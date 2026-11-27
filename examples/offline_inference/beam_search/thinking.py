#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Run ordinary LLM thinking generation from the offline Beam Search example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

import regex as re

try:
    from . import _beam_common as common
except ImportError:
    import _beam_common as common  # type: ignore[no-redef]


def _read_thinking_input(path: Path) -> str:
    """Read plain text or unambiguously encoded ChatML, without guessing escapes."""
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith('"'):
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError as exc:
            if re.match(r'^"\s*<\|im_start\|>', text.lstrip()):
                raise ValueError("JSON-encoded ChatML must be a complete JSON string") from exc
        else:
            if isinstance(decoded, str) and decoded.lstrip().startswith("<|im_start|>"):
                text = decoded
    if (
        text.lstrip().startswith("<|im_start|>")
        and "\\n" in text
        and "\n" not in text.rstrip("\r\n")
    ):
        raise ValueError(
            "ambiguous single-line ChatML with literal \\n; use real newlines or a "
            "JSON string containing ChatML, with body backslashes escaped by JSON"
        )
    return text


def _template_thinking_prompt(tokenizer: Any, text: str) -> list[int]:
    tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=True,
        return_dict=False,
        enable_thinking=True,
        add_generation_prompt=True,
    )
    if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
        raise TypeError("chat template must return a flat list of token IDs")
    if not tokens:
        raise ValueError("chat template returned an empty prompt")
    return tokens


def single_thinking_token(tokenizer: Any, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1 or tokenizer.convert_ids_to_tokens(ids[0]) != text:
        raise ValueError(f"CoT requires a dedicated {text} token in the model tokenizer")
    return int(ids[0])


def _reopen_empty_thinking_span(tokenizer: Any, ids: list[int]) -> list[int]:
    rendered = tokenizer.decode(ids, skip_special_tokens=False)
    if not re.search(r"<think>\s*</think>\s*$", rendered):
        return ids
    # Keep every earlier ID: an empty template span may be removed only when
    # its opening tag is a real token boundary, not text inside another token.
    open_id = single_thinking_token(tokenizer, "<think>")
    try:
        open_index = len(ids) - 1 - ids[::-1].index(open_id)
    except ValueError as exc:
        raise ValueError("chat template did not preserve the <think> special token") from exc
    if not re.fullmatch(
        r"<think>\s*</think>\s*",
        tokenizer.decode(ids[open_index:], skip_special_tokens=False),
    ):
        raise ValueError("chat template did not preserve the terminal <think> token boundary")
    return [*ids[: open_index + 1], *tokenizer.encode("\n", add_special_tokens=False)]


def _serialized_thinking_prompt(tokenizer: Any, text: str) -> list[int]:
    single_thinking_token(tokenizer, "<think>")
    single_thinking_token(tokenizer, "</think>")
    user_end = text.rfind("<|im_end|>")
    prefix, suffix = text[:user_end], text[user_end:]
    if user_end < 0 or not re.fullmatch(
        r"<\|im_end\|>\s*<\|im_start\|>assistant\s*(?:<think>\s*(?:</think>\s*)?)?",
        suffix,
    ):
        raise ValueError("CoT expects serialized chat ending in an empty assistant/think prefix")
    prefix = re.sub(r"/no_think(?=\s*$)", "/think", prefix)
    text = prefix + "<|im_end|>\n<|im_start|>assistant\n<think>\n"
    return list(tokenizer.encode(text, add_special_tokens=False))


def make_thinking_prompt(tokenizer: Any, path: Path) -> list[int]:
    # Preserve the complete question and template; Beam's fixed-length prompt
    # padding/truncation and SID boundary tokens do not apply to normal chat.
    text = _read_thinking_input(path)
    if text.lstrip().startswith("<|im_start|>"):
        return _serialized_thinking_prompt(tokenizer, text)
    return _reopen_empty_thinking_span(tokenizer, _template_thinking_prompt(tokenizer, text))


def run_thinking(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.inputs import tokens_input

    from vllm_gr.entrypoints.gr import GRLLM

    tokenizer = AutoTokenizer.from_pretrained(str(args.model.resolve()), trust_remote_code=True)
    prompt_ids = make_thinking_prompt(tokenizer, args.prompt.resolve())
    if len(prompt_ids) + args.max_tokens > args.max_model_len:
        raise ValueError(
            f"thinking prompt ({len(prompt_ids)}) + --max-tokens ({args.max_tokens}) "
            f"exceeds --max-model-len ({args.max_model_len}); the prompt was not truncated"
        )
    sampling = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
    }
    params = SamplingParams(**sampling)
    prompts = [tokens_input(list(prompt_ids)) for _ in range(args.batch_size)]
    config = {
        "beam_width": None,
        "candidate_width": None,
        "constraint_mode": "none",
    }
    device_kwargs = {"block_size": 128} if args.constraint_backend == "npu" else {}
    print(
        f"mode=thinking batch_size={args.batch_size} input_tokens={len(prompt_ids)} "
        f"max_tokens={args.max_tokens}; ordinary LLM generation, constraints disabled"
    )
    with GRLLM(
        model=str(args.model.resolve()),
        trust_remote_code=True,
        seed=args.seed,
        gr=config,
        attention_config={"backend": "CUSTOM"},
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=max(args.max_model_len, args.batch_size * len(prompt_ids)),
        max_num_seqs=max(4, args.batch_size),
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.execution == "eager",
        **device_kwargs,
        **common.profiler_engine_kwargs(args),
    ) as llm:

        def generate() -> list[Any]:
            outputs: list[Any] = llm.generate(prompts, params, use_tqdm=False)
            if len(outputs) != args.batch_size:
                raise RuntimeError(f"expected {args.batch_size} outputs, got {len(outputs)}")
            return outputs

        latencies_ms: list[float] = []
        generated_tokens: list[int] = []
        if args.repeat is not None:
            for _ in range(args.warmup_runs):
                generate()
            for _ in range(args.repeat):
                started = perf_counter()
                outputs = generate()
                latencies_ms.append((perf_counter() - started) * 1000.0)
                generated_tokens.append(
                    sum(
                        len(candidate.token_ids)
                        for output in outputs
                        for candidate in output.outputs
                    )
                )
        elif args.profile_dir is not None:
            for _ in range(args.profile_warmup_runs):
                generate()
            llm.start_profile(f"offline_thinking_{args.execution}_b{args.batch_size}")
            try:
                for _ in range(args.profile_runs):
                    outputs = generate()
            finally:
                llm.stop_profile()
        else:
            outputs = generate()

        results = []
        for request_index, output in enumerate(outputs):
            for candidate in output.outputs:
                result = {
                    "request": request_index,
                    "text": candidate.text,
                    "token_ids": list(candidate.token_ids),
                    "generated_tokens": len(candidate.token_ids),
                    "finish_reason": candidate.finish_reason,
                    "stop_reason": candidate.stop_reason,
                }
                results.append(result)
                print(
                    f"request={request_index} generated_tokens={len(candidate.token_ids)} "
                    f"finish_reason={candidate.finish_reason}\n{candidate.text}"
                )

    report = {
        "mode": "thinking",
        "model": str(args.model.resolve()),
        "prompt": str(args.prompt.resolve()),
        "execution": args.execution,
        "batch_size": args.batch_size,
        "input_tokens": len(prompt_ids),
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "sampling": sampling,
        "results": results,
    }
    if latencies_ms:
        benchmark = {
            "timing_scope": "generate API until output objects return; excludes startup and warmup",
            "prefix_caching": False,
            "repeat": args.repeat,
            "warmup_runs": args.warmup_runs,
            "mean_batch_latency_ms": mean(latencies_ms),
            "runs": [
                {"batch_latency_ms": latency, "generated_tokens": count}
                for latency, count in zip(latencies_ms, generated_tokens, strict=True)
            ],
        }
        report["benchmark"] = benchmark
        print(json.dumps({"thinking_benchmark": benchmark}, ensure_ascii=False))
    if args.output_json is not None:
        output_path = args.output_json.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"thinking_json={output_path}")


def main() -> None:
    args = common.parse_args(description=__doc__, force_mode="thinking")
    common.validate_common_args(args)
    common.validate_input_paths(args)
    common.configure_runner_env(args)
    run_thinking(args)


if __name__ == "__main__":
    main()
