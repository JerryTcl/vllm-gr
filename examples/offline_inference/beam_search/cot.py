#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Run CoT-to-SID Beam Search from the offline Beam Search example."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Any

import regex as re

try:
    from . import _beam_common as common
    from .thinking import make_thinking_prompt, single_thinking_token
except ImportError:
    import _beam_common as common  # type: ignore[no-redef]
    from thinking import (  # type: ignore[no-redef]
        make_thinking_prompt,
        single_thinking_token,
    )


def make_cot_prompt(tokenizer: Any, path: Path) -> list[int]:
    """Preserve the full input; reopen only the terminal empty thinking span."""
    single_thinking_token(tokenizer, "<think>")
    single_thinking_token(tokenizer, "</think>")
    ids = make_thinking_prompt(tokenizer, path)
    # Some OneRec templates accept enable_thinking but leave the tag to the model.
    rendered = tokenizer.decode(ids, skip_special_tokens=False)
    if re.search(r"</think>\s*$", rendered):
        raise ValueError("CoT cannot replace a nonempty thinking span from the chat template")
    if not re.search(r"<think>\s*$", rendered):
        ids.extend(tokenizer.encode("<think>\n", add_special_tokens=False))
    if not ids:
        raise ValueError("CoT prompt is empty")
    return ids


def execute_cot_beam_batch(
    llm: Any, args: argparse.Namespace, prompts: list[Any], params: Any
) -> list[Any]:
    from vllm import SamplingParams

    tokenizer = llm.get_tokenizer()
    close_id = single_thinking_token(tokenizer, "</think>")
    started = perf_counter()
    completions = llm.generate(
        prompts,
        SamplingParams(
            max_tokens=args.cot_max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed,
            stop_token_ids=[close_id],
            detokenize=False,
        ),
        use_tqdm=False,
    )
    thinking_ms = (perf_counter() - started) * 1000.0
    if len(completions) != len(prompts):
        raise RuntimeError("CoT returned a different request count; Beam was not started")
    separator = list(tokenizer.encode("\n\n", add_special_tokens=False))
    beam_prompts = []
    records = []
    for index, (prompt, result) in enumerate(zip(prompts, completions, strict=True)):
        # LLM.generate returns results in input order. Verify their source IDs
        # before associating each continuation with its SID request.
        if result.prompt_token_ids != prompt["prompt_token_ids"]:
            raise RuntimeError(f"request {index}: CoT prompt/order mismatch; Beam was not started")
        if len(result.outputs) != 1:
            raise RuntimeError("CoT requires one continuation per request")
        completion = result.outputs[0]
        ids = list(completion.token_ids)
        forced_close = False
        if (
            ids
            and ids[-1] != close_id
            and completion.finish_reason == "length"
            and len(ids) == args.cot_max_tokens
            and getattr(args, "cot_on_limit", "error") == "truncate"
        ):
            ids.append(close_id)
            forced_close = True
        if not ids or ids[-1] != close_id:
            raise RuntimeError(
                f"request {index}: CoT did not reach </think> "
                f"(finish_reason={completion.finish_reason}); increase --cot-max-tokens "
                "or check the model/prompt. Beam was not started."
            )
        combined = [*prompt["prompt_token_ids"], *ids, *separator]
        if len(combined) + args.generated_steps + 2 > args.max_model_len:
            raise ValueError("CoT plus SID generation exceeds --max-model-len")
        # Keep the initial token-handoff feature independent of partial-prefix
        # reuse between ordinary Decode and the GR Beam runtime.
        beam_prompts.append(
            dict(
                prompt,
                prompt_token_ids=combined,
                cache_salt=f"{prompt.get('cache_salt', 'offline-cot')}-sid-row-{index}",
            )
        )
        records.append(
            {
                "request": index,
                "thinking_token_ids": ids,
                "generated_thinking_token_ids": list(completion.token_ids),
                "thinking_text": tokenizer.decode(ids, skip_special_tokens=False),
                "thinking_tokens": len(ids),
                "generated_thinking_tokens": len(completion.token_ids),
                "thinking_truncated": forced_close,
                "forced_close": forced_close,
                "beam_prompt_tokens": len(combined),
            }
        )
    started = perf_counter()
    outputs: list[Any] = llm.beam_search(beam_prompts, params, output_mode="full")
    if len(outputs) != len(prompts):
        raise RuntimeError("Beam returned a different request count after CoT")
    args.cot_last_report = {
        "thinking_ms": thinking_ms,
        "beam_ms": (perf_counter() - started) * 1000.0,
        "requests": records,
        "kv_handoff": "token IDs only; separate cache salts for thinking and SID Prefill",
    }
    return outputs


def main() -> None:
    args = common.parse_args(description=__doc__, force_mode="cot")
    common.validate_common_args(args)
    common.validate_input_paths(args)
    common.validate_run_args(args)
    common.configure_runner_env(args)
    try:
        from . import _beam_runner
    except ImportError:
        import _beam_runner  # type: ignore[no-redef]

    _beam_runner.run_entry(args)


if __name__ == "__main__":
    main()
