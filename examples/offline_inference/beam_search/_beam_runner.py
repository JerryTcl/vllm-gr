#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Beam Search execution shared by the offline example entrypoints."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, cast

try:
    from . import _beam_common as common
except ImportError:
    import _beam_common as common  # type: ignore[no-redef]


def _cot_module() -> Any:
    try:
        from . import cot
    except ImportError:
        import cot  # type: ignore[no-redef]

    return cot


def execute_beam_batch(
    llm: Any, args: argparse.Namespace, prompts: list[Any], params: Any
) -> list[Any]:
    if not getattr(args, "cot", False):
        return cast(list[Any], llm.beam_search(prompts, params, output_mode="full"))
    return cast(list[Any], _cot_module().execute_cot_beam_batch(llm, args, prompts, params))


def benchmark_beam_search(
    llm: Any, args: argparse.Namespace, prompts: list[Any], params: Any
) -> tuple[list[Any], dict[str, Any]]:
    for iteration in range(args.warmup_runs):
        execute_beam_batch(
            llm,
            args,
            common.isolate_prompts(prompts, f"warmup-{iteration}"),
            params,
        )

    latencies_ms: list[float] = []
    returned_beams: list[int] = []
    outputs: list[Any] = []
    for iteration in range(args.repeat):
        isolated = common.isolate_prompts(prompts, f"benchmark-{iteration}")
        started = perf_counter()
        outputs = execute_beam_batch(llm, args, isolated, params)
        latencies_ms.append((perf_counter() - started) * 1000.0)
        if len(outputs) != len(prompts):
            raise RuntimeError(f"expected {len(prompts)} request outputs, got {len(outputs)}")
        # Count candidates outside timing; do not tokenize/decode or print here.
        returned_beams.append(sum(len(output.sequences) for output in outputs))

    ordered = sorted(latencies_ms)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    total_seconds = sum(latencies_ms) / 1000.0
    report: dict[str, Any] = {
        "model": str(args.model.resolve()),
        "catalog": str(args.catalog.resolve()) if args.catalog is not None else None,
        "prompt": str(args.prompt.resolve()) if args.prompt is not None else None,
        "constraint_backend": args.constraint_backend,
        "execution": args.execution,
        "batch_size": args.batch_size,
        "beam_width": args.beam_width,
        "input_len": args.input_len,
        "generated_steps": args.generated_steps,
        "length_penalty": args.length_penalty,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "warmup_runs": args.warmup_runs,
        "repeat": args.repeat,
        "prefix_caching": "enabled; isolated salt per row and run (cold Prefill)",
        "timing_scope": "beam_search API call until output objects return; excludes engine startup, "
        "warmup, pre-call prompt preparation and post-call result inspection/printing",
        "batch_latency_ms": {
            "mean": mean(latencies_ms),
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "min": ordered[0],
            "max": ordered[-1],
        },
        "total_timed_seconds": total_seconds,
        "requests_per_second": len(prompts) * len(latencies_ms) / total_seconds,
        "returned_beams_per_second": sum(returned_beams) / total_seconds,
        "runs": [
            {"run": index + 1, "batch_latency_ms": latency, "returned_beams": beams}
            for index, (latency, beams) in enumerate(zip(latencies_ms, returned_beams, strict=True))
        ],
    }
    if getattr(args, "dynamic_beam_widths", None) is not None:
        report["dynamic_beam_widths"] = list(args.dynamic_beam_widths)
    if getattr(args, "result_width", None) is not None:
        report["result_width"] = args.result_width
    if getattr(args, "cot", False):
        report.update(
            mode="cot_beam",
            input_len=None,
            cot_input_tokens=len(prompts[0]["prompt_token_ids"]),
            cot_max_tokens=args.cot_max_tokens,
            cot_on_limit=args.cot_on_limit,
            timing_scope="CoT generation, token handoff and beam_search until outputs return; "
            "excludes startup, warmup and printing",
        )
    return outputs, report


def run_beam_search(
    llm: Any, args: argparse.Namespace, prompts: list[Any], params: Any
) -> list[Any]:
    if args.profile_dir is None:
        outputs: list[Any] = execute_beam_batch(llm, args, prompts, params)
        return outputs

    def run_once(iteration: int) -> list[Any]:
        isolated = common.isolate_prompts(prompts, f"profile-{iteration}")
        result: list[Any] = execute_beam_batch(llm, args, isolated, params)
        return result

    for iteration in range(args.profile_warmup_runs):
        run_once(iteration)
    llm.start_profile(f"offline_beam_v1_{args.execution}_b{args.batch_size}_w{args.beam_width}")
    try:
        for iteration in range(
            args.profile_warmup_runs, args.profile_warmup_runs + args.profile_runs
        ):
            outputs = run_once(iteration)
    finally:
        llm.stop_profile()
    return outputs


def run(args: argparse.Namespace, artifact_path: Path | None = None) -> None:
    from vllm.inputs import tokens_input

    from vllm_gr.entrypoints.gr import GRLLM
    from vllm_gr.sampling_params import BeamSearchParams

    is_npu = args.constraint_backend == "npu"
    # Candidate width K is the per-row top-k; None means it follows the beam
    # width (the pre-decoupling behaviour).
    candidate_width = args.candidate_width if args.candidate_width is not None else args.beam_width
    prompt_token_ids = None
    if getattr(args, "cot", False):
        from transformers import AutoTokenizer

        cot = _cot_module()
        tokenizer = AutoTokenizer.from_pretrained(str(args.model.resolve()), trust_remote_code=True)
        prompt_token_ids = cot.make_cot_prompt(tokenizer, args.prompt.resolve())
        separator_len = len(tokenizer.encode("\n\n", add_special_tokens=False))
        forced_close_reserve = int(args.cot_on_limit == "truncate")
        longest_prompt = (
            len(prompt_token_ids) + args.cot_max_tokens + forced_close_reserve + separator_len + 1
        )
        if longest_prompt + args.generated_steps + 1 > args.max_model_len:
            raise ValueError(
                "full prompt plus --cot-max-tokens and SID budget exceeds --max-model-len"
            )
        prefill_capacity = args.batch_size * longest_prompt
    elif args.input_len is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(args.model.resolve()), trust_remote_code=True)
        prompt_token_ids = common.make_prompt(tokenizer, args.prompt, None)
        # Reserve both SID boundary positions and all generated SID components.
        prompt_limit = args.max_model_len - args.generated_steps - 2
        if len(prompt_token_ids) > prompt_limit:
            raise ValueError(
                f"prompt has {len(prompt_token_ids)} tokens, but only {prompt_limit} fit "
                f"within --max-model-len {args.max_model_len} with the SID budget"
            )
        prefill_capacity = args.batch_size * (len(prompt_token_ids) + 2)
    else:
        prefill_capacity = args.batch_size * (args.input_len + 1)

    candidate_widths = [args.beam_width] + [candidate_width] * (args.generated_steps - 1)
    max_candidates = max(candidate_widths)
    table_config = None
    if args.catalog is not None:
        if artifact_path is None:
            raise ValueError("artifact_path is required with --catalog")
        from vllm_gr.constraints.artifact_builder import build_single_trie_artifact

        artifact_digest, tokenizer_digest = build_single_trie_artifact(
            args.model.resolve(), args.catalog.resolve(), artifact_path
        )
        table_config = {
            "path": str(artifact_path.resolve()),
            "format": "constraint_table_v1",
            "backend": args.constraint_backend,
            "max_top_k": max_candidates,
            "digests": {"artifact": artifact_digest, "tokenizer": tokenizer_digest},
        }
    beam_config: dict[str, object] = {
        "beam_graph_enabled": not is_npu or args.execution == "graph",
        "worker_decision": True,
        "beam_width": [args.beam_width] * args.generated_steps,
        "candidate_width": candidate_widths,
        "max_decode_steps": args.generated_steps,
        "constraint_mode": "table" if table_config is not None else "none",
        "constraint_table": table_config,
    }
    if getattr(args, "no_prefill_graph", False):
        beam_config["prefill_graph_enabled"] = False
    dynamic_widths = getattr(args, "dynamic_beam_widths", None)
    result_width = getattr(args, "result_width", None)
    if is_npu:
        beam_config["decode_graph_batch_size"] = args.batch_size
        if dynamic_widths is not None and len(dynamic_widths) > 1:
            beam_config["decode_graph_widths"] = sorted(set(dynamic_widths[:-1]))
    else:
        beam_config["beam_batch_sizes"] = sorted({1, args.batch_size})
    params = BeamSearchParams(
        beam_width=args.beam_width,
        dynamic_beam_widths=dynamic_widths,
        result_width=result_width,
        # Public max_tokens includes the begin and end boundary positions.
        max_tokens=args.generated_steps + 2,
        temperature=0.0,
        ignore_eos=True,
        include_stop_str_in_output=True,
        begin_token=common.BEGIN_TOKEN,
        end_token=common.END_TOKEN,
        length_penalty=args.length_penalty,
    )

    with GRLLM(
        model=str(args.model.resolve()),
        trust_remote_code=True,
        gr=beam_config,
        attention_config={"backend": "CUSTOM"},
        async_scheduling=True,
        enable_prefix_caching=True,
        enable_chunked_prefill=False,
        max_model_len=args.max_model_len,
        # The V1 input processor appends the begin token to every prompt, so a
        # B-wide Prefill schedules B * (input_len + 1) tokens, not B * input_len.
        max_num_batched_tokens=max(args.max_model_len, prefill_capacity),
        max_num_seqs=max(args.beam_width, args.batch_size),
        max_logprobs=max_candidates,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=is_npu and args.execution == "eager",
        **common.profiler_engine_kwargs(args),
    ) as llm:
        tokenizer = llm.get_tokenizer()
        if getattr(args, "cot", False):
            assert prompt_token_ids is not None
            print(f"mode=cot_beam input_tokens={len(prompt_token_ids)}; --input-len is not used")
        elif prompt_token_ids is None:
            prompt_token_ids = common.make_prompt(tokenizer, args.prompt, args.input_len)
        # Keep prefix caching enabled, but prevent duplicate batch rows from
        # sharing the same prompt block before their atomic Prefill completes.
        request_prompts = [
            tokens_input(
                list(prompt_token_ids),
                cache_salt=f"offline-beam-v1-row-{request_index}",
            )
            for request_index in range(args.batch_size)
        ]
        benchmark = None
        if args.repeat is not None:
            outputs, benchmark = benchmark_beam_search(llm, args, request_prompts, params)
        else:
            outputs = run_beam_search(llm, args, request_prompts, params)
        if getattr(args, "cot", False):
            print(json.dumps({"cot_beam": args.cot_last_report}, ensure_ascii=False))
            if any(row["thinking_truncated"] for row in args.cot_last_report["requests"]):
                print(
                    "CoT budget exhausted: explicitly truncated (--cot-on-limit=truncate); "
                    "this is not a naturally completed thinking span"
                )
        if is_npu and args.execution == "graph":
            stats = llm.collective_rpc("get_gr_beam_graph_runtime_stats")[0]
            print(
                json.dumps(
                    {"npu_beam_graph": stats},
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            decode_widths = dynamic_widths[:-1] if dynamic_widths is not None else [args.beam_width]
            expected_shapes = {(args.batch_size, width) for width in decode_widths}
            captured_shapes = {tuple(shape) for shape in stats.get("captured_shapes", ())}
            # N=1 finishes in Prefill and must not require a Decode graph.
            if args.generated_steps > 1 and (
                not expected_shapes.issubset(captured_shapes) or not stats["full_dispatches"]
            ):
                raise RuntimeError("NPU Beam graph capture/replay was not observed")
            if benchmark is not None:
                benchmark["npu_beam_graph"] = stats

        if benchmark is not None:
            print(json.dumps({"beam_benchmark": benchmark}, ensure_ascii=False, sort_keys=True))
            if args.output_json is not None:
                output_path = args.output_json.resolve()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    json.dumps(benchmark, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                print(f"benchmark_json={output_path}")

        token_field = "sid" if args.catalog is not None else "beam"
        for request_index, output in enumerate(outputs):
            sequences = output.sequences
            print(f"request={request_index} returned_beams={len(sequences)}")
            for rank, sequence in enumerate(sequences[: args.print_beams]):
                token_ids = [int(token) for token in sequence.tokens]
                beam_token_ids = token_ids[-(args.generated_steps + 2) :]
                print(
                    json.dumps(
                        {
                            "request": request_index,
                            "rank": rank,
                            "cumulative_logprob": float(sequence.cum_logprob),
                            "finish_reason": sequence.finish_reason,
                            "stop_reason": sequence.stop_reason,
                            f"{token_field}_token_ids": beam_token_ids,
                            f"{token_field}_tokens": tokenizer.convert_ids_to_tokens(
                                beam_token_ids
                            ),
                        },
                        ensure_ascii=False,
                    )
                )
        if getattr(args, "cot", False) and args.output_json is not None:
            report = {
                "mode": "cot_beam",
                "cot": args.cot_last_report,
                "benchmark": benchmark,
                "results": [
                    [
                        {"tokens": list(seq.tokens), "cumulative_logprob": float(seq.cum_logprob)}
                        for seq in output.sequences
                    ]
                    for output in outputs
                ],
            }
            path = args.output_json.resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )


def run_entry(args: argparse.Namespace) -> None:
    if args.catalog is None:
        run(args)
        return
    with tempfile.TemporaryDirectory(prefix="vllm-gr-beam-v1-") as directory:
        artifact_path = Path(directory).resolve() / "constraint_table.safetensors"
        run(args, artifact_path)
