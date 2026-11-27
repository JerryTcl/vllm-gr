#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Shared arguments and helpers for the offline Beam Search examples."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")

BEGIN_TOKEN = "<|sid_begin|>"
END_TOKEN = "<|sid_end|>"
DEFAULT_PROMPT = "Recommend the next video."


def parse_args(
    *,
    quickstart: bool = False,
    description: str | None = None,
    force_mode: str | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description or __doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--catalog",
        type=Path,
        help="Constraint triples JSON (default: full-vocabulary Beam Search; required with --cot)",
    )
    parser.add_argument(
        "--constraint-backend",
        choices=("cuda", "npu"),
        default="cuda",
        help="Device backend for the Beam example (CUDA or NPU)",
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        help=f'UTF-8 prompt file (default: "{DEFAULT_PROMPT}"; required for thinking modes)',
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Run ordinary LLM thinking instead of Beam Search",
    )
    parser.add_argument(
        "--cot",
        action="store_true",
        help="Use the same complete --prompt for thinking, then constrained SID Beam Search",
    )
    parser.add_argument(
        "--cot-max-tokens",
        type=int,
        default=1024,
        help="Thinking budget before Beam; default policy aborts if </think> is not reached",
    )
    parser.add_argument(
        "--cot-on-limit",
        choices=("error", "truncate"),
        default="error",
        help="On an exhausted CoT budget: fail, or explicitly close the span and continue",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=1024, help="CoT plus final answer output budget"
    )
    parser.add_argument("--temperature", type=float, default=0.6, help="Thinking mode sampling")
    parser.add_argument("--top-p", type=float, default=0.95, help="Thinking mode sampling")
    parser.add_argument("--top-k", type=int, default=20, help="Thinking mode sampling")
    parser.add_argument("--seed", type=int, default=0, help="Thinking mode random seed")
    parser.add_argument(
        "--execution",
        choices=("eager", "graph"),
        default="eager",
        help="NPU Decode execution mode; CUDA keeps its existing graph behavior",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=None if quickstart else 1024,
        help="Fixed prompt length including the SID begin token (optional in the quickstart)",
    )
    parser.add_argument(
        "--generated-steps",
        type=int,
        choices=(1, 2, 3),
        default=3,
        help="Effective SID length; constrained Beam requires 3 (unused with --enable-thinking)",
    )
    parser.add_argument("--beam-width", type=int, default=128)
    parser.add_argument(
        "--dynamic-beam-widths",
        help="NPU per-stage retained widths, e.g. 128,64,32 (beam-width remains capacity)",
    )
    parser.add_argument(
        "--result-width",
        type=int,
        help="NPU final sequence count; defaults to the last stage width",
    )
    parser.add_argument(
        "--candidate-width",
        type=int,
        help="Candidates per beam row (1..256); defaults to --beam-width",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument(
        "--no-prefill-graph",
        action="store_true",
        help="Disable GR Prefill graph capture and replay",
    )
    parser.add_argument("--print-beams", type=int, default=5)
    parser.add_argument("--repeat", type=int, help="Benchmark this many batches without a profiler")
    parser.add_argument(
        "--warmup-runs", type=int, default=3, help="Untimed warmup batches (requires --repeat)"
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Save Beam benchmark (--repeat required), or thinking output and timings",
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        help="Save Worker profiler traces; exclude engine startup and output printing",
    )
    parser.add_argument(
        "--profile-warmup-runs",
        type=int,
        default=1,
        help="Unprofiled warmup batches before collection (requires --profile-dir)",
    )
    parser.add_argument(
        "--profile-runs",
        type=int,
        default=1,
        help="Complete batches to profile; print only the last result (requires --profile-dir)",
    )
    args = parser.parse_args()
    if force_mode == "thinking":
        args.enable_thinking = True
        args.cot = False
    elif force_mode == "cot":
        args.cot = True
        args.enable_thinking = False
    if args.cot and args.enable_thinking:
        parser.error("--cot and --enable-thinking are separate modes; choose one")
    if args.cot and args.cot_max_tokens <= 0:
        parser.error("--cot-max-tokens must be positive")
    if args.cot and args.catalog is None:
        parser.error("--cot requires --catalog for constrained SID generation")
    if args.catalog is not None and not args.enable_thinking and args.generated_steps != 3:
        parser.error(
            "constrained Beam requires --generated-steps 3 to match constraint_table_v1 depth"
        )
    if args.enable_thinking or args.cot:
        if args.prompt is None:
            parser.error("--prompt is required with --enable-thinking or --cot")
        if args.enable_thinking and (
            args.dynamic_beam_widths is not None or args.result_width is not None
        ):
            parser.error("--enable-thinking cannot be combined with dynamic Beam options")
        if args.max_tokens <= 0 or args.max_model_len <= 0:
            parser.error("--max-tokens and --max-model-len must be positive")
        if not 0.0 <= args.temperature <= 2.0 or not 0.0 < args.top_p <= 1.0:
            parser.error("--temperature must be in [0, 2] and --top-p in (0, 1]")
        if args.top_k < -1:
            parser.error("--top-k must be -1, 0, or positive")
    if args.dynamic_beam_widths is not None:
        try:
            widths = [int(value.strip()) for value in args.dynamic_beam_widths.split(",")]
        except ValueError:
            parser.error("--dynamic-beam-widths must contain comma-separated integers")
        if len(widths) != args.generated_steps or any(
            not 1 <= width <= args.beam_width for width in widths
        ):
            parser.error(
                "--dynamic-beam-widths needs one width per generated step, each in [1, beam-width]"
            )
        args.dynamic_beam_widths = widths
    if args.dynamic_beam_widths is not None or args.result_width is not None:
        if args.constraint_backend != "npu":
            parser.error("dynamic-beam-widths and result-width currently require the NPU backend")
        final_width = (
            args.dynamic_beam_widths[-1]
            if args.dynamic_beam_widths is not None
            else args.beam_width
        )
        if args.result_width is not None and not 1 <= args.result_width <= final_width:
            parser.error("--result-width must be in [1, last-stage-width]")
    if args.profile_warmup_runs < 0 or args.profile_runs <= 0:
        parser.error("--profile-warmup-runs must be nonnegative and --profile-runs positive")
    if args.warmup_runs < 0 or (args.repeat is not None and args.repeat <= 0):
        parser.error("--warmup-runs must be nonnegative and --repeat positive")
    if args.repeat is not None and args.profile_dir is not None:
        parser.error(
            "--repeat cannot be combined with --profile-dir; profile overhead affects latency"
        )
    if (
        args.output_json is not None
        and args.repeat is None
        and not (args.enable_thinking or args.cot)
    ):
        parser.error("--output-json requires --repeat")
    return args


def validate_common_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")


def validate_input_paths(args: argparse.Namespace) -> None:
    paths = [args.model]
    if args.prompt is not None:
        paths.append(args.prompt)
    if args.catalog is not None and not args.enable_thinking:
        paths.append(args.catalog)
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)


def configure_runner_env(args: argparse.Namespace) -> None:
    if args.constraint_backend == "npu":
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    elif not (args.enable_thinking or args.cot) and not os.environ.get("VLLM_USE_V2_MODEL_RUNNER"):
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"


def profiler_engine_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    if args.profile_dir is None:
        return {}
    profile_dir = args.profile_dir.resolve()
    profile_dir.mkdir(parents=True, exist_ok=True)
    return {
        "profiler_config": {
            "profiler": "torch",
            "torch_profiler_dir": str(profile_dir),
            "torch_profiler_with_stack": False,
            "torch_profiler_record_shapes": False,
            "torch_profiler_with_memory": False,
            "torch_profiler_use_gzip": False,
            "torch_profiler_dump_cuda_time_total": False,
            "ignore_frontend": True,
        }
    }


def isolate_prompts(prompts: list[Any], suffix: str) -> list[Any]:
    # Keep prefix caching enabled, but force full Prefill for every row/run.
    return [
        dict(
            prompt,
            prompt_token_ids=list(prompt["prompt_token_ids"]),
            cache_salt=f"{prompt['cache_salt']}-{suffix}",
        )
        for prompt in prompts
    ]


def make_prompt(tokenizer: Any, path: Path | None, input_len: int | None) -> list[int]:
    text = DEFAULT_PROMPT
    if path is not None:
        with path.open(encoding="utf-8", newline="") as prompt_file:
            text = prompt_file.read()
    source = tokenizer.encode(text, add_special_tokens=False)
    if input_len is None:
        return [int(token) for token in source]

    # input_len includes the begin token appended by beam_search.
    submitted_len = input_len - 1
    if submitted_len <= 0:
        raise ValueError("--input-len must be at least 2")
    if len(source) >= submitted_len:
        return [int(token) for token in source[-submitted_len:]]
    newline = tokenizer.encode("\n", add_special_tokens=False)
    if len(newline) != 1:
        raise RuntimeError(f"newline must map to exactly one token, got {newline}")
    return [int(newline[0])] * (submitted_len - len(source)) + [int(token) for token in source]
