#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Run offline OneRec Beam Search, with an optional Constraint Table."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

try:
    from . import _beam_common as common
    from ._beam_common import (
        BEGIN_TOKEN,
        DEFAULT_PROMPT,
        END_TOKEN,
        isolate_prompts,
        make_prompt,
        parse_args,
        profiler_engine_kwargs,
    )
    from ._beam_runner import (
        benchmark_beam_search,
        execute_beam_batch,
        run,
        run_beam_search,
    )
    from .cot import make_cot_prompt
    from .thinking import make_thinking_prompt, run_thinking
except ImportError:
    import _beam_common as common  # type: ignore[no-redef]
    from _beam_common import (  # type: ignore[no-redef]
        BEGIN_TOKEN,
        DEFAULT_PROMPT,
        END_TOKEN,
        isolate_prompts,
        make_prompt,
        parse_args,
        profiler_engine_kwargs,
    )
    from _beam_runner import (  # type: ignore[no-redef]
        benchmark_beam_search,
        execute_beam_batch,
        run,
        run_beam_search,
    )
    from cot import make_cot_prompt  # type: ignore[no-redef]
    from thinking import make_thinking_prompt, run_thinking  # type: ignore[no-redef]

__all__ = [
    "BEGIN_TOKEN",
    "DEFAULT_PROMPT",
    "END_TOKEN",
    "benchmark_beam_search",
    "execute_beam_batch",
    "isolate_prompts",
    "main",
    "make_cot_prompt",
    "make_prompt",
    "make_thinking_prompt",
    "parse_args",
    "profiler_engine_kwargs",
    "run",
    "run_beam_search",
    "run_thinking",
]


def main(*, quickstart: bool = False) -> None:
    args = common.parse_args(quickstart=quickstart, description=__doc__)
    common.validate_input_paths(args)
    common.validate_common_args(args)
    common.configure_runner_env(args)
    if args.enable_thinking:
        if args.catalog is not None:
            print("--catalog is ignored in --enable-thinking mode", file=sys.stderr)
        run_thinking(args)
        return
    if (
        not args.cot
        and args.input_len is not None
        and args.input_len + args.generated_steps + 1 > args.max_model_len
    ):
        raise ValueError("prompt and generated tokens exceed --max-model-len")
    if args.print_beams <= 0:
        raise ValueError("--print-beams must be positive")

    if args.catalog is None:
        run(args)
        return

    with tempfile.TemporaryDirectory(prefix="vllm-gr-beam-v1-") as directory:
        artifact_path = Path(directory).resolve() / "constraint_table.safetensors"
        run(args, artifact_path)


if __name__ == "__main__":
    main()
