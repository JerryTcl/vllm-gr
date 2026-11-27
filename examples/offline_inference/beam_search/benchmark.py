#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Benchmark offline Beam Search batches without starting a profiler."""

from __future__ import annotations

try:
    from . import _beam_common as common
except ImportError:
    import _beam_common as common  # type: ignore[no-redef]


def main() -> None:
    args = common.parse_args(description=__doc__)
    if args.enable_thinking:
        raise SystemExit("benchmark.py covers Beam Search; use thinking.py --repeat instead")
    if args.repeat is None:
        raise SystemExit("benchmark.py requires --repeat")
    common.validate_common_args(args)
    common.validate_input_paths(args)
    common.configure_runner_env(args)
    try:
        from . import _beam_runner
    except ImportError:
        import _beam_runner  # type: ignore[no-redef]

    _beam_runner.run_entry(args)


if __name__ == "__main__":
    main()
