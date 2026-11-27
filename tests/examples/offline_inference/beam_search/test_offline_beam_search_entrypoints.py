# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Entrypoint coverage for the split offline Beam Search scripts."""

import sys
from pathlib import Path
from typing import Any

import pytest

from examples.offline_inference.beam_search import _beam_runner
from examples.offline_inference.beam_search import benchmark as benchmark_module
from examples.offline_inference.beam_search import cot as cot_module
from examples.offline_inference.beam_search import thinking as thinking_module


def _prompt(tmp_path: Path) -> Path:
    path = tmp_path / "prompt.txt"
    path.write_text("Recommend the next video.", encoding="utf-8")
    return path


def _catalog(tmp_path: Path) -> Path:
    path = tmp_path / "catalog.json"
    path.write_text("[]", encoding="utf-8")
    return path


def test_thinking_entrypoint_selects_thinking(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    monkeypatch.setattr(
        sys,
        "argv",
        ["thinking", "--model", str(tmp_path), "--prompt", str(_prompt(tmp_path))],
    )
    received: list[Any] = []
    monkeypatch.setattr(thinking_module, "run_thinking", received.append)

    thinking_module.main()

    assert len(received) == 1
    assert received[0].enable_thinking is True
    assert received[0].cot is False


def test_thinking_entrypoint_rejects_cot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "thinking",
            "--model",
            str(tmp_path),
            "--prompt",
            str(_prompt(tmp_path)),
            "--catalog",
            str(_catalog(tmp_path)),
            "--cot",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        thinking_module.main()
    assert exc.value.code == 2


def test_cot_entrypoint_selects_cot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cot",
            "--model",
            str(tmp_path),
            "--prompt",
            str(_prompt(tmp_path)),
            "--catalog",
            str(_catalog(tmp_path)),
        ],
    )
    received: list[Any] = []
    monkeypatch.setattr(_beam_runner, "run_entry", received.append)

    cot_module.main()

    assert len(received) == 1
    assert received[0].cot is True
    assert received[0].enable_thinking is False


def test_cot_entrypoint_rejects_thinking(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cot",
            "--model",
            str(tmp_path),
            "--prompt",
            str(_prompt(tmp_path)),
            "--catalog",
            str(_catalog(tmp_path)),
            "--enable-thinking",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        cot_module.main()
    assert exc.value.code == 2


def test_benchmark_entrypoint_requires_repeat(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sys, "argv", ["benchmark", "--model", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        benchmark_module.main()
    assert "--repeat" in str(exc.value)


def test_benchmark_entrypoint_runs_repeated_batches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmark", "--model", str(tmp_path), "--repeat", "2"],
    )
    received: list[Any] = []
    monkeypatch.setattr(_beam_runner, "run_entry", received.append)

    benchmark_module.main()

    assert len(received) == 1
    assert received[0].repeat == 2


def test_benchmark_entrypoint_rejects_thinking(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--model",
            str(tmp_path),
            "--prompt",
            str(_prompt(tmp_path)),
            "--enable-thinking",
            "--repeat",
            "1",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        benchmark_module.main()
    assert "thinking.py" in str(exc.value)


pytestmark = pytest.mark.cpu_test
