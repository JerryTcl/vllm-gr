# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Capture/replay contract for the opt-in dense CUDA Prefill.

Dense Prefill attends over a layer's freshly projected K/V instead of the paged
cache, which is only correct when a request has no earlier context. A CPU check
at call time is enough for eager execution, but a CUDA graph freezes the choice
made at capture: a decode graph captured from ``seq_len == query_len == 1``
dummies would replay over cached context and silently drop it.

The contract therefore has two halves:

* Capture: while a stream is capturing, dense attention may only be selected
  inside :func:`fresh_prefill_capture`, which only the GR Prefill graph runners
  open. Native decode, mixed and other captures keep the paged path.
* Replay: when dense Prefill is requested, those runners replay a Prefill graph
  only if every scheduled request has ``num_computed_tokens == 0``
  (checked by the graph runners using the same GR configuration).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

_fresh_prefill_capture: ContextVar[bool] = ContextVar(
    "vllm_gr_fresh_prefill_capture", default=False
)


def admitted_dense_prefill_mode(
    requested: str | None,
    *,
    prefix_caching: bool,
    chunked_prefill: bool,
    kv_transfer: bool,
    batch_invariant: bool,
) -> str | None:
    """Dense attention can consume fresh K/V only when no cached prefix is needed."""
    if requested == "off" or prefix_caching or chunked_prefill or kv_transfer or batch_invariant:
        return None
    return requested


def requires_fresh_prefill_replay(runner: Any) -> bool:
    """Use the same mode that was fixed on this runner's captured layers."""
    binding = getattr(runner, "_gr_backend_binding", None)
    if binding is not None:
        return bool(getattr(binding.attention_executor, "requires_fresh_prefill", False))
    config = getattr(runner, "vllm_config", None)
    if config is not None:
        from vllm_gr.platforms.plan import read_backend_plan

        plan = read_backend_plan(config)
        if plan is not None:
            # Native graph warmup can precede publication of the complete binding.
            return plan.choices.to_dict().get("dense_prefill_mode") is not None
    from vllm_gr.config.models import deployment_for

    return deployment_for(config).dense_prefill != "off"


@contextmanager
def fresh_prefill_capture() -> Iterator[None]:
    """Mark a capture whose graphs only replay for fresh prompts."""
    token = _fresh_prefill_capture.set(True)
    try:
        yield
    finally:
        _fresh_prefill_capture.reset(token)


def in_fresh_prefill_capture() -> bool:
    return _fresh_prefill_capture.get()
