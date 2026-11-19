# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Platform adapter factory for canonical constraint tables."""

from __future__ import annotations

from typing import Any

from vllm_gr.config import GRConfig
from vllm_gr.platforms.plan import BackendPlan
from vllm_gr.platforms.services import constraint_execution_override


def _backend_label(provider: Any) -> str:
    """Describe the active provider for a startup diagnostic.

    The backend ID is the GR implementation; tensor placement stays available
    through the platform identity when the two differ.
    """
    backend_id = getattr(provider, "backend_id", None)
    return repr(backend_id) if isinstance(backend_id, str) else repr(provider)


def create_constraint_execution(
    *,
    device: str | Any | None = None,
    plan: Any = None,
) -> Any:
    """Create the platform constrained-top-k adapter, failing explicitly."""
    override = constraint_execution_override(device=device)
    if override is not None:
        return override

    from vllm_gr.platforms.loader import activate_current_backend

    provider = activate_current_backend()
    if plan is None:
        adapter = provider.create_constrained_topk_execution(device=device)
    else:
        from vllm_gr.platforms.services import planning_provider

        adapter = planning_provider(provider).create_constrained_topk_execution(
            plan=plan, device=device
        )
    if adapter is None:
        raise RuntimeError(
            "constraint.mode='table' cannot start: the active backend "
            f"{_backend_label(provider)} does not implement Constraint Table execution."
        )
    if not adapter.is_available():
        raise RuntimeError(
            "constraint.mode='table' cannot start: the active backend "
            f"{_backend_label(provider)} provides a Constraint Table adapter but its custom "
            "operator is unavailable (is_available()=False)."
        )
    return adapter


def validate_constraint_table_platform(
    gr_config: GRConfig,
    *,
    device_type: str | None = None,
    plan: BackendPlan | None = None,
) -> None:
    """Reject configurations Constraint Tables cannot support in this release.

    The selected implementation declares whether it handles full-vocabulary
    token IDs against a sliced logits window.
    """
    slicing_enabled = gr_config.vocab_offset is not None or gr_config.vocab_slice_size is not None
    if not slicing_enabled:
        return
    from vllm_gr.constraints.registry import (
        implementation_by_id,
        resolve_constraint_implementation,
    )

    table = gr_config.constraint_table
    implementation_id = (plan.constraint_implementation if plan else None) or device_type
    if table is not None and table.enabled:
        implementation = resolve_constraint_implementation(table, plan)
    elif implementation_id:
        implementation = implementation_by_id(implementation_id)
    else:
        implementation = None
    if implementation is None or not implementation.supports_vocab_slicing:
        raise ValueError(
            "constraint_backend='constraint_table' cannot be combined with "
            f"vocab slicing (vocab_offset={gr_config.vocab_offset!r}, "
            f"vocab_slice_size={gr_config.vocab_slice_size!r}) on platform "
            f"{device_type or implementation_id or 'unknown'!r}"
        )
