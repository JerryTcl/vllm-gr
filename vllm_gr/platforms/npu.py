# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Thin NPU provider with lazy Ascend-specific imports."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, ClassVar, cast

from .protocol import BeamAttentionExecution, ModelRunnerPatchHooks

if TYPE_CHECKING:
    import torch
    from vllm.config import VllmConfig

    from vllm_gr.config import ConstraintBackend
    from vllm_gr.constraints.execution import NPUConstrainedTopKExecution


@dataclass(frozen=True)
class NPUBackendProvider:
    device_type: str = "npu"
    # Semantic ownership includes deferred hooks and existing shared dispatch.
    patch_ids: ClassVar[tuple[str, ...]] = (
        "GR-PATCH-002",
        "GR-PATCH-012",
        "GR-PATCH-016",
        "GR-PATCH-018",
        "GR-PATCH-019",
        "GR-PATCH-021",
    )

    def check_environment(self) -> dict[str, str]:
        from vllm_gr._env_check import check_ascend_environment

        return check_ascend_environment(require_npu_platform=True)

    def apply_runtime_patches(self, *, force: bool = False) -> int:
        del force  # Installers inspect their current targets on every call.
        from vllm_gr import npu_spawn_patch
        from vllm_gr.v1 import ascend_compat

        # Preserve spawn protection before native worker imports or process
        # creation. NPUWorker initializes Ascend ops before its ModelRunner;
        # keep that delayed import before the common Beam installer runs.
        spawn_count = int(npu_spawn_patch.apply_npu_spawn_patch())
        worker_count = ascend_compat.patch_npu_worker_init_device()
        return spawn_count + worker_count

    def get_model_runner_hooks(self) -> ModelRunnerPatchHooks:
        return cast(
            ModelRunnerPatchHooks,
            import_module("vllm_gr.v1.worker.npu.npu_model_runner_patch"),
        )

    def get_attention_execution(self) -> BeamAttentionExecution:
        return cast(
            BeamAttentionExecution,
            import_module("vllm_gr.v1.attention.backends.beam_attn_npu"),
        )

    def create_constrained_topk_execution(
        self,
        *,
        device: str | torch.device | None = None,
    ) -> "NPUConstrainedTopKExecution":
        """Create a fresh NPU constrained top-k execution adapter.

        The adapter is stateful (holds loaded tables + digest), so a new
        instance is created per call; the ModelRunner owns it for one
        lifetime and creates another after a reset.
        """
        from vllm_gr.constraints.execution import NPUConstrainedTopKExecution

        return NPUConstrainedTopKExecution(device=device)


NPU_BACKEND = NPUBackendProvider()


def apply_default_block_size(config: VllmConfig) -> None:
    """Use an Ascend-compatible default without overriding user intent.

    vLLM-Ascend 0.30 sets the block size to 128 itself when prefix caching or
    chunked prefill is enabled, but leaves the platform default in place
    otherwise, while the beam attention backends require the 128 kernel
    geometry. This rule closes that gap; it was validated on the 0.30.0rc2
    runtime.
    """
    import logging

    try:
        from vllm.platforms import current_platform
    except Exception:
        return
    if getattr(current_platform, "device_type", None) != "npu":
        return

    cache_config = getattr(config, "cache_config", None)
    model_config = getattr(config, "model_config", None)
    if cache_config is None:
        return
    if getattr(cache_config, "user_specified_block_size", False):
        return
    # Hybrid attention/state models have model-specific cache sizing rules in
    # vLLM-Ascend and must not be forced onto the dense-attention default.
    if model_config is not None and getattr(model_config, "is_hybrid", False):
        return
    if cache_config.block_size != 128:
        logging.getLogger(__name__).info(
            "Using Ascend-compatible default KV cache block size 128 "
            "(vLLM resolved the unspecified value to %s).",
            cache_config.block_size,
        )
        cache_config.block_size = 128


def default_constraint_backend(
    backend: ConstraintBackend,
    *,
    explicit: bool,
    constrained_generation: bool | None,
    beam_width_configured: bool,
    catalog_path: str | None,
) -> ConstraintBackend:
    """Select Constraint Table for implicit NPU Beam with a triples catalog."""
    from vllm_gr.config import ConstraintBackend

    if (
        explicit
        or constrained_generation is False
        or not beam_width_configured
        or catalog_path is None
        or backend is not ConstraintBackend.TRIE
    ):
        return backend

    from vllm_gr.utils.catalog import catalog_is_triples_document

    if catalog_is_triples_document(catalog_path):
        return ConstraintBackend.CONSTRAINT_TABLE
    return backend
