# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Thin CUDA provider with lazy hardware-specific imports."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, ClassVar, cast

from .protocol import BeamAttentionExecution, ModelRunnerPatchHooks

if TYPE_CHECKING:
    import torch
    from vllm.config import VllmConfig

    from vllm_gr.config import GRConfig
    from vllm_gr.constraints.execution import ConstrainedTopKExecution


@dataclass(frozen=True)
class CUDABackendProvider:
    device_type: str = "cuda"
    # Semantic ownership includes hooks still installed by shared dispatch.
    patch_ids: ClassVar[tuple[str, ...]] = ("GR-PATCH-011",)

    def check_environment(self) -> dict[str, str]:
        from vllm.platforms import current_platform

        from vllm_gr._env_check import check_runtime_versions

        if str(current_platform.device_type) != "cuda":
            raise RuntimeError("CUDA provider requires the CUDA platform")

        report = check_runtime_versions()
        report["platform"] = "cuda"
        return report

    def apply_runtime_patches(self, *, force: bool = False) -> int:
        del force
        return 0

    def get_model_runner_hooks(self) -> ModelRunnerPatchHooks:
        return cast(
            ModelRunnerPatchHooks,
            import_module("vllm_gr.v1.worker.cuda.v1.gpu_model_runner_patch"),
        )

    def get_attention_execution(self) -> BeamAttentionExecution:
        return cast(
            BeamAttentionExecution,
            import_module("vllm_gr.v1.attention.backends.beam_attn_gpu"),
        )

    def create_constrained_topk_execution(
        self,
        *,
        device: str | torch.device | None = None,
    ) -> "ConstrainedTopKExecution | None":
        """Create the production CUDA Constraint Table execution adapter."""
        from vllm_gr.constraints.execution import CUDAConstrainedTopKExecution

        return CUDAConstrainedTopKExecution(device=device)


CUDA_BACKEND = CUDABackendProvider()


def validate_beam_runner(config: VllmConfig, effective_gr_config: GRConfig) -> None:
    from vllm_gr.config import BeamExecutionMode

    if (
        getattr(config, "use_v2_model_runner", False)
        and (
            effective_gr_config.beam_max_width is not None
            or (
                effective_gr_config.constraint_table is not None
                and effective_gr_config.constraint_table.enabled
            )
        )
        and effective_gr_config.beam_execution_mode is not BeamExecutionMode.V1
    ):
        raise ValueError(
            "ModelRunner V2 Beam requires beam.execution_mode='v1'; "
            "select it explicitly before engine initialization, or use ModelRunner V1"
        )


def apply_beam_compilation_defaults(config: VllmConfig) -> None:
    compilation = getattr(config, "compilation_config", None)
    if (
        not getattr(config, "use_v2_model_runner", False)
        and compilation is not None
        and compilation.backend == "inductor"
    ):
        # Inductor can benchmark an extra, halved reduction block on each
        # AOT load. Q/K norm rounding then changes across independent
        # engines, including Beam candidate order and scores. Keep the
        # compiled reduction geometry; record this in the native config
        # so the AOT cache key includes it. Explicit overrides still win.
        compilation.inductor_compile_config.setdefault("dynamic_scale_rblock", False)
