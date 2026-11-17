# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Registration of backend providers shipped with vllm-gr."""

from __future__ import annotations

import logging

from .loader import disabled_builtin_backend_ids

logger = logging.getLogger("vllm_gr.platforms.builtins")


def register_builtin_backends() -> None:
    """Register the in-repository providers that were not explicitly disabled.

    ``VLLM_GR_DISABLE_BUILTIN_BACKENDS`` is read before registration so that an
    external distribution can replace a built-in ID deliberately. The switch
    must be set consistently in every participating process.
    """
    from .cuda import CUDA_BACKEND
    from .npu import NPU_BACKEND
    from .registry import register_backend

    disabled = disabled_builtin_backend_ids()
    for provider in (CUDA_BACKEND, NPU_BACKEND):
        if provider.backend_id in disabled:
            logger.warning(
                "vllm-gr built-in backend %r is disabled by configuration; "
                "an external provider must register this backend ID",
                provider.backend_id,
            )
            continue
        register_backend(provider, source=f"builtin:{provider.backend_id}")
