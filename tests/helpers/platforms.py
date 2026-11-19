# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Select the same simulated provider for config defaults and backend-plan admission."""

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import patch

from vllm_gr.platforms.ascend import NPU_BACKEND
from vllm_gr.platforms.cuda import CUDA_BACKEND
from vllm_gr.platforms.interface import GRBackendProvider
from vllm_gr.platforms.planning import admit_backend_plan


@contextmanager
def config_provider(device: str) -> Iterator[GRBackendProvider]:
    providers: dict[str, GRBackendProvider] = {"cuda": CUDA_BACKEND, "npu": NPU_BACKEND}
    provider = providers[device]
    with (
        patch("vllm_gr.platforms.config.get_active_backend", return_value=provider),
        patch("vllm_gr.platforms.loader.activate_current_backend", return_value=provider),
        # Config-only test modules may isolate planning by default.
        patch("vllm_gr.platforms.planning.admit_backend_plan", admit_backend_plan),
    ):
        yield provider
