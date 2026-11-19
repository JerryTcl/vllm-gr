# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from ..config import BackendConfigService

if TYPE_CHECKING:
    from vllm_gr.config import GRConfig


class CudaConfigService(BackendConfigService):
    default_worker_decision = True
    supports_constraint_vocab_slicing = True

    def startup_inputs(self, gr: GRConfig, admitted: Mapping[str, Any] | None) -> dict[str, Any]:
        if admitted is not None:
            invariant = admitted.get("dense_prefill_batch_invariant", False)
        else:
            invariant = gr.deployment.dense_prefill != "off" and bool(
                int(os.environ.get("VLLM_BATCH_INVARIANT", "0"))
            )
        return {"dense_prefill_batch_invariant": invariant}

    def finalize_native_config(self, config: Any, gr: GRConfig) -> None:
        compilation = getattr(config, "compilation_config", None)
        if (
            gr.beam_max_width is not None
            and not getattr(config, "use_v2_model_runner", False)
            and compilation is not None
            and compilation.backend == "inductor"
        ):
            # Fix reduction geometry across AOT loads and preserve explicit overrides.
            compilation.inductor_compile_config.setdefault("dynamic_scale_rblock", False)
