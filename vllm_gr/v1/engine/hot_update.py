# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Compatibility import for :mod:`vllm_gr.engine.hot_update`."""

import sys

from vllm_gr.engine import hot_update as _impl

sys.modules[__name__] = _impl
