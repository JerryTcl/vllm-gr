# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Compatibility import for :mod:`vllm_gr.engine.core_client`."""

import sys

from vllm_gr.engine import core_client as _impl

sys.modules[__name__] = _impl
