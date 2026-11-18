# mypy: disable-error-code="no-untyped-def,no-untyped-call"
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
from types import SimpleNamespace

import pytest

from vllm_gr.v1.attention.backends import beam_attn


def _config(prefix_caching=False, chunked=False):
    return SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=prefix_caching),
        scheduler_config=SimpleNamespace(enable_chunked_prefill=chunked),
    )


@pytest.mark.parametrize(
    "raw,cuda,config,expected",
    [
        (None, True, _config(), None),
        ("off", True, _config(), None),
        ("fa2", True, _config(), "fa2"),
        (" CUDNN ", True, _config(), "cudnn"),
        ("flash", True, _config(), None),
        ("cudnn", False, _config(), None),
        # Cached context lives only in the paged cache.
        ("cudnn", True, _config(prefix_caching=True), None),
        ("cudnn", True, _config(chunked=True), None),
        ("cudnn", True, None, None),
    ],
)
def test_dense_prefill_mode_requires_fresh_prompts(monkeypatch, raw, cuda, config, expected):
    import vllm.config

    if raw is None:
        monkeypatch.delenv(beam_attn.ENV_DENSE_PREFILL, raising=False)
    else:
        monkeypatch.setenv(beam_attn.ENV_DENSE_PREFILL, raw)
    monkeypatch.setattr(beam_attn, "CUDA_DEVICE", cuda)
    monkeypatch.setattr(vllm.config, "get_current_vllm_config_or_none", lambda: config)
    assert beam_attn._resolve_dense_prefill_mode() == expected


@pytest.mark.parametrize(
    "changes,forwards",
    [
        ({}, True),
        ({"_dense_prefill_mode": None}, False),
        ({"kv_cache_dtype": "fp8_e4m3"}, False),
        ({"kv_sharing_target_layer_name": "layers.0.attn"}, False),
    ],
)
def test_dense_prefill_kwargs_forward_fresh_kv_only(changes, forwards):
    impl = SimpleNamespace(
        _dense_prefill_mode="fa2", kv_cache_dtype="auto", kv_sharing_target_layer_name=None
    )
    for name, value in changes.items():
        setattr(impl, name, value)
    key, value = object(), object()
    kwargs = beam_attn._dense_prefill_kwargs(impl, key, value)
    if forwards:
        assert kwargs == {"dense_key": key, "dense_value": value, "dense_mode": "fa2"}
    else:
        assert kwargs == {}
    assert beam_attn._dense_prefill_kwargs(impl, None, value) == {}


pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]
