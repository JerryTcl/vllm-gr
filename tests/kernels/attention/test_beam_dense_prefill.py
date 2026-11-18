# mypy: disable-error-code="no-untyped-def,no-untyped-call"
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""The opt-in dense CUDA Prefill must match the paged Prefill on fresh prompts."""

from types import SimpleNamespace

import pytest
import torch

from vllm_gr.v1.attention.backends.beam_attn_gpu import prefill_attention

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    pytest.mark.l2,
]

HEADS, KV_HEADS, DIM, PAGE = 16, 8, 128, 16


def _case(lengths, padding):
    torch.manual_seed(611)
    total = sum(lengths)
    q = torch.randn(total + padding, HEADS, DIM, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(total + padding, KV_HEADS, DIM, device="cuda", dtype=q.dtype)
    v = torch.randn_like(k)
    request_pages = [-(-length // PAGE) for length in lengths]
    num_pages = sum(request_pages) + 3
    key_cache = torch.zeros(num_pages, PAGE, KV_HEADS, DIM, device="cuda", dtype=q.dtype)
    value_cache = torch.zeros_like(key_cache)
    # Shuffled pages catch accidental reliance on a contiguous cache layout.
    order = torch.randperm(num_pages, device="cuda").to(torch.int32)
    table = torch.zeros(len(lengths), max(request_pages), device="cuda", dtype=torch.int32)
    start = used = 0
    for b, length in enumerate(lengths):
        table[b, : request_pages[b]] = order[used : used + request_pages[b]]
        used += request_pages[b]
        rows = torch.arange(length, device="cuda")
        blocks = table[b, rows // PAGE].long()
        key_cache[blocks, rows % PAGE] = k[start : start + length]
        value_cache[blocks, rows % PAGE] = v[start : start + length]
        start += length
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda")
    meta = SimpleNamespace(
        query_start_loc=cu.to(torch.int32),
        max_query_len=max(lengths),
        seq_lens=torch.tensor(lengths, device="cuda", dtype=torch.int32),
        max_seq_len=max(lengths),
        block_table_tensor=table,
        causal=True,
        num_actual_tokens=total + padding,
        num_reqs=len(lengths),
    )
    return q, k, v, key_cache, value_cache, meta, total


@torch.inference_mode()
@pytest.mark.parametrize(
    "mode,lengths,padding",
    [
        ("fa2", (1000,), 0),
        ("fa2", (700, 1300, 64), 0),
        ("fa2", (900,), 124),
        ("cudnn", (1500,), 0),
        # Captured buckets pad past cu_seqlens_q[-1]; real rows must not change.
        ("cudnn", (1500,), 36),
        # Multi-request and short Prefill fall back to dense FA2.
        ("cudnn", (700, 1300), 0),
        ("cudnn", (1,), 0),
    ],
)
def test_dense_prefill_matches_paged(mode, lengths, padding):
    q, k, v, key_cache, value_cache, meta, total = _case(lengths, padding)
    common = {
        "query": q,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "attn_metadata": meta,
        "sliding_window_size": (-1, -1),
        "scale": DIM**-0.5,
        "logits_soft_cap": 0.0,
        "vllm_flash_attn_version": 2,
    }
    paged = torch.zeros_like(q)
    prefill_attention(output=paged, **common)
    dense = torch.zeros_like(q)
    prefill_attention(output=dense, dense_key=k, dense_value=v, dense_mode=mode, **common)
    torch.testing.assert_close(dense[:total].float(), paged[:total].float(), atol=2e-2, rtol=2e-2)


@torch.inference_mode()
def test_dense_prefill_declines_unsupported_layer():
    """Softcap is not handled by the dense kernels, so the paged path runs."""
    q, k, v, key_cache, value_cache, meta, total = _case((300,), 0)
    common = {
        "query": q,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "attn_metadata": meta,
        "sliding_window_size": (-1, -1),
        "scale": DIM**-0.5,
        "logits_soft_cap": 30.0,
        "vllm_flash_attn_version": 2,
    }
    paged = torch.zeros_like(q)
    prefill_attention(output=paged, **common)
    dense = torch.zeros_like(q)
    prefill_attention(output=dense, dense_key=k * 0, dense_value=v * 0, dense_mode="fa2", **common)
    torch.testing.assert_close(dense, paged, atol=0, rtol=0)
