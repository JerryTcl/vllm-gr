# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
import torch

from vllm_gr.ops.cuda.rec_constrained_topk import rec_constrained_topk_cuda

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")


def _tables(device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        torch.tensor([1, 1, 7], dtype=torch.int32, device=device),
        torch.tensor(
            [
                0,
                0,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                3,
                3,
                3,
                3,
                3,
                3,
                4,
            ],
            dtype=torch.int32,
            device=device,
        ),
        torch.tensor([2, 4, 2, 0], dtype=torch.int16, device=device),
        torch.tensor([10, 12, 74, 120], dtype=torch.int32, device=device),
        torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32, device=device),
        torch.tensor([3, 5, 6, 1], dtype=torch.int16, device=device),
    )


@pytest.mark.parametrize(  # type: ignore[misc]
    ("step", "prefixes", "expected"),
    [
        (0, [[-1, -1, -1]] * 3, [[1], [7, 1], [6, 7]]),
        (1, [[1, -1, -1], [1, -1, -1], [5, -1, -1]], [[4, 2], [2], [6, 7]]),
        (2, [[1, 2, -1], [1, 2, -1], [5, 5, -1]], [[3], [6], [6, 7]]),
    ],
)
def test_region_rows_and_unknown_rows_select_independently(
    step: int, prefixes: list[list[int]], expected: list[list[int]]
) -> None:
    device = torch.device("cuda")
    logits = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 6.0],
        ],
        dtype=torch.float16,
        device=device,
    )
    first, offsets1, values1, keys, offsets2, values2 = _tables(device)
    tokens, _, valid, counts = rec_constrained_topk_cuda(
        logits,
        torch.tensor(prefixes, dtype=torch.int32, device=device),
        first,
        offsets1,
        values1,
        keys,
        offsets2,
        values2,
        torch.ones(3, dtype=torch.float32, device=device),
        step,
        2,
        2,
        1,
        region_ids=torch.tensor([0, 1, -1], dtype=torch.int32, device=device),
        first_token_offsets=torch.tensor([0, 1, 3], dtype=torch.int32, device=device),
        region_count=2,
        max_first_degree=2,
        allow_unconstrained=True,
    )
    torch.accelerator.synchronize()

    actual = [row[mask].tolist() for row, mask in zip(tokens.cpu(), valid.cpu())]
    assert actual == expected
    assert counts.cpu().tolist() == [len(expected[0]), len(expected[1]), 8]
