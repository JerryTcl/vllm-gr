# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from vllm_gr.constraints.region_table import load_region_constraint_table


def _write_toy_region_trie(path: Path) -> str:
    l1 = np.zeros((2, 8), dtype=np.bool_)
    l1[0, 1] = True
    l1[1, [1, 7]] = True
    np.savez(
        path,
        L1_mask=l1,
        offsets_SB=np.asarray(
            [
                [0, 0, 2, 2, 2, 2, 2, 2, 2],
                [2, 2, 3, 3, 3, 3, 3, 3, 4],
            ],
            dtype=np.int32,
        ),
        flat_SB=np.asarray([2, 4, 2, 0], dtype=np.int16),
        offsets_SC=np.asarray([0, 1, 2, 3, 4], dtype=np.int32),
        flat_SC=np.asarray([3, 5, 6, 1], dtype=np.int16),
        pair_keys=np.asarray([10, 12, 74, 120], dtype=np.int32),
        regions=np.asarray(["th", "ph"]),
    )
    data = path.read_bytes()
    return "sha256:" + hashlib.sha256(data).hexdigest()


def test_loads_gr_novi_combined_region_trie_without_widening_tokens(tmp_path: Path) -> None:
    path = tmp_path / "trie_combined.npz"
    digest = _write_toy_region_trie(path)

    table = load_region_constraint_table(
        path,
        expected_digest=digest,
        expected_tokenizer_digest="sha256:" + "1" * 64,
        expected_vocab_size=8,
    )

    assert table.regions == ("th", "ph")
    assert table.first_token_offsets.tolist() == [0, 1, 3]
    assert table.first_token_ids.tolist() == [1, 1, 7]
    assert table.flat_SB.dtype == np.int16
    assert table.flat_SC.dtype == np.int16
    assert table.pair_keys.dtype == np.int32
    assert table.max_first_degree == 2
    assert table.metadata.max_prefix1_degree == 2
    assert table.metadata.max_prefix2_degree == 1


def test_region_trie_digest_is_verified_before_loading(tmp_path: Path) -> None:
    path = tmp_path / "trie_combined.npz"
    _write_toy_region_trie(path)

    with pytest.raises(ValueError, match="digest mismatch"):
        load_region_constraint_table(
            path,
            expected_digest="sha256:" + "0" * 64,
            expected_tokenizer_digest="sha256:" + "1" * 64,
            expected_vocab_size=8,
        )
