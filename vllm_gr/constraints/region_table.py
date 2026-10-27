# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Validated host representation of a GR-NOVI compatible combined Region Trie."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vllm_gr.config.models import REGION_TRIE_FORMAT
from vllm_gr.config.validation import (
    validate_artifact_digest,
    validate_tokenizer_digest,
)
from vllm_gr.constraints.table import ConstraintTableMetadata

REGION_TRIE_TENSOR_NAMES = (
    "first_token_offsets",
    "first_token_ids",
    "offsets_SB",
    "flat_SB",
    "offsets_SC",
    "flat_SC",
    "pair_keys",
)
_NPZ_TENSOR_NAMES = (
    "L1_mask",
    "offsets_SB",
    "flat_SB",
    "offsets_SC",
    "flat_SC",
    "pair_keys",
)
_COMBINED_NAME = "trie_combined.npz"
# The production combined Trie is larger than the canonical single-catalog
# artifact. Keep a hard guard, but do not inherit the latter's 512 MiB cap.
MAX_REGION_TRIE_BYTES = 4 * 1024 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class RegionConstraintTable:
    """One immutable GPU-oriented Trie containing one or more named regions.

    ``offsets_SB`` retains the GR-NOVI ``[R, V + 1]`` absolute-offset layout.
    The remaining child arrays are concatenated across regions. Token arrays
    may stay int16, avoiding an eager 2x expansion of production artifacts.
    """

    first_token_offsets: NDArray[np.int32]
    first_token_ids: NDArray[np.int32]
    offsets_SB: NDArray[np.int32]
    flat_SB: NDArray[np.signedinteger[Any]]
    offsets_SC: NDArray[np.int32]
    flat_SC: NDArray[np.signedinteger[Any]]
    pair_keys: NDArray[np.signedinteger[Any]]
    metadata: ConstraintTableMetadata
    max_first_degree: int

    def __post_init__(self) -> None:
        for name in REGION_TRIE_TENSOR_NAMES:
            value = getattr(self, name)
            frozen = np.frombuffer(
                np.ascontiguousarray(value).tobytes(), dtype=value.dtype
            ).reshape(value.shape)
            object.__setattr__(self, name, frozen)

    @property
    def regions(self) -> tuple[str, ...]:
        return self.metadata.regions

    @property
    def region_count(self) -> int:
        return len(self.regions)

    @property
    def total_bytes(self) -> int:
        return sum(int(getattr(self, name).nbytes) for name in REGION_TRIE_TENSOR_NAMES)


def _resolve_npz(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate / _COMBINED_NAME if candidate.is_dir() else candidate


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                hasher.update(chunk)
    except OSError as exc:
        raise ValueError(f"cannot read Region Trie artifact {str(path)!r}: {exc}") from exc
    return "sha256:" + hasher.hexdigest()


def _manifest_regions(path: Path) -> tuple[str, ...]:
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.is_file():
        return ()
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read Region Trie manifest {str(manifest_path)!r}: {exc}") from exc
    values = document.get("combined", {}).get("regions")
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        return ()
    return tuple(values)


def _regions_from_archive(archive: Any, path: Path) -> tuple[str, ...]:
    if "regions" in archive.files:
        values = archive["regions"]
        if values.ndim != 1:
            raise ValueError("Region Trie regions must be a one-dimensional string array")
        regions = tuple(
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in values.tolist()
        )
    else:
        regions = _manifest_regions(path)
    if not regions or any(not value for value in regions) or len(set(regions)) != len(regions):
        raise ValueError("Region Trie requires non-empty unique region names")
    return regions


def _signed_array(array: NDArray[Any], name: str, allowed: tuple[np.dtype[Any], ...]) -> None:
    if array.dtype not in allowed:
        expected = "/".join(str(value) for value in allowed)
        raise ValueError(f"Region Trie {name}.dtype must be {expected}, got {array.dtype}")


def _int32_offsets(array: NDArray[Any], name: str) -> NDArray[np.int32]:
    """Match GR-NOVI's device ABI while rejecting lossy offset conversion."""

    if not np.issubdtype(array.dtype, np.signedinteger):
        raise ValueError(f"Region Trie {name} must contain signed integers")
    if array.size and (
        int(array.min()) < np.iinfo(np.int32).min or int(array.max()) > np.iinfo(np.int32).max
    ):
        raise ValueError(f"Region Trie {name} exceeds int32 capacity")
    return np.ascontiguousarray(array, dtype=np.int32)


def _token_array(array: NDArray[Any], name: str) -> NDArray[np.signedinteger[Any]]:
    """Retain GR-NOVI's compact token dtype, accepting safe wider inputs."""

    if not np.issubdtype(array.dtype, np.signedinteger):
        raise ValueError(f"Region Trie {name} must contain signed integers")
    if array.dtype in (np.dtype(np.int16), np.dtype(np.int32)):
        return np.ascontiguousarray(array)
    if array.size and (
        int(array.min()) < np.iinfo(np.int32).min or int(array.max()) > np.iinfo(np.int32).max
    ):
        raise ValueError(f"Region Trie {name} exceeds int32 capacity")
    return np.ascontiguousarray(array, dtype=np.int32)


def _pair_key_array(array: NDArray[Any]) -> NDArray[np.signedinteger[Any]]:
    if not np.issubdtype(array.dtype, np.signedinteger):
        raise ValueError("Region Trie pair_keys must contain signed integers")
    if array.dtype in (np.dtype(np.int32), np.dtype(np.int64)):
        return np.ascontiguousarray(array)
    return np.ascontiguousarray(array, dtype=np.int64)


def _validate_region_table(table: RegionConstraintTable) -> None:
    metadata = table.metadata
    vocab_size = metadata.vocab_size
    region_count = table.region_count
    if metadata.format != REGION_TRIE_FORMAT:
        raise ValueError(f"Region Trie format must be {REGION_TRIE_FORMAT!r}")
    if table.first_token_offsets.shape != (region_count + 1,):
        raise ValueError("first_token_offsets must contain region_count + 1 entries")
    if table.offsets_SB.shape != (region_count, vocab_size + 1):
        raise ValueError("offsets_SB must have shape [region_count, vocab_size + 1]")
    if table.first_token_ids.ndim != 1 or table.flat_SB.ndim != 1:
        raise ValueError("Region Trie token tables must be one-dimensional")
    if table.offsets_SC.ndim != 1 or table.flat_SC.ndim != 1 or table.pair_keys.ndim != 1:
        raise ValueError("Region Trie pair tables must be one-dimensional")
    _signed_array(table.first_token_ids, "first_token_ids", (np.dtype(np.int32),))
    _signed_array(table.offsets_SB, "offsets_SB", (np.dtype(np.int32),))
    _signed_array(table.offsets_SC, "offsets_SC", (np.dtype(np.int32),))
    _signed_array(table.flat_SB, "flat_SB", (np.dtype(np.int16), np.dtype(np.int32)))
    _signed_array(table.flat_SC, "flat_SC", (np.dtype(np.int16), np.dtype(np.int32)))
    _signed_array(table.pair_keys, "pair_keys", (np.dtype(np.int32), np.dtype(np.int64)))

    first_offsets = table.first_token_offsets
    if int(first_offsets[0]) != 0 or int(first_offsets[-1]) != table.first_token_ids.size:
        raise ValueError("first_token_offsets must cover first_token_ids")
    if np.any(np.diff(first_offsets.astype(np.int64)) < 0):
        raise ValueError("first_token_offsets must be non-decreasing")

    offsets_sb = table.offsets_SB.astype(np.int64, copy=False)
    if int(offsets_sb[0, 0]) != 0 or int(offsets_sb[-1, -1]) != table.flat_SB.size:
        raise ValueError("offsets_SB must cover flat_SB with absolute offsets")
    if np.any(np.diff(offsets_sb, axis=1) < 0):
        raise ValueError("offsets_SB rows must be non-decreasing")
    if region_count > 1 and np.any(offsets_sb[:-1, -1] != offsets_sb[1:, 0]):
        raise ValueError("adjacent offsets_SB region ranges must be contiguous")

    if table.pair_keys.size != table.flat_SB.size:
        raise ValueError("pair_keys must contain one key per (region, SA, SB) pair")
    if table.offsets_SC.size != table.pair_keys.size + 1:
        raise ValueError("offsets_SC must contain pair_count + 1 entries")
    offsets_sc = table.offsets_SC.astype(np.int64, copy=False)
    if int(offsets_sc[0]) != 0 or int(offsets_sc[-1]) != table.flat_SC.size:
        raise ValueError("offsets_SC must cover flat_SC")
    if np.any(np.diff(offsets_sc) < 0):
        raise ValueError("offsets_SC must be non-decreasing")
    if offsets_sc.size > 1 and np.any(np.diff(offsets_sc) <= 0):
        raise ValueError("every Region Trie (region, SA, SB) pair must have an SC child")
    keys = table.pair_keys.astype(np.int64, copy=False)
    if keys.size > 1 and np.any(np.diff(keys) <= 0):
        raise ValueError("pair_keys must be strictly increasing")
    key_limit = region_count * vocab_size * vocab_size
    if keys.size and (int(keys[0]) < 0 or int(keys[-1]) >= key_limit):
        raise ValueError("pair_keys contain an out-of-range region/SID key")
    for region_id in range(region_count):
        region_base = region_id * vocab_size * vocab_size
        for first_id in range(vocab_size):
            begin = int(offsets_sb[region_id, first_id])
            end = int(offsets_sb[region_id, first_id + 1])
            if begin == end:
                continue
            expected_keys = (
                region_base
                + first_id * vocab_size
                + table.flat_SB[begin:end].astype(np.int64, copy=False)
            )
            if not np.array_equal(keys[begin:end], expected_keys):
                raise ValueError(
                    "pair_keys must encode region * V^2 + SA * V + SB for every offsets_SB bucket"
                )

    for name, values in (
        ("first_token_ids", table.first_token_ids),
        ("flat_SB", table.flat_SB),
        ("flat_SC", table.flat_SC),
    ):
        if values.size and (int(values.min()) < 0 or int(values.max()) >= vocab_size):
            raise ValueError(f"Region Trie {name} contains an out-of-range token ID")
    active = np.diff(offsets_sb, axis=1) > 0
    expected_first = np.zeros((region_count, vocab_size), dtype=np.bool_)
    for region_id in range(region_count):
        begin = int(first_offsets[region_id])
        end = int(first_offsets[region_id + 1])
        expected_first[region_id, table.first_token_ids[begin:end]] = True
    if not np.array_equal(active, expected_first):
        raise ValueError("Region Trie first-token rows do not match offsets_SB active prefixes")
    if table.total_bytes != metadata.total_bytes:
        raise ValueError("Region Trie metadata total_bytes does not match its tensors")
    if table.total_bytes > MAX_REGION_TRIE_BYTES:
        raise ValueError(
            f"Region Trie total_bytes {table.total_bytes} exceeds {MAX_REGION_TRIE_BYTES}"
        )


def load_region_constraint_table(
    path: str | Path,
    *,
    expected_digest: str,
    expected_tokenizer_digest: str,
    expected_vocab_size: int,
) -> RegionConstraintTable:
    """Load the production ``trie_combined.npz`` representation fail-closed."""

    digest = validate_artifact_digest(expected_digest)
    tokenizer_digest = validate_tokenizer_digest(expected_tokenizer_digest)
    if digest is None or tokenizer_digest is None:
        raise ValueError("Region Trie artifact and tokenizer digests are required")
    artifact_path = _resolve_npz(path)
    actual_digest = _sha256_file(artifact_path)
    if actual_digest != digest:
        raise ValueError(
            f"Region Trie artifact digest mismatch: expected {digest}, got {actual_digest}"
        )
    try:
        with np.load(artifact_path, allow_pickle=False) as archive:
            missing = sorted(set(_NPZ_TENSOR_NAMES) - set(archive.files))
            if missing:
                raise ValueError(f"Region Trie artifact is missing tensors {missing}")
            regions = _regions_from_archive(archive, artifact_path)
            l1_mask = np.ascontiguousarray(archive["L1_mask"], dtype=np.bool_)
            offsets_sb = _int32_offsets(archive["offsets_SB"], "offsets_SB")
            flat_sb = _token_array(archive["flat_SB"], "flat_SB")
            offsets_sc = _int32_offsets(archive["offsets_SC"], "offsets_SC")
            flat_sc = _token_array(archive["flat_SC"], "flat_SC")
            pair_keys = _pair_key_array(archive["pair_keys"])
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("Region Trie"):
            raise
        raise ValueError(f"cannot load Region Trie artifact {str(artifact_path)!r}: {exc}") from exc

    if l1_mask.ndim != 2:
        raise ValueError("Region Trie L1_mask must have shape [region_count, vocab_size]")
    region_count, vocab_size = (int(value) for value in l1_mask.shape)
    if region_count != len(regions):
        raise ValueError("Region Trie names do not match L1_mask region_count")
    if vocab_size != expected_vocab_size:
        raise ValueError(
            f"Region Trie vocab_size mismatch: expected {expected_vocab_size}, got {vocab_size}"
        )
    if offsets_sb.shape != (region_count, vocab_size + 1):
        raise ValueError("Region Trie offsets_SB must have shape [region_count, vocab_size + 1]")
    if any(value.ndim != 1 for value in (flat_sb, offsets_sc, flat_sc, pair_keys)):
        raise ValueError("Region Trie flat and pair tables must be one-dimensional")

    counts = l1_mask.sum(axis=1, dtype=np.int64)
    if int(counts.sum()) > np.iinfo(np.int32).max:
        raise ValueError("Region Trie first-token count exceeds int32 capacity")
    first_token_offsets = np.zeros(region_count + 1, dtype=np.int32)
    np.cumsum(counts, out=first_token_offsets[1:])
    first_token_ids = np.nonzero(l1_mask)[1].astype(np.int32, copy=False)
    max_first_degree = int(counts.max(initial=0))
    max_prefix1_degree = int(
        np.diff(offsets_sb.astype(np.int64, copy=False), axis=1).max(initial=0)
    )
    max_prefix2_degree = int(np.diff(offsets_sc.astype(np.int64, copy=False)).max(initial=0))
    total_bytes = sum(
        int(value.nbytes)
        for value in (
            first_token_offsets,
            first_token_ids,
            offsets_sb,
            flat_sb,
            offsets_sc,
            flat_sc,
            pair_keys,
        )
    )
    table = RegionConstraintTable(
        first_token_offsets=first_token_offsets,
        first_token_ids=first_token_ids,
        offsets_SB=offsets_sb,
        flat_SB=flat_sb,
        offsets_SC=offsets_sc,
        flat_SC=flat_sc,
        pair_keys=pair_keys,
        metadata=ConstraintTableMetadata(
            format=REGION_TRIE_FORMAT,
            artifact_digest=actual_digest,
            tokenizer_digest=tokenizer_digest,
            vocab_size=vocab_size,
            max_prefix1_degree=max_prefix1_degree,
            max_prefix2_degree=max_prefix2_degree,
            total_bytes=total_bytes,
            regions=regions,
        ),
        max_first_degree=max_first_degree,
    )
    _validate_region_table(table)
    return table


__all__ = [
    "MAX_REGION_TRIE_BYTES",
    "REGION_TRIE_TENSOR_NAMES",
    "RegionConstraintTable",
    "load_region_constraint_table",
]
