# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Worker-side beam decision for incremental beam decode (asynchronous scheduling).

The V1 stage runner supplies candidate tensors directly. Device-resident
BeamDecisionBuffers own cumulative scores and sequence history through Prefill
initialization and subsequent Decode updates. Selected tokens and parent
indices return to the stage runner for the same-step KV reorder and terminal
result construction.

The runner's stage binds the group and final selection operators before dispatch.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from vllm_gr.v1.beam.decision_ops import DecisionOps, reorder_sessions
from vllm_gr.v1.worker.runner_lifecycle import get_stage_execution


def init_beam_decision_state(
    session: Any,
    top_tokens: torch.Tensor,
    top_logprobs: torch.Tensor,
) -> None:
    """Seed the persistent decision buffers from the first candidate round.

    At the first
    decision step each beam (one row of ``[B*W, W]`` candidates) contributes
    its top candidate token and score to ``sequence[..., 0]`` and
    ``beam_scores``, so the subsequent ``beam_search_group`` steps read
    cumulative state entirely from device-resident buffers.  Runs once per
    session before any ``decode_select``-style step.

    Args:
        session: the active BeamSearchSession (its ``context.decision`` buffers).
        top_tokens / top_logprobs: ``[B * W, W]`` candidate token ids / probs.
    """
    bufs = getattr(session.context, "decision", None)
    if bufs is None:
        return
    width = session.beam_width
    n_rows = top_tokens.shape[0]
    if width == 0 or n_rows == 0 or n_rows % width != 0:
        return
    batch = n_rows // width
    # [B, W, K]: the last dim holds the top-candidate ordering per beam, and K
    # is the candidate width, which is independent of the beam width.
    # Column-0 views are already contiguous per-row, and the dtype guards
    # keep the constrained path (int32 ids / float32 probs straight from
    # rec_constrained_topk) copy-free — the profiled ~12ms-per-copy host
    # stalls (aclnnInplaceCopyGetWorkspaceSize) came from casting first and
    # copying the cast result; matching dtypes lets copy_ take the
    # contiguous fast path directly.
    candidate_width = top_tokens.shape[1]
    if candidate_width < 1:
        return
    best_token = top_tokens.reshape(batch, width, candidate_width)[:, :, 0]
    best_score = top_logprobs.reshape(batch, width, candidate_width)[:, :, 0]
    if best_token.dtype is not bufs.sequence.dtype:
        best_token = best_token.to(bufs.sequence.dtype)
    if best_score.dtype is not bufs.beam_scores.dtype:
        best_score = best_score.to(bufs.beam_scores.dtype)
    n = min(width, bufs.sequence.shape[1])
    bufs.sequence[:batch, :n, 0].copy_(best_token[:, :n])
    bufs.beam_scores[:batch, :n].copy_(best_score[:, :n])
    # Worker-side token-count for the decision: one committed token per beam.
    bufs.worker_decode_steps = 1


def init_beam_decision_from_parents(
    session: Any,
    parent_sequences: Sequence[Sequence[int]],
    parent_cumulative_logprobs: Sequence[float],
) -> None:
    """Restore persistent decision state from already-selected parents.

    The first grouped decode evaluates candidates after the prefill winner
    (for OneRec, ``s_b`` after ``s_a``).  Seeding from the current candidate
    matrix would lose the prefill score and count the current local score
    twice, changing beam rank order.
    """
    import math

    width = session.beam_width
    if len(parent_sequences) != width or len(parent_cumulative_logprobs) != width:
        raise RuntimeError(
            f"worker beam decision parent state must contain exactly beam_width={width} rows"
        )
    lengths = {len(sequence) for sequence in parent_sequences}
    if len(lengths) != 1:
        raise RuntimeError("worker beam decision parent sequences must have equal lengths")
    steps = next(iter(lengths), 0)
    bufs = session.context.decision
    if steps < 1 or steps > bufs.sequence.shape[2]:
        raise RuntimeError(
            f"worker beam decision parent sequence length {steps} is outside "
            f"[1, {bufs.sequence.shape[2]}]"
        )
    scores = [float(score) for score in parent_cumulative_logprobs]
    # -inf is a legitimate parent-score sentinel at the EOS-mask protocol
    # boundary. Only NaN/+inf are protocol errors.
    if any(math.isnan(score) or score == math.inf for score in scores):
        raise RuntimeError(
            "worker beam decision parent scores must not be NaN or +inf; "
            "-inf marks an EOS-masked beam and is a valid score"
        )
    device = bufs.sequence.device
    sequence_tensor = torch.tensor(parent_sequences, dtype=torch.int32, device=device)
    score_tensor = torch.tensor(scores, dtype=torch.float32, device=device)
    bufs.sequence[0, :width, :steps].copy_(sequence_tensor)
    bufs.beam_scores[0, :width].copy_(score_tensor)
    bufs.worker_decode_steps = steps


def _finalize_decision(
    session: Any,
    bufs: Any,
    out_ids: torch.Tensor,
    out_idx: torch.Tensor,
    out_lp: torch.Tensor,
    out_seq: torch.Tensor,
    prefix: torch.Tensor | None,
    batch: int,
    beam_width: int,
    result_width: int,
    *,
    reorder: Any = reorder_sessions,
) -> dict[str, torch.Tensor]:
    """Persist cumulative state and apply the asynchronous Beam decision.

    Non-final steps (``prefix`` is not None): convert the operator's GLOBAL
    ``request * beam_width`` offsets to request-local form for both the
    parent mapping and the prefix sums, persist
    ``group_token_num`` for the fused ``select_unshared_kv`` kernel, and
    reorder the KV pool in the SAME step.  Final steps (``prefix is None``):
    no reorder (select_unshared_kv is skipped on the last step).
    """
    # Persist cumulative state back into the fixed-address buffers.
    write_w = min(out_seq.shape[1], bufs.sequence.shape[1])
    bufs.sequence[:batch, :write_w, : out_seq.shape[2]].copy_(out_seq[:, :write_w])
    bufs.beam_scores[:batch, :result_width].copy_(out_lp[:, :result_width])

    decision = {
        "out_token_ids": out_ids,
        "out_token_index": out_idx,
        "out_log_probs": out_lp,
        "out_sequence": out_seq,
    }
    if prefix is None:
        return decision

    request_offset = (
        torch.arange(batch, dtype=torch.int32, device=prefix.device).unsqueeze(1) * beam_width
    )
    local_idx = out_idx[:batch, :beam_width].to(torch.int32) - request_offset
    local_prefix = prefix[:batch, :beam_width].to(torch.int32) - request_offset
    bufs.group_token_num[:batch, :beam_width].copy_(local_prefix)

    sessions = getattr(session, "sessions", (session,))
    if len(sessions) != batch:
        raise RuntimeError("Beam decision batch does not match its request sessions")
    if not all(item._state.is_last_step for item in sessions):
        reorder(session, sessions, local_idx, local_prefix)
    return decision


def _run_decision(
    ops: DecisionOps,
    session: Any,
    tokens: torch.Tensor,
    probs: torch.Tensor,
    last: bool,
    result_width: int,
    *,
    strict: bool,
) -> dict[str, torch.Tensor]:
    bufs = session.context.decision
    width = session.beam_width
    batch = tokens.shape[0] // width
    step = bufs.worker_decode_steps
    scores = bufs.beam_scores[:batch, :width].reshape(batch * width, 1)
    history = bufs.sequence[:batch, :width, : step + 1].contiguous()
    ids, parents, log_probs, prefix, sequence, result_width = ops.select(
        scores, tokens, probs, history, step, width, result_width, last, strict
    )
    bufs.worker_decode_steps = step + 1
    return _finalize_decision(
        session,
        bufs,
        ids.reshape(batch, result_width),
        parents.reshape(batch, result_width),
        log_probs.reshape(batch, result_width),
        sequence,
        prefix,
        batch,
        width,
        result_width,
        reorder=ops.reorder,
    )


def run_beam_decision_candidates(
    runner: Any,
    session: Any,
    top_tokens: torch.Tensor,
    top_probs: torch.Tensor,
    *,
    is_last_step: bool,
    result_width: int | None = None,
) -> dict[str, torch.Tensor]:
    """Run a worker decision from canonical fixed-width candidates.

    The NPU eager GR runner already owns ``ConstraintOutput`` tensors, so
    routing them through vLLM's sampler container would only reconstruct an
    object that the constrained backend has just deconstructed. This narrow
    entry point shares the persistent decision and KV-reorder machinery while
    keeping the canonical candidate ABI explicit.
    """
    width = session.beam_width
    if (
        top_tokens.ndim != 2
        or top_probs.shape != top_tokens.shape
        or top_tokens.shape[1] < 1
        or top_tokens.shape[0] == 0
        or top_tokens.shape[0] % width
    ):
        raise ValueError("Beam candidates must have shape [B*W, K] with K >= 1")
    if top_tokens.dtype is not torch.int32 or top_probs.dtype is not torch.float32:
        raise ValueError("Beam candidates must use int32 token IDs and float32 scores")
    if session.context.decision.worker_decode_steps == 0:
        raise RuntimeError("Beam decision state must be initialized from Prefill")

    result_width = result_width or width
    if not 1 <= result_width <= width:
        raise ValueError("result_width must be in [1, beam_width]")
    decision = _run_decision(
        get_stage_execution(runner).decision,
        session,
        top_tokens,
        top_probs,
        is_last_step,
        result_width,
        strict=True,
    )
    batch = top_tokens.shape[0] // width
    decision["probs"] = session.context.decision.beam_scores[:batch, :width].clone()
    return decision
