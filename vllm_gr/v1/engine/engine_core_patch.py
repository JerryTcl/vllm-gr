# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Legacy scheduler and KV-cache integration pending the RFC 751 PR1 extraction."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from functools import wraps

import msgspec
import numpy as np
from vllm.logger import init_logger
from vllm.utils.hashing import sha256 as _vllm_sha256

from vllm_gr.engine.engine_core_patch import (
    apply_engine_core_child_patches as apply_engine_core_child_patches,
)
from vllm_gr.engine.engine_core_patch import (
    apply_engine_core_request_patches as apply_engine_core_request_patches,
)
from vllm_gr.engine.engine_core_patch import (
    make_patched_engine_init as make_patched_engine_init,
)
from vllm_gr.engine.engine_core_patch import (
    make_patched_handle_client_request as make_patched_handle_client_request,
)
from vllm_gr.engine.engine_core_patch import (
    make_patched_process_engine_step as make_patched_process_engine_step,
)
from vllm_gr.engine.engine_core_patch import (
    patch_engine_core_process_entrypoint as patch_engine_core_process_entrypoint,
)
from vllm_gr.engine.engine_core_patch import run_engine_core as run_engine_core

logger = init_logger(__name__)

_SCHEDULER_PATCH_MARKER = "_vllm_gr_beam_scheduler_patch"
_GR_FINALIZING_IDS_ATTR = "_gr_v1_finalizing_request_ids"
_GR_NATIVE_UPDATE_IDS_ATTR = "_gr_v1_native_update_request_ids"
_GR_RESET_PREFIX_CACHE_PATCH_MARKER = "_vllm_gr_v1_reset_prefix_cache"
_GR_SCHEDULE_PATCH_DEPTH_ATTR = "_vllm_gr_schedule_patch_depth"


def __getattr__(name: str):
    # The upstream socket loop is captured at installation time.
    if name == "UPSTREAM_PROCESS_INPUT_SOCKETS":
        from vllm_gr.engine import engine_core_patch

        return engine_core_patch.UPSTREAM_PROCESS_INPUT_SOCKETS
    raise AttributeError(name)


def _expand_gr_abort_request_ids(scheduler, request_ids, finished_status):
    """Expand cancellation of any GR member to its complete fixed batch."""
    if request_ids is None or getattr(finished_status, "name", "") != "FINISHED_ABORTED":
        return request_ids
    if request_ids is None:
        return None
    requested = (request_ids,) if isinstance(request_ids, str) else tuple(request_ids)
    sessions = getattr(scheduler, "_gr_v1_sessions", {})
    request_sessions = getattr(scheduler, "_gr_v1_request_sessions", {})
    expanded: list[str] = []
    seen: set[str] = set()
    for request_id in requested:
        session_id = request_sessions.get(request_id, request_id)
        session = sessions.get(session_id)
        members = session.request_ids if session is not None else (request_id,)
        for member_id in members:
            if member_id in seen:
                continue
            seen.add(member_id)
            expanded.append(member_id)
    return expanded


def _sync_gr_sessions_after_native_finish(scheduler, finished_requests, finished_status) -> None:
    """Mirror the native finished Request objects into GR session state."""
    from vllm_gr.v1.engine.beam_search_v1 import GRSessionStatus

    status_name = getattr(finished_status, "name", "")
    sessions = getattr(scheduler, "_gr_v1_sessions", {})
    request_sessions = getattr(scheduler, "_gr_v1_request_sessions", {})
    gr_finalizing_ids = getattr(scheduler, _GR_FINALIZING_IDS_ATTR, ())
    gr_native_update_ids = getattr(scheduler, _GR_NATIVE_UPDATE_IDS_ATTR, ())
    for request in finished_requests:
        req_id = request.request_id
        session = sessions.get(request_sessions.get(req_id, req_id))
        if session is None:
            continue
        # Native EOS/stop detection may finish a Request while processing a GR
        # dispatch. The update wrapper reconciles that transition after GR has
        # consumed the same output; it is not an external abort/error.
        if req_id in gr_native_update_ids:
            continue
        gr_terminal_delivery = (
            req_id in gr_finalizing_ids
            and session.status == GRSessionStatus.FINISHING
            and session.terminal_result_pending
        )
        if gr_terminal_delivery:
            continue
        if status_name == "FINISHED_ABORTED":
            session.status = GRSessionStatus.ABORTED
        elif status_name == "FINISHED_ERROR":
            session.status = GRSessionStatus.FAILED
        else:
            session.status = GRSessionStatus.FINISHING
        session.terminal_result_pending = False

        from vllm_gr.v1.engine.gr_async_scheduler import _drain_resources

        _drain_resources(scheduler, session)


def _patch_scheduler_reset_prefix_cache(scheduler_cls) -> None:
    original_reset = scheduler_cls.reset_prefix_cache
    if getattr(original_reset, _GR_RESET_PREFIX_CACHE_PATCH_MARKER, False):
        return

    @wraps(original_reset)
    def patched_reset_prefix_cache(self, *args, **kwargs):
        reset_running_requests = kwargs.get("reset_running_requests", args[0] if args else False)
        from vllm_gr.v1.engine.beam_search_v1 import GRResourceStatus, GRSessionStatus

        # Native forced reset cannot safely preempt active GR resources. Weight
        # updates may leave unstarted sessions queued after pause/drain, so allow
        # cache clearing for those while still rejecting executing/retiring work.
        sessions = getattr(self, "_gr_v1_sessions", {}).values()
        if reset_running_requests and any(
            session.status is not GRSessionStatus.WAITING
            or session.resource_status is not GRResourceStatus.WAITING
            or session.in_flight_stages
            for session in sessions
        ):
            logger.warning(
                "Rejecting reset_prefix_cache(reset_running_requests=True) "
                "while GR sessions are active"
            )
            return False
        return original_reset(self, *args, **kwargs)

    setattr(patched_reset_prefix_cache, _GR_RESET_PREFIX_CACHE_PATCH_MARKER, True)
    scheduler_cls.reset_prefix_cache = patched_reset_prefix_cache


def apply_scheduler_patch():
    """Monkey-patch Scheduler.schedule to propagate custom attributes via SchedulerOutput."""
    from vllm.v1.core.sched.scheduler import Scheduler

    from vllm_gr.v1.engine.gr_async_scheduler import (
        GR_STAGE_OUTPUT_ATTR,
        GRAdmissionError,
        _prepare_gr_session,
        _register_gr_session,
        has_gr_cleanup_work,
        install_gr_scheduler_methods,
    )

    install_gr_scheduler_methods(Scheduler)

    if not getattr(Scheduler.has_requests, "_gr_retirement_work", False):
        original_has_requests = Scheduler.has_requests

        @wraps(original_has_requests)
        def has_requests(self):
            return original_has_requests(self) or has_gr_cleanup_work(self)

        has_requests._gr_retirement_work = True
        Scheduler.has_requests = has_requests

    if not getattr(Scheduler.add_request, "_vllm_gr_v1_add_request", False):
        original_add_request = Scheduler.add_request

        @wraps(original_add_request)
        def patched_add_request(self, request):
            batch_request = getattr(request, "gr_batch_request", None)
            sessions = getattr(self, "_gr_v1_sessions", {})
            if (batch_request is None and sessions) or (
                batch_request is not None
                and any(
                    getattr(r, "gr_batch_request", None) is None for r in self.requests.values()
                )
            ):
                raise GRAdmissionError(
                    "Asynchronous scheduling requests cannot mix with ordinary or legacy Beam requests"
                )
            if batch_request is not None:
                member_index = getattr(request, "gr_batch_member_index", None)
                matching_indices = tuple(
                    index
                    for index, item in enumerate(batch_request.items)
                    if item.request.request_id == request.request_id
                )
                if len(matching_indices) != 1 or (
                    member_index is not None
                    and (type(member_index) is not int or member_index != matching_indices[0])
                ):
                    raise GRAdmissionError("native request does not match its GR batch member")
                if request.request_id in self.requests:
                    raise GRAdmissionError(f"GR request already exists: {request.request_id}")
                session = sessions.get(batch_request.batch_id)
                if session is not None:
                    if session.request is not batch_request:
                        raise GRAdmissionError(
                            f"GR session already exists: {batch_request.batch_id}"
                        )
                    return original_add_request(self, request)
                # The caller supplied the complete immutable batch. Validate
                # it once before publishing the first native member; the core
                # handles every member synchronously before it can schedule.
                duplicate_id = next(
                    (
                        item.request.request_id
                        for item in batch_request.items
                        if item.request.request_id in self.requests
                    ),
                    None,
                )
                if duplicate_id is not None:
                    raise GRAdmissionError(f"GR request already exists: {duplicate_id}")
                session = _prepare_gr_session(self, batch_request)
                result = original_add_request(self, request)
                _register_gr_session(self, session)
                return result
            return original_add_request(self, request)

        setattr(patched_add_request, "_vllm_gr_v1_add_request", True)
        Scheduler.add_request = patched_add_request

    if not getattr(Scheduler.finish_requests, "_vllm_gr_v1_finish_requests", False):
        original_finish_requests = Scheduler.finish_requests

        @wraps(original_finish_requests)
        def patched_finish_requests(self, request_ids, finished_status):
            request_ids = _expand_gr_abort_request_ids(self, request_ids, finished_status)
            result = original_finish_requests(self, request_ids, finished_status)
            _sync_gr_sessions_after_native_finish(self, result, finished_status)
            return result

        setattr(patched_finish_requests, "_vllm_gr_v1_finish_requests", True)
        Scheduler.finish_requests = patched_finish_requests

    _patch_scheduler_reset_prefix_cache(Scheduler)

    if getattr(Scheduler.schedule, _SCHEDULER_PATCH_MARKER, False):
        return

    _original_schedule = Scheduler.schedule

    @wraps(_original_schedule)
    def patched_schedule(self, throttle_prefills: bool = False):
        # Some platform plugins replace Scheduler with a subclass whose
        # schedule() delegates to super().schedule(). If both implementations
        # were patched, that produces nested GR wrappers around one native
        # scheduling pass. Only the outermost wrapper may apply admission and
        # annotate the SchedulerOutput; the inner wrapper must preserve the
        # subclass' normal super() chain without processing the output twice.
        schedule_patch_depth = getattr(self, _GR_SCHEDULE_PATCH_DEPTH_ATTR, 0)
        if schedule_patch_depth:
            return _original_schedule(self, throttle_prefills=throttle_prefills)

        # Cache only real prompt KV for V1: generated tokens are device-owned
        # placeholders. Ordinary requests retain native cache semantics.
        original_cache_blocks = self.kv_cache_manager.coordinator.cache_blocks
        cached_patch = getattr(self, "_gr_cache_blocks_patch", None)
        if cached_patch is None or cached_patch[0] is not original_cache_blocks:

            def cache_prompt_blocks(request, num_computed_tokens):
                if getattr(request, "gr_batch_request", None) is not None:
                    num_computed_tokens = min(num_computed_tokens, request.num_prompt_tokens)
                return original_cache_blocks(request, num_computed_tokens)

            cached_patch = (original_cache_blocks, cache_prompt_blocks)
            self._gr_cache_blocks_patch = cached_patch
        self.kv_cache_manager.coordinator.cache_blocks = cached_patch[1]
        from vllm_gr.v1.engine.gr_async_scheduler import (
            attach_gr_retirement,
            gr_admission,
        )

        setattr(self, _GR_SCHEDULE_PATCH_DEPTH_ATTR, schedule_patch_depth + 1)
        try:
            with gr_admission(self):
                output = _original_schedule(self, throttle_prefills=throttle_prefills)

            gr_stages = {}
            sessions = getattr(self, "_gr_v1_sessions", {})
            request_sessions = getattr(self, "_gr_v1_request_sessions", {})
            scheduled_session_ids = dict.fromkeys(
                request_sessions[req_id]
                for req_id in output.num_scheduled_tokens
                if req_id in request_sessions
            )
            for session_id in scheduled_session_ids:
                session = sessions.get(session_id)
                if session is None:
                    continue
                metadata = self.attach_gr_stage_metadata(output, session)
                if metadata is not None:
                    gr_stages[session_id] = metadata
            setattr(output, GR_STAGE_OUTPUT_ATTR, gr_stages)
            attach_gr_retirement(self, output)
            return output
        finally:
            try:
                self.kv_cache_manager.coordinator.cache_blocks = original_cache_blocks
            finally:
                if schedule_patch_depth:
                    setattr(self, _GR_SCHEDULE_PATCH_DEPTH_ATTR, schedule_patch_depth)
                else:
                    delattr(self, _GR_SCHEDULE_PATCH_DEPTH_ATTR)

    Scheduler.schedule = patched_schedule
    setattr(Scheduler.schedule, _SCHEDULER_PATCH_MARKER, True)
    Scheduler._patched_for_beam_requests = True
    logger.debug("Scheduler.schedule patched for beam-requests propagation.")

    # Publish the V1 terminal result after every physical dispatch drains.
    _patch_scheduler_update_from_output(Scheduler)


def _find_gr_output(result, request_id):
    for engine_outputs in result.values():
        for engine_output in engine_outputs.outputs:
            if engine_output.request_id == request_id:
                return engine_outputs, engine_output
    return None, None


def _native_request_terminated(scheduler, result, request_id) -> bool:
    _, target_output = _find_gr_output(result, request_id)
    if target_output is not None and target_output.finish_reason is not None:
        return True
    for engine_outputs in result.values():
        if request_id in (engine_outputs.finished_requests or ()):
            return True
    return request_id not in scheduler.requests


def _reconcile_native_gr_termination(scheduler, result, session) -> None:
    from vllm_gr.v1.engine.beam_search_v1 import GRResourceStatus, GRSessionStatus
    from vllm_gr.v1.engine.gr_async_scheduler import _make_failure_result

    if session.status not in (GRSessionStatus.WAITING, GRSessionStatus.RUNNING):
        return
    terminated_member = next(
        (
            request_id
            for request_id in session.request_ids
            if _native_request_terminated(scheduler, result, request_id)
        ),
        None,
    )
    if terminated_member is None:
        return

    session.resource_status = GRResourceStatus.DRAINING
    session.status = GRSessionStatus.FINISHING
    session.terminal_result_pending = True
    if session._pending_terminal_result is None:
        session._pending_terminal_result = _make_failure_result(
            session,
            "native_termination_without_gr_result",
            f"Native batch member {terminated_member} terminated before GR produced "
            "a terminal result",
        )


def _suppress_native_terminal(result, request_id, target_output) -> None:
    if target_output is not None:
        target_output.finish_reason = None
    for engine_outputs in result.values():
        if engine_outputs.finished_requests is not None:
            engine_outputs.finished_requests.discard(request_id)


def _suppress_batch_member_outputs(result, session) -> None:
    """Keep internal batch members off the single batch-result channel."""
    internal_ids = frozenset(session.request_ids[1:])
    if not internal_ids:
        return
    for engine_outputs in result.values():
        engine_outputs.outputs[:] = [
            output for output in engine_outputs.outputs if output.request_id not in internal_ids
        ]
        if engine_outputs.finished_requests is not None:
            engine_outputs.finished_requests.difference_update(internal_ids)


def _finish_native_gr_request(scheduler, request, request_id, request_status) -> None:
    if request is None or request.is_finished():
        return

    # Every physical dispatch has drained. Any remaining placeholders are
    # speculative reservations and no longer represent real outputs.
    request.num_output_placeholders = 0
    finalizing_ids = getattr(scheduler, _GR_FINALIZING_IDS_ATTR, None)
    if finalizing_ids is None:
        finalizing_ids = set()
        setattr(scheduler, _GR_FINALIZING_IDS_ATTR, finalizing_ids)
    finalizing_ids.add(request_id)
    try:
        scheduler.finish_requests(request_id, request_status)
    finally:
        finalizing_ids.discard(request_id)
        if not finalizing_ids:
            delattr(scheduler, _GR_FINALIZING_IDS_ATTR)


def _discard_pending_native_finish(
    scheduler, client_index, request_id, *, worker_released: bool = False
) -> None:
    finished_by_client = getattr(scheduler, "finished_req_ids_dict", None)
    if finished_by_client is not None:
        pending_finished = finished_by_client.get(client_index)
        if pending_finished is not None:
            pending_finished.discard(request_id)
            if not pending_finished:
                finished_by_client.pop(client_index, None)
    if not worker_released:
        return
    # A fused release retires the Worker's cached request in the consuming
    # frame, so no retirement control frame will carry this ID to the Worker.
    # Leaving it pending would keep has_requests() true forever: while
    # PAUSED_ALL, schedule() early-returns before it can consume the set, so
    # the engine would never look idle and the pause Future would never
    # complete.
    pending_finished_ids = getattr(scheduler, "finished_req_ids", None)
    if pending_finished_ids is not None:
        pending_finished_ids.discard(request_id)


def _finalize_ready_gr_session(scheduler, result, session) -> None:
    """Publish one terminal native output after speculative work drains."""
    from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
    from vllm.v1.request import RequestStatus

    from vllm_gr.v1.engine.beam_search_v1 import GRSessionStatus
    from vllm_gr.v1.engine.gr_async_scheduler import (
        _close_gr_session,
        _gr_session_ready_to_finalize,
        _gr_terminal_result,
    )

    request_id = session.session_id
    target_outputs, target_output = _find_gr_output(result, request_id)

    if not _gr_session_ready_to_finalize(session):
        # Native stop detection can race ahead of a queued GR successor. Keep
        # the request open at the frontend until the GR session has drained;
        # the finalizer will synthesize the one authoritative terminal frame.
        if session.status == GRSessionStatus.FINISHING and session.terminal_result_pending:
            for member_id in session.request_ids:
                _, member_output = _find_gr_output(result, member_id)
                _suppress_native_terminal(result, member_id, member_output)
        return

    terminal_result = _gr_terminal_result(session)
    native_requests = {
        member_id: scheduler.requests.get(member_id) for member_id in session.request_ids
    }
    native_request = native_requests[request_id]
    client_index = (
        native_request.client_index
        if native_request is not None
        else session.request.items[0].request.client_index
    )

    request_status = (
        RequestStatus.FINISHED_ERROR
        if terminal_result.failure is not None
        else RequestStatus.FINISHED_LENGTH_CAPPED
    )
    finish_reason = (
        FinishReason.ERROR if terminal_result.failure is not None else FinishReason.LENGTH
    )

    for member_id, member_request in native_requests.items():
        _finish_native_gr_request(scheduler, member_request, member_id, request_status)

    if target_outputs is None:
        target_outputs = result.get(client_index)
        if target_outputs is None:
            target_outputs = EngineCoreOutputs(outputs=[])
            result[client_index] = target_outputs
    if target_output is None:
        target_output = EngineCoreOutput(
            request_id=request_id,
            new_token_ids=[],
            finish_reason=finish_reason,
        )
        target_outputs.outputs.append(target_output)
    else:
        # The GR result, rather than the native placeholder token, owns the
        # externally visible terminal status.
        target_output.finish_reason = finish_reason

    kv_params = dict(target_output.kv_transfer_params or {})
    kv_params["gr_batch_result"] = terminal_result
    target_output.kv_transfer_params = kv_params
    finished_requests = set(target_outputs.finished_requests or ())
    finished_requests.add(request_id)
    target_outputs.finished_requests = finished_requests

    # finish_requests() ran after upstream assembled this step's outputs. Avoid
    # publishing the same finished ID again on the next update. A Worker that
    # already proved the release has no cached request left, so the pending
    # native finish must not survive either.
    for member_id in session.request_ids:
        _discard_pending_native_finish(
            scheduler,
            client_index,
            member_id,
            worker_released=not session.worker_resources_allocated,
        )

    _close_gr_session(scheduler, session)


def _patch_scheduler_update_from_output(scheduler_cls) -> None:
    """Consume V1 stage/retirement receipts and publish the terminal batch result."""
    marker = "_vllm_gr_update_from_output_patch"
    if getattr(scheduler_cls.update_from_output, marker, False):
        return

    _original_update = scheduler_cls.update_from_output

    @wraps(_original_update)
    def patched_update_from_output(self, scheduler_output, model_runner_output, *args, **kwargs):
        from vllm_gr.v1.engine.gr_async_scheduler import (
            GR_STAGE_OUTPUT_ATTR,
            consume_gr_retirement,
        )

        retired_session = consume_gr_retirement(self, scheduler_output, model_runner_output)
        gr_stages = getattr(scheduler_output, GR_STAGE_OUTPUT_ATTR, {})
        sessions = getattr(self, "_gr_v1_sessions", {})
        dispatched_sessions = {
            sid: sessions[sid]
            for sid, metadata in gr_stages.items()
            if sid in sessions
            and metadata.owner_generation == sessions[sid].owner_generation
            and metadata.dispatch_id in sessions[sid]._in_flight_dispatches
        }
        # GR dispatches never mix with ordinary requests. Stale results must
        # not reach native token accounting, even if a new owner reuses the ID.
        if gr_stages and not dispatched_sessions:
            return {}
        completions = [
            self.update_gr_from_output(session, scheduler_output, model_runner_output)
            for session in dispatched_sessions.values()
        ]
        native_update_ids = getattr(self, _GR_NATIVE_UPDATE_IDS_ATTR, None)
        if native_update_ids is None:
            native_update_ids = set()
            setattr(self, _GR_NATIVE_UPDATE_IDS_ATTR, native_update_ids)
        dispatched_request_ids = {
            request_id
            for session in dispatched_sessions.values()
            for request_id in session.request_ids
        }
        added_update_ids = dispatched_request_ids.difference(native_update_ids)
        native_update_ids.update(added_update_ids)
        try:
            # A malformed active receipt is a request failure, not a native
            # sampled token. The normal finalizer clears remaining placeholders
            # only after drain and matching Worker release confirmation.
            result = (
                {}
                if any(completion.error for completion in completions)
                else _original_update(self, scheduler_output, model_runner_output, *args, **kwargs)
            )
        finally:
            native_update_ids.difference_update(added_update_ids)
            if not native_update_ids:
                delattr(self, _GR_NATIVE_UPDATE_IDS_ATTR)

        for session in dispatched_sessions.values():
            _suppress_batch_member_outputs(result, session)

        updated_sessions = []
        for session in dispatched_sessions.values():
            _reconcile_native_gr_termination(self, result, session)
            updated_sessions.append(session)

        if retired_session is not None and retired_session.terminal_result_pending:
            updated_sessions.append(retired_session)
        for outputs in result.values():
            for output in outputs.outputs:
                if output.request_id in dispatched_sessions:
                    output.new_token_ids = []

        # A terminal result may arrive before a look-ahead successor. Buffer it
        # in GRSessionState and publish only after every physical dispatch has
        # drained, together with a genuinely finished native output.
        for session in updated_sessions:
            _finalize_ready_gr_session(self, result, session)

        return result

    setattr(patched_update_from_output, marker, True)
    scheduler_cls.update_from_output = patched_update_from_output


_KV_HASH_MEMO_LIMIT = 65536

# True while the whole-chain memo's fallback hashes a from-scratch request
# that MISSED the chain memo.  Such traffic is (statistically) never
# repeated: inserting its ~64 blocks per request into the per-block memo
# costs key builds, big-dict probes, and FIFO evictions that evict hot
# entries, with no hits in return — the "varied prompts after memo
# saturation" regression.  The per-block wrapper checks this flag first and
# stays transparent, so chain-miss requests hash at stock vLLM speed while
# the per-block memo keeps serving the paths where repeats are real:
# incremental (block_hashes-seeded) beam requests and the mm / lora /
# salt / prompt-embeds fallbacks.  The two memos' recency windows coincide
# (65,536 blocks ~= 1,024 x 64-block prompts = the chain-memo limit), so a
# from-scratch prompt the chain memo cannot serve is beyond the per-block
# memo's reach anyway.  EngineCore is single-threaded; even if a concurrent
# caller ever observed the flag, bypassing the memo is semantically
# transparent (it only memoizes a pure function).
_CHAIN_FALLBACK_ACTIVE = False


def _token_bytes(tokens) -> memoryview:
    """Expose token storage as contiguous bytes without copying.

    GR requests already carry 1-D contiguous int64 arrays. Native requests
    (the ADD path) still pass Python lists, which normalize through the same
    boundary contract instead of raising inside ``hashlib``.
    """
    if not (
        isinstance(tokens, np.ndarray)
        and tokens.dtype == np.int64
        and tokens.ndim == 1
        and tokens.flags.c_contiguous
    ):
        from vllm_gr.v1.beam.batch_contracts import to_int64_tokens

        tokens = to_int64_tokens(tokens, name="block tokens")
    return memoryview(tokens).cast("B")


def _hash_block_bytes(block, parent_hash, extra_keys) -> bytes:
    """Hash token bytes, then a domain tag plus parent/extra-key metadata.

    The one-byte tag keeps the common no-extra-keys path free of per-block
    MessagePack encoding while staying unambiguous with the extra-keys path.
    """
    digest = hashlib.sha256(block)
    if extra_keys is None:
        digest.update(b"\x00")
        digest.update(parent_hash)
        return digest.digest()
    digest.update(b"\x01")
    try:
        digest.update(msgspec.msgpack.encode(extra_keys))
    except TypeError:
        # Preserve support for opaque extra keys through vLLM's native object
        # hasher instead of maintaining another serializer here.
        digest.update(b"\x02")
        digest.update(_vllm_sha256(extra_keys))
    digest.update(parent_hash)
    return digest.digest()


def apply_kv_hash_memo_patch() -> None:
    """Memoize vLLM's per-block prefix-cache hashing (EngineCore process).

    ``vllm.v1.core.kv_cache_utils.hash_block_tokens`` is a pure function of
    (hash_function, parent_block_hash, block token ids, extra keys): the same
    inputs always produce the same ``BlockHash``.  Repeated blocks
    (re-sent beam suffixes, popular prefixes) re-hash identical blocks on
    every request — ~1.3ms of sha256 work per 1024-token prompt — so a
    bounded FIFO memo turns those hits into dict lookups.  Keys are immutable
    byte snapshots of the token view (``bytes(memoryview)``; never
    ``tobytes()``), so a miss hashes each block exactly once, and non-array
    token sequences normalize through the same boundary contract.  The wrapper
    is idempotent and installs on the module attribute
    (``request_block_hasher`` resolves it as a module global at call time,
    so every existing and future hasher picks it up).

    From-scratch requests that miss the whole-chain memo set
    ``_CHAIN_FALLBACK_ACTIVE`` and bypass this wrapper entirely (see there);
    the per-block memo serves incremental beam requests and the mm / lora /
    salt / prompt-embeds fallbacks, where block-level repeats are real.
    """
    global _CHAIN_FALLBACK_ACTIVE
    from vllm.v1.core import kv_cache_utils

    if getattr(kv_cache_utils.hash_block_tokens, "_gr_hash_memo", False):
        return

    _CHAIN_FALLBACK_ACTIVE = False
    original = kv_cache_utils.hash_block_tokens
    # OrderedDict (not plain dict): after the memo saturates, every miss
    # evicts the oldest entry.  ``dict.pop(next(iter(...)))`` is NOT
    # constant-time — the fresh iterator rescans the leading run of deleted
    # (dummy) slots and CPython only compacts the table on resize, so varied
    # traffic pays an ever-longer scan per eviction (measured 14-32x slowdown
    # on the saturated hashing path).  ``OrderedDict.popitem(last=False)``
    # unlinks the head of its doubly-linked list in O(1).
    memo: OrderedDict[tuple, object] = OrderedDict()

    def hash_block_tokens_memo(
        hash_function, parent_block_hash, curr_block_token_ids, extra_keys=None
    ):
        if hash_function is not _vllm_sha256:
            # The direct-buffer encoding below is SHA-256 specific; keep the
            # configured algorithm authoritative for other hash functions.
            # Those requests take the native path and therefore do NOT get memo
            # acceleration. Identity comparison is intentional: upstream hands
            # us the module-level ``sha256`` itself, so a wrapper falls back.
            # Native serializers require Python ints, not NumPy scalars.
            if isinstance(curr_block_token_ids, np.ndarray):
                curr_block_token_ids = curr_block_token_ids.tolist()
            return original(hash_function, parent_block_hash, curr_block_token_ids, extra_keys)
        parent_hash = parent_block_hash or kv_cache_utils.NONE_HASH
        # Immutable content snapshot: exact memo key, no second block hash.
        token_bytes = bytes(_token_bytes(curr_block_token_ids))
        if _CHAIN_FALLBACK_ACTIVE:
            return kv_cache_utils.BlockHash(_hash_block_bytes(token_bytes, parent_hash, extra_keys))
        key = (hash_function, parent_hash, token_bytes, extra_keys)
        hit = memo.get(key)
        if hit is not None:
            return hit
        hit = kv_cache_utils.BlockHash(_hash_block_bytes(token_bytes, parent_hash, extra_keys))
        if len(memo) >= _KV_HASH_MEMO_LIMIT:
            # Constant-time FIFO eviction keeps the memo bounded in
            # long-running servers under varied (never-repeating) traffic.
            memo.popitem(last=False)
        memo[key] = hit
        return hit

    hash_block_tokens_memo._gr_hash_memo = True  # type: ignore[attr-defined]
    # Exposed for tests / diagnostics: the live memo store behind this wrapper.
    hash_block_tokens_memo._gr_hash_memo_entries = memo  # type: ignore[attr-defined]
    kv_cache_utils.hash_block_tokens = hash_block_tokens_memo
    _apply_block_hash_chain_memo_patch(kv_cache_utils)
    logger.info("[vllm-gr] kv block-hash memo patch applied.")


_CHAIN_MEMO_LIMIT = 1024


def _apply_block_hash_chain_memo_patch(kv_cache_utils) -> None:
    """Whole-chain memo on top of the per-block memo (from-scratch requests).

    The per-block memo removes the sha256 work but a repeated prompt still
    walks the ``request_block_hasher`` loop (64 iterations of slicing,
    ``generate_block_hash_extra_keys``, tuple-key building — ~0.8ms for a
    1024-token prompt).  For a request hashing from scratch with no mm /
    lora / salt variation, the full hash chain is a pure function of
    (hash_function, block_size, salt, the full-block token prefix), so a
    hit can replay the whole list in one dict lookup.

    A MISS hashes the full-block byte view directly with SHA-256, using the
    same ``sha256(tokens || tag || parent)`` encoding as incremental hashing.
    Eligibility proves that extra keys are absent. The immutable token-byte
    snapshot doubles as the memo key, so no second whole-prompt hash is paid.
    No per-block memo bookkeeping is needed; the chain entry is still recorded
    for a repeat of this exact prompt.

    Fast-path eligibility is conservative: any incremental call
    (``block_hashes`` non-empty), mm_features, lora_request, a non-``None``
    cache_salt, or a non-``None`` prompt_embeds falls back to the original
    hasher.  prompt_embeds requests MUST bypass the chain memo: their
    ``all_token_ids`` are ``[0] * num_prompt_tokens`` placeholders (vLLM
    distinguishes their contents via per-block embeds hashes), so the token
    key alone would collide across distinct embeddings of equal length.
    """
    global _CHAIN_FALLBACK_ACTIVE
    if getattr(kv_cache_utils.get_request_block_hasher, "_gr_chain_memo", False):
        return

    original_factory = kv_cache_utils.get_request_block_hasher
    # Same constant-time FIFO rationale as the per-block memo above.
    chain_memo: OrderedDict[tuple, list] = OrderedDict()

    def get_request_block_hasher_memo(block_size, caching_hash_fn):
        original_hasher = original_factory(block_size, caching_hash_fn)

        def request_block_hasher_chain_memo(request):
            global _CHAIN_FALLBACK_ACTIVE
            if (
                request.block_hashes
                or request.mm_features
                or request.lora_request is not None
                or request.cache_salt is not None
                or getattr(request, "prompt_embeds", None) is not None
            ):
                return original_hasher(request)
            if caching_hash_fn is not _vllm_sha256:
                # Same contract as the per-block memo: non-default algorithms
                # keep native hashing and lose the chain memo.
                return original_hasher(request)

            num_tokens = request.num_tokens
            num_full_blocks = num_tokens // block_size
            if num_full_blocks == 0:
                return original_hasher(request)

            # Only the full-block prefix participates in hashing; trailing
            # partial-block tokens never reach hash_block_tokens.
            tokens = request.all_token_ids[: num_full_blocks * block_size]
            # Immutable content snapshot: an exact memo key that costs one
            # memcpy instead of a whole-prompt SHA-256.
            token_bytes = bytes(_token_bytes(tokens))
            key = (caching_hash_fn, block_size, token_bytes)
            hit = chain_memo.get(key)
            if hit is not None:
                return list(hit)

            _CHAIN_FALLBACK_ACTIVE = True
            try:
                # Eligibility above proves every block has no extra keys.
                # Use the same direct-buffer encoding as hash_block_tokens_memo.
                result = []
                parent_hash = None
                block_view = memoryview(token_bytes)
                step = block_size * 8
                for start in range(0, len(block_view), step):
                    parent_hash = kv_cache_utils.BlockHash(
                        _hash_block_bytes(
                            block_view[start : start + step],
                            parent_hash or kv_cache_utils.NONE_HASH,
                            None,
                        )
                    )
                    result.append(parent_hash)
            finally:
                _CHAIN_FALLBACK_ACTIVE = False
            if len(chain_memo) >= _CHAIN_MEMO_LIMIT:
                chain_memo.popitem(last=False)
            chain_memo[key] = list(result)
            return result

        return request_block_hasher_chain_memo

    get_request_block_hasher_memo._gr_chain_memo = True  # type: ignore[attr-defined]
    # Exposed for tests / diagnostics: the live chain-memo store.
    get_request_block_hasher_memo._gr_chain_memo_entries = chain_memo  # type: ignore[attr-defined]
    kv_cache_utils.get_request_block_hasher = get_request_block_hasher_memo
    # ``vllm.v1.engine.core`` binds this factory via ``from ... import`` at
    # module load, so it holds its OWN reference — patching only the
    # kv_cache_utils attribute never reaches ``core.get_request_block_hasher``
    # and Requests keep using the original per-block loop.  Install the same
    # wrapper object on both module attributes so a single shared chain memo
    # serves every path.  Import the submodule explicitly: relying on the
    # ``vllm.v1.engine.core`` attribute of the parent package would silently
    # no-op whenever nothing else has bound it yet.
    try:
        import vllm.v1.engine.core as _vllm_engine_core

        _vllm_engine_core.get_request_block_hasher = get_request_block_hasher_memo
    except ImportError:
        # Fallback: the core module may not be importable in exotic
        # processes; the kv_cache_utils binding alone still helps any
        # caller that resolves it as a module global.
        pass
