# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark-only Worker extension: deferred, paired CUDA interval timing."""

from collections import Counter, deque
from functools import wraps

import torch

_POOL = deque()
_PENDING = []
_ACTIVE = None
_ENABLED = False
_INSTALLED = False
_GRAPHS = Counter()


def _install():
    global _INSTALLED
    if _INSTALLED:
        return
    from vllm_gr.v1.worker.beam_decode_graph import BeamDecodeGraph
    from vllm_gr.v1.worker.gpu_beam_stage_runner import GPUBeamStageRunner

    capture, replay = BeamDecodeGraph.capture, BeamDecodeGraph.replay

    @wraps(capture)
    def counted_capture(self, *args, **kwargs):
        result = capture(self, *args, **kwargs)
        _GRAPHS["captures"] += 1
        return result

    @wraps(replay)
    def counted_replay(self, *args, **kwargs):
        result = replay(self, *args, **kwargs)
        _GRAPHS["replays"] += 1
        return result

    BeamDecodeGraph.capture = counted_capture
    BeamDecodeGraph.replay = counted_replay

    execute, sample = GPUBeamStageRunner.execute, GPUBeamStageRunner.sample

    def timed(kind, stage, operation):
        if _ACTIVE is None:
            return operation()
        if len(_POOL) < 2:
            raise RuntimeError("Probe event pool exhausted; drain between measurement batches")
        start, end = _POOL.popleft(), _POOL.popleft()
        start.record()
        try:
            return operation()
        finally:
            end.record()
            _ACTIVE["events"].append((stage, kind, start, end))

    @wraps(execute)
    def measured_execute(self, scheduled, *args, **kwargs):
        global _ACTIVE
        stages = getattr(scheduled, "gr_stage_metadata", None) or {}
        metadata = next(iter(stages.values()), None)
        if metadata is None:
            return execute(self, scheduled, *args, **kwargs)
        if metadata.stage_index == 0 and _ENABLED:
            _ACTIVE = dict(
                session_id=metadata.session_id,
                events=[],
                per_parent_beam_widths=list(metadata.beam_params.per_parent_beam_widths or ()),
            )
        self._frontend_probe_stage = metadata.stage_index
        return timed(
            "execute", metadata.stage_index, lambda: execute(self, scheduled, *args, **kwargs)
        )

    @wraps(sample)
    def measured_sample(self, *args, **kwargs):
        global _ACTIVE
        stage = getattr(self, "_frontend_probe_stage", -1)
        result = timed("sample", stage, lambda: sample(self, *args, **kwargs))
        if _ACTIVE is not None and stage == 2:
            _PENDING.append(_ACTIVE)
            _ACTIVE = None
        return result

    GPUBeamStageRunner.execute = measured_execute
    GPUBeamStageRunner.sample = measured_sample
    _INSTALLED = True


class WorkerExtension:
    def frontend_probe_status(self):
        return dict(captures=_GRAPHS["captures"], replays=_GRAPHS["replays"])

    def configure_frontend_probe(self, enabled: bool):
        global _ENABLED
        _install()
        if not _POOL:
            # Allocate and materialize Events outside the measured requests.
            for _ in range(8192):
                event = torch.Event(device="cuda", enable_timing=True)
                event.record()
                _POOL.append(event)
            torch.accelerator.synchronize()
        if _PENDING or _ACTIVE is not None:
            raise RuntimeError("Drain the probe before changing its configuration")
        _ENABLED = enabled
        runner = self.model_runner
        return dict(
            enabled=enabled,
            device=torch.cuda.get_device_name(),
            torch=str(torch.__version__),
            model_class=type(runner.get_model()).__name__,
            compilation_mode=int(runner.compilation_config.mode),
            cudagraph_mode=str(runner.compilation_config.cudagraph_mode),
        )

    def drain_frontend_probe(self):
        # One excluded control RPC, after all timed GenerateBatch calls.
        torch.accelerator.synchronize()
        rows = []
        for state in _PENDING:
            events = state["events"]
            if len(events) != 6 or len({(s, k) for s, k, _, _ in events}) != 6:
                raise RuntimeError("Expected execute/sample pairs for three generation stages")
            segments = []
            for stage, kind, start, end in events:
                segments.append(dict(stage=stage, kind=kind, ms=start.elapsed_time(end)))
                _POOL.extend((start, end))
            rows.append(
                dict(
                    session_id=state["session_id"],
                    segments=segments,
                    per_parent_beam_widths=state["per_parent_beam_widths"],
                )
            )
        _PENDING.clear()
        return rows
