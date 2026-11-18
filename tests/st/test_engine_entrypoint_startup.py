# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc"
"""Exercise the process bootstrap and session transport through public entry points."""

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
ROWS = [[7, 8, 9], [10, 11, 12]]


def _offline(model: str, config: str) -> None:
    from vllm_gr.entrypoints.gr import GRLLM
    from vllm_gr.outputs import CompactBeamSearchOutput
    from vllm_gr.plugin import initialize_runtime
    from vllm_gr.sampling_params import BeamSearchParams

    # GRLLM calls initialize_runtime again before building EngineArgs.
    initialize_runtime()
    with GRLLM(
        model=model,
        vllm_gr_config=config,
        dtype="float16",
        enforce_eager=True,
        async_scheduling=True,
        max_model_len=64,
        max_num_batched_tokens=128,
        max_num_seqs=16,
        gpu_memory_utilization=0.1,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
    ) as llm:
        prompts = [{"prompt_token_ids": row} for row in ROWS]

        def submit(steps: int) -> list[CompactBeamSearchOutput]:
            params = BeamSearchParams(
                beam_width=4, max_tokens=steps, temperature=0.0, ignore_eos=True
            )
            outputs: list[CompactBeamSearchOutput] = llm.beam_search_v1(prompts, params)
            return outputs

        with pytest.raises(ValueError, match="step|token"):
            submit(4)
        result = submit(3)
        assert [item.prompt_token_ids for item in result] == list(map(tuple, ROWS))
        assert all(len(item.sequences) == 4 for item in result)
        assert all(len(beam.token_ids) == 3 for item in result for beam in item.sequences)
        assert submit(3) == result


def _http(address: str, model: Path) -> None:
    import requests

    body = dict(
        model=str(model),
        prompt=ROWS,
        use_beam_search=True,
        n=4,
        temperature=0.0,
        ignore_eos=True,
        return_token_ids=True,
    )

    def submit(steps: int) -> requests.Response:
        return requests.post(
            f"http://{address}/v1/completions", json={**body, "max_tokens": steps}, timeout=90
        )

    invalid = submit(4)
    assert invalid.status_code == 400, invalid.text

    def choices() -> list[dict[str, Any]]:
        response = submit(3)
        assert response.status_code == 200, response.text
        items: list[dict[str, Any]] = response.json()["choices"]
        return items

    result = choices()
    assert [item["index"] for item in result] == list(range(8))
    assert [item["prompt_token_ids"] for item in result[::4]] == ROWS
    assert all(len(item["token_ids"]) == 3 for item in result)
    assert choices() == result


def _grpc(address: str) -> None:
    import grpc

    from examples.online_serving.grpc.client import generate
    from vllm_gr.entrypoints.grpc.generated import vllm_engine_pb2_grpc as rpc

    with grpc.insecure_channel(address) as channel:
        grpc.channel_ready_future(channel).result(timeout=30)
        stub = rpc.VllmEngineStub(channel)
        with pytest.raises(grpc.RpcError) as exc:
            generate(stub, ROWS, max_new_tokens=4)
        assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        result = generate(stub, ROWS)["items"]
        assert len(result) == len(ROWS)
        assert all(len(item["token_ids"]) == 4 for item in result)
        assert all(len(beam) == 3 for item in result for beam in item["token_ids"])
        assert generate(stub, ROWS)["items"] == result


@pytest.mark.slow
@pytest.mark.model_test
@pytest.mark.timeout(420)
@pytest.mark.parametrize("entrypoint", ["offline", "http", "grpc"])
def test_engine_entrypoint_startup(tmp_path: Path, entrypoint: str) -> None:
    import torch

    from examples.online_serving.grpc.prepare_model import prepare_model

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if entrypoint == "grpc":
        pytest.importorskip("smg_grpc_servicer", reason="install vllm-gr[grpc]")

    model, config = prepare_model(tmp_path)
    env = dict(
        os.environ,
        VLLM_USE_V2_MODEL_RUNNER="0",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        HF_HUB_OFFLINE="1",
        PYTHONPATH=os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")]),
    )
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    address = f"127.0.0.1:{port}"
    if entrypoint == "offline":
        cmd = [sys.executable, __file__, str(model), str(config)]
    else:
        cmd = [
            sys.executable,
            "-m",
            "vllm_gr.entrypoints.cli.main",
            "serve",
            str(model),
            "--gr",
            "--vllm-gr-config",
            str(config),
            "--dtype",
            "float16",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--async-scheduling",
            "--no-enable-chunked-prefill",
            "--no-enable-prefix-caching",
            "--max-model-len",
            "64",
            "--max-num-batched-tokens",
            "128",
            "--max-num-seqs",
            "16",
            "--gpu-memory-utilization",
            "0.1",
            "--enforce-eager",
        ]
        if entrypoint == "grpc":
            cmd.append("--grpc")

    log = tmp_path / "engine.log"
    with log.open("w") as stream:
        child = subprocess.Popen(
            cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            if entrypoint == "offline":
                assert child.wait(timeout=300) == 0, log.read_text()[-8000:]
                return
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline:
                assert child.poll() is None, log.read_text()[-8000:]
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=1):
                        break
                except OSError:
                    time.sleep(0.2)
            else:
                pytest.fail(f"Server startup timed out: {log.read_text()[-8000:]}")
            if entrypoint == "http":
                _http(address, model)
            else:
                _grpc(address)
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()


if __name__ == "__main__":
    _offline(*sys.argv[1:])
