# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402 -- Resolve this checkout before importing the runtime.
"""Real standard-model gRPC calls, paired with Worker CUDA stage intervals."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

import grpc
import numpy as np
from assets import VOCAB, WIDTHS, prepare

from vllm_gr.entrypoints.grpc.generated import vllm_engine_pb2 as pb
from vllm_gr.entrypoints.grpc.generated import vllm_engine_pb2_grpc as rpc


def integers(value):
    return [int(x) for x in value.split(",")]


def consume(response, batch, count):
    """Count the same required client-side array conversion in both formats."""
    if len(response.items) != batch:
        raise RuntimeError("Incomplete batch response")
    rows = []
    for item in response.items:
        if item.HasField("binary_beams"):
            beams = item.binary_beams
            if beams.count != count or beams.steps != 3:
                raise RuntimeError("Unexpected binary result shape")
            tokens = np.frombuffer(beams.output_ids_i32_le, dtype="<i4").reshape(count, 3)
            scores = np.frombuffer(beams.cumulative_logprobs_f32_le, dtype="<f4")
        else:
            beams = item.beams
            if len(beams.output_offsets) != count + 1:
                raise RuntimeError("Unexpected ordinary result shape")
            tokens = np.asarray(beams.output_ids, dtype="<i4").reshape(count, 3)
            scores = np.asarray(beams.cumulative_logprobs, dtype="<f4")
        if scores.shape != (count,) or not np.isfinite(scores).all():
            raise RuntimeError("Invalid result scores")
        rows.append((tokens, scores))
    return rows


def signature(rows):
    return hashlib.sha256(b"".join(t.tobytes() + s.tobytes() for t, s in rows)).hexdigest()


def request(length, batch, count, binary):
    rng = np.random.default_rng(314159 + length)
    messages = []
    for index in range(batch):
        tokens = rng.integers(0, VOCAB, length, dtype="<i8")
        messages.append(
            pb.GenerateRequest(
                request_id=f"synthetic-{index}",
                tokenized=pb.TokenizedInput(input_ids_i64_le=tokens.tobytes()),
                sampling_params=pb.SamplingParams(
                    temperature=0.0,
                    ignore_eos=True,
                    n=1024,
                    max_tokens=3,
                ),
                beam_search=pb.BeamSearchOptions(
                    beam_width=1024,
                    dynamic_beam_widths=WIDTHS,
                    result_width=count,
                    length_penalty=0.0,
                ),
            )
        )
    return pb.BatchGenerateRequest(requests=messages, binary_beam_output=binary)


def wait_ready(client, process, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Server exited; inspect server.log")
        try:
            if client.HealthCheck(pb.HealthCheckRequest(), timeout=1).healthy:
                return
        except grpc.RpcError:
            pass
        time.sleep(0.5)
    raise TimeoutError("Server startup timed out; inspect server.log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--architecture", choices=("dense", "moe"), default="moe")
    parser.add_argument("--tiny", action="store_true", help="Two standard layers for a quick smoke")
    parser.add_argument("--lengths", type=integers, default=[65, 4097])
    parser.add_argument("--batches", type=integers, default=[1, 4])
    parser.add_argument("--result-widths", type=integers, default=[64, 1000])
    parser.add_argument("--formats", default="default,binary")
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--control-samples", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--prefill-mode", choices=("eager", "compiled"), default="eager")
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    args = parser.parse_args()
    formats = args.formats.split(",")
    if not set(formats) <= {"default", "binary"} or not formats:
        parser.error("--formats must contain default and/or binary")
    if not 1 <= args.samples <= 512 or not 1 <= args.warmup <= 32:
        parser.error("Use 1..512 samples and 1..32 warmups between probe drains")
    if args.control_samples < 1 or not set(args.batches) <= {1, 2, 4}:
        parser.error("Use positive control samples and batch sizes 1,2,4")
    if any(n < 1 or n > 1024 for n in args.result_widths):
        parser.error("Result widths must be in 1..1024")
    if any(n < 1 or n + 3 > 8192 for n in args.lengths):
        parser.error("Prompt lengths must be in 1..8189")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        parser.error("Choose a fresh output directory; existing results will not be overwritten")
    assets = prepare(output, args.architecture, args.tiny)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    compilation = dict(mode=0, cudagraph_mode="NONE")
    if args.prefill_mode == "compiled":
        compilation = dict(mode=3, cudagraph_mode="FULL_AND_PIECEWISE")
    command = [
        sys.executable,
        str(HERE / "server.py"),
        "serve",
        assets["model"],
        "--grpc",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--vllm-gr-config",
        str(output / "gr.json"),
        "--load-format",
        "dummy",
        "--skip-tokenizer-init",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(max(args.lengths) + 3),
        "--max-num-batched-tokens",
        str(max(4096, 4 * max(args.lengths))),
        "--max-num-seqs",
        str(max(args.batches)),
        "--max-logprobs",
        "256",
        "--no-enable-prefix-caching",
        "--no-enable-chunked-prefill",
        "--async-scheduling",
        "--disable-log-stats",
        "--seed",
        "0",
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--worker-extension-cls",
        "frontend_probe.WorkerExtension",
        "--compilation-config",
        json.dumps(compilation),
    ]
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join([str(HERE), str(REPO)]),
        TOKENIZERS_PARALLELISM="false",
    )
    report = dict(
        status="starting",
        model=assets,
        cells=[],
        samples=[],
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        prefill_mode=args.prefill_mode,
        decode_graph_enabled=True,
        load_format="dummy",
        dtype="bfloat16",
        prefix_cache=False,
        measurement="Full RPC return E2E; six CUDA intervals Forward; Outside = E2E - Forward",
        max_num_seqs=max(args.batches),
        length_penalty=0.0,
        command=command,
        versions={
            name: importlib.metadata.version(name)
            for name in ("vllm", "torch", "grpcio", "protobuf", "numpy")
        },
        omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
    )
    process = channel = None
    try:
        with (output / "server.log").open("w") as log:
            process = subprocess.Popen(
                command, cwd=REPO, env=env, stdout=log, stderr=log, start_new_session=True
            )
        channel = grpc.insecure_channel(
            f"127.0.0.1:{port}",
            options=[
                ("grpc.max_receive_message_length", 64 * 1024 * 1024),
            ],
        )
        client = rpc.VllmEngineStub(channel)
        control = channel.unary_unary(
            "/beam_frontend_benchmark.Control/Call",
            request_serializer=lambda value: json.dumps(value).encode(),
            response_deserializer=json.loads,
        )
        wait_ready(client, process, args.startup_timeout)
        report["worker"] = control(dict(command="configure", enabled=True), timeout=60)
        expected = {}
        for length in args.lengths:
            for batch in args.batches:
                for count in args.result_widths:
                    for wire in formats:
                        tag = f"L{length}-B{batch}-N{count}-{wire}"
                        message = request(length, batch, count, wire == "binary")

                        def invoke(timeout=120):
                            begin = time.perf_counter_ns()
                            response = client.GenerateBatch(message, timeout=timeout)
                            rpc_end = time.perf_counter_ns()
                            rows = consume(response, batch, count)
                            arrays_end = time.perf_counter_ns()
                            return rows, (rpc_end - begin) / 1e6, (arrays_end - rpc_end) / 1e6

                        control(dict(command="configure", enabled=True), timeout=60)
                        for _ in range(args.warmup):
                            rows, _, _ = invoke(timeout=args.startup_timeout)
                        control(dict(command="drain"), timeout=60)
                        before_graphs = control(dict(command="status"), timeout=60)[0]
                        key = (length, batch, count)
                        observed = signature(rows)
                        if key in expected and observed != expected[key]:
                            raise RuntimeError("Default/binary outputs differ")
                        expected[key] = observed
                        samples = []
                        for index in range(args.samples):
                            rows, elapsed, conversion = invoke()
                            if signature(rows) != expected[key]:
                                raise RuntimeError("Repeated requests changed output IDs/scores")
                            samples.append(
                                dict(
                                    cell=tag,
                                    index=index,
                                    e2e_ms=elapsed,
                                    client_conversion_ms=conversion,
                                    arrays_ready_ms=elapsed + conversion,
                                )
                            )
                        workers = control(dict(command="drain"), timeout=60)
                        after_graphs = control(dict(command="status"), timeout=60)[0]
                        graph_delta = {
                            k: after_graphs[k] - before_graphs[k] for k in ("captures", "replays")
                        }
                        if graph_delta["captures"] or graph_delta["replays"] < 2 * args.samples:
                            raise RuntimeError(
                                f"Expected warmed Decode graph replay: {graph_delta}"
                            )
                        if len(workers) != 1 or len(workers[0]) != len(samples):
                            raise RuntimeError(
                                "Expected one Worker and one six-interval row per RPC"
                            )
                        for sample, events in zip(samples, workers[0], strict=True):
                            segments = events["segments"]
                            gpu = sum(s["ms"] for s in segments)
                            sample.update(
                                forward_ms=gpu,
                                outside_ms=sample["e2e_ms"] - gpu,
                                segments=segments,
                            )
                            if sample["outside_ms"] < -0.1:
                                raise RuntimeError(
                                    "Invalid timing boundaries; intervals exceed RPC"
                                )
                        control(dict(command="configure", enabled=False), timeout=60)
                        off = [invoke()[1] for _ in range(args.control_samples)]
                        means = {
                            k: statistics.mean(s[k] for s in samples)
                            for k in ("e2e_ms", "forward_ms", "outside_ms")
                        }
                        p99s = {
                            k: float(np.percentile([s[k] for s in samples], 99, method="linear"))
                            for k in ("e2e_ms", "forward_ms", "outside_ms")
                        }
                        diagnostics = {
                            k: dict(
                                mean=statistics.mean(s[k] for s in samples),
                                p99=float(
                                    np.percentile([s[k] for s in samples], 99, method="linear")
                                ),
                            )
                            for k in ("client_conversion_ms", "arrays_ready_ms")
                        }
                        diagnostics["probe_off_e2e_ms"] = dict(
                            mean=statistics.mean(off),
                            p99=float(np.percentile(off, 99, method="linear")),
                        )
                        report["cells"].append(
                            dict(
                                tag=tag,
                                prompt_tokens=length,
                                batch=batch,
                                result_width=count,
                                wire=wire,
                                means=means,
                                p99s=p99s,
                                diagnostics=diagnostics,
                                output_sha256=observed,
                                probe_off_samples_ms=off,
                                graph_delta=graph_delta,
                            )
                        )
                        report["samples"].extend(samples)
                        print(tag, json.dumps(dict(mean=means, p99=p99s)), flush=True)
                        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        report["status"] = "passed"
        with (output / "metrics.csv").open("w") as stream:
            rows = [
                dict(
                    cell=c["tag"],
                    input_tokens=c["prompt_tokens"],
                    batch=c["batch"],
                    result_width=c["result_width"],
                    wire=c["wire"],
                    **{
                        f"{key.removesuffix('_ms')}_{stat}_ms": value
                        for stat, field in (("mean", "means"), ("p99", "p99s"))
                        for key, value in c[field].items()
                    },
                )
                for c in report["cells"]
            ]
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with (output / "diagnostics.csv").open("w") as stream:
            rows = [
                dict(
                    cell=c["tag"],
                    **{
                        f"{key.removesuffix('_ms')}_{stat}_ms": value
                        for key, stats in c["diagnostics"].items()
                        for stat, value in stats.items()
                    },
                )
                for c in report["cells"]
            ]
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        if channel is not None:
            channel.close()
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
