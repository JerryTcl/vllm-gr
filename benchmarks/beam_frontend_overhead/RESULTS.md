# Local validation example

This is one independent H20 start on code revision `a85136ae461f`.
All measurements report arithmetic mean and P99 in milliseconds. Every sample,
including slow calls, is retained. P99 uses linear percentile interpolation.
This finite-sample diagnostic is not a production latency target or SLO.

## Configuration

- Standard `Qwen3MoeForCausalLM`, locally initialized dummy weights.
- 12 layers, hidden size 1024, 32 experts, four selected per token, BF16.
- Synthetic integer vocabulary/catalog; one explicit B4 request at a time.
- Eager Prefill and warmed GR Decode CUDA Graphs, no quantization or prefix cache.
- W=[256,512,1024], per-parent K=[256,256,256], final result count 64 or 1000.
- Default Python GC; `OMP_NUM_THREADS=1`; no serving-heap changes.
- vLLM 0.30.0, Torch 2.13.0+cu130, grpcio 1.85.1, protobuf 7.36.2.
- 300 paired calls and 100 separate probe-off calls per cell: 2400 paired calls total.
- Every cell recorded zero measured captures and 600 Decode graph replays.
- Repeated requests and default/binary replies matched token IDs and FP32 scores.

## Mean / P99 (milliseconds)

Each timing cell below is `mean / P99`. Probe-off statistics use their own
100 samples; the other columns use 300 paired samples per configuration.

| Prompt tokens | Batch | Results/input | Wire | RPC through arrays | Worker stage intervals | Outside stages | Probe-off RPC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 65 | 4 | 64 | default | 21.83 / 23.42 | 18.90 / 19.96 | 2.93 / 3.93 | 21.75 / 22.19 |
| 65 | 4 | 64 | binary | 21.53 / 22.65 | 18.87 / 19.25 | 2.65 / 3.02 | 21.44 / 21.78 |
| 65 | 4 | 1000 | default | 23.22 / 24.86 | 18.93 / 19.73 | 4.29 / 5.26 | 23.33 / 26.34 |
| 65 | 4 | 1000 | binary | 21.90 / 23.43 | 18.93 / 20.04 | 2.97 / 4.04 | 21.71 / 22.89 |
| 4097 | 4 | 64 | default | 74.76 / 75.92 | 71.06 / 71.49 | 3.70 / 4.52 | 74.74 / 76.33 |
| 4097 | 4 | 64 | binary | 74.72 / 76.11 | 71.08 / 71.47 | 3.64 / 4.80 | 74.50 / 75.07 |
| 4097 | 4 | 1000 | default | 76.34 / 77.61 | 71.12 / 71.99 | 5.22 / 6.24 | 76.42 / 77.39 |
| 4097 | 4 | 1000 | binary | 75.07 / 76.37 | 71.17 / 71.44 | 3.91 / 4.71 | 74.63 / 74.94 |

Outside-stage mean and P99 are calculated from each request's paired
`RPC - sum(Worker intervals)` difference. P99 values are not additive:
`P99(RPC) - P99(Worker)` does not equal the outside-stage P99.

The residual remains visible with all three generic optimizations present,
using only the standard model path and generated assets. CUDA intervals include
model/logits and selection, and may include launch gaps inside a stage. The
residual includes frontend, IPC, EngineCore/Worker host work between stages,
output transfer, transport and client conversion. It is not pure frontend time
and is not a cross-framework comparison.

## Reproduce this configuration

```bash
OMP_NUM_THREADS=1 python benchmarks/beam_frontend_overhead/run.py \
  --output ./benchmark-output/validation \
  --architecture moe --lengths 65,4097 --batches 4 \
  --result-widths 64,1000 --samples 300 --control-samples 100 --warmup 5
```

`summary.json` records raw paired samples, environment, exact command, output
hashes, means and P99 values. `metrics.csv` has explicit `_mean_ms` and `_p99_ms`
columns. Repeat in fresh output directories for independent-start variance.
Dense-model and compiled-Prefill options were not exercised by this run.

Additional checks: 106 host gRPC/columnar tests, 116 CUDA constrained-TopK and
batch-switch graph tests, protobuf regeneration consistency, and targeted
formatting/documentation checks passed. The full repository test suite and
NPU model validation were not run.
