# Local validation example

This is one independent H20 start on code revision `0d06f57d7131`.
It is a reproducibility example, not a latency target or an SLO. All measured
samples, including slow calls, were retained. The preceding development runs
are not pooled into this table.

## Configuration

- Standard `Qwen3MoeForCausalLM`, locally initialized dummy weights.
- 12 layers, hidden size 1024, 32 experts, four selected per token, BF16.
- Synthetic integer vocabulary/catalog; one explicit B4 request at a time.
- Eager Prefill and warmed GR Decode CUDA Graphs, no quantization or prefix cache.
- W=[256,512,1024], per-parent K=[256,256,256], final result count 64 or 1000.
- Default Python GC; `OMP_NUM_THREADS=1`; no serving-heap changes.
- vLLM 0.30.0, Torch 2.13.0+cu130, grpcio 1.85.1, protobuf 7.36.2.
- 30 paired calls per cell and 10 separate probe-off calls; 240 paired calls total.
- Every cell recorded zero measured captures and 60 Decode graph replays.
- Repeated requests and default/binary replies matched token IDs and FP32 scores.

## Means (milliseconds)

| Prompt tokens | Batch | Results/input | Wire | RPC through arrays | Worker stage intervals | Outside stages | Probe-off RPC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 65 | 4 | 64 | default | 21.50 | 18.83 | 2.67 | 21.42 |
| 65 | 4 | 64 | binary | 21.32 | 18.80 | 2.52 | 21.26 |
| 65 | 4 | 1000 | default | 22.94 | 18.79 | 4.15 | 22.97 |
| 65 | 4 | 1000 | binary | 21.48 | 18.78 | 2.70 | 21.43 |
| 4097 | 4 | 64 | default | 74.60 | 71.02 | 3.58 | 74.72 |
| 4097 | 4 | 64 | binary | 74.52 | 71.09 | 3.43 | 74.52 |
| 4097 | 4 | 1000 | default | 76.53 | 71.17 | 5.36 | 76.27 |
| 4097 | 4 | 1000 | binary | 75.05 | 71.11 | 3.94 | 75.32 |

The long-input, 1000-result cell retains about 5.36 ms outside the six Worker
intervals with default replies, and about 3.94 ms with binary replies. This
shows residual host/transport/client cost with all three generic optimizations
present, using only the standard model path and generated assets.

CUDA intervals include model/logits and selection, and may include launch gaps
inside a stage. The residual includes frontend, IPC, EngineCore/Worker host
work between stages, output transfer, transport and client conversion. It is
not pure frontend time and is not a cross-framework comparison.

## Reproduce this configuration

```bash
OMP_NUM_THREADS=1 python benchmarks/beam_frontend_overhead/run.py \
  --output ./benchmark-output/validation \
  --architecture moe --lengths 65,4097 --batches 4 \
  --result-widths 64,1000 --samples 30 --control-samples 10 --warmup 5
```

`summary.json` records the raw paired samples, environment, exact command and
output hashes in the selected output directory. Repeat in fresh directories
for independent-start variance. Dense-model and compiled-Prefill options were
not exercised by this validation run.

Additional checks: 106 host gRPC/columnar tests, 116 CUDA constrained-TopK and
batch-switch graph tests, protobuf regeneration consistency, and targeted
formatting/documentation checks passed. The full repository test suite and
NPU model validation were not run.
