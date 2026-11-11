# Public-model serving measurements

One independent H20 start on code revision `8807861f9f1c`.
Each timing cell is arithmetic mean / P99 in milliseconds. All samples,
including slow requests, are retained. P99 uses linear interpolation; 300
samples per cell provide a finite-sample diagnostic, not a production SLO.

## Timing definitions

| Metric | Definition |
| --- | --- |
| Pure Forward | Sum of six CUDA Events intervals: three model executions plus three Trie/TopK/Beam selections. Includes input preparation, LM head and idle gaps inside each interval; excludes gaps between them. |
| E2E | Client RPC call start through complete response return, including protobuf encoding/decoding and transport. Excludes request construction and business/array conversion after return. |
| Outside | E2E minus Pure Forward for the same request. Includes interval-external request handling, IPC, result handling, communication and gaps between the six intervals. |

Outside P99 is calculated from the paired per-request differences. It is not
`P99(E2E) - P99(Forward)`. Outside time must not be called pure frontend time.

## Workload and configuration

- Standard `Qwen3MoeForCausalLM`, dummy weights: 12 layers, hidden size 1024,
  32 experts, four selected per token, BF16, no quantization.
- Deterministic synthetic token IDs and integer-only three-token Trie.
  65/4097 are token counts per input; they do not encode item-feature history.
- Explicit gRPC GenerateBatch, B1/B4; one RPC batch in flight.
- Fixed 1000 returned sequences per input, each with three generated tokens.
- W=[256,512,1024], per-parent K=[256,256,256], length_penalty=0.
- Native max_num_seqs=4; GR manages Decode Beam fanout. max_model_len=4100,
  max_num_batched_tokens=16388. Prefix caching is disabled; no result cache.
- Eager Prefill and GR Decode CUDA Graph replay; normal Python GC,
  OMP_NUM_THREADS=1. All three generic optimizations remain enabled.
- vLLM 0.30.0, Torch 2.13.0+cu130, grpcio 1.85.1, protobuf 7.36.2.
- Each cell: 300 paired calls, 100 separate probe-disabled controls, five
  excluded warmups. 2400 paired calls total; zero measured captures and
  600 Decode graph replays per cell. Repeated/default/binary token IDs and
  FP32 scores matched.

## Default gRPC output

| Input tokens / Batch | Outside | Pure Forward | E2E |
| --- | --- | --- | --- |
| 65 / B1 | 2.25 / 2.76 | 9.76 / 10.62 | 12.01 / 13.37 |
| 65 / B4 | 4.29 / 5.07 | 18.88 / 19.21 | 23.17 / 23.99 |
| 4097 / B1 | 2.53 / 3.02 | 22.14 / 22.59 | 24.67 / 25.28 |
| 4097 / B4 | 5.40 / 6.39 | 71.14 / 71.60 | 76.54 / 77.72 |

## Optional binary gRPC output

| Input tokens / Batch | Outside | Pure Forward | E2E |
| --- | --- | --- | --- |
| 65 / B1 | 1.93 / 2.53 | 9.74 / 10.10 | 11.67 / 12.52 |
| 65 / B4 | 2.93 / 3.56 | 18.95 / 20.80 | 21.88 / 24.21 |
| 4097 / B1 | 2.32 / 2.82 | 22.16 / 22.47 | 24.48 / 25.22 |
| 4097 / B4 | 3.72 / 4.40 | 71.06 / 71.44 | 74.77 / 75.66 |

These standard-model measurements demonstrate residual outside time; they
are not an equal-model or equal-catalog comparison against another framework.
The metric boundaries above are independent of the generated model and data.

## Reproduce

```bash
OMP_NUM_THREADS=1 python benchmarks/beam_frontend_overhead/run.py \
  --output ./benchmark-output/validation \
  --architecture moe --lengths 65,4097 --batches 1,4 \
  --result-widths 1000 --samples 300 --control-samples 100 --warmup 5
```

`summary.json` retains all raw samples; `metrics.csv` contains E2E, Forward
and Outside means/P99 values. Use fresh directories for independent starts.
Dense models and compiled-Prefill mode were not exercised by this run.

## Auxiliary diagnostics

This appendix is not another set of business performance metrics. Post-RPC
array conversion/validation is outside E2E. Array-ready time includes it.
Probe-disabled RPCs check timing-observer effects; they have no paired Forward
measurement. Conversion/array-ready values use 300 calls; probe-disabled values
use their own 100 calls. Each cell below is mean / P99 in milliseconds.

| Input tokens / Batch | Output format | Post-RPC conversion | Array-ready time | RPC without timing probe |
| --- | --- | --- | --- | --- |
| 65 / B1 | default | 0.03 / 0.08 | 12.04 / 13.42 | 11.95 / 12.54 |
| 65 / B1 | binary | 0.02 / 0.05 | 11.69 / 12.54 | 11.63 / 12.75 |
| 65 / B4 | default | 0.08 / 0.14 | 23.24 / 24.09 | 23.12 / 24.05 |
| 65 / B4 | binary | 0.04 / 0.09 | 21.92 / 24.26 | 21.79 / 24.27 |
| 4097 / B1 | default | 0.03 / 0.07 | 24.70 / 25.32 | 24.70 / 25.44 |
| 4097 / B1 | binary | 0.03 / 0.06 | 24.51 / 25.25 | 24.41 / 25.03 |
| 4097 / B4 | default | 0.10 / 0.14 | 76.64 / 77.88 | 76.41 / 76.95 |
| 4097 / B4 | binary | 0.06 / 0.08 | 74.83 / 75.72 | 74.80 / 75.63 |

Previous tables labelled RPC through arrays included post-RPC conversion and
used native max_num_seqs=4096. They remain in the original experiment outputs;
the corrected boundary/configuration above was measured again, not inferred
by subtracting aggregate statistics.

Additional validation: 106 host gRPC/columnar tests, 116 CUDA constrained-TopK
and batch-switch graph tests, protobuf consistency, and targeted static checks
passed. Full repository CI and NPU model validation were not run.
