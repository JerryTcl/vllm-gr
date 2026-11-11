# Public-model serving measurements

One independent H20 start on code revision `fb734fda5cf5`.
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
- W=[256,512,1024], per-parent K=[256,128,64], length_penalty=0.
- Native max_num_seqs=4; GR manages Decode Beam fanout. max_model_len=4100,
  max_num_batched_tokens=16388. Prefix caching is disabled; no result cache.
- Eager Prefill and GR Decode CUDA Graph replay; normal Python GC,
  OMP_NUM_THREADS=1. All three generic optimizations remain enabled.
- vLLM 0.30.0, Torch 2.13.0+cu130, grpcio 1.85.1, protobuf 7.36.2.
- Each cell: 300 paired calls, 100 separate probe-disabled controls, five
  excluded warmups. 2400 paired calls total; zero measured captures and
  600 Decode graph replays per cell. Repeated/default/binary token IDs and
  FP32 scores matched.

## Matched search and resource configuration

The request schedule is checked again on the Worker for every measured RPC.
The candidate workspace capacity stays at 256 while stage K changes.

| Setting | Value |
| --- | --- |
| `beam.execution_mode` | `v1` |
| `beam.worker_decision` | `true` |
| `beam.graph_enabled` | `true` |
| `beam.max_width` | `1024` |
| `beam.max_decode_steps` | `3` |
| `beam.online_batch_sizes` | `[1,2,4]` |
| `max_num_seqs` | `4` |
| `max_num_batched_tokens` | `16388` |
| `max_model_len` | `4100` |
| `constraint_table.max_top_k` | `256` |
| `constraint_table.candidate_width` | `256` |
| `beam_width` | `1024` |
| `dynamic_beam_widths` | `[256,512,1024]` |
| `per_parent_beam_widths` | `[256,128,64]` |
| `result_width` | `1000` |
| `max_tokens` | `3` |
| `ignore_eos` | `true` |
| `length_penalty` | `0` |

## Default gRPC output

| Input tokens / Batch | Outside | Pure Forward | E2E |
| --- | --- | --- | --- |
| 65 / B1 | 2.43 / 3.29 | 9.69 / 12.20 | 12.13 / 15.48 |
| 65 / B4 | 4.38 / 5.39 | 18.14 / 20.99 | 22.53 / 26.39 |
| 4097 / B1 | 3.00 / 3.84 | 22.05 / 22.54 | 25.05 / 26.35 |
| 4097 / B4 | 5.87 / 7.53 | 70.27 / 70.78 | 76.14 / 77.71 |

## Optional binary gRPC output

| Input tokens / Batch | Outside | Pure Forward | E2E |
| --- | --- | --- | --- |
| 65 / B1 | 1.98 / 2.35 | 9.64 / 10.18 | 11.62 / 12.26 |
| 65 / B4 | 3.08 / 4.03 | 18.24 / 22.35 | 21.33 / 26.16 |
| 4097 / B1 | 2.41 / 3.18 | 21.96 / 22.28 | 24.37 / 25.42 |
| 4097 / B4 | 4.21 / 5.70 | 70.34 / 70.68 | 74.55 / 75.98 |

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
| 65 / B1 | default | 0.04 / 0.07 | 12.17 / 15.53 | 12.04 / 13.39 |
| 65 / B1 | binary | 0.02 / 0.05 | 11.64 / 12.29 | 11.51 / 12.04 |
| 65 / B4 | default | 0.08 / 0.14 | 22.60 / 26.53 | 22.79 / 25.07 |
| 65 / B4 | binary | 0.05 / 0.07 | 21.37 / 26.21 | 21.17 / 23.46 |
| 4097 / B1 | default | 0.05 / 0.09 | 25.10 / 26.40 | 25.09 / 26.21 |
| 4097 / B1 | binary | 0.03 / 0.05 | 24.40 / 25.46 | 24.52 / 25.43 |
| 4097 / B4 | default | 0.12 / 0.17 | 76.25 / 77.87 | 75.36 / 76.65 |
| 4097 / B4 | binary | 0.07 / 0.10 | 74.62 / 76.06 | 74.29 / 75.33 |

Previous tables labelled RPC through arrays included post-RPC conversion and
used native max_num_seqs=4096. They remain in the original experiment outputs;
the corrected boundary/configuration above was measured again, not inferred
by subtracting aggregate statistics.

Additional validation: 115 host gRPC/columnar tests, 116 CUDA constrained-TopK
and batch-switch graph tests, protobuf consistency, and targeted static checks
passed. Full repository CI and NPU model validation were not run.

## Independent restart confirmation

An independent server restart at `b950c66ba88f` re-ran the long B4
case: 4097 tokens per input, 1000 returned sequences, W=[256,512,1024],
K=[256,128,64]. Model configuration, catalog digest, runtime versions, native
capacities and timing boundaries match the earlier run. Runtime/benchmark
code is identical between the two recorded revisions; intervening changes
were documentation only. The restart checks this subset directly after
warmup, rather than repeating all earlier short/B1 cases first.

Each output mode has 300 paired calls and 100 probe-disabled controls.
No samples are discarded or pooled across starts. Each value is mean / P99
in milliseconds.

| Start | Output format | Outside | Pure Forward | E2E |
| --- | --- | --- | --- | --- |
| Original | default | 5.87 / 7.53 | 70.27 / 70.78 | 76.14 / 77.71 |
| Original | binary | 4.21 / 5.70 | 70.34 / 70.68 | 74.55 / 75.98 |
| Restart | default | 5.11 / 6.11 | 70.20 / 70.78 | 75.30 / 76.42 |
| Restart | binary | 3.76 / 4.82 | 70.15 / 70.57 | 73.91 / 75.23 |

Both restart cells had zero measured captures and 600 Decode graph replays.
Repeated requests and default/binary output checks passed. The residual
outside time is reproduced without a model adapter or repeated graph capture.

```bash
OMP_NUM_THREADS=1 python benchmarks/beam_frontend_overhead/run.py \
  --output ./benchmark-output/restart-confirmation \
  --architecture moe --lengths 4097 --batches 4 --result-widths 1000 \
  --samples 300 --control-samples 100 --warmup 5
```
