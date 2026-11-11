# RFC: Track and reduce Beam serving latency outside execution intervals

Status: Draft for discussion. Updated: 2026-11-11.
Owner: Unassigned. Root-cause attribution and optimization proposals are pending.

## Motivation

Constrained Beam serving still spends several milliseconds outside its model
and candidate-selection intervals. This overhead remains observable with a
standard Qwen3 MoE model, synthetic inputs and no custom model adapter, after
enabling columnar results, partial TopK and reuse of Decode graphs across batch
sizes. The measured residual increases with batch size and prompt length.

This RFC proposes a shared measurement and investigation plan, rather than a
specific replacement for the scheduler, IPC or RPC implementation. The goal is
to identify the dominant remaining costs and validate focused improvements
without changing search behavior or moving work outside the reported boundary.

The [public reproduction](../benchmarks/beam_frontend_overhead/README.md) and
[measurement record](../benchmarks/beam_frontend_overhead/RESULTS.md) provide
the starting point. The branch is
[`repro/beam_frontend_overhead`](https://github.com/JiusiServe/vllm-gr/tree/repro/beam_frontend_overhead),
based on `main` at `142b8b2109f7e03979247ef15017f16c03a94f72`.

## Measurement contract

Report arithmetic mean and P99 in milliseconds, retaining all measured samples.

| Metric | Boundary |
| --- | --- |
| Pure Forward | Sum of six CUDA Events intervals: three model/logits executions and three Trie/TopK/Beam selections. Includes input preparation, LM head and idle time inside those intervals; excludes gaps between them. |
| E2E | Client `GenerateBatch` call start through complete protobuf response return. Includes RPC encoding/decoding, communication and serving. Excludes prebuilt request construction and post-return array/business conversion. |
| Outside | `E2E - Pure Forward`, calculated for each paired request. Includes interval-external processing, IPC, stage handoffs, result handling and communication. |

Pure Forward is an execution-interval measurement, not a sum of kernel times.
Outside is not exclusively frontend time and is not all necessarily removable.
Calculate its P99 from the per-request differences, using linear percentile
interpolation; never subtract aggregate P99 values. Component P99 values are
not additive.

The [client timer](../benchmarks/beam_frontend_overhead/run.py) and
[Worker probe](../benchmarks/beam_frontend_overhead/frontend_probe.py) implement
these boundaries. CUDA Events are preallocated; collection happens through a
separate drain RPC outside the measured request. Separate probe-disabled calls
check observer effects, but do not provide a paired Forward/Outside measurement.

## Current evidence

The validated workload uses a generated standard `Qwen3MoeForCausalLM`
configuration with dummy weights: 12 layers, hidden size 1024, 32 experts and
four selected experts per token. It runs BF16 without quantization on one H20,
using vLLM 0.30.0, Torch 2.13.0+cu130, grpcio 1.85.1 and protobuf 7.36.2.

Requests use deterministic synthetic token IDs and a synthetic three-token
catalog. They use W=[256,512,1024], per-parent K=[256,128,64], Beam capacity
1024, 1000 returned sequences and three generation steps. Native capacities
are `max_num_seqs=4`, `max_num_batched_tokens=16388`, `max_model_len=4100`;
the constraint backend's TopK and candidate capacities are both 256.

Prefill is eager; Decode uses GR CUDA Graphs. Prefix caching is disabled,
request-local KV remains enabled, and no result-cache layer is present.
Python GC uses its normal policy. One explicitly constructed RPC batch is in
flight at a time; this is a closed-loop latency test without an external batcher.

The full matrix measures 65/4097 tokens per input, B1/B4, and default/binary
output. Its Outside mean / P99 values are:

| Tokens / Batch | Default output | Binary output |
| --- | --- | --- |
| 65 / B1 | 2.43 / 3.29 | 1.98 / 2.35 |
| 65 / B4 | 4.38 / 5.39 | 3.08 / 4.03 |
| 4097 / B1 | 3.00 / 3.84 | 2.41 / 3.18 |
| 4097 / B4 | 5.87 / 7.53 | 4.21 / 5.70 |

An independent restart confirms the long B4 residual. Each cell below is
mean / P99; starts are reported separately, without pooling or selecting a
preferred run.

| Start / Output | Outside | Pure Forward | E2E |
| --- | --- | --- | --- |
| A / default | 5.87 / 7.53 | 70.27 / 70.78 | 76.14 / 77.71 |
| A / binary | 4.21 / 5.70 | 70.34 / 70.68 | 74.55 / 75.98 |
| B / default | 5.11 / 6.11 | 70.20 / 70.78 | 75.30 / 76.42 |
| B / binary | 3.76 / 4.82 | 70.15 / 70.57 | 73.91 / 75.23 |

Every cell contains 300 paired calls, 100 separate probe-disabled controls and
five excluded warmups. Start A contains 2400 paired calls across its full
matrix; start B contains 600 across the long B4 subset. Every cell recorded
zero captures and 600 Decode graph replays. Repeated/default/binary token IDs
and FP32 scores matched. Recorded source revisions are `fb734fda5cf5` and
`b950c66`; runtime and benchmark code are identical between them.

This establishes residual overhead on the standard serving path. It does not
establish which component dominates, reproduce historical GC stalls, or measure
a same-model gap against another framework. The between-start variation also
requires replication before attributing small differences to a code change.

## Proposed investigation

### 1. Preserve a reproducible baseline

Run from the repository root in the documented CUDA environment, with an idle
assigned GPU and a fresh output directory for each start:

```bash
OMP_NUM_THREADS=1 python benchmarks/beam_frontend_overhead/run.py \
  --output ./benchmark-output/rfc-baseline \
  --architecture moe --lengths 65,4097 --batches 1,4 \
  --result-widths 1000 --samples 300 --control-samples 100 --warmup 5
```

The harness tests both output formats. Preserve `summary.json` with raw samples,
`metrics.csv`, configuration, catalog identity, runtime versions and logs. Record
the exact source revision and any local patch; repeat independent starts without
dropping slow samples. The full configuration and restart-only command are in
the measurement record linked above.

### 2. Attribute the critical path

Add opt-in diagnostic tracing with a request/session correlation ID across the
following boundaries. These are investigation areas, not measured attributions.

| Area | Boundaries to observe |
| --- | --- |
| Client / RPC | Call start, request encode/send, response receive/decode, call return |
| Frontend ingress | Handler entry, conversion/validation, Beam submission |
| Core and Worker | IPC send/receive, admission/scheduling, dispatch, host work and gaps between execution intervals |
| Terminal output | Completion readiness, device-to-host result transfer, packing, return IPC, frontend result handling |
| RPC return | Response construction/encoding and transport handoff |

Reuse existing trace boundaries where possible. Keep observations bounded and
dump traces after measurement; avoid new per-request file writes or device
synchronization. Use causal correlation and a verified common host clock for
cross-process spans. CUDA Event durations and host timestamps are different
clock domains and cannot be directly subtracted as absolute timestamps.

Report non-overlapping critical-path contributions and an explicit unassigned
remainder. CPU spans can overlap GPU work or each other, so summing all host
durations is not a valid decomposition of Outside. Profiled traces explain
mechanisms; unprofiled measurements establish performance changes.

### 3. Test one mechanism at a time

- Vary result count, batch size and prompt length separately around the baseline
  to distinguish output-size costs from input preparation and dispatch costs.
- Compare default and optional binary responses through the same E2E boundary;
  report post-RPC conversion separately if consumers need arrays.
- Measure IPC, terminal-result handling and stage handoffs before proposing
  changes to queues, scheduling or completion delivery.
- Observe allocations, GC and host scheduling when tails occur, keeping the
  default policy in acceptance runs. Do not infer a GC cause from latency alone.
- Check observer effects with separate probe-disabled runs. Do not subtract a
  fixed estimated tracing cost from samples.

Prioritize the largest measured critical-path contributor. Dense-model and
compiled-Prefill checks can later test generality; neither has been validated
in this public measurement record. Open-loop/batcher throughput is a separate
experiment and must not be mixed with this fixed-load latency baseline.

## Acceptance and compatibility

Each proposed optimization should supply:

1. A focused before/after comparison with identical inputs, capacities, output
   format, cache policy and search parameters. Use at least three independent
   starts per variant as the initial replication plan, retaining every start.
2. Paired Outside, Pure Forward and E2E mean/P99 for the baseline matrix, raw
   samples, sample counts, runtime identity and probe-disabled controls. Increase
   sample counts when tail conclusions remain unstable.
3. Output equivalence and no unexpected steady-state captures. Preserve request
   cancellation, failure propagation, cleanup and result-buffer ownership;
   optional binary output must preserve the compatible default response.
4. Evidence connecting the change to the measured cost, including any remaining
   unassigned time and regressions. Moving work beyond response return, reducing
   search widths, or enabling caches does not count as an improvement here.

Agree on a latency budget after attribution and replication. This RFC does not
assume a zero-overhead target or promise that the entire residual is recoverable.

## Tracking

Maintain this document as evidence arrives. Link each experiment or focused PR
with its exact source revision, configuration, artifacts, mean/P99 and conclusion.
Keep superseded measurements labeled rather than silently replacing them.

| Item | State | Evidence / next action |
| --- | --- | --- |
| T1: Public standard-model reproduction | Done | Start A, `fb734fda5cf5`; full matrix in the measurement record |
| T2: Independent restart confirmation | Done | Start B, `b950c66`; long B4 default/binary subset |
| T3: Correlated component tracing | Pending | Identify boundary owners and collect diagnostic traces |
| T4: Dominant-cost ranking | Pending | Separate critical-path contributions and unassigned remainder |
| T5: Focused optimization and ablation | Pending | Implement only after T4; one mechanism per comparison |
| T6: Regression matrix and agreed budget | Pending | Replicate gains, verify lifecycle behavior, document residual costs |

## Related work and discussion

- [E2E attribution tracking #353](https://github.com/JiusiServe/vllm-gr/issues/353):
  broader historical work; this RFC adds a standard-model residual-latency
  reproduction with a fixed measurement contract. Its timings are not directly
  comparable across models, hardware, revisions or execution policies.
- [Offline GC issue #426](https://github.com/JiusiServe/vllm-gr/issues/426):
  a separately attributed tail case, not an established cause of this residual.
- [Columnar output #630](https://github.com/JiusiServe/vllm-gr/pull/630),
  [partial TopK #631](https://github.com/JiusiServe/vllm-gr/pull/631), and
  [batch-switch graph reuse #632](https://github.com/JiusiServe/vllm-gr/pull/632):
  included in the reproduction branch; this baseline does not measure their
  individual before/after gains.
- [Graph lifecycle RFC #644](https://github.com/JiusiServe/vllm-gr/issues/644):
  related lifecycle design; no measured recapture explains the residual here.

Discussion should first confirm the timing contract, trace boundaries and
ownership of investigation items. A feedback window can be assigned when this
draft is opened for community discussion. No API change or implementation plan
is considered accepted by the existence of this document.
