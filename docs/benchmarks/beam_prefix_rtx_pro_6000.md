# Beam prefix attention on RTX PRO 6000

This page records the offline Beam Search latency measured for the CUDA prefix
splitting fixes and the opt-in dense Prefill described in
[Beam decode prefix optimization](../beam_prefix_optimization.md) and
[GPU Prefill capture configuration](../prefill_graph_capture.md). It compares
vLLM-GR with NVIDIA `sid-gr-inference` (NVIDIA-GR) and the SGLang Beam Search
branch on one matrix of input lengths, beam widths and batch sizes.

> Measured on NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB, SM120,
> 188 SMs, driver 580.126.09) / 2026-11-18 / vLLM 0.30.0, torch
> 2.13.0+cu130, CUDA 13.0, cuDNN 9.20, FlashInfer 0.6.18.post1.
> Not a performance guarantee.

## Workload and protocol

- Model: [OneRec-1.7B snapshot](https://huggingface.co/OpenOneRec/OneRec-1.7B/tree/cc4d0b5b7294ecf75e40be1c77fa6b7d284bb84b),
  BF16, one GPU.
- Inputs: deterministic synthetic token IDs from
  `tools/make_qwen3_beam_workload.py --no-tokenizer` in
  [NVIDIA/recsys-examples@2995c4148346](https://github.com/NVIDIA/recsys-examples/tree/2995c4148346/examples/sid-gr-inference),
  vocabulary 151936. Each input length has one 8-request file; batch size B
  uses its first B rows. All engines read the same files.
- Unconstrained Beam Search (no Constraint Table), 3 generated tokens,
  prefix caching and chunked Prefill disabled.
- One engine per (input, beam, batch) point. The whole batch is submitted in
  one call; 3 warmup calls, then 10 timed calls. Tables report the mean wall
  time of the timed calls in milliseconds.
- Matrix: input 1024 and 8192 with beam 32-1024, and beam 64 and 1024 with
  input 256-10240, each at batch 1 and 8 (44 distinct points).

vLLM-GR timing wraps `GRLLM.beam_search_v1` with this configuration
(`k = min(beam, 256)`):

```python
vllm_gr_config = {
    "schema_version": 1,
    "attention_backend": "CUSTOM",
    "constrained_generation": False,
    "beam": {
        "execution_mode": "v1",
        "graph_enabled": True,
        "worker_decision": True,
        "max_width": beam,
        "max_decode_steps": 3,
        "online_batch_sizes": sorted({1, batch}),
    },
    # "candidate_width": 256 is added for beam > 256
    "constraint_table": {"enabled": False, "max_top_k": k},
}
GRLLM(
    vllm_gr_config=vllm_gr_config, model=model, dtype="bfloat16",
    async_scheduling=True, enable_prefix_caching=False,
    enable_chunked_prefill=False, max_model_len=input_len + 3 + 16,
    max_num_batched_tokens=max(batch * (input_len + 3 + 16), batch * beam),
    max_num_seqs=max(4, batch), max_logprobs=k,
    gpu_memory_utilization=0.2, seed=0,
)
BeamSearchParams(beam_width=beam, max_tokens=3, temperature=0.0, ignore_eos=True)
```

The three vLLM-GR columns are:

| Column | Revision | Environment |
| --- | --- | --- |
| `origin/main` | `1afafaf` | defaults |
| PR default | `830f335` | defaults |
| PR + opt-in | `830f335` | `VLLM_GR_DENSE_PREFILL=cudnn`, `VLLM_GR_PREFIX_MIN_CHAIN_US=8` |

Each column was measured as one sweep with a fresh
`VLLM_GR_PREFIX_CALIBRATION_PATH`, back to back under the same conditions:
another idle process kept about 64 GB resident on the GPU. The later review
fixes do not change these code paths for fresh Prefill; re-measuring input
10240 / beam 64 after them gave 135.9 ms (B=1) and 1051.2 ms (B=8).

The reference engines were measured on the same day with the GPU otherwise
idle:

- NVIDIA-GR: `tools/run_qwen3_real_weight_serving.py --decode-backend real
  --batched-decode --continuous --decode-steps 2` (Prefill produces the first
  token), with the Beam and context KV pools sized to the batch.
- SGLang: [cswuyg/sglang@af4e3f42a5a1](https://github.com/cswuyg/sglang/tree/af4e3f42a5a1)
  (`feature/beam_search`) through `tools/run_sglang_beam_benchmark.py` from the
  same recsys-examples revision, with `--disable-radix-cache`,
  `--mem-fraction-static 0.6`, `max_running_requests = B * beam + B` and
  `context_length = input + 64`. Its request-slot cap of 4096 was raised
  locally to fit `B * beam + B` requests.

Re-measuring NVIDIA-GR under the shared-GPU conditions reproduced 12 of the
batch-1 and batch-8 beam-64 points within 1.2%, in both directions. At one
checked point the shared GPU cost `origin/main` about 1%, so the comparison
slightly favors the reference engines. Beam outputs were not checked for equivalence; with
flat synthetic score distributions, low-ranked beams can reorder on ties.

## Results

Mean latency reduction relative to NVIDIA-GR; positive means vLLM-GR is
faster.

| Panel | Batch | `origin/main` | PR default | PR + opt-in |
| --- | --- | ---: | ---: | ---: |
| Input 1024, beam 32-1024 | 1 | +1.6% | +1.7% | +2.9% |
| Input 1024, beam 32-1024 | 8 | +7.0% | +7.0% | +7.4% |
| Input 8192, beam 32-1024 | 1 | -7.2% | -3.2% | +1.1% |
| Input 8192, beam 32-1024 | 8 | -0.9% | -0.9% | +3.3% |
| Beam 64, input 256-10240 | 1 | -8.0% | -6.4% | -2.8% |
| Beam 64, input 256-10240 | 8 | +1.3% | +1.2% | +2.9% |
| Beam 1024, input 256-10240 | 1 | +8.0% | +9.2% | +10.2% |
| Beam 1024, input 256-10240 | 8 | +11.0% | +10.9% | +12.2% |
| All 44 points | | +1.8% | +2.6% | +4.7% |
| Points slower than NVIDIA-GR | | 22 | 22 | 10 |

Run-to-run noise is about +/-2.5%: where PR default runs the same kernels as
`origin/main` (no prefix split selected, dense Prefill off), the two sweeps
still differ by up to 2.5%. Against `origin/main`, PR default is 0.7% faster
on average and PR + opt-in 2.7% faster. With opt-in, three short-input points
measured 2.7-3.2% slower (at most 1.4 ms), within about that noise. Batch-1
inputs up to 2048 tokens remain 0.5-1.1 ms behind NVIDIA-GR; traces attribute
this to per-call host work rather than kernels.

Latency per point (ms):

| Batch | Beam | Input | `origin/main` | PR default | PR + opt-in | NVIDIA-GR | SGLang |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 1024 | 23.8 | 23.8 | 23.6 | 22.5 | 27.2 |
| 1 | 32 | 8192 | 120.2 | 111.6 | 106.6 | 107.3 | 119.4 |
| 1 | 64 | 256 | 14.8 | 14.8 | 14.9 | 14.3 | 18.0 |
| 1 | 64 | 512 | 18.5 | 18.2 | 19.0 | 17.9 | 21.8 |
| 1 | 64 | 1024 | 24.4 | 24.3 | 23.3 | 22.5 | 28.3 |
| 1 | 64 | 2048 | 36.0 | 36.6 | 34.3 | 33.2 | 40.4 |
| 1 | 64 | 4096 | 59.6 | 60.2 | 55.9 | 54.7 | 66.6 |
| 1 | 64 | 8192 | 119.8 | 113.8 | 108.6 | 107.8 | 127.1 |
| 1 | 64 | 10240 | 154.5 | 145.7 | 136.7 | 138.8 | 164.0 |
| 1 | 128 | 1024 | 25.2 | 24.9 | 24.4 | 24.0 | 31.6 |
| 1 | 128 | 8192 | 120.1 | 114.4 | 108.9 | 109.4 | 146.7 |
| 1 | 256 | 1024 | 28.2 | 28.4 | 29.1 | 28.9 | 42.9 |
| 1 | 256 | 8192 | 123.0 | 119.7 | 113.8 | 115.4 | 207.1 |
| 1 | 512 | 1024 | 35.6 | 36.5 | 35.7 | 40.9 | 60.6 |
| 1 | 512 | 8192 | 131.2 | 129.6 | 125.6 | 128.4 | 262.9 |
| 1 | 1024 | 256 | 41.5 | 40.0 | 40.6 | 49.0 | 63.4 |
| 1 | 1024 | 512 | 45.4 | 43.8 | 44.5 | 53.0 | 76.1 |
| 1 | 1024 | 1024 | 50.6 | 49.0 | 49.3 | 58.1 | 97.8 |
| 1 | 1024 | 2048 | 61.6 | 61.6 | 62.2 | 70.5 | 137.1 |
| 1 | 1024 | 4096 | 89.0 | 89.5 | 87.4 | 94.0 | 218.9 |
| 1 | 1024 | 8192 | 154.0 | 152.7 | 148.0 | 152.2 | 402.1 |
| 1 | 1024 | 10240 | 189.8 | 190.6 | 181.0 | 183.9 | 495.6 |
| 8 | 32 | 1024 | 94.1 | 93.7 | 93.2 | 92.5 | 106.6 |
| 8 | 32 | 8192 | 823.1 | 823.5 | 785.4 | 810.7 | 912.4 |
| 8 | 64 | 256 | 44.3 | 43.2 | 45.7 | 46.6 | 60.2 |
| 8 | 64 | 512 | 61.9 | 62.9 | 61.1 | 63.9 | 85.3 |
| 8 | 64 | 1024 | 99.9 | 101.1 | 98.9 | 102.1 | 124.3 |
| 8 | 64 | 2048 | 184.9 | 185.0 | 184.2 | 189.2 | 232.1 |
| 8 | 64 | 4096 | 374.4 | 374.2 | 367.6 | 381.2 | 441.4 |
| 8 | 64 | 8192 | 843.9 | 844.2 | 806.2 | 827.6 | 972.9 |
| 8 | 64 | 10240 | 1112.3 | 1112.3 | 1051.4 | 1076.6 | 1232.3 |
| 8 | 128 | 1024 | 114.2 | 113.3 | 114.2 | 120.7 | 159.6 |
| 8 | 128 | 8192 | 865.5 | 865.0 | 828.2 | 843.0 | 1115.1 |
| 8 | 256 | 1024 | 143.1 | 143.4 | 142.6 | 155.6 | 269.9 |
| 8 | 256 | 8192 | 903.1 | 902.9 | 864.6 | 895.7 | 1462.7 |
| 8 | 512 | 1024 | 196.0 | 196.8 | 196.3 | 224.0 | 366.2 |
| 8 | 512 | 8192 | 975.6 | 973.0 | 934.0 | 970.2 | 2054.5 |
| 8 | 1024 | 256 | 246.1 | 244.9 | 245.4 | 305.3 | 644.6 |
| 8 | 1024 | 512 | 264.5 | 268.5 | 266.8 | 325.3 | 717.4 |
| 8 | 1024 | 1024 | 309.1 | 308.7 | 309.5 | 367.5 | 910.1 |
| 8 | 1024 | 2048 | 404.8 | 404.6 | 403.2 | 462.4 | 1230.0 |
| 8 | 1024 | 4096 | 621.8 | 623.2 | 614.0 | 677.4 | 1887.9 |
| 8 | 1024 | 8192 | 1134.0 | 1135.8 | 1095.9 | 1156.7 | 3494.8 |
| 8 | 1024 | 10240 | 1410.6 | 1407.7 | 1348.9 | 1415.9 | 4135.4 |
