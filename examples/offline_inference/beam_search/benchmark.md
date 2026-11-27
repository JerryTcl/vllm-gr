# Offline Beam Search benchmark

`benchmark.py` measures Beam Search batch latency without starting a profiler.
It requires `--repeat`.

```bash
python examples/offline_inference/beam_search/benchmark.py \
  --model "$ONEREC_MODEL" \
  --repeat 5 \
  --output-json /absolute/path/to/beam_benchmark.json
```

Use `--catalog` for constrained batches and `--warmup-runs` to set the number
of untimed batches. See
[Measure latency without profiling](README.md#measure-latency-without-profiling).
For paired cold/warm A/B measurements against the OneRec dataset, use the
[offline A/B benchmark](../../../tools/AB_Tests/README.md) instead.
