# Offline Beam Search advanced entrypoint

`offline_beam_search.py` keeps the complete flag surface, including fixed
prompt length, dynamic Beam widths, catalog backends, graph execution, and
profiling. Prefer [`quickstart.py`](quickstart.md) for a first run.

```bash
python examples/offline_inference/beam_search/offline_beam_search.py \
  --model "$ONEREC_MODEL" \
  --input-len 1024
```

The directory [README](README.md#beam-search-v1) documents the public
entrypoint, engine configuration, request parameters, NPU graph behavior, and
profiling commands.
