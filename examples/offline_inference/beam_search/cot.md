# Offline CoT-to-SID Beam Search

`cot.py` generates a thinking span first, then runs constrained SID Beam
Search with the same complete prompt. It requires `--catalog` and `--prompt`.

```bash
python examples/offline_inference/beam_search/cot.py \
  --model "$ONEREC_MODEL" \
  --prompt /absolute/path/to/question.txt \
  --catalog /absolute/path/to/constraint_triples.json
```

`--cot-max-tokens` and `--cot-on-limit` control the thinking budget and
handoff behavior. See
[Optional thinking before SID generation](README.md#optional-thinking-before-sid-generation).
