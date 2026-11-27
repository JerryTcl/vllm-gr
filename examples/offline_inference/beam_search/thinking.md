# Offline thinking generation

`thinking.py` runs ordinary LLM thinking instead of Beam Search. It requires a
UTF-8 prompt file.

```bash
python examples/offline_inference/beam_search/thinking.py \
  --model "$ONEREC_MODEL" \
  --prompt /absolute/path/to/question.txt
```

`--repeat` benchmarks generation, and `--output-json` saves the report. See
[Optional thinking before SID generation](README.md#optional-thinking-before-sid-generation)
for the prompt formats and budget rules.
