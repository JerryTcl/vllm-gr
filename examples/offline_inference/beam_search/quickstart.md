# Offline Beam Search quickstart

`quickstart.py` is the teaching entrypoint for the offline example. It uses a
natural-length prompt and an optional Constraint Table.

```bash
python examples/offline_inference/beam_search/quickstart.py --model "$ONEREC_MODEL"
```

Pass `--catalog` after generating constraint triples to restrict generation.
See [Run the V1 example](README.md#run-the-v1-example) and
[Run with a Constraint Table](README.md#run-with-a-constraint-table).

Use [`offline_beam_search.py`](offline_beam_search.md) when you need the full
flag surface, [`thinking.py`](thinking.md) for ordinary thinking, and
[`cot.py`](cot.md) for thinking before SID Beam Search.
