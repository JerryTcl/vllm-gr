# Region Trie (CUDA V1)

`region_trie_v1` loads the combined `trie_combined.npz` layout used by
GR-NOVI and selects one Trie region per logical request. The complete batch
still shares one model forward and one Trie artifact.

The archive must contain:

- `L1_mask [R, V]`
- `offsets_SB [R, V + 1]`, `flat_SB`
- `pair_keys`, `offsets_SC`, `flat_SC`
- `regions [R]`, or a sibling `manifest.json` with `combined.regions`

Configure the artifact as the active CUDA constraint table:

```yaml
constraint_table:
  enabled: true
  backend: cuda
  path: /opt/tiger/trie/trie_combined.npz
  format: region_trie_v1
  artifact_digest: sha256:<sha256-of-trie_combined.npz>
  tokenizer_digest: sha256:<tokenizer-digest>
  max_top_k: 256
```

Offline callers pass one region per prompt:

```python
outputs = llm.beam_search_v1(
    prompts=[prompt_th, prompt_ph],
    params=params,
    regions=["th", "ph"],
)
```

OpenAI callers can pass one region for the request through
`vllm_xargs.region`. Missing, empty, and unknown region names reproduce the
GR-NOVI behavior: only that logical request runs unconstrained. Valid rows in
the same batch continue to use their selected Region Trie.

The first implementation is CUDA-only and uses static startup loading. It does
not add Region Trie reload/rollback or NPU execution.
