# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate a standard-model config and an integer-only synthetic catalog."""

import hashlib
import json
from pathlib import Path

import numpy as np
from transformers import Qwen3Config, Qwen3MoeConfig

from vllm_gr.constraints.artifact import save_constraint_table_artifact
from vllm_gr.constraints.table import ConstraintTable, ConstraintTableMetadata

VOCAB = 8192
WIDTHS = [256, 512, 1024]
PER_PARENT_WIDTHS = [256, 128, 64]


def prepare(folder: Path, architecture: str, tiny: bool) -> dict:
    """Use vLLM's dummy loader; no model weights or external tokenizer are needed."""
    model = folder / "model"
    model.mkdir(parents=True, exist_ok=True)
    hidden, layers = (256, 2) if tiny else (1024, 12)
    common = dict(
        vocab_size=VOCAB,
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=4 if tiny else 8,
        num_key_value_heads=2 if tiny else 4,
        head_dim=64 if tiny else 128,
        max_position_embeddings=8192,
        tie_word_embeddings=False,
        torch_dtype="bfloat16",
    )
    if architecture == "moe":
        config = Qwen3MoeConfig(
            **common,
            intermediate_size=hidden * 3,
            moe_intermediate_size=128 if tiny else 512,
            num_experts=8 if tiny else 32,
            num_experts_per_tok=2 if tiny else 4,
        )
        config.architectures = ["Qwen3MoeForCausalLM"]
    else:
        config = Qwen3Config(**common, intermediate_size=hidden * 3)
        config.architectures = ["Qwen3ForCausalLM"]
    config.save_pretrained(model)

    # 256 first tokens, 32 second tokens per first token, 128 final tokens.
    # One real row has degree 8192 to retain a large candidate work capacity.
    first = np.arange(256, dtype=np.int32)
    second = np.tile(np.arange(32, dtype=np.int32), 256)
    offsets1 = np.minimum(np.arange(VOCAB + 1), 256).astype(np.int32) * 32
    keys = np.repeat(first.astype(np.int64), 32) * VOCAB + second
    degrees = np.full(keys.size, 128, dtype=np.int32)
    degrees[-1] = VOCAB
    offsets2 = np.concatenate(([0], np.cumsum(degrees))).astype(np.int32)
    third = np.concatenate(
        (np.tile(np.arange(128, dtype=np.int32), keys.size - 1), np.arange(VOCAB, dtype=np.int32))
    )
    arrays = [first, offsets1, second, keys, offsets2, third]
    token_digest = "sha256:" + hashlib.sha256(b"synthetic-integer-vocabulary-8192-v1").hexdigest()
    table = ConstraintTable(
        *arrays,
        metadata=ConstraintTableMetadata(
            format="constraint_table_v1",
            artifact_digest=None,
            tokenizer_digest=token_digest,
            vocab_size=VOCAB,
            max_prefix1_degree=32,
            max_prefix2_degree=VOCAB,
            total_bytes=sum(a.nbytes for a in arrays),
        ),
    )
    table_path = folder / "catalog.safetensors"
    digest = save_constraint_table_artifact(table, table_path)
    gr = dict(
        schema_version=1,
        attention_backend="CUSTOM",
        beam=dict(
            execution_mode="v1",
            max_width=1024,
            max_decode_steps=3,
            graph_enabled=True,
            worker_decision=True,
            online_batch_sizes=[1, 2, 4],
        ),
        constraint_table=dict(
            enabled=True,
            backend="cuda",
            max_top_k=256,
            candidate_width=256,
            path=str(table_path.resolve()),
            format="constraint_table_v1",
            artifact_digest=digest,
            tokenizer_digest=token_digest,
        ),
    )
    (folder / "gr.json").write_text(json.dumps(gr, indent=2) + "\n")
    return dict(
        model=str(model.resolve()),
        gr=gr,
        architecture=architecture,
        preset="tiny" if tiny else "repro",
        artifact_digest=digest,
        artifact_bytes=table.total_bytes,
    )
