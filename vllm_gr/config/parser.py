# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import MISSING, replace
from enum import Enum
from pathlib import Path
from typing import Any, Final, Literal, TypeAlias, cast

from vllm_gr.config.models import (
    AUTO_CONSTRAINT_TABLE_PATH,
    ConstraintBackend,
    ConstraintTableConfig,
    GRAttentionBackend,
    GRConfig,
)
from vllm_gr.config.schema import (
    canonical_keys,
    config_fields,
    field_default,
    legacy_key,
    should_emit,
)

GR_CONFIG_KEY: Final = "vllm_gr"
GRConfigInput: TypeAlias = str | Path | Mapping[str, object] | GRConfig
CATALOG_PATH_KEY: Final = legacy_key(GRConfig, "catalog_path")
CONSTRAINT_BACKEND_KEY: Final = legacy_key(GRConfig, "constraint_backend")
ATTENTION_BACKEND_KEY: Final = legacy_key(GRConfig, "attention_backend")
VOCAB_OFFSET_KEY: Final = legacy_key(GRConfig, "vocab_offset")
VOCAB_SLICE_SIZE_KEY: Final = legacy_key(GRConfig, "vocab_slice_size")
PREFILL_FA_VERSION_KEY: Final = legacy_key(GRConfig, "prefill_flash_attn_version")
SHORT_SUFFIX_ATTENTION_KEY: Final = legacy_key(GRConfig, "enable_short_suffix_attention")
BEAM_GRAPH_ENABLED_KEY: Final = legacy_key(GRConfig, "beam_graph_enabled")
BEAM_ADAPTER_CACHE_ENABLED_KEY: Final = legacy_key(GRConfig, "beam_adapter_cache_enabled")
BEAM_GRAPH_BATCH_SIZE_KEY: Final = legacy_key(GRConfig, "beam_graph_batch_size")
BEAM_GRAPH_WIDTHS_KEY: Final = legacy_key(GRConfig, "beam_graph_widths")
BEAM_ONLINE_BATCH_SIZES_KEY: Final = legacy_key(GRConfig, "beam_online_batch_sizes")
BEAM_MAX_WIDTH_KEY: Final = legacy_key(GRConfig, "beam_max_width")
BEAM_MAX_DECODE_STEPS_KEY: Final = legacy_key(GRConfig, "beam_max_decode_steps")
NPU_DECODE_EXECUTION_KEY: Final = legacy_key(GRConfig, "npu_decode_execution")
BEAM_WORKER_DECISION_KEY: Final = legacy_key(GRConfig, "beam_worker_decision")
BEAM_ENGINE_DRIVEN_KEY: Final = legacy_key(GRConfig, "beam_engine_driven")
BEAM_EXECUTION_MODE_KEY: Final = legacy_key(GRConfig, "beam_execution_mode")
BEAM_SID_STAGE_VOCAB_KEY: Final = legacy_key(GRConfig, "beam_sid_stage_vocab")
CONSTRAINED_GENERATION_KEY: Final = legacy_key(GRConfig, "constrained_generation")
GR_CONFIG_KEY_PREFIX: Final = "vllm_gr_"
KNOWN_GR_CONFIG_KEYS: Final = frozenset(
    spec.legacy_key for _, spec in config_fields(GRConfig) if spec.legacy_key is not None
)
_TOP_LEVEL_KEYS: Final = canonical_keys(GRConfig)
_BEAM_KEYS: Final = canonical_keys(GRConfig, ("beam",))
_CONSTRAINT_TABLE_KEYS: Final = canonical_keys(ConstraintTableConfig)


def _reject_unknown_gr_keys(additional_config: Mapping[object, object]) -> None:
    unknown_keys = sorted(
        key
        for key in additional_config
        if isinstance(key, str)
        and key.startswith(GR_CONFIG_KEY_PREFIX)
        and key not in KNOWN_GR_CONFIG_KEYS
    )
    if unknown_keys:
        unknown = ", ".join(repr(key) for key in unknown_keys)
        raise ValueError(f"unknown vLLM-GR additional_config key(s): {unknown}")


def _as_string_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"vllm-gr {name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"vllm-gr {name} keys must be strings")
    return cast(Mapping[str, object], value)


def _reject_unknown_mapping_keys(
    mapping: Mapping[str, object],
    allowed: frozenset[str],
    name: str,
) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        rendered = ", ".join(repr(key) for key in unknown)
        raise ValueError(f"unknown vllm-gr {name} key(s): {rendered}")


def _normalize_artifact_path(
    value: object,
    *,
    base_dir: str | Path | None,
) -> object:
    if value is None or not isinstance(value, str) or not value.strip():
        return value
    artifact_path = Path(value).expanduser()
    if artifact_path.is_absolute():
        return str(artifact_path.resolve(strict=False))
    if base_dir is None:
        raise ValueError("vllm-gr constraint_table.path must be absolute for mapping input")
    normalized_base = Path(base_dir).expanduser().resolve(strict=False)
    return str((normalized_base / artifact_path).resolve(strict=False))


def _canonical_kwargs(
    cls: type[Any], mapping: Mapping[str, object], *, prefix: str = ""
) -> dict[str, Any]:
    """Supply only explicit inputs; constructor defaults remain authoritative."""
    result: dict[str, Any] = {}
    for item, spec in config_fields(cls):
        if spec.canonical_path is None:
            continue
        value: object = mapping
        for part in spec.canonical_path:
            value = value.get(part, MISSING) if isinstance(value, Mapping) else MISSING
        name = prefix + ".".join(spec.canonical_path)
        if value is MISSING:
            if spec.canonical_required:
                raise ValueError(f"vllm-gr configuration requires {name}")
            continue
        if spec.canonical_strict_bool and not isinstance(value, bool):
            raise ValueError(f"vllm-gr {name} must be a boolean")
        result[item.name] = value
    return result


def _primitive(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, ConstraintTableConfig):
        return _canonical_mapping(value)
    return value


def _canonical_mapping(config: GRConfig | ConstraintTableConfig) -> dict[str, object]:
    result: dict[str, object] = {}
    for item, spec in config_fields(type(config)):
        path = spec.canonical_path
        value = getattr(config, item.name)
        if path is None or not should_emit(item, value, spec.canonical_emit):
            continue
        target = result
        for part in path[:-1]:
            target = cast(dict[str, object], target.setdefault(part, {}))
        target[path[-1]] = _primitive(value)
    return result


def gr_config_from_mapping(
    raw: object,
    *,
    base_dir: str | Path | None = None,
) -> GRConfig:
    """Build a validated config from a canonical schema-v1 mapping."""
    mapping = _as_string_mapping(raw, "configuration")
    _reject_unknown_mapping_keys(mapping, _TOP_LEVEL_KEYS, "configuration")
    beam_raw = _as_string_mapping(mapping.get("beam", {}), "beam")
    _reject_unknown_mapping_keys(beam_raw, _BEAM_KEYS, "beam")
    table_raw = _as_string_mapping(mapping.get("constraint_table", {}), "constraint_table")
    _reject_unknown_mapping_keys(table_raw, _CONSTRAINT_TABLE_KEYS, "constraint_table")

    values = _canonical_kwargs(GRConfig, {**mapping, "beam": beam_raw})
    table_values = _canonical_kwargs(ConstraintTableConfig, table_raw, prefix="constraint_table.")
    if "path" in table_values:
        table_values["path"] = _normalize_artifact_path(table_values["path"], base_dir=base_dir)
    values["constraint_table"] = ConstraintTableConfig(**table_values)
    return GRConfig(**values)


def _reject_lossy_canonical_conversion(config: GRConfig) -> None:
    legacy_values = [
        item.name
        for item, spec in config_fields(GRConfig)
        if spec.canonical_path is None
        and spec.legacy_key is not None
        and getattr(config, item.name) != field_default(item)
    ]
    if legacy_values:
        raise ValueError(
            "cannot encode legacy-only GRConfig field(s) in canonical schema-v1: "
            + ", ".join(legacy_values)
        )


def gr_config_to_mapping(config: GRConfig) -> dict[str, object]:
    """Convert a canonical config to JSON/pickle-safe primitive values."""
    if not isinstance(config, GRConfig):
        raise TypeError("config must be a GRConfig")
    _reject_lossy_canonical_conversion(config)
    if config.constraint_table is None:
        raise ValueError("cannot encode GRConfig without canonical constraint_table state")
    return _canonical_mapping(config)


def load_gr_config(path: str | Path) -> GRConfig:
    """Load canonical schema-v1 YAML on the Driver and normalize its paths."""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - packaging contract
        raise RuntimeError("PyYAML is required to load a vLLM-GR config") from exc

    config_path = Path(path).expanduser().resolve(strict=False)
    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            raw = yaml.safe_load(config_file)
    except OSError as exc:
        raise ValueError(f"cannot read vllm-gr config {str(config_path)!r}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid vllm-gr YAML {str(config_path)!r}: {exc}") from exc
    if raw is None:
        raise ValueError(f"vllm-gr config {str(config_path)!r} is empty")
    return gr_config_from_mapping(raw, base_dir=config_path.parent)


def normalize_gr_config_input(value: GRConfigInput) -> GRConfig:
    if isinstance(value, GRConfig):
        return value
    if isinstance(value, (str, Path)):
        return load_gr_config(value)
    if isinstance(value, Mapping):
        return gr_config_from_mapping(value)
    raise TypeError(
        f"vllm_gr_config must be a YAML path, mapping, or GRConfig; got {type(value).__name__}"
    )


def _legacy_keys_present(
    additional_config: Mapping[object, object],
) -> list[str]:
    return sorted(key for key in KNOWN_GR_CONFIG_KEYS if key in additional_config)


def parse_gr_config(vllm_config: object | None) -> GRConfig:
    if vllm_config is None:
        return GRConfig()
    additional_config = getattr(vllm_config, "additional_config", None)
    if additional_config is None:
        return GRConfig()
    if not isinstance(additional_config, Mapping):
        # Upstream also accepts opaque SupportsHash configuration objects.
        # They cannot carry primitive vllm_gr_* keys, so leave them untouched
        # and treat them as an empty GR configuration on the read path.
        return GRConfig()
    _reject_unknown_gr_keys(additional_config)
    canonical_present = GR_CONFIG_KEY in additional_config
    legacy_keys = _legacy_keys_present(additional_config)
    if canonical_present and legacy_keys:
        raise ValueError(
            "canonical additional_config['vllm_gr'] cannot be combined with "
            "legacy vLLM-GR additional_config key(s): "
            + ", ".join(repr(key) for key in legacy_keys)
        )
    if canonical_present:
        return gr_config_from_mapping(additional_config[GR_CONFIG_KEY])
    return GRConfig(
        **{
            item.name: additional_config[spec.legacy_key]
            for item, spec in config_fields(GRConfig)
            if spec.legacy_key is not None and spec.legacy_key in additional_config
        }
    )


def write_gr_config(vllm_config: object, gr_config: GRConfig) -> None:
    additional_config = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional_config, MutableMapping):
        raise ValueError("vLLM-GR additional_config must be a mutable mapping")
    _reject_unknown_gr_keys(additional_config)
    if GR_CONFIG_KEY in additional_config:
        raise ValueError(
            "legacy vLLM-GR config cannot be combined with canonical additional_config['vllm_gr']"
        )
    if gr_config.constraint_table is not None:
        raise ValueError(
            "write_gr_config cannot serialize canonical constraint_table state; "
            "use write_canonical_gr_config"
        )

    for item, spec in config_fields(GRConfig):
        if spec.legacy_key is None:
            continue
        value = getattr(gr_config, item.name)
        if should_emit(item, value, spec.legacy_emit):
            additional_config[spec.legacy_key] = _primitive(value)
        else:
            additional_config.pop(spec.legacy_key, None)


def write_canonical_gr_config(vllm_config: object, gr_config: GRConfig) -> None:
    """Write a complete canonical primitive mapping into ``VllmConfig``."""

    additional_config = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional_config, MutableMapping):
        raise ValueError("vLLM-GR additional_config must be a mutable mapping")
    _reject_unknown_gr_keys(additional_config)
    legacy_keys = _legacy_keys_present(additional_config)
    if legacy_keys:
        raise ValueError(
            "canonical vLLM-GR config cannot be combined with legacy "
            "additional_config key(s): " + ", ".join(repr(key) for key in legacy_keys)
        )
    additional_config[GR_CONFIG_KEY] = gr_config_to_mapping(gr_config)
    # EngineArgs reads the defaults before applying the canonical mapping.
    # UniProc keeps that same config object; it must not retain the old parse.
    try:
        object.__setattr__(vllm_config, "_vllm_gr_parsed_config", None)
    except (AttributeError, TypeError):
        pass


def _legacy_engine_config(
    args: Any,
    existing_gr_config: GRConfig,
    backend_is_explicit: bool,
    backend_override: GRAttentionBackend | None,
    device_type: str,
) -> GRConfig:
    from vllm_gr.config.validation import validate_beam_engine_driven

    catalog_path = getattr(args, "catalog_path", None)
    constraint_backend = getattr(args, "constraint_backend", None)
    constrained_generation = getattr(args, "constrained_generation", None)
    effective_catalog_path = (
        catalog_path if catalog_path is not None else existing_gr_config.catalog_path
    )
    effective_constraint_backend = (
        ConstraintBackend(constraint_backend)
        if constraint_backend is not None
        else existing_gr_config.constraint_backend
    )
    effective_constrained_generation = (
        constrained_generation
        if constrained_generation is not None
        else existing_gr_config.constrained_generation
    )
    effective_beam_worker_decision = (
        getattr(args, "beam_worker_decision", None)
        if getattr(args, "beam_worker_decision", None) is not None
        else existing_gr_config.beam_worker_decision
    )
    validate_beam_engine_driven(getattr(args, "beam_engine_driven", None))
    beam_width_configured = (
        getattr(args, "beam_max_width", None) is not None
        or existing_gr_config.beam_max_width is not None
    )
    if device_type == "npu":
        from vllm_gr.platforms.npu import default_constraint_backend

        effective_constraint_backend = default_constraint_backend(
            effective_constraint_backend,
            explicit=constraint_backend is not None,
            constrained_generation=effective_constrained_generation,
            beam_width_configured=beam_width_configured,
            catalog_path=effective_catalog_path,
        )
    if (
        effective_beam_worker_decision is None
        and beam_width_configured
        and device_type in ("cuda", "npu")
    ):
        effective_beam_worker_decision = True
    offset, slice_size = None, None
    # CUDA Constraint Tables support sliced logits while keeping global
    # table token ids (CUDAConstrainedTopKExecution._vocab_window).  The
    # slice helper accepts both Trie lists and triples documents, so keep
    # the old lm_head fast path on CUDA. NPU retains its full-vocabulary
    # ABI and validate_constraint_table_platform rejects slicing there.
    constraint_slice_supported = (
        effective_constraint_backend is not ConstraintBackend.CONSTRAINT_TABLE
        or device_type == "cuda"
    )
    if (
        effective_constrained_generation is not False
        and effective_catalog_path is not None
        and effective_catalog_path != AUTO_CONSTRAINT_TABLE_PATH
        and constraint_slice_supported
    ):
        model_name = getattr(args, "model", None)
        if model_name is not None:
            from vllm_gr.utils.catalog import compute_catalog_slice_config

            trust_remote_code = bool(getattr(args, "trust_remote_code", False))
            offset, slice_size = compute_catalog_slice_config(
                effective_catalog_path,
                model_name,
                trust_remote_code=trust_remote_code,
            )
            if offset is not None and slice_size is not None:
                from vllm_gr.ops.sliced_logits import set_vocab_slice_config

                set_vocab_slice_config(offset, slice_size)

    return replace(
        existing_gr_config,
        catalog_path=effective_catalog_path,
        constraint_backend=effective_constraint_backend,
        attention_backend=(
            backend_override if backend_is_explicit else existing_gr_config.attention_backend
        ),
        vocab_offset=offset if offset is not None else existing_gr_config.vocab_offset,
        vocab_slice_size=(
            slice_size if slice_size is not None else existing_gr_config.vocab_slice_size
        ),
        beam_graph_enabled=(
            existing_gr_config.beam_graph_enabled and not getattr(args, "no_beam_graph", False)
        ),
        beam_max_width=(
            getattr(args, "beam_max_width", None)
            if getattr(args, "beam_max_width", None) is not None
            else existing_gr_config.beam_max_width
        ),
        beam_max_decode_steps=(
            getattr(args, "beam_max_decode_steps", None)
            if getattr(args, "beam_max_decode_steps", None) is not None
            else existing_gr_config.beam_max_decode_steps
        ),
        beam_worker_decision=effective_beam_worker_decision,
        beam_execution_mode=existing_gr_config.beam_execution_mode,
        schema_version=existing_gr_config.schema_version,
        constraint_table=existing_gr_config.constraint_table,
        constrained_generation=effective_constrained_generation,
    )


def resolve_engine_gr_config(
    args: Any,
    config: Any,
    attention_backend: tuple[bool, GRAttentionBackend | None],
    device_type: str,
) -> tuple[GRConfig, Literal["canonical", "legacy"] | None]:
    """Resolve CLI/Python inputs and report canonical or legacy write-back requirements."""
    # Parse unconditionally so invalid raw vllm_gr_* options fail before
    # the EngineConfig is returned and before runtime execution begins.
    additional_config = getattr(config, "additional_config", None)
    canonical_in_additional_config = (
        isinstance(additional_config, Mapping) and GR_CONFIG_KEY in additional_config
    )
    legacy_keys_in_additional_config = (
        sorted(key for key in KNOWN_GR_CONFIG_KEYS if key in additional_config)
        if isinstance(additional_config, Mapping)
        else []
    )
    existing_gr_config = GRConfig.from_vllm_config(config)
    # Parsing consumes the retired value, so config equality alone cannot tell
    # whether the raw additional_config sent to workers still needs cleanup.
    retired_input_present = BEAM_ENGINE_DRIVEN_KEY in legacy_keys_in_additional_config
    canonical_input = getattr(args, "_vllm_gr_config_input", None)
    catalog_path = getattr(args, "catalog_path", None)
    constraint_backend = getattr(args, "constraint_backend", None)
    constrained_generation = getattr(args, "constrained_generation", None)
    backend_is_explicit, backend_override = attention_backend

    # GRLLM historically forwards its default ConstraintBackend.TRIE value.
    # Treat that exact enum default as omitted so a canonical config can pass
    # through GRLLM's **kwargs path without changing the legacy signature.
    legacy_argument_present = (
        catalog_path is not None
        or (constraint_backend is not None and constraint_backend is not ConstraintBackend.TRIE)
        or getattr(args, "beam_max_width", None) is not None
        or getattr(args, "beam_max_decode_steps", None) is not None
        or bool(getattr(args, "no_beam_graph", False))
        or getattr(args, "beam_worker_decision", None) is not None
        or getattr(args, "beam_engine_driven", None) is not None
        or constrained_generation is not None
        or backend_is_explicit
    )

    if canonical_input is not None:
        if canonical_in_additional_config:
            raise ValueError(
                "vllm_gr_config cannot be combined with an existing canonical "
                "additional_config['vllm_gr']"
            )
        if legacy_argument_present or legacy_keys_in_additional_config:
            raise ValueError(
                "canonical vllm_gr_config cannot be combined with legacy "
                "vLLM-GR CLI, Python, or additional_config inputs"
            )
        effective_gr_config = normalize_gr_config_input(canonical_input)
        canonical_source = True
    elif canonical_in_additional_config:
        if legacy_argument_present:
            raise ValueError(
                "canonical additional_config['vllm_gr'] cannot be combined "
                "with legacy vLLM-GR CLI or Python inputs"
            )
        effective_gr_config = existing_gr_config
        canonical_source = True
    else:
        canonical_source = False
        effective_gr_config = _legacy_engine_config(
            args, existing_gr_config, backend_is_explicit, backend_override, device_type
        )
    if (
        canonical_source
        and effective_gr_config.beam_worker_decision is None
        and effective_gr_config.beam_max_width is not None
        and device_type in ("cuda", "npu")
    ):
        effective_gr_config = replace(effective_gr_config, beam_worker_decision=True)
    if (
        effective_gr_config.constrained_generation is False
        and effective_gr_config.beam_max_width is not None
        and device_type == "npu"
    ):
        from vllm_gr.platforms.npu_admission import validate_npu_unconstrained_runner

        validate_npu_unconstrained_runner(config)

    if effective_gr_config.constrained_generation is False:
        from vllm_gr.ops.sliced_logits import set_vocab_slice_config

        set_vocab_slice_config(None, None)

    execution_mode = getattr(args, "beam_execution_mode", None)
    if execution_mode is not None:
        from vllm_gr.config.validation import validate_beam_execution_mode

        effective_gr_config = replace(
            effective_gr_config, beam_execution_mode=validate_beam_execution_mode(execution_mode)
        )

    if canonical_source:
        return effective_gr_config, "canonical"
    if effective_gr_config != existing_gr_config or retired_input_present:
        return effective_gr_config, "legacy"
    return effective_gr_config, None
