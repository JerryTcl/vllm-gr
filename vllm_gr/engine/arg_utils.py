# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
import argparse
import inspect
import json
from collections.abc import Mapping
from typing import Final

from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_gr.config import (
    BeamExecutionMode,
    ConstraintBackend,
    GRAttentionBackend,
    GRConfigInput,
    write_canonical_gr_config,
)
from vllm_gr.config.parser import resolve_engine_gr_config
from vllm_gr.config.validation import validate_online_batch_capacity
from vllm_gr.platforms.npu import apply_default_block_size

origin_add_cli_args = EngineArgs.add_cli_args
origin_create_engine_config = EngineArgs.create_engine_config
origin_from_cli_args = EngineArgs.from_cli_args.__func__
origin_engine_args_init = EngineArgs.__init__
_NESTED_ATTENTION_BACKEND_EXPLICIT_ATTR: Final = "_vllm_gr_nested_attention_backend_explicit"
VLLM_GR_ENGINE_DEFAULTS: Final[dict[str, object]] = {
    "max_model_len": 8192,
    "max_num_batched_tokens": 8192,
    "enable_chunked_prefill": False,
    "enable_prefix_caching": True,
}
_ORIGIN_ENGINE_ARG_POSITIONAL_NAMES: Final = tuple(
    list(inspect.signature(origin_engine_args_init).parameters)[1:]
)


def _apply_vllm_gr_engine_defaults(args: tuple[object, ...], kwargs: dict[str, object]) -> None:
    """Fill vllm-gr engine defaults without replacing explicit arguments."""
    explicit_values = dict(zip(_ORIGIN_ENGINE_ARG_POSITIONAL_NAMES, args))
    explicit_values.update(kwargs)
    explicitly_set = set(explicit_values)
    for name, value in VLLM_GR_ENGINE_DEFAULTS.items():
        if name not in explicitly_set:
            kwargs[name] = value

    # Disabling chunked prefill makes vLLM require one scheduling batch to
    # accommodate the full model context. Keep the 8K default for ordinary
    # vLLM-GR construction, but follow an explicitly enlarged model length
    # when the caller did not independently choose a batched-token limit.
    # Explicit (including intentionally invalid) pairs remain upstream's
    # responsibility to validate.
    max_model_len = explicit_values.get("max_model_len", kwargs.get("max_model_len"))
    enable_chunked_prefill = explicit_values.get(
        "enable_chunked_prefill", kwargs.get("enable_chunked_prefill")
    )
    if (
        "max_num_batched_tokens" not in explicitly_set
        and enable_chunked_prefill is False
        and isinstance(max_model_len, int)
        and not isinstance(max_model_len, bool)
    ):
        default_batched_tokens = kwargs.get("max_num_batched_tokens")
        if isinstance(default_batched_tokens, int):
            kwargs["max_num_batched_tokens"] = max(default_batched_tokens, max_model_len)


def _init_engine_args(
    self: EngineArgs,
    *args: object,
    catalog_path: str | None = None,
    constraint_backend: ConstraintBackend | str | None = None,
    constrained_generation: bool | None = None,
    vllm_gr_config: GRConfigInput | None = None,
    beam_execution_mode: str | None = None,
    **kwargs: object,
) -> None:
    """Apply vllm-gr defaults and accept GR-only Python API arguments.

    The CLI path (``--catalog-path`` / ``--constraint-backend``) stashes these
    values as attributes in :func:`from_cli_args` after construction. Shared
    engine defaults are filled here only when callers did not pass
    those arguments explicitly. The offline ``GRLLM`` Python API builds its
    ``EngineArgs`` indirectly through
    ``LLM.__init__`` -> ``EngineArgs(model=..., **kwargs)``, so it cannot reach
    the instance to set the attributes afterwards.  Threading the two values as
    ``**kwargs`` here stashes them identically, and the patched
    :func:`create_engine_config` reads them exactly like the CLI path —
    otherwise ``EngineCore`` would not learn that the backend is
    ``constraint_table`` and would never run the constrained hooks.
    """
    _apply_vllm_gr_engine_defaults(args, kwargs)
    origin_engine_args_init(self, *args, **kwargs)
    self.catalog_path = catalog_path
    self.constraint_backend = constraint_backend
    self.constrained_generation = constrained_generation
    self._vllm_gr_config_input = vllm_gr_config
    setattr(self, "beam_execution_mode", beam_execution_mode)


def _nested_attention_backend_is_explicit(value: object) -> bool:
    if isinstance(value, Mapping):
        return "backend" in value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return False
        return isinstance(decoded, Mapping) and "backend" in decoded
    return bool(getattr(value, _NESTED_ATTENTION_BACKEND_EXPLICIT_ATTR, False))


def _preserve_nested_attention_backend_source(
    parser: FlexibleArgumentParser,
) -> None:
    """Keep source information that upstream AttentionConfig normalizes away."""
    action = next(
        (action for action in parser._actions if action.dest == "attention_config"),
        None,
    )
    if action is None or not callable(action.type):
        return

    parse_attention_config = action.type

    def parse_with_source(value: str) -> object:
        parsed = parse_attention_config(value)
        setattr(
            parsed,
            _NESTED_ATTENTION_BACKEND_EXPLICIT_ATTR,
            _nested_attention_backend_is_explicit(value),
        )
        return parsed

    action.type = parse_with_source


def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
    parser = origin_add_cli_args(parser)
    parser.set_defaults(**VLLM_GR_ENGINE_DEFAULTS)
    _preserve_nested_attention_backend_source(parser)
    model_group = parser.add_argument_group(title="vLLM-GR")
    model_group.add_argument(
        "--catalog-path",
        type=str,
        default=None,
        help="Optional JSON catalog used to constrain generation results.",
    )
    model_group.add_argument(
        "--constraint-backend",
        type=str,
        choices=[backend.value for backend in ConstraintBackend],
        default=None,
        help="Constraint implementation: trie (default) or constraint_table.",
    )
    model_group.add_argument(
        "--constrained-generation",
        action="store_true",
        default=None,
        help="Enable catalog-constrained Beam candidate generation.",
    )
    model_group.add_argument(
        "--no-constrained-generation",
        action="store_false",
        dest="constrained_generation",
        default=None,
        help="Disable catalog constraints and use full-vocabulary Beam candidates.",
    )
    model_group.add_argument(
        "--vllm-gr-config",
        type=str,
        default=None,
        help="Path to a schema-versioned vLLM-GR YAML config.",
    )
    model_group.add_argument(
        "--beam-execution-mode",
        choices=[mode.value for mode in BeamExecutionMode],
        default=None,
        help="Enable GR Beam Search V1 for this engine (requires async scheduling). "
        "Unset leaves GR Beam disabled; the legacy interface has been removed.",
    )
    model_group.add_argument(
        "--beam-max-width",
        type=int,
        default=None,
        help=(
            "Upper bound on beam width for the persistent beam KV pool and "
            "the captured CUDA graph shape (num_tokens=width). Must be >= "
            "any served request's beam_width and <= 1024. Widths above 256 need an "
            "explicit constraint_table.candidate_width, because K follows the Beam "
            "width when it is unset and the operators cap K at 256. "
            "When set, the engine allocates "
            "the beam KV pool and (unless --no-beam-graph) captures a FULL "
            "CUDA graph for this width at startup. Unset (default) disables "
            "beam decode — no pool allocated, no graph captured."
        ),
    )
    model_group.add_argument(
        "--no-beam-graph",
        action="store_true",
        default=False,
        help="Disable the beam-decode CUDA graph (beam decode runs eager).",
    )
    model_group.add_argument(
        "--beam-max-decode-steps",
        type=int,
        default=None,
        help="Upper bound on beam decode steps for the persistent beam KV "
        "pool (default 3, matching OneRec SID: max_tokens=5 minus begin/end).",
    )
    model_group.add_argument(
        "--beam-worker-decision",
        action="store_true",
        default=None,
        help="Move the beam decision (beam_search_group / rec_final_select + "
        "select_unshared_kv) into the accelerator worker and return results via "
        "ModelRunnerOutput.beam_search_result (asynchronous result flow). CUDA "
        "beam deployments enable this by default; NPU requires xllm decode "
        "execution and its operator library. Explicit False does not restore "
        "the retired frontend decision path.",
    )
    model_group.add_argument(
        "--no-beam-worker-decision",
        action="store_false",
        dest="beam_worker_decision",
        default=None,
        help="Explicitly disable worker decisions; this does not enable legacy Beam.",
    )
    model_group.add_argument(
        "--beam-engine-driven",
        action="store_true",
        default=None,
        help="Retired compatibility option. True is accepted with a warning and "
        "ignored; Beam V1 always owns scheduling.",
    )
    model_group.add_argument(
        "--no-beam-engine-driven",
        action="store_false",
        dest="beam_engine_driven",
        default=None,
        help="Retired frontend step-loop option; engine configuration rejects it.",
    )
    return parser


def from_cli_args(cls, args: argparse.Namespace) -> EngineArgs:
    raw_attention_config = getattr(args, "attention_config", None)
    nested_backend_is_explicit = _nested_attention_backend_is_explicit(raw_attention_config)
    engine_args = origin_from_cli_args(cls, args)
    engine_args.catalog_path = None
    if hasattr(args, "catalog_path"):
        engine_args.catalog_path = args.catalog_path
    engine_args.constraint_backend = None
    if hasattr(args, "constraint_backend"):
        engine_args.constraint_backend = args.constraint_backend
    engine_args.constrained_generation = getattr(args, "constrained_generation", None)
    engine_args._vllm_gr_config_input = getattr(args, "vllm_gr_config", None)
    for attr in ("beam_max_width", "beam_max_decode_steps"):
        setattr(engine_args, attr, getattr(args, attr, None))
    engine_args.no_beam_graph = getattr(args, "no_beam_graph", False)
    engine_args.beam_worker_decision = getattr(args, "beam_worker_decision", None)
    engine_args.beam_engine_driven = getattr(args, "beam_engine_driven", None)
    engine_args.beam_execution_mode = getattr(args, "beam_execution_mode", None)
    # AttentionConfig normalizes explicit "auto" and JSON null to None. Keep
    # the CLI source information so both values can override a stale CUSTOM
    # marker rather than becoming indistinguishable from an omitted backend.
    setattr(
        engine_args,
        _NESTED_ATTENTION_BACKEND_EXPLICIT_ATTR,
        nested_backend_is_explicit,
    )
    return engine_args


def _explicit_gr_attention_backend(
    self,
) -> tuple[bool, GRAttentionBackend | None]:
    """Return whether a backend was explicit and its GR representation.

    ``None`` with ``explicit=True`` means the user selected a non-GR backend,
    which intentionally clears a stale GR CUSTOM marker.
    """
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    requested_backend = getattr(self, "attention_backend", None)
    if requested_backend is not None:
        is_custom = (
            isinstance(requested_backend, str) and requested_backend.upper() == "CUSTOM"
        ) or requested_backend == AttentionBackendEnum.CUSTOM
        return True, GRAttentionBackend.CUSTOM if is_custom else None

    attention_config = getattr(self, "attention_config", None)
    configured_backend = getattr(attention_config, "backend", None)
    nested_backend_is_explicit = getattr(
        self,
        _NESTED_ATTENTION_BACKEND_EXPLICIT_ATTR,
        False,
    )
    if nested_backend_is_explicit or configured_backend is not None:
        return (
            True,
            GRAttentionBackend.CUSTOM
            if configured_backend == AttentionBackendEnum.CUSTOM
            else None,
        )
    return False, None


def _is_npu_platform() -> bool:
    """Return whether the active platform is an Ascend NPU."""
    try:
        from vllm.platforms import current_platform
    except Exception:
        return False
    return getattr(current_platform, "device_type", None) == "npu"


def _is_cuda_platform() -> bool:
    """Return whether the active platform is CUDA."""
    try:
        from vllm.platforms import current_platform
    except Exception:
        return False
    return getattr(current_platform, "device_type", None) == "cuda"


def create_engine_config(self, usage_context=None, headless=False):
    config = origin_create_engine_config(self, usage_context, headless)
    apply_default_block_size(config)
    device_type = "npu" if _is_npu_platform() else "cuda" if _is_cuda_platform() else ""
    effective_gr_config, config_format = resolve_engine_gr_config(
        self, config, _explicit_gr_attention_backend(self), device_type
    )
    if device_type == "cuda":
        from vllm_gr.platforms.cuda import validate_beam_runner

        validate_beam_runner(config, effective_gr_config)
    max_num_seqs = getattr(getattr(config, "scheduler_config", None), "max_num_seqs", None)
    validate_online_batch_capacity(effective_gr_config, max_num_seqs)
    if config_format == "canonical":
        # vLLM hashes the raw additional_config for compilation. Normalize both
        # canonical input paths even when their parsed values are unchanged, so
        # omitted defaults cannot give equivalent engines different AOT caches.
        write_canonical_gr_config(config, effective_gr_config)
    elif config_format == "legacy":
        effective_gr_config.write_to_vllm_config(config)

    if effective_gr_config.attention_backend is GRAttentionBackend.CUSTOM:
        attention_config = getattr(config, "attention_config", None)
        if attention_config is not None and getattr(attention_config, "backend", None) is None:
            from vllm.v1.attention.backends.registry import AttentionBackendEnum

            attention_config.backend = AttentionBackendEnum.CUSTOM
    if effective_gr_config.beam_execution_mode is BeamExecutionMode.V1:
        from vllm_gr.config.execution import validate_gr_execution_config

        validate_gr_execution_config(config)
        if device_type == "cuda":
            from vllm_gr.platforms.cuda import apply_beam_compilation_defaults

            apply_beam_compilation_defaults(config)

    # On Ascend, vllm-ascend wraps EngineCoreProc.run_engine_core and validates
    # additional_config (pydantic extra="forbid") in the spawned EngineCore
    # process *before* the vllm-gr plugin loads. Install the GR-key sanitizer
    # here -- after origin_create_engine_config and every general plugin, and
    # before EngineCoreClient spawns the process -- so it is the outermost
    # entrypoint and filters GR keys during vllm-ascend's init_ascend_config.
    if device_type == "npu":
        from vllm.v1.engine.core import EngineCoreProc

        from vllm_gr.v1.ascend_compat import apply_engine_core_config_sanitizer

        apply_engine_core_config_sanitizer(EngineCoreProc)

    return config


def patch_add_cli_args() -> None:
    EngineArgs.add_cli_args = staticmethod(add_cli_args)
    EngineArgs.create_engine_config = create_engine_config
    EngineArgs.from_cli_args = classmethod(from_cli_args)
    EngineArgs.__init__ = _init_engine_args
