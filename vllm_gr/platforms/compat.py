# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Declared compatibility ranges checked before a provider is activated."""

from __future__ import annotations

import logging
import os

from .errors import InvalidBackendProviderError
from .protocol import GRBackendProvider

logger = logging.getLogger("vllm_gr.platforms.compat")

#: When set to a truthy value, an unsatisfied declared range is a hard failure.
#: The default only warns: whether a declared range still holds is a vendor
#: compatibility decision, and failing hard by default would refuse startup on
#: any patch-level runtime bump.
_STRICT_ENV = "VLLM_GR_STRICT_PROVIDER_RANGES"
_TRUTHY = ("1", "true", "yes", "on")


def strict_ranges_enabled() -> bool:
    """Whether declared range mismatches must fail startup instead of warn."""
    return os.environ.get(_STRICT_ENV, "").strip().lower() in _TRUTHY


def _installed_version(distribution: str) -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - stdlib always provides it
        return None
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _specifier_allows(specifier: str, version: str) -> bool:
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    try:
        return Version(version) in SpecifierSet(specifier)
    except Exception:
        # An unparsable declared range is a provider defect, not a runtime one.
        raise InvalidBackendProviderError(
            f"Provider declared an unparsable version range {specifier!r}"
        ) from None


def _check_range(
    *,
    label: str,
    declared: str | None,
    distribution: str,
    provider: GRBackendProvider,
    failures: list[str],
) -> None:
    if declared is None:
        return
    installed = _installed_version(distribution)
    if installed is None:
        failures.append(
            f"{label} distribution {distribution!r} is not installed "
            f"(provider {provider.backend_id!r} declares {declared!r})"
        )
        return
    if not _specifier_allows(declared, installed):
        failures.append(
            f"{label} version {installed} is outside the range {declared!r} "
            f"declared by provider {provider.backend_id!r}"
        )


def check_declared_ranges(provider: GRBackendProvider) -> tuple[str, ...]:
    """Return every declared-range violation for one provider.

    Empty means all declared ranges hold. ``None`` declares no constraint.
    """
    failures: list[str] = []
    gr_version = _installed_version("vllm-gr") or _source_tree_version()

    declared_gr = provider.gr_version_range
    if declared_gr is not None and gr_version is not None:
        if not _specifier_allows(declared_gr, gr_version):
            failures.append(
                f"vllm-gr version {gr_version} is outside the range "
                f"{declared_gr!r} declared by provider {provider.backend_id!r}"
            )

    _check_range(
        label="vLLM",
        declared=provider.vllm_version_range,
        distribution="vllm",
        provider=provider,
        failures=failures,
    )
    if provider.vendor_version_range is not None:
        _check_range(
            label="vendor runtime",
            declared=provider.vendor_version_range,
            distribution=provider.vendor_distribution,
            provider=provider,
            failures=failures,
        )
    return tuple(failures)


def _source_tree_version() -> str | None:
    try:
        from vllm_gr.version import __version__
    except Exception:  # pragma: no cover - import-time diagnostics only
        return None
    return __version__


def enforce_declared_ranges(provider: GRBackendProvider) -> None:
    """Raise on violations when strict mode is enabled; otherwise warn once."""
    failures = check_declared_ranges(provider)
    if not failures:
        return
    message = "; ".join(failures)
    if strict_ranges_enabled():
        raise InvalidBackendProviderError(
            f"{_STRICT_ENV} is set and a declared compatibility range failed: {message}"
        )
    logger.warning(
        "vllm-gr backend %r declares a compatibility range that does not hold: %s. "
        "Set %s=1 to make this a startup failure.",
        provider.backend_id,
        message,
        _STRICT_ENV,
    )
