"""Typing Protocols for slow-path Python callbacks.

These are documentation / type-checking aids. Native bridges accept any
callable matching the shapes below; they reacquire the GIL and force serial
execution (non-native performance).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class CiBatchTest(Protocol):
    """Batch conditional-independence test.

    Parameters
    ----------
    columns:
        List of 1-d float64 columns.
    queries:
        List of ``(x, y, z_idxs)`` where ``z_idxs`` is a list of conditioning
        column indexes.
    """

    def __call__(
        self,
        columns: Sequence[NDArray[np.float64]],
        queries: Sequence[tuple[int, int, list[int]]],
    ) -> Sequence[tuple[float, float]]:
        """Return ``(statistic, p_value)`` per query."""


@runtime_checkable
class MechanismWrapper(Protocol):
    """Per-node mechanism override for GCM sampling / abduction.

    Required: ``sample_noise``, ``evaluate``.
    Optional: ``infer_noise``, ``log_prob`` — when omitted, Rust uses an
    additive-noise default (``noise = y - f(pa, 0)``, Gaussian ``N(0,1)`` log-prob).
    """

    def sample_noise(self, n: int) -> NDArray[np.float64]:
        """Draw structural noise of length ``n``.

        Declare a second parameter named ``rng`` (``def sample_noise(self, n, rng)``) to
        receive a NumPy ``Generator`` seeded from the run's ``seed``; otherwise the draws
        come from your own source and ``seed`` does not govern them. Every callback must
        return exactly one value per row: a longer array is refused, not truncated.
        """

    def evaluate(
        self,
        parents: Sequence[NDArray[np.float64]],
        noise: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Map parents + noise → child values (length ``noise``)."""

    def infer_noise(
        self,
        value: NDArray[np.float64],
        parents: Sequence[NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """Optional: abduce noise from factual ``value`` and parents."""

    def log_prob(
        self,
        values: NDArray[np.float64],
        parents: Sequence[NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """Optional: log-density of ``values`` given parents."""


@runtime_checkable
class UtilityFn(Protocol):
    """Batch utility for decision evaluation."""

    def __call__(
        self,
        actions: NDArray[np.float64],
        outcomes: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return flat utilities of length ``len(actions) * len(outcomes)``."""


@runtime_checkable
class EffectValidator(Protocol):
    """Custom effect refuter returning a report dict."""

    def __call__(
        self,
        *,
        ate: float,
        se_analytic: float,
        method: str,
        adjustment_set: list[str],
    ) -> dict:
        """Must include ``passed: bool``; optional ``refuted_ate``, ``comparison``."""


class ProviderTrust(StrEnum):
    """Evidence status attached to a custom causal provider."""

    NATIVE_LICENSED = "native_licensed"
    VERIFIED_EXTENSION = "verified_extension"
    EXTERNALLY_ATTESTED = "externally_attested"


@dataclass(frozen=True, slots=True)
class CausalProviderSpec:
    """Machine-readable claims required before an extension can run.

    These declarations describe an extension's contract; they do not make its
    scientific claims true. Native licensing and extension verification remain
    separate gates.
    """

    query_family: str
    identification_requirements: tuple[str, ...]
    observed_distributions: tuple[str, ...]
    nuisance_functions: tuple[str, ...]
    support_conditions: tuple[str, ...]
    data_dependence: tuple[str, ...]
    inference_claims: tuple[str, ...]
    influence_function: str | None
    fold_policy: str
    output_shape: tuple[int, ...] | str
    uncertainty_semantics: str
    artifact_codec: str
    deterministic: bool
    provenance: Mapping[str, str]

    def __post_init__(self) -> None:
        for field_name in (
            "identification_requirements",
            "observed_distributions",
            "nuisance_functions",
            "support_conditions",
            "data_dependence",
            "inference_claims",
        ):
            values = tuple(getattr(self, field_name))
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"{field_name} entries must be non-empty strings")
            object.__setattr__(self, field_name, values)
        required_text = {
            "query_family": self.query_family,
            "fold_policy": self.fold_policy,
            "uncertainty_semantics": self.uncertainty_semantics,
            "artifact_codec": self.artifact_codec,
        }
        missing = [
            name
            for name, value in required_text.items()
            if not isinstance(value, str) or not value.strip()
        ]
        if missing:
            raise ValueError(f"provider spec fields must be non-empty: {', '.join(missing)}")
        for field_name in (
            "identification_requirements",
            "support_conditions",
            "data_dependence",
            "inference_claims",
        ):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} must not be empty")
        if isinstance(self.output_shape, tuple):
            if any(type(size) is not int or size < 0 for size in self.output_shape):
                raise ValueError("output_shape dimensions must be non-negative integers")
        elif not isinstance(self.output_shape, str) or not self.output_shape.strip():
            raise ValueError("output_shape must be dimensions or a non-empty shape label")
        if not self.provenance:
            raise ValueError("provenance must identify the provider")
        if any(
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(value, str)
            or not value.strip()
            for key, value in self.provenance.items()
        ):
            raise ValueError("provenance keys and values must be non-empty strings")
        if type(self.deterministic) is not bool:
            raise ValueError("deterministic must be a bool")
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


@runtime_checkable
class CausalProvider(Protocol):
    """Declarative provider contract, optionally executable by the registry.

    The EconML handoff implements this declaration without owning a learner.
    Executable providers additionally implement ``execute(request)`` and can
    be registered explicitly. Python plugins need no Rust rebuild, and
    importing a package never executes plugin discovery.
    """

    @property
    def spec(self) -> CausalProviderSpec:
        """Declared data, identification, fold, output, and inference contract."""

@dataclass(frozen=True, slots=True)
class ProviderExecution:
    """Untrusted provider output with explicit assumptions and uncertainty."""

    estimate: Any
    uncertainty: Any | None
    assumptions: tuple[str, ...]
    support_status: str
    provenance: Mapping[str, str]
    artifact: bytes | None = None

    def __post_init__(self) -> None:
        assumptions = tuple(self.assumptions)
        if any(not isinstance(item, str) or not item.strip() for item in assumptions):
            raise ValueError("provider assumptions must be non-empty strings")
        if not isinstance(self.support_status, str) or not self.support_status.strip():
            raise ValueError("provider support_status must be a non-empty string")
        provenance = dict(self.provenance)
        if not provenance or any(
            not isinstance(k, str) or not k.strip() or not isinstance(v, str) or not v.strip()
            for k, v in provenance.items()
        ):
            raise ValueError("provider result provenance must contain non-empty strings")
        object.__setattr__(self, "assumptions", assumptions)
        object.__setattr__(self, "provenance", MappingProxyType(provenance))
        if self.artifact is not None and not isinstance(self.artifact, bytes):
            raise TypeError("provider artifact must be bytes")


@dataclass(frozen=True, slots=True)
class ProviderResult:
    """Validated output envelope; extensions cannot self-assign native status."""

    estimate: NDArray[np.float64]
    uncertainty: NDArray[np.float64] | None
    assumptions: tuple[str, ...]
    support_status: str
    provenance: Mapping[str, str]
    trust: ProviderTrust
    uncertainty_semantics: str
    artifact: bytes | None = None


@dataclass(frozen=True, slots=True)
class ProviderQuery:
    """Explicit request to run a registered provider through ``analyze``.

    The ordinary ``analyze(data, query=...)`` data argument supplies the
    caller's dataset. ``request`` contains only provider-specific controls.
    This route deliberately does not translate an arbitrary provider output
    into a native causal estimand.
    """

    provider: str
    request: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("ProviderQuery.provider must be a non-empty registered name")
        if not isinstance(self.request, Mapping):
            raise TypeError("ProviderQuery.request must be a mapping")
        object.__setattr__(self, "request", MappingProxyType(dict(self.request)))


class ProviderRegistry:
    """Small explicit registry for separately installed Python providers."""

    def __init__(self) -> None:
        self._providers: dict[str, tuple[CausalProvider, ProviderTrust]] = {}

    def register(
        self,
        name: str,
        provider: CausalProvider,
        *,
        trust: ProviderTrust = ProviderTrust.EXTERNALLY_ATTESTED,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("provider name must be a non-empty string")
        if name in self._providers:
            raise ValueError(f"provider {name!r} is already registered")
        spec = getattr(provider, "spec", None)
        if not isinstance(spec, CausalProviderSpec):
            raise TypeError("provider.spec must be a CausalProviderSpec")
        if not callable(getattr(provider, "execute", None)):
            raise TypeError("provider must implement execute(request)")
        if trust is ProviderTrust.NATIVE_LICENSED:
            raise ValueError("Python extensions cannot register as native licensed providers")
        if trust is not ProviderTrust.EXTERNALLY_ATTESTED:
            raise ValueError(
                "Python provider registration defaults to externally attested; "
                "verified and native status require the separate evidence gate"
            )
        self._providers[name] = (provider, trust)

    def get(self, name: str) -> CausalProvider:
        try:
            return self._providers[name][0]
        except KeyError as exc:
            raise KeyError(f"provider {name!r} is not explicitly registered") from exc

    def execute(self, name: str, request: Mapping[str, Any]) -> ProviderResult:
        try:
            provider, trust = self._providers[name]
        except KeyError as exc:
            raise KeyError(f"provider {name!r} is not explicitly registered") from exc
        spec = provider.spec
        raw = provider.execute(MappingProxyType(dict(request)))
        if not isinstance(raw, ProviderExecution):
            raise TypeError("provider execute() must return ProviderExecution")
        estimate = _provider_array(raw.estimate, "estimate")
        if isinstance(spec.output_shape, tuple) and estimate.shape != spec.output_shape:
            raise ValueError(
                f"provider estimate shape {estimate.shape} does not match declared {spec.output_shape}"
            )
        uncertainty = (
            None if raw.uncertainty is None else _provider_array(raw.uncertainty, "uncertainty")
        )
        if uncertainty is not None and uncertainty.shape not in ((), estimate.shape):
            raise ValueError("provider uncertainty must be scalar or match the estimate shape")
        provenance = dict(spec.provenance)
        provenance.update(raw.provenance)
        provenance["registry_name"] = name
        provenance["trust_boundary"] = trust.value
        return ProviderResult(
            estimate,
            uncertainty,
            raw.assumptions,
            raw.support_status,
            MappingProxyType(provenance),
            trust,
            spec.uncertainty_semantics,
            raw.artifact,
        )


def _provider_array(value: Any, label: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"provider {label} must contain only finite values")
    array = np.array(array, dtype=np.float64, copy=True)
    array.flags.writeable = False
    return array


providers = ProviderRegistry()


__all__ = [
    "CiBatchTest",
    "CausalProvider",
    "CausalProviderSpec",
    "EffectValidator",
    "MechanismWrapper",
    "ProviderTrust",
    "ProviderExecution",
    "ProviderRegistry",
    "ProviderResult",
    "ProviderQuery",
    "UtilityFn",
    "providers",
]
