"""Typing Protocols for slow-path Python callbacks.

These are documentation / type-checking aids. Native bridges accept any
callable matching the shapes below; they reacquire the GIL and force serial
execution (non-native performance).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from importlib import metadata
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
    host_artifact: bytes | None = None


@dataclass(frozen=True, slots=True)
class ProviderVerificationFixture:
    """Caller supplied, independently specified reference for one provider run.

    ``artifact_digest`` checks the encoded bytes. A provider-specific artifact
    decoder can additionally be supplied to check that those bytes round trip
    to the expected scientific payload; it must not be the provider under test.
    """

    name: str
    request: Mapping[str, Any]
    expected_estimate: Any
    expected_uncertainty: Any | None
    expected_assumptions: tuple[str, ...]
    expected_support_status: str
    expected_provenance: Mapping[str, str]
    artifact_digest: str | None = None
    artifact_decoder: Any | None = None
    expected_decoded_artifact: Any | None = None
    atol: float = 0.0

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("verification fixture needs a name")
        if not isinstance(self.request, Mapping):
            raise TypeError("verification request must be a mapping")
        if self.atol < 0 or not np.isfinite(self.atol):
            raise ValueError("verification atol must be finite and non-negative")
        if self.artifact_decoder is not None and not callable(self.artifact_decoder):
            raise TypeError("artifact_decoder must be callable")
        if self.artifact_decoder is not None and self.artifact_digest is None:
            raise ValueError("artifact_decoder requires an expected artifact_digest")
        object.__setattr__(self, "request", MappingProxyType(dict(self.request)))
        object.__setattr__(
            self, "expected_provenance", MappingProxyType(dict(self.expected_provenance))
        )


@dataclass(frozen=True, slots=True)
class ProviderVerificationReport:
    """Host recorded evidence for a bounded set of external fixtures."""

    provider: str
    fixture_names: tuple[str, ...]
    spec_digest: str
    evidence_digest: str
    deterministic_replay_checked: bool
    evidence_origin: str
    verified_request_digests: tuple[str, ...]


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
        self._entry_points: dict[str, str] = {}
        self._verification: dict[str, ProviderVerificationReport] = {}

    def load_entry_point(self, name: str) -> CausalProvider:
        """Explicitly load one installed ``antecedent.providers`` entry point.

        An entry point must expose a zero-argument factory returning an
        executable provider. Discovery and provider code run only when this
        method is called. Installing a separate Python distribution therefore
        needs no Antecedent or Rust rebuild. Every loaded provider remains
        externally attested under the same validation as ``register``.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("provider name must be a non-empty string")
        if name in self._providers:
            raise ValueError(f"provider {name!r} is already registered")
        matches = tuple(metadata.entry_points().select(group="antecedent.providers", name=name))
        if not matches:
            raise KeyError(f"provider entry point {name!r} is not installed")
        if len(matches) != 1:
            raise ValueError(f"provider entry point {name!r} is ambiguous")
        entry_point = matches[0]
        factory = entry_point.load()
        if not callable(factory):
            raise TypeError(f"provider entry point {name!r} must expose a factory")
        provider = factory()
        self.register(name, provider)
        self._entry_points[name] = entry_point.value
        return provider

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

    def verification_report(self, name: str) -> ProviderVerificationReport:
        """Return the host's evidence receipt for a verified extension."""
        try:
            return self._verification[name]
        except KeyError as exc:
            raise KeyError(f"provider {name!r} has no verification report") from exc

    def verify(
        self,
        name: str,
        fixtures: Sequence[ProviderVerificationFixture],
        *,
        evidence_origin: str,
    ) -> ProviderVerificationReport:
        """Run independent caller fixtures before promoting an external provider.

        This checks the executable contract and supplied references; it does
        not certify the scientific validity of caller supplied truth or grant
        native licensing. Promotion is atomic after every fixture passes.
        """
        provider, trust = self._providers[name]
        if trust is not ProviderTrust.EXTERNALLY_ATTESTED:
            raise ValueError("only externally attested providers can be verified")
        if not isinstance(evidence_origin, str) or not evidence_origin.strip():
            raise ValueError("evidence_origin must identify the independent fixture source")
        fixtures = tuple(fixtures)
        if any(not isinstance(fixture, ProviderVerificationFixture) for fixture in fixtures):
            raise TypeError("verification fixtures must be ProviderVerificationFixture")
        if not fixtures or len({fixture.name for fixture in fixtures}) != len(fixtures):
            raise ValueError("verification requires distinct, non-empty fixtures")
        spec = provider.spec
        spec_digest = _digest(_spec_evidence(spec))
        observations = []
        request_digests = []
        for fixture in fixtures:
            result = self.execute(name, fixture.request)
            expected = _provider_array(fixture.expected_estimate, "expected estimate")
            if result.estimate.shape != expected.shape or not np.allclose(
                result.estimate, expected, atol=fixture.atol, rtol=0
            ):
                raise ValueError(f"fixture {fixture.name!r}: estimate differs from reference")
            if fixture.expected_uncertainty is None:
                if result.uncertainty is not None:
                    raise ValueError(f"fixture {fixture.name!r}: unexpected uncertainty")
            else:
                expected_uncertainty = _provider_array(
                    fixture.expected_uncertainty, "expected uncertainty"
                )
                if (
                    result.uncertainty is None
                    or result.uncertainty.shape != expected_uncertainty.shape
                    or not np.allclose(
                        result.uncertainty, expected_uncertainty, atol=fixture.atol, rtol=0
                    )
                ):
                    raise ValueError(
                        f"fixture {fixture.name!r}: uncertainty differs from reference"
                    )
            if result.uncertainty is not None and spec.uncertainty_semantics == "point_only":
                raise ValueError(
                    f"fixture {fixture.name!r}: point-only provider returned uncertainty"
                )
            if result.assumptions != tuple(fixture.expected_assumptions):
                raise ValueError(f"fixture {fixture.name!r}: assumptions differ from reference")
            if result.support_status != fixture.expected_support_status:
                raise ValueError(f"fixture {fixture.name!r}: support differs from reference")
            for key, value in fixture.expected_provenance.items():
                if result.provenance.get(key) != value:
                    raise ValueError(f"fixture {fixture.name!r}: provenance {key!r} differs")
            if fixture.artifact_digest is not None:
                if (
                    result.artifact is None
                    or hashlib.sha256(result.artifact).hexdigest() != fixture.artifact_digest
                ):
                    raise ValueError(f"fixture {fixture.name!r}: artifact digest differs")
            elif result.artifact is not None:
                raise ValueError(f"fixture {fixture.name!r}: unverified artifact")
            if fixture.artifact_decoder is not None:
                decoded = fixture.artifact_decoder(result.artifact)
                if decoded != fixture.expected_decoded_artifact:
                    raise ValueError(f"fixture {fixture.name!r}: artifact round trip differs")
            if spec.deterministic:
                replay = self.execute(name, fixture.request)
                if not _same_provider_result(result, replay):
                    raise ValueError(f"fixture {fixture.name!r}: deterministic replay differs")
            observations.append(
                {
                    "name": fixture.name,
                    "request": fixture.request,
                    "expected_estimate": expected,
                    "expected_uncertainty": fixture.expected_uncertainty,
                    "atol": fixture.atol,
                    "result_estimate": result.estimate,
                    "result_uncertainty": result.uncertainty,
                    "assumptions": result.assumptions,
                    "support_status": result.support_status,
                    "provenance": dict(result.provenance),
                    "artifact_digest": None
                    if result.artifact is None
                    else hashlib.sha256(result.artifact).hexdigest(),
                    "artifact_round_trip_checked": fixture.artifact_decoder is not None,
                }
            )
            request_digests.append(_digest(fixture.request))
        if _digest(_spec_evidence(provider.spec)) != spec_digest:
            raise ValueError("provider spec changed during verification")
        report = ProviderVerificationReport(
            name,
            tuple(fixture.name for fixture in fixtures),
            spec_digest,
            _digest(
                {
                    "spec_digest": spec_digest,
                    "evidence_origin": evidence_origin,
                    "observations": observations,
                }
            ),
            spec.deterministic,
            evidence_origin,
            tuple(request_digests),
        )
        self._verification[name] = report
        self._providers[name] = (provider, ProviderTrust.VERIFIED_EXTENSION)
        return report

    def execute(self, name: str, request: Mapping[str, Any]) -> ProviderResult:
        try:
            provider, trust = self._providers[name]
        except KeyError as exc:
            raise KeyError(f"provider {name!r} is not explicitly registered") from exc
        spec = provider.spec
        report = self._verification.get(name)
        if report is not None and _digest(_spec_evidence(spec)) != report.spec_digest:
            raise ValueError("verified provider spec changed after verification")
        effective_trust = trust
        if report is not None:
            try:
                covered_request = _digest(request) in report.verified_request_digests
            except TypeError:
                covered_request = False
            if not covered_request:
                effective_trust = ProviderTrust.EXTERNALLY_ATTESTED
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
        if uncertainty is not None and spec.uncertainty_semantics == "point_only":
            raise ValueError("point-only provider returned uncertainty")
        provenance = dict(spec.provenance)
        provenance.update(raw.provenance)
        provenance["registry_name"] = name
        if name in self._entry_points:
            provenance["entry_point"] = self._entry_points[name]
        provenance["trust_boundary"] = effective_trust.value
        if report is not None and effective_trust is ProviderTrust.VERIFIED_EXTENSION:
            provenance["verification_evidence_digest"] = report.evidence_digest
            provenance["verification_evidence_origin"] = report.evidence_origin
        try:
            request_digest = _digest(request)
        except TypeError:
            if effective_trust is ProviderTrust.VERIFIED_EXTENSION:
                raise ValueError("verified request must have a canonical digest") from None
            request_digest = hashlib.sha256(repr(dict(request)).encode("utf-8")).hexdigest()
            provenance["request_digest_method"] = "repr_fallback_external_only"
        from . import _native

        header = {
            "version": 1,
            "provider": name,
            "query_family": spec.query_family,
            "trust": effective_trust.value,
            "spec_digest": _digest(_spec_evidence(spec)),
            "request_digest": request_digest,
            "verification_evidence_digest": (
                report.evidence_digest
                if effective_trust is ProviderTrust.VERIFIED_EXTENSION and report is not None
                else None
            ),
            "estimate": estimate.reshape(-1).tolist(),
            "shape": list(estimate.shape),
            "uncertainty": None if uncertainty is None else uncertainty.reshape(-1).tolist(),
            "uncertainty_shape": None if uncertainty is None else list(uncertainty.shape),
            "uncertainty_semantics": spec.uncertainty_semantics,
            "assumptions": list(raw.assumptions),
            "provider_support_status": raw.support_status,
            "provenance": provenance,
            "external_artifact_digest": None,
        }
        host_artifact = _native.seal_provider_result(
            json.dumps(header, sort_keys=True, allow_nan=False), raw.artifact
        )
        return ProviderResult(
            estimate,
            uncertainty,
            raw.assumptions,
            raw.support_status,
            MappingProxyType(provenance),
            effective_trust,
            spec.uncertainty_semantics,
            raw.artifact,
            host_artifact,
        )


def _provider_array(value: Any, label: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"provider {label} must contain only finite values")
    array = np.array(array, dtype=np.float64, copy=True)
    array.flags.writeable = False
    return array


def _spec_evidence(spec: CausalProviderSpec) -> dict[str, Any]:
    return {
        field: dict(value) if field == "provenance" else value
        for field, value in ((name, getattr(spec, name)) for name in spec.__dataclass_fields__)
    }


def _digest(value: Any) -> str:
    def encode(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Mapping):
            return {str(key): encode(value) for key, value in item.items()}
        if isinstance(item, (tuple, list)):
            return [encode(value) for value in item]
        if isinstance(item, (str, int, float, bool)) or item is None:
            return item
        raise TypeError("verification evidence must be JSON serializable")

    payload = json.dumps(encode(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _same_provider_result(a: ProviderResult, b: ProviderResult) -> bool:
    return (
        np.array_equal(a.estimate, b.estimate)
        and (
            (a.uncertainty is None and b.uncertainty is None)
            or (
                a.uncertainty is not None
                and b.uncertainty is not None
                and np.array_equal(a.uncertainty, b.uncertainty)
            )
        )
        and a.assumptions == b.assumptions
        and a.support_status == b.support_status
        and dict(a.provenance) == dict(b.provenance)
        and a.artifact == b.artifact
    )


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
    "ProviderVerificationFixture",
    "ProviderVerificationReport",
    "UtilityFn",
    "providers",
]
