"""Typing Protocols for slow-path Python callbacks.

These are documentation / type-checking aids. Native bridges accept any
callable matching the shapes below; they reacquire the GIL and force serial
execution (non-native performance).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from importlib import metadata
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, cast, get_args, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ._artifact_protocol import (
    ExportedArtifact,
    artifact_export,
    artifact_from_receipt,
    container_bytes,
)
from ._columns import coerce_fields
from ._digest import canonical_digest
from .errors import CausalValueError, StructuredRefusal

if TYPE_CHECKING:
    from .external import ProviderObject

# Provenance keys only the registry writes: they record which registry entry ran,
# how it was trusted, and which verification evidence backs it. A provider's own
# values for them are dropped, so it cannot forge its trust record.
_HOST_PROVENANCE_KEYS = frozenset(
    {
        "registry_name",
        "entry_point",
        "trust_boundary",
        "verification_evidence_digest",
        "verification_evidence_origin",
        "request_digest_method",
    }
)


def _provider_provenance(provenance: Mapping[str, str]) -> dict[str, str]:
    return {k: v for k, v in provenance.items() if k not in _HOST_PROVENANCE_KEYS}


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
    A mechanism whose noise cannot be recovered from an observation declares
    ``supports_noise_inference = False``: counterfactual abduction is then refused naming the
    mechanism and the operation, while sampling and likelihood remain available.
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


class ProviderRefusal(StructuredRefusal):
    """A refusal by the provider registry or the adjustment handoff.

    It carries the same structured fields as :class:`~antecedent.external.ExternalRefusal`:
    :attr:`reason_code` is a registered refusal code, :attr:`detail` the namespaced
    ``family.slot`` (``provider.*`` or ``handoff.*``), :attr:`offending` what it is
    about and :attr:`remedy` what the caller can change to proceed. It is a
    :class:`~antecedent.errors.CausalUnsupportedError`, as an identification refusal
    always was.
    """


class ProviderValueRefusal(ProviderRefusal, CausalValueError):
    """A :class:`ProviderRefusal` that is also a :class:`ValueError`.

    The refusals that were a ``ValueError`` before they gained a reason code (an
    argument, a fixture that differs from its reference, a provider output that
    breaks its declared contract) keep that class whatever their reason code is,
    so an ``except ValueError`` written against them still catches.
    """


class UnknownProviderError(ProviderRefusal, KeyError):
    """A provider name that is not registered or installed; also a :class:`KeyError`."""

    __str__ = Exception.__str__


def _refuse(
    detail: str,
    message: str,
    *,
    code: str = "invalid_argument",
    remedy: str,
    offending: str | None = None,
    stage: str = "provider",
    error: type[ProviderRefusal] = ProviderValueRefusal,
) -> ProviderRefusal:
    """A typed refusal; ``raise`` it. ``code`` must be a registered refusal code.

    A :class:`ProviderValueRefusal` (a ``ValueError`` too) unless ``error`` names another class.
    """
    return error(
        {
            "code": code,
            "stage": stage,
            "detail": detail,
            "message": message,
            "offending": offending,
            "remedy": remedy,
        },
        text=message,
    )


class ProviderTrust(StrEnum):
    """Evidence status attached to a custom causal provider.

    The members are exactly the Rust ``ProviderTrust`` levels and ``value`` is
    their canonical wire (``as_wire``). Read-compat aliases parse to the same
    member: ``external_attested`` for :attr:`EXTERNALLY_ATTESTED`;
    ``exact_request_verified`` and ``external_exact_request_verified`` for
    :attr:`VERIFIED_EXTENSION`. :attr:`UNVERIFIED` is what a value carries when
    nobody has claimed anything for it; a provider is never registered with it.
    """

    NATIVE_LICENSED = "native_licensed"
    VERIFIED_EXTENSION = "verified_extension"
    EXTERNALLY_ATTESTED = "externally_attested"
    UNVERIFIED = "unverified"

    @classmethod
    def _missing_(cls, value: object) -> ProviderTrust | None:
        if not isinstance(value, str):
            return None
        aliases = {
            "external_attested": cls.EXTERNALLY_ATTESTED,
            "exact_request_verified": cls.VERIFIED_EXTENSION,
            "external_exact_request_verified": cls.VERIFIED_EXTENSION,
        }
        return aliases.get(value)


class ProviderCapability(StrEnum):
    """An operation a provider can supply; the Rust ``ExternalCapability`` names.

    Declared independently: ``SAMPLE`` or ``MEAN`` never stand in for ``CDF``.
    """

    SAMPLE = "sample"
    CDF = "cdf"
    QUANTILE = "quantile"
    LOG_PROBABILITY = "log_probability"
    MEAN = "mean"
    COVARIANCE = "covariance"
    CONDITIONAL = "conditional"
    INTERVENTION = "intervention"
    POSTERIOR_PREDICTIVE = "posterior_predictive"
    UPDATE = "update"
    FACTOR = "factor"
    EVALUATE_UTILITY = "evaluate_utility"


class UncertaintySemantics(StrEnum):
    """The uncertainty claims the host understands without a provider method name.

    Any other non-blank ``uncertainty_semantics`` string is a provider-declared
    method identity (Rust ``ExternalUncertaintyMeaning::ProviderDeclared``), never
    promoted to an Antecedent interval.
    """

    POINT_ONLY = "point_only"
    ATTESTED_NOT_REVERIFIABLE = "externally_attested_not_reverifiable"


@dataclass(frozen=True, slots=True)
class CausalProviderSpec:
    """Machine-readable claims required before an extension can run.

    These declarations describe an extension's contract; they do not make its
    scientific claims true. Native licensing and extension verification remain
    separate gates.

    ``capabilities`` and ``meaning`` are the typed counterparts of what a
    ``ProviderObject`` declares: when given they are what
    :meth:`ProviderRegistry.provider_object` derives it from, so a provider
    states its operations and what one value means once. ``uncertainty_semantics``
    is an :class:`UncertaintySemantics` member or a provider-declared method name.
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
    capabilities: tuple[ProviderCapability, ...] = ()
    meaning: str | None = None

    @property
    def uncertainty_method(self) -> str | None:
        """The provider-declared uncertainty method, or ``None`` for a point-only provider."""
        if self.uncertainty_semantics == UncertaintySemantics.POINT_ONLY:
            return None
        return str(self.uncertainty_semantics)

    def __post_init__(self) -> None:
        coerce_fields(self, capabilities=_capabilities(self.capabilities))
        if self.meaning is not None:
            from .joint_distribution import DistributionMeaning

            if self.meaning not in get_args(DistributionMeaning):
                raise ValueError(
                    f"meaning must be one of {', '.join(get_args(DistributionMeaning))}; "
                    f"got {self.meaning!r}"
                )
        if isinstance(self.uncertainty_semantics, str):
            with contextlib.suppress(ValueError):
                coerce_fields(
                    self, uncertainty_semantics=UncertaintySemantics(self.uncertainty_semantics)
                )
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
            coerce_fields(self, **{field_name: values})
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
        coerce_fields(self, provenance=MappingProxyType(dict(self.provenance)))


@runtime_checkable
class CausalProvider(Protocol):
    """Declarative provider contract, optionally executable by the registry.

    The adjustment handoff implements this declaration without owning a learner.
    ``handoff.econml`` / ``EconMLSpec`` are aliases of that handoff.
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
        coerce_fields(self, assumptions=assumptions)
        coerce_fields(self, provenance=MappingProxyType(provenance))
        if self.artifact is not None and not isinstance(self.artifact, bytes):
            raise TypeError("provider artifact must be bytes")


@runtime_checkable
class ExecutableProvider(CausalProvider, Protocol):
    """A :class:`CausalProvider` the registry can run; ``register`` requires this."""

    def execute(self, request: Mapping[str, Any]) -> ProviderExecution:
        """Return an untrusted execution result for the given request."""


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
    #: Identity of the input data when the request refers to one by id. The request
    #: digest stands in when it is omitted: a request carries its whole input.
    snapshot: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("verification fixture needs a name")
        if self.snapshot is not None and (
            not isinstance(self.snapshot, str) or not self.snapshot.strip()
        ):
            raise ValueError("verification snapshot must be a non-empty string")
        if not isinstance(self.request, Mapping):
            raise TypeError("verification request must be a mapping")
        if self.atol < 0 or not np.isfinite(self.atol):
            raise ValueError("verification atol must be finite and non-negative")
        if self.artifact_decoder is not None and not callable(self.artifact_decoder):
            raise TypeError("artifact_decoder must be callable")
        if self.artifact_decoder is not None and self.artifact_digest is None:
            raise ValueError("artifact_decoder requires an expected artifact_digest")
        coerce_fields(self, request=MappingProxyType(dict(self.request)))
        coerce_fields(self, expected_provenance=MappingProxyType(dict(self.expected_provenance)))


@dataclass(frozen=True, slots=True)
class VerifiedRequest:
    """One exact request a provider was verified at.

    ``spec_digest`` is the digest of the provider's declarative spec at that time;
    ``request_digest`` identifies the exact request; ``contract_digest`` is the
    digest of everything the probes were checked against.
    """

    provider: str
    query_family: str
    spec_digest: str
    snapshot: str
    request_digest: str
    kind: str
    contract_digest: str
    probes: tuple[Any, ...]


class ProviderVerificationReceipt:
    """Evidence that a provider passed independent probes at exact requests.

    Issued by Rust: the probe values are stated here, and whether they pass, what
    the receipt covers, and its digest are decided natively. One receipt serves
    every surface. It is exportable, loadable under the digest a consumer
    retained, and consumable by composition through :meth:`trust_evidence`. It
    covers only the listed requests and never grants native licensing or
    calibration.
    """

    __slots__ = ("_fixture_names", "_native_value", "_requests")

    def __init__(self, native: Any, fixture_names: Sequence[str] = ()) -> None:
        from . import _native

        if not isinstance(native, _native.ProviderVerificationReceipt):
            raise TypeError("ProviderVerificationReceipt wraps library-issued receipts only")
        self._native_value = native
        self._fixture_names = tuple(fixture_names)
        from .external import VerificationProbe

        wire = json.loads(native.wire_json)
        self._requests = tuple(
            VerifiedRequest(
                provider=item["identity"]["provider_id"],
                query_family=item["identity"]["object_id"],
                spec_digest=item["identity"]["version_id"],
                snapshot=item["identity"]["snapshot_id"],
                request_digest=item["identity"]["request_id"],
                kind=item["kind"],
                contract_digest=item["contract_digest"],
                probes=tuple(
                    VerificationProbe(p["kind"], p["observed"], p["expected"], p["tolerance"])
                    for p in item["probes"]
                ),
            )
            for item in wire["requests"]
        )

    @property
    def digest(self) -> str:
        """Canonical BLAKE3 digest of the whole receipt; retain it to load an export."""
        return str(self._native_value.digest)

    @property
    def evidence_digest(self) -> str:
        """Alias of :attr:`digest`, the name provenance records carry."""
        return self.digest

    @property
    def evidence_origin(self) -> str | None:
        """Who held the independent reference."""
        return cast("str | None", json.loads(self._native_value.wire_json)["evidence_origin"])

    @property
    def requests(self) -> tuple[VerifiedRequest, ...]:
        """Every exact request verified, in the order supplied."""
        return self._requests

    @property
    def provider(self) -> str:
        """The registered provider name every request belongs to."""
        return self._requests[0].provider

    @property
    def spec_digest(self) -> str:
        """Digest of the provider's spec at verification time."""
        return self._requests[0].spec_digest

    @property
    def verified_request_digests(self) -> tuple[str, ...]:
        """Digests of the exact requests covered, in order."""
        return tuple(request.request_digest for request in self._requests)

    @property
    def fixture_names(self) -> tuple[str, ...]:
        """Caller labels of the fixtures, when this receipt came from ``verify``.

        Labels only: they are not part of the digest and are empty after ``load``.
        """
        return self._fixture_names

    @property
    def deterministic_replay_checked(self) -> bool:
        """Whether every request was replayed and agreed exactly."""
        return all(
            any(probe.kind == "seeded_behavior" for probe in request.probes)
            for request in self._requests
        )

    def covers(self, request: Mapping[str, Any], *, spec_digest: str | None = None) -> bool:
        """Whether ``request`` is exactly one this receipt verified."""
        try:
            digest = _digest("request", request)
        except (TypeError, ValueError):
            return False
        return any(
            item.request_digest == digest
            and (spec_digest is None or item.spec_digest == spec_digest)
            for item in self._requests
        )

    @artifact_export
    def export(self, *, artifact_id: str = "provider-verification") -> ExportedArtifact:
        """Serialize through the shared container; reload with :meth:`load` and :attr:`digest`."""
        return artifact_from_receipt(self, bytes(self._native_value.export(artifact_id)))

    @classmethod
    def load(cls, data: bytes, *, expected_digest: str) -> ProviderVerificationReceipt:
        """Load an export under the digest the consumer retained independently."""
        data = container_bytes(data)
        from . import _native

        return cls(_native.ProviderVerificationReceipt.load(data, expected_digest))

    def trust_evidence(self):
        """Composition evidence that the provider was verified at exactly these requests.

        An input is backed only when its provider id and snapshot id are ones a
        request here names (``provider`` and ``snapshot``).
        """
        from . import _native
        from .composition import TrustEvidence

        return TrustEvidence(_native.TrustEvidence.from_verification(self._native_value))

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form: origin, digest and every request with its probes."""
        return {
            "digest": self.digest,
            "evidence_origin": self.evidence_origin,
            "requests": [
                {
                    "provider": r.provider,
                    "query_family": r.query_family,
                    "spec_digest": r.spec_digest,
                    "snapshot": r.snapshot,
                    "request_digest": r.request_digest,
                    "kind": r.kind,
                    "contract_digest": r.contract_digest,
                    "probes": [
                        {
                            "kind": p.kind,
                            "observed": p.observed,
                            "expected": p.expected,
                            "tolerance": p.tolerance,
                        }
                        for p in r.probes
                    ],
                }
                for r in self._requests
            ],
        }

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ProviderVerificationReceipt) and self.digest == other.digest

    def __hash__(self) -> int:
        return hash(self.digest)

    def __repr__(self) -> str:
        return (
            f"<ProviderVerificationReceipt {self.provider} requests={len(self._requests)} "
            f"digest={self.digest[:12]}>"
        )


#: Name of the pre-unification Python-only report; it is the shared receipt.
ProviderVerificationReport = ProviderVerificationReceipt


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
        coerce_fields(self, request=MappingProxyType(dict(self.request)))


class ProviderRegistry:
    """Small explicit registry for separately installed Python providers."""

    def __init__(self) -> None:
        self._providers: dict[str, tuple[ExecutableProvider, ProviderTrust]] = {}
        self._entry_points: dict[str, str] = {}
        self._verification: dict[str, ProviderVerificationReceipt] = {}

    def load_entry_point(self, name: str) -> CausalProvider:
        """Explicitly load one installed ``antecedent.providers`` entry point.

        An entry point must expose a zero-argument factory returning an
        executable provider. Discovery and provider code run only when this
        method is called. Installing a separate Python distribution therefore
        needs no Antecedent or Rust rebuild. Every loaded provider remains
        externally attested under the same validation as ``register``.
        """
        self._require_new(name)
        matches = tuple(metadata.entry_points().select(group="antecedent.providers", name=name))
        if not matches:
            raise _refuse(
                "provider.entry_point_not_installed",
                f"provider entry point {name!r} is not installed",
                offending=name,
                remedy="install the distribution that declares this antecedent.providers "
                "entry point, or register the provider object directly",
                error=UnknownProviderError,
            )
        if len(matches) != 1:
            raise _refuse(
                "provider.entry_point_ambiguous",
                f"provider entry point {name!r} is ambiguous",
                offending=name,
                remedy="uninstall the duplicate distribution so one entry point names it",
            )
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
        self._require_new(name)
        spec = getattr(provider, "spec", None)
        if not isinstance(spec, CausalProviderSpec):
            raise TypeError("provider.spec must be a CausalProviderSpec")
        if not callable(getattr(provider, "execute", None)):
            raise TypeError("provider must implement execute(request)")
        executable = cast(ExecutableProvider, provider)
        if trust is ProviderTrust.NATIVE_LICENSED:
            raise _refuse(
                "provider.native_licence_not_available",
                "Python extensions cannot register as native licensed providers",
                code="external_verification_failed",
                offending="trust",
                remedy="register as externally attested; native licensing is granted only to "
                "providers implemented in the library",
            )
        if trust is not ProviderTrust.EXTERNALLY_ATTESTED:
            raise _refuse(
                "provider.trust_requires_evidence",
                "Python provider registration defaults to externally attested; "
                "verified and native status require the separate evidence gate",
                code="external_verification_failed",
                offending="trust",
                remedy="register without trust= and call ProviderRegistry.verify with "
                "independent fixtures",
            )
        self._providers[name] = (executable, trust)

    def _require_new(self, name: object) -> None:
        if not isinstance(name, str) or not name.strip():
            raise _refuse(
                "provider.name_invalid",
                "provider name must be a non-empty string",
                offending="name",
                remedy="pass the provider name as a non-empty string",
            )
        if name in self._providers:
            raise _refuse(
                "provider.already_registered",
                f"provider {name!r} is already registered",
                offending=name,
                remedy="use the registered provider, or register this one under another name",
            )

    def get(self, name: str) -> CausalProvider:
        return self._entry(name)[0]

    def verification_report(self, name: str) -> ProviderVerificationReceipt:
        """Return the host's verification receipt for a verified extension."""
        try:
            return self._verification[name]
        except KeyError as exc:
            raise _refuse(
                "provider.not_verified",
                f"provider {name!r} has no verification report",
                code="external_verification_failed",
                offending=name,
                remedy="call ProviderRegistry.verify with independent fixtures first",
                error=UnknownProviderError,
            ) from exc

    def verify(
        self,
        name: str,
        fixtures: Sequence[ProviderVerificationFixture],
        *,
        evidence_origin: str,
    ) -> ProviderVerificationReceipt:
        """Run independent caller fixtures before promoting an external provider.

        Fixtures are sugar: each becomes a typed request with numeric probes
        (shape, known truth, replay, optional uncertainty) that Rust checks and
        seals into one :class:`ProviderVerificationReceipt`. The non-numeric
        references (assumptions, support label, provenance, artifact bytes) are
        compared here and must hold before any probe is submitted. This checks
        the executable contract and supplied references; it does not certify the
        scientific validity of caller supplied truth or grant native licensing.
        Promotion is atomic after every fixture passes.
        """
        provider, trust = self._entry(name)
        if trust is not ProviderTrust.EXTERNALLY_ATTESTED:
            raise _refuse(
                "provider.already_verified",
                "only externally attested providers can be verified",
                code="external_verification_failed",
                offending=name,
                remedy="a provider is verified once; register a new provider object to "
                "verify it again",
            )
        if not isinstance(evidence_origin, str) or not evidence_origin.strip():
            raise _refuse(
                "provider.evidence_origin_missing",
                "evidence_origin must identify the independent fixture source",
                offending="evidence_origin",
                remedy="name who held the independent reference, for example 'lab-benchmark/v3'",
            )
        fixtures = tuple(fixtures)
        if any(not isinstance(fixture, ProviderVerificationFixture) for fixture in fixtures):
            raise TypeError("verification fixtures must be ProviderVerificationFixture")
        if not fixtures or len({fixture.name for fixture in fixtures}) != len(fixtures):
            raise _refuse(
                "provider.fixtures_invalid",
                "verification requires distinct, non-empty fixtures",
                offending="fixtures",
                remedy="pass at least one fixture and give each a distinct name",
            )
        spec = provider.spec
        spec_digest = _digest("spec", _spec_evidence(spec))
        wire_requests = []
        for fixture in fixtures:
            result = self.execute(name, fixture.request)
            wire_requests.append(self._fixture_request(name, spec, spec_digest, fixture, result))
        if _digest("spec", _spec_evidence(provider.spec)) != spec_digest:
            raise _refuse(
                "provider.spec_changed",
                "provider spec changed during verification",
                code="external_binding_mismatch",
                offending=name,
                remedy="keep the provider's declarative spec fixed while it is verified",
            )
        from . import _native

        native, refusal = _native.ProviderVerificationReceipt.verify(
            json.dumps({"evidence_origin": evidence_origin, "requests": wire_requests})
        )
        if refusal is not None or native is None:
            raise _fixture_refusal(json.loads(refusal or "{}"), fixtures)
        receipt = ProviderVerificationReceipt(native, tuple(fixture.name for fixture in fixtures))
        self._verification[name] = receipt
        self._providers[name] = (provider, ProviderTrust.VERIFIED_EXTENSION)
        return receipt

    def _fixture_request(
        self,
        name: str,
        spec: CausalProviderSpec,
        spec_digest: str,
        fixture: ProviderVerificationFixture,
        result: ProviderResult,
    ) -> dict[str, Any]:
        """Compare one fixture and state it as a typed request with numeric probes."""

        def differs(slot: str, text: str, offending: str) -> ProviderRefusal:
            return _refuse(
                f"provider.fixture_{slot}",
                f"fixture {fixture.name!r}: {text}",
                code="external_verification_failed",
                offending=offending,
                remedy="fix the provider so it reproduces the independent reference, or "
                "correct a reference that was wrong",
            )

        expected = _provider_array(fixture.expected_estimate, "expected estimate")
        if result.estimate.shape != expected.shape:
            raise differs("estimate", "estimate differs from reference", fixture.name)
        probes = [
            _probe("shape", float(result.estimate.size), float(expected.size), 0.0),
            _probe("known_truth", _max_abs_diff(result.estimate, expected), 0.0, fixture.atol),
        ]
        if fixture.expected_uncertainty is None:
            if result.uncertainty is not None:
                raise differs("uncertainty", "unexpected uncertainty", fixture.name)
        else:
            expected_uncertainty = _provider_array(
                fixture.expected_uncertainty, "expected uncertainty"
            )
            if result.uncertainty is None or result.uncertainty.shape != expected_uncertainty.shape:
                raise differs("uncertainty", "uncertainty differs from reference", fixture.name)
            probes.append(
                _probe(
                    "uncertainty",
                    _max_abs_diff(result.uncertainty, expected_uncertainty),
                    0.0,
                    fixture.atol,
                )
            )
        if result.uncertainty is not None and spec.uncertainty_semantics == "point_only":
            raise differs("uncertainty", "point-only provider returned uncertainty", fixture.name)
        if result.assumptions != tuple(fixture.expected_assumptions):
            raise differs("assumptions", "assumptions differ from reference", fixture.name)
        if result.support_status != fixture.expected_support_status:
            raise differs("support", "support differs from reference", fixture.name)
        for key, value in fixture.expected_provenance.items():
            if result.provenance.get(key) != value:
                raise differs("provenance", f"provenance {key!r} differs", fixture.name)
        if fixture.artifact_digest is not None:
            if (
                result.artifact is None
                or hashlib.sha256(result.artifact).hexdigest() != fixture.artifact_digest
            ):
                raise differs("artifact", "artifact digest differs", fixture.name)
        elif result.artifact is not None:
            raise differs("artifact", "unverified artifact", fixture.name)
        if fixture.artifact_decoder is not None:
            decoded = fixture.artifact_decoder(result.artifact)
            if decoded != fixture.expected_decoded_artifact:
                raise differs("artifact", "artifact round trip differs", fixture.name)
        if spec.deterministic:
            replay = self.execute(name, fixture.request)
            probes.append(
                _probe(
                    "seeded_behavior",
                    0.0 if _same_provider_result(result, replay) else 1.0,
                    0.0,
                    0.0,
                )
            )
        from . import _native

        request_digest = _request_digest(fixture.request)
        return {
            "identity": {
                "provider_id": name,
                "object_id": spec.query_family,
                "version_id": spec_digest,
                "snapshot_id": fixture.snapshot or request_digest,
                "request_id": request_digest,
            },
            "deterministic": spec.deterministic,
            "reports_uncertainty": fixture.expected_uncertainty is not None,
            "artifact_blake3": None
            if result.artifact is None
            else _native.blake3_digest_hex(result.artifact),
            "probes": probes,
        }

    def provider_object(
        self,
        name: str,
        request: Mapping[str, Any],
        *,
        snapshot: str | None = None,
        meaning: str | None = None,
    ) -> ProviderObject:
        """The :class:`~antecedent.external.ProviderObject` of a registered provider at ``request``.

        Derived, never restated: the provider id is the registered name, the
        object is its query family, the version is the digest of its declarative
        spec, the request is the canonical request digest, and operations and
        meaning come from the spec. The snapshot defaults to the request digest
        (a request carries its whole input). This is the identity a verification
        receipt records for the same request.
        """
        from .external import ProviderObject

        spec = self._entry(name)[0].spec
        declared = meaning if meaning is not None else spec.meaning
        if declared is None:
            raise _refuse(
                "provider.meaning_undeclared",
                f"provider {name!r} declares no distribution meaning; "
                "set spec.meaning or pass meaning=",
                code="distribution_meaning_mismatch",
                offending="meaning",
                remedy="declare what one value of the provider's law means in "
                "CausalProviderSpec.meaning, or pass meaning= here",
            )
        if not spec.capabilities:
            raise _refuse(
                "provider.capabilities_undeclared",
                f"provider {name!r} declares no capabilities; set spec.capabilities",
                code="external_capability_missing",
                offending="capabilities",
                remedy="list the operations the provider supplies in "
                "CausalProviderSpec.capabilities",
            )
        digest = _request_digest(request)
        return ProviderObject(
            provider_id=name,
            object_id=spec.query_family,
            version=_digest("spec", _spec_evidence(spec)),
            snapshot=snapshot if snapshot is not None else digest,
            request=digest,
            meaning=cast("Any", declared),
            capabilities=tuple(str(c) for c in spec.capabilities),
        )

    def _entry(self, name: str) -> tuple[ExecutableProvider, ProviderTrust]:
        try:
            return self._providers[name]
        except (KeyError, TypeError) as exc:
            raise _refuse(
                "provider.not_registered",
                f"provider {name!r} is not explicitly registered",
                offending=str(name),
                remedy="register it with ProviderRegistry.register, or load an installed "
                "one with load_entry_point",
                error=UnknownProviderError,
            ) from exc

    def execute(self, name: str, request: Mapping[str, Any]) -> ProviderResult:
        provider, trust = self._entry(name)
        spec = provider.spec
        report = self._verification.get(name)
        current_spec_digest = _digest("spec", _spec_evidence(spec))
        if report is not None and current_spec_digest != report.spec_digest:
            raise _refuse(
                "provider.spec_changed",
                "verified provider spec changed after verification",
                code="external_binding_mismatch",
                offending=name,
                remedy="verify the changed provider again, or restore the verified spec",
            )
        request_digest = _request_digest(request)
        effective_trust = trust
        if report is not None and not report.covers(request, spec_digest=current_spec_digest):
            effective_trust = ProviderTrust.EXTERNALLY_ATTESTED
        raw = provider.execute(MappingProxyType(dict(request)))
        if not isinstance(raw, ProviderExecution):
            raise TypeError("provider execute() must return ProviderExecution")
        estimate = _provider_array(raw.estimate, "estimate")
        if isinstance(spec.output_shape, tuple) and estimate.shape != spec.output_shape:
            raise _refuse(
                "provider.output_shape_mismatch",
                f"provider estimate shape {estimate.shape} does not match "
                f"declared {spec.output_shape}",
                code="external_binding_mismatch",
                offending="estimate",
                remedy="return an estimate of the declared output_shape, or declare the "
                "shape the provider returns",
            )
        uncertainty = (
            None if raw.uncertainty is None else _provider_array(raw.uncertainty, "uncertainty")
        )
        if uncertainty is not None and uncertainty.shape not in ((), estimate.shape):
            raise _refuse(
                "provider.uncertainty_shape_mismatch",
                "provider uncertainty must be scalar or match the estimate shape",
                code="external_binding_mismatch",
                offending="uncertainty",
                remedy="return one uncertainty value, or one per estimate entry",
            )
        if uncertainty is not None and spec.uncertainty_semantics == "point_only":
            raise _refuse(
                "provider.uncertainty_undeclared",
                "point-only provider returned uncertainty",
                code="external_binding_mismatch",
                offending="uncertainty",
                remedy="declare how the uncertainty is to be read in "
                "CausalProviderSpec.uncertainty_semantics, or return none",
            )
        provenance = _provider_provenance(spec.provenance)
        provenance.update(_provider_provenance(raw.provenance))
        provenance["registry_name"] = name
        if name in self._entry_points:
            provenance["entry_point"] = self._entry_points[name]
        provenance["trust_boundary"] = effective_trust.value
        if report is not None and effective_trust is ProviderTrust.VERIFIED_EXTENSION:
            provenance["verification_evidence_digest"] = report.evidence_digest
            provenance["verification_evidence_origin"] = cast(str, report.evidence_origin)
        provenance["request_digest_method"] = _REQUEST_DIGEST_METHOD
        from . import _native

        header = {
            "version": 1,
            "provider": name,
            "query_family": spec.query_family,
            "trust": effective_trust.value,
            "spec_digest": current_spec_digest,
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


def _capabilities(value: Any) -> tuple[ProviderCapability, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError("capabilities must be a sequence of capability names, not one string")
    names = []
    for item in value:
        try:
            names.append(ProviderCapability(item))
        except ValueError:
            known = ", ".join(member.value for member in ProviderCapability)
            raise ValueError(f"unknown provider capability {item!r}; use one of {known}") from None
    if len(set(names)) != len(names):
        raise ValueError("capabilities must not repeat an operation")
    return tuple(names)


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


_DIGEST_DOMAINS = {
    "spec": "antecedent.provider.spec.v1",
    # v2: arrays and tables enter by dtype, shape and byte digest.
    "request": "antecedent.provider.request.v2",
}


def _digest(domain: str, value: Any) -> str:
    """Canonical digest of a provider spec or request; a value with no JSON form raises."""
    return canonical_digest(_DIGEST_DOMAINS[domain], value)


_REQUEST_DIGEST_METHOD = "canonical_json_v2"


def _request_digest(request: Mapping[str, Any]) -> str:
    """The canonical digest of a request, or the refusal that it has none.

    A request is hashed by its content: JSON values, and arrays and column tables
    by dtype, shape and bytes. There is no fallback to a textual ``repr``, which
    truncates large data so two datasets could share a digest; an object with no
    canonical form is refused and must be passed as columns.
    """
    try:
        return _digest("request", request)
    except (TypeError, ValueError) as error:
        raise _refuse(
            "provider.request_not_canonical",
            f"provider request has no canonical digest: {error}",
            offending="request",
            remedy="pass JSON values, NumPy arrays or columns (a dict of arrays, a pandas "
            "DataFrame or an Arrow table); finite numbers only in lists",
        ) from error


def _probe(kind: str, observed: float, expected: float, tolerance: float) -> dict[str, Any]:
    return {
        "kind": kind,
        "observed": float(observed),
        "expected": float(expected),
        "tolerance": float(tolerance),
    }


def _max_abs_diff(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    return float(np.max(np.abs(a - b))) if a.size else 0.0


_PROBE_FAILURES = {
    "shape": "estimate differs from reference",
    "known_truth": "estimate differs from reference",
    "uncertainty": "uncertainty differs from reference",
    "seeded_behavior": "deterministic replay differs",
}


def _fixture_refusal(
    refusal: Mapping[str, Any], fixtures: Sequence[ProviderVerificationFixture]
) -> ProviderValueRefusal:
    """The refusal for a Rust verification refusal, naming the failing fixture."""
    offending = str(refusal.get("offending") or "")
    match = re.fullmatch(r"request\[(\d+)\]\.(\w+)", offending)
    detail = str(refusal.get("detail", "provider verification refused"))
    subject = offending or None
    if match is not None:
        fixture = fixtures[int(match.group(1))]
        text = _PROBE_FAILURES.get(match.group(2), f"{detail} at {match.group(2)}")
        message = f"fixture {fixture.name!r}: {text}"
        subject = fixture.name
    else:
        message = f"{detail}: {offending}" if offending else detail
    return ProviderValueRefusal(
        {
            "code": refusal.get("code") or "external_verification_failed",
            "stage": str(refusal.get("stage") or "verify"),
            "detail": detail,
            "message": message,
            "offending": subject,
            "remedy": refusal.get("remedy")
            or "fix the provider so it reproduces the independent reference, or correct "
            "a reference that was wrong",
        },
        text=message,
    )


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
    "ExecutableProvider",
    "MechanismWrapper",
    "ProviderCapability",
    "ProviderTrust",
    "ProviderExecution",
    "ProviderRegistry",
    "ProviderResult",
    "ProviderQuery",
    "ProviderRefusal",
    "ProviderValueRefusal",
    "ProviderVerificationFixture",
    "ProviderVerificationReceipt",
    "ProviderVerificationReport",
    "UncertaintySemantics",
    "UnknownProviderError",
    "UtilityFn",
    "VerifiedRequest",
    "providers",
]
