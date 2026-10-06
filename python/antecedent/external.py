"""Bind a foreign scientific result to an identified causal contract.

Antecedent checks the *causal claim*; another system supplies the numbers. This
follows the :mod:`antecedent.handoff` flow: identify first, derive a spec from
the identification, then bind what the provider produced and get an
inspectable, exportable claim back::

    ident = ac.identify(graph=edges, names=names, query=ac.ResponseCurve("a", "y", grid=[0, 1, 2]))
    spec = ac.external.response(ident, outcome_units="mmHg", population="target")
    claim = spec.bind(ac.external.Response(provider=provider, values=[...], attested_by="lab"))
    claim.inspect(); claim.export()

Every identity, check and refusal rule lives in Rust; this module builds
typed declarations and raises the Rust refusal as
:class:`~antecedent.errors.CausalUnsupportedError` with its registered
``reason_code`` and ``remedy``. Foreign numerical work is never presented as
native estimation: :attr:`BoundExternalClaim.native` is always ``False``, trust
is a :class:`~antecedent.extensibility.ProviderTrust` (never ``native_licensed``),
and provider-declared uncertainty is a declaration, not an Antecedent interval.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._native import ExternalClaimArtifact as _NativeClaim
from ._native import bind_external_response as _bind_external_response
from .errors import CausalUnsupportedError, CausalValueError
from .extensibility import ProviderTrust
from .joint_distribution import DistributionMeaning, QuantityCondition, ScientificQuantity

_ID_STATUS = {
    "NonparametricallyIdentified": "nonparametrically_identified",
    "IdentifiedUnderParametricRestrictions": "identified_under_parametric_restrictions",
    "IdentifiedUnderPriorRestrictions": "identified_under_prior_restrictions",
    "PartiallyIdentified": "partially_identified",
    "GraphDependent": "graph_dependent",
    "NotIdentified": "not_identified",
}
_SUPPORT_SEVERITY = (
    "supported",
    "weak_overlap",
    "extrapolative",
    "outside_empirical_support",
    "missing_evidence",
)
OBSERVATIONAL = "observational"


@dataclass(frozen=True, slots=True)
class ProviderObject:
    """One foreign provider object at an exact request.

    ``capabilities`` names the operations the provider really supplies
    (``sample``, ``cdf``, ``quantile``, ``log_probability``, ``mean``,
    ``covariance``, ``conditional``, ``intervention``, ``posterior_predictive``).
    ``sample`` or ``mean`` never stand in for ``cdf``.
    """

    provider_id: str
    object_id: str
    version: str
    snapshot: str
    request: str
    meaning: DistributionMeaning
    capabilities: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "capabilities", tuple(self.capabilities))

    def _wire(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "object_id": self.object_id,
            "version_id": self.version,
            "snapshot_id": self.snapshot,
            "request_id": self.request,
            "meaning": self.meaning,
            "capabilities": list(self.capabilities),
        }


@dataclass(frozen=True, slots=True)
class VerificationProbe:
    """An independently computed check of one provider property.

    ``kind`` is one of ``shape``, ``normalization``, ``moments``, ``known_truth``,
    ``seeded_behavior``, ``support``, ``update_coherence``, ``monotonicity``. The
    probe values must not come from the provider under test.
    """

    kind: str
    observed: float
    expected: float
    tolerance: float

    def _wire(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "observed": float(self.observed),
            "expected": float(self.expected),
            "tolerance": float(self.tolerance),
        }


@dataclass(frozen=True, slots=True)
class Response:
    """A finite response grid computed by a foreign provider.

    Supply ``attested_by`` for an attested provider, or ``probes`` to have the
    exact provider object verified (``verified_extension``). ``support`` is the
    provider's per-coordinate label; absent means support was not assessed and is
    recorded as ``missing_evidence``, never ``supported``. ``quantities`` is
    normally left unset so the spec's coordinates are used.
    """

    provider: ProviderObject
    values: Sequence[float]
    evidence: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    attested_by: str | None = None
    probes: tuple[VerificationProbe, ...] = ()
    uncertainty_method: str | None = None
    support: tuple[str, ...] | None = None
    quantities: tuple[ScientificQuantity, ...] | None = None
    graph_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(float(v) for v in self.values))
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        object.__setattr__(self, "probes", tuple(self.probes))


@dataclass(frozen=True, slots=True)
class Equivalence:
    """A separately checked licence to read an observational law as interventional."""

    graph_id: str
    interventional_regime: str
    justification: str


@dataclass(frozen=True, slots=True)
class LineageLink:
    """One derivation step behind a reported number."""

    id: str
    stage: str
    parents: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ClaimInspection:
    """What stands behind a bound external claim, as data rather than a renderer."""

    statement: str
    trust: ProviderTrust
    native: bool
    support_status: str
    provenance_label: str
    lineage: tuple[LineageLink, ...]
    uncertainty_method: str | None

    def __str__(self) -> str:
        chain = " <- ".join(link.id for link in reversed(self.lineage))
        return (
            f"{self.statement}\n  trust: {self.trust.value}; native: {self.native}\n  from: {chain}"
        )


@dataclass(frozen=True, slots=True)
class ExternalSpec:
    """What an identified contract requires of an external result.

    Built by :func:`response`. ``bind`` checks a provider's :class:`Response`
    against it and returns the claim, or refuses.
    """

    contract_id: str
    graph_id: str
    identification: str
    quantities: tuple[ScientificQuantity, ...]
    accepted_meanings: tuple[DistributionMeaning, ...] = ("interventional_predictive",)
    require_evidence: tuple[str, ...] = ()
    require_assumptions: tuple[str, ...] = ()
    equivalences: tuple[Equivalence, ...] = ()
    statement: str = ""

    def _contract_wire(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "identification": self.identification,
            "estimand": [q._wire() for q in self.quantities],
            "accepted_meanings": list(self.accepted_meanings),
            "required_evidence_ids": list(self.require_evidence),
            "required_assumption_ids": list(self.require_assumptions),
            "equivalences": [
                {
                    "graph_id": e.graph_id,
                    "interventional_regime_id": e.interventional_regime,
                    "justification_id": e.justification,
                }
                for e in self.equivalences
            ],
        }

    def bind(self, response: Response) -> BoundExternalClaim:
        """Check ``response`` against this contract and bind it, or refuse."""
        if not isinstance(response, Response):
            raise TypeError("ExternalSpec.bind expects an antecedent.external.Response")
        quantities = response.quantities if response.quantities is not None else self.quantities
        wire = {
            "provider": response.provider._wire(),
            "graph_id": response.graph_id or self.graph_id,
            "quantities": [q._wire() for q in quantities],
            "values": list(response.values),
            "evidence_ids": list(response.evidence),
            "assumption_ids": list(response.assumptions),
            "attestor": response.attested_by,
            "probes": [p._wire() for p in response.probes] if response.probes else None,
            "uncertainty_method": response.uncertainty_method,
            "point_support": None if response.support is None else list(response.support),
        }
        native, refusal = _bind_external_response(
            json.dumps(self._contract_wire()), json.dumps(wire), self.contract_id
        )
        if refusal is not None or native is None:
            raise _refusal_error(json.loads(refusal or "{}"))
        return BoundExternalClaim(native, self.statement)

    def load(self, data: bytes, *, expected: Mapping[str, Any]) -> BoundExternalClaim:
        """Load a claim exported elsewhere under the consumer's retained identity.

        ``expected`` is the identity mapping the consumer holds independently of
        the bytes (for example a producer's ``claim.identity`` recorded
        out-of-band); an artifact whose identity differs refuses.
        """
        native = _NativeClaim.load(data, json.dumps(dict(expected)))
        return BoundExternalClaim(native, self.statement)


class BoundExternalClaim:
    """A foreign result bound to an identified contract: inspectable, exportable."""

    def __init__(self, native: _NativeClaim, statement: str = "") -> None:
        self._native = native
        self._statement = statement
        self._meta: dict[str, Any] = json.loads(native.metadata_json)

    @property
    def identity(self) -> dict[str, Any]:
        """The full identity a consumer should retain to load this claim."""
        return dict(self._meta["identity"])

    @property
    def native(self) -> bool:
        """Always ``False``: the numbers were computed outside Antecedent."""
        return bool(self._meta["native_estimation"])

    @property
    def trust(self) -> ProviderTrust:
        return ProviderTrust(self._meta["identity"]["trust"])

    @property
    def values(self) -> NDArray[np.float64]:
        """One bounded copy of the response values, in coordinate order."""
        return np.asarray(self._native.values_copy())

    @property
    def quantities(self) -> tuple[ScientificQuantity, ...]:
        return tuple(ScientificQuantity._from_wire(q) for q in self._meta["identity"]["quantities"])

    @property
    def support(self) -> tuple[str, ...]:
        """Per-coordinate support labels (``missing_evidence`` when undeclared)."""
        return tuple(self._meta["identity"]["point_status"])

    @property
    def support_status(self) -> str:
        """Worst per-coordinate label, the summary an old consumer reads."""
        return max(self.support, key=_SUPPORT_SEVERITY.index)

    @property
    def identification(self) -> str:
        return str(self._meta["identity"]["identification"])

    @property
    def uncertainty_method(self) -> str | None:
        return self._meta["identity"]["uncertainty_method"]

    @property
    def lineage(self) -> tuple[LineageLink, ...]:
        """Derivation chain, parents before children."""
        return tuple(
            LineageLink(item["id"], item["stage"], tuple(item["parents"]))
            for item in self._meta["identity"]["lineage"]
        )

    def stages_behind(self, link: str = "claim") -> frozenset[str]:
        """Stages standing behind ``link`` (default: the reported claim)."""
        by_id = {item.id: item for item in self.lineage}
        if link not in by_id:
            raise CausalValueError(f"unknown lineage link {link!r}")
        seen: set[str] = set()
        stack = [link]
        while stack:
            current = stack.pop()
            if current not in seen:
                seen.add(current)
                stack.extend(by_id[current].parents)
        return frozenset(by_id[item].stage for item in seen)

    @property
    def provenance_label(self) -> str:
        return self._native.provenance_label

    def claim(self) -> str:
        """One sentence that names the external execution; never a native claim."""
        subject = self._statement or "The requested quantity"
        return (
            f"{subject.rstrip('.')}; values supplied by {self.provenance_label} "
            f"({self.trust.value}), not estimated natively."
        )

    def inspect(self) -> ClaimInspection:
        return ClaimInspection(
            statement=self.claim(),
            trust=self.trust,
            native=self.native,
            support_status=self.support_status,
            provenance_label=self.provenance_label,
            lineage=self.lineage,
            uncertainty_method=self.uncertainty_method,
        )

    def export(self, *, artifact_id: str = "external-claim") -> bytes:
        return self._native.export(artifact_id)

    def __repr__(self) -> str:
        return (
            f"<BoundExternalClaim {self.provenance_label} trust={self.trust.value} "
            f"support={self.support_status}>"
        )


class ExternalRefusal(CausalUnsupportedError):
    """A :class:`CausalUnsupportedError` carrying the structured Rust refusal fields.

    Existing ``except CausalUnsupportedError`` handlers keep working;
    ``reason_code`` and ``remedy`` are the inherited, registered fields.
    """

    def __init__(self, refusal: Mapping[str, Any]) -> None:
        parts = [str(refusal["detail"])]
        if refusal.get("offending"):
            parts.append(f"at {refusal['offending']}")
        if refusal.get("expected") is not None or refusal.get("supplied") is not None:
            parts.append(
                f"(expected {refusal.get('expected')!r}, supplied {refusal.get('supplied')!r})"
            )
        super().__init__(" ".join(parts), reason_code=refusal["code"], remedy=refusal.get("remedy"))
        #: Refusing stage: ``declare``, ``negotiate``, ``verify`` or ``bind``.
        self.stage: str = refusal["stage"]
        #: Namespaced ``family.slot`` detail.
        self.detail: str = refusal["detail"]
        #: Offending coordinate (``coordinate[i]``) or probe, when there is one.
        self.offending: str | None = refusal.get("offending")
        #: Expected and supplied semantics when a comparison failed.
        self.expected: str | None = refusal.get("expected")
        self.supplied: str | None = refusal.get("supplied")
        #: Missing operation, for a capability refusal.
        self.capability: str | None = refusal.get("capability")


def _refusal_error(refusal: Mapping[str, Any]) -> ExternalRefusal:
    return ExternalRefusal(refusal)


def _status_name(status: str) -> str:
    return _ID_STATUS.get(status, status)


def _graph_identity(identification: Any) -> str:
    graph = identification.graph
    if not isinstance(graph, (list, tuple)):
        raise CausalValueError(
            "this graph type has no canonical identity here; pass graph_id= naming the "
            "graph the external provider used"
        )
    canonical = json.dumps(
        {
            "edges": sorted([str(a), str(b)] for a, b in graph),
            "names": list(identification.names or ()),
            "identifier": identification.identifier or identification.method,
            "adjustment_set": sorted(identification.adjustment_set),
        },
        sort_keys=True,
    )
    return "graph:" + hashlib.sha256(canonical.encode()).hexdigest()[:32]


def _query_quantities(
    query: Any, *, outcome_units: str, population: str, transform: str
) -> tuple[ScientificQuantity, ...]:
    kind = getattr(query, "kind", None)
    outcome = query.outcome

    def make(regime: str, functional: str) -> ScientificQuantity:
        return ScientificQuantity(
            variable_id=outcome,
            variable_name=outcome,
            role="outcome",
            units=outcome_units,
            population_id=population,
            regime_id=regime,
            horizon=0,
            functional_id=functional,
            conditioning=(),
            transform_id=transform,
        )

    if kind == "response_curve" and getattr(query, "horizons", None) is None:
        return tuple(make(f"do({query.treatment}={dose:g})", "mean") for dose in query.grid)
    if kind == "average":
        return (
            make(
                f"do({query.treatment}={query.active_level:g})"
                f" - do({query.treatment}={query.control_level:g})",
                "ate",
            ),
        )
    raise CausalValueError(
        f"cannot derive coordinates for {type(query).__name__}; pass quantities= "
        "naming each requested coordinate"
    )


def response(
    identification: Any,
    *,
    outcome_units: str | None = None,
    population: str = "target",
    transform: str = "identity",
    quantities: Sequence[ScientificQuantity] | None = None,
    graph_id: str | None = None,
    contract_id: str | None = None,
    accepted_meanings: Sequence[DistributionMeaning] = ("interventional_predictive",),
    require_evidence: Sequence[str] = (),
    require_assumptions: Sequence[str] = (),
    equivalences: Sequence[Equivalence] = (),
) -> ExternalSpec:
    """Derive what an identification requires of an external response grid.

    Coordinates come from the identified query (a ``ResponseCurve`` grid becomes
    one ``do(treatment=dose)`` coordinate per point); ``outcome_units`` is
    required because the query cannot know it, and no unit is inferred. Pass
    ``quantities=`` to override the derivation for any other grid. An
    unidentified query refuses at ``bind`` with ``effect_not_identified``.
    """
    status = _status_name(str(identification.status))
    if quantities is None:
        if outcome_units is None or not outcome_units.strip():
            raise CausalValueError(
                "outcome_units= is required: units are never inferred or converted"
            )
        coordinates = _query_quantities(
            identification.query,
            outcome_units=outcome_units,
            population=population,
            transform=transform,
        )
    else:
        coordinates = tuple(quantities)
    identity = graph_id or _graph_identity(identification)
    return ExternalSpec(
        contract_id=contract_id or "contract:" + identity.removeprefix("graph:"),
        graph_id=identity,
        identification=status,
        quantities=coordinates,
        accepted_meanings=tuple(accepted_meanings),
        require_evidence=tuple(require_evidence),
        require_assumptions=tuple(require_assumptions),
        equivalences=tuple(equivalences),
        statement=identification.statement,
    )


__all__ = [
    "BoundExternalClaim",
    "ClaimInspection",
    "Equivalence",
    "ExternalRefusal",
    "ExternalSpec",
    "LineageLink",
    "ProviderObject",
    "QuantityCondition",
    "Response",
    "ScientificQuantity",
    "VerificationProbe",
    "response",
]
