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
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from ._native import ExternalClaimArtifact as _NativeClaim
from ._native import bind_external_response as _bind_external_response
from ._native import bind_external_to_program as _bind_external_to_program
from .errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from .extensibility import ProviderTrust
from .joint_distribution import DistributionMeaning, QuantityCondition, ScientificQuantity

if TYPE_CHECKING:
    from .program_claims import ProgramBinding

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
#: Dose units a spec carries when the caller names none; units are never inferred, so a
#: program that names real dose units refuses this one (``program_binding.dose_grid_changed``).
UNSPECIFIED_UNITS = "unspecified"


def _strings(what: str, values: object) -> tuple[str, ...]:
    """A tuple of strings; a bare string (which would split into characters) is refused."""
    if isinstance(values, (str, bytes)):
        raise CausalTypeError(f"{what} must be a sequence of strings, not a single string")
    try:
        items = tuple(values)  # type: ignore[var-annotated,arg-type]
    except TypeError as error:
        raise CausalTypeError(f"{what} must be a sequence of strings") from error
    if any(not isinstance(item, str) for item in items):
        raise CausalTypeError(f"{what} must hold only strings")
    return items


def _real(what: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as error:
            raise CausalTypeError(f"{what} must be a real number, got {value!r}") from error
    return float(value)


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
        for name in ("provider_id", "object_id", "version", "snapshot", "request", "meaning"):
            if not isinstance(getattr(self, name), str):
                raise CausalTypeError(f"ProviderObject.{name} must be a string")
        object.__setattr__(
            self, "capabilities", _strings("ProviderObject.capabilities", self.capabilities)
        )

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

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str):
            raise CausalTypeError("VerificationProbe.kind must be a string")
        for name in ("observed", "expected", "tolerance"):
            _real(f"VerificationProbe.{name}", getattr(self, name))

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
        if not isinstance(self.provider, ProviderObject):
            raise CausalTypeError("Response.provider must be a ProviderObject")
        if isinstance(self.values, (str, bytes)):
            raise CausalTypeError("Response.values must be a sequence of numbers, not a string")
        try:
            values = tuple(_real("Response.values entry", v) for v in self.values)
        except TypeError as error:
            raise CausalTypeError("Response.values must be an iterable of numbers") from error
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "evidence", _strings("Response.evidence", self.evidence))
        object.__setattr__(self, "assumptions", _strings("Response.assumptions", self.assumptions))
        probes = tuple(self.probes)
        if any(not isinstance(probe, VerificationProbe) for probe in probes):
            raise CausalTypeError("Response.probes must be VerificationProbe values")
        object.__setattr__(self, "probes", probes)
        if self.attested_by is not None and not isinstance(self.attested_by, str):
            raise CausalTypeError("Response.attested_by must be a string or None")


@dataclass(frozen=True, slots=True)
class Equivalence:
    """A separately checked licence to read an observational law as interventional.

    Two scopes, exactly one of which is given. ``interventional_regime`` covers
    a coordinate that differs from the request only in its regime.
    ``treatment`` with ``value_to_regime`` covers ``P(Y | T=v)`` standing for
    ``P(Y | do(T=v))`` over a grid: each conditioning value maps to exactly the
    regime it licenses, and nothing else about the coordinate may differ. Use
    :meth:`conditioned` or :meth:`ExternalSpec.observational_equivalence`.
    ``justification`` names the check, for example an adjustment-set proof.
    """

    graph_id: str
    interventional_regime: str | None
    justification: str
    treatment: str | None = None
    value_to_regime: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if (self.interventional_regime is None) == (self.treatment is None):
            raise CausalValueError(
                "an Equivalence names exactly one scope: interventional_regime, or "
                "treatment with value_to_regime"
            )
        if (self.treatment is None) != (self.value_to_regime is None):
            raise CausalValueError("treatment and value_to_regime go together")

    @classmethod
    def conditioned(
        cls, graph_id: str, treatment: str, value_to_regime: Mapping[str, str], justification: str
    ) -> Equivalence:
        """``P(Y | treatment=v)`` read as ``P(Y | do(treatment=v))`` for each listed value."""
        return cls(graph_id, None, justification, treatment, dict(value_to_regime))

    def _wire(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "justification_id": self.justification,
            "interventional_regime_id": self.interventional_regime,
            "conditioned_treatment": None
            if self.treatment is None
            else {
                "variable_id": self.treatment,
                "value_to_regime": [[v, r] for v, r in (self.value_to_regime or {}).items()],
            },
        }


@dataclass(frozen=True, slots=True)
class LineageLink:
    """One derivation step behind a reported number."""

    id: str
    stage: str
    parents: tuple[str, ...]
    #: Merkle digest (BLAKE3 of id, stage and parent digests), lowercase hex. Empty
    #: only for links this surface appends itself and cannot hash.
    digest: str = ""
    #: Digests of ``parents`` in the same order, as this link believes them.
    parent_digests: tuple[str, ...] = ()


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
class ProgramRequest:
    """The identified request an external result answers.

    Read from the identified query, never from the spec's ``quantities``: a
    caller's coordinate override can describe a different question, and this is
    what it is checked against (see :func:`antecedent.program_claims.bind_to_program`).
    """

    treatment: str
    outcome: str
    population: str
    doses: tuple[float, ...]
    dose_units: str
    outcome_units: str
    transform: str = "identity"


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
    treatment: str | None = None
    doses: tuple[float, ...] = ()
    #: The identified dose-grid request, when the query is one; ``None`` otherwise.
    request: ProgramRequest | None = None

    def _content_id(self, *, with_coordinates: bool = False) -> str:
        """Digest of the contract's premises: graph, identification, meanings, evidence.

        The coordinates are left out by default so a ``quantities`` override shares
        the contract identity of the request it claims to answer and is refused on
        its coordinates, not mistaken for another contract.
        """
        wire = self._contract_wire()
        if not with_coordinates:
            del wire["estimand"]
        digest = hashlib.sha256(json.dumps(wire, sort_keys=True).encode()).hexdigest()
        return f"contract:{digest}"

    def _require_request(self) -> ProgramRequest:
        """The identified dose-grid request, or the typed refusal for a spec with none."""
        if self.request is None:
            raise ExternalRefusal(
                {
                    "code": "invalid_argument",
                    "stage": "bind",
                    "detail": "program_binding.no_identified_request",
                    "offending": "request",
                    "expected": "a response-curve query with a dose grid",
                    "supplied": None,
                    "remedy": "derive the spec from an identified ResponseCurve",
                }
            )
        return self.request

    def _program_claim(self) -> dict[str, Any]:
        """What this spec declares about the program it answers, for the Rust check."""
        request = self._require_request()
        return {
            "contract_id": self._content_id(),
            "graph_id": self.graph_id,
            "declared_identity": self.contract_id,
            "treatment_id": request.treatment,
            "outcome_id": request.outcome,
            "population_id": request.population,
            "doses": list(request.doses),
            "dose_units": request.dose_units,
            "quantities": [q._wire() for q in self.quantities],
        }

    def _contract_wire(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "identification": self.identification,
            "estimand": [q._wire() for q in self.quantities],
            "accepted_meanings": list(self.accepted_meanings),
            "required_evidence_ids": list(self.require_evidence),
            "required_assumption_ids": list(self.require_assumptions),
            "equivalences": [e._wire() for e in self.equivalences],
        }

    def observational_quantities(self) -> tuple[ScientificQuantity, ...]:
        """The coordinates of ``P(Y | treatment=dose)`` over this spec's dose grid.

        What an observational provider reports for the same grid: the regime is
        ``observational`` and each coordinate conditions on its dose.
        """
        if self.treatment is None:
            raise CausalValueError(
                "this spec has no treatment dose grid; build the observational "
                "coordinates explicitly"
            )
        return tuple(
            replace(
                q,
                regime_id=OBSERVATIONAL,
                conditioning=(QuantityCondition(self.treatment, f"{dose:g}"),),
            )
            for q, dose in zip(self.quantities, self.doses, strict=True)
        )

    def observational_equivalence(self, justification: str) -> Equivalence:
        """The checked equivalence licensing :meth:`observational_quantities`.

        Only call this when ``justification`` names a real check (for example the
        backdoor adjustment set that identified the query); the license is
        yours to assert, and it is recorded in the claim's lineage.
        """
        if self.treatment is None:
            raise CausalValueError("this spec has no treatment dose grid")
        return Equivalence.conditioned(
            self.graph_id,
            self.treatment,
            {f"{dose:g}": q.regime_id for q, dose in zip(self.quantities, self.doses, strict=True)},
            justification,
        )

    def bind(
        self, response: Response, *, program: ProgramBinding | None = None
    ) -> BoundExternalClaim:
        """Check ``response`` against this contract and bind it, or refuse.

        With ``program=`` (an :class:`~antecedent.program_claims.ProgramBinding`) the
        spec is first checked against the identified program: a substituted
        treatment or outcome, another target population, a changed dose grid or
        units, an incompatible ``quantities`` override, or a declared identity that
        is not the program's refuses with ``program_binding.*`` before anything binds.
        """
        if not isinstance(response, Response):
            raise CausalTypeError("ExternalSpec.bind expects an antecedent.external.Response")
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
        if program is None:
            native, refusal = _bind_external_response(
                json.dumps(self._contract_wire()), json.dumps(wire), self.contract_id
            )
        else:
            native, refusal = _bind_external_to_program(
                json.dumps(program._wire()),
                json.dumps(self._program_claim()),
                json.dumps(self._contract_wire()),
                json.dumps(wire),
                self.contract_id,
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
            LineageLink(
                item["id"],
                item["stage"],
                tuple(item["parents"]),
                item["digest"],
                tuple(item["parent_digests"]),
            )
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

    @property
    def source_evidence(self):
        """Independently consumed original attestation; no native authority is issued."""
        from .source_evidence import SourceEvidence

        return SourceEvidence._deferred(lambda: self._native.source_evidence)

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
    from ._native import ResponseAnalysisResult

    raw = getattr(identification, "_native_response", None)
    if isinstance(raw, ResponseAnalysisResult):
        basis = raw.program_basis_json()
        if basis is not None:
            return str(json.loads(basis)["graph_id"])
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
    dose_units: str | None = None,
) -> ExternalSpec:
    """Derive what an identification requires of an external response grid.

    Coordinates come from the identified query (a ``ResponseCurve`` grid becomes
    one ``do(treatment=dose)`` coordinate per point); ``outcome_units`` is
    required because the query cannot know it, and no unit is inferred. Pass
    ``quantities=`` to override the derivation for any other grid. An
    unidentified query refuses at ``bind`` with ``effect_not_identified``.

    Without an explicit ``contract_id=`` the spec's contract identity is durable
    and derived from the full contract and request: for a dose-grid query the
    :attr:`~antecedent.program_claims.ProgramBinding.identity` of the identified
    program (graph, contract premises, treatment, outcome, population, dose grid and
    units), otherwise a digest of the whole contract including its coordinates. It is
    never a hash of the graph alone. ``dose_units`` names the units of the treatment
    grid for that identity (``"unspecified"`` when omitted, and never inferred).
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
    query = identification.query
    dose_grid = (
        quantities is None
        and getattr(query, "kind", None) == "response_curve"
        and getattr(query, "horizons", None) is None
    )
    treatment = query.treatment if dose_grid else None
    doses = tuple(float(dose) for dose in query.grid) if dose_grid else ()
    identity = graph_id or _graph_identity(identification)
    spec = ExternalSpec(
        contract_id=contract_id or "",
        graph_id=identity,
        identification=status,
        quantities=coordinates,
        accepted_meanings=tuple(accepted_meanings),
        require_evidence=tuple(require_evidence),
        require_assumptions=tuple(require_assumptions),
        equivalences=tuple(equivalences),
        statement=identification.statement,
        treatment=treatment,
        doses=doses,
        request=_program_request(
            query,
            outcome_units=outcome_units,
            population=population,
            transform=transform,
            dose_units=dose_units,
        ),
    )
    if contract_id:
        return spec
    faithful = quantities is None or (
        spec.request is not None
        and coordinates
        == _query_quantities(
            query,
            outcome_units=spec.request.outcome_units,
            population=population,
            transform=transform,
        )
    )
    return replace(spec, contract_id=_derived_contract_id(spec, faithful=faithful))


def _program_request(
    query: Any,
    *,
    outcome_units: str | None,
    population: str,
    transform: str,
    dose_units: str | None,
) -> ProgramRequest | None:
    """The identified dose-grid request, or ``None`` for a query that is not one."""
    if (
        getattr(query, "kind", None) != "response_curve"
        or getattr(query, "horizons", None) is not None
        or outcome_units is None
        or not outcome_units.strip()
    ):
        return None
    return ProgramRequest(
        treatment=query.treatment,
        outcome=query.outcome,
        population=population,
        doses=tuple(float(dose) for dose in query.grid),
        dose_units=(dose_units or "").strip() or UNSPECIFIED_UNITS,
        outcome_units=outcome_units,
        transform=transform,
    )


def _derived_contract_id(spec: ExternalSpec, *, faithful: bool) -> str:
    """Durable contract identity from the full contract and request, never the graph alone.

    A spec that faithfully carries the identified dose-grid request takes the identity
    of its program. One whose coordinates differ from that request's, or whose query
    is not a dose grid, takes a digest of the whole contract including its coordinates,
    so it can never share the identity of a request it does not exactly answer.
    """
    if spec.request is None or not faithful:
        return spec._content_id(with_coordinates=True)
    from .program_claims import ProgramBinding

    return ProgramBinding.from_spec(spec).identity


__all__ = [
    "BoundExternalClaim",
    "ClaimInspection",
    "Equivalence",
    "ExternalRefusal",
    "ExternalSpec",
    "LineageLink",
    "ProgramRequest",
    "ProviderObject",
    "QuantityCondition",
    "Response",
    "ScientificQuantity",
    "VerificationProbe",
    "response",
]
