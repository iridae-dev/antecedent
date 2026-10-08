"""Bind claims to the identified program and request they answer.

A causal contract is evidence about one concrete question: this graph, this
treatment and outcome, this target population, this dose grid. A hash of the graph
alone certifies none of those. A :class:`ProgramBinding` names every one of them and
derives one durable identity from all of them; Rust refuses any claim whose
treatment, outcome, population, dose grid or declared identity differs from it, each
with its own typed detail under ``program_binding``::

    ident = ac.identify(graph=edges, names=names, query=ac.ResponseCurve("a", "y", grid=[0, 1, 2]))
    spec = ac.external.response(ident, outcome_units="mmHg", dose_units="mg")
    program = program_claims.ProgramBinding.from_identification(
        ident, outcome_units="mmHg", dose_units="mg"
    )
    claim = program_claims.bind_to_program(spec, program).bind(response)   # or refuses

The same binding turns a native response into a typed :class:`NativeClaim` a decision
or inverse flow can consume::

    claim = program_claims.native_claim(result, program)
    claim.coordinates, claim.support, claim.trust        # native_licensed, never external
    source = claim.as_decision_source(contract)          # a mean source, or a refusal
    decision = contract.evaluate(source.source)

A point or mean response is never an outcome law: asking it for a probability, a
quantile or any criterion that needs a distribution refuses with
``native_claims.source_not_supplied``. Rust owns every identity, check and refusal;
this module builds declarations and raises each refusal as
:class:`~antecedent.external.ExternalRefusal`, a
:class:`~antecedent.errors.CausalUnsupportedError` with its registered ``reason_code``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from ._native import NativeResponseClaim as _NativeResponseClaim
from ._native import check_external_program as _check_external_program
from ._native import native_response_claim as _native_response_claim
from ._native import program_binding_identity as _program_binding_identity
from .decision import Contract, MeanSource
from .errors import CausalUnsupportedError, CausalValueError
from .extensibility import ProviderTrust
from .external import (
    BoundExternalClaim,
    Equivalence,
    ExternalRefusal,
    ExternalSpec,
    Response,
    _status_name,
)
from .external import response as _external_response
from .inverse_query import MeanClaim
from .joint_distribution import DistributionMeaning, JointDistributionArtifact, ScientificQuantity

if TYPE_CHECKING:
    from .results.response import CausalResponseView

Calibration = Literal["unmeasured", "point_only"]
SourceKind = Literal["mean", "joint_draws"]


def _raise(refusal: str | None) -> None:
    if refusal is not None:
        raise ExternalRefusal(json.loads(refusal))


@dataclass(frozen=True, slots=True)
class ProgramCheck:
    """A spec that passed every program check."""

    #: The verified durable identity of the full program and request.
    identity: str
    #: Number of coordinates verified against the program.
    coordinates: int


@dataclass(frozen=True, slots=True)
class ProgramBinding:
    """The full identified program and request an external or native claim must answer.

    ``contract_id`` identifies the contract's premises (graph, identification
    status, accepted meanings, required evidence and assumptions, equivalences);
    :meth:`from_spec` and :meth:`from_identification` derive it. ``identity`` is a
    BLAKE3 digest over every field, so changing any one of them, including the
    contract, changes it. ``dose_units`` and ``outcome_units`` are never inferred or
    converted.
    """

    graph_id: str
    contract_id: str
    treatment_id: str
    outcome_id: str
    population_id: str
    dose_grid: tuple[float, ...]
    dose_units: str
    outcome_units: str
    intervention_kind: str = "do"
    horizon: int = 0
    functional_id: str = "mean"
    transform_id: str = "identity"
    _contract_premises_json: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        grid = tuple(float(dose) for dose in self.dose_grid)
        if not all(math.isfinite(dose) for dose in grid):
            raise CausalValueError("dose_grid must be finite")
        object.__setattr__(self, "dose_grid", grid)

    def _wire(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "contract_id": self.contract_id,
            "treatment_id": self.treatment_id,
            "outcome_id": self.outcome_id,
            "population_id": self.population_id,
            "intervention_kind": self.intervention_kind,
            "horizon": self.horizon,
            "dose_grid": list(self.dose_grid),
            "dose_units": self.dose_units,
            "outcome_units": self.outcome_units,
            "functional_id": self.functional_id,
            "transform_id": self.transform_id,
        }

    @property
    def identity(self) -> str:
        """Durable identity of the full program and request; refuses an invalid binding."""
        value, refusal = _program_binding_identity(json.dumps(self._wire()))
        _raise(refusal)
        assert value is not None
        return value

    @classmethod
    def from_spec(cls, spec: ExternalSpec) -> ProgramBinding:
        """The program ``spec`` was derived for: its identified request and contract premises.

        Read from :attr:`ExternalSpec.request`, never from the spec's ``quantities``, so
        a caller's coordinate override cannot change the program it is checked against.
        """
        request = spec._require_request()
        premises = spec._contract_wire()
        del premises["estimand"]
        return cls(
            graph_id=spec.graph_id,
            contract_id=spec._content_id(),
            treatment_id=request.treatment,
            outcome_id=request.outcome,
            population_id=request.population,
            dose_grid=request.doses,
            dose_units=request.dose_units,
            outcome_units=request.outcome_units,
            transform_id=request.transform,
            _contract_premises_json=json.dumps(premises, sort_keys=True),
        )

    @classmethod
    def from_response(
        cls,
        response: CausalResponseView,
        *,
        outcome_units: str,
        dose_units: str,
        population: str = "target",
        transform: str = "identity",
    ) -> ProgramBinding:
        """Derive the identified scientific binding from an actual native response.

        The complete executed contract remains native execution authority. This
        scientific contract also binds external results for the same identified query.
        """
        return cls.from_spec(
            _external_response(
                _identification_from_response(response),
                outcome_units=outcome_units,
                dose_units=dose_units,
                population=population,
                transform=transform,
            )
        )

    @classmethod
    def from_response(
        cls,
        response: CausalResponseView,
        *,
        outcome_units: str,
        dose_units: str,
        population: str = "target",
        transform: str = "identity",
    ) -> ProgramBinding:
        """The program a native response curve itself answers, from its own compiled identity.

        Treatment, outcome and dose grid come from the response; the graph and contract
        identities are the response's compiled ``program_id`` (so changing the executed
        program changes the binding). Units are required and never inferred or converted.
        Use :meth:`from_identification` instead when the binding must carry the contract
        premises (accepted meanings, required evidence) of an external spec.

        Raises:
            CausalValueError: a unit is blank.
            CausalUnsupportedError: the response is not a static single-treatment mean
                curve, or carries no compiled program identity.
        """
        for name, value in (("outcome_units", outcome_units), ("dose_units", dose_units)):
            if not isinstance(value, str) or not value.strip():
                raise CausalValueError(f"{name}= is required: units are never inferred")
        projection = _projection(response)
        program_id = response.program_id
        if not program_id:
            raise CausalUnsupportedError(
                "this response carries no compiled program identity to bind a decision to",
                reason_code="route_not_supported",
            )
        identity = f"program:{program_id}"
        return cls(
            graph_id=identity,
            contract_id=identity,
            treatment_id=projection["treatment"],
            outcome_id=projection["outcome"],
            population_id=population,
            dose_grid=tuple(projection["grid"]),
            dose_units=dose_units,
            outcome_units=outcome_units,
            transform_id=transform,
        )

    @classmethod
    def from_identification(
        cls,
        identification: Any,
        *,
        outcome_units: str,
        dose_units: str,
        population: str = "target",
        transform: str = "identity",
        graph_id: str | None = None,
        accepted_meanings: Sequence[DistributionMeaning] = ("interventional_predictive",),
        require_evidence: Sequence[str] = (),
        require_assumptions: Sequence[str] = (),
        equivalences: Sequence[Equivalence] = (),
    ) -> ProgramBinding:
        """The program an identified ``ResponseCurve`` asks for, derived the way
        :func:`antecedent.external.response` derives its coordinates.

        Units are required and never inferred. The contract premises
        (``accepted_meanings``, ``require_evidence``, ``require_assumptions``,
        ``equivalences``) are those of the spec this program is meant to check.
        """
        return cls.from_spec(
            _external_response(
                identification,
                outcome_units=outcome_units,
                dose_units=dose_units,
                population=population,
                transform=transform,
                graph_id=graph_id,
                accepted_meanings=accepted_meanings,
                require_evidence=require_evidence,
                require_assumptions=require_assumptions,
                equivalences=equivalences,
            )
        )


def check_external_program(spec: ExternalSpec, program: ProgramBinding) -> ProgramCheck:
    """Check ``spec`` against ``program`` without binding anything.

    Refuses with ``program_binding.treatment_outcome_substitution``,
    ``population_mismatch``, ``dose_grid_changed`` (a changed grid or units),
    ``quantities_override_mismatch`` (an incompatible coordinate override),
    ``graph_only_identity``, ``contract_identity_mismatch`` or ``invalid_binding``.
    """
    value, refusal = _check_external_program(
        json.dumps(program._wire()), json.dumps(spec._program_claim())
    )
    _raise(refusal)
    assert value is not None
    body = json.loads(value)
    return ProgramCheck(identity=str(body["identity"]), coordinates=int(body["coordinates"]))


@dataclass(frozen=True, slots=True)
class ProgramBoundSpec:
    """An :class:`~antecedent.external.ExternalSpec` verified against its program.

    ``bind`` binds a provider response under both the checked contract and the
    program, and returns an ordinary external claim: it stays external and keeps its
    external trust.
    """

    spec: ExternalSpec
    program: ProgramBinding
    #: The verified durable identity of the full program and request.
    identity: str

    def bind(self, response: Response) -> BoundExternalClaim:
        return self.spec.bind(response, program=self.program)


def bind_to_program(spec: ExternalSpec, program: ProgramBinding) -> ProgramBoundSpec:
    """Verify ``spec`` against ``program`` now, or refuse; see :func:`check_external_program`."""
    checked = check_external_program(spec, program)
    return ProgramBoundSpec(spec, program, checked.identity)


@dataclass(frozen=True, slots=True)
class WithheldCoordinate:
    """A coordinate left out of a decision source for lack of support."""

    index: int
    coordinate: ScientificQuantity
    status: str


@dataclass(frozen=True, slots=True)
class NativeDecisionSource:
    """What the decision engine receives from a native claim, with its provenance.

    ``source`` is a :class:`~antecedent.decision.MeanSource` or an actual retained
    joint posterior artifact for ``Contract.evaluate``. ``mean_claim()`` supplies
    point means as an inverse-query forward claim. Coordinates outside the empirical region or
    lacks evidence are in ``withheld``, never silently used.
    """

    representation: SourceKind
    coordinates: tuple[ScientificQuantity, ...]
    point_status: tuple[str, ...]
    withheld: tuple[WithheldCoordinate, ...]
    trust: ProviderTrust
    calibration: str
    program_identity: str
    source: MeanSource | JointDistributionArtifact

    def mean_claim(self) -> MeanClaim:
        """The same means as an :class:`~antecedent.inverse_query.MeanClaim`."""
        if not isinstance(self.source, MeanSource):
            raise CausalUnsupportedError(
                "a joint law requires a distribution consumer", reason_code="route_not_supported"
            )
        return MeanClaim(
            coordinates=self.source.coordinates,
            means=self.source.means,
            provider_id=self.source.provider_id,
            snapshot_id=self.source.snapshot_id,
            causal_contract_id=self.source.causal_contract_id,
            rng_id=self.source.rng_id,
            _original_native=self.source._original_native,
        )

    @classmethod
    def _from_wire(
        cls,
        body: Mapping[str, Any],
        joint: JointDistributionArtifact | None = None,
        original: Any = None,
    ) -> NativeDecisionSource:
        mean = body.get("mean_source")
        if mean is None and joint is None:
            raise CausalUnsupportedError(
                "native joint state is unavailable", reason_code="route_not_supported"
            )
        mean = {} if mean is None else mean
        coordinates = tuple(ScientificQuantity._from_wire(q) for q in body["coordinates"])
        return cls(
            representation="joint_draws" if joint is not None else "mean",
            coordinates=coordinates,
            point_status=tuple(body["point_status"]),
            withheld=tuple(
                WithheldCoordinate(
                    item["index"], ScientificQuantity._from_wire(item["coordinate"]), item["status"]
                )
                for item in body["withheld"]
            ),
            trust=ProviderTrust(body["trust"]),
            calibration=str(body["calibration"]),
            program_identity=str(body["program_identity"]),
            source=joint
            if joint is not None
            else MeanSource(
                coordinates=coordinates,
                means=tuple(mean["means"]),
                provider_id=str(mean["provider_id"]),
                snapshot_id=str(mean["snapshot_id"]),
                causal_contract_id=str(mean["causal_contract_id"]),
                rng_id=str(mean["rng_id"]),
                _original_native=original,
            ),
        )


class NativeClaim:
    """A native response as a typed claim: coordinates, support, trust and provenance.

    Built by :func:`native_claim` from a response this library produced. Trust is
    always ``native_licensed`` because the claim is built in process from a typed
    response, never from artifact metadata; ``calibration`` is ``unmeasured`` or
    ``point_only`` and is never upgraded here.
    """

    def __init__(self, native: _NativeResponseClaim) -> None:
        self._native = native

    @property
    def coordinates(self) -> tuple[ScientificQuantity, ...]:
        """One scientific coordinate per grid point, in grid order."""
        wires = json.loads(self._native.coordinates_json)
        return tuple(ScientificQuantity._from_wire(wire) for wire in wires)

    @property
    def means(self) -> tuple[float, ...]:
        """One mean per coordinate, in grid order."""
        return tuple(self._native.means)

    @property
    def support(self) -> tuple[str, ...]:
        """Support label of each coordinate."""
        return tuple(self._native.point_status)

    @property
    def support_status(self) -> str:
        """The worst per-coordinate label."""
        return self._native.support_status

    @property
    def trust(self) -> ProviderTrust:
        """Always ``native_licensed``: the claim is built in process, never from metadata."""
        return ProviderTrust(self._native.trust)

    @property
    def calibration(self) -> str:
        """``unmeasured`` or ``point_only``; never upgraded here."""
        return self._native.calibration

    @property
    def program_identity(self) -> str:
        """Durable identity of the full program and request this response was checked against."""
        return self._native.program_identity

    @property
    def lineage(self):
        """Original typed source ancestry; querying it consumes the original source artifact."""
        return self.source_evidence.lineage

    def stages_behind(self, link: str = "claim") -> frozenset[str]:
        return self.source_evidence.stages_behind(link)

    @property
    def source_evidence(self):
        """Actual original scoped diagnostics and independently consumable source."""
        from .source_evidence import SourceEvidence

        return SourceEvidence._deferred(lambda: self._native.source_evidence)

    @property
    def execution_program_id(self) -> str | None:
        """Actual checked executed program, separate from scientific declarations."""
        return self._native.execution_program_id

    @property
    def provenance_id(self) -> str:
        """Identity of the native execution that produced the means."""
        return self._native.provenance_id

    @property
    def has_joint_law(self) -> bool:
        """Whether the response retained aligned credible draws of every coordinate."""
        return bool(self._native.has_joint_law)

    def as_decision_source(self, requirement: Contract) -> NativeDecisionSource:
        """The source ``requirement`` (a decision contract) needs, or the Rust refusal.

        A mean source is available for an affine expectation. A criterion that needs
        a distribution (a probability, a quantile, a nonlinear utility) refuses with
        ``native_claims.source_not_supplied``; a point or mean response is not an
        outcome law. Every coordinate lacking support refuses with
        ``native_claims.no_supported_coordinate``.
        """
        body, refusal = self._native.decision_source(json.dumps(requirement._wire()))
        _raise(refusal)
        assert body is not None
        decoded = json.loads(body)
        joint = None
        if "joint_law" in decoded:
            artifact, refusal = self._native.joint_artifact(json.dumps(requirement._wire()))
            _raise(refusal)
            assert artifact is not None
            joint = JointDistributionArtifact._from_native(artifact)
        return NativeDecisionSource._from_wire(decoded, joint, self)

    def explain(self) -> str:
        """What the numbers are, that Antecedent produced them, and how far they are trusted."""
        joint = "retains" if self.has_joint_law else "does not retain"
        return (
            f"{len(self.means)} mean(s) estimated natively by Antecedent for program "
            f"{self.program_identity[:12]}. Provider trust: {self.trust.value}; calibration: "
            f"{self.calibration}; worst support: {self.support_status}. The response {joint} "
            "aligned draws, so it answers an affine expectation but not a distribution; "
            "calibration is never upgraded here."
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form: means, support, trust and provenance."""
        return {
            "means": list(self.means),
            "coordinates": [q._wire() for q in self.coordinates],
            "support": list(self.support),
            "support_status": self.support_status,
            "provider_trust": self.trust.value,
            "calibration": self.calibration,
            "program_identity": self.program_identity,
            "provenance_id": self.provenance_id,
            "has_joint_law": self.has_joint_law,
        }

    def __repr__(self) -> str:
        return (
            f"<NativeClaim {len(self.means)} coordinates trust={self.trust.value} "
            f"support={self.support_status} calibration={self.calibration}>"
        )


def _unsupported_estimand(supplied: str) -> ExternalRefusal:
    return ExternalRefusal(
        {
            "code": "route_not_supported",
            "stage": "bind",
            "detail": "native_claims.unsupported_estimand",
            "offending": None,
            "expected": "mean_curve",
            "supplied": supplied,
            "remedy": "only a static mean response curve carries one coordinate per dose",
        }
    )


@dataclass(frozen=True, slots=True)
class _NativeProgramIdentification:
    query: Any
    status: str
    statement: str
    _native_response: Any
    graph: tuple[()] = ()
    names: tuple[()] = ()
    identifier: str = "checked.native"
    method: str = "checked.native"
    adjustment_set: tuple[()] = ()


def _identification_from_response(response: CausalResponseView) -> _NativeProgramIdentification:
    from .query import ResponseCurve

    raw = response._raw
    value = None if raw is None else raw.program_basis_json()
    if value is None:
        raise ExternalRefusal(
            {
                "code": "invalid_argument",
                "stage": "bind",
                "detail": "native_claims.native_state_unavailable",
                "remedy": "use a response issued by actual checked native execution",
            }
        )
    basis = json.loads(value)
    return _NativeProgramIdentification(
        ResponseCurve(basis["treatment"], basis["outcome"], grid=basis["grid"]),
        basis["status"],
        basis["statement"],
        raw,
    )


def _projection(response: CausalResponseView) -> dict[str, Any]:
    """The response's own estimand, grid, means, support and identification, as data."""
    query = response.estimand
    view = response.response
    if (
        getattr(query, "kind", None) != "response_curve"
        or getattr(query, "horizons", None) is not None
    ):
        raise _unsupported_estimand(str(getattr(query, "kind", type(query).__name__)))
    if (
        view is None
        or len(view.treatments) != 1
        or len(view.outcomes) != 1
        or any(len(point) != 1 for point in view.points)
    ):
        raise _unsupported_estimand("non_static_or_multivariate_response")
    return {
        "treatment": view.treatments[0],
        "outcome": view.outcomes[0],
        "grid": [point[0] for point in view.points],
        "means": [row[0] for row in view.values],
        "identification": _status_name(str(response.identification.status)),
        "has_envelope": response.envelope is not None,
        "support_status": response.support.status,
        "point_status": None
        if response.support.point_status is None
        else list(response.support.point_status),
        "provenance_id": str(response.provenance.get("operation_id") or ""),
        "uncertainty": response.uncertainty.model_dump(mode="json"),
        "data_snapshot_id": response.data_snapshot_id,
        "program_id": response.program_id,
    }


def native_claim(
    response: CausalResponseView,
    program: ProgramBinding,
    *,
    snapshot_id: str | None = None,
    rng_id: str | None = None,
    calibration: Calibration | None = None,
) -> NativeClaim:
    """The typed claim a native response makes about the program it answers.

    Coordinates come from the response's own treatment, outcome and grid and are
    checked against ``program``: a response computed for another outcome, treatment,
    dose grid or population refuses with ``program_binding.*`` instead of being
    relabeled, and a schema unit is never converted. ``snapshot_id`` defaults to the
    response's data snapshot. ``calibration`` defaults to ``point_only`` for a
    response with no uncertainty and ``unmeasured`` otherwise; only those two are
    accepted because a calibration status comes from a measured record, not from the
    caller. A response without a label for every coordinate, or one that is not point
    identified, refuses.
    """
    projection = _projection(response)
    binding_json = json.dumps(program._wire())
    if response.quantities is not None:
        declared = {
            "contract_id": program.contract_id,
            "graph_id": program.graph_id,
            "declared_identity": program.identity,
            "treatment_id": projection["treatment"],
            "outcome_id": projection["outcome"],
            "population_id": response.quantities[0].population_id
            if response.quantities
            else program.population_id,
            "doses": projection["grid"],
            "dose_units": program.dose_units,
            "quantities": [quantity._wire() for quantity in response.quantities],
        }
        # Rust owns the full coordinate comparison and registered refusals.
        # A descriptor-bearing response cannot be relabelled by a new program.
        _, refusal = _check_external_program(binding_json, json.dumps(declared))
        _raise(refusal)
    claim, refusal = _native_response_claim(
        response._raw,
        json.dumps(projection),
        binding_json,
        snapshot_id,
        rng_id,
        calibration,
        program._contract_premises_json,
    )
    _raise(refusal)
    assert claim is not None
    return NativeClaim(claim)


__all__ = [
    "Calibration",
    "NativeClaim",
    "NativeDecisionSource",
    "ProgramBinding",
    "ProgramBoundSpec",
    "ProgramCheck",
    "WithheldCoordinate",
    "bind_to_program",
    "check_external_program",
    "native_claim",
]
