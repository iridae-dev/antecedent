"""Generalized finite inverse decision query (F7).

"Which of these declared actions satisfy these constraints?" over a finite, ordered
action grid. Target-mean, target-quantile and probability-threshold constraints all
run through the one typed functional engine that evaluates every other decision
functional, on a forward claim that actually supplies the law each one needs::

    contract = decision.Contract(actions=[...], utility_units="units", ...)
    query = inverse_query.InverseQuery(
        contract,
        grid=("a0", "a1", "a2"),                      # the declared order
        constraints=[inverse_query.target_quantile(0.5, 1.0, direction="at_least")],
        selection="first_in_grid_order",
    )
    result = query.evaluate(point=joint_distribution)
    result.selected, result.feasible_actions, result.selection_certified

What each claim can answer:

* a :class:`~antecedent.joint_distribution.JointDistributionArtifact` with aligned
  joint draws answers all three constraints, including a nonlinear utility;
* a :class:`MeanClaim` (or a bound external response grid) answers an affine target
  mean and nothing else: asking it for a quantile or a probability refuses with
  ``decision_evaluation.mean_source_insufficient``, because a mean never yields an
  outcome probability or quantile;
* independent marginals refuse a nonlinear utility
  (``decision_evaluation.joint_law_required``).

The quantile is the engine's left inverse ``inf {x : F(x) >= p}`` of the weighted
CDF (no interpolation); ``P(U <= t)`` and ``P(U >= t)`` both include the atom at
``t``. The action grid is enumerated in the declared order, which is also the order
of the multiple-action rule. Feasibility is reported per action in distinct fields:
``point`` (the single forward claim), ``interval_region``, ``identified_set``,
``all_scenario`` and ``posterior_probability``; none is derived from another, and
an absent kind of evidence is ``None``, never ``feasible``. A ``continuous_sample``
grid can find a feasible point, but one found point is never a global feasibility
claim. No coverage or calibration claim is made here.

Rust owns the engine, the artifact and every refusal; this module builds
declarations and raises each refusal as :class:`InverseQueryRefusal`, a
:class:`~antecedent.errors.CausalUnsupportedError` with its registered
``reason_code`` and the engine's own ``detail``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

from ._native import InverseQueryArtifact as _NativeInverseQueryArtifact
from ._native import inverse_query_baseline as _baseline
from .decision import Contract
from .errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from .external import BoundExternalClaim
from .joint_distribution import JointDistributionArtifact, ScientificQuantity

if TYPE_CHECKING:
    from .program_claims import NativeClaim

Direction = Literal["at_least", "at_most"]
Tail = Literal["lower", "upper"]
GridScope = Literal["finite_enumeration", "continuous_sample"]
Selection = Literal["first_in_grid_order", "last_in_grid_order", "require_unique"]
SelectionOutcome = Literal["selected", "no_feasible_action", "multiple_feasible", "no_point_claim"]
Feasibility = Literal[
    "feasible",
    "infeasible",
    "unsupported",
    "unevaluated",
    "structurally_ambiguous",
    "unidentified",
]
FEASIBILITY_STATUSES: tuple[str, ...] = (
    "feasible",
    "infeasible",
    "unsupported",
    "unevaluated",
    "structurally_ambiguous",
    "unidentified",
)


class InverseQueryRefusal(CausalUnsupportedError):
    """An inverse-query refusal carrying the structured Rust fields.

    ``reason_code`` is registered. ``detail`` is the namespaced ``family.slot``: the
    engine's own detail for a forward claim that cannot answer a constraint
    (``decision_evaluation.joint_law_required``,
    ``decision_evaluation.mean_source_insufficient``,
    ``decision_evaluation.meaning_mismatch``), ``inverse_query.*`` for a malformed
    query, or ``functional_inverse_query.*`` for an artifact (for example
    ``functional_inverse_query.global_feasibility_claim`` for a continuous sample that
    claims global feasibility, ``functional_inverse_query.wrong_contract`` for a
    resealed mutation).
    """

    def __init__(self, refusal: Mapping[str, Any]) -> None:
        offending = refusal.get("offending")
        text = str(refusal["detail"]) + (f": {offending}" if offending else "")
        super().__init__(text, reason_code=refusal["code"], remedy=refusal.get("remedy"))
        self.stage: str = refusal.get("stage", "")
        self.detail: str = refusal["detail"]
        self.offending: str | None = offending
        self.expected: str | None = refusal.get("expected")
        self.supplied: str | None = refusal.get("supplied")


def _raise(refusal: str | None) -> None:
    if refusal is not None:
        raise InverseQueryRefusal(json.loads(refusal))


def _direction(value: str) -> Direction:
    if value not in ("at_least", "at_most"):
        raise CausalValueError("direction is 'at_least' or 'at_most'")
    return value  # type: ignore[return-value]


# ------------------------------------------------------------------- constraints


@dataclass(frozen=True, slots=True)
class Constraint:
    """One typed constraint on an action's forward utility law; build with the helpers."""

    kind: Literal["target_mean", "target_quantile", "probability_threshold"]
    direction: Direction
    target: float | None = None
    p: float | None = None
    outcome_threshold: float | None = None
    tail: Tail | None = None
    probability: float | None = None

    def _wire(self) -> dict[str, Any]:
        if self.kind == "target_mean":
            body: dict[str, Any] = {"target": self.target}
        elif self.kind == "target_quantile":
            body = {"p": self.p, "target": self.target}
        else:
            body = {
                "outcome_threshold": self.outcome_threshold,
                "tail": self.tail,
                "probability": self.probability,
            }
        return {self.kind: {**body, "comparison": self.direction}}

    @classmethod
    def _from_wire(cls, wire: Mapping[str, Any]) -> Constraint:
        ((kind, body),) = wire.items()
        return cls(
            kind=cast("Literal['target_mean', 'target_quantile', 'probability_threshold']", kind),
            direction=body["comparison"],
            target=body.get("target"),
            p=body.get("p"),
            outcome_threshold=body.get("outcome_threshold"),
            tail=body.get("tail"),
            probability=body.get("probability"),
        )

    def __str__(self) -> str:
        sign = ">=" if self.direction == "at_least" else "<="
        if self.kind == "target_mean":
            return f"E[U] {sign} {self.target:g}"
        if self.kind == "target_quantile":
            return f"Q_{self.p:g}(U) {sign} {self.target:g}"
        side = "<=" if self.tail == "lower" else ">="
        return f"P(U {side} {self.outcome_threshold:g}) {sign} {self.probability:g}"


def _finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise CausalValueError(f"{name} must be a finite number", reason_code="invalid_argument")
    return float(value)


def target_mean(target: float, *, direction: Direction = "at_least") -> Constraint:
    """``E[U_a]`` at least (or at most) ``target``; answerable from means when affine."""
    return Constraint("target_mean", _direction(direction), target=_finite("target", target))


def target_quantile(p: float, target: float, *, direction: Direction = "at_least") -> Constraint:
    """The left-inverse ``p``-quantile of ``U_a`` at least (or at most) ``target``.

    Needs a forward claim that supplies the law; a mean never answers it.
    """
    return Constraint(
        "target_quantile",
        _direction(direction),
        target=_finite("target", target),
        p=_finite("p", p),
    )


def probability_threshold(
    outcome_threshold: float,
    probability: float,
    *,
    tail: Tail = "upper",
    direction: Direction = "at_least",
) -> Constraint:
    """``P(U_a >= t)`` (``tail="upper"``) or ``P(U_a <= t)`` (``"lower"``) against ``probability``.

    Both tails include the atom at ``t``. Needs a forward claim that supplies the
    law; a mean posterior never yields an outcome probability.
    """
    if tail not in ("lower", "upper"):
        raise CausalValueError("tail is 'lower' or 'upper'")
    return Constraint(
        "probability_threshold",
        _direction(direction),
        outcome_threshold=_finite("outcome_threshold", outcome_threshold),
        tail=tail,
        probability=_finite("probability", probability),
    )


# ------------------------------------------------------------------------ claims


@dataclass(frozen=True, slots=True)
class MeanClaim:
    """A forward claim that supplies only the mean of each coordinate.

    It answers an affine target mean and nothing else.
    """

    coordinates: tuple[ScientificQuantity, ...]
    means: tuple[float, ...]
    provider_id: str
    snapshot_id: str
    causal_contract_id: str
    rng_id: str = "none:mean_grid"
    _original_external: BoundExternalClaim | None = field(default=None, repr=False, compare=False)
    _original_native: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "coordinates", tuple(self.coordinates))
        object.__setattr__(self, "means", tuple(float(m) for m in self.means))

    @classmethod
    def from_external(cls, claim: BoundExternalClaim) -> MeanClaim:
        """The mean grid of a bound external response."""
        identity = claim.identity_fields
        return cls(
            coordinates=tuple(claim.quantities),
            means=tuple(float(v) for v in claim.values),
            provider_id=str(identity["provider_id"]),
            snapshot_id=str(identity["snapshot_id"]),
            causal_contract_id=str(identity["causal_contract_id"]),
            _original_external=claim,
        )

    def _wire(self) -> dict[str, Any]:
        return {
            "coordinates": [q._wire() for q in self.coordinates],
            "means": list(self.means),
            "provider_id": self.provider_id,
            "snapshot_id": self.snapshot_id,
            "causal_contract_id": self.causal_contract_id,
            "rng_id": self.rng_id,
        }


if TYPE_CHECKING:
    ForwardClaim: TypeAlias = (
        JointDistributionArtifact | MeanClaim | BoundExternalClaim | NativeClaim
    )
else:
    ForwardClaim = "JointDistributionArtifact | MeanClaim | BoundExternalClaim | NativeClaim"


def _claim(claim: ForwardClaim) -> Any:
    from .program_claims import NativeClaim

    if isinstance(claim, NativeClaim):
        return claim._native
    if isinstance(claim, JointDistributionArtifact):
        return claim._native
    if isinstance(claim, BoundExternalClaim):
        return claim._native
    if isinstance(claim, MeanClaim):
        if claim._original_native is not None:
            from . import _native

            return _native.source_backed_native_mean(
                claim._original_native._native, json.dumps(claim._wire())
            )
        if claim._original_external is not None:
            if claim._wire() != MeanClaim.from_external(claim._original_external)._wire():
                raise CausalValueError(
                    "source_evidence.point_binding_mismatch: changed mean declaration differs from original external claim",
                    reason_code="invalid_argument",
                )
            return claim._original_external._native
        return json.dumps(claim._wire())
    raise CausalTypeError(
        "a forward claim is a JointDistributionArtifact, a MeanClaim or a bound external claim"
    )


@dataclass(frozen=True, slots=True)
class Scenario:
    """A structural scenario or identified-set member.

    Build with :meth:`evaluated`, :meth:`unidentified` or :meth:`unevaluated`.
    ``probability`` is a genuine probability, supplied for every scenario or for none;
    scenario completion counts are not probabilities.
    """

    id: str
    state: Literal["evaluated", "unidentified", "unevaluated"]
    law: JointDistributionArtifact | None = None
    reason: str | None = None
    probability: float | None = None

    @classmethod
    def evaluated(
        cls,
        id: str,
        law: JointDistributionArtifact,
        probability: float | None = None,  # noqa: A002
    ) -> Scenario:
        return cls(id, "evaluated", law=law, probability=probability)

    @classmethod
    def unidentified(cls, id: str, probability: float | None = None) -> Scenario:  # noqa: A002
        return cls(id, "unidentified", probability=probability)

    @classmethod
    def unevaluated(
        cls,
        id: str,
        reason: str,
        probability: float | None = None,  # noqa: A002
    ) -> Scenario:
        return cls(id, "unevaluated", reason=reason, probability=probability)

    def _tuple(self) -> tuple[str, float | None, str, Any]:
        if self.state == "evaluated":
            if not isinstance(self.law, JointDistributionArtifact):
                raise CausalTypeError("an evaluated scenario carries a JointDistributionArtifact")
            return (self.id, self.probability, "evaluated", self.law._native)
        if self.state == "unevaluated":
            return (self.id, self.probability, "unevaluated", self.reason or "unevaluated")
        return (self.id, self.probability, "unidentified", None)


@dataclass(frozen=True, slots=True)
class IntervalRegion:
    """A published interval region, as its two endpoint claims.

    ``endpoints_bound_functional`` is the caller's declaration that the endpoint laws
    bound the functional over the whole region; it is not verified. Without it an
    endpoint answer is ``unevaluated``.
    """

    lower: ForwardClaim
    upper: ForwardClaim
    endpoints_bound_functional: bool = False


@dataclass(frozen=True, slots=True)
class IdentifiedSet:
    """Member laws of an identified set; ``exhaustive`` declares they enumerate the whole set."""

    members: tuple[Scenario, ...]
    exhaustive: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "members", tuple(self.members))


# ------------------------------------------------------------------------ result


@dataclass(frozen=True, slots=True)
class ConstraintValue:
    """One constraint's value for one action and claim."""

    constraint: int
    value: float | None
    standard_error: float | None
    status: Feasibility
    reason: str | None


@dataclass(frozen=True, slots=True)
class MemberStatus:
    """One scenario or set member's status for one action."""

    id: str
    status: Feasibility
    reason: str | None


@dataclass(frozen=True, slots=True)
class PosteriorFeasibility:
    """Probability mass over scenarios for one action; never renormalized."""

    feasible_mass: float
    infeasible_mass: float
    unresolved_mass: float


@dataclass(frozen=True, slots=True)
class ActionReport:
    """One action across every feasibility field; each field is independent of the others."""

    id: str
    position: int
    point: Feasibility | None
    point_values: tuple[ConstraintValue, ...]
    interval_region: Feasibility | None
    identified_set: Feasibility | None
    identified_set_members: tuple[MemberStatus, ...]
    all_scenario: Feasibility | None
    scenario_members: tuple[MemberStatus, ...]
    posterior_probability: PosteriorFeasibility | None

    @classmethod
    def _from_wire(cls, wire: Mapping[str, Any]) -> ActionReport:
        posterior = wire["posterior_probability"]
        return cls(
            id=wire["id"],
            position=int(wire["position"]),
            point=wire["point"],
            point_values=tuple(
                ConstraintValue(
                    int(v["constraint"]), v["value"], v["standard_error"], v["status"], v["reason"]
                )
                for v in wire["point_values"]
            ),
            interval_region=wire["interval_region"],
            identified_set=wire["identified_set"],
            identified_set_members=tuple(
                MemberStatus(m["id"], m["status"], m["reason"])
                for m in wire["identified_set_members"]
            ),
            all_scenario=wire["all_scenario"],
            scenario_members=tuple(
                MemberStatus(m["id"], m["status"], m["reason"]) for m in wire["scenario_members"]
            ),
            posterior_probability=None if posterior is None else PosteriorFeasibility(**posterior),
        )


@dataclass(frozen=True, slots=True)
class SourceReceipt:
    """Lineage of the point claim."""

    provider_id: str
    snapshot_id: str
    rng_id: str
    causal_contract_id: str


class InverseResult:
    """The answer to an inverse query, with the evidence it stands on; exportable."""

    def __init__(self, query: InverseQuery, native: _NativeInverseQueryArtifact) -> None:
        self._query = query
        self._native = native
        self._body: dict[str, Any] = json.loads(native.result_json)

    @property
    def source_evidence(self):
        """Original diagnostics and source bytes retained through native inverse evaluation."""
        from .source_evidence import SourceEvidence

        return tuple(SourceEvidence(handle) for handle in self._native.source_evidence)

    @property
    def query(self) -> InverseQuery:
        """The query this result answers."""
        return self._query

    @property
    def identity(self) -> dict[str, str]:
        """``premises_digest``, ``data_digest`` and ``digest``: retain it to consume."""
        return dict(json.loads(self._native.identity_json))

    @property
    def contract_identity(self) -> str:
        """Identity of the decision contract the query ranges over."""
        return str(self._body["contract_identity"])

    @property
    def actions(self) -> tuple[ActionReport, ...]:
        """Per-action reports in the declared grid order."""
        return tuple(ActionReport._from_wire(a) for a in self._body["actions"])

    @property
    def feasible_actions(self) -> tuple[str, ...]:
        """Point-feasible actions in grid order."""
        return tuple(self._body["feasible_actions"])

    @property
    def selected(self) -> str | None:
        """The action the selection rule picks, when it picks one."""
        return self._body["selected"]

    @property
    def selection(self) -> SelectionOutcome:
        """How selection ended: one action selected, none feasible, several feasible
        under ``require_unique``, or no point claim."""
        return self._body["selection"]

    @property
    def selection_certified(self) -> bool:
        """``True`` only when every action the rule passed over is definitely infeasible,
        never merely unevaluated or unsupported."""
        return bool(self._body["selection_certified"])

    @property
    def grid_fully_decided(self) -> bool:
        """Whether every grid action has a decided (feasible or infeasible) point status."""
        return bool(self._body["grid_fully_decided"])

    @property
    def existence(
        self,
    ) -> Literal["found_feasible_action", "no_feasible_action_in_declared_set", "undetermined"]:
        """What the point field says about existence: a found witness, none in a decided
        finite enumeration, or nothing."""
        return self._body["existence"]

    @property
    def exhaustive(self) -> bool:
        """``True`` only for a finite enumeration whose every action is decided; always
        ``False`` for a continuous sample."""
        return bool(self._body["exhaustive_over_declared_set"])

    @property
    def evaluations_used(self) -> int:
        """How many forward evaluations were spent."""
        return int(self._body["evaluations_used"])

    @property
    def budget_exhausted(self) -> bool:
        """Whether the evaluation budget stopped the search before it finished."""
        return bool(self._body["budget_exhausted"])

    @property
    def point_source(self) -> SourceReceipt | None:
        """Lineage (provider and snapshot) of the point claim, or ``None`` without one."""
        wire = self._body["point_source"]
        return None if wire is None else SourceReceipt(**wire)

    @property
    def scope_note(self) -> str:
        """What the answer does and does not claim."""
        return str(self._body["scope_note"])

    def action(self, id: str) -> ActionReport:  # noqa: A002
        """The report of one action by id."""
        for report in self.actions:
            if report.id == id:
                return report
        raise CausalValueError(f"unknown action {id!r}")

    def explain(self) -> str:
        """The answer, how it was decided and what it does not claim."""
        constraints = " and ".join(str(c) for c in self._query.constraints)
        grid = ", ".join(self._query.grid)
        if self.selected is not None:
            text = f"{self.selected!r} is selected for {constraints} over [{grid}]"
            text += (
                " (every action passed over is definitely infeasible)"
                if self.selection_certified
                else " (not certified: an action passed over is not definitely infeasible)"
            )
        elif self.selection == "no_point_claim":
            text = f"no point claim was supplied for {constraints} over [{grid}]"
        elif self.selection == "multiple_feasible":
            text = f"several actions are feasible for {constraints} over [{grid}]: " + ", ".join(
                repr(a) for a in self.feasible_actions
            )
        else:
            text = f"no action is point-feasible for {constraints} over [{grid}]"
        if self.existence == "no_feasible_action_in_declared_set":
            text += "; every action of the finite enumeration is decided and none is feasible"
        elif self.existence == "found_feasible_action" and not self.exhaustive:
            text += "; a feasible point was found, which is not a global feasibility claim"
        return text + ". " + self.scope_note

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form: selection, existence, per-action feasibility fields and lineage."""
        source = self.point_source
        return {
            "contract_identity": self.contract_identity,
            "identity": self.identity,
            "selected": self.selected,
            "selection": self.selection,
            "selection_certified": self.selection_certified,
            "feasible_actions": list(self.feasible_actions),
            "grid_fully_decided": self.grid_fully_decided,
            "existence": self.existence,
            "exhaustive": self.exhaustive,
            "evaluations_used": self.evaluations_used,
            "budget_exhausted": self.budget_exhausted,
            "point_source": None if source is None else asdict(source),
            "actions": [asdict(a) for a in self.actions],
            "scope_note": self.scope_note,
        }

    def export(self, *, artifact_id: str = "inverse-query") -> bytes:
        """The query, its forward evidence and this result table, replayable.

        Embeds the finite laws and means compactly, with premises and data digests kept
        apart. A consumer re-evaluates it through the same engine.
        """
        return bytes(self._native.export(artifact_id))

    @classmethod
    def consume(
        cls, data: bytes, *, expected_identity: Mapping[str, str] | None = None
    ) -> InverseResult:
        """Consume by re-evaluation; refuses a resealed mutation.

        The contract, query and evidence are rebuilt and evaluated again; the
        recomputed identity and result table must equal the stored ones, and, when the
        consumer retained ``expected_identity`` (:attr:`identity`) independently of the
        bytes, that identity must match too. A continuous sample that stores a global
        feasibility claim refuses with
        ``functional_inverse_query.global_feasibility_claim``.
        """
        if not isinstance(data, bytes):
            raise CausalTypeError("artifact must be bytes")
        native, refusal = _NativeInverseQueryArtifact.consume(
            data, None if expected_identity is None else json.dumps(dict(expected_identity))
        )
        _raise(refusal)
        assert native is not None
        contract = Contract._from_wire(json.loads(native.contract_json))
        return cls(InverseQuery._from_wire(contract, json.loads(native.query_json)), native)

    def __repr__(self) -> str:
        return f"<InverseResult selected={self.selected!r} feasible={list(self.feasible_actions)}>"


# ------------------------------------------------------------------------- query


@dataclass(frozen=True, slots=True)
class InverseQuery:
    """A typed inverse decision query over a declared, ordered, finite action grid.

    ``contract`` declares the actions, their input quantities and their utilities (its
    criterion and hard constraints are not used by this query). ``grid`` is the
    ordered list of action ids; the order is the multiple-action rule's order, not an
    id sort. ``constraints`` are a conjunction. ``selection`` resolves several
    feasible actions: ``first_in_grid_order``, ``last_in_grid_order`` or
    ``require_unique``. ``tolerance`` is non-negative slack on each comparison, for
    roundoff in exact laws. ``max_evaluations`` bounds the (action, claim,
    constraint) functional evaluations; the rest stay ``unevaluated``. ``scope`` is
    ``finite_enumeration`` or ``continuous_sample`` (points sampled from a continuous
    domain, which never yields a global claim).
    """

    contract: Contract
    grid: tuple[str, ...]
    constraints: tuple[Constraint, ...]
    scope: GridScope = "finite_enumeration"
    selection: Selection = "first_in_grid_order"
    tolerance: float = 1e-12
    max_evaluations: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "grid", tuple(self.grid))
        object.__setattr__(self, "constraints", tuple(self.constraints))
        if not isinstance(self.contract, Contract):
            raise CausalTypeError("contract must be a decision.Contract")
        if any(not isinstance(c, Constraint) for c in self.constraints):
            raise CausalTypeError(
                "constraints are built with target_mean, target_quantile or probability_threshold"
            )

    def _wire(self) -> dict[str, Any]:
        return {
            "grid_order": list(self.grid),
            "grid_scope": self.scope,
            "constraints": [c._wire() for c in self.constraints],
            "selection": self.selection,
            "tolerance": float(self.tolerance),
            "max_evaluations": self.max_evaluations,
        }

    @classmethod
    def _from_wire(cls, contract: Contract, wire: Mapping[str, Any]) -> InverseQuery:
        return cls(
            contract=contract,
            grid=tuple(wire["grid_order"]),
            constraints=tuple(Constraint._from_wire(c) for c in wire["constraints"]),
            scope=wire["grid_scope"],
            selection=wire["selection"],
            tolerance=float(wire["tolerance"]),
            max_evaluations=wire["max_evaluations"],
        )

    def evaluate(
        self,
        point: ForwardClaim | None = None,
        *,
        interval_region: IntervalRegion | None = None,
        identified_set: IdentifiedSet | None = None,
        scenarios: Sequence[Scenario] | None = None,
    ) -> InverseResult:
        """Evaluate on forward evidence, one optional slot per feasibility field.

        ``point`` answers the ``point`` field (a native law, a :class:`MeanClaim` or a
        bound external response). ``interval_region``, ``identified_set`` and
        ``scenarios`` answer their own fields; ``posterior_probability`` is reported
        only when every scenario carries a genuine probability. Nothing is derived
        from another field.

        Refuses with :class:`InverseQueryRefusal` an invalid grid, constraint or
        tolerance, no forward evidence, and any engine refusal on a point or
        interval-endpoint claim: a joint-law requirement the claim cannot meet
        (independent marginals with a nonlinear utility, a quantile or a probability),
        or a mean-only claim asked for a quantile or a probability.
        """
        region = None
        if interval_region is not None:
            region = (
                _claim(interval_region.lower),
                _claim(interval_region.upper),
                bool(interval_region.endpoints_bound_functional),
            )
        members = None
        if identified_set is not None:
            members = (
                [m._tuple() for m in identified_set.members],
                bool(identified_set.exhaustive),
            )
        native, refusal = _NativeInverseQueryArtifact.build(
            json.dumps(self.contract._wire()),
            json.dumps(self._wire()),
            None if point is None else _claim(point),
            region,
            members,
            None if scenarios is None else [s._tuple() for s in scenarios],
        )
        _raise(refusal)
        assert native is not None
        return InverseResult(self, native)


def evaluate(
    query: InverseQuery,
    point: ForwardClaim | None = None,
    *,
    interval_region: IntervalRegion | None = None,
    identified_set: IdentifiedSet | None = None,
    scenarios: Sequence[Scenario] | None = None,
) -> InverseResult:
    """Evaluate ``query`` on forward evidence; see :meth:`InverseQuery.evaluate`."""
    return query.evaluate(
        point, interval_region=interval_region, identified_set=identified_set, scenarios=scenarios
    )


def consume(data: bytes, *, expected_identity: Mapping[str, str] | None = None) -> InverseResult:
    """Consume an exported inverse query; see :meth:`InverseResult.consume`."""
    return InverseResult.consume(data, expected_identity=expected_identity)


# --------------------------------------------------------------------- baseline


@dataclass(frozen=True, slots=True)
class EnumeratedPoint:
    """A forward point of the finite-enumeration baseline: an action, its mean, its support."""

    id: str
    value: float | None
    supported: bool = True


def finite_enumeration_baseline(
    points: Sequence[EnumeratedPoint],
    target: float,
    *,
    direction: Direction = "at_least",
    tolerance: float = 0.0,
) -> dict[str, Feasibility]:
    """The 2.2 finite-enumeration baseline: classify each enumerated action by its forward
    mean against ``target``.

    An action off the evaluated grid is ``unevaluated`` (never interpolated) and an
    unsupported one is ``unsupported``. It uses no distribution engine, so it is the
    independent reference the generalized query agrees with on target-mean cases.
    """
    rows = [{"id": p.id, "value": p.value, "supported": p.supported} for p in points]
    text, refusal = _baseline(
        json.dumps(rows), _finite("target", target), _direction(direction), float(tolerance)
    )
    _raise(refusal)
    assert text is not None
    return {action: status for action, status in json.loads(text)}


__all__ = [
    "FEASIBILITY_STATUSES",
    "ActionReport",
    "Constraint",
    "ConstraintValue",
    "EnumeratedPoint",
    "IdentifiedSet",
    "IntervalRegion",
    "InverseQuery",
    "InverseQueryRefusal",
    "InverseResult",
    "MeanClaim",
    "MemberStatus",
    "PosteriorFeasibility",
    "Scenario",
    "SourceReceipt",
    "consume",
    "evaluate",
    "finite_enumeration_baseline",
    "probability_threshold",
    "target_mean",
    "target_quantile",
]
