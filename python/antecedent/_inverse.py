"""Finite-action inverse-outcome query (target mean) over a licensed forward response.

The forward question ``a -> E[Y | do(a)]`` is answered by an existing response
route (``analyze`` / ``prepare`` with a :class:`~antecedent.query.ResponseCurve`).
This module inverts it over an enumerated, caller-supplied action grid: each
:class:`Action` carries a declared cost and named constraints, and the query
asks whether ``E[Y^do(a)] >= threshold`` (or ``<=``). Every action comes back
``feasible``, ``infeasible``, ``unsupported`` or ``unevaluated`` with its
assumptions, cost, support status and numerical tolerance.

What a result is:

* Classification is point-based. When the forward route published a pointwise or
  simultaneous interval, an action is additionally flagged ``robustly_feasible``
  when that interval, used exactly as published, lies on the feasible side of the
  threshold. The flag adds no coverage claim of its own.
* "Feasible", "infeasible", "necessary" and "sufficient" are relative to the
  enumerated action set and the forward response's declared assumptions. An
  action that was not enumerated, or that is off the evaluated grid, is
  unresolved, never excluded or interpolated.
* A mean threshold is not a chance constraint ``P(Y^do(a) >= y) >= q``, and a
  target quantile needs the interventional distribution (2.3B); an observational
  ``P(X | Y)`` is not an action. All three refuse with ``cell_not_licensed``.

Missing-evidence study options (the study planner's output) and sensitivity
tipping points are attached as separate views. Neither is a confidence interval,
and a hypothetical study is never observed evidence; neither enters the
classification or the report identity.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from . import _native
from .errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from .query import ResponseCurve
from .results.response import CausalResponseView

_DIRECTIONS = ("at_least", "at_most")

_STUDY_NOTE = (
    "Hypothetical catalog delta from the study planner: studies that would repair a failed "
    "identification if they delivered exactly the declared regimes. Not observed evidence, not "
    "an estimate of this outcome, and not part of the classification; its relevance to this "
    "query is the caller's declaration."
)
_SENSITIVITY_NOTE = (
    "An assumption range with tipping points from a separate sensitivity analysis. It is not a "
    "confidence interval, adds no coverage to the classification, and speaks about its own "
    "estimand, not about this action grid unless the caller built it so."
)


@dataclass(frozen=True, slots=True)
class Action:
    """One enumerated intervention.

    ``point`` is the dose (a float) or, on a temporal dose-by-horizon surface,
    the ``(dose, horizon)`` coordinates, exactly as evaluated by the forward
    response. ``cost`` is in the caller's units. ``constraints`` maps a named
    constraint to whether the action satisfies it; the caller owns their meaning.
    """

    label: str
    point: float | Sequence[float]
    cost: float = 0.0
    constraints: Mapping[str, bool] = field(default_factory=dict)

    def coordinates(self) -> tuple[float, ...]:
        if isinstance(self.point, (bool, str, bytes)):
            raise CausalTypeError("action point must be a number or a sequence of numbers")
        if isinstance(self.point, (int, float)):
            return (float(self.point),)
        if any(isinstance(value, bool) for value in self.point):
            raise CausalTypeError("action coordinates must be numbers, not booleans")
        return tuple(float(value) for value in self.point)


def _action_wire(action: Action) -> tuple[str, list[float], float, list[tuple[str, bool]]]:
    constraints = []
    for name, satisfied in action.constraints.items():
        if not isinstance(name, str) or not isinstance(satisfied, bool):
            raise CausalTypeError("action constraints need string names and boolean values")
        constraints.append((name, satisfied))
    return (action.label, list(action.coordinates()), float(action.cost), constraints)


@dataclass(frozen=True, slots=True)
class TargetMean:
    """Goal ``E[Y^do(a)] >= threshold`` (``at_least``) or ``<= threshold`` (``at_most``)."""

    threshold: float
    direction: Literal["at_least", "at_most"] = "at_least"


@dataclass(frozen=True, slots=True)
class ChanceConstraint:
    """``P(Y^do(a) >= threshold) >= probability``: always refused (``cell_not_licensed``)."""

    threshold: float
    probability: float


@dataclass(frozen=True, slots=True)
class TargetQuantile:
    """A target quantile of ``Y^do(a)``: always refused (``cell_not_licensed``, 2.3B)."""

    level: float
    threshold: float


@dataclass(frozen=True, slots=True)
class ObservationalScenarios:
    """Observational scenarios compatible with an outcome: always refused."""


InverseQuery = TargetMean | ChanceConstraint | TargetQuantile | ObservationalScenarios


@dataclass(frozen=True, slots=True)
class ActionResult:
    """One action's classification.

    ``status`` is ``feasible``, ``infeasible``, ``unsupported`` or ``unevaluated``
    and ``reason`` says why (``meets_target``, ``misses_target``,
    ``constraint_violated``, ``over_budget``, ``outside_empirical_support``,
    ``missing_evidence``, ``not_on_evaluated_grid``, ``non_finite_estimate``).
    ``margin`` is the signed distance to the goal side of the threshold (positive
    meets it); ``interval`` is the forward route's interval at this point as
    published; ``robustly_feasible`` is ``None`` without an interval or for an
    action that is not point-feasible.
    """

    label: str
    point: tuple[float, ...]
    cost: float
    status: str
    reason: str
    violated_constraints: tuple[str, ...]
    estimate: float | None
    margin: float | None
    within_tolerance: bool
    support_status: str | None
    interval: tuple[float | None, float | None] | None
    robustly_feasible: bool | None


def _action_result(row: Mapping[str, Any]) -> ActionResult:
    return ActionResult(
        label=row["label"],
        point=tuple(row["point"]),
        cost=row["cost"],
        status=row["status"],
        reason=row["reason"],
        violated_constraints=tuple(row["violated_constraints"]),
        estimate=row["estimate"],
        margin=row["margin"],
        within_tolerance=row["within_tolerance"],
        support_status=row["support_status"],
        interval=None if row["interval"] is None else tuple(row["interval"]),
        robustly_feasible=row["robustly_feasible"],
    )


@dataclass(frozen=True, slots=True)
class MissingEvidenceView:
    """The study planner's proposals, projected verbatim as a separate view."""

    kind: str
    evidence_status: str
    plan_outcome: str
    proposals: tuple[Mapping[str, Any], ...]
    minimal: bool | None
    note: str


@dataclass(frozen=True, slots=True)
class SensitivityView:
    """A sensitivity analysis's assumption range and tipping points, as a separate view."""

    kind: str
    is_confidence_interval: bool
    assumption_range: Mapping[str, float]
    fields: Mapping[str, Any]
    note: str


@dataclass(frozen=True, slots=True)
class InverseOutcomeReport:
    """The classified enumeration.

    ``enumerated`` is ``reachable`` (some action is feasible),
    ``unreachable_within_set`` (every enumerated action is infeasible) or
    ``undetermined`` (none feasible, some unsupported or unevaluated). It never
    speaks about actions that were not enumerated. ``inference_claim`` is
    ``point_only``. ``identity`` is an order-invariant digest of every input;
    :meth:`verify` recomputes and compares the report from its stored inputs;
    this local check does not authenticate those inputs or the forward model.
    """

    query: TargetMean
    tolerance: float
    budget: float | None
    outcome: str
    population: str
    forward_claim_id: str | None
    forward_program_id: str | None
    forward_data_snapshot_id: str | None
    horizons: tuple[int, ...] | None
    assumptions: tuple[str, ...]
    support_basis: str
    interval: Mapping[str, Any] | None
    actions: tuple[ActionResult, ...]
    feasible: tuple[str, ...]
    infeasible: tuple[str, ...]
    unsupported: tuple[str, ...]
    unevaluated: tuple[str, ...]
    cheapest_feasible: tuple[str, ...]
    robustly_feasible: tuple[str, ...]
    enumerated: str
    inference_claim: str
    scope_note: str
    identity: str
    missing_evidence: MissingEvidenceView | None = None
    sensitivity: SensitivityView | None = None
    _call: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)
    _payload: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)
    _declared_horizons: tuple[int, ...] | None = field(default=None, repr=False, compare=False)

    def action(self, label: str) -> ActionResult:
        """The result for one enumerated action."""
        for result in self.actions:
            if result.label == label:
                return result
        raise CausalValueError(f"no enumerated action is labelled {label!r}")

    def verify(self) -> bool:
        """Recompute the report from its stored inputs; ``True`` when identical.

        The report is a pure function of the supplied forward response, action
        grid, target, budget and tolerance. Recompute and compare its identity,
        public classification fields and stored native payload. This does not
        authenticate edited inputs or certify the forward model.
        """
        recomputed = _native.classify_inverse_outcome_stage(**self._call)
        if recomputed != self._payload:
            return False
        expected_actions = tuple(_action_result(row) for row in recomputed["actions"])
        expected_horizons = (
            (int(self._call["actions"][0][1][1]),)
            if self._declared_horizons is not None and self._call["actions"]
            else None
        )
        return bool(
            self.query
            == TargetMean(recomputed["query"]["threshold"], recomputed["query"]["direction"])
            and self.tolerance == recomputed["tolerance"]
            and self.budget == recomputed["budget"]
            and self.outcome == recomputed["outcome"]
            and self.population == recomputed["population"]
            and self.forward_claim_id == recomputed["forward_claim_id"]
            and self.forward_program_id == recomputed["forward_program_id"]
            and self.forward_data_snapshot_id == recomputed["forward_data_snapshot_id"]
            and self.horizons == expected_horizons
            and self.assumptions == tuple(recomputed["assumptions"])
            and self.support_basis == recomputed["support_basis"]
            and self.interval == recomputed["interval"]
            and self.actions == expected_actions
            and self.feasible == tuple(recomputed["feasible"])
            and self.infeasible == tuple(recomputed["infeasible"])
            and self.unsupported == tuple(recomputed["unsupported"])
            and self.unevaluated == tuple(recomputed["unevaluated"])
            and self.cheapest_feasible == tuple(recomputed["cheapest_feasible"])
            and self.robustly_feasible == tuple(recomputed["robustly_feasible"])
            and self.enumerated == recomputed["enumerated"]
            and self.inference_claim == recomputed["inference_claim"]
            and self.scope_note == recomputed["scope_note"]
            and self.identity == recomputed["identity"]
        )


def _finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise CausalValueError(f"{name} must be a finite number")
    return float(value)


def _forward(response: CausalResponseView) -> tuple[Any, ...]:
    """The forward response exactly as the licensed route published it."""
    if not isinstance(response, CausalResponseView):
        raise CausalTypeError(
            "inverse_outcome requires the CausalResponseView of a forward response route "
            "(ResponseCurve); run analyze/prepare first",
            reason_code="invalid_argument",
        )
    if response.evidence_status == "allowed_unlicensed":
        raise CausalUnsupportedError(
            "inverse.forward_not_licensed: the forward response is allowed for compatibility "
            "but does not carry a licensed causal claim",
            reason_code="cell_not_licensed",
        )
    view = response.response
    points = [] if view is None else [[float(c) for c in point] for point in view.points]
    outcomes = [] if view is None else list(view.outcomes)
    mean = [] if view is None else [float(row[0]) for row in view.values]
    estimand = response.estimand
    # A scalar-outcome mean curve (or temporal dose-by-horizon surface) only: a
    # derivative, contrast or other functional has no target-mean reading.
    mean_response = isinstance(estimand, ResponseCurve) and len(outcomes) == 1
    point_identified = (
        bool(response.identification)
        and response.envelope is None
        and response.uncertainty.kind != "identified_set"
    )
    cells = response.support.point_status
    if cells is not None and len(cells) == len(points):
        support, basis = [str(label) for label in cells], "per_point"
    else:
        support, basis = [str(response.support.status)] * len(points), "surface_worst_case"
    interval = None
    uncertainty = response.uncertainty
    simultaneous = response.simultaneous_band
    if (
        simultaneous is not None
        and uncertainty.interpretation is not None
        and len(simultaneous) == len(points)
    ):
        interval = (
            "simultaneous",
            uncertainty.interpretation,
            float(simultaneous.level),
            [float(row[0]) for row in simultaneous.lower],
            [float(row[0]) for row in simultaneous.upper],
        )
    if (
        interval is None
        and uncertainty.kind in ("pointwise", "simultaneous")
        and uncertainty.interpretation is not None
        and uncertainty.level is not None
        and uncertainty.lower is not None
        and uncertainty.upper is not None
        and len(uncertainty.lower) == len(points)
    ):
        interval = (
            uncertainty.kind,
            uncertainty.interpretation,
            float(uncertainty.level),
            [float(row[0]) for row in uncertainty.lower],
            [float(row[0]) for row in uncertainty.upper],
        )
    target = getattr(estimand, "target_population", None)
    population = (
        "source population (no target population declared)" if target is None else repr(target)
    )
    return (
        outcomes[0] if len(outcomes) == 1 else "",
        population,
        (response.claim_id, response.program_id, response.data_snapshot_id),
        mean_response,
        point_identified,
        points,
        mean,
        support,
        basis,
        interval,
        [str(assumption) for assumption in response.assumptions],
    )


def _plan_dict(source: Any) -> Mapping[str, Any]:
    plan = source.plan() if callable(getattr(source, "plan", None)) else source
    if not isinstance(plan, Mapping) or not isinstance(plan.get("proposals"), Sequence):
        raise CausalValueError(
            "study_options must be a study plan (the stage returned by plan_studies, or its "
            "plan() dictionary)"
        )
    return plan


def _missing_evidence(source: Any) -> MissingEvidenceView:
    plan = _plan_dict(source)
    keys = ("rank", "candidates", "cost_units", "sample_budget", "deliver", "repairs")
    proposals = tuple(
        {key: proposal[key] for key in keys if key in proposal} for proposal in plan["proposals"]
    )
    return MissingEvidenceView(
        kind="study_options",
        evidence_status="hypothetical_catalog_delta",
        plan_outcome=str(plan.get("outcome", "")),
        proposals=proposals,
        minimal=plan.get("minimal"),
        note=_STUDY_NOTE,
    )


def _sensitivity(source: Mapping[str, Any]) -> SensitivityView:
    if not isinstance(source, Mapping) or not isinstance(source.get("assumption_range"), Mapping):
        raise CausalValueError(
            "sensitivity must be a sensitivity result carrying an assumption_range "
            "(minimum and maximum)"
        )
    bounds = source["assumption_range"]
    low, high = (
        _finite("assumption_range minimum", bounds["minimum"]),
        _finite("assumption_range maximum", bounds["maximum"]),
    )
    keep = (
        "estimand",
        "baseline",
        "decision_threshold",
        "tipping_fraction",
        "axis_tipping",
        "frontier",
        "interval_interpretation",
        "interpretation",
    )
    return SensitivityView(
        kind="assumption_range",
        is_confidence_interval=False,
        assumption_range={"minimum": low, "maximum": high},
        fields={key: source[key] for key in keep if key in source},
        note=_SENSITIVITY_NOTE,
    )


def _query_arguments(query: InverseQuery) -> dict[str, Any]:
    if isinstance(query, TargetMean):
        if query.direction not in _DIRECTIONS:
            raise CausalValueError(f"direction must be one of {_DIRECTIONS}")
        return {
            "query_kind": "target_mean",
            "threshold": _finite("threshold", query.threshold),
            "direction": query.direction,
        }
    if isinstance(query, ChanceConstraint):
        return {
            "query_kind": "probability_target",
            "threshold": float(query.threshold),
            "direction": "at_least",
            "probability": float(query.probability),
        }
    if isinstance(query, TargetQuantile):
        return {
            "query_kind": "quantile_target",
            "threshold": float(query.threshold),
            "direction": "at_least",
            "level": float(query.level),
        }
    if isinstance(query, ObservationalScenarios):
        return {"query_kind": "observational_scenarios", "threshold": 0.0, "direction": "at_least"}
    raise CausalTypeError(
        "query must be TargetMean, ChanceConstraint, TargetQuantile or ObservationalScenarios"
    )


def inverse_outcome(
    response: CausalResponseView,
    *,
    query: InverseQuery,
    actions: Sequence[Action],
    budget: float | None = None,
    tolerance: float = 1e-9,
    study_options: Any = None,
    sensitivity: Mapping[str, Any] | None = None,
    cancel: Any = None,
) -> InverseOutcomeReport:
    """Classify an enumerated action grid against a licensed forward response.

    ``response`` is the result of an existing response route evaluated on a grid
    that contains every action's point (an action off that grid is
    ``unevaluated``). ``query`` is a :class:`TargetMean`; the probability,
    quantile and observational-scenario queries refuse with ``cell_not_licensed``
    (``inverse.probability_target``, ``inverse.quantile_target``,
    ``inverse.observational_scenarios``). A forward response that is not a point
    identified mean curve refuses with ``route_not_supported``
    (``inverse.forward_not_mean_response`` / ``inverse.forward_not_point_identified``).
    ``budget`` caps an action's cost, ``tolerance`` is the numerical tolerance on
    the margin (a margin within ``-tolerance`` meets the target).

    ``study_options`` (a study plan) and ``sensitivity`` (a sensitivity result)
    are attached as separate views and never change the classification.
    """
    arguments = _query_arguments(query)
    if isinstance(actions, (str, bytes)) or not all(isinstance(a, Action) for a in actions):
        raise CausalTypeError("actions must be a sequence of Action values")
    forward_wire = _forward(response)
    action_rows = [_action_wire(action) for action in actions]
    declared_horizons = getattr(response.estimand, "horizons", None)
    selected_horizon = None
    if declared_horizons is not None:
        allowed_horizons = set(int(h) for h in declared_horizons)
        for _, point, _, _ in action_rows:
            if (
                len(point) != 2
                or not point[1].is_integer()
                or int(point[1]) not in allowed_horizons
            ):
                raise CausalValueError(
                    "a temporal action needs a dose and one horizon declared by the forward query"
                )
        selected_horizon = (int(action_rows[0][1][1]),) if action_rows else None
    call: dict[str, Any] = {
        **arguments,
        "forward": forward_wire,
        "actions": action_rows,
        "budget": None if budget is None else float(budget),
        "tolerance": float(tolerance),
    }
    payload = _native.classify_inverse_outcome_stage(**call, cancel=cancel)
    return InverseOutcomeReport(
        query=TargetMean(payload["query"]["threshold"], payload["query"]["direction"]),
        tolerance=payload["tolerance"],
        budget=payload["budget"],
        outcome=payload["outcome"],
        population=payload["population"],
        forward_claim_id=payload["forward_claim_id"],
        forward_program_id=payload["forward_program_id"],
        forward_data_snapshot_id=payload["forward_data_snapshot_id"],
        horizons=selected_horizon,
        assumptions=tuple(payload["assumptions"]),
        support_basis=payload["support_basis"],
        interval=payload["interval"],
        actions=tuple(_action_result(row) for row in payload["actions"]),
        feasible=tuple(payload["feasible"]),
        infeasible=tuple(payload["infeasible"]),
        unsupported=tuple(payload["unsupported"]),
        unevaluated=tuple(payload["unevaluated"]),
        cheapest_feasible=tuple(payload["cheapest_feasible"]),
        robustly_feasible=tuple(payload["robustly_feasible"]),
        enumerated=payload["enumerated"],
        inference_claim=payload["inference_claim"],
        scope_note=payload["scope_note"],
        identity=payload["identity"],
        missing_evidence=None if study_options is None else _missing_evidence(study_options),
        sensitivity=None if sensitivity is None else _sensitivity(sensitivity),
        _call=call,
        _payload=payload,
        _declared_horizons=None if declared_horizons is None else tuple(declared_horizons),
    )


__all__ = [
    "Action",
    "ActionResult",
    "ChanceConstraint",
    "InverseOutcomeReport",
    "InverseQuery",
    "MissingEvidenceView",
    "ObservationalScenarios",
    "SensitivityView",
    "TargetMean",
    "TargetQuantile",
    "inverse_outcome",
]
