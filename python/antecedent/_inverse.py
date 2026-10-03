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
from .errors import CausalTypeError, CausalValueError
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
        return tuple(float(value) for value in self.point)


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
    :meth:`verify` recomputes the report from the stored inputs.
    """

    query: TargetMean
    tolerance: float
    budget: float | None
    outcome: str
    population: str
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

    def action(self, label: str) -> ActionResult:
        """The result for one enumerated action."""
        for result in self.actions:
            if result.label == label:
                return result
        raise CausalValueError(f"no enumerated action is labelled {label!r}")

    def verify(self) -> bool:
        """Recompute the report from its stored inputs; ``True`` when identical.

        The report is a pure function of the forward response as published, the
        action grid, the target, the budget and the tolerance, so an independent
        recomputation reproduces the identity and every classification.
        """
        recomputed = _native.classify_inverse_outcome_stage(**self._call)
        return bool(recomputed == self._payload)


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
    if (
        uncertainty.kind in ("pointwise", "simultaneous")
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
    call: dict[str, Any] = {
        **arguments,
        "forward": _forward(response),
        "actions": [
            (
                action.label,
                list(action.coordinates()),
                float(action.cost),
                [(str(name), bool(ok)) for name, ok in action.constraints.items()],
            )
            for action in actions
        ],
        "budget": None if budget is None else float(budget),
        "tolerance": float(tolerance),
    }
    payload = _native.classify_inverse_outcome_stage(**call, cancel=cancel)
    horizons = getattr(response.estimand, "horizons", None)
    return InverseOutcomeReport(
        query=TargetMean(payload["query"]["threshold"], payload["query"]["direction"]),
        tolerance=payload["tolerance"],
        budget=payload["budget"],
        outcome=payload["outcome"],
        population=payload["population"],
        horizons=None if horizons is None else tuple(int(h) for h in horizons),
        assumptions=tuple(payload["assumptions"]),
        support_basis=payload["support_basis"],
        interval=payload["interval"],
        actions=tuple(
            ActionResult(
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
            for row in payload["actions"]
        ),
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
