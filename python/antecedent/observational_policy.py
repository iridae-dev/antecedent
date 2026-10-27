"""Observational policy value under a declared treatment support.

The value of a treatment policy is the mean outcome if every unit were given the action the
policy picks from its covariates. From observational data it is estimated by a cross-fitted
doubly robust score: a multinomial-logit propensity and one outcome regression per action are
fitted on the other folds, so the estimate is consistent when either side is correct. The
treatment support (the actions the policy may take, each of which must have positive
propensity), the objective and the evaluation have no default and are refused when absent::

    from antecedent import observational_policy

    # A policy fixed before seeing these rows, scored on every row.
    value = observational_policy.policy_value(
        outcome, treatment,
        support=[0.0, 1.0], objective="maximize",
        evaluation="prespecified", policy_actions=treat_if_age_over_60,
        policy_origin="prespecified",
        adjust={"age": age},
    )

    # A policy learned on part of the data and scored on independent held-out rows.
    held_out = observational_policy.policy_value(
        outcome, treatment, support=[0.0, 1.0], objective="maximize",
        evaluation="held_out", evaluation_fraction=0.4, adjust={"age": age},
    )

    # A policy learned on the same data: selection-aware cross-fitted learning and scoring.
    honest = observational_policy.policy_value(
        outcome, treatment, support=[0.0, 1.0], objective="maximize",
        evaluation="cross_fitted_learning", adjust={"age": age},
    )

A policy declared ``policy_origin="learned_on_evaluation_rows"`` is refused
(``policy_learned_on_evaluation_rows``): its value on the rows it was fitted to is estimated
too high. ``held_out`` learns the greedy policy (the action with the best predicted outcome)
on a seeded learning part and reports that one policy's value on the held-out rows;
``cross_fitted_learning`` is the value of the learning procedure fed ``(K - 1) / K`` of the
data, not of one final policy.

What the answer is **not**:

* No graph is checked: the caller owns that ``adjust`` is a valid adjustment set.
* The standard error is an influence-function standard error. Its calibration is
  **unmeasured** (``calibration == "unmeasured"``): no interval is produced and no coverage
  claim is made.
* An action outside the declared support, or one whose propensity falls below the floor,
  refuses instead of being extrapolated.

Everything numerical, and every refusal, is computed in Rust.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ._native import evaluate_observational_policy as _evaluate
from .errors import CausalTypeError, CausalValueError
from .execution import Execution, resolve_execution

__all__ = ["ActionValue", "ObservationalPolicyValue", "ReferenceComparison", "policy_value"]

EvaluationName = Literal["prespecified", "held_out", "cross_fitted_learning"]
ObjectiveName = Literal["maximize", "minimize"]
PolicyOriginName = Literal["prespecified", "learned_on_evaluation_rows"]


@dataclass(frozen=True, slots=True)
class ActionValue:
    """The value of one action given to everyone, on the scored rows."""

    level: float
    value: float
    se: float
    #: Fraction of the scored rows to which the policy gives this action.
    policy_share: float
    #: Smallest out-of-sample propensity of the action over the scored rows.
    propensity_min: float
    #: Complete-case rows observed at the action.
    rows: int


@dataclass(frozen=True, slots=True)
class ReferenceComparison:
    """The policy against a single declared action given to everyone."""

    level: float
    incremental_value: float
    incremental_se: float


@dataclass(frozen=True, slots=True)
class ObservationalPolicyValue:
    """The value of a policy with the evidence needed to read it."""

    evaluation: str
    estimand: str
    assumptions: tuple[str, ...]
    objective: str
    support: tuple[float, ...]
    #: ``"estimated"`` (multinomial logit) or ``"known"`` (declared assignment probabilities).
    propensity_source: str
    value: float
    #: Influence-function standard error (calibration unmeasured).
    se: float
    action_values: tuple[ActionValue, ...]
    reference: ReferenceComparison | None
    evaluation_rows: int
    learning_rows: int
    rows_dropped_missing: int
    #: Original row index, policy action and score of each scored row.
    row_index: tuple[int, ...]
    policy_actions: tuple[float, ...]
    scores: tuple[float, ...]
    calibration: str

    def to_dict(self) -> dict[str, Any]:
        """A plain, JSON-serializable view."""
        return {
            "evaluation": self.evaluation,
            "estimand": self.estimand,
            "assumptions": list(self.assumptions),
            "objective": self.objective,
            "support": list(self.support),
            "propensity_source": self.propensity_source,
            "value": self.value,
            "se": self.se,
            "action_values": [
                {
                    "level": a.level,
                    "value": a.value,
                    "se": a.se,
                    "policy_share": a.policy_share,
                    "propensity_min": a.propensity_min,
                    "rows": a.rows,
                }
                for a in self.action_values
            ],
            "reference": None
            if self.reference is None
            else {
                "level": self.reference.level,
                "incremental_value": self.reference.incremental_value,
                "incremental_se": self.reference.incremental_se,
            },
            "evaluation_rows": self.evaluation_rows,
            "learning_rows": self.learning_rows,
            "rows_dropped_missing": self.rows_dropped_missing,
            "row_index": list(self.row_index),
            "policy_actions": list(self.policy_actions),
            "scores": list(self.scores),
            "calibration": self.calibration,
        }


def _column(values: object, what: str) -> list[float]:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise CausalTypeError(f"{what} must be a one-dimensional numeric array") from error
    if array.ndim != 1:
        raise CausalValueError(f"{what} must be one-dimensional")
    return [float(v) for v in array]


def policy_value(
    outcome: Sequence[float],
    treatment: Sequence[float],
    *,
    support: Sequence[float] | None = None,
    objective: ObjectiveName | None = None,
    evaluation: EvaluationName | None = None,
    policy_actions: Sequence[float] | None = None,
    policy_origin: PolicyOriginName | None = None,
    evaluation_fraction: float | None = None,
    reference: float | None = None,
    known_propensity: Sequence[float] | Sequence[Sequence[float]] | None = None,
    adjust: Mapping[str, Sequence[float]] | None = None,
    folds: int | None = None,
    min_propensity: float | None = None,
    execution: Execution | None = None,
) -> ObservationalPolicyValue:
    """Estimate the value of a treatment policy from observational data.

    ``treatment`` holds the observed action of each row; ``support`` declares the actions the
    policy may take, and every observed treatment must be one of them. ``objective`` says
    whether a larger or smaller outcome is better (it directs learned policies).
    ``evaluation`` chooses how the policy is obtained and scored (see the module docstring);
    ``"prespecified"`` needs ``policy_actions`` (one action per row) and ``policy_origin``,
    ``"held_out"`` needs ``evaluation_fraction`` in ``(0, 1)``. ``reference`` names an action
    to compare the policy against. ``adjust`` maps column names to the covariates the caller
    declares sufficient for adjustment. ``known_propensity`` declares the assignment
    probabilities of a randomized design and replaces the fitted propensity model: one
    probability per action in ``support`` order (e.g. ``[0.5, 0.5]``), or one column per action
    with a value per row. ``folds`` is the cross-fit fold count (default 5),
    ``min_propensity`` the positivity floor of every declared action and ``execution.seed``
    seeds the folds or the held-out split. A ``NaN`` is a missing cell: its row is dropped from the
    complete cases and counted in ``rows_dropped_missing``.

    Raises:
        CausalValueError: A malformed argument.
        CausalUnsupportedError: A refusal with a registered ``reason_code`` such as
            ``required_option_missing``, ``policy_learned_on_evaluation_rows``,
            ``positivity_violated`` or ``treatment_level_undeclared``.
    """
    execution = resolve_execution(execution)
    declaration: dict[str, Any] = {"seed": execution.seed}
    if support is not None:
        declaration["support"] = _column(support, "support")
    if objective is not None:
        declaration["objective"] = str(objective)
    if evaluation is not None:
        declaration["evaluation"] = str(evaluation)
    if policy_actions is not None:
        declaration["policy_actions"] = _column(policy_actions, "policy_actions")
    if policy_origin is not None:
        declaration["policy_origin"] = str(policy_origin)
    if evaluation_fraction is not None:
        declaration["evaluation_fraction"] = float(evaluation_fraction)
    if reference is not None:
        declaration["reference"] = float(reference)
    if known_propensity is not None:
        known = np.asarray(known_propensity, dtype=np.float64)
        if known.ndim == 1:
            declaration["known_propensity"] = [float(v) for v in known]
        elif known.ndim == 2:
            declaration["known_propensity_columns"] = [[float(v) for v in col] for col in known]
        else:
            raise CausalValueError("known_propensity must be one-dimensional or two-dimensional")
    if folds is not None:
        declaration["folds"] = int(folds)
    if min_propensity is not None:
        declaration["min_propensity"] = float(min_propensity)
    adjustment = dict(adjust or {})
    report = json.loads(
        execution._call(
            _evaluate,
            _column(outcome, "outcome"),
            _column(treatment, "treatment"),
            list(adjustment),
            [_column(values, f"adjust[{name!r}]") for name, values in adjustment.items()],
            json.dumps(declaration),
        )
    )
    ref = report["reference"]
    return ObservationalPolicyValue(
        evaluation=report["evaluation"],
        estimand=report["estimand"],
        assumptions=tuple(report["assumptions"]),
        objective=report["objective"],
        support=tuple(report["support"]),
        propensity_source=report["propensity_source"],
        value=report["value"],
        se=report["se"],
        action_values=tuple(ActionValue(**a) for a in report["action_values"]),
        reference=None if ref is None else ReferenceComparison(**ref),
        evaluation_rows=report["evaluation_rows"],
        learning_rows=report["learning_rows"],
        rows_dropped_missing=report["rows_dropped_missing"],
        row_index=tuple(report["row_index"]),
        policy_actions=tuple(report["policy_actions"]),
        scores=tuple(report["scores"]),
        calibration=report["calibration"],
    )
