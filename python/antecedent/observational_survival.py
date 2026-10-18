"""Observational survival effects at a declared horizon.

For an observed (not randomized) binary treatment, estimate the survival probability
difference ``S_1(t*) - S_0(t*)`` and the restricted mean survival time difference through
``t*``. Three declarations have no default and are refused when absent::

    from antecedent import observational_survival

    effect = observational_survival.survival_effect(
        duration, event, treatment,
        active=1.0, control=0.0,
        horizon=2.0,                          # t*
        censoring="at_random_given_covariates",
        adjust={"age": age, "stage": stage},  # the declared sufficient adjustment set
    )
    effect.survival_difference.estimate, effect.survival_difference.se
    effect.rmst_difference.estimate, effect.rmst_difference.se
    effect.assumptions  # what the numbers rest on

``censoring`` is ``"independent_within_arm"`` (censoring does not depend on the event time or
on the covariates within an arm) or ``"at_random_given_covariates"`` (coarsening at random:
given the arm and the declared covariates the censoring hazard does not depend on the event
time). Informative censoring is not identified and is not offered.

The estimator is cross-fitted and doubly robust: a propensity model, a Cox event model and a
Cox censoring model per arm are fitted on the other folds, and the estimate is the mean of
the efficient-influence-function scores, which is consistent when either the outcome model
or the propensity and censoring models are correct.

What the answer is **not**:

* No graph is checked: the caller owns that ``adjust`` is a valid adjustment set.
* The standard errors are influence-function standard errors. Their calibration is
  **unmeasured** (``calibration == "unmeasured"``): no interval is produced and no coverage
  claim is made.
* A horizon beyond an arm's follow-up, a propensity below the floor, or a probability of
  remaining uncensored below the floor refuses ``positivity_violated`` instead of weighting
  by an unbounded inverse probability.

Everything numerical, and every refusal, is computed in Rust; this module builds the
declaration and presents the result.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ._native import evaluate_observational_survival as _evaluate
from .errors import CausalTypeError, CausalValueError

__all__ = [
    "ArmSurvival",
    "ObservationalSurvivalEffect",
    "SurvivalContrast",
    "survival_effect",
]

CensoringName = Literal["independent_within_arm", "at_random_given_covariates"]


@dataclass(frozen=True, slots=True)
class ArmSurvival:
    """One arm's standardized survival summaries at the horizon."""

    level: float
    survival: float
    survival_se: float
    rmst: float
    rmst_se: float
    rows: int
    events: int
    #: Smallest out-of-fold propensity of the arm.
    propensity_min: float
    #: Smallest out-of-fold probability of remaining uncensored through the horizon.
    censoring_survival_min: float


@dataclass(frozen=True, slots=True)
class SurvivalContrast:
    """An active-minus-control contrast and its influence-function standard error."""

    estimate: float
    se: float


@dataclass(frozen=True, slots=True)
class ObservationalSurvivalEffect:
    """Survival effects of an observational treatment at a declared horizon."""

    horizon: float
    censoring: str
    estimands: tuple[str, ...]
    assumptions: tuple[str, ...]
    active: ArmSurvival
    control: ArmSurvival
    survival_difference: SurvivalContrast
    rmst_difference: SurvivalContrast
    n_obs: int
    rows_dropped_missing: int
    #: Original row index of each used row, with the per-row scores.
    row_index: tuple[int, ...]
    survival_difference_scores: tuple[float, ...]
    rmst_difference_scores: tuple[float, ...]
    calibration: str

    def to_dict(self) -> dict[str, Any]:
        """A plain, JSON-serializable view."""

        def arm(a: ArmSurvival) -> dict[str, Any]:
            return {
                "level": a.level,
                "survival": a.survival,
                "survival_se": a.survival_se,
                "rmst": a.rmst,
                "rmst_se": a.rmst_se,
                "rows": a.rows,
                "events": a.events,
                "propensity_min": a.propensity_min,
                "censoring_survival_min": a.censoring_survival_min,
            }

        return {
            "horizon": self.horizon,
            "censoring": self.censoring,
            "estimands": list(self.estimands),
            "assumptions": list(self.assumptions),
            "active": arm(self.active),
            "control": arm(self.control),
            "survival_difference": {
                "estimate": self.survival_difference.estimate,
                "se": self.survival_difference.se,
            },
            "rmst_difference": {
                "estimate": self.rmst_difference.estimate,
                "se": self.rmst_difference.se,
            },
            "n_obs": self.n_obs,
            "rows_dropped_missing": self.rows_dropped_missing,
            "row_index": list(self.row_index),
            "survival_difference_scores": list(self.survival_difference_scores),
            "rmst_difference_scores": list(self.rmst_difference_scores),
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


def _arm(report: Mapping[str, Any]) -> ArmSurvival:
    return ArmSurvival(
        level=report["level"],
        survival=report["survival"],
        survival_se=report["survival_se"],
        rmst=report["rmst"],
        rmst_se=report["rmst_se"],
        rows=report["rows"],
        events=report["events"],
        propensity_min=report["propensity_min"],
        censoring_survival_min=report["censoring_survival_min"],
    )


def survival_effect(
    duration: Sequence[float],
    event: Sequence[float],
    treatment: Sequence[float],
    *,
    active: float,
    control: float,
    horizon: float | None = None,
    censoring: CensoringName | None = None,
    adjust: Mapping[str, Sequence[float]] | None = None,
    folds: int | None = None,
    min_propensity: float | None = None,
    min_censoring_survival: float | None = None,
    seed: int = 0,
) -> ObservationalSurvivalEffect:
    """Estimate survival probability and restricted mean survival time differences.

    ``duration`` is the follow-up time of each row, ``event`` is 1 where the event was
    observed and 0 where the row was censored, and ``treatment`` holds the observed level.
    ``active`` and ``control`` are the declared levels of the contrast; a row at any other
    level refuses ``treatment_level_undeclared``. ``horizon`` and ``censoring`` must be
    declared (see the module docstring). ``adjust`` maps column names to the covariates the
    caller declares sufficient for adjustment. ``folds`` is the cross-fit fold count
    (default 5), ``min_propensity`` and ``min_censoring_survival`` the positivity floors and
    ``seed`` seeds the fold plan. A ``NaN`` is a missing cell: its row is dropped from the
    complete cases and counted in ``rows_dropped_missing``.

    Raises:
        CausalValueError: A malformed argument.
        CausalUnsupportedError: A refusal with a registered ``reason_code`` such as
            ``required_option_missing`` (no horizon or censoring assumption),
            ``positivity_violated`` or ``treatment_level_undeclared``.
    """
    declaration: dict[str, Any] = {
        "active": float(active),
        "control": float(control),
        "seed": int(seed),
    }
    if horizon is not None:
        declaration["horizon"] = float(horizon)
    if censoring is not None:
        declaration["censoring"] = str(censoring)
    if folds is not None:
        declaration["folds"] = int(folds)
    if min_propensity is not None:
        declaration["min_propensity"] = float(min_propensity)
    if min_censoring_survival is not None:
        declaration["min_censoring_survival"] = float(min_censoring_survival)
    adjustment = dict(adjust or {})
    report = json.loads(
        _evaluate(
            _column(duration, "duration"),
            _column(event, "event"),
            _column(treatment, "treatment"),
            list(adjustment),
            [_column(values, f"adjust[{name!r}]") for name, values in adjustment.items()],
            json.dumps(declaration),
        )
    )
    return ObservationalSurvivalEffect(
        horizon=report["horizon"],
        censoring=report["censoring"],
        estimands=tuple(report["estimands"]),
        assumptions=tuple(report["assumptions"]),
        active=_arm(report["active"]),
        control=_arm(report["control"]),
        survival_difference=SurvivalContrast(**report["survival_difference"]),
        rmst_difference=SurvivalContrast(**report["rmst_difference"]),
        n_obs=report["n_obs"],
        rows_dropped_missing=report["rows_dropped_missing"],
        row_index=tuple(report["row_index"]),
        survival_difference_scores=tuple(report["survival_difference_scores"]),
        rmst_difference_scores=tuple(report["rmst_difference_scores"]),
        calibration=report["calibration"],
    )
