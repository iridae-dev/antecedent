"""Point evaluation of prespecified longitudinal treatment regimes.

This low-level utility evaluates static or history-adaptive binary regimes using
user-supplied sequential treatment and censoring probabilities. The regime
value and sequential g-formula paths are point-only; the binary MSM path
returns pointwise subject-clustered standard errors. No path fits the nuisance
probabilities or adds a support-matrix license.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from math import isfinite
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike

from . import _native

HistoryPolicy: TypeAlias = Callable[[int, tuple[bool, ...], tuple[tuple[float, ...], ...]], bool]
Regime: TypeAlias = Sequence[bool] | HistoryPolicy


@dataclass(frozen=True, slots=True)
class LongitudinalRegimeQuery:
    """Prespecified binary regime over complete subject histories.

    Each outcome row belongs to one subject. Assignment probabilities are
    known from the sequential randomized design; the retained route reports
    a regime value without an interval.
    """

    outcome: str
    treatment_history: Sequence[Sequence[bool]]
    actions: Sequence[bool] | Sequence[Sequence[bool]]
    treatment_probabilities: Sequence[Sequence[float]]
    subject_ids: Sequence[str]
    method: Literal["ipw", "g_formula", "sequential_dr", "marginal_structural_model"] = "ipw"
    period_outcome_predictions: Sequence[Sequence[float]] | None = None
    stabilizing_numerator_probabilities: Sequence[float] | None = None
    q_predictions: Sequence[Sequence[float]] | None = None
    observation_history: Sequence[Sequence[bool]] | None = None
    prediction_fold_ids: Sequence[int] | None = None
    censoring_probabilities: Sequence[Sequence[float]] | None = None
    outcome_observed: Sequence[bool] | None = None
    fold_ids: Sequence[int] | None = None
    excluded_fold_predictions: bool = False
    probabilities_known_by_design: bool = True
    minimum_probability: float = 0.01
    rule_id: str | None = None
    rule_version: str | None = None
    rule_provenance: str | None = None
    kind: Literal["longitudinal_regime"] = field(default="longitudinal_regime", init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, str) or not self.outcome.strip():
            raise ValueError("outcome must name a non-empty column")
        history = np.asarray(self.treatment_history)
        if history.ndim != 2 or not history.size or not np.isin(history, (0, 1, False, True)).all():
            raise ValueError("treatment_history must be a non-empty binary subject-by-period array")
        n, periods = history.shape
        identity = (self.rule_id, self.rule_version, self.rule_provenance)
        if any(value is not None for value in identity):
            if any(not isinstance(value, str) or not value.strip() for value in identity):
                raise ValueError("dynamic rule requires non-empty rule_id, rule_version, and rule_provenance")
            if self.method == "marginal_structural_model":
                raise ValueError("dynamic rule identity is not applicable to marginal_structural_model")
        if self.method not in ("ipw", "g_formula", "sequential_dr", "marginal_structural_model"):
            raise ValueError("unknown longitudinal method")
        numerator = None if self.stabilizing_numerator_probabilities is None else np.asarray(self.stabilizing_numerator_probabilities, dtype=np.float64)
        if self.method == "marginal_structural_model":
            if numerator is None or numerator.shape != (periods,) or not np.isfinite(numerator).all():
                raise ValueError("marginal_structural_model requires one finite stabilizing numerator probability per period")
        elif numerator is not None:
            raise ValueError("stabilizing numerator probabilities require marginal_structural_model")
        predictions = None if self.period_outcome_predictions is None else np.asarray(self.period_outcome_predictions, dtype=np.float64)
        if self.method == "g_formula":
            if predictions is None or predictions.shape != (n, periods) or not np.isfinite(predictions).all():
                raise ValueError("g_formula requires finite subject-by-period outcome predictions")
        elif predictions is not None:
            raise ValueError("ipw does not accept period_outcome_predictions")
        q = None if self.q_predictions is None else np.asarray(self.q_predictions, dtype=np.float64)
        observation = None if self.observation_history is None else np.asarray(self.observation_history)
        if self.method == "sequential_dr":
            if q is None or q.shape != (n, periods) or not np.isfinite(q).all():
                raise ValueError("sequential_dr requires finite subject-by-period Q predictions")
            if observation is None or observation.shape != (n, periods) or not np.isin(observation, (0, 1, False, True)).all():
                raise ValueError("sequential_dr requires binary subject-by-period observation history")
            if self.prediction_fold_ids is None:
                raise ValueError("sequential_dr requires Q prediction fold ownership")
        elif q is not None or observation is not None or self.prediction_fold_ids is not None:
            raise ValueError("Q predictions and observation history require sequential_dr")
        actions = np.asarray(self.actions)
        if actions.shape == (periods,):
            actions = np.broadcast_to(actions, (n, periods))
        if actions.shape != (n, periods) or not np.isin(actions, (0, 1, False, True)).all():
            raise ValueError("actions must be binary, with one period row or one subject-by-period row")
        probabilities = np.asarray(self.treatment_probabilities, dtype=np.float64)
        if probabilities.shape != (n, periods) or not np.isfinite(probabilities).all():
            raise ValueError("treatment_probabilities must match treatment_history")
        censor = np.ones((n, periods)) if self.censoring_probabilities is None else np.asarray(self.censoring_probabilities, dtype=np.float64)
        if censor.shape != (n, periods) or not np.isfinite(censor).all():
            raise ValueError("censoring_probabilities must match treatment_history")
        observed = np.ones(n, dtype=bool) if self.outcome_observed is None else np.asarray(self.outcome_observed)
        if observed.shape != (n,) or not np.isin(observed, (0, 1, False, True)).all():
            raise ValueError("outcome_observed must contain one binary value per subject")
        ids = tuple(self.subject_ids)
        if len(ids) != n or any(not isinstance(value, str) or not value.strip() for value in ids) or len(set(ids)) != n:
            raise ValueError("subject_ids must contain one distinct non-empty ID per outcome row")
        folds = (0,) * n if self.fold_ids is None else tuple(self.fold_ids)
        if len(folds) != n or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > 2**32 - 1 for value in folds):
            raise ValueError("fold_ids must contain one non-negative u32 per subject")
        if self.excluded_fold_predictions and len(set(folds)) < 2:
            raise ValueError("excluded-fold predictions require at least two subject folds")
        if self.method == "sequential_dr":
            if not self.excluded_fold_predictions:
                raise ValueError("sequential_dr requires declared excluded-fold Q predictions")
            if tuple(self.prediction_fold_ids) != folds:
                raise ValueError("Q prediction fold ownership must match subject folds")
            if not np.array_equal(observation[:, -1], observed):
                raise ValueError("terminal observation must match the final observation-history period")
            if np.any((~observation[:, :-1].astype(bool)) & observation[:, 1:].astype(bool)):
                raise ValueError("observation_history must be monotone after dropout")
            if not observed.any():
                raise ValueError("sequential_dr requires at least one observed terminal outcome")
        if not self.probabilities_known_by_design and not self.excluded_fold_predictions:
            raise ValueError("probabilities require known sequential randomization or excluded-fold predictions")
        floor = self.minimum_probability
        if not isfinite(floor) or not 0 < floor <= 0.5 or np.any(probabilities < floor) or np.any(probabilities > 1 - floor) or np.any(censor < floor) or np.any(censor > 1):
            raise ValueError("sequential treatment or censoring positivity fails at the declared floor")
        if numerator is not None and (np.any(numerator < floor) or np.any(numerator > 1 - floor)):
            raise ValueError("stabilizing numerator probabilities violate the declared positivity floor")
        if self.method == "marginal_structural_model" and np.count_nonzero(observed) <= periods + 1:
            raise ValueError("MSM requires more observed subjects than coefficients")
        object.__setattr__(self, "treatment_history", tuple(tuple(map(bool, row)) for row in history))
        object.__setattr__(self, "actions", tuple(tuple(map(bool, row)) for row in actions))
        object.__setattr__(self, "treatment_probabilities", tuple(tuple(map(float, row)) for row in probabilities))
        object.__setattr__(self, "censoring_probabilities", tuple(tuple(map(float, row)) for row in censor))
        object.__setattr__(self, "outcome_observed", tuple(map(bool, observed)))
        object.__setattr__(self, "subject_ids", ids)
        object.__setattr__(self, "fold_ids", folds)
        if predictions is not None:
            object.__setattr__(self, "period_outcome_predictions", tuple(tuple(map(float, row)) for row in predictions))
        if numerator is not None:
            object.__setattr__(self, "stabilizing_numerator_probabilities", tuple(map(float, numerator)))
        if q is not None:
            object.__setattr__(self, "q_predictions", tuple(tuple(map(float, row)) for row in q))
            object.__setattr__(self, "observation_history", tuple(tuple(map(bool, row)) for row in observation))
            object.__setattr__(self, "prediction_fold_ids", tuple(self.prediction_fold_ids))

    @property
    def periods(self) -> int:
        return len(self.treatment_history[0])

    @classmethod
    def from_dynamic_rule(
        cls,
        *,
        outcome: str,
        treatment_history: Sequence[Sequence[bool]],
        predecision_covariates: Sequence[Sequence[Sequence[float]]],
        rule: HistoryPolicy,
        rule_id: str,
        rule_version: str,
        rule_provenance: str,
        treatment_probabilities: Sequence[Sequence[float]],
        subject_ids: Sequence[str],
        method: Literal["ipw", "g_formula", "sequential_dr"] = "ipw",
        period_outcome_predictions: Sequence[Sequence[float]] | None = None,
        q_predictions: Sequence[Sequence[float]] | None = None,
        observation_history: Sequence[Sequence[bool]] | None = None,
        prediction_fold_ids: Sequence[int] | None = None,
        censoring_probabilities: Sequence[Sequence[float]] | None = None,
        outcome_observed: Sequence[bool] | None = None,
        fold_ids: Sequence[int] | None = None,
        excluded_fold_predictions: bool = False,
        probabilities_known_by_design: bool = True,
        minimum_probability: float = 0.01,
    ) -> LongitudinalRegimeQuery:
        """Freeze a binary rule against observed pre-decision histories.

        At period ``t``, ``rule(t, past_actions, covariates_through_t)`` sees
        actions from periods ``<t`` and covariates from periods ``<=t`` only.
        The callable is caller code: artifacts retain its identity and resolved
        actions, but cannot replay or verify its implementation.
        """
        history = np.asarray(treatment_history)
        if history.ndim != 2 or not history.size or not np.isin(history, (0, 1, False, True)).all():
            raise ValueError("treatment_history must be a non-empty binary subject-by-period array")
        n, periods = history.shape
        covariates = np.asarray(predecision_covariates, dtype=np.float64)
        if covariates.ndim != 3 or covariates.shape[:2] != (n, periods) or covariates.shape[2] == 0:
            raise ValueError("predecision_covariates must be finite subject-by-period-by-feature values")
        observed = np.ones((n, periods), dtype=bool) if observation_history is None else np.asarray(observation_history)
        if observed.shape != (n, periods) or not np.isin(observed, (0, 1, False, True)).all():
            raise ValueError("observation_history must be binary subject-by-period values")
        observed = observed.astype(bool)
        if np.any((~observed[:, :-1]) & observed[:, 1:]):
            raise ValueError("observation_history must be monotone after dropout")
        if not np.isfinite(covariates[observed]).all():
            raise ValueError("observed predecision_covariates must be finite")
        if not callable(rule):
            raise ValueError("rule must be callable")
        if any(not isinstance(value, str) or not value.strip() for value in (rule_id, rule_version, rule_provenance)):
            raise ValueError("dynamic rule requires non-empty rule_id, rule_version, and rule_provenance")
        actions: list[list[bool]] = []
        for subject in range(n):
            row: list[bool] = []
            for period in range(periods):
                if not observed[subject, period]:
                    row.append(False)
                    continue
                past = tuple(bool(value) for value in history[subject, :period])
                available = tuple(tuple(float(value) for value in covariates[subject, t]) for t in range(period + 1))
                decision = rule(period, past, available)
                if not isinstance(decision, (bool, np.bool_)):
                    raise ValueError("dynamic rule must return a binary bool at every decision")
                row.append(bool(decision))
            actions.append(row)
        return cls(
            outcome=outcome, treatment_history=treatment_history, actions=actions,
            treatment_probabilities=treatment_probabilities, subject_ids=subject_ids,
            method=method, period_outcome_predictions=period_outcome_predictions,
            q_predictions=q_predictions, observation_history=observation_history,
            prediction_fold_ids=prediction_fold_ids,
            censoring_probabilities=censoring_probabilities, outcome_observed=outcome_observed,
            fold_ids=fold_ids, excluded_fold_predictions=excluded_fold_predictions,
            probabilities_known_by_design=probabilities_known_by_design,
            minimum_probability=minimum_probability, rule_id=rule_id,
            rule_version=rule_version, rule_provenance=rule_provenance,
        )

    @classmethod
    def marginal_structural_model(
        cls,
        *,
        outcome: str,
        treatment_history: Sequence[Sequence[bool]],
        treatment_probabilities: Sequence[Sequence[float]],
        stabilizing_numerator_probabilities: Sequence[float],
        subject_ids: Sequence[str],
        censoring_probabilities: Sequence[Sequence[float]] | None = None,
        outcome_observed: Sequence[bool] | None = None,
        fold_ids: Sequence[int] | None = None,
        excluded_fold_predictions: bool = False,
        probabilities_known_by_design: bool = True,
        minimum_probability: float = 0.01,
    ) -> LongitudinalRegimeQuery:
        """Specify an additive MSM without an irrelevant regime-action argument.

        The model still uses the ordinary subject-history Study query and the
        same validation, artifacts, and pointwise CR1 result contract.
        """
        history = np.asarray(treatment_history)
        if history.ndim != 2 or history.shape[1] == 0:
            raise ValueError("treatment_history must be a non-empty subject-by-period array")
        return cls(
            outcome=outcome, treatment_history=treatment_history,
            actions=[False] * history.shape[1],
            treatment_probabilities=treatment_probabilities,
            subject_ids=subject_ids, method="marginal_structural_model",
            stabilizing_numerator_probabilities=stabilizing_numerator_probabilities,
            censoring_probabilities=censoring_probabilities,
            outcome_observed=outcome_observed, fold_ids=fold_ids,
            excluded_fold_predictions=excluded_fold_predictions,
            probabilities_known_by_design=probabilities_known_by_design,
            minimum_probability=minimum_probability,
        )


@dataclass(frozen=True, slots=True)
class LongitudinalRegimeEstimate:
    """Regime mean and bounded IPW uncertainty, or an additive MSM summary."""
    value: float
    effective_sample_size: float
    matched_observed_fraction: float
    maximum_weight: float
    minimum_action_probability: float
    minimum_censoring_probability: float
    value_standard_error: float | None = None
    value_interval_95: tuple[float, float] | None = None
    interval_reason: str | None = None
    method: str = "ipw"
    uncertainty: str = "point_only_no_interval"
    probability_ownership: str = "known_sequential_randomization"
    support_status: str = "unlicensed_point_utility"
    period_effects: tuple[float, ...] | None = None
    standard_errors: tuple[float, ...] | None = None
    stabilizing_numerator_probabilities: tuple[float, ...] | None = None
    observed_subjects: int | None = None
    rule_id: str | None = None
    rule_version: str | None = None
    rule_provenance: str | None = None


@dataclass(frozen=True, slots=True)
class RegimeValue:
    """Uncertainty-free value summary for a longitudinal regime."""

    value: float
    effective_sample_size: float
    matched_observed_fraction: float
    maximum_weight: float
    uncertainty: str = "not_estimated"
    support_status: str = "caller_supplied_sequential_probabilities"
    assumptions: tuple[str, ...] = (
        "consistency for the specified binary treatment history",
        "sequential exchangeability conditional on the supplied pre-decision histories",
        "sequential positivity for treatment and remaining uncensored",
    )


@dataclass(frozen=True, slots=True)
class GFormulaValue:
    """Point-only plug-in value from supplied sequential conditional means."""

    value: float
    subject_count: int
    minimum_regime_action_probability: float
    minimum_censoring_survival: float
    fold_ownership: tuple[tuple[str, int], ...]
    uncertainty: str = "not_estimated"
    support_status: str = "caller_supplied_conditional_outcome_predictions"
    crossfit_status: str = "not_claimed"
    assumptions: tuple[str, ...] = (
        "consistency for the specified treatment regime",
        "sequential exchangeability conditional on supplied histories",
        "sequential treatment and censoring positivity at the declared floor",
        "conditional reward predictions are valid under the specified regime",
    )


@dataclass(frozen=True, slots=True)
class DoublyRobustRegimeValue:
    """Point-only sequentially augmented regime value from supplied Q scores."""

    value: float
    subject_count: int
    minimum_regime_action_probability: float
    minimum_censoring_probability: float
    fold_ownership: tuple[tuple[str, int], ...]
    uncertainty: str = "not_estimated"
    support_status: str = "caller_supplied_sequential_Q_and_probabilities"
    crossfit_status: str = "fold_ids_aligned_but_crossfit_not_independently_verified"
    assumptions: tuple[str, ...] = (
        "consistency for the specified treatment regime",
        "sequential exchangeability conditional on supplied histories",
        "sequential treatment and censoring positivity at the declared floor",
        "q_prediction is the conditional mean of the next recursive pseudo-outcome under the regime",
        "Q nuisance predictions are cross-fitted by subject fold as declared by the caller",
        "the treatment and censoring probability nuisances satisfy sequential doubly robust conditions",
    )


@dataclass(frozen=True, slots=True)
class MarginalStructuralModelResult:
    """Additive binary-treatment MSM coefficients and pointwise clustered SEs."""

    intercept: float
    period_effects: tuple[float, ...]
    standard_errors: tuple[float, ...]
    stabilizing_numerator_probabilities: tuple[float, ...]
    effective_sample_size: float
    maximum_weight: float
    observed_subjects: int
    fold_ownership: tuple[tuple[str, int], ...]
    uncertainty_kind: str = "subject_clustered_sandwich_standard_error"
    uncertainty_semantics: str = "pointwise CR1 standard errors; no confidence intervals; no simultaneous coverage claim"
    crossfit_status: str = "caller_supplied propensity data dependence is not independently verified"
    support_status: str = "unlicensed_point_utility"
    assumptions: tuple[str, ...] = (
        "consistency for the observed longitudinal treatment histories",
        "sequential exchangeability conditional on the histories used for treatment probabilities",
        "sequential positivity for treatment and remaining uncensored",
        "the supplied treatment and censoring probabilities are correctly specified",
        "the additive marginal structural mean is correctly specified; period effects are additive and have no interactions",
        "no interference between subjects",
    )


def evaluate_regime_value(
    outcomes: ArrayLike,
    treatment_history: ArrayLike,
    regime: Regime,
    treatment_probabilities: ArrayLike,
    *,
    covariate_history: ArrayLike | None = None,
    outcome_observed: ArrayLike | None = None,
    censoring_survival: ArrayLike | None = None,
    minimum_probability: float = 0.01,
) -> RegimeValue:
    """Evaluate a prespecified regime with a sequential inverse-probability score.

    ``treatment_probabilities[i, t]`` is the known/externally estimated
    conditional probability of treatment at time ``t`` given that unit's
    pre-decision history; the score uses this probability for treated rows and
    its complement for untreated rows. ``censoring_survival[i, t]`` is the
    conditional probability of remaining observed through that period. The
    product of those probabilities weights each complete observed trajectory.

    A regime is either a static sequence of binary actions or a callback
    ``regime(t, past_treatments, covariates_through_t)``. The callback is
    evaluated separately for each subject and period. This is a point-only
    Horvitz–Thompson estimator; uncertainty and nuisance estimation are outside
    this API. Sequential exchangeability, consistency, and positivity are
    assumptions supplied by the caller, not verified by the package.
    """
    y = np.asarray(outcomes, dtype=np.float64)
    a = np.asarray(treatment_history)
    p = np.asarray(treatment_probabilities, dtype=np.float64)
    if y.ndim != 1 or y.size == 0 or not np.isfinite(y).all():
        raise ValueError("outcomes must be a non-empty finite one-dimensional array")
    if a.ndim != 2 or a.shape[0] != y.size or a.shape[1] == 0:
        raise ValueError("treatment_history must have shape (subjects, periods)")
    if not np.isin(a, (0, 1, False, True)).all():
        raise ValueError("treatment_history must contain only binary actions")
    a = a.astype(bool, copy=False)
    n, periods = a.shape
    if p.shape != a.shape or not np.isfinite(p).all():
        raise ValueError("treatment_probabilities must be finite and match treatment_history")
    if not isfinite(minimum_probability) or not 0 < minimum_probability <= 0.5:
        raise ValueError("minimum_probability must be finite and in (0, 0.5]")
    if np.any(p < minimum_probability) or np.any(p > 1.0 - minimum_probability):
        raise ValueError("sequential treatment positivity is violated at the declared probability floor")
    if callable(regime):
        x = np.empty((n, periods, 0), dtype=np.float64) if covariate_history is None else np.asarray(covariate_history, dtype=np.float64)
        if x.ndim != 3 or x.shape[:2] != a.shape or not np.isfinite(x).all():
            raise ValueError("dynamic regimes require finite covariate_history shaped (subjects, periods, covariates)")
        actions = np.empty_like(a)
        for i in range(n):
            past: list[bool] = []
            for t in range(periods):
                history = tuple(tuple(float(v) for v in x[i, s]) for s in range(t + 1))
                action = regime(t, tuple(past), history)
                if not isinstance(action, (bool, np.bool_)):
                    raise ValueError("dynamic regime callback must return a bool for every subject-period")
                actions[i, t] = bool(action)
                past.append(bool(a[i, t]))
    else:
        actions = np.asarray(regime)
        if actions.shape != (periods,) or not np.isin(actions, (0, 1, False, True)).all():
            raise ValueError("static regime must contain one binary action per treatment period")
        actions = np.broadcast_to(actions.astype(bool), a.shape).copy()
    observed = np.ones(n, dtype=bool) if outcome_observed is None else np.asarray(outcome_observed)
    if observed.shape != (n,) or not np.isin(observed, (0, 1, False, True)).all():
        raise ValueError("outcome_observed must contain one bool per subject")
    observed = observed.astype(bool, copy=False)
    censor = np.ones_like(p) if censoring_survival is None else np.asarray(censoring_survival, dtype=np.float64)
    if censor.shape != a.shape or not np.isfinite(censor).all():
        raise ValueError("censoring_survival must be finite and match treatment_history")
    if np.any(censor < minimum_probability) or np.any(censor > 1.0):
        raise ValueError("sequential censoring positivity is violated at the declared probability floor")
    value, ess, matched, max_weight = _native.evaluate_longitudinal_regime_value(
        y, a, actions, p, observed, censor
    )
    return RegimeValue(value, ess, matched, max_weight)


def evaluate_sequential_gformula(
    period_outcome_predictions: ArrayLike,
    treatment_history: ArrayLike,
    regime: Regime,
    treatment_probabilities: ArrayLike,
    *,
    subject_ids: Sequence[str],
    fold_ids: Sequence[int],
    covariate_history: ArrayLike | None = None,
    censoring_survival: ArrayLike | None = None,
    minimum_probability: float = 0.01,
) -> GFormulaValue:
    """Average supplied period-reward predictions under a static/dynamic regime.

    ``period_outcome_predictions[i, t]`` must already be the caller's
    conditional mean reward for subject ``i`` at period ``t`` under the
    specified regime and its downstream continuation. The returned value is
    the mean across subjects of the sum over periods. This function does not
    fit or cross-fit conditional models. Subject IDs and fold IDs preserve
    subject-level ownership in the returned provenance; one row per unique
    subject is required.

    Treatment probabilities and censoring survival are supplied only for
    sequential positivity checks; they are not weights in this plug-in
    estimator. Censoring/dropout must be handled by the caller's prediction
    construction under the declared assumptions.
    """

    q = np.asarray(period_outcome_predictions, dtype=np.float64)
    a = np.asarray(treatment_history)
    p = np.asarray(treatment_probabilities, dtype=np.float64)
    if q.ndim != 2 or q.shape[0] == 0 or q.shape[1] == 0 or not np.isfinite(q).all():
        raise ValueError("period_outcome_predictions must be a non-empty finite (subjects, periods) array")
    if a.ndim != 2 or a.shape != q.shape or not np.isin(a, (0, 1, False, True)).all():
        raise ValueError("treatment_history must be binary and match period_outcome_predictions")
    a = a.astype(bool, copy=False)
    n, periods = q.shape
    if p.shape != q.shape or not np.isfinite(p).all():
        raise ValueError("treatment_probabilities must be finite and match period_outcome_predictions")
    if not isfinite(minimum_probability) or not 0 < minimum_probability <= 0.5:
        raise ValueError("minimum_probability must be finite and in (0, 0.5]")
    if np.any(p < minimum_probability) or np.any(p > 1.0 - minimum_probability):
        raise ValueError("sequential treatment positivity is violated at the declared probability floor")

    ids = tuple(subject_ids)
    if len(ids) != n or any(not isinstance(subject_id, str) or not subject_id.strip() for subject_id in ids):
        raise ValueError("subject_ids must contain one non-empty string per subject row")
    if len(set(ids)) != n:
        raise ValueError("g-formula inputs require one row per unique subject to preserve fold ownership")
    raw_folds = np.asarray(fold_ids)
    if (
        raw_folds.shape != (n,)
        or not np.issubdtype(raw_folds.dtype, np.signedinteger)
        or np.any(raw_folds < np.iinfo(np.int64).min)
        or np.any(raw_folds > np.iinfo(np.int64).max)
        or np.any(raw_folds < 0)
    ):
        raise ValueError("fold_ids must contain one non-negative signed 64-bit integer per subject")
    folds = raw_folds.astype(np.int64, copy=False)

    if callable(regime):
        x = (
            np.empty((n, periods, 0), dtype=np.float64)
            if covariate_history is None
            else np.asarray(covariate_history, dtype=np.float64)
        )
        if x.ndim != 3 or x.shape[:2] != a.shape or not np.isfinite(x).all():
            raise ValueError("dynamic regimes require finite covariate_history shaped (subjects, periods, covariates)")
        actions = np.empty_like(a)
        for i in range(n):
            past: list[bool] = []
            for t in range(periods):
                history = tuple(tuple(float(v) for v in x[i, s]) for s in range(t + 1))
                action = regime(t, tuple(past), history)
                if not isinstance(action, (bool, np.bool_)):
                    raise ValueError("dynamic regime callback must return a bool for every subject-period")
                actions[i, t] = bool(action)
                past.append(bool(a[i, t]))
    else:
        actions = np.asarray(regime)
        if actions.shape != (periods,) or not np.isin(actions, (0, 1, False, True)).all():
            raise ValueError("static regime must contain one binary action per treatment period")
        actions = np.broadcast_to(actions.astype(bool), a.shape).copy()

    censor = np.ones_like(p) if censoring_survival is None else np.asarray(censoring_survival, dtype=np.float64)
    if censor.shape != q.shape or not np.isfinite(censor).all():
        raise ValueError("censoring_survival must be finite and match period_outcome_predictions")
    if np.any(censor < minimum_probability) or np.any(censor > 1.0):
        raise ValueError("sequential censoring positivity is violated at the declared probability floor")
    try:
        value, subject_count, min_action_p, min_censor_p = _native.evaluate_sequential_gformula(
            q,
            actions,
            p,
            censor,
            list(ids),
            folds.tolist(),
            float(minimum_probability),
        )
    except ValueError as error:
        raise ValueError(str(error)) from error
    return GFormulaValue(
        float(value),
        int(subject_count),
        float(min_action_p),
        float(min_censor_p),
        tuple(zip(ids, (int(fold) for fold in folds), strict=True)),
    )


def evaluate_sequential_doubly_robust(
    outcomes: ArrayLike,
    outcome_observed: ArrayLike,
    observation_history: ArrayLike,
    treatment_history: ArrayLike,
    regime: Regime,
    q_predictions: ArrayLike,
    treatment_probabilities: ArrayLike,
    *,
    subject_ids: Sequence[str],
    fold_ids: Sequence[int],
    prediction_fold_ids: Sequence[int],
    covariate_history: ArrayLike | None = None,
    censoring_probabilities: ArrayLike | None = None,
    minimum_probability: float = 0.01,
) -> DoublyRobustRegimeValue:
    """Compute a backward-recursive sequentially doubly robust regime score.

    ``q_predictions[i, t]`` is the caller-supplied cross-fitted conditional
    mean of the next recursive pseudo-outcome for subject ``i`` under the
    regime at period ``t``. ``outcomes`` is the terminal outcome; a finite
    placeholder is ignored when ``outcome_observed`` is false. Observation
    history records whether each subject remains uncensored through each
    period and must be monotone. ``censoring_probabilities[i, t]`` is the
    conditional probability of remaining observed from the preceding period
    through period ``t`` (not the cumulative survival).

    Each subject must have one row, a subject fold ID, and a matching Q
    prediction fold ID. This checks ownership alignment but cannot verify the
    caller actually trained Q outside that subject's fold. No nuisance model
    is fit and no interval is produced.
    """

    y = np.asarray(outcomes, dtype=np.float64)
    terminal_observed = np.asarray(outcome_observed)
    observed = np.asarray(observation_history)
    a = np.asarray(treatment_history)
    q = np.asarray(q_predictions, dtype=np.float64)
    p = np.asarray(treatment_probabilities, dtype=np.float64)
    if q.ndim != 2 or q.shape[0] == 0 or q.shape[1] == 0 or not np.isfinite(q).all():
        raise ValueError("q_predictions must be a non-empty finite (subjects, periods) array")
    n, periods = q.shape
    if y.shape != (n,) or terminal_observed.shape != (n,):
        raise ValueError("outcomes and outcome_observed must contain one value per subject")
    if not np.isin(terminal_observed, (0, 1, False, True)).all():
        raise ValueError("outcome_observed must contain one bool per subject")
    terminal_observed = terminal_observed.astype(bool, copy=False)
    if observed.shape != (n, periods) or not np.isin(observed, (0, 1, False, True)).all():
        raise ValueError("observation_history must be binary and match q_predictions")
    observed = observed.astype(bool, copy=False)
    if np.any((~observed[:, :-1]) & observed[:, 1:]):
        raise ValueError("observation_history must be monotone after censoring or dropout")
    if not np.array_equal(terminal_observed, observed[:, -1]):
        raise ValueError("outcome_observed must match the final observation-history period")
    if not np.isfinite(y[terminal_observed]).all():
        raise ValueError("observed terminal outcomes must be finite")
    y = np.where(terminal_observed, y, 0.0)
    if not terminal_observed.any():
        raise ValueError("at least one terminal outcome must be observed for sequential augmentation")
    if a.shape != (n, periods) or not np.isin(a, (0, 1, False, True)).all():
        raise ValueError("treatment_history must be binary and match q_predictions")
    a = a.astype(bool, copy=False)
    if p.shape != (n, periods) or not np.isfinite(p).all():
        raise ValueError("treatment_probabilities must be finite and match q_predictions")
    if not isfinite(minimum_probability) or not 0 < minimum_probability <= 0.5:
        raise ValueError("minimum_probability must be finite and in (0, 0.5]")
    if np.any(p < minimum_probability) or np.any(p > 1.0 - minimum_probability):
        raise ValueError("sequential treatment positivity is violated at the declared probability floor")

    ids = tuple(subject_ids)
    if len(ids) != n or any(not isinstance(subject_id, str) or not subject_id.strip() for subject_id in ids):
        raise ValueError("subject_ids must contain one non-empty string per subject row")
    if len(set(ids)) != n:
        raise ValueError("doubly robust inputs require one row per unique subject to preserve fold ownership")

    def integer_folds(values: Sequence[int], name: str) -> np.ndarray:
        raw = np.asarray(values)
        if (
            raw.shape != (n,)
            or not np.issubdtype(raw.dtype, np.signedinteger)
            or np.any(raw < np.iinfo(np.int64).min)
            or np.any(raw > np.iinfo(np.int64).max)
            or np.any(raw < 0)
        ):
            raise ValueError(f"{name} must contain one non-negative signed 64-bit integer per subject")
        return raw.astype(np.int64, copy=False)

    folds = integer_folds(fold_ids, "fold_ids")
    prediction_folds = integer_folds(prediction_fold_ids, "prediction_fold_ids")
    if not np.array_equal(folds, prediction_folds):
        raise ValueError("Q prediction fold ownership must match the subject fold IDs")

    if callable(regime):
        x = (
            np.empty((n, periods, 0), dtype=np.float64)
            if covariate_history is None
            else np.asarray(covariate_history, dtype=np.float64)
        )
        if x.ndim != 3 or x.shape[:2] != a.shape or not np.isfinite(x).all():
            raise ValueError("dynamic regimes require finite covariate_history shaped (subjects, periods, covariates)")
        actions = np.empty_like(a)
        for i in range(n):
            past: list[bool] = []
            for t in range(periods):
                if not observed[i, t]:
                    # No residual is taken after dropout, so the dynamic
                    # policy need not inspect unavailable later history.
                    actions[i, t] = False
                    continue
                history = tuple(tuple(float(v) for v in x[i, s]) for s in range(t + 1))
                action = regime(t, tuple(past), history)
                if not isinstance(action, (bool, np.bool_)):
                    raise ValueError("dynamic regime callback must return a bool for every subject-period")
                actions[i, t] = bool(action)
                past.append(bool(a[i, t]))
    else:
        actions = np.asarray(regime)
        if actions.shape != (periods,) or not np.isin(actions, (0, 1, False, True)).all():
            raise ValueError("static regime must contain one binary action per treatment period")
        actions = np.broadcast_to(actions.astype(bool), a.shape).copy()

    censor = (
        np.ones_like(p)
        if censoring_probabilities is None
        else np.asarray(censoring_probabilities, dtype=np.float64)
    )
    if censor.shape != (n, periods) or not np.isfinite(censor).all():
        raise ValueError("censoring_probabilities must be finite and match q_predictions")
    if np.any(censor < minimum_probability) or np.any(censor > 1.0):
        raise ValueError("sequential censoring positivity is violated at the declared probability floor")
    try:
        value, subject_count, min_action_p, min_censor_p = _native.evaluate_sequential_doubly_robust(
            y,
            terminal_observed,
            observed,
            a,
            actions,
            q,
            p,
            censor,
            list(ids),
            folds.tolist(),
            prediction_folds.tolist(),
            float(minimum_probability),
        )
    except ValueError as error:
        raise ValueError(str(error)) from error
    return DoublyRobustRegimeValue(
        float(value),
        int(subject_count),
        float(min_action_p),
        float(min_censor_p),
        tuple(zip(ids, (int(fold) for fold in folds), strict=True)),
    )


def fit_marginal_structural_model(
    outcomes: ArrayLike,
    treatment_history: ArrayLike,
    treatment_probabilities: ArrayLike,
    *,
    stabilizing_numerator_probabilities: ArrayLike,
    subject_ids: Sequence[str],
    fold_ids: Sequence[int] | None = None,
    outcome_observed: ArrayLike | None = None,
    censoring_survival: ArrayLike | None = None,
    minimum_probability: float = 0.01,
) -> MarginalStructuralModelResult:
    """Fit an additive binary marginal structural model with stabilized IPTW.

    Terminal outcomes are one scalar per subject. Each period coefficient is
    the marginal additive effect of its treatment, conditional on other period
    indicators in the specified structural mean. Caller-supplied numerator
    probabilities define stabilization: each is ``P(A_t=1)``. Censoring
    weights are unstabilized and use the supplied conditional probability of
    remaining observed at each period. This function fits no propensities.

    Optional fold IDs record subject ownership, but do not verify whether the
    supplied propensities were estimated out of fold. Standard errors use a
    subject-clustered CR1 sandwich and are pointwise; no interval, simultaneous
    coverage, or calibration claim is made. Identification requires consistency,
    sequential exchangeability and positivity, correctly specified supplied
    nuisance probabilities, no interference, and a correctly specified
    additive marginal structural mean with no treatment interactions.
    """
    y = np.asarray(outcomes, dtype=np.float64)
    a = np.asarray(treatment_history)
    p = np.asarray(treatment_probabilities, dtype=np.float64)
    numerator = np.asarray(stabilizing_numerator_probabilities, dtype=np.float64)
    if y.ndim != 1 or y.size == 0:
        raise ValueError("outcomes must be a non-empty one-dimensional subject vector")
    if a.ndim != 2 or a.shape[0] != y.size or a.shape[1] == 0:
        raise ValueError("treatment_history must have shape (subjects, periods)")
    if not np.isin(a, (0, 1, False, True)).all():
        raise ValueError("treatment_history must contain only binary actions")
    a = a.astype(bool, copy=False)
    n, periods = a.shape
    if p.shape != a.shape or not np.isfinite(p).all():
        raise ValueError("treatment_probabilities must be finite and match treatment_history")
    if numerator.shape != (periods,) or not np.isfinite(numerator).all():
        raise ValueError("stabilizing_numerator_probabilities must have one finite value per period")
    if not isfinite(minimum_probability) or not 0 < minimum_probability <= 0.5:
        raise ValueError("minimum_probability must be finite and in (0, 0.5]")
    if np.any(p < minimum_probability) or np.any(p > 1.0 - minimum_probability):
        raise ValueError("sequential treatment positivity is violated at the declared probability floor")
    if np.any(numerator < minimum_probability) or np.any(numerator > 1.0 - minimum_probability):
        raise ValueError("stabilizing numerator probabilities violate the declared positivity floor")
    observed = np.ones(n, dtype=bool) if outcome_observed is None else np.asarray(outcome_observed)
    if observed.shape != (n,) or not np.isin(observed, (0, 1, False, True)).all():
        raise ValueError("outcome_observed must contain one bool per subject")
    observed = observed.astype(bool, copy=False)
    if not observed.any():
        raise ValueError("at least one terminal outcome must be observed")
    if not np.isfinite(y[observed]).all():
        raise ValueError("observed terminal outcomes must be finite")
    y = np.where(observed, y, 0.0)
    censor = np.ones_like(p) if censoring_survival is None else np.asarray(censoring_survival, dtype=np.float64)
    if censor.shape != a.shape or not np.isfinite(censor).all():
        raise ValueError("censoring_survival must be finite and match treatment_history")
    if np.any(censor < minimum_probability) or np.any(censor > 1.0):
        raise ValueError("sequential censoring positivity is violated at the declared probability floor")

    ids = tuple(subject_ids)
    if len(ids) != n or any(not isinstance(value, str) or not value.strip() for value in ids):
        raise ValueError("subject_ids must contain one non-empty string per subject")
    if len(set(ids)) != n:
        raise ValueError("subject_ids must be unique because each row is one subject cluster")
    if fold_ids is None:
        ownership: tuple[tuple[str, int], ...] = ()
        crossfit = "fold IDs not supplied; supplied propensity data dependence is unknown"
    else:
        raw_folds = np.asarray(fold_ids)
        if (
            raw_folds.shape != (n,)
            or not np.issubdtype(raw_folds.dtype, np.signedinteger)
            or np.any(raw_folds < 0)
        ):
            raise ValueError("fold_ids must contain one non-negative integer per subject")
        ownership = tuple((subject, int(fold)) for subject, fold in zip(ids, raw_folds, strict=True))
        crossfit = "subject fold ownership recorded; propensity cross-fitting is declared by caller and not verified"

    try:
        intercept, effects, standard_errors, ess, max_weight, included = _native.fit_binary_msm(
            y,
            a,
            p,
            numerator,
            observed,
            censor,
            float(minimum_probability),
        )
    except ValueError as error:
        raise ValueError(str(error)) from error
    return MarginalStructuralModelResult(
        float(intercept),
        tuple(float(value) for value in effects),
        tuple(float(value) for value in standard_errors),
        tuple(float(value) for value in numerator),
        float(ess),
        float(max_weight),
        int(included),
        ownership,
        crossfit_status=crossfit,
    )


__all__ = [
    "GFormulaValue",
    "DoublyRobustRegimeValue",
    "HistoryPolicy",
    "MarginalStructuralModelResult",
    "Regime",
    "RegimeValue",
    "evaluate_regime_value",
    "fit_marginal_structural_model",
    "evaluate_sequential_doubly_robust",
    "evaluate_sequential_gformula",
]
