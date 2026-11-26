"""Point evaluation of prespecified longitudinal treatment regimes.

This low-level utility evaluates static or history-adaptive binary regimes using
user-supplied sequential treatment and censoring probabilities. The regime
value and sequential g-formula paths are point-only; the binary MSM path
returns pointwise subject-clustered standard errors. No path fits the nuisance
probabilities or adds a support-matrix support.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from math import isfinite
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _deprecation, _direct_status, _native
from ._columns import coerce_fields, hold_unresolved, resolved
from .errors import CausalValueError
from .execution import Execution, resolve_execution

HistoryPolicy: TypeAlias = Callable[[int, tuple[bool, ...], tuple[tuple[float, ...], ...]], bool]
Regime: TypeAlias = Sequence[bool] | HistoryPolicy


@dataclass(frozen=True, slots=True)
class FittedRegimeQ:
    """A fitted terminal Q mean within a binary pre-treatment baseline stratum.

    The fitted g-formula scope is a common two-decision schedule, fair-coin
    randomization, and complete outcomes. Each replicate resamples whole subjects
    in their folds and refits Q. An external training fold identifies a separate
    sample and the other fold provides the target baseline distribution. Without
    a training fold, every fold is held out in turn.
    """

    baseline: str
    bootstrap_replicates: int = 399
    external_training_fold: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.baseline, str) or not self.baseline.strip():
            raise CausalValueError("fitted Q needs a named binary baseline column")
        if (
            type(self.bootstrap_replicates) is not int
            or not 99 <= self.bootstrap_replicates <= 100_000
        ):
            raise CausalValueError("fitted Q needs 99 to 100000 whole-subject bootstrap replicates")
        if self.external_training_fold is not None and (
            type(self.external_training_fold) is not int
            or not 0 <= self.external_training_fold < 2**32
        ):
            raise CausalValueError("external_training_fold must be a nonnegative 32-bit fold ID")


Schedule = tuple[bool, bool]


def _schedule_code(schedule: Schedule) -> int:
    if len(schedule) != 2 or any(action not in (False, True) for action in schedule):
        raise CausalValueError("each candidate schedule must have two binary actions")
    return 2 * int(schedule[0]) + int(schedule[1])


def _schedule_mask(schedules: Sequence[Schedule]) -> int:
    codes = tuple(map(_schedule_code, schedules))
    if len(set(codes)) != len(codes):
        raise CausalValueError("schedule declarations may name each schedule once")
    return sum(1 << code for code in codes)


@dataclass(frozen=True, slots=True)
class FixedScheduleContrasts:
    """Prespecified candidates and simultaneous differences from one reference."""

    schedules: tuple[Schedule, ...]
    reference: Schedule
    contrasts: tuple[Schedule, ...]
    draws: int = 399

    def _wire(self) -> tuple[int, int, tuple[int, int], None]:
        family = _schedule_mask(self.schedules)
        reference = _schedule_code(self.reference)
        contrasts = _schedule_mask(self.contrasts)
        if (
            family.bit_count() < 2
            or not family & (1 << reference)
            or not contrasts
            or contrasts & ~family
            or contrasts & (1 << reference)
        ):
            raise CausalValueError(
                "declare a candidate reference and distinct nonreference contrasts from the class"
            )
        return family, self.draws, (reference, contrasts), None


@dataclass(frozen=True, slots=True)
class LearnedScheduleClass:
    """Choose a fixed schedule using only outcomes from the declared training fold.

    Candidate actions are fixed sequences, so the selection uses no subject information at
    the individual treatment decisions. The other fold evaluates the selected regime and
    all prespecified pairs. Selection breaks training-score ties by binary schedule order.
    """

    schedules: tuple[Schedule, ...]
    training_fold: int
    draws: int = 399

    def _wire(self) -> tuple[int, int, None, int]:
        family = _schedule_mask(self.schedules)
        if (
            family.bit_count() < 2
            or type(self.training_fold) is not int
            or not 0 <= self.training_fold < 2**32
        ):
            raise CausalValueError("declare at least two schedules and one training fold")
        return family, self.draws, None, self.training_fold


LongitudinalCalibrationDesign = Literal[
    "four_period_sequential_dr",
    "fitted_q_internal_crossfit",
    "fitted_q_external_split",
    "fixed_schedule_family",
    "learned_schedule_unique_optimum",
    "learned_schedule_tied_optima",
]


@dataclass(frozen=True, slots=True)
class LongitudinalRegime:
    """Prespecified binary regime over complete subject histories.

    Each outcome row belongs to one subject. Assignment probabilities are
    known from the sequential randomized design. Interval construction and
    support depend on the declared method and its subject histories.
    """

    outcome: str
    treatment_history: Sequence[Sequence[bool]]
    actions: Sequence[bool] | Sequence[Sequence[bool]]
    treatment_probabilities: Sequence[Sequence[float]]
    subject_ids: Sequence[str] | str
    method: Literal[
        "ipw",
        "g_formula",
        "fitted_g_formula",
        "compare_schedules",
        "learn_schedule",
        "sequential_dr",
        "marginal_structural_model",
    ] = "ipw"
    period_outcome_predictions: Sequence[Sequence[float]] | None = None
    known_fixed_outcome_predictions: bool = False
    fitted_q: FittedRegimeQ | None = None
    schedule_family: FixedScheduleContrasts | LearnedScheduleClass | None = None
    stabilizing_numerator_probabilities: Sequence[float] | None = None
    q_predictions: Sequence[Sequence[float]] | None = None
    observation_history: Sequence[Sequence[bool]] | None = None
    prediction_fold_ids: Sequence[int] | str | None = None
    censoring_probabilities: Sequence[Sequence[float]] | None = None
    outcome_observed: Sequence[bool] | str | None = None
    fold_ids: Sequence[int] | str | None = None
    excluded_fold_predictions: bool = False
    probabilities_known_by_design: bool = True
    minimum_probability: float = 0.01
    rule_id: str | None = None
    rule_version: str | None = None
    rule_provenance: str | None = None
    #: Numeric columns the prepared study retains beside the outcome, for follow-up operations
    #: on the study's own rows (the refitted weight models of
    #: ``antecedent.validation.sensitivity_range(study, "msm_parameters_refit", ...)``). No
    #: estimator of the regime reads them; a study that declares none retains the outcome alone.
    covariates: Sequence[str] = ()
    #: Name the prespecified study law only if it applies to these data and this fit.
    #: Undeclared or mismatched designs retain a nominal reading.
    calibration_design: LongitudinalCalibrationDesign | None = None
    kind: Literal["longitudinal_regime"] = field(
        default="longitudinal_regime", init=False, repr=False
    )
    _deferred_columns: bool = field(default=False, init=False, repr=False, compare=False)

    #: Row-aligned (per-subject) inputs that may instead name a data column. The
    #: subject-by-period matrices (treatment history/probabilities, predictions,
    #: Q values) are not single columns and stay inline.
    _COLUMN_FIELDS = {
        "subject_ids": "str",
        "fold_ids": "int",
        "outcome_observed": "bool",
        "prediction_fold_ids": "int",
    }

    def __post_init__(self) -> None:
        if hold_unresolved(self):
            return
        if not isinstance(self.outcome, str) or not self.outcome.strip():
            raise CausalValueError("outcome must name a non-empty column")
        if self.calibration_design not in (
            None,
            "four_period_sequential_dr",
            "fitted_q_internal_crossfit",
            "fitted_q_external_split",
            "fixed_schedule_family",
            "learned_schedule_unique_optimum",
            "learned_schedule_tied_optima",
        ):
            raise CausalValueError("unknown longitudinal calibration design")
        covariates = (
            (self.covariates,) if isinstance(self.covariates, str) else tuple(self.covariates)
        )
        if (
            any(not isinstance(name, str) or not name.strip() for name in covariates)
            or len(set(covariates)) != len(covariates)
            or self.outcome in covariates
        ):
            raise CausalValueError(
                "covariates must name distinct non-empty columns other than the outcome"
            )
        object.__setattr__(self, "covariates", covariates)
        history = np.asarray(self.treatment_history)
        if history.ndim != 2 or not history.size or not np.isin(history, (0, 1, False, True)).all():
            raise CausalValueError(
                "treatment_history must be a non-empty binary subject-by-period array"
            )
        n, periods = history.shape
        identity = (self.rule_id, self.rule_version, self.rule_provenance)
        if any(value is not None for value in identity):
            if any(not isinstance(value, str) or not value.strip() for value in identity):
                raise CausalValueError(
                    "dynamic rule requires non-empty rule_id, rule_version, and rule_provenance"
                )
            if self.method == "marginal_structural_model":
                raise CausalValueError(
                    "dynamic rule identity is not applicable to marginal_structural_model"
                )
        if self.method not in (
            "ipw",
            "g_formula",
            "fitted_g_formula",
            "compare_schedules",
            "learn_schedule",
            "sequential_dr",
            "marginal_structural_model",
        ):
            raise CausalValueError("unknown longitudinal method")
        if self.method == "fitted_g_formula":
            if (
                not isinstance(self.fitted_q, FittedRegimeQ)
                or self.fitted_q.baseline == self.outcome
            ):
                raise CausalValueError(
                    "fitted_g_formula needs a fitted Q declaration with a distinct baseline"
                )
        elif self.fitted_q is not None:
            raise CausalValueError("fitted_q requires fitted_g_formula")
        if self.method in ("compare_schedules", "learn_schedule"):
            expected = (
                FixedScheduleContrasts
                if self.method == "compare_schedules"
                else LearnedScheduleClass
            )
            if not isinstance(self.schedule_family, expected):
                raise CausalValueError(
                    "schedule_family must match the comparison or learning method"
                )
            if (
                type(self.schedule_family.draws) is not int
                or not 199 <= self.schedule_family.draws <= 100_000
            ):
                raise CausalValueError("schedule inference needs 199 to 100000 whole-subject draws")
            self.schedule_family._wire()
        elif self.schedule_family is not None:
            raise CausalValueError("schedule_family requires compare_schedules or learn_schedule")
        numerator = (
            None
            if self.stabilizing_numerator_probabilities is None
            else np.asarray(self.stabilizing_numerator_probabilities, dtype=np.float64)
        )
        if self.method == "marginal_structural_model":
            if (
                numerator is None
                or numerator.shape != (periods,)
                or not np.isfinite(numerator).all()
            ):
                raise CausalValueError(
                    "marginal_structural_model requires one finite stabilizing numerator probability per period"
                )
        elif numerator is not None:
            raise CausalValueError(
                "stabilizing numerator probabilities require marginal_structural_model"
            )
        predictions = (
            None
            if self.period_outcome_predictions is None
            else np.asarray(self.period_outcome_predictions, dtype=np.float64)
        )
        if self.method == "g_formula":
            if (
                predictions is None
                or predictions.shape != (n, periods)
                or not np.isfinite(predictions).all()
            ):
                raise CausalValueError(
                    "g_formula requires finite subject-by-period outcome predictions"
                )
        elif predictions is not None:
            raise CausalValueError("period_outcome_predictions require g_formula")
        if not isinstance(self.known_fixed_outcome_predictions, bool) or (
            self.known_fixed_outcome_predictions and self.method != "g_formula"
        ):
            raise CausalValueError(
                "known_fixed_outcome_predictions requires g_formula and a bool declaration"
            )
        q = None if self.q_predictions is None else np.asarray(self.q_predictions, dtype=np.float64)
        observation = (
            None if self.observation_history is None else np.asarray(self.observation_history)
        )
        if self.method == "sequential_dr":
            if q is None or q.shape != (n, periods) or not np.isfinite(q).all():
                raise CausalValueError(
                    "sequential_dr requires finite subject-by-period Q predictions"
                )
            if (
                observation is None
                or observation.shape != (n, periods)
                or not np.isin(observation, (0, 1, False, True)).all()
            ):
                raise CausalValueError(
                    "sequential_dr requires binary subject-by-period observation history"
                )
            if self.prediction_fold_ids is None:
                raise CausalValueError("sequential_dr requires Q prediction fold ownership")
        elif q is not None or observation is not None or self.prediction_fold_ids is not None:
            raise CausalValueError("Q predictions and observation history require sequential_dr")
        actions = np.asarray(self.actions)
        if actions.shape == (periods,):
            actions = np.broadcast_to(actions, (n, periods))
        if actions.shape != (n, periods) or not np.isin(actions, (0, 1, False, True)).all():
            raise CausalValueError(
                "actions must be binary, with one period row or one subject-by-period row"
            )
        probabilities = np.asarray(self.treatment_probabilities, dtype=np.float64)
        if probabilities.shape != (n, periods) or not np.isfinite(probabilities).all():
            raise CausalValueError("treatment_probabilities must match treatment_history")
        censor = (
            np.ones((n, periods))
            if self.censoring_probabilities is None
            else np.asarray(self.censoring_probabilities, dtype=np.float64)
        )
        if censor.shape != (n, periods) or not np.isfinite(censor).all():
            raise CausalValueError("censoring_probabilities must match treatment_history")
        observed = (
            np.ones(n, dtype=bool)
            if self.outcome_observed is None
            else np.asarray(self.outcome_observed)
        )
        if observed.shape != (n,) or not np.isin(observed, (0, 1, False, True)).all():
            raise CausalValueError("outcome_observed must contain one binary value per subject")
        ids = tuple(self.subject_ids)
        if (
            len(ids) != n
            or any(not isinstance(value, str) or not value.strip() for value in ids)
            or len(set(ids)) != n
        ):
            raise CausalValueError(
                "subject_ids must contain one distinct non-empty ID per outcome row"
            )
        folds = (0,) * n if self.fold_ids is None else tuple(self.fold_ids)
        if len(folds) != n or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > 2**32 - 1
            for value in folds
        ):
            raise CausalValueError("fold_ids must contain one non-negative u32 per subject")
        if self.excluded_fold_predictions and len(set(folds)) < 2:
            raise CausalValueError("excluded-fold predictions require at least two subject folds")
        if self.method == "sequential_dr":
            # The first sequential_dr block above rejected a missing observation
            # history or fold ownership, so both are present here.
            assert observation is not None
            assert self.prediction_fold_ids is not None
            if not self.excluded_fold_predictions:
                raise CausalValueError(
                    "sequential_dr requires declared excluded-fold Q predictions"
                )
            if tuple(resolved(self.prediction_fold_ids)) != folds:
                raise CausalValueError("Q prediction fold ownership must match subject folds")
            if not np.array_equal(observation[:, -1], observed):
                raise CausalValueError(
                    "terminal observation must match the final observation-history period"
                )
            if np.any((~observation[:, :-1].astype(bool)) & observation[:, 1:].astype(bool)):
                raise CausalValueError("observation_history must be monotone after dropout")
            if not observed.any():
                raise CausalValueError(
                    "sequential_dr requires at least one observed terminal outcome"
                )
        if not self.probabilities_known_by_design and not self.excluded_fold_predictions:
            raise CausalValueError(
                "probabilities require known sequential randomization or excluded-fold predictions"
            )
        floor = self.minimum_probability
        if (
            not isfinite(floor)
            or not 0 < floor <= 0.5
            or np.any(probabilities < floor)
            or np.any(probabilities > 1 - floor)
            or np.any(censor < floor)
            or np.any(censor > 1)
        ):
            raise CausalValueError(
                "sequential treatment or censoring positivity fails at the declared floor"
            )
        if numerator is not None and (np.any(numerator < floor) or np.any(numerator > 1 - floor)):
            raise CausalValueError(
                "stabilizing numerator probabilities violate the declared positivity floor"
            )
        if self.method == "marginal_structural_model" and np.count_nonzero(observed) <= periods + 1:
            raise CausalValueError("MSM requires more observed subjects than coefficients")
        coerce_fields(self, treatment_history=tuple(tuple(map(bool, row)) for row in history))
        coerce_fields(self, actions=tuple(tuple(map(bool, row)) for row in actions))
        coerce_fields(
            self, treatment_probabilities=tuple(tuple(map(float, row)) for row in probabilities)
        )
        coerce_fields(self, censoring_probabilities=tuple(tuple(map(float, row)) for row in censor))
        coerce_fields(self, outcome_observed=tuple(map(bool, observed)))
        coerce_fields(self, subject_ids=ids)
        coerce_fields(self, fold_ids=folds)
        if predictions is not None:
            coerce_fields(
                self,
                period_outcome_predictions=tuple(tuple(map(float, row)) for row in predictions),
            )
        if numerator is not None:
            coerce_fields(self, stabilizing_numerator_probabilities=tuple(map(float, numerator)))
        if q is not None:
            # Q predictions only pass validation for sequential_dr, which also
            # requires the observation history and fold ownership set here.
            assert observation is not None
            assert self.prediction_fold_ids is not None
            coerce_fields(self, q_predictions=tuple(tuple(map(float, row)) for row in q))
            coerce_fields(
                self, observation_history=tuple(tuple(map(bool, row)) for row in observation)
            )
            coerce_fields(self, prediction_fold_ids=tuple(resolved(self.prediction_fold_ids)))

    @property
    def periods(self) -> int:
        """The number of periods in each subject's treatment history."""
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
        known_fixed_outcome_predictions: bool = False,
        q_predictions: Sequence[Sequence[float]] | None = None,
        observation_history: Sequence[Sequence[bool]] | None = None,
        prediction_fold_ids: Sequence[int] | None = None,
        censoring_probabilities: Sequence[Sequence[float]] | None = None,
        outcome_observed: Sequence[bool] | None = None,
        fold_ids: Sequence[int] | None = None,
        excluded_fold_predictions: bool = False,
        probabilities_known_by_design: bool = True,
        minimum_probability: float = 0.01,
    ) -> LongitudinalRegime:
        """Freeze a binary rule against observed pre-decision histories.

        At period ``t``, ``rule(t, past_actions, covariates_through_t)`` sees
        actions from periods ``<t`` and covariates from periods ``<=t`` only.
        The callable is caller code: artifacts retain its identity and resolved
        actions, but cannot replay or verify its implementation.
        """
        history = np.asarray(treatment_history)
        if history.ndim != 2 or not history.size or not np.isin(history, (0, 1, False, True)).all():
            raise CausalValueError(
                "treatment_history must be a non-empty binary subject-by-period array"
            )
        n, periods = history.shape
        covariates = np.asarray(predecision_covariates, dtype=np.float64)
        if covariates.ndim != 3 or covariates.shape[:2] != (n, periods) or covariates.shape[2] == 0:
            raise CausalValueError(
                "predecision_covariates must be finite subject-by-period-by-feature values"
            )
        observed = (
            np.ones((n, periods), dtype=bool)
            if observation_history is None
            else np.asarray(observation_history)
        )
        if observed.shape != (n, periods) or not np.isin(observed, (0, 1, False, True)).all():
            raise CausalValueError("observation_history must be binary subject-by-period values")
        observed = observed.astype(bool)
        if np.any((~observed[:, :-1]) & observed[:, 1:]):
            raise CausalValueError("observation_history must be monotone after dropout")
        if not np.isfinite(covariates[observed]).all():
            raise CausalValueError("observed predecision_covariates must be finite")
        if not callable(rule):
            raise CausalValueError("rule must be callable")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (rule_id, rule_version, rule_provenance)
        ):
            raise CausalValueError(
                "dynamic rule requires non-empty rule_id, rule_version, and rule_provenance"
            )
        actions: list[list[bool]] = []
        for subject in range(n):
            row: list[bool] = []
            for period in range(periods):
                if not observed[subject, period]:
                    row.append(False)
                    continue
                past = tuple(bool(value) for value in history[subject, :period])
                available = tuple(
                    tuple(float(value) for value in covariates[subject, t])
                    for t in range(period + 1)
                )
                decision = rule(period, past, available)
                if not isinstance(decision, (bool, np.bool_)):
                    raise CausalValueError(
                        "dynamic rule must return a binary bool at every decision"
                    )
                row.append(bool(decision))
            actions.append(row)
        return cls(
            outcome=outcome,
            treatment_history=treatment_history,
            actions=actions,
            treatment_probabilities=treatment_probabilities,
            subject_ids=subject_ids,
            method=method,
            period_outcome_predictions=period_outcome_predictions,
            known_fixed_outcome_predictions=known_fixed_outcome_predictions,
            q_predictions=q_predictions,
            observation_history=observation_history,
            prediction_fold_ids=prediction_fold_ids,
            censoring_probabilities=censoring_probabilities,
            outcome_observed=outcome_observed,
            fold_ids=fold_ids,
            excluded_fold_predictions=excluded_fold_predictions,
            probabilities_known_by_design=probabilities_known_by_design,
            minimum_probability=minimum_probability,
            rule_id=rule_id,
            rule_version=rule_version,
            rule_provenance=rule_provenance,
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
    ) -> LongitudinalRegime:
        """Specify an additive MSM without an irrelevant regime-action argument.

        The model still uses the ordinary subject-history Study query and the
        same validation, artifacts, and pointwise CR1 result contract.
        """
        history = np.asarray(treatment_history)
        if history.ndim != 2 or history.shape[1] == 0:
            raise CausalValueError("treatment_history must be a non-empty subject-by-period array")
        return cls(
            outcome=outcome,
            treatment_history=treatment_history,
            actions=[False] * history.shape[1],
            treatment_probabilities=treatment_probabilities,
            subject_ids=subject_ids,
            method="marginal_structural_model",
            stabilizing_numerator_probabilities=stabilizing_numerator_probabilities,
            censoring_probabilities=censoring_probabilities,
            outcome_observed=outcome_observed,
            fold_ids=fold_ids,
            excluded_fold_predictions=excluded_fold_predictions,
            probabilities_known_by_design=probabilities_known_by_design,
            minimum_probability=minimum_probability,
        )


@dataclass(frozen=True, slots=True)
class ScheduleContrast:
    """One coordinate of the paired simultaneous schedule interval."""

    schedule: int
    reference: int
    difference: float
    standard_error: float
    interval: tuple[float, float]


@dataclass(frozen=True, slots=True)
class ScheduleLearningResult:
    """Selected schedule, compatible optima and training-refit variability."""

    selected_schedule: int
    optimum_set: tuple[int, ...]
    refit_selections: tuple[tuple[int, int], ...]
    refit_value_sd: float


@dataclass(frozen=True, slots=True)
class LongitudinalScheduleResult:
    """Paired candidate values and simultaneous contrasts in two-bit schedule order."""

    values: tuple[tuple[int, float], ...]
    contrasts: tuple[ScheduleContrast, ...]
    critical_value: float
    evaluation_subjects: int
    learning: ScheduleLearningResult | None


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
    fitted_q_bootstrap_replicates_ok: int | None = None
    schedule_family: LongitudinalScheduleResult | None = None
    period_intervals_95: tuple[tuple[float, float], ...] | None = None
    interval_reason: str | None = None
    method: str = "ipw"
    uncertainty: str = "point_only_no_interval"
    probability_ownership: str = "known_sequential_randomization"
    support_status: str = _direct_status.UNSUPPORTED_POINT_UTILITY
    period_effects: tuple[float, ...] | None = None
    standard_errors: tuple[float, ...] | None = None
    stabilizing_numerator_probabilities: tuple[float, ...] | None = None
    observed_subjects: int | None = None
    rule_id: str | None = None
    rule_version: str | None = None
    rule_provenance: str | None = None


@dataclass(frozen=True, slots=True)
class SequentialContrast:
    """Observational contrast of two static two-decision regimes, one row per subject.

    The first binary action, the covariates measured after it, the second binary action and the
    terminal outcome are data columns. ``active`` and ``control`` are the actions ``(first,
    second)`` of each regime and have no default. ``folds`` is a fold count (row ``i`` goes to fold
    ``i % folds``) or one fold index per row for the cross-fitting. Sequential conditional
    exchangeability given the measured history, positivity, consistency and no interference are
    declared, not derived. Frequentist inference reports the normal interval of the cross-fit
    contrast; ``Bayesian`` inference reports the modular Bayesian-bootstrap posterior of the same
    scores. Either is supported only for the always-treat against never-treat contrast with at
    least five folds (none above a fifth of the rows), at least 200 subjects and fitted action
    probabilities at least 0.05 (and at least 2,000 draws for the posterior): the interval's
    support is ``SequentialContrastEstimate.supported`` and the posterior's is the
    ``estimate.design_posterior.graphless_support`` diagnostic. Every other fit is unsupported.
    """

    outcome: str
    first_treatment: str
    second_treatment: str
    active: tuple[int, int]
    control: tuple[int, int]
    baseline: Sequence[str] = ()
    intermediate: Sequence[str] = ()
    folds: int | Sequence[int] = 2
    min_probability: float = 0.05
    kind: Literal["sequential_contrast"] = field(
        default="sequential_contrast", init=False, repr=False
    )

    def __post_init__(self) -> None:
        names = [self.outcome, self.first_treatment, self.second_treatment]
        names.extend(self.baseline)
        names.extend(self.intermediate)
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise CausalValueError("every column name must be a non-empty string")
        if len(set(names)) != len(names):
            raise CausalValueError(
                "outcome, both actions and every covariate must be distinct columns"
            )
        for label, regime in (("active", self.active), ("control", self.control)):
            if len(regime) != 2 or any(action not in (0, 1) for action in regime):
                raise CausalValueError(f"{label} must be two actions, each 0 or 1")
        if isinstance(self.folds, bool):
            raise CausalValueError("folds must be a count or one fold per row")
        if isinstance(self.folds, int):
            if self.folds < 2:
                raise CausalValueError("folds must be at least 2")
        elif len(self.folds) < 2 or any(int(fold) < 0 for fold in self.folds):
            raise CausalValueError("folds must be a count or one non-negative fold per row")
        if not (isfinite(self.min_probability) and 0.0 < self.min_probability <= 0.5):
            raise CausalValueError("min_probability must be in (0, 0.5]")
        coerce_fields(self, baseline=tuple(self.baseline), intermediate=tuple(self.intermediate))


@dataclass(frozen=True, slots=True)
class SequentialContrastEstimate:
    """Cross-fit contrast of two regimes with its scores.

    ``supported`` is ``True`` only when the normal interval meets the sequential-contrast support
    row: the always-treat against never-treat contrast, at least five folds with none above a
    fifth of the rows, at least 200 subjects and every fitted regime-action probability at least
    0.05. A posterior states its row in the result's diagnostics instead. Every other fit is
    unsupported.
    """

    active_regime: tuple[int, int]
    control_regime: tuple[int, int]
    contrast: float
    active_mean: float
    control_mean: float
    evaluated_rows: int
    minimum_action_probability: float
    min_probability: float
    coverage: float
    active_scores: tuple[float, ...]
    control_scores: tuple[float, ...]
    contrast_scores: tuple[float, ...]
    uncertainty: str
    #: Normal-interval report: standard error and interval of the mean contrast score.
    standard_error: float | None = None
    interval: tuple[float, float] | None = None
    fits_computed: int = 0
    fits_reused: int = 0
    supported: bool = False
    calibration_measured: bool = False
    support_status: str = _direct_status.UNSUPPORTED_UNCALIBRATED
    assumptions: tuple[str, ...] = (
        "sequential_conditional_exchangeability_given_measured_history",
        "positivity_at_each_decision",
        "consistency_and_no_interference",
        "no_unmeasured_second_decision_confounder",
        "nuisance_product_rate_o_n_minus_half",
    )

    #: Symmetric credible interval around the contrast from the Bayesian-bootstrap draws.
    posterior_symmetric_interval: tuple[float, float] | None = None


@dataclass(frozen=True, slots=True)
class RegimeValue:
    """Uncertainty-free value summary for a longitudinal regime."""

    value: float
    effective_sample_size: float
    matched_observed_fraction: float
    maximum_weight: float
    uncertainty: str = "not_estimated"
    support_status: str = _direct_status.CALLER_SUPPLIED_SEQUENTIAL_PROBABILITIES
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
    support_status: str = _direct_status.CALLER_SUPPLIED_CONDITIONAL_OUTCOME_PREDICTIONS
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
    support_status: str = _direct_status.CALLER_SUPPLIED_SEQUENTIAL_Q_AND_PROBABILITIES
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
    uncertainty_semantics: str = (
        "pointwise CR1 standard errors; no confidence intervals; no simultaneous coverage claim"
    )
    crossfit_status: str = (
        "caller_supplied propensity data dependence is not independently verified"
    )
    support_status: str = _direct_status.UNSUPPORTED_POINT_UTILITY
    assumptions: tuple[str, ...] = (
        "consistency for the observed longitudinal treatment histories",
        "sequential exchangeability conditional on the histories used for treatment probabilities",
        "sequential positivity for treatment and remaining uncensored",
        "the supplied treatment and censoring probabilities are correctly specified",
        "the additive marginal structural mean is correctly specified; period effects are additive and have no interactions",
        "no interference between subjects",
    )


def _evaluate_regime_value_direct(
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
        raise CausalValueError("outcomes must be a non-empty finite one-dimensional array")
    if a.ndim != 2 or a.shape[0] != y.size or a.shape[1] == 0:
        raise CausalValueError("treatment_history must have shape (subjects, periods)")
    if not np.isin(a, (0, 1, False, True)).all():
        raise CausalValueError("treatment_history must contain only binary actions")
    a = a.astype(bool, copy=False)
    n, periods = a.shape
    if p.shape != a.shape or not np.isfinite(p).all():
        raise CausalValueError("treatment_probabilities must be finite and match treatment_history")
    if not isfinite(minimum_probability) or not 0 < minimum_probability <= 0.5:
        raise CausalValueError("minimum_probability must be finite and in (0, 0.5]")
    if np.any(p < minimum_probability) or np.any(p > 1.0 - minimum_probability):
        raise CausalValueError(
            "sequential treatment positivity is violated at the declared probability floor"
        )
    if callable(regime):
        x = (
            np.empty((n, periods, 0), dtype=np.float64)
            if covariate_history is None
            else np.asarray(covariate_history, dtype=np.float64)
        )
        if x.ndim != 3 or x.shape[:2] != a.shape or not np.isfinite(x).all():
            raise CausalValueError(
                "dynamic regimes require finite covariate_history shaped (subjects, periods, covariates)"
            )
        actions = np.empty_like(a)
        for i in range(n):
            past: list[bool] = []
            for t in range(periods):
                history = tuple(tuple(float(v) for v in x[i, s]) for s in range(t + 1))
                action = regime(t, tuple(past), history)
                if not isinstance(action, (bool, np.bool_)):
                    raise CausalValueError(
                        "dynamic regime callback must return a bool for every subject-period"
                    )
                actions[i, t] = bool(action)
                past.append(bool(a[i, t]))
    else:
        actions = np.asarray(regime)
        if actions.shape != (periods,) or not np.isin(actions, (0, 1, False, True)).all():
            raise CausalValueError(
                "static regime must contain one binary action per treatment period"
            )
        actions = np.broadcast_to(actions.astype(bool), a.shape).copy()
    observed = np.ones(n, dtype=bool) if outcome_observed is None else np.asarray(outcome_observed)
    if observed.shape != (n,) or not np.isin(observed, (0, 1, False, True)).all():
        raise CausalValueError("outcome_observed must contain one bool per subject")
    observed = observed.astype(bool, copy=False)
    censor = (
        np.ones_like(p)
        if censoring_survival is None
        else np.asarray(censoring_survival, dtype=np.float64)
    )
    if censor.shape != a.shape or not np.isfinite(censor).all():
        raise CausalValueError("censoring_survival must be finite and match treatment_history")
    if np.any(censor < minimum_probability) or np.any(censor > 1.0):
        raise CausalValueError(
            "sequential censoring positivity is violated at the declared probability floor"
        )
    value, ess, matched, max_weight = _native.evaluate_longitudinal_regime_value(
        y, a, actions, p, observed, censor
    )
    return RegimeValue(value, ess, matched, max_weight)


def _evaluate_sequential_gformula_direct(
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
        raise CausalValueError(
            "period_outcome_predictions must be a non-empty finite (subjects, periods) array"
        )
    if a.ndim != 2 or a.shape != q.shape or not np.isin(a, (0, 1, False, True)).all():
        raise CausalValueError(
            "treatment_history must be binary and match period_outcome_predictions"
        )
    a = a.astype(bool, copy=False)
    n, periods = q.shape
    if p.shape != q.shape or not np.isfinite(p).all():
        raise CausalValueError(
            "treatment_probabilities must be finite and match period_outcome_predictions"
        )
    if not isfinite(minimum_probability) or not 0 < minimum_probability <= 0.5:
        raise CausalValueError("minimum_probability must be finite and in (0, 0.5]")
    if np.any(p < minimum_probability) or np.any(p > 1.0 - minimum_probability):
        raise CausalValueError(
            "sequential treatment positivity is violated at the declared probability floor"
        )

    ids = tuple(subject_ids)
    if len(ids) != n or any(
        not isinstance(subject_id, str) or not subject_id.strip() for subject_id in ids
    ):
        raise CausalValueError("subject_ids must contain one non-empty string per subject row")
    if len(set(ids)) != n:
        raise CausalValueError(
            "g-formula inputs require one row per unique subject to preserve fold ownership"
        )
    raw_folds = np.asarray(fold_ids)
    if (
        raw_folds.shape != (n,)
        or not np.issubdtype(raw_folds.dtype, np.signedinteger)
        or np.any(raw_folds < np.iinfo(np.int64).min)
        or np.any(raw_folds > np.iinfo(np.int64).max)
        or np.any(raw_folds < 0)
    ):
        raise CausalValueError(
            "fold_ids must contain one non-negative signed 64-bit integer per subject"
        )
    folds = raw_folds.astype(np.int64, copy=False)

    if callable(regime):
        x = (
            np.empty((n, periods, 0), dtype=np.float64)
            if covariate_history is None
            else np.asarray(covariate_history, dtype=np.float64)
        )
        if x.ndim != 3 or x.shape[:2] != a.shape or not np.isfinite(x).all():
            raise CausalValueError(
                "dynamic regimes require finite covariate_history shaped (subjects, periods, covariates)"
            )
        actions = np.empty_like(a)
        for i in range(n):
            past: list[bool] = []
            for t in range(periods):
                history = tuple(tuple(float(v) for v in x[i, s]) for s in range(t + 1))
                action = regime(t, tuple(past), history)
                if not isinstance(action, (bool, np.bool_)):
                    raise CausalValueError(
                        "dynamic regime callback must return a bool for every subject-period"
                    )
                actions[i, t] = bool(action)
                past.append(bool(a[i, t]))
    else:
        actions = np.asarray(regime)
        if actions.shape != (periods,) or not np.isin(actions, (0, 1, False, True)).all():
            raise CausalValueError(
                "static regime must contain one binary action per treatment period"
            )
        actions = np.broadcast_to(actions.astype(bool), a.shape).copy()

    censor = (
        np.ones_like(p)
        if censoring_survival is None
        else np.asarray(censoring_survival, dtype=np.float64)
    )
    if censor.shape != q.shape or not np.isfinite(censor).all():
        raise CausalValueError(
            "censoring_survival must be finite and match period_outcome_predictions"
        )
    if np.any(censor < minimum_probability) or np.any(censor > 1.0):
        raise CausalValueError(
            "sequential censoring positivity is violated at the declared probability floor"
        )
    try:
        value, subject_count, min_action_p, min_censor_p = _native.evaluate_sequential_gformula(
            q,
            actions,
            p,
            censor,
            list(ids),
            folds,
            float(minimum_probability),
        )
    except ValueError as error:
        raise CausalValueError(str(error)) from error
    return GFormulaValue(
        float(value),
        int(subject_count),
        float(min_action_p),
        float(min_censor_p),
        tuple(zip(ids, (int(fold) for fold in folds), strict=True)),
    )


def _evaluate_sequential_doubly_robust_direct(
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
        raise CausalValueError("q_predictions must be a non-empty finite (subjects, periods) array")
    n, periods = q.shape
    if y.shape != (n,) or terminal_observed.shape != (n,):
        raise CausalValueError("outcomes and outcome_observed must contain one value per subject")
    if not np.isin(terminal_observed, (0, 1, False, True)).all():
        raise CausalValueError("outcome_observed must contain one bool per subject")
    terminal_observed = terminal_observed.astype(bool, copy=False)
    if observed.shape != (n, periods) or not np.isin(observed, (0, 1, False, True)).all():
        raise CausalValueError("observation_history must be binary and match q_predictions")
    observed = observed.astype(bool, copy=False)
    if np.any((~observed[:, :-1]) & observed[:, 1:]):
        raise CausalValueError("observation_history must be monotone after censoring or dropout")
    if not np.array_equal(terminal_observed, observed[:, -1]):
        raise CausalValueError("outcome_observed must match the final observation-history period")
    if not np.isfinite(y[terminal_observed]).all():
        raise CausalValueError("observed terminal outcomes must be finite")
    y = np.where(terminal_observed, y, 0.0)
    if not terminal_observed.any():
        raise CausalValueError(
            "at least one terminal outcome must be observed for sequential augmentation"
        )
    if a.shape != (n, periods) or not np.isin(a, (0, 1, False, True)).all():
        raise CausalValueError("treatment_history must be binary and match q_predictions")
    a = a.astype(bool, copy=False)
    if p.shape != (n, periods) or not np.isfinite(p).all():
        raise CausalValueError("treatment_probabilities must be finite and match q_predictions")
    if not isfinite(minimum_probability) or not 0 < minimum_probability <= 0.5:
        raise CausalValueError("minimum_probability must be finite and in (0, 0.5]")
    if np.any(p < minimum_probability) or np.any(p > 1.0 - minimum_probability):
        raise CausalValueError(
            "sequential treatment positivity is violated at the declared probability floor"
        )

    ids = tuple(subject_ids)
    if len(ids) != n or any(
        not isinstance(subject_id, str) or not subject_id.strip() for subject_id in ids
    ):
        raise CausalValueError("subject_ids must contain one non-empty string per subject row")
    if len(set(ids)) != n:
        raise CausalValueError(
            "doubly robust inputs require one row per unique subject to preserve fold ownership"
        )

    def integer_folds(values: Sequence[int], name: str) -> np.ndarray:
        raw = np.asarray(values)
        if (
            raw.shape != (n,)
            or not np.issubdtype(raw.dtype, np.signedinteger)
            or np.any(raw < np.iinfo(np.int64).min)
            or np.any(raw > np.iinfo(np.int64).max)
            or np.any(raw < 0)
        ):
            raise CausalValueError(
                f"{name} must contain one non-negative signed 64-bit integer per subject"
            )
        return raw.astype(np.int64, copy=False)

    folds = integer_folds(fold_ids, "fold_ids")
    prediction_folds = integer_folds(prediction_fold_ids, "prediction_fold_ids")
    if not np.array_equal(folds, prediction_folds):
        raise CausalValueError("Q prediction fold ownership must match the subject fold IDs")

    if callable(regime):
        x = (
            np.empty((n, periods, 0), dtype=np.float64)
            if covariate_history is None
            else np.asarray(covariate_history, dtype=np.float64)
        )
        if x.ndim != 3 or x.shape[:2] != a.shape or not np.isfinite(x).all():
            raise CausalValueError(
                "dynamic regimes require finite covariate_history shaped (subjects, periods, covariates)"
            )
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
                    raise CausalValueError(
                        "dynamic regime callback must return a bool for every subject-period"
                    )
                actions[i, t] = bool(action)
                past.append(bool(a[i, t]))
    else:
        actions = np.asarray(regime)
        if actions.shape != (periods,) or not np.isin(actions, (0, 1, False, True)).all():
            raise CausalValueError(
                "static regime must contain one binary action per treatment period"
            )
        actions = np.broadcast_to(actions.astype(bool), a.shape).copy()

    censor = (
        np.ones_like(p)
        if censoring_probabilities is None
        else np.asarray(censoring_probabilities, dtype=np.float64)
    )
    if censor.shape != (n, periods) or not np.isfinite(censor).all():
        raise CausalValueError("censoring_probabilities must be finite and match q_predictions")
    if np.any(censor < minimum_probability) or np.any(censor > 1.0):
        raise CausalValueError(
            "sequential censoring positivity is violated at the declared probability floor"
        )
    try:
        value, subject_count, min_action_p, min_censor_p = (
            _native.evaluate_sequential_doubly_robust(
                y,
                terminal_observed,
                observed,
                a,
                actions,
                q,
                p,
                censor,
                list(ids),
                folds,
                prediction_folds,
                float(minimum_probability),
            )
        )
    except ValueError as error:
        raise CausalValueError(str(error)) from error
    return DoublyRobustRegimeValue(
        float(value),
        int(subject_count),
        float(min_action_p),
        float(min_censor_p),
        tuple(zip(ids, (int(fold) for fold in folds), strict=True)),
    )


@dataclass(frozen=True, slots=True)
class TrajectorySurvivalIntervals:
    """Subject-bootstrap intervals of the weighted product-limit contrast of two regimes.

    The resampling conditions on the supplied treatment probabilities and censoring weights:
    the nuisance models that produced them are not refitted, so their estimation error is not
    included, and the interval is not a posterior.
    """

    survival_difference: tuple[float, float]
    restricted_mean_difference: tuple[float, float]
    replicates_requested: int
    replicates_ok: int


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
    """Deprecated direct path; use an inverse-probability ``LongitudinalRegime`` through ``analyze``."""
    _deprecation.direct_path_deprecated(
        "antecedent.regimes.evaluate_regime_value",
        'antecedent.analyze(data, query=LongitudinalRegime(..., method="ipw"))',
    )
    return _evaluate_regime_value_direct(
        outcomes,
        treatment_history,
        regime,
        treatment_probabilities,
        covariate_history=covariate_history,
        outcome_observed=outcome_observed,
        censoring_survival=censoring_survival,
        minimum_probability=minimum_probability,
    )


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
    """Deprecated direct path; use a g-formula ``LongitudinalRegime`` through ``analyze``."""
    _deprecation.direct_path_deprecated(
        "antecedent.regimes.evaluate_sequential_gformula",
        'antecedent.analyze(data, query=LongitudinalRegime(..., method="g_formula", period_outcome_predictions=...))',
    )
    return _evaluate_sequential_gformula_direct(
        period_outcome_predictions,
        treatment_history,
        regime,
        treatment_probabilities,
        subject_ids=subject_ids,
        fold_ids=fold_ids,
        covariate_history=covariate_history,
        censoring_survival=censoring_survival,
        minimum_probability=minimum_probability,
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
    """Deprecated direct path; use a sequential doubly robust ``LongitudinalRegime`` through ``analyze``."""
    _deprecation.direct_path_deprecated(
        "antecedent.regimes.evaluate_sequential_doubly_robust",
        'antecedent.analyze(data, query=LongitudinalRegime(..., method="sequential_dr", q_predictions=..., observation_history=...))',
    )
    return _evaluate_sequential_doubly_robust_direct(
        outcomes,
        outcome_observed,
        observation_history,
        treatment_history,
        regime,
        q_predictions,
        treatment_probabilities,
        subject_ids=subject_ids,
        fold_ids=fold_ids,
        prediction_fold_ids=prediction_fold_ids,
        covariate_history=covariate_history,
        censoring_probabilities=censoring_probabilities,
        minimum_probability=minimum_probability,
    )


_IndicatorArray: TypeAlias = NDArray[np.bool_] | NDArray[np.float64]


def _trajectory_indicator(values: ArrayLike) -> _IndicatorArray:
    column = np.asarray(values)
    if column.dtype == np.dtype(np.float64):
        return column
    return np.asarray(column, dtype=bool)


@dataclass(frozen=True, slots=True)
class _Trajectories:
    subjects: int
    periods: int
    treatment: _IndicatorArray
    event: _IndicatorArray
    at_risk: _IndicatorArray
    treatment_probability: NDArray[np.float64]
    censoring_weight: NDArray[np.float64]
    minimum_probability: float

    def args(
        self,
    ) -> tuple[
        int,
        int,
        _IndicatorArray,
        _IndicatorArray,
        _IndicatorArray,
        NDArray[np.float64],
        NDArray[np.float64],
        float,
    ]:
        return (
            self.subjects,
            self.periods,
            self.treatment,
            self.event,
            self.at_risk,
            self.treatment_probability,
            self.censoring_weight,
            self.minimum_probability,
        )


def _trajectories(
    treatment: ArrayLike,
    event: ArrayLike,
    at_risk: ArrayLike,
    treatment_probability: ArrayLike,
    censoring_weight: ArrayLike,
    minimum_probability: float,
) -> _Trajectories:
    flags = [_trajectory_indicator(value) for value in (treatment, event, at_risk)]
    numbers = [
        np.asarray(value, dtype=float) for value in (treatment_probability, censoring_weight)
    ]
    shapes = {array.shape for array in (*flags, *numbers)}
    if len(shapes) != 1 or len(next(iter(shapes))) != 2:
        raise CausalValueError(
            "treatment, event, at_risk, treatment_probability and censoring_weight must share one "
            "(subjects, periods) shape"
        )
    subjects, periods = flags[0].shape
    return _Trajectories(
        subjects,
        periods,
        flags[0].ravel(),
        flags[1].ravel(),
        flags[2].ravel(),
        numbers[0].ravel(),
        numbers[1].ravel(),
        float(minimum_probability),
    )


def trajectory_survival_bootstrap(
    treatment: ArrayLike,
    event: ArrayLike,
    at_risk: ArrayLike,
    treatment_probability: ArrayLike,
    censoring_weight: ArrayLike,
    *,
    active: Sequence[bool],
    control: Sequence[bool],
    minimum_probability: float = 0.05,
    replicates: int = 999,
    execution: Execution | None = None,
) -> TrajectorySurvivalIntervals:
    """Percentile subject-bootstrap intervals of the weighted product-limit contrast.

    Arrays are ``(subjects, periods)``; ``active`` and ``control`` are the two prescribed
    treatment trajectories, one action per period. The intervals condition on the supplied
    treatment probabilities and censoring weights. For a credible interval see
    :func:`trajectory_hazard_credible_interval`, which refuses.
    """
    execution = resolve_execution(execution)
    data = _trajectories(
        treatment, event, at_risk, treatment_probability, censoring_weight, minimum_probability
    )
    low, high, rmst_low, rmst_high, requested, ok = _native.trajectory_bootstrap_native(
        *data.args(),
        [bool(a) for a in active],
        [bool(c) for c in control],
        int(replicates),
        int(execution.seed),
    )
    return TrajectorySurvivalIntervals((low, high), (rmst_low, rmst_high), requested, ok)


def trajectory_survival_contrast(
    treatment: ArrayLike,
    event: ArrayLike,
    at_risk: ArrayLike,
    treatment_probability: ArrayLike,
    censoring_weight: ArrayLike,
    *,
    active: Sequence[bool],
    control: Sequence[bool],
    minimum_probability: float = 0.05,
) -> tuple[float, float]:
    """The weighted product-limit ``(survival difference, restricted-mean difference)``."""
    data = _trajectories(
        treatment, event, at_risk, treatment_probability, censoring_weight, minimum_probability
    )
    return _native.trajectory_contrast_native(
        *data.args(), [bool(a) for a in active], [bool(c) for c in control]
    )


def trajectory_hazard_credible_interval(
    treatment: ArrayLike,
    event: ArrayLike,
    at_risk: ArrayLike,
    treatment_probability: ArrayLike,
    censoring_weight: ArrayLike,
    *,
    active: Sequence[bool],
    control: Sequence[bool],
    minimum_probability: float = 0.05,
) -> tuple[float, float]:
    """A credible interval for the trajectory survival contrast: always refused.

    Raises a ``route_not_supported`` refusal (``trajectory_survival.credible_interval_not_supported``)
    naming the missing premise, a joint likelihood for the confounders, treatment, censoring and
    event hazard (a parametric g-formula posterior), and the next attempt: supply a longitudinal
    confounder, treatment, censoring and event model with priors, or report the conditional
    bootstrap interval of :func:`trajectory_survival_bootstrap`.
    """
    data = _trajectories(
        treatment, event, at_risk, treatment_probability, censoring_weight, minimum_probability
    )
    return _native.trajectory_credible_interval_native(
        *data.args(), [bool(a) for a in active], [bool(c) for c in control]
    )


__all__ = [
    "LongitudinalCalibrationDesign",
    "FittedRegimeQ",
    "FixedScheduleContrasts",
    "LearnedScheduleClass",
    "LongitudinalScheduleResult",
    "ScheduleContrast",
    "ScheduleLearningResult",
    "TrajectorySurvivalIntervals",
    "trajectory_hazard_credible_interval",
    "trajectory_survival_bootstrap",
    "trajectory_survival_contrast",
    "SequentialContrast",
    "SequentialContrastEstimate",
    "LongitudinalRegime",
    "LongitudinalRegimeEstimate",
    "GFormulaValue",
    "DoublyRobustRegimeValue",
    "HistoryPolicy",
    "MarginalStructuralModelResult",
    "Regime",
    "RegimeValue",
    "evaluate_regime_value",
    "evaluate_sequential_doubly_robust",
    "evaluate_sequential_gformula",
]
