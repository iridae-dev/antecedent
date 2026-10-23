"""Point estimation utilities for explicitly scoped quasi-experimental designs.

These utilities do not add a support-matrix license or interval guarantee.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Literal

import numpy as np

from ._data import as_columns
from ._native import difference_in_differences as _difference_in_differences
from ._native import group_time_att as _group_time_att
from .errors import CausalUnsupportedError, CausalValueError


def _binary(values: Iterable[Any], name: str) -> list[bool]:
    encoded: list[bool] = []
    for value in values:
        if (
            isinstance(value, (bool, np.bool_))
            or isinstance(value, (int, float, np.integer, np.floating))
            and value in (0, 1)
        ):
            encoded.append(bool(value))
        else:
            raise CausalValueError(f"{name} values must be bool or encoded as 0/1")
    return encoded


def _raw_columns(data: Any) -> tuple[list[str], list[np.ndarray]]:
    """Read aligned columns without coercing subject identifiers to numbers."""
    if isinstance(data, Mapping):
        names = [str(name) for name in data]
        columns = [np.asarray(data[name]) for name in data]
    elif hasattr(data, "columns") and hasattr(data, "__getitem__"):
        names = [str(name) for name in data.columns]
        columns = [np.asarray(data[name]) for name in data.columns]
    else:
        names, numeric = as_columns(data)
        columns = [np.asarray(column) for column in numeric]
    if len(set(names)) != len(names):
        raise CausalValueError("quasi-experimental input column names must be unique")
    if any(column.ndim != 1 for column in columns):
        raise CausalValueError("quasi-experimental inputs require one-dimensional columns")
    if columns and len({len(column) for column in columns}) != 1:
        raise CausalValueError("quasi-experimental input columns must have equal row counts")
    return names, columns


@dataclass(frozen=True, slots=True)
class DifferenceInDifferences:
    """A repeated-cross-section 2x2 DiD query with declared design columns.

    This point-only query assumes parallel untreated trends, no anticipation,
    stable treatment assignment, and no interference. The function cannot
    verify those assumptions from the observed table.
    """

    outcome: str
    treated: str
    post: str

    def __post_init__(self) -> None:
        for name in ("outcome", "treated", "post"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise CausalValueError(f"{name} must be a non-empty column name")
        if self.treated == self.post or self.outcome in (self.treated, self.post):
            raise CausalValueError("outcome, treated, and post must name distinct columns")


@dataclass(frozen=True, slots=True)
class DifferenceInDifferencesEstimate:
    """Point-only repeated-cross-section 2x2 DiD result."""

    estimate: float
    uncertainty: str = "point_only"
    design: str = "repeated_cross_section_2x2"
    assumptions: tuple[str, ...] = (
        "parallel_untreated_trends",
        "no_anticipation",
        "stable_treatment_assignment",
        "no_interference",
    )
    support_status: str = "unlicensed_point_utility"


@dataclass(frozen=True, slots=True)
class PanelDifferenceInDifferences:
    """A two-period panel or repeated-cross-section DiD design.

    Subject IDs must identify the same unit at pre and post. Treatment is
    stable within subject. Repeated cross sections sample each subject once.
    Estimation reports a pointwise cluster-robust
    standard error, using subject IDs by default or an optional higher-level
    cluster column. Clusters must belong to one treatment group; a shared
    cluster across groups is refused. With two periods, parallel untreated
    trends is an identifying assumption and cannot be tested from pre-trends.
    The design does not support staggered adoption or missing waves.

    ``weights="w"`` names a column of sampling weights of the balanced panel (both rows of a
    subject carry the subject's weight). The estimate is the difference of the two groups'
    weighted mean change scores in the weighted population; the standard error is the cluster
    score sandwich over sampling-weighted scores, treating clusters as the independent primary
    sampling units. Weights are normalized to mean one (the estimate does not depend on their
    scale), subjects of weight zero are excluded and counted, the Kish effective sample size is
    reported, and no interval is published.
    """

    outcome: str
    subject: str
    treated: str
    post: str
    cluster: str | None = None
    sampling: Literal["balanced_panel", "repeated_cross_section"] = "balanced_panel"
    weights: str | None = None

    @classmethod
    def repeated_cross_section(
        cls,
        outcome: str,
        subject: str,
        treated: str,
        post: str,
        *,
        cluster: str | None = None,
    ) -> PanelDifferenceInDifferences:
        """Bind one sampled subject per row across two periods."""
        return cls(outcome, subject, treated, post, cluster, "repeated_cross_section")

    def __post_init__(self) -> None:
        fields = (self.outcome, self.subject, self.treated, self.post)
        if any(not isinstance(value, str) or not value.strip() for value in fields):
            raise CausalValueError(
                "outcome, subject, treated, and post must be non-empty column names"
            )
        if len(set(fields)) != len(fields):
            raise CausalValueError("outcome, subject, treated, and post must name distinct columns")
        if self.cluster is not None and (
            not isinstance(self.cluster, str) or not self.cluster.strip() or self.cluster in fields
        ):
            raise CausalValueError("cluster must be a distinct non-empty column name")
        if self.sampling not in ("balanced_panel", "repeated_cross_section"):
            raise CausalValueError("sampling must be balanced_panel or repeated_cross_section")
        if self.weights is not None:
            if (
                not isinstance(self.weights, str)
                or not self.weights.strip()
                or self.weights in fields
                or self.weights == self.cluster
            ):
                raise CausalValueError("weights must be a distinct non-empty column name")
            if self.sampling != "balanced_panel":
                raise CausalUnsupportedError(
                    "weights= are read by the balanced two-period panel DiD; a repeated "
                    "cross-section has no sampling-weighted contrast",
                    reason_code="option_not_applicable",
                )


@dataclass(frozen=True, slots=True)
class PanelDifferenceInDifferencesEstimate:
    """DiD estimate with a support-gated pointwise cluster interval."""

    estimate: float
    standard_error: float
    treated_subjects: int
    control_subjects: int
    clusters: int
    uncertainty: str = "cluster_robust_se_only_pointwise_cr1_unlicensed"
    interval_95: tuple[float, float] | None = None
    design: str = "balanced_two_period_panel"
    assumptions: tuple[str, ...] = (
        "parallel_untreated_trends",
        "no_anticipation",
        "stable_treatment_assignment_within_subject",
        "complete_pre_post_panel",
        "no_interference",
        "independent_sampling_clusters",
    )
    support_status: str = "unlicensed_point_utility"
    cohort: int | None = None
    period: int | None = None


@dataclass(frozen=True, slots=True)
class StaggeredAdoption:
    """Balanced-panel staggered-adoption design using cohort 0 as never treated."""

    outcome: str
    subject: str
    period: str
    cohort: str
    target_cohort: int | None = None
    target_period: int | None = None
    cluster: str | None = None
    event_study: bool = False

    def __post_init__(self) -> None:
        fields = (self.outcome, self.subject, self.period, self.cohort)
        if any(not isinstance(value, str) or not value.strip() for value in fields):
            raise CausalValueError(
                "outcome, subject, period, and cohort must be non-empty column names"
            )
        if len(set(fields)) != len(fields):
            raise CausalValueError(
                "outcome, subject, period, and cohort must name distinct columns"
            )
        if self.event_study and (self.target_cohort is not None or self.target_period is not None):
            raise CausalValueError(
                "event study compares all cohorts and cannot select a single cohort-period target"
            )
        if (self.target_cohort is None) != (self.target_period is None):
            raise CausalValueError("target_cohort and target_period must be supplied together")
        if self.target_cohort is not None and (
            isinstance(self.target_cohort, bool)
            or not isinstance(self.target_cohort, int)
            or isinstance(self.target_period, bool)
            or not isinstance(self.target_period, int)
            or self.target_cohort <= 1
            or self.target_period < self.target_cohort
        ):
            raise CausalValueError(
                "selected staggered comparison needs cohort > 1 and period >= cohort"
            )
        if self.cluster is not None and (
            not isinstance(self.cluster, str) or not self.cluster.strip() or self.cluster in fields
        ):
            raise CausalValueError("cluster must be a distinct non-empty column name")


@dataclass(frozen=True, slots=True)
class GroupTimeATT:
    cohort: int
    period: int
    event_time: int
    estimate: float
    standard_error: float
    treated_subjects: int
    control_subjects: int
    clusters: int


@dataclass(frozen=True, slots=True)
class StaggeredAdoptionEstimate:
    effects: tuple[GroupTimeATT, ...]
    control_group: str = "never_treated_cohort_0"
    uncertainty: str = "cluster_robust_se_only_pointwise_cr1_unlicensed"
    design: str = "balanced_staggered_adoption_group_time_att"
    assumptions: tuple[str, ...] = (
        "cohort_specific_parallel_untreated_trends",
        "no_anticipation",
        "absorbing_treatment_after_adoption",
        "never_treated_controls_are_valid",
        "no_interference",
        "balanced_panel",
        "independent_sampling_clusters",
    )
    support_status: str = "unlicensed_point_utility"
    diagnostics: tuple[str, ...] = (
        "comparison_support_counts_reported_per_cohort_period",
        "pretrend_not_tested",
        "positivity_not_inferred_from_counts",
        "pointwise_cluster_robust_se_uses_g_over_g_minus_one_correction",
        "no_p_values_or_confidence_intervals_reported",
    )


@dataclass(frozen=True, slots=True)
class StaggeredEventTimeEffect:
    cohort: int
    period: int
    event_time: int
    estimate: float
    standard_error: float
    treated_subjects: int
    control_subjects: int
    clusters: int
    interval_95: tuple[float, float] | None = None


@dataclass(frozen=True, slots=True)
class StaggeredEventStudyEstimate:
    """Cohort event times with supported post-adoption pointwise intervals."""

    effects: tuple[StaggeredEventTimeEffect, ...]
    control_group: str = "never_treated_cohort_0"
    uncertainty: str = "cluster_robust_se_only_pointwise_cr1_unlicensed"
    design: str = "balanced_staggered_adoption_event_study"
    assumptions: tuple[str, ...] = (
        "cohort_specific_parallel_untreated_trends",
        "no_anticipation",
        "absorbing_treatment_after_adoption",
        "never_treated_controls_are_valid",
        "no_interference",
        "balanced_panel",
        "independent_sampling_clusters",
    )
    support_status: str = "unlicensed_point_utility"
    diagnostics: tuple[str, ...] = (
        "cohort_specific_effects_reported_by_event_time",
        "event_time_minus_one_is_reference_and_omitted",
        "pre_adoption_estimates_are_descriptive_diagnostics_not_a_test",
        "comparison_support_counts_reported_per_cohort_event_time",
        "positivity_not_inferred_from_counts",
        "pointwise_cluster_robust_se_uses_g_over_g_minus_one_correction",
        "post_adoption_intervals_are_pointwise_not_simultaneous",
        "pre_adoption_contrasts_have_no_intervals_or_pretrend_test",
    )


@dataclass(frozen=True, slots=True)
class StaggeredRollout:
    """Heterogeneity-robust staggered rollout that never uses already-treated controls.

    ``cohort`` holds each row's first treated period (0 for never treated); the panel may
    be unbalanced for ``callaway_santanna`` and ``imputation``. ``estimator`` is
    ``callaway_santanna`` (group-time 2x2 comparisons), ``sun_abraham`` (interaction-weighted
    cohort-by-event-time contrasts) or ``imputation`` (untreated outcomes imputed from unit and
    period effects fitted on untreated rows). ``comparison`` is ``never_treated`` or
    ``not_yet_treated``; ``nuisance`` says how the baseline ``covariates`` (read at each unit's
    first observed row) adjust a comparison; ``base_period`` is ``varying`` or ``universal``;
    ``aggregation`` (``simple``, ``dynamic``, ``group`` or ``calendar``) defines the scalar
    summary; ``anticipation`` is the number of periods units may respond before adoption.
    ``resampling`` (``cluster_bootstrap`` or ``cluster_multiplier``) refits the whole procedure
    on ``replicates`` reweightings of whole clusters and reports pointwise intervals and
    simultaneous bands. The result is ``result.quasi_rollout``: unlicensed, calibration
    unmeasured.
    """

    outcome: str
    subject: str
    period: str
    cohort: str
    cluster: str | None = None
    estimator: Literal["callaway_santanna", "sun_abraham", "imputation"] = "callaway_santanna"
    comparison: Literal["never_treated", "not_yet_treated"] = "never_treated"
    nuisance: Literal["outcome_regression", "inverse_probability_weighting", "doubly_robust"] = (
        "doubly_robust"
    )
    base_period: Literal["varying", "universal"] = "varying"
    aggregation: Literal["simple", "dynamic", "group", "calendar"] = "simple"
    anticipation: int = 0
    covariates: Sequence[str] = ()
    resampling: Literal["cluster_bootstrap", "cluster_multiplier"] | None = None
    replicates: int = 999

    def __post_init__(self) -> None:
        fields = (self.outcome, self.subject, self.period, self.cohort)
        if any(not isinstance(value, str) or not value.strip() for value in fields):
            raise CausalValueError(
                "outcome, subject, period, and cohort must be non-empty column names"
            )
        if len(set(fields)) != len(fields):
            raise CausalValueError(
                "outcome, subject, period, and cohort must name distinct columns"
            )
        if self.cluster is not None and (
            not isinstance(self.cluster, str) or not self.cluster.strip() or self.cluster in fields
        ):
            raise CausalValueError("cluster must be a distinct non-empty column name")
        _check_choice(self.estimator, "estimator", _ROLLOUT_ESTIMATORS)
        _check_choice(self.comparison, "comparison", _DID_COMPARISONS)
        _check_choice(self.nuisance, "nuisance", _DID_NUISANCE)
        _check_choice(self.base_period, "base_period", _DID_BASE_PERIODS)
        _check_choice(self.aggregation, "aggregation", _DID_AGGREGATIONS)
        if (
            isinstance(self.anticipation, (bool, np.bool_))
            or not isinstance(self.anticipation, (int, np.integer))
            or self.anticipation < 0
        ):
            raise CausalValueError("anticipation must be a non-negative integer")
        taken = (*fields, *((self.cluster,) if self.cluster is not None else ()))
        object.__setattr__(self, "covariates", _column_tuple(self.covariates, "covariates", taken))
        _check_resampling(self.resampling, self.replicates)


@dataclass(frozen=True, slots=True)
class CovariateAdjustedDiD:
    """Two-period panel DiD with parallel untreated trends conditional on baseline covariates.

    Columns follow :class:`PanelDifferenceInDifferences` (a balanced two-period panel of
    ``subject`` rows); ``covariates`` name the baseline covariates, read at each subject's
    first row. ``nuisance`` is ``outcome_regression``, ``inverse_probability_weighting`` or
    ``doubly_robust``. ``resampling`` refits the whole procedure on whole-cluster
    reweightings. The result is ``result.quasi_rollout``: unlicensed, calibration unmeasured.
    """

    outcome: str
    subject: str
    treated: str
    post: str
    covariates: Sequence[str]
    cluster: str | None = None
    nuisance: Literal["outcome_regression", "inverse_probability_weighting", "doubly_robust"] = (
        "doubly_robust"
    )
    resampling: Literal["cluster_bootstrap", "cluster_multiplier"] | None = None
    replicates: int = 999

    def __post_init__(self) -> None:
        fields = (self.outcome, self.subject, self.treated, self.post)
        if any(not isinstance(value, str) or not value.strip() for value in fields):
            raise CausalValueError(
                "outcome, subject, treated, and post must be non-empty column names"
            )
        if len(set(fields)) != len(fields):
            raise CausalValueError("outcome, subject, treated, and post must name distinct columns")
        if self.cluster is not None and (
            not isinstance(self.cluster, str) or not self.cluster.strip() or self.cluster in fields
        ):
            raise CausalValueError("cluster must be a distinct non-empty column name")
        _check_choice(self.nuisance, "nuisance", _DID_NUISANCE)
        taken = (*fields, *((self.cluster,) if self.cluster is not None else ()))
        covariates = _column_tuple(self.covariates, "covariates", taken)
        if not covariates:
            raise CausalValueError(
                "CovariateAdjustedDiD needs at least one covariate; use "
                "PanelDifferenceInDifferences for an unadjusted two-period panel"
            )
        object.__setattr__(self, "covariates", covariates)
        _check_resampling(self.resampling, self.replicates)


@dataclass(frozen=True, slots=True)
class DesignEffect:
    """One reported effect of a rollout, synthetic-panel or local-ratio design.

    ``kind`` is ``overall`` (the scalar summary), ``group_time``, ``event_time``, ``group``,
    ``calendar``, ``pre_trend`` (a falsification contrast), ``period`` or ``unit``.
    ``interval`` is pointwise 95%; ``band`` is the simultaneous 95% band over the effects of
    the same kind. Neither is calibrated.
    """

    kind: str
    estimate: float
    standard_error: float | None = None
    interval: tuple[float, float] | None = None
    band: tuple[float, float] | None = None
    group: int | None = None
    period: int | None = None
    event_time: int | None = None
    unit: str | None = None
    treated: int = 0
    comparison: int = 0


@dataclass(frozen=True, slots=True)
class QuasiRolloutEstimate:
    """Heterogeneity-robust or covariate-adjusted DiD, conformal or multi-unit synthetic panel,
    or optioned local ratio. The first effect is the scalar summary. Never licensed; the
    calibration of every interval is unmeasured."""

    design: str
    method: str
    effects: tuple[DesignEffect, ...]
    uncertainty: str
    replicates_requested: int = 0
    replicates_ok: int = 0
    replicates_failed: int = 0
    critical_value: float | None = None
    diagnostics: tuple[tuple[str, float], ...] = ()
    calibration: str = "unmeasured"
    licensed: bool = False

    @property
    def overall(self) -> DesignEffect | None:
        """The scalar summary."""
        return self.effects[0] if self.effects else None

    def effects_of(self, kind: str) -> tuple[DesignEffect, ...]:
        """The reported effects of one ``kind``."""
        return tuple(effect for effect in self.effects if effect.kind == kind)

    def diagnostic(self, name: str) -> float | None:
        """The named numeric diagnostic, when present."""
        for key, value in self.diagnostics:
            if key == name:
                return value
        return None


@dataclass(frozen=True, slots=True)
class SyntheticControl:
    """Balanced-panel synthetic control with one treated unit and donor pool."""

    outcome: str
    unit: str
    period: str
    treated_unit: str
    intervention_period: int
    uniform_unit_randomization: bool = False
    sharp_null_effect: float | None = None
    augmentation_ridge: float | None = None
    #: Conformal intervals for the single treated unit (unlicensed, calibration unmeasured).
    conformal: bool = False
    #: Staggered adoptions ``(unit, adoption period)`` fitted against never-treated donors,
    #: including ``(treated_unit, intervention_period)``; placebo-variance intervals.
    adoptions: Sequence[tuple[str, int]] | None = None
    placebo_replicates: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.conformal, bool):
            raise CausalValueError("conformal must be boolean")
        if self.conformal and self.adoptions is not None:
            raise CausalValueError(
                "conformal intervals cover one treated unit; staggered adoptions use "
                "placebo-variance intervals, so declare one or the other"
            )
        _bind_adoptions(self)
        if not isinstance(self.uniform_unit_randomization, bool):
            raise CausalValueError("uniform_unit_randomization must be boolean")
        if self.sharp_null_effect is not None and (
            not self.uniform_unit_randomization
            or isinstance(self.sharp_null_effect, (bool, np.bool_))
            or not isinstance(self.sharp_null_effect, (int, float, np.integer, np.floating))
            or not np.isfinite(self.sharp_null_effect)
        ):
            raise CausalValueError("finite sharp_null_effect requires uniform_unit_randomization")
        if self.augmentation_ridge is not None and (
            not isinstance(self.augmentation_ridge, (int, float, np.integer, np.floating))
            or isinstance(self.augmentation_ridge, (bool, np.bool_))
            or not np.isfinite(self.augmentation_ridge)
            or self.augmentation_ridge <= 0
        ):
            raise CausalValueError("augmentation_ridge must be finite and positive")
        fields = (self.outcome, self.unit, self.period)
        if any(not isinstance(value, str) or not value.strip() for value in fields):
            raise CausalValueError("outcome, unit, and period must be non-empty column names")
        if len(set(fields)) != len(fields):
            raise CausalValueError("outcome, unit, and period must name distinct columns")
        if not isinstance(self.treated_unit, str) or not self.treated_unit:
            raise CausalValueError("treated_unit must be a non-empty string unit ID")
        if (
            isinstance(self.intervention_period, (bool, np.bool_))
            or not isinstance(self.intervention_period, (int, np.integer))
            or self.intervention_period <= 0
        ):
            raise CausalValueError("intervention_period must be a positive integer")


@dataclass(frozen=True, slots=True)
class SyntheticControlEstimate:
    estimate: float
    pre_treatment_rmse: float
    donor_weights: tuple[tuple[str, float], ...]
    placebo_effects: tuple[float, ...]
    placebo_rank_p_value: float
    effective_donors: float
    n_donors: int
    n_pre_periods: int
    n_post_periods: int
    randomization_p_value: float | None = None
    randomization_null_effect: float | None = None
    randomization_statistics: tuple[tuple[str, float], ...] = ()
    unadjusted_effect: float | None = None
    outcome_model_correction: float | None = None
    augmentation_ridge: float | None = None
    uncertainty: str = "point_only_with_unlicensed_placebo_rank"
    support_status: str = "unlicensed_point_utility"
    diagnostics: tuple[str, ...] = (
        "placebo_rank_assumes_exchangeable_donors_and_is_not_calibrated",
    )
    design: str = "balanced_panel_synthetic_control"
    assumptions: tuple[str, ...] = (
        "no_anticipation",
        "stable_treatment_after_intervention",
        "convex_donor_combination_is_a_valid_counterfactual",
        "no_interference_between_units",
        "no_concurrent_treated_unit_specific_shock",
    )


@dataclass(frozen=True, slots=True)
class SyntheticDifferenceInDifferences:
    """Balanced-panel SDID with optional exact uniform-unit assignment test."""

    outcome: str
    unit: str
    period: str
    treated_unit: str
    intervention_period: int
    uniform_unit_randomization: bool = False
    sharp_null_effect: float | None = None
    #: Staggered adoptions ``(unit, adoption period)``, including
    #: ``(treated_unit, intervention_period)``; placebo-variance intervals.
    adoptions: Sequence[tuple[str, int]] | None = None
    placebo_replicates: int = 0

    def __post_init__(self) -> None:
        _bind_adoptions(self)
        if not isinstance(self.uniform_unit_randomization, bool):
            raise CausalValueError("uniform_unit_randomization must be boolean")
        if self.sharp_null_effect is not None and (
            not self.uniform_unit_randomization
            or isinstance(self.sharp_null_effect, (bool, np.bool_))
            or not isinstance(self.sharp_null_effect, (int, float, np.integer, np.floating))
            or not np.isfinite(self.sharp_null_effect)
        ):
            raise CausalValueError("finite sharp_null_effect requires uniform_unit_randomization")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (self.outcome, self.unit, self.period)
        ):
            raise CausalValueError("outcome, unit, and period must be non-empty column names")
        if len({self.outcome, self.unit, self.period}) != 3:
            raise CausalValueError("outcome, unit, and period must name distinct columns")
        if not isinstance(self.treated_unit, str) or not self.treated_unit:
            raise CausalValueError("treated_unit must be a non-empty string unit ID")
        if (
            isinstance(self.intervention_period, (bool, np.bool_))
            or not isinstance(self.intervention_period, (int, np.integer))
            or self.intervention_period <= 0
        ):
            raise CausalValueError("intervention_period must be a positive integer")


@dataclass(frozen=True, slots=True)
class SyntheticDifferenceInDifferencesEstimate:
    estimate: float
    pre_treatment_rmse: float
    donor_weights: tuple[tuple[str, float], ...]
    time_weights: tuple[tuple[int, float], ...]
    n_donors: int
    n_pre_periods: int
    n_post_periods: int
    randomization_p_value: float | None = None
    randomization_null_effect: float | None = None
    randomization_statistics: tuple[tuple[str, float], ...] = ()
    uncertainty: str = "point_only"
    design: str = "balanced_panel_synthetic_difference_in_differences"
    assumptions: tuple[str, ...] = (
        "no_anticipation",
        "stable_treatment_after_intervention",
        "convex_unit_and_time_weights_represent_untreated_counterfactual_trends",
        "no_interference_between_units",
        "no_concurrent_treated_unit_specific_shock",
    )
    support_status: str = "unlicensed_point_utility"
    diagnostics: tuple[str, ...] = (
        "balanced_panel_and_two_pre_periods_required",
        "unit_and_time_weights_are_simplex_constrained",
        "pre_fit_rmse_reported_without_acceptance_threshold",
        "no_interval_or_calibrated_inference",
    )


@dataclass(frozen=True, slots=True)
class AugmentedPanelDiD:
    """Panel ATT using supplied propensity and untreated-change predictions."""

    outcome_pre: str
    outcome_post: str
    subject: str
    treated: str
    propensity: str
    untreated_change_prediction: str
    predictions_cross_fitted: bool = False
    cluster: str | None = None
    #: Whole-cluster resample of the procedure (``cluster_bootstrap`` or
    #: ``cluster_multiplier``); the supplied nuisance columns are held fixed.
    resampling: Literal["cluster_bootstrap", "cluster_multiplier"] | None = None
    replicates: int = 999

    def __post_init__(self) -> None:
        _check_resampling(self.resampling, self.replicates)
        names = (
            self.outcome_pre,
            self.outcome_post,
            self.subject,
            self.treated,
            self.propensity,
            self.untreated_change_prediction,
        )
        if any(not isinstance(value, str) or not value.strip() for value in names):
            raise CausalValueError("all augmented panel DiD column names must be non-empty")
        if len(set(names)) != len(names):
            raise CausalValueError("augmented panel DiD columns must be distinct")
        if not isinstance(self.predictions_cross_fitted, bool):
            raise CausalValueError("predictions_cross_fitted must be boolean")
        if self.cluster is not None and (
            not isinstance(self.cluster, str) or not self.cluster.strip() or self.cluster in names
        ):
            raise CausalValueError("cluster must be a distinct non-empty column name")


@dataclass(frozen=True, slots=True)
class AugmentedPanelDiDEstimate:
    estimate: float
    treated_subjects: int
    control_subjects: int
    propensity_min: float
    propensity_max: float
    effective_control_sample_size: float
    nuisance_predictions_cross_fitted: bool
    clusters: int = 0
    uncertainty: str = "point_only"
    design: str = "augmented_panel_difference_in_differences"
    assumptions: tuple[str, ...] = (
        "conditional_parallel_untreated_trends_or_correct_untreated_change_model",
        "correct_propensity_model_or_correct_untreated_change_model",
        "no_anticipation",
        "no_interference",
        "strict_propensity_overlap",
        "supplied_nuisance_predictions_are_valid_for_the_evaluation_rows",
    )
    support_status: str = "unlicensed_point_utility"
    diagnostics: tuple[str, ...] = (
        "propensity_range_and_weighted_control_effective_sample_size_reported",
        "cross_fitting_is_caller_declared_not_verified",
        "no_interval_or_calibrated_inference",
    )
    #: Whole-cluster resample interval when ``resampling`` was declared (calibration
    #: unmeasured, nuisance predictions held fixed).
    interval_95: tuple[float, float] | None = None


@dataclass(frozen=True, slots=True)
class FuzzyRegressionDiscontinuity:
    """Fuzzy RD at a declared cutoff.

    ``bandwidth_rule="mse_optimal"`` selects the MSE-optimal plug-in window, capped at
    ``bandwidth``; ``temporal_replicates`` treats the running variable as calendar time and
    reports a block-bootstrap interval; ``density_test`` runs the manipulation test of the
    running variable at the cutoff; ``balance_covariates`` report their jumps at the cutoff.
    Any of these returns ``result.quasi_rollout`` (unlicensed, calibration unmeasured).
    """

    outcome: str
    treatment: str
    running: str
    cutoff: float
    bandwidth: float
    bandwidth_rule: Literal["declared", "mse_optimal"] = "declared"
    temporal_replicates: int | None = None
    density_test: bool = False
    balance_covariates: Sequence[str] = ()

    def __post_init__(self) -> None:
        _validate_rd_spec(self.outcome, self.treatment, self.running, self.cutoff, self.bandwidth)
        _bind_local_ratio_options(self)


@dataclass(frozen=True, slots=True)
class SharpRegressionDiscontinuity:
    """Sharp regression-discontinuity: treatment is a deterministic threshold rule.

    The design (the running variable is the treatment's only cause, and both act on
    the outcome) is fixed, so ``analyze`` needs neither a graph nor estimator/kwargs::

        ant.analyze(data, query=ant.quasi.SharpRegressionDiscontinuity(
            "outcome", "treatment", "running", cutoff=0.0, bandwidth=1.5))

    The effect is the local jump at the cutoff, reported as a
    ``local_at_cutoff`` target, not a population average. A treatment column that is
    not the threshold rule is refused rather than estimated as an ITT contrast (use
    :class:`FuzzyRegressionDiscontinuity` for imperfect compliance).

    ``se`` selects the analytic standard error of the jump coefficient: ``None``
    keeps the ``hc1`` residual-sandwich default; ``hc0`` / ``hc2`` / ``hc3`` are the
    other heteroskedasticity-robust variants and ``homoskedastic`` opts into the
    classical constant-variance formula.
    """

    _SE_KINDS = ("hc1", "homoskedastic", "hc0", "hc2", "hc3")

    outcome: str
    treatment: str
    running: str
    cutoff: float
    bandwidth: float
    se: str | None = None

    def __post_init__(self) -> None:
        _validate_rd_spec(self.outcome, self.treatment, self.running, self.cutoff, self.bandwidth)
        if self.se is not None and self.se not in self._SE_KINDS:
            raise CausalValueError(
                f"SharpRegressionDiscontinuity se must be one of {', '.join(self._SE_KINDS)}; "
                f"got {self.se!r}"
            )


@dataclass(frozen=True, slots=True)
class RegressionKink:
    """Regression kink at a declared cutoff; the options are those of
    :class:`FuzzyRegressionDiscontinuity`."""

    outcome: str
    treatment: str
    running: str
    cutoff: float
    bandwidth: float
    bandwidth_rule: Literal["declared", "mse_optimal"] = "declared"
    temporal_replicates: int | None = None
    density_test: bool = False
    balance_covariates: Sequence[str] = ()

    def __post_init__(self) -> None:
        _validate_rd_spec(self.outcome, self.treatment, self.running, self.cutoff, self.bandwidth)
        _bind_local_ratio_options(self)


@dataclass(frozen=True, slots=True)
class PeriodHistoryMatch:
    """Nearest-neighbor contrast on pre-outcome histories, matched inside calendar periods.

    ``history`` names the time-ordered history columns; their row-wise mean is the match score.
    A treated unit is matched only to a control of its own ``period`` (one period when absent), so
    a match never crosses a period. ``weight`` is a sampling-weight column and ``caliper`` a
    maximum absolute score distance. The estimand is the weighted effect on the matched treated
    units given the period and the history-mean score. The standard error is Abadie-Imbens for one
    match with replacement; no interval is reported and the route is unlicensed. A posterior of a
    matching contrast does not exist (a matching rule has no likelihood): ``Bayesian`` inference
    raises ``CausalUnsupportedError`` naming that premise and the next attempt.
    """

    outcome: str
    treated: str
    history: Sequence[str]
    period: str | None = None
    weight: str | None = None
    caliper: float | None = None
    kind: Literal["period_history_match"] = field(
        default="period_history_match", init=False, repr=False
    )

    def __post_init__(self) -> None:
        names = [self.outcome, self.treated, *self.history]
        names.extend(name for name in (self.period, self.weight) if name is not None)
        if not self.history or any(not isinstance(name, str) or not name.strip() for name in names):
            raise CausalValueError("declare non-empty outcome, treated and history column names")
        if len(set(names)) != len(names):
            raise CausalValueError("outcome, treated, history, period and weight must be distinct")
        if self.caliper is not None and not (isfinite(self.caliper) and self.caliper >= 0.0):
            raise CausalValueError("caliper must be finite and non-negative")
        object.__setattr__(self, "history", tuple(self.history))


@dataclass(frozen=True, slots=True)
class PeriodHistoryMatchEstimate:
    """History-matched contrast with its matched pairs. Unlicensed, no interval."""

    contrast: float
    weight_total: float
    matched_queries: tuple[int, ...]
    matched_donors: tuple[int, ...]
    differences: tuple[float, ...]
    pair_weights: tuple[float, ...]
    unmatched_treated: tuple[int, ...]
    periods_matched: int
    uncertainty: str
    #: Abadie-Imbens standard error; ``None`` with fewer than two matched units.
    standard_error: float | None = None
    caliper: float | None = None
    licensed: bool = False
    calibration_measured: bool = False
    support_status: str = "unlicensed_uncalibrated"


@dataclass(frozen=True, slots=True)
class LocalPolynomialRatioEstimate:
    estimate: float
    reduced_form_discontinuity: float
    first_stage_discontinuity: float
    observations_left: int
    observations_right: int
    standard_error: float
    ci_lower: float | None
    ci_upper: float | None
    reduced_form_standard_error: float | None
    first_stage_standard_error: float | None
    cutoff: float | None = None
    bandwidth: float | None = None
    kink: bool | None = None
    uncertainty: str = "rbc_hc0_delta_normal_fixed_bandwidth"
    design: str = "fuzzy_regression_discontinuity_local_quadratic"
    assumptions: tuple[str, ...] = (
        "potential_outcome_regression_is_smooth_at_cutoff",
        "no_precise_manipulation_of_running_variable",
        "exclusion_restriction_for_threshold_instrument",
        "monotonicity_for_local_complier_interpretation",
        "independent_observations_within_bandwidth",
        "no_interference",
    )
    support_status: str = "off_axis_interval_evidence"
    diagnostics: tuple[str, ...] = (
        "local_quadratic_triangular_kernel",
        "left_right_window_counts_reported",
        "cubic_pilot_bias_correction_at_same_bandwidth",
        "hc0_sandwich_covariance_with_delta_method_ratio_se",
        "nominal_95_interval_calibrated_on_strong_first_stage_fixtures",
    )


def estimate_did(
    data: Any,
    query: DifferenceInDifferences,
    *,
    treated: Sequence[bool] | None = None,
    post: Sequence[bool] | None = None,
) -> DifferenceInDifferencesEstimate:
    """Estimate a 2x2 repeated-cross-section DiD using native Rust execution.

    Optional explicit ``treated`` and ``post`` vectors override the named
    columns. Panel, staggered-adoption, weighted, and clustered-inference
    designs are outside this utility's contract.
    """

    if not isinstance(query, DifferenceInDifferences):
        raise CausalValueError("query must be a DifferenceInDifferences")
    names, columns = _raw_columns(data)
    for name in (query.outcome, query.treated, query.post):
        if name not in names:
            raise CausalValueError(f"required difference-in-differences column {name!r} is missing")
    outcome = np.asarray(columns[names.index(query.outcome)], dtype=np.float64)
    group = list(columns[names.index(query.treated)]) if treated is None else list(treated)
    period = list(columns[names.index(query.post)]) if post is None else list(post)
    if len(group) != len(outcome) or len(period) != len(outcome):
        raise CausalValueError("treated and post must match the outcome row count")
    group_values = _binary(group, "treated")
    period_values = _binary(period, "post")
    try:
        value = _difference_in_differences(outcome, group_values, period_values)
    except ValueError as error:
        raise CausalValueError(str(error)) from error
    return DifferenceInDifferencesEstimate(float(value))


def _integer_column(values: Iterable[Any], name: str, *, minimum: int) -> list[int]:
    encoded: list[int] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise CausalValueError(f"{name} values must be integers")
        if not np.isfinite(value):
            raise CausalValueError(f"{name} values must be finite integers")
        integer = int(value)
        if value != integer:
            raise CausalValueError(f"{name} values must be integers")
        if integer > 2**63 - 1:
            raise CausalValueError(f"{name} values exceed the supported integer range")
        if integer < minimum:
            raise CausalValueError(f"{name} values must be at least {minimum}")
        encoded.append(integer)
    return encoded


def estimate_group_time_att(
    data: Any, query: StaggeredAdoption, *, cluster: str | None = None
) -> StaggeredAdoptionEstimate:
    """Estimate cohort-by-period ATT using never-treated controls and cohort-specific baselines.

    For adoption cohort ``g`` and each observed ``t >= g``, the kernel
    differences the cohort's mean change since ``g - 1`` from the never-treated
    mean change over the same periods. Periods are positive integers and cohort
    0 marks never treated. Balanced support is required; the returned counts
    describe sampled units, not propensity or inferential support. It also
    reports pointwise cluster-robust standard errors using subject IDs by
    default, or an optional higher-level cluster column.
    """

    if not isinstance(query, StaggeredAdoption):
        raise CausalValueError("query must be a StaggeredAdoption")
    if cluster is None:
        cluster = query.cluster
    names, columns = _raw_columns(data)
    for name in (query.outcome, query.subject, query.period, query.cohort):
        if name not in names:
            raise CausalValueError(f"required staggered-adoption column {name!r} is missing")
    if cluster is not None and (not isinstance(cluster, str) or cluster not in names):
        raise CausalValueError("cluster must name a present cluster column")
    outcome = np.asarray(columns[names.index(query.outcome)], dtype=np.float64)
    subjects = list(columns[names.index(query.subject)])
    if any(not isinstance(subject, str) or not subject for subject in subjects):
        raise CausalValueError("subject IDs must be non-empty strings")
    periods = _integer_column(columns[names.index(query.period)], "period", minimum=1)
    cohorts = _integer_column(columns[names.index(query.cohort)], "cohort", minimum=0)
    clusters = subjects if cluster is None else list(columns[names.index(cluster)])
    if any(not isinstance(value, str) or not value for value in clusters):
        raise CausalValueError("cluster IDs must be non-empty strings")
    try:
        raw_effects = _group_time_att(outcome, subjects, periods, cohorts, clusters)
    except ValueError as error:
        raise CausalValueError(str(error)) from error
    effects = tuple(
        GroupTimeATT(
            cohort=int(cohort),
            period=int(period),
            event_time=int(period - cohort),
            estimate=float(estimate),
            standard_error=float(se),
            treated_subjects=int(n_treated),
            control_subjects=int(n_control),
            clusters=int(n_clusters),
        )
        for cohort, period, estimate, n_treated, n_control, se, n_clusters in raw_effects
    )
    return StaggeredAdoptionEstimate(effects)


def _validate_rd_spec(
    outcome: str, treatment: str, running: str, cutoff: float, bandwidth: float
) -> None:
    fields = (outcome, treatment, running)
    if any(not isinstance(value, str) or not value.strip() for value in fields):
        raise CausalValueError("outcome, treatment, and running must be non-empty column names")
    if len(set(fields)) != len(fields):
        raise CausalValueError("outcome, treatment, and running must name distinct columns")
    if (
        isinstance(cutoff, (bool, np.bool_))
        or not isinstance(cutoff, (int, float, np.integer, np.floating))
        or not np.isfinite(cutoff)
    ):
        raise CausalValueError("cutoff must be finite")
    if (
        isinstance(bandwidth, (bool, np.bool_))
        or not isinstance(bandwidth, (int, float, np.integer, np.floating))
        or not np.isfinite(bandwidth)
        or bandwidth <= 0
    ):
        raise CausalValueError("bandwidth must be finite and positive")


_ROLLOUT_ESTIMATORS = ("callaway_santanna", "sun_abraham", "imputation")
_DID_COMPARISONS = ("never_treated", "not_yet_treated")
_DID_NUISANCE = ("outcome_regression", "inverse_probability_weighting", "doubly_robust")
_DID_BASE_PERIODS = ("varying", "universal")
_DID_AGGREGATIONS = ("simple", "dynamic", "group", "calendar")
_DID_RESAMPLING = ("cluster_bootstrap", "cluster_multiplier")
_LOCAL_BANDWIDTH_RULES = ("declared", "mse_optimal")


def _check_choice(value: Any, name: str, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise CausalValueError(f"{name} must be one of {', '.join(allowed)}; got {value!r}")


def _positive_int(value: Any) -> bool:
    return (
        not isinstance(value, (bool, np.bool_))
        and isinstance(value, (int, np.integer))
        and value > 0
    )


def _check_resampling(resampling: Any, replicates: Any) -> None:
    if resampling is not None:
        _check_choice(resampling, "resampling", _DID_RESAMPLING)
    if not _positive_int(replicates) or replicates > 2**32 - 1:
        raise CausalValueError("replicates must be a positive integer")


def _column_tuple(values: Any, name: str, taken: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(values, str):
        raise CausalValueError(f"{name} must be a sequence of column names, not one string")
    columns = tuple(values)
    if any(not isinstance(value, str) or not value.strip() for value in columns):
        raise CausalValueError(f"{name} must be non-empty column names")
    if len(set(columns)) != len(columns) or set(columns) & set(taken):
        raise CausalValueError(f"{name} must be distinct from each other and the design columns")
    return columns


def _bind_adoptions(query: Any) -> None:
    """Validate and freeze the staggered adoptions of a synthetic panel."""
    replicates = query.placebo_replicates
    if (
        isinstance(replicates, (bool, np.bool_))
        or not isinstance(replicates, (int, np.integer))
        or not 0 <= replicates <= 2**32 - 1
    ):
        raise CausalValueError("placebo_replicates must be a non-negative integer")
    if query.adoptions is None:
        if replicates:
            raise CausalValueError("placebo_replicates requires staggered adoptions")
        return
    if isinstance(query.adoptions, (str, Mapping)):
        raise CausalValueError("adoptions must be a sequence of (unit, adoption period) pairs")
    adoptions: list[tuple[str, int]] = []
    for pair in query.adoptions:
        if not isinstance(pair, Sequence) or isinstance(pair, str) or len(pair) != 2:
            raise CausalValueError("adoptions must be (unit, adoption period) pairs")
        unit, period = pair
        if not isinstance(unit, str) or not unit:
            raise CausalValueError("adoption units must be non-empty strings")
        if (
            isinstance(period, (bool, np.bool_))
            or not isinstance(period, (int, np.integer))
            or period <= 0
        ):
            raise CausalValueError("adoption periods must be positive integers")
        adoptions.append((unit, int(period)))
    if len({unit for unit, _ in adoptions}) != len(adoptions):
        raise CausalValueError("each adopting unit is listed once")
    if (query.treated_unit, int(query.intervention_period)) not in adoptions:
        raise CausalValueError("adoptions must include (treated_unit, intervention_period)")
    object.__setattr__(query, "adoptions", tuple(adoptions))


def _bind_local_ratio_options(query: Any) -> None:
    """Validate and freeze the design options of a fuzzy RD or regression kink."""
    _check_choice(query.bandwidth_rule, "bandwidth_rule", _LOCAL_BANDWIDTH_RULES)
    if query.temporal_replicates is not None and (
        not _positive_int(query.temporal_replicates) or query.temporal_replicates > 2**32 - 1
    ):
        raise CausalValueError("temporal_replicates must be a positive integer")
    if not isinstance(query.density_test, bool):
        raise CausalValueError("density_test must be boolean")
    taken = (query.outcome, query.treatment, query.running)
    object.__setattr__(
        query,
        "balance_covariates",
        _column_tuple(query.balance_covariates, "balance_covariates", taken),
    )


__all__ = [
    "PeriodHistoryMatch",
    "PeriodHistoryMatchEstimate",
    "DifferenceInDifferences",
    "DifferenceInDifferencesEstimate",
    "PanelDifferenceInDifferences",
    "PanelDifferenceInDifferencesEstimate",
    "StaggeredAdoption",
    "GroupTimeATT",
    "StaggeredAdoptionEstimate",
    "StaggeredEventStudyEstimate",
    "StaggeredEventTimeEffect",
    "SyntheticControl",
    "SyntheticControlEstimate",
    "SyntheticDifferenceInDifferences",
    "SyntheticDifferenceInDifferencesEstimate",
    "AugmentedPanelDiD",
    "AugmentedPanelDiDEstimate",
    "StaggeredRollout",
    "CovariateAdjustedDiD",
    "DesignEffect",
    "QuasiRolloutEstimate",
    "FuzzyRegressionDiscontinuity",
    "RegressionKink",
    "SharpRegressionDiscontinuity",
    "LocalPolynomialRatioEstimate",
    "estimate_did",
    "estimate_group_time_att",
]
