"""Point estimation utilities for explicitly scoped quasi-experimental designs.

These utilities do not add a support-matrix license or interval guarantee.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np

from ._data import as_columns
from ._native import augmented_panel_difference_in_differences as _augmented_panel_did
from ._native import difference_in_differences as _difference_in_differences
from ._native import group_time_att as _group_time_att
from ._native import local_polynomial_fuzzy_discontinuity as _local_polynomial_fuzzy_discontinuity
from ._native import panel_difference_in_differences as _panel_difference_in_differences
from ._native import staggered_event_study as _staggered_event_study
from ._native import synthetic_control as _synthetic_control
from ._native import synthetic_difference_in_differences as _synthetic_did
from .errors import CausalValueError


def _binary(values: Sequence[Any], name: str) -> list[bool]:
    encoded: list[bool] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or isinstance(value, (int, float, np.integer, np.floating)) and value in (0, 1):
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
    cluster column. The design does not support staggered adoption or missing
    waves.
    """

    outcome: str
    subject: str
    treated: str
    post: str
    cluster: str | None = None
    sampling: Literal["balanced_panel", "repeated_cross_section"] = "balanced_panel"

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
            raise CausalValueError("outcome, subject, treated, and post must be non-empty column names")
        if len(set(fields)) != len(fields):
            raise CausalValueError("outcome, subject, treated, and post must name distinct columns")
        if self.cluster is not None and (not isinstance(self.cluster, str) or not self.cluster.strip() or self.cluster in fields):
            raise CausalValueError("cluster must be a distinct non-empty column name")
        if self.sampling not in ("balanced_panel", "repeated_cross_section"):
            raise CausalValueError("sampling must be balanced_panel or repeated_cross_section")


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
            raise CausalValueError("outcome, subject, period, and cohort must be non-empty column names")
        if len(set(fields)) != len(fields):
            raise CausalValueError("outcome, subject, period, and cohort must name distinct columns")
        if self.event_study and (self.target_cohort is not None or self.target_period is not None):
            raise CausalValueError("event study compares all cohorts and cannot select a single cohort-period target")
        if (self.target_cohort is None) != (self.target_period is None):
            raise CausalValueError("target_cohort and target_period must be supplied together")
        if self.target_cohort is not None and (
            isinstance(self.target_cohort, bool) or not isinstance(self.target_cohort, int)
            or isinstance(self.target_period, bool) or not isinstance(self.target_period, int)
            or self.target_cohort <= 1 or self.target_period < self.target_cohort
        ):
            raise CausalValueError("selected staggered comparison needs cohort > 1 and period >= cohort")
        if self.cluster is not None and (not isinstance(self.cluster, str) or not self.cluster.strip() or self.cluster in fields):
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
class SyntheticControl:
    """Balanced-panel synthetic control with one treated unit and donor pool."""

    outcome: str
    unit: str
    period: str
    treated_unit: str
    intervention_period: int
    uniform_unit_randomization: bool = False
    augmentation_ridge: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.uniform_unit_randomization, bool):
            raise CausalValueError("uniform_unit_randomization must be boolean")
        if self.augmentation_ridge is not None:
            if (not isinstance(self.augmentation_ridge, (int, float, np.integer, np.floating))
                or isinstance(self.augmentation_ridge, (bool, np.bool_))
                or not np.isfinite(self.augmentation_ridge) or self.augmentation_ridge <= 0):
                raise CausalValueError("augmentation_ridge must be finite and positive")
            if self.uniform_unit_randomization:
                raise CausalValueError("augmentation cannot be combined with exact unit randomization")
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

    def __post_init__(self) -> None:
        if not isinstance(self.uniform_unit_randomization, bool):
            raise CausalValueError("uniform_unit_randomization must be boolean")
        if any(not isinstance(value, str) or not value.strip() for value in (self.outcome, self.unit, self.period)):
            raise CausalValueError("outcome, unit, and period must be non-empty column names")
        if len({self.outcome, self.unit, self.period}) != 3:
            raise CausalValueError("outcome, unit, and period must name distinct columns")
        if not isinstance(self.treated_unit, str) or not self.treated_unit:
            raise CausalValueError("treated_unit must be a non-empty string unit ID")
        if isinstance(self.intervention_period, (bool, np.bool_)) or not isinstance(
            self.intervention_period, (int, np.integer)
        ) or self.intervention_period <= 0:
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

    def __post_init__(self) -> None:
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
        if self.cluster is not None and (not isinstance(self.cluster, str) or not self.cluster.strip() or self.cluster in names):
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


@dataclass(frozen=True, slots=True)
class FuzzyRegressionDiscontinuity:
    outcome: str
    treatment: str
    running: str
    cutoff: float
    bandwidth: float

    def __post_init__(self) -> None:
        _validate_rd_spec(self.outcome, self.treatment, self.running, self.cutoff, self.bandwidth)


@dataclass(frozen=True, slots=True)
class RegressionKink:
    outcome: str
    treatment: str
    running: str
    cutoff: float
    bandwidth: float

    def __post_init__(self) -> None:
        _validate_rd_spec(self.outcome, self.treatment, self.running, self.cutoff, self.bandwidth)


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
        value = _difference_in_differences(
            outcome, group_values, period_values
        )
    except ValueError as error:
        raise CausalValueError(str(error)) from error
    return DifferenceInDifferencesEstimate(float(value))


def estimate_panel_did(
    data: Any, query: PanelDifferenceInDifferences, *, cluster: str | None = None
) -> PanelDifferenceInDifferencesEstimate:
    """Estimate balanced-panel DiD as the treated-control mean difference in subject changes.

    Standard errors use subject clusters by default, or the optional higher-level
    cluster column. They use a CR1-style G/(G-1) multiplier on summed influence
    contributions. Only pointwise SEs are reported; no p-values or intervals.
    """

    if not isinstance(query, PanelDifferenceInDifferences):
        raise CausalValueError("query must be a PanelDifferenceInDifferences")
    if query.sampling == "repeated_cross_section":
        from ._analyze import analyze

        effective = replace(query, cluster=cluster) if cluster is not None else query
        result = analyze(data, query=effective)
        if result.panel_did is None:
            raise CausalValueError("repeated-cross-section DiD did not return a DiD section")
        return result.panel_did
    names, columns = _raw_columns(data)
    for name in (query.outcome, query.subject, query.treated, query.post):
        if name not in names:
            raise CausalValueError(f"required panel difference-in-differences column {name!r} is missing")
    if cluster is not None and (not isinstance(cluster, str) or cluster not in names):
        raise CausalValueError("cluster must name a present cluster column")
    outcome = np.asarray(columns[names.index(query.outcome)], dtype=np.float64)
    subjects = list(columns[names.index(query.subject)])
    if any(not isinstance(subject, str) or not subject for subject in subjects):
        raise CausalValueError("subject IDs must be non-empty strings")
    group = _binary(columns[names.index(query.treated)], "treated")
    period = _binary(columns[names.index(query.post)], "post")
    clusters = subjects if cluster is None else list(columns[names.index(cluster)])
    if any(not isinstance(value, str) or not value for value in clusters):
        raise CausalValueError("cluster IDs must be non-empty strings")
    try:
        estimate, standard_error, n_treated, n_control, n_clusters = _panel_difference_in_differences(
            outcome, subjects, group, period, clusters
        )
    except ValueError as error:
        raise CausalValueError(str(error)) from error
    return PanelDifferenceInDifferencesEstimate(
        float(estimate), float(standard_error), int(n_treated), int(n_control), int(n_clusters)
    )


def _integer_column(values: Sequence[Any], name: str, *, minimum: int) -> list[int]:
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


def estimate_staggered_event_study(
    data: Any, query: StaggeredAdoption, *, cluster: str | None = None
) -> StaggeredEventStudyEstimate:
    """Return cohort-by-event-time contrasts relative to event time -1.

    Pre-adoption coefficients are descriptive checks, not a parallel-trends
    test. Pointwise standard errors aggregate subject-level influence scores
    into independent clusters and apply G/(G-1). Supply a higher-level cluster
    column when appropriate; subject IDs are the default. The retained event
    study reports separate post-adoption pointwise intervals only with the
    calibrated cluster support. It does not validate parallel trends or form
    simultaneous bands.
    """
    if not isinstance(query, StaggeredAdoption):
        raise CausalValueError("query must be a StaggeredAdoption")
    if query.event_study:
        from ._analyze import analyze
        result = analyze(data, query=query)
        if result.panel_did is None or not isinstance(result.panel_did, StaggeredEventStudyEstimate):
            raise CausalValueError("retained event study did not return cohort-specific effects")
        return result.panel_did
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
        raw = _staggered_event_study(outcome, subjects, periods, cohorts, clusters)
    except ValueError as error:
        raise CausalValueError(str(error)) from error
    effects = tuple(
        StaggeredEventTimeEffect(int(g), int(t), int(e), float(estimate), float(se), int(nt), int(nc), int(gc))
        for g, t, e, estimate, nt, nc, se, gc in raw
    )
    return StaggeredEventStudyEstimate(effects)


def estimate_synthetic_control(data: Any, query: SyntheticControl) -> SyntheticControlEstimate:
    """Fit simplex donor weights to pre-period outcomes and compare post means.

    The returned placebo rank is the finite donor-pool tail fraction from
    leave-one-donor-out fits. It is explicitly uncalibrated, makes no inference
    claim, and does not change the point-only uncertainty status.
    """

    if not isinstance(query, SyntheticControl):
        raise CausalValueError("query must be a SyntheticControl")
    if query.uniform_unit_randomization or query.augmentation_ridge is not None:
        from .estimation import PreparedAnalysis

        result = PreparedAnalysis.prepare(data, query=query).estimate(data).synthetic_control
        assert result is not None
        return result
    names, columns = _raw_columns(data)
    for name in (query.outcome, query.unit, query.period):
        if name not in names:
            raise CausalValueError(f"required synthetic-control column {name!r} is missing")
    outcome = np.asarray(columns[names.index(query.outcome)], dtype=np.float64)
    units = list(columns[names.index(query.unit)])
    if any(not isinstance(unit, str) or not unit for unit in units):
        raise CausalValueError("unit IDs must be non-empty strings")
    periods = _integer_column(columns[names.index(query.period)], "period", minimum=1)
    try:
        estimate, pre_rmse, raw_weights, placebo_effects, placebo_rank, n_pre, n_post = (
            _synthetic_control(
                outcome,
                units,
                periods,
                query.treated_unit,
                int(query.intervention_period),
            )
        )
    except ValueError as error:
        raise CausalValueError(str(error)) from error
    weights = tuple((str(unit), float(weight)) for unit, weight in raw_weights)
    squared_mass = sum(weight**2 for _, weight in weights)
    effective_donors = 1.0 / squared_mass if squared_mass > 0 else 0.0
    return SyntheticControlEstimate(
        estimate=float(estimate),
        pre_treatment_rmse=float(pre_rmse),
        donor_weights=weights,
        placebo_effects=tuple(float(value) for value in placebo_effects),
        placebo_rank_p_value=float(placebo_rank),
        effective_donors=float(effective_donors),
        n_donors=len(weights),
        n_pre_periods=int(n_pre),
        n_post_periods=int(n_post),
    )


def estimate_synthetic_did(
    data: Any, query: SyntheticDifferenceInDifferences
) -> SyntheticDifferenceInDifferencesEstimate:
    """Estimate SDID with native simplex weights on units and pre-periods.

    The balanced-panel kernel reports a point estimate, pre-fit RMSE and both
    weight vectors. A declared uniform single-unit assignment requests an
    exact sharp-null p-value through retained analysis; it reports no interval.
    """

    if not isinstance(query, SyntheticDifferenceInDifferences):
        raise CausalValueError("query must be a SyntheticDifferenceInDifferences")
    if query.uniform_unit_randomization:
        from .estimation import PreparedAnalysis

        result = PreparedAnalysis.prepare(data, query=query).estimate(data).synthetic_did
        assert result is not None
        return result
    names, columns = _raw_columns(data)
    for name in (query.outcome, query.unit, query.period):
        if name not in names:
            raise CausalValueError(f"required synthetic DiD column {name!r} is missing")
    outcome = np.asarray(columns[names.index(query.outcome)], dtype=np.float64)
    units = list(columns[names.index(query.unit)])
    if any(not isinstance(unit, str) or not unit for unit in units):
        raise CausalValueError("unit IDs must be non-empty strings")
    periods = _integer_column(columns[names.index(query.period)], "period", minimum=1)
    try:
        estimate, pre_rmse, raw_units, raw_times, n_donors, n_pre, n_post = (
            _synthetic_did(outcome, units, periods, query.treated_unit, int(query.intervention_period))
        )
    except ValueError as error:
        raise CausalValueError(str(error)) from error
    return SyntheticDifferenceInDifferencesEstimate(
        estimate=float(estimate),
        pre_treatment_rmse=float(pre_rmse),
        donor_weights=tuple((str(unit), float(weight)) for unit, weight in raw_units),
        time_weights=tuple((int(period), float(weight)) for period, weight in raw_times),
        n_donors=int(n_donors),
        n_pre_periods=int(n_pre),
        n_post_periods=int(n_post),
    )


def estimate_augmented_panel_did(
    data: Any, query: AugmentedPanelDiD
) -> AugmentedPanelDiDEstimate:
    """Estimate panel ATT with supplied propensity and untreated-change nuisance.

    The caller supplies one row per subject. The propensity and untreated
    change prediction columns are passed unchanged to the Rust estimator;
    the point utility does not fit or validate nuisance models.
    """

    if not isinstance(query, AugmentedPanelDiD):
        raise CausalValueError("query must be an AugmentedPanelDiD")
    names, columns = _raw_columns(data)
    required = (
        query.outcome_pre,
        query.outcome_post,
        query.subject,
        query.treated,
        query.propensity,
        query.untreated_change_prediction,
    )
    if query.cluster is not None:
        required += (query.cluster,)
    for name in required:
        if name not in names:
            raise CausalValueError(f"required augmented panel DiD column {name!r} is missing")
    subjects = list(columns[names.index(query.subject)])
    if any(not isinstance(subject, str) or not subject for subject in subjects):
        raise CausalValueError("subject IDs must be non-empty strings")
    if len(set(subjects)) != len(subjects):
        raise CausalValueError("augmented panel DiD requires one row per unique subject")
    treated = _binary(columns[names.index(query.treated)], "treated")
    try:
        pre = np.asarray(columns[names.index(query.outcome_pre)], dtype=np.float64)
        post = np.asarray(columns[names.index(query.outcome_post)], dtype=np.float64)
        propensity = np.asarray(columns[names.index(query.propensity)], dtype=np.float64)
        prediction = np.asarray(
            columns[names.index(query.untreated_change_prediction)], dtype=np.float64
        )
    except (TypeError, ValueError) as error:
        raise CausalValueError("outcomes, propensities, and predictions must be numeric") from error
    try:
        estimate, n_treated, n_control, p_min, p_max, ess = _augmented_panel_did(
            pre, post, treated, propensity, prediction
        )
    except ValueError as error:
        raise CausalValueError(str(error)) from error
    return AugmentedPanelDiDEstimate(
        estimate=float(estimate),
        treated_subjects=int(n_treated),
        control_subjects=int(n_control),
        propensity_min=float(p_min),
        propensity_max=float(p_max),
        effective_control_sample_size=float(ess),
        nuisance_predictions_cross_fitted=query.predictions_cross_fitted,
        clusters=len(set(subjects if query.cluster is None else columns[names.index(query.cluster)])),
    )


def _validate_rd_spec(outcome: str, treatment: str, running: str, cutoff: float, bandwidth: float) -> None:
    fields = (outcome, treatment, running)
    if any(not isinstance(value, str) or not value.strip() for value in fields):
        raise CausalValueError("outcome, treatment, and running must be non-empty column names")
    if len(set(fields)) != len(fields):
        raise CausalValueError("outcome, treatment, and running must name distinct columns")
    if isinstance(cutoff, (bool, np.bool_)) or not isinstance(
        cutoff, (int, float, np.integer, np.floating)
    ) or not np.isfinite(cutoff):
        raise CausalValueError("cutoff must be finite")
    if isinstance(bandwidth, (bool, np.bool_)) or not isinstance(
        bandwidth, (int, float, np.integer, np.floating)
    ) or not np.isfinite(bandwidth) or bandwidth <= 0:
        raise CausalValueError("bandwidth must be finite and positive")


def _estimate_local_polynomial_ratio(data: Any, query: Any, *, kink: bool) -> LocalPolynomialRatioEstimate:
    names, columns = _raw_columns(data)
    for name in (query.outcome, query.treatment, query.running):
        if name not in names:
            raise CausalValueError(f"required local-polynomial column {name!r} is missing")
    outcome = np.asarray(columns[names.index(query.outcome)], dtype=np.float64)
    treatment = np.asarray(columns[names.index(query.treatment)], dtype=np.float64)
    running = np.asarray(columns[names.index(query.running)], dtype=np.float64)
    try:
        (
            estimate,
            reduced,
            first_stage,
            n_left,
            n_right,
            standard_error,
            ci_lower,
            ci_upper,
            reduced_se,
            first_stage_se,
        ) = _local_polynomial_fuzzy_discontinuity(
            running, outcome, treatment, float(query.cutoff), float(query.bandwidth), kink
        )
    except ValueError as error:
        raise CausalValueError(str(error)) from error
    return LocalPolynomialRatioEstimate(
        estimate=float(estimate),
        reduced_form_discontinuity=float(reduced),
        first_stage_discontinuity=float(first_stage),
        observations_left=int(n_left),
        observations_right=int(n_right),
        standard_error=float(standard_error),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        reduced_form_standard_error=float(reduced_se),
        first_stage_standard_error=float(first_stage_se),
        cutoff=float(query.cutoff),
        bandwidth=float(query.bandwidth),
        kink=bool(kink),
        design=(
            "fuzzy_regression_kink_local_quadratic"
            if kink
            else "fuzzy_regression_discontinuity_local_quadratic"
        ),
        assumptions=(
            (
                "potential_outcome_derivatives_are_smooth_at_cutoff",
                "no_precise_manipulation_of_running_variable",
                "exclusion_restriction_for_threshold_induced_slope_change",
                "monotonicity_for_local_complier_interpretation",
                "no_interference",
            )
            if kink
            else LocalPolynomialRatioEstimate.__dataclass_fields__["assumptions"].default
        ),
        diagnostics=(
            "local_quadratic_triangular_kernel",
            "left_right_window_counts_reported",
            "quartic_pilot_bias_correction_at_same_bandwidth" if kink else "cubic_pilot_bias_correction_at_same_bandwidth",
            "hc0_sandwich_covariance_with_delta_method_ratio_se",
            "nominal_95_interval_calibrated_on_strong_first_stage_fixtures",
        ),
    )


def estimate_fuzzy_rd(data: Any, query: FuzzyRegressionDiscontinuity) -> LocalPolynomialRatioEstimate:
    """Estimate fuzzy RD as local-quadratic outcome jump / treatment jump.

    The implementation corrects the local-quadratic leading cubic bias with a
    same-bandwidth local-cubic pilot and uses HC0 covariance with a delta-method
    normal interval. Repeated-sampling evidence covers a strong-first-stage,
    fixed-bandwidth fixture; the design remains outside the support matrix.
    """
    if not isinstance(query, FuzzyRegressionDiscontinuity):
        raise CausalValueError("query must be a FuzzyRegressionDiscontinuity")
    return _estimate_local_polynomial_ratio(data, query, kink=False)


def estimate_regression_kink(data: Any, query: RegressionKink) -> LocalPolynomialRatioEstimate:
    """Estimate a fuzzy regression-kink ratio from local-quadratic slope changes."""
    if not isinstance(query, RegressionKink):
        raise CausalValueError("query must be a RegressionKink")
    return _estimate_local_polynomial_ratio(data, query, kink=True)


__all__ = [
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
    "FuzzyRegressionDiscontinuity",
    "RegressionKink",
    "LocalPolynomialRatioEstimate",
    "estimate_did",
    "estimate_panel_did",
    "estimate_group_time_att",
    "estimate_staggered_event_study",
    "estimate_synthetic_control",
    "estimate_synthetic_did",
    "estimate_augmented_panel_did",
    "estimate_fuzzy_rd",
    "estimate_regression_kink",
]
