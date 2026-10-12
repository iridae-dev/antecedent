"""High-level estimation entry points."""

from __future__ import annotations

import importlib
import json
import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, cast

import numpy as np

from ._api import describe_refusal
from ._coerce import coerce_latency, coerce_query, coerce_refute
from ._data import as_columns, ingest_columns, try_as_arrow_c_columns
from ._native import (
    AnalysisResult as TemporalAnalysisResult,
)
from ._native import (
    AteAnalysisResult,
    MediationEffectsSummary,
    mediation_effects_summary,
)
from ._native import (
    analyze_ate_many as _analyze_ate_many,
)
from ._native import (
    identify_ate as _identify_ate,
)
from ._native import (
    identify_ate_admg as _identify_ate_admg,
)
from ._native import (
    prepare_ate_batch as _prepare_ate_batch,
)
from ._native import (
    prepare_cells_batch as _prepare_cells_batch,
)
from ._unset_license import unset_dose_license, unset_panel_license, unset_policy_license
from .discovery import (
    DbnPosterior,
    ExactDagPosterior,
    GraphPosterior,
    cpdag_oriented_edges,
)
from .errors import (
    CausalTypeError,
    CausalUnsupportedError,
    CausalValueError,
)
from .experiment import (
    ComplierEffect,
    RandomizedEffect,
    RandomizedExperimentEstimate,
    SwitchbackEffect,
    TreatmentOnTreated,
)
from .graph import Admg, Cpdag, Dag, Pag, TemporalCpdag, TemporalDag, TemporalPag, TieredBackground
from .ids import Estimator, Identifier, Latency, Refute
from .inference import (
    Bayesian,
    ClassPrior,
    Frequentist,
)
from .interference import (
    InterferenceEstimate,
    InterferencePointwiseInterval,
    InterferenceQuery,
    InterferenceSupport,
    RandomizationContrast,
)
from .policy import (
    ConditionalDoseResponse,
    ConditionalDoseResponseEstimate,
    ConditionalDoseResponsePoint,
    DoublyRobustPolicyEvaluation,
    FixedDosePolicyValueEstimate,
    MultiActionPolicyValue,
    PolicyValue,
)
from .population import coerce_target_population
from .quasi import (
    AugmentedPanelDiD,
    AugmentedPanelDiDEstimate,
    FuzzyRegressionDiscontinuity,
    LocalPolynomialRatioEstimate,
    PanelDifferenceInDifferences,
    PanelDifferenceInDifferencesEstimate,
    RegressionKink,
    SharpRegressionDiscontinuity,
    StaggeredAdoption,
    StaggeredEventStudyEstimate,
    StaggeredEventTimeEffect,
    SyntheticControl,
    SyntheticControlEstimate,
    SyntheticDifferenceInDifferences,
    SyntheticDifferenceInDifferencesEstimate,
)
from .query import (
    AnomalyAttribution,
    AverageDerivative,
    AverageEffect,
    ChangeAttribution,
    ConditionalEffect,
    Counterfactual,
    DirectionalDerivative,
    Elasticity,
    InterventionalDistribution,
    InterventionResponse,
    MediationEffect,
    NestedCounterfactual,
    PathSpecificEffect,
    PointDerivative,
    PulseEffect,
    ResponseCurve,
    ResponseJacobian,
    SemiElasticity,
    SustainedEffect,
    TemporalMediationEffect,
)
from .regimes import LongitudinalRegime, LongitudinalRegimeEstimate
from .results import (
    AnalysisResult,
    CausalResponseView,
    ConflictSummaryView,
    DistributionAtomView,
    EffectEnvelope,
    EstimateView,
    IdentificationView,
    InspectionReport,
    MediationView,
    PerformanceView,
    PhysicalPlanView,
    PlanView,
    PosteriorView,
    PredictiveCheckReport,
    PriorSensitivityReport,
    ProbabilityIntervalView,
    ReasoningSlots,
    RefutationReport,
    ResponseEnvelopeView,
    ResponseUncertainty,
    ResponseValidationCheck,
    ResponseValidationView,
    ResponseView,
    SupportDiagnostic,
    SupportReport,
    TemporalMediationGridView,
    TemporalMediationSliceView,
    ValidationView,
)
from .results._views import (
    PenalizedFallback,
    PenalizedFoldSupport,
    RefitBootstrapReport,
    ReplicatePenalties,
)
from .results.response import IntervalInterpretation, SupportStatus, UncertaintyKind
from .survival import (
    CompetingRisksOutcome,
    CumulativeIncidenceEstimate,
    SurvivalDifferenceBand,
    SurvivalEstimate,
    SurvivalOutcome,
)
from .transport import (
    Transport,
    TransportControls,
    TransportInference,
)
from .transport._impl import (
    ExactTransportDistribution,
    OverlapDiagnostic,
    StatisticalTransportDistribution,
    TransportOverlapReport,
    TransportResponseGrid,
    TransportStage,
)
from .transport.advanced import (
    ExactTransportQuery,
    StatisticalTransportQuery,
    TransportQuery,
    TransportResponseGridQuery,
)

if TYPE_CHECKING:
    from .preflight import (
        BatchCostEstimate,
        BatchPreflightReport,
        CostEstimate,
        NuisanceFitDiagnostics,
        PreflightReport,
        RankDropPlan,
    )


def _refutation_reports_from_raw(validation: Any) -> list[RefutationReport]:
    """One :class:`RefutationReport` per entry in a nested ``validation.reports``."""
    return [
        RefutationReport(
            refuter=r.refuter,
            original_ate=r.original_ate,
            refuted_ate=r.refuted_ate,
            comparison=r.comparison,
            informative=r.informative,
            passed=r.passed,
            failure_condition=r.failure_condition,
            replicates=r.replicates,
        )
        for r in getattr(validation, "reports", None) or ()
    ]


def _plan_from_raw(raw: Any) -> PlanView:
    return PlanView(
        plan_id=str(getattr(raw, "plan_id", "") or ""),
        modality=getattr(raw, "modality", None),
        discovery_algorithm=getattr(raw, "discovery_algorithm", None),
        structure_source=getattr(raw, "structure_source", None),
        graph_review_required=bool(getattr(raw, "graph_review_required", False)),
        identifier=getattr(raw, "plan_identifier", None),
        estimator=getattr(raw, "plan_estimator", None)
        or (getattr(raw, "estimator_id", None) or None),
        validation_suite=getattr(raw, "validation_suite", None),
    )


def _optional_finite_ate(value: Any) -> float | None:
    """Omit non-finite sentinels so function-valued results have no scalar ate."""
    if value is None:
        return None
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    return as_float if math.isfinite(as_float) else None


def _probability_interval_from_raw(raw: Any) -> ProbabilityIntervalView | None:
    """Native ``ProbabilityIntervalSection`` → view (``None`` passes through)."""
    if raw is None:
        return None
    return ProbabilityIntervalView(
        level=raw.level,
        lower=raw.lower,
        upper=raw.upper,
        unavailable=raw.unavailable,
    )


def _penalized_fallback_from_raw(raw: Any) -> PenalizedFallback | None:
    """Native fallback tuple → view (``None`` passes through)."""
    if raw is None:
        return None
    stage, fold, reason, message, destination = raw
    return PenalizedFallback(
        stage=str(stage),
        fold=int(fold),
        reason=str(reason),
        message=str(message),
        destination=str(destination),
    )


def _penalized_bootstrap_from_raw(raw: Any) -> RefitBootstrapReport | None:
    """Native refit-bootstrap tuple → view (``None`` passes through)."""
    if raw is None:
        return None
    kind, se, requested, ok, failed, cancelled, penalties, influence_se = raw
    return RefitBootstrapReport(
        uncertainty_kind=str(kind),
        se=se,
        replicates_requested=int(requested),
        replicates_ok=int(ok),
        replicates_failed=int(failed),
        cancelled=bool(cancelled),
        replicate_penalties=tuple(
            ReplicatePenalties(replicate=int(r), lambdas=tuple(float(v) for v in lambdas))
            for r, lambdas in penalties
        ),
        influence_se=float(influence_se),
    )


def _unit_effect_intervals_from_raw(raw: Any) -> list[tuple[float, float]] | None:
    """Native per-unit ``(lower, upper)`` pairs → tuples (``None`` passes through)."""
    pairs = getattr(raw, "unit_effect_intervals", None)
    if pairs is None:
        return None
    return [(float(lower), float(upper)) for lower, upper in pairs]


def _distribution_atoms_from_raw(sec_estimate: Any) -> tuple[DistributionAtomView, ...] | None:
    """Native distribution atoms with their bounded probability intervals."""
    atoms = getattr(sec_estimate, "distribution_atoms", None)
    if atoms is None:
        return None
    return tuple(
        DistributionAtomView(
            outcomes=tuple((str(name), value) for name, value in atom.outcomes),
            conditioning=tuple((str(name), value) for name, value in atom.conditioning),
            probability=float(atom.probability),
            se_bootstrap=atom.se_bootstrap,
            interval=_probability_interval_from_raw(atom.interval),
        )
        for atom in atoms
    )


def _transport_overlap_from_raw(raw: Any) -> TransportOverlapReport | None:
    section = getattr(raw, "transport", None)
    if section is None:
        return None
    return TransportOverlapReport(
        OverlapDiagnostic(
            section.selection_probability_min,
            section.selection_probability_max,
            section.selection_effective_sample_size,
            section.selection_extreme_weight_count,
        ),
        OverlapDiagnostic(
            section.treatment_probability_min,
            section.treatment_probability_max,
            section.treatment_effective_sample_size,
            section.treatment_extreme_weight_count,
        ),
    )


def _interference_from_raw(raw: Any) -> InterferenceEstimate | None:
    section = getattr(raw, "interference", None)
    if section is None:
        return None
    estimator_id = getattr(raw, "estimator_id", "")
    observational = section.from_probability_method.startswith("supplied_")
    native_interval = section.pointwise_interval
    pointwise_interval = (
        InterferencePointwiseInterval(
            lower=native_interval[0],
            upper=native_interval[1],
            standard_error=native_interval[2],
            degrees_of_freedom=native_interval[3],
            first_stage_arm_clusters=(native_interval[4], native_interval[5]),
            method=native_interval[6],
        )
        if native_interval is not None
        else None
    )
    if observational:
        uncertainty = (
            "Pointwise 95% cluster t interval for the HT contrast under known fixed exposure "
            "probabilities, independent clusters, and declared network exchangeability."
            if pointwise_interval is not None
            else "no interval: descriptive cluster CR1 variance for the HT contrast; pointwise interval withheld "
            "because exposure probabilities were fitted or cluster support failed."
        )
    elif estimator_id == "interference.saturation_exact":
        uncertainty = (
            "Independent-cluster Neyman pointwise 95% interval with Welch degrees of freedom; "
            "conditional on the declared two-stage randomization and within-cluster exposure mapping."
            if pointwise_interval is not None
            else "Descriptive covariance-free saturation variance proxy; pointwise interval withheld because exposure or cluster support failed."
        )
    elif estimator_id == "interference.cluster_neyman":
        uncertainty = (
            "Independent-cluster Neyman pointwise 95% interval for the total effect with Welch degrees of freedom."
            if pointwise_interval is not None
            else "Conservative cluster-level Neyman variance for the total effect; pointwise interval withheld because cluster support failed."
        )
    else:
        uncertainty = (
            "Conservative Young variance bound under the declared randomization; "
            "see the support matrix for interval licensing."
        )
    return InterferenceEstimate(
        RandomizationContrast(
            section.horvitz_thompson, section.hajek, section.conservative_variance
        ),
        section.from_probability_method,
        section.to_probability_method,
        section.minimum_exposure_probability,
        support=(
            InterferenceSupport(
                section.from_exposed_units,
                section.to_exposed_units,
                section.from_exposed_clusters,
                section.to_exposed_clusters,
                section.minimum_exposure_probability,
                True,
                section.maximum_exposure_probability,
                section.clusters,
            )
            if section.from_exposed_units is not None
            else None
        ),
        uncertainty_semantics=uncertainty,
        provenance={
            "operation_ids": [
                "estimate.interference.observational_ipw"
                if observational
                else "estimate.interference.saturation_exact"
                if estimator_id == "interference.saturation_exact"
                else "stats.randomized_interference"
            ]
        },
        pointwise_interval=pointwise_interval,
        interval_unavailable_reason=section.interval_unavailable_reason,
        support_status=(
            "licensed"
            if section.graphless_support_status == "licensed"
            else "unlicensed_point_utility"
        ),
    )


def _randomized_effect_from_raw(raw: Any) -> RandomizedExperimentEstimate | None:
    section = getattr(raw, "randomized_effect", None)
    if section is None:
        return None
    return RandomizedExperimentEstimate(
        effect=section.effect,
        variance_upper_bound=section.variance_upper_bound,
        assignment_design=section.assignment_design,
        assignment_units=tuple(section.assignment_units),
        outcome_units=tuple(section.outcome_units),
        blocks=tuple(section.blocks) or None,
        periods=tuple(section.periods) or None,
        estimand=section.estimand,
        intention_to_treat_effect=section.intention_to_treat_effect,
        first_stage_effect=section.first_stage_effect,
        received_treatment=tuple(section.received_treatment)
        if section.received_treatment is not None
        else None,
        randomization_p_value=section.randomization_p_value,
        randomization_allocations=section.randomization_allocations,
        second_factor_effect=section.second_factor_effect,
        factorial_interaction=section.factorial_interaction,
        second_factor_variance=section.second_factor_variance,
        factorial_interaction_variance=section.factorial_interaction_variance,
        multi_arm_values=tuple(tuple(item) for item in section.multi_arm_values),
        standard_error=section.standard_error,
        interval_95=section.interval_95,
        second_factor_interval_95=section.second_factor_interval_95,
        factorial_interaction_interval_95=section.factorial_interaction_interval_95,
        multi_arm_intervals_95=tuple(section.multi_arm_intervals_95),
        treatment_arms=(section.control_arm, section.treatment_arm),
        control_units=section.control_units,
        treatment_units=section.treatment_units,
        minimum_assignment_probability=section.minimum_assignment_probability,
        uncertainty=section.uncertainty,
        support_status=(
            section.graphless_support_status
            or (
                "off_axis_interval_evidence"
                if section.interval_95 is not None
                else "unlicensed_off_matrix"
            )
        ),
    )


def _panel_did_from_raw(
    raw: Any, query: Any = None
) -> (
    PanelDifferenceInDifferencesEstimate
    | StaggeredEventStudyEstimate
    | AugmentedPanelDiDEstimate
    | None
):
    section = getattr(raw, "panel_did", None)
    if section is None:
        return None
    if isinstance(query, AugmentedPanelDiD):
        if section.augmented is None:
            raise CausalValueError("augmented panel DiD result is missing overlap diagnostics")
        p_min, p_max, ess, declared = section.augmented
        return AugmentedPanelDiDEstimate(
            estimate=section.effect,
            treated_subjects=section.treated_subjects,
            control_subjects=section.comparison_subjects,
            propensity_min=p_min,
            propensity_max=p_max,
            effective_control_sample_size=ess,
            nuisance_predictions_cross_fitted=declared,
            clusters=section.clusters,
            uncertainty=section.uncertainty,
        )
    if isinstance(query, StaggeredAdoption) and query.event_study:
        intervals = section.event_time_intervals_95
        if len(intervals) != len(section.event_time_effects):
            raise CausalValueError("event-time intervals must align with cohort contrasts")
        effects = tuple(
            StaggeredEventTimeEffect(
                int(cohort),
                int(period),
                int(event_time),
                float(effect),
                float(se),
                int(treated),
                int(controls),
                int(clusters),
                tuple(interval) if interval is not None else None,
            )
            for (
                cohort,
                period,
                event_time,
                effect,
                treated,
                controls,
                se,
                clusters,
            ), interval in zip(section.event_time_effects, intervals, strict=True)
        )
        supported = any(effect.interval_95 is not None for effect in effects)
        return StaggeredEventStudyEstimate(
            effects,
            uncertainty=(
                "event_time_pointwise_normal_intervals_independent_clusters"
                if supported
                else "cluster_robust_se_only_pointwise_cr1_unlicensed"
            ),
            support_status=(
                "licensed"
                if section.graphless_support_status == "licensed"
                else "off_axis_pointwise_95"
                if supported
                else "unlicensed_point_utility"
            ),
        )
    repeated = (
        isinstance(query, PanelDifferenceInDifferences)
        and query.sampling == "repeated_cross_section"
    )
    staggered = isinstance(query, StaggeredAdoption)
    return PanelDifferenceInDifferencesEstimate(
        estimate=section.effect,
        standard_error=section.standard_error,
        treated_subjects=section.treated_subjects,
        control_subjects=section.comparison_subjects,
        clusters=section.clusters,
        uncertainty=section.uncertainty,
        interval_95=section.interval_95,
        design="balanced_staggered_adoption_group_time_att"
        if staggered
        else ("repeated_cross_section_2x2" if repeated else "balanced_two_period_panel"),
        assumptions=(
            "cohort_specific_parallel_untreated_trends",
            "no_anticipation",
            "absorbing_treatment_after_adoption",
            "never_treated_controls_are_valid",
            "no_interference",
            "balanced_panel",
            "independent_sampling_clusters",
        )
        if staggered
        else (
            "parallel_untreated_trends",
            "no_anticipation",
            "stable_group_definition",
            "no_interference",
            "independent_sampling_clusters",
        )
        if repeated
        else (
            "parallel_untreated_trends",
            "no_anticipation",
            "stable_treatment_assignment_within_subject",
            "complete_pre_post_panel",
            "no_interference",
            "independent_sampling_clusters",
        ),
        support_status=(
            section.graphless_support_status
            or unset_panel_license(has_interval=section.interval_95 is not None)
        ),
        cohort=query.target_cohort if staggered else None,
        period=query.target_period if staggered else None,
    )


def _synthetic_control_from_raw(raw: Any) -> SyntheticControlEstimate | None:
    section = getattr(raw, "synthetic_control", None)
    if section is None:
        return None
    return SyntheticControlEstimate(
        estimate=section.effect,
        pre_treatment_rmse=section.pre_treatment_rmse,
        donor_weights=tuple((str(unit), float(weight)) for unit, weight in section.donor_weights),
        placebo_effects=tuple(section.placebo_effects),
        placebo_rank_p_value=section.placebo_rank,
        effective_donors=section.effective_donors,
        n_donors=len(section.donor_weights),
        n_pre_periods=section.n_pre_periods,
        n_post_periods=section.n_post_periods,
        randomization_p_value=section.randomization_p_value,
        randomization_null_effect=section.randomization_null_effect,
        randomization_statistics=tuple(
            (str(unit), float(statistic)) for unit, statistic in section.randomization_statistics
        ),
        unadjusted_effect=section.unadjusted_effect,
        outcome_model_correction=section.outcome_model_correction,
        augmentation_ridge=section.augmentation_ridge,
        uncertainty=section.uncertainty,
        assumptions=(
            "no_anticipation",
            "stable_treatment_after_intervention",
            "convex_donor_combination_is_a_valid_counterfactual",
            "no_interference_between_units",
            "no_concurrent_treated_unit_specific_shock",
        )
        + (
            ("donor_ridge_outcome_model_transports", "donor_outcome_model_correction_is_valid")
            if section.augmentation_ridge is not None
            else ()
        )
        + (
            ("uniform_single_treated_unit_assignment",)
            if section.randomization_p_value is not None
            else ()
        ),
        diagnostics=(
            "donor_ridge_outcome_model_correction",
            "unadjusted_placebo_rank_is_descriptive",
            "no_confidence_interval",
        )
        if section.augmentation_ridge is not None
        else (
            ("exact_uniform_unit_randomization_sharp_null_test", "no_confidence_interval")
            if section.randomization_p_value is not None
            else ("placebo_rank_assumes_exchangeable_donors_and_is_not_calibrated",)
        ),
    )


def _synthetic_did_from_raw(raw: Any) -> SyntheticDifferenceInDifferencesEstimate | None:
    section = getattr(raw, "synthetic_did", None)
    if section is None:
        return None
    return SyntheticDifferenceInDifferencesEstimate(
        estimate=section.effect,
        pre_treatment_rmse=section.pre_treatment_rmse,
        donor_weights=tuple((str(unit), float(weight)) for unit, weight in section.donor_weights),
        time_weights=tuple((int(period), float(weight)) for period, weight in section.time_weights),
        n_donors=section.n_donors,
        n_pre_periods=section.n_pre_periods,
        n_post_periods=section.n_post_periods,
        randomization_p_value=section.randomization_p_value,
        randomization_null_effect=section.randomization_null_effect,
        randomization_statistics=tuple(
            (str(unit), float(statistic)) for unit, statistic in section.randomization_statistics
        ),
        uncertainty=(
            section.uncertainty if section.randomization_p_value is not None else "point_only"
        ),
        assumptions=(
            "no_anticipation",
            "stable_treatment_after_intervention",
            "convex_unit_and_time_weights_represent_untreated_counterfactual_trends",
            "no_interference_between_units",
            "no_concurrent_treated_unit_specific_shock",
        )
        + (
            ("uniform_single_treated_unit_assignment",)
            if section.randomization_p_value is not None
            else ()
        ),
        diagnostics=(
            ("exact_uniform_unit_randomization_sharp_null_test", "no_confidence_interval")
            if section.randomization_p_value is not None
            else (
                "balanced_panel_and_two_pre_periods_required",
                "unit_and_time_weights_are_simplex_constrained",
                "pre_fit_rmse_reported_without_acceptance_threshold",
                "no_interval_or_calibrated_inference",
            )
        ),
    )


def _local_polynomial_ratio_from_raw(raw: Any, query: Any) -> LocalPolynomialRatioEstimate | None:
    section = getattr(raw, "local_polynomial_ratio", None)
    if section is None:
        return None
    return LocalPolynomialRatioEstimate(
        estimate=section.effect,
        reduced_form_discontinuity=section.reduced_form,
        first_stage_discontinuity=section.first_stage,
        observations_left=section.n_left,
        observations_right=section.n_right,
        standard_error=section.standard_error,
        ci_lower=section.ci_lower,
        ci_upper=section.ci_upper,
        reduced_form_standard_error=section.reduced_form_standard_error,
        first_stage_standard_error=section.first_stage_standard_error,
        cutoff=section.cutoff,
        bandwidth=section.bandwidth,
        kink=section.kink,
        uncertainty=section.uncertainty,
        design="regression_kink_local_polynomial"
        if isinstance(query, RegressionKink)
        else "fuzzy_regression_discontinuity_local_polynomial",
        assumptions=(
            "continuous_untreated_potential_outcome_and_treatment_derivatives_at_cutoff"
            if isinstance(query, RegressionKink)
            else "continuous_untreated_potential_outcomes_at_cutoff",
            "no_precise_running_variable_manipulation",
            "exclusion_restriction",
            "local_monotonicity",
            "independent_local_observations",
        ),
        support_status=(
            "licensed"
            if getattr(section, "graphless_support_status", None) == "licensed"
            else "off_axis_interval_evidence"
        ),
        diagnostics=(
            "local_quadratic_triangular_kernel",
            "quartic_pilot_bias_correction_at_same_bandwidth"
            if isinstance(query, RegressionKink)
            else "cubic_pilot_bias_correction_at_same_bandwidth",
            "fixed_bandwidth_local_support_reported",
            "hc0_delta_method_standard_error",
            "calibrated_strong_first_stage_fixture",
        ),
    )


def _survival_from_raw(
    raw: Any, query: Any = None
) -> SurvivalEstimate | CumulativeIncidenceEstimate | None:
    section = getattr(raw, "survival", None)
    if section is None:
        return None
    times = tuple(section.times)
    control = tuple(section.control)
    treated = tuple(section.treated)
    fixed_g = getattr(query, "known_censoring", None) is not None
    has_interval = section.difference_at_tau_interval is not None
    native_band = section.difference_band
    # The band license is the sixth tuple element on builds that carry it; older
    # builds return a five-element tuple, so read it defensively.
    band_status = native_band[5] if native_band is not None and len(native_band) > 5 else None
    difference_band = (
        SurvivalDifferenceBand(
            times=tuple(native_band[0]),
            difference=tuple(native_band[1]),
            lower=tuple(native_band[2]),
            upper=tuple(native_band[3]),
            replicates_ok=native_band[4],
            support_status=band_status or "unlicensed_simultaneous_band",
        )
        if native_band is not None
        else None
    )
    observation_assumptions: tuple[str, ...] = (
        (
            "correct_caller_supplied_conditional_censoring_survival",
            "independent_censoring_given_supplied_history",
            "sequential_censoring_positivity",
        )
        if fixed_g
        else ("independent_right_censoring_within_arm",)
    )
    if getattr(query, "delayed_entry", None) is not None:
        observation_assumptions += (
            "marginal_independent_delayed_entry_and_censoring_within_arm"
            if fixed_g
            else "independent_delayed_entry_within_arm",
        )
    provenance = section.censoring_survival_provenance
    scalar_licensed = getattr(raw, "evidence_status", None) == "licensed"
    band_licensed = band_status == "licensed"
    # The paired RMST + horizon survival scalar license and the simultaneous
    # band license are distinct graphless claims. The scalar license takes the
    # top-level slot when present; a licensed band that stands alone (for example
    # an off-axis competing-incidence scalar) is reported as such; the band's own
    # license is always carried on ``difference_band.support_status``.
    support_status = (
        "licensed"
        if scalar_licensed
        else "licensed_simultaneous_band"
        if band_licensed
        else "unlicensed_simultaneous_band"
        if difference_band is not None
        else "unlicensed_pointwise_interval"
        if has_interval
        else "unlicensed_point_utility"
    )
    if isinstance(query, CompetingRisksOutcome) or section.target_cause is not None:
        return CumulativeIncidenceEstimate(
            target_cause=int(section.target_cause),
            times=times,
            control_incidence=control,
            treated_incidence=treated,
            incidence_difference=treated[-1] - control[-1],
            tau=section.tau,
            uncertainty=section.uncertainty,
            incidence_difference_interval=(
                tuple(section.difference_at_tau_interval)
                if section.difference_at_tau_interval is not None
                else None
            ),
            bootstrap_replicates_requested=section.bootstrap_replicates_requested,
            bootstrap_replicates_ok=section.bootstrap_replicates_ok,
            censoring_survival_provenance=provenance,
            difference_band=difference_band,
            band_unavailable_reason=section.band_unavailable_reason,
            support_status=support_status,
            assumptions=(
                "individual_random_assignment",
                "all_event_causes_coded_distinctly",
                *observation_assumptions,
                "consistency",
                "no_interference",
            ),
        )
    return SurvivalEstimate(
        times=times,
        control_survival=control,
        treated_survival=treated,
        rmst_control=float(section.rmst_control),
        rmst_treated=float(section.rmst_treated),
        rmst_difference=float(section.rmst_treated - section.rmst_control),
        tau=section.tau,
        uncertainty=section.uncertainty,
        rmst_difference_interval=(
            tuple(section.rmst_difference_interval)
            if section.rmst_difference_interval is not None
            else None
        ),
        survival_at_tau_difference_interval=(
            tuple(section.difference_at_tau_interval)
            if section.difference_at_tau_interval is not None
            else None
        ),
        bootstrap_replicates_requested=section.bootstrap_replicates_requested,
        bootstrap_replicates_ok=section.bootstrap_replicates_ok,
        censoring_survival_provenance=provenance,
        difference_band=difference_band,
        band_unavailable_reason=section.band_unavailable_reason,
        support_status=support_status,
        assumptions=(
            "individual_random_assignment",
            *observation_assumptions,
            "consistency",
            "no_interference",
        ),
    )


def _longitudinal_regime_from_raw(raw: Any) -> LongitudinalRegimeEstimate | None:
    section = getattr(raw, "longitudinal_regime", None)
    if section is None:
        return None
    return LongitudinalRegimeEstimate(
        value=section.value,
        value_standard_error=section.value_standard_error,
        value_interval_95=section.value_interval_95,
        period_intervals_95=(
            tuple(tuple(bounds) for bounds in section.period_intervals_95)
            if section.period_intervals_95 is not None
            else None
        ),
        interval_reason=section.interval_reason,
        support_status=(
            "licensed"
            if section.graphless_support_status == "licensed"
            else "off_axis_pointwise_95"
            if section.value_interval_95 is not None
            else "unlicensed_point_utility"
        ),
        effective_sample_size=section.effective_sample_size,
        matched_observed_fraction=section.matched_observed_fraction,
        maximum_weight=section.maximum_weight,
        minimum_action_probability=section.minimum_action_probability,
        minimum_censoring_probability=section.minimum_censoring_probability,
        method=section.method,
        uncertainty=section.uncertainty,
        probability_ownership=section.probability_ownership,
        period_effects=tuple(section.period_effects)
        if section.period_effects is not None
        else None,
        standard_errors=tuple(section.standard_errors)
        if section.standard_errors is not None
        else None,
        stabilizing_numerator_probabilities=tuple(section.stabilizing_numerator_probabilities)
        if section.stabilizing_numerator_probabilities is not None
        else None,
        observed_subjects=section.observed_subjects,
        rule_id=section.rule_id,
        rule_version=section.rule_version,
        rule_provenance=section.rule_provenance,
    )


def _continuous_dose_from_raw(raw: Any) -> ConditionalDoseResponseEstimate | None:
    section = getattr(raw, "continuous_dose_response", None)
    if section is None:
        return None
    policy = getattr(section, "fixed_policy", None)
    fixed_policy = (
        None
        if policy is None
        else FixedDosePolicyValueEstimate(
            policy_doses=tuple((str(group), float(dose)) for group, dose in policy.policy_doses),
            reference_doses=tuple(
                (str(group), float(dose)) for group, dose in policy.reference_doses
            ),
            policy_value=float(policy.policy_value),
            reference_value=float(policy.reference_value),
            incremental_value=float(policy.incremental_value),
            policy_variance=float(policy.policy_variance),
            reference_variance=float(policy.reference_variance),
            incremental_variance=float(policy.incremental_variance),
            policy_interval_95=tuple(policy.policy_interval_95)
            if policy.policy_interval_95 is not None
            else None,
            reference_interval_95=tuple(policy.reference_interval_95)
            if policy.reference_interval_95 is not None
            else None,
            incremental_interval_95=tuple(policy.incremental_interval_95)
            if policy.incremental_interval_95 is not None
            else None,
            minimum_local_rows=int(policy.minimum_local_rows),
            minimum_effective_sample_size=float(policy.minimum_effective_sample_size),
            maximum_normalized_weight=float(policy.maximum_normalized_weight),
            minimum_dose_density=float(policy.minimum_dose_density),
        )
    )
    return ConditionalDoseResponseEstimate(
        points=tuple(
            ConditionalDoseResponsePoint(
                str(group),
                float(target),
                float(response),
                int(local_rows),
                float(effective_n),
                float(min_density),
                float(max_weight),
                float(local_sd),
            )
            for group, target, response, local_rows, effective_n, min_density, max_weight, local_sd in section.points
        ),
        bandwidth=float(section.bandwidth),
        density_provenance=str(section.density_provenance),
        uncertainty=str(section.uncertainty),
        policy_value_estimated=fixed_policy is not None,
        fixed_policy=fixed_policy,
        support_status=(
            section.graphless_support_status
            or unset_dose_license(
                has_incremental_interval=fixed_policy is not None
                and fixed_policy.incremental_interval_95 is not None
            )
        ),
        evaluation_method=(
            "fixed_group_kernel_smoothed_inverse_density_policy"
            if fixed_policy is not None
            else "stratified_triangular_kernel_inverse_density"
        ),
    )


def _policy_value_from_raw(raw: Any) -> DoublyRobustPolicyEvaluation | None:
    section = getattr(raw, "policy_value", None)
    if section is None:
        return None
    ipw = section.prediction_ownership == "no_outcome_nuisance_predictions"
    multi_action = section.uncertainty.startswith("multi_action_")
    from .policy import FiniteClassRegretEvaluation, MultiActionCatePoint, UpliftBin

    return DoublyRobustPolicyEvaluation(
        policy_value=section.policy_value,
        reference_value=section.reference_value,
        incremental_value=section.incremental_value,
        relative_value_gap=section.relative_value_gap,
        treatment_rate=section.treatment_rate,
        total_treatment_cost=section.total_cost,
        policy_value_standard_error=section.policy_standard_error,
        reference_value_standard_error=section.reference_standard_error,
        incremental_value_standard_error=section.incremental_standard_error,
        policy_value_interval_95=section.policy_interval_95,
        incremental_value_interval_95=section.incremental_interval_95,
        support_status=(
            section.graphless_support_status
            or unset_policy_license(
                has_regret=section.regret is not None,
                has_interval=section.policy_interval_95 is not None
                or any(interval is not None for _, _, _, _, interval in section.uplift_bins)
                or any(interval is not None for *_, interval in section.multi_action_cate),
            )
        ),
        prediction_ownership=section.prediction_ownership,
        propensity_min=section.propensity_min,
        propensity_max=section.propensity_max,
        uplift_bins=tuple(
            UpliftBin(int(rank), float(effect), float(se), int(rows), interval)
            for rank, effect, se, rows, interval in section.uplift_bins
        ),
        multi_action_cate=tuple(
            MultiActionCatePoint(
                group,
                action,
                float(effect),
                int(rows),
                int(action_rows),
                int(control_rows),
                float(se),
                interval,
                "pointwise_95" if interval is not None else "point_only",
            )
            for group, action, effect, rows, action_rows, control_rows, se, interval in section.multi_action_cate
        ),
        finite_class_regret=(
            FiniteClassRegretEvaluation(
                float(section.regret[0]),
                tuple(section.regret[1]),
                tuple(float(value) for value in section.regret[2]),
                tuple(float(se) for se in section.regret[3]),
                int(section.regret[4]),
            )
            if section.regret is not None
            else None
        ),
        uncertainty=section.uncertainty,
        evaluation_method=(
            "randomized_multi_action_ipw_fixed_policy"
            if multi_action
            else "randomized_ipw_fixed_policy"
            if ipw
            else "doubly_robust_randomized_heldout_or_cross_fitted"
        ),
        assumptions=(
            (
                (
                    "Known randomized action probabilities with positive support for each declared action.",
                    "Consistency and no interference between evaluation subjects.",
                    "The multi-action policy and reference were fixed without using evaluation outcomes.",
                    "The first action label is the control action for treatment-rate reporting.",
                )
                if multi_action
                else (
                    "Known randomized treatment propensities with strict treatment overlap.",
                    "Consistency and no interference between evaluation subjects.",
                    "The policy and reference were fixed without using evaluation outcomes.",
                )
                if ipw
                else (
                    "Known randomized treatment propensities with strict treatment overlap.",
                    "The supplied randomization propensities are correct; nuisance outcome predictions may be misspecified.",
                    "Consistency and no interference between evaluation subjects.",
                    "Policy recommendations and both outcome nuisance predictions were generated without using the corresponding evaluation subject's outcome.",
                )
            )
            + (
                (
                    "Rank scores were fitted on declared training subjects disjoint from evaluation subjects and frozen before outcome evaluation.",
                )
                if section.uplift_bins
                else ()
            )
            + (
                (
                    "Baseline groups for conditional effects were fixed before evaluation outcomes were observed.",
                )
                if section.multi_action_cate
                else ()
            )
            + (
                (
                    "The finite candidate class and selected member were fixed using declared training subjects disjoint from evaluation subjects.",
                )
                if section.regret is not None
                else ()
            )
        ),
        diagnostics=(
            (
                (
                    "row-level standard errors assume independent evaluation subjects",
                    "randomization and policy selection claims are caller-declared",
                    "action availability, capacities, budgets, and positive probability rows were checked",
                )
                if ipw
                else (
                    "row-level standard errors assume independent evaluation subjects",
                    "training/test disjointness or excluded-fold correspondence is checked from caller-supplied IDs only",
                    "nuisance predictions and randomization claims are not independently authenticated",
                )
            )
            + (
                (
                    "uplift-bin pointwise 95% intervals require at least 300 independent evaluation subjects and 50 observed treated and control subjects per bin",
                    "ranking-model ownership is checked from caller-supplied subject IDs only",
                )
                if section.uplift_bins
                else ()
            )
            + (
                (
                    "multi-action conditional intervals require at least 300 independent evaluation subjects in each fixed baseline group, 50 observed subjects per compared arm, and each compared action probability at least 0.2",
                )
                if section.multi_action_cate
                else ()
            )
            + (
                (
                    "finite-class regret uses simultaneous paired-score bounds for 2–16 fixed candidates; training ownership and randomization are caller-declared",
                )
                if section.regret is not None
                else ()
            )
        ),
    )


def _wrap_ate(
    raw: AteAnalysisResult | TemporalAnalysisResult,
    prepared: Any | None = None,
    *,
    query: Any = None,
) -> AnalysisResult:
    """Build the nested :class:`AnalysisResult` view from either native DTO.

    ``AteAnalysisResult`` (static) and ``AnalysisResult`` (temporal, aliased here
    as ``TemporalAnalysisResult``) both expose the same five nested sections —
    ``identification`` / ``estimate`` / ``posterior`` / ``validation`` /
    ``performance`` (see ``antecedent._native`` and the doc comments there) —
    built and kept in sync with their flat-field siblings on the Rust side. This
    one function reads those sections instead of the ~30 hand-written
    ``getattr(raw, "field_name", default)`` calls per DTO shape that used to live
    in two near-duplicate functions (``_wrap_ate`` / ``_wrap_temporal``). Fields
    the temporal DTO genuinely cannot supply already come through their section
    as ``None`` (see each section's doc comment in ``_native.pyi``), so no
    per-DTO branching is needed here beyond `getattr` presence gates for the
    handful of fields that only ever exist on the static DTO (predictive checks,
    prior sensitivity, external-prior conflict, mediation) — those `getattr`
    calls resolve to ``None`` on the temporal DTO exactly as the old
    `_wrap_temporal` left them.
    """
    execution = None if prepared is None else prepared._native.snapshot()
    slots = (
        None if execution is None else ReasoningSlots.from_contract(execution.execution_contract())
    )

    def _conflict_from_raw(r: Any) -> ConflictSummaryView | None:
        ids = getattr(r, "conflict_source_ids", None)
        if ids is None:
            return None
        return ConflictSummaryView(
            source_ids=list(ids),
            alphas_requested=list(getattr(r, "conflict_alphas_requested", None) or []),
            alphas_applied=list(getattr(r, "conflict_alphas_applied", None) or []),
        )

    sec_identification = raw.identification
    sec_estimate = raw.estimate
    sec_posterior = raw.posterior
    sec_validation = raw.validation
    sec_performance = raw.performance

    mediation = None
    if (
        getattr(raw, "mediation_total", None) is not None
        or getattr(raw, "mediation_mediated", None) is not None
    ):
        mediation = MediationView(
            total=getattr(raw, "mediation_total", None),
            direct=getattr(raw, "mediation_direct", None),
            mediated=getattr(raw, "mediation_mediated", None),
        )

    posterior = None
    if sec_posterior.n_draws is not None:
        mass = sec_posterior.unidentified_mass
        skipped = float(getattr(sec_posterior, "subsampled_out_mass", None) or 0.0)
        envelope = None
        if (mass is not None and float(mass) > 0.0) or skipped > 0.0:
            envelope = EffectEnvelope(
                effect_mean=sec_posterior.effect_mean,
                effect_sd=sec_posterior.effect_sd,
                q025=sec_posterior.q025,
                q975=sec_posterior.q975,
                unidentified_mass=float(mass or 0.0),
                n_draws=sec_posterior.n_draws,
                backend=sec_posterior.backend,
                subsampled_out_mass=skipped,
            )
        posterior_estimator_id = str(getattr(sec_estimate, "estimator_id", "") or "")
        posterior_interval_type = (
            "modular_bootstrap_pushforward"
            if posterior_estimator_id == "bayesian.robust_ate"
            else "equal_tailed_95"
        )
        posterior = PosteriorView(
            effect_mean=sec_posterior.effect_mean,
            effect_sd=sec_posterior.effect_sd,
            q025=sec_posterior.q025,
            q975=sec_posterior.q975,
            n_draws=sec_posterior.n_draws,
            p_below_zero=sec_posterior.p_below_zero,
            backend=sec_posterior.backend,
            interval_type=posterior_interval_type,
            artifact=sec_posterior.artifact,
            unidentified_mass=None if mass is None else float(mass),
            envelope=envelope,
            conflict=_conflict_from_raw(raw),
            subsampled_out_mass=skipped,
        )

    # The predictive-check fields are static-DTO only; the temporal DTO does not
    # declare them, so every read goes through `getattr` rather than asserting a
    # union member that may not have the attribute at all.
    def _ppc(prefix: str, kind: str) -> PredictiveCheckReport | None:
        p_value = getattr(raw, f"{prefix}_p_value", None)
        if p_value is None:
            return None
        observed = getattr(raw, f"{prefix}_observed", None)
        predictive_mean = getattr(raw, f"{prefix}_predictive_mean", None)
        predictive_sd = getattr(raw, f"{prefix}_predictive_sd", None)
        n_sims = getattr(raw, f"{prefix}_n_sims", None)
        if observed is None or predictive_mean is None:
            return None
        if predictive_sd is None or n_sims is None:
            return None
        return PredictiveCheckReport(
            kind=kind,
            observed=float(observed),
            predictive_mean=float(predictive_mean),
            predictive_sd=float(predictive_sd),
            p_value=float(p_value),
            n_sims=int(n_sims),
        )

    prior_predictive = _ppc("prior_ppc", "prior_predictive")
    posterior_predictive = _ppc("posterior_ppc", "posterior_predictive")
    prior_sensitivity = None
    means = getattr(raw, "prior_sensitivity_means", None)
    if means is not None:
        alphas_raw = getattr(raw, "prior_sensitivity_alphas", None)
        scales_raw = getattr(raw, "prior_sensitivity_scales", None)
        sds = getattr(raw, "prior_sensitivity_sds", None)
        multipliers_raw = getattr(raw, "prior_sensitivity_variance_multipliers", None)
        family = getattr(raw, "prior_sensitivity_family", None)
        prior_sensitivity = PriorSensitivityReport(
            scales=list(scales_raw or ()),
            effect_means=list(means),
            effect_sds=list(sds or ()),
            alphas=None if alphas_raw is None else list(alphas_raw),
            variance_multipliers=None if multipliers_raw is None else list(multipliers_raw),
            family=str(family) if family is not None else "isotropic_scale",
        )
    certificate_json = getattr(raw, "certificate_json", None)
    mediation_grid = None
    horizons = list(getattr(raw, "mediation_horizons", None) or ())
    if horizons:
        temporal_raw = cast(TemporalAnalysisResult, raw)
        effects = list(temporal_raw.mediation_effects)
        totals = list(temporal_raw.mediation_totals)
        directs = list(temporal_raw.mediation_directs)
        mediated_effects = list(temporal_raw.mediation_mediated_effects)
        statuses = list(temporal_raw.mediation_identification_statuses)
        methods = list(temporal_raw.mediation_methods)
        adjustments = list(temporal_raw.mediation_adjustments)
        uncertainty_kinds = list(temporal_raw.mediation_uncertainty_kinds)
        standard_deviations = list(temporal_raw.mediation_standard_deviations)
        q025 = list(temporal_raw.mediation_q025)
        q975 = list(temporal_raw.mediation_q975)
        identified_lower = list(temporal_raw.mediation_identified_lower)
        identified_upper = list(temporal_raw.mediation_identified_upper)
        slices = tuple(
            TemporalMediationSliceView(
                horizon=int(values[0]),
                effect=float(values[1]),
                total=float(values[2]),
                direct=float(values[3]),
                mediated=float(values[4]),
                identification_status=str(values[5]),
                method=str(values[6]),
                adjustment=tuple((int(variable), int(offset)) for variable, offset in values[7]),
                uncertainty_kind=str(values[8]),
                standard_deviation=None if values[9] is None else float(values[9]),
                q025=None if values[10] is None else float(values[10]),
                q975=None if values[11] is None else float(values[11]),
                identified_lower=None if values[12] is None else float(values[12]),
                identified_upper=None if values[13] is None else float(values[13]),
            )
            for values in zip(
                horizons,
                effects,
                totals,
                directs,
                mediated_effects,
                statuses,
                methods,
                adjustments,
                uncertainty_kinds,
                standard_deviations,
                q025,
                q975,
                identified_lower,
                identified_upper,
                strict=True,
            )
        )
        mediation_grid = TemporalMediationGridView(
            slices=slices,
            joint_posterior=bool(getattr(raw, "mediation_joint_posterior", False)),
        )
    # ResultModel accepts private retained handles outside its public Pydantic fields.
    return AnalysisResult(  # type: ignore[call-arg]
        certificate=json.loads(certificate_json) if certificate_json else None,
        query=query if query is not None else getattr(prepared, "_query", None),
        identification=IdentificationView(
            status=sec_identification.status,
            method=sec_identification.method,
            adjustment_set=list(sec_identification.adjustment_set),
            assumption_count=sec_identification.assumption_count,
            derivation_step_count=sec_identification.derivation_step_count,
        ),
        estimate=EstimateView(
            ate=_optional_finite_ate(getattr(sec_estimate, "ate", None)),
            se_analytic=sec_estimate.se_analytic,
            se_bootstrap=sec_estimate.se_bootstrap,
            estimator_id=sec_estimate.estimator_id,
            method=sec_estimate.method,
            overlap_ess=sec_estimate.overlap_ess,
            overlap_propensity_min=sec_estimate.overlap_propensity_min,
            mediation=mediation,
            functional_means=tuple(functional_means)
            if (functional_means := getattr(sec_estimate, "functional_means", None)) is not None
            else None,
            exceedance_cdf=tuple(exceedance_cdf)
            if (exceedance_cdf := getattr(sec_estimate, "exceedance_cdf", None)) is not None
            else None,
            monotone_rearranged=bool(getattr(sec_estimate, "monotone_rearranged", False)),
            interaction_structurally_zero=getattr(
                sec_estimate, "interaction_structurally_zero", None
            ),
            unit_effects_homogeneous=getattr(sec_estimate, "unit_effects_homogeneous", None),
            score_table=getattr(sec_estimate, "score_table", None),
            joint_covariance=getattr(sec_estimate, "joint_covariance", None),
            score_inference=getattr(sec_estimate, "score_inference", None),
            scenario_effects=getattr(sec_estimate, "scenario_effects", None),
            scenario_intervals=getattr(sec_estimate, "scenario_intervals", None),
            simultaneous_interval=getattr(sec_estimate, "simultaneous_interval", None),
            adjusted_p_values=getattr(sec_estimate, "adjusted_p_values", None),
            family_contrast=getattr(sec_estimate, "family_contrast", None),
            family_contrast_interval=getattr(sec_estimate, "family_contrast_interval", None),
            candidate_selection=getattr(sec_estimate, "candidate_selection", None),
            evalue=getattr(sec_estimate, "evalue", None),
            evalue_threshold=getattr(sec_estimate, "evalue_threshold", None),
            distribution=_distribution_atoms_from_raw(sec_estimate),
            mean_interval=_probability_interval_from_raw(
                getattr(sec_estimate, "mean_interval", None)
            ),
            outcome_oof_r2=getattr(sec_estimate, "outcome_oof_r2", None),
            treatment_oof_logloss=getattr(sec_estimate, "treatment_oof_logloss", None),
            crossfit_folds=getattr(sec_estimate, "crossfit_folds", None),
            crossfit_seed=getattr(sec_estimate, "crossfit_seed", None),
            learner_provenance=tuple(getattr(sec_estimate, "learner_provenance", ())),
            penalized_fallback=_penalized_fallback_from_raw(
                getattr(sec_estimate, "penalized_fallback", None)
            ),
            penalized_support=tuple(
                PenalizedFoldSupport(fold=int(fold), names=tuple(str(n) for n in names))
                for fold, names in getattr(sec_estimate, "penalized_support", ())
            ),
            penalized_bootstrap=_penalized_bootstrap_from_raw(
                getattr(sec_estimate, "penalized_bootstrap", None)
            ),
            cate=tuple(cate) if (cate := getattr(sec_estimate, "cate", None)) is not None else None,
            cate_se=tuple(cate_se)
            if (cate_se := getattr(sec_estimate, "cate_se", None)) is not None
            else None,
            cate_leaf_dispersion=tuple(cate_leaf_dispersion)
            if (cate_leaf_dispersion := getattr(sec_estimate, "cate_leaf_dispersion", None))
            is not None
            else None,
        ),
        posterior=posterior,
        unit_effects=getattr(raw, "unit_effects", None),
        unit_effect_intervals=_unit_effect_intervals_from_raw(raw),
        unit_effect_intervals_level=getattr(raw, "unit_effect_intervals_level", None),
        unit_effect_intervals_method=getattr(raw, "unit_effect_intervals_method", None),
        unit_extrapolative=getattr(raw, "unit_extrapolative", None),
        assumptions=getattr(raw, "assumptions", None),
        support=getattr(raw, "support_diagnostics", None),
        mediation=mediation,
        mediation_grid=mediation_grid,
        validation=ValidationView(
            passed=sec_validation.passed,
            ran=sec_validation.ran,
            count=sec_validation.count,
            prior_predictive=prior_predictive,
            posterior_predictive=posterior_predictive,
            prior_sensitivity=prior_sensitivity,
            reports=_refutation_reports_from_raw(sec_validation),
            computation_failures=list(getattr(sec_validation, "computation_failures", [])),
        ),
        performance=PerformanceView(
            plan_id=sec_performance.plan_id,
            modality=sec_performance.modality,
            peak_memory_bytes=sec_performance.peak_memory_bytes,
            latency_mode=sec_performance.latency_mode,
            wall_time_ns=sec_performance.wall_time_ns,
            bootstrap_replicates_requested=sec_performance.bootstrap_replicates_requested,
            bootstrap_replicates_ok=sec_performance.bootstrap_replicates_ok,
            n_draws=sec_performance.n_draws,
            cancelled=bool(sec_performance.cancelled),
            early_stopped=bool(sec_performance.early_stopped),
            stage_timings={str(k): int(v) for k, v in (sec_performance.stage_timings or [])}
            or None,
            bytes_borrowed=getattr(sec_performance, "bytes_borrowed", None),
        ),
        diagnostics=list(raw.diagnostics),
        provenance={
            "node_count": raw.provenance_node_count,
            "worker_threads": getattr(raw, "worker_threads", None),
            "expected_python_crossings": getattr(raw, "expected_python_crossings", None),
        },
        plan=_plan_from_raw(raw),
        evidence_status=getattr(raw, "evidence_status", None),
        allowlist_reason=getattr(raw, "allowlist_reason", None),
        allowlist_parent=getattr(raw, "allowlist_parent", None),
        structural_weight_basis=getattr(raw, "structural_weight_basis", None),
        structural_identified_mass=getattr(raw, "structural_identified_mass", None),
        structural_unidentified_mass=getattr(raw, "structural_unidentified_mass", None),
        structural_unevaluable_mass=getattr(raw, "structural_unevaluable_mass", None),
        structural_identified_set=getattr(raw, "structural_identified_set", None),
        structural_identified_set_interval=getattr(raw, "structural_identified_set_interval", None),
        structural_identified_set_interval_level=getattr(
            raw, "structural_identified_set_interval_level", None
        ),
        structural_identified_set_interval_method=getattr(
            raw, "structural_identified_set_interval_method", None
        ),
        structural_identified_set_interval_truncated=getattr(
            raw, "structural_identified_set_interval_truncated", None
        ),
        transport_overlap=_transport_overlap_from_raw(raw),
        interference=_interference_from_raw(raw),
        randomized_effect=_randomized_effect_from_raw(raw),
        panel_did=_panel_did_from_raw(raw, query),
        synthetic_control=_synthetic_control_from_raw(raw),
        synthetic_did=_synthetic_did_from_raw(raw),
        local_polynomial_ratio=_local_polynomial_ratio_from_raw(raw, query),
        policy_value=_policy_value_from_raw(raw),
        continuous_dose_response=_continuous_dose_from_raw(raw),
        survival=_survival_from_raw(raw, query),
        longitudinal_regime=_longitudinal_regime_from_raw(raw),
        anomaly=getattr(raw, "anomaly", None),
        change_attribution=getattr(raw, "change_attribution", None),
        _raw=raw,
        _prepared=prepared,
        _execution=execution,
        reasoning=slots,
        program_id=None if slots is None else slots.program_id,
        claim_id=None if slots is None else slots.claim_id,
        data_snapshot_id=None if slots is None else slots.data_snapshot_id,
    )


def _static_edges(
    graph: Dag | Cpdag | Sequence[tuple[str, str]] | None,
) -> list[tuple[str, str]]:
    if graph is None:
        raise CausalValueError("graph= is required")
    if isinstance(graph, Dag):
        return [(str(a), str(b)) for a, b in graph.edges()]
    if isinstance(graph, (Pag, TemporalDag, TemporalCpdag, TemporalPag, Admg)):
        raise CausalValueError(
            f"this query reads a fully oriented static Dag; got {type(graph).__name__}"
        )
    if isinstance(graph, Cpdag):
        # Compatibility normalization for fixed-DAG query families: a fully
        # oriented CPDAG has a unique DAG and reports that Dag coordinate.
        # Incomplete CPDAGs refuse; this does not license class execution.
        return cpdag_oriented_edges(graph, require_oriented=True)
    return [(str(a), str(b)) for a, b in graph]


_RESPONSE_FAMILY = (
    ResponseCurve,
    InterventionResponse,
    AverageDerivative,
    PointDerivative,
    Elasticity,
    SemiElasticity,
    DirectionalDerivative,
    ResponseJacobian,
)


def _bayesian_inference_kwargs(inference: Bayesian) -> dict[str, Any]:
    backend = str(inference.backend).strip().lower()
    if backend == "laplace":
        inference_s = "bayesian"
    elif backend == "conjugate":
        inference_s = "conjugate"
    elif backend == "hmc":
        inference_s = "hmc"
    else:
        raise CausalValueError(
            f"unknown Bayesian backend {inference.backend!r}; use laplace|conjugate|hmc"
        )
    likelihood = str(inference.likelihood).strip().lower()
    if likelihood not in ("gaussian", "logit", "probit", "poisson"):
        raise CausalValueError(
            f"unknown Bayesian likelihood {inference.likelihood!r}; "
            "use gaussian|logit|probit|poisson",
            reason_code="invalid_argument",
        )
    kw: dict[str, Any] = {
        "inference": inference_s,
        "prior_scale": inference.prior_scale,
        "likelihood": likelihood,
    }
    if inference.n_draws_explicit:
        kw["n_draws"] = inference.n_draws
    prior_from = inference.prior_from
    if prior_from is not None:
        # Local import avoids circular import with priors ↔ estimation.
        from .priors import ComposedPrior

        if isinstance(prior_from, ComposedPrior):
            kw["composed_prior"] = prior_from.to_native_dict()
        else:
            kw["prior_artifact"] = bytes(prior_from)
    if inference.mapping is not None:
        kw["prior_mapping"] = inference.mapping.to_dict()
    return kw


def analyze_many(
    data: Mapping[str, Any] | Any,
    *,
    graph: Dag | TieredBackground | Sequence[tuple[str, str]],
    queries: Sequence[AverageEffect],
    identifier: str | None = None,
    estimator: str | Estimator | Any | None = None,
    estimator_config: Mapping[str, Any] | None = None,
    refute: bool | Literal["full", "placebo", "none", "cheap"] | None = None,
    seed: int = 1,
    bootstrap: int | None = None,
    threads: int | None = None,
    latency: Literal["interactive", "standard", "report"] | None = None,
    candidate_screen: CandidateScreen | None = None,
) -> list[AnalysisResult]:
    """Estimate many average effects on one shared table ingest.

    Queries that would fit the identical cross-fitted AIPW nuisance (same
    treatment coding, adjustment set, complete-case rows, fold plan and
    learner options) share one fit; each result is bit-identical to the
    query's own ``analyze`` fit. The ``batch.shared_design`` diagnostic
    records, per result, whether its propensity / outcome fit was shared.

    Only ``AverageEffect`` queries batch here; discrete joint
    ``InterventionResponse`` cells batch through
    :meth:`PreparedBatch.prepare_cells`. Any other query is refused with
    ``reason_code="route_not_supported"`` and a ``remedy`` naming
    :func:`antecedent.analyze`, which estimates it one query at a time.

    Parameters
    ----------
    data:
        Column mapping / DataFrame (ingested once).
    graph:
        Static DAG, edge list, or ``TieredBackground`` shared by every query.
    queries:
        Non-empty sequence of ``AverageEffect`` queries. Outcome functionals
        are executed, not dropped to means.
    refute:
        ``False`` or a suite name; leave unset (``None``) for the default
        suite. Explicit ``refute=True`` raises ``TypeError`` — see
        :func:`antecedent._coerce.coerce_refute`.
    estimator, estimator_config:
        A bare estimator id (``str`` or :class:`antecedent.Estimator`) or a typed estimator
        configuration such as ``Aipw(bootstrap=0, propensity_penalty=...)``, exactly as
        :func:`antecedent.analyze` takes them; one configuration governs every query. A
        nuisance fit is shared only between queries whose fit inputs (rows, treatment coding,
        adjustment design, folds, learner options) are identical, and a penalized propensity
        is never shared. Each result's ``batch.shared_design`` diagnostic carries the
        canonical ``estimator_config`` fingerprint.
    candidate_screen:
        Optional screen/estimate split recorded on every result.
    """
    if not queries:
        raise CausalValueError("analyze_many requires at least one query")
    if not all(isinstance(q, AverageEffect) for q in queries):
        raise _batch_route_refusal("analyze_many", "AverageEffect queries")
    estimator, estimator_config = _unwrap_estimator(estimator, estimator_config)
    resolved_refute = None if refute is None else coerce_refute(refute)
    names, columns = ingest_columns(data)
    from .query import coerce_outcome_functional

    specs = [
        (
            q.treatment,
            q.outcome,
            float(q.control_level),
            float(q.active_level),
            coerce_outcome_functional(q.outcome_functional),
            coerce_target_population(q.target_population),
        )
        for q in queries
    ]
    kwargs: dict[str, Any] = dict(
        identifier=identifier,
        estimator=estimator,
        estimator_config=estimator_config,
        refute=resolved_refute,
        seed=seed,
        bootstrap=bootstrap,
        threads=threads,
    )
    if latency is not None:
        kwargs["latency"] = latency
    kwargs.update(_screen_kwargs(candidate_screen))
    if isinstance(graph, TieredBackground):
        raws = _analyze_ate_many(
            names,
            columns,
            [],
            specs,
            tiers=[list(tier) for tier in graph.tiers],
            within_tier=str(graph.within_tier),
            **kwargs,
        )
    else:
        raws = _analyze_ate_many(names, columns, _static_edges(graph), specs, **kwargs)
    return [_wrap_ate(r, query=q) for r, q in zip(raws, queries, strict=True)]


_BATCH_ROUTE_REMEDY = (
    "estimate it with antecedent.analyze(data, graph=..., query=...), one query at a "
    "time; the batch routes cover AverageEffect (analyze_many, PreparedBatch.prepare) "
    "and discrete joint InterventionResponse cells (PreparedBatch.prepare_cells)"
)


def _batch_route_refusal(entry: str, covered: str) -> CausalUnsupportedError:
    """Typed refusal for a query outside a batch route, naming ``analyze``."""
    return _refused_with_remedy(
        "route_not_supported",
        f"{entry} covers {covered} only; use analyze for any other query",
        _BATCH_ROUTE_REMEDY,
    )


@dataclass(frozen=True)
class CandidateScreen:
    """Declared screen/estimate split for a batch family."""

    screen_id: str
    procedure: Literal["max_t", "bh", "by", "unrecorded"]
    screen_rows: Sequence[int]
    estimate_rows: Sequence[int]

    @classmethod
    def from_units(
        cls,
        screen_id: str,
        procedure: Literal["max_t", "bh", "by", "unrecorded"],
        *,
        seed: int,
        screen_fraction: float = 0.5,
        entity_ids: Sequence[int] | None = None,
        first: Sequence[int] | None = None,
        second: Sequence[int] | None = None,
    ) -> CandidateScreen:
        """Deterministic split of whole entities, clusters or dyad components.

        Declare ownership with ``entity_ids`` (one entity or cluster label per data row) or
        with ``first`` and ``second`` (the two endpoint labels of each dyad row, in one label
        namespace). Dyads that share an endpoint, directly or through a chain of rows, form a
        connected component and always land in the same half, so no endpoint appears in both
        halves; a single component is refused. The split is a seeded function of the set of
        units, not of row order. ``screen_id`` is extended with the seed and a digest of the
        sorted unit ids (``"<id>;seed=<hex>;units=<digest>"``), so every recorded selection
        names them.
        """
        from ._native import candidate_screen_from_units

        recorded_id, screen_rows, estimate_rows, *_receipt = candidate_screen_from_units(
            screen_id,
            procedure,
            screen_fraction=float(screen_fraction),
            seed=int(seed),
            entity_ids=None if entity_ids is None else [int(i) for i in entity_ids],
            first=None if first is None else [int(i) for i in first],
            second=None if second is None else [int(i) for i in second],
        )
        return cls(recorded_id, procedure, screen_rows, estimate_rows)


def _screen_kwargs(screen: CandidateScreen | None) -> dict[str, Any]:
    if screen is None:
        return {}
    return {
        "screen_id": screen.screen_id,
        "screen_procedure": screen.procedure,
        "screen_rows": [int(i) for i in screen.screen_rows],
        "estimate_rows": [int(i) for i in screen.estimate_rows],
    }


def _joint_cell_batch_specs(
    queries: Sequence[InterventionResponse],
) -> list[tuple[str, list[str], list[str], list[list[float]], dict[str, Any] | None]]:
    from . import intervention as intervention_specs
    from .query import coerce_outcome_functional

    specs: list[tuple[str, list[str], list[str], list[list[float]], dict[str, Any] | None]] = []
    for query in queries:
        if getattr(query, "is_temporal", False):
            raise _batch_route_refusal(
                "PreparedBatch.prepare_cells", "static discrete joint InterventionResponse cells"
            )
        supplied = query.intervention
        interventions = (
            list(supplied)
            if isinstance(supplied, Sequence) and not isinstance(supplied, (str, bytes))
            else [supplied]
        )
        if len(interventions) < 2:
            raise _batch_route_refusal(
                "PreparedBatch.prepare_cells", "joint InterventionResponse cells (two or more Set)"
            )
        treatments: list[str] = []
        kinds: list[str] = []
        parameters: list[list[float]] = []
        for spec in interventions:
            if not isinstance(spec, intervention_specs.Set):
                raise _batch_route_refusal(
                    "PreparedBatch.prepare_cells", "discrete joint cells of Set interventions"
                )
            treatments.append(spec.variable)
            kinds.append("set")
            parameters.append([spec.value])
        specs.append(
            (
                query.outcome,
                treatments,
                kinds,
                parameters,
                coerce_outcome_functional(query.outcome_functional),
            )
        )
    return specs


@dataclass(frozen=True)
class SharedBatchDesign:
    """Fold assignment and covariate design frozen on a prepared batch.

    Folds and, when adjustment sets agree, the ``[1 | Z]`` matrix are shared.
    The frozen design holds no nuisance fit, so ``shares_propensity`` and
    ``shares_outcome_residualization`` stay ``False`` here. Each
    :meth:`PreparedBatch.estimate` shares identical cross-fitted AIPW
    propensity / outcome fits across its queries and records that per result
    in the ``batch.shared_design`` diagnostic.
    """

    n_folds: int
    fold_ids: tuple[int, ...]
    adjustment_set: tuple[int, ...] | None
    shares_covariates: bool
    shares_propensity: bool = False
    shares_outcome_residualization: bool = False


@dataclass(frozen=True)
class RetargetClaim:
    """One declared retargeted claim of a :meth:`PreparedBatch.retarget` family.

    ``query`` is the index of a batch query or the query object itself. ``weights`` are the
    target row weights aligned with :meth:`PreparedBatch.retarget_rows`; ``depends_on`` names
    the covariates the weights are a function of (a subset of the certified adjustment set).
    """

    name: str
    query: int | AverageEffect | InterventionResponse
    weights: Any
    depends_on: Sequence[str] = ()


@dataclass(frozen=True)
class RetargetContrast:
    """A named linear contrast of declared claims, ``sum_k c_k * claim_k``.

    ``route A minus route B`` is ``RetargetContrast("a_minus_b", {"a": 1.0, "b": -1.0})``.
    """

    name: str
    coefficients: Mapping[str, float]


@dataclass(frozen=True)
class RetargetMember:
    """One claim or contrast of a :class:`BatchRetarget`, failed members included."""

    kind: Literal["claim", "contrast"]
    name: str
    estimand: str
    status: Literal["ok", "point_only", "failed"]
    value: float | None
    std_error: float | None
    uncertainty_kind: str
    query_index: int | None = None
    n_eff: float | None = None
    support_status: str = "not_assessed"
    refusal_code: str | None = None
    refusal_detail: str | None = None
    refusal_message: str | None = None
    # Structured refusal diagnostics of a failed member (stage, the claim or contrast as
    # subject, reason, and for a failed overlap gate the per-arm effective sample size and the
    # propensity range); ``None`` for a member that did not fail, absent entries ``None``.
    refusal_fields: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class SimultaneousBandMember:
    """One claim of a :class:`SimultaneousInterval`: ``value +/- critical_value * std_error``."""

    name: str
    value: float
    std_error: float
    lower: float
    upper: float


@dataclass(frozen=True)
class SimultaneousInterval:
    """Family-level max-t band of a complete :class:`BatchRetarget` family.

    ``critical_value`` is the ``level`` quantile of ``max_j |Z_j|`` for ``Z ~ N(0, R)``, ``R``
    the correlation matrix of the family's plug-in score covariance: a Monte-Carlo value that
    is a deterministic function of ``(R, level, seed, draws)``. The band is a nominal
    asymptotic construction under iid rows, fixed declared weights, positivity and nuisance
    convergence; selection and weight-estimation uncertainty and the Monte-Carlo error of
    ``critical_value`` are excluded.
    """

    level: float
    critical_value: float
    seed: int
    draws: int
    members: tuple[SimultaneousBandMember, ...]


@dataclass(frozen=True)
class BatchRetarget:
    """Points, joint score covariance and contrasts of a retargeted family.

    The covariance is the plug-in joint covariance of the retargeted claims from the shared
    row-aligned scores; a standard error is not an interval. For a complete family
    :meth:`simultaneous_interval` returns the max-t band ``value +/- c * std_error`` of the
    claims (level, seed and draws are the ``simultaneous_*`` arguments of
    :meth:`PreparedBatch.retarget`); a partial family (any failed or point-only member) lists
    every member, has no band and is never a complete-family claim (:meth:`complete_family`
    and :meth:`simultaneous_interval` raise). Penalized-propensity scores retarget to points
    only.
    """

    family_id: str
    snapshot_id: str | None
    scores_source: Literal["prepared", "estimated"]
    estimator_fingerprint: str
    claims: tuple[RetargetMember, ...]
    contrasts: tuple[RetargetMember, ...]
    covariance_names: tuple[str, ...]
    covariance: tuple[tuple[float, ...], ...] | None
    complete: bool
    failed_members: tuple[str, ...]
    point_only_members: tuple[str, ...]
    inference_claim: str
    scope_note: str
    _closed: Mapping[str, Any]
    _complete_refusal: Mapping[str, Any] | None
    _rows: tuple[Mapping[str, Any], ...]

    def covariance_between(self, a: str, b: str) -> float:
        """Entry of the joint covariance for two named claims."""
        if self.covariance is None or a not in self.covariance_names:
            raise CausalValueError(f"claim {a!r} carries no covariance in this family")
        if b not in self.covariance_names:
            raise CausalValueError(f"claim {b!r} carries no covariance in this family")
        return self.covariance[self.covariance_names.index(a)][self.covariance_names.index(b)]

    def simultaneous_interval(self) -> SimultaneousInterval:
        """The family-level max-t band; a typed refusal for a partial or point-only family."""
        raw = self._closed
        if raw["status"] != "published":
            raise _refused(raw["reason_code"], f"{raw['detail']}: {raw['message']}")
        return SimultaneousInterval(
            level=float(raw["level"]),
            critical_value=float(raw["critical_value"]),
            seed=int(raw["seed"]),
            draws=int(raw["draws"]),
            members=tuple(
                SimultaneousBandMember(
                    name=m["name"],
                    value=float(m["value"]),
                    std_error=float(m["std_error"]),
                    lower=float(m["lower"]),
                    upper=float(m["upper"]),
                )
                for m in raw["members"]
            ),
        )

    def complete_family(self) -> BatchRetarget:
        """This report when it is a complete-family claim; otherwise a typed refusal."""
        refusal = self._complete_refusal
        if refusal is not None:
            raise _refused(refusal["reason_code"], f"{refusal['detail']}: {refusal['message']}")
        return self

    def to_rows(self) -> list[dict[str, Any]]:
        """Tidy export: one row per claim then contrast, failed members included, each with
        its family identity, completeness, support and refusal status, uncertainty kind,
        diagnostics and provenance."""
        return [dict(row) for row in self._rows]


def _retarget_member(raw: Mapping[str, Any], kind: str) -> RetargetMember:
    failure = raw.get("failure") or {}
    return RetargetMember(
        kind=kind,  # type: ignore[arg-type]
        name=raw["name"],
        estimand=raw.get("estimand") or " ".join(f"{c:+}*{n}" for n, c in raw.get("terms", [])),
        status=raw["status"],
        value=raw.get("value"),
        std_error=raw.get("std_error"),
        uncertainty_kind=raw.get("uncertainty_kind", "none"),
        query_index=raw.get("query_index"),
        n_eff=raw.get("n_eff"),
        support_status=(
            "refused"
            if failure.get("support_refused")
            else ("not_assessed" if failure else "supported")
        ),
        refusal_code=failure.get("reason_code"),
        refusal_detail=failure.get("detail"),
        refusal_message=failure.get("message"),
        refusal_fields=failure.get("refusal_fields"),
    )


def _retarget_estimand(query: Any) -> str | None:
    """Readable estimand of an average-effect query (joint cells keep the native label)."""
    if not isinstance(query, AverageEffect):
        return None
    t, y = query.treatment, query.outcome
    return f"E[{y}(do {t}={query.active_level})] - E[{y}(do {t}={query.control_level})]"


@dataclass
class PreparedBatch:
    """Compile-once batch of average-effect or joint-cell plans.

    ``prepare`` and ``prepare_cells`` freeze one fold-assignment object and,
    when every query shares a certified adjustment set, one covariate design.
    Average-effect queries that would fit the identical cross-fitted AIPW
    propensity or outcome model share one fit (bit-identical to the per-query
    fit); joint-cell nuisances are fit per query. Other query types are
    refused with ``reason_code="route_not_supported"`` and a ``remedy``
    naming :func:`antecedent.analyze`.
    A family of two or more average-effect claims attaches joint IF covariance
    and max-t / BH / BY on those contrasts. Joint-cell families keep
    ``simultaneous_interval`` on cell levels and, unless ``family_contrast``
    is ``None``, test a declared score-difference contrast (default
    ``cell_minus_control``) with max-t and ``family_contrast_interval`` on
    that contrast family.
    """

    _native: Any
    _queries: tuple[AverageEffect | InterventionResponse, ...]
    _names: tuple[str, ...]

    @property
    def shared_design(self) -> SharedBatchDesign | None:
        n_folds = self._native.shared_n_folds()
        if n_folds is None:
            return None
        folds = self._native.shared_fold_ids() or []
        adj = self._native.shared_adjustment_set()
        return SharedBatchDesign(
            n_folds=int(n_folds),
            fold_ids=tuple(int(i) for i in folds),
            adjustment_set=None if adj is None else tuple(int(i) for i in adj),
            shares_covariates=bool(self._native.shares_covariates()),
        )

    def diagnose(self, *, seed: int = 1, threads: int | None = None) -> BatchPreflightReport:
        """Fit-free preflight of every plan (see ``PreparedAnalysis.diagnose``)."""
        from .preflight import diagnose_batch

        return diagnose_batch(self._native, seed=seed, threads=threads)

    def diagnose_fit(
        self, *, seed: int = 1, threads: int | None = None
    ) -> tuple[NuisanceFitDiagnostics, ...]:
        """Propensity-fit diagnostics of every plan (fit-requiring)."""
        from .preflight import diagnose_fit_batch

        return diagnose_fit_batch(self._native, seed=seed, threads=threads)

    def plan_rank_drop(
        self,
        priority: Sequence[str] | None = None,
        *,
        seed: int = 1,
        threads: int | None = None,
    ) -> tuple[RankDropPlan, ...]:
        """Opt-in rank-drop plan of every plan under one declared column priority."""
        from .preflight import plan_rank_drop_batch

        return plan_rank_drop_batch(self._native, priority, seed=seed, threads=threads)

    def estimate_cost(self) -> BatchCostEstimate:
        """Planning counts of every plan and their totals; the fit total is an upper bound
        because identical cross-fitted nuisances are shared. A planning hint, not a runtime
        guarantee."""
        from .preflight import estimate_cost_batch

        return estimate_cost_batch(self._native)

    @classmethod
    def prepare(
        cls,
        data: Mapping[str, Any] | Any,
        *,
        graph: Dag | TieredBackground | Sequence[tuple[str, str]],
        queries: Sequence[AverageEffect],
        identifier: str | None = None,
        estimator: str | Estimator | Any | None = None,
        estimator_config: Mapping[str, Any] | None = None,
        refute: bool | Literal["full", "placebo", "none", "cheap"] | None = None,
        seed: int = 1,
        bootstrap: int | None = None,
        threads: int | None = None,
        latency: Literal["interactive", "standard", "report"] | None = None,
        candidate_screen: CandidateScreen | None = None,
    ) -> PreparedBatch:
        if not queries:
            raise CausalValueError("PreparedBatch.prepare requires at least one query")
        if not all(isinstance(q, AverageEffect) for q in queries):
            raise _batch_route_refusal("PreparedBatch.prepare", "AverageEffect queries")
        estimator, estimator_config = _unwrap_estimator(estimator, estimator_config)
        resolved_refute: bool | str | None = None if refute is None else coerce_refute(refute)
        names, columns = ingest_columns(data)
        from .query import coerce_outcome_functional

        specs = [
            (
                q.treatment,
                q.outcome,
                float(q.control_level),
                float(q.active_level),
                coerce_outcome_functional(q.outcome_functional),
                coerce_target_population(q.target_population),
            )
            for q in queries
        ]
        kwargs: dict[str, Any] = dict(
            identifier=identifier,
            estimator=estimator,
            estimator_config=estimator_config,
            refute=resolved_refute,
            seed=seed,
            bootstrap=bootstrap,
            threads=threads,
        )
        if latency is not None:
            kwargs["latency"] = latency
        kwargs.update(_screen_kwargs(candidate_screen))
        if isinstance(graph, TieredBackground):
            native = _prepare_ate_batch(
                names,
                columns,
                [],
                specs,
                tiers=[list(tier) for tier in graph.tiers],
                within_tier=str(graph.within_tier),
                **kwargs,
            )
        else:
            native = _prepare_ate_batch(names, columns, _static_edges(graph), specs, **kwargs)
        return cls(_native=native, _queries=tuple(queries), _names=tuple(names))

    @classmethod
    def prepare_cells(
        cls,
        data: Mapping[str, Any] | Any,
        *,
        graph: Dag | TieredBackground | Sequence[tuple[str, str]],
        queries: Sequence[InterventionResponse],
        identifier: str | None = None,
        estimator: str | Estimator | Any | None = None,
        estimator_config: Mapping[str, Any] | None = None,
        refute: bool | Literal["full", "placebo", "none", "cheap"] | None = None,
        seed: int = 1,
        bootstrap: int | None = None,
        threads: int | None = None,
        latency: Literal["interactive", "standard", "report"] | None = None,
        candidate_screen: CandidateScreen | None = None,
        family_contrast: Literal["cell_minus_control", "interaction"] | None = "cell_minus_control",
    ) -> PreparedBatch:
        """Compile discrete joint ``InterventionResponse`` cells into one batch.

        Licensed on a DAG and on CoDetermined ``TieredBackground``. Unknown-tier
        joint has no single ADMG and refuses. Pair families share folds and, when
        adjustment sets agree, the covariate design; estimation attaches joint
        IF covariance on cell **levels**. Family p-values / FDR use
        ``family_contrast`` (default ``cell_minus_control``, the same contrast
        as cell.aipw refuters). Max-t and ``family_contrast_interval`` are
        formed on that contrast family. ``family_contrast=None`` publishes no
        p-values. ``estimate.ate`` remains the requested cell level, not a
        contrast.
        """
        if not queries:
            raise CausalValueError("PreparedBatch.prepare_cells requires at least one query")
        if not all(isinstance(q, InterventionResponse) for q in queries):
            raise _batch_route_refusal(
                "PreparedBatch.prepare_cells", "discrete joint InterventionResponse cells"
            )
        estimator, estimator_config = _unwrap_estimator(estimator, estimator_config)
        if isinstance(graph, TieredBackground):
            if identifier not in (None, "generalized.adjustment"):
                raise CausalUnsupportedError(
                    "CoDetermined joint cells require identifier generalized.adjustment"
                )
            if estimator not in (None, "cell.aipw"):
                raise CausalUnsupportedError("CoDetermined joint cells require estimator cell.aipw")
        resolved_refute: bool | str | None = None if refute is None else coerce_refute(refute)
        specs = _joint_cell_batch_specs(queries)
        names, columns = ingest_columns(data)
        kwargs: dict[str, Any] = dict(
            identifier=None if isinstance(graph, TieredBackground) else identifier,
            estimator=estimator,
            estimator_config=estimator_config,
            refute=resolved_refute,
            seed=seed,
            bootstrap=bootstrap,
            threads=threads,
            family_contrast=family_contrast,
        )
        if latency is not None:
            kwargs["latency"] = latency
        kwargs.update(_screen_kwargs(candidate_screen))
        if isinstance(graph, TieredBackground):
            native = _prepare_cells_batch(
                names,
                columns,
                [],
                specs,
                tiers=[list(tier) for tier in graph.tiers],
                within_tier=str(graph.within_tier),
                **kwargs,
            )
        else:
            native = _prepare_cells_batch(names, columns, _static_edges(graph), specs, **kwargs)
        return cls(_native=native, _queries=tuple(queries), _names=tuple(names))

    def estimate(
        self,
        data: Mapping[str, Any] | Any,
        *,
        seed: int = 1,
        threads: int | None = None,
    ) -> list[AnalysisResult]:
        """Re-estimate every plan on ``data``.

        The estimate's score tables are retained on this handle, so a later
        :meth:`retarget` reweights this estimate's rows (see :meth:`retarget_rows`) rather
        than the prepare-time rows; a plan whose estimate kept no table refuses its members
        instead of silently losing the retarget.
        """
        names, columns = ingest_columns(data)
        raws = self._native.estimate(names, columns, seed=seed, threads=threads)
        return [_wrap_ate(r, query=q) for r, q in zip(raws, self._queries, strict=True)]

    def retarget_rows(self) -> tuple[int, ...] | None:
        """Original-row index the weights of every claim must align with.

        These are the rows of the scores a retarget now reads: the last :meth:`estimate`'s
        (retained on this handle), else the prepare-time rows. ``None`` when no plan holds a
        score table. Plans built on different rows raise
        ``reason_code="row_weights_bound_to_snapshot"``.
        """
        rows = self._native.retarget_rows()
        return None if rows is None else tuple(int(r) for r in rows)

    def retarget_snapshot(self) -> str | None:
        """Identity of the row snapshot the scores a retarget now reads share."""
        return self._native.retarget_snapshot()

    @describe_refusal
    def retarget(
        self,
        claims: Sequence[RetargetClaim],
        contrasts: Sequence[RetargetContrast] = (),
        *,
        expected_snapshot: str | None = None,
        simultaneous_level: float = 0.95,
        simultaneous_seed: int = 0,
        simultaneous_draws: int = 100_000,
    ) -> BatchRetarget:
        """Retarget a declared family of claims over one row snapshot. No refit.

        Each :class:`RetargetClaim` reweights one plan's cross-fitted score table (an
        AllObserved iid AIPW or cell-AIPW plan) under row weights aligned with
        :meth:`retarget_rows`. The result carries every point, the joint plug-in score
        covariance (cross-claim terms come from the shared rows), the named
        :class:`RetargetContrast` values with their standard errors, the failed members, and
        for a complete family the max-t simultaneous band at ``simultaneous_level`` (the
        Monte-Carlo critical value uses ``simultaneous_seed`` and ``simultaneous_draws``,
        1000 to 2000000). After :meth:`estimate` the scores are that
        estimate's; a plan whose estimate kept no score table fails its members with
        ``reason_code="score_table_unavailable"`` rather than reusing prepare-time rows.
        Mixed snapshots, an undeclared ``expected_snapshot`` and malformed families raise
        typed refusals; a member that cannot be retargeted is reported, never dropped.
        """
        import numpy as np

        native_claims = []
        labels: dict[str, str] = {}
        for claim in claims:
            if isinstance(claim.query, int) and not isinstance(claim.query, bool):
                index = claim.query
            else:
                matches = [i for i, q in enumerate(self._queries) if q == claim.query]
                if not matches:
                    raise CausalValueError(
                        f"claim {claim.name!r} names a query that is not in this batch"
                    )
                index = matches[0]
            if 0 <= index < len(self._queries):
                label = _retarget_estimand(self._queries[index])
                if label is not None:
                    labels[claim.name] = label
            native_claims.append(
                (
                    claim.name,
                    index,
                    np.asarray(claim.weights, dtype=float).tolist(),
                    [str(n) for n in claim.depends_on],
                )
            )
        native_contrasts = [
            (c.name, [(str(n), float(v)) for n, v in c.coefficients.items()]) for c in contrasts
        ]
        raw = self._native.retarget_family(
            native_claims,
            native_contrasts,
            expected_snapshot=expected_snapshot,
            simultaneous_level=float(simultaneous_level),
            simultaneous_seed=int(simultaneous_seed),
            simultaneous_draws=int(simultaneous_draws),
        )
        members = []
        for entry in raw["claims"]:
            if entry["name"] in labels:
                entry["estimand"] = labels[entry["name"]]
            members.append(_retarget_member(entry, "claim"))
        rows = []
        for row in raw["rows"]:
            if row["kind"] == "claim" and row["name"] in labels:
                row["estimand"] = labels[row["name"]]
            rows.append(row)
        cov = raw["covariance"]
        return BatchRetarget(
            family_id=raw["family_id"],
            snapshot_id=raw["snapshot_id"],
            scores_source=raw["scores_source"],
            estimator_fingerprint=raw["estimator_fingerprint"],
            claims=tuple(members),
            contrasts=tuple(_retarget_member(c, "contrast") for c in raw["contrasts"]),
            covariance_names=tuple(cov["names"]) if cov else (),
            covariance=tuple(tuple(float(v) for v in row) for row in cov["matrix"])
            if cov
            else None,
            complete=bool(raw["complete"]),
            failed_members=tuple(raw["failed_members"]),
            point_only_members=tuple(raw["point_only_members"]),
            inference_claim=raw["inference_claim"],
            scope_note=raw["scope_note"],
            _closed=raw["simultaneous_interval"],
            _complete_refusal=raw["complete_refusal"],
            _rows=tuple(rows),
        )


@dataclass(frozen=True)
class IdentifyResult:
    """Identify-only result (no estimate), retaining its certified query."""

    status: str
    method: str
    adjustment_set: list[str]
    query: Any = None


def identify(
    *,
    graph: Dag | Admg | Sequence[tuple[str, str]],
    query: AverageEffect,
    names: Sequence[str] | None = None,
    identifier: str | Identifier | None = None,
) -> IdentifyResult:
    """Identify without estimating.

    Accepts a ``Dag``, an ``Admg``, or an edge list. Pass ``names`` with an edge
    list (variable order); with a typed graph the names come from
    ``graph.nodes()``.

    ``identify(TieredBackground)`` stays Rust-only
    (``identify_tiered`` / ``identify_tiered_joint``). Prepared identification
    already returns the certificate; this entry does not accept a tier rule.
    Pair-family joint cells use :meth:`PreparedBatch.prepare_cells`.

    Prefer an ``Admg`` whenever a confounder is unmeasured. A ``Dag`` has no way
    to say a variable cannot be observed, so a latent common cause flattened
    into one is treated as an ordinary adjustable node and the effect is
    reported identified by adjusting on something no study can measure.
    ``Dag.latent_project(observed)`` produces the ``Admg`` for that graph.
    """
    if isinstance(identifier, Identifier):
        identifier = str(identifier)
    if isinstance(graph, Admg):
        status, method, adjustment = _identify_ate_admg(
            list(graph.nodes()),
            graph,
            query.treatment,
            query.outcome,
            identifier=identifier,
        )
        return IdentifyResult(
            status=status, method=method, adjustment_set=list(adjustment), query=query
        )
    if isinstance(graph, Dag):
        node_names = list(graph.nodes())
        edges = list(graph.edges())
    else:
        if names is None:
            raise CausalValueError("identify(edge_list) requires names=")
        node_names = list(names)
        edges = list(graph)
    status, method, adjustment = _identify_ate(
        node_names,
        edges,
        query.treatment,
        query.outcome,
        identifier=identifier,
    )
    return IdentifyResult(
        status=status, method=method, adjustment_set=list(adjustment), query=query
    )


def _response_support_bounds(raw: Any) -> dict[str, tuple[float, float]]:
    """Map native support minima/maxima onto axis names.

    Static curves align one interval per treatment name. Temporal dose × horizon
    surfaces report a multi-axis query region wider than the treatment list.
    """
    minima = list(getattr(raw, "support_minima", ()) or ())
    maxima = list(getattr(raw, "support_maxima", ()) or ())
    treatments = list(getattr(raw, "treatments", ()) or ())
    if len(minima) == len(treatments):
        return {
            name: (lower, upper)
            for name, lower, upper in zip(treatments, minima, maxima, strict=True)
        }
    axis_names = (
        ["dose", "horizon"] if len(minima) == 2 else [f"axis_{i}" for i in range(len(minima))]
    )
    return {
        name: (lower, upper) for name, lower, upper in zip(axis_names, minima, maxima, strict=True)
    }


def _horizon_adjustment_sets(raw: Any) -> tuple[tuple[str, ...], ...] | None:
    sets = tuple(
        tuple(str(name) for name in group)
        for group in (getattr(raw, "horizon_adjustment_sets", ()) or ())
    )
    return sets or None


def _support_point_status(raw: Any) -> tuple[SupportStatus, ...] | None:
    """Native empty vec means a static curve with no per-cell support grid."""
    cells = tuple(getattr(raw, "support_point_status", ()) or ())
    return tuple(cast(SupportStatus, status) for status in cells) if cells else None


def _wrap_prepared_response(
    raw: Any,
    query: ResponseCurve | InterventionResponse | None = None,
    prepared: Any | None = None,
) -> CausalResponseView:
    """Build a :class:`CausalResponseView` from a prepared-response native DTO."""
    execution = None if prepared is None else prepared._native.snapshot()
    slots = (
        None if execution is None else ReasoningSlots.from_contract(execution.execution_contract())
    )
    from typing import cast

    response = (
        ResponseView(
            treatments=raw.treatments, outcomes=raw.outcomes, points=raw.points, values=raw.values
        )
        if raw.points and raw.values
        else None
    )
    temporal = bool(getattr(query, "is_temporal", False))
    identifier = getattr(raw, "identifier", None)
    validation: ResponseValidationView | None
    if identifier == "generalized.adjustment":
        method = "generalized.adjustment"
        identify_op = "identify.generalized_adjustment"
        validation = None
    elif temporal:
        method = "temporal.backdoor.unfolded"
        identify_op = "identify.temporal_backdoor"
        validation = ResponseValidationView(
            checks=(
                ResponseValidationCheck(
                    id="refute.temporal_response.skipped",
                    status="skipped",
                    statistic=None,
                    threshold=None,
                    detail="scalar ATE refuters are not applicable to a function-valued temporal response",
                ),
            )
        )
    else:
        method = "response.backdoor"
        identify_op = "identify.response"
        validation = None
    certificate_json = getattr(raw, "certificate_json", None)
    envelope = None
    if getattr(raw, "identified_mass", None) is not None:
        if raw.lower is None or raw.upper is None:
            raise RuntimeError("native structural response omitted its identified envelope")
        envelope = ResponseEnvelopeView(
            treatments=raw.treatments,
            outcomes=raw.outcomes,
            points=raw.points,
            lower=raw.lower,
            upper=raw.upper,
            identified_mass=float(raw.identified_mass),
            unidentified_mass=float(raw.unidentified_mass),
            completion_count=int(raw.completion_count),
            truncated_completions=int(raw.truncated_completions),
            enumeration_capped=bool(raw.enumeration_capped),
            mass_scope=cast(Literal["full_class", "examined_completions"], raw.mass_scope),
            weight_basis=cast(
                Literal[
                    "posterior_probability",
                    "completion_enumeration",
                    "caller_supplied_class_prior",
                ],
                raw.weight_basis,
            ),
            atom_keys=tuple(raw.atom_keys),
            atom_weights=tuple(raw.atom_weights),
            atom_statuses=tuple(raw.atom_statuses),
            atom_values=tuple(tuple(values) for values in raw.atom_values),
            unevaluable_mass=float(getattr(raw, "unevaluable_mass", None) or 0.0),
            subsampled_out_mass=float(getattr(raw, "subsampled_out_mass", None) or 0.0),
        )
    basis_json = raw.program_basis_json()
    native_basis = {} if basis_json is None else json.loads(basis_json)
    # ResultModel accepts private retained handles outside its public Pydantic fields.
    return CausalResponseView(  # type: ignore[call-arg]
        certificate=json.loads(certificate_json) if certificate_json else None,
        estimand=query,
        response=response,
        estimate=raw.scalar if raw.scalar is not None else raw.matrix,
        uncertainty=ResponseUncertainty(
            kind=cast(UncertaintyKind, raw.uncertainty_kind),
            lower=raw.lower,
            upper=raw.upper,
            level=raw.level,
            standard_error=raw.standard_error,
            replicates=raw.replicates,
            artifact_id=raw.artifact_id,
            interpretation=cast(
                IntervalInterpretation | None, getattr(raw, "interval_interpretation", None)
            ),
        ),
        support=SupportReport(
            status=cast(SupportStatus, raw.support_status),
            query_region=_response_support_bounds(raw),
            diagnostics=[
                SupportDiagnostic(id=identifier, values=values, detail=detail, scope=scope)
                for identifier, values, detail, scope in zip(
                    raw.diagnostic_ids,
                    raw.diagnostic_values,
                    raw.diagnostic_details,
                    raw.diagnostic_scopes,
                    strict=True,
                )
            ],
            warnings=raw.warnings,
            point_status=_support_point_status(raw),
        ),
        identification=IdentificationView(
            status=raw.identification,
            method=method,
            adjustment_set=list(getattr(raw, "adjustment_set", ())),
            assumption_count=len(raw.assumptions),
            derivation_step_count=0,
            horizon_adjustment_sets=_horizon_adjustment_sets(raw),
        ),
        assumptions=raw.assumptions,
        provenance={
            "operation_id": raw.provenance_id,
            "operation_ids": [identify_op, raw.provenance_id],
        },
        envelope=envelope,
        validation=validation,
        evidence_status=getattr(raw, "evidence_status", None),
        allowlist_reason=getattr(raw, "allowlist_reason", None),
        allowlist_parent=getattr(raw, "allowlist_parent", None),
        diagnostics=tuple(getattr(raw, "diagnostics", ()) or ()),
        _prepared=prepared,
        _execution=execution,
        _raw=raw,
        reasoning=slots,
        program_id=native_basis.get("program_id") if slots is None else slots.program_id,
        claim_id=None if slots is None else slots.claim_id,
        data_snapshot_id=native_basis.get("snapshot_id")
        if slots is None
        else slots.data_snapshot_id,
    )


def _prepared_columns(data: Any) -> tuple[list[str], list[Any], bool]:
    arrow = try_as_arrow_c_columns(data)
    if arrow is not None:
        names, columns = arrow
        return names, columns, True
    names, columns = as_columns(data)
    return names, columns, False


_PreparedQuery = (
    AverageEffect
    | ResponseCurve
    | ConditionalEffect
    | PathSpecificEffect
    | InterventionalDistribution
    | InterventionResponse
    | PulseEffect
    | SustainedEffect
    | MediationEffect
    | NestedCounterfactual
    | Counterfactual
    | PointDerivative
    | Elasticity
    | SemiElasticity
    | AverageDerivative
    | DirectionalDerivative
    | ResponseJacobian
    | TemporalMediationEffect
    | TransportQuery
    | Transport
    | ExactTransportQuery
    | StatisticalTransportQuery
    | TransportResponseGridQuery
    | InterferenceQuery
    | RandomizedEffect
    | ComplierEffect
    | TreatmentOnTreated
    | SwitchbackEffect
    | PolicyValue
    | ConditionalDoseResponse
    | MultiActionPolicyValue
    | AnomalyAttribution
    | ChangeAttribution
    | PanelDifferenceInDifferences
    | AugmentedPanelDiD
    | StaggeredAdoption
    | SyntheticControl
    | SyntheticDifferenceInDifferences
    | FuzzyRegressionDiscontinuity
    | RegressionKink
    | SharpRegressionDiscontinuity
    | SurvivalOutcome
    | CompetingRisksOutcome
    | LongitudinalRegime
)


@dataclass(frozen=True)
class _Controls:
    """Execution controls a study applies to every click unless overridden.

    In-process only: none of them is part of the contract, program, or claim.
    """

    cancel: Any | None = None
    on_progress: Any | None = None
    on_stage: Any | None = None

    def kwargs(self) -> dict[str, Any]:
        return {
            name: value
            for name, value in (
                ("cancel", self.cancel),
                ("on_progress", self.on_progress),
                ("on_stage", self.on_stage),
            )
            if value is not None
        }


_UNSET: Any = object()


def _refused(reason_code: str, message: str) -> CausalUnsupportedError:
    return CausalUnsupportedError(message, reason_code=reason_code)


def _refused_with_remedy(reason_code: str, message: str, remedy: str) -> CausalUnsupportedError:
    return CausalUnsupportedError(message, reason_code=reason_code, remedy=remedy)


def _not_applicable(option: str, route: str) -> CausalUnsupportedError:
    return _refused("option_not_applicable", f"{option} does not apply to {route}")


def _unwrap_estimator(
    estimator: Any, estimator_config: Mapping[str, Any] | None
) -> tuple[str | None, Mapping[str, Any] | None]:
    """Split a typed estimator config into its id and wire configuration."""
    if estimator is None or isinstance(estimator, str):
        return estimator, estimator_config
    if isinstance(estimator, Estimator):
        return str(estimator), estimator_config
    if estimator_config is not None:
        raise ValueError(
            "estimator= already carries its configuration; do not also pass estimator_config="
        )
    return estimator.estimator_id, estimator._wire()


def _frame_payload(data: Any) -> tuple[list[str], list[Any], dict[str, Any] | None]:
    """Columns plus the panel / multi-environment / event frame the natives rebuild."""
    from .data import EventFrame, MultiEnvFrame, PanelFrame

    if isinstance(data, EventFrame):
        return (
            list(data.names),
            list(data.columns),
            {
                "kind": "events",
                "event_times_ns": [int(t) for t in data.event_times_ns],
                "align_interval_ns": int(data.align_interval_ns),
            },
        )
    if isinstance(data, PanelFrame):
        return (
            list(data.names),
            [],
            {
                "kind": "panel",
                "unit_columns": [list(cols) for cols in data.unit_columns],
                "unit_ids": [int(i) for i in data.unit_ids],
            },
        )
    if isinstance(data, MultiEnvFrame):
        return (
            list(data.names),
            [],
            {"kind": "multi_env", "env_columns": [list(cols) for cols in data.env_columns]},
        )
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes, Mapping)):
        # A sequence of tables is multi-environment data (the J-PCMCI+ spelling).
        from ._data import as_multi_env_columns

        names, env_columns = as_multi_env_columns(list(data))
        return names, [], {"kind": "multi_env", "env_columns": env_columns}
    names, columns = ingest_columns(data)
    return names, columns, None


def _panel_did_payload(
    data: Any, query: PanelDifferenceInDifferences
) -> tuple[list[str], list[Any], dict[str, Any]]:
    """Keep panel IDs as query metadata and send only numeric outcomes to Rust."""
    from .quasi import _binary, _raw_columns

    raw_names, raw_columns = _raw_columns(data)
    raw = dict(zip(raw_names, raw_columns, strict=True))
    required: tuple[str, ...] = (query.outcome, query.subject, query.treated, query.post)
    if query.cluster is not None:
        required += (query.cluster,)
    missing = [name for name in required if name not in raw]
    if missing:
        raise CausalValueError(f"required panel DiD columns are missing: {missing}")
    subjects = tuple(str(value) for value in raw[query.subject])
    clusters = (
        subjects if query.cluster is None else tuple(str(value) for value in raw[query.cluster])
    )
    if any(not value.strip() for value in (*subjects, *clusters)):
        raise CausalValueError("subject and cluster IDs must be non-empty")
    design = {
        "subjects": subjects,
        "clusters": clusters,
        "treated": tuple(_binary(raw[query.treated], query.treated)),
        "post": tuple(_binary(raw[query.post], query.post)),
        "repeated_cross_section": query.sampling == "repeated_cross_section",
    }
    names, columns = ingest_columns({query.outcome: raw[query.outcome]})
    return names, columns, design


def _augmented_panel_did_payload(
    data: Any, query: AugmentedPanelDiD
) -> tuple[list[str], list[Any], dict[str, tuple[Any, ...]]]:
    """Bind subject and cluster ownership while keeping outcome/nuisance columns numeric."""
    from .quasi import _binary, _raw_columns

    raw_names, raw_columns = _raw_columns(data)
    raw = dict(zip(raw_names, raw_columns, strict=True))
    numeric = (
        query.outcome_pre,
        query.outcome_post,
        query.propensity,
        query.untreated_change_prediction,
    )
    required: tuple[str, ...] = (*numeric, query.subject, query.treated)
    if query.cluster is not None:
        required += (query.cluster,)
    missing = [name for name in required if name not in raw]
    if missing:
        raise CausalValueError(f"required augmented panel DiD columns are missing: {missing}")
    subjects = tuple(raw[query.subject])
    clusters = subjects if query.cluster is None else tuple(raw[query.cluster])
    if any(not isinstance(value, str) or not value.strip() for value in (*subjects, *clusters)):
        raise CausalValueError("subject and cluster IDs must be non-empty strings")
    if len(set(subjects)) != len(subjects):
        raise CausalValueError("augmented panel DiD requires one unique row per subject")
    design = {
        "subjects": subjects,
        "clusters": clusters,
        "treated": tuple(_binary(raw[query.treated], query.treated)),
    }
    names, columns = ingest_columns({name: raw[name] for name in numeric})
    return names, columns, design


def _staggered_payload(
    data: Any, query: StaggeredAdoption
) -> tuple[list[str], list[Any], dict[str, Any]]:
    from .quasi import _integer_column, _raw_columns

    if not query.event_study and (query.target_cohort is None or query.target_period is None):
        raise CausalValueError(
            "analyze with StaggeredAdoption requires target_cohort and target_period"
        )
    raw_names, raw_columns = _raw_columns(data)
    raw = dict(zip(raw_names, raw_columns, strict=True))
    required: tuple[str, ...] = (query.outcome, query.subject, query.period, query.cohort)
    if query.cluster is not None:
        required += (query.cluster,)
    missing = [name for name in required if name not in raw]
    if missing:
        raise CausalValueError(f"required staggered DiD columns are missing: {missing}")
    subjects = tuple(str(value) for value in raw[query.subject])
    clusters = (
        subjects if query.cluster is None else tuple(str(value) for value in raw[query.cluster])
    )
    if any(not value.strip() for value in (*subjects, *clusters)):
        raise CausalValueError("subject and cluster IDs must be non-empty")
    periods = tuple(_integer_column(raw[query.period], "period", minimum=1))
    cohorts = tuple(_integer_column(raw[query.cohort], "cohort", minimum=0))
    design = {
        "subjects": subjects,
        "clusters": clusters,
        "periods": periods,
        "cohorts": cohorts,
        "target_cohort": query.target_cohort,
        "target_period": query.target_period,
    }
    names, columns = ingest_columns({query.outcome: raw[query.outcome]})
    return names, columns, design


def _synthetic_control_payload(
    data: Any, query: SyntheticControl | SyntheticDifferenceInDifferences
) -> tuple[list[str], list[Any], dict[str, Any]]:
    from .quasi import _integer_column, _raw_columns

    raw_names, raw_columns = _raw_columns(data)
    raw = dict(zip(raw_names, raw_columns, strict=True))
    required = (query.outcome, query.unit, query.period)
    missing = [name for name in required if name not in raw]
    if missing:
        raise CausalValueError(f"required synthetic-control columns are missing: {missing}")
    units = tuple(str(value) for value in raw[query.unit])
    if any(not value.strip() for value in units):
        raise CausalValueError("unit IDs must be non-empty")
    periods = tuple(_integer_column(raw[query.period], "period", minimum=1))
    names, columns = ingest_columns({query.outcome: raw[query.outcome]})
    return names, columns, {"units": units, "periods": periods}


def _continuous_dose_payload(
    data: Any, query: ConditionalDoseResponse
) -> tuple[list[str], list[Any], dict[str, Any]]:
    from .quasi import _raw_columns

    raw_names, raw_columns = _raw_columns(data)
    raw = dict(zip(raw_names, raw_columns, strict=True))
    required = (query.outcome, query.dose, query.baseline_group, query.dose_density)
    missing = [name for name in required if name not in raw]
    if missing:
        raise CausalValueError(f"required continuous-dose columns are missing: {missing}")
    groups = tuple(raw[query.baseline_group])
    if any(not isinstance(group, str) or not group for group in groups):
        raise CausalValueError("baseline group labels must be non-empty strings")
    names, columns = ingest_columns(
        {name: raw[name] for name in (query.outcome, query.dose, query.dose_density)}
    )
    return names, columns, {"baseline_groups": groups}


def _longitudinal_regime_payload(
    data: Any, query: LongitudinalRegime
) -> tuple[list[str], list[Any]]:
    """Send the numeric endpoint to Rust; histories remain frozen query metadata."""
    from .quasi import _raw_columns

    raw_names, raw_columns = _raw_columns(data)
    if query.outcome not in raw_names:
        raise CausalValueError(f"longitudinal outcome column {query.outcome!r} is missing")
    outcome = raw_columns[raw_names.index(query.outcome)]
    if len(outcome) != len(query.subject_ids):
        raise CausalValueError("longitudinal outcome rows must align with subject histories")
    if query.method in ("sequential_dr", "marginal_structural_model"):
        values = np.asarray(outcome, dtype=np.float64)
        observed = np.asarray(query.outcome_observed, dtype=bool)
        if not np.isfinite(values[observed]).all():
            raise CausalValueError("observed terminal outcomes must be finite")
        outcome = np.where(observed, values, 0.0)
    return ingest_columns({query.outcome: outcome})


def _inference_wire(inference: Frequentist | Bayesian) -> dict[str, Any]:
    if isinstance(inference, Frequentist):
        return {"mode": "frequentist", "prior_scale": 10.0}
    kw = _bayesian_inference_kwargs(inference)
    if kw.get("prior_mapping") is not None and kw.get("prior_artifact") is None:
        raise CausalUnsupportedError(
            "prior_mapping names which estimand of a prior artifact to read; it has nothing "
            "to map without prior_from=<artifact>",
            reason_code="option_not_applicable",
        )
    return {
        "mode": kw.pop("inference"),
        "n_draws": kw.get("n_draws"),
        "prior_scale": kw["prior_scale"],
        "prior_artifact": kw.get("prior_artifact"),
        "prior_mapping": kw.get("prior_mapping"),
        "composed_prior": kw.get("composed_prior"),
        "likelihood": kw["likelihood"],
    }


def _accept_live_discovery(
    data: Any,
    discovery: Any,
    *,
    accept_discovered: bool,
    regimes: Sequence[int] | None,
    seed: int,
    threads: int | None,
    controls: _Controls,
) -> Any:
    """Run a live discovery config once and accept it through its review gate.

    Cancellation is checked at the discovery boundary and progress reports the
    discovery stage; estimation on the accepted structure honours both fully.
    """
    from .accepted_graph import AcceptedGraph

    token = controls.cancel
    if token is not None and token.is_cancelled():
        from .errors import CausalCancelledError

        raise CausalCancelledError("cancelled before discovery")
    if controls.on_progress is not None:
        controls.on_progress(0.0, "discovery")
    from .discovery import RPCMCI

    # Only RPCMCI labels regimes; prepare refuses `regimes=` for every other config.
    labelled = {"regimes": regimes} if isinstance(discovery, RPCMCI) else {}
    accepted = discovery.accept(
        data,
        seed=seed,
        threads=threads,
        accept_discovered=accept_discovered,
        **labelled,
    )
    if controls.on_progress is not None:
        controls.on_progress(1.0, "discovery")
    if token is not None and token.is_cancelled():
        from .errors import CausalCancelledError

        raise CausalCancelledError("cancelled after discovery")
    return accepted if isinstance(accepted, AcceptedGraph) else AcceptedGraph(accepted)


_GRAPH_POSTERIOR_TYPES = (ExactDagPosterior, DbnPosterior, GraphPosterior)

# Loaded after `_static_edges`, `_RESPONSE_FAMILY`, and `_not_applicable` exist.
# A top-level import would cycle: `_prepare_route` imports those helpers back.
_PrepareRoute = importlib.import_module("._prepare_route", __package__)._PrepareRoute


def _check_response_observation(query: Any, inference: Any) -> None:
    """Observation mechanism and its assumptions on a response.

    A declared target population is licensed by the Rust study builder, which
    refuses it with ``population_not_estimable`` on every response route.
    """
    from .observation import Complete

    if query.observation is not None and not isinstance(query.observation, Complete):
        if isinstance(inference, Bayesian) and not getattr(query, "is_temporal", False):
            raise CausalUnsupportedError("these response cells require complete observations")
        if not getattr(query, "is_temporal", False) and isinstance(query, InterventionResponse):
            raise CausalUnsupportedError("these response cells require complete observations")
    if query.observation_assumptions and (
        query.observation is None or isinstance(query.observation, Complete)
    ):
        raise CausalUnsupportedError(
            "observation_assumptions require an explicit observation mechanism"
        )


_PreparedResult = (
    AnalysisResult
    | CausalResponseView
    | ExactTransportDistribution
    | StatisticalTransportDistribution
    | TransportResponseGrid
)

ResultT = TypeVar("ResultT", covariant=True)


class PreparedAnalysis(Generic[ResultT]):
    """Compile-once / re-estimate-many handle for licensed analysis cells.

    **Frozen at prepare:** schema (names, types, order); graph, accepted graph,
    or licensed graph-posterior atoms and weights; query identity; identifier;
    estimator and its configuration; validation suite, custom validators and
    replicate / draw budgets; observation / transport / interference
    assumptions; panel, multi-environment, or event data layout.

    **Estimate click:** same-schema data (or the retained data); seeds /
    threads; the study's execution controls (``cancel`` / ``on_progress`` /
    ``on_stage``), which a click may override. Does not re-identify or
    recompile the logical plan.

    **Refute click:** AverageEffect and scalar Dag ``InterventionResponse``.
    :meth:`refute` raises ``CausalUnsupportedError``, prefixed with the
    matrix's own wire id for the cell: ``not_applicable:`` for ResponseCurve
    and for temporal / class-aware InterventionResponse (those remain
    function-valued or envelope-mixed surfaces). ConditionalEffect and Dag
    InterventionResponse cheap/full run the scalar ATE-shaped suite.

    **Re-prepare required:** any frozen field change, including schema mismatch.

    ``analyze`` is ``prepare(...).estimate()``; its result retains this handle
    as ``result.study``. For streaming append + incremental OLS, use
    :class:`antecedent.CausalState`.
    """

    def __init__(
        self,
        native: Any,
        *,
        kind: Literal[
            "average",
            "response_curve",
            "intervention_response",
            "exact_transport",
            "statistical_transport",
            "transport_grid",
            "learned_trial",
            "z_transport",
        ] = "average",
        query: _PreparedQuery | None = None,
        display_query: Any = None,
        seed: int = 1,
        threads: int | None = None,
        controls: _Controls | None = None,
        deferred_suite: str | None = None,
        snapshot_data: Any = None,
        design_columns: dict[str, Any] | None = None,
        named_design_columns: frozenset[str] = frozenset(),
    ) -> None:
        self._native = native
        self._kind = kind
        from ._transport_lifecycle import transport_lifecycle

        self._transport = transport_lifecycle(kind)
        self._query = query
        self._display_query = query if display_query is None else display_query
        # A scalar Dag InterventionResponse runs its refuter suite as the
        # second click of every estimate, so `refute=` is honoured on the
        # prepared route exactly as it is on a one-shot analyze.
        self._deferred_suite = deferred_suite
        self._snapshot_data = snapshot_data
        self._design_columns = design_columns
        # Data columns the query named for its design inputs (see antecedent._columns).
        # They were read and dropped at prepare; a later click drops them again so the
        # same table can be passed back, but never re-reads the frozen design.
        self._named_design_columns = named_design_columns
        self._seed = seed
        self._threads = threads
        self._controls = controls or _Controls()
        self._cancelled = False
        # Set by transport/_day1.py::prepare_transport when identification is
        # deferred (no native execution yet): the frozen inputs needed to
        # re-derive identification (inspect()) or re-bind data (refresh())
        # without one. Absent for every non-transport study.
        self._transport_stage: TransportStage | None = None

    def _frozen(self, execution: Any) -> PreparedAnalysis[ResultT]:
        """A handle over one retained execution with this study's seed and controls."""
        return PreparedAnalysis(
            execution,
            kind=self._kind,
            query=self._query,
            display_query=self._display_query,
            seed=self._seed,
            threads=self._threads,
            controls=self._controls,
            design_columns=self._design_columns,
            named_design_columns=self._named_design_columns,
        )

    @property
    def validator_names(self) -> tuple[str, ...]:
        """Attested custom-validator names frozen at prepare (their claim identity)."""
        if self._transport is not None:
            return ()
        return tuple(self._native.validator_names())

    def rebind_validators(self, validators: Mapping[str, Any]) -> None:
        """Re-bind caller custom validators by their exact attested names.

        Validators are in-process callbacks: a claim carries their attested
        results, never the callables. The names must equal
        :attr:`validator_names`; otherwise the rebind is refused.
        """
        if self._transport is not None:
            if validators:
                raise ValueError("Transport studies do not fit or invoke sampled-data validators")
            return
        self._native.rebind_validators(dict(validators))

    @classmethod
    @describe_refusal
    def prepare(
        cls,
        data: Mapping[str, Any] | Any,
        *,
        query: _PreparedQuery,
        graph: Dag | Sequence[tuple[str, str]] | Any | None = None,
        discovery: Any | None = None,
        inference: Frequentist | Bayesian | TransportInference | None = None,
        identifier: str | Identifier | None = None,
        estimator: str | Estimator | Any | None = None,
        estimator_config: Mapping[str, Any] | None = None,
        refute: bool | Refute | Literal["full", "placebo", "none", "cheap"] | None = None,
        seed: int = 1,
        bootstrap: int | None = None,
        threads: int | None = None,
        latency: Latency | Literal["interactive", "standard", "report"] | None = None,
        class_prior: ClassPrior | None = None,
        max_completions: int | None = None,
        population_registry: Any | None = None,
        cancel: Any | None = None,
        on_progress: Any | None = None,
        on_stage: Any | None = None,
        validators: Sequence[Any] | Mapping[str, Any] | None = None,
        accept_discovered: bool = True,
        regimes: Sequence[int] | None = None,
        provider: Any | None = None,
        controls: TransportControls | None = None,
    ) -> PreparedAnalysis[_PreparedResult]:
        """Compile a durable plan for a licensed analysis cell.

        Every argument either reaches the compiled study or is refused with a
        reason code; none is dropped. Omitted ``refute`` / ``bootstrap`` /
        ``latency`` reach the Rust study builder as omitted, so its one
        omitted-default table, latency-tier mapping and refute downgrade apply
        (``antecedent._defaults.OMITTED`` reads that table).

        Supports ``AverageEffect``, ``ResponseCurve``, ``ConditionalEffect``,
        ``PathSpecificEffect``, ``InterventionalDistribution``,
        ``InterventionResponse``, ``PulseEffect``, ``SustainedEffect``,
        ``TemporalMediationEffect``, static ``MediationEffect``,
        ``Counterfactual``, and the six Frequentist derivative query types
        on an explicit graph or accepted wrapper.
        ``AverageEffect`` also prepares on a ``Pag`` or bidirected ``Admg``; the
        generalized-adjustment envelope or general-ID result is frozen at
        prepare and reused by every estimate click (``exec.identify.cached``).
        Temporal ``ResponseCurve`` / ``InterventionResponse`` (keyword ``horizons``)
        prepare on a ``TemporalDag`` or lagged edge list; their Frequentist
        ``bootstrap`` is the joint circular-block replicate count behind the
        pointwise and simultaneous bands. An omitted ``bootstrap`` takes the
        ``latency`` tier's replicate count from the omitted-default table
        (``antecedent._native.omitted_defaults()``), as Pulse / Sustained do;
        ``0`` publishes the point surface with no band and an
        ``estimate.temporal_response.band_withheld`` warning. Every other
        prepared response refuses a positive ``bootstrap``. Pulse / Sustained /
        TemporalMediation prepare on a ``TemporalDag`` or lagged edge list; a
        Frequentist ``TemporalMediationEffect`` with ``bootstrap=None`` follows the
        same ``latency`` tier for its shared circular-block replicates.
        ``discovery=ExactDagPosterior()`` / ``DbnPosterior()`` / a constructed
        ``GraphPosterior`` compiles the licensed graph-posterior cells
        (AverageEffect / ConditionalEffect / ResponseCurve / one-coordinate
        InterventionResponse on DAG, CPDAG, PAG, and ADMG atoms; Pulse /
        Sustained / TemporalMediationEffect / temporal ResponseCurve /
        one-coordinate InterventionResponse on a DBN or temporal-class
        posterior). Frequentist DBN mixtures
        take their shared circular-block replicates from the ``latency`` tier
        (or an explicit ``bootstrap``).

        ``graph=`` takes a ``Dag`` / ``Cpdag`` / ``Pag`` / ``Admg`` /
        ``TieredBackground`` / ``TemporalDag`` / ``TemporalCpdag`` /
        ``TemporalPag``, an edge list, or an :class:`antecedent.AcceptedGraph`.
        ``discovery=`` also takes a live configuration (``PC`` / ``GES`` /
        ``LiNGAM`` / ``NOTEARS`` / ``FCI`` / ``RFCI`` / ``PCMCI`` /
        ``PCMCIPlus`` / ``LPCMCI`` / ``JPCMCIPlus`` / ``RPCMCI``), which runs
        once and is accepted through its review gate (``accept_discovered=False``
        raises ``ReviewRequired`` while anything is pending).

        ``data`` may be a table, a series, or an ``antecedent.data`` panel /
        multi-environment / event frame; frames compile on the matching Rust
        data modality for every query it licenses and are refused otherwise.

        ``cancel`` / ``on_progress`` apply to prepare-time identification and to
        every estimate click; ``on_stage`` streams the stages of each click on
        routes that have them. The handle retains all three; a click may
        override them.
        """
        from .transport import ExactTransportData, StatisticalTransportData
        from .transport._day1 import prepare_transport, refuse_transport_only_kwargs
        from .transport.advanced import (
            ExactTransportQuery,
            StatisticalTransportQuery,
            TransportResponseGridQuery,
            prepare_exact,
            prepare_response_grid,
            prepare_statistical,
        )

        refuse_transport_only_kwargs(
            query, provider=provider, inference=inference, controls=controls
        )
        # A design-carrying query may name its row-aligned inputs by data column;
        # resolve those from the raw data (any dtype) and drop them before the numeric
        # ingest. No-op when the caller passed the arrays inline.
        from ._columns import collect_column_names
        from ._columns import resolve_query as _resolve_query_columns

        named_design_columns = frozenset(collect_column_names(query))
        query, data = _resolve_query_columns(query, data)
        if isinstance(query, Transport):
            if not isinstance(graph, Admg):
                raise CausalTypeError("transport.Transport requires graph=Admg(...)")
            if isinstance(inference, (Frequentist, Bayesian)):
                raise CausalUnsupportedError(
                    "transport.Transport uses inference=TransportInference(...); "
                    "Frequentist/Bayesian do not apply",
                    reason_code="option_not_applicable",
                )
            if identifier not in (None, "transport.sid"):
                raise CausalUnsupportedError(
                    "transport.Transport is identified by transport.sid; "
                    "another identifier= does not apply",
                    reason_code="option_not_applicable",
                )
            if (
                any(
                    option is not None
                    for option in (
                        discovery,
                        estimator,
                        estimator_config,
                        refute,
                        bootstrap,
                        threads,
                        latency,
                        class_prior,
                        max_completions,
                        population_registry,
                        on_progress,
                        on_stage,
                        validators,
                        regimes,
                    )
                )
                or seed != 1
                or not accept_discovered
            ):
                raise CausalUnsupportedError(
                    "transport.Transport uses provider=, inference=TransportInference, "
                    "and controls=; ordinary analyze knobs do not apply",
                    reason_code="option_not_applicable",
                )
            return prepare_transport(
                data,
                query=query,
                graph=graph,
                provider=provider,
                inference=inference,
                controls=controls,
                cancel=cancel,
            )

        if isinstance(query, TransportResponseGridQuery):
            if not isinstance(data, (ExactTransportData, StatisticalTransportData)):
                raise ValueError("Transport grid requires explicit exact or statistical providers")
            if (
                any(
                    option is not None
                    for option in (
                        graph,
                        discovery,
                        inference,
                        identifier,
                        estimator,
                        estimator_config,
                        refute,
                        bootstrap,
                        threads,
                        latency,
                        class_prior,
                        max_completions,
                        population_registry,
                        on_progress,
                        on_stage,
                        validators,
                        regimes,
                    )
                )
                or seed != 1
                or not accept_discovered
            ):
                raise ValueError(
                    "Transport grids use their retained graph, catalog, and inference settings"
                )
            return prepare_response_grid(
                query.identification,
                query.catalog,
                data,
                at=query.at,
                bootstrap=query.bootstrap,
                coverage_level=query.coverage_level,
                seed=query.seed,
                estimator=query.estimator,
                cancel=cancel,
            )
        if isinstance(query, StatisticalTransportQuery) or isinstance(
            data, StatisticalTransportData
        ):
            if not isinstance(query, StatisticalTransportQuery) or not isinstance(
                data, StatisticalTransportData
            ):
                raise ValueError(
                    "Statistical transport requires StatisticalTransportData and StatisticalTransportQuery"
                )
            if (
                any(
                    option is not None
                    for option in (
                        graph,
                        discovery,
                        inference,
                        identifier,
                        estimator,
                        estimator_config,
                        refute,
                        threads,
                        latency,
                        class_prior,
                        max_completions,
                        population_registry,
                        on_progress,
                        on_stage,
                        validators,
                        regimes,
                    )
                )
                or not accept_discovered
            ):
                raise ValueError(
                    "Statistical transport uses its retained graph, catalog, and inference settings"
                )
            statistical_prepared = prepare_statistical(
                query.identification,
                query.catalog,
                data,
                at=query.at,
                cancel=cancel,
                bootstrap=query.bootstrap,
                coverage_level=query.coverage_level,
                estimator=query.estimator,
                seed=query.seed,
            )
            statistical_prepared._controls = _Controls(cancel=cancel)
            return statistical_prepared

        if isinstance(query, ExactTransportQuery) or isinstance(data, ExactTransportData):
            if not isinstance(query, ExactTransportQuery) or not isinstance(
                data, ExactTransportData
            ):
                raise ValueError(
                    "Exact transport requires ExactTransportData and ExactTransportQuery"
                )
            if (
                any(
                    option is not None
                    for option in (
                        graph,
                        discovery,
                        inference,
                        identifier,
                        estimator,
                        estimator_config,
                        refute,
                        bootstrap,
                        threads,
                        latency,
                        class_prior,
                        max_completions,
                        population_registry,
                        on_progress,
                        on_stage,
                        validators,
                        regimes,
                    )
                )
                or seed != 1
                or not accept_discovered
            ):
                raise ValueError(
                    "Exact transport uses its retained graph and evidence contract; sampled-data preparation options do not apply"
                )
            exact_prepared = prepare_exact(
                query.identification, query.catalog, data, at=query.at, cancel=cancel
            )
            exact_prepared._controls = _Controls(cancel=cancel)
            return exact_prepared

        from .population import registry_wire

        display_query = query
        query = coerce_query(query)
        if isinstance(identifier, Identifier):
            identifier = str(identifier)
        estimator, estimator_config = _unwrap_estimator(estimator, estimator_config)
        if latency is not None:
            latency = str(coerce_latency(latency))  # type: ignore[assignment]
        # The native side takes a suite name or False; the parameter keeps its
        # richer public type.
        suite: str | bool | None = None
        if refute is not None:
            coerced = coerce_refute(refute)
            suite = str(coerced) if isinstance(coerced, Refute) else coerced
        if bootstrap is not None and (
            isinstance(bootstrap, bool)
            or not isinstance(bootstrap, numbers.Integral)
            or bootstrap < 0
        ):
            raise CausalValueError("bootstrap must be a non-negative integer or None")
        inference = inference or Frequentist()
        if not isinstance(inference, (Frequentist, Bayesian)):
            raise CausalTypeError("inference must be Frequentist or Bayesian")
        execution_controls = _Controls(cancel=cancel, on_progress=on_progress, on_stage=on_stage)
        if isinstance(query, (ResponseCurve, InterventionResponse)):
            _check_response_observation(query, inference)

        from .accepted_graph import AcceptedGraph
        from .discovery import RPCMCI

        live_discovery = discovery is not None and not isinstance(discovery, _GRAPH_POSTERIOR_TYPES)
        if isinstance(graph, AcceptedGraph) and discovery is not None:
            raise CausalUnsupportedError(
                "graph=AcceptedGraph(...) rejects discovery=; the structure artifact is "
                "already accepted (call rediscover() explicitly to replace it)"
            )
        if graph is not None and discovery is not None:
            raise CausalValueError("PreparedAnalysis.prepare rejects both graph= and discovery=")
        if regimes is not None and not isinstance(discovery, RPCMCI):
            raise _not_applicable("regimes", "a discovery other than RPCMCI")
        if not accept_discovered and not live_discovery:
            raise _not_applicable(
                "accept_discovered=False", "a study without live discovery (nothing to review)"
            )
        if live_discovery and latency == "interactive":
            raise CausalUnsupportedError(
                "discovery= is not on the interactive estimate path; run discovery once "
                "(Config.accept(data) -> AcceptedGraph), then prepare(graph=..., "
                "latency='interactive')",
                reason_code="option_not_applicable",
            )
        if live_discovery:
            graph = _accept_live_discovery(
                data,
                discovery,
                accept_discovered=accept_discovered,
                regimes=regimes,
                seed=seed,
                threads=threads,
                controls=execution_controls,
            )
            discovery = None
        structure_accepted = isinstance(graph, AcceptedGraph)
        discovery_algorithm = None
        if structure_accepted:
            discovery_algorithm = cast(AcceptedGraph, graph).algorithm_id
            graph = cast(AcceptedGraph, graph).graph
        if class_prior is not None and (
            not isinstance(graph, (TemporalCpdag, TemporalPag))
            or not isinstance(inference, Bayesian)
        ):
            raise CausalUnsupportedError(
                "class_prior requires Bayesian inference on TemporalCpdag or TemporalPag",
                reason_code="option_not_applicable",
            )
        if max_completions is not None and not isinstance(graph, (TemporalCpdag, TemporalPag)):
            raise _not_applicable("max_completions", "a structure without class completions")

        design_columns = None
        if isinstance(
            query,
            (
                PanelDifferenceInDifferences,
                AugmentedPanelDiD,
                StaggeredAdoption,
                SyntheticControl,
                SyntheticDifferenceInDifferences,
            ),
        ):
            if isinstance(query, AugmentedPanelDiD):
                names, columns, design_columns = _augmented_panel_did_payload(data, query)
            elif isinstance(query, StaggeredAdoption):
                names, columns, design_columns = _staggered_payload(data, query)
            elif isinstance(query, (SyntheticControl, SyntheticDifferenceInDifferences)):
                names, columns, design_columns = _synthetic_control_payload(data, query)
            else:
                names, columns, design_columns = _panel_did_payload(data, query)
            frame = None
        elif isinstance(query, ConditionalDoseResponse):
            names, columns, design_columns = _continuous_dose_payload(data, query)
            frame = None
        elif isinstance(query, LongitudinalRegime):
            names, columns = _longitudinal_regime_payload(data, query)
            frame = None
        else:
            names, columns, frame = _frame_payload(data)
        predicates, distributions = registry_wire(population_registry)
        population = coerce_target_population(getattr(query, "target_population", None))
        options: dict[str, Any] = {
            "refute": suite,
            "bootstrap": bootstrap,
            "latency": latency,
            "validators": validators,
            "target_population": population,
            "population_predicates": predicates or None,
            "population_distributions": distributions or None,
            "cancel": cancel,
            "on_progress": on_progress,
            "inference": _inference_wire(inference),
            "discovery_algorithm": discovery_algorithm,
        }
        route = _PrepareRoute(
            names=names,
            columns=columns,
            frame=frame,
            query=query,
            graph=graph,
            discovery=discovery,
            inference=inference,
            identifier=identifier,
            estimator=estimator,
            estimator_config=estimator_config,
            refute=suite,
            bootstrap=bootstrap,
            threads=threads,
            seed=seed,
            class_prior=class_prior,
            max_completions=max_completions,
            accepted=structure_accepted,
            options=options,
            design_columns=design_columns,
        )
        native, kind = route.compile()
        prepared = cls(
            native,
            kind=kind,
            query=query,
            display_query=display_query,
            seed=seed,
            threads=threads,
            controls=execution_controls,
            deferred_suite=route.deferred_suite,
            snapshot_data=data if route.deferred_suite else None,
            design_columns=design_columns,
            named_design_columns=named_design_columns,
        )
        if on_stage is not None and not native.streams_stages():
            raise CausalUnsupportedError(
                "on_stage streams identify, estimate_point, uncertainty and validate for one "
                "identification and one scalar effect whose point is fitted before its "
                "uncertainty; this route fits its point and uncertainty together or mixes "
                "several identifications, so it has no such stages (use on_progress)",
                reason_code="stage_stream_unavailable",
            )
        return cast(PreparedAnalysis[_PreparedResult], prepared)

    def export_artifact(
        self,
        *,
        artifact_id: str = "prepared-result",
        payload: Literal["query", "result"] = "result",
    ) -> bytes:
        """Export the last full posterior/response, or its query, without refitting.

        ``payload="query"`` also supports functional path/distribution queries.

        Decode Bayesian scalar results with ``inference.decode_posterior_artifact``;
        decode response results with ``artifacts.loads``. Posterior artifacts retain
        draws, quantity names, identification and backend metadata. Response
        artifacts also retain support and assumptions; retain the analysis result
        separately for posterior assumptions and validation reports.
        """
        if self._transport is not None:
            if payload != "result":
                raise ValueError(
                    "Transport artifacts export a checked execution and its full proof"
                )
            return self._native.export()
        return self._native.export_artifact(artifact_id=artifact_id, payload=payload)

    def export(self, *, artifact_id: str = "analysis-result") -> bytes:
        """Export the last execution as a contracted ``analysis_result``, or refuse
        if no claim was produced."""
        if self._transport is not None:
            # A transport handle retains its previous claim through a cancelled
            # click (every refresh is atomic) and refuses on its own when it
            # holds none, so the native export decides.
            return self._native.export()
        if getattr(self, "_cancelled", False):
            raise CausalUnsupportedError(
                "Cancelled estimate produced no claim.",
                reason_code="cancelled_no_claim",
            )
        return self._native.export_contracted_artifact(artifact_id=artifact_id)

    @property
    def structure_source(self) -> str:
        """Support-matrix structure axis frozen at prepare (`explicit` or `accepted`)."""
        return str(self._native.plan_summary().get("structure_source", "explicit"))

    def checked_static_dag_response_info(self) -> dict[str, Any] | None:
        """Inspect the frozen checked DAG response query and procedure, if present.

        The returned mapping is read-only evidence from the prepared native handle;
        it remains available after the original query and graph objects are dropped.
        """
        info = self._native.checked_static_dag_response_info()
        return None if info is None else dict(info)

    def checked_conditional_effect_info(self) -> dict[str, Any] | None:
        """Inspect the retained DAG conditional target, procedure, and bound rows."""
        info = self._native.checked_conditional_effect_info()
        return None if info is None else dict(info)

    def checked_static_mediation_info(self) -> dict[str, Any] | None:
        """Inspect the retained static mediation contrast, graph, and procedure."""
        info = self._native.checked_static_mediation_info()
        return None if info is None else dict(info)

    def checked_bayesian_dag_ate_info(self) -> dict[str, Any] | None:
        """Inspect the retained Bayesian DAG ATE target, backend, and prior source."""
        info = self._native.checked_bayesian_dag_ate_info()
        return None if info is None else dict(info)

    def checked_graph_posterior_effect_info(self) -> dict[str, Any] | None:
        """Inspect frozen graph atoms, weights, and the selected effect procedure."""
        info = self._native.checked_graph_posterior_effect_info()
        return None if info is None else dict(info)

    def checked_temporal_effect_info(self) -> dict[str, Any] | None:
        """Inspect the retained temporal contrast, graph, and estimator choice."""
        info = self._native.checked_temporal_effect_info()
        return None if info is None else dict(info)

    def checked_temporal_response_info(self) -> dict[str, Any] | None:
        """Inspect the frozen temporal response grid, horizons, and uncertainty method."""
        info = self._native.checked_temporal_response_info()
        return None if info is None else dict(info)

    @property
    def evidence_status(self) -> str | None:
        """`licensed` or `allowed_unlicensed`, or ``None`` if the query is off-axis."""
        raw = self._native.plan_summary().get("evidence_status")
        return str(raw) if raw is not None else None

    @property
    def allowlist_reason(self) -> str | None:
        raw = self._native.plan_summary().get("allowlist_reason")
        return str(raw) if raw is not None else None

    @property
    def allowlist_parent(self) -> str | None:
        raw = self._native.plan_summary().get("allowlist_parent")
        return str(raw) if raw is not None else None

    def inspect(self) -> InspectionReport:
        """Everything known about this study, including cached identification.

        ``to_dict()`` gives the structured report; its ``contract`` field is the
        study's domain-separated identities and reasoning slots (ADR 0022), and
        its ``calibration`` is unavailable (``not_executed``) until an estimate
        runs. :meth:`preflight` is the cheap structural-only view.
        """
        if self._kind == "z_transport":
            from .transport._day1 import identification_from_transport

            query = self._query
            if not isinstance(query, Transport):
                raise CausalValueError("transport inspect requires a Transport query")
            stage = getattr(self, "_transport_stage", None) or {}
            graph = stage.get("graph")
            if graph is None:
                raise CausalValueError("transport inspect requires a retained graph")
            return identification_from_transport(
                graph,
                query,
                catalog=stage.get("catalog"),
                identified=stage.get("identified"),
            ).inspect()
        if self._transport is not None:
            if getattr(self, "_native", None) is None:
                from .transport._day1 import identification_from_transport

                query = self._query
                if not isinstance(query, Transport):
                    raise CausalValueError("transport inspect requires a Transport query")
                stage = getattr(self, "_transport_stage", None) or {}
                graph = stage.get("graph")
                if graph is None:
                    raise CausalValueError("transport inspect requires a retained graph")
                return identification_from_transport(
                    graph,
                    query,
                    catalog=stage.get("catalog"),
                    identified=stage.get("identified"),
                ).inspect()
            return InspectionReport(**json.loads(self._native.inspection_json()))

        from dataclasses import replace

        from .results._execution import Answer, CalibrationInfo
        from .results._report import as_inspection

        return as_inspection(
            replace(
                ReasoningSlots.from_contract(dict(self._native.contract())),
                answer=Answer("unavailable", detail="not_executed"),
                calibration=CalibrationInfo(status="unavailable", reason="not_executed"),
            )
        )

    def preflight(self) -> InspectionReport:
        """Cheap structural-only inspection; identification and fitting are not run."""
        from .results._report import as_inspection

        if self._transport is not None:
            return self.inspect()
        return as_inspection(ReasoningSlots.from_contract(self._native.inspect()))

    def diagnose(self, *, seed: int = 1, threads: int | None = None) -> PreflightReport:
        """Fit-free preflight of the retained table and certified adjustment set.

        Reports complete-case and arm counts, missingness, exact duplicate columns, the
        numerical rank of the ``[1 | Z]`` design with the dependent columns named, and
        review flags for near-deterministic relations (never a change to the adjustment
        set). Nothing is fit; :meth:`diagnose_fit` is the check that fits a propensity
        model. Unlike :meth:`preflight`, which inspects the plan's structure, this reads
        the data.
        """
        from .preflight import diagnose_prepared

        return diagnose_prepared(self._native, seed=seed, threads=threads)

    def diagnose_fit(self, *, seed: int = 1, threads: int | None = None) -> NuisanceFitDiagnostics:
        """Fit a diagnostic propensity model and report its score range and arm ESS.

        Not preflight: the fit is part of the evidence. A fit that fails before any score
        exists reports every fitted quantity as absent.
        """
        from .preflight import diagnose_fit_prepared

        return diagnose_fit_prepared(self._native, seed=seed, threads=threads)

    def plan_rank_drop(
        self,
        priority: Sequence[str] | None = None,
        *,
        seed: int = 1,
        threads: int | None = None,
    ) -> RankDropPlan:
        """Opt-in rank-deficiency drop plan under a declared column priority (a record,
        not an execution). See :func:`antecedent.preflight.plan_rank_drop`."""
        from .preflight import plan_rank_drop_prepared

        return plan_rank_drop_prepared(self._native, priority, seed=seed, threads=threads)

    def estimate_cost(self) -> CostEstimate:
        """Planning-time counts (nuisance fits, folds, bootstrap replicates, design bytes)
        and the active inference default. A planning hint, not a runtime guarantee; no
        seconds are estimated."""
        from .preflight import estimate_cost_prepared

        return estimate_cost_prepared(self._native)

    def preview_transform(self, intent: str) -> dict[str, str]:
        """Pure preview of a transformation of this study; nothing is re-executed.

        ``intent`` is one of ``display_precision``, ``filter_display``,
        ``compatible_data_replace``, ``retarget``, ``filter_population``,
        ``new_conditional_query``, ``change_graph``, ``change_prior``,
        ``change_physical_policy`` or ``average_unweighted_class``. The report
        names the intent, whether it is refused, its obligations, and the frozen
        input identities it would carry under ``input_<domain>`` keys.
        """
        return dict(self._native.preview_transform(intent))

    @property
    def plan(self) -> PhysicalPlanView:
        """Physical-plan summary retained from prepare."""
        raw = self._native.plan_summary()
        return PhysicalPlanView(
            plan_id=str(raw.get("plan_id", "")),
            estimated_peak_memory_bytes=(
                int(raw["estimated_peak_memory_bytes"])
                if "estimated_peak_memory_bytes" in raw
                else None
            ),
            workspace_bytes=(int(raw["workspace_bytes"]) if "workspace_bytes" in raw else None),
            batch_size=int(raw["batch_size"]) if "batch_size" in raw else None,
            worker_threads=int(raw.get("worker_threads", 0)),
            expected_python_crossings=int(raw.get("expected_python_crossings", 0)),
            deterministic_reductions=str(raw.get("deterministic_reductions", "true")).lower()
            in ("1", "true"),
            kernels=raw.get("kernels") or None,
        )

    def replace_snapshot(self, data: Any, *, cancel: Any = _UNSET) -> None:
        """Replace exact-law providers and invalidate execution claims atomically."""
        if self._transport is not None:
            self._native.replace_snapshot(
                self._transport.payload(data),
                cancel=self._controls.cancel if cancel is _UNSET else cancel,
            )
            return
        raise ValueError("Use refresh for this sampled-data modality")

    def _click_controls(self, cancel: Any, on_progress: Any, on_stage: Any) -> dict[str, Any]:
        """The study's retained controls, with any per-click override applied."""
        controls = self._controls
        chosen = _Controls(
            cancel=controls.cancel if cancel is _UNSET else cancel,
            on_progress=controls.on_progress if on_progress is _UNSET else on_progress,
            on_stage=controls.on_stage if on_stage is _UNSET else on_stage,
        )
        return chosen.kwargs()

    def _run_click(self, call: Any) -> Any:
        from .errors import CausalCancelledError

        try:
            raw = call()
        except CausalCancelledError:
            # The handle still holds the previous execution; it must not export as this one.
            self._cancelled = True
            raise
        self._cancelled = False
        return raw

    def _click_payload(self, data: Any) -> tuple[list[str], list[Any], dict[str, Any] | None]:
        from ._columns import drop_columns

        data = drop_columns(data, self._named_design_columns)
        query = self._query
        if isinstance(query, ConditionalDoseResponse):
            names, columns, design = _continuous_dose_payload(data, query)
            if design != self._design_columns:
                raise CausalUnsupportedError(
                    "continuous-dose refresh requires prepared baseline-group row order",
                    reason_code="invalid_argument",
                )
            return names, columns, None
        if isinstance(
            query,
            (
                PanelDifferenceInDifferences,
                AugmentedPanelDiD,
                StaggeredAdoption,
                SyntheticControl,
                SyntheticDifferenceInDifferences,
            ),
        ):
            names, columns, design = (
                _augmented_panel_did_payload(data, query)
                if isinstance(query, AugmentedPanelDiD)
                else _staggered_payload(data, query)
                if isinstance(query, StaggeredAdoption)
                else _synthetic_control_payload(data, query)
                if isinstance(query, (SyntheticControl, SyntheticDifferenceInDifferences))
                else _panel_did_payload(data, query)
            )
            if design != self._design_columns:
                raise CausalUnsupportedError(
                    "design refresh requires the prepared unit and period row order",
                    reason_code="invalid_argument",
                )
            return names, columns, None
        if isinstance(query, LongitudinalRegime):
            names, columns = _longitudinal_regime_payload(data, query)
            return names, columns, None
        if isinstance(query, ResponseCurve) and query.observation is not None:
            from .observation import Complete, _ensure_latent_schema_column

            if not isinstance(query.observation, Complete):
                names, columns = ingest_columns(data)
                names, columns = _ensure_latent_schema_column(names, columns, query.observation)
                return names, columns, None
        return _frame_payload(data)

    def _wrap(
        self, raw: Any
    ) -> (
        AnalysisResult
        | CausalResponseView
        | StatisticalTransportDistribution
        | TransportResponseGrid
    ):
        if self._kind in ("response_curve", "intervention_response"):
            query = self._query if isinstance(self._query, _RESPONSE_FAMILY) else None
            return _wrap_prepared_response(raw, query=cast(Any, query), prepared=self)
        return _wrap_ate(raw, query=self._display_query, prepared=self)

    def _click(
        self,
        data: Any,
        *,
        refresh: bool,
        seed: int | None,
        threads: int | None,
        controls: dict[str, Any],
    ) -> (
        AnalysisResult
        | CausalResponseView
        | StatisticalTransportDistribution
        | TransportResponseGrid
    ):
        if self._transport is not None:
            if getattr(self, "_native", None) is None:
                from .transport._wrap import unavailable_from_stage

                return unavailable_from_stage(self)
            raw = self._run_click(
                lambda: self._transport.execute(
                    self._native,
                    data,
                    refresh=refresh,
                    seed=seed,
                    threads=threads,
                    controls=controls,
                )
            )
            if isinstance(self._query, Transport):
                from .transport._wrap import wrap_transport_result

                return wrap_transport_result(self, raw)
            return raw
        seed = self._seed if seed is None else seed
        threads = self._threads if threads is None else threads
        response = self._kind in ("response_curve", "intervention_response")
        native = self._native
        if data is None:
            bound = native.estimate_response_bound if response else native.estimate_bound
            raw = self._run_click(lambda: bound(seed=seed, threads=threads, **controls))
            return self._with_deferred_suite(
                self._wrap(raw), self._snapshot_data, seed=seed, threads=threads
            )
        if isinstance(self._query, (PolicyValue, MultiActionPolicyValue)):
            raise CausalUnsupportedError(
                "PolicyValue recommendations, nuisance predictions, and subject ownership are bound to the prepared evaluation rows; prepare a new study for new data",
                reason_code="option_not_applicable",
            )
        names, columns, frame = self._click_payload(data)
        if frame is not None:
            raw = self._run_click(
                lambda: native.estimate_frame(
                    names,
                    columns,
                    frame,
                    response=response,
                    refresh=refresh,
                    seed=seed,
                    threads=threads,
                    **controls,
                )
            )
            return self._wrap(raw)
        if refresh:
            fn = native.refresh_response if response else native.refresh
        else:
            fn = native.estimate_response if response else native.estimate
        raw = self._run_click(lambda: fn(names, columns, seed=seed, threads=threads, **controls))
        return self._with_deferred_suite(self._wrap(raw), data, seed=seed, threads=threads)

    def _with_deferred_suite(
        self, result: Any, data: Any, *, seed: int, threads: int | None
    ) -> Any:
        """Run the study's refuter suite as this click's second click.

        A scalar ``InterventionResponse`` on a Dag is estimated by the response
        executor, which fits its point and uncertainty together and has no
        validation stage of its own; the ATE-shaped suite denotes on it and
        runs against the estimate just produced.
        """
        if self._deferred_suite is None or data is None:
            return result
        from .results._report import copy_model

        refuted = self.refute(data, suite=self._deferred_suite, seed=seed, threads=threads)
        # The suite ran against this estimate, so the click's claim is the
        # refuted one; the estimate itself is the response executor's.
        return copy_model(
            result,
            validation=refuted.validation,
            certificate=refuted.certificate,
            reasoning=refuted.reasoning,
            evidence_status=refuted.evidence_status,
            claim_id=refuted.claim_id,
        )

    def mechanism_sensitivity(
        self,
        *,
        outcome_values: Sequence[float],
        parent_cardinalities: Sequence[int],
        treatment_levels: tuple[int, int],
        max_fraction: float,
        source_kernel_regime: int,
        source_kernel_snapshot: str,
        target_parent_regime: int,
        target_parent_snapshot: str,
        source_kernel: Sequence[tuple[Sequence[int], Sequence[float]]],
        source_parent_law: Sequence[tuple[Sequence[int], float]],
        target_parent_law: Sequence[tuple[int, Sequence[int], float]],
        decision_threshold: float | None = None,
        perturbed_treatment_level: int | None = None,
        perturbed_root_mechanism: str | None = None,
        perturbed_conditional_mechanism: str | None = None,
        cancel: Any = None,
    ) -> dict[str, Any]:
        """Evaluate checked fixed-graph mechanism sensitivity for exact transport."""
        if self._kind != "exact_transport":
            raise CausalUnsupportedError(
                "mechanism_sensitivity applies only to prepared exact transport",
                reason_code="option_not_applicable",
            )
        raw = self._native.mechanism_sensitivity(
            list(outcome_values),
            list(parent_cardinalities),
            treatment_levels,
            max_fraction,
            source_kernel_regime,
            source_kernel_snapshot,
            target_parent_regime,
            target_parent_snapshot,
            [(list(levels), list(probabilities)) for levels, probabilities in source_kernel],
            [(list(levels), probability) for levels, probability in source_parent_law],
            [
                (treatment, list(levels), probability)
                for treatment, levels, probability in target_parent_law
            ],
            decision_threshold=decision_threshold,
            perturbed_treatment_level=perturbed_treatment_level,
            perturbed_root_mechanism=perturbed_root_mechanism,
            perturbed_conditional_mechanism=perturbed_conditional_mechanism,
            cancel=cancel,
        )
        return json.loads(raw)

    @describe_refusal
    def estimate(
        self,
        data: Mapping[str, Any] | Any = None,
        *,
        seed: int | None = None,
        threads: int | None = None,
        cancel: Any = _UNSET,
        on_progress: Any = _UNSET,
        on_stage: Any = _UNSET,
    ) -> ResultT:
        """Re-estimate without recompiling.

        With no ``data`` this re-executes the prepared program on the retained
        data. New ``data`` must match the prepared schema (and, for a panel /
        multi-environment / event study, its frame kind). ``seed`` / ``threads``
        default to the values given at prepare; ``cancel`` / ``on_progress`` /
        ``on_stage`` default to the study's retained controls and may be
        overridden (``None`` turns one off for this click).
        """
        return cast(
            ResultT,
            self._click(
                data,
                refresh=False,
                seed=seed,
                threads=threads,
                controls=self._click_controls(cancel, on_progress, on_stage),
            ),
        )

    @describe_refusal
    def retarget(
        self,
        weights: Any,
        depends_on: Sequence[str],
        *,
        seed: int = 1,
        threads: int | None = None,
    ) -> AnalysisResult:
        """Estimate a declared target population from frozen scores. No refit.

        Requires a prepared AllObserved iid AIPW or cell-AIPW score table.
        Nonempty ``depends_on`` needs a directed graph (DAG or ADMG) so
        descendant closure can be checked. Nonconstant weights require a
        nonempty ``depends_on``.
        """
        if self._transport is not None:
            raise CausalUnsupportedError(
                "This operation is not licensed on the prepared transport handle",
                reason_code="option_not_applicable",
            )
        import numpy as np

        raw = self._native.retarget(
            np.asarray(weights, dtype=float).tolist(),
            list(depends_on),
            seed=seed,
            threads=threads,
        )
        return _wrap_ate(raw, prepared=self)

    @describe_refusal
    def reexecute_retarget(
        self,
        artifact: bytes,
        *,
        seed: int = 1,
        threads: int | None = None,
    ) -> AnalysisResult:
        """Re-execute the row-weight retarget an exported contract carries.

        The weights travel in the artifact with the identity that binds them to
        one data snapshot and one score table. Re-executing them on a study
        that holds a different snapshot raises ``CausalUnsupportedError`` with
        ``reason_code="row_weights_bound_to_snapshot"`` instead of silently
        reweighting other rows.
        """
        if self._transport is not None:
            raise CausalUnsupportedError(
                "This operation is not licensed on the prepared transport handle",
                reason_code="option_not_applicable",
            )
        raw = self._native.reexecute_retarget(bytes(artifact), seed=seed, threads=threads)
        return _wrap_ate(raw, prepared=self)

    @describe_refusal
    def refresh(
        self,
        data: Mapping[str, Any] | Any,
        *,
        seed: int | None = None,
        threads: int | None = None,
        cancel: Any = _UNSET,
        on_progress: Any = _UNSET,
        on_stage: Any = _UNSET,
    ) -> ResultT:
        """Replace retained data and re-estimate (controls as in :meth:`estimate`)."""
        if isinstance(self._query, Transport):
            from .transport._day1 import catalog_from_evidence

            stage = getattr(self, "_transport_stage", None) or {}
            _catalog, bound = catalog_from_evidence(self._query, data, graph=stage.get("graph"))
            if bound is not None:
                data = bound
            if self._transport is not None and getattr(self, "_native", None) is not None:
                self.replace_snapshot(data)
        return cast(
            ResultT,
            self._click(
                data,
                refresh=True,
                seed=seed,
                threads=threads,
                controls=self._click_controls(cancel, on_progress, on_stage),
            ),
        )

    @describe_refusal
    def refute(
        self,
        data: Mapping[str, Any] | Any,
        suite: Refute | Literal["placebo", "full", "cheap"] | bool | str = "placebo",
        *,
        seed: int | None = None,
        threads: int | None = None,
        cancel: Any = _UNSET,
    ) -> AnalysisResult:
        """Second-click refute against the last :meth:`estimate` / :meth:`refresh`.

        Interactive first clicks typically use ``refute=False`` or ``cheap``;
        call this with ``suite="placebo"`` or ``"full"`` for the deferred suite.
        ``seed`` / ``threads`` / ``cancel`` default to the study's own.
        """
        if self._transport is not None:
            raise CausalUnsupportedError(
                "This operation is not licensed on the prepared transport handle",
                reason_code="option_not_applicable",
            )
        if self._kind == "response_curve":
            raise CausalUnsupportedError(
                "not_applicable: PreparedAnalysis.refute is AverageEffect and scalar "
                "Dag InterventionResponse; ResponseCurve cheap/full/placebo do not denote."
            )
        if isinstance(self._query, (TransportQuery, InterferenceQuery)):
            raise CausalUnsupportedError(
                f"{type(self._query).__name__} has no refuter suite: its estimand is "
                "defined by the design (selection diagram and trial probabilities, or the "
                "randomization and network), and the average-effect refuters do not apply",
                reason_code="option_not_applicable",
            )
        if isinstance(self._query, (AnomalyAttribution, ChangeAttribution)):
            raise CausalUnsupportedError(
                f"{type(self._query).__name__} has no refuter suite: GCM attribution "
                "scores are not an average treatment effect, and the average-effect "
                "refuters do not apply",
                reason_code="option_not_applicable",
            )
        if isinstance(suite, Refute):
            suite = str(suite)
        from ._columns import drop_columns

        names, columns, arrow = _prepared_columns(drop_columns(data, self._named_design_columns))
        kwargs: dict[str, Any] = dict(
            seed=self._seed if seed is None else seed,
            threads=self._threads if threads is None else threads,
        )
        token = self._controls.cancel if cancel is _UNSET else cancel
        if token is not None:
            kwargs["cancel"] = token
        fn = self._native.refute_arrow_c if arrow else self._native.refute
        raw = fn(names, columns, suite, **kwargs)
        return _wrap_ate(raw, prepared=self)


__all__ = [
    "AnalysisResult",
    "ConflictSummaryView",
    "EffectEnvelope",
    "EstimateView",
    "MediationView",
    "IdentificationView",
    "IdentifyResult",
    "MediationEffectsSummary",
    "PerformanceView",
    "PhysicalPlanView",
    "PlanView",
    "PosteriorView",
    "PredictiveCheckReport",
    "PreparedAnalysis",
    "PreparedBatch",
    "BatchRetarget",
    "RetargetClaim",
    "RetargetContrast",
    "RetargetMember",
    "SimultaneousBandMember",
    "SimultaneousInterval",
    "SharedBatchDesign",
    "CandidateScreen",
    "PriorSensitivityReport",
    "RefutationReport",
    "ValidationView",
    "analyze_many",
    "identify",
    "mediation_effects_summary",
]
