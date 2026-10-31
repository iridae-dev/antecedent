"""Wrap native estimation payloads as Python result views.

Moved out of ``estimation.py`` so the facade module is the study lifecycle,
not the per-route field mapping. Names stay importable from ``estimation``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

from ._native import AnalysisResult as TemporalAnalysisResult
from ._native import AteAnalysisResult
from ._posterior_row import PosteriorRowEstimate
from ._quasi_rollout import quasi_rollout_from_raw as _quasi_rollout_from_raw
from ._temporal_design import temporal_design_from_raw as _temporal_design_from_raw
from ._unset_license import unset_dose_license, unset_panel_license, unset_policy_license
from .descriptive import association_from_raw as _association_from_raw
from .errors import CausalValueError
from .experiment import RandomizedExperimentEstimate
from .interference import (
    InterferenceEstimate,
    InterferencePointwiseInterval,
    InterferenceSupport,
    RandomizationContrast,
)
from .policy import (
    ConditionalDoseResponseEstimate,
    ConditionalDoseResponsePoint,
    DoublyRobustPolicyEvaluation,
    FixedDosePolicyValueEstimate,
)
from .quasi import (
    AugmentedPanelDiD,
    AugmentedPanelDiDEstimate,
    IvBoundsEstimate,
    IvBoundsInterval,
    IvBoundsPosterior,
    LocalPolynomialRatioEstimate,
    ObservedWindowSupport,
    PanelDifferenceInDifferences,
    PanelDifferenceInDifferencesEstimate,
    PeriodHistoryMatchEstimate,
    RegressionKink,
    StaggeredAdoption,
    StaggeredEventStudyEstimate,
    StaggeredEventTimeEffect,
    SyntheticControlEstimate,
    SyntheticDifferenceInDifferencesEstimate,
)
from .regimes import LongitudinalRegimeEstimate, SequentialContrastEstimate
from .results import (
    AnalysisResult,
    ConflictSummaryView,
    DistributionAtomView,
    EffectEnvelope,
    EstimateView,
    IdentificationView,
    ImputationExtraView,
    ImputationVectorView,
    ImputationView,
    ImputationWaldView,
    IntervalBindingView,
    MediationView,
    MissingDataView,
    PerformanceView,
    PlanView,
    PosteriorView,
    PredictiveCheckReport,
    PriorSensitivityReport,
    ProbabilityIntervalView,
    ReasoningSlots,
    RefutationReport,
    TemporalMediationGridView,
    TemporalMediationSliceView,
    ValidationView,
)
from .results._views import (
    CounterfactualContrastView,
    DropReasonView,
    GroupAccountingView,
    MissingVariableView,
    PenalizedFallback,
    PenalizedFoldSupport,
    RefitBootstrapReport,
    ReplicatePenalties,
    RowAccountingView,
    SamplingWeightsView,
)
from .survival import (
    CompetingRisksOutcome,
    CumulativeIncidenceEstimate,
    SurvivalDifferenceBand,
    SurvivalEstimate,
)
from .transport._impl import (
    OverlapDiagnostic,
    TransportOverlapReport,
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
            scope_kind=getattr(r, "scope_kind", None),
            scope_target_identity=getattr(r, "scope_target_identity", None),
            scope_outer_atom=getattr(r, "scope_outer_atom", None),
            scope_completion_digest=getattr(r, "scope_completion_digest", None),
            scope_horizon_index=getattr(r, "scope_horizon_index", None),
            scope_horizon=getattr(r, "scope_horizon", None),
        )
        for r in getattr(validation, "reports", None) or ()
    ]


def _predictive_replicates_from_raw(raw: Any) -> Any:
    native = getattr(raw, "predictive_replicates", None)
    if native is None:
        return None
    from .predictive_assessment import PosteriorReplicates  # noqa: PLC0415

    return PosteriorReplicates.from_native(native)


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


def _sampling_weights_view(raw: Any) -> SamplingWeightsView | None:
    """Native ``(rows_used, zero_weight_rows, effective_sample_size, weight_sum)`` → view."""
    if raw is None:
        return None
    rows_used, zero_weight_rows, effective_sample_size, weight_sum = raw
    return SamplingWeightsView(
        rows_used=int(rows_used),
        zero_weight_rows=int(zero_weight_rows),
        effective_sample_size=float(effective_sample_size),
        weight_sum=float(weight_sum),
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


def _counterfactual_contrasts_from_raw(raw: Any) -> tuple[CounterfactualContrastView, ...] | None:
    """Native per-outcome contrast rows → views (``None`` for a single-outcome query)."""
    rows = getattr(raw, "counterfactual_contrasts", None)
    if rows is None:
        return None
    return tuple(
        CounterfactualContrastView(
            outcome=str(outcome),
            mean_effect=float(mean),
            standard_error=None if se is None else float(se),
            unit_effects=[float(v) for v in effects],
            unit_effect_intervals=None
            if intervals is None
            else [(float(lo), float(hi)) for lo, hi in intervals],
            unit_extrapolative=None if flags is None else [bool(f) for f in flags],
            unit_effects_homogeneous=bool(homogeneous),
        )
        for outcome, mean, se, effects, intervals, flags, homogeneous in rows
    )


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


def _label_digest(labels: Sequence[str]) -> str:
    """BLAKE2b-256 of the labels, each length-prefixed so no two label lists collide."""
    import hashlib

    digest = hashlib.blake2b(digest_size=32, person=b"antecedent.drops")
    digest.update(len(labels).to_bytes(8, "little"))
    for label in labels:
        encoded = label.encode()
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def row_accounting_from_dict(
    raw: Mapping[str, Any], labels: Sequence[str] | None = None
) -> RowAccountingView:
    """The :class:`RowAccountingView` of a native accounting mapping.

    ``labels`` are the original participant label of every submitted row; they name the
    dropped rows and digest the names, so the exclusion can be audited by participant.
    """
    dropped = tuple(int(row) for row in raw["dropped_rows"])
    units = raw.get("units")
    named = (
        tuple(labels[row] for row in dropped)
        if labels is not None and len(labels) == raw["rows_total"]
        else None
    )
    return RowAccountingView(
        policy=raw["policy"],
        rows_total=raw["rows_total"],
        rows_kept=raw["rows_kept"],
        rows_dropped=raw["rows_dropped"],
        rows_in_denominator=raw["rows_in_denominator"],
        missing_by_variable=tuple(
            MissingVariableView(**entry) for entry in raw["missing_by_variable"]
        ),
        reasons=tuple(DropReasonView(**entry) for entry in raw["reasons"]),
        dropped_rows=dropped,
        dropped_digest=raw["dropped_digest"],
        groups=tuple(GroupAccountingView(**entry) for entry in raw["groups"]),
        units_total=None if units is None else units["total"],
        units_dropped=None if units is None else units["dropped"],
        dropped_labels=named,
        dropped_labels_digest=None if named is None else _label_digest(named),
    )


def _interval_bindings_from_raw(raw: Any) -> tuple[IntervalBindingView, ...]:
    """The intervals a native result reported, primary first."""
    payload = getattr(raw, "interval_bindings_json", None)
    if not payload:
        return ()
    return tuple(IntervalBindingView(**entry) for entry in json.loads(payload))


def _missing_data_from_raw(raw: Any, prepared: Any | None) -> MissingDataView | None:
    """The missing-data view of a native result, with participant labels when bound."""
    payload = getattr(raw, "missing_data_json", None)
    if not payload:
        return None
    parsed = json.loads(payload)
    labels = None if prepared is None else getattr(prepared, "participants", None)
    imputation = parsed.get("imputation")
    return MissingDataView(
        declared=parsed["declared"],
        assumption=parsed["assumption"],
        covariates=tuple(parsed["covariates"]),
        accounting=row_accounting_from_dict(parsed["accounting"], labels),
        imputation=None
        if imputation is None
        else ImputationView(
            **{
                **imputation,
                "imputed_cells": {
                    entry["variable"]: entry["cells"] for entry in imputation["imputed_cells"]
                },
                "points": tuple(imputation["points"]),
                "vector": _imputation_vector(imputation.get("vector")),
            }
        ),
    )


def _imputation_vector(raw: Mapping[str, Any] | None) -> ImputationVectorView | None:
    """The joint pooling of a multi-coordinate multiple-imputation answer."""
    if raw is None:
        return None
    wald = raw.get("wald")
    return ImputationVectorView(
        labels=tuple(raw["labels"]),
        covariance_basis=raw["covariance_basis"],
        points=tuple(tuple(point) for point in raw["points"]),
        point=tuple(raw["point"]),
        within=tuple(raw["within"]),
        between=tuple(raw["between"]),
        total=tuple(raw["total"]),
        coordinate_df=tuple(raw["coordinate_df"]),
        coordinate_fraction_missing_information=tuple(
            raw["coordinate_fraction_missing_information"]
        ),
        average_relative_increase=raw.get("average_relative_increase"),
        wald=None
        if wald is None
        else ImputationWaldView(
            null=tuple(wald["null"]),
            statistic=wald["statistic"],
            # A reference with no denominator degrees of freedom to spare is the chi-square law.
            df_numerator=wald["df_numerator"],
            df_denominator=math.inf if wald["df_denominator"] is None else wald["df_denominator"],
        ),
        extras=tuple(
            ImputationExtraView(
                label=extra["label"],
                points=tuple(extra["points"]),
                point=extra["point"],
                between=extra["between"],
            )
            for extra in raw.get("extras", ())
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
            interval_95=section.interval_95,
            standard_error=section.standard_error if section.interval_95 is not None else None,
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


def _observed_window_from_raw(raw: Any) -> ObservedWindowSupport | None:
    if raw is None:
        return None
    pre, post, used, dropped, unread = raw
    return ObservedWindowSupport(
        pre_periods=tuple(int(period) for period in pre),
        post_periods=tuple(int(period) for period in post),
        donors_used=int(used),
        donors_dropped=tuple((str(unit), int(missed)) for unit, missed in dropped),
        unread_rows=int(unread),
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
        observed_window=_observed_window_from_raw(section.observed_window),
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
        randomization_ranking_tolerance=section.randomization_ranking_tolerance,
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


def _posterior_row_from_raw(raw: Any) -> PosteriorRowEstimate | None:
    section = getattr(raw, "posterior_row", None)
    return None if section is None else PosteriorRowEstimate._from_raw(section)


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


def _sequential_contrast_from_raw(raw: Any) -> SequentialContrastEstimate | None:
    section = getattr(raw, "sequential_contrast", None)
    if section is None:
        return None
    return SequentialContrastEstimate(
        active_regime=tuple(section.active_regime),
        control_regime=tuple(section.control_regime),
        contrast=section.contrast,
        active_mean=section.active_mean,
        control_mean=section.control_mean,
        evaluated_rows=section.evaluated_rows,
        minimum_action_probability=section.minimum_action_probability,
        min_probability=section.min_probability,
        coverage=section.coverage,
        active_scores=tuple(section.active_scores),
        control_scores=tuple(section.control_scores),
        contrast_scores=tuple(section.contrast_scores),
        uncertainty=section.uncertainty,
        standard_error=section.standard_error,
        interval=None if section.interval is None else tuple(section.interval),
        fits_computed=section.fits_computed,
        fits_reused=section.fits_reused,
    )


def _history_match_from_raw(raw: Any) -> PeriodHistoryMatchEstimate | None:
    section = getattr(raw, "history_match", None)
    if section is None:
        return None
    return PeriodHistoryMatchEstimate(
        contrast=section.contrast,
        weight_total=section.weight_total,
        matched_queries=tuple(section.matched_queries),
        matched_donors=tuple(section.matched_donors),
        differences=tuple(section.differences),
        pair_weights=tuple(section.pair_weights),
        unmatched_treated=tuple(section.unmatched_treated),
        periods_matched=section.periods_matched,
        uncertainty=section.uncertainty,
        standard_error=section.standard_error,
        caliper=section.caliper,
    )


def _iv_bounds_from_raw(raw: Any) -> IvBoundsEstimate | None:
    section = getattr(raw, "iv_bounds", None)
    if section is None:
        return None
    interval = section.interval
    posterior = section.posterior
    return IvBoundsEstimate(
        lower=section.lower,
        upper=section.upper,
        cells=tuple(section.cells),
        arm_weights=tuple(section.arm_weights),
        rows=section.rows,
        rows_dropped=section.rows_dropped,
        uncertainty=interval.method
        if interval is not None
        else "identified_set_without_sampling_interval",
        interval=None
        if interval is None
        else IvBoundsInterval(
            level=interval.level,
            lower=interval.lower,
            upper=interval.upper,
            lower_se=interval.lower_se,
            upper_se=interval.upper_se,
            critical_value=interval.critical_value,
            width_retained=interval.width_retained,
            replicates=interval.replicates,
            infeasible=interval.infeasible,
            method=interval.method,
        ),
        posterior=None
        if posterior is None
        else IvBoundsPosterior(
            n_draws=posterior.n_draws,
            infeasible_draws=posterior.infeasible_draws,
            lower_mean=posterior.lower_mean,
            upper_mean=posterior.upper_mean,
            lower_sd=posterior.lower_sd,
            upper_sd=posterior.upper_sd,
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


def _execution_slots(execution: Any | None) -> ReasoningSlots | None:
    """The reasoning slots of a frozen execution, or ``None`` when it has no contract.

    A functional-validation result (a vector refitted on perturbed data) is not described by
    the handle's contract, which names a run with no validation, so it carries no slots.
    """
    if execution is None or not execution.has_execution_contract():
        return None
    # Completion-level response checks are inspectable but cannot be sealed as
    # whole-result evidence. Keep the native result without inventing a claim.
    if getattr(execution, "response_checks_are_scoped", lambda: False)():
        return None
    # Capture the native contract and scientific evidence once, at execution.
    # Inspection reads these immutable presentation records; it never exports an
    # artifact or reaches back into the native execution to materialize a payload.
    record = json.loads(execution.execution_record_json())
    return ReasoningSlots.from_result_section(record["contract"], record["evidence"])


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
    slots = _execution_slots(execution)

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
            predictive_replicates=_predictive_replicates_from_raw(raw),
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
            dispersion_p_value=getattr(raw, f"{prefix}_dispersion_p_value", None),
            noise=getattr(raw, f"{prefix}_noise", None),
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
            overlap_excluded_fraction=getattr(sec_estimate, "overlap_excluded_fraction", None),
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
            sampling_weights=_sampling_weights_view(
                getattr(sec_estimate, "sampling_weights", None)
            ),
            scenario_effects=getattr(sec_estimate, "scenario_effects", None),
            scenario_intervals=getattr(sec_estimate, "scenario_intervals", None),
            simultaneous_interval=getattr(sec_estimate, "simultaneous_interval", None),
            adjusted_p_values=getattr(sec_estimate, "adjusted_p_values", None),
            family_contrast=getattr(sec_estimate, "family_contrast", None),
            family_contrast_interval=getattr(sec_estimate, "family_contrast_interval", None),
            family_contrast_posterior=getattr(sec_estimate, "family_contrast_posterior", None),
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
        counterfactual_contrasts=_counterfactual_contrasts_from_raw(raw),
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
            resampling=_resampling_from_raw(getattr(sec_performance, "resampling", None)),
        ),
        diagnostics=list(raw.diagnostics),
        route_diagnostics=tuple(
            {"route": route, "outcome": outcome, "failed": failed, "unevaluable_mass": mass}
            for route, outcome, failed, mass in getattr(raw, "route_diagnostics", ())
        ),
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
        structural_aggregation_policy=getattr(raw, "structural_aggregation_policy", None),
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
        missing_data=_missing_data_from_raw(raw, prepared),
        interval_bindings=_interval_bindings_from_raw(raw),
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
        posterior_row=_posterior_row_from_raw(raw),
        longitudinal_regime=_longitudinal_regime_from_raw(raw),
        sequential_contrast=_sequential_contrast_from_raw(raw),
        history_match=_history_match_from_raw(raw),
        iv_bounds=_iv_bounds_from_raw(raw),
        association=_association_from_raw(raw),
        quasi_rollout=_quasi_rollout_from_raw(raw),
        temporal_design=_temporal_design_from_raw(raw),
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


_WAVE_FIELDS = (
    "wave",
    "first_replicate",
    "attempted",
    "ok",
    "estimator_errors",
    "non_finite",
    "declined",
)


def _resampling_from_raw(section: Any) -> dict[str, Any] | None:
    """The resampling section of a requested bootstrap interval as plain data."""
    if section is None:
        return None
    return {
        "level": section.level,
        "wave_size": section.wave_size,
        "min_replicates": section.min_replicates,
        "max_replicates": section.max_replicates,
        "se_rel_tol": section.se_rel_tol,
        "half_width_rel_tol": section.half_width_rel_tol,
        "interval": None if section.interval is None else tuple(section.interval),
        "stop": section.stop,
        "attempted": section.attempted,
        "waves": tuple(dict(zip(_WAVE_FIELDS, wave, strict=True)) for wave in section.waves),
        "se_rel_mc_error": section.se_rel_mc_error,
        "half_width_rel_mc_error": section.half_width_rel_mc_error,
    }
