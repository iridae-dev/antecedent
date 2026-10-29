//! Wire sections of an executed result's non-scalar payloads.
//!
//! Each estimate family converts to its portable wire section here, so the
//! result body is assembled by dispatch rather than by inline conversion. An
//! estimate whose section carries the graphless license the executing result
//! resolved converts through [`LicensedWire`]; the rest are plain `From`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_io::analysis_result_artifact::{
    ContinuousDosePointWire, ContinuousDoseResponseWire, DosePolicyValueWire,
    FixedCandidateRegretWire, MultiActionCateWire,
};
use antecedent_io::{
    DistributionAtomWire, HistoryMatchWire, InterferenceInferenceWire,
    InterferencePointwiseIntervalWire, InterventionalDistributionWire, LocalPolynomialRatioWire,
    LongitudinalRegimeWire, PanelDidWire, PolicyValueWire, RandomizedEffectWire,
    SequentialContrastWire, SurvivalDifferenceBandWire, SurvivalWire, SyntheticControlWire,
    SyntheticDidWire, UpliftBinWire, ValueWire,
};

use antecedent_estimate::InterventionalDistributionEstimate;

use crate::result::{
    ContinuousDoseResponseEstimate, HistoryMatchEstimate, InterferenceInference,
    LocalPolynomialRatioEstimate, LongitudinalRegimeEstimate, PanelDidEstimate,
    PolicyValueEstimate, RandomizedEffectEstimate, SequentialContrastEstimate, SurvivalEstimate,
    SyntheticControlEstimate, SyntheticDidEstimate,
};
use crate::support::CellStatus;

/// Wire section of an estimate whose graphless license is decided by the
/// executing result rather than by the estimate itself.
pub(super) trait LicensedWire {
    type Wire;

    /// `support` is the result's graphless support status.
    fn to_wire(&self, support: Option<CellStatus>) -> Self::Wire;
}

fn support_label(support: Option<CellStatus>) -> Option<String> {
    support.map(CellStatus::as_str).map(str::to_string)
}

fn unit_values(values: &[(impl ToString, f64)]) -> Vec<(String, f64)> {
    values.iter().map(|(unit, value)| (unit.to_string(), *value)).collect()
}

impl LicensedWire for PolicyValueEstimate {
    type Wire = PolicyValueWire;

    fn to_wire(&self, support: Option<CellStatus>) -> PolicyValueWire {
        PolicyValueWire {
            policy_value: self.policy_value,
            reference_value: self.reference_value,
            incremental_value: self.incremental_value,
            relative_value_gap: self.relative_value_gap,
            treatment_rate: self.treatment_rate,
            total_cost: self.total_cost,
            policy_standard_error: self.policy_standard_error,
            reference_standard_error: self.reference_standard_error,
            incremental_standard_error: self.incremental_standard_error,
            policy_interval_95: self.policy_interval_95,
            incremental_interval_95: self.incremental_interval_95,
            prediction_ownership: self.prediction_ownership.to_string(),
            propensity_min: self.propensity_min,
            propensity_max: self.propensity_max,
            uncertainty: self.uncertainty.to_string(),
            graphless_support_status: support_label(support),
            uplift_bins: self
                .uplift_bins
                .iter()
                .map(|bin| UpliftBinWire {
                    rank: bin.rank,
                    effect: bin.effect,
                    standard_error: bin.standard_error,
                    evaluation_rows: bin.evaluation_rows,
                    interval_95: bin.interval_95,
                })
                .collect(),
            multi_action_cate: self
                .multi_action_cate
                .iter()
                .map(|point| MultiActionCateWire {
                    group: point.group.clone(),
                    action: point.action.clone(),
                    effect: point.effect,
                    standard_error: point.standard_error,
                    interval_95: point.interval_95,
                    evaluation_rows: point.evaluation_rows,
                    observed_action_rows: point.observed_action_rows,
                    observed_control_rows: point.observed_control_rows,
                })
                .collect(),
            regret: self.regret.as_ref().map(|regret| FixedCandidateRegretWire {
                candidate_values: regret.candidate_values.clone(),
                contrast_standard_errors: regret.contrast_standard_errors.clone(),
                regret: regret.regret,
                interval_95: regret.interval_95,
                selected_index: regret.selected_index,
            }),
        }
    }
}

impl LicensedWire for ContinuousDoseResponseEstimate {
    type Wire = ContinuousDoseResponseWire;

    fn to_wire(&self, support: Option<CellStatus>) -> ContinuousDoseResponseWire {
        ContinuousDoseResponseWire {
            points: self
                .points
                .iter()
                .map(|point| ContinuousDosePointWire {
                    baseline_group: point.baseline_group.clone(),
                    target_dose: point.target_dose,
                    response: point.response,
                    local_rows: point.local_rows,
                    effective_sample_size: point.effective_sample_size,
                    minimum_dose_density: point.minimum_dose_density,
                    maximum_normalized_weight: point.maximum_normalized_weight,
                    local_outcome_sd: point.local_outcome_sd,
                })
                .collect(),
            bandwidth: self.bandwidth,
            density_provenance: self.density_provenance.to_string(),
            uncertainty: self.uncertainty.to_string(),
            graphless_support_status: support_label(support),
            fixed_policy: self.fixed_policy.as_ref().map(|value| DosePolicyValueWire {
                policy_doses: value.policy_doses.clone(),
                reference_doses: value.reference_doses.clone(),
                policy_value: value.policy_value,
                reference_value: value.reference_value,
                incremental_value: value.incremental_value,
                policy_variance: value.policy_variance,
                reference_variance: value.reference_variance,
                incremental_variance: value.incremental_variance,
                policy_interval_95: value.policy_interval_95,
                reference_interval_95: value.reference_interval_95,
                incremental_interval_95: value.incremental_interval_95,
                minimum_local_rows: value.minimum_local_rows,
                minimum_effective_sample_size: value.minimum_effective_sample_size,
                maximum_normalized_weight: value.maximum_normalized_weight,
                minimum_dose_density: value.minimum_dose_density,
            }),
        }
    }
}

impl LicensedWire for PanelDidEstimate {
    type Wire = PanelDidWire;

    fn to_wire(&self, support: Option<CellStatus>) -> PanelDidWire {
        PanelDidWire {
            effect: self.effect,
            // An augmented panel has a standard error only when its whole-cluster resample
            // published an interval; otherwise the field is a placeholder.
            standard_error: if self.augmented.is_some() && self.interval_95.is_none() {
                None
            } else {
                Some(self.standard_error)
            },
            interval_95: self.interval_95,
            treated_subjects: self.treated_subjects,
            comparison_subjects: self.comparison_subjects,
            clusters: self.clusters,
            uncertainty: self.uncertainty.to_string(),
            graphless_support_status: support_label(support),
            event_time_effects: self
                .event_time_effects
                .iter()
                .map(|effect| {
                    (
                        effect.cohort,
                        effect.period,
                        effect.event_time,
                        effect.effect,
                        effect.treated_subjects,
                        effect.comparison_subjects,
                        effect.standard_error,
                        effect.clusters,
                    )
                })
                .collect(),
            event_time_intervals_95: self.event_time_intervals_95.to_vec(),
            augmented: self.augmented,
        }
    }
}

impl LicensedWire for LocalPolynomialRatioEstimate {
    type Wire = LocalPolynomialRatioWire;

    fn to_wire(&self, support: Option<CellStatus>) -> LocalPolynomialRatioWire {
        LocalPolynomialRatioWire {
            effect: self.effect,
            reduced_form: self.reduced_form,
            first_stage: self.first_stage,
            cutoff: self.cutoff,
            bandwidth: self.bandwidth,
            kink: self.kink,
            n_left: self.n_left,
            n_right: self.n_right,
            standard_error: self.standard_error,
            ci_lower: self.ci_lower,
            ci_upper: self.ci_upper,
            reduced_form_standard_error: self.reduced_form_standard_error,
            first_stage_standard_error: self.first_stage_standard_error,
            uncertainty: self.uncertainty.to_string(),
            graphless_support_status: support_label(support),
        }
    }
}

impl LicensedWire for RandomizedEffectEstimate {
    type Wire = RandomizedEffectWire;

    fn to_wire(&self, support: Option<CellStatus>) -> RandomizedEffectWire {
        RandomizedEffectWire {
            effect: self.effect,
            estimand: self.estimand.to_string(),
            intention_to_treat_effect: self.intention_to_treat_effect,
            first_stage_effect: self.first_stage_effect,
            received_treatment: self.received_treatment.as_ref().map(|receipt| receipt.to_vec()),
            randomization_p_value: self.randomization_p_value,
            randomization_allocations: self.randomization_allocations,
            second_factor_effect: self.second_factor_effect,
            factorial_interaction: self.factorial_interaction,
            second_factor_variance: self.second_factor_variance,
            factorial_interaction_variance: self.factorial_interaction_variance,
            multi_arm_values: self
                .multi_arm_values
                .iter()
                .map(|(label, value, variance, arm_support)| {
                    (label.to_string(), *value, *variance, *arm_support)
                })
                .collect(),
            variance: self.variance_upper_bound,
            standard_error: self.standard_error,
            interval_95: self.interval_95,
            second_factor_interval_95: self.second_factor_interval_95,
            factorial_interaction_interval_95: self.factorial_interaction_interval_95,
            multi_arm_intervals_95: self.multi_arm_intervals_95.to_vec(),
            assignment_design: self.assignment_design.to_string(),
            assignment_units: self.assignment_units.iter().map(ToString::to_string).collect(),
            outcome_units: self.outcome_units.iter().map(ToString::to_string).collect(),
            blocks: self.blocks.iter().map(ToString::to_string).collect(),
            periods: self.periods.iter().map(ToString::to_string).collect(),
            treatment_arms: (self.treatment_arms.0.to_string(), self.treatment_arms.1.to_string()),
            control_units: self.control_units,
            treatment_units: self.treatment_units,
            minimum_assignment_probability: self.minimum_assignment_probability,
            uncertainty: self.uncertainty.to_string(),
            graphless_support_status: support_label(support),
        }
    }
}

impl LicensedWire for SurvivalEstimate {
    type Wire = SurvivalWire;

    fn to_wire(&self, support: Option<CellStatus>) -> SurvivalWire {
        SurvivalWire {
            times: self.times.to_vec(),
            control: self.control.to_vec(),
            treated: self.treated.to_vec(),
            rmst_control: self.rmst_control,
            rmst_treated: self.rmst_treated,
            target_cause: self.target_cause,
            tau: self.tau,
            minimum_event_risk_set: self.minimum_event_risk_set,
            uncertainty: self.uncertainty.to_string(),
            rmst_difference_interval: self.rmst_difference_interval,
            difference_at_tau_interval: self.difference_at_tau_interval,
            bootstrap_replicates_requested: self.bootstrap_replicates_requested,
            bootstrap_replicates_ok: self.bootstrap_replicates_ok,
            assignment_counts: Some(self.assignment_counts),
            graphless_support_status: support_label(support),
            censoring_survival_provenance: self
                .censoring_survival_provenance
                .as_ref()
                .map(ToString::to_string),
            difference_band: self.difference_band.as_ref().map(|band| SurvivalDifferenceBandWire {
                times: band.times.to_vec(),
                difference: band.difference.to_vec(),
                lower: band.lower.to_vec(),
                upper: band.upper.to_vec(),
                replicates_ok: band.replicates_ok,
                graphless_support_status: support_label(band.support_status),
            }),
            band_unavailable_reason: self.band_unavailable_reason.as_ref().map(ToString::to_string),
        }
    }
}

impl From<&SyntheticControlEstimate> for SyntheticControlWire {
    fn from(fit: &SyntheticControlEstimate) -> Self {
        Self {
            effect: fit.effect,
            pre_treatment_rmse: fit.pre_treatment_rmse,
            donor_weights: unit_values(&fit.donor_weights),
            placebo_effects: fit.placebo_effects.to_vec(),
            placebo_rank: fit.placebo_rank,
            effective_donors: fit.effective_donors,
            n_pre_periods: fit.n_pre_periods,
            n_post_periods: fit.n_post_periods,
            uncertainty: fit.uncertainty.to_string(),
            randomization_p_value: fit.randomization_p_value,
            randomization_null_effect: fit.randomization_null_effect,
            randomization_statistics: unit_values(&fit.randomization_statistics),
            unadjusted_effect: fit.unadjusted_effect,
            outcome_model_correction: fit.outcome_model_correction,
            augmentation_ridge: fit.augmentation_ridge,
            observed_window: fit.observed_window.as_ref().map(|window| {
                antecedent_io::analysis_result_artifact::ObservedWindowWire {
                    pre_periods: window.pre_periods.to_vec(),
                    post_periods: window.post_periods.to_vec(),
                    donors_used: window.donors_used,
                    donors_dropped: unit_counts(&window.donors_dropped),
                    unread_rows: window.unread_rows,
                }
            }),
        }
    }
}

fn unit_counts(values: &[(std::sync::Arc<str>, usize)]) -> Vec<(String, usize)> {
    values.iter().map(|(unit, count)| (unit.to_string(), *count)).collect()
}

impl From<&SyntheticDidEstimate> for SyntheticDidWire {
    fn from(fit: &SyntheticDidEstimate) -> Self {
        Self {
            effect: fit.effect,
            pre_treatment_rmse: fit.pre_treatment_rmse,
            donor_weights: unit_values(&fit.donor_weights),
            time_weights: fit.time_weights.to_vec(),
            n_donors: fit.n_donors,
            n_pre_periods: fit.n_pre_periods,
            n_post_periods: fit.n_post_periods,
            uncertainty: fit.uncertainty.to_string(),
            randomization_p_value: fit.randomization_p_value,
            randomization_null_effect: fit.randomization_null_effect,
            randomization_statistics: unit_values(&fit.randomization_statistics),
        }
    }
}

impl From<&LongitudinalRegimeEstimate> for LongitudinalRegimeWire {
    fn from(regime: &LongitudinalRegimeEstimate) -> Self {
        let vec_or_empty = |values: &Option<std::sync::Arc<[f64]>>| {
            values.as_ref().map_or_else(Vec::new, |v| v.to_vec())
        };
        Self {
            method: regime.method.to_string(),
            graphless_support_status: support_label(regime.graphless_support_status),
            rule_id: regime.rule_id.as_ref().map(ToString::to_string),
            rule_version: regime.rule_version.as_ref().map(ToString::to_string),
            rule_provenance: regime.rule_provenance.as_ref().map(ToString::to_string),
            value: regime.value,
            value_standard_error: regime.value_standard_error,
            value_interval_95: regime.value_interval_95,
            period_intervals_95: regime
                .period_intervals_95
                .as_ref()
                .map_or_else(Vec::new, |v| v.to_vec()),
            interval_reason: regime.interval_reason.as_ref().map(ToString::to_string),
            effective_sample_size: regime.effective_sample_size,
            matched_observed_fraction: regime.matched_observed_fraction,
            maximum_weight: regime.maximum_weight,
            minimum_action_probability: regime.minimum_action_probability,
            minimum_censoring_probability: regime.minimum_censoring_probability,
            uncertainty: regime.uncertainty.to_string(),
            probability_ownership: regime.probability_ownership.to_string(),
            period_effects: vec_or_empty(&regime.period_effects),
            standard_errors: vec_or_empty(&regime.standard_errors),
            stabilizing_numerator_probabilities: vec_or_empty(
                &regime.stabilizing_numerator_probabilities,
            ),
            observed_subjects: regime.observed_subjects.unwrap_or(0),
        }
    }
}

impl From<&HistoryMatchEstimate> for HistoryMatchWire {
    fn from(fit: &HistoryMatchEstimate) -> Self {
        Self {
            contrast: fit.contrast,
            standard_error: fit.standard_error,
            weight_total: fit.weight_total,
            matched_queries: fit.matched_queries.to_vec(),
            matched_donors: fit.matched_donors.to_vec(),
            differences: fit.differences.to_vec(),
            pair_weights: fit.pair_weights.to_vec(),
            query_variances: fit.query_variances.to_vec(),
            donor_variances: fit.donor_variances.to_vec(),
            unmatched_treated: fit.unmatched_treated.to_vec(),
            periods_matched: fit.periods_matched,
            pair_periods: fit.pair_periods.to_vec(),
            serial_standard_error: fit.serial_standard_error,
            interval_lower: fit.interval_lower,
            interval_upper: fit.interval_upper,
            serial_bandwidth: fit.serial_bandwidth,
            caliper: fit.caliper,
            uncertainty: fit.uncertainty.to_string(),
            calibration: "unmeasured".to_owned(),
            licensed: false,
        }
    }
}

impl From<&SequentialContrastEstimate> for SequentialContrastWire {
    fn from(fit: &SequentialContrastEstimate) -> Self {
        Self {
            active_regime: fit.active_regime,
            control_regime: fit.control_regime,
            contrast: fit.contrast,
            active_mean: fit.active_mean,
            control_mean: fit.control_mean,
            evaluated_rows: fit.evaluated_rows,
            minimum_action_probability: fit.minimum_action_probability,
            min_probability: fit.min_probability,
            standard_error: fit.standard_error,
            interval: fit.interval,
            coverage: fit.coverage,
            active_scores: fit.active_scores.to_vec(),
            control_scores: fit.control_scores.to_vec(),
            contrast_scores: fit.contrast_scores.to_vec(),
            uncertainty: fit.uncertainty.to_string(),
            calibration: "unmeasured".to_owned(),
            licensed: false,
        }
    }
}

impl From<&InterferenceInference> for InterferenceInferenceWire {
    fn from(inference: &InterferenceInference) -> Self {
        Self {
            method: inference.method.into(),
            graphless_support_status: support_label(inference.graphless_support_status),
            interval: inference.interval.as_ref().map(|interval| {
                InterferencePointwiseIntervalWire {
                    lower: interval.lower,
                    upper: interval.upper,
                    standard_error: interval.standard_error,
                    degrees_of_freedom: interval.degrees_of_freedom,
                    first_stage_arm_clusters: interval.first_stage_arm_clusters,
                }
            }),
            interval_unavailable_reason: inference.interval_unavailable_reason.map(str::to_string),
            from_exposed_units: inference.from_exposed_units,
            to_exposed_units: inference.to_exposed_units,
            from_exposed_clusters: inference.from_exposed_clusters,
            to_exposed_clusters: inference.to_exposed_clusters,
        }
    }
}

/// Free function: both the estimate and its wire live in other crates.
pub(super) fn distribution_wire(
    distribution: &InterventionalDistributionEstimate,
) -> InterventionalDistributionWire {
    InterventionalDistributionWire {
        atoms: distribution
            .atoms
            .iter()
            .map(|atom| DistributionAtomWire {
                outcomes: atom
                    .outcomes
                    .iter()
                    .map(|(variable, value)| (variable.raw(), ValueWire::from_value(value)))
                    .collect(),
                conditioning: atom
                    .conditioning
                    .iter()
                    .map(|(variable, value)| (variable.raw(), ValueWire::from_value(value)))
                    .collect(),
                probability: atom.probability,
            })
            .collect(),
    }
}

impl From<&crate::QuasiRolloutEstimate>
    for antecedent_io::analysis_result_artifact::QuasiRolloutWire
{
    fn from(fit: &crate::QuasiRolloutEstimate) -> Self {
        Self {
            design: fit.design.to_string(),
            method: fit.method.to_string(),
            effects: fit
                .effects
                .iter()
                .map(|effect| antecedent_io::analysis_result_artifact::DesignEffectWire {
                    kind: effect.kind.as_str().to_string(),
                    group: effect.group,
                    period: effect.period,
                    event_time: effect.event_time,
                    unit: effect.unit.clone(),
                    estimate: effect.estimate,
                    standard_error: effect.standard_error,
                    interval: effect.interval,
                    band: effect.band,
                    treated: effect.treated,
                    comparison: effect.comparison,
                })
                .collect(),
            uncertainty: fit.uncertainty.to_string(),
            calibration: fit.calibration.to_string(),
            replicates_requested: fit.replicates_requested,
            replicates_ok: fit.replicates_ok,
            replicates_failed: fit.replicates_failed,
            critical_value: fit.critical_value,
            diagnostics: fit
                .diagnostics
                .iter()
                .map(|(name, value)| (name.to_string(), *value))
                .collect(),
            licensed: fit.licensed,
        }
    }
}

impl From<&crate::TemporalDesignEstimate>
    for antecedent_io::analysis_result_artifact::TemporalDesignWire
{
    fn from(fit: &crate::TemporalDesignEstimate) -> Self {
        Self {
            design: fit.design.to_string(),
            functional: fit.functional.to_string(),
            coordinates: fit.coordinates.iter().map(ToString::to_string).collect(),
            estimate: fit.estimate.to_vec(),
            standard_error: fit.standard_error.to_vec(),
            lower: fit.lower.to_vec(),
            upper: fit.upper.to_vec(),
            covariance: fit.covariance.as_ref().map(|values| values.to_vec()),
            band_coordinates: fit.band_coordinates.iter().map(|index| *index as u64).collect(),
            band_critical_value: fit.band_critical_value,
            band_lower: fit.band_lower.as_ref().map(|values| values.to_vec()),
            band_upper: fit.band_upper.as_ref().map(|values| values.to_vec()),
            holm_adjusted_p: fit.holm_adjusted_p.as_ref().map(|values| values.to_vec()),
            smoothing_bias_bound: fit.smoothing_bias_bound.as_ref().map(|values| values.to_vec()),
            pairs: fit.pairs,
            clusters: fit.clusters,
            unpaired: fit.unpaired,
            minimum_probability: fit
                .minimum_probability
                .is_finite()
                .then_some(fit.minimum_probability),
            replicates: fit.replicates.map(|(ok, failed)| [ok, failed]),
            draws: fit.draws,
            uncertainty: fit.uncertainty.to_string(),
            calibration: fit.calibration.to_string(),
            licensed: fit.licensed,
        }
    }
}

/// The artifact section of a compiled instrumental-variable bounds estimate.
pub(crate) fn iv_bounds_wire(
    fit: &antecedent_estimate::IvBoundsEstimate,
) -> antecedent_io::analysis_result_artifact::IvBoundsWire {
    {
        use antecedent_io::analysis_result_artifact::{
            IV_BOUNDS_POINT_UNCERTAINTY, IvBoundsIntervalWire, IvBoundsPosteriorWire,
        };
        let interval = fit.interval.map(|interval| IvBoundsIntervalWire {
            level: interval.level,
            lower: interval.lower,
            upper: interval.upper,
            lower_se: interval.lower_se,
            upper_se: interval.upper_se,
            critical_value: interval.critical_value,
            width_retained: interval.width_retained,
            replicates: interval.replicates as u64,
            infeasible: interval.infeasible as u64,
            method: interval.method.as_str().to_owned(),
        });
        let uncertainty = interval
            .as_ref()
            .map_or_else(|| IV_BOUNDS_POINT_UNCERTAINTY.to_owned(), |i| i.method.clone());
        antecedent_io::analysis_result_artifact::IvBoundsWire {
            lower: fit.lower,
            upper: fit.upper,
            cells: fit.law.cells.iter().flatten().copied().collect(),
            arm_weights: fit.arm_weights.to_vec(),
            rows: fit.rows as u64,
            rows_dropped: fit.rows_dropped as u64,
            interval,
            posterior: fit.posterior.map(|posterior| IvBoundsPosteriorWire {
                n_draws: posterior.n_draws as u64,
                infeasible_draws: posterior.infeasible_draws as u64,
                lower_mean: posterior.lower_mean,
                upper_mean: posterior.upper_mean,
                lower_sd: posterior.lower_sd,
                upper_sd: posterior.upper_sd,
            }),
            uncertainty,
            calibration: "unmeasured".to_owned(),
            licensed: false,
        }
    }
}

/// The artifact section of a compiled noncausal association.
pub(crate) fn association_wire(
    payload: &crate::analysis::association::AssociationPayload,
) -> antecedent_io::analysis_result_artifact::AssociationWire {
    use antecedent_io::analysis_result_artifact::{
        AssociationPosteriorWire, AssociationPredictionWire, AssociationResamplingWire,
        AssociationTermWire, AssociationWire,
    };
    let estimate = &payload.estimate;
    let terms = estimate
        .terms
        .iter()
        .map(|term| AssociationTermWire {
            label: term.label.clone(),
            scale: term.scale.to_owned(),
            estimate: term.estimate,
            standard_error: term.standard_error,
            standard_error_method: term.standard_error_method.map(str::to_owned),
            spread_lower: term.spread.map(|(lower, _)| lower),
            spread_upper: term.spread.map(|(_, upper)| upper),
            posterior_mean: term.posterior.map(|posterior| posterior.mean),
            posterior_sd: term.posterior.map(|posterior| posterior.sd),
            posterior_lower: term.posterior.map(|posterior| posterior.lower),
            posterior_upper: term.posterior.map(|posterior| posterior.upper),
            rows: term.rows as u64,
        })
        .collect();
    let uncertainty = payload.uncertainty_label();
    AssociationWire {
        analysis: estimate.selection.analysis.as_str().to_owned(),
        rule: estimate.selection.rule.to_owned(),
        requested: estimate.selection.requested.to_owned(),
        outcome_kind: estimate.selection.outcome_kind.as_str().to_owned(),
        modality: estimate.selection.modality.as_str().to_owned(),
        terms,
        rows: estimate.rows as u64,
        rows_dropped: estimate.rows_dropped as u64,
        resampling: estimate.resampling.map(|record| AssociationResamplingWire {
            method: record.method.to_owned(),
            unit: record.unit.to_owned(),
            replicates_requested: u64::from(record.replicates_requested),
            replicates_ok: u64::from(record.replicates_ok),
            block_length: record.block_length.map(|length| length as u64),
            spread_level: antecedent_estimate::SPREAD_LEVEL,
        }),
        posterior: estimate.posterior.map(|record| AssociationPosteriorWire {
            method: record.method.to_owned(),
            draws: record.draws as u64,
            level: antecedent_estimate::SPREAD_LEVEL,
        }),
        prediction: payload.scores.as_ref().zip(estimate.prediction.as_ref()).map(
            |(scores, fit)| AssociationPredictionWire {
                folds: u64::from(fit.folds),
                fold_design: fit.fold_design.to_owned(),
                cases: scores.cases as u64,
                score: scores.score.to_owned(),
                mean_score: scores.mean_score,
                baseline_score: scores.baseline_score,
                mean_crps: scores.mean_crps,
                calibration: crate::analysis::association::ASSOCIATION_CALIBRATION.to_owned(),
            },
        ),
        assumptions: estimate.assumptions.iter().map(|premise| (*premise).to_owned()).collect(),
        refused_after: payload.refused_after.as_deref().map(str::to_owned),
        uncertainty: uncertainty.to_owned(),
        calibration: crate::analysis::association::ASSOCIATION_CALIBRATION.to_owned(),
        licensed: false,
        ranks_actions: false,
        is_intervention_effect: false,
    }
}
