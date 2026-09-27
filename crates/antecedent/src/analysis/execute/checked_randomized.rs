//! Prepared native execution for randomized two-arm ITT.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSource, AssumptionStatus,
};

/// Frozen randomized-design analysis. Assignment metadata is part of query identity.
#[derive(Clone)]
pub(crate) struct CheckedRandomizedOperation {
    query: antecedent_core::RandomizedEffectQuery,
    source_schema: antecedent_core::CausalSchema,
    source_rows: usize,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    physical: PhysicalExecutionPlan,
    result_context: IdentifiedResultContext,
}

impl std::fmt::Debug for CheckedRandomizedOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CheckedRandomizedOperation")
            .field("query", &self.query)
            .field("source_rows", &self.source_rows)
            .finish_non_exhaustive()
    }
}

impl CheckedRandomizedOperation {
    pub(crate) fn checked(
        study: &Study,
        data: &TabularData,
        physical: &PhysicalExecutionPlan,
    ) -> Result<Self, CausalError> {
        let CausalQuery::RandomizedEffect(query) = &study.query else {
            return Err(CausalError::Compile {
                message: "randomized execution requires RandomizedEffectQuery".into(),
            });
        };
        query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
        if study.graph.class() != GraphClass::RandomizedTrial
            || study.structure_source != crate::support::StructureSource::RandomizedTrial
            || !matches!(study.inference, InferenceMode::Frequentist)
            || study.refute != RefuteSuite::None
            || study.bootstrap_replicates != 0
            || !study.custom_validators.is_empty()
            || study.graph_posterior.is_some()
            || study.tiered.is_some()
        {
            return Err(CausalError::Unsupported {
                message: "randomized ITT supports Bernoulli, complete, stratified, cluster, and switchback estimation only, without a graph, validation suite, or bootstrap",
            });
        }
        if query.realized_assignment.len() != data.row_count() {
            return Err(CausalError::Compile {
                message: "randomized assignment metadata must align with the table rows".into(),
            });
        }
        data.schema()
            .get(query.outcome)
            .map_err(|e| CausalError::Compile { message: e.to_string() })?;
        if let Some((covariate, _)) = query.fixed_cuped {
            data.schema().get(covariate)
                .map_err(|e| CausalError::Compile { message: e.to_string() })?;
        }
        for &covariate in query.ancova_covariates.iter() {
            data.schema().get(covariate)
                .map_err(|e| CausalError::Compile { message: e.to_string() })?;
        }
        if physical.logical.query != study.query
            || physical.logical.record.identifier.as_deref()
                != Some(IdentifierId::RandomizedDesign.as_str())
            || physical.logical.record.estimator.as_deref()
                != Some(randomized_estimator_id(query).as_str())
        {
            return Err(CausalError::Compile {
                message: "randomized target or design differs from its compiled plan".into(),
            });
        }
        let (identification, estimand) = randomized_identification(query);
        Ok(Self {
            query: query.clone(),
            source_schema: data.schema().clone(),
            source_rows: data.row_count(),
            identification,
            estimand,
            physical: physical.clone(),
            result_context: IdentifiedResultContext::from_study(study),
        })
    }

    pub(crate) fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if data.schema() != &self.source_schema || data.row_count() != self.source_rows {
            return Err(CausalError::Unsupported {
                message: "randomized ITT refresh requires the prepared schema and row order",
            });
        }
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled {
                stage: super::super::stage::STAGE_ESTIMATE_POINT,
            });
        }
        self.query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let outcomes = match data.column(self.query.outcome).map_err(CausalError::from)? {
            antecedent_data::ColumnView::Float64(column) => column.values.as_slice(),
            _ => {
                return Err(CausalError::Unsupported {
                    message: "randomized ITT outcome must be continuous",
                });
            }
        };
        let n = outcomes.len();
        if outcomes.iter().any(|y| !y.is_finite()) {
            return Err(CausalError::Unsupported {
                message: "randomized ITT requires complete finite outcomes",
            });
        }
        let adjusted_outcomes;
        let outcomes = if let Some((covariate, coefficient)) = self.query.fixed_cuped {
            let values = match data.column(covariate).map_err(CausalError::from)? {
                antecedent_data::ColumnView::Float64(column) => column.values.as_slice(),
                _ => return Err(CausalError::Unsupported {
                    message: "fixed CUPED pre-assignment covariate must be continuous",
                }),
            };
            if values.len() != n || values.iter().any(|x| !x.is_finite()) {
                return Err(CausalError::Unsupported {
                    message: "fixed CUPED requires a complete finite pre-assignment covariate",
                });
            }
            adjusted_outcomes = outcomes.iter().zip(values.iter())
                .map(|(y, x)| y - coefficient * x).collect::<Vec<_>>();
            adjusted_outcomes.as_slice()
        } else {
            outcomes
        };
        let sample_variance = |values: &[f64]| -> f64 {
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            values.iter().map(|value| (value - mean).powi(2)).sum::<f64>()
                / (values.len() - 1) as f64
        };
        let mut complier_components = None;
        let mut factorial_contrasts = None;
        #[allow(clippy::type_complexity, reason = "labeled multi-arm value tuple mirrors the public result field type")]
        let mut multi_arm_values: Arc<[(Arc<str>, f64, f64, usize)]> = Arc::from([]);
        let mut interval_95 = None;
        let mut second_factor_interval_95 = None;
        let mut factorial_interaction_interval_95 = None;
        let mut multi_arm_intervals_95: Arc<[Option<[f64; 2]>]> = Arc::from([]);
        let mut interval_standard_error = None;
        let (
            effect,
            variance,
            control_units,
            treatment_units,
            blocks,
            periods,
            assignment_design,
            uncertainty,
            diagnostic,
        ) = if !self.query.ancova_covariates.is_empty() {
            let covariates = self.query.ancova_covariates.iter().map(|&id| {
                match data.column(id).map_err(CausalError::from)? {
                    antecedent_data::ColumnView::Float64(column) => Ok(column.values.as_slice().to_vec()),
                    _ => Err(CausalError::Unsupported { message: "ANCOVA pre-assignment covariates must be continuous" }),
                }
            }).collect::<Result<Vec<_>, _>>()?;
            let refs = covariates.iter().map(Vec::as_slice).collect::<Vec<_>>();
            let fit = antecedent_estimate::ancova::fit_ancova(outcomes, &self.query.realized_assignment, &refs)
                .map_err(|message| CausalError::Unsupported { message })?;
            interval_95 = antecedent_estimate::ancova::calibrated_bernoulli_interval_95(
                &fit, self.query.assignment_probabilities[0]);
            if interval_95.is_some() {
                interval_standard_error = Some(fit.hc0_variance.sqrt());
            }
            (fit.effect, fit.hc0_variance, fit.control, fit.treated, Arc::from([]), Arc::from([]),
                Arc::<str>::from("bernoulli"),
                Arc::<str>::from("bernoulli_ancova_hc0_variance_no_interval"),
                "OLS ANCOVA with pre-assignment covariates and independent-row HC0 sandwich variance; no calibrated confidence interval")
        } else if let Some(received) = &self.query.received_treatment {
            let mut outcome_scores = Vec::with_capacity(n);
            let mut receipt_scores = Vec::with_capacity(n);
            let mut controls = 0;
            let mut treated = 0;
            for i in 0..n {
                let probability = self.query.assignment_probabilities[i];
                let sign = if self.query.realized_assignment[i] {
                    treated += 1;
                    1.0 / probability
                } else {
                    controls += 1;
                    -1.0 / (1.0 - probability)
                };
                outcome_scores.push(sign * outcomes[i]);
                receipt_scores.push(sign * f64::from(received[i]));
            }
            let outcome_itt = outcome_scores.iter().sum::<f64>() / n as f64;
            let first_stage = receipt_scores.iter().sum::<f64>() / n as f64;
            if !first_stage.is_finite() || first_stage <= f64::EPSILON {
                return Err(CausalError::Unsupported {
                    message: "receipt-adjusted randomized effects require a finite positive receipt first stage",
                });
            }
            let cace = outcome_itt / first_stage;
            let influence = outcome_scores.iter().zip(&receipt_scores)
                .map(|(outcome, receipt)| (outcome - cace * receipt) / first_stage)
                .collect::<Vec<_>>();
            let variance = sample_variance(&influence) / n as f64;
            complier_components = Some((outcome_itt, first_stage));
            (cace, variance, controls, treated, Arc::from([]), Arc::from([]),
                Arc::<str>::from("bernoulli"),
                Arc::<str>::from(if self.query.estimand == antecedent_core::RandomizedEstimand::TreatmentOnTreated {
                    "bernoulli_one_sided_tot_influence_variance_no_interval"
                } else { "bernoulli_wald_cace_influence_variance_no_interval" }),
                if self.query.estimand == antecedent_core::RandomizedEstimand::TreatmentOnTreated {
                    "Treatment-on-treated among recipients under one-sided noncompliance, exclusion, and independent assignment; Wald point and independent-unit influence variance, no interval"
                } else { "Wald CACE/LATE from randomized encouragement and observed receipt; independent-unit influence variance, no interval" })
        } else { match &self.query.design {
            antecedent_core::RandomizationDesign::MultiArm { assignment, probabilities, arms } => {
                let fit = antecedent_estimate::multi_arm::estimate_multi_arm(outcomes, assignment, probabilities)
                    .map_err(|message| CausalError::Unsupported { message })?;
                let primary = fit[1].value - fit[0].value;
                let primary_variance = 2.0 * (fit[1].variance_bound + fit[0].variance_bound);
                multi_arm_values = arms.iter().zip(fit.iter()).map(|(label, arm)|
                    (Arc::clone(label), arm.value, arm.variance_bound, arm.observed_support)
                ).collect::<Vec<_>>().into();
                let mut intervals = (0..arms.len()).map(|arm| {
                    if arm == 0 { return None; }
                    antecedent_estimate::randomized_scores::independent_action_contrast(
                        outcomes, assignment, probabilities, 0, arm,
                    ).and_then(|contrast| {
                        if arm == 1 && contrast.interval_95.is_some() {
                            interval_standard_error = Some(contrast.variance_upper_bound.sqrt());
                        }
                        contrast.interval_95
                    })
                }).collect::<Vec<_>>();
                if intervals.iter().skip(1).any(Option::is_none) {
                    intervals.fill(None);
                    interval_standard_error = None;
                }
                interval_95 = intervals[1];
                multi_arm_intervals_95 = intervals.into();
                (primary, primary_variance, fit[0].observed_support, fit[1].observed_support,
                    Arc::from([]), Arc::from([]), Arc::<str>::from("multi_arm"),
                    Arc::<str>::from("multi_arm_covariance_free_variance_bound_no_interval"),
                    "Independent multi-arm assignment with known action probabilities; all contrasts versus reference retain covariance-free variance bounds, no calibrated interval")
            }
            antecedent_core::RandomizationDesign::Bernoulli => {
                let mut effect_sum = 0.0;
                let mut variance_sum = 0.0;
                let mut control_units = 0;
                let mut treatment_units = 0;
                // With a fixed pre-assignment adjustment, E[observed y² / assignment²]
                // equals Y(1)²/p + Y(0)²/(1-p). This exceeds the Bernoulli
                // variance by (Y(1)-Y(0))² for each unit. No interval follows
                // from the random bound estimate without calibration.
                for ((&y, &treated), &p) in outcomes
                    .iter()
                    .zip(self.query.realized_assignment.iter())
                    .zip(self.query.assignment_probabilities.iter())
                {
                    if treated {
                        treatment_units += 1;
                        effect_sum += y / p;
                        variance_sum += if self.query.fixed_cuped.is_some() {
                            y * y / (p * p)
                        } else {
                            y * y * (1.0 - p) / (p * p)
                        };
                    } else {
                        control_units += 1;
                        effect_sum -= y / (1.0 - p);
                        variance_sum += if self.query.fixed_cuped.is_some() {
                            y * y / ((1.0 - p) * (1.0 - p))
                        } else {
                            y * y * p / ((1.0 - p) * (1.0 - p))
                        };
                    }
                }
                let assignments = self.query.realized_assignment.iter().map(|assigned|
                    usize::from(*assigned)).collect::<Vec<_>>();
                let probabilities = self.query.assignment_probabilities.iter().map(|p|
                    vec![1.0 - p, *p]).collect::<Vec<_>>();
                if let Some(contrast) = antecedent_estimate::randomized_scores::independent_action_contrast(
                    outcomes, &assignments, &probabilities, 0, 1,
                ) {
                    interval_95 = contrast.interval_95;
                    if interval_95.is_some() {
                        interval_standard_error = Some(contrast.variance_upper_bound.sqrt());
                    }
                }
                (
                    effect_sum / n as f64,
                    variance_sum / (n * n) as f64,
                    control_units,
                    treatment_units,
                    Arc::from([]),
                    Arc::from([]),
                    Arc::<str>::from("bernoulli"),
                    Arc::<str>::from(if self.query.fixed_cuped.is_some() {
                        "bernoulli_fixed_cuped_ht_conservative_variance_no_interval"
                    } else {
                        "bernoulli_ht_design_variance_no_interval"
                    }),
                    if self.query.fixed_cuped.is_some() {
                        "Horvitz-Thompson ITT on outcomes adjusted by a declared pre-assignment covariate and externally fixed coefficient; observed-arm upper-bound estimator is conservative in randomization expectation, no interval"
                    } else {
                        "Horvitz-Thompson ITT with Bernoulli design variance; no confidence interval is reported"
                    },
                )
            }
            antecedent_core::RandomizationDesign::Complete { treated_units } => {
                let treated: Vec<f64> = outcomes
                    .iter()
                    .zip(self.query.realized_assignment.iter())
                    .filter_map(|(&y, &z)| z.then_some(y))
                    .collect();
                let control: Vec<f64> = outcomes
                    .iter()
                    .zip(self.query.realized_assignment.iter())
                    .filter_map(|(&y, &z)| (!z).then_some(y))
                    .collect();
                let fit = antecedent_estimate::randomized_neyman::complete_unit_itt(
                    &treated, &control,
                ).ok_or(CausalError::Unsupported {
                    message: "complete-randomized ITT requires finite outcomes and estimable variance",
                })?;
                interval_95 = fit.interval_95;
                if interval_95.is_some() { interval_standard_error = Some(fit.variance_upper_bound.sqrt()); }
                (
                    fit.effect,
                    fit.variance_upper_bound,
                    n - treated_units,
                    *treated_units,
                    Arc::from([]),
                    Arc::from([]),
                    Arc::<str>::from("complete"),
                    Arc::<str>::from("complete_neyman_variance_upper_bound_no_interval"),
                    "Complete-randomization difference in means with Neyman conservative variance estimate; no confidence interval is reported",
                )
            }
            antecedent_core::RandomizationDesign::Cluster { treated_clusters } => {
                let mut cluster_totals: std::collections::BTreeMap<&str, (bool, f64)> =
                    std::collections::BTreeMap::new();
                for (i, outcome) in outcomes.iter().enumerate() {
                    let entry = cluster_totals
                        .entry(self.query.assignment_units[i].as_ref())
                        .or_insert((self.query.realized_assignment[i], 0.0));
                    entry.1 += outcome;
                }
                let treated: Vec<f64> = cluster_totals
                    .values()
                    .filter_map(|(assigned, total)| assigned.then_some(*total))
                    .collect();
                let control: Vec<f64> = cluster_totals
                    .values()
                    .filter_map(|(assigned, total)| (!assigned).then_some(*total))
                    .collect();
                let clusters = cluster_totals.len();
                let fit = antecedent_estimate::randomized_neyman::complete_cluster_itt(
                    &treated, &control, n,
                ).ok_or(CausalError::Unsupported {
                    message: "cluster-randomized ITT requires finite cluster totals and estimable variance",
                })?;
                interval_95 = fit.interval_95;
                if interval_95.is_some() { interval_standard_error = Some(fit.variance_upper_bound.sqrt()); }
                (
                    fit.effect,
                    fit.variance_upper_bound,
                    clusters - treated_clusters,
                    *treated_clusters,
                    Arc::from([]),
                    Arc::from([]),
                    Arc::<str>::from("cluster"),
                    Arc::<str>::from("cluster_neyman_variance_upper_bound_no_interval"),
                    "Complete cluster-randomization HT ITT over cluster outcome totals, with Neyman conservative variance; no confidence interval is reported",
                )
            }
            antecedent_core::RandomizationDesign::Stratified { blocks, treated_per_row } => {
                let mut grouped: std::collections::BTreeMap<&str, (Vec<f64>, Vec<f64>, usize)> =
                    std::collections::BTreeMap::new();
                for i in 0..n {
                    let group = grouped
                        .entry(&blocks[i])
                        .or_insert_with(|| (Vec::new(), Vec::new(), treated_per_row[i]));
                    if self.query.realized_assignment[i] {
                        group.0.push(outcomes[i]);
                    } else {
                        group.1.push(outcomes[i]);
                    }
                }
                let cells = grouped.values().map(|(treated, control, _)|
                    (treated.as_slice(), control.as_slice())).collect::<Vec<_>>();
                let fit = antecedent_estimate::randomized_neyman::blocked_unit_itt(&cells)
                    .ok_or(CausalError::Unsupported {
                        message: "blocked-randomized ITT requires finite outcomes and estimable within-block variances",
                    })?;
                interval_95 = fit.interval_95;
                if interval_95.is_some() { interval_standard_error = Some(fit.variance_upper_bound.sqrt()); }
                let treatment_units = cells.iter().map(|(treated, _)| treated.len()).sum();
                let control_units = cells.iter().map(|(_, control)| control.len()).sum();
                (
                    fit.effect,
                    fit.variance_upper_bound,
                    control_units,
                    treatment_units,
                    Arc::clone(blocks),
                    Arc::from([]),
                    Arc::<str>::from("stratified"),
                    Arc::<str>::from("stratified_neyman_variance_upper_bound_no_interval"),
                    "Stratified difference in means weighted by block size, with blockwise Neyman conservative variance estimate; no confidence interval is reported",
                )
            }
            antecedent_core::RandomizationDesign::Factorial2x2 { second_factor_assignment, .. } => {
                let mut cells: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
                for i in 0..n {
                    let cell = usize::from(self.query.realized_assignment[i])
                        + 2 * usize::from(second_factor_assignment[i]);
                    cells[cell].push(outcomes[i]);
                }
                let fit = antecedent_estimate::randomized_neyman::factorial_2x2(
                    cells.each_ref().map(Vec::as_slice),
                ).ok_or(CausalError::Unsupported {
                    message: "factorial ITT requires finite outcomes and estimable cell variances",
                })?;
                interval_95 = fit.primary_interval_95;
                second_factor_interval_95 = fit.secondary_interval_95;
                factorial_interaction_interval_95 = fit.interaction_interval_95;
                if interval_95.is_some() { interval_standard_error = Some(fit.main_effect_variance_upper_bound.sqrt()); }
                factorial_contrasts = Some((fit.secondary, fit.interaction,
                    fit.main_effect_variance_upper_bound, fit.interaction_variance_upper_bound));
                (
                    fit.primary,
                    fit.main_effect_variance_upper_bound,
                    cells[0].len() + cells[2].len(),
                    cells[1].len() + cells[3].len(),
                    Arc::from([]), Arc::from([]),
                    Arc::<str>::from("factorial_2x2"),
                    Arc::<str>::from("factorial_cell_neyman_variance_upper_bound_no_interval"),
                    "Fixed-cell 2×2 factorial main effects and interaction from four cell means, with cellwise Neyman conservative variance estimates; no confidence interval is reported",
                )
            }
            antecedent_core::RandomizationDesign::Switchback { periods } => {
                let sequence_ids = self.query.assignment_units.iter()
                    .map(AsRef::as_ref).collect::<Vec<&str>>();
                let fit = antecedent_estimate::switchback::switchback_itt(
                    outcomes, &self.query.realized_assignment,
                    &self.query.assignment_probabilities, &sequence_ids,
                ).ok_or(CausalError::Unsupported {
                    message: "switchback ITT requires valid outcomes, probabilities, and independent sequences with global arm support",
                })?;
                interval_95 = fit.interval_95;
                if interval_95.is_some() { interval_standard_error = Some(fit.variance.sqrt()); }
                (
                    fit.effect,
                    fit.variance,
                    self.query.realized_assignment.iter().filter(|&&a| !a).count(),
                    self.query.realized_assignment.iter().filter(|&&a| a).count(),
                    Arc::from([]),
                    Arc::clone(periods),
                    Arc::<str>::from("switchback"),
                    Arc::<str>::from("switchback_independent_sequence_sandwich_variance_no_interval"),
                    "Unit-period HT ITT with independent-sequence score sandwich variance; arbitrary within-sequence dependence, no interval",
                )
            }
        }};
        if !effect.is_finite() || !variance.is_finite() || variance < 0.0 {
            return Err(CausalError::Unsupported {
                message: "randomized effect or design variance is not finite under the declared probabilities",
            });
        }
        let uncertainty = if interval_95.is_some() {
            Arc::<str>::from(match self.query.design {
                antecedent_core::RandomizationDesign::Bernoulli if self.query.fixed_cuped.is_some() =>
                    "bernoulli_fixed_cuped_ht_score_normal_interval",
                antecedent_core::RandomizationDesign::Bernoulli if !self.query.ancova_covariates.is_empty() =>
                    "bernoulli_ancova_hc0_normal_interval",
                antecedent_core::RandomizationDesign::Bernoulli => "bernoulli_ht_score_normal_interval",
                antecedent_core::RandomizationDesign::Complete { .. } => "complete_neyman_normal_interval",
                antecedent_core::RandomizationDesign::Cluster { .. } => "cluster_neyman_normal_interval",
                antecedent_core::RandomizationDesign::Stratified { .. } => "stratified_neyman_normal_interval",
                antecedent_core::RandomizationDesign::Factorial2x2 { .. } => "factorial_cell_neyman_pointwise_normal_intervals",
                antecedent_core::RandomizationDesign::MultiArm { .. } => "multi_arm_ht_score_pointwise_normal_intervals",
                antecedent_core::RandomizationDesign::Switchback { .. } => "switchback_independent_sequence_student_interval",
            })
        } else { uncertainty };
        let diagnostic = if interval_95.is_some() {
            match self.query.design {
                antecedent_core::RandomizationDesign::Bernoulli if self.query.fixed_cuped.is_some() =>
                    "Bernoulli ITT with an externally fixed pre-assignment CUPED coefficient and independent-unit score-sandwich pointwise 95% interval",
                antecedent_core::RandomizationDesign::Bernoulli if !self.query.ancova_covariates.is_empty() =>
                    "Bernoulli ANCOVA with pre-assignment covariates and independent-unit HC0 pointwise 95% interval",
                antecedent_core::RandomizationDesign::Bernoulli =>
                    "Bernoulli Horvitz-Thompson ITT with known probabilities and an independent-unit score-sandwich pointwise 95% interval",
                antecedent_core::RandomizationDesign::Complete { .. } =>
                    "Complete-randomization difference in means with a conservative Neyman variance and pointwise 95% normal interval",
                antecedent_core::RandomizationDesign::Cluster { .. } =>
                    "Cluster-randomized ITT over cluster outcome totals with a conservative cluster-level Neyman variance and pointwise 95% normal interval",
                antecedent_core::RandomizationDesign::Stratified { .. } =>
                    "Blocked difference in means with blockwise conservative Neyman variance and pointwise 95% normal interval",
                antecedent_core::RandomizationDesign::Factorial2x2 { .. } =>
                    "Fixed-cell factorial main effects and interaction with separate conservative cellwise variances and pointwise 95% normal intervals; simultaneous coverage is not claimed",
                antecedent_core::RandomizationDesign::MultiArm { .. } =>
                    "Independent multi-arm HT effects with known probabilities; the retained variance is a covariance-free bound, while separate score-sandwich pointwise 95% intervals cover each action versus reference; simultaneous coverage is not claimed",
                antecedent_core::RandomizationDesign::Switchback { .. } =>
                    "Unit-period HT ITT with independent-sequence sandwich variance and a pointwise 95% Student interval; no carryover and no between-sequence interference are declared assumptions",
            }
        } else { diagnostic };
        let exact_test = if self.query.exact_randomization_test {
            let antecedent_core::RandomizationDesign::Complete { treated_units } = &self.query.design else {
                unreachable!("validated exact test requires complete randomization")
            };
            let observed = effect.abs();
            let mut extreme = 0_u64;
            let mut allocations = 0_u64;
            let total = outcomes.iter().sum::<f64>();
            for mask in 0..(1_u64 << n) {
                if mask.count_ones() as usize != *treated_units { continue; }
                let treated_sum = outcomes.iter().enumerate()
                    .filter_map(|(i, value)| ((mask >> i) & 1 == 1).then_some(*value))
                    .sum::<f64>();
                let contrast = treated_sum / *treated_units as f64
                    - (total - treated_sum) / (n - *treated_units) as f64;
                allocations += 1;
                if contrast.abs() + 1e-12 >= observed { extreme += 1; }
            }
            Some((extreme as f64 / allocations as f64, allocations))
        } else { None };
        let mut estimate = EffectEstimate::new(
            effect,
            interval_standard_error.unwrap_or(f64::NAN),
            self.identification.required_assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        if interval_95.is_some() && matches!(self.query.design,
            antecedent_core::RandomizationDesign::Cluster { .. } | antecedent_core::RandomizationDesign::Switchback { .. }) {
            estimate = estimate.with_se_kind(antecedent_estimate::AnalyticSeKind::Cluster);
        }
        let started = Instant::now();
        let mut result = finish_identified_execute_with_context(
            &self.result_context,
            Some(data),
            IdentifiedExecuteFinish {
                physical: &self.physical,
                identification: self.identification.clone(),
                estimand: self.estimand.clone(),
                estimate,
                identifier_id: IdentifierId::RandomizedDesign,
                estimator_id: randomized_estimator_id(&self.query),
                treatment: self.query.outcome,
                outcome: self.query.outcome,
                identify_cached: true,
                extra_diagnostics: vec![Diagnostic::new(
                    match self.query.design {
                        antecedent_core::RandomizationDesign::Bernoulli if self.query.received_treatment.is_some() => "estimate.randomized.wald_cace_late",
                        antecedent_core::RandomizationDesign::Bernoulli if self.query.fixed_cuped.is_some() => "estimate.randomized.fixed_cuped_ht_itt",
                        antecedent_core::RandomizationDesign::Bernoulli if !self.query.ancova_covariates.is_empty() => "estimate.randomized.ancova_itt",
                        antecedent_core::RandomizationDesign::Bernoulli => "estimate.randomized.ht_itt",
                        antecedent_core::RandomizationDesign::Switchback { .. } => "estimate.randomized.switchback_ht_itt",
                        antecedent_core::RandomizationDesign::MultiArm { .. } => "estimate.randomized.multi_arm_ht_itt",
                        _ => "estimate.randomized.neyman_itt",
                    },
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    diagnostic,
                )],
                refutations: Vec::new(),
                distribution: None,
                mediation: None,
                wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
                bootstrap_replicates_ok: None,
                cancelled: false,
                early_stopped: false,
                extras: IdentifiedExecuteExtras::default(),
            },
        );
        result.randomized_effect = Some(crate::RandomizedEffectEstimate {
            effect,
            estimand: Arc::from(if self.query.estimand == antecedent_core::RandomizedEstimand::TreatmentOnTreated { "treatment_on_treated" } else if complier_components.is_some() { "cace_late" } else if factorial_contrasts.is_some() { "factorial_primary_main_effect" } else if !multi_arm_values.is_empty() { "multi_arm_itt" } else { "itt" }),
            intention_to_treat_effect: complier_components.map(|(itt, _)| itt),
            first_stage_effect: complier_components.map(|(_, stage)| stage),
            received_treatment: self.query.received_treatment.clone(),
            randomization_p_value: exact_test.map(|(p, _)| p),
            randomization_allocations: exact_test.map(|(_, count)| count),
            second_factor_effect: factorial_contrasts.map(|(effect, _, _, _)| effect),
            factorial_interaction: factorial_contrasts.map(|(_, effect, _, _)| effect),
            second_factor_variance: factorial_contrasts.map(|(_, _, variance, _)| variance),
            factorial_interaction_variance: factorial_contrasts.map(|(_, _, _, variance)| variance),
            multi_arm_values,
            variance_upper_bound: variance,
            standard_error: interval_standard_error,
            interval_95,
            second_factor_interval_95,
            factorial_interaction_interval_95,
            multi_arm_intervals_95,
            minimum_assignment_probability: if let antecedent_core::RandomizationDesign::MultiArm { probabilities, .. } = &self.query.design {
                probabilities.iter().flat_map(|row| row.iter().copied()).fold(f64::INFINITY, f64::min)
            } else {
                self.query.assignment_probabilities.iter().copied()
                    .map(|p| if matches!(self.query.design, antecedent_core::RandomizationDesign::Switchback { .. }) {
                        p.min(1.0 - p)
                    } else { p })
                    .fold(f64::INFINITY, f64::min)
            },
            assignment_design,
            blocks,
            periods,
            control_units,
            treatment_units,
            uncertainty,
            assignment_units: Arc::clone(&self.query.assignment_units),
            outcome_units: Arc::clone(&self.query.outcome_units),
            treatment_arms: self.query.treatment_arms.clone(),
        });
        // Graphless licenses are exact design/method/inference claims. A
        // published interval alone is insufficient: sparse or other designs
        // must not inherit a geometric matrix coordinate by analogy.
        let (design, method, assignment_unit, claim) = match self.query.design {
            antecedent_core::RandomizationDesign::Bernoulli if self.query.fixed_cuped.is_some() =>
                ("bernoulli", "fixed_cuped_ht_score", "unit", "pointwise_95_normal_interval"),
            antecedent_core::RandomizationDesign::Bernoulli if !self.query.ancova_covariates.is_empty() =>
                ("bernoulli", "ancova_hc0", "unit", "pointwise_95_normal_interval"),
            antecedent_core::RandomizationDesign::Bernoulli =>
                ("bernoulli", "independent_action_ht_score", "unit", "pointwise_95_normal_interval"),
            antecedent_core::RandomizationDesign::Complete { .. } =>
                ("complete", "neyman_difference_in_means", "unit", "pointwise_95_normal_interval"),
            antecedent_core::RandomizationDesign::Cluster { .. } =>
                ("cluster", "neyman_unit_weighted_cluster_totals", "cluster", "pointwise_95_normal_interval"),
            antecedent_core::RandomizationDesign::Stratified { .. } =>
                ("stratified", "blocked_neyman_difference_in_means", "unit", "pointwise_95_normal_interval"),
            antecedent_core::RandomizationDesign::Factorial2x2 { .. } =>
                ("factorial_2x2", "fixed_cell_neyman_contrasts", "unit", "three_pointwise_95_normal_intervals"),
            antecedent_core::RandomizationDesign::MultiArm { .. } =>
                ("multi_arm", "independent_action_ht_scores", "unit", "all_action_pointwise_95_normal_intervals"),
            antecedent_core::RandomizationDesign::Switchback { .. } =>
                ("switchback", "independent_sequence_ht_score", "sequence", "pointwise_95_student_interval"),
        };
        let fit = result.randomized_effect.as_ref().expect("randomized result was just set");
        let mut block_counts = std::collections::BTreeMap::<&str, (usize, usize)>::new();
        let support_blocks = if matches!(self.query.design, antecedent_core::RandomizationDesign::Switchback { .. }) {
            &self.query.assignment_units
        } else { &fit.blocks };
        for (block, assigned) in support_blocks.iter().zip(self.query.realized_assignment.iter()) {
            let counts = block_counts.entry(block).or_default();
            if *assigned { counts.1 += 1; } else { counts.0 += 1; }
        }
        let min_block_arm = block_counts.values().flat_map(|(control, treated)| [*control, *treated]).min().unwrap_or(0);
        let balanced_sequences = matches!(self.query.design, antecedent_core::RandomizationDesign::Switchback { .. })
            && block_counts.values().next().is_some_and(|first| {
                let size = first.0 + first.1;
                size > 0 && block_counts.values().all(|counts| counts.0 + counts.1 == size)
            });
        let min_factorial_cell = if let antecedent_core::RandomizationDesign::Factorial2x2 { second_factor_assignment, .. } = &self.query.design {
            let mut counts = [0_usize; 4];
            for (primary, secondary) in self.query.realized_assignment.iter().zip(second_factor_assignment.iter()) {
                counts[usize::from(*primary) + 2 * usize::from(*secondary)] += 1;
            }
            *counts.iter().min().unwrap_or(&0)
        } else { 0 };
        let reported_intervals = if design == "multi_arm" {
            fit.multi_arm_intervals_95.iter().filter(|interval| interval.is_some()).count()
        } else {
            usize::from(fit.interval_95.is_some())
                + usize::from(fit.second_factor_interval_95.is_some())
                + usize::from(fit.factorial_interaction_interval_95.is_some())
        };
        let all_reported_intervals = match design {
            "factorial_2x2" => reported_intervals == 3,
            "multi_arm" => fit.multi_arm_intervals_95.len() == fit.multi_arm_values.len()
                && reported_intervals + 1 == fit.multi_arm_values.len(),
            _ => fit.interval_95.is_some(),
        };
        let min_probability = if let antecedent_core::RandomizationDesign::MultiArm { probabilities, .. } = &self.query.design {
            probabilities.iter().flat_map(|row| row.iter().copied()).fold(f64::INFINITY, f64::min)
        } else {
            self.query.assignment_probabilities.iter().copied()
                .map(|p| p.min(1.0 - p)).fold(f64::INFINITY, f64::min)
        };
        if let crate::support::GraphlessSupportStatus::Licensed { .. } = crate::support::classify_graphless(
            crate::support::GraphlessSupportKey {
                family: "randomized_effect", design, method, inference_claim: claim,
            },
            crate::support::GraphlessAssignmentSupport {
                assignment_unit, treated: treatment_units, control: control_units,
                interval_95_published: interval_95.is_some(),
                rows: n, blocks: block_counts.len(), min_block_arm, min_factorial_cell,
                min_action_rows: if fit.multi_arm_values.is_empty() {
                    control_units.min(treatment_units)
                } else {
                    fit.multi_arm_values.iter().map(|(_, _, _, count)| *count).min().unwrap_or(0)
                },
                min_probability, reported_intervals, all_reported_intervals,
                covariates: self.query.ancova_covariates.len(),
                balanced_sequences,
                ..Default::default()
            },
        ) {
            result.support_status = Some(crate::support::CellStatus::Licensed);
        }
        result.rebind_interval(false);
        result.treatment = None;
        Ok(result)
    }
}

impl Study {
    pub(super) fn execute_policy_value(
        &self,
        data: &TabularData,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        CheckedPolicyValueOperation::checked(self, data, physical)?.execute(data, ctx)
    }

    pub(super) fn execute_randomized(
        &self,
        data: &TabularData,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        CheckedRandomizedOperation::checked(self, data, physical)?.execute(data, ctx)
    }
}

/// Frozen, held-out policy scoring operation. Prediction vectors and ownership
/// declarations are part of the retained query identity.
#[derive(Clone)]
pub(crate) struct CheckedPolicyValueOperation {
    query: antecedent_core::PolicyValueQuery,
    source_schema: antecedent_core::CausalSchema,
    source_rows: usize,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    physical: PhysicalExecutionPlan,
    result_context: IdentifiedResultContext,
}

impl std::fmt::Debug for CheckedPolicyValueOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CheckedPolicyValueOperation")
            .field("query", &self.query)
            .field("source_rows", &self.source_rows)
            .finish_non_exhaustive()
    }
}

impl CheckedPolicyValueOperation {
    pub(crate) fn checked(
        study: &Study,
        data: &TabularData,
        physical: &PhysicalExecutionPlan,
    ) -> Result<Self, CausalError> {
        let CausalQuery::PolicyValue(query) = &study.query else {
            return Err(CausalError::Compile {
                message: "policy-value execution requires PolicyValueQuery".into(),
            });
        };
        query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
        if query.multi_action.as_ref().map_or(query.assignment.len(), |multi| multi.assignment.len()) != data.row_count()
            || !matches!(study.inference, InferenceMode::Frequentist)
            || study.refute != RefuteSuite::None
            || study.bootstrap_replicates != 0
            || !study.custom_validators.is_empty()
            || study.graph.class() != GraphClass::RandomizedTrial
        {
            return Err(CausalError::Unsupported {
                message: "policy value requires aligned graphless randomized rows and does not accept refuters, bootstrap, or Bayesian inference",
            });
        }
        data.schema()
            .get(query.outcome)
            .map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let n = data.row_count();
        let (binary_assignment, binary_propensity) = if let Some(multi) = &query.multi_action {
            let k = multi.action_labels.len();
            (
                multi.assignment.iter().map(|&a| a != 0).collect::<Vec<_>>(),
                (0..n).map(|i| 1.0 - multi.propensities[i * k]).collect::<Vec<_>>(),
            )
        } else {
            (
                query.assignment.to_vec(),
                (0..n).map(|i| query.propensity[if query.propensity.len() == 1 { 0 } else { i }]).collect::<Vec<_>>(),
            )
        };
        let synthetic = antecedent_core::RandomizedEffectQuery::bernoulli_itt(
            query.outcome,
            binary_assignment,
            binary_propensity,
            query.evaluation_subject_ids.clone(),
            query.evaluation_subject_ids.clone(),
            ("control", "treated"),
        );
        let (mut identification, mut estimand) = randomized_identification(&synthetic);
        identification.query = CausalQuery::PolicyValue(query.clone());
        estimand.method = Arc::from(antecedent_expr::EstimandMethod::RandomizedPolicyValue);
        identification.estimands = vec![estimand.clone()];
        identification.arena.set_derivation(
            estimand.functional,
            antecedent_expr::DerivationMeta::rule(
                "randomized.policy_value",
                Some(Arc::from("known randomized action probabilities identify the value of a fixed policy")),
            ),
        );
        identification.derivation = DerivationTrace::default();
        identification.derivation.push(
            "randomized.policy_value",
            "known randomized action probabilities identify the value of fixed policy and reference recommendations",
        );
        identification.required_assumptions.push(AssumptionRecord {
            assumption: Assumption::Custom {
                id: Arc::from("policy_fixed_without_evaluation_outcomes"),
                description: Arc::from("policy and reference recommendations were fixed without using the corresponding evaluation outcomes"),
            },
            source: AssumptionSource::UserDeclared,
            scope: AssumptionScope::Identification,
            status: AssumptionStatus::Declared,
        });
        if query.multi_action.is_some() {
            identification.required_assumptions.push(AssumptionRecord {
                assumption: Assumption::Custom {
                    id: Arc::from("known_random_action_probabilities"),
                    description: Arc::from("each evaluation row followed the declared randomized multi-action probability vector"),
                },
                source: AssumptionSource::UserDeclared,
                scope: AssumptionScope::Identification,
                status: AssumptionStatus::Declared,
            });
            if query.multi_action.as_ref().is_some_and(|policy| !policy.cate_groups.is_empty()) {
                identification.required_assumptions.push(AssumptionRecord {
                    assumption: Assumption::Custom {
                        id: Arc::from("fixed_pre_outcome_cate_groups"),
                        description: Arc::from("baseline groups for multi-action conditional contrasts were fixed before observing evaluation outcomes"),
                    },
                    source: AssumptionSource::UserDeclared,
                    scope: AssumptionScope::Identification,
                    status: AssumptionStatus::Declared,
                });
            }
        }
        Ok(Self {
            query: query.clone(),
            source_schema: data.schema().clone(),
            source_rows: n,
            identification,
            estimand,
            physical: physical.clone(),
            result_context: IdentifiedResultContext::from_study(study),
        })
    }

    pub(crate) fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if data.schema() != &self.source_schema || data.row_count() != self.source_rows {
            return Err(CausalError::Unsupported {
                message: "policy value refresh requires the prepared schema and row order",
            });
        }
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled {
                stage: super::super::stage::STAGE_ESTIMATE_POINT,
            });
        }
        let y = match data.column(self.query.outcome).map_err(CausalError::from)? {
            antecedent_data::ColumnView::Float64(values) => values.values.as_slice(),
            _ => {
                return Err(CausalError::Unsupported {
                    message: "policy value outcome must be continuous",
                });
            }
        };
        let multi = self.query.multi_action.as_ref();
        let multi_action_cate = if let Some(policy) = multi {
            antecedent_estimate::policy_value::evaluate_multi_action_cate(y, policy)
                .map_err(|message| CausalError::Unsupported { message })?
        } else { Vec::new() };
        let regret = self.query.regret.as_ref().map(|design| {
            antecedent_estimate::policy_value::evaluate_fixed_candidate_regret(
                y, &self.query.assignment, &self.query.propensity,
                &design.candidates.iter().map(|candidate| candidate.to_vec()).collect::<Vec<_>>(),
                design.selected_index, &self.query.costs,
                !design.training_subject_ids.is_empty(),
            ).map_err(|message| CausalError::Unsupported { message })
        }).transpose()?;
        let ipw = self.query.mu0.is_empty();
        let uplift_bins = if self.query.uplift_bin_count > 0 {
            antecedent_estimate::policy_value::evaluate_uplift_bins(
                y, &self.query.assignment, &self.query.propensity,
                &self.query.uplift_bins, self.query.uplift_bin_count,
            ).map_err(|message| CausalError::Unsupported { message })?
        } else { Vec::new() };
        let score = if let Some(multi) = multi {
            antecedent_estimate::policy_value::evaluate_multi_action_policy_value_scores(y, multi)
        } else if ipw {
            antecedent_estimate::policy_value::evaluate_policy_value_ipw_scores(
                y, &self.query.assignment, &self.query.actions, &self.query.propensity,
                &self.query.reference, &self.query.costs, &self.query.reference_costs,
            )
        } else {
            antecedent_estimate::policy_value::evaluate_policy_value_scores(
                y, &self.query.assignment, &self.query.actions, &self.query.propensity,
                &self.query.mu0, &self.query.mu1, &self.query.reference,
                &self.query.costs, &self.query.reference_costs,
            )
        }
        .map_err(|message| CausalError::Unsupported { message })?;
        let (policy_matches, reference_matches) = if let Some(policy) = multi {
            (
                policy.assignment.iter().zip(policy.actions.iter()).filter(|(a, b)| a == b).count(),
                policy.assignment.iter().zip(policy.reference.iter()).filter(|(a, b)| a == b).count(),
            )
        } else {
            (
                self.query.assignment.iter().zip(self.query.actions.iter()).filter(|(a, b)| a == b).count(),
                self.query.assignment.iter().zip(self.query.reference.iter()).filter(|(a, b)| a == b).count(),
            )
        };
        // Cross-fitted predictions on caller-declared excluded folds do not use
        // a row's own outcome, so the independent-row influence-function
        // variance is valid; IPW and disjoint held-out AIPW have fixed score
        // rules. Global capacity/budget coupling still withholds the interval.
        let independent_policy_rows = !self.query.global_constraints_present
            && multi.is_none_or(|policy| !policy.global_constraints_couple_rows());
        let minimum_calibrated_rows = if multi.is_some() || !ipw { 300 } else { 120 };
        let intervals = if y.len() >= minimum_calibrated_rows
            && independent_policy_rows
            && (ipw || self.query.disjoint_training_subjects || self.query.crossfit_fold_ownership_valid) {
            antecedent_estimate::policy_value::pointwise_intervals_95(
                &score, y.len(), policy_matches, reference_matches,
            )
        } else { None };
        let mut result = finish_identified_execute_with_context(
            &self.result_context,
            Some(data),
            IdentifiedExecuteFinish {
                physical: &self.physical,
                identification: self.identification.clone(),
                estimand: self.estimand.clone(),
                estimate: EffectEstimate::new(
                    0.0,
                    f64::NAN,
                    self.identification.required_assumptions.clone(),
                    OverlapPolicy::ExplicitOverride,
                ),
                identifier_id: IdentifierId::RandomizedDesign,
                estimator_id: if multi.is_some() {
                    EstimatorId::RandomizedMultiActionIpwPolicy
                } else if ipw { EstimatorId::RandomizedIpwPolicy } else { EstimatorId::RandomizedDrPolicy },
                treatment: self.query.outcome,
                outcome: self.query.outcome,
                identify_cached: true,
                extra_diagnostics: vec![Diagnostic::new(
                    if multi.is_some() { "estimate.policy_value.multi_action_ipw" }
                    else if ipw { "estimate.policy_value.ipw" } else { "estimate.policy_value.doubly_robust" },
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    if multi.is_some() {
                        "fixed randomized multi-action IPW policy value; row-score standard errors assume independent subjects"
                    } else if ipw {
                        "fixed randomized IPW policy value; row-score standard errors assume independent subjects"
                    } else {
                        "held-out randomized AIPW policy value; row-score standard errors assume independent subjects"
                    },
                )],
                refutations: Vec::new(),
                distribution: None,
                mediation: None,
                wall_time_ns: 0,
                bootstrap_replicates_ok: None,
                cancelled: false,
                early_stopped: false,
                extras: IdentifiedExecuteExtras::default(),
            },
        );
        result.estimate = crate::PrimaryEstimate::NotAnEffect;
        result.treatment = None;
        result.policy_value = Some(crate::PolicyValueEstimate {
            policy_value: score.policy_value,
            reference_value: score.reference_value,
            incremental_value: score.incremental_value,
            relative_value_gap: score.relative_value_gap,
            treatment_rate: score.treatment_rate,
            total_cost: score.total_cost,
            policy_standard_error: score.policy_standard_error,
            reference_standard_error: score.reference_standard_error,
            incremental_standard_error: score.incremental_standard_error,
            policy_interval_95: intervals.map(|value| value.policy),
            incremental_interval_95: intervals.map(|value| value.incremental),
            prediction_ownership: Arc::from(if ipw {
                "no_outcome_nuisance_predictions"
            } else if self.query.disjoint_training_subjects {
                "declared_disjoint_training_subject_ids"
            } else {
                "caller_declared_cross_fitted_excluded_fold_ids"
            }),
            propensity_min: score.propensity_min,
            propensity_max: score.propensity_max,
            uncertainty: Arc::from(if multi.is_some() {
                "multi_action_ipw_row_score_standard_error_independent_subjects"
            } else if ipw {
                "ipw_row_score_standard_error_independent_subjects"
            } else {
                "row_score_standard_error_independent_subjects"
            }),
            uplift_bins,
            multi_action_cate,
            regret,
        });
        let policy = result.policy_value.as_ref().expect("policy value was just attached");
        let (design, method, claim) = antecedent_core::policy_graphless_coordinate(
            multi.is_some(), ipw, !policy.uplift_bins.is_empty(), !policy.multi_action_cate.is_empty(),
            self.query.regret.is_some(),
            !ipw && self.query.crossfit_fold_ownership_valid && !self.query.disjoint_training_subjects,
        );
        let (treated, control, min_action_rows, min_probability) = if let Some(multi) = multi {
            let k = multi.action_labels.len();
            (multi.assignment.iter().filter(|&&action| action != 0).count(),
             multi.assignment.iter().filter(|&&action| action == 0).count(),
             (0..k).map(|action| multi.assignment.iter().filter(|&&observed| observed == action).count())
                .min().unwrap_or(0),
             multi.propensities.iter().copied().fold(f64::INFINITY, f64::min))
        } else {
            let treated = self.query.assignment.iter().filter(|&&assigned| assigned).count();
            (treated, y.len() - treated, treated.min(y.len() - treated),
             self.query.propensity.iter().copied().map(|p| p.min(1.0 - p))
                .fold(f64::INFINITY, f64::min))
        };
        let min_bin_rows = policy.uplift_bins.iter().map(|bin| bin.evaluation_rows).min().unwrap_or(0);
        let min_bin_arm_rows = if policy.uplift_bins.is_empty() { 0 } else {
            (0..policy.uplift_bins.len()).flat_map(|bin| {
                let treated = self.query.uplift_bins.iter().enumerate()
                    .filter(|&(row, &rank)| rank == bin && self.query.assignment[row]).count();
                let total = policy.uplift_bins[bin].evaluation_rows;
                [treated, total - treated]
            }).min().unwrap_or(0)
        };
        let min_group_rows = policy.multi_action_cate.iter().map(|point| point.evaluation_rows).min().unwrap_or(0);
        let min_group_arm_rows = policy.multi_action_cate.iter()
            .flat_map(|point| [point.observed_action_rows, point.observed_control_rows]).min().unwrap_or(0);
        let scalar_intervals = usize::from(policy.policy_interval_95.is_some())
            + usize::from(policy.incremental_interval_95.is_some());
        let extras = policy.uplift_bins.iter().filter(|bin| bin.interval_95.is_some()).count()
            + policy.multi_action_cate.iter().filter(|point| point.interval_95.is_some()).count();
        let all_reported_intervals = scalar_intervals == 2
            && policy.uplift_bins.iter().all(|bin| bin.interval_95.is_some())
            && policy.multi_action_cate.iter().all(|point| point.interval_95.is_some());
        let observed = crate::support::GraphlessAssignmentSupport {
            assignment_unit: "unit", treated, control, rows: y.len(), min_action_rows,
            min_probability, interval_95_published: scalar_intervals == 2,
            reported_intervals: scalar_intervals + extras, all_reported_intervals,
            policy_matches, reference_matches, min_bin_rows, min_bin_arm_rows,
            min_group_rows, min_group_arm_rows,
            uncoupled_constraints: independent_policy_rows,
            disjoint_nuisance_training: self.query.disjoint_training_subjects,
            rank_ownership: !self.query.uplift_training_subject_ids.is_empty()
                && self.query.uplift_training_subject_ids.iter()
                    .all(|id| !self.query.evaluation_subject_ids.contains(id)),
            ..Default::default()
        };
        if matches!(crate::support::classify_graphless(
            crate::support::GraphlessSupportKey { family: "policy_value", design, method, inference_claim: claim }, observed,
        ), crate::support::GraphlessSupportStatus::Licensed { .. }) {
            result.support_status = Some(crate::support::CellStatus::Licensed);
        }
        Ok(result)
    }
}

pub(crate) fn randomized_identification(
    query: &antecedent_core::RandomizedEffectQuery,
) -> (IdentificationResult, IdentifiedEstimand) {
    let mut assumptions = antecedent_core::AssumptionSet::default();
    let mut standard = vec![Assumption::Consistency, Assumption::Positivity];
    if !matches!(query.design, antecedent_core::RandomizationDesign::Cluster { .. } | antecedent_core::RandomizationDesign::Switchback { .. }) {
        standard.push(Assumption::NoInterference);
    }
    for assumption in standard {
        assumptions.push(AssumptionRecord {
            assumption,
            source: AssumptionSource::UserDeclared,
            scope: AssumptionScope::Identification,
            status: AssumptionStatus::Declared,
        });
    }
    if query.received_treatment.is_some() {
        for (id, description) in [
            ("exclusion_restriction", "encouragement affects the outcome only through treatment receipt"),
            ("monotonicity_no_defiers", "encouragement does not reduce treatment receipt for any unit"),
            ("nonzero_receipt_first_stage", "randomized encouragement changes the probability of treatment receipt"),
        ] {
            assumptions.push(AssumptionRecord {
                assumption: Assumption::Custom { id: Arc::from(id), description: Arc::from(description) },
                source: AssumptionSource::UserDeclared,
                scope: AssumptionScope::Identification,
                status: AssumptionStatus::Declared,
            });
        }
        if query.estimand == antecedent_core::RandomizedEstimand::TreatmentOnTreated {
            assumptions.push(AssumptionRecord {
                assumption: Assumption::Custom {
                    id: Arc::from("one_sided_noncompliance"),
                    description: Arc::from("no control-assigned unit could receive treatment; the observed control-arm receipt check does not establish this counterfactual claim"),
                },
                source: AssumptionSource::UserDeclared,
                scope: AssumptionScope::Identification,
                status: AssumptionStatus::Declared,
            });
        }
    }
    if query.exact_randomization_test {
        assumptions.push(AssumptionRecord {
            assumption: Assumption::Custom {
                id: Arc::from("fisher_sharp_null_two_sided"),
                description: Arc::from("the reported exact p-value tests the sharp null of no unit-level assignment effect using the absolute difference in means over every fixed-count assignment"),
            },
            source: AssumptionSource::UserDeclared,
            scope: AssumptionScope::Estimation,
            status: AssumptionStatus::Declared,
        });
    }
    if matches!(query.design, antecedent_core::RandomizationDesign::Cluster { .. }) {
        assumptions.push(AssumptionRecord {
            assumption: Assumption::Custom {
                id: Arc::from("no_between_cluster_interference"),
                description: Arc::from("one cluster's assignment does not affect outcomes in another cluster; effects within a cluster are included in the cluster-assignment ITT"),
            },
            source: AssumptionSource::UserDeclared,
            scope: AssumptionScope::Identification,
            status: AssumptionStatus::Declared,
        });
    }
    if matches!(query.design, antecedent_core::RandomizationDesign::Switchback { .. }) {
        assumptions.push(AssumptionRecord {
            assumption: Assumption::Custom {
                id: Arc::from("switchback_no_carryover"),
                description: Arc::from("a unit-period outcome depends on its current assignment, not assignments in earlier periods"),
            },
            source: AssumptionSource::UserDeclared,
            scope: AssumptionScope::Identification,
            status: AssumptionStatus::Declared,
        });
        assumptions.push(AssumptionRecord {
            assumption: Assumption::Custom {
                id: Arc::from("independent_sequences_no_between_sequence_interference"),
                description: Arc::from("switching sequences are independent and treatment in one sequence does not affect another sequence's outcomes"),
            },
            source: AssumptionSource::UserDeclared,
            scope: AssumptionScope::Identification,
            status: AssumptionStatus::Declared,
        });
    }
    assumptions.push(AssumptionRecord {
        assumption: Assumption::Custom {
            id: Arc::from("known_random_assignment"),
            description: Arc::from(match &query.design {
                antecedent_core::RandomizationDesign::Bernoulli =>
                    "Treatment assignment follows the declared independent Bernoulli probabilities",
                antecedent_core::RandomizationDesign::Complete { .. } =>
                    "Treatment assignment follows complete randomization with the declared fixed treatment count",
                antecedent_core::RandomizationDesign::Stratified { .. } =>
                    "Treatment assignment follows independent complete randomization within each declared block",
                antecedent_core::RandomizationDesign::Factorial2x2 { .. } =>
                    "The two factors are jointly completely randomized to the declared four fixed cell counts",
                antecedent_core::RandomizationDesign::MultiArm { .. } =>
                    "Each unit is independently assigned to one of the declared actions with its known probability vector",
                antecedent_core::RandomizationDesign::Cluster { .. } =>
                    "Treatment assignment follows complete randomization of independent clusters; arbitrary dependence is allowed within clusters",
                antecedent_core::RandomizationDesign::Switchback { .. } =>
                    "Unit-period treatment has the declared known marginal assignment probabilities within independent switching sequences",
            }),
        },
        source: AssumptionSource::UserDeclared,
        scope: AssumptionScope::Identification,
        status: AssumptionStatus::Declared,
    });
    if query.fixed_cuped.is_some() {
        assumptions.push(AssumptionRecord {
            assumption: Assumption::Custom {
                id: Arc::from("fixed_pre_assignment_cuped"),
                description: Arc::from("the CUPED covariate was measured before randomization and its coefficient was fixed independently of these outcomes"),
            },
            source: AssumptionSource::UserDeclared,
            scope: AssumptionScope::Identification,
            status: AssumptionStatus::Declared,
        });
    }
    if !query.ancova_covariates.is_empty() {
        assumptions.push(AssumptionRecord {
            assumption: Assumption::Custom {
                id: Arc::from("ancova_pre_assignment_covariates"),
                description: Arc::from("ANCOVA covariates were measured before randomization; its HC0 variance treats independently assigned rows as the sampling units"),
            },
            source: AssumptionSource::UserDeclared,
            scope: AssumptionScope::Estimation,
            status: AssumptionStatus::Declared,
        });
    }
    let mut arena = CausalExprArena::new();
    let outcomes = arena.intern_var_set([query.outcome]);
    let empty = arena.empty_var_set();
    let no_intervention = arena.intern_intervention_set([]);
    let distribution = arena.intern_distribution(
        outcomes,
        empty,
        no_intervention,
        antecedent_expr::DomainRef::Observational,
    );
    let functional = arena.intern(antecedent_expr::ExprNode::Expectation {
        function: antecedent_expr::OutcomeExprId::identity(query.outcome),
        distribution,
    });
    let (identification_rule, identification_note) = if matches!(query.design, antecedent_core::RandomizationDesign::MultiArm { .. }) {
        ("randomized.multi_arm_itt", "known randomized action probabilities identify all arm means and their assignment contrasts")
    } else if query.estimand == antecedent_core::RandomizedEstimand::TreatmentOnTreated {
        ("randomized.one_sided_treatment_on_treated", "one-sided noncompliance identifies the effect among recipients through the Wald ratio under exclusion and random encouragement")
    } else if query.received_treatment.is_some() {
        ("randomized.wald_cace_late", "randomized encouragement identifies the Wald complier contrast under exclusion and monotonicity")
    } else {
        ("randomized.itt", "randomized assignment identifies the intention-to-treat contrast")
    };
    arena.set_derivation(
        functional,
        antecedent_expr::DerivationMeta::rule(
            identification_rule,
            Some(Arc::from(identification_note)),
        ),
    );
    let estimand = IdentifiedEstimand::new(
        identification_rule,
        Arc::from([]),
        Arc::from([]),
        Arc::from([]),
        functional,
        None,
    );
    let mut trace = DerivationTrace::default();
    trace.push(
        identification_rule,
        identification_note,
    );
    let result = IdentificationResult::identified(
        CausalQuery::RandomizedEffect(query.clone()),
        vec![estimand.clone()],
        arena,
        trace,
        assumptions,
        IdentificationPerformanceRecord::default(),
    );
    (result, estimand)
}

fn randomized_estimator_id(query: &antecedent_core::RandomizedEffectQuery) -> EstimatorId {
    match &query.design {
        antecedent_core::RandomizationDesign::Bernoulli => {
            if query.received_treatment.is_some() {
                EstimatorId::RandomizedWaldCace
            } else if query.fixed_cuped.is_some() {
                EstimatorId::RandomizedFixedCupedHt
            } else if !query.ancova_covariates.is_empty() {
                EstimatorId::RandomizedAncova
            } else {
                EstimatorId::RandomizedHt
            }
        },
        antecedent_core::RandomizationDesign::Complete { .. }
        | antecedent_core::RandomizationDesign::Stratified { .. }
        | antecedent_core::RandomizationDesign::Factorial2x2 { .. }
        | antecedent_core::RandomizationDesign::Cluster { .. } => EstimatorId::RandomizedNeyman,
        antecedent_core::RandomizationDesign::Switchback { .. } => EstimatorId::RandomizedSwitchbackHt,
        antecedent_core::RandomizationDesign::MultiArm { .. } => EstimatorId::RandomizedHt,
    }
}
