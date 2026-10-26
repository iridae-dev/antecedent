//! Streamed reduction for cross-fitted augmented inverse-probability weighting.
//!
//! The design `[1 | Z…]` is never built. The fold plan is the resident one
//! ([`crate::crossfit_plan::crossfit_fold_plan`] over the arm strata and the row identities,
//! under the fit's seed), held as one `u32` per kept row. For each of the
//! [`crate::crossfit_aipw::DEFAULT_AIPW_FOLDS`] folds the nuisances are fit on the training
//! rows by passes of the weighted-Gaussian capacity engine, each pass reading rows straight from
//! the source columns through the complete-case mask and a fold filter: the logistic propensity
//! by iteratively reweighted least squares (one pass per iteration), and the two arm outcome
//! regressions by one pass each. A scoring pass then reads the fold's validation rows once,
//! forms the out-of-fold scores `μ₁ + T/e (Y−μ₁)` and `μ₀ + (1−T)/(1−e) (Y−μ₀)`, and folds them
//! into running means and a running 2×2 covariance, so the estimate and the cross-fit iid
//! standard error are the resident score table's `summarize` and `linear_contrast` without the
//! `n × 2` table.
//!
//! The reduction is the normal equation, so it squares the design's condition number: it agrees
//! with the materialized fit to a relative tolerance set by that condition number, not bit for
//! bit. Only the closed-form route is streamed: the untrimmed all-observed mean effect, the
//! default unpenalized logistic propensity, the iid standard error, no sampling weights,
//! clustering or bootstrap, and a treatment that sits at the two declared levels. The fold
//! assignment is the only `O(n)` structure and holds four bytes per kept row. Per-row scores,
//! influence values and the retargetable score table are not retained.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::execution_facts::OutputRetention;
use antecedent_core::physical_choice::PhysicalChoice;
use antecedent_core::{AssumptionSet, AverageEffectQuery, OutcomeFunctional, TargetPopulation};
use antecedent_data::materialize::{PreparationControls, PreparationProduct, prepare_table};
use antecedent_data::{ColumnView, GlobalId, MissingPolicy, TableView, TabularData};
use antecedent_expr::IdentifiedEstimand;
use antecedent_io_core::capacity_account::CapacityOptions;

use crate::adjustment::EffectEstimate;
use crate::aipw::AipwAte;
use crate::crossfit_aipw::DEFAULT_AIPW_FOLDS;
use crate::crossfit_plan::crossfit_fold_plan;
use crate::error::EstimationError;
use crate::joint_if::JointCovariance;
use crate::util::dot;
use crate::overlap::{ClipSensitivity, IpwTarget, OverlapPolicy, OverlapReport, clip_propensity};
use crate::prepare::treatment_contrast;
use crate::propensity::{
    DeclaredTreatmentLevels, OtherLevels, PropensityNuisance, clip_of, trim_of,
};
use crate::se::AnalyticSeKind;
use crate::streamed_rows::{
    ArmFilter, EngineFacts, MaskedRows, RowFilter, RowMode, fit_logistic_streamed, run_engine,
};

/// Weight above which a weight is counted extreme, as the resident overlap report counts it.
const EXTREME_WEIGHT: f64 = 10.0;
/// Fitted probabilities this close to 0 or 1 are refused, as the resident fit refuses them.
const BOUNDARY: f64 = 1e-8;

/// Effect, cross-fit standard error and overlap report from a streamed reduction, with the facts
/// of the run.
#[derive(Clone, Debug)]
pub struct StreamedAipwEstimate {
    /// Mean of the out-of-fold treated scores minus the mean of the control scores.
    pub ate: f64,
    /// Cross-fit iid standard error of [`Self::ate`].
    pub se_analytic: f64,
    /// Covariance of the two arm-score means, column-major `[control, treated]`.
    pub joint_covariance: JointCovariance,
    /// Positivity diagnostics of the out-of-fold propensities.
    pub overlap_report: OverlapReport,
    /// The declared treatment levels and the rows at each.
    pub treatment_levels: DeclaredTreatmentLevels,
    /// Folds the nuisances were cross-fit over.
    pub folds: usize,
    /// Seed of the fold plan.
    pub fold_seed: u64,
    /// Logistic IRLS iterations of each fold's propensity fit.
    pub propensity_iterations: Vec<u32>,
    /// Largest number of bytes reserved against the ceiling by one engine pass.
    pub reserved_bytes: u64,
    /// Passes over the source rows, the engine's and the accumulating ones.
    pub source_passes: u64,
    /// Batches read in the last engine pass.
    pub batches: u64,
    /// The executed backend choice, with the alternatives rejected.
    pub choice: PhysicalChoice,
}

impl StreamedAipwEstimate {
    /// The estimate the resident fit would publish, less the per-row score table.
    #[must_use]
    pub fn into_effect(self, assumptions: AssumptionSet, overlap: OverlapPolicy) -> EffectEstimate {
        EffectEstimate::new(self.ate, self.se_analytic, assumptions, overlap)
            .with_se_kind(AnalyticSeKind::Homoskedastic)
            .with_overlap_report(Some(self.overlap_report))
            .with_joint_covariance(Some(self.joint_covariance))
            .with_treatment_levels(self.treatment_levels)
    }
}

/// One kept row: the propensity design row, its arm and its outcome.
struct RowView<'r> {
    design: &'r [f64],
    treated: bool,
    outcome: f64,
}

/// Visit every kept row with its place among the kept rows.
#[allow(
    clippy::float_cmp,
    reason = "the declared treatment level is an exact category label, not a computed value"
)]
fn for_each_kept_row(
    mask: &[bool],
    covariates: &[&[f64]],
    treatment: &[f64],
    outcome: &[f64],
    active: f64,
    mut visit: impl FnMut(usize, RowView<'_>),
) {
    let mut design = vec![0.0; 1 + covariates.len()];
    design[0] = 1.0;
    let mut place = 0;
    for (row, &kept) in mask.iter().enumerate() {
        if !kept {
            continue;
        }
        for (slot, column) in design[1..].iter_mut().zip(covariates) {
            *slot = column[row];
        }
        visit(
            place,
            RowView { design: &design, treated: treatment[row] == active, outcome: outcome[row] },
        );
        place += 1;
    }
}

/// Fitted propensity of a design row, kept off 0 and 1 as the resident scores are.
fn propensity(design: &[f64], gamma: &[f64]) -> f64 {
    let eta: f64 = design.iter().zip(gamma).map(|(x, g)| x * g).sum();
    antecedent_kernels::sigmoid(eta).clamp(1e-9, 1.0 - 1e-9)
}

/// Running sums of one weight vector: all rows and each arm.
#[derive(Clone, Copy, Debug, Default)]
struct WeightSums {
    sum: f64,
    sum_sq: f64,
    extreme: u64,
    treated_sum: f64,
    treated_sum_sq: f64,
    control_sum: f64,
    control_sum_sq: f64,
}

impl WeightSums {
    fn push(&mut self, weight: f64, treated: bool) {
        self.sum += weight;
        self.sum_sq += weight * weight;
        if weight > EXTREME_WEIGHT {
            self.extreme += 1;
        }
        if treated {
            self.treated_sum += weight;
            self.treated_sum_sq += weight * weight;
        } else {
            self.control_sum += weight;
            self.control_sum_sq += weight * weight;
        }
    }

    fn ess(sum: f64, sum_sq: f64) -> f64 {
        if sum_sq > 0.0 { (sum * sum) / sum_sq } else { 0.0 }
    }

    fn extreme_count(&self) -> u32 {
        u32::try_from(self.extreme).unwrap_or(u32::MAX)
    }
}

/// Running means and sums of centered cross-products of the two arm scores.
#[derive(Clone, Copy, Debug, Default)]
struct ScoreMoments {
    count: f64,
    mean: [f64; 2],
    m2: [[f64; 2]; 2],
}

impl ScoreMoments {
    fn push(&mut self, control: f64, treated: f64) {
        self.count += 1.0;
        let before = [control - self.mean[0], treated - self.mean[1]];
        self.mean[0] += before[0] / self.count;
        self.mean[1] += before[1] / self.count;
        let after = [control - self.mean[0], treated - self.mean[1]];
        self.m2[0][0] += before[0] * after[0];
        self.m2[1][1] += before[1] * after[1];
        self.m2[0][1] += before[0] * after[1];
        self.m2[1][0] = self.m2[0][1];
    }
}

impl AipwAte {
    /// Whether this estimator's result follows in closed form from streamed row passes: the
    /// default unpenalized logistic propensity, untrimmed weights, the iid standard error, and
    /// no sampling weights, clustering, cluster-DML, bootstrap or rows at undeclared levels.
    #[must_use]
    pub fn supports_streamed_reduction(&self) -> bool {
        self.bootstrap_replicates == 0
            && self.bootstrap_interval.is_none()
            && matches!(self.se_kind, AnalyticSeKind::Homoskedastic)
            && self.cluster_ids.is_none()
            && self.source_cluster_ids.is_none()
            && self.cluster_ids_second.is_none()
            && self.source_cluster_ids_second.is_none()
            && self.multiway_ids.is_none()
            && self.panel_times.is_none()
            && self.propensity == PropensityNuisance::default()
            && self.cluster_dml.is_none()
            && self.other_levels == OtherLevels::Refuse
            && self.sampling_weights.is_none()
            && trim_of(self.overlap).is_none()
    }

    /// Estimate the cross-fitted average effect by streaming the rows in batches of
    /// `batch_rows`, holding at most `ceiling_bytes` per engine pass, without building the
    /// design. `fold_seed` is the seed the resident fit takes from its context.
    ///
    /// # Errors
    ///
    /// The estimator is not streamable, the estimand or query is not an all-observed mean
    /// adjustment, a column is not float64, a treatment row sits at neither declared level, a
    /// fold is empty or misses an arm, a fold's logistic fit does not converge or reaches the
    /// boundary, an outcome regression is rank deficient, a propensity is 0 or 1 without a
    /// clip, or the reduction exceeds its ceiling.
    #[allow(
        clippy::too_many_lines,
        clippy::float_cmp,
        reason = "one reduction: the fold plan, then per fold the nuisance passes and the scoring pass, read in order; the declared treatment levels are exact category labels"
    )]
    pub fn estimate_streamed(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
        fold_seed: u64,
        batch_rows: usize,
        ceiling_bytes: u64,
    ) -> Result<StreamedAipwEstimate, EstimationError> {
        if !self.supports_streamed_reduction() {
            return Err(EstimationError::unsupported(
                "a streamed reduction needs the default logistic propensity, untrimmed weights, \
                 the iid standard error, and no sampling weights, clustering, cluster-DML or \
                 bootstrap",
            ));
        }
        crate::util::refuse_explicit_override(
            self.overlap,
            "propensity estimators require RequireDiagnostics overlap policy; positivity is mandatory",
        )?;
        self.overlap.validate()?;
        if !estimand.is_adjustment_shaped() {
            return Err(EstimationError::IncompatibleEstimand {
                message: "propensity estimators expect an adjustment-shaped estimand",
            });
        }
        query.validate()?;
        if !query.effect_modifiers.is_empty() {
            return Err(EstimationError::unsupported(
                "propensity estimators do not support effect modifiers",
            ));
        }
        if query.target_population != TargetPopulation::AllObserved
            || query.outcome_functional != OutcomeFunctional::Mean
        {
            return Err(EstimationError::unsupported(
                "a streamed reduction estimates the all-observed mean effect",
            ));
        }
        let (active, control, _) = treatment_contrast(&query.active, &query.control)?;
        let clip = clip_of(self.overlap);
        let folds = DEFAULT_AIPW_FOLDS;

        let mut ids = Vec::with_capacity(2 + estimand.adjustment_set.len());
        ids.push(query.treatment);
        ids.push(query.outcome);
        ids.extend_from_slice(&estimand.adjustment_set);
        let PreparationProduct { row_mask, .. } = prepare_table(
            data,
            &ids,
            MissingPolicy::CompleteCase,
            &[],
            PreparationControls::finite_resident(),
        )?;
        let column = |id| -> Result<&[f64], EstimationError> {
            match data.column(id)? {
                ColumnView::Float64(c) => Ok(c.values.as_slice()),
                _ => Err(EstimationError::unsupported(
                    "a streamed reduction reads float64 design columns",
                )),
            }
        };
        let mut covariates: Vec<&[f64]> = Vec::with_capacity(estimand.adjustment_set.len());
        for &id in estimand.adjustment_set.iter() {
            covariates.push(column(id)?);
        }
        let treatment = column(query.treatment)?;
        let outcome = column(query.outcome)?;

        // The kept rows: the level each sits at, its arm stratum and its identity.
        let active_is_higher = active > control;
        let (mut at_active, mut at_control, mut elsewhere) = (0_u64, 0_u64, 0_u64);
        let mut strata: Vec<u32> = Vec::new();
        let mut units: Vec<GlobalId> = Vec::new();
        for (row, &kept) in row_mask.iter().enumerate() {
            if !kept {
                continue;
            }
            let level = treatment[row];
            if level == active {
                at_active += 1;
                strata.push(u32::from(active_is_higher));
            } else if level == control {
                at_control += 1;
                strata.push(u32::from(!active_is_higher));
            } else {
                elsewhere += 1;
                continue;
            }
            units.push(GlobalId::new(u64::try_from(row).unwrap_or(u64::MAX)));
        }
        if at_active + at_control == 0 {
            return Err(EstimationError::data_msg("no complete-case rows for propensity design"));
        }
        if elsewhere > 0 {
            return Err(EstimationError::unsupported(
                "the treatment has rows at neither declared level; a streamed reduction does \
                 not exclude them",
            ));
        }
        let rows = at_active + at_control;
        if usize::try_from(rows).unwrap_or(usize::MAX) < folds {
            return Err(EstimationError::data_msg(
                "cross-fitting folds cannot exceed complete rows",
            ));
        }
        let fold_ids = crossfit_fold_plan(&strata, &units, folds, fold_seed)?;
        drop((strata, units));

        // Rows of each fold and arm, so an empty fold or a missing arm refuses before any fit.
        let mut counts = vec![[0_u64; 2]; folds];
        for_each_kept_row(&row_mask, &covariates, treatment, outcome, active, |place, row| {
            counts[fold_ids[place] as usize][usize::from(row.treated)] += 1;
        });
        let arm_totals =
            [counts.iter().map(|c| c[0]).sum::<u64>(), counts.iter().map(|c| c[1]).sum::<u64>()];
        for fold_counts in &counts {
            if fold_counts[0] + fold_counts[1] == 0
                || arm_totals[0] == fold_counts[0]
                || arm_totals[1] == fold_counts[1]
            {
                return Err(EstimationError::data_msg(if fold_counts[0] + fold_counts[1] == 0 {
                    "AIPW cross-fit fold is empty"
                } else {
                    "AIPW cross-fit fold is missing a treatment arm"
                }));
            }
        }

        let options = CapacityOptions::new(batch_rows, ceiling_bytes, OutputRetention::SummaryOnly);
        let mut facts = EngineFacts::default();
        let mut iterations = Vec::with_capacity(folds);
        let thresholds: Vec<f64> = match clip {
            Some(c) => {
                let mut kept: Vec<f64> = Vec::with_capacity(3);
                for candidate in [c * 0.5, c, (c * 2.0).min(0.49)] {
                    if kept.last().is_some_and(|previous| (previous - candidate).abs() < 1e-15) {
                        continue;
                    }
                    kept.push(candidate);
                }
                kept
            }
            None => Vec::new(),
        };
        let mut sensitivity = vec![WeightSums::default(); thresholds.len()];
        let mut moments = ScoreMoments::default();
        let (mut min_p, mut max_p) = (f64::INFINITY, f64::NEG_INFINITY);
        let mut in_support = 0_u64;
        let support_lo = clip.unwrap_or(0.0);
        let support_hi = 1.0 - support_lo;
        let mut refusal: Option<EstimationError> = None;
        for fold in 0..folds {
            let fold_id = u32::try_from(fold).unwrap_or(u32::MAX);
            let training =
                RowFilter { fold_ids: &fold_ids, exclude_fold: Some(fold_id), arm: None };
            let (gamma, steps) = fit_logistic_streamed(
                &covariates,
                treatment,
                active,
                &row_mask,
                Some(training),
                &options,
                &self.glm_options,
                &mut facts,
            )?;
            iterations.push(steps);
            facts.passes += 1;
            let mut saturated = false;
            for_each_kept_row(&row_mask, &covariates, treatment, outcome, active, |place, row| {
                if fold_ids[place] != fold_id {
                    let e = propensity(row.design, &gamma);
                    saturated |= e <= BOUNDARY || e >= 1.0 - BOUNDARY;
                }
            });
            if saturated {
                return Err(EstimationError::data_msg(
                    "GLM fitted probabilities lie within 1e-8 of 0 or 1 (extreme scores, not \
                     necessarily separation); refuse propensity/outcome scores",
                ));
            }
            // The arm outcome regressions on the training rows.
            let mut beta = [Vec::new(), Vec::new()];
            for treated in [false, true] {
                let filter = RowFilter {
                    fold_ids: &fold_ids,
                    exclude_fold: Some(fold_id),
                    arm: Some(ArmFilter { treatment, active, treated }),
                };
                let mut source = MaskedRows::new(
                    true,
                    covariates.clone(),
                    RowMode::Plain { outcome },
                    &row_mask,
                )
                .filtered(filter);
                beta[usize::from(treated)]
                    .clone_from(&run_engine(&mut source, &options, &mut facts)?.coefficients);
            }
            // The scoring pass over the fold's validation rows.
            facts.passes += 1;
            for_each_kept_row(&row_mask, &covariates, treatment, outcome, active, |place, row| {
                if fold_ids[place] != fold_id || refusal.is_some() {
                    return;
                }
                let raw = propensity(row.design, &gamma);
                let e = clip.map_or(raw, |c| clip_propensity(raw, c));
                if !(e > 0.0 && e < 1.0) {
                    refusal = Some(EstimationError::Overlap {
                        message: "a fitted propensity is 0 or 1 (or not finite) and no clip is \
                                  set; the inverse-probability weight is infinite — set an \
                                  overlap clip",
                    });
                    return;
                }
                let t = if row.treated { 1.0 } else { 0.0 };
                let m0 = dot(row.design, &beta[0]);
                let m1 = dot(row.design, &beta[1]);
                let control_score = m0 + ((1.0 - t) / (1.0 - e)) * (row.outcome - m0);
                let treated_score = m1 + (t / e) * (row.outcome - m1);
                moments.push(control_score, treated_score);
                min_p = min_p.min(raw);
                max_p = max_p.max(raw);
                if raw >= support_lo && raw <= support_hi {
                    in_support += 1;
                }
                for (sums, &threshold) in sensitivity.iter_mut().zip(&thresholds) {
                    sums.push(
                        IpwTarget::Ate.weight(t, clip_propensity(raw, threshold)),
                        row.treated,
                    );
                }
            });
            if let Some(error) = refusal.take() {
                return Err(error);
            }
        }

        let nf = moments.count;
        if nf < 2.0 {
            return Err(EstimationError::data_msg("joint IF requires at least two rows"));
        }
        let scale = nf * (nf - 1.0);
        let covariance = [
            moments.m2[0][0] / scale,
            moments.m2[1][0] / scale,
            moments.m2[0][1] / scale,
            moments.m2[1][1] / scale,
        ];
        let ate = (-1.0) * moments.mean[0] + moments.mean[1];
        let variance = covariance[0] + covariance[3] - covariance[1] - covariance[2];
        if !ate.is_finite() || !variance.is_finite() {
            return Err(EstimationError::data_msg("contrast value or variance is non-finite"));
        }
        let overlap_report = OverlapReport {
            propensity_min: min_p,
            propensity_max: max_p,
            ess: None,
            extreme_weight_count: 0,
            excluded_fraction: 0.0,
            target_population_support: in_support as f64 / nf.max(1.0),
            excluded_regions: Arc::from([]),
            clip,
            trim: None,
            retained_fraction: 1.0,
            clip_sensitivity: clip.map(|_| ClipSensitivity {
                thresholds: Arc::from(thresholds.clone()),
                ess: sensitivity.iter().map(|s| WeightSums::ess(s.sum, s.sum_sq)).collect(),
                treated_ess: sensitivity
                    .iter()
                    .map(|s| WeightSums::ess(s.treated_sum, s.treated_sum_sq))
                    .collect(),
                control_ess: sensitivity
                    .iter()
                    .map(|s| WeightSums::ess(s.control_sum, s.control_sum_sq))
                    .collect(),
                extreme_weight_counts: sensitivity.iter().map(WeightSums::extreme_count).collect(),
            }),
        };
        let choice = facts.choice.ok_or_else(|| EstimationError::data_msg("no engine pass ran"))?;
        Ok(StreamedAipwEstimate {
            ate,
            se_analytic: variance.max(0.0).sqrt(),
            joint_covariance: JointCovariance { dim: 2, values: Arc::from(covariance) },
            overlap_report,
            treatment_levels: DeclaredTreatmentLevels {
                active,
                control,
                other_levels: OtherLevels::Refuse,
                rows_at_active: at_active,
                rows_at_control: at_control,
                excluded_rows: 0,
            },
            folds,
            fold_seed,
            propensity_iterations: iterations,
            reserved_bytes: facts.reserved,
            // The engine's passes, the count pass, and a boundary and a scoring pass per fold.
            source_passes: facts.passes + 2,
            batches: facts.batches,
            choice,
        })
    }
}

#[cfg(test)]
mod tests {
    use antecedent_core::{AssumptionSet, AverageEffectQuery, ExecutionContext, VariableId};
    use antecedent_expr::{ExprId, IdentifiedEstimand};

    use super::*;
    use crate::aipw::AipwWorkspace;
    use crate::sampling_weights::SamplingWeights;

    fn noise(i: usize, k: usize) -> f64 {
        (((i * 31 + k * 977) as f64 * 12.9898).sin() * 43758.5453).fract() - 0.5
    }

    /// Treatment is a logistic function of `z1`, `z2`; `y = 1 + 2 t + 0.5 z1 - 0.3 z2 + noise`.
    fn fixture(
        rows: usize,
        stray_level: bool,
    ) -> (TabularData, IdentifiedEstimand, AverageEffectQuery) {
        let z1: Vec<f64> = (0..rows).map(|i| noise(i, 1) * 3.0).collect();
        let z2: Vec<f64> = (0..rows).map(|i| noise(i, 2) * 2.0).collect();
        let t: Vec<f64> = (0..rows)
            .map(|i| {
                if stray_level && i == 0 {
                    return 2.0;
                }
                let eta = 0.6 * z1[i] - 0.4 * z2[i];
                f64::from(0.5 + noise(i, 3) < crate::util::sigmoid(eta))
            })
            .collect();
        let y: Vec<f64> = (0..rows)
            .map(|i| 1.0 + 2.0 * t[i] + 0.5 * z1[i] - 0.3 * z2[i] + 0.4 * noise(i, 4))
            .collect();
        let data = TabularData::from_f64_columns([
            ("t", t.as_slice()),
            ("y", y.as_slice()),
            ("z1", z1.as_slice()),
            ("z2", z2.as_slice()),
        ])
        .unwrap();
        let estimand = IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from([VariableId::from_raw(2), VariableId::from_raw(3)]),
            ExprId::from_raw(0),
        );
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        (data, estimand, query)
    }

    fn close(left: f64, right: f64, tolerance: f64) -> bool {
        (left - right).abs() <= tolerance * left.abs().max(right.abs()).max(1.0)
    }

    #[test]
    fn the_streamed_reduction_equals_the_resident_cross_fit_estimate_standard_error_and_report() {
        let (data, estimand, query) = fixture(3_000, false);
        let estimator = AipwAte::new().with_bootstrap_replicates(0);
        assert!(estimator.supports_streamed_reduction());
        let problem = estimator.prepare(&data, &estimand, &query).unwrap();
        let ctx = ExecutionContext::for_tests(7);
        let mut workspace = AipwWorkspace::default();
        let resident = estimator.fit(&problem, &mut workspace, &ctx, AssumptionSet::new()).unwrap();
        // The same fold plan: the seed the resident fit reads from its context.
        let streamed = estimator
            .estimate_streamed(&data, &estimand, &query, ctx.rng.master_seed(), 97, 1 << 24)
            .unwrap();
        assert!(streamed.batches > 1, "the rows were read in several batches");
        assert_eq!(streamed.folds, 5);
        assert!(
            close(streamed.ate, resident.ate, 1e-7),
            "ate {} vs {}",
            streamed.ate,
            resident.ate
        );
        assert!(
            close(streamed.se_analytic, resident.se_analytic(), 1e-6),
            "se {} vs {}",
            streamed.se_analytic,
            resident.se_analytic()
        );
        let want = resident.joint.covariance.as_ref().expect("resident covariance");
        for (a, b) in streamed.joint_covariance.values.iter().zip(want.values.iter()) {
            assert!(close(*a, *b, 1e-6), "covariance {a} vs {b}");
        }
        let want = resident.diagnostics.overlap_report.as_ref().expect("resident report");
        let got = &streamed.overlap_report;
        assert!(close(got.propensity_min, want.propensity_min, 1e-7));
        assert!(close(got.propensity_max, want.propensity_max, 1e-7));
        assert_eq!(got.ess, want.ess);
        assert_eq!(got.extreme_weight_count, want.extreme_weight_count);
        assert!(close(got.target_population_support, want.target_population_support, 1e-12));
        let (got_sens, want_sens) =
            (got.clip_sensitivity.as_ref().unwrap(), want.clip_sensitivity.as_ref().unwrap());
        assert_eq!(got_sens.thresholds, want_sens.thresholds);
        assert_eq!(got_sens.extreme_weight_counts, want_sens.extreme_weight_counts);
        for (a, b) in got_sens.ess.iter().zip(want_sens.ess.iter()) {
            assert!(close(*a, *b, 1e-7));
        }
        assert_eq!(Some(streamed.treatment_levels), resident.diagnostics.treatment_levels);
        // A different seed is a different fold plan and a different estimate.
        let other = estimator
            .estimate_streamed(
                &data,
                &estimand,
                &query,
                ctx.rng.master_seed() ^ 0x5eed,
                97,
                1 << 24,
            )
            .unwrap();
        assert!(other.ate != streamed.ate);
    }

    #[test]
    fn a_configuration_the_cross_fit_reduction_does_not_cover_is_refused() {
        let (data, estimand, query) = fixture(600, false);
        let estimator = AipwAte::new().with_bootstrap_replicates(0);
        // A ceiling below the window refuses instead of reading.
        assert!(estimator.estimate_streamed(&data, &estimand, &query, 1, 64, 8).is_err());
        let refused = [
            AipwAte::new(),
            estimator.clone().with_se_kind(AnalyticSeKind::Hc1),
            estimator.clone().with_cluster_ids(vec![0; 600]),
            estimator.clone().with_overlap(OverlapPolicy::RequireDiagnostics {
                clip: Some(0.01),
                trim: Some(0.05),
            }),
            estimator.clone().with_sampling_weights(SamplingWeights::new(vec![1.0; 600]).unwrap()),
        ];
        for refused in &refused {
            assert!(!refused.supports_streamed_reduction());
            assert!(refused.estimate_streamed(&data, &estimand, &query, 1, 64, 1 << 20).is_err());
        }
        // A treatment row at neither declared level is not silently dropped.
        let (stray, estimand, query) = fixture(600, true);
        assert!(estimator.estimate_streamed(&stray, &estimand, &query, 1, 64, 1 << 20).is_err());
    }
}
