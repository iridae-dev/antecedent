//! Streamed reduction for inverse-probability weighting.
//!
//! The propensity design `[1 | Z…]` is never built. The logistic propensity model is fit by
//! iteratively reweighted least squares in which every iteration is one pass of the
//! weighted-Gaussian capacity engine over rows read straight from the source columns: the
//! working weight `μ(1−μ)` and working response `η + (t−μ)/(μ(1−μ))` are formed per row at the
//! current coefficients, so the engine's solution is exactly the next IRLS iterate and the
//! iteration is the resident one (start at zero, the same clamp, the same coefficient-change
//! tolerance). The Hájek difference, the overlap report and the analytic standard error then
//! follow from three more row passes that hold only `p`-sized accumulators: the weighted arm
//! sums, the logistic information and the weight-derivative, and the corrected influence.
//!
//! The reduction is the normal equation, so it squares the design's condition number: it
//! agrees with the materialized fit to a relative tolerance set by that condition number, not
//! bit for bit. Only the closed-form route is streamed: the average effect on all observed
//! rows, clipped but not trimmed weights, an iid analytic standard error, no sampling weights,
//! clustering, calibration or bootstrap, and a treatment that sits at the two declared levels.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::execution_facts::OutputRetention;
use antecedent_core::physical_choice::PhysicalChoice;
use antecedent_core::{AssumptionSet, AverageEffectQuery, TargetPopulation};
use antecedent_data::materialize::{PreparationControls, PreparationProduct, prepare_table};
use antecedent_data::{ColumnView, MissingPolicy, TableView, TabularData};
use antecedent_expr::IdentifiedEstimand;
use antecedent_io_core::capacity_account::{
    CapacityControl, CapacityOptions, run_weighted_capacity,
};

use crate::adjustment::EffectEstimate;
use crate::error::EstimationError;
use crate::overlap::{ClipSensitivity, IpwTarget, OverlapPolicy, OverlapReport, clip_propensity};
use crate::prepare::treatment_contrast;
use crate::propensity::{DeclaredTreatmentLevels, OtherLevels, PropensityWeighting};
use crate::propensity::{clip_of, trim_of};
use crate::streamed_rows::{MaskedRows, RowMode};

/// Weight above which a weight is counted extreme, as the resident overlap report counts it.
const EXTREME_WEIGHT: f64 = 10.0;
/// Fitted probabilities this close to 0 or 1 are refused, as the resident fit refuses them.
const BOUNDARY: f64 = 1e-8;

/// Effect, standard error and overlap report from a streamed reduction, with the facts of the
/// run.
#[derive(Clone, Debug)]
pub struct StreamedPropensityEstimate {
    /// Hájek-weighted average effect.
    pub ate: f64,
    /// Analytic standard error with the logistic-estimation correction.
    pub se_analytic: f64,
    /// Rows with a positive finite weight.
    pub n_obs: u64,
    /// Propensity coefficients: the intercept, then one per adjustment column.
    pub coefficients: Vec<f64>,
    /// IRLS iterations until the coefficient change fell below the tolerance.
    pub iterations: u32,
    /// Positivity diagnostics, built from running accumulators.
    pub overlap_report: OverlapReport,
    /// The declared treatment levels and the rows at each.
    pub treatment_levels: DeclaredTreatmentLevels,
    /// Largest number of bytes reserved against the ceiling by one engine pass.
    pub reserved_bytes: u64,
    /// Passes over the source rows, the engine's and the accumulating ones.
    pub source_passes: u64,
    /// Batches read in the final engine pass.
    pub batches: u64,
    /// The executed backend choice, with the alternatives rejected.
    pub choice: PhysicalChoice,
}

impl StreamedPropensityEstimate {
    /// The estimate the resident fit would publish.
    #[must_use]
    pub fn into_effect(self, assumptions: AssumptionSet, overlap: OverlapPolicy) -> EffectEstimate {
        EffectEstimate::new(self.ate, self.se_analytic, assumptions, overlap)
            .with_n_obs(self.n_obs)
            .with_overlap_report(Some(self.overlap_report))
            .with_treatment_levels(self.treatment_levels)
    }
}

/// One row of the propensity design with its treatment and outcome.
struct RowView<'r> {
    design: &'r [f64],
    treated: bool,
    outcome: f64,
}

/// Visit every row the mask keeps.
#[allow(
    clippy::float_cmp,
    reason = "the declared treatment level is an exact category label, not a computed value"
)]
fn for_each_row(
    mask: &[bool],
    covariates: &[&[f64]],
    treatment: &[f64],
    outcome: &[f64],
    active: f64,
    mut visit: impl FnMut(RowView<'_>),
) {
    let mut design = vec![0.0; 1 + covariates.len()];
    design[0] = 1.0;
    for (row, &kept) in mask.iter().enumerate() {
        if !kept {
            continue;
        }
        for (slot, column) in design[1..].iter_mut().zip(covariates) {
            *slot = column[row];
        }
        visit(RowView {
            design: &design,
            treated: treatment[row] == active,
            outcome: outcome[row],
        });
    }
}

/// Fitted propensity of a design row: the unclamped logistic mean, kept off 0 and 1 as the
/// resident scores are.
fn score(design: &[f64], beta: &[f64]) -> f64 {
    let eta: f64 = design.iter().zip(beta).map(|(x, b)| x * b).sum();
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

impl PropensityWeighting {
    /// Whether this estimator's result follows in closed form from streamed row passes: the
    /// iid analytic standard error, untrimmed weights, no sampling weights, clustering,
    /// calibration, bootstrap, or rows at undeclared treatment levels.
    #[must_use]
    pub fn supports_streamed_reduction(&self) -> bool {
        self.bootstrap_replicates == 0
            && self.bootstrap_interval.is_none()
            && !self.covariate_balance
            && self.sampling_weights.is_none()
            && self.cluster_ids.is_none()
            && self.source_cluster_ids.is_none()
            && self.other_levels == OtherLevels::Refuse
            && trim_of(self.overlap).is_none()
    }

    /// Estimate the average effect by streaming the rows in batches of `batch_rows`, holding
    /// at most `ceiling_bytes` per engine pass, without building the design.
    ///
    /// # Errors
    ///
    /// The estimator is not streamable, the estimand or query is not an all-observed
    /// adjustment, a column is not float64, a treatment row sits at neither declared level,
    /// the logistic fit does not converge or its probabilities reach the boundary, an arm
    /// has no weight, or the reduction exceeds its ceiling.
    #[allow(
        clippy::too_many_lines,
        clippy::float_cmp,
        reason = "one reduction: the IRLS passes, then the three accumulating passes, read in order; the declared treatment levels are exact category labels"
    )]
    pub fn estimate_streamed(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
        batch_rows: usize,
        ceiling_bytes: u64,
    ) -> Result<StreamedPropensityEstimate, EstimationError> {
        if !self.supports_streamed_reduction() {
            return Err(EstimationError::unsupported(
                "a streamed reduction needs untrimmed weights, the iid analytic standard error, \
                 and no sampling weights, clustering, calibration or bootstrap",
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
        if query.target_population != TargetPopulation::AllObserved {
            return Err(EstimationError::unsupported(
                "a streamed reduction estimates the all-observed population",
            ));
        }
        let target = IpwTarget::Ate;
        let (active, control, _) = treatment_contrast(&query.active, &query.control)?;
        let clip = clip_of(self.overlap);

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

        // Rows at the declared levels, and none at any other.
        let (mut rows, mut at_active, mut at_control, mut elsewhere) = (0_u64, 0_u64, 0_u64, 0_u64);
        for (row, &kept) in row_mask.iter().enumerate() {
            if !kept {
                continue;
            }
            rows += 1;
            if treatment[row] == active {
                at_active += 1;
            } else if treatment[row] == control {
                at_control += 1;
            } else {
                elsewhere += 1;
            }
        }
        if rows == 0 {
            return Err(EstimationError::data_msg("no complete-case rows for propensity design"));
        }
        if elsewhere > 0 {
            return Err(EstimationError::unsupported(
                "the treatment has rows at neither declared level; a streamed reduction does \
                 not exclude them",
            ));
        }

        // Logistic IRLS: one engine pass per iteration.
        let ncols = 1 + covariates.len();
        let options = CapacityOptions::new(batch_rows, ceiling_bytes, OutputRetention::SummaryOnly);
        let mut beta = vec![0.0; ncols];
        let mut converged = false;
        let mut iterations = 0_u32;
        let mut source_passes = 0_u64;
        let mut reserved_bytes = 0_u64;
        let mut batches = 0_u64;
        let mut choice = None;
        for iteration in 1..=self.glm_options.max_iter {
            iterations = iteration;
            let result = {
                let mut source = MaskedRows::new(
                    true,
                    covariates.clone(),
                    RowMode::LogisticWorking { treatment, active, beta: &beta },
                    &row_mask,
                );
                run_weighted_capacity(CapacityControl::Glm, &mut source, &options)
                    .map_err(|error| EstimationError::data_msg(error.to_string()))?
                    .into_resident()
                    .ok_or_else(|| {
                        EstimationError::data_msg("the reduction summary was not resident")
                    })?
            };
            source_passes += result.source_passes;
            reserved_bytes = reserved_bytes.max(result.charges.total_bytes);
            batches = result.batches;
            choice = Some(*result.physical_choice());
            if result.coefficients.len() != ncols
                || result.coefficients.iter().any(|value| !value.is_finite())
            {
                return Err(EstimationError::data_msg(
                    "the logistic propensity step produced a non-finite coefficient",
                ));
            }
            let max_delta = result
                .coefficients
                .iter()
                .zip(&beta)
                .map(|(new, old)| (new - old).abs())
                .fold(0.0_f64, f64::max);
            beta.clone_from(&result.coefficients);
            if max_delta < self.glm_options.tol {
                converged = true;
                break;
            }
        }
        if !converged {
            return Err(EstimationError::data_msg(
                "GLM IRLS did not converge; refuse propensity/outcome scores",
            ));
        }
        let choice = choice.ok_or_else(|| EstimationError::data_msg("no IRLS pass ran"))?;

        // Pass A: arm sums of the Hájek means, the overlap accumulators, the boundary check.
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
        let mut applied = WeightSums::default();
        let (mut sw1, mut sw0, mut sy1, mut sy0) = (0.0, 0.0, 0.0, 0.0);
        let (mut min_p, mut max_p) = (f64::INFINITY, f64::NEG_INFINITY);
        let (mut in_support, mut boundary, mut used) = (0_u64, 0_u64, 0_u64);
        let support_lo = clip.unwrap_or(0.0);
        let support_hi = 1.0 - support_lo;
        for_each_row(&row_mask, &covariates, treatment, outcome, active, |row| {
            let raw = score(row.design, &beta);
            min_p = min_p.min(raw);
            max_p = max_p.max(raw);
            if raw >= support_lo && raw <= support_hi {
                in_support += 1;
            }
            if raw <= BOUNDARY || raw >= 1.0 - BOUNDARY {
                boundary += 1;
            }
            let t = if row.treated { 1.0 } else { 0.0 };
            let weight_score = clip.map_or(raw, |c| clip_propensity(raw, c));
            let weight = target.weight(t, weight_score);
            applied.push(weight, row.treated);
            if weight.is_finite() && weight > 0.0 {
                used += 1;
            }
            if row.treated {
                sw1 += weight;
                sy1 += weight * row.outcome;
            } else {
                sw0 += weight;
                sy0 += weight * row.outcome;
            }
            for (sums, &threshold) in sensitivity.iter_mut().zip(&thresholds) {
                sums.push(target.weight(t, clip_propensity(raw, threshold)), row.treated);
            }
        });
        if boundary > 0 {
            return Err(EstimationError::data_msg(
                "GLM fitted probabilities lie within 1e-8 of 0 or 1 (extreme scores, not \
                 necessarily separation); refuse propensity/outcome scores",
            ));
        }
        if sw1 <= 0.0 || sw0 <= 0.0 {
            return Err(EstimationError::data_msg(
                "IPW weighting left an arm with zero total weight (trimming/clipping removed all treated or all control units)",
            ));
        }
        let (mu1, mu0) = (sy1 / sw1, sy0 / sw0);
        let ate = mu1 - mu0;
        let nf = rows as f64;

        // Pass B: the logistic information and the derivative of the weighted means with
        // respect to the propensity coefficients.
        let mut information = vec![0.0; ncols * ncols];
        let mut derivative = vec![0.0; ncols];
        for_each_row(&row_mask, &covariates, treatment, outcome, active, |row| {
            let raw = score(row.design, &beta);
            let t = if row.treated { 1.0 } else { 0.0 };
            let weight_score = clip.map_or(raw, |c| clip_propensity(raw, c));
            let weight = target.weight(t, weight_score);
            let psi = linearized(weight, row.treated, row.outcome, (sw1, sw0), (mu1, mu0), nf);
            let log_weight_derivative = if clip.is_some_and(|c| raw <= c || raw >= 1.0 - c) {
                0.0
            } else if row.treated {
                -(1.0 - raw)
            } else {
                raw
            };
            let derivative_weight = psi * log_weight_derivative;
            let information_weight = raw * (1.0 - raw);
            for c in 0..ncols {
                derivative[c] += derivative_weight * row.design[c] / nf;
                for d in 0..ncols {
                    information[c * ncols + d] +=
                        information_weight * row.design[c] * row.design[d] / nf;
                }
            }
        });
        let alpha = crate::util::solve_spd(&information, &derivative, ncols).ok_or_else(|| {
            EstimationError::stats_msg("singular logistic information in Hajek SE")
        })?;

        // Pass C: the corrected influence, accumulated as a running mean and sum of squares.
        let (mut mean, mut m2, mut seen) = (0.0_f64, 0.0_f64, 0.0_f64);
        for_each_row(&row_mask, &covariates, treatment, outcome, active, |row| {
            let raw = score(row.design, &beta);
            let t = if row.treated { 1.0 } else { 0.0 };
            let weight_score = clip.map_or(raw, |c| clip_propensity(raw, c));
            let weight = target.weight(t, weight_score);
            let psi = linearized(weight, row.treated, row.outcome, (sw1, sw0), (mu1, mu0), nf);
            let adjustment: f64 = row.design.iter().zip(&alpha).map(|(x, a)| x * a).sum();
            let corrected = psi + adjustment * (t - raw);
            seen += 1.0;
            let delta = corrected - mean;
            mean += delta / seen;
            m2 += delta * (corrected - mean);
        });
        let se_analytic = if rows < 2 {
            f64::NAN
        } else {
            let variance = m2 / (nf - 1.0);
            let df = (nf - ncols.max(1) as f64).max(1.0);
            (variance / nf * (nf / df)).max(0.0).sqrt()
        };

        let n = nf.max(1.0);
        let overlap_report = OverlapReport {
            propensity_min: min_p,
            propensity_max: max_p,
            ess: Some(WeightSums::ess(applied.sum, applied.sum_sq)),
            extreme_weight_count: applied.extreme_count(),
            excluded_fraction: 0.0,
            target_population_support: in_support as f64 / n,
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
        Ok(StreamedPropensityEstimate {
            ate,
            se_analytic,
            n_obs: used,
            coefficients: beta,
            iterations,
            overlap_report,
            treatment_levels: DeclaredTreatmentLevels {
                active,
                control,
                other_levels: OtherLevels::Refuse,
                rows_at_active: at_active,
                rows_at_control: at_control,
                excluded_rows: 0,
            },
            reserved_bytes,
            // The engine's passes, the level count, and the three accumulating passes.
            source_passes: source_passes + 4,
            batches,
            choice,
        })
    }
}

/// The linearized Hájek contribution of one row, before the propensity-estimation term.
fn linearized(
    weight: f64,
    treated: bool,
    outcome: f64,
    sums: (f64, f64),
    means: (f64, f64),
    nf: f64,
) -> f64 {
    let (w1, w0) = if treated { (weight, 0.0) } else { (0.0, weight) };
    nf * (w1 / sums.0) * (outcome - means.0) - nf * (w0 / sums.1) * (outcome - means.1)
}

#[cfg(test)]
mod tests {
    use antecedent_core::{AssumptionSet, AverageEffectQuery, ExecutionContext, VariableId};
    use antecedent_expr::{ExprId, IdentifiedEstimand};

    use super::*;
    use crate::propensity::PropensityEstimationWorkspace;

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
    fn the_streamed_reduction_equals_the_resident_fit_standard_error_and_overlap_report() {
        let (data, estimand, query) = fixture(4_000, false);
        let estimator = PropensityWeighting::new().with_bootstrap_replicates(0);
        assert!(estimator.supports_streamed_reduction());
        let problem = estimator.prepare(&data, &estimand, &query).unwrap();
        let ctx = ExecutionContext::for_tests(1);
        let mut workspace = PropensityEstimationWorkspace::default();
        let resident = estimator.fit(&problem, &mut workspace, &ctx, AssumptionSet::new()).unwrap();
        // A batch far smaller than the row count: many batches in every engine pass.
        let streamed = estimator.estimate_streamed(&data, &estimand, &query, 97, 1 << 24).unwrap();
        assert!(streamed.batches > 1, "the rows were read in several batches");
        assert!(streamed.iterations >= 2, "a logistic fit takes more than one step");
        assert_eq!(Some(streamed.n_obs), resident.provenance.n_obs, "rows with a weight");
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
        let want = resident.diagnostics.overlap_report.as_ref().expect("resident report");
        let got = &streamed.overlap_report;
        assert!(close(got.propensity_min, want.propensity_min, 1e-7));
        assert!(close(got.propensity_max, want.propensity_max, 1e-7));
        assert!(close(got.ess.unwrap(), want.ess.unwrap(), 1e-7));
        assert_eq!(got.extreme_weight_count, want.extreme_weight_count);
        assert!(close(got.target_population_support, want.target_population_support, 1e-12));
        assert_eq!((got.clip, got.trim), (want.clip, want.trim));
        let (got_sens, want_sens) =
            (got.clip_sensitivity.as_ref().unwrap(), want.clip_sensitivity.as_ref().unwrap());
        assert_eq!(got_sens.thresholds, want_sens.thresholds);
        assert_eq!(got_sens.extreme_weight_counts, want_sens.extreme_weight_counts);
        for (a, b) in got_sens.ess.iter().zip(want_sens.ess.iter()) {
            assert!(close(*a, *b, 1e-7));
        }
        for (a, b) in got_sens.treated_ess.iter().zip(want_sens.treated_ess.iter()) {
            assert!(close(*a, *b, 1e-7));
        }
        for (a, b) in got_sens.control_ess.iter().zip(want_sens.control_ess.iter()) {
            assert!(close(*a, *b, 1e-7));
        }
        let levels = resident.diagnostics.treatment_levels.expect("resident levels");
        assert_eq!(streamed.treatment_levels, levels);
    }

    #[test]
    fn a_reduction_the_closed_form_does_not_cover_is_refused() {
        let (data, estimand, query) = fixture(600, false);
        let estimator = PropensityWeighting::new().with_bootstrap_replicates(0);
        // A ceiling below the window refuses instead of reading.
        assert!(estimator.estimate_streamed(&data, &estimand, &query, 64, 8).is_err());
        // A bootstrap, a trim, or sampling weights have no closed form here.
        let bootstrapped = PropensityWeighting::new();
        assert!(!bootstrapped.supports_streamed_reduction());
        assert!(bootstrapped.estimate_streamed(&data, &estimand, &query, 64, 1 << 20).is_err());
        let trimmed = PropensityWeighting::new()
            .with_bootstrap_replicates(0)
            .with_overlap(OverlapPolicy::RequireDiagnostics { clip: Some(0.01), trim: Some(0.05) });
        assert!(!trimmed.supports_streamed_reduction());
        // A treatment row at neither declared level is not silently dropped.
        let (stray, estimand, query) = fixture(600, true);
        let refused = estimator.estimate_streamed(&stray, &estimand, &query, 64, 1 << 20);
        assert!(refused.is_err(), "a row at level 2 is refused");
    }
}
