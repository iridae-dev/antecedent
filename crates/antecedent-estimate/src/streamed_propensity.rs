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
use antecedent_io_core::capacity_account::CapacityOptions;

use crate::adjustment::EffectEstimate;
use crate::error::{EstimationError, ExactF64, RefusalFields};
use crate::overlap::{IpwTarget, OverlapPolicy, OverlapReport, clip_propensity};
use crate::prepare::treatment_contrast;
use crate::propensity::{DeclaredTreatmentLevels, OtherLevels, PropensityWeighting};
use crate::propensity::{clip_of, trim_of};
use crate::streamed_rows::{
    BOUNDARY, EngineFacts, RowView, WeightSums, clip_sensitivity_report,
    clip_sensitivity_thresholds, fit_logistic_streamed, fitted_propensity, for_each_kept_row,
};

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

/// The complete-case rows of a streamed reduction with the float columns they are read from,
/// the declared treatment levels and the clip threshold the weights are formed under.
struct StreamedInputs<'a> {
    row_mask: Vec<bool>,
    covariates: Vec<&'a [f64]>,
    treatment: &'a [f64],
    outcome: &'a [f64],
    active: f64,
    control: f64,
    /// Rows the mask keeps.
    rows: u64,
    at_active: u64,
    at_control: u64,
    clip: Option<f64>,
    /// A declared overlap clip or trim governs rows at the boundary, so they are admitted.
    admit_saturation: bool,
}

impl StreamedInputs<'_> {
    /// Visit every kept row with its design row, treatment arm and outcome.
    fn for_each_row(&self, mut visit: impl FnMut(RowView<'_>)) {
        for_each_kept_row(
            &self.row_mask,
            &self.covariates,
            self.treatment,
            self.outcome,
            self.active,
            |_, row| visit(row),
        );
    }

    /// The weight of a row at fitted propensity `raw`, under the clip.
    fn weight(&self, target: IpwTarget, treated: bool, raw: f64) -> f64 {
        let weight_score = self.clip.map_or(raw, |c| clip_propensity(raw, c));
        target.weight(if treated { 1.0 } else { 0.0 }, weight_score)
    }
}

/// The logistic IRLS fit: coefficients and what the engine passes cost.
struct LogisticPasses {
    beta: Vec<f64>,
    iterations: u32,
    source_passes: u64,
    reserved_bytes: u64,
    batches: u64,
    choice: PhysicalChoice,
}

/// The first accumulating pass: the weighted arm sums of the Hájek means, the overlap
/// accumulators and the clip sensitivity.
struct ArmPass {
    applied: WeightSums,
    sensitivity: Vec<WeightSums>,
    thresholds: Vec<f64>,
    /// Total weight of the treated and control arms.
    totals: (f64, f64),
    /// Hájek mean of the treated and control arms.
    means: (f64, f64),
    min_p: f64,
    max_p: f64,
    in_support: u64,
    /// Rows with a positive finite weight.
    used: u64,
}

impl PropensityWeighting {
    /// Whether this estimator's result follows in closed form from streamed row passes: the
    /// iid analytic standard error, untrimmed weights, no sampling weights, clustering,
    /// calibration, bootstrap, or rows at undeclared treatment levels.
    #[must_use]
    pub fn supports_streamed_reduction(&self) -> bool {
        self.inference.bootstrap_replicates == 0
            && self.inference.bootstrap_interval.is_none()
            && !self.covariate_balance
            && self.sampling_weights.is_none()
            && self.inference.cluster_ids.is_none()
            && self.inference.source_cluster_ids.is_none()
            && crate::propensity::require_supported_inference(&self.inference).is_ok()
            && self.other_levels == OtherLevels::Refuse
            && trim_of(self.overlap).is_none()
    }

    /// Estimate the average effect by streaming the rows in batches of `batch_rows`, holding
    /// at most `ceiling_bytes` per engine pass, without building the design.
    ///
    /// The reduction runs in stages: validate the request and read the columns, fit the
    /// propensity by streamed IRLS, accumulate the arm sums and overlap report, then the
    /// logistic information and the corrected influence of the standard error.
    ///
    /// # Errors
    ///
    /// The estimator is not streamable, the estimand or query is not an all-observed
    /// adjustment, a column is not float64, a treatment row sits at neither declared level,
    /// the logistic fit does not converge or its probabilities reach the boundary, an arm
    /// has no weight, or the reduction exceeds its ceiling.
    pub fn estimate_streamed(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
        batch_rows: usize,
        ceiling_bytes: u64,
    ) -> Result<StreamedPropensityEstimate, EstimationError> {
        let inputs = self.streamed_inputs(data, estimand, query)?;
        let target = IpwTarget::Ate;
        let options = CapacityOptions::new(batch_rows, ceiling_bytes, OutputRetention::SummaryOnly);
        let logistic = self.fit_propensity_streamed(&inputs, &options)?;
        let arm = arm_pass(&inputs, target, &logistic.beta)?;
        let se_analytic = influence_se(&inputs, target, &logistic.beta, &arm)?;
        let overlap_report = overlap_report_of(&inputs, &arm);
        let LogisticPasses { beta, iterations, source_passes, reserved_bytes, batches, choice } =
            logistic;
        Ok(StreamedPropensityEstimate {
            ate: arm.means.0 - arm.means.1,
            se_analytic,
            n_obs: arm.used,
            coefficients: beta,
            iterations,
            overlap_report,
            treatment_levels: DeclaredTreatmentLevels {
                active: inputs.active,
                control: inputs.control,
                other_levels: OtherLevels::Refuse,
                rows_at_active: inputs.at_active,
                rows_at_control: inputs.at_control,
                excluded_rows: 0,
            },
            reserved_bytes,
            // The engine's passes, the level count, and the three accumulating passes.
            source_passes: source_passes + 4,
            batches,
            choice,
        })
    }

    /// Validate the estimator and query, then read the float columns, the complete-case mask
    /// and the rows at each declared treatment level (none may sit elsewhere).
    #[allow(
        clippy::float_cmp,
        reason = "the declared treatment levels are exact category labels, not computed values"
    )]
    fn streamed_inputs<'a>(
        &self,
        data: &'a TabularData,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
    ) -> Result<StreamedInputs<'a>, EstimationError> {
        if !self.supports_streamed_reduction() {
            return Err(EstimationError::unsupported_coded(
                antecedent_core::reason_code!("route_not_supported"),
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
            return Err(EstimationError::unsupported_coded(
                antecedent_core::reason_code!("option_not_applicable"),
                "propensity estimators do not support effect modifiers",
            ));
        }
        if query.target_population != TargetPopulation::AllObserved {
            return Err(EstimationError::unsupported_coded(
                antecedent_core::reason_code!("route_not_supported"),
                "a streamed reduction estimates the all-observed population",
            ));
        }
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
        let column = |id| -> Result<&'a [f64], EstimationError> {
            match data.column(id)? {
                ColumnView::Float64(c) => Ok(c.values.as_slice()),
                _ => Err(EstimationError::unsupported_coded(
                    antecedent_core::reason_code!("route_not_supported"),
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
        Ok(StreamedInputs {
            row_mask,
            covariates,
            treatment,
            outcome,
            active,
            control,
            rows,
            at_active,
            at_control,
            clip,
            admit_saturation: admits_saturation(self.overlap),
        })
    }

    /// Logistic IRLS: one engine pass per iteration.
    fn fit_propensity_streamed(
        &self,
        inputs: &StreamedInputs<'_>,
        options: &CapacityOptions,
    ) -> Result<LogisticPasses, EstimationError> {
        let mut facts = EngineFacts::default();
        let (beta, iterations) = fit_logistic_streamed(
            &inputs.covariates,
            inputs.treatment,
            inputs.active,
            &inputs.row_mask,
            None,
            options,
            &self.glm_options,
            &mut facts,
        )?;
        let choice = facts.choice.ok_or_else(|| EstimationError::data_msg("no IRLS pass ran"))?;
        Ok(LogisticPasses {
            beta,
            iterations,
            source_passes: facts.passes,
            reserved_bytes: facts.reserved,
            batches: facts.batches,
            choice,
        })
    }
}

/// Whether `overlap` declares a clip or trim, which governs propensities at the boundary: an
/// undeclared default does not decide a positivity question.
pub(crate) fn admits_saturation(overlap: OverlapPolicy) -> bool {
    overlap.is_declared() && (clip_of(overlap).is_some() || trim_of(overlap).is_some())
}

/// The refusal of a converged fit whose propensities reach within `1e-8` of 0 or 1 when no
/// declared overlap clip or trim governs them: those rows have essentially no chance of the
/// other treatment, a positivity problem in the data.
pub(crate) fn boundary_saturated_refusal(route: &str, count: u64, margin: f64) -> EstimationError {
    let rows = format!("{count} row{}", if count == 1 { "" } else { "s" });
    EstimationError::refused_detail_with_fields(
        antecedent_core::reason_code!("positivity_violated"),
        format!("{route}.boundary_saturated"),
        format!(
            "the propensity model gives {rows} a probability of treatment within 1e-8 of 0 or 1, \
             so those rows have essentially no chance of the other treatment and an \
             inverse-probability weight is unsupported there. Declare an overlap trim or clip, \
             or restrict the population to where both treatments occur"
        ),
        RefusalFields {
            stage: Some("glm_fit".to_owned()),
            reason: Some("boundary_saturated".to_owned()),
            remedy: Some(
                "declare an overlap clip that bounds the weights (an overlap trim needs the \
                 resident estimator), restrict the population or covariates to where both \
                 treatments occur, or obtain data or a design that produces both treatments there"
                    .to_owned(),
            ),
            boundary_margin: margin.is_finite().then_some(ExactF64(margin)),
            boundary_count: Some(count),
            ..RefusalFields::default()
        },
    )
}

/// Pass A: arm sums of the Hájek means, the overlap accumulators, the boundary check.
fn arm_pass(
    inputs: &StreamedInputs<'_>,
    target: IpwTarget,
    beta: &[f64],
) -> Result<ArmPass, EstimationError> {
    let clip = inputs.clip;
    let thresholds = clip_sensitivity_thresholds(clip);
    let mut sensitivity = vec![WeightSums::default(); thresholds.len()];
    let mut applied = WeightSums::default();
    let (mut sw1, mut sw0, mut sy1, mut sy0) = (0.0, 0.0, 0.0, 0.0);
    let (mut min_p, mut max_p) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut in_support, mut boundary, mut used) = (0_u64, 0_u64, 0_u64);
    let mut margin = f64::INFINITY;
    let support_lo = clip.unwrap_or(0.0);
    let support_hi = 1.0 - support_lo;
    inputs.for_each_row(|row| {
        let raw = fitted_propensity(row.design, beta);
        min_p = min_p.min(raw);
        max_p = max_p.max(raw);
        if raw >= support_lo && raw <= support_hi {
            in_support += 1;
        }
        if raw <= BOUNDARY || raw >= 1.0 - BOUNDARY {
            boundary += 1;
            margin = margin.min(raw.min(1.0 - raw));
        }
        let t = if row.treated { 1.0 } else { 0.0 };
        let weight = inputs.weight(target, row.treated, raw);
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
    if boundary > 0 && !inputs.admit_saturation {
        return Err(boundary_saturated_refusal("propensity", boundary, margin));
    }
    if sw1 <= 0.0 || sw0 <= 0.0 {
        return Err(EstimationError::data_msg(
            "IPW weighting left an arm with zero total weight (trimming/clipping removed all treated or all control units)",
        ));
    }
    Ok(ArmPass {
        applied,
        sensitivity,
        thresholds,
        totals: (sw1, sw0),
        means: (sy1 / sw1, sy0 / sw0),
        min_p,
        max_p,
        in_support,
        used,
    })
}

/// Passes B and C: the analytic standard error with the logistic-estimation correction.
///
/// Pass B accumulates the logistic information and the derivative of the weighted means with
/// respect to the propensity coefficients; pass C the corrected influence, as a running mean and
/// sum of squares.
fn influence_se(
    inputs: &StreamedInputs<'_>,
    target: IpwTarget,
    beta: &[f64],
    arm: &ArmPass,
) -> Result<f64, EstimationError> {
    let ncols = beta.len();
    let clip = inputs.clip;
    let nf = inputs.rows as f64;
    let (totals, means) = (arm.totals, arm.means);
    let mut information = vec![0.0; ncols * ncols];
    let mut derivative = vec![0.0; ncols];
    inputs.for_each_row(|row| {
        let raw = fitted_propensity(row.design, beta);
        let weight = inputs.weight(target, row.treated, raw);
        let psi = linearized(weight, row.treated, row.outcome, totals, means, nf);
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
    let alpha = crate::util::solve_spd(&information, &derivative, ncols)
        .ok_or_else(|| EstimationError::stats_msg("singular logistic information in Hajek SE"))?;

    let (mut mean, mut m2, mut seen) = (0.0_f64, 0.0_f64, 0.0_f64);
    inputs.for_each_row(|row| {
        let raw = fitted_propensity(row.design, beta);
        let t = if row.treated { 1.0 } else { 0.0 };
        let weight = inputs.weight(target, row.treated, raw);
        let psi = linearized(weight, row.treated, row.outcome, totals, means, nf);
        let adjustment: f64 = row.design.iter().zip(&alpha).map(|(x, a)| x * a).sum();
        let corrected = psi + adjustment * (t - raw);
        seen += 1.0;
        let delta = corrected - mean;
        mean += delta / seen;
        m2 += delta * (corrected - mean);
    });
    Ok(if inputs.rows < 2 {
        f64::NAN
    } else {
        let variance = m2 / (nf - 1.0);
        let df = (nf - ncols.max(1) as f64).max(1.0);
        (variance / nf * (nf / df)).max(0.0).sqrt()
    })
}

/// The positivity diagnostics built from the running accumulators of pass A.
fn overlap_report_of(inputs: &StreamedInputs<'_>, arm: &ArmPass) -> OverlapReport {
    let n = (inputs.rows as f64).max(1.0);
    let ArmPass { applied, sensitivity, thresholds, min_p, max_p, in_support, .. } = arm;
    OverlapReport {
        propensity_min: *min_p,
        propensity_max: *max_p,
        ess: Some(WeightSums::ess(applied.sum, applied.sum_sq)),
        extreme_weight_count: applied.extreme_count(),
        excluded_fraction: 0.0,
        target_population_support: *in_support as f64 / n,
        excluded_regions: Arc::from([]),
        clip: inputs.clip,
        trim: None,
        retained_fraction: 1.0,
        clip_sensitivity: clip_sensitivity_report(inputs.clip, thresholds, sensitivity),
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

    /// 400 rows, `x` uniform on `[-3, 3]`, treated iff `x > 0` except eight flips next to 0:
    /// the logistic fit converges and is not separated, yet saturates far from the flips.
    fn saturated_fixture() -> (TabularData, IdentifiedEstimand, AverageEffectQuery) {
        let x: Vec<f64> = (0..400).map(|i| -3.0 + 6.0 * (i as f64 + 0.5) / 400.0).collect();
        let flips = [190, 193, 196, 198, 201, 203, 206, 209];
        let t: Vec<f64> = (0..400).map(|i| f64::from((x[i] > 0.0) != flips.contains(&i))).collect();
        let y: Vec<f64> =
            (0..400).map(|i| 1.0 + 2.0 * t[i] + 0.5 * x[i] + 0.4 * noise(i, 4)).collect();
        let data = TabularData::from_f64_columns([
            ("t", t.as_slice()),
            ("y", y.as_slice()),
            ("x", x.as_slice()),
        ])
        .unwrap();
        let estimand = IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from([VariableId::from_raw(2)]),
            ExprId::from_raw(0),
        );
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        (data, estimand, query)
    }

    fn assert_positivity_refusal(error: &EstimationError) {
        assert_eq!(error.reason_code(), Some(antecedent_core::reason_code!("positivity_violated")));
        let fields = error.refusal_fields().expect("structured fields");
        assert_eq!(fields.reason.as_deref(), Some("boundary_saturated"));
        assert!(fields.boundary_count.is_some_and(|count| count > 0));
        assert!(fields.remedy.as_deref().is_some_and(|remedy| remedy.contains("overlap")));
    }

    #[test]
    fn a_saturated_streamed_fit_refuses_as_positivity_unless_a_declared_clip_governs_it() {
        let (data, estimand, query) = saturated_fixture();
        let undeclared = PropensityWeighting::new().with_bootstrap_replicates(0);
        let error = undeclared
            .estimate_streamed(&data, &estimand, &query, 64, 1 << 20)
            .expect_err("an undeclared default refuses a saturated fit");
        assert_positivity_refusal(&error);
        let declared = PropensityWeighting::new()
            .with_bootstrap_replicates(0)
            .with_overlap(OverlapPolicy::RequireDiagnostics { clip: Some(0.01), trim: None });
        declared
            .estimate_streamed(&data, &estimand, &query, 64, 1 << 20)
            .expect("a declared clip governs the saturated rows");
    }
}
