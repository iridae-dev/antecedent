//! Query-native refuters for a mediation decomposition evaluated by the discrete plug-in.
//!
//! The total, natural direct and natural indirect effects over a mixed graph, a graph class
//! or the unrolled window of a temporal graph are three functionals of one table, identified
//! by the path-specific identifier and evaluated by the finite-discrete plug-in. The ATE
//! refuters do not apply to them, and neither do the linear-model mediation refuters: the
//! plug-in keeps the support of every column, so a perturbation must too. The suite refits
//! every identified component together on each perturbed table:
//!
//! * `mediation.sequential_ignorability`: the sensitivity of the indirect effect to a
//!   violation of sequential ignorability (Imai, Keele and Yamamoto 2010), read from the
//!   linear working model of the treatment, the mediator index and the outcome with the
//!   variables the total-effect functional adjusts for. The error correlation at which the
//!   indirect effect vanishes must lie beyond the probed range.
//! * `mediation.placebo_mediator_*`: the mediator columns are permuted together (rotated by
//!   whole windows on a window table) so they keep their support and their joint law and lose
//!   every link to the treatment and the outcome. The indirect effect must vanish, and the
//!   total and direct effects must equal the total effect of the unpermuted table, which does
//!   not read the mediator.
//! * `mediation.subset_*` and `mediation.joint_stability` (and the 50% variants under full
//!   validation): all components refitted on the same random row subsets (contiguous stretches
//!   of windows on a window table). Each subsample refit is centred on the full-sample plug-in
//!   by construction, so this is a pipeline stability check, not evidence for the effect.
//! * `mediation.joint_bootstrap_range` (full): the components evaluated on the same bootstrap
//!   tables under the study's resampling design; every full-sample value must lie in the 95%
//!   percentile range of its replicates and few replicates may be lost.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::{TableView, TabularData};
use antecedent_estimate::util::SamplingDeclaration;
use antecedent_estimate::{FunctionalDistributionWorkspace, FunctionalEffect, RowResampling};
use antecedent_expr::IdentifiedEstimand;
use antecedent_identify::IdentificationResult;
use antecedent_stats::{DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace};

use crate::ValidationError;
use crate::common::{
    RefutationReport, STABILITY_ALPHA, STABILITY_REPLICATES, finish_p_report, map_replicates,
    with_contiguous_row_window, with_replaced_float, with_row_subset,
};
use crate::mediation_sensitivity::mediation_sensitivity_fit;

/// Names of the components, in the order the suite reads them.
pub const COMPONENT_NAMES: [&str; 3] = ["total", "direct", "indirect"];

/// Mediator-outcome error correlation the indirect effect must survive.
const SENSITIVITY_RHO: f64 = 0.2;
/// Smallest share of the replicates that must evaluate every component.
const MIN_COMPLETE_SHARE: f64 = 0.9;
/// Bootstrap replicates when the study requested none.
const DEFAULT_BOOTSTRAP_REPLICATES: u32 = 49;
/// Upper bound on the bootstrap replicates.
const MAX_BOOTSTRAP_REPLICATES: u32 = 199;

/// Subset streams of the 80% and 50% refits, mediator permutations.
const SUBSET_80: u64 = 0xF040_0000;
const SUBSET_50: u64 = 0xF041_0000;
const PLACEBO: u64 = 0xF042_0000;

/// One identified component of the decomposition.
#[derive(Clone, Copy)]
pub struct MediationComponent<'a> {
    /// The proof the component's functional came from.
    pub identification: &'a IdentificationResult,
    /// The functional the plug-in evaluates.
    pub estimand: &'a IdentifiedEstimand,
    /// The value the study published for the component.
    pub published: f64,
}

/// Everything one refutation of a mediation decomposition reads.
#[derive(Clone, Copy)]
pub struct MediationSuite<'a> {
    /// The table the plug-in was evaluated on.
    pub data: &'a TabularData,
    /// Total, natural direct and natural indirect component; `None` where the identifier
    /// refused it.
    pub components: [Option<MediationComponent<'a>>; 3],
    /// Slot of the contrast the study asked for.
    pub requested: usize,
    /// The columns the plug-in reads (the roles passed to the functional estimator).
    pub roles: &'a [VariableId],
    /// Treatment column.
    pub treatment: VariableId,
    /// Outcome column.
    pub outcome: VariableId,
    /// Mediator columns (every window coordinate of a mediator on a window table).
    pub mediators: &'a [VariableId],
    /// Active treatment level minus control level.
    pub contrast_width: f64,
    /// Series times one row of a window table spans; `None` for independent rows.
    pub window_span: Option<u32>,
    /// How the rows are resampled by the study's own uncertainty procedure.
    pub resampling: RowResampling,
    /// The sampling design the study declared.
    pub design: SamplingDeclaration,
    /// The study's bootstrap replicate count (zero: the default budget).
    pub bootstrap_replicates: u32,
}

/// Run the mediation suite: the cheap checks, plus the 50% refits and the joint bootstrap
/// range under `full`.
///
/// # Errors
///
/// A cancelled run, a table the perturbations cannot rebuild, or a spread that is not
/// finite.
pub fn refute_mediation_components(
    suite: &MediationSuite<'_>,
    full: bool,
    ctx: &ExecutionContext,
) -> Result<Vec<RefutationReport>, ValidationError> {
    let plug_in = suite.refit(suite.data, ctx)?;
    let mut reports = vec![suite.sequential_ignorability()];
    reports.extend(suite.placebo(&plug_in, ctx)?);
    reports.extend(suite.stability(&plug_in, 0.8, ("subset", SUBSET_80), ctx)?);
    if full {
        reports.extend(suite.stability(&plug_in, 0.5, ("half_sample", SUBSET_50), ctx)?);
        reports.push(suite.bootstrap_range(&plug_in, ctx)?);
    }
    Ok(reports)
}

impl<'a> MediationSuite<'a> {
    fn present(&self) -> impl Iterator<Item = (usize, &MediationComponent<'a>)> + '_ {
        self.components.iter().enumerate().filter_map(|(slot, c)| c.as_ref().map(|c| (slot, c)))
    }

    fn published_requested(&self) -> f64 {
        self.components[self.requested].as_ref().map_or(f64::NAN, |c| c.published)
    }

    /// Every component's plug-in on `data`; `NaN` where the component is absent or the
    /// table leaves a required empirical cell empty.
    fn refit(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<[f64; 3], ValidationError> {
        let estimator = FunctionalEffect::new();
        let mut values = [f64::NAN; 3];
        for (slot, component) in self.present() {
            let estimate = estimator
                .prepare(
                    data,
                    component.estimand,
                    &component.identification.arena,
                    component.identification.required_assumptions.clone(),
                    self.roles,
                )
                .and_then(|prepared| {
                    estimator.estimate(
                        &prepared,
                        &mut FunctionalDistributionWorkspace::default(),
                        ctx,
                    )
                });
            if let Ok(estimate) = estimate {
                values[slot] = estimate.ate;
            }
        }
        if ctx.cancellation.is_cancelled() {
            return Err(ValidationError::Cancelled);
        }
        Ok(values)
    }

    fn subset(
        &self,
        fraction: f64,
        ctx: &ExecutionContext,
        stream: u64,
    ) -> Result<TabularData, ValidationError> {
        if self.window_span.is_some() {
            with_contiguous_row_window(self.data, fraction, ctx, stream)
        } else {
            with_row_subset(self.data, fraction, ctx, stream)
        }
    }

    /// Variables the total-effect functional (else the first identified component) reads
    /// beyond the treatment, the mediators and the outcome: the covariates of the working
    /// model of the sensitivity analysis.
    fn adjustment_columns(&self) -> Vec<VariableId> {
        let estimator = FunctionalEffect::new();
        for (_, component) in self.present() {
            let Ok(prepared) = estimator.prepare(
                self.data,
                component.estimand,
                &component.identification.arena,
                component.identification.required_assumptions.clone(),
                self.roles,
            ) else {
                continue;
            };
            return prepared
                .variables_read()
                .into_iter()
                .filter(|v| {
                    *v != self.treatment && *v != self.outcome && !self.mediators.contains(v)
                })
                .collect();
        }
        Vec::new()
    }

    fn sequential_ignorability(&self) -> RefutationReport {
        let published = self.components[2].as_ref().map_or(f64::NAN, |c| c.published);
        let covariates = self.adjustment_columns();
        let columns = |ids: &[VariableId]| -> Result<Vec<Vec<f64>>, ValidationError> {
            ids.iter()
                .map(|&id| self.data.float64_values(id).map_err(ValidationError::from))
                .collect()
        };
        let read = (|| -> Result<_, ValidationError> {
            Ok((
                self.data.float64_values(self.treatment)?,
                columns(self.mediators)?,
                self.data.float64_values(self.outcome)?,
                columns(&covariates)?,
            ))
        })();
        match read.and_then(|(t, m, y, x)| {
            sequential_ignorability_report(&SensitivityInput {
                treatment: &t,
                mediators: &m,
                outcome: &y,
                covariates: &x,
                contrast_width: self.contrast_width,
                published,
            })
        }) {
            Ok(report) => report,
            Err(error) => RefutationReport::builder("mediation.sequential_ignorability")
                .original_ate(published)
                .refuted_ate(f64::NAN)
                .comparison(f64::NAN)
                .informative(false)
                .passed(false)
                .failure_condition(Some(Arc::from(format!(
                    "the linear working model of the sensitivity analysis could not be fitted: \
                     {error}"
                ))))
                .replicates(0)
                .build(),
        }
    }

    /// The table with the mediator columns permuted together (rotated by at least a whole
    /// window on a window table).
    fn permuted_mediators(
        &self,
        ctx: &ExecutionContext,
        replicate: u32,
    ) -> Result<TabularData, ValidationError> {
        let rows = self.data.row_count();
        let mut rng = ctx.rng.stream(PLACEBO + u64::from(replicate));
        let order: Vec<usize> = if let Some(span) = self.window_span {
            let span = span as usize;
            let rotations = (rows + 1).saturating_sub(2 * span);
            if rotations == 0 {
                return Err(ValidationError::NotApplicable {
                    message: "the series is too short to rotate the mediator by a full window",
                });
            }
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "next_f64 is in [0, 1), so the floored product is in [0, rotations) and is bounded to the last rotation"
            )]
            let offset = ((rng.next_f64() * rotations as f64) as usize).min(rotations - 1);
            let shift = span + offset;
            (0..rows).map(|row| (row + shift) % rows).collect()
        } else {
            let mut order: Vec<usize> = (0..rows).collect();
            for position in (1..rows).rev() {
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "next_f64 is in [0, 1), so the floored product is at most the position"
                )]
                let other = ((rng.next_f64() * (position + 1) as f64) as usize).min(position);
                order.swap(position, other);
            }
            order
        };
        let mut table = self.data.clone();
        for &mediator in self.mediators {
            let values = self.data.float64_values(mediator)?;
            let moved: Vec<f64> = order.iter().map(|&row| values[row]).collect();
            table = with_replaced_float(&table, mediator, Arc::from(moved))?;
        }
        Ok(table)
    }

    fn placebo(
        &self,
        plug_in: &[f64; 3],
        ctx: &ExecutionContext,
    ) -> Result<Vec<RefutationReport>, ValidationError> {
        let draws = map_replicates(
            ctx,
            STABILITY_REPLICATES,
            || (),
            |(), ctx, replicate| {
                let permuted = self.permuted_mediators(ctx, replicate)?;
                self.refit(&permuted, ctx)
            },
        )?;
        // The permuted mediator is independent of everything: the indirect effect is zero and
        // the direct effect is the whole total effect, which does not read the mediator.
        let targets = [plug_in[0], plug_in[0], 0.0];
        let mut reports = Vec::new();
        for (slot, component) in self.present() {
            let id = format!("mediation.placebo_mediator_{}", COMPONENT_NAMES[slot]);
            let values: Vec<f64> =
                draws.iter().map(|draw| draw[slot]).filter(|v| v.is_finite()).collect();
            if !targets[slot].is_finite() {
                continue;
            }
            reports.push(replicate_report(
                &id,
                component.published,
                &values,
                targets[slot],
                draws.len(),
                |mean, p_value| {
                    format!(
                        "with the mediator permuted the {} effect is {mean:.4} where {:.4} is \
                         expected (p = {p_value:.4})",
                        COMPONENT_NAMES[slot], targets[slot]
                    )
                },
            )?);
        }
        Ok(reports)
    }

    fn stability(
        &self,
        plug_in: &[f64; 3],
        fraction: f64,
        (label, base): (&str, u64),
        ctx: &ExecutionContext,
    ) -> Result<Vec<RefutationReport>, ValidationError> {
        let draws = map_replicates(
            ctx,
            STABILITY_REPLICATES,
            || (),
            |(), ctx, replicate| {
                let subset = self.subset(fraction, ctx, base + u64::from(replicate))?;
                self.refit(&subset, ctx)
            },
        )?;
        let mut reports = Vec::new();
        let mut all_passed = true;
        for (slot, component) in self.present() {
            let values: Vec<f64> =
                draws.iter().map(|draw| draw[slot]).filter(|v| v.is_finite()).collect();
            let report = replicate_report(
                &format!("mediation.{label}_{}", COMPONENT_NAMES[slot]),
                component.published,
                &values,
                plug_in[slot],
                draws.len(),
                |_, _| {
                    format!(
                        "the {} effect refitted on {:.0}% of the table is unstable",
                        COMPONENT_NAMES[slot],
                        fraction * 100.0
                    )
                },
            )?;
            all_passed &= report.passed;
            reports.push(report);
        }
        let complete = draws
            .iter()
            .filter(|draw| self.present().all(|(slot, _)| draw[slot].is_finite()))
            .count();
        let share = complete as f64 / draws.len().max(1) as f64;
        let passed = all_passed && share >= MIN_COMPLETE_SHARE;
        let requested_mean = mean_of(
            &draws
                .iter()
                .map(|draw| draw[self.requested])
                .filter(|v| v.is_finite())
                .collect::<Vec<_>>(),
        );
        reports.push(
            RefutationReport::builder(format!("mediation.joint_stability_{label}"))
                .original_ate(self.published_requested())
                .refuted_ate(requested_mean)
                .comparison(share)
                .informative(false)
                .passed(passed)
                .failure_condition((!passed).then(|| {
                    Arc::from(format!(
                        "the components do not stay stable together on {:.0}% of the table: \
                         {complete} of {} refits evaluated every component and \
                         {} component check(s) failed",
                        fraction * 100.0,
                        draws.len(),
                        reports.iter().filter(|r| !r.passed).count()
                    ))
                }))
                .replicates(u32::try_from(complete).unwrap_or(u32::MAX))
                .build(),
        );
        Ok(reports)
    }

    fn bootstrap_range(
        &self,
        plug_in: &[f64; 3],
        ctx: &ExecutionContext,
    ) -> Result<RefutationReport, ValidationError> {
        let replicates = if self.bootstrap_replicates == 0 {
            DEFAULT_BOOTSTRAP_REPLICATES
        } else {
            self.bootstrap_replicates.min(MAX_BOOTSTRAP_REPLICATES)
        };
        let estimator = FunctionalEffect::new()
            .with_bootstrap_replicates(replicates)
            .with_row_resampling(self.resampling, self.design);
        let mut slots = Vec::new();
        let mut prepared = Vec::new();
        for (slot, component) in self.present() {
            prepared.push(estimator.prepare(
                self.data,
                component.estimand,
                &component.identification.arena,
                component.identification.required_assumptions.clone(),
                self.roles,
            )?);
            slots.push(slot);
        }
        let refs: Vec<&antecedent_estimate::PreparedFunctionalEffect> = prepared.iter().collect();
        estimator.assert_same_rows(&refs)?;
        let rows = estimator.shared_replicate_draws(&refs, ctx)?;
        let kept = rows.len();
        let share = kept as f64 / f64::from(replicates);
        let mut outside = Vec::new();
        for (column, &slot) in slots.iter().enumerate() {
            let mut values: Vec<f64> = rows.iter().map(|row| row[column]).collect();
            values.sort_by(f64::total_cmp);
            let (low, high) = (quantile(&values, 0.025), quantile(&values, 0.975));
            let value = plug_in[slot];
            if !(low <= value && value <= high) {
                outside.push(format!(
                    "{} {value:.4} outside [{low:.4}, {high:.4}]",
                    COMPONENT_NAMES[slot]
                ));
            }
        }
        let requested_column = slots.iter().position(|&slot| slot == self.requested);
        let requested_mean = requested_column.map_or(f64::NAN, |column| {
            mean_of(&rows.iter().map(|row| row[column]).collect::<Vec<_>>())
        });
        let passed = kept >= 2 && outside.is_empty() && share >= MIN_COMPLETE_SHARE;
        Ok(RefutationReport::builder("mediation.joint_bootstrap_range")
            .original_ate(self.published_requested())
            .refuted_ate(requested_mean)
            .comparison(share)
            .informative(false)
            .passed(passed)
            .failure_condition((!passed).then(|| {
                Arc::from(format!(
                    "{kept} of {replicates} joint bootstrap tables evaluated every component; \
                     {}",
                    if outside.is_empty() {
                        "too few replicates survive to read a range".to_owned()
                    } else {
                        outside.join("; ")
                    }
                ))
            }))
            .replicates(u32::try_from(kept).unwrap_or(u32::MAX))
            .build())
    }
}

/// A replicate report of `values` against `centre`; a refit that lost too many replicates is
/// a failed report rather than an error.
fn replicate_report(
    id: &str,
    published: f64,
    values: &[f64],
    centre: f64,
    attempted: usize,
    failure: impl FnOnce(f64, f64) -> String,
) -> Result<RefutationReport, ValidationError> {
    if values.len() < 2
        || (values.len() as f64) < MIN_COMPLETE_SHARE * attempted as f64
        || !centre.is_finite()
    {
        return Ok(RefutationReport::builder(id)
            .original_ate(published)
            .refuted_ate(mean_of(values))
            .comparison(values.len() as f64 / attempted.max(1) as f64)
            .informative(false)
            .passed(false)
            .failure_condition(Some(Arc::from(format!(
                "only {} of {attempted} refits left every empirical cell the functional reads \
                 evaluable, or the full-sample value ({centre}) is not finite",
                values.len()
            ))))
            .replicates(u32::try_from(values.len()).unwrap_or(u32::MAX))
            .build());
    }
    finish_p_report(id, published, values, centre, STABILITY_ALPHA, failure)
}

fn mean_of(values: &[f64]) -> f64 {
    if values.is_empty() { f64::NAN } else { values.iter().sum::<f64>() / values.len() as f64 }
}

/// Linear-interpolated quantile `q` of `sorted`.
fn quantile(sorted: &[f64], q: f64) -> f64 {
    match sorted.len() {
        0 => f64::NAN,
        1 => sorted[0],
        n => {
            let position = q.clamp(0.0, 1.0) * (n - 1) as f64;
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "position is in [0, n - 1]"
            )]
            let lower = position.floor() as usize;
            let upper = (lower + 1).min(n - 1);
            sorted[lower] + (position - lower as f64) * (sorted[upper] - sorted[lower])
        }
    }
}

/// Columns the sensitivity analysis reads.
pub struct SensitivityInput<'a> {
    /// Treatment values.
    pub treatment: &'a [f64],
    /// One column per mediator.
    pub mediators: &'a [Vec<f64>],
    /// Outcome values.
    pub outcome: &'a [f64],
    /// One column per covariate.
    pub covariates: &'a [Vec<f64>],
    /// Active treatment level minus control level.
    pub contrast_width: f64,
    /// The published indirect effect (`NaN` when the identifier refused it).
    pub published: f64,
}

/// Sensitivity of the indirect effect to a mediator-outcome error correlation, as a report.
///
/// With several mediators the working model collapses them to the index the outcome
/// regression weights them by, so the indirect effect through the mediator set is the one
/// the single-mediator analysis is about.
///
/// # Errors
///
/// A working model that cannot be fitted: mismatched lengths, too few rows, a rank-deficient
/// design, or a residual with no variance.
pub fn sequential_ignorability_report(
    input: &SensitivityInput<'_>,
) -> Result<RefutationReport, ValidationError> {
    let n = input.outcome.len();
    if input.mediators.is_empty() {
        return Err(ValidationError::NotApplicable {
            message: "the sensitivity analysis needs a mediator column",
        });
    }
    let all_columns = || {
        std::iter::once(input.treatment)
            .chain(std::iter::once(input.outcome))
            .chain(input.mediators.iter().map(Vec::as_slice))
            .chain(input.covariates.iter().map(Vec::as_slice))
    };
    if all_columns().any(|column| column.len() != n) {
        return Err(ValidationError::NotApplicable {
            message: "treatment, mediators, outcome and covariates must have the same length",
        });
    }
    // Complete cases: a row with a missing value in any column is dropped from the working
    // model as it is from the plug-in.
    let keep: Vec<usize> =
        (0..n).filter(|&row| all_columns().all(|column| column[row].is_finite())).collect();
    let select = |column: &[f64]| -> Vec<f64> { keep.iter().map(|&row| column[row]).collect() };
    let (treatment, outcome) = (select(input.treatment), select(input.outcome));
    let mediators: Vec<Vec<f64>> = input.mediators.iter().map(|c| select(c)).collect();
    let covariates: Vec<Vec<f64>> = input.covariates.iter().map(|c| select(c)).collect();
    let rows = keep.len();
    let index = if mediators.len() == 1 {
        mediators[0].clone()
    } else {
        let p = 2 + mediators.len() + covariates.len();
        if rows <= p + 1 {
            return Err(ValidationError::NotApplicable {
                message: "the sensitivity analysis needs more rows than regression parameters",
            });
        }
        let mut design = vec![1.0; rows];
        design.extend_from_slice(&treatment);
        for column in mediators.iter().chain(&covariates) {
            design.extend_from_slice(column);
        }
        let fit = FaerBackend.least_squares(
            &design,
            rows,
            p,
            &outcome,
            &mut LeastSquaresWorkspace::default(),
        )?;
        let mut index = vec![0.0; rows];
        for (j, column) in mediators.iter().enumerate() {
            let weight = fit.coefficients[2 + j];
            for (slot, value) in index.iter_mut().zip(column) {
                *slot += weight * value;
            }
        }
        index
    };
    let summary = mediation_sensitivity_fit(&treatment, &index, &outcome, &covariates)?;
    let width = input.contrast_width;
    let at = |rho: f64| summary.acme(rho).map(|acme| acme * width);
    let (centre, low, high) = (at(0.0)?, at(-SENSITIVITY_RHO)?, at(SENSITIVITY_RHO)?);
    let sign = centre.signum();
    let survives = centre.abs() > 0.0 && sign * low > 0.0 && sign * high > 0.0;
    let worst = if sign * low < sign * high { low } else { high };
    let published = if input.published.is_finite() { input.published } else { centre };
    Ok(RefutationReport::builder("mediation.sequential_ignorability")
        .original_ate(published)
        .refuted_ate(worst)
        .comparison(summary.rho_tilde)
        .informative(true)
        .passed(survives)
        .failure_condition((!survives).then(|| {
            Arc::from(format!(
                "the indirect effect of the working linear model is {centre:.4} under \
                 sequential ignorability and vanishes at a mediator-outcome error correlation \
                 of {:.3}, inside the probed range of +-{SENSITIVITY_RHO}",
                summary.rho_tilde
            ))
        }))
        .replicates(1)
        .build())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linear_data(effect_of_mediator: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let (mut t, mut m, mut y, mut x) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for i in 0..200 {
            let a = f64::from(i);
            let xi = (a * 0.3).sin();
            let ti = f64::from(i % 2);
            let mi = 0.9 * ti + 0.4 * xi + 0.5 * (a * 1.7).sin();
            let yi = 0.2 * ti + effect_of_mediator * mi - 0.6 * xi + 0.5 * (a * 0.9).cos();
            t.push(ti);
            m.push(mi);
            y.push(yi);
            x.push(xi);
        }
        (t, m, y, x)
    }

    #[test]
    fn a_strong_mediated_effect_survives_the_probed_error_correlation() {
        let (t, m, y, x) = linear_data(1.5);
        let report = sequential_ignorability_report(&SensitivityInput {
            treatment: &t,
            mediators: &[m.clone()],
            outcome: &y,
            covariates: &[x.clone()],
            contrast_width: 1.0,
            published: 1.35,
        })
        .unwrap();
        assert!(report.passed, "{:?}", report.failure_condition);
        let summary = mediation_sensitivity_fit(&t, &m, &y, &[x]).unwrap();
        assert!((report.comparison - summary.rho_tilde).abs() < 1e-12);
        // The reported worst case is the curve at one end of the probed range.
        let ends =
            [summary.acme(-SENSITIVITY_RHO).unwrap(), summary.acme(SENSITIVITY_RHO).unwrap()];
        assert!(ends.iter().any(|end| (end - report.refuted_ate).abs() < 1e-12));
    }

    #[test]
    fn a_vanishing_mediated_effect_is_flagged_and_scales_with_the_contrast_width() {
        let (t, m, y, x) = linear_data(0.0);
        let narrow = sequential_ignorability_report(&SensitivityInput {
            treatment: &t,
            mediators: &[m.clone()],
            outcome: &y,
            covariates: &[x.clone()],
            contrast_width: 1.0,
            published: f64::NAN,
        })
        .unwrap();
        assert!(!narrow.passed);
        assert!(narrow.failure_condition.is_some());
        let wide = sequential_ignorability_report(&SensitivityInput {
            treatment: &t,
            mediators: &[m],
            outcome: &y,
            covariates: &[x],
            contrast_width: 2.0,
            published: f64::NAN,
        })
        .unwrap();
        assert!((wide.refuted_ate - 2.0 * narrow.refuted_ate).abs() < 1e-12);
        assert!((wide.comparison - narrow.comparison).abs() < 1e-12);
    }

    #[test]
    fn several_mediators_collapse_to_the_index_the_outcome_weights_them_by() {
        let (t, m, y, x) = linear_data(1.5);
        // A second mediator that is the first plus independent wiggle: the index weights
        // them by the outcome regression, so the indirect effect stays that of the mediator
        // set and the working model stays estimable.
        let second: Vec<f64> =
            m.iter().enumerate().map(|(i, v)| 0.5 * v + (i as f64 * 2.3).cos()).collect();
        let report = sequential_ignorability_report(&SensitivityInput {
            treatment: &t,
            mediators: &[m, second],
            outcome: &y,
            covariates: &[x],
            contrast_width: 1.0,
            published: 1.0,
        })
        .unwrap();
        assert!(report.refuted_ate.is_finite());
        assert!(report.comparison.abs() < 1.0);
    }

    #[test]
    fn missing_values_are_dropped_and_a_ragged_input_is_refused() {
        let (t, mut m, y, x) = linear_data(1.5);
        m[3] = f64::NAN;
        assert!(
            sequential_ignorability_report(&SensitivityInput {
                treatment: &t,
                mediators: &[m],
                outcome: &y,
                covariates: &[x.clone()],
                contrast_width: 1.0,
                published: 1.0,
            })
            .is_ok()
        );
        assert!(
            sequential_ignorability_report(&SensitivityInput {
                treatment: &t[1..],
                mediators: &[y.clone()],
                outcome: &y,
                covariates: &[x],
                contrast_width: 1.0,
                published: 1.0,
            })
            .is_err()
        );
    }

    #[test]
    fn quantiles_interpolate_between_order_statistics() {
        let sorted = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((quantile(&sorted, 0.0) - 1.0).abs() < 1e-12);
        assert!((quantile(&sorted, 0.5) - 3.0).abs() < 1e-12);
        assert!((quantile(&sorted, 0.125) - 1.5).abs() < 1e-12);
        assert!((quantile(&sorted, 1.0) - 5.0).abs() < 1e-12);
        assert!(quantile(&[], 0.5).is_nan());
    }

    #[test]
    fn a_report_over_too_few_replicates_fails_without_an_error() {
        let report =
            replicate_report("mediation.test", 0.5, &[0.1], 0.0, 20, |_, _| String::new()).unwrap();
        assert!(!report.passed);
        assert!(report.failure_condition.is_some());
        let enough: Vec<f64> = (0..20).map(|i| 0.01 * f64::from(i % 5)).collect();
        let report =
            replicate_report("mediation.test", 0.5, &enough, 0.02, 20, |_, _| String::new())
                .unwrap();
        assert!(report.passed);
    }
}
