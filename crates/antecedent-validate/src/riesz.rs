//! Riesz-representer diagnostics for binary ATE.
//!
//! For binary treatment the Riesz representer (Chernozhukov, Newey & Singh 2022) of the ATE functional under
//! unconfoundedness is
//!
//! ```text
//! α(T, Z) = T / π(Z) − (1 − T) / (1 − π(Z))
//! ```
//!
//! with propensity `π(Z) = P(T=1 | Z)`. The IPW ATE is `E[α Y]`. Sensitivity
//! bounds use the representer norm: an unobserved confounder that shifts the
//! outcome residual by at most `δ` in L2 can change the ATE by at most
//! `δ · ||α||_2 / √n`-scaled norms; we report the smallest `δ` on a grid that
//! can push the estimate through zero (or flip sign). The interval is centred on
//! the published estimate the report is attached to (`original_ate`), not on the
//! diagnostic fit's clipped IPW contrast, which supplies only `α`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        clippy::float_cmp,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

use std::sync::Arc;

use antecedent_core::ExecutionContext;
use antecedent_estimate::EstimationWorkspace;
use antecedent_stats::GlmOptions;

use crate::common::{RefutationProblem, RefutationReport, fit_diagnostic_propensity};
use crate::error::ValidationError;

/// Default confounding-strength grid (L2 residual shift, in units of `sd(Y)`).
fn default_delta_grid() -> Vec<f64> {
    vec![0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
}

/// Riesz-representer robustness diagnostics for binary ATE.
#[derive(Clone, Debug)]
pub struct RieszSensitivity {
    /// Ascending grid of residual confounding strengths `δ`, in units of `sd(Y)` so the
    /// verdict is invariant to outcome units.
    pub delta_grid: Vec<f64>,
    /// Pass only if a probed `δ` at or above this threshold left the effect standing, so a
    /// bar between two grid points is judged by the lower one. Equality of the robustness `δ`
    /// with the bar fails: a residual shift at the bar already kills the effect.
    pub pass_threshold: f64,
    /// Propensity clip for numerical stability.
    pub clip: f64,
    /// GLM options for the propensity fit.
    pub glm_options: GlmOptions,
}

impl Default for RieszSensitivity {
    fn default() -> Self {
        Self::new()
    }
}

impl RieszSensitivity {
    /// Defaults: delta grid through 1.0, pass threshold 0.1, clip 0.01.
    #[must_use]
    pub fn new() -> Self {
        Self {
            delta_grid: default_delta_grid(),
            pass_threshold: 0.1,
            clip: 0.01,
            glm_options: GlmOptions::default(),
        }
    }

    /// Run Riesz-representer sensitivity.
    ///
    /// # Errors
    ///
    /// Empty grid, data/GLM failures, or non-binary treatment.
    pub fn refute(
        &self,
        problem: &RefutationProblem<'_>,
        _workspace: &mut EstimationWorkspace,
        _ctx: &ExecutionContext,
    ) -> Result<RefutationReport, ValidationError> {
        let mut local = antecedent_stats::PropensityWorkspace::default();
        self.refute_with_propensity(problem, &mut local)
    }

    /// Like [`Self::refute`], reusing a warmed propensity workspace for the diagnostic fit.
    ///
    /// # Errors
    ///
    /// Empty grid, data/GLM failures, or non-binary treatment.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the refutation reports its grid size as u32, and a delta grid is far below 2^32 points"
    )]
    pub fn refute_with_propensity(
        &self,
        problem: &RefutationProblem<'_>,
        propensity: &mut antecedent_stats::PropensityWorkspace,
    ) -> Result<RefutationReport, ValidationError> {
        if self.delta_grid.is_empty() {
            return Err(ValidationError::NotApplicable {
                message: "Riesz sensitivity requires a non-empty delta_grid",
            });
        }
        let representer = self.representer_and_ipw(problem, propensity)?;
        if let Some(defect) = representer.defect {
            // Separation makes the representer unbounded: no finite δ certifies robustness.
            return Ok(RefutationReport {
                refuter: Arc::from("sensitivity.riesz"),
                original_ate: problem.original.ate,
                refuted_ate: problem.original.ate,
                comparison: 0.0,
                informative: true,
                passed: false,
                failure_condition: Some(Arc::from(defect)),
                replicates: 0,
            });
        }
        let Representer { alpha, y, ipw_ate, .. } = representer;
        // Centring. `|E[αΔ]| ≤ δ·sd(Y)·‖α‖₂` bounds the confounding bias of the ATE functional,
        // not of any one estimator of it, so the interval is centred on the published estimate:
        // the report carries that estimate as `original_ate`, and the tipping δ must be the
        // confounding needed to explain *it* away. The clipped IPW contrast is only the
        // diagnostic fit's own reading of the effect; it supplies the representer norm and a
        // consistency check, never the centre. An IPW contrast of the opposite sign means the
        // diagnostic propensity fit does not describe the published adjustment, so its norm
        // is not used.
        let centre = problem.original.ate;
        if !centre.is_finite() {
            return Err(ValidationError::NotApplicable {
                message: "Riesz sensitivity requires a finite published estimate",
            });
        }
        if ipw_ate != 0.0 && centre != 0.0 && ipw_ate.signum() != centre.signum() {
            return Err(ValidationError::NotApplicable {
                message: "the inverse-probability-weighted ATE has the opposite sign of the \
                          published estimate, so the diagnostic fit's representer does not describe it",
            });
        }
        let sd_y = crate::common::sample_sd(&y).max(1e-12);
        let n = alpha.len() as f64;
        let alpha_l2 = (alpha.iter().map(|a| a * a).sum::<f64>() / n.max(1.0)).sqrt();
        if alpha_l2 < 1e-15 {
            return Err(ValidationError::NotApplicable {
                message: "Riesz representer has near-zero L2 norm",
            });
        }

        let mut sorted = self.delta_grid.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let original_sign = centre.signum();
        let mut last_bound_ate = centre;
        let mut explained_away_at = None;
        for &delta in &sorted {
            // Worst-case shift: |bias| ≤ δ·sd(Y) · ||α||_2 (population L2 product bound;
            // δ expressed in sd(Y) units keeps the grid scale-free).
            let bias = delta * sd_y * alpha_l2;
            let lower = centre - bias;
            let upper = centre + bias;
            // "Explained away" once the interval [ate − bias, ate + bias] covers 0. The bias is
            // non-negative, so the interval always contains the published estimate and covering
            // zero is exactly the nearer endpoint reaching or crossing it.
            let covers_zero = lower <= 0.0 && upper >= 0.0;
            last_bound_ate = if original_sign >= 0.0 { lower } else { upper };
            if covers_zero {
                explained_away_at = Some(delta);
                break;
            }
            let _ = &y; // y retained for future DR extensions
        }
        // Smallest grid δ that tips the estimate; +∞ if the bound never covers zero.
        let robustness = explained_away_at.unwrap_or(f64::INFINITY);
        let passed = crate::sensitivity::robustness_passes(
            sorted.iter().copied(),
            robustness,
            self.pass_threshold,
        );
        Ok(RefutationReport {
            refuter: Arc::from("sensitivity.riesz"),
            original_ate: problem.original.ate,
            refuted_ate: last_bound_ate,
            comparison: robustness,
            informative: true,
            passed,
            failure_condition: if passed {
                None
            } else {
                Some(Arc::from(format!(
                    "{} (||α||₂={alpha_l2})",
                    crate::sensitivity::grid_failure_text(
                        "Riesz bound: effect",
                        "δ",
                        robustness,
                        self.pass_threshold,
                    )
                )))
            },
            replicates: self.delta_grid.len() as u32,
        })
    }

    fn representer_and_ipw(
        &self,
        problem: &RefutationProblem<'_>,
        propensity: &mut antecedent_stats::PropensityWorkspace,
    ) -> Result<Representer, ValidationError> {
        let cols = fit_diagnostic_propensity(problem, &self.glm_options, true, propensity)?;
        let defect = cols.defect;
        let y = cols.outcome.expect("outcome requested");
        let nrows = cols.treatment.len();
        #[allow(
            clippy::float_cmp,
            reason = "binary treatment is coded exactly 0/1, so exact equality is the membership test"
        )]
        for &ti in &cols.treatment {
            if !(ti == 0.0 || ti == 1.0) {
                return Err(ValidationError::NotApplicable {
                    message: "RieszSensitivity requires binary 0/1 treatment",
                });
            }
        }
        let lo = self.clip.clamp(1e-6, 0.49);
        let hi = 1.0 - lo;
        let mut alpha = Vec::with_capacity(nrows);
        let mut weighted = 0.0;
        for (score, (&ti, &yi)) in cols.scores.iter().zip(cols.treatment.iter().zip(y.iter())) {
            let p = score.clamp(lo, hi);
            let a = if ti >= 0.5 { 1.0 / p } else { -1.0 / (1.0 - p) };
            alpha.push(a);
            weighted += a * yi;
        }
        let ipw_ate = weighted / nrows as f64;
        Ok(Representer { alpha, y, ipw_ate, defect })
    }
}

/// Riesz representer values, outcome and IPW ATE of the diagnostic propensity fit.
struct Representer {
    alpha: Vec<f64>,
    y: Vec<f64>,
    ipw_ate: f64,
    /// Separation / non-convergence of the diagnostic fit (a positivity failure).
    defect: Option<&'static str>,
}

#[cfg(test)]
mod tests {
    use antecedent_core::{
        AssumptionSet, AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec,
        RoleHint, SmallRoleSet, ValueType, VariableId,
    };
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
    };
    use antecedent_estimate::{EstimationWorkspace, LinearAdjustmentAte};
    use antecedent_expr::ExprId;
    use antecedent_identify::IdentifiedEstimand;

    use super::*;
    use crate::common::RefutationProblem;

    fn toy() -> (TabularData, IdentifiedEstimand) {
        let n = 300usize;
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "t",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "z",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let t: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
        let z: Vec<f64> = (0..n).map(|i| (i as f64) / n as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * t[i] + 0.5 * z[i]).collect();
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(t),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(y),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(2),
                    Arc::from(z),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let estimand = IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from([VariableId::from_raw(2)]),
            ExprId::from_raw(0),
        );
        (TabularData::new(storage), estimand)
    }

    #[test]
    fn riesz_reports_positive_robustness() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../conformance/validate/riesz_sensitivity/expected.json"
        ))
        .unwrap();
        assert_eq!(fixture["balanced_case"]["alpha_l2"].as_f64().unwrap(), 2.0);
        let (data, estimand) = toy();
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let est = LinearAdjustmentAte { bootstrap_replicates: 0, ..LinearAdjustmentAte::new() };
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = EstimationWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let original = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        let problem = RefutationProblem::new(
            &data,
            &estimand,
            &query,
            &original,
            Some("linear.adjustment.ate"),
            None,
        );
        let report = RieszSensitivity::new().refute(&problem, &mut ws, &ctx).unwrap();
        assert_eq!(report.refuter.as_ref(), "sensitivity.riesz");
        assert!(report.comparison > 0.0, "comparison={}", report.comparison);
        assert!(
            report.comparison.is_infinite()
                || fixture["delta_grid"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|delta| delta.as_f64() == Some(report.comparison)),
            "comparison={} must be a grid δ or +∞ when never explained away",
            report.comparison
        );
        assert!(report.informative);
    }

    /// The worst-case bias `δ·sd(Y)·‖α‖₂` at the reported robustness must reach the magnitude
    /// of the estimate the report is attached to, the previous grid point's must not, and
    /// `refuted_ate` must be that estimate moved toward zero by the bias.
    fn assert_robustness_magnitude_matches(
        report: &RefutationReport,
        grid: &[f64],
        sd_y: f64,
        alpha_l2: f64,
    ) {
        let ate = report.original_ate;
        let unit = sd_y * alpha_l2;
        let tip = grid.iter().copied().find(|&d| d * unit >= ate.abs());
        match tip {
            Some(delta) => {
                assert_eq!(report.comparison, delta, "robustness for |ate|={}", ate.abs());
                let expected_bound = ate - ate.signum() * delta * unit;
                assert!(
                    (report.refuted_ate - expected_bound).abs() <= 1e-12 * unit.max(1.0),
                    "refuted_ate={} expected {expected_bound}",
                    report.refuted_ate
                );
            }
            None => assert!(report.comparison.is_infinite(), "comparison={}", report.comparison),
        }
        if let Some(prev) = grid.iter().copied().filter(|&d| d < report.comparison).last() {
            assert!(prev * unit < ate.abs(), "δ={prev} already explains away |ate|={}", ate.abs());
        }
    }

    #[test]
    fn riesz_robustness_is_centred_on_the_published_estimate() {
        let (data, estimand) = toy();
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let est = LinearAdjustmentAte { bootstrap_replicates: 0, ..LinearAdjustmentAte::new() };
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = EstimationWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let fitted = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        let riesz = RieszSensitivity::new();
        let grid = riesz.delta_grid.clone();
        // The clipped IPW contrast of the diagnostic fit is ≈ 2; publish smaller and larger
        // same-sign estimates so a bound centred on the IPW contrast would tip at the wrong δ.
        for published in [fitted.ate, 0.5, 0.3, 4.0] {
            let mut original = fitted.clone();
            original.ate = published;
            let problem = RefutationProblem::new(
                &data,
                &estimand,
                &query,
                &original,
                Some("linear.adjustment.ate"),
                None,
            );
            let mut prop = antecedent_stats::PropensityWorkspace::default();
            let rep = riesz.representer_and_ipw(&problem, &mut prop).unwrap();
            assert!((rep.ipw_ate - 2.0).abs() < 0.1, "ipw={}", rep.ipw_ate);
            let sd_y = crate::common::sample_sd(&rep.y);
            let n = rep.alpha.len() as f64;
            let alpha_l2 = (rep.alpha.iter().map(|a| a * a).sum::<f64>() / n).sqrt();
            let report = riesz.refute(&problem, &mut ws, &ctx).unwrap();
            assert_eq!(report.original_ate, published);
            assert_robustness_magnitude_matches(&report, &grid, sd_y, alpha_l2);
        }
    }

    #[test]
    fn riesz_equality_at_threshold_fails_strict_above_passes() {
        let (data, estimand) = toy();
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let est = LinearAdjustmentAte { bootstrap_replicates: 0, ..LinearAdjustmentAte::new() };
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = EstimationWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let original = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        let problem = RefutationProblem::new(
            &data,
            &estimand,
            &query,
            &original,
            Some("linear.adjustment.ate"),
            None,
        );
        let probe = RieszSensitivity { pass_threshold: 0.0, ..RieszSensitivity::new() };
        let tipped = probe.refute(&problem, &mut ws, &ctx).unwrap();
        assert!(
            tipped.comparison.is_finite() && tipped.comparison > 0.0,
            "expected a finite tipping δ, got {}",
            tipped.comparison
        );
        let rv = tipped.comparison;

        let at_bar = RieszSensitivity {
            pass_threshold: rv,
            delta_grid: probe.delta_grid.clone(),
            ..RieszSensitivity::new()
        };
        let equal = at_bar.refute(&problem, &mut ws, &ctx).unwrap();
        assert_eq!(equal.comparison, rv);
        assert!(!equal.passed, "RV == threshold must fail");

        let below_bar = RieszSensitivity {
            pass_threshold: rv * 0.5,
            delta_grid: probe.delta_grid.clone(),
            ..RieszSensitivity::new()
        };
        let past = below_bar.refute(&problem, &mut ws, &ctx).unwrap();
        assert_eq!(past.comparison, rv);
        assert!(past.passed, "RV > threshold must pass");
    }
}
