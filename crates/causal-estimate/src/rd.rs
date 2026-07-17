//! Sharp regression discontinuity estimator .
//!
//! Treatment is defined deterministically by the running variable: `T = 1{running ≥ cutoff}`.
//! The local effect at the cutoff is the coefficient on `T` in a local-linear OLS of `Y` on
//! `[1, T, (R − c), T·(R − c)]`, restricted to rows within `bandwidth` of the cutoff.
//!
//! Bandwidth is explicit configuration in — no data-driven bandwidth selector
//! (Imbens–Kalyanaraman, cross-validation, etc.) is implemented yet.
//!
//! Uses the dedicated method tag `"rd.sharp"` rather than `backdoor.adjustment`, since RD
//! identification does not rely on a backdoor adjustment set: [`prepare`](SharpRegressionDiscontinuity::prepare)
//! accepts any [`IdentifiedEstimand`] carrying that tag, including a synthetic one built for
//! tests via `IdentifiedEstimand::backdoor("rd.sharp", ..)`.
//!
//! Positivity is not meaningful for RD — it is not a propensity-based method — so
//! [`OverlapPolicy::ExplicitOverride`] is the only supported policy, matching
//! [`crate::adjustment::LinearAdjustmentAte`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::similar_names)]

use std::sync::Arc;

use causal_core::{
    AssumptionSet, AverageEffectQuery, ExecutionContext, TargetPopulation, VariableId,
};
use causal_data::TabularData;
use causal_expr::IdentifiedEstimand;
use causal_stats::{
    DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace, form_xtx, invert_square,
};

use crate::adjustment::{EffectEstimate, intervention_f64};
use crate::error::EstimationError;
use crate::overlap::OverlapPolicy;
use crate::util::{bootstrap_se, BootstrapSeResult, stats_err};

/// Local-linear RD design column count: `[1, T, (R-c), T·(R-c)]`.
const RD_NCOLS: usize = 4;
/// Column index of the treatment indicator within the RD design.
const RD_TREATMENT_COL: usize = 1;

/// Prepared sharp-RD problem: local-linear design windowed to `|R − cutoff| ≤ bandwidth`.
#[derive(Clone, Debug)]
pub struct PreparedRdProblem {
    /// Column-major `[1, T, (R-c), T·(R-c)]` design, restricted to the bandwidth window.
    pub matrix: Arc<[f64]>,
    /// Row count within the bandwidth window.
    pub nrows: usize,
    /// Outcome, length `nrows`.
    pub outcome: Arc<[f64]>,
    /// Estimand method tag (always `"rd.sharp"`).
    pub method: Arc<str>,
    /// Cutoff applied.
    pub cutoff: f64,
    /// Bandwidth applied.
    pub bandwidth: f64,
    /// Overlap policy applied.
    pub overlap: OverlapPolicy,
}

/// Estimation workspace (reusable across bootstrap replicates).
#[derive(Clone, Debug, Default)]
pub struct RdWorkspace {
    /// OLS scratch.
    pub ols: LeastSquaresWorkspace,
}

/// Sharp regression discontinuity estimator.
///
/// `running_variable`, `cutoff`, and `bandwidth` are explicit configuration; there is no
/// data-driven bandwidth selector in .
#[derive(Clone, Debug)]
pub struct SharpRegressionDiscontinuity {
    /// Dense linear-algebra backend.
    pub backend: FaerBackend,
    /// Bootstrap replicates (0 = skip bootstrap).
    pub bootstrap_replicates: u32,
    /// Overlap policy (must be [`OverlapPolicy::ExplicitOverride`]).
    pub overlap: OverlapPolicy,
    /// Running (assignment) variable.
    pub running_variable: VariableId,
    /// Discontinuity cutoff.
    pub cutoff: f64,
    /// Symmetric bandwidth around the cutoff (`|R − cutoff| ≤ bandwidth` is retained).
    pub bandwidth: f64,
}

impl SharpRegressionDiscontinuity {
    /// Construct with explicit running variable, cutoff, and bandwidth.
    ///
    /// Defaults: 200 bootstrap replicates, explicit overlap override.
    #[must_use]
    pub fn new(running_variable: VariableId, cutoff: f64, bandwidth: f64) -> Self {
        Self {
            backend: FaerBackend,
            bootstrap_replicates: 200,
            overlap: OverlapPolicy::ExplicitOverride,
            running_variable,
            cutoff,
            bandwidth,
        }
    }

    /// Prepare the windowed local-linear design from tabular data, identified estimand, and
    /// query.
    ///
    /// Accepts any estimand tagged `"rd.sharp"` (including a synthetic one built via
    /// `IdentifiedEstimand::backdoor("rd.sharp", ..)` for tests).
    ///
    /// # Errors
    ///
    /// Overlap policy is not `ExplicitOverride`, incompatible estimand, unsupported query,
    /// missing/invalid data columns, no rows within the bandwidth window, or a window with only
    /// one treatment arm represented.
    pub fn prepare(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
    ) -> Result<PreparedRdProblem, EstimationError> {
        crate::util::require_explicit_override(
            self.overlap,
            "SharpRegressionDiscontinuity requires ExplicitOverride overlap policy",
        )?;
        if estimand.method_kind().ok() != Some(causal_expr::EstimandMethod::RdSharp) {
            return Err(EstimationError::IncompatibleEstimand {
                message: "SharpRegressionDiscontinuity expects an \"rd.sharp\" estimand",
            });
        }
        if self.bandwidth <= 0.0 {
            return Err(EstimationError::UnsupportedQuery("bandwidth must be positive".into()));
        }
        query.validate().map_err(|e| EstimationError::UnsupportedQuery(e.to_string()))?;
        if !query.effect_modifiers.is_empty() {
            return Err(EstimationError::UnsupportedQuery(
                "sharp RD does not support effect modifiers".into(),
            ));
        }
        if query.target_population != TargetPopulation::AllObserved {
            return Err(EstimationError::UnsupportedQuery(
                "sharp RD only supports TargetPopulation::AllObserved".into(),
            ));
        }
        // The sharp-RD estimand is the outcome jump at the cutoff for the 0/1 crossing
        // indicator `T = 1{R ≥ c}` — a local ATE, not a per-unit-of-treatment slope. Scaling
        // the jump by arbitrary query levels (e.g. levels 0/2 doubling the reported effect)
        // would be semantically wrong, so require the canonical binary coding and report the
        // raw jump.
        let active = intervention_f64(&query.active)?;
        let control = intervention_f64(&query.control)?;
        if (active - 1.0).abs() > 1e-12 || control.abs() > 1e-12 {
            return Err(EstimationError::UnsupportedQuery(
                "sharp RD requires binary treatment levels coded active=1.0, control=0.0; the \
                 RD estimand is the raw outcome jump at the cutoff for the 0/1 crossing \
                 indicator and does not scale with query levels"
                    .into(),
            ));
        }

        let ids = [query.outcome, self.running_variable];
        let row_mask = data.complete_case_mask(&ids).map_err(EstimationError::from)?;
        let outcome_full =
            data.float64_masked(query.outcome, &row_mask).map_err(EstimationError::from)?;
        let running_full =
            data.float64_masked(self.running_variable, &row_mask).map_err(EstimationError::from)?;

        let mut y_sel = Vec::new();
        let mut centered_sel = Vec::new();
        let mut treated_sel = Vec::new();
        for i in 0..running_full.len() {
            let centered = running_full[i] - self.cutoff;
            if centered.abs() <= self.bandwidth {
                y_sel.push(outcome_full[i]);
                centered_sel.push(centered);
                treated_sel.push(if centered >= 0.0 { 1.0 } else { 0.0 });
            }
        }
        let nrows = y_sel.len();
        if nrows == 0 {
            return Err(EstimationError::data_msg(
                "no rows within the bandwidth window of the cutoff",
            ));
        }
        let has_treated = treated_sel.iter().any(|&t| t > 0.5);
        let has_control = treated_sel.iter().any(|&t| t < 0.5);
        if !has_treated || !has_control {
            return Err(EstimationError::data_msg(
                "bandwidth window must contain rows on both sides of the cutoff",
            ));
        }

        let matrix = build_rd_matrix(&treated_sel, &centered_sel);

        Ok(PreparedRdProblem {
            matrix: Arc::from(matrix),
            nrows,
            outcome: Arc::from(y_sel),
            method: Arc::clone(&estimand.method),
            cutoff: self.cutoff,
            bandwidth: self.bandwidth,
            overlap: self.overlap,
        })
    }

    /// Fit the local-linear OLS and return the raw jump at the cutoff, with optional
    /// bootstrap. The query levels are constrained to 0/1 in `prepare`, so no level scaling
    /// is applied.
    ///
    /// # Errors
    ///
    /// Backend/rank failure.
    pub fn fit(
        &self,
        problem: &PreparedRdProblem,
        workspace: &mut RdWorkspace,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        let fit = self
            .backend
            .least_squares(
                &problem.matrix,
                problem.nrows,
                RD_NCOLS,
                &problem.outcome,
                &mut workspace.ols,
            )
            .map_err(stats_err)?;
        let ate = fit.coefficients[RD_TREATMENT_COL];
        let n = problem.nrows as f64;
        let p = RD_NCOLS as f64;
        let sigma2 = fit.rss / (n - p).max(1.0);
        let se_analytic = analytic_se_treatment(&problem.matrix, problem.nrows, sigma2);

        let boot = if self.bootstrap_replicates == 0 {
            None
        } else {
            Some(self.bootstrap_se(problem, workspace, ctx)?)
        };

        Ok(EffectEstimate {
            ate,
            se_analytic,
            se_bootstrap: None,
            bootstrap_replicates_ok: None,
            bootstrap_replicates_failed: None,
            assumptions,
            overlap: problem.overlap,
            overlap_report: None,
            retained_memory_bytes: None,
        }
        .with_bootstrap(boot))
    }

    fn bootstrap_se(
        &self,
        problem: &PreparedRdProblem,
        workspace: &mut RdWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<BootstrapSeResult, EstimationError> {
                let n = problem.nrows;
        let mut x_boot = vec![0.0; n * RD_NCOLS];
        let mut y_boot = vec![0.0; n];
        bootstrap_se(self.bootstrap_replicates, ctx, 0x5D0C_u64, n, |idx| {
            for (r, &src) in idx.iter().enumerate() {
                y_boot[r] = problem.outcome[src];
                for c in 0..RD_NCOLS {
                    x_boot[c * n + r] = problem.matrix[c * n + src];
                }
            }
            match self.backend.least_squares(&x_boot, n, RD_NCOLS, &y_boot, &mut workspace.ols) {
                Ok(fit) => Ok(Some(fit.coefficients[RD_TREATMENT_COL])),
                Err(_) => Ok(None),
            }
        })
    }
}

/// Build the column-major `[1, T, (R-c), T·(R-c)]` local-linear design.
fn build_rd_matrix(treated: &[f64], centered: &[f64]) -> Vec<f64> {
    let n = treated.len();
    let mut matrix = vec![0.0; n * RD_NCOLS];
    for r in 0..n {
        matrix[r] = 1.0;
        matrix[n + r] = treated[r];
        matrix[2 * n + r] = centered[r];
        matrix[3 * n + r] = treated[r] * centered[r];
    }
    matrix
}

fn analytic_se_treatment(x_colmajor: &[f64], nrows: usize, sigma2: f64) -> f64 {
    let mut xtx = vec![0.0; RD_NCOLS * RD_NCOLS];
    form_xtx(x_colmajor, nrows, RD_NCOLS, &mut xtx);
    let Some(inv) = invert_square(&xtx, RD_NCOLS) else {
        return f64::NAN;
    };
    (sigma2 * inv[RD_TREATMENT_COL * RD_NCOLS + RD_TREATMENT_COL].max(0.0)).sqrt()
}

#[cfg(test)]
#[allow(clippy::many_single_char_names, clippy::float_cmp)]
mod tests {
    use std::sync::Arc;

    use causal_core::{
        CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint, SmallRoleSet,
        TargetPopulation, ValueType, VariableId,
    };
    use causal_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
    };
    use causal_expr::ExprId;
    use causal_expr::IdentifiedEstimand;

    use super::*;
    use crate::overlap::OverlapPolicy;

    /// `R ~ U(-1, 1)`, `T = 1{R ≥ 0}`, `Y = 2 + 0.5R + 3T − 0.8T·R + noise`. Jump at cutoff = 3.
    fn sharp_rd_scm(n: usize, seed: u64) -> (TabularData, IdentifiedEstimand) {
        let mut rng = ExecutionContext::for_tests(seed).rng.stream(0x8D15_u64);
        let mut r = vec![0.0; n];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let ri = 2.0 * rng.next_f64() - 1.0;
            let ti = if ri >= 0.0 { 1.0 } else { 0.0 };
            let noise = (rng.next_f64() - 0.5) * 0.2;
            r[i] = ri;
            y[i] = 2.0 + 0.5 * ri + 3.0 * ti - 0.8 * ti * ri + noise;
        }

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
            "r",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        // Treatment column (id 0) is unused by RD (T is derived from the running variable),
        // but the query still needs a nominal treatment variable id.
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(vec![0.0; n]),
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
                    Arc::from(r),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let estimand = IdentifiedEstimand::backdoor("rd.sharp", Arc::from([]), ExprId::from_raw(0));
        (TabularData::new(storage), estimand)
    }

    fn ctx() -> ExecutionContext {
        ExecutionContext::for_tests(31)
    }

    #[test]
    fn recovers_jump_of_three() {
        let (data, estimand) = sharp_rd_scm(6000, 1);
        let est = SharpRegressionDiscontinuity {
            bootstrap_replicates: 30,
            ..SharpRegressionDiscontinuity::new(VariableId::from_raw(2), 0.0, 1.0)
        };
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = RdWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        assert!((effect.ate - 3.0).abs() < 0.5, "ate={}", effect.ate);
        assert!(effect.se_bootstrap.is_some());
    }

    #[test]
    fn rejects_non_rd_estimand() {
        let (data, mut estimand) = sharp_rd_scm(200, 2);
        estimand.method = Arc::from("backdoor.adjustment");
        let est = SharpRegressionDiscontinuity::new(VariableId::from_raw(2), 0.0, 1.0);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let err = est.prepare(&data, &estimand, &query).unwrap_err();
        assert!(matches!(err, EstimationError::IncompatibleEstimand { .. }));
    }

    #[test]
    fn rejects_require_diagnostics_overlap() {
        let (data, estimand) = sharp_rd_scm(200, 3);
        let est = SharpRegressionDiscontinuity {
            overlap: OverlapPolicy::require_diagnostics(),
            ..SharpRegressionDiscontinuity::new(VariableId::from_raw(2), 0.0, 1.0)
        };
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let err = est.prepare(&data, &estimand, &query).unwrap_err();
        assert!(matches!(err, EstimationError::Overlap { .. }));
    }

    #[test]
    fn rejects_non_binary_treatment_levels() {
        // Levels 0/2 must be refused rather than doubling the reported jump: the sharp-RD
        // estimand is the raw outcome jump at the cutoff for the 0/1 crossing indicator.
        let (data, estimand) = sharp_rd_scm(200, 6);
        let est = SharpRegressionDiscontinuity::new(VariableId::from_raw(2), 0.0, 1.0);
        let query = AverageEffectQuery::with_levels(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            0.0,
            2.0,
        );
        let err = est.prepare(&data, &estimand, &query).unwrap_err();
        assert!(matches!(err, EstimationError::UnsupportedQuery(_)), "err={err:?}");
    }

    #[test]
    fn rejects_empty_bandwidth_window() {
        let (data, estimand) = sharp_rd_scm(200, 4);
        let est = SharpRegressionDiscontinuity::new(VariableId::from_raw(2), 100.0, 0.01);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let err = est.prepare(&data, &estimand, &query).unwrap_err();
        assert!(matches!(err, EstimationError::Data(_)));
    }

    #[test]
    fn rejects_unsupported_target_population() {
        let (data, estimand) = sharp_rd_scm(200, 5);
        let est = SharpRegressionDiscontinuity::new(VariableId::from_raw(2), 0.0, 1.0);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
                .with_target_population(TargetPopulation::Treated);
        let err = est.prepare(&data, &estimand, &query).unwrap_err();
        assert!(matches!(err, EstimationError::UnsupportedQuery(_)));
    }
}
