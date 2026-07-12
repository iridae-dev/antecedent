//! Linear adjustment ATE estimator.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names
)]

use std::sync::Arc;

use causal_core::{AssumptionSet, ExecutionContext, VariableId};
use causal_data::{TableView, TabularData};
use causal_identify::IdentifiedEstimand;
use causal_stats::{
    CompiledDesign, DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace, form_xtx,
    invert_square,
};

use crate::error::EstimationError;

/// Overlap / positivity handling for Phase 1.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum OverlapPolicy {
    /// Explicitly skip propensity-based overlap (Phase 1 OLS path).
    ExplicitOverride,
}

/// Prepared estimation problem (compiled design retained).
#[derive(Clone, Debug)]
pub struct PreparedEstimationProblem {
    /// Compiled design.
    pub design: CompiledDesign,
    /// Estimand method tag.
    pub method: Arc<str>,
    /// Adjustment set.
    pub adjustment_set: Arc<[VariableId]>,
    /// Overlap policy applied.
    pub overlap: OverlapPolicy,
}

/// Estimation workspace (reusable across bootstrap replicates).
#[derive(Clone, Debug, Default)]
pub struct EstimationWorkspace {
    /// OLS scratch.
    pub ols: LeastSquaresWorkspace,
}

/// Point estimate with uncertainty.
#[derive(Clone, Debug)]
pub struct EffectEstimate {
    /// ATE point estimate (treatment coefficient).
    pub ate: f64,
    /// Analytic IID standard error (homoskedastic).
    pub se_analytic: f64,
    /// Bootstrap standard error (if requested).
    pub se_bootstrap: Option<f64>,
    /// Assumptions carried from identification.
    pub assumptions: AssumptionSet,
    /// Overlap policy recorded on the artifact.
    pub overlap: OverlapPolicy,
}

/// Linear adjustment estimator for backdoor ATE.
#[derive(Clone, Debug)]
pub struct LinearAdjustmentAte {
    /// Backend.
    pub backend: FaerBackend,
    /// Bootstrap replicates (0 = skip bootstrap).
    pub bootstrap_replicates: u32,
    /// Overlap policy (must be explicit in Phase 1).
    pub overlap: OverlapPolicy,
}

impl Default for LinearAdjustmentAte {
    fn default() -> Self {
        Self::new()
    }
}

impl LinearAdjustmentAte {
    /// Default: 200 bootstrap replicates, explicit overlap override.
    #[must_use]
    pub fn new() -> Self {
        Self {
            backend: FaerBackend,
            bootstrap_replicates: 200,
            overlap: OverlapPolicy::ExplicitOverride,
        }
    }

    /// Prepare design from tabular data and an identified backdoor estimand.
    ///
    /// # Errors
    ///
    /// Missing columns, type errors, or overlap policy not set.
    pub fn prepare(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        treatment: VariableId,
        outcome: VariableId,
    ) -> Result<PreparedEstimationProblem, EstimationError> {
        if self.overlap != OverlapPolicy::ExplicitOverride {
            return Err(EstimationError::Overlap {
                message: "Phase 1 requires ExplicitOverride overlap policy",
            });
        }
        if &*estimand.method != "backdoor.adjustment" {
            return Err(EstimationError::IncompatibleEstimand {
                message: "LinearAdjustmentAte expects backdoor.adjustment",
            });
        }
        let t = data
            .float64_values(treatment)
            .map_err(|e| EstimationError::Data(e.to_string()))?;
        let y = data
            .float64_values(outcome)
            .map_err(|e| EstimationError::Data(e.to_string()))?;
        let mut covs: Vec<(VariableId, Vec<f64>)> = Vec::new();
        for &z in estimand.adjustment_set.iter() {
            covs.push((
                z,
                data.float64_values(z).map_err(|e| EstimationError::Data(e.to_string()))?,
            ));
        }
        let cov_refs: Vec<(VariableId, &[f64])> =
            covs.iter().map(|(id, v)| (*id, v.as_slice())).collect();
        let design = CompiledDesign::linear_adjustment(&t, &cov_refs, &y)
            .map_err(|e| EstimationError::Stats(e.to_string()))?;
        Ok(PreparedEstimationProblem {
            design,
            method: Arc::clone(&estimand.method),
            adjustment_set: Arc::clone(&estimand.adjustment_set),
            overlap: self.overlap,
        })
    }

    /// Fit ATE with optional IID bootstrap.
    ///
    /// # Errors
    ///
    /// OLS failure.
    pub fn fit(
        &self,
        problem: &PreparedEstimationProblem,
        workspace: &mut EstimationWorkspace,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        let fit = problem
            .design
            .fit_ols(&self.backend, &mut workspace.ols)
            .map_err(|e| EstimationError::Stats(e.to_string()))?;
        let t_col = problem
            .design
            .treatment_column()
            .ok_or_else(|| EstimationError::Stats("missing treatment column".into()))?;
        let ate = fit.coefficients[t_col];
        let n = problem.design.nrows as f64;
        let p = problem.design.ncols as f64;
        let sigma2 = fit.rss / (n - p).max(1.0);
        let se_analytic = analytic_se_treatment(
            &problem.design.matrix,
            problem.design.nrows,
            problem.design.ncols,
            t_col,
            sigma2,
        );

        let se_bootstrap = if self.bootstrap_replicates == 0 {
            None
        } else {
            Some(self.bootstrap_se(problem, workspace, ctx, t_col)?)
        };

        Ok(EffectEstimate {
            ate,
            se_analytic,
            se_bootstrap,
            assumptions,
            overlap: problem.overlap,
        })
    }

    fn bootstrap_se(
        &self,
        problem: &PreparedEstimationProblem,
        workspace: &mut EstimationWorkspace,
        ctx: &ExecutionContext,
        t_col: usize,
    ) -> Result<f64, EstimationError> {
        let mut rng = ctx.rng.stream(0xA7E_u64);
        let n = problem.design.nrows;
        let p = problem.design.ncols;
        let mut ates = Vec::with_capacity(self.bootstrap_replicates as usize);
        let mut x_boot = vec![0.0; n * p];
        let mut y_boot = vec![0.0; n];
        for _ in 0..self.bootstrap_replicates {
            for r in 0..n {
                let idx = (rng.next_u64() as usize) % n;
                y_boot[r] = problem.design.outcome[idx];
                for c in 0..p {
                    x_boot[c * n + r] = problem.design.matrix[c * n + idx];
                }
            }
            let fit = self
                .backend
                .least_squares(&x_boot, n, p, &y_boot, &mut workspace.ols)
                .map_err(|e| EstimationError::Stats(e.to_string()))?;
            ates.push(fit.coefficients[t_col]);
        }
        let mean = ates.iter().sum::<f64>() / ates.len() as f64;
        let var = ates
            .iter()
            .map(|a| {
                let d = a - mean;
                d * d
            })
            .sum::<f64>()
            / (ates.len() as f64 - 1.0).max(1.0);
        Ok(var.sqrt())
    }
}

fn analytic_se_treatment(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    t_col: usize,
    sigma2: f64,
) -> f64 {
    let mut xtx = vec![0.0; ncols * ncols];
    form_xtx(x_colmajor, nrows, ncols, &mut xtx);
    let Some(inv) = invert_square(&xtx, ncols) else {
        return f64::NAN;
    };
    (sigma2 * inv[t_col * ncols + t_col].max(0.0)).sqrt()
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
mod tests {
    use std::sync::Arc;

    use causal_core::{
        AssumptionSet, CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint,
        SmallRoleSet, ValueType, VariableId,
    };
    use causal_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
    };
    use causal_expr::ExprId;
    use causal_identify::IdentifiedEstimand;

    use super::*;

    fn toy() -> (TabularData, IdentifiedEstimand) {
        let n = 100usize;
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
        let y: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * t[i] + z[i]).collect();
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
        let estimand = IdentifiedEstimand {
            method: Arc::from("backdoor.adjustment"),
            adjustment_set: Arc::from([VariableId::from_raw(2)]),
            functional: ExprId::from_raw(0),
        };
        (TabularData::new(storage), estimand)
    }

    #[test]
    fn recovers_ate_two() {
        let (data, estimand) = toy();
        let est = LinearAdjustmentAte {
            bootstrap_replicates: 50,
            ..LinearAdjustmentAte::new()
        };
        let prep = est
            .prepare(
                &data,
                &estimand,
                VariableId::from_raw(0),
                VariableId::from_raw(1),
            )
            .unwrap();
        let mut ws = EstimationWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let effect = est
            .fit(&prep, &mut ws, &ctx, AssumptionSet::new())
            .unwrap();
        assert!((effect.ate - 2.0).abs() < 1e-8);
        assert!(effect.se_bootstrap.is_some());
    }
}
