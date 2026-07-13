//! Generalized linear model (logistic) adjustment ATE estimator for binary outcomes (Phase 4).
//!
//! Fits a logistic GLM `Y ~ T + Z` and recovers the ATE by finite-difference g-computation:
//! the fitted model is evaluated at `T = active` and `T = control` for every row (holding `Z`
//! fixed), and the ATE is the mean of the per-row predicted-probability contrast. This is the
//! standard g-computation contrast for a non-identity link, since the coefficient on `T` alone
//! is a log-odds-ratio, not a probability-scale effect.
//!
//! Positivity is handled the same way as [`crate::adjustment::LinearAdjustmentAte`]:
//! [`OverlapPolicy::ExplicitOverride`] is the only supported policy, since this is a regression
//! (not propensity-based) path.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::float_cmp
)]

use std::sync::Arc;

use causal_core::{
    AssumptionSet, AverageEffectQuery, ExecutionContext, TargetPopulation, VariableId,
};
use causal_data::TabularData;
use causal_identify::IdentifiedEstimand;
use causal_stats::{
    CompiledDesign, FaerBackend, GlmDesignRef, GlmFamily, GlmOptions, LeastSquaresWorkspace,
    fit_glm,
};

use crate::adjustment::{EffectEstimate, OverlapPolicy, intervention_f64};
use crate::error::EstimationError;
use crate::propensity::{sample_std, stats_err};

/// Prepared GLM adjustment problem (compiled design retained).
#[derive(Clone, Debug)]
pub struct PreparedGlmProblem {
    /// Compiled `[1 | T | Z…]` design; outcome must be binary (0/1).
    pub design: CompiledDesign,
    /// Estimand method tag.
    pub method: Arc<str>,
    /// Adjustment set.
    pub adjustment_set: Arc<[VariableId]>,
    /// Overlap policy applied.
    pub overlap: OverlapPolicy,
    /// Active treatment level used for the g-computation contrast.
    pub active: f64,
    /// Control treatment level used for the g-computation contrast.
    pub control: f64,
    /// GLM family used for this problem.
    pub family: GlmFamily,
}

/// Estimation workspace (reusable across bootstrap replicates).
#[derive(Clone, Debug, Default)]
pub struct GlmAdjustmentWorkspace {
    /// IRLS least-squares scratch.
    pub ols: LeastSquaresWorkspace,
}

/// Logistic GLM adjustment estimator for binary-outcome backdoor ATE.
///
/// ATE is estimated by finite-difference g-computation: `mean(μ(T=active, Z) − μ(T=control, Z))`
/// over the complete-case sample, where `μ` is the fitted logistic mean function.
#[derive(Clone, Debug)]
pub struct GlmAdjustmentAte {
    /// Dense linear-algebra backend used by the IRLS inner loop.
    pub backend: FaerBackend,
    /// Bootstrap replicates (0 = skip bootstrap).
    pub bootstrap_replicates: u32,
    /// Overlap policy (must be [`OverlapPolicy::ExplicitOverride`]).
    pub overlap: OverlapPolicy,
    /// GLM fitting options (max iterations, convergence tolerance).
    pub glm_options: GlmOptions,
    /// Outcome family / link.
    pub family: GlmFamily,
}

impl Default for GlmAdjustmentAte {
    fn default() -> Self {
        Self::new()
    }
}

impl GlmAdjustmentAte {
    /// Default: 200 bootstrap replicates, explicit overlap override.
    #[must_use]
    pub fn new() -> Self {
        Self {
            backend: FaerBackend,
            bootstrap_replicates: 200,
            overlap: OverlapPolicy::ExplicitOverride,
            glm_options: GlmOptions::default(),
            family: GlmFamily::BinomialLogit,
        }
    }

    /// Prepare design from tabular data, identified estimand, and query levels.
    ///
    /// Accepts `backdoor.adjustment` / `backdoor.efficient` estimands.
    ///
    /// # Errors
    ///
    /// Overlap policy is not `ExplicitOverride`, incompatible estimand, unsupported query, or
    /// missing/invalid data columns.
    pub fn prepare(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
    ) -> Result<PreparedGlmProblem, EstimationError> {
        crate::util::require_explicit_override(
            self.overlap,
            "GlmAdjustmentAte requires ExplicitOverride overlap policy",
        )?;
        if &*estimand.method != "backdoor.adjustment" && &*estimand.method != "backdoor.efficient"
        {
            return Err(EstimationError::IncompatibleEstimand {
                message: "GlmAdjustmentAte expects backdoor.adjustment or backdoor.efficient",
            });
        }
        query.validate().map_err(|e| EstimationError::UnsupportedQuery(e.to_string()))?;
        if !query.effect_modifiers.is_empty() {
            return Err(EstimationError::UnsupportedQuery(
                "GLM adjustment (Phase 4) does not support effect modifiers".into(),
            ));
        }
        if query.target_population != TargetPopulation::AllObserved {
            return Err(EstimationError::UnsupportedQuery(
                "GLM adjustment (Phase 4) only supports TargetPopulation::AllObserved".into(),
            ));
        }
        let treatment = query.treatment;
        let outcome = query.outcome;
        let active = intervention_f64(&query.active)?;
        let control = intervention_f64(&query.control)?;
        if active == control {
            return Err(EstimationError::UnsupportedQuery(
                "active and control treatment levels must differ".into(),
            ));
        }

        let mut ids = Vec::with_capacity(2 + estimand.adjustment_set.len());
        ids.push(treatment);
        ids.push(outcome);
        ids.extend_from_slice(&estimand.adjustment_set);
        let row_mask =
            data.complete_case_mask(&ids).map_err(|e| EstimationError::Data(e.to_string()))?;
        let t = data
            .float64_masked(treatment, &row_mask)
            .map_err(|e| EstimationError::Data(e.to_string()))?;
        let y = data
            .float64_masked(outcome, &row_mask)
            .map_err(|e| EstimationError::Data(e.to_string()))?;
        match self.family {
            GlmFamily::BinomialLogit => {
                for &yi in &y {
                    if !(yi == 0.0 || yi == 1.0) {
                        return Err(EstimationError::UnsupportedQuery(
                            "BinomialLogit GlmAdjustmentAte requires a binary (0/1) outcome".into(),
                        ));
                    }
                }
            }
            GlmFamily::PoissonLog => {
                for &yi in &y {
                    if !(yi.is_finite() && yi >= 0.0) {
                        return Err(EstimationError::UnsupportedQuery(
                            "PoissonLog GlmAdjustmentAte requires non-negative outcomes".into(),
                        ));
                    }
                }
            }
            GlmFamily::GaussianIdentity => {}
        }
        let mut covs: Vec<(VariableId, Vec<f64>)> = Vec::new();
        for &z in estimand.adjustment_set.iter() {
            covs.push((
                z,
                data.float64_masked(z, &row_mask)
                    .map_err(|e| EstimationError::Data(e.to_string()))?,
            ));
        }
        let cov_refs: Vec<(VariableId, &[f64])> =
            covs.iter().map(|(id, v)| (*id, v.as_slice())).collect();
        let selected_rows: Vec<usize> =
            row_mask.iter().enumerate().filter_map(|(i, keep)| keep.then_some(i)).collect();
        let design = CompiledDesign::linear_adjustment(&t, &cov_refs, &y, &selected_rows)
            .map_err(|e| EstimationError::Stats(e.to_string()))?;
        Ok(PreparedGlmProblem {
            design,
            method: Arc::clone(&estimand.method),
            adjustment_set: Arc::clone(&estimand.adjustment_set),
            overlap: self.overlap,
            active,
            control,
            family: self.family,
        })
    }

    /// Fit the logistic GLM and compute the g-computation ATE, with optional bootstrap.
    ///
    /// # Errors
    ///
    /// GLM/backend failure.
    pub fn fit(
        &self,
        problem: &PreparedGlmProblem,
        workspace: &mut GlmAdjustmentWorkspace,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        let t_col = problem
            .design
            .treatment_column()
            .ok_or_else(|| EstimationError::Stats("missing treatment column".into()))?;
        let glm_fit = fit_glm(
            problem.family,
            GlmDesignRef {
                x_colmajor: &problem.design.matrix,
                nrows: problem.design.nrows,
                ncols: problem.design.ncols,
                y: &problem.design.outcome,
            },
            &self.backend,
            &mut workspace.ols,
            &self.glm_options,
        )
        .map_err(stats_err)?;

        let diffs = g_computation_diffs(
            problem.family,
            &problem.design.matrix,
            problem.design.nrows,
            problem.design.ncols,
            t_col,
            &glm_fit.coefficients,
            problem.active,
            problem.control,
        );
        let n = diffs.len() as f64;
        let ate = diffs.iter().sum::<f64>() / n;
        // Naive per-unit dispersion of the g-computation contrast; ignores GLM parameter
        // estimation uncertainty (ok as a rough spread indicator — the bootstrap SE below
        // refits the GLM on every resample and is the recommended uncertainty estimate).
        let se_analytic = sample_std(&diffs) / n.sqrt();

        let se_bootstrap = if self.bootstrap_replicates == 0 {
            None
        } else {
            Some(self.bootstrap_se(problem, workspace, ctx, t_col))
        };

        Ok(EffectEstimate {
            ate,
            se_analytic,
            se_bootstrap,
            assumptions,
            overlap: problem.overlap,
            overlap_report: None,
        })
    }

    fn bootstrap_se(
        &self,
        problem: &PreparedGlmProblem,
        workspace: &mut GlmAdjustmentWorkspace,
        ctx: &ExecutionContext,
        t_col: usize,
    ) -> f64 {
        let mut rng = ctx.rng.stream(0xC17A_u64);
        let n = problem.design.nrows;
        let p = problem.design.ncols;
        let mut x_boot = vec![0.0; n * p];
        let mut y_boot = vec![0.0; n];
        let mut estimates = Vec::with_capacity(self.bootstrap_replicates as usize);
        for _ in 0..self.bootstrap_replicates {
            for r in 0..n {
                let idx = (rng.next_u64() as usize) % n;
                y_boot[r] = problem.design.outcome[idx];
                for c in 0..p {
                    x_boot[c * n + r] = problem.design.matrix[c * n + idx];
                }
            }
            let Ok(fit) = fit_glm(
                problem.family,
                GlmDesignRef { x_colmajor: &x_boot, nrows: n, ncols: p, y: &y_boot },
                &self.backend,
                &mut workspace.ols,
                &self.glm_options,
            ) else {
                continue;
            };
            let diffs = g_computation_diffs(
                problem.family,
                &x_boot,
                n,
                p,
                t_col,
                &fit.coefficients,
                problem.active,
                problem.control,
            );
            let m = diffs.len() as f64;
            estimates.push(diffs.iter().sum::<f64>() / m);
        }
        if estimates.len() < 2 {
            return f64::NAN;
        }
        sample_std(&estimates)
    }
}

/// Per-row mean-scale contrast `μ(T=active, Z) − μ(T=control, Z)`.
fn g_computation_diffs(
    family: GlmFamily,
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    t_col: usize,
    coefficients: &[f64],
    active: f64,
    control: f64,
) -> Vec<f64> {
    let mut diffs = Vec::with_capacity(nrows);
    for r in 0..nrows {
        let mut eta_active = 0.0;
        let mut eta_control = 0.0;
        for c in 0..ncols {
            let coef = coefficients[c];
            if c == t_col {
                eta_active += active * coef;
                eta_control += control * coef;
            } else {
                let val = x_colmajor[c * nrows + r];
                eta_active += val * coef;
                eta_control += val * coef;
            }
        }
        let mu_active = match family {
            GlmFamily::BinomialLogit => 1.0 / (1.0 + (-eta_active).exp()),
            GlmFamily::GaussianIdentity => eta_active,
            GlmFamily::PoissonLog => eta_active.exp(),
        };
        let mu_control = match family {
            GlmFamily::BinomialLogit => 1.0 / (1.0 + (-eta_control).exp()),
            GlmFamily::GaussianIdentity => eta_control,
            GlmFamily::PoissonLog => eta_control.exp(),
        };
        diffs.push(mu_active - mu_control);
    }
    diffs
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
    use causal_identify::IdentifiedEstimand;

    use super::*;
    use crate::adjustment::OverlapPolicy;

    /// Binary-outcome SCM: `Z ~ U(-0.5, 0.5)`, `T ∈ {0,1}`, `logit(Y=1) = -0.5 + 2T + Z`.
    fn binary_scm(n: usize, seed: u64) -> (TabularData, IdentifiedEstimand) {
        let mut rng = ExecutionContext::for_tests(seed).rng.stream(0xABCD_u64);
        let mut t = vec![0.0; n];
        let mut z = vec![0.0; n];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let ti = (i % 2) as f64;
            let zi = (i as f64) / (n as f64) - 0.5;
            let logit = -0.5 + 2.0 * ti + zi;
            let p = 1.0 / (1.0 + (-logit).exp());
            let yi = if rng.next_f64() < p { 1.0 } else { 0.0 };
            t[i] = ti;
            z[i] = zi;
            y[i] = yi;
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
            "z",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(0), Arc::from(t), ValidityBitmap::all_valid(n))
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(1), Arc::from(y), ValidityBitmap::all_valid(n))
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(2), Arc::from(z), ValidityBitmap::all_valid(n))
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

    fn ctx() -> ExecutionContext {
        ExecutionContext::for_tests(11)
    }

    #[test]
    fn recovers_positive_ate_on_binary_outcome() {
        let (data, estimand) = binary_scm(4000, 1);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let est = GlmAdjustmentAte { bootstrap_replicates: 30, ..GlmAdjustmentAte::new() };
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = GlmAdjustmentWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        assert!(effect.ate > 0.0, "ate={}", effect.ate);
        assert!(effect.ate < 1.0, "ate={}", effect.ate);
        assert!(effect.se_bootstrap.is_some());
    }

    #[test]
    fn works_with_efficient_backdoor_estimand() {
        let (data, mut estimand) = binary_scm(2000, 2);
        estimand.method = Arc::from("backdoor.efficient");
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let est = GlmAdjustmentAte { bootstrap_replicates: 0, ..GlmAdjustmentAte::new() };
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = GlmAdjustmentWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        assert!(effect.ate > 0.0, "ate={}", effect.ate);
    }

    #[test]
    fn rejects_require_diagnostics_overlap() {
        let (data, estimand) = binary_scm(200, 3);
        let est = GlmAdjustmentAte {
            overlap: OverlapPolicy::require_diagnostics(),
            ..GlmAdjustmentAte::new()
        };
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let err = est.prepare(&data, &estimand, &query).unwrap_err();
        assert!(matches!(err, EstimationError::Overlap { .. }));
    }

    #[test]
    fn rejects_non_binary_outcome() {
        let (data, estimand) = binary_scm(200, 4);
        // Replace outcome with a non-binary value to trigger the validation path.
        let (data, _) = data.with_appended_float("dummy", Arc::from(vec![0.0; 200])).unwrap();
        let bad_y = (0..200).map(f64::from).collect::<Vec<_>>();
        let data = data.with_replaced_float(VariableId::from_raw(1), Arc::from(bad_y)).unwrap();
        let est = GlmAdjustmentAte::new();
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let err = est.prepare(&data, &estimand, &query).unwrap_err();
        assert!(matches!(err, EstimationError::UnsupportedQuery(_)));
    }

    #[test]
    fn rejects_unsupported_target_population() {
        let (data, estimand) = binary_scm(200, 5);
        let est = GlmAdjustmentAte::new();
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
                .with_target_population(TargetPopulation::Treated);
        let err = est.prepare(&data, &estimand, &query).unwrap_err();
        assert!(matches!(err, EstimationError::UnsupportedQuery(_)));
    }
}
