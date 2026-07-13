//! Linear adjustment ATE estimator.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::similar_names)]

use std::sync::Arc;

use causal_core::{
    AssumptionSet, AverageEffectQuery, ExecutionContext, Intervention, TargetPopulation, VariableId,
};
use causal_data::TabularData;
use causal_identify::IdentifiedEstimand;
use causal_stats::{
    CompiledDesign, DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace, form_xtx, invert_square,
};

use crate::error::EstimationError;

/// Overlap / positivity handling (DESIGN §14.3).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OverlapPolicy {
    /// Explicitly skip propensity-based overlap (linear adjustment path).
    ExplicitOverride,
    /// Require propensity diagnostics; optional clip/trim thresholds in `(0, 0.5)`.
    RequireDiagnostics {
        /// Clip propensities into `[clip, 1 - clip]` when `Some`.
        clip: Option<f64>,
        /// Drop units outside `[trim, 1 - trim]` when `Some`.
        trim: Option<f64>,
    },
}

impl OverlapPolicy {
    /// Require diagnostics with no clipping or trimming.
    #[must_use]
    pub const fn require_diagnostics() -> Self {
        Self::RequireDiagnostics { clip: None, trim: None }
    }
}

/// Propensity overlap / positivity report retained on estimates.
#[derive(Clone, Debug, PartialEq)]
pub struct OverlapReport {
    /// Minimum fitted propensity (before clipping).
    pub propensity_min: f64,
    /// Maximum fitted propensity (before clipping).
    pub propensity_max: f64,
    /// Kish effective sample size of the applied weights.
    pub ess: f64,
    /// Count of weights above the extreme-weight threshold (default 10).
    pub extreme_weight_count: u32,
    /// Fraction of rows excluded by trimming (0 if no trim).
    pub excluded_fraction: f64,
    /// Clip threshold applied, if any.
    pub clip: Option<f64>,
    /// Trim threshold applied, if any.
    pub trim: Option<f64>,
}

impl OverlapReport {
    /// Build a report from fitted propensities and optional IPW weights.
    #[must_use]
    pub fn from_propensities(
        propensities: &[f64],
        weights: Option<&[f64]>,
        policy: OverlapPolicy,
    ) -> Self {
        let (clip, trim) = match policy {
            OverlapPolicy::ExplicitOverride => (None, None),
            OverlapPolicy::RequireDiagnostics { clip, trim } => (clip, trim),
        };
        let mut min_p = f64::INFINITY;
        let mut max_p = f64::NEG_INFINITY;
        let mut excluded = 0u32;
        for &p in propensities {
            min_p = min_p.min(p);
            max_p = max_p.max(p);
            if let Some(t) = trim {
                if p < t || p > 1.0 - t {
                    excluded = excluded.saturating_add(1);
                }
            }
        }
        if propensities.is_empty() {
            min_p = f64::NAN;
            max_p = f64::NAN;
        }
        let n = propensities.len().max(1) as f64;
        let excluded_fraction = f64::from(excluded) / n;
        let (ess, extreme_weight_count) = match weights {
            Some(w) if !w.is_empty() => {
                let sum: f64 = w.iter().sum();
                let sum_sq: f64 = w.iter().map(|x| x * x).sum();
                let ess = if sum_sq > 0.0 { (sum * sum) / sum_sq } else { 0.0 };
                let extreme = w.iter().filter(|&&x| x > 10.0).count();
                (ess, u32::try_from(extreme).unwrap_or(u32::MAX))
            }
            _ => (f64::from(u32::try_from(propensities.len()).unwrap_or(u32::MAX)), 0),
        };
        Self {
            propensity_min: min_p,
            propensity_max: max_p,
            ess,
            extreme_weight_count,
            excluded_fraction,
            clip,
            trim,
        }
    }
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
    /// Active − control treatment contrast used for the ATE scaling.
    pub treatment_delta: f64,
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
    /// ATE point estimate `β_T * (active − control)`.
    pub ate: f64,
    /// Analytic IID standard error (homoskedastic).
    pub se_analytic: f64,
    /// Bootstrap standard error (if requested).
    pub se_bootstrap: Option<f64>,
    /// Assumptions carried from identification.
    pub assumptions: AssumptionSet,
    /// Overlap policy recorded on the artifact.
    pub overlap: OverlapPolicy,
    /// Propensity overlap diagnostics when computed.
    pub overlap_report: Option<OverlapReport>,
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

    /// Prepare design from tabular data, identified estimand, and query levels.
    ///
    /// # Errors
    ///
    /// Missing columns, unsupported query options, type errors, or overlap policy not set.
    pub fn prepare(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
    ) -> Result<PreparedEstimationProblem, EstimationError> {
        if self.overlap != OverlapPolicy::ExplicitOverride {
            return Err(EstimationError::Overlap {
                message: "LinearAdjustmentAte requires ExplicitOverride overlap policy",
            });
        }
        if &*estimand.method != "backdoor.adjustment" {
            return Err(EstimationError::IncompatibleEstimand {
                message: "LinearAdjustmentAte expects backdoor.adjustment",
            });
        }
        query.validate().map_err(|e| EstimationError::UnsupportedQuery(e.to_string()))?;
        if !query.effect_modifiers.is_empty() {
            return Err(EstimationError::UnsupportedQuery(
                "Phase 1 linear adjustment does not support effect modifiers".into(),
            ));
        }
        if query.target_population != TargetPopulation::AllObserved {
            return Err(EstimationError::UnsupportedQuery(
                "Phase 1 linear adjustment only supports TargetPopulation::AllObserved".into(),
            ));
        }
        let treatment = query.treatment;
        let outcome = query.outcome;
        let active = intervention_f64(&query.active)?;
        let control = intervention_f64(&query.control)?;
        let treatment_delta = active - control;
        if treatment_delta == 0.0 {
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
        Ok(PreparedEstimationProblem {
            design,
            method: Arc::clone(&estimand.method),
            adjustment_set: Arc::clone(&estimand.adjustment_set),
            overlap: self.overlap,
            treatment_delta,
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
        let ate = fit.coefficients[t_col] * problem.treatment_delta;
        let n = problem.design.nrows as f64;
        let p = problem.design.ncols as f64;
        let sigma2 = fit.rss / (n - p).max(1.0);
        let se_coef = analytic_se_treatment(
            &problem.design.matrix,
            problem.design.nrows,
            problem.design.ncols,
            t_col,
            sigma2,
        );
        let se_analytic = se_coef * problem.treatment_delta.abs();

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
            overlap_report: None,
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
            ates.push(fit.coefficients[t_col] * problem.treatment_delta);
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

pub(crate) fn intervention_f64(intervention: &Intervention) -> Result<f64, EstimationError> {
    match intervention {
        Intervention::Set { value, .. } => value.as_f64().ok_or_else(|| {
            EstimationError::UnsupportedQuery(
                "Phase 1 linear adjustment requires numeric treatment levels".into(),
            )
        }),
        _ => Err(EstimationError::UnsupportedQuery(
            "Phase 1 linear adjustment requires Set interventions".into(),
        )),
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
        AssumptionSet, AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec,
        RoleHint, SmallRoleSet, TargetPopulation, ValueType, VariableId,
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
        let estimand = IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from([VariableId::from_raw(2)]),
            ExprId::from_raw(0),
        );
        (TabularData::new(storage), estimand)
    }

    #[test]
    fn overlap_report_from_propensities() {
        let ps = [0.1, 0.5, 0.9];
        let ws = [10.0, 2.0, 1.111];
        let report = OverlapReport::from_propensities(
            &ps,
            Some(&ws),
            OverlapPolicy::RequireDiagnostics { clip: Some(0.05), trim: Some(0.05) },
        );
        assert!((report.propensity_min - 0.1).abs() < 1e-12);
        assert!((report.propensity_max - 0.9).abs() < 1e-12);
        assert_eq!(report.extreme_weight_count, 0); // none strictly > 10
        assert_eq!(report.clip, Some(0.05));
        assert!((report.excluded_fraction - 0.0).abs() < 1e-12);
    }

    #[test]
    fn rejects_require_diagnostics_on_linear_path() {
        let (data, estimand) = toy();
        let est = LinearAdjustmentAte {
            overlap: OverlapPolicy::require_diagnostics(),
            ..LinearAdjustmentAte::new()
        };
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let err = est.prepare(&data, &estimand, &query).unwrap_err();
        assert!(matches!(err, EstimationError::Overlap { .. }));
    }

    #[test]
    fn recovers_ate_two() {
        let (data, estimand) = toy();
        let est = LinearAdjustmentAte { bootstrap_replicates: 50, ..LinearAdjustmentAte::new() };
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = EstimationWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let effect = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        assert!((effect.ate - 2.0).abs() < 1e-8);
        assert!(effect.se_bootstrap.is_some());
    }

    #[test]
    fn scales_ate_by_level_delta() {
        let (data, estimand) = toy();
        let est = LinearAdjustmentAte { bootstrap_replicates: 0, ..LinearAdjustmentAte::new() };
        let query = AverageEffectQuery::with_levels(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            0.0,
            2.0,
        );
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = EstimationWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let effect = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        // β_T ≈ 2, delta = 2 → ATE ≈ 4
        assert!((effect.ate - 4.0).abs() < 1e-8);
    }

    #[test]
    fn rejects_unsupported_target_population() {
        let (data, estimand) = toy();
        let est = LinearAdjustmentAte::new();
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
                .with_target_population(TargetPopulation::Treated);
        let err = est.prepare(&data, &estimand, &query).unwrap_err();
        assert!(matches!(err, EstimationError::UnsupportedQuery(_)));
    }
}
