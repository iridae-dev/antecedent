//! Linear temporal mediation effects .
//!
//! Path-product decomposition on lagged samples: total = direct + mediated
//! under a linear SEM with a single mediator.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::similar_names
)]

use std::sync::Arc;

use causal_core::{AssumptionSet, ExecutionContext, Lag, MediationContrast, MediationQuery};
use causal_data::{LaggedColumn, LaggedSampleWorkspace, TimeSeriesData};
use causal_expr::IdentifiedEstimand;
use causal_stats::{DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace};

use crate::adjustment::{EffectEstimate, intervention_f64};
use crate::error::EstimationError;
use crate::util::{coefficient_variance, ols_sigma2};

/// Temporal mediation effect estimate with optional decomposition.
#[derive(Clone, Debug)]
pub struct TemporalMediationEstimate {
    /// Requested contrast estimate.
    pub effect: EffectEstimate,
    /// Total effect (when computed).
    pub total: Option<f64>,
    /// Direct effect (when computed).
    pub direct: Option<f64>,
    /// Mediated / indirect effect (when computed).
    pub mediated: Option<f64>,
}

/// Linear temporal mediation estimator (two-stage / path-product).
#[derive(Clone, Debug, Default)]
pub struct TemporalMediationEstimator {
    /// Linear algebra backend.
    pub backend: FaerBackend,
}

impl TemporalMediationEstimator {
    /// Create with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Estimate mediation contrasts from lag-aligned series.
    ///
    /// Treatment at lag 1, mediator and outcome contemporaneous (linear SEM path).
    ///
    /// # Errors
    ///
    /// Incompatible estimand, multi-mediator sets, or OLS failures.
    pub fn estimate(
        &self,
        data: &TimeSeriesData,
        estimand: &IdentifiedEstimand,
        query: &MediationQuery,
        ctx: &ExecutionContext,
    ) -> Result<TemporalMediationEstimate, EstimationError> {
        let _ = ctx;
        query.validate().map_err(|e| EstimationError::UnsupportedQuery(e.to_string()))?;
        if !(estimand.method_kind().ok().is_some_and(|m| {
            m.is_temporal_mediation() || m == causal_expr::EstimandMethod::FrontDoor
        })) {
            return Err(EstimationError::IncompatibleEstimand {
                message: "TemporalMediationEstimator expects temporal_mediation.* or frontdoor",
            });
        }
        if estimand.mediators.len() != 1 {
            return Err(EstimationError::UnsupportedQuery(
                "TemporalMediationEstimator supports exactly one mediator".into(),
            ));
        }
        let mediator = estimand.mediators[0];
        let active = intervention_f64(&query.active)?;
        let control = intervention_f64(&query.control)?;
        let delta = active - control;
        if delta == 0.0 {
            return Err(EstimationError::UnsupportedQuery(
                "active and control treatment levels must differ".into(),
            ));
        }

        let cols = Arc::from([
            LaggedColumn { variable: query.treatment, lag: Lag::from_raw(1) },
            LaggedColumn { variable: mediator, lag: Lag::CONTEMPORANEOUS },
            LaggedColumn { variable: query.outcome, lag: Lag::CONTEMPORANEOUS },
        ]);
        let plan = data.plan_lagged_sample(1, cols).map_err(EstimationError::from)?;
        let mut ws = LaggedSampleWorkspace::default();
        let prep = plan.prepare(data, &mut ws).map_err(EstimationError::from)?;
        let t = prep.column(0);
        let m = prep.column(1);
        let y = prep.column(2);
        let n = prep.n;
        if n < 4 {
            return Err(EstimationError::data_msg("insufficient effective samples for mediation"));
        }

        // Stage 1: M ~ [1, T] → a = β_T
        let (a, _intercept_m, design_a, sigma2_a) = ols_two_col(self.backend, t, m)?;
        // Stage 2: Y ~ [1, T, M] → c' = β_T (direct), b = β_M
        let (c_prime, b, design_b, sigma2_b) = ols_three_col(self.backend, t, m, y)?;
        // Reduced form: Y ~ [1, T] → c = total
        let (c, _intercept_y, design_c, sigma2_c) = ols_two_col(self.backend, t, y)?;

        let total = c * delta;
        let direct = c_prime * delta;
        let mediated = a * b * delta;

        let point = match query.contrast {
            MediationContrast::Total => total,
            MediationContrast::Direct | MediationContrast::NaturalDirect => direct,
            MediationContrast::Mediated | MediationContrast::NaturalIndirect => mediated,
        };

        let se_analytic = match query.contrast {
            MediationContrast::Total => {
                let var_c = coefficient_variance(&design_c, n, 2, 1, sigma2_c);
                (var_c * delta * delta).max(0.0).sqrt()
            }
            MediationContrast::Direct | MediationContrast::NaturalDirect => {
                let var_cp = coefficient_variance(&design_b, n, 3, 1, sigma2_b);
                (var_cp * delta * delta).max(0.0).sqrt()
            }
            MediationContrast::Mediated | MediationContrast::NaturalIndirect => {
                let var_a = coefficient_variance(&design_a, n, 2, 1, sigma2_a);
                let var_b = coefficient_variance(&design_b, n, 3, 2, sigma2_b);
                // Sobel: SE(ab) ≈ sqrt(b² Var(a) + a² Var(b)), then scale by |δ|.
                let var_ab = b * b * var_a + a * a * var_b;
                (var_ab * delta * delta).max(0.0).sqrt()
            }
        };

        Ok(TemporalMediationEstimate {
            effect: EffectEstimate {
                ate: point,
                se_analytic,
                se_bootstrap: None,
                bootstrap_replicates_ok: None,
                bootstrap_replicates_failed: None,
                assumptions: AssumptionSet::default(),
                overlap: crate::overlap::OverlapPolicy::ExplicitOverride,
                overlap_report: None,
                retained_memory_bytes: None,
            },
            total: Some(total),
            direct: Some(direct),
            mediated: Some(mediated),
        })
    }
}

/// Returns `(slope_x, intercept, design [1,x], σ²)`.
fn ols_two_col(
    backend: FaerBackend,
    x: &[f64],
    y: &[f64],
) -> Result<(f64, f64, Vec<f64>, f64), EstimationError> {
    let n = x.len();
    let mut design = vec![0.0; n * 2];
    for i in 0..n {
        design[i] = 1.0;
        design[n + i] = x[i];
    }
    let coef = ols_fit(backend, &design, 2, y)?;
    let sigma2 = ols_sigma2(&design, n, 2, y, &coef);
    Ok((coef[1], coef[0], design, sigma2))
}

/// Returns `(c' = β_T, b = β_M, design [1,T,M], σ²)`.
fn ols_three_col(
    backend: FaerBackend,
    t: &[f64],
    m: &[f64],
    y: &[f64],
) -> Result<(f64, f64, Vec<f64>, f64), EstimationError> {
    let n = t.len();
    let mut design = vec![0.0; n * 3];
    for i in 0..n {
        design[i] = 1.0;
        design[n + i] = t[i];
        design[2 * n + i] = m[i];
    }
    let coef = ols_fit(backend, &design, 3, y)?;
    let sigma2 = ols_sigma2(&design, n, 3, y, &coef);
    Ok((coef[1], coef[2], design, sigma2))
}

fn ols_fit(
    backend: FaerBackend,
    design_colmajor: &[f64],
    ncols: usize,
    y: &[f64],
) -> Result<Vec<f64>, EstimationError> {
    let mut ws = LeastSquaresWorkspace::default();
    let fit = backend
        .least_squares(design_colmajor, y.len(), ncols, y, &mut ws)
        .map_err(crate::util::stats_err)?;
    Ok(fit.coefficients)
}

/// Temporal effect surface aligning with Tigramite (direct / total / mediated / conditional).
#[derive(Clone, Debug)]
pub struct TemporalEffectSurface {
    /// Total effect.
    pub total: f64,
    /// Direct effect.
    pub direct: f64,
    /// Mediated effect.
    pub mediated: f64,
    /// Optional conditional effect at a modifier level (same as total when unmodified).
    pub conditional: Option<f64>,
}

impl TemporalMediationEstimator {
    /// Convenience: return the full Tigramite-style effect surface.
    ///
    /// # Errors
    ///
    /// Propagates [`Self::estimate`].
    pub fn effect_surface(
        &self,
        data: &TimeSeriesData,
        estimand: &IdentifiedEstimand,
        query: &MediationQuery,
        ctx: &ExecutionContext,
    ) -> Result<TemporalEffectSurface, EstimationError> {
        let est = self.estimate(data, estimand, query, ctx)?;
        Ok(TemporalEffectSurface {
            total: est.total.unwrap_or(est.effect.ate),
            direct: est.direct.unwrap_or(0.0),
            mediated: est.mediated.unwrap_or(0.0),
            conditional: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use causal_core::{
        CausalSchemaBuilder, ExecutionContext, MeasurementSpec, MediationContrast, RoleHint,
        SmallRoleSet, ValueType, VariableId,
    };
    use causal_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
        TimeSeriesData, ValidityBitmap,
    };
    use causal_expr::{CausalExprArena, IdentifiedEstimand};

    use super::*;

    fn mediated_series(n: usize) -> (TimeSeriesData, MediationQuery, IdentifiedEstimand) {
        let mut b = CausalSchemaBuilder::new();
        for name in ["t", "m", "y"] {
            b.add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        }
        let schema = b.build().unwrap();
        let mut t = vec![0.0; n];
        let mut m = vec![0.0; n];
        let mut y = vec![0.0; n];
        for i in 1..n {
            t[i] = 0.3 * t[i - 1] + 0.1 * (i as f64).sin();
            m[i] = 0.8 * t[i - 1] + 0.05 * (i as f64).cos();
            y[i] = 0.5 * m[i] + 0.2 * t[i - 1] + 0.02 * (i as f64).sin();
        }
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
                    Arc::from(m),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(2),
                    Arc::from(y),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let data = TimeSeriesData::try_new(
            storage,
            TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
        )
        .unwrap();
        let q = MediationQuery::binary(
            VariableId::from_raw(0),
            VariableId::from_raw(2),
            [VariableId::from_raw(1)],
            MediationContrast::Mediated,
        );
        let mut arena = CausalExprArena::new();
        let functional = arena.frontdoor_ate(
            q.treatment,
            q.outcome,
            &q.mediators,
            causal_core::Value::f64(1.0),
            causal_core::Value::f64(0.0),
        );
        let estimand = IdentifiedEstimand::frontdoor(
            "temporal_mediation.mediated",
            Arc::clone(&q.mediators),
            functional,
        );
        (data, q, estimand)
    }

    #[test]
    fn recovers_positive_mediated_effect() {
        let (data, q, estimand) = mediated_series(300);
        let est = TemporalMediationEstimator::new()
            .estimate(&data, &estimand, &q, &ExecutionContext::for_tests(1))
            .unwrap();
        assert!(est.mediated.unwrap() > 0.1);
        assert!((est.total.unwrap() - est.direct.unwrap() - est.mediated.unwrap()).abs() < 0.15);
        assert!(
            est.effect.se_analytic.is_finite() && est.effect.se_analytic > 0.0,
            "se={}",
            est.effect.se_analytic
        );
    }
}
