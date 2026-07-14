//! Linear temporal mediation effects (Phase 9 / Tigramite effects parity).
//!
//! Path-product decomposition on lagged samples: total = direct + mediated
//! under a linear SEM with a single mediator.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::similar_names)]

use std::sync::Arc;

use causal_core::{ExecutionContext, Lag, MediationContrast, MediationQuery};
use causal_data::{LaggedColumn, SampleWorkspace, TimeSeriesData};
use causal_expr::IdentifiedEstimand;
use causal_stats::{FaerBackend, form_xtx, invert_square};

use crate::adjustment::{EffectEstimate, intervention_f64};
use crate::error::EstimationError;

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
        if !(estimand.method.starts_with("temporal_mediation") || estimand.method.as_ref() == "frontdoor")
        {
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
        let plan = data
            .plan_lagged_sample(1, cols)
            .map_err(|e| EstimationError::Data(e.to_string()))?;
        let mut ws = SampleWorkspace::default();
        let prep = plan.prepare(data, &mut ws).map_err(|e| EstimationError::Data(e.to_string()))?;
        let t = prep.column(0);
        let m = prep.column(1);
        let y = prep.column(2);
        let n = prep.n;
        if n < 4 {
            return Err(EstimationError::Data("insufficient effective samples for mediation".into()));
        }

        // Stage 1: M ~ [1, T] → a = β_T
        let (a, _) = ols_two_col(self.backend, t, m)?;
        // Stage 2: Y ~ [1, T, M] → c' = β_T (direct), b = β_M
        let (c_prime, b) = ols_three_col(self.backend, t, m, y)?;
        // Reduced form: Y ~ [1, T] → c = total
        let (c, _) = ols_two_col(self.backend, t, y)?;

        let total = c * delta;
        let direct = c_prime * delta;
        let mediated = a * b * delta;

        let point = match query.contrast {
            MediationContrast::Total => total,
            MediationContrast::Direct | MediationContrast::NaturalDirect => direct,
            MediationContrast::Mediated | MediationContrast::NaturalIndirect => mediated,
        };

        Ok(TemporalMediationEstimate {
            effect: EffectEstimate {
                ate: point,
                se_analytic: 0.0,
                se_bootstrap: None,
                assumptions: Default::default(),
                overlap: crate::adjustment::OverlapPolicy::ExplicitOverride,
                overlap_report: None,
                retained_memory_bytes: None,
            },
            total: Some(total),
            direct: Some(direct),
            mediated: Some(mediated),
        })
    }
}

fn ols_two_col(backend: FaerBackend, x: &[f64], y: &[f64]) -> Result<(f64, f64), EstimationError> {
    let n = x.len();
    let mut design = vec![0.0; n * 2];
    for i in 0..n {
        design[i] = 1.0;
        design[n + i] = x[i];
    }
    let coef = ols_fit(backend, &design, 2, y)?;
    Ok((coef[1], coef[0]))
}

fn ols_three_col(
    backend: FaerBackend,
    t: &[f64],
    m: &[f64],
    y: &[f64],
) -> Result<(f64, f64), EstimationError> {
    let n = t.len();
    let mut design = vec![0.0; n * 3];
    for i in 0..n {
        design[i] = 1.0;
        design[n + i] = t[i];
        design[2 * n + i] = m[i];
    }
    let coef = ols_fit(backend, &design, 3, y)?;
    Ok((coef[1], coef[2])) // c', b
}

fn ols_fit(
    backend: FaerBackend,
    design_colmajor: &[f64],
    ncols: usize,
    y: &[f64],
) -> Result<Vec<f64>, EstimationError> {
    let n = y.len();
    let mut xtx = vec![0.0; ncols * ncols];
    let mut xty = vec![0.0; ncols];
    form_xtx(design_colmajor, n, ncols, &mut xtx);
    for c in 0..ncols {
        let col = &design_colmajor[c * n..(c + 1) * n];
        xty[c] = col.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
    }
    let inv = invert_square(&xtx, ncols).ok_or_else(|| {
        EstimationError::Stats("singular design in temporal mediation OLS".into())
    })?;
    let mut coef = vec![0.0; ncols];
    for i in 0..ncols {
        let mut s = 0.0;
        for j in 0..ncols {
            s += inv[i * ncols + j] * xty[j];
        }
        coef[i] = s;
    }
    let _ = backend;
    Ok(coef)
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
        SmallRoleSet, ValueType,
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
                Float64Column::new(VariableId::from_raw(0), Arc::from(t), ValidityBitmap::all_valid(n))
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(1), Arc::from(m), ValidityBitmap::all_valid(n))
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(2), Arc::from(y), ValidityBitmap::all_valid(n))
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
    }
}
