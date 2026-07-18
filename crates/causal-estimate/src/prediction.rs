//! Temporal linear model prediction under interventions .
//!
//! Fit a lagged linear SEM once, then batch-predict under `do()` without
//! Python-per-horizon crossings.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::needless_range_loop,
    clippy::similar_names
)]

use std::sync::Arc;

use causal_core::{Lag, VariableId};
use causal_data::{LaggedColumn, LaggedSampleWorkspace, TimeSeriesData};
use causal_stats::{DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace};

use crate::error::EstimationError;

/// Fitted lagged linear predictor for one target.
#[derive(Clone, Debug)]
pub struct TemporalLinearPredictor {
    /// Target variable.
    pub target: VariableId,
    /// Parent columns used in the design (excluding intercept).
    pub parents: Arc<[LaggedColumn]>,
    /// Coefficients `[intercept, parent_0, ...]`.
    pub coefficients: Arc<[f64]>,
    /// Max lag in the design.
    pub max_lag: u32,
}

impl TemporalLinearPredictor {
    /// Fit `target(t) ~ 1 + parents` on lag-aligned samples.
    ///
    /// # Errors
    ///
    /// Sample / OLS failures.
    pub fn fit(
        data: &TimeSeriesData,
        target: VariableId,
        parents: impl Into<Arc<[LaggedColumn]>>,
    ) -> Result<Self, EstimationError> {
        let parents = parents.into();
        let max_lag = parents.iter().map(|p| p.lag.raw()).max().unwrap_or(0);
        let mut cols = Vec::with_capacity(1 + parents.len());
        cols.push(LaggedColumn { variable: target, lag: Lag::CONTEMPORANEOUS });
        cols.extend_from_slice(&parents);
        let plan = data
            .plan_lagged_sample(max_lag, Arc::<[LaggedColumn]>::from(cols))
            .map_err(EstimationError::from)?;
        let mut ws = LaggedSampleWorkspace::default();
        let prep = plan
            .prepare(data, &mut ws, &causal_core::KernelPolicy::default_policy())
            .map_err(EstimationError::from)?;
        let n = prep.n;
        let y = prep.column(0);
        let ncols = 1 + parents.len();
        let mut design = vec![0.0; n * ncols];
        for design_cell in design.iter_mut().take(n) {
            *design_cell = 1.0;
        }
        for (p, _) in parents.iter().enumerate() {
            let col = prep.column(1 + p);
            for i in 0..n {
                design[(1 + p) * n + i] = col[i];
            }
        }
        let mut ws = LeastSquaresWorkspace::default();
        let fit = FaerBackend
            .least_squares(&design, n, ncols, y, &mut ws)
            .map_err(EstimationError::from)?;
        Ok(Self { target, parents, coefficients: Arc::from(fit.coefficients), max_lag })
    }

    /// Batch-predict under a hard intervention on one parent variable (all horizons).
    ///
    /// Sets every lag of `intervene_var` in the design to `level` and evaluates
    /// the linear predictor on the same row geometry (no per-horizon data clone).
    ///
    /// # Errors
    ///
    /// Sample preparation failures.
    pub fn predict_intervened(
        &self,
        data: &TimeSeriesData,
        intervene_var: VariableId,
        level: f64,
    ) -> Result<Arc<[f64]>, EstimationError> {
        let mut cols = Vec::with_capacity(1 + self.parents.len());
        cols.push(LaggedColumn { variable: self.target, lag: Lag::CONTEMPORANEOUS });
        cols.extend_from_slice(&self.parents);
        let plan = data
            .plan_lagged_sample(self.max_lag, Arc::<[LaggedColumn]>::from(cols))
            .map_err(EstimationError::from)?;
        let mut ws = LaggedSampleWorkspace::default();
        let prep = plan
            .prepare(data, &mut ws, &causal_core::KernelPolicy::default_policy())
            .map_err(EstimationError::from)?;
        let n = prep.n;
        let mut out = vec![0.0; n];
        for i in 0..n {
            let mut yhat = self.coefficients[0];
            for (p, parent) in self.parents.iter().enumerate() {
                let x =
                    if parent.variable == intervene_var { level } else { prep.column(1 + p)[i] };
                yhat += self.coefficients[1 + p] * x;
            }
            out[i] = yhat;
        }
        Ok(Arc::from(out))
    }
}

#[cfg(test)]
mod tests {
    use causal_core::{
        CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType, VariableId,
    };
    use causal_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
        TimeSeriesData, ValidityBitmap,
    };

    use super::*;

    #[test]
    fn intervene_and_predict_batch() {
        let n = 80usize;
        let mut b = CausalSchemaBuilder::new();
        for name in ["x", "y"] {
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
        let mut x = vec![0.0; n];
        let mut y = vec![0.0; n];
        for t in 1..n {
            x[t] = 0.5 * x[t - 1] + 0.1;
            y[t] = 2.0 * x[t - 1] + 0.01;
        }
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(x),
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
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let data = TimeSeriesData::try_new(
            storage,
            TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
        )
        .unwrap();
        let pred = TemporalLinearPredictor::fit(
            &data,
            VariableId::from_raw(1),
            [LaggedColumn { variable: VariableId::from_raw(0), lag: Lag::from_raw(1) }],
        )
        .unwrap();
        let yhat = pred.predict_intervened(&data, VariableId::from_raw(0), 1.0).unwrap();
        assert_eq!(yhat.len(), n - 1);
        let mean: f64 = yhat.iter().sum::<f64>() / yhat.len() as f64;
        assert!((mean - 2.0).abs() < 0.2);
    }
}
