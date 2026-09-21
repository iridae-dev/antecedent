//! Conditional (associational) prediction from a lagged linear regression.
//!
//! [`TemporalLinearPredictor`] regresses a target on caller-listed lagged columns and
//! evaluates the fitted line with one variable's columns held at a level. There is no graph,
//! no identification step and no adjustment, so the output is the conditional expectation
//! `E[target_t | listed lags, variable = level]` under the linear fit. It equals an
//! interventional mean only if the listed columns happen to be a valid adjustment set, the
//! relation is linear, and no other listed column is a descendant of the held variable;
//! none of that is checked here. Identified interventional quantities, with assumptions,
//! support and uncertainty, come from the temporal analysis path of the `antecedent` facade.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::needless_range_loop)]

use std::sync::Arc;

use antecedent_core::{KernelPolicy, Lag, VariableId};
use antecedent_data::{LaggedColumn, LaggedSampleWorkspace, TimeSeriesData};
use antecedent_stats::{DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace};

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
    /// Sample / OLS failures, non-finite values, or a rank-deficient design (collinear or
    /// constant lagged columns), which the least-squares backend refuses.
    pub fn fit(
        data: &TimeSeriesData,
        target: VariableId,
        parents: impl Into<Arc<[LaggedColumn]>>,
        policy: &KernelPolicy,
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
        let prep = plan.prepare(data, &mut ws, policy).map_err(EstimationError::from)?;
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
        if y.iter().chain(design.iter()).any(|v| !v.is_finite()) {
            return Err(EstimationError::data_msg(
                "lagged linear predictor needs finite target and parent values",
            ));
        }
        let fit = FaerBackend
            .least_squares(&design, n, ncols, y, &mut ws)
            .map_err(EstimationError::from)?;
        Ok(Self { target, parents, coefficients: Arc::from(fit.coefficients), max_lag })
    }

    /// Batch conditional prediction with one variable's columns held at `level`.
    ///
    /// Sets every lag of `held_var` in the design to `level`, leaves every other column at its
    /// observed value, and evaluates the fitted line on the same row geometry. This is a
    /// conditional prediction, not `do(held_var = level)`: see the module docs.
    ///
    /// # Errors
    ///
    /// Sample preparation failures, a non-finite `level`, or a `held_var` that is not one
    /// of the fitted columns (the prediction would silently ignore it).
    pub fn predict_conditional(
        &self,
        data: &TimeSeriesData,
        held_var: VariableId,
        level: f64,
        policy: &KernelPolicy,
    ) -> Result<Arc<[f64]>, EstimationError> {
        if !level.is_finite() {
            return Err(EstimationError::data_msg("prediction level must be finite"));
        }
        if !self.parents.iter().any(|p| p.variable == held_var) {
            return Err(EstimationError::data_msg(
                "held variable is not one of the predictor's lagged columns",
            ));
        }
        let mut cols = Vec::with_capacity(1 + self.parents.len());
        cols.push(LaggedColumn { variable: self.target, lag: Lag::CONTEMPORANEOUS });
        cols.extend_from_slice(&self.parents);
        let plan = data
            .plan_lagged_sample(self.max_lag, Arc::<[LaggedColumn]>::from(cols))
            .map_err(EstimationError::from)?;
        let mut ws = LaggedSampleWorkspace::default();
        let prep = plan.prepare(data, &mut ws, policy).map_err(EstimationError::from)?;
        let n = prep.n;
        let mut out = vec![0.0; n];
        for i in 0..n {
            let mut yhat = self.coefficients[0];
            for (p, parent) in self.parents.iter().enumerate() {
                let x = if parent.variable == held_var { level } else { prep.column(1 + p)[i] };
                yhat += self.coefficients[1 + p] * x;
            }
            out[i] = yhat;
        }
        Ok(Arc::from(out))
    }
}

#[cfg(test)]
mod tests {
    use antecedent_core::StreamDomain;

    use antecedent_core::{
        CausalSchemaBuilder, KernelPolicy, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
        VariableId,
    };
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
        TimeSeriesData, ValidityBitmap,
    };

    use super::*;

    fn series(columns: Vec<(&str, Vec<f64>)>) -> TimeSeriesData {
        let n = columns[0].1.len();
        let mut b = CausalSchemaBuilder::new();
        for (name, _) in &columns {
            b.add_variable(
                *name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        }
        let cols = columns
            .into_iter()
            .enumerate()
            .map(|(idx, (_, values))| {
                OwnedColumn::Float64(
                    Float64Column::new(
                        VariableId::from_raw(idx as u32),
                        Arc::from(values),
                        ValidityBitmap::all_valid(n),
                    )
                    .unwrap(),
                )
            })
            .collect();
        let storage = OwnedColumnarStorage::try_new(b.build().unwrap(), cols, None, None).unwrap();
        TimeSeriesData::try_new(
            storage,
            TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
        )
        .unwrap()
    }

    fn lag1(variable: u32) -> LaggedColumn {
        LaggedColumn { variable: VariableId::from_raw(variable), lag: Lag::from_raw(1) }
    }

    #[test]
    fn conditional_prediction_is_not_the_interventional_mean_under_confounding() {
        // U_t drives X_t and Y_{t+1}; X has no effect on Y, so E[Y | do(X=1)] = E[Y] = 0.
        // The fitted predictor can only return the association E[Y_t | X_{t-1}=1] = 0.8.
        let n = 20_000usize;
        let mut rng = antecedent_core::ExecutionContext::for_tests(7)
            .rng
            .stream_for(StreamDomain::Estimate, 0x9ED1_u64);
        let mut x = vec![0.0; n];
        let mut y = vec![0.0; n];
        let mut u_prev = 0.0;
        for t in 0..n {
            let u = antecedent_kernels::standard_normal(&mut rng);
            x[t] = u + 0.5 * antecedent_kernels::standard_normal(&mut rng);
            y[t] = u_prev + 0.5 * antecedent_kernels::standard_normal(&mut rng);
            u_prev = u;
        }
        let data = series(vec![("x", x), ("y", y)]);
        let policy = KernelPolicy::default_policy();
        let pred = TemporalLinearPredictor::fit(&data, VariableId::from_raw(1), [lag1(0)], &policy)
            .unwrap();
        let yhat = pred.predict_conditional(&data, VariableId::from_raw(0), 1.0, &policy).unwrap();
        let mean: f64 = yhat.iter().sum::<f64>() / yhat.len() as f64;
        assert!((mean - 0.8).abs() < 0.03, "conditional mean={mean}");
    }

    #[test]
    fn fit_refuses_a_rank_deficient_design() {
        let n = 60usize;
        let x: Vec<f64> = (0..n).map(|t| (t as f64 * 0.7).sin()).collect();
        let copy: Vec<f64> = x.iter().map(|v| 2.0 * v).collect();
        let y: Vec<f64> = (0..n).map(|t| (t as f64 * 0.3).cos()).collect();
        let data = series(vec![("x", x), ("y", y), ("x2", copy)]);
        let policy = KernelPolicy::default_policy();
        let err = TemporalLinearPredictor::fit(
            &data,
            VariableId::from_raw(1),
            [lag1(0), lag1(2)],
            &policy,
        )
        .unwrap_err();
        assert!(err.to_string().contains("rank"), "err={err}");
    }

    #[test]
    fn prediction_refuses_a_non_finite_level_and_an_absent_column() {
        let n = 40usize;
        let x: Vec<f64> = (0..n).map(|t| (t as f64 * 0.7).sin()).collect();
        let y: Vec<f64> = x.iter().map(|v| 2.0 * v + 0.1).collect();
        let data = series(vec![("x", x), ("y", y)]);
        let policy = KernelPolicy::default_policy();
        let pred = TemporalLinearPredictor::fit(&data, VariableId::from_raw(1), [lag1(0)], &policy)
            .unwrap();
        let x_id = VariableId::from_raw(0);
        assert!(pred.predict_conditional(&data, x_id, f64::NAN, &policy).is_err());
        assert!(pred.predict_conditional(&data, x_id, f64::INFINITY, &policy).is_err());
        // Holding a variable that is not in the design would silently change nothing.
        assert!(pred.predict_conditional(&data, VariableId::from_raw(1), 1.0, &policy).is_err());
    }

    #[test]
    fn conditional_prediction_batch() {
        let n = 80usize;
        let mut x = vec![0.0; n];
        let mut y = vec![0.0; n];
        for t in 1..n {
            x[t] = 0.5 * x[t - 1] + 0.1;
            y[t] = 2.0 * x[t - 1] + 0.01;
        }
        let data = series(vec![("x", x), ("y", y)]);
        let policy = KernelPolicy::default_policy();
        let pred = TemporalLinearPredictor::fit(
            &data,
            VariableId::from_raw(1),
            [LaggedColumn { variable: VariableId::from_raw(0), lag: Lag::from_raw(1) }],
            &policy,
        )
        .unwrap();
        let yhat = pred.predict_conditional(&data, VariableId::from_raw(0), 1.0, &policy).unwrap();
        assert_eq!(yhat.len(), n - 1);
        let mean: f64 = yhat.iter().sum::<f64>() / yhat.len() as f64;
        assert!((mean - 2.0).abs() < 0.2);
    }
}
