//! Synthetic-null discovery calibration.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::Arc;

use causal_core::{
    CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
    VariableId,
};
use causal_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
    TimeSeriesData, ValidityBitmap,
};
use causal_discovery::{DiscoveryWorkspace, Pcmci};
use causal_kernels::standard_normal;

use crate::error::ValidationError;

/// Empirical false-positive calibration under independent noise.
#[derive(Clone, Debug)]
pub struct NullCalibrationReport {
    /// Nominal α used for discovery.
    pub alpha: f64,
    /// Simulations run.
    pub n_sim: u32,
    /// Empirical edge rate (edges per possible directed lag-1 pair per sim, averaged).
    pub empirical_fpr: f64,
    /// Binomial SE under H0 rate = α (rough guide).
    pub se: f64,
    /// Whether `|empirical_fpr − α| ≤ band_tol * se` (or absolute floor).
    pub within_band: bool,
    /// Tolerance multiplier used for `within_band`.
    pub band_tol: f64,
}

/// Monte Carlo FPR calibration for PCMCI under independent Gaussian noise.
#[derive(Clone, Debug)]
pub struct SyntheticNullCalibration {
    /// PCMCI configuration (α should match the intended type-I level).
    pub pcmci: Pcmci,
    /// Significance level expected under the null.
    pub alpha: f64,
    /// Number of independent simulations.
    pub n_sim: u32,
    /// Observations per simulation.
    pub n_obs: usize,
    /// Variables per simulation (≥2).
    pub n_vars: usize,
    /// Band width in SE units for `within_band` (default 3).
    pub band_tol: f64,
}

impl SyntheticNullCalibration {
    /// Build a null calibrator.
    #[must_use]
    pub fn new(pcmci: Pcmci, alpha: f64, n_sim: u32, n_obs: usize, n_vars: usize) -> Self {
        Self { pcmci, alpha, n_sim, n_obs, n_vars, band_tol: 3.0 }
    }

    /// Run synthetic-null calibration.
    ///
    /// # Errors
    ///
    /// Invalid config or discovery failures.
    pub fn run(
        &self,
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<NullCalibrationReport, ValidationError> {
        if self.n_sim == 0 || self.n_obs < 8 || self.n_vars < 2 {
            return Err(ValidationError::NotApplicable {
                message: "synthetic-null needs n_sim>0, n_obs≥8, n_vars≥2",
            });
        }
        if !(self.alpha > 0.0 && self.alpha <= 1.0) {
            return Err(ValidationError::NotApplicable {
                message: "synthetic-null alpha must be in (0, 1]",
            });
        }
        let variables: Vec<VariableId> =
            (0..self.n_vars as u32).map(VariableId::from_raw).collect();
        // Possible directed lag edges under max_lag (approx family size for FPR).
        let max_lag = self.pcmci.engine().constraints.temporal.max_lag.raw().max(1);
        let family = (self.n_vars * self.n_vars.saturating_sub(0) * max_lag as usize).max(1);
        let mut rng = ctx.rng.stream(0x5011_u64);
        let mut edge_hits = 0u64;
        let mut trials = 0u64;
        for _ in 0..self.n_sim {
            let data = independent_noise_series(self.n_obs, self.n_vars, &mut rng)?;
            let result =
                self.pcmci.run(&data, &variables, workspace, ctx).map_err(ValidationError::from)?;
            edge_hits += result.evidence.links.len() as u64;
            trials += family as u64;
        }
        let empirical_fpr = if trials == 0 { 0.0 } else { edge_hits as f64 / trials as f64 };
        let se = (self.alpha * (1.0 - self.alpha) / f64::from(self.n_sim)).sqrt();
        let abs_floor = 0.05;
        let within_band = (empirical_fpr - self.alpha).abs() <= (self.band_tol * se).max(abs_floor);
        Ok(NullCalibrationReport {
            alpha: self.alpha,
            n_sim: self.n_sim,
            empirical_fpr,
            se,
            within_band,
            band_tol: self.band_tol,
        })
    }
}

fn independent_noise_series(
    n_obs: usize,
    n_vars: usize,
    rng: &mut causal_core::CausalRng,
) -> Result<TimeSeriesData, ValidationError> {
    let mut b = CausalSchemaBuilder::new();
    for i in 0..n_vars {
        b.add_variable(
            format!("v{i}"),
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .map_err(|_| ValidationError::NotApplicable {
            message: "synthetic-null schema variable rejected",
        })?;
    }
    let schema = b.build().map_err(|_| ValidationError::NotApplicable {
        message: "synthetic-null schema build failed",
    })?;
    let mut cols = Vec::with_capacity(n_vars);
    for v in 0..n_vars {
        let values: Vec<f64> = (0..n_obs).map(|_| standard_normal(rng)).collect();
        cols.push(OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(v as u32),
                Arc::from(values),
                ValidityBitmap::all_valid(n_obs),
            )
            .map_err(ValidationError::from)?,
        ));
    }
    let storage =
        OwnedColumnarStorage::try_new(schema, cols, None, None).map_err(ValidationError::from)?;
    TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n_obs },
    )
    .map_err(ValidationError::from)
}

#[cfg(test)]
mod tests {
    use causal_core::{ExecutionContext, Lag};
    use causal_discovery::{DiscoveryConstraints, DiscoveryWorkspace, TemporalConstraints};

    use super::*;

    #[test]
    fn synthetic_null_smoke() {
        let constraints = DiscoveryConstraints {
            temporal: TemporalConstraints { max_lag: Lag::from_raw(1), min_lag: Lag::from_raw(1) },
            max_cond_size: 1,
            alpha: 0.05,
            ..Default::default()
        };
        let cal = SyntheticNullCalibration::new(
            Pcmci::new().with_fdr(false).with_constraints(constraints),
            0.05,
            2,
            80,
            2,
        );
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(2);
        let report = cal.run(&mut ws, &ctx).unwrap();
        assert_eq!(report.n_sim, 2);
        assert!(report.empirical_fpr >= 0.0);
    }

    /// Scheduled calibration gate: empirical FPR near α under independent noise.
    #[test]
    #[ignore = "scheduled calibration gate; run via scripts/gate_calibration.sh"]
    fn synthetic_null_fpr_near_alpha_gate() {
        let constraints = DiscoveryConstraints {
            temporal: TemporalConstraints { max_lag: Lag::from_raw(1), min_lag: Lag::from_raw(1) },
            max_cond_size: 1,
            alpha: 0.05,
            ..Default::default()
        };
        let mut cal = SyntheticNullCalibration::new(
            Pcmci::new().with_fdr(false).with_constraints(constraints),
            0.05,
            40,
            200,
            3,
        );
        cal.band_tol = 4.0;
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(42);
        let report = cal.run(&mut ws, &ctx).unwrap();
        assert!(
            report.within_band,
            "FPR={} α={} se={} band_tol={}",
            report.empirical_fpr, report.alpha, report.se, report.band_tol
        );
    }
}
