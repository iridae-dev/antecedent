//! Synthetic-null discovery calibration.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

use std::sync::Arc;

use antecedent_core::{
    CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
    VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
    TimeSeriesData, ValidityBitmap,
};
use antecedent_discovery::{DiscoveryWorkspace, Pcmci};
use antecedent_kernels::standard_normal;

use crate::error::ValidationError;
use crate::stability::{lagged_link_family, null_rate_se};

/// Empirical false-positive calibration under independent noise.
#[derive(Clone, Debug)]
pub struct NullCalibrationReport {
    /// Nominal α used for discovery.
    pub alpha: f64,
    /// Simulations run.
    pub n_sim: u32,
    /// Empirical edge rate: retained links over `n_sim × family` candidate lagged links, where
    /// the family is every ordered variable pair at each lag of `min_lag..=max_lag`.
    pub empirical_fpr: f64,
    /// Standard error of `empirical_fpr`: the larger of the binomial value over all
    /// `n_sim × family` link trials and the empirical run-to-run spread (links of one run
    /// share a sample, so their hits are dependent).
    pub se: f64,
    /// Whether `|empirical_fpr − α| ≤ band_tol * se`. Both a rate that is too high and one that
    /// is too low (a test that never rejects) leave the band.
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
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the synthetic-null variable count is a small simulation size, far below 2^32"
    )]
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
        // Candidate lagged links PCMCI scores under its own lag constraints.
        let lags = &self.pcmci.engine().constraints.temporal;
        let family = lagged_link_family(self.n_vars, lags.min_lag.raw(), lags.max_lag.raw())?;
        let mut rng = ctx.rng.stream(0x5011_u64);
        let mut hits_per_run = Vec::with_capacity(self.n_sim as usize);
        for _ in 0..self.n_sim {
            let data = independent_noise_series(self.n_obs, self.n_vars, &mut rng)?;
            let result =
                self.pcmci.run(&data, &variables, workspace, ctx).map_err(ValidationError::from)?;
            hits_per_run.push(result.evidence.links.len() as u64);
        }
        let trials = u64::from(self.n_sim) * family as u64;
        let edge_hits: u64 = hits_per_run.iter().sum();
        let empirical_fpr = edge_hits as f64 / trials as f64;
        let se = null_rate_se(self.alpha, family, &hits_per_run);
        let within_band = (empirical_fpr - self.alpha).abs() <= self.band_tol * se;
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

#[allow(
    clippy::cast_possible_truncation,
    reason = "the synthetic-null variable count is a small simulation size, far below 2^32"
)]
fn independent_noise_series(
    n_obs: usize,
    n_vars: usize,
    rng: &mut antecedent_core::CausalRng,
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
    use antecedent_core::{ExecutionContext, Lag};
    use antecedent_discovery::{DiscoveryConstraints, DiscoveryWorkspace, TemporalConstraints};

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

    /// Empirical FPR near α under independent noise. Runs in well under a second
    /// (`n_sim=40`, `n_obs=200`, `n_vars=3`), so it stays in the default `cargo
    /// test` suite rather than the locally measured `scripts/gate_calibration.sh` gate — a
    /// regression in the PCMCI significance calibration should fail CI, not wait
    /// a week to be caught.
    #[test]
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
