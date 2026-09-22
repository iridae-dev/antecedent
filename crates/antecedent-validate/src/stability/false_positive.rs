//! False-positive checks via permute / phase-randomize surrogates.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::{TimeSeriesData, surrogate_permute_columns, surrogate_phase_randomize};
use antecedent_discovery::{DiscoveryWorkspace, Pcmci};

use crate::error::ValidationError;
use crate::stability::{lagged_link_family, null_rate_se};

/// Null transform applied to observed series before rediscovery.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NullTransform {
    /// Independently permute each column.
    ColumnPermute,
    /// Phase-randomize each column (preserve spectrum).
    PhaseRandomize,
}

/// Report from [`FalsePositiveCheck`].
#[derive(Clone, Debug)]
pub struct FalsePositiveCheckReport {
    /// Transform used.
    pub method: NullTransform,
    /// Surrogate replicates.
    pub replicates: u32,
    /// Mean retained edge count after nullification.
    pub mean_edge_count: f64,
    /// Empirical edge rate: retained links per replicate over the candidate lagged links (every
    /// ordered variable pair at each lag of `min_lag..=max_lag`).
    pub empirical_fpr: f64,
    /// Standard error of `empirical_fpr` (see [`SyntheticNullCalibration`]).
    pub se: f64,
    /// Whether the empirical rate is at most `α + 3·se`. One-sided: surrogates that remove
    /// structure should not raise the false-positive rate, and a lower rate is not a failure.
    pub passed: bool,
}

/// Apply surrogate nulls to observed data and re-run PCMCI.
#[derive(Clone, Debug)]
pub struct FalsePositiveCheck {
    /// PCMCI configuration.
    pub pcmci: Pcmci,
    /// Null transform.
    pub transform: NullTransform,
    /// Surrogate replicates.
    pub replicates: u32,
}

impl FalsePositiveCheck {
    /// Build a false-positive check.
    #[must_use]
    pub fn new(pcmci: Pcmci, transform: NullTransform, replicates: u32) -> Self {
        Self { pcmci, transform, replicates }
    }

    /// Run surrogate false-positive assessment on observed `data`.
    ///
    /// # Errors
    ///
    /// Invalid config, surrogate, or discovery failures.
    pub fn run(
        &self,
        data: &TimeSeriesData,
        variables: &[VariableId],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<FalsePositiveCheckReport, ValidationError> {
        if self.replicates == 0 {
            return Err(ValidationError::NotApplicable {
                message: "false-positive check requires positive replicates",
            });
        }
        let alpha = self.pcmci.engine().constraints.alpha;
        let lags = &self.pcmci.engine().constraints.temporal;
        let family = lagged_link_family(variables.len(), lags.min_lag.raw(), lags.max_lag.raw())?;
        let mut rng = ctx.rng.stream(0xF41E_u64);
        let mut hits_per_run = Vec::with_capacity(self.replicates as usize);
        for _ in 0..self.replicates {
            let null = match self.transform {
                NullTransform::ColumnPermute => {
                    surrogate_permute_columns(data, &mut rng).map_err(ValidationError::from)?
                }
                NullTransform::PhaseRandomize => {
                    surrogate_phase_randomize(data, &mut rng).map_err(ValidationError::from)?
                }
            };
            let result =
                self.pcmci.run(&null, variables, workspace, ctx).map_err(ValidationError::from)?;
            hits_per_run.push(result.evidence.links.len() as u64);
        }
        let mean_edge_count = hits_per_run.iter().sum::<u64>() as f64 / f64::from(self.replicates);
        let empirical_fpr = mean_edge_count / family as f64;
        let se = null_rate_se(alpha, family, &hits_per_run);
        let passed = empirical_fpr <= alpha + 3.0 * se;
        Ok(FalsePositiveCheckReport {
            method: self.transform,
            replicates: self.replicates,
            mean_edge_count,
            empirical_fpr,
            se,
            passed,
        })
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use std::sync::Arc;

    use antecedent_core::{
        CausalSchemaBuilder, ExecutionContext, Lag, MeasurementSpec, RoleHint, SmallRoleSet,
        ValueType, VariableId,
    };
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
        ValidityBitmap,
    };
    use antecedent_discovery::{DiscoveryConstraints, TemporalConstraints};

    use super::*;

    fn linked_series() -> (TimeSeriesData, Vec<VariableId>) {
        let n = 200usize;
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
            x[t] = ((t as f64) * 0.02).sin();
            y[t] = 0.9 * x[t - 1];
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
        (data, vec![VariableId::from_raw(0), VariableId::from_raw(1)])
    }

    #[test]
    fn permute_null_reduces_edges() {
        let (data, vars) = linked_series();
        let constraints = DiscoveryConstraints {
            temporal: TemporalConstraints { max_lag: Lag::from_raw(1), min_lag: Lag::from_raw(1) },
            max_cond_size: 1,
            alpha: 0.05,
            ..Default::default()
        };
        let pcmci = Pcmci::new().with_fdr(false).with_constraints(constraints);
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(8);
        let before = pcmci.run(&data, &vars, &mut ws, &ctx).unwrap().evidence.links.len();
        // The smooth series carry at least the two autocorrelation links (x→x, y→y at lag 1).
        assert!(before >= 2, "structured series must keep links, got {before}");
        let check = FalsePositiveCheck::new(pcmci, NullTransform::ColumnPermute, 4);
        let report = check.run(&data, &vars, &mut ws, &ctx).unwrap();
        // Permuting each column destroys the structure: the surviving links are false positives
        // at roughly alpha per candidate link (2 x 2 links at one lag), far below the structured
        // graph's.
        assert!(
            report.mean_edge_count < before as f64,
            "mean surrogate edges {} not below the structured {before}",
            report.mean_edge_count
        );
        assert_eq!(report.method, NullTransform::ColumnPermute);
        // Candidate links: 2 variables squared at a single lag.
        assert!((report.empirical_fpr - report.mean_edge_count / 4.0).abs() < 1e-15);
    }
}
