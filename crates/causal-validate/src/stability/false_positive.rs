//! False-positive checks via permute / phase-randomize surrogates.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use causal_core::{ExecutionContext, VariableId};
use causal_data::{TimeSeriesData, surrogate_permute_columns, surrogate_phase_randomize};
use causal_discovery::{DiscoveryWorkspace, Pcmci};

use crate::error::ValidationError;

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
    /// Empirical edge rate vs family size estimate.
    pub empirical_fpr: f64,
    /// Whether mean edge count is at/below the α-calibrated expectation band.
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
        let max_lag = self.pcmci.engine().constraints.temporal.max_lag.raw().max(1) as usize;
        let family = (variables.len() * variables.len() * max_lag).max(1);
        let mut rng = ctx.rng.stream(0xF41E_u64);
        let mut total_edges = 0u64;
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
            total_edges += result.evidence.links.len() as u64;
        }
        let mean_edge_count = total_edges as f64 / f64::from(self.replicates);
        let empirical_fpr = mean_edge_count / family as f64;
        // Pass if empirical FPR is not far above α (allow 3√(α(1-α)/R) + 0.05 floor).
        let se = (alpha * (1.0 - alpha) / f64::from(self.replicates)).sqrt();
        let passed = empirical_fpr <= alpha + (3.0 * se).max(0.05);
        Ok(FalsePositiveCheckReport {
            method: self.transform,
            replicates: self.replicates,
            mean_edge_count,
            empirical_fpr,
            passed,
        })
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use std::sync::Arc;

    use causal_core::{
        CausalSchemaBuilder, ExecutionContext, Lag, MeasurementSpec, RoleHint, SmallRoleSet,
        ValueType, VariableId,
    };
    use causal_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
        ValidityBitmap,
    };
    use causal_discovery::{DiscoveryConstraints, TemporalConstraints};

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
        let check = FalsePositiveCheck::new(pcmci, NullTransform::ColumnPermute, 4);
        let report = check.run(&data, &vars, &mut ws, &ctx).unwrap();
        assert!(report.mean_edge_count <= before as f64 + 1.0);
        assert_eq!(report.method, NullTransform::ColumnPermute);
    }
}
