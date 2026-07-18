//! Regime stability via RPCMCI block bootstrap (DESIGN.md §18.3).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::collections::BTreeMap;

use causal_core::{ExecutionContext, RegimeId, VariableId};
use causal_data::{ResamplingPlan, TableView, TimeSeriesData, resample_timeseries};
use causal_discovery::{DiscoveryWorkspace, LaggedLink, RegimeAssignment, Rpcmci};

use crate::error::ValidationError;

use super::pcmci_grid::{DiscoveryStabilityReport, report_from_counts};

/// Per-regime link-frequency stability under fixed caller labels.
#[derive(Clone, Debug)]
pub struct RegimeStabilityReport {
    /// Per-regime discovery stability reports.
    pub per_regime: BTreeMap<RegimeId, DiscoveryStabilityReport>,
    /// Bootstrap replicates.
    pub replicates: u32,
    /// Moving-block length.
    pub block_size: usize,
}

/// Block-bootstrap regime stability around [`Rpcmci`] with fixed assignments.
#[derive(Clone, Debug)]
pub struct RegimeStability {
    /// RPCMCI configuration (`alternating_iters` should be 0 for fixed labels).
    pub rpcmci: Rpcmci,
    /// Caller-supplied regime labels (no unsupervised search).
    pub assignment: RegimeAssignment,
    /// Bootstrap replicates.
    pub replicates: u32,
    /// Block length.
    pub block_size: usize,
}

impl RegimeStability {
    /// Build with fixed assignment; forces `alternating_iters = 0`.
    #[must_use]
    pub fn new(rpcmci: Rpcmci, assignment: RegimeAssignment) -> Self {
        Self {
            rpcmci: rpcmci.with_alternating_iters(0),
            assignment,
            replicates: 10,
            block_size: 20,
        }
    }

    /// Run regime stability assessment.
    ///
    /// Bootstraps the series, re-applies the same regime labels, and re-runs RPCMCI.
    ///
    /// # Errors
    ///
    /// Length mismatch, empty configs, or discovery failures.
    pub fn run(
        &self,
        data: &TimeSeriesData,
        variables: &[VariableId],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<RegimeStabilityReport, ValidationError> {
        if self.replicates == 0 || self.block_size == 0 {
            return Err(ValidationError::NotApplicable {
                message: "regime stability requires positive replicates and block_size",
            });
        }
        if self.assignment.len() != data.row_count() {
            return Err(ValidationError::NotApplicable {
                message: "regime assignment length must match series length",
            });
        }
        if self.block_size > data.row_count() {
            return Err(ValidationError::NotApplicable {
                message: "block_size exceeds series length",
            });
        }
        let regimes = self.assignment.unique_regimes();
        let mut counts: BTreeMap<RegimeId, BTreeMap<LaggedLink, u32>> = BTreeMap::new();
        for &r in &regimes {
            counts.insert(r, BTreeMap::new());
        }
        let mut rng = ctx.rng.stream(0x5E61_u64);
        let mut index_scratch = Vec::new();
        for _ in 0..self.replicates {
            let boot = resample_timeseries(
                data,
                ResamplingPlan::MovingBlock { length: self.block_size },
                &mut rng,
                &mut index_scratch,
            )
            .map_err(ValidationError::from)?;
            // Re-apply fixed labels by mapping bootstrap row indexes back to original regimes.
            let boot_labels: Vec<_> = index_scratch
                .iter()
                .map(|&i| {
                    self.assignment
                        .at(i as usize)
                        .expect("bootstrap index in range")
                })
                .collect();
            let boot_assign = RegimeAssignment::try_new(boot_labels).map_err(ValidationError::from)?;
            let result = self
                .rpcmci
                .run(&boot, variables, &boot_assign, workspace, ctx)
                .map_err(ValidationError::from)?;
            for (idx, &(regime, _)) in result.graphs.graphs.iter().enumerate() {
                let Some(per) = result.per_regime.get(idx) else {
                    continue;
                };
                let entry = counts.entry(regime).or_default();
                for s in per.evidence.links.iter() {
                    *entry.entry(s.link).or_insert(0) += 1;
                }
            }
        }
        let mut per_regime = BTreeMap::new();
        for (regime, c) in counts {
            per_regime.insert(regime, report_from_counts(c, self.replicates, self.block_size));
        }
        Ok(RegimeStabilityReport {
            per_regime,
            replicates: self.replicates,
            block_size: self.block_size,
        })
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use causal_core::{
        ExecutionContext, Lag, MeasurementSpec, RoleHint, SmallRoleSet, ValueType, VariableId,
        CausalSchemaBuilder,
    };
    use causal_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
        TimeSeriesData, ValidityBitmap,
    };
    use causal_discovery::{
        DiscoveryConstraints, DiscoveryWorkspace, PcmciPlus, TemporalConstraints,
        two_regime_half_split,
    };
    use std::sync::Arc;

    use super::*;

    fn two_regime_series(n: usize) -> (TimeSeriesData, Vec<VariableId>) {
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
        let mid = n / 2;
        let mut x = vec![0.0; n];
        let mut y = vec![0.0; n];
        for t in 1..n {
            x[t] = 0.5 * x[t - 1] + (t as f64 * 0.01).sin() * 0.1;
            if t < mid {
                y[t] = 0.8 * x[t - 1] + 0.2 * y[t - 1];
            } else {
                y[t] = -0.7 * x[t - 1] + 0.2 * y[t - 1];
            }
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
    fn regime_stability_runs_two_regimes() {
        let n = 200usize;
        let (data, vars) = two_regime_series(n);
        let assign = two_regime_half_split(n);
        let mut constraints = DiscoveryConstraints::default();
        constraints.temporal = TemporalConstraints {
            max_lag: Lag::from_raw(1),
            min_lag: Lag::from_raw(1),
        };
        constraints.max_cond_size = 1;
        constraints.alpha = 0.15;
        let rpcmci = Rpcmci::new()
            .with_pcmci_plus(PcmciPlus::new().with_fdr(false).with_constraints(constraints))
            .with_min_regime_len(40)
            .with_alternating_iters(0);
        let stab = RegimeStability {
            rpcmci,
            assignment: assign,
            replicates: 3,
            block_size: 25,
        };
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(3);
        let report = stab.run(&data, &vars, &mut ws, &ctx).unwrap();
        assert_eq!(report.replicates, 3);
        assert!(!report.per_regime.is_empty());
    }
}
