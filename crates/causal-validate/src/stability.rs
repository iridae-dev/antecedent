//! Block-bootstrap discovery stability (DESIGN.md §18.3 subset).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::collections::BTreeMap;
use std::sync::Arc;

use causal_core::{ExecutionContext, VariableId};
use causal_data::{ResamplingPlan, TimeSeriesData, resample_timeseries};
use causal_discovery::{DiscoveryWorkspace, LaggedLink, Pcmci};

use crate::error::ValidationError;

/// Stability frequency for one lagged link.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinkStability {
    /// Link.
    pub link: LaggedLink,
    /// Fraction of bootstrap replicates retaining the link.
    pub frequency: f64,
}

/// Report from block-bootstrap discovery stability.
#[derive(Clone, Debug)]
pub struct DiscoveryStabilityReport {
    /// Per-link frequencies (links seen in ≥1 replicate).
    pub frequencies: Arc<[LinkStability]>,
    /// Replicates run.
    pub replicates: u32,
    /// Block size used.
    pub block_size: usize,
}

/// Block-bootstrap stability around a [`Pcmci`] configuration.
#[derive(Clone, Debug)]
pub struct BlockBootstrapStability {
    /// PCMCI configuration to re-run.
    pub pcmci: Pcmci,
    /// Bootstrap replicates.
    pub replicates: u32,
    /// Block length.
    pub block_size: usize,
}

impl Default for BlockBootstrapStability {
    fn default() -> Self {
        Self::new()
    }
}

impl BlockBootstrapStability {
    /// Defaults: 20 replicates, block size 20.
    #[must_use]
    pub fn new() -> Self {
        Self { pcmci: Pcmci::new().with_fdr(false), replicates: 20, block_size: 20 }
    }

    /// Run stability assessment.
    ///
    /// # Errors
    ///
    /// Data or discovery failures.
    pub fn run(
        &self,
        data: &TimeSeriesData,
        variables: &[VariableId],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<DiscoveryStabilityReport, ValidationError> {
        if self.replicates == 0 || self.block_size == 0 {
            return Err(ValidationError::NotApplicable {
                message: "stability requires positive replicates and block_size",
            });
        }
        let mut counts: BTreeMap<LaggedLink, u32> = BTreeMap::new();
        let mut rng = ctx.rng.stream(0x57AB_u64);
        let mut index_scratch = Vec::new();
        for _ in 0..self.replicates {
            let boot = resample_timeseries(
                data,
                ResamplingPlan::MovingBlock { length: self.block_size },
                &mut rng,
                &mut index_scratch,
            )
            .map_err(|e| ValidationError::Data(e.to_string()))?;
            let result = self
                .pcmci
                .run(&boot, variables, workspace, ctx)
                .map_err(|e| ValidationError::Estimation(e.to_string()))?;
            for s in result.evidence.links.iter() {
                *counts.entry(s.link).or_insert(0) += 1;
            }
        }
        let mut frequencies = Vec::with_capacity(counts.len());
        for (link, c) in counts {
            frequencies
                .push(LinkStability { link, frequency: f64::from(c) / f64::from(self.replicates) });
        }
        frequencies.sort_by(|a, b| {
            b.frequency.partial_cmp(&a.frequency).unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(DiscoveryStabilityReport {
            frequencies: Arc::from(frequencies),
            replicates: self.replicates,
            block_size: self.block_size,
        })
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
mod tests {
    use causal_core::{
        CausalSchemaBuilder, ExecutionContext, Lag, MeasurementSpec, RoleHint, SmallRoleSet,
        ValueType, VariableId,
    };
    use causal_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
        TimeSeriesData, ValidityBitmap,
    };
    use causal_discovery::{DiscoveryConstraints, DiscoveryWorkspace, TemporalConstraints};

    use super::*;

    fn linked_series() -> (TimeSeriesData, Vec<VariableId>) {
        let n = 300usize;
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "x",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
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
    fn true_link_is_stable() {
        let (data, vars) = linked_series();
        let mut stab = BlockBootstrapStability::new();
        stab.replicates = 8;
        stab.block_size = 25;
        stab.pcmci =
            causal_discovery::Pcmci::new().with_fdr(false).with_constraints(DiscoveryConstraints {
                temporal: TemporalConstraints {
                    max_lag: Lag::from_raw(2),
                    min_lag: Lag::from_raw(1),
                },
                max_cond_size: 1,
                alpha: 0.05,
                ..DiscoveryConstraints::default()
            });
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(5);
        let report = stab.run(&data, &vars, &mut ws, &ctx).unwrap();
        let freq = report
            .frequencies
            .iter()
            .find(|f| {
                f.link.source == VariableId::from_raw(0)
                    && f.link.target == VariableId::from_raw(1)
                    && f.link.source_lag.raw() == 1
            })
            .map_or(0.0, |f| f.frequency);
        assert!(freq > 0.0, "expected true link to appear; report={:?}", report.frequencies);
    }
}
