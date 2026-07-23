//! LPCMCI discovery returning a temporal PAG.
//!
//! Implements Gerhardus & Runge (2020): middle marks, weakly-minimal sepsets,
//! interleaved ancestral / non-ancestral removal with orientation (Alg. 1).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use antecedent_core::{ExecutionContext, Lag, VariableId};
use antecedent_data::TimeSeriesData;
use antecedent_stats::FdrAdjustment;

use crate::constraints::DiscoveryConstraints;
use crate::engine::{DiscoveryWorkspace, PcmciEngine};
use crate::error::DiscoveryError;
use crate::lpcmci_phases::run_lpcmci_algorithm;
use crate::pcmci_family::pcmci_family_builders;
use crate::result::PagDiscoveryResult;

/// LPCMCI: latent-confounder-aware PCMCI → oriented [`antecedent_graph::TemporalPag`].
#[derive(Clone, Debug)]
pub struct Lpcmci {
    /// Shared PCMCI engine (`min_lag` typically 0; crate-private — use builders / [`Self::engine`]).
    pub(crate) engine: PcmciEngine,
    /// Multiple-testing adjustment (`None` = off).
    pub fdr: Option<FdrAdjustment>,
    /// Preliminary Alg-S2 iterations before the final ancestral/non-ancestral pass
    /// (pinned baseline `n_preliminary_iterations`, default 1).
    pub n_preliminary_iterations: u32,
}

impl Default for Lpcmci {
    fn default() -> Self {
        Self::new()
    }
}

impl Lpcmci {
    /// Default LPCMCI with `min_lag = 0`.
    #[must_use]
    pub fn new() -> Self {
        let mut constraints = DiscoveryConstraints::default();
        constraints.temporal.min_lag = Lag::CONTEMPORANEOUS;
        Self {
            engine: PcmciEngine::new().with_constraints(constraints),
            fdr: Some(FdrAdjustment::bh()),
            n_preliminary_iterations: 1,
        }
    }

    pcmci_family_builders!();

    /// Number of preliminary ancestral phases (pinned baseline default: 1).
    #[must_use]
    pub fn with_n_preliminary_iterations(mut self, n: u32) -> Self {
        self.n_preliminary_iterations = n;
        self
    }

    /// Run LPCMCI and return a PAG-backed discovery result.
    ///
    /// # Errors
    ///
    /// Engine / orientation failures.
    pub fn run(
        &self,
        data: &TimeSeriesData,
        variables: &[VariableId],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<PagDiscoveryResult, DiscoveryError> {
        run_lpcmci_algorithm(
            &self.engine,
            data,
            variables,
            workspace,
            ctx,
            self.fdr,
            self.n_preliminary_iterations,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use antecedent_core::{
        CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
    };
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
        TimeSeriesData, ValidityBitmap,
    };

    use super::*;
    use crate::constraints::TemporalConstraints;

    fn tiny_xy(n: usize) -> (TimeSeriesData, Vec<VariableId>) {
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
            x[t] = 0.5 * x[t - 1] + 0.1 * (t as f64).sin();
            y[t] = 0.7 * x[t] + 0.2 * y[t - 1] + 0.05 * (t as f64).cos();
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
    fn lpcmci_returns_temporal_pag() {
        let (data, vars) = tiny_xy(80);
        let alg = Lpcmci::new().with_fdr(false).with_constraints(DiscoveryConstraints {
            temporal: TemporalConstraints {
                max_lag: Lag::from_raw(1),
                min_lag: Lag::CONTEMPORANEOUS,
            },
            alpha: 0.2,
            max_cond_size: 2,
            ..DiscoveryConstraints::default()
        });
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let result = alg.run(&data, &vars, &mut ws, &ctx).unwrap();
        assert_eq!(result.algorithm.id.as_ref(), "lpcmci");
        assert!(result.evidence.graph.node_count() > 0);
        assert!(matches!(result.evidence.source, crate::result::EvidenceSource::Discovery { .. }));
    }
}
