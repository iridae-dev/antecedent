//! PCMCI+ returning a temporal CPDAG (DESIGN.md §13.4–13.5, Phase 5).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::sync::Arc;

use causal_core::{ExecutionContext, Lag, VariableId};
use causal_data::TimeSeriesData;
use causal_graph::{DenseNodeId, TemporalCpdag, TemporalGraphReview};
use causal_stats::ConditionalIndependence;

use crate::constraints::DiscoveryConstraints;
use crate::engine::{DiscoveryWorkspace, PcmciEngine};
use crate::error::DiscoveryError;
use crate::evidence::{graph_evidence_from_scored, threshold_scored_links};
use crate::orientation::{
    MeekR1, MeekR2, MeekR3, MeekR4, OrientCollider, OrientationRule, OrientationState,
    run_orientation_to_fixed_point,
};
use crate::result::{AlgorithmRecord, DiscoveryDiagnostic, DiscoveryResult};

/// PCMCI+ discovery: contemporaneous + lagged links → oriented [`TemporalCpdag`].
#[derive(Clone, Debug)]
pub struct PcmciPlus {
    /// Shared engine (`min_lag` typically 0).
    pub engine: PcmciEngine,
    /// Apply FDR before alpha keep.
    pub fdr: bool,
}

impl Default for PcmciPlus {
    fn default() -> Self {
        Self::new()
    }
}

impl PcmciPlus {
    /// Default PCMCI+ with `min_lag = 0`.
    #[must_use]
    pub fn new() -> Self {
        let mut constraints = DiscoveryConstraints::default();
        constraints.temporal.min_lag = Lag::CONTEMPORANEOUS;
        Self { engine: PcmciEngine::new().with_constraints(constraints), fdr: true }
    }

    /// Configure constraints (caller should keep `min_lag = 0` for contemporaneous discovery).
    #[must_use]
    pub fn with_constraints(mut self, constraints: DiscoveryConstraints) -> Self {
        self.engine.constraints = constraints;
        self
    }

    /// Enable / disable FDR.
    #[must_use]
    pub fn with_fdr(mut self, fdr: bool) -> Self {
        self.fdr = fdr;
        self
    }

    /// Replace the CI test on the shared engine.
    #[must_use]
    pub fn with_ci(mut self, ci: Arc<dyn ConditionalIndependence + Send + Sync>) -> Self {
        self.engine = self.engine.with_ci(ci);
        self
    }

    /// Run PCMCI+ and return discovery result plus oriented [`TemporalCpdag`].
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
    ) -> Result<(DiscoveryResult, TemporalCpdag), DiscoveryError> {
        let mut result = self.engine.run_pc_mci(data, variables, workspace, ctx)?;
        let alpha = self.engine.constraints.alpha;

        let scored = threshold_scored_links(
            result.evidence.links.iter().copied().collect(),
            self.fdr,
            alpha,
        );

        let mut cpdag = TemporalCpdag::empty();
        let max_lag = self.engine.constraints.temporal.max_lag.raw();
        let mut node_ids = HashMap::<(u32, u32), DenseNodeId>::new();
        for &v in variables {
            for lag in 0..=max_lag {
                let id = cpdag
                    .add_lagged(v, Lag::from_raw(lag))
                    .map_err(|e| DiscoveryError::Data(e.to_string()))?;
                node_ids.insert((v.raw(), lag), id);
            }
        }

        let mut state = OrientationState::default();
        for link in &scored {
            let Some(&src) = node_ids.get(&(link.link.source.raw(), link.link.source_lag.raw()))
            else {
                continue;
            };
            let Some(&tgt) = node_ids.get(&(link.link.target.raw(), link.link.target_lag.raw()))
            else {
                continue;
            };
            if link.link.source_lag.is_contemporaneous()
                && link.link.target_lag.is_contemporaneous()
            {
                if !cpdag.has_edge(src, tgt) {
                    cpdag
                        .insert_undirected(src, tgt)
                        .map_err(|e| DiscoveryError::Data(e.to_string()))?;
                }
            } else if !cpdag.has_edge(src, tgt) {
                cpdag.insert_directed(src, tgt).map_err(|e| DiscoveryError::Data(e.to_string()))?;
            }
        }

        for ((s, slag, t, tlag), sep) in &result.sepsets {
            let Some(&sa) = node_ids.get(&(s.raw(), slag.raw())) else {
                continue;
            };
            let Some(&tb) = node_ids.get(&(t.raw(), tlag.raw())) else {
                continue;
            };
            let mapped: Vec<DenseNodeId> = sep
                .iter()
                .filter_map(|(v, l)| node_ids.get(&(v.raw(), l.raw())).copied())
                .collect();
            state.set_sepset(sa, tb, Arc::from(mapped));
        }

        let rules: [&dyn OrientationRule; 5] =
            [&OrientCollider, &MeekR1, &MeekR2, &MeekR3, &MeekR4];
        let _delta = run_orientation_to_fixed_point(&mut cpdag, &rules, &mut state)?;

        result.algorithm = AlgorithmRecord {
            id: Arc::from("pcmci_plus"),
            config: Arc::from(format!(
                "alpha={},max_lag={},fdr={},min_lag={}",
                alpha,
                self.engine.constraints.temporal.max_lag.raw(),
                self.fdr,
                self.engine.constraints.temporal.min_lag.raw()
            )),
        };
        result.evidence = graph_evidence_from_scored(scored)?;
        result.review = TemporalGraphReview::from_graph(
            result.evidence.graph.clone(),
            result.algorithm.id.clone(),
        );
        result.performance.links_retained = result.evidence.links.len() as u64;
        result.diagnostics.push(DiscoveryDiagnostic {
            code: Arc::from("pcmci_plus.cpdag"),
            message: Arc::from(format!(
                "oriented temporal CPDAG with {} nodes",
                cpdag.node_count()
            )),
        });

        Ok((result, cpdag))
    }
}

#[cfg(test)]
mod tests {
    use causal_core::{
        CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
    };
    use causal_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
        TimeSeriesData, ValidityBitmap,
    };

    use super::*;
    use crate::constraints::TemporalConstraints;

    fn toy_series() -> (TimeSeriesData, Vec<VariableId>) {
        let n = 400usize;
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
            x[t] = ((t as f64) * 0.01).sin();
            y[t] = 0.8 * x[t - 1] + 0.3 * x[t] + 0.01 * ((t as f64) * 0.03).cos();
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
        let series = TimeSeriesData::try_new(
            storage,
            TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
        )
        .unwrap();
        (series, vec![VariableId::from_raw(0), VariableId::from_raw(1)])
    }

    #[test]
    fn pcmci_plus_returns_cpdag() {
        let (series, vars) = toy_series();
        let plus = PcmciPlus::new().with_fdr(false).with_constraints(DiscoveryConstraints {
            temporal: TemporalConstraints {
                max_lag: Lag::from_raw(1),
                min_lag: Lag::CONTEMPORANEOUS,
            },
            alpha: 0.05,
            max_cond_size: 2,
            ..DiscoveryConstraints::default()
        });
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let (res, cpdag) = plus.run(&series, &vars, &mut ws, &ctx).unwrap();
        assert_eq!(res.algorithm.id.as_ref(), "pcmci_plus");
        assert!(cpdag.node_count() >= 2);
        assert!(!res.evidence.links.is_empty());
    }
}
