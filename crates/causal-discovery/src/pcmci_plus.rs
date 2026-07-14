//! PCMCI+ returning a temporal CPDAG (DESIGN.md §13.4–13.5, Phase 5).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::Arc;

use causal_core::{ExecutionContext, Lag, VariableId};
use causal_data::TimeSeriesData;
use causal_graph::TemporalCpdagReview;
use causal_stats::ConditionalIndependence;

use crate::constraints::DiscoveryConstraints;
use crate::engine::{DiscoveryWorkspace, PcmciEngine};
use crate::error::DiscoveryError;
use crate::evidence::{
    cpdag_evidence_from_oriented, cpdag_from_scored_links, threshold_scored_links,
};
use crate::orientation::{
    MeekR1, MeekR2, MeekR3, MeekR4, OrientCollider, OrientationRule, run_orientation_to_fixed_point,
};
use crate::pipeline::{
    algorithm_record, lagged_node_index, orientation_state_from_sepsets, push_diagnostic,
    with_links_retained,
};
use crate::result::CpdagDiscoveryResult;

/// PCMCI+ discovery: contemporaneous + lagged links → oriented [`causal_graph::TemporalCpdag`].
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

    /// Run PCMCI+ and return a CPDAG-backed discovery result.
    ///
    /// Evidence and review both carry the oriented [`causal_graph::TemporalCpdag`]
    /// (DESIGN.md §13.5); undirected contemporaneous marks are preserved.
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
    ) -> Result<CpdagDiscoveryResult, DiscoveryError> {
        let engine_result = self.engine.run_pc_mci(data, variables, workspace, ctx)?;
        let alpha = self.engine.constraints.alpha;

        let scored = threshold_scored_links(engine_result.evidence.links.to_vec(), self.fdr, alpha);

        let max_lag = self.engine.constraints.temporal.max_lag.raw();
        let mut cpdag = cpdag_from_scored_links(&scored, variables, max_lag)?;

        let node_ids = lagged_node_index(cpdag.nodes());
        let mut state = orientation_state_from_sepsets(&node_ids, &engine_result.sepsets);

        let rules: [&dyn OrientationRule; 5] =
            [&OrientCollider, &MeekR1, &MeekR2, &MeekR3, &MeekR4];
        let _delta = run_orientation_to_fixed_point(&mut cpdag, &rules, &mut state)?;

        let algorithm = algorithm_record(
            "pcmci_plus",
            format!(
                "alpha={},max_lag={},fdr={},min_lag={}",
                alpha,
                self.engine.constraints.temporal.max_lag.raw(),
                self.fdr,
                self.engine.constraints.temporal.min_lag.raw()
            ),
        );
        let evidence = cpdag_evidence_from_oriented(cpdag.clone(), scored, &engine_result.sepsets);
        let review = TemporalCpdagReview::from_cpdag(cpdag, algorithm.id.clone());
        let links_retained = evidence.links.len();
        let mut diagnostics = engine_result.diagnostics;
        push_diagnostic(
            &mut diagnostics,
            "pcmci_plus.cpdag",
            format!(
                "oriented temporal CPDAG with {} nodes ({} directed, {} undirected pending orientation)",
                evidence.graph.node_count(),
                evidence.graph.directed_edge_count(),
                review.pending_undirected.len()
            ),
        );

        Ok(CpdagDiscoveryResult {
            evidence,
            review,
            algorithm,
            assumptions: engine_result.assumptions,
            iterations: engine_result.iterations,
            diagnostics,
            performance: with_links_retained(engine_result.performance, links_retained),
            sepsets: engine_result.sepsets,
        })
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
    fn pcmci_plus_evidence_is_cpdag() {
        let (data, vars) = tiny_xy(200);
        let plus = PcmciPlus::new().with_fdr(false).with_constraints(DiscoveryConstraints {
            temporal: TemporalConstraints {
                max_lag: Lag::from_raw(1),
                min_lag: Lag::CONTEMPORANEOUS,
            },
            alpha: 0.2,
            max_cond_size: 2,
            ..DiscoveryConstraints::default()
        });
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(7);
        let result = plus.run(&data, &vars, &mut ws, &ctx).unwrap();
        assert_eq!(result.algorithm.id.as_ref(), "pcmci_plus");
        assert!(result.evidence.graph.node_count() >= 2);
        // Review tracks the same CPDAG (not a collapsed DAG).
        assert_eq!(result.review.graph.node_count(), result.evidence.graph.node_count());
    }
}
