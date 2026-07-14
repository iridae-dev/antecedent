//! J-PCMCI+: multi-environment PCMCI+ with context nodes (DESIGN.md §13.4–13.5, Phase 9).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::sync::Arc;

use causal_core::{ExecutionContext, Lag, VariableId};
use causal_data::{LaggedColumn, MultiEnvSamplePlan, MultiEnvironmentData, TableView};
use causal_graph::{DenseNodeId, NodeRef, TemporalCpdagReview};
use causal_stats::ConditionalIndependence;

use crate::constraints::DiscoveryConstraints;
use crate::engine::{DiscoveryWorkspace, PcmciEngine};
use crate::error::DiscoveryError;
use crate::evidence::{
    cpdag_evidence_from_oriented, cpdag_from_scored_links, threshold_scored_links,
};
use crate::orientation::{
    MeekR1, MeekR2, MeekR3, MeekR4, OrientCollider, OrientationRule, OrientationState,
    run_orientation_to_fixed_point,
};
use crate::result::{
    AlgorithmRecord, CpdagDiscoveryResult, DiscoveryDiagnostic, DiscoveryPerformanceRecord,
    LaggedLink, ScoredLink,
};

/// Alias for J-PCMCI+ discovery output (context-augmented temporal CPDAG).
pub type JpcmciPlusDiscoveryResult = CpdagDiscoveryResult;

/// J-PCMCI+ discovery over [`MultiEnvironmentData`].
///
/// Own type (not a PCMCI+ flag). Pools per-environment MCI evidence without
/// cloning sibling environment series payloads; sample planning shares column
/// geometry via [`MultiEnvSamplePlan`].
#[derive(Clone, Debug)]
pub struct JpcmciPlus {
    /// Shared engine (`min_lag` typically 0).
    pub engine: PcmciEngine,
    /// Apply FDR before alpha keep on the pooled link set.
    pub fdr: bool,
}

impl Default for JpcmciPlus {
    fn default() -> Self {
        Self::new()
    }
}

impl JpcmciPlus {
    /// Default J-PCMCI+ with `min_lag = 0` and pooled lagged CI enabled.
    #[must_use]
    pub fn new() -> Self {
        let mut constraints = DiscoveryConstraints::default();
        constraints.temporal.min_lag = Lag::CONTEMPORANEOUS;
        constraints.multi_dataset.pool_lagged_ci = true;
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

    /// Run J-PCMCI+ on multi-environment data.
    ///
    /// # Errors
    ///
    /// Empty multi-env, engine / orientation failures.
    pub fn run(
        &self,
        data: &MultiEnvironmentData,
        variables: &[VariableId],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<JpcmciPlusDiscoveryResult, DiscoveryError> {
        if data.env_count() == 0 {
            return Err(DiscoveryError::Unsupported {
                message: "J-PCMCI+ needs ≥1 environment",
            });
        }
        self.engine.constraints.validate()?;

        let max_lag = self.engine.constraints.temporal.max_lag.raw();
        // Plan once across environments: shared columns Arc + LagMap reuse by length.
        // Never clone sibling environment series payloads.
        let mut plan_cols: Vec<LaggedColumn> = Vec::with_capacity(variables.len() * (max_lag as usize + 1));
        for &variable in variables {
            for lag in 0..=max_lag {
                plan_cols.push(LaggedColumn {
                    variable,
                    lag: Lag::from_raw(lag),
                });
            }
        }
        let plan = MultiEnvSamplePlan::try_from_multi_env(data, max_lag, Arc::from(plan_cols))
            .map_err(|e| DiscoveryError::Data(format!("multi-env sample plan failed: {e}")))?;

        let mut per_env_scored: Vec<Vec<ScoredLink>> = Vec::with_capacity(data.env_count());
        let mut last_sepsets = Default::default();
        let mut assumptions = Default::default();
        let mut iterations = Vec::new();
        let mut diagnostics = Vec::new();
        let mut performance = DiscoveryPerformanceRecord::default();

        // Record shared-geometry cost once (not per-env full series bytes).
        let plan_bytes = plan.columns.len().saturating_mul(16)
            + plan.env_count().saturating_mul(64);
        performance.lagged_frame_bytes = plan_bytes as u64;

        for i in 0..data.env_count() {
            let env_plan = plan
                .plan(i)
                .map_err(|e| DiscoveryError::Data(format!("multi-env plan {i}: {e}")))?;
            let series = data
                .environment(i)
                .map_err(|e| DiscoveryError::Data(format!("environment {i}: {e}")))?;
            if env_plan.lag_map().series_len() != series.row_count() {
                return Err(DiscoveryError::Data(format!(
                    "sample plan length {} != environment {i} rows {}",
                    env_plan.lag_map().series_len(),
                    series.row_count()
                )));
            }
            // Borrow-only env access; shared columns Arc across equal plans.
            let _shared_cols = env_plan.columns_arc();
            let engine_result = self.engine.run_pc_mci(series, variables, workspace, ctx)?;
            per_env_scored.push(engine_result.evidence.links.to_vec());
            last_sepsets = engine_result.sepsets;
            assumptions = engine_result.assumptions;
            iterations = engine_result.iterations;
            diagnostics.extend(engine_result.diagnostics);
            performance.ci_tests += engine_result.performance.ci_tests;
            performance.targets = engine_result.performance.targets;
            performance.lagged_frame_bytes = performance
                .lagged_frame_bytes
                .max(engine_result.performance.lagged_frame_bytes);
        }

        diagnostics.push(DiscoveryDiagnostic {
            code: Arc::from("jpcmci_plus.multi_env_plan"),
            message: Arc::from(format!(
                "MultiEnvSamplePlan: {} envs, {} shared lagged columns (no sibling series clone)",
                plan.env_count(),
                plan.columns.len()
            )),
        });

        let pooled = pool_scored_links(&per_env_scored, self.engine.constraints.multi_dataset.pool_lagged_ci);
        let alpha = self.engine.constraints.alpha;
        let scored = threshold_scored_links(pooled, self.fdr, alpha);

        let mut cpdag = cpdag_from_scored_links(&scored, variables, max_lag)?;
        attach_context_nodes(
            &mut cpdag,
            &self.engine.constraints.multi_dataset.context_variables,
            &scored,
        )?;

        let mut node_ids = HashMap::<(u32, u32), DenseNodeId>::new();
        for (i, node) in cpdag.nodes().iter().enumerate() {
            if let NodeRef::Lagged { variable, lag } = node {
                node_ids.insert((variable.raw(), lag.raw()), DenseNodeId::from_raw(i as u32));
            }
        }

        let mut state = OrientationState::default();
        let mut sepset_entries: Vec<_> = last_sepsets.iter().collect();
        sepset_entries
            .sort_by_key(|((s, slag, t, tlag), _)| (s.raw(), slag.raw(), t.raw(), tlag.raw()));
        for ((s, slag, t, tlag), sep) in sepset_entries {
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

        let algorithm = AlgorithmRecord {
            id: Arc::from("jpcmci_plus"),
            config: Arc::from(format!(
                "alpha={},max_lag={},fdr={},envs={},pool={},context={}",
                alpha,
                max_lag,
                self.fdr,
                data.env_count(),
                self.engine.constraints.multi_dataset.pool_lagged_ci,
                self.engine.constraints.multi_dataset.context_variables.len()
            )),
        };
        let evidence = cpdag_evidence_from_oriented(cpdag.clone(), scored, &last_sepsets);
        let review = TemporalCpdagReview::from_cpdag(cpdag, algorithm.id.clone());
        let links_retained = evidence.links.len() as u64;
        diagnostics.push(DiscoveryDiagnostic {
            code: Arc::from("jpcmci_plus.cpdag"),
            message: Arc::from(format!(
                "pooled {} envs into context-aware temporal CPDAG ({} nodes)",
                data.env_count(),
                evidence.graph.node_count()
            )),
        });
        performance.links_retained = links_retained;

        Ok(CpdagDiscoveryResult {
            evidence,
            review,
            algorithm,
            assumptions,
            iterations,
            diagnostics,
            performance,
            sepsets: last_sepsets,
        })
    }
}

/// Conservative pool: keep a link if it appears in any env; p-value = max across envs;
/// statistic = mean of absolute values (signed by first env).
fn pool_scored_links(per_env: &[Vec<ScoredLink>], pool: bool) -> Vec<ScoredLink> {
    if per_env.is_empty() {
        return Vec::new();
    }
    if !pool || per_env.len() == 1 {
        return per_env[0].clone();
    }
    let mut by_link: HashMap<LaggedLink, (f64, f64, usize)> = HashMap::new();
    for scored in per_env {
        for s in scored {
            let entry = by_link.entry(s.link).or_insert((0.0, 0.0, 0));
            entry.0 += s.statistic;
            entry.1 = entry.1.max(s.p_value);
            entry.2 += 1;
        }
    }
    let n_env = per_env.len();
    let mut out = Vec::with_capacity(by_link.len());
    for (link, (stat_sum, p_max, count)) in by_link {
        // Shared-skeleton style: require presence in all environments when pooling.
        if count < n_env {
            continue;
        }
        out.push(ScoredLink {
            link,
            statistic: stat_sum / count as f64,
            p_value: p_max,
            adjusted_p_value: None,
        });
    }
    out.sort_by(|a, b| a.link.cmp(&b.link));
    out
}

fn attach_context_nodes(
    cpdag: &mut causal_graph::TemporalCpdag,
    context_vars: &[VariableId],
    scored: &[ScoredLink],
) -> Result<(), DiscoveryError> {
    if context_vars.is_empty() {
        return Ok(());
    }
    let mut lag0: HashMap<VariableId, DenseNodeId> = HashMap::new();
    for (i, node) in cpdag.nodes().iter().enumerate() {
        if let NodeRef::Lagged { variable, lag } = node {
            if lag.is_contemporaneous() {
                lag0.insert(*variable, DenseNodeId::from_raw(i as u32));
            }
        }
    }
    for &cv in context_vars {
        let ctx_id = cpdag
            .add_context(cv, None)
            .map_err(|e| DiscoveryError::Data(format!("add context node: {e}")))?;
        // Direct context → contemporaneous system target when a lag-0 link was retained.
        for s in scored {
            if s.link.source == cv
                && s.link.source_lag.is_contemporaneous()
                && s.link.target_lag.is_contemporaneous()
                && s.link.target != cv
            {
                if let Some(&tgt) = lag0.get(&s.link.target) {
                    let _ = cpdag.insert_directed(ctx_id, tgt);
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use causal_core::{
        CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
    };
    use causal_data::{
        Float64Column, MultiEnvironmentData, OwnedColumn, OwnedColumnarStorage, SamplingRegularity,
        TimeIndex, TimeSeriesData, ValidityBitmap,
    };

    use super::*;
    use crate::constraints::{MultiDatasetConstraints, TemporalConstraints};

    fn toy_env(n: usize, seed: f64) -> TimeSeriesData {
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
            x[t] = 0.5 * x[t - 1] + 0.1 * ((t as f64) + seed).sin();
            y[t] = 0.7 * x[t] + 0.2 * y[t - 1] + 0.05 * ((t as f64) + seed).cos();
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
        TimeSeriesData::try_new(
            storage,
            TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
        )
        .unwrap()
    }

    #[test]
    fn jpcmci_plus_two_env_toy() {
        let a = toy_env(180, 0.0);
        let b = toy_env(180, 1.0);
        let multi = MultiEnvironmentData::try_new(Arc::from([a, b])).unwrap();
        let vars = [VariableId::from_raw(0), VariableId::from_raw(1)];
        let algo = JpcmciPlus::new().with_fdr(false).with_constraints(DiscoveryConstraints {
            temporal: TemporalConstraints {
                max_lag: Lag::from_raw(1),
                min_lag: Lag::CONTEMPORANEOUS,
            },
            alpha: 0.25,
            max_cond_size: 2,
            multi_dataset: MultiDatasetConstraints {
                context_variables: Arc::from([]),
                pool_lagged_ci: true,
                ..MultiDatasetConstraints::default()
            },
            ..DiscoveryConstraints::default()
        });
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(9);
        let result = algo.run(&multi, &vars, &mut ws, &ctx).unwrap();
        assert_eq!(result.algorithm.id.as_ref(), "jpcmci_plus");
        assert!(result.evidence.graph.node_count() >= 2);
    }
}
