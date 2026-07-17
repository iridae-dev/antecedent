//! J-PCMCI+: multi-environment PCMCI+ with context nodes (DESIGN.md §13.4–13.5).
//!
//! # Current scope (honest limitations)
//!
//! This implementation runs PCMCI **independently per environment**, pools surviving
//! links by **intersection** (`p = max` across envs), merges per-env sepsets, then applies
//! Meek orientation. Context variables listed in constraints are attached as decoration
//! nodes after pooling — they do **not** enter CI tests. The published Günther et al.
//! algorithm (pooled PCMCI+ once with observed context + dataset/time dummies under link
//! assumptions) is not yet implemented.
//!
//! [`MultiEnvSamplePlan`] validates shared lagged-column geometry across environments;
//! each environment still materializes its own lagged frame inside the PCMCI engine.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::sync::Arc;

use causal_core::{AssumptionSet, ExecutionContext, Lag, VariableId};
use causal_data::{LaggedColumn, MultiEnvSamplePlan, MultiEnvironmentData, TableView};
use causal_graph::{DenseNodeId, NodeRef, TemporalCpdagReview};
use causal_stats::{ConditionalIndependence, FdrAdjustment};

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
};
use crate::result::{
    CpdagDiscoveryResult, DiscoveryPerformanceRecord, LaggedLink, PcSepsets, ScoredLink, SepsetKey,
};

/// Alias for J-PCMCI+ discovery output (context-augmented temporal CPDAG).
pub type JpcmciPlusDiscoveryResult = CpdagDiscoveryResult;

/// J-PCMCI+ discovery over [`MultiEnvironmentData`].
///
/// Own type (not a PCMCI+ flag). See module docs for pooling / context limitations.
#[derive(Clone, Debug)]
pub struct JpcmciPlus {
    /// Shared engine (`min_lag` typically 0).
    pub engine: PcmciEngine,
    /// Multiple-testing adjustment on the pooled link set (`None` = off).
    pub fdr: Option<FdrAdjustment>,
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
        Self {
            engine: PcmciEngine::new().with_constraints(constraints),
            fdr: Some(FdrAdjustment::bh()),
        }
    }

    /// Configure constraints (caller should keep `min_lag = 0` for contemporaneous discovery).
    #[must_use]
    pub fn with_constraints(mut self, constraints: DiscoveryConstraints) -> Self {
        self.engine.constraints = constraints;
        self
    }

    /// Enable / disable BH FDR.
    #[must_use]
    pub fn with_fdr(mut self, fdr: bool) -> Self {
        self.fdr = fdr.then(FdrAdjustment::bh);
        self
    }

    /// Full FDR / FWER configuration.
    #[must_use]
    pub fn with_fdr_adjustment(mut self, fdr: Option<FdrAdjustment>) -> Self {
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
    #[allow(clippy::too_many_lines)]
    pub fn run(
        &self,
        data: &MultiEnvironmentData,
        variables: &[VariableId],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<JpcmciPlusDiscoveryResult, DiscoveryError> {
        if data.env_count() == 0 {
            return Err(DiscoveryError::Unsupported { message: "J-PCMCI+ needs ≥1 environment" });
        }
        self.engine.constraints.validate()?;

        let max_lag = self.engine.constraints.temporal.max_lag.raw();
        // Validate shared lagged-column geometry across environments (no sibling series clone).
        let mut plan_cols: Vec<LaggedColumn> =
            Vec::with_capacity(variables.len() * (max_lag as usize + 1));
        for &variable in variables {
            for lag in 0..=max_lag {
                plan_cols.push(LaggedColumn { variable, lag: Lag::from_raw(lag) });
            }
        }
        let plan = MultiEnvSamplePlan::try_from_multi_env(data, max_lag, Arc::from(plan_cols))
            .map_err(|e| DiscoveryError::data_msg(format!("multi-env sample plan failed: {e}")))?;

        let mut per_env_scored: Vec<Vec<ScoredLink>> = Vec::with_capacity(data.env_count());
        let mut per_env_sepsets: Vec<PcSepsets> = Vec::with_capacity(data.env_count());
        let mut assumptions = AssumptionSet::default();
        let mut iterations = Vec::new();
        let mut diagnostics = Vec::new();
        let mut performance = DiscoveryPerformanceRecord::default();

        for i in 0..data.env_count() {
            let env_plan = plan
                .plan(i)
                .map_err(|e| DiscoveryError::data_msg(format!("multi-env plan {i}: {e}")))?;
            let series = data
                .environment(i)
                .map_err(|e| DiscoveryError::data_msg(format!("environment {i}: {e}")))?;
            if env_plan.lag_map().series_len() != series.row_count() {
                return Err(DiscoveryError::data_msg(format!(
                    "sample plan length {} != environment {i} rows {}",
                    env_plan.lag_map().series_len(),
                    series.row_count()
                )));
            }
            let engine_result = self.engine.run_pc_mci(series, variables, workspace, ctx)?;
            per_env_scored.push(engine_result.evidence.links.to_vec());
            per_env_sepsets.push(engine_result.sepsets);
            assumptions = engine_result.assumptions;
            iterations = engine_result.iterations;
            diagnostics.extend(engine_result.diagnostics);
            performance.ci_tests += engine_result.performance.ci_tests;
            performance.targets = engine_result.performance.targets;
            performance.lagged_frame_bytes =
                performance.lagged_frame_bytes.max(engine_result.performance.lagged_frame_bytes);
        }

        diagnostics.push(crate::result::DiscoveryDiagnostic {
            code: Arc::from("jpcmci_plus.multi_env_plan"),
            message: Arc::from(format!(
                "MultiEnvSamplePlan validated {} envs / {} lagged columns; per-env frames built by engine",
                plan.env_count(),
                plan.columns.len()
            )),
        });
        if !self.engine.constraints.multi_dataset.context_variables.is_empty() {
            push_diagnostic(
                &mut diagnostics,
                "jpcmci_plus.context_decoration",
                "context_variables are attached post-hoc and do not enter CI tests (Günther pooled redesign pending)",
            );
        }

        let pooled = pool_scored_links(
            &per_env_scored,
            self.engine.constraints.multi_dataset.pool_lagged_ci,
        );
        let pooled_sepsets = merge_sepsets(&per_env_sepsets);
        let alpha = self.engine.constraints.alpha;
        let scored = threshold_scored_links(pooled, self.fdr, alpha);

        let mut cpdag = cpdag_from_scored_links(&scored, variables, max_lag)?;
        attach_context_nodes(
            &mut cpdag,
            &self.engine.constraints.multi_dataset.context_variables,
            &scored,
        )?;

        let node_ids = lagged_node_index(cpdag.nodes());
        let mut state = orientation_state_from_sepsets(&node_ids, &pooled_sepsets);

        let rules: [&dyn OrientationRule; 5] =
            [&OrientCollider, &MeekR1, &MeekR2, &MeekR3, &MeekR4];
        let delta = run_orientation_to_fixed_point(&mut cpdag, &rules, &mut state)?;

        let algorithm = algorithm_record(
            "jpcmci_plus",
            format!(
                "alpha={},max_lag={},fdr={:?},envs={},pool={},context={}",
                alpha,
                max_lag,
                self.fdr,
                data.env_count(),
                self.engine.constraints.multi_dataset.pool_lagged_ci,
                self.engine.constraints.multi_dataset.context_variables.len()
            ),
        );
        let evidence = cpdag_evidence_from_oriented(cpdag.clone(), scored, &pooled_sepsets);
        let review = TemporalCpdagReview::from_cpdag(cpdag, algorithm.id.clone());
        let links_retained = evidence.links.len() as u64;
        push_diagnostic(
            &mut diagnostics,
            "jpcmci_plus.cpdag",
            format!(
                "pooled {} envs into context-aware temporal CPDAG ({} nodes)",
                data.env_count(),
                evidence.graph.node_count()
            ),
        );
        if state.conflicts > 0 || delta.conflicts > 0 {
            push_diagnostic(
                &mut diagnostics,
                "orientation.conflicts",
                format!(
                    "{} orientation conflict(s) recorded (cycle or opposite direction)",
                    state.conflicts
                ),
            );
        }
        performance.links_retained = links_retained;

        Ok(CpdagDiscoveryResult {
            evidence,
            review,
            algorithm,
            assumptions,
            iterations,
            diagnostics,
            performance,
            sepsets: pooled_sepsets,
        })
    }
}

/// Intersection pool when `pool` is set: keep a link only if it appears in **every** env;
/// p-value = max across envs; statistic = mean across envs.
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

/// Merge per-environment sepsets: keep keys present in every environment; value is the
/// intersection of sepset members (conservative for collider orientation).
fn merge_sepsets(per_env: &[PcSepsets]) -> PcSepsets {
    if per_env.is_empty() {
        return PcSepsets::default();
    }
    if per_env.len() == 1 {
        return per_env[0].clone();
    }
    let mut keys: Vec<SepsetKey> = per_env[0].keys().copied().collect();
    keys.retain(|k| per_env.iter().all(|s| s.contains_key(k)));
    let mut out = PcSepsets::default();
    for key in keys {
        let mut inter: Option<Vec<_>> = None;
        for sep in per_env {
            let members = sep.get(&key).map(|s| s.to_vec()).unwrap_or_default();
            inter = Some(match inter {
                None => members,
                Some(prev) => prev.into_iter().filter(|m| members.contains(m)).collect(),
            });
        }
        if let Some(members) = inter {
            out.insert(key, Arc::from(members));
        }
    }
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
            .map_err(|e| DiscoveryError::data_msg(format!("add context node: {e}")))?;
        for s in scored {
            if s.link.source == cv
                && s.link.source_lag.is_contemporaneous()
                && s.link.target_lag.is_contemporaneous()
                && s.link.target != cv
            {
                if let Some(&tgt) = lag0.get(&s.link.target) {
                    match cpdag.insert_directed(ctx_id, tgt) {
                        Ok(()) => {}
                        Err(
                            causal_graph::GraphError::Cycle { .. }
                            | causal_graph::GraphError::DuplicateEdge { .. },
                        ) => {}
                        Err(e) => return Err(DiscoveryError::from(e)),
                    }
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

    #[test]
    fn merge_sepsets_intersects_members() {
        let key: SepsetKey = (
            VariableId::from_raw(0),
            Lag::from_raw(1),
            VariableId::from_raw(1),
            Lag::CONTEMPORANEOUS,
        );
        let a = (VariableId::from_raw(2), Lag::from_raw(1));
        let b = (VariableId::from_raw(3), Lag::from_raw(1));
        let mut s0 = PcSepsets::default();
        s0.insert(key, Arc::from([a, b]));
        let mut s1 = PcSepsets::default();
        s1.insert(key, Arc::from([a]));
        let merged = merge_sepsets(&[s0, s1]);
        let members = merged.get(&key).unwrap();
        assert_eq!(members.len(), 1);
        assert_eq!(members[0], a);
    }

    #[test]
    fn pool_requires_all_environments() {
        let link = LaggedLink {
            source: VariableId::from_raw(0),
            source_lag: Lag::from_raw(1),
            target: VariableId::from_raw(1),
            target_lag: Lag::CONTEMPORANEOUS,
        };
        let only_first = vec![ScoredLink {
            link,
            statistic: 1.0,
            p_value: 0.01,
            adjusted_p_value: None,
        }];
        let second_empty = [only_first.clone(), Vec::new()];
        assert!(pool_scored_links(&second_empty, true).is_empty());
        let both = [
            only_first,
            vec![ScoredLink {
                link,
                statistic: 2.0,
                p_value: 0.02,
                adjusted_p_value: None,
            }],
        ];
        let pooled = pool_scored_links(&both, true);
        assert_eq!(pooled.len(), 1);
        assert!((pooled[0].statistic - 1.5).abs() < 1e-12);
        assert!((pooled[0].p_value - 0.02).abs() < 1e-12);
    }
}
