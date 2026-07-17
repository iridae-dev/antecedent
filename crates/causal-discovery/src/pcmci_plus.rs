//! PCMCI+ returning a temporal CPDAG (DESIGN.md §13.4–13.5).
//!
//! Implements Runge (2020) / tigramite `run_pcmciplus`:
//! 1. Lagged-only PC1 skeleton \(\widehat{\mathcal{B}}^-\).
//! 2. Contemporaneous MCI phase with conditioning on contemp neighbors plus lagged parents.
//! 3. Majority collider orientation (sepset subset re-tests) with out-of-band conflicts.
//! 4. Meek R1–R3 restricted to contemporaneous undirected links.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::too_many_lines
)]

use std::collections::HashMap;
use std::sync::Arc;

use causal_core::{AssumptionSet, ExecutionContext, Lag, VariableId};
use causal_data::{LaggedFrame, TimeSeriesData};
use causal_graph::{DenseNodeId, NodeRef, TemporalCpdagReview};
use causal_stats::{ConfidenceMethod, ConditionalIndependence, FdrAdjustment};

use crate::combinations::for_each_combination;
use crate::constraints::DiscoveryConstraints;
use crate::engine::{
    DiscoveryWorkspace, PcmciEngine, mci_conditioning, parents_of_target,
};
use crate::error::DiscoveryError;
use crate::evidence::{
    cpdag_evidence_from_oriented, cpdag_from_scored_links, symmetrize_contemporaneous_links,
    threshold_scored_links,
};
use crate::orientation::{
    ContempMeekR1, ContempMeekR2, ContempMeekR3, OrientationRule, OrientationState, RuleDelta,
    run_orientation_to_fixed_point, try_orient_undirected,
};
use crate::pipeline::{
    algorithm_record, lagged_node_index, orientation_state_from_sepsets, push_diagnostic,
    with_links_retained,
};
use crate::result::{
    CpdagDiscoveryResult, DiscoveryDiagnostic, DiscoveryIteration, DiscoveryPerformanceRecord,
    LaggedLink, PcSepsets, ScoredLink,
};

/// PCMCI+ discovery: contemporaneous + lagged links → oriented [`causal_graph::TemporalCpdag`].
#[derive(Clone, Debug)]
pub struct PcmciPlus {
    /// Shared engine (`min_lag` typically 0).
    pub engine: PcmciEngine,
    /// Multiple-testing adjustment (`None` = off). Contemporaneous links are
    /// excluded from the family by default (tigramite).
    pub fdr: Option<FdrAdjustment>,
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

    /// Enable / disable BH FDR (excludes contemporaneous by default).
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
        let max_lag = self.engine.constraints.temporal.max_lag.raw();
        let frame_depth = 2 * max_lag;
        let frame =
            LaggedFrame::from_series(data, variables, frame_depth).map_err(DiscoveryError::from)?;
        self.run_on_frame(&frame, variables, workspace, ctx)
    }

    /// Run PCMCI+ on a pre-built (optionally row-filtered) lagged frame.
    ///
    /// # Errors
    ///
    /// Engine / orientation failures.
    pub fn run_on_frame(
        &self,
        frame: &LaggedFrame,
        variables: &[VariableId],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<CpdagDiscoveryResult, DiscoveryError> {
        let alpha = self.engine.constraints.alpha;
        let max_lag = self.engine.constraints.temporal.max_lag.raw();
        if let Some(hard) = ctx.memory.hard_limit_bytes {
            if frame.values_bytes() > hard {
                return Err(DiscoveryError::Unsupported {
                    message: "lagged frames exceed ExecutionContext memory hard limit",
                });
            }
        }

        let threads = ctx.parallelism.max_threads.get().max(1);
        let compiled = self.engine.constraints.compile(variables)?;
        {
            let cols: Vec<&[f64]> = (0..frame.ncols()).map(|i| frame.column(i)).collect();
            let plan = causal_stats::CiPreparationPlan {
                significance: self.engine.constraints.significance,
                confidence: ConfidenceMethod::default(),
            };
            workspace.prepared_ci = Some(
                self.engine.ci.prepare(&cols, &plan, ctx).map_err(DiscoveryError::from)?,
            );
        }

        // --- Step 1: lagged-only PC1 → B̂⁻ ---
        let mut lagged_constraints = self.engine.constraints.clone();
        // Lagged phase never includes τ=0; bump min_lag to at least 1.
        if lagged_constraints.temporal.min_lag.raw() == 0 {
            lagged_constraints.temporal.min_lag = Lag::from_raw(1);
        }
        let (lagged_parents, mut iterations, mut ci_tests, mut sepsets) =
            if lagged_constraints.temporal.min_lag.raw()
                > lagged_constraints.temporal.max_lag.raw()
            {
                // max_lag = 0: no lagged candidates.
                let empty: Vec<_> = variables.iter().map(|&t| (t, Vec::new())).collect();
                (empty, Vec::new(), 0u64, PcSepsets::default())
            } else {
                let lagged_engine = PcmciEngine {
                    constraints: lagged_constraints,
                    ci: Arc::clone(&self.engine.ci),
                };
                let lagged_compiled = lagged_engine.constraints.compile(variables)?;
                let (parents, iters, tests) = lagged_engine.select_parents_all(
                    &frame,
                    variables,
                    &lagged_compiled,
                    workspace,
                    ctx,
                    threads,
                )?;
                let sep = std::mem::take(&mut workspace.sepsets);
                (parents, iters, tests, sep)
            };

        // --- Step 2: contemporaneous MCI phase ---
        let (scored, contemp_sepsets, contemp_tests, truncated) = contemp_mci_phase(
            &self.engine,
            &frame,
            variables,
            &compiled,
            &lagged_parents,
            workspace,
            ctx,
        )?;
        ci_tests += contemp_tests;
        iterations.push(DiscoveryIteration {
            label: Arc::from("pcmci_plus.contemp_mci"),
            ci_tests: contemp_tests,
        });
        for (k, v) in contemp_sepsets {
            sepsets.insert(k, v);
        }

        let scored = threshold_scored_links(scored, self.fdr, alpha);
        let scored = symmetrize_contemporaneous_links(scored);

        let mut cpdag = cpdag_from_scored_links(&scored, variables, max_lag)?;
        let node_ids = lagged_node_index(cpdag.nodes());
        let mut state = orientation_state_from_sepsets(&node_ids, &sepsets);

        // --- Step 3: majority collider (with subset re-tests) ---
        let majority_delta = orient_majority_colliders(
            &self.engine,
            &frame,
            &lagged_parents,
            &mut cpdag,
            &mut state,
            workspace,
            ctx,
        )?;

        // --- Step 4: Meek R1–R3 contemporaneous only ---
        let rules: [&dyn OrientationRule; 3] =
            [&ContempMeekR1, &ContempMeekR2, &ContempMeekR3];
        let meek_delta = run_orientation_to_fixed_point(&mut cpdag, &rules, &mut state)?;

        let algorithm = algorithm_record(
            "pcmci_plus",
            format!(
                "alpha={},max_lag={},fdr={:?},min_lag={},collider=majority,meek=r1-r3-contemp",
                alpha,
                max_lag,
                self.fdr,
                self.engine.constraints.temporal.min_lag.raw()
            ),
        );
        let evidence = cpdag_evidence_from_oriented(cpdag.clone(), scored, &sepsets);
        let review = TemporalCpdagReview::from_cpdag(cpdag, algorithm.id.clone());
        let links_retained = evidence.links.len();
        let mut diagnostics = Vec::new();
        if truncated > 0 {
            diagnostics.push(DiscoveryDiagnostic {
                code: Arc::from("mci.conditioning_truncated"),
                message: Arc::from(format!(
                    "MCI conditioning sets dropped {truncated} weakest condition(s) at the column cap"
                )),
            });
        }
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
        let conflicts = state.conflicts + majority_delta.conflicts + meek_delta.conflicts;
        if conflicts > 0 {
            push_diagnostic(
                &mut diagnostics,
                "orientation.conflicts",
                format!(
                    "{conflicts} orientation conflict(s) recorded (cycle, opposite direction, or ambiguous majority); edges left unmarked where conflicting"
                ),
            );
        }

        Ok(CpdagDiscoveryResult {
            evidence,
            review,
            algorithm,
            assumptions: AssumptionSet::new(),
            iterations,
            diagnostics,
            performance: with_links_retained(
                DiscoveryPerformanceRecord {
                    ci_tests,
                    links_retained: 0,
                    targets: variables.len() as u64,
                    lagged_frame_bytes: frame.values_bytes(),
                    worker_threads: threads,
                },
                links_retained,
            ),
            sepsets,
        })
    }
}

type AdjMap = HashMap<VariableId, Vec<(VariableId, Lag)>>;
type ScoreMap = HashMap<(VariableId, Lag, VariableId), (f64, f64)>;

/// Contemporaneous + lagged MCI skeleton (Runge 2020 Alg. 2 / tigramite `contemp_conds`).
///
/// Initializes adjacencies with \(\widehat{\mathcal{B}}^-\) lagged parents plus all
/// contemporaneous pairs; removes edges by PC1-style tests whose contemporaneous
/// conditioning sets are augmented with lagged parents of both endpoints (MCI).
fn contemp_mci_phase(
    engine: &PcmciEngine,
    frame: &LaggedFrame,
    variables: &[VariableId],
    compiled: &crate::constraints::CompiledConstraints,
    lagged_parents: &[(VariableId, Vec<(VariableId, Lag)>)],
    workspace: &mut DiscoveryWorkspace,
    ctx: &ExecutionContext,
) -> Result<(Vec<ScoredLink>, PcSepsets, u64, u64), DiscoveryError> {
    let alpha = engine.constraints.alpha;
    let max_cond = engine.constraints.max_cond_size;
    let mut adj: AdjMap = HashMap::new();
    for &t in variables {
        let mut parents = parents_of_target(lagged_parents, t).to_vec();
        for &v in variables {
            if v == t {
                continue;
            }
            let link = LaggedLink {
                source: v,
                source_lag: Lag::CONTEMPORANEOUS,
                target: t,
                target_lag: Lag::CONTEMPORANEOUS,
            };
            if compiled.allows(link) && !parents.contains(&(v, Lag::CONTEMPORANEOUS)) {
                parents.push((v, Lag::CONTEMPORANEOUS));
            }
        }
        adj.insert(t, parents);
    }

    let mut scores: ScoreMap = HashMap::new();
    let mut sepsets: PcSepsets = HashMap::new();
    let mut ci_tests = 0u64;
    let mut truncated = 0u64;
    let mut min_stat: HashMap<(VariableId, VariableId, Lag), f64> = HashMap::new();

    for cond_size in 0..=max_cond {
        let mut removed: Vec<(VariableId, VariableId, Lag)> = Vec::new();
        let targets: Vec<VariableId> = variables.to_vec();
        for &target in &targets {
            let Some(parents) = adj.get(&target).cloned() else {
                continue;
            };
            if parents.is_empty() || parents.len() <= cond_size {
                continue;
            }
            // Rank by descending |stat| for PC1 strongest-q selection.
            let mut order = parents.clone();
            order.sort_by(|a, b| {
                let sa = min_stat.get(&(target, a.0, a.1)).copied().unwrap_or(f64::INFINITY);
                let sb = min_stat.get(&(target, b.0, b.1)).copied().unwrap_or(f64::INFINITY);
                sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal).then_with(|| {
                    (a.0.raw(), a.1.raw()).cmp(&(b.0.raw(), b.1.raw()))
                })
            });

            for pi in 0..order.len() {
                let (src, slag) = order[pi];
                let link = LaggedLink {
                    source: src,
                    source_lag: slag,
                    target,
                    target_lag: Lag::CONTEMPORANEOUS,
                };
                // Contemporaneous conditions only (Alg. 2); lagged MCI parents always added.
                let contemp_others: Vec<(VariableId, Lag)> = order
                    .iter()
                    .enumerate()
                    .filter(|(j, (v, l))| {
                        *j != pi && l.is_contemporaneous() && !(*v == src && slag.is_contemporaneous())
                    })
                    .map(|(_, x)| *x)
                    .take(cond_size)
                    .collect();

                let lagged_tgt = parents_of_target(lagged_parents, target);
                let lagged_src = parents_of_target(lagged_parents, src);
                truncated += mci_conditioning(link, lagged_tgt, lagged_src, &mut workspace.others);
                // Prepend contemporaneous S (not already present).
                for &c in &contemp_others {
                    if !workspace.others.contains(&c) {
                        workspace.others.insert(0, c);
                    }
                }
                // Cap again after inserting S.
                if workspace.others.len() > 30 {
                    let drop = workspace.others.len() - 30;
                    workspace.others.truncate(30);
                    truncated += drop as u64;
                }

                let cond = std::mem::take(&mut workspace.others);
                let result = engine.ci_statistic(
                    frame,
                    src,
                    slag,
                    target,
                    Lag::CONTEMPORANEOUS,
                    &cond,
                    workspace,
                    ctx,
                );
                workspace.others = cond;
                let (stat, p) = result?;
                ci_tests += 1;
                let key_stat = (target, src, slag);
                let prev = min_stat.get(&key_stat).copied().unwrap_or(f64::INFINITY);
                min_stat.insert(key_stat, prev.min(stat.abs()));

                let sk = (src, slag, target);
                let entry = scores.entry(sk).or_insert((0.0, 0.0));
                if p >= entry.0 {
                    *entry = (p, stat);
                }

                if p > alpha && !compiled.requires(link) {
                    removed.push((target, src, slag));
                    sepsets.insert(
                        (src, slag, target, Lag::CONTEMPORANEOUS),
                        Arc::from(contemp_others.clone().into_boxed_slice()),
                    );
                }
            }
        }
        for (target, src, slag) in removed {
            if let Some(list) = adj.get_mut(&target) {
                list.retain(|&p| p != (src, slag));
            }
        }
        let any_left = adj.values().any(|p| p.len() > cond_size);
        if !any_left {
            break;
        }
    }

    // Emit surviving adjacencies (conservative p = max over tests).
    let mut scored = Vec::new();
    for (&target, parents) in &adj {
        for &(src, slag) in parents {
            let Some(&(p, stat)) = scores.get(&(src, slag, target)) else {
                continue;
            };
            scored.push(ScoredLink {
                link: LaggedLink {
                    source: src,
                    source_lag: slag,
                    target,
                    target_lag: Lag::CONTEMPORANEOUS,
                },
                statistic: stat,
                p_value: p,
                adjusted_p_value: None,
            });
        }
    }
    Ok((scored, sepsets, ci_tests, truncated))
}

/// Majority collider orientation with contemporaneous-neighbor subset re-tests.
///
/// Matches tigramite `contemp_collider_rule='majority'`. Conflicts / ambiguous triples
/// are recorded out-of-band (`conflict_edges`); `x-x` Endpoint marks remain deferred.
#[allow(clippy::too_many_arguments)]
fn orient_majority_colliders(
    engine: &PcmciEngine,
    frame: &LaggedFrame,
    lagged_parents: &[(VariableId, Vec<(VariableId, Lag)>)],
    graph: &mut causal_graph::TemporalCpdag,
    state: &mut OrientationState,
    workspace: &mut DiscoveryWorkspace,
    ctx: &ExecutionContext,
) -> Result<RuleDelta, DiscoveryError> {
    let alpha = engine.constraints.alpha;
    let max_cond = engine.constraints.max_cond_size;
    let mut delta = RuleDelta { fixed_point: true, ..RuleDelta::default() };
    let n = graph.node_count();

    let mut contemp_nodes = Vec::new();
    for i in 0..n {
        let id = DenseNodeId::from_raw(i as u32);
        if is_contemp_node(graph, id) {
            contemp_nodes.push(id);
        }
    }

    for &c in &contemp_nodes {
        let neighbors: Vec<DenseNodeId> = graph
            .undirected_neighbors(c)
            .into_iter()
            .filter(|&nb| is_contemp_node(graph, nb))
            .collect();
        let mut legs: Vec<(DenseNodeId, bool)> =
            neighbors.iter().map(|&nb| (nb, true)).collect();
        for p in graph.parents(c) {
            if !legs.iter().any(|(x, _)| *x == p) {
                legs.push((p, false));
            }
        }

        for i in 0..legs.len() {
            for j in (i + 1)..legs.len() {
                let (a, a_und) = legs[i];
                let (b, b_und) = legs[j];
                if !a_und && !b_und {
                    continue;
                }
                if graph.has_edge(a, b) {
                    continue;
                }
                let (n_sep, n_with_c) = majority_sep_counts(
                    engine,
                    frame,
                    lagged_parents,
                    graph,
                    a,
                    b,
                    c,
                    max_cond,
                    alpha,
                    workspace,
                    ctx,
                )?;
                if n_sep == 0 {
                    state.record_conflict(&mut delta, a, b, "ambiguous_majority");
                    continue;
                }
                let frac = f64::from(n_with_c) / f64::from(n_sep);
                if (frac - 0.5).abs() < f64::EPSILON {
                    state.record_conflict(&mut delta, a, b, "ambiguous_majority");
                    continue;
                }
                if frac < 0.5 {
                    // Collider at c.
                    if a_und {
                        let premise = format!(
                            "majority.collider: {}→{}←{} (frac={frac:.2})",
                            a.raw(),
                            c.raw(),
                            b.raw()
                        );
                        let _ = try_orient_undirected(graph, state, &mut delta, a, c, premise)?;
                    }
                    if b_und {
                        let premise = format!(
                            "majority.collider: {}→{}←{} (frac={frac:.2})",
                            a.raw(),
                            c.raw(),
                            b.raw()
                        );
                        let _ = try_orient_undirected(graph, state, &mut delta, b, c, premise)?;
                    }
                }
            }
        }
    }
    Ok(delta)
}

fn is_contemp_node(graph: &causal_graph::TemporalCpdag, id: DenseNodeId) -> bool {
    match graph.nodes().get(id.raw() as usize) {
        Some(NodeRef::Lagged { lag, .. }) => lag.is_contemporaneous(),
        _ => false,
    }
}

fn node_var_lag(graph: &causal_graph::TemporalCpdag, id: DenseNodeId) -> Option<(VariableId, Lag)> {
    match graph.nodes().get(id.raw() as usize) {
        Some(NodeRef::Lagged { variable, lag }) => Some((*variable, *lag)),
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn majority_sep_counts(
    engine: &PcmciEngine,
    frame: &LaggedFrame,
    lagged_parents: &[(VariableId, Vec<(VariableId, Lag)>)],
    graph: &causal_graph::TemporalCpdag,
    a: DenseNodeId,
    b: DenseNodeId,
    c: DenseNodeId,
    max_cond: usize,
    alpha: f64,
    workspace: &mut DiscoveryWorkspace,
    ctx: &ExecutionContext,
) -> Result<(u32, u32), DiscoveryError> {
    let (va, la) = node_var_lag(graph, a).ok_or_else(|| {
        DiscoveryError::stats_msg("majority collider: missing node a")
    })?;
    let (vb, lb) = node_var_lag(graph, b).ok_or_else(|| {
        DiscoveryError::stats_msg("majority collider: missing node b")
    })?;
    let (vc, lc) = node_var_lag(graph, c).ok_or_else(|| {
        DiscoveryError::stats_msg("majority collider: missing node c")
    })?;

    // Candidate contemporaneous neighbors of a (excl b) and of b (excl a).
    let mut cand: Vec<(VariableId, Lag)> = Vec::new();
    for n in graph.undirected_neighbors(a) {
        if n == b || n == c {
            continue;
        }
        if let Some((v, l)) = node_var_lag(graph, n) {
            if l.is_contemporaneous() && !cand.contains(&(v, l)) {
                cand.push((v, l));
            }
        }
    }
    for n in graph.undirected_neighbors(b) {
        if n == a || n == c {
            continue;
        }
        if let Some((v, l)) = node_var_lag(graph, n) {
            if l.is_contemporaneous() && !cand.contains(&(v, l)) {
                cand.push((v, l));
            }
        }
    }

    let mut n_sep = 0u32;
    let mut n_with_c = 0u32;
    let c_key = (vc, lc);
    let mut scratch = Vec::new();
    for q in 0..=max_cond.min(cand.len()) {
        for_each_combination(&cand, q, &mut scratch, |s| {
            // Build MCI-style Z = S ∪ lagged parents.
            let link = LaggedLink {
                source: va,
                source_lag: la,
                target: vb,
                target_lag: lb,
            };
            let _ = mci_conditioning(
                link,
                parents_of_target(lagged_parents, vb),
                parents_of_target(lagged_parents, va),
                &mut workspace.others,
            );
            for &x in s {
                if !workspace.others.contains(&x) {
                    workspace.others.push(x);
                }
            }
            let cond = std::mem::take(&mut workspace.others);
            let result = engine.ci_statistic(
                frame,
                va,
                la,
                vb,
                lb,
                &cond,
                workspace,
                ctx,
            );
            workspace.others = cond;
            match result {
                Ok((_, p)) if p > alpha => {
                    n_sep = n_sep.saturating_add(1);
                    if s.contains(&c_key) {
                        n_with_c = n_with_c.saturating_add(1);
                    }
                }
                Ok(_) => {}
                Err(_) => {}
            }
            true
        });
    }
    Ok((n_sep, n_with_c))
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
        assert_eq!(result.review.graph.node_count(), result.evidence.graph.node_count());
        assert!(result.algorithm.config.as_ref().contains("collider=majority"));
        assert!(result.algorithm.config.as_ref().contains("meek=r1-r3-contemp"));
    }
}
