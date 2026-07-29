//! PCMCI+ returning a temporal CPDAG.
//!
//! Implements Runge (2020) / pinned baseline `run_pcmciplus`:
//! 1. Lagged-only PC1 skeleton \(\widehat{\mathcal{B}}^-\).
//! 2. Contemporaneous MCI phase with conditioning on contemp neighbors plus lagged parents.
//! 3. Majority collider orientation (sepset subset re-tests) with out-of-band conflicts.
//! 4. Meek R1–R3 restricted to contemporaneous undirected links.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::match_same_arms,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::type_complexity
)]

use std::collections::HashMap;
use std::sync::Arc;

use antecedent_core::{AssumptionSet, ExecutionContext, Lag, VariableId};
use antecedent_data::{LaggedFrame, TimeSeriesData};
use antecedent_graph::{DenseNodeId, MarkedEdge, NodeRef, TemporalCpdagReview};
use antecedent_stats::{ConfidenceMethod, FdrAdjustment};

use crate::combinations::for_each_combination;
use crate::constraints::DiscoveryConstraints;
use crate::engine::{DiscoveryWorkspace, PcmciEngine, mci_conditioning, parents_of_target};
use crate::error::DiscoveryError;
use crate::evidence::{
    cpdag_evidence_from_oriented, cpdag_from_scored_links, symmetrize_contemporaneous_links,
    threshold_scored_links,
};
use crate::orientation::{
    ContempMeekR1, ContempMeekR2, ContempMeekR3, ContempMeekR4, OrientationRule, OrientationState,
    RuleDelta, run_orientation_to_fixed_point, try_orient_undirected,
};
use crate::pcmci_family::pcmci_family_builders;
use crate::pipeline::{
    algorithm_record, lagged_node_index, orientation_state_from_sepsets, push_diagnostic,
    with_links_retained,
};
use crate::result::{
    CpdagDiscoveryResult, DiscoveryDiagnostic, DiscoveryIteration, DiscoveryPerformanceRecord,
    LaggedLink, PcSepsets, ScoredLink,
};

/// PCMCI+ discovery: contemporaneous + lagged links → oriented [`antecedent_graph::TemporalCpdag`].
#[derive(Clone, Debug)]
pub struct PcmciPlus {
    /// Shared engine (`min_lag` typically 0; crate-private — use builders / [`Self::engine`]).
    pub(crate) engine: PcmciEngine,
    /// Multiple-testing adjustment (`None` = off). Contemporaneous links are
    /// excluded from the family by default (pinned baseline).
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

    pcmci_family_builders!();

    /// Run PCMCI+ and return a CPDAG-backed discovery result.
    ///
    /// Evidence and review both carry the oriented [`antecedent_graph::TemporalCpdag`]
    ///; undirected contemporaneous marks are preserved.
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
        let frame = LaggedFrame::from_series(data, variables, frame_depth, &ctx.kernel_policy)
            .map_err(DiscoveryError::from)?;
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
            let plan = antecedent_stats::CiPreparationPlan {
                significance: self.engine.constraints.significance,
                confidence: ConfidenceMethod::default(),
            };
            workspace.prepared_ci =
                Some(self.engine.ci.prepare(&cols, &plan, ctx).map_err(DiscoveryError::from)?);
        }

        // --- Step 1: lagged-only PC1 → B̂⁻ ---
        let (lagged_parents, mut iterations, mut ci_tests, mut sepsets) =
            lagged_pc1_parents(&self.engine, frame, variables, workspace, ctx, threads)?;

        // --- Step 2: contemporaneous MCI phase ---
        let (scored, contemp_sepsets, contemp_tests, truncated) = contemp_mci_phase(
            &self.engine,
            frame,
            variables,
            &compiled,
            &lagged_parents,
            workspace,
            ctx,
            None,
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
            frame,
            &lagged_parents,
            &mut cpdag,
            &mut state,
            workspace,
            ctx,
        )?;

        // --- Step 4: Meek R1–R4 contemporaneous only ---
        let rules: [&dyn OrientationRule<antecedent_graph::TemporalCpdag>; 4] =
            [&ContempMeekR1, &ContempMeekR2, &ContempMeekR3, &ContempMeekR4];
        let meek_delta = run_orientation_to_fixed_point(&mut cpdag, &rules, &mut state)?;

        let algorithm = algorithm_record(
            "pcmci_plus",
            format!(
                "alpha={},max_lag={},fdr={:?},min_lag={},collider=majority,meek=r1-r4-contemp",
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

/// Lagged-only PC1 parent selection (\(\widehat{\mathcal{B}}^-\)).
///
/// # Errors
///
/// Engine / CI failures.
pub(crate) fn lagged_pc1_parents(
    engine: &PcmciEngine,
    frame: &LaggedFrame,
    variables: &[VariableId],
    workspace: &mut DiscoveryWorkspace,
    ctx: &ExecutionContext,
    threads: u32,
) -> Result<
    (Vec<(VariableId, Vec<(VariableId, Lag)>)>, Vec<DiscoveryIteration>, u64, PcSepsets),
    DiscoveryError,
> {
    let mut lagged_constraints = engine.constraints.clone();
    if lagged_constraints.temporal.min_lag.raw() == 0 {
        lagged_constraints.temporal.min_lag = Lag::from_raw(1);
    }
    if lagged_constraints.temporal.min_lag.raw() > lagged_constraints.temporal.max_lag.raw() {
        let empty: Vec<_> = variables.iter().map(|&t| (t, Vec::new())).collect();
        return Ok((empty, Vec::new(), 0, PcSepsets::default()));
    }
    let lagged_engine = PcmciEngine {
        constraints: lagged_constraints,
        ci: Arc::clone(&engine.ci),
        column_blocks: Arc::clone(&engine.column_blocks),
    };
    let lagged_compiled = lagged_engine.constraints.compile(variables)?;
    let (parents, iters, tests) = lagged_engine.select_parents_all(
        frame,
        variables,
        &lagged_compiled,
        workspace,
        ctx,
        threads,
    )?;
    let sep = std::mem::take(&mut workspace.sepsets);
    Ok((parents, iters, tests, sep))
}

/// Contemporaneous + lagged MCI skeleton (Runge 2020 Alg. 2 / pinned baseline `contemp_conds`).
///
/// Initializes adjacencies with \(\widehat{\mathcal{B}}^-\) lagged parents plus all
/// contemporaneous pairs; removes edges by PC1-style tests whose contemporaneous
/// conditioning sets are augmented with lagged parents of both endpoints (MCI).
///
/// When `search` is `Some`, only links for which `search(link)` is true are tested for
/// removal; other adjacencies stay as fixed conditioning parents (J-PCMCI+ phases).
#[allow(clippy::too_many_arguments)]
pub(crate) fn contemp_mci_phase(
    engine: &PcmciEngine,
    frame: &LaggedFrame,
    variables: &[VariableId],
    compiled: &crate::constraints::CompiledConstraints,
    lagged_parents: &[(VariableId, Vec<(VariableId, Lag)>)],
    workspace: &mut DiscoveryWorkspace,
    ctx: &ExecutionContext,
    search: Option<&dyn Fn(LaggedLink) -> bool>,
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
                sb.partial_cmp(&sa)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| (a.0.raw(), a.1.raw()).cmp(&(b.0.raw(), b.1.raw())))
            });

            for pi in 0..order.len() {
                let (src, slag) = order[pi];
                let link = LaggedLink {
                    source: src,
                    source_lag: slag,
                    target,
                    target_lag: Lag::CONTEMPORANEOUS,
                };
                if search.is_some_and(|f| !f(link)) {
                    continue;
                }
                // Contemporaneous conditions only (Alg. 2); lagged MCI parents always added.
                let contemp_others: Vec<(VariableId, Lag)> = order
                    .iter()
                    .enumerate()
                    .filter(|(j, (v, l))| {
                        *j != pi
                            && l.is_contemporaneous()
                            && !(*v == src && slag.is_contemporaneous())
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
    // Fixed (non-search) links are omitted unless they have scores from a prior test.
    let mut scored = Vec::new();
    for (&target, parents) in &adj {
        for &(src, slag) in parents {
            let link = LaggedLink {
                source: src,
                source_lag: slag,
                target,
                target_lag: Lag::CONTEMPORANEOUS,
            };
            if search.is_some_and(|f| !f(link)) {
                continue;
            }
            let Some(&(p, stat)) = scores.get(&(src, slag, target)) else {
                continue;
            };
            scored.push(ScoredLink { link, statistic: stat, p_value: p, adjusted_p_value: None });
        }
    }
    Ok((scored, sepsets, ci_tests, truncated))
}

/// Majority collider orientation with contemporaneous-neighbor subset re-tests.
///
/// Matches pinned baseline `contemp_collider_rule='majority'`. Conflicts / ambiguous triples
/// are recorded out-of-band (`conflict_edges`) and conflicting edges are marked `x-x`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn orient_majority_colliders(
    engine: &PcmciEngine,
    frame: &LaggedFrame,
    lagged_parents: &[(VariableId, Vec<(VariableId, Lag)>)],
    graph: &mut antecedent_graph::TemporalCpdag,
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
        let mut legs: Vec<(DenseNodeId, bool)> = neighbors.iter().map(|&nb| (nb, true)).collect();
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
                        orient_majority_leg(graph, state, &mut delta, a, c, premise)?;
                    }
                    if b_und {
                        let premise = format!(
                            "majority.collider: {}→{}←{} (frac={frac:.2})",
                            a.raw(),
                            c.raw(),
                            b.raw()
                        );
                        orient_majority_leg(graph, state, &mut delta, b, c, premise)?;
                    }
                }
            }
        }
    }
    Ok(delta)
}

/// Orient leg `endpoint → c` if the edge is still undirected.
///
/// `legs` in [`orient_majority_colliders`] is snapshotted once per center `c` before its
/// nested pair loop runs, so a leg flagged undirected there may already have been oriented
/// by an earlier pair in the same loop that shares it. Re-read the edge's current state
/// instead of trusting that stale flag: no-op if already oriented `endpoint → c`
/// (consistent with this collider's conclusion), record a conflict if somehow oriented the
/// opposite way, and otherwise orient it as before.
fn orient_majority_leg(
    graph: &mut antecedent_graph::TemporalCpdag,
    state: &mut OrientationState,
    delta: &mut RuleDelta,
    endpoint: DenseNodeId,
    c: DenseNodeId,
    premise: impl Into<Arc<str>>,
) -> Result<(), DiscoveryError> {
    let current = graph.edge_between(endpoint, c);
    let oriented = current.and_then(MarkedEdge::parent_child);
    if oriented == Some((endpoint, c)) || current.is_some_and(MarkedEdge::is_conflict) {
        // Already oriented endpoint → c by an earlier pair sharing this leg, or already
        // pinned as a conflict by one; either way, nothing to do.
    } else if oriented == Some((c, endpoint)) {
        // Oriented the opposite way — conflict.
        state.record_conflict(delta, endpoint, c, "opposite_direction");
        if graph.mark_conflict(endpoint, c).is_ok() {
            delta.edges_changed += 1;
            delta.fixed_point = false;
        }
    } else {
        let _ = try_orient_undirected(graph, state, delta, endpoint, c, premise)?;
    }
    Ok(())
}

fn is_contemp_node(graph: &antecedent_graph::TemporalCpdag, id: DenseNodeId) -> bool {
    match graph.nodes().get(id.raw() as usize) {
        Some(NodeRef::Lagged { lag, .. }) => lag.is_contemporaneous(),
        Some(NodeRef::Context { .. }) => true,
        _ => false,
    }
}

fn node_var_lag(
    graph: &antecedent_graph::TemporalCpdag,
    id: DenseNodeId,
) -> Option<(VariableId, Lag)> {
    match graph.nodes().get(id.raw() as usize) {
        Some(NodeRef::Lagged { variable, lag }) => Some((*variable, *lag)),
        Some(NodeRef::Context { variable, .. }) => Some((*variable, Lag::CONTEMPORANEOUS)),
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn majority_sep_counts(
    engine: &PcmciEngine,
    frame: &LaggedFrame,
    lagged_parents: &[(VariableId, Vec<(VariableId, Lag)>)],
    graph: &antecedent_graph::TemporalCpdag,
    a: DenseNodeId,
    b: DenseNodeId,
    c: DenseNodeId,
    max_cond: usize,
    alpha: f64,
    workspace: &mut DiscoveryWorkspace,
    ctx: &ExecutionContext,
) -> Result<(u32, u32), DiscoveryError> {
    let (va, la) = node_var_lag(graph, a)
        .ok_or_else(|| DiscoveryError::unsupported("majority collider: missing node a"))?;
    let (vb, lb) = node_var_lag(graph, b)
        .ok_or_else(|| DiscoveryError::unsupported("majority collider: missing node b"))?;
    let (vc, lc) = node_var_lag(graph, c)
        .ok_or_else(|| DiscoveryError::unsupported("majority collider: missing node c"))?;

    // Candidate contemporaneous neighbors of a (excl b) and of b (excl a). `c` itself must
    // remain eligible: the majority vote asks "of the subsets that separate a and b, what
    // fraction contain c?", so excluding c from `cand` would make that fraction always 0.
    // `c` may not turn up via `undirected_neighbors` at all (a leg into c can already be
    // directed), so it is added explicitly below rather than merely un-excluded here.
    let mut cand: Vec<(VariableId, Lag)> = Vec::new();
    for n in graph.undirected_neighbors(a) {
        if n == b {
            continue;
        }
        if let Some((v, l)) = node_var_lag(graph, n) {
            if l.is_contemporaneous() && !cand.contains(&(v, l)) {
                cand.push((v, l));
            }
        }
    }
    for n in graph.undirected_neighbors(b) {
        if n == a {
            continue;
        }
        if let Some((v, l)) = node_var_lag(graph, n) {
            if l.is_contemporaneous() && !cand.contains(&(v, l)) {
                cand.push((v, l));
            }
        }
    }
    if lc.is_contemporaneous() && !cand.contains(&(vc, lc)) {
        cand.push((vc, lc));
    }

    let mut n_sep = 0u32;
    let mut n_with_c = 0u32;
    let c_key = (vc, lc);
    let mut scratch = Vec::new();
    for q in 0..=max_cond.min(cand.len()) {
        for_each_combination(&cand, q, &mut scratch, |s| {
            // Build MCI-style Z = S ∪ lagged parents.
            let link = LaggedLink { source: va, source_lag: la, target: vb, target_lag: lb };
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
            let result = engine.ci_statistic(frame, va, la, vb, lb, &cond, workspace, ctx);
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
    use antecedent_core::{
        CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
    };
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
        TimeSeriesData, ValidityBitmap,
    };

    use antecedent_graph::TemporalCpdag;
    use antecedent_stats::{
        CiBatchRequest, CiBatchResult, CiResult, CiWorkspace, ConditionalIndependence,
        PreparedCiTest, StatsError,
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
        assert!(result.algorithm.config.as_ref().contains("meek=r1-r4-contemp"));
    }

    /// CI test double whose verdict depends only on whether `c_col` is present in the
    /// tested conditioning set — never on the data. Drives the majority-rule vote in
    /// [`majority_sep_counts`] deterministically in both directions:
    /// - `independent_iff_c_present = true`: `a ⫫ b | S` iff `c ∈ S` — chain/fork ground
    ///   truth (`a→c→b` or `a←c→b`): dependent marginally, independent once you condition
    ///   on `c`.
    /// - `independent_iff_c_present = false`: `a ⫫ b | S` iff `c ∉ S` — collider ground
    ///   truth (`a→c←b`): independent marginally, dependent once you condition on the
    ///   collider `c`.
    ///
    /// [`antecedent_stats::OracleCi`] can't exercise this: it decides purely from the
    /// `(x, y)` column pair and ignores the conditioning set, so every subset would return
    /// the same verdict and the vote would stay degenerate regardless of the fix.
    struct SepGivenC {
        c_col: usize,
        independent_iff_c_present: bool,
    }

    impl ConditionalIndependence for SepGivenC {
        fn test_batch(
            &self,
            prepared: &PreparedCiTest,
            request: &CiBatchRequest<'_>,
            _workspace: &mut CiWorkspace,
            _ctx: &ExecutionContext,
        ) -> Result<CiBatchResult, StatsError> {
            prepared.ensure_compatible(request)?;
            let request = &prepared.bind_request(request);
            let results = request
                .queries
                .iter()
                .map(|q| {
                    let cond = &request.z_flat[q.z_start..q.z_start + q.z_len];
                    let has_c = cond.contains(&self.c_col);
                    let independent = has_c == self.independent_iff_c_present;
                    CiResult {
                        statistic: if independent { 0.0 } else { 1.0 },
                        p_value: if independent { 1.0 } else { 0.0 },
                        df: 0.0,
                        ci: None,
                    }
                })
                .collect();
            Ok(CiBatchResult { results })
        }
    }

    /// Three contemporaneous variables `a`, `c`, `b`. Values are arbitrary — [`SepGivenC`]
    /// decides purely from which columns are in the conditioning set, never from the data.
    fn tiny_abc(n: usize) -> (TimeSeriesData, Vec<VariableId>) {
        let mut b = CausalSchemaBuilder::new();
        for name in ["a", "c", "b"] {
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
        let cols = (0..3u32)
            .map(|i| {
                let vals: Vec<f64> = (0..n).map(|t| t as f64 + f64::from(i)).collect();
                OwnedColumn::Float64(
                    Float64Column::new(
                        VariableId::from_raw(i),
                        Arc::from(vals),
                        ValidityBitmap::all_valid(n),
                    )
                    .unwrap(),
                )
            })
            .collect();
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let data = TimeSeriesData::try_new(
            storage,
            TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
        )
        .unwrap();
        (data, vec![VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2)])
    }

    /// Unshielded triple `a — c — b` (no `a — b` edge) as a bare graph, plus the
    /// `DenseNodeId`s for `a`, `c`, `b` in that order.
    fn unshielded_triple() -> (TemporalCpdag, DenseNodeId, DenseNodeId, DenseNodeId) {
        let mut graph = TemporalCpdag::empty();
        let a = graph.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
        let c = graph.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        let b = graph.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
        graph.insert_undirected(a, c).unwrap();
        graph.insert_undirected(c, b).unwrap();
        (graph, a, c, b)
    }

    /// Regression for "the majority rule never votes": `cand` used to exclude `c` from
    /// every candidate conditioning subset, so `n_with_c` was always 0 and `frac` could
    /// never land anywhere but < 0.5. With `c` correctly eligible, the vote swings both
    /// ways depending on whether the true separating set contains `c`.
    #[test]
    fn majority_sep_counts_votes_both_ways() {
        let ctx = ExecutionContext::for_tests(1);
        let (data, vars) = tiny_abc(5);
        let frame = LaggedFrame::from_series(&data, &vars, 0, &ctx.kernel_policy).unwrap();
        let c_col = frame.column_index(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        let (graph, a, c, b) = unshielded_triple();

        // Collider ground truth: only the empty set separates a, b, and it never contains
        // c ⇒ vote must conclude collider (frac < 0.5).
        let collider_engine = PcmciEngine::new()
            .with_ci(Arc::new(SepGivenC { c_col, independent_iff_c_present: false }));
        let mut ws = DiscoveryWorkspace::default();
        let (n_sep, n_with_c) = majority_sep_counts(
            &collider_engine,
            &frame,
            &[],
            &graph,
            a,
            b,
            c,
            1,
            0.05,
            &mut ws,
            &ctx,
        )
        .unwrap();
        assert_eq!((n_sep, n_with_c), (1, 0), "collider case: c must never appear in a sepset");
        assert!(f64::from(n_with_c) / f64::from(n_sep) < 0.5);

        // Chain/fork ground truth: only {c} separates a, b ⇒ vote must conclude
        // non-collider (frac > 0.5) — unreachable before the fix.
        let chain_engine = PcmciEngine::new()
            .with_ci(Arc::new(SepGivenC { c_col, independent_iff_c_present: true }));
        let mut ws = DiscoveryWorkspace::default();
        let (n_sep, n_with_c) = majority_sep_counts(
            &chain_engine,
            &frame,
            &[],
            &graph,
            a,
            b,
            c,
            1,
            0.05,
            &mut ws,
            &ctx,
        )
        .unwrap();
        assert_eq!((n_sep, n_with_c), (1, 1), "chain/fork case: every sepset must contain c");
        assert!(f64::from(n_with_c) / f64::from(n_sep) > 0.5);
    }

    /// End-to-end: a true chain/fork `a — c — b` must NOT be oriented as a collider by
    /// [`orient_majority_colliders`]. Before the fix this always fired as a collider,
    /// since `frac` was stuck at 0.0 regardless of ground truth.
    #[test]
    fn majority_vote_leaves_chain_undirected() {
        let ctx = ExecutionContext::for_tests(1);
        let (data, vars) = tiny_abc(5);
        let frame = LaggedFrame::from_series(&data, &vars, 0, &ctx.kernel_policy).unwrap();
        let c_col = frame.column_index(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        let (mut graph, a, c, b) = unshielded_triple();
        let engine = PcmciEngine::new()
            .with_ci(Arc::new(SepGivenC { c_col, independent_iff_c_present: true }))
            .with_constraints(DiscoveryConstraints {
                alpha: 0.05,
                max_cond_size: 1,
                ..DiscoveryConstraints::default()
            });
        let mut state = OrientationState::default();
        let mut ws = DiscoveryWorkspace::default();
        orient_majority_colliders(&engine, &frame, &[], &mut graph, &mut state, &mut ws, &ctx)
            .unwrap();

        assert!(
            graph.edge_between(a, c).unwrap().parent_child().is_none(),
            "a—c must stay undirected"
        );
        assert!(
            graph.edge_between(b, c).unwrap().parent_child().is_none(),
            "b—c must stay undirected"
        );
    }

    /// End-to-end: a true collider `a → c ← b` IS oriented as a collider.
    #[test]
    fn majority_vote_orients_true_collider() {
        let ctx = ExecutionContext::for_tests(1);
        let (data, vars) = tiny_abc(5);
        let frame = LaggedFrame::from_series(&data, &vars, 0, &ctx.kernel_policy).unwrap();
        let c_col = frame.column_index(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        let (mut graph, a, c, b) = unshielded_triple();
        let engine = PcmciEngine::new()
            .with_ci(Arc::new(SepGivenC { c_col, independent_iff_c_present: false }))
            .with_constraints(DiscoveryConstraints {
                alpha: 0.05,
                max_cond_size: 1,
                ..DiscoveryConstraints::default()
            });
        let mut state = OrientationState::default();
        let mut ws = DiscoveryWorkspace::default();
        orient_majority_colliders(&engine, &frame, &[], &mut graph, &mut state, &mut ws, &ctx)
            .unwrap();

        assert_eq!(graph.edge_between(a, c).unwrap().parent_child(), Some((a, c)));
        assert_eq!(graph.edge_between(b, c).unwrap().parent_child(), Some((b, c)));
    }
}
