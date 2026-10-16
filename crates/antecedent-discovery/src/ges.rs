//! Greedy Equivalence Search (GES) → static [`Cpdag`] (Chickering 2002).
//!
//! Forward Insert / turning Reverse / backward Delete on the CPDAG, scored with
//! Gaussian BIC via [`antecedent_state::LocalScoreCache`]. Optional PC-skeleton
//! screening restricts Insert candidates.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use antecedent_core::{ExecutionContext, Lag, VariableId};
use antecedent_data::TabularData;
use antecedent_graph::{Cpdag, CpdagReview, Dag, DenseNodeId, Endpoint, NodeRef};
use antecedent_state::{GraphScoreData, GraphScoreFamily, LocalScoreCache};
use antecedent_stats::{ConditionalIndependence, FdrAdjustment, PartialCorrelation};

use crate::constraints::{DiscoveryConstraints, KnowledgeScope};
use crate::engine::DiscoveryWorkspace;
use crate::error::DiscoveryError;
use crate::orientation::{
    MeekR1, MeekR2, MeekR3, MeekR4, OrientationRule, OrientationState,
    run_static_orientation_to_fixed_point,
};
use crate::pc::{Pc, StaticCpdagDiscoveryResult, collect_float_columns};
use crate::result::{
    DiscoveryDiagnostic, DiscoveryIteration, DiscoveryPerformanceRecord, DiscoveryResult,
    EdgeEvidence, EvidenceSource, GraphEvidence, LaggedLink, ScoredLink, discovery_assumptions,
};

/// Chickering GES over tabular data → CPDAG.
#[derive(Clone)]
pub struct Ges {
    /// Constraints / alpha (alpha used only when PC screening is on).
    pub constraints: DiscoveryConstraints,
    /// CI test for optional PC skeleton screening.
    pub ci: Arc<dyn ConditionalIndependence + Send + Sync>,
    /// FDR for PC screening (`None` = off).
    pub fdr: Option<FdrAdjustment>,
    /// Soft PC-skeleton screening for Insert candidates.
    pub screen_pc: bool,
    /// Cap on T/H subset enumeration during Insert/Delete/Reverse (`None` → 12).
    pub max_subset: Option<usize>,
}

crate::debug_with_ci!(Ges, screen_pc, max_subset);

impl Default for Ges {
    fn default() -> Self {
        Self::new()
    }
}

impl Ges {
    /// Default GES (Gaussian BIC; no PC screening).
    #[must_use]
    pub fn new() -> Self {
        Self {
            constraints: DiscoveryConstraints {
                temporal: crate::constraints::TemporalConstraints {
                    max_lag: Lag::CONTEMPORANEOUS,
                    min_lag: Lag::CONTEMPORANEOUS,
                },
                ..DiscoveryConstraints::default()
            },
            ci: Arc::new(PartialCorrelation),
            fdr: Some(FdrAdjustment::bh().with_exclude_contemporaneous(false)),
            screen_pc: false,
            max_subset: None,
        }
    }

    /// Configure constraints.
    #[must_use]
    pub fn with_constraints(mut self, constraints: DiscoveryConstraints) -> Self {
        self.constraints = constraints;
        self
    }

    /// Enable / disable BH FDR for PC screening.
    #[must_use]
    pub fn with_fdr(mut self, fdr: bool) -> Self {
        self.fdr = fdr.then(|| FdrAdjustment::bh().with_exclude_contemporaneous(false));
        self
    }

    /// Full FDR configuration (PC screening).
    #[must_use]
    pub fn with_fdr_adjustment(mut self, fdr: Option<FdrAdjustment>) -> Self {
        self.fdr = fdr;
        self
    }

    /// Replace the CI test used by PC screening.
    #[must_use]
    pub fn with_ci(mut self, ci: Arc<dyn ConditionalIndependence + Send + Sync>) -> Self {
        self.ci = ci;
        self
    }

    /// Soft PC-skeleton screening for Insert (and reverse) candidates.
    #[must_use]
    pub fn with_pc_screening(mut self, screen: bool) -> Self {
        self.screen_pc = screen;
        self
    }

    /// Cap T/H subset enumeration (`None` uses default 12).
    #[must_use]
    pub fn with_max_subset(mut self, max_subset: Option<usize>) -> Self {
        self.max_subset = max_subset;
        self
    }

    /// Run GES.
    ///
    /// # Errors
    ///
    /// Data, score, graph, or screening failures.
    #[allow(
        clippy::too_many_lines,
        reason = "one algorithm pipeline over shared mutable graph state; the phases are marked in place"
    )]
    pub fn run(
        &self,
        data: &TabularData,
        variables: &[VariableId],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<StaticCpdagDiscoveryResult, DiscoveryError> {
        self.constraints.validate()?;
        self.constraints
            .check_background_knowledge(KnowledgeScope::ScoreSearch, variables)
            .map_err(DiscoveryError::Knowledge)?;
        if self.screen_pc {
            crate::ci::ensure_ci_decisions_meaningful(
                &*self.ci,
                self.constraints.significance,
                self.constraints.alpha,
                self.fdr.is_some(),
            )?;
        }
        if variables.is_empty() {
            return Err(DiscoveryError::Unsupported {
                message: "GES requires at least one variable",
            });
        }

        let col_owned = collect_float_columns(data, variables)?;
        let n = col_owned[0].len();
        if n < 2 {
            return Err(DiscoveryError::stats_msg("insufficient rows for GES"));
        }
        for c in &col_owned {
            if c.len() != n {
                return Err(DiscoveryError::data_msg("column length mismatch"));
            }
        }

        let n_vars = variables.len();
        let mut flat = Vec::with_capacity(n_vars.saturating_mul(n));
        for c in &col_owned {
            flat.extend_from_slice(c.as_ref());
        }
        let score_data = GraphScoreData::new(n, n_vars, Arc::from(flat))?;
        let mut cache = LocalScoreCache::new(crate::graph_score::graph_score_cache_key(
            GraphScoreFamily::GaussianBic,
            n_vars,
            n,
        ));
        let mut deltas = GesDeltaCache::default();

        let screen: Option<HashSet<(u32, u32)>> = if self.screen_pc {
            Some(pc_skeleton_pairs(self, data, variables, workspace, ctx)?)
        } else {
            None
        };

        let mut cpdag = empty_cpdag(variables)?;
        seed_required_edges(&mut cpdag, variables, &self.constraints)?;

        let max_parents = self.constraints.max_parents.unwrap_or(n_vars.saturating_sub(1));
        let max_subset = max_parents.min(self.max_subset.unwrap_or(12));
        let search = GesSearch {
            data: &score_data,
            max_parents,
            max_subset,
            screen: screen.as_ref(),
            constraints: &self.constraints,
            variables,
        };

        // Forward equivalence search (Insert).
        loop {
            if ctx.cancellation.is_cancelled() {
                return Err(DiscoveryError::Cancelled);
            }
            let Some(best) = best_insert(&cpdag, search, &mut cache, &mut deltas)? else {
                break;
            };
            if best.delta <= 0.0 {
                break;
            }
            apply_insert(&mut cpdag, best.x, best.y, &best.set)?;
        }

        // Turning phase (Reverse = Delete then Insert opposite).
        loop {
            if ctx.cancellation.is_cancelled() {
                return Err(DiscoveryError::Cancelled);
            }
            let Some(best) = best_reverse(&cpdag, search, &mut cache, &mut deltas)? else {
                break;
            };
            if best.delta <= 0.0 {
                break;
            }
            apply_delete(&mut cpdag, best.x, best.y, &best.h)?;
            apply_insert(&mut cpdag, best.y, best.x, &best.t)?;
        }

        // Backward equivalence search (Delete).
        loop {
            if ctx.cancellation.is_cancelled() {
                return Err(DiscoveryError::Cancelled);
            }
            let Some(best) = best_delete(&cpdag, search, &mut cache, &mut deltas)? else {
                break;
            };
            if best.delta <= 0.0 {
                break;
            }
            apply_delete(&mut cpdag, best.x, best.y, &best.set)?;
        }

        // Background knowledge orients what the score leaves undirected: the search forbids an
        // insertion against a forbidden direction, but an equivalence class can still contain an
        // edge whose direction knowledge fixes. Meek's rules then propagate the orientations.
        let mut diagnostics = Vec::new();
        if self.constraints.has_directional_knowledge() {
            let var_index: HashMap<VariableId, usize> =
                variables.iter().enumerate().map(|(i, v)| (*v, i)).collect();
            let dense_of = |v: VariableId| crate::pipeline::dense_of(&var_index, v);
            let mut state = OrientationState::default();
            crate::pc::orient_fixed_directions(
                &mut cpdag,
                &self.constraints,
                variables,
                &dense_of,
                &mut state,
            )?;
            let rules: [&dyn OrientationRule<Cpdag>; 4] = [&MeekR1, &MeekR2, &MeekR3, &MeekR4];
            run_static_orientation_to_fixed_point(&mut cpdag, &rules, &mut state)?;
            if state.conflicts > 0 {
                diagnostics.push(DiscoveryDiagnostic {
                    code: Arc::from("ges.orientation_conflict"),
                    message: Arc::from(format!(
                        "{} edge(s) have a direction the score search compelled against the \
                         background knowledge, or knowledge that closes a directed cycle; they \
                         are marked as conflicts",
                        state.conflicts
                    )),
                });
            }
        }

        let total_score = {
            // Sync cache parents from a consistent DAG extension for reporting.
            let dag = pdag_to_dag(&cpdag)?;
            for node in 0..crate::indexing::dense_u32(n_vars) {
                let d = DenseNodeId::from_raw(node);
                let parents: Vec<u32> = dag.parents(d).iter().map(|p| p.raw()).collect();
                let _ = cache.local_score(&score_data, node, &Arc::from(parents))?;
            }
            cache.score_graph(&score_data)?
        };

        let edge_evidence: Vec<EdgeEvidence> = cpdag
            .edges()
            .into_iter()
            .filter_map(|e| {
                let (a, b) = if e.a.raw() <= e.b.raw() { (e.a, e.b) } else { (e.b, e.a) };
                let va = cpdag.variable_id(a)?;
                let vb = cpdag.variable_id(b)?;
                Some(EdgeEvidence {
                    link: LaggedLink {
                        source: va,
                        source_lag: Lag::CONTEMPORANEOUS,
                        target: vb,
                        target_lag: Lag::CONTEMPORANEOUS,
                    },
                    statistic: None,
                    p_value: None,
                    adjusted_p_value: None,
                    interval: None,
                    separating_set: None,
                    provenance: Arc::from([Arc::from("ges")]),
                })
            })
            .collect();

        let links: Vec<ScoredLink> = edge_evidence
            .iter()
            .map(|e| ScoredLink {
                link: e.link,
                // GES scores graphs, not edges: there is no per-edge test statistic.
                statistic: f64::NAN,
                p_value: f64::NAN,
                adjusted_p_value: None,
            })
            .collect();

        let evidence = GraphEvidence {
            graph: cpdag.clone(),
            edge_evidence: Arc::from(edge_evidence),
            links: Arc::from(links),
            source: EvidenceSource::Discovery { algorithm: Arc::from("ges") },
        };
        let review = CpdagReview::from_cpdag(cpdag, "ges");
        let edge_count = evidence.graph.edges().len();

        Ok(DiscoveryResult {
            evidence,
            review,
            algorithm: crate::pipeline::algorithm_record(
                "ges",
                format!(
                    "family=gaussian_bic screen_pc={} score={total_score:.6}{}",
                    self.screen_pc,
                    self.constraints.knowledge_config()
                ),
            ),
            assumptions: discovery_assumptions("ges", true),
            iterations: Vec::<DiscoveryIteration>::new(),
            diagnostics,
            performance: DiscoveryPerformanceRecord {
                ci_tests: 0,
                links_retained: u64::try_from(edge_count).unwrap_or(u64::MAX),
                targets: u64::try_from(n_vars).unwrap_or(u64::MAX),
                lagged_frame_bytes: 0,
                worker_threads: 1,
            },
            sepsets: crate::result::PcSepsets::default(),
        })
    }
}

#[derive(Clone, Debug)]
struct OpCand {
    x: DenseNodeId,
    y: DenseNodeId,
    set: Vec<DenseNodeId>,
    delta: f64,
}

#[derive(Clone, Debug)]
struct ReverseCand {
    x: DenseNodeId,
    y: DenseNodeId,
    h: Vec<DenseNodeId>,
    t: Vec<DenseNodeId>,
    delta: f64,
}

fn empty_cpdag(variables: &[VariableId]) -> Result<Cpdag, DiscoveryError> {
    let mut cpdag = Cpdag::empty();
    for &v in variables {
        cpdag.add_node(NodeRef::Static(v))?;
    }
    Ok(cpdag)
}

fn seed_required_edges(
    cpdag: &mut Cpdag,
    variables: &[VariableId],
    constraints: &DiscoveryConstraints,
) -> Result<(), DiscoveryError> {
    let var_set: HashSet<VariableId> = variables.iter().copied().collect();
    let index: HashMap<VariableId, DenseNodeId> = variables
        .iter()
        .enumerate()
        .map(|(i, v)| (*v, DenseNodeId::from_raw(crate::indexing::dense_u32(i))))
        .collect();
    for r in constraints.required.iter() {
        if r.source_lag != Lag::CONTEMPORANEOUS || r.target_lag != Lag::CONTEMPORANEOUS {
            continue;
        }
        if !var_set.contains(&r.source) || !var_set.contains(&r.target) {
            continue;
        }
        let a = index[&r.source];
        let b = index[&r.target];
        if !cpdag.has_edge(a, b) {
            cpdag.insert_undirected(a, b)?;
        }
    }
    Ok(())
}

fn pc_skeleton_pairs(
    ges: &Ges,
    data: &TabularData,
    variables: &[VariableId],
    workspace: &mut DiscoveryWorkspace,
    ctx: &ExecutionContext,
) -> Result<HashSet<(u32, u32)>, DiscoveryError> {
    // The screen only reads the uncapped skeleton; GES enforces the limit in its own search.
    let mut screening = ges.constraints.clone();
    screening.max_parents = None;
    let pc = Pc::new()
        .with_constraints(screening)
        .with_fdr_adjustment(ges.fdr)
        .with_ci(Arc::clone(&ges.ci));
    let result = pc.run(data, variables, workspace, ctx)?;
    let mut pairs = HashSet::new();
    for e in result.evidence.graph.edges() {
        let lo = e.a.raw().min(e.b.raw());
        let hi = e.a.raw().max(e.b.raw());
        pairs.insert((lo, hi));
    }
    Ok(pairs)
}

fn screened(screen: Option<&HashSet<(u32, u32)>>, a: DenseNodeId, b: DenseNodeId) -> bool {
    let Some(s) = screen else {
        return true;
    };
    let lo = a.raw().min(b.raw());
    let hi = a.raw().max(b.raw());
    s.contains(&(lo, hi))
}

/// Whether an operator that makes `from` a parent of `to` is forbidden by the background
/// knowledge. A variable outside the search is forbidden.
fn forbidden_direction(
    constraints: &DiscoveryConstraints,
    variables: &[VariableId],
    from: DenseNodeId,
    to: DenseNodeId,
) -> bool {
    let (Some(&vf), Some(&vt)) = (variables.get(from.as_usize()), variables.get(to.as_usize()))
    else {
        return true;
    };
    constraints.is_forbidden(LaggedLink {
        source: vf,
        source_lag: Lag::CONTEMPORANEOUS,
        target: vt,
        target_lag: Lag::CONTEMPORANEOUS,
    })
}

fn required_pair(
    constraints: &DiscoveryConstraints,
    variables: &[VariableId],
    a: DenseNodeId,
    b: DenseNodeId,
) -> bool {
    let Some(va) = variables.get(a.as_usize()).copied() else {
        return false;
    };
    let Some(vb) = variables.get(b.as_usize()).copied() else {
        return false;
    };
    let link_ab = LaggedLink {
        source: va,
        source_lag: Lag::CONTEMPORANEOUS,
        target: vb,
        target_lag: Lag::CONTEMPORANEOUS,
    };
    let link_ba = LaggedLink {
        source: vb,
        source_lag: Lag::CONTEMPORANEOUS,
        target: va,
        target_lag: Lag::CONTEMPORANEOUS,
    };
    constraints.is_required(link_ab) || constraints.is_required(link_ba)
}

fn na_yx(cpdag: &Cpdag, y: DenseNodeId, x: DenseNodeId) -> Vec<DenseNodeId> {
    let mut out: Vec<_> =
        cpdag.undirected_neighbors(y).into_iter().filter(|&n| cpdag.has_edge(n, x)).collect();
    out.sort_unstable_by_key(|node| node.raw());
    out
}

fn is_clique(cpdag: &Cpdag, nodes: &[DenseNodeId]) -> bool {
    for i in 0..nodes.len() {
        for j in i + 1..nodes.len() {
            if !cpdag.has_edge(nodes[i], nodes[j]) {
                return false;
            }
        }
    }
    true
}

/// Semi-directed reachability: undirected either way; directed only forward.
fn semi_directed_reaches_avoiding(
    cpdag: &Cpdag,
    from: DenseNodeId,
    to: DenseNodeId,
    avoid: &HashSet<DenseNodeId>,
) -> bool {
    if from == to {
        return true;
    }
    let n = cpdag.node_count();
    let mut seen = vec![false; n];
    let mut q = VecDeque::new();
    q.push_back(from);
    seen[from.as_usize()] = true;
    while let Some(u) = q.pop_front() {
        for v in cpdag.children(u) {
            if avoid.contains(&v) && v != to {
                continue;
            }
            if v == to {
                return true;
            }
            if !seen[v.as_usize()] {
                seen[v.as_usize()] = true;
                q.push_back(v);
            }
        }
        for v in cpdag.undirected_neighbors(u) {
            if avoid.contains(&v) && v != to {
                continue;
            }
            if v == to {
                return true;
            }
            if !seen[v.as_usize()] {
                seen[v.as_usize()] = true;
                q.push_back(v);
            }
        }
    }
    false
}

fn insert_valid(cpdag: &Cpdag, x: DenseNodeId, y: DenseNodeId, t: &[DenseNodeId]) -> bool {
    if cpdag.has_edge(x, y) || x == y {
        return false;
    }
    let na = na_yx(cpdag, y, x);
    let mut clique_nodes = na.clone();
    clique_nodes.extend_from_slice(t);
    if !is_clique(cpdag, &clique_nodes) {
        return false;
    }
    let avoid: HashSet<DenseNodeId> = clique_nodes.into_iter().collect();
    // Every semi-directed path Y ↝ X hits NA∪T ⇔ no path avoiding NA∪T.
    !semi_directed_reaches_avoiding(cpdag, y, x, &avoid)
}

fn delete_valid(cpdag: &Cpdag, x: DenseNodeId, y: DenseNodeId, h: &[DenseNodeId]) -> bool {
    let Some(e) = cpdag.edge_between(x, y) else {
        return false;
    };
    // Edge must be X→Y or X—Y (not Y→X alone as X,Y ordered for operator).
    let xy_directed = matches!(
        (e.at_a, e.at_b),
        (Endpoint::Tail, Endpoint::Arrow) | (Endpoint::Arrow, Endpoint::Tail)
    ) && e.parent_child() == Some((x, y));
    let undirected = e.is_undirected();
    if !xy_directed && !undirected {
        return false;
    }
    let na = na_yx(cpdag, y, x);
    let h_set: HashSet<DenseNodeId> = h.iter().copied().collect();
    if h.iter().any(|n| !na.contains(n)) {
        return false;
    }
    let rest: Vec<DenseNodeId> = na.into_iter().filter(|n| !h_set.contains(n)).collect();
    is_clique(cpdag, &rest)
}

/// Parent set that `insert_delta` actually scores: `Pa(Y) ∪ NA(Y,X) ∪ T ∪ {X}`.
///
/// Counting only `Pa` and `T` lets neighbours in `NA(Y,X)` push a node past
/// `max_parents` while the operator is still accepted.
fn insert_scored_parents(
    cpdag: &Cpdag,
    x: DenseNodeId,
    y: DenseNodeId,
    t: &[DenseNodeId],
) -> usize {
    let na = na_yx(cpdag, y, x);
    let mut base = parents_u32(cpdag, y);
    for n in na.iter().chain(t.iter()) {
        base.push(n.raw());
    }
    base.push(x.raw());
    base.sort_unstable();
    base.dedup();
    base.len()
}

fn parents_u32(cpdag: &Cpdag, y: DenseNodeId) -> Vec<u32> {
    let mut out: Vec<_> =
        cpdag.parents(y).into_iter().map(antecedent_graph::DenseNodeId::raw).collect();
    out.sort_unstable();
    out
}

fn insert_delta(
    cpdag: &Cpdag,
    cache: &mut LocalScoreCache,
    deltas: &mut GesDeltaCache,
    data: &GraphScoreData,
    x: DenseNodeId,
    y: DenseNodeId,
    t: &[DenseNodeId],
) -> Result<f64, DiscoveryError> {
    // Corollary 16: s(Y, NA∪T∪Pa∪{X}) − s(Y, NA∪T∪Pa)
    let na = na_yx(cpdag, y, x);
    let pa = parents_u32(cpdag, y);
    let mut base: Vec<u32> = pa;
    for n in na.iter().chain(t.iter()) {
        base.push(n.raw());
    }
    base.sort_unstable();
    base.dedup();
    let mut neu = base.clone();
    neu.push(x.raw());
    neu.sort_unstable();
    neu.dedup();
    cached_operator_delta(cache, deltas, data, y.raw(), &Arc::from(base), &Arc::from(neu))
}

fn delete_delta(
    cpdag: &Cpdag,
    cache: &mut LocalScoreCache,
    deltas: &mut GesDeltaCache,
    data: &GraphScoreData,
    x: DenseNodeId,
    y: DenseNodeId,
    h: &[DenseNodeId],
) -> Result<f64, DiscoveryError> {
    // Corollary 18: s(Y, (NA\H) ∪ (Pa\{X})) − s(Y, (NA\H) ∪ Pa)
    let na = na_yx(cpdag, y, x);
    let h_set: HashSet<DenseNodeId> = h.iter().copied().collect();
    let na_rest: Vec<u32> = na
        .into_iter()
        .filter(|n| !h_set.contains(n))
        .map(antecedent_graph::DenseNodeId::raw)
        .collect();
    let mut pa = parents_u32(cpdag, y);
    let mut with_x = pa.clone();
    with_x.push(x.raw());
    with_x.extend_from_slice(&na_rest);
    with_x.sort_unstable();
    with_x.dedup();
    pa.retain(|&p| p != x.raw());
    let mut without_x = pa;
    without_x.extend_from_slice(&na_rest);
    without_x.sort_unstable();
    without_x.dedup();
    cached_operator_delta(cache, deltas, data, y.raw(), &Arc::from(with_x), &Arc::from(without_x))
}

/// Exact cache key for a GES operator delta.
type GesDeltaKey = (u32, Arc<[u32]>, Arc<[u32]>);

/// Operator deltas keyed by the exact `(node, old parents, new parents)` pair that was scored.
#[derive(Default)]
struct GesDeltaCache {
    entries: HashMap<GesDeltaKey, f64>,
}

fn cached_operator_delta(
    cache: &mut LocalScoreCache,
    deltas: &mut GesDeltaCache,
    data: &GraphScoreData,
    node: u32,
    old_parents: &Arc<[u32]>,
    new_parents: &Arc<[u32]>,
) -> Result<f64, DiscoveryError> {
    let key = (node, Arc::clone(old_parents), Arc::clone(new_parents));
    if let Some(delta) = deltas.entries.get(&key).copied() {
        return Ok(delta);
    }
    let old = local_score_or_degenerate(cache, data, node, old_parents)?;
    let new = local_score_or_degenerate(cache, data, node, new_parents)?;
    let delta = operator_delta(old, new);
    deltas.entries.insert(key, delta);
    Ok(delta)
}

/// A zero-residual (deterministic) local score is not a finite BIC. Treat that parent set as
/// `−∞` so the operator is never selected, instead of aborting the search.
fn local_score_or_degenerate(
    cache: &mut LocalScoreCache,
    data: &GraphScoreData,
    node: u32,
    parents: &Arc<[u32]>,
) -> Result<f64, DiscoveryError> {
    match cache.local_score(data, node, parents) {
        Ok(score) => Ok(score),
        Err(antecedent_state::StateError::Numerical(message)) if message.contains("degenerate") => {
            Ok(f64::NEG_INFINITY)
        }
        Err(error) => Err(error.into()),
    }
}

/// `new − old`, with a non-finite candidate scoring `−∞` (never chosen) and a non-finite
/// current score yielding `+∞` when the candidate is finite (leaving the degenerate set).
fn operator_delta(old: f64, new: f64) -> f64 {
    if !new.is_finite() {
        f64::NEG_INFINITY
    } else if !old.is_finite() {
        f64::INFINITY
    } else {
        new - old
    }
}

fn for_each_subset(items: &[DenseNodeId], max_k: usize, mut visit: impl FnMut(&[DenseNodeId])) {
    let m = items.len().min(max_k);
    let mut scratch = Vec::new();
    for k in 0..=m {
        crate::combinations::for_each_combination(items, k, &mut scratch, |combo| {
            visit(combo);
            true
        });
    }
}

/// The score data and limits GES operator searches read.
#[derive(Clone, Copy)]
struct GesSearch<'a> {
    data: &'a GraphScoreData,
    max_parents: usize,
    max_subset: usize,
    screen: Option<&'a HashSet<(u32, u32)>>,
    constraints: &'a DiscoveryConstraints,
    variables: &'a [VariableId],
}

fn best_insert(
    cpdag: &Cpdag,
    search: GesSearch<'_>,
    cache: &mut LocalScoreCache,
    deltas: &mut GesDeltaCache,
) -> Result<Option<OpCand>, DiscoveryError> {
    let GesSearch { data, max_parents, max_subset, screen, constraints, variables } = search;
    let n = cpdag.node_count();
    let mut best: Option<OpCand> = None;
    for xi in 0..n {
        let x = DenseNodeId::from_raw(crate::indexing::dense_u32(xi));
        for yi in 0..n {
            if xi == yi {
                continue;
            }
            let y = DenseNodeId::from_raw(crate::indexing::dense_u32(yi));
            if cpdag.has_edge(x, y) {
                continue;
            }
            if !screened(screen, x, y) {
                continue;
            }
            if forbidden_direction(constraints, variables, x, y) {
                continue;
            }
            // T ⊆ Ne(Y) \ adj(X)
            let adj_x: HashSet<DenseNodeId> = cpdag.adjacent(x).into_iter().collect();
            let mut candidates: Vec<DenseNodeId> =
                cpdag.undirected_neighbors(y).into_iter().filter(|n| !adj_x.contains(n)).collect();
            candidates.sort_unstable_by_key(|node| node.raw());
            let mut subsets: Vec<Vec<DenseNodeId>> = Vec::new();
            if candidates.len() > max_subset {
                return Err(DiscoveryError::Resource(format!(
                    "GES Insert T-subset cap {max_subset} < {} neighbours; refusing an incomplete operator search",
                    candidates.len()
                )));
            }
            for_each_subset(&candidates, max_subset, |t| subsets.push(t.to_vec()));
            for t in subsets {
                if !insert_valid(cpdag, x, y, &t) {
                    continue;
                }
                let pa_len = insert_scored_parents(cpdag, x, y, &t);
                if pa_len > max_parents {
                    continue;
                }
                let delta = insert_delta(cpdag, cache, deltas, data, x, y, &t)?;
                if best.as_ref().is_none_or(|b| delta > b.delta) {
                    best = Some(OpCand { x, y, set: t, delta });
                }
            }
        }
    }
    Ok(best)
}

fn best_delete(
    cpdag: &Cpdag,
    search: GesSearch<'_>,
    cache: &mut LocalScoreCache,
    deltas: &mut GesDeltaCache,
) -> Result<Option<OpCand>, DiscoveryError> {
    let GesSearch { data, max_subset, constraints, variables, .. } = search;
    let mut best: Option<OpCand> = None;
    let mut edges = cpdag.edges();
    edges.sort_unstable_by_key(|edge| (edge.a.raw(), edge.b.raw()));
    for e in edges {
        // Consider both orientations for undirected; directed as parent→child.
        let pairs: Vec<(DenseNodeId, DenseNodeId)> = if e.is_undirected() {
            vec![(e.a, e.b), (e.b, e.a)]
        } else if let Some((from, to)) = e.parent_child() {
            vec![(from, to)]
        } else {
            continue;
        };
        for (x, y) in pairs {
            if required_pair(constraints, variables, x, y) {
                continue;
            }
            let na = na_yx(cpdag, y, x);
            if na.len() > max_subset {
                return Err(DiscoveryError::Resource(format!(
                    "GES Delete H-subset cap {max_subset} < {} neighbours; refusing an incomplete operator search",
                    na.len()
                )));
            }
            let mut subsets: Vec<Vec<DenseNodeId>> = Vec::new();
            for_each_subset(&na, max_subset, |h| subsets.push(h.to_vec()));
            for h in subsets {
                if !delete_valid(cpdag, x, y, &h) {
                    continue;
                }
                let delta = delete_delta(cpdag, cache, deltas, data, x, y, &h)?;
                if best.as_ref().is_none_or(|b| delta > b.delta) {
                    best = Some(OpCand { x, y, set: h, delta });
                }
            }
        }
    }
    Ok(best)
}

fn best_reverse(
    cpdag: &Cpdag,
    search: GesSearch<'_>,
    cache: &mut LocalScoreCache,
    deltas: &mut GesDeltaCache,
) -> Result<Option<ReverseCand>, DiscoveryError> {
    let GesSearch { data, max_parents, max_subset, screen, constraints, variables } = search;
    let mut best: Option<ReverseCand> = None;
    let mut edges = cpdag.edges();
    edges.sort_unstable_by_key(|edge| (edge.a.raw(), edge.b.raw()));
    for e in edges {
        let Some((x, y)) = e.parent_child() else {
            continue;
        };
        if required_pair(constraints, variables, x, y) {
            continue;
        }
        if !screened(screen, x, y) {
            continue;
        }
        if forbidden_direction(constraints, variables, y, x) {
            continue;
        }
        let na = na_yx(cpdag, y, x);
        if na.len() > max_subset {
            return Err(DiscoveryError::Resource(format!(
                "GES Reverse H-subset cap {max_subset} < {} neighbours; refusing an incomplete operator search",
                na.len()
            )));
        }
        let mut h_subsets: Vec<Vec<DenseNodeId>> = Vec::new();
        for_each_subset(&na, max_subset, |h| h_subsets.push(h.to_vec()));
        for h in h_subsets {
            if !delete_valid(cpdag, x, y, &h) {
                continue;
            }
            let d_del = delete_delta(cpdag, cache, deltas, data, x, y, &h)?;
            let mut tmp = cpdag.clone();
            apply_delete(&mut tmp, x, y, &h)?;
            // Insert opposite Y → X
            let adj_y: HashSet<DenseNodeId> = tmp.adjacent(y).into_iter().collect();
            let mut candidates: Vec<DenseNodeId> =
                tmp.undirected_neighbors(x).into_iter().filter(|n| !adj_y.contains(n)).collect();
            candidates.sort_unstable_by_key(|node| node.raw());
            if candidates.len() > max_subset {
                return Err(DiscoveryError::Resource(format!(
                    "GES Reverse T-subset cap {max_subset} < {} neighbours; refusing an incomplete operator search",
                    candidates.len()
                )));
            }
            let mut t_subsets: Vec<Vec<DenseNodeId>> = Vec::new();
            for_each_subset(&candidates, max_subset, |t| t_subsets.push(t.to_vec()));
            for t in t_subsets {
                if !insert_valid(&tmp, y, x, &t) {
                    continue;
                }
                let pa_len = insert_scored_parents(&tmp, y, x, &t);
                if pa_len > max_parents {
                    continue;
                }
                let d_ins = insert_delta(&tmp, cache, deltas, data, y, x, &t)?;
                let delta = d_del + d_ins;
                if best.as_ref().is_none_or(|b| delta > b.delta) {
                    best = Some(ReverseCand { x, y, h: h.clone(), t, delta });
                }
            }
        }
    }
    Ok(best)
}

fn apply_insert(
    cpdag: &mut Cpdag,
    x: DenseNodeId,
    y: DenseNodeId,
    t: &[DenseNodeId],
) -> Result<(), DiscoveryError> {
    cpdag.insert_directed(x, y)?;
    for &ti in t {
        if let Some(e) = cpdag.edge_between(ti, y) {
            if e.is_undirected() {
                cpdag.orient_undirected(ti, y)?;
            }
        }
    }
    let dag = pdag_to_dag(cpdag)?;
    *cpdag = dag_to_cpdag(&dag)?;
    Ok(())
}

fn apply_delete(
    cpdag: &mut Cpdag,
    x: DenseNodeId,
    y: DenseNodeId,
    h: &[DenseNodeId],
) -> Result<(), DiscoveryError> {
    cpdag.remove_edge(x, y)?;
    for &hi in h {
        if let Some(e) = cpdag.edge_between(y, hi) {
            if e.is_undirected() {
                cpdag.orient_undirected(y, hi)?;
            }
        }
        if let Some(e) = cpdag.edge_between(x, hi) {
            if e.is_undirected() {
                cpdag.orient_undirected(x, hi)?;
            }
        }
    }
    let dag = pdag_to_dag(cpdag)?;
    *cpdag = dag_to_cpdag(&dag)?;
    Ok(())
}

/// Dor–Tarsi consistent extension of a PDAG to a DAG.
fn pdag_to_dag(pdag: &Cpdag) -> Result<Dag, DiscoveryError> {
    let mut work = pdag.clone();
    let n = work.node_count();
    let mut dag = Dag::empty();
    for node in work.nodes() {
        dag.add_node(*node)?;
    }
    // Copy already-directed edges.
    for e in work.edges() {
        if let Some((from, to)) = e.parent_child() {
            let _ = dag.insert_directed(from, to);
        }
    }

    let mut remaining: HashSet<DenseNodeId> =
        (0..crate::indexing::dense_u32(n)).map(DenseNodeId::from_raw).collect();
    while !remaining.is_empty() {
        // Select x ∈ remaining: no directed edge out of x to remaining, and the
        // adjacent set (remaining parents ∪ remaining undirected neighbors) is a clique.
        let mut selected = None;
        let mut candidates: Vec<_> = remaining.iter().copied().collect();
        candidates.sort_unstable_by_key(|node| node.raw());
        for x in candidates {
            let out_to_remaining = work.children(x).into_iter().any(|c| remaining.contains(&c));
            if out_to_remaining {
                continue;
            }
            let remaining_undirected: Vec<DenseNodeId> = work
                .undirected_neighbors(x)
                .into_iter()
                .filter(|n| remaining.contains(n))
                .collect();
            let mut adjacent: Vec<DenseNodeId> =
                work.parents(x).into_iter().filter(|p| remaining.contains(p)).collect();
            adjacent.extend(remaining_undirected.iter().copied());
            adjacent.sort_unstable_by_key(|node| node.raw());
            adjacent.dedup();
            // Dor–Tarsi (1992): every undirected neighbor of `x` must be adjacent to every
            // *other* vertex adjacent to `x` (parent or undirected neighbor alike). A pair of
            // `x`'s parents that are not adjacent to each other is not itself disqualifying —
            // those edges are already fixed, so nothing new is being oriented between them;
            // requiring it here would reject every plain collider `a → x ← b` (a, b
            // non-adjacent) as having "no consistent extension", when the collider already
            // *is* its own extension.
            let mut ok = true;
            'outer: for &u in &remaining_undirected {
                for &z in &adjacent {
                    if z != u && !work.has_edge(u, z) {
                        ok = false;
                        break 'outer;
                    }
                }
            }
            if ok {
                selected = Some(x);
                break;
            }
        }
        let Some(x) = selected else {
            return Err(DiscoveryError::Unsupported {
                message: "GES PDAG has no consistent DAG extension",
            });
        };
        let mut neighbors = work.undirected_neighbors(x);
        neighbors.sort_unstable_by_key(|node| node.raw());
        for n in neighbors {
            if remaining.contains(&n) {
                work.orient_undirected(n, x)?;
                let _ = dag.insert_directed(n, x);
            }
        }
        remaining.remove(&x);
    }
    Ok(dag)
}

/// Essential graph: skeleton + DAG v-structures + Meek R1–R4.
fn dag_to_cpdag(dag: &Dag) -> Result<Cpdag, DiscoveryError> {
    let mut cpdag = Cpdag::empty();
    for node in dag.nodes() {
        cpdag.add_node(*node)?;
    }
    for e in dag.edges() {
        if let Some((from, to)) = e.parent_child() {
            if !cpdag.has_edge(from, to) {
                cpdag.insert_undirected(from, to)?;
            }
        }
    }
    // Orient unshielded colliders present in the DAG.
    let n = dag.node_count();
    for zi in 0..n {
        let z = DenseNodeId::from_raw(crate::indexing::dense_u32(zi));
        let parents = dag.parents(z);
        for i in 0..parents.len() {
            for j in i + 1..parents.len() {
                let p = parents[i];
                let q = parents[j];
                if !dag.has_edge(p, q) && !dag.has_edge(q, p) {
                    // p → z ← q
                    if cpdag
                        .edge_between(p, z)
                        .is_some_and(antecedent_graph::MarkedEdge::is_undirected)
                    {
                        cpdag.orient_undirected(p, z)?;
                    }
                    if cpdag
                        .edge_between(q, z)
                        .is_some_and(antecedent_graph::MarkedEdge::is_undirected)
                    {
                        cpdag.orient_undirected(q, z)?;
                    }
                }
            }
        }
    }
    let mut state = OrientationState::default();
    let rules: [&dyn OrientationRule<antecedent_graph::Cpdag>; 4] =
        [&MeekR1, &MeekR2, &MeekR3, &MeekR4];
    let _ = run_static_orientation_to_fixed_point(&mut cpdag, &rules, &mut state)?;
    Ok(cpdag)
}

// Dag adjacency helper used by `dag_to_cpdag`.
trait DagEdgeHelpers {
    fn has_edge(&self, a: DenseNodeId, b: DenseNodeId) -> bool;
}

impl DagEdgeHelpers for Dag {
    fn has_edge(&self, a: DenseNodeId, b: DenseNodeId) -> bool {
        self.children(a).contains(&b) || self.children(b).contains(&a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::VariableId;
    use antecedent_data::{TableView, TabularData};

    fn gaussian_chain_data(n: usize) -> (TabularData, Vec<VariableId>) {
        // X0 → X1 → X2 linear Gaussian.
        let mut x0 = vec![0.0; n];
        let mut x1 = vec![0.0; n];
        let mut x2 = vec![0.0; n];
        for i in 0..n {
            let e0 = ((i as f64 * 0.017) % 1.0) - 0.5;
            let e1 = ((i as f64 * 0.029) % 1.0) - 0.5;
            let e2 = ((i as f64 * 0.041) % 1.0) - 0.5;
            x0[i] = e0;
            x1[i] = 0.8 * x0[i] + e1;
            x2[i] = 0.8 * x1[i] + e2;
        }
        let data = crate::test_tables::continuous_table(&["x0", "x1", "x2"], vec![x0, x1, x2]);
        let vars: Vec<_> = data.schema().variables().iter().map(|v| v.id).collect();
        (data, vars)
    }

    #[test]
    fn ges_chain_recovers_adjacent_skeleton() {
        let (data, vars) = gaussian_chain_data(400);
        let ges = Ges::new();
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let result = ges.run(&data, &vars, &mut ws, &ctx).unwrap();
        assert_eq!(result.algorithm.id.as_ref(), "ges");
        let g = &result.evidence.graph;
        // Score recorded before the operator-delta cache. The cached search must keep these bits.
        let dag = pdag_to_dag(g).unwrap();
        let mut flat = Vec::new();
        for v in &vars {
            flat.extend(data.float64_values(*v).unwrap());
        }
        let score_data = GraphScoreData::new(400, vars.len(), Arc::from(flat)).unwrap();
        let mut score_cache = LocalScoreCache::new(crate::graph_score::graph_score_cache_key(
            GraphScoreFamily::GaussianBic,
            vars.len(),
            400,
        ));
        for node in 0..crate::indexing::dense_u32(vars.len()) {
            let parents: Vec<u32> =
                dag.parents(DenseNodeId::from_raw(node)).iter().map(|p| p.raw()).collect();
            let _ = score_cache.local_score(&score_data, node, &Arc::from(parents)).unwrap();
        }
        assert_eq!(
            score_cache.score_graph(&score_data).unwrap().to_bits(),
            13_869_790_556_363_832_396
        );
        let d = |i: u32| DenseNodeId::from_raw(i);
        // Chain skeleton should keep 0—1 and 1—2; 0—2 may be absent.
        assert!(g.has_edge(d(0), d(1)), "expected edge 0-1");
        assert!(g.has_edge(d(1), d(2)), "expected edge 1-2");
    }

    #[test]
    fn delta_cache_hit_returns_the_stored_value() {
        let (data, vars) = gaussian_chain_data(40);
        let mut flat = Vec::new();
        for v in &vars {
            flat.extend(data.float64_values(*v).unwrap());
        }
        let score = GraphScoreData::new(40, vars.len(), Arc::from(flat)).unwrap();
        let mut cache = LocalScoreCache::new(crate::graph_score::graph_score_cache_key(
            GraphScoreFamily::GaussianBic,
            vars.len(),
            40,
        ));
        let cpdag = Cpdag::with_variables(3);
        let x = DenseNodeId::from_raw(0);
        let y = DenseNodeId::from_raw(1);
        let mut deltas = GesDeltaCache::default();
        let first = insert_delta(&cpdag, &mut cache, &mut deltas, &score, x, y, &[]).unwrap();
        let mut fresh = LocalScoreCache::new(crate::graph_score::graph_score_cache_key(
            GraphScoreFamily::GaussianBic,
            vars.len(),
            40,
        ));
        let old = local_score_or_degenerate(&mut fresh, &score, y.raw(), &Arc::from([])).unwrap();
        let new =
            local_score_or_degenerate(&mut fresh, &score, y.raw(), &Arc::from([x.raw()])).unwrap();
        assert_eq!(first.to_bits(), operator_delta(old, new).to_bits());
        for stored in deltas.entries.values_mut() {
            *stored = 123.0;
        }
        let hit = insert_delta(&cpdag, &mut cache, &mut deltas, &score, x, y, &[]).unwrap();
        assert_eq!(hit.to_bits(), 123.0f64.to_bits());
    }

    #[test]
    fn max_parents_counts_na_neighbors_that_are_scored() {
        // Undirected edges 0—2 and 1—2. Inserting 0→1 scores parents {0, 2}
        // because 2 ∈ NA(1, 0), even though T is empty and 1 has no directed parents.
        // The old gate counted `|Pa| + 1 + |T| = 1` and accepted the operator.
        let mut cpdag = Cpdag::with_variables(3);
        let x = DenseNodeId::from_raw(0);
        let y = DenseNodeId::from_raw(1);
        let z = DenseNodeId::from_raw(2);
        cpdag.insert_undirected(x, z).unwrap();
        cpdag.insert_undirected(y, z).unwrap();
        assert_eq!(insert_scored_parents(&cpdag, x, y, &[]), 2);
        assert!(insert_valid(&cpdag, x, y, &[]));

        let (data, vars) = gaussian_chain_data(40);
        let mut flat = Vec::new();
        for v in &vars {
            flat.extend(data.float64_values(*v).unwrap());
        }
        let score = GraphScoreData::new(40, vars.len(), Arc::from(flat)).unwrap();
        let mut cache = LocalScoreCache::new(crate::graph_score::graph_score_cache_key(
            GraphScoreFamily::GaussianBic,
            vars.len(),
            40,
        ));
        let constraints = DiscoveryConstraints::default();
        let search = GesSearch {
            data: &score,
            max_parents: 1,
            max_subset: 12,
            screen: None,
            constraints: &constraints,
            variables: &vars,
        };
        let best = best_insert(&cpdag, search, &mut cache, &mut GesDeltaCache::default()).unwrap();
        assert!(
            best.is_none(),
            "an insert whose scored parent set exceeds max_parents must be skipped"
        );
    }

    #[test]
    fn empty_clique_and_insert_on_empty_graph() {
        let mut cpdag = Cpdag::with_variables(3);
        let x = DenseNodeId::from_raw(0);
        let y = DenseNodeId::from_raw(1);
        assert!(insert_valid(&cpdag, x, y, &[]));
        apply_insert(&mut cpdag, x, y, &[]).unwrap();
        assert!(cpdag.has_edge(x, y));
    }

    /// PDAG `0 → 2 — 1` (no 0—1): the weakened undirected-only clique test would
    /// accept 2 as a sink and invent the v-structure `0 → 2 ← 1`. With parents in
    /// the clique, the unique consistent extension is `0 → 2 → 1`.
    #[test]
    fn pdag_to_dag_rejects_sink_whose_parents_miss_undirected_neighbor() {
        let mut pdag = Cpdag::with_variables(3);
        let a = DenseNodeId::from_raw(0);
        let b = DenseNodeId::from_raw(1);
        let c = DenseNodeId::from_raw(2);
        pdag.insert_directed(a, c).unwrap();
        pdag.insert_undirected(b, c).unwrap();

        let dag = pdag_to_dag(&pdag).expect("consistent extension exists");
        // Must be the chain 0 → 2 → 1, not the invented collider 0 → 2 ← 1.
        assert!(dag.children(a).contains(&c));
        assert!(dag.children(c).contains(&b));
        assert!(!dag.children(b).contains(&c));
        assert!(dag.parents(c).iter().all(|&p| p == a));
    }

    #[test]
    fn pdag_to_dag_extends_directed_chain() {
        let mut pdag = Cpdag::with_variables(3);
        let n0 = DenseNodeId::from_raw(0);
        let n1 = DenseNodeId::from_raw(1);
        let n2 = DenseNodeId::from_raw(2);
        pdag.insert_directed(n0, n1).unwrap();
        pdag.insert_directed(n1, n2).unwrap();

        let dag = pdag_to_dag(&pdag).expect("chain is a consistent extension");
        assert!(dag.children(n0).contains(&n1));
        assert!(dag.children(n1).contains(&n2));
        assert!(dag.children(n2).is_empty());
    }

    #[test]
    fn pdag_to_dag_extends_undirected_clique() {
        let mut pdag = Cpdag::with_variables(3);
        let n0 = DenseNodeId::from_raw(0);
        let n1 = DenseNodeId::from_raw(1);
        let n2 = DenseNodeId::from_raw(2);
        pdag.insert_undirected(n0, n1).unwrap();
        pdag.insert_undirected(n1, n2).unwrap();
        pdag.insert_undirected(n0, n2).unwrap();

        let dag = pdag_to_dag(&pdag).expect("clique has a consistent extension");
        // Dor–Tarsi orients into successive sinks; every undirected edge becomes directed.
        assert_eq!(dag.edges().count(), 3);
        for e in dag.edges() {
            assert!(e.parent_child().is_some());
        }
    }

    #[test]
    fn ges_scores_a_degenerate_parent_as_negative_infinity() {
        let n = 80;
        let x: Vec<f64> = (0..n).map(|i| f64::from(i) * 0.1).collect();
        let data = crate::test_tables::continuous_table(&["d0", "d1"], vec![x.clone(), x]);
        let vars: Vec<_> = data.schema().variables().iter().map(|v| v.id).collect();
        let result = Ges::new().run(
            &data,
            &vars,
            &mut DiscoveryWorkspace::default(),
            &ExecutionContext::for_tests(1),
        );
        assert!(result.is_ok(), "{result:?}");
    }

    /// Run GES on the chain data under `constraints`.
    fn ges_chain_under(constraints: DiscoveryConstraints) -> Cpdag {
        let (data, vars) = gaussian_chain_data(400);
        let ges = Ges::new().with_constraints(constraints);
        let mut ws = DiscoveryWorkspace::default();
        ges.run(&data, &vars, &mut ws, &ExecutionContext::for_tests(1)).unwrap().evidence.graph
    }

    fn parent_child(g: &Cpdag, a: u32, b: u32) -> Option<(u32, u32)> {
        g.edge_between(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b))
            .and_then(antecedent_graph::MarkedEdge::parent_child)
            .map(|(p, c)| (p.raw(), c.raw()))
    }

    fn contemp(a: u32, b: u32) -> LaggedLink {
        crate::constraints::contemp_link(VariableId::from_raw(a), VariableId::from_raw(b))
    }

    /// One present edge between variables `lo < hi`, with `(parent, child)` when directed.
    type Orientation = (u32, u32, Option<(u32, u32)>);

    /// Every present edge between variables `a < b`, as `(parent, child)` when directed.
    fn orientations(g: &Cpdag) -> Vec<Orientation> {
        g.edges()
            .into_iter()
            .map(|e| {
                let (lo, hi) = (e.a.raw().min(e.b.raw()), e.a.raw().max(e.b.raw()));
                (lo, hi, e.parent_child().map(|(p, c)| (p.raw(), c.raw())))
            })
            .collect()
    }

    // The tests below assert what the knowledge guarantees for whatever skeleton the score
    // search finds: a tier or a forbidden direction fixes the direction of every adjacent pair
    // it covers, and a required link is an edge that is never oriented against the statement.

    #[test]
    fn tiers_orient_every_present_edge_from_the_earlier_tier_to_the_later() {
        let tier = |ids: &[u32]| -> Arc<[VariableId]> {
            ids.iter().copied().map(VariableId::from_raw).collect()
        };
        let mut forward = Ges::new().constraints;
        forward.tiers = Arc::from([tier(&[0]), tier(&[1]), tier(&[2])]);
        let edges = orientations(&ges_chain_under(forward));
        assert!(!edges.is_empty());
        for (lo, hi, direction) in edges {
            assert_eq!(direction, Some((lo, hi)), "edge {lo}-{hi}");
        }

        let mut backward = Ges::new().constraints;
        backward.tiers = Arc::from([tier(&[2]), tier(&[1]), tier(&[0])]);
        let edges = orientations(&ges_chain_under(backward));
        assert!(!edges.is_empty());
        for (lo, hi, direction) in edges {
            assert_eq!(direction, Some((hi, lo)), "edge {lo}-{hi}");
        }
    }

    #[test]
    fn a_forbidden_direction_is_never_present_in_the_output() {
        let mut c = Ges::new().constraints;
        c.forbidden = Arc::from([contemp(1, 0), contemp(2, 1)]);
        let g = ges_chain_under(c);
        assert_ne!(parent_child(&g, 0, 1), Some((1, 0)));
        assert_ne!(parent_child(&g, 1, 2), Some((2, 1)));
        // Neither pair is left undirected: the single allowed direction is fixed.
        for (lo, hi, direction) in orientations(&g) {
            if (lo, hi) == (0, 1) || (lo, hi) == (1, 2) {
                assert_eq!(direction, Some((lo, hi)), "edge {lo}-{hi}");
            }
        }
    }

    #[test]
    fn a_required_link_is_an_edge_and_is_not_oriented_against_the_statement() {
        let mut c = Ges::new().constraints;
        c.required = Arc::from([contemp(2, 1)]);
        let g = ges_chain_under(c);
        assert!(g.has_edge(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)));
        assert_ne!(parent_child(&g, 1, 2), Some((1, 2)));
    }

    #[test]
    fn a_parent_limit_is_enforced_and_unknown_variables_are_refused() {
        // Every insertion scores at least the new parent, so a limit of zero admits no edge.
        let mut capped = Ges::new().constraints;
        capped.max_parents = Some(0);
        assert!(ges_chain_under(capped).edges().is_empty());

        let (data, vars) = gaussian_chain_data(50);
        let mut unknown = Ges::new().constraints;
        unknown.forbidden = Arc::from([contemp(0, 9)]);
        let mut ws = DiscoveryWorkspace::default();
        let error = Ges::new()
            .with_constraints(unknown)
            .run(&data, &vars, &mut ws, &ExecutionContext::for_tests(1))
            .unwrap_err();
        assert!(matches!(
            error,
            DiscoveryError::Knowledge(crate::constraints::KnowledgeFault::UnknownVariable(_))
        ));
    }
}
