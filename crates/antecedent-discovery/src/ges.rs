//! Greedy Equivalence Search (GES) → static [`Cpdag`] (Chickering 2002).
//!
//! Forward Insert / turning Reverse / backward Delete on the CPDAG, scored by the local
//! score family the search is given (Gaussian BIC by default) via
//! [`antecedent_state::LocalScoreCache`]. Optional PC-skeleton screening restricts Insert
//! candidates.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::cell::Cell;
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

/// Identity of this algorithm in its records.
const ALGORITHM: &str = crate::pipeline::StaticAlgorithm::Ges.as_str();

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
    /// Work budget of the operator search per node pair: the power set of this many
    /// neighbours (`2^max_subset` candidate T/H sets; `None` → 12). Only sets that can be valid
    /// operators (cliques) are enumerated, smallest first, so the budget bites only on dense
    /// neighbourhoods; when it does, the search is pruned to the smaller sets and the result
    /// says so in a `ges.operator_search_pruned` diagnostic.
    pub max_subset: Option<usize>,
    /// Local score family the search maximises.
    pub score_family: GraphScoreFamily,
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
            fdr: Some(crate::pipeline::static_bh_fdr()),
            screen_pc: false,
            max_subset: None,
            score_family: GraphScoreFamily::GaussianBic,
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
        self.fdr = crate::pipeline::static_fdr(fdr);
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

    /// Per-pair operator-search budget (`None` uses default 12; see [`Self::max_subset`]).
    #[must_use]
    pub fn with_max_subset(mut self, max_subset: Option<usize>) -> Self {
        self.max_subset = max_subset;
        self
    }

    /// Score the search with `family` instead of Gaussian BIC.
    #[must_use]
    pub fn with_score_family(mut self, family: GraphScoreFamily) -> Self {
        self.score_family = family;
        self
    }

    /// Run GES.
    ///
    /// # Errors
    ///
    /// Data, score, graph, or screening failures.
    pub fn run(
        &self,
        data: &TabularData,
        variables: &[VariableId],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<StaticCpdagDiscoveryResult, DiscoveryError> {
        self.validate_request(variables)?;
        let score_data = ges_score_data(data, variables)?;
        let n_vars = variables.len();
        let mut cache = LocalScoreCache::new(crate::graph_score::graph_score_cache_key(
            self.score_family,
            n_vars,
            score_data.n_rows,
        ));
        let mut deltas = GesDeltaCache::default();

        let screen: Option<HashSet<(u32, u32)>> = if self.screen_pc {
            Some(pc_skeleton_pairs(self, data, variables, workspace, ctx)?)
        } else {
            None
        };

        let mut cpdag = empty_cpdag(variables)?;
        seed_required_edges(&mut cpdag, variables, &self.constraints)?;

        let pruned = Cell::new(0_u64);
        let search = GesSearch {
            data: &score_data,
            max_parents: self.constraints.max_parents.unwrap_or(n_vars.saturating_sub(1)),
            subset_budget: 1_usize << self.max_subset.unwrap_or(12).min(30),
            screen: screen.as_ref(),
            constraints: &self.constraints,
            variables,
            tolerance: score_gain_tolerance(&score_data, &mut cache)?,
            pruned: &pruned,
        };
        for phase in [SearchPhase::Insert, SearchPhase::Turn, SearchPhase::Delete] {
            phase.run(&mut cpdag, search, &mut cache, &mut deltas, ctx)?;
        }

        let mut diagnostics = Vec::new();
        if pruned.get() > 0 {
            diagnostics.push(DiscoveryDiagnostic {
                code: Arc::from("ges.operator_search_pruned"),
                message: Arc::from(format!(
                    "{} operator search(es) over a dense neighbourhood hit the per-pair budget \
                     of {} candidate sets and examined only the smaller sets; raise max_subset \
                     to search them exhaustively",
                    pruned.get(),
                    search.subset_budget
                )),
            });
        }
        diagnostics.extend(self.orient_by_knowledge(&mut cpdag, variables)?);

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
        self.assemble(cpdag, total_score, diagnostics, n_vars)
    }

    /// Validate the constraints, knowledge and variable list before any data is read.
    fn validate_request(&self, variables: &[VariableId]) -> Result<(), DiscoveryError> {
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
        Ok(())
    }

    /// Background knowledge orients what the score leaves undirected: the search forbids an
    /// insertion against a forbidden direction, but an equivalence class can still contain an
    /// edge whose direction knowledge fixes. Meek's rules then propagate the orientations.
    fn orient_by_knowledge(
        &self,
        cpdag: &mut Cpdag,
        variables: &[VariableId],
    ) -> Result<Vec<DiscoveryDiagnostic>, DiscoveryError> {
        let mut diagnostics = Vec::new();
        if !self.constraints.has_directional_knowledge() {
            return Ok(diagnostics);
        }
        let var_index: HashMap<VariableId, usize> =
            variables.iter().enumerate().map(|(i, v)| (*v, i)).collect();
        let dense_of = |v: VariableId| crate::pipeline::dense_of(&var_index, v);
        let mut state = OrientationState::default();
        crate::pc::orient_fixed_directions(
            cpdag,
            &self.constraints,
            variables,
            &dense_of,
            &mut state,
        )?;
        let rules: [&dyn OrientationRule<Cpdag>; 4] = [&MeekR1, &MeekR2, &MeekR3, &MeekR4];
        run_static_orientation_to_fixed_point(cpdag, &rules, &mut state)?;
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
        Ok(diagnostics)
    }

    /// The discovery result for the finished equivalence class.
    fn assemble(
        &self,
        cpdag: Cpdag,
        total_score: f64,
        diagnostics: Vec<DiscoveryDiagnostic>,
        n_vars: usize,
    ) -> Result<StaticCpdagDiscoveryResult, DiscoveryError> {
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
                    provenance: Arc::from([Arc::from(ALGORITHM)]),
                })
            })
            .collect();

        // GES scores graphs, not edges: there is no per-edge test statistic.
        let links: Vec<ScoredLink> =
            edge_evidence.iter().map(|e| ScoredLink::untested(e.link)).collect();

        let edge_count = cpdag.edges().len();
        let evidence = GraphEvidence {
            graph: cpdag.clone(),
            edge_evidence: Arc::from(edge_evidence),
            links: Arc::from(links),
            source: EvidenceSource::Discovery { algorithm: Arc::from(ALGORITHM) },
        };
        let review = CpdagReview::from_cpdag(cpdag, ALGORITHM);

        Ok(DiscoveryResult {
            evidence,
            review,
            algorithm: crate::pipeline::algorithm_record(
                ALGORITHM,
                format!(
                    "family={} screen_pc={} score={total_score:.6}{}",
                    score_family_label(self.score_family),
                    self.screen_pc,
                    self.constraints.knowledge_config()
                ),
            ),
            assumptions: discovery_assumptions(ALGORITHM, true),
            iterations: Vec::<DiscoveryIteration>::new(),
            diagnostics,
            performance: DiscoveryPerformanceRecord {
                ci_tests: 0,
                links_retained: crate::pipeline::count_u64(edge_count)?,
                targets: crate::pipeline::count_u64(n_vars)?,
                lagged_frame_bytes: 0,
                worker_threads: 1,
            },
            sepsets: crate::result::PcSepsets::default(),
        })
    }
}

/// Label of a score family in the algorithm record.
fn score_family_label(family: GraphScoreFamily) -> &'static str {
    match family {
        GraphScoreFamily::GaussianBic => "gaussian_bic",
        GraphScoreFamily::LaplaceBic => "laplace_bic",
        _ => "other",
    }
}

/// Column-major score data of the searched variables.
fn ges_score_data(
    data: &TabularData,
    variables: &[VariableId],
) -> Result<GraphScoreData, DiscoveryError> {
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
    let mut flat = Vec::with_capacity(variables.len().saturating_mul(n));
    for c in &col_owned {
        flat.extend_from_slice(c.as_ref());
    }
    Ok(GraphScoreData::new(n, variables.len(), Arc::from(flat))?)
}

/// Smallest score gain a move must exceed to be applied.
///
/// A gain at the rounding level of the scores being compared is noise: accepting it can cycle
/// between equivalent graphs whose scores differ only in their last bits. The tolerance is
/// `64 ε` times the total magnitude of the empty graph's local scores (at least 1), so it
/// scales with the data. A move that leaves a degenerate (non-finite) score has an infinite gain
/// and always passes.
fn score_gain_tolerance(
    data: &GraphScoreData,
    cache: &mut LocalScoreCache,
) -> Result<f64, DiscoveryError> {
    let empty: Arc<[u32]> = Arc::from([]);
    let mut magnitude = 0.0_f64;
    for node in 0..data.n_vars {
        let score =
            local_score_or_degenerate(cache, data, crate::indexing::dense_u32(node), &empty)?;
        if score.is_finite() {
            magnitude += score.abs();
        }
    }
    Ok(64.0 * f64::EPSILON * magnitude.max(1.0))
}

/// One phase of the equivalence search: the best operator of its kind is applied while it
/// improves the score by more than the tolerance.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SearchPhase {
    /// Forward equivalence search (Insert).
    Insert,
    /// Turning phase (Reverse = Delete then Insert the opposite direction).
    Turn,
    /// Backward equivalence search (Delete).
    Delete,
}

impl SearchPhase {
    /// Apply the best operator of this phase when it improves the score; whether one was applied.
    fn step(
        self,
        cpdag: &mut Cpdag,
        search: GesSearch<'_>,
        cache: &mut LocalScoreCache,
        deltas: &mut GesDeltaCache,
    ) -> Result<bool, DiscoveryError> {
        match self {
            Self::Insert => {
                let Some(best) = best_insert(cpdag, search, cache, deltas)? else {
                    return Ok(false);
                };
                if best.delta <= search.tolerance {
                    return Ok(false);
                }
                apply_insert(cpdag, best.x, best.y, &best.set)?;
            }
            Self::Turn => {
                let Some(best) = best_reverse(cpdag, search, cache, deltas)? else {
                    return Ok(false);
                };
                if best.delta <= search.tolerance {
                    return Ok(false);
                }
                apply_delete(cpdag, best.x, best.y, &best.h)?;
                apply_insert(cpdag, best.y, best.x, &best.t)?;
            }
            Self::Delete => {
                let Some(best) = best_delete(cpdag, search, cache, deltas)? else {
                    return Ok(false);
                };
                if best.delta <= search.tolerance {
                    return Ok(false);
                }
                apply_delete(cpdag, best.x, best.y, &best.set)?;
            }
        }
        Ok(true)
    }

    fn run(
        self,
        cpdag: &mut Cpdag,
        search: GesSearch<'_>,
        cache: &mut LocalScoreCache,
        deltas: &mut GesDeltaCache,
        ctx: &ExecutionContext,
    ) -> Result<(), DiscoveryError> {
        loop {
            if ctx.cancellation.is_cancelled() {
                return Err(DiscoveryError::Cancelled);
            }
            if !self.step(cpdag, search, cache, deltas)? {
                return Ok(());
            }
        }
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
///
/// Every node whose edges the search reads (each node it expands) is added to `footprint`.
fn semi_directed_reaches_avoiding(
    cpdag: &Cpdag,
    from: DenseNodeId,
    to: DenseNodeId,
    avoid: &HashSet<DenseNodeId>,
    mut footprint: Option<&mut Footprint>,
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
        if let Some(footprint) = footprint.as_deref_mut() {
            footprint.add(u);
        }
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

#[cfg(test)]
fn insert_valid(cpdag: &Cpdag, x: DenseNodeId, y: DenseNodeId, t: &[DenseNodeId]) -> bool {
    if cpdag.has_edge(x, y) || x == y {
        return false;
    }
    insert_valid_given(cpdag, x, y, &na_yx(cpdag, y, x), t, None)
}

/// Insert validity for an `x`, `y` known to be non-adjacent, with `NA(Y,X)` already computed.
fn insert_valid_given(
    cpdag: &Cpdag,
    x: DenseNodeId,
    y: DenseNodeId,
    na: &[DenseNodeId],
    t: &[DenseNodeId],
    footprint: Option<&mut Footprint>,
) -> bool {
    let mut clique_nodes = na.to_vec();
    clique_nodes.extend_from_slice(t);
    if !is_clique(cpdag, &clique_nodes) {
        return false;
    }
    let avoid: HashSet<DenseNodeId> = clique_nodes.into_iter().collect();
    // Every semi-directed path Y ↝ X hits NA∪T ⇔ no path avoiding NA∪T.
    !semi_directed_reaches_avoiding(cpdag, y, x, &avoid, footprint)
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

/// The parent sets Insert(X, Y, T) compares: `Pa(Y) ∪ NA(Y,X) ∪ T` and the same with `{X}`
/// added, each sorted and without repeats.
///
/// The larger set is what the operator scores, so it is also what counts against
/// `max_parents`: counting only `Pa` and `T` would let neighbours in `NA(Y,X)` push a node past
/// the limit while the operator is still accepted.
fn insert_parent_sets(
    cpdag: &Cpdag,
    x: DenseNodeId,
    y: DenseNodeId,
    na: &[DenseNodeId],
    t: &[DenseNodeId],
) -> (Vec<u32>, Vec<u32>) {
    let mut base = parents_u32(cpdag, y);
    for n in na.iter().chain(t.iter()) {
        base.push(n.raw());
    }
    base.sort_unstable();
    base.dedup();
    let mut neu = base.clone();
    neu.push(x.raw());
    neu.sort_unstable();
    neu.dedup();
    (base, neu)
}

fn parents_u32(cpdag: &Cpdag, y: DenseNodeId) -> Vec<u32> {
    let mut out: Vec<_> =
        cpdag.parents(y).into_iter().map(antecedent_graph::DenseNodeId::raw).collect();
    out.sort_unstable();
    out
}

#[cfg(test)]
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
    let (base, neu) = insert_parent_sets(cpdag, x, y, &na_yx(cpdag, y, x), t);
    cached_operator_delta(cache, deltas, data, y.raw(), &base, &neu)
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
    cached_operator_delta(cache, deltas, data, y.raw(), &with_x, &without_x)
}

/// A set of nodes, as a bitset.
#[derive(Clone, Debug, Default)]
struct Footprint {
    words: Vec<u64>,
}

impl Footprint {
    fn new(n: usize) -> Self {
        Self { words: vec![0; n.div_ceil(64)] }
    }

    fn add(&mut self, node: DenseNodeId) {
        self.add_index(node.as_usize());
    }

    fn add_index(&mut self, index: usize) {
        if let Some(word) = self.words.get_mut(index / 64) {
            *word |= 1u64 << (index % 64);
        }
    }

    fn contains(&self, node: DenseNodeId) -> bool {
        let index = node.as_usize();
        self.words.get(index / 64).is_some_and(|word| word & (1u64 << (index % 64)) != 0)
    }

    fn meets(&self, other: &Self) -> bool {
        self.words.iter().zip(&other.words).any(|(a, b)| a & b != 0)
    }
}

/// What the Insert search found for one ordered pair, and the nodes whose edges it read.
struct PairInsert {
    /// The highest-scoring valid operator of the pair (the first, among equals).
    best: Option<(Vec<DenseNodeId>, f64)>,
    /// The clique enumeration stopped at the per-pair budget.
    truncated: bool,
    footprint: Footprint,
}

/// Memory the stored footprints may take, in `u64` words (128 MiB). A pair past it is searched
/// again at every iteration instead of being kept.
const PAIR_CACHE_WORD_BUDGET: usize = 1 << 24;

/// Insert results per ordered pair, kept across the iterations of the forward phase.
///
/// Whether `Insert(X, Y, T)` is valid, and which `T` there are, is a function of the edges
/// incident to `X`, `Y`, `Ne(Y)` and to every node the semi-directed path search `Y ↝ X`
/// expanded; its score delta is a function of the parent sets alone. A pair's result therefore
/// holds until an edge incident to one of those nodes (its footprint) changes. Each call diffs
/// the graph against the one the entries were computed on and drops exactly the entries whose
/// footprint meets a changed edge, so a graph it was not told about is still handled correctly.
#[derive(Default)]
struct InsertPairCache {
    /// Edges of the graph the entries describe, keyed by `(low, high)` node with the marks at
    /// the low and high end.
    snapshot: HashMap<(u32, u32), (Endpoint, Endpoint)>,
    node_count: usize,
    entries: HashMap<(u32, u32), PairInsert>,
    words: usize,
    /// The one result that did not fit the budget, for the pair being read.
    overflow: Option<PairInsert>,
}

impl InsertPairCache {
    /// Bring the cache to `cpdag`: drop every entry that read a changed edge.
    fn sync(&mut self, cpdag: &Cpdag) {
        let n = cpdag.node_count();
        let mut now: HashMap<(u32, u32), (Endpoint, Endpoint)> = HashMap::new();
        for e in cpdag.edges() {
            let (lo, hi, at_lo, at_hi) = if e.a.raw() <= e.b.raw() {
                (e.a.raw(), e.b.raw(), e.at_a, e.at_b)
            } else {
                (e.b.raw(), e.a.raw(), e.at_b, e.at_a)
            };
            now.insert((lo, hi), (at_lo, at_hi));
        }
        if n != self.node_count {
            self.entries.clear();
            self.words = 0;
        } else {
            let mut changed = Footprint::new(n);
            let mut any = false;
            for (&(lo, hi), marks) in &now {
                if self.snapshot.get(&(lo, hi)) != Some(marks) {
                    changed.add_index(lo as usize);
                    changed.add_index(hi as usize);
                    any = true;
                }
            }
            for &(lo, hi) in self.snapshot.keys() {
                if !now.contains_key(&(lo, hi)) {
                    changed.add_index(lo as usize);
                    changed.add_index(hi as usize);
                    any = true;
                }
            }
            if any {
                self.entries.retain(|_, entry| !entry.footprint.meets(&changed));
                self.words = self.entries.values().map(|entry| entry.footprint.words.len()).sum();
            }
        }
        self.snapshot = now;
        self.node_count = n;
        self.overflow = None;
    }

    fn contains(&self, key: (u32, u32)) -> bool {
        self.entries.contains_key(&key)
    }

    fn store(&mut self, key: (u32, u32), entry: PairInsert) {
        if self.words + entry.footprint.words.len() <= PAIR_CACHE_WORD_BUDGET {
            self.words += entry.footprint.words.len();
            self.entries.insert(key, entry);
        } else {
            self.overflow = Some(entry);
        }
    }

    fn get(&self, key: (u32, u32)) -> Option<&PairInsert> {
        self.entries.get(&key).or(self.overflow.as_ref())
    }
}

/// Operator deltas keyed by the exact `(node, old parents, new parents)` triple that was
/// scored, flattened to `[node, old.len(), old.., new..]` so a lookup borrows a scratch slice
/// instead of allocating a key.
#[derive(Default)]
struct GesDeltaCache {
    entries: HashMap<Box<[u32]>, f64>,
    scratch: Vec<u32>,
    /// Per-pair Insert results of the forward phase.
    pairs: InsertPairCache,
    /// Per-candidate Insert results of the turning phase.
    reverses: ReverseCache,
}

fn cached_operator_delta(
    cache: &mut LocalScoreCache,
    deltas: &mut GesDeltaCache,
    data: &GraphScoreData,
    node: u32,
    old_parents: &[u32],
    new_parents: &[u32],
) -> Result<f64, DiscoveryError> {
    deltas.scratch.clear();
    deltas.scratch.push(node);
    deltas.scratch.push(crate::indexing::dense_u32(old_parents.len()));
    deltas.scratch.extend_from_slice(old_parents);
    deltas.scratch.extend_from_slice(new_parents);
    if let Some(delta) = deltas.entries.get(deltas.scratch.as_slice()).copied() {
        return Ok(delta);
    }
    let old = local_score_or_degenerate(cache, data, node, &Arc::from(old_parents))?;
    let new = local_score_or_degenerate(cache, data, node, &Arc::from(new_parents))?;
    let delta = operator_delta(old, new);
    deltas.entries.insert(deltas.scratch.as_slice().into(), delta);
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
        Err(antecedent_state::StateError::DegenerateScore(_)) => Ok(f64::NEG_INFINITY),
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

/// Subsets of `nodes` of at most `max_size` members whose members are pairwise adjacent, the
/// empty set included, smallest first and in node order within a size; and whether the
/// enumeration stopped at `budget` sets with more to come.
///
/// Both operator families need exactly these sets: Insert's `NA ∪ T` and Delete's `NA \ H` must
/// be cliques, so an enumeration that grows cliques never visits the (exponentially many)
/// subsets that cannot be valid.
fn clique_subsets(
    cpdag: &Cpdag,
    nodes: &[DenseNodeId],
    max_size: usize,
    budget: usize,
) -> (Vec<Vec<DenseNodeId>>, bool) {
    let mut out: Vec<Vec<DenseNodeId>> = Vec::new();
    // Each entry is a clique and the index of the first node it may still grow with.
    let mut level: Vec<(Vec<DenseNodeId>, usize)> = vec![(Vec::new(), 0)];
    loop {
        for (set, _) in &level {
            if out.len() == budget {
                return (out, true);
            }
            out.push(set.clone());
        }
        if level.first().is_none_or(|(set, _)| set.len() >= max_size) {
            return (out, false);
        }
        let mut next: Vec<(Vec<DenseNodeId>, usize)> = Vec::new();
        'grow: for (set, start) in &level {
            for (i, &candidate) in nodes.iter().enumerate().skip(*start) {
                if set.iter().all(|&member| cpdag.has_edge(member, candidate)) {
                    if out.len() + next.len() > budget {
                        // Everything past the budget is discarded by the next pass.
                        break 'grow;
                    }
                    let mut grown = set.clone();
                    grown.push(candidate);
                    next.push((grown, i + 1));
                }
            }
        }
        if next.is_empty() {
            return (out, false);
        }
        level = next;
    }
}

/// The score data and limits GES operator searches read.
#[derive(Clone, Copy)]
struct GesSearch<'a> {
    data: &'a GraphScoreData,
    max_parents: usize,
    /// Most candidate T/H sets examined per node pair.
    subset_budget: usize,
    screen: Option<&'a HashSet<(u32, u32)>>,
    constraints: &'a DiscoveryConstraints,
    variables: &'a [VariableId],
    /// A move improving the score by no more than this is not applied.
    tolerance: f64,
    /// Operator searches stopped by `subset_budget`.
    pruned: &'a Cell<u64>,
}

impl GesSearch<'_> {
    fn note_pruned(&self, truncated: bool) {
        if truncated {
            self.pruned.set(self.pruned.get() + 1);
        }
    }
}

/// A valid Insert(X, Y, T) operator with the parent sets it compares.
struct InsertOperator {
    t: Vec<DenseNodeId>,
    base: Vec<u32>,
    neu: Vec<u32>,
}

/// The valid Insert(X, Y, T) operators of a non-adjacent pair within `max_parents`, and whether
/// the clique enumeration stopped at the budget.
///
/// `T ⊆ Ne(Y) \ adj(X)` with `NA(Y,X) ∪ T` a clique and every semi-directed path `Y ↝ X`
/// blocked by it. Only cliques are enumerated (see [`clique_subsets`]). Every node whose edges
/// the answer depends on is added to `footprint`: the pair, `Ne(Y)` (the candidate and clique
/// tests read edges among them and to `X`), and each node the path search expands.
fn collect_inserts(
    cpdag: &Cpdag,
    x: DenseNodeId,
    y: DenseNodeId,
    adj_x: &HashSet<DenseNodeId>,
    search: &GesSearch<'_>,
    mut footprint: Option<&mut Footprint>,
) -> (Vec<InsertOperator>, bool) {
    if let Some(footprint) = footprint.as_deref_mut() {
        footprint.add(x);
        footprint.add(y);
        for neighbour in cpdag.undirected_neighbors(y) {
            footprint.add(neighbour);
        }
    }
    let na = na_yx(cpdag, y, x);
    if !is_clique(cpdag, &na) {
        return (Vec::new(), false);
    }
    let mut candidates: Vec<DenseNodeId> = cpdag
        .undirected_neighbors(y)
        .into_iter()
        .filter(|n| !adj_x.contains(n) && na.iter().all(|&a| cpdag.has_edge(*n, a)))
        .collect();
    candidates.sort_unstable_by_key(|node| node.raw());
    // A T larger than the parent limit scores more parents than the limit allows, so the
    // subsets that can be accepted are exactly those of at most `max_parents` members.
    let (subsets, truncated) =
        clique_subsets(cpdag, &candidates, search.max_parents, search.subset_budget);
    let mut out = Vec::new();
    for t in subsets {
        if !insert_valid_given(cpdag, x, y, &na, &t, footprint.as_deref_mut()) {
            continue;
        }
        let (base, neu) = insert_parent_sets(cpdag, x, y, &na, &t);
        if neu.len() > search.max_parents {
            continue;
        }
        out.push(InsertOperator { t, base, neu });
    }
    (out, truncated)
}

/// The valid Delete(X, Y, H) sets: `H ⊆ NA(Y,X)` with `NA \ H` a clique, smallest `H` first.
fn valid_deletes(
    cpdag: &Cpdag,
    x: DenseNodeId,
    y: DenseNodeId,
    search: &GesSearch<'_>,
) -> Vec<Vec<DenseNodeId>> {
    let na = na_yx(cpdag, y, x);
    let (kept, truncated) = clique_subsets(cpdag, &na, na.len(), search.subset_budget);
    search.note_pruned(truncated);
    let mut sets: Vec<Vec<DenseNodeId>> = kept
        .into_iter()
        .map(|rest| na.iter().copied().filter(|n| !rest.contains(n)).collect())
        .collect();
    sets.sort_by(|a: &Vec<DenseNodeId>, b: &Vec<DenseNodeId>| {
        a.len().cmp(&b.len()).then_with(|| a.iter().map(|n| n.raw()).cmp(b.iter().map(|n| n.raw())))
    });
    sets.retain(|h| delete_valid(cpdag, x, y, h));
    sets
}

fn best_insert(
    cpdag: &Cpdag,
    search: GesSearch<'_>,
    cache: &mut LocalScoreCache,
    deltas: &mut GesDeltaCache,
) -> Result<Option<OpCand>, DiscoveryError> {
    // The pair cache is moved out so the operator deltas can be filled while it is read.
    let mut pairs = std::mem::take(&mut deltas.pairs);
    let best = best_insert_with(cpdag, search, cache, deltas, &mut pairs);
    deltas.pairs = pairs;
    best
}

/// The best Insert over all ordered pairs, reusing each pair's result from the previous call
/// when no edge it read has changed (see [`InsertPairCache`]).
fn best_insert_with(
    cpdag: &Cpdag,
    search: GesSearch<'_>,
    cache: &mut LocalScoreCache,
    deltas: &mut GesDeltaCache,
    pairs: &mut InsertPairCache,
) -> Result<Option<OpCand>, DiscoveryError> {
    pairs.sync(cpdag);
    let n = cpdag.node_count();
    let mut best: Option<OpCand> = None;
    for xi in 0..n {
        let x = DenseNodeId::from_raw(crate::indexing::dense_u32(xi));
        let mut adj_x: Option<HashSet<DenseNodeId>> = None;
        for yi in 0..n {
            if xi == yi {
                continue;
            }
            let y = DenseNodeId::from_raw(crate::indexing::dense_u32(yi));
            if cpdag.has_edge(x, y)
                || !screened(search.screen, x, y)
                || forbidden_direction(search.constraints, search.variables, x, y)
            {
                continue;
            }
            let key = (x.raw(), y.raw());
            if !pairs.contains(key) {
                let adj_x = adj_x.get_or_insert_with(|| cpdag.adjacent(x).into_iter().collect());
                let mut footprint = Footprint::new(n);
                let (ops, truncated) =
                    collect_inserts(cpdag, x, y, adj_x, &search, Some(&mut footprint));
                let mut best_op: Option<(Vec<DenseNodeId>, f64)> = None;
                for op in ops {
                    let delta = cached_operator_delta(
                        cache,
                        deltas,
                        search.data,
                        y.raw(),
                        &op.base,
                        &op.neu,
                    )?;
                    if best_op.as_ref().is_none_or(|(_, d)| delta > *d) {
                        best_op = Some((op.t, delta));
                    }
                }
                pairs.store(key, PairInsert { best: best_op, truncated, footprint });
            }
            let Some(entry) = pairs.get(key) else {
                continue;
            };
            search.note_pruned(entry.truncated);
            if let Some((set, delta)) = &entry.best {
                if best.as_ref().is_none_or(|b| *delta > b.delta) {
                    best = Some(OpCand { x, y, set: set.clone(), delta: *delta });
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
            if required_pair(search.constraints, search.variables, x, y) {
                continue;
            }
            for h in valid_deletes(cpdag, x, y, &search) {
                let delta = delete_delta(cpdag, cache, deltas, search.data, x, y, &h)?;
                if best.as_ref().is_none_or(|b| delta > b.delta) {
                    best = Some(OpCand { x, y, set: h, delta });
                }
            }
        }
    }
    Ok(best)
}

/// The Insert(Y, X, T) operators of a reversal candidate on its post-delete graph, with the
/// score gain of each, and the edges the search read.
struct ReverseInsert {
    /// Edges of the post-delete graph incident to the nodes the Insert search read, sorted.
    read_edges: Vec<((u32, u32), (Endpoint, Endpoint))>,
    footprint: Footprint,
    /// Each valid `T` in search order with the delta of the Insert it names.
    ops: Vec<(Vec<DenseNodeId>, f64)>,
    truncated: bool,
}

/// Most reversal results kept across the iterations of the turning phase.
const REVERSE_CACHE_ENTRY_BUDGET: usize = 1 << 16;

/// Insert results of the reversal candidates `(X, Y, H)`, kept across turning iterations.
///
/// The post-delete graph of a candidate is a global function of the current CPDAG (the
/// conversion to a DAG and back can re-orient edges far from `X`, `Y`), so it is rebuilt every
/// iteration. The Insert search on it, which dominates the cost, reads only the edges incident to
/// its footprint, so its result is reused whenever those edges are the same in the rebuilt graph.
#[derive(Default)]
struct ReverseCache {
    entries: HashMap<(u32, u32, Vec<u32>), ReverseInsert>,
}

/// The edges of `cpdag` incident to a node of `footprint`, normalized and sorted.
fn edges_incident_to(
    cpdag: &Cpdag,
    footprint: &Footprint,
) -> Vec<((u32, u32), (Endpoint, Endpoint))> {
    let mut out = Vec::new();
    for e in cpdag.edges() {
        if !footprint.contains(e.a) && !footprint.contains(e.b) {
            continue;
        }
        let (lo, hi, at_lo, at_hi) = if e.a.raw() <= e.b.raw() {
            (e.a.raw(), e.b.raw(), e.at_a, e.at_b)
        } else {
            (e.b.raw(), e.a.raw(), e.at_b, e.at_a)
        };
        out.push(((lo, hi), (at_lo, at_hi)));
    }
    out.sort_unstable_by_key(|(key, _)| *key);
    out
}

fn best_reverse(
    cpdag: &Cpdag,
    search: GesSearch<'_>,
    cache: &mut LocalScoreCache,
    deltas: &mut GesDeltaCache,
) -> Result<Option<ReverseCand>, DiscoveryError> {
    let mut previous = std::mem::take(&mut deltas.reverses.entries);
    let mut kept: HashMap<(u32, u32, Vec<u32>), ReverseInsert> = HashMap::new();
    let best = best_reverse_with(cpdag, search, cache, deltas, &mut previous, &mut kept);
    // Only the candidates of this iteration are carried over, which bounds the cache by the
    // number of candidates a single search visits.
    deltas.reverses.entries = kept;
    best
}

fn best_reverse_with(
    cpdag: &Cpdag,
    search: GesSearch<'_>,
    cache: &mut LocalScoreCache,
    deltas: &mut GesDeltaCache,
    previous: &mut HashMap<(u32, u32, Vec<u32>), ReverseInsert>,
    kept: &mut HashMap<(u32, u32, Vec<u32>), ReverseInsert>,
) -> Result<Option<ReverseCand>, DiscoveryError> {
    let mut best: Option<ReverseCand> = None;
    let mut edges = cpdag.edges();
    edges.sort_unstable_by_key(|edge| (edge.a.raw(), edge.b.raw()));
    for e in edges {
        let Some((x, y)) = e.parent_child() else {
            continue;
        };
        if required_pair(search.constraints, search.variables, x, y)
            || !screened(search.screen, x, y)
            || forbidden_direction(search.constraints, search.variables, y, x)
        {
            continue;
        }
        for h in valid_deletes(cpdag, x, y, &search) {
            let d_del = delete_delta(cpdag, cache, deltas, search.data, x, y, &h)?;
            let mut tmp = cpdag.clone();
            apply_delete(&mut tmp, x, y, &h)?;
            let key = (x.raw(), y.raw(), h.iter().map(|n| n.raw()).collect::<Vec<u32>>());
            let reusable = previous
                .remove(&key)
                .filter(|entry| edges_incident_to(&tmp, &entry.footprint) == entry.read_edges);
            let entry = match reusable {
                Some(entry) => entry,
                None => {
                    // Insert opposite Y → X
                    let adj_y: HashSet<DenseNodeId> = tmp.adjacent(y).into_iter().collect();
                    let mut footprint = Footprint::new(tmp.node_count());
                    let (inserts, truncated) =
                        collect_inserts(&tmp, y, x, &adj_y, &search, Some(&mut footprint));
                    let mut ops = Vec::with_capacity(inserts.len());
                    for op in inserts {
                        let d_ins = cached_operator_delta(
                            cache,
                            deltas,
                            search.data,
                            x.raw(),
                            &op.base,
                            &op.neu,
                        )?;
                        ops.push((op.t, d_ins));
                    }
                    let read_edges = edges_incident_to(&tmp, &footprint);
                    ReverseInsert { read_edges, footprint, ops, truncated }
                }
            };
            search.note_pruned(entry.truncated);
            for (t, d_ins) in &entry.ops {
                let delta = d_del + d_ins;
                if best.as_ref().is_none_or(|b| delta > b.delta) {
                    best = Some(ReverseCand { x, y, h: h.clone(), t: t.clone(), delta });
                }
            }
            if kept.len() < REVERSE_CACHE_ENTRY_BUDGET {
                kept.insert(key, entry);
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
    let directed: Vec<(u32, u32)> = pdag
        .edges()
        .iter()
        .filter_map(|edge| edge.parent_child())
        .map(|(from, to)| (from.raw(), to.raw()))
        .collect();
    if let Err(refusal) = antecedent_graph::error::refuse_directed_cycle(
        antecedent_graph::error::DagOperation::DagProjection,
        antecedent_graph::error::CyclicNextAttempt::Condensation,
        pdag.node_count(),
        &directed,
    ) {
        return Err(DiscoveryError::Orientation(refusal.to_string()));
    }
    let mut work = pdag.clone();
    let n = work.node_count();
    let mut dag = Dag::empty();
    for node in work.nodes() {
        dag.add_node(*node)?;
    }
    // Copy already-directed edges. The check above accepted this directed set.
    for e in work.edges() {
        if let Some((from, to)) = e.parent_child() {
            dag.insert_directed(from, to)?;
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
                dag.insert_directed(n, x)?;
            }
        }
        remaining.remove(&x);
    }
    Ok(dag)
}

/// Essential graph: skeleton + DAG v-structures + Meek R1–R4.
pub(crate) fn dag_to_cpdag(dag: &Dag) -> Result<Cpdag, DiscoveryError> {
    dag_to_cpdag_with_known(dag, |_, _| false)
}

/// Essential graph under known orientations: the edges `known(from, to)` names start directed
/// (time order, for an unrolled window), then v-structures and Meek R1–R4 orient the rest.
pub(crate) fn dag_to_cpdag_with_known(
    dag: &Dag,
    known: impl Fn(DenseNodeId, DenseNodeId) -> bool,
) -> Result<Cpdag, DiscoveryError> {
    let mut cpdag = Cpdag::empty();
    for node in dag.nodes() {
        cpdag.add_node(*node)?;
    }
    for e in dag.edges() {
        if let Some((from, to)) = e.parent_child() {
            if !cpdag.has_edge(from, to) {
                if known(from, to) {
                    cpdag.insert_directed(from, to)?;
                } else {
                    cpdag.insert_undirected(from, to)?;
                }
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
        // Score of the chosen graph under the centered, compensated Gram. The cached search must
        // keep these bits.
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
            13_869_790_556_363_832_392
        );
        let d = |i: u32| DenseNodeId::from_raw(i);
        // Chain skeleton should keep 0—1 and 1—2; 0—2 may be absent.
        assert!(g.has_edge(d(0), d(1)), "expected edge 0-1");
        assert!(g.has_edge(d(1), d(2)), "expected edge 1-2");
    }

    /// Deterministic noise in `[-1, 1)` from a linear congruential stream.
    fn lcg_noise(n: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
            })
            .collect()
    }

    /// The per-pair cache must return exactly what a search from scratch returns at every step
    /// of the forward phase, including through the Meek re-orientations an insert triggers.
    #[test]
    fn pair_cache_matches_a_fresh_search_at_every_insert_step() {
        let n = 300;
        let e: Vec<Vec<f64>> = (0..6).map(|k| lcg_noise(n, 11 + k)).collect();
        let x0 = e[0].clone();
        let x1 = e[1].clone();
        let x2: Vec<f64> = (0..n).map(|i| 0.9 * x0[i] + 0.8 * x1[i] + 0.3 * e[2][i]).collect();
        let x3: Vec<f64> = (0..n).map(|i| 0.9 * x2[i] + 0.3 * e[3][i]).collect();
        let x4: Vec<f64> = (0..n).map(|i| 0.8 * x1[i] + 0.3 * e[4][i]).collect();
        let x5: Vec<f64> = (0..n).map(|i| 0.7 * x3[i] + 0.7 * x4[i] + 0.3 * e[5][i]).collect();
        let data = crate::test_tables::continuous_table(
            &["x0", "x1", "x2", "x3", "x4", "x5"],
            vec![x0, x1, x2, x3, x4, x5],
        );
        let vars: Vec<_> = data.schema().variables().iter().map(|v| v.id).collect();
        let score = ges_score_data(&data, &vars).unwrap();
        let key = || {
            crate::graph_score::graph_score_cache_key(
                GraphScoreFamily::GaussianBic,
                vars.len(),
                score.n_rows,
            )
        };
        let constraints = DiscoveryConstraints::default();
        let pruned = Cell::new(0);
        let search = GesSearch {
            data: &score,
            max_parents: vars.len() - 1,
            subset_budget: 1 << 12,
            screen: None,
            constraints: &constraints,
            variables: &vars,
            tolerance: 0.0,
            pruned: &pruned,
        };
        let mut cpdag = empty_cpdag(&vars).unwrap();
        let mut kept = LocalScoreCache::new(key());
        let mut kept_deltas = GesDeltaCache::default();
        let mut steps = 0;
        loop {
            let reused = best_insert(&cpdag, search, &mut kept, &mut kept_deltas).unwrap();
            let fresh = best_insert(
                &cpdag,
                search,
                &mut LocalScoreCache::new(key()),
                &mut GesDeltaCache::default(),
            )
            .unwrap();
            match (&reused, &fresh) {
                (None, None) => break,
                (Some(a), Some(b)) => {
                    assert_eq!((a.x, a.y), (b.x, b.y), "step {steps}");
                    assert_eq!(a.set, b.set, "step {steps}");
                    assert_eq!(a.delta.to_bits(), b.delta.to_bits(), "step {steps}");
                }
                _ => panic!("cached and fresh searches disagree at step {steps}"),
            }
            // The turning phase keeps its per-candidate results across graphs too.
            let turned = best_reverse(&cpdag, search, &mut kept, &mut kept_deltas).unwrap();
            let turned_fresh = best_reverse(
                &cpdag,
                search,
                &mut LocalScoreCache::new(key()),
                &mut GesDeltaCache::default(),
            )
            .unwrap();
            match (&turned, &turned_fresh) {
                (None, None) => {}
                (Some(a), Some(b)) => {
                    assert_eq!((a.x, a.y), (b.x, b.y), "turn step {steps}");
                    assert_eq!(a.h, b.h, "turn step {steps}");
                    assert_eq!(a.t, b.t, "turn step {steps}");
                    assert_eq!(a.delta.to_bits(), b.delta.to_bits(), "turn step {steps}");
                }
                _ => panic!("cached and fresh turning searches disagree at step {steps}"),
            }
            let Some(best) = reused else { break };
            if best.delta <= 0.0 {
                break;
            }
            apply_insert(&mut cpdag, best.x, best.y, &best.set).unwrap();
            steps += 1;
        }
        assert!(steps >= 3, "the search should have inserted several edges, got {steps}");
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
        let (_base, scored) = insert_parent_sets(&cpdag, x, y, &na_yx(&cpdag, y, x), &[]);
        assert_eq!(scored.len(), 2);
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
        let pruned = Cell::new(0);
        let search = GesSearch {
            data: &score,
            max_parents: 1,
            subset_budget: 1 << 12,
            screen: None,
            constraints: &constraints,
            variables: &vars,
            tolerance: 0.0,
            pruned: &pruned,
        };
        let best = best_insert(&cpdag, search, &mut cache, &mut GesDeltaCache::default()).unwrap();
        assert!(
            best.is_none(),
            "an insert whose scored parent set exceeds max_parents must be skipped"
        );
    }

    #[test]
    fn clique_subsets_are_exactly_the_cliques_smallest_first() {
        // Nodes 0..4; edges 0-1, 1-2, 0-2 (a triangle) and 3 isolated from it.
        let mut cpdag = Cpdag::with_variables(4);
        let d = DenseNodeId::from_raw;
        cpdag.insert_undirected(d(0), d(1)).unwrap();
        cpdag.insert_undirected(d(1), d(2)).unwrap();
        cpdag.insert_undirected(d(0), d(2)).unwrap();
        let nodes = [d(0), d(1), d(2), d(3)];
        let (sets, truncated) = clique_subsets(&cpdag, &nodes, 4, 1 << 12);
        assert!(!truncated);
        let raw: Vec<Vec<u32>> =
            sets.iter().map(|set| set.iter().map(|n| n.raw()).collect()).collect();
        assert_eq!(
            raw,
            vec![
                vec![],
                vec![0],
                vec![1],
                vec![2],
                vec![3],
                vec![0, 1],
                vec![0, 2],
                vec![1, 2],
                vec![0, 1, 2],
            ]
        );
        let (limited, truncated) = clique_subsets(&cpdag, &nodes, 4, 5);
        assert!(truncated, "nine cliques cannot fit a budget of five");
        assert_eq!(limited.len(), 5);
        let (exact, truncated) = clique_subsets(&cpdag, &nodes, 4, 9);
        assert!(!truncated, "a budget equal to the clique count is not a truncation");
        assert_eq!(exact.len(), 9);
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
    fn a_parent_limit_below_the_neighbour_count_restricts_the_search_without_refusing_it() {
        // The middle variable of the chain has two undirected neighbours in the class, more
        // than a limit of one: the subsets larger than the limit cannot be accepted, so the
        // search is complete over the subsets that can, and it still finds the chain.
        let mut limited = Ges::new().constraints;
        limited.max_parents = Some(1);
        let graph = ges_chain_under(limited);
        let node = DenseNodeId::from_raw;
        assert!(graph.has_edge(node(0), node(1)) && graph.has_edge(node(1), node(2)));
        assert!(!graph.has_edge(node(0), node(2)));
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
