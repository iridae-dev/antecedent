//! Greedy Equivalence Search (GES) → static [`Cpdag`] (Chickering 2002).
//!
//! Forward Insert / turning Reverse / backward Delete on the CPDAG, scored with
//! Gaussian BIC via [`causal_state::LocalScoreCache`]. Optional PC-skeleton
//! screening restricts Insert candidates.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use causal_core::{AssumptionSet, ExecutionContext, Lag, VariableId};
use causal_data::TabularData;
use causal_graph::{Cpdag, CpdagReview, Dag, DenseNodeId, Endpoint, NodeRef};
use causal_state::{GraphScoreCacheKey, GraphScoreData, GraphScoreFamily, LocalScoreCache};
use causal_stats::{ConditionalIndependence, FdrAdjustment, PartialCorrelation};

use crate::constraints::DiscoveryConstraints;
use crate::engine::DiscoveryWorkspace;
use crate::error::DiscoveryError;
use crate::orientation::{
    MeekR1, MeekR2, MeekR3, MeekR4, OrientationState, StaticOrientationRule,
    run_static_orientation_to_fixed_point,
};
use crate::pc::{Pc, StaticCpdagDiscoveryResult, collect_float_columns};
use crate::result::{
    AlgorithmRecord, DiscoveryDiagnostic, DiscoveryIteration, DiscoveryPerformanceRecord,
    DiscoveryResult, EdgeEvidence, EvidenceSource, GraphEvidence, LaggedLink, ScoredLink,
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

impl std::fmt::Debug for Ges {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ges")
            .field("constraints", &self.constraints)
            .field("ci", &"<dyn ConditionalIndependence>")
            .field("fdr", &self.fdr)
            .field("screen_pc", &self.screen_pc)
            .field("max_subset", &self.max_subset)
            .finish()
    }
}

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
    pub fn run(
        &self,
        data: &TabularData,
        variables: &[VariableId],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<StaticCpdagDiscoveryResult, DiscoveryError> {
        self.constraints.validate()?;
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
        let mut cache = LocalScoreCache::new(GraphScoreCacheKey {
            data_version: 1,
            family: GraphScoreFamily::GaussianBic,
            var_fingerprint: n_vars as u64,
            penalty_fingerprint: n as u64,
        });

        let screen: Option<HashSet<(u32, u32)>> = if self.screen_pc {
            Some(pc_skeleton_pairs(self, data, variables, workspace, ctx)?)
        } else {
            None
        };

        let mut cpdag = empty_cpdag(variables)?;
        seed_required_edges(&mut cpdag, variables, &self.constraints)?;

        let max_parents = self.constraints.max_parents.unwrap_or(n_vars.saturating_sub(1));
        let max_subset = max_parents.min(self.max_subset.unwrap_or(12));

        // Forward equivalence search (Insert).
        loop {
            let Some(best) = best_insert(
                &cpdag,
                &score_data,
                &mut cache,
                max_parents,
                max_subset,
                screen.as_ref(),
                &self.constraints,
                variables,
            )?
            else {
                break;
            };
            if best.delta <= 0.0 {
                break;
            }
            apply_insert(&mut cpdag, best.x, best.y, &best.set)?;
        }

        // Turning phase (Reverse = Delete then Insert opposite).
        loop {
            let Some(best) = best_reverse(
                &cpdag,
                &score_data,
                &mut cache,
                max_parents,
                max_subset,
                screen.as_ref(),
                &self.constraints,
                variables,
            )?
            else {
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
            let Some(best) = best_delete(
                &cpdag,
                &score_data,
                &mut cache,
                max_subset,
                &self.constraints,
                variables,
            )?
            else {
                break;
            };
            if best.delta <= 0.0 {
                break;
            }
            apply_delete(&mut cpdag, best.x, best.y, &best.set)?;
        }

        let total_score = {
            // Sync cache parents from a consistent DAG extension for reporting.
            let dag = pdag_to_dag(&cpdag)?;
            for node in 0..n_vars as u32 {
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
                    separating_sets: Arc::from([]),
                    provenance: Arc::from([Arc::from("ges")]),
                })
            })
            .collect();

        let links: Vec<ScoredLink> = edge_evidence
            .iter()
            .map(|e| ScoredLink {
                link: e.link,
                statistic: 0.0,
                p_value: 1.0,
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

        let _ = workspace;
        Ok(DiscoveryResult {
            evidence,
            review,
            algorithm: AlgorithmRecord {
                id: Arc::from("ges"),
                config: Arc::from(format!(
                    "family=gaussian_bic screen_pc={} score={total_score:.6}",
                    self.screen_pc
                )),
            },
            assumptions: AssumptionSet::default(),
            iterations: Vec::<DiscoveryIteration>::new(),
            diagnostics: Vec::<DiscoveryDiagnostic>::new(),
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
    let index: HashMap<VariableId, DenseNodeId> =
        variables.iter().enumerate().map(|(i, v)| (*v, DenseNodeId::from_raw(i as u32))).collect();
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
    let pc = Pc::new()
        .with_constraints(ges.constraints.clone())
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

fn forbidden_pair(
    constraints: &DiscoveryConstraints,
    variables: &[VariableId],
    a: DenseNodeId,
    b: DenseNodeId,
) -> bool {
    let Some(va) = variables.get(a.as_usize()).copied() else {
        return true;
    };
    let Some(vb) = variables.get(b.as_usize()).copied() else {
        return true;
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
    constraints.is_forbidden(link_ab)
        || constraints.is_forbidden(link_ba)
        || constraints.tier_forbids(va, vb)
        || constraints.tier_forbids(vb, va)
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
    cpdag.undirected_neighbors(y).into_iter().filter(|&n| cpdag.has_edge(n, x)).collect()
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

fn parents_u32(cpdag: &Cpdag, y: DenseNodeId) -> Vec<u32> {
    cpdag.parents(y).into_iter().map(causal_graph::DenseNodeId::raw).collect()
}

fn insert_delta(
    cpdag: &Cpdag,
    cache: &mut LocalScoreCache,
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
    let old = cache.local_score(data, y.raw(), &Arc::from(base))?;
    let new = cache.local_score(data, y.raw(), &Arc::from(neu))?;
    Ok(new - old)
}

fn delete_delta(
    cpdag: &Cpdag,
    cache: &mut LocalScoreCache,
    data: &GraphScoreData,
    x: DenseNodeId,
    y: DenseNodeId,
    h: &[DenseNodeId],
) -> Result<f64, DiscoveryError> {
    // Corollary 18: s(Y, (NA\H) ∪ (Pa\{X})) − s(Y, (NA\H) ∪ Pa)
    let na = na_yx(cpdag, y, x);
    let h_set: HashSet<DenseNodeId> = h.iter().copied().collect();
    let na_rest: Vec<u32> =
        na.into_iter().filter(|n| !h_set.contains(n)).map(causal_graph::DenseNodeId::raw).collect();
    let mut pa = parents_u32(cpdag, y);
    let mut with_x = pa.clone();
    with_x.extend_from_slice(&na_rest);
    with_x.sort_unstable();
    with_x.dedup();
    pa.retain(|&p| p != x.raw());
    let mut without_x = pa;
    without_x.extend_from_slice(&na_rest);
    without_x.sort_unstable();
    without_x.dedup();
    let old = cache.local_score(data, y.raw(), &Arc::from(with_x))?;
    let new = cache.local_score(data, y.raw(), &Arc::from(without_x))?;
    Ok(new - old)
}

fn for_each_subset(items: &[DenseNodeId], max_k: usize, mut visit: impl FnMut(&[DenseNodeId])) {
    let m = items.len().min(max_k);
    let mut scratch = Vec::new();
    for k in 0..=m {
        if k == 0 {
            visit(&[]);
            continue;
        }
        if k > items.len() {
            break;
        }
        scratch.resize(k, DenseNodeId::from_raw(0));
        let mut idx: Vec<usize> = (0..k).collect();
        loop {
            for (slot, &i) in idx.iter().enumerate() {
                scratch[slot] = items[i];
            }
            visit(&scratch);
            let mut i = k;
            let mut advanced = false;
            while i > 0 {
                i -= 1;
                if idx[i] != i + items.len() - k {
                    idx[i] += 1;
                    for j in i + 1..k {
                        idx[j] = idx[j - 1] + 1;
                    }
                    advanced = true;
                    break;
                }
            }
            if !advanced {
                break;
            }
        }
    }
}

fn best_insert(
    cpdag: &Cpdag,
    data: &GraphScoreData,
    cache: &mut LocalScoreCache,
    max_parents: usize,
    max_subset: usize,
    screen: Option<&HashSet<(u32, u32)>>,
    constraints: &DiscoveryConstraints,
    variables: &[VariableId],
) -> Result<Option<OpCand>, DiscoveryError> {
    let n = cpdag.node_count();
    let mut best: Option<OpCand> = None;
    for xi in 0..n {
        let x = DenseNodeId::from_raw(xi as u32);
        for yi in 0..n {
            if xi == yi {
                continue;
            }
            let y = DenseNodeId::from_raw(yi as u32);
            if cpdag.has_edge(x, y) {
                continue;
            }
            if !screened(screen, x, y) {
                continue;
            }
            if forbidden_pair(constraints, variables, x, y) {
                continue;
            }
            // T ⊆ Ne(Y) \ adj(X)
            let adj_x: HashSet<DenseNodeId> = cpdag.adjacent(x).into_iter().collect();
            let candidates: Vec<DenseNodeId> =
                cpdag.undirected_neighbors(y).into_iter().filter(|n| !adj_x.contains(n)).collect();
            let mut subsets: Vec<Vec<DenseNodeId>> = Vec::new();
            for_each_subset(&candidates, max_subset, |t| subsets.push(t.to_vec()));
            for t in subsets {
                if !insert_valid(cpdag, x, y, &t) {
                    continue;
                }
                let pa_len = parents_u32(cpdag, y).len() + 1 + t.len();
                if pa_len > max_parents {
                    continue;
                }
                let delta = insert_delta(cpdag, cache, data, x, y, &t)?;
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
    data: &GraphScoreData,
    cache: &mut LocalScoreCache,
    max_subset: usize,
    constraints: &DiscoveryConstraints,
    variables: &[VariableId],
) -> Result<Option<OpCand>, DiscoveryError> {
    let mut best: Option<OpCand> = None;
    for e in cpdag.edges() {
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
            let mut subsets: Vec<Vec<DenseNodeId>> = Vec::new();
            for_each_subset(&na, max_subset, |h| subsets.push(h.to_vec()));
            for h in subsets {
                if !delete_valid(cpdag, x, y, &h) {
                    continue;
                }
                let delta = delete_delta(cpdag, cache, data, x, y, &h)?;
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
    data: &GraphScoreData,
    cache: &mut LocalScoreCache,
    max_parents: usize,
    max_subset: usize,
    screen: Option<&HashSet<(u32, u32)>>,
    constraints: &DiscoveryConstraints,
    variables: &[VariableId],
) -> Result<Option<ReverseCand>, DiscoveryError> {
    let mut best: Option<ReverseCand> = None;
    for e in cpdag.edges() {
        let Some((x, y)) = e.parent_child() else {
            continue;
        };
        if required_pair(constraints, variables, x, y) {
            continue;
        }
        if !screened(screen, x, y) {
            continue;
        }
        if forbidden_pair(constraints, variables, y, x) {
            continue;
        }
        let na = na_yx(cpdag, y, x);
        let mut h_subsets: Vec<Vec<DenseNodeId>> = Vec::new();
        for_each_subset(&na, max_subset, |h| h_subsets.push(h.to_vec()));
        for h in h_subsets {
            if !delete_valid(cpdag, x, y, &h) {
                continue;
            }
            let d_del = delete_delta(cpdag, cache, data, x, y, &h)?;
            let mut tmp = cpdag.clone();
            apply_delete(&mut tmp, x, y, &h)?;
            // Insert opposite Y → X
            let adj_y: HashSet<DenseNodeId> = tmp.adjacent(y).into_iter().collect();
            let candidates: Vec<DenseNodeId> =
                tmp.undirected_neighbors(x).into_iter().filter(|n| !adj_y.contains(n)).collect();
            let mut t_subsets: Vec<Vec<DenseNodeId>> = Vec::new();
            for_each_subset(&candidates, max_subset, |t| t_subsets.push(t.to_vec()));
            for t in t_subsets {
                if !insert_valid(&tmp, y, x, &t) {
                    continue;
                }
                let pa_len = parents_u32(&tmp, x).len() + 1 + t.len();
                if pa_len > max_parents {
                    continue;
                }
                let d_ins = insert_delta(&tmp, cache, data, y, x, &t)?;
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

    let mut remaining: HashSet<DenseNodeId> = (0..n as u32).map(DenseNodeId::from_raw).collect();
    while !remaining.is_empty() {
        // Select x ∈ remaining: no directed edge out of x to remaining,
        // and undirected neighbors (in remaining) form a clique.
        let mut selected = None;
        for &x in &remaining {
            let out_to_remaining = work.children(x).into_iter().any(|c| remaining.contains(&c));
            if out_to_remaining {
                continue;
            }
            let und_nbrs: Vec<DenseNodeId> = work
                .undirected_neighbors(x)
                .into_iter()
                .filter(|n| remaining.contains(n))
                .collect();
            // Clique among {und_nbrs} in the full PDAG adjacency among remaining.
            let mut ok = true;
            for i in 0..und_nbrs.len() {
                for j in i + 1..und_nbrs.len() {
                    if !work.has_edge(und_nbrs[i], und_nbrs[j]) {
                        ok = false;
                        break;
                    }
                }
                if !ok {
                    break;
                }
            }
            if ok {
                selected = Some(x);
                break;
            }
        }
        let Some(x) = selected else {
            // Fallback: orient remaining undirected arbitrarily in a topo-safe way.
            return pdag_to_dag_fallback(&work, &dag, &remaining);
        };
        for n in work.undirected_neighbors(x) {
            if remaining.contains(&n) {
                work.orient_undirected(x, n)?;
                let _ = dag.insert_directed(x, n);
            }
        }
        remaining.remove(&x);
    }
    Ok(dag)
}

fn pdag_to_dag_fallback(
    work: &Cpdag,
    dag: &Dag,
    remaining: &HashSet<DenseNodeId>,
) -> Result<Dag, DiscoveryError> {
    let mut dag = dag.clone();
    let mut work = work.clone();
    // Orient undirected edges among remaining by endpoint order when acyclic.
    let mut pairs: Vec<(DenseNodeId, DenseNodeId)> = Vec::new();
    for e in work.edges() {
        if e.is_undirected() && remaining.contains(&e.a) && remaining.contains(&e.b) {
            pairs.push((e.a, e.b));
        }
    }
    for (a, b) in pairs {
        if !work.has_edge(a, b) {
            continue;
        }
        if let Some(e) = work.edge_between(a, b) {
            if !e.is_undirected() {
                continue;
            }
        }
        // Prefer lower → higher if acyclic.
        if dag.insert_directed(a, b).is_ok() {
            let _ = work.orient_undirected(a, b);
        } else if dag.insert_directed(b, a).is_ok() {
            let _ = work.orient_undirected(b, a);
        } else {
            return Err(DiscoveryError::Unsupported {
                message: "GES PDAG has no consistent DAG extension",
            });
        }
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
        let z = DenseNodeId::from_raw(zi as u32);
        let parents = dag.parents(z);
        for i in 0..parents.len() {
            for j in i + 1..parents.len() {
                let p = parents[i];
                let q = parents[j];
                if !dag.has_edge(p, q) && !dag.has_edge(q, p) {
                    // p → z ← q
                    if cpdag.edge_between(p, z).is_some_and(causal_graph::MarkedEdge::is_undirected)
                    {
                        cpdag.orient_undirected(p, z)?;
                    }
                    if cpdag.edge_between(q, z).is_some_and(causal_graph::MarkedEdge::is_undirected)
                    {
                        cpdag.orient_undirected(q, z)?;
                    }
                }
            }
        }
    }
    let mut state = OrientationState::default();
    let rules: [&dyn StaticOrientationRule; 4] = [&MeekR1, &MeekR2, &MeekR3, &MeekR4];
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
    use causal_core::{
        CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType, VariableId,
    };
    use causal_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TableView, TabularData, ValidityBitmap,
    };

    fn gaussian_chain_data(n: usize) -> (TabularData, Vec<VariableId>) {
        // X0 → X1 → X2 linear Gaussian.
        let mut b = CausalSchemaBuilder::new();
        for i in 0..3 {
            b.add_variable(
                format!("x{i}"),
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        }
        let schema = b.build().unwrap();
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
        let owned = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(x0),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(x1),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(2),
                    Arc::from(x2),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, owned, None, None).unwrap();
        let data = TabularData::new(storage);
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
        let d = |i: u32| DenseNodeId::from_raw(i);
        // Chain skeleton should keep 0—1 and 1—2; 0—2 may be absent.
        assert!(g.has_edge(d(0), d(1)), "expected edge 0-1");
        assert!(g.has_edge(d(1), d(2)), "expected edge 1-2");
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
}
