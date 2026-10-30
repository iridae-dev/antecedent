//! Structure MCMC for DAG posteriors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{CausalRng, ExecutionContext, StreamDomain, VariableId};
use antecedent_data::TabularData;
use antecedent_state::GraphScoreFamily;

use crate::adjacency_mask::AdjacencyMask;
use crate::adjacency_mask::{mask_is_dag, parent_bits};
use crate::engine::DiscoveryWorkspace;
use crate::error::DiscoveryError;
use crate::graph_mcmc::{
    DEFAULT_TEMPERATURES, EQUIVALENCE_WALK_MIN_NODES, EQUIVALENCE_WALK_PROBABILITY,
    FinishMaskPosterior, GraphMcmcSchedule, MaskAccept, MaskCancel, MaskChain, MaskChainPlan,
    accept_covered_reversal, propose_covered_reversal, reversed_edge, run_parallel_mask_chains,
    run_tempered_mask_chain, tempering_ladder, validate_temperatures,
};
use crate::graph_posterior::{
    GraphPosterior, GraphPosteriorEngine, GraphPrior, GraphPriorTable, edge_bit,
};
use crate::graph_score::{
    ChainTarget, ScoredTarget, SharedTarget, initial_scored_mask, tabular_score_data,
};

/// Probability that an iteration resamples the parent sets of a random node pair exactly.
pub const DEFAULT_PAIR_MOVE_PROBABILITY: f64 = 0.1;

/// Most parent sets per node for which the paired move is enabled: it enumerates them.
const PAIR_MOVE_MAX_SETS: usize = 4096;

/// Structure MCMC with single-edge add / delete / reverse proposals, an exact paired
/// parent-set resampling move, and parallel tempering.
///
/// The paired move picks two nodes `u`, `v` uniformly and redraws their parent sets jointly from
/// the full conditional of the target given every other node's parents (all sets the constraints
/// allow, minus those that would close a cycle). It contains reversals of an edge between `u`
/// and `v` *together with* the changes of parents that reversal needs (the idea of the
/// new-edge-reversal move of Grzegorczyk and Husmeier, 2008), so structures that differ in the
/// orientation of several interacting edges are one move apart. Being a draw from a conditional
/// it is always accepted and leaves the target invariant. It is disabled when a node has more
/// than 4096 allowed parent sets (set `max_parents`).
///
/// Above eight variables a quarter of the iterations instead reverse a uniformly chosen covered
/// edge (one whose parents are its child's parents less itself), accepted with the
/// Metropolis–Hastings ratio of the move. Reversing a covered edge stays inside the Markov
/// equivalence class, whose members the Gaussian-BIC score does not tell apart, so the move
/// walks the class at the acceptance of a plateau; the orientation of an edge the data leave
/// open is otherwise decided only by the rare proposal that reverses a covered edge, and its
/// chain diagnostics mix at the pace of that rare event.
///
/// Replicas at the inverse temperatures of [`Self::temperatures`] run in every chain and
/// exchange temperatures (replica exchange), which carries chains across the local modes
/// single-edge moves cannot leave; only the replica at temperature 1 is recorded.
///
/// Parent sets are node sets of any width, so the node count is bounded only by the edge bit
/// index ([`crate::adjacency_mask::edge_bits_fit`]). Cap the parents of any node with
/// `GraphPrior::constraints.max_parents`; the diagnostics gate reads per-edge split R-hat, ESS and
/// Monte-Carlo error across chains.
#[derive(Clone, Debug)]
pub struct StructureMcmc {
    /// Number of chains (≥ 2 for R-hat).
    pub n_chains: u32,
    /// Warmup draws discarded per chain.
    pub n_warmup: u32,
    /// Post-warmup draws retained per chain (before thinning).
    pub n_draws: u32,
    /// Keep every `thin`-th post-warmup draw.
    pub thin: u32,
    /// Optional undirected candidate pairs `(lo, hi)` from CI screening.
    pub candidate_pairs: Option<Arc<[(u32, u32)]>>,
    /// When true (default), refuse if [`crate::graph_posterior::allows_graph_posterior`] fails.
    pub require_diagnostics_gate: bool,
    /// Inverse-temperature ladder of the replicas in every chain: `1 = b_0 > b_1 > ... > 0`
    /// (default `[1, 1/2, 1/4, 1/8]`, run as a denser eight-replica ladder from nine variables on;
    /// `[1]` is a single untempered chain).
    pub temperatures: Vec<f64>,
    /// Probability that an iteration is a paired parent-set resampling instead of a
    /// single-edge proposal, in `[0, 1]`.
    pub pair_move_probability: f64,
}

impl Default for StructureMcmc {
    fn default() -> Self {
        Self::new()
    }
}

impl StructureMcmc {
    /// Default sampler (4 chains × 500 warmup × 1000 draws, thin 1).
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_chains: 4,
            n_warmup: 500,
            n_draws: 1000,
            thin: 1,
            candidate_pairs: None,
            require_diagnostics_gate: true,
            temperatures: DEFAULT_TEMPERATURES.to_vec(),
            pair_move_probability: DEFAULT_PAIR_MOVE_PROBABILITY,
        }
    }

    /// Set the inverse-temperature ladder (validated when the sampler runs).
    #[must_use]
    pub fn with_temperatures(mut self, temperatures: impl Into<Vec<f64>>) -> Self {
        self.temperatures = temperatures.into();
        self
    }

    /// Set the probability of the paired parent-set resampling move.
    #[must_use]
    pub fn with_pair_move_probability(mut self, probability: f64) -> Self {
        self.pair_move_probability = probability;
        self
    }

    /// Configure chain lengths.
    #[must_use]
    pub fn with_schedule(mut self, n_chains: u32, n_warmup: u32, n_draws: u32, thin: u32) -> Self {
        let s = GraphMcmcSchedule::normalize(n_chains, n_warmup, n_draws, thin);
        self.n_chains = s.n_chains;
        self.n_warmup = s.n_warmup;
        self.n_draws = s.n_draws;
        self.thin = s.thin;
        self
    }

    /// Restrict proposals to undirected candidate pairs.
    #[must_use]
    pub fn with_candidate_pairs(mut self, pairs: impl Into<Arc<[(u32, u32)]>>) -> Self {
        self.candidate_pairs = Some(pairs.into());
        self
    }

    /// Enable / disable the multi-chain diagnostics publish gate.
    #[must_use]
    pub fn with_diagnostics_gate(mut self, require: bool) -> Self {
        self.require_diagnostics_gate = require;
        self
    }

    fn schedule(&self) -> GraphMcmcSchedule {
        GraphMcmcSchedule {
            n_chains: self.n_chains,
            n_warmup: self.n_warmup,
            n_draws: self.n_draws,
            thin: self.thin,
        }
    }

    /// Run structure MCMC.
    ///
    /// # Errors
    ///
    /// Data/score failures or diagnostics gate refusal.
    #[allow(
        clippy::too_many_lines,
        reason = "one algorithm pipeline over shared mutable graph state; the phases are marked in place"
    )]
    pub fn run(
        &self,
        data: &TabularData,
        variables: &[VariableId],
        prior: &GraphPrior,
        score_family: GraphScoreFamily,
        _workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<GraphPosterior, DiscoveryError> {
        prior.validate_edge_inclusion()?;
        prior.constraints.validate()?;
        let n = variables.len();
        if n == 0 {
            return Err(DiscoveryError::unsupported(
                "structure MCMC requires at least one variable",
            ));
        }
        crate::graph_posterior::ensure_nodes_fit(
            n,
            "structure MCMC edge bit index exceeds u32 for this many variables",
        )?;
        if !matches!(score_family, GraphScoreFamily::GaussianBic) {
            return Err(DiscoveryError::unsupported(
                "structure MCMC currently supports GaussianBic only",
            ));
        }
        let schedule = self.schedule();
        schedule.require_min_chains(2, "structure MCMC requires at least 2 chains for R-hat")?;
        validate_temperatures(&self.temperatures)?;
        let ladder = tempering_ladder(&self.temperatures, n);
        if !(0.0..=1.0).contains(&self.pair_move_probability) {
            return Err(DiscoveryError::unsupported(
                "structure MCMC pair move probability must lie in [0, 1]",
            ));
        }

        let score_data = tabular_score_data(data, variables)?;
        let pairs = candidate_directed_list(n, self.candidate_pairs.as_deref());
        if pairs.is_empty() {
            return Err(DiscoveryError::unsupported(
                "structure MCMC has empty candidate edge list",
            ));
        }

        let (initial_mask, initial_score) =
            initial_scored_mask(n, &score_data, score_family, prior, variables)?;
        let prior_table = GraphPriorTable::new(n, prior, variables);
        let catalog = if self.pair_move_probability > 0.0 {
            ParentSetCatalog::build(&prior_table, self.candidate_pairs.as_deref())
        } else {
            None
        };

        let (n_chains, n_warmup, n_draws, thin) = schedule.as_usize();
        let n_params = pairs.len();

        let (traces, sample_masks, rejected, scores) =
            run_parallel_mask_chains(ctx, n_chains, n_draws, n_params, |chain, inner| {
                let mut rng = inner.rng.stream_for(StreamDomain::McmcStructure, chain as u64);
                let target = SharedTarget(std::rc::Rc::new(std::cell::RefCell::new(
                    ScoredTarget::new(n, &score_data, score_family, &prior_table),
                )));
                let mut replicas: Vec<_> = ladder
                    .iter()
                    .map(|_| StructureWalker {
                        n,
                        pairs: &pairs,
                        target: target.clone(),
                        catalog: catalog.as_ref(),
                        pair_probability: self.pair_move_probability,
                        mask: initial_mask.clone(),
                        cur: initial_score,
                        pending: initial_mask.clone(),
                    })
                    .collect();
                let run = run_tempered_mask_chain(
                    &MaskChainPlan { chain_index: chain, n_warmup, n_draws, thin, n_params },
                    inner,
                    &mut rng,
                    &mut replicas,
                    &ladder,
                )?;
                Ok((run.traces, run.samples, run.rejected, run.scores))
            })?;
        if ctx.cancellation.is_cancelled() {
            return Err(DiscoveryError::Cancelled);
        }

        FinishMaskPosterior {
            n,
            schedule: &schedule,
            traces: &traces,
            scores: &scores,
            sample_masks: &sample_masks,
            rejected,
            n_params,
            param_edges: &pairs,
            require_gate: self.require_diagnostics_gate,
            refuse_msg: "structure MCMC diagnostics gate refused posterior",
            empty_msg: "structure MCMC produced no samples",
        }
        .run()
    }
}

impl GraphPosteriorEngine for StructureMcmc {
    fn infer_graphs(
        &self,
        data: &TabularData,
        variables: &[VariableId],
        prior: &GraphPrior,
        score_family: GraphScoreFamily,
        ctx: &ExecutionContext,
    ) -> Result<GraphPosterior, DiscoveryError> {
        let mut ws = DiscoveryWorkspace::default();
        self.run(data, variables, prior, score_family, &mut ws, ctx)
    }
}

fn candidate_directed_list(n: usize, undirected: Option<&[(u32, u32)]>) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    match undirected {
        None => {
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        out.push((i, j));
                    }
                }
            }
        }
        Some(pairs) => {
            for &(lo, hi) in pairs {
                let a = lo as usize;
                let b = hi as usize;
                if a < n && b < n && a != b {
                    out.push((a, b));
                    out.push((b, a));
                }
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Hastings factor `q(old|new) / q(new|old)` for the symmetric directed-pair proposal.
///
/// The candidate list contains both orientations of every pair. An add has probability
/// `1/|pairs|`; its reverse delete has the same total probability because either orientation
/// can select the existing edge and each takes the delete branch with probability 1/2.
/// Reversals are symmetric too, so every valid structure move has ratio 1.
#[must_use]
pub(crate) fn structure_kernel_ratio(_move_kind: StructureMove) -> f64 {
    1.0
}

/// Which edit [`propose_structure`] attempted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum StructureMove {
    /// Absent pair → one directed edge.
    Add,
    /// Directed edge → absent.
    Delete,
    /// Directed edge → the opposite direction.
    Reverse,
}

/// Every parent set the constraints allow for each node, restricted to a node's candidate
/// neighbours when the search is restricted to candidate pairs.
struct ParentSetCatalog {
    sets: Vec<Vec<AdjacencyMask>>,
}

impl ParentSetCatalog {
    /// `None` when some node has more than [`PAIR_MOVE_MAX_SETS`] allowed sets.
    fn build(table: &GraphPriorTable, candidate_pairs: Option<&[(u32, u32)]>) -> Option<Self> {
        let n = table.n();
        // Per node, the nodes its parents may come from; `None` is every node.
        let neighbours: Option<Vec<AdjacencyMask>> = candidate_pairs.map(|pairs| {
            let mut near = vec![AdjacencyMask::EMPTY; n];
            for &(lo, hi) in pairs {
                let (a, b) = (lo as usize, hi as usize);
                if a < n && b < n && a != b {
                    near[a].set_bit(hi, true);
                    near[b].set_bit(lo, true);
                }
            }
            near
        });
        let sets = (0..n)
            .map(|node| {
                table.allowed_parent_sets(node, PAIR_MOVE_MAX_SETS).map(|all| {
                    all.into_iter()
                        .filter(|set| {
                            neighbours.as_ref().is_none_or(|near| near[node].contains_all(set))
                        })
                        .collect()
                })
            })
            .collect::<Option<Vec<Vec<AdjacencyMask>>>>()?;
        Some(Self { sets })
    }
}

/// One option of a node in the paired move: parent set, tempered log target, and whether the
/// set reaches the other node of the pair.
struct PairOption {
    set: AdjacencyMask,
    log_weight: f64,
    reaches: bool,
}

/// The options of one node in one reach class, summarised for stable sampling: the best log
/// weight and the sum of `exp(log_weight - best)`.
#[derive(Clone, Copy)]
struct ReachClass {
    best: f64,
    total: f64,
}

impl ReachClass {
    fn of(options: &[PairOption], reaches: bool) -> Self {
        let best = options
            .iter()
            .filter(|o| o.reaches == reaches)
            .map(|o| o.log_weight)
            .fold(f64::NEG_INFINITY, f64::max);
        let total = options
            .iter()
            .filter(|o| o.reaches == reaches)
            .map(|o| (o.log_weight - best).exp())
            .sum();
        Self { best, total }
    }

    fn is_empty(self) -> bool {
        self.best == f64::NEG_INFINITY
    }

    /// Log of the class's total weight.
    fn log_mass(self) -> f64 {
        self.best + self.total.ln()
    }
}

/// One structure-MCMC replica. `target` is the log target of a mask: BIC + log prior in a run,
/// and whatever a test needs to isolate the kernel.
struct StructureWalker<'a, T> {
    n: usize,
    pairs: &'a [(usize, usize)],
    target: T,
    catalog: Option<&'a ParentSetCatalog>,
    pair_probability: f64,
    mask: AdjacencyMask,
    cur: f64,
    pending: AdjacencyMask,
}

impl<T: ChainTarget> StructureWalker<'_, T> {
    /// Metropolis–Hastings step that reverses a uniformly chosen covered edge at inverse
    /// temperature `beta`. Reversing a covered edge keeps the graph acyclic.
    fn reverse_covered_edge(&mut self, beta: f64, rng: &mut CausalRng) {
        let Some((x, y, covered)) = propose_covered_reversal(&self.mask, self.n, rng) else {
            return;
        };
        let proposal = reversed_edge(&self.mask, self.n, x, y);
        debug_assert!(mask_is_dag(&proposal, self.n));
        let Some(score) = self.target.score(&proposal) else {
            return;
        };
        if accept_covered_reversal((beta, self.cur, score), covered, &proposal, self.n, rng) {
            self.mask = proposal;
            self.cur = score;
        }
    }

    /// Nodes reachable from `from` along child edges, `child[x]` being the bitset of `x`'s children.
    fn descendants(child: &[AdjacencyMask], from: usize) -> AdjacencyMask {
        let mut reach = AdjacencyMask::EMPTY;
        let mut frontier = child[from].clone();
        loop {
            let fresh = frontier.difference(&reach);
            if fresh.is_empty() {
                return reach;
            }
            reach = reach.union(&fresh);
            frontier = fresh
                .set_bits()
                .filter(|&y| (y as usize) < child.len())
                .fold(AdjacencyMask::EMPTY, |f, y| f.union(&child[y as usize]));
        }
    }

    /// The options of `node` whose other pair member is `other`. `own` are the nodes the new
    /// parents may not include (they would close a cycle through `node` alone); `across` are
    /// the nodes from which `other` is reached.
    fn pair_options(
        &mut self,
        catalog: &ParentSetCatalog,
        (node, other): (usize, usize),
        (own, across): (&AdjacencyMask, &AdjacencyMask),
        beta: f64,
    ) -> Vec<PairOption> {
        let other_bit = crate::indexing::dense_u32(other);
        let mut options = Vec::new();
        for set in &catalog.sets[node] {
            if set.intersects(own) {
                continue;
            }
            if let Some(log_target) = self.target.node_score(node, set) {
                options.push(PairOption {
                    reaches: set.intersects(across) || set.bit(other_bit),
                    set: set.clone(),
                    log_weight: beta * log_target,
                });
            }
        }
        options
    }

    /// Redraw the parent sets of `u` and `v` jointly from their full conditional at inverse
    /// temperature `beta` given everybody else's parents.
    ///
    /// With the incoming edges of `u` and `v` removed, `u` and `v` are sources; a choice of
    /// parent sets `(A, B)` closes a cycle exactly when `A` contains a descendant of `u`, or `B`
    /// one of `v`, or both `A` reaches `v` and `B` reaches `u` (each through the other's
    /// descendants). Weights factorise over the two nodes, so the pair is drawn in three stages:
    /// the reach classes of `(A, B)` (not both reaching), then `A` and `B` within their classes.
    /// Returns whether the sampled graph was installed (always, for a finitely scored target).
    fn resample_pair(
        &mut self,
        catalog: &ParentSetCatalog,
        (u, v): (usize, usize),
        beta: f64,
        rng: &mut CausalRng,
    ) -> bool {
        let n = self.n;
        let mut child = vec![AdjacencyMask::EMPTY; n];
        for node in (0..n).filter(|&node| node != u && node != v) {
            let node_bit = crate::indexing::dense_u32(node);
            for p in parent_bits(&self.mask, n, node).set_bits() {
                child[p as usize].set_bit(node_bit, true);
            }
        }
        let (below_u, below_v) = (Self::descendants(&child, u), Self::descendants(&child, v));
        let a = self.pair_options(catalog, (u, v), (&below_u, &below_v), beta);
        let b = self.pair_options(catalog, (v, u), (&below_v, &below_u), beta);
        // Classes (a reaches, b reaches) that close no cycle: not both. Their masses are compared
        // in log space, so a pair far below an excluded best pair still has a usable weight.
        let (ca, cb) = (
            [ReachClass::of(&a, false), ReachClass::of(&a, true)],
            [ReachClass::of(&b, false), ReachClass::of(&b, true)],
        );
        let admissible = [(0usize, 0usize), (0, 1), (1, 0)];
        let log_mass = |(i, j): (usize, usize)| {
            if ca[i].is_empty() || cb[j].is_empty() {
                f64::NEG_INFINITY
            } else {
                ca[i].log_mass() + cb[j].log_mass()
            }
        };
        let top = admissible.iter().map(|&c| log_mass(c)).fold(f64::NEG_INFINITY, f64::max);
        if top == f64::NEG_INFINITY {
            return false;
        }
        let masses: Vec<f64> = admissible.iter().map(|&c| (log_mass(c) - top).exp()).collect();
        let total: f64 = masses.iter().sum();
        let mut pick = rng.next_f64() * total;
        let mut class = admissible[0];
        for (&c, &m) in admissible.iter().zip(&masses) {
            if m > 0.0 {
                class = c;
                if pick < m {
                    break;
                }
                pick -= m;
            }
        }
        let chosen_a = draw(&a, class.0 == 1, ca[class.0], rng.next_f64());
        let chosen_b = draw(&b, class.1 == 1, cb[class.1], rng.next_f64());
        let mut mask = self.mask.clone();
        for (node, set) in [(u, &chosen_a.set), (v, &chosen_b.set)] {
            for p in (0..n).filter(|&p| p != node) {
                mask.set_bit(edge_bit(n, p, node), set.bit(crate::indexing::dense_u32(p)));
            }
        }
        let Some(score) = self.target.score(&mask) else {
            return false;
        };
        self.mask = mask;
        self.cur = score;
        true
    }
}

/// The option of reach class `reaches` that the uniform draw `x` in `[0, 1)` selects.
fn draw(options: &[PairOption], reaches: bool, class: ReachClass, x: f64) -> &PairOption {
    let mut remaining = x * class.total;
    let mut last = &options[0];
    for option in options.iter().filter(|o| o.reaches == reaches) {
        let w = (option.log_weight - class.best).exp();
        if w > 0.0 {
            last = option;
            if remaining < w {
                return option;
            }
            remaining -= w;
        }
    }
    last
}

impl<T: ChainTarget> MaskChain for StructureWalker<'_, T> {
    type Sample = AdjacencyMask;

    fn current_score(&self) -> f64 {
        self.cur
    }

    fn accept_policy(&self) -> MaskAccept {
        // Every valid move has Hastings factor 1 for the symmetric directed-pair list.
        MaskAccept::FiniteOnly
    }

    fn cancel_policy(&self) -> MaskCancel {
        MaskCancel::Break
    }

    fn exact_move(&mut self, rng: &mut CausalRng, beta: f64) -> bool {
        if self.n >= EQUIVALENCE_WALK_MIN_NODES && rng.next_f64() < EQUIVALENCE_WALK_PROBABILITY {
            self.reverse_covered_edge(beta, rng);
            return true;
        }
        let Some(catalog) = self.catalog else {
            return false;
        };
        if rng.next_f64() >= self.pair_probability {
            return false;
        }
        let u = crate::indexing::bounded_index(rng.next_u64(), self.n);
        let v = crate::indexing::bounded_index(rng.next_u64(), self.n);
        if u != v {
            let applied = self.resample_pair(catalog, (u, v), beta, rng);
            debug_assert!(
                applied,
                "a drawn parent-set pair is always a valid, finitely scored graph"
            );
        }
        true
    }

    fn propose(&mut self, rng: &mut CausalRng) -> u64 {
        let (prop, _kernel_ratio, rejected) =
            propose_structure(&self.mask, self.n, self.pairs, rng);
        self.pending = prop;
        rejected
    }

    fn proposed_score(&mut self) -> Option<f64> {
        self.target.score(&self.pending)
    }

    fn commit(&mut self, score: f64) {
        self.mask = self.pending.clone();
        self.cur = score;
    }

    fn record(&self, row: &mut [f64]) {
        for (p, &(i, j)) in self.pairs.iter().enumerate() {
            row[p] = if self.mask.has_edge(self.n, i, j) { 1.0 } else { 0.0 };
        }
    }

    fn sample(&self) -> AdjacencyMask {
        self.mask.clone()
    }
}

/// Propose add/delete/reverse; returns `(new_mask, kernel Hastings ratio, rejected)`.
/// `rejected` counts a proposal that is not a DAG.
fn propose_structure(
    mask: &AdjacencyMask,
    n: usize,
    pairs: &[(usize, usize)],
    rng: &mut CausalRng,
) -> (AdjacencyMask, f64, u64) {
    if pairs.is_empty() {
        return (mask.clone(), 1.0, 0);
    }
    let idx = crate::indexing::bounded_index(rng.next_u64(), pairs.len());
    let u = rng.next_f64();
    propose_structure_at(mask, n, pairs, idx, u)
}

/// Apply a selected candidate and branch draw. Kept separate so the finite proposal
/// kernel can be exhaustively checked on a two-node state space.
fn propose_structure_at(
    mask: &AdjacencyMask,
    n: usize,
    pairs: &[(usize, usize)],
    idx: usize,
    u: f64,
) -> (AdjacencyMask, f64, u64) {
    let (i, j) = pairs[idx];
    let forward = mask.has_edge(n, i, j);
    let backward = mask.has_edge(n, j, i);
    // `mask` with the given `(from, to, present)` edits applied in order.
    let edited = |edits: &[(usize, usize, bool)]| {
        let mut next = mask.clone();
        for &(from, to, present) in edits {
            next.set_edge(n, from, to, present);
        }
        next
    };
    if !forward && !backward {
        let prop = edited(&[(i, j, true)]);
        let ratio = structure_kernel_ratio(StructureMove::Add);
        if mask_is_dag(&prop, n) { (prop, ratio, 0) } else { (mask.clone(), ratio, 1) }
    } else if forward && !backward {
        if u < 0.5 {
            (edited(&[(i, j, false)]), structure_kernel_ratio(StructureMove::Delete), 0)
        } else {
            let prop = edited(&[(i, j, false), (j, i, true)]);
            let ratio = structure_kernel_ratio(StructureMove::Reverse);
            if mask_is_dag(&prop, n) { (prop, ratio, 0) } else { (mask.clone(), ratio, 1) }
        }
    } else if backward && !forward {
        if u < 0.5 {
            (edited(&[(j, i, false)]), structure_kernel_ratio(StructureMove::Delete), 0)
        } else {
            let prop = edited(&[(j, i, false), (i, j, true)]);
            let ratio = structure_kernel_ratio(StructureMove::Reverse);
            if mask_is_dag(&prop, n) { (prop, ratio, 0) } else { (mask.clone(), ratio, 1) }
        }
    } else {
        (edited(&[(i, j, false)]), structure_kernel_ratio(StructureMove::Delete), 1)
    }
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::needless_range_loop,
    reason = "test fixtures index small matrices by node"
)]
mod tests {
    use super::*;

    use antecedent_data::TabularData;

    use crate::adjacency_mask::set_edge;
    use crate::graph_posterior::GraphPrior;

    fn fork_data(n_rows: usize) -> (TabularData, Vec<VariableId>) {
        let mut rng = antecedent_core::CausalRng::from_seed(11);
        let mut x = vec![0.0; n_rows];
        let mut y = vec![0.0; n_rows];
        let mut z = vec![0.0; n_rows];
        for i in 0..n_rows {
            x[i] = rng.next_f64() * 2.0 - 1.0;
            y[i] = 1.4 * x[i] + 0.2 * (rng.next_f64() * 2.0 - 1.0);
            z[i] = 1.1 * x[i] + 0.2 * (rng.next_f64() * 2.0 - 1.0);
        }
        let data = crate::test_tables::continuous_table(&["x", "y", "z"], vec![x, y, z]);
        let vars: Vec<_> = (0..3).map(VariableId::from_raw).collect();
        (data, vars)
    }

    #[test]
    fn structure_mcmc_fork_edges() {
        let (data, vars) = fork_data(250);
        // 2 chains / 400 draws (old) left the per-edge MCSE above the publication gate's bar
        // often enough to flake; 4 chains / 1200 draws converges comfortably without masking a
        // real defect (the fork itself is trivially easy to recover).
        let eng = StructureMcmc::new().with_schedule(4, 400, 1200, 1);
        let ctx = ExecutionContext::for_tests(42);
        let mut ws = DiscoveryWorkspace::default();
        let post = eng
            .run(&data, &vars, &GraphPrior::uniform(), GraphScoreFamily::GaussianBic, &mut ws, &ctx)
            .unwrap();
        let n = 3;
        let sk_xy = post.edge_marginals()[1] + post.edge_marginals()[n];
        let sk_xz = post.edge_marginals()[2] + post.edge_marginals()[2 * n];
        assert!(sk_xy > 0.4, "P(X—Y)={sk_xy}");
        assert!(sk_xz > 0.4, "P(X—Z)={sk_xz}");
        assert!(crate::graph_posterior::allows_graph_posterior(post.diagnostics()));
    }
    #[test]
    fn mask_samplers_start_inside_required_edge_support() {
        let (data, vars) = fork_data(100);
        let mut prior = GraphPrior::uniform();
        prior.constraints.required = Arc::from([
            crate::graph_posterior::static_link(&vars, 2, 1),
            crate::graph_posterior::static_link(&vars, 1, 0),
        ]);
        let ctx = ExecutionContext::for_tests(17);
        let mut ws = DiscoveryWorkspace::default();
        let structure = StructureMcmc::new()
            .with_schedule(2, 0, 12, 1)
            .with_diagnostics_gate(false)
            .run(&data, &vars, &prior, GraphScoreFamily::GaussianBic, &mut ws, &ctx)
            .unwrap();
        let order = crate::OrderMcmc::new()
            .with_schedule(2, 0, 12, 1)
            .with_diagnostics_gate(false)
            .run(&data, &vars, &prior, GraphScoreFamily::GaussianBic, &mut ws, &ctx)
            .unwrap();
        for posterior in [structure, order] {
            for mask in posterior.atoms().iter() {
                assert!(mask.has_edge(3, 2, 1));
                assert!(mask.has_edge(3, 1, 0));
                assert!(mask_is_dag(mask, 3));
            }
        }
    }

    #[test]
    fn invalid_schedule_is_reported_without_panicking() {
        let (data, vars) = fork_data(20);
        let mut engine = StructureMcmc::new().with_diagnostics_gate(false);
        engine.thin = 0;
        assert!(
            engine
                .run(
                    &data,
                    &vars,
                    &GraphPrior::uniform(),
                    GraphScoreFamily::GaussianBic,
                    &mut DiscoveryWorkspace::default(),
                    &ExecutionContext::for_tests(1)
                )
                .is_err()
        );
    }

    /// The proposal kernel, the paired parent-set move and replica exchange, with the data
    /// removed: the target is the uniform prior on rooted forests (at most one parent per node),
    /// whose edge-count law is exact. On twelve nodes the probed edges sit in the second and
    /// third packed words.
    #[test]
    fn prior_only_tempered_chain_follows_the_rooted_forest_law_on_twelve_nodes() {
        let n = 12;
        let table = crate::test_tables::rooted_forest_prior_table(n);
        let catalog = ParentSetCatalog::build(&table, None).expect("forest parent sets are few");
        let pairs = candidate_directed_list(n, None);
        let ctx = ExecutionContext::for_tests(23);
        let (draws, thin) = (10_000usize, 10usize);
        let ladder = [1.0, 0.5];
        let (_, samples, _, _) =
            run_parallel_mask_chains(&ctx, 4, draws, pairs.len(), |chain, inner| {
                let mut rng = inner.rng.stream_for(StreamDomain::McmcStructure, chain as u64);
                let mut replicas: Vec<_> = ladder
                    .iter()
                    .map(|_| StructureWalker {
                        n,
                        pairs: &pairs,
                        target: crate::test_tables::PriorOnlyTarget(&table),
                        catalog: Some(&catalog),
                        pair_probability: DEFAULT_PAIR_MOVE_PROBABILITY,
                        mask: AdjacencyMask::EMPTY,
                        cur: 0.0,
                        pending: AdjacencyMask::EMPTY,
                    })
                    .collect();
                let plan = MaskChainPlan {
                    chain_index: chain,
                    n_warmup: 5_000,
                    n_draws: draws,
                    thin,
                    n_params: pairs.len(),
                };
                let run = run_tempered_mask_chain(&plan, inner, &mut rng, &mut replicas, &ladder)?;
                Ok((run.traces, run.samples, run.rejected, run.scores))
            })
            .unwrap();
        crate::test_tables::assert_rooted_forest_law(
            n,
            &samples,
            500,
            &[(0, 1), (5, 9), (11, 10), (10, 11), (11, 0)],
        );
    }

    /// The covered-edge reversal alone, on a three-leaf star under a flat target: the star's
    /// Markov equivalence class is the four orientations away from a chosen root, with three
    /// covered edges when the root is the centre and one otherwise. The class is uniform only
    /// with the move's Hastings factor; without it the chain would favour the centre.
    #[test]
    fn covered_reversal_walks_a_markov_class_uniformly() {
        let n = 4;
        let prior = GraphPrior::uniform();
        let vars: Vec<_> = (0..n).map(|i| VariableId::from_raw(i as u32)).collect();
        let table = GraphPriorTable::new(n, &prior, &vars);
        let start = [(0, 1), (0, 2), (0, 3)]
            .iter()
            .fold(AdjacencyMask::EMPTY, |m, &(a, b)| set_edge(m, n, a, b, true));
        let mut walker = StructureWalker {
            n,
            pairs: &[],
            target: crate::test_tables::PriorOnlyTarget(&table),
            catalog: None,
            pair_probability: 0.0,
            mask: start.clone(),
            cur: 0.0,
            pending: start,
        };
        let mut rng = CausalRng::from_seed(21);
        let steps = 80_000usize;
        let mut visits = std::collections::HashMap::new();
        for _ in 0..steps {
            walker.reverse_covered_edge(1.0, &mut rng);
            assert!(mask_is_dag(&walker.mask, n));
            *visits.entry(walker.mask.clone()).or_insert(0usize) += 1;
        }
        assert_eq!(visits.len(), 4, "the class has four members: {visits:?}");
        for (mask, count) in &visits {
            let share = *count as f64 / steps as f64;
            assert!((share - 0.25).abs() < 0.02, "{mask:?}: {share}");
        }
    }

    /// The paired move draws exactly from its conditional: from a fixed state on four nodes, every
    /// pair of parent sets that keeps the graph acyclic (and within three parents) is reached with
    /// the same frequency under a flat target, and nothing else is.
    #[test]
    fn paired_move_draws_uniformly_from_the_valid_parent_set_pairs() {
        let n = 4;
        let prior = GraphPrior::uniform().with_constraints(crate::DiscoveryConstraints {
            max_parents: Some(3),
            ..Default::default()
        });
        let vars: Vec<_> = (0..n).map(|i| VariableId::from_raw(i as u32)).collect();
        let table = GraphPriorTable::new(n, &prior, &vars);
        let catalog = ParentSetCatalog::build(&table, None).unwrap();
        let (u, v) = (1usize, 3usize);
        // Start: 0 -> 1, 1 -> 2, 2 -> 3, 0 -> 2.
        let start = [(0, 1), (1, 2), (2, 3), (0, 2)]
            .iter()
            .fold(AdjacencyMask::EMPTY, |m, &(a, b)| set_edge(m, n, a, b, true));
        let mut walker = StructureWalker {
            n,
            pairs: &[],
            target: crate::test_tables::PriorOnlyTarget(&table),
            catalog: Some(&catalog),
            pair_probability: 1.0,
            mask: start.clone(),
            cur: 0.0,
            pending: start.clone(),
        };
        // Brute force: every parent set of u and of v, others fixed.
        let others_parents: Vec<u64> =
            (0..n).map(|k| parent_bits(&start, n, k).to_word().unwrap()).collect();
        let mut valid = Vec::new();
        for a in 0u64..16 {
            for b in 0u64..16 {
                if (a >> u) & 1 == 1
                    || (b >> v) & 1 == 1
                    || a.count_ones() > 3
                    || b.count_ones() > 3
                {
                    continue;
                }
                let mut mask = AdjacencyMask::EMPTY;
                for k in 0..n {
                    let set = if k == u {
                        a
                    } else if k == v {
                        b
                    } else {
                        others_parents[k]
                    };
                    for p in (0..n).filter(|&p| (set >> p) & 1 == 1) {
                        mask.set_edge(n, p, k, true);
                    }
                }
                if mask_is_dag(&mask, n) {
                    valid.push((a, b));
                }
            }
        }
        let mut rng = CausalRng::from_seed(5);
        let draws = 40_000usize;
        let mut hits = std::collections::HashMap::new();
        for _ in 0..draws {
            walker.mask = start.clone();
            assert!(walker.resample_pair(&catalog, (u, v), 1.0, &mut rng));
            assert!(mask_is_dag(&walker.mask, n));
            let key = (
                parent_bits(&walker.mask, n, u).to_word().unwrap(),
                parent_bits(&walker.mask, n, v).to_word().unwrap(),
            );
            *hits.entry(key).or_insert(0usize) += 1;
        }
        assert_eq!(hits.len(), valid.len(), "reached sets differ from the valid pairs");
        let expect = draws as f64 / valid.len() as f64;
        for pair in &valid {
            let got = hits[pair] as f64;
            assert!((got - expect).abs() < 5.0 * expect.sqrt(), "{pair:?}: {got} vs {expect}");
        }
    }

    /// The same check under the Gaussian-BIC target: the paired move's draws follow the
    /// normalised product of the two nodes' local scores over the valid pairs.
    #[test]
    fn paired_move_follows_the_bic_conditional_of_the_valid_parent_set_pairs() {
        let n = 4;
        let mut cols = vec![vec![0.0; 120]; n];
        let mut source = antecedent_core::CausalRng::from_seed(5);
        for r in 0..120 {
            let a = source.next_f64() * 2.0 - 1.0;
            let b = source.next_f64() * 2.0 - 1.0;
            let c = 0.7 * a + 0.7 * b + 0.8 * (source.next_f64() * 2.0 - 1.0);
            let d = 0.6 * c + 0.8 * (source.next_f64() * 2.0 - 1.0);
            for (col, x) in cols.iter_mut().zip([a, b, c, d]) {
                col[r] = x;
            }
        }
        let data = crate::test_tables::continuous_table(&["a", "b", "c", "d"], cols);
        let vars: Vec<_> = (0..n).map(|i| VariableId::from_raw(i as u32)).collect();
        let score_data = tabular_score_data(&data, &vars).unwrap();
        let prior = GraphPrior::uniform();
        let table = GraphPriorTable::new(n, &prior, &vars);
        let catalog = ParentSetCatalog::build(&table, None).unwrap();
        let (u, v) = (1usize, 2usize);
        let start = [(0, 1), (3, 2)]
            .iter()
            .fold(AdjacencyMask::EMPTY, |m, &(a, b)| set_edge(m, n, a, b, true));
        let mut reference =
            ScoredTarget::new(n, &score_data, GraphScoreFamily::GaussianBic, &table);
        let mut walker = StructureWalker {
            n,
            pairs: &[],
            target: ScoredTarget::new(n, &score_data, GraphScoreFamily::GaussianBic, &table),
            catalog: Some(&catalog),
            pair_probability: 1.0,
            mask: start.clone(),
            cur: 0.0,
            pending: start.clone(),
        };
        let fixed: Vec<u64> =
            (0..n).map(|k| parent_bits(&start, n, k).to_word().unwrap()).collect();
        let mut weights = std::collections::HashMap::new();
        for a in 0u64..16 {
            for b in 0u64..16 {
                if (a >> u) & 1 == 1 || (b >> v) & 1 == 1 {
                    continue;
                }
                let mut mask = AdjacencyMask::EMPTY;
                for k in 0..n {
                    let set = if k == u {
                        a
                    } else if k == v {
                        b
                    } else {
                        fixed[k]
                    };
                    for p in (0..n).filter(|&p| (set >> p) & 1 == 1) {
                        mask.set_edge(n, p, k, true);
                    }
                }
                if let Some(log_target) = reference.score(&mask) {
                    weights.insert((a, b), log_target);
                }
            }
        }
        let top = weights.values().copied().fold(f64::NEG_INFINITY, f64::max);
        let z: f64 = weights.values().map(|l| (l - top).exp()).sum();
        let mut rng = CausalRng::from_seed(9);
        let draws = 60_000usize;
        let mut hits = std::collections::HashMap::new();
        for _ in 0..draws {
            walker.mask = start.clone();
            assert!(walker.resample_pair(&catalog, (u, v), 1.0, &mut rng));
            let key = (
                parent_bits(&walker.mask, n, u).to_word().unwrap(),
                parent_bits(&walker.mask, n, v).to_word().unwrap(),
            );
            *hits.entry(key).or_insert(0usize) += 1;
        }
        for (pair, log_target) in &weights {
            let p = (log_target - top).exp() / z;
            let got = hits.get(pair).copied().unwrap_or(0) as f64 / draws as f64;
            let sd = (p * (1.0 - p) / draws as f64).sqrt();
            assert!((got - p).abs() < 5.0 * sd + 1e-4, "{pair:?}: {got} vs {p}");
        }
    }

    /// From every one of the 543 four-node DAGs, for several node pairs, the paired move's draws
    /// follow the exact normalised conditional (BIC target) of the valid parent-set pairs. This
    /// is the check that caught a cancellation: the admissible mass was once obtained by
    /// subtracting an excluded best pair's mass from the total.
    #[test]
    fn paired_move_matches_the_exact_conditional_from_every_four_node_dag() {
        let n = 4;
        let mut cols = vec![vec![0.0; 120]; n];
        let mut source = antecedent_core::CausalRng::from_seed(5);
        for r in 0..120 {
            let a = source.next_f64() * 2.0 - 1.0;
            let b = source.next_f64() * 2.0 - 1.0;
            let c = 0.7 * a + 0.7 * b + 0.8 * (source.next_f64() * 2.0 - 1.0);
            let d = 0.6 * c + 0.8 * (source.next_f64() * 2.0 - 1.0);
            for (col, x) in cols.iter_mut().zip([a, b, c, d]) {
                col[r] = x;
            }
        }
        let data = crate::test_tables::continuous_table(&["a", "b", "c", "d"], cols);
        let vars: Vec<_> = (0..n).map(|i| VariableId::from_raw(i as u32)).collect();
        let score_data = tabular_score_data(&data, &vars).unwrap();
        let prior = GraphPrior::uniform();
        let table = GraphPriorTable::new(n, &prior, &vars);
        let catalog = ParentSetCatalog::build(&table, None).unwrap();
        let mut reference =
            ScoredTarget::new(n, &score_data, GraphScoreFamily::GaussianBic, &table);
        let mut rng = CausalRng::from_seed(9);
        let mut states = 0;
        for word in 0u64..(1 << 12) {
            let start = AdjacencyMask::from_word(word);
            if !mask_is_dag(&start, n) {
                continue;
            }
            states += 1;
            for (u, v) in [(2usize, 3usize), (1, 2), (3, 1)] {
                let fixed: Vec<u64> =
                    (0..n).map(|k| parent_bits(&start, n, k).to_word().unwrap()).collect();
                let mut weights = std::collections::HashMap::new();
                for a in 0u64..16 {
                    for b in 0u64..16 {
                        if (a >> u) & 1 == 1 || (b >> v) & 1 == 1 {
                            continue;
                        }
                        let mut mask = AdjacencyMask::EMPTY;
                        for k in 0..n {
                            let set = if k == u {
                                a
                            } else if k == v {
                                b
                            } else {
                                fixed[k]
                            };
                            for p in (0..n).filter(|&p| (set >> p) & 1 == 1) {
                                mask.set_edge(n, p, k, true);
                            }
                        }
                        if let Some(l) = reference.score(&mask) {
                            weights.insert((a, b), l);
                        }
                    }
                }
                let top = weights.values().copied().fold(f64::NEG_INFINITY, f64::max);
                let z: f64 = weights.values().map(|l| (l - top).exp()).sum();
                let mut walker = StructureWalker {
                    n,
                    pairs: &[],
                    target: ScoredTarget::new(
                        n,
                        &score_data,
                        GraphScoreFamily::GaussianBic,
                        &table,
                    ),
                    catalog: Some(&catalog),
                    pair_probability: 1.0,
                    mask: start.clone(),
                    cur: 0.0,
                    pending: start.clone(),
                };
                let draws = 1500usize;
                let mut hits = std::collections::HashMap::new();
                for _ in 0..draws {
                    walker.mask = start.clone();
                    assert!(walker.resample_pair(&catalog, (u, v), 1.0, &mut rng));
                    let key = (
                        parent_bits(&walker.mask, n, u).to_word().unwrap(),
                        parent_bits(&walker.mask, n, v).to_word().unwrap(),
                    );
                    *hits.entry(key).or_insert(0usize) += 1;
                }
                let mut worst = 0.0f64;
                for (pair, l) in &weights {
                    let p = (l - top).exp() / z;
                    let got = hits.get(pair).copied().unwrap_or(0) as f64 / draws as f64;
                    let sd = (p * (1.0 - p) / draws as f64).sqrt().max(1e-3);
                    worst = worst.max((got - p).abs() / sd);
                }
                let extra = hits.keys().filter(|k| !weights.contains_key(k)).count();
                assert!(
                    worst < 6.0 && extra == 0,
                    "state {word:#x}, pair ({u}, {v}): worst z {worst:.1}, {extra} unexpected sets"
                );
            }
        }
        assert_eq!(states, 543);
    }

    /// Whole-sampler check of the new moves against exhaustive enumeration on four variables
    /// (543 DAGs, Gaussian BIC): paired moves only, no tempering; then the default sampler.
    #[test]
    fn paired_move_and_tempering_match_exhaustive_enumeration_on_four_variables() {
        let mut cols = vec![vec![0.0; 120]; 4];
        let mut rng = antecedent_core::CausalRng::from_seed(77);
        for r in 0..120 {
            let a = rng.next_f64() * 2.0 - 1.0;
            let b = rng.next_f64() * 2.0 - 1.0;
            let c = 0.7 * a + 0.7 * b + 0.8 * (rng.next_f64() * 2.0 - 1.0);
            let d = 0.6 * c + 0.8 * (rng.next_f64() * 2.0 - 1.0);
            for (col, x) in cols.iter_mut().zip([a, b, c, d]) {
                col[r] = x;
            }
        }
        let data = crate::test_tables::continuous_table(&["a", "b", "c", "d"], cols);
        let vars: Vec<_> = (0..4).map(VariableId::from_raw).collect();
        let prior = GraphPrior::uniform();
        let ctx = ExecutionContext::for_tests(3);
        let exact = crate::ExactDagPosterior::new()
            .run(
                &data,
                &vars,
                &prior,
                GraphScoreFamily::GaussianBic,
                &mut DiscoveryWorkspace::default(),
                &ctx,
            )
            .unwrap();
        let paired_only = StructureMcmc::new()
            .with_schedule(4, 1000, 6000, 2)
            .with_temperatures([1.0])
            .with_pair_move_probability(1.0)
            .with_diagnostics_gate(false);
        let default =
            StructureMcmc::new().with_schedule(4, 1000, 6000, 2).with_diagnostics_gate(false);
        for (label, engine) in [("default", default), ("paired only", paired_only)] {
            let post = engine
                .run(
                    &data,
                    &vars,
                    &prior,
                    GraphScoreFamily::GaussianBic,
                    &mut DiscoveryWorkspace::default(),
                    &ctx,
                )
                .unwrap();
            let mcse = post.edge_mcse().unwrap();
            for cell in 0..16 {
                let band = 0.03 + 5.0 * mcse[cell];
                assert!(
                    (post.edge_marginals()[cell] - exact.edge_marginals()[cell]).abs() <= band,
                    "{label}: cell {cell}: {} vs exact {} (band {band})",
                    post.edge_marginals()[cell],
                    exact.edge_marginals()[cell]
                );
            }
        }
    }

    /// Past sixteen nodes the packed edge set leaves its inline words and a parent set is a node
    /// set wider than sixteen bits: the proposal kernel and the paired move still draw only
    /// forests under the one-parent prior, and reach edges of the highest nodes.
    #[test]
    fn prior_only_tempered_chain_runs_on_eighteen_nodes() {
        let n = 18;
        let table = crate::test_tables::rooted_forest_prior_table(n);
        let catalog = ParentSetCatalog::build(&table, None).expect("forest parent sets are few");
        assert!(catalog.sets.iter().all(|sets| sets.len() == n), "no parent, or one of n - 1");
        let pairs = candidate_directed_list(n, None);
        let ctx = ExecutionContext::for_tests(31);
        let (draws, thin) = (400usize, 5usize);
        let ladder = [1.0, 0.5];
        let (_, samples, _, _) =
            run_parallel_mask_chains(&ctx, 2, draws, pairs.len(), |chain, inner| {
                let mut rng = inner.rng.stream_for(StreamDomain::McmcStructure, chain as u64);
                let mut replicas: Vec<_> = ladder
                    .iter()
                    .map(|_| StructureWalker {
                        n,
                        pairs: &pairs,
                        target: crate::test_tables::PriorOnlyTarget(&table),
                        catalog: Some(&catalog),
                        pair_probability: DEFAULT_PAIR_MOVE_PROBABILITY,
                        mask: AdjacencyMask::EMPTY,
                        cur: 0.0,
                        pending: AdjacencyMask::EMPTY,
                    })
                    .collect();
                let plan = MaskChainPlan {
                    chain_index: chain,
                    n_warmup: 500,
                    n_draws: draws,
                    thin,
                    n_params: pairs.len(),
                };
                let run = run_tempered_mask_chain(&plan, inner, &mut rng, &mut replicas, &ladder)?;
                Ok((run.traces, run.samples, run.rejected, run.scores))
            })
            .unwrap();
        let mut wide = false;
        for mask in samples.iter().flatten() {
            assert!(mask_is_dag(mask, n));
            let mut in_degree = vec![0usize; n];
            for (_, to) in mask.edges(n) {
                in_degree[to] += 1;
            }
            assert!(in_degree.iter().all(|&d| d <= 1), "{mask:#x}");
            wide |= mask.words().len() > 4 || mask.edges(n).any(|(from, to)| from.max(to) >= 16);
        }
        assert!(wide, "no sample used an edge of the nodes past the sixteenth");
    }

    #[test]
    fn structure_kernel_is_detailed_balanced_for_two_nodes() {
        let pairs = candidate_directed_list(2, None);
        assert_eq!(pairs, [(0, 1), (1, 0)]);
        let states = [
            AdjacencyMask::EMPTY,
            set_edge(AdjacencyMask::EMPTY, 2, 0, 1, true),
            set_edge(AdjacencyMask::EMPTY, 2, 1, 0, true),
        ];
        let mut transition_mass = [[0.0_f64; 3]; 3];
        // Each directed-pair choice has probability 1/2, and each branch interval
        // below represents probability 1/2 under the uniform branch draw.
        for (from, mask) in states.iter().enumerate() {
            for idx in 0..pairs.len() {
                for u in [0.25, 0.75] {
                    let (next, _, _) = propose_structure_at(mask, 2, &pairs, idx, u);
                    let to = states.iter().position(|state| *state == next).unwrap();
                    transition_mass[from][to] += 0.25;
                }
            }
        }
        for i in 0..states.len() {
            for j in 0..states.len() {
                assert_eq!(transition_mass[i][j].to_bits(), transition_mass[j][i].to_bits());
            }
        }
        assert_eq!(structure_kernel_ratio(StructureMove::Add).to_bits(), 1.0_f64.to_bits());
        assert_eq!(structure_kernel_ratio(StructureMove::Delete).to_bits(), 1.0_f64.to_bits());
        assert_eq!(structure_kernel_ratio(StructureMove::Reverse).to_bits(), 1.0_f64.to_bits());
    }
}
