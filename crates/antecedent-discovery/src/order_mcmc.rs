//! Order MCMC for DAG posteriors.
//!
//! State = topological order + forward-edge subset. Proposals: adjacent
//! transposition (reorienting the skeleton) and forward-edge flips.
//!
//! The Metropolis target is the DAG posterior on graphs, not the order-augmented
//! joint: each candidate score is corrected by `log|# topological orders|` so
//! DAGs with more linear extensions are not over-weighted (MM-011).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{CausalRng, ExecutionContext, StreamDomain, VariableId};
use antecedent_data::TabularData;
use antecedent_state::GraphScoreFamily;

use crate::adjacency_mask::AdjacencyMask;
use crate::adjacency_mask::{has_edge, set_edge};
use crate::engine::DiscoveryWorkspace;
use crate::error::DiscoveryError;
use crate::graph_mcmc::{
    DEFAULT_TEMPERATURES, EQUIVALENCE_WALK_MIN_NODES, EQUIVALENCE_WALK_PROBABILITY,
    FinishMaskPosterior, GraphMcmcSchedule, MaskAccept, MaskCancel, MaskChain, MaskChainPlan,
    accept_covered_reversal, propose_covered_reversal, reversed_edge, run_parallel_mask_chains,
    run_tempered_mask_chain, tempering_ladder, validate_temperatures,
};
use crate::graph_posterior::{
    GraphPosterior, GraphPosteriorEngine, GraphPrior, GraphPriorTable, n_directed_edges,
};
use crate::graph_score::{
    ChainTarget, ScoredTarget, SharedTarget, initial_scored_mask, tabular_score_data,
};

/// Order MCMC over topological orders and compatible forward edges.
///
/// Supports up to [`crate::MAX_POSTERIOR_NODES`] variables; each step costs a `2^n`-state dynamic
/// program (the topological-order correction), so it doubles with every variable. Cap the parents
/// of any node with `GraphPrior::constraints.max_parents`.
///
/// Above eight variables a quarter of the iterations instead reverse a uniformly chosen covered
/// edge and exchange the two endpoints in the order (a deterministic partner of the reversal,
/// valid when the tail has no child between the two), accepted with the Metropolis–Hastings
/// ratio of the move. The reversal stays inside the Markov equivalence class, so the
/// orientation of an edge the data leave open is walked at the acceptance of a plateau rather
/// than waiting for a transposition that happens to separate the two nodes by no child.
///
/// The order move set also relocates a uniformly chosen node to a uniformly chosen position
/// (`from` and `to` independent and uniform, so the move and its inverse are equally likely and
/// the proposal is symmetric). It replaces part of the adjacent transpositions above eight
/// variables, where an order space of `n!` states is crossed too slowly by single-place moves.
///
/// Every chain runs replicas at the inverse temperatures of [`Self::temperatures`] and exchanges
/// them (parallel tempering), which carries chains across the local modes single-step moves
/// cannot leave; only the replica at temperature 1 is recorded. The order-modular prior is
/// corrected, not tempered away: the tempered family is `(exp(score) / |orders|)^beta`.
#[derive(Clone, Debug)]
pub struct OrderMcmc {
    /// Number of chains (≥ 2 for R-hat).
    pub n_chains: u32,
    /// Warmup draws discarded per chain.
    pub n_warmup: u32,
    /// Post-warmup draws retained per chain (before thinning).
    pub n_draws: u32,
    /// Keep every `thin`-th post-warmup draw.
    pub thin: u32,
    /// When true (default), refuse if [`crate::graph_posterior::allows_graph_posterior`] fails.
    pub require_diagnostics_gate: bool,
    /// Inverse-temperature ladder of the replicas in every chain: `1 = b_0 > b_1 > ... > 0`
    /// (default `[1, 1/2, 1/4, 1/8]`, run as a denser eight-replica ladder from nine variables on;
    /// `[1]` is a single untempered chain).
    pub temperatures: Vec<f64>,
}

impl Default for OrderMcmc {
    fn default() -> Self {
        Self::new()
    }
}

impl OrderMcmc {
    /// Default sampler.
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_chains: 4,
            n_warmup: 500,
            n_draws: 1000,
            thin: 1,
            require_diagnostics_gate: true,
            temperatures: DEFAULT_TEMPERATURES.to_vec(),
        }
    }

    /// Set the inverse-temperature ladder (validated when the sampler runs).
    #[must_use]
    pub fn with_temperatures(mut self, temperatures: impl Into<Vec<f64>>) -> Self {
        self.temperatures = temperatures.into();
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

    /// Run order MCMC.
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
        prior.constraints.validate()?;
        let n = variables.len();
        if n == 0 {
            return Err(DiscoveryError::unsupported("order MCMC requires at least one variable"));
        }
        crate::graph_posterior::ensure_nodes_fit(
            n,
            "order MCMC supports at most 16 variables (MAX_POSTERIOR_NODES)",
        )?;
        if !matches!(score_family, GraphScoreFamily::GaussianBic | GraphScoreFamily::LaplaceBic) {
            return Err(DiscoveryError::unsupported(
                "order MCMC supports the Gaussian and Laplace BIC scores only",
            ));
        }
        let schedule = self.schedule();
        schedule.require_min_chains(2, "order MCMC requires at least 2 chains for R-hat")?;
        validate_temperatures(&self.temperatures)?;
        let ladder = tempering_ladder(&self.temperatures, n);

        let score_data = tabular_score_data(data, variables)?;
        let (initial_mask, initial_score) =
            initial_scored_mask(n, &score_data, score_family, prior, variables)?;
        let prior_table = GraphPriorTable::new(n, prior, variables);
        let mut initial_order = Vec::with_capacity(n);
        while initial_order.len() < n {
            let next = (0..n)
                .find(|node| {
                    !initial_order.contains(node)
                        && (0..n).all(|parent| {
                            parent == *node
                                || !has_edge(initial_mask, n, parent, *node)
                                || initial_order.contains(&parent)
                        })
                })
                .expect("validated required-edge DAG has a topological order");
            initial_order.push(next);
        }
        let initial_score = initial_score - log_topological_order_count(initial_mask, n);

        let (n_chains, n_warmup, n_draws, thin) = schedule.as_usize();
        let n_params = n_directed_edges(n);

        let (traces, sample_masks, rejected, scores) =
            run_parallel_mask_chains(ctx, n_chains, n_draws, n_params, |chain, inner| {
                let mut rng = inner.rng.stream_for(StreamDomain::McmcOrder, chain as u64);
                let target = SharedTarget(std::rc::Rc::new(std::cell::RefCell::new(
                    ScoredTarget::new(n, &score_data, score_family, &prior_table),
                )));
                let mut replicas: Vec<_> = ladder
                    .iter()
                    .map(|_| OrderWalker {
                        n,
                        target: target.clone(),
                        order: initial_order.clone(),
                        mask: initial_mask,
                        cur: initial_score,
                        pending_order: initial_order.clone(),
                        pending_mask: initial_mask,
                        pos_scratch: Vec::with_capacity(n),
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

        let param_edges: Vec<(usize, usize)> =
            (0..n).flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j))).collect();
        FinishMaskPosterior {
            n,
            schedule: &schedule,
            traces: &traces,
            scores: &scores,
            sample_masks: &sample_masks,
            rejected,
            n_params,
            param_edges: &param_edges,
            require_gate: self.require_diagnostics_gate,
            refuse_msg: "order MCMC diagnostics gate refused posterior",
            empty_msg: "order MCMC produced no samples",
        }
        .run()
    }
}

impl GraphPosteriorEngine for OrderMcmc {
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

/// One order-MCMC replica. `target` is the log target of a mask before the topological-order
/// correction: BIC + log prior in a run, and whatever a test needs to isolate the kernel.
struct OrderWalker<T> {
    n: usize,
    target: T,
    order: Vec<usize>,
    mask: AdjacencyMask,
    cur: f64,
    pending_order: Vec<usize>,
    pending_mask: AdjacencyMask,
    pos_scratch: Vec<usize>,
}

impl<T: ChainTarget> OrderWalker<T> {
    /// Metropolis–Hastings step that reverses a uniformly chosen covered edge `x -> y` and
    /// swaps `x` and `y` in the order, at inverse temperature `beta`.
    ///
    /// A node between `x` and `y` in the order is neither a parent of `y` (those are `x` and the
    /// parents of `x`, which precede `x`) nor a parent of `x`; it may only be a child of `x`,
    /// which the swap would put before its parent, so such a state is left as it is. The swap is
    /// its own inverse, and no child of `y` lies between the two, so the reverse move exists
    /// exactly when the forward one does and the proposal ratio is the ratio of covered counts.
    fn reverse_covered_edge(&mut self, beta: f64, rng: &mut CausalRng) {
        let Some((x, y, covered)) = propose_covered_reversal(self.mask, self.n, rng) else {
            return;
        };
        fill_position_map(&self.order, &mut self.pos_scratch);
        let (a, b) = (self.pos_scratch[x], self.pos_scratch[y]);
        debug_assert!(a < b, "an edge points forward in the order");
        if a >= b || self.order[a + 1..b].iter().any(|&z| has_edge(self.mask, self.n, x, z)) {
            return;
        }
        let proposal = reversed_edge(self.mask, self.n, x, y);
        let Some(score) = self
            .target
            .score(proposal)
            .map(|score| score - log_topological_order_count(proposal, self.n))
        else {
            return;
        };
        if accept_covered_reversal((beta, self.cur, score), covered, proposal, self.n, rng) {
            self.order.swap(a, b);
            self.mask = proposal;
            self.cur = score;
        }
    }
}

impl<T: ChainTarget> MaskChain for OrderWalker<T> {
    type Sample = AdjacencyMask;

    fn current_score(&self) -> f64 {
        self.cur
    }

    fn accept_policy(&self) -> MaskAccept {
        // A non-finite score is an invalid proposal. A Metropolis rejection of a finite
        // score is not counted.
        MaskAccept::FiniteOnlyCounted
    }

    fn cancel_policy(&self) -> MaskCancel {
        MaskCancel::Break
    }

    fn exact_move(&mut self, rng: &mut CausalRng, beta: f64) -> bool {
        if self.n >= EQUIVALENCE_WALK_MIN_NODES && rng.next_f64() < EQUIVALENCE_WALK_PROBABILITY {
            self.reverse_covered_edge(beta, rng);
            return true;
        }
        false
    }

    fn propose(&mut self, rng: &mut CausalRng) -> u64 {
        self.pending_mask = propose_order(
            &self.order,
            self.mask,
            self.n,
            rng,
            &mut self.pending_order,
            &mut self.pos_scratch,
        );
        0
    }

    fn proposed_score(&mut self) -> Option<f64> {
        self.target
            .score(self.pending_mask)
            .map(|score| score - log_topological_order_count(self.pending_mask, self.n))
    }

    fn commit(&mut self, score: f64) {
        self.order.clone_from(&self.pending_order);
        self.mask = self.pending_mask;
        self.cur = score;
    }

    fn record(&self, row: &mut [f64]) {
        let mut p = 0usize;
        for i in 0..self.n {
            for j in 0..self.n {
                if i == j {
                    continue;
                }
                row[p] = if has_edge(self.mask, self.n, i, j) { 1.0 } else { 0.0 };
                p += 1;
            }
        }
    }

    fn sample(&self) -> AdjacencyMask {
        self.mask
    }
}

fn propose_order(
    order: &[usize],
    mask: AdjacencyMask,
    n: usize,
    rng: &mut CausalRng,
    new_order: &mut Vec<usize>,
    pos: &mut Vec<usize>,
) -> AdjacencyMask {
    new_order.clear();
    new_order.extend_from_slice(order);
    let mut new_mask = mask;
    let u = rng.next_f64();
    // Above the equivalence-walk size a share of the transpositions becomes node relocations:
    // an adjacent transposition moves an order by one place, so over `n!` orders it is a slow
    // random walk, while a relocation moves one node anywhere and reshapes the order at once.
    let relocate = n >= EQUIVALENCE_WALK_MIN_NODES;
    let (adjacent_below, relocate_below, swap_below) =
        if relocate { (0.2, 0.45, 0.55) } else { (0.35, 0.35, 0.55) };
    if u < adjacent_below && n >= 2 {
        let k = crate::indexing::bounded_index(rng.next_u64(), n - 1);
        new_order.swap(k, k + 1);
        new_mask = reorient_skeleton(mask, new_order, n, pos);
    } else if relocate && u < relocate_below {
        let from = crate::indexing::bounded_index(rng.next_u64(), n);
        let to = crate::indexing::bounded_index(rng.next_u64(), n);
        if from != to {
            let node = new_order.remove(from);
            new_order.insert(to, node);
            new_mask = reorient_skeleton(mask, new_order, n, pos);
        }
    } else if u < swap_below && n >= 3 {
        let a = crate::indexing::bounded_index(rng.next_u64(), n);
        let b = crate::indexing::bounded_index(rng.next_u64(), n);
        if a != b {
            new_order.swap(a, b);
            new_mask = reorient_skeleton(mask, new_order, n, pos);
        }
    } else {
        let a = crate::indexing::bounded_index(rng.next_u64(), n);
        let b = crate::indexing::bounded_index(rng.next_u64(), n);
        if a != b {
            fill_position_map(new_order, pos);
            let (from, to) = if pos[a] < pos[b] { (a, b) } else { (b, a) };
            let on = has_edge(new_mask, n, from, to);
            new_mask = set_edge(new_mask, n, from, to, !on);
        }
    }
    new_mask
}

fn fill_position_map(order: &[usize], pos: &mut Vec<usize>) {
    pos.clear();
    pos.resize(order.len(), 0);
    for (p, &node) in order.iter().enumerate() {
        pos[node] = p;
    }
}

fn reorient_skeleton(
    mask: AdjacencyMask,
    order: &[usize],
    n: usize,
    pos: &mut Vec<usize>,
) -> AdjacencyMask {
    fill_position_map(order, pos);
    let mut out = AdjacencyMask::EMPTY;
    for a in 0..n {
        for b in a + 1..n {
            if has_edge(mask, n, a, b) || has_edge(mask, n, b, a) {
                let (from, to) = if pos[a] < pos[b] { (a, b) } else { (b, a) };
                out = set_edge(out, n, from, to, true);
            }
        }
    }
    out
}

/// `ln` of the number of linear extensions of the DAG `mask`: a `2^n`-state subset DP, so the
/// per-step cost of order MCMC grows as `n * 2^n`.
fn log_topological_order_count(mask: AdjacencyMask, n: usize) -> f64 {
    let states = 1usize << n;
    let parents: Vec<usize> =
        (0..n).map(|node| usize::from(crate::adjacency_mask::parent_bits(mask, n, node))).collect();
    let mut ways = vec![0u64; states];
    ways[0] = 1;
    for placed in 0..states {
        let w = ways[placed];
        if w == 0 {
            continue;
        }
        let mut open = !placed & (states - 1);
        while open != 0 {
            let node = open.trailing_zeros() as usize;
            open &= open - 1;
            if parents[node] & !placed == 0 {
                let next = placed | (1 << node);
                ways[next] = ways[next].saturating_add(w);
            }
        }
    }
    (ways[states - 1].max(1) as f64).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    use antecedent_data::TabularData;

    use crate::graph_posterior::GraphPrior;

    #[test]
    fn topological_order_count_matches_known_dags() {
        let n = 3;
        // Empty DAG: all 3! = 6 orders.
        assert!(
            (log_topological_order_count(AdjacencyMask::EMPTY, n) - (6.0f64).ln()).abs() < 1e-12
        );
        // Chain 0→1→2: unique order.
        let chain = set_edge(set_edge(AdjacencyMask::EMPTY, n, 0, 1, true), n, 1, 2, true);
        assert!((log_topological_order_count(chain, n) - 0.0).abs() < 1e-12);
        // V-structure 0→2←1: two orders (0 before 1, or 1 before 0).
        let v = set_edge(set_edge(AdjacencyMask::EMPTY, n, 0, 2, true), n, 1, 2, true);
        assert!((log_topological_order_count(v, n) - (2.0f64).ln()).abs() < 1e-12);
    }

    /// The order kernel with its `log |orders|` correction and replica exchange, with the data
    /// removed: the target is the uniform prior on rooted forests (at most one parent per node),
    /// whose edge-count law is exact. Without the correction the DAGs with many topological
    /// orders are over-weighted and the law is off; a wrong exchange rule shifts it too.
    #[test]
    fn prior_only_tempered_chain_follows_the_rooted_forest_law_on_ten_nodes() {
        let n = 10;
        let table = crate::test_tables::rooted_forest_prior_table(n);
        let ctx = ExecutionContext::for_tests(31);
        let (draws, thin) = (6_000usize, 5usize);
        let n_params = n_directed_edges(n);
        let ladder = [1.0, 0.5];
        let (_, samples, _, _) =
            run_parallel_mask_chains(&ctx, 4, draws, n_params, |chain, inner| {
                let mut rng = inner.rng.stream_for(StreamDomain::McmcOrder, chain as u64);
                let mut replicas: Vec<_> = ladder
                    .iter()
                    .map(|_| OrderWalker {
                        n,
                        target: crate::test_tables::PriorOnlyTarget(&table),
                        order: (0..n).collect(),
                        mask: AdjacencyMask::EMPTY,
                        cur: -log_topological_order_count(AdjacencyMask::EMPTY, n),
                        pending_order: (0..n).collect(),
                        pending_mask: AdjacencyMask::EMPTY,
                        pos_scratch: Vec::with_capacity(n),
                    })
                    .collect();
                let plan = MaskChainPlan {
                    chain_index: chain,
                    n_warmup: 5_000,
                    n_draws: draws,
                    thin,
                    n_params,
                };
                let run = run_tempered_mask_chain(&plan, inner, &mut rng, &mut replicas, &ladder)?;
                Ok((run.traces, run.samples, run.rejected, run.scores))
            })
            .unwrap();
        crate::test_tables::assert_rooted_forest_law(
            n,
            &samples,
            600,
            &[(0, 1), (4, 7), (9, 8), (8, 9), (9, 0)],
        );
    }

    /// Every proposal at ten nodes (relocations included) is a permutation of the order and a
    /// mask whose edges all point forward in it, and proposals change every position of
    /// the order. The prior-only law test above then checks the invariance of the whole
    /// move set, since at ten nodes it proposes relocations.
    #[test]
    fn order_proposals_keep_the_mask_compatible_with_the_order() {
        let n = 10;
        let mut rng = antecedent_core::CausalRng::from_seed(5);
        let (mut order, mut mask) = ((0..n).collect::<Vec<_>>(), AdjacencyMask::EMPTY);
        let (mut proposed, mut pos) = (Vec::new(), Vec::new());
        let mut moved_to = [false; 10];
        for _ in 0..4_000 {
            let new_mask = propose_order(&order, mask, n, &mut rng, &mut proposed, &mut pos);
            let mut sorted = proposed.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, (0..n).collect::<Vec<_>>());
            fill_position_map(&proposed, &mut pos);
            for i in 0..n {
                for j in 0..n {
                    assert!(!has_edge(new_mask, n, i, j) || pos[i] < pos[j]);
                }
            }
            for p in (0..n).filter(|&p| proposed[p] != order[p]) {
                moved_to[p] = true;
            }
            (order, mask) = (proposed.clone(), new_mask);
        }
        assert!(moved_to.iter().all(|&m| m), "{moved_to:?}");
    }

    #[test]
    fn more_than_sixteen_variables_are_refused_before_any_work() {
        let data = crate::test_tables::tabular_n(17, 20);
        let vars: Vec<_> = (0..17).map(VariableId::from_raw).collect();
        let err = OrderMcmc::new()
            .run(
                &data,
                &vars,
                &GraphPrior::uniform(),
                GraphScoreFamily::GaussianBic,
                &mut DiscoveryWorkspace::default(),
                &ExecutionContext::for_tests(1),
            )
            .unwrap_err();
        assert!(err.to_string().contains("at most 16 variables"), "{err}");
    }

    fn chain_data(n_rows: usize) -> (TabularData, Vec<VariableId>) {
        let mut rng = antecedent_core::CausalRng::from_seed(3);
        let mut a = vec![0.0; n_rows];
        let mut bb = vec![0.0; n_rows];
        let mut c = vec![0.0; n_rows];
        for i in 0..n_rows {
            a[i] = rng.next_f64() * 2.0 - 1.0;
            bb[i] = 1.6 * a[i] + 0.15 * (rng.next_f64() * 2.0 - 1.0);
            c[i] = 1.3 * bb[i] + 0.15 * (rng.next_f64() * 2.0 - 1.0);
        }
        let data = crate::test_tables::continuous_table(&["a", "b", "c"], vec![a, bb, c]);
        let vars: Vec<_> = (0..3).map(VariableId::from_raw).collect();
        (data, vars)
    }

    #[test]
    fn order_mcmc_chain_signal() {
        let (data, vars) = chain_data(220);
        // Gate off: a short schedule recovers edge mass but is not intended as a
        // publishable multi-chain run (see the refusal test below).
        let eng = OrderMcmc::new().with_schedule(2, 40, 50, 1).with_diagnostics_gate(false);
        let ctx = ExecutionContext::for_tests(9);
        let mut ws = DiscoveryWorkspace::default();
        let post = eng
            .run(&data, &vars, &GraphPrior::uniform(), GraphScoreFamily::GaussianBic, &mut ws, &ctx)
            .unwrap();
        let n = 3;
        let sk01 = post.edge_marginals[1] + post.edge_marginals[n];
        assert!(sk01 > 0.25, "P(A—B)={sk01}");
        assert!(
            !crate::graph_posterior::allows_graph_posterior(&post.diagnostics),
            "short order-MCMC schedule should not clear the diagnostics bar (ess_bulk={:?}, rhat={:?})",
            post.diagnostics.ess_bulk_min,
            post.diagnostics.rhat_max
        );
    }

    #[test]
    fn order_mcmc_short_schedule_gate_refuses() {
        let (data, vars) = chain_data(220);
        let eng = OrderMcmc::new().with_schedule(2, 40, 50, 1).with_diagnostics_gate(true);
        let ctx = ExecutionContext::for_tests(9);
        let mut ws = DiscoveryWorkspace::default();
        let err = eng
            .run(&data, &vars, &GraphPrior::uniform(), GraphScoreFamily::GaussianBic, &mut ws, &ctx)
            .unwrap_err();
        assert!(err.to_string().contains("diagnostics gate"), "err={err}");
    }

    #[test]
    fn order_mcmc_counts_forbidden_proposals() {
        let (data, vars) = chain_data(40);
        let mut prior = GraphPrior::uniform();
        prior.constraints.forbidden = std::sync::Arc::from([
            crate::graph_posterior::static_link(&vars, 0, 1),
            crate::graph_posterior::static_link(&vars, 1, 0),
            crate::graph_posterior::static_link(&vars, 0, 2),
            crate::graph_posterior::static_link(&vars, 2, 0),
            crate::graph_posterior::static_link(&vars, 1, 2),
            crate::graph_posterior::static_link(&vars, 2, 1),
        ]);
        let eng = OrderMcmc::new().with_schedule(2, 0, 30, 1).with_diagnostics_gate(false);
        let post = eng
            .run(
                &data,
                &vars,
                &prior,
                GraphScoreFamily::GaussianBic,
                &mut DiscoveryWorkspace::default(),
                &ExecutionContext::for_tests(3),
            )
            .unwrap();
        assert!(post.rejected_invalid > 0, "forbidden edges were reported as accepted proposals");
    }
}
