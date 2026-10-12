//! Structure MCMC for DAG posteriors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{CausalRng, ExecutionContext, StreamDomain, VariableId};
use antecedent_data::TabularData;
use antecedent_state::{GraphScoreData, GraphScoreFamily, LocalScoreCache};

use crate::engine::DiscoveryWorkspace;
use crate::error::DiscoveryError;
use crate::graph_mcmc::{
    FinishMaskPosterior, GraphMcmcSchedule, MaskAccept, MaskCancel, MaskChain, MaskChainPlan,
    run_mask_chain, run_parallel_mask_chains,
};
use crate::graph_posterior::{
    GraphPosterior, GraphPosteriorEngine, GraphPrior, has_edge, mask_is_dag, set_edge,
};
use crate::graph_score::{initial_scored_mask, score_dag_mask, tabular_score_data};

/// Structure MCMC with single-edge add / delete / reverse proposals.
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
        }
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
        prior.constraints.validate()?;
        let n = variables.len();
        if n == 0 {
            return Err(DiscoveryError::unsupported(
                "structure MCMC requires at least one variable",
            ));
        }
        crate::graph_posterior::ensure_directed_mask_fits(
            n,
            "structure MCMC adjacency exceeds 63 directed edges",
        )?;
        if !matches!(score_family, GraphScoreFamily::GaussianBic) {
            return Err(DiscoveryError::unsupported(
                "structure MCMC currently supports GaussianBic only",
            ));
        }
        let schedule = self.schedule();
        schedule.require_min_chains(2, "structure MCMC requires at least 2 chains for R-hat")?;

        let score_data = tabular_score_data(data, variables)?;
        let pairs = candidate_directed_list(n, self.candidate_pairs.as_deref());
        if pairs.is_empty() {
            return Err(DiscoveryError::unsupported(
                "structure MCMC has empty candidate edge list",
            ));
        }

        let (initial_mask, initial_score) =
            initial_scored_mask(n, &score_data, score_family, prior, variables)?;

        let (n_chains, n_warmup, n_draws, thin) = schedule.as_usize();
        let n_params = pairs.len();

        let (traces, sample_masks, rejected) =
            run_parallel_mask_chains(ctx, n_chains, n_draws, n_params, |chain, inner| {
                let mut rng = inner.rng.stream_for(StreamDomain::McmcStructure, chain as u64);
                let mut walker = StructureWalker {
                    n,
                    pairs: &pairs,
                    score_data: &score_data,
                    cache: LocalScoreCache::new(crate::graph_score::graph_score_cache_key(
                        score_family,
                        n,
                        score_data.n_rows,
                    )),
                    prior,
                    variables,
                    mask: initial_mask,
                    cur: initial_score,
                    pending: initial_mask,
                };
                let run = run_mask_chain(
                    &MaskChainPlan { chain_index: chain, n_warmup, n_draws, thin, n_params },
                    inner,
                    &mut rng,
                    &mut walker,
                )?;
                Ok((run.traces, run.samples, run.rejected))
            })?;
        if ctx.cancellation.is_cancelled() {
            return Err(DiscoveryError::Cancelled);
        }

        FinishMaskPosterior {
            n,
            schedule: &schedule,
            traces: &traces,
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

/// Hastings factor `q(old|new) / q(new|old)` of one structure move.
///
/// Add selects a directed pair and always inserts it (`q = 1/|pairs|`); the reverse delete
/// succeeds on only half of the selections of that pair. Delete is the reciprocal. Reverse
/// is symmetric when both directions are candidates. The sampler does **not** multiply by
/// this factor (it uses 1), so these values describe the kernel, not the chain that runs.
#[must_use]
pub(crate) fn structure_kernel_ratio(move_kind: StructureMove) -> f64 {
    match move_kind {
        StructureMove::Add => 0.5,
        StructureMove::Delete => 2.0,
        StructureMove::Reverse => 1.0,
    }
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

struct StructureWalker<'a> {
    n: usize,
    pairs: &'a [(usize, usize)],
    score_data: &'a GraphScoreData,
    cache: LocalScoreCache,
    prior: &'a GraphPrior,
    variables: &'a [VariableId],
    mask: u64,
    cur: f64,
    pending: u64,
}

impl MaskChain for StructureWalker<'_> {
    type Sample = u64;

    fn current_score(&self) -> f64 {
        self.cur
    }

    fn accept_policy(&self) -> MaskAccept {
        // `propose_structure` still returns the kernel ratio. This policy does not
        // read it, so add/delete keep Hastings factor 1.
        MaskAccept::FiniteOnly
    }

    fn cancel_policy(&self) -> MaskCancel {
        MaskCancel::Break
    }

    fn propose(&mut self, rng: &mut CausalRng) -> u64 {
        let (prop, _kernel_ratio, rejected) = propose_structure(self.mask, self.n, self.pairs, rng);
        self.pending = prop;
        rejected
    }

    fn proposed_score(&mut self) -> Option<f64> {
        score_dag_mask(
            self.pending,
            self.n,
            self.score_data,
            &mut self.cache,
            self.prior,
            self.variables,
        )
    }

    fn commit(&mut self, score: f64) {
        self.mask = self.pending;
        self.cur = score;
    }

    fn record(&self, row: &mut [f64]) {
        for (p, &(i, j)) in self.pairs.iter().enumerate() {
            row[p] = if has_edge(self.mask, self.n, i, j) { 1.0 } else { 0.0 };
        }
    }

    fn sample(&self) -> u64 {
        self.mask
    }
}

/// Propose add/delete/reverse; returns `(new_mask, kernel Hastings ratio, rejected)`.
///
/// `rejected` counts a proposal that is not a DAG. The ratio is [`structure_kernel_ratio`];
/// the chain applies 1 instead, so seeds keep their historical graphs.
fn propose_structure(
    mask: u64,
    n: usize,
    pairs: &[(usize, usize)],
    rng: &mut CausalRng,
) -> (u64, f64, u64) {
    if pairs.is_empty() {
        return (mask, 1.0, 0);
    }
    let idx = crate::indexing::bounded_index(rng.next_u64(), pairs.len());
    let (i, j) = pairs[idx];
    let forward = has_edge(mask, n, i, j);
    let backward = has_edge(mask, n, j, i);
    let u = rng.next_f64();
    if !forward && !backward {
        let prop = set_edge(mask, n, i, j, true);
        let ratio = structure_kernel_ratio(StructureMove::Add);
        if mask_is_dag(prop, n) { (prop, ratio, 0) } else { (mask, ratio, 1) }
    } else if forward && !backward {
        if u < 0.5 {
            (set_edge(mask, n, i, j, false), structure_kernel_ratio(StructureMove::Delete), 0)
        } else {
            let mut prop = set_edge(mask, n, i, j, false);
            prop = set_edge(prop, n, j, i, true);
            let ratio = structure_kernel_ratio(StructureMove::Reverse);
            if mask_is_dag(prop, n) { (prop, ratio, 0) } else { (mask, ratio, 1) }
        }
    } else if backward && !forward {
        if u < 0.5 {
            (set_edge(mask, n, j, i, false), structure_kernel_ratio(StructureMove::Delete), 0)
        } else {
            let mut prop = set_edge(mask, n, j, i, false);
            prop = set_edge(prop, n, i, j, true);
            let ratio = structure_kernel_ratio(StructureMove::Reverse);
            if mask_is_dag(prop, n) { (prop, ratio, 0) } else { (mask, ratio, 1) }
        }
    } else {
        (set_edge(mask, n, i, j, false), structure_kernel_ratio(StructureMove::Delete), 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use antecedent_data::TabularData;

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
        let sk_xy = post.edge_marginals[1] + post.edge_marginals[n];
        let sk_xz = post.edge_marginals[2] + post.edge_marginals[2 * n];
        assert!(sk_xy > 0.4, "P(X—Y)={sk_xy}");
        assert!(sk_xz > 0.4, "P(X—Z)={sk_xz}");
        assert!(crate::graph_posterior::allows_graph_posterior(&post.diagnostics));
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
            for mask in posterior.adjacency.iter() {
                assert!(has_edge(*mask, 3, 2, 1));
                assert!(has_edge(*mask, 3, 1, 0));
                assert!(mask_is_dag(*mask, 3));
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

    #[test]
    fn structure_kernel_ratio_is_not_one_for_add_or_delete() {
        assert_eq!(structure_kernel_ratio(StructureMove::Add), 0.5);
        assert_eq!(structure_kernel_ratio(StructureMove::Delete), 2.0);
        assert_eq!(structure_kernel_ratio(StructureMove::Reverse), 1.0);
    }
}
