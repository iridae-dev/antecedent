//! Shared graph-structure MCMC schedule, parallel driver, and posterior assembly.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use antecedent_core::{CausalRng, ExecutionContext};
use antecedent_prob::{InferenceDiagnostics, kish_ess};

use crate::adjacency_mask::{AdjacencyMask, parent_bits, set_edge};
use crate::error::DiscoveryError;
use crate::graph_posterior::{
    ChainAgreement, GraphChainSummary, GraphMcmcRun, GraphPosterior, MAX_EDGE_MCSE,
    SamplingSummary, accumulate_marginals, graph_chain_agreement, graph_chain_summary,
    mcmc_graph_diagnostics, publish_graph_posterior,
};

/// One chain: flat edge-indicator trace, sampled states, constraint-invalid proposals, and the
/// log target at each kept draw.
///
/// `S` is an [`AdjacencyMask`] for structure and order MCMC, and a contemporaneous/lag
/// pair for the DBN posterior. The three samplers used to each spell this quadruple.
type ChainOutput<S> = (Vec<f64>, Vec<S>, u64, Vec<f64>);

/// Merged `(traces, per-chain samples, rejected proposals, log targets)`.
type MaskChainMerge<S> = (Vec<f64>, Vec<Vec<S>>, u64, Vec<f64>);

/// Shared MCMC schedule knobs for mask-based graph samplers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GraphMcmcSchedule {
    pub n_chains: u32,
    pub n_warmup: u32,
    pub n_draws: u32,
    pub thin: u32,
}

impl GraphMcmcSchedule {
    /// Normalize schedule fields (at least one chain, four draws, thin ≥ 1).
    #[must_use]
    pub fn normalize(n_chains: u32, n_warmup: u32, n_draws: u32, thin: u32) -> Self {
        Self { n_chains: n_chains.max(1), n_warmup, n_draws: n_draws.max(4), thin: thin.max(1) }
    }

    /// Require at least `min` chains (R-hat needs ≥ 2).
    pub fn require_min_chains(
        &self,
        min: u32,
        refuse_msg: &'static str,
    ) -> Result<(), DiscoveryError> {
        if self.n_draws < 4 || self.thin == 0 {
            return Err(DiscoveryError::unsupported(
                "graph MCMC requires at least four draws and positive thinning",
            ));
        }
        if self.n_chains < min {
            return Err(DiscoveryError::unsupported(refuse_msg));
        }
        Ok(())
    }

    /// `(n_chains, n_warmup, n_draws, thin)` as `usize`.
    #[must_use]
    pub fn as_usize(self) -> (usize, usize, usize, usize) {
        (self.n_chains as usize, self.n_warmup as usize, self.n_draws as usize, self.thin as usize)
    }
}

/// Merge per-chunk worker outputs into global edge-indicator traces and samples.
pub(crate) fn merge_chain_outputs<S>(
    n_draws: usize,
    n_params: usize,
    chains: impl IntoIterator<Item = ChainOutput<S>>,
) -> MaskChainMerge<S> {
    let mut traces = Vec::new();
    let mut sample_masks = Vec::new();
    let mut scores = Vec::new();
    let mut rejected = 0u64;
    for (local_traces, local_samples, local_rej, local_scores) in chains {
        debug_assert_eq!(local_traces.len(), n_draws * n_params);
        debug_assert_eq!(local_scores.len(), n_draws);
        rejected += local_rej;
        traces.extend_from_slice(&local_traces);
        scores.extend_from_slice(&local_scores);
        sample_masks.push(local_samples);
    }
    (traces, sample_masks, rejected, scores)
}

/// Run one worker per chain and merge their outputs.
///
/// Each worker runs [`run_mask_chain`]. That runner applies Hastings factor 1.
/// Structure MCMC still computes a kernel ratio and does not pass it in.
pub(crate) fn run_parallel_mask_chains<S, F>(
    ctx: &ExecutionContext,
    n_chains: usize,
    n_draws: usize,
    n_params: usize,
    worker: F,
) -> Result<MaskChainMerge<S>, DiscoveryError>
where
    S: Send,
    F: Fn(usize, &ExecutionContext) -> Result<ChainOutput<S>, DiscoveryError> + Sync,
{
    // Every chain keeps a trace and a score per draw, and the merge copies them once more;
    // both are leased before any chain starts.
    let bytes = n_chains
        .checked_mul(n_draws)
        .and_then(|cells| cells.checked_mul(n_params.saturating_add(1)))
        .and_then(|cells| cells.checked_mul(2 * size_of::<f64>()))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| DiscoveryError::Resource("chain trace size overflow".to_owned()))?;
    let _traces = ctx.acquire_admission(0, bytes).map_err(|error| match error {
        antecedent_core::execution::AdmissionError::Cancelled => DiscoveryError::Cancelled,
        other => DiscoveryError::Resource(format!(
            "admission pool cannot fund the chain traces: {other}"
        )),
    })?;
    // One call per chain index. `map_indexed` assigns chain `i` the stream the worker
    // opens for `i`, and a panic in `worker` propagates with its payload.
    let outputs = ctx.map_indexed(n_chains, |chain, inner| worker(chain, inner))?;
    Ok(merge_chain_outputs(n_draws, n_params, outputs))
}

/// Schedule for one [`MaskChain`] worker.
#[derive(Clone, Copy)]
pub(crate) struct MaskChainPlan {
    pub chain_index: usize,
    pub n_warmup: usize,
    pub n_draws: usize,
    pub thin: usize,
    pub n_params: usize,
}

/// How a missing or non-finite proposal score is treated.
///
/// The Metropolis ratio is always `exp(proposed - current)`. No kernel Hastings
/// factor is multiplied in.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MaskAccept {
    /// Reject without drawing `u` and without counting the rejection.
    FiniteOnly,
    /// Reject without drawing `u`, and count one invalid proposal.
    FiniteOnlyCounted,
    /// Missing score: reject. A proposal while the current score is non-finite is
    /// adopted and is not an accept. Otherwise Metropolis, including a non-finite
    /// proposal (that draw of `u` is historical unless `proposed - current >= 0`).
    AdoptIfCurrentNonFinite,
}

/// Cancellation inside one chain.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MaskCancel {
    /// Leave the step loop. The caller checks the parent token after the merge.
    Break,
    /// Refuse at entry and whenever `step % 64 == 0`.
    AbortEvery64,
}

/// One mask sampler: propose, score, commit, record.
///
/// Order, structure, and DBN MCMC share [`run_mask_chain`]. Each walker keeps its
/// own invalid-proposal count and its own accept policy.
pub(crate) trait MaskChain {
    type Sample;

    fn current_score(&self) -> f64;
    fn accept_policy(&self) -> MaskAccept;
    fn cancel_policy(&self) -> MaskCancel;
    /// Invalid proposals in this draw. A kernel Hastings ratio is not returned.
    fn propose(&mut self, rng: &mut CausalRng) -> u64;
    fn proposed_score(&mut self) -> Option<f64>;
    /// Replace the Metropolis step by a kernel move that keeps the target tempered at `beta`
    /// invariant by itself, returning whether it did: a draw from a full conditional, which needs
    /// no accept test, or a Metropolis–Hastings step with its own.
    fn exact_move(&mut self, _rng: &mut CausalRng, _beta: f64) -> bool {
        false
    }
    fn commit(&mut self, score: f64);
    fn record(&self, row: &mut [f64]);
    fn sample(&self) -> Self::Sample;
}

/// `(extra_rejections, score_to_commit)` of one Metropolis decision at inverse temperature
/// `beta` (`1` is the target itself).
fn mask_step(
    policy: MaskAccept,
    current: f64,
    proposed: Option<f64>,
    beta: f64,
    rng: &mut CausalRng,
) -> (u64, Option<f64>) {
    match policy {
        MaskAccept::FiniteOnly | MaskAccept::FiniteOnlyCounted => match proposed {
            Some(ps) if ps.is_finite() => {
                let log_r = beta * (ps - current);
                let accept = log_r >= 0.0 || rng.next_f64() < log_r.exp();
                (0, accept.then_some(ps))
            }
            _ => (u64::from(matches!(policy, MaskAccept::FiniteOnlyCounted)), None),
        },
        MaskAccept::AdoptIfCurrentNonFinite => match proposed {
            Some(ps) if current.is_finite() => {
                let log_r = beta * (ps - current);
                let accept = log_r >= 0.0 || rng.next_f64() < log_r.exp();
                (0, accept.then_some(ps))
            }
            Some(ps) => (0, Some(ps)),
            None => (0, None),
        },
    }
}

/// Traces, retained samples, and invalid proposals from one chain.
pub(crate) struct MaskChainOutput<S> {
    pub traces: Vec<f64>,
    pub samples: Vec<S>,
    pub rejected: u64,
    /// Log target (before tempering) of the state recorded at each kept draw.
    pub scores: Vec<f64>,
}

/// Inverse temperature of the random walk that scatters a later chain's starting state.
const START_BETA: f64 = 0.1;

/// Default inverse-temperature ladder of the graph samplers: the target and three tempered
/// copies, each half as sharp as the one before.
pub const DEFAULT_TEMPERATURES: [f64; 4] = [1.0, 0.5, 0.25, 0.125];

/// Ladder used in place of [`DEFAULT_TEMPERATURES`] on at least [`EQUIVALENCE_WALK_MIN_NODES`]
/// nodes: eight replicas, each about three quarters as sharp as the one before. The log target
/// of a graph spreads in proportion to the number of variables, so the exchange acceptance of
/// neighbouring replicas at the default spacing falls as the graph grows; a denser ladder keeps
/// it usable and lets the cold replica receive states from the hot ones.
pub(crate) const DENSE_TEMPERATURES: [f64; 8] = [1.0, 0.75, 0.56, 0.42, 0.32, 0.24, 0.18, 0.13];

/// The ladder a sampler on `n` nodes runs: `configured`, except that the default ladder is
/// replaced by [`DENSE_TEMPERATURES`] from [`EQUIVALENCE_WALK_MIN_NODES`] nodes on. Samplers on
/// fewer nodes run the configured ladder unchanged.
pub(crate) fn tempering_ladder(configured: &[f64], n: usize) -> Vec<f64> {
    if n >= EQUIVALENCE_WALK_MIN_NODES && configured == DEFAULT_TEMPERATURES.as_slice() {
        DENSE_TEMPERATURES.to_vec()
    } else {
        configured.to_vec()
    }
}

/// Iterations between replica-exchange attempts.
const SWAP_EVERY: usize = 10;

/// Graphs on at least this many nodes (nine) also take equivalence-class steps (see
/// [`propose_covered_reversal`]). Samplers on fewer nodes draw exactly the random numbers they
/// always did.
pub(crate) const EQUIVALENCE_WALK_MIN_NODES: usize = 9;

/// Probability that an iteration of a sampler on at least [`EQUIVALENCE_WALK_MIN_NODES`] nodes is
/// an equivalence-class step instead of its usual proposal.
pub(crate) const EQUIVALENCE_WALK_PROBABILITY: f64 = 0.25;

/// The covered edges `x -> y` of a DAG: `Pa(y) = Pa(x) ∪ {x}` (Chickering, 1995).
///
/// Reversing a covered edge yields a DAG of the same Markov equivalence class, so the two have
/// the same Gaussian-BIC score, and every two members of a class are joined by a sequence of such
/// reversals. Single-edge moves that propose an arbitrary reversal rarely hit one of the few
/// covered edges, which is why the orientation of an edge inside a class mixes slowly.
pub(crate) fn covered_edges(mask: AdjacencyMask, n: usize) -> Vec<(usize, usize)> {
    let parents: Vec<u16> = (0..n).map(|node| parent_bits(mask, n, node)).collect();
    let mut covered = Vec::new();
    for (y, &pa_y) in parents.iter().enumerate() {
        for x in (0..n).filter(|&x| (pa_y >> x) & 1 == 1) {
            if pa_y == parents[x] | (1 << x) {
                covered.push((x, y));
            }
        }
    }
    covered
}

/// Pick a covered edge `x -> y` of `mask` uniformly: `(x, y, number of covered edges)`, or `None`
/// when there is none.
///
/// Completing the move is [`accept_covered_reversal`], which supplies the Hastings factor
/// `|covered(G)| / |covered(G')|` of the uniform choice.
pub(crate) fn propose_covered_reversal(
    mask: AdjacencyMask,
    n: usize,
    rng: &mut CausalRng,
) -> Option<(usize, usize, usize)> {
    let covered = covered_edges(mask, n);
    if covered.is_empty() {
        return None;
    }
    let (x, y) = covered[crate::indexing::bounded_index(rng.next_u64(), covered.len())];
    Some((x, y, covered.len()))
}

/// `mask` with the edge `x -> y` replaced by `y -> x`.
pub(crate) fn reversed_edge(mask: AdjacencyMask, n: usize, x: usize, y: usize) -> AdjacencyMask {
    set_edge(set_edge(mask, n, x, y, false), n, y, x, true)
}

/// Metropolis–Hastings decision of a covered-edge reversal at inverse temperature `beta`, from a
/// state of log target `current` with `covered_before` covered edges to `after` with log target
/// `proposed`.
pub(crate) fn accept_covered_reversal(
    (beta, current, proposed): (f64, f64, f64),
    covered_before: usize,
    after: AdjacencyMask,
    n: usize,
    rng: &mut CausalRng,
) -> bool {
    let covered_after = covered_edges(after, n).len();
    let log_r = beta * (proposed - current) + (covered_before as f64).ln()
        - (covered_after.max(1) as f64).ln();
    log_r >= 0.0 || rng.next_f64() < log_r.exp()
}

/// Refuse a temperature ladder that is not `1 = beta_0 > beta_1 > ... > 0`.
///
/// # Errors
///
/// [`DiscoveryError::Unsupported`] naming the violated rule.
pub(crate) fn validate_temperatures(ladder: &[f64]) -> Result<(), DiscoveryError> {
    use std::cmp::Ordering::{Greater, Less};
    if ladder.first() != Some(&1.0) {
        return Err(DiscoveryError::unsupported(
            "temperature ladder must start at inverse temperature 1 (the target)",
        ));
    }
    if ladder.windows(2).any(|w| w[1].partial_cmp(&w[0]) != Some(Less))
        || ladder.iter().any(|b| b.partial_cmp(&0.0) != Some(Greater))
    {
        return Err(DiscoveryError::unsupported(
            "temperature ladder must be strictly decreasing and positive",
        ));
    }
    Ok(())
}

/// Warm up, then keep thinned draws, with a single replica. Hastings factor is 1.
///
/// Chain 0 starts at the walker's state. Later chains first take a `(4 * n_params).min(400)`-step
/// walk at inverse temperature [`START_BETA`], so they begin at dispersed states that are still
/// biased toward the target rather than at arbitrary graphs.
pub(crate) fn run_mask_chain<C: MaskChain>(
    plan: &MaskChainPlan,
    ctx: &ExecutionContext,
    rng: &mut CausalRng,
    chain: &mut C,
) -> Result<MaskChainOutput<C::Sample>, DiscoveryError> {
    run_tempered_mask_chain(plan, ctx, rng, std::slice::from_mut(chain), &[1.0])
}

/// Parallel tempering (replica exchange): one replica per entry of `ladder`, replica `k`
/// targeting the posterior raised to `ladder[k]`. Every iteration each replica takes one step at
/// its own temperature; every [`SWAP_EVERY`] iterations neighbouring temperatures exchange
/// replicas with the Metropolis probability `min(1, exp((b_k - b_{k+1}) (f_{k+1} - f_k)))`, where
/// `f` is a replica's current log target, so the target (`ladder[0] == 1`) stays exactly
/// invariant. Temperatures swap between replicas rather than states between temperatures. Only
/// the replica currently at temperature 1 is recorded and counted.
pub(crate) fn run_tempered_mask_chain<C: MaskChain>(
    plan: &MaskChainPlan,
    ctx: &ExecutionContext,
    rng: &mut CausalRng,
    replicas: &mut [C],
    ladder: &[f64],
) -> Result<MaskChainOutput<C::Sample>, DiscoveryError> {
    debug_assert_eq!(replicas.len(), ladder.len());
    let cancel = replicas[0].cancel_policy();
    if matches!(cancel, MaskCancel::AbortEvery64) && ctx.cancellation.is_cancelled() {
        return Err(DiscoveryError::Cancelled);
    }
    if plan.chain_index > 0 {
        for chain in replicas.iter_mut() {
            for _ in 0..(4 * plan.n_params).min(400) {
                let _rejected = chain.propose(rng);
                let (_, adopt) = mask_step(
                    chain.accept_policy(),
                    chain.current_score(),
                    chain.proposed_score(),
                    START_BETA,
                    rng,
                );
                if let Some(score) = adopt {
                    chain.commit(score);
                }
            }
        }
    }
    // `holder[k]` is the replica currently at temperature `ladder[k]`.
    let mut holder: Vec<usize> = (0..replicas.len()).collect();
    let mut traces = vec![0.0f64; plan.n_draws * plan.n_params];
    let mut scores = vec![0.0f64; plan.n_draws];
    let mut samples = Vec::new();
    let mut rejected = 0u64;
    let total_steps = plan.n_warmup + plan.n_draws * plan.thin;
    let mut kept = 0usize;
    for step in 0..total_steps {
        match cancel {
            MaskCancel::Break => {
                if ctx.cancellation.is_cancelled() {
                    break;
                }
            }
            MaskCancel::AbortEvery64 => {
                if step % 64 == 0 && ctx.cancellation.is_cancelled() {
                    return Err(DiscoveryError::Cancelled);
                }
            }
        }
        for (slot, &beta) in ladder.iter().enumerate() {
            let chain = &mut replicas[holder[slot]];
            if chain.exact_move(rng, beta) {
                continue;
            }
            let invalid = chain.propose(rng);
            let (extra, adopt) = mask_step(
                chain.accept_policy(),
                chain.current_score(),
                chain.proposed_score(),
                beta,
                rng,
            );
            if slot == 0 {
                rejected += invalid + extra;
            }
            if let Some(score) = adopt {
                chain.commit(score);
            }
        }
        if replicas.len() > 1 && step % SWAP_EVERY == 0 {
            for slot in (step / SWAP_EVERY % 2..replicas.len() - 1).step_by(2) {
                let (hot, cold) = (&replicas[holder[slot + 1]], &replicas[holder[slot]]);
                let log_r = (ladder[slot] - ladder[slot + 1])
                    * (hot.current_score() - cold.current_score());
                if log_r >= 0.0 || rng.next_f64() < log_r.exp() {
                    holder.swap(slot, slot + 1);
                }
            }
        }
        if step >= plan.n_warmup && (step - plan.n_warmup) % plan.thin == 0 && kept < plan.n_draws {
            let cold = &replicas[holder[0]];
            cold.record(&mut traces[kept * plan.n_params..(kept + 1) * plan.n_params]);
            scores[kept] = cold.current_score();
            samples.push(cold.sample());
            kept += 1;
        }
    }
    Ok(MaskChainOutput { traces, samples, rejected, scores })
}

/// Diagnostics from edge-indicator traces, then optional publish gate.
///
/// Besides R̂ / ESS the gate requires every edge marginal's Monte-Carlo standard error to be
/// within [`MAX_EDGE_MCSE`], and the chains to agree ([`graph_chain_agreement`]): no chain's
/// marginal of an edge far from the pooled one, every edge changed often enough in every chain to
/// be judged, and a rank-normalised split R̂ of the log target within bounds. Each criterion
/// leaves a `gate_*` note with its outcome. Returns the diagnostics and the per-indicator chain summary (MCSE,
/// split R̂ and bulk ESS, each aligned with the trace parameters).
pub(crate) fn diagnostics_from_traces(
    schedule: &GraphMcmcSchedule,
    traces: &[f64],
    scores: &[f64],
    n_params: usize,
    require_gate: bool,
    refuse_msg: &'static str,
) -> Result<(InferenceDiagnostics, GraphChainSummary), DiscoveryError> {
    let (n_chains, _, n_draws, _) = schedule.as_usize();
    let summary = graph_chain_summary(traces, n_chains, n_draws, n_params);
    let diagnostics = mcmc_graph_diagnostics(&GraphMcmcRun {
        n_chains: schedule.n_chains,
        n_warmup: schedule.n_warmup,
        n_draws: schedule.n_draws,
        ess_bulk_min: summary.ess_bulk_min,
        ess_tail_min: summary.ess_tail_min,
        rhat_max: summary.rhat_max,
        n_divergences: 0,
        converged: true,
        chains_moved: summary.all_chains_moved,
    });
    let mut diagnostics = publish_graph_posterior(diagnostics, false, refuse_msg)?;
    let mcse_max = summary.mcse.iter().copied().fold(0.0_f64, f64::max);
    diagnostics.notes.push(Arc::from(format!("edge_mcse_max={mcse_max:.4}")));
    if summary.rhat_max.is_infinite() {
        diagnostics.notes.push(Arc::from(
            "chains disagree on an edge indicator that never varies within a chain (rhat=inf)",
        ));
    }
    let mcse_ok = mcse_max <= MAX_EDGE_MCSE;
    let agreement = graph_chain_agreement(traces, scores, n_chains, n_draws, n_params);
    push_gate_notes(&mut diagnostics, &summary, mcse_ok, &agreement);
    if !mcse_ok || !agreement.agrees() {
        diagnostics.converged = false;
    }
    if require_gate && !diagnostics.converged {
        return Err(DiscoveryError::unsupported(refuse_msg));
    }
    Ok((diagnostics, summary))
}

/// Record the outcome of each publication criterion as a `gate_<name>=pass|fail` note.
fn push_gate_notes(
    diagnostics: &mut InferenceDiagnostics,
    summary: &GraphChainSummary,
    mcse_ok: bool,
    agreement: &ChainAgreement,
) {
    let verdict = |ok: bool| if ok { "pass" } else { "fail" };
    let rhat_ok = summary.rhat_max.is_finite() && summary.rhat_max < 1.2;
    let ess_ok = summary.ess_bulk_min.is_finite() && summary.ess_bulk_min > 10.0;
    let notes = [
        format!("gate_rhat={}", verdict(rhat_ok)),
        format!("gate_ess={}", verdict(ess_ok)),
        format!("gate_moved={}", verdict(summary.all_chains_moved)),
        format!("gate_mcse={}", verdict(mcse_ok)),
        format!(
            "gate_chain_deviation={} z={:.2} at_param={}",
            verdict(agreement.deviation_ok()),
            agreement.max_deviation_z,
            agreement.worst_param.map_or_else(|| "none".to_string(), |p| p.to_string())
        ),
        format!(
            "indicator_changes={} unresolved={}",
            verdict(agreement.resolved()),
            agreement.unresolved
        ),
        format!(
            "gate_score_rhat={} rhat={:.3}",
            verdict(agreement.score_ok()),
            agreement.score_rhat
        ),
    ];
    diagnostics.notes.extend(notes.into_iter().map(|n| Arc::from(n.as_str())));
}

/// Aggregate visit-frequency weights into a [`GraphPosterior`].
pub(crate) fn aggregate_mask_posterior(
    n: usize,
    sample_masks: &[Vec<AdjacencyMask>],
    diagnostics: InferenceDiagnostics,
    rejected: u64,
    empty_msg: &'static str,
) -> Result<GraphPosterior, DiscoveryError> {
    let mut counts: HashMap<AdjacencyMask, u64> = HashMap::new();
    for chain_samples in sample_masks {
        for &m in chain_samples {
            *counts.entry(m).or_insert(0) += 1;
        }
    }
    let total: f64 = counts.values().map(|&c| c as f64).sum();
    if total.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return Err(DiscoveryError::unsupported(empty_msg));
    }
    // Atoms in mask order, so one seed gives one posterior object (a hash map's order differs
    // from run to run).
    let mut counted: Vec<(AdjacencyMask, u64)> = counts.into_iter().collect();
    counted.sort_unstable_by_key(|&(mask, _)| mask);
    let mut masks = Vec::with_capacity(counted.len());
    let mut weights = Vec::with_capacity(counted.len());
    for (m, c) in counted {
        masks.push(m);
        weights.push(c as f64 / total);
    }
    let ess = kish_ess(&weights);
    let (edge, orient) = accumulate_marginals(n, &weights, &masks);
    GraphPosterior::from_atoms(
        n,
        weights,
        masks,
        edge,
        orient,
        SamplingSummary { ess, diagnostics, rejected_invalid: rejected },
    )
}

/// Diagnostics + aggregation for mask-based parallel MCMC.
pub(crate) struct FinishMaskPosterior<'a> {
    pub n: usize,
    pub schedule: &'a GraphMcmcSchedule,
    pub traces: &'a [f64],
    /// Log target at each kept draw, chain-major (one value per draw).
    pub scores: &'a [f64],
    pub sample_masks: &'a [Vec<AdjacencyMask>],
    pub rejected: u64,
    pub n_params: usize,
    /// `(from, to)` of each trace parameter, in trace order (for placing per-edge MCSE).
    pub param_edges: &'a [(usize, usize)],
    pub require_gate: bool,
    pub refuse_msg: &'static str,
    pub empty_msg: &'static str,
}

impl FinishMaskPosterior<'_> {
    pub fn run(self) -> Result<GraphPosterior, DiscoveryError> {
        let (diagnostics, summary) = diagnostics_from_traces(
            self.schedule,
            self.traces,
            self.scores,
            self.n_params,
            self.require_gate,
            self.refuse_msg,
        )?;
        // Edges outside the sampled parameters never vary (forbidden / excluded): their
        // Monte-Carlo error is zero and they report the constant-indicator R̂ and ESS.
        let cells = self.n * self.n;
        let mut edge_mcse = vec![0.0; cells];
        let mut edge_rhat = vec![1.0; cells];
        let mut edge_ess =
            vec![f64::from(self.schedule.n_chains) * f64::from(self.schedule.n_draws); cells];
        for (p, &(i, j)) in self.param_edges.iter().enumerate() {
            edge_mcse[i * self.n + j] = summary.mcse[p];
            edge_rhat[i * self.n + j] = summary.rhat[p];
            edge_ess[i * self.n + j] = summary.ess_bulk[p];
        }
        aggregate_mask_posterior(
            self.n,
            self.sample_masks,
            diagnostics,
            self.rejected,
            self.empty_msg,
        )?
        .with_edge_mcse(edge_mcse)?
        .with_edge_chain_diagnostics(edge_rhat, edge_ess)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn dense_ladder_replaces_only_the_default_above_eight_nodes() {
        assert!(validate_temperatures(&DENSE_TEMPERATURES).is_ok());
        assert_eq!(tempering_ladder(&DEFAULT_TEMPERATURES, 8), DEFAULT_TEMPERATURES.to_vec());
        assert_eq!(tempering_ladder(&DEFAULT_TEMPERATURES, 9), DENSE_TEMPERATURES.to_vec());
        assert_eq!(tempering_ladder(&[1.0, 0.3], 12), vec![1.0, 0.3]);
    }

    #[test]
    fn worker_partition_respects_nondivisible_thread_limit() {
        let workers = AtomicUsize::new(0);
        let ctx = ExecutionContext::production(0, 2);
        let (traces, samples, _, _) = run_parallel_mask_chains(&ctx, 5, 4, 1, |chain, _| {
            workers.fetch_add(1, Ordering::Relaxed);
            Ok((vec![chain as f64; 4], vec![chain as u64; 4], 0, vec![0.0; 4]))
        })
        .unwrap();
        assert_eq!(workers.load(Ordering::Relaxed), 5);
        assert_eq!(samples[3], vec![3, 3, 3, 3]);
        assert_eq!(traces.len(), 20);
        assert_eq!(samples.len(), 5);
        assert!(samples.iter().all(|chain| chain.len() == 4));
    }

    /// Slowly mixing indicators (two long blocks per chain, ESS of a handful of draws) carry a
    /// Monte-Carlo standard error far above the publication bar even though the chains
    /// agree on the mean, so the gate must refuse them.
    #[test]
    fn slow_mixing_indicator_fails_the_mcse_gate() {
        let (n_chains, n_draws) = (2usize, 50usize);
        let mut traces = vec![0.0; n_chains * n_draws];
        for c in 0..n_chains {
            for d in 0..n_draws {
                let high = (d >= n_draws / 2) == (c == 0);
                traces[c * n_draws + d] = f64::from(u8::from(high));
            }
        }
        let summary = graph_chain_summary(&traces, n_chains, n_draws, 1);
        assert!(summary.mcse[0] > MAX_EDGE_MCSE, "mcse={}", summary.mcse[0]);

        let schedule = GraphMcmcSchedule { n_chains: 2, n_warmup: 0, n_draws: 50, thin: 1 };
        let scores = vec![0.0; n_chains * n_draws];
        assert!(diagnostics_from_traces(&schedule, &traces, &scores, 1, true, "refused").is_err());
        let (d, _) =
            diagnostics_from_traces(&schedule, &traces, &scores, 1, false, "refused").unwrap();
        assert!(!d.converged);
        assert!(d.notes.iter().any(|n| n.starts_with("edge_mcse_max=")));
    }

    /// Covered edges against hand-checked graphs: a fork's edges are all covered, a chain's last
    /// edge and a collider's edges are not.
    #[test]
    fn covered_edges_match_hand_checked_graphs() {
        let n = 3;
        let build = |edges: &[(usize, usize)]| {
            edges.iter().fold(AdjacencyMask::EMPTY, |m, &(a, b)| set_edge(m, n, a, b, true))
        };
        assert_eq!(covered_edges(build(&[(0, 1), (0, 2)]), n), [(0, 1), (0, 2)]);
        assert_eq!(covered_edges(build(&[(0, 1), (1, 2)]), n), [(0, 1)]);
        assert!(covered_edges(build(&[(0, 2), (1, 2)]), n).is_empty());
        // Reversing a covered edge keeps the covered edge covered, in the other direction.
        let fork = build(&[(0, 1), (0, 2)]);
        let flipped = reversed_edge(fork, n, 0, 1);
        assert_eq!(covered_edges(flipped, n), [(1, 0)]);
    }

    /// The ladder must start at the target and cool strictly.
    #[test]
    fn temperature_ladders_are_validated() {
        assert!(validate_temperatures(&DEFAULT_TEMPERATURES).is_ok());
        assert!(validate_temperatures(&[1.0]).is_ok());
        for bad in [
            &[][..],
            &[0.5][..],
            &[1.0, 1.0][..],
            &[1.0, 0.5, 0.7][..],
            &[1.0, 0.0][..],
            &[1.0, f64::NAN][..],
        ] {
            assert!(validate_temperatures(bad).is_err(), "{bad:?}");
        }
    }

    /// An improving move accepts without drawing `u`. Multiplying the structure
    /// add-move ratio 0.5 would make `0.1 + ln(0.5)` negative and might reject.
    #[test]
    fn mask_step_keeps_hastings_factor_one() {
        let mut used = CausalRng::from_seed(7);
        let mut untouched = CausalRng::from_seed(7);
        let (extra, adopt) = mask_step(MaskAccept::FiniteOnly, 0.0, Some(0.1), 1.0, &mut used);
        assert_eq!(extra, 0);
        assert_eq!(adopt, Some(0.1));
        assert_eq!(used.next_f64().to_bits(), untouched.next_f64().to_bits());

        let mut draw = CausalRng::from_seed(7);
        let mut expected = CausalRng::from_seed(7);
        let u = expected.next_f64();
        let (extra, adopt) = mask_step(MaskAccept::FiniteOnly, 0.0, Some(-1.0), 1.0, &mut draw);
        assert_eq!(extra, 0);
        assert_eq!(adopt.is_some(), u < (-1.0f64).exp());
        assert_eq!(draw.next_f64().to_bits(), expected.next_f64().to_bits());

        let mut counted = CausalRng::from_seed(3);
        let mut counted_copy = CausalRng::from_seed(3);
        let (extra, adopt) = mask_step(MaskAccept::FiniteOnlyCounted, 1.0, None, 1.0, &mut counted);
        assert_eq!((extra, adopt), (1, None));
        let (extra, adopt) = mask_step(MaskAccept::FiniteOnly, 1.0, None, 1.0, &mut counted_copy);
        assert_eq!((extra, adopt), (0, None));
        assert_eq!(counted.next_f64().to_bits(), counted_copy.next_f64().to_bits());

        let mut adopt_rng = CausalRng::from_seed(9);
        let mut adopt_copy = CausalRng::from_seed(9);
        let (extra, adopt) = mask_step(
            MaskAccept::AdoptIfCurrentNonFinite,
            f64::NAN,
            Some(2.5),
            1.0,
            &mut adopt_rng,
        );
        assert_eq!((extra, adopt), (0, Some(2.5)));
        let (extra, adopt) = mask_step(
            MaskAccept::AdoptIfCurrentNonFinite,
            0.0,
            Some(f64::NAN),
            1.0,
            &mut adopt_rng,
        );
        assert_eq!(extra, 0);
        assert!(adopt.is_none());
        let _ = adopt_copy.next_f64();
        assert_eq!(adopt_rng.next_f64().to_bits(), adopt_copy.next_f64().to_bits());
    }
}
