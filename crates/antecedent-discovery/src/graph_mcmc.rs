//! Shared graph-structure MCMC schedule, parallel driver, and posterior assembly.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use antecedent_core::{CausalRng, ExecutionContext};
use antecedent_prob::{InferenceDiagnostics, kish_ess};

use crate::error::DiscoveryError;
use crate::graph_posterior::{
    GraphMcmcRun, GraphPosterior, MAX_EDGE_MCSE, SamplingSummary, accumulate_marginals,
    graph_chain_summary, mcmc_graph_diagnostics, publish_graph_posterior,
};

/// One chain: flat edge-indicator trace, sampled states, constraint-invalid proposals.
///
/// `S` is a `u64` adjacency for structure and order MCMC, and a contemporaneous/lag
/// pair for the DBN posterior. The three samplers used to each spell this triple.
type ChainOutput<S> = (Vec<f64>, Vec<S>, u64);

/// Merged `(traces, per-chain samples, rejected proposals)`.
type MaskChainMerge<S> = (Vec<f64>, Vec<Vec<S>>, u64);

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
    let mut rejected = 0u64;
    for (local_traces, local_samples, local_rej) in chains {
        debug_assert_eq!(local_traces.len(), n_draws * n_params);
        rejected += local_rej;
        traces.extend_from_slice(&local_traces);
        sample_masks.push(local_samples);
    }
    (traces, sample_masks, rejected)
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
    fn commit(&mut self, score: f64);
    fn record(&self, row: &mut [f64]);
    fn sample(&self) -> Self::Sample;
}

/// `(extra_rejections, score_to_commit)`.
fn mask_step(
    policy: MaskAccept,
    current: f64,
    proposed: Option<f64>,
    rng: &mut CausalRng,
) -> (u64, Option<f64>) {
    match policy {
        MaskAccept::FiniteOnly | MaskAccept::FiniteOnlyCounted => match proposed {
            Some(ps) if ps.is_finite() => {
                let log_r = ps - current;
                let accept = log_r >= 0.0 || rng.next_f64() < log_r.exp();
                (0, accept.then_some(ps))
            }
            _ => (u64::from(matches!(policy, MaskAccept::FiniteOnlyCounted)), None),
        },
        MaskAccept::AdoptIfCurrentNonFinite => match proposed {
            Some(ps) if current.is_finite() => {
                let log_r = ps - current;
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
}

/// Warm up, then keep thinned draws. Hastings factor is 1.
///
/// Chain 0 starts at the walker's state. Later chains take the same
/// `(4 * n_params).min(400)` finite-score walk the three samplers used.
pub(crate) fn run_mask_chain<C: MaskChain>(
    plan: &MaskChainPlan,
    ctx: &ExecutionContext,
    rng: &mut CausalRng,
    chain: &mut C,
) -> Result<MaskChainOutput<C::Sample>, DiscoveryError> {
    if matches!(chain.cancel_policy(), MaskCancel::AbortEvery64) && ctx.cancellation.is_cancelled()
    {
        return Err(DiscoveryError::Cancelled);
    }
    if plan.chain_index > 0 {
        for _ in 0..(4 * plan.n_params).min(400) {
            let _rejected = chain.propose(rng);
            if let Some(ps) = chain.proposed_score().filter(|score| score.is_finite()) {
                chain.commit(ps);
            }
        }
    }
    let mut traces = vec![0.0f64; plan.n_draws * plan.n_params];
    let mut samples = Vec::new();
    let mut rejected = 0u64;
    let total_steps = plan.n_warmup + plan.n_draws * plan.thin;
    let mut kept = 0usize;
    for step in 0..total_steps {
        match chain.cancel_policy() {
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
        rejected += chain.propose(rng);
        let (extra, adopt) =
            mask_step(chain.accept_policy(), chain.current_score(), chain.proposed_score(), rng);
        rejected += extra;
        if let Some(score) = adopt {
            chain.commit(score);
        }
        if step >= plan.n_warmup && (step - plan.n_warmup) % plan.thin == 0 && kept < plan.n_draws {
            chain.record(&mut traces[kept * plan.n_params..(kept + 1) * plan.n_params]);
            samples.push(chain.sample());
            kept += 1;
        }
    }
    Ok(MaskChainOutput { traces, samples, rejected })
}

/// Diagnostics from edge-indicator traces, then optional publish gate.
///
/// Besides R̂ / ESS the gate requires every edge marginal's Monte-Carlo standard error to be
/// within [`MAX_EDGE_MCSE`]. Returns the diagnostics and the per-indicator MCSE (aligned with
/// the trace parameters).
pub(crate) fn diagnostics_from_traces(
    schedule: &GraphMcmcSchedule,
    traces: &[f64],
    n_params: usize,
    require_gate: bool,
    refuse_msg: &'static str,
) -> Result<(InferenceDiagnostics, Vec<f64>), DiscoveryError> {
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
    if mcse_max > MAX_EDGE_MCSE || mcse_max.is_nan() {
        diagnostics.converged = false;
    }
    if require_gate && !diagnostics.converged {
        return Err(DiscoveryError::unsupported(refuse_msg));
    }
    Ok((diagnostics, summary.mcse))
}

/// Aggregate visit-frequency weights into a [`GraphPosterior`].
pub(crate) fn aggregate_mask_posterior(
    n: usize,
    sample_masks: &[Vec<u64>],
    diagnostics: InferenceDiagnostics,
    rejected: u64,
    empty_msg: &'static str,
) -> Result<GraphPosterior, DiscoveryError> {
    let mut counts: HashMap<u64, u64> = HashMap::new();
    for chain_samples in sample_masks {
        for &m in chain_samples {
            *counts.entry(m).or_insert(0) += 1;
        }
    }
    let total: f64 = counts.values().map(|&c| c as f64).sum();
    if total.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return Err(DiscoveryError::unsupported(empty_msg));
    }
    let mut masks = Vec::with_capacity(counts.len());
    let mut weights = Vec::with_capacity(counts.len());
    for (m, c) in counts {
        masks.push(m);
        weights.push(c as f64 / total);
    }
    let ess = kish_ess(&weights);
    let (edge, orient) = accumulate_marginals(n, &weights, &masks);
    GraphPosterior::new(
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
    pub sample_masks: &'a [Vec<u64>],
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
        let (diagnostics, param_mcse) = diagnostics_from_traces(
            self.schedule,
            self.traces,
            self.n_params,
            self.require_gate,
            self.refuse_msg,
        )?;
        // Edges outside the sampled parameters never vary (forbidden / excluded), so their
        // Monte-Carlo error is zero.
        let mut edge_mcse = vec![0.0; self.n * self.n];
        for (&(i, j), &m) in self.param_edges.iter().zip(&param_mcse) {
            edge_mcse[i * self.n + j] = m;
        }
        aggregate_mask_posterior(
            self.n,
            self.sample_masks,
            diagnostics,
            self.rejected,
            self.empty_msg,
        )?
        .with_edge_mcse(edge_mcse)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn worker_partition_respects_nondivisible_thread_limit() {
        let workers = AtomicUsize::new(0);
        let ctx = ExecutionContext::production(0, 2);
        let (traces, samples, _) = run_parallel_mask_chains(&ctx, 5, 4, 1, |chain, _| {
            workers.fetch_add(1, Ordering::Relaxed);
            Ok((vec![chain as f64; 4], vec![chain as u64; 4], 0))
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
        assert!(diagnostics_from_traces(&schedule, &traces, 1, true, "refused").is_err());
        let (d, _) = diagnostics_from_traces(&schedule, &traces, 1, false, "refused").unwrap();
        assert!(!d.converged);
        assert!(d.notes.iter().any(|n| n.starts_with("edge_mcse_max=")));
    }

    /// An improving move accepts without drawing `u`. Multiplying the structure
    /// add-move ratio 0.5 would make `0.1 + ln(0.5)` negative and might reject.
    #[test]
    fn mask_step_keeps_hastings_factor_one() {
        let mut used = CausalRng::from_seed(7);
        let mut untouched = CausalRng::from_seed(7);
        let (extra, adopt) = mask_step(MaskAccept::FiniteOnly, 0.0, Some(0.1), &mut used);
        assert_eq!(extra, 0);
        assert_eq!(adopt, Some(0.1));
        assert_eq!(used.next_f64().to_bits(), untouched.next_f64().to_bits());

        let mut draw = CausalRng::from_seed(7);
        let mut expected = CausalRng::from_seed(7);
        let u = expected.next_f64();
        let (extra, adopt) = mask_step(MaskAccept::FiniteOnly, 0.0, Some(-1.0), &mut draw);
        assert_eq!(extra, 0);
        assert_eq!(adopt.is_some(), u < (-1.0f64).exp());
        assert_eq!(draw.next_f64().to_bits(), expected.next_f64().to_bits());

        let mut counted = CausalRng::from_seed(3);
        let mut counted_copy = CausalRng::from_seed(3);
        let (extra, adopt) = mask_step(MaskAccept::FiniteOnlyCounted, 1.0, None, &mut counted);
        assert_eq!((extra, adopt), (1, None));
        let (extra, adopt) = mask_step(MaskAccept::FiniteOnly, 1.0, None, &mut counted_copy);
        assert_eq!((extra, adopt), (0, None));
        assert_eq!(counted.next_f64().to_bits(), counted_copy.next_f64().to_bits());

        let mut adopt_rng = CausalRng::from_seed(9);
        let mut adopt_copy = CausalRng::from_seed(9);
        let (extra, adopt) =
            mask_step(MaskAccept::AdoptIfCurrentNonFinite, f64::NAN, Some(2.5), &mut adopt_rng);
        assert_eq!((extra, adopt), (0, Some(2.5)));
        let (extra, adopt) =
            mask_step(MaskAccept::AdoptIfCurrentNonFinite, 0.0, Some(f64::NAN), &mut adopt_rng);
        assert_eq!(extra, 0);
        assert!(adopt.is_none());
        let _ = adopt_copy.next_f64();
        assert_eq!(adopt_rng.next_f64().to_bits(), adopt_copy.next_f64().to_bits());
    }
}
