//! Shared graph-structure MCMC schedule, parallel driver, and posterior assembly.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::sync::Arc;

use antecedent_prob::{InferenceDiagnostics, kish_ess};

use crate::error::DiscoveryError;
use crate::graph_posterior::{
    GraphPosterior, MAX_EDGE_MCSE, accumulate_marginals, graph_chain_summary,
    mcmc_graph_diagnostics, publish_graph_posterior,
};

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
pub(crate) fn merge_chunk_outputs(
    n_chains: usize,
    n_draws: usize,
    n_params: usize,
    chunks: impl IntoIterator<Item = (usize, Vec<f64>, Vec<Vec<u64>>, u64)>,
) -> (Vec<f64>, Vec<Vec<u64>>, u64) {
    let mut traces = vec![0.0f64; n_chains * n_draws * n_params];
    let mut sample_masks: Vec<Vec<u64>> = vec![Vec::new(); n_chains];
    let mut rejected = 0u64;
    for (start, local_traces, local_samples, local_rej) in chunks {
        rejected += local_rej;
        let n_local = local_samples.len();
        for li in 0..n_local {
            let chain = start + li;
            sample_masks[chain].clone_from(&local_samples[li]);
            for d in 0..n_draws {
                for p in 0..n_params {
                    traces[(chain * n_draws + d) * n_params + p] =
                        local_traces[(li * n_draws + d) * n_params + p];
                }
            }
        }
    }
    (traces, sample_masks, rejected)
}

/// Run chunked parallel chain workers and merge their outputs.
pub(crate) fn run_parallel_mask_chains<F>(
    n_chains: usize,
    n_draws: usize,
    n_params: usize,
    max_threads: usize,
    worker: F,
) -> Result<(Vec<f64>, Vec<Vec<u64>>, u64), DiscoveryError>
where
    F: Fn(usize, usize) -> (usize, Vec<f64>, Vec<Vec<u64>>, u64) + Send + Sync,
{
    let threads = max_threads.max(1);
    let chunk = n_chains.div_ceil(threads).max(1);
    let mut outputs = Vec::new();
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for start in (0..n_chains).step_by(chunk) {
            let end = (start + chunk).min(n_chains);
            let w = &worker;
            handles.push(scope.spawn(move || w(start, end)));
        }
        for h in handles {
            outputs.push(h.join().map_err(|_| DiscoveryError::Unsupported {
                message: "graph MCMC worker panicked",
            }));
        }
    });
    let outputs = outputs.into_iter().collect::<Result<Vec<_>, _>>()?;
    Ok(merge_chunk_outputs(n_chains, n_draws, n_params, outputs))
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
    let diagnostics = mcmc_graph_diagnostics(
        schedule.n_chains,
        schedule.n_warmup,
        schedule.n_draws,
        summary.ess_bulk_min,
        summary.ess_tail_min,
        summary.rhat_max,
        0,
        true,
        summary.all_chains_moved,
    );
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
    GraphPosterior::new(n, weights, masks, edge, orient, ess, diagnostics, rejected)
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
        let (traces, samples, _) = run_parallel_mask_chains(5, 4, 1, 2, |start, end| {
            workers.fetch_add(1, Ordering::Relaxed);
            (start, vec![0.0; (end - start) * 4], vec![vec![0; 4]; end - start], 0)
        })
        .unwrap();
        assert_eq!(workers.load(Ordering::Relaxed), 2);
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
}
