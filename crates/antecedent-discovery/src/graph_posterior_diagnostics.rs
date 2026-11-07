// SPDX-License-Identifier: MIT OR Apache-2.0
//! Publication diagnostics for graph posteriors.

use super::{
    Arc, DiscoveryError, HessianFactorization, InferenceDiagnostics, all_chains_moved,
    parameter_mcmc_diagnostics,
};

/// Analytic diagnostics for exact / closed-form graph posteriors.
#[must_use]
pub(crate) fn analytic_graph_diagnostics(n_graphs: usize, ess: f64) -> InferenceDiagnostics {
    InferenceDiagnostics {
        converged: true,
        iterations: crate::indexing::dense_u32(n_graphs),
        grad_inf_norm: 0.0,
        hessian_condition: 1.0,
        factorization: HessianFactorization::Analytic,
        separation_warning: false,
        notes: vec![Arc::from(format!(
            "exact_enumeration_of_bic_weights effective_graph_count={ess:.4}"
        ))],
        backend_id: Arc::from("graph_posterior_analytic"),
        n_chains: None,
        n_warmup: None,
        // Exact enumeration has no sampling ESS; the Kish graph count is a concentration index
        // (see the note above), not a chain diagnostic.
        ess_bulk_min: None,
        ess_tail_min: None,
        rhat_max: None,
        n_divergences: None,
        mean_accept_prob: None,
        n_warmup_divergences: None,
        n_postwarmup_divergences: None,
        max_abs_delta_h: None,
        all_chains_moved: None,
    }
}

/// Summary of a graph-structure MCMC run (edge-indicator traces) fed to
/// [`mcmc_graph_diagnostics`].
#[derive(Debug, Clone, Copy)]
pub struct GraphMcmcRun {
    /// Chains run.
    pub n_chains: u32,
    /// Warmup iterations per chain.
    pub n_warmup: u32,
    /// Retained draws per chain.
    pub n_draws: u32,
    /// Smallest bulk effective sample size over the indicators.
    pub ess_bulk_min: f64,
    /// Smallest tail effective sample size over the indicators.
    pub ess_tail_min: f64,
    /// Largest split R-hat over the indicators.
    pub rhat_max: f64,
    /// Divergent transitions.
    pub n_divergences: u32,
    /// Whether the run passed its convergence checks.
    pub converged: bool,
    /// Whether every chain made at least one structural move.
    pub chains_moved: bool,
}

/// MCMC diagnostics gate for graph chains (edge-indicator traces).
#[must_use]
pub fn mcmc_graph_diagnostics(run: &GraphMcmcRun) -> InferenceDiagnostics {
    let GraphMcmcRun {
        n_chains,
        n_warmup,
        n_draws,
        ess_bulk_min,
        ess_tail_min,
        rhat_max,
        n_divergences,
        converged,
        chains_moved,
    } = *run;
    InferenceDiagnostics {
        converged,
        iterations: n_draws,
        grad_inf_norm: 0.0,
        hessian_condition: f64::NAN,
        factorization: HessianFactorization::Mcmc,
        separation_warning: false,
        notes: Vec::new(),
        backend_id: Arc::from("graph_structure_mcmc"),
        n_chains: Some(n_chains),
        n_warmup: Some(n_warmup),
        ess_bulk_min: Some(ess_bulk_min),
        ess_tail_min: Some(ess_tail_min),
        rhat_max: Some(rhat_max),
        n_divergences: Some(n_divergences),
        // Not fabricated: this function has no accept/reject counts to derive a real mean
        // Metropolis acceptance probability from. The `rejected` counter threaded through
        // `FinishMaskPosterior`/`GraphPosterior.rejected_invalid` (see `structure_mcmc.rs`,
        // `order_mcmc.rs`) counts constraint-invalid proposals (e.g. would-create-a-cycle),
        // not the sampler's actual per-step Metropolis accept/reject decision -- and it
        // isn't even threaded into this function's signature. `allows_graph_posterior`
        // below does not consult `mean_accept_prob`, so this is safe for the publication
        // gate; callers reading `diagnostics.mean_accept_prob` for reporting get an honest
        // "unknown" instead of a fabricated constant.
        mean_accept_prob: None,
        n_warmup_divergences: Some(0),
        n_postwarmup_divergences: Some(n_divergences),
        max_abs_delta_h: Some(0.0),
        all_chains_moved: Some(chains_moved),
    }
}

/// Largest Monte-Carlo standard error any edge marginal may carry for a graph-MCMC posterior
/// to publish: `0.05` is the standard error of a probability-½ indicator at ESS 100, the
/// library's per-chain-scale bar. It binds where it matters (the marginals users read) rather
/// than on the ESS of near-constant indicators, whose error is small at any ESS.
pub(crate) const MAX_EDGE_MCSE: f64 = 0.05;

/// Whether graph-MCMC diagnostics are sufficient to publish a posterior.
///
/// Binary edge-indicator traces mix slower than continuous HMC parameters, so
/// the R-hat bar is `1.2` (vs `1.01` on [`InferenceDiagnostics::allows_posterior`]).
/// Divergences must be zero (not merely reported). Marginal accuracy is enforced separately
/// on the per-edge MCSE ([`MAX_EDGE_MCSE`]) by the graph-MCMC engines.
#[must_use]
pub(crate) fn allows_graph_posterior(diagnostics: &InferenceDiagnostics) -> bool {
    if diagnostics.factorization != HessianFactorization::Mcmc {
        return diagnostics.allows_posterior();
    }
    let rhat_ok = diagnostics.rhat_max.is_some_and(|r| r.is_finite() && r < 1.2);
    let ess_ok = diagnostics.ess_bulk_min.is_some_and(|e| e.is_finite() && e > 10.0);
    let div_ok =
        diagnostics.n_postwarmup_divergences.or(diagnostics.n_divergences).is_some_and(|n| n == 0);
    // A chain that never accepted a structural move has bit-identical edge traces, which
    // the chain diagnostics report as R̂ = 1.0 and ESS = `n_chains·n_draws` — perfect
    // scores that mean "nothing varied", not "everything converged". `all_chains_moved` is
    // the only signal that separates the two, so it has to be in the conjunction (as it is
    // in `InferenceDiagnostics::mcmc_publication_ok`).
    let moved_ok = diagnostics.all_chains_moved == Some(true);
    diagnostics.converged && rhat_ok && ess_ok && div_ok && moved_ok
}

/// Set `converged` from [`allows_graph_posterior`]; optionally refuse publication.
///
/// # Errors
///
/// When `require_gate` is true and the graph-MCMC diagnostics bar fails.
pub(crate) fn publish_graph_posterior(
    mut diagnostics: InferenceDiagnostics,
    require_gate: bool,
    refuse_msg: &'static str,
) -> Result<InferenceDiagnostics, DiscoveryError> {
    diagnostics.converged = true;
    let ok = allows_graph_posterior(&diagnostics);
    diagnostics.converged = ok;
    if require_gate && !ok {
        return Err(DiscoveryError::unsupported(refuse_msg));
    }
    Ok(diagnostics)
}

/// Edge-indicator chain diagnostics.
#[derive(Clone, Debug)]
pub(crate) struct GraphChainSummary {
    /// Maximum split R̂ over varying indicators. `∞` when any indicator's chains disagree
    /// without varying within a chain (the clearest non-convergence there is); never
    /// replaced by a finite value from another indicator.
    pub rhat_max: f64,
    /// Minimum bulk ESS over varying indicators.
    pub ess_bulk_min: f64,
    /// Minimum tail ESS over varying indicators.
    pub ess_tail_min: f64,
    /// Whether every chain moved on at least one indicator.
    pub all_chains_moved: bool,
    /// Monte-Carlo standard error of each input indicator's mean (`sqrt(p(1-p)/ESS_bulk)`),
    /// `0` for an indicator constant over every draw of every chain.
    pub mcse: Vec<f64>,
    /// Split R̂ of each input indicator; `1` for one constant over every draw of every chain.
    pub rhat: Vec<f64>,
    /// Bulk ESS of each input indicator; the total draw count for a constant one.
    pub ess_bulk: Vec<f64>,
}

/// Edge-indicator chain diagnostics (R-hat / ESS / MCSE), dropping constant parameters.
///
/// Zero-variance indicators (never/always present) would otherwise inflate R-hat
/// to infinity and fail the diagnostics gate spuriously. Indicators that are constant
/// within every chain but differ between chains are *not* dropped: their R̂ is `∞`.
#[must_use]
pub(crate) fn graph_chain_summary(
    traces: &[f64],
    n_chains: usize,
    n_draws: usize,
    n_params: usize,
) -> GraphChainSummary {
    if n_params == 0 || n_chains == 0 || n_draws == 0 {
        return GraphChainSummary {
            rhat_max: f64::INFINITY,
            ess_bulk_min: 0.0,
            ess_tail_min: 0.0,
            all_chains_moved: false,
            mcse: vec![0.0; n_params],
            rhat: vec![f64::INFINITY; n_params],
            ess_bulk: vec![0.0; n_params],
        };
    }
    let mut keep = Vec::new();
    for p in 0..n_params {
        let mut min_v = f64::INFINITY;
        let mut max_v = f64::NEG_INFINITY;
        for c in 0..n_chains {
            for d in 0..n_draws {
                let v = traces[(c * n_draws + d) * n_params + p];
                min_v = min_v.min(v);
                max_v = max_v.max(v);
            }
        }
        if max_v - min_v > 1e-12 {
            keep.push(p);
        }
    }
    if keep.is_empty() {
        // All indicators constant across chains — treat as perfect agreement.
        let n_total = (n_chains * n_draws) as f64;
        return GraphChainSummary {
            rhat_max: 1.0,
            ess_bulk_min: n_total,
            ess_tail_min: n_total,
            all_chains_moved: all_chains_moved(traces, n_chains, n_draws, n_params),
            mcse: vec![0.0; n_params],
            rhat: vec![1.0; n_params],
            ess_bulk: vec![n_total; n_params],
        };
    }
    let k = keep.len();
    let mut filtered = vec![0.0; n_chains * n_draws * k];
    for c in 0..n_chains {
        for d in 0..n_draws {
            for (j, &p) in keep.iter().enumerate() {
                filtered[(c * n_draws + d) * k + j] = traces[(c * n_draws + d) * n_params + p];
            }
        }
    }
    // Same guards as `mcmc_summary`: too few chains / draws cannot support the statistic.
    let rhat_guard = n_chains < 2 || n_draws < 8;
    let ess_guard = n_draws < 8;
    let diags = parameter_mcmc_diagnostics(&filtered, n_chains, n_draws, k);
    let mut rhat_max = 0.0_f64;
    let mut ess_bulk_min = f64::INFINITY;
    let mut ess_tail_min = f64::INFINITY;
    let mut mcse = vec![0.0; n_params];
    let n_total = (n_chains * n_draws) as f64;
    let mut rhat = vec![1.0; n_params];
    let mut ess_bulk = vec![n_total; n_params];
    for (j, d) in diags.iter().enumerate() {
        rhat[keep[j]] = if !rhat_guard && d.rhat.is_finite() { d.rhat } else { f64::INFINITY };
        ess_bulk[keep[j]] = if !ess_guard && d.ess_bulk.is_finite() { d.ess_bulk } else { 0.0 };
        rhat_max = if d.rhat.is_finite() { rhat_max.max(d.rhat) } else { f64::INFINITY };
        ess_bulk_min = ess_bulk_min.min(d.ess_bulk);
        ess_tail_min = ess_tail_min.min(d.ess_tail);
        let mean: f64 = (0..n_chains * n_draws).map(|i| filtered[i * k + j]).sum::<f64>() / n_total;
        mcse[keep[j]] = if !ess_guard && d.ess_bulk.is_finite() && d.ess_bulk > 0.0 {
            (mean * (1.0 - mean) / d.ess_bulk).max(0.0).sqrt()
        } else {
            f64::INFINITY
        };
    }
    GraphChainSummary {
        rhat_max: if rhat_guard { f64::INFINITY } else { rhat_max },
        ess_bulk_min: if ess_guard || !ess_bulk_min.is_finite() { 0.0 } else { ess_bulk_min },
        ess_tail_min: if ess_guard || !ess_tail_min.is_finite() { 0.0 } else { ess_tail_min },
        all_chains_moved: all_chains_moved(&filtered, n_chains, n_draws, k),
        mcse,
        rhat,
        ess_bulk,
    }
}

/// Fewest value changes an edge indicator must make in every chain for that chain's mean to be
/// compared with the others. A two-state series with `t` changes holds about `t / 2` independent
/// excursions into its rarer state, so its mean has no usable standard error below a handful.
pub(crate) const MIN_INDICATOR_TRANSITIONS: usize = 6;

/// An indicator whose rarer value occurs fewer than this many times over all draws of all chains
/// is judged by the pooled count alone: its marginal is within `MIN_RARE_EVENTS / draws` of 0 or 1
/// whatever the chains do, so a chain that rarely changes it cannot move the answer.
const MIN_RARE_EVENTS: usize = 4 * MIN_INDICATOR_TRANSITIONS;

/// Least standardised distance between one chain's marginal of an edge and the pooled marginal
/// that refuses a run. A run holds a few hundred comparisons (indicators times chains) and the
/// batch-means error behind each rests on [`AGREEMENT_BATCHES`] batches, so its ratio has a
/// Student-t tail; the limit actually applied is the larger of this floor and the Student-t
/// quantile that keeps the false refusal rate of a correct run at [`FALSE_REFUSAL_RATE`] over
/// all the comparisons the run makes (it grows with the number of indicators).
pub(crate) const MAX_CHAIN_DEVIATION_Z: f64 = 6.0;

/// Probability that a correct run is refused by the chain-deviation criterion, over all its
/// indicator-by-chain comparisons.
pub(super) const FALSE_REFUSAL_RATE: f64 = 1e-3;

/// Largest rank-normalised split R̂ of the log target along the chains. The log target is a
/// continuous summary of the whole graph, so it exposes chains sitting in different regions that
/// every individual edge indicator would hide behind a rarely flipping value.
pub(crate) const MAX_SCORE_RHAT: f64 = 1.1;

/// Batches the draws of one chain are cut into to estimate the standard error of its marginal.
pub(super) const AGREEMENT_BATCHES: usize = 20;

/// Whether the chains of a graph sampler describe the same distribution, judged on what the
/// indicator R̂ and ESS cannot see.
///
/// R̂ of an indicator that a chain never changes has no within-chain variance, and the Geyer ESS
/// of an indicator that changes a few times is far too large. This asks the questions directly.
/// Per indicator: how far is each chain's marginal from the pooled marginal in units of that
/// chain's standard error (the batch-means error, which cannot be trusted when batches hold
/// few changes of the indicator; there the excursion bound `sqrt(p (1 - p) / (changes / 2))`
/// replaces it), and does every chain change the indicator often enough to
/// be judged at all. Overall: do the chains agree on the log target.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ChainAgreement {
    /// Largest standardised distance of a chain's marginal from the pooled one.
    pub max_deviation_z: f64,
    /// Standardised distance above which a run is refused: the floor
    /// [`MAX_CHAIN_DEVIATION_Z`], or the Student-t quantile holding the false refusal rate at
    /// [`FALSE_REFUSAL_RATE`] over the comparisons made, whichever is larger.
    pub deviation_limit: f64,
    /// Trace parameter attaining [`Self::max_deviation_z`].
    pub worst_param: Option<usize>,
    /// Indicators some chain changes fewer than [`MIN_INDICATOR_TRANSITIONS`] times although
    /// their pooled marginal is not within the rare-event bound of 0 or 1.
    pub unresolved: usize,
    /// Rank-normalised split R̂ of the log target (`∞` when the chains cannot be compared).
    pub score_rhat: f64,
}

impl ChainAgreement {
    /// Chains agree on every judged edge marginal.
    #[must_use]
    pub(crate) fn deviation_ok(&self) -> bool {
        self.max_deviation_z.is_finite() && self.max_deviation_z <= self.deviation_limit
    }

    /// Every indicator is changed often enough in every chain to be judged. Reported, not
    /// gated: an edge the data nearly fix flips rarely in a converged chain, so a transition
    /// count cannot tell a stuck chain from a decided edge; [`Self::deviation_ok`] judges
    /// agreement with the excursion error of such an indicator instead.
    #[must_use]
    pub(crate) fn resolved(&self) -> bool {
        self.unresolved == 0
    }

    /// The chains agree on the log target.
    #[must_use]
    pub(crate) fn score_ok(&self) -> bool {
        self.score_rhat.is_finite() && self.score_rhat <= MAX_SCORE_RHAT
    }

    /// The gated criteria hold: chain agreement on every edge marginal and on the log target.
    #[must_use]
    pub(crate) fn agrees(&self) -> bool {
        self.deviation_ok() && self.score_ok()
    }
}

/// [`ChainAgreement`] of `traces` (edge indicators) and `scores` (the log target at each draw),
/// both laid out chain-major as in [`graph_chain_summary`] (`scores` has one value per draw).
#[must_use]
pub(crate) fn graph_chain_agreement(
    traces: &[f64],
    scores: &[f64],
    n_chains: usize,
    n_draws: usize,
    n_params: usize,
) -> ChainAgreement {
    let unjudgeable = ChainAgreement {
        max_deviation_z: f64::INFINITY,
        deviation_limit: MAX_CHAIN_DEVIATION_Z,
        worst_param: None,
        unresolved: n_params,
        score_rhat: f64::INFINITY,
    };
    if n_chains < 2
        || n_draws < 8
        || traces.len() != n_chains * n_draws * n_params
        || scores.len() != n_chains * n_draws
    {
        return unjudgeable;
    }
    let score_rhat = score_rank_rhat(scores, n_chains, n_draws);
    let total = (n_chains * n_draws) as f64;
    let batch_len = (n_draws / AGREEMENT_BATCHES).max(1);
    let n_batches = n_draws / batch_len;
    let mut max_deviation_z = 0.0_f64;
    let mut comparisons = 0usize;
    let mut worst_param = None;
    let mut unresolved = 0usize;
    let mut means = vec![0.0; n_chains];
    let mut changes = vec![0usize; n_chains];
    let mut errors = vec![0.0; n_chains];
    for p in 0..n_params {
        for c in 0..n_chains {
            let at = |d: usize| traces[(c * n_draws + d) * n_params + p];
            let mut sum = 0.0;
            let mut flips = 0usize;
            let mut batch_means = vec![0.0; n_batches];
            for d in 0..n_draws {
                let v = at(d);
                sum += v;
                if d > 0 && v.to_bits() != at(d - 1).to_bits() {
                    flips += 1;
                }
                if d / batch_len < n_batches {
                    batch_means[d / batch_len] += v / batch_len as f64;
                }
            }
            means[c] = sum / n_draws as f64;
            changes[c] = flips;
            errors[c] = if n_batches >= 2 {
                let grand = batch_means.iter().sum::<f64>() / n_batches as f64;
                let var = batch_means.iter().map(|m| (m - grand).powi(2)).sum::<f64>()
                    / (n_batches as f64 - 1.0);
                (var / n_batches as f64).sqrt()
            } else {
                0.0
            };
        }
        let pooled = means.iter().sum::<f64>() / n_chains as f64;
        let minority = (pooled.min(1.0 - pooled) * total).round();
        if minority < 1.0 {
            continue;
        }
        let variance = pooled * (1.0 - pooled);
        for c in 0..n_chains {
            // Batches that each hold several changes carry the chain's autocorrelation; with
            // fewer the batch means are too coarse, and the independent units of a two-state
            // series are its excursions into the rarer state, about half its changes.
            let units = if changes[c] >= 5 * n_batches {
                n_draws as f64
            } else {
                (changes[c] as f64 / 2.0).max(1.0)
            };
            let error = errors[c].max((variance / units).sqrt());
            comparisons += 1;
            let z = (means[c] - pooled).abs() / error;
            if z > max_deviation_z {
                max_deviation_z = z;
                worst_param = Some(p);
            }
        }
        let thin_chain = changes.iter().any(|&t| t < MIN_INDICATOR_TRANSITIONS);
        if thin_chain && minority >= MIN_RARE_EVENTS as f64 {
            unresolved += 1;
        }
    }
    let deviation_limit = deviation_limit(comparisons, n_batches);
    ChainAgreement { max_deviation_z, deviation_limit, worst_param, unresolved, score_rhat }
}

/// The chain-deviation limit for `comparisons` indicator-by-chain comparisons whose errors rest
/// on `n_batches` batches: the Student-t quantile (batches less one degrees of freedom) with a
/// two-sided tail of [`FALSE_REFUSAL_RATE`] split over the comparisons, never below
/// [`MAX_CHAIN_DEVIATION_Z`].
pub(super) fn deviation_limit(comparisons: usize, n_batches: usize) -> f64 {
    if comparisons == 0 || n_batches < 2 {
        return MAX_CHAIN_DEVIATION_Z;
    }
    let tail = FALSE_REFUSAL_RATE / (2.0 * comparisons as f64);
    let quantile = antecedent_stats::special::student_t_ppf(1.0 - tail, n_batches as f64 - 1.0);
    if quantile.is_finite() { quantile.max(MAX_CHAIN_DEVIATION_Z) } else { MAX_CHAIN_DEVIATION_Z }
}

/// Rank-normalised split R̂ (Vehtari et al., 2021) of one scalar trace; `1` when the trace is
/// constant over every draw of every chain, `∞` when it cannot be computed.
fn score_rank_rhat(scores: &[f64], n_chains: usize, n_draws: usize) -> f64 {
    let (lo, hi) = scores
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    if !lo.is_finite() || !hi.is_finite() {
        return f64::INFINITY;
    }
    if hi - lo <= 1e-12 {
        return 1.0;
    }
    let d = parameter_mcmc_diagnostics(scores, n_chains, n_draws, 1)[0];
    if d.rhat.is_finite() { d.rhat } else { f64::INFINITY }
}
