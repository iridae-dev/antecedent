//! Continuous eleven-dimensional Bayesian Verma nested-Markov candidate.
//!
//! The prior is a product of Beta kernels on the ORIGINAL Möbius coordinates,
//! restricted to the strictly positive feasible c-factor polytope. Its constant
//! normalizer cancels in the posterior; conditional-width factors are NOT added.
//! The three independent singleton coordinates are updated by exact conjugacy.
//! The eight coupled coordinates use full-feasible-bracket coordinate slice
//! shrinkage (Neal, 2003). No latent binary DAG replaces the eleven-dimensional law.
//! This numerical candidate makes no calibrated public inference claim. Fixed 95%
//! credible endpoints are Monte Carlo quantile estimates; 5%/95% tail ESS does
//! not guarantee the precision of their 2.5%/97.5% endpoints.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::needless_range_loop, reason = "fixed-size eleven-coordinate algebra")]
use antecedent_core::{CausalRng, ExecutionContext, RngFactory, StreamDomain};
use antecedent_kernels::{quantile_type7_sorted, standard_normal};
use antecedent_prob::mcmc_stats::parameter_mcmc_diagnostics;
use serde::{Deserialize, Serialize};

/// Coordinate order: a, c0, c1, q20, q21, q40, q41, g00, g01, g10, g11.
pub const PARAMETERS: usize = 11;
/// Retained coordinates followed by do(X2=0), do(X2=1) means and their contrast.
pub const OUTPUTS: usize = 14;
/// Frozen posterior column identity, independent of machine byte order.
pub const COORDINATES: [&str; OUTPUTS] = [
    "a",
    "c0",
    "c1",
    "q20",
    "q21",
    "q40",
    "q41",
    "g00",
    "g01",
    "g10",
    "g11",
    "mean_do_x2_0",
    "mean_do_x2_1",
    "contrast_1_minus_0",
];
/// Frozen stream generator and non-additive domain/chain derivation.
pub const RNG: &str = "causal_splitmix64_top53_rngfactory_bayesian_domain_chain_index_v1";
/// Maximum retained draws per chain; diagnostics have quadratic lag work.
pub const MAX_DRAWS: usize = 8192;
/// Immutable algorithm identity, including RNG and diagnostic semantics.
pub const METHOD: &str = "verma_beta_kernel_conjugate_coordinate_slice_splitmix64_rank_folded_v1";

/// Typed rejection of the internal Bayesian candidate.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum BayesianRefusal {
    /// Unsupported count/prior/request domain.
    #[error("invalid Bayesian declaration: {0}")]
    Invalid(&'static str),
    /// Cancelled or exhausted work budget; no partial posterior is returned.
    #[error("Bayesian sampling work refused: {0}")]
    Budget(&'static str),
    /// Numerical boundary or nonfinite state; no draw is silently removed.
    #[error("Bayesian numerical failure: {0}")]
    Numerical(&'static str),
    /// Rank/folded split Rhat or bulk/tail ESS gate failed.
    #[error("Bayesian chains failed convergence")]
    Nonconvergence,
}

/// Proper log-concave Beta kernels restricted to positive feasible parameters.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Prior {
    /// Shapes on probability of level zero, finite and at least one.
    pub alpha: [f64; PARAMETERS],
    /// Shapes on its complement, finite and at least one.
    pub beta: [f64; PARAMETERS],
}
impl Default for Prior {
    fn default() -> Self {
        Self { alpha: [1.0; PARAMETERS], beta: [1.0; PARAMETERS] }
    }
}

/// Bounded seeded request. Diagnostic gates cannot be relaxed by this request.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Options {
    /// Four or more independently seeded chains, at most eight.
    pub chains: usize,
    /// Warmup sweeps per chain, at most 65536.
    pub warmup: usize,
    /// Retained sweeps per chain, 256..=8192.
    pub draws: usize,
    /// Total gamma/slice proposals across all chains, at most 100 million.
    pub max_proposals: usize,
    /// Master seed; stream derivation is part of METHOD.
    pub seed: u64,
    /// Fixed 0.95 equal-tailed posterior mass; not coverage calibration.
    pub credible_mass: f64,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            chains: 4,
            warmup: 1024,
            draws: 2048,
            max_proposals: 2_000_000,
            seed: 0,
            credible_mass: 0.95,
        }
    }
}

/// One parameter's modern convergence diagnostics.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Diagnostics {
    /// Rank-normalized split Rhat.
    pub rank_rhat: f64,
    /// Folded rank-normalized split Rhat.
    pub folded_rhat: f64,
    /// Geyer ESS of rank-normalized draws.
    pub bulk_ess: f64,
    /// Minimum ESS of five/ninety-five percent tail indicators.
    pub tail_ess: f64,
}
/// Joint posterior receipt, including estimand dependence and Monte Carlo standing.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Posterior {
    /// Chain-major, draw-major rows of OUTPUTS values.
    pub samples: Vec<[f64; OUTPUTS]>,
    /// Monte Carlo mean, not an exact posterior moment.
    pub mean: [f64; OUTPUTS],
    /// Row-major Monte Carlo covariance, OUTPUTS squared entries.
    pub covariance: Vec<f64>,
    /// Monte Carlo equal-tailed credible quantile estimates.
    pub credible: [[f64; 2]; OUTPUTS],
    /// Diagnostics for every parameter AND derived estimand.
    pub diagnostics: Vec<Diagnostics>,
    /// Actual total proposal count.
    pub proposals: usize,
}

fn q(p: &[f64; PARAMETERS], i: usize, j: usize) -> [f64; 4] {
    let g = p[7 + i * 2 + j];
    [g, p[3 + i] - g, p[5 + j] - g, 1.0 - p[3 + i] - p[5 + j] + g]
}
/// Original normalized joint law; returns None outside strictly positive model.
#[must_use]
pub fn probabilities(p: &[f64; PARAMETERS]) -> Option<[f64; 16]> {
    if p.iter().any(|v| !v.is_finite() || *v <= 0.0 || *v >= 1.0) {
        return None;
    }
    let mut out = [0.0; 16];
    for i in 0..2 {
        for t in 0..2 {
            for j in 0..2 {
                for y in 0..2 {
                    let cell = q(p, i, j)[t * 2 + y];
                    if cell <= 0.0 {
                        return None;
                    }
                    out[i * 8 + t * 4 + j * 2 + y] = if i == 0 { p[0] } else { 1.0 - p[0] }
                        * if j == 0 { p[1 + t] } else { 1.0 - p[1 + t] }
                        * cell;
                }
            }
        }
    }
    if out.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return None;
    }
    Some(out)
}

fn log_block(p: &[f64; PARAMETERS], counts: &[u64; 16], prior: &Prior) -> f64 {
    let mut out = 0.0;
    for k in 3..PARAMETERS {
        if p[k] <= 0.0 || p[k] >= 1.0 {
            return f64::NEG_INFINITY;
        }
        out += (prior.alpha[k] - 1.0) * p[k].ln() + (prior.beta[k] - 1.0) * (-p[k]).ln_1p();
    }
    for i in 0..2 {
        for j in 0..2 {
            let cells = q(p, i, j);
            for t in 0..2 {
                for y in 0..2 {
                    if cells[t * 2 + y] <= 0.0 {
                        return f64::NEG_INFINITY;
                    }
                    out += counts[i * 8 + t * 4 + j * 2 + y] as f64 * cells[t * 2 + y].ln();
                }
            }
        }
    }
    out
}
fn bracket(p: &[f64; PARAMETERS], k: usize) -> (f64, f64) {
    if k < 5 {
        let i = k - 3;
        (
            (p[7 + i * 2]).max(p[8 + i * 2]),
            (1.0 - p[5] + p[7 + i * 2]).min(1.0 - p[6] + p[8 + i * 2]),
        )
    } else if k < 7 {
        let j = k - 5;
        (p[7 + j].max(p[9 + j]), (1.0 - p[3] + p[7 + j]).min(1.0 - p[4] + p[9 + j]))
    } else {
        let (i, j) = ((k - 7) / 2, (k - 7) % 2);
        ((p[3 + i] + p[5 + j] - 1.0).max(0.0), p[3 + i].min(p[5 + j]))
    }
}
struct Work<'a> {
    used: usize,
    limit: usize,
    ctx: &'a ExecutionContext,
}
impl Work<'_> {
    fn spend(&mut self) -> Result<(), BayesianRefusal> {
        if self.ctx.cancellation.is_cancelled() {
            return Err(BayesianRefusal::Budget("cancelled"));
        }
        if self.used >= self.limit {
            return Err(BayesianRefusal::Budget("proposal limit"));
        }
        self.used += 1;
        Ok(())
    }
}
fn uniform(rng: &mut CausalRng) -> f64 {
    rng.next_f64().max(1.0 / 9_007_199_254_740_992.0)
}
// Bounded Marsaglia-Tsang gamma; existing kernel has unbounded inner loops, so it
// cannot supply this candidate's cancellation/proposal contract.
fn gamma(shape: f64, rng: &mut CausalRng, work: &mut Work<'_>) -> Result<f64, BayesianRefusal> {
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        work.spend()?;
        let x = standard_normal(rng);
        let root = 1.0 + c * x;
        if root <= 0.0 {
            continue;
        }
        let v = root * root * root;
        let u = uniform(rng);
        if u < 1.0 - 0.0331 * x.powi(4) || u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return Ok(d * v);
        }
    }
}
fn beta(a: f64, b: f64, rng: &mut CausalRng, work: &mut Work<'_>) -> Result<f64, BayesianRefusal> {
    let x = gamma(a, rng, work)?;
    let y = gamma(b, rng, work)?;
    let v = x / (x + y);
    if v.is_finite() && v > 0.0 && v < 1.0 {
        Ok(v)
    } else {
        Err(BayesianRefusal::Numerical("conjugate Beta boundary"))
    }
}
fn slice(
    p: &mut [f64; PARAMETERS],
    k: usize,
    counts: &[u64; 16],
    prior: &Prior,
    rng: &mut CausalRng,
    work: &mut Work<'_>,
) -> Result<(), BayesianRefusal> {
    let old = p[k];
    let threshold = log_block(p, counts, prior) + uniform(rng).ln();
    if !threshold.is_finite() {
        return Err(BayesianRefusal::Numerical("slice threshold"));
    }
    let (mut low, mut high) = bracket(p, k);
    loop {
        work.spend()?;
        let proposed = low + (high - low) * uniform(rng);
        if proposed <= low || proposed >= high {
            return Err(BayesianRefusal::Numerical("collapsed slice bracket"));
        }
        p[k] = proposed;
        if log_block(p, counts, prior) >= threshold {
            return Ok(());
        }
        if proposed < old {
            low = proposed;
        } else {
            high = proposed;
        }
    }
}
/// Fit the frozen full-dimensional candidate. No partial or failed-chain output.
/// # Errors
/// Invalid declaration, cancellation/budget, numerical boundary or convergence gate.
pub fn fit(
    counts: &[u64; 16],
    prior: &Prior,
    options: &Options,
    ctx: &ExecutionContext,
) -> Result<Posterior, BayesianRefusal> {
    validate(counts, prior, options)?;
    let required = (options.chains * options.draws * OUTPUTS * 8 * 4 + 65536) as u64;
    if ctx
        .memory
        .hard_limit_bytes
        .into_iter()
        .chain(ctx.memory.soft_limit_bytes)
        .any(|limit| required > limit)
    {
        return Err(BayesianRefusal::Budget("posterior/diagnostic memory limit"));
    }
    let mut shapes = [(0.0, 0.0); 3];
    for k in 0..3 {
        shapes[k] = (prior.alpha[k], prior.beta[k]);
    }
    for i in 0..2 {
        for t in 0..2 {
            for j in 0..2 {
                for y in 0..2 {
                    let n = counts[i * 8 + t * 4 + j * 2 + y] as f64;
                    if i == 0 {
                        shapes[0].0 += n;
                    } else {
                        shapes[0].1 += n;
                    }
                    if j == 0 {
                        shapes[1 + t].0 += n;
                    } else {
                        shapes[1 + t].1 += n;
                    }
                }
            }
        }
    }
    let mut work = Work { used: 0, limit: options.max_proposals, ctx };
    let mut samples = Vec::with_capacity(options.chains * options.draws);
    for chain in 0..options.chains {
        let mut rng =
            RngFactory::from_seed(options.seed).stream_for(StreamDomain::Bayesian, chain as u64);
        let mut p = [0.5; PARAMETERS];
        // Dispersed feasible starts, distinct margins and association per chain.
        for k in 3..7 {
            p[k] = 0.2 + 0.6 * uniform(&mut rng);
        }
        for k in 7..11 {
            let (lo, hi) = bracket(&p, k);
            p[k] = lo + (hi - lo) * (0.1 + 0.8 * uniform(&mut rng));
        }
        for sweep in 0..options.warmup + options.draws {
            for k in 0..3 {
                p[k] = beta(shapes[k].0, shapes[k].1, &mut rng, &mut work)?;
            }
            for k in 3..11 {
                slice(&mut p, k, counts, prior, &mut rng, &mut work)?;
            }
            if probabilities(&p).is_none() {
                return Err(BayesianRefusal::Numerical("positive joint law boundary"));
            }
            if sweep >= options.warmup {
                let mut row = [0.0; OUTPUTS];
                row[..PARAMETERS].copy_from_slice(&p);
                for t in 0..2 {
                    row[11 + t] = p[1 + t] * (1.0 - p[5]) + (1.0 - p[1 + t]) * (1.0 - p[6]);
                }
                row[13] = row[12] - row[11];
                samples.push(row);
            }
        }
    }
    summarize(samples, options, work.used, ctx)
}
fn validate(counts: &[u64; 16], prior: &Prior, o: &Options) -> Result<(), BayesianRefusal> {
    if counts.iter().any(|n| *n == 0 || *n > 1_000_000_000) {
        return Err(BayesianRefusal::Invalid(
            "positive integer counts at most one billion per cell",
        ));
    }
    if prior.alpha.iter().chain(&prior.beta).any(|v| !v.is_finite() || *v < 1.0 || *v > 1_000_000.0)
    {
        return Err(BayesianRefusal::Invalid("Beta shapes in [1,1000000]"));
    }
    if !(4..=8).contains(&o.chains)
        || !(256..=MAX_DRAWS).contains(&o.draws)
        || o.warmup < 128
        || o.warmup > 65536
        || o.max_proposals == 0
        || o.max_proposals > 100_000_000
        || o.credible_mass.to_bits() != 0.95_f64.to_bits()
    {
        return Err(BayesianRefusal::Invalid(
            "bounded sampler request and fixed 95% credible mass",
        ));
    }
    Ok(())
}
fn summarize(
    samples: Vec<[f64; OUTPUTS]>,
    o: &Options,
    proposals: usize,
    ctx: &ExecutionContext,
) -> Result<Posterior, BayesianRefusal> {
    let mut diagnostics = Vec::with_capacity(OUTPUTS);
    for k in 0..OUTPUTS {
        if ctx.cancellation.is_cancelled() {
            return Err(BayesianRefusal::Budget("cancelled during diagnostics"));
        }
        let column: Vec<_> = samples.iter().map(|row| row[k]).collect();
        let d = parameter_mcmc_diagnostics(&column, o.chains, o.draws, 1)[0];
        diagnostics.push(Diagnostics {
            rank_rhat: d.rhat_rank,
            folded_rhat: d.rhat_folded,
            bulk_ess: d.ess_bulk,
            tail_ess: d.ess_tail,
        });
    }
    if diagnostics.iter().any(|d| {
        !d.rank_rhat.is_finite()
            || !d.folded_rhat.is_finite()
            || d.rank_rhat.max(d.folded_rhat) > 1.01
            || !d.bulk_ess.is_finite()
            || !d.tail_ess.is_finite()
            || d.bulk_ess < 400.0
            || d.tail_ess < 400.0
    }) {
        return Err(BayesianRefusal::Nonconvergence);
    }
    let n = samples.len() as f64;
    let mut mean = [0.0; OUTPUTS];
    for row in &samples {
        for k in 0..OUTPUTS {
            mean[k] += row[k] / n;
        }
    }
    let mut covariance = vec![0.0; OUTPUTS * OUTPUTS];
    for row in &samples {
        for i in 0..OUTPUTS {
            for j in 0..OUTPUTS {
                covariance[i * OUTPUTS + j] += (row[i] - mean[i]) * (row[j] - mean[j]) / (n - 1.0);
            }
        }
    }
    let mut credible = [[0.0; 2]; OUTPUTS];
    let tail = (1.0 - o.credible_mass) / 2.0;
    for k in 0..OUTPUTS {
        let mut col: Vec<_> = samples.iter().map(|r| r[k]).collect();
        col.sort_by(f64::total_cmp);
        credible[k] = [quantile_type7_sorted(&col, tail), quantile_type7_sorted(&col, 1.0 - tail)];
    }
    if ctx.cancellation.is_cancelled() {
        return Err(BayesianRefusal::Budget("cancelled during summary"));
    }
    Ok(Posterior { samples, mean, covariance, credible, diagnostics, proposals })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn model_is_original_eleven_dimensional_law() {
        let p = [0.3, 0.2, 0.8, 0.4, 0.6, 0.3, 0.7, 0.15, 0.25, 0.2, 0.4];
        let cells = probabilities(&p).unwrap();
        assert!((cells.iter().sum::<f64>() - 1.0).abs() < 1e-14);
        assert!(probabilities(&[0.5; 11]).is_none());
        assert!((cells[0] - 0.3 * 0.2 * 0.15).abs() < 1e-14);
    }
    #[test]
    fn domain_budget_cancellation_and_nonconvergence_refuse() {
        let ctx = ExecutionContext::for_tests(0);
        let prior = Prior::default();
        for shape in [0.5, f64::NAN, f64::INFINITY, 1_000_001.0] {
            let mut malformed = prior.clone();
            malformed.alpha[7] = shape;
            assert!(matches!(
                fit(&[1; 16], &malformed, &Options::default(), &ctx),
                Err(BayesianRefusal::Invalid(_))
            ));
        }
        let mut small_memory = ExecutionContext::for_tests(0);
        small_memory.memory.hard_limit_bytes = Some(0);
        assert!(matches!(
            fit(&[1; 16], &prior, &Options::default(), &small_memory),
            Err(BayesianRefusal::Budget(_))
        ));
        assert!(matches!(
            fit(&[0; 16], &prior, &Options::default(), &ctx),
            Err(BayesianRefusal::Invalid(_))
        ));
        let mut options = Options { credible_mass: 0.90, ..Options::default() };
        assert!(matches!(fit(&[1; 16], &prior, &options, &ctx), Err(BayesianRefusal::Invalid(_))));
        options = Options { max_proposals: 1, ..Options::default() };
        assert!(matches!(fit(&[1; 16], &prior, &options, &ctx), Err(BayesianRefusal::Budget(_))));
        options.max_proposals = 1_000_000;
        options.draws = 256;
        options.warmup = 128;
        assert!(matches!(
            fit(&[1; 16], &prior, &options, &ctx),
            Err(BayesianRefusal::Nonconvergence)
        ));
        ctx.cancellation.cancel();
        assert!(matches!(
            fit(&[1; 16], &prior, &Options::default(), &ctx),
            Err(BayesianRefusal::Budget(_))
        ));
    }
    #[test]
    fn low_count_candidate_has_joint_posterior_and_exact_conjugate_moments() {
        let options = Options { seed: 817, ..Options::default() };
        let p =
            fit(&[1; 16], &Prior::default(), &options, &ExecutionContext::for_tests(0)).unwrap();
        for k in 0..3 {
            assert!((p.mean[k] - 0.5).abs() < 0.012, "{:?}", p.mean);
        }
        assert!((p.covariance[0] - 1.0 / 76.0).abs() < 0.0015);
        assert!((p.covariance[15] - 1.0 / 44.0).abs() < 0.002);
        assert!((p.mean[13]).abs() < 0.01);
        assert_eq!(p.samples.len(), options.chains * options.draws);
        assert_eq!(p.diagnostics.len(), OUTPUTS);
    }
}
