//! The exact posterior of the arm success probabilities of a participant-level
//! trial, with the intra-participant correlation either declared or estimated
//! jointly.
//!
//! # The model
//!
//! A participant of arm `k` gives `m` binary responses. Their success count
//! `s` is beta-binomial with mean `p_k` and intra-participant correlation
//! `rho`: `s | p_k, rho ~ BetaBinomial(m, p_k (1 - rho) / rho,
//! (1 - p_k) (1 - rho) / rho)`, the binomial at `rho = 0`. Written with
//! `d = rho / (1 - rho)`,
//!
//! ```text
//! P(s | p, rho) = C(m, s) prod_{j < s}(p + j d) prod_{j < m - s}(1 - p + j d)
//!                 / prod_{j < m}(1 + j d).
//! ```
//!
//! The arm means have independent `Beta(a0_k, b0_k)` priors (the resolved
//! priors, including any discounted historical borrowing). The likelihood of
//! an arm depends on its participants only through the histogram
//! `n_k[s]` of success counts, so the posterior is a function of the
//! histograms and the prior.
//!
//! # Two modes for the correlation
//!
//! * **Declared**: `rho` is a stated value. The posterior of the arm means is
//!   the product of the arms' exact posteriors `p_k | data ~ prior_k(p) L_k(p,
//!   rho)`.
//! * **Estimated**: `rho` has a `Beta(a_rho, b_rho)` prior (shapes at least
//!   one), shared by all arms, and is integrated *jointly* with the arm means.
//!   The posterior is a mixture over `rho` of products of the arms' posteriors,
//!   with mixing weight `pi(rho) prod_k Z_k(rho)` where `Z_k(rho) = integral
//!   prior_k(p) L_k(p, rho) dp` is the arm's marginal likelihood at `rho`.
//!
//! # The computation and its error
//!
//! Everything is a two-dimensional integral over `(p, rho)` evaluated by
//! composite Gauss-Legendre quadrature, with the log-sum-exp shift that keeps
//! thousands of participants from underflowing.
//!
//! * The arm means are integrated in `u = logit p`, where the posterior
//!   density `prior(p) L(p, rho) p (1 - p)` is smooth and tends to zero at both
//!   ends. A window is placed from the moments of the Beta shapes the design
//!   effect gives (a locator only); panels are at most 1.5 standard deviations
//!   wide of the narrowest posterior active over them, with 8 nodes each. The
//!   density at the window edges is then *checked* against its peak
//!   (relative size below `1e-13`); a window that fails is widened and the
//!   computation repeated, so the truncation is bounded by the check rather
//!   than assumed.
//! * Probabilities of being best, the expected maximum, and the exceedance of a
//!   margin are integrals of the arms' densities times the cumulative
//!   distribution functions of the others. The cumulative functions are the
//!   exact integrals of the degree-7 panel interpolants (the integration matrix
//!   at the nodes, Legendre coefficients elsewhere), so no integral is
//!   approximated by a coarser rule than the density quadrature.
//! * The correlation is integrated over a window placed from the posterior of
//!   the correlation itself. A first pass over the whole interval locates the
//!   mass, the window is narrowed to nine posterior standard deviations around
//!   the mean, and passes repeat until the panels are at most 3.5 posterior
//!   standard deviations wide and the density at the window edges (other than
//!   at 0 and 1) is below `1e-13` of its peak.
//! * [`ExactPosterior::probability_error`] reports `|sum_k P(k best) - 1|`,
//!   which is zero for exact integration and so reads the quadrature error of
//!   the whole computation.
//!
//! The functionals are exact posterior quantities under the stated model and
//! priors; no coverage or error-rate claim follows from them.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::OnceLock;

use antecedent_kernels::sigmoid;

use super::AdaptiveInferenceError as Error;
use crate::special::{digamma, gauss_legendre, trigamma};

const NODES: usize = 8;
/// The logit window never extends beyond this magnitude; the prior shapes are
/// at least one, so the density there is below `e^-40` of its peak.
const U_LIMIT: f64 = 40.0;
const MAX_PANELS: usize = 4096;
const WINDOW_SIGMAS: f64 = 12.0;
const TAIL_DECAY: f64 = 36.0;
const PANEL_SIGMAS: f64 = 1.5;
/// Relative log size below which a density at a window edge is negligible.
const EDGE_LOG_TOLERANCE: f64 = -30.0;
const MAX_WIDENINGS: usize = 6;
const RHO_WINDOW_SIGMAS: f64 = 9.0;
const RHO_PANELS_FIRST: usize = 4;
const RHO_PANELS_REFINED: usize = 6;
const RHO_PANELS_MAX: usize = 40;
const RHO_PANEL_SIGMAS: f64 = 3.5;
const MAX_RHO_PASSES: usize = 10;
/// Posterior weight (relative to the largest) of a correlation node below
/// which the node is dropped from the functionals.
const NEGLIGIBLE_WEIGHT_LOG: f64 = -40.0;
const CANCEL_STRIDE: usize = 4;

/// Most binary responses per participant.
pub const MAX_RESPONSES: u32 = 32;
/// Most arms.
pub const MAX_ARMS: usize = 64;

/// A `Beta(alpha, beta)` prior on the intra-participant correlation, shapes at
/// least one so that the density is bounded.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CorrelationPrior {
    /// First shape, at least one.
    pub alpha: f64,
    /// Second shape, at least one.
    pub beta: f64,
}

impl CorrelationPrior {
    /// Whether the shapes are finite, at least one, and at most `1e4`.
    #[must_use]
    pub fn is_valid(self) -> bool {
        [self.alpha, self.beta].iter().all(|&s| s.is_finite() && (1.0..=1.0e4).contains(&s))
    }

    /// The prior mean `alpha / (alpha + beta)`.
    #[must_use]
    pub fn mean(self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }
}

/// How the intra-participant correlation enters the posterior.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CorrelationModel {
    /// A stated value in `[0, 1)`.
    Declared(f64),
    /// A prior, integrated jointly with the arm means and shared by the arms.
    Estimated(CorrelationPrior),
}

/// What one arm contributes: its resolved prior and the histogram of its
/// completed participants' success counts.
#[derive(Clone, Copy, Debug)]
pub struct ArmEvidence<'a> {
    /// First Beta prior shape, at least one.
    pub prior_alpha: f64,
    /// Second Beta prior shape, at least one.
    pub prior_beta: f64,
    /// `counts[s]` participants with `s` successes, `s = 0..=responses`.
    pub counts: &'a [u64],
}

/// A posterior to compute.
#[derive(Clone, Copy, Debug)]
pub struct PosteriorRequest<'a> {
    /// Responses per participant, from 2 to [`MAX_RESPONSES`].
    pub responses: u32,
    /// The correlation model.
    pub correlation: CorrelationModel,
    /// The arms; arm 0 is the control.
    pub arms: &'a [ArmEvidence<'a>],
    /// When set, the posterior probability that each treatment arm exceeds the
    /// control by more than this margin (in `[0, 1)`) is computed.
    pub exceedance_margin: Option<f64>,
}

/// The posterior functionals.
#[derive(Clone, Debug, PartialEq)]
pub struct ExactPosterior {
    /// Posterior mean of each arm's success probability.
    pub mean: Vec<f64>,
    /// Posterior standard deviation of each arm's success probability.
    pub sd: Vec<f64>,
    /// Posterior probability that each arm has the largest success probability.
    pub probability_best: Vec<f64>,
    /// Posterior expected loss of choosing each arm, `E[max_j p_j] - E[p_k]`.
    pub expected_loss: Vec<f64>,
    /// For arms `1..`, the posterior probability of exceeding the control by
    /// the requested margin; empty when no margin was requested.
    pub exceedance: Vec<f64>,
    /// Posterior mean of the correlation (the declared value when declared).
    pub correlation_mean: f64,
    /// Posterior standard deviation of the correlation (zero when declared).
    pub correlation_sd: f64,
    /// `|sum_k P(k best) - 1|`: the quadrature error read from the identity
    /// that the arms' best-probabilities sum to one.
    pub probability_error: f64,
    /// Correlation nodes of the final integration (one when declared).
    pub correlation_nodes: usize,
    /// Logit nodes of the final integration.
    pub arm_nodes: usize,
}

// -- Quadrature rule ------------------------------------------------------------------------------

struct Rule {
    nodes: [f64; NODES],
    weights: [f64; NODES],
    /// `integral[i][l]`: the integral from -1 to node `i` of the Lagrange basis `l`.
    integral: [[f64; NODES]; NODES],
    /// `projection[n][q]`: coefficient of `P_n` from the value at node `q`.
    projection: [[f64; NODES]; NODES],
}

fn legendre_values(t: f64) -> [f64; NODES + 1] {
    let mut p = [0.0; NODES + 1];
    p[0] = 1.0;
    p[1] = t;
    for n in 1..NODES {
        let nf = n as f64;
        p[n + 1] = ((2.0 * nf + 1.0) * t * p[n] - nf * p[n - 1]) / (nf + 1.0);
    }
    p
}

fn build_rule() -> Rule {
    let (xs, ws) = gauss_legendre(NODES);
    let mut nodes = [0.0; NODES];
    let mut weights = [0.0; NODES];
    nodes.copy_from_slice(&xs);
    weights.copy_from_slice(&ws);
    let lagrange = |l: usize, y: f64| -> f64 {
        (0..NODES).filter(|&j| j != l).map(|j| (y - nodes[j]) / (nodes[l] - nodes[j])).product()
    };
    let mut integral = [[0.0; NODES]; NODES];
    for (i, row) in integral.iter_mut().enumerate() {
        let span = 0.5 * (nodes[i] + 1.0);
        for (l, cell) in row.iter_mut().enumerate() {
            *cell = (0..NODES)
                .map(|q| {
                    let y = -1.0 + (nodes[i] + 1.0) * (nodes[q] + 1.0) * 0.5;
                    span * weights[q] * lagrange(l, y)
                })
                .sum();
        }
    }
    let mut projection = [[0.0; NODES]; NODES];
    for q in 0..NODES {
        let p = legendre_values(nodes[q]);
        for (n, row) in projection.iter_mut().enumerate() {
            row[q] = (2.0 * n as f64 + 1.0) * 0.5 * weights[q] * p[n];
        }
    }
    Rule { nodes, weights, integral, projection }
}

fn rule() -> &'static Rule {
    static RULE: OnceLock<Rule> = OnceLock::new();
    RULE.get_or_init(build_rule)
}

// -- Elementary functions -------------------------------------------------------------------------

fn ln_sigmoid(u: f64) -> f64 {
    if u >= 0.0 { -(-u).exp().ln_1p() } else { u - u.exp().ln_1p() }
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the value is a nonnegative ceiling already bounded by a panel cap before the cast"
)]
fn whole(value: f64) -> usize {
    value as usize
}

fn design_effect(responses: u32, rho: f64) -> f64 {
    1.0 + f64::from(responses - 1) * rho
}

// -- Arm statistics and windows -------------------------------------------------------------------

struct ArmStats {
    prior: (f64, f64),
    successes: f64,
    trials: f64,
    counts: Vec<u64>,
}

impl ArmStats {
    fn new(arm: &ArmEvidence<'_>, responses: u32) -> Result<Self, Error> {
        if arm.counts.len() != responses as usize + 1 {
            return Err(Error::Shape("an arm needs a count for each number of successes"));
        }
        if !(arm.prior_alpha.is_finite()
            && arm.prior_beta.is_finite()
            && arm.prior_alpha >= 1.0
            && arm.prior_beta >= 1.0)
        {
            return Err(Error::InvalidParameter(
                "a Beta prior shape must be finite and at least one",
            ));
        }
        let mut participants = 0.0_f64;
        let mut successes = 0.0_f64;
        for (s, &n) in arm.counts.iter().enumerate() {
            participants += n as f64;
            successes += s as f64 * n as f64;
        }
        Ok(Self {
            prior: (arm.prior_alpha, arm.prior_beta),
            successes,
            trials: participants * f64::from(responses),
            counts: arm.counts.to_vec(),
        })
    }

    /// Locator window and narrowest scale over the design effects `d_lo..=d_hi`,
    /// from the logit moments of the Beta shapes the design effect gives.
    fn window(&self, d_lo: f64, d_hi: f64) -> ((f64, f64), f64) {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        let mut narrowest = f64::INFINITY;
        for d in [d_lo, d_hi] {
            let a = self.prior.0 + self.successes / d;
            let b = self.prior.1 + (self.trials - self.successes) / d;
            let centre = digamma(a) - digamma(b);
            let sigma = (trigamma(a) + trigamma(b)).sqrt();
            lo = lo.min(centre - (WINDOW_SIGMAS * sigma).max(TAIL_DECAY / a));
            hi = hi.max(centre + (WINDOW_SIGMAS * sigma).max(TAIL_DECAY / b));
            narrowest = narrowest.min(sigma);
        }
        ((lo.max(-U_LIMIT), hi.min(U_LIMIT)), narrowest)
    }
}

// -- The logit layout -----------------------------------------------------------------------------

struct Layout {
    /// Panels `(left, right)`, ascending and disjoint.
    panels: Vec<(f64, f64)>,
    u: Vec<f64>,
    weight: Vec<f64>,
    mu: Vec<f64>,
    mu_c: Vec<f64>,
    ln_mu: Vec<f64>,
    ln_mu_c: Vec<f64>,
}

fn build_layout(windows: &[(f64, f64)], scales: &[f64]) -> Result<Layout, Error> {
    let mut cuts: Vec<f64> = windows.iter().flat_map(|w| [w.0, w.1]).collect();
    cuts.sort_by(f64::total_cmp);
    cuts.dedup();
    let mut panels: Vec<(f64, f64)> = Vec::new();
    for pair in cuts.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        if b <= a {
            continue;
        }
        let mid = 0.5 * (a + b);
        let scale = windows
            .iter()
            .zip(scales)
            .filter(|(w, _)| w.0 <= mid && mid <= w.1)
            .map(|(_, &s)| s)
            .fold(f64::INFINITY, f64::min);
        if !scale.is_finite() {
            continue;
        }
        let count = ((b - a) / (PANEL_SIGMAS * scale)).ceil().max(1.0);
        if !count.is_finite() || count + panels.len() as f64 > MAX_PANELS as f64 {
            return Err(Error::Limit(
                "the participant posteriors are too concentrated for the quadrature",
            ));
        }
        let count = whole(count);
        for i in 0..count {
            let left = a + (b - a) * i as f64 / count as f64;
            let right = a + (b - a) * (i + 1) as f64 / count as f64;
            panels.push((left, right));
        }
    }
    if panels.is_empty() {
        return Err(Error::Numerical("the quadrature window is empty"));
    }
    let rule = rule();
    let capacity = panels.len() * NODES;
    let mut layout = Layout {
        u: Vec::with_capacity(capacity),
        weight: Vec::with_capacity(capacity),
        mu: Vec::with_capacity(capacity),
        mu_c: Vec::with_capacity(capacity),
        ln_mu: Vec::with_capacity(capacity),
        ln_mu_c: Vec::with_capacity(capacity),
        panels,
    };
    for &(a, b) in &layout.panels.clone() {
        let centre = 0.5 * (a + b);
        let half = 0.5 * (b - a);
        for q in 0..NODES {
            let u = centre + half * rule.nodes[q];
            layout.u.push(u);
            layout.weight.push(half * rule.weights[q]);
            layout.mu.push(sigmoid(u));
            layout.mu_c.push(sigmoid(-u));
            layout.ln_mu.push(ln_sigmoid(u));
            layout.ln_mu_c.push(ln_sigmoid(-u));
        }
    }
    Ok(layout)
}

/// `table[s][g] = ln P(s | mu_g, rho)` up to the binomial coefficient, which
/// does not depend on `(mu, rho)`.
fn log_pmf_table(layout: &Layout, responses: u32, rho: f64) -> Vec<Vec<f64>> {
    let m = responses as usize;
    let n = layout.u.len();
    let d = rho / (1.0 - rho);
    let mut a = vec![vec![0.0; n]; m + 1];
    let mut b = vec![vec![0.0; n]; m + 1];
    for j in 0..m {
        let shift = j as f64 * d;
        for g in 0..n {
            let (ta, tb) = if j == 0 || d == 0.0 {
                (layout.ln_mu[g], layout.ln_mu_c[g])
            } else {
                ((layout.mu[g] + shift).ln(), (layout.mu_c[g] + shift).ln())
            };
            a[j + 1][g] = a[j][g] + ta;
            b[j + 1][g] = b[j][g] + tb;
        }
    }
    let log_normalizer: f64 = (0..m).map(|j| (1.0 + j as f64 * d).ln()).sum();
    (0..=m).map(|s| (0..n).map(|g| a[s][g] + b[m - s][g] - log_normalizer).collect()).collect()
}

fn arm_log_density(layout: &Layout, arm: &ArmStats, table: &[Vec<f64>]) -> Vec<f64> {
    let n = layout.u.len();
    let mut out: Vec<f64> =
        (0..n).map(|g| arm.prior.0 * layout.ln_mu[g] + arm.prior.1 * layout.ln_mu_c[g]).collect();
    for (s, &count) in arm.counts.iter().enumerate() {
        if count > 0 {
            let c = count as f64;
            for (value, entry) in out.iter_mut().zip(&table[s]) {
                *value += c * entry;
            }
        }
    }
    out
}

/// `ln Σ wᵢ exp(vᵢ)` and the peak it was shifted by. When no term carries mass (every value is
/// `-inf`, or every weight is zero) the log-sum is `-inf`, never `NaN`, so the caller's finiteness
/// check refuses it.
fn log_sum_exp_weighted(values: &[f64], weights: &[f64]) -> (f64, f64) {
    let peak = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !peak.is_finite() {
        return (if peak.is_nan() { f64::NAN } else { peak }, peak);
    }
    let total: f64 = values.iter().zip(weights).map(|(v, w)| w * (v - peak).exp()).sum();
    if total > 0.0 { (peak + total.ln(), peak) } else { (f64::NEG_INFINITY, peak) }
}

// -- One evaluation of the arm integrals over a set of correlation nodes --------------------------

struct Evaluation {
    layout: Layout,
    /// `lw[r][k][g]`.
    lw: Vec<Vec<Vec<f64>>>,
    /// `ln_z[r][k]`.
    ln_z: Vec<Vec<f64>>,
    /// Log posterior weight of each correlation node, up to a constant.
    ln_weight: Vec<f64>,
}

fn evaluate(
    request: &PosteriorRequest<'_>,
    arms: &[ArmStats],
    nodes: &[(f64, f64)],
    cancelled: &dyn Fn() -> bool,
) -> Result<Evaluation, Error> {
    let rho_lo = nodes.iter().map(|n| n.0).fold(f64::INFINITY, f64::min);
    let rho_hi = nodes.iter().map(|n| n.0).fold(f64::NEG_INFINITY, f64::max);
    let d_lo = design_effect(request.responses, rho_lo);
    let d_hi = design_effect(request.responses, rho_hi);
    let located: Vec<((f64, f64), f64)> = arms.iter().map(|arm| arm.window(d_lo, d_hi)).collect();
    let mut windows: Vec<(f64, f64)> = located.iter().map(|w| w.0).collect();
    let scales: Vec<f64> = located.iter().map(|w| w.1).collect();
    for attempt in 0..=MAX_WIDENINGS {
        let layout = build_layout(&windows, &scales)?;
        let mut lw = Vec::with_capacity(nodes.len());
        let mut ln_z = Vec::with_capacity(nodes.len());
        let mut peaks = Vec::with_capacity(nodes.len());
        for (index, &(rho, _)) in nodes.iter().enumerate() {
            if index % CANCEL_STRIDE == 0 && cancelled() {
                return Err(Error::Cancelled);
            }
            let table = log_pmf_table(&layout, request.responses, rho);
            let mut row = Vec::with_capacity(arms.len());
            let mut zs = Vec::with_capacity(arms.len());
            let mut pk = Vec::with_capacity(arms.len());
            for arm in arms {
                let values = arm_log_density(&layout, arm, &table);
                let (z, peak) = log_sum_exp_weighted(&values, &layout.weight);
                if !z.is_finite() {
                    return Err(Error::Numerical("an arm posterior has no finite mass"));
                }
                row.push(values);
                zs.push(z);
                pk.push(peak);
            }
            lw.push(row);
            ln_z.push(zs);
            peaks.push(pk);
        }
        let ln_weight: Vec<f64> =
            nodes.iter().zip(&ln_z).map(|(n, z)| n.1 + z.iter().sum::<f64>()).collect();
        let best = ln_weight.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        // Edge check: the density of every arm at its window edges, at every node
        // that matters, is negligible against its peak.
        let mut widen = vec![(false, false); arms.len()];
        for (r, row) in lw.iter().enumerate() {
            if ln_weight[r] < best + NEGLIGIBLE_WEIGHT_LOG {
                continue;
            }
            for (k, values) in row.iter().enumerate() {
                let (lo, hi) = windows[k];
                let first = layout.u.partition_point(|&x| x < lo);
                let last = layout.u.partition_point(|&x| x <= hi);
                if first < layout.u.len()
                    && lo > -U_LIMIT
                    && values[first] - peaks[r][k] > EDGE_LOG_TOLERANCE
                {
                    widen[k].0 = true;
                }
                if last > 0 && hi < U_LIMIT && values[last - 1] - peaks[r][k] > EDGE_LOG_TOLERANCE {
                    widen[k].1 = true;
                }
            }
        }
        if widen.iter().all(|w| !w.0 && !w.1) {
            return Ok(Evaluation { layout, lw, ln_z, ln_weight });
        }
        if attempt == MAX_WIDENINGS {
            break;
        }
        for (window, flag) in windows.iter_mut().zip(&widen) {
            let extra = 0.5 * (window.1 - window.0) + 2.0;
            if flag.0 {
                window.0 = (window.0 - extra).max(-U_LIMIT);
            }
            if flag.1 {
                window.1 = (window.1 + extra).min(U_LIMIT);
            }
        }
    }
    Err(Error::Numerical("the participant posterior does not vanish at the edge of its window"))
}

// -- Functionals of one correlation node ----------------------------------------------------------

struct NodeFunctionals {
    mean: Vec<f64>,
    second: Vec<f64>,
    best: Vec<f64>,
    expected_max: f64,
    exceedance: Vec<f64>,
}

/// Cumulative distribution function of one arm: panel prefix masses and the
/// Legendre coefficients of the density on every panel.
struct Cdf<'a> {
    layout: &'a Layout,
    coefficients: Vec<[f64; NODES]>,
    prefix: Vec<f64>,
}

impl<'a> Cdf<'a> {
    fn new(layout: &'a Layout, density: &[f64]) -> Self {
        let rule = rule();
        let mut coefficients = Vec::with_capacity(layout.panels.len());
        let mut prefix = Vec::with_capacity(layout.panels.len() + 1);
        let mut running = 0.0;
        prefix.push(0.0);
        for (q, &(a, b)) in layout.panels.iter().enumerate() {
            let values = &density[q * NODES..(q + 1) * NODES];
            let mut c = [0.0; NODES];
            for (n, slot) in c.iter_mut().enumerate() {
                *slot = (0..NODES).map(|i| rule.projection[n][i] * values[i]).sum();
            }
            // The integral of P_0 over the panel is its width; the others integrate to zero.
            running += (b - a) * c[0];
            prefix.push(running);
            coefficients.push(c);
        }
        Self { layout, coefficients, prefix }
    }

    fn at(&self, u: f64) -> f64 {
        let q = self.layout.panels.partition_point(|p| p.1 <= u);
        if q == self.layout.panels.len() {
            return self.prefix[q].clamp(0.0, 1.0);
        }
        let (a, b) = self.layout.panels[q];
        if u <= a {
            return self.prefix[q].clamp(0.0, 1.0);
        }
        let half = 0.5 * (b - a);
        let t = ((u - a) / half - 1.0).clamp(-1.0, 1.0);
        let p = legendre_values(t);
        let c = &self.coefficients[q];
        let mut sum = c[0] * (t + 1.0);
        for n in 1..NODES {
            sum += c[n] * (p[n + 1] - p[n - 1]) / (2.0 * n as f64 + 1.0);
        }
        (self.prefix[q] + half * sum).clamp(0.0, 1.0)
    }
}

/// The cumulative distribution of every arm at the layout nodes.
fn cdf_at_nodes(layout: &Layout, density: &[f64]) -> Vec<f64> {
    let rule = rule();
    let mut out = vec![0.0; density.len()];
    let mut running = 0.0;
    for (q, &(a, b)) in layout.panels.iter().enumerate() {
        let half = 0.5 * (b - a);
        let values = &density[q * NODES..(q + 1) * NODES];
        for i in 0..NODES {
            let inner: f64 = (0..NODES).map(|l| rule.integral[i][l] * values[l]).sum();
            out[q * NODES + i] = (running + half * inner).clamp(0.0, 1.0);
        }
        running += half * (0..NODES).map(|l| rule.weights[l] * values[l]).sum::<f64>();
    }
    out
}

fn node_functionals(
    layout: &Layout,
    lw: &[Vec<f64>],
    ln_z: &[f64],
    margin: Option<f64>,
) -> NodeFunctionals {
    let count = lw.len();
    let n = layout.u.len();
    let density: Vec<Vec<f64>> = lw
        .iter()
        .zip(ln_z)
        .map(|(values, &z)| values.iter().map(|&v| (v - z).exp()).collect())
        .collect();
    let cdf: Vec<Vec<f64>> = density.iter().map(|d| cdf_at_nodes(layout, d)).collect();
    let mut mean = vec![0.0; count];
    let mut second = vec![0.0; count];
    let mut best = vec![0.0; count];
    let mut expected_max = 0.0;
    let mut suffix = vec![1.0; count + 1];
    for g in 0..n {
        for k in (0..count).rev() {
            suffix[k] = suffix[k + 1] * cdf[k][g];
        }
        let mut prefix = 1.0;
        let w = layout.weight[g];
        let mu = layout.mu[g];
        for k in 0..count {
            let mass = w * density[k][g];
            mean[k] += mass * mu;
            second[k] += mass * mu * mu;
            let others = prefix * suffix[k + 1];
            best[k] += mass * others;
            expected_max += mass * mu * others;
            prefix *= cdf[k][g];
        }
    }
    let exceedance = match margin {
        None => Vec::new(),
        Some(margin) => {
            let control = Cdf::new(layout, &density[0]);
            (1..count)
                .map(|k| {
                    (0..n)
                        .map(|g| {
                            let x = layout.mu[g] - margin;
                            if x <= 0.0 {
                                0.0
                            } else {
                                let u0 = (x / (1.0 - x)).ln();
                                layout.weight[g] * density[k][g] * control.at(u0)
                            }
                        })
                        .sum()
                })
                .collect()
        }
    };
    NodeFunctionals { mean, second, best, expected_max, exceedance }
}

// -- Correlation nodes ----------------------------------------------------------------------------

fn correlation_nodes(
    prior: CorrelationPrior,
    window: (f64, f64),
    panels: usize,
) -> Vec<(f64, f64)> {
    let rule = rule();
    let width = (window.1 - window.0) / panels as f64;
    let mut nodes = Vec::with_capacity(panels * NODES);
    for panel in 0..panels {
        let centre = window.0 + width * (panel as f64 + 0.5);
        for q in 0..NODES {
            let rho = centre + 0.5 * width * rule.nodes[q];
            let ln_density = (prior.alpha - 1.0) * rho.ln() + (prior.beta - 1.0) * (-rho).ln_1p();
            nodes.push((rho, (0.5 * width * rule.weights[q]).ln() + ln_density));
        }
    }
    nodes
}

fn normalized(ln_weight: &[f64]) -> Vec<f64> {
    let peak = ln_weight.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let raw: Vec<f64> = ln_weight.iter().map(|w| (w - peak).exp()).collect();
    let total: f64 = raw.iter().sum();
    raw.iter().map(|r| r / total).collect()
}

// -- The posterior --------------------------------------------------------------------------------

/// The exact posterior functionals of the arms' success probabilities.
///
/// # Errors
/// A request outside its bounds (responses, arms, prior shapes, counts, the
/// correlation or its prior, the margin), a posterior too concentrated for the
/// quadrature panels, a posterior that does not vanish at the edge of its
/// window, a correlation integration that does not settle, or cancellation.
pub fn exact_posterior(
    request: &PosteriorRequest<'_>,
    cancelled: &dyn Fn() -> bool,
) -> Result<ExactPosterior, Error> {
    if !(2..=MAX_RESPONSES).contains(&request.responses) {
        return Err(Error::InvalidParameter("responses per participant must lie in 2..=32"));
    }
    if !(2..=MAX_ARMS).contains(&request.arms.len()) {
        return Err(Error::Shape("a posterior comparison needs between 2 and 64 arms"));
    }
    if let Some(margin) = request.exceedance_margin {
        if !(margin.is_finite() && (0.0..1.0).contains(&margin)) {
            return Err(Error::InvalidParameter("a margin must lie in [0, 1)"));
        }
    }
    let arms = request
        .arms
        .iter()
        .map(|arm| ArmStats::new(arm, request.responses))
        .collect::<Result<Vec<_>, _>>()?;
    let (evaluation, nodes) = match request.correlation {
        CorrelationModel::Declared(rho) => {
            if !(rho.is_finite() && (0.0..1.0).contains(&rho)) {
                return Err(Error::InvalidParameter("a correlation must lie in [0, 1)"));
            }
            let nodes = vec![(rho, 0.0)];
            (evaluate(request, &arms, &nodes, cancelled)?, nodes)
        }
        CorrelationModel::Estimated(prior) => {
            if !prior.is_valid() {
                return Err(Error::InvalidParameter(
                    "a correlation prior needs Beta shapes between one and 10000",
                ));
            }
            settle_correlation(request, &arms, prior, cancelled)?
        }
    };
    assemble(request, &evaluation, &nodes)
}

fn settle_correlation(
    request: &PosteriorRequest<'_>,
    arms: &[ArmStats],
    prior: CorrelationPrior,
    cancelled: &dyn Fn() -> bool,
) -> Result<(Evaluation, Vec<(f64, f64)>), Error> {
    let mut window = (0.0, 1.0);
    let mut panels = RHO_PANELS_FIRST;
    // How many posterior standard deviations each side of the window reaches; a side whose
    // edge density is not negligible reaches further on the next pass.
    let mut reach = (RHO_WINDOW_SIGMAS, RHO_WINDOW_SIGMAS);
    for _ in 0..MAX_RHO_PASSES {
        let nodes = correlation_nodes(prior, window, panels);
        let evaluation = evaluate(request, arms, &nodes, cancelled)?;
        let q = normalized(&evaluation.ln_weight);
        let mean: f64 = nodes.iter().zip(&q).map(|(n, w)| n.0 * w).sum();
        let variance: f64 = nodes.iter().zip(&q).map(|(n, w)| (n.0 - mean).powi(2) * w).sum();
        let sd = variance.max(0.0).sqrt();
        let panel_width = (window.1 - window.0) / panels as f64;
        // Density (without the quadrature weight) at the window edges.
        let densities: Vec<f64> = evaluation
            .ln_weight
            .iter()
            .enumerate()
            .map(|(i, w)| w - rule().weights[i % NODES].ln() - (0.5 * panel_width).ln())
            .collect();
        let peak = densities.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let first_ok = window.0 <= 0.0 || densities[0] - peak <= EDGE_LOG_TOLERANCE;
        let last_ok =
            window.1 >= 1.0 || densities[densities.len() - 1] - peak <= EDGE_LOG_TOLERANCE;
        if first_ok && last_ok && panel_width <= RHO_PANEL_SIGMAS * sd {
            return Ok((evaluation, nodes));
        }
        if !first_ok {
            reach.0 *= 1.5;
        }
        if !last_ok {
            reach.1 *= 1.5;
        }
        let effective = sd.max(panel_width / 8.0);
        let lo = (mean - reach.0 * effective).max(0.0);
        let hi = (mean + reach.1 * effective).min(1.0);
        if hi - lo < 1e-9 {
            return Err(Error::Numerical("the correlation posterior collapsed to a point"));
        }
        window = (lo, hi);
        let wanted = ((hi - lo) / (0.85 * RHO_PANEL_SIGMAS * effective)).ceil();
        panels = whole(wanted.max(RHO_PANELS_REFINED as f64)).min(RHO_PANELS_MAX);
    }
    Err(Error::Numerical("the correlation integration did not settle"))
}

fn assemble(
    request: &PosteriorRequest<'_>,
    evaluation: &Evaluation,
    nodes: &[(f64, f64)],
) -> Result<ExactPosterior, Error> {
    let count = request.arms.len();
    let q = normalized(&evaluation.ln_weight);
    let mut mean = vec![0.0; count];
    let mut second = vec![0.0; count];
    let mut best = vec![0.0; count];
    let mut loss_max = 0.0;
    let mut exceedance = vec![0.0; if request.exceedance_margin.is_some() { count - 1 } else { 0 }];
    for (r, &weight) in q.iter().enumerate() {
        if weight < 1e-18 {
            continue;
        }
        let f = node_functionals(
            &evaluation.layout,
            &evaluation.lw[r],
            &evaluation.ln_z[r],
            request.exceedance_margin,
        );
        for k in 0..count {
            mean[k] += weight * f.mean[k];
            second[k] += weight * f.second[k];
            best[k] += weight * f.best[k];
        }
        loss_max += weight * f.expected_max;
        for (slot, value) in exceedance.iter_mut().zip(&f.exceedance) {
            *slot += weight * value;
        }
    }
    let sd = mean.iter().zip(&second).map(|(m, s)| (s - m * m).max(0.0).sqrt()).collect();
    let expected_loss = mean.iter().map(|m| (loss_max - m).max(0.0)).collect();
    let total: f64 = best.iter().sum();
    let correlation_mean: f64 = nodes.iter().zip(&q).map(|(n, w)| n.0 * w).sum();
    let correlation_variance: f64 =
        nodes.iter().zip(&q).map(|(n, w)| (n.0 - correlation_mean).powi(2) * w).sum();
    if !(total.is_finite() && mean.iter().all(|m| m.is_finite())) {
        return Err(Error::Numerical("the participant posterior functionals are not finite"));
    }
    Ok(ExactPosterior {
        mean,
        sd,
        probability_best: best,
        expected_loss,
        exceedance: exceedance.into_iter().map(|e| e.clamp(0.0, 1.0)).collect(),
        correlation_mean,
        correlation_sd: correlation_variance.max(0.0).sqrt(),
        probability_error: (total - 1.0).abs(),
        correlation_nodes: nodes.len(),
        arm_nodes: evaluation.layout.u.len(),
    })
}

#[cfg(test)]
mod log_sum_exp_tests {
    use super::log_sum_exp_weighted;

    #[test]
    fn massless_inputs_give_negative_infinity_not_nan() {
        let (z, _) = log_sum_exp_weighted(&[f64::NEG_INFINITY, f64::NEG_INFINITY], &[1.0, 1.0]);
        assert_eq!(z, f64::NEG_INFINITY);
        let (z, _) = log_sum_exp_weighted(&[0.0, 1.0], &[0.0, 0.0]);
        assert_eq!(z, f64::NEG_INFINITY);
        let (z, peak) = log_sum_exp_weighted(&[0.0, 0.0], &[1.0, 1.0]);
        assert!((z - 2.0_f64.ln()).abs() < 1e-15 && peak == 0.0);
    }
}
