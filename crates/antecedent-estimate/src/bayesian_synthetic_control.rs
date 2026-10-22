//! Bayesian simplex synthetic control with a Dirichlet donor-weight prior.
//!
//! The estimand is the one of [`crate::synthetic_control::fit_synthetic_control`]: the average
//! post-intervention gap between the treated unit and its counterfactual, the donor-weighted
//! average of the donors' outcomes. The design premise is unchanged and no prior supplies it: the
//! treated unit's untreated path lies in the convex hull of the donors' (a donor-weight simplex
//! exists), donors are unaffected by the intervention, and the pre-period relation persists.
//! The pre-fit and range diagnostics below are target-native descriptions of that premise; a good
//! pre-fit does not establish it for the post period.
//!
//! Model: pre-period treated outcomes `y_t = sum_j w_j x_jt + e_t`, `w ~ Dirichlet(alpha 1)` on
//! the simplex, `e` a stationary Gaussian AR(1) with declared coefficient `rho` and innovation
//! variance `sigma2 ~ InvGamma(a, b)`. The counterfactual post-period mean carries the same
//! error process continued from the last pre-period residual, so the effect draw is the realized
//! average effect on the treated, not only the weight-driven gap. The posterior has no closed form
//! and is sampled by pairwise-transfer Metropolis steps on the simplex with a Gibbs update of
//! `sigma2`, over four overdispersed chains whose split R-hat and effective sample size gate the
//! result.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use antecedent_core::CausalRng;
use antecedent_kernels::{sample_gamma, sample_inv_gamma, standard_normal};

use crate::bayesian_panel_did::summarize;
use crate::error::EstimationError;
use crate::util::DrawSpec;

const CHAINS: usize = 4;
/// Metropolis sweeps between kept draws.
const THIN: usize = 5;
/// Largest split R-hat of the gap and the residual variance that is accepted.
const MAX_RHAT: f64 = 1.05;
/// Smallest total bulk effective sample size that is accepted.
const MIN_BULK_ESS: f64 = 100.0;
/// Pairwise-transfer proposal scales, one chosen uniformly per proposal.
const STEP_SCALES: [f64; 3] = [0.3, 0.05, 0.01];
/// Seed offset that separates the chains.
const CHAIN_SEED_STRIDE: u64 = 0xD1B5_4A32_D192_ED03;

/// Declared prior and sampler of a Bayesian synthetic control.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BayesianSyntheticControlSpec<'a> {
    /// The single treated unit.
    pub treated_unit: &'a str,
    /// First post-intervention period.
    pub intervention_period: i64,
    /// Symmetric Dirichlet concentration of the donor weights (1 is uniform on the simplex).
    pub dirichlet_concentration: f64,
    /// Inverse-Gamma shape of the AR(1) innovation variance.
    pub variance_shape: f64,
    /// Inverse-Gamma scale of the AR(1) innovation variance.
    pub variance_scale: f64,
    /// Declared AR(1) coefficient of the pre-period fit errors, in (-1, 1); 0 is independent errors.
    pub ar1_rho: f64,
    /// Sweeps discarded before draws are kept, per chain.
    pub burn_in: usize,
}

impl<'a> BayesianSyntheticControlSpec<'a> {
    /// A uniform Dirichlet prior, a vague variance prior, independent errors and 1,000 burn-in sweeps.
    #[must_use]
    pub const fn new(treated_unit: &'a str, intervention_period: i64) -> Self {
        Self {
            treated_unit,
            intervention_period,
            dirichlet_concentration: 1.0,
            variance_shape: 1e-3,
            variance_scale: 1e-3,
            ar1_rho: 0.0,
            burn_in: 1_000,
        }
    }
}

/// Posterior of the average post-intervention effect on the treated unit.
#[derive(Clone, Debug, PartialEq)]
pub struct BayesianSyntheticControlEstimate {
    /// Effect draws, chain-major.
    pub effect_draws: Vec<f64>,
    /// Posterior mean.
    pub mean: f64,
    /// Posterior standard deviation.
    pub sd: f64,
    /// Equal-tailed 95% lower posterior quantile.
    pub lower: f64,
    /// Equal-tailed 95% upper posterior quantile.
    pub upper: f64,
    /// Posterior mean donor weights in stable unit-name order.
    pub donor_weight_means: Vec<(String, f64)>,
    /// Posterior standard deviation of each donor weight, in the same order.
    pub donor_weight_sds: Vec<f64>,
    /// Root mean squared pre-period gap at the posterior-mean weights.
    pub pre_fit_rmse: f64,
    /// Pre-periods in which the treated outcome lies outside the range of the donors' outcomes,
    /// where no convex combination can match it.
    pub pre_periods_outside_donor_range: usize,
    /// Fraction of Metropolis proposals accepted.
    pub acceptance_rate: f64,
    /// Largest split R-hat of the weight-driven gap and the innovation variance.
    pub max_split_rhat: f64,
    /// Smallest bulk effective sample size of the same two quantities, across all chains.
    pub min_bulk_ess: f64,
    /// Observed pre-intervention periods.
    pub n_pre_periods: usize,
    /// Observed post-intervention periods.
    pub n_post_periods: usize,
    /// Explicit model scope.
    pub model: &'static str,
}

struct ScPanel {
    donors: Vec<String>,
    target_pre: Vec<f64>,
    donor_pre: Vec<Vec<f64>>,
    target_post_mean: f64,
    donor_post_mean: Vec<f64>,
    n_post: usize,
}

fn parse_panel(
    outcome: &[f64],
    units: &[String],
    periods: &[i64],
    spec: &BayesianSyntheticControlSpec<'_>,
) -> Result<ScPanel, EstimationError> {
    let n = outcome.len();
    if n == 0 || units.len() != n || periods.len() != n || outcome.iter().any(|v| !v.is_finite()) {
        return Err(EstimationError::stats_msg(
            "synthetic control needs equal non-empty finite outcome, unit and period vectors",
        ));
    }
    let mut panel: BTreeMap<&str, BTreeMap<i64, f64>> = BTreeMap::new();
    let mut all_periods = BTreeSet::new();
    for i in 0..n {
        if units[i].is_empty() || periods[i] <= 0 {
            return Err(EstimationError::stats_msg(
                "unit IDs must be non-empty and periods positive",
            ));
        }
        all_periods.insert(periods[i]);
        if panel.entry(units[i].as_str()).or_default().insert(periods[i], outcome[i]).is_some() {
            return Err(EstimationError::stats_msg(
                "each unit must have exactly one outcome per period",
            ));
        }
    }
    let pre: Vec<i64> =
        all_periods.iter().copied().filter(|p| *p < spec.intervention_period).collect();
    let post: Vec<i64> =
        all_periods.iter().copied().filter(|p| *p >= spec.intervention_period).collect();
    if pre.len() < 2 || post.is_empty() {
        return Err(EstimationError::stats_msg(
            "synthetic control requires at least two pre-periods and one post-period",
        ));
    }
    if panel.len() < 4 || !panel.contains_key(spec.treated_unit) {
        return Err(EstimationError::stats_msg(
            "synthetic control requires one observed treated unit and at least three donor units",
        ));
    }
    if panel.values().any(|observed| all_periods.iter().any(|p| !observed.contains_key(p))) {
        return Err(EstimationError::stats_msg(
            "synthetic control requires a balanced panel with identical periods for every unit",
        ));
    }
    let treated = &panel[spec.treated_unit];
    let donors: Vec<(&&str, &BTreeMap<i64, f64>)> =
        panel.iter().filter(|(unit, _)| **unit != spec.treated_unit).collect();
    let mean_post = |values: &BTreeMap<i64, f64>| {
        post.iter().map(|p| values[p]).sum::<f64>() / post.len() as f64
    };
    Ok(ScPanel {
        donors: donors.iter().map(|(unit, _)| (**unit).to_string()).collect(),
        target_pre: pre.iter().map(|p| treated[p]).collect(),
        donor_pre: donors.iter().map(|(_, v)| pre.iter().map(|p| v[p]).collect()).collect(),
        target_post_mean: mean_post(treated),
        donor_post_mean: donors.iter().map(|(_, v)| mean_post(v)).collect(),
        n_post: post.len(),
    })
}

/// Pre-period residuals `y_t - sum_j w_j x_jt`.
fn residuals(panel: &ScPanel, weights: &[f64]) -> Vec<f64> {
    panel
        .target_pre
        .iter()
        .enumerate()
        .map(|(t, y)| y - panel.donor_pre.iter().zip(weights).map(|(x, w)| w * x[t]).sum::<f64>())
        .collect()
}

/// Sum of squared innovations of a stationary AR(1) error path.
fn whitened_ss(residual: &[f64], rho: f64) -> f64 {
    let first = residual.first().map_or(0.0, |r| (1.0 - rho * rho) * r * r);
    first + residual.windows(2).map(|w| (w[1] - rho * w[0]).powi(2)).sum::<f64>()
}

struct ChainState {
    weights: Vec<f64>,
    residual: Vec<f64>,
    ss: f64,
    sigma2: f64,
}

/// A uniform index in `0..n`.
fn below(rng: &mut CausalRng, n: usize) -> usize {
    let bound = u64::try_from(n).unwrap_or(u64::MAX);
    usize::try_from(rng.next_u64() % bound).unwrap_or(0)
}

/// `rho^k` for a small non-negative `k`.
fn power(rho: f64, k: usize) -> f64 {
    rho.powi(i32::try_from(k).unwrap_or(i32::MAX))
}

/// One pairwise-transfer Metropolis proposal; true when accepted.
fn transfer_step(
    state: &mut ChainState,
    panel: &ScPanel,
    spec: &BayesianSyntheticControlSpec<'_>,
    rng: &mut CausalRng,
) -> bool {
    let donors = state.weights.len();
    let i = below(rng, donors);
    let j = (i + 1 + below(rng, donors - 1)) % donors;
    let scale = STEP_SCALES[below(rng, STEP_SCALES.len())];
    let delta = scale * standard_normal(rng);
    let (wi, wj) = (state.weights[i] + delta, state.weights[j] - delta);
    let uniform = rng.next_f64();
    if wi <= 0.0 || wj <= 0.0 {
        return false;
    }
    let proposed: Vec<f64> = state
        .residual
        .iter()
        .zip(&panel.donor_pre[i])
        .zip(&panel.donor_pre[j])
        .map(|((r, xi), xj)| r - delta * (xi - xj))
        .collect();
    let ss = whitened_ss(&proposed, spec.ar1_rho);
    let prior = (spec.dirichlet_concentration - 1.0)
        * (wi.ln() + wj.ln() - state.weights[i].ln() - state.weights[j].ln());
    let log_ratio = -(ss - state.ss) / (2.0 * state.sigma2) + prior;
    if uniform.ln() < log_ratio {
        state.weights[i] = wi;
        state.weights[j] = wj;
        state.residual = proposed;
        state.ss = ss;
        true
    } else {
        false
    }
}

/// Draw of the realized average post-period effect given the weights and the error process.
fn effect_draw(
    gap: f64,
    last_residual: f64,
    sigma2: f64,
    spec: &BayesianSyntheticControlSpec<'_>,
    n_post: usize,
    rng: &mut CausalRng,
) -> f64 {
    let rho = spec.ar1_rho;
    let horizon = n_post as f64;
    let carried: f64 = (1..=n_post).map(|k| power(rho, k)).sum();
    let innovation_weight: f64 =
        (1..=n_post).map(|m| (0..=(n_post - m)).map(|j| power(rho, j)).sum::<f64>().powi(2)).sum();
    let mean = last_residual * carried / horizon;
    let sd = (sigma2 * innovation_weight).sqrt() / horizon;
    gap - (mean + sd * standard_normal(rng))
}

struct ChainDraws {
    weights: Vec<Vec<f64>>,
    gaps: Vec<f64>,
    sigma2: Vec<f64>,
    effects: Vec<f64>,
    accepted: usize,
    proposals: usize,
}

fn run_chain(
    panel: &ScPanel,
    spec: &BayesianSyntheticControlSpec<'_>,
    seed: u64,
    kept: usize,
) -> ChainDraws {
    let mut rng = CausalRng::from_seed(seed);
    let donors = panel.donors.len();
    let mut start: Vec<f64> = (0..donors)
        .map(|_| sample_gamma(spec.dirichlet_concentration.max(1.0), 1.0, &mut rng).max(1e-6))
        .collect();
    let total: f64 = start.iter().sum();
    start.iter_mut().for_each(|w| *w /= total);
    let residual = residuals(panel, &start);
    let ss = whitened_ss(&residual, spec.ar1_rho);
    let shape = spec.variance_shape + 0.5 * panel.target_pre.len() as f64;
    let sigma2 = sample_inv_gamma(shape, spec.variance_scale + 0.5 * ss, &mut rng);
    let mut state = ChainState { weights: start, residual, ss, sigma2 };
    let mut out = ChainDraws {
        weights: Vec::with_capacity(kept),
        gaps: Vec::with_capacity(kept),
        sigma2: Vec::with_capacity(kept),
        effects: Vec::with_capacity(kept),
        accepted: 0,
        proposals: 0,
    };
    for sweep in 0..spec.burn_in + kept * THIN {
        for _ in 0..2 * donors {
            out.proposals += 1;
            out.accepted += usize::from(transfer_step(&mut state, panel, spec, &mut rng));
        }
        state.sigma2 = sample_inv_gamma(shape, spec.variance_scale + 0.5 * state.ss, &mut rng);
        if sweep >= spec.burn_in && (sweep - spec.burn_in) % THIN == THIN - 1 {
            let gap = panel.target_post_mean
                - panel.donor_post_mean.iter().zip(&state.weights).map(|(x, w)| x * w).sum::<f64>();
            let last = state.residual.last().copied().unwrap_or(0.0);
            out.effects.push(effect_draw(gap, last, state.sigma2, spec, panel.n_post, &mut rng));
            out.gaps.push(gap);
            out.sigma2.push(state.sigma2);
            out.weights.push(state.weights.clone());
        }
    }
    out
}

fn validate(
    spec: &BayesianSyntheticControlSpec<'_>,
    sampling: DrawSpec,
) -> Result<(), EstimationError> {
    let positive = |v: f64| v.is_finite() && v > 0.0;
    let valid = !spec.treated_unit.is_empty()
        && spec.intervention_period > 0
        && positive(spec.dirichlet_concentration)
        && positive(spec.variance_shape)
        && positive(spec.variance_scale)
        && spec.ar1_rho > -1.0
        && spec.ar1_rho < 1.0
        && sampling.draws >= 400;
    if !valid {
        return Err(EstimationError::stats_msg(
            "Bayesian synthetic control needs a treated unit, a positive intervention period, positive prior parameters, |rho| < 1 and at least 400 draws",
        ));
    }
    Ok(())
}

/// Posterior of the average post-intervention effect on the treated unit.
///
/// `sampling.draws` is the total over four chains. Refuses, naming the next attempt, when the
/// chains disagree (split R-hat above 1.05) or the gap's effective sample size is below 100.
pub fn fit_bayesian_synthetic_control(
    outcome: &[f64],
    units: &[String],
    periods: &[i64],
    spec: &BayesianSyntheticControlSpec<'_>,
    sampling: DrawSpec,
) -> Result<BayesianSyntheticControlEstimate, EstimationError> {
    validate(spec, sampling)?;
    let panel = parse_panel(outcome, units, periods, spec)?;
    let per_chain = sampling.draws / CHAINS;
    let chains: Vec<ChainDraws> = (0..CHAINS)
        .map(|c| {
            let seed = sampling.seed.wrapping_add(CHAIN_SEED_STRIDE.wrapping_mul(c as u64 + 1));
            run_chain(&panel, spec, seed, per_chain)
        })
        .collect();
    let _admitted = crate::util::admit_bound_array::<f64>(CHAINS * per_chain, 4)?;
    let mut samples = Vec::with_capacity(CHAINS * per_chain * 2);
    for chain in &chains {
        for (gap, sigma2) in chain.gaps.iter().zip(&chain.sigma2) {
            samples.extend([*gap, *sigma2]);
        }
    }
    let diagnostics = antecedent_prob::mcmc_summary(&samples, CHAINS, per_chain, 2);
    let rhat_ok = matches!(
        diagnostics.max_rhat.partial_cmp(&MAX_RHAT),
        Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
    );
    if !rhat_ok || diagnostics.min_bulk_ess < MIN_BULK_ESS {
        return Err(crate::util::refuse(
            antecedent_core::reason_code!("posterior_draws_not_converged"),
            "bayesian_synthetic_control_not_converged",
            &format!(
                "split R-hat {:.4} (limit {MAX_RHAT}), bulk ESS {:.1} (minimum {MIN_BULK_ESS}). \
                 Next attempt: more draws or burn-in, or a larger donor pool with a less \
                 concentrated prior",
                diagnostics.max_rhat, diagnostics.min_bulk_ess
            ),
        ));
    }
    Ok(assemble(&panel, spec, &chains, diagnostics.max_rhat, diagnostics.min_bulk_ess))
}

fn assemble(
    panel: &ScPanel,
    spec: &BayesianSyntheticControlSpec<'_>,
    chains: &[ChainDraws],
    max_rhat: f64,
    min_bulk_ess: f64,
) -> BayesianSyntheticControlEstimate {
    let effect_draws: Vec<f64> = chains.iter().flat_map(|c| c.effects.iter().copied()).collect();
    let all_weights: Vec<&Vec<f64>> = chains.iter().flat_map(|c| c.weights.iter()).collect();
    let count = all_weights.len() as f64;
    let weight_means: Vec<f64> = (0..panel.donors.len())
        .map(|j| all_weights.iter().map(|w| w[j]).sum::<f64>() / count)
        .collect();
    let weight_sds: Vec<f64> = weight_means
        .iter()
        .enumerate()
        .map(|(j, m)| {
            (all_weights.iter().map(|w| (w[j] - m).powi(2)).sum::<f64>() / (count - 1.0)).sqrt()
        })
        .collect();
    let fit_residual = residuals(panel, &weight_means);
    let pre_fit_rmse =
        (fit_residual.iter().map(|r| r * r).sum::<f64>() / fit_residual.len() as f64).sqrt();
    let outside = panel
        .target_pre
        .iter()
        .enumerate()
        .filter(|(t, y)| {
            let column = panel.donor_pre.iter().map(|x| x[*t]);
            **y < column.clone().fold(f64::INFINITY, f64::min)
                || **y > column.fold(f64::NEG_INFINITY, f64::max)
        })
        .count();
    let (mean, sd, lower, upper) = summarize(&effect_draws);
    let proposals: usize = chains.iter().map(|c| c.proposals).sum();
    let accepted: usize = chains.iter().map(|c| c.accepted).sum();
    BayesianSyntheticControlEstimate {
        effect_draws,
        mean,
        sd,
        lower,
        upper,
        donor_weight_means: panel.donors.iter().cloned().zip(weight_means).collect(),
        donor_weight_sds: weight_sds,
        pre_fit_rmse,
        pre_periods_outside_donor_range: outside,
        acceptance_rate: accepted as f64 / proposals as f64,
        max_split_rhat: max_rhat,
        min_bulk_ess,
        n_pre_periods: panel.target_pre.len(),
        n_post_periods: panel.n_post,
        model: if spec.ar1_rho.abs() <= 0.0 {
            "simplex synthetic control; Dirichlet donor-weight prior; independent Gaussian pre-fit errors; pairwise-transfer Metropolis with Gibbs variance"
        } else {
            "simplex synthetic control; Dirichlet donor-weight prior; declared-coefficient AR(1) Gaussian pre-fit errors continued into the post period; pairwise-transfer Metropolis with Gibbs variance"
        },
    }
}

#[cfg(test)]
mod tests;
