//! Bayesian posterior for a kernel-smoothed group-dose policy value.
//!
//! The target is the one of [`crate::continuous_dose`]: for fixed baseline groups `g` with
//! observed fractions `pi_g` and a fixed group-to-dose rule `t_g`, the value
//! `V = sum_g pi_g * integral K_h(d - t_g) m_g(d) dd`, where `m_g(d) = E[Y | D = d, group g]`
//! and `K_h` is the Epanechnikov density of half-width `h`. It is the average response over
//! the prespecified window, not `do(D = d)`. The frequentist cell reaches it with inverse
//! known-density weights; here it is a functional of the fitted curve.
//!
//! The likelihood is stated: within a group, `y_i = m_g(d_i) + e_i` with independent
//! `N(0, sigma^2)` errors, and `m_g` a polynomial in the dose rescaled to `[-1, 1]` on the
//! declared support, with the same explicit coefficient prior as the dose-grid posterior
//! (`beta_j | sigma^2 ~ N(0, sigma^2 s_j^2)`; the scale vector is the smoothing prior).
//! Groups are fitted separately and are independent a posteriori. Conditional exchangeability
//! within groups and local positivity are the identification premises, as in the
//! frequentist cell; the dose density does not enter a likelihood for `Y | D`, so it is not
//! an input. The group fractions are held at their observed values, so the posterior does not
//! carry sampling uncertainty of the group mix.
//!
//! The policy and the reference are read from the same coefficient draws, so the incremental
//! value is a paired draw. Windows must lie inside the declared support (no boundary
//! kernels) and hold at least `min_local_support` observed rows in every group.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use antecedent_stats::special::gauss_legendre;
use antecedent_stats::{QuantileRule, equal_tail_interval_sorted};

use crate::continuous_dose::DoseRules;
use crate::dose_basis_regression::{BasisPosterior, ResidualPrior, fit_basis_regression};
use crate::error::EstimationError;

/// Gauss-Legendre nodes of the window quadrature: exact for polynomial curves of degree 29.
const WINDOW_NODES: usize = 16;
/// Most polynomial coefficients per group, keeping the window quadrature exact.
const MAX_GROUP_WIDTH: usize = 30;

/// Rows of the policy evaluation, aligned by row.
#[derive(Clone, Copy, Debug)]
pub struct DoseResponseRows<'a> {
    /// Outcome per row.
    pub outcome: &'a [f64],
    /// Observed dose per row.
    pub dose: &'a [f64],
    /// Baseline group label per row.
    pub groups: &'a [String],
}

/// The declared model of the group curves.
#[derive(Clone, Copy, Debug)]
pub struct DoseCurvePrior<'a> {
    /// Declared dose support `(lo, hi)` the polynomial is rescaled on.
    pub support: (f64, f64),
    /// Prior scale of every polynomial coefficient.
    pub coefficient_scales: &'a [f64],
    /// Prior of the residual variance, per group.
    pub residual: ResidualPrior,
}

/// The kernel window and the posterior sample size.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolicyWindow {
    /// Epanechnikov half-width, in dose units.
    pub bandwidth: f64,
    /// Fewest rows any group window must hold (at least two).
    pub min_local_support: usize,
    /// Posterior draws.
    pub draws: usize,
    /// Seed of the draws.
    pub seed: u64,
}

/// Posterior of the policy value, the reference value and their paired difference.
#[derive(Clone, Debug, PartialEq)]
pub struct DosePolicyPosterior {
    /// Observed group fractions, in group order.
    pub group_fractions: Vec<(String, f64)>,
    /// Posterior mean of the policy value (from the draws).
    pub policy_mean: f64,
    /// Posterior mean of the reference value.
    pub reference_mean: f64,
    /// Posterior mean of the policy-minus-reference value.
    pub incremental_mean: f64,
    /// Equal-tailed 95% credible interval of the policy value.
    pub policy_interval_95: [f64; 2],
    /// Equal-tailed 95% credible interval of the reference value.
    pub reference_interval_95: [f64; 2],
    /// Equal-tailed 95% credible interval of the incremental value.
    pub incremental_interval_95: [f64; 2],
    /// Posterior probability that the incremental value is positive.
    pub probability_incremental_positive: f64,
    /// Exact posterior mean of the policy, reference and incremental values (known variance).
    pub exact_means: Option<[f64; 3]>,
    /// Exact posterior variance of the policy, reference and incremental values.
    pub exact_variances: Option<[f64; 3]>,
    /// Smallest number of rows in any group window.
    pub minimum_local_rows: usize,
    /// Number of draws.
    pub draws: usize,
    /// Draws of the policy value, in draw order.
    pub policy_draws: Vec<f64>,
    /// Draws of the reference value; draw `k` shares its coefficients with the policy draw.
    pub reference_draws: Vec<f64>,
}

/// Posterior of a fixed group-dose policy value under per-group polynomial dose responses.
///
/// # Errors
///
/// Misaligned or non-finite rows, a rule that does not name every observed group exactly
/// once, a window outside the declared support, thin local support, an invalid prior or draw
/// count.
#[allow(
    clippy::too_many_lines,
    reason = "validate, then per group the window functionals and the fit feed the paired draws and exact moments"
)]
pub fn fixed_dose_policy_posterior(
    rows: &DoseResponseRows<'_>,
    rules: &DoseRules<'_>,
    curve: &DoseCurvePrior<'_>,
    window: &PolicyWindow,
) -> Result<DosePolicyPosterior, EstimationError> {
    let n = rows.outcome.len();
    let (lo, hi) = curve.support;
    let width = curve.coefficient_scales.len();
    if n < 2
        || rows.dose.len() != n
        || rows.groups.len() != n
        || rows.outcome.iter().chain(rows.dose).any(|v| !v.is_finite())
        || rows.groups.iter().any(String::is_empty)
        || !window.bandwidth.is_finite()
        || window.bandwidth <= 0.0
        || window.min_local_support < 2
        || !lo.is_finite()
        || !hi.is_finite()
        || lo >= hi
        || width == 0
        || width > MAX_GROUP_WIDTH
        || rows.dose.iter().any(|d| *d < lo || *d > hi)
    {
        return Err(EstimationError::data_msg(
            "a fixed dose policy posterior requires aligned finite rows inside the declared support, a positive bandwidth, support of at least two and a polynomial of at most 30 coefficients",
        ));
    }
    let observed: BTreeSet<&str> = rows.groups.iter().map(String::as_str).collect();
    let policy = mapping(rules.policy, &observed)?;
    let reference = mapping(rules.reference, &observed)?;
    let (nodes, weights) = gauss_legendre(WINDOW_NODES);
    let h = window.bandwidth;
    let mut minimum_local_rows = usize::MAX;
    let mut fractions = Vec::new();
    let _admitted = crate::util::admit_bound_array::<f64>(window.draws, 2)?;
    let mut policy_draws = vec![0.0; window.draws];
    let mut reference_draws = vec![0.0; window.draws];
    let mut exact = Exact::default();
    for (index, group) in observed.iter().enumerate() {
        let members: Vec<usize> = (0..n).filter(|i| rows.groups[*i] == *group).collect();
        #[allow(clippy::cast_precision_loss, reason = "row counts are far below 2^52")]
        let fraction = members.len() as f64 / n as f64;
        fractions.push(((*group).to_owned(), fraction));
        let mut functionals = Vec::new();
        for target in [policy[*group], reference[*group]] {
            if target - h < lo || target + h > hi {
                return Err(EstimationError::data_msg(format!(
                    "the window of group {group:?} at dose {target} leaves the declared support; boundary kernels are not supported"
                )));
            }
            let local = members.iter().filter(|i| (rows.dose[**i] - target).abs() < h).count();
            if local < window.min_local_support {
                return Err(EstimationError::data_msg(format!(
                    "dose policy local support failure: group {group:?} at dose {target} has {local} rows, below min_local_support={}",
                    window.min_local_support
                )));
            }
            minimum_local_rows = minimum_local_rows.min(local);
            functionals.push(window_functional(&nodes, &weights, target, h, curve));
        }
        let dose: Vec<f64> = members.iter().map(|i| rows.dose[*i]).collect();
        let outcome: Vec<f64> = members.iter().map(|i| rows.outcome[*i]).collect();
        let fit = fit_group(&dose, &outcome, curve, window, index)?;
        accumulate(&fit, &functionals, fraction, &mut policy_draws, &mut reference_draws);
        exact.add(&fit, &functionals, fraction);
    }
    let incremental: Vec<f64> =
        policy_draws.iter().zip(&reference_draws).map(|(p, r)| p - r).collect();
    #[allow(clippy::cast_precision_loss, reason = "draw counts are far below 2^52")]
    let count = window.draws as f64;
    let mean = |values: &[f64]| values.iter().sum::<f64>() / count;
    #[allow(clippy::cast_precision_loss, reason = "draw counts are far below 2^52")]
    let positive = incremental.iter().filter(|v| **v > 0.0).count() as f64;
    Ok(DosePolicyPosterior {
        group_fractions: fractions,
        policy_mean: mean(&policy_draws),
        reference_mean: mean(&reference_draws),
        incremental_mean: mean(&incremental),
        policy_interval_95: interval_95(&policy_draws),
        reference_interval_95: interval_95(&reference_draws),
        incremental_interval_95: interval_95(&incremental),
        probability_incremental_positive: positive / count,
        exact_means: exact.known.then_some(exact.means),
        exact_variances: exact.known.then_some(exact.variances),
        minimum_local_rows,
        draws: window.draws,
        policy_draws,
        reference_draws,
    })
}

fn mapping<'a>(
    rule: &'a [(String, f64)],
    observed: &BTreeSet<&str>,
) -> Result<BTreeMap<&'a str, f64>, EstimationError> {
    let message = "dose rule must name every observed group exactly once with a finite target";
    let mut doses = BTreeMap::new();
    for (group, target) in rule {
        if !target.is_finite()
            || !observed.contains(group.as_str())
            || doses.insert(group.as_str(), *target).is_some()
        {
            return Err(EstimationError::data_msg(message));
        }
    }
    if doses.len() != observed.len() {
        return Err(EstimationError::data_msg(message));
    }
    Ok(doses)
}

fn rescale(dose: f64, lo: f64, hi: f64) -> f64 {
    (2.0 * dose - lo - hi) / (hi - lo)
}

/// `sum_k w_k K(u_k) phi(z(t + h u_k))`: the window average of every basis function.
fn window_functional(
    nodes: &[f64],
    weights: &[f64],
    target: f64,
    h: f64,
    curve: &DoseCurvePrior<'_>,
) -> Vec<f64> {
    let (lo, hi) = curve.support;
    let width = curve.coefficient_scales.len();
    let mut out = vec![0.0; width];
    for (u, w) in nodes.iter().zip(weights) {
        let kernel = 0.75 * (1.0 - u * u);
        let z = rescale(target + h * u, lo, hi);
        let mut power = 1.0;
        for slot in &mut out {
            *slot += w * kernel * power;
            power *= z;
        }
    }
    out
}

fn fit_group(
    dose: &[f64],
    outcome: &[f64],
    curve: &DoseCurvePrior<'_>,
    window: &PolicyWindow,
    index: usize,
) -> Result<BasisPosterior, EstimationError> {
    let (lo, hi) = curve.support;
    let rows = dose.len();
    let width = curve.coefficient_scales.len();
    let mut design = vec![0.0; rows * width];
    for (i, d) in dose.iter().enumerate() {
        let z = rescale(*d, lo, hi);
        let mut power = 1.0;
        for j in 0..width {
            design[j * rows + i] = power;
            power *= z;
        }
    }
    fit_basis_regression(
        &design,
        rows,
        outcome,
        curve.coefficient_scales,
        curve.residual,
        window.draws,
        window.seed.wrapping_add((index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
    )
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

/// Add `fraction * functional . beta_k` to the policy and reference draws.
fn accumulate(
    fit: &BasisPosterior,
    functionals: &[Vec<f64>],
    fraction: f64,
    policy: &mut [f64],
    reference: &mut [f64],
) {
    for (k, draw) in fit.coefficients.chunks(fit.width).enumerate() {
        if let (Some(p), Some(r)) = (policy.get_mut(k), reference.get_mut(k)) {
            *p += fraction * dot(&functionals[0], draw);
            *r += fraction * dot(&functionals[1], draw);
        }
    }
}

/// Exact Gaussian moments of the three values, accumulated across independent groups.
struct Exact {
    means: [f64; 3],
    variances: [f64; 3],
    known: bool,
}

impl Default for Exact {
    fn default() -> Self {
        Self { means: [0.0; 3], variances: [0.0; 3], known: true }
    }
}

impl Exact {
    fn add(&mut self, fit: &BasisPosterior, functionals: &[Vec<f64>], fraction: f64) {
        let Some(covariance) = fit.covariance.as_ref() else {
            self.known = false;
            return;
        };
        let difference: Vec<f64> =
            functionals[0].iter().zip(&functionals[1]).map(|(a, b)| a - b).collect();
        let rows = [&functionals[0], &functionals[1], &difference];
        for (slot, row) in rows.iter().enumerate() {
            self.means[slot] += fraction * dot(row, &fit.mean);
            let spread: Vec<f64> = (0..fit.width)
                .map(|i| (0..fit.width).map(|j| covariance[i * fit.width + j] * row[j]).sum())
                .collect();
            self.variances[slot] += fraction * fraction * dot(&spread, row);
        }
    }
}

fn interval_95(values: &[f64]) -> [f64; 2] {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let (lower, upper) = equal_tail_interval_sorted(&sorted, 0.95, QuantileRule::ExchangeableRank);
    [lower, upper]
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCALES: [f64; 3] = [3.0, 2.0, 1.0];

    fn rows() -> (Vec<f64>, Vec<f64>, Vec<String>) {
        let mut dose = vec![0.2, 0.6, 1.1, 1.5, 1.9, 2.4, 2.8, 3.3];
        let mut outcome = vec![1.0, 1.6, 2.2, 2.5, 2.4, 2.0, 1.7, 1.1];
        let mut groups = vec!["A".to_owned(); 8];
        dose.extend([0.4, 0.9, 1.3, 1.8, 2.2, 2.6, 3.0, 3.7]);
        outcome.extend([0.5, 0.9, 1.0, 1.4, 1.9, 2.0, 2.4, 2.9]);
        groups.extend(vec!["B".to_owned(); 8]);
        (dose, outcome, groups)
    }

    type Rule = Vec<(String, f64)>;

    fn rules() -> (Rule, Rule) {
        (
            vec![("A".to_owned(), 1.5), ("B".to_owned(), 2.5)],
            vec![("A".to_owned(), 2.0), ("B".to_owned(), 1.0)],
        )
    }

    fn evaluate(
        policy: &[(String, f64)],
        reference: &[(String, f64)],
        bandwidth: f64,
    ) -> Result<DosePolicyPosterior, EstimationError> {
        let (dose, outcome, groups) = rows();
        fixed_dose_policy_posterior(
            &DoseResponseRows { outcome: &outcome, dose: &dose, groups: &groups },
            &DoseRules { policy, reference },
            &DoseCurvePrior {
                support: (0.0, 4.0),
                coefficient_scales: &SCALES,
                residual: ResidualPrior::KnownVariance(0.25),
            },
            &PolicyWindow { bandwidth, min_local_support: 2, draws: 40_000, seed: 5 },
        )
    }

    #[test]
    fn policy_posterior_matches_the_per_group_conjugate_reference() {
        // Reference: per-group ridge posteriors from numpy, the window average integrated
        // with scipy adaptive quadrature (not the Gauss-Legendre rule used here).
        let (policy, reference) = rules();
        let posterior = evaluate(&policy, &reference, 1.0).unwrap();
        let means = posterior.exact_means.unwrap();
        let variances = posterior.exact_variances.unwrap();
        for (got, want) in means.iter().zip([1.915_595_66, 1.431_440_23, 0.484_155_43]) {
            assert!((got - want).abs() < 1e-7, "{got} != {want}");
        }
        for (got, want) in variances.iter().zip([0.019_981_54, 0.023_910_98, 0.018_089_61]) {
            assert!((got - want).abs() < 1e-7, "{got} != {want}");
        }
        assert!((posterior.policy_mean - means[0]).abs() < 0.01);
        assert!((posterior.reference_mean - means[1]).abs() < 0.01);
        assert!((posterior.incremental_mean - means[2]).abs() < 0.01);
        let [lower, upper] = posterior.incremental_interval_95;
        let half = 1.96 * variances[2].sqrt();
        assert!(
            (lower - (means[2] - half)).abs() < 0.015 && (upper - (means[2] + half)).abs() < 0.015
        );
        assert!(posterior.probability_incremental_positive > 0.99);
        assert_eq!(posterior.minimum_local_rows, 4);
        assert_eq!(posterior.group_fractions.len(), 2);
    }

    #[test]
    fn windows_off_the_support_or_thin_groups_and_bad_rules_are_refused() {
        let (policy, reference) = rules();
        // Bandwidth 2 puts group A's reference window at [0, 4] and B's at [-1, 3].
        let boundary = evaluate(&policy, &reference, 2.0).unwrap_err();
        assert!(boundary.to_string().contains("boundary kernels"));
        // A very narrow window holds fewer than two rows.
        let thin = evaluate(&policy, &reference, 0.12).unwrap_err();
        assert!(thin.to_string().contains("local support failure"));
        let missing = vec![("A".to_owned(), 1.5)];
        assert!(evaluate(&missing, &reference, 1.0).is_err());
        let duplicate = vec![("A".to_owned(), 1.5), ("A".to_owned(), 2.0)];
        assert!(evaluate(&duplicate, &reference, 1.0).is_err());
    }
}
