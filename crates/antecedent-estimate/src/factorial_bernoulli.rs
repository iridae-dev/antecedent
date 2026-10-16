//! Independent-Bernoulli 2×2 factorial Horvitz–Thompson contrasts.
//!
//! Cell order is `[a=0,b=0]`, `[a=0,b=1]`, `[a=1,b=0]`, `[a=1,b=1]`.
//! This is not [`crate::randomized_neyman::factorial_2x2`], which estimates
//! fixed-cell sample means and Neyman variances.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// Horvitz–Thompson cell means and factorial contrasts under independent factors.
#[derive(Clone, Debug, PartialEq)]
pub struct BernoulliFactorial2x2 {
    /// Inverse-probability cell means in cell order.
    pub cell_means: Vec<f64>,
    /// Observed rows in each cell.
    pub cell_support: Vec<usize>,
    /// Main effect of factor A.
    pub factor_a_effect: f64,
    /// Main effect of factor B.
    pub factor_b_effect: f64,
    /// Interaction contrast.
    pub interaction_effect: f64,
    /// Young's inequality bound on the factor-A contrast variance.
    pub factor_a_variance_bound: f64,
    /// Young's inequality bound on the factor-B contrast variance.
    pub factor_b_variance_bound: f64,
    /// Young's inequality bound on the interaction variance.
    pub interaction_variance_bound: f64,
}

/// Finite-row cell means and factorial contrasts for two independent Bernoulli factors.
pub fn estimate_factorial_2x2(
    outcome: &[f64],
    factor_a: &[bool],
    factor_b: &[bool],
    probability_a: &[f64],
    probability_b: &[f64],
) -> Result<BernoulliFactorial2x2, &'static str> {
    let n = outcome.len();
    if n == 0
        || factor_a.len() != n
        || factor_b.len() != n
        || probability_a.len() != n
        || probability_b.len() != n
    {
        return Err(
            "outcome, assignments, and probability vectors must have equal non-zero length",
        );
    }
    if outcome.iter().any(|value| !value.is_finite()) {
        return Err("outcomes must be finite");
    }
    if probability_a.iter().chain(probability_b).any(|p| !p.is_finite() || *p <= 0.0 || *p >= 1.0) {
        return Err("both factor assignment probabilities must be strictly between zero and one");
    }

    let mut sums = [0.0; 4];
    let mut variance_terms = [0.0; 4];
    let mut support = [0_usize; 4];
    for i in 0..n {
        let a = usize::from(factor_a[i]);
        let b = usize::from(factor_b[i]);
        let cell = 2 * a + b;
        let qa = if a == 1 { probability_a[i] } else { 1.0 - probability_a[i] };
        let qb = if b == 1 { probability_b[i] } else { 1.0 - probability_b[i] };
        let q = qa * qb;
        support[cell] += 1;
        sums[cell] += outcome[i] / q;
        variance_terms[cell] += (1.0 - q) * outcome[i].powi(2) / q;
    }
    if support.contains(&0) {
        return Err("factorial positivity failure: every treatment cell needs observed support");
    }
    let means = sums.map(|sum| sum / n as f64);
    let variances = variance_terms.map(|term| term / (n as f64).powi(2));
    let factor_a_weights = [-0.5, -0.5, 0.5, 0.5];
    let factor_b_weights = [-0.5, 0.5, -0.5, 0.5];
    let interaction_weights = [1.0, -1.0, -1.0, 1.0];
    let contrast = |weights: [f64; 4]| {
        means.iter().zip(weights).map(|(mean, weight)| mean * weight).sum::<f64>()
    };
    // Young's inequality gives Var(sum_j w_j X_j) <= 4 sum_j w_j^2 Var(X_j).
    let variance_bound = |weights: [f64; 4]| {
        4.0 * variances
            .iter()
            .zip(weights)
            .map(|(variance, weight)| variance * weight.powi(2))
            .sum::<f64>()
    };
    Ok(BernoulliFactorial2x2 {
        cell_means: means.to_vec(),
        cell_support: support.to_vec(),
        factor_a_effect: contrast(factor_a_weights),
        factor_b_effect: contrast(factor_b_weights),
        interaction_effect: contrast(interaction_weights),
        factor_a_variance_bound: variance_bound(factor_a_weights),
        factor_b_variance_bound: variance_bound(factor_b_weights),
        interaction_variance_bound: variance_bound(interaction_weights),
    })
}
