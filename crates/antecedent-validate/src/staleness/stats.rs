//! Test statistics of the staleness diagnostic.
//!
//! Every function is deterministic: sums run in a fixed order and the permutation test draws
//! from keyed streams ([`antecedent_core::RngKey`]) seeded by the policy, so the same inputs
//! give the same bits on any thread count.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{ChainId, FoldId, ReductionPolicy, ReplicateId, RngKey, SemanticOperation};
use antecedent_kernels::{norm_sf, shuffle, sigmoid};
use antecedent_stats::{chi_square_sf, chol_solve, cholesky_spd};

use super::policy::Multiplicity;

/// Operation tag of the baseline subsample stream.
pub(crate) const BASELINE_SUBSAMPLE_OP: SemanticOperation =
    SemanticOperation::new(0x57A1_E000_0000_0001);
/// Operation tag of the new-data subsample stream.
pub(crate) const NEW_SUBSAMPLE_OP: SemanticOperation =
    SemanticOperation::new(0x57A1_E000_0000_0002);
/// Operation tag of the energy permutation streams.
pub(crate) const ENERGY_PERMUTATION_OP: SemanticOperation =
    SemanticOperation::new(0x57A1_E000_0000_0003);

/// Open the keyed stream of `operation`, replicate `replicate`, for `seed`.
pub(crate) fn stream(
    operation: SemanticOperation,
    replicate: u64,
    seed: u64,
) -> antecedent_core::CausalRng {
    RngKey::new(
        operation,
        FoldId::new(0),
        ChainId::new(0),
        ReplicateId::new(replicate),
        ReductionPolicy::Strict,
    )
    .open(seed)
}

/// Sample variance from a count and a centered sum of squares (`0` below two rows).
pub(crate) fn variance(n: u64, m2: f64) -> f64 {
    if n < 2 { 0.0 } else { (m2 / (n - 1) as f64).max(0.0) }
}

/// Welch two-sample test of equal means: `(t, two-sided p)`. `None` below two rows a side.
#[allow(clippy::float_cmp, reason = "two constant samples are equal only when their means are")]
pub(crate) fn welch(
    n0: u64,
    mean0: f64,
    var0: f64,
    n1: u64,
    mean1: f64,
    var1: f64,
) -> Option<(f64, f64)> {
    if n0 < 2 || n1 < 2 {
        return None;
    }
    let (a, b) = (var0 / n0 as f64, var1 / n1 as f64);
    let se2 = a + b;
    if !(se2.is_finite() && se2 > 0.0) {
        // Two constant samples: equal means are no shift, unequal ones are a certain shift.
        return Some(if mean0 == mean1 { (0.0, 1.0) } else { (f64::INFINITY, 0.0) });
    }
    let t = (mean1 - mean0) / se2.sqrt();
    let df = se2 * se2 / (a * a / (n0 - 1) as f64 + b * b / (n1 - 1) as f64);
    let p = if df.is_finite() && df > 0.0 {
        2.0 * antecedent_stats::student_t_sf(t.abs(), df)
    } else {
        2.0 * norm_sf(t.abs())
    };
    Some((t, p.clamp(0.0, 1.0)))
}

/// Standardized mean difference with the average of the two variances.
#[allow(clippy::float_cmp, reason = "two constant samples are equal only when their means are")]
pub(crate) fn smd(mean0: f64, var0: f64, mean1: f64, var1: f64) -> f64 {
    let pooled = (0.5 * (var0 + var1)).sqrt();
    if pooled > 0.0 && pooled.is_finite() {
        (mean1 - mean0) / pooled
    } else if mean0 == mean1 {
        0.0
    } else {
        f64::INFINITY.copysign(mean1 - mean0)
    }
}

/// Chi-square test of homogeneity of two count vectors: `(statistic, df, p)`.
/// Cells empty on both sides are dropped; `None` when fewer than two cells remain.
pub(crate) fn chi_square_homogeneity(old: &[u64], new: &[u64]) -> Option<(f64, usize, f64)> {
    let n0: u64 = old.iter().sum();
    let n1: u64 = new.iter().sum();
    if n0 == 0 || n1 == 0 || old.len() != new.len() {
        return None;
    }
    let total = (n0 + n1) as f64;
    let (f0, f1) = (n0 as f64 / total, n1 as f64 / total);
    let mut statistic = 0.0;
    let mut cells = 0usize;
    for (&o, &n) in old.iter().zip(new) {
        if o + n == 0 {
            continue;
        }
        cells += 1;
        let t = (o + n) as f64;
        let (e0, e1) = (t * f0, t * f1);
        statistic += (o as f64 - e0).powi(2) / e0 + (n as f64 - e1).powi(2) / e1;
    }
    if cells < 2 {
        return None;
    }
    let df = cells - 1;
    Some((statistic, df, chi_square_sf(statistic, df)))
}

/// Population stability index of two count vectors (a symmetric binned divergence of the
/// density ratio), with a half-count added to every cell, and the largest cell density ratio.
pub(crate) fn population_stability(old: &[u64], new: &[u64]) -> (f64, f64) {
    let cells = old.len().max(1) as f64;
    let n0 = old.iter().sum::<u64>() as f64 + 0.5 * cells;
    let n1 = new.iter().sum::<u64>() as f64 + 0.5 * cells;
    let mut psi = 0.0;
    let mut max_ratio = 0.0_f64;
    for (&o, &n) in old.iter().zip(new) {
        let p = (o as f64 + 0.5) / n0;
        let q = (n as f64 + 0.5) / n1;
        psi += (q - p) * (q / p).ln();
        max_ratio = max_ratio.max(q / p);
    }
    (psi, max_ratio)
}

/// Survival function of the F distribution with `(d1, d2)` degrees of freedom.
pub(crate) fn f_sf(f: f64, d1: f64, d2: f64) -> f64 {
    if f.is_nan() || f <= 0.0 {
        return 1.0;
    }
    if f.is_infinite() {
        return 0.0;
    }
    let x = d2 / (d2 + d1 * f);
    antecedent_stats::adaptive_inference::regularized_incomplete_beta(x, 0.5 * d2, 0.5 * d1)
        .map_or(f64::NAN, |value| value.clamp(0.0, 1.0))
}

/// `P(sup |B(t)| > c)` for a Brownian bridge `B` (the Kolmogorov distribution).
pub(crate) fn kolmogorov_sf(c: f64) -> f64 {
    if c.is_nan() || c < 0.2 {
        return 1.0;
    }
    let mut total = 0.0;
    for i in 1..=100_u32 {
        let k = f64::from(i);
        let term = (-2.0 * k * k * c * c).exp();
        total += if i % 2 == 1 { term } else { -term };
        if term < 1e-16 {
            break;
        }
    }
    (2.0 * total).clamp(0.0, 1.0)
}

/// Probability that at least `k` of `n1` new values lie beyond the extreme of `n0` exchangeable
/// old values: the top `k` of the pooled sample are all new.
pub(crate) fn exceedance_p(n0: u64, n1: u64, k: u64) -> f64 {
    if k == 0 {
        return 1.0;
    }
    if k > n1 {
        return 0.0;
    }
    let mut log_p = 0.0;
    for i in 0..k {
        log_p += ((n1 - i) as f64).ln() - ((n0 + n1 - i) as f64).ln();
    }
    log_p.exp()
}

/// Two-proportion z test: `(z, two-sided p)`. `None` when neither sample has an event or a
/// non-event (no variance to test against).
pub(crate) fn two_proportion(x0: u64, n0: u64, x1: u64, n1: u64) -> Option<(f64, f64)> {
    if n0 == 0 || n1 == 0 {
        return None;
    }
    let pooled = (x0 + x1) as f64 / (n0 + n1) as f64;
    let se = (pooled * (1.0 - pooled) * (1.0 / n0 as f64 + 1.0 / n1 as f64)).sqrt();
    if se.is_nan() || se <= 0.0 {
        return None;
    }
    let z = (x1 as f64 / n1 as f64 - x0 as f64 / n0 as f64) / se;
    Some((z, (2.0 * norm_sf(z.abs())).clamp(0.0, 1.0)))
}

/// Multiplicity-adjusted p-values in input order.
pub(crate) fn adjust(p_values: &[f64], multiplicity: Multiplicity) -> Vec<f64> {
    match multiplicity {
        Multiplicity::Holm => antecedent_stats::holm(p_values),
        Multiplicity::BenjaminiHochberg => antecedent_stats::benjamini_hochberg(p_values),
    }
}

/// Energy two-sample statistic of the pooled standardized points (`old` rows first), and its
/// permutation p-value over `permutations` keyed label shuffles.
///
/// `distances` is caller-leased scratch of at least `m²` entries, `m` the pooled row count.
pub(crate) fn energy_permutation(
    old: &[f64],
    new: &[f64],
    dim: usize,
    permutations: u32,
    seed: u64,
    distances: &mut Vec<f64>,
) -> Option<(f64, f64)> {
    if dim == 0 {
        return None;
    }
    let (m0, m1) = (old.len() / dim, new.len() / dim);
    if m0 < 2 || m1 < 2 {
        return None;
    }
    let m = m0 + m1;
    let point = |i: usize| {
        if i < m0 { &old[i * dim..(i + 1) * dim] } else { &new[(i - m0) * dim..(i - m0 + 1) * dim] }
    };
    distances.clear();
    distances.resize(m * m, 0.0);
    for i in 0..m {
        for j in (i + 1)..m {
            let d =
                point(i).iter().zip(point(j)).map(|(a, b)| (a - b) * (a - b)).sum::<f64>().sqrt();
            distances[i * m + j] = d;
            distances[j * m + i] = d;
        }
    }
    let total: f64 = (0..m)
        .flat_map(|i| ((i + 1)..m).map(move |j| (i, j)))
        .map(|(i, j)| distances[i * m + j])
        .sum();
    let statistic = |labels: &[bool]| -> f64 {
        let (mut within_old, mut within_new) = (0.0, 0.0);
        for i in 0..m {
            for j in (i + 1)..m {
                match (labels[i], labels[j]) {
                    (false, false) => within_old += distances[i * m + j],
                    (true, true) => within_new += distances[i * m + j],
                    _ => {}
                }
            }
        }
        let between = total - within_old - within_new;
        let (a, b) = (m0 as f64, m1 as f64);
        let energy =
            2.0 * between / (a * b) - 2.0 * within_old / (a * a) - 2.0 * within_new / (b * b);
        a * b / (a + b) * energy
    };
    let mut labels: Vec<bool> = (0..m).map(|i| i >= m0).collect();
    let observed = statistic(&labels);
    let tolerance = 1e-12 * observed.abs().max(1.0);
    let mut at_least = 0u64;
    for replicate in 0..permutations {
        labels.iter_mut().enumerate().for_each(|(i, label)| *label = i >= m0);
        let mut rng = stream(ENERGY_PERMUTATION_OP, u64::from(replicate), seed);
        shuffle(&mut rng, &mut labels);
        if statistic(&labels) >= observed - tolerance {
            at_least += 1;
        }
    }
    let p = (1 + at_least) as f64 / (1 + u64::from(permutations)) as f64;
    Some((observed, p))
}

/// Ridge-penalized logistic regression of `t` on `[1, x…]` by Newton iterations.
///
/// `columns` are the covariates (already standardized), all complete over the same rows.
/// The ridge `lambda` (not applied to the intercept) keeps the fit finite under separation,
/// where the fitted propensities move to the boundary: that is the positivity signal itself.
pub(crate) fn logistic_ridge(columns: &[Vec<f64>], t: &[f64], lambda: f64) -> Option<Vec<f64>> {
    let p = columns.len() + 1;
    let n = t.len();
    if n == 0 || columns.iter().any(|c| c.len() != n) {
        return None;
    }
    let mut beta = vec![0.0; p];
    let mut x = vec![0.0; p];
    for _ in 0..60 {
        let mut gradient: Vec<f64> =
            beta.iter().enumerate().map(|(j, b)| if j == 0 { 0.0 } else { -lambda * b }).collect();
        let mut hessian = vec![0.0; p * p];
        for j in 1..p {
            hessian[j * p + j] = lambda;
        }
        for (i, &ti) in t.iter().enumerate() {
            x[0] = 1.0;
            for (slot, column) in x[1..].iter_mut().zip(columns) {
                *slot = column[i];
            }
            let eta: f64 = x.iter().zip(&beta).map(|(a, b)| a * b).sum::<f64>().clamp(-30.0, 30.0);
            let mu = sigmoid(eta);
            let w = mu * (1.0 - mu);
            let residual = ti - mu;
            for (a, &xa) in x.iter().enumerate() {
                gradient[a] += residual * xa;
                for (b, &xb) in x.iter().enumerate().skip(a) {
                    hessian[a * p + b] += w * xa * xb;
                }
            }
        }
        for a in 0..p {
            for b in 0..a {
                hessian[a * p + b] = hessian[b * p + a];
            }
        }
        let chol = cholesky_spd(&hessian, p)?;
        let step = chol_solve(&chol, p, &gradient)?;
        let largest = step.iter().fold(0.0_f64, |m, s| m.max(s.abs()));
        for (b, s) in beta.iter_mut().zip(&step) {
            *b += s;
        }
        if !largest.is_finite() {
            return None;
        }
        if largest < 1e-9 {
            break;
        }
    }
    Some(beta)
}

/// Fitted propensities `σ(xᵀβ)` of [`logistic_ridge`] coefficients.
pub(crate) fn propensities(columns: &[Vec<f64>], beta: &[f64], n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let eta = beta[0] + columns.iter().zip(&beta[1..]).map(|(c, b)| c[i] * b).sum::<f64>();
            sigmoid(eta.clamp(-30.0, 30.0))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closed_forms() {
        assert!((kolmogorov_sf(1.358) - 0.05).abs() < 2e-3);
        assert!((f_sf(3.0, 2.0, 1e6) - (-3.0_f64).exp()).abs() < 1e-3);
        // Top 1 of 10 old and 10 new being new: 10 / 20.
        assert!((exceedance_p(10, 10, 1) - 0.5).abs() < 1e-12);
        assert!((exceedance_p(10, 10, 2) - 0.5 * 9.0 / 19.0).abs() < 1e-12);
        let (stat, df, p) = chi_square_homogeneity(&[10, 10], &[10, 10]).unwrap();
        assert!(stat.abs() < 1e-12);
        assert_eq!(df, 1);
        assert!((p - 1.0).abs() < 1e-12);
        let (psi, _) = population_stability(&[50, 50], &[50, 50]);
        assert!(psi.abs() < 1e-12);
        assert!(welch(5, 0.0, 1.0, 5, 0.0, 1.0).unwrap().1 > 0.99);
    }

    #[test]
    fn energy_permutation_is_reproducible_and_detects_a_shift() {
        let old: Vec<f64> = (0..40_u32).map(|i| f64::from(i % 7) / 7.0).collect();
        let shifted: Vec<f64> = (0..40_u32).map(|i| 3.0 + f64::from(i % 5) / 5.0).collect();
        let mut scratch = Vec::new();
        let first = energy_permutation(&old, &shifted, 1, 99, 7, &mut scratch).unwrap();
        let again = energy_permutation(&old, &shifted, 1, 99, 7, &mut scratch).unwrap();
        assert_eq!(first.0.to_bits(), again.0.to_bits());
        assert_eq!(first.1.to_bits(), again.1.to_bits());
        assert!(first.1 <= 0.011, "{first:?}");
    }

    #[test]
    fn logistic_ridge_recovers_a_balanced_intercept() {
        let x = vec![(0..100_u32).map(|i| f64::from((i / 2) % 5) - 2.0).collect::<Vec<_>>()];
        let t: Vec<f64> = (0..100_u32).map(|i| f64::from(i % 2)).collect();
        let beta = logistic_ridge(&x, &t, 1e-4).unwrap();
        let fitted = propensities(&x, &beta, 100);
        assert!(fitted.iter().all(|e| (e - 0.5).abs() < 0.05), "{beta:?}");
    }
}
