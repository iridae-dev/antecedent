//! Split-R-hat and bulk ESS for multi-chain MCMC.
//!
//! Chain layout: `samples[(chain * n_draws + draw) * n_params + param]`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::needless_range_loop)]

/// Maximum split-Ř across parameters (Vehtari / Gelman–Rubin style).
#[must_use]
pub fn max_split_rhat(samples: &[f64], n_chains: usize, n_draws: usize, n_params: usize) -> f64 {
    if n_chains < 2 || n_draws < 4 || n_params == 0 {
        return f64::INFINITY;
    }
    let mut max_r = 0.0_f64;
    for p in 0..n_params {
        let r = split_rhat_one(samples, n_chains, n_draws, n_params, p);
        if r.is_finite() {
            max_r = max_r.max(r);
        } else {
            return f64::INFINITY;
        }
    }
    max_r
}

/// Minimum bulk ESS across parameters.
#[must_use]
pub fn min_bulk_ess(samples: &[f64], n_chains: usize, n_draws: usize, n_params: usize) -> f64 {
    if n_chains == 0 || n_draws < 4 || n_params == 0 {
        return 0.0;
    }
    let mut min_ess = f64::INFINITY;
    for p in 0..n_params {
        let e = bulk_ess_one(samples, n_chains, n_draws, n_params, p);
        min_ess = min_ess.min(e);
    }
    if min_ess.is_finite() { min_ess } else { 0.0 }
}

fn split_rhat_one(
    samples: &[f64],
    n_chains: usize,
    n_draws: usize,
    n_params: usize,
    param: usize,
) -> f64 {
    // Split each chain in half → m = 2 * n_chains segments of length n = n_draws/2.
    let half = n_draws / 2;
    if half < 2 {
        return f64::INFINITY;
    }
    let m = n_chains * 2;
    let n = half as f64;
    let mut means = vec![0.0; m];
    let mut vars = vec![0.0; m];
    for c in 0..n_chains {
        for split in 0..2 {
            let seg = c * 2 + split;
            let start = split * half;
            let mut mean = 0.0;
            for d in 0..half {
                mean += sample_at(samples, c, start + d, n_draws, n_params, param);
            }
            mean /= n;
            means[seg] = mean;
            let mut var = 0.0;
            for d in 0..half {
                let v = sample_at(samples, c, start + d, n_draws, n_params, param) - mean;
                var += v * v;
            }
            vars[seg] = var / (n - 1.0);
        }
    }
    let mut w = 0.0;
    let mut grand = 0.0;
    for i in 0..m {
        w += vars[i];
        grand += means[i];
    }
    w /= m as f64;
    grand /= m as f64;
    let mut b = 0.0;
    for i in 0..m {
        let d = means[i] - grand;
        b += d * d;
    }
    b = n * b / (m as f64 - 1.0);
    if !(w > 0.0) {
        return if b > 0.0 { f64::INFINITY } else { 1.0 };
    }
    let var_hat = ((n - 1.0) / n) * w + b / n;
    (var_hat / w).sqrt()
}

fn bulk_ess_one(
    samples: &[f64],
    n_chains: usize,
    n_draws: usize,
    n_params: usize,
    param: usize,
) -> f64 {
    // Pool chains; estimate lag-1 autocorrelation and ESS ≈ N / (1 + 2 Σ ρ).
    let n_total = n_chains * n_draws;
    let mut mean = 0.0;
    for c in 0..n_chains {
        for d in 0..n_draws {
            mean += sample_at(samples, c, d, n_draws, n_params, param);
        }
    }
    mean /= n_total as f64;
    let mut var = 0.0;
    for c in 0..n_chains {
        for d in 0..n_draws {
            let v = sample_at(samples, c, d, n_draws, n_params, param) - mean;
            var += v * v;
        }
    }
    var /= (n_total as f64) - 1.0;
    if !(var > 0.0) {
        return n_total as f64;
    }

    let max_lag = (n_draws / 2).clamp(1, 50);
    let mut tau = 1.0;
    for lag in 1..=max_lag {
        let mut num = 0.0;
        let mut count = 0.0;
        for c in 0..n_chains {
            for d in 0..(n_draws - lag) {
                let a = sample_at(samples, c, d, n_draws, n_params, param) - mean;
                let b = sample_at(samples, c, d + lag, n_draws, n_params, param) - mean;
                num += a * b;
                count += 1.0;
            }
        }
        let rho = (num / count) / var;
        if rho < 0.05 {
            break;
        }
        tau += 2.0 * rho;
    }
    (n_total as f64) / tau.max(1.0)
}

#[inline]
fn sample_at(
    samples: &[f64],
    chain: usize,
    draw: usize,
    n_draws: usize,
    n_params: usize,
    param: usize,
) -> f64 {
    samples[(chain * n_draws + draw) * n_params + param]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_chains_rhat_near_one() {
        // 2 chains, 40 draws, 1 param — i.i.d. around the same mean.
        let n_chains = 2;
        let n_draws = 40;
        let n_params = 1;
        let mut samples = vec![0.0; n_chains * n_draws * n_params];
        for c in 0..n_chains {
            for d in 0..n_draws {
                // Tiny deterministic jitter; same distribution both chains.
                let jitter = (((c + 1) * (d + 3)) % 7) as f64 * 1e-6;
                samples[(c * n_draws + d) * n_params] = 1.0 + jitter;
            }
        }
        let r = max_split_rhat(&samples, n_chains, n_draws, n_params);
        assert!(r < 1.1, "rhat={r}");
        let ess = min_bulk_ess(&samples, n_chains, n_draws, n_params);
        assert!(ess > 5.0, "ess={ess}");
    }
}
