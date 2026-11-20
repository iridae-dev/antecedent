//! Exact donor-pair Gibbs updates under the uniform simplex prior and independent errors.

use antecedent_core::CausalRng;
use antecedent_kernels::standard_normal;

use super::{BayesianSyntheticControlSpec, ChainState, ScPanel, below, whitened_ss};

pub(super) fn uniform_independent(spec: &BayesianSyntheticControlSpec<'_>) -> bool {
    spec.dirichlet_concentration.to_bits() == 1.0_f64.to_bits() && spec.ar1_rho == 0.0
}

/// Sample N(0,1) on [a,b]. Reflect left tails. A narrow interval uses a uniform
/// envelope at its mode; a wide right tail uses the exponential tangent envelope.
/// Central intervals use normal rejection. All three produce the same conditional law.
fn truncated_normal(a: f64, b: f64, rng: &mut CausalRng) -> Option<f64> {
    if !a.is_finite() || !b.is_finite() || a >= b {
        return None;
    }
    let reflect = b < 0.0;
    let (lo, hi) = if reflect { (-b, -a) } else { (a, b) };
    let mode = lo.max(0.0);
    let narrow = (hi - lo) * (1.0 + mode) <= 1.0;
    let rate = 0.5 * lo + 0.5 * lo.hypot(2.0);
    // Failure to draw from the envelope leaves the pair at its current value.
    for _ in 0..128 {
        let sample = if narrow {
            let z = lo + (hi - lo) * rng.next_f64();
            if rng.next_f64().ln() > -0.5 * (z - mode) * (z + mode) {
                continue;
            }
            z
        } else if lo > 0.0 {
            let z = lo - (-rng.next_f64()).ln_1p() / rate;
            if z > hi || rng.next_f64().ln() > -0.5 * (z - rate).powi(2) {
                continue;
            }
            z
        } else {
            let z = standard_normal(rng);
            if z < lo || z > hi {
                continue;
            }
            z
        };
        return Some(if reflect { -sample } else { sample });
    }
    None
}

/// Hold the pair sum and all other weights fixed. With a uniform Dirichlet prior,
/// transferring delta from j to i has density proportional to
/// exp(-||residual - delta * difference||² / (2 sigma²)) on [-wi,wj].
pub(super) fn update(state: &mut ChainState, panel: &ScPanel, rng: &mut CausalRng) {
    let donors = state.weights.len();
    let i = below(rng, donors);
    let j = (i + 1 + below(rng, donors - 1)) % donors;
    let differences = || panel.donor_pre[i].iter().zip(&panel.donor_pre[j]).map(|(x, y)| x - y);
    let (mut norm, mut product) = (0.0, 0.0);
    for (residual, difference) in state.residual.iter().zip(differences()) {
        norm += difference * difference;
        product += residual * difference;
    }
    let (lo, hi) = (-state.weights[i], state.weights[j]);
    let delta = if norm == 0.0 {
        lo + (hi - lo) * rng.next_f64()
    } else {
        let mean = product / norm;
        let sd = (state.sigma2 / norm).sqrt();
        let Some(z) = truncated_normal((lo - mean) / sd, (hi - mean) / sd, rng) else {
            return;
        };
        mean + sd * z
    };
    if delta > lo && delta < hi {
        state.weights[i] += delta;
        state.weights[j] -= delta;
        for (residual, difference) in state.residual.iter_mut().zip(differences()) {
            *residual -= delta * difference;
        }
        state.ss = whitened_ss(&state.residual, 0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pair_updates_match_direct_quadrature_at_interior_and_boundary_modes() {
        // Integrate the observation likelihood on the simplex independently of the
        // sampler's completed square, truncation envelopes and maintained residual.
        let left = [4.0, -1.0, 2.0];
        let right = [1.0, 0.0, 0.5];
        for mode in [-0.2, 0.35, 1.3] {
            let target: Vec<_> =
                left.iter().zip(right).map(|(x, y)| mode * x + (1.0 - mode) * y).collect();
            // This isolates one conditional of a wider panel with all other
            // donors fixed. The public estimator requires at least three donors.
            let panel = ScPanel {
                donors: vec!["a".into(), "b".into()],
                target_pre: target.clone(),
                donor_pre: vec![left.to_vec(), right.to_vec()],
                target_post_mean: 0.0,
                donor_post_mean: vec![],
                n_post: 0,
                post_periods: vec![],
                target_post: vec![],
                donor_post: vec![],
                step_units: vec![],
            };
            let variance = 0.2;
            let mut mass = 0.0;
            let mut first = 0.0;
            let mut second = 0.0;
            for k in 0..8192 {
                let w = (f64::from(k) + 0.5) / 8192.0;
                let sq: f64 =
                    (0..3).map(|t| (target[t] - w * left[t] - (1.0 - w) * right[t]).powi(2)).sum();
                let density = (-sq / (2.0 * variance)).exp();
                mass += density;
                first += density * w;
                second += density * w * w;
            }
            let weights = vec![0.5, 0.5];
            let residual = super::super::residuals(&panel, &weights);
            let mut state =
                ChainState { weights, ss: whitened_ss(&residual, 0.0), residual, sigma2: variance };
            let mut rng = CausalRng::from_seed(0x74a5);
            let (mut empirical_first, mut empirical_second) = (0.0, 0.0);
            for _ in 0..16_384 {
                update(&mut state, &panel, &mut rng);
                let w = state.weights[0];
                assert!((0.0..1.0).contains(&w));
                assert!((state.weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
                empirical_first += w;
                empirical_second += w * w;
            }
            assert!((empirical_first / 16_384.0 - first / mass).abs() < 0.004);
            assert!((empirical_second / 16_384.0 - second / mass).abs() < 0.004);
            let direct = super::super::residuals(&panel, &state.weights);
            assert!(state.residual.iter().zip(direct).all(|(x, y)| (x - y).abs() < 1e-11));
        }
    }

    #[test]
    fn truncated_normal_matches_analytic_means_in_the_central_and_tail_intervals() {
        // E[Z | a <= Z <= b] = (phi(a) - phi(b)) / (Phi(b) - Phi(a)).
        for (lo, hi, expected) in [
            (-1.0, 2.0, 0.229_637),
            (2.0, 4.0, 2.370_633),
            (-4.0, -2.0, -2.370_633),
            (1.0, 1.1, 1.049_126),
        ] {
            let mut rng = CausalRng::from_seed(7389);
            let mean = (0..16_384)
                .map(|_| {
                    let draw = truncated_normal(lo, hi, &mut rng).unwrap();
                    assert!((lo..=hi).contains(&draw));
                    draw
                })
                .sum::<f64>()
                / 16_384.0;
            assert!((mean - expected).abs() < 0.02, "{lo}..{hi}: {mean}");
        }
    }
}
