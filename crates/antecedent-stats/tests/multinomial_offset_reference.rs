//! The unpenalised multinomial-logit fit is invariant to a location shift of a covariate.
//!
//! The maximum-likelihood fitted probabilities of a baseline-category logit do not depend on where
//! a covariate is centred: adding a constant to a column only reparametrises the intercepts. A
//! fit of the shifted design must therefore converge to the same probabilities and deviance as the
//! centred design. The reference is the centred fit itself, which is well conditioned and checked
//! against the score equations `X'(y - p) = 0` here, so the shifted fits are compared with a
//! solution verified by definition rather than with themselves. Offsets of 1e2, 1e3 and 1e4 are
//! covered, as is a weighted fit.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::{
    FaerBackend, GlmOptions, LeastSquaresWorkspace, MultinomialDesignRef, MultinomialFit,
    fit_multinomial_logit_weighted,
};

fn lcg(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

/// Three categories whose odds depend on one covariate and a second covariate.
fn data(n: usize) -> (Vec<f64>, Vec<f64>, Vec<u32>) {
    let mut state = 0x2545_F491_4F6C_DD1D_u64;
    let (mut z1, mut z2, mut y) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..n {
        let a = 2.0 * lcg(&mut state) - 1.0;
        let b = 2.0 * lcg(&mut state) - 1.0;
        let logits = [0.0, 0.3 + 1.2 * a - 0.5 * b, -0.2 - 0.9 * a + 0.8 * b];
        let top = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = logits.iter().map(|l| (l - top).exp()).collect();
        let total: f64 = weights.iter().sum();
        let u = lcg(&mut state) * total;
        let class = if u < weights[0] {
            0
        } else if u < weights[0] + weights[1] {
            1
        } else {
            2
        };
        z1.push(a);
        z2.push(b);
        y.push(class);
    }
    (z1, z2, y)
}

fn design(z1: &[f64], z2: &[f64], offset: f64) -> Vec<f64> {
    let n = z1.len();
    let mut out = vec![1.0; n];
    out.extend(z1.iter().map(|v| v + offset));
    out.extend(z2.iter().map(|v| v + offset));
    out
}

fn fit(design: &[f64], y: &[u32], weights: Option<&[f64]>) -> MultinomialFit {
    let mut workspace = LeastSquaresWorkspace::default();
    fit_multinomial_logit_weighted(
        MultinomialDesignRef {
            x_colmajor: design,
            nrows: y.len(),
            ncols: 3,
            y_category: y,
            n_categories: 3,
        },
        weights,
        &FaerBackend,
        &mut workspace,
        &GlmOptions::default(),
    )
    .unwrap()
}

fn probabilities(fit: &MultinomialFit, design: &[f64], n: usize) -> Vec<[f64; 3]> {
    (0..n)
        .map(|r| {
            let mut eta = [0.0; 3];
            for (c, slot) in eta.iter_mut().enumerate() {
                *slot = (0..3).map(|j| design[j * n + r] * fit.coefficients[c * 3 + j]).sum();
            }
            let top = eta.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mass: f64 = eta.iter().map(|e| (e - top).exp()).sum();
            [(eta[0] - top).exp() / mass, (eta[1] - top).exp() / mass, (eta[2] - top).exp() / mass]
        })
        .collect()
}

/// The largest score-equation residual `|X'W(y - p)|`.
fn score_residual(design: &[f64], y: &[u32], p: &[[f64; 3]], weights: Option<&[f64]>) -> f64 {
    let n = y.len();
    let mut worst = 0.0_f64;
    for class in 1..3 {
        for j in 0..3 {
            let s: f64 = (0..n)
                .map(|r| {
                    let w = weights.map_or(1.0, |w| w[r]);
                    w * design[j * n + r]
                        * (f64::from(u8::from(y[r] as usize == class)) - p[r][class])
                })
                .sum();
            worst = worst.max(s.abs());
        }
    }
    worst
}

#[test]
fn a_location_shift_of_a_covariate_does_not_change_the_fitted_probabilities() {
    let (z1, z2, y) = data(240);
    let n = y.len();
    for weighted in [false, true] {
        let weights: Vec<f64> = (0..n).map(|i| 0.5 + (i % 7) as f64 * 0.25).collect();
        let w = weighted.then_some(weights.as_slice());
        let centred_design = design(&z1, &z2, 0.0);
        let centred = fit(&centred_design, &y, w);
        assert!(centred.converged && !centred.separated, "the centred fit is the reference");
        let reference = probabilities(&centred, &centred_design, n);
        assert!(
            score_residual(&centred_design, &y, &reference, w) < 1e-6,
            "the centred solution satisfies the score equations"
        );
        for offset in [1e2, 1e3, 1e4] {
            let shifted_design = design(&z1, &z2, offset);
            let shifted = fit(&shifted_design, &y, w);
            assert!(
                shifted.converged && !shifted.separated,
                "offset {offset} weighted {weighted}: converged {} separated {} after {} iterations",
                shifted.converged,
                shifted.separated,
                shifted.iterations
            );
            let got = probabilities(&shifted, &shifted_design, n);
            let worst = got
                .iter()
                .zip(&reference)
                .flat_map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x - y).abs()))
                .fold(0.0_f64, f64::max);
            assert!(
                worst < 1e-6,
                "offset {offset} weighted {weighted}: probabilities move by {worst}"
            );
            let scale = centred.deviance.abs().max(1.0);
            assert!(
                (shifted.deviance - centred.deviance).abs() < 1e-6 * scale,
                "offset {offset}: deviance {} vs {}",
                shifted.deviance,
                centred.deviance
            );
        }
    }
}
