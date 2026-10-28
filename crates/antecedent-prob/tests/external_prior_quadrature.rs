//! Numerical-integration checks of the Gaussian external-prior composition.
//!
//! Each composed prior is compared with the moments of the density it claims to summarise,
//! integrated on a grid without using the closed forms in the implementation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_prob::{
    CoefficientCorrelation, ExternalPriorSource, ExternalPriorWeight, GaussianCoefficientPrior,
    PriorSet, PriorSpec, compose_external_priors,
};
use std::sync::Arc;

fn gauss(mean: f64, var: f64) -> PriorSet {
    let mut p = PriorSet::new();
    p.push(PriorSpec::GaussianCoefficients(
        GaussianCoefficientPrior::shared(1, mean, var).unwrap(),
    ));
    p
}

fn dense_source(alpha: f64, mixture: Option<f64>) -> ExternalPriorSource {
    let mut prior = PriorSet::new();
    prior.push(PriorSpec::GaussianCoefficients(GaussianCoefficientPrior {
        mean: Arc::from(vec![1.0, -1.0]),
        variance: Arc::from(vec![0.04, 0.09]),
    }));
    prior.push(PriorSpec::CoefficientCorrelation(
        CoefficientCorrelation::new(2, vec![1.0, -0.8, -0.8, 1.0]).unwrap(),
    ));
    let weight = match mixture {
        Some(w) => ExternalPriorWeight::power_mixture(alpha, w).unwrap(),
        None => ExternalPriorWeight::power(alpha).unwrap(),
    };
    ExternalPriorSource { id: Arc::from("dense"), prior, weight, ess: None }
}

/// Independent trapezoid-quadrature moments of a density given by its log on a grid.
fn grid_moments_1d(log_density: impl Fn(f64) -> f64) -> (f64, f64) {
    let (lo, hi, steps) = (-40.0, 40.0, 160_000usize);
    let h = (hi - lo) / steps as f64;
    let (mut z, mut m1, mut m2) = (0.0, 0.0, 0.0);
    for i in 0..=steps {
        let x = lo + h * i as f64;
        let w = log_density(x).exp() * if i == 0 || i == steps { 0.5 } else { 1.0 };
        z += w;
        m1 += w * x;
        m2 += w * x * x;
    }
    let mean = m1 / z;
    (mean, m2 / z - mean * mean)
}

fn log_normal(x: f64, mean: f64, var: f64) -> f64 {
    -0.5 * ((x - mean).powi(2) / var + (2.0 * std::f64::consts::PI * var).ln())
}

fn source(id: &str, mean: f64, var: f64, weight: ExternalPriorWeight) -> ExternalPriorSource {
    ExternalPriorSource { id: Arc::from(id), prior: gauss(mean, var), weight, ess: None }
}

/// The precision-add path is the Gaussian closed form of `baseline * prod f_k^alpha_k`;
/// numerical integration of that product density, which never uses the closed form,
/// must give the composed mean and variance (an applied alpha of zero drops its source).
#[test]
fn power_composition_matches_quadrature_of_the_tempered_density_product() {
    let (b_mean, b_var) = (0.5, 9.0);
    let specs = [(2.0, 1.0, 0.5), (-1.5, 0.25, 0.8), (4.0, 2.0, 1.0), (30.0, 0.01, 0.0)];
    let sources: Vec<_> = specs
        .iter()
        .enumerate()
        .map(|(i, &(m, v, a))| {
            source(&format!("s{i}"), m, v, ExternalPriorWeight::power(a).unwrap())
        })
        .collect();
    let composed = compose_external_priors(&sources, &gauss(b_mean, b_var)).unwrap();
    let coef = composed.prior.gaussian_coefficients().unwrap();
    let (mean, var) = grid_moments_1d(|x| {
        log_normal(x, b_mean, b_var)
            + specs.iter().map(|&(m, v, a)| a * log_normal(x, m, v)).sum::<f64>()
    });
    assert!((coef.mean[0] - mean).abs() < 1e-8, "{} vs {mean}", coef.mean[0]);
    assert!((coef.variance[0] - var).abs() < 1e-8, "{} vs {var}", coef.variance[0]);
}

/// The mixture path moment-matches `sum_k w_k N(m_k, V_k / alpha_k)` plus the leftover
/// baseline mass; integrating that mixture density numerically gives the same moments,
/// including when a zero-alpha source folds its weight back onto the baseline.
#[test]
fn mixture_composition_matches_quadrature_of_the_mixture_density() {
    let (b_mean, b_var) = (-1.0, 16.0);
    let specs = [(3.0, 1.0, 0.5, 0.3), (-4.0, 0.5, 1.0, 0.2), (9.0, 4.0, 0.0, 0.25)];
    let sources: Vec<_> = specs
        .iter()
        .enumerate()
        .map(|(i, &(m, v, a, w))| {
            source(&format!("s{i}"), m, v, ExternalPriorWeight::power_mixture(a, w).unwrap())
        })
        .collect();
    let composed = compose_external_priors(&sources, &gauss(b_mean, b_var)).unwrap();
    let coef = composed.prior.gaussian_coefficients().unwrap();
    // Active weights 0.3 and 0.2; the dropped 0.25 and the unassigned 0.25 stay on the baseline.
    let (mean, var) = grid_moments_1d(|x| {
        let density = 0.3 * log_normal(x, 3.0, 1.0 / 0.5).exp()
            + 0.2 * log_normal(x, -4.0, 0.5).exp()
            + 0.5 * log_normal(x, b_mean, b_var).exp();
        density.ln()
    });
    assert!((coef.mean[0] - mean).abs() < 1e-8, "{} vs {mean}", coef.mean[0]);
    assert!((coef.variance[0] - var).abs() < 1e-8, "{} vs {var}", coef.variance[0]);
}

/// The correlated path agrees with a two-dimensional quadrature of the tempered product.
#[test]
fn dense_power_composition_matches_two_dimensional_quadrature() {
    let baseline = {
        let mut p = PriorSet::new();
        p.push(PriorSpec::GaussianCoefficients(GaussianCoefficientPrior {
            mean: Arc::from(vec![0.0, 0.5]),
            variance: Arc::from(vec![4.0, 4.0]),
        }));
        p
    };
    let alpha = 0.6;
    let composed = compose_external_priors(&[dense_source(alpha, None)], &baseline).unwrap();
    let coef = composed.prior.gaussian_coefficients().unwrap();
    let corr = composed.prior.coefficient_correlation().unwrap();
    // Source: mean (1, -1), sd (0.2, 0.3), correlation -0.8.
    let (sd0, sd1, rho) = (0.2_f64, 0.3_f64, -0.8_f64);
    let log_source = |x: f64, y: f64| {
        let (u, v) = ((x - 1.0) / sd0, (y + 1.0) / sd1);
        -(u * u - 2.0 * rho * u * v + v * v) / (2.0 * (1.0 - rho * rho))
    };
    let (lo, hi, steps) = (-3.0, 3.0, 1200usize);
    let h = (hi - lo) / steps as f64;
    let (mut z, mut mx, mut my, mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for i in 0..=steps {
        let x = lo + h * i as f64;
        for j in 0..=steps {
            let y = lo + h * j as f64;
            let w =
                (-0.5 * (x * x / 4.0 + (y - 0.5).powi(2) / 4.0) + alpha * log_source(x, y)).exp();
            z += w;
            mx += w * x;
            my += w * y;
            sxx += w * x * x;
            syy += w * y * y;
            sxy += w * x * y;
        }
    }
    let (mean_x, mean_y) = (mx / z, my / z);
    let (var_x, var_y) = (sxx / z - mean_x * mean_x, syy / z - mean_y * mean_y);
    let cov = sxy / z - mean_x * mean_y;
    assert!((coef.mean[0] - mean_x).abs() < 1e-6 && (coef.mean[1] - mean_y).abs() < 1e-6);
    assert!((coef.variance[0] - var_x).abs() < 1e-6 && (coef.variance[1] - var_y).abs() < 1e-6);
    assert!((corr.matrix()[1] - cov / (var_x * var_y).sqrt()).abs() < 1e-6);
}
