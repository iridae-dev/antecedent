//! Posterior correctness checks for the conjugate Gaussian and Laplace GLM backends that do
//! not reuse the product's closed-form update.
//!
//! * Conjugate known-variance posterior is checked by Bayes' rule: the log-density ratio of the
//!   published `N(map, cov)` between two coefficient vectors must equal the difference of the
//!   unnormalised log prior plus weighted log-likelihood computed row by row, with weights,
//!   offsets, a zero-weight row and a dense `V0`.
//! * Integer prior weights must equal row replication, and a zero weight must equal deleting
//!   the row, for the unknown-variance (Normal-Inv-Gamma) fit; its residual-variance draws must
//!   have the mean `beta_n / (alpha_n - 1)` computed through the `y'Wy + m0'P m0 - m'Pn m` route
//!   and its coefficient covariance must equal `E[sigma^2] V_n` of the known-variance fit.
//! * Laplace logit, probit and Poisson modes and covariances are compared with 50-digit values
//!   from `oracle_data/laplace_glm_reference.py` (derivatives by numerical differentiation of
//!   the log-likelihood, separate Newton iteration).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_prob::{
    BayesDesignRef, BayesFitOptions, BayesLikelihood, CoefficientCorrelation,
    GaussianCoefficientPrior, InvGammaPrior, LaplaceWorkspace, PriorSet, PriorSpec,
    fit_conjugate_gaussian, fit_laplace_glm,
};

const X1: [f64; 14] =
    [-1.5, -1.1, -0.8, -0.4, -0.1, 0.2, 0.5, 0.7, 1.0, 1.3, 1.6, -0.6, 0.9, 0.0];
const Y_BIN: [f64; 14] = [0., 0., 1., 0., 0., 1., 0., 1., 1., 1., 1., 1., 0., 1.];
const Y_CNT: [f64; 14] = [0., 1., 0., 1., 1., 2., 1., 3., 2., 4., 6., 0., 3., 1.];
const Y_GAUSS: [f64; 14] =
    [-0.9, -0.2, 0.1, 0.4, 0.3, 1.1, 0.9, 1.2, 1.9, 2.3, 2.2, 0.0, 1.4, 0.5];
const WEIGHTS: [f64; 14] = [1., 1., 2., 1., 0.5, 1., 1., 3., 1., 1., 1., 0., 1., 2.];
const OFFSETS: [f64; 14] = [0.0, 0.1, -0.1, 0.0, 0.2, 0.0, 0.3, -0.2, 0.0, 0.1, 0.0, 0.5, -0.3, 0.0];
const INT_WEIGHTS: [f64; 14] = [1., 1., 2., 1., 3., 1., 1., 3., 1., 1., 1., 0., 1., 2.];
const M0: [f64; 2] = [0.25, -0.5];

fn colmajor(x1: &[f64]) -> Vec<f64> {
    let mut x = vec![1.0; x1.len()];
    x.extend_from_slice(x1);
    x
}

fn gaussian_prior(v0: [f64; 2], dense: Option<[f64; 4]>, tail: PriorSpec) -> PriorSet {
    let mut prior = PriorSet::new();
    prior.push(PriorSpec::GaussianCoefficients(GaussianCoefficientPrior {
        mean: Arc::from(M0.to_vec()),
        variance: Arc::from(v0.to_vec()),
    }));
    if let Some(v) = dense {
        prior.push(PriorSpec::CoefficientCorrelation(
            CoefficientCorrelation::from_covariance(&v, 2).unwrap().unwrap(),
        ));
    }
    prior.push(tail);
    prior.validate().unwrap();
    prior
}

fn inv2(m: [f64; 4]) -> [f64; 4] {
    let det = m[0] * m[3] - m[1] * m[2];
    [m[3] / det, -m[1] / det, -m[2] / det, m[0] / det]
}

fn quad2(p: [f64; 4], d: [f64; 2]) -> f64 {
    d[0] * (p[0] * d[0] + p[1] * d[1]) + d[1] * (p[2] * d[0] + p[3] * d[1])
}

#[test]
fn known_variance_posterior_obeys_bayes_rule_with_weights_offsets_and_dense_prior() {
    let sigma2 = 0.7;
    let v0 = [2.0, 0.6, 0.6, 1.0];
    let prior = gaussian_prior([2.0, 1.0], Some(v0), PriorSpec::KnownResidualVariance(sigma2));
    let x = colmajor(&X1);
    let design = BayesDesignRef {
        x_colmajor: &x,
        nrows: 14,
        ncols: 2,
        y: &Y_GAUSS,
        weights: Some(&WEIGHTS),
        offsets: Some(&OFFSETS),
    };
    let opts = BayesFitOptions { n_draws: 10, seed: 1, ..BayesFitOptions::default() };
    let fit = fit_conjugate_gaussian(design, &prior, &opts, &mut LaplaceWorkspace::default())
        .unwrap();
    let cov: [f64; 4] = fit.cov.clone().unwrap().try_into().unwrap();
    let prec = inv2(cov);
    let v0_inv = inv2(v0);
    let log_post = |b: [f64; 2]| {
        let mut rss = 0.0;
        for r in 0..14 {
            let res = Y_GAUSS[r] - OFFSETS[r] - b[0] - X1[r] * b[1];
            rss += WEIGHTS[r] * res * res;
        }
        -0.5 * (rss + quad2(v0_inv, [b[0] - M0[0], b[1] - M0[1]])) / sigma2
    };
    let gauss = |b: [f64; 2]| -0.5 * quad2(prec, [b[0] - fit.map[0], b[1] - fit.map[1]]);
    for (a, b) in [
        ([0.0, 0.0], [1.0, -1.0]),
        ([0.8, 0.9], [-0.4, 0.3]),
        ([fit.map[0] + 0.2, fit.map[1]], [2.0, 2.0]),
    ] {
        let direct = log_post(a) - log_post(b);
        let published = gauss(a) - gauss(b);
        assert!(
            (direct - published).abs() <= 1e-9 * (1.0 + direct.abs()),
            "log density ratio {direct} vs published Gaussian {published}"
        );
    }
}

fn nig_prior() -> PriorSet {
    gaussian_prior(
        [2.0, 1.0],
        Some([2.0, 0.6, 0.6, 1.0]),
        PriorSpec::ResidualInvGamma(InvGammaPrior { shape: 3.0, scale: 1.5 }),
    )
}

#[test]
fn nig_integer_weights_equal_replication_and_zero_weight_equals_deletion() {
    let opts = BayesFitOptions { n_draws: 400, seed: 9, ..BayesFitOptions::default() };
    let x = colmajor(&X1);
    let weighted = BayesDesignRef {
        x_colmajor: &x,
        nrows: 14,
        ncols: 2,
        y: &Y_GAUSS,
        weights: Some(&INT_WEIGHTS),
        offsets: Some(&OFFSETS),
    };
    let a = fit_conjugate_gaussian(weighted, &nig_prior(), &opts, &mut LaplaceWorkspace::default())
        .unwrap();

    let (mut x1, mut y, mut off) = (Vec::new(), Vec::new(), Vec::new());
    for r in 0..14 {
        for _ in 0..(INT_WEIGHTS[r] as usize) {
            x1.push(X1[r]);
            y.push(Y_GAUSS[r]);
            off.push(OFFSETS[r]);
        }
    }
    let xr = colmajor(&x1);
    let replicated = BayesDesignRef {
        x_colmajor: &xr,
        nrows: y.len(),
        ncols: 2,
        y: &y,
        weights: None,
        offsets: Some(&off),
    };
    let b = fit_conjugate_gaussian(replicated, &nig_prior(), &opts, &mut LaplaceWorkspace::default())
        .unwrap();
    for i in 0..2 {
        assert!((a.map[i] - b.map[i]).abs() <= 1e-10, "mean {i}: {} vs {}", a.map[i], b.map[i]);
    }
    // Same posterior parameters and seed give the same draws up to rounding; the shape
    // increment sum(w)/2 is therefore the replicated row count over two.
    let (sa, sb) = (a.draws.summarize(), b.draws.summarize());
    for q in 0..3 {
        assert!(
            (sa.mean[q] - sb.mean[q]).abs() <= 1e-8 * (1.0 + sb.mean[q].abs()),
            "draw mean {q}: {} vs {}",
            sa.mean[q],
            sb.mean[q]
        );
    }
}

#[test]
fn nig_residual_variance_mean_and_coefficient_covariance_match_independent_algebra() {
    let x = colmajor(&X1);
    let design = BayesDesignRef {
        x_colmajor: &x,
        nrows: 14,
        ncols: 2,
        y: &Y_GAUSS,
        weights: Some(&WEIGHTS),
        offsets: Some(&OFFSETS),
    };
    let opts = BayesFitOptions { n_draws: 60_000, seed: 21, ..BayesFitOptions::default() };
    let nig = fit_conjugate_gaussian(design, &nig_prior(), &opts, &mut LaplaceWorkspace::default())
        .unwrap();

    // Independent route: y'Wy + m0' P0 m0 - m' Pn m with explicit 2x2 algebra.
    let v0 = [2.0, 0.6, 0.6, 1.0];
    let p0 = inv2(v0);
    let (mut sxx, mut sxy, mut syy, mut n_eff) = ([0.0; 4], [0.0; 2], 0.0, 0.0);
    for r in 0..14 {
        let (w, yr) = (WEIGHTS[r], Y_GAUSS[r] - OFFSETS[r]);
        let row = [1.0, X1[r]];
        n_eff += w;
        syy += w * yr * yr;
        for i in 0..2 {
            sxy[i] += w * row[i] * yr;
            for j in 0..2 {
                sxx[2 * i + j] += w * row[i] * row[j];
            }
        }
    }
    let pn = [sxx[0] + p0[0], sxx[1] + p0[1], sxx[2] + p0[2], sxx[3] + p0[3]];
    let vn = inv2(pn);
    let rhs = [
        p0[0] * M0[0] + p0[1] * M0[1] + sxy[0],
        p0[2] * M0[0] + p0[3] * M0[1] + sxy[1],
    ];
    let m = [vn[0] * rhs[0] + vn[1] * rhs[1], vn[2] * rhs[0] + vn[3] * rhs[1]];
    let (alpha_n, beta_n) = (3.0 + 0.5 * n_eff, {
        let q = syy + quad2(p0, M0) - quad2(pn, m);
        1.5 + 0.5 * q
    });
    let sigma2_mean = beta_n / (alpha_n - 1.0);
    let s = nig.draws.summarize();
    assert!((nig.map[0] - m[0]).abs() < 1e-10 && (nig.map[1] - m[1]).abs() < 1e-10);
    assert!(
        (s.mean[2] - sigma2_mean).abs() / sigma2_mean < 0.02,
        "sigma2 mean {} vs {sigma2_mean}",
        s.mean[2]
    );
    // Var(beta_0) = E[sigma^2] V_n[0][0] because the conditional mean does not depend on sigma^2.
    let var0 = s.sd[0] * s.sd[0];
    let expected = sigma2_mean * vn[0];
    assert!((var0 - expected).abs() / expected < 0.05, "Var(beta0) {var0} vs {expected}");
}

struct Case {
    likelihood: BayesLikelihood,
    y: &'static [f64; 14],
    weighted: bool,
    mode: [f64; 2],
    cov: [f64; 4],
}

#[test]
fn laplace_glm_mode_and_covariance_match_fifty_digit_reference() {
    let cases = [
        Case {
            likelihood: BayesLikelihood::BernoulliLogit,
            y: &Y_BIN,
            weighted: false,
            mode: [0.22840440040191323, 0.72581317688816202],
            cov: [0.2982428327076739, -0.017359529451955143, -0.017359529451955143, 0.36637704565812243],
        },
        Case {
            likelihood: BayesLikelihood::BernoulliLogit,
            y: &Y_BIN,
            weighted: true,
            mode: [0.6227672729994279, 0.81886438920941016],
            cov: [0.27928213249477221, -0.0080919837633195159, -0.0080919837633195159, 0.36299641836817794],
        },
        Case {
            likelihood: BayesLikelihood::BernoulliProbit,
            y: &Y_BIN,
            weighted: false,
            mode: [0.13803347390428118, 0.53478272538564599],
            cov: [0.12109813426116763, -0.011016946582769438, -0.011016946582769438, 0.16449205462833723],
        },
        Case {
            likelihood: BayesLikelihood::BernoulliProbit,
            y: &Y_BIN,
            weighted: true,
            mode: [0.39542447064307119, 0.59682863302825311],
            cov: [0.10837564400105809, -0.009465173616644354, -0.009465173616644354, 0.15729338506442991],
        },
        Case {
            likelihood: BayesLikelihood::PoissonLog,
            y: &Y_CNT,
            weighted: false,
            mode: [0.088046918284619129, 1.0063246611241229],
            cov: [0.087032232772890244, -0.059370076558422315, -0.059370076558422315, 0.074219449065314703],
        },
        Case {
            likelihood: BayesLikelihood::PoissonLog,
            y: &Y_CNT,
            weighted: true,
            mode: [0.17228426955591389, 0.99817614954494016],
            cov: [0.070638687219765763, -0.050814549893949583, -0.050814549893949583, 0.065934477267440708],
        },
    ];
    let x = colmajor(&X1);
    let prior = gaussian_prior([4.0, 2.25], None, PriorSpec::ResidualInvGamma(InvGammaPrior::weakly_informative()));
    for case in &cases {
        let design = BayesDesignRef {
            x_colmajor: &x,
            nrows: 14,
            ncols: 2,
            y: case.y,
            weights: case.weighted.then_some(&WEIGHTS[..]),
            offsets: case.weighted.then_some(&OFFSETS[..]),
        };
        let opts = BayesFitOptions { n_draws: 10, seed: 2, ..BayesFitOptions::default() };
        let fit = fit_laplace_glm(case.likelihood, design, &prior, &opts, &mut LaplaceWorkspace::default())
            .unwrap();
        let cov = fit.cov.as_ref().unwrap();
        for i in 0..2 {
            assert!(
                (fit.map[i] - case.mode[i]).abs() <= 1e-9,
                "{:?} weighted={} mode {i}: {} vs {}",
                case.likelihood,
                case.weighted,
                fit.map[i],
                case.mode[i]
            );
        }
        for k in 0..4 {
            assert!(
                (cov[k] - case.cov[k]).abs() <= 1e-8 * (1.0 + case.cov[k].abs()),
                "{:?} weighted={} cov[{k}]: {} vs {}",
                case.likelihood,
                case.weighted,
                cov[k],
                case.cov[k]
            );
        }
    }
}
