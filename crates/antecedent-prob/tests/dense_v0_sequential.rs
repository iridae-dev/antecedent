//! Sequential conjugate updating through a dense coefficient prior equals pooling.
//!
//! A known-σ² Gaussian linear posterior is `β | y_A ~ N(m_A, Σ_A)`. Fitting batch
//! B under the prior `N(m_A, σ² V0)` with `V0 = Σ_A / σ²` (the full matrix, not its
//! diagonal) reproduces the pooled A ∪ B posterior exactly. A diagonal `V0` drops
//! the batch-A coefficient correlation and is overconfident on the linear
//! combinations along which that correlation made batch A uncertain.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_prob::{
    BayesDesignRef, BayesFitOptions, BayesLikelihood, CoefficientCorrelation,
    GaussianCoefficientPrior, LaplaceWorkspace, PriorSet, PriorSpec, fit_conjugate_gaussian,
    fit_laplace_glm,
};

const P: usize = 3;
const SIGMA2: f64 = 0.5;

/// Batch with an intercept and two strongly correlated regressors (row-major rows).
fn batch(n: usize, shift: usize) -> (Vec<[f64; P]>, Vec<f64>) {
    let mut rows = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let k = i + shift;
        let a = ((k * 37 % 101) as f64) / 101.0 - 0.5;
        let b = ((k * 53 % 97) as f64) / 97.0 - 0.5;
        let x1 = a;
        let x2 = 0.9 * a + 0.2 * b;
        let e = (((k * 29) % 11) as f64 - 5.0) * 0.15;
        rows.push([1.0, x1, x2]);
        y.push(0.3 + 1.5 * x1 - 0.7 * x2 + e);
    }
    (rows, y)
}

fn colmajor(rows: &[[f64; P]]) -> Vec<f64> {
    let n = rows.len();
    let mut x = vec![0.0; n * P];
    for (r, row) in rows.iter().enumerate() {
        for c in 0..P {
            x[c * n + r] = row[c];
        }
    }
    x
}

fn invert(a: &[f64], n: usize) -> Vec<f64> {
    let mut m = a.to_vec();
    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.0;
    }
    for col in 0..n {
        let piv = (col..n).max_by(|&i, &j| m[i * n + col].abs().total_cmp(&m[j * n + col].abs()));
        let piv = piv.unwrap();
        for k in 0..n {
            m.swap(col * n + k, piv * n + k);
            inv.swap(col * n + k, piv * n + k);
        }
        let d = m[col * n + col];
        for k in 0..n {
            m[col * n + k] /= d;
            inv[col * n + k] /= d;
        }
        for i in 0..n {
            if i != col {
                let f = m[i * n + col];
                for k in 0..n {
                    m[i * n + k] -= f * m[col * n + k];
                    inv[i * n + k] -= f * inv[col * n + k];
                }
            }
        }
    }
    inv
}

/// Analytic known-σ² posterior `(m, Σ)` under `N(m0, σ² V0)` with dense `V0`.
fn analytic_posterior(
    rows: &[[f64; P]],
    y: &[f64],
    m0: &[f64],
    v0: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let v0_inv = invert(v0, P);
    let mut a = v0_inv.clone();
    let mut b = [0.0; P];
    for i in 0..P {
        for j in 0..P {
            b[i] += v0_inv[i * P + j] * m0[j];
        }
    }
    for (row, &yr) in rows.iter().zip(y) {
        for i in 0..P {
            b[i] += row[i] * yr;
            for j in 0..P {
                a[i * P + j] += row[i] * row[j];
            }
        }
    }
    let a_inv = invert(&a, P);
    let mut m = vec![0.0; P];
    for i in 0..P {
        for j in 0..P {
            m[i] += a_inv[i * P + j] * b[j];
        }
    }
    (m, a_inv.iter().map(|v| v * SIGMA2).collect())
}

fn known_sigma2_prior(mean: &[f64], v0: &[f64], dense: bool) -> PriorSet {
    let variance: Vec<f64> = (0..P).map(|i| v0[i * P + i]).collect();
    let mut prior = PriorSet::new();
    prior.push(PriorSpec::GaussianCoefficients(GaussianCoefficientPrior {
        mean: Arc::from(mean.to_vec()),
        variance: Arc::from(variance),
    }));
    if dense {
        if let Some(corr) = CoefficientCorrelation::from_covariance(v0, P).unwrap() {
            prior.push(PriorSpec::CoefficientCorrelation(corr));
        }
    }
    prior.push(PriorSpec::KnownResidualVariance(SIGMA2));
    prior.validate().unwrap();
    prior
}

fn fit(rows: &[[f64; P]], y: &[f64], prior: &PriorSet, laplace: bool) -> (Vec<f64>, Vec<f64>) {
    let x = colmajor(rows);
    let design = BayesDesignRef {
        x_colmajor: &x,
        nrows: rows.len(),
        ncols: P,
        y,
        weights: None,
        offsets: None,
    };
    let opts = BayesFitOptions { n_draws: 50, seed: 3, ..BayesFitOptions::default() };
    let mut ws = LaplaceWorkspace::default();
    let res = if laplace {
        fit_laplace_glm(BayesLikelihood::GaussianIdentity, design, prior, &opts, &mut ws)
    } else {
        fit_conjugate_gaussian(design, prior, &opts, &mut ws)
    }
    .unwrap();
    (res.map, res.cov.expect("known-σ² Gaussian fits publish the exact covariance"))
}

fn rel_close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
}

#[test]
fn dense_v0_sequential_update_equals_pooled_fit() {
    let (rows_a, y_a) = batch(25, 0);
    let (rows_b, y_b) = batch(30, 1000);
    let mut v0 = vec![0.0; P * P];
    for i in 0..P {
        v0[i * P + i] = 100.0;
    }
    let m0 = [0.0; P];
    let base = known_sigma2_prior(&m0, &v0, false);

    // Stage A, then the hydrated dense prior V0 = Σ_A / σ².
    let (m_a, cov_a) = analytic_posterior(&rows_a, &y_a, &m0, &v0);
    let v0_a: Vec<f64> = cov_a.iter().map(|c| c / SIGMA2).collect();

    let mut rows_pool = rows_a.clone();
    rows_pool.extend_from_slice(&rows_b);
    let mut y_pool = y_a.clone();
    y_pool.extend_from_slice(&y_b);

    for laplace in [false, true] {
        let (map_pool, cov_pool) = fit(&rows_pool, &y_pool, &base, laplace);
        let (map_seq, cov_seq) =
            fit(&rows_b, &y_b, &known_sigma2_prior(&m_a, &v0_a, true), laplace);
        for i in 0..P {
            assert!(
                rel_close(map_seq[i], map_pool[i], 1e-9),
                "laplace={laplace} coef {i}: sequential {} vs pooled {}",
                map_seq[i],
                map_pool[i]
            );
        }
        for k in 0..P * P {
            assert!(
                rel_close(cov_seq[k], cov_pool[k], 1e-9),
                "laplace={laplace} cov[{k}]: sequential {} vs pooled {}",
                cov_seq[k],
                cov_pool[k]
            );
        }

        // The diagonal approximation is overconfident on β1 − β2: positively
        // correlated regressors leave the batch-A coefficients negatively
        // correlated, so dropping that covariance shrinks the prior on their
        // difference, the direction batch A could not resolve.
        let (_, cov_diag) = fit(&rows_b, &y_b, &known_sigma2_prior(&m_a, &v0_a, false), laplace);
        let diff_var = |c: &[f64]| c[4] + c[8] - 2.0 * c[5];
        assert!(
            diff_var(&cov_diag) < 0.9 * diff_var(&cov_pool),
            "diagonal V0 should understate Var(β1 − β2): diag {} vs pooled {}",
            diff_var(&cov_diag),
            diff_var(&cov_pool)
        );
    }
}

/// Unknown-σ² conjugate fits use the dense prior precision too: the NIG posterior
/// mean equals `(V0⁻¹ + X'X)⁻¹ (V0⁻¹ m0 + X'y)` with the dense `V0`.
#[test]
fn nig_posterior_mean_uses_dense_v0() {
    let (rows, y) = batch(20, 7);
    let mut v0 = vec![0.0; P * P];
    let sd = [2.0, 0.5, 0.5];
    let corr = [[1.0, 0.1, 0.0], [0.1, 1.0, -0.8], [0.0, -0.8, 1.0]];
    for i in 0..P {
        for j in 0..P {
            v0[i * P + j] = sd[i] * sd[j] * corr[i][j];
        }
    }
    let m0 = [0.1, 1.0, -1.0];
    let (expected, _) = analytic_posterior(&rows, &y, &m0, &v0);
    let mut prior = PriorSet::weakly_informative(P);
    prior.specs[0] = PriorSpec::GaussianCoefficients(GaussianCoefficientPrior {
        mean: Arc::from(m0.to_vec()),
        variance: Arc::from((0..P).map(|i| v0[i * P + i]).collect::<Vec<_>>()),
    });
    prior.push(PriorSpec::CoefficientCorrelation(
        CoefficientCorrelation::from_covariance(&v0, P).unwrap().unwrap(),
    ));
    prior.validate().unwrap();
    let x = colmajor(&rows);
    let design = BayesDesignRef {
        x_colmajor: &x,
        nrows: rows.len(),
        ncols: P,
        y: &y,
        weights: None,
        offsets: None,
    };
    let opts = BayesFitOptions { n_draws: 20, seed: 1, ..BayesFitOptions::default() };
    let res =
        fit_conjugate_gaussian(design, &prior, &opts, &mut LaplaceWorkspace::default()).unwrap();
    for (i, (got, want)) in res.map.iter().zip(&expected).enumerate() {
        assert!(rel_close(*got, *want, 1e-9), "{i}: {got} vs {want}");
    }
}

/// The GLM Laplace mode Hessian adds the dense prior precision: at the mode,
/// `Cov⁻¹ = X' diag(p(1−p)) X + V0⁻¹`.
#[test]
fn laplace_logit_hessian_adds_dense_prior_precision() {
    let n = 40;
    let mut rows = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let a = ((i * 37 % 41) as f64) / 41.0 - 0.5;
        let b = ((i * 11 % 43) as f64) / 43.0 - 0.5;
        rows.push([1.0, a, 0.8 * a + 0.3 * b]);
        y.push(if (i * 7) % 5 < 2 { 1.0 } else { 0.0 });
    }
    let sd = [1.5, 1.0, 1.0];
    let corr = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.7], [0.0, 0.7, 1.0]];
    let mut v0 = vec![0.0; P * P];
    for i in 0..P {
        for j in 0..P {
            v0[i * P + j] = sd[i] * sd[j] * corr[i][j];
        }
    }
    let mut prior = PriorSet::new();
    prior.push(PriorSpec::GaussianCoefficients(GaussianCoefficientPrior {
        mean: Arc::from(vec![0.0, 0.2, -0.2]),
        variance: Arc::from((0..P).map(|i| v0[i * P + i]).collect::<Vec<_>>()),
    }));
    prior.push(PriorSpec::CoefficientCorrelation(
        CoefficientCorrelation::from_covariance(&v0, P).unwrap().unwrap(),
    ));
    let x = colmajor(&rows);
    let design =
        BayesDesignRef { x_colmajor: &x, nrows: n, ncols: P, y: &y, weights: None, offsets: None };
    let opts = BayesFitOptions { n_draws: 20, seed: 1, grad_tol: 1e-12, max_iter: 100 };
    let res = fit_laplace_glm(
        BayesLikelihood::BernoulliLogit,
        design,
        &prior,
        &opts,
        &mut LaplaceWorkspace::default(),
    )
    .unwrap();
    let prec = invert(res.cov.as_ref().unwrap(), P);
    let v0_inv = invert(&v0, P);
    let mut expected = v0_inv;
    for row in &rows {
        let eta: f64 = (0..P).map(|c| row[c] * res.map[c]).sum();
        let pr = 1.0 / (1.0 + (-eta).exp());
        let w = pr * (1.0 - pr);
        for i in 0..P {
            for j in 0..P {
                expected[i * P + j] += w * row[i] * row[j];
            }
        }
    }
    for k in 0..P * P {
        assert!(rel_close(prec[k], expected[k], 1e-7), "[{k}] {} vs {}", prec[k], expected[k]);
    }
}
