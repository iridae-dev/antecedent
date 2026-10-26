//! Tests for the fused normal-equation paths against their row-wise references.
#![allow(unused_imports)]

use super::*;
use crate::faer_backend::FaerBackend;
use crate::gram::{column_is_constant, invert_square};
use antecedent_kernels::norm_cdf;

/// With `max_iter = 1`, IRLS returns after a single coefficient update. Reported
/// deviance must match a direct recalculation from those returned coefficients
/// (not the pre-update iterate used inside the working-response pass).
#[test]
#[allow(
    clippy::too_many_lines,
    reason = "one fixture exercised end to end; later assertions read locals built above them"
)]
fn glm_deviance_matches_returned_coefficients_at_max_iter_one() {
    let n = 40usize;
    let mut x = vec![0.0; n * 2];
    let mut y_bin = vec![0.0; n];
    let mut y_count = vec![0.0; n];
    for i in 0..n {
        let t = if i < n / 2 { 0.0 } else { 1.0 };
        x[i] = 1.0;
        x[n + i] = t;
        y_bin[i] = if i % 7 == 0 { 1.0 - t } else { t };
        y_count[i] = if t < 0.5 { 2.0 } else { 5.0 };
    }
    let mut ws = LeastSquaresWorkspace::default();
    let opts = GlmOptions {
        max_iter: 1,
        tol: 0.0, // force a full update even if delta is tiny
        ridge_on_separation: None,
        ..GlmOptions::default()
    };
    let design_bin = GlmDesignRef { x_colmajor: &x, nrows: n, ncols: 2, y: &y_bin };
    let design_count = GlmDesignRef { x_colmajor: &x, nrows: n, ncols: 2, y: &y_count };

    let poisson =
        fit_glm(GlmFamily::PoissonLog, design_count, &FaerBackend, &mut ws, &opts).unwrap();
    assert_eq!(poisson.iterations, 1);
    let mut poisson_re = 0.0;
    for r in 0..n {
        let eta = poisson.coefficients[0] + x[n + r] * poisson.coefficients[1];
        let mu = eta.exp().max(MEAN_FLOOR);
        let yi = y_count[r];
        if yi > 0.0 {
            poisson_re += 2.0 * (yi * (yi / mu).ln() - (yi - mu));
        } else {
            poisson_re += 2.0 * mu;
        }
    }
    assert!(
        (poisson.deviance - poisson_re).abs() < 1e-12,
        "poisson deviance {} vs recomputed {}",
        poisson.deviance,
        poisson_re
    );

    let logit =
        fit_glm(GlmFamily::BinomialLogit, design_bin, &FaerBackend, &mut ws, &opts).unwrap();
    assert_eq!(logit.iterations, 1);
    let mut logit_re = 0.0;
    for r in 0..n {
        let eta = logit.coefficients[0] + x[n + r] * logit.coefficients[1];
        let mu = sigmoid(eta).clamp(PROB_CLIP, 1.0 - PROB_CLIP);
        if y_bin[r] > 0.0 {
            logit_re += -2.0 * mu.ln();
        } else {
            logit_re += -2.0 * (1.0 - mu).ln();
        }
    }
    assert!(
        (logit.deviance - logit_re).abs() < 1e-12,
        "logit deviance {} vs recomputed {}",
        logit.deviance,
        logit_re
    );

    let probit =
        fit_glm(GlmFamily::BinomialProbit, design_bin, &FaerBackend, &mut ws, &opts).unwrap();
    assert_eq!(probit.iterations, 1);
    let mut probit_re = 0.0;
    for r in 0..n {
        let eta = probit.coefficients[0] + x[n + r] * probit.coefficients[1];
        let mu = norm_cdf(eta).clamp(PROB_CLIP, 1.0 - PROB_CLIP);
        if y_bin[r] > 0.0 {
            probit_re += -2.0 * mu.ln();
        } else {
            probit_re += -2.0 * (1.0 - mu).ln();
        }
    }
    assert!(
        (probit.deviance - probit_re).abs() < 1e-12,
        "probit deviance {} vs recomputed {}",
        probit.deviance,
        probit_re
    );

    let nb_opts = GlmOptions { nb_alpha: NbAlphaPolicy::Fixed(0.5), ..opts };
    let nb =
        fit_glm(GlmFamily::NegativeBinomial, design_count, &FaerBackend, &mut ws, &nb_opts)
            .unwrap();
    assert_eq!(nb.iterations, 1);
    let theta = 2.0; // 1 / 0.5
    let mut nb_re = 0.0;
    for r in 0..n {
        let eta = nb.coefficients[0] + x[n + r] * nb.coefficients[1];
        let mu = eta.exp().max(MEAN_FLOOR);
        let yi = y_count[r];
        if yi > 0.0 {
            nb_re +=
                2.0 * (yi * (yi / mu).ln() - (yi + theta) * ((yi + theta) / (mu + theta)).ln());
        } else {
            nb_re += 2.0 * theta * ((theta + mu) / theta).ln();
        }
    }
    assert!(
        (nb.deviance - nb_re).abs() < 1e-12,
        "negbin deviance {} vs recomputed {}",
        nb.deviance,
        nb_re
    );
}

fn coefs_match(fused: &[f64], reference: &[f64]) {
    assert_eq!(fused.len(), reference.len());
    for (i, (left, right)) in fused.iter().zip(reference).enumerate() {
        let scale = 1.0 + left.abs().max(right.abs());
        assert!(
            (left - right).abs() <= 1e-10 * scale,
            "coefficient {i}: fused {left} vs row formula {right}"
        );
    }
}

/// Row-wise IRLS, the pass the fused column scaling replaces.
#[allow(
    clippy::needless_range_loop,
    reason = "index-based loop over parallel arrays; the subscript also names the matrix coordinate"
)]
fn row_logit(x: &[f64], y: &[f64], max_iter: u32, tol: f64) -> Vec<f64> {
    let nrows = y.len();
    let ncols = x.len() / nrows;
    let mut beta = vec![0.0; ncols];
    let mut x_w = vec![0.0; nrows * ncols];
    let mut z = vec![0.0; nrows];
    let mut ws = LeastSquaresWorkspace::default();
    for _ in 1..=max_iter {
        let mut max_delta = 0.0_f64;
        for r in 0..nrows {
            let mut eta = 0.0;
            for c in 0..ncols {
                eta += x[c * nrows + r] * beta[c];
            }
            let mu = sigmoid(eta);
            let mu_c = mu.clamp(PROB_CLIP, 1.0 - PROB_CLIP);
            let w = (mu_c * (1.0 - mu_c)).sqrt();
            z[r] = (eta + (y[r] - mu_c) / (mu_c * (1.0 - mu_c))) * w;
            for c in 0..ncols {
                x_w[c * nrows + r] = x[c * nrows + r] * w;
            }
        }
        let fit =
            FaerBackend.least_squares(&x_w, nrows, ncols, &z, &mut ws).expect("row logit");
        for c in 0..ncols {
            max_delta = max_delta.max((fit.coefficients[c] - beta[c]).abs());
            beta[c] = fit.coefficients[c];
        }
        if max_delta < tol {
            break;
        }
    }
    beta
}

/// Row-wise penalized normal equations, including a prior weight.
#[allow(
    clippy::needless_range_loop,
    reason = "index-based loop over parallel arrays; the subscript also names the matrix coordinate"
)]
fn row_ridge(
    x: &[f64],
    y: &[f64],
    weights: &[f64],
    lambda: f64,
    max_iter: u32,
    tol: f64,
) -> Vec<f64> {
    let nrows = y.len();
    let ncols = x.len() / nrows;
    let mut beta = vec![0.0; ncols];
    let mut xtwx = vec![0.0; ncols * ncols];
    let mut xtwz = vec![0.0; ncols];
    let mut row = vec![0.0; ncols];
    let unpenalize0 = ncols > 0 && column_is_constant(x, nrows, 0);
    for _ in 1..=max_iter {
        xtwx.fill(0.0);
        xtwz.fill(0.0);
        for r in 0..nrows {
            for c in 0..ncols {
                row[c] = x[c * nrows + r];
            }
            let mut eta = 0.0;
            for c in 0..ncols {
                eta += row[c] * beta[c];
            }
            let mu = sigmoid(eta).clamp(PROB_CLIP, 1.0 - PROB_CLIP);
            let w = mu * (1.0 - mu);
            let z = eta + (y[r] - mu) / w.max(MEAN_FLOOR);
            let w_fisher = w * weights[r];
            for i in 0..ncols {
                let xi = row[i];
                xtwz[i] += xi * w_fisher * z;
                let xiw = xi * w_fisher;
                for j in 0..ncols {
                    xtwx[i * ncols + j] += xiw * row[j];
                }
            }
        }
        for c in 0..ncols {
            if c == 0 && unpenalize0 {
                continue;
            }
            xtwx[c * ncols + c] += lambda;
        }
        let inv = invert_square(&xtwx, ncols).expect("row ridge");
        let mut max_delta = 0.0_f64;
        let mut beta_new = vec![0.0; ncols];
        for i in 0..ncols {
            let mut s = 0.0;
            for j in 0..ncols {
                s += inv[i * ncols + j] * xtwz[j];
            }
            beta_new[i] = s;
            max_delta = max_delta.max((s - beta[i]).abs());
        }
        beta = beta_new;
        if max_delta < tol {
            break;
        }
    }
    beta
}

#[test]
#[allow(
    clippy::needless_range_loop,
    reason = "index-based loop over parallel arrays; the subscript also names the matrix coordinate"
)]
fn fused_glm_matches_row_formula_on_a_penalized_design() {
    let n = 6usize;
    let mut x = vec![0.0; n * 2];
    let mut y = vec![0.0; n];
    let mut weights = vec![0.0; n];
    for i in 0..n {
        x[i] = 1.0;
        x[n + i] = i as f64 - 2.5;
        y[i] = if i % 2 == 0 { 0.0 } else { 1.0 };
        weights[i] = 0.5 + i as f64 * 0.25;
    }
    let design = GlmDesignRef { x_colmajor: &x, nrows: n, ncols: 2, y: &y };
    let opts = GlmOptions { ridge_on_separation: None, ..GlmOptions::new(40, 1e-12) };
    let mut ws = LeastSquaresWorkspace::default();
    let fused = fit_glm(GlmFamily::BinomialLogit, design, &FaerBackend, &mut ws, &opts)
        .expect("fused logit");
    let again = fit_glm(GlmFamily::BinomialLogit, design, &FaerBackend, &mut ws, &opts)
        .expect("reused workspace");
    assert_eq!(
        ws.grow_count, 1,
        "the solve workspace grows once, then the second fit reuses it"
    );
    assert!(fused.converged && again.converged);
    coefs_match(&fused.coefficients, &row_logit(&x, &y, opts.max_iter, opts.tol));
    coefs_match(&again.coefficients, &fused.coefficients);

    let lambda = 0.35;
    let penalized = fit_glm_ridge_weighted(
        GlmFamily::BinomialLogit,
        design,
        &weights,
        &FaerBackend,
        &mut ws,
        &opts,
        lambda,
    )
    .expect("fused ridge");
    assert!(penalized.converged && penalized.penalized);
    coefs_match(
        &penalized.coefficients,
        &row_ridge(&x, &y, &weights, lambda, opts.max_iter, opts.tol),
    );
    let mut score = [0.0; 2];
    for r in 0..n {
        let eta = penalized.coefficients[0] + x[n + r] * penalized.coefficients[1];
        let resid = weights[r] * (y[r] - sigmoid(eta));
        score[0] += resid;
        score[1] += resid * x[n + r];
    }
    score[1] -= lambda * penalized.coefficients[1];
    assert!(score[0].abs() < 1e-6 && score[1].abs() < 1e-6, "penalized score {score:?}");
}

#[test]
#[allow(
    clippy::needless_range_loop,
    reason = "index-based loop over parallel arrays; the subscript also names the matrix coordinate"
)]
fn fused_multinomial_score_matches_the_row_formula() {
    let nrows = 4usize;
    let ncols = 2usize;
    let k = 3usize;
    let n_free = k - 1;
    let m = n_free * ncols;
    let mut x = vec![0.0; nrows * ncols];
    for r in 0..nrows {
        x[r] = 1.0;
        x[nrows + r] = r as f64 - 1.5;
    }
    let y = [0u32, 1, 2, 1];
    let weights = [1.0, 0.5, 1.5, 0.25];
    let mut pi_rows = vec![0.0; nrows * k];
    for r in 0..nrows {
        let eta = [0.0, 0.2 * r as f64 - 0.1, -0.15 * r as f64];
        let max_eta = eta[0].max(eta[1]).max(eta[2]);
        let exponentials =
            [(eta[0] - max_eta).exp(), (eta[1] - max_eta).exp(), (eta[2] - max_eta).exp()];
        let total = exponentials[0] + exponentials[1] + exponentials[2];
        for j in 0..k {
            pi_rows[r * k + j] = exponentials[j] / total;
        }
    }
    let design =
        MultinomialDesignRef { x_colmajor: &x, nrows, ncols, y_category: &y, n_categories: k };
    let mut score = vec![0.0; m];
    let mut h = vec![0.0; m * m];
    accumulate_multinomial_score_and_information(
        design,
        &pi_rows,
        Some(&weights),
        &mut score,
        &mut h,
    );
    let mut score_ref = vec![0.0; m];
    let mut h_ref = vec![0.0; m * m];
    for r in 0..nrows {
        let yi = y[r] as usize;
        let obs = weights[r];
        for j in 1..k {
            let pj = pi_rows[r * k + j];
            let resid = if yi == j { 1.0 } else { 0.0 } - pj;
            let jb = (j - 1) * ncols;
            for c in 0..ncols {
                score_ref[jb + c] += obs * resid * x[c * nrows + r];
            }
            for jp in 1..k {
                let pjp = pi_rows[r * k + jp];
                let class_w = if j == jp { pj * (1.0 - pj) } else { -pj * pjp };
                let jpb = (jp - 1) * ncols;
                for c1 in 0..ncols {
                    let xc1 = x[c1 * nrows + r];
                    for c2 in 0..ncols {
                        h_ref[(jb + c1) * m + (jpb + c2)] +=
                            obs * class_w * xc1 * x[c2 * nrows + r];
                    }
                }
            }
        }
    }
    for (i, (left, right)) in score.iter().zip(&score_ref).enumerate() {
        assert!((left - right).abs() <= 1e-12, "score {i}: {left} vs {right}");
    }
    for (i, (left, right)) in h.iter().zip(&h_ref).enumerate() {
        assert!((left - right).abs() <= 1e-12, "information {i}: {left} vs {right}");
    }
}

#[test]
#[allow(
    clippy::needless_range_loop,
    reason = "index-based loop over parallel arrays; the subscript also names the matrix coordinate"
)]
fn separation_rank_deficiency_and_nonfinite_updates_stay_visible() {
    let n = 8usize;
    let mut x = vec![0.0; n * 2];
    let mut y = vec![0.0; n];
    for i in 0..n {
        let t = if i < n / 2 { 0.0 } else { 1.0 };
        x[i] = 1.0;
        x[n + i] = t;
        y[i] = t;
    }
    let mut ws = LeastSquaresWorkspace::default();
    let separated = fit_glm(
        GlmFamily::BinomialLogit,
        GlmDesignRef { x_colmajor: &x, nrows: n, ncols: 2, y: &y },
        &FaerBackend,
        &mut ws,
        &GlmOptions::new(30, 1e-8).without_separation_ridge(),
    )
    .expect("separation is a fit with a refusal, not a dropped error");
    assert!(separated.separated || !separated.converged);
    assert!(separated.require_ok().is_err());

    let mut collinear = vec![0.0; n * 3];
    for i in 0..n {
        let t = i as f64 / n as f64;
        collinear[i] = 1.0;
        collinear[n + i] = t;
        collinear[2 * n + i] = t;
        y[i] = if i % 2 == 0 { 0.0 } else { 1.0 };
    }
    let rank = fit_glm(
        GlmFamily::BinomialLogit,
        GlmDesignRef { x_colmajor: &collinear, nrows: n, ncols: 3, y: &y },
        &FaerBackend,
        &mut ws,
        &GlmOptions::new(20, 1e-8).without_separation_ridge(),
    );
    assert!(
        matches!(rank, Err(StatsError::RankDeficient { .. })),
        "collinear design must stay rank-deficient, got {rank:?}"
    );

    x[n] = f64::NAN;
    let nonfinite = fit_glm_ridge(
        GlmFamily::BinomialLogit,
        GlmDesignRef { x_colmajor: &x, nrows: n, ncols: 2, y: &y },
        &FaerBackend,
        &mut ws,
        &GlmOptions::new(10, 1e-8),
        0.2,
    );
    match nonfinite {
        Err(StatsError::Backend(_) | StatsError::RankDeficient { .. }) => {}
        Ok(fit) => {
            assert!(
                !fit.converged || fit.coefficients.iter().any(|coef| !coef.is_finite()),
                "a non-finite column must not become a finite converged fit"
            );
        }
        Err(other) => panic!("unexpected refusal {other:?}"),
    }

    let fisher = [1.0, f64::NAN];
    let working = [0.5, 0.5];
    let mut xtwx = [0.0; 4];
    let mut xtwz = [0.0; 2];
    accumulate_weighted_normal_equations(
        &[1.0, 1.0, 0.0, 1.0],
        2,
        2,
        &fisher,
        &working,
        &mut xtwx,
        &mut xtwz,
    );
    assert!(
        xtwx.iter().any(|v| !v.is_finite()) || xtwz.iter().any(|v| !v.is_finite()),
        "a non-finite weight stays in the normal equations"
    );
}
