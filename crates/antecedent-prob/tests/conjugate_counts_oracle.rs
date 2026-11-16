//! Conjugate count posteriors and power-prior discount selection against a multiprecision
//! oracle (`oracle_data/conjugate_counts_reference.py`, 50 digits, no Antecedent code).
//!
//! * Beta-Binomial and Gamma-Poisson posteriors are the closed forms
//!   `Beta(a + k, b + n - k)` and `Gamma(a + Σy, b + Σt)`; quantiles, CDFs, marginal
//!   likelihoods and posterior predictives equal the oracle's.
//! * The empirical-Bayes discount is the maximizer of the oracle's marginal-likelihood scores
//!   over the grid (smallest `α` on a tie), and the library's sensitivity table carries the
//!   oracle's scores and estimates.
//! * The linear target's composed power-path prior scores equal the oracle's direct
//!   multivariate normal (known σ²) and multivariate t (Inv-Gamma σ²) log densities.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    reason = "oracle counts are stored as JSON numbers and closed-form values are compared exactly"
)]

use std::sync::Arc;

use antecedent_prob::{
    AlphaCriterion, AlphaSelectionOptions, BetaBinomialPosterior, BetaHyperparameters, BetaSource,
    ExternalPriorSource, ExternalPriorWeight, GammaHyperparameters, GammaPoissonPosterior,
    GammaSource, GaussianCoefficientPrior, LinearTarget, PriorSet, PriorSpec,
    select_beta_power_prior, select_gamma_power_prior, select_linear_power_prior,
};
use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(include_str!("oracle_data/conjugate_counts_reference.json"))
        .expect("conjugate counts reference")
}

fn num(v: &Value) -> f64 {
    v.as_f64().expect("number")
}

fn nums(v: &Value) -> Vec<f64> {
    v.as_array().expect("array").iter().map(num).collect()
}

fn close_rel(got: f64, want: f64, rel: f64, what: &str) {
    let scale = want.abs().max(f64::MIN_POSITIVE);
    assert!((got - want).abs() <= rel * scale, "{what}: got {got:e}, want {want:e}");
}

fn close_abs(got: f64, want: f64, abs: f64, what: &str) {
    assert!((got - want).abs() <= abs, "{what}: got {got:e}, want {want:e}");
}

#[test]
fn beta_binomial_matches_the_oracle() {
    let doc = fixture();
    let tol = &doc["tolerance"];
    for case in doc["beta"].as_array().unwrap() {
        let prior = nums(&case["prior"]);
        let post = BetaBinomialPosterior::new(
            BetaHyperparameters { alpha: prior[0], beta: prior[1] },
            case["successes"].as_u64().unwrap(),
            case["trials"].as_u64().unwrap(),
        )
        .unwrap();
        let want = nums(&case["posterior"]);
        // Beta(a + k, b + n - k): exact in floating point for these inputs.
        assert_eq!(post.posterior().alpha, want[0]);
        assert_eq!(post.posterior().beta, want[1]);
        close_rel(post.mean(), num(&case["mean"]), 1e-14, "mean");
        close_rel(post.variance(), num(&case["variance"]), 1e-14, "variance");
        for (p, q) in nums(&case["probs"]).into_iter().zip(nums(&case["quantiles"])) {
            close_rel(post.quantile(p).unwrap(), q, num(&tol["quantile_rel"]), "quantile");
        }
        for pair in case["cdf_at"].as_array().unwrap() {
            let pair = nums(pair);
            close_abs(post.cdf(pair[0]), pair[1], 1e-12, "cdf");
        }
        close_abs(
            post.log_marginal_likelihood(),
            num(&case["log_marginal_likelihood"]),
            num(&tol["log_ml_abs"]),
            "log marginal likelihood",
        );
        let m = case["predictive_trials"].as_u64().unwrap();
        for (j, pmf) in nums(&case["predictive_pmf"]).into_iter().enumerate() {
            close_abs(
                post.predictive_pmf(m, j as u64).unwrap(),
                pmf,
                num(&tol["pmf_abs"]),
                "predictive pmf",
            );
        }
        close_rel(post.predictive_mean(m), num(&case["predictive_mean"]), 1e-13, "pred mean");
        close_rel(
            post.predictive_variance(m),
            num(&case["predictive_variance"]),
            1e-12,
            "pred var",
        );
    }
}

#[test]
fn gamma_poisson_matches_the_oracle() {
    let doc = fixture();
    let tol = &doc["tolerance"];
    for case in doc["gamma"].as_array().unwrap() {
        let prior = nums(&case["prior"]);
        let counts: Vec<u64> =
            case["counts"].as_array().unwrap().iter().map(|c| c.as_u64().unwrap()).collect();
        let exposures = case["exposures"].as_array().map(|_| nums(&case["exposures"]));
        let post = GammaPoissonPosterior::from_counts(
            GammaHyperparameters { shape: prior[0], rate: prior[1] },
            &counts,
            exposures.as_deref(),
        )
        .unwrap();
        let want = nums(&case["posterior"]);
        close_rel(post.posterior().shape, want[0], 1e-15, "shape");
        close_rel(post.posterior().rate, want[1], 1e-15, "rate");
        close_rel(post.mean(), num(&case["mean"]), 1e-14, "mean");
        close_rel(post.variance(), num(&case["variance"]), 1e-14, "variance");
        for (p, q) in nums(&case["probs"]).into_iter().zip(nums(&case["quantiles"])) {
            close_rel(post.quantile(p).unwrap(), q, num(&tol["quantile_rel"]), "quantile");
        }
        for pair in case["cdf_at"].as_array().unwrap() {
            let pair = nums(pair);
            close_abs(post.cdf(pair[0]), pair[1], 1e-12, "cdf");
        }
        close_abs(
            post.log_marginal_likelihood(),
            num(&case["log_marginal_likelihood"]),
            num(&tol["log_ml_abs"]),
            "log marginal likelihood",
        );
        let t = num(&case["predictive_exposure"]);
        for (y, pmf) in nums(&case["predictive_pmf"]).into_iter().enumerate() {
            close_abs(
                post.predictive_pmf(y as u64, t).unwrap(),
                pmf,
                num(&tol["pmf_abs"]),
                "predictive pmf",
            );
        }
        close_rel(post.predictive_mean(t), num(&case["predictive_mean"]), 1e-13, "pred mean");
        close_rel(
            post.predictive_variance(t),
            num(&case["predictive_variance"]),
            1e-12,
            "pred var",
        );
    }
}

fn eb_options(grid: Vec<f64>) -> AlphaSelectionOptions {
    let mut options = AlphaSelectionOptions::new(AlphaCriterion::EmpiricalBayes);
    options.grid = grid;
    options
}

#[test]
fn empirical_bayes_discount_is_the_marginal_likelihood_maximizer_on_a_beta_case() {
    let doc = fixture();
    for case in doc["empirical_bayes_beta"].as_array().unwrap() {
        let base = nums(&case["base"]);
        let src = nums(&case["source"]);
        let tgt = nums(&case["target"]);
        let (k, n) = (tgt[0] as usize, tgt[1] as usize);
        let outcomes: Vec<bool> = (0..n).map(|i| i < k).collect();
        let grid = nums(&case["grid"]);
        let report = select_beta_power_prior(
            BetaSource {
                base: BetaHyperparameters { alpha: base[0], beta: base[1] },
                successes: src[0] as u64,
                trials: src[1] as u64,
            },
            &outcomes,
            &eb_options(grid.clone()),
        )
        .unwrap();
        let index = case["selected_index"].as_u64().unwrap() as usize;
        assert_eq!(report.selected_index, index);
        assert_eq!(report.selected_alpha, grid[index]);
        let scores = nums(&case["scores"]);
        let estimates = nums(&case["estimates"]);
        for (i, row) in report.sensitivity.iter().enumerate() {
            close_abs(row.score, scores[i], 1e-10, "score");
            close_rel(row.estimate, estimates[i], 1e-13, "estimate");
        }
        // The grid maximizer brackets the continuous maximizer.
        let continuous = num(&case["continuous_maximizer"]);
        assert!((report.selected_alpha - continuous).abs() <= 0.05 + 1e-12, "{continuous}");
        // The reported interval is ordered and contains the estimate.
        assert!(report.interval.0 <= report.estimate && report.estimate <= report.interval.1);
    }
}

#[test]
fn empirical_bayes_discount_is_the_marginal_likelihood_maximizer_on_a_gamma_case() {
    let doc = fixture();
    for case in doc["empirical_bayes_gamma"].as_array().unwrap() {
        let base = nums(&case["base"]);
        let src = nums(&case["source"]);
        let counts: Vec<u64> =
            case["counts"].as_array().unwrap().iter().map(|c| c.as_u64().unwrap()).collect();
        let grid = nums(&case["grid"]);
        let report = select_gamma_power_prior(
            GammaSource {
                base: GammaHyperparameters { shape: base[0], rate: base[1] },
                total_count: src[0] as u64,
                exposure: src[1],
            },
            &counts,
            &eb_options(grid.clone()),
        )
        .unwrap();
        let index = case["selected_index"].as_u64().unwrap() as usize;
        assert_eq!(report.selected_index, index);
        let scores = nums(&case["scores"]);
        let estimates = nums(&case["estimates"]);
        for (i, row) in report.sensitivity.iter().enumerate() {
            close_abs(row.score, scores[i], 1e-10, "score");
            close_rel(row.estimate, estimates[i], 1e-13, "estimate");
        }
        let continuous = num(&case["continuous_maximizer"]);
        assert!((report.selected_alpha - continuous).abs() <= 0.05 + 1e-12, "{continuous}");
    }
}

#[test]
fn linear_target_scores_equal_the_direct_marginal_density() {
    let doc = fixture();
    for case in doc["linear"].as_array().unwrap() {
        let rows = case["x"].as_array().unwrap();
        let n_rows = rows.len();
        let n_cols = rows[0].as_array().unwrap().len();
        let x: Vec<f64> = rows.iter().flat_map(nums).collect();
        let y = nums(&case["y"]);
        let functional = nums(&case["functional"]);
        let gaussian = |block: &Value| {
            PriorSpec::GaussianCoefficients(GaussianCoefficientPrior {
                mean: Arc::from(nums(&block["mean"])),
                variance: Arc::from(nums(&block["variance"])),
            })
        };
        let mut baseline = PriorSet::new();
        baseline.push(gaussian(&case["baseline"]));
        if let Some(sigma2) = case["known_residual_variance"].as_f64() {
            baseline.push(PriorSpec::KnownResidualVariance(sigma2));
        }
        let mut source_prior = PriorSet::new();
        source_prior.push(gaussian(&case["source"]));
        let sources = [ExternalPriorSource {
            id: Arc::from("source"),
            prior: source_prior,
            weight: ExternalPriorWeight::power(1.0).unwrap(),
            ess: None,
        }];
        let grid = nums(&case["grid"]);
        let report = select_linear_power_prior(
            &sources,
            &baseline,
            LinearTarget { x: &x, y: &y, n_rows, n_cols, functional: &functional },
            &eb_options(grid.clone()),
        )
        .unwrap();
        let scores = nums(&case["scores"]);
        let estimates = nums(&case["estimates"]);
        let scales = nums(&case["scales"]);
        let index = case["selected_index"].as_u64().unwrap() as usize;
        assert_eq!(report.selected_index, index);
        for (i, row) in report.sensitivity.iter().enumerate() {
            close_abs(row.score, scores[i], 1e-8, "score");
            close_abs(row.estimate, estimates[i], 1e-9, "estimate");
            if case["known_residual_variance"].as_f64().is_some() {
                // Normal posterior: the 95% interval is the estimate ± 1.959963984540054 sd.
                let half = 1.959_963_984_540_054 * scales[i];
                close_abs(row.upper - row.estimate, half, 1e-8, "upper half-width");
                close_abs(row.estimate - row.lower, half, 1e-8, "lower half-width");
            }
        }
    }
}

#[test]
fn a_hand_set_discount_is_the_composed_prior_unchanged() {
    // alpha = 1 on the grid reproduces the composition the hand-set path builds: the power
    // prior is the sum of baseline and source precisions, so the posterior mean at alpha = 1
    // equals the closed form for a single coefficient.
    let mut baseline = PriorSet::new();
    baseline.push(PriorSpec::GaussianCoefficients(GaussianCoefficientPrior {
        mean: Arc::from(vec![0.0]),
        variance: Arc::from(vec![4.0]),
    }));
    baseline.push(PriorSpec::KnownResidualVariance(1.0));
    let mut source_prior = PriorSet::new();
    source_prior.push(PriorSpec::GaussianCoefficients(GaussianCoefficientPrior {
        mean: Arc::from(vec![2.0]),
        variance: Arc::from(vec![1.0]),
    }));
    let sources = [ExternalPriorSource {
        id: Arc::from("s"),
        prior: source_prior,
        weight: ExternalPriorWeight::power(1.0).unwrap(),
        ess: None,
    }];
    let x = [1.0, 1.0, 1.0, 1.0];
    let y = [1.0, 2.0, 3.0, 2.0];
    let report = select_linear_power_prior(
        &sources,
        &baseline,
        LinearTarget { x: &x, y: &y, n_rows: 4, n_cols: 1, functional: &[1.0] },
        &eb_options(vec![0.0, 0.5, 1.0]),
    )
    .unwrap();
    // Precision 1/4 + 1 = 1.25, mean (0 + 2) / 1.25 = 1.6; data precision 4, sum 8:
    // posterior mean (1.25 * 1.6 + 8) / 5.25.
    let want = (1.25 * 1.6 + 8.0) / 5.25;
    close_abs(report.sensitivity[2].estimate, want, 1e-12, "alpha = 1 posterior mean");
}
