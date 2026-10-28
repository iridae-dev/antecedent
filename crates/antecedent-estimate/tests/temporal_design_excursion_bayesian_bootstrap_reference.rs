//! The cluster Bayesian bootstrap of the temporal design ratio against its exact distribution.
//!
//! `oracle_data/temporal_design_excursion_bayesian_bootstrap_reference.py` (frozen in the JSON
//! beside it) takes per-cluster score sums and pair counts, and treats a draw as the ratio
//! `sum w_g S_g / sum w_g n_g` under Dirichlet(1, ..., 1) cluster weights. For two clusters the
//! ratio is a monotone function of one uniform weight, so its mean, standard deviation and
//! quantiles are exact quadrature values; for three clusters the mean and standard deviation are
//! integrals over the simplex. The product draws are compared with those values to Monte Carlo
//! accuracy: the mean and standard deviation within five standard errors, and the share of draws
//! below the exact 2.5, 50 and 97.5 percent quantiles within five binomial standard errors of the
//! nominal share. The cases include a ratio that decreases in the weight, a sign change, an
//! 8 to 1 cluster-count imbalance and a heavy cluster.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_estimate::temporal_design::{ClusterScores, bayesian_bootstrap};
use serde_json::Value;

const DRAWS: usize = 400_000;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().expect("array").iter().map(|x| x.as_f64().expect("number")).collect()
}

#[test]
fn the_posterior_of_the_ratio_matches_the_exact_dirichlet_integrals() {
    let cases: Vec<Value> = serde_json::from_str::<Value>(include_str!(
        "oracle_data/temporal_design_excursion_bayesian_bootstrap_reference.json"
    ))
    .unwrap()
    .as_array()
    .cloned()
    .unwrap();
    let ctx = ExecutionContext::for_tests(11);
    for case in &cases {
        let name = case["name"].as_str().unwrap();
        let k = usize::try_from(case["coordinates"].as_u64().unwrap()).unwrap();
        let scores = ClusterScores {
            coordinates: k,
            sums: floats(&case["sums"]),
            counts: floats(&case["counts"]),
        };
        // The point estimate is the unweighted ratio and is not the posterior mean.
        for (got, want) in scores.estimate().iter().zip(floats(&case["estimate"])) {
            assert!((got - want).abs() <= 1e-12 * want.abs().max(1.0), "{name} estimate");
        }
        let posterior = bayesian_bootstrap(&scores, DRAWS, &ctx).unwrap();
        assert_eq!(posterior.method, "bayesian_bootstrap_over_clusters");
        let (mean, sd) = (floats(&case["mean"]), floats(&case["sd"]));
        for j in 0..k {
            let draws = &posterior.draws[j];
            assert_eq!(draws.len(), DRAWS, "{name}");
            let n = DRAWS as f64;
            let se_mean = sd[j] / n.sqrt();
            assert!(
                (posterior.mean[j] - mean[j]).abs() <= 5.0 * se_mean,
                "{name} mean {j}: {} vs {} (se {se_mean})",
                posterior.mean[j],
                mean[j]
            );
            // The standard error of a standard deviation is at most sd * sqrt((kurtosis - 1) / 4n);
            // a 3 percent band covers five of them for these bounded-weight ratios.
            assert!(
                (posterior.standard_deviation[j] - sd[j]).abs() <= 0.03 * sd[j],
                "{name} sd {j}: {} vs {}",
                posterior.standard_deviation[j],
                sd[j]
            );
            if let Some(quantiles) = case["quantiles"].as_object().filter(|q| !q.is_empty()) {
                for (p_text, row) in quantiles {
                    let p: f64 = p_text.parse().unwrap();
                    let cut = floats(row)[j];
                    let below = draws.iter().filter(|v| **v <= cut).count() as f64 / n;
                    let tolerance = 5.0 * (p * (1.0 - p) / n).sqrt();
                    assert!(
                        (below - p).abs() <= tolerance,
                        "{name} share below the {p} quantile: {below}"
                    );
                }
                let [lo, hi] = [floats(&quantiles["0.025"])[j], floats(&quantiles["0.975"])[j]];
                let (lo, hi) = (lo.min(hi), lo.max(hi));
                let width = hi - lo;
                assert!((posterior.lower[j] - lo).abs() <= 0.02 * width, "{name} lower {j}");
                assert!((posterior.upper[j] - hi).abs() <= 0.02 * width, "{name} upper {j}");
            }
        }
    }
}

#[test]
fn draws_are_reproducible_and_depend_on_the_seed() {
    let scores =
        ClusterScores { coordinates: 1, sums: vec![3.0, 1.0, 2.0], counts: vec![4.0, 2.0, 3.0] };
    let a = bayesian_bootstrap(&scores, 64, &ExecutionContext::for_tests(5)).unwrap();
    let b = bayesian_bootstrap(&scores, 64, &ExecutionContext::for_tests(5)).unwrap();
    let c = bayesian_bootstrap(&scores, 64, &ExecutionContext::for_tests(6)).unwrap();
    assert_eq!(a, b);
    assert_ne!(a.draws, c.draws);
    // One draw, one cluster and no coordinates are refused, not summarised.
    assert!(bayesian_bootstrap(&scores, 1, &ExecutionContext::for_tests(5)).is_err());
    let single = ClusterScores { coordinates: 1, sums: vec![1.0], counts: vec![1.0] };
    assert!(bayesian_bootstrap(&single, 64, &ExecutionContext::for_tests(5)).is_err());
}
