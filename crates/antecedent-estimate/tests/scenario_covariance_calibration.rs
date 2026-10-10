//! Numerical precision of the actual shared-row bootstrap covariance estimator.
//! IID A~Bernoulli(.3), E~Bernoulli(.4), B=A+E. For sample means,
//! population covariance is [[.21,.21],[.21,.45]]/n. Each sample's conditional
//! multinomial bootstrap covariance is independently computed from centered rows.
//! This measures covariance estimation and Monte Carlo precision, not CI coverage.
#![allow(clippy::cast_precision_loss, reason = "bounded simulation dimensions")]
#[path = "../../antecedent/tests/common/calibration.rs"]
mod calibration;
#[path = "common/candidate_calibration.rs"]
mod candidate;
use antecedent_core::ExecutionContext;
use antecedent_estimate::scenario_covariance::{
    ScenarioRowEstimator, SharedRowBootstrapOptions, shared_row_bootstrap_covariance,
};
use calibration::{grid_n, grid_seed, map_replicates, n_sim};
use std::sync::Arc;

#[test]
#[ignore = "calibration: final measurement only"]
fn shared_row_mean_covariance_precision() {
    let n = grid_n(256);
    let results = map_replicates(n_sim(), |rep| {
        let mut rng = candidate::Generator::new(grid_seed(0x5317_0000 + rep));
        let rows: Vec<(f64, f64)> = (0..n)
            .map(|_| {
                let a = rng.binary(0.3) as f64;
                (a, a + rng.binary(0.4) as f64)
            })
            .collect();
        let means = [
            rows.iter().map(|r| r.0).sum::<f64>() / n as f64,
            rows.iter().map(|r| r.1).sum::<f64>() / n as f64,
        ];
        // Direct centered score arithmetic; no production enumerator/covariance routine.
        let exact = [
            rows.iter().map(|r| (r.0 - means[0]).powi(2)).sum::<f64>(),
            rows.iter().map(|r| (r.0 - means[0]) * (r.1 - means[1])).sum::<f64>(),
            rows.iter().map(|r| (r.1 - means[1]).powi(2)).sum::<f64>(),
        ]
        .map(|v| v / (n * n) as f64);
        let units: Vec<Arc<str>> = (0..n).map(|i| Arc::from(format!("unit:{i}"))).collect();
        let scenarios: Vec<_> = (0..2)
            .map(|column| {
                let source = rows.clone();
                ScenarioRowEstimator::new(
                    format!("mean:{column}"),
                    "actual_shared_rows",
                    units.clone(),
                    move |counts| {
                        Ok(source
                            .iter()
                            .zip(counts)
                            .map(|(r, &c)| f64::from(c) * if column == 0 { r.0 } else { r.1 })
                            .sum::<f64>()
                            / n as f64)
                    },
                )
            })
            .collect();
        let estimate = shared_row_bootstrap_covariance(
            &scenarios,
            &SharedRowBootstrapOptions {
                replicates: 1000,
                seed: grid_seed(0x7519_0000 + rep),
                max_failure_fraction: 0.0,
            },
            &ExecutionContext::for_tests(grid_seed(0x9153_0000 + rep)),
        )
        .expect("bounded supported shared-row means cannot fail");
        assert_eq!(estimate.dimension(), 2);
        assert_eq!(estimate.failed_replicates, 0);
        ([estimate.entry(0, 0), estimate.entry(0, 1), estimate.entry(1, 1)], exact)
    });
    let mut mean_relative_error = [0.; 3];
    let mut squared_relative_error = [0.; 3];
    let mut scaled_covariance = [0.; 3];
    for (estimated, exact) in &results {
        for i in 0..3 {
            let error = (estimated[i] - exact[i]) * n as f64 / [0.21, 0.21, 0.45][i];
            mean_relative_error[i] += error / results.len() as f64;
            squared_relative_error[i] += error * error / results.len() as f64;
            scaled_covariance[i] += estimated[i] * n as f64 / results.len() as f64;
        }
    }
    for i in 0..3 {
        // 1000 inner draws: conditional covariance MC precision, fixed in advance.
        assert!(
            mean_relative_error[i].abs() < 0.03,
            "conditional covariance bias {mean_relative_error:?}"
        );
        assert!(
            squared_relative_error[i].sqrt() < 0.12,
            "conditional covariance RMSE {squared_relative_error:?}"
        );
        // Unconditional empirical plug-in covariance has known finite-n factor1-1/n.
        let expected = [0.21, 0.21, 0.45][i] * (1. - 1. / n as f64);
        assert!(
            (scaled_covariance[i] / expected - 1.).abs() < 0.12,
            "sampling covariance ratio {scaled_covariance:?}"
        );
    }
    println!(
        "diagnostic-precision shared_row_mean_covariance_precision n={n} samples={} inner_replicates=1000 mean_relative_error={mean_relative_error:?} scaled_covariance={scaled_covariance:?}; point_only_no_interval_license",
        results.len()
    );
}
