//! Actual Gaussian adapter posterior summary precision against independent conjugate algebra.
//! Balanced binary T, Y=1+2T+N(0,1); known residual variance1, zero prior mean,
//! independent coefficient variance100. Complete checked fit and native sampling;
//! summary changes reuse exactly those issued draws. No frequentist interval license.
#![allow(clippy::cast_precision_loss, reason = "bounded simulation dimensions")]
#[path = "common/calibration.rs"]
mod calibration;
#[path = "../../antecedent-estimate/tests/common/candidate_calibration.rs"]
mod candidate;
use antecedent::BayesianConfig;
use antecedent::analysis::recalc_bayesian::{
    BayesianModel, BayesianRequest, BayesianSession, PosteriorSummarySpec,
    execute_bayesian_with_receipt,
};
use antecedent::analysis::recalc_receipt::UtilitySpec;
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_prob::{GaussianCoefficientPrior, PriorSet, PriorSpec};
use calibration::{grid_n, grid_seed, map_replicates, n_sim};

#[test]
#[ignore = "calibration: final measurement only"]
fn gaussian_adapter_posterior_summary_precision() {
    let n = grid_n(256);
    let draws = grid_n(4096);
    let results = map_replicates(n_sim(), |rep| {
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(move || {
                let mut rng = candidate::Generator::new(grid_seed(0x61ba_0000 + rep));
                let t: Vec<_> = (0..n).map(|i| (i % 2) as f64).collect();
                let y: Vec<_> = t.iter().map(|&a| 1. + 2. * a + rng.normal()).collect();
                let sy = y.iter().sum::<f64>();
                let sty = t.iter().zip(&y).map(|(t, y)| t * y).sum::<f64>();
                let a = n as f64 + 0.01;
                let b = n as f64 / 2.;
                let c = b + 0.01;
                let determinant = a * c - b * b;
                let mean = (a * sty - b * sy) / determinant;
                let sd = (a / determinant).sqrt();
                let mut prior = PriorSet::new();
                prior.push(PriorSpec::GaussianCoefficients(GaussianCoefficientPrior::isotropic(
                    2, 10.,
                )));
                prior.push(PriorSpec::KnownResidualVariance(1.));
                let mut request = BayesianRequest {
                    columns: vec![("t".into(), t), ("y".into(), y)],
                    edges: vec![(0, 1)],
                    treatment: VariableId::from_raw(0),
                    outcome: VariableId::from_raw(1),
                    model: BayesianModel::Gaussian,
                    inference: BayesianConfig::conjugate().n_draws(draws).prior(prior),
                    summary: PosteriorSummarySpec {
                        lower_probability: 0.025,
                        upper_probability: 0.975,
                        threshold: mean,
                    },
                    utility: UtilitySpec { benefit_per_unit: 1., cost: 0. },
                };
                let context = ExecutionContext::for_tests(grid_seed(0x72cb_0000 + rep));
                let mut session = BayesianSession::new();
                let result =
                    execute_bayesian_with_receipt(&mut session, &request, &context).unwrap();
                assert_eq!(result.receipt.totals().model_fits, 1);
                assert_eq!(result.receipt.totals().posterior_draws, draws as u64);
                let z = 1.959_963_984_540_054_f64;
                let density = (-z * z / 2.).exp() / std::f64::consts::TAU.sqrt();
                let quantile_scale = (0.025 * 0.975 / draws as f64).sqrt() / density;
                let mut errors = vec![
                    (result.law.mean - mean) / sd * (draws as f64).sqrt(),
                    (result.law.standard_deviation / sd - 1.) * (2. * draws as f64).sqrt(),
                    ((result.law.lower_quantile - mean) / sd + z) / quantile_scale,
                    ((result.law.upper_quantile - mean) / sd - z) / quantile_scale,
                ];
                for (offset, truth) in
                    [(-1., 0.158_655_253_931_457_07), (0., 0.5), (1., 0.841_344_746_068_542_9)]
                {
                    request.summary.threshold = mean + offset * sd;
                    let summary =
                        execute_bayesian_with_receipt(&mut session, &request, &context).unwrap();
                    assert_eq!(summary.receipt.totals().model_fits, 0);
                    assert_eq!(summary.receipt.totals().posterior_draws, 0);
                    errors.push(
                        (summary.law.probability_below - truth)
                            / (truth * (1. - truth) / draws as f64).sqrt(),
                    );
                }
                errors
            })
            .expect("bounded checked-engine stack")
            .join()
            .expect("checked replicate")
    });
    let mut bias = [0.; 7];
    let mut squared = [0.; 7];
    for result in &results {
        for i in 0..7 {
            bias[i] += result[i] / results.len() as f64;
            squared[i] += result[i] * result[i] / results.len() as f64;
        }
    }
    for i in 0..7 {
        assert!(bias[i].abs() < 0.3, "native posterior standardized bias {bias:?}");
        assert!(squared[i].sqrt() < 1.4, "native posterior standardized RMSE {squared:?}");
    }
    println!(
        "diagnostic-precision gaussian_adapter_posterior_summary_precision rows={n} draws={draws} repetitions={} standardized_bias={bias:?} standardized_squared_error={squared:?}; conditional_gaussian_only_no_sampling_coverage",
        results.len()
    );
}
