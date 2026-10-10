//! Conditional posterior-summary precision for actual checked basis and source-prior adapters.
//! Independent finite-dimensional algebra and quadrature; no sampling coverage license.
//! SPDX-License-Identifier: MIT OR Apache-2.0
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
use antecedent::analysis::recalc_receipt::{RecalcRunError, UtilitySpec};
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_io::{PriorMapping, consume_analysis_result};
use antecedent_prob::{GaussianCoefficientPrior, PosteriorQuantityKind, PriorSet, PriorSpec};
use calibration::{grid_n, grid_seed, map_replicates, n_sim};

fn inverse(mut a: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut result = vec![vec![0.; n]; n];
    for (i, row) in result.iter_mut().enumerate() {
        row[i] = 1.;
    }
    for i in 0..n {
        let pivot = a[i][i];
        assert!(pivot > 1e-12);
        for j in 0..n {
            a[i][j] /= pivot;
            result[i][j] /= pivot;
        }
        for k in 0..n {
            if k == i {
                continue;
            }
            let factor = a[k][i];
            for j in 0..n {
                a[k][j] -= factor * a[i][j];
                result[k][j] -= factor * result[i][j];
            }
        }
    }
    result
}
fn multiply(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    a.iter().map(|row| row.iter().zip(b).map(|(x, y)| x * y).sum()).collect()
}
struct StudentOracle {
    mean: f64,
    scale: f64,
    df: f64,
}
impl StudentOracle {
    fn kernel(&self, x: f64) -> f64 {
        (1. + x * x / self.df).powf(-(self.df + 1.) / 2.)
    }
    fn integral(&self, end: f64) -> f64 {
        let steps = 2048;
        let h = end / f64::from(steps);
        let mut sum = self.kernel(0.) + self.kernel(end);
        for i in 1..steps {
            sum += if i % 2 == 0 { 2. } else { 4. } * self.kernel(f64::from(i) * h);
        }
        sum * h / 3.
    }
    fn cdf(&self, x: f64) -> f64 {
        0.5 + self.integral(x) / (2. * self.integral(16.))
    }
    fn quantile(&self, p: f64) -> f64 {
        let (mut low, mut high) = (-8., 8.);
        for _ in 0..55 {
            let mid = (low + high) / 2.;
            if self.cdf(mid) < p {
                low = mid;
            } else {
                high = mid;
            }
        }
        (low + high) / 2.
    }
    fn sd(&self) -> f64 {
        self.scale * (self.df / (self.df - 2.)).sqrt()
    }
}
fn nig(
    design: &[Vec<f64>],
    y: &[f64],
    prior_mean: &[f64],
    prior_precision: &[Vec<f64>],
) -> StudentOracle {
    let p = prior_mean.len();
    let mut precision = prior_precision.to_vec();
    let mut moment = multiply(prior_precision, prior_mean);
    for (row, y) in design.iter().zip(y) {
        for i in 0..p {
            moment[i] += row[i] * y;
            for j in 0..p {
                precision[i][j] += row[i] * row[j];
            }
        }
    }
    let covariance = inverse(precision);
    let mean = multiply(&covariance, &moment);
    let residual = design
        .iter()
        .zip(y)
        .map(|(row, y)| (y - row.iter().zip(&mean).map(|(x, b)| x * b).sum::<f64>()).powi(2))
        .sum::<f64>();
    let difference = mean.iter().zip(prior_mean).map(|(a, b)| a - b).collect::<Vec<_>>();
    let penalty = difference
        .iter()
        .zip(multiply(prior_precision, &difference))
        .map(|(a, b)| a * b)
        .sum::<f64>();
    let alpha = 0.001 + design.len() as f64 / 2.;
    let beta = 0.001 + (residual + penalty) / 2.;
    StudentOracle { mean: mean[1], scale: (beta / alpha * covariance[1][1]).sqrt(), df: 2. * alpha }
}
fn request(t: Vec<f64>, z: Option<Vec<f64>>, y: Vec<f64>, draws: usize) -> BayesianRequest {
    let (columns, edges, outcome, model) = if let Some(z) = z {
        (
            vec![("t".into(), t), ("z".into(), z), ("y".into(), y)],
            vec![(0, 2), (1, 0), (1, 2)],
            2,
            BayesianModel::QuadraticBasis,
        )
    } else {
        (vec![("t".into(), t), ("y".into(), y)], vec![(0, 1)], 1, BayesianModel::Gaussian)
    };
    BayesianRequest {
        columns,
        edges,
        treatment: VariableId::from_raw(0),
        outcome: VariableId::from_raw(outcome),
        model,
        inference: BayesianConfig::conjugate().n_draws(draws),
        summary: PosteriorSummarySpec::default(),
        utility: UtilitySpec { benefit_per_unit: 1., cost: 0. },
    }
}
fn errors(
    mut request: BayesianRequest,
    oracle: &StudentOracle,
    context: &ExecutionContext,
) -> Vec<f64> {
    let draws = request.inference.n_draws;
    let expected_fits = if request.inference.prior_artifact.is_some() { 2 } else { 1 };
    let mut session = BayesianSession::new();
    let result = execute_bayesian_with_receipt(&mut session, &request, context).unwrap();
    assert_eq!(result.receipt.totals().model_fits, expected_fits);
    assert_eq!(result.receipt.totals().posterior_draws, expected_fits * draws as u64);
    let sd = oracle.sd();
    let mut errors = vec![
        (result.law.mean - oracle.mean) / sd * (draws as f64).sqrt(),
        (result.law.standard_deviation / sd - 1.)
            * (4. * draws as f64 / (2. + 6. / (oracle.df - 4.))).sqrt(),
    ];
    for (reported, p) in [(result.law.lower_quantile, 0.025), (result.law.upper_quantile, 0.975)] {
        let q = oracle.quantile(p);
        let density = oracle.kernel(q) / (2. * oracle.integral(16.));
        errors.push(
            ((reported - oracle.mean) / oracle.scale - q)
                / ((p * (1. - p) / draws as f64).sqrt() / density),
        );
    }
    let pointer = session.posterior().unwrap().draws.column(0).unwrap().as_ptr();
    for offset in [-1., 0., 1.] {
        let truth = oracle.cdf(offset * sd / oracle.scale);
        request.summary.threshold = oracle.mean + offset * sd;
        let summary = execute_bayesian_with_receipt(&mut session, &request, context).unwrap();
        assert_eq!(
            summary.receipt.totals().model_fits + summary.receipt.totals().posterior_draws,
            0
        );
        assert_eq!(session.posterior().unwrap().draws.column(0).unwrap().as_ptr(), pointer);
        errors.push(
            (summary.law.probability_below - truth) / (truth * (1. - truth) / draws as f64).sqrt(),
        );
    }
    consume_analysis_result(&session.export_result(context).unwrap()).unwrap();
    errors
}
fn assert_precision(name: &str, results: &[Vec<f64>], n: usize, draws: usize) {
    let mut bias = [0.; 7];
    let mut squared = [0.; 7];
    for row in results {
        for i in 0..7 {
            bias[i] += row[i] / results.len() as f64;
            squared[i] += row[i].powi(2) / results.len() as f64;
        }
    }
    for i in 0..7 {
        assert!(bias[i].abs() < 0.3, "{name} standardized bias {bias:?}");
        assert!(squared[i].sqrt() < 1.4, "{name} standardized RMSE {squared:?}");
    }
    println!(
        "diagnostic-precision {name} rows={n} draws={draws} repetitions={} standardized_bias={bias:?} standardized_squared_error={squared:?}; conditional_student_t_only_no_sampling_coverage",
        results.len()
    );
}
fn large_stack<F: FnOnce() -> Vec<f64> + Send + 'static>(work: F) -> Vec<f64> {
    std::thread::Builder::new().stack_size(16 * 1024 * 1024).spawn(work).unwrap().join().unwrap()
}
#[test]
#[ignore = "calibration: final measurement only"]
fn quadratic_basis_nig_adapter_summary_precision() {
    let n = grid_n(240);
    let draws = grid_n(4096);
    let results = map_replicates(n_sim(), |rep| {
        large_stack(move || {
            let mut rng = candidate::Generator::new(grid_seed(0x81bc_0000 + rep));
            let t = (0..n).map(|i| (i % 2) as f64).collect::<Vec<_>>();
            let z = (0..n).map(|i| ((i / 2) % 3) as f64 - 1.).collect::<Vec<_>>();
            let design =
                t.iter().zip(&z).map(|(t, z)| vec![1., *t, *z, t * z, z * z]).collect::<Vec<_>>();
            let y = design
                .iter()
                .map(|x| {
                    x.iter().zip([1., 2., 0.5, 0.4, 0.3]).map(|(x, b)| x * b).sum::<f64>()
                        + rng.normal()
                })
                .collect::<Vec<_>>();
            let prior: Vec<Vec<f64>> =
                (0..5).map(|i| (0..5).map(|j| if i == j { 0.01 } else { 0. }).collect()).collect();
            let oracle = nig(&design, &y, &[0.; 5], &prior);
            errors(
                request(t, Some(z), y, draws),
                &oracle,
                &ExecutionContext::for_tests(grid_seed(0x82bc_0000 + rep)),
            )
        })
    });
    assert_precision("quadratic_basis_nig_adapter_summary_precision", &results, n, draws);
}
#[test]
#[ignore = "calibration: final measurement only"]
fn mapped_native_source_prior_adapter_summary_precision() {
    let n = grid_n(256);
    let draws = grid_n(4096);
    let results =
        map_replicates(n_sim(), |rep| large_stack(move || mapped_prior_replicate(n, draws, rep)));
    assert_precision("mapped_native_source_prior_adapter_summary_precision", &results, n, draws);
}
fn mapped_prior_replicate(n: usize, draws: usize, rep: u64) -> Vec<f64> {
    let mut rng = candidate::Generator::new(grid_seed(0x91bc_0000 + rep));
    let t = (0..n).map(|i| (i % 2) as f64).collect::<Vec<_>>();
    let source_y = t.iter().map(|t| 1. + 2. * t + rng.normal()).collect::<Vec<_>>();
    let target_y = t.iter().map(|t| 1. + 3. * t + rng.normal()).collect::<Vec<_>>();
    let mut source = request(t.clone(), None, source_y, draws);
    let mut prior = PriorSet::new();
    prior.push(PriorSpec::GaussianCoefficients(GaussianCoefficientPrior::isotropic(2, 10.)));
    prior.push(PriorSpec::KnownResidualVariance(1.));
    source.inference = source.inference.prior(prior);
    let context = ExecutionContext::for_tests(grid_seed(0x92bc_0000 + rep));
    let mut session = BayesianSession::new();
    execute_bayesian_with_receipt(&mut session, &source, &context).unwrap();
    let posterior = session.posterior().unwrap();
    let coefficients = (0..2)
        .map(|index| {
            let col = posterior
                .draws
                .schema
                .quantities
                .iter()
                .position(
                    |q| matches!(q,PosteriorQuantityKind::Coefficient{index:i,..} if *i==index),
                )
                .unwrap();
            posterior.draws.column(col).unwrap()
        })
        .collect::<Vec<_>>();
    let means = coefficients
        .iter()
        .map(|column| column.iter().sum::<f64>() / draws as f64)
        .collect::<Vec<_>>();
    let covariance = (0..2)
        .map(|i| {
            (0..2)
                .map(|j| {
                    coefficients[i]
                        .iter()
                        .zip(coefficients[j])
                        .map(|(x, y)| (x - means[i]) * (y - means[j]))
                        .sum::<f64>()
                        / (draws - 1) as f64
                })
                .collect()
        })
        .collect();
    let design = t.iter().map(|t| vec![1., *t]).collect::<Vec<_>>();
    let oracle = nig(&design, &target_y, &means, &inverse(covariance));
    let bytes = session.export_prior_source().unwrap();
    let mut same = source.clone();
    same.inference.prior = None;
    same.inference = same
        .inference
        .prior_from_artifact(bytes.clone(), Some(PriorMapping::IdenticalCoefficientSubspace));
    assert!(matches!(
        execute_bayesian_with_receipt(&mut BayesianSession::new(), &same, &context),
        Err(RecalcRunError::Request("recalc.bayesian_prior_likelihood_double_use"))
    ));
    let mut target = request(t, None, target_y, draws);
    target.inference = target
        .inference
        .prior_from_artifact(bytes, Some(PriorMapping::IdenticalCoefficientSubspace));
    let target_context = ExecutionContext::for_tests(grid_seed(0x93bc_0000 + rep));
    errors(target, &oracle, &target_context)
}
