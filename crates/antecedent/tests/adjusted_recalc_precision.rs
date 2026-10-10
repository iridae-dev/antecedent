//! Full-fit retained adjusted OLS and logistic delta-variance diagnostics.
//! Balanced fixed (Z,T) design, independently generated outcome noise. Target Z
//! masses are supplied/fixed (.25,.75); no estimated-target-law variance claim.
//! All repetitions fit the checked adapter, then retarget with zero model solves.
//! No confidence intervals or coverage IDs are manufactured by these diagnostics.
#![allow(clippy::cast_precision_loss, reason = "bounded simulation dimensions")]
#[path = "../../antecedent-estimate/tests/common/candidate_calibration.rs"]
mod candidate;
mod common;
use antecedent::analysis::recalc_adjusted::{
    AdjustedContrast, AdjustedModel, AdjustedRequest, AdjustedSession,
    execute_adjusted_with_receipt,
};
use antecedent::analysis::recalc_receipt::{TargetWeights, UtilitySpec};
use antecedent_core::ExecutionContext;
use antecedent_estimate::vector_treatment::VectorCovariance;
use antecedent_stats::{GlmFamily, GlmOptions};
use common::calibration;
use common::calibration::{grid_n, grid_seed, n_sim};

fn inverse(m: [f64; 9]) -> [f64; 9] {
    let [a, b, c, _, d, e, _, _, f] = m;
    let det = a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d);
    assert!(det > 0.);
    [
        d * f - e * e,
        c * e - b * f,
        b * e - c * d,
        c * e - b * f,
        a * f - c * c,
        b * c - a * e,
        b * e - c * d,
        b * c - a * e,
        a * d - b * b,
    ]
    .map(|x| x / det)
}
fn logistic_oracle(n: usize, mass: f64) -> (f64, f64, [f64; 9]) {
    let cells = [(0., 0., 0.25), (0., 1., 0.5), (1., 0., 0.4), (1., 1., 2. / 3.)];
    let mut info = [0.; 9];
    for (z, t, p) in cells {
        let x = [1., z, t];
        for i in 0..3 {
            for j in 0..3 {
                info[3 * i + j] += n as f64 / 4. * p * (1. - p) * x[i] * x[j];
            }
        }
    }
    let cov = inverse(info);
    let g = [
        (1. - mass) * (0.25 - 3. / 16.) + mass * (2. / 9. - 6. / 25.),
        mass * (2. / 9. - 6. / 25.),
        (1. - mass) * 0.25 + mass * 2. / 9.,
    ];
    let variance = (0..3).flat_map(|i| (0..3).map(move |j| g[i] * cov[3 * i + j] * g[j])).sum();
    ((1. - mass) * 0.25 + mass * (2. / 3. - 0.4), variance, cov)
}
fn request(n: usize, seed: u64, logistic: bool) -> AdjustedRequest {
    let mut rng = candidate::Generator::new(seed);
    let (mut z, mut t, mut y) =
        (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    for i in 0..n {
        let a = f64::from(i % 2 == 1);
        let b = f64::from((i / 2) % 2 == 1);
        let p = 1. / (1. + (-(-3_f64.ln() + b * 2_f64.ln() + a * 3_f64.ln())).exp());
        t.push(a);
        z.push(b);
        y.push(if logistic { rng.binary(p) as f64 } else { 1. + 2. * a + 0.3 * b + rng.normal() });
    }
    AdjustedRequest {
        columns: vec![("t".into(), t), ("y".into(), y), ("z".into(), z)],
        edges: vec![(2, 0), (2, 1), (0, 1)],
        treatments: vec![0],
        outcome: 1,
        adjustment: vec![2],
        model: if logistic {
            AdjustedModel::Glm { family: GlmFamily::BinomialLogit, options: GlmOptions::default() }
        } else {
            AdjustedModel::Linear { covariance: VectorCovariance::ModelBased }
        },
        contrast: AdjustedContrast::Numeric { active: vec![1.], control: vec![0.] },
        target: None,
        utility: UtilitySpec { benefit_per_unit: 1., cost: 0. },
    }
}
fn measure(test: &str, logistic: bool) {
    let n = grid_n(1024);
    assert_eq!(n % 4, 0);
    let attempts = n_sim().max(2000);
    let mut sums = [[0.; 3]; 2];
    let mut covariance = [0.; 9];
    let oracle_cov = if logistic {
        logistic_oracle(n, 0.25).2
    } else {
        inverse([
            n as f64,
            n as f64 / 2.,
            n as f64 / 2.,
            n as f64 / 2.,
            n as f64 / 2.,
            n as f64 / 4.,
            n as f64 / 2.,
            n as f64 / 4.,
            n as f64 / 2.,
        ])
    };
    for rep in 0..attempts {
        let seed = grid_seed(0xa631_0000 + u64::from(rep));
        let mut r = request(n, seed, logistic);
        let mut s = AdjustedSession::new();
        let ctx = ExecutionContext::for_tests(seed);
        let first = execute_adjusted_with_receipt(&mut s, &r, &ctx)
            .expect("every fixed full-rank source fit must succeed");
        assert_eq!(first.receipt.totals().model_fits, 1);
        for (total, actual) in covariance.iter_mut().zip(s.covariance().unwrap()) {
            *total += actual;
        }
        for (index, mass) in [0.25, 0.75].into_iter().enumerate() {
            r.target = Some(TargetWeights {
                depends_on: vec![antecedent_core::VariableId::from_raw(2)],
                weights: r.columns[2]
                    .1
                    .iter()
                    .map(|&z| if z > 0. { mass * 2. } else { (1. - mass) * 2. })
                    .collect(),
            });
            let out = execute_adjusted_with_receipt(&mut s, &r, &ctx)
                .expect("fixed target retarget must succeed");
            assert_eq!(out.receipt.totals().model_fits, 0);
            let truth = if logistic { logistic_oracle(n, mass).0 } else { 2. };
            let error = out.law.ate - truth;
            sums[index][0] += error;
            sums[index][1] += error * error;
            sums[index][2] += out.law.std_error.powi(2);
        }
    }
    let reps = f64::from(attempts);
    for (actual, truth) in covariance.into_iter().zip(oracle_cov) {
        let scale = (oracle_cov[0] * oracle_cov[8]).sqrt();
        assert!((actual / reps - truth).abs() / scale < 0.15, "full covariance precision");
    }
    for (index, mass) in [0.25, 0.75].into_iter().enumerate() {
        let truth_var = if logistic { logistic_oracle(n, mass).1 } else { 4. / n as f64 };
        let bias = sums[index][0] / reps;
        let empirical = sums[index][1] / reps - bias * bias;
        let estimated = sums[index][2] / reps;
        assert!(bias.abs() < 5. * (truth_var / reps).sqrt(), "fixed target bias");
        assert!((empirical / truth_var - 1.).abs() < 0.15, "sampling variance");
        assert!((estimated / truth_var - 1.).abs() < 0.15, "reported SE squared");
        eprintln!(
            "INFERENCE_DIAGNOSTIC_RECORD test={test} n={n} repetitions={attempts} target_z_mass={mass} failed=0 bias={bias} empirical_variance={empirical} reported_variance={estimated} oracle_variance={truth_var}"
        );
    }
}
#[test]
#[ignore = "calibration: final measurement only"]
fn adjusted_ols_retained_covariance_precision() {
    measure("adjusted_ols_retained_covariance_precision", false);
}
#[test]
#[ignore = "calibration: final measurement only"]
fn adjusted_logit_fixed_target_delta_precision() {
    measure("adjusted_logit_fixed_target_delta_precision", true);
}
