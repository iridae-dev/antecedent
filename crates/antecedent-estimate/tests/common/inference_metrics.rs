//! Frozen diagnostic measurements, distinct from interval coverage and activation.
//! Gate logs retain actual counts and metrics; no audit file or covariance CI is minted.
#![allow(clippy::cast_precision_loss, reason = "bounded calibration counts")]
use serde_json::json;

pub struct FitMetrics {
    pub rows: usize,
    pub estimates: [f64; 2],
    pub covariance: [f64; 4],
    pub joint_p: f64,
    pub contrast_p_holm: [f64; 3],
    pub contrast_variance: [f64; 3],
}
#[derive(Clone, Copy)]
pub struct MetricDesign<'a> {
    pub test: &'a str,
    pub file: &'a str,
    pub method: &'a str,
    pub n: usize,
    pub seed: u64,
    pub true_coefficients: [f64; 2],
    pub true_covariance: [f64; 4],
    pub true_contrasts: [f64; 3],
    pub true_contrast_variance: [f64; 3],
}
fn complete_sample<'a>(
    design: MetricDesign<'_>,
    attempts: usize,
    results: &'a [Option<FitMetrics>],
) -> Vec<&'a FitMetrics> {
    let failed = results.iter().filter(|fit| fit.is_none()).count();
    let fits: Vec<_> = results.iter().filter_map(Option::as_ref).collect();
    let invalid = fits
        .iter()
        .filter(|fit| {
            fit.rows != design.n
                || !fit
                    .estimates
                    .iter()
                    .chain(&fit.covariance)
                    .chain(&fit.contrast_variance)
                    .all(|v| v.is_finite())
                || fit.contrast_variance.iter().any(|v| *v <= 0.)
                || !std::iter::once(&fit.joint_p)
                    .chain(&fit.contrast_p_holm)
                    .all(|p| p.is_finite() && (0. ..=1.).contains(p))
        })
        .count();
    if failed > 0 || invalid > 0 || results.len() != attempts {
        println!(
            "INFERENCE_DIAGNOSTIC_RECORD {}",
            json!({"kind":"full_fit_regression_inference", "test":design.test,
            "method":design.method,"rows":design.n,"seed":design.seed,"attempts":attempts,"failed":failed,
            "invalid_outputs":invalid,"metrics":null,"pass":false,"activation":"none"})
        );
        panic!("full attempted sample incomplete; no conditional-success inference is assessed");
    }
    fits
}

/// Every attempted full fit participates. Any failed fit fails this regular-design
/// suite; it is not silently removed from the sampling covariance or error rate.
pub fn assess(design: MetricDesign<'_>, attempts: usize, results: &[Option<FitMetrics>]) {
    let fits = complete_sample(design, attempts, results);
    let r = attempts as f64;
    let alpha: f64 = 0.05;
    let slack = 3. * (alpha * (1. - alpha) / r).sqrt();
    let mut estimated_covariance = [0.; 4];
    let mut mean = [0.; 2];
    let mut estimated_contrast_variance = [0.; 3];
    let (mut joint_rejections, mut false_holm, mut complete_power) = (0, 0, 0);
    for fit in &fits {
        for (a, b) in mean.iter_mut().zip(fit.estimates) {
            *a += b / r;
        }
        for (a, b) in estimated_covariance.iter_mut().zip(fit.covariance) {
            *a += b / r;
        }
        for (a, b) in estimated_contrast_variance.iter_mut().zip(fit.contrast_variance) {
            *a += b / r;
        }
        joint_rejections += usize::from(fit.joint_p <= alpha);
        false_holm += usize::from(
            fit.contrast_p_holm
                .iter()
                .zip(design.true_contrasts)
                .any(|(p, truth)| truth.abs() < f64::EPSILON && *p <= alpha),
        );
        complete_power += usize::from(
            fit.contrast_p_holm
                .iter()
                .zip(design.true_contrasts)
                .filter(|(_, truth)| truth.abs() >= f64::EPSILON)
                .all(|(p, _)| *p <= alpha),
        );
    }
    let mut empirical_covariance = [0.; 4];
    for fit in &fits {
        for i in 0..2 {
            for j in 0..2 {
                empirical_covariance[2 * i + j] +=
                    (fit.estimates[i] - mean[i]) * (fit.estimates[j] - mean[j]) / (r - 1.);
            }
        }
    }
    let mut covariance_error: f64 = 0.;
    let mut empirical_error: f64 = 0.;
    for i in 0..2 {
        for j in 0..2 {
            let scale =
                (design.true_covariance[2 * i + i] * design.true_covariance[2 * j + j]).sqrt();
            covariance_error = covariance_error.max(
                (estimated_covariance[2 * i + j] - design.true_covariance[2 * i + j]).abs() / scale,
            );
            empirical_error = empirical_error.max(
                (empirical_covariance[2 * i + j] - design.true_covariance[2 * i + j]).abs() / scale,
            );
        }
    }
    let contrast_error = estimated_contrast_variance
        .iter()
        .zip(design.true_contrast_variance)
        .map(|(actual, truth)| (actual / truth - 1.).abs())
        .fold(0., f64::max);
    let mean_z = (0..2)
        .map(|i| {
            (mean[i] - design.true_coefficients[i]).abs()
                / (design.true_covariance[2 * i + i] / r).sqrt()
        })
        .fold(0., f64::max);
    let joint_rate = joint_rejections as f64 / r;
    let family_rate = false_holm as f64 / r;
    let power = complete_power as f64 / r;
    let global_null = design.true_coefficients.iter().all(|x| x.abs() < f64::EPSILON);
    let has_alternative = design.true_contrasts.iter().any(|x| x.abs() >= f64::EPSILON);
    let pass = covariance_error <= 0.15
        && empirical_error <= 0.15
        && contrast_error <= 0.15
        && mean_z <= 5.
        && family_rate <= alpha + slack
        && (if global_null { (joint_rate - alpha).abs() <= slack } else { joint_rate >= 0.9 })
        && (!has_alternative || power >= 0.9);
    println!(
        "INFERENCE_DIAGNOSTIC_RECORD {}",
        json!({
            "kind":"full_fit_regression_inference", "file":design.file,"test":design.test,
            "method":design.method,"rows":design.n,"seed":design.seed,"attempts":attempts,"failed":0,
            "truth_coefficients":design.true_coefficients,"truth_covariance":design.true_covariance,
            "estimated_covariance":estimated_covariance,"empirical_covariance":empirical_covariance,
            "estimated_contrast_variance":estimated_contrast_variance,"truth_contrast_variance":design.true_contrast_variance,
            "mean_standardized_error":mean_z,"joint_rejection_rate":joint_rate,"false_holm_rate":family_rate,
            "complete_alternative_power":if has_alternative {Some(power)} else {None},"alpha":alpha,"error_rate_slack":slack,
            "maximum_scaled_covariance_error":0.15,"maximum_relative_contrast_variance_error":0.15,
            "maximum_mean_standardized_error":5.,"minimum_strong_alternative_power":0.9,
            "pass":pass,"activation":"none; asymptotic point/diagnostic standing unchanged"
        })
    );
    assert!(
        pass,
        "{} {} n{}: full-fit diagnostic rules failed",
        design.test, design.method, design.n
    );
}
/// Conservative ordered-null test: failures count as rejection under the null
/// and as non-rejection under an alternative. All failures also fail the cell.
pub fn assess_monotonicity(
    test: &str,
    method: &str,
    n: usize,
    seed: u64,
    alternative: bool,
    results: &[Option<f64>],
) {
    let attempts = results.len();
    let failed = results.iter().filter(|x| x.is_none()).count();
    let invalid =
        results.iter().flatten().filter(|p| !p.is_finite() || !(0. ..=1.).contains(*p)).count();
    let rejected = results.iter().filter(|p| p.is_some_and(|p| p <= 0.05)).count()
        + if alternative { 0 } else { failed + invalid };
    let rate = rejected as f64 / attempts as f64;
    let bound = 0.05 + 3. * (0.05 * 0.95 / attempts as f64).sqrt();
    let pass = failed == 0 && invalid == 0 && if alternative { rate >= 0.9 } else { rate <= bound };
    println!(
        "INFERENCE_DIAGNOSTIC_RECORD {}",
        json!({"kind":"ordered_null_monotonicity",
        "test":test,"method":method,"rows":n,"seed":seed,"attempts":attempts,"failed":failed,"invalid_outputs":invalid,
        "alternative":alternative,"rejection_rate":rate,"null_upper_bound":bound,"minimum_strong_alternative_power":0.9,
        "pass":pass,"activation":"none; conservative diagnostic standing unchanged"})
    );
    assert!(pass, "{test} {method} n{n}: monotonicity rules failed");
}
