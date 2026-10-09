//! Frozen bootstrap-SE diagnostics, not confidence intervals or coverage claims.
#![allow(clippy::cast_precision_loss, reason = "bounded final measurement dimensions")]
use serde_json::json;
/// One full fit plus every actual inner bootstrap attempt.
pub struct Fit {
    pub estimates: Vec<f64>,
    pub standard_errors: Vec<f64>,
    pub inner_requested: usize,
    pub inner_succeeded: usize,
}
/// Compare reported bootstrap variances with independent full-fit sampling variance.
/// Every missing, invalid or incompletely bootstrapped result fails the cell.
pub fn assess(test: &str, rows: usize, seed: u64, truth: &[f64], results: &[Option<Fit>]) {
    let attempts = results.len();
    let failed = results.iter().filter(|r| r.is_none()).count();
    let fits: Vec<_> = results.iter().flatten().collect();
    let invalid = fits
        .iter()
        .filter(|f| {
            f.estimates.len() != truth.len()
                || f.standard_errors.len() != truth.len()
                || f.estimates.iter().any(|v| !v.is_finite())
                || f.standard_errors.iter().any(|v| !v.is_finite() || *v <= 0.)
                || f.inner_requested != 300
                || f.inner_succeeded != f.inner_requested
        })
        .count();
    if failed + invalid > 0 {
        println!(
            "INFERENCE_DIAGNOSTIC_RECORD {}",
            json!({"test":test,"rows":rows,"seed":seed,"attempts":attempts,"failed":failed,"invalid_or_incomplete_bootstrap":invalid,"pass":false,"failure_counting":"all attempts; no successful-fit conditioning"})
        );
        panic!("bootstrap precision cell contains failed or invalid full methods");
    }
    let r = attempts as f64;
    let means: Vec<_> =
        (0..truth.len()).map(|j| fits.iter().map(|f| f.estimates[j]).sum::<f64>() / r).collect();
    let empirical_variance: Vec<_> = (0..truth.len())
        .map(|j| fits.iter().map(|f| (f.estimates[j] - means[j]).powi(2)).sum::<f64>() / (r - 1.))
        .collect();
    let bootstrap_variance: Vec<_> = (0..truth.len())
        .map(|j| fits.iter().map(|f| f.standard_errors[j].powi(2)).sum::<f64>() / r)
        .collect();
    let relative_error = empirical_variance
        .iter()
        .zip(&bootstrap_variance)
        .map(|(e, b)| (b / e - 1.).abs())
        .fold(0., f64::max);
    let bias_mcse = means
        .iter()
        .zip(truth)
        .zip(&empirical_variance)
        .map(|((m, t), v)| (m - t).abs() / (v / r).sqrt())
        .fold(0., f64::max);
    let pass = relative_error <= 0.2 && bias_mcse <= 5.;
    println!(
        "INFERENCE_DIAGNOSTIC_RECORD {}",
        json!({"kind":"whole_method_bootstrap_standard_error_precision","test":test,"rows":rows,"seed":seed,"attempts":attempts,"failed":failed,"inner_requested_per_fit":300,"inner_succeeded_total":fits.iter().map(|f|f.inner_succeeded).sum::<usize>(),"truth":truth,"mean_estimate":means,"independent_empirical_variance":empirical_variance,"mean_reported_bootstrap_variance":bootstrap_variance,"maximum_relative_variance_error":relative_error,"maximum_bias_mcse":bias_mcse,"relative_variance_error_bound":0.2,"bias_mcse_bound":5.,"pass":pass,"activation":"none; standard errors remain unmeasured diagnostics without intervals"})
    );
    assert!(pass, "{test} n{rows}: bootstrap diagnostic rules failed");
}
