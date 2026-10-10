//! Frozen exact-quadratic randomized-dose design. D~Uniform[0,4] and independent
//! N(0,1) outcome noise with mean1+.5d+.25d². The declared conditional mean is
//! exactly quadratic, so smoothing bias is zero at fixed bandwidth .6. HC0 local
//! covariance and shared-row contrast covariance are refitted per independent sample.
//! Level at2, derivative at2 and contrast1→3 are separate pointwise claims. No band,
//! unrestricted smoothing-bias license or calibration run occurs in ordinary tests.
#![allow(clippy::cast_precision_loss, reason = "bounded simulation dimensions")]
#[path = "../../antecedent/tests/common/calibration.rs"]
mod calibration;
#[path = "common/candidate_calibration.rs"]
mod candidate;
use antecedent_estimate::dose_grid_functional::{
    ClaimSet, DoseDesign, DoseFunctional, DoseGridRequest, estimate_dose_grid_quadratic_mean,
};
use calibration::{CoverageTally, RecordKey, grid_n, grid_seed, map_replicates, n_sim};
fn measure(test: &'static str, expected_id: &str, functional: DoseFunctional, truth: f64) {
    let n = grid_n(500);
    let mut tally = CoverageTally::for_record(
        RecordKey {
            test,
            dgp: "randomized_uniform_quadratic_mean_gaussian_errors",
            interval: "analytic_se",
        },
        0.95,
    );
    let results = map_replicates(n_sim(), |rep| {
        let mut rng = candidate::Generator::new(grid_seed(0x47ab_0000 + rep));
        let (mut dose, mut outcome) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for _ in 0..n {
            let d = 4. * rng.uniform();
            dose.push(d);
            outcome.push(1. + 0.5 * d + 0.25 * d * d + rng.normal());
        }
        estimate_dose_grid_quadratic_mean(&DoseGridRequest {
            dose: &dose,
            outcome: &outcome,
            grid: &[2.],
            bandwidth: 0.6,
            bandwidth_range: (0.1, 1.),
            minimum_local_ess: 30.,
            functional,
            claims: ClaimSet {
                pointwise_level: !matches!(functional, DoseFunctional::Derivative),
                derivative: matches!(functional, DoseFunctional::Derivative),
                simultaneous_band: false,
            },
            design: DoseDesign::RandomizedDose,
        })
    });
    for result in results {
        match result {
            Ok(row) => {
                candidate::bind(
                    &mut tally,
                    &row.quadratic_calibration_basis(0).expect("actual zero-bias candidate"),
                );
                let interval = match functional {
                    DoseFunctional::Level => row.levels[0].interval,
                    DoseFunctional::Derivative => row.derivatives[0].interval,
                    DoseFunctional::Contrast { .. } => row.contrast.unwrap().interval,
                };
                tally.record(Some((interval.lower, interval.upper)), truth);
            }
            Err(_) => tally.skip(),
        }
    }
    assert_eq!(tally.record_id().as_deref(), Some(expected_id));
    tally.assert();
}
#[test]
#[ignore = "calibration: final measurement only"]
fn randomized_exact_quadratic_level_l95() {
    measure(
        "randomized_exact_quadratic_level_l95",
        "cov.dose_response.dag.frequentist.analytic_se.l95.randomized_exact_quadratic_level_l95",
        DoseFunctional::Level,
        3.,
    );
}
#[test]
#[ignore = "calibration: final measurement only"]
fn randomized_exact_quadratic_derivative_l95() {
    measure(
        "randomized_exact_quadratic_derivative_l95",
        "cov.dose_response.dag.frequentist.analytic_se.l95.randomized_exact_quadratic_derivative_l95",
        DoseFunctional::Derivative,
        1.5,
    );
}
#[test]
#[ignore = "calibration: final measurement only"]
fn randomized_exact_quadratic_contrast_l95() {
    measure(
        "randomized_exact_quadratic_contrast_l95",
        "cov.dose_response.dag.frequentist.analytic_se.l95.randomized_exact_quadratic_contrast_l95",
        DoseFunctional::Contrast { from: 1., to: 3. },
        3.,
    );
}
