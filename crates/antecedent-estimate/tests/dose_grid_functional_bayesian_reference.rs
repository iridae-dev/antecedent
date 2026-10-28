//! The joint polynomial dose-grid posterior against a 50-digit conjugate reference.
//!
//! `oracle_data/dose_grid_functional_bayesian_reference.py` (frozen in the JSON beside it) rescales
//! the dose to [-1, 1] on the declared support, forms the ridge posterior of the polynomial
//! coefficients (mean `(X'X + diag(1/s^2))^-1 X'y`, covariance `v (X'X + diag(1/s^2))^-1` for a
//! known residual variance and `b_n/(a_n - 1)` times that inverse for an inverse-gamma prior) and
//! pushes it through the linear map of a level, a derivative or a two-dose contrast. The cases
//! include doses and support offset by 1e6 and -3e5, more coefficients than rows, the smallest
//! admitted sample, a support wider than the data and prior scales from 1e-3 to 1e6. The exact
//! moments are compared tightly; the retained draws, which carry the interval and the joint
//! covariance, are compared with the same moments to Monte Carlo accuracy.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::dose_basis_regression::ResidualPrior;
use antecedent_estimate::dose_grid_functional::{DoseDesign, DoseFunctional};
use antecedent_estimate::dose_grid_posterior::{
    DoseGridPosterior, DoseGridPosteriorRequest, estimate_dose_grid_posterior,
};
use serde_json::Value;

const DRAWS: usize = 100_000;

fn reference() -> Vec<Value> {
    serde_json::from_str::<Value>(include_str!(
        "oracle_data/dose_grid_functional_bayesian_reference.json"
    ))
    .expect("frozen reference parses")
    .as_array()
    .cloned()
    .expect("array")
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().expect("array").iter().map(|x| x.as_f64().expect("number")).collect()
}

struct Case {
    name: String,
    dose: Vec<f64>,
    outcome: Vec<f64>,
    grid: Vec<f64>,
    support: (f64, f64),
    scales: Vec<f64>,
    residual: ResidualPrior,
    functional: DoseFunctional,
    known: bool,
}

impl Case {
    fn new(case: &Value) -> Self {
        let kind = case["functional"].as_str().unwrap();
        let known = case.get("known_variance").is_some();
        Self {
            name: case["name"].as_str().unwrap().to_owned(),
            dose: floats(&case["dose"]),
            outcome: floats(&case["outcome"]),
            grid: case.get("grid").map(floats).unwrap_or_default(),
            support: (case["lo"].as_f64().unwrap(), case["hi"].as_f64().unwrap()),
            scales: floats(&case["scales"]),
            residual: if known {
                ResidualPrior::KnownVariance(case["known_variance"].as_f64().unwrap())
            } else {
                ResidualPrior::InverseGamma {
                    shape: case["ig_shape"].as_f64().unwrap(),
                    scale: case["ig_scale"].as_f64().unwrap(),
                }
            },
            functional: match kind {
                "level" => DoseFunctional::Level,
                "derivative" => DoseFunctional::Derivative,
                _ => DoseFunctional::Contrast {
                    from: case["from"].as_f64().unwrap(),
                    to: case["to"].as_f64().unwrap(),
                },
            },
            known,
        }
    }

    fn request(&self) -> DoseGridPosteriorRequest<'_> {
        DoseGridPosteriorRequest {
            dose: &self.dose,
            outcome: &self.outcome,
            grid: &self.grid,
            support: self.support,
            coefficient_scales: &self.scales,
            residual: self.residual,
            functional: self.functional,
            design: DoseDesign::RandomizedDose,
            draws: DRAWS,
            seed: 7,
        }
    }

    fn fit(&self) -> DoseGridPosterior {
        estimate_dose_grid_posterior(&self.request())
            .unwrap_or_else(|error| panic!("{}: {error:?}", self.name))
    }
}

#[test]
fn known_variance_exact_moments_equal_the_multiprecision_conjugate_posterior() {
    let mut seen = 0;
    for json in reference() {
        let case = Case::new(&json);
        if !case.known {
            continue;
        }
        let posterior = case.fit();
        let want_mean = floats(&json["expected"]["mean"]);
        let want_cov = floats(&json["expected"]["cov"]);
        let mean = posterior.exact_mean.as_ref().expect("exact mean under a known variance");
        let cov = posterior.exact_covariance.as_ref().expect("exact covariance");
        assert_eq!(mean.len(), want_mean.len(), "{}", case.name);
        // Doses offset by 1e6 lose about six digits in the rescaling to the support; the
        // diffuse and tight priors lose conditioning, not accuracy of the identity.
        let rel = if case.name.starts_with("offset") { 1e-5 } else { 1e-8 };
        for (g, (&got, &want)) in mean.iter().zip(&want_mean).enumerate() {
            let scale = want_mean.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
            assert!(
                (got - want).abs() <= rel * scale.max(1e-12),
                "{} mean {g}: {got} {want}",
                case.name
            );
        }
        for (k, (&got, &want)) in cov.iter().zip(&want_cov).enumerate() {
            let scale = want_cov.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
            assert!(
                (got - want).abs() <= rel * scale.max(1e-300),
                "{} cov {k}: {got} {want}",
                case.name
            );
        }
        seen += 1;
    }
    assert_eq!(seen, 11);
}

#[test]
fn retained_draws_reproduce_the_reference_moments_and_the_joint_covariance() {
    for json in reference() {
        let case = Case::new(&json);
        let posterior = case.fit();
        let want_mean = floats(&json["expected"]["mean"]);
        let want_cov = floats(&json["expected"]["cov"]);
        let values = posterior.values;
        assert_eq!(values, want_mean.len(), "{}", case.name);
        for g in 0..values {
            let sd = want_cov[g * values + g].sqrt();
            // The posterior mean of 1e5 draws is within 5 standard errors of the reference.
            assert!(
                (posterior.mean[g] - want_mean[g]).abs()
                    <= 5.0 * sd / (DRAWS as f64).sqrt() + 1e-9 * want_mean[g].abs(),
                "{} mean {g}: {} {}",
                case.name,
                posterior.mean[g],
                want_mean[g]
            );
            for h in 0..values {
                let scale = (want_cov[g * values + g] * want_cov[h * values + h]).sqrt();
                let tolerance = if case.known { 0.03 } else { 0.06 } * scale;
                assert!(
                    (posterior.covariance[g * values + h] - want_cov[g * values + h]).abs()
                        <= tolerance,
                    "{} covariance {g},{h}: {} {}",
                    case.name,
                    posterior.covariance[g * values + h],
                    want_cov[g * values + h]
                );
            }
            // The equal-tailed interval brackets the mean and is wider than zero.
            let [lower, upper] = posterior.interval_95[g];
            assert!(lower < posterior.mean[g] && posterior.mean[g] < upper, "{}", case.name);
        }
        if !case.known {
            assert!(
                posterior.exact_mean.is_none() && posterior.exact_covariance.is_none(),
                "{}",
                case.name
            );
        }
    }
}

#[test]
fn gaussian_intervals_are_the_normal_quantiles_of_the_exact_moments() {
    for json in reference() {
        let case = Case::new(&json);
        if !case.known || !matches!(case.functional, DoseFunctional::Level) {
            continue;
        }
        let posterior = case.fit();
        let mean = posterior.exact_mean.as_ref().unwrap();
        let cov = posterior.exact_covariance.as_ref().unwrap();
        let values = posterior.values;
        for g in 0..values {
            let sd = cov[g * values + g].sqrt();
            let [lower, upper] = posterior.interval_95[g];
            assert!(
                ((upper - lower) / (2.0 * 1.959_963_984_540_054 * sd) - 1.0).abs() < 0.03,
                "{}",
                case.name
            );
            assert!(((lower + upper) / 2.0 - mean[g]).abs() < 0.03 * sd + 1e-9, "{}", case.name);
        }
    }
}

#[test]
fn declarations_are_used_or_refused_never_ignored() {
    let json = reference().into_iter().find(|c| c["name"] == "level_known").unwrap();
    let case = Case::new(&json);
    let base = case.fit();
    // The prior scales, the residual variance and the support change the answer.
    let mut scales = case.scales.clone();
    scales[2] = 0.01;
    let changed = estimate_dose_grid_posterior(&DoseGridPosteriorRequest {
        coefficient_scales: &scales,
        ..case.request()
    })
    .unwrap();
    assert_ne!(changed.exact_mean, base.exact_mean);
    let noisier = estimate_dose_grid_posterior(&DoseGridPosteriorRequest {
        residual: ResidualPrior::KnownVariance(4.0),
        ..case.request()
    })
    .unwrap();
    assert!(
        noisier.exact_covariance.as_ref().unwrap()[0] > base.exact_covariance.as_ref().unwrap()[0]
    );
    let widened = estimate_dose_grid_posterior(&DoseGridPosteriorRequest {
        support: (case.support.0 - 1.0, case.support.1 + 1.0),
        ..case.request()
    })
    .unwrap();
    assert_ne!(widened.exact_mean, base.exact_mean);
    // The seed moves the draws and leaves the exact moments alone.
    let reseeded =
        estimate_dose_grid_posterior(&DoseGridPosteriorRequest { seed: 8, ..case.request() })
            .unwrap();
    assert_eq!(reseeded.exact_mean, base.exact_mean);
    assert_ne!(reseeded.draws, base.draws);
    // An observational dose is refused, not adjusted for.
    assert!(
        estimate_dose_grid_posterior(&DoseGridPosteriorRequest {
            design: DoseDesign::Observational,
            ..case.request()
        })
        .is_err()
    );
    // A grid dose outside the observed range, a dose outside the support, a derivative of a
    // constant curve, too few rows, too few draws and a bad prior scale are all refused.
    let outside = [case.dose[0] - 0.01];
    assert!(
        estimate_dose_grid_posterior(&DoseGridPosteriorRequest {
            grid: &outside,
            ..case.request()
        })
        .is_err()
    );
    let narrow = (case.dose[1], case.dose[2]);
    assert!(
        estimate_dose_grid_posterior(&DoseGridPosteriorRequest {
            support: narrow,
            ..case.request()
        })
        .is_err()
    );
    let constant = [3.0];
    assert!(
        estimate_dose_grid_posterior(&DoseGridPosteriorRequest {
            coefficient_scales: &constant,
            functional: DoseFunctional::Derivative,
            ..case.request()
        })
        .is_err()
    );
    assert!(
        estimate_dose_grid_posterior(&DoseGridPosteriorRequest {
            dose: &case.dose[..2],
            outcome: &case.outcome[..2],
            ..case.request()
        })
        .is_err()
    );
    assert!(
        estimate_dose_grid_posterior(&DoseGridPosteriorRequest { draws: 10, ..case.request() })
            .is_err()
    );
    let bad = [3.0, -2.0, 1.0];
    assert!(
        estimate_dose_grid_posterior(&DoseGridPosteriorRequest {
            coefficient_scales: &bad,
            ..case.request()
        })
        .is_err()
    );
}
