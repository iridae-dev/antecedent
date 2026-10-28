//! Point derivative of the Kennedy curve against exact local fits, bias correction and transforms.
//!
//! `oracle_data/estimate_response_point_derivative_reference.py` (frozen in the JSON beside it) takes
//! the cross-fitted pseudo-outcome and the covariate term of the exact Kennedy pipeline, fits at
//! 50 digits the Gaussian-kernel local polynomials of degree 2 (the conventional point), 3 (the
//! corrected slope) and 4 (the corrected level and curvature), forms their robust row influences and
//! the joint covariance of the corrected coordinates with the covariate marginalization term on the
//! level, and evaluates each scale's transform, its delta-method error from the numerical gradient of
//! the transform's definition, its Fieller interval for the elasticity, and the interval centred at
//! the corrected coordinate. Cases: a positive level, a level within sampling noise of zero, and a
//! treatment offset of 1e3 with an outcome offset of 1e5.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    AssumptionSet, DerivativeScale, IdentificationStatus, ResponseFunctional,
    ResponseIdentification, ResponseQuery, ResponseUncertainty, ResponseValue, VariableId,
};
use antecedent_data::TabularData;
use antecedent_estimate::response::ContinuousResponseEstimator;
use serde_json::Value;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_str().unwrap().parse().unwrap()).collect()
}

fn num(v: &Value) -> f64 {
    v.as_str().unwrap().parse().unwrap()
}

fn close(got: f64, want: f64, rel: f64, scale: f64, what: &str) {
    assert!(
        (got - want).abs() <= rel * want.abs() + scale,
        "{what}: got {got:.12e} want {want:.12e}"
    );
}

#[test]
fn derivatives_intervals_and_transforms_match_the_exact_local_fits() {
    let frozen: Value = serde_json::from_str(include_str!(
        "oracle_data/estimate_response_point_derivative_reference.json"
    ))
    .unwrap();
    for case in frozen["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let (a, x, y) = (floats(&case["a"]), floats(&case["x"]), floats(&case["y"]));
        let data = TabularData::from_f64_columns([
            ("a", a.as_slice()),
            ("y", y.as_slice()),
            ("x", x.as_slice()),
        ])
        .unwrap();
        let mut estimator = ContinuousResponseEstimator::new([VariableId::from_raw(2)]);
        estimator.options.folds = usize::try_from(case["folds"].as_u64().unwrap()).unwrap();
        estimator.options.nuisance_basis =
            usize::try_from(case["basis"].as_u64().unwrap()).unwrap();
        estimator.options.nuisance_lambda = case["lambda"].as_f64().unwrap();
        estimator.options.bandwidth = Some(case["bandwidth"].as_f64().unwrap());
        let at = case["at"].as_f64().unwrap();
        // The level within noise of zero amplifies the nuisance resolution into every scale that
        // divides by it.
        let rel = if name == "ordinary_positive_level" { 3e-5 } else { 3e-3 };
        // With an outcome of 1e5 the backfit stops at an absolute 1e-5 in the smooths (its scale is
        // the largest fitted value), which a second derivative at bandwidth 0.75 carries over.
        let floor: f64 = if name == "offset_treatment_and_outcome" { 1e-4 } else { 1e-12 };
        for result in case["results"].as_array().unwrap() {
            let order = u8::try_from(result["order"].as_u64().unwrap()).unwrap();
            let scale_name = result["scale"].as_str().unwrap();
            let scale = match scale_name {
                "identity" => DerivativeScale::Identity,
                "log_treatment" => DerivativeScale::LogTreatment,
                "log_outcome" => DerivativeScale::LogOutcome,
                _ => DerivativeScale::LogLog,
            };
            let label = format!("{name} order {order} {scale_name}");
            let query = ResponseQuery::new(ResponseFunctional::PointDerivative {
                outcome: VariableId::from_raw(1),
                treatment: VariableId::from_raw(0),
                at,
                order,
                scale,
            });
            let response = match estimator.estimate_identified(
                &data,
                &query,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
            ) {
                Ok(response) => response,
                Err(error) => {
                    // A scale that needs a positive level is refused when the fit is not.
                    assert!(
                        error.to_string().contains("positive fitted response"),
                        "{label}: {error}"
                    );
                    continue;
                }
            };
            let ResponseIdentification::PointIdentified(ResponseValue::Scalar(estimate)) =
                &response.estimate
            else {
                panic!("{label}: a point derivative is a scalar");
            };
            let scale_of = num(&result["estimate"]).abs().max(num(&result["center"]).abs());
            close(
                *estimate,
                num(&result["estimate"]),
                rel,
                floor + 1e-12 * scale_of,
                &format!("{label} estimate"),
            );
            match (&response.uncertainty, result.get("fieller")) {
                (ResponseUncertainty::Scalar { standard_error, lower, upper, .. }, fieller) => {
                    close(
                        *standard_error,
                        num(&result["se"]),
                        rel * 10.0,
                        0.0,
                        &format!("{label} se"),
                    );
                    let (want_low, want_up) = match fieller.filter(|f| !f.is_null()) {
                        Some(f) => (num(&f[0]), num(&f[1])),
                        None => {
                            let i = &result["interval"];
                            (num(&i[0]), num(&i[1]))
                        }
                    };
                    close(
                        *lower,
                        want_low,
                        rel * 10.0,
                        floor + scale_of * 1e-3 * rel,
                        &format!("{label} lower"),
                    );
                    close(
                        *upper,
                        want_up,
                        rel * 10.0,
                        floor + scale_of * 1e-3 * rel,
                        &format!("{label} upper"),
                    );
                }
                (ResponseUncertainty::None, Some(f)) if f.is_null() => {}
                (other, _) => panic!("{label}: unexpected uncertainty {other:?}"),
            }
        }
    }
}
