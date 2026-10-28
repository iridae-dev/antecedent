//! Additive-GAM plug-in Jacobian and directional derivative against exact rational fits.
//!
//! `oracle_data/estimate_response_gam_derivative_reference.py` (frozen in the JSON beside it) fits
//! the outcome by the unique minimizer of the penalized additive cubic-spline problem (treatment
//! smooths unpenalized, adjuster smooths at the caller's lambda, type-7 knots) in exact arithmetic,
//! differentiates each treatment smooth by the degree-drop formula (zero outside the half-open knot
//! interior), and evaluates the plug-in level as the covariate average of the fitted function. Cases:
//! one treatment at interior points, at the left edge, at the open right edge and outside the range;
//! two treatments with two outcomes (Jacobian, every scale, and a directional derivative); a
//! treatment offset of 1e3 with an outcome offset of 1e5. The route publishes no interval, which is
//! asserted, as are the refusal of differential missingness across outcomes and of a log scale at a
//! non-positive level.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

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

fn close(got: f64, want: f64, rel: f64, abs: f64, what: &str) {
    assert!(
        (got - want).abs() <= rel * want.abs() + abs,
        "{what}: got {got:.12e} want {want:.12e}"
    );
}

fn run(
    estimator: &ContinuousResponseEstimator,
    data: &TabularData,
    functional: ResponseFunctional,
) -> Result<antecedent_core::CausalResponse, antecedent_estimate::EstimationError> {
    estimator.estimate_identified(
        data,
        &ResponseQuery::new(functional),
        IdentificationStatus::NonparametricallyIdentified,
        AssumptionSet::new(),
    )
}

#[test]
fn the_plug_in_gradient_level_and_scales_match_the_exact_fits() {
    let frozen: Value = serde_json::from_str(include_str!(
        "oracle_data/estimate_response_gam_derivative_reference.json"
    ))
    .unwrap();
    for case in frozen["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let t = usize::try_from(case["treatments"].as_u64().unwrap()).unwrap();
        let y: Vec<Vec<f64>> = case["y"].as_array().unwrap().iter().map(floats).collect();
        let (a1, a2, x) = (floats(&case["a1"]), floats(&case["a2"]), floats(&case["x"]));
        // Column order: y1, y2, a1, [a2], x.
        let mut columns: Vec<(&str, &[f64])> =
            vec![("y1", y[0].as_slice()), ("y2", y[1].as_slice()), ("a1", a1.as_slice())];
        if t == 2 {
            columns.push(("a2", a2.as_slice()));
        }
        columns.push(("x", x.as_slice()));
        let data = TabularData::from_f64_columns(columns.iter().copied()).unwrap();
        let outcomes: Arc<[VariableId]> =
            Arc::from([VariableId::from_raw(0), VariableId::from_raw(1)]);
        let treatments: Arc<[VariableId]> =
            (0..t).map(|j| VariableId::from_raw(2 + u32::try_from(j).unwrap())).collect();
        let adjuster = VariableId::from_raw(2 + u32::try_from(t).unwrap());
        let mut estimator = ContinuousResponseEstimator::new([adjuster]);
        estimator.options.nuisance_basis =
            usize::try_from(case["basis"].as_u64().unwrap()).unwrap();
        estimator.options.nuisance_lambda = case["lambda"].as_f64().unwrap();
        // The offset design carries an outcome of 1e5: the backfit stops relative to it.
        let (rel, abs) = if name == "offsets_and_scales" { (3e-5, 2e-4) } else { (1e-6, 0.0) };
        for point in case["points"].as_array().unwrap() {
            let at: Vec<f64> =
                point["at"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
            let gradient: Vec<Vec<f64>> =
                point["gradient"].as_array().unwrap().iter().map(floats).collect();
            let level = floats(&point["level"]);
            for scale_name in ["identity", "log_treatment", "log_outcome", "log_log"] {
                let scale = match scale_name {
                    "identity" => DerivativeScale::Identity,
                    "log_treatment" => DerivativeScale::LogTreatment,
                    "log_outcome" => DerivativeScale::LogOutcome,
                    _ => DerivativeScale::LogLog,
                };
                let label = format!("{name} at {at:?} {scale_name}");
                let logs_a_treatment =
                    scale != DerivativeScale::Identity && scale != DerivativeScale::LogOutcome;
                if logs_a_treatment && at.iter().any(|v| *v <= 0.0) {
                    continue; // the query itself requires a positive treatment point
                }
                let result = run(
                    &estimator,
                    &data,
                    ResponseFunctional::Jacobian {
                        outcomes: Arc::clone(&outcomes),
                        treatments: Arc::clone(&treatments),
                        at: Arc::from(at.clone()),
                        scale,
                    },
                );
                let needs_level =
                    matches!(scale, DerivativeScale::LogOutcome | DerivativeScale::LogLog);
                if needs_level && level.iter().any(|l| *l <= 0.0) {
                    let error = result
                        .err()
                        .unwrap_or_else(|| panic!("{label}: a log scale at a non-positive level"));
                    assert!(
                        error.to_string().contains("positive fitted response"),
                        "{label}: {error}"
                    );
                    continue;
                }
                let response = result.unwrap_or_else(|e| panic!("{label}: {e}"));
                let ResponseIdentification::PointIdentified(ResponseValue::Jacobian {
                    values, ..
                }) = &response.estimate
                else {
                    panic!("{label}: a Jacobian");
                };
                assert!(matches!(response.uncertainty, ResponseUncertainty::None), "{label}");
                assert!(
                    response
                        .support
                        .warnings
                        .iter()
                        .any(|w| w.code.as_ref() == "response.derivative_interval_withheld"),
                    "{label}: the withheld interval is disclosed"
                );
                for (o, row) in gradient.iter().enumerate() {
                    for (j, g) in row.iter().enumerate() {
                        let factor = match scale {
                            DerivativeScale::Identity => *g,
                            DerivativeScale::LogTreatment => at[j] * g,
                            DerivativeScale::LogOutcome => g / level[o],
                            DerivativeScale::LogLog => at[j] * g / level[o],
                        };
                        close(values[o * t + j], factor, rel, abs, &format!("{label} [{o},{j}]"));
                    }
                }
            }
            // The directional derivative is the gradient dotted with the direction.
            let direction: Vec<f64> = (0..t).map(|j| if j == 0 { 0.6 } else { -0.8 }).collect();
            let response = run(
                &estimator,
                &data,
                ResponseFunctional::DirectionalDerivative {
                    outcomes: Arc::clone(&outcomes),
                    treatments: Arc::clone(&treatments),
                    at: Arc::from(at.clone()),
                    direction: Arc::from(direction.clone()),
                },
            )
            .unwrap_or_else(|e| panic!("{name} directional: {e}"));
            let ResponseIdentification::PointIdentified(ResponseValue::Vector(values)) =
                &response.estimate
            else {
                panic!("{name}: a directional derivative is a vector");
            };
            for (o, row) in gradient.iter().enumerate() {
                let want: f64 = row.iter().zip(&direction).map(|(g, d)| g * d).sum();
                close(values[o], want, rel, abs, &format!("{name} at {at:?} directional {o}"));
            }
        }
    }
    let _ = num;
}

#[test]
fn differential_missingness_across_outcomes_is_refused() {
    let n = 60usize;
    let x: Vec<f64> = (0..n).map(|i| ((i * 37) % 60) as f64 / 30.0 - 1.0).collect();
    let a: Vec<f64> = (0..n).map(|i| 0.5 * x[i] + ((i * 53) % 60) as f64 / 120.0 - 0.25).collect();
    let y1: Vec<f64> =
        (0..n).map(|i| 1.5 * a[i] + 0.8 * x[i] + ((i * 29) % 60) as f64 / 120.0).collect();
    let mut y2 = y1.clone();
    y2[7] = f64::NAN;
    let data = TabularData::from_f64_columns([
        ("y1", y1.as_slice()),
        ("y2", y2.as_slice()),
        ("a", a.as_slice()),
        ("x", x.as_slice()),
    ])
    .unwrap();
    let estimator = ContinuousResponseEstimator::new([VariableId::from_raw(3)]);
    let error = run(
        &estimator,
        &data,
        ResponseFunctional::Jacobian {
            outcomes: Arc::from([VariableId::from_raw(0), VariableId::from_raw(1)]),
            treatments: Arc::from([VariableId::from_raw(2)]),
            at: Arc::from([0.1]),
            scale: DerivativeScale::Identity,
        },
    )
    .err()
    .expect("a shared complete-case row set is required across outcomes");
    assert!(error.to_string().contains("shared complete-case row set"), "{error}");
}
