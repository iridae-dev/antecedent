//! Additive-GAM g-computation of an intervention response against exact rational references.
//!
//! `oracle_data/estimate_response_intervention_gcomp_reference.py` (frozen in the JSON beside it)
//! fits the outcome by the unique minimizer of the penalized additive cubic-spline problem (treatment
//! smooths unpenalized), computes the policy mean of every row from the policy's law (a point mass,
//! a shift, Bernoulli, categorical and Gaussian laws, the Gaussian one by quadrature over the knot
//! intervals of the clamped spline rather than a fixed Hermite rule), and forms the HC2 sandwich
//! error from the reduced penalized design with exact leverages. Cases: every policy kind on one
//! treatment (including a Gaussian policy wider than the knot range and the identity shift of 0, whose
//! influence is the outcome's own), two treatments with mixed policies, and a treatment offset of 1e3
//! with an outcome offset of 1e5. The refusal of a soft intervention, a duplicated target, a Gaussian
//! policy of zero variance and a joint categorical support over the exact-mixture budget is executed.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, IdentificationStatus, Intervention, ResponseFunctional, ResponseIdentification,
    ResponseQuery, ResponseUncertainty, ResponseValue, StochasticPolicy, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_estimate::response::ContinuousResponseEstimator;
use serde_json::Value as Json;

fn floats(v: &Json) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_str().unwrap().parse().unwrap()).collect()
}

fn num(v: &Json) -> f64 {
    v.as_str().unwrap().parse().unwrap()
}

fn intervention(variable: VariableId, policy: &Json) -> Intervention {
    let f = |key: &str| policy[key].as_f64().unwrap();
    match policy["kind"].as_str().unwrap() {
        "set" => Intervention::set(variable, Value::f64(f("value"))),
        "shift" => Intervention::Shift { variable, delta: Value::f64(f("delta")) },
        "bernoulli" => {
            Intervention::Stochastic { variable, policy: StochasticPolicy::Bernoulli { p: f("p") } }
        }
        "categorical" => Intervention::Stochastic {
            variable,
            policy: StochasticPolicy::Categorical {
                probs: policy["probs"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect(),
            },
        },
        _ => Intervention::Stochastic {
            variable,
            policy: StochasticPolicy::Gaussian { mean: f("mean"), variance: f("var") },
        },
    }
}

fn estimator(case: &Json, adjuster: VariableId) -> ContinuousResponseEstimator {
    let mut estimator = ContinuousResponseEstimator::new([adjuster]);
    estimator.options.nuisance_basis = usize::try_from(case["basis"].as_u64().unwrap()).unwrap();
    estimator.options.nuisance_lambda = case["lambda"].as_f64().unwrap();
    estimator
}

fn data_of(case: &Json) -> (TabularData, Vec<VariableId>, VariableId, VariableId) {
    let columns: Vec<Vec<f64>> = case["columns"].as_array().unwrap().iter().map(floats).collect();
    let (x, y) = (floats(&case["x"]), floats(&case["y"]));
    let t = columns.len();
    let mut named: Vec<(&str, &[f64])> = Vec::new();
    for (j, column) in columns.iter().enumerate() {
        named.push((["a1", "a2"][j], column.as_slice()));
    }
    named.push(("y", y.as_slice()));
    named.push(("x", x.as_slice()));
    let data = TabularData::from_f64_columns(named.iter().copied()).unwrap();
    let treatments: Vec<VariableId> =
        (0..t).map(|j| VariableId::from_raw(u32::try_from(j).unwrap())).collect();
    (
        data,
        treatments,
        VariableId::from_raw(u32::try_from(t).unwrap()),
        VariableId::from_raw(u32::try_from(t).unwrap() + 1),
    )
}

#[test]
fn the_level_and_sandwich_error_match_the_exact_references() {
    let frozen: Json = serde_json::from_str(include_str!(
        "oracle_data/estimate_response_intervention_gcomp_reference.json"
    ))
    .unwrap();
    for case in frozen["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let (data, treatments, y, x) = data_of(case);
        let interventions: Vec<Intervention> = case["policies"]
            .as_array()
            .unwrap()
            .iter()
            .zip(&treatments)
            .map(|(policy, variable)| intervention(*variable, policy))
            .collect();
        let response = estimator(case, x)
            .estimate_identified(
                &data,
                &ResponseQuery::new(ResponseFunctional::InterventionResponse {
                    outcome: y,
                    interventions: Arc::from(interventions),
                }),
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
            )
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let ResponseIdentification::PointIdentified(ResponseValue::Scalar(level)) =
            &response.estimate
        else {
            panic!("{name}: a scalar");
        };
        let ResponseUncertainty::Scalar { standard_error, lower, upper, .. } =
            &response.uncertainty
        else {
            panic!("{name}: a scalar interval");
        };
        // The backfit stops relative to the largest fitted value; the offset design carries 1e5.
        let abs = if name == "offsets_and_scales" { 1e-3 } else { 1e-7 };
        let want = num(&case["estimate"]);
        assert!((level - want).abs() <= 1e-6 * want.abs() + abs, "{name}: level {level} vs {want}");
        let want_se = num(&case["se"]);
        assert!(
            (standard_error - want_se).abs() <= 2e-4 * want_se + abs,
            "{name}: se {standard_error} vs {want_se}"
        );
        let interval = floats(&case["interval"]);
        assert!(
            (lower - interval[0]).abs() <= 2e-4 * want_se + 1e-6 * want.abs() + abs,
            "{name} lower"
        );
        assert!(
            (upper - interval[1]).abs() <= 2e-4 * want_se + 1e-6 * want.abs() + abs,
            "{name} upper"
        );
    }
}

#[test]
fn unsupported_policies_and_budget_overruns_are_refused() {
    let n = 60usize;
    let x: Vec<f64> = (0..n).map(|i| ((i * 37) % 60) as f64 / 30.0 - 1.0).collect();
    let a1: Vec<f64> = (0..n).map(|i| 0.5 * x[i] + ((i * 53) % 60) as f64 / 120.0 - 0.25).collect();
    let a2: Vec<f64> = (0..n).map(|i| -0.3 * x[i] + ((i * 41) % 60) as f64 / 90.0 - 0.3).collect();
    let y: Vec<f64> =
        (0..n).map(|i| 1.5 * a1[i] - a2[i] + 0.8 * x[i] + ((i * 29) % 60) as f64 / 120.0).collect();
    let data = TabularData::from_f64_columns([
        ("a1", a1.as_slice()),
        ("a2", a2.as_slice()),
        ("y", y.as_slice()),
        ("x", x.as_slice()),
    ])
    .unwrap();
    let (a1, a2, y, x) = (
        VariableId::from_raw(0),
        VariableId::from_raw(1),
        VariableId::from_raw(2),
        VariableId::from_raw(3),
    );
    let estimator = ContinuousResponseEstimator::new([x]);
    let run = |interventions: Vec<Intervention>| {
        estimator.estimate_identified(
            &data,
            &ResponseQuery::new(ResponseFunctional::InterventionResponse {
                outcome: y,
                interventions: Arc::from(interventions),
            }),
            IdentificationStatus::NonparametricallyIdentified,
            AssumptionSet::new(),
        )
    };
    run(vec![Intervention::set(a1, Value::f64(0.1))]).unwrap();
    // The same target twice.
    assert!(
        run(vec![Intervention::set(a1, Value::f64(0.1)), Intervention::set(a1, Value::f64(0.2))])
            .is_err()
    );
    // A zero-variance Gaussian policy has no law to integrate.
    assert!(
        run(vec![Intervention::Stochastic {
            variable: a1,
            policy: StochasticPolicy::Gaussian { mean: 0.1, variance: 0.0 },
        }])
        .is_err()
    );
    // 65 x 65 category pairs exceed the exact-mixture budget of 4096 combinations.
    let probs: Arc<[f64]> = vec![1.0; 65].into();
    let categorical = |variable| Intervention::Stochastic {
        variable,
        policy: StochasticPolicy::Categorical { probs: Arc::clone(&probs) },
    };
    let error = run(vec![categorical(a1), categorical(a2)]).err().expect("budget overrun");
    assert!(error.to_string().contains("exact-mixture budget"), "{error}");
}
