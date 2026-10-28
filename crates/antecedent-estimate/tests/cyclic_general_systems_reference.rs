//! Independent references for the numerics the umbrella cyclic-systems record owns on the
//! estimation side: projected Adam on the feedback likelihood and the Gaussian (Euler-type)
//! quasi-likelihood of a diffusion, direct and through the extended Kalman filter.
//!
//! `oracle_data/cyclic_general_systems_reference.py` (frozen in the JSON beside it) writes the
//! feedback likelihood with its analytic scores, replays Adam step by step at 60 digits for
//! full-batch epochs, and evaluates the Euler transition density, the exact Ornstein-Uhlenbeck
//! transition density and the Kalman innovation likelihood of the discretised model.  Nothing in
//! it reads product code and no paper was retrieved.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_estimate::cyclic_adam::{CyclicAdamPolicy, fit_cyclic_adam};
use antecedent_estimate::cyclic_likelihood::CyclicParameter;
use antecedent_estimate::nonlinear_sde_fit::{
    SdeFilterPrior, SdeFitPolicy, SdeObservationModel, SdeTrajectory, sde_quasi_log_likelihood,
};
use antecedent_model::nonlinear_equilibrium::{EquilibriumModel, RegimeRow};
use antecedent_model::nonlinear_sde::SdeSystem;
use serde_json::{Value, json};

fn document() -> Value {
    serde_json::from_str(include_str!("oracle_data/cyclic_general_systems_reference.json")).unwrap()
}

fn numbers(value: &Value) -> Vec<f64> {
    value.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()
}

fn close(actual: f64, expected: f64, relative: f64, label: &str) {
    let tolerance = relative * (1.0 + expected.abs());
    assert!((actual - expected).abs() <= tolerance, "{label}: {actual} vs {expected}");
}

// ---------------------------------------------------------------- Adam

fn regime_rows(rows: &Value) -> Vec<RegimeRow> {
    rows.as_array()
        .unwrap()
        .iter()
        .map(|row| {
            let row = numbers(row);
            RegimeRow {
                values: row.clone(),
                clamps: vec![antecedent_model::nonlinear_equilibrium::Clamp {
                    variable: 1,
                    value: row[1],
                }],
            }
        })
        .collect()
}

#[test]
fn cyclic_general_systems_projected_adam_replays_the_exact_recurrence() {
    let ctx = ExecutionContext::for_tests(0);
    for case in document()["adam"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let model: EquilibriumModel = serde_json::from_value(case["model"].clone()).unwrap();
        let parameters: Vec<CyclicParameter> =
            serde_json::from_value(case["learned"].clone()).unwrap();
        let policy: CyclicAdamPolicy = serde_json::from_value(case["policy"].clone()).unwrap();
        let (training, validation) =
            (regime_rows(&case["train"]), regime_rows(&case["validation"]));
        let fit =
            fit_cyclic_adam(&model, &parameters, &training, &validation, policy, &ctx).unwrap();
        let expected = &case["expected"];
        // The trace is the validation log likelihood per row after every epoch of the exact
        // recurrence; agreement epoch by epoch pins the moments, the bias corrections, the
        // epsilon placement, the L1 proximal step and the contraction projection.
        let trace = numbers(&expected["trace"]);
        assert_eq!(fit.trace.len(), trace.len(), "{name} epochs");
        for (k, (epoch, reference)) in fit.trace.iter().zip(&trace).enumerate() {
            close(
                epoch.validation_log_likelihood_per_row,
                *reference,
                1e-9,
                &format!("{name} trace {k}"),
            );
            assert_eq!(epoch.rejected_updates, 0, "{name}");
        }
        assert_eq!(
            fit.best_epoch as u64,
            expected["best_epoch"].as_u64().unwrap(),
            "{name} best epoch"
        );
        assert_eq!(fit.epochs as u64, expected["epochs"].as_u64().unwrap(), "{name} epochs");
        assert_eq!(
            fit.gradient_steps as u64,
            expected["gradient_steps"].as_u64().unwrap(),
            "{name} steps"
        );
        let equation = &fit.model.equations[0];
        close(
            equation.intercept,
            expected["intercept"].as_f64().unwrap(),
            1e-9,
            &format!("{name} intercept"),
        );
        close(
            equation.weights[0],
            expected["self"].as_f64().unwrap(),
            1e-9,
            &format!("{name} self"),
        );
        close(
            equation.weights[1],
            expected["weight"].as_f64().unwrap(),
            1e-9,
            &format!("{name} weight"),
        );
        close(
            fit.training_log_likelihood,
            expected["training_log_likelihood"].as_f64().unwrap(),
            1e-9,
            name,
        );
        close(
            fit.validation_log_likelihood,
            expected["validation_log_likelihood"].as_f64().unwrap(),
            1e-9,
            name,
        );
        close(
            fit.training_score_per_row,
            expected["training_score_per_row"].as_f64().unwrap(),
            1e-8,
            name,
        );
        assert_eq!(
            fit.validation_converged,
            expected["validation_converged"].as_bool().unwrap(),
            "{name}"
        );
        assert_eq!(
            fit.training_score_converged,
            expected["training_score_converged"].as_bool().unwrap(),
            "{name}"
        );
        assert_eq!(fit.stopping_reason, expected["stopping_reason"].as_str().unwrap(), "{name}");
        // The selected model keeps every learned row inside the declared contraction margin.
        let row_sum: f64 = equation.weights.iter().map(|w| w.abs()).sum();
        assert!(row_sum <= policy.contraction_margin, "{name}: row sum {row_sum}");
    }
}

#[test]
fn cyclic_general_systems_adam_minibatch_epochs_match_one_exact_batch_partition() {
    let document = document();
    let case = &document["minibatch"];
    let ctx = ExecutionContext::for_tests(0);
    let model: EquilibriumModel = serde_json::from_value(case["model"].clone()).unwrap();
    let policy: CyclicAdamPolicy = serde_json::from_value(case["policy"].clone()).unwrap();
    let parameters = [
        CyclicParameter::Intercept { equation: 0 },
        CyclicParameter::Weight { equation: 0, parent: 1 },
    ];
    let fit = fit_cyclic_adam(
        &model,
        &parameters,
        &regime_rows(&case["train"]),
        &regime_rows(&case["validation"]),
        policy,
        &ctx,
    )
    .unwrap();
    assert_eq!(fit.gradient_steps, 6, "three minibatches per epoch, two epochs");
    assert_eq!(fit.trace.len(), 2);
    let first = numbers(&case["epoch_one"]);
    let matches: Vec<usize> = (0..first.len())
        .filter(|&k| {
            (fit.trace[0].validation_log_likelihood_per_row - first[k]).abs()
                <= 1e-9 * (1.0 + first[k].abs())
        })
        .collect();
    // exactly one partition of the six rows into three ordered pairs reproduces epoch one
    assert_eq!(matches.len(), 1, "epoch one matches partitions {matches:?}");
    let second = numbers(&case["epoch_two"][matches[0]]);
    let again = second
        .iter()
        .filter(|&&v| {
            (fit.trace[1].validation_log_likelihood_per_row - v).abs() <= 1e-9 * (1.0 + v.abs())
        })
        .count();
    assert_eq!(again, 1, "epoch two matches {again} partitions");
}

// ---------------------------------------------------------------- Gaussian quasi-likelihood

fn compiled(system: &Value) -> antecedent_model::nonlinear_sde::CompiledSde {
    serde_json::from_value::<SdeSystem>(system.clone()).unwrap().compile().unwrap()
}

fn trajectories(list: &Value) -> Vec<SdeTrajectory> {
    list.as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(k, observations)| {
            serde_json::from_value(json!({"id": format!("unit-{k}"), "observations": observations}))
                .unwrap()
        })
        .collect()
}

fn quasi(
    system: &antecedent_model::nonlinear_sde::CompiledSde,
    theta: &[f64],
    observation: Option<&SdeObservationModel>,
    prior: Option<&SdeFilterPrior>,
    units: &[SdeTrajectory],
    policy: SdeFitPolicy,
) -> f64 {
    let ctx = ExecutionContext::for_tests(0);
    sde_quasi_log_likelihood(system, theta, observation, prior, units, policy, &ctx).unwrap()
}

#[test]
fn cyclic_general_systems_euler_quasi_likelihood_is_the_euler_density_and_tracks_the_exact_ou_law()
{
    let document = document();
    let ou = &document["ql"]["ou"];
    let system = compiled(&ou["system"]);
    let points: Vec<Vec<f64>> = ou["points"].as_array().unwrap().iter().map(numbers).collect();
    let mut gaps = Vec::new();
    let mut estimate_gaps = Vec::new();
    for level in ou["levels"].as_array().unwrap() {
        let interval = level["interval"].as_f64().unwrap();
        let unit = trajectories(&json!([level["observations"]]));
        for (point, value) in points.iter().zip(level["values"].as_array().unwrap()) {
            let library = quasi(&system, point, None, None, &unit, SdeFitPolicy::default());
            // the library value is exactly the Euler Gaussian transition density, summed
            close(library, value["euler"].as_f64().unwrap(), 1e-11, &format!("ou dt {interval}"));
        }
        // at the generating parameters the gap to the exact transition density shrinks with the
        // interval; elsewhere the two log likelihoods differ by an O(1) amount that does not
        // vanish (the sum of ln(1 + theta dt) over T / dt transitions), so only their maximisers
        // are comparable
        let truth = &level["values"][0];
        gaps.push((truth["euler"].as_f64().unwrap() - truth["exact"].as_f64().unwrap()).abs());
        // the library surface peaks at the Euler maximiser in theta
        let hat = level["theta_hat_euler"].as_f64().unwrap();
        let at = |theta: f64| {
            quasi(&system, &[theta, 0.8, 0.6], None, None, &unit, SdeFitPolicy::default())
        };
        assert!(
            at(hat) > at(hat + 2e-3) && at(hat) > at(hat - 2e-3),
            "dt {interval}: not a maximum at {hat}"
        );
        estimate_gaps.push((hat - level["theta_hat_exact"].as_f64().unwrap()).abs());
    }
    assert!(gaps[gaps.len() - 1] < 0.1 && gaps[gaps.len() - 1] < gaps[0] / 5.0, "{gaps:?}");
    assert!(
        estimate_gaps[estimate_gaps.len() - 1] < 0.1
            && estimate_gaps[estimate_gaps.len() - 1] < estimate_gaps[0] / 5.0,
        "{estimate_gaps:?}"
    );
    // irregular times and two trajectories
    let irregular = &document["ql"]["ou_irregular"];
    let units = trajectories(&irregular["trajectories"]);
    for (point, value) in points.iter().zip(irregular["values"].as_array().unwrap()) {
        let library = quasi(&system, point, None, None, &units, SdeFitPolicy::default());
        close(library, value["euler"].as_f64().unwrap(), 1e-11, "irregular");
    }
}

#[test]
fn cyclic_general_systems_euler_quasi_likelihood_handles_state_dependent_and_coupled_systems() {
    let document = document();
    for key in ["square_root", "coupled_direct"] {
        let case = &document["ql"][key];
        let system = compiled(&case["system"]);
        let units = trajectories(&case["trajectories"]);
        for (point, value) in
            case["points"].as_array().unwrap().iter().zip(case["values"].as_array().unwrap())
        {
            let library =
                quasi(&system, &numbers(point), None, None, &units, SdeFitPolicy::default());
            close(library, value["euler"].as_f64().unwrap(), 1e-11, key);
        }
    }
}

#[test]
fn cyclic_general_systems_extended_kalman_likelihood_equals_the_discretised_kalman_filter() {
    let document = document();
    for key in ["kalman_partial", "kalman_full_rank"] {
        let case = &document["ql"][key];
        let system = compiled(&case["system"]);
        let observation: SdeObservationModel =
            serde_json::from_value(case["observation"].clone()).unwrap();
        let prior: SdeFilterPrior = serde_json::from_value(case["prior"].clone()).unwrap();
        let times = numbers(&case["times"]);
        let observations: Vec<Value> = times
            .iter()
            .zip(case["values_y"].as_array().unwrap())
            .map(|(t, y)| json!({"time": t, "values": y}))
            .collect();
        let units = trajectories(&json!([observations]));
        for result in case["results"].as_array().unwrap() {
            let policy = SdeFitPolicy {
                filter_step: result["filter_step"].as_f64().unwrap(),
                ..SdeFitPolicy::default()
            };
            let library = quasi(
                &system,
                &numbers(&result["theta"]),
                Some(&observation),
                Some(&prior),
                &units,
                policy,
            );
            close(library, result["log_likelihood"].as_f64().unwrap(), 1e-10, key);
        }
    }
}
