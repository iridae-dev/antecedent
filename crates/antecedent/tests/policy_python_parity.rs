//! A price-floor policy sampled from a fitted model is bit-identical in Rust and Python.
//!
//! Both sides fit an explicitly discrete/constant fixture and draw with the same seed.
//! Gaussian libm/linear-fit bits are not portable across platforms; this fixture
//! avoids those operations while pinning every drawn bit here and in
//! `python/tests/test_intervention_programs.py`. Python builds program text and calls this
//! same entry point, so a difference would mean a second implementation had crept in.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::gcm::{
    FittedGcm, PreparedPolicy, TemporalPlan, counterfactual_policy, counterfactual_temporal_policy,
    fit_gcm, fit_gcm_with, noise_table_from_columns, policy_value, registry_from_candidates,
    replay_noise, sample_do, sample_with_noise,
};
use antecedent_core::{
    CausalRng, ExecutionContext, Intervention, InterventionSequence, SequencedIntervention,
    TemporalPolicy, Value, ValueExpr, VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_model::{MechanismFamily, NodeFamilyConstraint, SelectionOptions};

/// The refusal of `result` names `needle` (its typed message), rather than failing for any reason.
fn refused_with<T, E: std::fmt::Debug>(result: Result<T, E>, needle: &str) {
    match result {
        Ok(_) => panic!("expected a refusal naming {needle:?}, got a value"),
        Err(error) => {
            let text = format!("{error:?}");
            assert!(text.contains(needle), "refusal {text} does not name {needle:?}");
        }
    }
}

/// Sum (mod 2^64) of the IEEE-754 bits of every draw of the policy run in the Python test.
const DISCRETE_FLOOR_DRAW_CHECKSUM: u64 = 13_792_273_858_822_143_950;
const DISCRETE_FLOOR_IDENTITY: &str =
    "3a08cbc918b28367814b2f0aa4ea318d963ae5dfb7af90f12bf7c3e7d6c15654";

fn price_floor() -> Intervention {
    Intervention::transform(
        VariableId::from_raw(0),
        ValueExpr::constant(10.0).max(ValueExpr::constant(0.85).mul(ValueExpr::generated())),
    )
}

/// The deterministic two-variable data of the Python test, fitted with default selection.
fn fitted_price_model(shift: f64) -> (FittedGcm, TabularData) {
    let price: Vec<f64> =
        (0..600_u32).map(|i| 11.0 + shift + (f64::from((i * 37) % 101) - 50.0) * 0.04).collect();
    let demand: Vec<f64> = (0..600_u32)
        .zip(&price)
        .map(|(i, p)| 50.0 - 0.8 * p + (f64::from((i * 13) % 23) - 11.0) * 0.05)
        .collect();
    let data =
        TabularData::from_f64_columns([("price", price.as_slice()), ("demand", demand.as_slice())])
            .unwrap();
    let mut graph = Dag::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    (fit_gcm(graph, &data).unwrap(), data)
}

#[test]
#[allow(clippy::float_cmp, reason = "a replay of retained noise is bit-identical to its sample")]
fn a_sample_with_its_noise_replays_bit_for_bit_and_accepts_caller_noise() {
    let (fitted, _) = fitted_price_model(0.0);
    let ctx = ExecutionContext::for_tests(3);
    let floor = [price_floor()];
    let (drawn, noise) = sample_with_noise(&fitted.model, &floor, 256, &ctx).unwrap();
    let replayed = replay_noise(&fitted.model, &floor, &noise, 0, &ctx).unwrap();
    assert_eq!(drawn.values, replayed.values);

    // The same noise handed back as plain columns is the same table.
    let caller = noise_table_from_columns(&fitted.model, 256, noise.values(), &ctx).unwrap();
    let again = replay_noise(&fitted.model, &floor, &caller, 0, &ctx).unwrap();
    assert_eq!(drawn.values, again.values);

    // Replay under another policy keeps the noise: the unfloored sample is the floor's
    // counterfactual on the same units.
    let free = replay_noise(&fitted.model, &[], &noise, 0, &ctx).unwrap();
    let (price, floored) = (free.column(0).unwrap(), drawn.column(0).unwrap());
    for (free, floored) in price.iter().zip(floored) {
        assert_eq!(*floored, (0.85 * free).max(10.0));
    }

    // A noise column of the wrong length or with a non-finite value is refused.
    refused_with(
        noise_table_from_columns(&fitted.model, 256, &noise.values()[1..], &ctx),
        "noise must hold n_rows values for each of the model's nodes",
    );
    let mut poisoned = noise.values().to_vec();
    poisoned[3] = f64::NAN;
    refused_with(
        noise_table_from_columns(&fitted.model, 256, &poisoned, &ctx),
        "fixed noise must be finite, got NaN",
    );
}

#[test]
#[allow(clippy::float_cmp, reason = "the temporal route replays the static route's noise")]
fn the_temporal_counterfactual_without_a_lag_read_is_the_static_counterfactual_per_step() {
    let (fitted, first) = fitted_price_model(0.0);
    let (_, second) = fitted_price_model(0.5);
    let ctx = ExecutionContext::for_tests(11);
    let floor = price_floor();
    let outcomes = [VariableId::from_raw(0), VariableId::from_raw(1)];

    let schedule = InterventionSequence::new(vec![SequencedIntervention::new(
        floor.clone(),
        TemporalPolicy::sustained(0, 100),
    )]);
    let plan = TemporalPlan {
        sequence: &schedule,
        n_steps: 2,
        n_rows: first.row_count(),
        window: 0,
        initial_history: &[],
    };
    let steps = [first.clone(), second.clone()];
    let path = counterfactual_temporal_policy(fitted.model.clone(), &steps, &plan, &ctx).unwrap();
    for (t, data) in steps.iter().enumerate() {
        let statics = counterfactual_policy(
            fitted.model.clone(),
            data,
            std::slice::from_ref(&floor),
            &outcomes,
            &ctx,
        )
        .unwrap();
        for (node, id) in outcomes.iter().enumerate() {
            let dense = fitted.model.dense_of(*id).unwrap();
            let column = path.column(t, node).unwrap();
            for (unit, value) in column.iter().enumerate() {
                assert_eq!(*value, statics.get(0, dense, unit), "step {t} node {node} unit {unit}");
            }
        }
    }

    // One table per step is required.
    let short = counterfactual_temporal_policy(fitted.model.clone(), &steps[..1], &plan, &ctx);
    refused_with(short, "one exogenous posterior per step is required");
}

#[test]
fn a_discrete_parent_policy_has_portable_rust_python_bits() {
    let price = vec![8.0; 64];
    let demand: Vec<f64> = (0..64_u32).map(|i| f64::from(i % 2)).collect();
    let data =
        TabularData::from_f64_columns([("price", price.as_slice()), ("demand", demand.as_slice())])
            .unwrap();
    let mut graph = Dag::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(0)).unwrap();
    let options = SelectionOptions {
        constraints: vec![
            (
                VariableId::from_raw(1),
                NodeFamilyConstraint {
                    required: Some(MechanismFamily::Discrete),
                    ..Default::default()
                },
            ),
            (
                VariableId::from_raw(0),
                NodeFamilyConstraint {
                    required: Some(MechanismFamily::Constant),
                    ..Default::default()
                },
            ),
        ],
        ..Default::default()
    };
    let registry = registry_from_candidates(&["discrete"]).unwrap();
    let fitted = fit_gcm_with(graph, &data, &registry, &options).unwrap();
    let program = |rate| {
        Intervention::transform(
            VariableId::from_raw(0),
            ValueExpr::constant(10.0).max(
                ValueExpr::constant(rate)
                    .mul(ValueExpr::generated().add(
                        ValueExpr::constant(8.0).mul(ValueExpr::var(VariableId::from_raw(1))),
                    )),
            ),
        )
    };
    let floor = program(0.85);
    let identity = |intervention: &Intervention| {
        blake3::Hash::from_bytes(
            antecedent_io::policy_program_artifact::policy_program_identity(std::slice::from_ref(
                intervention,
            ))
            .unwrap(),
        )
        .to_hex()
        .to_string()
    };
    assert_eq!(identity(&floor), DISCRETE_FLOOR_IDENTITY);
    let draw = |policy: &Intervention, seed| {
        sample_do(
            &fitted.model,
            std::slice::from_ref(policy),
            512,
            &mut CausalRng::from_seed(seed),
            &ExecutionContext::for_tests(seed),
        )
        .unwrap()
    };
    let draws = draw(&floor, 7);
    let checksum = draws.values.iter().fold(0_u64, |sum, v| sum.wrapping_add(v.to_bits()));
    assert_eq!(checksum, DISCRETE_FLOOR_DRAW_CHECKSUM);
    let prices = draws.column(0).unwrap();
    for (&value, &parent) in prices.iter().zip(draws.column(1).unwrap()) {
        assert_eq!(value.to_bits(), (0.85 * (8.0 + 8.0 * parent)).max(10.0_f64).to_bits());
    }
    assert!(prices.contains(&10.0) && prices.iter().any(|value| *value > 10.0));
    assert_ne!(draws.values, draw(&floor, 8).values);
    let other = program(0.75);
    assert_ne!(identity(&other), DISCRETE_FLOOR_IDENTITY);
    assert_ne!(draws.values, draw(&other, 7).values);
}

#[test]
fn a_policy_value_is_the_paired_difference_against_the_same_noise_with_its_licence() {
    let (fitted, _) = fitted_price_model(0.0);
    let ctx = ExecutionContext::for_tests(21);
    let n = 2_000;
    let (price, demand) = (VariableId::from_raw(0), VariableId::from_raw(1));
    let level = Intervention::set(price, Value::f64(20.0));

    // The noise does not depend on the overlay, so the unintervened draws on the same context
    // are an independent baseline.
    let (baseline, _) = sample_with_noise(&fitted.model, &[], n, &ctx).unwrap();
    let base_price = baseline.column(0).unwrap();
    let mean = base_price.iter().sum::<f64>() / n as f64;
    let sd = (base_price.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0)).sqrt();

    let on_price =
        policy_value(&fitted.model, std::slice::from_ref(&level), price, n, &ctx).unwrap();
    assert_eq!(on_price.n_draws, n);
    assert!((on_price.policy_mean - 20.0).abs() < 1e-12);
    assert!((on_price.baseline_mean - mean).abs() < 1e-9);
    assert!((on_price.mean_difference - (20.0 - mean)).abs() < 1e-9);
    assert!((on_price.difference_simulation_stderr - sd / (n as f64).sqrt()).abs() < 1e-9);
    assert!(on_price.policy_simulation_stderr.abs() < 1e-12, "a constant has no spread");

    // Demand follows price with the generating slope, and the noise cancels in the pairing.
    let on_demand =
        policy_value(&fitted.model, std::slice::from_ref(&level), demand, n, &ctx).unwrap();
    let slope = on_demand.mean_difference / on_price.mean_difference;
    assert!((slope + 0.8).abs() < 0.02, "slope {slope}");
    // The pairing removes the demand noise both arms share: the paired error is below the
    // error of differencing two independent samples of the same sizes.
    let base_demand = baseline.column(demand.as_usize()).unwrap();
    let demand_mean = base_demand.iter().sum::<f64>() / n as f64;
    let demand_var =
        base_demand.iter().map(|d| (d - demand_mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    let unpaired = (on_demand.policy_simulation_stderr.powi(2) + demand_var / n as f64).sqrt();
    assert!(on_demand.difference_simulation_stderr < unpaired);

    // A hard level is identifiable; a transform of the generated value is simulation-only.
    assert!(on_price.support.identification.is_licensed());
    let floor = policy_value(&fitted.model, &[price_floor()], price, n, &ctx).unwrap();
    assert!(floor.support.simulation.is_licensed() && floor.support.counterfactual.is_licensed());
    match &floor.support.identification {
        antecedent::gcm::RouteDecision::Refused { code, .. } => {
            assert_eq!(code, "policy_route_not_licensed");
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
    assert!(floor.mean_difference < 0.0, "a 15% cut outweighs the units the floor lifts");

    // One draw cannot report an error; an unknown outcome is refused.
    refused_with(
        policy_value(&fitted.model, &[], price, 1, &ctx),
        "needs at least two draws to report its simulation error",
    );
    refused_with(
        policy_value(&fitted.model, &[], VariableId::from_raw(9), n, &ctx),
        "outcome V9 is absent from the fitted model",
    );
}

#[test]
#[allow(clippy::float_cmp, reason = "the prepared value is the one-shot value, bit for bit")]
fn a_prepared_policy_reruns_the_one_shot_value_without_compiling_the_program_again() {
    let (fitted, _) = fitted_price_model(0.0);
    let (price, demand) = (VariableId::from_raw(0), VariableId::from_raw(1));
    let floor = price_floor();

    let prepared = PreparedPolicy::prepare(&fitted.model, vec![floor.clone()]).unwrap();
    assert_eq!((prepared.program_compilations(), prepared.model_bindings()), (1, 1));
    assert!(prepared.support().simulation.is_licensed());
    assert!(!prepared.support().identification.is_licensed());

    // Each rerun, at any seed and outcome, is the one-shot call to the last bit, and compiles
    // nothing.
    for seed in [3_u64, 4, 5] {
        let ctx = ExecutionContext::for_tests(seed);
        for outcome in [price, demand] {
            let rerun = prepared.value(&fitted.model, outcome, 800, &ctx).unwrap();
            let one_shot =
                policy_value(&fitted.model, std::slice::from_ref(&floor), outcome, 800, &ctx)
                    .unwrap();
            assert_eq!(rerun, one_shot);
        }
    }
    assert_eq!(prepared.program_compilations(), 1);

    // A refit model rebinds the program: the licence and overlay are resolved against it, the
    // program is not compiled again, and the value equals the one-shot value on that model.
    let (refit, _) = fitted_price_model(0.5);
    let rebound = prepared.rebind(&refit.model).unwrap();
    assert_eq!((rebound.program_compilations(), rebound.model_bindings()), (1, 2));
    let ctx = ExecutionContext::for_tests(9);
    assert_eq!(
        rebound.value(&refit.model, demand, 800, &ctx).unwrap(),
        policy_value(&refit.model, std::slice::from_ref(&floor), demand, 800, &ctx).unwrap(),
    );

    // A program the model refuses is refused at preparation; a request that cannot report a
    // simulation error is refused when run.
    let unknown = Intervention::set(VariableId::from_raw(9), Value::f64(1.0));
    refused_with(
        PreparedPolicy::prepare(&fitted.model, vec![unknown]),
        "variable V9 not in compiled model",
    );
    refused_with(
        prepared.value(&fitted.model, price, 1, &ctx),
        "needs at least two draws to report its simulation error",
    );
    refused_with(
        prepared.value(&fitted.model, VariableId::from_raw(9), 800, &ctx),
        "outcome V9 is absent from the fitted model",
    );
}
