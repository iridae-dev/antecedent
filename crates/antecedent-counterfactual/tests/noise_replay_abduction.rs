//! Abduced noise replayed through the batch noise-replay API matches the counterfactual
//! engine's own prediction, and the noise inference disclosure travels with the table.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{ExecutionContext, Intervention, Value, VariableId};
use antecedent_counterfactual::{
    AbductionMissingPolicy, CounterfactualEngine, CounterfactualWorld, NoiseInferenceKind,
};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_model::{
    CompiledCausalModel, CompiledMechanismStore, InterventionOverlay, MechanismSlot,
    MechanismWorkspace, ModelView, NoiseInferenceMode, NoiseOrigin, NoiseReplayer, NoiseSource,
    NoiseTable,
};

fn engine_for(slots: Vec<MechanismSlot>) -> CounterfactualEngine {
    let n = u32::try_from(slots.len()).unwrap();
    let mut graph = Dag::with_variables(n);
    for i in 1..n {
        graph.insert_directed(DenseNodeId::from_raw(i - 1), DenseNodeId::from_raw(i)).unwrap();
    }
    let compiled = CompiledCausalModel::compile(graph)
        .unwrap()
        .with_mechanisms(CompiledMechanismStore { slots: Arc::from(slots) });
    CounterfactualEngine::new(compiled)
}

fn do_x(value: f64) -> CounterfactualWorld {
    CounterfactualWorld {
        unit_rows: None,
        interventions: Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(value))]),
    }
}

/// Replay the abduced noise under the overlay of `world` and compare it with `predict`.
fn assert_replay_matches_predict(
    engine: &CounterfactualEngine,
    data: &TabularData,
    world: &CounterfactualWorld,
) -> NoiseTable {
    let ctx = ExecutionContext::for_tests(9);
    let exo = engine.abduct(data, AbductionMissingPolicy::Error, &ctx).unwrap();
    let n = exo.n_units;
    let n_nodes = exo.n_nodes;
    let outcomes: Vec<VariableId> =
        (0..u32::try_from(n_nodes).unwrap()).map(VariableId::from_raw).collect();
    let mut ws = MechanismWorkspace::default();
    let predicted =
        engine.predict(&exo, std::slice::from_ref(world), &outcomes, false, &mut ws, &ctx).unwrap();

    let table =
        NoiseTable::from_abducted(&engine.model, n, exo.noise.to_vec(), &exo.assumed_columns)
            .unwrap();
    let overlay =
        InterventionOverlay::from_interventions(&engine.model, &world.interventions).unwrap();
    let view = ModelView::with_overlay(&engine.model, overlay);
    let replayed = NoiseReplayer::new().replay(&view, NoiseSource::Table(&table), 0, &ctx).unwrap();
    for node in 0..n_nodes {
        let column = replayed.column(node).unwrap();
        for unit in 0..n {
            let want = predicted.get(0, DenseNodeId::from_raw(u32::try_from(node).unwrap()), unit);
            assert_eq!(column[unit].to_bits(), want.to_bits(), "node {node} unit {unit}");
        }
    }
    table
}

#[test]
fn abduction_then_replay_matches_the_counterfactual_prediction() {
    let engine = engine_for(vec![
        MechanismSlot::LinearGaussian { intercept: 0.0, coeffs: Arc::from([]), sigma: 1.0 },
        MechanismSlot::LinearGaussian { intercept: 0.5, coeffs: Arc::from([2.0]), sigma: 0.5 },
        MechanismSlot::LinearGaussian { intercept: -1.0, coeffs: Arc::from([3.0]), sigma: 2.0 },
    ]);
    let n = 60;
    let x: Vec<f64> = (0..n).map(|i| f64::from((i * 7) % 13) / 6.0 - 1.0).collect();
    let y: Vec<f64> =
        x.iter().enumerate().map(|(i, v)| 0.5 + 2.0 * v + ((i % 5) as f64 - 2.0) * 0.1).collect();
    let z: Vec<f64> =
        y.iter().enumerate().map(|(i, v)| -1.0 + 3.0 * v + ((i % 3) as f64 - 1.0) * 0.3).collect();
    let data = TabularData::from_f64_columns([
        ("x", x.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
    ])
    .unwrap();

    let table = assert_replay_matches_predict(&engine, &data, &do_x(1.5));
    assert!(!table.has_posterior_noise());
    assert!(
        table.origins().iter().all(|o| *o == NoiseOrigin::Abducted(NoiseInferenceMode::Invertible))
    );
    // The factual world (no intervention) replays back to the observed data.
    let ctx = ExecutionContext::for_tests(9);
    let view = ModelView::observational(&engine.model);
    let factual = NoiseReplayer::new().replay(&view, NoiseSource::Table(&table), 0, &ctx).unwrap();
    for (node, observed) in [&x, &y, &z].into_iter().enumerate() {
        for (a, b) in factual.column(node).unwrap().iter().zip(observed) {
            assert!((a - b).abs() < 1e-12);
        }
    }
}

#[test]
fn constant_abduction_uses_the_same_support_as_factual_replay() {
    let engine = engine_for(vec![MechanismSlot::Constant { value: 0.0 }]);
    let data = TabularData::from_f64_columns([("x", &[0.0, 0.5e-12][..])]).unwrap();
    let world = CounterfactualWorld { unit_rows: None, interventions: Arc::from([]) };
    let table = assert_replay_matches_predict(&engine, &data, &world);
    let ctx = ExecutionContext::for_tests(9);
    let view = ModelView::observational(&engine.model);
    let factual = NoiseReplayer::new().replay(&view, NoiseSource::Table(&table), 0, &ctx).unwrap();
    for (replayed, observed) in factual.column(0).unwrap().iter().zip([0.0, 0.5e-12]) {
        assert!((replayed - observed).abs() < 1e-12);
    }
    let off_support = TabularData::from_f64_columns([("x", &[1e-10][..])]).unwrap();
    assert!(matches!(
        engine.abduct(&off_support, AbductionMissingPolicy::Error, &ctx),
        Err(antecedent_counterfactual::CounterfactualError::Model(
            antecedent_model::ModelError::Numerical { .. }
        ))
    ));
}

#[test]
fn posterior_noise_disclosure_is_carried_into_the_table() {
    let engine = engine_for(vec![
        MechanismSlot::LinearGaussian { intercept: 0.0, coeffs: Arc::from([]), sigma: 1.0 },
        MechanismSlot::Discrete {
            support: Arc::from(vec![0.0, 1.0, 2.0]),
            probs: Arc::from(vec![0.2, 0.5, 0.3]),
            logit_coeffs: None,
        },
    ]);
    let n = 40;
    let x: Vec<f64> = (0..n).map(|i| f64::from(i % 9) * 0.25).collect();
    let y: Vec<f64> = (0..n).map(|i| f64::from(i % 3)).collect();
    let data = TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())]).unwrap();

    let ctx = ExecutionContext::for_tests(9);
    let exo = engine.abduct(&data, AbductionMissingPolicy::Error, &ctx).unwrap();
    assert_eq!(exo.kind, NoiseInferenceKind::PosteriorNoise);

    let table = assert_replay_matches_predict(&engine, &data, &do_x(0.5));
    assert!(table.has_posterior_noise());
    assert_eq!(table.inference_mode(0), Some(NoiseInferenceMode::Invertible));
    let mode = table.inference_mode(1).unwrap();
    assert_eq!(NoiseInferenceKind::from(mode), exo.kind);
}
