//! The direct fitted-model operations of `antecedent::gcm` and the compiled routes that express
//! them give the same numbers.
//!
//! * `counterfactual_contrasts` and the `Counterfactual` query: the route fits the
//!   heterogeneity-capable registry (`fit_gcm_counterfactual`), abducts once and reads every
//!   outcome from the same two replays, the call `counterfactual_contrasts` makes after its own
//!   abduction. Every mean and unit effect is compared on `to_bits()`.
//! * `interventional_mean` and the `Simulate` interventional sample: both draw through
//!   `sample_interventional`; the route returns the draws and the direct entry their mean, so the
//!   test averages the route's column the way the direct entry does and starts both from one
//!   seeded stream.
//! * `simultaneous_hard_counterfactual` and the `Simulate` counterfactual: the pair of
//!   intervention sets of the direct entry is one world in the route, so the route's outcome mean
//!   is compared with the direct value on `to_bits()`.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    reason = "bit-identical equality is the property under test; columns come from small indices"
)]

use std::sync::Arc;

use antecedent::gcm::{
    CounterfactualEngine, MechanismWorkspace, counterfactual_contrasts, fit_gcm,
    fit_gcm_counterfactual, interventional_mean, simultaneous_hard_counterfactual,
};
use antecedent::{CompiledExecution, RefuteSuite, SimulationOutcome, Study};
use antecedent_core::{
    CausalQuery, CausalRng, CounterfactualQuery, ExecutionContext, Intervention, SimulateQuery,
    SimulationTarget, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};

const ROWS: usize = 300;

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|x| x.to_bits()).collect()
}

/// `x1`, `x2` unrelated causes of `y1` and `y2`, each with its own bounded wiggle. Variables
/// `0 = x1`, `1 = x2`, `2 = y1`, `3 = y2`.
fn scm() -> (TabularData, Dag) {
    let x1: Vec<f64> = (0..ROWS).map(|i| (i % 17) as f64 * 0.2).collect();
    let x2: Vec<f64> = (0..ROWS).map(|i| ((i * 7) % 11) as f64 * 0.25).collect();
    let y1: Vec<f64> = (0..ROWS)
        .map(|i| 1.0 + 2.0 * x1[i] + 3.0 * x2[i] + 0.2 * (i as f64 * 0.37).sin())
        .collect();
    let y2: Vec<f64> =
        (0..ROWS).map(|i| x1[i] - 2.0 * x2[i] + 0.2 * (i as f64 * 0.91).cos()).collect();
    let data = TabularData::from_f64_columns([
        ("x1", x1.as_slice()),
        ("x2", x2.as_slice()),
        ("y1", y1.as_slice()),
        ("y2", y2.as_slice()),
    ])
    .unwrap();
    let mut graph = Dag::with_variables(4);
    for (from, to) in [(0, 2), (1, 2), (0, 3), (1, 3)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    (data, graph)
}

fn hard(variable: u32, level: f64) -> Intervention {
    Intervention::set(v(variable), Value::f64(level))
}

#[test]
fn counterfactual_contrasts_equal_the_compiled_counterfactual_route() {
    let (data, graph) = scm();
    let ctx = ExecutionContext::for_tests(31);
    let active = [hard(0, 1.0), hard(1, 1.0)];
    let control = [hard(0, 0.0), hard(1, 0.0)];

    let fitted = fit_gcm_counterfactual(&Arc::new(graph.clone()), &data).unwrap();
    let direct =
        counterfactual_contrasts(fitted.model, &data, &[v(2), v(3)], (&active, &control), &ctx)
            .unwrap();

    let mut query = CounterfactualQuery::new(v(2), active.to_vec()).with_joint_control(control);
    query.outcomes = Arc::from([v(2), v(3)]);
    let result = Study::tabular(data.clone())
        .graph(graph)
        .query(CausalQuery::Counterfactual(query))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap()
        .estimate(&data, &ctx)
        .unwrap();
    let set = result.counterfactual_contrasts().expect("a contrast set payload");

    assert_eq!(set.contrasts.len(), direct.contrasts.len());
    for (compiled, direct) in set.contrasts.iter().zip(&direct.contrasts) {
        assert_eq!(compiled.outcome, direct.outcome);
        assert_eq!(compiled.mean_ite.to_bits(), direct.mean_ite.to_bits());
        assert_eq!(bits(&compiled.unit_effects), bits(&direct.unit_effects));
    }
}

#[test]
fn interventional_mean_equals_the_mean_of_the_compiled_interventional_sample() {
    let (data, graph) = scm();
    let ctx = ExecutionContext::for_tests(0x5EED_0017);
    let fitted = fit_gcm(graph.clone(), &data).unwrap();
    let interventions: Arc<[Intervention]> = Arc::from([hard(0, 1.0)]);
    let draws = 64;

    let mut rng = CausalRng::from_seed(4);
    let mut workspace = MechanismWorkspace::default();
    let direct = interventional_mean(
        &fitted.model,
        &interventions,
        v(2),
        draws,
        &mut rng,
        &mut workspace,
        &ctx,
    )
    .unwrap();

    let study = Study::tabular(data)
        .graph(graph)
        .query(CausalQuery::Simulate(SimulateQuery::new(SimulationTarget::InterventionalSample {
            interventions: Arc::clone(&interventions),
            draws,
            seed: 4,
        })))
        .build()
        .unwrap();
    let execution = CompiledExecution::run(&study, &ctx).unwrap();
    let Some(SimulationOutcome::Sample(batch)) =
        execution.result().simulation().map(|payload| payload.outcome.clone())
    else {
        panic!("an interventional sample returns draws");
    };
    let column = batch.column(2).unwrap();
    let compiled = column.iter().sum::<f64>() / column.len().max(1) as f64;
    assert_eq!(compiled.to_bits(), direct.to_bits());
}

#[test]
fn simultaneous_hard_counterfactual_equals_the_mean_of_the_compiled_counterfactual() {
    let (data, graph) = scm();
    let ctx = ExecutionContext::for_tests(0x5EED_0017);
    let fitted = fit_gcm(graph.clone(), &data).unwrap();
    let outer = [hard(0, 1.0)];
    let inner = [hard(1, 0.5)];

    let engine = CounterfactualEngine::new(fitted.model.clone());
    let mut workspace = MechanismWorkspace::default();
    let direct = simultaneous_hard_counterfactual(
        &engine,
        &data,
        &outer,
        &inner,
        v(2),
        &mut workspace,
        &ctx,
    )
    .unwrap();

    let study = Study::tabular(data)
        .graph(graph)
        .query(CausalQuery::Simulate(SimulateQuery::new(SimulationTarget::Counterfactual {
            interventions: Arc::from([hard(0, 1.0), hard(1, 0.5)]),
            outcomes: Arc::from([v(2)]),
        })))
        .build()
        .unwrap();
    let execution = CompiledExecution::run(&study, &ctx).unwrap();
    let Some(SimulationOutcome::Counterfactual(values)) =
        execution.result().simulation().map(|payload| payload.outcome.clone())
    else {
        panic!("a counterfactual returns its values");
    };
    let compiled = values.outcome_mean(0, DenseNodeId::from_raw(2));
    assert_eq!(compiled.n_used, compiled.n_units, "every unit enters the mean");
    assert_eq!(compiled.mean.to_bits(), direct.to_bits());
}
