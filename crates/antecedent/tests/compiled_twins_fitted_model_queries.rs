//! The nested counterfactual and the population do-contrast as compiled queries give the numbers
//! of their direct `antecedent::gcm` entries and of their fitted-model operation forms.
//!
//! * `SimulationTarget::NestedCounterfactual` (route `GcmSimulateDag`) and the direct
//!   `nested_counterfactual`, `nested_counterfactual_with_exo` and `nested_hard_counterfactual`:
//!   abduction is deterministic on a complete table, so the three entries, the operation form
//!   and the query are one number.
//! * `AttributeTarget::PopulationDoContrast` (route `GcmAttributeDag`) and the direct
//!   `population_do_contrast`: the query draws both arms from the attribution stream of a
//!   generator seeded with the target's seed, whatever seed the study runs under; a direct call
//!   under a context seeded with that seed draws the same numbers.
//!
//! The study fits the standard mechanism program, which selects the mechanisms `fit_gcm`
//! selects. Neither direct entry has a measured coverage record, and the query forms are point
//! answers with calibration status `not_measured`, as the operation forms are. Each query
//! carries the usual query-level contract and claim.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    deprecated,
    reason = "these tests call the direct entries the compiled queries are compared with"
)]
#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    reason = "bit-identical equality is the property under test; columns come from small indices"
)]

use std::sync::Arc;

use antecedent::analysis::model_operation::ModelOperationAnswer;
use antecedent::gcm::{
    AbductionMissingPolicy, CounterfactualEngine, MechanismWorkspace, fit_gcm,
    nested_counterfactual, nested_counterfactual_with_exo, nested_hard_counterfactual,
    population_do_contrast,
};
use antecedent::{
    AttributionOutcome, CompiledExecution, InferenceMode, RefuteSuite, SimulationOutcome, Study,
};
use antecedent_core::{
    AttributeQuery, AttributeTarget, CausalQuery, ExecutionContext, Intervention, ModelOperation,
    NestedCounterfactualTarget, PopulationDoContrastTarget, SimulateQuery, SimulationTarget, Value,
    VariableId,
};
use antecedent_counterfactual::NestedSpec;
use antecedent_data::{TableView, TabularData};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_io_core::{AttributeTargetWire, CausalQueryWire, SimulationTargetWire};

const ROWS: usize = 300;
const SEED: u64 = 0x5EED_0D0C;

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(SEED)
}

fn hard(variable: u32, level: f64) -> Intervention {
    Intervention::set(v(variable), Value::f64(level))
}

fn on_large_stack(name: &str, body: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name(name.into())
        .stack_size(32 * 1024 * 1024)
        .spawn(body)
        .unwrap()
        .join()
        .unwrap();
}

/// `x -> y`, `x -> z`, `y -> z`, linear with bounded wiggles. Variables `0 = x`, `1 = y`,
/// `2 = z`.
fn linear() -> (TabularData, Dag) {
    let x: Vec<f64> =
        (0..ROWS).map(|i| (i % 17) as f64 * 0.2 - 1.6 + 0.2 * (i as f64 * 1.3).cos()).collect();
    let y: Vec<f64> = (0..ROWS).map(|i| 1.0 + 2.0 * x[i] + 0.4 * (i as f64 * 0.37).sin()).collect();
    let z: Vec<f64> =
        (0..ROWS).map(|i| -0.5 + 0.7 * x[i] + 1.5 * y[i] + 0.4 * (i as f64 * 0.91).cos()).collect();
    let data = TabularData::from_f64_columns([
        ("x", x.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
    ])
    .unwrap();
    let mut graph = Dag::with_variables(3);
    for (from, to) in [(0, 1), (0, 2), (1, 2)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    (data, graph)
}

fn study(data: &TabularData, graph: &Dag, query: CausalQuery) -> Study {
    Study::tabular(data.clone())
        .graph(graph.clone())
        .query(query)
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
}

/// Compile and run `query`, checking the route it lowers to and that the run seals the
/// query-level contract and claim of its query kind.
fn run(
    data: &TabularData,
    graph: &Dag,
    query: CausalQuery,
    route: &str,
    kind: &str,
    ctx: &ExecutionContext,
) -> CompiledExecution {
    let study = study(data, graph, query);
    let plan = study.compile(ctx).unwrap();
    assert_eq!(plan.lowered().expect("a lowered program").route(), route);
    let execution = CompiledExecution::run(&study, ctx).unwrap();
    let (contract, _claim) = execution.contract_and_claim(ctx).unwrap();
    assert_eq!(&*contract.query_kind, kind);
    execution
}

/// The query the contract section of `execution`'s record states.
fn contract_query(execution: &CompiledExecution, ctx: &ExecutionContext) -> CausalQueryWire {
    let (contract, _) = execution
        .prepared()
        .execution_record_with_quantity_labels(execution.result(), ctx, None)
        .unwrap();
    contract.target.query
}

/// Run `operation` on the fitted model of a study prepared on `data` and `graph`.
fn operate(
    data: &TabularData,
    graph: &Dag,
    operation: &ModelOperation,
    ctx: &ExecutionContext,
) -> ModelOperationAnswer {
    let carrier =
        CausalQuery::Simulate(SimulateQuery::new(SimulationTarget::InterventionalSample {
            interventions: Arc::from([]),
            draws: 1,
            seed: 0,
        }));
    let prepared = study(data, graph, carrier).prepare(ctx).unwrap();
    prepared.model_operation(operation, ctx).unwrap().into_answer()
}

fn mediation_target() -> NestedCounterfactualTarget {
    // Y_{x = 1, M_{x = 0}} with M = y: the mediation contrast of x on z through y.
    NestedCounterfactualTarget {
        outer: Arc::from([hard(0, 0.0)]),
        inner: Arc::from([hard(0, 1.0)]),
        frozen: Arc::from([v(1)]),
        outcome: v(2),
    }
}

#[test]
fn the_nested_counterfactual_query_equals_the_direct_entries_and_the_operation_form() {
    on_large_stack("nested_counterfactual_query_twin", || {
        let (data, graph) = linear();
        let fitted = fit_gcm(graph.clone(), &data).unwrap();
        let target = mediation_target();
        let spec = NestedSpec {
            outer: &target.outer,
            inner: &target.inner,
            frozen: &target.frozen,
            outcome: target.outcome,
        };
        let engine = CounterfactualEngine::new(fitted.model.clone());
        let mut ws = MechanismWorkspace::default();
        let direct = nested_counterfactual(&engine, &data, &spec, &mut ws, &ctx()).unwrap();
        let hard_alias =
            nested_hard_counterfactual(&engine, &data, &spec, &mut ws, &ctx()).unwrap();
        let exo = engine.abduct(&data, AbductionMissingPolicy::Error, &ctx()).unwrap();
        let with_exo =
            nested_counterfactual_with_exo(&engine, &exo, &spec, &mut ws, &ctx()).unwrap();
        assert!(direct.is_finite());
        assert_eq!(hard_alias.to_bits(), direct.to_bits());
        assert_eq!(with_exo.to_bits(), direct.to_bits());

        let query = CausalQuery::Simulate(SimulateQuery::new(
            SimulationTarget::NestedCounterfactual(target.clone()),
        ));
        let execution = run(&data, &graph, query, "GcmSimulateDag", "Simulate", &ctx());
        let Some(SimulationOutcome::NestedCounterfactual(compiled)) =
            execution.result().simulation().map(|payload| payload.outcome.clone())
        else {
            panic!("a nested counterfactual query returns its mean");
        };
        assert_eq!(compiled.to_bits(), direct.to_bits());
        let query = contract_query(&execution, &ctx());
        let CausalQueryWire::Simulate(wire) = &query else {
            panic!("the contract states the simulation query");
        };
        assert!(matches!(wire.target, SimulationTargetWire::NestedCounterfactual { .. }));

        let ModelOperationAnswer::NestedCounterfactual(operated) =
            operate(&data, &graph, &ModelOperation::NestedCounterfactual(target), &ctx())
        else {
            panic!("a nested counterfactual operation returns its mean");
        };
        assert_eq!(operated.to_bits(), direct.to_bits());
    });
}

#[test]
fn the_population_do_contrast_query_draws_from_the_target_seed_like_the_direct_entry() {
    on_large_stack("population_do_contrast_query_twin", || {
        let (data, graph) = linear();
        let fitted = fit_gcm(graph.clone(), &data).unwrap();
        let target_seed = 41;
        let direct = population_do_contrast(
            &fitted.model,
            &data,
            v(0),
            v(2),
            0.5,
            data.row_count(),
            &ExecutionContext::for_tests(target_seed),
        )
        .unwrap();
        assert!(direct.contrast.is_finite());
        let target = PopulationDoContrastTarget {
            parent: v(0),
            child: v(2),
            delta: 0.5,
            max_units: None,
            seed: target_seed,
        };
        // The study runs under another seed: the draws follow the target's.
        let other = ExecutionContext::for_tests(target_seed ^ 0xFFFF);
        let query = CausalQuery::Attribute(AttributeQuery::new(
            AttributeTarget::PopulationDoContrast(target),
        ));
        let execution = run(&data, &graph, query, "GcmAttributeDag", "Attribute", &other);
        let Some(AttributionOutcome::PopulationDoContrast(compiled)) =
            execution.result().attribution().map(|payload| payload.outcome.clone())
        else {
            panic!("a do-contrast query returns the contrast");
        };
        assert_eq!(compiled.contrast.to_bits(), direct.contrast.to_bits());
        assert_eq!(compiled.stderr.to_bits(), direct.stderr.to_bits());
        let query = contract_query(&execution, &other);
        let CausalQueryWire::Attribute(wire) = &query else {
            panic!("the contract states the attribution query");
        };
        assert!(matches!(wire.target, AttributeTargetWire::PopulationDoContrast { .. }));

        let ModelOperationAnswer::PopulationDoContrast(operated) =
            operate(&data, &graph, &ModelOperation::PopulationDoContrast(target), &other)
        else {
            panic!("a do-contrast operation returns the contrast");
        };
        assert_eq!(operated.contrast.to_bits(), direct.contrast.to_bits());
        assert_eq!(operated.stderr.to_bits(), direct.stderr.to_bits());
    });
}

#[test]
fn the_query_forms_refuse_what_their_operation_forms_refuse() {
    on_large_stack("fitted_model_query_refusals", || {
        let (data, graph) = linear();
        let refused = |query: CausalQuery| {
            Study::tabular(data.clone())
                .graph(graph.clone())
                .query(query)
                .bootstrap_replicates(0)
                .build()
                .and_then(|study| study.compile(&ctx()).map(|_| ()))
                .is_err()
        };
        let frozen_outcome =
            NestedCounterfactualTarget { frozen: Arc::from([v(2)]), ..mediation_target() };
        assert!(refused(CausalQuery::Simulate(SimulateQuery::new(
            SimulationTarget::NestedCounterfactual(frozen_outcome),
        ))));
        let self_contrast = PopulationDoContrastTarget {
            parent: v(0),
            child: v(0),
            delta: 0.5,
            max_units: None,
            seed: 1,
        };
        assert!(refused(CausalQuery::Attribute(AttributeQuery::new(
            AttributeTarget::PopulationDoContrast(self_contrast),
        ))));
    });
}
