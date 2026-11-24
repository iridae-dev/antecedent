//! The compiled z-transport and the direct `StudyBuilder::z_transport` / `PreparedZTransport`
//! path give the same target law, interval and interval reason, bit for bit.
//!
//! The fixtures are those of the prepared route's own tests: the registered four-node surrogate
//! graph (W -> Z -> X -> Y, W -> Y, with W <-> Y, Z <-> Y and Z <-> X) whose cited source joint
//! under do(Z = 0) is P(W) P(X) P(Y | X), exact and as a 10 000-unit counted table; two disconnected
//! single-intervention components combined as independent components; and the two factors of
//! X -> Z, X -> Y combined as intervention-factorized components. Both paths take the caller's
//! context, so the same `ExecutionContext::for_tests(seed)` gives the same bootstrap and
//! posterior draws. The compiled program names the `ZTransportEstimate` route and reports the
//! decision of its unit, `transport.single_source_z_transport`.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::float_cmp, reason = "the bits are compared exactly")]
#![allow(deprecated, reason = "exercises the deprecated direct facade path")]

use std::sync::Arc;

use antecedent::analysis::kernel_programs::z_transport::CompiledZTransport;
use antecedent::planner::PlanRoute;
use antecedent::{PreparedZTransport, Study, StudyBuilder, ZTransportResult};
use antecedent_core::{
    DependenceGroup, DistributionAvailability, Environment, EvidenceCatalog, EvidenceKind,
    EvidenceRegime, ExecutionContext, InterventionAssignment, RegimeBinding, RegimeId, RegimeKind,
    SamplingDesign, Value, VariableCoordinate, VariableDomain, VariableId,
};
use antecedent_estimate::BayesianTransportLawProvider;
use antecedent_expr::{
    Assignment, DiscreteAxis, ExactDiscreteLaw, ExactEvaluationLimits, ExactTransportData,
    LawTolerance,
};
use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram};
use antecedent_identify::{
    BoundZTransportFunctional, SidLimits, ZTransportDecision, ZTransportQuery,
    ZTransportResult as Identified, bind_z_transport_catalog, decide_z_transport_with_catalog,
    identify_z_transport,
};

const W: VariableId = VariableId::from_raw(0);
const Z: VariableId = VariableId::from_raw(1);
const X: VariableId = VariableId::from_raw(2);
const Y: VariableId = VariableId::from_raw(3);

fn binary(variable: VariableId) -> DiscreteAxis {
    DiscreteAxis { variable, values: Arc::from([Value::Bool(false), Value::Bool(true)]) }
}

fn binding(regime: u32, snapshot: &str) -> RegimeBinding {
    RegimeBinding {
        dataset_identity: None,
        regime: RegimeId::from_raw(regime),
        snapshot_identity: Arc::from(snapshot),
        schema_names: Arc::from([]),
        sampling: SamplingDesign::Independent,
        weights: None,
        dependence: DependenceGroup::IndependentStudies,
    }
}

fn coordinates(variables: &[VariableId]) -> Vec<VariableCoordinate> {
    variables
        .iter()
        .map(|&variable| VariableCoordinate {
            variable,
            domain: VariableDomain::Binary,
            unit: None,
        })
        .collect()
}

struct Surrogate {
    diagram: SelectionDiagram,
    functional: BoundZTransportFunctional,
    data: ExactTransportData,
}

/// The registered surrogate fixture; `empirical` makes the cited law a counted table.
fn surrogate(empirical: bool) -> Surrogate {
    let mut graph = Admg::with_variables(4);
    for (from, to) in [(0, 1), (1, 2), (2, 3), (0, 3)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    for (a, b) in [(0, 3), (1, 3), (1, 2)] {
        graph.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    let diagram = SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from([])).unwrap();
    let query = ZTransportQuery {
        outcomes: Arc::from([Y]),
        treatments: Arc::from([X]),
        controllable: Arc::from([Z]),
        experiment_assignment: Arc::from([InterventionAssignment {
            variable: Z,
            value: Value::Bool(false),
        }]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    };
    let Identified::Identified(proof) = identify_z_transport(
        &diagram,
        &query,
        SidLimits::default(),
        &ExecutionContext::for_tests(0),
    )
    .unwrap() else {
        panic!("the registered surrogate formula identifies");
    };
    let regimes = [false, true].map(|level| {
        EvidenceRegime::try_new(
            RegimeId::from_raw(u32::from(level)),
            RegimeKind::Experimental,
            EvidenceKind::Available,
            [Z],
            [InterventionAssignment { variable: Z, value: Value::Bool(level) }],
            [W, Z, X, Y],
            "source",
            DistributionAvailability::Joint,
        )
        .unwrap()
    });
    let catalog = EvidenceCatalog::try_new(
        [Environment::try_new("source", coordinates(&[W, Z, X, Y]), Arc::<[VariableId]>::from([]))
            .unwrap()],
        regimes,
        [binding(0, "do-z-0"), binding(1, "do-z-1")],
        None,
    )
    .unwrap();
    let functional = bind_z_transport_catalog(&diagram, &query, &proof, &catalog).unwrap();
    let mut probabilities = Vec::with_capacity(8);
    let mut counts = Vec::with_capacity(8);
    for w in [false, true] {
        for x in [false, true] {
            for y in [false, true] {
                let p = (if w { 0.25_f64 } else { 0.75 })
                    * (if x { 0.35_f64 } else { 0.65 })
                    * (if y == x { 0.80_f64 } else { 0.20 });
                probabilities.push(p);
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "a probability in [0, 1] scaled by 10 000 and rounded to a count"
                )]
                counts.push((p * 10_000.0).round() as u64);
            }
        }
    }
    if empirical {
        let total = counts.iter().sum::<u64>() as f64;
        probabilities = counts.iter().map(|count| *count as f64 / total).collect();
    }
    let mut law = ExactDiscreteLaw::try_new(
        "source",
        RegimeId::from_raw(0),
        [antecedent_expr::InterventionAssignment::concrete(Z, Value::Bool(false))],
        [binary(W), binary(X), binary(Y)],
        probabilities,
        "do-z-0",
        LawTolerance::default(),
    )
    .unwrap();
    if empirical {
        law = law.with_empirical_counts(counts).unwrap();
    }
    Surrogate { diagram, functional, data: ExactTransportData::try_new([law], 128).unwrap() }
}

fn request() -> Assignment {
    Assignment::from_pairs([(X, Value::Bool(false))])
}

fn direct(fixture: Surrogate, empirical: bool, ctx: &ExecutionContext) -> PreparedZTransport {
    let limits = ExactEvaluationLimits::default();
    if empirical {
        StudyBuilder::z_transport_empirical(
            fixture.diagram,
            fixture.functional,
            fixture.data,
            request(),
            limits,
            ctx,
        )
    } else {
        StudyBuilder::z_transport(
            fixture.diagram,
            fixture.functional,
            fixture.data,
            request(),
            limits,
            ctx,
        )
    }
    .unwrap()
}

fn compiled(fixture: Surrogate, empirical: bool, ctx: &ExecutionContext) -> CompiledZTransport {
    let limits = ExactEvaluationLimits::default();
    if empirical {
        Study::compile_z_transport_empirical(
            fixture.diagram,
            fixture.functional,
            fixture.data,
            request(),
            limits,
            ctx,
        )
    } else {
        Study::compile_z_transport(
            fixture.diagram,
            fixture.functional,
            fixture.data,
            request(),
            limits,
            ctx,
        )
    }
    .unwrap()
}

fn same_result(direct: &ZTransportResult, compiled: &ZTransportResult) {
    let (left, right) = (direct.distribution(), compiled.distribution());
    assert_eq!(left.outcomes, right.outcomes);
    assert_eq!(left.atoms, right.atoms);
    assert_eq!(left.probabilities.len(), right.probabilities.len());
    for (a, b) in left.probabilities.iter().zip(right.probabilities.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    assert_eq!(format!("{:?}", left.support), format!("{:?}", right.support));
    assert_eq!(direct.interval_method(), compiled.interval_method());
    assert_eq!(direct.interval_reason(), compiled.interval_reason());
    assert_eq!(direct.interval_type(), compiled.interval_type());
    assert_eq!(
        direct.coverage_target().map(f64::to_bits),
        compiled.coverage_target().map(f64::to_bits)
    );
    assert_eq!(direct.mean_intervals().len(), compiled.mean_intervals().len());
    for (a, b) in direct.mean_intervals().iter().zip(compiled.mean_intervals()) {
        assert_eq!(a.0, b.0);
        assert_eq!(a.1.to_bits(), b.1.to_bits());
        assert_eq!(a.2.to_bits(), b.2.to_bits());
    }
}

#[test]
fn compiled_exact_and_empirical_estimates_equal_the_direct_estimates_bit_for_bit() {
    for empirical in [false, true] {
        let ctx = ExecutionContext::for_tests(7);
        let prepared = direct(surrogate(empirical), empirical, &ctx);
        let program = compiled(surrogate(empirical), empirical, &ctx);
        let left = prepared.estimate(&ExecutionContext::for_tests(9)).unwrap();
        let right = program.estimate(&ExecutionContext::for_tests(9)).unwrap();
        same_result(&left, &right);
        assert_eq!(left.interval_method().is_some(), empirical);
        // The exported point artifacts are the same bytes.
        assert_eq!(left.export(&prepared).unwrap(), right.export(program.prepared()).unwrap());
    }
}

#[test]
fn compiled_bayesian_estimates_equal_the_direct_estimates_bit_for_bit() {
    let ctx = ExecutionContext::for_tests(7);
    let prepared = direct(surrogate(true), true, &ctx);
    let program = compiled(surrogate(true), true, &ctx);
    for provider in [
        BayesianTransportLawProvider::EmpiricalSupport,
        BayesianTransportLawProvider::DeclaredStateSpaceDirichlet,
    ] {
        let left =
            prepared.estimate_bayesian(provider, 199, &ExecutionContext::for_tests(31)).unwrap();
        let right =
            program.estimate_bayesian(provider, 199, &ExecutionContext::for_tests(31)).unwrap();
        same_result(&left, &right);
    }
}

#[test]
fn a_compiled_refresh_rebinds_the_laws_as_the_direct_refresh_does() {
    let ctx = ExecutionContext::for_tests(7);
    let prepared = direct(surrogate(false), false, &ctx);
    let program = compiled(surrogate(false), false, &ctx);
    let data = surrogate(false).data;
    let refreshed = prepared.refresh(data.clone(), &ctx).unwrap();
    let recompiled = program.refresh(data, &ctx).unwrap();
    assert_eq!(recompiled.lowered().identity(), program.lowered().identity());
    same_result(&refreshed.estimate(&ctx).unwrap(), &recompiled.estimate(&ctx).unwrap());
}

/// One single-intervention component: `P(outcome | do(treatment = level))` from `population`.
struct Component {
    diagram: SelectionDiagram,
    functional: BoundZTransportFunctional,
    data: ExactTransportData,
    request: Assignment,
}

fn component(
    graph: &SelectionDiagram,
    population: &str,
    treatment: VariableId,
    outcome: VariableId,
    level: bool,
    probability_true: f64,
) -> Component {
    let nodes: Vec<VariableId> = graph
        .causal_graph()
        .nodes()
        .iter()
        .filter_map(|node| match node {
            antecedent_graph::NodeRef::Static(variable) => Some(*variable),
            _ => None,
        })
        .collect();
    let regime = EvidenceRegime::try_new(
        RegimeId::from_raw(u32::from(level)),
        RegimeKind::Experimental,
        EvidenceKind::Available,
        [treatment],
        [InterventionAssignment { variable: treatment, value: Value::Bool(level) }],
        [outcome],
        population,
        DistributionAvailability::Joint,
    )
    .unwrap();
    let snapshot = format!("{population}-joint-{level}");
    let catalog = EvidenceCatalog::try_new(
        ["alpha", "beta", "target"]
            .map(|name| Environment::try_new(name, coordinates(&nodes), []).unwrap()),
        [regime],
        [binding(u32::from(level), &snapshot)],
        None,
    )
    .unwrap();
    let query = ZTransportQuery {
        outcomes: Arc::from([outcome]),
        treatments: Arc::from([treatment]),
        controllable: Arc::from([treatment]),
        experiment_assignment: Arc::from([InterventionAssignment {
            variable: treatment,
            value: Value::Bool(level),
        }]),
        source: Arc::from(population),
        target: Arc::from("target"),
    };
    let ZTransportDecision::Identified(proof) = decide_z_transport_with_catalog(
        graph,
        &query,
        &catalog,
        SidLimits::default(),
        &ExecutionContext::for_tests(0),
    )
    .unwrap() else {
        panic!("the component's joint intervention identifies");
    };
    let functional = bind_z_transport_catalog(graph, &query, &proof, &catalog).unwrap();
    let law = ExactDiscreteLaw::try_new(
        population,
        RegimeId::from_raw(u32::from(level)),
        [antecedent_expr::InterventionAssignment::concrete(treatment, Value::Bool(level))],
        [binary(outcome)],
        [1.0 - probability_true, probability_true],
        snapshot,
        LawTolerance::default(),
    )
    .unwrap();
    Component {
        diagram: graph.clone(),
        functional,
        data: ExactTransportData::try_new([law], 8).unwrap(),
        request: Assignment::from_pairs([(treatment, Value::Bool(level))]),
    }
}

fn prepare_both(
    component: Component,
    ctx: &ExecutionContext,
) -> (PreparedZTransport, CompiledZTransport) {
    let limits = ExactEvaluationLimits::default();
    let prepared = StudyBuilder::z_transport(
        component.diagram.clone(),
        component.functional.clone(),
        component.data.clone(),
        component.request.clone(),
        limits,
        ctx,
    )
    .unwrap();
    let program = Study::compile_z_transport(
        component.diagram,
        component.functional,
        component.data,
        component.request,
        limits,
        ctx,
    )
    .unwrap();
    (prepared, program)
}

#[test]
fn compiled_independent_components_equal_the_direct_combination_bit_for_bit() {
    // W -> Z and X -> Y in disconnected components; alpha controls W, beta controls X.
    let mut graph = Admg::with_variables(4);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(3)).unwrap();
    let diagram = SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from([])).unwrap();
    let ctx = ExecutionContext::for_tests(0);
    for (level, alpha_true, beta_true) in [(false, 0.25, 0.60), (true, 0.70, 0.40)] {
        let (alpha, alpha_program) =
            prepare_both(component(&diagram, "alpha", W, Z, level, alpha_true), &ctx);
        let (beta, beta_program) =
            prepare_both(component(&diagram, "beta", X, Y, level, beta_true), &ctx);
        let left =
            PreparedZTransport::estimate_independent_components(&alpha, &beta, 4, &ctx).unwrap();
        let right = CompiledZTransport::estimate_independent_components(
            &alpha_program,
            &beta_program,
            4,
            &ctx,
        )
        .unwrap();
        same_result(&left, &right);
        // The atom limit refuses on both paths alike.
        assert!(
            PreparedZTransport::estimate_independent_components(&alpha, &beta, 3, &ctx).is_err()
        );
        assert!(
            CompiledZTransport::estimate_independent_components(
                &alpha_program,
                &beta_program,
                3,
                &ctx
            )
            .is_err()
        );
    }
}

#[test]
fn compiled_factorized_components_equal_the_direct_combination_bit_for_bit() {
    // X -> Z and X -> Y: Z and Y factorize given X, both factors at X = 0.
    let mut graph = Admg::with_variables(3);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    let diagram = SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from([])).unwrap();
    let (x, z, y) = (VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2));
    let ctx = ExecutionContext::for_tests(0);
    let (left, left_program) = prepare_both(component(&diagram, "alpha", x, z, false, 0.3), &ctx);
    let (right, right_program) = prepare_both(component(&diagram, "beta", x, y, false, 0.6), &ctx);
    let direct =
        PreparedZTransport::estimate_intervention_factorized_components(&left, &right, 4, &ctx)
            .unwrap();
    let compiled = CompiledZTransport::estimate_intervention_factorized_components(
        &left_program,
        &right_program,
        4,
        &ctx,
    )
    .unwrap();
    same_result(&direct, &compiled);
    assert_eq!(compiled.distribution().outcomes.as_ref(), [z, y]);
}

#[test]
fn the_compiled_program_names_its_route_and_reports_its_unit() {
    let ctx = ExecutionContext::for_tests(7);
    let exact = compiled(surrogate(false), false, &ctx);
    let empirical = compiled(surrogate(true), true, &ctx);
    assert_ne!(exact.lowered().identity(), empirical.lowered().identity());
    for program in [&exact, &empirical] {
        let view = program.plan_view();
        assert_eq!(view.route(), PlanRoute::ZTransportEstimate);
        let decision = view.unit_support().expect("the route is bound to a unit");
        assert_eq!(decision.unit(), "transport.single_source_z_transport");
        // Until the unit's record is earned the gate is off, so the estimate runs whatever the
        // unit's decision; afterwards a supported unit lets it run.
        assert!(program.estimate(&ctx).is_ok());
    }
    // A different request is a different program.
    let other = surrogate(false);
    let at_one = Study::compile_z_transport(
        other.diagram,
        other.functional,
        other.data,
        Assignment::from_pairs([(X, Value::Bool(true))]),
        ExactEvaluationLimits::default(),
        &ctx,
    )
    .unwrap();
    assert_ne!(at_one.lowered().identity(), exact.lowered().identity());
}
