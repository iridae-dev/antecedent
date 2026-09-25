//! Positive z-transport from the joints a formula cites.
//!
//! Truth is enumerated from structural equations. These tests do not call the
//! identifier to compute a probability.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    DependenceGroup, DistributionAvailability, Environment, EvidenceCatalog, EvidenceKind,
    EvidenceRegime, ExecutionContext, InterventionAssignment, RegimeBinding, RegimeId, RegimeKind,
    SamplingDesign, Value, VariableCoordinate, VariableDomain, VariableId,
};
use antecedent_estimate::{
    evaluate_exact_z_transport, nominal_z_transport_interval, Z_TRANSPORT_INTERVAL_NOT_MEASURED,
};
use antecedent_expr::{
    Assignment, DiscreteAxis, ExactDiscreteLaw, ExactEvaluationLimits, ExactTransportData,
    LawTolerance,
};
use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram};
use antecedent_identify::{
    bind_z_transport_catalog, decide_z_transport_with_catalog, identify_z_transport, SidLimits,
    ZTransportDecision, ZTransportQuery, ZTransportResult,
};

const W: VariableId = VariableId::from_raw(0);
const Z: VariableId = VariableId::from_raw(1);
const X: VariableId = VariableId::from_raw(2);
const Y: VariableId = VariableId::from_raw(3);

fn binary_axis(variable: VariableId) -> DiscreteAxis {
    DiscreteAxis { variable, values: Arc::from([Value::Bool(false), Value::Bool(true)]) }
}

fn y_risk(distribution: &antecedent_expr::ExactDistribution) -> f64 {
    distribution
        .atoms
        .iter()
        .zip(distribution.probabilities.iter())
        .filter(|(atom, _)| atom[0] == Value::Bool(true))
        .map(|(_, probability)| probability)
        .sum()
}

fn surrogate_diagram() -> SelectionDiagram {
    let mut graph = Admg::with_variables(4);
    for (from, to) in [(0, 1), (1, 2), (2, 3), (0, 3)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    for (a, b) in [(0, 3), (1, 3), (1, 2)] {
        graph.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from([])).unwrap()
}

fn surrogate_query() -> ZTransportQuery {
    ZTransportQuery {
        outcomes: Arc::from([Y]),
        treatments: Arc::from([X]),
        controllable: Arc::from([Z]),
        experiment_assignment: Arc::from([InterventionAssignment {
            variable: Z,
            value: Value::Bool(false),
        }]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    }
}

fn binding(regime: u32) -> RegimeBinding {
    RegimeBinding {
        dataset_identity: None,
        regime: RegimeId::from_raw(regime),
        snapshot_identity: Arc::from(format!("snapshot-{regime}")),
        schema_names: Arc::from([]),
        sampling: SamplingDesign::Independent,
        weights: None,
        dependence: DependenceGroup::IndependentStudies,
    }
}

fn environment(name: &str, variables: impl IntoIterator<Item = VariableId>) -> Environment {
    Environment::try_new(
        name,
        variables
            .into_iter()
            .map(|variable| VariableCoordinate {
                variable,
                domain: VariableDomain::Binary,
                unit: None,
            })
            .collect::<Vec<_>>(),
        [],
    )
    .unwrap()
}

/// Product law P(W)P(X)P(Y|X) on axes (W, X, Y). P(Y=1 | X=0) = 0.2.
fn cited_wyx_probabilities() -> Vec<f64> {
    let mut probabilities = Vec::new();
    for w in [false, true] {
        for x in [false, true] {
            for y in [false, true] {
                let p_w = if w { 0.25 } else { 0.75 };
                let p_x = if x { 0.35 } else { 0.65 };
                let p_y = if y == x { 0.8 } else { 0.2 };
                probabilities.push(p_w * p_x * p_y);
            }
        }
    }
    probabilities
}

#[test]
fn surrogate_decision_uses_only_the_cited_source_margin() {
    let diagram = surrogate_diagram();
    let query = surrogate_query();
    let measured = Arc::<[VariableId]>::from([W, X, Y]);
    let regime = EvidenceRegime::try_new(
        RegimeId::from_raw(0),
        RegimeKind::Experimental,
        EvidenceKind::Available,
        [Z],
        [InterventionAssignment { variable: Z, value: Value::Bool(false) }],
        Arc::clone(&measured),
        "source",
        DistributionAvailability::Joint,
    )
    .unwrap();
    let catalog = EvidenceCatalog::try_new(
        [environment("source", [W, Z, X, Y])],
        [regime],
        [binding(0)],
        None,
    )
    .unwrap();
    let context = ExecutionContext::for_tests(61);
    let ZTransportDecision::Identified(derivation) =
        decide_z_transport_with_catalog(&diagram, &query, &catalog, SidLimits::default(), &context)
            .unwrap()
    else {
        panic!("the cited do(Z) margin must identify without the experiment power set");
    };
    assert_eq!(derivation.to_record().rules, ["ztr.surrogate_factorization"]);
    let law = ExactDiscreteLaw::try_new(
        "source",
        RegimeId::from_raw(0),
        [antecedent_expr::InterventionAssignment::concrete(Z, Value::Bool(false))],
        [binary_axis(W), binary_axis(X), binary_axis(Y)],
        cited_wyx_probabilities(),
        "snapshot-0",
        LawTolerance::default(),
    )
    .unwrap();
    let data = ExactTransportData::try_new([law], 64).unwrap();
    let bound = bind_z_transport_catalog(&diagram, &query, &derivation, &catalog).unwrap();
    let result = evaluate_exact_z_transport(
        &bound,
        data,
        Assignment::from_pairs([(X, Value::Bool(false))]),
        ExactEvaluationLimits::default(),
        &context,
    )
    .unwrap();
    let risk = y_risk(&result);
    assert!((risk - 0.2).abs() < 1e-12, "P(Y=1 | do(X=0))={risk}");
}

#[test]
fn empirical_cited_margin_publishes_a_nominal_interval_around_the_exact_point() {
    let diagram = surrogate_diagram();
    let query = surrogate_query();
    let regime = EvidenceRegime::try_new(
        RegimeId::from_raw(0),
        RegimeKind::Experimental,
        EvidenceKind::Available,
        [Z],
        [InterventionAssignment { variable: Z, value: Value::Bool(false) }],
        [W, X, Y],
        "source",
        DistributionAvailability::Joint,
    )
    .unwrap();
    let catalog = EvidenceCatalog::try_new(
        [environment("source", [W, Z, X, Y])],
        [regime],
        [binding(0)],
        None,
    )
    .unwrap();
    let probabilities = cited_wyx_probabilities();
    let counts = probabilities.iter().map(|p| (p * 20_000.0).round() as u64).collect::<Vec<_>>();
    let total = counts.iter().sum::<u64>() as f64;
    let empirical = counts.iter().map(|count| *count as f64 / total).collect::<Vec<_>>();
    let law = ExactDiscreteLaw::try_empirical(
        "source",
        RegimeId::from_raw(0),
        [antecedent_expr::InterventionAssignment::concrete(Z, Value::Bool(false))],
        [binary_axis(W), binary_axis(X), binary_axis(Y)],
        empirical,
        "snapshot-0",
        LawTolerance::default(),
    )
    .unwrap()
    .with_empirical_counts(counts)
    .unwrap();
    let data = ExactTransportData::try_new([law], 64).unwrap();
    let context = ExecutionContext::for_tests(64);
    let ZTransportDecision::Identified(derivation) =
        decide_z_transport_with_catalog(&diagram, &query, &catalog, SidLimits::default(), &context)
            .unwrap()
    else {
        panic!("cited margin identifies");
    };
    let bound = bind_z_transport_catalog(&diagram, &query, &derivation, &catalog).unwrap();
    let request = Assignment::from_pairs([(X, Value::Bool(false))]);
    let point = evaluate_exact_z_transport(
        &bound,
        data.clone(),
        request.clone(),
        ExactEvaluationLimits::default(),
        &context,
    )
    .unwrap();
    let risk = y_risk(&point);
    let interval = nominal_z_transport_interval(
        &bound,
        &data,
        request,
        ExactEvaluationLimits::default(),
        99,
        0.95,
        &context,
    )
    .unwrap()
    .expect("iid counts publish a nominal interval");
    assert_eq!(interval.reason.as_ref(), Z_TRANSPORT_INTERVAL_NOT_MEASURED);
    assert!(interval.mean_intervals[0].1 <= risk && risk <= interval.mean_intervals[0].2);
    assert!((0.2 - risk).abs() < 1e-3);

    let mut dependent = catalog;
    let mut bindings = dependent.bindings.to_vec();
    bindings[0].dependence = DependenceGroup::LinkedUnits;
    dependent.bindings = bindings.into();
    let bound = bind_z_transport_catalog(&diagram, &query, &derivation, &dependent).unwrap();
    let withheld = nominal_z_transport_interval(
        &bound,
        &data,
        Assignment::from_pairs([(X, Value::Bool(false))]),
        ExactEvaluationLimits::default(),
        99,
        0.95,
        &context,
    )
    .unwrap()
    .unwrap_err();
    assert_eq!(withheld, "transport.unsupported_dependence");
}

#[test]
fn selection_on_a_non_outcome_still_transports_and_matches_scm_truth() {
    let mut graph = Admg::with_variables(5);
    for (from, to) in [(0, 1), (1, 2), (2, 3), (0, 3)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    for (a, b) in [(0, 3), (1, 3), (1, 2)] {
        graph.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    let u = VariableId::from_raw(4);
    let diagram = SelectionDiagram::try_new(graph.clone(), Arc::<[VariableId]>::from([u])).unwrap();
    let blocked = SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from([Y])).unwrap();
    let query = surrogate_query();
    assert!(
        matches!(
            identify_z_transport(&blocked, &query).unwrap(),
            ZTransportResult::NotCertified { .. }
        ),
        "selection on the outcome stays non-identified"
    );
    let context = ExecutionContext::for_tests(62);
    let ZTransportResult::Identified(proof) = identify_z_transport(&diagram, &query).unwrap()
    else {
        panic!("selection on an isolated non-outcome must still transport");
    };
    let rules = proof.to_record().rules;
    assert!(
        rules.iter().any(|rule| rule.starts_with("ztr.line10.source_exchange")),
        "selection-positive transport must be the recursive exchange: {rules:?}"
    );

    let variables = [W, Z, X, Y, u];
    let mut target = vec![0.0; 32];
    let mut source = vec![0.0; 16];
    let mut truth = 0.0_f64;
    let index = |values: &[usize]| values.iter().fold(0, |index, bit| (index << 1) | bit);
    for a in 0..=1usize {
        for b in 0..=1usize {
            for c in 0..=1usize {
                for extra in 0..=1usize {
                    let mass = (if a == 1 { 0.25 } else { 0.75 })
                        * (if b == 1 { 0.20 } else { 0.80 })
                        * (if c == 1 { 0.35 } else { 0.65 })
                        * 0.5;
                    let w = a;
                    let z = w ^ b ^ c;
                    let x = z ^ c;
                    let y = x ^ (w & b);
                    target[index(&[w, z, x, y, extra])] += mass;
                    // do(Z=0) sets Z. X then copies C. U stays an independent bit.
                    let x_do_z = c;
                    let y_do_z = x_do_z ^ (w & b);
                    source[index(&[w, x_do_z, y_do_z, extra])] += mass;
                    // do(X=0): Y = W & B. U is an independent non-outcome.
                    if (w & b) == 1 {
                        truth += mass;
                    }
                }
            }
        }
    }
    assert!((truth - 0.05).abs() < 1e-12, "independent SCM truth={truth}");

    let target_regime = EvidenceRegime::try_new(
        RegimeId::from_raw(0),
        RegimeKind::Observational,
        EvidenceKind::Available,
        [],
        [],
        variables,
        "target",
        DistributionAvailability::Joint,
    )
    .unwrap();
    let source_regime = EvidenceRegime::try_new(
        RegimeId::from_raw(1),
        RegimeKind::Experimental,
        EvidenceKind::Available,
        [Z],
        [InterventionAssignment { variable: Z, value: Value::Bool(false) }],
        [W, X, Y, u],
        "source",
        DistributionAvailability::Joint,
    )
    .unwrap();
    let catalog = EvidenceCatalog::try_new(
        [environment("source", variables), environment("target", variables)],
        [target_regime, source_regime],
        [binding(0), binding(1)],
        None,
    )
    .unwrap();
    let ZTransportDecision::Identified(derivation) =
        decide_z_transport_with_catalog(&diagram, &query, &catalog, SidLimits::default(), &context)
            .unwrap()
    else {
        panic!("cited selection-diagram factors must decide as identified");
    };
    let target_law = ExactDiscreteLaw::try_new(
        "target",
        RegimeId::from_raw(0),
        [],
        variables.map(binary_axis),
        target,
        "snapshot-0",
        LawTolerance::default(),
    )
    .unwrap();
    let source_law = ExactDiscreteLaw::try_new(
        "source",
        RegimeId::from_raw(1),
        [antecedent_expr::InterventionAssignment::concrete(Z, Value::Bool(false))],
        [W, X, Y, u].map(binary_axis),
        source,
        "snapshot-1",
        LawTolerance::default(),
    )
    .unwrap();
    let data = ExactTransportData::try_new([target_law, source_law], 128).unwrap();
    let bound = bind_z_transport_catalog(&diagram, &query, &derivation, &catalog).unwrap();
    let result = evaluate_exact_z_transport(
        &bound,
        data,
        Assignment::from_pairs([(X, Value::Bool(false))]),
        ExactEvaluationLimits::default(),
        &context,
    )
    .unwrap();
    let risk = y_risk(&result);
    assert!((risk - truth).abs() < 1e-12, "P(Y=1 | do(X=0))={risk}, truth={truth}");
}

#[test]
fn seven_node_three_control_exchange_matches_scm_truth() {
    let mut graph = Admg::with_variables(7);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let diagram = SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from([])).unwrap();
    let variables = (0..7).map(VariableId::from_raw).collect::<Vec<_>>();
    let query = ZTransportQuery {
        outcomes: Arc::from([VariableId::from_raw(1)]),
        treatments: Arc::from([VariableId::from_raw(0)]),
        controllable: Arc::from([
            VariableId::from_raw(0),
            VariableId::from_raw(2),
            VariableId::from_raw(3),
        ]),
        experiment_assignment: Arc::from([InterventionAssignment {
            variable: VariableId::from_raw(0),
            value: Value::Bool(true),
        }]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    };
    let regime = EvidenceRegime::try_new(
        RegimeId::from_raw(0),
        RegimeKind::Experimental,
        EvidenceKind::Available,
        [VariableId::from_raw(0)],
        [InterventionAssignment { variable: VariableId::from_raw(0), value: Value::Bool(true) }],
        [VariableId::from_raw(1)],
        "source",
        DistributionAvailability::Joint,
    )
    .unwrap();
    let catalog =
        EvidenceCatalog::try_new([environment("source", variables)], [regime], [binding(0)], None)
            .unwrap();
    let context = ExecutionContext::for_tests(63);
    let ZTransportDecision::Identified(derivation) =
        decide_z_transport_with_catalog(&diagram, &query, &catalog, SidLimits::default(), &context)
            .unwrap()
    else {
        panic!("one joint source exchange must identify past the old six-node bound");
    };
    assert_eq!(derivation.to_record().rules, ["ztr.source_exchange_joint"]);
    // Y copies X. Under do(X=1) the outcome is 1, independently of the other five nodes.
    let law = ExactDiscreteLaw::try_new(
        "source",
        RegimeId::from_raw(0),
        [antecedent_expr::InterventionAssignment::concrete(
            VariableId::from_raw(0),
            Value::Bool(true),
        )],
        [binary_axis(VariableId::from_raw(1))],
        vec![0.0, 1.0],
        "snapshot-0",
        LawTolerance::default(),
    )
    .unwrap();
    let data = ExactTransportData::try_new([law], 32).unwrap();
    let bound = bind_z_transport_catalog(&diagram, &query, &derivation, &catalog).unwrap();
    let result = evaluate_exact_z_transport(
        &bound,
        data,
        Assignment::from_pairs([(VariableId::from_raw(0), Value::Bool(true))]),
        ExactEvaluationLimits::default(),
        &context,
    )
    .unwrap();
    let risk = y_risk(&result);
    assert!((risk - 1.0).abs() < 1e-12, "P(Y=1 | do(X=1))={risk}");
}
