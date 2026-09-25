//! Two sources searched separately. Cross-source factor combination is refused.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    DependenceGroup, DistributionAvailability, Environment, EvidenceCatalog, EvidenceKind,
    EvidenceRegime, ExecutionContext, InterventionAssignment, RegimeBinding, RegimeId, RegimeKind,
    SamplingDesign, Value, VariableCoordinate, VariableDomain, VariableId,
};
use antecedent_graph::{Admg, DenseNodeId};
use antecedent_identify::{
    decide_two_source_z_transport, SidLimits, TwoSourceZTransportDecision,
    TwoSourceZTransportQuery, ZTransportSourceSpec,
};

const W: VariableId = VariableId::from_raw(0);
const Z: VariableId = VariableId::from_raw(1);
const X: VariableId = VariableId::from_raw(2);
const Y: VariableId = VariableId::from_raw(3);

fn surrogate_graph() -> Admg {
    let mut graph = Admg::with_variables(4);
    for (from, to) in [(0, 1), (1, 2), (2, 3), (0, 3)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    for (a, b) in [(0, 3), (1, 3), (1, 2)] {
        graph.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    graph
}

fn disconnected_graph() -> Admg {
    let mut graph = Admg::with_variables(4);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    graph.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    graph
}

fn environment(name: &str) -> Environment {
    Environment::try_new(
        name,
        [W, Z, X, Y]
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

fn experiment_catalog(population: &str) -> EvidenceCatalog {
    let regime = EvidenceRegime::try_new(
        RegimeId::from_raw(0),
        RegimeKind::Experimental,
        EvidenceKind::Available,
        [Z],
        [InterventionAssignment { variable: Z, value: Value::Bool(false) }],
        [W, X, Y],
        population,
        DistributionAvailability::Joint,
    )
    .unwrap();
    EvidenceCatalog::try_new(
        [environment("alpha"), environment("beta"), environment("target")],
        [regime],
        [RegimeBinding {
            dataset_identity: None,
            regime: RegimeId::from_raw(0),
            snapshot_identity: Arc::from(format!("do-z-{population}")),
            schema_names: Arc::from([]),
            sampling: SamplingDesign::Independent,
            weights: None,
            dependence: DependenceGroup::IndependentStudies,
        }],
        None,
    )
    .unwrap()
}

fn empty_catalog() -> EvidenceCatalog {
    EvidenceCatalog::try_new(
        [environment("alpha"), environment("beta"), environment("target")],
        [],
        [],
        None,
    )
    .unwrap()
}

fn surrogate_query() -> TwoSourceZTransportQuery {
    let source = |population: &str| ZTransportSourceSpec {
        population: Arc::from(population),
        controllable: Arc::from([Z]),
        experiment_assignment: Arc::from([InterventionAssignment {
            variable: Z,
            value: Value::Bool(false),
        }]),
        selection_targets: Arc::from([]),
    };
    TwoSourceZTransportQuery {
        outcomes: Arc::from([Y]),
        treatments: Arc::from([X]),
        target: Arc::from("target"),
        sources: [source("alpha"), source("beta")],
    }
}

#[test]
fn one_source_with_the_surrogate_experiment_identifies_and_names_that_source() {
    let decision = decide_two_source_z_transport(
        &surrogate_graph(),
        &surrogate_query(),
        [&experiment_catalog("alpha"), &empty_catalog()],
        SidLimits::default(),
        &ExecutionContext::for_tests(71),
    )
    .unwrap();
    let TwoSourceZTransportDecision::Identified { source, derivation } = decision else {
        panic!("the source that holds the cited experiment must identify on its own");
    };
    assert_eq!(source.as_ref(), "alpha");
    assert_eq!(derivation.to_record().rules, ["ztr.surrogate_factorization"]);
}

#[test]
fn both_disconnected_controls_return_one_obstruction_with_both_terminals() {
    let source = |population: &str| ZTransportSourceSpec {
        population: Arc::from(population),
        controllable: Arc::from([VariableId::from_raw(2), VariableId::from_raw(3)]),
        experiment_assignment: Arc::from([]),
        selection_targets: Arc::from([]),
    };
    let query = TwoSourceZTransportQuery {
        outcomes: Arc::from([VariableId::from_raw(1)]),
        treatments: Arc::from([VariableId::from_raw(0)]),
        target: Arc::from("target"),
        sources: [source("alpha"), source("beta")],
    };
    let empty = empty_catalog();
    let decision = decide_two_source_z_transport(
        &disconnected_graph(),
        &query,
        [&empty, &empty],
        SidLimits::default(),
        &ExecutionContext::for_tests(72),
    )
    .unwrap();
    let TwoSourceZTransportDecision::ProvenNonTransportable { obstructions } = decision else {
        panic!("both line-11 sources must be one obstruction");
    };
    assert_eq!(obstructions[0].query().source.as_ref(), "alpha");
    assert_eq!(obstructions[1].query().source.as_ref(), "beta");
    assert_eq!(obstructions[0].to_record().terminal, obstructions[1].to_record().terminal);
    assert_eq!(
        obstructions[0].to_record().terminal.rules.last().map(String::as_str),
        Some("ztr.line11.fail")
    );
}

#[test]
fn a_factor_from_each_source_is_not_combined() {
    let decision = decide_two_source_z_transport(
        &surrogate_graph(),
        &surrogate_query(),
        [&experiment_catalog("beta"), &experiment_catalog("alpha")],
        SidLimits::default(),
        &ExecutionContext::for_tests(73),
    )
    .unwrap();
    assert!(matches!(
        decision,
        TwoSourceZTransportDecision::NotCertified {
            reason: "z_transport.multi_source_combination_not_searched",
        }
    ));
}
