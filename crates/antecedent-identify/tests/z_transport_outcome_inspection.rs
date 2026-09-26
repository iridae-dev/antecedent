//! Every bounded single-source z-transport outcome exposes structured
//! inspection, not only a reason string: an identified proof carries its
//! factor obligations, missing evidence names the unbound factor, a
//! not-certified identification carries the explored region and a typed
//! reason, a line-11 obstruction carries its terminal record, and an exhausted
//! search carries a limits receipt with what it consumed.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    DependenceGroup, DistributionAvailability, Environment, EvidenceCatalog, EvidenceKind,
    EvidenceRegime, ExecutionContext, InterventionAssignment, RegimeBinding, RegimeId, RegimeKind,
    SamplingDesign, Value, VariableCoordinate, VariableDomain, VariableId,
};
use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram};
use antecedent_identify::{
    SidLimits, ZTransportBudgetKind, ZTransportDecision, ZTransportNotCertifiedKind,
    ZTransportOutcome, ZTransportQuery, ZTransportResult, decide_z_transport_inspecting,
    decide_z_transport_with_catalog, identify_z_transport,
};

const W: VariableId = VariableId::from_raw(0);
const Z: VariableId = VariableId::from_raw(1);
const X: VariableId = VariableId::from_raw(2);
const Y: VariableId = VariableId::from_raw(3);

/// The registered surrogate graph W→Z→X→Y, W→Y, W↔Y, Z↔Y, Z↔X.
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

/// A do(Z=false) joint over {W, X, Y} for `source`, sufficient to bind the
/// registered surrogate formula.
fn surrogate_catalog() -> EvidenceCatalog {
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
    EvidenceCatalog::try_new(
        [environment("source"), environment("target")],
        [regime],
        [RegimeBinding {
            dataset_identity: None,
            regime: RegimeId::from_raw(0),
            snapshot_identity: Arc::from("do-z-source"),
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
    EvidenceCatalog::try_new([environment("source"), environment("target")], [], [], None).unwrap()
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

#[test]
fn identified_exposes_a_proof_graph_whose_factors_all_bind() {
    let diagram =
        SelectionDiagram::try_new(surrogate_graph(), Arc::<[VariableId]>::from([])).unwrap();
    let query = surrogate_query();
    let catalog = surrogate_catalog();
    let decision = decide_z_transport_with_catalog(
        &diagram,
        &query,
        &catalog,
        SidLimits::default(),
        &ExecutionContext::for_tests(101),
    )
    .unwrap();
    let ZTransportDecision::Identified(proof) = decision else {
        panic!("the registered surrogate graph with its cited joint must identify");
    };
    let inspection = proof.inspect_proof(&catalog);
    assert!(!inspection.rules.is_empty(), "the identified proof exposes its checked rules");
    assert!(!inspection.factors.is_empty(), "the identified proof exposes factor obligations");
    assert!(
        inspection.factors.iter().all(|factor| factor.failure.is_none()),
        "every factor of an identified proof binds to available evidence",
    );
}

#[test]
fn missing_evidence_names_the_unbound_cited_factor() {
    let diagram =
        SelectionDiagram::try_new(surrogate_graph(), Arc::<[VariableId]>::from([])).unwrap();
    let query = surrogate_query();
    let empty = empty_catalog();
    let decision = decide_z_transport_with_catalog(
        &diagram,
        &query,
        &empty,
        SidLimits::default(),
        &ExecutionContext::for_tests(102),
    )
    .unwrap();
    let ZTransportDecision::MissingEvidence { missing } = decision else {
        panic!("a checked formula whose cited joint is absent is missing evidence: {decision:?}");
    };
    // The structured object names the missing cited factor, not just a string.
    assert!(format!("{missing:?}").contains("CitedFactor"));
}

#[test]
fn not_certified_exposes_the_explored_region_and_a_typed_reason() {
    // A selection target on the registered surrogate graph disqualifies the
    // registered formula; the bounded identifier reaches a checked line-11
    // terminal and, with no catalog, reports the query as not certified while
    // exposing the region it explored.
    let diagram =
        SelectionDiagram::try_new(surrogate_graph(), Arc::<[VariableId]>::from([Y])).unwrap();
    let query = surrogate_query();
    let result = identify_z_transport(
        &diagram,
        &query,
        SidLimits::default(),
        &ExecutionContext::for_tests(103),
    )
    .unwrap();
    let ZTransportResult::NotCertified { reason, inspection } = result else {
        panic!("a selected surrogate graph the bounded identifier cannot certify");
    };
    assert_eq!(reason, "z_transport.no_checked_recursive_formula");
    // Honesty: the search built no expression, so the record is the explored
    // region, not a fabricated proof graph. It names a typed reason and the
    // recursive rules the search actually applied.
    assert_eq!(inspection.kind, ZTransportNotCertifiedKind::CheckedLine11Terminal);
    assert!(
        inspection.steps_explored >= 1,
        "a not-certified search charges at least one recursive step",
    );
    assert!(
        !inspection.explored_rules.is_empty(),
        "the explored region names the rules the search applied",
    );
    assert!(inspection.explored_rules.iter().all(|rule| rule.starts_with("ztr.")));
    assert!(
        inspection.explored_rules.iter().any(|rule| rule == "ztr.line11.fail"),
        "the explored trace records the line-11 terminal the search reached",
    );
}

#[test]
fn proven_non_transportable_exposes_a_terminal_record() {
    // X→Y, X↔Y is a treatment/outcome hedge; the declared controllables cannot
    // exchange the active treatment, so the search reaches a checked line 11.
    let mut graph = Admg::with_variables(4);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    graph.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let diagram = SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from([])).unwrap();
    let query = ZTransportQuery {
        outcomes: Arc::from([VariableId::from_raw(1)]),
        treatments: Arc::from([VariableId::from_raw(0)]),
        controllable: Arc::from([VariableId::from_raw(2), VariableId::from_raw(3)]),
        experiment_assignment: Arc::from([]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    };
    let decision = decide_z_transport_with_catalog(
        &diagram,
        &query,
        &hedge_catalog(),
        SidLimits::default(),
        &ExecutionContext::for_tests(104),
    )
    .unwrap();
    let ZTransportDecision::ProvenNonTransportable(obstruction) = decision else {
        panic!("a treatment/outcome hedge with uncontrollable X reaches line 11: {decision:?}");
    };
    let record = obstruction.to_record();
    assert!(!record.terminal.treatments.is_empty(), "the terminal record names its treatments");
    assert_eq!(record.terminal.rules.last().map(String::as_str), Some("ztr.line11.fail"));
}

/// A catalog with only the two environments the hedge query names.
fn hedge_catalog() -> EvidenceCatalog {
    let variables = (0..4)
        .map(|raw| VariableCoordinate {
            variable: VariableId::from_raw(raw),
            domain: VariableDomain::Binary,
            unit: None,
        })
        .collect::<Vec<_>>();
    EvidenceCatalog::try_new(
        [
            Environment::try_new("source", variables.clone(), []).unwrap(),
            Environment::try_new("target", variables, []).unwrap(),
        ],
        [],
        [],
        None,
    )
    .unwrap()
}

#[test]
fn exhausted_search_exposes_a_limits_receipt_with_what_it_consumed() {
    let diagram =
        SelectionDiagram::try_new(surrogate_graph(), Arc::<[VariableId]>::from([Y])).unwrap();
    let query = surrogate_query();
    let catalog = surrogate_catalog();
    // A one-step budget stops the recursive search before it can decide.
    let outcome = decide_z_transport_inspecting(
        &diagram,
        &query,
        &catalog,
        SidLimits { steps: 1, depth: 256 },
        &ExecutionContext::for_tests(105),
    )
    .unwrap();
    let ZTransportOutcome::Exhausted(receipt) = outcome else {
        panic!("a one-step budget must exhaust rather than decide: {outcome:?}");
    };
    assert_eq!(receipt.budget, ZTransportBudgetKind::Steps);
    assert_eq!(receipt.steps_limit, 1);
    assert_eq!(receipt.depth_limit, 256);
    assert!(
        receipt.steps_consumed.is_some_and(|steps| steps >= 1),
        "the receipt reports the engine's own step count at the point it stopped",
    );
    assert!(receipt.depth_reached.is_some(), "the receipt reports the deepest level reached");
}

#[test]
fn a_zero_step_budget_receipt_reports_no_search_was_entered() {
    let diagram =
        SelectionDiagram::try_new(surrogate_graph(), Arc::<[VariableId]>::from([])).unwrap();
    let query = surrogate_query();
    let catalog = surrogate_catalog();
    let outcome = decide_z_transport_inspecting(
        &diagram,
        &query,
        &catalog,
        SidLimits { steps: 0, depth: 256 },
        &ExecutionContext::for_tests(106),
    )
    .unwrap();
    let ZTransportOutcome::Exhausted(receipt) = outcome else {
        panic!("a zero-step budget exhausts before deciding: {outcome:?}");
    };
    assert_eq!(receipt.budget, ZTransportBudgetKind::Steps);
    assert_eq!(receipt.steps_limit, 0);
    assert_eq!(receipt.steps_consumed, Some(0));
    assert!(receipt.explored_rules.is_empty());
}
