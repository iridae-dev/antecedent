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
    ComponentFactorization, SidLimits, TwoSourceZTransportDecision, TwoSourceZTransportQuery,
    ZTransportSourceSpec, decide_two_source_z_transport,
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

fn split_graph() -> Admg {
    let mut graph = Admg::with_variables(4);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(3)).unwrap();
    graph
}

fn component_catalog(
    population: &str,
    intervention: VariableId,
    outcome: VariableId,
) -> EvidenceCatalog {
    let regime = EvidenceRegime::try_new(
        RegimeId::from_raw(0),
        RegimeKind::Experimental,
        EvidenceKind::Available,
        [intervention],
        [InterventionAssignment { variable: intervention, value: Value::Bool(false) }],
        [outcome],
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
            snapshot_identity: Arc::from(format!("do-{intervention}-{population}")),
            schema_names: Arc::from([]),
            sampling: SamplingDesign::Independent,
            weights: None,
            dependence: DependenceGroup::IndependentStudies,
        }],
        None,
    )
    .unwrap()
}

#[test]
fn complementary_joint_regimes_identify_disconnected_outcome_components() {
    let graph = split_graph();
    let sources = [
        ZTransportSourceSpec {
            population: Arc::from("alpha"),
            controllable: Arc::from([W]),
            experiment_assignment: Arc::from([InterventionAssignment {
                variable: W,
                value: Value::Bool(false),
            }]),
            selection_targets: Arc::from([]),
        },
        ZTransportSourceSpec {
            population: Arc::from("beta"),
            controllable: Arc::from([X]),
            experiment_assignment: Arc::from([InterventionAssignment {
                variable: X,
                value: Value::Bool(false),
            }]),
            selection_targets: Arc::from([]),
        },
    ];
    let query = TwoSourceZTransportQuery {
        outcomes: Arc::from([Z, Y]),
        treatments: Arc::from([W, X]),
        target: Arc::from("target"),
        sources: sources.clone(),
    };
    let alpha = component_catalog("alpha", W, Z);
    let beta = component_catalog("beta", X, Y);
    let decision = decide_two_source_z_transport(
        &graph,
        &query,
        [&alpha, &beta],
        SidLimits::default(),
        &ExecutionContext::for_tests(75),
    )
    .unwrap();
    let TwoSourceZTransportDecision::CombinedIdentified { components, factorization } = decision
    else {
        panic!(
            "complementary joint factors should identify both disconnected components: {decision:?}"
        );
    };
    assert_eq!(factorization, ComponentFactorization::DisconnectedGraphComponents);
    assert_eq!(components[0].source.as_ref(), "alpha");
    assert_eq!(components[0].derivation.to_record().rules, ["ztr.source_exchange_joint"]);
    assert_eq!(components[1].source.as_ref(), "beta");
    assert_eq!(components[1].derivation.to_record().rules, ["ztr.source_exchange_joint"]);
    assert!(
        components[0]
            .derivation
            .inspect_proof(&alpha)
            .factors
            .iter()
            .all(|factor| factor.failure.is_none())
    );
    assert!(
        components[1]
            .derivation
            .inspect_proof(&beta)
            .factors
            .iter()
            .all(|factor| factor.failure.is_none())
    );

    // A separately measured marginal cannot bind either joint component law.
    let separate = EvidenceRegime::try_new(
        RegimeId::from_raw(0),
        RegimeKind::Experimental,
        EvidenceKind::Available,
        [W],
        [InterventionAssignment { variable: W, value: Value::Bool(false) }],
        [Z],
        "alpha",
        DistributionAvailability::SeparateMarginals { variables: Arc::from([Z]) },
    )
    .unwrap();
    let alpha_marginal = EvidenceCatalog::try_new(
        [environment("alpha"), environment("beta"), environment("target")],
        [separate],
        [RegimeBinding {
            dataset_identity: None,
            regime: RegimeId::from_raw(0),
            snapshot_identity: Arc::from("alpha-marginal"),
            schema_names: Arc::from([]),
            sampling: SamplingDesign::Independent,
            weights: None,
            dependence: DependenceGroup::IndependentStudies,
        }],
        None,
    )
    .unwrap();
    let refused = decide_two_source_z_transport(
        &graph,
        &query,
        [&alpha_marginal, &beta],
        SidLimits::default(),
        &ExecutionContext::for_tests(76),
    )
    .unwrap();
    assert!(matches!(refused, TwoSourceZTransportDecision::NotCertified { .. }));
}

/// One connected graph W→Z, X→Y, W↔X. It is a single component (the disconnected
/// route does not apply), yet under do(W, X) the confounding arc W↔X is cut, so Z
/// and Y are m-separated given {W, X} and P*_{w,x}(z, y) = P*_w(z)·P*_x(y). Alpha
/// controls W and supplies the Z factor; beta controls X and supplies the Y
/// factor. Neither source alone measures the joint {Z, Y}.
fn connected_confounded_graph() -> Admg {
    let mut graph = Admg::with_variables(4);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap(); // W→Z
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(3)).unwrap(); // X→Y
    graph.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap(); // W↔X
    graph
}

fn connected_complementary_query() -> TwoSourceZTransportQuery {
    TwoSourceZTransportQuery {
        outcomes: Arc::from([Z, Y]),
        treatments: Arc::from([W, X]),
        target: Arc::from("target"),
        sources: [
            ZTransportSourceSpec {
                population: Arc::from("alpha"),
                controllable: Arc::from([W]),
                experiment_assignment: Arc::from([InterventionAssignment {
                    variable: W,
                    value: Value::Bool(false),
                }]),
                selection_targets: Arc::from([]),
            },
            ZTransportSourceSpec {
                population: Arc::from("beta"),
                controllable: Arc::from([X]),
                experiment_assignment: Arc::from([InterventionAssignment {
                    variable: X,
                    value: Value::Bool(false),
                }]),
                selection_targets: Arc::from([]),
            },
        ],
    }
}

#[test]
fn connected_complementary_factors_identify_across_two_sources() {
    let graph = connected_confounded_graph();
    let query = connected_complementary_query();
    let alpha = component_catalog("alpha", W, Z);
    let beta = component_catalog("beta", X, Y);
    let decision = decide_two_source_z_transport(
        &graph,
        &query,
        [&alpha, &beta],
        SidLimits::default(),
        &ExecutionContext::for_tests(80),
    )
    .unwrap();
    let TwoSourceZTransportDecision::CombinedIdentified { components, factorization } = decision
    else {
        panic!("a connected intervention-separated target must combine two sources: {decision:?}");
    };
    assert_eq!(factorization, ComponentFactorization::InterventionSeparatedGroups);
    // The Z factor is transported from alpha under do(W); the Y factor from beta
    // under do(X). Each derivation binds only to its own source's regimes.
    assert_eq!(components[0].source.as_ref(), "alpha");
    assert_eq!(components[0].outcomes.as_ref(), [Z]);
    assert_eq!(components[0].treatments.as_ref(), [W]);
    assert_eq!(components[1].source.as_ref(), "beta");
    assert_eq!(components[1].outcomes.as_ref(), [Y]);
    assert_eq!(components[1].treatments.as_ref(), [X]);
    assert!(
        components[0]
            .derivation
            .inspect_proof(&alpha)
            .factors
            .iter()
            .all(|factor| factor.failure.is_none())
    );
    assert!(
        components[1]
            .derivation
            .inspect_proof(&beta)
            .factors
            .iter()
            .all(|factor| factor.failure.is_none())
    );
}

/// The same connected graph plus a bidirected Z↔Y between the two outcomes. That
/// arc is not cut by do(W, X), so Z and Y stay m-connected given the treatments
/// and P*_{w,x}(z, y) has one connected c-factor. Identifying it would require a
/// single joint law over do(W, X) measuring {Z, Y}, which no single source
/// supplies and which the joint-regime rule forbids fabricating. Both sources
/// hold binding do experiments, so this is a structural refusal, not missing
/// evidence.
#[test]
fn connected_confounded_outcomes_refuse_a_fabricated_cross_source_joint() {
    let mut graph = connected_confounded_graph();
    graph.insert_bidirected(DenseNodeId::from_raw(1), DenseNodeId::from_raw(3)).unwrap(); // Z↔Y
    let query = connected_complementary_query();
    let alpha = component_catalog("alpha", W, Z);
    let beta = component_catalog("beta", X, Y);
    let decision = decide_two_source_z_transport(
        &graph,
        &query,
        [&alpha, &beta],
        SidLimits::default(),
        &ExecutionContext::for_tests(81),
    )
    .unwrap();
    assert!(
        matches!(
            decision,
            TwoSourceZTransportDecision::NotCertified {
                reason: "z_transport.multi_source_combination_not_searched",
            }
        ),
        "a connected c-factor spanning both sources must be a typed refusal, not missing \
         evidence: {decision:?}"
    );
}

/// Two disjoint bows X1→Y1, X1↔Y1 and X2→Y2, X2↔Y2 with outcomes {Y1, Y2} and
/// treatments {X1, X2}. Alone, a source that controls only X1 (or only X2)
/// reaches line 11, yet P*_{x1,x2}(y1, y2) = P^α_{x1}(y1)·P^β_{x2}(y2) combines
/// one factor from each source. Two single-family terminals therefore never
/// certify non-transportability for the union of the families; the decision
/// stays a named refusal to search that combination.
#[test]
fn two_line11_terminals_from_different_families_do_not_certify_an_obstruction() {
    let mut graph = Admg::with_variables(4);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    graph.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(3)).unwrap();
    graph.insert_bidirected(DenseNodeId::from_raw(2), DenseNodeId::from_raw(3)).unwrap();
    let source = |population: &str, controllable: VariableId| ZTransportSourceSpec {
        population: Arc::from(population),
        controllable: Arc::from([controllable]),
        experiment_assignment: Arc::from([InterventionAssignment {
            variable: controllable,
            value: Value::Bool(false),
        }]),
        selection_targets: Arc::from([]),
    };
    let query = TwoSourceZTransportQuery {
        outcomes: Arc::from([VariableId::from_raw(1), VariableId::from_raw(3)]),
        treatments: Arc::from([VariableId::from_raw(0), VariableId::from_raw(2)]),
        target: Arc::from("target"),
        sources: [
            source("alpha", VariableId::from_raw(0)),
            source("beta", VariableId::from_raw(2)),
        ],
    };
    let empty = empty_catalog();
    let decision = decide_two_source_z_transport(
        &graph,
        &query,
        [&empty, &empty],
        SidLimits::default(),
        &ExecutionContext::for_tests(74),
    )
    .unwrap();
    assert!(
        matches!(
            decision,
            TwoSourceZTransportDecision::NotCertified {
                reason: "z_transport.multi_source_combination_not_searched",
            }
        ),
        "{decision:?}"
    );
}
