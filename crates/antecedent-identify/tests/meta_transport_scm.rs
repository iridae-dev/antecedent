//! Independent finite SCM enumeration for classical meta-transportability.
#![allow(clippy::cast_precision_loss, clippy::cast_possible_wrap)]
#![allow(
    clippy::cast_possible_truncation,
    reason = "test scaffolding compares exact constants and indexes with small literals"
)] // Exhaustive three-bit coordinates.
use antecedent_core::{
    DistributionAvailability, Environment, EvidenceCatalog, EvidenceKind, EvidenceRegime,
    ExecutionContext, RegimeId, RegimeKind, Value, VariableCoordinate, VariableDomain, VariableId,
};
use antecedent_expr::{
    Assignment, DiscreteAxis, ExactDiscreteLaw, ExactEvaluationLimits, ExactEvaluationPlan,
    ExactTransportData, InterventionAssignment, LawTolerance,
};
use antecedent_graph::{Admg, DenseNodeId};
use antecedent_identify::{
    ClassicalTransportResult, MetaSource, MetaTransportQuery, SidLimits, identify_meta_transport,
    verify_meta_s_hedge, verify_meta_transport,
};
use std::sync::Arc;
fn v(n: u32) -> VariableId {
    VariableId::from_raw(n)
}
fn d(n: u32) -> DenseNodeId {
    DenseNodeId::from_raw(n)
}
const EDGES: [(usize, usize); 3] = [(0, 1), (0, 2), (1, 2)];
// Integrate independent latent binary causes and independent Bernoulli mechanism noise.
// Each changed source mechanism is explicitly listed by its selection targets.
fn enumerate(
    directed: usize,
    bidirected: usize,
    selected: usize,
    setting: usize,
    intervention: usize,
    world: usize,
) -> (Vec<u32>, Vec<f64>) {
    let axes: Vec<_> = (0..3).filter(|i| intervention & (1 << i) == 0).collect();
    let mut law = vec![0.; 1 << axes.len()];
    for latent in 0..8usize {
        for state in 0..8usize {
            if state & intervention != world & intervention {
                continue;
            }
            let mut mass = 0.125;
            for node in 0..3 {
                if intervention & (1 << node) != 0 {
                    continue;
                }
                let parents: usize = EDGES
                    .iter()
                    .enumerate()
                    .filter(|(edge, (_, b))| *b == node && directed & (1 << edge) != 0)
                    .map(|(_, (a, _))| (state >> a) & 1)
                    .sum();
                let shared: usize = EDGES
                    .iter()
                    .enumerate()
                    .filter(|(edge, (a, b))| {
                        (*a == node || *b == node) && bidirected & (1 << edge) != 0
                    })
                    .map(|(edge, _)| (latent >> edge) & 1)
                    .sum();
                let probability = 0.07
                    + 0.03 * setting as f64
                    + 0.13 * parents as f64
                    + 0.11 * shared as f64
                    + if setting == 1 {
                        0.035 * (parents * shared) as f64 + if parents == 2 { 0.02 } else { 0.0 }
                    } else {
                        0.0
                    }
                    + if selected & (1 << node) != 0 { 0.09 } else { 0. };
                mass *= if state & (1 << node) != 0 { probability } else { 1. - probability };
            }
            let row = axes.iter().fold(0, |row, node| 2 * row + ((state >> node) & 1));
            law[row] += mass;
        }
    }
    (axes.into_iter().map(|n| u32::try_from(n).unwrap()).collect(), law)
}
#[test]
#[allow(clippy::too_many_lines)]
fn ordered_three_node_meta_graphs_match_every_target_atom_and_contrast() {
    let ctx = ExecutionContext::for_tests(13);
    let mut branches = std::collections::BTreeSet::new();
    let mut positives = 0;
    let mut negatives = 0;
    for directed in 0..8usize {
        for bidirected in 0..8usize {
            for (selection_a, selection_b) in
                [(1, 2), (2, 4), (4, 2), (3, 5), (6, 7), (7, 7), (0, 7)]
            {
                let mut graph = Admg::with_variables(3);
                for (edge, (a, b)) in EDGES.iter().enumerate() {
                    if directed & (1 << edge) != 0 {
                        graph.insert_directed(d(*a as u32), d(*b as u32)).unwrap();
                    }
                    if bidirected & (1 << edge) != 0 {
                        graph.insert_bidirected(d(*a as u32), d(*b as u32)).unwrap();
                    }
                }
                let selections =
                    |mask: usize| (0..3).filter(|n| mask & (1 << n) != 0).collect::<Vec<_>>();
                for outcomes in [vec![v(2)], vec![v(1), v(2)]] {
                    let query = MetaTransportQuery {
                        outcomes: outcomes.clone().into(),
                        treatments: Arc::from([v(0)]),
                        target: Arc::from("target"),
                        sources: vec![
                            MetaSource {
                                population: "a".into(),
                                selections: selections(selection_a),
                            },
                            MetaSource {
                                population: "b".into(),
                                selections: selections(selection_b),
                            },
                        ],
                    };
                    let proof = match identify_meta_transport(
                        &graph,
                        &query,
                        SidLimits::default(),
                        &ctx,
                    )
                    .unwrap()
                    {
                        ClassicalTransportResult::Identified(proof) => {
                            positives += 1;
                            verify_meta_transport(
                                &graph,
                                &query,
                                &proof,
                                SidLimits::default(),
                                &ctx,
                            )
                            .unwrap();
                            proof
                        }
                        ClassicalTransportResult::ProvenNonTransportable(w) => {
                            negatives += 1;
                            verify_meta_s_hedge(&graph, &query, &w, &ctx).unwrap();
                            continue;
                        }
                        ClassicalTransportResult::NotCertified => panic!(
                            "uncapped meta case unresolved: {directed} {bidirected} {selection_a} {selection_b}"
                        ),
                    };
                    branches.extend(proof.rules());
                    for setting in 0..2 {
                        let mut regimes = vec![];
                        let mut laws = vec![];
                        for (population, selected, offset) in
                            [("a", selection_a, 0), ("b", selection_b, 8), ("target", 0, 16)]
                        {
                            for mask in 0..if population == "target" { 1 } else { 8 } {
                                let id = RegimeId::from_raw((offset + mask) as u32);
                                let interventions: Vec<_> =
                                    (0..3).filter(|n| mask & (1 << n) != 0).map(v).collect();
                                let measured: Vec<_> =
                                    (0..3).filter(|n| mask & (1 << n) == 0).map(v).collect();
                                regimes.push(
                                    EvidenceRegime::try_new(
                                        id,
                                        if mask == 0 {
                                            RegimeKind::Observational
                                        } else {
                                            RegimeKind::Experimental
                                        },
                                        EvidenceKind::Available,
                                        interventions.clone(),
                                        [],
                                        measured,
                                        population,
                                        DistributionAvailability::Joint,
                                    )
                                    .unwrap(),
                                );
                                for world in 0..8 {
                                    if world & !mask != 0 {
                                        continue;
                                    }
                                    let (axes, masses) = enumerate(
                                        directed, bidirected, selected, setting, mask, world,
                                    );
                                    laws.push(
                                        ExactDiscreteLaw::try_new(
                                            population,
                                            id,
                                            interventions
                                                .iter()
                                                .map(|v| InterventionAssignment {
                                                    variable: *v,
                                                    value: Value::Int64(
                                                        ((world >> v.raw()) & 1) as i64,
                                                    ),
                                                })
                                                .collect::<Vec<_>>(),
                                            axes.into_iter()
                                                .map(|n| DiscreteAxis {
                                                    variable: v(n),
                                                    values: Arc::from([
                                                        Value::Int64(0),
                                                        Value::Int64(1),
                                                    ]),
                                                })
                                                .collect::<Vec<_>>(),
                                            masses,
                                            "oracle",
                                            LawTolerance::default(),
                                        )
                                        .unwrap(),
                                    );
                                }
                            }
                        }
                        let environments = [("a", selection_a), ("b", selection_b), ("target", 0)]
                            .map(|(name, selected)| {
                                Environment::try_new(
                                    name,
                                    (0..3)
                                        .map(|n| VariableCoordinate {
                                            variable: v(n),
                                            domain: VariableDomain::Binary,
                                            unit: None,
                                        })
                                        .collect::<Vec<_>>(),
                                    selections(selected).into_iter().map(v).collect::<Vec<_>>(),
                                )
                                .unwrap()
                            });
                        let catalog =
                            EvidenceCatalog::try_new(environments, regimes, [], None).unwrap();
                        let bound = proof.bind_catalog(&catalog).unwrap();
                        let data = ExactTransportData::try_new(laws, 1000).unwrap();
                        let mut means = vec![];
                        let mut expected_means = vec![];
                        for x in 0..2 {
                            let (_, truth) = enumerate(directed, bidirected, 0, setting, 1, x);
                            let expected = truth[1] + truth[3];
                            let plan = ExactEvaluationPlan::compile(
                                bound.arena(),
                                bound.root(),
                                data.clone(),
                                outcomes.clone(),
                                Assignment::from_pairs([(v(0), Value::Int64(x as i64))]),
                                ExactEvaluationLimits::default(),
                                LawTolerance::default(),
                                &ctx,
                            )
                            .unwrap();
                            let actual = plan.evaluate(&ctx).unwrap();
                            let expected_atoms = if outcomes.len() == 1 {
                                vec![1. - expected, expected]
                            } else {
                                truth
                            };
                            assert!(
                                actual
                                    .probabilities
                                    .iter()
                                    .zip(expected_atoms)
                                    .all(|(actual, expected)| (actual - expected).abs() < 1e-10),
                                "meta SCM mismatch {directed} {bidirected} {selection_a} {selection_b} {setting} {x}"
                            );
                            means.push(actual.mean(v(2)).unwrap());
                            expected_means.push(expected);
                        }
                        assert!(
                            ((means[1] - means[0]) - (expected_means[1] - expected_means[0])).abs()
                                < 1e-10
                        );
                    }
                }
            }
        }
    }
    let mut graph = Admg::with_variables(3);
    graph.insert_directed(d(0), d(1)).unwrap();
    graph.insert_directed(d(1), d(2)).unwrap();
    let query = MetaTransportQuery {
        outcomes: Arc::from([v(2)]),
        treatments: Arc::from([v(1)]),
        target: Arc::from("target"),
        sources: vec![MetaSource { population: "source".into(), selections: vec![2] }],
    };
    let ClassicalTransportResult::Identified(proof) =
        identify_meta_transport(&graph, &query, SidLimits::default(), &ctx).unwrap()
    else {
        panic!("target DAG identifies")
    };
    branches.extend(proof.rules());
    assert!(positives > 100 && negatives > 0);
    // Multi-source recursion never reaches Figure 5 line 10 proper: an available source
    // answers a state directly, or the state is an obstruction.
    for rule in [
        "sid.line1",
        "sid.line2",
        "sid.line3",
        "sid.line4",
        "sid.line7",
        "sid.line8",
        "transport.direct",
    ] {
        assert!(branches.contains(rule), "unexercised rule {rule}: {branches:?}");
    }
}

#[test]
fn four_node_meta_conformance_has_no_unchecked_obstructions() {
    let ctx = ExecutionContext::for_tests(17);
    let edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    let mut positives = 0;
    let mut negatives = 0;
    for directed in 0..64usize {
        for bidirected in 0..64usize {
            let mut graph = Admg::with_variables(4);
            for (edge, (a, b)) in edges.iter().enumerate() {
                if directed & (1 << edge) != 0 {
                    graph.insert_directed(d(*a), d(*b)).unwrap();
                }
                if bidirected & (1 << edge) != 0 {
                    graph.insert_bidirected(d(*a), d(*b)).unwrap();
                }
            }
            for (a, b) in [(2, 4), (4, 8), (6, 9), (15, 15), (0, 15)] {
                let query = MetaTransportQuery {
                    outcomes: Arc::from([v(3)]),
                    treatments: Arc::from([v(0)]),
                    target: Arc::from("target"),
                    sources: [("a", a), ("b", b)]
                        .into_iter()
                        .map(|(population, mask)| MetaSource {
                            population: population.into(),
                            selections: (0..4).filter(|n| mask & (1 << n) != 0).collect(),
                        })
                        .collect(),
                };
                match identify_meta_transport(&graph, &query, SidLimits::default(), &ctx).unwrap() {
                    ClassicalTransportResult::Identified(proof) => {
                        verify_meta_transport(&graph, &query, &proof, SidLimits::default(), &ctx)
                            .unwrap();
                        positives += 1;
                    }
                    ClassicalTransportResult::ProvenNonTransportable(witness) => {
                        verify_meta_s_hedge(&graph, &query, &witness, &ctx).unwrap();
                        negatives += 1;
                    }
                    ClassicalTransportResult::NotCertified => {
                        panic!("unchecked four-node obstruction: {directed} {bidirected} {a} {b}")
                    }
                }
            }
        }
    }
    assert!(positives > 1000 && negatives > 1000);
}
