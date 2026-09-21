//! Numerical oracle independent of symbolic ID: enumerate latent binary SCMs.
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_possible_wrap)] // All casts are binary levels or three-node coordinates.
use antecedent_core::{
    DistributionAvailability, EvidenceCatalog, EvidenceKind, EvidenceRegime, ExecutionContext,
    RegimeId, RegimeKind, Value, VariableId,
};
use antecedent_expr::{
    Assignment, DiscreteAxis, ExactDiscreteLaw, ExactEvaluationLimits, ExactEvaluationPlan,
    ExactTransportData, LawTolerance,
};
use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram};
use antecedent_identify::{
    ClassicalTransportQuery, ClassicalTransportResult, SidLimits, identify_classical_transport,
};
use std::sync::Arc;
fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}
fn d(i: u32) -> DenseNodeId {
    DenseNodeId::from_raw(i)
}
fn bernoulli(value: usize, probability: f64) -> f64 {
    if value == 1 { probability } else { 1.0 - probability }
}

#[test]
fn recursive_frontdoor_matches_independent_latent_scm_enumeration() {
    let mut graph = Admg::with_variables(3);
    graph.insert_directed(d(0), d(1)).unwrap();
    graph.insert_directed(d(1), d(2)).unwrap();
    graph.insert_bidirected(d(0), d(2)).unwrap();
    let diagram = SelectionDiagram::try_new(graph, [v(2)]).unwrap();
    let query = ClassicalTransportQuery {
        outcomes: Arc::from([v(2)]),
        treatments: Arc::from([v(0)]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    };
    let ctx = ExecutionContext::for_tests(12);
    let ClassicalTransportResult::Identified(derivation) =
        identify_classical_transport(&diagram, &query, SidLimits::default(), &ctx).unwrap()
    else {
        panic!("identifiable frontdoor");
    };
    let regime = EvidenceRegime::try_new(
        RegimeId::from_raw(7),
        RegimeKind::Observational,
        EvidenceKind::Available,
        [],
        [],
        [v(0), v(1), v(2)],
        "target",
        DistributionAvailability::Joint,
    )
    .unwrap();
    let catalog = EvidenceCatalog::try_new([], [regime], [], None).unwrap();
    let functional = derivation.bind_catalog(&catalog).unwrap();
    for latent_mass in [0.2, 0.5, 0.8] {
        for mediator_shift in [0.25, 0.55] {
            let mut observational = vec![0.0; 8];
            for u in 0..2 {
                for x in 0..2 {
                    for m in 0..2 {
                        for y in 0..2 {
                            let mass = bernoulli(u, latent_mass)
                                * bernoulli(x, if u == 1 { 0.8 } else { 0.2 })
                                * bernoulli(m, 0.15 + mediator_shift * x as f64)
                                * bernoulli(y, 0.1 + 0.45 * m as f64 + 0.2 * u as f64);
                            observational[x * 4 + m * 2 + y] += mass;
                        }
                    }
                }
            }
            let axes: Vec<_> = (0..3)
                .map(|i| DiscreteAxis {
                    variable: v(i),
                    values: Arc::from([Value::Int64(0), Value::Int64(1)]),
                })
                .collect();
            let law = ExactDiscreteLaw::try_new(
                "target",
                RegimeId::from_raw(7),
                [],
                axes,
                observational,
                "oracle",
                LawTolerance::default(),
            )
            .unwrap();
            let data = ExactTransportData::try_new([law], 1000).unwrap();
            for x in 0..2 {
                let mut truth = [0.0; 2];
                for u in 0..2 {
                    for m in 0..2 {
                        for (y, mass) in truth.iter_mut().enumerate() {
                            *mass += bernoulli(u, latent_mass)
                                * bernoulli(m, 0.15 + mediator_shift * x as f64)
                                * bernoulli(y, 0.1 + 0.45 * m as f64 + 0.2 * u as f64);
                        }
                    }
                }
                let plan = ExactEvaluationPlan::compile(
                    functional.arena(),
                    functional.root(),
                    data.clone(),
                    [v(2)],
                    Assignment::from_pairs([(v(0), Value::Int64(x))]),
                    ExactEvaluationLimits::default(),
                    LawTolerance::default(),
                    &ctx,
                )
                .unwrap();
                let result = plan.evaluate(&ctx).unwrap();
                for (actual, expected) in result.probabilities.iter().zip(truth.iter()) {
                    assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
                }
                assert!((result.mean(v(2)).unwrap() - truth[1]).abs() < 1e-12);
            }
        }
    }
}

#[test]
fn negative_witness_mutations_and_evidence_substitution_fail() {
    use antecedent_identify::sid::verify_s_hedge;
    let mut graph = Admg::with_variables(2);
    graph.insert_directed(d(0), d(1)).unwrap();
    graph.insert_bidirected(d(0), d(1)).unwrap();
    let diagram = SelectionDiagram::try_new(graph, [v(1)]).unwrap();
    let query = ClassicalTransportQuery {
        outcomes: Arc::from([v(1)]),
        treatments: Arc::from([v(0)]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    };
    let ctx = ExecutionContext::for_tests(1);
    let ClassicalTransportResult::ProvenNonTransportable(witness) =
        identify_classical_transport(&diagram, &query, SidLimits::default(), &ctx).unwrap()
    else {
        panic!("selected confounded outcome is an s-hedge");
    };
    verify_s_hedge(&diagram, &query, &witness, &ctx).unwrap();
    let bytes = serde_json::to_vec(&witness.to_record()).unwrap();
    let record = serde_json::from_slice(&bytes).unwrap();
    antecedent_identify::sid::SHedgeCertificate::from_record_checked(
        record, &diagram, &query, &ctx,
    )
    .unwrap();
    let mut altered = witness.to_record();
    altered.evidence_setting = "finite_catalog".into();
    assert!(
        antecedent_identify::sid::SHedgeCertificate::from_record_checked(
            altered, &diagram, &query, &ctx
        )
        .is_err()
    );

    let mut mutated = witness.clone();
    mutated.larger.bidirected = Arc::from([]);
    assert!(verify_s_hedge(&diagram, &query, &mutated, &ctx).is_err());
    let mut mutated = witness.clone();
    mutated.larger.directed = Arc::from([]);
    assert!(verify_s_hedge(&diagram, &query, &mutated, &ctx).is_err());
    let mut different_evidence = query;
    different_evidence.source = Arc::from("another-source");
    assert!(verify_s_hedge(&diagram, &different_evidence, &witness, &ctx).is_err());
}

#[test]
#[allow(clippy::too_many_lines)] // Independent exhaustive oracle and provider construction.
fn three_node_selection_graphs_agree_with_full_experimental_oracle() {
    // All 64 ordered three-node ADMGs and all eight selection patterns.
    // Independent Bernoulli latent causes generate every bidirected edge.
    let ctx = ExecutionContext::for_tests(42);
    let edges = [(0usize, 1usize), (0, 2), (1, 2)];
    let query = ClassicalTransportQuery {
        outcomes: Arc::from([v(2)]),
        treatments: Arc::from([v(0)]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    };
    for directed_mask in 0..8usize {
        for bidirected_mask in 0..8usize {
            for selected in 0..8usize {
                let mut graph = Admg::with_variables(3);
                for (bit, &(a, b)) in edges.iter().enumerate() {
                    if directed_mask & (1 << bit) != 0 {
                        graph.insert_directed(d(a as u32), d(b as u32)).unwrap();
                    }
                    if bidirected_mask & (1 << bit) != 0 {
                        graph.insert_bidirected(d(a as u32), d(b as u32)).unwrap();
                    }
                }
                let diagram = SelectionDiagram::try_new(
                    graph,
                    (0..3).filter(|i| selected & (1 << i) != 0).map(v).collect::<Vec<_>>(),
                )
                .unwrap();
                let result =
                    identify_classical_transport(&diagram, &query, SidLimits::default(), &ctx)
                        .unwrap();
                let proof = match result {
                    ClassicalTransportResult::Identified(proof) => proof,
                    ClassicalTransportResult::ProvenNonTransportable(witness) => {
                        antecedent_identify::sid::verify_s_hedge(&diagram, &query, &witness, &ctx)
                            .unwrap();
                        continue;
                    }
                    ClassicalTransportResult::NotCertified => panic!(
                        "inconclusive uncapped three-node case: directed={directed_mask} bidirected={bidirected_mask} selected={selected}"
                    ),
                };
                for (base, parent_w, shared_w, sel_w) in
                    [(0.12_f64, 0.16, 0.12, 0.09), (0.08, 0.14, 0.11, 0.07)]
                {
                    let enumerate =
                        |target: bool, intervention_mask: usize, intervention_values: usize| {
                            let axes: Vec<usize> =
                                (0..3).filter(|i| intervention_mask & (1 << i) == 0).collect();
                            let mut law = vec![0.0; 1 << axes.len()];
                            for latent in 0..8usize {
                                for values in 0..8usize {
                                    if (values & intervention_mask)
                                        != (intervention_values & intervention_mask)
                                    {
                                        continue;
                                    }
                                    let bit = |node: usize| (values >> node) & 1;
                                    let mut mass = 0.125;
                                    for node in 0..3 {
                                        if intervention_mask & (1 << node) != 0 {
                                            continue;
                                        }
                                        let parents: usize = edges
                                            .iter()
                                            .enumerate()
                                            .filter(|(e, (_, b))| {
                                                *b == node && directed_mask & (1 << e) != 0
                                            })
                                            .map(|(_, (a, _))| bit(*a))
                                            .sum();
                                        let shared: usize = edges
                                            .iter()
                                            .enumerate()
                                            .filter(|(e, (a, b))| {
                                                (*a == node || *b == node)
                                                    && bidirected_mask & (1 << e) != 0
                                            })
                                            .map(|(e, _)| (latent >> e) & 1)
                                            .sum();
                                        let p = base
                                            + parent_w * parents as f64
                                            + shared_w * shared as f64
                                            + if target && selected & (1 << node) != 0 {
                                                sel_w
                                            } else {
                                                0.0
                                            };
                                        mass *= bernoulli(bit(node), p);
                                    }
                                    let row = axes.iter().fold(0, |row, i| row * 2 + bit(*i));
                                    law[row] += mass;
                                }
                            }
                            (axes, law)
                        };
                    let mut laws = Vec::new();
                    let mut regimes = Vec::new();
                    for mask in 0..8usize {
                        let interventions: Vec<_> =
                            (0..3).filter(|i| mask & (1 << i) != 0).map(v).collect();
                        let measured: Vec<_> =
                            (0..3).filter(|i| mask & (1 << i) == 0).map(v).collect();
                        regimes.push(
                            EvidenceRegime::try_new(
                                RegimeId::from_raw(mask as u32),
                                if mask == 0 {
                                    RegimeKind::Observational
                                } else {
                                    RegimeKind::Experimental
                                },
                                EvidenceKind::Available,
                                interventions.clone(),
                                [],
                                measured,
                                "source",
                                DistributionAvailability::Joint,
                            )
                            .unwrap(),
                        );
                        for values in 0..8usize {
                            if values & !mask != 0 {
                                continue;
                            }
                            let (axes, mass) = enumerate(false, mask, values);
                            let assignments = interventions
                                .iter()
                                .map(|v| antecedent_expr::InterventionAssignment {
                                    variable: *v,
                                    value: Value::Int64(((values >> v.raw()) & 1) as i64),
                                })
                                .collect::<Vec<_>>();
                            let axes = axes
                                .into_iter()
                                .map(|i| DiscreteAxis {
                                    variable: v(i as u32),
                                    values: Arc::from([Value::Int64(0), Value::Int64(1)]),
                                })
                                .collect::<Vec<_>>();
                            laws.push(
                                ExactDiscreteLaw::try_new(
                                    "source",
                                    RegimeId::from_raw(mask as u32),
                                    assignments,
                                    axes,
                                    mass,
                                    "oracle",
                                    LawTolerance::default(),
                                )
                                .unwrap(),
                            );
                        }
                    }
                    let (axes, mass) = enumerate(true, 0, 0);
                    regimes.push(
                        EvidenceRegime::try_new(
                            RegimeId::from_raw(8),
                            RegimeKind::Observational,
                            EvidenceKind::Available,
                            [],
                            [],
                            [v(0), v(1), v(2)],
                            "target",
                            DistributionAvailability::Joint,
                        )
                        .unwrap(),
                    );
                    let axes = axes
                        .into_iter()
                        .map(|i| DiscreteAxis {
                            variable: v(i as u32),
                            values: Arc::from([Value::Int64(0), Value::Int64(1)]),
                        })
                        .collect::<Vec<_>>();
                    laws.push(
                        ExactDiscreteLaw::try_new(
                            "target",
                            RegimeId::from_raw(8),
                            [],
                            axes,
                            mass,
                            "oracle",
                            LawTolerance::default(),
                        )
                        .unwrap(),
                    );
                    let catalog = EvidenceCatalog::try_new([], regimes, [], None).unwrap();
                    let bound = proof.bind_catalog(&catalog).unwrap();
                    let data = ExactTransportData::try_new(laws, 1000).unwrap();
                    for level in 0..2 {
                        let (_, truth) = enumerate(true, 1, level);
                        let expected = truth[1] + truth[3];
                        let plan = ExactEvaluationPlan::compile(
                            bound.arena(),
                            bound.root(),
                            data.clone(),
                            [v(2)],
                            Assignment::from_pairs([(v(0), Value::Int64(level as i64))]),
                            ExactEvaluationLimits::default(),
                            LawTolerance::default(),
                            &ctx,
                        )
                        .unwrap();
                        let result = plan.evaluate(&ctx).unwrap();
                        assert!(
                            (result.mean(v(2)).unwrap() - expected).abs() < 1e-10,
                            "graph directed={directed_mask} bidirected={bidirected_mask} selection={selected} level={level} base={base}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn original_coordinates_and_intervention_enlargement_are_preserved() {
    let mut graph = Admg::empty();
    for id in [17, 41, 99] {
        graph.add_node(antecedent_core::NodeRef::Static(v(id))).unwrap();
    }
    graph.insert_directed(d(0), d(1)).unwrap();
    graph.insert_directed(d(1), d(2)).unwrap();
    let diagram = SelectionDiagram::try_new(graph, [v(99)]).unwrap();
    let query = ClassicalTransportQuery {
        outcomes: Arc::from([v(99)]),
        treatments: Arc::from([v(41)]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    };
    let ClassicalTransportResult::Identified(proof) = identify_classical_transport(
        &diagram,
        &query,
        SidLimits::default(),
        &ExecutionContext::for_tests(0),
    )
    .unwrap() else {
        panic!("target DAG identifies");
    };
    assert!(proof.rules().contains(&"sid.line3"));
    let mut arena = proof.arena().clone();
    assert_eq!(arena.free_variables(proof.root()), vec![v(41), v(99)]);
}

#[test]
fn unavailable_target_formula_does_not_hide_available_source_alternative() {
    use antecedent_identify::{CatalogTransportResult, identify_catalog_transport};
    let mut graph = Admg::with_variables(2);
    graph.insert_directed(d(0), d(1)).unwrap();
    let diagram = SelectionDiagram::try_new(graph, []).unwrap();
    let query = ClassicalTransportQuery {
        outcomes: Arc::from([v(1)]),
        treatments: Arc::from([v(0)]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    };
    let regime = EvidenceRegime::try_new(
        RegimeId::from_raw(9),
        RegimeKind::Experimental,
        EvidenceKind::Available,
        [v(0)],
        [],
        [v(1)],
        "source",
        DistributionAvailability::Joint,
    )
    .unwrap();
    let catalog = EvidenceCatalog::try_new([], [regime], [], None).unwrap();
    let ctx = ExecutionContext::for_tests(0);
    let CatalogTransportResult::Identified(bound) =
        identify_catalog_transport(&diagram, &query, &catalog, SidLimits::default(), &ctx).unwrap()
    else {
        panic!("source alternative must bind");
    };
    assert!(
        bound
            .arena()
            .leaf_bindings(bound.root())
            .iter()
            .all(|binding| binding.population.as_ref() == "source")
    );
    let empty = identify_catalog_transport(
        &diagram,
        &query,
        &EvidenceCatalog::empty(),
        SidLimits::default(),
        &ctx,
    )
    .unwrap();
    let CatalogTransportResult::MissingEvidence { searched, .. } = empty else {
        panic!("missing catalog is not an impossibility proof");
    };
    assert_eq!(searched.len(), 3);
}

#[test]
fn four_node_branch_conformance_has_no_unchecked_obstructions() {
    let edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    let query = ClassicalTransportQuery {
        outcomes: Arc::from([v(3)]),
        treatments: Arc::from([v(0)]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    };
    let ctx = ExecutionContext::for_tests(0);
    let mut rules = std::collections::BTreeSet::new();
    for directed in 0..64 {
        for bidirected in 0..64 {
            let mut graph = Admg::with_variables(4);
            for (index, (a, b)) in edges.iter().enumerate() {
                if directed & (1 << index) != 0 {
                    graph.insert_directed(d(*a), d(*b)).unwrap();
                }
                if bidirected & (1 << index) != 0 {
                    graph.insert_bidirected(d(*a), d(*b)).unwrap();
                }
            }
            for selections in 0..16 {
                let diagram = SelectionDiagram::try_new(
                    graph.clone(),
                    (0..4).filter(|i| selections & (1 << i) != 0).map(v).collect::<Vec<_>>(),
                )
                .unwrap();
                match identify_classical_transport(&diagram, &query, SidLimits::default(), &ctx)
                    .unwrap()
                {
                    ClassicalTransportResult::Identified(proof) => rules.extend(proof.rules()),
                    ClassicalTransportResult::ProvenNonTransportable(witness) => {
                        antecedent_identify::sid::verify_s_hedge(&diagram, &query, &witness, &ctx)
                            .unwrap()
                    }
                    ClassicalTransportResult::NotCertified => panic!(
                        "unchecked obstruction directed={directed} bidirected={bidirected} selections={selections}"
                    ),
                }
            }
        }
    }
    // Line 3 requires an ancestor preceding the treatment in topological order.
    let mut graph = Admg::with_variables(4);
    for (a, b) in [(0, 1), (1, 3), (2, 3)] {
        graph.insert_directed(d(a), d(b)).unwrap();
    }
    let diagram = SelectionDiagram::try_new(graph, []).unwrap();
    let mut query = query.clone();
    query.treatments = Arc::from([v(1)]);
    let ClassicalTransportResult::Identified(proof) =
        identify_classical_transport(&diagram, &query, SidLimits::default(), &ctx).unwrap()
    else {
        panic!("pre-treatment ancestor");
    };
    rules.extend(proof.rules());
    for rule in
        ["sid.line1", "sid.line2", "sid.line3", "sid.line4", "sid.line7", "sid.line8", "sid.line10"]
    {
        assert!(rules.contains(rule), "missing branch {rule}");
    }
}

#[test]
#[allow(clippy::too_many_lines)]
fn four_node_frontdoor_matches_independent_parameterizations() {
    let mut graph = Admg::with_variables(4);
    graph.insert_directed(d(0), d(1)).unwrap();
    graph.insert_directed(d(1), d(3)).unwrap();
    graph.insert_directed(d(2), d(3)).unwrap();
    graph.insert_bidirected(d(0), d(3)).unwrap();
    let diagram = SelectionDiagram::try_new(graph, [v(3)]).unwrap();
    let query = ClassicalTransportQuery {
        outcomes: Arc::from([v(3)]),
        treatments: Arc::from([v(0)]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    };
    let ctx = ExecutionContext::for_tests(4);
    let ClassicalTransportResult::Identified(proof) =
        identify_classical_transport(&diagram, &query, SidLimits::default(), &ctx).unwrap()
    else {
        panic!("four-node frontdoor with covariate is identifiable");
    };
    let catalog = EvidenceCatalog::try_new(
        [],
        [EvidenceRegime::try_new(
            RegimeId::from_raw(0),
            RegimeKind::Observational,
            EvidenceKind::Available,
            [],
            [],
            (0..4).map(v).collect::<Vec<_>>(),
            "target",
            DistributionAvailability::Joint,
        )
        .unwrap()],
        [],
        None,
    )
    .unwrap();
    let functional = proof.bind_catalog(&catalog).unwrap();
    for (latent_mass, mediator_shift, z_shift) in
        [(0.2_f64, 0.25, 0.15), (0.45, 0.55, 0.05), (0.8, 0.35, 0.25)]
    {
        let mut observational = vec![0.0; 16];
        for u in 0..2 {
            for x in 0..2 {
                for m in 0..2 {
                    for z in 0..2 {
                        for y in 0..2 {
                            let mass = bernoulli(u, latent_mass)
                                * bernoulli(x, if u == 1 { 0.75 } else { 0.25 })
                                * bernoulli(m, 0.2 + mediator_shift * x as f64)
                                * bernoulli(z, 0.4)
                                * bernoulli(
                                    y,
                                    0.1 + 0.35 * m as f64 + z_shift * z as f64 + 0.2 * u as f64,
                                );
                            observational[x * 8 + m * 4 + z * 2 + y] += mass;
                        }
                    }
                }
            }
        }
        let law = ExactDiscreteLaw::try_new(
            "target",
            RegimeId::from_raw(0),
            [],
            (0..4)
                .map(|i| DiscreteAxis {
                    variable: v(i),
                    values: Arc::from([Value::Int64(0), Value::Int64(1)]),
                })
                .collect::<Vec<_>>(),
            observational,
            "four-node-oracle",
            LawTolerance::default(),
        )
        .unwrap();
        let data = ExactTransportData::try_new([law], 10_000).unwrap();
        for x in 0..2 {
            let mut truth = 0.0;
            for u in 0..2 {
                for m in 0..2 {
                    for z in 0..2 {
                        truth += bernoulli(u, latent_mass)
                            * bernoulli(m, 0.2 + mediator_shift * x as f64)
                            * bernoulli(z, 0.4)
                            * (0.1 + 0.35 * m as f64 + z_shift * z as f64 + 0.2 * u as f64);
                    }
                }
            }
            let result = ExactEvaluationPlan::compile(
                functional.arena(),
                functional.root(),
                data.clone(),
                [v(3)],
                Assignment::from_pairs([(v(0), Value::Int64(x))]),
                ExactEvaluationLimits::default(),
                LawTolerance::default(),
                &ctx,
            )
            .unwrap()
            .evaluate(&ctx)
            .unwrap();
            assert!(
                (result.mean(v(3)).unwrap() - truth).abs() < 1e-12,
                "x={x} latent={latent_mass} mediator={mediator_shift} z={z_shift}"
            );
        }
    }
}

#[test]
#[allow(clippy::too_many_lines)] // Keep independently enumerated joint truth and its contrasts together.
fn six_node_recursive_districts_match_joint_latent_scm_truth() {
    let mut graph = Admg::with_variables(6);
    for offset in [0, 3] {
        graph.insert_directed(d(offset), d(offset + 1)).unwrap();
        graph.insert_directed(d(offset + 1), d(offset + 2)).unwrap();
        graph.insert_bidirected(d(offset), d(offset + 2)).unwrap();
    }
    let diagram = SelectionDiagram::try_new(graph, [v(2), v(5)]).unwrap();
    let query = ClassicalTransportQuery {
        outcomes: Arc::from([v(2), v(5)]),
        treatments: Arc::from([v(0), v(3)]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    };
    let ctx = ExecutionContext::for_tests(0);
    let ClassicalTransportResult::Identified(proof) =
        identify_classical_transport(&diagram, &query, SidLimits::default(), &ctx).unwrap()
    else {
        panic!("two frontdoor districts");
    };
    assert!(proof.rules().iter().filter(|r| **r == "sid.line8").count() >= 2);
    let catalog = EvidenceCatalog::try_new(
        [],
        [EvidenceRegime::try_new(
            RegimeId::from_raw(0),
            RegimeKind::Observational,
            EvidenceKind::Available,
            [],
            [],
            (0..6).map(v).collect::<Vec<_>>(),
            "target",
            DistributionAvailability::Joint,
        )
        .unwrap()],
        [],
        None,
    )
    .unwrap();
    let functional = proof.bind_catalog(&catalog).unwrap();
    for latent_mass in [0.2, 0.45, 0.8] {
        let block = |x: usize, m: usize, y: usize, intervened: bool| {
            (0..2)
                .map(|u| {
                    bernoulli(u, latent_mass)
                        * if intervened { 1.0 } else { bernoulli(x, 0.15 + 0.7 * u as f64) }
                        * bernoulli(m, 0.2 + 0.5 * x as f64)
                        * bernoulli(y, 0.1 + 0.4 * m as f64 + 0.3 * u as f64)
                })
                .sum::<f64>()
        };
        let mut observational = Vec::new();
        for bits in 0..64usize {
            let b = |i: usize| (bits >> (5usize - i)) & 1usize;
            observational.push(block(b(0), b(1), b(2), false) * block(b(3), b(4), b(5), false));
        }
        let law = ExactDiscreteLaw::try_new(
            "target",
            RegimeId::from_raw(0),
            [],
            (0..6)
                .map(|i| DiscreteAxis {
                    variable: v(i),
                    values: Arc::from([Value::Int64(0), Value::Int64(1)]),
                })
                .collect::<Vec<_>>(),
            observational,
            "six-node-oracle",
            LawTolerance::default(),
        )
        .unwrap();
        let data = ExactTransportData::try_new([law], 100_000).unwrap();
        let mut reference = None;
        for x0 in 0..2 {
            for x1 in 0..2 {
                let result = ExactEvaluationPlan::compile(
                    functional.arena(),
                    functional.root(),
                    data.clone(),
                    [v(2), v(5)],
                    Assignment::from_pairs([
                        (v(0), Value::Int64(x0 as i64)),
                        (v(3), Value::Int64(x1 as i64)),
                    ]),
                    ExactEvaluationLimits { operations: 100_000_000, depth: 256 },
                    LawTolerance::default(),
                    &ctx,
                )
                .unwrap()
                .evaluate(&ctx)
                .unwrap();
                for (outcome, x) in [(v(2), x0), (v(5), x1)] {
                    let truth = 0.18 + 0.2 * x as f64 + 0.3 * latent_mass;
                    assert!((result.mean(outcome).unwrap() - truth).abs() < 1e-12);
                }
                if x0 == 0 && x1 == 0 {
                    reference = Some(result.clone());
                }
                for (outcome, x) in [(v(2), x0), (v(5), x1)] {
                    assert!(
                        (result.mean_difference(reference.as_ref().unwrap(), outcome).unwrap()
                            - 0.2 * x as f64)
                            .abs()
                            < 1e-12
                    );
                }
                for (atom, p) in result.atoms.iter().zip(result.probabilities.iter()) {
                    let y0 = usize::from(atom[0] == Value::Int64(1));
                    let y1 = usize::from(atom[1] == Value::Int64(1));
                    let truth = (0..2).map(|m| block(x0, m, y0, true)).sum::<f64>()
                        * (0..2).map(|m| block(x1, m, y1, true)).sum::<f64>();
                    assert!((p - truth).abs() < 1e-12, "{p} != {truth}");
                }
            }
        }
    }
}
