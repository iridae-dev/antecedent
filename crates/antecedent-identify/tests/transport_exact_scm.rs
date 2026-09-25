//! Numerical oracle independent of symbolic ID: enumerate latent binary SCMs.
#![allow(clippy::cast_precision_loss, clippy::cast_possible_wrap)]
#![allow(
    clippy::cast_possible_truncation,
    reason = "test scaffolding compares exact constants and indexes with small literals"
)] // All casts are binary levels or three-node coordinates.
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
    identify_classical_transport, ClassicalTransportQuery, ClassicalTransportResult, SidLimits,
};
use std::sync::Arc;
fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}
fn d(i: u32) -> DenseNodeId {
    DenseNodeId::from_raw(i)
}
fn bernoulli(value: usize, probability: f64) -> f64 {
    if value == 1 {
        probability
    } else {
        1.0 - probability
    }
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
    assert!(antecedent_identify::sid::SHedgeCertificate::from_record_checked(
        altered, &diagram, &query, &ctx
    )
    .is_err());

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
fn three_node_selection_graphs_agree_with_full_experimental_oracle() {
    three_node_selection_sweep(0);
}

#[test]
fn three_node_selection_graphs_with_pre_treatment_ancestor_agree_with_full_experimental_oracle() {
    // X = v(1) has the potential parent W = v(0), so sID line 3 (enlargement over
    // non-ancestors of Y) and the weighted marginalization over W are exercised
    // numerically, not only by rule name.
    three_node_selection_sweep(1);
}

#[allow(clippy::too_many_lines)] // Independent exhaustive oracle and provider construction.
fn three_node_selection_sweep(x: u32) {
    // Domain: all 64 ordered three-node ADMGs × 8 selection patterns × 3
    // independent Bernoulli latent-SCM parameterizations. Truth is the target
    // interventional P(Y=1 | do(X=x)), not a second wrapper of the same formula.
    sweep(&[x], 2, &[(0.12, 0.16, 0.12, 0.09), (0.08, 0.14, 0.11, 0.07), (0.18, 0.10, 0.08, 0.05)]);
}

#[test]
fn every_treatment_position_and_joint_interventions_agree_with_the_oracle() {
    // Edges always point from lower to higher index, so a treatment other than
    // node 0 has parents and pretreatment confounders, and enlargement (line 3),
    // recursion through carried kernels (line 8) and external-parent source
    // leaves are all exercised numerically, as are two-treatment queries.
    let parameterization = [(0.12, 0.16, 0.12, 0.09)];
    for (treatments, outcome) in [
        (&[0u32][..], 1u32),
        (&[1][..], 2),
        (&[1][..], 0),
        (&[2][..], 0),
        (&[2][..], 1),
        (&[0, 1][..], 2),
        (&[0, 2][..], 1),
        (&[1, 2][..], 0),
    ] {
        sweep(treatments, outcome, &parameterization);
    }
}

#[allow(clippy::too_many_lines)] // Independent exhaustive oracle and provider construction.
fn sweep(treatments: &[u32], outcome: u32, parameterizations: &[(f64, f64, f64, f64)]) {
    let ctx = ExecutionContext::for_tests(42);
    let edges = [(0usize, 1usize), (0, 2), (1, 2)];
    let query = ClassicalTransportQuery {
        outcomes: Arc::from([v(outcome)]),
        treatments: treatments.iter().map(|t| v(*t)).collect::<Vec<_>>().into(),
        source: Arc::from("source"),
        target: Arc::from("target"),
    };
    let treatment_mask: usize = treatments.iter().map(|t| 1usize << t).sum();
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
                for &(base, parent_w, shared_w, sel_w) in parameterizations {
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
                    for level in 0..(1usize << treatments.len()) {
                        let assigned: usize = treatments
                            .iter()
                            .enumerate()
                            .filter(|(j, _)| (level >> j) & 1 == 1)
                            .map(|(_, t)| 1usize << t)
                            .sum();
                        let (truth_axes, truth) = enumerate(true, treatment_mask, assigned);
                        let outcome_axis =
                            truth_axes.iter().position(|axis| *axis == outcome as usize).unwrap();
                        let expected: f64 = truth
                            .iter()
                            .enumerate()
                            .filter(|(row, _)| {
                                (row >> (truth_axes.len() - 1 - outcome_axis)) & 1 == 1
                            })
                            .map(|(_, mass)| *mass)
                            .sum();
                        let plan =
                            ExactEvaluationPlan::compile(
                                bound.arena(),
                                bound.root(),
                                data.clone(),
                                [v(outcome)],
                                Assignment::from_pairs(treatments.iter().enumerate().map(
                                    |(j, t)| (v(*t), Value::Int64(((level >> j) & 1) as i64)),
                                )),
                                ExactEvaluationLimits::default(),
                                LawTolerance::default(),
                                &ctx,
                            )
                            .unwrap();
                        let result = plan.evaluate(&ctx).unwrap();
                        assert!(
                            (result.mean(v(outcome)).unwrap() - expected).abs() < 1e-10,
                            "graph directed={directed_mask} bidirected={bidirected_mask} selection={selected} treatments={treatments:?} outcome={outcome} level={level} base={base}"
                        );
                    }
                }
            }
        }
    }
}

/// Laws for X -> M -> Y with X <-> M, where only Y's mechanism differs in the target.
/// Returns the source `do(x)` worlds over (M, Y) and one observational joint over (X, M, Y).
fn mediator_bow_laws(observational_population: &str, y_shift: f64) -> Vec<ExactDiscreteLaw> {
    let binary =
        |i| DiscreteAxis { variable: v(i), values: Arc::from([Value::Int64(0), Value::Int64(1)]) };
    let m_given = |m, x: usize, u: usize| bernoulli(m, 0.15 + 0.5 * x as f64 + 0.2 * u as f64);
    let y_given = |y, m: usize, shift: f64| bernoulli(y, 0.1 + 0.6 * m as f64 + shift);
    let mut laws = Vec::new();
    for x in 0..2usize {
        let mut world = vec![0.0; 4];
        for u in 0..2 {
            for m in 0..2 {
                for y in 0..2 {
                    world[m * 2 + y] += 0.5 * m_given(m, x, u) * y_given(y, m, 0.0);
                }
            }
        }
        laws.push(
            ExactDiscreteLaw::try_new(
                "source",
                RegimeId::from_raw(1),
                [antecedent_expr::InterventionAssignment {
                    variable: v(0),
                    value: Value::Int64(x as i64),
                }],
                [binary(1), binary(2)],
                world,
                "oracle",
                LawTolerance::default(),
            )
            .unwrap(),
        );
    }
    let mut joint = vec![0.0; 8];
    for u in 0..2 {
        for x in 0..2 {
            for m in 0..2 {
                for y in 0..2 {
                    joint[x * 4 + m * 2 + y] += 0.5
                        * bernoulli(x, 0.2 + 0.6 * u as f64)
                        * m_given(m, x, u)
                        * y_given(y, m, y_shift);
                }
            }
        }
    }
    laws.push(
        ExactDiscreteLaw::try_new(
            observational_population,
            RegimeId::from_raw(2),
            [],
            [binary(0), binary(1), binary(2)],
            joint,
            "oracle",
            LawTolerance::default(),
        )
        .unwrap(),
    );
    laws
}

#[test]
fn two_population_district_recursion_fails_truth_under_kernel_population_substitution() {
    use antecedent_identify::{identify_catalog_transport, CatalogTransportResult};
    const TARGET_Y_SHIFT: f64 = 0.25;
    let mut graph = Admg::with_variables(3);
    graph.insert_directed(d(0), d(1)).unwrap();
    graph.insert_directed(d(1), d(2)).unwrap();
    graph.insert_bidirected(d(0), d(1)).unwrap();
    let diagram = SelectionDiagram::try_new(graph, [v(2)]).unwrap();
    let query = ClassicalTransportQuery {
        outcomes: Arc::from([v(2)]),
        treatments: Arc::from([v(0)]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    };
    let regimes = [
        EvidenceRegime::try_new(
            RegimeId::from_raw(1),
            RegimeKind::Experimental,
            EvidenceKind::Available,
            [v(0)],
            [],
            [v(1), v(2)],
            "source",
            DistributionAvailability::Joint,
        )
        .unwrap(),
        EvidenceRegime::try_new(
            RegimeId::from_raw(2),
            RegimeKind::Observational,
            EvidenceKind::Available,
            [],
            [],
            [v(0), v(1), v(2)],
            "target",
            DistributionAvailability::Joint,
        )
        .unwrap(),
    ];
    let catalog = EvidenceCatalog::try_new([], regimes, [], None).unwrap();
    let ctx = ExecutionContext::for_tests(3);
    let CatalogTransportResult::Identified(bound) =
        identify_catalog_transport(&diagram, &query, &catalog, SidLimits::default(), &ctx).unwrap()
    else {
        panic!("the bow on M needs the source experiment and the selected Y needs the target");
    };
    let populations: std::collections::BTreeSet<_> = bound
        .arena()
        .leaf_bindings(bound.root())
        .iter()
        .map(|binding| binding.population.to_string())
        .collect();
    assert_eq!(populations.into_iter().collect::<Vec<_>>(), ["source", "target"]);

    let mean = |laws: Vec<ExactDiscreteLaw>, x: i64| {
        ExactEvaluationPlan::compile(
            bound.arena(),
            bound.root(),
            ExactTransportData::try_new(laws, 1000).unwrap(),
            [v(2)],
            Assignment::from_pairs([(v(0), Value::Int64(x))]),
            ExactEvaluationLimits::default(),
            LawTolerance::default(),
            &ctx,
        )
        .and_then(|plan| plan.evaluate(&ctx))
        .map(|result| result.mean(v(2)).unwrap())
    };
    for x in 0..2usize {
        // Target truth: M keeps the shared mechanism, Y uses the target mechanism.
        let p_m1 = 0.15 + 0.5 * x as f64 + 0.2 * 0.5;
        let truth = 0.1 + 0.6 * p_m1 + TARGET_Y_SHIFT;
        let actual = mean(mediator_bow_laws("target", TARGET_Y_SHIFT), x as i64).unwrap();
        assert!((actual - truth).abs() < 1e-12, "{actual} != {truth}");
        // The source's Y kernel filed under the target's identity is a different law.
        let substituted = mean(mediator_bow_laws("target", 0.0), x as i64).unwrap();
        assert!((substituted - truth).abs() > 0.2, "population substitution went unnoticed");
        // An honestly labelled source observation cannot stand in for the target leaf.
        assert!(mean(mediator_bow_laws("source", 0.0), x as i64).is_err());
    }
}

#[test]
fn exhausted_identification_budget_is_an_error_never_a_negative_witness() {
    let mut graph = Admg::with_variables(2);
    graph.insert_directed(d(0), d(1)).unwrap();
    graph.insert_bidirected(d(0), d(1)).unwrap();
    let query = ClassicalTransportQuery {
        outcomes: Arc::from([v(1)]),
        treatments: Arc::from([v(0)]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    };
    let ctx = ExecutionContext::for_tests(5);
    let starved = SidLimits { steps: 1, depth: 1 };
    // The same budget starves an obstructed and an identifiable diagram alike, so the
    // refusal carries no information about transportability.
    for selected in [vec![v(1)], vec![]] {
        let diagram = SelectionDiagram::try_new(graph.clone(), selected.clone()).unwrap();
        let error = identify_classical_transport(&diagram, &query, starved, &ctx)
            .expect_err("a starved search has no result");
        assert!(error.to_string().contains("transport.identification_budget"), "{error}");
        let full =
            identify_classical_transport(&diagram, &query, SidLimits::default(), &ctx).unwrap();
        if selected.is_empty() {
            assert!(matches!(full, ClassicalTransportResult::Identified(_)));
        } else {
            assert!(matches!(full, ClassicalTransportResult::ProvenNonTransportable(_)));
        }
    }
}

#[test]
fn identification_within_an_explicit_budget_matches_the_default_search() {
    // X -> Y with X <-> Y; an unselected diagram identifies, a Y-selected one is
    // an s-hedge. A generous but non-default budget must not change either verdict.
    let mut graph = Admg::with_variables(2);
    graph.insert_directed(d(0), d(1)).unwrap();
    graph.insert_bidirected(d(0), d(1)).unwrap();
    let query = ClassicalTransportQuery {
        outcomes: Arc::from([v(1)]),
        treatments: Arc::from([v(0)]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    };
    let ctx = ExecutionContext::for_tests(5);
    let explicit = SidLimits { steps: 1_000, depth: 32 };
    for selected in [vec![v(1)], vec![]] {
        let diagram = SelectionDiagram::try_new(graph.clone(), selected.clone()).unwrap();
        let bounded = identify_classical_transport(&diagram, &query, explicit, &ctx).unwrap();
        let default =
            identify_classical_transport(&diagram, &query, SidLimits::default(), &ctx).unwrap();
        match (bounded, default) {
            (ClassicalTransportResult::Identified(a), ClassicalTransportResult::Identified(b)) => {
                assert!(selected.is_empty());
                assert_eq!(a.rules(), b.rules());
            }
            (
                ClassicalTransportResult::ProvenNonTransportable(_),
                ClassicalTransportResult::ProvenNonTransportable(_),
            ) => assert!(!selected.is_empty()),
            _ => panic!("an explicit budget that is not exhausted changed the verdict"),
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
    let arena = proof.arena().clone();
    assert_eq!(arena.free_variables(proof.root()), vec![v(41), v(99)]);
}

#[test]
fn unavailable_target_formula_does_not_hide_available_source_alternative() {
    use antecedent_identify::{identify_catalog_transport, CatalogTransportResult};
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
    assert!(bound
        .arena()
        .leaf_bindings(bound.root())
        .iter()
        .all(|binding| binding.population.as_ref() == "source"));
    // Empty standardization is direct transport, not Figure 5 line 10.
    assert_eq!(bound.derivation().rules(), ["transport.direct"]);
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

/// Same ordered four-node family as [`four_node_branch_conformance_has_no_unchecked_obstructions`],
/// with every identified formula compared to an independent latent-SCM oracle.
/// A parameterization is skipped when a node's Bernoulli weight would leave `(0, 1)`.
#[test]
fn four_node_branch_positives_match_latent_scm_truth() {
    const PARAMS: [(f64, f64, f64, f64); 3] =
        [(0.12, 0.16, 0.12, 0.09), (0.08, 0.14, 0.11, 0.07), (0.18, 0.10, 0.08, 0.05)];
    let failed = std::sync::atomic::AtomicBool::new(false);
    let message = std::sync::Mutex::new(String::new());
    let checked = std::sync::atomic::AtomicU64::new(0);
    std::thread::scope(|scope| {
        for worker in 0..8u32 {
            let failed = &failed;
            let message = &message;
            let checked = &checked;
            scope.spawn(move || {
                let ctx = ExecutionContext::for_tests(u64::from(worker));
                let catalog = four_node_power_set_catalog();
                let query = ClassicalTransportQuery {
                    outcomes: Arc::from([v(3)]),
                    treatments: Arc::from([v(0)]),
                    source: Arc::from("source"),
                    target: Arc::from("target"),
                };
                for directed in (worker..64).step_by(8) {
                    for bidirected in 0..64u32 {
                        if failed.load(std::sync::atomic::Ordering::Relaxed) {
                            return;
                        }
                        let graph = four_node_graph(directed, bidirected);
                        for selections in 0..16u32 {
                            let diagram = SelectionDiagram::try_new(
                                graph.clone(),
                                (0..4).filter(|i| selections & (1 << i) != 0).map(v).collect::<Vec<_>>(),
                            )
                            .unwrap();
                            let proof = match identify_classical_transport(
                                &diagram,
                                &query,
                                SidLimits::default(),
                                &ctx,
                            )
                            .unwrap()
                            {
                                ClassicalTransportResult::Identified(proof) => proof,
                                ClassicalTransportResult::ProvenNonTransportable(_) => continue,
                                ClassicalTransportResult::NotCertified => {
                                    *message.lock().unwrap() = format!(
                                        "unchecked obstruction directed={directed} bidirected={bidirected} selections={selections}"
                                    );
                                    failed.store(true, std::sync::atomic::Ordering::Relaxed);
                                    return;
                                }
                            };
                            for (base, parent_w, shared_w, sel_w) in PARAMS {
                                if !four_node_param_in_unit_interval(
                                    directed, bidirected, selections, base, parent_w, shared_w, sel_w,
                                ) {
                                    continue;
                                }
                                if let Err(err) = four_node_formula_matches_scm(
                                    &proof,
                                    &catalog,
                                    directed,
                                    bidirected,
                                    selections,
                                    base,
                                    parent_w,
                                    shared_w,
                                    sel_w,
                                    &ctx,
                                ) {
                                    *message.lock().unwrap() = err;
                                    failed.store(true, std::sync::atomic::Ordering::Relaxed);
                                    return;
                                }
                                checked.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            }
                        }
                    }
                }
            });
        }
    });
    assert!(!failed.load(std::sync::atomic::Ordering::Relaxed), "{}", message.lock().unwrap());
    assert!(checked.load(std::sync::atomic::Ordering::Relaxed) > 0);
}

const FOUR_EDGES: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

fn four_node_graph(directed: u32, bidirected: u32) -> Admg {
    let mut graph = Admg::with_variables(4);
    for (index, (a, b)) in FOUR_EDGES.iter().enumerate() {
        if directed & (1 << index) != 0 {
            graph.insert_directed(d(*a as u32), d(*b as u32)).unwrap();
        }
        if bidirected & (1 << index) != 0 {
            graph.insert_bidirected(d(*a as u32), d(*b as u32)).unwrap();
        }
    }
    graph
}

fn four_node_param_in_unit_interval(
    directed: u32,
    bidirected: u32,
    selections: u32,
    base: f64,
    parent_w: f64,
    shared_w: f64,
    sel_w: f64,
) -> bool {
    (0..4).all(|node| {
        let parents = FOUR_EDGES
            .iter()
            .enumerate()
            .filter(|(e, (_, b))| *b == node && directed & (1 << e) != 0)
            .count() as f64;
        let shared = FOUR_EDGES
            .iter()
            .enumerate()
            .filter(|(e, (a, b))| (*a == node || *b == node) && bidirected & (1 << e) != 0)
            .count() as f64;
        let sel = if selections & (1 << node) != 0 { sel_w } else { 0.0 };
        let max = base + parent_w * parents + shared_w * shared + sel;
        (0.0..=1.0).contains(&base) && (0.0..=1.0).contains(&max)
    })
}

fn four_node_power_set_catalog() -> EvidenceCatalog {
    let mut regimes = Vec::with_capacity(17);
    for mask in 0..16u32 {
        let interventions: Vec<_> = (0..4).filter(|i| mask & (1 << i) != 0).map(v).collect();
        let measured: Vec<_> = (0..4).filter(|i| mask & (1 << i) == 0).map(v).collect();
        regimes.push(
            EvidenceRegime::try_new(
                RegimeId::from_raw(mask),
                if mask == 0 { RegimeKind::Observational } else { RegimeKind::Experimental },
                EvidenceKind::Available,
                interventions,
                [],
                measured,
                "source",
                DistributionAvailability::Joint,
            )
            .unwrap(),
        );
    }
    regimes.push(
        EvidenceRegime::try_new(
            RegimeId::from_raw(16),
            RegimeKind::Observational,
            EvidenceKind::Available,
            [],
            [],
            [v(0), v(1), v(2), v(3)],
            "target",
            DistributionAvailability::Joint,
        )
        .unwrap(),
    );
    EvidenceCatalog::try_new([], regimes, [], None).unwrap()
}

fn four_node_weight(
    node: usize,
    values: usize,
    latent: usize,
    directed: u32,
    bidirected: u32,
    base: f64,
    parent_w: f64,
    shared_w: f64,
    sel: f64,
) -> f64 {
    let mut parents = 0.0;
    let mut shared = 0.0;
    for (edge, &(from, to)) in FOUR_EDGES.iter().enumerate() {
        if to == node && directed & (1 << edge) != 0 {
            parents += ((values >> from) & 1) as f64;
        }
        if (from == node || to == node) && bidirected & (1 << edge) != 0 {
            shared += ((latent >> edge) & 1) as f64;
        }
    }
    base + parent_w * parents + shared_w * shared + sel
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn four_node_formula_matches_scm(
    proof: &antecedent_identify::ClassicalTransportDerivation,
    catalog: &EvidenceCatalog,
    directed: u32,
    bidirected: u32,
    selections: u32,
    base: f64,
    parent_w: f64,
    shared_w: f64,
    sel_w: f64,
    ctx: &ExecutionContext,
) -> Result<(), String> {
    let mut source = vec![Vec::new(); 81];
    for mask in 0..16usize {
        let free = 4 - mask.count_ones() as usize;
        for values in 0..16usize {
            if values & !mask != 0 {
                continue;
            }
            source[four_node_slot(mask, values)] = vec![0.0; 1 << free];
        }
    }
    let mut target = vec![0.0; 16];
    let mut truth = [0.0; 2];
    let prior = 1.0 / 64.0;
    for latent in 0..64usize {
        for values in 0..16usize {
            let source_p: [f64; 4] = std::array::from_fn(|node| {
                four_node_weight(
                    node, values, latent, directed, bidirected, base, parent_w, shared_w, 0.0,
                )
            });
            let target_p: [f64; 4] = std::array::from_fn(|node| {
                let sel = if selections & (1 << node) != 0 { sel_w } else { 0.0 };
                four_node_weight(
                    node, values, latent, directed, bidirected, base, parent_w, shared_w, sel,
                )
            });
            let mut observational = prior;
            for node in 0..4 {
                observational *= bernoulli((values >> node) & 1, target_p[node]);
            }
            target[four_node_free_row(values, 0)] += observational;
            for mask in 0..16usize {
                let mut mass = prior;
                for node in 0..4 {
                    if mask & (1 << node) != 0 {
                        continue;
                    }
                    mass *= bernoulli((values >> node) & 1, source_p[node]);
                }
                let row = four_node_free_row(values, mask);
                source[four_node_slot(mask, values & mask)][row] += mass;
            }
            let x = values & 1;
            if (values >> 3) & 1 == 1 {
                let mut mass = prior;
                for node in 1..4 {
                    mass *= bernoulli((values >> node) & 1, target_p[node]);
                }
                truth[x] += mass;
            }
        }
    }
    let mut laws = Vec::with_capacity(82);
    for mask in 0..16usize {
        for values in 0..16usize {
            if values & !mask != 0 {
                continue;
            }
            let interventions = (0..4)
                .filter(|node| mask & (1 << node) != 0)
                .map(|node| antecedent_expr::InterventionAssignment {
                    variable: v(node as u32),
                    value: Value::Int64(((values >> node) & 1) as i64),
                })
                .collect::<Vec<_>>();
            let axes = (0..4)
                .filter(|node| mask & (1 << node) == 0)
                .map(|node| DiscreteAxis {
                    variable: v(node as u32),
                    values: Arc::from([Value::Int64(0), Value::Int64(1)]),
                })
                .collect::<Vec<_>>();
            laws.push(
                ExactDiscreteLaw::try_new(
                    "source",
                    RegimeId::from_raw(mask as u32),
                    interventions,
                    axes,
                    source[four_node_slot(mask, values)].clone(),
                    "oracle",
                    LawTolerance::default(),
                )
                .map_err(|err| {
                    format!("source law directed={directed} bidirected={bidirected} selections={selections} mask={mask} values={values}: {err}")
                })?,
            );
        }
    }
    laws.push(
        ExactDiscreteLaw::try_new(
            "target",
            RegimeId::from_raw(16),
            [],
            (0..4)
                .map(|node| DiscreteAxis {
                    variable: v(node as u32),
                    values: Arc::from([Value::Int64(0), Value::Int64(1)]),
                })
                .collect::<Vec<_>>(),
            target,
            "oracle",
            LawTolerance::default(),
        )
        .map_err(|err| format!("target law directed={directed} bidirected={bidirected}: {err}"))?,
    );
    let bound = proof.bind_catalog(catalog).map_err(|err| format!("bind: {err}"))?;
    let data = ExactTransportData::try_new(laws, 10_000).map_err(|err| format!("data: {err}"))?;
    for x in 0..2usize {
        let result = ExactEvaluationPlan::compile(
            bound.arena(),
            bound.root(),
            data.clone(),
            [v(3)],
            Assignment::from_pairs([(v(0), Value::Int64(x as i64))]),
            ExactEvaluationLimits::default(),
            LawTolerance::default(),
            ctx,
        )
        .and_then(|plan| plan.evaluate(ctx))
        .map_err(|err| format!("evaluate x={x} directed={directed} bidirected={bidirected} selections={selections}: {err}"))?;
        let got = result.mean(v(3)).map_err(|err| format!("mean: {err}"))?;
        if (got - truth[x]).abs() > 1e-8 {
            return Err(format!(
                "directed={directed} bidirected={bidirected} selections={selections} x={x} base={base} got={got} truth={}",
                truth[x]
            ));
        }
    }
    Ok(())
}

fn four_node_slot(mask: usize, values: usize) -> usize {
    let mut slot = 0;
    let mut place = 1;
    for node in 0..4 {
        let digit = if mask & (1 << node) == 0 { 0 } else { 1 + ((values >> node) & 1) };
        slot += digit * place;
        place *= 3;
    }
    slot
}

fn four_node_free_row(values: usize, mask: usize) -> usize {
    let mut row = 0;
    for node in 0..4 {
        if mask & (1 << node) != 0 {
            continue;
        }
        row = row * 2 + ((values >> node) & 1);
    }
    row
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

#[allow(clippy::too_many_lines)] // one linear derivation; splitting it would scatter the argument
#[test]
#[allow(
    clippy::too_many_lines,
    reason = "one sweep over every graph and parameterization, each compared against the enumerated truth in sequence"
)]
fn four_node_graph_parameter_sweep_matches_target_interventions() {
    // Domain: X→M→Y, Z→Y, X↔Y; Z→M on/off; selection {Y} vs empty; three
    // latent-SCM parameterizations. Truth is P(Y=1 | do(X=x)) from independent
    // enumeration. Unidentifiable variants must carry a checked s-hedge.
    let ctx = ExecutionContext::for_tests(5);
    let query = ClassicalTransportQuery {
        outcomes: Arc::from([v(3)]),
        treatments: Arc::from([v(0)]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    };
    for z_to_m in [false, true] {
        for select_y in [false, true] {
            let mut graph = Admg::with_variables(4);
            graph.insert_directed(d(0), d(1)).unwrap();
            graph.insert_directed(d(1), d(3)).unwrap();
            graph.insert_directed(d(2), d(3)).unwrap();
            graph.insert_bidirected(d(0), d(3)).unwrap();
            if z_to_m {
                graph.insert_directed(d(2), d(1)).unwrap();
            }
            let diagram =
                SelectionDiagram::try_new(graph, if select_y { vec![v(3)] } else { Vec::new() })
                    .unwrap();
            let proof =
                match identify_classical_transport(&diagram, &query, SidLimits::default(), &ctx)
                    .unwrap()
                {
                    ClassicalTransportResult::Identified(proof) => proof,
                    ClassicalTransportResult::ProvenNonTransportable(witness) => {
                        antecedent_identify::sid::verify_s_hedge(&diagram, &query, &witness, &ctx)
                            .unwrap();
                        continue;
                    }
                    ClassicalTransportResult::NotCertified => {
                        panic!("unchecked four-node sweep z_to_m={z_to_m} select_y={select_y}")
                    }
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
                                    let mediator = 0.2
                                        + mediator_shift * x as f64
                                        + if z_to_m { 0.2 * z as f64 } else { 0.0 };
                                    let mass = bernoulli(u, latent_mass)
                                        * bernoulli(x, if u == 1 { 0.75 } else { 0.25 })
                                        * bernoulli(m, mediator)
                                        * bernoulli(z, 0.4)
                                        * bernoulli(
                                            y,
                                            0.1 + 0.35 * m as f64
                                                + z_shift * z as f64
                                                + 0.2 * u as f64,
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
                    "four-node-sweep-oracle",
                    LawTolerance::default(),
                )
                .unwrap();
                let data = ExactTransportData::try_new([law], 10_000).unwrap();
                for x in 0..2 {
                    let mut truth = 0.0;
                    for u in 0..2 {
                        for m in 0..2 {
                            for z in 0..2 {
                                let mediator = 0.2
                                    + mediator_shift * x as f64
                                    + if z_to_m { 0.2 * z as f64 } else { 0.0 };
                                truth += bernoulli(u, latent_mass)
                                    * bernoulli(m, mediator)
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
                        "z_to_m={z_to_m} select_y={select_y} x={x} latent={latent_mass}"
                    );
                }
            }
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

fn binary_axis(i: u32) -> DiscreteAxis {
    DiscreteAxis { variable: v(i), values: Arc::from([Value::Int64(0), Value::Int64(1)]) }
}

#[test]
fn enlargement_does_not_require_the_child_at_levels_the_data_cannot_reach() {
    // W -> X -> Y with the Y mechanism selected. P*(y | do x) = P*(y | x). The
    // enlarged child P*(y | x, w) is a conditional on a null event at (w=0, x=1),
    // because P*(X=1 | W=0) = 0 while P*(X=1) > 0 and P*(W=0) > 0. Weighting by
    // P*(w) would need that undefined cell; weighting by P*(w | x) does not.
    let mut graph = Admg::with_variables(3);
    graph.insert_directed(d(0), d(1)).unwrap();
    graph.insert_directed(d(1), d(2)).unwrap();
    let diagram = SelectionDiagram::try_new(graph, [v(2)]).unwrap();
    let query = ClassicalTransportQuery {
        outcomes: Arc::from([v(2)]),
        treatments: Arc::from([v(1)]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    };
    let ctx = ExecutionContext::for_tests(21);
    let ClassicalTransportResult::Identified(derivation) =
        identify_classical_transport(&diagram, &query, SidLimits::default(), &ctx).unwrap()
    else {
        panic!("target DAG identifies");
    };
    assert!(derivation.rules().contains(&"sid.line3"));
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
    let mut observational = vec![0.0; 8];
    for w in 0..2usize {
        for x in 0..2usize {
            for y in 0..2usize {
                let p_x1 = if w == 1 { 0.6 } else { 0.0 };
                observational[w * 4 + x * 2 + y] +=
                    0.5 * bernoulli(x, p_x1) * bernoulli(y, 0.2 + 0.5 * x as f64);
            }
        }
    }
    let law = ExactDiscreteLaw::try_new(
        "target",
        RegimeId::from_raw(7),
        [],
        [binary_axis(0), binary_axis(1), binary_axis(2)],
        observational,
        "oracle",
        LawTolerance::default(),
    )
    .unwrap();
    let data = ExactTransportData::try_new([law], 1000).unwrap();
    for x in 0..2i64 {
        let plan = ExactEvaluationPlan::compile(
            functional.arena(),
            functional.root(),
            data.clone(),
            [v(2)],
            Assignment::from_pairs([(v(1), Value::Int64(x))]),
            ExactEvaluationLimits::default(),
            LawTolerance::default(),
            &ctx,
        )
        .unwrap();
        let result = plan.evaluate(&ctx).unwrap();
        // Y depends on X alone, so do(x) and conditioning on x agree: P(Y=1) = 0.2 + 0.5 x.
        let truth = 0.2 + 0.5 * x as f64;
        assert!((result.mean(v(2)).unwrap() - truth).abs() < 1e-12, "x={x}");
        assert!((result.probabilities[0] - (1.0 - truth)).abs() < 1e-12, "x={x}");
    }
}

/// W -> X -> Y, X <-> Y with selection on W: only the law of W (hence of X) differs.
fn confounded_treatment_with_selected_parent() -> (SelectionDiagram, ClassicalTransportQuery) {
    let mut graph = Admg::with_variables(3);
    graph.insert_directed(d(0), d(1)).unwrap();
    graph.insert_directed(d(1), d(2)).unwrap();
    graph.insert_bidirected(d(1), d(2)).unwrap();
    let diagram = SelectionDiagram::try_new(graph, [v(0)]).unwrap();
    let query = ClassicalTransportQuery {
        outcomes: Arc::from([v(2)]),
        treatments: Arc::from([v(1)]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    };
    (diagram, query)
}

#[test]
fn irrelevant_selection_upstream_of_the_treatment_is_direct_transport_at_the_root() {
    let (diagram, query) = confounded_treatment_with_selected_parent();
    let ctx = ExecutionContext::for_tests(22);
    let ClassicalTransportResult::Identified(derivation) =
        identify_classical_transport(&diagram, &query, SidLimits::default(), &ctx).unwrap()
    else {
        panic!("S is independent of Y under do(x), so the source experiment answers the query");
    };
    // Not enlarged over W: that would demand do(X, W) experiments and a target law of W.
    assert_eq!(derivation.rules(), ["transport.direct"]);
    let regime = EvidenceRegime::try_new(
        RegimeId::from_raw(1),
        RegimeKind::Experimental,
        EvidenceKind::Available,
        [v(1)],
        [],
        [v(2)],
        "source",
        DistributionAvailability::Joint,
    )
    .unwrap();
    let catalog = EvidenceCatalog::try_new([], [regime], [], None).unwrap();
    let bound = derivation.bind_catalog(&catalog).unwrap();
    // Latent U confounds X and Y; W only shifts X. P(Y=1 | do(x)) = sum_u 0.5 (0.1 + 0.5 x + 0.3 u).
    let mut laws = Vec::new();
    for x in 0..2usize {
        let mut world = vec![0.0; 2];
        for u in 0..2usize {
            for (y, mass) in world.iter_mut().enumerate() {
                *mass += 0.5 * bernoulli(y, 0.1 + 0.5 * x as f64 + 0.3 * u as f64);
            }
        }
        laws.push(
            ExactDiscreteLaw::try_new(
                "source",
                RegimeId::from_raw(1),
                [antecedent_expr::InterventionAssignment {
                    variable: v(1),
                    value: Value::Int64(x as i64),
                }],
                [binary_axis(2)],
                world,
                "oracle",
                LawTolerance::default(),
            )
            .unwrap(),
        );
    }
    let data = ExactTransportData::try_new(laws, 1000).unwrap();
    for x in 0..2i64 {
        let plan = ExactEvaluationPlan::compile(
            bound.arena(),
            bound.root(),
            data.clone(),
            [v(2)],
            Assignment::from_pairs([(v(1), Value::Int64(x))]),
            ExactEvaluationLimits::default(),
            LawTolerance::default(),
            &ctx,
        )
        .unwrap();
        let truth = 0.25 + 0.5 * x as f64;
        assert!((plan.evaluate(&ctx).unwrap().mean(v(2)).unwrap() - truth).abs() < 1e-12);
    }
}

#[test]
fn a_direct_transport_step_cannot_be_relabelled_as_figure_five_line_ten() {
    use antecedent_identify::sid::{ClassicalTransportDerivation, Rule};
    let (diagram, query) = confounded_treatment_with_selected_parent();
    let ctx = ExecutionContext::for_tests(23);
    let ClassicalTransportResult::Identified(derivation) =
        identify_classical_transport(&diagram, &query, SidLimits::default(), &ctx).unwrap()
    else {
        panic!("direct transport");
    };
    let record = derivation.to_record();
    assert_eq!(record.steps[record.root_step].rule, Rule::DirectTransport);
    let check = |record| {
        ClassicalTransportDerivation::from_record_checked(
            record,
            derivation.arena().clone(),
            &diagram,
            &query,
            SidLimits::default(),
            &ctx,
        )
    };
    check(record.clone()).unwrap();
    // The root state is enlargeable (W is not an ancestor of Y in G_X-bar), so line 10 cannot fire.
    let mut relabelled = record;
    let root = relabelled.root_step;
    relabelled.steps[root].rule = Rule::Source;
    assert!(check(relabelled).is_err());
}

#[test]
fn exhausting_the_standardizer_subset_budget_is_an_obligation_not_an_error() {
    use antecedent_identify::{identify_catalog_transport, CatalogTransportResult};
    // X, Y and 17 pretreatment parents of Y; the first parent's mechanism is selected, so
    // every admissible standardizer contains it and none can be bound from a catalog that
    // holds only the source experiment. 2^17 subsets exceed the step budget.
    let n = 19u32;
    let mut graph = Admg::with_variables(n);
    graph.insert_directed(d(0), d(1)).unwrap();
    for z in 2..n {
        graph.insert_directed(d(z), d(1)).unwrap();
    }
    let diagram = SelectionDiagram::try_new(graph, [v(2)]).unwrap();
    let query = ClassicalTransportQuery {
        outcomes: Arc::from([v(1)]),
        treatments: Arc::from([v(0)]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    };
    let regime = EvidenceRegime::try_new(
        RegimeId::from_raw(1),
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
    let ctx = ExecutionContext::for_tests(24);
    let limits = SidLimits { steps: 20_000, depth: 256 };
    let result = identify_catalog_transport(&diagram, &query, &catalog, limits, &ctx)
        .expect("subset-search exhaustion must not fail the call");
    let CatalogTransportResult::MissingEvidence { searched, obligations } = result else {
        panic!("nothing binds, and nothing here is a proof of impossibility: {result:?}");
    };
    assert_eq!(searched.len(), 3, "the source-first strategy must still run");
    assert!(
        obligations.iter().any(|o| o.contains("stopped after 20000 candidate subsets of 17")),
        "{obligations:?}"
    );
}
