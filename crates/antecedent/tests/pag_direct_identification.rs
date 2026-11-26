//! Published IDP and CIDP graphs and independent finite-law references.

use std::sync::Arc;

use antecedent::analysis::composition::{
    BundleBuilder, NodeKind, SuppliedSources, consume, detect_node_kind, export_bundle,
};
use antecedent::analysis::pag_identification::{
    PagIdError, PagIdentificationProblem, PagIdentificationProof, PagReplayError,
    compile_pag_identification, consume_pag_identification,
};
use antecedent_core::{ExecutionContext, RegimeId, Value, VariableId};
use antecedent_expr::{
    Assignment, DiscreteAxis, ExactDiscreteLaw, ExactEvaluationLimits, LawTolerance,
};
use antecedent_graph::{DenseNodeId, Endpoint, MarkedEdge, MiddleMark, Pag};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    names: Vec<String>,
    edges: Vec<(u32, u32, String, String)>,
    actions: Vec<u32>,
    outcomes: Vec<u32>,
    given: Vec<u32>,
}

fn fixture(case: &str) -> PagIdentificationProblem {
    let fixtures: std::collections::BTreeMap<String, Fixture> =
        serde_json::from_str(include_str!("../../../conformance/pag_direct_reference.json"))
            .unwrap();
    let Fixture { names, edges, actions, outcomes, given } =
        fixtures.into_iter().find(|(name, _)| name == case).unwrap().1;
    let endpoint = |mark: &str| match mark {
        "arrow" => Endpoint::Arrow,
        "tail" => Endpoint::Tail,
        "circle" => Endpoint::Circle,
        _ => panic!("unexpected mark"),
    };
    let mut graph = Pag::with_variables(u32::try_from(names.len()).unwrap());
    for (a, b, at_a, at_b) in edges {
        graph
            .insert_marked(MarkedEdge {
                a: DenseNodeId::from_raw(a),
                b: DenseNodeId::from_raw(b),
                at_a: endpoint(&at_a),
                at_b: endpoint(&at_b),
                middle: MiddleMark::Empty,
            })
            .unwrap();
    }
    PagIdentificationProblem {
        graph: antecedent_io::pag_to_wire(&graph).unwrap(),
        names,
        actions,
        outcomes,
        given,
        population: "observed".into(),
    }
}

fn atom(mask: usize, size: usize) -> Vec<usize> {
    (0..size).map(|i| (mask >> (size - i - 1)) & 1).collect()
}

fn binary_mass(bit: usize, p: f64) -> f64 {
    [1.0 - p, p][bit]
}

fn law(width: u32, mass: impl Fn(&[usize]) -> f64) -> ExactDiscreteLaw {
    let axes: Arc<[_]> = (0..width)
        .map(|id| DiscreteAxis {
            variable: VariableId::from_raw(id),
            values: Arc::from([Value::Int64(0), Value::Int64(1)]),
        })
        .collect::<Vec<_>>()
        .into();
    let probabilities: Arc<[_]> =
        (0..(1usize << width)).map(|i| mass(&atom(i, width as usize))).collect::<Vec<_>>().into();
    ExactDiscreteLaw::try_new(
        "observed",
        RegimeId::from_raw(0),
        [],
        axes,
        probabilities,
        "published-reference",
        LawTolerance::default(),
    )
    .unwrap()
}

fn assign(entries: &[(u32, i64)]) -> Assignment {
    Assignment::from_pairs(
        entries
            .iter()
            .map(|&(variable, level)| (VariableId::from_raw(variable), Value::Int64(level))),
    )
}

fn reference_eq_14(x1: usize, x2: usize, y: &[usize]) -> f64 {
    binary_mass(y[0], 0.1 + [0.0, 0.2][x1] + [0.0, 0.2][y[4]])
        * binary_mass(y[1], [0.2, 0.7][x2])
        * binary_mass(y[2], [0.2, 0.6][y[3]])
        * binary_mass(y[3], 0.4)
        * binary_mass(y[4], [0.3, 0.8][y[3]])
}

#[test]
fn published_beyond_adjustment_formula_replays_and_executes() {
    let ctx = ExecutionContext::for_tests(7);
    let problem = fixture("marginal_beyond_adjustment");
    let plan = compile_pag_identification(problem).unwrap();
    assert_eq!(plan.plan_view().route().as_str(), "DirectPagIdentification");
    let artifact = plan.execute(&ctx).unwrap();
    assert!(matches!(artifact.proof(), PagIdentificationProof::Identified { .. }));
    assert!(artifact.work() > 0);
    let laws = law(7, |a| {
        binary_mass(a[0], 0.4)
            * binary_mass(a[1], [0.3, 0.55][a[0]])
            * reference_eq_14(a[0], a[1], &a[2..])
    });
    let result = artifact
        .evaluate(laws, &assign(&[(0, 1), (1, 0)]), ExactEvaluationLimits::default(), &ctx)
        .unwrap();
    for (i, value) in result.probabilities.iter().enumerate() {
        let expected = reference_eq_14(1, 0, &atom(i, 5));
        assert!((value - expected).abs() < 1e-12, "atom {i}: {value} != {expected}");
    }
    let bytes = artifact.to_bytes().unwrap();
    let replay = consume_pag_identification(
        &bytes,
        artifact.identity(),
        plan.lowered().identity(),
        &ExecutionContext::for_tests(8),
    )
    .unwrap();
    assert_eq!(replay.proof(), artifact.proof());
    assert_eq!(detect_node_kind(&bytes), Some(NodeKind::PagIdentification));
    let mut bundle = BundleBuilder::new();
    bundle.add_detected_artifact(Some("pag"), &bytes).unwrap();
    let graph = bundle.build().unwrap();
    assert!(
        consume(
            &export_bundle(&graph, "direct-pag").unwrap(),
            graph.identity(),
            &SuppliedSources::default()
        )
        .unwrap()
        .all_replayed()
    );
}

#[test]
fn published_conditional_exchange_and_obstruction_are_distinct_from_resource_stop() {
    let ctx = ExecutionContext::for_tests(9);
    let problem = fixture("conditional_exchange");
    let plan = compile_pag_identification(problem.clone()).unwrap();
    let artifact = plan.execute(&ctx).unwrap();
    let obs = law(4, |a| {
        binary_mass(a[0], 0.35)
            * binary_mass(a[3], 0.5)
            * binary_mass(a[1], 0.1 + [0.0, 0.3][a[0]] + [0.0, 0.2][a[3]])
            * binary_mass(a[2], 0.1 + [0.0, 0.2][a[1]] + [0.0, 0.4][a[3]])
    });
    let answer = artifact
        .evaluate(obs, &assign(&[(0, 0), (1, 1), (3, 1)]), ExactEvaluationLimits::default(), &ctx)
        .unwrap();
    assert!((answer.probabilities[1] - 0.7).abs() < 1e-12);
    let mut marginal = problem;
    marginal.outcomes = vec![1, 2, 3];
    marginal.given.clear();
    assert!(matches!(
        compile_pag_identification(marginal).unwrap().execute(&ctx).unwrap().proof(),
        PagIdentificationProof::MarginalObstruction { .. }
    ));
    let reverse = compile_pag_identification(fixture("conditional_reverse_exchange")).unwrap();
    assert!(matches!(
        reverse.execute(&ctx).unwrap().proof(),
        PagIdentificationProof::Identified { .. }
    ));
    let failed = compile_pag_identification(fixture("conditional_obstruction")).unwrap();
    let evidence = failed.execute(&ctx).unwrap();
    assert!(
        matches!(evidence.proof(), PagIdentificationProof::ConditionalObstruction { path, .. } if path == &[0, 3, 2])
    );
    assert!(matches!(
        failed.execute(&ctx.with_id_work_limit(2)),
        Err(PagReplayError::Identification(PagIdError::Budget(2)))
    ));
    let bytes = evidence.to_bytes().unwrap();
    let replay =
        consume_pag_identification(&bytes, evidence.identity(), failed.lowered().identity(), &ctx)
            .unwrap();
    assert_eq!(replay.proof(), evidence.proof());
}

#[test]
fn all_two_vertex_pag_classes_match_the_semantic_definition() {
    let ctx = ExecutionContext::for_tests(9);
    for adjacency in [false, true] {
        let mut graph = Pag::with_variables(2);
        if adjacency {
            graph
                .insert_marked(MarkedEdge {
                    a: DenseNodeId::from_raw(0),
                    b: DenseNodeId::from_raw(1),
                    at_a: Endpoint::Circle,
                    at_b: Endpoint::Circle,
                    middle: MiddleMark::Empty,
                })
                .unwrap();
        }
        for action in [0, 1] {
            let problem = PagIdentificationProblem {
                graph: antecedent_io::pag_to_wire(&graph).unwrap(),
                names: vec!["X".into(), "Y".into()],
                actions: vec![action],
                outcomes: vec![1 - action],
                given: vec![],
                population: "observed".into(),
            };
            let artifact = compile_pag_identification(problem).unwrap().execute(&ctx).unwrap();
            if adjacency {
                // X -> Y with uniform X and independent 0.1 flip and its reverse
                // both give table [0.45, 0.05, 0.05, 0.45]. Their P(Y=1 | do(X=1))
                // are 0.9 and 0.5, respectively. The same witness works in reverse.
                assert!(matches!(
                    artifact.proof(),
                    PagIdentificationProof::MarginalObstruction { .. }
                ));
            } else {
                let product = law(2, |a| binary_mass(a[0], 0.25) * binary_mass(a[1], 0.6));
                let result = artifact
                    .evaluate(
                        product,
                        &assign(&[(action, 1)]),
                        ExactEvaluationLimits::default(),
                        &ctx,
                    )
                    .unwrap();
                assert!((result.probabilities[1] - [0.6, 0.25][action as usize]).abs() < 1e-12);
            }
        }
    }
}

#[test]
fn all_three_vertex_pag_classes_match_the_path_and_collider_reference() {
    // At three vertices the only observational-equivalence distinction beyond
    // skeleton is the single unshielded collider on a two-edge path.
    let ctx = ExecutionContext::for_tests(91);
    let pairs = [(0, 1), (0, 2), (1, 2)];
    for mask in 0u32..8 {
        let present: Vec<_> = pairs
            .iter()
            .enumerate()
            .filter_map(|(index, &pair)| (mask & (1 << index) != 0).then_some(pair))
            .collect();
        let middle = (present.len() == 2).then(|| {
            (0..3)
                .find(|node| present.iter().filter(|(a, b)| a == node || b == node).count() == 2)
                .unwrap()
        });
        for collider in [None, middle].into_iter().take(if middle.is_some() { 2 } else { 1 }) {
            let mut graph = Pag::with_variables(3);
            for &(a, b) in &present {
                let mark = |node| {
                    if collider == Some(node) { Endpoint::Arrow } else { Endpoint::Circle }
                };
                graph
                    .insert_marked(MarkedEdge {
                        a: DenseNodeId::from_raw(a),
                        b: DenseNodeId::from_raw(b),
                        at_a: mark(a),
                        at_b: mark(b),
                        middle: MiddleMark::Empty,
                    })
                    .unwrap();
            }
            for x in 0u32..3 {
                for y in (0u32..3).filter(|y| *y != x) {
                    let z = 3 - x - y;
                    for condition in [false, true] {
                        let expected = if let Some(center) = collider {
                            x == center || (y != center && !condition)
                        } else if present.contains(&(x.min(y), x.max(y))) {
                            false
                        } else {
                            present.len() != 2 || condition
                        };
                        let problem = PagIdentificationProblem {
                            graph: antecedent_io::pag_to_wire(&graph).unwrap(),
                            names: vec!["A".into(), "B".into(), "C".into()],
                            actions: vec![x],
                            outcomes: vec![y],
                            given: if condition { vec![z] } else { vec![] },
                            population: "observed".into(),
                        };
                        let direct =
                            compile_pag_identification(problem).unwrap().execute(&ctx).unwrap();
                        assert_eq!(
                            matches!(direct.proof(), PagIdentificationProof::Identified { .. }),
                            expected,
                            "skeleton={present:?}, collider={collider:?}, {x} -> {y}, condition={condition}",
                        );
                        if expected && !condition {
                            let query = antecedent_core::ResponseQuery::new(
                                antecedent_core::ResponseFunctional::InterventionResponse {
                                    outcome: VariableId::from_raw(y),
                                    interventions: Arc::from([antecedent_core::Intervention::set(
                                        VariableId::from_raw(x),
                                        Value::Int64(1),
                                    )]),
                                },
                            );
                            let other =
                                antecedent_identify::identify_pag_response_general(&graph, &query)
                                    .unwrap();
                            if mask == 0 {
                                assert_eq!(other.status, antecedent_core::IdentificationStatus::NonparametricallyIdentified);
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn direct_and_completion_values_agree_on_their_shared_identified_scope() {
    use antecedent::Study;
    use antecedent_core::{CausalQuery, Intervention, ResponseFunctional, ResponseQuery};
    use antecedent_data::TabularData;

    let graph = Pag::with_variables(2);
    let direct = compile_pag_identification(PagIdentificationProblem {
        graph: antecedent_io::pag_to_wire(&graph).unwrap(),
        names: vec!["X".into(), "Y".into()],
        actions: vec![0],
        outcomes: vec![1],
        given: vec![],
        population: "observed".into(),
    })
    .unwrap();
    let ctx = ExecutionContext::for_tests(2913);
    let joint = |atom: &[usize]| binary_mass(atom[0], 0.5) * binary_mass(atom[1], 0.625);
    let answer = direct
        .execute(&ctx)
        .unwrap()
        .evaluate(law(2, joint), &assign(&[(0, 1)]), ExactEvaluationLimits::default(), &ctx)
        .unwrap();
    let mut x = Vec::new();
    let mut y = Vec::new();
    for a in 0..4 {
        let cell = atom(a, 2);
        let count = if cell[1] == 0 { 96 } else { 160 };
        x.extend(std::iter::repeat_n(f64::from(cell[0] == 1), count));
        y.extend(std::iter::repeat_n(f64::from(cell[1] == 1), count));
    }
    let observations =
        TabularData::from_f64_columns([("X", x.as_slice()), ("Y", y.as_slice())]).unwrap();
    let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(1),
        interventions: Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(1.0))]),
    });
    let completion = Study::tabular(observations)
        .graph(graph)
        .query(CausalQuery::Response(query))
        .refute(antecedent::RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    assert!((answer.probabilities[1] - 0.625).abs() < 1e-12);
    assert!((completion.effect() - answer.probabilities[1]).abs() < 1e-10);
}

#[test]
fn visible_nonnull_effect_matches_the_completion_route_and_shared_oracle() {
    use antecedent::Study;
    use antecedent_core::{CausalQuery, Intervention, ResponseFunctional, ResponseQuery};
    use antecedent_data::TabularData;

    let problem = fixture("visible_collider_parent");
    let graph = antecedent_io::pag_from_wire(&problem.graph).unwrap();
    let ctx = ExecutionContext::for_tests(93);
    let direct = compile_pag_identification(problem).unwrap().execute(&ctx).unwrap();
    let joint = |a: &[usize]| {
        0.25 * binary_mass(a[1], 0.25 + 0.25 * (a[0] + a[2]) as f64)
            * binary_mass(a[3], [0.125, 0.875][a[1]])
    };
    let mut columns: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
    for bits in 0..16 {
        let cell = atom(bits, 4);
        let observations = (128.0 * joint(&cell)) as usize;
        for (column, value) in columns.iter_mut().zip(cell) {
            column.extend(std::iter::repeat_n(value as f64, observations));
        }
    }
    let data = TabularData::from_f64_columns(
        ["A", "B", "C", "D"].into_iter().zip(columns.iter().map(Vec::as_slice)),
    )
    .unwrap();
    for (b, expected) in [(0, 0.125), (1, 0.875)] {
        let answer = direct
            .evaluate(law(4, joint), &assign(&[(1, b)]), ExactEvaluationLimits::default(), &ctx)
            .unwrap();
        assert!((answer.probabilities[1] - expected).abs() < 1e-12);
        let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
            outcome: VariableId::from_raw(3),
            interventions: Arc::from([Intervention::set(
                VariableId::from_raw(1),
                Value::f64(b as f64),
            )]),
        });
        let completion = Study::tabular(data.clone())
            .graph(graph.clone())
            .query(CausalQuery::Response(query))
            .refute(antecedent::RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .run(&ctx)
            .unwrap();
        assert!((completion.effect() - answer.probabilities[1]).abs() < 1e-10);
    }
}

#[test]
fn retained_search_stop_replays_under_its_original_budget() {
    use antecedent::analysis::pag_identification::PagResourceStop;
    let plan = compile_pag_identification(fixture("conditional_obstruction")).unwrap();
    let ctx = ExecutionContext::for_tests(90);
    let short = ctx.clone().with_id_work_limit(2);
    let stopped = plan.execute_retaining_stop(&short).unwrap();
    assert!(matches!(
        stopped.proof(),
        PagIdentificationProof::ResourceStop {
            limit: PagResourceStop::Work,
            work: 2,
            work_limit: 2,
            depth_limit: None,
        }
    ));
    let problem = stopped.problem();
    let mass = 0.5_f64.powi(i32::try_from(problem.graph.node_count).unwrap());
    let observational = law(problem.graph.node_count, |_| mass);
    let bindings =
        problem.actions.iter().chain(&problem.given).map(|&id| (id, 0)).collect::<Vec<_>>();
    assert!(matches!(
        stopped.evaluate(observational, &assign(&bindings), ExactEvaluationLimits::default(), &ctx),
        Err(PagReplayError::Invalid("PAG result is not identified"))
    ));
    let bytes = stopped.to_bytes().unwrap();
    let replayed =
        consume_pag_identification(&bytes, stopped.identity(), plan.lowered().identity(), &ctx)
            .unwrap();
    assert_eq!(replayed.proof(), stopped.proof());
    assert!(
        consume_pag_identification(
            &bytes,
            stopped.identity(),
            plan.lowered().identity(),
            &ctx.clone().with_id_work_limit(1)
        )
        .is_err()
    );
    let completed = plan.execute_retaining_stop(&ctx).unwrap();
    assert!(matches!(completed.proof(), PagIdentificationProof::ConditionalObstruction { .. }));
    assert_ne!(completed.identity(), stopped.identity());
    let mut builder = BundleBuilder::new();
    builder.add_detected_artifact(Some("stopped"), &bytes).unwrap();
    let bundle = builder.build().unwrap();
    let encoded = export_bundle(&bundle, "retained-PAG-stop").unwrap();
    assert!(
        consume(&encoded, bundle.identity(), &SuppliedSources::default()).unwrap().all_replayed()
    );
}

/// Both SCMs use an independent uniform binary root and the same 10% flip noise.
/// The first graph directs X into Y and the second graph directs Y into X.
fn reversing_binary_world(x: usize, y: usize, do_x: Option<usize>, x_causes_y: bool) -> f64 {
    if do_x.is_some_and(|action| x != action) {
        return 0.0;
    }
    let transition = if x == y { 0.9 } else { 0.1 };
    match (do_x, x_causes_y) {
        (None, _) => 0.5 * transition,
        (Some(_), true) => transition,
        (Some(_), false) => 0.5,
    }
}

#[test]
fn observationally_equal_two_vertex_models_have_different_do_laws() {
    use antecedent_graph::MarkedEdge;

    let left: [f64; 4] = std::array::from_fn(|i| reversing_binary_world(i / 2, i % 2, None, true));
    let right: [f64; 4] =
        std::array::from_fn(|i| reversing_binary_world(i / 2, i % 2, None, false));
    assert_eq!(left, right);
    assert!((left.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    let effect = |world| (0..2).map(|x| reversing_binary_world(x, 1, Some(1), world)).sum::<f64>();
    assert_eq!(effect(true), 0.9);
    assert_eq!(effect(false), 0.5);

    // These DAGs share a complete two-node skeleton and have no unshielded collider,
    // so their invariant PAG has circles at both endpoints.
    let mut graph = Pag::with_variables(2);
    graph
        .insert_marked(MarkedEdge {
            a: DenseNodeId::from_raw(0),
            b: DenseNodeId::from_raw(1),
            at_a: Endpoint::Circle,
            at_b: Endpoint::Circle,
            middle: MiddleMark::Empty,
        })
        .unwrap();
    let identified = compile_pag_identification(PagIdentificationProblem {
        graph: antecedent_io::pag_to_wire(&graph).unwrap(),
        names: vec!["X".into(), "Y".into()],
        actions: vec![0],
        outcomes: vec![1],
        given: vec![],
        population: "observed".into(),
    })
    .unwrap()
    .execute(&ExecutionContext::for_tests(1))
    .unwrap();
    assert!(matches!(identified.proof(), PagIdentificationProof::MarginalObstruction { .. }));
}

#[test]
fn retained_depth_stop_is_distinct_from_the_work_budget_and_replays() {
    use antecedent::analysis::pag_identification::PagResourceStop;
    use antecedent_expr::ProgramLimits;

    // Removing singleton buckets reaches the expression-depth ceiling first.
    let depth_limit = ProgramLimits::default().max_depth;
    let count = u32::try_from(depth_limit + 2).unwrap();
    let graph = Pag::with_variables(count);
    let ctx = ExecutionContext::for_tests(8).with_id_work_limit(1_048_576);
    let plan = compile_pag_identification(PagIdentificationProblem {
        graph: antecedent_io::pag_to_wire(&graph).unwrap(),
        names: (0..count).map(|id| format!("variable-{id}")).collect(),
        actions: vec![0],
        outcomes: vec![count - 1],
        given: vec![],
        population: "observed".into(),
    })
    .unwrap();
    let stopped = plan.execute_retaining_stop(&ctx).unwrap();
    assert!(matches!(stopped.proof(), PagIdentificationProof::ResourceStop {
        limit: PagResourceStop::Depth, depth_limit: Some(limit), work, work_limit,
    } if *limit == depth_limit && *work < *work_limit));
    let replay = consume_pag_identification(
        &stopped.to_bytes().unwrap(),
        stopped.identity(),
        plan.lowered().identity(),
        &ctx,
    )
    .unwrap();
    assert_eq!(replay.proof(), stopped.proof());
}
