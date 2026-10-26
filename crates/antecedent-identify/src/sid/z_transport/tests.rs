use super::*;
use antecedent_core::EvidenceKind;
use antecedent_core::{
    DependenceGroup, Environment, RegimeBinding, RegimeId, SamplingDesign, Value,
    VariableCoordinate,
};
use antecedent_expr::{Assignment, DistributionProvider, EvalContext, EvalError, FactorSpec};
use antecedent_graph::{Admg, DenseNodeId};

fn query(graph_size: usize, controllable: &[u32]) -> (SelectionDiagram, ZTransportQuery) {
    let diagram = SelectionDiagram::try_new(
        Admg::with_variables(index_u32(graph_size)),
        Arc::<[VariableId]>::from([VariableId::from_raw(2)]),
    )
    .unwrap();
    let query = ZTransportQuery {
        outcomes: Arc::from([VariableId::from_raw(2)]),
        treatments: Arc::from([VariableId::from_raw(0)]),
        controllable: controllable
            .iter()
            .copied()
            .map(VariableId::from_raw)
            .collect::<Vec<_>>()
            .into(),
        experiment_assignment: controllable.first().map_or_else(
            || Arc::from([]),
            |variable| {
                Arc::from([CatalogInterventionAssignment {
                    variable: VariableId::from_raw(*variable),
                    value: Value::Bool(false),
                }])
            },
        ),
        source: Arc::from("source"),
        target: Arc::from("target"),
    };
    (diagram, query)
}

#[test]
fn bounded_query_contract_accepts_declared_controllable_set() {
    let (diagram, query) = query(12, &[1, 2, 3, 4]);
    validate_z_transport_query(&diagram, &query).unwrap();
}

#[test]
fn query_contract_has_no_observed_variable_ceiling() {
    let (diagram, query) = query(64, &[1, 2, 3, 4]);
    validate_z_transport_query(&diagram, &query).unwrap();
}

#[test]
fn query_contract_has_no_controllable_count_ceiling_but_needs_one_controllable() {
    let (diagram, wide) = query(12, &[0, 1, 2, 3, 4]);
    validate_z_transport_query(&diagram, &wide).unwrap();
    let (diagram, none) = query(12, &[]);
    assert_eq!(
        validate_z_transport_query(&diagram, &none).unwrap_err().to_string(),
        "z_transport.unsupported_controllable_count"
    );
}

/// A Markovian chain `0 -> 1 -> ... -> n-1` (no latent confounding): the effect of a
/// middle node on the last is identified by ordinary adjustment, so the recursion reaches
/// a formula without any source experiment on the treatment.
fn chain_fixture(n: u32, treatment: u32) -> (SelectionDiagram, ZTransportQuery) {
    let mut graph = Admg::with_variables(n);
    for i in 0..(n - 1) {
        graph.insert_directed(DenseNodeId::from_raw(i), DenseNodeId::from_raw(i + 1)).unwrap();
    }
    let diagram = SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from([])).unwrap();
    let query = ZTransportQuery {
        outcomes: Arc::from([VariableId::from_raw(n - 1)]),
        treatments: Arc::from([VariableId::from_raw(treatment)]),
        controllable: Arc::from([VariableId::from_raw(0)]),
        experiment_assignment: Arc::from([]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    };
    (diagram, query)
}

#[test]
fn graphs_beyond_the_old_twelve_variable_ceiling_are_identified_and_replayed() {
    // Sixteen observed variables, fourteen past the old ceiling's reach in the chain.
    let ctx = antecedent_core::ExecutionContext::for_tests(0);
    let (diagram, query) = chain_fixture(16, 5);
    let ZTransportResult::Identified(proof) =
        identify_z_transport(&diagram, &query, super::super::SidLimits::default(), &ctx)
            .unwrap()
    else {
        panic!("an unconfounded chain of sixteen variables must be identified");
    };
    assert_eq!(proof.kind, ZFormulaKind::Recursive);
    verify_z_transport_derivation(
        &diagram,
        &query,
        &proof,
        super::super::SidLimits::default(),
        &ctx,
    )
    .unwrap();
}

#[test]
fn a_fourteen_node_graph_completes_and_its_derivation_audit_is_reproduced() {
    // Fourteen observed variables, past the old ceiling of twelve: the search completes,
    // the derivation verifies against the graph and the query, and replaying the search
    // reproduces it. The treatment sits mid-graph so the recursion has to descend.
    let ctx = antecedent_core::ExecutionContext::for_tests(0);
    let (diagram, query) = chain_fixture(14, 6);
    let limits = super::super::SidLimits::default();
    let identify = || match identify_z_transport(&diagram, &query, limits, &ctx).unwrap() {
        ZTransportResult::Identified(proof) => proof,
        other @ ZTransportResult::NotCertified { .. } => {
            panic!("an unconfounded chain of fourteen variables is identified: {other:?}")
        }
    };
    let proof = identify();
    assert_eq!(proof.kind, ZFormulaKind::Recursive);
    assert!(!proof.rules.is_empty());
    verify_z_transport_derivation(&diagram, &query, &proof, limits, &ctx).unwrap();
    let replay = identify();
    assert_eq!(replay.rules, proof.rules);
    assert_eq!(replay.graph_signature, proof.graph_signature);
    // The audit is of this graph: a derivation checked against another query is refused.
    let (_, other_query) = chain_fixture(14, 7);
    assert!(
        verify_z_transport_derivation(&diagram, &other_query, &proof, limits, &ctx).is_err()
    );
}

#[test]
fn an_exhausted_search_budget_is_refused_naming_the_budget_and_its_limits() {
    let (diagram, query) = chain_fixture(14, 6);
    let ctx = antecedent_core::ExecutionContext::for_tests(0);
    let limits = super::super::SidLimits { steps: 1, depth: 14 };
    let mut receipt = None;
    let error = identify_z_transport_reporting(&diagram, &query, limits, &ctx, &mut receipt)
        .unwrap_err();
    assert!(error.is_budget_or_cancel());
    // The refusal names the z-transport search budget, never a verdict about the effect;
    // the receipt says which of its limits stopped it.
    assert_eq!(error, IdentificationError::budget(IdentificationBudget::ZTransport));
    assert_eq!(error.to_string(), "z_transport.exhausted_computation");
    let receipt = receipt.expect("a stopped search records what it consumed");
    assert_eq!(receipt.budget, ZTransportBudgetKind::Steps);
    assert_eq!((receipt.steps_limit, receipt.depth_limit), (1, 14));
    // The depth budget is named as its own.
    let shallow = super::super::SidLimits { steps: 10_000, depth: 1 };
    let mut receipt = None;
    let error = identify_z_transport_reporting(&diagram, &query, shallow, &ctx, &mut receipt)
        .unwrap_err();
    assert!(error.is_budget_or_cancel(), "{error:?}");
    assert_eq!(receipt.expect("a stopped search records its limits").depth_limit, 1);
}

#[test]
fn a_direct_source_exchange_is_identified_on_a_wide_graph() {
    // The treatment is the controllable variable with a concrete level, on 20 variables.
    let (diagram, mut query) = chain_fixture(20, 3);
    query.controllable = Arc::from([VariableId::from_raw(3)]);
    query.experiment_assignment = Arc::from([CatalogInterventionAssignment {
        variable: VariableId::from_raw(3),
        value: Value::Bool(true),
    }]);
    let ctx = antecedent_core::ExecutionContext::for_tests(0);
    let ZTransportResult::Identified(proof) =
        identify_z_transport(&diagram, &query, super::super::SidLimits::default(), &ctx)
            .unwrap()
    else {
        panic!("a concrete source exchange must be identified");
    };
    assert_eq!(proof.kind, ZFormulaKind::DirectJoint);
}

#[test]
fn a_wide_graph_search_stops_at_its_budget_with_a_receipt_not_a_verdict() {
    let (diagram, query) = chain_fixture(24, 2);
    let ctx = antecedent_core::ExecutionContext::for_tests(0);
    // Cutting the treatment's incoming edge leaves the two upstream variables outside the
    // outcome's ancestry, so the first rule recurses and needs a second step.
    let limits = super::super::SidLimits { steps: 1, depth: 24 };
    let mut receipt = None;
    let error = identify_z_transport_reporting(&diagram, &query, limits, &ctx, &mut receipt)
        .unwrap_err();
    assert!(error.is_budget_or_cancel());
    let receipt = receipt.expect("a stopped search records what it consumed");
    assert_eq!(receipt.steps_limit, 1);
    let cancelled = antecedent_core::ExecutionContext::for_tests(0);
    cancelled.cancellation.cancel();
    let mut receipt = None;
    let error = identify_z_transport_reporting(
        &diagram,
        &query,
        super::super::SidLimits::default(),
        &cancelled,
        &mut receipt,
    )
    .unwrap_err();
    assert!(matches!(error, IdentificationError::Cancelled));
    assert_eq!(receipt.unwrap().budget, ZTransportBudgetKind::Cancelled);
}

#[test]
fn negative_family_enumeration_stops_at_the_first_absent_regime() {
    let (diagram, query) = query(4, &[0, 1, 2, 3]);
    let coordinates = (0..4)
        .map(|raw| VariableCoordinate {
            variable: VariableId::from_raw(raw),
            domain: VariableDomain::Categorical { cardinality: 10 },
            unit: None,
        })
        .collect::<Vec<_>>();
    let source = Environment::try_new("source", coordinates.clone(), []).unwrap();
    let target = Environment::try_new("target", coordinates, []).unwrap();
    let catalog = EvidenceCatalog::try_new([source, target], [], [], None).unwrap();
    assert!(matches!(
        validate_z_experiment_family(&diagram, &query, &catalog),
        Err(ZExperimentFamilyError::MissingJointLaw { .. })
    ));
}

#[test]
fn family_check_names_a_missing_law_for_a_controllable_set_too_wide_to_enumerate() {
    // 64 binary controllables: the family has 3^64 - 1 regimes, far past any catalog.
    // It is reported as missing the first absent law, never an empty "complete" family,
    // and is never materialized.
    let variables = (0..64).map(VariableId::from_raw).collect::<Vec<_>>();
    let coordinates = variables
        .iter()
        .map(|&variable| VariableCoordinate {
            variable,
            domain: VariableDomain::Binary,
            unit: None,
        })
        .collect::<Vec<_>>();
    let catalog = EvidenceCatalog::try_new(
        [
            Environment::try_new("source", coordinates.clone(), []).unwrap(),
            Environment::try_new("target", coordinates, []).unwrap(),
        ],
        [],
        [],
        None,
    )
    .unwrap();
    let diagram =
        SelectionDiagram::try_new(Admg::with_variables(64), Arc::<[VariableId]>::from([]))
            .unwrap();
    let query = ZTransportQuery {
        outcomes: Arc::from([variables[0]]),
        treatments: Arc::from([variables[1]]),
        controllable: variables.clone().into(),
        experiment_assignment: Arc::from([]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    };
    assert!(matches!(
        validate_z_experiment_family(&diagram, &query, &catalog),
        Err(ZExperimentFamilyError::MissingJointLaw { .. })
    ));
}

#[test]
fn contract_rejects_unknown_and_duplicate_coordinates() {
    let (diagram, mut query) = query(3, &[1]);
    query.controllable = Arc::from([VariableId::from_raw(9)]);
    assert_eq!(
        validate_z_transport_query(&diagram, &query).unwrap_err().to_string(),
        "z_transport.unknown_variable"
    );
    query.controllable = Arc::from([VariableId::from_raw(1), VariableId::from_raw(1)]);
    assert_eq!(
        validate_z_transport_query(&diagram, &query).unwrap_err().to_string(),
        "z_transport.duplicate_variable"
    );
}

fn surrogate_fixture() -> (SelectionDiagram, ZTransportQuery) {
    let mut graph = Admg::with_variables(4);
    for (from, to) in [(0, 1), (1, 2), (2, 3), (0, 3)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    for (a, b) in [(0, 3), (1, 3), (1, 2)] {
        graph.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    let diagram = SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from([])).unwrap();
    let query = ZTransportQuery {
        outcomes: Arc::from([VariableId::from_raw(3)]),
        treatments: Arc::from([VariableId::from_raw(2)]),
        controllable: Arc::from([VariableId::from_raw(1)]),
        experiment_assignment: Arc::from([CatalogInterventionAssignment {
            variable: VariableId::from_raw(1),
            value: Value::Bool(false),
        }]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    };
    (diagram, query)
}

#[test]
fn registered_surrogate_query_has_checked_formula_and_rejects_substitution() {
    let (diagram, query) = surrogate_fixture();
    let ZTransportResult::Identified(proof) = identify_z_transport(
        &diagram,
        &query,
        super::super::SidLimits::default(),
        &antecedent_core::ExecutionContext::for_tests(0),
    )
    .unwrap() else {
        panic!("registered surrogate graph should be identified");
    };
    assert_eq!(proof.surrogate(), Some(VariableId::from_raw(1)));
    assert_eq!(proof.confounder(), Some(VariableId::from_raw(0)));
    let ExprNode::SumOut { variables, expr } = proof.arena().node(proof.root()) else {
        panic!("formula root must sum out W");
    };
    assert_eq!(proof.arena().var_set(*variables), &[VariableId::from_raw(0)]);
    let ExprNode::Product(factors) = proof.arena().node(*expr) else {
        panic!("formula must multiply two source factors");
    };
    assert_eq!(proof.arena().list(*factors).len(), 2);
    verify_z_transport_derivation(
        &diagram,
        &query,
        &proof,
        super::super::SidLimits::default(),
        &antecedent_core::ExecutionContext::for_tests(0),
    )
    .unwrap();
    let mut altered = query.clone();
    altered.controllable = Arc::from([VariableId::from_raw(0)]);
    assert!(
        verify_z_transport_derivation(
            &diagram,
            &altered,
            &proof,
            super::super::SidLimits::default(),
            &antecedent_core::ExecutionContext::for_tests(0)
        )
        .is_err()
    );
}

#[test]
fn direct_exchange_requires_concrete_joint_intervention_and_checks_proof() {
    let mut graph = Admg::with_variables(3);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    let diagram = SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from([])).unwrap();
    let query = ZTransportQuery {
        outcomes: Arc::from([VariableId::from_raw(2)]),
        treatments: Arc::from([VariableId::from_raw(0), VariableId::from_raw(1)]),
        controllable: Arc::from([VariableId::from_raw(0), VariableId::from_raw(1)]),
        experiment_assignment: Arc::from([
            CatalogInterventionAssignment {
                variable: VariableId::from_raw(0),
                value: Value::Bool(true),
            },
            CatalogInterventionAssignment {
                variable: VariableId::from_raw(1),
                value: Value::Bool(false),
            },
        ]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    };
    let ZTransportResult::Identified(proof) = identify_z_transport(
        &diagram,
        &query,
        super::super::SidLimits::default(),
        &antecedent_core::ExecutionContext::for_tests(0),
    )
    .unwrap() else {
        panic!("direct joint source exchange must be identified")
    };
    assert_eq!(proof.to_record().rules, ["ztr.source_exchange_joint"]);
    assert!(matches!(proof.arena().node(proof.root()), ExprNode::Distribution { .. }));
    let reconstructed = ZTransportDerivation::from_record_checked(
        &diagram,
        &query,
        &proof.to_record(),
        proof.arena().clone(),
        super::super::SidLimits::default(),
        &antecedent_core::ExecutionContext::for_tests(0),
    )
    .unwrap();
    verify_z_transport_derivation(
        &diagram,
        &query,
        &reconstructed,
        super::super::SidLimits::default(),
        &antecedent_core::ExecutionContext::for_tests(0),
    )
    .unwrap();
    let mut changed = query;
    changed.experiment_assignment = Arc::from([
        CatalogInterventionAssignment {
            variable: VariableId::from_raw(0),
            value: Value::Bool(false),
        },
        CatalogInterventionAssignment {
            variable: VariableId::from_raw(1),
            value: Value::Bool(false),
        },
    ]);
    assert!(
        verify_z_transport_derivation(
            &diagram,
            &changed,
            &proof,
            super::super::SidLimits::default(),
            &antecedent_core::ExecutionContext::for_tests(0)
        )
        .is_err()
    );
}

#[test]
fn formula_matches_independent_binary_scm_truth() {
    // Independent binary exogenous causes: A→W,Y; B→Z,Y; C→Z,X.
    // The three shared exogenous variables induce precisely the fixture's
    // bidirected edges. Non-symmetric probabilities make the check numerical.
    let p_a = 0.25;
    let p_b = 0.20;
    let p_c = 0.35;
    let mut target = [0.0_f64; 2];
    let mut experiment = [[[0.0_f64; 2]; 2]; 2]; // w,x,y under do(Z)
    for a in 0..=1 {
        for b in 0..=1 {
            for c in 0..=1 {
                let mass = if a == 1 { p_a } else { 1.0 - p_a }
                    * if b == 1 { p_b } else { 1.0 - p_b }
                    * if c == 1 { p_c } else { 1.0 - p_c };
                let w = a;
                // Effect truth under do(X=0), marginalizing the graph's
                // exogenous variables independently of the intervention.
                let y_do_x = w ^ a ^ b;
                target[y_do_x] += mass;
                // The source experiment is do(Z=0); all W, X, Y are jointly
                // measured and X still shares C with Z.
                let z = 0;
                let x = z ^ c;
                let y = x ^ w ^ a ^ b;
                experiment[w][x][y] += mass;
            }
        }
    }
    let mut formula = 0.0;
    for stratum in &experiment {
        let pw = stratum.iter().flatten().sum::<f64>();
        let y_given_w_x = stratum[0][1] / (stratum[0][0] + stratum[0][1]);
        formula += pw * y_given_w_x;
    }
    assert!((formula - target[1]).abs() < 1e-12, "formula={formula}, truth={}", target[1]);
}

#[test]
fn recursive_trz_search_reaches_registered_surrogate() {
    let (diagram, query) = surrogate_fixture();
    let answer = search_trz(
        &diagram,
        &query,
        super::super::SidLimits::default(),
        &antecedent_core::ExecutionContext::for_tests(7),
    )
    .unwrap();
    assert!(answer.is_some(), "TRz recursion should find the surrogate case");
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "one fixture exercised end to end; later assertions read locals built above them"
)]
fn recursive_trz_surrogate_formula_matches_independent_scm() {
    use antecedent_expr::{
        Assignment, DiscreteAxis, EvalContext, ExactDiscreteLaw, ExactTransportData,
        InterventionAssignment as ExprAssignment, LawTolerance,
    };
    let (diagram, query) = surrogate_fixture();
    let (arena, root, trace) = search_trz(
        &diagram,
        &query,
        super::super::SidLimits::default(),
        &antecedent_core::ExecutionContext::for_tests(7),
    )
    .unwrap()
    .unwrap();
    assert!(trace.iter().any(|rule| rule.starts_with("ztr.line10.source_exchange:")));
    let mut bound = arena.clone();
    let mut remap = Vec::<ExprId>::new();
    for index in 0..arena.len() {
        let node = match arena.node(ExprId::from_raw(index_u32(index))).clone() {
            ExprNode::Distribution {
                variables,
                conditioned_on,
                intervention,
                domain,
                population,
                ..
            } => ExprNode::Distribution {
                variables,
                conditioned_on,
                intervention,
                domain,
                population,
                regime: Some(if arena.population(population) == "source" {
                    RegimeId::from_raw(1)
                } else {
                    RegimeId::from_raw(0)
                }),
            },
            ExprNode::SumOut { variables, expr } => {
                ExprNode::SumOut { variables, expr: remap[expr.raw() as usize] }
            }
            ExprNode::Ratio { numerator, denominator } => ExprNode::Ratio {
                numerator: remap[numerator.raw() as usize],
                denominator: remap[denominator.raw() as usize],
            },
            ExprNode::Product(list) => {
                let children = arena
                    .list(list)
                    .iter()
                    .map(|id| remap[id.raw() as usize])
                    .collect::<Vec<_>>();
                ExprNode::Product(bound.intern_list(children))
            }
            other => panic!("unexpected recursive node: {other:?}"),
        };
        remap.push(bound.intern(node));
    }
    let root = remap[root.raw() as usize];
    let mut target = vec![0.0; 16];
    let mut source = vec![0.0; 8];
    for a in 0..=1usize {
        for b in 0..=1usize {
            for c in 0..=1usize {
                let mass = (if a == 1 { 0.25 } else { 0.75 })
                    * (if b == 1 { 0.20 } else { 0.80 })
                    * (if c == 1 { 0.35 } else { 0.65 });
                let w = a;
                let z = w ^ b ^ c;
                let x = z ^ c;
                let y = x ^ (w & b);
                target[w * 8 + z * 4 + x * 2 + y] += mass;
                let sx = c;
                let sy = sx ^ (w & b);
                source[w * 4 + sx * 2 + sy] += mass;
            }
        }
    }
    let axis = |raw| DiscreteAxis {
        variable: VariableId::from_raw(raw),
        values: Arc::from([Value::Bool(false), Value::Bool(true)]),
    };
    let target_law = ExactDiscreteLaw::try_new(
        "target",
        RegimeId::from_raw(0),
        [],
        [axis(0), axis(1), axis(2), axis(3)],
        target,
        "target-obs",
        LawTolerance::default(),
    )
    .unwrap();
    let source_law = ExactDiscreteLaw::try_new(
        "source",
        RegimeId::from_raw(1),
        [ExprAssignment::concrete(VariableId::from_raw(1), Value::Bool(false))],
        [axis(0), axis(2), axis(3)],
        source,
        "source-do-z0",
        LawTolerance::default(),
    )
    .unwrap();
    let provider = ExactTransportData::try_new([target_law, source_law], 128).unwrap();
    let assignment = Assignment::from_pairs([
        (VariableId::from_raw(2), Value::Bool(false)),
        (VariableId::from_raw(3), Value::Bool(true)),
    ]);
    let answer = bound
        .compile(root)
        .unwrap()
        .evaluate_with(&bound, &provider, &EvalContext::default(), &assignment)
        .unwrap();
    assert!((answer - 0.05).abs() < 1e-12, "recursive answer={answer}");
}

#[test]
fn recursive_route_binds_nonregistered_graph_using_actual_margins() {
    let mut graph = Admg::with_variables(5);
    for (a, b) in [(0, 1), (1, 2), (2, 3), (0, 3)] {
        graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    for (a, b) in [(0, 3), (1, 3), (1, 2)] {
        graph.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    let diagram = SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from([])).unwrap();
    let (_, query) = surrogate_fixture();
    let ZTransportResult::Identified(proof) = identify_z_transport(
        &diagram,
        &query,
        super::super::SidLimits::default(),
        &antecedent_core::ExecutionContext::for_tests(0),
    )
    .unwrap() else {
        panic!("recursive TRz should identify the graph with an isolated extra node")
    };
    assert_eq!(
        proof.to_record().rules.first().map(String::as_str),
        Some("ztr.recursive_reduction")
    );
    let coords = (0..5)
        .map(|raw| VariableCoordinate {
            variable: VariableId::from_raw(raw),
            domain: VariableDomain::Binary,
            unit: None,
        })
        .collect::<Vec<_>>();
    let source =
        Environment::try_new("source", coords.clone(), Arc::<[VariableId]>::from([])).unwrap();
    let target = Environment::try_new("target", coords, Arc::<[VariableId]>::from([])).unwrap();
    let obs = antecedent_core::EvidenceRegime::try_new(
        RegimeId::from_raw(0),
        RegimeKind::Observational,
        EvidenceKind::Available,
        [],
        [],
        (0..5).map(VariableId::from_raw).collect::<Vec<_>>(),
        "target",
        DistributionAvailability::Joint,
    )
    .unwrap();
    let experiment = antecedent_core::EvidenceRegime::try_new(
        RegimeId::from_raw(1),
        RegimeKind::Experimental,
        EvidenceKind::Available,
        [VariableId::from_raw(1)],
        [CatalogInterventionAssignment {
            variable: VariableId::from_raw(1),
            value: Value::Bool(false),
        }],
        [VariableId::from_raw(0), VariableId::from_raw(2), VariableId::from_raw(3)],
        "source",
        DistributionAvailability::Joint,
    )
    .unwrap();
    let bindings = [0, 1].map(|raw| RegimeBinding {
        dataset_identity: None,
        regime: RegimeId::from_raw(raw),
        snapshot_identity: Arc::from(format!("snapshot-{raw}")),
        schema_names: Arc::from([]),
        sampling: SamplingDesign::Independent,
        weights: None,
        dependence: DependenceGroup::IndependentStudies,
    });
    let catalog =
        EvidenceCatalog::try_new([source, target], [obs, experiment], bindings, None).unwrap();
    let bound = bind_z_transport_catalog(&diagram, &query, &proof, &catalog).unwrap();
    assert_eq!(bound.catalog().regimes.len(), 2);
    ZTransportDerivation::from_record_checked(
        &diagram,
        &query,
        &proof.to_record(),
        proof.arena().clone(),
        super::super::SidLimits::default(),
        &antecedent_core::ExecutionContext::for_tests(0),
    )
    .unwrap();
}

/// An available, bound joint regime for the preference tests. `levels`
/// `None` declares the whole intervention family.
fn preference_regime(
    raw: u32,
    population: &str,
    interventions: &[u32],
    levels: Option<&[bool]>,
    measured: &[u32],
) -> antecedent_core::EvidenceRegime {
    antecedent_core::EvidenceRegime::try_new(
        RegimeId::from_raw(raw),
        if interventions.is_empty() {
            RegimeKind::Observational
        } else {
            RegimeKind::Experimental
        },
        EvidenceKind::Available,
        interventions.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>(),
        levels.map_or_else(Vec::new, |levels| {
            interventions
                .iter()
                .zip(levels)
                .map(|(variable, level)| CatalogInterventionAssignment {
                    variable: VariableId::from_raw(*variable),
                    value: Value::Bool(*level),
                })
                .collect()
        }),
        measured.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>(),
        population,
        DistributionAvailability::Joint,
    )
    .unwrap()
}

/// A binary source/target catalog over `n` variables binding every regime.
fn preference_catalog(
    n: u32,
    regimes: Vec<antecedent_core::EvidenceRegime>,
) -> EvidenceCatalog {
    let coords = (0..n)
        .map(|raw| VariableCoordinate {
            variable: VariableId::from_raw(raw),
            domain: VariableDomain::Binary,
            unit: None,
        })
        .collect::<Vec<_>>();
    let bindings = regimes
        .iter()
        .map(|regime| RegimeBinding {
            dataset_identity: None,
            regime: regime.id,
            snapshot_identity: Arc::from(format!("snapshot-{}", regime.id.raw())),
            schema_names: Arc::from([]),
            sampling: SamplingDesign::Independent,
            weights: None,
            dependence: DependenceGroup::IndependentStudies,
        })
        .collect::<Vec<_>>();
    EvidenceCatalog::try_new(
        [
            Environment::try_new("source", coords.clone(), Arc::<[VariableId]>::from([]))
                .unwrap(),
            Environment::try_new("target", coords, Arc::<[VariableId]>::from([])).unwrap(),
        ],
        regimes,
        bindings,
        None,
    )
    .unwrap()
}

fn cited(bound: &BoundZTransportFunctional) -> Vec<u32> {
    bound.cited_regimes().iter().map(|regime| regime.raw()).collect()
}

/// Z→X→Y with an observed confounder W→X, W→Y: the recursive formula cites
/// the target's observational joint. Two target regimes that both supply
/// it bind the lowest id, whatever the catalog order.
#[test]
fn recursive_concrete_factor_binds_the_lowest_of_two_supplying_regimes() {
    let mut graph = Admg::with_variables(5);
    for (a, b) in [(1, 2), (2, 3), (0, 2), (0, 3)] {
        graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    let diagram = SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from([])).unwrap();
    let (_, query) = surrogate_fixture();
    let ZTransportResult::Identified(proof) = identify_z_transport(
        &diagram,
        &query,
        super::super::SidLimits::default(),
        &antecedent_core::ExecutionContext::for_tests(0),
    )
    .unwrap() else {
        panic!("the observed-confounder graph identifies recursively")
    };
    let all = [0, 1, 2, 3, 4];
    for order in [[1, 4], [4, 1]] {
        let catalog = preference_catalog(
            5,
            order
                .iter()
                .map(|raw| preference_regime(*raw, "target", &[], None, &all))
                .collect(),
        );
        let bound = bind_z_transport_catalog(&diagram, &query, &proof, &catalog).unwrap();
        assert_eq!(cited(&bound), [1], "catalog order {order:?}");
    }
}

/// The recursive route's concrete source experiment do(X1 = 0): a family
/// regime on X1 (no declared levels) supplies it too and is preferred over
/// a concrete regime with a lower id; two concrete regimes bind the lowest.
#[test]
fn recursive_concrete_experiment_prefers_a_family_regime_then_the_lowest_id() {
    let mut graph = Admg::with_variables(5);
    for (a, b) in [(0, 1), (1, 2), (2, 3), (0, 3)] {
        graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    for (a, b) in [(0, 3), (1, 3), (1, 2)] {
        graph.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    let diagram = SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from([])).unwrap();
    let (_, query) = surrogate_fixture();
    let ZTransportResult::Identified(proof) = identify_z_transport(
        &diagram,
        &query,
        super::super::SidLimits::default(),
        &antecedent_core::ExecutionContext::for_tests(0),
    )
    .unwrap() else {
        panic!("recursive TRz should identify the graph with an isolated extra node")
    };
    let observational = preference_regime(0, "target", &[], None, &[0, 1, 2, 3, 4]);
    let measured = [0, 2, 3];
    let concrete = |raw| preference_regime(raw, "source", &[1], Some(&[false]), &measured);
    let family = |raw| preference_regime(raw, "source", &[1], None, &measured);
    let catalog = preference_catalog(5, vec![observational.clone(), concrete(1), family(7)]);
    let bound = bind_z_transport_catalog(&diagram, &query, &proof, &catalog).unwrap();
    assert_eq!(cited(&bound), [7]);
    let catalog = preference_catalog(5, vec![observational, concrete(6), concrete(2)]);
    let bound = bind_z_transport_catalog(&diagram, &query, &proof, &catalog).unwrap();
    assert_eq!(cited(&bound), [2]);
}

/// Bow graph W→Y, X→Y, X↔Y with selection on W: the exchanged treatment
/// stays symbolic and binds one regime per world. A world supplied twice
/// binds the lowest id; a family regime supplies every world and wins.
#[test]
fn per_world_binding_prefers_a_family_regime_then_the_lowest_id_per_world() {
    let mut graph = Admg::with_variables(3);
    for (a, b) in [(0, 2), (1, 2)] {
        graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    graph.insert_bidirected(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    let diagram =
        SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from([VariableId::from_raw(0)]))
            .unwrap();
    let query = ZTransportQuery {
        outcomes: Arc::from([VariableId::from_raw(2)]),
        treatments: Arc::from([VariableId::from_raw(1)]),
        controllable: Arc::from([VariableId::from_raw(1)]),
        experiment_assignment: Arc::from([CatalogInterventionAssignment {
            variable: VariableId::from_raw(1),
            value: Value::Bool(false),
        }]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    };
    let ZTransportResult::Identified(proof) = identify_z_transport(
        &diagram,
        &query,
        super::super::SidLimits::default(),
        &antecedent_core::ExecutionContext::for_tests(0),
    )
    .unwrap() else {
        panic!("the bow graph identifies")
    };
    let observational = preference_regime(0, "target", &[], None, &[0, 1, 2]);
    let at = |raw, level| preference_regime(raw, "source", &[1], Some(&[level]), &[0, 2]);
    let catalog = preference_catalog(
        3,
        vec![observational.clone(), at(5, false), at(2, true), at(3, false)],
    );
    let bound = bind_z_transport_catalog(&diagram, &query, &proof, &catalog).unwrap();
    assert_eq!(cited(&bound), [0, 2, 3]);
    let family = preference_regime(9, "source", &[1], None, &[0, 2]);
    let catalog = preference_catalog(3, vec![observational, at(1, false), at(2, true), family]);
    let bound = bind_z_transport_catalog(&diagram, &query, &proof, &catalog).unwrap();
    assert_eq!(cited(&bound), [0, 9]);
}

/// The registered surrogate formula binds one source joint; two supplying
/// experiments bind the lowest id, whatever the catalog order.
#[test]
fn surrogate_formula_binds_the_lowest_of_two_supplying_experiments() {
    let (diagram, query) = surrogate_fixture();
    let ZTransportResult::Identified(proof) = identify_z_transport(
        &diagram,
        &query,
        super::super::SidLimits::default(),
        &antecedent_core::ExecutionContext::for_tests(0),
    )
    .unwrap() else {
        panic!("registered surrogate graph should be identified");
    };
    assert_eq!(proof.kind, ZFormulaKind::Surrogate);
    let experiment = |raw| preference_regime(raw, "source", &[1], Some(&[false]), &[0, 2, 3]);
    for order in [[3, 1], [1, 3]] {
        let catalog = preference_catalog(4, order.iter().map(|raw| experiment(*raw)).collect());
        let bound = bind_z_transport_catalog(&diagram, &query, &proof, &catalog).unwrap();
        assert_eq!(cited(&bound), [1], "catalog order {order:?}");
    }
}

/// The direct joint exchange binds one source joint; two supplying
/// experiments bind the lowest id, whatever the catalog order.
#[test]
fn direct_joint_formula_binds_the_lowest_of_two_supplying_experiments() {
    let mut graph = Admg::with_variables(3);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    let diagram = SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from([])).unwrap();
    let query = ZTransportQuery {
        outcomes: Arc::from([VariableId::from_raw(2)]),
        treatments: Arc::from([VariableId::from_raw(0), VariableId::from_raw(1)]),
        controllable: Arc::from([VariableId::from_raw(0), VariableId::from_raw(1)]),
        experiment_assignment: Arc::from([
            CatalogInterventionAssignment {
                variable: VariableId::from_raw(0),
                value: Value::Bool(true),
            },
            CatalogInterventionAssignment {
                variable: VariableId::from_raw(1),
                value: Value::Bool(false),
            },
        ]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    };
    let ZTransportResult::Identified(proof) = identify_z_transport(
        &diagram,
        &query,
        super::super::SidLimits::default(),
        &antecedent_core::ExecutionContext::for_tests(0),
    )
    .unwrap() else {
        panic!("direct joint source exchange must be identified")
    };
    assert_eq!(proof.kind, ZFormulaKind::DirectJoint);
    let experiment =
        |raw| preference_regime(raw, "source", &[0, 1], Some(&[true, false]), &[2]);
    for order in [[4, 2], [2, 4]] {
        let catalog = preference_catalog(3, order.iter().map(|raw| experiment(*raw)).collect());
        let bound = bind_z_transport_catalog(&diagram, &query, &proof, &catalog).unwrap();
        assert_eq!(cited(&bound), [2], "catalog order {order:?}");
    }
}

#[test]
fn bounded_search_reports_exhaustion_separately_from_noncertification() {
    let (diagram, query) = surrogate_fixture();
    let exhausted = identify_z_transport(
        &diagram,
        &query,
        super::super::SidLimits { steps: 0, depth: 1 },
        &antecedent_core::ExecutionContext::for_tests(1),
    )
    .unwrap_err();
    assert_eq!(exhausted.to_string(), "z_transport.exhausted_computation");
}

/// Dense slot of `variable` in the `Admg::with_variables(4)` fixture this SCM evaluates.
fn exchange_slot(variable: VariableId) -> usize {
    Admg::with_variables(4)
        .dense_of_static(variable)
        .unwrap_or_else(|| panic!("sequential exchange fixture has no node {}", variable.raw()))
        .as_usize()
}

struct SequentialExchangeScm;

#[allow(clippy::float_cmp, reason = "The provider accepts only binary values 0.0 and 1.0")]
impl DistributionProvider for SequentialExchangeScm {
    fn probability(
        &self,
        spec: &FactorSpec<'_>,
        assignment: &Assignment,
        _ctx: &EvalContext,
    ) -> Result<f64, EvalError> {
        let mut intervention = [None; 4];
        for item in spec.intervention {
            let value = item.value.as_f64().ok_or(EvalError::MissingBinding(item.variable))?;
            intervention[exchange_slot(item.variable)] = Some(usize::from(value == 1.0));
        }
        let mut denominator = 0.0;
        let mut numerator = 0.0;
        for exogenous in 0..16 {
            let a = exogenous & 1;
            let b = (exogenous >> 1) & 1;
            let noise_z = (exogenous >> 2) & 1;
            let noise_t = (exogenous >> 3) & 1;
            let x = intervention[0].unwrap_or(a ^ b);
            let t = intervention[1].unwrap_or(noise_t);
            let z = intervention[2].unwrap_or(x ^ a ^ noise_z);
            let y = intervention[3].unwrap_or(z ^ b);
            let state = [x, t, z, y];
            let matches = |variables: &[VariableId]| {
                variables.iter().all(|variable| {
                    assignment.get(*variable).and_then(Value::as_f64).is_some_and(|value| {
                        usize::from(value == 1.0) == state[exchange_slot(*variable)]
                    })
                })
            };
            if matches(spec.conditioned_on) {
                denominator += 1.0 / 16.0;
                if matches(spec.variables) {
                    numerator += 1.0 / 16.0;
                }
            }
        }
        if denominator == 0.0 {
            return Err(EvalError::DivisionByZero);
        }
        Ok(numerator / denominator)
    }

    fn support(
        &self,
        variables: &[VariableId],
        _ctx: &EvalContext,
    ) -> Result<Arc<[Arc<[Value]>]>, EvalError> {
        Ok((0..(1_usize << variables.len()))
            .map(|mask| {
                variables
                    .iter()
                    .enumerate()
                    .map(|(index, _)| Value::f64(((mask >> index) & 1) as f64))
                    .collect::<Arc<[_]>>()
            })
            .collect::<Vec<_>>()
            .into())
    }

    fn outcome(
        &self,
        variable: VariableId,
        assignment: &Assignment,
        _ctx: &EvalContext,
    ) -> Result<f64, EvalError> {
        assignment
            .get(variable)
            .and_then(Value::as_f64)
            .ok_or(EvalError::MissingBinding(variable))
    }

    fn n_draws(&self) -> Option<usize> {
        None
    }
}

#[test]
fn recursive_line10_keeps_remaining_controllables_and_matches_exact_scm() {
    // SCM: A and B are independent binary causes of X; A also causes Z,
    // B also causes Y, and the directed chain is X -> Z -> Y. T is
    // independent. The latent projection has X<->Z and X<->Y.
    let mut graph = Admg::with_variables(4);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(3)).unwrap();
    graph.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    graph.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(3)).unwrap();
    let diagram = SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from([])).unwrap();
    let query = ZTransportQuery {
        outcomes: Arc::from([VariableId::from_raw(3)]),
        treatments: Arc::from([VariableId::from_raw(0), VariableId::from_raw(1)]),
        controllable: Arc::from([VariableId::from_raw(0), VariableId::from_raw(2)]),
        experiment_assignment: Arc::from([
            CatalogInterventionAssignment {
                variable: VariableId::from_raw(0),
                value: Value::Bool(false),
            },
            CatalogInterventionAssignment {
                variable: VariableId::from_raw(2),
                value: Value::Bool(false),
            },
        ]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    };
    let Some((arena, root, trace)) = search_trz(
        &diagram,
        &query,
        super::super::SidLimits::default(),
        &antecedent_core::ExecutionContext::for_tests(0),
    )
    .unwrap() else {
        panic!("two-stage TRz exchange should identify this query");
    };
    assert_eq!(
        trace.iter().filter(|rule| rule.starts_with("ztr.line10.source_exchange")).count(),
        2,
        "the second controllable must remain available after the first exchange: {trace:?}"
    );

    let evaluator = arena.compile(root).unwrap();
    let provider = SequentialExchangeScm;
    for treatment in [0.0, 1.0] {
        for outcome in [0.0, 1.0] {
            let assignment = Assignment::from_pairs([
                (VariableId::from_raw(0), Value::f64(treatment)),
                (VariableId::from_raw(1), Value::f64(0.0)),
                (VariableId::from_raw(3), Value::f64(outcome)),
            ]);
            let actual = evaluator
                .evaluate_with(&arena, &provider, &EvalContext::default(), &assignment)
                .unwrap();
            // Under do(X=x,T=0), Z has fair A/E noise and Y adds
            // independent fair B, so P(Y=y)=1/2 for both levels.
            assert!((actual - 0.5).abs() < 1e-12, "x={treatment}, y={outcome}: {actual}");
        }
    }
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "one fixture exercised end to end; later assertions read locals built above them"
)]
fn recursive_search_handles_selected_mediator_surrogate_graph() {
    use antecedent_expr::{
        Assignment, DiscreteAxis, EvalContext, ExactDiscreteLaw, ExactTransportData,
        InterventionAssignment as ExprAssignment, LawTolerance,
    };
    // A source Z experiment combines with target W distribution. This is
    // structurally distinct from the registered W→Z surrogate graph.
    let mut graph = Admg::with_variables(4);
    for (a, b) in [(0, 1), (1, 2), (1, 3), (2, 3)] {
        graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    for (a, b) in [(0, 1), (0, 3)] {
        graph.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    let diagram =
        SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from([VariableId::from_raw(2)]))
            .unwrap();
    let query = ZTransportQuery {
        outcomes: Arc::from([VariableId::from_raw(3)]),
        treatments: Arc::from([VariableId::from_raw(1)]),
        controllable: Arc::from([VariableId::from_raw(0)]),
        experiment_assignment: Arc::from([CatalogInterventionAssignment {
            variable: VariableId::from_raw(0),
            value: Value::Bool(false),
        }]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    };
    let result = identify_z_transport(
        &diagram,
        &query,
        super::super::SidLimits::default(),
        &antecedent_core::ExecutionContext::for_tests(0),
    )
    .unwrap();
    let ZTransportResult::Identified(proof) = result else { panic!("{result:?}") };
    let coords = (0..4)
        .map(|raw| VariableCoordinate {
            variable: VariableId::from_raw(raw),
            domain: VariableDomain::Binary,
            unit: None,
        })
        .collect::<Vec<_>>();
    let source_env =
        Environment::try_new("source", coords.clone(), Arc::<[VariableId]>::from([])).unwrap();
    let target_env =
        Environment::try_new("target", coords, Arc::<[VariableId]>::from([])).unwrap();
    let target_regime = antecedent_core::EvidenceRegime::try_new(
        RegimeId::from_raw(0),
        RegimeKind::Observational,
        EvidenceKind::Available,
        [],
        [],
        (0..4).map(VariableId::from_raw).collect::<Vec<_>>(),
        "target",
        DistributionAvailability::Joint,
    )
    .unwrap();
    let source_regime = antecedent_core::EvidenceRegime::try_new(
        RegimeId::from_raw(1),
        RegimeKind::Experimental,
        EvidenceKind::Available,
        [VariableId::from_raw(0)],
        [CatalogInterventionAssignment {
            variable: VariableId::from_raw(0),
            value: Value::Bool(false),
        }],
        [VariableId::from_raw(1), VariableId::from_raw(2), VariableId::from_raw(3)],
        "source",
        DistributionAvailability::Joint,
    )
    .unwrap();
    let bindings = [0, 1].map(|raw| RegimeBinding {
        dataset_identity: None,
        regime: RegimeId::from_raw(raw),
        snapshot_identity: Arc::from(format!("selected-{raw}")),
        schema_names: Arc::from([]),
        sampling: SamplingDesign::Independent,
        weights: None,
        dependence: DependenceGroup::IndependentStudies,
    });
    let catalog = EvidenceCatalog::try_new(
        [source_env, target_env],
        [target_regime, source_regime],
        bindings,
        None,
    )
    .unwrap();
    let functional = bind_z_transport_catalog(&diagram, &query, &proof, &catalog).unwrap();
    let mut target = vec![0.0; 16];
    let mut source = vec![0.0; 8];
    let mass = |a: usize, b: usize, e: usize, p_e: f64| {
        (if a == 1 { 0.35 } else { 0.65 })
            * (if b == 1 { 0.20 } else { 0.80 })
            * (if e == 1 { p_e } else { 1.0 - p_e })
    };
    for a in 0..=1usize {
        for b in 0..=1usize {
            for e in 0..=1usize {
                let z = a ^ b;
                let x = z ^ a;
                let w = x ^ e;
                let y = x ^ w ^ b;
                target[z * 8 + x * 4 + w * 2 + y] += mass(a, b, e, 0.25);
                let sx = a;
                let sw = sx ^ e;
                let sy = sx ^ sw ^ b;
                source[sx * 4 + sw * 2 + sy] += mass(a, b, e, 0.65);
            }
        }
    }
    let axis = |raw| DiscreteAxis {
        variable: VariableId::from_raw(raw),
        values: Arc::from([Value::Bool(false), Value::Bool(true)]),
    };
    let target_law = ExactDiscreteLaw::try_new(
        "target",
        RegimeId::from_raw(0),
        [],
        [axis(0), axis(1), axis(2), axis(3)],
        target,
        "selected-0",
        LawTolerance::default(),
    )
    .unwrap();
    let source_law = ExactDiscreteLaw::try_new(
        "source",
        RegimeId::from_raw(1),
        [ExprAssignment::concrete(VariableId::from_raw(0), Value::Bool(false))],
        [axis(1), axis(2), axis(3)],
        source,
        "selected-1",
        LawTolerance::default(),
    )
    .unwrap();
    let data = ExactTransportData::try_new([target_law, source_law], 128).unwrap();
    for treatment in [false, true] {
        let request = Assignment::from_pairs([
            (VariableId::from_raw(1), Value::Bool(treatment)),
            (VariableId::from_raw(3), Value::Bool(true)),
        ]);
        let value = functional
            .arena()
            .compile(functional.root())
            .unwrap()
            .evaluate_with(functional.arena(), &data, &EvalContext::default(), &request)
            .unwrap();
        assert!((value - 0.35).abs() < 1e-12, "x={treatment}, formula={value}");
    }
}

#[test]
fn a_starved_search_reports_the_z_transport_budget_never_the_sid_step_budget() {
    // The five-node graph is not a registered surrogate, so every level
    // goes through the recursion whose step budget is the SID engine's.
    let mut graph = Admg::with_variables(5);
    for (a, b) in [(0, 1), (1, 2), (2, 3), (0, 3)] {
        graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    for (a, b) in [(0, 3), (1, 3), (1, 2)] {
        graph.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    let diagram = SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from([])).unwrap();
    let (_, query) = surrogate_fixture();
    let ctx = antecedent_core::ExecutionContext::for_tests(0);
    for steps in [1, 2, 3] {
        let error = identify_z_transport(
            &diagram,
            &query,
            super::super::SidLimits { steps, depth: 8 },
            &ctx,
        )
        .expect_err("a starved search has no result");
        assert!(
            matches!(
                error,
                IdentificationError::Budget { budget: IdentificationBudget::ZTransport }
            ),
            "steps={steps}: {error}"
        );
        assert_eq!(error.to_string(), "z_transport.exhausted_computation");
    }
    assert!(matches!(
        identify_z_transport(&diagram, &query, super::super::SidLimits::default(), &ctx)
            .unwrap(),
        ZTransportResult::Identified(_)
    ));
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "one fixture exercised under every family-validation failure"
)]
fn family_validation_requires_each_joint_intervention_law() {
    let (diagram, query) = surrogate_fixture();
    let coords = (0..4)
        .map(|raw| VariableCoordinate {
            variable: VariableId::from_raw(raw),
            domain: VariableDomain::Binary,
            unit: None,
        })
        .collect::<Vec<_>>();
    let environment =
        Environment::try_new("source", coords, Arc::<[VariableId]>::from([])).unwrap();
    let measured: Arc<[VariableId]> = (0..4).map(VariableId::from_raw).collect();
    let regimes = (0..2)
        .map(|level| {
            antecedent_core::EvidenceRegime::try_new(
                RegimeId::from_raw(level),
                RegimeKind::Experimental,
                EvidenceKind::Available,
                [VariableId::from_raw(1)],
                [CatalogInterventionAssignment {
                    variable: VariableId::from_raw(1),
                    value: Value::Bool(level == 1),
                }],
                Arc::clone(&measured),
                "source",
                DistributionAvailability::Joint,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let bindings = (0..2)
        .map(|level| RegimeBinding {
            dataset_identity: None,
            regime: RegimeId::from_raw(level),
            snapshot_identity: Arc::from(format!("source-do-z-{level}")),
            schema_names: Arc::from([]),
            sampling: SamplingDesign::Independent,
            weights: None,
            dependence: DependenceGroup::IndependentStudies,
        })
        .collect::<Vec<_>>();
    let catalog = EvidenceCatalog {
        environments: Arc::from([environment]),
        regimes: regimes.into(),
        bindings: bindings.into(),
        target_sampling: None,
    };
    assert_eq!(validate_z_experiment_family(&diagram, &query, &catalog).unwrap().len(), 2);

    // The positive formula uses only the selected do(Z=0) margin on
    // (W,X,Y). The unselected level and the measured Z coordinate are not
    // premises of this certificate.
    let ZTransportResult::Identified(proof) = identify_z_transport(
        &diagram,
        &query,
        super::super::SidLimits::default(),
        &antecedent_core::ExecutionContext::for_tests(0),
    )
    .unwrap() else {
        panic!()
    };
    let mut sufficient = catalog.clone();
    let mut regimes = sufficient.regimes.to_vec();
    regimes.truncate(1);
    regimes[0].measured =
        Arc::from([VariableId::from_raw(0), VariableId::from_raw(2), VariableId::from_raw(3)]);
    sufficient.regimes = regimes.into();
    sufficient.bindings = Arc::from([sufficient.bindings[0].clone()]);
    bind_z_transport_catalog(&diagram, &query, &proof, &sufficient).unwrap();
    let inspection = proof.inspect_proof(&sufficient);
    assert_eq!(inspection.factors.len(), 2);
    assert!(inspection.factors.iter().all(|factor| factor.failure.is_none()));
    assert!(inspection.factors.iter().all(|factor| factor.supplied_by == [0]));

    let mut missing_joint = catalog.clone();
    let mut altered = missing_joint.regimes.to_vec();
    altered.pop();
    missing_joint.regimes = altered.into();
    assert!(matches!(
        validate_z_experiment_family(&diagram, &query, &missing_joint),
        Err(ZExperimentFamilyError::MissingJointLaw { .. })
    ));
    let mut marginal_only = catalog.clone();
    let mut altered = marginal_only.regimes.to_vec();
    altered[0].distribution = DistributionAvailability::SeparateMarginals {
        variables: Arc::from([VariableId::from_raw(0), VariableId::from_raw(3)]),
    };
    marginal_only.regimes = altered.into();
    let inspection = proof.inspect_proof(&marginal_only);
    assert!(
        inspection
            .factors
            .iter()
            .any(|factor| factor.failure.as_deref() == Some("joint_law_required"))
    );
    assert!(matches!(
        validate_z_experiment_family(&diagram, &query, &marginal_only),
        Err(ZExperimentFamilyError::MissingJointLaw { .. })
    ));

    let mut insufficient_measurement = catalog;
    let mut altered = insufficient_measurement.regimes.to_vec();
    altered[0].measured =
        Arc::from([VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(3)]);
    insufficient_measurement.regimes = altered.into();
    assert!(matches!(
        validate_z_experiment_family(&diagram, &query, &insufficient_measurement),
        Err(ZExperimentFamilyError::MissingJointLaw { .. })
    ));
}
