//! 2.2B X10: decision, refusals, budget and verification of exact binary
//! observation recovery. Numerical truth lives in
//! `antecedent-estimate/tests/observation_recovery_execution.rs`.

#[path = "../../antecedent-estimate/tests/support/recovery_scm.rs"]
mod recovery_scm;

use std::sync::Arc;

use antecedent_core::{
    DistributionAvailability, EvidenceCatalog, EvidenceKind, EvidenceRegime, ExecutionContext,
    LawOrigin, RegimeId, RegimeKind, SamplingSelection, SearchLimits, SearchReceipt, SearchStop,
    VariableDomain,
};
use antecedent_expr::ExprNode;
use antecedent_graph::{Admg, DenseNodeId, NodeRef};
use antecedent_identify::{
    MixedSourceQuery, PartiallyObserved, RECOVERY_DEFAULT_LIMITS, RecoveredEffectQuery,
    RecoveryDecision, RecoveryDetail, RecoveryLimits, decide_observation_recovery,
    validate_mixed_source_query, verify_observation_recovery, verify_recovery_witness,
};
use recovery_scm::{MModel, POPULATION, v};

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

fn decide(model: &MModel) -> Result<RecoveryDecision, antecedent_identify::RecoveryError> {
    decide_observation_recovery(
        &model.graph,
        &model.query(),
        &model.catalog(),
        None,
        RecoveryLimits::default(),
        &ctx(),
    )
}

fn recovered(decision: RecoveryDecision) -> Box<antecedent_identify::RecoveryDerivation> {
    match decision {
        RecoveryDecision::Recovered(d) => d,
        RecoveryDecision::NonRecoverable(_) => panic!("expected a recovered derivation"),
    }
}

/// k = 2, m = 1: O -> X0, X0 -> X1, R0 <- O, R1 <- X0 (no self-censoring).
fn two_partial() -> MModel {
    MModel::new(2, 1, &[(2, 0), (0, 1)], &[vec![2], vec![0]], 11).unwrap()
}

#[test]
fn a_compliant_m_graph_is_recovered_with_margins_bound_to_the_named_distribution() {
    let model = two_partial();
    let derivation = recovered(decide(&model).unwrap());
    let record = derivation.record();
    assert_eq!(record.rule_version, "x10.recovery.v1");
    assert_eq!(record.factors.len(), 2);
    // R0's propensity conditions on O only; R1's on its partially observed parent
    // X0 through (R0, X*0), never on X0 itself.
    assert_eq!(record.factors[0].conditioning, vec![2]);
    assert_eq!(record.factors[1].conditioning, vec![3, 5]);
    // The root is complete case over a product of propensities, every leaf an
    // observational margin of the named regime.
    let arena = derivation.arena();
    assert!(matches!(arena.node(derivation.root()), ExprNode::Ratio { .. }));
    let leaves = arena.distribution_leaves(derivation.root());
    assert_eq!(leaves.len(), record.margins.len());
    let observed = derivation.observed();
    for margin in &record.margins {
        let ExprNode::Distribution { regime, population, .. } =
            arena.node(antecedent_expr::ExprId::from_raw(margin.expression))
        else {
            panic!("a margin is a leaf");
        };
        assert_eq!(*regime, Some(RegimeId::from_raw(7)));
        assert_eq!(arena.population(*population), POPULATION);
        assert!(margin.identity.starts_with("catalog_distribution.v1|source_regime=7"));
        if margin.variables.len() < observed.measured.len() {
            assert!(margin.identity.contains("projections=marginalize:"), "{}", margin.identity);
        }
    }
    assert!(record.premises.contains(&"no_self_censoring:4".to_owned()));
    // The recovered law's descriptor is derived with provenance, never measured.
    let descriptor = derivation.recovered_descriptor();
    assert!(
        matches!(&descriptor.origin, LawOrigin::Recovered { derivation: d } if d.as_ref() == derivation.identity())
    );
    assert_eq!(descriptor.measured.as_ref(), &[v(0), v(1), v(2)]);
    assert!(descriptor.canonical_identity().contains("origin=recovered:x10.recovery.v1"));
    assert!(record.receipt.operations_consumed > 0);
}

#[test]
fn self_censoring_is_nonrecoverable_with_a_verified_witness() {
    // X0 -> R0 (self-censoring), X1 fine.
    let model = MModel::new(2, 1, &[(2, 0), (0, 1)], &[vec![0, 2], vec![0]], 3).unwrap();
    let RecoveryDecision::NonRecoverable(witness) = decide(&model).unwrap() else {
        panic!("self-censoring is nonrecoverable");
    };
    assert_eq!(witness.edge, (0, 3));
    let check = verify_recovery_witness(&model.graph, &model.query(), &witness).unwrap();
    // Every observed-law cell agrees exactly (integer masses) and the targets differ.
    assert!(check.observed_cells >= 8, "{}", check.observed_cells);
    assert_ne!(check.masses.0, check.masses.1);
    assert_eq!(check.denominator, 60u128.pow(5));
    assert_eq!(
        antecedent_identify::RecoveryDetail::NonrecoverableWitness.reason_code(),
        "transport_proven_non_transportable"
    );
    assert_eq!(
        antecedent_identify::RecoveryDetail::NonrecoverableWitness.detail(),
        "recovery.nonrecoverable_witness"
    );
}

#[test]
fn a_tampered_witness_fails_the_exact_verifier() {
    let model = MModel::new(1, 0, &[], &[vec![0]], 5).unwrap();
    let RecoveryDecision::NonRecoverable(witness) = decide(&model).unwrap() else {
        panic!("self-censoring");
    };
    verify_recovery_witness(&model.graph, &model.query(), &witness).unwrap();
    // Observed laws no longer agree.
    let mut disagree = (*witness).clone();
    disagree.second.iter_mut().find(|m| m.node == 1).unwrap().numerators[1] = 44;
    // A non-positive model.
    let mut degenerate = (*witness).clone();
    degenerate.first[0].numerators[0] = 0;
    // A mechanism that is not Markov to the graph (drops the parent).
    let mut not_markov = (*witness).clone();
    let r = not_markov.first.iter_mut().find(|m| m.node == 1).unwrap();
    r.parents.clear();
    r.numerators = vec![30];
    // Two identical models: the target does not differ.
    let mut same = (*witness).clone();
    same.second = same.first.clone();
    for (tampered, why) in [
        (disagree, "disagree on the observed law"),
        (degenerate, "positivity"),
        (not_markov, "not Markov"),
        (same, "agree on the target"),
    ] {
        let error = verify_recovery_witness(&model.graph, &model.query(), &tampered).unwrap_err();
        assert_eq!(error.detail, RecoveryDetail::InvalidDerivation);
        assert!(error.message.contains(why), "{why}: {error}");
    }
}

#[test]
fn separate_marginals_or_a_missing_coordinate_refuse_as_a_missing_margin() {
    let model = two_partial();
    let observed = model.observed_variables();
    let q = model.query();
    let cases = [
        model.catalog_with(
            DistributionAvailability::SeparateMarginals {
                variables: observed.iter().map(|r| v(*r)).collect::<Vec<_>>().into(),
            },
            &observed,
        ),
        model.catalog_with(DistributionAvailability::Joint, &observed[1..]),
    ];
    for catalog in cases {
        let error = decide_observation_recovery(
            &model.graph,
            &q,
            &catalog,
            None,
            RecoveryLimits::default(),
            &ctx(),
        )
        .unwrap_err();
        assert_eq!(error.detail, RecoveryDetail::MissingMargin, "{error}");
        assert_eq!(error.reason_code(), "transport_missing_evidence");
    }
    // An unavailable regime, a model artifact, another population, an absent regime.
    let mut unavailable = model.catalog();
    Arc::make_mut(&mut unavailable.regimes)[0].evidence_kind = EvidenceKind::Proposed;
    unavailable.bindings = Arc::from([]);
    let mut artifact = model.catalog();
    Arc::make_mut(&mut artifact.regimes)[0].origin =
        LawOrigin::ModelArtifact { artifact: Arc::from("posterior") };
    let mut recovered_origin = model.catalog();
    Arc::make_mut(&mut recovered_origin.regimes)[0].origin =
        LawOrigin::Recovered { derivation: Arc::from("earlier") };
    let mut absent = q.clone();
    absent.observed_regime = RegimeId::from_raw(99);
    for (catalog, query) in [
        (unavailable, q.clone()),
        (artifact, q.clone()),
        (recovered_origin, q.clone()),
        (model.catalog(), absent),
    ] {
        let error = decide_observation_recovery(
            &model.graph,
            &query,
            &catalog,
            None,
            RecoveryLimits::default(),
            &ctx(),
        )
        .unwrap_err();
        assert_eq!(error.detail, RecoveryDetail::MissingMargin, "{error}");
    }
}

#[test]
fn each_unsupported_mechanism_is_refused() {
    let model = two_partial();
    let q = model.query();
    let d = DenseNodeId::from_raw;
    let check = |graph: &Admg, catalog: &antecedent_core::EvidenceCatalog, why: &str| {
        let error = decide_observation_recovery(
            graph,
            &q,
            catalog,
            None,
            RecoveryLimits::default(),
            &ctx(),
        )
        .unwrap_err();
        assert_eq!(error.detail, RecoveryDetail::UnsupportedMechanism, "{why}: {error}");
        assert_eq!(error.reason_code(), "route_not_supported");
    };
    // Bidirected edge (unmeasured confounding).
    let mut bidirected = model.graph.clone();
    bidirected.insert_bidirected(d(0), d(1)).unwrap();
    check(&bidirected, &model.catalog(), "bidirected");
    // R -> R edge (colluder shape).
    let mut rr = model.graph.clone();
    rr.insert_directed(d(3), d(4)).unwrap();
    check(&rr, &model.catalog(), "R -> R");
    // Response causes a substantive variable.
    let mut r_to_x = model.graph.clone();
    r_to_x.insert_directed(d(3), d(1)).unwrap();
    check(&r_to_x, &model.catalog(), "R -> X");
    // Proxy noise: an O parent of a proxy.
    let mut noisy = model.graph.clone();
    noisy.insert_directed(d(2), d(5)).unwrap();
    check(&noisy, &model.catalog(), "proxy noise");
    // A proxy with a child.
    let mut proxy_child = model.graph.clone();
    proxy_child.insert_directed(d(5), d(4)).unwrap();
    check(&proxy_child, &model.catalog(), "proxy child");
    // Non-binary variable.
    let mut non_binary = model.catalog();
    let env = &mut Arc::make_mut(&mut non_binary.environments)[0];
    let mut coordinates = env.variables.to_vec();
    coordinates[1].domain = VariableDomain::Categorical { cardinality: 4 };
    env.variables = coordinates.into();
    check(&model.graph, &non_binary, "non-binary");
    // Selection: a selected-sample law and a selection target.
    let mut selected = model.catalog();
    Arc::make_mut(&mut selected.regimes)[0].selection =
        SamplingSelection::SelectedOn { variables: Arc::from([v(2)]) };
    check(&model.graph, &selected, "selected sample");
    let mut selection_node = model.catalog();
    Arc::make_mut(&mut selection_node.environments)[0].selection_targets = Arc::from([v(2)]);
    check(&model.graph, &selection_node, "selection node");
}

#[test]
fn bounds_refuse_at_the_cap_plus_one() {
    // Three partially observed and two fully observed are accepted.
    let at_cap = MModel::new(3, 2, &[(3, 0), (4, 1)], &[vec![3], vec![4], vec![0]], 2).unwrap();
    assert!(matches!(decide(&at_cap).unwrap(), RecoveryDecision::Recovered(_)));
    for (k, m) in [(4, 0), (1, 3)] {
        let over = MModel::new(k, m, &[], &vec![vec![]; k as usize], 2).unwrap();
        let error = decide(&over).unwrap_err();
        assert_eq!(error.detail, RecoveryDetail::BoundsExceeded, "k={k} m={m}");
        assert_eq!(error.reason_code(), "route_not_supported");
    }
    let model = two_partial();
    for limits in [
        RecoveryLimits {
            search: SearchLimits { operations: 50_001, depth: 32 },
            ..RecoveryLimits::default()
        },
        RecoveryLimits {
            search: SearchLimits { operations: 50_000, depth: 33 },
            ..RecoveryLimits::default()
        },
    ] {
        let error = decide_observation_recovery(
            &model.graph,
            &model.query(),
            &model.catalog(),
            None,
            limits,
            &ctx(),
        )
        .unwrap_err();
        assert_eq!(error.detail, RecoveryDetail::BoundsExceeded);
    }
    assert_eq!(RECOVERY_DEFAULT_LIMITS.search, SearchLimits { operations: 50_000, depth: 32 });
}

#[test]
fn malformed_queries_are_invalid_not_unrecoverable() {
    let model = two_partial();
    let mut duplicate = model.query();
    duplicate.fully_observed = Arc::from([v(3)]);
    let mut empty_population = model.query();
    empty_population.population = Arc::from(" ");
    let mut unknown = model.query();
    unknown.fully_observed = Arc::from([v(40)]);
    let mut none = model.query();
    none.partially_observed = Arc::from([]);
    for query in [duplicate, empty_population, unknown, none] {
        let error = decide_observation_recovery(
            &model.graph,
            &query,
            &model.catalog(),
            None,
            RecoveryLimits::default(),
            &ctx(),
        )
        .unwrap_err();
        assert_eq!(error.detail, RecoveryDetail::InvalidQuery, "{error}");
        assert_eq!(error.reason_code(), "invalid_argument");
    }
    // A graph node without a role.
    let mut extra = model.graph.clone();
    extra.add_node(NodeRef::Static(v(9))).unwrap();
    let error = decide_observation_recovery(
        &extra,
        &model.query(),
        &model.catalog(),
        None,
        RecoveryLimits::default(),
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(error.detail, RecoveryDetail::InvalidQuery);
}

#[test]
fn permutations_leave_the_decision_unchanged_and_proxy_relabelling_is_refused() {
    let model = two_partial();
    let reference = recovered(decide(&model).unwrap());
    // Declaration order of the roles does not matter.
    let mut reversed = model.query();
    reversed.partially_observed =
        reversed.partially_observed.iter().rev().copied().collect::<Vec<_>>().into();
    let again = recovered(
        decide_observation_recovery(
            &model.graph,
            &reversed,
            &model.catalog(),
            None,
            RecoveryLimits::default(),
            &ctx(),
        )
        .unwrap(),
    );
    assert_eq!(again.record(), reference.record());
    assert_eq!(again.identity(), reference.identity());
    // Proxy relabelling: declaring X*1 as the proxy of X0 (and X*0 of X1) is refused,
    // never silently aligned.
    let mut swapped = model.query();
    swapped.partially_observed = Arc::from([
        PartiallyObserved { variable: v(0), response: v(3), proxy: v(6) },
        PartiallyObserved { variable: v(1), response: v(4), proxy: v(5) },
    ]);
    let error = decide_observation_recovery(
        &model.graph,
        &swapped,
        &model.catalog(),
        None,
        RecoveryLimits::default(),
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(error.detail, RecoveryDetail::InvalidQuery);
    assert!(error.message.contains("relabelled"), "{error}");
    // A proxy that also depends on another partially observed variable is wired to
    // a foreign variable: refused as relabelled, not as proxy noise.
    let mut foreign = model.graph.clone();
    foreign.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(5)).unwrap();
    let error = decide_observation_recovery(
        &foreign,
        &model.query(),
        &model.catalog(),
        None,
        RecoveryLimits::default(),
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(error.detail, RecoveryDetail::InvalidQuery);
    assert!(error.message.contains("relabelled"), "{error}");
    // Responses swapped between variables is refused the same way.
    let mut responses = model.query();
    responses.partially_observed = Arc::from([
        PartiallyObserved { variable: v(0), response: v(4), proxy: v(5) },
        PartiallyObserved { variable: v(1), response: v(3), proxy: v(6) },
    ]);
    let error = decide_observation_recovery(
        &model.graph,
        &responses,
        &model.catalog(),
        None,
        RecoveryLimits::default(),
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(error.detail, RecoveryDetail::InvalidQuery);
}

#[test]
fn exhausted_budget_returns_a_receipt_never_a_verdict() {
    let model = two_partial();
    let full = recovered(decide(&model).unwrap()).record().receipt.operations_consumed;
    let limits = RecoveryLimits {
        search: SearchLimits { operations: full - 1, depth: 32 },
        ..RecoveryLimits::default()
    };
    let error = decide_observation_recovery(
        &model.graph,
        &model.query(),
        &model.catalog(),
        None,
        limits,
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(error.detail, RecoveryDetail::Budget);
    assert_eq!(error.reason_code(), "transport_budget_cancel");
    let receipt = error.receipt.unwrap();
    assert_eq!(receipt.stop, SearchStop::Operations);
    assert_eq!(receipt.operations_limit, full - 1);
    assert_eq!(receipt.operations_consumed, Some(full - 1));
    assert!(!receipt.unevaluated.is_empty());
    // The witness enumeration is charged too: a self-censoring graph under a tiny
    // budget stops with a receipt, never a nonrecoverability claim.
    let censored = MModel::new(2, 1, &[(2, 0)], &[vec![0], vec![2]], 1).unwrap();
    let tiny = RecoveryLimits {
        search: SearchLimits { operations: 12, depth: 32 },
        ..RecoveryLimits::default()
    };
    let error = decide_observation_recovery(
        &censored.graph,
        &censored.query(),
        &censored.catalog(),
        None,
        tiny,
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(error.receipt.unwrap().stop, SearchStop::Operations);
}

#[test]
fn memory_and_cancellation_are_observed_by_the_shared_budget() {
    let model = two_partial();
    let small = RecoveryLimits { memory_bytes: 64, ..RecoveryLimits::default() };
    let error = decide_observation_recovery(
        &model.graph,
        &model.query(),
        &model.catalog(),
        None,
        small,
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(error.detail, RecoveryDetail::Budget);
    let receipt = error.receipt.unwrap();
    assert_eq!(receipt.stop, SearchStop::Memory);
    assert_eq!(receipt.memory_limit_bytes, Some(64));
    // The context's hard limit also bounds the cap.
    let mut limited = ctx();
    limited.memory.hard_limit_bytes = Some(100);
    let error = decide_observation_recovery(
        &model.graph,
        &model.query(),
        &model.catalog(),
        None,
        RecoveryLimits::default(),
        &limited,
    )
    .unwrap_err();
    assert_eq!(error.receipt.unwrap().stop, SearchStop::Memory);
    let cancelled = ctx();
    cancelled.cancellation.cancel();
    let error = decide_observation_recovery(
        &model.graph,
        &model.query(),
        &model.catalog(),
        None,
        RecoveryLimits::default(),
        &cancelled,
    )
    .unwrap_err();
    assert_eq!(error.receipt.unwrap().stop, SearchStop::Cancelled);
}

fn effect_graph(model: &MModel, edges: &[(u32, u32)]) -> Admg {
    let mut graph = Admg::empty();
    for n in 0..model.k + model.m {
        graph.add_node(NodeRef::Static(v(n))).unwrap();
    }
    for (a, b) in edges {
        graph.insert_directed(DenseNodeId::from_raw(*a), DenseNodeId::from_raw(*b)).unwrap();
    }
    graph
}

#[test]
fn the_effect_is_identified_by_ordinary_target_id_on_the_causal_restriction() {
    let model = two_partial();
    let effect = RecoveredEffectQuery {
        graph: effect_graph(&model, &[(2, 0), (0, 1)]),
        outcomes: Arc::from([v(1)]),
        treatments: Arc::from([v(0)]),
    };
    let derivation = recovered(
        decide_observation_recovery(
            &model.graph,
            &model.query(),
            &model.catalog(),
            Some(&effect),
            RecoveryLimits::default(),
            &ctx(),
        )
        .unwrap(),
    );
    let identified = derivation.effect().unwrap();
    for leaf in identified.arena().distribution_leaves(identified.root()) {
        let ExprNode::Distribution { population, regime, .. } = identified.arena().node(leaf)
        else {
            panic!()
        };
        assert_eq!(identified.arena().population(*population), POPULATION);
        assert_eq!(*regime, None);
    }
    assert!(!derivation.record().effect.as_ref().unwrap().rules.is_empty());
    // Handoff mismatches: another factorization, a bidirected edge, other variables.
    let dropped = RecoveredEffectQuery { graph: effect_graph(&model, &[(0, 1)]), ..effect.clone() };
    let mut confounded_graph = effect_graph(&model, &[(2, 0), (0, 1)]);
    confounded_graph.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let confounded = RecoveredEffectQuery { graph: confounded_graph, ..effect.clone() };
    let mut with_response = effect_graph(&model, &[(2, 0), (0, 1)]);
    with_response.add_node(NodeRef::Static(v(3))).unwrap();
    let foreign = RecoveredEffectQuery { graph: with_response, ..effect.clone() };
    let response_outcome = RecoveredEffectQuery { outcomes: Arc::from([v(3)]), ..effect.clone() };
    for bad in [dropped, confounded, foreign, response_outcome] {
        let error = decide_observation_recovery(
            &model.graph,
            &model.query(),
            &model.catalog(),
            Some(&bad),
            RecoveryLimits::default(),
            &ctx(),
        )
        .unwrap_err();
        assert_eq!(error.detail, RecoveryDetail::HandoffMismatch, "{error}");
        assert_eq!(error.reason_code(), "invalid_argument");
    }
}

#[test]
fn a_tampered_record_or_arena_fails_verification() {
    let model = two_partial();
    let effect = RecoveredEffectQuery {
        graph: effect_graph(&model, &[(2, 0), (0, 1)]),
        outcomes: Arc::from([v(1)]),
        treatments: Arc::from([v(0)]),
    };
    let q = model.query();
    let c = model.catalog();
    let d = recovered(
        decide_observation_recovery(
            &model.graph,
            &q,
            &c,
            Some(&effect),
            RecoveryLimits::default(),
            &ctx(),
        )
        .unwrap(),
    );
    let e_arena = d.effect().unwrap().arena().clone();
    let verified = verify_observation_recovery(
        &model.graph,
        &q,
        &c,
        Some(&effect),
        d.record(),
        d.arena(),
        Some(&e_arena),
        RECOVERY_DEFAULT_LIMITS,
        &ctx(),
    )
    .unwrap();
    assert_eq!(verified.identity(), d.identity());
    let verify = |record: &antecedent_identify::RecoveryDerivationRecord,
                  arena: &antecedent_expr::CausalExprArena,
                  effect_arena: Option<&antecedent_expr::CausalExprArena>| {
        verify_observation_recovery(
            &model.graph,
            &q,
            &c,
            Some(&effect),
            record,
            arena,
            effect_arena,
            RECOVERY_DEFAULT_LIMITS,
            &ctx(),
        )
        .unwrap_err()
    };
    // A margin identity edited.
    let mut margin = d.record().clone();
    margin.margins[0].identity.push('x');
    assert_eq!(
        verify(&margin, d.arena(), Some(&e_arena)).detail,
        RecoveryDetail::InvalidDerivation
    );
    // A premise edited.
    let mut premise = d.record().clone();
    premise.premises.push("mar".to_owned());
    assert_eq!(
        verify(&premise, d.arena(), Some(&e_arena)).detail,
        RecoveryDetail::InvalidDerivation
    );
    // The receipt edited.
    let mut receipt = d.record().clone();
    receipt.receipt.operations_consumed += 1;
    assert_eq!(
        verify(&receipt, d.arena(), Some(&e_arena)).detail,
        RecoveryDetail::InvalidDerivation
    );
    // The effect arena dropped.
    assert_eq!(verify(d.record(), d.arena(), None).detail, RecoveryDetail::InvalidDerivation);
    // Stored limits above the consumer's maxima refuse before any work.
    let mut limits = d.record().clone();
    limits.receipt.operations_limit = 60_000;
    let error = verify_observation_recovery(
        &model.graph,
        &q,
        &c,
        Some(&effect),
        &limits,
        d.arena(),
        Some(&e_arena),
        RECOVERY_DEFAULT_LIMITS,
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(error.detail, RecoveryDetail::BoundsExceeded);
    // Another catalog snapshot changes the margin identities: refused.
    let mut other = c.clone();
    Arc::make_mut(&mut other.bindings)[0].snapshot_identity = Arc::from("snap-other");
    let error = verify_observation_recovery(
        &model.graph,
        &q,
        &other,
        Some(&effect),
        d.record(),
        d.arena(),
        Some(&e_arena),
        RECOVERY_DEFAULT_LIMITS,
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(error.detail, RecoveryDetail::InvalidDerivation);
}

#[test]
fn the_colluder_archetype_is_refused_as_unsupported_not_as_nonrecoverable() {
    // X0 -> R1 <- R0: a colluder. The class excludes R -> R edges, so the decision
    // refuses the mechanism; it never claims nonrecoverability (for the target
    // P(X) the archetype is recoverable, see the execution test of the same name).
    let mut model = MModel::new(2, 0, &[], &[vec![], vec![0]], 4).unwrap();
    model.graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(3)).unwrap();
    let error = decide(&model).unwrap_err();
    assert_eq!(error.detail, RecoveryDetail::UnsupportedMechanism);
    assert!(error.message.contains("R -> R"));
    let _ = v(0);
}

/// Every budget stop of `run` for operation limits `1..` until it decides:
/// `(explored, unevaluated)` of each receipt, in limit order.
fn stops_by_limit(
    run: impl Fn(RecoveryLimits) -> Result<RecoveryDecision, antecedent_identify::RecoveryError>,
) -> Vec<(Vec<String>, Vec<String>)> {
    let mut out = Vec::new();
    for limit in 1..=RECOVERY_DEFAULT_LIMITS.search.operations {
        let limits = RecoveryLimits {
            search: SearchLimits { operations: limit, depth: 32 },
            ..RecoveryLimits::default()
        };
        match run(limits) {
            Ok(_) => return out,
            Err(error) => {
                assert_eq!(error.detail, RecoveryDetail::Budget, "{error}");
                let SearchReceipt { stop, explored, unevaluated, .. } = *error.receipt.unwrap();
                assert_eq!(stop, SearchStop::Operations);
                out.push((explored, unevaluated));
            }
        }
    }
    panic!("the decision never completed under the maximum budget");
}

fn names(raw: &[&str]) -> Vec<String> {
    raw.iter().map(|s| (*s).to_owned()).collect()
}

#[test]
fn budget_receipts_name_the_stage_the_stop_reached() {
    // Recoverable, with a downstream effect: stops walk class check -> formula ->
    // downstream effect, and each receipt names exactly the stage reached.
    let model = two_partial();
    let effect = RecoveredEffectQuery {
        graph: effect_graph(&model, &[(2, 0), (0, 1)]),
        outcomes: Arc::from([v(1)]),
        treatments: Arc::from([v(0)]),
    };
    let stops = stops_by_limit(|limits| {
        decide_observation_recovery(
            &model.graph,
            &model.query(),
            &model.catalog(),
            Some(&effect),
            limits,
            &ctx(),
        )
    });
    let class = (vec![], names(&["class_check", "formula_or_witness", "downstream_effect"]));
    let formula = (names(&["class_check"]), names(&["formula", "downstream_effect"]));
    let downstream = (names(&["class_check", "formula"]), names(&["downstream_effect"]));
    for stage in [&class, &formula, &downstream] {
        assert!(stops.contains(stage), "no stop in {stage:?}: {stops:?}");
    }
    // Monotone: once a later stage is reached, no earlier stage is reported.
    let rank = |s: &(Vec<String>, Vec<String>)| {
        [&class, &formula, &downstream].iter().position(|x| *x == s).expect("a known stage")
    };
    assert!(stops.windows(2).all(|w| rank(&w[0]) <= rank(&w[1])), "{stops:?}");
    // The last stops are inside the downstream effect stage.
    assert_eq!(stops.last(), Some(&downstream));

    // Self-censoring: stops walk class check -> witness; the witness stage is
    // named, and no effect stage follows a nonrecoverable decision.
    let censored = MModel::new(2, 1, &[(2, 0)], &[vec![0], vec![2]], 1).unwrap();
    let stops = stops_by_limit(|limits| {
        decide_observation_recovery(
            &censored.graph,
            &censored.query(),
            &censored.catalog(),
            None,
            limits,
            &ctx(),
        )
    });
    let class = (vec![], names(&["class_check", "formula_or_witness"]));
    let witness = (names(&["class_check"]), names(&["witness"]));
    assert!(stops.contains(&class), "{stops:?}");
    assert!(stops.contains(&witness), "{stops:?}");
    assert!(stops.iter().all(|s| *s == class || *s == witness), "{stops:?}");
    assert_eq!(stops.last(), Some(&witness));
}

#[test]
fn a_recovered_regime_is_excluded_from_the_mixed_source_search_as_a_recovered_law() {
    // X -> Z -> Y in the target population: a measured {X, Z} law is an input; a
    // recovered law (observational or experimental) is excluded under its own
    // reason, never mislabelled a model artifact or refused as a posterior.
    let mut graph = Admg::with_variables(3);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    let regime = |id: u32, on: &[u32], measured: &[u32], origin: LawOrigin| {
        let mut regime = EvidenceRegime::try_new(
            RegimeId::from_raw(id),
            if on.is_empty() { RegimeKind::Observational } else { RegimeKind::Experimental },
            EvidenceKind::Available,
            on.iter().copied().map(v).collect::<Vec<_>>(),
            [],
            measured.iter().copied().map(v).collect::<Vec<_>>(),
            "target",
            DistributionAvailability::Joint,
        )
        .unwrap();
        regime.origin = origin;
        regime
    };
    let recovered = || LawOrigin::Recovered { derivation: Arc::from("x10.recovery.v1|d") };
    let catalog = EvidenceCatalog::try_new(
        [],
        [
            regime(1, &[], &[0, 1], LawOrigin::Measured),
            regime(2, &[], &[1, 2], recovered()),
            regime(3, &[1], &[0, 2], recovered()),
            regime(4, &[], &[1, 2], LawOrigin::ModelArtifact { artifact: Arc::from("fit") }),
        ],
        [],
        None,
    )
    .unwrap();
    let query = MixedSourceQuery {
        outcomes: Arc::from([v(2)]),
        treatments: Arc::from([v(0)]),
        target: Arc::from("target"),
        sources: Arc::from([]),
    };
    let validated = validate_mixed_source_query(&graph, &query, &catalog).unwrap();
    assert_eq!(
        validated.inputs.iter().map(|i| i.regime.raw()).collect::<Vec<_>>(),
        vec![1],
        "only the measured law is an input"
    );
    let reasons =
        validated.exclusions.iter().map(|e| (e.regime.raw(), e.reason)).collect::<Vec<_>>();
    assert_eq!(reasons, vec![(2, "recovered_law"), (3, "recovered_law"), (4, "model_artifact")]);
}

/// What a user sees of a budget stop (the error text a binding surfaces) names
/// the stop and the regions left unevaluated, never only the successes.
#[test]
fn a_budget_stop_displays_its_receipt() {
    let model = two_partial();
    let full = recovered(decide(&model).unwrap()).record().receipt.operations_consumed;
    let limits = RecoveryLimits {
        search: SearchLimits { operations: full - 1, depth: 32 },
        ..RecoveryLimits::default()
    };
    let error = decide_observation_recovery(
        &model.graph,
        &model.query(),
        &model.catalog(),
        None,
        limits,
        &ctx(),
    )
    .unwrap_err();
    let receipt = error.receipt.clone().unwrap();
    let text = error.to_string();
    assert!(text.contains("search.operations"), "{text}");
    assert!(receipt.unevaluated.iter().all(|r| text.contains(r.as_str())), "{text}");
}
