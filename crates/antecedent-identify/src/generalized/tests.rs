// Completion weights here are exact counts of unit-weight cases.
#![cfg_attr(
    test,
    allow(
        clippy::float_cmp,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]
use super::*;
use crate::result::IdentificationStatus;
use antecedent_graph::Pag;

#[test]
fn default_max_candidates_matches_dag_backdoor() {
    assert_eq!(GeneralizedAdjustmentConfig::default().max_candidates, usize::MAX);
    assert_eq!(
        GeneralizedAdjustmentConfig::default().max_candidates,
        crate::backdoor::AdjustmentSearchConfig::default().max_candidates
    );
}

#[test]
fn conditional_modifier_cannot_be_a_mediator() {
    let mut cpdag = Cpdag::with_variables(3);
    let t = DenseNodeId::from_raw(0);
    let w = DenseNodeId::from_raw(1);
    let y = DenseNodeId::from_raw(2);
    cpdag.insert_directed(t, w).unwrap();
    cpdag.insert_directed(w, y).unwrap();
    let id = GeneralizedAdjustmentIdentifier::new();
    let ate = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(2));
    assert_eq!(id.identify_cpdag_envelope(&cpdag, &ate).unwrap().identified_weight.0, 1.0);
    let conditional = ate.with_effect_modifiers([VariableId::from_raw(1)]);
    let env = id.identify_cpdag_envelope(&cpdag, &conditional).unwrap();
    assert_eq!(env.identified_weight.0, 0.0);
    assert_eq!(env.unidentified_weight.0, 1.0);
    assert!(
        env.cases[0]
            .result
            .diagnostics
            .iter()
            .any(|d| d.code.as_ref() == "identify.conditional.adjustment_unverified")
    );
}

#[test]
fn pretreatment_modifier_can_open_a_collider_path() {
    // T <- A -> W <- B -> Y, T -> Y. Empty Z identifies ATE,
    // Conditioning on pre-treatment W opens the noncausal path; adding A closes it.
    let mut pag = Pag::with_variables(5);
    for (a, b) in [(1, 0), (1, 2), (3, 2), (3, 4), (0, 4)] {
        pag.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    let id = GeneralizedAdjustmentIdentifier::new();
    let ate = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(4));
    assert!(id.identify_pag_envelope(&pag, &ate).unwrap().identified_weight.0 > 0.0);
    let conditional = ate.with_effect_modifiers([VariableId::from_raw(2)]);
    let env = id.identify_pag_envelope(&pag, &conditional).unwrap();
    assert!(env.identified_weight.0 > 0.0);
    assert_eq!(env.unidentified_weight.0, 0.0);
    assert_eq!(
        env.cases[0].result.estimands[0].adjustment_set.as_ref(),
        &[VariableId::from_raw(1)]
    );
}

/// `T <- A -> W <- B -> Y`, `T -> Y`, conditioning on the pre-treatment collider `W`:
/// the empty set fails and `{A}` and `{B}` succeed. A budget of two separation tests
/// is spent on the ancestral set and the empty set, so the search for smaller sets stops
/// before it reaches `{A}`; the ancestral set `{A, B}` was already tested valid, so the
/// query is identified with it and the unfinished search is marked, not refuted.
#[test]
fn conditional_maximal_search_charges_pruned_subsets() {
    let mut dag = antecedent_graph::Dag::with_variables(5);
    for (a, b) in [(1, 0), (1, 2), (3, 2), (3, 4), (0, 4)] {
        dag.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    let graph = crate::prepared::dag_to_admg(&dag).unwrap();
    let cut = mutilate_outgoing(&graph, DenseNodeId::from_raw(0));
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(4))
        .with_effect_modifiers([VariableId::from_raw(2)]);
    let config = crate::backdoor::AdjustmentSearchConfig {
        maximal_only: true,
        minimal_only: false,
        max_examinations: 2,
        ..Default::default()
    };
    let result = constrained_conditional_set(
        &graph,
        &cut,
        dag.nodes(),
        &query,
        &[DenseNodeId::from_raw(2)],
        &config,
    )
    .unwrap();
    match result {
        ConditionalSearch::Found(sets, examined, bounded, _) => {
            assert_eq!(sets.len(), 1);
            assert_eq!(examined, 2);
            assert!(bounded);
        }
        ConditionalSearch::Absent => panic!("ancestral set is valid"),
    }
}

#[test]
fn conditional_search_budget_exit_keeps_the_certified_ancestral_set() {
    let mut dag = antecedent_graph::Dag::with_variables(5);
    for (a, b) in [(1, 0), (1, 2), (3, 2), (3, 4), (0, 4)] {
        dag.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    let query = CausalQuery::AverageEffect(
        AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(4))
            .with_effect_modifiers([VariableId::from_raw(2)]),
    );
    let run = |max_examinations: u64| {
        let id = crate::BackdoorIdentifier {
            config: crate::backdoor::AdjustmentSearchConfig {
                max_examinations,
                ..Default::default()
            },
        };
        let prepared = id.prepare(&dag).unwrap();
        crate::Identifier::identify(
            &id,
            &prepared,
            &query,
            &mut crate::IdentificationWorkspace::default(),
        )
        .unwrap()
    };

    let bounded = run(2);
    assert_eq!(bounded.status, IdentificationStatus::NonparametricallyIdentified);
    assert_eq!(
        bounded.estimands[0].adjustment_set.as_ref(),
        &[VariableId::from_raw(1), VariableId::from_raw(3)]
    );
    let kinds: Vec<_> =
        bounded.diagnostics.iter().map(|d| (d.code.as_ref().to_owned(), d.kind)).collect();
    assert_eq!(
        kinds,
        vec![(CONDITIONAL_SEARCH_BOUNDED_DIAGNOSTIC_CODE.to_owned(), DiagnosticKind::Execution)],
        "an unfinished search carries no scientific negative"
    );

    let full = run(1_000_000);
    assert_eq!(full.status, IdentificationStatus::NonparametricallyIdentified);
    assert_eq!(full.estimands[0].adjustment_set.as_ref(), &[VariableId::from_raw(1)]);
    assert!(full.diagnostics.is_empty());
}

/// Same graph with both `A` and `B` forbidden: conditioning on the collider `W` opens
/// `T <- A -> W <- B -> Y` and nothing admissible can close it. One separation test on
/// the ancestral conditioning set proves that; it is not a budget miss.
#[test]
fn conditional_adjustment_with_no_admissible_set_is_absent_not_bounded() {
    let mut dag = antecedent_graph::Dag::with_variables(5);
    for (a, b) in [(1, 0), (1, 2), (3, 2), (3, 4), (0, 4)] {
        dag.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    let query = CausalQuery::AverageEffect(
        AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(4))
            .with_effect_modifiers([VariableId::from_raw(2)]),
    );
    let id = crate::BackdoorIdentifier {
        config: crate::backdoor::AdjustmentSearchConfig {
            max_examinations: 2,
            forbidden: Arc::from([VariableId::from_raw(1), VariableId::from_raw(3)]),
            ..Default::default()
        },
    };
    let prepared = id.prepare(&dag).unwrap();
    let result = crate::Identifier::identify(
        &id,
        &prepared,
        &query,
        &mut crate::IdentificationWorkspace::default(),
    )
    .unwrap();
    assert_eq!(result.status, IdentificationStatus::NotIdentified);
    assert!(
        !result.diagnostics.iter().any(|d| d.kind == DiagnosticKind::Execution),
        "{:?}",
        result.diagnostics
    );
}

/// `0 -> 1` beside an all-circle K4 on `{2, 3, 4, 5}`: the effect is unconfounded and so
/// identified in every completion, which leaves a partial enumeration as the only reason
/// to withdraw the class-wide claim. The K4 has 12 circle endpoints and 219 members
/// (the ancestral complete graphs on four nodes).
fn effect_beside_circle_clique() -> Pag {
    let mut pag = Pag::with_variables(6);
    pag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    for a in 2..6 {
        for b in (a + 1)..6 {
            pag.insert_circle_circle(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
    }
    pag
}

fn audit_withdrawn(env: &IdentificationEnvelope<Pag>) -> Option<String> {
    env.critical_graph_features
        .iter()
        .find(|f| &*f.kind == "completion_equivalence_audit_skipped")
        .map(|f| f.detail.to_string())
}

#[test]
fn a_completed_completion_search_keeps_the_class_wide_claim() {
    let pag = effect_beside_circle_clique();
    let id = GeneralizedAdjustmentIdentifier {
        config: GeneralizedAdjustmentConfig {
            max_completions: 1_000,
            completion_operations: 10_000_000,
            ..Default::default()
        },
    };
    let q = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let env = id.identify_pag_envelope(&pag, &q).unwrap();
    assert_eq!(env.cases.len(), 219);
    assert!(audit_withdrawn(&env).is_none(), "a completed search withdraws nothing");
    assert!(
        env.critical_graph_features.iter().all(|f| &*f.kind != "completion_enumeration_capped")
    );
}

#[test]
fn a_partial_completion_search_refuses_the_class_wide_claim() {
    let pag = effect_beside_circle_clique();
    // One audited signature costs 6 nodes: 15 pairs x 2^4 subsets = 240 operations, so
    // 400 operations classify one member and stop during the next.
    let id = GeneralizedAdjustmentIdentifier {
        config: GeneralizedAdjustmentConfig {
            max_completions: 1_000,
            completion_operations: 400,
            ..Default::default()
        },
    };
    let q = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let env = id.identify_pag_envelope(&pag, &q).unwrap();
    assert!(env.cases.len() < 219);
    assert_ne!(
        env.status,
        IdentificationStatus::NonparametricallyIdentified,
        "a prefix of the class cannot support identification in every member"
    );
    let detail = audit_withdrawn(&env).expect("the withdrawal is recorded");
    assert!(detail.contains("search.operations"), "{detail}");
}

#[test]
fn a_partial_cpdag_completion_search_refuses_the_class_wide_claim() {
    // Six nodes, all pairs undirected: 720 members, more than 100 operations can visit.
    let mut cpdag = Cpdag::with_variables(6);
    for a in 0..6 {
        for b in (a + 1)..6 {
            cpdag.insert_undirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
    }
    let id = GeneralizedAdjustmentIdentifier {
        config: GeneralizedAdjustmentConfig {
            max_completions: usize::MAX,
            completion_operations: 100,
            ..Default::default()
        },
    };
    let q = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let env = id.identify_cpdag_envelope(&cpdag, &q).unwrap();
    assert!(env.cases.len() < 720);
    assert_ne!(env.status, IdentificationStatus::NonparametricallyIdentified);
    assert!(
        env.critical_graph_features.iter().any(|f| &*f.kind == "completion_enumeration_capped"
            && f.detail.contains("search.operations"))
    );
}

#[test]
fn envelope_preserves_mass_on_mixed_pag() {
    let mut pag = Pag::with_variables(2);
    pag.insert_circle_arrow(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let id = GeneralizedAdjustmentIdentifier {
        config: GeneralizedAdjustmentConfig {
            max_completions: 8,
            per_completion_weight: 1.0,
            max_candidates: 16,
            ..Default::default()
        },
    };
    let q = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let env = id.identify_pag_envelope(&pag, &q).unwrap();
    let total = env.identified_weight.0 + env.unidentified_weight.0;
    assert!(total > 0.0);
    assert!((total - env.cases.len() as f64).abs() < 1e-9);
}

#[test]
fn invisible_directed_edge_does_not_identify_even_conditionally() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../conformance/identify/generalized_adjustment/expected.json"
    ))
    .unwrap();
    assert_eq!(fixture["cases"][0]["status"], "not_identified_by_adjustment");
    let mut pag = Pag::with_variables(3);
    pag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let id = GeneralizedAdjustmentIdentifier::new();
    let q = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    for query in [q.clone(), q.with_effect_modifiers([VariableId::from_raw(2)])] {
        let env = id.identify_pag_envelope(&pag, &query).unwrap();
        assert_eq!(env.status, IdentificationStatus::NotIdentified);
        assert_eq!(env.identified_weight.0, 0.0);
        assert!(env.cases[0].result.estimands.is_empty());
    }
    // R -> T with R nonadjacent to Y makes T -> Y visible.
    pag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    let env = id
        .identify_pag_envelope(
            &pag,
            &AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)),
        )
        .unwrap();
    assert_eq!(env.status, IdentificationStatus::NonparametricallyIdentified);
    assert!(env.cases[0].result.estimands[0].adjustment_set.is_empty());
}

#[test]
fn proper_backdoor_preserves_side_effect_collider_edges() {
    // R -> T -> Y; T -> W <- U -> Y. R witnesses visibility of T -> Y.
    let mut mag = Pag::with_variables(5);
    for (a, b) in [(4, 0), (0, 1), (0, 2), (3, 2), (3, 1)] {
        mag.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    let t = DenseNodeId::from_raw(0);
    let y = DenseNodeId::from_raw(1);
    let w = DenseNodeId::from_raw(2);
    let cut = proper_backdoor_mag(&mag, t, y).unwrap();
    assert!(cut.children(t).contains(&w));
    assert!(!cut.children(t).contains(&y));
    let mut workspace = DSeparationWorkspace::default();
    assert!(cut.is_m_separated(t, y, &[], &mut workspace).unwrap());
    assert!(!cut.is_m_separated(t, y, &[w], &mut workspace).unwrap());
}

#[test]
fn confounder_identifies_with_nonempty_z() {
    // Z → T, Z → Y, T → Y  (backdoor {Z}).
    let mut pag = Pag::with_variables(4);
    let z = DenseNodeId::from_raw(0);
    let t = DenseNodeId::from_raw(1);
    let y = DenseNodeId::from_raw(2);
    pag.insert_directed(z, t).unwrap();
    pag.insert_directed(z, y).unwrap();
    pag.insert_directed(t, y).unwrap();
    pag.insert_directed(DenseNodeId::from_raw(3), t).unwrap();
    let id = GeneralizedAdjustmentIdentifier::new();
    let q = AverageEffectQuery::binary_ate(VariableId::from_raw(1), VariableId::from_raw(2));
    let env = id.identify_pag_envelope(&pag, &q).unwrap();
    assert_eq!(env.status, IdentificationStatus::NonparametricallyIdentified);
    assert_eq!(env.cases[0].result.status, IdentificationStatus::NonparametricallyIdentified);
    let z_set = &env.cases[0].result.estimands[0].adjustment_set;
    assert!(z_set.iter().any(|v| v.raw() == 0), "expected Z in adjustment, got {z_set:?}");
}

#[test]
fn a_declared_ceiling_keeps_the_certified_set_and_proved_nonidentification_stays_distinct() {
    // Case A: Z → T, Z → Y, T → Y, but max_candidates=0 puts the candidate set {Z} over the
    // declared ceiling before enumeration can start. The ancestral set was already tested
    // valid, so the effect is identified by it; only the search for a smaller set is
    // reported as bounded.
    let mut capped_pag = Pag::with_variables(4);
    let z = DenseNodeId::from_raw(0);
    let t = DenseNodeId::from_raw(1);
    let y = DenseNodeId::from_raw(2);
    capped_pag.insert_directed(z, t).unwrap();
    capped_pag.insert_directed(z, y).unwrap();
    capped_pag.insert_directed(t, y).unwrap();
    capped_pag.insert_directed(DenseNodeId::from_raw(3), t).unwrap();
    let capped_id = GeneralizedAdjustmentIdentifier {
        config: GeneralizedAdjustmentConfig {
            max_completions: 8,
            per_completion_weight: 1.0,
            max_candidates: 0,
            ..Default::default()
        },
    };
    let q = AverageEffectQuery::binary_ate(VariableId::from_raw(1), VariableId::from_raw(2));
    let capped_env = capped_id.identify_pag_envelope(&capped_pag, &q).unwrap();
    assert_eq!(capped_env.unidentified_weight.0, 0.0);
    assert_eq!(capped_env.status, IdentificationStatus::NonparametricallyIdentified);
    let capped_case = &capped_env.cases[0].result;
    assert_eq!(capped_case.status, IdentificationStatus::NonparametricallyIdentified);
    assert!(
        capped_case.diagnostics.iter().any(|d| {
            d.code.as_ref() == MAG_SEARCH_BOUNDED_DIAGNOSTIC_CODE
                && d.kind == DiagnosticKind::Execution
        }),
        "the unfinished search is an execution diagnostic: {:?}",
        capped_case.diagnostics
    );
    assert!(
        !capped_case.diagnostics.iter().any(|d| d.kind == DiagnosticKind::Scientific),
        "the bound must not be stamped scientific: {:?}",
        capped_case.diagnostics
    );

    // Case B: bare T ↔ Y (unmeasured confounding, no other variables). No adjustment set
    // can ever separate T from Y, so this is a genuine non-identification — proved, not
    // merely "could not tell".
    let mut confounded_pag = Pag::with_variables(2);
    confounded_pag.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let genuine_id = GeneralizedAdjustmentIdentifier::new();
    let q2 = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let genuine_env = genuine_id.identify_pag_envelope(&confounded_pag, &q2).unwrap();
    assert!(genuine_env.unidentified_weight.0 > 0.0);
    assert_eq!(
        genuine_env.truncated_completions, 0,
        "genuinely-blocked completions must not count as truncated"
    );

    // Only the proved non-identification carries unidentified mass.
    assert!(genuine_env.unidentified_weight.0 > capped_env.unidentified_weight.0);
}

#[test]
fn gac_candidates_allow_side_effect_descendants_of_t() {
    // T → M → Y, T → W, U → T, U → Y. W ∈ De(T) but W ∉ Forb(T,Y)=De({M,Y}).
    let mut g = Admg::with_variables(5);
    let t = DenseNodeId::from_raw(0);
    let y = DenseNodeId::from_raw(1);
    let m = DenseNodeId::from_raw(2);
    let w = DenseNodeId::from_raw(3);
    let u = DenseNodeId::from_raw(4);
    g.insert_directed(t, m).unwrap();
    g.insert_directed(m, y).unwrap();
    g.insert_directed(t, w).unwrap();
    g.insert_directed(u, t).unwrap();
    g.insert_directed(u, y).unwrap();
    let c = adjustment_candidates(&g, t, y);
    assert!(c.contains(&u), "confounder must remain a candidate: {c:?}");
    assert!(c.contains(&w), "GAC allows side-effect descendant W: {c:?}");
    assert!(!c.contains(&m), "mediator is in Forb: {c:?}");
    assert!(!c.contains(&t) && !c.contains(&y));
}

#[test]
fn cpdag_confounded_undirected_has_two_identified_completions() {
    // Z — T → Y, Z → Y. Completions: Z→T (backdoor {Z}) and T→Z (empty Z).
    let mut cpdag = Cpdag::with_variables(3);
    let z = DenseNodeId::from_raw(0);
    let t = DenseNodeId::from_raw(1);
    let y = DenseNodeId::from_raw(2);
    cpdag.insert_undirected(z, t).unwrap();
    cpdag.insert_directed(z, y).unwrap();
    cpdag.insert_directed(t, y).unwrap();
    let id = GeneralizedAdjustmentIdentifier::new();
    let q = AverageEffectQuery::binary_ate(VariableId::from_raw(1), VariableId::from_raw(2));
    let env = id.identify_cpdag_envelope(&cpdag, &q).unwrap();
    assert_eq!(env.cases.len(), 2);
    assert!(env.unidentified_weight.0 == 0.0);
    assert_eq!(env.identified_weight.0, 2.0);
    assert!(
        env.cases
            .iter()
            .all(|c| { c.result.status == IdentificationStatus::NonparametricallyIdentified })
    );
}

#[test]
fn fully_oriented_cpdag_is_a_one_case_envelope() {
    let mut cpdag = Cpdag::with_variables(2);
    cpdag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let env = GeneralizedAdjustmentIdentifier::new()
        .identify_cpdag_envelope(
            &cpdag,
            &AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)),
        )
        .unwrap();
    assert_eq!(env.cases.len(), 1);
    assert_eq!(env.status, IdentificationStatus::NonparametricallyIdentified);
    assert!(env.unidentified_weight.0 == 0.0);
}
