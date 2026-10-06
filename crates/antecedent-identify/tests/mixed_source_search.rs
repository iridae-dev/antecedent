//! Bounded mixed-source proof search (X9).
//!
//! Graphs are small binary ADMGs. The positive fixture is the chain
//! `X -> Z -> Y`: one study measures `{X, Z}` and another `{Z, Y}`, so no study
//! holds the joint the named routes need, but `P(y | do(x)) = Σ_z P(z | x) P(y | z)`
//! follows from the two together by the probability rules and do-calculus rules 1
//! and 2.

use std::sync::Arc;

use antecedent_core::{
    DependenceGroup, DistributionAvailability, Environment, EvidenceCatalog, EvidenceKind,
    EvidenceRegime, ExecutionContext, InterventionAssignment, LawOrigin, MemoryBudget,
    RegimeBinding, RegimeId, RegimeKind, SamplingDesign, SearchLimits, SearchStop, Value,
    VariableCoordinate, VariableDomain, VariableId,
};
use antecedent_expr::CausalExprArena;
use antecedent_graph::{Admg, DenseNodeId};
use antecedent_identify::{
    CatalogTransportResult, IdentificationError, MIXED_SOURCE_DEFAULT_LIMITS, MetaSource,
    MetaTransportQuery, MixedRule, MixedSourceDecision, MixedSourceQuery, MixedStageRecord,
    SidLimits, ZTransportSourceSpec, bind_mixed_source_catalog, decide_mixed_source,
    identify_meta_catalog, mixed_source_rule_names, verify_mixed_source_derivation,
};

const X: u32 = 0;
const Z: u32 = 1;
const Y: u32 = 2;

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

fn graph(n: u32, directed: &[(u32, u32)], bidirected: &[(u32, u32)]) -> Admg {
    let mut g = Admg::with_variables(n);
    let d = DenseNodeId::from_raw;
    for (a, b) in directed {
        g.insert_directed(d(*a), d(*b)).unwrap();
    }
    for (a, b) in bidirected {
        g.insert_bidirected(d(*a), d(*b)).unwrap();
    }
    g
}

fn chain() -> Admg {
    graph(3, &[(X, Z), (Z, Y)], &[])
}

fn query() -> MixedSourceQuery {
    MixedSourceQuery {
        outcomes: Arc::from([v(Y)]),
        treatments: Arc::from([v(X)]),
        target: Arc::from("target"),
        sources: Arc::from([]),
    }
}

/// An available joint regime of study `study` in the target population.
fn regime(id: u32, study: &str, on: &[u32], measured: &[u32]) -> EvidenceRegime {
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
    regime.study = Some(Arc::from(study));
    regime
}

fn catalog(regimes: Vec<EvidenceRegime>) -> EvidenceCatalog {
    let bindings = regimes
        .iter()
        .filter(|r| r.evidence_kind == EvidenceKind::Available)
        .map(|r| RegimeBinding {
            dataset_identity: None,
            regime: r.id,
            snapshot_identity: Arc::from(format!("snapshot-{}", r.id.raw())),
            schema_names: Arc::from([]),
            sampling: SamplingDesign::Independent,
            weights: None,
            dependence: DependenceGroup::IndependentStudies,
        })
        .collect::<Vec<_>>();
    EvidenceCatalog::try_new([], regimes, bindings, None).unwrap()
}

/// Study 1 measures `{X, Z}`, study 2 measures `{Z, Y}`.
fn complementary() -> EvidenceCatalog {
    catalog(vec![regime(1, "study-1", &[], &[X, Z]), regime(2, "study-2", &[], &[Z, Y])])
}

fn decide(g: &Admg, q: &MixedSourceQuery, c: &EvidenceCatalog) -> MixedSourceDecision {
    decide_mixed_source(g, q, c, MIXED_SOURCE_DEFAULT_LIMITS, &ExecutionContext::for_tests(1))
        .unwrap()
}

fn budget_receipt(decision: MixedSourceDecision) -> antecedent_core::SearchReceipt {
    match decision {
        MixedSourceDecision::Exhausted(receipt) => receipt,
        other => panic!("expected an exhausted search, got {other:?}"),
    }
}

#[test]
fn complementary_studies_identify_what_no_single_study_can() {
    let g = chain();
    let MixedSourceDecision::Identified { derivation, cited, alternatives } =
        decide(&g, &query(), &complementary())
    else {
        panic!("the two studies together identify the effect");
    };
    // Both studies are cited, each through the regime that supplies it.
    assert_eq!(cited.iter().map(|r| r.raw()).collect::<Vec<_>>(), [1, 2]);
    let leaves = derivation.leaves();
    assert_eq!(leaves.len(), 2);
    assert_eq!(
        leaves.iter().map(|(_, l)| l.study.as_ref()).collect::<Vec<_>>(),
        ["study-1", "study-2"]
    );
    assert!(leaves.iter().all(|(_, l)| l.population.as_ref() == "target"));
    assert!(leaves.iter().all(|(_, l)| l.snapshot.is_some() && !l.identity.is_empty()));
    assert!(alternatives.is_empty(), "alternatives appear only when found");
    // Every step records its premises; the proof exchanges the observation X for an action
    // (rule 2) and inserts the action X into the conditional of Y given Z (rule 3).
    let rules = derivation.steps().iter().map(|s| s.rule).collect::<Vec<_>>();
    assert!(rules.contains(&MixedRule::Rule2ToDo) && rules.contains(&MixedRule::Rule3Insert));
    assert!(rules.contains(&MixedRule::Product) && rules.contains(&MixedRule::Marginalize));
    for (index, step) in derivation.steps().iter().enumerate() {
        assert!(step.premises.iter().all(|p| *p < index));
        assert_eq!(step.source.is_some(), step.rule == MixedRule::Input);
    }
    // The named sID route ran first and could not bind.
    assert_eq!(
        derivation.stages(),
        [MixedStageRecord { stage: "target_first_sid".into(), outcome: "missing_evidence" }]
    );
    // The independent checker and the binding accept it; the record is reproduced.
    verify_mixed_source_derivation(&g, &derivation, &complementary()).unwrap();
    let bound = bind_mixed_source_catalog(&g, &derivation, &complementary()).unwrap();
    assert_eq!(bound.cited_regimes(), [RegimeId::from_raw(1), RegimeId::from_raw(2)]);
    let ctx = ExecutionContext::for_tests(1);
    let again = antecedent_identify::MixedSourceDerivation::from_record_checked(
        &g,
        &query(),
        &complementary(),
        &derivation.to_record(),
        derivation.arena(),
        MIXED_SOURCE_DEFAULT_LIMITS,
        &ctx,
    )
    .unwrap();
    assert_eq!(again.to_record(), derivation.to_record());
    // No single study identifies it.
    for one in [
        catalog(vec![regime(1, "study-1", &[], &[X, Z])]),
        catalog(vec![regime(2, "study-2", &[], &[Z, Y])]),
    ] {
        let alone = decide(&g, &query(), &one);
        assert!(matches!(alone, MixedSourceDecision::NotCertified(_)), "{alone:?}");
    }
}

#[test]
fn the_proof_graph_names_every_source_distribution() {
    let g = chain();
    let MixedSourceDecision::Identified { derivation, .. } = decide(&g, &query(), &complementary())
    else {
        panic!("identified");
    };
    let names = ["X", "Z", "Y"].map(String::from);
    let lines = derivation.proof_graph(&names);
    assert_eq!(lines.len(), derivation.steps().len());
    assert!(lines.iter().any(|l| l.contains("input study \"study-1\" regime 1")));
    assert!(lines.iter().any(|l| l.contains("input study \"study-2\" regime 2")));
    let last = lines.last().unwrap();
    assert!(last.contains("P(Y | do(X))") && last.ends_with("<- target"), "{last}");
    // Determinism: the same decision renders the same proof.
    let MixedSourceDecision::Identified { derivation: again, .. } =
        decide(&g, &query(), &complementary())
    else {
        panic!("identified");
    };
    assert_eq!(again.proof_graph(&names), lines);
}

#[test]
fn a_surrogate_experiment_and_an_observational_study_combine() {
    // Front-door graph with X <-> Y: the observational study alone cannot identify the
    // effect. Study 2 experiments on Z (a family regime: every level) and measures {X, Y}.
    let g = graph(3, &[(X, Z), (Z, Y)], &[(X, Y)]);
    let c = catalog(vec![regime(1, "study-1", &[], &[X, Z]), regime(2, "trial", &[Z], &[X, Y])]);
    let MixedSourceDecision::Identified { derivation, .. } = decide(&g, &query(), &c) else {
        panic!("the observational study and the surrogate trial together identify the effect");
    };
    let rules = derivation.steps().iter().map(|s| s.rule).collect::<Vec<_>>();
    assert!(rules.contains(&MixedRule::Rule3Insert), "{rules:?}");
    assert!(rules.contains(&MixedRule::Rule2ToObservation), "{rules:?}");
    assert!(rules.contains(&MixedRule::Rule2ToDo), "{rules:?}");
    assert_eq!(derivation.cited_regimes().len(), 2);
    verify_mixed_source_derivation(&g, &derivation, &c).unwrap();
    // Neither study identifies it alone.
    for one in [
        catalog(vec![regime(1, "study-1", &[], &[X, Z])]),
        catalog(vec![regime(2, "trial", &[Z], &[X, Y])]),
    ] {
        assert!(matches!(decide(&g, &query(), &one), MixedSourceDecision::NotCertified(_)));
    }
}

#[test]
fn separate_marginals_never_stand_in_for_a_joint() {
    let g = chain();
    let mut marginals = regime(2, "study-2", &[], &[Z, Y]);
    marginals.distribution =
        DistributionAvailability::SeparateMarginals { variables: Arc::from([v(Z), v(Y)]) };
    let c = catalog(vec![regime(1, "study-1", &[], &[X, Z]), marginals]);
    let decision = decide(&g, &query(), &c);
    assert_eq!(decision.reason_code(), Some("transport_missing_evidence"));
    assert_eq!(decision.detail_code(), Some("mixed_search.missing_joint"));
    let MixedSourceDecision::MissingEvidence(missing) = decision else {
        panic!("the relaxed pass names the joint");
    };
    // The exact missing leaf: study 2's joint over {Z, Y}, held only as marginals.
    assert_eq!(missing.leaves.len(), 1);
    let leaf = &missing.leaves[0];
    assert_eq!((leaf.regime.raw(), leaf.study.as_ref()), (2, "study-2"));
    assert_eq!(leaf.variables, [v(Z), v(Y)]);
    assert_eq!(leaf.marginals, [v(Z), v(Y)]);
    assert!(leaf.intervened.is_empty());
    let names = ["X", "Z", "Y"].map(String::from);
    let lines = missing.proof_graph(&names);
    let marked = lines.iter().filter(|l| l.contains("MISSING")).collect::<Vec<_>>();
    assert_eq!(marked.len(), 1, "{lines:?}");
    assert!(marked[0].contains("P(Z, Y)") && marked[0].contains("study-2"), "{lines:?}");
    // A joint over the same variables identifies it, so the marginals were the gap.
    assert!(matches!(
        decide(&g, &query(), &complementary()),
        MixedSourceDecision::Identified { .. }
    ));
}

#[test]
fn an_unsolved_query_is_not_certified_never_proven_unidentifiable() {
    // Bow arc X -> Y with X <-> Y is not identifiable; the search still only reports
    // that its frozen rule set found nothing.
    let g = graph(2, &[(0, 1)], &[(0, 1)]);
    let q = MixedSourceQuery {
        outcomes: Arc::from([v(1)]),
        treatments: Arc::from([v(0)]),
        target: Arc::from("target"),
        sources: Arc::from([]),
    };
    let c = catalog(vec![regime(1, "study", &[], &[0, 1])]);
    let decision = decide(&g, &q, &c);
    assert_eq!(decision.reason_code(), Some("transport_not_certified"));
    assert_eq!(decision.detail_code(), Some("mixed_search.not_certified"));
    let MixedSourceDecision::NotCertified(inspection) = decision else {
        panic!("not certified");
    };
    assert_eq!(inspection.goal.y, [v(1)]);
    assert_eq!(inspection.goal.intervened, [v(0)]);
    assert!(inspection.quantities > 0 && inspection.generations > 0 && inspection.operations > 0);
    assert!(!inspection.frontier.is_empty());
    assert!(inspection.rule_counts.iter().any(|(rule, n)| *rule == "input" && *n == 1));
    assert_eq!(
        inspection.stages.last().unwrap(),
        &MixedStageRecord { stage: "rule_search".into(), outcome: "not_certified" }
    );
}

#[test]
fn excluded_regimes_are_listed_with_their_reason() {
    let g = chain();
    let mut elsewhere = regime(3, "study-3", &[], &[X, Y]);
    elsewhere.population = Arc::from("elsewhere");
    let mut proposed = regime(4, "study-4", &[Z], &[X, Y]);
    proposed.evidence_kind = EvidenceKind::Proposed;
    let mut conditioned = regime(5, "study-5", &[], &[X, Z]);
    conditioned.conditioned_on = Arc::from([v(X)]);
    let c = catalog(vec![regime(1, "study-1", &[], &[X, Z]), elsewhere, proposed, conditioned]);
    let MixedSourceDecision::NotCertified(inspection) = decide(&g, &query(), &c) else {
        panic!("not certified");
    };
    let reasons =
        inspection.exclusions.iter().map(|e| (e.regime.raw(), e.reason)).collect::<Vec<_>>();
    assert_eq!(
        reasons,
        [(3, "other_population"), (4, "not_available"), (5, "conditioned_projection")]
    );
}

#[test]
fn a_model_artifact_offered_as_an_experimental_law_is_refused() {
    let g = chain();
    let mut posterior = regime(2, "posterior", &[Z], &[X, Y]);
    posterior.origin = LawOrigin::ModelArtifact { artifact: Arc::from("posterior-1") };
    let c = catalog(vec![regime(1, "study-1", &[], &[X, Z]), posterior]);
    let error = decide_mixed_source(
        &g,
        &query(),
        &c,
        MIXED_SOURCE_DEFAULT_LIMITS,
        &ExecutionContext::for_tests(1),
    )
    .unwrap_err();
    assert_eq!(
        error,
        IdentificationError::UnsupportedInput { code: "mixed_search.posterior_as_law" }
    );
    // An observational model artifact is only excluded.
    let mut fitted = regime(2, "fitted", &[], &[Z, Y]);
    fitted.origin = LawOrigin::ModelArtifact { artifact: Arc::from("posterior-2") };
    let c = catalog(vec![regime(1, "study-1", &[], &[X, Z]), fitted]);
    let MixedSourceDecision::NotCertified(inspection) = decide(&g, &query(), &c) else {
        panic!("an observational model artifact never supplies a leaf");
    };
    assert_eq!(inspection.exclusions[0].reason, "model_artifact");
}

#[test]
fn named_routes_run_first_and_a_solved_query_is_never_searched() {
    // The full observational joint lets sID identify the chain.
    let g = chain();
    let c = catalog(vec![regime(1, "full", &[], &[X, Z, Y])]);
    let decision = decide(&g, &query(), &c);
    assert_eq!(decision.detail_code(), Some("mixed_search.named_route"));
    assert_eq!(decision.reason_code(), Some("route_not_supported"));
    let MixedSourceDecision::NamedRoute { route, stages } = decision else {
        panic!("a named route solves it");
    };
    assert_eq!(route, "target_first_sid");
    assert_eq!(
        stages,
        [MixedStageRecord { stage: "target_first_sid".into(), outcome: "identified" }]
    );
}

/// Classical meta-transportability fixture (Bareinboim and Pearl 2013): the
/// chain `X -> Z -> Y` with `X <-> Z` and `X <-> Y`. Source `a` differs in `Y`'s
/// mechanism and holds a trial of `do(X)` measuring `Z`; source `b` differs in
/// `Z`'s mechanism and holds a trial of `do(Z)` measuring `Y`. Neither source
/// alone identifies `P(y | do(x))` in the target; together they do.
fn meta_graph() -> Admg {
    graph(3, &[(X, Z), (Z, Y)], &[(X, Z), (X, Y)])
}

fn source_regime(id: u32, population: &str, on: u32, measured: u32) -> EvidenceRegime {
    let mut regime = regime(id, &format!("trial-{population}"), &[on], &[measured]);
    regime.population = Arc::from(population);
    regime
}

fn meta_source(population: &str, controllable: &[u32], selection: u32) -> ZTransportSourceSpec {
    ZTransportSourceSpec {
        population: Arc::from(population),
        controllable: controllable.iter().copied().map(v).collect::<Vec<_>>().into(),
        experiment_assignment: Arc::from([]),
        selection_targets: Arc::from([v(selection)]),
    }
}

fn meta_query(controllable: &[u32]) -> MixedSourceQuery {
    let mut q = query();
    q.sources = Arc::from([meta_source("a", controllable, Y), meta_source("b", controllable, Z)]);
    q
}

/// The declared environments (with their selection targets) of the meta fixture.
fn meta_environments() -> Vec<Environment> {
    let coordinates = || {
        [X, Z, Y]
            .map(|i| VariableCoordinate {
                variable: v(i),
                domain: VariableDomain::Binary,
                unit: None,
            })
            .to_vec()
    };
    [("a", vec![v(Y)]), ("b", vec![v(Z)]), ("target", vec![])]
        .into_iter()
        .map(|(name, selection)| Environment::try_new(name, coordinates(), selection).unwrap())
        .collect()
}

fn with_environments(mut evidence: EvidenceCatalog) -> EvidenceCatalog {
    evidence = EvidenceCatalog::try_new(
        meta_environments(),
        evidence.regimes.to_vec(),
        evidence.bindings.to_vec(),
        None,
    )
    .unwrap();
    evidence
}

fn meta_catalog() -> EvidenceCatalog {
    with_environments(catalog(vec![source_regime(1, "a", X, Z), source_regime(2, "b", Z, Y)]))
}

#[test]
fn a_meta_only_query_is_returned_as_the_meta_named_route_and_never_searched() {
    let g = meta_graph();
    let q = meta_query(&[X, Z, Y]);
    let c = meta_catalog();
    let decision = decide(&g, &q, &c);
    assert_eq!(decision.detail_code(), Some("mixed_search.named_route"));
    let MixedSourceDecision::NamedRoute { route, stages } = decision else {
        panic!("the meta route solves it");
    };
    assert_eq!(route, "meta_transport");
    // Every earlier named stage ran, and none of them solved it, before meta did.
    assert_eq!(
        stages,
        [
            MixedStageRecord { stage: "target_first_sid".into(), outcome: "missing_evidence" },
            MixedStageRecord { stage: "mz_transport".into(), outcome: "missing_evidence" },
            MixedStageRecord { stage: "meta_transport".into(), outcome: "identified" },
        ]
    );
}

#[test]
fn the_meta_route_is_outside_the_contract_for_restricted_or_single_sources() {
    let g = meta_graph();
    let c = meta_catalog();
    // A source that cannot experiment on every variable is the mz route's, not
    // meta's: the meta theorem assumes the unrestricted family. Nothing searches
    // the source populations' regimes, so the frozen refusal is returned.
    let restricted = decide(&g, &meta_query(&[X, Z]), &c);
    assert_eq!(restricted.detail_code(), Some("mixed_search.not_certified"));
    let MixedSourceDecision::NotCertified(inspection) = restricted else {
        panic!("outside the meta contract and the target population's regimes");
    };
    let stages = inspection.stages;
    assert!(stages.iter().all(|s| s.stage != "meta_transport"), "{stages:?}");
    let mut single = meta_query(&[X, Z, Y]);
    single.sources = Arc::from([single.sources[0].clone()]);
    let MixedSourceDecision::NotCertified(inspection) = decide(&g, &single, &c) else {
        panic!("one source is the z route's and is not a meta query");
    };
    let stages = inspection.stages;
    assert!(stages.iter().all(|s| s.stage != "meta_transport"), "{stages:?}");
}

#[test]
fn earlier_named_routes_take_precedence_over_meta() {
    // Unconfounded chain: the target's observational joint identifies the effect
    // by target-first sID, although the same catalog also feeds the meta route.
    let g = chain();
    let q = meta_query(&[X, Z, Y]);
    let mut regimes = vec![source_regime(1, "a", X, Z), source_regime(2, "b", Z, Y)];
    regimes.push(regime(3, "full", &[], &[X, Z, Y]));
    let MixedSourceDecision::NamedRoute { route, stages } =
        decide(&g, &q, &with_environments(catalog(regimes)))
    else {
        panic!("a named route solves it");
    };
    assert_eq!(route, "target_first_sid");
    assert_eq!(
        stages,
        [MixedStageRecord { stage: "target_first_sid".into(), outcome: "identified" }]
    );
}

#[test]
fn meta_is_charged_to_the_one_shared_budget() {
    let g = meta_graph();
    let q = meta_query(&[X, Z, Y]);
    let c = meta_catalog();
    let ctx = ExecutionContext::for_tests(1);
    let used = |limits| decide_mixed_source(&g, &q, &c, limits, &ctx).unwrap();
    // Run out of operations inside the meta stage: earlier stages are explored.
    let mut stopped = None;
    for operations in (1..400).step_by(3) {
        if let MixedSourceDecision::Exhausted(receipt) =
            used(SearchLimits { operations, depth: 16 })
        {
            if receipt.unevaluated.iter().any(|r| r == "stage:meta_transport")
                && receipt.explored.iter().any(|r| r == "stage:mz_transport")
            {
                stopped = Some(receipt);
            }
        }
    }
    let receipt = stopped.expect("a meta-stage stop is observed under some operation limit");
    assert_eq!(receipt.stop, SearchStop::Operations);
    assert!(receipt.unevaluated.iter().any(|r| r == "stage:rule_search"), "{receipt:?}");
}

#[test]
fn a_declared_source_runs_the_z_route_before_the_rule_search() {
    let g = chain();
    let mut q = query();
    q.sources = Arc::from([ZTransportSourceSpec {
        population: Arc::from("source"),
        controllable: Arc::from([v(Z)]),
        experiment_assignment: Arc::from([]),
        selection_targets: Arc::from([v(Y)]),
    }]);
    let MixedSourceDecision::Identified { derivation, .. } = decide(&g, &q, &complementary())
    else {
        panic!("the rule search still identifies it");
    };
    let stages = derivation.stages().iter().map(|s| s.stage.as_str()).collect::<Vec<_>>();
    assert_eq!(stages, ["target_first_sid", "z_transport"]);
}

#[test]
fn two_studies_supplying_one_distribution_are_an_alternative_derivation() {
    let g = chain();
    // Two trials of do(X) measuring {Y} both supply the target itself.
    let c = catalog(vec![regime(1, "trial-a", &[X], &[Y]), regime(2, "trial-b", &[X], &[Y])]);
    let MixedSourceDecision::Identified { derivation, alternatives, .. } = decide(&g, &query(), &c)
    else {
        panic!("a trial of do(X) supplies the target");
    };
    assert_eq!(derivation.steps().len(), 1);
    assert_eq!(derivation.cited_regimes(), [RegimeId::from_raw(1)]);
    assert_eq!(alternatives.len(), 1);
    assert_eq!(alternatives[0].cited_regimes(), [RegimeId::from_raw(2)]);
    verify_mixed_source_derivation(&g, &alternatives[0], &c).unwrap();
}

#[test]
fn exhausted_search_returns_a_receipt_never_a_verdict() {
    let g = chain();
    let ctx = ExecutionContext::for_tests(1);
    let run = |limits: SearchLimits, ctx: &ExecutionContext| {
        budget_receipt(decide_mixed_source(&g, &query(), &complementary(), limits, ctx).unwrap())
    };
    let ops = run(SearchLimits { operations: 30, depth: 16 }, &ctx);
    assert_eq!(ops.stop, SearchStop::Operations);
    assert_eq!(ops.operations_consumed, Some(30));
    assert!(ops.explored.iter().any(|r| r == "stage:target_first_sid"), "{ops:?}");
    assert!(ops.unevaluated.iter().any(|r| r == "stage:rule_search"), "{ops:?}");
    assert!(ops.unevaluated.iter().any(|r| r.starts_with("rule_search:generation:")), "{ops:?}");
    let depth = run(SearchLimits { operations: 20_000, depth: 1 }, &ctx);
    assert_eq!(depth.stop, SearchStop::Depth);
    assert_eq!(depth.depth_limit, 1);
    let cancelled = ExecutionContext::for_tests(1);
    cancelled.cancellation.cancel();
    assert_eq!(run(MIXED_SOURCE_DEFAULT_LIMITS, &cancelled).stop, SearchStop::Cancelled);
    let decision = decide_mixed_source(
        &g,
        &query(),
        &complementary(),
        SearchLimits { operations: 30, depth: 16 },
        &ctx,
    )
    .unwrap();
    assert_eq!(decision.reason_code(), Some("transport_budget_cancel"));
    assert_eq!(decision.detail_code(), Some("mixed_search.budget"));
}

#[test]
fn memory_exhaustion_is_a_receipt_and_is_checked_on_every_charge() {
    let g = chain();
    for limit in [16u64, 512, 2048] {
        let mut ctx = ExecutionContext::for_tests(1);
        ctx.memory = MemoryBudget { soft_limit_bytes: None, hard_limit_bytes: Some(limit) };
        let decision =
            decide_mixed_source(&g, &query(), &complementary(), MIXED_SOURCE_DEFAULT_LIMITS, &ctx)
                .unwrap();
        // A small hard limit stops the search; the live state grows with each charge, so a
        // limit the search outgrows is observed mid-search rather than only at the start.
        match decision {
            MixedSourceDecision::Exhausted(receipt) => {
                assert_eq!(receipt.stop, SearchStop::Memory);
                assert_eq!(receipt.memory_limit_bytes, Some(limit));
            }
            MixedSourceDecision::Identified { .. } => assert!(limit >= 2048),
            other => panic!("memory never yields a verdict: {other:?}"),
        }
    }
}

#[test]
fn one_budget_is_shared_by_every_stage() {
    let g = chain();
    let MixedSourceDecision::Identified { derivation, .. } = decide(&g, &query(), &complementary())
    else {
        panic!("identified");
    };
    let alone = derivation.summary().operations_to_proof;
    assert!(alone > 0);
    // A limit the rule search alone would fit still exhausts once the named stage that ran
    // first is charged to the same budget.
    let receipt = budget_receipt(
        decide_mixed_source(
            &g,
            &query(),
            &complementary(),
            SearchLimits { operations: alone, depth: 16 },
            &ExecutionContext::for_tests(1),
        )
        .unwrap(),
    );
    assert_eq!(receipt.stop, SearchStop::Operations);
    assert_eq!(receipt.operations_consumed, Some(alone));
    assert!(receipt.explored.iter().any(|r| r == "stage:target_first_sid"), "{receipt:?}");
}

#[test]
fn bounds_refuse_as_bounds_exceeded() {
    let ctx = ExecutionContext::for_tests(1);
    let bounds = IdentificationError::UnsupportedInput { code: "mixed_search.bounds_exceeded" };
    let big = graph(11, &[], &[]);
    let q = MixedSourceQuery {
        outcomes: Arc::from([v(1)]),
        treatments: Arc::from([v(0)]),
        target: Arc::from("target"),
        sources: Arc::from([]),
    };
    let c = catalog(vec![regime(1, "study", &[], &[0, 1])]);
    assert_eq!(
        decide_mixed_source(&big, &q, &c, MIXED_SOURCE_DEFAULT_LIMITS, &ctx).unwrap_err(),
        bounds
    );
    let g = chain();
    for limits in [
        SearchLimits { operations: MIXED_SOURCE_DEFAULT_LIMITS.operations + 1, depth: 16 },
        SearchLimits { operations: 100, depth: MIXED_SOURCE_DEFAULT_LIMITS.depth + 1 },
    ] {
        assert_eq!(
            decide_mixed_source(&g, &query(), &complementary(), limits, &ctx).unwrap_err(),
            bounds
        );
    }
    let many = catalog((1..=17).map(|i| regime(i, &format!("s{i}"), &[], &[X, Z])).collect());
    assert_eq!(
        decide_mixed_source(&g, &query(), &many, MIXED_SOURCE_DEFAULT_LIMITS, &ctx).unwrap_err(),
        bounds
    );
}

#[test]
fn malformed_queries_are_invalid_not_unidentified() {
    let g = chain();
    let ctx = ExecutionContext::for_tests(1);
    let mut overlapping = query();
    overlapping.treatments = Arc::from([v(Y)]);
    let unknown = MixedSourceQuery { outcomes: Arc::from([v(9)]), ..query() };
    for bad in [overlapping, unknown] {
        let error =
            decide_mixed_source(&g, &bad, &complementary(), MIXED_SOURCE_DEFAULT_LIMITS, &ctx)
                .unwrap_err();
        assert!(
            matches!(&error, IdentificationError::InvalidInput { message }
            if message.starts_with("mixed_search.invalid_query")),
            "{error:?}"
        );
    }
}

type Mutation = Box<dyn Fn(&mut antecedent_identify::MixedSourceDerivationRecord)>;

#[test]
fn a_tampered_record_or_expression_fails_the_replay() {
    let g = chain();
    let ctx = ExecutionContext::for_tests(1);
    let MixedSourceDecision::Identified { derivation, .. } = decide(&g, &query(), &complementary())
    else {
        panic!("identified");
    };
    let replay = |record: &antecedent_identify::MixedSourceDerivationRecord,
                  arena: &CausalExprArena,
                  catalog: &EvidenceCatalog| {
        antecedent_identify::MixedSourceDerivation::from_record_checked(
            &g,
            &query(),
            catalog,
            record,
            arena,
            MIXED_SOURCE_DEFAULT_LIMITS,
            &ctx,
        )
    };
    let bad = IdentificationError::InvalidDerivation { code: "mixed_search.invalid_derivation" };
    let record = derivation.to_record();
    replay(&record, derivation.arena(), &complementary()).unwrap();
    let mutations: Vec<Mutation> = vec![
        Box::new(|r| r.rule_set = "x9.rules.v0".into()),
        Box::new(|r| r.steps[0].rule = "product".into()),
        Box::new(|r| r.steps.last_mut().unwrap().params.push(7)),
        Box::new(|r| r.steps[0].study = Some("someone-else".into())),
        Box::new(|r| r.steps[0].source = Some("catalog_distribution.v1|forged".into())),
        Box::new(|r| r.steps[0].snapshot = Some("snapshot-9".into())),
        Box::new(|r| r.graph_signature.push('x')),
        Box::new(|r| r.root_step = 0),
        Box::new(|r| r.operations_to_proof += 1),
        Box::new(|r| r.stages.clear()),
    ];
    for mutate in mutations {
        let mut tampered = record.clone();
        mutate(&mut tampered);
        assert_eq!(replay(&tampered, derivation.arena(), &complementary()).unwrap_err(), bad);
    }
    assert_eq!(replay(&record, &CausalExprArena::new(), &complementary()).unwrap_err(), bad);
    // Another catalog re-decides differently.
    let other =
        catalog(vec![regime(1, "study-1", &[], &[X, Z]), regime(2, "renamed", &[], &[Z, Y])]);
    assert_eq!(replay(&record, derivation.arena(), &other).unwrap_err(), bad);
    // Another graph does not check.
    assert!(
        verify_mixed_source_derivation(
            &graph(3, &[(X, Z), (Z, Y)], &[(X, Z)]),
            &derivation,
            &complementary()
        )
        .is_err()
    );
}

#[test]
fn the_rule_set_is_frozen() {
    assert_eq!(
        mixed_source_rule_names(),
        [
            "input",
            "marginalize",
            "condition",
            "rule1_insert",
            "rule1_delete",
            "rule2_to_do",
            "rule2_to_observation",
            "rule3_insert",
            "rule3_delete",
            "product",
        ]
    );
    assert_eq!(antecedent_identify::MIXED_SOURCE_RULE_SET, "x9.rules.v1");
}

/// A trial of `do(X = true)` only is not a family regime: it supplies the
/// response at one level, and the symbolic search states identification for every
/// level. The named routes refuse it through `EvidenceRegime::satisfies`; the rule
/// search excludes it as `restricted_levels`, so no identification is stated for a
/// level nobody measured.
#[test]
fn a_value_restricted_trial_is_excluded_not_identified_for_every_level() {
    let g = chain();
    let mut restricted = regime(1, "trial-x1", &[X], &[Y]);
    restricted.intervention_values =
        Arc::from([InterventionAssignment { variable: v(X), value: Value::Bool(true) }]);
    let c = catalog(vec![restricted]);
    let decision = decide(&g, &query(), &c);
    let MixedSourceDecision::NotCertified(inspection) = decision else {
        panic!("a restricted trial identifies nothing symbolically: {decision:?}");
    };
    assert_eq!(
        inspection.exclusions.iter().map(|e| (e.regime.raw(), e.reason)).collect::<Vec<_>>(),
        [(1, "restricted_levels")]
    );
    // The same trial with every level is the target itself.
    let family = catalog(vec![regime(1, "trial-x1", &[X], &[Y])]);
    assert!(matches!(decide(&g, &query(), &family), MixedSourceDecision::Identified { .. }));
    // A restricted regime is excluded wherever it appears, next to a usable one.
    let mut restricted_z = regime(2, "trial-z", &[Z], &[X, Y]);
    restricted_z.intervention_values =
        Arc::from([InterventionAssignment { variable: v(Z), value: Value::Bool(false) }]);
    let mixed = catalog(vec![regime(1, "study-1", &[], &[X, Z]), restricted_z]);
    let MixedSourceDecision::NotCertified(inspection) = decide(&g, &query(), &mixed) else {
        panic!("not certified");
    };
    assert_eq!(inspection.exclusions[0].reason, "restricted_levels");
    // Editing an identified proof's catalog to a restricted regime breaks it.
    let MixedSourceDecision::Identified { derivation, .. } = decide(&g, &query(), &family) else {
        panic!("identified");
    };
    let mut edited = regime(1, "trial-x1", &[X], &[Y]);
    edited.intervention_values =
        Arc::from([InterventionAssignment { variable: v(X), value: Value::Bool(true) }]);
    assert!(bind_mixed_source_catalog(&g, &derivation, &catalog(vec![edited])).is_err());
}

/// The same measured set decides identification only through `availability`: a
/// joint identifies, separate marginals of the same variables (or of a proper
/// subset) do not.
#[test]
fn availability_alone_decides_identification() {
    let g = chain();
    let with = |availability: DistributionAvailability| {
        let mut second = regime(2, "study-2", &[], &[Z, Y]);
        second.distribution = availability;
        catalog(vec![regime(1, "study-1", &[], &[X, Z]), second])
    };
    assert!(matches!(
        decide(&g, &query(), &with(DistributionAvailability::Joint)),
        MixedSourceDecision::Identified { .. }
    ));
    for variables in [vec![v(Z), v(Y)], vec![v(Z)]] {
        let decision = decide(
            &g,
            &query(),
            &with(DistributionAvailability::SeparateMarginals { variables: variables.into() }),
        );
        assert_eq!(decision.detail_code(), Some("mixed_search.missing_joint"), "{decision:?}");
    }
}

/// Two sources that can each experiment on everything, where the second one's
/// mechanism changes are irrelevant to `Y`: its `do(X)` trial transports. The mz
/// route identifies it, and the classical meta route would identify it too;
/// meta is then never run.
#[test]
fn mz_takes_precedence_over_meta_when_both_could_solve() {
    let g = graph(3, &[(X, Y)], &[(X, Y)]);
    let mut q = query();
    q.sources = Arc::from([meta_source("a", &[X, Z, Y], Y), meta_source("b", &[X, Z, Y], Z)]);
    let mut trial_a = regime(1, "trial-a", &[X], &[Z, Y]);
    trial_a.population = Arc::from("a");
    let mut trial_b = regime(2, "trial-b", &[X], &[Z, Y]);
    trial_b.population = Arc::from("b");
    let environments = [("a", vec![v(Y)]), ("b", vec![v(Z)]), ("target", vec![])]
        .into_iter()
        .map(|(name, selection)| {
            let coordinates = [X, Z, Y]
                .map(|i| VariableCoordinate {
                    variable: v(i),
                    domain: VariableDomain::Binary,
                    unit: None,
                })
                .to_vec();
            Environment::try_new(name, coordinates, selection).unwrap()
        })
        .collect::<Vec<_>>();
    let base = catalog(vec![trial_a, trial_b]);
    let c =
        EvidenceCatalog::try_new(environments, base.regimes.to_vec(), base.bindings.to_vec(), None)
            .unwrap();
    let MixedSourceDecision::NamedRoute { route, stages } = decide(&g, &q, &c) else {
        panic!("a named route solves it");
    };
    assert_eq!(route, "mz_transport");
    assert_eq!(
        stages,
        [
            MixedStageRecord { stage: "target_first_sid".into(), outcome: "missing_evidence" },
            MixedStageRecord { stage: "mz_transport".into(), outcome: "identified" },
        ],
        "meta never ran"
    );
    // The precondition that makes the precedence meaningful: meta alone solves it.
    let meta = MetaTransportQuery {
        outcomes: Arc::clone(&q.outcomes),
        treatments: Arc::clone(&q.treatments),
        target: Arc::clone(&q.target),
        sources: q
            .sources
            .iter()
            .map(|s| MetaSource {
                population: s.population.to_string(),
                selections: s.selection_targets.iter().map(|t| t.raw()).collect(),
            })
            .collect(),
    };
    let alone =
        identify_meta_catalog(&g, &meta, &c, SidLimits::default(), &ExecutionContext::for_tests(1))
            .unwrap();
    assert!(matches!(alone, CatalogTransportResult::Identified(_)), "{alone:?}");
}

/// A single declared source whose `do(X)` trial transports (its selection changes
/// only `X`'s mechanism): the z route solves it, so the rule search does not run
/// even though the target population's own `do(X)` trial would let it derive a
/// proof.
#[test]
fn a_solving_z_route_precedes_the_rule_search() {
    let g = graph(3, &[(X, Y)], &[(X, Y)]);
    let mut q = query();
    q.sources = Arc::from([ZTransportSourceSpec {
        population: Arc::from("source"),
        controllable: Arc::from([v(X)]),
        experiment_assignment: Arc::from([]),
        selection_targets: Arc::from([v(X)]),
    }]);
    let mut trial = regime(3, "source-trial", &[X], &[Z, Y]);
    trial.population = Arc::from("source");
    let c = catalog(vec![
        trial,
        regime(1, "observational", &[], &[Z, Y]),
        regime(5, "target-trial", &[X], &[Y]),
    ]);
    // The rule search alone identifies it from the target's own trial.
    let mut no_source = q.clone();
    no_source.sources = Arc::from([]);
    assert!(matches!(decide(&g, &no_source, &c), MixedSourceDecision::Identified { .. }));
    let decision = decide(&g, &q, &c);
    let MixedSourceDecision::NamedRoute { route, stages } = decision else {
        panic!("the z route solves it before the rule search: {decision:?}");
    };
    assert_eq!(route, "z_transport");
    assert_eq!(
        stages,
        [
            MixedStageRecord { stage: "target_first_sid".into(), outcome: "missing_evidence" },
            MixedStageRecord { stage: "z_transport".into(), outcome: "identified" },
        ],
        "no rule search ran"
    );
}

/// The counterexample rule 3 needs its `W(Z)` exception for: `W -> Z -> Y` and
/// `W <-> Y`. The observational `P(Z, Y)` and a trial `P(Z | do(W))` would give
/// `P(y | do(w)) = sum_z P(z | do(w)) P(y | z)` if `do(W)` could be inserted into
/// `P(y | z)`. That is unsound because `W` is an ancestor of `Z`: conditioning on
/// `Z` re-opens the confounded path, and the effect is not identified from these
/// studies (the front-door formula needs `W` measured with `Z` and `Y`). The
/// search refuses to insert it and returns nothing instead of the wrong formula.
#[test]
fn rule_three_insertion_is_refused_when_the_action_is_an_ancestor_of_the_observed_set() {
    let (w, zed, y) = (0u32, 1u32, 2u32);
    let g = graph(3, &[(w, zed), (zed, y)], &[(w, y)]);
    let q = MixedSourceQuery {
        outcomes: Arc::from([v(y)]),
        treatments: Arc::from([v(w)]),
        target: Arc::from("target"),
        sources: Arc::from([]),
    };
    let c = catalog(vec![
        regime(1, "observational", &[], &[zed, y]),
        regime(2, "trial-on-w", &[w], &[zed]),
    ]);
    let decision = decide(&g, &q, &c);
    assert!(matches!(decision, MixedSourceDecision::NotCertified(_)), "{decision:?}");
}

/// The rule search alone is incomplete against ID on a single fully observed study
/// (module docs of `sid::mixed_source`: the napkin family and the three-intervention
/// bound are `not_certified` there), but the decision as a whole is not: the named
/// target-first sID route runs first and solves both, so the incompleteness never
/// reaches a caller who supplies the whole joint.
#[test]
fn a_query_the_rule_search_alone_misses_is_solved_by_the_named_route_on_a_full_joint() {
    let napkin = graph(4, &[(0, 1), (1, 2), (2, 3)], &[(0, 2), (0, 3)]);
    let napkin_query = MixedSourceQuery {
        outcomes: Arc::from([v(3)]),
        treatments: Arc::from([v(2)]),
        target: Arc::from("target"),
        sources: Arc::from([]),
    };
    let four_treatments = graph(5, &[], &[]);
    let four_query = MixedSourceQuery {
        outcomes: Arc::from([v(0)]),
        treatments: Arc::from([v(1), v(2), v(3), v(4)]),
        target: Arc::from("target"),
        sources: Arc::from([]),
    };
    for (g, q, measured) in [
        (&napkin, &napkin_query, vec![0, 1, 2, 3]),
        (&four_treatments, &four_query, vec![0, 1, 2, 3, 4]),
    ] {
        let evidence = catalog(vec![regime(1, "study", &[], &measured)]);
        match decide(g, q, &evidence) {
            MixedSourceDecision::NamedRoute { route, .. } => assert_eq!(route, "target_first_sid"),
            other => panic!("expected the named sID route, got {other:?}"),
        }
    }
}

/// Cancellation observed in the middle of the search (a token that trips after
/// `checks` observations, one per budget charge) is a receipt of what was
/// explored, never a verdict; enough checks let the same search identify.
#[test]
fn a_cancellation_in_the_middle_of_the_search_is_a_receipt_never_a_verdict() {
    let g = chain();
    let mut mid_search = 0usize;
    for checks in 1..20_000 {
        let mut ctx = ExecutionContext::for_tests(1);
        ctx.cancellation = antecedent_core::CancellationToken::cancel_after_checks(checks);
        match decide_mixed_source(&g, &query(), &complementary(), MIXED_SOURCE_DEFAULT_LIMITS, &ctx)
            .unwrap()
        {
            MixedSourceDecision::Exhausted(receipt) => {
                assert_eq!(receipt.stop, SearchStop::Cancelled, "{receipt:?}");
                if receipt.operations_consumed.is_some_and(|n| n > 0) {
                    mid_search += 1;
                    assert!(!receipt.unevaluated.is_empty(), "{receipt:?}");
                }
            }
            MixedSourceDecision::Identified { .. } => break,
            other => panic!("a cancellation never yields a verdict: {other:?}"),
        }
        assert!(checks < 19_999, "the search never finished");
    }
    assert!(mid_search > 0);
}
