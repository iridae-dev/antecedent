use super::*;
use crate::test_support::{d, v};
fn query() -> ClassicalTransportQuery {
    ClassicalTransportQuery {
        outcomes: Arc::from([v(2)]),
        treatments: Arc::from([v(0)]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    }
}
#[test]
fn live_bytes_are_cumulative_across_the_stages_of_one_decision() {
    use antecedent_core::SearchBudget;
    let mut ctx = ExecutionContext::for_tests(1);
    ctx.set_memory_limit(Some(250));
    let limits = SearchLimits { operations: 100, depth: 8 };
    // Each stage alone fits 250 bytes; the retained first stage does not leave
    // room for the second.
    let mut alone = SharedSearch::new(SearchBudget::new(limits, &ctx).unwrap());
    alone.begin_stage();
    alone.charge(0, 200).unwrap();
    alone.charge(0, 200).unwrap();
    let mut staged = SharedSearch::new(SearchBudget::new(limits, &ctx).unwrap());
    staged.begin_stage();
    staged.charge(0, 200).unwrap();
    staged.end_stage();
    staged.begin_stage();
    assert_eq!(staged.charge(0, 100), Err(SearchStop::Memory));
    assert_eq!(staged.peak_bytes(), 300);
    // Without a context limit the mandatory default cap still bounds it.
    let free = ExecutionContext::for_tests(1);
    let mut capped = SharedSearch::new(SearchBudget::new(limits, &free).unwrap());
    assert_eq!(capped.charge(0, u64::MAX), Err(SearchStop::Memory));
}

#[test]
fn a_shared_budget_is_charged_by_every_pretreatment_subset_test() {
    use antecedent_core::{
        DistributionAvailability, EvidenceCatalog, EvidenceKind, EvidenceRegime, RegimeId,
        RegimeKind, SearchBudget,
    };
    // X, Y and 17 pretreatment parents of Y, the first selected: no standardizer
    // binds, so the subset search runs to its cap. Under its own limits that is an
    // obligation; under a shared budget every separation test is charged to it.
    let n = 19u32;
    let mut graph = Admg::with_variables(n);
    graph.insert_directed(d(0), d(1)).unwrap();
    for z in 2..n {
        graph.insert_directed(d(z), d(1)).unwrap();
    }
    let diagram = SelectionDiagram::try_new(graph, [v(2)]).unwrap();
    let query = ClassicalTransportQuery { outcomes: Arc::from([v(1)]), ..query() };
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
    let ctx = ExecutionContext::for_tests(1);
    let limits = SidLimits { steps: 2_000, depth: 256 };
    let own = identify_catalog_transport(&diagram, &query, &catalog, limits, &ctx).unwrap();
    assert!(matches!(own, CatalogTransportResult::MissingEvidence { .. }), "{own:?}");

    let budget = SearchBudget::new(SearchLimits { operations: 2_000, depth: 256 }, &ctx).unwrap();
    let mut search = SharedSearch::new(budget);
    let error =
        identify_catalog_transport_metered(&diagram, &query, &catalog, search.meter(), &ctx)
            .expect_err("the subset tests exhaust the shared budget");
    assert!(error.is_budget_or_cancel(), "{error:?}");
    assert_eq!(search.stop_of(&error), SearchStop::Operations);
}
#[test]
fn target_frontdoor_uses_recursive_multinode_district() {
    let mut graph = Admg::with_variables(3);
    graph.insert_directed(d(0), d(1)).unwrap();
    graph.insert_directed(d(1), d(2)).unwrap();
    graph.insert_bidirected(d(0), d(2)).unwrap();
    let diagram = SelectionDiagram::try_new(graph, [v(2)]).unwrap();
    let ctx = ExecutionContext::for_tests(1);
    let ClassicalTransportResult::Identified(proof) =
        identify_classical_transport(&diagram, &query(), SidLimits::default(), &ctx).unwrap()
    else {
        panic!("frontdoor must identify");
    };
    assert!(proof.rules().contains(&"sid.line8"));
    assert!(
        proof.arena().leaf_bindings(proof.root()).iter().all(|b| b.population.as_ref() == "target")
    );
    verify_classical_transport(&diagram, &query(), &proof, SidLimits::default(), &ctx).unwrap();
    let mut corrupted = proof.clone();
    corrupted.proof[corrupted.root_step].output = corrupted.proof[0].output;
    assert!(
        verify_classical_transport(&diagram, &query(), &corrupted, SidLimits::default(), &ctx)
            .is_err()
    );
}
#[test]
fn source_rescues_unconfounded_selection_mechanism_and_rejects_changed_premise() {
    let mut graph = Admg::with_variables(3);
    graph.insert_directed(d(0), d(2)).unwrap();
    graph.insert_bidirected(d(0), d(2)).unwrap();
    let diagram = SelectionDiagram::try_new(graph.clone(), []).unwrap();
    let ctx = ExecutionContext::for_tests(1);
    let ClassicalTransportResult::Identified(proof) =
        identify_classical_transport(&diagram, &query(), SidLimits::default(), &ctx).unwrap()
    else {
        panic!("direct transport");
    };
    assert!(proof.rules().contains(&"sid.line10"));
    let changed = SelectionDiagram::try_new(graph, [v(2)]).unwrap();
    assert!(
        verify_classical_transport(&changed, &query(), &proof, SidLimits::default(), &ctx).is_err()
    );
    assert!(
        identify_classical_transport(&diagram, &query(), SidLimits { steps: 1, depth: 1 }, &ctx)
            .is_err()
    );
    ctx.cancellation.cancel();
    assert!(identify_classical_transport(&diagram, &query(), SidLimits::default(), &ctx).is_err());
}
