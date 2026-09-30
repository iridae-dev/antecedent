//! 2.2B B1 (X2): ADMG conditional transport on the classical complete-source family.
//!
//! Decision, reduction, obstruction-candidate and budget evidence. Numerical truth
//! against enumerated latent SCMs lives in
//! crates/antecedent-estimate/tests/admg_conditional_transport_scm.rs.
#![allow(clippy::cast_possible_truncation, reason = "small graph coordinates")]

use antecedent_core::{
    DistributionAvailability, EvidenceCatalog, EvidenceKind, EvidenceRegime, ExecutionContext,
    Intervention, MemoryBudget, RegimeId, RegimeKind, SearchLimits, SearchStop, Value, VariableId,
};
use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram};
use antecedent_identify::{
    ADMG_CONDITIONAL_DEFAULT_LIMITS, ClassicalTransportQuery, ClassicalTransportResult,
    ConditionalObstructionCandidate, ConditionalTransportDecision, ConditionalTransportDerivation,
    ConditionalTransportQuery, IdcIdentifier, IdentificationError, IdentificationWorkspace,
    SidLimits, admg_conditional_refusal, decide_admg_conditional_transport,
    identify_classical_transport,
};
use std::sync::Arc;

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}
fn d(i: u32) -> DenseNodeId {
    DenseNodeId::from_raw(i)
}

fn graph(n: u32, directed: &[(u32, u32)], bidirected: &[(u32, u32)]) -> Admg {
    let mut graph = Admg::with_variables(n);
    for &(a, b) in directed {
        graph.insert_directed(d(a), d(b)).unwrap();
    }
    for &(a, b) in bidirected {
        graph.insert_bidirected(d(a), d(b)).unwrap();
    }
    graph
}

fn query(outcomes: &[u32], treatments: &[u32], conditioned: &[u32]) -> ConditionalTransportQuery {
    ConditionalTransportQuery {
        base: ClassicalTransportQuery {
            outcomes: outcomes.iter().copied().map(v).collect(),
            treatments: treatments.iter().copied().map(v).collect(),
            source: Arc::from("source"),
            target: Arc::from("target"),
        },
        conditioned_on: conditioned.iter().copied().map(v).collect(),
    }
}

/// Every source experiment `do(Z)` over `n` nodes plus the target observational
/// joint, except the source regimes whose intervention mask is in `skip`.
fn full_catalog(n: u32, skip: &[u32]) -> EvidenceCatalog {
    let mut regimes = Vec::new();
    for mask in 0..(1u32 << n) {
        if skip.contains(&mask) {
            continue;
        }
        let interventions: Vec<_> = (0..n).filter(|i| mask & (1 << i) != 0).map(v).collect();
        let measured: Vec<_> = (0..n).filter(|i| mask & (1 << i) == 0).map(v).collect();
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
            RegimeId::from_raw(1 << n),
            RegimeKind::Observational,
            EvidenceKind::Available,
            [],
            [],
            (0..n).map(v).collect::<Vec<_>>(),
            "target",
            DistributionAvailability::Joint,
        )
        .unwrap(),
    );
    EvidenceCatalog::try_new([], regimes, [], None).unwrap()
}

fn decide(
    diagram: &SelectionDiagram,
    q: &ConditionalTransportQuery,
    catalog: &EvidenceCatalog,
) -> ConditionalTransportDecision {
    let ctx = ExecutionContext::for_tests(1);
    decide_admg_conditional_transport(diagram, q, catalog, ADMG_CONDITIONAL_DEFAULT_LIMITS, &ctx)
        .unwrap()
}

/// X(0) -> Y(1), W(2) -> Y, X <-> Y, selection on W: W is a parent of Y with no
/// other path, so rule 2 moves it and the query is P*(y | do(x, w)), which the
/// source experiment on (X, W) answers (its mechanism difference is cut).
#[test]
fn a_movable_conditioning_variable_identifies_through_rule_two() {
    let diagram =
        SelectionDiagram::try_new(graph(3, &[(0, 1), (2, 1)], &[(0, 1)]), [v(2)]).unwrap();
    let q = query(&[1], &[0], &[2]);
    let ConditionalTransportDecision::Identified(bound) =
        decide(&diagram, &q, &full_catalog(3, &[]))
    else {
        panic!("a movable parent identifies");
    };
    let derivation = bound.derivation();
    assert_eq!(derivation.moves(), [v(2)]);
    assert!(derivation.remaining().is_empty());
    assert_eq!(*derivation.reduced_query().treatments, [v(0), v(2)]);
    assert_eq!(*derivation.reduced_query().outcomes, [v(1)]);
    // Nothing is left to condition on: the formula is the reduced joint itself.
    assert_eq!(derivation.root(), derivation.joint().root());
    // Selection on the confounded outcome: the joint needs the source experiment.
    assert!(bound.cited_leaves().iter().any(|(population, _)| population.as_ref() == "source"));
    let ctx = ExecutionContext::for_tests(1);
    derivation.recheck(&diagram, ADMG_CONDITIONAL_DEFAULT_LIMITS, &ctx).unwrap();
}

/// X(0) -> Y(1) -> W(2), X <-> Y, selection on W: W is a child of Y and cannot
/// move. The reduced joint P*(y, w | do(x)) mixes a source experiment (Y's
/// confounded factor) and the target law (W's selected mechanism).
#[test]
fn a_non_movable_conditioning_variable_is_normalized_out_of_the_joint() {
    let diagram =
        SelectionDiagram::try_new(graph(3, &[(0, 1), (1, 2)], &[(0, 1)]), [v(2)]).unwrap();
    let q = query(&[1], &[0], &[2]);
    let ConditionalTransportDecision::Identified(bound) =
        decide(&diagram, &q, &full_catalog(3, &[]))
    else {
        panic!("identified");
    };
    let derivation = bound.derivation();
    assert!(derivation.moves().is_empty());
    assert_eq!(derivation.remaining(), [v(2)]);
    assert_eq!(*derivation.reduced_query().outcomes, [v(1), v(2)]);
    assert_ne!(derivation.root(), derivation.joint().root(), "a ratio normalizes the joint");
    assert!(matches!(
        derivation.arena().node(derivation.root()),
        antecedent_expr::ExprNode::Ratio { numerator, .. } if *numerator == derivation.joint().root()
    ));
    let populations: Vec<_> =
        bound.cited_leaves().into_iter().map(|(p, _)| p.to_string()).collect();
    assert!(
        populations.contains(&"source".to_owned()) && populations.contains(&"target".to_owned())
    );
}

/// X(0) -> Y(1) -> W(2), X <-> Y, selection on Y: the reduced joint has an
/// s-hedge. It is `not_certified` with a verified candidate, never proven.
#[test]
fn a_reduced_joint_s_hedge_is_not_certified_never_proven() {
    let diagram =
        SelectionDiagram::try_new(graph(3, &[(0, 1), (1, 2)], &[(0, 1)]), [v(1)]).unwrap();
    let q = query(&[1], &[0], &[2]);
    let decision = decide(&diagram, &q, &full_catalog(3, &[]));
    assert_eq!(decision.reason_code(), Some("transport_not_certified"));
    assert_eq!(decision.detail_code(), Some("admg_transport.not_certified"));
    let ConditionalTransportDecision::NotCertified(inspection) = decision else {
        panic!("not certified");
    };
    assert_eq!(inspection.remaining, [v(2)]);
    let candidate = inspection.candidate.expect("the reduced joint's s-hedge is kept");
    assert_eq!(*candidate.reduced_query().outcomes, [v(1), v(2)]);
    let ctx = ExecutionContext::for_tests(1);
    candidate.recheck(&diagram, &q, &ctx).unwrap();
    // It round-trips as data and re-checks from the record alone.
    let json = serde_json::to_vec(&candidate.to_record()).unwrap();
    ConditionalObstructionCandidate::from_record_checked(
        serde_json::from_slice(&json).unwrap(),
        &diagram,
        &q,
        &ctx,
    )
    .unwrap();
}

#[test]
fn candidate_mutations_fail_verification() {
    let diagram =
        SelectionDiagram::try_new(graph(3, &[(0, 1), (1, 2)], &[(0, 1)]), [v(1)]).unwrap();
    let q = query(&[1], &[0], &[2]);
    let ConditionalTransportDecision::NotCertified(inspection) =
        decide(&diagram, &q, &full_catalog(3, &[]))
    else {
        panic!("not certified");
    };
    let candidate = inspection.candidate.unwrap();
    let ctx = ExecutionContext::for_tests(1);
    let check = |record| {
        ConditionalObstructionCandidate::from_record_checked(record, &diagram, &q, &ctx)
            .expect_err("mutated candidate")
    };
    let invalid = |error: IdentificationError| {
        assert_eq!(
            admg_conditional_refusal(&error),
            Some(("transport_not_certified", "admg_transport.invalid_derivation")),
            "{error:?}"
        );
    };
    // Dropped edge: the larger forest no longer spans one district.
    let mut record = candidate.to_record();
    assert!(!record.s_hedge.larger.bidirected.is_empty());
    record.s_hedge.larger.bidirected.clear();
    invalid(check(record));
    // Wrong root: without its directed edges the larger forest gains a root.
    let mut record = candidate.to_record();
    assert!(!record.s_hedge.larger.directed.is_empty());
    record.s_hedge.larger.directed.clear();
    invalid(check(record));
    // A non-movable conditioning variable claimed moved.
    let mut record = candidate.to_record();
    record.moves = vec![2];
    record.remaining.clear();
    invalid(check(record));
    // Changed selection premise: the same forest on a diagram without selections.
    let unselected = SelectionDiagram::try_new(graph(3, &[(0, 1), (1, 2)], &[(0, 1)]), []).unwrap();
    invalid(
        ConditionalObstructionCandidate::from_record_checked(
            candidate.to_record(),
            &unselected,
            &q,
            &ctx,
        )
        .unwrap_err(),
    );
    // Swapped query: outcome and conditioned variable exchanged.
    let swapped = query(&[2], &[0], &[1]);
    assert!(candidate.recheck(&diagram, &swapped, &ctx).is_err());
    invalid(
        ConditionalObstructionCandidate::from_record_checked(
            candidate.to_record(),
            &diagram,
            &swapped,
            &ctx,
        )
        .unwrap_err(),
    );
    // The unmutated record still checks.
    ConditionalObstructionCandidate::from_record_checked(candidate.to_record(), &diagram, &q, &ctx)
        .unwrap();
}

#[test]
fn a_missing_source_experiment_is_missing_evidence_naming_the_leaf() {
    // As in the non-movable case, Y's confounded factor needs the source do(X).
    let diagram =
        SelectionDiagram::try_new(graph(3, &[(0, 1), (1, 2)], &[(0, 1)]), [v(2)]).unwrap();
    let q = query(&[1], &[0], &[2]);
    // Drop every source regime that intervenes on X.
    let catalog = full_catalog(3, &[1, 3, 5, 7]);
    let decision = decide(&diagram, &q, &catalog);
    assert_eq!(decision.reason_code(), Some("transport_missing_evidence"));
    assert_eq!(decision.detail_code(), Some("admg_transport.missing_evidence"));
    let ConditionalTransportDecision::MissingEvidence { derivation, obligations } = decision else {
        panic!("missing evidence");
    };
    assert_eq!(derivation.remaining(), [v(2)]);
    assert!(
        obligations.iter().any(|o| o.contains("source") && o.contains("VariableId(0)")),
        "the unmet source leaf under do(X) is named: {obligations:?}"
    );
}

#[test]
fn graph_treatment_conditioned_and_limit_bounds_refuse_as_bounds_exceeded() {
    let ctx = ExecutionContext::for_tests(1);
    let bounds = Some(("route_not_supported", "admg_transport.bounds_exceeded"));
    let chain = |n: u32| {
        let edges: Vec<_> = (0..n - 1).map(|i| (i, i + 1)).collect();
        SelectionDiagram::try_new(graph(n, &edges, &[]), []).unwrap()
    };
    let run = |diagram: &SelectionDiagram, q: &ConditionalTransportQuery, limits| {
        decide_admg_conditional_transport(diagram, q, &full_catalog(3, &[]), limits, &ctx)
    };
    let limits = ADMG_CONDITIONAL_DEFAULT_LIMITS;
    // Six observed variables are admitted; seven refuse.
    let q = query(&[5], &[0], &[4]);
    assert!(run(&chain(6), &q, limits).is_ok());
    assert_eq!(admg_conditional_refusal(&run(&chain(7), &q, limits).unwrap_err()), bounds);
    // Three treatments and three conditioned variables are admitted; four refuse.
    assert!(run(&chain(6), &query(&[5], &[0, 1, 2], &[3]), limits).is_ok());
    let error = run(&chain(6), &query(&[5], &[0, 1, 2, 3], &[4]), limits).unwrap_err();
    assert_eq!(admg_conditional_refusal(&error), bounds);
    assert!(run(&chain(6), &query(&[5], &[0], &[1, 2, 3]), limits).is_ok());
    let error = run(&chain(6), &query(&[5], &[], &[0, 1, 2, 3]), limits).unwrap_err();
    assert_eq!(admg_conditional_refusal(&error), bounds);
    // Limits at the maxima run; one more operation or depth level refuses.
    let small = chain(3);
    let q = query(&[2], &[0], &[1]);
    assert!(run(&small, &q, limits).is_ok());
    for over in [
        SearchLimits { operations: limits.operations + 1, depth: limits.depth },
        SearchLimits { operations: limits.operations, depth: limits.depth + 1 },
    ] {
        assert_eq!(admg_conditional_refusal(&run(&small, &q, over).unwrap_err()), bounds);
    }
}

#[test]
fn invalid_queries_and_catalogs_refuse_by_name() {
    let ctx = ExecutionContext::for_tests(1);
    let diagram = SelectionDiagram::try_new(graph(3, &[(0, 1), (1, 2)], &[]), [v(2)]).unwrap();
    let catalog = full_catalog(3, &[]);
    let invalid = Some(("invalid_argument", "admg_transport.invalid_query"));
    let mut same_population = query(&[2], &[0], &[1]);
    same_population.base.source = Arc::from("target");
    for bad in [
        query(&[2], &[0], &[]),
        query(&[], &[0], &[1]),
        query(&[2], &[0], &[2]),
        query(&[2], &[0, 0], &[1]),
        query(&[9], &[0], &[1]),
        same_population,
    ] {
        let error = decide_admg_conditional_transport(
            &diagram,
            &bad,
            &catalog,
            ADMG_CONDITIONAL_DEFAULT_LIMITS,
            &ctx,
        )
        .unwrap_err();
        assert_eq!(admg_conditional_refusal(&error), invalid, "{bad:?}");
    }
    // A source environment whose selections disagree with the diagram.
    let environment = |population: &str, selections: Vec<VariableId>| {
        antecedent_core::Environment::try_new(
            population,
            (0..3)
                .map(|i| antecedent_core::VariableCoordinate {
                    variable: v(i),
                    domain: antecedent_core::VariableDomain::Binary,
                    unit: None,
                })
                .collect::<Vec<_>>(),
            selections,
        )
        .unwrap()
    };
    let catalog = EvidenceCatalog::try_new(
        [environment("source", vec![v(1)]), environment("target", vec![])],
        catalog.regimes.to_vec(),
        [],
        None,
    )
    .unwrap();
    let error = decide_admg_conditional_transport(
        &diagram,
        &query(&[2], &[0], &[1]),
        &catalog,
        ADMG_CONDITIONAL_DEFAULT_LIMITS,
        &ctx,
    )
    .unwrap_err();
    assert_eq!(
        admg_conditional_refusal(&error),
        Some(("invalid_argument", "admg_transport.invalid_catalog"))
    );
}

#[test]
fn a_tampered_record_fails_the_independent_checker() {
    let ctx = ExecutionContext::for_tests(1);
    let limits = ADMG_CONDITIONAL_DEFAULT_LIMITS;
    let invalid = Some(("transport_not_certified", "admg_transport.invalid_derivation"));
    // Movable case: W(2) -> Y(1) <- X(0), X <-> Y, selection on W.
    let diagram =
        SelectionDiagram::try_new(graph(3, &[(0, 1), (2, 1)], &[(0, 1)]), [v(2)]).unwrap();
    let q = query(&[1], &[0], &[2]);
    let ConditionalTransportDecision::Identified(bound) =
        decide(&diagram, &q, &full_catalog(3, &[]))
    else {
        panic!("identified");
    };
    let derivation = bound.derivation();
    let arena = derivation.joint().arena().clone();
    let check = |record, q: &ConditionalTransportQuery, diagram: &SelectionDiagram| {
        ConditionalTransportDerivation::from_record_checked(
            record,
            arena.clone(),
            diagram,
            q,
            limits,
            &ctx,
        )
    };
    check(derivation.to_record(), &q, &diagram).unwrap();
    // Not moving a movable variable breaks maximality. The dropped move is paired
    // with a VALID sID derivation of the joint it implies, P*(y, w | do(x)), so
    // only the maximality check can refuse it.
    let implied = ClassicalTransportQuery {
        outcomes: Arc::from([v(1), v(2)]),
        treatments: Arc::from([v(0)]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    };
    let ClassicalTransportResult::Identified(implied_joint) =
        identify_classical_transport(&diagram, &implied, SidLimits::default(), &ctx).unwrap()
    else {
        panic!("the implied joint has a valid derivation");
    };
    let mut record = derivation.to_record();
    record.moves.clear();
    record.remaining = vec![2];
    record.joint = implied_joint.to_record();
    let error = ConditionalTransportDerivation::from_record_checked(
        record,
        implied_joint.arena().clone(),
        &diagram,
        &q,
        limits,
        &ctx,
    )
    .unwrap_err();
    assert_eq!(admg_conditional_refusal(&error), invalid);
    // A different conditioned set.
    let mut record = derivation.to_record();
    record.conditioned_on = vec![0];
    assert_eq!(admg_conditional_refusal(&check(record, &q, &diagram).unwrap_err()), invalid);
    // A tampered joint proof step.
    let mut record = derivation.to_record();
    let last = record.joint.steps.len() - 1;
    record.joint.steps[last].output = record.joint.steps[last].kernel;
    assert_eq!(admg_conditional_refusal(&check(record, &q, &diagram).unwrap_err()), invalid);
    // The joint of another query: the proof's query is not the reduced one.
    let mut record = derivation.to_record();
    record.joint.treatments = vec![0];
    assert_eq!(admg_conditional_refusal(&check(record, &q, &diagram).unwrap_err()), invalid);
    // A different graph: the separation premise and the signature both change.
    let confounded =
        SelectionDiagram::try_new(graph(3, &[(0, 1), (2, 1)], &[(0, 1), (1, 2)]), [v(2)]).unwrap();
    assert_eq!(
        admg_conditional_refusal(&check(derivation.to_record(), &q, &confounded).unwrap_err()),
        invalid
    );
    // Non-movable case: claiming W moved fails its rule-2 premise.
    let diagram =
        SelectionDiagram::try_new(graph(3, &[(0, 1), (1, 2)], &[(0, 1)]), [v(2)]).unwrap();
    let ConditionalTransportDecision::Identified(bound) =
        decide(&diagram, &q, &full_catalog(3, &[]))
    else {
        panic!("identified");
    };
    let mut record = bound.derivation().to_record();
    record.moves = vec![2];
    record.remaining.clear();
    let error = ConditionalTransportDerivation::from_record_checked(
        record,
        bound.derivation().joint().arena().clone(),
        &diagram,
        &q,
        limits,
        &ctx,
    )
    .unwrap_err();
    assert_eq!(admg_conditional_refusal(&error), invalid);
}

/// Every three-node ADMG (edges low -> high) as `(directed, bidirected)` edge lists.
fn three_node_admgs() -> Vec<(Vec<(u32, u32)>, Vec<(u32, u32)>)> {
    let pairs = [(0u32, 1u32), (0, 2), (1, 2)];
    let edges = |mask: u32| -> Vec<(u32, u32)> {
        pairs.iter().enumerate().filter(|(i, _)| mask & (1 << i) != 0).map(|(_, e)| *e).collect()
    };
    (0..8u32).flat_map(|d| (0..8u32).map(move |b| (edges(d), edges(b)))).collect()
}

/// Every query over three nodes: one outcome, one treatment or none, and a
/// non-empty conditioned subset of the rest.
fn three_node_queries() -> Vec<ConditionalTransportQuery> {
    let mut queries = Vec::new();
    for y in 0..3u32 {
        for x in (0..3u32).map(Some).chain([None]) {
            if x == Some(y) {
                continue;
            }
            let treatments: Vec<u32> = x.into_iter().collect();
            let rest: Vec<u32> = (0..3).filter(|i| *i != y && Some(*i) != x).collect();
            for mask in 1..(1u32 << rest.len()) {
                let conditioned: Vec<u32> = rest
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| mask & (1 << i) != 0)
                    .map(|(_, node)| *node)
                    .collect();
                queries.push(query(&[y], &treatments, &conditioned));
            }
        }
    }
    queries
}

/// Every selection pattern over three nodes.
fn three_node_selections() -> Vec<Vec<VariableId>> {
    (0..8u32).map(|mask| (0..3).filter(|i| mask & (1 << i) != 0).map(v).collect()).collect()
}

/// `(moves, remaining)` of a finished decision.
fn reduction_of(decision: ConditionalTransportDecision) -> (Vec<VariableId>, Vec<VariableId>) {
    match decision {
        ConditionalTransportDecision::Identified(bound) => {
            (bound.derivation().moves().to_vec(), bound.derivation().remaining().to_vec())
        }
        ConditionalTransportDecision::MissingEvidence { derivation, .. } => {
            (derivation.moves().to_vec(), derivation.remaining().to_vec())
        }
        ConditionalTransportDecision::NotCertified(inspection) => {
            (inspection.moves, inspection.remaining)
        }
        ConditionalTransportDecision::Exhausted(receipt) => {
            panic!("uncapped small graph exhausted: {receipt:?}")
        }
    }
}

/// Every outcome `decide` returns carries a reduction the independent checker
/// accepts: an identified or missing-evidence derivation re-checks from its
/// record, a not-certified candidate re-checks too. Every three-node ADMG, every
/// selection pattern, every query.
#[test]
fn every_decided_reduction_passes_the_independent_checker() {
    let ctx = ExecutionContext::for_tests(1);
    let limits = ADMG_CONDITIONAL_DEFAULT_LIMITS;
    let catalog = full_catalog(3, &[]);
    let (mut identified, mut candidates, mut moved) = (0usize, 0usize, 0usize);
    for (directed, bidirected) in three_node_admgs() {
        for selections in three_node_selections() {
            let diagram =
                SelectionDiagram::try_new(graph(3, &directed, &bidirected), selections).unwrap();
            for q in three_node_queries() {
                match decide_admg_conditional_transport(&diagram, &q, &catalog, limits, &ctx)
                    .unwrap()
                {
                    ConditionalTransportDecision::Identified(bound) => {
                        let derivation = bound.derivation();
                        ConditionalTransportDerivation::from_record_checked(
                            derivation.to_record(),
                            derivation.joint().arena().clone(),
                            &diagram,
                            &q,
                            limits,
                            &ctx,
                        )
                        .unwrap_or_else(|e| panic!("{q:?} on {directed:?}/{bidirected:?}: {e:?}"));
                        identified += 1;
                        moved += usize::from(!derivation.moves().is_empty());
                    }
                    ConditionalTransportDecision::MissingEvidence { derivation, .. } => {
                        derivation.recheck(&diagram, limits, &ctx).unwrap();
                    }
                    ConditionalTransportDecision::NotCertified(inspection) => {
                        if let Some(candidate) = inspection.candidate {
                            candidate.recheck(&diagram, &q, &ctx).unwrap();
                            candidates += 1;
                        }
                    }
                    ConditionalTransportDecision::Exhausted(receipt) => {
                        panic!("uncapped small graph exhausted: {receipt:?}")
                    }
                }
            }
        }
    }
    eprintln!("checked: {identified} identified ({moved} with moves), {candidates} candidates");
    assert!(
        identified > 5_000 && moved > 1_000 && candidates > 100,
        "{identified} {moved} {candidates}"
    );
}

/// The moved set does not depend on the selection targets (see the module note
/// in crates/antecedent-identify/src/sid/conditional.rs: selection nodes are
/// parentless and conditioned on, so they neither open nor keep open a path).
#[test]
fn the_moved_set_is_selection_independent() {
    let ctx = ExecutionContext::for_tests(1);
    let catalog = full_catalog(3, &[]);
    let decide_with = |diagram: &SelectionDiagram, q: &ConditionalTransportQuery| {
        reduction_of(
            decide_admg_conditional_transport(
                diagram,
                q,
                &catalog,
                ADMG_CONDITIONAL_DEFAULT_LIMITS,
                &ctx,
            )
            .unwrap(),
        )
    };
    // Paired case: W(2) -> Y(1) <- X(0), X <-> Y. W moves with no selection and
    // with a selection on W itself.
    let g = graph(3, &[(0, 1), (2, 1)], &[(0, 1)]);
    let q = query(&[1], &[0], &[2]);
    let unselected = SelectionDiagram::try_new(g.clone(), []).unwrap();
    let selected = SelectionDiagram::try_new(g, [v(2)]).unwrap();
    assert_eq!(decide_with(&unselected, &q), (vec![v(2)], vec![]));
    assert_eq!(decide_with(&selected, &q), (vec![v(2)], vec![]));
    // Every three-node ADMG, query and selection pattern: the moves equal the
    // unselected diagram's.
    let (mut compared, mut selected_moved) = (0usize, 0usize);
    for (directed, bidirected) in three_node_admgs() {
        let g = graph(3, &directed, &bidirected);
        let unselected = SelectionDiagram::try_new(g.clone(), []).unwrap();
        for q in three_node_queries() {
            let expected = decide_with(&unselected, &q);
            for selections in three_node_selections() {
                let diagram = SelectionDiagram::try_new(g.clone(), selections.clone()).unwrap();
                let got = decide_with(&diagram, &q);
                assert_eq!(got, expected, "{directed:?}/{bidirected:?} {q:?} S={selections:?}");
                compared += 1;
                selected_moved += usize::from(got.0.iter().any(|m| selections.contains(m)));
            }
        }
    }
    assert!(compared > 10_000, "{compared}");
    assert!(selected_moved > 1_000, "a selection on a moved variable: {selected_moved}");
}

/// Moves the repo's IDC identifier made, parsed from its derivation trace.
fn idc_moves(graph: &Admg, y: &[u32], x: &[u32], w: &[u32]) -> Vec<u32> {
    let idc = IdcIdentifier::new();
    let prepared = idc.prepare(graph).unwrap();
    let interventions: Vec<_> =
        x.iter().map(|t| Intervention::set(v(*t), Value::Int64(0))).collect();
    let outcomes: Vec<_> = y.iter().copied().map(v).collect();
    let conditioning: Vec<_> = w.iter().copied().map(v).collect();
    let result = idc
        .identify_conditional(
            &prepared,
            &outcomes,
            &interventions,
            &conditioning,
            &mut IdentificationWorkspace::default(),
        )
        .unwrap();
    result
        .derivation
        .steps
        .iter()
        .filter(|step| step.rule.as_ref() == "general.idc.line1")
        .map(|step| {
            let detail = step.detail.as_ref();
            let start = detail.find("Z=").unwrap() + 2;
            let end = detail[start..].find(' ').unwrap() + start;
            detail[start..end].parse().unwrap()
        })
        .collect()
}

#[test]
fn the_moved_set_equals_the_repo_idc_moves() {
    let ctx = ExecutionContext::for_tests(1);
    let mut compared = 0usize;
    let mut moved_some = 0usize;
    let mut kept_some = 0usize;
    let mut partial = 0usize;
    // Every three-node ADMG (edges low -> high) under every selection pattern,
    // and a stride of four-node ADMGs with selection on node 0, under every role
    // assignment of one outcome, one treatment (or none) and a conditioned subset
    // of the rest.
    for n in [3u32, 4] {
        let pairs: Vec<(u32, u32)> = (0..n).flat_map(|a| (a + 1..n).map(move |b| (a, b))).collect();
        let masks = 1u32 << pairs.len();
        let stride = if n == 3 { 1 } else { 3 };
        let catalog = full_catalog(n, &[]);
        for directed in (0..masks).step_by(stride) {
            for bidirected in (0..masks).step_by(stride) {
                let edges = |mask: u32| -> Vec<(u32, u32)> {
                    pairs
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| mask & (1 << i) != 0)
                        .map(|(_, e)| *e)
                        .collect()
                };
                let g = graph(n, &edges(directed), &edges(bidirected));
                // Every selection pattern on three nodes; selection {0} on four.
                let patterns: Vec<Vec<VariableId>> =
                    if n == 3 { three_node_selections() } else { vec![vec![v(0)]] };
                for selections in patterns {
                    let diagram = SelectionDiagram::try_new(g.clone(), selections).unwrap();
                    for y in 0..n {
                        for x in (0..n).map(Some).chain([None]) {
                            if x == Some(y) {
                                continue;
                            }
                            let treatments: Vec<u32> = x.into_iter().collect();
                            let rest: Vec<u32> =
                                (0..n).filter(|i| *i != y && Some(*i) != x).collect();
                            // Every non-empty conditioned subset of the rest, so nodes
                            // outside the query (neither conditioned nor intervened)
                            // are exercised too.
                            for mask in 1..(1u32 << rest.len()) {
                                let conditioned: Vec<u32> = rest
                                    .iter()
                                    .enumerate()
                                    .filter(|(i, _)| mask & (1 << i) != 0)
                                    .map(|(_, node)| *node)
                                    .collect();
                                let q = query(&[y], &treatments, &conditioned);
                                let decision = decide_admg_conditional_transport(
                                    &diagram,
                                    &q,
                                    &catalog,
                                    ADMG_CONDITIONAL_DEFAULT_LIMITS,
                                    &ctx,
                                )
                                .unwrap();
                                let (moves, remaining) = reduction_of(decision);
                                let ours: Vec<u32> = moves.iter().map(|m| m.raw()).collect();
                                assert_eq!(
                                    ours,
                                    idc_moves(&g, &[y], &treatments, &conditioned),
                                    "n={n} directed={directed} bidirected={bidirected} y={y} x={x:?} w={conditioned:?}"
                                );
                                compared += 1;
                                moved_some += usize::from(!moves.is_empty());
                                kept_some += usize::from(!remaining.is_empty());
                                partial += usize::from(conditioned.len() < rest.len());
                            }
                        }
                    }
                }
            }
        }
    }
    eprintln!(
        "idc agreement: {compared} compared, {moved_some} moved, {kept_some} kept, {partial} partial"
    );
    assert!(compared > 20_000, "{compared}");
    assert!(moved_some > 5_000 && kept_some > 5_000, "{moved_some} {kept_some}");
    assert!(partial > 10_000, "queries leaving a node outside the query: {partial}");
}

#[test]
fn query_coordinate_order_does_not_change_the_decision() {
    // Two outcomes, two conditioned variables, in both declaration orders.
    let diagram =
        SelectionDiagram::try_new(graph(5, &[(0, 1), (1, 2), (3, 1), (2, 4)], &[(0, 2)]), [v(4)])
            .unwrap();
    let catalog = full_catalog(5, &[]);
    let forward = decide(&diagram, &query(&[1, 2], &[0], &[3, 4]), &catalog);
    let backward = decide(&diagram, &query(&[2, 1], &[0], &[4, 3]), &catalog);
    let summary = |decision: &ConditionalTransportDecision| match decision {
        ConditionalTransportDecision::Identified(bound) => {
            let mut moves = bound.derivation().moves().to_vec();
            let mut remaining = bound.derivation().remaining().to_vec();
            moves.sort_unstable();
            remaining.sort_unstable();
            ("identified", moves, remaining)
        }
        other => panic!("identified expected: {other:?}"),
    };
    let (kind, moves, remaining) = summary(&forward);
    assert_eq!((kind, moves.clone(), remaining.clone()), summary(&backward));
    assert_eq!(moves, [v(3)], "the unconfounded parent of Y1 moves");
    assert_eq!(remaining, [v(4)], "the selected descendant stays conditioned");
}

#[test]
fn an_operation_stop_is_a_receipt_never_a_verdict() {
    let ctx = ExecutionContext::for_tests(1);
    let diagram =
        SelectionDiagram::try_new(graph(3, &[(0, 1), (1, 2)], &[(0, 1)]), [v(2)]).unwrap();
    let q = query(&[1], &[0], &[2]);
    let catalog = full_catalog(3, &[]);
    let mut identified_at = None;
    for operations in 1..200 {
        let limits = SearchLimits { operations, depth: 24 };
        match decide_admg_conditional_transport(&diagram, &q, &catalog, limits, &ctx).unwrap() {
            ConditionalTransportDecision::Exhausted(receipt) => {
                assert_eq!(receipt.stop, SearchStop::Operations);
                assert_eq!(receipt.operations_limit, operations);
                assert_eq!(receipt.operations_consumed, Some(operations));
                assert!(receipt.memory_limit_bytes.is_some());
                // Explored and unevaluated stages partition the three stages.
                assert_eq!(receipt.explored.len() + receipt.unevaluated.len(), 3, "{receipt:?}");
                assert!(receipt.unevaluated.iter().all(|s| s.starts_with("stage:")));
            }
            ConditionalTransportDecision::Identified(_) => {
                identified_at = Some(operations);
                break;
            }
            other => panic!("a budget never yields a verdict: {other:?}"),
        }
    }
    assert!(identified_at.is_some_and(|at| at > 3), "{identified_at:?}");
}

#[test]
fn depth_and_memory_stops_are_receipts_and_memory_is_checked_on_every_charge() {
    let diagram =
        SelectionDiagram::try_new(graph(3, &[(0, 1), (1, 2)], &[(0, 1)]), [v(2)]).unwrap();
    let q = query(&[1], &[0], &[2]);
    let catalog = full_catalog(3, &[]);
    let ctx = ExecutionContext::for_tests(1);
    // Depth: the reduction charges at depth 1, the sID recursion deeper.
    let ConditionalTransportDecision::Exhausted(receipt) = decide_admg_conditional_transport(
        &diagram,
        &q,
        &catalog,
        SearchLimits { operations: 4096, depth: 1 },
        &ctx,
    )
    .unwrap() else {
        panic!("depth stop");
    };
    assert_eq!(receipt.stop, SearchStop::Depth);
    assert_eq!(receipt.depth_limit, 1);
    assert_eq!(receipt.explored, ["stage:rule_two_reduction"]);
    // Memory: a hard limit below one live state stops at the first charge.
    let mut tight = ExecutionContext::for_tests(1);
    tight.memory = MemoryBudget { soft_limit_bytes: None, hard_limit_bytes: Some(600) };
    let ConditionalTransportDecision::Exhausted(receipt) = decide_admg_conditional_transport(
        &diagram,
        &q,
        &catalog,
        ADMG_CONDITIONAL_DEFAULT_LIMITS,
        &tight,
    )
    .unwrap() else {
        panic!("memory stop");
    };
    assert_eq!(receipt.stop, SearchStop::Memory);
    assert_eq!(receipt.memory_limit_bytes, Some(600));
    assert_eq!(receipt.unevaluated.len(), 3);
    // A limit the reduction fits but the cumulative stages do not: the stop lands
    // in a later stage, so memory was charged after the first stage too.
    let mut later = ExecutionContext::for_tests(1);
    later.memory = MemoryBudget { soft_limit_bytes: None, hard_limit_bytes: Some(4_000) };
    let ConditionalTransportDecision::Exhausted(receipt) = decide_admg_conditional_transport(
        &diagram,
        &q,
        &catalog,
        ADMG_CONDITIONAL_DEFAULT_LIMITS,
        &later,
    )
    .unwrap() else {
        panic!("memory stop in a later stage");
    };
    assert_eq!(receipt.stop, SearchStop::Memory);
    assert!(receipt.explored.contains(&"stage:rule_two_reduction".to_owned()), "{receipt:?}");
    // Cancellation before the search is a receipt too.
    let cancelled = ExecutionContext::for_tests(1);
    cancelled.cancellation.cancel();
    let ConditionalTransportDecision::Exhausted(receipt) = decide_admg_conditional_transport(
        &diagram,
        &q,
        &catalog,
        ADMG_CONDITIONAL_DEFAULT_LIMITS,
        &cancelled,
    )
    .unwrap() else {
        panic!("cancelled");
    };
    assert_eq!(receipt.stop, SearchStop::Cancelled);
}

/// A forged reduction paired with a VALID sID derivation of the query it implies:
/// only the independent rule-2 checks can refuse it.
#[test]
fn forged_reductions_with_valid_joints_fail_the_independent_checker() {
    use antecedent_identify::ConditionalTransportRecord;
    let ctx = ExecutionContext::for_tests(1);
    let invalid = Some(("transport_not_certified", "admg_transport.invalid_derivation"));
    let q = query(&[1], &[0], &[2]);
    let forge =
        |diagram: &SelectionDiagram, outcomes: &[u32], treatments: &[u32], moves, remaining| {
            let reduced = ClassicalTransportQuery {
                outcomes: outcomes.iter().copied().map(v).collect(),
                treatments: treatments.iter().copied().map(v).collect(),
                source: Arc::from("source"),
                target: Arc::from("target"),
            };
            let ClassicalTransportResult::Identified(joint) =
                identify_classical_transport(diagram, &reduced, SidLimits::default(), &ctx)
                    .unwrap()
            else {
                panic!("the forged reduced joint is itself a valid sID derivation");
            };
            let record = ConditionalTransportRecord {
                conditioned_on: vec![2],
                moves,
                remaining,
                joint: joint.to_record(),
            };
            ConditionalTransportDerivation::from_record_checked(
                record,
                joint.arena().clone(),
                diagram,
                &q,
                ADMG_CONDITIONAL_DEFAULT_LIMITS,
                &ctx,
            )
        };
    // W(2) is a descendant of Y(1): moving it is an unsound rule-2 step, although
    // P*(y | do(x, w)) itself has a valid sID derivation.
    let descendant =
        SelectionDiagram::try_new(graph(3, &[(0, 1), (1, 2)], &[(0, 1)]), [v(2)]).unwrap();
    let error = forge(&descendant, &[1], &[0, 2], vec![2], vec![]).unwrap_err();
    assert_eq!(admg_conditional_refusal(&error), invalid);
    // The honest reduction of the same graph checks.
    forge(&descendant, &[1, 2], &[0], vec![], vec![2]).unwrap();
    // W(2) is an unconfounded parent of Y(1): leaving it conditioned breaks the
    // maximality of the moved set, although P*(y, w | do(x)) has a valid derivation.
    let parent = SelectionDiagram::try_new(graph(3, &[(0, 1), (2, 1)], &[(0, 1)]), [v(2)]).unwrap();
    let error = forge(&parent, &[1, 2], &[0], vec![], vec![2]).unwrap_err();
    assert_eq!(admg_conditional_refusal(&error), invalid);
    forge(&parent, &[1], &[0, 2], vec![2], vec![]).unwrap();
}
