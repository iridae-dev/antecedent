//! Bounded DAG completions of a supplied CPDAG (2.3A X2).
//!
//! The expected completion sets come from an independent brute force: every
//! orientation of the skeleton, filtered to the acyclic, v-structure-preserving
//! ones that keep the CPDAG's directed edges. Nothing here reuses the library's
//! enumeration.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;
use std::sync::Arc;

use antecedent_core::{
    CancellationToken, DistributionAvailability, EvidenceCatalog, EvidenceKind, EvidenceRegime,
    ExecutionContext, MemoryBudget, RegimeId, RegimeKind, SearchLimits, SearchStop, VariableDomain,
    VariableId,
};
use antecedent_graph::{Cpdag, DenseNodeId, NodeRef};
use antecedent_identify::ClassicalTransportQuery;
use antecedent_identify::sid::cpdag_completion::{
    CPDAG_BOUNDS_EXCEEDED_DETAIL, CPDAG_EVIDENCE_MISMATCH_DETAIL, CPDAG_NO_EVIDENCE_DETAIL,
    CPDAG_NOT_A_CPDAG_DETAIL, CPDAG_SELECTION_OR_LATENT_DETAIL, CompletionEvidenceBinding,
    CpdagCompletionDecision, CpdagCompletionInput, CpdagEnumeration, CpdagEvidence,
    CpdagScenarioError, decide_cpdag_completions, enumerate_cpdag_completions,
};
use antecedent_identify::sid::scenarios::{ScenarioCoordinate, ScenarioOutcome};

const BIG: SearchLimits = SearchLimits { operations: 1_000_000, depth: 64 };

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

fn d(i: u32) -> DenseNodeId {
    DenseNodeId::from_raw(i)
}

type Edges = Vec<(u32, u32)>;

struct Case {
    name: &'static str,
    nodes: u32,
    undirected: Edges,
    directed: Edges,
    total: usize,
}

fn cases() -> Vec<Case> {
    let case = |name, nodes, undirected: &[(u32, u32)], directed: &[(u32, u32)], total| Case {
        name,
        nodes,
        undirected: undirected.to_vec(),
        directed: directed.to_vec(),
        total,
    };
    vec![
        case("chain", 3, &[(0, 1), (1, 2)], &[], 3),
        case("collider", 3, &[], &[(0, 1), (2, 1)], 1),
        case("triangle", 3, &[(0, 1), (1, 2), (0, 2)], &[], 6),
        // An undirected tree has one completion per root.
        case("star_five", 5, &[(0, 1), (0, 2), (0, 3), (0, 4)], &[], 5),
        case("collider_and_pair", 5, &[(3, 4)], &[(0, 2), (1, 2)], 2),
        // A 3-path (3 completions) beside a triangle (6): 18.
        case("six_nodes", 6, &[(0, 1), (1, 2), (3, 4), (4, 5), (3, 5)], &[], 18),
    ]
}

fn cpdag(case: &Case) -> Cpdag {
    let mut graph = Cpdag::with_variables(case.nodes);
    for (a, b) in &case.undirected {
        graph.insert_undirected(d(*a), d(*b)).unwrap();
    }
    for (a, b) in &case.directed {
        graph.insert_directed(d(*a), d(*b)).unwrap();
    }
    graph
}

fn acyclic(nodes: u32, edges: &[(u32, u32)]) -> bool {
    let mut indegree = vec![0_usize; nodes as usize];
    for (_, child) in edges {
        indegree[*child as usize] += 1;
    }
    let mut ready = (0..nodes).filter(|i| indegree[*i as usize] == 0).collect::<Vec<_>>();
    let mut seen = 0;
    while let Some(node) = ready.pop() {
        seen += 1;
        for (parent, child) in edges {
            if *parent == node {
                indegree[*child as usize] -= 1;
                if indegree[*child as usize] == 0 {
                    ready.push(*child);
                }
            }
        }
    }
    seen == nodes as usize
}

/// Every orientation of the skeleton that keeps the CPDAG's directed edges, has
/// no directed cycle and has exactly the CPDAG's unshielded colliders.
fn oracle(case: &Case) -> BTreeSet<Edges> {
    let skeleton = case
        .undirected
        .iter()
        .chain(&case.directed)
        .map(|(a, b)| (*a.min(b), *a.max(b)))
        .collect::<Vec<_>>();
    let adjacent = |x: u32, y: u32| skeleton.contains(&(x.min(y), x.max(y)));
    let colliders = |edges: &[(u32, u32)]| {
        let mut out = BTreeSet::new();
        for (x, y) in edges {
            for (z, y2) in edges {
                if y == y2 && x < z && !adjacent(*x, *z) {
                    out.insert((*x, *y, *z));
                }
            }
        }
        out
    };
    let target = colliders(&case.directed);
    let mut result = BTreeSet::new();
    for mask in 0..(1_u32 << skeleton.len()) {
        let edges = skeleton
            .iter()
            .enumerate()
            .map(|(i, (a, b))| if (mask >> i) & 1 == 1 { (*a, *b) } else { (*b, *a) })
            .collect::<Vec<_>>();
        let keeps_directed = case.directed.iter().all(|e| edges.contains(e));
        if keeps_directed && colliders(&edges) == target && acyclic(case.nodes, &edges) {
            let mut sorted = edges;
            sorted.sort_unstable();
            result.insert(sorted);
        }
    }
    result
}

fn raw(edges: &[(VariableId, VariableId)]) -> Edges {
    edges.iter().map(|(a, b)| (a.raw(), b.raw())).collect()
}

fn enumerate(case: &Case) -> CpdagEnumeration {
    enumerate_cpdag_completions(
        &CpdagCompletionInput::new(cpdag(case)),
        BIG,
        &ExecutionContext::for_tests(1),
    )
    .unwrap()
}

fn found(enumeration: &CpdagEnumeration) -> BTreeSet<Edges> {
    enumeration.completions.iter().map(|c| raw(&c.edges)).collect()
}

fn case_named(name: &str) -> Case {
    cases().into_iter().find(|c| c.name == name).unwrap()
}

#[test]
fn x2_cpdag_enumerated_truth_sets() {
    for case in cases() {
        let enumeration = enumerate(&case);
        let expected = oracle(&case);
        assert_eq!(expected.len(), case.total, "{}: the oracle count", case.name);
        assert_eq!(found(&enumeration), expected, "{}: completion sets", case.name);
        assert_eq!(enumeration.completions.len(), case.total, "{}", case.name);
        assert!(enumeration.is_exportable() && enumeration.not_enumerated == 0, "{}", case.name);
        assert_eq!(enumeration.total(), case.total);
        // Identities are 64 hex characters, distinct, and in ascending order.
        let ids =
            enumeration.completions.iter().map(|c| c.identity.to_string()).collect::<Vec<_>>();
        assert!(ids.iter().all(|id| id.len() == 64 && id.bytes().all(|b| b.is_ascii_hexdigit())));
        assert!(ids.windows(2).all(|w| w[0] < w[1]), "{}: identity order", case.name);
        // Each completion's edge list is sorted and keeps the CPDAG's directed edges.
        for completion in &enumeration.completions {
            let edges = raw(&completion.edges);
            assert!(edges.windows(2).all(|w| w[0] < w[1]));
            assert!(case.directed.iter().all(|e| edges.contains(e)));
        }
    }
    // The chain's three completions, by name.
    let chain = found(&enumerate(&case_named("chain")));
    let expected: BTreeSet<Edges> =
        [vec![(0, 1), (1, 2)], vec![(1, 0), (1, 2)], vec![(1, 0), (2, 1)]].into_iter().collect();
    assert_eq!(chain, expected);
}

fn binary_coordinates(n: u32) -> Arc<[ScenarioCoordinate]> {
    (0..n)
        .map(|i| ScenarioCoordinate {
            variable: v(i),
            name: Arc::from(format!("v{i}")),
            domain: VariableDomain::Binary,
            unit: None,
        })
        .collect()
}

fn query() -> ClassicalTransportQuery {
    // do(v1) -> v2 in a chain v0 - v1 - v2.
    ClassicalTransportQuery {
        outcomes: Arc::from([v(2)]),
        treatments: Arc::from([v(1)]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    }
}

fn target_catalog(measured: &[u32]) -> EvidenceCatalog {
    let regime = EvidenceRegime::try_new(
        RegimeId::from_raw(0),
        RegimeKind::Observational,
        EvidenceKind::Available,
        Vec::<VariableId>::new(),
        [],
        measured.iter().copied().map(v).collect::<Vec<_>>(),
        "target",
        DistributionAvailability::Joint,
    )
    .unwrap();
    EvidenceCatalog::try_new([], vec![regime], [], None).unwrap()
}

const SEARCH: SearchLimits = SearchLimits { operations: 100_000, depth: 256 };

fn decide(evidence: &CpdagEvidence) -> Result<CpdagCompletionDecision, CpdagScenarioError> {
    decide_cpdag_completions(
        &CpdagCompletionInput::new(cpdag(&case_named("chain"))),
        &binary_coordinates(3),
        &query(),
        evidence,
        SEARCH,
        &ExecutionContext::for_tests(1),
    )
}

fn id_of(enumeration: &CpdagEnumeration, edges: &[(u32, u32)]) -> Arc<str> {
    let wanted = edges.to_vec();
    let found = enumeration.completions.iter().find(|c| raw(&c.edges) == wanted).unwrap();
    Arc::clone(&found.identity)
}

fn binding(
    id: &Arc<str>,
    certified: &Arc<str>,
    name: &str,
    measured: &[u32],
) -> CompletionEvidenceBinding {
    CompletionEvidenceBinding {
        completion: Arc::clone(id),
        certified_for: Arc::clone(certified),
        evidence_identity: Arc::from(name),
        catalog: target_catalog(measured),
    }
}

fn status_of(decision: &CpdagCompletionDecision, edges: &[(u32, u32)]) -> &'static str {
    let wanted = edges.to_vec();
    let index = decision.completions.iter().position(|c| raw(&c.edges) == wanted).unwrap();
    decision.decision.as_ref().unwrap().decisions[index].outcome.status()
}

fn refusal(error: CpdagScenarioError) -> (&'static str, &'static str) {
    match error {
        CpdagScenarioError::Refused(r) => (r.code, r.detail),
        other @ CpdagScenarioError::Identification(_) => panic!("expected a refusal, got {other}"),
    }
}

#[test]
fn x2_cpdag_missing_evidence_refusals() {
    let ctx = ExecutionContext::for_tests(1);
    // More than six nodes.
    let mut seven = Cpdag::with_variables(7);
    for i in 0..6 {
        seven.insert_undirected(d(i), d(i + 1)).unwrap();
    }
    let refused =
        enumerate_cpdag_completions(&CpdagCompletionInput::new(seven), BIG, &ctx).unwrap_err();
    assert_eq!(
        (refused.code, refused.detail),
        ("route_not_supported", CPDAG_BOUNDS_EXCEEDED_DETAIL)
    );
    assert_eq!(CPDAG_BOUNDS_EXCEEDED_DETAIL, "cpdag_scenarios.bounds_exceeded");
    // A selection target and a latent pair.
    let chain = || CpdagCompletionInput::new(cpdag(&case_named("chain")));
    let mut selected = chain();
    selected.selection_targets = Arc::from([v(1)]);
    let mut latent = chain();
    latent.latent_pairs = Arc::from([(v(0), v(2))]);
    for input in [selected, latent] {
        let refused = enumerate_cpdag_completions(&input, BIG, &ctx).unwrap_err();
        assert_eq!(
            (refused.code, refused.detail),
            ("route_not_supported", CPDAG_SELECTION_OR_LATENT_DETAIL)
        );
    }
    assert_eq!(CPDAG_SELECTION_OR_LATENT_DETAIL, "cpdag_scenarios.selection_or_latent");
    // a -> b - c is not maximally oriented (b -> c is compelled), and a conflict edge is no CPDAG.
    let compelled = Case {
        name: "compelled",
        nodes: 3,
        undirected: vec![(1, 2)],
        directed: vec![(0, 1)],
        total: 1,
    };
    let refused =
        enumerate_cpdag_completions(&CpdagCompletionInput::new(cpdag(&compelled)), BIG, &ctx)
            .unwrap_err();
    assert_eq!((refused.code, refused.detail), ("invalid_argument", CPDAG_NOT_A_CPDAG_DETAIL));
    let mut conflicted = Cpdag::with_variables(2);
    conflicted.insert_undirected(d(0), d(1)).unwrap();
    conflicted.mark_conflict(d(0), d(1)).unwrap();
    let refused =
        enumerate_cpdag_completions(&CpdagCompletionInput::new(conflicted), BIG, &ctx).unwrap_err();
    assert_eq!(refused.detail, CPDAG_NOT_A_CPDAG_DETAIL);

    // Evidence is bound per completion.
    let chain_set = enumerate(&case_named("chain"));
    let forward = [(0, 1), (1, 2)];
    let fork = [(1, 0), (1, 2)];
    let backward = [(1, 0), (2, 1)];
    let (a, b, c) =
        (id_of(&chain_set, &forward), id_of(&chain_set, &fork), id_of(&chain_set, &backward));
    // A completion whose catalog lacks the factor stays missing_evidence.
    let decision = decide(&CpdagEvidence::PerCompletion(vec![
        binding(&a, &a, "ev-a", &[0, 1, 2]),
        binding(&b, &b, "ev-b", &[0, 1, 2]),
        binding(&c, &c, "ev-c", &[0, 1]),
    ]))
    .unwrap();
    assert_eq!(status_of(&decision, &forward), "identified");
    assert_eq!(status_of(&decision, &fork), "identified");
    assert_eq!(status_of(&decision, &backward), "missing_evidence");
    assert!(decision.is_exportable());
    let identities = decision
        .completions
        .iter()
        .map(|r| (r.identity.to_string(), r.evidence_identity.as_deref().map(str::to_owned)))
        .collect::<Vec<_>>();
    assert!(identities.contains(&(a.to_string(), Some("ev-a".into()))));
    assert!(identities.contains(&(c.to_string(), Some("ev-c".into()))));
    // A completion with no binding is missing_evidence, with no evidence identity.
    let decision =
        decide(&CpdagEvidence::PerCompletion(vec![binding(&a, &a, "ev-a", &[0, 1, 2])])).unwrap();
    assert_eq!(status_of(&decision, &forward), "identified");
    assert_eq!(status_of(&decision, &fork), "missing_evidence");
    assert_eq!(status_of(&decision, &backward), "missing_evidence");
    let unbound = decision
        .decision
        .as_ref()
        .unwrap()
        .decisions
        .iter()
        .filter_map(|d| match &d.outcome {
            ScenarioOutcome::MissingEvidence { obligations } => Some(obligations.join(";")),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(unbound.len(), 2);
    assert!(unbound.iter().all(|text| text.contains(CPDAG_NO_EVIDENCE_DETAIL)));
    assert!(decision.completions.iter().filter(|r| r.evidence_identity.is_none()).count() == 2);
    assert_evidence_mismatch_refusals(&a, &b);
}

/// Evidence certified for one completion never binds another: the binding names
/// a different identity, an unknown identity, or the same completion twice.
fn assert_evidence_mismatch_refusals(a: &Arc<str>, b: &Arc<str>) {
    let wrong = decide(&CpdagEvidence::PerCompletion(vec![binding(b, a, "ev", &[0, 1, 2])]));
    assert_eq!(refusal(wrong.unwrap_err()), ("invalid_argument", CPDAG_EVIDENCE_MISMATCH_DETAIL));
    let unknown: Arc<str> = Arc::from("not-a-completion");
    let wrong =
        decide(&CpdagEvidence::PerCompletion(vec![binding(&unknown, &unknown, "ev", &[0, 1, 2])]));
    assert_eq!(refusal(wrong.unwrap_err()).1, CPDAG_EVIDENCE_MISMATCH_DETAIL);
    let twice = decide(&CpdagEvidence::PerCompletion(vec![
        binding(a, a, "ev-1", &[0, 1, 2]),
        binding(a, a, "ev-2", &[0, 1, 2]),
    ]));
    assert_eq!(refusal(twice.unwrap_err()).1, CPDAG_EVIDENCE_MISMATCH_DETAIL);
    assert_eq!(CPDAG_EVIDENCE_MISMATCH_DETAIL, "cpdag_scenarios.evidence_identity_mismatch");
}

#[test]
fn x2_cpdag_permutation_identities() {
    // Nodes added and edges inserted in a different order: dense ids change, the
    // variable ids and so the completion identities and order do not.
    for case in cases() {
        let canonical = enumerate(&case);
        let order = (0..case.nodes).rev().collect::<Vec<_>>();
        let mut graph = Cpdag::empty();
        for variable in &order {
            graph.add_node(NodeRef::Static(v(*variable))).unwrap();
        }
        let dense = |variable: u32| {
            d(u32::try_from(order.iter().position(|o| *o == variable).unwrap()).unwrap())
        };
        for (a, b) in case.undirected.iter().rev() {
            graph.insert_undirected(dense(*b), dense(*a)).unwrap();
        }
        for (a, b) in case.directed.iter().rev() {
            graph.insert_directed(dense(*a), dense(*b)).unwrap();
        }
        let permuted = enumerate_cpdag_completions(
            &CpdagCompletionInput::new(graph),
            BIG,
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
        assert_eq!(permuted.cpdag_identity, canonical.cpdag_identity, "{}", case.name);
        assert_eq!(permuted.completions, canonical.completions, "{}", case.name);
        assert_eq!(permuted.variables, canonical.variables);
    }
    // The decision is permutation invariant too: same statuses by completion identity.
    let ctx = ExecutionContext::for_tests(1);
    let shared = CpdagEvidence::Shared {
        evidence_identity: Arc::from("target-law"),
        catalog: target_catalog(&[0, 1, 2]),
    };
    let first = decide(&shared).unwrap();
    let mut graph = Cpdag::empty();
    for variable in [2, 0, 1] {
        graph.add_node(NodeRef::Static(v(variable))).unwrap();
    }
    graph.insert_undirected(d(1), d(2)).unwrap(); // v0 - v1
    graph.insert_undirected(d(2), d(0)).unwrap(); // v1 - v2
    let second = decide_cpdag_completions(
        &CpdagCompletionInput::new(graph),
        &binary_coordinates(3),
        &query(),
        &shared,
        SEARCH,
        &ctx,
    )
    .unwrap();
    assert_eq!(first.completions, second.completions);
    let statuses = |d: &CpdagCompletionDecision| {
        d.decision
            .as_ref()
            .unwrap()
            .decisions
            .iter()
            .map(|x| (x.scenario.name.to_string(), x.outcome.status()))
            .collect::<Vec<_>>()
    };
    assert_eq!(statuses(&first), statuses(&second));
    assert_eq!(first.cpdag_identity, second.cpdag_identity);
}

fn six() -> Case {
    case_named("six_nodes")
}

fn run(case: &Case, limits: SearchLimits, ctx: &ExecutionContext) -> CpdagEnumeration {
    enumerate_cpdag_completions(&CpdagCompletionInput::new(cpdag(case)), limits, ctx).unwrap()
}

#[test]
fn x2_cpdag_budget_stop_enumeration() {
    let case = six();
    let total = oracle(&case).len();
    assert_eq!(total, 18);
    let ctx = ExecutionContext::for_tests(1);
    let mut previous = 0;
    // One operation per orientation attempt at each internal search node: the
    // 3-path and the triangle give 2 * (1 + 2 + 3 + 6 + 12) = 48 attempts.
    for operations in [1, 2, 3, 5, 8, 13, 21, 34, 47] {
        let limits = SearchLimits { operations, depth: 64 };
        let partial = run(&case, limits, &ctx);
        let found = partial.completions.len();
        // Explored plus unevaluated is the oracle's total, whatever the limit.
        assert_eq!(found + partial.not_enumerated, total, "operations {operations}");
        assert_eq!(partial.total(), total);
        assert!(found >= previous, "more budget never finds fewer completions");
        previous = found;
        if found < total {
            let receipt = partial.receipt.as_ref().unwrap();
            assert_eq!(receipt.stop, SearchStop::Operations);
            assert_eq!(receipt.operations_consumed, Some(operations));
            assert_eq!(receipt.explored.len(), found);
            assert!(!partial.is_exportable());
            // What was found is a subset of the true completions.
            assert!(found_in(&partial, &oracle(&case)));
        }
    }
    assert!(previous < total, "47 of 48 operations cannot enumerate every completion");
    // A limit that never binds finishes and is exportable; so does exactly 48.
    let whole = run(&case, BIG, &ctx);
    assert!(whole.is_exportable() && whole.completions.len() == total);
    let exact = run(&case, SearchLimits { operations: 48, depth: 64 }, &ctx);
    assert!(exact.is_exportable() && exact.completions.len() == total);
    assert_eq!(exact.completions, whole.completions);

    // Depth: five undirected edges need depth five.
    let shallow = run(&case, SearchLimits { operations: 1_000_000, depth: 2 }, &ctx);
    assert_eq!(shallow.receipt.as_ref().unwrap().stop, SearchStop::Depth);
    assert_eq!(shallow.total(), total);

    // Memory: live completion bytes are charged against the hard limit.
    let mut small = ExecutionContext::for_tests(1);
    small.memory = MemoryBudget { soft_limit_bytes: None, hard_limit_bytes: Some(400) };
    let limited = run(&case, BIG, &small);
    assert_eq!(limited.receipt.as_ref().unwrap().stop, SearchStop::Memory);
    assert_eq!(limited.completions.len(), 1);
    assert_eq!(limited.total(), total);

    // Cancellation before the search, and in the middle: non-exportable partials.
    let cancelled = ExecutionContext::for_tests(1);
    cancelled.cancellation.cancel();
    let before = run(&case, BIG, &cancelled);
    assert_eq!(before.receipt.as_ref().unwrap().stop, SearchStop::Cancelled);
    assert_eq!((before.completions.len(), before.not_enumerated), (0, total));
    assert!(!before.is_exportable());
    let mut mid = ExecutionContext::for_tests(1);
    mid.cancellation = CancellationToken::cancel_after_checks(8);
    let during = run(&case, BIG, &mid);
    assert_eq!(during.receipt.as_ref().unwrap().stop, SearchStop::Cancelled);
    assert_eq!(during.total(), total);
    assert!(!during.is_exportable());
    assert!(!during.completions.is_empty() || during.not_enumerated == total);
}

fn found_in(enumeration: &CpdagEnumeration, truth: &BTreeSet<Edges>) -> bool {
    found(enumeration).is_subset(truth)
}

#[test]
fn x2_cpdag_budget_stop_decisions() {
    // A budget that stops inside enumeration leaves every found completion
    // unevaluated and reports the unenumerated count separately.
    let case = six();
    let evidence = CpdagEvidence::Shared {
        evidence_identity: Arc::from("target-law"),
        catalog: target_catalog(&[0, 1, 2, 3, 4, 5]),
    };
    let ctx = ExecutionContext::for_tests(1);
    let partial = decide_cpdag_completions(
        &CpdagCompletionInput::new(cpdag(&case)),
        &binary_coordinates(6),
        &ClassicalTransportQuery {
            outcomes: Arc::from([v(2)]),
            treatments: Arc::from([v(1)]),
            source: Arc::from("source"),
            target: Arc::from("target"),
        },
        &evidence,
        SearchLimits { operations: 12, depth: 256 },
        &ctx,
    )
    .unwrap();
    assert!(!partial.is_exportable());
    assert_eq!(partial.total(), 18);
    assert!(partial.not_enumerated > 0);
    let decision = partial.decision.as_ref().unwrap();
    assert_eq!(decision.decisions.len(), partial.completions.len());
    assert!(decision.decisions.iter().all(|d| d.outcome.status() == "unevaluated"));
    assert_eq!(decision.receipt.as_ref().unwrap().stop, SearchStop::Operations);
}
