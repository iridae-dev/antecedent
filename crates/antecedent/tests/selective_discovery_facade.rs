//! Facade integration of selective discovery: a graph, completion or selection change reaches
//! the stage planner and the claim-level invalidation graph; the discovery rows are an
//! overlap in the evidence ledger; a transport request over a discovered structure is
//! refused with its unresolved edges; and a fresh process resumes from the portable artifact
//! only with the raw data.
//!
//! Hand oracles: the collider `0 -> 2 <- 1`, `2 -> 3` is its own CPDAG under a d-separation
//! oracle (v-structure, then Meek rule 1); a chain has no v-structure and stays an
//! equivalence class.

use std::sync::Arc;

use antecedent::analysis::selective_discovery::{
    CiTestSpec, DataIdentity, DeclaredCompletion, DiscoveryClaimKeys, DiscoveryOutcome,
    DiscoverySnapshot, GraphAcceptance, StageInputs, StaticAlgorithm, StaticDiscoveryContract,
    StaticSelectiveDiscovery, accepted_graph_stage_identity, add_discovery_claims,
    discovered_graph_source, discovered_structure_refusal, discovery_data_source,
    request_selection_diagram, revalidate_after_discovery,
};
use antecedent_contract::composition_verification::{
    CompositionGraph, CompositionNode, DependencyKind, NodeKind, NumericalCause, Revalidation,
    WithdrawalKind,
};
use antecedent_contract::evidence_ledger::{EvidenceLedger, Overlap};
use antecedent_core::recalc::{
    RecalcCapabilities, RecalcPlan, RetargetSupport, Stage, StageIdentities, StageIdentity,
};
use antecedent_core::{ExecutionContext, SemanticDigest, VariableId};
use antecedent_data::TabularData;
use antecedent_io::selective_discovery_artifact::SelectiveDiscoveryArtifact;
use antecedent_stats::{
    CiBatchRequest, CiBatchResult, CiResult, CiWorkspace, ConditionalIndependenceTest,
    PreparedCiTest, StatsError,
};

const SEED: u64 = 99_173;

// -- oracle ------------------------------------------------------------------------------------

/// d-separation by moralization of the ancestral set (independent of the library's graphs).
fn d_separated(parents: &[Vec<usize>], x: usize, y: usize, z: &[usize]) -> bool {
    let n = parents.len();
    let mut ancestral = vec![false; n];
    let mut stack: Vec<usize> = vec![x, y];
    stack.extend_from_slice(z);
    for &s in &stack {
        ancestral[s] = true;
    }
    while let Some(v) = stack.pop() {
        for &p in &parents[v] {
            if !ancestral[p] {
                ancestral[p] = true;
                stack.push(p);
            }
        }
    }
    let mut adj = vec![vec![false; n]; n];
    for v in 0..n {
        if !ancestral[v] {
            continue;
        }
        for &p in &parents[v] {
            adj[v][p] = true;
            adj[p][v] = true;
            for &q in &parents[v] {
                if p != q {
                    adj[p][q] = true;
                    adj[q][p] = true;
                }
            }
        }
    }
    let mut seen = vec![false; n];
    let mut frontier = vec![x];
    seen[x] = true;
    while let Some(v) = frontier.pop() {
        if v == y {
            return false;
        }
        for u in 0..n {
            if adj[v][u] && ancestral[u] && !seen[u] && !z.contains(&u) {
                seen[u] = true;
                frontier.push(u);
            }
        }
    }
    true
}

struct DsepOracle {
    parents: Vec<Vec<usize>>,
}

impl ConditionalIndependenceTest for DsepOracle {
    fn test_batch(
        &self,
        prepared: &PreparedCiTest,
        request: &CiBatchRequest<'_>,
        _workspace: &mut CiWorkspace,
        _ctx: &ExecutionContext,
    ) -> Result<CiBatchResult, StatsError> {
        prepared.ensure_compatible(request)?;
        let results = request
            .queries
            .iter()
            .map(|q| {
                let z = &request.z_flat[q.z_start..q.z_start + q.z_len];
                let separated = d_separated(&self.parents, q.x, q.y, z);
                CiResult {
                    statistic: if separated { 0.0 } else { 1.0 },
                    p_value: if separated { 1.0 } else { 0.0 },
                    df: 0.0,
                    ci: None,
                }
            })
            .collect();
        Ok(CiBatchResult { results })
    }
}

// -- data ----------------------------------------------------------------------------------------

struct Lcg(u64);

impl Lcg {
    fn normal(&mut self) -> f64 {
        let mut unit = || {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (f64::from(u32::try_from(self.0 >> 40).unwrap()) + 0.5) / f64::from(1u32 << 24)
        };
        let (u, v) = (unit(), unit());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

fn table(n: usize, seed: u64) -> TabularData {
    let mut rng = Lcg(seed);
    let (mut a, mut b, mut c, mut d) = (vec![], vec![], vec![], vec![]);
    for _ in 0..n {
        let (x, y) = (rng.normal(), rng.normal());
        let z = 0.8 * x + 0.8 * y + 0.6 * rng.normal();
        let w = 0.9 * z + 0.5 * rng.normal();
        a.push(x);
        b.push(y);
        c.push(z);
        d.push(w);
    }
    TabularData::from_f64_columns([
        ("a", a.as_slice()),
        ("b", b.as_slice()),
        ("c", c.as_slice()),
        ("d", d.as_slice()),
    ])
    .unwrap()
}

fn vars(n: u32) -> Vec<VariableId> {
    (0..n).map(VariableId::from_raw).collect()
}

fn oracle_contract(label: &str, parents: Vec<Vec<usize>>, n: u32) -> StaticDiscoveryContract {
    let mut contract = StaticDiscoveryContract::new(
        StaticAlgorithm::Pc,
        vars(n),
        CiTestSpec::declared(label, Arc::new(DsepOracle { parents })),
        SEED,
    );
    contract.fdr = None;
    contract
}

fn collider() -> StaticDiscoveryContract {
    oracle_contract("dsep:collider", vec![vec![], vec![], vec![0, 1], vec![2]], 4)
}

fn chain() -> StaticDiscoveryContract {
    oracle_contract("dsep:chain", vec![vec![], vec![0], vec![1], vec![2]], 4)
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(SEED)
}

struct Run {
    inputs: StageInputs,
    data: DataIdentity,
    outcome: DiscoveryOutcome,
}

fn run(contract: &StaticDiscoveryContract, data: &TabularData) -> Run {
    let mut adapter = StaticSelectiveDiscovery::new();
    adapter.run(contract, data, &ctx()).unwrap();
    Run {
        inputs: adapter.inputs().unwrap().clone(),
        data: adapter.data_identity().unwrap().clone(),
        outcome: adapter.outcome().unwrap().clone(),
    }
}

// -- stage planner --------------------------------------------------------------------------------

const STAGES: [Stage; 15] = [
    Stage::Graph,
    Stage::Query,
    Stage::Regime,
    Stage::Evidence,
    Stage::SourcePopulation,
    Stage::TargetPopulation,
    Stage::DataSnapshot,
    Stage::RowDesign,
    Stage::TreatmentGrid,
    Stage::LearnerFoldsRng,
    Stage::Utility,
    Stage::Identification,
    Stage::ScoreArtifact,
    Stage::Law,
    Stage::Decision,
];

fn identities(graph: StageIdentity) -> StageIdentities {
    let mut ids = StageIdentities::new();
    for stage in STAGES {
        ids.set(stage, StageIdentity::of(&stage.label(), &[b"v1"]));
    }
    ids.set(Stage::Graph, graph);
    ids
}

fn plan(previous: &StageIdentities, requested: &StageIdentities) -> RecalcPlan {
    RecalcPlan::plan(
        previous,
        requested,
        &RecalcCapabilities::in_process(RetargetSupport::Licensed),
    )
}

fn accept(outcome: &DiscoveryOutcome) -> StageIdentity {
    let acceptance = GraphAcceptance { declared_by: "reviewer".to_owned(), completion: None };
    accepted_graph_stage_identity(outcome, &acceptance).unwrap()
}

#[test]
fn a_resolved_structure_is_accepted_and_a_class_needs_a_declared_completion() {
    let data = table(40, 1);
    let resolved = run(&collider(), &data);
    assert!(resolved.outcome.structure.unresolved_edges().is_empty());
    let first = accept(&resolved.outcome);
    assert_eq!(first, accept(&resolved.outcome));

    let class = run(&chain(), &data);
    let acceptance = GraphAcceptance { declared_by: "reviewer".to_owned(), completion: None };
    let refusal = accepted_graph_stage_identity(&class.outcome, &acceptance).unwrap_err();
    assert_eq!(refusal.code, "discovered_structure_unresolved");
    assert_eq!(refusal.detail, "selective_discovery.unresolved_structure");
    assert_eq!(refusal.unresolved_edges.len(), 3);

    // A declared completion is accepted, and its identity and label are part of the result.
    let completion = |label: &str, byte: u8| GraphAcceptance {
        declared_by: "reviewer".to_owned(),
        completion: Some(DeclaredCompletion { label: label.to_owned(), identity: [byte; 32] }),
    };
    let a = accepted_graph_stage_identity(&class.outcome, &completion("0->1->2->3", 1)).unwrap();
    let b = accepted_graph_stage_identity(&class.outcome, &completion("3->2->1->0", 2)).unwrap();
    let relabelled =
        accepted_graph_stage_identity(&class.outcome, &completion("0->1->2->3 (v2)", 1)).unwrap();
    assert_ne!(a, b);
    assert_ne!(a, relabelled);

    // An exhausted search bound cannot be accepted at all.
    let mut bounded = resolved.outcome.clone();
    bounded.search_incomplete = true;
    bounded.bounded_search_receipts = vec![("pc.depth_budget".to_owned(), "exhausted".to_owned())];
    let refusal = accepted_graph_stage_identity(&bounded, &acceptance).unwrap_err();
    assert_eq!(refusal.detail, "selective_discovery.search_incomplete");
    assert_eq!(refusal.bounded_search_receipts.len(), 1);
}

#[test]
fn a_changed_discovered_graph_reaches_identification_fit_law_and_decision_only() {
    let data = table(40, 2);
    let before = identities(accept(&run(&collider(), &data).outcome));
    let same = plan(&before, &identities(accept(&run(&collider(), &data).outcome)));
    assert!(same.recomputed().is_empty());

    // A different discovered structure: the class of the chain, with a completion declared.
    let class = run(&chain(), &data);
    let acceptance = GraphAcceptance {
        declared_by: "reviewer".to_owned(),
        completion: Some(DeclaredCompletion { label: "chain".to_owned(), identity: [5; 32] }),
    };
    let after = identities(accepted_graph_stage_identity(&class.outcome, &acceptance).unwrap());
    let changed = plan(&before, &after);
    let mut recomputed = changed.recomputed();
    recomputed.sort_unstable();
    let mut expected = vec![
        Stage::Graph,
        Stage::Identification,
        Stage::ScoreArtifact,
        Stage::Law,
        Stage::Decision,
    ];
    expected.sort_unstable();
    assert_eq!(recomputed, expected);
    // Inputs that did not change stay reused.
    assert!(changed.reused().contains(&Stage::DataSnapshot));
    assert!(changed.reused().contains(&Stage::Query));
}

// -- claim graph ----------------------------------------------------------------------------------------

fn claim_graph(run: &Run, completion: Option<&DeclaredCompletion>) -> CompositionGraph {
    let mut graph = CompositionGraph::new();
    let snapshot = DiscoverySnapshot {
        inputs: &run.inputs,
        data: &run.data,
        outcome: &run.outcome,
        completion,
    };
    let keys: DiscoveryClaimKeys =
        add_discovery_claims(&mut graph, "discovery", &snapshot).unwrap();
    let d = |b: u8| SemanticDigest::from_bytes([b; 32]);
    graph.add_node(CompositionNode::new("identification", NodeKind::Claim, d(1))).unwrap();
    graph.add_node(CompositionNode::new("fit", NodeKind::LedgerResult, d(2))).unwrap();
    graph.add_node(CompositionNode::new("transport", NodeKind::TransportedResult, d(3))).unwrap();
    graph.add_node(CompositionNode::new("decision", NodeKind::Decision, d(4))).unwrap();
    let anchor = keys.downstream_anchor().to_owned();
    graph.add_edge("identification", &anchor, DependencyKind::StructuralDerivation).unwrap();
    graph.add_edge("fit", "identification", DependencyKind::StructuralDerivation).unwrap();
    graph.add_edge("fit", &keys.data, DependencyKind::Data).unwrap();
    graph.add_edge("transport", "fit", DependencyKind::Data).unwrap();
    graph.add_edge("decision", "transport", DependencyKind::Data).unwrap();
    graph
}

const DOWNSTREAM: [&str; 4] = ["identification", "fit", "transport", "decision"];

#[test]
fn unchanged_discovery_leaves_every_downstream_claim_valid() {
    let data = table(40, 3);
    let base = claim_graph(&run(&collider(), &data), None);
    let again = claim_graph(&run(&collider(), &data), None);
    let result = revalidate_after_discovery(&base, &again).unwrap();
    for key in DOWNSTREAM.into_iter().chain(["discovery.graph", "discovery.statistics"]) {
        assert_eq!(result.claim(key).unwrap().class, Revalidation::StillValid, "{key}");
    }
}

#[test]
fn new_data_recomputes_the_graph_and_everything_that_rests_on_it_without_withdrawing() {
    let before = claim_graph(&run(&collider(), &table(40, 4)), None);
    let after = claim_graph(&run(&collider(), &table(40, 5)), None);
    let result = revalidate_after_discovery(&before, &after).unwrap();
    for key in
        ["discovery.data", "discovery.statistics", "discovery.graph"].into_iter().chain(DOWNSTREAM)
    {
        let claim = result.claim(key).unwrap();
        assert_eq!(claim.class, Revalidation::Recompute, "{key}");
        assert!(claim.withdrawals.is_empty(), "{key}");
    }
    // The structure is reused and the fit is not retained: the earlier fit carries no authority.
    let identification = result.claim("identification").unwrap();
    assert!(
        identification
            .numerical
            .iter()
            .any(|r| matches!(r.cause, NumericalCause::StructureReusedFitNotRetained))
    );
}

#[test]
fn a_threshold_change_keeps_the_statistics_and_recomputes_what_rests_on_the_graph() {
    let data = table(40, 6);
    let before = claim_graph(&run(&collider(), &data), None);
    let mut strict = collider();
    strict.constraints.alpha = 0.01;
    let after = claim_graph(&run(&strict, &data), None);
    let result = revalidate_after_discovery(&before, &after).unwrap();
    assert_eq!(result.claim("discovery.data").unwrap().class, Revalidation::StillValid);
    assert_eq!(result.claim("discovery.statistics").unwrap().class, Revalidation::StillValid);
    assert_eq!(result.claim("discovery.graph").unwrap().class, Revalidation::Recompute);
    for key in DOWNSTREAM {
        assert_eq!(result.claim(key).unwrap().class, Revalidation::Recompute, "{key}");
    }
}

#[test]
fn a_changed_structure_recomputes_and_a_changed_completion_reaches_the_anchor() {
    let data = table(40, 7);
    let completion = |byte: u8| DeclaredCompletion { label: "c".to_owned(), identity: [byte; 32] };
    let class = run(&chain(), &data);
    let first = claim_graph(&class, Some(&completion(1)));
    let second = claim_graph(&class, Some(&completion(2)));
    let result = revalidate_after_discovery(&first, &second).unwrap();
    // The discovered structure is untouched; only the declared completion changed.
    assert_eq!(result.claim("discovery.graph").unwrap().class, Revalidation::StillValid);
    for key in ["discovery.completion"].into_iter().chain(DOWNSTREAM) {
        assert_eq!(result.claim(key).unwrap().class, Revalidation::Recompute, "{key}");
    }
    let completion_node = first.node("discovery.completion").unwrap();
    assert!(
        completion_node
            .assumptions
            .contains(&"declared_completion_of_equivalence_class".to_owned()),
        "a completion is recorded as a declaration, not a discovery"
    );

    // A different discovered structure: recompute with a changed structure.
    let before = claim_graph(&run(&collider(), &data), None);
    let after = claim_graph(&class, None);
    let result = revalidate_after_discovery(&before, &after).unwrap();
    let graph = result.claim("discovery.graph").unwrap();
    assert!(graph.numerical.iter().any(|r| matches!(r.cause, NumericalCause::StructureChanged)));
    assert_eq!(result.claim("decision").unwrap().class, Revalidation::Recompute);
}

#[test]
fn an_exhausted_search_bound_withdraws_every_downstream_claim() {
    let data = table(40, 8);
    let complete = run(&collider(), &data);
    let before = claim_graph(&complete, None);
    let mut bounded = run(&collider(), &data);
    bounded.outcome.search_incomplete = true;
    bounded.outcome.bounded_search_receipts =
        vec![("fci.discriminating_path_budget".to_owned(), "exhausted on 2 edges".to_owned())];
    let after = claim_graph(&bounded, None);
    let node = after.node("discovery.graph").unwrap();
    assert!(node.assumptions.iter().any(|a| a.starts_with("bounded_search:")));
    let result = revalidate_after_discovery(&before, &after).unwrap();
    for key in ["discovery.graph"].into_iter().chain(DOWNSTREAM) {
        let claim = result.claim(key).unwrap();
        assert_eq!(claim.class, Revalidation::Withdrawn, "{key}");
        assert!(claim.withdrawn_for(WithdrawalKind::Support), "{key}");
    }
}

// -- evidence ledger --------------------------------------------------------------------------------------

#[test]
fn reusing_the_discovery_rows_in_an_analysis_is_a_shared_observation_overlap() {
    let snapshot = b"dataset-snapshot-7";
    let mut ledger = EvidenceLedger::new();
    let discovery =
        ledger.add_source(discovery_data_source("discovery rows", snapshot, 0..60)).unwrap();
    let same_rows =
        ledger.add_source(discovery_data_source("analysis, all rows", snapshot, 0..100)).unwrap();
    let held_out =
        ledger.add_source(discovery_data_source("analysis, held out", snapshot, 60..100)).unwrap();
    assert_eq!(
        ledger.overlap(discovery, same_rows).unwrap(),
        Overlap::SharedObservations { shared: 60 }
    );
    assert_eq!(ledger.overlap(discovery, held_out).unwrap(), Overlap::Independent);

    // The discovered graph is a model built from data already used: its lineage is explicit.
    let graph =
        ledger.add_source(discovered_graph_source("discovered graph", vec![discovery])).unwrap();
    assert!(ledger.source(graph).is_some());
    // Another snapshot's rows are not the same observations.
    let other = ledger
        .add_source(discovery_data_source("other dataset", b"dataset-snapshot-8", 0..60))
        .unwrap();
    assert_eq!(ledger.overlap(discovery, other).unwrap(), Overlap::Independent);
}

// -- transport ------------------------------------------------------------------------------------------------

#[test]
fn a_discovered_structure_transport_request_returns_the_unresolved_edges() {
    let data = table(40, 9);
    let class = run(&chain(), &data);
    let refusal =
        request_selection_diagram(&class.outcome, &[VariableId::from_raw(2)]).unwrap_err();
    let external = discovered_structure_refusal(&refusal);
    external.validate().unwrap();
    assert_eq!(external.code, "discovered_structure_unresolved");
    assert_eq!(external.detail, "selective_discovery.unresolved_structure");
    let offending = external.offending.as_deref().unwrap();
    assert_eq!(offending.split(';').count(), 3);
    assert!(offending.contains("undirected_in_equivalence_class"));
    assert!(external.remedy.unwrap().contains("not computed across an equivalence class"));
}

// -- fresh process ------------------------------------------------------------------------------------------------

#[test]
fn a_fresh_process_resumes_from_the_artifact_only_with_the_raw_data() {
    let data = table(250, 10);
    let contract = StaticDiscoveryContract::new(
        StaticAlgorithm::Pc,
        vars(4),
        CiTestSpec::named("parcorr"),
        SEED,
    );
    let mut producer = StaticSelectiveDiscovery::new();
    producer.run(&contract, &data, &ctx()).unwrap();
    let state = producer.export_state().unwrap();
    let bytes = SelectiveDiscoveryArtifact::to_bytes(&state, "run-0001").unwrap();
    let retained =
        SelectiveDiscoveryArtifact::from_bytes(&bytes, None).unwrap().identity().to_owned();

    // Fresh consumer: bytes and retained identity only, then the raw data.
    let artifact = SelectiveDiscoveryArtifact::from_bytes(&bytes, Some(&retained)).unwrap();
    let missing = StaticSelectiveDiscovery::resume(&contract, None, artifact.state(), &ctx());
    assert_eq!(missing.unwrap_err().detail(), Some("selective_discovery.raw_data_missing"));
    let (resumed, report) =
        StaticSelectiveDiscovery::resume(&contract, Some(&data), artifact.state(), &ctx()).unwrap();
    assert_eq!(report.work.ci_tests_attempted, 0);
    assert_eq!(
        resumed.outcome().unwrap().result_identity,
        producer.outcome().unwrap().result_identity
    );

    // Different raw data of the same shape: refused, not resumed.
    let other = table(250, 11);
    let err = StaticSelectiveDiscovery::resume(&contract, Some(&other), artifact.state(), &ctx())
        .unwrap_err();
    assert_eq!(err.detail(), Some("selective_discovery.data_mismatch"));

    // A semantic mutation, resealed so the container is consistent: each statistic is a valid
    // statistic, and the replay reaches another graph than the recorded one.
    let mut edited_state = state.clone();
    for receipt in &mut edited_state.receipts {
        receipt.p_value = 1.0;
    }
    edited_state.reseal();
    let edited_bytes = SelectiveDiscoveryArtifact::to_bytes(&edited_state, "run-0001").unwrap();
    let edited = SelectiveDiscoveryArtifact::from_bytes(&edited_bytes, None).unwrap();
    let err = StaticSelectiveDiscovery::resume(&contract, Some(&data), edited.state(), &ctx())
        .unwrap_err();
    assert_eq!(err.detail(), Some("selective_discovery.replay_diverged"));
    // And a consumer that retained the identity refuses the resealed artifact outright.
    assert!(SelectiveDiscoveryArtifact::from_bytes(&edited_bytes, Some(&retained)).is_err());
}
