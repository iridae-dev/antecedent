//! Selective discovery against independent oracles.
//!
//! Three kinds of check, kept apart on purpose:
//!
//! * **Known-graph oracle.** A d-separation oracle over a hand-written DAG drives PC; the
//!   asserted CPDAG is the one worked out by hand (v-structure, then Meek rule 1). This
//!   checks the adapter wiring against a result that does not come from the library.
//! * **Algorithmic match.** A selective run must equal a full run of the same request on a
//!   fresh adapter, bit for bit, through the result identity and the per-edge evidence bits.
//!   This is a statement about reuse, not about recovery.
//! * **Work accounting.** Counters are asserted from what the algorithm did.
//!
//! No test here asserts, or could assert, that a discovered graph is the true graph.

use std::sync::Arc;

use antecedent_core::{CancellationToken, ExecutionContext, Lag, VariableId};
use antecedent_data::{TabularData, TimeSeriesData};
use antecedent_discovery::constraints::CiMaskType;
use antecedent_discovery::result::LaggedLink;
use antecedent_discovery::selective::{
    CiTestSpec, DiscoveryOutcome, DiscoveryStage, Mark, NativeResult, PortableDiscoveryState,
    StageDecision, StaticAlgorithm, StaticDiscoveryContract, StaticSelectiveDiscovery,
    TemporalAlgorithm, TemporalDiscoveryContract, TemporalSelectiveDiscovery, WorkUnits,
    request_downstream_interval, request_recovery_claim, request_score_search_reuse,
    request_selection_diagram,
};
use antecedent_stats::{
    CiBatchRequest, CiBatchResult, CiResult, CiWorkspace, ConditionalIndependenceTest,
    PreparedCiTest, StatsError,
};

const SEED: u64 = 20_240_611;

// -- independent oracles -------------------------------------------------------------------

/// d-separation in a known DAG by the moralization criterion (Lauritzen et al. 1990):
/// `x ⊥ y | z` iff `x` and `y` are disconnected in the moral graph of the ancestral set of
/// `{x, y} ∪ z` after removing `z`. Written independently of the library's graph code.
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
    let blocked = |v: usize| z.contains(&v);
    let mut seen = vec![false; n];
    let mut frontier = vec![x];
    seen[x] = true;
    while let Some(v) = frontier.pop() {
        if v == y {
            return false;
        }
        for u in 0..n {
            if adj[v][u] && ancestral[u] && !seen[u] && !blocked(u) {
                seen[u] = true;
                frontier.push(u);
            }
        }
    }
    true
}

struct DsepOracle {
    parents: Vec<Vec<usize>>,
    /// DAG node of each column position.
    nodes: Vec<usize>,
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
        let mut results = Vec::new();
        for q in request.queries {
            let z: Vec<usize> = request.z_flat[q.z_start..q.z_start + q.z_len]
                .iter()
                .map(|&c| self.nodes[c])
                .collect();
            let separated = d_separated(&self.parents, self.nodes[q.x], self.nodes[q.y], &z);
            results.push(CiResult {
                statistic: if separated { 0.0 } else { 1.0 },
                p_value: if separated { 1.0 } else { 0.0 },
                df: 0.0,
                ci: None,
            });
        }
        Ok(CiBatchResult { results })
    }
}

// -- data ----------------------------------------------------------------------------------

struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 =
            self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        let top = u32::try_from(self.0 >> 40).unwrap();
        (f64::from(top) + 0.5) / f64::from(1u32 << 24)
    }

    fn normal(&mut self) -> f64 {
        let (u, v) = (self.unit(), self.unit());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

/// `0 → 2 ← 1`, `2 → 3`.
fn collider_table(n: usize, seed: u64) -> TabularData {
    let mut rng = Lcg(seed);
    let (mut c0, mut c1, mut c2, mut c3) = (vec![], vec![], vec![], vec![]);
    for _ in 0..n {
        let a = rng.normal();
        let b = rng.normal();
        let c = 0.8 * a + 0.8 * b + 0.6 * rng.normal();
        let d = 0.9 * c + 0.5 * rng.normal();
        c0.push(a);
        c1.push(b);
        c2.push(c);
        c3.push(d);
    }
    TabularData::from_f64_columns([
        ("a", c0.as_slice()),
        ("b", c1.as_slice()),
        ("c", c2.as_slice()),
        ("d", c3.as_slice()),
    ])
    .unwrap()
}

fn collider_parents() -> Vec<Vec<usize>> {
    vec![vec![], vec![], vec![0, 1], vec![2]]
}

fn vars(order: &[u32]) -> Vec<VariableId> {
    order.iter().copied().map(VariableId::from_raw).collect()
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(SEED)
}

fn pc_contract(order: &[u32], test: &str) -> StaticDiscoveryContract {
    StaticDiscoveryContract::new(StaticAlgorithm::Pc, vars(order), CiTestSpec::named(test), SEED)
}

fn oracle_contract(order: &[u32]) -> StaticDiscoveryContract {
    let oracle = DsepOracle {
        parents: collider_parents(),
        nodes: order.iter().map(|&v| v as usize).collect(),
    };
    let mut contract = StaticDiscoveryContract::new(
        StaticAlgorithm::Pc,
        vars(order),
        CiTestSpec::declared(format!("dsep-oracle:collider:{order:?}"), Arc::new(oracle)),
        SEED,
    );
    contract.fdr = None;
    contract
}

fn full_static(
    contract: &StaticDiscoveryContract,
    data: &TabularData,
) -> (DiscoveryOutcome, WorkUnits) {
    let mut adapter = StaticSelectiveDiscovery::new();
    let report = adapter.run(contract, data, &ctx()).unwrap();
    (adapter.outcome().unwrap().clone(), report.work)
}

/// Per-edge evidence bits of a static PC outcome, for bit-for-bit comparison.
fn evidence_bits(outcome: &DiscoveryOutcome) -> Vec<(LaggedLink, u64, u64, Option<u64>)> {
    let links = match &outcome.native {
        NativeResult::StaticCpdag(r) => r.evidence.links.to_vec(),
        NativeResult::StaticPag(r) => r.evidence.links.to_vec(),
        NativeResult::TemporalDag(r) => r.evidence.links.to_vec(),
        NativeResult::TemporalCpdag(r) => r.evidence.links.to_vec(),
        _ => unreachable!("every native result of this crate is listed"),
    };
    let mut bits: Vec<_> = links
        .into_iter()
        .map(|l| {
            (
                l.link,
                l.statistic.to_bits(),
                l.p_value.to_bits(),
                l.adjusted_p_value.map(f64::to_bits),
            )
        })
        .collect();
    bits.sort_by_key(|b| b.0);
    bits
}

fn assert_algorithmic_match(selective: &DiscoveryOutcome, full: &DiscoveryOutcome) {
    assert_eq!(selective.result_identity, full.result_identity);
    assert_eq!(selective.structure_identity, full.structure_identity);
    assert_eq!(selective.structure, full.structure);
    assert_eq!(evidence_bits(selective), evidence_bits(full));
    assert_eq!(selective.diagnostics, full.diagnostics);
}

// -- known-graph oracle ----------------------------------------------------------------------

#[test]
fn oracle_recovers_the_hand_derived_cpdag_of_a_collider_with_a_descendant() {
    // True DAG 0 -> 2 <- 1, 2 -> 3. Skeleton: 0-2, 1-2, 2-3. 0 and 1 are separated by the
    // empty set, so 0 -> 2 <- 1 is a v-structure; 0 -> 2 - 3 with 0, 3 non-adjacent orients
    // 2 -> 3 by Meek rule 1. The CPDAG is the DAG itself.
    let data = collider_table(40, 1);
    let (outcome, work) = full_static(&oracle_contract(&[0, 1, 2, 3]), &data);
    let directed: Vec<(u32, u32)> = outcome
        .structure
        .edges()
        .iter()
        .map(|e| {
            assert!(e.is_directed(), "{e:?}");
            match (e.at_a, e.at_b) {
                (Mark::Tail, Mark::Arrow) => (e.a.variable, e.b.variable),
                (Mark::Arrow, Mark::Tail) => (e.b.variable, e.a.variable),
                other => panic!("not directed: {other:?}"),
            }
        })
        .collect();
    let mut sorted = directed;
    sorted.sort_unstable();
    assert_eq!(sorted, vec![(0, 2), (1, 2), (2, 3)]);
    assert!(outcome.structure.unresolved_edges().is_empty());
    assert!(outcome.structure.is_fully_directed());
    assert!(work.ci_tests_attempted > 0);
    assert!(work.orientation_runs >= 1);
    assert!(work.orientations_applied >= 3);
    assert_eq!(work.score_evaluations, 0);
}

#[test]
fn oracle_chain_stays_an_equivalence_class_with_two_unresolved_edges() {
    // 0 -> 1 -> 2: no v-structure, so the CPDAG is undirected.
    let data = collider_table(40, 2);
    let oracle = DsepOracle { parents: vec![vec![], vec![0], vec![1]], nodes: vec![0, 1, 2] };
    let mut contract = StaticDiscoveryContract::new(
        StaticAlgorithm::Pc,
        vars(&[0, 1, 2]),
        CiTestSpec::declared("dsep-oracle:chain", Arc::new(oracle)),
        SEED,
    );
    contract.fdr = None;
    let (outcome, _) = full_static(&contract, &data);
    let unresolved = outcome.structure.unresolved_edges();
    assert_eq!(unresolved.len(), 2);
    assert!(unresolved.iter().all(|u| u.reason == "undirected_in_equivalence_class"));
    assert!(!outcome.structure.is_fully_directed());
}

#[test]
fn oracle_result_is_invariant_to_the_search_order_of_the_variables() {
    let data = collider_table(40, 3);
    let (reference, _) = full_static(&oracle_contract(&[0, 1, 2, 3]), &data);
    for order in [[3, 2, 1, 0], [2, 0, 3, 1], [1, 3, 0, 2]] {
        let (permuted, _) = full_static(&oracle_contract(&order), &data);
        assert_eq!(permuted.structure, reference.structure, "{order:?}");
    }
}

// -- selective vs full -----------------------------------------------------------------------

#[test]
fn repeated_request_is_pure_reuse_with_no_executed_work() {
    let data = collider_table(300, 11);
    let contract = pc_contract(&[0, 1, 2, 3], "parcorr");
    let mut adapter = StaticSelectiveDiscovery::new();
    let first = adapter.run(&contract, &data, &ctx()).unwrap();
    assert!(first.work.ci_tests_attempted > 0);
    assert_eq!(first.work.ci_tests_attempted, first.work.ci_tests_completed);
    assert!(first.work.ci_tests_attempted >= first.work.ci_tests_consumed);
    assert!(first.work.search_passes >= 1);
    let before = adapter.outcome().unwrap().result_identity;
    let retained = adapter.retained_statistics();
    let second = adapter.run(&contract, &data, &ctx()).unwrap();
    assert!(second.plan.is_pure_reuse());
    assert!(second.work.is_pure_reuse());
    assert!(!second.outcome_changed);
    assert_eq!(adapter.outcome().unwrap().result_identity, before);
    assert_eq!(adapter.retained_statistics(), retained);
}

#[test]
fn threshold_change_reruns_decisions_over_retained_statistics_and_matches_a_full_run() {
    let data = collider_table(300, 12);
    let mut adapter = StaticSelectiveDiscovery::new();
    let mut loose = pc_contract(&[0, 1, 2, 3], "parcorr");
    loose.constraints.alpha = 0.2;
    adapter.run(&loose, &data, &ctx()).unwrap();

    let mut strict = loose.clone();
    strict.constraints.alpha = 0.01;
    let report = adapter.run(&strict, &data, &ctx()).unwrap();
    assert!(report.plan.decision(DiscoveryStage::Statistics).is_reused());
    assert_eq!(
        report.plan.decision(DiscoveryStage::Search),
        &StageDecision::Recomputed { because: vec!["thresholds"] }
    );
    let (full, full_work) = full_static(&strict, &data);
    assert_algorithmic_match(adapter.outcome().unwrap(), &full);
    assert!(report.work.ci_tests_reused > 0, "no statistic was reused");
    assert!(report.work.ci_tests_attempted < full_work.ci_tests_attempted);
    assert_eq!(report.work.ci_tests_consumed, full_work.ci_tests_consumed);
}

#[test]
fn background_knowledge_change_reuses_statistics_and_matches_a_full_run() {
    let data = collider_table(300, 13);
    let mut adapter = StaticSelectiveDiscovery::new();
    let open = pc_contract(&[0, 1, 2, 3], "parcorr");
    adapter.run(&open, &data, &ctx()).unwrap();

    let mut knowledge = open.clone();
    knowledge.constraints.forbidden = Arc::from([
        LaggedLink {
            source: VariableId::from_raw(0),
            source_lag: Lag::CONTEMPORANEOUS,
            target: VariableId::from_raw(3),
            target_lag: Lag::CONTEMPORANEOUS,
        },
        LaggedLink {
            source: VariableId::from_raw(3),
            source_lag: Lag::CONTEMPORANEOUS,
            target: VariableId::from_raw(0),
            target_lag: Lag::CONTEMPORANEOUS,
        },
    ]);
    let report = adapter.run(&knowledge, &data, &ctx()).unwrap();
    assert!(report.plan.decision(DiscoveryStage::Statistics).is_reused());
    assert_eq!(
        report.plan.decision(DiscoveryStage::Search),
        &StageDecision::Recomputed { because: vec!["background_knowledge"] }
    );
    let (full, full_work) = full_static(&knowledge, &data);
    assert_algorithmic_match(adapter.outcome().unwrap(), &full);
    assert!(report.work.ci_tests_attempted < full_work.ci_tests_attempted);
}

#[test]
fn permuted_variable_order_and_a_changed_ci_test_match_full_runs() {
    let data = collider_table(300, 14);
    let mut adapter = StaticSelectiveDiscovery::new();
    adapter.run(&pc_contract(&[0, 1, 2, 3], "parcorr"), &data, &ctx()).unwrap();

    let permuted = pc_contract(&[3, 1, 0, 2], "parcorr");
    let report = adapter.run(&permuted, &data, &ctx()).unwrap();
    assert!(!report.plan.decision(DiscoveryStage::Statistics).is_reused());
    let (full, _) = full_static(&permuted, &data);
    assert_algorithmic_match(adapter.outcome().unwrap(), &full);

    let other_test = pc_contract(&[3, 1, 0, 2], "robust_parcorr");
    let report = adapter.run(&other_test, &data, &ctx()).unwrap();
    match report.plan.decision(DiscoveryStage::Statistics) {
        StageDecision::Recomputed { because } => assert!(because.contains(&"ci_test")),
        StageDecision::Reused => panic!("a different test must not reuse statistics"),
    }
    assert_eq!(report.work.ci_tests_reused, 0);
    let (full, _) = full_static(&other_test, &data);
    assert_algorithmic_match(adapter.outcome().unwrap(), &full);
}

#[test]
fn permutation_test_statistics_reuse_matches_a_full_run() {
    // A stochastic null: the p-value is a function of (seed, query), which the coordinate
    // includes. Reuse must reproduce a full run exactly.
    let data = collider_table(120, 15);
    let mut loose = pc_contract(&[0, 1, 2, 3], "knn_dependence");
    loose.constraints.significance =
        antecedent_stats::SignificanceMethod::BlockShuffle { replicates: 99, block_size: 1 };
    loose.constraints.alpha = 0.1;
    loose.constraints.max_cond_size = 1;
    loose.fdr = None;
    let mut adapter = StaticSelectiveDiscovery::new();
    adapter.run(&loose, &data, &ctx()).unwrap();
    let mut strict = loose.clone();
    strict.constraints.alpha = 0.05;
    let report = adapter.run(&strict, &data, &ctx()).unwrap();
    assert!(report.plan.decision(DiscoveryStage::Statistics).is_reused());
    let (full, _) = full_static(&strict, &data);
    assert_algorithmic_match(adapter.outcome().unwrap(), &full);
}

// -- cache, identity and atomicity ------------------------------------------------------------

#[test]
fn changed_input_at_unchanged_shape_retains_nothing_and_matches_a_fresh_run() {
    let original = collider_table(200, 21);
    let contract = pc_contract(&[0, 1, 2, 3], "parcorr");
    let mut adapter = StaticSelectiveDiscovery::new();
    adapter.run(&contract, &original, &ctx()).unwrap();
    let before = adapter.data_identity().unwrap().clone();

    // One value of one column changes; the schema, the row count and every label are equal.
    let mut columns: Vec<Vec<f64>> = (0..4)
        .map(|i| {
            use antecedent_data::TableView;
            original.float64_values(VariableId::from_raw(i)).unwrap()
        })
        .collect();
    columns[1][17] += 0.5;
    let mutated = TabularData::from_f64_columns([
        ("a", columns[0].as_slice()),
        ("b", columns[1].as_slice()),
        ("c", columns[2].as_slice()),
        ("d", columns[3].as_slice()),
    ])
    .unwrap();

    let report = adapter.run(&contract, &mutated, &ctx()).unwrap();
    let after = adapter.data_identity().unwrap().clone();
    assert_ne!(before.digest, after.digest);
    assert_eq!(after.changed_variables(&before), vec![VariableId::from_raw(1)]);
    assert_eq!(
        report.plan.decision(DiscoveryStage::Statistics),
        &StageDecision::Recomputed { because: vec!["data"] }
    );
    assert_eq!(report.work.ci_tests_reused, 0, "new data must not be served from old statistics");
    let (fresh, _) = full_static(&contract, &mutated);
    assert_algorithmic_match(adapter.outcome().unwrap(), &fresh);
}

#[test]
fn failed_or_cancelled_update_leaves_the_retained_state_untouched() {
    let data = collider_table(200, 22);
    let contract = pc_contract(&[0, 1, 2, 3], "parcorr");
    let mut adapter = StaticSelectiveDiscovery::new();
    adapter.run(&contract, &data, &ctx()).unwrap();
    let identity = adapter.outcome().unwrap().result_identity;
    let inputs = adapter.inputs().unwrap().clone();
    let retained = adapter.retained_statistics();

    // Cancellation part-way through a candidate with a different threshold.
    let mut changed = contract.clone();
    changed.constraints.alpha = 0.001;
    let mut cancelled = ctx();
    cancelled.cancellation = CancellationToken::cancel_after_checks(1);
    assert!(adapter.run(&changed, &data, &cancelled).is_err());
    // A refusal before any work: the context's seed is not the contract's.
    let wrong_seed = ExecutionContext::for_tests(SEED + 1);
    let refusal = adapter.run(&contract, &data, &wrong_seed).unwrap_err();
    assert_eq!(refusal.detail(), Some("selective_discovery.rng_mismatch"));

    assert_eq!(adapter.outcome().unwrap().result_identity, identity);
    assert_eq!(adapter.inputs().unwrap(), &inputs);
    assert_eq!(adapter.retained_statistics(), retained);
    // The retained state is still the one a repeated request is served from.
    assert!(adapter.run(&contract, &data, &ctx()).unwrap().work.is_pure_reuse());
}

#[test]
fn closed_routes_are_refused_with_a_remedy() {
    let data = collider_table(50, 23);
    let mut multi = pc_contract(&[0, 1, 2, 3], "parcorr");
    multi.additional_populations = vec!["second_site".to_owned()];
    let err = StaticSelectiveDiscovery::new().run(&multi, &data, &ctx()).unwrap_err();
    assert_eq!(err.code(), Some("route_not_supported"));
    assert_eq!(err.detail(), Some("selective_discovery.multi_population"));
    assert!(err.to_string().contains("one declared population"));
}

#[test]
fn closed_claims_are_refused_with_cell_not_licensed_and_a_remedy() {
    let data = collider_table(40, 71);
    let (outcome, _) = full_static(&oracle_contract(&[0, 1, 2, 3]), &data);
    let recovery = request_recovery_claim(&outcome).unwrap_err();
    assert_eq!(recovery.code(), Some("cell_not_licensed"));
    assert_eq!(recovery.detail(), Some("selective_discovery.recovery_claim_not_licensed"));
    assert!(recovery.to_string().contains("not testable"));
    let interval = request_downstream_interval(&outcome).unwrap_err();
    assert_eq!(interval.code(), Some("cell_not_licensed"));
    assert_eq!(interval.detail(), Some("selective_discovery.downstream_interval_not_licensed"));
    assert!(interval.to_string().contains("disjoint discovery and analysis rows"));
    let score = request_score_search_reuse().unwrap_err();
    assert_eq!(score.code(), Some("route_not_supported"));
    assert_eq!(score.detail(), Some("selective_discovery.score_search_closed"));
}

// -- transport under graph uncertainty ----------------------------------------------------------

#[test]
fn a_discovered_structure_never_supplies_a_selection_diagram() {
    let data = collider_table(40, 31);
    // Equivalence class: the refusal names the unresolved edges.
    let oracle = DsepOracle { parents: vec![vec![], vec![0], vec![1]], nodes: vec![0, 1, 2] };
    let mut chain = StaticDiscoveryContract::new(
        StaticAlgorithm::Pc,
        vars(&[0, 1, 2]),
        CiTestSpec::declared("dsep-oracle:chain-transport", Arc::new(oracle)),
        SEED,
    );
    chain.fdr = None;
    let (class, _) = full_static(&chain, &data);
    let refusal = request_selection_diagram(&class, &[VariableId::from_raw(1)]).unwrap_err();
    assert_eq!(refusal.code, "discovered_structure_unresolved");
    assert_eq!(refusal.detail, "selective_discovery.unresolved_structure");
    assert_eq!(refusal.unresolved_edges.len(), 2);
    assert!(refusal.remedy.contains("not computed across the class"));

    // Fully directed and search-complete: still a discovered estimate, still refused.
    let (directed, _) = full_static(&oracle_contract(&[0, 1, 2, 3]), &data);
    let refusal = request_selection_diagram(&directed, &[VariableId::from_raw(2)]).unwrap_err();
    assert_eq!(refusal.detail, "selective_discovery.unaccepted_discovery");
    assert!(refusal.unresolved_edges.is_empty());

    // A target that is not in the structure is named.
    let refusal = request_selection_diagram(&directed, &[VariableId::from_raw(9)]).unwrap_err();
    assert!(refusal.remedy.contains("v9"), "{}", refusal.remedy);

    // Search-incomplete status wins and preserves the bounded-search receipts.
    let mut bounded = directed.clone();
    bounded.search_incomplete = true;
    bounded.bounded_search_receipts =
        vec![("fci.discriminating_path_budget".to_owned(), "exhausted".to_owned())];
    let refusal = request_selection_diagram(&bounded, &[VariableId::from_raw(2)]).unwrap_err();
    assert_eq!(refusal.detail, "selective_discovery.search_incomplete");
    assert_eq!(refusal.bounded_search_receipts.len(), 1);
}

// -- portable state ---------------------------------------------------------------------------------

fn exported(contract: &StaticDiscoveryContract, data: &TabularData) -> PortableDiscoveryState {
    let mut adapter = StaticSelectiveDiscovery::new();
    adapter.run(contract, data, &ctx()).unwrap();
    adapter.export_state().unwrap()
}

#[test]
fn resume_replays_over_receipts_only_and_reproduces_the_result() {
    let data = collider_table(250, 41);
    let contract = pc_contract(&[0, 1, 2, 3], "parcorr");
    let state = exported(&contract, &data);
    assert!(!state.receipts.is_empty());
    state.validate().unwrap();

    let (resumed, report) =
        StaticSelectiveDiscovery::resume(&contract, Some(&data), &state, &ctx()).unwrap();
    assert_eq!(report.work.ci_tests_attempted, 0, "resume computed a statistic");
    assert!(report.work.ci_tests_reused > 0);
    assert_eq!(resumed.outcome().unwrap().result_identity, state.result_identity);
    assert_eq!(resumed.outcome().unwrap().structure.identity(), state.graph_identity());

    // The resumed adapter keeps working: the same request is pure reuse.
    let mut resumed = resumed;
    assert!(resumed.run(&contract, &data, &ctx()).unwrap().work.is_pure_reuse());
}

#[test]
fn resume_refuses_missing_data_other_data_and_another_contract() {
    let data = collider_table(250, 42);
    let contract = pc_contract(&[0, 1, 2, 3], "parcorr");
    let state = exported(&contract, &data);

    let err = StaticSelectiveDiscovery::resume(&contract, None, &state, &ctx()).unwrap_err();
    assert_eq!(err.code(), Some("discovery_statistics_invalid"));
    assert_eq!(err.detail(), Some("selective_discovery.raw_data_missing"));

    let other = collider_table(250, 43);
    let err =
        StaticSelectiveDiscovery::resume(&contract, Some(&other), &state, &ctx()).unwrap_err();
    assert_eq!(err.detail(), Some("selective_discovery.data_mismatch"));

    let mut changed = contract.clone();
    changed.constraints.alpha = 0.01;
    let err = StaticSelectiveDiscovery::resume(&changed, Some(&data), &state, &ctx()).unwrap_err();
    assert_eq!(err.detail(), Some("selective_discovery.contract_mismatch"));
}

#[test]
fn resume_refuses_invalid_receipts_and_a_resealed_semantic_mutation() {
    let data = collider_table(250, 44);
    let contract = pc_contract(&[0, 1, 2, 3], "parcorr");
    let state = exported(&contract, &data);

    // A statistic that is not a usable statistic.
    let mut invalid = state.clone();
    invalid.receipts[0].p_value = 1.5;
    let err =
        StaticSelectiveDiscovery::resume(&contract, Some(&data), &invalid, &ctx()).unwrap_err();
    assert_eq!(err.detail(), Some("selective_discovery.receipt_invalid"));

    // A receipt dropped: the replay needs a statistic the state does not hold.
    let mut dropped = state.clone();
    dropped.receipts.truncate(1);
    dropped.receipts_identity =
        antecedent_discovery::selective::StatisticStore::receipts_identity(&dropped.receipts);
    let err =
        StaticSelectiveDiscovery::resume(&contract, Some(&data), &dropped, &ctx()).unwrap_err();
    assert_eq!(err.detail(), Some("selective_discovery.replay_miss"));

    // Every statistic rewritten to "independent" and the receipt identity resealed: each
    // value is a valid statistic, the container is consistent, and the replay reaches a
    // different graph than the one recorded.
    let mut mutated = state.clone();
    for receipt in &mut mutated.receipts {
        receipt.p_value = 1.0;
    }
    mutated.receipts_identity =
        antecedent_discovery::selective::StatisticStore::receipts_identity(&mutated.receipts);
    let err =
        StaticSelectiveDiscovery::resume(&contract, Some(&data), &mutated, &ctx()).unwrap_err();
    assert_eq!(err.detail(), Some("selective_discovery.replay_diverged"));

    // The same edit without resealing is caught by the receipt identity.
    let mut unsealed = state;
    for receipt in &mut unsealed.receipts {
        receipt.p_value = 1.0;
    }
    let err =
        StaticSelectiveDiscovery::resume(&contract, Some(&data), &unsealed, &ctx()).unwrap_err();
    assert_eq!(err.detail(), Some("selective_discovery.receipts_identity"));
}

// -- temporal ---------------------------------------------------------------------------------------

/// `x_t = 0.5 x_{t-1} + e`, `y_t = 0.7 x_{t-1} + 0.3 y_{t-1} + e`, `z` white noise.
fn temporal_series(n: usize, seed: u64, missing: &[usize]) -> TimeSeriesData {
    let mut rng = Lcg(seed);
    let (mut x, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for t in 1..n {
        x[t] = 0.5 * x[t - 1] + rng.normal();
        y[t] = 0.7 * x[t - 1] + 0.3 * y[t - 1] + 0.7 * rng.normal();
        z[t] = rng.normal();
    }
    for &t in missing {
        y[t] = f64::NAN;
    }
    TimeSeriesData::from_f64_columns(
        [("x", x.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())],
        1,
    )
    .unwrap()
}

fn pcmci_contract() -> TemporalDiscoveryContract {
    let mut contract = TemporalDiscoveryContract::new(
        TemporalAlgorithm::Pcmci,
        vars(&[0, 1, 2]),
        CiTestSpec::named("parcorr"),
        SEED,
    );
    contract.constraints.temporal.max_lag = Lag::from_raw(2);
    contract.constraints.temporal.min_lag = Lag::from_raw(1);
    contract.constraints.alpha = 0.001;
    contract
}

fn full_temporal(
    contract: &TemporalDiscoveryContract,
    data: &TimeSeriesData,
) -> (DiscoveryOutcome, WorkUnits) {
    let mut adapter = TemporalSelectiveDiscovery::new();
    let report = adapter.run(contract, data, &ctx()).unwrap();
    (adapter.outcome().unwrap().clone(), report.work)
}

#[test]
fn temporal_run_finds_the_planted_lagged_link_and_counts_search_work() {
    let data = temporal_series(400, 51, &[]);
    let (outcome, work) = full_temporal(&pcmci_contract(), &data);
    // Planted: x at lag 1 causes y at lag 0.
    let planted = outcome.structure.edges().iter().any(|e| {
        e.is_directed()
            && ((e.a.variable == 0
                && e.a.kind == 1
                && e.a.aux == 1
                && e.b.variable == 1
                && e.b.aux == 0
                && e.at_a == Mark::Tail)
                || (e.b.variable == 0
                    && e.b.kind == 1
                    && e.b.aux == 1
                    && e.a.variable == 1
                    && e.a.aux == 0
                    && e.at_b == Mark::Tail))
    });
    assert!(planted, "{:?}", outcome.structure.edges());
    assert!(work.ci_tests_attempted > 0);
    assert!(work.search_passes >= 1);
    assert_eq!(work.score_evaluations, 0);
}

#[test]
fn temporal_threshold_lag_and_mask_changes_match_full_runs() {
    let data = temporal_series(300, 52, &[40, 41, 120, 200]);
    let mut adapter = TemporalSelectiveDiscovery::new();
    let base = pcmci_contract();
    adapter.run(&base, &data, &ctx()).unwrap();

    // Threshold: statistics reused.
    let mut tighter = base.clone();
    tighter.constraints.alpha = 0.0001;
    let report = adapter.run(&tighter, &data, &ctx()).unwrap();
    assert!(report.plan.decision(DiscoveryStage::Statistics).is_reused());
    let (full, _) = full_temporal(&tighter, &data);
    assert_algorithmic_match(adapter.outcome().unwrap(), &full);

    // Minimum lag with the same frame depth: statistics still reused.
    let mut min_lag = tighter.clone();
    min_lag.constraints.temporal.min_lag = Lag::from_raw(2);
    let report = adapter.run(&min_lag, &data, &ctx()).unwrap();
    assert!(report.plan.decision(DiscoveryStage::Statistics).is_reused());
    let (full, _) = full_temporal(&min_lag, &data);
    assert_algorithmic_match(adapter.outcome().unwrap(), &full);

    // Maximum lag changes the lagged frame: nothing is reused, and the result is a full run's.
    let mut longer = min_lag.clone();
    longer.constraints.temporal.max_lag = Lag::from_raw(3);
    let report = adapter.run(&longer, &data, &ctx()).unwrap();
    match report.plan.decision(DiscoveryStage::Statistics) {
        StageDecision::Recomputed { because } => assert!(because.contains(&"frame_depth")),
        StageDecision::Reused => panic!("a new lag frame must not reuse statistics"),
    }
    let (full, _) = full_temporal(&longer, &data);
    assert_algorithmic_match(adapter.outcome().unwrap(), &full);

    // Mask type changes which rows each test sees.
    let mut masked = longer.clone();
    masked.constraints.mask_type = CiMaskType::Y;
    let report = adapter.run(&masked, &data, &ctx()).unwrap();
    match report.plan.decision(DiscoveryStage::Statistics) {
        StageDecision::Recomputed { because } => assert!(because.contains(&"mask")),
        StageDecision::Reused => panic!("a new mask type must not reuse statistics"),
    }
    let (full, _) = full_temporal(&masked, &data);
    assert_algorithmic_match(adapter.outcome().unwrap(), &full);
}

#[test]
fn temporal_resume_and_data_change_follow_the_static_rules() {
    let data = temporal_series(300, 53, &[]);
    let contract = pcmci_contract();
    let mut adapter = TemporalSelectiveDiscovery::new();
    adapter.run(&contract, &data, &ctx()).unwrap();
    let state = adapter.export_state().unwrap();

    let (resumed, report) =
        TemporalSelectiveDiscovery::resume(&contract, Some(&data), &state, &ctx()).unwrap();
    assert_eq!(report.work.ci_tests_attempted, 0);
    assert_eq!(resumed.outcome().unwrap().result_identity, state.result_identity);

    let other = temporal_series(300, 54, &[]);
    let err =
        TemporalSelectiveDiscovery::resume(&contract, Some(&other), &state, &ctx()).unwrap_err();
    assert_eq!(err.detail(), Some("selective_discovery.data_mismatch"));
    let err = TemporalSelectiveDiscovery::resume(&contract, None, &state, &ctx()).unwrap_err();
    assert_eq!(err.detail(), Some("selective_discovery.raw_data_missing"));

    // New data: the retained statistics are not used and the result is a fresh run's.
    let report = adapter.run(&contract, &other, &ctx()).unwrap();
    // Within-run repeats of one coordinate are hits too, so "reused" is not zero for a fresh
    // run; what must hold is that nothing came from the old data: the counts equal a fresh run's.
    let (fresh, fresh_work) = full_temporal(&contract, &other);
    assert_eq!(report.work.ci_tests_reused, fresh_work.ci_tests_reused);
    assert_eq!(report.work.ci_tests_attempted, fresh_work.ci_tests_attempted);
    assert_algorithmic_match(adapter.outcome().unwrap(), &fresh);
}

#[test]
fn exhausted_search_bound_is_preserved_identically_by_selective_and_full_runs() {
    // A Possible-D-Sep budget of zero nodes cannot expand any node. Whatever the algorithm
    // does with that (a typed failure or a recorded diagnostic), a selective run must do
    // exactly the same as a full run.
    let data = collider_table(200, 61);
    let mut contract = StaticDiscoveryContract::new(
        StaticAlgorithm::Fci,
        vars(&[0, 1, 2, 3]),
        CiTestSpec::named("parcorr"),
        SEED,
    );
    contract.possible_d_sep_max_nodes = 0;
    contract.discriminating_path_budget.max_paths_per_edge = 1;
    contract.discriminating_path_budget.max_len = 3;
    let mut selective = StaticSelectiveDiscovery::new();
    let mut warm = contract.clone();
    warm.possible_d_sep_max_nodes = 1_000;
    warm.discriminating_path_budget =
        antecedent_discovery::discriminating_paths::DiscriminatingPathBudget::default();
    selective.run(&warm, &data, &ctx()).unwrap();
    let from_retained = selective.run(&contract, &data, &ctx());
    let mut fresh = StaticSelectiveDiscovery::new();
    let from_scratch = fresh.run(&contract, &data, &ctx());
    match (from_retained, from_scratch) {
        (Ok(_), Ok(_)) => {
            assert_algorithmic_match(selective.outcome().unwrap(), fresh.outcome().unwrap());
            let outcome = fresh.outcome().unwrap();
            // The bound's exhaustion is a recorded fact, and a transport request sees it.
            if outcome.search_incomplete {
                let refusal = request_selection_diagram(outcome, &[]).unwrap_err();
                assert_eq!(refusal.detail, "selective_discovery.search_incomplete");
            }
        }
        (Err(a), Err(b)) => assert_eq!(a, b),
        (a, b) => panic!("selective and full disagree: {a:?} vs {b:?}"),
    }
}

// -- background knowledge --------------------------------------------------------------------

fn contemp(source: u32, target: u32) -> LaggedLink {
    LaggedLink {
        source: VariableId::from_raw(source),
        source_lag: Lag::CONTEMPORANEOUS,
        target: VariableId::from_raw(target),
        target_lag: Lag::CONTEMPORANEOUS,
    }
}

fn tiers(layers: &[&[u32]]) -> Arc<[Arc<[VariableId]>]> {
    layers.iter().map(|ids| ids.iter().copied().map(VariableId::from_raw).collect()).collect()
}

/// `0 -> 1 -> 2` as a d-separation oracle: no v-structure, so PC returns the undirected chain.
fn chain_contract(algorithm: StaticAlgorithm) -> StaticDiscoveryContract {
    let oracle = DsepOracle { parents: vec![vec![], vec![0], vec![1]], nodes: vec![0, 1, 2] };
    let mut contract = StaticDiscoveryContract::new(
        algorithm,
        vars(&[0, 1, 2]),
        CiTestSpec::declared("dsep-oracle:chain", Arc::new(oracle)),
        SEED,
    );
    contract.fdr = None;
    contract
}

/// The marks `(at a, at b)` of the edge between variables `a < b`, if it is in the structure.
fn marks(outcome: &DiscoveryOutcome, a: u32, b: u32) -> Option<(Mark, Mark)> {
    outcome
        .structure
        .edges()
        .iter()
        .find(|e| (e.a.variable, e.b.variable) == (a, b))
        .map(|e| (e.at_a, e.at_b))
}

#[test]
fn a_required_link_orients_the_chain_and_meek_propagates_it() {
    let data = collider_table(40, 70);
    let (plain, _) = full_static(&chain_contract(StaticAlgorithm::Pc), &data);
    assert_eq!(marks(&plain, 0, 1), Some((Mark::Tail, Mark::Tail)));

    let mut contract = chain_contract(StaticAlgorithm::Pc);
    contract.constraints.required = Arc::from([contemp(0, 1)]);
    let (outcome, _) = full_static(&contract, &data);
    // 0 -> 1 is stated; 0 and 2 are non-adjacent, so Meek rule 1 gives 1 -> 2.
    assert_eq!(marks(&outcome, 0, 1), Some((Mark::Tail, Mark::Arrow)));
    assert_eq!(marks(&outcome, 1, 2), Some((Mark::Tail, Mark::Arrow)));
    assert!(outcome.structure.is_fully_directed());
    assert_ne!(outcome.structure_identity, plain.structure_identity);
}

#[test]
fn tiers_orient_pc_and_place_arrowheads_only_in_fci() {
    let data = collider_table(40, 71);
    let mut pc = chain_contract(StaticAlgorithm::Pc);
    pc.constraints.tiers = tiers(&[&[0], &[1], &[2]]);
    let (outcome, _) = full_static(&pc, &data);
    assert_eq!(marks(&outcome, 0, 1), Some((Mark::Tail, Mark::Arrow)));
    assert_eq!(marks(&outcome, 1, 2), Some((Mark::Tail, Mark::Arrow)));

    // In a PAG a later tier excludes a tail at the later end, so the earlier end keeps its
    // circle; rule R1 then turns 1 o-> 2 after the arrowhead at 1 into 1 -> 2.
    let mut fci = chain_contract(StaticAlgorithm::Fci);
    fci.constraints.tiers = tiers(&[&[0], &[1], &[2]]);
    let (outcome, _) = full_static(&fci, &data);
    assert_eq!(marks(&outcome, 0, 1), Some((Mark::Circle, Mark::Arrow)));
    assert_eq!(marks(&outcome, 1, 2), Some((Mark::Tail, Mark::Arrow)));
}

#[test]
fn a_required_link_in_fci_is_adjacency_plus_tail_and_arrow() {
    let data = collider_table(40, 72);
    let mut fci = chain_contract(StaticAlgorithm::Fci);
    fci.constraints.required = Arc::from([contemp(1, 0)]);
    let (outcome, _) = full_static(&fci, &data);
    assert_eq!(marks(&outcome, 0, 1), Some((Mark::Arrow, Mark::Tail)));
}

#[test]
fn refused_knowledge_names_its_reason_and_detail() {
    let refusal = |contract: &StaticDiscoveryContract| contract.validate().unwrap_err();
    let mut contradictory = chain_contract(StaticAlgorithm::Pc);
    contradictory.constraints.required = Arc::from([contemp(0, 1)]);
    contradictory.constraints.forbidden = Arc::from([contemp(0, 1)]);
    let error = refusal(&contradictory);
    assert_eq!(error.detail(), Some("selective_discovery.background_knowledge_contradictory"));
    assert_eq!(error.code(), Some("invalid_argument"));

    let mut unknown = chain_contract(StaticAlgorithm::Pc);
    unknown.constraints.forbidden = Arc::from([contemp(0, 9)]);
    let error = refusal(&unknown);
    assert_eq!(error.detail(), Some("selective_discovery.background_knowledge_unknown_variable"));
    assert_eq!(error.code(), Some("unknown_variable"));

    // A static contract with a lagged link has nothing to search it in.
    let mut lagged = chain_contract(StaticAlgorithm::Pc);
    lagged.constraints.forbidden =
        Arc::from([LaggedLink { source_lag: Lag::from_raw(1), ..contemp(0, 1) }]);
    let error = refusal(&lagged);
    assert_eq!(error.detail(), Some("selective_discovery.background_knowledge_not_applicable"));
    assert_eq!(error.code(), Some("option_not_applicable"));
    // The parent limit is enforced by PC and FCI, so their contracts accept it.
    for algorithm in [StaticAlgorithm::Pc, StaticAlgorithm::Fci] {
        let mut capped = chain_contract(algorithm);
        capped.constraints.max_parents = Some(1);
        capped.validate().unwrap();
    }

    // PCMCI enforces a parent limit, but a required link outside its lag window cannot hold.
    let mut series = TemporalDiscoveryContract::new(
        TemporalAlgorithm::Pcmci,
        vars(&[0, 1]),
        CiTestSpec::named("parcorr"),
        SEED,
    );
    series.constraints.max_parents = Some(1);
    series.validate().unwrap();
    series.constraints.required = Arc::from([contemp(0, 1)]);
    let error = series.validate().unwrap_err();
    assert_eq!(error.detail(), Some("selective_discovery.background_knowledge_not_applicable"));
}

#[test]
fn the_contract_identity_changes_with_each_background_knowledge_statement() {
    let base = chain_contract(StaticAlgorithm::Pc);
    let identity = base.identity().unwrap();
    let mut seen = vec![identity];
    let mut edit = |change: &dyn Fn(&mut StaticDiscoveryContract)| {
        let mut contract = base.clone();
        change(&mut contract);
        let id = contract.identity().unwrap();
        assert!(!seen.contains(&id), "an edit left the identity unchanged");
        seen.push(id);
    };
    edit(&|c| c.constraints.required = Arc::from([contemp(0, 1)]));
    edit(&|c| c.constraints.forbidden = Arc::from([contemp(2, 0)]));
    edit(&|c| c.constraints.tiers = tiers(&[&[0], &[1, 2]]));

    // The parent limit is background knowledge too.
    let series = TemporalDiscoveryContract::new(
        TemporalAlgorithm::Pcmci,
        vars(&[0, 1]),
        CiTestSpec::named("parcorr"),
        SEED,
    );
    let mut capped = series.clone();
    capped.constraints.max_parents = Some(2);
    assert_ne!(series.identity().unwrap(), capped.identity().unwrap());
    let mut capped_again = series.clone();
    capped_again.constraints.max_parents = Some(3);
    assert_ne!(capped.identity().unwrap(), capped_again.identity().unwrap());
}

// -- background knowledge in PCMCI+ -------------------------------------------------------------

/// `x` white noise, `y = 0.9 x + 0.4 e` in the same step, `z` white noise: one contemporaneous
/// edge `x - y` that no collider can orient.
fn contemporaneous_pair(n: usize, seed: u64) -> TimeSeriesData {
    let mut rng = Lcg(seed);
    let (mut x, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for t in 0..n {
        x[t] = rng.normal();
        y[t] = 0.9 * x[t] + 0.4 * rng.normal();
        z[t] = rng.normal();
    }
    TimeSeriesData::from_f64_columns(
        [("x", x.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())],
        1,
    )
    .unwrap()
}

fn pcmci_plus_contract() -> TemporalDiscoveryContract {
    let mut contract = TemporalDiscoveryContract::new(
        TemporalAlgorithm::PcmciPlus,
        vars(&[0, 1, 2]),
        CiTestSpec::named("parcorr"),
        SEED,
    );
    contract.constraints.temporal.min_lag = Lag::CONTEMPORANEOUS;
    contract.constraints.temporal.max_lag = Lag::from_raw(1);
    contract.constraints.alpha = 0.001;
    contract.fdr = None;
    contract
}

/// Marks `(at x, at y)` of the contemporaneous edge between variables 0 and 1.
fn contemporaneous_marks(outcome: &DiscoveryOutcome) -> Option<(Mark, Mark)> {
    outcome.structure.edges().iter().find_map(|e| {
        let both_now = e.a.aux == 0 && e.b.aux == 0;
        match (e.a.variable, e.b.variable) {
            (0, 1) if both_now => Some((e.at_a, e.at_b)),
            (1, 0) if both_now => Some((e.at_b, e.at_a)),
            _ => None,
        }
    })
}

#[test]
fn pcmci_plus_orients_a_contemporaneous_edge_by_knowledge() {
    let data = contemporaneous_pair(1500, 90);
    let (plain, _) = full_temporal(&pcmci_plus_contract(), &data);
    assert_eq!(contemporaneous_marks(&plain), Some((Mark::Tail, Mark::Tail)));

    let mut tiered = pcmci_plus_contract();
    tiered.constraints.tiers = tiers(&[&[0], &[1]]);
    let (outcome, _) = full_temporal(&tiered, &data);
    assert_eq!(contemporaneous_marks(&outcome), Some((Mark::Tail, Mark::Arrow)));

    let mut required = pcmci_plus_contract();
    required.constraints.required = Arc::from([contemp(1, 0)]);
    let (outcome, _) = full_temporal(&required, &data);
    assert_eq!(contemporaneous_marks(&outcome), Some((Mark::Arrow, Mark::Tail)));
    assert_ne!(outcome.result_identity, plain.result_identity);

    let mut forbidden = pcmci_plus_contract();
    forbidden.constraints.forbidden = Arc::from([contemp(0, 1)]);
    let (outcome, _) = full_temporal(&forbidden, &data);
    assert_eq!(contemporaneous_marks(&outcome), Some((Mark::Arrow, Mark::Tail)));
}

#[test]
fn a_required_link_is_in_the_output_even_when_its_own_test_would_drop_it() {
    // z is independent noise: its lagged link to y fails every test, but the statement stands.
    let data = contemporaneous_pair(1500, 91);
    let lagged = LaggedLink { source_lag: Lag::from_raw(1), ..contemp(2, 1) };
    let mut contract = pcmci_plus_contract();
    contract.constraints.required = Arc::from([lagged]);
    let (outcome, _) = full_temporal(&contract, &data);
    let present = outcome.structure.edges().iter().any(|e| {
        let vars = (e.a.variable, e.b.variable);
        (vars == (2, 1) || vars == (1, 2)) && (e.a.aux == 1 || e.b.aux == 1)
    });
    assert!(present, "{:?}", outcome.structure.edges());
}

#[test]
fn contradictory_knowledge_is_refused_by_pcmci_plus() {
    let mut contract = pcmci_plus_contract();
    contract.constraints.required = Arc::from([contemp(0, 1)]);
    contract.constraints.tiers = tiers(&[&[1], &[0]]);
    let error = contract.validate().unwrap_err();
    assert_eq!(error.detail(), Some("selective_discovery.background_knowledge_contradictory"));
}
