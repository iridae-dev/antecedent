//! Transport across a discovered equivalence class, from data through the facade.
//!
//! Hand oracles: a d-separation oracle over a declared DAG makes PC return exactly that DAG's
//! CPDAG. The chain `0 - 1 - 2` has no v-structure, so its class holds the three DAGs that
//! create no collider; the collider `0 -> 2 <- 1` is compelled and is its own class. The
//! structural verdicts per member are the ones the structural identifier fixes for the same
//! DAGs and the same query (see the identify crate's discovered-transport tests).

use std::sync::Arc;

use antecedent::analysis::discovered_transport::{
    DiscoveredTransportError, DiscoveredTransportRefusal, EnvelopeStatus, MemberVerdict,
    TEMPORAL_STRUCTURE_DETAIL, discover_and_transport, discovered_transport_refusal, formula_kind,
    transport_discovered_structure,
};
use antecedent_core::{
    ContinuousDomain, ExecutionContext, GridSpec, ResponseFunctional, ResponseQuery,
    TransportQuery, VariableId,
};
use antecedent_data::{TabularData, TimeSeriesData};
use antecedent_discovery::selective::{
    CiTestSpec, StaticAlgorithm, StaticDiscoveryContract, StaticSelectiveDiscovery,
    TemporalAlgorithm, TemporalDiscoveryContract, TemporalSelectiveDiscovery,
};
use antecedent_stats::{
    CiBatchRequest, CiBatchResult, CiResult, CiWorkspace, ConditionalIndependenceTest,
    PreparedCiTest, StatsError,
};

const SEED: u64 = 4_417;

// -- oracle ---------------------------------------------------------------------------------------

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

// -- data and declarations --------------------------------------------------------------------------

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

/// `n_vars` named columns of independent noise: the oracle test ignores the values, so the
/// table only fixes the shape and the content identity.
fn table(n_vars: usize, rows: usize, seed: u64) -> TabularData {
    let mut rng = Lcg(seed);
    let columns: Vec<(String, Vec<f64>)> =
        (0..n_vars).map(|i| (format!("v{i}"), (0..rows).map(|_| rng.normal()).collect())).collect();
    TabularData::from_f64_columns(columns.iter().map(|(n, c)| (n.as_str(), c.as_slice()))).unwrap()
}

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn vars(n: u32) -> Vec<VariableId> {
    (0..n).map(VariableId::from_raw).collect()
}

fn contract(
    algorithm: StaticAlgorithm,
    label: &str,
    parents: Vec<Vec<usize>>,
) -> StaticDiscoveryContract {
    let n = u32::try_from(parents.len()).unwrap();
    let mut contract = StaticDiscoveryContract::new(
        algorithm,
        vars(n),
        CiTestSpec::declared(label, Arc::new(DsepOracle { parents })),
        SEED,
    );
    contract.fdr = None;
    contract
}

fn chain3() -> StaticDiscoveryContract {
    contract(StaticAlgorithm::Pc, "dsep:chain3", vec![vec![], vec![0], vec![1]])
}

fn collider3() -> StaticDiscoveryContract {
    contract(StaticAlgorithm::Pc, "dsep:collider3", vec![vec![], vec![], vec![0, 1]])
}

fn chain4() -> StaticDiscoveryContract {
    contract(StaticAlgorithm::Pc, "dsep:chain4", vec![vec![], vec![0], vec![1], vec![2]])
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(SEED)
}

/// `E[outcome | do(v0)]` from an experiment on `v0`, trial to clinic.
fn query(outcome: u32) -> TransportQuery {
    TransportQuery::new(
        ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: v(outcome),
            treatment: ContinuousDomain::new(v(0), GridSpec::Values(Arc::from([0.0, 1.0]))),
        }),
        "trial",
        "clinic",
        [v(0)],
    )
}

fn refused(error: DiscoveredTransportError) -> DiscoveredTransportRefusal {
    match error {
        DiscoveredTransportError::Refused(refusal) => refusal,
        other => panic!("expected a scientific refusal, got {other}"),
    }
}

// -- envelopes from data ----------------------------------------------------------------------------

#[test]
fn a_chain_is_discovered_as_a_class_and_each_member_carries_its_structural_verdict() {
    let run = discover_and_transport(&chain3(), &table(3, 60, 1), &[v(1)], &query(2), 64, &ctx())
        .unwrap();
    let report = run.transport.unwrap();
    let envelope = report.envelope();

    assert_eq!(report.kind(), "structural_envelope");
    assert_eq!(report.selection_targets(), &[v(1)]);
    assert!(report.assumptions().iter().any(|a| a.starts_with("causal sufficiency")));
    assert!(envelope.enumeration_complete());
    assert_eq!(envelope.members().len(), 3);
    assert_eq!(envelope.unresolved().len(), 2);
    assert_eq!(envelope.status(), EnvelopeStatus::IdentifiedInEveryMember);
    // The class does not determine the estimand's expression.
    assert!(!envelope.formulas_agree());

    let kind_of = |edges: &[(u32, u32)]| -> (String, &'static str) {
        let want: Vec<(VariableId, VariableId)> =
            edges.iter().map(|&(p, c)| (v(p), v(c))).collect();
        let member = envelope
            .members()
            .iter()
            .find(|m| m.directed_edges == want)
            .unwrap_or_else(|| panic!("member {edges:?} missing"));
        match &member.verdict {
            MemberVerdict::Identified { rule, formula } => (rule.clone(), formula_kind(formula)),
            other => panic!("{edges:?}: {other:?}"),
        }
    };
    // 0 -> 1 -> 2: the mediator's mechanism differs, so population factors are recursive.
    assert_eq!(kind_of(&[(0, 1), (1, 2)]).1, "recursive_factorization");
    // 0 <- 1 -> 2: Markovian, so the truncated factorization identifies it.
    assert!(matches!(
        envelope.members().iter().find(|m| m.directed_edges == [(v(1), v(0)), (v(1), v(2))]),
        Some(m) if matches!(m.verdict, MemberVerdict::Identified { .. })
    ));
    // 0 <- 1 <- 2: the varying mechanism is not an ancestor of the outcome.
    let (rule, kind) = kind_of(&[(1, 0), (2, 1)]);
    assert_eq!((rule.as_str(), kind), ("transport.sid.direct", "direct"));
}

#[test]
fn a_compelled_structure_is_one_member_with_one_formula() {
    let run =
        discover_and_transport(&collider3(), &table(3, 60, 2), &[v(1)], &query(2), 64, &ctx())
            .unwrap();
    let report = run.transport.unwrap();
    let envelope = report.envelope();
    assert_eq!(envelope.kind(), "structural_envelope");
    assert_eq!(envelope.members().len(), 1);
    assert!(envelope.unresolved().is_empty());
    assert_eq!(envelope.status(), EnvelopeStatus::IdentifiedInEveryMember);
    assert!(envelope.formulas_agree());
    match &envelope.members()[0].verdict {
        MemberVerdict::Identified { formula, .. } => {
            assert_eq!(formula_kind(formula), "standardize");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn the_report_identity_follows_the_class_and_not_the_rows_at_an_unchanged_class() {
    let make = |c: &StaticDiscoveryContract, data: &TabularData| {
        *discover_and_transport(c, data, &[v(1)], &query(2), 64, &ctx())
            .unwrap()
            .transport
            .unwrap()
            .identity()
    };
    let base = make(&chain3(), &table(3, 60, 3));
    // Same class from other rows of the same shape: outcome-only data keeps the envelope.
    assert_eq!(base, make(&chain3(), &table(3, 60, 4)));
    // Another class at the same shape and the same query is another report.
    assert_ne!(base, make(&collider3(), &table(3, 60, 3)));
}

#[test]
fn a_retained_outcome_serves_a_second_request_without_rediscovery() {
    let mut adapter = StaticSelectiveDiscovery::new();
    adapter.run(&chain3(), &table(3, 60, 5), &ctx()).unwrap();
    let outcome = adapter.outcome().unwrap();
    let first = transport_discovered_structure(outcome, &[v(1)], &query(2), 64).unwrap();
    let second = transport_discovered_structure(outcome, &[v(1)], &query(2), 64).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.structure_identity(), &outcome.structure_identity);
    assert_eq!(first.result_identity(), &outcome.result_identity);
}

// -- refusals and partial results ----------------------------------------------------------------

#[test]
fn an_exhausted_search_bound_refuses_and_names_the_unresolved_pairs() {
    let mut adapter = StaticSelectiveDiscovery::new();
    adapter.run(&chain3(), &table(3, 60, 6), &ctx()).unwrap();
    let mut bounded = adapter.outcome().unwrap().clone();
    bounded.search_incomplete = true;
    bounded.bounded_search_receipts = vec![("pc.depth_budget".to_owned(), "exhausted".to_owned())];

    let refusal =
        refused(transport_discovered_structure(&bounded, &[v(1)], &query(2), 64).unwrap_err());
    assert_eq!(refusal.code, "discovered_structure_unresolved");
    assert_eq!(refusal.detail, "discovered_transport.search_incomplete");
    assert_eq!(refusal.exhausted_search_bounds, 1);
    assert_eq!(refusal.unresolved.len(), 2);

    let external = discovered_transport_refusal(&refusal);
    external.validate().unwrap();
    assert_eq!(external.detail, "discovered_transport.search_incomplete");
    let offending = external.offending.as_deref().unwrap();
    assert_eq!(offending.split(';').count(), 2);
    assert!(offending.contains("undirected_in_equivalence_class"));
    assert!(external.remedy.unwrap().contains("selection diagram"));
}

#[test]
fn a_selection_target_outside_the_structure_and_a_zero_bound_are_refused() {
    let mut adapter = StaticSelectiveDiscovery::new();
    adapter.run(&chain3(), &table(3, 60, 7), &ctx()).unwrap();
    let outcome = adapter.outcome().unwrap();

    let unknown =
        refused(transport_discovered_structure(outcome, &[v(9)], &query(2), 64).unwrap_err());
    assert_eq!(unknown.detail, "discovered_transport.selection_target_unknown");
    assert_eq!(unknown.unknown_selection_targets, vec![v(9)]);
    let external = discovered_transport_refusal(&unknown);
    external.validate().unwrap();
    assert_eq!(external.offending.as_deref(), Some("v9"));

    let zero = refused(transport_discovered_structure(outcome, &[v(1)], &query(2), 0).unwrap_err());
    assert_eq!(zero.detail, "discovered_transport.member_bound_zero");
}

#[test]
fn a_member_bound_that_stops_the_enumeration_makes_no_every_member_statement() {
    // A chain of four variables has four members.
    let full = discover_and_transport(&chain4(), &table(4, 60, 8), &[v(1)], &query(3), 64, &ctx())
        .unwrap()
        .transport
        .unwrap();
    assert_eq!(full.envelope().members().len(), 4);
    assert!(full.envelope().enumeration_complete());

    let bounded =
        discover_and_transport(&chain4(), &table(4, 60, 8), &[v(1)], &query(3), 2, &ctx())
            .unwrap()
            .transport
            .unwrap();
    assert_eq!(bounded.envelope().status(), EnvelopeStatus::Partial);
    assert!(!bounded.envelope().enumeration_complete());
    assert_eq!(bounded.envelope().members().len(), 2);
    assert_eq!(bounded.envelope().member_bound(), 2);
    // The bound is part of the report: a partial report is not the complete one.
    assert_ne!(bounded.identity(), full.identity());
}

#[test]
fn a_pag_is_refused_with_its_unresolved_pairs() {
    let fci = contract(StaticAlgorithm::Fci, "dsep:chain3:fci", vec![vec![], vec![0], vec![1]]);
    let run =
        discover_and_transport(&fci, &table(3, 60, 9), &[v(1)], &query(2), 64, &ctx()).unwrap();
    let refusal = refused(run.transport.unwrap_err());
    assert_eq!(refusal.code, "discovered_structure_unresolved");
    assert_eq!(refusal.detail, "discovered_transport.pag_members_not_causal_graphs");
    assert_eq!(refusal.exhausted_search_bounds, 0);
    // The discovery that ran is kept: the refusal is of the transport request only.
    assert_eq!(run.outcome.structure.class().as_str(), "pag");
    discovered_transport_refusal(&refusal).validate().unwrap();
}

#[test]
fn a_temporal_structure_is_refused() {
    let mut rng = Lcg(77);
    let n = 160;
    let (mut a, mut b) = (vec![0.0_f64; n], vec![0.0_f64; n]);
    for t in 1..n {
        a[t] = 0.6 * a[t - 1] + rng.normal();
        b[t] = 0.7 * a[t - 1] + 0.3 * b[t - 1] + rng.normal();
    }
    let series =
        TimeSeriesData::from_f64_columns([("a", a.as_slice()), ("b", b.as_slice())], 1_000_000_000)
            .unwrap();
    let contract = TemporalDiscoveryContract::new(
        TemporalAlgorithm::Pcmci,
        vars(2),
        CiTestSpec::named("parcorr"),
        SEED,
    );
    let mut adapter = TemporalSelectiveDiscovery::new();
    adapter.run(&contract, &series, &ctx()).unwrap();
    let refusal = refused(
        transport_discovered_structure(adapter.outcome().unwrap(), &[v(1)], &query(1), 64)
            .unwrap_err(),
    );
    assert_eq!(refusal.detail, TEMPORAL_STRUCTURE_DETAIL);
    assert_eq!(refusal.code, "discovered_structure_unresolved");
    discovered_transport_refusal(&refusal).validate().unwrap();
}
