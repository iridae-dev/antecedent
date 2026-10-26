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
use antecedent::analysis::discovered_transport_estimate::{
    AnswerKind, TransportEvidence, estimate_discovered_transport,
    verify_discovered_transport_estimate,
};
use antecedent::analysis::discovered_transport_temporal::{
    TemporalClass, TemporalHorizon, estimate_discovered_temporal_transport,
    transport_temporal_class,
};
use antecedent_core::{
    ContinuousDomain, ExecutionContext, GridSpec, Lag, ResponseFunctional, ResponseQuery,
    TransportQuery, VariableId,
};
use antecedent_data::{TabularData, TimeSeriesData};
use antecedent_discovery::selective::{
    CiTestSpec, StaticAlgorithm, StaticDiscoveryContract, StaticSelectiveDiscovery,
    TemporalAlgorithm, TemporalDiscoveryContract, TemporalSelectiveDiscovery,
};
use antecedent_graph::{TemporalDag, TemporalPag};
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
fn a_pag_chain_is_enumerated_as_its_five_markov_equivalent_mags() {
    // FCI on the chain 0 - 1 - 2 returns circle endpoints throughout. The MAGs of its class
    // share the skeleton and have no collider, so the only excluded pairs of marks are those
    // with an arrowhead at 1 on both edges. Each edge is `->`, `<-` or `<->` (9 pairs); the
    // pairs with a head at 1 on both edges are {0->1, 0<->1} x {2->1, 1<->2} = 4, leaving 5:
    // the chain's three orderings and the two latent readings 0 <-> 1 -> 2 and 0 <- 1 <-> 2,
    // which imply the same independences. Collapsing them would read a circle as no
    // confounding.
    let fci = contract(StaticAlgorithm::Fci, "dsep:chain3:fci", vec![vec![], vec![0], vec![1]]);
    let run =
        discover_and_transport(&fci, &table(3, 60, 9), &[v(1)], &query(2), 64, &ctx()).unwrap();
    assert_eq!(run.outcome.structure.class().as_str(), "pag");
    let report = run.transport.unwrap();
    let envelope = report.envelope();
    assert_eq!(envelope.members().len(), 5);
    assert!(envelope.enumeration_complete());
    assert_eq!(envelope.members().iter().filter(|m| !m.bidirected_edges.is_empty()).count(), 2);
    assert!(envelope.unresolved().iter().all(|p| p.reason == "circle_endpoint"));
}

#[test]
fn a_temporal_structure_without_a_horizon_is_refused() {
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
    assert!(refusal.remedy.contains("horizon"));
    assert_eq!(refusal.code, "discovered_structure_unresolved");
    discovered_transport_refusal(&refusal).validate().unwrap();
}

// -- estimates across a class, against enumerated truth ------------------------------------------------

impl Lcg {
    fn uniform(&mut self) -> f64 {
        self.0 =
            self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (f64::from(u32::try_from(self.0 >> 40).unwrap()) + 0.5) / f64::from(1u32 << 24)
    }

    fn bernoulli(&mut self, p: f64) -> f64 {
        f64::from(u8::from(self.uniform() < p))
    }
}

/// An oracle contract over the first `observed` variables of a DAG that also has latent ones.
fn latent_contract(
    algorithm: StaticAlgorithm,
    label: &str,
    parents: Vec<Vec<usize>>,
    observed: u32,
) -> StaticDiscoveryContract {
    let mut contract = StaticDiscoveryContract::new(
        algorithm,
        vars(observed),
        CiTestSpec::declared(label, Arc::new(DsepOracle { parents })),
        SEED,
    );
    contract.fdr = None;
    contract
}

/// `x -> y <- z <- {c, d}`, `y -> r`, `z -> q`, and a latent `L` with `L -> w` and `L -> y`.
/// Variables: x0 y1 z2 c3 d4 w5 q6 r7 (latent 8).
fn latent_parents() -> Vec<Vec<usize>> {
    vec![vec![], vec![0, 2, 8], vec![3, 4], vec![], vec![], vec![8], vec![2], vec![1], vec![]]
}

const Y_BASE: f64 = 0.1;

/// `P(y = 1 | x, z, L)`.
fn p_y(x: f64, z: f64, l: f64) -> f64 {
    Y_BASE + 0.3 * x + 0.2 * z + 0.2 * l + 0.15 * x * z
}

/// `P(z = 1 | c, d)` in the source and in the target: the mechanism of `z` is what differs.
fn p_z(c: f64, d: f64, target: bool) -> f64 {
    if target { 0.85 - 0.3 * c - 0.3 * d } else { 0.15 + 0.3 * c + 0.3 * d }
}

/// Rows of the study; the treatment is randomized in both populations.
fn study_table(rows: usize, target: bool, seed: u64) -> TabularData {
    let mut rng = Lcg(seed);
    let mut columns = vec![Vec::with_capacity(rows); 8];
    for _ in 0..rows {
        let (c, d, l, x) =
            (rng.bernoulli(0.5), rng.bernoulli(0.5), rng.bernoulli(0.5), rng.bernoulli(0.5));
        let z = rng.bernoulli(p_z(c, d, target));
        let w = rng.bernoulli(0.2 + 0.6 * l);
        let y = rng.bernoulli(p_y(x, z, l));
        let q = rng.bernoulli(0.1 + 0.7 * z);
        let r = rng.bernoulli(0.2 + 0.6 * y);
        for (column, value) in columns.iter_mut().zip([x, y, z, c, d, w, q, r]) {
            column.push(value);
        }
    }
    let names: Vec<String> = (0..8).map(|i| format!("v{i}")).collect();
    TabularData::from_f64_columns(
        names.iter().map(String::as_str).zip(columns.iter().map(Vec::as_slice)),
    )
    .unwrap()
}

/// `E*[y | do(x = a)]` by enumerating the target population's structural model.
fn target_truth(a: f64) -> f64 {
    let mut total = 0.0;
    for c in [0.0, 1.0] {
        for d in [0.0, 1.0] {
            for l in [0.0, 1.0] {
                for z in [0.0, 1.0] {
                    let pz = p_z(c, d, true);
                    let weight = 0.125 * if z > 0.5 { pz } else { 1.0 - pz };
                    total += weight * p_y(a, z, l);
                }
            }
        }
    }
    total
}

fn effect_query() -> TransportQuery {
    TransportQuery::new(
        ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: v(1),
            treatment: ContinuousDomain::new(v(0), GridSpec::Values(Arc::from([0.0, 1.0]))),
        }),
        "trial",
        "clinic",
        [v(0)],
    )
}

#[test]
fn an_identified_pag_class_is_estimated_and_covers_the_enumerated_truth() {
    let fci = latent_contract(StaticAlgorithm::Fci, "dsep:latent:fci", latent_parents(), 8);
    let source = study_table(30_000, false, 11);
    let target = study_table(30_000, true, 12);
    let run = discover_and_transport(&fci, &source, &[v(2)], &effect_query(), 256, &ctx()).unwrap();
    let report = run.transport.unwrap();
    let envelope = report.envelope();
    // Latent confounding is enumerated: some members carry the bidirected edge, some do not.
    assert!(envelope.members().iter().any(|m| m.bidirected_edges.contains(&(v(1), v(5)))));
    assert!(envelope.members().iter().any(|m| m.directed_edges.contains(&(v(5), v(1)))));
    assert_eq!(envelope.status(), EnvelopeStatus::IdentifiedInEveryMember);

    let evidence = TransportEvidence::new(&source, &target);
    let estimate =
        estimate_discovered_transport(&report, &effect_query(), &evidence, &ctx()).unwrap();
    let answer = estimate.answer();
    let tolerance = 0.03;
    for (level, truth) in answer.levels.iter().zip([target_truth(0.0), target_truth(1.0)]) {
        assert!(
            level.low - tolerance <= truth && truth <= level.high + tolerance,
            "{level:?} vs {truth}"
        );
    }
    let (low, high) = answer.contrast.unwrap();
    let truth = target_truth(1.0) - target_truth(0.0);
    assert!(low - tolerance <= truth && truth <= high + tolerance, "{low} {high} vs {truth}");
    // A structural range is a labelled statement with no interval; a point has one level value.
    match answer.kind {
        AnswerKind::Point => {
            assert!(answer.levels.iter().all(|l| (l.high - l.low) < 1e-9))
        }
        AnswerKind::StructuralRange => {
            assert_eq!(answer.interval_reason, Some("structural_range_has_no_interval"))
        }
    }

    // The exported artifact replays: every member grid consumes and the answer recomputes.
    let bytes = estimate.export().unwrap();
    let replayed = verify_discovered_transport_estimate(
        &bytes,
        antecedent_expr::ExactEvaluationLimits::default(),
        &ctx(),
    )
    .unwrap();
    assert_eq!(&replayed.answer.levels, &answer.levels);
    assert_eq!(&replayed.report_identity, report.identity());
}

#[test]
fn a_selected_cause_that_may_share_a_latent_cause_with_the_outcome_is_refused_with_its_members() {
    // x -> y <- z with a latent L -> z and L -> y: z o-> y leaves z -> y and z <-> y.
    let parents = vec![vec![], vec![0, 2, 3], vec![3], vec![]];
    let fci = latent_contract(StaticAlgorithm::Fci, "dsep:latent:small", parents, 3);
    let rows = 4_000;
    let make = |target: bool, seed: u64| {
        let mut rng = Lcg(seed);
        let mut columns = vec![Vec::new(); 3];
        for _ in 0..rows {
            let (l, x) = (rng.bernoulli(0.5), rng.bernoulli(0.5));
            let z = rng.bernoulli(if target { 0.2 + 0.1 * l } else { 0.5 + 0.3 * l });
            let y = rng.bernoulli(p_y(x, z, l));
            for (column, value) in columns.iter_mut().zip([x, y, z]) {
                column.push(value);
            }
        }
        TabularData::from_f64_columns(
            ["v0", "v1", "v2"].into_iter().zip(columns.iter().map(Vec::as_slice)),
        )
        .unwrap()
    };
    let (source, target) = (make(false, 21), make(true, 22));
    let report = discover_and_transport(&fci, &source, &[v(2)], &effect_query(), 64, &ctx())
        .unwrap()
        .transport
        .unwrap();
    assert_eq!(report.envelope().status(), EnvelopeStatus::MembersDisagree);
    let error = estimate_discovered_transport(
        &report,
        &effect_query(),
        &TransportEvidence::new(&source, &target),
        &ctx(),
    )
    .unwrap_err();
    let refusal = refused(error);
    assert_eq!(refusal.detail, "discovered_transport.class_not_identified");
    assert!(!refusal.member_reasons.is_empty());
    let external = discovered_transport_refusal(&refusal);
    external.validate().unwrap();
    assert!(external.offending.unwrap().contains("member"));
}

#[test]
fn evidence_that_is_not_integer_coded_or_is_incomplete_is_refused_precisely() {
    let fci = latent_contract(StaticAlgorithm::Fci, "dsep:latent:fci2", latent_parents(), 8);
    let source = study_table(2_000, false, 31);
    let report = discover_and_transport(&fci, &source, &[v(2)], &effect_query(), 256, &ctx())
        .unwrap()
        .transport
        .unwrap();
    let fractional = table(8, 100, 5);
    let error = estimate_discovered_transport(
        &report,
        &effect_query(),
        &TransportEvidence::new(&source, &fractional),
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(refused(error).detail, "discovered_transport.evidence_invalid");
}

// -- temporal ----------------------------------------------------------------------------------------

/// `y_t <- x_{t-1}, z_{t-1}` and an autoregressive `z`; `z` is selected. Variables x0 y1 z2.
fn temporal_template() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let x1 = graph.add_lagged(v(0), Lag::from_raw(1)).unwrap();
    let y0 = graph.add_lagged(v(1), Lag::from_raw(0)).unwrap();
    let z1 = graph.add_lagged(v(2), Lag::from_raw(1)).unwrap();
    let z0 = graph.add_lagged(v(2), Lag::from_raw(0)).unwrap();
    for (from, to) in [(x1, y0), (z1, y0), (z1, z0)] {
        graph.insert_directed(from, to).unwrap();
    }
    graph
}

fn p_y_lagged(x: f64, z: f64) -> f64 {
    0.1 + 0.4 * x + 0.3 * z + 0.1 * x * z
}

fn series_table(steps: usize, target: bool, seed: u64) -> TabularData {
    let mut rng = Lcg(seed);
    let (mut xs, mut ys, mut zs): (Vec<f64>, Vec<f64>, Vec<f64>) =
        (vec![0.0], vec![0.0], vec![0.0]);
    for t in 1..steps {
        let pz = if target { 0.6 - 0.4 * zs[t - 1] } else { 0.2 + 0.5 * zs[t - 1] };
        zs.push(rng.bernoulli(pz));
        xs.push(rng.bernoulli(0.5));
        ys.push(rng.bernoulli(p_y_lagged(xs[t - 1], zs[t - 1])));
    }
    TabularData::from_f64_columns([
        ("v0", xs.as_slice()),
        ("v1", ys.as_slice()),
        ("v2", zs.as_slice()),
    ])
    .unwrap()
}

#[test]
fn a_lagged_class_is_estimated_at_the_declared_horizon_against_the_stationary_truth() {
    let template = temporal_template();
    let horizon = TemporalHorizon { horizon_steps: 1, history: None };
    let report = transport_temporal_class(
        TemporalClass::Dag(&template),
        0,
        horizon,
        &[v(2)],
        &effect_query(),
        64,
    )
    .unwrap();
    assert_eq!(report.envelope().status(), EnvelopeStatus::IdentifiedInEveryMember);
    assert_eq!(report.envelope().horizon_steps(), 1);
    assert_eq!(report.envelope().max_lag(), 1);
    let source = series_table(40_000, false, 41);
    let target = series_table(40_000, true, 42);
    let estimate = estimate_discovered_temporal_transport(
        &report,
        &effect_query(),
        &TransportEvidence::new(&source, &target),
        &ctx(),
    )
    .unwrap();
    // The target chain's stationary law: pi = 0.6 - 0.4 pi.
    let pi = 0.6 / 1.4;
    let truth = |a: f64| (1.0 - pi) * p_y_lagged(a, 0.0) + pi * p_y_lagged(a, 1.0);
    let tolerance = 0.03;
    for (level, expected) in estimate.answer().levels.iter().zip([truth(0.0), truth(1.0)]) {
        assert!(
            level.low - tolerance <= expected && expected <= level.high + tolerance,
            "{level:?} vs {expected}"
        );
    }
    assert_eq!(estimate.answer().kind, AnswerKind::Point);

    // Windows overlap, so an interval comes from the circular block bootstrap and is labelled
    // as unmeasured; the point answer is unchanged by asking for it.
    let mut with_interval = TransportEvidence::new(&source, &target);
    with_interval.bootstrap_replicates = 200;
    let bootstrapped =
        estimate_discovered_temporal_transport(&report, &effect_query(), &with_interval, &ctx())
            .unwrap();
    assert_eq!(bootstrapped.answer().levels, estimate.answer().levels);
    let interval = bootstrapped.answer().interval.expect("a block-bootstrap interval");
    assert_eq!(interval.calibration, "not_bound_to_this_execution");
    assert!(interval.low < interval.high && interval.high - interval.low < 0.1, "{interval:?}");

    let bytes = estimate.export().unwrap();
    verify_discovered_transport_estimate(
        &bytes,
        antecedent_expr::ExactEvaluationLimits::default(),
        &ctx(),
    )
    .unwrap();
}

#[test]
fn a_temporal_pag_with_a_confoundable_selected_cause_is_refused_not_estimated() {
    let mut class = TemporalPag::empty();
    let x1 = class.add_lagged(v(0), Lag::from_raw(1)).unwrap();
    let y0 = class.add_lagged(v(1), Lag::from_raw(0)).unwrap();
    let z1 = class.add_lagged(v(2), Lag::from_raw(1)).unwrap();
    class.insert_circle_arrow(x1, y0).unwrap();
    class.insert_circle_arrow(z1, y0).unwrap();
    let horizon = TemporalHorizon { horizon_steps: 1, history: None };
    let report = transport_temporal_class(
        TemporalClass::Pag(&class),
        0,
        horizon,
        &[v(2)],
        &effect_query(),
        64,
    )
    .unwrap();
    assert_eq!(report.envelope().status(), EnvelopeStatus::MembersDisagree);
    let source = series_table(5_000, false, 51);
    let target = series_table(5_000, true, 52);
    let error = estimate_discovered_temporal_transport(
        &report,
        &effect_query(),
        &TransportEvidence::new(&source, &target),
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(refused(error).detail, "discovered_transport.class_not_identified");
}

#[test]
fn a_variable_with_more_levels_than_supported_is_refused() {
    let fci = latent_contract(StaticAlgorithm::Fci, "dsep:latent:fci3", latent_parents(), 8);
    let source = study_table(2_000, false, 61);
    let report = discover_and_transport(&fci, &source, &[v(2)], &effect_query(), 256, &ctx())
        .unwrap()
        .transport
        .unwrap();
    let wide: Vec<Vec<f64>> =
        (0..8).map(|_| (0..2_000).map(|i| f64::from(i % 41)).collect()).collect();
    let names: Vec<String> = (0..8).map(|i| format!("v{i}")).collect();
    let target = TabularData::from_f64_columns(
        names.iter().map(String::as_str).zip(wide.iter().map(Vec::as_slice)),
    )
    .unwrap();
    let error = estimate_discovered_transport(
        &report,
        &effect_query(),
        &TransportEvidence::new(&source, &target),
        &ctx(),
    )
    .unwrap_err();
    let refusal = refused(error);
    assert_eq!(refusal.detail, "discovered_transport.evidence_invalid");
    assert!(refusal.remedy.contains("at most 32"));
}

// -- discovery from data, then transport --------------------------------------------------------------

/// `x -> y <- z <- {c, d}` with strong effects, so FCI with the G-squared test recovers the
/// skeleton and the colliders at this size. Variables: x0 y1 z2 c3 d4. The mechanism of `z`
/// differs between the populations.
fn strong_effects_table(rows: usize, target: bool, seed: u64) -> TabularData {
    let mut rng = Lcg(seed);
    let mut columns = vec![Vec::with_capacity(rows); 5];
    for _ in 0..rows {
        let (x, c, d) = (rng.bernoulli(0.5), rng.bernoulli(0.5), rng.bernoulli(0.5));
        let pz = if target { 0.9 - 0.4 * c - 0.4 * d } else { 0.1 + 0.4 * c + 0.4 * d };
        let z = rng.bernoulli(pz);
        let y = rng.bernoulli(0.05 + 0.4 * x + 0.35 * z + 0.1 * x * z);
        for (column, value) in columns.iter_mut().zip([x, y, z, c, d]) {
            column.push(value);
        }
    }
    let names: Vec<String> = (0..5).map(|i| format!("v{i}")).collect();
    TabularData::from_f64_columns(
        names.iter().map(String::as_str).zip(columns.iter().map(Vec::as_slice)),
    )
    .unwrap()
}

#[test]
fn fci_on_simulated_rows_then_transport_recovers_the_enumerated_target_effect() {
    let source = strong_effects_table(20_000, false, 71);
    let target = strong_effects_table(20_000, true, 72);
    let mut fci = StaticDiscoveryContract::new(
        StaticAlgorithm::Fci,
        vars(5),
        CiTestSpec::named("gsquared"),
        SEED,
    );
    fci.fdr = None;
    fci.constraints.alpha = 0.001;
    let run = discover_and_transport(&fci, &source, &[v(2)], &effect_query(), 256, &ctx()).unwrap();
    assert_eq!(run.outcome.structure.class().as_str(), "pag");
    let report = run.transport.unwrap();
    // The collider at z puts an arrowhead into z from both causes, so z -> y is visible and the
    // class is identified in every member.
    assert_eq!(report.envelope().status(), EnvelopeStatus::IdentifiedInEveryMember);
    let estimate = estimate_discovered_transport(
        &report,
        &effect_query(),
        &TransportEvidence::new(&source, &target),
        &ctx(),
    )
    .unwrap();
    // E*[y | do(x = a)] = 0.05 + 0.4 a + (0.35 + 0.1 a) E*[z], with E*[z] = 0.9 - 0.4 = 0.5.
    let truth = |a: f64| 0.05 + 0.4 * a + (0.35 + 0.1 * a) * 0.5;
    for (level, expected) in estimate.answer().levels.iter().zip([truth(0.0), truth(1.0)]) {
        assert!(
            level.low - 0.02 <= expected && expected <= level.high + 0.02,
            "{level:?} vs {expected}"
        );
    }
}

#[test]
fn retained_discovered_class_transport_uses_the_callers_pool_and_cancellation() {
    use antecedent::analysis::discovered_transport::transport_discovered_structure_with_context;
    use antecedent_identify::{IdentificationBudget, IdentificationError};
    let data = table(3, 60, 1);
    let run = discover_and_transport(&chain3(), &data, &[v(1)], &query(2), 64, &ctx()).unwrap();
    let original = run.transport.unwrap();
    let ample = ctx().with_memory_limit(Some(1 << 24));
    let before = ample.admission().available();
    let again =
        transport_discovered_structure_with_context(&run.outcome, &[v(1)], &query(2), 64, &ample)
            .unwrap();
    assert_eq!(again.identity(), original.identity());
    assert_eq!(again.envelope(), original.envelope());
    assert_eq!(ample.admission().available(), before);
    let tiny = ctx().with_memory_limit(Some(1));
    let before = tiny.admission().available();
    let error =
        transport_discovered_structure_with_context(&run.outcome, &[v(1)], &query(2), 64, &tiny)
            .unwrap_err();
    assert!(matches!(
        error,
        DiscoveredTransportError::Identification(IdentificationError::Budget {
            budget: IdentificationBudget::Memory
        })
    ));
    assert_eq!(tiny.admission().available(), before);
    ample.cancellation.cancel();
    let error =
        transport_discovered_structure_with_context(&run.outcome, &[v(1)], &query(2), 64, &ample)
            .unwrap_err();
    assert!(matches!(
        error,
        DiscoveredTransportError::Identification(IdentificationError::Cancelled)
    ));
    // Resource refusal leaves the retained discovery and its scientific report intact.
    let replay = transport_discovered_structure(&run.outcome, &[v(1)], &query(2), 64).unwrap();
    assert_eq!(replay.identity(), original.identity());
}
