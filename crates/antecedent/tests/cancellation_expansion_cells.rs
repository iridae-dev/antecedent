//! Cancellation of the facade-level 2.2 E-cells (cost and cancellation checks): preflight and
//! the rank-drop estimate, batch retarget and its max-t evaluator, the finite-action inverse
//! outcome and the tier diagnostics. A context cancelled before the call yields the cell's
//! typed cancellation stop and no partial report; a context tripped after `k` clean polls
//! stops at the loop's own granularity; and a sweep over every `k` shows each stop is the
//! typed one (never another error, never a half-built result) until the call completes, after
//! which every larger budget completes too.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{
    ActionSpec, BatchRetargetRequest, BatchStudy, CausalError, ColumnPriority, EstimatorId,
    ForwardEvaluation, InverseQuery, PreflightInput, RankDropPolicy, RefuteSuite, RetargetClaim,
    Study, SupportBasis, TargetDirection, classify_inverse_outcome, estimate_with_rank_drop,
    fit_diagnostics_design, max_t_critical_value, plan_rank_drop, preflight_design,
    tier_diagnostics,
};
use antecedent_core::{
    AverageEffectQuery, CancellationToken, CausalRng, ExecutionContext, StreamDomain,
    SupportStatus, VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_estimate::JointCovariance;
use antecedent_graph::{Dag, DenseNodeId, TieredBackground, WithinTier};
use antecedent_kernels::standard_normal;

fn tripping(checks: usize) -> ExecutionContext {
    let mut ctx = ExecutionContext::for_tests(11);
    ctx.cancellation = CancellationToken::cancel_after_checks(checks);
    ctx
}

fn pre_cancelled() -> ExecutionContext {
    let ctx = ExecutionContext::for_tests(11);
    ctx.cancellation.cancel();
    ctx
}

fn stream(seed: u64) -> CausalRng {
    ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Test, 0xD4)
}

fn id(data: &TabularData, name: &str) -> VariableId {
    data.schema().id_of(name).unwrap()
}

/// Run `call` under a token that trips after `k` clean polls for every `k` until it completes
/// (at most `limit` polls). Every earlier `k` must stop with the typed cancellation judged by
/// `is_stop`; returns the first `k` that completes, which is the number of polls the call makes.
fn sweep<T, E: std::fmt::Debug>(
    limit: usize,
    call: impl Fn(&ExecutionContext) -> Result<T, E>,
    is_stop: impl Fn(&E) -> bool,
) -> usize {
    for checks in 0..limit {
        match call(&tripping(checks)) {
            Ok(_) => {
                // A larger budget never stops what a smaller one completed.
                assert!(call(&tripping(checks + 3)).is_ok());
                return checks;
            }
            Err(error) => assert!(is_stop(&error), "k = {checks}: not the typed stop: {error:?}"),
        }
    }
    panic!("the call still polled after {limit} budgets");
}

// ---- preflight, fit diagnostics, rank-drop plan and estimate --------------------------------

struct Collinear {
    data: TabularData,
}

/// Binary `t` driven by `a`, outcome linear, `c = 2a - 3b + 1` exactly.
fn collinear(seed: u64) -> Collinear {
    let n = 400;
    let mut rng = stream(seed);
    let a: Vec<f64> = (0..n).map(|_| standard_normal(&mut rng)).collect();
    let b: Vec<f64> = (0..n).map(|_| standard_normal(&mut rng)).collect();
    let c: Vec<f64> = a.iter().zip(&b).map(|(x, z)| 2.0 * x - 3.0 * z + 1.0).collect();
    let t: Vec<f64> =
        a.iter().map(|x| f64::from(rng.next_f64() < 1.0 / (1.0 + (-0.8 * x).exp()))).collect();
    let y: Vec<f64> =
        (0..n).map(|i| 1.5 * t[i] + a[i] + 0.5 * b[i] + 0.3 * standard_normal(&mut rng)).collect();
    let data = TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("a", a.as_slice()),
        ("b", b.as_slice()),
        ("c", c.as_slice()),
    ])
    .unwrap();
    Collinear { data }
}

fn is_cancelled(error: &CausalError) -> bool {
    matches!(error, CausalError::Cancelled { .. })
        || error.peeled().reason_code() == Some(antecedent_core::reason_code!("cancelled_no_claim"))
}

/// The rank scan polls per column, the fit diagnostics on entry, and the rank-drop estimate
/// threads one context through the plan, the span check, the diagnostic fit and the study.
#[test]
fn preflight_and_the_rank_drop_estimate_stop_typed_at_every_poll() {
    let fixture = collinear(1);
    let data = &fixture.data;
    let adjustment = [id(data, "a"), id(data, "b"), id(data, "c")];
    let input =
        PreflightInput::binary_effect(data, id(data, "t"), id(data, "y"), &adjustment, 0.0, 1.0);
    let order = RankDropPolicy { priority: ColumnPriority::AdjustmentOrder };

    assert!(is_cancelled(&preflight_design(&input, &pre_cancelled()).unwrap_err()));
    assert!(is_cancelled(&fit_diagnostics_design(&input, &pre_cancelled()).unwrap_err()));
    assert!(is_cancelled(&plan_rank_drop(&input, &order, &pre_cancelled()).unwrap_err()));
    for estimator in [EstimatorId::LinearAdjustmentAte, EstimatorId::Aipw] {
        let stopped = estimate_with_rank_drop(&input, &order, estimator, &pre_cancelled());
        assert!(is_cancelled(&stopped.unwrap_err()), "{estimator:?}");
    }

    let scan = sweep(500, |ctx| preflight_design(&input, ctx), is_cancelled);
    assert!(scan >= 2, "the scan polls once per column, so a one-poll budget must stop it");
    sweep(500, |ctx| plan_rank_drop(&input, &order, ctx), is_cancelled);
    sweep(500, |ctx| fit_diagnostics_design(&input, ctx), is_cancelled);
    for estimator in [EstimatorId::LinearAdjustmentAte, EstimatorId::Aipw] {
        let polls = sweep(
            2000,
            |ctx| estimate_with_rank_drop(&input, &order, estimator, ctx),
            is_cancelled,
        );
        assert!(polls >= 2, "{estimator:?}: the plan scan alone polls once per column");
    }
}

// ---- batch preflight and batch retarget -----------------------------------------------------

const T1: u32 = 0;
const T2: u32 = 1;
const Y1: u32 = 2;
const Z: u32 = 3;

/// Two binary treatments, one outcome, one confounder.
fn batch_data(n: usize, seed: u64) -> (TabularData, Dag) {
    let mut rng = stream(seed);
    let (mut t1, mut t2, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let zi = standard_normal(&mut rng);
        z[i] = zi;
        t1[i] = f64::from(rng.next_f64() < 1.0 / (1.0 + (0.2 - 0.8 * zi).exp()));
        t2[i] = f64::from(rng.next_f64() < 1.0 / (1.0 + (-0.1 + 0.6 * zi).exp()));
        y[i] = 2.0 * t1[i] + 0.5 * t2[i] + zi + 0.3 * standard_normal(&mut rng);
    }
    let data = TabularData::from_f64_columns([
        ("t1", t1.as_slice()),
        ("t2", t2.as_slice()),
        ("y1", y.as_slice()),
        ("z", z.as_slice()),
    ])
    .unwrap();
    let mut graph = Dag::with_variables(4);
    for (from, to) in [(Z, T1), (Z, T2), (Z, Y1), (T1, Y1), (T2, Y1)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    (data, graph)
}

fn ate(treatment: u32) -> AverageEffectQuery {
    AverageEffectQuery::binary_ate(VariableId::from_raw(treatment), VariableId::from_raw(Y1))
}

/// The batch retarget polls once per claim, the max-t evaluator every 1024 draws, and batch
/// preflight reaches every plan's own scan.
#[test]
fn batch_retarget_max_t_and_batch_preflight_stop_typed_at_every_poll() {
    let (data, graph) = batch_data(300, 21);
    let ctx = ExecutionContext::for_tests(21);
    let prepared = BatchStudy::new(data.clone(), graph)
        .estimator(EstimatorId::Aipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .prepare(&[ate(T1), ate(T2)], &ctx)
        .unwrap();
    let scores = prepared.prepared_scores();
    let rows = scores.common_rows().unwrap().unwrap();
    let weights = |sign: f64| -> Vec<f64> {
        let z = data.float64_values(VariableId::from_raw(Z)).unwrap();
        rows.iter().map(|&r| (sign * 0.4 * z[r as usize]).exp()).collect()
    };
    let claim = |name: &str, query_index: usize, sign: f64| RetargetClaim {
        name: name.into(),
        query_index,
        weights: weights(sign),
        depends_on: vec![VariableId::from_raw(Z)],
    };
    let request = BatchRetargetRequest {
        claims: vec![claim("a", 0, 1.0), claim("b", 1, -1.0)],
        contrasts: vec![],
        expected_snapshot: None,
    };

    let is_stop = |e: &antecedent::BatchRetargetError| {
        (e.code, e.detail) == ("cancelled_no_claim", "batch_retarget.cancelled")
    };
    assert!(is_stop(&prepared.retarget(&scores, &request, &pre_cancelled()).unwrap_err()));
    let polls = sweep(50, |ctx| prepared.retarget(&scores, &request, ctx), is_stop);
    assert!(polls >= 2, "one poll per claim: a one-poll budget stops a two-claim family");
    let report = prepared.retarget(&scores, &request, &tripping(polls)).unwrap();
    assert!(report.complete_family().is_ok(), "the exact budget completes the whole family");

    // The unpublished max-t evaluator polls every 1024 draws: 5000 draws make five polls.
    let corr = JointCovariance { dim: 2, values: Arc::from(vec![1.0, 0.3, 0.3, 1.0]) };
    let draw_stop = |e: &antecedent::BatchRetargetError| e.detail == "batch_retarget.cancelled";
    let stopped = max_t_critical_value(&corr, 0.95, 1, 5000, &pre_cancelled()).unwrap_err();
    assert!(draw_stop(&stopped));
    let draws = sweep(50, |ctx| max_t_critical_value(&corr, 0.95, 1, 5000, ctx), draw_stop);
    assert!((2..=6).contains(&draws), "a poll every 1024 of 5000 draws, got {draws}");

    // Batch preflight and fit diagnostics reach every plan's scan under the same context.
    assert!(is_cancelled(&prepared.diagnose(&pre_cancelled()).unwrap_err()));
    assert!(is_cancelled(&prepared.diagnose_fit(&pre_cancelled()).unwrap_err()));
    sweep(500, |ctx| prepared.diagnose(ctx), is_cancelled);
}

// ---- finite-action inverse outcome ----------------------------------------------------------

fn forward(grid: &[f64]) -> ForwardEvaluation {
    ForwardEvaluation {
        outcome: "y".into(),
        population: "source".into(),
        mean_response: true,
        point_identified: true,
        dimension: 1,
        points: grid.iter().map(|a| vec![*a]).collect(),
        mean: grid.iter().map(|a| 1.0 + 2.0 * a).collect(),
        support: vec![SupportStatus::Supported; grid.len()],
        support_basis: SupportBasis::SurfaceWorstCase,
        interval: None,
        assumptions: vec!["backdoor adjustment on z".into()],
    }
}

/// The action enumeration polls once per action, and a stop reports no classification.
#[test]
fn inverse_outcome_stops_typed_at_an_action_without_a_classification() {
    let grid = [0.0, 0.5, 1.0, 1.5, 2.0];
    let actions: Vec<ActionSpec> = grid
        .iter()
        .map(|a| ActionSpec {
            label: format!("dose_{a}"),
            point: vec![*a],
            cost: a.abs(),
            constraints: Vec::new(),
        })
        .collect();
    let query = InverseQuery::TargetMean { threshold: 3.0, direction: TargetDirection::AtLeast };
    let run = |token: &CancellationToken| {
        classify_inverse_outcome(&query, &forward(&grid), &actions, None, 1e-12, token)
    };
    let cancelled = CancellationToken::new();
    cancelled.cancel();
    let stop = |e: &antecedent::InverseOutcomeError| {
        (e.code, e.detail) == ("transport_budget_cancel", "inverse.cancelled")
    };
    assert!(stop(&run(&cancelled).unwrap_err()));
    for checks in 0..grid.len() {
        assert!(
            stop(&run(&CancellationToken::cancel_after_checks(checks)).unwrap_err()),
            "{checks}"
        );
    }
    let report = run(&CancellationToken::cancel_after_checks(grid.len())).unwrap();
    assert_eq!(report.outcomes.len(), grid.len());
    assert_eq!(report.feasible, vec!["dose_1", "dose_1.5", "dose_2"]);
}

// ---- tier diagnostics -----------------------------------------------------------------------

fn tiered_data(n: usize, seed: u64) -> TabularData {
    let mut rng = stream(seed);
    let (mut z, mut u, mut t, mut y) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let common = standard_normal(&mut rng);
        z[i] = 0.7 * common + 0.7 * standard_normal(&mut rng);
        u[i] = 0.7 * common + 0.7 * standard_normal(&mut rng);
        let p = 1.0 / (1.0 + (-(0.6 * z[i] - 0.4 * u[i])).exp());
        t[i] = f64::from(rng.next_f64() < p);
        y[i] = 2.0 * t[i] + z[i] - 0.5 * u[i] + standard_normal(&mut rng);
    }
    TabularData::from_f64_columns([
        ("z", z.as_slice()),
        ("u", u.as_slice()),
        ("t", t.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap()
}

fn unknown_tiered_data(n: usize, seed: u64) -> TabularData {
    let mut rng = stream(seed);
    let (mut era, mut t, mut m, mut y) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        era[i] = standard_normal(&mut rng);
        t[i] = 0.8 * era[i] + standard_normal(&mut rng);
        m[i] = t[i] + 0.2 * era[i] + 0.5 * standard_normal(&mut rng);
        y[i] = -t[i] + 2.0 * m[i] + 0.2 * era[i] + standard_normal(&mut rng);
    }
    TabularData::from_f64_columns([
        ("era", era.as_slice()),
        ("t", t.as_slice()),
        ("m", m.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap()
}

fn tiered_result(
    data: &TabularData,
    tiers: &[Vec<&str>],
    within: WithinTier,
    aipw: bool,
) -> antecedent::StudyResult {
    let background = TieredBackground::from_named(data.schema(), tiers, within).unwrap();
    let query = AverageEffectQuery::binary_ate(id(data, "t"), id(data, "y"));
    let study = Study::tabular(data.clone())
        .tiered_background(background)
        .unwrap()
        .query(query)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0);
    let study = if aipw { study.estimator(EstimatorId::Aipw) } else { study };
    study.build().unwrap().run(&ExecutionContext::for_tests(11)).unwrap()
}

/// Reading the diagnostics polls on entry, before each tier read and once per scenario:
/// both tiered designs stop typed and report nothing, then complete.
#[test]
fn tier_diagnostics_stop_typed_at_every_poll_without_a_partial_read() {
    let co = tiered_data(1_500, 31);
    let co_result =
        tiered_result(&co, &[vec!["z", "u"], vec!["t"], vec!["y"]], WithinTier::CoDetermined, true);
    let unknown = unknown_tiered_data(1_000, 32);
    let unknown_result = tiered_result(
        &unknown,
        &[vec!["era"], vec!["t", "m"], vec!["y"]],
        WithinTier::Unknown,
        false,
    );
    let stop = |e: &antecedent::TierDiagnosticsError| {
        (e.code, e.detail) == ("cancelled_no_claim", "tier_diagnostics.cancelled")
    };
    for (name, result, minimum) in
        [("co-determined", &co_result, 2), ("unknown", &unknown_result, 3)]
    {
        assert!(stop(&tier_diagnostics(result, &pre_cancelled()).unwrap_err()), "{name}");
        let polls = sweep(50, |ctx| tier_diagnostics(result, ctx), stop);
        assert!(polls >= minimum, "{name}: {polls} polls");
    }
}
