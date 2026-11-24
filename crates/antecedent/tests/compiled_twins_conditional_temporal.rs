//! The conditional linear adjustment, the temporal mediation estimator and the partial-correlation
//! test are the computation their compiled routes run.
//!
//! These are kernel types the facade re-exports. The compiled conditional-effect route fits
//! `ConditionalLinearAdjustment` (`estimate`, which reads `estimate_with_means` and
//! `estimate_ate`; and `estimate_with_arm_scores` per exceedance threshold, seeded with the run's
//! master seed). The compiled temporal mediation routes fit `TemporalMediationEstimator`
//! (`estimate_with_block_bootstrap` on a DAG, `estimate_with_adjustment` per class member,
//! `prepare_shared` per posterior atom, `window_selection` for the row accounting of a series with
//! gaps; `estimate`, `estimate_with_extras` and `effect_surface` are the empty-adjustment form of
//! `estimate_with_adjustment`). The compiled discovery routes resolve the CI test by name; the
//! pairwise block test (`pairwise_mv`) answers a one-column block with
//! `PartialCorrelation::test_one`.
//!
//! Each test runs the compiled route, then calls the kernel method directly with the estimand,
//! adjustment and seed the route used, and compares every number the route publishes bit for
//! bit. The compiled route runs the kernel with a dispatch-scoped copy of the caller's context;
//! the copy keeps the master seed, so a seeded fit draws the same folds.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    reason = "the bits are compared exactly and the fixtures build columns from small counts"
)]

use std::sync::Arc;

use antecedent::estimate::{ConditionalLinearAdjustment, TemporalMediationEstimator};
use antecedent::{RefuteSuite, Study, StudyResult};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ConditionalEffectQuery, ExecutionContext, Lag,
    MediationContrast, MediationQuery, OutcomeFunctional, StreamDomain, VariableId,
};
use antecedent_data::{
    LaggedColumn, MissingDeclaration, MissingPolicy, TabularData, TimeSeriesData,
};
use antecedent_estimate::{BlockBootstrap, EffectEstimate, TemporalMediationEstimate};
use antecedent_graph::{Dag, DenseNodeId, TemporalDag, ensure_lagged};
use antecedent_kernels::standard_normal;
use antecedent_stats::{
    CiBatchRequest, CiQuery, CiResult, CiWorkspace, ConditionalIndependence, ConfidenceMethod,
    PartialCorrelation, SignificanceMethod, ci_from_name,
};

fn bits(value: f64) -> u64 {
    value.to_bits()
}

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn assert_effect(label: &str, direct: &EffectEstimate, compiled: &EffectEstimate) {
    assert_eq!(
        bits(direct.ate),
        bits(compiled.ate),
        "{label}: point {} vs {}",
        direct.ate,
        compiled.ate
    );
    assert_eq!(
        bits(direct.se_analytic()),
        bits(compiled.se_analytic()),
        "{label}: analytic se {} vs {}",
        direct.se_analytic(),
        compiled.se_analytic()
    );
    assert_eq!(
        direct.se_bootstrap().map(bits),
        compiled.se_bootstrap().map(bits),
        "{label}: bootstrap se"
    );
}

// --- conditional effect --------------------------------------------------------------------

/// `Y = 2 T + 0.3 W + 0.5 noise` with a 0/1 treatment independent of the modifier `W`.
fn conditional_data() -> TabularData {
    let mut rng = ExecutionContext::for_tests(615).rng.stream_for(StreamDomain::Test, 0x51);
    let (mut t, mut y, mut w) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..1200 {
        let modifier = standard_normal(&mut rng);
        let treatment = f64::from(rng.next_f64() < 0.5);
        t.push(treatment);
        w.push(modifier);
        y.push(2.0 * treatment + 0.3 * modifier + 0.5 * standard_normal(&mut rng));
    }
    TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice()), ("w", w.as_slice())])
        .unwrap()
}

fn conditional_graph() -> Dag {
    let mut graph = Dag::with_variables(3);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    graph
}

fn conditional_query(functional: OutcomeFunctional) -> ConditionalEffectQuery {
    ConditionalEffectQuery::try_new(
        AverageEffectQuery::binary_ate(v(0), v(1))
            .with_effect_modifiers([v(2)])
            .with_outcome_functional(functional),
    )
    .unwrap()
}

fn run_conditional(
    data: &TabularData,
    query: &ConditionalEffectQuery,
    ctx: &ExecutionContext,
) -> StudyResult {
    Study::tabular(data.clone())
        .graph(conditional_graph())
        .query(CausalQuery::ConditionalEffect(query.clone()))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(ctx)
        .unwrap()
}

#[test]
fn conditional_mean_effect_is_the_linear_adjustment_fit() {
    let data = conditional_data();
    let ctx = ExecutionContext::for_tests(41);
    let query = conditional_query(OutcomeFunctional::Mean);
    let compiled = run_conditional(&data, &query, &ctx);
    let published = compiled.estimate().as_effect().expect("an effect estimate");
    let estimator = ConditionalLinearAdjustment::new().with_fold_seed(ctx.rng.master_seed());

    let direct = estimator.estimate(&data, &compiled.estimand, &query, &ctx).unwrap();
    assert_effect("estimate", &direct, published);

    let (with_means, arms) =
        estimator.estimate_with_means(&data, &compiled.estimand, &query).unwrap();
    assert_effect("estimate_with_means", &with_means, published);

    let (ate, ate_arms, _) =
        estimator.estimate_ate(&data, &compiled.estimand, &query.inner).unwrap();
    assert_effect("estimate_ate", &ate, published);
    assert_eq!(arms.map(bits), ate_arms.map(bits), "the arm means of both entries");
}

#[test]
fn conditional_exceedance_effect_is_the_seeded_arm_score_fit() {
    let data = conditional_data();
    let ctx = ExecutionContext::for_tests(42);
    let threshold = 1.0;
    let query = conditional_query(OutcomeFunctional::exceedance(threshold));
    let compiled = run_conditional(&data, &query, &ctx);
    let published = compiled.estimate().as_effect().expect("an effect estimate");

    // The route reads each threshold as the mean of the exceedance indicator `1{Y > c}`.
    let outcome = data.float64_values(v(1)).unwrap();
    let indicator: Arc<[f64]> = outcome.iter().map(|&y| f64::from(y > threshold)).collect();
    let indicator_data = data.with_replaced_float(v(1), indicator).unwrap();
    let mean_query = conditional_query(OutcomeFunctional::Mean);
    let (direct, scores) = ConditionalLinearAdjustment::new()
        .with_fold_seed(ctx.rng.master_seed())
        .estimate_with_arm_scores(&indicator_data, &compiled.estimand, &mean_query)
        .unwrap();
    assert_effect("estimate_with_arm_scores", &direct, published);
    let cdf = published.joint.exceedance_cdf.clone().expect("the arm CDFs at the threshold");
    assert_eq!(
        cdf.iter().copied().map(bits).collect::<Vec<_>>(),
        [1.0 - scores.means[0], 1.0 - scores.means[1]].map(bits).to_vec(),
        "the published arm CDFs are one minus the arm means of the indicator"
    );
}

// --- temporal mediation --------------------------------------------------------------------

const N: usize = 241;

fn mediation_query() -> MediationQuery {
    MediationQuery::binary(v(0), v(2), [v(1)], MediationContrast::Mediated)
        .with_horizons(vec![1])
        .unwrap()
}

/// `M_t = 0.8 T_{t-1} + 0.6 Z_{t-1} + noise`, `Y_t = 1 + 0.25 T_{t-1} + 0.55 M_t + 5 Z_{t-1}`:
/// the horizon-1 mediation adjusts for `Z@-1`.
fn confounded_series() -> (TimeSeriesData, TemporalDag) {
    let z: Vec<f64> = (0..N).map(|i| if i % 4 == 0 || i % 4 == 1 { 1.0 } else { -1.0 }).collect();
    let t: Vec<f64> =
        (0..N).map(|i| z[i] + if i % 4 == 0 || i % 4 == 2 { 1.0 } else { -1.0 }).collect();
    let mut m = vec![0.0; N];
    let mut y = vec![1.0; N];
    for i in 1..N {
        let noise = (i % 5) as f64 - 2.0;
        m[i] = 0.8 * t[i - 1] + 0.6 * z[i - 1] + 0.15 * noise;
        y[i] = 1.0 + 0.25 * t[i - 1] + 0.55 * m[i] + 5.0 * z[i - 1];
    }
    let data = TimeSeriesData::from_f64_columns(
        [("t", t.as_slice()), ("m", m.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())],
        1,
    )
    .unwrap();
    let mut graph = TemporalDag::empty();
    let t0 = ensure_lagged(&mut graph, v(0), Lag::CONTEMPORANEOUS).unwrap();
    let t1 = ensure_lagged(&mut graph, v(0), Lag::from_raw(1)).unwrap();
    let m0 = ensure_lagged(&mut graph, v(1), Lag::CONTEMPORANEOUS).unwrap();
    let y0 = ensure_lagged(&mut graph, v(2), Lag::CONTEMPORANEOUS).unwrap();
    let z0 = ensure_lagged(&mut graph, v(3), Lag::CONTEMPORANEOUS).unwrap();
    let z1 = ensure_lagged(&mut graph, v(3), Lag::from_raw(1)).unwrap();
    for (from, to) in [(z0, t0), (z1, y0), (z1, m0), (t1, y0), (t1, m0), (m0, y0)] {
        graph.insert_directed(from, to).unwrap();
    }
    (data, graph)
}

/// `M_t = 0.7 T_{t-1} + noise`, `Y_t = 0.3 T_{t-1} + 0.5 M_t + noise`, with no confounder, so the
/// horizon-1 mediation adjusts for nothing. `gaps` marks treatment times as missing.
fn unconfounded_series(gaps: &[usize]) -> (TimeSeriesData, TemporalDag) {
    let t: Vec<f64> =
        (0..N).map(|i| (0.37 * i as f64).sin() + 0.5 * (1.3 * i as f64).cos()).collect();
    let mut m = vec![0.0; N];
    let mut y = vec![0.0; N];
    for i in 1..N {
        m[i] = 0.7 * t[i - 1] + 0.2 * (2.9 * i as f64).sin();
        y[i] = 0.3 * t[i - 1] + 0.5 * m[i] + 0.1 * (4.1 * i as f64).cos();
    }
    let observed: Vec<f64> =
        t.iter().enumerate().map(|(i, x)| if gaps.contains(&i) { f64::NAN } else { *x }).collect();
    let data = TimeSeriesData::from_f64_columns(
        [("t", observed.as_slice()), ("m", m.as_slice()), ("y", y.as_slice())],
        1,
    )
    .unwrap();
    let mut graph = TemporalDag::empty();
    let t1 = ensure_lagged(&mut graph, v(0), Lag::from_raw(1)).unwrap();
    let m0 = ensure_lagged(&mut graph, v(1), Lag::CONTEMPORANEOUS).unwrap();
    let y0 = ensure_lagged(&mut graph, v(2), Lag::CONTEMPORANEOUS).unwrap();
    for (from, to) in [(t1, m0), (t1, y0), (m0, y0)] {
        graph.insert_directed(from, to).unwrap();
    }
    (data, graph)
}

fn run_mediation(data: &TimeSeriesData, graph: &TemporalDag, gaps: bool) -> StudyResult {
    let builder = Study::series(data.clone())
        .graph(graph.clone())
        .query(CausalQuery::Mediation(mediation_query()))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0);
    let builder = if gaps {
        builder.missing(MissingDeclaration::new(MissingPolicy::CompleteCase))
    } else {
        builder
    };
    builder.build().unwrap().run(&ExecutionContext::for_tests(7)).unwrap()
}

/// The estimator the compiled DAG route builds.
fn route_estimator(gaps: bool) -> TemporalMediationEstimator {
    TemporalMediationEstimator::new()
        .with_allow_natural_controlled_alias(true)
        .with_allow_gaps(gaps)
}

fn assert_mediation(label: &str, direct: &TemporalMediationEstimate, compiled: &StudyResult) {
    let published = compiled.mediation().expect("a single-horizon mediation estimate");
    assert_effect(label, &direct.effect, &published.effect);
    for (part, a, b) in [
        ("total", direct.total, published.total),
        ("direct", direct.direct, published.direct),
        ("mediated", direct.mediated, published.mediated),
    ] {
        assert_eq!(a.map(bits), b.map(bits), "{label}: {part} {a:?} vs {b:?}");
    }
}

#[test]
fn adjusted_mediation_is_the_block_bootstrap_and_the_adjusted_fit() {
    let (data, graph) = confounded_series();
    let compiled = run_mediation(&data, &graph, false);
    let ctx = ExecutionContext::for_tests(7);
    let query = mediation_query();
    // The horizon-1 adjustment set is Z@-1, one lag before the outcome.
    let adjustment = [LaggedColumn { variable: v(3), lag: Lag::from_raw(1) }];
    let estimator = route_estimator(false);

    let (boot, block) = estimator
        .estimate_with_block_bootstrap(
            &data,
            &compiled.estimand,
            &query,
            &adjustment,
            BlockBootstrap { replicates: 0, stream_base: 0 },
            &ctx,
        )
        .unwrap();
    assert_eq!(block.replicates_attempted, 0, "no replicate is drawn without a request");
    assert_mediation("estimate_with_block_bootstrap", &boot, &compiled);

    let adjusted = estimator
        .estimate_with_adjustment(&data, &compiled.estimand, &query, &adjustment, &[], &ctx)
        .unwrap();
    assert_mediation("estimate_with_adjustment", &adjusted, &compiled);

    let shared =
        estimator.prepare_shared(&data, &compiled.estimand, &query, &adjustment, &ctx).unwrap();
    assert_mediation("prepare_shared", shared.estimate(), &compiled);
}

#[test]
fn unadjusted_mediation_is_the_estimate_the_extras_fit_and_the_effect_surface() {
    let (data, graph) = unconfounded_series(&[]);
    let compiled = run_mediation(&data, &graph, false);
    let ctx = ExecutionContext::for_tests(7);
    let query = mediation_query();
    let estimator = route_estimator(false);

    let estimate = estimator.estimate(&data, &compiled.estimand, &query, &ctx).unwrap();
    assert_mediation("estimate", &estimate, &compiled);
    let extras =
        estimator.estimate_with_extras(&data, &compiled.estimand, &query, &[], &ctx).unwrap();
    assert_mediation("estimate_with_extras", &extras, &compiled);

    let surface = estimator.effect_surface(&data, &compiled.estimand, &query, &ctx).unwrap();
    let published = compiled.mediation().expect("a single-horizon mediation estimate");
    assert_eq!(Some(bits(surface.total)), published.total.map(bits), "surface total");
    assert_eq!(Some(bits(surface.direct)), published.direct.map(bits), "surface direct");
    assert_eq!(Some(bits(surface.mediated)), published.mediated.map(bits), "surface mediated");
}

#[test]
fn gapped_mediation_reads_the_window_selection_of_the_kernel() {
    let gaps = [60, 61, 150];
    let (data, graph) = unconfounded_series(&gaps);
    let compiled = run_mediation(&data, &graph, true);
    let ctx = ExecutionContext::for_tests(7);
    let query = mediation_query();

    let selection = route_estimator(false)
        .window_selection(&data, &compiled.estimand, &query, &[], &ctx)
        .unwrap();
    let shared =
        route_estimator(true).prepare_shared(&data, &compiled.estimand, &query, &[], &ctx).unwrap();
    assert_eq!(shared.window_selection().complete, selection.complete, "prepared windows");
    assert!(selection.complete.iter().any(|intact| !intact), "the gaps drop windows");

    let report = compiled.missing_data().as_ref().expect("the declared complete-case report");
    assert_eq!(report.accounting.rows_total, selection.complete.len(), "windows on the axis");
    assert_eq!(
        report.accounting.dropped_rows.to_vec(),
        selection.dropped_times().into_iter().map(|time| time as u64).collect::<Vec<_>>(),
        "dropped windows"
    );

    let gapped = route_estimator(true)
        .estimate_with_adjustment(&data, &compiled.estimand, &query, &[], &[], &ctx)
        .unwrap();
    assert_mediation("estimate_with_adjustment on intact windows", &gapped, &compiled);
}

// --- partial correlation --------------------------------------------------------------------

fn ci_columns() -> [Vec<f64>; 3] {
    let mut rng = ExecutionContext::for_tests(913).rng.stream_for(StreamDomain::Test, 0x77);
    let (mut x, mut y, mut z) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..300 {
        let common = standard_normal(&mut rng);
        z.push(common);
        x.push(0.8 * common + standard_normal(&mut rng));
        y.push(0.5 * common + 0.3 * x.last().copied().unwrap_or(0.0) + standard_normal(&mut rng));
    }
    [x, y, z]
}

fn assert_ci(label: &str, direct: &CiResult, routed: &CiResult) {
    assert_eq!(bits(direct.statistic), bits(routed.statistic), "{label}: statistic");
    assert_eq!(bits(direct.p_value), bits(routed.p_value), "{label}: p-value");
    assert_eq!(bits(direct.df), bits(routed.df), "{label}: df");
    assert_eq!(
        direct.ci.map(|(lo, hi)| (bits(lo), bits(hi))),
        routed.ci.map(|(lo, hi)| (bits(lo), bits(hi))),
        "{label}: interval"
    );
}

#[test]
fn a_one_column_block_test_is_the_partial_correlation_test_one() {
    let [x, y, z] = ci_columns();
    let columns: [&[f64]; 3] = [&x, &y, &z];
    let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 1 }];
    for significance in [
        SignificanceMethod::Analytic,
        SignificanceMethod::BlockShuffle { replicates: 99, block_size: 5 },
    ] {
        let ctx = ExecutionContext::for_tests(29);
        let direct = PartialCorrelation::new()
            .test_one(&columns, &[2], significance, &mut CiWorkspace::default(), &ctx)
            .unwrap();
        // The pairwise block test, resolved by the name a compiled discovery route declares,
        // answers a one-column block through the multivariate block test's scalar path.
        let request = CiBatchRequest {
            columns: &columns,
            queries: &queries,
            z_flat: &[2],
            significance,
            confidence: ConfidenceMethod::default(),
        };
        let test: Arc<dyn ConditionalIndependence + Send + Sync> =
            ci_from_name("pairwise_mv").unwrap();
        let routed =
            test.test_batch_adhoc(&request, &mut CiWorkspace::default(), &ctx).unwrap().results;
        assert_eq!(routed.len(), 1, "one query, one result");
        assert_ci(&format!("pairwise_mv {significance:?}"), &direct, &routed[0]);
    }
}
