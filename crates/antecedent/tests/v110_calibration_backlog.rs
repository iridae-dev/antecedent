//! Coverage tests for constructions whose interval is the one the facade
//! reports: agreeing TemporalCpdag mediation, Bayesian ADMG intervention and
//! curve, and the frequentist joint influence-function band of an agreeing Pag.
//!
//! Ignored tests run via `scripts/gate_calibration.sh`. The non-ignored tests
//! check one replicate's match key and do not emit a coverage record.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::doc_markdown)]
#![allow(
    clippy::float_cmp,
    reason = "test scaffolding compares exact constants and indexes with small literals"
)]

mod common;

use std::sync::Arc;

use antecedent::{AcceptedGraph, BayesianConfig, InferenceMode, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, Intervention,
    Lag, MediationContrast, MediationQuery, ResponseFunctional, ResponseQuery, ResponseUncertainty,
    TemporalPolicy, TemporalResponseSpec, Value, VariableId,
};
use antecedent_data::{TableView, TabularData, TimeSeriesData};
use antecedent_graph::{
    Admg, DenseNodeId, Endpoint, MarkedEdge, MiddleMark, Pag, TemporalCpdag, TieredBackground,
    WithinTier,
};
use common::calibration::{
    gaussian, grid_n, map_replicates, n_sim, stream_seed, CoverageTally, RecordKey, SampleGrid,
};
use common::calibration_bind::{bind_all, constructions};
use common::fixtures::{self, mediation_cpdag_two, mediation_series};
use common::reported::{
    gate, posterior_pair, record_pair, response_band, response_scalar, skip_pair, REPORTED_LEVEL,
};

const MEDIATION_N: usize = 160;
const FRONTDOOR_N: usize = 1000;
const P_Y1_DO_T1: f64 = 0.57;

fn keyed(test: &'static str, dgp: &'static str, level: f64) -> CoverageTally {
    CoverageTally::for_record(RecordKey { test, dgp, interval: "posterior_quantile" }, level)
}

fn keyed_pair(test: &'static str, dgp: &'static str) -> [CoverageTally; 2] {
    [keyed(test, dgp, REPORTED_LEVEL), keyed(test, dgp, common::reported::GATE_LEVEL)]
}

fn bayes() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::laplace())
}

fn mediated_query() -> CausalQuery {
    CausalQuery::Mediation(
        MediationQuery::binary(
            VariableId::from_raw(0),
            VariableId::from_raw(2),
            [VariableId::from_raw(1)],
            MediationContrast::Mediated,
        )
        .with_horizons(vec![1])
        .unwrap(),
    )
}

fn intervention_query() -> CausalQuery {
    CausalQuery::Response(ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(2),
        interventions: Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(1.0))]),
    }))
}

fn scored_column(result: &StudyResult) -> Option<usize> {
    let posterior = result.posterior.as_ref()?;
    Some(posterior.effect_column().unwrap_or(0))
}

fn posterior_intervals(result: &StudyResult) -> [Option<(f64, f64)>; 2] {
    scored_column(result).map_or([None, None], |column| posterior_pair(result, column))
}

fn assert_posterior_quantile(study: &Study, result: &StudyResult) {
    let contract = study.inspect().expect("inspect");
    let reported = constructions(&contract, result);
    let methods: Vec<&str> =
        reported.iter().map(|(construction, _)| construction.interval_method.as_str()).collect();
    assert!(
        methods.contains(&"posterior_quantile"),
        "the execution reported {methods:?}, not posterior_quantile"
    );
}

fn uniform(seed: u64) -> impl FnMut() -> f64 {
    common::calibration::uniform(seed, 0xF0_0D00_0F0D)
}

fn frontdoor_data(n: usize, seed: u64) -> TabularData {
    let mut u01 = uniform(seed);
    let (mut t, mut m, mut y) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let u = f64::from(u01() < 0.5);
        t[i] = f64::from(u01() < 0.3 + 0.4 * u);
        m[i] = f64::from(u01() < 0.2 + 0.6 * t[i]);
        y[i] = f64::from(u01() < 0.1 + 0.4 * m[i] + 0.3 * u);
    }
    TabularData::from_f64_columns([("t", t.as_slice()), ("m", m.as_slice()), ("y", y.as_slice())])
        .unwrap()
}

fn frontdoor_admg() -> Admg {
    let mut graph = Admg::with_variables(3);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    graph.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    graph
}

fn run_mediation(
    kappa: f64,
    n: usize,
    accepted: bool,
    suite: RefuteSuite,
    seed: u64,
) -> Option<(Study, StudyResult)> {
    let data = mediation_series(n, kappa, seed);
    let builder = Study::series(data);
    let builder = if accepted {
        builder.graph(AcceptedGraph::temporal_cpdag(mediation_cpdag_two()).ok()?)
    } else {
        builder.graph(mediation_cpdag_two())
    };
    let study = builder.query(mediated_query()).inference(bayes()).refute(suite).build().ok()?;
    let result = study.run(&ExecutionContext::for_tests(seed)).ok()?;
    Some((study, result))
}

fn run_intervention(
    n: usize,
    accepted: bool,
    suite: RefuteSuite,
    seed: u64,
) -> Option<(Study, StudyResult)> {
    let data = frontdoor_data(n, seed);
    let builder = Study::tabular(data);
    let builder = if accepted {
        builder.graph(AcceptedGraph::from(frontdoor_admg()))
    } else {
        builder.graph(frontdoor_admg())
    };
    let study =
        builder.query(intervention_query()).inference(bayes()).refute(suite).build().ok()?;
    let result = study.run(&ExecutionContext::for_tests(seed)).ok()?;
    Some((study, result))
}

fn same_interval<R>(run: R, intervals: [Option<(f64, f64)>; 2])
where
    R: Fn(bool, RefuteSuite) -> Option<(Study, StudyResult)>,
{
    for (accepted, suite) in
        [(true, RefuteSuite::None), (false, RefuteSuite::Cheap), (false, RefuteSuite::Full)]
    {
        let (_, other) = run(accepted, suite).expect("licensed coordinate");
        assert_eq!(posterior_intervals(&other), intervals, "accepted={accepted} {suite:?}");
    }
}

fn mediation_coverage(test: &'static str, kappa: f64, seed_base: u64) {
    let mut tallies = keyed_pair(test, "mediation_series");
    let runs = map_replicates(n_sim(), |rep| {
        let seed = seed_base + rep;
        let (study, result) =
            run_mediation(kappa, grid_n(MEDIATION_N), false, RefuteSuite::None, seed)?;
        let intervals = posterior_intervals(&result);
        if rep == 0 {
            assert_eq!(
                result.logical_plan.estimator.as_deref(),
                Some("temporal.mediation.bayesian")
            );
            assert!(intervals[0].is_some(), "agreeing completions must publish the posterior");
            same_interval(
                |accepted, suite| run_mediation(kappa, grid_n(MEDIATION_N), accepted, suite, seed),
                intervals,
            );
        }
        Some((study, result, intervals))
    });
    for scored in &runs {
        let Some((study, result, intervals)) = scored else {
            skip_pair(&mut tallies);
            continue;
        };
        let [first, second] = &mut tallies;
        bind_all(&mut [first, second], study, result);
        record_pair(&mut tallies, *intervals, fixtures::mediation_truth());
    }
    gate(&tallies, &[None, None]);
}

fn publishes_mediation(kappa: f64, seed: u64) {
    let n = grid_n(MEDIATION_N) / 2;
    let (study, result) =
        run_mediation(kappa, n, false, RefuteSuite::None, seed).expect("mediation runs");
    assert_posterior_quantile(&study, &result);
    let intervals = posterior_intervals(&result);
    assert!(intervals[0].is_some());
    same_interval(|accepted, suite| run_mediation(kappa, n, accepted, suite, seed), intervals);
}

#[test]
fn temporal_cpdag_mediation_bayesian_confounded_publishes_posterior_quantile() {
    publishes_mediation(fixtures::MED_KAPPA, 0x110_0411);
}

#[test]
fn temporal_cpdag_mediation_bayesian_unconfounded_publishes_posterior_quantile() {
    publishes_mediation(0.0, 0x110_0412);
}

#[test]
fn intervention_response_admg_bayesian_publishes_posterior_quantile() {
    let n = 400;
    let seed = 0x110_0413;
    let (study, result) =
        run_intervention(n, false, RefuteSuite::None, seed).expect("intervention runs");
    assert_eq!(result.logical_plan.estimator.as_deref(), Some("functional.effect"));
    assert_posterior_quantile(&study, &result);
    let intervals = posterior_intervals(&result);
    assert!(intervals[0].is_some());
    same_interval(|accepted, suite| run_intervention(n, accepted, suite, seed), intervals);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn temporal_cpdag_mediation_bayesian_confounded_nominal_coverage() {
    mediation_coverage(
        "temporal_cpdag_mediation_bayesian_confounded_nominal_coverage",
        fixtures::MED_KAPPA,
        0x110_0411,
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn temporal_cpdag_mediation_bayesian_unconfounded_nominal_coverage() {
    mediation_coverage(
        "temporal_cpdag_mediation_bayesian_unconfounded_nominal_coverage",
        0.0,
        0x110_0412 << 20,
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn intervention_response_admg_bayesian_nominal_coverage() {
    let mut tallies =
        keyed_pair("intervention_response_admg_bayesian_nominal_coverage", "frontdoor_data");
    let runs = map_replicates(n_sim(), |rep| {
        let seed = stream_seed(0x110_0413, rep);
        let (study, result) =
            run_intervention(SampleGrid::HEAVY.n(FRONTDOOR_N), false, RefuteSuite::None, seed)?;
        let intervals = posterior_intervals(&result);
        if rep == 0 {
            assert_eq!(result.logical_plan.estimator.as_deref(), Some("functional.effect"));
            assert!(intervals[0].is_some(), "the intervention must publish the posterior");
            same_interval(
                |accepted, suite| {
                    run_intervention(SampleGrid::HEAVY.n(FRONTDOOR_N), accepted, suite, seed)
                },
                intervals,
            );
        }
        Some((study, result, intervals))
    });
    for scored in &runs {
        let Some((study, result, intervals)) = scored else {
            skip_pair(&mut tallies);
            continue;
        };
        let [first, second] = &mut tallies;
        bind_all(&mut [first, second], study, result);
        record_pair(&mut tallies, *intervals, P_Y1_DO_T1);
    }
    gate(&tallies, &[None, None]);
}

const PAG_GRID: [f64; 5] = [-1.0, -0.5, 0.0, 0.5, 1.0];
const ADMG_CURVE: [f64; 2] = [0.0, 1.0];
const ADMG_CURVE_TRUTH: [f64; 2] = [0.33, 0.57];

fn d(i: u32) -> DenseNodeId {
    DenseNodeId::from_raw(i)
}

fn assert_interval_method(study: &Study, result: &StudyResult, method: &str) {
    let contract = study.inspect().expect("inspect");
    let reported = constructions(&contract, result);
    let methods: Vec<&str> =
        reported.iter().map(|(construction, _)| construction.interval_method.as_str()).collect();
    assert!(methods.contains(&method), "the execution reported {methods:?}, not {method}");
}

fn pag_data(n: usize, seed: u64) -> TabularData {
    let mut g = common::calibration::gaussian(seed);
    let (mut t, mut y, mut z, mut r) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        z[i] = g();
        r[i] = g();
        t[i] = 0.5 * z[i] + 0.4 * r[i] + g();
        y[i] = 1.0 + 2.0 * t[i] + 0.8 * z[i] + g();
    }
    TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
        ("r", r.as_slice()),
    ])
    .unwrap()
}

fn agreeing_pag() -> Pag {
    let mut pag = Pag::with_variables(4);
    for (a, b, at_a, at_b) in [
        (3, 0, Endpoint::Circle, Endpoint::Arrow),
        (2, 0, Endpoint::Circle, Endpoint::Arrow),
        (2, 1, Endpoint::Tail, Endpoint::Arrow),
        (0, 1, Endpoint::Tail, Endpoint::Arrow),
    ] {
        pag.insert_marked(MarkedEdge { a: d(a), b: d(b), at_a, at_b, middle: MiddleMark::Empty })
            .unwrap();
    }
    pag
}

fn pag_level_query(level: f64) -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(1),
        interventions: Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(level))]),
    })
}

fn pag_curve_query() -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(PAG_GRID.to_vec().into()),
        ),
    })
}

fn run_pag(
    query: ResponseQuery,
    n: usize,
    accepted: bool,
    seed: u64,
) -> Option<(Study, StudyResult)> {
    let data = pag_data(n, seed);
    let builder = Study::tabular(data);
    let builder = if accepted {
        builder.graph(AcceptedGraph::from(agreeing_pag()))
    } else {
        builder.graph(agreeing_pag())
    };
    let study = builder
        .query(CausalQuery::Response(query))
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .build()
        .ok()?;
    let result = study.run(&ExecutionContext::for_tests(seed)).ok()?;
    Some((study, result))
}

fn admg_curve_query() -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(2),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(ADMG_CURVE.to_vec().into()),
        ),
    })
}

fn run_admg_curve(n: usize, accepted: bool, seed: u64) -> Option<(Study, StudyResult)> {
    let data = frontdoor_data(n, seed);
    let builder = Study::tabular(data);
    let builder = if accepted {
        builder.graph(AcceptedGraph::from(frontdoor_admg()))
    } else {
        builder.graph(frontdoor_admg())
    };
    let study = builder
        .query(CausalQuery::Response(admg_curve_query()))
        .inference(bayes())
        .refute(RefuteSuite::None)
        .build()
        .ok()?;
    let result = study.run(&ExecutionContext::for_tests(seed)).ok()?;
    Some((study, result))
}

fn keyed_analytic(test: &'static str, dgp: &'static str, label: String) -> CoverageTally {
    CoverageTally::for_record(RecordKey { test, dgp, interval: "analytic_se" }, REPORTED_LEVEL)
        .labelled(label)
}

#[test]
fn intervention_response_pag_frequentist_publishes_analytic_se() {
    let seed = 0x110_0420;
    let (study, result) =
        run_pag(pag_level_query(1.0), 800, false, seed).expect("pag intervention runs");
    assert_eq!(result.logical_plan.estimator.as_deref(), Some("response.intervention_gcomp"));
    assert_interval_method(&study, &result, "analytic_se");
    let interval = response_scalar(&result);
    assert!(interval.is_some());
    let (_, accepted) = run_pag(pag_level_query(1.0), 800, true, seed).expect("accepted pag");
    assert_eq!(response_scalar(&accepted), interval);
}

#[test]
fn response_curve_pag_frequentist_publishes_analytic_se() {
    let seed = 0x110_0421;
    let (study, result) = run_pag(pag_curve_query(), 800, false, seed).expect("pag curve runs");
    assert_eq!(result.logical_plan.estimator.as_deref(), Some("response.kennedy_dr"));
    assert_interval_method(&study, &result, "analytic_se");
    let band = response_band(&result).expect("pointwise band");
    assert_eq!(band.0.len(), PAG_GRID.len());
    let (_, accepted) = run_pag(pag_curve_query(), 800, true, seed).expect("accepted pag");
    assert_eq!(response_band(&accepted), Some(band));
}

#[test]
fn response_curve_admg_bayesian_publishes_posterior_quantile() {
    let seed = 0x110_0422;
    let (study, result) = run_admg_curve(400, false, seed).expect("admg curve runs");
    assert_eq!(result.logical_plan.estimator.as_deref(), Some("functional.effect"));
    assert_interval_method(&study, &result, "posterior_quantile");
    let band = response_band(&result).expect("pointwise credible band");
    assert_eq!(band.0.len(), ADMG_CURVE.len());
    let (_, accepted) = run_admg_curve(400, true, seed).expect("accepted admg");
    assert_eq!(response_band(&accepted), Some(band));
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn intervention_response_pag_frequentist_nominal_coverage() {
    let mut tally = keyed_analytic(
        "intervention_response_pag_frequentist_nominal_coverage",
        "pag_data",
        "a=1".into(),
    );
    let runs = map_replicates(n_sim(), |rep| {
        let seed = stream_seed(0x110_0420, rep);
        let (study, result) = run_pag(pag_level_query(1.0), grid_n(500), false, seed)?;
        let interval = response_scalar(&result).map(|(lo, hi, _, _)| (lo, hi));
        if rep == 0 {
            assert!(interval.is_some(), "agreeing completions must publish the joint-IF interval");
            let (_, accepted) = run_pag(pag_level_query(1.0), grid_n(500), true, seed).unwrap();
            assert_eq!(response_scalar(&accepted).map(|(lo, hi, _, _)| (lo, hi)), interval);
        }
        Some((study, result, interval))
    });
    for scored in &runs {
        let Some((study, result, interval)) = scored else {
            tally.skip();
            continue;
        };
        bind_all(&mut [&mut tally], &study, result);
        tally.record(*interval, 3.0);
    }
    gate(&[tally], &[None]);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn response_curve_pag_frequentist_pointwise_nominal_coverage() {
    let mut tallies: Vec<CoverageTally> = PAG_GRID
        .iter()
        .map(|a| {
            keyed_analytic(
                "response_curve_pag_frequentist_pointwise_nominal_coverage",
                "pag_data",
                format!("a={a}"),
            )
        })
        .collect();
    let runs = map_replicates(n_sim(), |rep| {
        let seed = stream_seed(0x110_0421, rep);
        run_pag(pag_curve_query(), grid_n(500), false, seed)
    });
    for run in &runs {
        let bands = run.as_ref().and_then(|(_, result)| response_band(result));
        if let (Some((study, result)), Some(_)) = (run, &bands) {
            bind_all(&mut tallies.iter_mut().collect::<Vec<_>>(), study, result);
        }
        for (j, tally) in tallies.iter_mut().enumerate() {
            match &bands {
                Some((lower, upper, _)) => {
                    tally.record(Some((lower[j], upper[j])), 1.0 + 2.0 * PAG_GRID[j])
                }
                None => tally.skip(),
            }
        }
    }
    gate(&tallies, &[None; 5]);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn response_curve_admg_bayesian_pointwise_nominal_coverage() {
    let mut tallies: Vec<CoverageTally> = ADMG_CURVE
        .iter()
        .map(|a| {
            CoverageTally::for_record(
                RecordKey {
                    test: "response_curve_admg_bayesian_pointwise_nominal_coverage",
                    dgp: "frontdoor_data",
                    interval: "posterior_quantile",
                },
                REPORTED_LEVEL,
            )
            .labelled(format!("a={a}"))
        })
        .collect();
    let runs = map_replicates(n_sim(), |rep| {
        let seed = stream_seed(0x110_0422, rep);
        run_admg_curve(SampleGrid::HEAVY.n(FRONTDOOR_N), false, seed)
    });
    for run in &runs {
        let bands = run.as_ref().and_then(|(_, result)| response_band(result));
        if let (Some((study, result)), Some(_)) = (run, &bands) {
            bind_all(&mut tallies.iter_mut().collect::<Vec<_>>(), study, result);
        }
        for (j, tally) in tallies.iter_mut().enumerate() {
            match &bands {
                Some((lower, upper, _)) => {
                    tally.record(Some((lower[j], upper[j])), ADMG_CURVE_TRUTH[j])
                }
                None => tally.skip(),
            }
        }
    }
    gate(&tallies, &[None; 2]);
}

const UNKNOWN_TRUTH: [f64; 2] = [1.0, -1.0];

fn unknown_data(n: usize, seed: u64) -> TabularData {
    let mut noise = gaussian(seed);
    let (mut era, mut t, mut m, mut y) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        era[i] = noise();
        t[i] = 0.8 * era[i] + noise();
        m[i] = t[i] + 0.2 * era[i] + 0.5 * noise();
        y[i] = -t[i] + 2.0 * m[i] + 0.2 * era[i] + noise();
    }
    TabularData::from_f64_columns([
        ("era", era.as_slice()),
        ("t", t.as_slice()),
        ("m", m.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap()
}

fn run_unknown(n: usize, seed: u64) -> Option<(Study, StudyResult)> {
    let data = unknown_data(n, seed);
    let schema = data.schema().clone();
    let background = TieredBackground::from_named(
        &schema,
        &[vec!["era"], vec!["t", "m"], vec!["y"]],
        WithinTier::Unknown,
    )
    .ok()?;
    let query = AverageEffectQuery::binary_ate(schema.id_of("t").ok()?, schema.id_of("y").ok()?);
    let study = Study::tabular(data)
        .tiered_background(background)
        .ok()?
        .query(query)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .ok()?;
    let result = study.run(&ExecutionContext::for_tests(seed)).ok()?;
    Some((study, result))
}

/// Joint coverage as one interval of the summed band width: truth `0` lies
/// inside exactly when both scenario truths lie in their max-t bands.
fn joint_band_interval(result: &StudyResult) -> Option<(f64, f64)> {
    let bands = result.estimate.scenario_intervals.as_ref()?;
    if bands.len() != UNKNOWN_TRUTH.len() {
        return None;
    }
    let covered =
        bands.iter().zip(UNKNOWN_TRUTH).all(|(&(lo, hi), truth)| lo <= truth && truth <= hi);
    let width = bands.iter().map(|(lo, hi)| hi - lo).sum::<f64>();
    Some(if covered { (-width / 2.0, width / 2.0) } else { (1.0, 1.0 + width) })
}

#[test]
fn average_effect_unknown_publishes_simultaneous_band() {
    let (study, result) = run_unknown(400, 0x110_0430).expect("unknown orientation runs");
    assert_eq!(result.logical_plan.estimator.as_deref(), Some("linear.adjustment.ate"));
    assert_interval_method(&study, &result, "simultaneous_band");
    assert!(joint_band_interval(&result).is_some());
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn average_effect_unknown_joint_band_nominal_coverage() {
    let mut tally = CoverageTally::for_record(
        RecordKey {
            test: "average_effect_unknown_joint_band_nominal_coverage",
            dgp: "unknown_data",
            interval: "simultaneous_band",
        },
        REPORTED_LEVEL,
    );
    let runs = map_replicates(n_sim(), |rep| {
        let seed = 20_300 + rep;
        run_unknown(grid_n(400), seed)
    });
    for run in &runs {
        let Some((study, result)) = run else {
            tally.skip();
            continue;
        };
        bind_all(&mut [&mut tally], study, result);
        tally.record(joint_band_interval(result), 0.0);
    }
    tally.assert_boundary_at([Some(0.940), None, None]);
}

fn temporal_response(horizons: &[u32]) -> TemporalResponseSpec {
    TemporalResponseSpec::new(horizons.to_vec(), TemporalPolicy::pulse(-1), None).unwrap()
}

fn agreeing_temporal_series(n: usize, seed: u64) -> TimeSeriesData {
    const BURN: usize = 10;
    let len = n + BURN;
    let mut noise = gaussian(seed);
    let (mut t, mut y, mut z, mut w) =
        (vec![0.0; len], vec![0.0; len], vec![0.0; len], vec![0.0; len]);
    for s in 0..len {
        z[s] = noise();
        w[s] = 0.4 * z[s] + noise();
        t[s] = 0.5 * z[s] + noise();
        let (t_lag, z_lag) = if s == 0 { (0.0, 0.0) } else { (t[s - 1], z[s - 1]) };
        y[s] = 1.0 + 2.0 * t_lag + 0.8 * z_lag + noise();
    }
    TimeSeriesData::from_f64_columns(
        [("t", &t[BURN..]), ("y", &y[BURN..]), ("z", &z[BURN..]), ("w", &w[BURN..])],
        1,
    )
    .unwrap()
}

/// `Z@-1 -> T@-1`, `Z@-1 -> Y`, `T@-1 -> Y`, `W@-1 — Z@-1`. Both orientations
/// of the ambiguous edge adjust `{Z@-1}`, so they share one band.
fn agreeing_temporal_cpdag() -> TemporalCpdag {
    let mut graph = TemporalCpdag::empty();
    let t1 = graph.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = graph.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let z1 = graph.add_lagged(VariableId::from_raw(2), Lag::from_raw(1)).unwrap();
    let w1 = graph.add_lagged(VariableId::from_raw(3), Lag::from_raw(1)).unwrap();
    graph.insert_directed(z1, t1).unwrap();
    graph.insert_directed(z1, y0).unwrap();
    graph.insert_directed(t1, y0).unwrap();
    graph.insert_undirected(w1, z1).unwrap();
    graph
}

fn disagreeing_temporal_cpdag() -> TemporalCpdag {
    let mut graph = TemporalCpdag::empty();
    let t1 = graph.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = graph.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let z1 = graph.add_lagged(VariableId::from_raw(2), Lag::from_raw(1)).unwrap();
    graph.insert_directed(z1, y0).unwrap();
    graph.insert_directed(t1, y0).unwrap();
    graph.insert_undirected(z1, t1).unwrap();
    graph
}

fn temporal_intervention() -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(1),
        interventions: Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(1.0))]),
    })
    .with_temporal(temporal_response(&[1]))
}

fn temporal_curve() -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from([0.0, 1.0])),
        ),
    })
    .with_temporal(temporal_response(&[1]))
}

fn run_temporal_class(
    data: TimeSeriesData,
    graph: impl antecedent::IntoGraphInput,
    query: ResponseQuery,
    inference: InferenceMode,
    replicates: u32,
    seed: u64,
) -> (Study, StudyResult) {
    let study = Study::series(data)
        .graph(graph)
        .query(CausalQuery::Response(query))
        .inference(inference)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(replicates)
        .build()
        .unwrap();
    let result = study.run(&ExecutionContext::for_tests(seed)).unwrap();
    (study, result)
}

fn temporal_bayes() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(400).prior_scale(8.0))
}

#[test]
fn agreeing_temporal_cpdag_publishes_shared_band() {
    let data = agreeing_temporal_series(80, 0x110_0440);
    let (study, result) = run_temporal_class(
        data,
        agreeing_temporal_cpdag(),
        temporal_intervention(),
        InferenceMode::Frequentist,
        19,
        0x110_0440,
    );
    assert_interval_method(&study, &result, "circular_block_se");
    let response = result.response.as_ref().expect("class response");
    assert!(matches!(response.uncertainty, ResponseUncertainty::PointwiseBand { .. }));
    let atoms: Vec<_> = result
        .structural_response
        .as_ref()
        .expect("atoms")
        .atoms
        .iter()
        .filter(|atom| atom.value.is_some())
        .collect();
    assert!(atoms.len() >= 2, "both orientations evaluate");
    assert!(
        atoms.iter().all(|atom| atom.response.as_ref().is_some_and(|response| {
            response.uncertainty == atoms[0].response.as_ref().unwrap().uncertainty
        })),
        "completions share one band"
    );

    let (study, result) = run_temporal_class(
        agreeing_temporal_series(80, 0x110_0441),
        agreeing_temporal_cpdag(),
        temporal_intervention(),
        temporal_bayes(),
        0,
        0x110_0441,
    );
    assert_interval_method(&study, &result, "posterior_quantile");

    let (study, result) = run_temporal_class(
        agreeing_temporal_series(80, 0x110_0442),
        agreeing_temporal_cpdag(),
        temporal_curve(),
        InferenceMode::Frequentist,
        19,
        0x110_0442,
    );
    assert_interval_method(&study, &result, "circular_block_se");
    match response_band(&result) {
        Some((lower, _, _)) => assert_eq!(lower.len(), 2),
        None => panic!("the shared curve band is pointwise"),
    }

    let (_, result) = run_temporal_class(
        agreeing_temporal_series(80, 0x110_0443),
        disagreeing_temporal_cpdag(),
        temporal_intervention(),
        InferenceMode::Frequentist,
        19,
        0x110_0443,
    );
    assert!(matches!(
        result.response.as_ref().map(|response| &response.uncertainty),
        Some(ResponseUncertainty::None)
    ));
}

fn covariate_mean(data: &TimeSeriesData) -> f64 {
    let antecedent_data::ColumnView::Float64(column) =
        data.column(VariableId::from_raw(2)).expect("z")
    else {
        panic!("z is float");
    };
    let z = &column.values[..column.values.len() - 1];
    z.iter().sum::<f64>() / z.len() as f64
}

fn record_temporal_band(
    tally: &mut CoverageTally,
    band: &Option<(Vec<f64>, Vec<f64>, f64)>,
    index: usize,
    truth: f64,
) {
    match band {
        Some((lower, upper, _)) => tally.record(Some((lower[index], upper[index])), truth),
        None => tally.skip(),
    }
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn temporal_cpdag_intervention_frequentist_nominal_coverage() {
    let mut tally = CoverageTally::for_record(
        RecordKey {
            test: "temporal_cpdag_intervention_frequentist_nominal_coverage",
            dgp: "agreeing_temporal_series",
            interval: "circular_block_se",
        },
        REPORTED_LEVEL,
    );
    let runs = map_replicates(n_sim(), |rep| {
        let seed = stream_seed(0x110_0450, rep);
        run_temporal_class(
            agreeing_temporal_series(grid_n(160), seed),
            agreeing_temporal_cpdag(),
            temporal_intervention(),
            InferenceMode::Frequentist,
            199,
            seed,
        )
    });
    for (study, result) in &runs {
        bind_all(&mut [&mut tally], study, result);
        record_temporal_band(&mut tally, &response_band(result), 0, 3.0);
    }
    gate(&[tally], &[None]);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn temporal_cpdag_intervention_bayesian_nominal_coverage() {
    let mut tally = CoverageTally::for_record(
        RecordKey {
            test: "temporal_cpdag_intervention_bayesian_nominal_coverage",
            dgp: "agreeing_temporal_series",
            interval: "posterior_quantile",
        },
        REPORTED_LEVEL,
    );
    let runs = map_replicates(n_sim(), |rep| {
        let seed = stream_seed(0x110_0451, rep);
        let data = agreeing_temporal_series(grid_n(160), seed);
        let truth = 3.0 + 0.8 * covariate_mean(&data);
        let (study, result) = run_temporal_class(
            data,
            agreeing_temporal_cpdag(),
            temporal_intervention(),
            temporal_bayes(),
            0,
            seed,
        );
        (study, result, truth)
    });
    for (study, result, truth) in &runs {
        bind_all(&mut [&mut tally], study, result);
        record_temporal_band(&mut tally, &response_band(result), 0, *truth);
    }
    gate(&[tally], &[None]);
}

const TEMPORAL_DOSES: [f64; 2] = [0.0, 1.0];

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn temporal_cpdag_curve_frequentist_pointwise_nominal_coverage() {
    let mut tallies: Vec<CoverageTally> = TEMPORAL_DOSES
        .iter()
        .map(|dose| {
            CoverageTally::for_record(
                RecordKey {
                    test: "temporal_cpdag_curve_frequentist_pointwise_nominal_coverage",
                    dgp: "agreeing_temporal_series",
                    interval: "circular_block_se",
                },
                REPORTED_LEVEL,
            )
            .labelled(format!("a={dose}"))
        })
        .collect();
    let runs = map_replicates(n_sim(), |rep| {
        let seed = stream_seed(0x110_0452, rep);
        run_temporal_class(
            agreeing_temporal_series(grid_n(160), seed),
            agreeing_temporal_cpdag(),
            temporal_curve(),
            InferenceMode::Frequentist,
            199,
            seed,
        )
    });
    for (study, result) in &runs {
        bind_all(&mut tallies.iter_mut().collect::<Vec<_>>(), study, result);
        let band = response_band(result);
        for (index, tally) in tallies.iter_mut().enumerate() {
            record_temporal_band(tally, &band, index, 1.0 + 2.0 * TEMPORAL_DOSES[index]);
        }
    }
    gate(&tallies, &[None, None]);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn temporal_cpdag_curve_bayesian_pointwise_nominal_coverage() {
    let mut tallies: Vec<CoverageTally> = TEMPORAL_DOSES
        .iter()
        .map(|dose| {
            CoverageTally::for_record(
                RecordKey {
                    test: "temporal_cpdag_curve_bayesian_pointwise_nominal_coverage",
                    dgp: "agreeing_temporal_series",
                    interval: "posterior_quantile",
                },
                REPORTED_LEVEL,
            )
            .labelled(format!("a={dose}"))
        })
        .collect();
    let runs = map_replicates(n_sim(), |rep| {
        let seed = stream_seed(0x110_0453, rep);
        let data = agreeing_temporal_series(grid_n(160), seed);
        let z_bar = covariate_mean(&data);
        let (study, result) = run_temporal_class(
            data,
            agreeing_temporal_cpdag(),
            temporal_curve(),
            temporal_bayes(),
            0,
            seed,
        );
        (study, result, z_bar)
    });
    for (study, result, z_bar) in &runs {
        bind_all(&mut tallies.iter_mut().collect::<Vec<_>>(), study, result);
        let band = response_band(result);
        for (index, tally) in tallies.iter_mut().enumerate() {
            record_temporal_band(
                tally,
                &band,
                index,
                1.0 + 2.0 * TEMPORAL_DOSES[index] + 0.8 * z_bar,
            );
        }
    }
    gate(&tallies, &[None, None]);
}
