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
    CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, Intervention, MediationContrast,
    MediationQuery, ResponseFunctional, ResponseQuery, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Admg, DenseNodeId, Endpoint, MarkedEdge, MiddleMark, Pag};
use common::calibration::{
    CoverageTally, RecordKey, SampleGrid, grid_n, map_replicates, n_sim, stream_seed,
};
use common::calibration_bind::{bind_all, constructions};
use common::fixtures::{self, mediation_cpdag_two, mediation_series};
use common::reported::{
    REPORTED_LEVEL, gate, posterior_pair, record_pair, response_band, response_scalar, skip_pair,
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
