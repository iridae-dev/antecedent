//! Repeated-sampling coverage for the Frequentist front-door distribution
//! interval on an explicit or accepted ADMG.

mod common;

use antecedent::{AcceptedGraph, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ExecutionContext, Intervention, InterventionalDistributionQuery, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_estimate::logit_probability_interval;
use antecedent_graph::{Admg, DenseNodeId};
use common::calibration::{
    CoverageTally, RecordKey, SampleGrid, map_replicates, n_sim, stream_seed,
};
use common::calibration_bind::bind_all;
use common::reported::{GATE_LEVEL, REPORTED_LEVEL, gate};

const TRUTH: f64 = 0.57;
const TEST: &str = "interventional_distribution_admg_frontdoor_frequentist_nominal_coverage";

fn data(n: usize, seed: u64) -> TabularData {
    let mut u01 = common::calibration::uniform(seed, 0xF0_0D00_0F0D);
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

fn graph() -> Admg {
    let mut g = Admg::with_variables(3);
    g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    g.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    g.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    g
}

fn query() -> CausalQuery {
    CausalQuery::Distribution(InterventionalDistributionQuery::new(
        VariableId::from_raw(2),
        [Intervention::set(VariableId::from_raw(0), Value::f64(1.0))],
    ))
}

fn run(data: &TabularData, accepted: bool, seed: u64) -> (Study, antecedent::StudyResult) {
    let builder = Study::tabular(data.clone());
    let builder =
        if accepted { builder.graph(AcceptedGraph::from(graph())) } else { builder.graph(graph()) };
    let study = builder
        .query(query())
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let result = study.run(&ExecutionContext::for_tests(seed)).unwrap();
    (study, result)
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn interventional_distribution_admg_frontdoor_frequentist_nominal_coverage() {
    let mut tallies = [REPORTED_LEVEL, GATE_LEVEL].map(|level| {
        CoverageTally::for_record(
            RecordKey { test: TEST, dgp: "frontdoor_data", interval: "bootstrap_se" },
            level,
        )
    });
    let runs = map_replicates(n_sim(), |rep| {
        let seed = stream_seed(0x110_0104, rep);
        let data = data(SampleGrid::HEAVY.n(1000), seed);
        let (study, result) = run(&data, false, seed);
        let distribution = result.distribution.as_ref().unwrap();
        let reported = distribution.mean_interval.and_then(|interval| interval.bounds());
        let gated =
            logit_probability_interval(distribution.mean, distribution.se_bootstrap, GATE_LEVEL)
                .bounds();
        if rep == 0 {
            assert_eq!(result.logical_plan.estimator.as_deref(), Some("functional.distribution"));
            assert_eq!(distribution.bootstrap_replicates_ok, Some(199));
            assert!(reported.is_some());
            let (_, accepted) = run(&data, true, seed);
            assert_eq!(
                accepted.distribution.as_ref().unwrap().mean_interval,
                distribution.mean_interval
            );
        }
        (study, result, [reported, gated])
    });
    for (study, result, intervals) in &runs {
        let [reported, gated] = &mut tallies;
        bind_all(&mut [reported, gated], study, result);
        for (tally, interval) in tallies.iter_mut().zip(intervals) {
            tally.record(*interval, TRUTH);
        }
    }
    gate(&tallies, &[None, None]);
}
