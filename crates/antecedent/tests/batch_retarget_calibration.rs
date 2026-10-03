//! Coverage of the unpublished family-level max-t band of a batch retarget (record
//! `2.2E.E3.batch_retarget_covariance_contrasts`, closed route
//! `antecedent.PreparedBatch.simultaneous_interval`).
//!
//! Four retargeted AIPW claims share one row snapshot: two binary treatments and two
//! outcomes, all confounded by one covariate, with constant additive effects so that every
//! claim's truth is the same under every declared target weighting (2.0, 0.5, -1.0, 1.5).
//! The band is `point_j ± c · se_j` from the family's plug-in score covariance with `c`
//! the Monte-Carlo max-t critical value of its correlation. The scored event is the joint
//! one, "all four truths inside the band", recorded as the unit interval against 0.5
//! (covered) or 2.0 (missed), the same way the response-curve simultaneous band is scored.
//!
//! `batch_retarget_max_t_simultaneous_band` is registered in `scripts/gate_calibration.sh`
//! (`run_br`, measured over the sample-size grid) and is measured once at the release cut;
//! until then the published interval stays closed (`cell_not_licensed`) and no coverage
//! record exists. The evaluator is `antecedent::simultaneous_band_unpublished`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss)]

mod common;

use antecedent::{
    BatchRetargetRequest, BatchStudy, EstimatorId, PreparedBatch, RefuteSuite, RetargetClaim,
    SimultaneousBand, simultaneous_band_unpublished,
};
use antecedent_core::{AverageEffectQuery, ExecutionContext, StreamDomain, VariableId};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_kernels::standard_normal;
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, grid_n, map_replicates,
    n_sim,
};

const INTERVAL: &str = "simultaneous_band";
/// Monte-Carlo draws of the critical value.
const BAND_DRAWS: u32 = 10_000;
const T1: u32 = 0;
const T2: u32 = 1;
const Y1: u32 = 2;
const Y2: u32 = 3;
const Z: u32 = 4;
/// Truth of each claim, in query order: `(T1, Y1)`, `(T2, Y1)`, `(T1, Y2)`, `(T2, Y2)`.
const TRUTHS: [f64; 4] = [2.0, 0.5, -1.0, 1.5];

struct Fixture {
    data: TabularData,
    graph: Dag,
    z: Vec<f64>,
}

/// Two binary treatments and two outcomes, all confounded by one `z`.
fn fixture(n: usize, seed: u64) -> Fixture {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Test, 0xE3);
    let (mut t1, mut t2, mut y1, mut y2, mut z) =
        (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let zi = standard_normal(&mut rng);
        z[i] = zi;
        let p1 = 1.0 / (1.0 + (-(-0.2 + 0.8 * zi)).exp());
        let p2 = 1.0 / (1.0 + (-(0.1 - 0.6 * zi)).exp());
        t1[i] = f64::from(rng.next_f64() < p1);
        t2[i] = f64::from(rng.next_f64() < p2);
        y1[i] = 2.0 * t1[i] + 0.5 * t2[i] + zi + 0.3 * standard_normal(&mut rng);
        y2[i] = -t1[i] + 1.5 * t2[i] - 0.5 * zi + 0.3 * standard_normal(&mut rng);
    }
    let pairs = [("t1", t1), ("t2", t2), ("y1", y1), ("y2", y2), ("z", z.clone())];
    let borrowed: Vec<(&str, &[f64])> = pairs.iter().map(|(n, v)| (*n, v.as_slice())).collect();
    let data = TabularData::from_f64_columns(borrowed).unwrap();
    let mut graph = Dag::with_variables(5);
    for (from, to) in [(Z, T1), (Z, T2), (Z, Y1), (Z, Y2), (T1, Y1), (T1, Y2), (T2, Y1), (T2, Y2)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    Fixture { data, graph, z }
}

fn ate(treatment: u32, outcome: u32) -> AverageEffectQuery {
    AverageEffectQuery::binary_ate(VariableId::from_raw(treatment), VariableId::from_raw(outcome))
}

fn prepare(fx: &Fixture, ctx: &ExecutionContext) -> PreparedBatch {
    BatchStudy::new(fx.data.clone(), fx.graph.clone())
        .estimator(EstimatorId::Aipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .prepare(&[ate(T1, Y1), ate(T2, Y1), ate(T1, Y2), ate(T2, Y2)], ctx)
        .unwrap()
}

/// The four claims on the snapshot's rows: two unweighted and two reweighted by
/// `exp(± 0.4 z)`. Constant effects make the truth the same under every weighting.
fn request(fx: &Fixture, rows: &[u32]) -> BatchRetargetRequest {
    let tilt = |sign: f64| -> Vec<f64> {
        rows.iter().map(|&r| (sign * 0.4 * fx.z[r as usize]).exp()).collect()
    };
    let claim = |name: &str, query_index: usize, weights: Vec<f64>| RetargetClaim {
        name: name.into(),
        query_index,
        weights,
        depends_on: vec![VariableId::from_raw(Z)],
    };
    BatchRetargetRequest {
        claims: vec![
            claim("c0", 0, vec![1.0; rows.len()]),
            claim("c1", 1, tilt(1.0)),
            claim("c2", 2, tilt(-1.0)),
            claim("c3", 3, vec![1.0; rows.len()]),
        ],
        contrasts: Vec::new(),
        expected_snapshot: None,
    }
}

/// One replicate: simulate, prepare, retarget the four claims and form the unpublished band.
fn replicate(n: usize, data_seed: u64, band_seed: u64) -> Option<SimultaneousBand> {
    let fx = fixture(n, data_seed);
    let ctx = ExecutionContext::for_tests(band_seed);
    let prepared = prepare(&fx, &ctx);
    let scores = prepared.prepared_scores();
    let rows = scores.common_rows().ok()??;
    let report = prepared.retarget(&scores, &request(&fx, &rows), &ctx).ok()?;
    simultaneous_band_unpublished(&report, REPORTED_LEVEL, band_seed, BAND_DRAWS, &ctx).ok()
}

fn construction() -> Construction {
    Construction {
        query: "AverageEffect".into(),
        graph_class: "Dag".into(),
        structure: "fixed".into(),
        modality: "tabular".into(),
        inference: "Frequentist".into(),
        estimator: "aipw".into(),
        interval_method: INTERVAL.into(),
        se_kind: String::new(),
        dependence: "iid".into(),
        posterior: String::new(),
        functional: "all_observed.mean".into(),
        identification: "point".into(),
        reported_level: REPORTED_LEVEL,
    }
}

/// The band's structure on one replicate at a large sample: four members in claim order,
/// each centred on its point near the known truth, with `c` between the pointwise and the
/// independent four-claim critical value.
#[test]
fn the_unpublished_band_has_the_declared_members_and_a_max_t_critical_value() {
    let band = replicate(2_000, 90_001, 5).expect("a complete covariance-bearing family");
    assert_eq!(band.members.len(), TRUTHS.len());
    for (i, (member, truth)) in band.members.iter().zip(TRUTHS).enumerate() {
        assert_eq!(member.name, format!("c{i}"));
        assert!((member.value - truth).abs() < 0.5, "{}: {} vs {truth}", member.name, member.value);
        assert!(member.std_error > 0.0 && member.std_error.is_finite());
        assert!(
            (member.upper - member.value - band.critical_value * member.std_error).abs() < 1e-12
        );
        assert!(
            (member.value - member.lower - band.critical_value * member.std_error).abs() < 1e-12
        );
    }
    // 1.96 is the single-claim value and about 2.57 the independent four-claim value.
    assert!(band.critical_value > 1.96 && band.critical_value < 2.7, "{}", band.critical_value);
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn batch_retarget_max_t_simultaneous_band() {
    let n = grid_n(400);
    let mut tally = CoverageTally::for_record(
        RecordKey {
            test: "batch_retarget_max_t_simultaneous_band",
            dgp: "fixture",
            interval: INTERVAL,
        },
        REPORTED_LEVEL,
    );
    let bands = map_replicates(n_sim(), |rep| replicate(n, 41_000 + rep, rep));
    for band in bands {
        let Some(band) = band else {
            tally.skip();
            continue;
        };
        tally.bind(
            &construction(),
            ScopeFacts {
                row_count: n as u64,
                replicates_ok: None,
                posterior_draws: None,
                unidentified_mass: 0.0,
            },
        );
        let covered =
            band.members.iter().zip(TRUTHS).all(|(m, truth)| m.lower <= truth && truth <= m.upper);
        tally.record(Some((0.0, 1.0)), if covered { 0.5 } else { 2.0 });
    }
    tally.assert();
}
