//! Repeated-sampling known-truth test of the batch-retarget max-t family band.
//!
//! This is a TEST with fixed seeds, not a coverage record: it simulates iid data from a known
//! SCM, builds a three-claim family on one AIPW plan, forms the band
//! `point_j ± c · se_j` with `BatchRetargetReport::simultaneous_interval` and asserts the
//! family-wise coverage over the replications is at least the nominal level minus a stated
//! Monte-Carlo tolerance of three binomial standard errors. It also asserts the max-t band is
//! strictly wider than every marginal Wald band. Calibrated coverage of the band is measured
//! separately by `batch_retarget_calibration.rs` (wired, run at the release cut).
//!
//! SCM: `Z ~ N(0, 1)`, `P(T = 1 | Z) = logistic(-0.2 + 0.8 Z)`,
//! `Y = Z + 2 T + 0.5 T Z + 0.5 e`, so the conditional effect is `2 + 0.5 Z`. Under the
//! exponential tilt `w(z) = exp(s z)` of a standard normal the retargeted covariate mean is
//! `E_w[Z] = s`, so the retargeted ATE is `2 + 0.5 s` exactly: 2.0 (uniform weights), 2.2
//! (`s = 0.4`) and 1.8 (`s = -0.4`).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    reason = "replicate counts are small literals converted for a coverage fraction"
)]

use antecedent::{
    BatchRetargetRequest, BatchStudy, EstimatorId, RefuteSuite, RetargetClaim, SimultaneousBand,
};
use antecedent_core::{AverageEffectQuery, ExecutionContext, StreamDomain, VariableId};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_kernels::standard_normal;

const T: u32 = 0;
const Y: u32 = 1;
const Z: u32 = 2;
const REPS: u64 = 300;
const N: usize = 600;
const LEVEL: f64 = 0.95;
/// Monte-Carlo draws of the critical value inside each replication.
const BAND_DRAWS: u32 = 10_000;
/// Two-sided 95% normal quantile (the marginal Wald multiplier).
const Z975: f64 = 1.959_963_984_540_054;
/// Retargeted ATEs under weights `exp(s z)`: `2 + 0.5 s`.
const TILTS: [f64; 3] = [0.0, 0.4, -0.4];
const TRUTHS: [f64; 3] = [2.0, 2.2, 1.8];

fn fixture(seed: u64) -> (TabularData, Dag, Vec<f64>) {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Test, 0xE3A);
    let (mut t, mut y, mut z) = (vec![0.0; N], vec![0.0; N], vec![0.0; N]);
    for i in 0..N {
        let zi = standard_normal(&mut rng);
        z[i] = zi;
        let p = 1.0 / (1.0 + (-(-0.2 + 0.8 * zi)).exp());
        t[i] = f64::from(rng.next_f64() < p);
        y[i] = zi + 2.0 * t[i] + 0.5 * t[i] * zi + 0.5 * standard_normal(&mut rng);
    }
    let columns = [("t", t), ("y", y), ("z", z.clone())];
    let borrowed: Vec<(&str, &[f64])> = columns.iter().map(|(n, v)| (*n, v.as_slice())).collect();
    let data = TabularData::from_f64_columns(borrowed).unwrap();
    let mut graph = Dag::with_variables(3);
    for (from, to) in [(Z, T), (Z, Y), (T, Y)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    (data, graph, z)
}

/// One replication: simulate, prepare, retarget the three claims, form the band.
fn replicate(data_seed: u64, band_seed: u64, draws: u32) -> SimultaneousBand {
    let (data, graph, z) = fixture(data_seed);
    let ctx = ExecutionContext::for_tests(band_seed);
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(T), VariableId::from_raw(Y));
    let prepared = BatchStudy::new(data, graph)
        .estimator(EstimatorId::Aipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .prepare(&[query], &ctx)
        .unwrap();
    let scores = prepared.prepared_scores();
    let rows = scores.common_rows().unwrap().unwrap();
    let claims = TILTS
        .iter()
        .enumerate()
        .map(|(k, &s)| RetargetClaim {
            name: format!("c{k}"),
            query_index: 0,
            weights: rows.iter().map(|&r| (s * z[r as usize]).exp()).collect(),
            depends_on: vec![VariableId::from_raw(Z)],
        })
        .collect();
    let request = BatchRetargetRequest { claims, contrasts: Vec::new(), expected_snapshot: None };
    let report = prepared.retarget(&scores, &request, &ctx).unwrap();
    report.simultaneous_interval(LEVEL, band_seed, draws, &ctx).unwrap()
}

/// Family-wise coverage of the max-t band over 300 fixed-seed replications is within three
/// binomial standard errors of nominal (one-sided: it must not undercover by more than that),
/// and the band is strictly wider than each marginal band.
#[test]
fn the_max_t_band_covers_the_known_retargeted_truths_family_wise() {
    let tolerance = 3.0 * (LEVEL * (1.0 - LEVEL) / REPS as f64).sqrt();
    let mut covered = 0_u64;
    let mut marginal_covered = [0_u64; 3];
    let mut critical_sum = 0.0;
    for rep in 0..REPS {
        let band = replicate(52_000 + rep, rep, BAND_DRAWS);
        assert_eq!(band.members.len(), TRUTHS.len());
        let inside = |m: &antecedent::BandMember, truth: f64| m.lower <= truth && truth <= m.upper;
        let all = band.members.iter().zip(TRUTHS).all(|(m, truth)| inside(m, truth));
        covered += u64::from(all);
        for (k, (m, truth)) in band.members.iter().zip(TRUTHS).enumerate() {
            marginal_covered[k] += u64::from((m.value - truth).abs() <= Z975 * m.std_error);
        }
        critical_sum += band.critical_value;
    }
    let coverage = covered as f64 / REPS as f64;
    eprintln!(
        "batch_retarget_known_truth reps={REPS} n={N} family_coverage={coverage} \
         marginal_coverage={:?} mean_critical={}",
        marginal_covered.map(|c| c as f64 / REPS as f64),
        critical_sum / REPS as f64
    );
    assert!(
        coverage >= LEVEL - tolerance,
        "family-wise coverage {coverage} below {LEVEL} - {tolerance}"
    );
    // The max-t multiplier averaged over the replications exceeds the marginal 1.96: the
    // Monte-Carlo error of the mean of 300 critical values is far below the gap.
    let mean_critical = critical_sum / REPS as f64;
    assert!(mean_critical > Z975 + 0.01, "mean critical value {mean_critical}");
}

/// The family band is strictly wider than each marginal Wald band on a replication with enough
/// draws that the Monte-Carlo error of `c` is negligible.
#[test]
fn the_max_t_band_is_strictly_wider_than_each_marginal_band() {
    let band = replicate(52_000, 0, 400_000);
    assert!(band.critical_value > Z975, "c = {} vs {Z975}", band.critical_value);
    for member in &band.members {
        let marginal_half = Z975 * member.std_error;
        assert!(member.upper - member.value > marginal_half, "{}", member.name);
        assert!(member.value - member.lower > marginal_half, "{}", member.name);
    }
    // Three correlated claims need less than the independent three-claim multiplier (about 2.39).
    assert!(band.critical_value < 2.4, "c = {}", band.critical_value);
}
