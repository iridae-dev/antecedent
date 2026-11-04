//! Independent hand-computed checks of the *published intervals* of graphless rows whose
//! other exact-reference tests cover only the point estimate and variance.
//!
//! Every expected value below is written from the definition in this file, from the data alone,
//! with the normal quantile hard-coded (1.959963984540054) and Student quantiles taken from the
//! published t table (interpolated in 1/df, error below 1e-3), never from the library:
//!
//! * cluster-total interference (Welch interval, df);
//! * observational known-exposure interference with KNOWN propensities and 40 clusters
//!   (cluster-t interval, df = clusters - 1);
//! * switchback ITT (Student interval over sequences);
//! * fixed-coefficient CUPED (adjust the outcome by the declared coefficient, then the
//!   Horvitz-Thompson contrast): exact enumeration of every assignment, and a published interval;
//! * two-sided noncompliance Wald ratio (control-arm receipt present), exact enumeration;
//! * uplift bins, IPW and AIPW policy value and incremental intervals, and multi-action CATE
//!   group intervals.
//!
//! The repeated-sampling coverage tests are calibration evidence and are a separate claim.
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::needless_range_loop,
    reason = "fixtures build small counters; hand formulas are written by index"
)]

use std::sync::Arc;

use antecedent_core::{
    AssignmentDesign, ExposureLevel, ExposureMapping, ExposurePropensityProvenance,
    InterferenceFunctional, InterferenceQuery, MultiActionPolicyInputs, VariableId,
};
use antecedent_data::{NetworkData, NetworkEdge, TabularData};
use antecedent_estimate::policy_value::{
    BinaryPolicyDesign, evaluate_multi_action_cate, evaluate_policy_value_ipw_scores,
    evaluate_policy_value_scores, evaluate_uplift_bins, pointwise_intervals_95,
};
use antecedent_estimate::randomized_scores::{complier_wald_effect, independent_action_contrast};
use antecedent_estimate::switchback::switchback_itt;
use antecedent_estimate::{
    ObservationalExposureSpec, estimate_cluster_interference_total_with_inference,
    estimate_observational_exposure,
};

const Z975: f64 = 1.959_963_984_540_054;

fn spread(key: u64) -> f64 {
    let mut z = key.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

fn close(got: f64, want: f64, tol: f64, what: &str) {
    assert!((got - want).abs() <= tol * (1.0 + want.abs()), "{what}: got {got}, want {want}");
}

/// Published two-sided 97.5% Student-t critical values, interpolated in 1 / df.
const T_TABLE: [(f64, f64); 11] = [
    (10.0, 2.2281),
    (12.0, 2.1788),
    (15.0, 2.1314),
    (20.0, 2.0860),
    (24.0, 2.0639),
    (30.0, 2.0423),
    (40.0, 2.0211),
    (50.0, 2.0086),
    (60.0, 2.0003),
    (80.0, 1.9901),
    (120.0, 1.9799),
];

fn table_critical(df: f64) -> f64 {
    let (lo, hi) = T_TABLE
        .windows(2)
        .map(|w| (w[0], w[1]))
        .find(|(a, b)| a.0 <= df && df <= b.0)
        .expect("degrees of freedom inside the t table");
    let share = (1.0 / lo.0 - 1.0 / df) / (1.0 / lo.0 - 1.0 / hi.0);
    lo.1 + share * (hi.1 - lo.1)
}

fn sample_variance(values: &[f64]) -> f64 {
    let m = values.iter().sum::<f64>() / values.len() as f64;
    values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (values.len() - 1) as f64
}

// ------------------------------------------------------------------ cluster-total interference

#[test]
fn cluster_total_welch_interval_matches_the_hand_computed_construction() {
    let k = 20_usize;
    let sizes: Vec<usize> = (0..k).map(|c| 3 + c % 2).collect();
    let n: usize = sizes.iter().sum();
    let mut clusters = Vec::new();
    let mut edges = Vec::new();
    let mut first = 0_u64;
    for (c, &size) in sizes.iter().enumerate() {
        for j in 0..size {
            clusters.push(c as u32);
            let (from, to) = (first + j as u64, first + ((j + 1) % size) as u64);
            edges.push(NetworkEdge::new(from, to, 1.0));
            edges.push(NetworkEdge::new(to, from, 1.0));
        }
        first += size as u64;
    }
    let treated_cluster = |c: usize| c % 2 == 0 || c == 5; // 10 + 1 = 11 treated clusters
    let treated_count = (0..k).filter(|&c| treated_cluster(c)).count();
    let assignment: Vec<bool> = clusters.iter().map(|&c| treated_cluster(c as usize)).collect();
    let outcomes: Vec<f64> = (0..n)
        .map(|i| {
            let c = clusters[i] as usize;
            2.0 + 1.5 * f64::from(assignment[i])
                + 2.0 * (spread(100 + c as u64) - 0.5)
                + 0.5 * (spread(900 + i as u64) - 0.5)
        })
        .collect();
    let table = TabularData::from_f64_columns([("y", outcomes.as_slice())]).unwrap();
    let network = NetworkData::try_new(table, edges).unwrap();
    let query = InterferenceQuery::new(
        AssignmentDesign::ClusterRandomization {
            clusters: Arc::from(clusters.clone()),
            treated_clusters: treated_count,
        },
        ExposureMapping::NeighborFraction,
        InterferenceFunctional::ExposureContrast {
            outcome: VariableId::from_raw(0),
            from: ExposureLevel { own: 0.0, neighbors: 0.0 },
            to: ExposureLevel { own: 1.0, neighbors: 1.0 },
        },
    );
    let (estimate, interval) =
        estimate_cluster_interference_total_with_inference(&query, &network, &assignment).unwrap();
    let interval = interval.expect("11 and 9 clusters publish an interval");

    let scale = k as f64 / n as f64;
    let mut totals = vec![0.0; k];
    for i in 0..n {
        totals[clusters[i] as usize] += outcomes[i] * scale;
    }
    let treated: Vec<f64> = (0..k).filter(|&c| treated_cluster(c)).map(|c| totals[c]).collect();
    let control: Vec<f64> = (0..k).filter(|&c| !treated_cluster(c)).map(|c| totals[c]).collect();
    let effect = treated.iter().sum::<f64>() / treated.len() as f64
        - control.iter().sum::<f64>() / control.len() as f64;
    let vt = sample_variance(&treated) / treated.len() as f64;
    let vc = sample_variance(&control) / control.len() as f64;
    let df = (vt + vc).powi(2)
        / (vt.powi(2) / (treated.len() - 1) as f64 + vc.powi(2) / (control.len() - 1) as f64);
    let se = (vt + vc).sqrt();
    close(estimate.contrast.horvitz_thompson, effect, 1e-11, "HT total effect");
    close(estimate.contrast.conservative_variance, vt + vc, 1e-11, "Neyman variance");
    close(interval.standard_error, se, 1e-11, "interval standard error");
    close(interval.degrees_of_freedom, df, 1e-9, "Welch-Satterthwaite df");
    assert_eq!((interval.treated_clusters, interval.control_clusters), (11, 9));
    let critical = (interval.bounds[1] - effect) / se;
    close((effect - interval.bounds[0]) / se, critical, 1e-9, "interval is symmetric");
    assert!((critical - table_critical(df)).abs() < 1e-3, "t critical {critical} at df {df}");
}

// ------------------------------------------------------------- observational known exposure

#[test]
fn observational_known_propensity_cluster_t_interval_matches_the_hand_computation() {
    let clusters_n = 40_usize;
    let n = 2 * clusters_n;
    let mut edges = Vec::new();
    let mut assignment = vec![false; n];
    for c in 0..clusters_n {
        let (a, b) = ((2 * c) as u64, (2 * c + 1) as u64);
        edges.push(NetworkEdge::new(a, b, 1.0));
        edges.push(NetworkEdge::new(b, a, 1.0));
        match c % 10 {
            0..=2 => {
                assignment[2 * c] = true;
                assignment[2 * c + 1] = true; // both treated: exposure (own 1, neighbours 1)
            }
            3..=5 => {}                    // both untreated: exposure (0, 0)
            _ => assignment[2 * c] = true, // mixed: neither requested level
        }
    }
    let cluster_of: Vec<u32> = (0..n).map(|i| (i / 2) as u32).collect();
    let outcomes: Vec<f64> = (0..n)
        .map(|i| {
            1.0 + 0.7 * f64::from(assignment[i])
                + (spread(77 + (i / 2) as u64) - 0.5)
                + 0.4 * (spread(3_000 + i as u64) - 0.5)
        })
        .collect();
    let table = TabularData::from_f64_columns([("y", outcomes.as_slice())]).unwrap();
    let network = NetworkData::try_new(table, edges).unwrap();
    let (p_from, p_to) = (0.3_f64, 0.25_f64);
    let (from_probabilities, to_probabilities) = (vec![p_from; n], vec![p_to; n]);
    let mapping = ExposureMapping::NeighborCount;
    let spec = ObservationalExposureSpec {
        assignment: &assignment,
        clusters: &cluster_of,
        exposure: &mapping,
        from: ExposureLevel { own: 0.0, neighbors: 0.0 },
        to: ExposureLevel { own: 1.0, neighbors: 1.0 },
        propensity_from: &from_probabilities,
        propensity_to: &to_probabilities,
        propensity_provenance: ExposurePropensityProvenance::Known,
    };
    let got = estimate_observational_exposure(&network, VariableId::from_raw(0), &spec).unwrap();

    // Hand computation: unit i is "to" iff both cluster members are treated, "from" iff neither.
    let (mut to_sum, mut from_sum, mut to_w, mut from_w) = (0.0, 0.0, 0.0, 0.0);
    let mut influence = vec![0.0; clusters_n];
    let (mut to_clusters, mut from_clusters) = (0, 0);
    for c in 0..clusters_n {
        let (a, b) = (assignment[2 * c], assignment[2 * c + 1]);
        let is_to = a && b;
        let is_from = !a && !b;
        to_clusters += usize::from(is_to);
        from_clusters += usize::from(is_from);
        for i in [2 * c, 2 * c + 1] {
            let to_score = if is_to { outcomes[i] / p_to } else { 0.0 };
            let from_score = if is_from { outcomes[i] / p_from } else { 0.0 };
            to_sum += to_score;
            from_sum += from_score;
            if is_to {
                to_w += 1.0 / p_to;
            }
            if is_from {
                from_w += 1.0 / p_from;
            }
            influence[c] += (to_score - from_score) / n as f64;
        }
    }
    let ht = (to_sum - from_sum) / n as f64;
    let g = clusters_n as f64;
    let mean_inf = influence.iter().sum::<f64>() / g;
    let variance = g / (g - 1.0) * influence.iter().map(|v| (v - mean_inf).powi(2)).sum::<f64>();
    let se = variance.sqrt();
    assert_eq!((to_clusters, from_clusters), (12, 12));
    close(got.horvitz_thompson, ht, 1e-11, "HT contrast");
    close(got.hajek, to_sum / to_w - from_sum / from_w, 1e-11, "Hajek contrast");
    close(got.cluster_robust_variance, variance, 1e-11, "CR1 cluster variance");
    let interval = got.pointwise_interval.expect("known propensities and 40 clusters publish");
    close(interval.standard_error, se, 1e-11, "interval standard error");
    close(interval.degrees_of_freedom, g - 1.0, 1e-12, "df = clusters - 1");
    let critical = (interval.bounds[1] - ht) / se;
    close((ht - interval.bounds[0]) / se, critical, 1e-9, "interval is symmetric");
    assert!((critical - table_critical(g - 1.0)).abs() < 1e-3, "t critical {critical}");
}

// ------------------------------------------------------------------------------- switchback

#[test]
fn switchback_student_interval_matches_the_hand_computation() {
    let (sequences, periods) = (40_usize, 6_usize);
    let n = sequences * periods;
    let assignment: Vec<bool> = (0..n).map(|i| spread(5_000 + i as u64) < 0.5).collect();
    let probability = 0.5;
    let outcomes: Vec<f64> = (0..n)
        .map(|i| {
            let s = i / periods;
            1.0 + 1.2 * f64::from(assignment[i])
                + (spread(60 + s as u64) - 0.5)
                + 0.5 * (spread(8_000 + i as u64) - 0.5)
        })
        .collect();
    let ids: Vec<String> = (0..n).map(|i| format!("seq-{}", i / periods)).collect();
    let refs: Vec<&str> = ids.iter().map(String::as_str).collect();
    let got = switchback_itt(&outcomes, &assignment, &vec![probability; n], &refs).unwrap();

    let totals: Vec<f64> = (0..sequences)
        .map(|s| {
            (s * periods..(s + 1) * periods)
                .map(|i| {
                    if assignment[i] {
                        outcomes[i] / probability
                    } else {
                        -outcomes[i] / (1.0 - probability)
                    }
                })
                .sum()
        })
        .collect();
    let mean_total = totals.iter().sum::<f64>() / sequences as f64;
    let effect = mean_total * sequences as f64 / n as f64;
    let s = sequences as f64;
    let variance = s / (s - 1.0) * totals.iter().map(|t| (t - mean_total).powi(2)).sum::<f64>()
        / (n * n) as f64;
    close(got.effect, effect, 1e-11, "switchback effect");
    close(got.variance, variance, 1e-11, "sequence sandwich variance");
    assert_eq!(got.sequences, sequences);
    assert!(assignment.iter().filter(|a| **a).count() >= 30);
    let [lo, hi] = got.interval_95.expect("40 balanced sequences publish an interval");
    let se = variance.sqrt();
    let critical = (hi - effect) / se;
    close((effect - lo) / se, critical, 1e-9, "interval is symmetric");
    assert!((critical - table_critical(s - 1.0)).abs() < 1e-3, "t critical {critical}");
}

// ------------------------------------------------------------------- fixed-coefficient CUPED

#[test]
fn fixed_coefficient_cuped_is_unbiased_over_every_assignment_and_publishes_the_hand_interval() {
    // The declared coefficient c is fixed outside the outcomes; the contrast is the HT contrast
    // of Y - c * x. Because x is fixed before assignment the adjusted contrast has the same
    // expectation as the unadjusted one, whatever c is.
    let c = 1.25;
    let y0 = [0.0, 1.0, 3.0, 2.5, 0.0, 4.0, 1.5, 2.0, 7.0, 0.5];
    let tau = [2.0, -0.5, 0.0, 3.5, -1.0, 0.5, -1.5, 2.5, 4.0, -3.0];
    let x = [0.3, -1.2, 0.8, 2.0, -0.5, 1.1, -2.2, 0.0, 1.7, -0.9];
    let p1 = [0.5, 0.3, 0.7, 0.4, 0.5, 0.6, 0.25, 0.5, 0.35, 0.8];
    let n = 10;
    let nf = n as f64;
    let (mut mean, mut mean_sq, mut mean_v, mut matched) = (0.0, 0.0, 0.0, 0.0);
    for mask in 0_u32..(1 << n) {
        let mut weight = 1.0;
        let (mut adjusted, mut assignments, mut scores) = (vec![], vec![], vec![]);
        for i in 0..n {
            let treated = (mask >> i) & 1 == 1;
            weight *= if treated { p1[i] } else { 1.0 - p1[i] };
            let y = if treated { y0[i] + tau[i] } else { y0[i] } - c * x[i];
            adjusted.push(y);
            assignments.push(usize::from(treated));
            scores.push(if treated { y / p1[i] } else { -y / (1.0 - p1[i]) });
        }
        let effect = scores.iter().sum::<f64>() / nf;
        let variance = scores.iter().map(|s| (s - effect).powi(2)).sum::<f64>() / (nf * (nf - 1.0));
        mean += weight * effect;
        mean_sq += weight * effect * effect;
        mean_v += weight * variance;
        let table: Vec<Vec<f64>> = p1.iter().map(|&p| vec![1.0 - p, p]).collect();
        let Some(got) =
            independent_action_contrast(&adjusted, &assignments, table.as_slice(), 0, 1)
        else {
            assert!(mask == 0 || mask == (1 << n) - 1, "only an empty arm is refused");
            continue;
        };
        close(got.effect, effect, 1e-11, "adjusted contrast");
        close(got.variance_upper_bound, variance, 1e-11, "adjusted score variance");
        matched += weight;
    }
    assert!(matched > 0.99);
    let tau_mean = tau.iter().sum::<f64>() / nf;
    close(mean, tau_mean, 1e-11, "unbiased for the unadjusted ITT for any fixed coefficient");
    let slack = tau.iter().map(|t| (t - tau_mean).powi(2)).sum::<f64>() / (nf * (nf - 1.0));
    close(
        mean_v,
        mean_sq - mean * mean + slack,
        1e-10,
        "E[variance] = variance + effect variation",
    );

    // Published interval: 80 rows, probability 0.5, exactly 40 per arm.
    let m = 80_usize;
    let assignment: Vec<usize> =
        (0..m).map(|i| usize::from(spread(300 + i as u64) < 0.5)).collect();
    let treated_n = assignment.iter().filter(|a| **a == 1).count();
    assert!(treated_n >= 30 && m - treated_n >= 30);
    let covariate: Vec<f64> = (0..m).map(|i| (0.19 * i as f64).sin()).collect();
    let adjusted: Vec<f64> = (0..m)
        .map(|i| {
            let raw = 2.0
                + c * covariate[i]
                + 0.4 * (spread(700 + i as u64) - 0.5)
                + 1.8 * assignment[i] as f64;
            raw - c * covariate[i]
        })
        .collect();
    let table: Vec<Vec<f64>> = vec![vec![0.5, 0.5]; m];
    let got = independent_action_contrast(&adjusted, &assignment, table.as_slice(), 0, 1).unwrap();
    let scores: Vec<f64> = (0..m)
        .map(|i| if assignment[i] == 1 { adjusted[i] / 0.5 } else { -adjusted[i] / 0.5 })
        .collect();
    let effect = scores.iter().sum::<f64>() / m as f64;
    let variance = scores.iter().map(|s| (s - effect).powi(2)).sum::<f64>() / ((m - 1) * m) as f64;
    let [lo, hi] = got.interval_95.expect("80 rows at probability one half publish");
    close(lo, effect - Z975 * variance.sqrt(), 1e-11, "interval lower");
    close(hi, effect + Z975 * variance.sqrt(), 1e-11, "interval upper");
}

// --------------------------------------------------------------- two-sided noncompliance Wald

#[test]
fn wald_ratio_with_control_arm_receipt_matches_the_delta_method_over_every_assignment() {
    // Two-sided noncompliance: always-takers (rows 2 and 5) receive treatment in BOTH arms,
    // compliers (0, 3, 6, 7) only when encouraged, never-takers (1, 4) never.
    let d0 = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let d1 = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0];
    let base = [1.0, 0.0, 2.0, 3.0, 1.5, 0.5, 2.5, 4.0];
    let gain = [2.0, 1.0, -0.5, 3.0, 0.0, 1.5, -1.0, 2.5];
    let p = [0.5, 0.4, 0.6, 0.5, 0.3, 0.5, 0.45, 0.55];
    let n = 8;
    let nf = n as f64;
    let (mut itt, mut stage) = (0.0, 0.0);
    let mut control_receipts = 0;
    for mask in 0_u32..(1 << n) {
        let mut weight = 1.0;
        let (mut outcomes, mut assignment, mut received) = (vec![], vec![], vec![]);
        for i in 0..n {
            let z = (mask >> i) & 1 == 1;
            weight *= if z { p[i] } else { 1.0 - p[i] };
            let d: f64 = if z { d1[i] } else { d0[i] };
            outcomes.push(base[i] + gain[i] * d); // exclusion: assignment acts through receipt
            assignment.push(z);
            let received_now = d.to_bits() == 1.0_f64.to_bits();
            received.push(received_now);
            control_receipts += usize::from(!z && received_now);
        }
        let ys: Vec<f64> = (0..n)
            .map(|i| if assignment[i] { outcomes[i] / p[i] } else { -outcomes[i] / (1.0 - p[i]) })
            .collect();
        let ds: Vec<f64> = (0..n)
            .map(|i| {
                let r = f64::from(received[i]);
                if assignment[i] { r / p[i] } else { -r / (1.0 - p[i]) }
            })
            .collect();
        let (itt_hat, stage_hat) = (ys.iter().sum::<f64>() / nf, ds.iter().sum::<f64>() / nf);
        itt += weight * itt_hat;
        stage += weight * stage_hat;
        let Some(got) = complier_wald_effect(&outcomes, &assignment, &received, &p) else {
            assert!(stage_hat <= f64::EPSILON, "refusal only for a non-positive first stage");
            continue;
        };
        close(got.intention_to_treat_effect, itt_hat, 1e-11, "itt");
        close(got.first_stage, stage_hat, 1e-11, "first stage");
        let effect = itt_hat / stage_hat;
        close(got.effect, effect, 1e-11, "wald");
        let moment = |a: &[f64], b: &[f64]| {
            let (ma, mb) = (a.iter().sum::<f64>() / nf, b.iter().sum::<f64>() / nf);
            a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum::<f64>() / (nf - 1.0)
        };
        let variance = (moment(&ys, &ys) - 2.0 * effect * moment(&ys, &ds)
            + effect * effect * moment(&ds, &ds))
            / (nf * stage_hat * stage_hat);
        close(got.variance, variance, 1e-10, "delta-method variance");
    }
    assert!(control_receipts > 0, "the enumeration exercises control-arm receipt");
    let itt_truth = (0..n).map(|i| gain[i] * (d1[i] - d0[i])).sum::<f64>() / nf;
    let stage_truth = (0..n).map(|i| d1[i] - d0[i]).sum::<f64>() / nf;
    close(itt, itt_truth, 1e-11, "ITT unbiased under two-sided noncompliance");
    close(stage, stage_truth, 1e-11, "first stage is the complier share");
}

// ----------------------------------------------------------------------------- policy value

struct PolicyFixture {
    outcome: Vec<f64>,
    assignment: Vec<bool>,
    actions: Vec<bool>,
    reference: Vec<bool>,
    bins: Vec<usize>,
}

fn policy_fixture() -> PolicyFixture {
    let n = 600;
    PolicyFixture {
        outcome: (0..n)
            .map(|i| 1.0 + (i % 300) as f64 / 300.0 + 0.8 * (spread(40 + i as u64) - 0.5))
            .collect(),
        assignment: (0..n).map(|i| i % 2 == 0).collect(),
        actions: (0..n).map(|i| i % 3 != 0).collect(),
        reference: vec![false; n],
        bins: (0..n).map(|i| i / 300).collect(),
    }
}

fn hand_uplift_bins(f: &PolicyFixture, p: f64) -> Vec<(f64, f64, [f64; 2])> {
    (0..2)
        .map(|b| {
            let scores: Vec<f64> = (0..f.outcome.len())
                .filter(|&i| f.bins[i] == b)
                .map(|i| if f.assignment[i] { f.outcome[i] / p } else { -f.outcome[i] / (1.0 - p) })
                .collect();
            let m = scores.len() as f64;
            let effect = scores.iter().sum::<f64>() / m;
            let se =
                (scores.iter().map(|s| (s - effect).powi(2)).sum::<f64>() / (m * (m - 1.0))).sqrt();
            (effect, se, [effect - Z975 * se, effect + Z975 * se])
        })
        .collect()
}

fn check_uplift_bins(f: &PolicyFixture) {
    let bins = evaluate_uplift_bins(&f.outcome, &f.assignment, &[0.5], &f.bins, 2).unwrap();
    for (b, (effect, se, interval)) in hand_uplift_bins(f, 0.5).into_iter().enumerate() {
        assert_eq!(bins[b].evaluation_rows, 300);
        close(bins[b].effect, effect, 1e-11, "bin effect");
        close(bins[b].standard_error, se, 1e-11, "bin standard error");
        let got = bins[b].interval_95.expect("300 rows and 150 per arm publish a bin interval");
        close(got[0], interval[0], 1e-11, "bin interval lower");
        close(got[1], interval[1], 1e-11, "bin interval upper");
    }
}

fn hand_policy_scores(
    f: &PolicyFixture,
    p: f64,
    nuisance: Option<(&[f64], &[f64])>,
    cost: f64,
) -> [Vec<f64>; 3] {
    let n = f.outcome.len();
    let score = |i: usize, a: bool| {
        let expected = nuisance.map_or(0.0, |(m0, m1)| if a { m1[i] } else { m0[i] });
        expected
            + if f.assignment[i] == a {
                (f.outcome[i] - expected) / if a { p } else { 1.0 - p }
            } else {
                0.0
            }
    };
    let policy: Vec<f64> =
        (0..n).map(|i| score(i, f.actions[i]) - cost * f64::from(f.actions[i])).collect();
    let reference: Vec<f64> = (0..n).map(|i| score(i, f.reference[i])).collect();
    let incremental = policy.iter().zip(&reference).map(|(a, b)| a - b).collect();
    [policy, reference, incremental]
}

fn check_policy_intervals(f: &PolicyFixture, nuisance: Option<(&[f64], &[f64])>) {
    let n = f.outcome.len();
    let (p, cost) = (0.5, 0.1);
    let design = BinaryPolicyDesign {
        assignment: &f.assignment,
        actions: &f.actions,
        propensity: &[p],
        reference: &f.reference,
        costs: &[cost],
        reference_costs: &[0.0],
    };
    let scores = match nuisance {
        Some((m0, m1)) => evaluate_policy_value_scores(&f.outcome, &design, m0, m1).unwrap(),
        None => evaluate_policy_value_ipw_scores(&f.outcome, &design).unwrap(),
    };
    let [policy, reference, incremental] = hand_policy_scores(f, p, nuisance, cost);
    let nf = n as f64;
    let mean = |v: &[f64]| v.iter().sum::<f64>() / nf;
    let se = |v: &[f64]| {
        let m = mean(v);
        (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (nf * (nf - 1.0))).sqrt()
    };
    close(scores.policy_value, mean(&policy), 1e-11, "policy value");
    close(scores.reference_value, mean(&reference), 1e-11, "reference value");
    close(scores.incremental_standard_error, se(&incremental), 1e-11, "incremental se");
    let matches = |actions: &[bool]| (0..n).filter(|&i| f.assignment[i] == actions[i]).count();
    let got = pointwise_intervals_95(&scores, n, matches(&f.actions), matches(&f.reference))
        .expect("600 rows and matched rows publish the policy intervals");
    close(got.policy[0], mean(&policy) - Z975 * se(&policy), 1e-11, "policy lower");
    close(got.policy[1], mean(&policy) + Z975 * se(&policy), 1e-11, "policy upper");
    close(
        got.incremental[0],
        mean(&incremental) - Z975 * se(&incremental),
        1e-11,
        "incremental lower",
    );
    close(
        got.incremental[1],
        mean(&incremental) + Z975 * se(&incremental),
        1e-11,
        "incremental upper",
    );
}

#[test]
fn ipw_uplift_bins_and_policy_intervals_match_the_hand_computation() {
    let f = policy_fixture();
    check_uplift_bins(&f);
    check_policy_intervals(&f, None);
}

#[test]
fn aipw_uplift_bins_and_policy_intervals_match_the_hand_computation() {
    // The bins are the held-out Horvitz-Thompson bins; the policy and incremental intervals use
    // the doubly robust scores with frozen held-out nuisance predictions.
    let f = policy_fixture();
    let n = f.outcome.len();
    let mu0: Vec<f64> = (0..n)
        .map(|i| 1.0 + 0.5 * (i % 300) as f64 / 300.0 + 0.1 * spread(90 + i as u64))
        .collect();
    let mu1: Vec<f64> = (0..n).map(|i| mu0[i] + 0.3 + 0.2 * spread(95 + i as u64)).collect();
    check_uplift_bins(&f);
    check_policy_intervals(&f, Some((&mu0, &mu1)));
}

#[test]
fn multi_action_cate_group_intervals_match_the_hand_computation() {
    let (g1, g2) = (360_usize, 330_usize);
    let n = g1 + g2;
    let k = 3;
    let labels: Vec<Arc<str>> = vec![Arc::from("control"), Arc::from("a"), Arc::from("b")];
    let assignment: Vec<usize> = (0..n)
        .map(|i| match i % 10 {
            0..=3 => 0,
            4..=6 => 1,
            _ => 2,
        })
        .collect();
    let outcome: Vec<f64> = (0..n)
        .map(|i| {
            1.0 + 0.6 * assignment[i] as f64
                + (if i < g1 { 0.0 } else { 0.5 })
                + (spread(500 + i as u64) - 0.5)
        })
        .collect();
    let group = |i: usize| if i < g1 { "g1" } else { "g2" };
    let propensities: Vec<f64> = (0..n).flat_map(|_| [0.4, 0.3, 0.3]).collect();
    let policy = MultiActionPolicyInputs {
        action_labels: Arc::from(labels),
        assignment: Arc::from(assignment.clone()),
        actions: Arc::from(vec![0_usize; n]),
        reference: Arc::from(vec![0_usize; n]),
        propensities: Arc::from(propensities),
        available: Arc::from(vec![true; n * k]),
        costs: Arc::from(vec![0.0; k]),
        reference_costs: Arc::from(vec![0.0; k]),
        capacities: Arc::from(vec![n; k]),
        reference_capacities: Arc::from(vec![n; k]),
        budget: None,
        reference_budget: None,
        cate_groups: Arc::from((0..n).map(|i| Arc::<str>::from(group(i))).collect::<Vec<_>>()),
    };
    let got = evaluate_multi_action_cate(&outcome, &policy).unwrap();
    assert_eq!(got.len(), 4);
    let mut index = 0;
    for name in ["g1", "g2"] {
        let rows: Vec<usize> = (0..n).filter(|&i| group(i) == name).collect();
        for action in 1..k {
            let prob = [0.4, 0.3, 0.3];
            let scores: Vec<f64> = rows
                .iter()
                .map(|&i| {
                    if assignment[i] == action {
                        outcome[i] / prob[action]
                    } else if assignment[i] == 0 {
                        -outcome[i] / prob[0]
                    } else {
                        0.0
                    }
                })
                .collect();
            let m = rows.len() as f64;
            let effect = scores.iter().sum::<f64>() / m;
            let se =
                (scores.iter().map(|s| (s - effect).powi(2)).sum::<f64>() / (m * (m - 1.0))).sqrt();
            let point = &got[index];
            index += 1;
            assert_eq!(point.group, name);
            close(point.effect, effect, 1e-11, "CATE effect");
            close(point.standard_error, se, 1e-11, "CATE standard error");
            let [lo, hi] = point.interval_95.expect("300+ rows, 50+ per arm, overlap 0.3 publish");
            close(lo, effect - Z975 * se, 1e-11, "CATE interval lower");
            close(hi, effect + Z975 * se, 1e-11, "CATE interval upper");
        }
    }
}
