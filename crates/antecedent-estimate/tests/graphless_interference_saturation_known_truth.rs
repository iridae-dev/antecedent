//! Independent exact-value check of the two-stage saturation exposure contrast and its
//! pointwise cluster interval.
//!
//! Design: 48 clusters of three units on a directed ring (each unit has the other two as
//! in-neighbours), 24 clusters allocated to the high saturation 0.8 and 24 to the low
//! saturation 0.2, then independent Bernoulli assignment inside each cluster. For the total
//! contrast `(own 0, neighbours 0) -> (own 1, neighbours 1)` under `NeighborFraction`, a unit's
//! exposure probability marginalizes its own cluster's allocation:
//!
//! `P(all three treated) = (0.8^3 + 0.2^3) / 2 = 0.26 = P(none treated) = (0.2^3 + 0.8^3) / 2`.
//!
//! Everything else is written out from the definitions in this file, from the assignment and
//! outcomes alone: the Horvitz-Thompson means, the Hajek means, the cluster scores
//! `(K / n) * (T_c(to) - T_c(from))`, the arm-share weighted Neyman variance, the
//! Welch-Satterthwaite degrees of freedom and the interval. The Student-t critical value is
//! checked against published table values (interpolated in `1 / df`), not against the library's
//! own quantile function. A noise-free outcome that depends only on exposure gives a Hajek
//! contrast equal to the planted exposure effect, whatever the assignment.
//!
//! The 2,000-replication coverage tests are calibration evidence and are a separate claim.
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "fixture builds small nonnegative cluster and unit counters"
)]

use antecedent_data::TableView;
use std::sync::Arc;

use antecedent_core::{
    AssignmentDesign, ExposureLevel, ExposureMapping, InterferenceFunctional, InterferenceQuery,
    VariableId,
};
use antecedent_data::{NetworkData, NetworkEdge, TabularData};
use antecedent_estimate::estimate_saturation_interference;

const CLUSTERS: usize = 48;
const SIZE: usize = 3;
const EXPOSURE_PROBABILITY: f64 = 0.26;

fn spread(key: u64) -> f64 {
    let mut z = key.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

/// Published two-sided 97.5% Student-t critical values.
const T_TABLE: [(f64, f64); 8] = [
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
        .expect("degrees of freedom inside the table");
    // The critical value is close to linear in 1 / df between table rows.
    let share = (1.0 / lo.0 - 1.0 / df) / (1.0 / lo.0 - 1.0 / hi.0);
    lo.1 + share * (hi.1 - lo.1)
}

struct Fixture {
    network: NetworkData,
    assignment: Vec<bool>,
    high: Vec<bool>,
    query: InterferenceQuery,
}

fn fixture(outcome_of: impl Fn(usize, f64, f64) -> f64) -> Fixture {
    let edges: Vec<NetworkEdge> = (0..CLUSTERS)
        .flat_map(|c| {
            (0..SIZE).flat_map(move |w| {
                let first = c * SIZE;
                [1, 2].map(|k| {
                    NetworkEdge::new((first + (w + k) % SIZE) as u64, (first + w) as u64, 1.0)
                })
            })
        })
        .collect();
    // Alternate high and low clusters; 24 each.
    let high: Vec<bool> = (0..CLUSTERS).map(|c| c % 2 == 0).collect();
    let mut assignment = vec![false; CLUSTERS * SIZE];
    for c in 0..CLUSTERS {
        let slot = c / 2;
        for w in 0..SIZE {
            let i = c * SIZE + w;
            assignment[i] = if high[c] {
                // Half of the high clusters are fully treated; the rest are mixed.
                slot % 2 == 0 || spread(i as u64) < 0.8
            } else {
                // Half of the low clusters are fully untreated; the rest are mixed.
                !(slot % 2 == 0) && spread(500 + i as u64) < 0.2
            };
        }
    }
    let outcomes: Vec<f64> = (0..CLUSTERS * SIZE)
        .map(|i| {
            let c = i / SIZE;
            let w = i % SIZE;
            let first = c * SIZE;
            let own = f64::from(assignment[i]);
            let nbr = (f64::from(assignment[first + (w + 1) % SIZE])
                + f64::from(assignment[first + (w + 2) % SIZE]))
                / 2.0;
            outcome_of(i, own, nbr)
        })
        .collect();
    let table = TabularData::from_f64_columns([("y", outcomes.as_slice())]).unwrap();
    let network = NetworkData::try_new(table, edges).unwrap();
    let query = InterferenceQuery::new(
        AssignmentDesign::TwoStageSaturation {
            clusters: Arc::from(
                (0..CLUSTERS * SIZE).map(|i| (i / SIZE) as u32).collect::<Vec<_>>(),
            ),
            low_probability: 0.2,
            high_probability: 0.8,
            high_clusters: CLUSTERS / 2,
            realized_saturation: Arc::from(
                (0..CLUSTERS * SIZE)
                    .map(|i| if high[i / SIZE] { 0.8 } else { 0.2 })
                    .collect::<Vec<_>>(),
            ),
        },
        ExposureMapping::NeighborFraction,
        InterferenceFunctional::ExposureContrast {
            outcome: VariableId::from_raw(0),
            from: ExposureLevel { own: 0.0, neighbors: 0.0 },
            to: ExposureLevel { own: 1.0, neighbors: 1.0 },
        },
    );
    Fixture { network, assignment, high, query }
}

fn exposure(assignment: &[bool], i: usize) -> Option<&'static str> {
    let first = (i / SIZE) * SIZE;
    let treated = (0..SIZE).filter(|w| assignment[first + w]).count();
    match treated {
        3 => Some("to"),
        0 => Some("from"),
        _ => None,
    }
}

#[test]
fn saturation_total_contrast_recovers_the_planted_exposure_effect_exactly() {
    // y depends on the exposure only: 1 + 2 own + 3 neighbours + own * neighbours.
    let f = fixture(|_, own, nbr| 1.0 + 2.0 * own + 3.0 * nbr + own * nbr);
    let result = estimate_saturation_interference(&f.query, &f.network, &f.assignment).unwrap();
    // y(1, 1) - y(0, 0) = 7 - 1: every exposed unit of a level has the same outcome, so the
    // Hajek mean of each level is that outcome whatever the assignment.
    assert!((result.estimate.contrast.hajek - 6.0).abs() < 1e-12);
    assert!(result.estimate.minimum_exposure_probability > 0.0);
    assert!((result.estimate.minimum_exposure_probability - EXPOSURE_PROBABILITY).abs() < 1e-12);
}

#[test]
fn saturation_pointwise_interval_matches_the_cluster_neyman_welch_construction() {
    let f = fixture(|i, own, nbr| {
        let cluster_shock = 1.5 * (spread(10_000 + (i / SIZE) as u64) - 0.5);
        let unit_noise = 0.5 * (spread(20_000 + i as u64) - 0.5);
        1.0 + 2.0 * own + 3.0 * nbr + own * nbr + cluster_shock + unit_noise
    });
    let n = CLUSTERS * SIZE;
    let outcomes: Vec<f64> = f.network.units().float64_values(VariableId::from_raw(0)).unwrap();
    let result = estimate_saturation_interference(&f.query, &f.network, &f.assignment).unwrap();

    // Horvitz-Thompson and Hajek means by their definitions.
    let (mut sum_to, mut sum_from) = (0.0, 0.0);
    let (mut raw_to, mut raw_from, mut count_to, mut count_from) = (0.0, 0.0, 0.0, 0.0);
    let mut total_to = vec![0.0; CLUSTERS];
    let mut total_from = vec![0.0; CLUSTERS];
    for i in 0..n {
        match exposure(&f.assignment, i) {
            Some("to") => {
                sum_to += outcomes[i] / EXPOSURE_PROBABILITY;
                raw_to += outcomes[i];
                count_to += 1.0;
                total_to[i / SIZE] += outcomes[i] / EXPOSURE_PROBABILITY;
            }
            Some("from") => {
                sum_from += outcomes[i] / EXPOSURE_PROBABILITY;
                raw_from += outcomes[i];
                count_from += 1.0;
                total_from[i / SIZE] += outcomes[i] / EXPOSURE_PROBABILITY;
            }
            _ => {}
        }
    }
    let point = sum_to / n as f64 - sum_from / n as f64;
    assert!((result.estimate.contrast.horvitz_thompson - point).abs() < 1e-11);
    // Equal probabilities: the Hajek mean of a level is the plain mean of its exposed units.
    assert!(
        (result.estimate.contrast.hajek - (raw_to / count_to - raw_from / count_from)).abs()
            < 1e-11
    );

    // Cluster scores, arm variance components and Welch-Satterthwaite degrees of freedom.
    let k = CLUSTERS as f64;
    let scale = k / n as f64;
    let scores: Vec<f64> = (0..CLUSTERS).map(|c| scale * (total_to[c] - total_from[c])).collect();
    let mean_score = scores.iter().sum::<f64>() / k;
    assert!((mean_score - point).abs() < 1e-11, "the point is the mean cluster score");
    let component = |want_high: bool| -> (f64, f64) {
        let arm: Vec<f64> =
            (0..CLUSTERS).filter(|&c| f.high[c] == want_high).map(|c| scores[c]).collect();
        let m = arm.len() as f64;
        let mean = arm.iter().sum::<f64>() / m;
        let sample_variance = arm.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (m - 1.0);
        ((m / k).powi(2) * sample_variance / m, m)
    };
    let (v_high, m_high) = component(true);
    let (v_low, m_low) = component(false);
    let variance = v_high + v_low;
    let df = variance.powi(2) / (v_high.powi(2) / (m_high - 1.0) + v_low.powi(2) / (m_low - 1.0));
    let se = variance.sqrt();

    let interval = result.pointwise_interval.expect("24 + 24 clusters publish an interval");
    assert_eq!((interval.high_clusters, interval.low_clusters), (24, 24));
    assert!((interval.standard_error - se).abs() <= 1e-11 * (1.0 + se));
    assert!((interval.degrees_of_freedom - df).abs() <= 1e-8 * df);
    let critical = (interval.bounds[1] - point) / se;
    assert!(((point - interval.bounds[0]) / se - critical).abs() < 1e-9, "interval is symmetric");
    assert!(
        (critical - table_critical(df)).abs() < 1e-3,
        "critical value {critical} at df {df} disagrees with the Student-t table"
    );
}
