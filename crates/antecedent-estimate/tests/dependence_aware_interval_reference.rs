//! Dependence-aware intervals of the period-matched history contrast and the penalized temporal
//! propensity, each against a calculation written here from the definition.
//!
//! The matched contrast aggregates its pairs by calendar period and reports the Newey-West
//! (Bartlett) standard error of the weighted mean over the period series, periods entering by
//! their labels. The references below match by brute force, form the period residuals
//! `sum_q w_q (d_q - contrast)` and sum the autocovariances over every pair of period labels at
//! each lag, so they share no code with the product. A hand-solved three-period case pins the
//! arithmetic. The penalized propensity interval is the subject-level influence-function sandwich
//! of the ridge mode, checked against a two-parameter Newton solve and explicit 2x2 algebra.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    reason = "reference arithmetic by index on small deterministic fixtures"
)]

use antecedent_estimate::history_match::{
    HistoryUnit, PeriodHistoryContrast, serial_history_interval,
};
use antecedent_estimate::temporal_penalized_score::{
    FrequentistTemporalPropensity, TemporalPropensityDesign, TrajectoryDependence,
};

const Z975: f64 = 1.959_963_984_540_054;

fn unit(history: &[f64], outcome: f64, treated: bool, weight: f64, period: i64) -> HistoryUnit {
    HistoryUnit::try_new(history, outcome, treated, weight).unwrap().in_period(period)
}

/// Newey-West standard error of the weighted mean of `d` aggregated over period labels.
fn newey_west_reference(periods: &[i64], d: &[f64], w: &[f64]) -> (f64, f64, usize) {
    let total: f64 = w.iter().sum();
    let contrast: f64 = w.iter().zip(d).map(|(w, d)| w * d).sum::<f64>() / total;
    let mut labels: Vec<i64> = periods.to_vec();
    labels.sort_unstable();
    labels.dedup();
    let t = labels.len();
    let residual: Vec<f64> = labels
        .iter()
        .map(|label| {
            (0..d.len()).filter(|&k| periods[k] == *label).map(|k| w[k] * (d[k] - contrast)).sum()
        })
        .collect();
    let bandwidth =
        ((4.0 * (t as f64 / 100.0).powf(2.0 / 9.0)).floor().max(1.0) as usize).min(t - 1);
    let mut meat: f64 = residual.iter().map(|r| r * r).sum();
    for lag in 1..=bandwidth {
        let kernel = 1.0 - lag as f64 / (bandwidth as f64 + 1.0);
        for a in 0..t {
            for b in 0..t {
                if labels[a] - labels[b] == lag as i64 {
                    meat += 2.0 * kernel * residual[a] * residual[b];
                }
            }
        }
    }
    let variance = meat * t as f64 / (t as f64 - 1.0) / (total * total);
    (contrast, variance.sqrt(), bandwidth)
}

struct Record {
    treated: bool,
    period: i64,
    outcome: f64,
    weight: f64,
    score: f64,
}

/// Three treated and four control units in each calendar period, scores all distinct, with the
/// plain record of each unit kept beside it for the brute-force reference.
fn period_panel(labels: &[i64]) -> (Vec<HistoryUnit>, Vec<Record>) {
    let (mut units, mut records) = (Vec::new(), Vec::new());
    for (index, &period) in labels.iter().enumerate() {
        let shock = ((index as f64) * 1.7).sin();
        for k in 0..3 {
            let score = 0.37 * k as f64 + 0.011 * index as f64 + 0.093;
            let outcome = 1.6 + 2.0 * score + 0.9 * shock + 0.31 * ((k * 5 + index * 3) % 7) as f64;
            let weight = 1.0 + 0.25 * ((k + index) % 3) as f64;
            units.push(unit(&[score - 0.05, score + 0.05], outcome, true, weight, period));
            records.push(Record { treated: true, period, outcome, weight, score });
        }
        for k in 0..4 {
            let score = 0.29 * k as f64 + 0.017 * index as f64 + 0.041;
            let outcome = 1.0 + 2.0 * score + 0.9 * shock + 0.23 * ((k * 3 + index * 5) % 5) as f64;
            units.push(unit(&[score - 0.03, score + 0.03], outcome, false, 1.0, period));
            records.push(Record { treated: false, period, outcome, weight: 1.0, score });
        }
    }
    (units, records)
}

/// Nearest control inside the period for each treated record, by brute force.
fn brute_force_pairs(records: &[Record]) -> (Vec<i64>, Vec<f64>, Vec<f64>) {
    let (mut periods, mut d, mut w) = (vec![], vec![], vec![]);
    for treated in records.iter().filter(|r| r.treated) {
        let donor = records
            .iter()
            .filter(|r| !r.treated && r.period == treated.period)
            .min_by(|a, b| {
                (a.score - treated.score).abs().total_cmp(&(b.score - treated.score).abs())
            })
            .unwrap();
        periods.push(treated.period);
        d.push(treated.outcome - donor.outcome);
        w.push(treated.weight);
    }
    (periods, d, w)
}

#[test]
fn a_three_period_hand_case_pins_the_newey_west_arithmetic() {
    // One treated and one control per period at equal scores, so each pair difference is the
    // treated outcome minus the control outcome: 1, 3, 2 with unit weights.
    // contrast = 2; period residuals -1, 1, 0; T = 3 gives bandwidth floor(4 (0.03)^(2/9)) = 1.
    // meat = 1 + 1 + 0 + 2 (1 - 1/2) (1 * -1 + 0 * 1) = 1; variance = 1 * 3/2 / 3^2 = 1/6.
    let mut units = Vec::new();
    for (period, (treated, control)) in [(1.0, 0.0), (4.0, 1.0), (3.0, 1.0)].into_iter().enumerate()
    {
        let period = period as i64 + 1;
        units.push(unit(&[0.5], treated, true, 1.0, period));
        units.push(unit(&[0.5], control, false, 1.0, period));
    }
    let estimate = PeriodHistoryContrast::try_new(None).unwrap().contrast(&units).unwrap();
    let serial = estimate.serial.expect("three matched periods");
    let expected = (1.0_f64 / 6.0).sqrt();
    assert!((estimate.contrast - 2.0).abs() < 1e-14);
    assert_eq!(serial.bandwidth, 1);
    assert_eq!(serial.periods, vec![1, 2, 3]);
    assert!((serial.standard_error - expected).abs() < 1e-14, "{}", serial.standard_error);
    assert!((serial.lower - (2.0 - Z975 * expected)).abs() < 1e-13);
    assert!((serial.upper - (2.0 + Z975 * expected)).abs() < 1e-13);
}

#[test]
fn the_serial_interval_equals_the_period_series_long_run_variance_of_the_matched_pairs() {
    for labels in [vec![1_i64, 2, 3, 4, 5, 6], vec![1_i64, 2, 3, 5, 6, 8]] {
        let (units, records) = period_panel(&labels);
        let (periods, d, w) = brute_force_pairs(&records);
        let (contrast, se, bandwidth) = newey_west_reference(&periods, &d, &w);
        let estimate = PeriodHistoryContrast::try_new(None).unwrap().contrast(&units).unwrap();
        let serial = estimate.serial.expect("six matched periods");
        assert!((estimate.contrast - contrast).abs() < 1e-12);
        assert_eq!(serial.bandwidth, bandwidth);
        assert!((serial.standard_error - se).abs() < 1e-12, "{} vs {se}", serial.standard_error);
        assert!((serial.lower - (contrast - Z975 * se)).abs() < 1e-12);
        assert!((serial.upper - (contrast + Z975 * se)).abs() < 1e-12);
        assert_eq!(serial.periods, labels);
    }
}

#[test]
fn a_missing_period_does_not_invent_an_adjacency() {
    // The same pairs under labels with a gap and under consecutive labels have different lag
    // structure, so the serial standard errors differ; each equals its own reference.
    let consecutive = [1_i64, 2, 3, 4, 5, 6];
    let gapped = [1_i64, 2, 3, 5, 6, 8];
    let d = [0.4, -0.2, 0.7, 0.1, -0.5, 0.3];
    let w = [1.0, 1.5, 1.0, 2.0, 1.0, 1.25];
    let a = serial_history_interval(&consecutive, &d, &w).unwrap();
    let b = serial_history_interval(&gapped, &d, &w).unwrap();
    let (_, se_a, _) = newey_west_reference(&consecutive, &d, &w);
    let (_, se_b, _) = newey_west_reference(&gapped, &d, &w);
    assert!((a.standard_error - se_a).abs() < 1e-13);
    assert!((b.standard_error - se_b).abs() < 1e-13);
    assert!((a.standard_error - b.standard_error).abs() > 1e-3);
}

#[test]
fn matches_in_a_single_period_refuse_a_serial_interval_with_a_typed_reason() {
    let units = vec![
        unit(&[0.5], 3.0, true, 1.0, 7),
        unit(&[0.5], 1.0, false, 1.0, 7),
        unit(&[0.4], 2.0, true, 1.0, 7),
        unit(&[0.45], 1.5, false, 1.0, 7),
    ];
    let estimate = PeriodHistoryContrast::try_new(None).unwrap().contrast(&units).unwrap();
    assert!(estimate.serial.is_none());
    let refusal = serial_history_interval(&[7, 7], &[2.0, 0.5], &[1.0, 1.0]).unwrap_err();
    assert_eq!(refusal.reason_code(), Some("route_not_supported"));
    // A non-positive matched weight leaves no linearization to aggregate.
    assert!(serial_history_interval(&[1, 2], &[1.0, 2.0], &[1.0, 0.0]).is_err());
}

struct Lcg(u64);

impl Lcg {
    fn uniform(&mut self) -> f64 {
        self.0 =
            self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 + 0.5) / (1_u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.uniform(), self.uniform());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn inverse2(m: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    [[m[1][1] / det, -m[0][1] / det], [-m[1][0] / det, m[0][0] / det]]
}

fn matmul2(a: [[f64; 2]; 2], b: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    let mut out = [[0.0; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j];
        }
    }
    out
}

#[test]
fn the_penalized_score_interval_is_the_subject_influence_sandwich_of_the_ridge_mode() {
    let n = 400_usize;
    let penalty = 3.0;
    let mut rng = Lcg(2024);
    let (mut lag, mut treat) = (vec![], vec![]);
    for _ in 0..n {
        let c = rng.normal();
        lag.push(c);
        treat.push(f64::from(rng.uniform() < sigmoid(-0.3 + 0.8 * c)));
    }
    // Newton solve of the penalized objective, intercept unpenalized.
    let mut beta = [0.0_f64; 2];
    let hessian = |b: &[f64; 2]| {
        let mut h = [[0.0; 2]; 2];
        for i in 0..n {
            let p = sigmoid(b[0] + b[1] * lag[i]);
            let x = [1.0, lag[i]];
            for r in 0..2 {
                for c in 0..2 {
                    h[r][c] += p * (1.0 - p) * x[r] * x[c];
                }
            }
        }
        h[1][1] += penalty;
        h
    };
    for _ in 0..100 {
        let mut g = [0.0; 2];
        for i in 0..n {
            let p = sigmoid(beta[0] + beta[1] * lag[i]);
            g[0] += treat[i] - p;
            g[1] += (treat[i] - p) * lag[i];
        }
        g[1] -= penalty * beta[1];
        let inv = inverse2(hessian(&beta));
        beta = [
            beta[0] + inv[0][0] * g[0] + inv[0][1] * g[1],
            beta[1] + inv[1][0] * g[0] + inv[1][1] * g[1],
        ];
    }
    let bread = inverse2(hessian(&beta));
    let mut mean = [0.0; 2];
    let scores: Vec<[f64; 2]> = (0..n)
        .map(|i| {
            let r = treat[i] - sigmoid(beta[0] + beta[1] * lag[i]);
            [r, r * lag[i]]
        })
        .collect();
    for s in &scores {
        mean[0] += s[0] / n as f64;
        mean[1] += s[1] / n as f64;
    }
    let mut meat = [[0.0; 2]; 2];
    for s in &scores {
        for r in 0..2 {
            for c in 0..2 {
                meat[r][c] += (s[r] - mean[r]) * (s[c] - mean[c]);
            }
        }
    }
    let scale = n as f64 / (n as f64 - 1.0);
    let mut cov = matmul2(matmul2(bread, meat), bread);
    cov.iter_mut().flatten().for_each(|v| *v *= scale);

    let mut x = vec![1.0; n];
    x.extend_from_slice(&lag);
    let design =
        TemporalPropensityDesign::try_new(x, treat, n, TrajectoryDependence::IndependentSubjects)
            .unwrap();
    let histories = vec![vec![0.0], vec![1.5], vec![-2.0]];
    let (fit, interval) = FrequentistTemporalPropensity::new(penalty)
        .unwrap()
        .fit_with_interval_borrowed(&design.as_ref(), &histories)
        .unwrap();
    for j in 0..2 {
        assert!((fit.coefficients[j] - beta[j]).abs() < 1e-7);
        assert!(
            (interval.coefficient_standard_error[j] - cov[j][j].sqrt()).abs() < 1e-6,
            "{j}: {} vs {}",
            interval.coefficient_standard_error[j],
            cov[j][j].sqrt()
        );
    }
    assert!((interval.coefficient_covariance[1] - cov[0][1]).abs() < 1e-6);
    for (band, history) in interval.scores.iter().zip([0.0_f64, 1.5, -2.0]) {
        let eta = beta[0] + beta[1] * history;
        let var = cov[0][0] + 2.0 * history * cov[0][1] + history * history * cov[1][1];
        let p = sigmoid(eta);
        assert!((band.score - p).abs() < 1e-7);
        assert!((band.standard_error - p * (1.0 - p) * var.sqrt()).abs() < 1e-6);
        assert!((band.lower - sigmoid(eta - Z975 * var.sqrt())).abs() < 1e-6);
        assert!((band.upper - sigmoid(eta + Z975 * var.sqrt())).abs() < 1e-6);
        assert!(band.lower < p && p < band.upper);
    }
}

#[test]
fn a_dependent_series_refuses_the_penalized_score_interval() {
    let n = 40_usize;
    let mut rng = Lcg(9);
    let mut x = vec![1.0; n];
    let mut y = Vec::new();
    for _ in 0..n {
        let c = rng.normal();
        x.push(c);
        y.push(f64::from(rng.uniform() < sigmoid(c)));
    }
    let design =
        TemporalPropensityDesign::try_new(x, y, n, TrajectoryDependence::OneDependentSeries)
            .unwrap();
    let refusal = FrequentistTemporalPropensity::new(1.0)
        .unwrap()
        .fit_with_interval_borrowed(&design.as_ref(), &[vec![0.0]])
        .unwrap_err();
    assert_eq!(refusal.reason_code(), Some("route_not_supported"));
}
