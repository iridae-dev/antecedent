//! Independent known-truth checks for the arm-stratified subject-bootstrap survival rows:
//! the scalar RMST and horizon survival-difference percentile intervals (plain, delayed
//! entry, fixed known censoring, delayed entry with fixed known censoring) and the two
//! simultaneous supremum bands (survival difference, competing incidence).
//!
//! The fixtures have one event time, so a bootstrap replicate is a function of the number of
//! events `K` redrawn in each arm, and `K ~ Binomial(n, observed event fraction)` exactly
//! (arms are resampled separately, subjects with replacement). The exact sampling law of every
//! bootstrap statistic is therefore computable here by enumerating `(K0, K1)` against binomial
//! probabilities written out below, with no reference to the library:
//!
//! * point values: Kaplan-Meier `1 - e / n` and restricted mean `1 + 2 S(1)` by hand; for
//!   fixed censoring weights `1 / G` the weighted hazard `w_e K / (w_e K + w_c (n - K))`;
//! * the interval endpoints must be the exact 2.5% and 97.5% (and, for a band, 95%) quantiles
//!   of that law, within the Monte-Carlo error of the stated number of draws (a window of six
//!   standard deviations of the empirical distribution function, computed from the draws).
//!
//! The 2,000-study coverage tests in `survival.rs` are calibration evidence and are a separate
//! claim.
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    reason = "exact enumeration over small integer counts"
)]

use antecedent_estimate::survival::{
    KnownCensoring, SurvivalEndpoint, SurvivalRows, randomized_survival_bootstrap_difference_band,
    randomized_survival_bootstrap_intervals, randomized_survival_ipcw_summary,
    randomized_survival_summary,
};

const TAU: f64 = 3.0;
const DRAWS: u32 = 40_000;

/// `Binomial(n, q)` probabilities for `k = 0..=n`.
fn binomial(n: usize, q: f64) -> Vec<f64> {
    let mut pmf = vec![0.0; n + 1];
    pmf[0] = (1.0 - q).powi(n as i32);
    for k in 0..n {
        pmf[k + 1] = pmf[k] * (n - k) as f64 / (k + 1) as f64 * q / (1.0 - q);
    }
    assert!((pmf.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    pmf
}

/// A finite exact law: ascending support with cumulative probabilities.
struct Law {
    support: Vec<(f64, f64)>,
}

impl Law {
    fn new(mut atoms: Vec<(f64, f64)>) -> Self {
        atoms.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut cumulative = 0.0;
        let support = atoms
            .into_iter()
            .map(|(x, p)| {
                cumulative += p;
                (x, cumulative)
            })
            .collect();
        Law { support }
    }

    /// Generalized inverse of the distribution function.
    fn quantile(&self, p: f64) -> f64 {
        let p = p.clamp(0.0, 1.0);
        self.support
            .iter()
            .find(|&&(_, cumulative)| cumulative >= p - 1e-15)
            .map_or(self.support.last().unwrap().0, |&(x, _)| x)
    }

    /// The exact quantile must be matched by the empirical one up to Monte-Carlo error.
    fn assert_quantile(&self, observed: f64, p: f64, what: &str) {
        let delta = 6.0 * (p * (1.0 - p) / f64::from(DRAWS)).sqrt();
        let (lo, hi) = (self.quantile(p - delta), self.quantile(p + delta));
        assert!(
            observed >= lo - 1e-9 && observed <= hi + 1e-9,
            "{what}: {observed} outside the exact [{lo}, {hi}] window at p = {p}"
        );
    }
}

/// Exact law of `f(K0, K1)` with `K_a ~ Binomial(n_a, q_a)`.
fn joint_law(n: [usize; 2], q: [f64; 2], f: impl Fn(usize, usize) -> f64) -> Law {
    let (p0, p1) = (binomial(n[0], q[0]), binomial(n[1], q[1]));
    let mut atoms = Vec::new();
    for (k0, a) in p0.iter().enumerate() {
        for (k1, b) in p1.iter().enumerate() {
            atoms.push((f(k0, k1), a * b));
        }
    }
    Law::new(atoms)
}

struct Arms {
    duration: Vec<f64>,
    event: Vec<i64>,
    treated: Vec<bool>,
    n: [usize; 2],
    events: [usize; 2],
}

/// `n[a]` subjects in arm `a`, the first `events[a]` of them with the event at time 1 and the
/// rest censored at the horizon.
fn arms(n: [usize; 2], events: [usize; 2]) -> Arms {
    let (mut duration, mut event, mut treated) = (vec![], vec![], vec![]);
    for arm in 0..2 {
        for i in 0..n[arm] {
            let is_event = i < events[arm];
            duration.push(if is_event { 1.0 } else { TAU });
            event.push(i64::from(is_event));
            treated.push(arm == 1);
        }
    }
    Arms { duration, event, treated, n, events }
}

/// Fixed known censoring: event subjects carry `G = 0.5`, censored subjects `G = 0.8`.
fn censoring_grid(a: &Arms) -> Vec<f64> {
    a.event
        .iter()
        .flat_map(|&e| {
            let g = if e == 1 { 0.5 } else { 0.8 };
            [1.0, g, g]
        })
        .collect()
}

fn survival_unweighted(n: usize, k: usize) -> f64 {
    1.0 - k as f64 / n as f64
}

fn survival_weighted(n: usize, k: usize) -> f64 {
    let (event_weight, other_weight) = (1.0 / 0.5, 1.0 / 0.8);
    let event_total = event_weight * k as f64;
    1.0 - event_total / (event_total + other_weight * (n - k) as f64)
}

fn scenario(weighted: bool) {
    let a = arms([120, 120], [36, 24]);
    let grid = [0.0, 1.0, TAU];
    let g = censoring_grid(&a);
    let rows = SurvivalRows {
        duration: &a.duration,
        event_code: &a.event,
        treated: &a.treated,
        delayed_entry: None,
    };
    let known = KnownCensoring { times: &grid, survival: &g, minimum_probability: 0.05 };
    let survival = |n: usize, k: usize| {
        if weighted { survival_weighted(n, k) } else { survival_unweighted(n, k) }
    };

    // Point values by hand: S(tau) = S(1) and RMST = 1 + 2 S(1).
    let summary = if weighted {
        randomized_survival_ipcw_summary(&rows, &known, TAU, SurvivalEndpoint::Survival).unwrap().0
    } else {
        randomized_survival_summary(&rows, TAU, SurvivalEndpoint::Survival).unwrap()
    };
    let (s0, s1) = (survival(a.n[0], a.events[0]), survival(a.n[1], a.events[1]));
    assert!((summary.control.last().unwrap() - s0).abs() < 1e-12);
    assert!((summary.treated.last().unwrap() - s1).abs() < 1e-12);
    assert!((summary.rmst_control.unwrap() - (1.0 + 2.0 * s0)).abs() < 1e-12);
    assert!((summary.rmst_treated.unwrap() - (1.0 + 2.0 * s1)).abs() < 1e-12);

    let interval = randomized_survival_bootstrap_intervals(
        &rows,
        weighted.then_some(&known),
        TAU,
        SurvivalEndpoint::Survival,
        DRAWS,
        20_261_103,
    )
    .unwrap();
    assert!(f64::from(interval.replicates_ok) >= 0.99 * f64::from(DRAWS));

    let q = [a.events[0] as f64 / a.n[0] as f64, a.events[1] as f64 / a.n[1] as f64];
    let survival_difference =
        joint_law(a.n, q, |k0, k1| survival(a.n[1], k1) - survival(a.n[0], k0));
    let rmst_difference =
        joint_law(a.n, q, |k0, k1| 2.0 * (survival(a.n[1], k1) - survival(a.n[0], k0)));
    let rmst = interval.rmst_difference.expect("survival endpoint reports RMST");
    survival_difference.assert_quantile(interval.difference_at_tau[0], 0.025, "survival lower");
    survival_difference.assert_quantile(interval.difference_at_tau[1], 0.975, "survival upper");
    rmst_difference.assert_quantile(rmst[0], 0.025, "RMST lower");
    rmst_difference.assert_quantile(rmst[1], 0.975, "RMST upper");
    // The point difference sits inside its own bootstrap interval.
    assert!(interval.difference_at_tau[0] < s1 - s0 && s1 - s0 < interval.difference_at_tau[1]);
}

/// `ln(n!)` for `n = 0..=max`.
fn ln_factorials(max: usize) -> Vec<f64> {
    let mut table = vec![0.0; max + 1];
    for k in 1..=max {
        table[k] = table[k - 1] + (k as f64).ln();
    }
    table
}

/// Exact law of a function of the trinomial counts `(K, E, L)` of `n` iid subjects drawn from
/// categories with probabilities `q = [event, early censored, late censored]`.
fn arm_atoms(n: usize, q: [f64; 3], value: impl Fn(usize, usize) -> f64) -> Vec<(f64, f64)> {
    let lf = ln_factorials(n);
    let mut atoms = Vec::new();
    for k in 0..=n {
        for e in 0..=(n - k) {
            let l = n - k - e;
            let log_p = lf[n] - lf[k] - lf[e] - lf[l]
                + k as f64 * q[0].ln()
                + e as f64 * q[1].ln()
                + l as f64 * q[2].ln();
            atoms.push((value(k, e), log_p.exp()));
        }
    }
    atoms
}

fn difference_law(a0: &[(f64, f64)], a1: &[(f64, f64)], scale: f64) -> Law {
    let mut atoms = Vec::with_capacity(a0.len() * a1.len());
    for &(s0, p0) in a0 {
        for &(s1, p1) in a1 {
            atoms.push((scale * (s1 - s0), p0 * p1));
        }
    }
    Law::new(atoms)
}

/// Delayed entry that binds: three kinds of subject per arm. Events and early-censored
/// subjects enter at time zero; late-censored subjects enter at time 2, after the event time 1,
/// so they are NOT in the risk set at the event time. The Kaplan-Meier step is therefore
/// `K / (K + E)` (events over events plus early censored), a function of the redrawn trinomial
/// counts whose exact law is enumerated here; ignoring left truncation would give
/// `K / (K + E + L)` and a different law.
fn late_entry_scenario(weighted: bool) {
    let n = 40_usize;
    let composition = [[12_usize, 16, 12], [8, 20, 12]];
    let (mut duration, mut event, mut treated, mut entry, mut g) =
        (vec![], vec![], vec![], vec![], vec![]);
    for arm in 0..2 {
        let [events, early, late] = composition[arm];
        for i in 0..n {
            let kind = if i < events { 0 } else if i < events + early { 1 } else { 2 };
            duration.push(if kind == 0 { 1.0 } else { TAU });
            event.push(i64::from(kind == 0));
            treated.push(arm == 1);
            entry.push(if kind == 2 { 2.0 } else { 0.0 });
            let gv = if kind == 0 { 0.5 } else { 0.8 };
            g.extend([1.0, gv, gv]);
        }
    }
    let grid = [0.0, 1.0, TAU];
    let rows = SurvivalRows {
        duration: &duration,
        event_code: &event,
        treated: &treated,
        delayed_entry: Some(&entry),
    };
    let known = KnownCensoring { times: &grid, survival: &g, minimum_probability: 0.05 };
    let survival = |k: usize, e: usize| -> f64 {
        if k == 0 {
            1.0
        } else if weighted {
            1.0 - 2.0 * k as f64 / (2.0 * k as f64 + 1.25 * e as f64)
        } else {
            1.0 - k as f64 / (k + e) as f64
        }
    };
    let (s0, s1) = (survival(12, 16), survival(8, 20));
    let summary = if weighted {
        randomized_survival_ipcw_summary(&rows, &known, TAU, SurvivalEndpoint::Survival).unwrap().0
    } else {
        randomized_survival_summary(&rows, TAU, SurvivalEndpoint::Survival).unwrap()
    };
    assert!((summary.control.last().unwrap() - s0).abs() < 1e-12);
    assert!((summary.treated.last().unwrap() - s1).abs() < 1e-12);
    assert!((summary.rmst_control.unwrap() - (1.0 + 2.0 * s0)).abs() < 1e-12);
    assert!((summary.rmst_treated.unwrap() - (1.0 + 2.0 * s1)).abs() < 1e-12);
    // Left truncation binds: the untruncated risk set would give a different curve.
    let untruncated = 1.0 - 12.0 / 40.0;
    assert!((s0 - untruncated).abs() > 0.05);

    let interval = randomized_survival_bootstrap_intervals(
        &rows,
        weighted.then_some(&known),
        TAU,
        SurvivalEndpoint::Survival,
        DRAWS,
        20_261_105,
    )
    .unwrap();
    assert!(f64::from(interval.replicates_ok) >= 0.99 * f64::from(DRAWS));
    let q = |arm: usize| composition[arm].map(|c| c as f64 / n as f64);
    let a0 = arm_atoms(n, q(0), &survival);
    let a1 = arm_atoms(n, q(1), &survival);
    let survival_law = difference_law(&a0, &a1, 1.0);
    let rmst_law = difference_law(&a0, &a1, 2.0);
    let rmst = interval.rmst_difference.expect("survival endpoint reports RMST");
    survival_law.assert_quantile(interval.difference_at_tau[0], 0.025, "survival lower");
    survival_law.assert_quantile(interval.difference_at_tau[1], 0.975, "survival upper");
    rmst_law.assert_quantile(rmst[0], 0.025, "RMST lower");
    rmst_law.assert_quantile(rmst[1], 0.975, "RMST upper");
}

#[test]
fn randomized_survival_scalar_intervals_match_the_exact_bootstrap_law() {
    scenario(false);
}

#[test]
fn delayed_entry_scalar_intervals_match_the_exact_bootstrap_law() {
    late_entry_scenario(false);
}

#[test]
fn fixed_known_censoring_scalar_intervals_match_the_exact_weighted_bootstrap_law() {
    scenario(true);
}

#[test]
fn delayed_entry_fixed_known_censoring_scalar_intervals_match_the_exact_weighted_bootstrap_law() {
    late_entry_scenario(true);
}

fn band_radius_check(
    endpoint: SurvivalEndpoint,
    event: Vec<i64>,
    n: [usize; 2],
    q: [f64; 2],
    sign: f64,
    point: f64,
) {
    let mut duration = vec![];
    let mut treated = vec![];
    for arm in 0..2 {
        for i in 0..n[arm] {
            duration.push(if event[arm * n[0] + i] > 0 { 1.0 } else { TAU });
            treated.push(arm == 1);
        }
    }
    let band = randomized_survival_bootstrap_difference_band(
        &duration,
        &event,
        &treated,
        TAU,
        endpoint,
        DRAWS / 2,
        20_261_104,
    )
    .unwrap();
    assert_eq!(band.times, [0.0, 1.0, TAU]);
    // Curve difference by hand: zero at time zero, then the horizon value.
    assert!(band.difference[0].abs() < 1e-12);
    assert!((band.difference[1] - point).abs() < 1e-12);
    assert!((band.difference[2] - point).abs() < 1e-12);
    // Common radius at every grid time (no clipping at these values), including time zero.
    let radius = band.upper[1] - band.difference[1];
    assert!(radius > 0.0);
    for j in 0..3 {
        assert!((band.upper[j] - band.difference[j] - radius).abs() < 1e-12);
        assert!((band.difference[j] - band.lower[j] - radius).abs() < 1e-12);
    }
    // Exact law of the supremum |D* - D| over the grid: only the horizon value moves.
    let law = joint_law(n, q, |k0, k1| {
        (sign * (k1 as f64 / n[1] as f64 - k0 as f64 / n[0] as f64) - point).abs()
    });
    let draws = f64::from(DRAWS / 2);
    let delta = 6.0 * (0.95 * 0.05 / draws).sqrt();
    let (lo, hi) = (law.quantile(0.95 - delta), law.quantile(0.95 + delta));
    assert!(radius >= lo - 1e-9 && radius <= hi + 1e-9, "radius {radius} outside [{lo}, {hi}]");
}

#[test]
fn simultaneous_survival_difference_band_radius_is_the_exact_supremum_quantile() {
    // 100 per arm; 30 and 20 events at time 1. Survival difference = K0 / n - K1 / n, 0.1 here.
    let mut event = vec![0_i64; 200];
    event[..30].fill(1);
    event[100..120].fill(1);
    band_radius_check(SurvivalEndpoint::Survival, event, [100, 100], [0.3, 0.2], -1.0, 0.1);
}

#[test]
fn simultaneous_competing_incidence_band_radius_is_the_exact_supremum_quantile() {
    // Cause one: 30 events in the control arm, 20 in the treated arm; cause two: 10 in each.
    // Cumulative incidence of cause one at the horizon is K / n, so the difference is
    // K1 / n - K0 / n, -0.1 here.
    let mut event = vec![0_i64; 200];
    event[..30].fill(1);
    event[30..40].fill(2);
    event[100..120].fill(1);
    event[120..130].fill(2);
    band_radius_check(
        SurvivalEndpoint::CumulativeIncidence { target_cause: 1 },
        event,
        [100, 100],
        [0.3, 0.2],
        1.0,
        -0.1,
    );
}
