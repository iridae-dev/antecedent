//! Conformal coverage and multi-unit recovery on simulated synthetic panels.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{ExecutionContext, SyntheticPanelMethod};

use super::{SyntheticRolloutPanel, conformal_synthetic_control, multi_unit_synthetic};
use crate::splitmix::FixtureRng;

fn normal(rng: &mut FixtureRng) -> f64 {
    let u = rng.uniform().max(1e-300);
    let v = rng.uniform();
    (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
}

struct Rows {
    outcome: Vec<f64>,
    units: Vec<Arc<str>>,
    periods: Vec<i64>,
}

impl Rows {
    fn panel(&self) -> SyntheticRolloutPanel<'_> {
        SyntheticRolloutPanel { outcome: &self.outcome, units: &self.units, periods: &self.periods }
    }
}

/// Ten donors with period-exchangeable outcomes; the treated unit is a fixed convex
/// combination of them plus noise, with effect `effect` from `intervention` on.
fn exchangeable_panel(seed: u64, periods: i64, intervention: i64, effect: f64) -> Rows {
    let mut rng = FixtureRng::new(seed);
    let weights = [0.5, 0.3, 0.2];
    let mut rows = Rows { outcome: Vec::new(), units: Vec::new(), periods: Vec::new() };
    let labels: Vec<Arc<str>> = (0..10).map(|j| Arc::from(format!("donor{j:02}"))).collect();
    for period in 1..=periods {
        let donors: Vec<f64> = (0..10).map(|j| f64::from(j) + normal(&mut rng)).collect();
        for (label, value) in labels.iter().zip(&donors) {
            rows.outcome.push(*value);
            rows.units.push(Arc::clone(label));
            rows.periods.push(period);
        }
        let mut treated: f64 = weights.iter().zip(&donors[3..6]).map(|(w, y)| w * y).sum::<f64>()
            + 0.5 * normal(&mut rng);
        if period >= intervention {
            treated += effect;
        }
        rows.outcome.push(treated);
        rows.units.push(Arc::from("treated"));
        rows.periods.push(period);
    }
    rows
}

#[test]
fn conformal_period_intervals_cover_the_effect_at_the_nominal_rate() {
    let seeds = 100;
    let mut covered = 0usize;
    let mut width_sum = 0.0;
    for seed in 0..seeds {
        let rows = exchangeable_panel(500 + seed, 40, 40, 1.0);
        let fit = conformal_synthetic_control(
            &rows.panel(),
            "treated",
            40,
            &ExecutionContext::for_tests(seed),
        )
        .unwrap();
        let interval = fit.periods[0].interval.unwrap();
        covered += usize::from(interval[0] <= 1.0 && 1.0 <= interval[1]);
        width_sum += interval[1] - interval[0];
    }
    let mean_width = width_sum / seeds as f64;
    assert!(
        (1.2..3.0).contains(&mean_width),
        "mean interval width {mean_width} (observed 2.19; residual sd 0.5 gives about 4 sd)"
    );
    // Exact 95% under exchangeable residuals; a sanity floor, not a coverage record.
    assert!(covered >= 88, "covered {covered} of {seeds}");
}

#[test]
fn conformal_point_and_constant_effect_interval_recover_a_known_effect() {
    let rows = exchangeable_panel(7, 45, 41, 3.0);
    let fit =
        conformal_synthetic_control(&rows.panel(), "treated", 41, &ExecutionContext::for_tests(1))
            .unwrap();
    assert_eq!(fit.post_periods, 5);

    assert!((fit.overall.estimate - 3.0).abs() < 0.5, "{}", fit.overall.estimate);
    let interval = fit.overall.interval.unwrap();
    assert!(interval[0] < fit.overall.estimate && fit.overall.estimate < interval[1]);
    assert!(interval[0] > 0.0, "{interval:?}");
    assert!(interval[0] < 3.0 && 3.0 < interval[1], "the true effect 3 lies in {interval:?}");
    let donor_mass: f64 = fit.donor_weights.iter().map(|(_, w)| w).sum();
    assert!((donor_mass - 1.0).abs() < 1e-8);
}

#[test]
fn a_short_pre_period_is_refused_with_the_next_attempt() {
    let rows = exchangeable_panel(8, 12, 10, 1.0);
    let error =
        conformal_synthetic_control(&rows.panel(), "treated", 10, &ExecutionContext::for_tests(1))
            .unwrap_err()
            .to_string();
    assert!(error.contains("19 pre-periods") && error.contains("next attempt"), "{error}");
}

/// Thirty never-treated units and three units adopting at 16, 18 and 20 in 22 periods;
/// outcomes are a unit level plus a common trend plus noise, and the effect is 2.
fn staggered_panel(seed: u64) -> (Rows, Vec<(Arc<str>, i64)>) {
    let mut rng = FixtureRng::new(seed);
    let mut rows = Rows { outcome: Vec::new(), units: Vec::new(), periods: Vec::new() };
    let mut trend = Vec::new();
    let mut level = 0.0;
    for _ in 0..22 {
        level += normal(&mut rng);
        trend.push(level);
    }
    let adoptions: Vec<(Arc<str>, i64)> =
        vec![(Arc::from("t0"), 16), (Arc::from("t1"), 18), (Arc::from("t2"), 20)];
    let mut units: Vec<(Arc<str>, f64, Option<i64>)> =
        (0..30).map(|j| (Arc::from(format!("c{j:02}")), f64::from(j) / 10.0, None)).collect();
    for (k, (label, adoption)) in adoptions.iter().enumerate() {
        units.push((
            Arc::clone(label),
            1.0 + f64::from(u8::try_from(k).unwrap()) / 2.0,
            Some(*adoption),
        ));
    }
    for (label, unit_level, adoption) in &units {
        for (t, common) in trend.iter().enumerate() {
            let period = i64::try_from(t).unwrap() + 1;
            let effect = if adoption.is_some_and(|a| period >= a) { 2.0 } else { 0.0 };
            rows.outcome.push(unit_level + common + effect + 0.3 * normal(&mut rng));
            rows.units.push(Arc::clone(label));
            rows.periods.push(period);
        }
    }
    (rows, adoptions)
}

#[test]
fn multi_unit_synthetic_control_recovers_a_staggered_effect_with_placebo_intervals() {
    let (rows, adoptions) = staggered_panel(31);
    let error = multi_unit_synthetic(
        &rows.panel(),
        &adoptions,
        SyntheticPanelMethod::DifferenceInDifferences,
        199,
        &ExecutionContext::for_tests(4),
    )
    .unwrap_err();
    assert_eq!(error.reason_code(), Some("route_not_supported"));
    assert!(error.to_string().contains("intercept and ridge"));
    for method in [SyntheticPanelMethod::Control] {
        let serial = multi_unit_synthetic(
            &rows.panel(),
            &adoptions,
            method,
            199,
            &ExecutionContext::for_tests(4),
        )
        .unwrap();
        let parallel = multi_unit_synthetic(
            &rows.panel(),
            &adoptions,
            method,
            199,
            &ExecutionContext::production(4, 4),
        )
        .unwrap();
        assert_eq!(serial, parallel);

        assert!(
            (serial.overall.estimate - 2.0).abs() < 0.25,
            "{method:?}: {}",
            serial.overall.estimate
        );
        let se = serial.overall.standard_error.unwrap();
        assert!((serial.overall.estimate - 2.0).abs() < 4.0 * se, "{method:?}: se {se}");
        let interval = serial.overall.interval.unwrap();
        assert!(interval[0] < serial.overall.estimate && serial.overall.estimate < interval[1]);
        assert_eq!(serial.units.len(), 3);
        assert!(serial.units.iter().all(|unit| unit.band.is_some()));
        assert!(serial.placebo.unwrap().replicates_ok >= 190);
    }
}

#[test]
fn too_few_never_treated_donors_are_refused() {
    let (rows, _) = staggered_panel(32);
    let adoptions: Vec<(Arc<str>, i64)> =
        (0..30).map(|j| (Arc::from(format!("c{j:02}")), 16)).collect();
    let error = multi_unit_synthetic(
        &rows.panel(),
        &adoptions,
        SyntheticPanelMethod::Control,
        10,
        &ExecutionContext::for_tests(1),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("next attempt"), "{error}");
}

#[test]
fn unconverged_simplex_is_an_optimizer_failure_not_support_or_interval() {
    let target = [1.0, 0.00008];
    let donors = vec![vec![1.0, 0.0], vec![1.0, 0.0001]];
    let fit = super::fit_simplex(&target, &donors, None).unwrap();
    assert!((fit[1] - 0.8).abs() < 1e-8);
    for offset in [0.0, 1e12] {
        let shifted =
            super::fit_simplex(&[offset + 0.25], &[vec![offset], vec![offset + 1.0]], None)
                .unwrap();
        assert!((shifted[1] - 0.25).abs() < 1e-9);
    }
    let tiny = f64::from_bits(1);
    let subnormal = super::fit_simplex(&[tiny], &[vec![0.0], vec![2.0 * tiny]], None).unwrap();
    assert!((subnormal[1] - 0.5).abs() < 1e-9);

    assert!(super::fit_simplex_with_limit(&target, &donors, None, 0).is_err());
    let panel = super::Panel {
        labels: vec!["treated".into(), "a".into(), "b".into()],
        periods: vec![0, 1, 2],
        values: vec![vec![1.0, 0.00008, 1.0], vec![1.0, 0.0, 1.0], vec![1.0, 0.0001, 1.0]],
    };
    assert_eq!(
        super::rollout_effects(&panel, &[(0, 1)], &[1, 2], SyntheticPanelMethod::Control),
        Err(super::RolloutFailure::Support)
    );
    let mut invalid = panel.clone();
    invalid.values[0][0] = f64::INFINITY;
    assert_eq!(
        super::rollout_effects(&invalid, &[(0, 2)], &[1, 2], SyntheticPanelMethod::Control),
        Err(super::RolloutFailure::Optimizer)
    );
    assert!(super::invert(0.0, 1.0, |_| Err(super::optimizer_refusal())).is_err());
    let ordinary =
        super::fit_simplex(&[1.0, 0.5], &[vec![1.0, 0.0], vec![1.0, 1.0]], None).unwrap();
    assert!((ordinary[0] - 0.5).abs() < 1e-12 && (ordinary[1] - 0.5).abs() < 1e-12);
}
