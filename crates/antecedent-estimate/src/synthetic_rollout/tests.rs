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
    {
        let method = SyntheticPanelMethod::Control;
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

/// With two donors the simplex fit is one weight in `[0, 1]` with a closed form, so the
/// residuals and the rank share of the tested positions are computed here.
fn two_donor_p(target: &[f64], first: &[f64], second: &[f64], post: usize) -> f64 {
    let n = target.len();
    let (mut cross, mut square) = (0.0, 0.0);
    for t in 0..n {
        let gap = first[t] - second[t];
        cross += (target[t] - second[t]) * gap;
        square += gap * gap;
    }
    let weight = (cross / square).clamp(0.0, 1.0);
    let residual: Vec<f64> = (0..n)
        .map(|t| (target[t] - weight * first[t] - (1.0 - weight) * second[t]).abs())
        .collect();
    let statistic =
        |shift: usize| -> f64 { (n - post..n).map(|at| residual[(at + n - shift) % n]).sum() };
    let observed = statistic(0);
    (0..n).filter(|shift| statistic(*shift) >= observed - 1e-12).count() as f64 / n as f64
}

/// The conformal p-value against the two-donor closed form, for one tested period and for a
/// block of two; and each end of a published set sits where that closed-form p-value
/// crosses 5%.
#[test]
fn conformal_p_value_and_set_match_the_two_donor_closed_form() {
    let mut rng = FixtureRng::new(41);
    let (pre, post) = (24_usize, 2_usize);
    let first: Vec<f64> = (0..pre + post).map(|_| 1.0 + normal(&mut rng)).collect();
    let second: Vec<f64> = (0..pre + post).map(|_| 3.0 + normal(&mut rng)).collect();
    let target: Vec<f64> = (0..pre + post)
        .map(|t| {
            0.7 * first[t]
                + 0.3 * second[t]
                + 0.4 * normal(&mut rng)
                + if t >= pre { 2.0 } else { 0.0 }
        })
        .collect();
    let donors = [first.clone(), second.clone()];
    for tested in [1, 2] {
        let got = super::conformal_p(&target, &donors, tested, &[0.5, 0.5]).unwrap();
        let want = two_donor_p(&target, &first, &second, tested);
        assert!((got - want).abs() < 1e-9, "block {tested}: {got} against {want}");
    }

    let labels: [Arc<str>; 3] = [Arc::from("a"), Arc::from("b"), Arc::from("treated")];
    let mut rows = Rows { outcome: Vec::new(), units: Vec::new(), periods: Vec::new() };
    for t in 0..pre + post {
        for (label, value) in labels.iter().zip([first[t], second[t], target[t]]) {
            rows.outcome.push(value);
            rows.units.push(Arc::clone(label));
            rows.periods.push(i64::try_from(t).unwrap() + 1);
        }
    }
    let fit = conformal_synthetic_control(
        &rows.panel(),
        "treated",
        i64::try_from(pre).unwrap() + 1,
        &ExecutionContext::for_tests(1),
    )
    .unwrap();
    // The first post period is tested on the pre-periods and itself, its value moved by theta.
    let pick = |series: &[f64]| -> Vec<f64> {
        series[..pre].iter().copied().chain(std::iter::once(series[pre])).collect()
    };
    let p_at = |theta: f64| -> f64 {
        let mut adjusted = pick(&target);
        adjusted[pre] -= theta;
        two_donor_p(&adjusted, &pick(&first), &pick(&second), 1)
    };
    let [lower, upper] = fit.periods[0].interval.unwrap();
    assert!(lower < upper);
    let step = 1e-4;
    assert!(p_at(lower + step) > 0.05 && p_at(upper - step) > 0.05);
    assert!(p_at(lower - step) <= 0.05 && p_at(upper + step) <= 0.05);
}

/// One adopting unit and three never-treated donors: each placebo refit assigns the adoption
/// to one donor and fits it on the other two, a one-weight problem with a closed form. The
/// placebo standard error is the spread of those three effects and the interval is its
/// normal interval.
#[test]
fn placebo_interval_is_the_normal_interval_of_the_enumerated_placebo_spread() {
    let mut rng = FixtureRng::new(77);
    let (periods, adoption) = (12_usize, 9_usize);
    let series: Vec<Vec<f64>> = (0..4)
        .map(|unit| {
            (0..periods)
                .map(|t| {
                    f64::from(unit) * 0.8
                        + 0.3 * t as f64
                        + normal(&mut rng)
                        + if unit == 3 && t + 1 >= adoption { 2.0 } else { 0.0 }
                })
                .collect()
        })
        .collect();
    let labels: [Arc<str>; 4] = [Arc::from("d0"), Arc::from("d1"), Arc::from("d2"), Arc::from("t")];
    let mut rows = Rows { outcome: Vec::new(), units: Vec::new(), periods: Vec::new() };
    for t in 0..periods {
        for (label, unit) in labels.iter().zip(&series) {
            rows.outcome.push(unit[t]);
            rows.units.push(Arc::clone(label));
            rows.periods.push(i64::try_from(t).unwrap() + 1);
        }
    }
    let replicates = 3_000;
    let fit = multi_unit_synthetic(
        &rows.panel(),
        &[(Arc::clone(&labels[3]), i64::try_from(adoption).unwrap())],
        SyntheticPanelMethod::Control,
        replicates,
        &ExecutionContext::for_tests(9),
    )
    .unwrap();
    assert_eq!(fit.placebo.unwrap().replicates_ok, replicates);

    let pre = adoption - 1;
    let post_mean = |unit: usize| series[unit][pre..].iter().sum::<f64>() / (periods - pre) as f64;
    let placebo: Vec<f64> = (0..3)
        .map(|unit| {
            let (a, b) = match unit {
                0 => (1, 2),
                1 => (0, 2),
                _ => (0, 1),
            };
            let (mut cross, mut square) = (0.0, 0.0);
            for t in 0..pre {
                let gap = series[a][t] - series[b][t];
                cross += (series[unit][t] - series[b][t]) * gap;
                square += gap * gap;
            }
            let weight = (cross / square).clamp(0.0, 1.0);
            post_mean(unit) - weight * post_mean(a) - (1.0 - weight) * post_mean(b)
        })
        .collect();
    let center = placebo.iter().sum::<f64>() / 3.0;
    let spread = (placebo.iter().map(|effect| (effect - center).powi(2)).sum::<f64>() / 3.0).sqrt();
    let se = fit.overall.standard_error.unwrap();
    // Each donor is drawn about a third of the time, so the replicate spread is the
    // enumerated one up to the draw frequencies.
    assert!((se / spread - 1.0).abs() < 0.05, "placebo se {se} against {spread}");
    let [lower, upper] = fit.overall.interval.unwrap();
    let z = 1.959_963_984_540_054;
    assert!((lower - (fit.overall.estimate - z * se)).abs() < 1e-12);
    assert!((upper - (fit.overall.estimate + z * se)).abs() < 1e-12);
}

/// A set narrower than the first grid spacing is found by the finer search, and a test that
/// accepts nothing leaves the set empty.
#[test]
fn a_set_narrower_than_the_first_grid_spacing_is_found() {
    let narrow = super::invert(0.0, 1.0, |theta| Ok((0.26..=0.36).contains(&theta)))
        .unwrap()
        .expect("the narrow set is found");
    assert!((narrow[0] - 0.26).abs() < 1e-6 && (narrow[1] - 0.36).abs() < 1e-6, "{narrow:?}");
    assert_eq!(super::invert(0.0, 1.0, |_| Ok(false)).unwrap(), None);
    assert_eq!(super::invert(0.0, 1.0, |_| Ok(true)).unwrap(), None);
}

#[test]
fn single_interval_selections_equal_the_matching_entries_of_the_full_fit() {
    use super::{ConformalIntervals, ROLLOUT_LEVEL, conformal_synthetic_control_for};
    let rows = exchangeable_panel(900, 40, 38, 1.0);
    let ctx = ExecutionContext::for_tests(1);
    let full = conformal_synthetic_control(&rows.panel(), "treated", 38, &ctx).unwrap();
    assert_eq!(full.periods.len(), 3);
    let same = |left: Option<[f64; 2]>, right: Option<[f64; 2]>| match (left, right) {
        (Some(a), Some(b)) => a[0].to_bits() == b[0].to_bits() && a[1].to_bits() == b[1].to_bits(),
        (None, None) => true,
        _ => false,
    };
    let select = |intervals| {
        conformal_synthetic_control_for(
            &rows.panel(),
            "treated",
            38,
            ROLLOUT_LEVEL,
            intervals,
            &ctx,
        )
        .unwrap()
    };
    for (index, expected) in full.periods.iter().enumerate() {
        let single = select(ConformalIntervals::Period(expected.period.unwrap()));
        assert!(same(single.periods[index].interval, expected.interval), "period {index}");
        assert_eq!(single.overall.estimate.to_bits(), full.overall.estimate.to_bits());
        assert!(single.overall.interval.is_none());
        for (other, entry) in single.periods.iter().enumerate() {
            assert_eq!(entry.estimate.to_bits(), full.periods[other].estimate.to_bits());
            assert_eq!(entry.interval.is_some(), other == index && expected.interval.is_some());
        }
        assert_eq!(single.donor_weights, full.donor_weights);
        assert_eq!(single.pre_treatment_rmse.to_bits(), full.pre_treatment_rmse.to_bits());
    }
    let overall = select(ConformalIntervals::Overall);
    assert!(same(overall.overall.interval, full.overall.interval));
    assert!(overall.periods.iter().all(|entry| entry.interval.is_none()));
    assert_eq!(select(ConformalIntervals::All), full);
}
