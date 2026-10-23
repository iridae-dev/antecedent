//! Known-truth recovery of the heterogeneity-robust rollout estimators.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    DidAggregation, DidBasePeriod, DidComparisonGroup, DidNuisanceMethod, ExecutionContext,
    RolloutDidEstimator,
};

use super::{RolloutDidSpec, RolloutPanel, RolloutResampling, estimate_rollout_did};
use crate::rollout_resampling::{DesignEffectKind, UnitWeighting};
use crate::splitmix::FixtureRng;

/// A standard normal draw by Box–Muller.
fn normal(rng: &mut FixtureRng) -> f64 {
    let u = rng.uniform().max(1e-300);
    let v = rng.uniform();
    (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
}

/// A simulated long panel.
struct Simulated {
    outcome: Vec<f64>,
    kept: Vec<bool>,
    units: Vec<Arc<str>>,
    clusters: Vec<Arc<str>>,
    periods: Vec<i64>,
    cohorts: Vec<i64>,
    covariates: Vec<Vec<f64>>,
}

impl Simulated {
    fn panel(&self) -> RolloutPanel<'_> {
        RolloutPanel {
            outcome: &self.outcome,
            kept: &self.kept,
            units: &self.units,
            clusters: &self.clusters,
            periods: &self.periods,
            cohorts: &self.cohorts,
            covariates: &self.covariates,
        }
    }
}

/// Cohorts 0, 2 and 4 of `per_cohort` units over five periods; the effect is `3·e` at event
/// time `e ≥ 0`, so it grows with exposure and differs by cohort at each calendar period.
/// The simple aggregate is `(0+3+6+9 + 0+3)/6 = 3.5` and the static two-way coefficient is
/// about `0.75`.
fn staggered(seed: u64, per_cohort: usize, drop_share: f64) -> Simulated {
    let mut rng = FixtureRng::new(seed);
    let mut sim = Simulated {
        outcome: Vec::new(),
        kept: Vec::new(),
        units: Vec::new(),
        clusters: Vec::new(),
        periods: Vec::new(),
        cohorts: Vec::new(),
        covariates: Vec::new(),
    };
    for (c, cohort) in [0_i64, 2, 4].into_iter().enumerate() {
        for i in 0..per_cohort {
            let label: Arc<str> = Arc::from(format!("u{c}_{i:04}"));
            let alpha = normal(&mut rng);
            for period in 1..=5_i64 {
                let effect = if cohort > 0 && period >= cohort {
                    3.0 * (period - cohort) as f64
                } else {
                    0.0
                };
                sim.outcome.push(alpha + 0.5 * period as f64 + effect + normal(&mut rng));
                sim.kept.push(rng.uniform() >= drop_share);
                sim.units.push(Arc::clone(&label));
                sim.clusters.push(Arc::clone(&label));
                sim.periods.push(period);
                sim.cohorts.push(cohort);
            }
        }
    }
    sim
}

fn spec(estimator: RolloutDidEstimator, comparison: DidComparisonGroup) -> RolloutDidSpec {
    RolloutDidSpec {
        estimator,
        comparison,
        nuisance: DidNuisanceMethod::DoublyRobust,
        base_period: DidBasePeriod::Varying,
        aggregation: DidAggregation::Simple,
        anticipation: 0,
    }
}

#[test]
fn every_robust_estimator_recovers_the_heterogeneous_effect_where_two_way_fixed_effects_fails() {
    let sim = staggered(11, 120, 0.0);
    let ctx = ExecutionContext::for_tests(3);
    for (estimator, comparison) in [
        (RolloutDidEstimator::CallawaySantAnna, DidComparisonGroup::NeverTreated),
        (RolloutDidEstimator::CallawaySantAnna, DidComparisonGroup::NotYetTreated),
        (RolloutDidEstimator::SunAbraham, DidComparisonGroup::NeverTreated),
        (RolloutDidEstimator::Imputation, DidComparisonGroup::NeverTreated),
    ] {
        let fit =
            estimate_rollout_did(&sim.panel(), &spec(estimator, comparison), None, &ctx).unwrap();
        assert!(
            (fit.overall.estimate - 3.5).abs() < 0.35,
            "{estimator:?} {comparison:?}: {}",
            fit.overall.estimate
        );
        let twfe = fit.twfe_static.unwrap();
        assert!((twfe - 3.5).abs() > 2.0, "two-way fixed effects {twfe} should be biased");
        assert!(fit.balanced);
        assert_eq!(fit.treated_units, 240);
    }
}

#[test]
fn the_dynamic_profile_recovers_the_exposure_path() {
    let sim = staggered(12, 120, 0.0);
    let ctx = ExecutionContext::for_tests(3);
    let mut declared = spec(RolloutDidEstimator::Imputation, DidComparisonGroup::NeverTreated);
    declared.aggregation = DidAggregation::Dynamic;
    let fit = estimate_rollout_did(&sim.panel(), &declared, None, &ctx).unwrap();
    for effect in &fit.aggregated {
        assert_eq!(effect.kind, DesignEffectKind::EventTime);
        let event_time = effect.event_time.unwrap();
        assert!((effect.estimate - 3.0 * event_time as f64).abs() < 0.45, "{effect:?}");
    }
    // Mean of the exposure path 0, 3, 6, 9.
    assert!((fit.overall.estimate - 4.5).abs() < 0.4);
    // Leads are pre-trend falsification contrasts near zero.
    assert!(fit.leads.iter().all(|lead| lead.estimate.abs() < 0.5));
}

#[test]
fn an_unbalanced_panel_is_read_by_the_two_by_two_and_refused_by_sun_abraham() {
    let sim = staggered(13, 150, 0.15);
    let ctx = ExecutionContext::for_tests(3);
    let fit = estimate_rollout_did(
        &sim.panel(),
        &spec(RolloutDidEstimator::CallawaySantAnna, DidComparisonGroup::NotYetTreated),
        None,
        &ctx,
    )
    .unwrap();
    assert!(!fit.balanced);
    assert!((fit.overall.estimate - 3.5).abs() < 0.4, "{}", fit.overall.estimate);
    let imputed = estimate_rollout_did(
        &sim.panel(),
        &spec(RolloutDidEstimator::Imputation, DidComparisonGroup::NeverTreated),
        None,
        &ctx,
    )
    .unwrap();
    assert!((imputed.overall.estimate - 3.5).abs() < 0.4, "{}", imputed.overall.estimate);
    let refused = estimate_rollout_did(
        &sim.panel(),
        &spec(RolloutDidEstimator::SunAbraham, DidComparisonGroup::NeverTreated),
        None,
        &ctx,
    )
    .unwrap_err()
    .to_string();
    assert!(refused.contains("balanced panel") && refused.contains("next attempt"), "{refused}");
}

/// Two periods; treatment probability rises with `x` and the untreated trend is `x`, so the
/// unadjusted comparison is biased and every covariate-adjusted 2×2 recovers the effect 2.
fn conditional_trends(seed: u64, units: usize) -> Simulated {
    let mut rng = FixtureRng::new(seed);
    let mut sim = Simulated {
        outcome: Vec::new(),
        kept: Vec::new(),
        units: Vec::new(),
        clusters: Vec::new(),
        periods: Vec::new(),
        cohorts: Vec::new(),
        covariates: vec![Vec::new()],
    };
    for i in 0..units {
        let x = normal(&mut rng);
        let treated = rng.uniform() < 1.0 / (1.0 + (-x).exp());
        let alpha = normal(&mut rng) + x;
        let label: Arc<str> = Arc::from(format!("unit{i:05}"));
        for period in 1..=2_i64 {
            let trend = if period == 2 { 1.0 + x } else { 0.0 };
            let effect = if treated && period == 2 { 2.0 } else { 0.0 };
            sim.outcome.push(alpha + trend + effect + 0.5 * normal(&mut rng));
            sim.kept.push(true);
            sim.units.push(Arc::clone(&label));
            sim.clusters.push(Arc::clone(&label));
            sim.periods.push(period);
            sim.cohorts.push(if treated { 2 } else { 0 });
            sim.covariates[0].push(x);
        }
    }
    sim
}

#[test]
fn covariate_adjusted_two_period_comparisons_recover_the_effect_under_conditional_trends() {
    let sim = conditional_trends(21, 3000);
    let ctx = ExecutionContext::for_tests(3);
    for nuisance in [
        DidNuisanceMethod::OutcomeRegression,
        DidNuisanceMethod::InverseProbabilityWeighting,
        DidNuisanceMethod::DoublyRobust,
    ] {
        let mut declared =
            spec(RolloutDidEstimator::CallawaySantAnna, DidComparisonGroup::NeverTreated);
        declared.nuisance = nuisance;
        let fit = estimate_rollout_did(&sim.panel(), &declared, None, &ctx).unwrap();
        assert!((fit.overall.estimate - 2.0).abs() < 0.2, "{nuisance:?}: {}", fit.overall.estimate);
    }
    let unadjusted = Simulated { covariates: Vec::new(), ..conditional_trends(21, 3000) };
    let fit = estimate_rollout_did(
        &unadjusted.panel(),
        &spec(RolloutDidEstimator::CallawaySantAnna, DidComparisonGroup::NeverTreated),
        None,
        &ctx,
    )
    .unwrap();
    assert!((fit.overall.estimate - 2.0).abs() > 0.4, "{}", fit.overall.estimate);
}

#[test]
fn covariates_are_refused_outside_the_two_by_two_estimator() {
    let sim = conditional_trends(22, 200);
    let ctx = ExecutionContext::for_tests(3);
    let error = estimate_rollout_did(
        &sim.panel(),
        &spec(RolloutDidEstimator::Imputation, DidComparisonGroup::NeverTreated),
        None,
        &ctx,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("next attempt"), "{error}");
}

#[test]
fn anticipation_moves_the_base_period_and_recovers_an_anticipated_effect() {
    // Units respond one period before adoption: the anticipation effect is 1 at e = -1.
    let mut sim = staggered(14, 120, 0.0);
    for row in 0..sim.outcome.len() {
        if sim.cohorts[row] > 0 && sim.periods[row] == sim.cohorts[row] - 1 {
            sim.outcome[row] += 1.0;
        }
    }
    let ctx = ExecutionContext::for_tests(3);
    let mut declared =
        spec(RolloutDidEstimator::CallawaySantAnna, DidComparisonGroup::NeverTreated);
    let naive = estimate_rollout_did(&sim.panel(), &declared, None, &ctx).unwrap();
    declared.anticipation = 1;
    let fit = estimate_rollout_did(&sim.panel(), &declared, None, &ctx).unwrap();
    let cell = |fit: &super::RolloutDidFit, group: i64, period: i64| {
        fit.group_time
            .iter()
            .find(|effect| effect.group == Some(group) && effect.period == Some(period))
            .map(|effect| effect.estimate)
            .unwrap()
    };
    // Ignoring anticipation compares adoption with an already-responding base period.
    assert!((cell(&naive, 4, 4) + 1.0).abs() < 0.45, "{}", cell(&naive, 4, 4));
    assert!(cell(&fit, 4, 4).abs() < 0.45, "{}", cell(&fit, 4, 4));
    // The anticipation window is reported, not aggregated.
    assert!((cell(&fit, 4, 3) - 1.0).abs() < 0.45, "{}", cell(&fit, 4, 3));
    // Only the late cohort keeps a period before the anticipation window.
    assert!(fit.dropped_cohorts.contains(&2));
    // Late-cohort cells are 0 and 3 relative to period 2.
    assert!((fit.overall.estimate - 1.5).abs() < 0.4, "{}", fit.overall.estimate);
}

#[test]
fn the_whole_procedure_resample_is_deterministic_across_threads_and_bands_dominate() {
    let sim = staggered(15, 60, 0.0);
    let mut declared =
        spec(RolloutDidEstimator::CallawaySantAnna, DidComparisonGroup::NotYetTreated);
    declared.aggregation = DidAggregation::Dynamic;
    for weighting in [UnitWeighting::Resample, UnitWeighting::Multiplier] {
        let resampling = RolloutResampling { weighting, replicates: 199 };
        let serial = estimate_rollout_did(
            &sim.panel(),
            &declared,
            Some(resampling),
            &ExecutionContext::for_tests(9),
        )
        .unwrap();
        let parallel = estimate_rollout_did(
            &sim.panel(),
            &declared,
            Some(resampling),
            &ExecutionContext::production(9, 4),
        )
        .unwrap();
        assert_eq!(serial, parallel);
        let record = serial.resampling.unwrap();
        assert!(record.replicates_ok >= 190, "{record:?}");
        for effect in &serial.aggregated {
            let (interval, band) = (effect.interval.unwrap(), effect.band.unwrap());
            assert!(band[0] <= interval[0] && band[1] >= interval[1]);
        }
        assert!(serial.pretrend_max_abs_t.is_some());
    }
}

#[test]
fn whole_procedure_intervals_cover_the_simulated_truth_in_most_samples() {
    let replicates = 40;
    let mut covered = 0;
    for seed in 0..replicates {
        let sim = staggered(1_000 + seed, 50, 0.0);
        let fit = estimate_rollout_did(
            &sim.panel(),
            &spec(RolloutDidEstimator::CallawaySantAnna, DidComparisonGroup::NeverTreated),
            Some(RolloutResampling { weighting: UnitWeighting::Multiplier, replicates: 149 }),
            &ExecutionContext::for_tests(seed),
        )
        .unwrap();
        let interval = fit.overall.interval.unwrap();
        covered += usize::from(interval[0] <= 3.5 && 3.5 <= interval[1]);
    }
    // Nominal 95%; a sanity floor, not a coverage record.
    assert!(covered >= 33, "covered {covered} of {replicates}");
}
