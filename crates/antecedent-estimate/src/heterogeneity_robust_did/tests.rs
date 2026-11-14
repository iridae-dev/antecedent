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
fn an_unbalanced_panel_is_read_as_observed_by_every_robust_estimator() {
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
    let interacted = estimate_rollout_did(
        &sim.panel(),
        &spec(RolloutDidEstimator::SunAbraham, DidComparisonGroup::NeverTreated),
        None,
        &ctx,
    )
    .unwrap();
    assert!(!interacted.balanced);
    assert!((interacted.overall.estimate - 3.5).abs() < 0.4, "{}", interacted.overall.estimate);
}

#[test]
fn the_attrition_slope_is_zero_when_nothing_is_missing_and_the_missing_share_otherwise() {
    let ctx = ExecutionContext::for_tests(3);
    let balanced = staggered(18, 100, 0.0);
    let sun_abraham = spec(RolloutDidEstimator::SunAbraham, DidComparisonGroup::NeverTreated);
    let fit = estimate_rollout_did(&balanced.panel(), &sun_abraham, None, &ctx).unwrap();
    assert!(fit.attrition_slope.abs() < 1e-12, "{}", fit.attrition_slope);
    // Each unit-period is observed with probability 0.8, so about a fifth of the cohort is
    // unobserved at a cell and the slope is near that share.
    let thinned = staggered(18, 400, 0.2);
    for estimator in [
        RolloutDidEstimator::CallawaySantAnna,
        RolloutDidEstimator::SunAbraham,
        RolloutDidEstimator::Imputation,
    ] {
        let declared = spec(estimator, DidComparisonGroup::NeverTreated);
        let fit = estimate_rollout_did(&thinned.panel(), &declared, None, &ctx).unwrap();
        assert!(
            fit.attrition_slope > 0.1 && fit.attrition_slope < 0.45,
            "{estimator:?} {}",
            fit.attrition_slope
        );
    }
}

#[test]
fn sun_abraham_on_an_unbalanced_panel_keeps_the_cohort_specific_effects_and_reports_support() {
    let sim = staggered(14, 200, 0.2);
    let ctx = ExecutionContext::for_tests(3);
    let fit = estimate_rollout_did(
        &sim.panel(),
        &spec(RolloutDidEstimator::SunAbraham, DidComparisonGroup::NeverTreated),
        None,
        &ctx,
    )
    .unwrap();
    assert!(!fit.balanced);
    // Each cohort's own exposure path 3·e is recovered, not only the pooled average.
    assert!(!fit.group_time.is_empty());
    for effect in &fit.group_time {
        let event = effect.event_time.unwrap();
        assert!((effect.estimate - 3.0 * event as f64).abs() < 0.8, "{effect:?}");
        assert!(effect.treated > 0 && effect.comparison > 0);
    }
    assert!(fit.leads.iter().all(|lead| lead.estimate.abs() < 0.6));
    assert_eq!(fit.cohort_support.len(), 2);
    for support in &fit.cohort_support {
        assert_eq!(support.units, 200);
        assert!(support.rows < 5 * support.units, "{support:?}");
        assert!(support.units_at_base < support.units, "{support:?}");
        assert!(support.post_cells > 0);
    }
    // Every observed row is either used or counted as dropped.
    let observed: usize = sim.kept.iter().filter(|kept| **kept).count();
    assert_eq!(fit.rows, observed);
}

/// The frozen unbalanced panel and the full-dummy numpy regression computed on it.
fn interacted_reference() -> (Simulated, serde_json::Value) {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../conformance/estimate/interacted_did_unbalanced_panel/expected.json"
    ))
    .unwrap();
    let column = |name: &str| fixture["rows"][name].as_array().unwrap().clone();
    let labels: Vec<Arc<str>> =
        column("unit").iter().map(|value| Arc::from(value.as_str().unwrap())).collect();
    let sim = Simulated {
        outcome: column("outcome").iter().map(|value| value.as_f64().unwrap()).collect(),
        kept: vec![true; labels.len()],
        clusters: labels.clone(),
        units: labels,
        periods: column("period").iter().map(|value| value.as_i64().unwrap()).collect(),
        cohorts: column("cohort").iter().map(|value| value.as_i64().unwrap()).collect(),
        covariates: Vec::new(),
    };
    (sim, fixture["reference"].clone())
}

#[test]
fn sun_abraham_on_an_unbalanced_panel_matches_the_full_dummy_regression_reference() {
    let (sim, reference) = interacted_reference();
    let ctx = ExecutionContext::for_tests(3);
    let fit = estimate_rollout_did(
        &sim.panel(),
        &spec(RolloutDidEstimator::SunAbraham, DidComparisonGroup::NeverTreated),
        None,
        &ctx,
    )
    .unwrap();
    assert!(!fit.balanced);
    for cell in reference["cells"].as_array().unwrap() {
        let (group, period) = (cell["cohort"].as_i64().unwrap(), cell["period"].as_i64().unwrap());
        let expected = cell["estimate"].as_f64().unwrap();
        if period >= group {
            let effect = fit
                .group_time
                .iter()
                .find(|effect| effect.group == Some(group) && effect.period == Some(period))
                .unwrap_or_else(|| panic!("cell ({group}, {period}) missing"));
            assert!((effect.estimate - expected).abs() < 1e-8, "{group} {period}: {effect:?}");
            assert_eq!(effect.treated as u64, cell["treated_units"].as_u64().unwrap());
            assert_eq!(effect.comparison as u64, cell["comparison_units"].as_u64().unwrap());
        }
    }
    // The cells at the period without any comparison unit are not estimated.
    let unsupported = reference["periods_without_comparison"].as_array().unwrap();
    for period in unsupported.iter().map(|value| value.as_i64().unwrap()) {
        assert!(fit.group_time.iter().all(|effect| effect.period != Some(period)));
    }
    for lead in &fit.leads {
        let expected =
            reference["leads_by_event"][lead.event_time.unwrap().to_string()].as_f64().unwrap();
        assert!((lead.estimate - expected).abs() < 1e-8, "{lead:?}");
    }
    assert!((fit.overall.estimate - reference["simple"].as_f64().unwrap()).abs() < 1e-8);
    // The cohort-specific effects are the declared truth up to sampling noise.
    assert!((fit.overall.estimate - reference["simple_truth"].as_f64().unwrap()).abs() < 0.1);
    let dropped: u64 = reference["dropped_rows_by_cohort"]
        .as_object()
        .unwrap()
        .values()
        .map(|n| n.as_u64().unwrap())
        .sum();
    assert_eq!(fit.dropped_rows as u64, dropped);
    assert_eq!(fit.dropped_units, 0);
}

#[test]
fn sun_abraham_on_an_unbalanced_panel_matches_the_fixest_sunab_regression() {
    // The same panel through the black-box `fixest::feols(y ~ sunab(cohort, period) | unit +
    // period)` reference. The reference's standard errors are analytic unit-clustered and are
    // recorded there only; this fit's uncertainty is a whole-procedure bootstrap that this
    // fixture does not freeze, so only the point estimates are asserted here.
    let (sim, _) = interacted_reference();
    let fixest: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../conformance/estimate/interacted_did_unbalanced_panel_fixest/expected.json"
    ))
    .unwrap();
    let ctx = ExecutionContext::for_tests(3);
    let fit = estimate_rollout_did(
        &sim.panel(),
        &spec(RolloutDidEstimator::SunAbraham, DidComparisonGroup::NeverTreated),
        None,
        &ctx,
    )
    .unwrap();
    let reference = &fixest["reference"];
    let cells = reference["cells"].as_array().unwrap();
    assert!(!cells.is_empty());
    for cell in cells {
        let (group, period) = (cell["cohort"].as_i64().unwrap(), cell["period"].as_i64().unwrap());
        let expected = cell["estimate"].as_f64().unwrap();
        if period >= group {
            let effect = fit
                .group_time
                .iter()
                .find(|effect| effect.group == Some(group) && effect.period == Some(period))
                .unwrap_or_else(|| panic!("cell ({group}, {period}) missing"));
            assert!((effect.estimate - expected).abs() < 1e-8, "{group} {period}: {effect:?}");
        }
    }
    let product = &reference["product_convention"];
    assert_eq!(fit.leads.len(), product["leads_by_event"].as_object().unwrap().len());
    for lead in &fit.leads {
        let expected = product["leads_by_event"][lead.event_time.unwrap().to_string()]["estimate"]
            .as_f64()
            .unwrap();
        assert!((lead.estimate - expected).abs() < 1e-8, "{lead:?}");
    }
    assert!((fit.overall.estimate - product["simple"]["estimate"].as_f64().unwrap()).abs() < 1e-8);
}

#[test]
fn removing_one_comparison_row_moves_the_interacted_fit_only_within_sampling_noise() {
    // A balanced panel is read through its 2×2 contrasts; removing one observed row sends the
    // same panel through the regression, which must stay within sampling noise of the 2×2 fit.
    let mut sim = staggered(15, 120, 0.0);
    let ctx = ExecutionContext::for_tests(3);
    let balanced = estimate_rollout_did(
        &sim.panel(),
        &spec(RolloutDidEstimator::SunAbraham, DidComparisonGroup::NeverTreated),
        None,
        &ctx,
    )
    .unwrap();
    sim.kept[0] = false;
    let nearly = estimate_rollout_did(
        &sim.panel(),
        &spec(RolloutDidEstimator::SunAbraham, DidComparisonGroup::NeverTreated),
        None,
        &ctx,
    )
    .unwrap();
    assert!(balanced.balanced && !nearly.balanced);
    assert!((balanced.overall.estimate - nearly.overall.estimate).abs() < 0.05);
    // Agreeing with each other is not enough: both are the simple aggregate 3.5 of the design.
    assert!((balanced.overall.estimate - 3.5).abs() < 0.35, "{}", balanced.overall.estimate);
    assert!((nearly.overall.estimate - 3.5).abs() < 0.35, "{}", nearly.overall.estimate);
}

#[test]
fn sun_abraham_refuses_when_no_comparison_unit_is_observed_in_two_periods() {
    let mut sim = staggered(16, 60, 0.0);
    // Every never-treated unit keeps one row only, so none can identify a period effect.
    for row in 0..sim.kept.len() {
        if sim.cohorts[row] == 0 && sim.periods[row] != 3 {
            sim.kept[row] = false;
        }
    }
    let ctx = ExecutionContext::for_tests(3);
    let error = estimate_rollout_did(
        &sim.panel(),
        &spec(RolloutDidEstimator::SunAbraham, DidComparisonGroup::NeverTreated),
        None,
        &ctx,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("comparison unit") && error.contains("next attempt"), "{error}");
}

#[test]
fn a_cohort_with_no_observed_base_period_is_dropped_and_reported_not_reweighted() {
    let mut sim = staggered(17, 80, 0.0);
    // Cohort 4's base period is 3; no unit of the cohort is observed there.
    for row in 0..sim.kept.len() {
        if sim.cohorts[row] == 4 && sim.periods[row] == 3 {
            sim.kept[row] = false;
        }
    }
    let ctx = ExecutionContext::for_tests(3);
    let fit = estimate_rollout_did(
        &sim.panel(),
        &spec(RolloutDidEstimator::SunAbraham, DidComparisonGroup::NeverTreated),
        None,
        &ctx,
    )
    .unwrap();
    assert_eq!(fit.dropped_cohorts, vec![4]);
    let late = fit.cohort_support.iter().find(|support| support.cohort == 4).unwrap();
    assert_eq!((late.units_at_base, late.post_cells), (0, 0));
    assert_eq!(late.dropped_rows, late.rows);
    let early = fit.cohort_support.iter().find(|support| support.cohort == 2).unwrap();
    assert_eq!(early.dropped_rows, 0);
    assert!(fit.group_time.iter().all(|effect| effect.group == Some(2)));
    // The surviving cohort keeps its own estimand: effects 0, 3, 6, 9.
    assert!((fit.overall.estimate - 4.5).abs() < 0.5, "{}", fit.overall.estimate);
    // With the only treated cohort lacking its base period there is nothing to estimate.
    for row in 0..sim.kept.len() {
        if sim.cohorts[row] == 2 && sim.periods[row] == 1 {
            sim.kept[row] = false;
        }
    }
    let error = estimate_rollout_did(
        &sim.panel(),
        &spec(RolloutDidEstimator::SunAbraham, DidComparisonGroup::NeverTreated),
        None,
        &ctx,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("next attempt"), "{error}");
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
        let treated = rng.uniform() < crate::util::sigmoid(x);
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

    declared.aggregation = DidAggregation::Dynamic;
    let dynamic = estimate_rollout_did(&sim.panel(), &declared, None, &ctx).unwrap();
    assert!(
        dynamic.aggregated.iter().all(|effect| effect.event_time.is_some_and(|event| event >= 0))
    );
    assert!(dynamic.group_time.iter().any(|effect| effect.event_time == Some(-1)));
    assert!(dynamic.leads.iter().all(|effect| effect.event_time.is_some_and(|event| event < -1)));
    assert!((dynamic.overall.estimate - 1.5).abs() < 0.4);
    let resampled = estimate_rollout_did(
        &sim.panel(),
        &declared,
        Some(RolloutResampling { weighting: UnitWeighting::Multiplier, replicates: 25 }),
        &ctx,
    )
    .unwrap();
    assert_eq!(
        resampled.aggregated.iter().map(|e| e.event_time).collect::<Vec<_>>(),
        dynamic.aggregated.iter().map(|e| e.event_time).collect::<Vec<_>>()
    );
    assert_eq!(resampled.group_time.len(), dynamic.group_time.len());
    assert!(resampled.aggregated.iter().all(|e| e.standard_error.is_some()));
    assert!(resampled.resampling.unwrap().replicates_ok >= 20);
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

#[test]
fn the_bayesian_bootstrap_posterior_recovers_the_effect_with_joint_draws() {
    let sim = staggered(16, 80, 0.1);
    let mut declared =
        spec(RolloutDidEstimator::CallawaySantAnna, DidComparisonGroup::NotYetTreated);
    declared.aggregation = DidAggregation::Dynamic;
    let serial =
        super::posterior_rollout_did(&sim.panel(), &declared, 400, &ExecutionContext::for_tests(2))
            .unwrap();
    let parallel = super::posterior_rollout_did(
        &sim.panel(),
        &declared,
        400,
        &ExecutionContext::production(2, 4),
    )
    .unwrap();
    assert_eq!(serial, parallel);
    let overall = &serial.fit.overall;
    assert!((overall.estimate - 4.5).abs() < 0.5, "{}", overall.estimate);
    let interval = overall.interval.unwrap();
    assert!(interval[0] < overall.estimate && overall.estimate < interval[1]);
    assert!(overall.standard_error.is_none());
    for effect in &serial.fit.aggregated {
        let (interval, band) = (effect.interval.unwrap(), effect.band.unwrap());
        assert!(band[0] <= interval[0] && band[1] >= interval[1]);
    }
    // Joint draws: the overall column is the mean of the post event-time columns draw by draw.
    let post: Vec<&Vec<f64>> = serial
        .columns
        .iter()
        .filter(|(label, _)| {
            label
                .strip_prefix("dynamic_")
                .and_then(|e| e.parse::<i64>().ok())
                .is_some_and(|e| e >= 0)
        })
        .map(|(_, column)| column)
        .collect();
    let draws = serial.columns[0].1.len();
    assert!(draws >= 380);
    for d in 0..draws {
        let mean = post.iter().map(|column| column[d]).sum::<f64>() / post.len() as f64;
        assert!((mean - serial.columns[0].1[d]).abs() < 1e-9);
    }
}

/// A balanced panel from per-unit cohorts, one unit per cluster: `outcome(unit, period)` and
/// one optional covariate value per unit.
fn hand_panel(
    cohorts: &[i64],
    periods: i64,
    outcome: impl Fn(usize, i64) -> f64,
    covariate: Option<&[f64]>,
) -> Simulated {
    let mut sim = Simulated {
        outcome: Vec::new(),
        kept: Vec::new(),
        units: Vec::new(),
        clusters: Vec::new(),
        periods: Vec::new(),
        cohorts: Vec::new(),
        covariates: covariate.map_or_else(Vec::new, |_| vec![Vec::new()]),
    };
    for (unit, cohort) in cohorts.iter().enumerate() {
        let label: Arc<str> = Arc::from(format!("h{unit:03}"));
        for period in 1..=periods {
            sim.outcome.push(outcome(unit, period));
            sim.kept.push(true);
            sim.units.push(Arc::clone(&label));
            sim.clusters.push(Arc::clone(&label));
            sim.periods.push(period);
            sim.cohorts.push(*cohort);
            if let Some(values) = covariate {
                sim.covariates[0].push(values[unit]);
            }
        }
    }
    sim
}

/// Mean of `change(unit)` over the units whose cohort is in `members`.
fn mean_change(cohorts: &[i64], members: &[i64], change: impl Fn(usize) -> f64) -> f64 {
    let units: Vec<usize> = (0..cohorts.len()).filter(|u| members.contains(&cohorts[*u])).collect();
    units.iter().map(|u| change(*u)).sum::<f64>() / units.len() as f64
}

/// The 2x2 contrasts and their cohort-size-weighted simple aggregate, computed here from the
/// outcome table, against the Callaway-Sant'Anna and Sun-Abraham fits; and the imputation
/// fit on an exactly additive untreated outcome, where it returns the planted effects.
#[test]
fn rollout_point_estimates_match_the_hand_computed_contrasts() {
    let cohorts = [0_i64, 0, 0, 2, 2, 3, 3];
    let y = |unit: usize, period: i64| -> f64 {
        let cohort = cohorts[unit];
        let wobble = ((7 * i64::try_from(unit).unwrap() + 3 * period) % 5) as f64 * 0.1;
        let effect =
            if cohort > 0 && period >= cohort { 1.0 + (period - cohort) as f64 } else { 0.0 };
        unit as f64 * 0.3 + 0.5 * period as f64 + wobble + effect
    };
    let contrast = |group: i64, period: i64, base: i64, controls: &[i64]| -> f64 {
        mean_change(&cohorts, &[group], |u| y(u, period) - y(u, base))
            - mean_change(&cohorts, controls, |u| y(u, period) - y(u, base))
    };
    let sim = hand_panel(&cohorts, 3, y, None);
    let ctx = ExecutionContext::for_tests(1);
    // Both adopting cohorts hold two units, so the simple aggregate is the plain cell mean.
    let never = [contrast(2, 2, 1, &[0]), contrast(2, 3, 1, &[0]), contrast(3, 3, 2, &[0])];
    let not_yet = [contrast(2, 2, 1, &[0, 3]), never[1], never[2]];
    for (estimator, comparison, cells) in [
        (RolloutDidEstimator::CallawaySantAnna, DidComparisonGroup::NeverTreated, never),
        (RolloutDidEstimator::CallawaySantAnna, DidComparisonGroup::NotYetTreated, not_yet),
        (RolloutDidEstimator::SunAbraham, DidComparisonGroup::NeverTreated, never),
    ] {
        let fit =
            estimate_rollout_did(&sim.panel(), &spec(estimator, comparison), None, &ctx).unwrap();
        for ((group, period), want) in [(2, 2), (2, 3), (3, 3)].into_iter().zip(cells) {
            let cell = fit
                .group_time
                .iter()
                .find(|cell| cell.group == Some(group) && cell.period == Some(period))
                .unwrap();
            assert!((cell.estimate - want).abs() < 1e-10, "{estimator:?} cell {group},{period}");
        }
        let aggregate = cells.iter().sum::<f64>() / 3.0;
        assert!((fit.overall.estimate - aggregate).abs() < 1e-10, "{estimator:?} aggregate");
        assert_eq!(fit.comparison_propensity_max, None);
    }
    // Additive untreated outcome: the imputed effects are the planted 1, 2 and 1.
    let additive = |unit: usize, period: i64| -> f64 {
        let cohort = cohorts[unit];
        let effect =
            if cohort > 0 && period >= cohort { 1.0 + (period - cohort) as f64 } else { 0.0 };
        unit as f64 * 0.3 + 0.5 * (period * period) as f64 + effect
    };
    let fit = estimate_rollout_did(
        &hand_panel(&cohorts, 3, additive, None).panel(),
        &spec(RolloutDidEstimator::Imputation, DidComparisonGroup::NotYetTreated),
        None,
        &ctx,
    )
    .unwrap();
    assert!((fit.overall.estimate - 4.0 / 3.0).abs() < 1e-8);
}

/// The two-point multiplier has mean, variance and third central moment one, and the
/// degrees-of-freedom scale is `sqrt(K / (K - p - 1))` at the Kish count of the cluster totals.
#[test]
#[allow(clippy::float_cmp, reason = "the multipliers and the covariate levels are exact constants")]
fn multiplier_law_and_degrees_of_freedom_scale_match_their_closed_forms() {
    let (high, low, p) =
        (super::MULTIPLIER_HIGH, super::MULTIPLIER_LOW, super::MULTIPLIER_HIGH_PROBABILITY);
    let moment = |power: i32| p * (high - 1.0).powi(power) + (1.0 - p) * (low - 1.0).powi(power);
    assert!(low > 0.0 && (p * high + (1.0 - p) * low - 1.0).abs() < 1e-12);
    assert!((moment(2) - 1.0).abs() < 1e-12 && (moment(3) - 1.0).abs() < 1e-12);
    let mut rng = antecedent_core::CausalRng::from_seed(3);
    let drawn = super::moment_matched_multipliers(40_000, &mut rng);
    assert!(drawn.iter().all(|weight| *weight == high || *weight == low));
    assert!((drawn.iter().sum::<f64>() / 40_000.0 - 1.0).abs() < 0.02);

    let singletons: Vec<usize> = (0..10).collect();
    let equal = super::comparison_scale(&singletons, None, 1);
    assert!((equal - (10.0_f64 / 8.0).sqrt()).abs() < 1e-12);
    // Odds 1, 1, 2 on three clusters: K = 16 / 6, one coefficient fitted.
    let weighted = super::comparison_scale(&[0, 1, 2], Some(&[1.0, 1.0, 2.0]), 0);
    assert!((weighted - 1.6_f64.sqrt()).abs() < 1e-12);
    // Two units of one cluster count once: totals 2 and 1, K = 1.8, denominator held at one.
    let clustered = super::comparison_scale(&[0, 0, 1], None, 0);
    assert!((clustered - 1.8_f64.sqrt()).abs() < 1e-12);
    // The reference quantile falls to the normal one as clusters and replicates grow.
    let normal = antecedent_stats::NORMAL_Q975;
    let (few, many) = (super::pointwise_critical(11, 199), super::pointwise_critical(101, 199));
    assert!(few > many && many > normal);
    assert!((super::pointwise_critical(1_000_001, 4_000_000) - normal).abs() < 1e-4);
}

/// With one binary covariate the outcome regression and the logistic propensity are both
/// saturated, so each adjusted 2x2 is the treated mean change minus the comparison mean
/// change standardized to the treated covariate shares, and the largest fitted comparison
/// propensity is the largest treated share of a covariate cell.
#[test]
#[allow(clippy::float_cmp, reason = "the multipliers and the covariate levels are exact constants")]
fn covariate_adjusted_contrasts_match_the_closed_form_on_a_binary_covariate() {
    // Covariate 0: three treated, four comparison; covariate 1: three treated, two comparison.
    let cohorts = [2_i64, 2, 2, 0, 0, 0, 0, 2, 2, 2, 0, 0];
    let x = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let change = |unit: usize| -> f64 {
        0.4 + 0.9 * x[unit]
            + ((5 * unit + 2) % 7) as f64 * 0.11
            + if cohorts[unit] == 2 { 1.5 } else { 0.0 }
    };
    let y = |unit: usize, period: i64| -> f64 {
        unit as f64 * 0.2 + if period == 2 { change(unit) } else { 0.0 }
    };
    let cell_mean = |cohort: i64, level: f64| -> f64 {
        let units: Vec<usize> =
            (0..cohorts.len()).filter(|u| cohorts[*u] == cohort && x[*u] == level).collect();
        units.iter().map(|u| change(*u)).sum::<f64>() / units.len() as f64
    };
    let treated = mean_change(&cohorts, &[2], change);
    let standardized = 0.5 * cell_mean(0, 0.0) + 0.5 * cell_mean(0, 1.0);
    let want = treated - standardized;
    let sim = hand_panel(&cohorts, 2, y, Some(&x));
    for nuisance in [
        DidNuisanceMethod::OutcomeRegression,
        DidNuisanceMethod::InverseProbabilityWeighting,
        DidNuisanceMethod::DoublyRobust,
    ] {
        let declared = RolloutDidSpec {
            nuisance,
            ..spec(RolloutDidEstimator::CallawaySantAnna, DidComparisonGroup::NeverTreated)
        };
        let fit =
            estimate_rollout_did(&sim.panel(), &declared, None, &ExecutionContext::for_tests(2))
                .unwrap();
        assert!((fit.overall.estimate - want).abs() < 1e-6, "{nuisance:?}");
        let overlap = fit.comparison_propensity_max.unwrap();
        assert!((overlap - 0.6).abs() < 1e-6, "{nuisance:?} overlap {overlap}");
    }
}

/// One cohort against never-treated units: the flat-Dirichlet posterior of a difference of two
/// weighted means has the plug-in contrast as its mean and, for each arm of `m` units, the
/// variance `sum (d - mean)^2 / (m (m + 1))`.
#[test]
fn bayesian_bootstrap_posterior_matches_the_dirichlet_moments_of_a_two_by_two() {
    let cohorts: Vec<i64> = (0..70).map(|unit| if unit < 30 { 2 } else { 0 }).collect();
    let change = |unit: usize| -> f64 {
        ((13 * unit + 5) % 17) as f64 * 0.25 + if unit < 30 { 2.0 } else { 0.0 }
    };
    let y = |unit: usize, period: i64| -> f64 {
        unit as f64 * 0.1 + if period == 2 { change(unit) } else { 0.0 }
    };
    let arm = |members: std::ops::Range<usize>| -> (f64, f64) {
        let m = members.len() as f64;
        let mean = members.clone().map(change).sum::<f64>() / m;
        let variance = members.map(|u| (change(u) - mean).powi(2)).sum::<f64>() / (m * (m + 1.0));
        (mean, variance)
    };
    let ((treated, treated_variance), (control, control_variance)) = (arm(0..30), arm(30..70));
    let posterior = super::posterior_rollout_did(
        &hand_panel(&cohorts, 2, y, None).panel(),
        &spec(RolloutDidEstimator::CallawaySantAnna, DidComparisonGroup::NeverTreated),
        4_000,
        &ExecutionContext::for_tests(5),
    )
    .unwrap();
    let draws = &posterior.columns[0].1;
    let n = draws.len() as f64;
    let mean = draws.iter().sum::<f64>() / n;
    let variance = draws.iter().map(|draw| (draw - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let want = treated_variance + control_variance;
    assert!((mean - (treated - control)).abs() < 5.0 * (want / n).sqrt(), "mean {mean}");
    assert!((variance / want - 1.0).abs() < 0.1, "variance {variance} against {want}");
}
