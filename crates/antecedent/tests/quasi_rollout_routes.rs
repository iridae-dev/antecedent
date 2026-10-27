//! Rollout designs through the compiler: heterogeneity-robust and covariate-adjusted
//! difference in differences, the resampled augmented panel, conformal and multi-unit
//! synthetic control, and the local ratio with a selected bandwidth or a calendar-time running
//! variable. Each route executes from its compiled plan, recovers a simulated known truth,
//! reports its whole-procedure intervals as unlicensed with calibration `unmeasured`, keeps the
//! design's own assumptions in the identification, and refuses what it must with the next
//! attempt.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::prelude::ExecutionContext;
use antecedent::{BayesianConfig, InferenceMode, PanelDidQuery, Study, StudyResult};
use antecedent_core::{
    Assumption, DidAggregation, DidComparisonGroup, DidNuisanceMethod, DidResamplingScheme,
    HeterogeneityRobustDid, LocalBandwidthRule, LocalPolynomialRatioQuery, LocalRatioOptions,
    RolloutDidEstimator, SyntheticControlQuery, VariableId,
};
use antecedent_data::TabularData;
use antecedent_estimate::rollout_resampling::DesignEffectKind;

struct Rng(u64);

impl Rng {
    fn uniform(&mut self) -> f64 {
        (antecedent_core::splitmix64(&mut self.0) >> 11) as f64 / (1_u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let u = self.uniform().max(1e-300);
        let v = self.uniform();
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

fn has_assumption(result: &StudyResult, id: &str) -> bool {
    result.identification.required_assumptions.entries.iter().any(
        |record| matches!(&record.assumption, Assumption::Custom { id: name, .. } if name.as_ref() == id),
    )
}

/// Cohorts 0, 2 and 4 over five periods with effect `3·e`: the simple aggregate is 3.5 and the
/// static two-way fixed-effects coefficient is about 0.75. `drop` removes rows to unbalance it.
fn staggered(
    seed: u64,
    per_cohort: usize,
    drop: f64,
) -> (TabularData, Vec<Arc<str>>, Vec<i64>, Vec<i64>) {
    let mut rng = Rng(seed);
    let (mut y, mut units, mut periods, mut cohorts) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for cohort in [0_i64, 2, 4] {
        for i in 0..per_cohort {
            let label: Arc<str> = Arc::from(format!("c{cohort}_{i:04}"));
            let alpha = rng.normal();
            for period in 1..=5_i64 {
                let keep = rng.uniform() >= drop;
                let effect = if cohort > 0 && period >= cohort {
                    3.0 * (period - cohort) as f64
                } else {
                    0.0
                };
                let value = alpha + 0.5 * period as f64 + effect + rng.normal();
                if keep {
                    y.push(value);
                    units.push(Arc::clone(&label));
                    periods.push(period);
                    cohorts.push(cohort);
                }
            }
        }
    }
    (TabularData::from_f64_columns([("y", y.as_slice())]).unwrap(), units, periods, cohorts)
}

fn robust(estimator: RolloutDidEstimator, aggregation: DidAggregation) -> HeterogeneityRobustDid {
    HeterogeneityRobustDid {
        estimator,
        comparison: DidComparisonGroup::NotYetTreated,
        aggregation,
        ..HeterogeneityRobustDid::default()
    }
}

fn rollout_query(
    units: &[Arc<str>],
    periods: &[i64],
    cohorts: &[i64],
    spec: HeterogeneityRobustDid,
) -> PanelDidQuery {
    PanelDidQuery::heterogeneity_robust(
        VariableId::from_raw(0),
        units.to_vec(),
        units.to_vec(),
        periods.to_vec(),
        cohorts.to_vec(),
        spec,
    )
}

#[test]
fn staggered_rollout_recovers_heterogeneous_effects_where_two_way_fixed_effects_is_biased() {
    let (data, units, periods, cohorts) = staggered(1, 120, 0.0);
    let ctx = ExecutionContext::for_tests(7);
    for estimator in [
        RolloutDidEstimator::CallawaySantAnna,
        RolloutDidEstimator::SunAbraham,
        RolloutDidEstimator::Imputation,
    ] {
        let mut spec = robust(estimator, DidAggregation::Simple);
        if estimator == RolloutDidEstimator::SunAbraham {
            spec.comparison = DidComparisonGroup::NeverTreated;
        }
        let query = rollout_query(&units, &periods, &cohorts, spec)
            .with_resampling(DidResamplingScheme::ClusterMultiplier, 99);
        let study = Study::tabular(data.clone()).query(query).build().unwrap();
        let prepared = study.prepare(&ctx).unwrap();
        let result = prepared.estimate(&data, &ctx).unwrap();
        assert_eq!(
            result.logical_plan.estimator.as_deref(),
            Some("quasi.staggered_heterogeneity_robust")
        );
        assert_eq!(result.support_status(), None, "the rollout route is unlicensed");
        let rollout = result.quasi_rollout().unwrap();
        assert_eq!(rollout.calibration.as_ref(), "unmeasured");
        assert!(!rollout.licensed);
        let overall = rollout.overall().unwrap();
        assert!((overall.estimate - 3.5).abs() < 0.35, "{estimator:?}: {}", overall.estimate);
        assert!(overall.interval.is_some());
        let twfe = rollout.diagnostic("twfe_static_coefficient").unwrap();
        assert!((twfe - 3.5).abs() > 2.0, "two-way fixed effects {twfe}");
        assert!(has_assumption(&result, "cohort_specific_parallel_untreated_trends"));
        assert!(has_assumption(&result, "target_population_adopted_cohorts"));
        assert!(result.panel_did().is_none());
    }
}

#[test]
fn dynamic_aggregation_publishes_simultaneous_bands_and_is_thread_invariant() {
    let (data, units, periods, cohorts) = staggered(2, 80, 0.0);
    let query = rollout_query(
        &units,
        &periods,
        &cohorts,
        robust(RolloutDidEstimator::CallawaySantAnna, DidAggregation::Dynamic),
    )
    .with_resampling(DidResamplingScheme::ClusterBootstrap, 199);
    let run = |ctx: &ExecutionContext| {
        Study::tabular(data.clone()).query(query.clone()).build().unwrap().run(ctx).unwrap()
    };
    let serial = run(&ExecutionContext::for_tests(3));
    let parallel = run(&ExecutionContext::production(3, 4));
    assert_eq!(serial.quasi_rollout(), parallel.quasi_rollout());
    let rollout = serial.quasi_rollout().unwrap();
    let dynamic: Vec<_> = rollout
        .effects
        .iter()
        .filter(|effect| effect.kind == DesignEffectKind::EventTime)
        .collect();
    assert!(!dynamic.is_empty());
    for effect in dynamic {
        let (interval, band) = (effect.interval.unwrap(), effect.band.unwrap());
        assert!(band[0] <= interval[0] && band[1] >= interval[1]);
        if effect.event_time.unwrap() >= 0 {
            assert!((effect.estimate - 3.0 * effect.event_time.unwrap() as f64).abs() < 0.7);
        }
    }
    assert!(rollout.critical_value.is_some());
}

#[test]
fn an_unbalanced_panel_runs_where_allowed_and_sun_abraham_names_the_next_attempt() {
    let (data, units, periods, cohorts) = staggered(3, 150, 0.15);
    let ctx = ExecutionContext::for_tests(5);
    let query = rollout_query(
        &units,
        &periods,
        &cohorts,
        robust(RolloutDidEstimator::Imputation, DidAggregation::Simple),
    );
    let result = Study::tabular(data.clone()).query(query).build().unwrap().run(&ctx).unwrap();
    let rollout = result.quasi_rollout().unwrap();
    assert!((rollout.overall().unwrap().estimate - 3.5).abs() < 0.4);
    assert_eq!(rollout.diagnostic("balanced_panel"), Some(0.0));
    let mut spec = robust(RolloutDidEstimator::SunAbraham, DidAggregation::Simple);
    spec.comparison = DidComparisonGroup::NeverTreated;
    let error = Study::tabular(data)
        .query(rollout_query(&units, &periods, &cohorts, spec))
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap_err()
        .to_string();
    assert!(error.contains("next attempt"), "{error}");
}

#[test]
fn covariate_adjusted_two_period_did_recovers_the_effect_under_conditional_trends() {
    let mut rng = Rng(4);
    let (mut y, mut x, mut treated, mut post, mut units) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 0..3000 {
        let covariate = rng.normal();
        let d = rng.uniform() < 1.0 / (1.0 + (-covariate).exp());
        let alpha = rng.normal() + covariate;
        let label: Arc<str> = Arc::from(format!("s{i:05}"));
        for after in [false, true] {
            let trend = if after { 1.0 + covariate } else { 0.0 };
            let effect = if d && after { 2.0 } else { 0.0 };
            y.push(alpha + trend + effect + 0.5 * rng.normal());
            x.push(covariate);
            treated.push(d);
            post.push(after);
            units.push(Arc::clone(&label));
        }
    }
    let data = TabularData::from_f64_columns([("y", y.as_slice()), ("x", x.as_slice())]).unwrap();
    let ctx = ExecutionContext::for_tests(6);
    for nuisance in [
        DidNuisanceMethod::OutcomeRegression,
        DidNuisanceMethod::InverseProbabilityWeighting,
        DidNuisanceMethod::DoublyRobust,
    ] {
        let query = PanelDidQuery::covariate_adjusted_two_period(
            VariableId::from_raw(0),
            treated.clone(),
            post.clone(),
            units.clone(),
            units.clone(),
            vec![VariableId::from_raw(1)],
            nuisance,
        );
        let result = Study::tabular(data.clone()).query(query).build().unwrap().run(&ctx).unwrap();
        let rollout = result.quasi_rollout().unwrap();
        assert_eq!(rollout.design.as_ref(), "covariate_adjusted_did");
        assert!((rollout.overall().unwrap().estimate - 2.0).abs() < 0.2, "{nuisance:?}");
        assert!(has_assumption(
            &result,
            "conditional_parallel_untreated_trends_given_baseline_covariates"
        ));
    }
}

#[test]
fn the_bayesian_rollout_is_a_joint_bayesian_bootstrap_posterior_with_credible_bands() {
    let (data, units, periods, cohorts) = staggered(5, 80, 0.1);
    let query = rollout_query(
        &units,
        &periods,
        &cohorts,
        robust(RolloutDidEstimator::CallawaySantAnna, DidAggregation::Dynamic),
    );
    let ctx = ExecutionContext::for_tests(1);
    let prepared = Study::tabular(data.clone())
        .query(query.clone())
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(400)))
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    assert_eq!(result.logical_plan.plan_id.as_ref(), "bayesian.panel_did");
    assert_eq!(result.support_status(), None);
    let posterior = result.posterior.as_ref().unwrap();
    assert!(posterior.draws.n_draws >= 380);
    let rollout = result.quasi_rollout().unwrap();
    assert!(rollout.uncertainty.starts_with("bayesian_"));
    assert_eq!(rollout.calibration.as_ref(), "unmeasured");
    let overall = rollout.overall().unwrap();
    assert!((overall.estimate - 4.5).abs() < 0.5, "{}", overall.estimate);
    for effect in rollout.effects.iter().filter(|effect| effect.kind == DesignEffectKind::EventTime)
    {
        let (interval, band) = (effect.interval.unwrap(), effect.band.unwrap());
        assert!(band[0] <= interval[0] && band[1] >= interval[1]);
        assert!(effect.standard_error.is_none());
    }
    // The Bayesian section round-trips beside its posterior.
    let bytes = prepared.encode_contracted_result(&result, "bayesian-rollout", &ctx).unwrap();
    let (_, _, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert!(body.quasi_rollout.is_some() && body.posterior_artifact.is_some());
    // A frequentist resample declared under Bayes is refused naming the next attempt.
    let error = Study::tabular(data)
        .query(query.with_resampling(DidResamplingScheme::ClusterBootstrap, 99))
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(400)))
        .build()
        .and_then(|study| study.run(&ctx))
        .unwrap_err()
        .to_string();
    assert!(error.contains("Next attempt"), "{error}");
}

#[test]
fn the_augmented_panel_publishes_a_whole_cluster_resample_interval() {
    let mut rng = Rng(8);
    let (mut pre, mut post, mut p, mut m, mut treated, mut ids) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 0..400 {
        let x = rng.uniform();
        let propensity = 0.25 + 0.5 * x;
        let d = rng.uniform() < propensity;
        let base = rng.normal();
        pre.push(base);
        post.push(base + x + if d { 1.5 } else { 0.0 } + 0.2 * rng.normal());
        p.push(propensity);
        m.push(x);
        treated.push(d);
        ids.push(Arc::<str>::from(format!("s{i}")));
    }
    let data = TabularData::from_f64_columns([
        ("pre", pre.as_slice()),
        ("post", post.as_slice()),
        ("p", p.as_slice()),
        ("m0", m.as_slice()),
    ])
    .unwrap();
    let query = PanelDidQuery::augmented_panel(
        VariableId::from_raw(1),
        VariableId::from_raw(0),
        VariableId::from_raw(2),
        VariableId::from_raw(3),
        treated,
        ids.clone(),
        ids,
        true,
    )
    .with_resampling(DidResamplingScheme::ClusterBootstrap, 199);
    let ctx = ExecutionContext::for_tests(2);
    let prepared =
        Study::tabular(data.clone()).query(query).build().unwrap().prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let bytes = prepared.encode_contracted_result(&result, "augmented-resampled", &ctx).unwrap();
    let (_, _, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    let section = body.panel_did.as_ref().unwrap();
    assert!(section.standard_error.is_some_and(|se| se > 0.0));
    assert_eq!(section.interval_95, result.panel_did().unwrap().interval_95);
    let fit = result.panel_did().unwrap();
    assert!((fit.effect - 1.5).abs() < 0.15, "{}", fit.effect);
    let interval = fit.interval_95.unwrap();
    assert!(interval[0] < fit.effect && fit.effect < interval[1]);
    assert!(fit.uncertainty.contains("calibration_unmeasured"));
    assert_eq!(result.support_status(), None);
    assert!(has_assumption(&result, "independent_resampling_units"));
}

/// Ten donors with period-exchangeable outcomes; the treated unit is a convex combination plus
/// noise, with effect `effect` from `intervention` on.
fn synthetic_panel(
    seed: u64,
    periods: i64,
    intervention: i64,
    effect: f64,
) -> (TabularData, Vec<Arc<str>>, Vec<i64>) {
    let mut rng = Rng(seed);
    let (mut y, mut units, mut times) = (Vec::new(), Vec::new(), Vec::new());
    for period in 1..=periods {
        let donors: Vec<f64> = (0..10).map(|j| f64::from(j) + rng.normal()).collect();
        for (j, value) in donors.iter().enumerate() {
            y.push(*value);
            units.push(Arc::from(format!("d{j:02}")));
            times.push(period);
        }
        let mut treated = 0.5 * donors[3] + 0.3 * donors[4] + 0.2 * donors[5] + 0.5 * rng.normal();
        if period >= intervention {
            treated += effect;
        }
        y.push(treated);
        units.push(Arc::from("treated"));
        times.push(period);
    }
    (TabularData::from_f64_columns([("y", y.as_slice())]).unwrap(), units, times)
}

#[test]
fn conformal_synthetic_control_reports_period_intervals_and_refuses_a_bayesian_request() {
    let (data, units, periods) = synthetic_panel(9, 45, 41, 3.0);
    let query = SyntheticControlQuery::new(
        VariableId::from_raw(0),
        units,
        periods,
        Arc::<str>::from("treated"),
        41,
    )
    .with_conformal_intervals();
    let ctx = ExecutionContext::for_tests(1);
    let result =
        Study::tabular(data.clone()).query(query.clone()).build().unwrap().run(&ctx).unwrap();
    assert_eq!(result.logical_plan.estimator.as_deref(), Some("quasi.synthetic_control_conformal"));
    let rollout = result.quasi_rollout().unwrap();
    let overall = rollout.overall().unwrap();
    assert!((overall.estimate - 3.0).abs() < 0.8);
    assert!(overall.interval.unwrap()[0] > 0.0);
    assert_eq!(
        rollout.effects.iter().filter(|effect| effect.kind == DesignEffectKind::Period).count(),
        5
    );
    assert!(has_assumption(
        &result,
        "post_period_residuals_exchangeable_with_pre_period_residuals"
    ));
    // The Bayesian counterpart is the posterior predictive interval of the counterfactual path.
    let bayesian = Study::tabular(data)
        .query(query)
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(2000)))
        .build()
        .and_then(|study| study.run(&ctx))
        .unwrap();
    let predictive = bayesian.quasi_rollout().unwrap();
    assert_eq!(predictive.design.as_ref(), "posterior_predictive_synthetic_control");
    assert!(bayesian.posterior.is_some());
    let periods: Vec<_> = predictive
        .effects
        .iter()
        .filter(|effect| effect.kind == DesignEffectKind::Period)
        .collect();
    assert_eq!(periods.len(), 5);
    assert!(periods.iter().all(|effect| effect.interval.is_some()));
    assert!((predictive.overall().unwrap().estimate - 3.0).abs() < 1.0);
}

#[test]
fn multi_unit_synthetic_did_recovers_a_staggered_effect_with_placebo_intervals() {
    let mut rng = Rng(10);
    let mut trend = Vec::new();
    let mut level = 0.0;
    for _ in 0..22 {
        level += rng.normal();
        trend.push(level);
    }
    let (mut y, mut units, mut periods) = (Vec::new(), Vec::new(), Vec::new());
    let adoptions: Vec<(Arc<str>, i64)> =
        vec![(Arc::from("t0"), 16), (Arc::from("t1"), 18), (Arc::from("t2"), 20)];
    let mut all: Vec<(Arc<str>, f64, Option<i64>)> =
        (0..30).map(|j| (Arc::from(format!("c{j:02}")), f64::from(j) / 10.0, None)).collect();
    for (k, (label, adoption)) in adoptions.iter().enumerate() {
        all.push((Arc::clone(label), 1.0 + k as f64 / 2.0, Some(*adoption)));
    }
    for (label, unit_level, adoption) in &all {
        for (t, common) in trend.iter().enumerate() {
            let period = i64::try_from(t).expect("a period index fits in i64") + 1;
            let effect = if adoption.is_some_and(|a| period >= a) { 2.0 } else { 0.0 };
            y.push(unit_level + common + effect + 0.3 * rng.normal());
            units.push(Arc::clone(label));
            periods.push(period);
        }
    }
    let data = TabularData::from_f64_columns([("y", y.as_slice())]).unwrap();
    let query = SyntheticControlQuery::new(
        VariableId::from_raw(0),
        units,
        periods,
        Arc::<str>::from("t0"),
        16,
    )
    .difference_in_differences()
    .with_staggered_adoptions(adoptions, 199);
    let result = Study::tabular(data)
        .query(query)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(4))
        .unwrap();
    let rollout = result.quasi_rollout().unwrap();
    assert_eq!(rollout.design.as_ref(), "multi_unit_synthetic_did");
    let overall = rollout.overall().unwrap();
    let se = overall.standard_error.unwrap();
    assert!((overall.estimate - 2.0).abs() < 4.0 * se + 0.1, "{} ± {se}", overall.estimate);
    assert_eq!(
        rollout.effects.iter().filter(|effect| effect.kind == DesignEffectKind::Unit).count(),
        3
    );
    assert!(rollout.replicates_ok >= 190);
}

#[test]
fn fuzzy_rd_with_a_selected_bandwidth_and_diagnostics_recovers_the_effect() {
    let mut rng = Rng(11);
    let (mut running, mut treatment, mut outcome, mut covariate) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..6000 {
        let r = 2.0 * rng.uniform() - 1.0;
        let t = if rng.uniform() < if r >= 0.0 { 0.8 } else { 0.2 } { 1.0 } else { 0.0 };
        running.push(r);
        treatment.push(t);
        outcome.push(1.0 + r + 0.5 * r * r - 0.8 * r.powi(3) + 2.0 * t + 0.5 * rng.normal());
        covariate.push(r + 0.3 * rng.normal());
    }
    let data = TabularData::from_f64_columns([
        ("running", running.as_slice()),
        ("treatment", treatment.as_slice()),
        ("outcome", outcome.as_slice()),
        ("z", covariate.as_slice()),
    ])
    .unwrap();
    let query = LocalPolynomialRatioQuery {
        outcome: VariableId::from_raw(2),
        treatment: VariableId::from_raw(1),
        running: VariableId::from_raw(0),
        cutoff: 0.0,
        bandwidth: 0.8,
        kink: false,
        options: Some(LocalRatioOptions {
            bandwidth_rule: LocalBandwidthRule::MseOptimal,
            temporal_replicates: None,
            density_test: true,
            balance_covariates: Arc::from([VariableId::from_raw(3)]),
        }),
    };
    let result = Study::tabular(data)
        .query(query)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(1))
        .unwrap();
    assert_eq!(
        result.logical_plan.estimator.as_deref(),
        Some("quasi.fuzzy_rd.mse_optimal_bandwidth_rbc")
    );
    let rollout = result.quasi_rollout().unwrap();
    let overall = rollout.overall().unwrap();
    let se = overall.standard_error.unwrap();
    assert!((overall.estimate - 2.0).abs() < 4.0 * se);
    assert!(rollout.diagnostic("bandwidth").unwrap() <= 0.8);
    assert!(rollout.diagnostic("density_p_value").unwrap() > 1e-3);
    assert!(rollout.diagnostic("balance_z_variable_3").unwrap().abs() < 4.0);
    assert!(result.local_polynomial_ratio().is_none());
}

#[test]
fn a_discontinuity_in_time_uses_a_block_bootstrap_and_respects_the_sampling_design() {
    let mut rng = Rng(12);
    let (mut time, mut treated, mut y) = (Vec::new(), Vec::new(), Vec::new());
    let mut error = 0.0;
    for period in 1..=2000_u32 {
        error = 0.7 * error + 0.5 * rng.normal();
        let t = f64::from(period);
        let d = if t >= 1000.0 { 1.0 } else { 0.0 };
        time.push(t);
        treated.push(d);
        y.push(0.001 * t + d + error);
    }
    let data = TabularData::from_f64_columns([
        ("time", time.as_slice()),
        ("treatment", treated.as_slice()),
        ("outcome", y.as_slice()),
    ])
    .unwrap();
    let query = LocalPolynomialRatioQuery {
        outcome: VariableId::from_raw(2),
        treatment: VariableId::from_raw(1),
        running: VariableId::from_raw(0),
        cutoff: 1000.0,
        bandwidth: 300.0,
        kink: false,
        options: Some(LocalRatioOptions {
            temporal_replicates: Some(199),
            ..LocalRatioOptions::default()
        }),
    };
    let result = Study::tabular(data.clone())
        .query(query.clone())
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(3))
        .unwrap();
    assert_eq!(result.logical_plan.plan_id.as_ref(), "quasi.temporal_fuzzy_rd");
    let rollout = result.quasi_rollout().unwrap();
    let overall = rollout.overall().unwrap();
    let block_se = overall.standard_error.unwrap();
    assert!(block_se > 1.3 * rollout.diagnostic("hc0_ratio_standard_error").unwrap());
    assert!((overall.estimate - 1.0).abs() < 4.0 * block_se);
    assert!(has_assumption(&result, "no_concurrent_discontinuity_at_the_cutoff_date"));
    let bayesian = Study::tabular(data.clone())
        .query(query.clone())
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(400)))
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(3))
        .unwrap();
    let posterior_rollout = bayesian.quasi_rollout().unwrap();
    assert_eq!(posterior_rollout.design.as_ref(), "temporal_fuzzy_rd");
    assert!(posterior_rollout.uncertainty.starts_with("bayesian_"));
    let posterior_sd = posterior_rollout.diagnostic("posterior_sd").unwrap();
    assert!(posterior_sd > 1.3 * rollout.diagnostic("hc0_ratio_standard_error").unwrap());
    assert!((posterior_rollout.overall().unwrap().estimate - 1.0).abs() < 4.0 * posterior_sd);
    let refused = Study::tabular(data)
        .query(query)
        .sampling_design(antecedent_core::DeclaredSamplingDesign::ClusteredPanel)
        .build()
        .and_then(|study| study.run(&ExecutionContext::for_tests(3)))
        .unwrap_err()
        .to_string();
    assert!(refused.contains("Next attempt"), "{refused}");
}

#[test]
#[allow(clippy::float_cmp, reason = "the artifact must carry every estimate exactly")]
fn the_rollout_section_round_trips_through_the_result_artifact_and_re_derives_its_summary() {
    let (data, units, periods, cohorts) = staggered(13, 60, 0.0);
    let query = rollout_query(
        &units,
        &periods,
        &cohorts,
        robust(RolloutDidEstimator::CallawaySantAnna, DidAggregation::Dynamic),
    )
    .with_resampling(DidResamplingScheme::ClusterMultiplier, 99);
    let ctx = ExecutionContext::for_tests(2);
    let prepared =
        Study::tabular(data.clone()).query(query).build().unwrap().prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let rollout = result.quasi_rollout().unwrap();
    let bytes = prepared.encode_contracted_result(&result, "quasi-rollout", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    let section = body.quasi_rollout.clone().unwrap();
    assert_eq!(section.design, rollout.design.as_ref());
    assert_eq!(section.calibration, "unmeasured");
    assert!(!section.licensed);
    assert_eq!(section.effects.len(), rollout.effects.len());
    for (wire, effect) in section.effects.iter().zip(rollout.effects.iter()) {
        assert_eq!(wire.kind, effect.kind.as_str());
        assert_eq!(wire.estimate, effect.estimate);
        assert_eq!(wire.interval, effect.interval);
        assert_eq!(wire.band, effect.band);
    }
    // A fresh consumer re-derives the dynamic summary from the event-time effects alone.
    let summary =
        antecedent_io::analysis_result_artifact::rederive_rollout_summary(&section).unwrap();
    assert!((summary - rollout.overall().unwrap().estimate).abs() < 1e-9);
    assert_eq!(body.estimate, Some(rollout.overall().unwrap().estimate));
    assert!(body.standard_error.is_none());
    // A tampered summary, a licence claim or a narrowed band is refused on re-encode.
    let names = header.variable_names;
    let mut tampered = body.clone();
    tampered.quasi_rollout.as_mut().unwrap().effects[0].estimate += 1.0;
    tampered.estimate = tampered.quasi_rollout.as_ref().map(|s| s.effects[0].estimate);
    assert!(antecedent_io::encode_analysis_result_artifact(&tampered, names.clone(), "t").is_err());
    let mut licensed = body.clone();
    licensed.quasi_rollout.as_mut().unwrap().licensed = true;
    assert!(antecedent_io::encode_analysis_result_artifact(&licensed, names.clone(), "t").is_err());
    let mut narrowed = body.clone();
    if let Some(effect) = narrowed
        .quasi_rollout
        .as_mut()
        .unwrap()
        .effects
        .iter_mut()
        .find(|effect| effect.band.is_some())
    {
        effect.band = effect.interval.map(|[lo, hi]| [lo + 1e-3, hi]);
        assert!(
            antecedent_io::encode_analysis_result_artifact(&narrowed, names.clone(), "t").is_err()
        );
    }
    assert!(antecedent_io::encode_analysis_result_artifact(&body, names, "t").is_ok());
}
