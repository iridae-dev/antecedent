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
        assert!((overall.estimate - 3.5).abs() < 0.15, "{estimator:?}: {}", overall.estimate);
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
            assert!((effect.estimate - 3.0 * effect.event_time.unwrap() as f64).abs() < 0.5);
        }
    }
    assert!(rollout.critical_value.is_some());
}

#[test]
fn an_unbalanced_panel_runs_for_every_estimator_and_an_informative_process_is_refused() {
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
    assert!((rollout.overall().unwrap().estimate - 3.5).abs() < 0.2);
    assert_eq!(rollout.diagnostic("balanced_panel"), Some(0.0));
    let mut spec = robust(RolloutDidEstimator::SunAbraham, DidAggregation::Simple);
    spec.comparison = DidComparisonGroup::NeverTreated;
    let result = Study::tabular(data.clone())
        .query(rollout_query(&units, &periods, &cohorts, spec.clone()))
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    let rollout = result.quasi_rollout().unwrap();
    assert!((rollout.overall().unwrap().estimate - 3.5).abs() < 0.2);
    assert_eq!(rollout.diagnostic("balanced_panel"), Some(0.0));
    // Per-cohort support and the dropped rows are reported with the fit.
    assert_eq!(rollout.diagnostic("cohort_2_units"), Some(150.0));
    assert_eq!(rollout.diagnostic("cohort_4_units"), Some(150.0));
    assert!(rollout.diagnostic("cohort_2_rows").unwrap() < 750.0);
    assert!(rollout.diagnostic("dropped_rows").is_some());
    // The observation process is recorded as an untestable assumption of the result.
    let assumption = result
        .identification
        .required_assumptions
        .entries
        .iter()
        .find(|record| {
            matches!(
                &record.assumption,
                antecedent_core::Assumption::Custom { id, .. }
                    if id.as_ref() == "attrition_ignorable_given_cohort"
            )
        })
        .expect("the observation process assumption is recorded");
    assert_eq!(assumption.status, antecedent_core::AssumptionStatus::Untestable);
    // A declared bound on an informative process gives an assumption-dependent range.
    spec.observation = antecedent_core::PanelObservationProcess::InformativeBounded(
        antecedent_core::AttritionShiftBound::new(1.0).unwrap(),
    );
    let bounded = Study::tabular(data.clone())
        .query(rollout_query(&units, &periods, &cohorts, spec.clone()))
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    let section = bounded.quasi_rollout().unwrap();
    let (estimate, slope) =
        (section.overall().unwrap().estimate, section.diagnostic("attrition_slope").unwrap());
    assert!(slope > 0.0 && slope < 1.0, "{slope}");
    let (lower, upper) = (
        section.diagnostic("assumption_range_lower").unwrap(),
        section.diagnostic("assumption_range_upper").unwrap(),
    );
    assert!(
        (lower - (estimate - slope)).abs() < 1e-12 && (upper - (estimate + slope)).abs() < 1e-12
    );
    assert_eq!(section.diagnostic("attrition_shift_bound"), Some(1.0));
    assert!(
        bounded
            .diagnostics
            .iter()
            .any(|d| d.code.as_ref() == "sensitivity.quasi.rollout.attrition_assumption_range")
    );
    // Without a declared bound there is nothing to bound the departure and it is refused.
    spec.observation = antecedent_core::PanelObservationProcess::Informative;
    let error = Study::tabular(data)
        .query(rollout_query(&units, &periods, &cohorts, spec))
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("informative") && error.to_lowercase().contains("next attempt"),
        "{error}"
    );
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
        if nuisance == DidNuisanceMethod::OutcomeRegression {
            // Independent outcome-regression ATT: the treated mean change minus the treated mean of
            // the control-group linear prediction of the change on the covariate.
            let (mut sx, mut sy, mut sxx, mut sxy, mut nc) = (0.0, 0.0, 0.0, 0.0, 0.0);
            let (mut treated_change, mut treated_pred_x, mut nt) = (0.0, 0.0, 0.0);
            let mut changes = Vec::new();
            for i in 0..3000 {
                changes.push((x[2 * i], y[2 * i + 1] - y[2 * i], treated[2 * i]));
            }
            for &(cx, dy, is_treated) in &changes {
                if is_treated {
                    treated_change += dy;
                    treated_pred_x += cx;
                    nt += 1.0;
                } else {
                    sx += cx;
                    sy += dy;
                    sxx += cx * cx;
                    sxy += cx * dy;
                    nc += 1.0;
                }
            }
            let slope = (sxy - sx * sy / nc) / (sxx - sx * sx / nc);
            let intercept = (sy - slope * sx) / nc;
            let reference = (treated_change - nt * intercept - slope * treated_pred_x) / nt;
            assert!(
                (rollout.overall().unwrap().estimate - reference).abs() < 1e-6,
                "outcome regression {} vs reference {reference}",
                rollout.overall().unwrap().estimate
            );
        }
        assert!((rollout.overall().unwrap().estimate - 2.0).abs() < 0.15, "{nuisance:?}");
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
    assert!((overall.estimate - 4.5).abs() < 0.35, "{}", overall.estimate);
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
    let reference_treated = treated.clone();
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
    assert_eq!(section.interval_95, result.panel_did().unwrap().interval_95);
    let fit = result.panel_did().unwrap();
    // Independent references from the definition of the supplied-nuisance augmented ATT with
    // r_i = (post_i - pre_i) - m0_i, odds o_i = p_i / (1 - p_i):
    //   theta = mean_{D=1} r - sum_{D=0} o r / sum_{D=0} o.
    // Its first-order variance is that of the influence values
    //   psi_i = D_i (r_i - A) / pi - (1 - D_i) o_i (r_i - B) / m,
    // with A, B the two terms, pi = mean D and m = mean (1 - D) o, so SE = sd(psi) / sqrt(n).
    let n = 400.0_f64;
    let (mut a_sum, mut n_treated, mut b_num, mut b_den) = (0.0, 0.0, 0.0, 0.0);
    let residual: Vec<f64> = (0..400).map(|i| post[i] - pre[i] - m[i]).collect();
    for i in 0..400 {
        if reference_treated[i] {
            a_sum += residual[i];
            n_treated += 1.0;
        } else {
            let odds = p[i] / (1.0 - p[i]);
            b_num += odds * residual[i];
            b_den += odds;
        }
    }
    let (a, b) = (a_sum / n_treated, b_num / b_den);
    assert!((fit.effect - (a - b)).abs() < 1e-10, "{} vs {}", fit.effect, a - b);
    assert!((fit.effect - 1.5).abs() < 0.12, "{}", fit.effect);
    let (pi, m) = (n_treated / n, b_den / n);
    let psi: Vec<f64> = (0..400)
        .map(|i| {
            if reference_treated[i] {
                (residual[i] - a) / pi
            } else {
                let odds = p[i] / (1.0 - p[i]);
                -odds * (residual[i] - b) / m
            }
        })
        .collect();
    let mean = psi.iter().sum::<f64>() / n;
    let influence_se =
        (psi.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt() / n.sqrt();
    let se = section.standard_error.expect("the cluster resample publishes a standard error");
    // 199 whole-cluster resamples estimate an SE to within about 10 percent (observed ratios across
    // seeds 0.87 to 1.06).
    assert!(
        (se / influence_se - 1.0).abs() < 0.2,
        "resample SE {se} vs influence-function SE {influence_se}"
    );
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
    assert!((overall.estimate - 3.0).abs() < 0.4);
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
    assert!((predictive.overall().unwrap().estimate - 3.0).abs() < 0.25);
}

/// Simplex-constrained least squares `min_w mean (target - X w)^2` over the donor columns,
/// by pairwise coordinate descent with exact line search from the uniform start: the largest
/// gradient entry among positive weights hands mass to the smallest gradient entry. This is an
/// independent solver for the product's accelerated projected gradient; the fitted values are
/// the unique minimiser's, the weights themselves are unique only when the donors are
/// linearly independent.
fn simplex_least_squares(target: &[f64], donors: &[Vec<f64>]) -> Vec<f64> {
    let k = donors.len();
    let n = target.len() as f64;
    let gram: Vec<Vec<f64>> = (0..k)
        .map(|a| {
            (0..k)
                .map(|b| donors[a].iter().zip(&donors[b]).map(|(x, y)| x * y).sum::<f64>() / n)
                .collect()
        })
        .collect();
    let cross: Vec<f64> =
        donors.iter().map(|d| d.iter().zip(target).map(|(x, y)| x * y).sum::<f64>() / n).collect();
    let mut w = vec![1.0 / k as f64; k];
    let mut grad: Vec<f64> =
        (0..k).map(|a| 2.0 * ((0..k).map(|b| gram[a][b] * w[b]).sum::<f64>() - cross[a])).collect();
    for _ in 0..200_000 {
        let i = (0..k).min_by(|&a, &b| grad[a].total_cmp(&grad[b])).unwrap();
        let j = (0..k)
            .filter(|&a| w[a] > 0.0)
            .max_by(|&a, &b| grad[a].total_cmp(&grad[b]))
            .unwrap();
        let gap = grad[j] - grad[i];
        if gap < 1e-13 {
            break;
        }
        let curvature = 2.0 * (gram[i][i] - 2.0 * gram[i][j] + gram[j][j]);
        let step = if curvature > 1e-18 { (gap / curvature).min(w[j]) } else { w[j] };
        w[i] += step;
        w[j] -= step;
        for a in 0..k {
            grad[a] += 2.0 * step * (gram[a][i] - gram[a][j]);
        }
    }
    w
}

/// Independent synthetic difference in differences of a staggered rollout, following the
/// definition: for each adopting unit, simplex unit weights fit to the pre-period outcomes of the
/// donors, simplex time weights fit to the demeaned donor post-period means from the demeaned
/// donor profiles of each pre-period, and the double-difference
/// `(treated post - time-weighted treated pre) - (unit-weighted donor post - time- and unit-weighted
/// donor pre)`; the overall effect weights the unit effects by their post-period counts.
fn reference_synthetic_did(
    values: &[Vec<f64>],
    treated: &[(usize, usize)],
    donors: &[usize],
) -> (f64, Vec<f64>) {
    let demean = |v: &[f64]| -> Vec<f64> {
        let m = v.iter().sum::<f64>() / v.len() as f64;
        v.iter().map(|x| x - m).collect()
    };
    let (mut effects, mut counts) = (Vec::new(), Vec::new());
    for &(unit, adoption) in treated {
        let periods = values[unit].len();
        let pre: Vec<usize> = (0..adoption).collect();
        let post: Vec<usize> = (adoption..periods).collect();
        let donor_pre: Vec<Vec<f64>> =
            donors.iter().map(|&u| pre.iter().map(|&t| values[u][t]).collect()).collect();
        let target_pre: Vec<f64> = pre.iter().map(|&t| values[unit][t]).collect();
        let unit_weights = simplex_least_squares(&target_pre, &donor_pre);
        let post_mean =
            |u: usize| post.iter().map(|&t| values[u][t]).sum::<f64>() / post.len() as f64;
        let donor_post: Vec<f64> = donors.iter().map(|&u| post_mean(u)).collect();
        let profiles: Vec<Vec<f64>> = pre
            .iter()
            .map(|&t| demean(&donors.iter().map(|&u| values[u][t]).collect::<Vec<_>>()))
            .collect();
        let time_weights = simplex_least_squares(&demean(&donor_post), &profiles);
        let treated_pre: f64 = pre.iter().zip(&time_weights).map(|(&t, l)| l * values[unit][t]).sum();
        let donor_pre_joint: f64 = donors
            .iter()
            .zip(&unit_weights)
            .map(|(&u, w)| w * pre.iter().zip(&time_weights).map(|(&t, l)| l * values[u][t]).sum::<f64>())
            .sum();
        let synthetic_post: f64 = donor_post.iter().zip(&unit_weights).map(|(p, w)| p * w).sum();
        effects.push((post_mean(unit) - treated_pre) - (synthetic_post - donor_pre_joint));
        counts.push(post.len() as f64);
    }
    let total: f64 = counts.iter().sum();
    (effects.iter().zip(&counts).map(|(e, c)| e * c).sum::<f64>() / total, effects)
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
    let reference_y = y.clone();
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
    // Independent references: the point estimate from an in-test solver of the same weighting
    // definition, and the placebo SE from an in-test placebo procedure (the three adoption dates
    // reassigned to randomly chosen never-treated units, the whole estimator refit on the
    // remaining donors, the population variance of the placebo overall estimates).
    let values: Vec<Vec<f64>> = (0..33).map(|u| reference_y[u * 22..(u + 1) * 22].to_vec()).collect();
    let treated_units: Vec<(usize, usize)> = vec![(30, 15), (31, 17), (32, 19)];
    let donor_units: Vec<usize> = (0..30).collect();
    let (reference, unit_effects) = reference_synthetic_did(&values, &treated_units, &donor_units);
    assert!((overall.estimate - reference).abs() < 1e-3, "{} vs {reference}", overall.estimate);
    let reported_units: Vec<f64> = rollout
        .effects
        .iter()
        .filter(|effect| effect.kind == DesignEffectKind::Unit)
        .map(|effect| effect.estimate)
        .collect();
    for (reported, expected) in reported_units.iter().zip(&unit_effects) {
        assert!((reported - expected).abs() < 1e-3, "{reported} vs {expected}");
    }
    let mut placebo_rng = Rng(0xfeed);
    let mut placebo = Vec::new();
    for _ in 0..300 {
        let mut pool = donor_units.clone();
        let mut assigned = Vec::new();
        for (index, &(_, adoption)) in treated_units.iter().enumerate() {
            let pick = index
                + (((placebo_rng.uniform() * (pool.len() - index) as f64) as usize).min(pool.len() - index - 1));
            pool.swap(index, pick);
            assigned.push((pool[index], adoption));
        }
        placebo.push(reference_synthetic_did(&values, &assigned, &pool[3..]).0);
    }
    let mean = placebo.iter().sum::<f64>() / placebo.len() as f64;
    let reference_se =
        (placebo.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / placebo.len() as f64).sqrt();
    assert!(
        (se / reference_se - 1.0).abs() < 0.25,
        "placebo SE {se} vs independent placebo procedure {reference_se}"
    );
    assert!((overall.estimate - 2.0).abs() < 4.0 * se + 0.1, "{} ± {se}", overall.estimate);
    assert_eq!(
        rollout.effects.iter().filter(|effect| effect.kind == DesignEffectKind::Unit).count(),
        3
    );
    assert!(rollout.replicates_ok >= 190);
}

/// Independent robust-bias-corrected fuzzy-RD reference. On each side of the cutoff it fits the
/// triangular-kernel weighted cubic polynomial `y ~ 1 + d + d^2 + d^3` (`d = (r - c) / h`) by
/// normal equations; the corrected side coefficient is the cubic fit's intercept (the quadratic
/// intercept minus the projection of its omitted cubic term is that intercept by
/// Frisch-Waugh). Returns `(estimate, standard error, reduced form, first stage)` with the
/// per-side HC0 covariance of the intercepts, combined by the delta method for the ratio.
fn reference_fuzzy_rd(
    running: &[f64],
    outcome: &[f64],
    treatment: &[f64],
    cutoff: f64,
    h: f64,
) -> (f64, f64, f64, f64) {
    fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
        let k = b.len();
        for col in 0..k {
            let pivot =
                (col..k).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs())).unwrap();
            a.swap(col, pivot);
            b.swap(col, pivot);
            for row in col + 1..k {
                let f = a[row][col] / a[col][col];
                for c in col..k {
                    a[row][c] -= f * a[col][c];
                }
                b[row] -= f * b[col];
            }
        }
        let mut x = vec![0.0; k];
        for row in (0..k).rev() {
            let tail: f64 = (row + 1..k).map(|c| a[row][c] * x[c]).sum();
            x[row] = (b[row] - tail) / a[row][row];
        }
        x
    }
    // (intercept_y, intercept_t, var_y, var_t, cov_yt) of one side.
    let side = |right: bool| -> (f64, f64, f64, f64, f64) {
        let rows: Vec<(usize, f64)> = (0..running.len())
            .filter_map(|i| {
                let d = (running[i] - cutoff) / h;
                (d.abs() < 1.0 && if right { d >= 0.0 } else { d < 0.0 }).then_some((i, d))
            })
            .collect();
        let basis = |d: f64| [1.0, d, d * d, d * d * d];
        let mut gram = vec![vec![0.0; 4]; 4];
        let (mut ry, mut rt) = (vec![0.0; 4], vec![0.0; 4]);
        for &(i, d) in &rows {
            let (x, w) = (basis(d), 1.0 - d.abs());
            for a in 0..4 {
                ry[a] += w * x[a] * outcome[i];
                rt[a] += w * x[a] * treatment[i];
                for b in 0..4 {
                    gram[a][b] += w * x[a] * x[b];
                }
            }
        }
        let beta_y = solve(gram.clone(), ry);
        let beta_t = solve(gram.clone(), rt);
        // First row of the inverse Gram matrix: bread[0] solves gram * bread[0] = e_0.
        let bread0 = solve(gram, vec![1.0, 0.0, 0.0, 0.0]);
        let (mut vy, mut vt, mut cyt) = (0.0, 0.0, 0.0);
        for &(i, d) in &rows {
            let (x, w) = (basis(d), 1.0 - d.abs());
            let ey = outcome[i] - (0..4).map(|a| beta_y[a] * x[a]).sum::<f64>();
            let et = treatment[i] - (0..4).map(|a| beta_t[a] * x[a]).sum::<f64>();
            // The intercept's influence on row i is (bread row 0) . x_i * w * e.
            let loading: f64 = (0..4).map(|a| bread0[a] * x[a]).sum::<f64>() * w;
            vy += (loading * ey).powi(2);
            vt += (loading * et).powi(2);
            cyt += loading * loading * ey * et;
        }
        (beta_y[0], beta_t[0], vy, vt, cyt)
    };
    let (left, right) = (side(false), side(true));
    let reduced = right.0 - left.0;
    let first = right.1 - left.1;
    let (vy, vt, cyt) = (left.2 + right.2, left.3 + right.3, left.4 + right.4);
    let variance = vy / first.powi(2) + reduced.powi(2) * vt / first.powi(4)
        - 2.0 * reduced * cyt / first.powi(3);
    (reduced / first, variance.sqrt(), reduced, first)
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
    // The estimate and its HC0 delta-method SE at the selected bandwidth are the independent
    // cubic-fit reference, not a band around the truth.
    let h = rollout.diagnostic("bandwidth").unwrap();
    let (reference, reference_se, _, first) =
        reference_fuzzy_rd(&running, &outcome, &treatment, 0.0, h);
    assert!((overall.estimate - reference).abs() < 1e-8, "{} vs {reference}", overall.estimate);
    assert!((se - reference_se).abs() < 1e-8 * (1.0 + reference_se), "{se} vs {reference_se}");
    assert!((rollout.diagnostic("hc0_ratio_standard_error").unwrap() - reference_se).abs() < 1e-8);
    assert!(first > 0.5 && first < 0.7, "first stage {first} should be near the 0.6 jump");
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
    // The point estimate and the HC0 SE are the independent cubic-fit reference at the declared
    // bandwidth (the first stage of a sharp calendar-time design is exactly one).
    let (reference, reference_hc0, _, first) =
        reference_fuzzy_rd(&time, &y, &treated, 1000.0, 300.0);
    assert!((first - 1.0).abs() < 1e-10, "{first}");
    assert!((overall.estimate - reference).abs() < 1e-8, "{} vs {reference}", overall.estimate);
    let hc0 = rollout.diagnostic("hc0_ratio_standard_error").unwrap();
    assert!((hc0 - reference_hc0).abs() < 1e-8 * (1.0 + hc0), "{hc0} vs {reference_hc0}");
    // The block-bootstrap SE is checked against an independent moving-block bootstrap run here:
    // rows inside the window in time order, blocks of round(cbrt(min side rows)) consecutive
    // rows drawn with replacement within each side and truncated to the side's length, the whole
    // local fit repeated on the resampled rows.
    let side_rows = |right: bool| -> Vec<usize> {
        (0..time.len())
            .filter(|&i| {
                (time[i] - 1000.0).abs() < 300.0 && if right { time[i] >= 1000.0 } else { time[i] < 1000.0 }
            })
            .collect()
    };
    let (left, right) = (side_rows(false), side_rows(true));
    assert_eq!((left.len(), right.len()), (299, 300));
    let block = (left.len().min(right.len()) as f64).cbrt().round() as usize;
    assert_eq!(block, 7);
    let mut reference_rng = Rng(0x5eed);
    let mut draws = Vec::new();
    for _ in 0..400 {
        let (mut r, mut o, mut d) = (Vec::new(), Vec::new(), Vec::new());
        for rows in [&left, &right] {
            let starts = rows.len() + 1 - block;
            let mut taken = 0;
            for _ in 0..rows.len().div_ceil(block) {
                let start = ((reference_rng.uniform() * starts as f64) as usize).min(starts - 1);
                for &row in &rows[start..(start + block).min(rows.len())] {
                    if taken == rows.len() {
                        break;
                    }
                    r.push(time[row]);
                    o.push(y[row]);
                    d.push(treated[row]);
                    taken += 1;
                }
            }
        }
        draws.push(reference_fuzzy_rd(&r, &o, &d, 1000.0, 300.0).0);
    }
    let mean = draws.iter().sum::<f64>() / draws.len() as f64;
    let reference_block_se =
        (draws.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (draws.len() as f64 - 1.0)).sqrt();
    assert!(
        (block_se / reference_block_se - 1.0).abs() < 0.2,
        "block bootstrap SE {block_se} vs independent moving-block bootstrap {reference_block_se}"
    );
    assert!(block_se > 1.3 * hc0);
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
    // Non-overlapping blocks under flat-Dirichlet weights are a different scheme from the moving
    // block bootstrap, so the posterior spread is only required to be of the same order as the
    // independent moving-block reference (observed ratio 0.78).
    assert!(
        posterior_sd > 0.6 * reference_block_se && posterior_sd < 1.05 * reference_block_se,
        "posterior sd {posterior_sd} vs moving-block bootstrap {reference_block_se}"
    );
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
