//! Observational survival effects at a declared horizon, against truths computed from the
//! generating process and not from the estimator.
//!
//! The generating process has exponential event times with a proportional-hazards effect of
//! the treatment and two discrete confounders, so the survival probability and restricted
//! mean survival time of each arm have closed forms: finite sums of exponentials over the six
//! confounder cells. Censoring is exponential. The tests check the closed-form truths under
//! independent and covariate-dependent (coarsening at random) censoring, that the unadjusted
//! contrast is biased where the adjusted one is not, that every refusal carries its own
//! detail, and that the reported standard error is the standard deviation of the retained
//! scores.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    reason = "test fixtures compare exact level codes and small counts"
)]

use antecedent_core::ExecutionContext;
use antecedent_estimate::EstimationError;
use antecedent_estimate::observational_survival::{
    CensoringAssumption, ObservationalSurvival, SurvivalData,
};

#[path = "common/observational_effects.rs"]
mod dgp;

use dgp::{
    Censoring, HORIZON, SurvivalStudy as Study, survival_study as study, survival_truth as truth,
};

fn run(
    data: &Study,
    censoring: CensoringAssumption,
    seed: u64,
) -> Result<
    antecedent_estimate::observational_survival::ObservationalSurvivalEstimate,
    EstimationError,
> {
    ObservationalSurvival::new(1.0, 0.0, HORIZON, censoring).estimate(
        &SurvivalData {
            duration: &data.duration,
            event: &data.event,
            treatment: &data.treatment,
            covariates: &[&data.z1, &data.z2],
        },
        &ExecutionContext::for_tests(seed),
    )
}

/// Kaplan-Meier survival at `t` and the restricted mean through `t` on the rows `keep`.
fn kaplan_meier(data: &Study, keep: impl Fn(usize) -> bool, t: f64) -> (f64, f64) {
    let mut rows: Vec<(f64, bool)> = (0..data.duration.len())
        .filter(|&i| keep(i))
        .map(|i| (data.duration[i], data.event[i] == 1.0))
        .collect();
    rows.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (mut survival, mut rmst, mut previous) = (1.0, 0.0, 0.0);
    let mut at_risk = rows.len() as f64;
    for (time, is_event) in rows {
        if time > t {
            break;
        }
        rmst += survival * (time - previous);
        previous = time;
        if is_event {
            survival *= 1.0 - 1.0 / at_risk;
        }
        at_risk -= 1.0;
    }
    (survival, rmst + survival * (t - previous))
}

fn within(estimate: f64, se: f64, truth: f64, multiples: f64) -> bool {
    (estimate - truth).abs() <= multiples * se
}

fn detail_of(error: &EstimationError) -> String {
    error.to_string()
}

#[test]
fn closed_form_truths_are_recovered_under_independent_censoring() {
    let data = study(4000, 11, Censoring::Independent(0.15), 1.0);
    let censored = data.event.iter().filter(|&&d| d == 0.0).count() as f64 / 4000.0;
    assert!((0.2..0.5).contains(&censored), "censoring share {censored}");
    let estimate = run(&data, CensoringAssumption::IndependentWithinArm, 3).unwrap();
    let (s1, r1) = truth(1.0, HORIZON);
    let (s0, r0) = truth(0.0, HORIZON);
    assert_eq!((estimate.active.level, estimate.control.level), (1.0, 0.0));
    assert!(within(estimate.active.survival, estimate.active.survival_se, s1, 4.0));
    assert!(within(estimate.control.survival, estimate.control.survival_se, s0, 4.0));
    assert!(within(estimate.active.rmst, estimate.active.rmst_se, r1, 4.0));
    assert!(within(estimate.control.rmst, estimate.control.rmst_se, r0, 4.0));
    let contrast = estimate.survival_difference;
    assert!(within(contrast.estimate, contrast.se, s1 - s0, 4.0), "{contrast:?} vs {}", s1 - s0);
    let contrast = estimate.rmst_difference;
    assert!(within(contrast.estimate, contrast.se, r1 - r0, 4.0), "{contrast:?} vs {}", r1 - r0);
    assert!(estimate.survival_difference.se > 0.0 && estimate.survival_difference.se < 0.05);
}

#[test]
fn covariate_dependent_censoring_is_handled_under_coarsening_at_random() {
    let data = study(5000, 12, Censoring::DependsOnCovariates(0.12), 1.6);
    let estimate = run(&data, CensoringAssumption::AtRandomGivenCovariates, 4).unwrap();
    let (s1, r1) = truth(1.0, HORIZON);
    let (s0, r0) = truth(0.0, HORIZON);
    let contrast = estimate.survival_difference;
    assert!(within(contrast.estimate, contrast.se, s1 - s0, 4.0), "{contrast:?} vs {}", s1 - s0);
    let contrast = estimate.rmst_difference;
    assert!(within(contrast.estimate, contrast.se, r1 - r0, 4.0), "{contrast:?} vs {}", r1 - r0);
    // The estimator is doubly robust: the Cox event model is correctly specified here, so the
    // estimate stays on the truth even when the censoring model is the (wrong) independent
    // one. That is the double-robustness guarantee, not a defect of the declaration; the
    // censoring assumption matters when the event model is wrong, which this fixture cannot
    // introduce. Both declarations recover the truth, while the unadjusted Kaplan-Meier
    // contrast, which ignores both the confounding and the covariate-dependent censoring,
    // is biased by more than the adjusted estimates' standard errors.
    let independent = run(&data, CensoringAssumption::IndependentWithinArm, 4).unwrap();
    let contrast = independent.survival_difference;
    assert!(within(contrast.estimate, contrast.se, s1 - s0, 4.0), "{contrast:?} vs {}", s1 - s0);
    let km = |arm: f64| kaplan_meier(&data, |i| data.treatment[i] == arm, HORIZON).0;
    let naive = km(1.0) - km(0.0);
    assert!(
        (naive - (s1 - s0)).abs() > 0.03
            && (naive - (s1 - s0)).abs() > 6.0 * estimate.survival_difference.se,
        "naive {naive} truth {}",
        s1 - s0
    );
}

#[test]
fn the_unadjusted_contrast_is_confounded_where_the_adjusted_one_is_not() {
    let data = study(6000, 13, Censoring::Independent(0.1), 1.6);
    let (s1, r1) = truth(1.0, HORIZON);
    let (s0, r0) = truth(0.0, HORIZON);
    let naive_s = kaplan_meier(&data, |i| data.treatment[i] == 1.0, HORIZON).0
        - kaplan_meier(&data, |i| data.treatment[i] == 0.0, HORIZON).0;
    let naive_r = kaplan_meier(&data, |i| data.treatment[i] == 1.0, HORIZON).1
        - kaplan_meier(&data, |i| data.treatment[i] == 0.0, HORIZON).1;
    let estimate = run(&data, CensoringAssumption::IndependentWithinArm, 5).unwrap();
    assert!((naive_s - (s1 - s0)).abs() > 0.05, "naive {naive_s} truth {}", s1 - s0);
    assert!((naive_r - (r1 - r0)).abs() > 0.15, "naive {naive_r} truth {}", r1 - r0);
    let contrast = estimate.survival_difference;
    assert!(within(contrast.estimate, contrast.se, s1 - s0, 4.0));
    let contrast = estimate.rmst_difference;
    assert!(within(contrast.estimate, contrast.se, r1 - r0, 4.0));
}

#[test]
fn the_standard_error_is_the_score_standard_deviation_and_the_fit_is_seeded() {
    let data = study(1500, 14, Censoring::Independent(0.1), 1.0);
    let first = run(&data, CensoringAssumption::IndependentWithinArm, 9).unwrap();
    let again = run(&data, CensoringAssumption::IndependentWithinArm, 9).unwrap();
    assert_eq!(first, again);
    let other_folds = run(&data, CensoringAssumption::IndependentWithinArm, 10).unwrap();
    assert_ne!(first.survival_difference.estimate, other_folds.survival_difference.estimate);
    let scores = &first.survival_difference_scores;
    let n = scores.len() as f64;
    let mean = scores.iter().sum::<f64>() / n;
    let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (n - 1.0);
    assert!((mean - first.survival_difference.estimate).abs() < 1e-12);
    assert!((variance.sqrt() / n.sqrt() - first.survival_difference.se).abs() < 1e-12);
    assert_eq!(first.row_index.len(), 1500);
    assert_eq!(first.n_obs, 1500);
    assert_eq!(first.active.rows + first.control.rows, 1500);
}

#[test]
fn missing_cells_are_dropped_and_counted() {
    let mut data = study(1200, 15, Censoring::Independent(0.1), 1.0);
    data.z1[3] = f64::NAN;
    data.duration[10] = f64::NAN;
    let estimate = run(&data, CensoringAssumption::IndependentWithinArm, 6).unwrap();
    assert_eq!(estimate.rows_dropped_missing, 2);
    assert_eq!(estimate.n_obs, 1198);
    assert!(
        !estimate.row_index.contains(&antecedent_data::GlobalId::new(3))
            && !estimate.row_index.contains(&antecedent_data::GlobalId::new(10))
    );
}

#[test]
fn a_horizon_beyond_an_arms_follow_up_is_refused_with_its_own_detail() {
    let data = study(800, 16, Censoring::Independent(0.1), 1.0);
    let too_far =
        ObservationalSurvival::new(1.0, 0.0, 1000.0, CensoringAssumption::IndependentWithinArm);
    let error = too_far
        .estimate(
            &SurvivalData {
                duration: &data.duration,
                event: &data.event,
                treatment: &data.treatment,
                covariates: &[&data.z1],
            },
            &ExecutionContext::for_tests(1),
        )
        .unwrap_err();
    assert!(detail_of(&error).contains("survival_effect.horizon_beyond_follow_up"), "{error}");
}

#[test]
fn near_deterministic_treatment_is_a_treatment_positivity_refusal() {
    let data = study(1500, 17, Censoring::Independent(0.1), 12.0);
    let error = run(&data, CensoringAssumption::IndependentWithinArm, 2).unwrap_err();
    assert!(detail_of(&error).contains("survival_effect.treatment_positivity"), "{error}");
}

#[test]
fn heavy_censoring_before_the_horizon_is_a_censoring_support_refusal() {
    // A censoring rate of 3 leaves about exp(-6) uncensored through the horizon of 2.
    let data = study(4000, 18, Censoring::Independent(3.0), 1.0);
    let error = run(&data, CensoringAssumption::IndependentWithinArm, 2).unwrap_err();
    let text = detail_of(&error);
    assert!(
        text.contains("survival_effect.censoring_support")
            || text.contains("survival_effect.horizon_beyond_follow_up"),
        "{text}"
    );
    // A shorter horizon that the censoring leaves support for is estimated.
    let short =
        ObservationalSurvival::new(1.0, 0.0, 0.2, CensoringAssumption::IndependentWithinArm);
    let ok = short.estimate(
        &SurvivalData {
            duration: &data.duration,
            event: &data.event,
            treatment: &data.treatment,
            covariates: &[&data.z1, &data.z2],
        },
        &ExecutionContext::for_tests(2),
    );
    assert!(ok.is_ok(), "{:?}", ok.err());
}

#[test]
fn undeclared_levels_invalid_events_and_missing_covariates_are_refused() {
    let mut data = study(600, 19, Censoring::Independent(0.1), 1.0);
    data.treatment[5] = 2.0;
    let error = run(&data, CensoringAssumption::IndependentWithinArm, 1).unwrap_err();
    assert!(detail_of(&error).contains("survival_effect.undeclared_level"), "{error}");

    let mut data = study(600, 19, Censoring::Independent(0.1), 1.0);
    data.event[5] = 2.0;
    let error = run(&data, CensoringAssumption::IndependentWithinArm, 1).unwrap_err();
    assert!(detail_of(&error).contains("survival_effect.invalid_event"), "{error}");

    let data = study(600, 19, Censoring::Independent(0.1), 1.0);
    let error =
        ObservationalSurvival::new(1.0, 0.0, HORIZON, CensoringAssumption::AtRandomGivenCovariates)
            .estimate(
                &SurvivalData {
                    duration: &data.duration,
                    event: &data.event,
                    treatment: &data.treatment,
                    covariates: &[],
                },
                &ExecutionContext::for_tests(1),
            )
            .unwrap_err();
    assert!(detail_of(&error).contains("survival_effect.censoring_covariates_missing"), "{error}");

    let error =
        ObservationalSurvival::new(1.0, 1.0, HORIZON, CensoringAssumption::IndependentWithinArm)
            .estimate(
                &SurvivalData {
                    duration: &data.duration,
                    event: &data.event,
                    treatment: &data.treatment,
                    covariates: &[&data.z1],
                },
                &ExecutionContext::for_tests(1),
            )
            .unwrap_err();
    assert!(detail_of(&error).contains("survival_effect.invalid_levels"), "{error}");
}

/// Repeated draws: the estimator is unbiased for the closed-form truth and its standard error
/// tracks the spread of the estimates. A finite check, not a coverage claim.
#[test]
fn estimates_are_centred_on_the_truth_over_repeated_samples() {
    let (s1, _) = truth(1.0, HORIZON);
    let (s0, _) = truth(0.0, HORIZON);
    let replicates = 40;
    let estimates: Vec<(f64, f64)> = (0..replicates)
        .map(|k| {
            let data = study(1000, 500 + k, Censoring::DependsOnCovariates(0.1), 1.0);
            let estimate = run(&data, CensoringAssumption::AtRandomGivenCovariates, k).unwrap();
            (estimate.survival_difference.estimate, estimate.survival_difference.se)
        })
        .collect();
    let mean = estimates.iter().map(|e| e.0).sum::<f64>() / replicates as f64;
    let spread = (estimates.iter().map(|e| (e.0 - mean).powi(2)).sum::<f64>()
        / (replicates - 1) as f64)
        .sqrt();
    let mean_se = estimates.iter().map(|e| e.1).sum::<f64>() / replicates as f64;
    assert!(
        (mean - (s1 - s0)).abs() < 3.0 * spread / (replicates as f64).sqrt(),
        "mean {mean} truth {} spread {spread}",
        s1 - s0
    );
    assert!(mean_se > 0.7 * spread && mean_se < 1.4 * spread, "se {mean_se} spread {spread}");
}
