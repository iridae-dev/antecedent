use super::*;

const DRAWS: usize = 40_000;

fn noise(i: usize) -> f64 {
    ((i * 7) % 11) as f64 / 10.0 - 0.5
}

/// Fourteen subjects over periods 1..=4: eight never treated, six adopting in period 3 with an
/// effect of 2 from adoption onward.
fn panel() -> (Vec<f64>, Vec<String>, Vec<i64>, Vec<i64>) {
    let (mut y, mut subjects, mut periods, mut cohorts) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 0..14_usize {
        let cohort = if i < 8 { 0 } else { 3 };
        for t in 1..=4_i64 {
            let effect = if cohort == 3 && t >= 3 { 2.0 } else { 0.0 };
            y.push(
                1.0 + 0.5 * t as f64
                    + 0.3 * (i % 4) as f64
                    + effect
                    + noise(i * 4 + usize::try_from(t).unwrap_or_default()),
            );
            subjects.push(format!("s{i}"));
            periods.push(t);
            cohorts.push(cohort);
        }
    }
    (y, subjects, periods, cohorts)
}

#[test]
fn event_time_posteriors_match_the_analytic_group_coefficient_posterior() {
    let (y, subjects, periods, cohorts) = panel();
    let fit = fit_bayesian_event_study(
        &y,
        &subjects,
        &periods,
        &cohorts,
        &subjects,
        DidPrior::diffuse(),
        DrawSpec::new(DRAWS, 3),
    )
    .unwrap();
    // Reference values: closed-form NIG posterior of the cohort coefficient in the change score
    // from reference period 2, computed outside the implementation.
    let expected = [
        (-2, -0.274_996_666_7, 0.297_086_764_6),
        (0, 1.954_116_078_1, 0.307_238_883_3),
        (1, 1.908_290_487_5, 0.288_740_001_4),
    ];
    assert_eq!(fit.len(), 3);
    for (effect, (event_time, mean, sd)) in fit.iter().zip(expected) {
        assert_eq!(effect.event_time, event_time);
        assert_eq!((effect.treated_subjects, effect.comparison_subjects), (6, 8));
        assert!((effect.mean - mean).abs() < 0.012, "{event_time}: {}", effect.mean);
        assert!((effect.sd - sd).abs() < 0.01, "{event_time}: {}", effect.sd);
    }
    // The pre-adoption lead is a descriptive posterior and the post-adoption effect is
    // separated from it: the lead interval covers zero and the adoption interval excludes it.
    assert!(fit[0].lower < 0.0 && fit[0].upper > 0.0);
    assert!(fit[1].lower > 0.0);
}

#[test]
fn clustered_unbalanced_and_uncontrolled_panels_are_refused() {
    let (y, subjects, periods, cohorts) = panel();
    let shared: Vec<String> = subjects.iter().map(|s| format!("c{}", s.len() % 2)).collect();
    let error = fit_bayesian_event_study(
        &y,
        &subjects,
        &periods,
        &cohorts,
        &shared,
        DidPrior::diffuse(),
        DrawSpec::new(100, 1),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("bayesian_did_dependent_units"), "{error}");

    let error = fit_bayesian_event_study(
        &y[1..],
        &subjects[1..],
        &periods[1..],
        &cohorts[1..],
        &subjects[1..],
        DidPrior::diffuse(),
        DrawSpec::new(100, 1),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("balanced panel"), "{error}");

    let treated_only: Vec<i64> = cohorts.iter().map(|_| 3).collect();
    let error = fit_bayesian_event_study(
        &y,
        &subjects,
        &periods,
        &treated_only,
        &subjects,
        DidPrior::diffuse(),
        DrawSpec::new(100, 1),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("never-treated"), "{error}");
}
