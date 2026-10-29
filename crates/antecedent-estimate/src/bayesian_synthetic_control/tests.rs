use super::*;

const UNITS: [&str; 4] = ["d1", "d2", "d3", "t"];

fn donors_at(t: i64) -> [f64; 3] {
    let tf = t as f64;
    [
        1.0 + 0.5 * tf + 0.3 * (((t * 5) % 7) as f64 / 3.0 - 1.0),
        3.0 - 0.2 * tf + 0.3 * (((t * 3) % 8) as f64 / 3.5 - 1.0),
        2.0 + 0.4 * ((t * 3) % 5) as f64,
    ]
}

/// Three donors and a treated unit that is `0.6 d1 + 0.4 d2` plus noise, with a post-period
/// effect of 2 from period 9 on. `lift` shifts the treated unit away from the donors' hull.
fn panel(lift: f64) -> (Vec<f64>, Vec<String>, Vec<i64>) {
    let (mut y, mut units, mut periods) = (Vec::new(), Vec::new(), Vec::new());
    for t in 1..=11_i64 {
        let d = donors_at(t);
        let treated = 0.6 * d[0]
            + 0.4 * d[1]
            + 0.15 * (((t * 11) % 13) as f64 / 6.0 - 1.0)
            + if t >= 9 { 2.0 } else { 0.0 }
            + lift;
        for (name, value) in UNITS.iter().zip([d[0], d[1], d[2], treated]) {
            y.push(value);
            units.push((*name).to_string());
            periods.push(t);
        }
    }
    (y, units, periods)
}

fn fit(
    lift: f64,
    spec: &BayesianSyntheticControlSpec<'_>,
) -> Result<BayesianSyntheticControlEstimate, EstimationError> {
    let (y, units, periods) = panel(lift);
    fit_bayesian_synthetic_control(&y, &units, &periods, spec, DrawSpec::new(8_000, 21))
}

#[test]
fn posterior_matches_the_simplex_quadrature_under_independent_errors() {
    let fit = fit(0.0, &BayesianSyntheticControlSpec::new("t", 9)).unwrap();
    // Reference: the posterior over the weight triangle integrated on a fine grid, with the
    // residual variance integrated out analytically, computed outside the implementation.
    // Observed deviations from the quadrature: 0.0002 (mean), 0.002 (sd); the Monte Carlo
    // standard error of the mean is about 0.002 at the observed effective sample size.
    assert!((fit.mean - 2.015_77).abs() < 0.005, "{}", fit.mean);
    assert!((fit.sd - 0.142_66).abs() < 0.006, "{}", fit.sd);
    let expected_means = [0.576_5, 0.362_3, 0.061_2];
    let expected_sds = [0.031_9, 0.033_3, 0.046_2];
    for (j, ((name, mean), sd)) in
        fit.donor_weight_means.iter().zip(&fit.donor_weight_sds).enumerate()
    {
        assert_eq!(name, ["d1", "d2", "d3"][j]);
        assert!((mean - expected_means[j]).abs() < 0.004, "{name}: {mean}");
        assert!((sd - expected_sds[j]).abs() < 0.003, "{name}: {sd}");
    }
    let total: f64 = fit.donor_weight_means.iter().map(|(_, w)| w).sum();
    assert!((total - 1.0).abs() < 1e-9);
    assert!(fit.lower < 2.0 && fit.upper > 2.0);
    assert!((fit.pre_fit_rmse - 0.096_6).abs() < 1e-4, "{}", fit.pre_fit_rmse);
    assert_eq!(fit.pre_periods_outside_donor_range, 0);
    assert_eq!((fit.n_pre_periods, fit.n_post_periods), (8, 3));
    assert!(fit.acceptance_rate > 0.2 && fit.acceptance_rate < 0.9);
    assert!(fit.max_split_rhat < 1.05 && fit.min_bulk_ess >= 100.0);
}

#[test]
fn ar1_errors_carry_the_last_residual_into_the_post_period_counterfactual() {
    let spec =
        BayesianSyntheticControlSpec { ar1_rho: 0.5, ..BayesianSyntheticControlSpec::new("t", 9) };
    let fit = fit(0.0, &spec).unwrap();
    // Reference quadrature with the stationary AR(1) whitened likelihood and the conditional
    // post-period error mean and variance. Omitting the carried residual gives 2.0380, not 2.0125.
    assert!((fit.mean - 2.012_52).abs() < 0.006, "{}", fit.mean);
    assert!((fit.sd - 0.183_06).abs() < 0.008, "{}", fit.sd);
    let expected = [0.579_7, 0.381_9, 0.038_4];
    for ((_, mean), want) in fit.donor_weight_means.iter().zip(expected) {
        assert!((mean - want).abs() < 0.004, "{mean} vs {want}");
    }
}

#[test]
fn a_concentrated_dirichlet_prior_pulls_the_weights_toward_equal() {
    let spec = BayesianSyntheticControlSpec {
        dirichlet_concentration: 20.0,
        ..BayesianSyntheticControlSpec::new("t", 9)
    };
    let fit = fit(0.0, &spec).unwrap();
    let expected = [0.443_2, 0.297_0, 0.259_9];
    for ((_, mean), want) in fit.donor_weight_means.iter().zip(expected) {
        assert!((mean - want).abs() < 0.004, "{mean} vs {want}");
    }
    assert!((fit.mean - 2.354_8).abs() < 0.02, "{}", fit.mean);
}

#[test]
fn a_treated_path_outside_the_donor_hull_is_reported_not_repaired() {
    let fit = fit(10.0, &BayesianSyntheticControlSpec::new("t", 9)).unwrap();
    assert_eq!(fit.pre_periods_outside_donor_range, 8);
    assert!(fit.pre_fit_rmse > 5.0, "{}", fit.pre_fit_rmse);
}

#[test]
fn malformed_panels_and_specs_are_refused() {
    let (y, units, periods) = panel(0.0);
    let spec = BayesianSyntheticControlSpec::new("t", 9);
    let few = fit_bayesian_synthetic_control(&y, &units, &periods, &spec, DrawSpec::new(100, 1));
    let text = few.unwrap_err().to_string();
    assert!(text.contains("reason=invalid_argument") && text.contains("at least 400 draws"), "{text}");
    let drop_rows = |keep: &dyn Fn(usize) -> bool| {
        let rows: Vec<usize> = (0..y.len()).filter(|i| keep(*i)).collect();
        (
            rows.iter().map(|&i| y[i]).collect::<Vec<_>>(),
            rows.iter().map(|&i| units[i].clone()).collect::<Vec<_>>(),
            rows.iter().map(|&i| periods[i]).collect::<Vec<_>>(),
        )
    };
    let (uy, uu, up) = drop_rows(&|i| i != 5);
    let unbalanced = fit_bayesian_synthetic_control(&uy, &uu, &up, &spec, DrawSpec::new(8_000, 1));
    assert!(unbalanced.unwrap_err().to_string().contains("balanced panel"));
    let (dy, du, dp) = drop_rows(&|i| units[i] != "d3");
    let thin = fit_bayesian_synthetic_control(&dy, &du, &dp, &spec, DrawSpec::new(8_000, 1));
    assert!(thin.unwrap_err().to_string().contains("three donor units"));
    let unit_root = BayesianSyntheticControlSpec { ar1_rho: 1.0, ..spec };
    let text =
        fit_bayesian_synthetic_control(&y, &units, &periods, &unit_root, DrawSpec::new(8_000, 1))
            .unwrap_err()
            .to_string();
    assert!(text.contains("reason=invalid_argument") && text.contains("|rho| < 1"), "{text}");
    let missing = BayesianSyntheticControlSpec::new("nobody", 9);
    let text =
        fit_bayesian_synthetic_control(&y, &units, &periods, &missing, DrawSpec::new(8_000, 1))
            .unwrap_err()
            .to_string();
    assert!(text.contains("one observed treated unit"), "{text}");
}
