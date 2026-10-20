use super::*;

const DRAWS: usize = 40_000;

fn noise(i: usize) -> f64 {
    ((i * 7) % 11) as f64 / 10.0 - 0.5
}

fn unit_ids(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("u{i}")).collect()
}

/// Four cells of six rows: `y = 2 + G + 0.5 Post + 1.5 G Post + noise`.
fn cross_section() -> (Vec<f64>, Vec<bool>, Vec<bool>) {
    let (mut y, mut group, mut post) = (Vec::new(), Vec::new(), Vec::new());
    let mut i = 0;
    for g in [false, true] {
        for p in [false, true] {
            for _ in 0..6 {
                y.push(
                    2.0 + f64::from(g) + 0.5 * f64::from(p) + 1.5 * f64::from(g && p) + noise(i),
                );
                group.push(g);
                post.push(p);
                i += 1;
            }
        }
    }
    (y, group, post)
}

/// Twenty-four subjects: the second twelve are treated; the change score also loads on `x`.
fn panel() -> (Vec<f64>, Vec<f64>, Vec<bool>, Vec<f64>) {
    let (mut pre, mut post, mut treated, mut x) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for j in 0..24 {
        let t = j >= 12;
        let xj = ((j * 5) % 7) as f64 / 3.0;
        let before = 1.0 + 0.3 * (j % 3) as f64 + noise(j);
        pre.push(before);
        post.push(before + 0.7 + 2.0 * f64::from(t) + 0.8 * xj + noise(j + 3));
        treated.push(t);
        x.push(xj);
    }
    (pre, post, treated, x)
}

#[test]
fn cross_section_posterior_matches_the_analytic_normal_inverse_gamma_posterior() {
    let (y, group, post) = cross_section();
    let ids = unit_ids(y.len());
    // Reference: conjugate NIG posterior of the interaction coefficient computed in closed form
    // (mean m_n, marginal variance b_n / (a_n - 1) V_n[3][3]) outside the implementation.
    let diffuse = fit_bayesian_cross_section_did(
        &y,
        &group,
        &post,
        &ids,
        DidPrior::diffuse(),
        DrawSpec::new(DRAWS, 5),
    )
    .unwrap();
    assert!((diffuse.mean - 1.316_600_562_8).abs() < 0.01, "{}", diffuse.mean);
    assert!((diffuse.sd - 0.266_700_253_3).abs() < 0.008, "{}", diffuse.sd);
    assert!(diffuse.lower < diffuse.mean && diffuse.mean < diffuse.upper);
    assert_eq!((diffuse.n_treated, diffuse.n_comparison), (12, 12));
    // The cell-mean DiD of the data is 1.3166667: a diffuse prior must reproduce it.
    assert!((diffuse.mean - 1.316_666_667).abs() < 0.01);
    // An informative prior shrinks the effect toward zero and widens nothing silently: the
    // posterior must move to the analytic value for that prior, not stay at the data value.
    let tight = DidPrior { coefficient_scale: 0.5, ..DidPrior::diffuse() };
    let shrunk =
        fit_bayesian_cross_section_did(&y, &group, &post, &ids, tight, DrawSpec::new(DRAWS, 6))
            .unwrap();
    assert!((shrunk.mean - 0.773_636_363_6).abs() < 0.015, "{}", shrunk.mean);
    assert!((shrunk.sd - 0.423_209_862_4).abs() < 0.015, "{}", shrunk.sd);
}

#[test]
fn panel_posterior_matches_the_analytic_change_score_posterior() {
    let (pre, post, treated, _) = panel();
    let ids = unit_ids(24);
    let fit = fit_bayesian_panel_did(
        &pre,
        &post,
        &treated,
        &ids,
        DidPrior::diffuse(),
        DrawSpec::new(DRAWS, 7),
    )
    .unwrap();
    assert!((fit.mean - 1.944_424_884_6).abs() < 0.01, "{}", fit.mean);
    assert!((fit.sd - 0.247_048_411_7).abs() < 0.008, "{}", fit.sd);
    assert_eq!((fit.n_treated, fit.n_comparison), (12, 12));
}

#[test]
fn augmented_posterior_matches_the_analytic_two_arm_regression_posterior() {
    let (pre, post, treated, x) = panel();
    let ids = unit_ids(24);
    let fit = fit_bayesian_augmented_panel_did(
        &pre,
        &post,
        &treated,
        &[x.clone()],
        &ids,
        DidPrior::diffuse(),
        DrawSpec::new(DRAWS, 8),
    )
    .unwrap();
    // Reference: per-arm NIG regressions on [1, x], contrast evaluated at the treated x mean,
    // variance the sum of the two independent Student-t marginal variances.
    assert!((fit.mean - 1.964_181_490_1).abs() < 0.01, "{}", fit.mean);
    assert!((fit.sd - 0.140_912_027_5).abs() < 0.006, "{}", fit.sd);
    // Adjusting for the covariate that drives the change score must tighten the posterior
    // relative to the unadjusted change-score model.
    let unadjusted = fit_bayesian_panel_did(
        &pre,
        &post,
        &treated,
        &ids,
        DidPrior::diffuse(),
        DrawSpec::new(DRAWS, 7),
    )
    .unwrap();
    assert!(fit.sd < unadjusted.sd);
}

#[test]
fn dependent_units_and_thin_designs_are_refused_with_the_next_attempt() {
    let (y, group, post) = cross_section();
    let shared: Vec<String> = (0..y.len()).map(|i| format!("c{}", i / 2)).collect();
    let error = fit_bayesian_cross_section_did(
        &y,
        &group,
        &post,
        &shared,
        DidPrior::diffuse(),
        DrawSpec::new(100, 1),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("bayesian_did_dependent_units"), "{error}");
    assert!(error.contains("hierarchical"), "{error}");

    let ids = unit_ids(y.len());
    let all_comparison = vec![false; y.len()];
    let error = fit_bayesian_cross_section_did(
        &y,
        &all_comparison,
        &post,
        &ids,
        DidPrior::diffuse(),
        DrawSpec::new(100, 1),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("bayesian_did_cell_support"), "{error}");

    let (pre, change, _, x) = panel();
    let ids = unit_ids(24);
    let few: Vec<bool> = (0..24).map(|j| j < 2).collect();
    let error = fit_bayesian_augmented_panel_did(
        &pre,
        &change,
        &few,
        &[x],
        &ids,
        DidPrior::diffuse(),
        DrawSpec::new(100, 1),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("bayesian_did_arm_support"), "{error}");
}

#[test]
fn invalid_priors_and_non_finite_outcomes_are_refused() {
    let (pre, post, treated, _) = panel();
    let ids = unit_ids(24);
    let bad = DidPrior { coefficient_scale: 0.0, ..DidPrior::diffuse() };
    assert!(
        fit_bayesian_panel_did(&pre, &post, &treated, &ids, bad, DrawSpec::new(100, 1)).is_err()
    );
    let mut broken = post.clone();
    broken[3] = f64::NAN;
    assert!(
        fit_bayesian_panel_did(
            &pre,
            &broken,
            &treated,
            &ids,
            DidPrior::diffuse(),
            DrawSpec::new(100, 1)
        )
        .is_err()
    );
}
