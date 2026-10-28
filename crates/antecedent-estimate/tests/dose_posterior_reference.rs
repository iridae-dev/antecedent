//! The local-quadratic posterior mean and the fixed group-dose policy posterior against 40-digit
//! references.
//!
//! `oracle_data/dose_posterior_reference.py` (frozen in the JSON beside it) forms the Gaussian
//! weighted normal equations of the local quadratic, the plug-in residual variance
//! `RSS / (W - 3)` and the ridge solve that leaves the intercept unpenalized; and, for three
//! groups of unequal size with a degree-three polynomial curve each, the conjugate coefficient
//! posterior, the Epanechnikov window functional by adaptive quadrature and the exact mean and
//! variance of the policy value, the reference value and their difference, for a known residual
//! variance and for an inverse-gamma prior on it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::continuous_dose::DoseRules;
use antecedent_estimate::continuous_dose_posterior::{
    DoseCurvePrior, DoseResponseRows, PolicyWindow, fixed_dose_policy_posterior,
};
use antecedent_estimate::dose_basis_regression::ResidualPrior;
use antecedent_estimate::posterior_mean::{
    LocalQuadraticNamespace, PosteriorRequest, consume_local_quadratic_posterior,
    local_quadratic_posterior,
};
use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(include_str!("oracle_data/dose_posterior_reference.json")).unwrap()
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn text(v: &Value) -> f64 {
    v.as_str().unwrap().parse().unwrap()
}

fn close(got: f64, want: f64, rel: f64, what: &str) {
    let tol = rel * want.abs().max(1e-300);
    assert!((got - want).abs() <= tol, "{what}: got {got} want {want} tol {tol}");
}

#[test]
fn local_quadratic_level_and_derivative_equal_the_penalized_weighted_fit() {
    for (index, case) in fixture()["local_quadratic"].as_array().unwrap().iter().enumerate() {
        let dose = floats(&case["dose"]);
        let outcome = floats(&case["outcome"]);
        let at = case["at"].as_f64().unwrap();
        let bandwidth = case["bandwidth"].as_f64().unwrap();
        let scale = case["scale"].as_f64().unwrap();
        for namespace in [LocalQuadraticNamespace::QuadraticDose, LocalQuadraticNamespace::DoseGrid]
        {
            let (level, derivative) = local_quadratic_posterior(
                &dose,
                &outcome,
                at,
                bandwidth,
                PosteriorRequest::IndependentNormal { scale },
                namespace,
            )
            .unwrap();
            close(level, text(&case["level"]), 1e-9, &format!("level {index}"));
            close(derivative, text(&case["derivative"]), 1e-9, &format!("derivative {index}"));
        }
        // The consumer replays the same number from the retained normal equations.
        let gram_rows = case["gram"].as_array().unwrap();
        let mut gram = [[0.0; 3]; 3];
        for (a, row) in gram_rows.iter().enumerate() {
            for (c, v) in row.as_array().unwrap().iter().enumerate() {
                gram[a][c] = text(v);
            }
        }
        let rhs_values: Vec<f64> = case["rhs"].as_array().unwrap().iter().map(text).collect();
        let (level, derivative) = consume_local_quadratic_posterior(
            gram,
            [rhs_values[0], rhs_values[1], rhs_values[2]],
            text(&case["weight_sum"]),
            text(&case["weighted_rss"]),
            scale,
        )
        .unwrap();
        close(level, text(&case["level"]), 1e-8, &format!("replayed level {index}"));
        close(derivative, text(&case["derivative"]), 1e-8, &format!("replayed derivative {index}"));
    }
}

#[test]
fn a_loose_prior_approaches_the_unpenalized_fit_and_a_tight_prior_flattens_the_curvature() {
    let fx = fixture();
    let cases = fx["local_quadratic"].as_array().unwrap();
    let wide = &cases[3];
    let tight = &cases[0];
    let run = |case: &Value, scale: f64| {
        local_quadratic_posterior(
            &floats(&case["dose"]),
            &floats(&case["outcome"]),
            case["at"].as_f64().unwrap(),
            case["bandwidth"].as_f64().unwrap(),
            PosteriorRequest::IndependentNormal { scale },
            LocalQuadraticNamespace::QuadraticDose,
        )
        .unwrap()
    };
    let (level, derivative) = run(wide, 1e8);
    close(level, text(&wide["unpenalized_level"]), 1e-6, "flat prior level");
    close(derivative, text(&wide["unpenalized_derivative"]), 1e-6, "flat prior derivative");
    // The unpenalized intercept stays at the weighted mean when the slope and curvature are
    // shrunk to zero.
    let (level, derivative) = run(tight, 1e-9);
    assert!(derivative.abs() < 1e-6, "{derivative}");
    assert!(level.is_finite());
}

struct PolicyCase {
    dose: Vec<f64>,
    outcome: Vec<f64>,
    groups: Vec<String>,
    policy: Vec<(String, f64)>,
    reference: Vec<(String, f64)>,
    scales: Vec<f64>,
    support: (f64, f64),
    bandwidth: f64,
}

fn policy_case(fx: &Value) -> PolicyCase {
    let p = &fx["policy"];
    let mut dose = Vec::new();
    let mut outcome = Vec::new();
    let mut groups = Vec::new();
    for g in p["groups"].as_array().unwrap() {
        let name = g["name"].as_str().unwrap().to_owned();
        for (d, y) in floats(&g["dose"]).into_iter().zip(floats(&g["outcome"])) {
            dose.push(d);
            outcome.push(y);
            groups.push(name.clone());
        }
    }
    let rule = |key: &str| -> Vec<(String, f64)> {
        let mut out: Vec<(String, f64)> = p["rules"][key]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.as_f64().unwrap()))
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    };
    let support = floats(&p["support"]);
    PolicyCase {
        dose,
        outcome,
        groups,
        policy: rule("policy"),
        reference: rule("reference"),
        scales: floats(&p["scales"]),
        support: (support[0], support[1]),
        bandwidth: p["bandwidth"].as_f64().unwrap(),
    }
}

fn run_policy(
    case: &PolicyCase,
    residual: ResidualPrior,
    draws: usize,
) -> antecedent_estimate::continuous_dose_posterior::DosePolicyPosterior {
    fixed_dose_policy_posterior(
        &DoseResponseRows { outcome: &case.outcome, dose: &case.dose, groups: &case.groups },
        &DoseRules { policy: &case.policy, reference: &case.reference },
        &DoseCurvePrior { support: case.support, coefficient_scales: &case.scales, residual },
        &PolicyWindow { bandwidth: case.bandwidth, min_local_support: 2, draws, seed: 20261028 },
    )
    .unwrap()
}

#[test]
fn known_variance_policy_moments_equal_the_exact_conjugate_window_integrals() {
    let fx = fixture();
    let case = policy_case(&fx);
    let known = &fx["policy"]["known"];
    let posterior = run_policy(&case, ResidualPrior::KnownVariance(0.3), 100_000);
    let means = posterior.exact_means.unwrap();
    let variances = posterior.exact_variances.unwrap();
    for slot in 0..3 {
        close(means[slot], text(&known["means"][slot]), 1e-10, "exact mean");
        close(variances[slot], text(&known["variances"][slot]), 1e-10, "exact variance");
    }
    assert_eq!(
        posterior.minimum_local_rows,
        fx["policy"]["minimum_local_rows"].as_u64().unwrap() as usize
    );
    let total: f64 = posterior.group_fractions.iter().map(|(_, f)| f).sum();
    close(total, 1.0, 1e-14, "fractions");
    close(posterior.group_fractions[1].1, 11.0 / 27.0, 1e-14, "fraction B");
    // The draws are paired and Gaussian: their moments match the exact ones within Monte Carlo
    // error (five standard errors of the mean, 3% of the variance at 100,000 draws).
    let n = posterior.draws as f64;
    let moments = [(&posterior.policy_draws, 0usize), (&posterior.reference_draws, 1usize)];
    for (draws, slot) in moments {
        let mean = draws.iter().sum::<f64>() / n;
        let var = draws.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
        assert!((mean - means[slot]).abs() < 5.0 * (variances[slot] / n).sqrt(), "mean {slot}");
        close(var, variances[slot], 0.03, "draw variance");
    }
    let diff: Vec<f64> =
        posterior.policy_draws.iter().zip(&posterior.reference_draws).map(|(p, r)| p - r).collect();
    let mean = diff.iter().sum::<f64>() / n;
    let var = diff.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    close(var, variances[2], 0.03, "paired difference variance");
    close(posterior.incremental_mean, mean, 1e-12, "incremental mean is the draw mean");
    // The published equal-tailed 95% intervals are the Gaussian ones to within the Monte Carlo
    // error of a 2.5% quantile (about 0.01 standard deviations at 100,000 draws; 0.05 allowed).
    for (interval, slot) in [
        (posterior.policy_interval_95, 0usize),
        (posterior.reference_interval_95, 1),
        (posterior.incremental_interval_95, 2),
    ] {
        let sd = variances[slot].sqrt();
        assert!((interval[0] - (means[slot] - 1.959_963_984_540_054 * sd)).abs() < 0.05 * sd);
        assert!((interval[1] - (means[slot] + 1.959_963_984_540_054 * sd)).abs() < 0.05 * sd);
    }
    let positive = diff.iter().filter(|v| **v > 0.0).count() as f64 / n;
    close(posterior.probability_incremental_positive, positive, 1e-12, "probability positive");
}

#[test]
fn inverse_gamma_residual_prior_gives_the_student_t_moments() {
    let fx = fixture();
    let case = policy_case(&fx);
    let want = &fx["policy"]["inverse_gamma_moments"];
    let ig = floats(&fx["policy"]["inverse_gamma"]);
    let posterior =
        run_policy(&case, ResidualPrior::InverseGamma { shape: ig[0], scale: ig[1] }, 100_000);
    assert!(posterior.exact_means.is_none() && posterior.exact_variances.is_none());
    let n = posterior.draws as f64;
    let sd = text(&want["variances"][0]).sqrt();
    let mean = posterior.policy_draws.iter().sum::<f64>() / n;
    assert!((mean - text(&want["means"][0])).abs() < 5.0 * sd / n.sqrt(), "mean {mean}");
    let var = posterior.policy_draws.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    // Student-t with at least 13 degrees of freedom: the sample variance has a relative
    // standard error near 0.01, so 5% is five standard errors.
    close(var, text(&want["variances"][0]), 0.05, "student-t variance");
    let ref_var = {
        let m = posterior.reference_draws.iter().sum::<f64>() / n;
        posterior.reference_draws.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n - 1.0)
    };
    close(ref_var, text(&want["variances"][1]), 0.05, "reference variance");
}
