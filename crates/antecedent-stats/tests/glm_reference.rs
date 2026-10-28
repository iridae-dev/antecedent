//! Generalised linear model fits against 50-digit maximum-likelihood references.
//!
//! The frozen values come from `oracle_data/glm_reference.py` (its output is
//! `oracle_data/glm_reference.json`).  Every design, outcome and prior weight is an exact binary64
//! value; the expected coefficients maximise the exact (penalised) log-likelihood, found by
//! damped Fisher scoring in `mpmath` with acceptance decided by the log-likelihood itself.  The
//! designs are the regimes IRLS handles worst: a large covariate offset, column scales six orders
//! apart, counts near 1e4, mostly-zero counts, prior weights spanning nine decades with a zero
//! weight, and completely or quasi-completely separated binary outcomes.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::{
    FaerBackend, GlmDesignRef, GlmFamily, GlmFit, GlmOptions, GlmRefusalKind,
    LeastSquaresWorkspace, NbAlphaPolicy, StatsError, fit_glm, fit_glm_ridge,
    fit_glm_ridge_weighted, fit_glm_weighted,
};
use serde_json::Value;

const REFERENCE: &str = include_str!("oracle_data/glm_reference.json");

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn optional_floats(v: &Value) -> Option<Vec<f64>> {
    (!v.is_null()).then(|| floats(v))
}

/// Largest coefficient error relative to the coefficient (absolute below 1e-9 of the largest).
fn coefficient_error(got: &[f64], want: &[f64]) -> f64 {
    let big = want.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    got.iter().zip(want).fold(0.0_f64, |m, (g, w)| m.max((g - w).abs() / w.abs().max(1e-9 * big)))
}

fn family_of(name: &str) -> GlmFamily {
    match name {
        "logit" => GlmFamily::BinomialLogit,
        "probit" => GlmFamily::BinomialProbit,
        "poisson" => GlmFamily::PoissonLog,
        "nb" => GlmFamily::NegativeBinomial,
        "gaussian" => GlmFamily::GaussianIdentity,
        other => panic!("unknown family {other}"),
    }
}

fn fit_case(
    family: GlmFamily,
    case: &Value,
    weights: Option<&[f64]>,
    options: &GlmOptions,
) -> Result<GlmFit, StatsError> {
    let (n, p) = (case["n"].as_u64().unwrap() as usize, case["p"].as_u64().unwrap() as usize);
    let (x, y) = (floats(&case["x"]), floats(&case["y"]));
    let design = GlmDesignRef { x_colmajor: &x, nrows: n, ncols: p, y: &y };
    let mut ws = LeastSquaresWorkspace::default();
    match weights {
        Some(w) => fit_glm_weighted(family, design, w, &FaerBackend, &mut ws, options),
        None => fit_glm(family, design, &FaerBackend, &mut ws, options),
    }
}

fn tight() -> GlmOptions {
    GlmOptions { ridge_on_separation: None, ..GlmOptions::new(200, 1e-10) }
}

#[test]
fn fits_match_the_exact_maximum_likelihood_estimates() {
    let doc: Value = serde_json::from_str(REFERENCE).unwrap();
    let mut failures = Vec::new();
    for (index, case) in doc["fits"].as_array().unwrap().iter().enumerate() {
        let family_name = case["family"].as_str().unwrap();
        let label = format!(
            "#{index} {family_name}/{}{}",
            case["design"].as_str().unwrap(),
            if case["w"].is_null() { "" } else { "/weighted" }
        );
        let family = family_of(family_name);
        let weights = optional_floats(&case["w"]);
        let mut options = tight();
        if family_name == "nb" {
            options.nb_alpha = NbAlphaPolicy::Fixed(case["alpha"].as_f64().unwrap());
        }
        let fit = match fit_case(family, case, weights.as_deref(), &options) {
            Ok(f) => f,
            Err(e) => {
                failures.push(format!("{label}: refused ({e})"));
                continue;
            }
        };
        let want = floats(&case["coef"]);
        let err = coefficient_error(&fit.coefficients, &want);
        // The deviance field: the weighted residual sum of squares for the Gaussian family,
        // and for every other weighted fit the unweighted deviance at the weighted maximiser.
        let want_dev = if family_name == "gaussian" && weights.is_some() {
            case["deviance_weighted"].as_f64().unwrap()
        } else {
            case["deviance"].as_f64().unwrap()
        };
        let dev_err = ((fit.deviance - want_dev) / want_dev.abs().max(1.0)).abs();
        eprintln!("{label:36} coef {err:.2e} deviance {dev_err:.2e} iterations {}", fit.iterations);
        if !fit.converged || fit.separated || err > 1e-8 || dev_err > 1e-9 {
            failures.push(format!(
                "{label}: converged {} separated {} coef err {err:.2e} deviance err {dev_err:.2e}",
                fit.converged, fit.separated
            ));
        }
        if let Some(a) = fit.nb_alpha {
            if (a - case["alpha"].as_f64().unwrap()).abs() > 0.0 {
                failures.push(format!("{label}: fixed alpha changed to {a}"));
            }
        }
    }
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}

#[test]
fn default_options_reach_the_same_estimates_as_tight_tolerances() {
    // The default convergence rule (relative change of the linear predictor below 1e-8) must
    // leave the coefficients accurate to far better than any standard error.
    let doc: Value = serde_json::from_str(REFERENCE).unwrap();
    let mut failures = Vec::new();
    for (index, case) in doc["fits"].as_array().unwrap().iter().enumerate() {
        let family_name = case["family"].as_str().unwrap();
        if family_name == "nb" {
            continue;
        }
        let weights = optional_floats(&case["w"]);
        let fit =
            fit_case(family_of(family_name), case, weights.as_deref(), &GlmOptions::default());
        let want = floats(&case["coef"]);
        match fit {
            Ok(f) => {
                let err = coefficient_error(&f.coefficients, &want);
                if !f.converged || err > 1e-6 {
                    failures.push(format!(
                        "#{index} {family_name}: converged {} err {err:.2e} after {} iterations",
                        f.converged, f.iterations
                    ));
                }
            }
            Err(e) => failures.push(format!("#{index} {family_name}: refused ({e})")),
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn negative_binomial_dispersion_policies_match_their_definitions() {
    let doc: Value = serde_json::from_str(REFERENCE).unwrap();
    let mut failures = Vec::new();
    // Method of moments: the Pearson excess over the residual degrees of freedom, from the
    // exact Poisson fit, with the same prior weights.
    for case in doc["fits"].as_array().unwrap().iter().filter(|c| c["family"] == "nb") {
        let weights = optional_floats(&case["w"]);
        let options = GlmOptions { nb_alpha: NbAlphaPolicy::MethodOfMoments, ..tight() };
        let fit =
            fit_case(GlmFamily::NegativeBinomial, case, weights.as_deref(), &options).unwrap();
        let want = case["mom_alpha"].as_f64().unwrap();
        let got = fit.nb_alpha.unwrap();
        eprintln!("method of moments: alpha {got:.10} want {want:.10}");
        if ((got - want) / want).abs() > 1e-6 {
            failures.push(format!("method of moments alpha {got} != {want}"));
        }
    }
    // Maximum likelihood: alpha and the coefficients jointly maximise the exact likelihood.
    for (index, case) in doc["nb_dispersion"].as_array().unwrap().iter().enumerate() {
        let weights = optional_floats(&case["w"]);
        let options = GlmOptions {
            nb_alpha: NbAlphaPolicy::NestedMle { max_outer: 200, tol_alpha: 1e-10 },
            ..tight()
        };
        let fit =
            fit_case(GlmFamily::NegativeBinomial, case, weights.as_deref(), &options).unwrap();
        let (want_alpha, want) = (case["alpha"].as_f64().unwrap(), floats(&case["coef"]));
        let alpha_err = ((fit.nb_alpha.unwrap() - want_alpha) / want_alpha).abs();
        let coef_err = coefficient_error(&fit.coefficients, &want);
        eprintln!(
            "nested MLE #{index}: alpha {alpha_err:.2e} coef {coef_err:.2e} converged {}",
            fit.converged
        );
        // The reference alpha is a 30-step golden-section optimum of a flat profile.
        if !fit.converged || alpha_err > 1e-3 || coef_err > 1e-3 {
            failures
                .push(format!("nested MLE #{index}: alpha {alpha_err:.2e}, coef {coef_err:.2e}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn ridge_fits_match_the_exact_penalised_maximiser() {
    let doc: Value = serde_json::from_str(REFERENCE).unwrap();
    let mut failures = Vec::new();
    for (index, case) in doc["ridge"].as_array().unwrap().iter().enumerate() {
        let family = family_of(case["family"].as_str().unwrap());
        let lambda = case["lambda"].as_f64().unwrap();
        let want = floats(&case["coef"]);
        let weights = optional_floats(&case["w"]);
        let (n, p) = (case["n"].as_u64().unwrap() as usize, case["p"].as_u64().unwrap() as usize);
        let (x, y) = (floats(&case["x"]), floats(&case["y"]));
        let design = GlmDesignRef { x_colmajor: &x, nrows: n, ncols: p, y: &y };
        let mut ws = LeastSquaresWorkspace::default();
        let options = GlmOptions::new(500, 1e-12);
        let fit = match &weights {
            Some(w) => {
                fit_glm_ridge_weighted(family, design, w, &FaerBackend, &mut ws, &options, lambda)
            }
            None => fit_glm_ridge(family, design, &FaerBackend, &mut ws, &options, lambda),
        };
        let label = format!(
            "#{index} {}/lambda {lambda:e}/separated {}",
            case["family"], case["separated_data"]
        );
        match fit {
            Ok(f) => {
                let err = coefficient_error(&f.coefficients, &want);
                eprintln!(
                    "{label}: coef {err:.2e} converged {} iterations {}",
                    f.converged, f.iterations
                );
                if !f.converged || !f.penalized || err > 1e-7 {
                    failures.push(format!(
                        "{label}: converged {} penalized {} err {err:.2e}",
                        f.converged, f.penalized
                    ));
                }
            }
            Err(e) => failures.push(format!("{label}: refused ({e})")),
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn separated_logistic_fits_are_flagged_refused_and_ridge_refit_on_request() {
    let doc: Value = serde_json::from_str(REFERENCE).unwrap();
    let mut failures = Vec::new();
    for case in doc["separation"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let exists = case["mle_exists"].as_bool().unwrap();
        // Without the ridge fallback the verdict is on the unpenalised iteration.
        let fit = fit_case(
            GlmFamily::BinomialLogit,
            case,
            None,
            &GlmOptions { ridge_on_separation: None, ..GlmOptions::new(50, 1e-8) },
        )
        .unwrap();
        let refused = fit.require_ok();
        eprintln!(
            "{name}: mle_exists {exists} separated {} converged {} refused {:?}",
            fit.separated,
            fit.converged,
            refused.as_ref().err().map(|e| e.to_string())
        );
        if exists {
            if fit.separated || refused.is_err() {
                failures.push(format!(
                    "{name}: a finite MLE exists but the fit was flagged or refused"
                ));
            }
        } else {
            if !fit.separated {
                failures.push(format!(
                    "{name}: no finite MLE exists but the fit was not flagged separated"
                ));
            }
            match refused {
                Err(StatsError::GlmRefused {
                    kind:
                        GlmRefusalKind::Separated
                        | GlmRefusalKind::NonConverged
                        | GlmRefusalKind::BoundarySaturated,
                    ..
                }) => {}
                other => failures.push(format!("{name}: expected a refusal, got {other:?}")),
            }
            // With the default ridge fallback the coefficients are finite and penalised, yet the
            // unpenalised verdict is kept, so estimation paths still refuse.
            let ridge =
                fit_case(GlmFamily::BinomialLogit, case, None, &GlmOptions::default()).unwrap();
            if !ridge.penalized
                || !ridge.separated
                || ridge.require_ok().is_ok()
                || ridge.coefficients.iter().any(|b| !b.is_finite())
            {
                failures.push(format!(
                    "{name}: ridge fallback must be finite, penalised, still flagged and refused"
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn an_iteration_limit_is_reported_as_non_convergence() {
    let doc: Value = serde_json::from_str(REFERENCE).unwrap();
    let case = &doc["fits"][0];
    let fit = fit_case(
        GlmFamily::BinomialLogit,
        case,
        None,
        &GlmOptions { ridge_on_separation: None, ..GlmOptions::new(1, 1e-12) },
    )
    .unwrap();
    assert!(!fit.converged);
    assert!(matches!(
        fit.require_ok(),
        Err(StatsError::GlmRefused { kind: GlmRefusalKind::NonConverged, iterations: 1, .. })
    ));
}

#[test]
fn invalid_outcomes_and_weights_are_refused() {
    let doc: Value = serde_json::from_str(REFERENCE).unwrap();
    let case = &doc["fits"][0];
    let (n, p) = (case["n"].as_u64().unwrap() as usize, case["p"].as_u64().unwrap() as usize);
    let x = floats(&case["x"]);
    let mut ws = LeastSquaresWorkspace::default();
    let mut y = floats(&case["y"]);
    y[0] = 0.5;
    let design = GlmDesignRef { x_colmajor: &x, nrows: n, ncols: p, y: &y };
    assert!(
        fit_glm(GlmFamily::BinomialLogit, design, &FaerBackend, &mut ws, &GlmOptions::default())
            .is_err()
    );
    assert!(
        fit_glm(
            GlmFamily::PoissonLog,
            GlmDesignRef { y: &[-1.0; 60], ..design },
            &FaerBackend,
            &mut ws,
            &GlmOptions::default()
        )
        .is_err()
    );
    let y = floats(&case["y"]);
    let design = GlmDesignRef { x_colmajor: &x, nrows: n, ncols: p, y: &y };
    for bad in [-1.0, f64::NAN, f64::INFINITY] {
        let mut w = vec![1.0; n];
        w[3] = bad;
        assert!(
            fit_glm_weighted(
                GlmFamily::BinomialLogit,
                design,
                &w,
                &FaerBackend,
                &mut ws,
                &GlmOptions::default()
            )
            .is_err()
        );
    }
    assert!(
        fit_glm_weighted(
            GlmFamily::BinomialLogit,
            design,
            &vec![1.0; n - 1],
            &FaerBackend,
            &mut ws,
            &GlmOptions::default()
        )
        .is_err()
    );
}
