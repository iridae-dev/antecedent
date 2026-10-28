//! Bayesian regime survival on tiny binary-confounder designs against an independent g-formula.
//!
//! `oracle_data/bayesian_temporal_regime_survival_reference.py` (frozen in the JSON beside it)
//! simulates three designs of 16 to 20 subjects with one binary confounder and two or three
//! periods. For each it computes, in 50-digit arithmetic, the three pooled-logistic Laplace
//! posteriors (hazard, treatment, censoring) and the conjugate Gaussian confounder law, and then
//!
//! * for two periods, the exact posterior mean of the always-treat and never-treat survival
//!   (a four-dimensional integral: Gauss-Hermite over the hazard coefficients and a Student-t
//!   marginal of the next confounder, the Dirichlet weights having mean 1/n), and
//! * for every design, an independent numpy Monte Carlo of the whole draw (2e6 draws): mean, sd,
//!   kurtosis, the distribution function and the share of draws below a positivity floor.
//!
//! The product's draws are compared with these within bounds derived from the iid-draw sampling
//! error of both sides; the shared Laplace kernel is also checked exactly on the same designs.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_estimate::temporal_design::survival::{
    SURVIVAL_COORDINATES, Trajectories, TrajectoryRows, fit_regime_survival_posterior,
};
use antecedent_stats::{GlmDesignRef, GlmOptions, LeastSquaresWorkspace, fit_logit_laplace};
use serde_json::Value;

/// Product draws per design.
const DRAWS: usize = 60_000;
/// Half-width, in standard errors, of every statistical acceptance band. A two-sided 5-sigma
/// band fails a correct, differently seeded product with probability about 6e-7 per statistic.
const SIGMAS: f64 = 5.0;

fn floats(value: &Value) -> Vec<f64> {
    value.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()
}

fn matrix(value: &Value) -> Vec<Vec<f64>> {
    value.as_array().unwrap().iter().map(floats).collect()
}

/// The shared penalised logistic kernel on the reference's own design rows: mode and Laplace
/// covariance against the 50-digit Newton fit.
fn check_laplace_kernel(name: &str, label: &str, block: &Value, scale: f64) {
    let x = matrix(&block["x"]);
    let y = floats(&block["y"]);
    let (n, p) = (y.len(), x[0].len());
    let mut colmajor = vec![0.0; n * p];
    for (i, row) in x.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            colmajor[j * n + i] = *v;
        }
    }
    let design = GlmDesignRef { x_colmajor: &colmajor, nrows: n, ncols: p, y: &y };
    let fit = fit_logit_laplace(
        design,
        None,
        1.0 / (scale * scale),
        &mut LeastSquaresWorkspace::default(),
        &GlmOptions::new(100, 1e-10),
    )
    .unwrap();
    for (j, want) in floats(&block["mode"]).iter().enumerate() {
        assert!(
            (fit.fit.coefficients[j] - want).abs() <= 1e-9 * want.abs().max(1.0),
            "{name} {label} mode[{j}]: {} vs {want}",
            fit.fit.coefficients[j]
        );
    }
    for (i, row) in matrix(&block["cov"]).iter().enumerate() {
        for (j, want) in row.iter().enumerate() {
            let got = fit.covariance[i * p + j];
            assert!(
                (got - want).abs() <= 1e-8 * want.abs().max(1e-3),
                "{name} {label} cov[{i}][{j}]: {got} vs {want}"
            );
        }
    }
}

/// Reference distribution function at `x`, by linear interpolation of the stored quantile grid.
fn reference_cdf(levels: &[f64], grid: &[f64], x: f64) -> f64 {
    if x <= grid[0] {
        return levels[0];
    }
    if x >= grid[grid.len() - 1] {
        return levels[levels.len() - 1];
    }
    let hi = grid.partition_point(|g| *g < x);
    let (a, b) = (grid[hi - 1], grid[hi]);
    levels[hi - 1] + (levels[hi] - levels[hi - 1]) * (x - a) / (b - a)
}

#[test]
fn bayesian_regime_survival_matches_the_independent_g_formula() {
    let reference: Value = serde_json::from_str(include_str!(
        "oracle_data/bayesian_temporal_regime_survival_reference.json"
    ))
    .unwrap();
    for case in reference["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let scale = case["prior_scale"].as_f64().unwrap();
        let horizon = usize::try_from(case["horizon"].as_u64().unwrap()).unwrap();
        for label in ["hazard", "treatment", "censoring"] {
            check_laplace_kernel(name, label, &case[label], scale);
        }
        let rows = case["rows"].as_array().unwrap();
        let unit: Vec<usize> =
            rows.iter().map(|r| usize::try_from(r["unit"].as_u64().unwrap()).unwrap()).collect();
        let period: Vec<i64> = rows.iter().map(|r| r["period"].as_i64().unwrap()).collect();
        let flag =
            |key: &str| -> Vec<bool> { rows.iter().map(|r| r[key].as_bool().unwrap()).collect() };
        let (treatment, censored, event) = (flag("treatment"), flag("censored"), flag("event"));
        let confounder: Vec<f64> = rows.iter().map(|r| r["L"].as_f64().unwrap()).collect();
        let data = Trajectories::new(
            TrajectoryRows {
                unit: &unit,
                period: &period,
                treatment: &treatment,
                censored: &censored,
                event: &event,
                confounders: &[confounder],
            },
            horizon,
        )
        .unwrap();
        assert_eq!(data.subjects() as u64, case["subjects"].as_u64().unwrap(), "{name}");
        assert_eq!(
            data.person_periods() as u64,
            case["person_periods"].as_u64().unwrap(),
            "{name}"
        );

        let fit = fit_regime_survival_posterior(
            &data,
            case["floor"].as_f64().unwrap(),
            scale,
            DRAWS,
            &ExecutionContext::for_tests(11),
        )
        .unwrap();
        let posterior = fit.posterior.as_ref().expect("a posterior summary");
        assert_eq!(posterior.draws.len(), SURVIVAL_COORDINATES.len(), "{name}");
        let n = DRAWS as f64;
        let mc = &case["mc"];
        let n_ref = mc["n"].as_f64().unwrap();
        // The contrast coordinate is the difference of the arm survivals draw by draw.
        for s in 0..DRAWS {
            let gap = posterior.draws[0][s] - (posterior.draws[1][s] - posterior.draws[2][s]);
            assert!(gap.abs() <= 1e-12, "{name} draw {s}");
        }
        for (j, coordinate) in SURVIVAL_COORDINATES.iter().enumerate() {
            let what = format!("{name} {coordinate}");
            assert!((fit.estimate[j] - posterior.mean[j]).abs() <= 1e-15, "{what}");
            let sd_ref = floats(&mc["sd"])[j];
            let kurtosis = floats(&mc["kurtosis"])[j];

            // Mean. Product draws are iid, so the sample mean has standard error sd/sqrt(N);
            // the exact reference (two periods) adds only 1e-5 of quadrature slack, the Monte
            // Carlo reference its own standard error.
            let mean_ref = floats(&mc["mean"])[j];
            let (target, slack) = if case.get("exact").is_some() {
                (floats(&case["exact"]["mean"])[j], 1e-5)
            } else {
                (mean_ref, SIGMAS * sd_ref / n_ref.sqrt())
            };
            let band = SIGMAS * sd_ref / n.sqrt() + slack;
            eprintln!(
                "{what}: mean {} target {target} (band {band}), sd {} reference {sd_ref}",
                posterior.mean[j], posterior.standard_deviation[j]
            );
            assert!(
                (posterior.mean[j] - target).abs() <= band,
                "{what} mean {} vs {target} (band {band})",
                posterior.mean[j]
            );
            if case.get("exact").is_some() {
                // The independent Monte Carlo agrees with the exact integral, so the quadrature
                // is not the source of the comparison.
                let band_ref = SIGMAS * sd_ref / n_ref.sqrt() + 1e-5;
                assert!((mean_ref - target).abs() <= band_ref, "{what} reference mean");
            }

            // Standard deviation. se(s) = s sqrt((kappa - 1) / (4 N)) for iid draws with
            // kurtosis kappa (the reference value, which the product shares up to its own
            // sampling error); the two independent errors add in quadrature.
            let se = |count: f64| sd_ref * ((kurtosis - 1.0) / (4.0 * count)).sqrt();
            let band = SIGMAS * (se(n).powi(2) + se(n_ref).powi(2)).sqrt();
            assert!(
                (posterior.standard_deviation[j] - sd_ref).abs() <= band,
                "{what} sd {} vs {sd_ref} (band {band})",
                posterior.standard_deviation[j]
            );

            // Equal-tailed endpoints: the reference distribution function at the product's
            // endpoint must be the nominal tail level within the binomial sampling error of
            // the product's order statistic and of the reference.
            let levels = floats(&mc["cdf_levels"]);
            let grid = floats(&mc["cdf_grid"][j]);
            for (value, tail) in [(posterior.lower[j], 0.025), (posterior.upper[j], 0.975)] {
                let cdf = reference_cdf(&levels, &grid, value);
                let band = SIGMAS * (tail * (1.0 - tail) * (1.0 / n + 1.0 / n_ref)).sqrt() + 5e-4;
                assert!((cdf - tail).abs() <= band, "{what} endpoint {value}: cdf {cdf} vs {tail}");
            }
        }

        // Share of draws whose smallest regime-consistent action probability is below the
        // floor (binomial error of the product and of the reference).
        let shortfall = fit.posterior_positivity_shortfall.unwrap();
        let reference_shortfall = mc["shortfall"].as_f64().unwrap();
        let band = SIGMAS
            * (reference_shortfall * (1.0 - reference_shortfall) * (1.0 / n + 1.0 / n_ref)).sqrt();
        assert!(
            (shortfall - reference_shortfall).abs() <= band,
            "{name} shortfall {shortfall} vs {reference_shortfall}"
        );
    }
}

#[test]
fn the_exact_check_resolves_the_posterior_spread_of_the_hazard() {
    // The first design is wide enough that the exact posterior mean of the survival contrast
    // differs from the hazard-mode plug-in by many standard errors of the product's draws, so
    // a posterior that collapsed to its mode would fail the comparison above.
    let reference: Value = serde_json::from_str(include_str!(
        "oracle_data/bayesian_temporal_regime_survival_reference.json"
    ))
    .unwrap();
    let case = &reference["cases"][0];
    let sd = floats(&case["mc"]["sd"])[0];
    let band = SIGMAS * sd / (DRAWS as f64).sqrt() + 1e-5;
    let exact = floats(&case["exact"]["mean"])[0];
    let plug_in = floats(&case["exact"]["plug_in"])[0];
    assert!((exact - plug_in).abs() > 4.0 * band, "{exact} vs {plug_in} (band {band})");
}
