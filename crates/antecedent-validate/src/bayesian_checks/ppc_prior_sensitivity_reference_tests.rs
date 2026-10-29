//! The predictive checks, simulation-based calibration and prior sensitivity against a 60-digit
//! evaluation of their definitions (`oracle_data/ppc_prior_sensitivity_reference.py`) and against
//! the exact conjugate laws.
//!
//! Compiled into the library's test build because the replicate-noise and generating-parameter
//! streams are crate-internal. The frozen file carries each case's inputs, the library draw
//! streams (coefficient draws of the fitted posterior, chi-square, standard-normal and uniform
//! replicate-noise draws, prior and generating draws) and the expected statistics evaluated from
//! them exactly. The live tests regenerate every stream and assert it equals the frozen one, then
//! run the product and compare with the exact values.
//!
//! Draw-dependent quantities with a closed-form law (the prior predictive tail of the mean, the
//! posterior-predictive residual-variance conditional, the rank of a generating draw, the
//! posterior mean under a perturbed prior) are checked against that law with a derived
//! concentration bound; the bound is stated at each use.

use std::sync::Arc;

use antecedent_core::{CausalRng, ExecutionContext};
use antecedent_estimate::{
    BayesianBackendKind, BayesianGCompWorkspace, BayesianGComputationAte, CausalPosterior,
    OverlapPolicy, PreparedBayesianProblem, SerialDependence,
};
use antecedent_identify::IdentificationStatus;
use antecedent_kernels::standard_normal;
use antecedent_prob::{
    BayesLikelihood, ExternalPriorSource, ExternalPriorWeight, GaussianCoefficientPrior,
    GaussianVarianceModel, InvGammaPrior, PriorSensitivityFamily, PriorSensitivitySummary,
    PriorSet, PriorSpec, compose_external_priors_with_alphas, sample_gamma, sample_inv_gamma,
};
use antecedent_stats::{CompiledDesign, GlmFamily};
use serde_json::{Value, json};

use super::predictive::{conditional_residual_sd, strided_draw_index};
use super::sbc::{
    GeneratingVariance, SBC_BINS, SBC_CHI2_CRITICAL_9DF_P99, draw_generating_parameters, sbc_bin,
};
use super::{
    ExternalAlphaSensitivity, PosteriorPredictiveCheck, PredictiveCheckKind, PredictiveCheckReport,
    PredictiveNoise, PriorPredictiveCheck, PriorSensitivity, SbcReport, SimulationBasedCalibration,
};
use crate::predictive_assessment::ForecastLaw;

const REPLICATE_NOISE: u64 = crate::streams::REPLICATE_NOISE;
const SERIAL_PPC: u64 = crate::streams::SERIAL_PPC;

fn reference() -> Value {
    serde_json::from_str(include_str!(
        "../../tests/oracle_data/ppc_prior_sensitivity_reference.json"
    ))
    .expect("frozen reference")
}

fn arr(v: &Value) -> Vec<f64> {
    v.as_array().expect("array").iter().map(|x| x.as_f64().expect("f64")).collect()
}

fn arr2(v: &Value) -> Vec<Vec<f64>> {
    v.as_array().expect("array").iter().map(arr).collect()
}

fn same(a: &Value, b: &Value, path: &str) {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            assert_eq!(x.as_f64(), y.as_f64(), "{path}");
        }
        (Value::Array(x), Value::Array(y)) => {
            assert_eq!(x.len(), y.len(), "{path} length");
            for (i, (p, q)) in x.iter().zip(y).enumerate() {
                same(p, q, &format!("{path}[{i}]"));
            }
        }
        (Value::Object(x), Value::Object(y)) => {
            for (k, v) in x {
                same(
                    v,
                    y.get(k).unwrap_or_else(|| panic!("{path}.{k} missing")),
                    &format!("{path}.{k}"),
                );
            }
        }
        _ => assert_eq!(a, b, "{path}"),
    }
}

fn close(got: f64, want: f64, rel: f64, what: &str) {
    assert!((got - want).abs() <= rel * want.abs().max(1.0), "{what}: {got} vs {want}");
}

fn problem(case: &Value) -> PreparedBayesianProblem {
    let t = arr(&case["t"]);
    let z = arr(&case["z"]);
    let y = arr(&case["y"]);
    PreparedBayesianProblem {
        design: CompiledDesign::linear_adjustment(
            &t,
            &[(antecedent_core::VariableId::from_raw(2), z.as_slice())],
            &y,
            &[],
        )
        .unwrap(),
        method: Arc::from("backdoor.adjustment"),
        adjustment_set: Arc::from([antecedent_core::VariableId::from_raw(2)]),
        active: 1.0,
        control: 0.0,
        overlap: OverlapPolicy::ExplicitOverride,
        coef_names: None,
        unit_ids: None,
        serial_dependence: SerialDependence::Iid,
        preparation: None,
    }
}

fn design_json(prep: &PreparedBayesianProblem) -> Value {
    json!({
        "nrows": prep.design.nrows,
        "ncols": prep.design.ncols,
        "matrix_colmajor": prep.design.matrix.to_vec(),
        "outcome": prep.design.outcome.to_vec(),
    })
}

fn merge(mut base: Value, extra: Value) -> Value {
    base.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
    base
}

fn glm_family(name: &str) -> GlmFamily {
    match name {
        "gaussian" => GlmFamily::GaussianIdentity,
        "logit" => GlmFamily::BinomialLogit,
        "probit" => GlmFamily::BinomialProbit,
        "poisson" => GlmFamily::PoissonLog,
        other => panic!("family {other}"),
    }
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

fn fit(estimator: &BayesianGComputationAte, prep: &PreparedBayesianProblem) -> CausalPosterior {
    estimator
        .fit(
            prep,
            IdentificationStatus::NonparametricallyIdentified,
            &mut BayesianGCompWorkspace::default(),
            &ctx(),
        )
        .unwrap()
}

// ------------------------------------------------------------------------------------------
// posterior predictive
// ------------------------------------------------------------------------------------------

fn post_estimator(case: &Value) -> BayesianGComputationAte {
    let draws = case["fit_draws"].as_u64().unwrap() as usize;
    let seed = case["fit_seed"].as_u64().unwrap();
    match case["family"].as_str().unwrap() {
        "gaussian" => BayesianGComputationAte {
            backend: BayesianBackendKind::ConjugateGaussian,
            n_draws: draws,
            seed,
            prior_scale: 10.0,
            ..BayesianGComputationAte::new()
        },
        other => BayesianGComputationAte::new()
            .with_likelihood(match other {
                "logit" => BayesLikelihood::BernoulliLogit,
                "probit" => BayesLikelihood::BernoulliProbit,
                _ => BayesLikelihood::PoissonLog,
            })
            .with_n_draws(draws)
            .with_seed(seed),
    }
}

fn post_check(case: &Value) -> PosteriorPredictiveCheck {
    PosteriorPredictiveCheck {
        n_sims: case["n_sims"].as_u64().unwrap() as u32,
        family: glm_family(case["family"].as_str().unwrap()),
        seed: case["check_seed"].as_u64().unwrap(),
    }
}

/// The library draw streams a posterior predictive check consumes, regenerated in the order the
/// definition consumes them.
fn regen_post(case: &Value) -> Value {
    let prep = problem(case);
    let post = fit(&post_estimator(case), &prep);
    let (n, p) = (prep.design.nrows, prep.design.ncols);
    let total = post.draws.n_draws;
    let take = total.min(case["n_sims"].as_u64().unwrap() as usize);
    let betas: Vec<Vec<f64>> =
        (0..total).map(|d| (0..p).map(|c| post.draws.get(d, c).unwrap()).collect()).collect();
    let mut rng = CausalRng::from_seed(case["check_seed"].as_u64().unwrap() ^ REPLICATE_NOISE);
    let (mut chi2, mut z, mut u) = (Vec::new(), Vec::new(), Vec::new());
    if case["family"] == "gaussian" {
        for _ in 0..take {
            chi2.push(sample_gamma((n + p) as f64 / 2.0, 0.5, &mut rng));
            z.push((0..n).map(|_| standard_normal(&mut rng)).collect::<Vec<_>>());
        }
    } else {
        u = (0..case["u_count"].as_u64().unwrap()).map(|_| rng.next_f64()).collect();
    }
    let mut out = merge(design_json(&prep), json!({"betas": betas, "chi2": chi2, "z": z, "u": u}));
    if let Some(seed) = case["temporal_seed"].as_u64() {
        let mut rng = CausalRng::from_seed(seed ^ SERIAL_PPC);
        let serial: Vec<Vec<f64>> =
            (0..take).map(|_| (0..n).map(|_| standard_normal(&mut rng)).collect()).collect();
        out["serial_z"] = json!(serial);
    }
    out
}

fn assert_report(got: &PredictiveCheckReport, want: &Value, label: &str) {
    let r = 1e-10;
    close(got.observed, want["observed"].as_f64().unwrap(), r, &format!("{label} observed"));
    close(got.predictive_mean, want["predictive_mean"].as_f64().unwrap(), r, label);
    close(got.predictive_sd, want["predictive_sd"].as_f64().unwrap(), r, &format!("{label} sd"));
    close(got.p_value, want["p_value"].as_f64().unwrap(), 1e-13, &format!("{label} p"));
    for (g, w) in got.location_tails.iter().zip(arr(&want["location_tails"])) {
        close(*g, w, 1e-13, &format!("{label} location tail"));
    }
    close(
        got.observed_dispersion,
        want["observed_dispersion"].as_f64().unwrap(),
        r,
        &format!("{label} observed dispersion"),
    );
    close(
        got.predictive_dispersion_mean,
        want["predictive_dispersion_mean"].as_f64().unwrap(),
        r,
        &format!("{label} dispersion mean"),
    );
    close(
        got.dispersion_p_value,
        want["dispersion_p_value"].as_f64().unwrap(),
        1e-13,
        &format!("{label} dispersion p"),
    );
    for (g, w) in got.dispersion_tails.iter().zip(arr(&want["dispersion_tails"])) {
        close(*g, w, 1e-13, &format!("{label} dispersion tail"));
    }
    assert_eq!(u64::from(got.n_sims), want["n_sims"].as_u64().unwrap(), "{label} n_sims");
    let noise = match want["noise"].as_str().unwrap() {
        "mean_only" => PredictiveNoise::MeanOnlyImproperScale,
        _ => PredictiveNoise::Likelihood,
    };
    assert_eq!(got.noise, noise, "{label} noise");
}

fn assert_residual(got: &PredictiveCheckReport, want: &Value, label: &str) {
    let axes = want["residual"].as_array().unwrap();
    assert_eq!(got.residual.len(), axes.len(), "{label} residual axes");
    for (g, w) in got.residual.iter().zip(axes) {
        assert_eq!(g.statistic.name(), w["name"].as_str().unwrap(), "{label}");
        close(
            g.observed,
            w["observed"].as_f64().unwrap(),
            1e-9,
            &format!("{label} {}", g.statistic.name()),
        );
        close(g.predictive_mean, w["predictive_mean"].as_f64().unwrap(), 1e-9, label);
        close(g.p_value, w["p_value"].as_f64().unwrap(), 1e-13, label);
        for (a, b) in g.tails.iter().zip(arr(&w["tails"])) {
            close(*a, b, 1e-13, label);
        }
    }
    close(got.residual_p_value, want["residual_p_value"].as_f64().unwrap(), 1e-13, label);
}

#[test]
fn ppc_prior_sensitivity_posterior_predictive_equals_the_exact_reference() {
    for case in reference()["post"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        same(&regen_post(case), &case["draws"], name);
        let prep = problem(case);
        let post = fit(&post_estimator(case), &prep);
        let check = post_check(case);
        let want = &case["expected"];
        let report = check.check(&prep, &post).unwrap();
        assert_eq!(report.kind, PredictiveCheckKind::Posterior);
        assert_report(&report, want, name);
        assert_residual(&report, want, name);
        // The replicate sets the check summarizes: draw-major, one sample law per row.
        let replicates = check.replicates(&prep, &post).unwrap();
        let expected = arr2(&want["replicates"]);
        assert_eq!(replicates.draws(), expected.len(), "{name}");
        for (row, law) in replicates.laws().into_iter().enumerate() {
            let ForecastLaw::EmpiricalSample { draws } = law else { panic!("sample law") };
            for (k, value) in draws.iter().enumerate() {
                close(*value, expected[k][row], 1e-10, &format!("{name} replicate {k},{row}"));
            }
        }
        if let Some(seed) = case["temporal_seed"].as_u64() {
            let temporal = check.check_temporal(&prep, &post, seed).unwrap();
            let serial = temporal.serial.expect("serial axis");
            let w = &want["serial"];
            assert_eq!(serial.lag, 1);
            close(serial.observed, w["observed"].as_f64().unwrap(), 1e-9, "serial observed");
            close(
                serial.predictive_mean,
                w["predictive_mean"].as_f64().unwrap(),
                1e-9,
                "serial mean",
            );
            close(serial.p_value, w["p_value"].as_f64().unwrap(), 1e-13, "serial p");
            assert!(serial.tempering_kappa.is_none());
        }
    }
}

#[test]
fn ppc_prior_sensitivity_posterior_predictive_reads_every_option() {
    let fixture = reference();
    let case = &fixture["post"][0];
    let prep = problem(case);
    let post = fit(&post_estimator(case), &prep);
    let base = post_check(case);
    let report = base.check(&prep, &post).unwrap();
    assert_eq!(report.n_sims, 45, "n_sims below the stored draws");
    // The draw cap: n_sims above the stored draw count uses them all.
    let capped =
        PosteriorPredictiveCheck { n_sims: 5000, ..base.clone() }.check(&prep, &post).unwrap();
    assert_eq!(capped.n_sims, 60);
    // The replicate-noise seed moves the replicates.
    let moved = PosteriorPredictiveCheck { seed: base.seed + 1, ..base.clone() }
        .check(&prep, &post)
        .unwrap();
    assert_ne!(moved.predictive_dispersion_mean, report.predictive_dispersion_mean);
    // The family selects the observation model: a Poisson reading of the same draws differs.
    let poisson = PosteriorPredictiveCheck { family: GlmFamily::PoissonLog, ..base.clone() }
        .check(&prep, &post)
        .unwrap();
    assert_ne!(poisson.predictive_mean, report.predictive_mean);
    // The stride that spreads used draws over the store.
    for (k, want) in [(0usize, 0usize), (1, 1), (2, 2), (44, 58)] {
        assert_eq!(strided_draw_index(k, 45, 60), want);
    }
    // n_sims = 0 and an empty design are refused.
    let refused = PosteriorPredictiveCheck { n_sims: 0, ..base }.check(&prep, &post).unwrap_err();
    assert!(format!("{refused:?}").contains("no posterior draws for PPC"), "{refused:?}");
}

/// The residual-variance conditional of a replicate noise draw: with the conjugate
/// normal-inverse-gamma posterior, sigma^2 given the coefficients is inverse-gamma with shape
/// `a + (n + p) / 2` and rate `b + (SSR + Q) / 2`, so the marginal precision `1 / sigma^2` of the
/// joint draw is Gamma(a_n, rate b_n) with a_n = a + n / 2 and b_n the posterior rate. The sample
/// mean of N precisions deviates from a_n / b_n by more than
/// `sqrt(2 v x / N) + x s / N` above or `sqrt(2 v x / N)` below with probability at most
/// `exp(-x)` each (a Gamma variable is sub-gamma with variance factor v and scale s);
/// `x = ln(2e9)`, so the failure probability is below 1e-9.
#[test]
fn ppc_prior_sensitivity_replicate_noise_variance_is_the_conjugate_conditional() {
    let fixture = reference();
    let shape = &fixture["shape"];
    let data = &shape["data"];
    let case = json!({"t": data["t"], "z": data["z"], "y": data["outcome"]});
    let prep = problem(&case);
    assert_eq!(arr(&data["outcome"]), prep.design.outcome.to_vec());
    let post = fit(
        &BayesianGComputationAte {
            backend: BayesianBackendKind::ConjugateGaussian,
            n_draws: data["draws"].as_u64().unwrap() as usize,
            seed: 61,
            prior_scale: 10.0,
            ..BayesianGComputationAte::new()
        },
        &prep,
    );
    let (n, p) = (prep.design.nrows, prep.design.ncols);
    let mut rng = CausalRng::from_seed(4242);
    let mut sum_precision = 0.0;
    let total = post.draws.n_draws;
    for d in 0..total {
        let beta: Vec<f64> = (0..p).map(|c| post.draws.get(d, c).unwrap()).collect();
        let ssr: f64 = (0..n)
            .map(|i| {
                let mu: f64 = (0..p).map(|c| prep.design.matrix[c * n + i] * beta[c]).sum();
                (prep.design.outcome[i] - mu).powi(2)
            })
            .sum();
        let sd = conditional_residual_sd(ssr, n, p, &mut rng);
        sum_precision += 1.0 / (sd * sd);
    }
    let mean = sum_precision / total as f64;
    let (centre, up, down) = (
        shape["mean_precision"].as_f64().unwrap(),
        shape["upper"].as_f64().unwrap(),
        shape["lower"].as_f64().unwrap(),
    );
    assert!(
        mean <= centre + up && mean >= centre - down,
        "mean precision {mean} outside [{}, {}] around the conjugate {centre}",
        centre - down,
        centre + up
    );
}

// ------------------------------------------------------------------------------------------
// prior predictive
// ------------------------------------------------------------------------------------------

fn prior_set(case: &Value) -> PriorSet {
    let mut prior = PriorSet::new();
    prior.push(PriorSpec::GaussianCoefficients(GaussianCoefficientPrior {
        mean: Arc::from(arr(&case["coef_mean"])),
        variance: Arc::from(arr(&case["coef_var"])),
    }));
    match case["kind"].as_str().unwrap() {
        "known" => prior.push(PriorSpec::KnownResidualVariance(case["sigma2"].as_f64().unwrap())),
        "ig" | "improper" => prior.push(PriorSpec::ResidualInvGamma(InvGammaPrior {
            shape: case["shape"].as_f64().unwrap(),
            scale: case["scale"].as_f64().unwrap(),
        })),
        _ => {}
    }
    prior
}

fn prior_check(case: &Value) -> PriorPredictiveCheck {
    PriorPredictiveCheck {
        n_sims: case["n_sims"].as_u64().unwrap() as u32,
        seed: case["seed"].as_u64().unwrap(),
        family: glm_family(case["family"].as_str().unwrap()),
    }
}

fn regen_prior(case: &Value) -> Value {
    let prep = problem(case);
    let (n, p) = (prep.design.nrows, prep.design.ncols);
    let seed = case["seed"].as_u64().unwrap();
    let mut rng = CausalRng::from_seed(seed);
    let mut noise_rng = CausalRng::from_seed(seed ^ REPLICATE_NOISE);
    let gaussian = case["family"] == "gaussian";
    let kind = case["kind"].as_str().unwrap();
    let (mut z_beta, mut sigma2, mut z_noise) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..case["n_sims"].as_u64().unwrap() {
        z_beta.push((0..p).map(|_| standard_normal(&mut rng)).collect::<Vec<_>>());
        if gaussian && kind != "improper" {
            sigma2.push(match kind {
                "known" => case["sigma2"].as_f64().unwrap(),
                _ => sample_inv_gamma(
                    case["shape"].as_f64().unwrap(),
                    case["scale"].as_f64().unwrap(),
                    &mut noise_rng,
                ),
            });
            z_noise.push((0..n).map(|_| standard_normal(&mut noise_rng)).collect::<Vec<_>>());
        }
    }
    let u: Vec<f64> = if gaussian {
        Vec::new()
    } else {
        let mut stream = CausalRng::from_seed(seed ^ REPLICATE_NOISE);
        (0..case["u_count"].as_u64().unwrap()).map(|_| stream.next_f64()).collect()
    };
    merge(
        design_json(&prep),
        json!({"z_beta": z_beta, "sigma2": sigma2, "z_noise": z_noise, "u": u}),
    )
}

#[test]
fn ppc_prior_sensitivity_prior_predictive_equals_the_exact_reference() {
    for case in reference()["prior"].as_array().unwrap() {
        if case["law"].as_bool().unwrap() {
            continue;
        }
        let name = case["name"].as_str().unwrap();
        same(&regen_prior(case), &case["draws"], name);
        let prep = problem(case);
        let report = prior_check(case).check_with_prior(&prep, &prior_set(case), &ctx()).unwrap();
        assert_eq!(report.kind, PredictiveCheckKind::Prior);
        assert_report(&report, &case["expected"], name);
        assert!(report.residual.is_empty() && report.serial.is_none());
    }
}

/// Exact prior predictive law of the mean. With `beta | sigma^2 ~ N(m, sigma^2 V0)` (the conjugate
/// scale V0 of the fitted prior) and `y_rep = X beta + sigma z`, the replicate mean is
/// `N(xbar'm, sigma^2 (1/n + xbar' V0 xbar))` for a known `sigma^2`, a Student t with `2a` degrees
/// of freedom, location `xbar'm` and squared scale `(b/a)(1/n + xbar' V0 xbar)` under an `IG(a, b)`
/// residual prior. The lower Monte Carlo tail `(1 + #{T <= t_obs}) / (1 + N)` has mean within
/// `1/(N+1)` of that law's CDF and each indicator is an independent Bernoulli, so by Hoeffding it is
/// within `sqrt(ln(2/delta) / (2N))` of the CDF with probability at least `1 - delta`
/// (`delta = 1e-9`).
#[test]
fn ppc_prior_sensitivity_prior_predictive_mean_follows_the_exact_law() {
    let fixture = reference();
    for case in fixture["prior"].as_array().unwrap() {
        if !case["law"].as_bool().unwrap() {
            continue;
        }
        let name = case["name"].as_str().unwrap();
        let prep = problem(case);
        let report = prior_check(case).check_with_prior(&prep, &prior_set(case), &ctx()).unwrap();
        let n_sims = f64::from(report.n_sims);
        let bound = ((2.0e9_f64).ln() / (2.0 * n_sims)).sqrt() + 1.0 / (n_sims + 1.0);
        let law = &case["expected_law"];
        let lower = law["lower"].as_f64().unwrap();
        let upper = law["upper"].as_f64().unwrap();
        assert!(
            (report.location_tails[0] - lower).abs() <= bound,
            "{name}: lower tail {} vs exact {lower} (bound {bound})",
            report.location_tails[0]
        );
        assert!(
            (report.location_tails[1] - upper).abs() <= bound,
            "{name}: upper tail {} vs exact {upper} (bound {bound})",
            report.location_tails[1]
        );
        if let Some(wrong) = law["convention_free_lower"].as_f64() {
            // The test has power against drawing the coefficients without the residual-variance
            // scale: that law's tail lies outside the bound around the exact one.
            assert!((wrong - lower).abs() > 2.0 * bound, "{name}: wrong law not separated");
        }
    }
}

#[test]
fn ppc_prior_sensitivity_prior_predictive_reads_every_option_and_refuses_bad_inputs() {
    let fixture = reference();
    let case = &fixture["prior"][0];
    let prep = problem(case);
    let prior = prior_set(case);
    let base = prior_check(case);
    let a = base.check_with_prior(&prep, &prior, &ctx()).unwrap();
    assert_eq!(a.n_sims, 40);
    let b = PriorPredictiveCheck { seed: base.seed + 1, ..base.clone() }
        .check_with_prior(&prep, &prior, &ctx())
        .unwrap();
    assert_ne!(a.predictive_mean, b.predictive_mean, "seed");
    let c = PriorPredictiveCheck { n_sims: 20, ..base.clone() }
        .check_with_prior(&prep, &prior, &ctx())
        .unwrap();
    assert_eq!(c.n_sims, 20);
    let d = PriorPredictiveCheck { family: GlmFamily::PoissonLog, ..base.clone() }
        .check_with_prior(&prep, &prior, &ctx())
        .unwrap();
    assert_ne!(a.predictive_mean, d.predictive_mean, "family");
    // A prior without Gaussian coefficients, or of the wrong dimension, is refused.
    let missing = base.check_with_prior(&prep, &PriorSet::new(), &ctx()).unwrap_err();
    assert!(format!("{missing:?}").contains("missing Gaussian coefficients"), "{missing:?}");
    let mut short = PriorSet::new();
    short.push(PriorSpec::GaussianCoefficients(GaussianCoefficientPrior::isotropic(2, 1.0)));
    let wrong_dimension = base.check_with_prior(&prep, &short, &ctx()).unwrap_err();
    assert!(
        format!("{wrong_dimension:?}").contains("coefficient dimension mismatch"),
        "{wrong_dimension:?}"
    );
    // A known residual variance is not read as a prior on the coefficient scale: doubling it
    // doubles the replicate-mean variance when the coefficient scale dominates.
    let mut loose = PriorSet::new();
    loose.push(PriorSpec::GaussianCoefficients(GaussianCoefficientPrior {
        mean: Arc::from(vec![0.0; 3]),
        variance: Arc::from(vec![100.0; 3]),
    }));
    loose.push(PriorSpec::KnownResidualVariance(1.0));
    let mut loose4 = loose.clone();
    loose4.specs[1] = PriorSpec::KnownResidualVariance(4.0);
    let sd1 = PriorPredictiveCheck { n_sims: 4000, ..base.clone() }
        .check_with_prior(&prep, &loose, &ctx())
        .unwrap()
        .predictive_sd;
    let sd4 = PriorPredictiveCheck { n_sims: 4000, ..base }
        .check_with_prior(&prep, &loose4, &ctx())
        .unwrap()
        .predictive_sd;
    assert!((sd4 / sd1 - 2.0).abs() < 1e-6, "sd ratio {}", sd4 / sd1);
}

// ------------------------------------------------------------------------------------------
// simulation-based calibration
// ------------------------------------------------------------------------------------------

fn sbc_prior(case: &Value) -> PriorSet {
    let mut prior = PriorSet::new();
    prior.push(PriorSpec::GaussianCoefficients(GaussianCoefficientPrior {
        mean: Arc::from(arr(&case["coef_mean"])),
        variance: Arc::from(arr(&case["coef_var"])),
    }));
    match case["kind"].as_str().unwrap() {
        "known" => prior.push(PriorSpec::KnownResidualVariance(case["sigma2"].as_f64().unwrap())),
        _ => prior.push(PriorSpec::ResidualInvGamma(InvGammaPrior {
            shape: case["shape"].as_f64().unwrap(),
            scale: case["scale"].as_f64().unwrap(),
        })),
    }
    prior
}

fn sbc_setup(
    case: &Value,
) -> (BayesianGComputationAte, PreparedBayesianProblem, SimulationBasedCalibration) {
    (
        BayesianGComputationAte {
            backend: BayesianBackendKind::ConjugateGaussian,
            n_draws: 99,
            seed: case["fit_seed"].as_u64().unwrap(),
            prior: Some(sbc_prior(case)),
            ..BayesianGComputationAte::new()
        },
        problem(case),
        SimulationBasedCalibration {
            n_reps: case["n_reps"].as_u64().unwrap() as u32,
            n_draws: case["n_draws"].as_u64().unwrap() as usize,
            seed: case["seed"].as_u64().unwrap(),
        },
    )
}

fn run_sbc(case: &Value) -> SbcReport {
    let (est, prep, sbc) = sbc_setup(case);
    sbc.check(
        &est,
        &prep,
        IdentificationStatus::NonparametricallyIdentified,
        &mut BayesianGCompWorkspace::default(),
        &ctx(),
    )
    .unwrap()
}

fn regen_sbc(case: &Value) -> Value {
    let (_, prep, sbc) = sbc_setup(case);
    let (n, p) = (prep.design.nrows, prep.design.ncols);
    let mut rng = CausalRng::from_seed(sbc.seed);
    let kind = case["kind"].as_str().unwrap();
    let (mut sigma2, mut z_beta, mut z_noise) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..sbc.n_reps {
        sigma2.push(match kind {
            "known" => case["sigma2"].as_f64().unwrap(),
            _ => sample_inv_gamma(
                case["shape"].as_f64().unwrap(),
                case["scale"].as_f64().unwrap(),
                &mut rng,
            ),
        });
        z_beta.push((0..p).map(|_| standard_normal(&mut rng)).collect::<Vec<_>>());
        z_noise.push((0..n).map(|_| standard_normal(&mut rng)).collect::<Vec<_>>());
    }
    let ranks: Vec<u32> = run_sbc(case).ranks.to_vec();
    merge(
        design_json(&prep),
        json!({"sigma2": sigma2, "z_beta": z_beta, "z_noise": z_noise, "ranks": ranks}),
    )
}

/// Each generating draw is a conjugate prior draw of the fitted model (`beta | sigma^2 ~
/// N(m, sigma^2 V0)`), the simulated data are `X beta + sigma z`, and the posterior of the
/// treatment coefficient is exactly known (normal for a known `sigma^2`, Student t under an
/// inverse-gamma residual prior). Given the data the rank is the count of `n_draws` independent
/// posterior draws below the generating value, so it is `Binomial(n_draws, F(theta))` with `F`
/// the exact posterior CDF at the generating value; averaged over the prior the rank is uniform.
/// Checks: the product's draws equal the definition, each rank is not in either tail
/// (`P <= 1e-7`) of its exact binomial, and the rank sum is within the Hoeffding bound
/// `sqrt(ln(2/delta)/2 * R L)` (`delta = 1e-9`, `R L` independent Bernoulli trials) of its exact
/// mean.
#[test]
fn ppc_prior_sensitivity_simulation_based_calibration_matches_the_exact_conjugate_ranks() {
    for case in reference()["sbc"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        same(&regen_sbc(case), &case["draws"], name);
        let want = &case["expected"];
        let report = run_sbc(case);
        assert_eq!(report.rank_draw_count, 99);
        assert_eq!(report.rank_bins, SBC_BINS);
        assert_eq!(report.ranks.len(), 60);
        assert_eq!(
            report.ranks.iter().map(|&r| f64::from(r)).collect::<Vec<_>>(),
            arr(&want["ranks"])
        );
        // Report arithmetic is the exact rational one.
        close(report.mean_rank_frac, want["mean_rank_frac"].as_f64().unwrap(), 1e-13, name);
        close(report.uniformity_stat, want["uniformity_stat"].as_f64().unwrap(), 1e-12, name);
        // Generating draws: the product helper reproduces the conjugate-scale draw.
        let (est, prep, sbc) = sbc_setup(case);
        let prior = est.prior_in_force(prep.design.ncols);
        let coefficients = prior.gaussian_coefficients().unwrap();
        let variance = match GaussianVarianceModel::from_prior_set(&prior).unwrap() {
            GaussianVarianceModel::Known { sigma2 } => GeneratingVariance::Known(sigma2),
            GaussianVarianceModel::InvGamma { shape, scale } => {
                GeneratingVariance::InvGamma { shape, scale }
            }
        };
        let mut rng = CausalRng::from_seed(sbc.seed);
        for (r, rep) in want["per_rep"].as_array().unwrap().iter().enumerate() {
            let (sigma2, beta) = draw_generating_parameters(coefficients, None, variance, &mut rng);
            close(sigma2, rep["sigma2"].as_f64().unwrap(), 1e-13, name);
            for (b, w) in beta.iter().zip(arr(&rep["beta"])) {
                close(*b, w, 1e-12, &format!("{name} rep {r} beta"));
            }
            for _ in 0..prep.design.nrows {
                let _ = standard_normal(&mut rng);
            }
            assert!(
                rep["p_le"].as_f64().unwrap() > 1e-7 && rep["p_ge"].as_f64().unwrap() > 1e-7,
                "{name} rep {r}: rank {} is in a binomial tail (p_le {}, p_ge {})",
                report.ranks[r],
                rep["p_le"],
                rep["p_ge"]
            );
        }
        let sum: f64 = report.ranks.iter().map(|&r| f64::from(r)).sum();
        let (centre, bound) =
            (want["rank_sum_expected"].as_f64().unwrap(), want["rank_sum_bound"].as_f64().unwrap());
        assert!((sum - centre).abs() <= bound, "{name}: rank sum {sum} vs {centre} +- {bound}");
        assert!(sbc.to_report(&report, 0.0).passed, "{name}");
    }
}

#[test]
fn ppc_prior_sensitivity_simulation_based_calibration_reads_every_option() {
    let fixture = reference();
    let case = &fixture["sbc"][1];
    let base = run_sbc(case);
    let (est, prep, sbc) = sbc_setup(case);
    let run = |sbc: &SimulationBasedCalibration| {
        sbc.check(
            &est,
            &prep,
            IdentificationStatus::NonparametricallyIdentified,
            &mut BayesianGCompWorkspace::default(),
            &ctx(),
        )
    };
    let reseeded = run(&SimulationBasedCalibration { seed: sbc.seed + 1, ..sbc.clone() }).unwrap();
    assert_ne!(reseeded.ranks, base.ranks, "seed");
    let longer = run(&SimulationBasedCalibration { n_reps: 70, ..sbc.clone() }).unwrap();
    assert_eq!(longer.ranks.len(), 70);
    assert_eq!(&longer.ranks[..60], &base.ranks[..], "replicates extend, they do not reshuffle");
    let fewer_draws = run(&SimulationBasedCalibration { n_draws: 49, ..sbc.clone() }).unwrap();
    assert_eq!(fewer_draws.rank_draw_count, 49);
    assert!(fewer_draws.ranks.iter().all(|&r| r <= 49));
    let unequal_bins = run(&SimulationBasedCalibration { n_draws: 98, ..sbc.clone() }).unwrap_err();
    assert!(
        format!("{unequal_bins:?}").contains("multiple of the 10 rank bins"),
        "{unequal_bins:?}"
    );
    let too_few = run(&SimulationBasedCalibration { n_reps: 49, ..sbc }).unwrap_err();
    assert!(format!("{too_few:?}").contains("at least 50 replicates"), "{too_few:?}");
}

#[test]
fn ppc_prior_sensitivity_rank_bins_and_verdict_thresholds() {
    let fixture = reference();
    let constants = &fixture["constants"];
    // Every rank belongs to exactly one bin and the bins are equal-mass when the count divides.
    for (n_draws, counts) in constants["bin_counts"].as_object().unwrap() {
        let n_draws: usize = n_draws.parse().unwrap();
        let mut got = [0u32; SBC_BINS];
        for rank in 0..=n_draws {
            got[sbc_bin(rank as u32, n_draws)] += 1;
        }
        assert_eq!(arr(counts), got.iter().map(|&c| f64::from(c)).collect::<Vec<_>>());
    }
    assert!(constants["rank_marginal_is_uniform"].as_bool().unwrap());
    // The tabulated chi-square 0.99 quantiles agree with the exact quantile to the printed digits.
    let q9 = constants["chi2_9df_p99"].as_f64().unwrap();
    let q4 = constants["chi2_4df_p99"].as_f64().unwrap();
    assert!((SBC_CHI2_CRITICAL_9DF_P99 - q9).abs() < 5e-4, "{SBC_CHI2_CRITICAL_9DF_P99} vs {q9}");
    let sbc = SimulationBasedCalibration::default();
    let report = |mean: f64, stat: f64, bins: usize| SbcReport {
        ranks: Arc::from(Vec::<u32>::new()),
        mean_rank_frac: mean,
        uniformity_stat: stat,
        rank_draw_count: 99,
        rank_bins: bins,
    };
    let verdict = |r: &SbcReport| sbc.to_report(r, 0.0).passed;
    // Ten independent-draw bins: critical value of nine degrees of freedom, inclusive.
    assert!(verdict(&report(0.5, SBC_CHI2_CRITICAL_9DF_P99, 10)));
    assert!(!verdict(&report(0.5, SBC_CHI2_CRITICAL_9DF_P99 + 1e-3, 10)));
    assert!(!verdict(&report(0.5, f64::NAN, 10)));
    // Five chain bins: critical value of four degrees of freedom.
    assert!((13.277 - q4).abs() < 5e-4);
    assert!(verdict(&report(0.5, 13.277, 5)));
    assert!(!verdict(&report(0.5, 13.278, 5)));
    // The mean-rank band [0.35, 0.65] is inclusive and independent of the uniformity clause.
    assert!(verdict(&report(0.35, 1.0, 10)) && verdict(&report(0.65, 1.0, 10)));
    assert!(!verdict(&report(0.3499, 1.0, 10)) && !verdict(&report(0.6501, 1.0, 10)));
    let both = sbc.to_report(&report(0.9, 99.0, 10), 0.0);
    assert!(!both.passed && both.failure_condition.as_deref().unwrap().contains("and"));
}

// ------------------------------------------------------------------------------------------
// prior sensitivity
// ------------------------------------------------------------------------------------------

fn sens_problem(spec: &Value) -> PreparedBayesianProblem {
    problem(&json!({"t": spec["t"], "z": spec["z"], "y": spec["y"]}))
}

fn sens_estimator(spec: &Value, prior: Option<PriorSet>) -> BayesianGComputationAte {
    BayesianGComputationAte {
        backend: BayesianBackendKind::ConjugateGaussian,
        n_draws: spec["fit_draws"].as_u64().unwrap() as usize,
        seed: spec["fit_seed"].as_u64().unwrap(),
        prior,
        ..BayesianGComputationAte::new()
    }
}

fn resolved_prior(spec: &Value) -> PriorSet {
    let r = &spec["resolved"];
    let mut prior = PriorSet::new();
    prior.push(PriorSpec::GaussianCoefficients(GaussianCoefficientPrior {
        mean: Arc::from(arr(&r["coef_mean"])),
        variance: Arc::from(arr(&r["coef_var"])),
    }));
    prior.push(PriorSpec::ResidualInvGamma(InvGammaPrior {
        shape: r["shape"].as_f64().unwrap(),
        scale: r["scale"].as_f64().unwrap(),
    }));
    prior
}

fn alpha_sources(spec: &Value, p: usize) -> Vec<ExternalPriorSource> {
    let a = &spec["alpha"];
    let mut source = PriorSet::new();
    source.push(PriorSpec::GaussianCoefficients(GaussianCoefficientPrior {
        mean: Arc::from(arr(&a["source_mean"])),
        variance: Arc::from(vec![a["source_var"].as_f64().unwrap(); p]),
    }));
    vec![ExternalPriorSource {
        id: Arc::from("survey"),
        prior: source,
        weight: ExternalPriorWeight::power(1.0).unwrap(),
        ess: None,
    }]
}

fn composed_json(spec: &Value, p: usize) -> Vec<Value> {
    let a = &spec["alpha"];
    let sources = alpha_sources(spec, p);
    let applied = arr(&a["alphas_applied"]);
    arr(&a["alphas"])
        .into_iter()
        .map(|mult| {
            let scaled: Vec<f64> = applied.iter().map(|x| (x * mult).clamp(0.0, 1.0)).collect();
            let composed = compose_external_priors_with_alphas(
                &sources,
                &[1.0],
                &scaled,
                &PriorSet::weakly_informative(p),
            )
            .unwrap();
            let coef = composed.prior.gaussian_coefficients().unwrap();
            assert!(composed.prior.coefficient_correlation().is_none());
            let resid = match GaussianVarianceModel::from_prior_set(&composed.prior).unwrap() {
                GaussianVarianceModel::Known { sigma2 } => {
                    json!({"kind": "known", "sigma2": sigma2})
                }
                GaussianVarianceModel::InvGamma { shape, scale } => {
                    json!({"kind": "ig", "shape": shape, "scale": scale})
                }
            };
            json!({"mean": coef.mean.to_vec(), "var": coef.variance.to_vec(), "resid": resid})
        })
        .collect()
}

fn regen_sens(spec: &Value) -> Value {
    let prep = sens_problem(spec);
    merge(design_json(&prep), json!({"composed": composed_json(spec, prep.design.ncols)}))
}

/// The posterior mean of the treatment effect under each perturbed prior is the exact conjugate
/// value `m_n[1]`; the product reports the sample mean of at most 200 independent posterior draws.
/// Those draws are `m_n + sigma sqrt(V_n) z` with `sigma^2 ~ IG(a_n, b_n)`, so given the
/// variances the mean is normal with variance `V_n[1,1] sum sigma_d^2 / N^2`. The sum
/// `sum sigma_d^2` exceeds `N c` with probability at most 5e-4 for
/// `c = E + sd sqrt((1 - d)/d) / sqrt(N)`, `d = 5e-4` (Cantelli on the mean of N iid draws with
/// mean `E = b_n/(a_n-1)` and sd `E/sqrt(a_n-2)`), and given the bound the normal tail beyond
/// `sqrt(2 V_n[1,1] c ln(4e3) / N)` has probability at most 5e-4: a joint failure probability of at
/// most 1e-3 per grid point.
fn assert_means(
    summary: &PriorSensitivitySummary,
    posts: &[CausalPosterior],
    exact: &Value,
    label: &str,
) {
    let exact = exact.as_array().unwrap();
    assert_eq!(summary.effect_means.len(), exact.len());
    for (i, w) in exact.iter().enumerate() {
        let post = &posts[i];
        let col = post.effect_column().unwrap();
        let draws = post.draws.column(col).unwrap();
        let mean = draws.iter().sum::<f64>() / draws.len() as f64;
        close(summary.effect_means[i], mean, 1e-12, &format!("{label}[{i}] summary mean"));
        let (centre, bound) = (w["mean"].as_f64().unwrap(), w["bound"].as_f64().unwrap());
        assert!(
            (summary.effect_means[i] - centre).abs() <= bound,
            "{label}[{i}]: posterior mean {} vs exact {centre} (bound {bound})",
            summary.effect_means[i]
        );
        // The posterior SD is a statistic of the same fixed-seed draws; its sampling error at 200
        // draws is about 5 percent and the observed relative deviation over all nine cases is at
        // most 2.1 percent, so the window is 6 percent (the former 35 percent admitted an SD off
        // by a third, e.g. a prior whose variance was misscaled by 1.8).
        let sd = w["sd"].as_f64().unwrap();
        assert!(
            (summary.effect_sds[i] / sd - 1.0).abs() < 0.06,
            "{label}[{i}]: posterior sd {} vs exact {sd}",
            summary.effect_sds[i]
        );
    }
}

#[test]
fn ppc_prior_sensitivity_grids_equal_the_exact_perturbed_posteriors() {
    let fixture = reference();
    let spec = &fixture["sens"];
    same(&regen_sens(spec), &spec["draws"], "sens");
    let prep = sens_problem(spec);
    let want = &spec["expected"];
    let id = IdentificationStatus::NonparametricallyIdentified;
    let mut ws = BayesianGCompWorkspace::default();

    // Isotropic scales around an unsupplied prior.
    let estimator = sens_estimator(spec, None);
    let grid = PriorSensitivity {
        scales: Arc::from(arr(&spec["scales"])),
        ..PriorSensitivity::standard_grid()
    };
    let (summary, posts) = grid.evaluate(&estimator, &prep, id, &mut ws, &ctx()).unwrap();
    assert_eq!(summary.family, PriorSensitivityFamily::IsotropicScale);
    assert_eq!(summary.prior_scales.to_vec(), arr(&spec["scales"]));
    assert!(posts.iter().all(|p| p.draws.n_draws == 200), "n_draws is capped at 200");
    assert_means(&summary, &posts, &want["isotropic"], "isotropic");

    // Variance multipliers around a supplied prior.
    let estimator = sens_estimator(spec, Some(resolved_prior(spec)));
    let grid = PriorSensitivity {
        scales: Arc::from([]),
        alphas: Arc::from([]),
        variance_multipliers: Arc::from(arr(&spec["resolved"]["multipliers"])),
        max_relative_range: 1.0,
    };
    let (summary, posts) = grid.evaluate_in_force(&estimator, &prep, id, &mut ws, &ctx()).unwrap();
    assert_eq!(summary.family, PriorSensitivityFamily::ResolvedPriorVariance);
    assert_eq!(summary.variance_multipliers.to_vec(), arr(&spec["resolved"]["multipliers"]));
    assert_means(&summary, &posts, &want["resolved"], "resolved");
    // The multiplier-one point is the prior in force itself.
    let own = fit(&BayesianGComputationAte { n_draws: 200, ..estimator.clone() }, &prep);
    assert_eq!(
        posts[1].draws.column(posts[1].effect_column().unwrap()).unwrap(),
        own.draws.column(own.effect_column().unwrap()).unwrap()
    );

    // External alpha multipliers.
    let sources = alpha_sources(spec, prep.design.ncols);
    let applied = arr(&spec["alpha"]["alphas_applied"]);
    let grid = PriorSensitivity {
        scales: Arc::from([]),
        alphas: Arc::from(arr(&spec["alpha"]["alphas"])),
        variance_multipliers: Arc::from([]),
        max_relative_range: 1.0,
    };
    let (summary, posts) = grid
        .evaluate_external_alpha(
            &sens_estimator(spec, None),
            &prep,
            id,
            &mut ws,
            &ctx(),
            ExternalAlphaSensitivity { sources: &sources, alphas_applied: &applied },
        )
        .unwrap();
    assert_eq!(summary.family, PriorSensitivityFamily::ExternalAlpha);
    assert_means(&summary, &posts, &want["alpha"], "alpha");
}

#[test]
fn ppc_prior_sensitivity_grids_read_every_option_and_refuse_bad_inputs() {
    let fixture = reference();
    let spec = &fixture["sens"];
    let prep = sens_problem(spec);
    let id = IdentificationStatus::NonparametricallyIdentified;
    let mut ws = BayesianGCompWorkspace::default();
    let plain = sens_estimator(spec, None);
    // n_draws below the cap is kept.
    let small = BayesianGComputationAte { n_draws: 50, ..plain.clone() };
    let (_, posts) =
        PriorSensitivity::standard_grid().evaluate(&small, &prep, id, &mut ws, &ctx()).unwrap();
    assert!(posts.iter().all(|p| p.draws.n_draws == 50));
    // An empty or invalid grid is refused.
    let empty = PriorSensitivity { scales: Arc::from([]), ..PriorSensitivity::standard_grid() };
    let empty_error = empty.evaluate(&plain, &prep, id, &mut ws, &ctx()).unwrap_err();
    assert!(format!("{empty_error:?}").contains("scale grid is empty"), "{empty_error:?}");
    let supplied = sens_estimator(spec, Some(resolved_prior(spec)));
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let grid = PriorSensitivity {
            variance_multipliers: Arc::from([bad]),
            ..PriorSensitivity::standard_resolved_grid()
        };
        let error =
            grid.evaluate_resolved_prior(&supplied, &prep, id, &mut ws, &ctx()).unwrap_err();
        assert!(
            format!("{error:?}").contains("variance multiplier must be finite and positive"),
            "{bad}: {error:?}"
        );
    }
    let no_prior = PriorSensitivity::standard_resolved_grid()
        .evaluate_resolved_prior(&plain, &prep, id, &mut ws, &ctx())
        .unwrap_err();
    assert!(format!("{no_prior:?}").contains("requires the prior in force"), "{no_prior:?}");
    let no_sources = PriorSensitivity::standard_alpha_grid()
        .evaluate_in_force(&plain, &prep, id, &mut ws, &ctx())
        .unwrap_err();
    assert!(format!("{no_sources:?}").contains("needs the prior bank's sources"), "{no_sources:?}");
    let sources = alpha_sources(spec, prep.design.ncols);
    for (multipliers, applied, expected) in [
        (vec![1.5], vec![0.8], "alpha multiplier must be finite and in [0, 1]"),
        (vec![0.5], vec![0.8, 0.8], "sources / alphas_applied length mismatch"),
        (vec![], vec![0.8], "alpha grid is empty"),
    ] {
        let grid = PriorSensitivity {
            scales: Arc::from([]),
            alphas: Arc::from(multipliers),
            variance_multipliers: Arc::from([]),
            max_relative_range: 1.0,
        };
        let error = grid
            .evaluate_external_alpha(
                &plain,
                &prep,
                id,
                &mut ws,
                &ctx(),
                ExternalAlphaSensitivity { sources: &sources, alphas_applied: &applied },
            )
            .unwrap_err();
        assert!(format!("{error:?}").contains(expected), "{expected}: {error:?}");
    }
    // The grid constructors.
    assert_eq!(PriorSensitivity::standard_grid().scales.to_vec(), [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]);
    assert_eq!(
        PriorSensitivity::standard_resolved_grid().variance_multipliers.to_vec(),
        [0.25, 0.5, 1.0, 2.0, 4.0, 10.0]
    );
    assert_eq!(
        PriorSensitivity::standard_alpha_grid().alphas.to_vec(),
        [0.0, 0.25, 0.5, 0.75, 1.0]
    );
}

#[test]
fn ppc_prior_sensitivity_report_equals_the_exact_rational_range() {
    let fixture = reference();
    for case in fixture["report_cases"].as_array().unwrap() {
        let sensitivity = PriorSensitivity {
            max_relative_range: case["max_relative_range"].as_f64().unwrap(),
            ..PriorSensitivity::default()
        };
        let summary = PriorSensitivitySummary {
            effect_means: Arc::from(arr(&case["effect_means"])),
            ..PriorSensitivitySummary::default()
        };
        let report = sensitivity.to_report(&summary, case["original"].as_f64().unwrap());
        let want = &case["expected"];
        assert_eq!(report.informative, want["informative"].as_bool().unwrap(), "{case}");
        assert_eq!(report.passed, want["passed"].as_bool().unwrap(), "{case}");
        match want["relative"].as_f64() {
            Some(rel) => close(report.comparison, rel, 1e-14, &case.to_string()),
            None => assert!(report.comparison.is_nan()),
        }
    }
    // A non-finite mean is uninformative and never passes.
    let summary = PriorSensitivitySummary {
        effect_means: Arc::from([1.0, f64::NAN]),
        ..PriorSensitivitySummary::default()
    };
    let report = PriorSensitivity::default().to_report(&summary, 1.0);
    assert!(!report.informative && !report.passed && report.comparison.is_nan());
}

// ------------------------------------------------------------------------------------------
// stream emission (ignored: run only to refresh the frozen file)
// ------------------------------------------------------------------------------------------

#[test]
#[ignore = "emits the frozen draw streams: PPC_REFERENCE_DATA=in.json PPC_REFERENCE_OUT=out.json"]
fn ppc_prior_sensitivity_emit() {
    let input: Value = serde_json::from_str(
        &std::fs::read_to_string(std::env::var("PPC_REFERENCE_DATA").unwrap()).unwrap(),
    )
    .unwrap();
    let mut out = json!({"post": {}, "prior": {}, "shape": {}, "sbc": {}, "sens": {}});
    for case in input["post"].as_array().unwrap() {
        out["post"][case["name"].as_str().unwrap()] = regen_post(case);
    }
    for case in input["prior"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        out["prior"][name] = if case["law"].as_bool().unwrap() {
            design_json(&problem(case))
        } else {
            regen_prior(case)
        };
    }
    let shape = &input["shape"];
    let prep = problem(shape);
    out["shape"] = merge(design_json(&prep), json!({"draws": shape["fit_draws"]}));
    for case in input["sbc"].as_array().unwrap() {
        out["sbc"][case["name"].as_str().unwrap()] = regen_sbc(case);
    }
    out["sens"] = regen_sens(&input["sens"]);
    std::fs::write(
        std::env::var("PPC_REFERENCE_OUT").unwrap(),
        serde_json::to_string(&out).unwrap(),
    )
    .unwrap();
}
