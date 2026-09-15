//! 1.9: Bayesian validation uses the prior in force (R-4, R-5), the temporal
//! serial-dependence posterior check (C-4), the tempering diagnostic (R-9) and
//! the HMC draw-floor diagnostic (B-5).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::too_many_lines, clippy::many_single_char_names)]

mod common;

use antecedent::validate::PredictiveCheckKind;
use antecedent::{BayesianConfig, ClassPrior, InferenceMode, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ExecutionContext, Lag, TemporalEffectQuery, TemporalPolicy,
    VariableId,
};
use antecedent_data::{TabularData, TimeSeriesData};
use antecedent_graph::{Dag, DenseNodeId, TemporalCpdag, TemporalDag, ensure_lagged};
use antecedent_prob::{GaussianCoefficientPrior, PriorSensitivityFamily, PriorSet, PriorSpec};
use common::calibration::{ar1_noise, gaussian};

const RESOLVED_GRID: [f64; 6] = [0.25, 0.5, 1.0, 2.0, 4.0, 10.0];

/// Treatment series of [`series_xy`] (AR(1), persistence at least 0.3).
fn treatment(n: usize, rho: f64, seed: u64) -> Vec<f64> {
    ar1_noise(n, rho.max(0.3), 0.5, 2 * seed + 1)
}

/// `y_t = beta x_{t-1} + e_t` with AR(1) treatment and residual (independent streams).
fn series_xy(n: usize, beta: f64, rho: f64, noise: f64, seed: u64) -> TimeSeriesData {
    let x = treatment(n, rho, seed);
    let e = ar1_noise(n, rho, noise, 2 * seed + 1_000_001);
    let mut y = vec![0.0; n];
    for t in 1..n {
        y[t] = beta * x[t - 1] + e[t];
    }
    TimeSeriesData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())], 1).unwrap()
}

fn lag_dag(lags: &[u32]) -> TemporalDag {
    let mut g = TemporalDag::empty();
    let y0 = ensure_lagged(&mut g, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    for &lag in lags {
        let x = ensure_lagged(&mut g, VariableId::from_raw(0), Lag::from_raw(lag)).unwrap();
        g.insert_directed(x, y0).unwrap();
    }
    g
}

fn pulse() -> TemporalEffectQuery {
    TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1)
}

fn multi_sustained() -> TemporalEffectQuery {
    TemporalEffectQuery::sustained(VariableId::from_raw(0), VariableId::from_raw(1), -2, 1.0)
        .with_policy(TemporalPolicy::sustained(-2, -1))
        .with_horizon_steps(1)
}

fn run_temporal(
    data: TimeSeriesData,
    graph: impl Into<antecedent::AcceptedGraph>,
    query: TemporalEffectQuery,
    cfg: BayesianConfig,
    suite: RefuteSuite,
) -> StudyResult {
    Study::series(data)
        .graph(graph.into())
        .query(CausalQuery::TemporalEffect(query))
        .inference(InferenceMode::Bayesian(cfg))
        .refute(suite)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(5))
        .unwrap()
}

fn effect_mean(result: &StudyResult) -> f64 {
    let post = result.posterior.as_ref().unwrap();
    post.summaries.mean[post.effect_column().unwrap()]
}

/// A tight source posterior (`beta ≈ 0.9`) banked as a transfer artifact.
fn informative_source() -> (Vec<u8>, [f64; 2]) {
    let source = run_temporal(
        series_xy(2_000, 0.9, 0.0, 0.02, 11),
        lag_dag(&[1]),
        pulse(),
        BayesianConfig::conjugate().n_draws(400),
        RefuteSuite::None,
    );
    let post = source.posterior.as_ref().unwrap();
    let names: Vec<_> = post
        .draws
        .schema
        .quantities
        .iter()
        .filter_map(|q| match q {
            antecedent_prob::PosteriorQuantityKind::Coefficient { name, .. } => name.clone(),
            _ => None,
        })
        .collect();
    assert_eq!(
        names.iter().map(AsRef::as_ref).collect::<Vec<&str>>(),
        ["intercept", "coef_x@lag1"],
        "temporal posteriors carry lag-aware coefficient names"
    );
    let bytes = antecedent::io::encode_causal_posterior_bytes(post, "source").unwrap();
    (bytes, [post.summaries.mean[0], post.summaries.mean[1]])
}

#[test]
fn temporal_full_validation_perturbs_and_checks_the_staged_prior() {
    let (bytes, source_coef) = informative_source();
    // Target data say beta = 0.2; the staged prior says 0.9.
    let target = series_xy(120, 0.2, 0.0, 0.35, 23);
    let x = treatment(120, 0.0, 23);
    let staged = run_temporal(
        target.clone(),
        lag_dag(&[1]),
        pulse(),
        BayesianConfig::conjugate().n_draws(400).prior_from_artifact(bytes, None),
        RefuteSuite::Full,
    );
    let isotropic = run_temporal(
        target,
        lag_dag(&[1]),
        pulse(),
        BayesianConfig::conjugate().n_draws(400),
        RefuteSuite::Full,
    );

    // R-4: the sensitivity grid is a variance-multiplier grid around the staged prior.
    let sens = staged.posterior.as_ref().unwrap().prior_sensitivity.as_ref().unwrap();
    assert_eq!(sens.family, PriorSensitivityFamily::ResolvedPriorVariance);
    assert_eq!(&*sens.variance_multipliers, &RESOLVED_GRID);
    assert!(sens.prior_scales.is_empty() && sens.alphas.is_empty());
    let reported = effect_mean(&staged);
    assert!((sens.effect_means[2] - reported).abs() < 0.02, "x1 grid point reproduces the fit");
    assert!(
        (sens.effect_means[0] - source_coef[1]).abs()
            < (sens.effect_means[5] - source_coef[1]).abs(),
        "tightening the staged prior must pull toward the source effect: {:?}",
        sens.effect_means
    );
    assert!(
        staged.refutations.iter().any(|r| r.refuter.as_ref() == "prior_sensitivity_resolved_prior"),
        "the report names the perturbed family"
    );
    // No prior supplied: the isotropic grid is the prior in force.
    let iso = isotropic.posterior.as_ref().unwrap().prior_sensitivity.as_ref().unwrap();
    assert_eq!(iso.family, PriorSensitivityFamily::IsotropicScale);
    assert!(isotropic.refutations.iter().any(|r| r.refuter.as_ref() == "prior_sensitivity"));

    // R-5: the prior predictive mean moves with the staged prior:
    // E[mean_t(b0 + b1 x_{t-1})] under the staged coefficient means.
    let prior_ppc = |result: &StudyResult| {
        result
            .predictive_checks
            .iter()
            .find(|c| c.kind == PredictiveCheckKind::Prior)
            .expect("prior PPC")
            .predictive_mean
    };
    let x_lag_mean = x[..x.len() - 1].iter().sum::<f64>() / (x.len() - 1) as f64;
    let expected = source_coef[0] + source_coef[1] * x_lag_mean;
    let staged_ppc = prior_ppc(&staged);
    assert!(
        (staged_ppc - expected).abs() < 0.02,
        "prior PPC must simulate from the staged prior: {staged_ppc} vs {expected}"
    );
    assert!((prior_ppc(&isotropic) - staged_ppc).abs() > 1e-6);
}

#[test]
fn class_envelope_full_validation_uses_each_completions_transferred_prior() {
    let (bytes, _) = informative_source();
    let mut gauss = gaussian(41);
    let n = 160;
    let mut t = vec![0.0; n];
    let mut z = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 1..n {
        z[i] = gauss();
        t[i] = 0.4 * z[i] + 0.5 * gauss();
        y[i] = 0.8 * t[i - 1] + 0.35 * gauss();
    }
    let data = TimeSeriesData::from_f64_columns(
        [("x", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())],
        1,
    )
    .unwrap();
    let mut g = TemporalCpdag::empty();
    let t1 = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let z1 = g.add_lagged(VariableId::from_raw(2), Lag::from_raw(1)).unwrap();
    g.insert_directed(z1, y0).unwrap();
    g.insert_directed(t1, y0).unwrap();
    g.insert_undirected(z1, t1).unwrap();
    let result = Study::series(data)
        .graph(g)
        .query(CausalQuery::TemporalEffect(pulse()))
        .class_prior(ClassPrior::from_ordered([0.5, 0.5]).unwrap())
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(200).prior_from_artifact(bytes, None),
        ))
        .refute(RefuteSuite::Full)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(9))
        .unwrap();
    let structural = result.structural_response.as_ref().expect("class atoms");
    let mut checked = 0;
    for atom in &structural.atoms {
        let Some(post) = atom.posterior.as_ref() else { continue };
        let sens = post.prior_sensitivity.as_ref().expect("full attaches sensitivity per atom");
        assert_eq!(sens.family, PriorSensitivityFamily::ResolvedPriorVariance);
        checked += 1;
    }
    assert!(checked >= 2, "both completions validated against their transferred prior");
    assert!(
        result
            .predictive_checks
            .iter()
            .filter(|c| c.kind == PredictiveCheckKind::Posterior)
            .all(|c| c.serial.is_some()),
        "temporal class atoms carry the serial discrepancy under full"
    );
    assert!(
        result
            .diagnostics
            .iter()
            .any(|d| d.code.as_ref() == "estimate.bayesian.temporal.dependence_correction"),
        "class atoms are fitted under the serial-dependence correction"
    );
}

#[test]
fn multi_step_isotropic_prior_is_the_prior_in_force() {
    let result = run_temporal(
        series_xy(160, 0.8, 0.5, 0.35, 3),
        lag_dag(&[1, 2]),
        multi_sustained(),
        BayesianConfig::conjugate().n_draws(200).prior_scale(4.0),
        RefuteSuite::Full,
    );
    let sens = result.posterior.as_ref().unwrap().prior_sensitivity.as_ref().unwrap();
    assert_eq!(sens.family, PriorSensitivityFamily::IsotropicScale);
    let posterior_checks: Vec<_> = result
        .predictive_checks
        .iter()
        .filter(|c| c.kind == PredictiveCheckKind::Posterior)
        .collect();
    assert!(!posterior_checks.is_empty() && posterior_checks.iter().all(|c| c.serial.is_some()));
    assert!(
        result
            .diagnostics
            .iter()
            .any(|d| d.code.as_ref() == "estimate.bayesian.temporal.dependence_correction")
    );
    // Transfer onto multi-step stays refused, so nothing but the isotropic prior can be in force.
    let (bytes, _) = informative_source();
    let err = Study::series(series_xy(160, 0.8, 0.5, 0.35, 3))
        .graph(lag_dag(&[1, 2]))
        .query(CausalQuery::TemporalEffect(multi_sustained()))
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().prior_from_artifact(bytes, None),
        ))
        .refute(RefuteSuite::Full)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(5))
        .unwrap_err();
    assert!(err.to_string().contains("isotropic per-mechanism priors"), "{err}");
}

fn static_fixture() -> (TabularData, Dag, AverageEffectQuery) {
    let mut gauss = gaussian(77);
    let n = 200;
    let mut t = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    for _ in 0..n {
        let zi = gauss();
        let ti = 0.5 * zi + gauss();
        t.push(ti);
        z.push(zi);
        y.push(0.3 * ti + 0.4 * zi + 0.5 * gauss());
    }
    let data = TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
    ])
    .unwrap();
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    (data, dag, AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)))
}

/// On a correctly specified confounded Gaussian model with default settings the
/// posterior predictive check passes: replicates carry the residual noise the
/// observed outcome carries. The default residual prior has no finite mean, so
/// the prior predictive scores only its one-sided dispersion axis, and a failure
/// there is worded as a statement about the diffuse prior.
#[test]
fn static_default_predictive_checks_pass_on_a_correct_gaussian_model() {
    let (data, dag, query) = static_fixture();
    let result = Study::tabular(data)
        .graph(dag)
        .query(query)
        .inference(InferenceMode::Bayesian(BayesianConfig::laplace()))
        .refute(RefuteSuite::Cheap)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(3))
        .unwrap();
    let verdict = |name: &str| {
        result.refutations.iter().find(|r| r.refuter.as_ref() == name).cloned().unwrap()
    };
    let posterior = verdict("posterior_predictive");
    assert!(posterior.passed, "{posterior:?}");
    let prior_check = result
        .predictive_checks
        .iter()
        .find(|c| c.kind == PredictiveCheckKind::Prior)
        .expect("prior PPC");
    assert_eq!(prior_check.noise, antecedent::validate::PredictiveNoise::MeanOnlyImproperScale);
    let prior = verdict("prior_predictive");
    assert!(
        prior.passed || prior.failure_condition.as_deref().is_some_and(|m| m.contains("diffuse")),
        "{prior:?}"
    );
}

#[test]
fn static_full_sensitivity_perturbs_an_explicit_prior() {
    let (data, dag, query) = static_fixture();
    let prior = PriorSet {
        specs: vec![PriorSpec::GaussianCoefficients(GaussianCoefficientPrior {
            mean: [0.0, 2.0, 0.0].into(),
            variance: [1.0, 1e-3, 1.0].into(),
        })],
        contrast: None,
        categorical: Vec::new(),
        restrictions: Vec::new(),
    };
    let result = Study::tabular(data)
        .graph(dag)
        .query(query)
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(200).prior(prior)))
        .refute(RefuteSuite::Full)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(3))
        .unwrap();
    let sens = result.posterior.as_ref().unwrap().prior_sensitivity.as_ref().unwrap();
    assert_eq!(sens.family, PriorSensitivityFamily::ResolvedPriorVariance);
    // Tightening the explicit prior (mean 2.0) pulls the effect up; weakening it lets the
    // data (0.3) win. An isotropic grid would have ignored the explicit prior entirely.
    assert!(sens.effect_means[0] > sens.effect_means[5], "{:?}", sens.effect_means);
    let prior_ppc = result
        .predictive_checks
        .iter()
        .find(|c| c.kind == PredictiveCheckKind::Prior)
        .expect("prior PPC");
    assert!(prior_ppc.serial.is_none(), "static rows are exchangeable: no serial axis");
}

#[test]
fn full_posterior_ppc_flags_serial_dependence_and_pins_a_value() {
    // AR(1) residual with a persistent treatment: the iid likelihood is misspecified.
    let ar = run_temporal(
        series_xy(200, 0.8, 0.7, 0.35, 5),
        lag_dag(&[1]),
        pulse(),
        BayesianConfig::conjugate().n_draws(200),
        RefuteSuite::Full,
    );
    let ppc = ar
        .predictive_checks
        .iter()
        .find(|c| c.kind == PredictiveCheckKind::Posterior)
        .expect("posterior PPC");
    let serial = ppc.serial.expect("full temporal PPC carries the serial axis");
    assert_eq!(serial.lag, 1);
    // Pinned on this fixture (seeded data, draws and replicates).
    assert!(
        (serial.observed - PINNED_LAG1_AUTOCORRELATION).abs() < 1e-9,
        "lag-1 residual autocorrelation drifted: {}",
        serial.observed
    );
    assert!(serial.p_value < 0.05, "AR(1) residuals must be flagged (p={})", serial.p_value);
    let report = ar
        .refutations
        .iter()
        .find(|r| r.refuter.as_ref() == "posterior_predictive")
        .expect("posterior predictive refutation");
    assert!(!report.passed);
    assert!(
        report.failure_condition.as_deref().is_some_and(|m| m.contains("serial-dependence")),
        "{:?}",
        report.failure_condition
    );
    // The failure is the dependence the tempering already widened the interval for,
    // and the verdict says so rather than presenting it as a refutation.
    assert!(serial.tempering_kappa.is_some_and(|k| k > 1.0), "{serial:?}");
    assert!(
        report
            .failure_condition
            .as_deref()
            .is_some_and(|m| m.contains("already tempered") && !m.contains("dispersion")),
        "{:?}",
        report.failure_condition
    );
    // The interval was tempered for that dependence, and the result says so.
    let correction = ar
        .diagnostics
        .iter()
        .find(|d| d.code.as_ref() == "estimate.bayesian.temporal.dependence_correction")
        .expect("dependence correction diagnostic");
    assert!(correction.message.contains("generalized (power) posterior"));
    let kappa = antecedent_estimate::tempering_kappa_from_notes(
        &ar.posterior.as_ref().unwrap().diagnostics.notes,
    )
    .expect("kappa note");
    assert!(kappa > 1.2, "persistent treatment + AR(1) residual inflates the scale: {kappa}");
    assert!(
        ar.estimate.assumptions.entries.iter().any(|a| matches!(
            &a.assumption,
            antecedent_core::Assumption::ParametricRestriction(p)
                if p.id.as_ref() == antecedent_estimate::DEPENDENCE_ASSUMPTION_ID
        )),
        "the generalized posterior is recorded as an assumption"
    );

    // iid residual: the serial axis passes, and cheap keeps the two-axis check.
    let iid = run_temporal(
        series_xy(200, 0.8, 0.0, 0.35, 6),
        lag_dag(&[1]),
        pulse(),
        BayesianConfig::conjugate().n_draws(200),
        RefuteSuite::Full,
    );
    let iid_serial = iid
        .predictive_checks
        .iter()
        .find(|c| c.kind == PredictiveCheckKind::Posterior)
        .and_then(|c| c.serial)
        .expect("serial axis");
    assert!(iid_serial.p_value > 0.05, "iid residuals must not be flagged ({iid_serial:?})");
    let cheap = run_temporal(
        series_xy(200, 0.8, 0.0, 0.35, 6),
        lag_dag(&[1]),
        pulse(),
        BayesianConfig::conjugate().n_draws(200),
        RefuteSuite::Cheap,
    );
    assert!(cheap.predictive_checks.iter().all(|c| c.serial.is_none()));
}

/// Mean lag-1 residual autocorrelation on the `full_posterior_ppc_flags_serial_dependence`
/// fixture (AR(1) rho 0.7, n 200, seed 5). The posterior draws depend on the
/// tempering factor, so a change to its estimator moves this pin.
const PINNED_LAG1_AUTOCORRELATION: f64 = 0.693_134_463_691_538_7;

#[test]
fn hmc_draw_floor_raises_draws_with_a_diagnostic() {
    // Randomized binary treatment, no confounder: a well-conditioned two-coefficient GLM.
    let mut gauss = gaussian(11);
    let t: Vec<f64> = (0..100).map(|i| f64::from(u8::from(i % 2 == 0))).collect();
    let y: Vec<f64> = t.iter().map(|t| 0.5 * t + 0.5 * gauss()).collect();
    let data = TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice())]).unwrap();
    let mut dag = Dag::with_variables(2);
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let result = Study::tabular(data)
        .graph(dag)
        .query(query)
        .inference(InferenceMode::Bayesian(BayesianConfig::hmc().n_draws(50)))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(3))
        .unwrap();
    let floor = result
        .diagnostics
        .iter()
        .find(|d| d.code.as_ref() == "estimate.bayesian.hmc_draw_floor")
        .expect("HMC floor diagnostic");
    assert!(floor.message.contains("from 50 to 3000"), "{}", floor.message);
    assert!(result.posterior.as_ref().unwrap().draws.n_draws >= 3_000);
}

/// Whether `bootstrap.ci_coverage` produced a report, or was skipped with a
/// not-applicable diagnostic naming it.
fn bootstrap_check(result: &StudyResult) -> (bool, bool) {
    let ran = result.refutations.iter().any(|r| r.refuter.as_ref() == "bootstrap.ci_coverage");
    let skipped = result.diagnostics.iter().any(|d| {
        d.code.as_ref() == "refute.validator.not_applicable"
            && format!("{d:?}").contains("bootstrap")
            && format!("{d:?}").contains("least-squares")
    });
    (ran, skipped)
}

#[test]
fn least_squares_stand_in_is_checked_only_when_it_matches_the_posterior_mean() {
    // A tight isotropic scale shrinks the posterior mean far from least squares;
    // the least-squares block bootstrap would then report a spurious refutation,
    // so the check is not applicable. The default weak scale keeps the check.
    for (query, label) in [(pulse(), "pulse"), (multi_sustained(), "multi-step")] {
        let lags: &[u32] = if label == "pulse" { &[1] } else { &[1, 2] };
        let weak = run_temporal(
            series_xy(200, 0.8, 0.0, 0.35, 11),
            lag_dag(lags),
            query.clone(),
            BayesianConfig::conjugate().n_draws(200).prior_scale(10.0),
            RefuteSuite::Full,
        );
        assert_eq!(bootstrap_check(&weak), (true, false), "{label}: weak scale runs the check");
        let tight = run_temporal(
            series_xy(200, 0.8, 0.0, 0.35, 11),
            lag_dag(lags),
            query,
            BayesianConfig::conjugate().n_draws(200).prior_scale(0.01),
            RefuteSuite::Full,
        );
        assert_eq!(bootstrap_check(&tight), (false, true), "{label}: tight scale skips it");
    }
}
