//! Bayesian conformance: load every `conformance/bayesian/*/expected.json`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::too_many_lines, clippy::many_single_char_names)]

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use causal::{decode_causal_posterior_bytes, encode_causal_posterior_bytes};
use causal_core::{
    AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint,
    SmallRoleSet, ValueType, VariableId,
};
use causal_data::{Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap};
use causal_estimate::{
    BayesianBackendKind, BayesianGCompWorkspace, BayesianGComputationAte, EnvelopeOptions,
    EstimationWorkspace, GraphEffectDraws, LinearAdjustmentAte, aggregate_effect_envelope,
    nonidentified_with_prior,
};
use causal_expr::{ExprId, IdentifiedEstimand};
use causal_identify::IdentificationStatus;
use causal_prob::{
    BayesDesignRef, BayesFitOptions, BayesLikelihood, ConjugateGaussianBackend,
    GaussianCoefficientPrior, GraphIdentFlag, InferenceBackend, InferenceDiagnostics,
    LaplaceGlmBackend, LaplaceWorkspace, PriorSet, PriorSpec, WeightedGraphSamples,
};
use causal_validate::{
    PosteriorPredictiveCheck, PredictiveCheckKind, PriorPredictiveCheck, PriorSensitivity,
};
use serde_json::Value as JsonValue;

fn fixture_dir(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../conformance/bayesian").join(name)
}

fn load_expected(name: &str) -> JsonValue {
    let raw = fs::read_to_string(fixture_dir(name).join("expected.json")).expect("expected.json");
    serde_json::from_str(&raw).expect("parse expected.json")
}

fn linear_scm_table(n: usize) -> (TabularData, VariableId, VariableId, VariableId) {
    let mut b = CausalSchemaBuilder::new();
    b.add_variable(
        "Z",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::Context),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "T",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "Y",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    let schema = b.build().unwrap();
    let z = VariableId::from_raw(0);
    let t = VariableId::from_raw(1);
    let y = VariableId::from_raw(2);
    let mut zv = vec![0.0; n];
    let mut tv = vec![0.0; n];
    let mut yv = vec![0.0; n];
    for i in 0..n {
        zv[i] = (i as f64) * 0.1;
        tv[i] = if i % 2 == 0 { 1.0 } else { 0.0 };
        yv[i] = 2.0 * tv[i] + 0.5 * zv[i];
    }
    let validity = ValidityBitmap::all_valid(n);
    let cols = vec![
        OwnedColumn::Float64(Float64Column::new(z, Arc::from(zv), validity.clone()).unwrap()),
        OwnedColumn::Float64(Float64Column::new(t, Arc::from(tv), validity.clone()).unwrap()),
        OwnedColumn::Float64(Float64Column::new(y, Arc::from(yv), validity).unwrap()),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    (TabularData::new(storage), t, y, z)
}

#[test]
fn shared_functional_ate() {
    let expected = load_expected("shared_functional_ate");
    let true_ate = expected["true_ate"].as_f64().unwrap();
    let tol = expected["tolerance"].as_f64().unwrap();
    assert_eq!(expected["frequentist"].as_str().unwrap(), "linear_adjustment");
    assert_eq!(expected["bayesian"].as_str().unwrap(), "conjugate_gcomp");

    let n = 80;
    let (data, t, y, z) = linear_scm_table(n);
    let estimand = IdentifiedEstimand::backdoor(
        "backdoor.adjustment",
        Arc::from(vec![z]),
        ExprId::from_raw(0),
    );
    let query = AverageEffectQuery::binary_ate(t, y);

    let freq = LinearAdjustmentAte { bootstrap_replicates: 0, ..LinearAdjustmentAte::new() };
    let prep = freq.prepare(&data, &estimand, &query).unwrap();
    let mut ws = EstimationWorkspace::default();
    let freq_est = freq
        .fit(&prep, &mut ws, &ExecutionContext::for_tests(1), causal_core::AssumptionSet::new())
        .unwrap();

    let bayes = BayesianGComputationAte {
        backend: BayesianBackendKind::ConjugateGaussian,
        n_draws: 400,
        seed: 5,
        prior_scale: 100.0,
        ..BayesianGComputationAte::new()
    };
    let bprep = bayes.prepare(&data, &estimand, &query).unwrap();
    let mut bws = BayesianGCompWorkspace::default();
    let post = bayes
        .fit(
            &bprep,
            IdentificationStatus::NonparametricallyIdentified,
            &mut bws,
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
    let eq = post.effect_column().unwrap();
    let mean = post.summaries.mean[eq];
    assert!((freq_est.ate - true_ate).abs() < 1e-6);
    assert!((mean - freq_est.ate).abs() < tol, "bayes={mean} freq={}", freq_est.ate);

    let bytes = encode_causal_posterior_bytes(&post, "shared-functional").unwrap();
    let (meta, _) = decode_causal_posterior_bytes(&bytes).unwrap();
    assert_eq!(meta.n_draws as usize, post.draws.n_draws);
}

#[test]
fn nonidentified_prior() {
    let expected = load_expected("nonidentified_prior");
    let prior = PriorSet::weakly_informative(3);
    let post = nonidentified_with_prior(&prior, InferenceDiagnostics::analytic("none"), 64, 1);
    assert_eq!(format!("{:?}", post.identification), expected["identification"].as_str().unwrap());
    assert!(
        (post.unidentified_mass - expected["unidentified_mass"].as_f64().unwrap()).abs() < 1e-12
    );
    assert_eq!(expected["prior_recorded"].as_bool().unwrap(), !post.assumptions.is_empty());
}

#[test]
fn conjugate_gaussian() {
    let expected = load_expected("conjugate_gaussian");
    let coefs = expected["true_coefficients"].as_array().unwrap();
    let true0 = coefs[0].as_f64().unwrap();
    let true1 = coefs[1].as_f64().unwrap();
    let tol = expected["tolerance"].as_f64().unwrap();
    assert_eq!(expected["backend"].as_str().unwrap(), "conjugate_gaussian");

    let n = 40;
    let mut x = vec![0.0; n * 2];
    let mut y = vec![0.0; n];
    for r in 0..n {
        let xi = r as f64;
        x[r] = 1.0;
        x[n + r] = xi;
        y[r] = true0 + true1 * xi;
    }
    let prior = PriorSet {
        specs: vec![
            PriorSpec::GaussianCoefficients(GaussianCoefficientPrior::isotropic(2, 100.0)),
            PriorSpec::KnownResidualVariance(1e-6),
        ],
        contrast: None,
        categorical: Vec::new(),
    };
    let mut ws = LaplaceWorkspace::default();
    let design =
        BayesDesignRef { x_colmajor: &x, nrows: n, ncols: 2, y: &y, weights: None, offsets: None };
    let opts = BayesFitOptions { n_draws: 200, seed: 42, ..BayesFitOptions::default() };
    let fit = ConjugateGaussianBackend
        .fit(
            BayesLikelihood::GaussianIdentity,
            design,
            &prior,
            &opts,
            &mut ws,
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
    assert!(fit.diagnostics.allows_posterior());
    assert!((fit.map[0] - true0).abs() < tol);
    assert!((fit.map[1] - true1).abs() < tol);
}

#[test]
fn laplace_glm() {
    let expected = load_expected("laplace_glm");
    let coefs = expected["true_coefficients"].as_array().unwrap();
    let true0 = coefs[0].as_f64().unwrap();
    let true1 = coefs[1].as_f64().unwrap();
    let tol = expected["tolerance"].as_f64().unwrap();
    assert_eq!(expected["backend"].as_str().unwrap(), "laplace");

    let n = 60;
    let mut x = vec![0.0; n * 2];
    let mut y = vec![0.0; n];
    for r in 0..n {
        let xi = (r as f64) * 0.1;
        x[r] = 1.0;
        x[n + r] = xi;
        y[r] = true0 + true1 * xi;
    }
    let prior = PriorSet::weakly_informative(2);
    let mut ws = LaplaceWorkspace::default();
    let design =
        BayesDesignRef { x_colmajor: &x, nrows: n, ncols: 2, y: &y, weights: None, offsets: None };
    let opts = BayesFitOptions { n_draws: 100, seed: 9, ..BayesFitOptions::default() };
    let fit = LaplaceGlmBackend
        .fit(
            BayesLikelihood::GaussianIdentity,
            design,
            &prior,
            &opts,
            &mut ws,
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
    assert!(fit.diagnostics.converged);
    assert!(fit.diagnostics.allows_posterior());
    assert!((fit.map[0] - true0).abs() < tol);
    assert!((fit.map[1] - true1).abs() < tol);
}

#[test]
fn graph_effect_envelope() {
    let expected = load_expected("graph_effect_envelope");
    let w_unid = expected["unidentified_mass"].as_f64().unwrap();
    let identified = expected["identified_weights"].as_array().unwrap();
    let effects = expected["effect_means"].as_array().unwrap();
    let mixture = expected["expected_mixture_mean"].as_f64().unwrap();

    let graphs = WeightedGraphSamples::new(
        vec![identified[0].as_f64().unwrap(), w_unid, identified[1].as_f64().unwrap()],
        vec![GraphIdentFlag::Identified, GraphIdentFlag::Unidentified, GraphIdentFlag::Identified],
        vec![1, 2, 3],
    )
    .unwrap();
    let e0 = effects[0].as_f64().unwrap();
    let e1 = effects[1].as_f64().unwrap();
    let per = vec![
        GraphEffectDraws { graph_key: 1, effect_draws: Arc::from(vec![e0, e0, e0]) },
        GraphEffectDraws { graph_key: 3, effect_draws: Arc::from(vec![e1, e1, e1]) },
    ];
    let env = aggregate_effect_envelope(
        &graphs,
        &per,
        InferenceDiagnostics::analytic("envelope"),
        EnvelopeOptions::default(),
    )
    .unwrap();
    assert!((env.unidentified_mass - w_unid).abs() < 1e-12);
    assert!((env.summaries.mean[0] - mixture).abs() < 1e-12);
}

#[test]
fn ppc() {
    let expected = load_expected("ppc");
    let checks = expected["checks"].as_array().unwrap();
    assert!(checks.iter().any(|c| c.as_str() == Some("prior_predictive")));
    assert!(checks.iter().any(|c| c.as_str() == Some("posterior_predictive")));

    let (data, t, y, z) = linear_scm_table(40);
    let estimand = IdentifiedEstimand::backdoor(
        "backdoor.adjustment",
        Arc::from(vec![z]),
        ExprId::from_raw(0),
    );
    let query = AverageEffectQuery::binary_ate(t, y);
    let bayes = BayesianGComputationAte {
        backend: BayesianBackendKind::ConjugateGaussian,
        n_draws: 100,
        seed: 2,
        prior_scale: 10.0,
        ..BayesianGComputationAte::new()
    };
    let prep = bayes.prepare(&data, &estimand, &query).unwrap();
    let ctx = ExecutionContext::for_tests(1);
    let prior_rep = PriorPredictiveCheck { n_sims: 50, seed: 3, ..PriorPredictiveCheck::new() }
        .check(&prep, &ctx)
        .unwrap();
    assert_eq!(prior_rep.kind, PredictiveCheckKind::Prior);
    if expected["require_finite_p_value"].as_bool().unwrap() {
        assert!(prior_rep.p_value.is_finite());
    }

    let mut ws = BayesianGCompWorkspace::default();
    let post =
        bayes.fit(&prep, IdentificationStatus::NonparametricallyIdentified, &mut ws, &ctx).unwrap();
    let post_rep = PosteriorPredictiveCheck { n_sims: 50, ..PosteriorPredictiveCheck::new() }
        .check(&prep, &post)
        .unwrap();
    assert_eq!(post_rep.kind, PredictiveCheckKind::Posterior);
    assert!(post_rep.p_value.is_finite());
}

#[test]
fn prior_sensitivity() {
    let expected = load_expected("prior_sensitivity");
    let scales: Vec<f64> =
        expected["scales"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();

    let (data, t, y, z) = linear_scm_table(40);
    let estimand = IdentifiedEstimand::backdoor(
        "backdoor.adjustment",
        Arc::from(vec![z]),
        ExprId::from_raw(0),
    );
    let query = AverageEffectQuery::binary_ate(t, y);
    let bayes = BayesianGComputationAte {
        backend: BayesianBackendKind::ConjugateGaussian,
        n_draws: 80,
        seed: 4,
        ..BayesianGComputationAte::new()
    };
    let prep = bayes.prepare(&data, &estimand, &query).unwrap();
    let mut ws = BayesianGCompWorkspace::default();
    let ctx = ExecutionContext::for_tests(1);
    let sens = PriorSensitivity {
        scales: Arc::from(scales.clone()),
        ..PriorSensitivity::standard_grid()
    };
    let (summary, _) = sens
        .evaluate(&bayes, &prep, IdentificationStatus::NonparametricallyIdentified, &mut ws, &ctx)
        .unwrap();
    assert_eq!(summary.prior_scales.len(), scales.len());
    if expected["require_finite_effect_means"].as_bool().unwrap() {
        assert!(summary.effect_means.iter().all(|m| m.is_finite()));
    }
}
