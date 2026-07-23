//! Bayesian conformance: load every `conformance/bayesian/*/expected.json`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::too_many_lines, clippy::many_single_char_names)]

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use antecedent::io::{decode_causal_posterior_bytes, encode_causal_posterior_bytes};
use antecedent::validate::PredictiveCheckKind as FacadeKind;
use antecedent::{BayesianConfig, CausalAnalysis, InferenceMode, RefuteSuite};
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
use causal_graph::{Dag, DenseNodeId, TemporalDag, ensure_lagged};
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
        restrictions: Vec::new(),
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

    // Facade attaches both prior and posterior PPC when refute ≠ none.
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    let facade = CausalAnalysis::builder()
        .data(data)
        .graph(dag)
        .query(query)
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(80).prior_scale(10.0),
        ))
        .refute(RefuteSuite::PlaceboAndRcc)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    assert!(facade.predictive_checks.iter().any(|c| c.kind == FacadeKind::Prior));
    assert!(facade.predictive_checks.iter().any(|c| c.kind == FacadeKind::Posterior));
    if expected["require_finite_p_value"].as_bool().unwrap() {
        assert!(facade.predictive_checks.iter().all(|c| c.p_value.is_finite()));
    }
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
    let sens =
        PriorSensitivity { scales: Arc::from(scales.clone()), ..PriorSensitivity::standard_grid() };
    let (summary, _) = sens
        .evaluate(&bayes, &prep, IdentificationStatus::NonparametricallyIdentified, &mut ws, &ctx)
        .unwrap();
    assert_eq!(summary.prior_scales.len(), scales.len());
    if expected["require_finite_effect_means"].as_bool().unwrap() {
        assert!(summary.effect_means.iter().all(|m| m.is_finite()));
    }
}

#[test]
fn temporal_pulse() {
    use causal_core::{Lag, TemporalEffectQuery, TemporalPolicy};
    use causal_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
        TimeSeriesData, ValidityBitmap,
    };

    let expected = load_expected("temporal_pulse");
    let true_ate = expected["expected_ate"].as_f64().unwrap();
    let tol = expected["tolerance"].as_f64().unwrap();
    let n = usize::try_from(expected["n"].as_u64().unwrap()).expect("fixture n");
    let n_draws = usize::try_from(expected["n_draws"].as_u64().unwrap()).expect("fixture n_draws");

    let mut b = CausalSchemaBuilder::new();
    b.add_variable(
        "pressure",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "defect",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    let schema = b.build().unwrap();
    let mut pressure = vec![0.0; n];
    let mut defect = vec![0.0; n];
    for t in 1..n {
        pressure[t] = ((t as f64) * 0.04).sin();
        defect[t] = true_ate * pressure[t - 1];
    }
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(0),
                Arc::from(pressure),
                ValidityBitmap::all_valid(n),
            )
            .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(1),
                Arc::from(defect),
                ValidityBitmap::all_valid(n),
            )
            .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let series = TimeSeriesData::try_new(
        storage,
        TimeIndex {
            regularity: SamplingRegularity::Regular { interval_ns: 3_600_000_000_000 },
            length: n,
        },
    )
    .unwrap();
    let mut g = TemporalDag::empty();
    let p1 = ensure_lagged(&mut g, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let d0 = ensure_lagged(&mut g, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_directed(p1, d0).unwrap();
    let q = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1);

    let analysis = CausalAnalysis::builder()
        .series(series)
        .temporal_graph(g)
        .temporal_query(q)
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(n_draws).prior_scale(100.0),
        ))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let result = analysis.run(&ExecutionContext::for_tests(42)).unwrap();
    let post = result.posterior.as_ref().expect("posterior");
    let eq = post.effect_column().unwrap();
    let mean = post.summaries.mean[eq];
    assert!((mean - true_ate).abs() < tol, "mean={mean} expected={true_ate}");
    if expected["require_finite_p_below_zero"].as_bool().unwrap() {
        assert!(post.probability_below(0.0).unwrap().is_finite());
    }
    if expected["require_artifact_round_trip"].as_bool().unwrap() {
        let bytes = encode_causal_posterior_bytes(post, "temporal-pulse").unwrap();
        let (meta, _) = decode_causal_posterior_bytes(&bytes).unwrap();
        assert_eq!(meta.n_draws as usize, post.draws.n_draws);
    }
}

#[test]
fn prior_bank_catalog() {
    use causal_io::{
        CausalPosteriorWire, CompatibilityRejectReason, CompatibilityReport, DesignVariableRole,
        DesignVariableSummary, EstimandFingerprint, PosteriorQuantityWire, PriorCatalog,
        PriorSourceMeta, PriorSourceRef, TargetDesign, encode_posterior_artifact,
    };

    fn pack(id: &str, coef_names: Option<Vec<&str>>) -> Vec<u8> {
        let mut quantities = Vec::new();
        if let Some(names) = coef_names {
            for (i, n) in names.into_iter().enumerate() {
                quantities.push(PosteriorQuantityWire::Coefficient {
                    index: u32::try_from(i).unwrap(),
                    name: Some(n.into()),
                });
            }
        } else {
            for i in 0..2u32 {
                quantities.push(PosteriorQuantityWire::Coefficient { index: i, name: None });
            }
        }
        quantities.push(PosteriorQuantityWire::Effect { name: "ate".into() });
        let n_q = quantities.len();
        let meta = CausalPosteriorWire {
            quantities,
            n_draws: 2,
            mean: vec![0.0; n_q],
            sd: vec![1.0; n_q],
            q025: vec![-1.0; n_q],
            q975: vec![1.0; n_q],
            identification: "NonparametricallyIdentified".into(),
            unidentified_mass: 0.0,
            backend_id: "laplace".into(),
            converged: true,
            hessian_condition: 1.0,
            draws_encoding: "f64_le_colmajor".into(),
        };
        let draws = vec![0.0f64; n_q * 2];
        let art = encode_posterior_artifact(&meta, &draws, id, "0.1.0").unwrap();
        let mut buf = Vec::new();
        art.write_to(&mut buf).unwrap();
        buf
    }

    let expected = load_expected("prior_bank_catalog");
    let ate = EstimandFingerprint::new("ate", "t", "y");
    let design = vec![
        DesignVariableSummary::new("t", DesignVariableRole::Treatment),
        DesignVariableSummary::new("y", DesignVariableRole::Outcome),
        DesignVariableSummary::new("z", DesignVariableRole::Covariate),
    ];

    let match_bytes = pack("match", Some(vec!["intercept", "coef_t", "coef_z"]));
    let unnamed_bytes = pack("unnamed", None);

    let catalog = PriorCatalog::from_sources(vec![
        PriorSourceRef::with_bytes(
            PriorSourceMeta::new("match", ate.clone(), "NonparametricallyIdentified")
                .with_design(design.clone()),
            match_bytes,
        ),
        PriorSourceRef::from_meta(
            PriorSourceMeta::new(
                "wrong",
                EstimandFingerprint::new("ate", "t", "other_y"),
                "NonparametricallyIdentified",
            )
            .with_design(design.clone()),
        ),
        PriorSourceRef::with_bytes(
            PriorSourceMeta::new("unnamed", ate.clone(), "NonparametricallyIdentified")
                .with_design(design),
            unnamed_bytes,
        ),
    ]);
    let target = TargetDesign::new(ate, ["t", "y", "z"]);
    let reports = catalog.filter_compatible(&target);

    let accepted: Vec<&str> = reports
        .iter()
        .filter_map(|r| match r {
            CompatibilityReport::Compatible { artifact_id } => Some(artifact_id.as_str()),
            _ => None,
        })
        .collect();
    let expected_accepted: Vec<&str> =
        expected["accepted_ids"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    assert_eq!(accepted, expected_accepted);

    let partial: Vec<&str> = reports
        .iter()
        .filter_map(|r| match r {
            CompatibilityReport::Partial { artifact_id, .. } => Some(artifact_id.as_str()),
            _ => None,
        })
        .collect();
    let expected_partial: Vec<&str> =
        expected["partial_ids"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    assert_eq!(partial, expected_partial);

    for rej in expected["rejected"].as_array().unwrap() {
        let id = rej["artifact_id"].as_str().unwrap();
        let code = rej["reason_code"].as_str().unwrap();
        let found = reports.iter().find(|r| r.artifact_id() == id).expect("rejected id");
        match found {
            CompatibilityReport::Rejected {
                reason: CompatibilityRejectReason::EstimandMismatch { .. },
                ..
            } => assert_eq!(code, "estimand_mismatch"),
            other => panic!("expected estimand_mismatch for {id}, got {other:?}"),
        }
    }

    for (id, needles) in expected["partial_missing_contains"].as_object().unwrap() {
        let CompatibilityReport::Partial { missing, mappable, .. } =
            reports.iter().find(|r| r.artifact_id() == id).unwrap()
        else {
            panic!("expected partial for {id}");
        };
        for n in needles.as_array().unwrap() {
            let s = n.as_str().unwrap();
            assert!(missing.iter().any(|m| m == s), "{id} missing {s} in {missing:?}");
        }
        if let Some(map_needles) = expected["partial_mappable_contains"].get(id) {
            for n in map_needles.as_array().unwrap() {
                let s = n.as_str().unwrap();
                assert!(mappable.iter().any(|m| m == s), "{id} mappable {s} in {mappable:?}");
            }
        }
    }
}

#[test]
fn prior_bank_effect_map() {
    use std::sync::Arc;

    use antecedent::inference::{hydrate_mapping_from_io, hydrate_prior_from_posterior_bytes};
    use antecedent::io::encode_causal_posterior_bytes;
    use antecedent::{BayesianConfig, CausalAnalysis, InferenceMode, RefuteSuite};
    use causal_core::{
        Assumption, AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec,
        RoleHint, SmallRoleSet, ValueType, VariableId,
    };
    use causal_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
    };
    use causal_graph::{Dag, DenseNodeId};
    use causal_io::PriorMapping;
    use causal_prob::PriorSet;

    let expected = load_expected("prior_bank_effect_map");
    let tol = expected["mapped_mean_tol"].as_f64().unwrap();
    let n = 80usize;

    // Source A: reuse linear SCM (true ATE = 2).
    let (data_a, t_a, y_a, z_a) = linear_scm_table(n);
    let _ = z_a;
    let mut dag_a = Dag::with_variables(3);
    dag_a.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    dag_a.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    dag_a.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    let result_a = CausalAnalysis::builder()
        .data(data_a)
        .graph(dag_a)
        .query(AverageEffectQuery::binary_ate(t_a, y_a))
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(80).prior_scale(10.0),
        ))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(1))
        .unwrap();
    let post_a = result_a.posterior.as_ref().unwrap();
    let source_mean = post_a.summaries.mean[post_a.effect_column().unwrap()];
    let bytes = encode_causal_posterior_bytes(post_a, "source-a").unwrap();

    // Target B: Z, W, T, Y with same T/Y relationship + noise covariate W.
    let mut b = CausalSchemaBuilder::new();
    for (name, hint) in [
        ("Z", RoleHint::Context),
        ("W", RoleHint::Context),
        ("T", RoleHint::TreatmentCandidate),
        ("Y", RoleHint::OutcomeCandidate),
    ] {
        b.add_variable(
            name,
            ValueType::Continuous,
            SmallRoleSet::from_hint(hint),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let z = VariableId::from_raw(0);
    let w = VariableId::from_raw(1);
    let t = VariableId::from_raw(2);
    let y = VariableId::from_raw(3);
    let mut zv = vec![0.0; n];
    let mut wv = vec![0.0; n];
    let mut tv = vec![0.0; n];
    let mut yv = vec![0.0; n];
    for i in 0..n {
        zv[i] = (i as f64) * 0.1;
        wv[i] = ((i * 3) % 7) as f64 * 0.05;
        tv[i] = if i % 2 == 0 { 1.0 } else { 0.0 };
        // Different DGP than source A (ATE=2) so baseline sits away from the banked prior.
        yv[i] = 0.5 * tv[i] + 0.5 * zv[i];
    }
    let validity = ValidityBitmap::all_valid(n);
    let cols = vec![
        OwnedColumn::Float64(Float64Column::new(z, Arc::from(zv), validity.clone()).unwrap()),
        OwnedColumn::Float64(Float64Column::new(w, Arc::from(wv), validity.clone()).unwrap()),
        OwnedColumn::Float64(Float64Column::new(t, Arc::from(tv), validity.clone()).unwrap()),
        OwnedColumn::Float64(Float64Column::new(y, Arc::from(yv), validity).unwrap()),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let data_b = TabularData::new(storage);
    let mut dag_b = Dag::with_variables(4);
    dag_b.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    dag_b.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(3)).unwrap();
    dag_b.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    dag_b.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(3)).unwrap();
    dag_b.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(3)).unwrap();

    let baseline = CausalAnalysis::builder()
        .data(data_b.clone())
        .graph(dag_b.clone())
        .query(AverageEffectQuery::binary_ate(t, y))
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(80).prior_scale(10.0),
        ))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(1))
        .unwrap();
    let base_post = baseline.posterior.as_ref().unwrap();
    let base_mean = base_post.summaries.mean[base_post.effect_column().unwrap()];

    let mapped = CausalAnalysis::builder()
        .data(data_b.clone())
        .graph(dag_b.clone())
        .query(AverageEffectQuery::binary_ate(t, y))
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(80).prior_scale(10.0).prior_from_artifact(
                bytes.clone(),
                Some(PriorMapping::EffectFunctional { source_quantity: "ate".into() }),
            ),
        ))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(1))
        .unwrap();
    let map_post = mapped.posterior.as_ref().unwrap();
    let map_mean = map_post.summaries.mean[map_post.effect_column().unwrap()];

    assert!(
        (map_mean - source_mean).abs() < tol,
        "mapped mean {map_mean} not within {tol} of source {source_mean}"
    );
    if expected["mapped_closer_than_baseline"].as_bool().unwrap() {
        assert!(
            (map_mean - source_mean).abs() < (base_mean - source_mean).abs(),
            "mapped {map_mean} should be closer to source {source_mean} than baseline {base_mean}"
        );
    }
    for id in expected["required_assumption_ids"].as_array().unwrap() {
        let needle = id.as_str().unwrap();
        assert!(
            map_post.assumptions.entries.iter().any(|a| {
                matches!(
                    &a.assumption,
                    Assumption::PriorRestriction(pa) if pa.id.as_ref() == needle
                )
            }),
            "missing assumption id {needle}"
        );
    }

    // Unset mapping auto-selects EffectFunctional for heterogeneous designs.
    let auto = CausalAnalysis::builder()
        .data(data_b)
        .graph(dag_b)
        .query(AverageEffectQuery::binary_ate(t, y))
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate()
                .n_draws(80)
                .prior_scale(10.0)
                .prior_from_artifact(bytes.clone(), None),
        ))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(1))
        .unwrap();
    let auto_post = auto.posterior.as_ref().unwrap();
    let auto_mean = auto_post.summaries.mean[auto_post.effect_column().unwrap()];
    assert!(
        (auto_mean - source_mean).abs() < (base_mean - source_mean).abs(),
        "auto-mapped {auto_mean} should be closer to source {source_mean} than baseline {base_mean}"
    );
    assert!(
        auto_post.assumptions.entries.iter().any(|a| {
            matches!(
                &a.assumption,
                Assumption::PriorRestriction(pa) if pa.id.as_ref() == "external_effect_prior"
            )
        }),
        "auto-mapped path should record external_effect_prior"
    );

    let names: Vec<Arc<str>> =
        ["intercept", "coef_T", "coef_Z", "coef_W"].into_iter().map(Arc::from).collect();
    let baseline_prior = PriorSet::weakly_informative(4);
    let mapping = hydrate_mapping_from_io(&PriorMapping::IdenticalCoefficientSubspace);
    assert!(
        hydrate_prior_from_posterior_bytes(&bytes, &mapping, &baseline_prior, &names, Some(1))
            .is_err(),
        "identical mapping should fail on ncols mismatch"
    );
}

#[test]
fn prior_bank_power_mixture() {
    use std::sync::Arc;

    use causal_prob::{
        ExternalPriorSource, ExternalPriorWeight, GaussianCoefficientPrior, PriorSet, PriorSpec,
        compose_external_priors,
    };

    let expected = load_expected("prior_bank_power_mixture");
    let tol = expected["tol"].as_f64().unwrap();
    let baseline_mean = expected["baseline_mean"].as_f64().unwrap();
    let baseline_var = expected["baseline_variance"].as_f64().unwrap();
    let source_mean = expected["source_mean"].as_f64().unwrap();
    let source_var = expected["source_variance"].as_f64().unwrap();
    let alpha = expected["alpha"].as_f64().unwrap();

    let mut baseline = PriorSet::new();
    baseline.push(PriorSpec::GaussianCoefficients(
        GaussianCoefficientPrior::shared(1, baseline_mean, baseline_var).unwrap(),
    ));
    let mut source_prior = PriorSet::new();
    source_prior.push(PriorSpec::GaussianCoefficients(
        GaussianCoefficientPrior::shared(1, source_mean, source_var).unwrap(),
    ));
    let sources = [ExternalPriorSource {
        id: Arc::from("old"),
        prior: source_prior,
        weight: ExternalPriorWeight::power(alpha).unwrap(),
    }];
    let composed = compose_external_priors(&sources, &baseline).unwrap();
    let coef = composed.prior.gaussian_coefficients().unwrap();
    let lam = 1.0 / coef.variance[0];
    assert!(
        (lam - expected["expected_precision"].as_f64().unwrap()).abs() < tol,
        "precision {lam}"
    );
    assert!((coef.mean[0] - expected["expected_mean"].as_f64().unwrap()).abs() < tol);
    assert!((coef.variance[0] - expected["expected_variance"].as_f64().unwrap()).abs() < tol);
    for id in expected["required_assumption_ids"].as_array().unwrap() {
        let needle = id.as_str().unwrap();
        assert!(
            composed.prior.restrictions.iter().any(|r| r.id.as_ref() == needle),
            "missing restriction {needle}"
        );
    }
}

#[test]
fn prior_bank_conflict_shrink() {
    use std::sync::Arc;

    use causal_prob::{
        ExternalPriorSource, ExternalPriorWeight, GaussianCoefficientPrior, PriorSet, PriorSpec,
    };
    use causal_validate::{ConflictPolicy, ConflictSignals, apply_conflict_and_compose};

    let expected = load_expected("prior_bank_conflict_shrink");
    let alpha = expected["alpha"].as_f64().unwrap();
    let policy = ConflictPolicy::try_new(
        expected["p_min"].as_f64().unwrap(),
        expected["kl_scale"].as_f64().unwrap(),
    )
    .unwrap();

    let mut baseline = PriorSet::new();
    baseline.push(PriorSpec::GaussianCoefficients(
        GaussianCoefficientPrior::shared(1, 0.0, 4.0).unwrap(),
    ));
    let mut source_prior = PriorSet::new();
    source_prior.push(PriorSpec::GaussianCoefficients(
        GaussianCoefficientPrior::shared(1, 50.0, 0.25).unwrap(),
    ));
    let sources = [ExternalPriorSource {
        id: Arc::from("src"),
        prior: source_prior,
        weight: ExternalPriorWeight::power(alpha).unwrap(),
    }];

    let conf = &expected["conflict"];
    let (composed_c, summary_c) = apply_conflict_and_compose(
        &sources,
        &baseline,
        &policy,
        &[ConflictSignals {
            p_value: Some(conf["p_value"].as_f64().unwrap()),
            kl: Some(conf["kl"].as_f64().unwrap()),
        }],
    )
    .unwrap();
    if conf["expect_alpha_strictly_less"].as_bool().unwrap() {
        assert!(summary_c.alphas_applied[0] < summary_c.alphas_requested[0]);
    }
    assert!(
        summary_c.alphas_applied[0] <= conf["expect_alpha_applied_max"].as_f64().unwrap() + 1e-15
    );
    assert!((composed_c.alphas_applied[0] - summary_c.alphas_applied[0]).abs() < 1e-15);

    let nc = &expected["no_conflict"];
    let (composed_n, summary_n) = apply_conflict_and_compose(
        &sources,
        &baseline,
        &policy,
        &[ConflictSignals {
            p_value: Some(nc["p_value"].as_f64().unwrap()),
            kl: Some(nc["kl"].as_f64().unwrap()),
        }],
    )
    .unwrap();
    let tol = nc["tol"].as_f64().unwrap();
    if nc["expect_alpha_unchanged"].as_bool().unwrap() {
        assert!((summary_n.alphas_applied[0] - alpha).abs() < tol);
        assert!((composed_n.alphas_applied[0] - alpha).abs() < tol);
    }
}

#[test]
fn prior_bank_transport() {
    use std::sync::Arc;

    use causal_prob::{
        ExternalPriorSource, ExternalPriorWeight, GaussianCoefficientPrior, PriorSet, PriorSpec,
        TransportContext, TransportError, TransportPolicy, apply_transport, compose_with_transport,
    };

    let expected = load_expected("prior_bank_transport");
    let tol = expected["tol"].as_f64().unwrap();
    let alpha = expected["alpha"].as_f64().unwrap();
    let source_pop = expected["source_population"].as_str().unwrap();
    let target_pop = expected["target_population"].as_str().unwrap();

    let mut baseline = PriorSet::new();
    baseline.push(PriorSpec::GaussianCoefficients(
        GaussianCoefficientPrior::shared(
            1,
            expected["baseline_mean"].as_f64().unwrap(),
            expected["baseline_variance"].as_f64().unwrap(),
        )
        .unwrap(),
    ));
    let mut source_prior = PriorSet::new();
    source_prior.push(PriorSpec::GaussianCoefficients(
        GaussianCoefficientPrior::shared(
            1,
            expected["source_mean"].as_f64().unwrap(),
            expected["source_variance"].as_f64().unwrap(),
        )
        .unwrap(),
    ));
    let sources = [ExternalPriorSource {
        id: Arc::from("src"),
        prior: source_prior,
        weight: ExternalPriorWeight::power(alpha).unwrap(),
    }];

    let missing = TransportContext {
        source_populations: &[Some(source_pop)],
        target_population: Some(target_pop),
        policy: None,
        adjustment: None,
        coef_index: None,
    };
    let err = apply_transport(&sources, &missing).unwrap_err();
    assert_eq!(err.code(), expected["error_code"].as_str().unwrap());
    assert!(matches!(err, TransportError::PolicyRequired { .. }));

    let with = &expected["with_policy"];
    let policy = TransportPolicy::parse(with["policy"].as_str().unwrap()).unwrap();
    let ctx = TransportContext {
        source_populations: &[Some(source_pop)],
        target_population: Some(target_pop),
        policy: Some(policy),
        adjustment: None,
        coef_index: None,
    };
    let (composed, outcomes) = compose_with_transport(&sources, &baseline, &ctx).unwrap();
    assert!(outcomes[0].required);
    let coef = composed.prior.gaussian_coefficients().unwrap();
    assert!(coef.mean[0].is_finite());
    assert!(coef.variance[0].is_finite() && coef.variance[0] > 0.0);
    if with["expect_alpha_unchanged"].as_bool().unwrap() {
        assert!((composed.alphas_applied[0] - alpha).abs() < tol);
    }
    for id in with["required_assumption_ids"].as_array().unwrap() {
        let needle = id.as_str().unwrap();
        assert!(
            composed.prior.restrictions.iter().any(|r| r.id.as_ref() == needle),
            "missing restriction {needle}"
        );
    }

    let prop = &expected["propensity_missing_weights"];
    let prop_policy = TransportPolicy::parse(prop["policy"].as_str().unwrap()).unwrap();
    let prop_ctx = TransportContext {
        source_populations: &[Some(source_pop)],
        target_population: Some(target_pop),
        policy: Some(prop_policy),
        adjustment: None,
        coef_index: None,
    };
    let (composed_p, _) = compose_with_transport(&sources, &baseline, &prop_ctx).unwrap();
    assert!(
        (composed_p.alphas_applied[0] - prop["expect_alpha_applied"].as_f64().unwrap()).abs() < tol
    );
    for id in prop["required_assumption_ids"].as_array().unwrap() {
        let needle = id.as_str().unwrap();
        assert!(
            composed_p.prior.restrictions.iter().any(|r| r.id.as_ref() == needle),
            "missing restriction {needle}"
        );
    }
}

#[test]
fn prior_bank_alpha_sensitivity() {
    use std::sync::Arc;

    use causal_core::{Assumption, CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet};
    use causal_prob::{
        ExternalPriorSource, ExternalPriorWeight, GaussianCoefficientPrior, PriorSet, PriorSpec,
        compose_external_priors,
    };

    let expected = load_expected("prior_bank_alpha_sensitivity");
    let n = usize::try_from(expected["n"].as_u64().unwrap()).expect("n");
    let n_draws = usize::try_from(expected["n_draws"].as_u64().unwrap()).expect("n_draws");
    let source_mean = expected["source_treatment_mean"].as_f64().unwrap();
    let source_var = expected["source_coef_variance"].as_f64().unwrap();
    let alpha = expected["alpha"].as_f64().unwrap();
    let multipliers: Vec<f64> = expected["alpha_multipliers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    let mut b = CausalSchemaBuilder::new();
    b.add_variable(
        "t",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "y",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "z",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::Context),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    let schema = b.build().unwrap();
    let t = VariableId::from_raw(0);
    let y = VariableId::from_raw(1);
    let z = VariableId::from_raw(2);
    let tv: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
    let zv: Vec<f64> = (0..n).map(|i| i as f64 * 0.05).collect();
    // Data ATE ≈ 2; banked prior pulls treatment coef toward `source_mean`.
    let yv: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * tv[i] + 0.3 * zv[i]).collect();
    let validity = ValidityBitmap::all_valid(n);
    let cols = vec![
        OwnedColumn::Float64(Float64Column::new(t, Arc::from(tv), validity.clone()).unwrap()),
        OwnedColumn::Float64(Float64Column::new(y, Arc::from(yv), validity.clone()).unwrap()),
        OwnedColumn::Float64(Float64Column::new(z, Arc::from(zv), validity).unwrap()),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let data = TabularData::new(storage);
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();

    // Probe design ncols / treatment column via a throwaway prepare.
    let probe = BayesianGComputationAte {
        backend: BayesianBackendKind::ConjugateGaussian,
        n_draws,
        seed: 11,
        ..BayesianGComputationAte::new()
    };
    let estimand =
        IdentifiedEstimand::backdoor("backdoor.adjustment", Arc::from([z]), ExprId::from_raw(0));
    let query = AverageEffectQuery::binary_ate(t, y);
    let prep = probe.prepare(&data, &estimand, &query).unwrap();
    let ncols = prep.design.ncols;
    let t_col = prep.design.treatment_column().expect("treatment column");
    let mut mean = vec![0.0; ncols];
    mean[t_col] = source_mean;
    let mut source_prior = PriorSet::new();
    source_prior.push(PriorSpec::GaussianCoefficients(GaussianCoefficientPrior {
        mean: Arc::from(mean),
        variance: Arc::from(vec![source_var; ncols]),
    }));
    let sources = Arc::<[ExternalPriorSource]>::from(vec![ExternalPriorSource {
        id: Arc::from("survey_a"),
        prior: source_prior,
        weight: ExternalPriorWeight::power(alpha).unwrap(),
    }]);
    let baseline = PriorSet::weakly_informative(ncols);
    let composed = compose_external_priors(&sources, &baseline).unwrap();

    let result = CausalAnalysis::builder()
        .data(data)
        .graph(dag)
        .query(query)
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(n_draws).prior_from_composed(
                Arc::clone(&sources),
                composed,
                None,
            ),
        ))
        .refute(RefuteSuite::Full)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(1))
        .unwrap();

    let post = result.posterior.as_ref().expect("posterior");
    let sens = post.prior_sensitivity.as_ref().expect("prior_sensitivity");
    assert!(sens.prior_scales.is_empty(), "external path should use α grid, not scales");
    assert_eq!(sens.alphas.as_ref(), multipliers.as_slice());
    if expected["require_finite_effect_means"].as_bool().unwrap() {
        assert!(sens.effect_means.iter().all(|m| m.is_finite()));
    }
    let m0 = sens.effect_means[0];
    let m1 = *sens.effect_means.last().unwrap();
    if expected["m1_closer_to_source_than_m0"].as_bool().unwrap() {
        assert!(
            (m1 - source_mean).abs() < (m0 - source_mean).abs(),
            "m=1 mean {m1} should be closer to {source_mean} than m=0 mean {m0}"
        );
    }
    // External compose must record prior restrictions (power-prior assumptions).
    assert!(
        post.assumptions
            .entries
            .iter()
            .any(|a| { matches!(&a.assumption, Assumption::PriorRestriction(_)) }),
        "expected prior restriction assumptions from composed prior"
    );
}

#[test]
fn temporal_composed_prior_conflict_and_alpha_grid() {
    use causal_core::{Lag, TemporalEffectQuery, TemporalPolicy};
    use causal_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
        TimeSeriesData, ValidityBitmap,
    };
    use causal_estimate::{BayesianTemporalGcomp, TemporalLinearAdjustment};
    use causal_identify::TemporalBackdoorIdentifier;
    use causal_prob::{
        ExternalPriorSource, ExternalPriorWeight, GaussianCoefficientPrior, PriorSet, PriorSpec,
        compose_external_priors,
    };
    use causal_validate::{ConflictPolicy, ExternalAlphaSensitivity, PriorSensitivity};

    let n = 80;
    let n_draws = 64;
    let true_ate = 0.9;

    let mut b = CausalSchemaBuilder::new();
    b.add_variable(
        "pressure",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "defect",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    let schema = b.build().unwrap();
    let mut pressure = vec![0.0; n];
    let mut defect = vec![0.0; n];
    for t in 1..n {
        pressure[t] = ((t as f64) * 0.04).sin();
        defect[t] = true_ate * pressure[t - 1];
    }
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(0),
                Arc::from(pressure),
                ValidityBitmap::all_valid(n),
            )
            .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(1),
                Arc::from(defect),
                ValidityBitmap::all_valid(n),
            )
            .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let series = TimeSeriesData::try_new(
        storage,
        TimeIndex {
            regularity: SamplingRegularity::Regular { interval_ns: 3_600_000_000_000 },
            length: n,
        },
    )
    .unwrap();
    let mut g = TemporalDag::empty();
    let p1 = ensure_lagged(&mut g, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let d0 = ensure_lagged(&mut g, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_directed(p1, d0).unwrap();
    let q = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1);

    let id_res = TemporalBackdoorIdentifier::new().identify_temporal(&g, &q).unwrap();
    let estimand = id_res.result.estimands[0].clone();
    let mut temporal_est = TemporalLinearAdjustment::new();
    temporal_est.inner.overlap = causal_estimate::OverlapPolicy::ExplicitOverride;
    let ctx = ExecutionContext::for_tests(42);
    let prep = temporal_est
        .prepare(&series, &estimand, &q, &id_res.indexer, None, &ctx.kernel_policy)
        .unwrap();
    let bprep = BayesianGComputationAte::from_prepared_estimation(&prep);
    let ncols = bprep.design.ncols;
    let t_col = bprep.design.treatment_column().unwrap_or(ncols.saturating_sub(1));

    let mut mean = vec![0.0; ncols];
    mean[t_col] = true_ate;
    let mut source_prior = PriorSet::new();
    source_prior.push(PriorSpec::GaussianCoefficients(GaussianCoefficientPrior {
        mean: Arc::from(mean),
        variance: Arc::from(vec![0.25; ncols]),
    }));
    let sources = Arc::<[ExternalPriorSource]>::from(vec![ExternalPriorSource {
        id: Arc::from("temporal_bank"),
        prior: source_prior,
        weight: ExternalPriorWeight::power(1.0).unwrap(),
    }]);
    let baseline = PriorSet::weakly_informative(ncols);
    let composed = compose_external_priors(&sources, &baseline).unwrap();
    let policy = ConflictPolicy::try_new(0.05, 1.0).unwrap();

    // Facade path: conflict re-shrink attaches (Full refute is separately limited on
    // temporal by DataSubset masks; α-grid is exercised directly below).
    let result = CausalAnalysis::builder()
        .series(series)
        .temporal_graph(g)
        .temporal_query(q)
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(n_draws).prior_from_composed(
                Arc::clone(&sources),
                composed.clone(),
                Some(policy),
            ),
        ))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();

    let post = result.posterior.as_ref().expect("posterior");
    assert!(
        post.conflict_summary.is_some(),
        "temporal path should attach conflict summary for banked prior"
    );
    assert!(
        result.diagnostics.iter().any(|d| d.code.as_ref() == "bayes.prior_bank.conflict"),
        "expected conflict diagnostic"
    );

    // α-grid on the temporal prepared design (same branch Full uses when reachable).
    let mut est = BayesianTemporalGcomp {
        inner: BayesianGComputationAte {
            backend: BayesianBackendKind::ConjugateGaussian,
            n_draws,
            seed: 42,
            prior: Some(composed.prior.clone()),
            ..BayesianGComputationAte::new()
        },
    };
    let sens = PriorSensitivity::standard_alpha_grid();
    let mut ws = BayesianGCompWorkspace::default();
    let alphas_applied = Arc::clone(&composed.alphas_applied);
    let (summary, _) = sens
        .evaluate_external_alpha(
            &est.inner,
            &bprep,
            IdentificationStatus::NonparametricallyIdentified,
            &mut ws,
            &ctx,
            ExternalAlphaSensitivity { sources: &sources, alphas_applied: &alphas_applied },
        )
        .unwrap();
    assert!(summary.prior_scales.is_empty());
    assert!(!summary.alphas.is_empty());
    assert!(summary.effect_means.iter().all(|m| m.is_finite()));
    let _ = &mut est; // keep mut for parity with execute path
}
