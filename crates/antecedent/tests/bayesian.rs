//! Bayesian conformance: load every `conformance/bayesian/*/expected.json`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::too_many_lines, clippy::many_single_char_names)]

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use antecedent::io::{decode_causal_posterior_bytes, encode_causal_posterior_bytes};
use antecedent::validate::PredictiveCheckKind as FacadeKind;
use antecedent::{AcceptedGraph, BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, Lag, MeasurementSpec, RoleHint,
    SmallRoleSet, TemporalEffectQuery, TemporalPolicy, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TabularData, TimeIndex,
    TimeSeriesData, ValidityBitmap,
};
use antecedent_estimate::{
    BayesianBackendKind, BayesianGCompWorkspace, BayesianGComputationAte, EnvelopeOptions,
    EstimationWorkspace, GraphEffectDraws, LinearAdjustmentAte, aggregate_effect_envelope,
    nonidentified_with_prior,
};
use antecedent_expr::{ExprId, IdentifiedEstimand};
use antecedent_graph::{Dag, DenseNodeId, TemporalDag, ensure_lagged};
use antecedent_identify::IdentificationStatus;
use antecedent_prob::{
    BayesDesignRef, BayesFitOptions, BayesLikelihood, ConjugateGaussianBackend,
    GaussianCoefficientPrior, GraphIdentFlag, InferenceBackend, InferenceDiagnostics,
    LaplaceGlmBackend, LaplaceWorkspace, PriorSet, PriorSpec, WeightedGraphSamples,
};
use antecedent_validate::{
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
        .fit(&prep, &mut ws, &ExecutionContext::for_tests(1), antecedent_core::AssumptionSet::new())
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
    assert!((mean - true_ate).abs() < tol, "bayes={mean} truth={true_ate}");
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
    let facade = Study::tabular(data)
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
    use antecedent_core::{Lag, TemporalEffectQuery, TemporalPolicy};
    use antecedent_data::{
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

    let analysis = Study::series(series)
        .graph(g)
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

/// Fixed-seed, dependency-free `SplitMix64` generator so the `temporal_sustained`
/// fixture's additive noise is exactly reproducible across runs and machines
/// (no external `rand` crate is a dependency of this test binary).
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform double in `(0, 1]` (never exactly 0, so `ln` below stays finite).
    fn next_f64(&mut self) -> f64 {
        // 53 bits of mantissa precision, then shifted into (0, 1].
        let bits = self.next_u64() >> 11;
        #[allow(clippy::cast_precision_loss)]
        let v = (bits as f64) / (1u64 << 53) as f64;
        1.0 - v
    }

    /// Standard-normal sample via Box-Muller (one of the two values per pair is discarded
    /// for simplicity; this is a test fixture, not a hot path).
    fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_f64();
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// Additive noise SD for the `temporal_sustained` fixture's outcome equation.
///
/// Chosen so the conjugate-Gaussian posterior spread is non-degenerate (large enough
/// that the `data.subset` stability refuter is comparing real replicate-to-replicate
/// estimation noise, not floating-point dust) while staying small enough that the
/// posterior mean recovers the true coefficient 0.7 comfortably within the fixture's
/// 0.05 tolerance at n=400. `pressure` has amplitude 1 and empirical SD ~0.71 over the
/// full sample (a sampled sinusoid), so with `sigma = 0.05` the OLS/conjugate standard
/// error on the slope is approximately `sigma / (sqrt(n) * sd(pressure))
/// ~= 0.05 / (20 * 0.71) ~= 0.0035` — about three orders of magnitude larger than the
/// ~1e-5 floating-point-only spread the noiseless SCM produced (making the posterior
/// genuinely non-degenerate) while still ~14x smaller than the 0.05 tolerance (so the
/// point estimate recovers 0.7 with a large, deterministic safety margin, not a
/// probabilistic one that could flake).
const SUSTAINED_NOISE_SD: f64 = 0.05;

/// Fixed seed for `SplitMix64`; any change here changes the fixture's exact numbers.
const SUSTAINED_NOISE_SEED: u64 = 0xA5EC_0DEF_5EED_0002;

/// Build the `temporal_sustained` fixture's series + lag-2 `TemporalDag`.
///
/// `defect_t = 0.7 * pressure_{t-2} + eps_t`, `eps_t ~ N(0, SUSTAINED_NOISE_SD^2)`,
/// drawn from a fixed-seed `SplitMix64` stream so the series is exactly reproducible
/// run to run — see `conformance/bayesian/temporal_sustained/expected.json` for the
/// full derivation of the true coefficient and the tolerance it implies. This graph
/// deliberately has *only* a lag-2 edge (no lag-1 edge), so a test that accidentally
/// reused the Pulse fixture's lag-1 wiring would fail to identify or recover the wrong
/// number.
fn sustained_scm_series(n: usize, true_effect: f64) -> (TimeSeriesData, TemporalDag) {
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
    let mut rng = SplitMix64::new(SUSTAINED_NOISE_SEED);
    // Draw noise for every row up front (including 0, 1) in a single fixed order so the
    // sequence — and therefore every downstream number — never depends on `n`.
    let noise: Vec<f64> = (0..n).map(|_| SUSTAINED_NOISE_SD * rng.next_gaussian()).collect();
    for t in 2..n {
        pressure[t] = ((t as f64) * 0.04).sin();
        defect[t] = true_effect * pressure[t - 2] + noise[t];
    }
    // pressure[0], pressure[1] also need values (they feed pressure[2], pressure[3]'s lag).
    pressure[0] = 0.0;
    pressure[1] = (0.04_f64).sin();
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
    let p2 = ensure_lagged(&mut g, VariableId::from_raw(0), Lag::from_raw(2)).unwrap();
    let d0 = ensure_lagged(&mut g, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_directed(p2, d0).unwrap();
    (series, g)
}

fn sustained_query() -> TemporalEffectQuery {
    TemporalEffectQuery::sustained(VariableId::from_raw(0), VariableId::from_raw(1), -2, 1.0)
        .with_policy(TemporalPolicy::sustained(-2, -2))
        .with_horizon_steps(1)
}

/// Multi-step `Sustained` (`until > from`) is refused by
/// `TemporalLinearAdjustment`/`BayesianGComputationAte::from_prepared_estimation`
/// (`refuse_multi_step_schedule`, `antecedent-estimate/src/temporal_adjustment.rs`):
/// a single-column regression cannot honor the multi-node contrast a genuine
/// multi-step Sustained estimand requires. Confirms that limitation still holds
/// so the single-step-only scope claimed by the fixture/limitations text stays
/// honest, rather than silently becoming stale if the estimator is ever extended.
#[test]
fn temporal_sustained_multi_step_is_refused() {
    let (series, g) = sustained_scm_series(400, 0.7);
    let q =
        TemporalEffectQuery::sustained(VariableId::from_raw(0), VariableId::from_raw(1), -1, 1.0)
            .with_policy(TemporalPolicy::sustained(-2, -1))
            .with_horizon_steps(1);
    let analysis = Study::series(series)
        .graph(g)
        .temporal_query(q)
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(64).prior_scale(100.0),
        ))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let err = analysis.run(&ExecutionContext::for_tests(42)).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("not supported") || msg.contains("single-step") || msg.contains("refus"),
        "expected a multi-step-refusal error, got: {msg}"
    );
}

/// Exercise one of the six licensed `SustainedEffect × TemporalDag × Bayesian`
/// coordinates on the **staged** `Study::prepare()` → `PreparedStudy::estimate_series()`
/// path (not the one-shot `.build().unwrap().run(&ctx)` sugar), for a given
/// structure source (`explicit` graph vs. `accepted` graph) and validation suite.
fn run_staged_sustained(
    accepted: bool,
    suite: RefuteSuite,
) -> Result<antecedent::StudyResult, antecedent::CausalError> {
    let expected = load_expected("temporal_sustained");
    let true_ate = expected["expected_ate"].as_f64().unwrap();
    let n = usize::try_from(expected["n"].as_u64().unwrap()).expect("fixture n");
    let n_draws = usize::try_from(expected["n_draws"].as_u64().unwrap()).expect("fixture n_draws");
    let (series, g) = sustained_scm_series(n, true_ate);
    let q = sustained_query();

    let mut builder = Study::series(series.clone())
        .temporal_query(q)
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(n_draws).prior_scale(100.0),
        ))
        .refute(suite)
        .bootstrap_replicates(0);
    builder =
        if accepted { builder.graph(AcceptedGraph::temporal_dag(g)) } else { builder.graph(g) };
    let study = builder.build().unwrap();
    let expected_structure = if accepted { "accepted" } else { "explicit" };
    assert_eq!(study.structure_source().as_str(), expected_structure);

    let ctx = ExecutionContext::for_tests(42);
    let prepared = study.prepare(&ctx).expect("staged prepare");
    prepared.estimate_series(&series, &ctx)
}

fn assert_sustained_mean(result: &antecedent::StudyResult) {
    let expected = load_expected("temporal_sustained");
    let true_ate = expected["expected_ate"].as_f64().unwrap();
    let tol = expected["tolerance"].as_f64().unwrap();
    let post = result.posterior.as_ref().expect("posterior");
    let eq = post.effect_column().unwrap();
    let mean = post.summaries.mean[eq];
    assert!((mean - true_ate).abs() < tol, "mean={mean} expected={true_ate}");
    if expected["require_finite_p_below_zero"].as_bool().unwrap() {
        assert!(post.probability_below(0.0).unwrap().is_finite());
    }
    if expected["require_artifact_round_trip"].as_bool().unwrap() {
        let bytes = encode_causal_posterior_bytes(post, "temporal-sustained").unwrap();
        let (meta, _) = decode_causal_posterior_bytes(&bytes).unwrap();
        assert_eq!(meta.n_draws as usize, post.draws.n_draws);
    }
    assert_eq!(result.support_status.unwrap().as_str(), "licensed");
}

#[test]
fn temporal_sustained_explicit_none() {
    let result = run_staged_sustained(false, RefuteSuite::None).expect("staged estimate_series");
    assert_sustained_mean(&result);
    assert!(
        result.refutations.is_empty(),
        "refute=None must not run refuters, got {:?}",
        result.refutations
    );
    assert!(
        result.predictive_checks.is_empty(),
        "refute=None must not run PPC, got {:?}",
        result.predictive_checks
    );
}

#[test]
fn temporal_sustained_accepted_none() {
    let result = run_staged_sustained(true, RefuteSuite::None).expect("staged estimate_series");
    assert_sustained_mean(&result);
    assert!(result.refutations.is_empty());
    assert!(result.predictive_checks.is_empty());
}

/// `RefuteSuite::Cheap` runs the overlap+E-value validator suite plus, on the
/// Bayesian temporal path, prior/posterior predictive checks whenever
/// `refute != None` (`execute_temporal`,
/// `crates/antecedent/src/analysis/execute/temporal_path.rs`).
///
/// **Known gap** (found while earning this cell, not fixed here — this test
/// file cannot touch `execute/temporal_path.rs`): unlike the static Bayesian
/// path (`execute_bayesian`, which pushes into both `refutations` *and*
/// `result.predictive_checks`), `execute_temporal` folds the prior/posterior
/// predictive checks into `refutations` only (as `RefutationReport`s named
/// `"prior_predictive"` / `"posterior_predictive"`) and never populates
/// `StudyResult::predictive_checks`. The PPC computation itself genuinely
/// runs (this is not the `execute_pag_bayesian`-style "claims a check it
/// skips" bug) — the check's *result* is just not surfaced through the field
/// callers would normally read it from. Assert against `refutations`, and
/// assert the gap explicitly so a future fix is visible as a test change
/// here, not a silent pass.
#[test]
fn temporal_sustained_explicit_cheap() {
    let result = run_staged_sustained(false, RefuteSuite::Cheap).expect("staged estimate_series");
    assert_sustained_mean(&result);
    assert!(
        result.refutations.iter().any(|r| r.refuter.as_ref() == "prior_predictive"),
        "cheap suite should run prior PPC on the Bayesian temporal path, got {:?}",
        result.refutations
    );
    assert!(
        result.refutations.iter().any(|r| r.refuter.as_ref() == "posterior_predictive"),
        "cheap suite should run posterior PPC on the Bayesian temporal path, got {:?}",
        result.refutations
    );
    assert!(result.refutations.iter().all(|r| r.comparison.is_finite()));
    // Gap: `execute_temporal` never populates `predictive_checks` (see doc comment above).
    assert!(
        result.predictive_checks.is_empty(),
        "if this starts failing, execute_temporal now populates predictive_checks — \
         update this test to assert on it instead of refutations and drop this comment"
    );
    // Prior sensitivity is Full-only; cheap must not add it.
    assert!(
        !result.refutations.iter().any(|r| r.refuter.contains("prior_sensitivity")),
        "cheap suite must not run prior sensitivity, got {:?}",
        result.refutations
    );
}

#[test]
fn temporal_sustained_accepted_cheap() {
    let result = run_staged_sustained(true, RefuteSuite::Cheap).expect("staged estimate_series");
    assert_sustained_mean(&result);
    assert!(result.refutations.iter().any(|r| r.refuter.as_ref() == "prior_predictive"));
    assert!(result.refutations.iter().any(|r| r.refuter.as_ref() == "posterior_predictive"));
    assert!(result.refutations.iter().all(|r| r.comparison.is_finite()));
    assert!(result.predictive_checks.is_empty());
}

/// `RefuteSuite::Full` now completes for Bayesian `TemporalDag` effect queries
/// (this used to hard-fail for *any* such query — the Pulse cell too, not just
/// Sustained — see the fixed root cause below).
///
/// `ValidationSuite::full_effect()` adds `stability_effect()`
/// (`ValidatorId::Bootstrap`, `ValidatorId::DataSubset`,
/// `crates/antecedent-validate/src/suite.rs`). `Bootstrap`'s row-resampling
/// (`with_resampled_rows`) never trips `ensure_unmasked`
/// (`crates/antecedent-data/src/sample.rs`) because it keeps every originally-valid
/// row (no mask is stamped when nothing was already invalid). `DataSubset` used to:
/// it kept a random ~80% of rows via a row-hiding `analysis_mask`
/// (`with_row_subset`), which `ensure_unmasked` unconditionally refuses for temporal
/// designs (lag-gather indexes raw row positions, so a masked-out row would corrupt
/// the lag alignment). `DataSubsetRefuter` now keeps one random *contiguous* window
/// of rows for temporal (non-panel) designs instead
/// (`with_contiguous_row_window`, `crates/antecedent-validate/src/common.rs`):
/// every retained row's immediate predecessor in the window is still its immediate
/// predecessor in the original series, so lag semantics survive, and `refit_effect`
/// rebuilds the series `TimeIndex` at the window's shorter length.
///
/// The `data.subset` refuter reports `passed: true` here — pinned below, not just
/// smoke-tested — now that the fixture carries a real (fixed-seed, reproducible)
/// additive noise term (`SUSTAINED_NOISE_SD` in `sustained_scm_series`) instead of the
/// exact-zero-residual SCM this fixture used to pin. With genuine noise in the outcome
/// equation, the conjugate-Gaussian posterior — and the spread of the ATE across
/// contiguous-window replicates — has a real, non-degenerate scale (on the order of the
/// OLS standard error, ~3.5e-3; see the constant's doc comment for the derivation), so
/// the ~6e-4 shift between the full-sample estimate and the replicate-mean estimate is
/// unremarkable sampling noise, not a many-sigma outlier. A zero/near-zero-residual SCM
/// cannot exercise this refuter meaningfully: any nonzero shift, however tiny, reads as
/// "many replicate-SDs out" against an essentially-zero spread, which is a fixture
/// artifact rather than a finding about the estimator (see the previous revision of
/// this file, which pinned exactly that `passed: false` artifact). This test now
/// verifies the refuter does its job on an estimator it can actually evaluate.
#[test]
fn temporal_sustained_explicit_full_completes_with_data_subset_refuter() {
    let result = run_staged_sustained(false, RefuteSuite::Full)
        .expect("RefuteSuite::Full now completes for Bayesian TemporalDag");
    assert_sustained_mean(&result);
    assert_full_suite_data_subset_refuter_ran(&result);
}

#[test]
fn temporal_sustained_accepted_full_completes_with_data_subset_refuter() {
    let result = run_staged_sustained(true, RefuteSuite::Full)
        .expect("RefuteSuite::Full now completes for Bayesian TemporalDag");
    assert_sustained_mean(&result);
    assert_full_suite_data_subset_refuter_ran(&result);
}

/// Pins the `data.subset` refuter's exact output on the Bayesian `temporal_sustained`
/// fixture (see the doc comment above): pinning the numbers (not just `is_finite()`)
/// pins that the refuter genuinely passes with a healthy, well-away-from-`alpha`
/// p-value on this now-non-degenerate estimator, and not merely "happens to be >=
/// 0.05" by an accident that a future change could silently erode.
fn assert_full_suite_data_subset_refuter_ran(result: &antecedent::StudyResult) {
    let subset = result
        .refutations
        .iter()
        .find(|r| r.refuter.as_ref() == "data.subset")
        .expect("data.subset refuter must have run under RefuteSuite::Full");
    assert!(
        (subset.original_ate - 0.703_919_322_460_684).abs() < 1e-9,
        "unexpected original_ate: {}",
        subset.original_ate
    );
    assert!(
        (subset.refuted_ate - 0.704_541_213_927_508_7).abs() < 1e-6,
        "contiguous-window subset ATE should stay close to the original (lag semantics \
         preserved, no gross corruption; the two differ only by ordinary sampling \
         variation from the additive noise, ~6.2e-4 here); got {}",
        subset.refuted_ate
    );
    assert!(
        (subset.comparison - 0.850_007_214_382_272_2).abs() < 1e-6,
        "expected a large p-value: with real additive noise the replicate spread across \
         contiguous windows is on the order of the estimator's own standard error, so the \
         ~6.2e-4 shift between original and subset ATE is unremarkable; got {}",
        subset.comparison
    );
    assert!(
        subset.passed,
        "data.subset is expected to report passed=true here: once the fixture has a real \
         (non-degenerate) noise term, the posterior/replicate spread is wide enough that a \
         genuinely correct estimator is not flagged as unstable"
    );
    assert_eq!(subset.replicates, 20);
    // The suite as a whole still completes even though one refuter reports a finding.
    assert!(
        result.refutations.iter().any(|r| r.refuter.as_ref() == "bootstrap.ci_coverage"),
        "Full must also run Bootstrap (unaffected by the DataSubset fix)"
    );
}

#[test]
fn prior_bank_catalog() {
    use antecedent_io::{
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
    use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study};
    use antecedent_core::{
        Assumption, AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec,
        RoleHint, SmallRoleSet, ValueType, VariableId,
    };
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
    };
    use antecedent_graph::{Dag, DenseNodeId};
    use antecedent_io::PriorMapping;
    use antecedent_prob::PriorSet;

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
    let result_a = Study::tabular(data_a)
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

    let baseline = Study::tabular(data_b.clone())
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

    let mapped = Study::tabular(data_b.clone())
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
    let auto = Study::tabular(data_b)
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

    use antecedent_prob::{
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
        ess: None,
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
fn prior_bank_ess_accounting() {
    use std::sync::Arc;

    use antecedent_prob::{
        ExternalPriorSource, ExternalPriorWeight, GaussianCoefficientPrior, PriorSet, PriorSpec,
        compose_external_priors,
    };

    fn gauss(mean: f64, var: f64) -> PriorSet {
        let mut p = PriorSet::new();
        p.push(PriorSpec::GaussianCoefficients(
            GaussianCoefficientPrior::shared(1, mean, var).unwrap(),
        ));
        p
    }

    fn assert_opt_vec(actual: &[Option<f64>], expected: &serde_json::Value, tol: f64) {
        let expected = expected.as_array().unwrap();
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected.iter()) {
            match (a, e.as_f64()) {
                (Some(av), Some(ev)) => assert!((av - ev).abs() < tol, "{av} vs {ev}"),
                (None, None) => {}
                other => panic!("mismatch: {other:?}"),
            }
        }
    }

    fn assert_opt(actual: Option<f64>, expected: &serde_json::Value, tol: f64) {
        match (actual, expected.as_f64()) {
            (Some(av), Some(ev)) => assert!((av - ev).abs() < tol, "{av} vs {ev}"),
            (None, None) => {}
            other => panic!("mismatch: {other:?}"),
        }
    }

    let expected = load_expected("prior_bank_ess");
    let tol = expected["tol"].as_f64().unwrap();

    // -- power: single contributing source sums exactly. --
    let p = &expected["power"];
    let baseline =
        gauss(p["baseline_mean"].as_f64().unwrap(), p["baseline_variance"].as_f64().unwrap());
    let sources = [ExternalPriorSource {
        id: Arc::from("old"),
        prior: gauss(p["source_mean"].as_f64().unwrap(), p["source_variance"].as_f64().unwrap()),
        weight: ExternalPriorWeight::power(p["alpha"].as_f64().unwrap()).unwrap(),
        ess: Some(p["ess"].as_f64().unwrap()),
    }];
    let composed = compose_external_priors(&sources, &baseline).unwrap();
    assert_opt_vec(&composed.effective_ess, &p["expected_effective_ess"], tol);
    assert_opt(composed.composed_ess, &p["expected_composed_ess"], tol);
    assert_opt(composed.kish_ess, &p["expected_kish_ess"], tol);

    // -- power_partial_coverage: a contributing source without ess forces
    //    composed_ess to None, even though the other source's effective_ess
    //    is reported. --
    let pc = &expected["power_partial_coverage"];
    let baseline =
        gauss(pc["baseline_mean"].as_f64().unwrap(), pc["baseline_variance"].as_f64().unwrap());
    let sa = &pc["source_a"];
    let sb = &pc["source_b"];
    let sources = [
        ExternalPriorSource {
            id: Arc::from("a"),
            prior: gauss(sa["mean"].as_f64().unwrap(), sa["variance"].as_f64().unwrap()),
            weight: ExternalPriorWeight::power(sa["alpha"].as_f64().unwrap()).unwrap(),
            ess: sa["ess"].as_f64(),
        },
        ExternalPriorSource {
            id: Arc::from("b"),
            prior: gauss(sb["mean"].as_f64().unwrap(), sb["variance"].as_f64().unwrap()),
            weight: ExternalPriorWeight::power(sb["alpha"].as_f64().unwrap()).unwrap(),
            ess: sb["ess"].as_f64(),
        },
    ];
    let composed = compose_external_priors(&sources, &baseline).unwrap();
    assert_opt_vec(&composed.effective_ess, &pc["expected_effective_ess"], tol);
    assert_opt(composed.composed_ess, &pc["expected_composed_ess"], tol);
    assert_opt(composed.kish_ess, &pc["expected_kish_ess"], tol);

    // -- power_dropped_source: a dropped (α=0) source with a declared ess
    //    contributes nothing and cannot poison composed_ess. --
    let pd = &expected["power_dropped_source"];
    let baseline =
        gauss(pd["baseline_mean"].as_f64().unwrap(), pd["baseline_variance"].as_f64().unwrap());
    let sa = &pd["source_a"];
    let sb = &pd["source_b"];
    let sources = [
        ExternalPriorSource {
            id: Arc::from("a"),
            prior: gauss(sa["mean"].as_f64().unwrap(), sa["variance"].as_f64().unwrap()),
            weight: ExternalPriorWeight::power(sa["alpha"].as_f64().unwrap()).unwrap(),
            ess: sa["ess"].as_f64(),
        },
        ExternalPriorSource {
            id: Arc::from("b"),
            prior: gauss(sb["mean"].as_f64().unwrap(), sb["variance"].as_f64().unwrap()),
            weight: ExternalPriorWeight::power(sb["alpha"].as_f64().unwrap()).unwrap(),
            ess: sb["ess"].as_f64(),
        },
    ];
    let composed = compose_external_priors(&sources, &baseline).unwrap();
    assert_opt_vec(&composed.effective_ess, &pd["expected_effective_ess"], tol);
    assert_opt(composed.composed_ess, &pd["expected_composed_ess"], tol);
    assert_opt(composed.kish_ess, &pd["expected_kish_ess"], tol);

    // -- mixture: composed_ess is always None regardless of a declared ess. --
    let m = &expected["mixture"];
    let baseline =
        gauss(m["baseline_mean"].as_f64().unwrap(), m["baseline_variance"].as_f64().unwrap());
    let sources = [ExternalPriorSource {
        id: Arc::from("s"),
        prior: gauss(m["source_mean"].as_f64().unwrap(), m["source_variance"].as_f64().unwrap()),
        weight: ExternalPriorWeight::power_mixture(
            m["alpha"].as_f64().unwrap(),
            m["mixture_weight"].as_f64().unwrap(),
        )
        .unwrap(),
        ess: Some(m["ess"].as_f64().unwrap()),
    }];
    let composed = compose_external_priors(&sources, &baseline).unwrap();
    assert_opt_vec(&composed.effective_ess, &m["expected_effective_ess"], tol);
    assert_opt(composed.composed_ess, &m["expected_composed_ess"], tol);
    assert_opt(composed.kish_ess, &m["expected_kish_ess"], tol);
}

#[test]
fn prior_conjugate_moment_match() {
    use antecedent_prob::{BetaHyperparameters, GammaHyperparameters};

    let expected = load_expected("prior_conjugate_moment_match");
    let tol = expected["tol"].as_f64().unwrap();

    // -- beta_moment_match: from_moments matches (mean, variance) exactly;
    //    the resulting ess is a derived consequence, not a request. --
    let br = &expected["beta_moment_match"];
    let h = BetaHyperparameters::from_moments(
        br["mean"].as_f64().unwrap(),
        br["variance"].as_f64().unwrap(),
    )
    .unwrap();
    assert!((h.alpha - br["expected_alpha"].as_f64().unwrap()).abs() < tol);
    assert!((h.beta - br["expected_beta"].as_f64().unwrap()).abs() < tol);
    assert!((h.mean() - br["expected_mean"].as_f64().unwrap()).abs() < tol);
    assert!((h.variance() - br["expected_variance"].as_f64().unwrap()).abs() < tol);
    assert!((h.ess() - br["expected_ess"].as_f64().unwrap()).abs() < tol);

    // -- beta_moment_match_negative_ess: a proper moment match weaker than
    //    the flat reference reports a negative ess -- truthful, not an
    //    error. --
    let bn = &expected["beta_moment_match_negative_ess"];
    let h = BetaHyperparameters::from_moments(
        bn["mean"].as_f64().unwrap(),
        bn["variance"].as_f64().unwrap(),
    )
    .unwrap();
    assert!(h.alpha > 0.0 && h.beta > 0.0);
    assert!(h.ess() < 0.0);
    assert!((h.mean() - bn["expected_mean"].as_f64().unwrap()).abs() < tol);

    // -- beta_mean_and_ess_zero: from_mean_and_ess(mean, 0.0) degrades to
    //    Beta(1,1)-equivalent strength at the requested mean, never
    //    vanishing or improper. No variance argument exists to satisfy any
    //    support check here. --
    let bz = &expected["beta_mean_and_ess_zero"];
    let h = BetaHyperparameters::from_mean_and_ess(
        bz["mean"].as_f64().unwrap(),
        bz["ess"].as_f64().unwrap(),
    )
    .unwrap();
    assert!((h.alpha - bz["expected_alpha"].as_f64().unwrap()).abs() < tol);
    assert!((h.beta - bz["expected_beta"].as_f64().unwrap()).abs() < tol);
    assert!((h.mean() - bz["expected_mean"].as_f64().unwrap()).abs() < tol);
    assert!((h.ess() - bz["expected_ess"].as_f64().unwrap()).abs() < tol);
    assert!(h.alpha > 0.0 && h.beta > 0.0);

    // -- beta_mean_and_ess_matches_any_request: every (mean, ess >= 0) pair
    //    is satisfiable -- no support gate to violate. --
    let bm = &expected["beta_mean_and_ess_matches_any_request"];
    let h = BetaHyperparameters::from_mean_and_ess(
        bm["mean"].as_f64().unwrap(),
        bm["ess"].as_f64().unwrap(),
    )
    .unwrap();
    assert!((h.alpha - bm["expected_alpha"].as_f64().unwrap()).abs() < tol);
    assert!((h.beta - bm["expected_beta"].as_f64().unwrap()).abs() < tol);
    assert!((h.ess() - bm["expected_ess"].as_f64().unwrap()).abs() < tol);

    // -- beta_moment_match_rejected_inputs: out-of-support (mean, variance)
    //    and out-of-range mean are rejected, never silently clamped. --
    for row in expected["beta_moment_match_rejected_inputs"].as_array().unwrap() {
        let err = BetaHyperparameters::from_moments(
            row["mean"].as_f64().unwrap(),
            row["variance"].as_f64().unwrap(),
        );
        assert!(err.is_err(), "expected rejection: {}", row["reason"]);
    }

    // -- beta_mean_and_ess_rejected_inputs: out-of-range mean and negative
    //    ess are rejected. --
    for row in expected["beta_mean_and_ess_rejected_inputs"].as_array().unwrap() {
        let err = BetaHyperparameters::from_mean_and_ess(
            row["mean"].as_f64().unwrap(),
            row["ess"].as_f64().unwrap(),
        );
        assert!(err.is_err(), "expected rejection: {}", row["reason"]);
    }

    // -- gamma_moment_match: same exact-match contract for Gamma. --
    let gr = &expected["gamma_moment_match"];
    let h = GammaHyperparameters::from_moments(
        gr["mean"].as_f64().unwrap(),
        gr["variance"].as_f64().unwrap(),
    )
    .unwrap();
    assert!((h.shape - gr["expected_shape"].as_f64().unwrap()).abs() < tol);
    assert!((h.rate - gr["expected_rate"].as_f64().unwrap()).abs() < tol);
    assert!((h.mean() - gr["expected_mean"].as_f64().unwrap()).abs() < tol);
    assert!((h.variance() - gr["expected_variance"].as_f64().unwrap()).abs() < tol);
    assert!((h.ess() - gr["expected_ess"].as_f64().unwrap()).abs() < tol);

    // -- gamma_moment_match_negative_ess: shape < 1 reports a negative ess,
    //    truthfully, not as an error. --
    let gn = &expected["gamma_moment_match_negative_ess"];
    let h = GammaHyperparameters::from_moments(
        gn["mean"].as_f64().unwrap(),
        gn["variance"].as_f64().unwrap(),
    )
    .unwrap();
    assert!(h.shape > 0.0 && h.rate > 0.0);
    assert!(h.ess() < 0.0);
    assert!((h.mean() - gn["expected_mean"].as_f64().unwrap()).abs() < tol);

    // -- gamma_mean_and_ess_zero: degrades to Gamma(shape=1, .), the
    //    reference exponential prior, at the requested mean. --
    let gz = &expected["gamma_mean_and_ess_zero"];
    let h = GammaHyperparameters::from_mean_and_ess(
        gz["mean"].as_f64().unwrap(),
        gz["ess"].as_f64().unwrap(),
    )
    .unwrap();
    assert!((h.shape - gz["expected_shape"].as_f64().unwrap()).abs() < tol);
    assert!((h.rate - gz["expected_rate"].as_f64().unwrap()).abs() < tol);
    assert!((h.mean() - gz["expected_mean"].as_f64().unwrap()).abs() < tol);
    assert!((h.ess() - gz["expected_ess"].as_f64().unwrap()).abs() < tol);
    assert!(h.shape > 0.0 && h.rate > 0.0);

    // -- gamma_mean_and_ess_matches_any_request: every (mean, ess >= 0)
    //    pair is satisfiable for Gamma too. --
    let gm = &expected["gamma_mean_and_ess_matches_any_request"];
    let h = GammaHyperparameters::from_mean_and_ess(
        gm["mean"].as_f64().unwrap(),
        gm["ess"].as_f64().unwrap(),
    )
    .unwrap();
    assert!((h.shape - gm["expected_shape"].as_f64().unwrap()).abs() < tol);
    assert!((h.rate - gm["expected_rate"].as_f64().unwrap()).abs() < tol);
    assert!((h.ess() - gm["expected_ess"].as_f64().unwrap()).abs() < tol);

    // -- gamma_moment_match_rejected_inputs: non-positive mean/variance are
    //    rejected. --
    for row in expected["gamma_moment_match_rejected_inputs"].as_array().unwrap() {
        let err = GammaHyperparameters::from_moments(
            row["mean"].as_f64().unwrap(),
            row["variance"].as_f64().unwrap(),
        );
        assert!(err.is_err(), "expected rejection: {}", row["reason"]);
    }

    // -- gamma_mean_and_ess_rejected_inputs: non-positive mean and negative
    //    ess are rejected. --
    for row in expected["gamma_mean_and_ess_rejected_inputs"].as_array().unwrap() {
        let err = GammaHyperparameters::from_mean_and_ess(
            row["mean"].as_f64().unwrap(),
            row["ess"].as_f64().unwrap(),
        );
        assert!(err.is_err(), "expected rejection: {}", row["reason"]);
    }
}

#[test]
fn prior_bank_conflict_shrink() {
    use std::sync::Arc;

    use antecedent_prob::{
        ExternalPriorSource, ExternalPriorWeight, GaussianCoefficientPrior, PriorSet, PriorSpec,
    };
    use antecedent_validate::{ConflictPolicy, ConflictSignals, apply_conflict_and_compose};

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
        ess: None,
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

    use antecedent_prob::{
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
        ess: None,
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

    use antecedent_core::{
        Assumption, CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet,
    };
    use antecedent_prob::{
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
        ess: None,
    }]);
    let baseline = PriorSet::weakly_informative(ncols);
    let composed = compose_external_priors(&sources, &baseline).unwrap();

    let result = Study::tabular(data)
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
    use antecedent_core::{Lag, TemporalEffectQuery, TemporalPolicy};
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
        TimeSeriesData, ValidityBitmap,
    };
    use antecedent_estimate::{BayesianTemporalGcomp, TemporalLinearAdjustment};
    use antecedent_identify::TemporalBackdoorIdentifier;
    use antecedent_prob::{
        ExternalPriorSource, ExternalPriorWeight, GaussianCoefficientPrior, PriorSet, PriorSpec,
        compose_external_priors,
    };
    use antecedent_validate::{ConflictPolicy, ExternalAlphaSensitivity, PriorSensitivity};

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
    temporal_est.inner.overlap = antecedent_estimate::OverlapPolicy::ExplicitOverride;
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
        ess: None,
    }]);
    let baseline = PriorSet::weakly_informative(ncols);
    let composed = compose_external_priors(&sources, &baseline).unwrap();
    let policy = ConflictPolicy::try_new(0.05, 1.0).unwrap();

    // Facade path: conflict re-shrink attaches (Full refute is separately limited on
    // temporal by DataSubset masks; α-grid is exercised directly below).
    let result = Study::series(series)
        .graph(g)
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
