//! 1.9 coverage of licensed temporal / mixture intervals.
//!
//! Every coverage test scores the interval the result actually reports over
//! `n_sim()` datasets (default 400) with a two-sided `level ± 3·MCSE` band
//! ([`CoverageTally`]). Multi-atom tests run on the heterogeneous fixtures in
//! `common::fixtures`, whose identified atoms disagree, so between-atom
//! covariance and weighting are exercised.
//!
//! Calibration targets:
//!
//! * Frequentist frozen-weight mixtures publish an interval for the reported
//!   aggregate, `Σ_g w_g θ_g / Σ_g w_g` over identified atoms (`θ_g` is the
//!   probability limit of atom `g`'s estimator, see `common::fixtures`).
//! * Bayesian class-prior / DBN mixtures publish the draw-level BMA posterior
//!   `P(τ | identified)`: a distribution over graph-specific effects, not an
//!   interval for the weighted mean (whose coverage is ~100% by construction
//!   once atoms separate). Its credible interval is scored against `θ_G` with
//!   `G` drawn from the identified-renormalized weights each replicate, which
//!   is the frequency with which the interval contains the effect of the
//!   structure it averages over. Coverage of the weighted mean is printed as
//!   an `info` line for the record.
//!
//! Ignored tests run via `scripts/gate_calibration.sh`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::too_many_lines,
    clippy::many_single_char_names
)]

mod common;

use std::sync::Arc;

use antecedent::discovery::GraphPosterior;
use antecedent::validate::PredictiveCheckKind;
use antecedent::{
    BayesianConfig, CausalPosterior, ClassPrior, InferenceMode, RefuteSuite, StructuralWeightBasis,
    Study, StudyResult,
};
use antecedent_core::{
    CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, Lag, MediationContrast,
    MediationQuery, ResponseFunctional, ResponseQuery, ResponseUncertainty, ResponseValue,
    TemporalEffectQuery, TemporalPolicy, TemporalResponseSpec, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_graph::{TemporalDag, ensure_lagged};
use antecedent_identify::IdentificationStatus;
use common::calibration::{
    CoverageTally, Z90, gaussian, n_sim, normal_interval, quantile_interval, unit_uniform,
};
use common::fixtures::{
    self, B1, B2, DBN_IDENTIFIED_MASS, DBN_UNIDENTIFIED_MASS, DBN_WEIGHTS, PAG_UNIDENTIFIED_MASS,
    circle_pag, confounded_cpdag, confounded_series, heterogeneous_dbn, mediation_cpdag_two,
    mediation_series, pag_series, two_lag_dag, two_lag_series,
};

const LEVEL: f64 = 0.9;
const N: usize = 160;
/// Outer circular-block replicates per fit. A bootstrap SE from `B` replicates
/// has relative SD ≈ `1/sqrt(2(B-1))`; at 199 that costs < 0.5 pp of coverage.
const BOOT: u32 = 199;
const DRAWS: usize = 1000;
/// Effect in the single-lag `noisy_xy` DGP.
const XY_TRUTH: f64 = 0.8;

// ---------------------------------------------------------------------------
// Data, queries, graphs
// ---------------------------------------------------------------------------

fn noisy_xy(n: usize, seed: u64) -> TimeSeriesData {
    let mut gauss = gaussian(seed);
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; n];
    for t in 1..n {
        x[t] = 0.4 * gauss();
        y[t] = XY_TRUTH * x[t - 1] + 0.35 * gauss();
    }
    TimeSeriesData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())], 1).unwrap()
}

fn pulse_query() -> TemporalEffectQuery {
    TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1)
}

fn single_sustained() -> TemporalEffectQuery {
    TemporalEffectQuery::sustained(VariableId::from_raw(0), VariableId::from_raw(1), -1, 1.0)
        .with_policy(TemporalPolicy::sustained(-1, -1))
        .with_horizon_steps(1)
}

fn multi_sustained() -> TemporalEffectQuery {
    TemporalEffectQuery::sustained(VariableId::from_raw(0), VariableId::from_raw(1), -2, 1.0)
        .with_policy(TemporalPolicy::sustained(-2, -1))
        .with_horizon_steps(1)
}

fn mediated_query() -> CausalQuery {
    CausalQuery::Mediation(
        MediationQuery::binary(
            VariableId::from_raw(0),
            VariableId::from_raw(2),
            [VariableId::from_raw(1)],
            MediationContrast::Mediated,
        )
        .with_horizons(vec![1])
        .unwrap(),
    )
}

fn xy_dag() -> TemporalDag {
    let mut g = TemporalDag::empty();
    let x1 = ensure_lagged(&mut g, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut g, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_directed(x1, y0).unwrap();
    g
}

fn bayes() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(DRAWS).prior_scale(8.0))
}

// ---------------------------------------------------------------------------
// Runners
// ---------------------------------------------------------------------------

fn run_dbn(
    data: TimeSeriesData,
    query: TemporalEffectQuery,
    gp: GraphPosterior,
    inference: InferenceMode,
    boot: u32,
    ctx_seed: u64,
) -> StudyResult {
    Study::series(data)
        .graph_posterior(gp)
        .temporal_query(query)
        .inference(inference)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(boot)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(ctx_seed))
        .unwrap()
}

fn run_freq_dbn(
    data: TimeSeriesData,
    query: TemporalEffectQuery,
    gp: GraphPosterior,
    boot: u32,
    ctx_seed: u64,
) -> StudyResult {
    run_dbn(data, query, gp, InferenceMode::Frequentist, boot, ctx_seed)
}

fn run_freq_class(
    data: TimeSeriesData,
    graph: impl Into<antecedent::AcceptedGraph>,
    query: CausalQuery,
    boot: u32,
    ctx_seed: u64,
) -> StudyResult {
    Study::series(data)
        .graph(graph.into())
        .query(query)
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(boot)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(ctx_seed))
        .unwrap()
}

fn run_bayes(
    data: TimeSeriesData,
    graph: impl Into<antecedent::AcceptedGraph>,
    query: CausalQuery,
    prior: Option<ClassPrior>,
    suite: RefuteSuite,
    ctx_seed: u64,
) -> StudyResult {
    let mut builder = Study::series(data).graph(graph.into()).query(query).inference(bayes());
    if let Some(prior) = prior {
        builder = builder.class_prior(prior);
    }
    builder
        .refute(suite)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(ctx_seed))
        .unwrap()
}

// ---------------------------------------------------------------------------
// Reported intervals
// ---------------------------------------------------------------------------

/// The Frequentist interval a temporal mixture reports: `ate ± z·se_bootstrap`.
fn freq_interval(result: &StudyResult) -> Option<(f64, f64)> {
    normal_interval(result.estimate.ate, result.estimate.se_bootstrap, Z90)
}

fn posterior_interval(post: Option<&CausalPosterior>) -> Option<(f64, f64)> {
    let post = post?;
    let col = post.effect_column()?;
    let draws = post.draws.column(col).ok()?;
    quantile_interval(draws, LEVEL)
}

/// Draw a structural atom index from normalized `weights` for replicate `key`.
fn pick_atom(weights: &[f64], key: u64) -> usize {
    let total: f64 = weights.iter().sum();
    let mut u = unit_uniform(key) * total;
    for (i, w) in weights.iter().enumerate() {
        if u < *w {
            return i;
        }
        u -= w;
    }
    weights.len() - 1
}

fn weighted_mean(weights: &[f64], truths: &[f64]) -> f64 {
    weights.iter().zip(truths).map(|(w, t)| w * t).sum::<f64>() / weights.iter().sum::<f64>()
}

/// Score a BMA posterior interval against `θ_G`, `G ~ weights`; also tally the
/// weighted mean for the record.
struct BmaTally {
    name: String,
    structural: CoverageTally,
    mean: CoverageTally,
    weights: Vec<f64>,
    truths: Vec<f64>,
    key: u64,
}

impl BmaTally {
    fn new(name: &str, weights: &[f64], truths: &[f64], key: u64) -> Self {
        assert_eq!(weights.len(), truths.len());
        Self {
            name: name.to_owned(),
            structural: CoverageTally::new(format!("{name} [θ_G, G~w]"), LEVEL),
            mean: CoverageTally::new(format!("{name} [Σwθ, info]"), LEVEL),
            weights: weights.to_vec(),
            truths: truths.to_vec(),
            key,
        }
    }

    fn record(&mut self, s: u32, interval: Option<(f64, f64)>) {
        let g = pick_atom(&self.weights, self.key.wrapping_mul(1_000_003) + u64::from(s));
        self.structural.record(interval, self.truths[g]);
        self.mean.record(interval, weighted_mean(&self.weights, &self.truths));
    }

    fn assert(&self) {
        eprintln!(
            "info {}: coverage of the weighted mean {:.4} = {:.3} (not asserted; see module docs)",
            self.name,
            weighted_mean(&self.weights, &self.truths),
            self.mean.rate()
        );
        self.structural.assert();
    }
}

/// Frequentist coverage tally that also records the Monte-Carlo SD of the
/// point estimate against the mean reported SE, so a failing gate says
/// whether the interval is mis-centred (bias) or mis-scaled (SE).
struct FreqTally {
    name: String,
    tally: CoverageTally,
    estimates: Vec<f64>,
    ses: Vec<f64>,
    truth: f64,
}

impl FreqTally {
    fn new(name: &str, truth: f64) -> Self {
        Self {
            name: name.to_owned(),
            tally: CoverageTally::new(name, LEVEL),
            estimates: Vec::new(),
            ses: Vec::new(),
            truth,
        }
    }

    fn record(&mut self, result: &StudyResult) {
        self.tally.record(freq_interval(result), self.truth);
        if result.estimate.ate.is_finite() {
            self.estimates.push(result.estimate.ate);
        }
        if let Some(se) = result.estimate.se_bootstrap.filter(|se| se.is_finite()) {
            self.ses.push(se);
        }
    }

    fn assert(&self) {
        let n = self.estimates.len().max(2) as f64;
        let mean = self.estimates.iter().sum::<f64>() / n;
        let mc_sd =
            (self.estimates.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
        let mean_se = self.ses.iter().sum::<f64>() / self.ses.len().max(1) as f64;
        eprintln!(
            "info {}: bias={:+.4} mc_sd={mc_sd:.4} mean_se={mean_se:.4} se/sd={:.3}",
            self.name,
            mean - self.truth,
            mean_se / mc_sd
        );
        self.tally.assert();
    }
}

fn diagnostic_field(result: &StudyResult, code: &str, key: &str) -> Option<f64> {
    let diag = result.diagnostics.iter().find(|d| d.code.as_ref() == code)?;
    let prefix = format!("{key}=");
    diag.message
        .split(|c: char| c == ';' || c == ',' || c.is_whitespace())
        .find_map(|part| part.strip_prefix(prefix.as_str()))
        .and_then(|value| value.parse::<f64>().ok())
}

// ---------------------------------------------------------------------------
// Frequentist DBN posterior (shared circular block)
// ---------------------------------------------------------------------------

fn frequentist_dbn_case(
    name: &str,
    query: &TemporalEffectQuery,
    truth: f64,
    rho: f64,
    n: usize,
    seed_base: u64,
) {
    let mut tally = FreqTally::new(name, truth);
    for s in 0..n_sim() {
        let data = confounded_series(n, B2, rho, seed_base + u64::from(s));
        let result = run_freq_dbn(data, query.clone(), heterogeneous_dbn(), BOOT, u64::from(s));
        tally.record(&result);
    }
    tally.assert();
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_dbn_pulse_shared_block_nominal_90_coverage() {
    let truth = fixtures::dbn_mixture_truth(fixtures::dbn_pulse_atom_truths(0.0));
    frequentist_dbn_case("frequentist DBN Pulse", &pulse_query(), truth, 0.0, N, 9_000);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_dbn_sustained_shared_block_nominal_90_coverage() {
    let truth = fixtures::dbn_mixture_truth(fixtures::dbn_pulse_atom_truths(0.0));
    frequentist_dbn_case(
        "frequentist DBN single-step Sustained",
        &single_sustained(),
        truth,
        0.0,
        N,
        10_000,
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_dbn_multistep_sustained_shared_block_nominal_90_coverage() {
    let truth = fixtures::dbn_mixture_truth(fixtures::dbn_multistep_atom_truths(0.0));
    frequentist_dbn_case(
        "frequentist DBN multi-step Sustained",
        &multi_sustained(),
        truth,
        0.0,
        N,
        11_000,
    );
}

/// Single-atom baseline: the same shared-block path with one contributing atom.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_dbn_pulse_single_atom_baseline_nominal_90_coverage() {
    let mut tally = FreqTally::new("frequentist DBN Pulse single-atom baseline", B1);
    for s in 0..n_sim() {
        let data = confounded_series(N, B2, 0.0, 12_000 + u64::from(s));
        let result =
            run_freq_dbn(data, pulse_query(), fixtures::dbn_atom_a_only(), BOOT, u64::from(s));
        tally.record(&result);
    }
    tally.assert();
}

// R-17: serially dependent errors (AR(1) in the exogenous confounder and the
// outcome innovation). The shared circular block is the dependence-honest
// interval these cells license, so these assert nominal coverage.

macro_rules! frequentist_dbn_ar1 {
    ($name:ident, $label:literal, $query:expr, $truths:path, $rho:expr, $n:expr, $seed:expr) => {
        #[test]
        #[ignore = "calibration: run via scripts/gate_calibration.sh"]
        fn $name() {
            let truth = fixtures::dbn_mixture_truth($truths($rho));
            frequentist_dbn_case($label, &$query, truth, $rho, $n, $seed);
        }
    };
}

frequentist_dbn_ar1!(
    frequentist_dbn_pulse_ar1_rho05_n160_nominal_90_coverage,
    "frequentist DBN Pulse AR(1) rho=0.5 n=160",
    pulse_query(),
    fixtures::dbn_pulse_atom_truths,
    0.5,
    160,
    30_000
);
frequentist_dbn_ar1!(
    frequentist_dbn_pulse_ar1_rho09_n400_nominal_90_coverage,
    "frequentist DBN Pulse AR(1) rho=0.9 n=400",
    pulse_query(),
    fixtures::dbn_pulse_atom_truths,
    0.9,
    400,
    31_000
);
frequentist_dbn_ar1!(
    frequentist_dbn_pulse_ar1_rho05_n60_nominal_90_coverage,
    "frequentist DBN Pulse AR(1) rho=0.5 n=60",
    pulse_query(),
    fixtures::dbn_pulse_atom_truths,
    0.5,
    60,
    32_000
);
frequentist_dbn_ar1!(
    frequentist_dbn_multistep_ar1_rho05_n160_nominal_90_coverage,
    "frequentist DBN multi-step Sustained AR(1) rho=0.5 n=160",
    multi_sustained(),
    fixtures::dbn_multistep_atom_truths,
    0.5,
    160,
    33_000
);
frequentist_dbn_ar1!(
    frequentist_dbn_multistep_ar1_rho09_n400_nominal_90_coverage,
    "frequentist DBN multi-step Sustained AR(1) rho=0.9 n=400",
    multi_sustained(),
    fixtures::dbn_multistep_atom_truths,
    0.9,
    400,
    34_000
);
frequentist_dbn_ar1!(
    frequentist_dbn_multistep_ar1_rho05_n60_nominal_90_coverage,
    "frequentist DBN multi-step Sustained AR(1) rho=0.5 n=60",
    multi_sustained(),
    fixtures::dbn_multistep_atom_truths,
    0.5,
    60,
    35_000
);

// ---------------------------------------------------------------------------
// Frequentist temporal class envelopes (shared circular block)
// ---------------------------------------------------------------------------

fn frequentist_cpdag_case(name: &str, query: &TemporalEffectQuery, rho: f64, n: usize, seed: u64) {
    let truth = fixtures::cpdag_completion_truths(rho).iter().sum::<f64>() / 2.0;
    let mut tally = FreqTally::new(name, truth);
    for s in 0..n_sim() {
        let data = confounded_series(n, 0.0, rho, seed + u64::from(s));
        let result = run_freq_class(
            data,
            confounded_cpdag(),
            CausalQuery::TemporalEffect(query.clone()),
            BOOT,
            u64::from(s),
        );
        assert!(
            result
                .diagnostics
                .iter()
                .any(|d| d.code.as_ref() == "estimate.temporal_class.frequentist.shared_block"),
            "class envelope must publish shared-block SE"
        );
        tally.record(&result);
    }
    tally.assert();
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_temporal_cpdag_pulse_envelope_nominal_90_coverage() {
    frequentist_cpdag_case("frequentist TemporalCpdag Pulse", &pulse_query(), 0.0, N, 13_000);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_temporal_cpdag_sustained_envelope_nominal_90_coverage() {
    frequentist_cpdag_case(
        "frequentist TemporalCpdag Sustained",
        &single_sustained(),
        0.0,
        N,
        14_000,
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_temporal_cpdag_pulse_ar1_rho05_n160_nominal_90_coverage() {
    frequentist_cpdag_case(
        "frequentist TemporalCpdag Pulse AR(1) rho=0.5 n=160",
        &pulse_query(),
        0.5,
        160,
        36_000,
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_temporal_cpdag_pulse_ar1_rho09_n400_nominal_90_coverage() {
    frequentist_cpdag_case(
        "frequentist TemporalCpdag Pulse AR(1) rho=0.9 n=400",
        &pulse_query(),
        0.9,
        400,
        37_000,
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_temporal_cpdag_pulse_ar1_rho05_n60_nominal_90_coverage() {
    frequentist_cpdag_case(
        "frequentist TemporalCpdag Pulse AR(1) rho=0.5 n=60",
        &pulse_query(),
        0.5,
        60,
        38_000,
    );
}

/// `circle_pag` has two completions: `t -> z` is not adjustment amenable
/// (unidentified, mass 1/2), `t <-> z` identifies via MAG adjustment on
/// `z[t-1]`. The reported aggregate is that single identified atom.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_temporal_pag_pulse_envelope_nominal_90_coverage() {
    let truth = pag_identified_truth();
    let mut tally = FreqTally::new("frequentist TemporalPag Pulse", truth);
    for s in 0..n_sim() {
        let result = run_freq_class(
            pag_series(N, 15_000 + u64::from(s)),
            circle_pag(),
            CausalQuery::TemporalEffect(pulse_query()),
            BOOT,
            u64::from(s),
        );
        let structural = result.structural_response.as_ref().expect("class mixture");
        assert_eq!(structural.unidentified_mass, PAG_UNIDENTIFIED_MASS);
        tally.record(&result);
    }
    tally.assert();
}

// ---------------------------------------------------------------------------
// Bayesian TemporalDag (staged path)
// ---------------------------------------------------------------------------

/// R-18: single-step Pulse through `Study::series(..).build().run`, so the
/// staged prior / dispatch path is what gets calibrated.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn bayesian_temporal_dag_pulse_staged_nominal_90_coverage() {
    let mut tally = CoverageTally::new("Bayesian TemporalDag Pulse (staged)", LEVEL);
    for s in 0..n_sim() {
        let result = run_bayes(
            noisy_xy(N, 16_000 + u64::from(s)),
            xy_dag(),
            CausalQuery::TemporalEffect(pulse_query()),
            None,
            RefuteSuite::None,
            u64::from(s),
        );
        tally.record(posterior_interval(result.posterior.as_ref()), XY_TRUTH);
    }
    tally.assert();
}

/// R-18: both intervened lags carry distinct nonzero effects, so the composed
/// truth `B1 + B2` tests propagation through every intervened time.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn bayesian_temporal_dag_multistep_sustained_nominal_90_coverage() {
    let mut tally = CoverageTally::new("Bayesian TemporalDag multi-step Sustained", LEVEL);
    for s in 0..n_sim() {
        let result = run_bayes(
            two_lag_series(N, 17_000 + u64::from(s)),
            two_lag_dag(),
            CausalQuery::TemporalEffect(multi_sustained()),
            None,
            RefuteSuite::None,
            u64::from(s),
        );
        tally.record(posterior_interval(result.posterior.as_ref()), B1 + B2);
    }
    tally.assert();
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn bayesian_temporal_dag_response_curve_pointwise_band_coverage() {
    let query = CausalQuery::Response(
        ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: VariableId::from_raw(1),
            treatment: ContinuousDomain::new(
                VariableId::from_raw(0),
                GridSpec::Values(Arc::from([0.0, 1.0])),
            ),
        })
        .with_temporal(
            TemporalResponseSpec::new(vec![1u32], TemporalPolicy::pulse(-1), None).unwrap(),
        ),
    );
    // The temporal response publishes a pointwise band at its own level (0.95
    // today); calibrate at the level the result reports, not at 90%.
    let mut tally: Option<CoverageTally> = None;
    for s in 0..n_sim() {
        let result = run_bayes(
            noisy_xy(N, 18_000 + u64::from(s)),
            xy_dag(),
            query.clone(),
            None,
            RefuteSuite::None,
            u64::from(s),
        );
        let response = result.response.as_ref().expect("curve");
        let ResponseUncertainty::PointwiseBand { level, lower, upper } = &response.uncertainty
        else {
            panic!("temporal MeanCurve must publish a pointwise band");
        };
        let tally = tally.get_or_insert_with(|| {
            CoverageTally::new("Bayesian TemporalDag ResponseCurve at x=1", *level)
        });
        // The DGP has no intercept, so the mean curve at x = 1 is the effect.
        let interval = (lower.len() >= 2 && upper.len() >= 2).then(|| (lower[1], upper[1]));
        tally.record(interval, XY_TRUTH);
    }
    tally.expect("replicates").assert();
}

// ---------------------------------------------------------------------------
// Bayesian class-prior / DBN mixtures (BMA posterior, θ_G target)
// ---------------------------------------------------------------------------

fn bayesian_cpdag_class_prior_case(
    name: &str,
    query: &TemporalEffectQuery,
    masses: [f64; 2],
    seed: u64,
) {
    let prior = ClassPrior::from_ordered(masses).unwrap();
    let mut tally = BmaTally::new(name, &masses, &fixtures::cpdag_completion_truths(0.0), seed);
    for s in 0..n_sim() {
        let result = run_bayes(
            confounded_series(N, 0.0, 0.0, seed + u64::from(s)),
            confounded_cpdag(),
            CausalQuery::TemporalEffect(query.clone()),
            Some(prior.clone()),
            RefuteSuite::None,
            u64::from(s),
        );
        let structural = result.structural_response.as_ref().expect("class mixture");
        assert_eq!(structural.weight_basis, StructuralWeightBasis::CallerSuppliedClassPrior);
        assert_eq!(structural.unidentified_mass, 0.0, "both completions identify");
        tally.record(s, posterior_interval(result.posterior.as_ref()));
    }
    tally.assert();
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn bayesian_temporal_cpdag_pulse_class_prior_nominal_90_coverage() {
    bayesian_cpdag_class_prior_case(
        "Bayesian TemporalCpdag Pulse class-prior",
        &pulse_query(),
        [0.5, 0.5],
        19_000,
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn bayesian_temporal_cpdag_sustained_class_prior_nominal_90_coverage() {
    bayesian_cpdag_class_prior_case(
        "Bayesian TemporalCpdag Sustained class-prior",
        &single_sustained(),
        [0.5, 0.5],
        20_000,
    );
}

/// Unequal masses: the mixture must weight atoms by the caller's class prior.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn class_prior_mixture_functional_nominal_90_coverage() {
    bayesian_cpdag_class_prior_case(
        "class-prior mixture functional [0.3, 0.7]",
        &pulse_query(),
        [0.3, 0.7],
        22_000,
    );
}

/// Class prior `[0.4, 0.6]` over `circle_pag`'s completions: the BMA over the
/// identified completion is that atom's posterior, and the unidentified
/// completion's 0.4 must stay a separate axis.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn bayesian_temporal_pag_pulse_nominal_90_coverage() {
    let prior = ClassPrior::from_ordered([0.4, 0.6]).unwrap();
    let truth = pag_identified_truth();
    let mut tally = CoverageTally::new("Bayesian TemporalPag Pulse class-prior", LEVEL);
    for s in 0..n_sim() {
        let result = run_bayes(
            pag_series(N, 23_000 + u64::from(s)),
            circle_pag(),
            CausalQuery::TemporalEffect(pulse_query()),
            Some(prior.clone()),
            RefuteSuite::None,
            u64::from(s),
        );
        let post = result.posterior.as_ref().expect("PAG class-prior posterior");
        assert!((post.unidentified_mass - 0.4).abs() < 1e-12, "got {}", post.unidentified_mass);
        tally.record(posterior_interval(Some(post)), truth);
    }
    tally.assert();
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn bayesian_dbn_posterior_pulse_nominal_90_coverage() {
    let truths = fixtures::dbn_pulse_atom_truths(0.0);
    let mut tally =
        BmaTally::new("Bayesian DBN posterior Pulse", &DBN_WEIGHTS[..2], &truths, 24_000);
    for s in 0..n_sim() {
        let result = run_dbn(
            confounded_series(N, B2, 0.0, 24_000 + u64::from(s)),
            pulse_query(),
            heterogeneous_dbn(),
            bayes(),
            0,
            u64::from(s),
        );
        let post = result.posterior.as_ref().expect("DBN BMA posterior");
        assert!(
            (post.unidentified_mass - DBN_UNIDENTIFIED_MASS).abs() < 1e-12,
            "unidentified DBN mass must stay a separate axis, got {}",
            post.unidentified_mass
        );
        tally.record(s, posterior_interval(Some(post)));
    }
    tally.assert();
}

/// A `TemporalCpdag` mediation result publishes no blended interval: the
/// aggregate estimate is NaN and each completion keeps its own composed
/// posterior (`structural_response.atoms[i].posterior`). What is reported is
/// therefore calibrated per completion, each against its own `θ_g` (equal
/// here; see `fixtures::mediation_cpdag_two`). The old "any completion
/// covers" rule was not a property of any reported interval.
fn bayesian_cpdag_mediation_case(name: &str, kappa: f64, seed: u64) {
    let truths = [fixtures::mediation_truth(); 2];
    let mut tallies: Vec<CoverageTally> = truths
        .iter()
        .enumerate()
        .map(|(i, t)| CoverageTally::new(format!("{name} completion {i} (θ={t:.3})"), LEVEL))
        .collect();
    let mut mean = [0.0; 2];
    for s in 0..n_sim() {
        let result = run_bayes(
            mediation_series(N, kappa, seed + u64::from(s)),
            mediation_cpdag_two(),
            mediated_query(),
            None,
            RefuteSuite::None,
            u64::from(s),
        );
        assert!(result.estimate.ate.is_nan(), "class mediation must not publish a blended effect");
        let atoms = &result.structural_response.as_ref().expect("atoms").atoms;
        assert_eq!(atoms.len(), truths.len());
        for (i, ((atom, truth), tally)) in atoms.iter().zip(&truths).zip(&mut tallies).enumerate() {
            if let Some(ResponseValue::Scalar(v)) = atom.value {
                mean[i] += v / f64::from(n_sim());
            }
            tally.record(posterior_interval(atom.posterior.as_ref()), *truth);
        }
    }
    eprintln!(
        "info {name}: mean completion estimates {mean:?}; identified truth {:.4}; \
         t->y-backdoor-only plim {:.4}",
        fixtures::mediation_truth(),
        fixtures::mediation_backdoor_only_plim(kappa)
    );
    for tally in &tallies {
        tally.assert();
    }
}

/// Mediator-outcome confounding through `z[t-1] -> w[t-1] -> y` that does not
/// confound `t -> y`: both completions identify the path product.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn bayesian_temporal_cpdag_mediation_envelope_nominal_90_coverage() {
    bayesian_cpdag_mediation_case("Bayesian TemporalCpdag mediation", fixtures::MED_KAPPA, 25_000);
}

/// Same class without mediator-outcome confounding (`kappa = 0`): isolates
/// interval calibration from the adjustment-set question.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn bayesian_temporal_cpdag_mediation_unconfounded_nominal_90_coverage() {
    bayesian_cpdag_mediation_case("Bayesian TemporalCpdag mediation (kappa=0)", 0.0, 26_000);
}

/// Bayesian `TemporalDag` mediation on the mediator-outcome-confounded DGP
/// (`kappa = 0.5`): the single-horizon composed posterior carries the Total,
/// Direct and Mediated draw columns; each 90% equal-tail interval is scored
/// against its path-product truth.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn bayesian_temporal_dag_mediation_confounded_nominal_90_coverage() {
    let targets = [
        ("Total", 1, fixtures::mediation_total_truth()),
        ("Direct", 2, fixtures::mediation_direct_truth()),
        ("Mediated", 3, fixtures::mediation_truth()),
    ];
    let mut tallies: Vec<CoverageTally> = targets
        .iter()
        .map(|(name, _, truth)| {
            CoverageTally::new(
                format!("Bayesian TemporalDag mediation {name} [kappa=0.5] (θ={truth:.3})"),
                LEVEL,
            )
        })
        .collect();
    for s in 0..n_sim() {
        let result = run_confounded_mediation(
            N,
            MediationContrast::Mediated,
            bayes(),
            0,
            27_000 + u64::from(s),
            u64::from(s),
        );
        let post = result.posterior.as_ref().expect("single-horizon mediation posterior");
        for ((_, column, truth), tally) in targets.iter().zip(&mut tallies) {
            let draws = post.draws.column(*column).expect("decomposition draws");
            tally.record(quantile_interval(draws, LEVEL), *truth);
        }
    }
    // Print every contrast's calibration line before failing on any of them.
    let failures: Vec<String> = tallies
        .iter()
        .filter_map(|tally| {
            std::panic::catch_unwind(|| tally.assert()).err().map(|e| {
                e.downcast_ref::<String>().cloned().unwrap_or_else(|| "coverage failure".into())
            })
        })
        .collect();
    assert!(failures.is_empty(), "{}", failures.join("; "));
}

// ---------------------------------------------------------------------------
// Fixture truths that depend on enumeration order (pinned below)
// ---------------------------------------------------------------------------

fn pag_identified_truth() -> f64 {
    fixtures::pag_completion_truths().into_iter().flatten().next().expect("identified")
}

// ---------------------------------------------------------------------------
// Non-ignored fixture sanity (R-15 / R-16)
// ---------------------------------------------------------------------------

const SANITY_N: usize = 20_000;

fn close(label: &str, got: f64, want: f64, tol: f64) {
    assert!((got - want).abs() <= tol, "{label}: got {got}, want {want} ± {tol}");
}

fn atom_values(result: &StudyResult) -> Vec<Option<f64>> {
    result
        .structural_response
        .as_ref()
        .expect("structural atoms")
        .atoms
        .iter()
        .map(|atom| match atom.value {
            Some(ResponseValue::Scalar(v)) => Some(v),
            _ => None,
        })
        .collect()
}

#[test]
fn heterogeneous_dbn_fixture_has_disagreeing_atoms_and_unidentified_mass() {
    for rho in [0.0, 0.5] {
        let data = confounded_series(SANITY_N, B2, rho, 1);
        let [ta, tb] = fixtures::dbn_pulse_atom_truths(rho);
        let a = run_freq_dbn(data.clone(), pulse_query(), fixtures::dbn_atom_a_only(), 0, 41);
        let b = run_freq_dbn(data.clone(), pulse_query(), fixtures::dbn_atom_b_only(), 0, 41);
        close("DBN atom A Pulse plim", a.estimate.ate, ta, 0.03);
        close("DBN atom B Pulse plim", b.estimate.ate, tb, 0.03);
        assert!(tb - ta > 0.4, "DBN atoms must disagree materially");
        let mix = run_freq_dbn(data.clone(), pulse_query(), heterogeneous_dbn(), 0, 41);
        close(
            "DBN mixture = frozen-weight mean of atom fits",
            mix.estimate.ate,
            (DBN_WEIGHTS[0] * a.estimate.ate + DBN_WEIGHTS[1] * b.estimate.ate)
                / DBN_IDENTIFIED_MASS,
            1e-10,
        );
        let [ma, mb] = fixtures::dbn_multistep_atom_truths(rho);
        let a = run_freq_dbn(data.clone(), multi_sustained(), fixtures::dbn_atom_a_only(), 0, 41);
        let b = run_freq_dbn(data, multi_sustained(), fixtures::dbn_atom_b_only(), 0, 41);
        close("DBN atom A multi-step plim", a.estimate.ate, ma, 0.04);
        close("DBN atom B multi-step plim", b.estimate.ate, mb, 0.04);
        assert!(mb - ma > 0.3, "DBN multi-step atoms must disagree materially");
    }
}

/// R-16: the Frequentist DBN path exposes its mass split on the
/// `estimate.dbn_posterior.frequentist` diagnostic and as `GraphDependent`
/// status; the Bayesian path on `posterior.unidentified_mass`. Both must equal
/// the fixture's known unidentified weight.
#[test]
fn dbn_mixture_functional_retains_unidentified_mass() {
    let data = confounded_series(N, B2, 0.0, 3);
    let freq = run_freq_dbn(data.clone(), pulse_query(), heterogeneous_dbn(), 8, 41);
    assert_eq!(freq.identification.status, IdentificationStatus::GraphDependent);
    let code = "estimate.dbn_posterior.frequentist";
    let unidentified = diagnostic_field(&freq, code, "unidentified_mass").expect("mass field");
    let identified = diagnostic_field(&freq, code, "identified_mass").expect("mass field");
    close("Frequentist unidentified mass", unidentified, DBN_UNIDENTIFIED_MASS, 1e-12);
    close("Frequentist identified mass", identified, DBN_IDENTIFIED_MASS, 1e-12);
    let bayes = run_dbn(data, pulse_query(), heterogeneous_dbn(), bayes(), 0, 41);
    let post = bayes.posterior.as_ref().expect("DBN BMA posterior");
    close("Bayesian unidentified mass", post.unidentified_mass, DBN_UNIDENTIFIED_MASS, 1e-12);
}

/// The shared circular-block SE on frozen weights must carry the cross-atom
/// covariance. With every atom refit on the same resample, the mixture SE lies
/// strictly between the independent-atom combination `sqrt(Σ (w̄_g se_g)²)`
/// (what bootstrapping atoms separately would give) and the comonotone bound
/// `Σ w̄_g se_g` (Cauchy–Schwarz: a frozen-weight mean can never exceed its
/// largest atom SE, so "exceeds each atom's own SE" is read as exceeding each
/// atom's weighted within-atom contribution and their independent combination).
///
/// The DBN atoms' estimates are only mildly correlated on this fixture
/// (sampling correlation ≈ 0.17 between the adjusted and unadjusted slopes), so
/// the shared SE sits a few percent above the independent combination. The
/// 1.1× margin the CPDAG pin uses was only reachable while the raw-row resample
/// inflated every mixture SE (1.9, F1); the strict inequality is what the
/// positive cross-atom covariance implies.
#[test]
fn heterogeneous_dbn_shared_block_se_carries_cross_atom_covariance() {
    let data = confounded_series(N, B2, 0.0, 5);
    let boot = 99;
    let mix = run_freq_dbn(data.clone(), pulse_query(), heterogeneous_dbn(), boot, 41);
    let a = run_freq_dbn(data.clone(), pulse_query(), fixtures::dbn_atom_a_only(), boot, 41);
    let b = run_freq_dbn(data, pulse_query(), fixtures::dbn_atom_b_only(), boot, 41);
    let se = mix.estimate.se_bootstrap.expect("shared-block SE");
    let se_a = a.estimate.se_bootstrap.expect("atom A SE");
    let se_b = b.estimate.se_bootstrap.expect("atom B SE");
    let (wa, wb) = (DBN_WEIGHTS[0] / DBN_IDENTIFIED_MASS, DBN_WEIGHTS[1] / DBN_IDENTIFIED_MASS);
    let independent = ((wa * se_a).powi(2) + (wb * se_b).powi(2)).sqrt();
    eprintln!("DBN shared SE {se:.5}; atoms {se_a:.5}, {se_b:.5}; independent {independent:.5}");
    assert!(se > wa * se_a && se > wb * se_b);
    assert!(se > independent, "shared SE {se} must exceed independent {independent}");
    assert!(se <= wa * se_a + wb * se_b + 1e-12, "Cauchy–Schwarz bound violated");
}

#[test]
fn heterogeneous_cpdag_fixture_has_disagreeing_completions() {
    for rho in [0.0, 0.5] {
        let data = confounded_series(SANITY_N, 0.0, rho, 7);
        let result = run_freq_class(
            data,
            confounded_cpdag(),
            CausalQuery::TemporalEffect(pulse_query()),
            0,
            41,
        );
        let values = atom_values(&result);
        let truths = fixtures::cpdag_completion_truths(rho);
        assert_eq!(values.len(), 2, "two completions");
        for (value, truth) in values.iter().zip(truths) {
            close("CPDAG completion plim", value.expect("identified"), truth, 0.03);
        }
        assert!(truths[0] - truths[1] > 0.4);
        let structural = result.structural_response.as_ref().unwrap();
        assert_eq!(structural.unidentified_mass, 0.0);
        close(
            "CPDAG mixture",
            result.estimate.ate,
            (values[0].unwrap() + values[1].unwrap()) / 2.0,
            1e-10,
        );
    }
}

#[test]
fn heterogeneous_cpdag_shared_block_se_carries_cross_atom_covariance() {
    let data = confounded_series(N, 0.0, 0.0, 9);
    let boot = 99;
    let query = CausalQuery::TemporalEffect(pulse_query());
    let mix = run_freq_class(data.clone(), confounded_cpdag(), query.clone(), boot, 41);
    let a = run_freq_class(
        data.clone(),
        fixtures::confounded_cpdag_oriented(false),
        query.clone(),
        boot,
        41,
    );
    let b = run_freq_class(data, fixtures::confounded_cpdag_oriented(true), query, boot, 41);
    let values = atom_values(&mix);
    close("oriented t->z matches completion 0", a.estimate.ate, values[0].unwrap(), 1e-10);
    close("oriented z->t matches completion 1", b.estimate.ate, values[1].unwrap(), 1e-10);
    let se = mix.estimate.se_bootstrap.expect("shared-block SE");
    let se_a = a.estimate.se_bootstrap.expect("completion 0 SE");
    let se_b = b.estimate.se_bootstrap.expect("completion 1 SE");
    let independent = ((0.5 * se_a).powi(2) + (0.5 * se_b).powi(2)).sqrt();
    eprintln!("CPDAG shared SE {se:.5}; atoms {se_a:.5}, {se_b:.5}; independent {independent:.5}");
    assert!(se > 0.5 * se_a && se > 0.5 * se_b);
    assert!(se > 1.1 * independent, "shared SE {se} must exceed independent {independent}");
    assert!(se <= 0.5 * se_a + 0.5 * se_b + 1e-12, "Cauchy–Schwarz bound violated");
}

/// `circle_pag`: two completions from one circle mark, one identified through
/// MAG adjustment and one not amenable. Pins enumeration order, the identified
/// atom's plim, and the retained unidentified mass. (Two identified,
/// disagreeing PAG completions were not constructible; see the fixture docs.)
#[test]
fn circle_pag_fixture_identifies_one_completion_and_retains_mass() {
    let data = pag_series(SANITY_N, 11);
    let result =
        run_freq_class(data, circle_pag(), CausalQuery::TemporalEffect(pulse_query()), 0, 41);
    let values = atom_values(&result);
    let truths = fixtures::pag_completion_truths();
    assert_eq!(values.len(), truths.len(), "two PAG completions");
    for (value, truth) in values.iter().zip(&truths) {
        match truth {
            Some(truth) => close("PAG completion plim", value.expect("identified"), *truth, 0.03),
            None => assert!(value.is_none(), "completion must stay unidentified"),
        }
    }
    let structural = result.structural_response.as_ref().unwrap();
    assert_eq!(structural.unidentified_mass, PAG_UNIDENTIFIED_MASS);
    assert_eq!(result.identification.status, IdentificationStatus::GraphDependent);
}

/// Two completions whose mediated effects necessarily agree (see
/// `fixtures::mediation_cpdag_two`); each keeps its own composed posterior.
/// With or without mediator-outcome confounding the estimate is the path
/// product: the mediation adjustment set blocks `m <- z[t-1] -> w[t-1] -> y`.
#[test]
fn mediation_cpdag_fixture_has_two_completions_with_own_posteriors() {
    for kappa in [0.0, fixtures::MED_KAPPA] {
        let data = mediation_series(SANITY_N, kappa, 13);
        let result =
            run_bayes(data, mediation_cpdag_two(), mediated_query(), None, RefuteSuite::None, 41);
        assert!(result.estimate.ate.is_nan());
        let atoms = &result.structural_response.as_ref().expect("atoms").atoms;
        assert_eq!(atoms.len(), 2, "two completions");
        let values: Vec<f64> = atoms
            .iter()
            .map(|atom| {
                assert!(atom.posterior.is_some(), "each completion keeps its own posterior");
                match atom.value {
                    Some(ResponseValue::Scalar(v)) => v,
                    _ => panic!("completion must be identified"),
                }
            })
            .collect();
        close("completions agree", values[0], values[1], 0.01);
        close(
            &format!("mediation completion plim (kappa={kappa})"),
            values[0],
            fixtures::mediation_truth(),
            0.02,
        );
    }
}

/// Mediation `Study` on `fixtures::mediation_dag` at `kappa != 0`.
fn run_confounded_mediation(
    n: usize,
    contrast: MediationContrast,
    inference: InferenceMode,
    boot: u32,
    data_seed: u64,
    ctx_seed: u64,
) -> StudyResult {
    Study::series(mediation_series(n, fixtures::MED_KAPPA, data_seed))
        .graph(fixtures::mediation_dag())
        .query(CausalQuery::Mediation(
            MediationQuery::binary(
                VariableId::from_raw(0),
                VariableId::from_raw(2),
                [VariableId::from_raw(1)],
                contrast,
            )
            .with_horizons(vec![1])
            .unwrap(),
        ))
        .inference(inference)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(boot)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(ctx_seed))
        .unwrap()
}

/// WP-F2: with the mediator-outcome confounder `w[t-1]` present (`kappa =
/// 0.5`), a plain `TemporalDag` recovers Total, Direct and Mediated in both
/// inference modes, for every requested contrast, and reports the adjustment
/// set it used. Adjusting only the `t -> y` back-door set (empty) gave
/// Mediated ≈ `mediation_backdoor_only_plim(0.5)` ≈ 0.519 against a truth of
/// 0.300.
#[test]
fn confounded_mediation_recovers_total_direct_mediated_in_both_modes() {
    let truths = [
        (MediationContrast::Total, fixtures::mediation_total_truth()),
        (MediationContrast::Direct, fixtures::mediation_direct_truth()),
        (MediationContrast::Mediated, fixtures::mediation_truth()),
    ];
    for inference in [InferenceMode::Frequentist, bayes()] {
        for (contrast, truth) in truths {
            let label = format!("{inference:?} {contrast:?}");
            let result = run_confounded_mediation(SANITY_N, contrast, inference.clone(), 0, 13, 41);
            close(&label, result.estimate.ate, truth, 0.02);
            let slice = &result.mediation_grid.as_ref().expect("mediation grid").slices[0];
            let adjustment: Vec<(u32, i32)> =
                slice.adjustment.iter().map(|k| (k.variable.raw(), k.offset)).collect();
            assert_eq!(adjustment, vec![(3, -1), (4, -1)], "{label}: S(1) = {{z[t-1], w[t-1]}}");
            let point = &slice.estimate;
            close(&format!("{label} total"), point.total.unwrap(), truths[0].1, 0.02);
            close(&format!("{label} direct"), point.direct.unwrap(), truths[1].1, 0.02);
            close(&format!("{label} mediated"), point.mediated.unwrap(), truths[2].1, 0.02);
            assert!(
                result.identification.required_assumptions.entries.iter().any(|a| matches!(
                    &a.assumption,
                    antecedent_core::Assumption::Custom { id, description }
                        if id.as_ref() == "temporal_mediation.adjustment_sets"
                            && description.contains("S = {v3[t-1], v4[t-1]}")
                )),
                "{label}: the adjustment sets must be recorded as an assumption"
            );
        }
    }
    // Same fixture, the pre-fix plim: the gap the gate found.
    assert!(
        (fixtures::mediation_backdoor_only_plim(fixtures::MED_KAPPA) - fixtures::mediation_truth())
            .abs()
            > 0.2
    );
}

#[test]
fn two_lag_fixture_composes_both_lags() {
    let data = two_lag_series(SANITY_N, 17);
    let result =
        run_freq_class(data, two_lag_dag(), CausalQuery::TemporalEffect(multi_sustained()), 0, 41);
    close("two-lag multi-step Sustained", result.estimate.ate, B1 + B2, 0.03);
}

#[test]
fn bayesian_full_pins_numeric_ppc_summary() {
    let result = run_bayes(
        noisy_xy(N, 7),
        xy_dag(),
        CausalQuery::TemporalEffect(pulse_query()),
        None,
        RefuteSuite::Full,
        41,
    );
    assert!(
        result.predictive_checks.iter().any(|c| c.kind == PredictiveCheckKind::Prior),
        "full must run prior PPC"
    );
    let ppc = result
        .predictive_checks
        .iter()
        .find(|c| c.kind == PredictiveCheckKind::Posterior)
        .expect("full must run posterior PPC");
    assert!(result.posterior.as_ref().and_then(|p| p.prior_sensitivity.as_ref()).is_some());
    assert!(ppc.p_value.is_finite());
    assert!(ppc.predictive_mean.is_finite());
    assert!(ppc.predictive_sd.is_finite());
    assert!(ppc.n_sims >= 64, "PPC must run a numeric simulation, n_sims={}", ppc.n_sims);
    assert!(
        (ppc.predictive_mean - ppc.observed).abs() < 0.35,
        "numeric PPC predictive_mean={} observed={}",
        ppc.predictive_mean,
        ppc.observed
    );
    assert!(
        ppc.p_value > 0.01,
        "well-specified conjugate DGP must not reject posterior PPC (p={})",
        ppc.p_value
    );
}
