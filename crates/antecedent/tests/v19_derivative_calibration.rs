//! 1.9 coverage of the licensed derivative family (R-19, R-8).
//!
//! Every in-assumption DGP here is inside the stated cell assumptions: an
//! additive outcome surface, a homoskedastic Gaussian treatment law given the
//! adjustment set, and a fixed caller bandwidth. The point-derivative DGP has
//! genuine third-order curvature at the evaluation point (`m(a) = 5 + 2 sin a`,
//! `m'''(0.5) ≈ -1.76`), so a local-quadratic interval that ignores smoothing
//! bias under-covers there at a mean-squared-error-sized bandwidth. The
//! licensed estimand is the true derivative `m'(a)`, not a bandwidth-smoothed
//! surrogate, so every point-type truth below is the analytic derivative.
//! Before 1.9 that interval covered 0.820 here; it is now robust
//! bias-corrected (local cubic at the caller bandwidth).
//!
//! The order-2 point derivative (`m''(0.5) = -2 sin 0.5`, `m''''(0.5) ≠ 0`) is
//! scored at a wider bandwidth sized for the curvature. There the leading
//! second-derivative bias is `h²·m''''`, which a local cubic does not remove;
//! the interval is bias-corrected by a local quartic instead.
//!
//! The Jacobian / directional DGP is additive with a quadratic component. Only
//! the Bayesian result publishes a band for these (the Frequentist result
//! withholds it, asserted below). A roughness-penalized plug-in target shrank
//! the gradient enough that the band covered well under nominal; the target is
//! now an unpenalized regression spline in the treatment coordinates.
//!
//! The skewed / heteroskedastic treatment-law ADE run is a misspecification
//! probe outside the Gaussian-score assumption: it records coverage and checks
//! that the assumption is stated, but does not gate on nominal coverage.
//!
//! Ignored tests run via `scripts/gate_calibration.sh`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::too_many_lines,
    clippy::many_single_char_names
)]

mod common;

use std::sync::Arc;

use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, CausalResponse, DerivativeScale, DerivativeWeighting, ExecutionContext,
    ResponseFunctional as F, ResponseIdentification, ResponseQuery, ResponseUncertainty,
    ResponseValue, VariableId,
};
use antecedent_data::TabularData;
use antecedent_estimate::ContinuousResponseOptions;
use antecedent_graph::{Dag, DenseNodeId};
use common::calibration::{CoverageTally, coverage_band, gaussian, n_sim};

const LEVEL: f64 = 0.9;
/// Rows per point-derivative / ADE replicate.
const N_POINT: usize = 1000;
/// Rows per multivariate (Jacobian / directional) replicate.
const N_GAM: usize = 1000;
/// Caller bandwidth: close to the MSE-optimal local-quadratic first-derivative
/// bandwidth for this DGP at `N_POINT` (≈ 0.34), i.e. the bandwidth a careful
/// user would pick, and the one the known-truth fixture uses.
const BANDWIDTH: f64 = 0.35;
const AT: f64 = 0.5;
/// Caller bandwidth for the order-2 point derivative: wide enough that the
/// curvature's smoothing bias is comparable to its standard error at `N_POINT`.
const SECOND_ORDER_BANDWIDTH: f64 = 0.7;
const DRAWS: usize = 200;

fn vid(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

/// Well-separated per-replicate seed. `gaussian(seed)` uses `seed | 1`, so
/// consecutive integers `2k`, `2k + 1` would replay the same dataset.
fn replicate_seed(family: u64, rep: u64) -> u64 {
    (family << 32) ^ (rep + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// Structural mean `m(a) = E[Y | do(A = a)] = 5 + 2 sin a` (E[X] = 0).
fn mu(a: f64) -> f64 {
    5.0 + 2.0 * a.sin()
}

fn mu_prime(a: f64) -> f64 {
    2.0 * a.cos()
}

fn mu_second(a: f64) -> f64 {
    -2.0 * a.sin()
}

/// Columns `a`(0), `x`(1), `y`(2): X ~ N(0,1); A = 0.5X + N(0,1);
/// Y = 5 + 2 sin A + X + N(0,1). A | X is homoskedastic Gaussian.
fn point_data(n: usize, seed: u64) -> TabularData {
    let mut z = gaussian(seed);
    let mut a = Vec::with_capacity(n);
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let xi = z();
        let ai = 0.5 * xi + z();
        x.push(xi);
        a.push(ai);
        y.push(mu(ai) + xi + z());
    }
    TabularData::from_f64_columns([("a", a.as_slice()), ("x", x.as_slice()), ("y", y.as_slice())])
        .unwrap()
}

/// Same outcome law, but A = 0.5X + (0.5 + 0.5|X|)(E − 1) with E ~ Exp(1):
/// skewed and heteroskedastic given X, outside the Gaussian treatment score.
fn skewed_treatment_draw(z: &mut impl FnMut() -> f64) -> (f64, f64) {
    let xi = z();
    let (u, v) = (z(), z());
    let e = 0.5 * (u * u + v * v);
    (xi, 0.5 * xi + (0.5 + 0.5 * xi.abs()) * (e - 1.0))
}

fn skewed_data(n: usize, seed: u64) -> TabularData {
    let mut z = gaussian(seed);
    let mut a = Vec::with_capacity(n);
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let (xi, ai) = skewed_treatment_draw(&mut z);
        x.push(xi);
        a.push(ai);
        y.push(mu(ai) + xi + z());
    }
    TabularData::from_f64_columns([("a", a.as_slice()), ("x", x.as_slice()), ("y", y.as_slice())])
        .unwrap()
}

fn point_graph() -> Dag {
    let mut graph = Dag::with_variables(3);
    for (s, t) in [(1, 0), (1, 2), (0, 2)] {
        graph.insert_directed(DenseNodeId::from_raw(s), DenseNodeId::from_raw(t)).unwrap();
    }
    graph
}

/// Columns `a1`(0), `a2`(1), `x`(2), `y1`(3), `y2`(4); X confounds both treatments:
/// Y1 = 1 + 1.5 A1 + 0.5 A1² − 0.5 A2 + X + ε, Y2 = 0.25 A1 + 1.5 A2 + 0.5 X + ε.
fn gam_data(n: usize, seed: u64) -> TabularData {
    let mut z = gaussian(seed);
    let mut cols: [Vec<f64>; 5] = std::array::from_fn(|_| Vec::with_capacity(n));
    for _ in 0..n {
        let x = z();
        let a1 = 0.5 * x + z();
        let a2 = 0.3 * x + z();
        cols[0].push(a1);
        cols[1].push(a2);
        cols[2].push(x);
        cols[3].push(1.0 + 1.5 * a1 + 0.5 * a1 * a1 - 0.5 * a2 + x + z());
        cols[4].push(0.25 * a1 + 1.5 * a2 + 0.5 * x + z());
    }
    TabularData::from_f64_columns([
        ("a1", cols[0].as_slice()),
        ("a2", cols[1].as_slice()),
        ("x", cols[2].as_slice()),
        ("y1", cols[3].as_slice()),
        ("y2", cols[4].as_slice()),
    ])
    .unwrap()
}

fn gam_graph() -> Dag {
    let mut graph = Dag::with_variables(5);
    for (s, t) in [(2, 0), (2, 1), (2, 3), (2, 4), (0, 3), (0, 4), (1, 3), (1, 4)] {
        graph.insert_directed(DenseNodeId::from_raw(s), DenseNodeId::from_raw(t)).unwrap();
    }
    graph
}

const GAM_AT: [f64; 2] = [0.5, 0.0];
/// Row-major outcomes × treatments at `GAM_AT`.
const JACOBIAN_TRUTH: [f64; 4] = [2.0, -0.5, 0.25, 1.5];
const DIRECTION: [f64; 2] = [1.0, 2.0];
const DIRECTIONAL_TRUTH: [f64; 2] = [1.0, 3.25];

fn point_query(scale: DerivativeScale) -> F {
    F::PointDerivative { outcome: vid(2), treatment: vid(0), at: AT, order: 1, scale }
}

fn ade_query() -> F {
    F::AverageDerivative {
        outcome: vid(2),
        treatment: vid(0),
        weighting: DerivativeWeighting::Observed,
    }
}

fn jacobian_query() -> F {
    F::Jacobian {
        outcomes: Arc::from([vid(3), vid(4)]),
        treatments: Arc::from([vid(0), vid(1)]),
        at: Arc::from(GAM_AT),
        scale: DerivativeScale::Identity,
    }
}

fn directional_query() -> F {
    F::DirectionalDerivative {
        outcomes: Arc::from([vid(3), vid(4)]),
        treatments: Arc::from([vid(0), vid(1)]),
        at: Arc::from(GAM_AT),
        direction: Arc::from(DIRECTION),
    }
}

fn bayes() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(DRAWS))
}

fn run(
    data: &TabularData,
    graph: &Dag,
    functional: F,
    bandwidth: Option<f64>,
    inference: Option<InferenceMode>,
    seed: u64,
) -> Result<CausalResponse, String> {
    let mut builder = Study::tabular(data.clone())
        .graph(graph.clone())
        .query(CausalQuery::Response(ResponseQuery::new(functional)))
        .response_options(ContinuousResponseOptions {
            bandwidth,
            confidence_level: LEVEL,
            ..Default::default()
        })
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0);
    if let Some(mode) = inference {
        builder = builder.inference(mode);
    }
    let study = builder.build().map_err(|e| e.to_string())?;
    let ctx = ExecutionContext::for_tests(seed);
    let result = study.run(&ctx).map_err(|e| e.to_string())?;
    result.response.ok_or_else(|| "no response".to_owned())
}

fn scalar_value(response: &CausalResponse) -> f64 {
    let ResponseIdentification::PointIdentified(ResponseValue::Scalar(v)) = &response.estimate
    else {
        panic!("expected a point-identified scalar, got {:?}", response.estimate)
    };
    *v
}

fn scalar_interval(response: &CausalResponse) -> Option<(f64, f64)> {
    match response.uncertainty {
        ResponseUncertainty::Scalar { lower, upper, level, .. } => {
            assert!((level - LEVEL).abs() < 1e-12, "interval level {level} != {LEVEL}");
            Some((lower, upper))
        }
        _ => None,
    }
}

fn band_interval(response: &CausalResponse, j: usize) -> Option<(f64, f64)> {
    match &response.uncertainty {
        ResponseUncertainty::PointwiseBand { lower, upper, level } => {
            assert!((level - LEVEL).abs() < 1e-12, "band level {level} != {LEVEL}");
            Some((lower[j], upper[j]))
        }
        _ => None,
    }
}

/// Score a scalar functional against `truth` over `n_sim()` replicates.
fn scalar_coverage(
    name: &str,
    data_fn: fn(usize, u64) -> TabularData,
    n: usize,
    functional: &F,
    bandwidth: Option<f64>,
    bayesian: bool,
    truth: f64,
) -> CoverageTally {
    let mut tally = CoverageTally::new(name, LEVEL);
    let graph = point_graph();
    for rep in 0..u64::from(n_sim()) {
        let seed = replicate_seed(0x0D0E, rep);
        let data = data_fn(n, seed);
        match run(&data, &graph, functional.clone(), bandwidth, bayesian.then(bayes), seed) {
            Ok(response) => tally.record(scalar_interval(&response), truth),
            // A documented refusal, e.g. a posterior draw whose fitted response
            // is nonpositive under a log-outcome transform. Capped at 5%.
            Err(error) => {
                eprintln!("{name}: replicate {rep} refused: {error}");
                tally.skip();
            }
        }
    }
    tally
}

/// Print a probe's measured coverage without gating on the nominal band.
fn report_probe(name: &str, tally: &CoverageTally) {
    let (lo, hi) = coverage_band(n_sim(), LEVEL);
    eprintln!(
        "calibration {name}: nominal={LEVEL:.2} coverage={:.3} band=[{lo:.3}, {hi:.3}] \
         mean_length={:.4} (misspecification probe; recorded, not gated)",
        tally.rate(),
        tally.mean_length()
    );
}

// --- Average derivative -----------------------------------------------------

/// E[m'(A)] with A ~ N(0, 1.25): 2·exp(−1.25/2).
fn ade_truth() -> f64 {
    2.0 * (-0.625_f64).exp()
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn ade_frequentist_gaussian_treatment_nominal_90_coverage() {
    scalar_coverage(
        "ade_frequentist_gaussian_treatment",
        point_data,
        N_POINT,
        &ade_query(),
        None,
        false,
        ade_truth(),
    )
    .assert();
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn ade_bayesian_gaussian_treatment_nominal_90_coverage() {
    scalar_coverage(
        "ade_bayesian_gaussian_treatment",
        point_data,
        N_POINT,
        &ade_query(),
        None,
        true,
        ade_truth(),
    )
    .assert();
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn ade_skewed_heteroskedastic_treatment_probe() {
    // Monte-Carlo truth E[2 cos A] under the skewed treatment law.
    let mut z = gaussian(0x5EED_ADE0);
    let draws = 2_000_000;
    let truth = (0..draws).map(|_| mu_prime(skewed_treatment_draw(&mut z).1)).sum::<f64>()
        / f64::from(draws);
    let graph = point_graph();
    // The Gaussian-score assumption must be stated on the result.
    let response = run(&skewed_data(N_POINT, 1), &graph, ade_query(), None, None, 1).unwrap();
    assert!(
        response.assumptions.entries.iter().any(|record| {
            let text = format!("{:?}", record.assumption);
            text.contains("response.riesz_ade.gaussian_score")
                && text.contains("homoskedastic Gaussian treatment")
        }),
        "ADE must state its Gaussian treatment-score assumption"
    );
    for (name, bayesian) in [
        ("ade_frequentist_skewed_treatment_probe", false),
        ("ade_bayesian_skewed_treatment_probe", true),
    ] {
        let tally =
            scalar_coverage(name, skewed_data, N_POINT, &ade_query(), None, bayesian, truth);
        report_probe(name, &tally);
    }
}

// --- Point derivative and its scales -----------------------------------------

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn point_derivative_frequentist_curvature_nominal_90_coverage() {
    scalar_coverage(
        "point_derivative_frequentist_curvature",
        point_data,
        N_POINT,
        &point_query(DerivativeScale::Identity),
        Some(BANDWIDTH),
        false,
        mu_prime(AT),
    )
    .assert();
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn point_derivative_bayesian_curvature_nominal_90_coverage() {
    scalar_coverage(
        "point_derivative_bayesian_curvature",
        point_data,
        N_POINT,
        &point_query(DerivativeScale::Identity),
        Some(BANDWIDTH),
        true,
        mu_prime(AT),
    )
    .assert();
}

fn second_order_query() -> F {
    F::PointDerivative {
        outcome: vid(2),
        treatment: vid(0),
        at: AT,
        order: 2,
        scale: DerivativeScale::Identity,
    }
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn point_derivative_order_2_frequentist_curvature_nominal_90_coverage() {
    scalar_coverage(
        "point_derivative_order_2_frequentist_curvature",
        point_data,
        N_POINT,
        &second_order_query(),
        Some(SECOND_ORDER_BANDWIDTH),
        false,
        mu_second(AT),
    )
    .assert();
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn point_derivative_order_2_bayesian_curvature_nominal_90_coverage() {
    scalar_coverage(
        "point_derivative_order_2_bayesian_curvature",
        point_data,
        N_POINT,
        &second_order_query(),
        Some(SECOND_ORDER_BANDWIDTH),
        true,
        mu_second(AT),
    )
    .assert();
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn semi_elasticity_log_treatment_frequentist_nominal_90_coverage() {
    scalar_coverage(
        "semi_elasticity_log_treatment_frequentist",
        point_data,
        N_POINT,
        &point_query(DerivativeScale::LogTreatment),
        Some(BANDWIDTH),
        false,
        AT * mu_prime(AT),
    )
    .assert();
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn semi_elasticity_log_treatment_bayesian_nominal_90_coverage() {
    scalar_coverage(
        "semi_elasticity_log_treatment_bayesian",
        point_data,
        N_POINT,
        &point_query(DerivativeScale::LogTreatment),
        Some(BANDWIDTH),
        true,
        AT * mu_prime(AT),
    )
    .assert();
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn semi_elasticity_log_outcome_bayesian_nominal_90_coverage() {
    scalar_coverage(
        "semi_elasticity_log_outcome_bayesian",
        point_data,
        N_POINT,
        &point_query(DerivativeScale::LogOutcome),
        Some(BANDWIDTH),
        true,
        mu_prime(AT) / mu(AT),
    )
    .assert();
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn elasticity_bayesian_nominal_90_coverage() {
    scalar_coverage(
        "elasticity_bayesian",
        point_data,
        N_POINT,
        &point_query(DerivativeScale::LogLog),
        Some(BANDWIDTH),
        true,
        AT * mu_prime(AT) / mu(AT),
    )
    .assert();
}

// --- Multivariate GAM plug-in (Bayesian publishes a pointwise band) ----------

fn gam_coverage(name: &str, functional: &F, truth: &[f64]) -> Vec<CoverageTally> {
    let mut tallies: Vec<_> =
        (0..truth.len()).map(|j| CoverageTally::new(format!("{name}[{j}]"), LEVEL)).collect();
    let graph = gam_graph();
    for rep in 0..u64::from(n_sim()) {
        let seed = replicate_seed(0x0D0F, rep);
        let data = gam_data(N_GAM, seed);
        match run(&data, &graph, functional.clone(), None, Some(bayes()), seed) {
            Ok(response) => {
                for (j, tally) in tallies.iter_mut().enumerate() {
                    tally.record(band_interval(&response, j), truth[j]);
                }
            }
            Err(error) => {
                eprintln!("{name}: replicate {rep} refused: {error}");
                for tally in &mut tallies {
                    tally.skip();
                }
            }
        }
    }
    tallies
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn response_jacobian_bayesian_nominal_90_coverage() {
    let tallies = gam_coverage("response_jacobian_bayesian", &jacobian_query(), &JACOBIAN_TRUTH);
    for tally in &tallies {
        tally.assert();
    }
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn directional_derivative_bayesian_nominal_90_coverage() {
    let tallies =
        gam_coverage("directional_derivative_bayesian", &directional_query(), &DIRECTIONAL_TRUTH);
    for tally in &tallies {
        tally.assert();
    }
}

// --- Withheld intervals stay withheld (not a coverage run) -------------------

#[test]
fn frequentist_withheld_derivative_intervals_stay_withheld() {
    let data = point_data(400, 7);
    let graph = point_graph();
    let withheld = |response: &CausalResponse| {
        matches!(response.uncertainty, ResponseUncertainty::None)
            && response
                .support
                .warnings
                .iter()
                .any(|w| w.code.as_ref() == "response.derivative_interval_withheld")
    };
    for (label, functional) in [
        ("elasticity", point_query(DerivativeScale::LogLog)),
        ("semi_elasticity_log_outcome", point_query(DerivativeScale::LogOutcome)),
        (
            "semi_elasticity_log_treatment_order_2",
            F::PointDerivative {
                outcome: vid(2),
                treatment: vid(0),
                at: AT,
                order: 2,
                scale: DerivativeScale::LogTreatment,
            },
        ),
    ] {
        let response = run(&data, &graph, functional, Some(BANDWIDTH), None, 7).unwrap();
        assert!(scalar_value(&response).is_finite(), "{label}: point value must publish");
        assert!(withheld(&response), "{label}: Frequentist interval must stay withheld");
    }
    for (label, functional) in [
        ("semi_elasticity_log_treatment", point_query(DerivativeScale::LogTreatment)),
        ("point_derivative", point_query(DerivativeScale::Identity)),
    ] {
        let response = run(&data, &graph, functional, Some(BANDWIDTH), None, 7).unwrap();
        assert!(scalar_interval(&response).is_some(), "{label}: interval is published");
    }
    let gam = gam_data(300, 7);
    for (label, functional) in
        [("jacobian", jacobian_query()), ("directional", directional_query())]
    {
        let response = run(&gam, &gam_graph(), functional, None, None, 7).unwrap();
        assert!(
            matches!(response.uncertainty, ResponseUncertainty::None),
            "{label}: Frequentist GAM plug-in publishes no interval"
        );
    }
}
