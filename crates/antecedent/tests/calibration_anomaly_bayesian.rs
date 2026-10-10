//! Coverage test for `AnomalyAttribution / Dag / explicit / Bayesian`.
//!
//! The IT anomaly score is `−log 2Φ(−|y − center|/scale)` against a reference
//! `(center, scale)`. With the default *empirical* reference the reference is the
//! observed sample's robust median / `1.4826·MAD`, so the mean anomaly score has
//! no fixed population value to cover. Injecting a **fixed** reference equal to the
//! target's true `(μ_Y, σ_Y)` turns the mean anomaly score into a fixed functional
//! of the target's law, and the Bayesian path publishes its Bayesian-bootstrap
//! (shared Dirichlet row-weight) posterior — a genuine credible interval for that
//! functional rather than a degenerate point.
//!
//! Ignored tests run via `scripts/gate_calibration.sh`. The non-ignored test
//! checks the population truth end to end and does not emit a coverage record.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::doc_markdown)]
#![allow(
    clippy::float_cmp,
    reason = "test scaffolding compares exact constants and indexes with small literals"
)]

mod common;

use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    AnomalyAttributionQuery, AnomalyReference, CausalQuery, ExecutionContext, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};
use common::calibration::{
    CoverageTally, RecordKey, SampleGrid, gaussian, map_replicates, n_sim, stream_seed,
};
use common::calibration_bind::bind_all;
use common::reported::{GATE_LEVEL, REPORTED_LEVEL, gate, posterior_pair, record_pair, skip_pair};

// DGP: DAG z → y with z ~ N(0, 1) and y = A + B·z + ε, ε ~ N(0, SIGMA²).
// Then y ~ N(A, B² + SIGMA²), i.e. (μ_Y, σ_Y) = (A, sqrt(B² + SIGMA²)).
const A: f64 = 1.0;
const B: f64 = 2.0;
const SIGMA: f64 = 0.7;
const ANOMALY_N: usize = 1000;

/// The target's true marginal scale σ_Y = sqrt(B² + SIGMA²) = sqrt(4.49).
fn fixed_scale() -> f64 {
    (B * B + SIGMA * SIGMA).sqrt()
}

/// Population mean anomaly score μ_A against the fixed reference `(A, σ_Y)`.
///
/// With the injected reference equal to the target's true `(μ_Y, σ_Y)`, the
/// standardized deviation `z = (Y − A)/σ_Y` is exactly standard normal, so the
/// two-sided Gaussian-tail probability `U = 2Φ(−|z|)` is the two-sided p-value of a
/// standard normal — which is `Uniform(0, 1)`. The score is `−log U`, so
///
/// `μ_A = E_U[−log U]`, `U ~ Uniform(0, 1)` `= ∫₀¹ −ln u du = 1`
///
/// (equivalently, `−log U ~ Exp(1)`, whose mean is 1). This is a closed form,
/// independent of `A`, `B`, `SIGMA`. It is cross-checked by Gauss–Hermite
/// quadrature of `E_Z[−log 2Φ(−|Z|)]` against the standard normal in
/// [`anomaly_fixed_reference_matches_population_truth`].
const ANOMALY_MEAN_SCORE_TRUTH: f64 = 1.0;

/// The estimator's IT tail score `−log 2Φ(−|z|)`, mirroring
/// `antecedent_attribution`'s `OutlierTail::score` including its log-space
/// asymptotic branch for `|z| ≳ 38` where `2Φ(−z)` underflows.
fn tail_score(z: f64) -> f64 {
    let z = z.abs();
    let two_sided = 2.0 * antecedent_kernels::norm_sf(z);
    if two_sided > f64::MIN_POSITIVE {
        -two_sided.ln()
    } else {
        0.5 * z * z + z.ln() + 0.5 * std::f64::consts::TAU.ln() - std::f64::consts::LN_2
    }
}

/// Gauss–Hermite quadrature of `μ_A = E_Z[−log 2Φ(−|Z|)]` against the standard
/// normal, reusing `antecedent_stats::gauss_hermite_standard_normal`
/// (`E[g(Z)] ≈ Σ w_i g(z_i)`). Converges to the closed form `1.0`, but the kink
/// of `|z|` at the origin (a derivative discontinuity) makes the convergence only
/// algebraic — ≈ 0.99 at 256 nodes — which is exactly why the recorded truth is
/// the exact closed form above rather than this quadrature value.
fn anomaly_mean_score_quadrature(nodes: usize) -> f64 {
    let (z, w) = antecedent_stats::gauss_hermite_standard_normal(nodes);
    z.iter().zip(w.iter()).map(|(&zi, &wi)| wi * tail_score(zi)).sum()
}

fn anomaly_dag_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let (mut z, mut y) = (vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        z[i] = g();
        y[i] = A + B * z[i] + SIGMA * g();
    }
    TabularData::from_f64_columns([("z", z.as_slice()), ("y", y.as_slice())]).unwrap()
}

fn anomaly_dag() -> Dag {
    let mut g = Dag::with_variables(2);
    g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    g
}

fn bayes() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(400))
}

/// Anomaly attribution on `y`, with the fixed reference set to the target's true
/// `(μ_Y, σ_Y)`.
fn anomaly_query() -> CausalQuery {
    CausalQuery::AnomalyAttribution(
        AnomalyAttributionQuery::new([VariableId::from_raw(1)], 1_000_000)
            .with_reference(AnomalyReference::fixed(A, fixed_scale())),
    )
}

fn run_anomaly(n: usize, seed: u64) -> Option<(Study, StudyResult)> {
    let data = anomaly_dag_data(n, seed);
    let study = Study::tabular(data)
        .graph(anomaly_dag())
        .query(anomaly_query())
        .inference(bayes())
        .refute(RefuteSuite::None)
        .build()
        .ok()?;
    let result = study.run(&ExecutionContext::for_tests(seed)).ok()?;
    Some((study, result))
}

fn keyed(test: &'static str, level: f64) -> CoverageTally {
    CoverageTally::for_record(
        RecordKey { test, dgp: "anomaly_dag_data", interval: "posterior_quantile" },
        level,
    )
}

/// End-to-end check that the fixed-reference Bayesian anomaly path publishes a
/// non-degenerate posterior centered on the derived population truth `μ_A = 1.0`.
#[test]
fn anomaly_fixed_reference_matches_population_truth() {
    // (a) Quadrature cross-check of the closed-form truth. GH converges only
    // algebraically here (kink at 0), so a coarse tolerance corroborates μ_A ≈ 1.0.
    let quad = anomaly_mean_score_quadrature(256);
    assert!(
        (quad - ANOMALY_MEAN_SCORE_TRUTH).abs() < 3e-2,
        "Gauss–Hermite μ_A = {quad}, expected ≈ {ANOMALY_MEAN_SCORE_TRUTH}"
    );

    // (b) End to end: a large sample with the fixed reference yields a
    // NON-degenerate posterior centered on the truth.
    let (_, result) = run_anomaly(4000, 0xA0_0417).expect("anomaly runs");
    assert_eq!(result.logical_plan.estimator.as_deref(), Some("gcm.fit.bayesian"));
    let posterior = result.posterior.as_ref().expect("fixed-reference anomaly posterior");
    let (lo, hi) = (posterior.summaries.q025[0], posterior.summaries.q975[0]);
    assert!(hi > lo, "posterior must be non-degenerate: [{lo}, {hi}]");
    assert!(
        (posterior.summaries.mean[0] - ANOMALY_MEAN_SCORE_TRUTH).abs() < 0.05,
        "posterior mean {} is far from μ_A = {ANOMALY_MEAN_SCORE_TRUTH}",
        posterior.summaries.mean[0]
    );
}

/// The default (empirical) reference Bayesian anomaly posterior must also be
/// non-degenerate: the shared-Dirichlet-row-weight Bayesian bootstrap reweights
/// the per-row scores, so the interval has positive width even though the factual
/// scores do not depend on the mechanism refit. (Before this fix the empirical
/// posterior was a zero-width point.) It is not calibrated to a fixed truth — the
/// reference is the observed-sample median/MAD — so only non-degeneracy is asserted.
#[test]
fn empirical_reference_posterior_is_non_degenerate() {
    let data = anomaly_dag_data(2000, 0xA0_0419);
    let study = Study::tabular(data)
        .graph(anomaly_dag())
        .query(CausalQuery::AnomalyAttribution(AnomalyAttributionQuery::new(
            [VariableId::from_raw(1)],
            1_000_000,
        )))
        .inference(bayes())
        .refute(RefuteSuite::None)
        .build()
        .expect("study builds");
    let result = study.run(&ExecutionContext::for_tests(0xA0_0419)).expect("anomaly runs");
    assert_eq!(result.logical_plan.estimator.as_deref(), Some("gcm.fit.bayesian"));
    let posterior = result.posterior.as_ref().expect("empirical-reference anomaly posterior");
    let (lo, hi) = (posterior.summaries.q025[0], posterior.summaries.q975[0]);
    assert!(hi > lo, "empirical posterior must be non-degenerate: [{lo}, {hi}]");
    assert!(
        posterior.summaries.mean[0] > 0.0,
        "mean {} must be positive",
        posterior.summaries.mean[0]
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn anomaly_attribution_dag_bayesian_nominal_coverage() {
    let mut tallies = [
        keyed("anomaly_attribution_dag_bayesian_nominal_coverage", REPORTED_LEVEL),
        keyed("anomaly_attribution_dag_bayesian_nominal_coverage", GATE_LEVEL),
    ];
    let runs = map_replicates(n_sim(), |rep| {
        let seed = stream_seed(0xA0_0417, rep);
        let (study, result) = run_anomaly(SampleGrid::HEAVY.n(ANOMALY_N), seed)?;
        let intervals = posterior_pair(&result, 0);
        if rep == 0 {
            assert_eq!(result.logical_plan.estimator.as_deref(), Some("gcm.fit.bayesian"));
            assert!(
                intervals[0].is_some(),
                "the fixed-reference anomaly posterior must publish an interval"
            );
            // The whole point of the fixed-reference Bayesian bootstrap: the
            // interval is not the degenerate point the empirical reference gives.
            let (lo, hi) = intervals[0].expect("interval present");
            assert!(hi > lo, "posterior interval must be non-degenerate: [{lo}, {hi}]");
        }
        Some((study, result, intervals))
    });
    for scored in &runs {
        let Some((study, result, intervals)) = scored else {
            skip_pair(&mut tallies);
            continue;
        };
        let [first, second] = &mut tallies;
        bind_all(&mut [first, second], study, result);
        // Truth μ_A = 1.0, derived on `ANOMALY_MEAN_SCORE_TRUTH` (uniform two-sided
        // p-value ⇒ −log U ~ Exp(1), mean 1; Gauss–Hermite cross-check above).
        record_pair(&mut tallies, *intervals, ANOMALY_MEAN_SCORE_TRUTH);
    }
    gate(&tallies, &[None, None]);
}
