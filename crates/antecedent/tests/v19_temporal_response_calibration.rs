//! 1.9 repeated-sampling calibration of licensed temporal response intervals.
//!
//! Cells: `ResponseCurve` / `InterventionResponse` × `TemporalDag` (Frequentist and
//! Bayesian), the observation-adjusted pair path, a horizon-dependent adjustment-set
//! surface, a two-step `Sequence` overlay (complete data and observation-adjusted), and
//! the per-completion atoms of `TemporalCpdag` / `TemporalPag` identified sets. Every
//! Frequentist band is the response family's joint circular-block bootstrap of
//! lag-aligned tuples (block `max(span, ceil(sqrt(n)))`, fixed-b and HC1 scaling); every
//! bootstrap cell uses the same 199 replicates. Every DGP is linear-Gaussian, so the
//! truth of the reported functional is analytic. Frequentist surfaces are calibrated
//! against the population level `E[Y_h | do(A)]`; the Bayesian
//! `response.temporal.bayesian` bands are declared conditional on the observed
//! covariate average and are calibrated against that functional evaluated with the true
//! coefficients.
//!
//! Each grid cell's pointwise band is tallied separately; the simultaneous band
//! (`response.simultaneous_band.*`) is tallied once per surface as "covers every cell".
//! A simultaneous tally records the slack interval
//! `[max_j (lower_j − truth_j), min_j (upper_j − truth_j)]` against zero, so its
//! reported `mean_length` is the narrowest cell's slack width, not a band width.
//!
//! Residual noise is iid (`rho = 0`) and AR(1) with `rho = 0.5`, `n = 160`. AR(1)
//! residuals are inside the stated assumptions of the block-bootstrap Frequentist
//! bands and of the per-horizon long-run-tempered Bayesian posterior (both asserted;
//! the class-atom Bayesian bands are asserted on iid residuals only). With zero replicates
//! the Frequentist surface publishes no band (the analytic band would assume
//! independent lag-aligned rows, which no DGP here satisfies); the tests assert that.
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

use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, Intervention, Lag,
    MechanismOverride, ObservationAssumption, ObservationSpec, ResponseFunctional, ResponseQuery,
    ResponseUncertainty, TemporalNodeKey, TemporalPolicy, TemporalResponseSpec, Value, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_graph::{TemporalCpdag, TemporalDag, TemporalPag, ensure_lagged};
use common::calibration::{CoverageTally, ar1_noise, gaussian, n_sim};

const N: usize = 160;
const BURN: usize = 10;
const LEVEL: f64 = 0.95;
const BOOT: u32 = 199;
const DRAWS: usize = 400;
const RHO_AR1: f64 = 0.5;
/// Lag-1 and lag-2 treatment effects of the dose × horizon DGP.
const BETA: [f64; 2] = [2.0, 1.5];

const SIM_LOWER: &str = "response.simultaneous_band.lower";
const SIM_UPPER: &str = "response.simultaneous_band.upper";

// ---------------------------------------------------------------------------
// Harness helpers
// ---------------------------------------------------------------------------

/// Assert every tally, printing all `calibration …` lines before failing.
fn assert_all(tallies: &[CoverageTally]) {
    let mut failures = Vec::new();
    for tally in tallies {
        if let Err(panic) =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| tally.assert()))
        {
            failures.push(
                panic
                    .downcast_ref::<String>()
                    .cloned()
                    .unwrap_or_else(|| "coverage assertion failed".to_owned()),
            );
        }
    }
    assert!(failures.is_empty(), "calibration failures:\n{}", failures.join("\n"));
}

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn uniform(seed: u64) -> impl FnMut() -> f64 {
    let mut state = splitmix(seed);
    move || {
        state = splitmix(state);
        (state >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn gaussian_vec(n: usize, sd: f64, seed: u64) -> Vec<f64> {
    let mut draw = gaussian(splitmix(seed));
    (0..n).map(|_| sd * draw()).collect()
}

fn lagged(values: &[f64], s: usize, lag: usize) -> f64 {
    s.checked_sub(lag).map_or(0.0, |i| values[i])
}

fn series(columns: &[(&str, &[f64])]) -> TimeSeriesData {
    let trimmed: Vec<(&str, &[f64])> =
        columns.iter().map(|(name, values)| (*name, &values[BURN..])).collect();
    TimeSeriesData::from_f64_columns(trimmed, 1).unwrap()
}

fn dag(edges: &[(u32, u32, u32, u32)]) -> TemporalDag {
    let mut graph = TemporalDag::empty();
    for &(from, from_lag, to, to_lag) in edges {
        let src =
            ensure_lagged(&mut graph, VariableId::from_raw(from), Lag::from_raw(from_lag)).unwrap();
        let dst =
            ensure_lagged(&mut graph, VariableId::from_raw(to), Lag::from_raw(to_lag)).unwrap();
        graph.insert_directed(src, dst).unwrap();
    }
    graph
}

fn temporal(horizons: &[u32]) -> TemporalResponseSpec {
    TemporalResponseSpec::new(horizons.to_vec(), TemporalPolicy::pulse(-1), None).unwrap()
}

fn curve_query(doses: &[f64], horizons: &[u32]) -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from(doses.to_vec())),
        ),
    })
    .with_temporal(temporal(horizons))
}

fn intervention_query(intervention: Intervention, horizons: &[u32]) -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(1),
        interventions: Arc::from([intervention]),
    })
    .with_temporal(temporal(horizons))
}

fn run(
    data: TimeSeriesData,
    graph: impl antecedent::IntoGraphInput,
    query: ResponseQuery,
    inference: InferenceMode,
    replicates: u32,
    seed: u64,
) -> StudyResult {
    Study::series(data)
        .graph(graph)
        .query(CausalQuery::Response(query))
        .inference(inference)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(replicates)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(seed))
        .unwrap()
}

fn bayes() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(DRAWS).prior_scale(8.0))
}

fn diagnostic<'a>(response: &'a antecedent_core::CausalResponse, id: &str) -> Option<&'a [f64]> {
    response.support.diagnostics.iter().find(|d| d.id.as_ref() == id).map(|d| d.values.as_ref())
}

fn pointwise(response: &antecedent_core::CausalResponse) -> Option<(Vec<f64>, Vec<f64>)> {
    match &response.uncertainty {
        ResponseUncertainty::PointwiseBand { lower, upper, level } if *level == LEVEL => {
            Some((lower.to_vec(), upper.to_vec()))
        }
        _ => None,
    }
}

/// Per-cell pointwise tallies plus one simultaneous tally for a response surface.
struct SurfaceTallies {
    cells: Vec<CoverageTally>,
    simultaneous: CoverageTally,
}

impl SurfaceTallies {
    fn new(name: &str, labels: &[String]) -> Self {
        Self {
            cells: labels
                .iter()
                .map(|label| CoverageTally::new(format!("{name} pointwise[{label}]"), LEVEL))
                .collect(),
            simultaneous: CoverageTally::new(format!("{name} simultaneous[slack]"), LEVEL),
        }
    }

    fn record(&mut self, response: &antecedent_core::CausalResponse, truth: &[f64]) {
        let band = pointwise(response);
        for (cell, tally) in self.cells.iter_mut().enumerate() {
            tally.record(band.as_ref().map(|(lo, hi)| (lo[cell], hi[cell])), truth[cell]);
        }
        self.simultaneous.record(simultaneous_slack(response, truth), 0.0);
    }

    fn all(&self) -> Vec<CoverageTally> {
        let mut out = self.cells.clone();
        out.push(self.simultaneous.clone());
        out
    }
}

/// `[max_j (lower_j − truth_j), min_j (upper_j − truth_j)]`: contains 0 iff the
/// simultaneous band covers every cell.
fn simultaneous_slack(
    response: &antecedent_core::CausalResponse,
    truth: &[f64],
) -> Option<(f64, f64)> {
    let lower = diagnostic(response, SIM_LOWER)?;
    let upper = diagnostic(response, SIM_UPPER)?;
    if lower.len() != truth.len() || upper.len() != truth.len() {
        return None;
    }
    let lo = lower.iter().zip(truth).map(|(l, t)| l - t).fold(f64::NEG_INFINITY, f64::max);
    let hi = upper.iter().zip(truth).map(|(u, t)| u - t).fold(f64::INFINITY, f64::min);
    Some((lo, hi))
}

// ---------------------------------------------------------------------------
// DGPs
// ---------------------------------------------------------------------------

/// Dose × horizon DGP: `T_s ~ N(0, 0.8²)` iid, `Y_s = 1 + 2 T_{s-1} + 1.5 T_{s-2} + e_s`,
/// `e` AR(1) with marginal SD 0.5. Truth: `E[Y_h | do(T_{-1} = a)] = 1 + BETA[h-1]·a`.
fn dose_horizon_series(rho: f64, seed: u64) -> TimeSeriesData {
    let n = N + BURN;
    let t = gaussian_vec(n, 0.8, seed);
    let e = ar1_noise(n, rho, 0.5, splitmix(seed ^ 0xE));
    let y: Vec<f64> = (0..n)
        .map(|s| 1.0 + BETA[0] * lagged(&t, s, 1) + BETA[1] * lagged(&t, s, 2) + e[s])
        .collect();
    series(&[("t", &t), ("y", &y)])
}

fn dose_horizon_dag() -> TemporalDag {
    dag(&[(0, 1, 1, 0), (0, 2, 1, 0)])
}

/// Horizon-dependent confounding: `Z_s ~ N(0,1)`, `A_s = Z_s + U_s`,
/// `Y_s = 1 + 2 A_{s-1} + A_{s-2} + 5 Z_{s-1} + e_s`. `I(1) = {Z@-1}`, `I(2) = {}`;
/// truth `1 + 2a` at h=1 and `1 + a` at h=2.
fn horizon_dependent_series(rho: f64, seed: u64) -> TimeSeriesData {
    let n = N + BURN;
    let z = gaussian_vec(n, 1.0, seed);
    let u = gaussian_vec(n, 1.0, splitmix(seed ^ 0xA));
    let a: Vec<f64> = z.iter().zip(&u).map(|(z, u)| z + u).collect();
    let e = ar1_noise(n, rho, 0.5, splitmix(seed ^ 0xE));
    let y: Vec<f64> = (0..n)
        .map(|s| 1.0 + 2.0 * lagged(&a, s, 1) + lagged(&a, s, 2) + 5.0 * lagged(&z, s, 1) + e[s])
        .collect();
    series(&[("t", &a), ("y", &y), ("z", &z)])
}

fn horizon_dependent_dag() -> TemporalDag {
    dag(&[(2, 0, 0, 0), (2, 1, 1, 0), (0, 1, 1, 0), (0, 2, 1, 0)])
}

/// Selected-outcome pair: `T ~ N(0, 0.8²)`, latent `Y_s = 1 + 2 T_{s-1} + e_s`,
/// `P(R_s = 1) = logistic(0.4 + 0.8 T_{s-1})`; unselected outcomes are recorded as 0.
/// The declared `OutcomeIndependentGiven([T])` holds by construction. Truth `1 + 2a`.
fn selected_series(rho: f64, seed: u64) -> TimeSeriesData {
    let n = N + BURN;
    let t = gaussian_vec(n, 0.8, seed);
    let e = ar1_noise(n, rho, 0.5, splitmix(seed ^ 0xE));
    let mut coin = uniform(seed ^ 0x5E1);
    let mut y = vec![0.0; n];
    let mut r = vec![0.0; n];
    for s in 0..n {
        let latent = 1.0 + 2.0 * lagged(&t, s, 1) + e[s];
        let p = 1.0 / (1.0 + (-(0.4 + 0.8 * lagged(&t, s, 1))).exp());
        if coin() < p {
            r[s] = 1.0;
            y[s] = latent;
        }
    }
    series(&[("t", &t), ("y", &y), ("r", &r)])
}

/// Class DGP: `R ~ N(0,1)`, `Z = 0.5 R + V`, `T = 0.5 R + 0.6 Z + U`,
/// `Y_s = 1 + 2 T_{s-1} + 1.5 Z_{s-1} + e_s`. `with_r = false` drops `R` (CPDAG
/// fixture, where `T = 0.6 Z + U` has no `R` term).
fn class_series(rho: f64, seed: u64, with_r: bool) -> TimeSeriesData {
    let n = N + BURN;
    let r = if with_r { gaussian_vec(n, 1.0, splitmix(seed ^ 0x77)) } else { vec![0.0; n] };
    let v = gaussian_vec(n, 1.0, seed);
    let u = gaussian_vec(n, 1.0, splitmix(seed ^ 0xA));
    let z: Vec<f64> = r.iter().zip(&v).map(|(r, v)| 0.5 * r + v).collect();
    let t: Vec<f64> = (0..n).map(|s| 0.5 * r[s] + 0.6 * z[s] + u[s]).collect();
    let e = ar1_noise(n, rho, 0.5, splitmix(seed ^ 0xE));
    let y: Vec<f64> =
        (0..n).map(|s| 1.0 + 2.0 * lagged(&t, s, 1) + 1.5 * lagged(&z, s, 1) + e[s]).collect();
    if with_r {
        series(&[("t", &t), ("y", &y), ("z", &z), ("r", &r)])
    } else {
        series(&[("t", &t), ("y", &y), ("z", &z)])
    }
}

fn class_cpdag() -> TemporalCpdag {
    let mut g = TemporalCpdag::empty();
    let t1 = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let z1 = g.add_lagged(VariableId::from_raw(2), Lag::from_raw(1)).unwrap();
    g.insert_directed(z1, y0).unwrap();
    g.insert_directed(t1, y0).unwrap();
    g.insert_undirected(z1, t1).unwrap();
    g
}

fn class_pag() -> TemporalPag {
    let mut g = TemporalPag::empty();
    let t1 = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let z1 = g.add_lagged(VariableId::from_raw(2), Lag::from_raw(1)).unwrap();
    let r1 = g.add_lagged(VariableId::from_raw(3), Lag::from_raw(1)).unwrap();
    g.insert_directed(r1, z1).unwrap();
    g.insert_directed(r1, t1).unwrap();
    g.insert_circle_circle_with_middle(z1, t1, antecedent_graph::MiddleMark::Empty).unwrap();
    g.insert_directed(z1, y0).unwrap();
    g.insert_directed(t1, y0).unwrap();
    g
}

fn noise_label(rho: f64) -> &'static str {
    if rho == 0.0 { "iid" } else { "AR(1) rho=0.5" }
}

fn cell_labels(doses: &[f64], horizons: &[u32]) -> Vec<String> {
    doses.iter().flat_map(|d| horizons.iter().map(move |h| format!("a={d},h={h}"))).collect()
}

// ---------------------------------------------------------------------------
// Frequentist TemporalDag ResponseCurve / InterventionResponse
// ---------------------------------------------------------------------------

const DOSES: [f64; 3] = [-1.0, 0.0, 1.0];
const HORIZONS: [u32; 2] = [1, 2];

fn curve_truth() -> Vec<f64> {
    DOSES.iter().flat_map(|&a| BETA.iter().map(move |b| 1.0 + b * a)).collect()
}

fn frequentist_curve_coverage(rho: f64, seed_base: u64) {
    let label = noise_label(rho);
    let labels = cell_labels(&DOSES, &HORIZONS);
    let mut boot = SurfaceTallies::new(
        &format!("freq TemporalDag ResponseCurve block-bootstrap {label}"),
        &labels,
    );
    let truth = curve_truth();
    for s in 0..n_sim() {
        let seed = seed_base + u64::from(s);
        let data = dose_horizon_series(rho, seed);
        let result = run(
            data.clone(),
            dose_horizon_dag(),
            curve_query(&DOSES, &HORIZONS),
            InferenceMode::Frequentist,
            BOOT,
            seed,
        );
        boot.record(result.response.as_ref().expect("surface"), &truth);
        let analytic_result = run(
            data,
            dose_horizon_dag(),
            curve_query(&DOSES, &HORIZONS),
            InferenceMode::Frequentist,
            0,
            seed,
        );
        // Zero replicates publish no band: the analytic band would treat
        // lag-aligned rows as independent.
        assert!(
            pointwise(analytic_result.response.as_ref().expect("surface")).is_none(),
            "zero replicates must not publish the analytic band"
        );
    }
    assert_all(&boot.all());
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_temporal_dag_response_curve_iid_nominal_95_coverage() {
    frequentist_curve_coverage(0.0, 190_000);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_temporal_dag_response_curve_ar1_nominal_95_coverage() {
    frequentist_curve_coverage(RHO_AR1, 191_000);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_temporal_dag_response_curve_ar1_rho09_nominal_95_coverage() {
    frequentist_curve_coverage(0.9, 191_900);
}

#[test]
fn frequentist_temporal_response_discloses_sqrt_n_block_at_ar1_rho_0_9() {
    let data = dose_horizon_series(0.9, 42);
    let result = run(
        data,
        dose_horizon_dag(),
        curve_query(&DOSES, &HORIZONS),
        InferenceMode::Frequentist,
        BOOT,
        42,
    );
    assert!(
        result
            .diagnostics
            .iter()
            .any(|d| d.code.as_ref() == "response.temporal.block.sqrt_n_rate")
            || result.response.as_ref().is_some_and(|r| {
                r.support.warnings.iter().any(|d| d.code.as_ref() == "response.temporal.block.sqrt_n_rate")
            }),
        "ρ=0.9 must disclose the √n testing-rate block; do not claim the PW-lengthened scalar construction"
    );
}

fn frequentist_intervention_coverage(rho: f64, seed_base: u64) {
    let label = noise_label(rho);
    let shift = 0.5;
    let labels: Vec<String> = HORIZONS.iter().map(|h| format!("shift=0.5,h={h}")).collect();
    let mut boot = SurfaceTallies::new(
        &format!("freq TemporalDag InterventionResponse block-bootstrap {label}"),
        &labels,
    );
    // E[T] = 0, so E[Y_h | do(T := T + 0.5)] = 1 + BETA[h-1]·0.5.
    let truth: Vec<f64> = BETA.iter().map(|b| 1.0 + b * shift).collect();
    for s in 0..n_sim() {
        let seed = seed_base + u64::from(s);
        let result = run(
            dose_horizon_series(rho, seed),
            dose_horizon_dag(),
            intervention_query(
                Intervention::soft(
                    VariableId::from_raw(0),
                    MechanismOverride::additive_shift(shift),
                ),
                &HORIZONS,
            ),
            InferenceMode::Frequentist,
            BOOT,
            seed,
        );
        boot.record(result.response.as_ref().expect("intervention path"), &truth);
    }
    assert_all(&boot.all());
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_temporal_dag_intervention_response_iid_nominal_95_coverage() {
    frequentist_intervention_coverage(0.0, 192_000);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_temporal_dag_intervention_response_ar1_nominal_95_coverage() {
    frequentist_intervention_coverage(RHO_AR1, 193_000);
}

/// Two-step Sequence `Set(T@-2 := 0.5)` then `Set(T@-1 := 1)` over horizons 1 and 2.
fn two_step_sequence() -> Intervention {
    let step = |level: f64, at: i32| antecedent_core::SequencedIntervention {
        intervention: Intervention::set(VariableId::from_raw(0), Value::f64(level)),
        temporal: TemporalPolicy::pulse(at),
    };
    Intervention::Sequence(antecedent_core::InterventionSequence::new(vec![
        step(0.5, -2),
        step(1.0, -1),
    ]))
}

/// Truth of [`two_step_sequence`] on the dose × horizon DGP: h = 1 is
/// `1 + 2·1 + 1.5·0.5 = 3.75`; h = 2 (outcome one step later, `T@0` left at its factual
/// law with mean 0) is `1 + 2·0 + 1.5·1 = 2.5`.
const SEQUENCE_TRUTH: [f64; 2] = [3.75, 2.5];

/// Complete-data multi-step Sequence on the unfolded sequential engine: one joint
/// circular-block tuple bootstrap of the whole horizon surface.
fn frequentist_sequence_coverage(rho: f64, seed_base: u64) {
    let label = noise_label(rho);
    let labels: Vec<String> = HORIZONS.iter().map(|h| format!("seq,h={h}")).collect();
    let mut boot = SurfaceTallies::new(
        &format!("freq TemporalDag Sequence tuple block-bootstrap {label}"),
        &labels,
    );
    for s in 0..n_sim() {
        let seed = seed_base + u64::from(s);
        let result = run(
            dose_horizon_series(rho, seed),
            dose_horizon_dag(),
            intervention_query(two_step_sequence(), &HORIZONS),
            InferenceMode::Frequentist,
            BOOT,
            seed,
        );
        boot.record(result.response.as_ref().expect("Sequence path"), &SEQUENCE_TRUTH);
    }
    assert_all(&boot.all());
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_temporal_dag_sequence_iid_nominal_95_coverage() {
    frequentist_sequence_coverage(0.0, 208_000);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_temporal_dag_sequence_ar1_nominal_95_coverage() {
    frequentist_sequence_coverage(RHO_AR1, 209_000);
}

// ---------------------------------------------------------------------------
// Bayesian TemporalDag InterventionResponse
// ---------------------------------------------------------------------------

fn bayesian_intervention_coverage(rho: f64, seed_base: u64) -> Vec<CoverageTally> {
    let label = noise_label(rho);
    let labels: Vec<String> = HORIZONS.iter().map(|h| format!("set=1,h={h}")).collect();
    let mut tallies = SurfaceTallies::new(
        &format!("Bayesian TemporalDag InterventionResponse credible {label}"),
        &labels,
    );
    // Both horizon designs are [1, T] (no adjustment covariates), so the declared
    // conditional-on-covariates functional equals the population level 1 + BETA[h-1].
    // At h = 2 the design omits T@-1, whose effect joins the residual; with iid T that
    // residual stays serially uncorrelated, so the untempered interval was already about
    // nominal in expectation (0.943-0.945 over 4000 fresh replicates: the 400-draw
    // quantiles run about half a point short). The per-horizon long-run tempering is what
    // keeps it nominal once the residual is serially dependent (the AR(1) cell below).
    let truth: Vec<f64> = BETA.iter().map(|b| 1.0 + b).collect();
    for s in 0..n_sim() {
        let seed = seed_base + u64::from(s);
        let result = run(
            dose_horizon_series(rho, seed),
            dose_horizon_dag(),
            intervention_query(
                Intervention::set(VariableId::from_raw(0), Value::f64(1.0)),
                &HORIZONS,
            ),
            bayes(),
            0,
            seed,
        );
        let response = result.response.as_ref().expect("Bayesian intervention path");
        if s == 0 {
            let per_horizon = response.horizon_identification.as_ref().expect("I(h)");
            assert!(per_horizon.iter().all(|h| h.adjustment.is_empty()), "{per_horizon:?}");
            let kappa = diagnostic(response, "response.temporal_bayesian.tempering")
                .expect("per-horizon tempering factor");
            assert!(kappa.len() == HORIZONS.len() && kappa.iter().all(|k| *k >= 1.0), "{kappa:?}");
        }
        tallies.record(response, &truth);
    }
    tallies.all()
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn bayesian_temporal_dag_intervention_response_iid_nominal_95_coverage() {
    assert_all(&bayesian_intervention_coverage(0.0, 194_000));
}

/// AR(1) residuals: each horizon's likelihood is tempered by its long-run-variance ratio
/// (`response.temporal_bayesian.tempering`), so serially dependent residuals are inside
/// the cell's stated generalized-posterior assumption. Before the tempering this cell was
/// a recorded misspecification probe.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn bayesian_temporal_dag_intervention_response_ar1_nominal_95_coverage() {
    assert_all(&bayesian_intervention_coverage(RHO_AR1, 195_000));
}

// ---------------------------------------------------------------------------
// Observation-adjusted pair: Selected × OutcomeIndependentGiven([T])
// ---------------------------------------------------------------------------

fn observation_coverage(rho: f64, seed_base: u64) {
    let label = noise_label(rho);
    let horizons = [1u32];
    let labels = cell_labels(&DOSES, &horizons);
    let mut tallies = SurfaceTallies::new(
        &format!("freq TemporalDag Selected-AIPW outer block-bootstrap {label}"),
        &labels,
    );
    let truth: Vec<f64> = DOSES.iter().map(|a| 1.0 + 2.0 * a).collect();
    let id = VariableId::from_raw;
    for s in 0..n_sim() {
        let seed = seed_base + u64::from(s);
        let query = curve_query(&DOSES, &horizons).with_observation(
            ObservationSpec::Selected { latent: id(1), observed: id(1), indicator: id(2) },
            [ObservationAssumption::OutcomeIndependentGiven(Arc::from([id(0)]))],
        );
        let result = run(
            selected_series(rho, seed),
            dag(&[(0, 1, 1, 0)]),
            query,
            InferenceMode::Frequentist,
            BOOT,
            seed,
        );
        let response = result.response.as_ref().expect("observation-adjusted surface");
        assert_eq!(
            response.provenance_id.as_ref(),
            "estimate.temporal_response.observation_adjusted"
        );
        tallies.record(response, &truth);
    }
    assert_all(&tallies.all());
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_temporal_observation_selected_iid_nominal_95_coverage() {
    observation_coverage(0.0, 196_000);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_temporal_observation_selected_ar1_nominal_95_coverage() {
    observation_coverage(RHO_AR1, 197_000);
}

// ---------------------------------------------------------------------------
// Observation-adjusted multi-step Sequence: Selected × OutcomeIndependentGiven([T])
// ---------------------------------------------------------------------------

/// Two-lag selected-outcome pair: `T ~ N(0, 0.8²)`, latent
/// `Y_s = 1 + 2 T_{s-1} + 1.5 T_{s-2} + e_s`, `P(R_s = 1) = logistic(0.4 + 0.8 T_{s-1})`;
/// unselected outcomes are recorded as 0. Selection depends only on `T_{s-1}`, so the
/// declared `OutcomeIndependentGiven([T])` (at the policy offset −1) holds.
///
/// The unfolded Sequence design regresses the pseudo-outcome on `T_{s-1}` and `T_{s-2}`,
/// so the selected-AIPW outcome nuisance conditions on both (the selection model keeps
/// the declared `T_{s-1}`). With the declared set alone the nuisance omitted `T_{s-2}`,
/// the pseudo-outcome regression was not orthogonal to the estimated selection
/// probability, and the iid band over-covered at the top of the acceptance band (0.983).
fn selected_two_lag_series(rho: f64, seed: u64) -> TimeSeriesData {
    let n = N + BURN;
    let t = gaussian_vec(n, 0.8, seed);
    let e = ar1_noise(n, rho, 0.5, splitmix(seed ^ 0xE));
    let mut coin = uniform(seed ^ 0x5E2);
    let mut y = vec![0.0; n];
    let mut r = vec![0.0; n];
    for s in 0..n {
        let latent = 1.0 + BETA[0] * lagged(&t, s, 1) + BETA[1] * lagged(&t, s, 2) + e[s];
        let p = 1.0 / (1.0 + (-(0.4 + 0.8 * lagged(&t, s, 1))).exp());
        if coin() < p {
            r[s] = 1.0;
            y[s] = latent;
        }
    }
    series(&[("t", &t), ("y", &y), ("r", &r)])
}

/// [`two_step_sequence`] under the selected pair (truth [`SEQUENCE_TRUTH`]). Every outer
/// replicate refits the selected-AIPW nuisance and every unfolded sequential mechanism on
/// the same resampled outcome-time tuples.
fn observation_sequence_coverage(rho: f64, seed_base: u64) {
    let label = noise_label(rho);
    let labels: Vec<String> = HORIZONS.iter().map(|h| format!("seq,h={h}")).collect();
    let mut tallies = SurfaceTallies::new(
        &format!("freq TemporalDag Selected-AIPW Sequence outer block-bootstrap {label}"),
        &labels,
    );
    let id = VariableId::from_raw;
    for s in 0..n_sim() {
        let seed = seed_base + u64::from(s);
        let query = intervention_query(two_step_sequence(), &HORIZONS).with_observation(
            ObservationSpec::Selected { latent: id(1), observed: id(1), indicator: id(2) },
            [ObservationAssumption::OutcomeIndependentGiven(Arc::from([id(0)]))],
        );
        let result = run(
            selected_two_lag_series(rho, seed),
            dose_horizon_dag(),
            query,
            InferenceMode::Frequentist,
            BOOT,
            seed,
        );
        let response = result.response.as_ref().expect("observation-adjusted Sequence");
        assert_eq!(
            response.provenance_id.as_ref(),
            "estimate.temporal_response.observation_adjusted"
        );
        tallies.record(response, &SEQUENCE_TRUTH);
    }
    assert_all(&tallies.all());
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_temporal_observation_sequence_iid_nominal_95_coverage() {
    observation_sequence_coverage(0.0, 206_000);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_temporal_observation_sequence_ar1_nominal_95_coverage() {
    observation_sequence_coverage(RHO_AR1, 207_000);
}

// ---------------------------------------------------------------------------
// Horizon-dependent adjustment sets (identify.temporal_response.horizon_dependent)
// ---------------------------------------------------------------------------

fn horizon_dependent_coverage(rho: f64, seed_base: u64) {
    let label = noise_label(rho);
    let doses = [0.0, 1.0];
    let labels = cell_labels(&doses, &HORIZONS);
    let mut boot = SurfaceTallies::new(
        &format!("freq TemporalDag horizon-dependent I(h) block-bootstrap {label}"),
        &labels,
    );
    let truth: Vec<f64> = doses.iter().flat_map(|&a| [1.0 + 2.0 * a, 1.0 + a]).collect();
    for s in 0..n_sim() {
        let seed = seed_base + u64::from(s);
        let data = horizon_dependent_series(rho, seed);
        let result = run(
            data.clone(),
            horizon_dependent_dag(),
            curve_query(&doses, &HORIZONS),
            InferenceMode::Frequentist,
            BOOT,
            seed,
        );
        if s == 0 {
            assert!(
                result
                    .diagnostics
                    .iter()
                    .any(|d| d.code.as_ref() == "identify.temporal_response.horizon_dependent"),
                "the fixture must exercise horizon-dependent identification"
            );
            let per_horizon = result
                .response
                .as_ref()
                .and_then(|r| r.horizon_identification.as_ref())
                .expect("I(h)");
            assert_eq!(
                per_horizon[0].adjustment.as_ref(),
                &[TemporalNodeKey { variable: VariableId::from_raw(2), offset: -1 }]
            );
            assert!(per_horizon[1].adjustment.is_empty());
        }
        boot.record(result.response.as_ref().expect("surface"), &truth);
        let analytic_result = run(
            data,
            horizon_dependent_dag(),
            curve_query(&doses, &HORIZONS),
            InferenceMode::Frequentist,
            0,
            seed,
        );
        // Zero replicates publish no band: the analytic band would treat
        // lag-aligned rows as independent.
        assert!(
            pointwise(analytic_result.response.as_ref().expect("surface")).is_none(),
            "zero replicates must not publish the analytic band"
        );
    }
    assert_all(&boot.all());
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_temporal_horizon_dependent_iid_nominal_95_coverage() {
    horizon_dependent_coverage(0.0, 198_000);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_temporal_horizon_dependent_ar1_nominal_95_coverage() {
    horizon_dependent_coverage(RHO_AR1, 199_000);
}

// ---------------------------------------------------------------------------
// TemporalCpdag / TemporalPag: per-completion atom bands
// ---------------------------------------------------------------------------
//
// The class response publishes an identified set of per-completion point surfaces
// and no class-level band. What carries an interval is each completion atom, whose
// band is calibrated here against that atom's own probability limit:
// - adjust {Z@-1}: 1 + 2a (Z blocks every backdoor);
// - adjust {} on the CPDAG (Z treated as a mediator of T): the population regression
//   E[Y | T_{-1} = a] = 1 + (2 + 1.5·Cov(Z,T)/Var(T))·a = 1 + (2 + 0.9/1.36)·a.

const CLASS_DOSES: [f64; 2] = [0.0, 1.0];

fn atom_adjustment(atom: &antecedent::result::StructuralResponseAtom) -> Vec<TemporalNodeKey> {
    atom.response
        .as_ref()
        .and_then(|r| r.horizon_identification.as_ref())
        .map(|h| h[0].adjustment.to_vec())
        .unwrap_or_default()
}

fn class_atom_coverage(
    class: &str,
    rho: f64,
    seed_base: u64,
    inference: &InferenceMode,
    replicates: u32,
) -> Vec<CoverageTally> {
    let label = noise_label(rho);
    let bayesian = matches!(inference, InferenceMode::Bayesian(_));
    let mode = if bayesian { "Bayesian credible" } else { "freq block-bootstrap" };
    let z_key = TemporalNodeKey { variable: VariableId::from_raw(2), offset: -1 };
    let mediator_slope = 2.0 + 1.5 * 0.6 / (0.36 + 1.0);
    let mut by_adjustment: Vec<(Vec<TemporalNodeKey>, SurfaceTallies)> = Vec::new();
    for s in 0..n_sim() {
        let seed = seed_base + u64::from(s);
        let data = class_series(rho, seed, class == "pag");
        let query = curve_query(&CLASS_DOSES, &[1]);
        let result = if class == "pag" {
            run(data.clone(), class_pag(), query, inference.clone(), replicates, seed)
        } else {
            run(data.clone(), class_cpdag(), query, inference.clone(), replicates, seed)
        };
        let structural = result.structural_response.as_ref().expect("identified set");
        assert!(
            matches!(
                result.response.as_ref().map(|r| &r.uncertainty),
                Some(ResponseUncertainty::None)
            ),
            "the class response publishes no class-level band"
        );
        // Sample mean of Z over the lag-aligned rows (anchor s = 1..n): the Bayesian
        // band is conditional on this covariate average.
        let z_bar = {
            let z = data_column(&data, 2);
            z[..z.len() - 1].iter().sum::<f64>() / (z.len() - 1) as f64
        };
        let mut seen_this_replicate: Vec<Vec<TemporalNodeKey>> = Vec::new();
        for atom in structural.atoms.iter().filter(|atom| atom.value.is_some()) {
            let adjustment = atom_adjustment(atom);
            // Completions sharing an adjustment set publish the identical band; count it
            // once per replicate so the tally's Monte Carlo error stays honest.
            if seen_this_replicate.contains(&adjustment) {
                continue;
            }
            seen_this_replicate.push(adjustment.clone());
            let adjusts_z = adjustment.contains(&z_key);
            let truth: Vec<f64> = CLASS_DOSES
                .iter()
                .map(|a| {
                    if adjusts_z {
                        1.0 + 2.0 * a + if bayesian { 1.5 * z_bar } else { 0.0 }
                    } else {
                        1.0 + mediator_slope * a
                    }
                })
                .collect();
            let position = by_adjustment.iter().position(|(adj, _)| *adj == adjustment);
            let index = position.unwrap_or_else(|| {
                let name = format!(
                    "{mode} Temporal{} atom adjust={} {label}",
                    if class == "pag" { "Pag" } else { "Cpdag" },
                    if adjusts_z { "{Z@-1}" } else { "{}" }
                );
                by_adjustment.push((
                    adjustment.clone(),
                    SurfaceTallies::new(&name, &cell_labels(&CLASS_DOSES, &[1])),
                ));
                by_adjustment.len() - 1
            });
            by_adjustment[index].1.record(atom.response.as_ref().expect("atom band"), &truth);
        }
    }
    assert!(!by_adjustment.is_empty(), "no identified completion atom");
    if class == "cpdag" {
        assert_eq!(by_adjustment.len(), 2, "the CPDAG fixture must keep two distinct completions");
    }
    by_adjustment.iter().flat_map(|(_, tallies)| tallies.all()).collect()
}

fn data_column(data: &TimeSeriesData, variable: u32) -> Vec<f64> {
    use antecedent_data::TableView;
    match data.column(VariableId::from_raw(variable)).unwrap() {
        antecedent_data::ColumnView::Float64(column) => column.values.to_vec(),
        _ => panic!("float column"),
    }
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_temporal_cpdag_response_atom_iid_nominal_95_coverage() {
    assert_all(&class_atom_coverage("cpdag", 0.0, 200_000, &InferenceMode::Frequentist, BOOT));
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_temporal_cpdag_response_atom_ar1_nominal_95_coverage() {
    assert_all(&class_atom_coverage("cpdag", RHO_AR1, 201_000, &InferenceMode::Frequentist, BOOT));
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_temporal_pag_response_atom_iid_nominal_95_coverage() {
    assert_all(&class_atom_coverage("pag", 0.0, 204_000, &InferenceMode::Frequentist, BOOT));
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn frequentist_temporal_pag_response_atom_ar1_nominal_95_coverage() {
    assert_all(&class_atom_coverage("pag", RHO_AR1, 202_000, &InferenceMode::Frequentist, BOOT));
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn bayesian_temporal_cpdag_response_atom_iid_nominal_95_coverage() {
    assert_all(&class_atom_coverage("cpdag", 0.0, 203_000, &bayes(), 0));
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn bayesian_temporal_pag_response_atom_iid_nominal_95_coverage() {
    assert_all(&class_atom_coverage("pag", 0.0, 205_000, &bayes(), 0));
}
