//! 1.9 repeated-sampling coverage for the remaining static cells (R-19, R-17).
//!
//! Static responses (Kennedy-DR curve bands, the g-computation intervention
//! level, the `cell.aipw` joint cell mean, and the CPDAG class-aware joint-IF
//! band; disagreeing completions publish an unbanded identified set, pinned
//! here), static mediation (NDE/NIE,
//! Frequentist bootstrap and the Bayesian product-of-Gaussians posterior),
//! path-specific effects, interventional distributions near 0 and 1, and the
//! Bayesian counterfactual mean ITE. Each DGP is inside the cell's stated
//! assumptions; the truth is derived next to each generator. Probes outside a
//! cell's assumptions (the R-17 `bayesian.gcomp` misspecification, the
//! Bayesian response scored against the population rather than the
//! sample-covariate target, a mediator–outcome-confounded mediation law)
//! print their measured coverage with `CoverageTally::report`-style lines and
//! are not gated.
//!
//! Ignored tests run via `scripts/gate_calibration.sh` (release build).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::doc_markdown,
    clippy::too_many_arguments
)]

mod common;

use std::sync::Arc;

use antecedent::{BayesianConfig, EstimatorId, InferenceMode, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ContinuousDomain, CounterfactualQuery, ExecutionContext,
    GridSpec, Intervention, InterventionalDistributionQuery, MediationContrast, MediationQuery,
    PathSpecificEffectQuery, ResponseFunctional, ResponseIdentification, ResponseQuery,
    ResponseUncertainty, ResponseValue, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_estimate::ContinuousResponseOptions;
use antecedent_graph::{Cpdag, Dag, DenseNodeId};

use common::calibration::{
    CoverageTally, Z90, gaussian, mix_seed, n_sim, normal_interval, quantile_interval,
};

const LEVEL: f64 = 0.9;
const DRAWS: usize = 400;

fn d(i: u32) -> DenseNodeId {
    DenseNodeId::from_raw(i)
}

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

fn dag(n: u32, edges: &[(u32, u32)]) -> Dag {
    let mut g = Dag::with_variables(n);
    for &(a, b) in edges {
        g.insert_directed(d(a), d(b)).unwrap();
    }
    g
}

/// Deterministic U(0, 1) stream, decorrelated from [`gaussian`] by the seed mix.
fn uniform(seed: u64) -> impl FnMut() -> f64 {
    let mut state = mix_seed(seed ^ 0x0F0F_F0F0_1234_5678) | 1;
    move || {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (state >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn bernoulli(u: &mut impl FnMut() -> f64, p: f64) -> f64 {
    f64::from(u() < p)
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn bayes() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(DRAWS).prior_scale(10.0))
}

fn table(columns: &[(&str, &[f64])]) -> TabularData {
    TabularData::from_f64_columns(columns.iter().map(|(name, col)| (*name, *col))).unwrap()
}

/// Posterior equal-tailed interval of the effect column.
fn posterior_interval(result: &StudyResult) -> Option<(f64, f64)> {
    let posterior = result.posterior.as_ref()?;
    let col = posterior.effect_column()?;
    quantile_interval(posterior.draws.column(col).ok()?, LEVEL)
}

/// Print a probe's measured coverage without gating.
fn report_probe(name: &str, covered: u32, scored: u32, extra: &str) {
    eprintln!(
        "calibration {name} (probe, not gated): nominal={LEVEL:.2} coverage={:.3} ({covered}/{scored}) {extra}",
        f64::from(covered) / f64::from(scored.max(1))
    );
}

fn mean_sd(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let sd = (values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0)).sqrt();
    (mean, sd)
}

// ======================================================================
// Static responses on a Dag
// ======================================================================

/// Continuous-treatment response law (columns `t, y, z`):
/// `z ~ N(0,1)`, `t = ρ z + √(1−ρ²) e` (Var t = 1), `y = 1 + 2t + 0.8z + e`.
///
/// The dose–response `m(a) = E[Y | do(T=a)] = 1 + 2a` is linear, so the
/// Kennedy local-quadratic smoother is unbiased at every bandwidth, the
/// additive-GAM outcome and Gaussian treatment-law nuisances are correctly
/// specified, and the Bayesian linear-additive model is the true model.
/// The gated tests use [`RHO`] = 0.3; the Kennedy overlap probe uses 0.6, where
/// the density ratio `f(t)/f(t|z)` is heavy-tailed at the grid ends.
/// Returns the sample mean of `z` for the sample-covariate target.
fn response_data_rho(n: usize, rho: f64, seed: u64) -> (TabularData, f64) {
    let mut g = gaussian(seed);
    let (mut t, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let sd = (1.0 - rho * rho).sqrt();
    for i in 0..n {
        z[i] = g();
        t[i] = rho * z[i] + sd * g();
        y[i] = 1.0 + 2.0 * t[i] + 0.8 * z[i] + g();
    }
    let z_bar = z.iter().sum::<f64>() / n as f64;
    (table(&[("t", &t), ("y", &y), ("z", &z)]), z_bar)
}

/// Confounding strength of the gated response DGP.
const RHO: f64 = 0.3;

fn response_data(n: usize, seed: u64) -> (TabularData, f64) {
    response_data_rho(n, RHO, seed)
}

fn response_dag() -> Dag {
    dag(3, &[(2, 0), (2, 1), (0, 1)])
}

const GRID: [f64; 5] = [-1.0, -0.5, 0.0, 0.5, 1.0];

fn curve_query() -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: v(1),
        treatment: ContinuousDomain::new(v(0), GridSpec::Values(GRID.to_vec().into())),
    })
}

fn level_query(level: f64) -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: v(1),
        interventions: Arc::from([Intervention::set(v(0), Value::f64(level))]),
    })
}

fn response_options(simultaneous: Option<u32>) -> ContinuousResponseOptions {
    ContinuousResponseOptions {
        bandwidth: Some(0.5),
        confidence_level: LEVEL,
        simultaneous_replicates: simultaneous,
        ..ContinuousResponseOptions::default()
    }
}

fn run_response(
    data: TabularData,
    graph: impl antecedent::IntoGraphInput,
    query: ResponseQuery,
    inference: InferenceMode,
    options: Option<ContinuousResponseOptions>,
    estimator: Option<EstimatorId>,
    seed: u64,
) -> Option<StudyResult> {
    let mut builder = Study::tabular(data)
        .graph(graph)
        .query(CausalQuery::Response(query))
        .inference(inference)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0);
    if let Some(options) = options {
        builder = builder.response_options(options);
    }
    if let Some(estimator) = estimator {
        builder = builder.estimator(estimator);
    }
    builder.build().ok()?.run(&ExecutionContext::for_tests(seed)).ok()
}

/// `(mean, lower, upper)` per grid coordinate of a pointwise or simultaneous band.
fn band(result: &StudyResult) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let response = result.response.as_ref()?;
    let ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) =
        &response.estimate
    else {
        return None;
    };
    let (ResponseUncertainty::PointwiseBand { lower, upper, .. }
    | ResponseUncertainty::SimultaneousBand { lower, upper, .. }) = &response.uncertainty
    else {
        return None;
    };
    Some((mean.to_vec(), lower.to_vec(), upper.to_vec()))
}

fn scalar_interval(result: &StudyResult) -> Option<(f64, f64)> {
    match result.response.as_ref()?.uncertainty {
        ResponseUncertainty::Scalar { lower, upper, .. } => Some((lower, upper)),
        _ => None,
    }
}

/// Kennedy-DR pointwise band: nominal 90% at each of the five grid points,
/// scored against `m(a) = 1 + 2a`.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn response_curve_dag_frequentist_pointwise_nominal_90_coverage() {
    let mut tallies: Vec<CoverageTally> = GRID
        .iter()
        .map(|a| {
            CoverageTally::new(format!("response_curve_dag_frequentist_pointwise_a={a}"), LEVEL)
        })
        .collect();
    for rep in 0..u64::from(n_sim()) {
        let seed = 21_000 + rep;
        let (data, _) = response_data(500, seed);
        let result = run_response(
            data,
            response_dag(),
            curve_query(),
            InferenceMode::Frequentist,
            Some(response_options(None)),
            None,
            seed,
        );
        let bands = result.as_ref().and_then(band);
        for (j, tally) in tallies.iter_mut().enumerate() {
            match &bands {
                Some((_, lower, upper)) => {
                    tally.record(Some((lower[j], upper[j])), 1.0 + 2.0 * GRID[j]);
                }
                None => tally.skip(),
            }
        }
    }
    for tally in &tallies {
        tally.assert();
    }
}

/// Overlap-stress probe (ρ = 0.6): same law and band, measured coverage only.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn response_curve_dag_frequentist_weak_overlap_probe() {
    let mut covered = [0u32; 5];
    let mut scored = 0u32;
    for rep in 0..u64::from(n_sim()) {
        let seed = 21_500 + rep;
        let (data, _) = response_data_rho(500, 0.6, seed);
        let result = run_response(
            data,
            response_dag(),
            curve_query(),
            InferenceMode::Frequentist,
            Some(response_options(None)),
            None,
            seed,
        );
        if let Some((_, lower, upper)) = result.as_ref().and_then(band) {
            scored += 1;
            for (j, a) in GRID.iter().enumerate() {
                let truth = 1.0 + 2.0 * a;
                covered[j] += u32::from(lower[j] <= truth && truth <= upper[j]);
            }
        }
    }
    for (j, a) in GRID.iter().enumerate() {
        report_probe(
            &format!("response_curve_dag_frequentist_weak_overlap_a={a}"),
            covered[j],
            scored,
            "t = 0.6 z + 0.8 e (heavy-tailed density ratio)",
        );
    }
}

/// Kennedy-DR simultaneous (sup-t multiplier) band: the joint event "all five
/// grid truths inside the band" at nominal 90%. Scored as in the Unknown-cell
/// test: the unit interval against 0.5 (covered) or 2.0 (missed).
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn response_curve_dag_frequentist_simultaneous_nominal_90_coverage() {
    let mut tally = CoverageTally::new("response_curve_dag_frequentist_simultaneous", LEVEL);
    let mut half_widths = Vec::new();
    for rep in 0..u64::from(n_sim()) {
        let seed = 22_000 + rep;
        let (data, _) = response_data(500, seed);
        let result = run_response(
            data,
            response_dag(),
            curve_query(),
            InferenceMode::Frequentist,
            Some(response_options(Some(1_000))),
            None,
            seed,
        );
        let Some(result) = result else {
            tally.skip();
            continue;
        };
        assert!(
            matches!(
                result.response.as_ref().map(|r| &r.uncertainty),
                Some(ResponseUncertainty::SimultaneousBand { .. })
            ),
            "simultaneous replicates must publish a SimultaneousBand"
        );
        let Some((mean, lower, upper)) = band(&result) else {
            tally.record(None, 0.5);
            continue;
        };
        let covered = GRID
            .iter()
            .enumerate()
            .all(|(j, a)| lower[j] <= 1.0 + 2.0 * a && 1.0 + 2.0 * a <= upper[j]);
        debug_assert_eq!(mean.len(), GRID.len());
        half_widths.push((upper[2] - lower[2]) / 2.0);
        tally.record(Some((0.0, 1.0)), if covered { 0.5 } else { 2.0 });
    }
    let (w, _) = mean_sd(&half_widths);
    eprintln!(
        "calibration response_curve_dag_frequentist_simultaneous: mean half-width at a=0 {w:.4}"
    );
    tally.assert();
}

/// Bayesian linear-additive response curve. The credible band holds the
/// empirical covariate distribution fixed (its stated assumption), so its
/// target is the sample-covariate curve `1 + 2a + 0.8·z̄`. The same band
/// scored against the population curve `1 + 2a` is printed as a probe.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn response_curve_dag_bayesian_pointwise_nominal_90_coverage() {
    let mut tallies: Vec<CoverageTally> = GRID
        .iter()
        .map(|a| CoverageTally::new(format!("response_curve_dag_bayesian_pointwise_a={a}"), LEVEL))
        .collect();
    let mut population = [0u32; 5];
    let mut scored = 0u32;
    for rep in 0..u64::from(n_sim()) {
        let seed = 23_000 + rep;
        let (data, z_bar) = response_data(500, seed);
        let result = run_response(
            data,
            response_dag(),
            curve_query(),
            bayes(),
            Some(response_options(None)),
            None,
            seed,
        );
        let bands = result.as_ref().and_then(band);
        if bands.is_some() {
            scored += 1;
        }
        for (j, tally) in tallies.iter_mut().enumerate() {
            match &bands {
                Some((_, lower, upper)) => {
                    tally.record(Some((lower[j], upper[j])), 1.0 + 2.0 * GRID[j] + 0.8 * z_bar);
                    let truth = 1.0 + 2.0 * GRID[j];
                    population[j] += u32::from(lower[j] <= truth && truth <= upper[j]);
                }
                None => tally.skip(),
            }
        }
    }
    for (j, a) in GRID.iter().enumerate() {
        report_probe(
            &format!("response_curve_dag_bayesian_population_target_a={a}"),
            population[j],
            scored,
            "target 1 + 2a (population covariate law)",
        );
    }
    for tally in &tallies {
        tally.assert();
    }
}

/// Additive-GAM g-computation level `E[Y | do(T = 1)] = 3`; its SE includes the
/// covariate-average influence, so the target is the population level.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn intervention_response_dag_frequentist_nominal_90_coverage() {
    let mut tally = CoverageTally::new("intervention_response_dag_frequentist", LEVEL);
    for rep in 0..u64::from(n_sim()) {
        let seed = 24_000 + rep;
        let (data, _) = response_data(500, seed);
        let result = run_response(
            data,
            response_dag(),
            level_query(1.0),
            InferenceMode::Frequentist,
            Some(response_options(None)),
            None,
            seed,
        );
        match result {
            Some(result) => tally.record(scalar_interval(&result), 3.0),
            None => tally.skip(),
        }
    }
    tally.assert();
}

/// Bayesian level at `do(T = 1)`: sample-covariate target `3 + 0.8·z̄` (gated),
/// population target `3` (probe).
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn intervention_response_dag_bayesian_nominal_90_coverage() {
    let mut tally = CoverageTally::new("intervention_response_dag_bayesian", LEVEL);
    let (mut population, mut scored) = (0u32, 0u32);
    for rep in 0..u64::from(n_sim()) {
        let seed = 25_000 + rep;
        let (data, z_bar) = response_data(500, seed);
        let result = run_response(
            data,
            response_dag(),
            level_query(1.0),
            bayes(),
            Some(response_options(None)),
            None,
            seed,
        );
        let Some(result) = result else {
            tally.skip();
            continue;
        };
        let interval = scalar_interval(&result);
        tally.record(interval, 3.0 + 0.8 * z_bar);
        if let Some((lo, hi)) = interval {
            scored += 1;
            population += u32::from(lo <= 3.0 && 3.0 <= hi);
        }
    }
    report_probe(
        "intervention_response_dag_bayesian_population_target",
        population,
        scored,
        "target 3 (population covariate law)",
    );
    tally.assert();
}

// ------------------------------------------------------------ cell.aipw

/// Joint binary treatments drawn from a multinomial logit, so the `cell.aipw`
/// four-cell assignment model is correctly specified (columns `t1, t2, y, z`):
/// `z ~ N(0,1)`, `P(cell = c | z) ∝ exp(α_c + β_c z)` over `(t1,t2) ∈ {00,01,10,11}`
/// with `α = (0, −0.2, 0.1, −0.3)`, `β = (0, 0.5, −0.4, 0.8)`;
/// `y = 1 + t1 + 0.5 t2 + 0.7 t1·t2 + z + 0.2 z·t1 + e`. The per-cell outcome
/// regressions are linear in `z`, and `E[Y | do(t1=1, t2=1)] = 3.2`.
fn cell_data(n: usize, seed: u64) -> TabularData {
    const ALPHA: [f64; 4] = [0.0, -0.2, 0.1, -0.3];
    const BETA: [f64; 4] = [0.0, 0.5, -0.4, 0.8];
    let mut g = gaussian(seed);
    let mut u = uniform(seed);
    let (mut t1, mut t2, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        z[i] = g();
        let weights: Vec<f64> = (0..4).map(|c| (ALPHA[c] + BETA[c] * z[i]).exp()).collect();
        let total: f64 = weights.iter().sum();
        let draw = u() * total;
        let mut acc = 0.0;
        let mut cell = 3;
        for (c, w) in weights.iter().enumerate() {
            acc += w;
            if draw < acc {
                cell = c;
                break;
            }
        }
        // cell index = 2·t1 + t2
        t1[i] = f64::from(u8::from(cell >= 2));
        t2[i] = f64::from(u8::from(cell % 2 == 1));
        y[i] = 1.0 + t1[i] + 0.5 * t2[i] + 0.7 * t1[i] * t2[i] + z[i] + 0.2 * z[i] * t1[i] + g();
    }
    table(&[("t1", &t1), ("t2", &t2), ("y", &y), ("z", &z)])
}

/// `cell.aipw` publishes a 95% normal interval for the requested cell mean.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn intervention_response_cell_aipw_nominal_95_coverage() {
    let graph = dag(4, &[(3, 0), (3, 1), (3, 2), (0, 1), (0, 2), (1, 2)]);
    let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: v(2),
        interventions: Arc::from([
            Intervention::set(v(0), Value::f64(1.0)),
            Intervention::set(v(1), Value::f64(1.0)),
        ]),
    });
    let mut tally = CoverageTally::new("intervention_response_cell_aipw", 0.95);
    for rep in 0..u64::from(n_sim()) {
        let seed = 26_000 + rep;
        let result = run_response(
            cell_data(800, seed),
            graph.clone(),
            query.clone(),
            InferenceMode::Frequentist,
            None,
            Some(EstimatorId::CellAipw),
            seed,
        );
        match result {
            Some(result) => {
                if rep == 0 {
                    assert_eq!(result.logical_plan.estimator.as_deref(), Some("cell.aipw"));
                }
                tally.record(scalar_interval(&result), 3.2);
            }
            None => tally.skip(),
        }
    }
    tally.assert();
}

// ------------------------------------------------ CPDAG class-aware response

/// `z — t`, `z -> y`, `t -> y`; data from `z -> t`: `z ~ N(0,1)`,
/// `t = 0.8 z + e`, `y = t + z + e` (columns `t, y, z`, plus an independent
/// `w ~ N(0,1)` column for the agreeing-completion graph).
///
/// Completion `z -> t` adjusts `{z}`: `E[Y | do(t=a)] = a`. Completion
/// `t -> z` adjusts nothing: its estimator's probability limit is
/// `E[Y | t = a] = (1 + κ) a` with `κ = 0.8 / 1.64`.
fn class_response_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let (mut t, mut y, mut z, mut w) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        z[i] = g();
        w[i] = g();
        t[i] = 0.8 * z[i] + g();
        y[i] = t[i] + z[i] + g();
    }
    table(&[("t", &t), ("y", &y), ("z", &z), ("w", &w)])
}

/// Two identified completions that disagree (see [`class_response_data`]).
fn disagreeing_cpdag() -> Cpdag {
    let mut g = Cpdag::with_variables(4);
    g.insert_directed(d(2), d(1)).unwrap();
    g.insert_directed(d(0), d(1)).unwrap();
    g.insert_undirected(d(2), d(0)).unwrap();
    g
}

/// `z -> t`, `z -> y`, `t -> y`, `w — z`: both orientations of `w — z` are
/// completions and both adjust `{z}`, so they agree and the class response is
/// point identified at `E[Y | do(t = a)] = a`, published with the joint
/// influence-function band over the two completions.
fn agreeing_cpdag() -> Cpdag {
    let mut g = Cpdag::with_variables(4);
    g.insert_directed(d(2), d(0)).unwrap();
    g.insert_directed(d(2), d(1)).unwrap();
    g.insert_directed(d(0), d(1)).unwrap();
    g.insert_undirected(d(3), d(2)).unwrap();
    g
}

/// Disagreeing completions publish the completion identified set `[a, (1+κ)a]`
/// with no band, and no stray scalar SE for an unreported mixture.
#[test]
fn class_response_cpdag_disagreeing_completions_publish_unbanded_identified_set() {
    for query in [level_query(1.0), curve_query()] {
        let result = run_response(
            class_response_data(500, 7),
            disagreeing_cpdag(),
            query,
            InferenceMode::Frequentist,
            Some(response_options(None)),
            None,
            7,
        )
        .expect("class response runs");
        let response = result.response.as_ref().unwrap();
        assert!(
            matches!(
                response.estimate,
                ResponseIdentification::PartiallyIdentified(ResponseValue::Envelope(_))
            ),
            "{:?}",
            response.estimate
        );
        assert!(matches!(response.uncertainty, ResponseUncertainty::None));
        assert!(!result.estimate.ate.is_finite());
        assert!(
            !result.estimate.se_analytic.is_finite(),
            "no SE may be attached to an unreported mixture: se={}",
            result.estimate.se_analytic
        );
        assert!(result.estimate.joint_covariance.is_none());
        assert!(
            result
                .diagnostics
                .iter()
                .any(|d| d.code.as_ref() == "estimate.envelope.response_identified_set_unbanded")
        );
        let mixture = result.structural_response.as_ref().unwrap();
        assert_eq!(mixture.atoms.iter().filter(|atom| atom.value.is_some()).count(), 2);
    }
}

/// Joint-IF scalar interval of the class-aware level `E[Y | do(t = 1)] = 1`
/// over two agreeing completions.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn class_response_cpdag_intervention_joint_if_nominal_90_coverage() {
    let mut tally = CoverageTally::new("class_response_cpdag_intervention_joint_if", LEVEL);
    for rep in 0..u64::from(n_sim()) {
        let seed = 27_000 + rep;
        let result = run_response(
            class_response_data(500, seed),
            agreeing_cpdag(),
            level_query(1.0),
            InferenceMode::Frequentist,
            Some(response_options(None)),
            None,
            seed,
        );
        let Some(result) = result else {
            tally.skip();
            continue;
        };
        if rep == 0 {
            let mixture = result.structural_response.as_ref().expect("structural response");
            assert_eq!(
                mixture.atoms.iter().filter(|atom| atom.value.is_some()).count(),
                2,
                "two identified completions must contribute"
            );
        }
        tally.record(scalar_interval(&result), 1.0);
    }
    tally.assert();
}

/// Joint-IF pointwise band of the class-aware Kennedy curve `m(a) = a` over two
/// agreeing completions.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn class_response_cpdag_curve_joint_if_pointwise_nominal_90_coverage() {
    let mut tallies: Vec<CoverageTally> = GRID
        .iter()
        .map(|a| CoverageTally::new(format!("class_response_cpdag_curve_joint_if_a={a}"), LEVEL))
        .collect();
    for rep in 0..u64::from(n_sim()) {
        let seed = 28_000 + rep;
        let result = run_response(
            class_response_data(500, seed),
            agreeing_cpdag(),
            curve_query(),
            InferenceMode::Frequentist,
            Some(response_options(None)),
            None,
            seed,
        );
        let bands = result.as_ref().and_then(band);
        for (j, tally) in tallies.iter_mut().enumerate() {
            match &bands {
                Some((_, lower, upper)) => tally.record(Some((lower[j], upper[j])), GRID[j]),
                None => tally.skip(),
            }
        }
    }
    for tally in &tallies {
        tally.assert();
    }
}

// ======================================================================
// Static mediation
// ======================================================================

/// Columns `t, m, y, x`: `x ~ N(0,1)`, `t = 0.5x + e`, `m = 0.6t + γ·x + 0.8e`,
/// `y = 0.4t + 0.5m + 0.5x + e`. The graph has `x -> t`, `x -> y`, `t -> m`,
/// `t -> y`, `m -> y`, plus `x -> m` when `γ ≠ 0` (a mediator–outcome
/// confounder that is not in the treatment's back-door set). NDE = 0.4,
/// NIE = 0.6·0.5 = 0.3 for a unit contrast.
fn mediation_data(n: usize, gamma: f64, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let (mut t, mut m, mut y, mut x) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        x[i] = g();
        t[i] = 0.5 * x[i] + g();
        m[i] = 0.6 * t[i] + gamma * x[i] + 0.8 * g();
        y[i] = 0.4 * t[i] + 0.5 * m[i] + 0.5 * x[i] + g();
    }
    table(&[("t", &t), ("m", &m), ("y", &y), ("x", &x)])
}

fn mediation_graph(confounded: bool) -> Dag {
    let mut edges = vec![(3, 0), (3, 2), (0, 1), (0, 2), (1, 2)];
    if confounded {
        edges.push((3, 1));
    }
    dag(4, &edges)
}

fn mediation_query(contrast: MediationContrast) -> CausalQuery {
    CausalQuery::Mediation(MediationQuery::binary(v(0), v(2), Arc::from([v(1)]), contrast))
}

fn run_mediation(
    gamma: f64,
    contrast: MediationContrast,
    inference: InferenceMode,
    seed: u64,
) -> Option<StudyResult> {
    let bayesian = matches!(inference, InferenceMode::Bayesian(_));
    Study::tabular(mediation_data(400, gamma, seed))
        .graph(mediation_graph(gamma != 0.0))
        .query(mediation_query(contrast))
        .inference(inference)
        .refute(RefuteSuite::None)
        // The Study default, so the gate calibrates the interval users get.
        .bootstrap_replicates(if bayesian { 0 } else { 50 })
        .build()
        .ok()?
        .run(&ExecutionContext::for_tests(seed))
        .ok()
}

fn mediation_interval(result: &StudyResult, bayesian: bool) -> Option<(f64, f64)> {
    if bayesian {
        posterior_interval(result)
    } else {
        normal_interval(result.estimate.ate, result.estimate.se_bootstrap, Z90)
    }
}

fn mediation_coverage(
    name: &str,
    contrast: MediationContrast,
    bayesian: bool,
    truth: f64,
    seed: u64,
) {
    let mut tally = CoverageTally::new(name, LEVEL);
    let mut points = Vec::new();
    for rep in 0..u64::from(n_sim()) {
        let inference = if bayesian { bayes() } else { InferenceMode::Frequentist };
        let Some(result) = run_mediation(0.0, contrast, inference, seed + rep) else {
            tally.skip();
            continue;
        };
        if !bayesian && rep == 0 {
            assert_eq!(result.estimate.bootstrap_replicates_ok, Some(200));
            assert_eq!(result.estimate.bootstrap_replicates_failed, Some(0));
        }
        points.push(result.estimate.ate);
        tally.record(mediation_interval(&result, bayesian), truth);
    }
    let (mean, sd) = mean_sd(&points);
    eprintln!("calibration {name}: mean point={mean:.4} truth={truth} empirical_sd={sd:.4}");
    tally.assert();
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn mediation_nde_frequentist_nominal_90_coverage() {
    mediation_coverage(
        "mediation_nde_frequentist",
        MediationContrast::NaturalDirect,
        false,
        0.4,
        29_000,
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn mediation_nie_frequentist_nominal_90_coverage() {
    mediation_coverage(
        "mediation_nie_frequentist",
        MediationContrast::NaturalIndirect,
        false,
        0.3,
        29_500,
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn mediation_nde_bayesian_nominal_90_coverage() {
    mediation_coverage(
        "mediation_nde_bayesian",
        MediationContrast::NaturalDirect,
        true,
        0.4,
        30_000,
    );
}

/// Product-of-Gaussians posterior `a·b` (the highest-risk Bayesian mediation cell).
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn mediation_nie_bayesian_nominal_90_coverage() {
    mediation_coverage(
        "mediation_nie_bayesian",
        MediationContrast::NaturalIndirect,
        true,
        0.3,
        30_500,
    );
}

/// Mediator–outcome confounder `x -> m` (γ = 0.7) that is in the graph but not
/// in the treatment's back-door set. The static linear estimator regresses every
/// node on its graph parents, so `y` is adjusted for `x`; this probe records the
/// point bias and coverage on that law (identification work is WP-F2's).
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn mediation_confounded_mediator_probe() {
    for (contrast, truth, label) in [
        (MediationContrast::NaturalDirect, 0.4, "nde"),
        (MediationContrast::NaturalIndirect, 0.3, "nie"),
    ] {
        for bayesian in [false, true] {
            let (mut covered, mut scored) = (0u32, 0u32);
            let mut points = Vec::new();
            for rep in 0..u64::from(n_sim()) {
                let inference = if bayesian { bayes() } else { InferenceMode::Frequentist };
                let Some(result) = run_mediation(0.7, contrast, inference, 31_000 + rep) else {
                    continue;
                };
                points.push(result.estimate.ate);
                scored += 1;
                if let Some((lo, hi)) = mediation_interval(&result, bayesian) {
                    covered += u32::from(lo <= truth && truth <= hi);
                }
            }
            let (mean, sd) = mean_sd(&points);
            report_probe(
                &format!(
                    "mediation_confounded_mediator_{label}_{}",
                    if bayesian { "bayesian" } else { "frequentist" }
                ),
                covered,
                scored,
                &format!("mean point={mean:.4} truth={truth} bias={:.4} sd={sd:.4}", mean - truth),
            );
        }
    }
}

// ======================================================================
// Path-specific effect (discrete functional.effect)
// ======================================================================

/// Binary chain `t -> m -> y` (columns `t, m, y`): `t ~ Bern(1/2)`,
/// `m | t ~ Bern(0.3 + 0.4t)`, `y | m ~ Bern(0.2 + 0.5m)`. The only directed
/// path runs through `m`, so the path-specific effect is `0.4·0.5 = 0.2`.
fn path_data(n: usize, seed: u64) -> TabularData {
    let mut u = uniform(seed);
    let (mut t, mut m, mut y) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        t[i] = bernoulli(&mut u, 0.5);
        m[i] = bernoulli(&mut u, 0.3 + 0.4 * t[i]);
        y[i] = bernoulli(&mut u, 0.2 + 0.5 * m[i]);
    }
    table(&[("t", &t), ("m", &m), ("y", &y)])
}

fn run_path(
    data: TabularData,
    graph: Dag,
    inference: InferenceMode,
    seed: u64,
) -> Result<StudyResult, String> {
    let bayesian = matches!(inference, InferenceMode::Bayesian(_));
    let query = PathSpecificEffectQuery::binary(v(0), v(2)).with_path_nodes([v(1)]);
    Study::tabular(data)
        .graph(graph)
        .query(CausalQuery::PathSpecific(query))
        .inference(inference)
        .refute(RefuteSuite::None)
        // The Study default, so the gate calibrates the interval users get.
        .bootstrap_replicates(if bayesian { 0 } else { 50 })
        .build()
        .map_err(|e| e.to_string())?
        .run(&ExecutionContext::for_tests(seed))
        .map_err(|e| e.to_string())
}

/// Two-path law with a confounder (columns `t, m, y, c`), the SCM of
/// `conformance/estimate/path_specific_edge_gformula`: `c ~ Bern(0.4)`,
/// `t | c ~ Bern(0.3 + 0.4c)`, `m | t, c ~ Bern(0.2 + 0.4t + 0.2c)`,
/// `y | t, m, c ~ Bern(0.1 + 0.3m + 0.1t + 0.2mt + 0.2c)`. The path through `m`
/// has effect `E[Y(0, M(1))] − E[Y(0, M(0))] = 0.4·0.3 = 0.12`; the total effect
/// is 0.356 and the natural direct effect 0.156, so an evaluator that binds one
/// treatment level everywhere cannot cover.
fn two_path_data(n: usize, seed: u64) -> TabularData {
    let mut u = uniform(seed);
    let (mut t, mut m, mut y, mut c) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        c[i] = bernoulli(&mut u, 0.4);
        t[i] = bernoulli(&mut u, 0.3 + 0.4 * c[i]);
        m[i] = bernoulli(&mut u, 0.2 + 0.4 * t[i] + 0.2 * c[i]);
        y[i] = bernoulli(&mut u, 0.1 + 0.3 * m[i] + 0.1 * t[i] + 0.2 * m[i] * t[i] + 0.2 * c[i]);
    }
    table(&[("t", &t), ("m", &m), ("y", &y), ("c", &c)])
}

fn two_path_graph() -> Dag {
    dag(4, &[(3, 0), (3, 1), (3, 2), (0, 1), (0, 2), (1, 2)])
}

fn path_coverage(
    name: &str,
    bayesian: bool,
    seed: u64,
    truth: f64,
    sample: impl Fn(u64) -> (TabularData, Dag),
) {
    let mut tally = CoverageTally::new(name, LEVEL);
    for rep in 0..u64::from(n_sim()) {
        let inference = if bayesian { bayes() } else { InferenceMode::Frequentist };
        let (data, graph) = sample(seed + rep);
        let Ok(result) = run_path(data, graph, inference, seed + rep) else {
            tally.skip();
            continue;
        };
        let interval = if bayesian {
            posterior_interval(&result)
        } else {
            normal_interval(result.estimate.ate, result.estimate.se_bootstrap, Z90)
        };
        tally.record(interval, truth);
    }
    tally.assert();
}

fn chain_sample(seed: u64) -> (TabularData, Dag) {
    (path_data(500, seed), dag(3, &[(0, 1), (1, 2)]))
}

fn two_path_sample(seed: u64) -> (TabularData, Dag) {
    (two_path_data(1_000, seed), two_path_graph())
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn path_specific_frequentist_nominal_90_coverage() {
    path_coverage("path_specific_frequentist", false, 32_000, 0.2, chain_sample);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn path_specific_bayesian_nominal_90_coverage() {
    path_coverage("path_specific_bayesian", true, 32_500, 0.2, chain_sample);
}

/// Edge g-formula: a direct `t -> y` path competes with the path through `m`.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn path_specific_two_path_frequentist_nominal_90_coverage() {
    path_coverage("path_specific_two_path_frequentist", false, 33_000, 0.12, two_path_sample);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn path_specific_two_path_bayesian_nominal_90_coverage() {
    path_coverage("path_specific_two_path_bayesian", true, 33_500, 0.12, two_path_sample);
}

// ======================================================================
// Interventional distribution near 0 and 1
// ======================================================================

/// Columns `t, y, z` (binary): `z ~ Bern(1/2)`, `t | z ~ Bern(0.3 + 0.4z)`,
/// `y | t, z ~ Bern(p(t, z))` with `p(1, z) = base + 0.03z` and
/// `p(0, z) = 0.5`. `P(Y = 1 | do(t = 1)) = base + 0.015`.
fn distribution_data(n: usize, base: f64, seed: u64) -> TabularData {
    let mut u = uniform(seed);
    let (mut t, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        z[i] = bernoulli(&mut u, 0.5);
        t[i] = bernoulli(&mut u, 0.3 + 0.4 * z[i]);
        let p = if t[i] > 0.5 { base + 0.03 * z[i] } else { 0.5 };
        y[i] = bernoulli(&mut u, p);
    }
    table(&[("t", &t), ("y", &y), ("z", &z)])
}

fn run_distribution(data: TabularData, inference: InferenceMode, seed: u64) -> Option<StudyResult> {
    let bayesian = matches!(inference, InferenceMode::Bayesian(_));
    let query =
        InterventionalDistributionQuery::new(v(1), [Intervention::set(v(0), Value::f64(1.0))]);
    Study::tabular(data)
        .graph(dag(3, &[(2, 0), (2, 1), (0, 1)]))
        .query(CausalQuery::Distribution(query))
        .inference(inference)
        .refute(RefuteSuite::None)
        // The Study default, so the gate calibrates the interval users get.
        .bootstrap_replicates(if bayesian { 0 } else { 50 })
        .build()
        .ok()?
        .run(&ExecutionContext::for_tests(seed))
        .ok()
}

/// Bayesian `P(Y = 1 | do(t = 1))` credible interval from the atom draws.
fn distribution_bayesian(name: &str, base: f64, seed: u64) {
    let truth = base + 0.015;
    let mut tally = CoverageTally::new(name, LEVEL);
    for rep in 0..u64::from(n_sim()) {
        let Some(result) =
            run_distribution(distribution_data(600, base, seed + rep), bayes(), seed + rep)
        else {
            tally.skip();
            continue;
        };
        let dist = result.distribution.as_ref().expect("distribution payload");
        let posterior = result.posterior.as_ref().expect("posterior");
        let offset = usize::from(posterior.effect_column().is_some());
        let atom = dist
            .atoms
            .iter()
            .position(|a| a.outcomes[0].1.as_f64() == Some(1.0))
            .expect("Y = 1 atom");
        let draws = posterior.draws.column(offset + atom).unwrap();
        assert!(draws.iter().all(|p| (0.0..=1.0).contains(p)), "atom draws must stay in [0,1]");
        tally.record(quantile_interval(draws, LEVEL), truth);
    }
    tally.assert();
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn interventional_distribution_bayesian_near_one_nominal_90_coverage() {
    distribution_bayesian("interventional_distribution_bayesian_near_one", 0.95, 33_000);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn interventional_distribution_bayesian_near_zero_nominal_90_coverage() {
    distribution_bayesian("interventional_distribution_bayesian_near_zero", 0.02, 33_500);
}

/// Frequentist published interval for `P(Y = 1 | do(t = 1))`: the logit-scale
/// delta-method interval from the per-atom bootstrap SE, at the estimator's
/// published level (95%). Every published atom bound must lie in `[0, 1]`, the
/// headline mean interval must be the `Y = 1` atom's interval, and the atom SE
/// of `Y = 1` must reproduce the scalar bootstrap SE of the mean.
fn distribution_frequentist(name: &str, base: f64, seed: u64) {
    const PUBLISHED_LEVEL: f64 = 0.95;
    let truth = base + 0.015;
    let mut tally = CoverageTally::new(name, PUBLISHED_LEVEL);
    for rep in 0..u64::from(n_sim()) {
        let Some(result) = run_distribution(
            distribution_data(600, base, seed + rep),
            InferenceMode::Frequentist,
            seed + rep,
        ) else {
            tally.skip();
            continue;
        };
        let dist = result.distribution.as_ref().expect("distribution payload");
        assert_eq!(dist.atom_uncertainty.len(), dist.atoms.len(), "one interval per atom");
        for (atom, u) in dist.atoms.iter().zip(dist.atom_uncertainty.iter()) {
            if let Some((lo, hi)) = u.interval.bounds() {
                assert!(
                    (0.0..=1.0).contains(&lo) && (0.0..=1.0).contains(&hi) && lo <= hi,
                    "published bounds [{lo}, {hi}] must lie in [0, 1]"
                );
                assert!(lo <= atom.probability && atom.probability <= hi, "p̂ inside its interval");
            }
        }
        let one = dist
            .atoms
            .iter()
            .position(|a| a.outcomes[0].1.as_f64() == Some(1.0))
            .expect("Y = 1 atom");
        let atom_interval = dist.atom_uncertainty[one].interval;
        assert_eq!(dist.mean_interval, Some(atom_interval), "binary mean uses the Y = 1 atom");
        if let (Some(atom_se), Some(mean_se)) =
            (dist.atom_uncertainty[one].se_bootstrap, result.estimate.se_bootstrap)
        {
            assert!((atom_se - mean_se).abs() <= 1e-12 * mean_se.max(1.0), "same replicates");
        }
        tally.record(atom_interval.bounds(), truth);
    }
    tally.assert();
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn interventional_distribution_frequentist_near_one_nominal_95_coverage() {
    distribution_frequentist("interventional_distribution_frequentist_near_one", 0.95, 34_000);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn interventional_distribution_frequentist_near_zero_nominal_95_coverage() {
    distribution_frequentist("interventional_distribution_frequentist_near_zero", 0.02, 34_500);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn interventional_distribution_frequentist_interior_nominal_95_coverage() {
    distribution_frequentist("interventional_distribution_frequentist_interior", 0.45, 35_500);
}

// ======================================================================
// Counterfactual (Bayesian mechanism-refit posterior of the mean ITE)
// ======================================================================

/// Linear additive SCM (columns `a, m, y`): `a ~ N(0,1)`, `m = 2a + e`,
/// `y = 3a + 4m + e`. Every unit's counterfactual contrast for
/// `do(a = 1)` vs `do(a = 0)` is `3 + 4·2 = 11`, so the mean ITE over the
/// observed units — the quantity the Bayesian interval claims — is 11.
fn counterfactual_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let (mut a, mut m, mut y) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        a[i] = g();
        m[i] = 2.0 * a[i] + g();
        y[i] = 3.0 * a[i] + 4.0 * m[i] + g();
    }
    table(&[("a", &a), ("m", &m), ("y", &y)])
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn counterfactual_bayesian_mean_ite_nominal_90_coverage() {
    let mut tally = CoverageTally::new("counterfactual_bayesian_mean_ite", LEVEL);
    let graph = dag(3, &[(0, 1), (0, 2), (1, 2)]);
    for rep in 0..u64::from(n_sim()) {
        let seed = 35_000 + rep;
        let query =
            CounterfactualQuery::new(v(2), Arc::from([Intervention::set(v(0), Value::f64(1.0))]))
                .with_control_level(0.0);
        let result = Study::tabular(counterfactual_data(300, seed))
            .graph(graph.clone())
            .query(CausalQuery::Counterfactual(query))
            .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(200)))
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .ok()
            .and_then(|s| s.run(&ExecutionContext::for_tests(seed)).ok());
        let Some(result) = result else {
            tally.skip();
            continue;
        };
        if rep == 0 {
            assert!(
                result.diagnostics.iter().any(|d| d.code.as_ref() == "gcm.counterfactual.bayesian"
                    && d.message.contains("mean_ite")),
                "the Bayesian counterfactual must say which quantity its interval covers"
            );
        }
        tally.record(posterior_interval(&result), 11.0);
    }
    tally.assert();
}

// ======================================================================
// R-17: bayesian.gcomp AverageEffect under a misspecified outcome
// ======================================================================

/// Columns `t, y, z`: `z ~ N(0,1)`, `t ~ Bern(σ(−0.8 + z))`,
/// `y = 2t + z + q·z² + h·t·z + e`. `ATE = 2 + h·E[z] = 2`. The fitted model is
/// `y ~ 1 + t + z` (linear Gaussian), misspecified whenever `q` or `h` ≠ 0.
fn misspecified_data(n: usize, q: f64, h: f64, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let mut u = uniform(seed);
    let (mut t, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        z[i] = g();
        t[i] = bernoulli(&mut u, sigmoid(-0.8 + z[i]));
        y[i] = 2.0 * t[i] + z[i] + q * z[i] * z[i] + h * t[i] * z[i] + g();
    }
    table(&[("t", &t), ("y", &y), ("z", &z)])
}

/// Measured coverage of the Bayesian gcomp credible interval on the correctly
/// specified law (gated) and on quadratic / interaction outcomes (probes).
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn bayesian_gcomp_misspecification_probe() {
    let graph = dag(3, &[(2, 0), (2, 1), (0, 1)]);
    let query = AverageEffectQuery::binary_ate(v(0), v(1));
    let mut gated = CoverageTally::new("bayesian_gcomp_ate_linear_outcome", LEVEL);
    for (q, h, label) in [(0.0, 0.0, "linear"), (0.6, 0.0, "quadratic"), (0.0, 1.0, "interaction")]
    {
        let (mut covered, mut scored) = (0u32, 0u32);
        let mut points = Vec::new();
        for rep in 0..u64::from(n_sim()) {
            let seed = 36_000 + rep;
            let result = Study::tabular(misspecified_data(500, q, h, seed))
                .graph(graph.clone())
                .query(query.clone())
                .inference(bayes())
                .refute(RefuteSuite::None)
                .bootstrap_replicates(0)
                .build()
                .unwrap()
                .run(&ExecutionContext::for_tests(seed))
                .unwrap();
            if rep == 0 {
                assert!(
                    result.estimate.assumptions.entries.iter().any(|a| matches!(
                        &a.assumption,
                        antecedent_core::Assumption::ParametricRestriction(p)
                            if p.id.as_ref() == "bayesian.gcomp.linear_gaussian_outcome"
                    )),
                    "bayesian.gcomp must declare its linear-Gaussian outcome model"
                );
            }
            let interval = posterior_interval(&result);
            points.push(result.estimate.ate);
            scored += 1;
            if let Some((lo, hi)) = interval {
                covered += u32::from(lo <= 2.0 && 2.0 <= hi);
            }
            if label == "linear" {
                gated.record(interval, 2.0);
            }
        }
        let (mean, sd) = mean_sd(&points);
        report_probe(
            &format!("bayesian_gcomp_ate_{label}_outcome"),
            covered,
            scored,
            &format!("mean point={mean:.4} truth=2 bias={:.4} sd={sd:.4}", mean - 2.0),
        );
    }
    gated.assert();
}
