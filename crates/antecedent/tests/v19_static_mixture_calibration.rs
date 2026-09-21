//! Coverage of static graph-posterior mixture intervals (R-19, R-11 / C-1).
//!
//! A hand-built `GraphPosterior` has two identified DAG atoms whose effects
//! differ plus one unidentified atom:
//!
//! | atom | structure | estimator target (probability limit) | weight |
//! | --- | --- | --- | --- |
//! | A | `T -> Y` (no adjustment) | `θ_A = 3` | 0.5 |
//! | B | `Z -> T`, `Z -> Y`, `T -> Y` (adjust `Z`) | `θ_B = 2` | 0.3 |
//! | C | `Y -> T` (no admissible adjustment) | unidentified | 0.2 |
//!
//! The DGP is `Z ~ Bern(1/2)`, `T | Z ~ Bern(1/4 + Z/2)`, `Y = 2T + 2Z + ε`
//! (`ConditionalEffect` adds `W ~ N(0, 1)` and `W + T·W/2` to `Y`, so both CATE
//! atoms keep the same 3 / 2 targets at `E[W] = 0`).
//!
//! * **Frequentist** on the disagreeing fixture withholds the scalar under
//!   `GraphDependentAtoms`. Joint-IF SE and scalar calibration use a
//!   same-estimand two-atom posterior (both unadjusted T→Y, truth 3).
//! * **Bayesian** reports the identified-atom BMA `P(τ | identified)`: each draw
//!   first samples a graph by its identified weight, then that graph's effect
//!   posterior (`aggregate_effect_envelope`). Its interval is a distribution over
//!   graph-specific effects, not an interval for the aggregate, and it does not
//!   shrink toward 2.625 with `n`. The matching calibration target is therefore
//!   the effect of a graph drawn from those weights: each replicate draws
//!   `g* ~ (0.625, 0.375)` independently of the data and scores the equal-tailed
//!   BMA interval against `θ_{g*}`. Scoring it against the aggregate 2.625
//!   instead would over-cover by construction.
//!
//! Ignored tests run via `scripts/gate_calibration.sh`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

mod common;

use antecedent::discovery::GraphPosterior;
use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ConditionalEffectQuery, ExecutionContext, VariableId,
};
use antecedent_data::TabularData;
use antecedent_discovery::set_edge;
use antecedent_prob::InferenceDiagnostics;
use common::calibration::{
    CoverageTally, REPORTED_LEVEL, RecordKey, Z90, Z95, gaussian, grid_n, n_sim, normal_interval,
    quantile_interval, stream_seed,
};
use common::calibration_bind::bind_all;

const N: usize = 400;
const LEVEL: f64 = 0.9;
const DRAWS: usize = 400;
const WEIGHTS: [f64; 3] = [0.5, 0.3, 0.2];
const THETA: [f64; 2] = [3.0, 2.0];
const TRUTH_SAME_ESTIMAND: f64 = 3.0;

/// Replicate seed of the frequentist and Bayesian sweeps: `BASE + r · STRIDE`.
/// The stride must stay odd and bigger than 1 — see [`uniform`].
const SEED_STRIDE: u64 = 7_919;
const FREQUENTIST_SEED_BASE: u64 = 0x19C0_0000;
const BAYESIAN_SEED_BASE: u64 = 0x19B0_0000;

/// Deterministic U(0, 1) stream (same LCG family as the harness generator).
///
/// This suite's uniform stream: the harness's one LCG, conditioned the way the
/// recorded coverage of these cells was measured.
///
/// `seed | 1` is the unscrambled conditioning `calibration::gaussian` warns
/// about — two seeds differing only in bit 0 share a stream, which would
/// silently halve the replicate count. The conditioning stays pinned because
/// changing it would regenerate every replicate and move the pinned rates;
/// [`replicate_seeds_stay_distinct_under_the_pinned_conditioning`] checks that
/// the seeds this suite strides through cannot collide, so the hazard cannot
/// arrive unnoticed.
fn uniform(seed: u64) -> impl FnMut() -> f64 {
    common::calibration::uniform_from_state(seed | 1)
}

/// Columns `[t, y, z]`, plus `w` when `modifier` is set.
fn draw_data(seed: u64, modifier: bool) -> TabularData {
    let mut unif = uniform(seed);
    let mut eps_noise = gaussian(stream_seed(seed, 0x5EED_0001));
    let mut w_noise = gaussian(stream_seed(seed, 0x5EED_0002));
    let rows = grid_n(N);
    let (mut t, mut y, mut z, mut w) = (
        Vec::with_capacity(rows),
        Vec::with_capacity(rows),
        Vec::with_capacity(rows),
        Vec::with_capacity(rows),
    );
    for _ in 0..rows {
        let zi = f64::from(u8::from(unif() < 0.5));
        let ti = f64::from(u8::from(unif() < 0.25 + 0.5 * zi));
        let wi = w_noise();
        let mut yi = 2.0 * ti + 2.0 * zi + eps_noise();
        if modifier {
            yi += wi + 0.5 * ti * wi;
        }
        t.push(ti);
        y.push(yi);
        z.push(zi);
        w.push(wi);
    }
    if modifier {
        TabularData::from_f64_columns([
            ("t", t.as_slice()),
            ("y", y.as_slice()),
            ("z", z.as_slice()),
            ("w", w.as_slice()),
        ])
        .unwrap()
    } else {
        TabularData::from_f64_columns([
            ("t", t.as_slice()),
            ("y", y.as_slice()),
            ("z", z.as_slice()),
        ])
        .unwrap()
    }
}

/// Atoms A / B / C over `[t, y, z]` (and `w -> y` in every atom when `modifier`).
fn disagreeing_mixture_posterior(modifier: bool) -> GraphPosterior {
    let n = if modifier { 4 } else { 3 };
    let base = if modifier { set_edge(0, n, 3, 1, true) } else { 0 };
    let direct = set_edge(base, n, 0, 1, true);
    let adjusted = set_edge(set_edge(set_edge(base, n, 0, 1, true), n, 2, 0, true), n, 2, 1, true);
    let unidentified = set_edge(base, n, 1, 0, true);
    let cells = n * n;
    GraphPosterior::new(
        n,
        WEIGHTS.to_vec(),
        vec![direct, adjusted, unidentified],
        vec![0.0; cells],
        vec![0.0; cells],
        1.0 / WEIGHTS.iter().map(|w| w * w).sum::<f64>(),
        InferenceDiagnostics::analytic("v19_static_mixture"),
        0,
    )
    .unwrap()
}

/// Two identified atoms share the empty adjustment set (both estimate 3).
fn same_estimand_mixture_posterior(modifier: bool) -> GraphPosterior {
    let n = if modifier { 4 } else { 3 };
    let base = if modifier { set_edge(0, n, 3, 1, true) } else { 0 };
    let direct = set_edge(base, n, 0, 1, true);
    let outcome_parent = set_edge(set_edge(base, n, 0, 1, true), n, 2, 1, true);
    let unidentified = set_edge(base, n, 1, 0, true);
    let cells = n * n;
    GraphPosterior::new(
        n,
        WEIGHTS.to_vec(),
        vec![direct, outcome_parent, unidentified],
        vec![0.0; cells],
        vec![0.0; cells],
        1.0 / WEIGHTS.iter().map(|w| w * w).sum::<f64>(),
        InferenceDiagnostics::analytic("v19_static_mixture_same_estimand"),
        0,
    )
    .unwrap()
}

fn mixture_posterior(modifier: bool) -> GraphPosterior {
    disagreeing_mixture_posterior(modifier)
}

fn query(conditional: bool) -> CausalQuery {
    let inner = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    if conditional {
        CausalQuery::ConditionalEffect(
            ConditionalEffectQuery::try_new(inner.with_effect_modifiers([VariableId::from_raw(3)]))
                .unwrap(),
        )
    } else {
        CausalQuery::AverageEffect(inner)
    }
}

fn run_with_posterior(
    conditional: bool,
    inference: InferenceMode,
    seed: u64,
    posterior: GraphPosterior,
) -> (Study, StudyResult) {
    let study = Study::tabular(draw_data(seed, conditional))
        .graph_posterior(posterior)
        .query(query(conditional))
        .inference(inference)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let result = study.run(&ExecutionContext::for_tests(seed)).unwrap();
    (study, result)
}

fn run(conditional: bool, inference: InferenceMode, seed: u64) -> (Study, StudyResult) {
    run_with_posterior(conditional, inference, seed, mixture_posterior(conditional))
}

fn assert_mixture_shape(result: &StudyResult) {
    let status =
        result.posterior.as_ref().map_or(result.identification.status, |p| p.identification);
    assert_eq!(status, antecedent_core::IdentificationStatus::GraphDependent);
    let unidentified = result.posterior.as_ref().map_or_else(
        || {
            result.diagnostics.iter().any(|d| {
                d.code.as_ref() == "estimate.graph_posterior.envelope"
                    && d.message.contains("unidentified_mass=0.2")
            })
        },
        |p| (p.unidentified_mass - WEIGHTS[2]).abs() < 1e-12,
    );
    assert!(unidentified, "the 0.2 unidentified atom must stay out of the mixture");
}

/// Gated at 90%; the runtime's reported 95% interval is scored on the same
/// replicates and recorded.
fn frequentist_coverage(conditional: bool, test: &'static str, name: &str) {
    let key = RecordKey { test, dgp: "draw_data", interval: "analytic_se" };
    let mut tally = CoverageTally::for_record(key, LEVEL);
    let mut reported = CoverageTally::for_record(key, REPORTED_LEVEL).unasserted();
    let mut points = Vec::with_capacity(n_sim() as usize);
    let mut se_sum = 0.0;
    for r in 0..n_sim() {
        let seed = FREQUENTIST_SEED_BASE + u64::from(r) * SEED_STRIDE;
        let (study, result) = run_with_posterior(
            conditional,
            InferenceMode::Frequentist,
            seed,
            same_estimand_mixture_posterior(conditional),
        );
        if r == 0 {
            assert_mixture_shape(&result);
            assert!(
                result.estimate.ate.is_finite(),
                "{name}: same-estimand fixture must publish a scalar ATE"
            );
            assert!(
                result
                    .diagnostics
                    .iter()
                    .any(|d| d.code.as_ref() == "estimate.graph_posterior.joint_if_se"),
                "{name}: multi-atom SE must come from the joint IF combiner"
            );
        }
        points.push(result.estimate.ate);
        se_sum += result.estimate.se_analytic;
        bind_all(&mut [&mut tally, &mut reported], &study, &result);
        tally.record(
            normal_interval(result.estimate.ate, Some(result.estimate.se_analytic), Z90),
            TRUTH_SAME_ESTIMAND,
        );
        reported.record(
            normal_interval(result.estimate.ate, Some(result.estimate.se_analytic), Z95),
            TRUTH_SAME_ESTIMAND,
        );
    }
    let reps = points.len() as f64;
    let mean = points.iter().sum::<f64>() / reps;
    let sd = (points.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / (reps - 1.0)).sqrt();
    eprintln!(
        "calibration {name}: mean point={mean:.4} truth={TRUTH_SAME_ESTIMAND} \
         empirical_sd={sd:.4} mean_se={:.4}",
        se_sum / reps
    );
    tally.assert();
    reported.emit();
}

/// Gated at 90%; the runtime's reported 95% interval is scored on the same
/// replicates (against the same drawn graph effect) and recorded.
fn bayesian_coverage(conditional: bool, test: &'static str) {
    let key = RecordKey { test, dgp: "draw_data", interval: "posterior_quantile" };
    let mut tally = CoverageTally::for_record(key, LEVEL);
    let mut reported = CoverageTally::for_record(key, REPORTED_LEVEL).unasserted();
    let mut graph_draw = uniform(0xB5A7_0019);
    let identified_mass = WEIGHTS[0] + WEIGHTS[1];
    for r in 0..n_sim() {
        let seed = BAYESIAN_SEED_BASE + u64::from(r) * SEED_STRIDE;
        let (study, result) = run(
            conditional,
            InferenceMode::Bayesian(
                BayesianConfig::conjugate().n_draws(DRAWS).prior_scale(1_000.0),
            ),
            seed,
        );
        if r == 0 {
            assert_mixture_shape(&result);
        }
        let truth = if graph_draw() < WEIGHTS[0] / identified_mass { THETA[0] } else { THETA[1] };
        let interval_at = |level: f64| {
            result.posterior.as_ref().and_then(|post| {
                let col = post.effect_column()?;
                let draws = post.draws.column(col).ok()?;
                quantile_interval(draws, level)
            })
        };
        bind_all(&mut [&mut tally, &mut reported], &study, &result);
        tally.record(interval_at(LEVEL), truth);
        reported.record(interval_at(REPORTED_LEVEL), truth);
    }
    tally.assert();
    reported.emit();
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn static_graph_posterior_frequentist_ate_joint_if_nominal_90_coverage() {
    frequentist_coverage(
        false,
        "static_graph_posterior_frequentist_ate_joint_if_nominal_90_coverage",
        "static_graph_posterior_frequentist_ate",
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn static_graph_posterior_frequentist_cate_joint_if_nominal_90_coverage() {
    frequentist_coverage(
        true,
        "static_graph_posterior_frequentist_cate_joint_if_nominal_90_coverage",
        "static_graph_posterior_frequentist_cate",
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn static_graph_posterior_bayesian_ate_bma_nominal_90_coverage() {
    bayesian_coverage(false, "static_graph_posterior_bayesian_ate_bma_nominal_90_coverage");
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn static_graph_posterior_bayesian_cate_bma_nominal_90_coverage() {
    bayesian_coverage(true, "static_graph_posterior_bayesian_cate_bma_nominal_90_coverage");
}

/// The pinned `seed | 1` conditioning in [`uniform`] maps the two seeds `2k`
/// and `2k + 1` onto one stream, so two replicates would silently share a
/// dataset and the effective replicate count would halve while the gate still
/// divided by `n_sim()`. The stride keeps that from happening; this asserts it
/// instead of leaving it to be true by accident.
///
/// Not `#[ignore]`: it is the guard on the conditioning, so it runs on every
/// PR, and it costs nothing — it generates no data.
#[test]
fn replicate_seeds_stay_distinct_under_the_pinned_conditioning() {
    // More than any gate run uses, so the property holds past the current N.
    const PROBE: u64 = 4_000;
    assert_eq!(SEED_STRIDE % 2, 1, "an even stride repeats one bit-0 class forever");
    for base in [FREQUENTIST_SEED_BASE, BAYESIAN_SEED_BASE] {
        let mut streams = std::collections::HashSet::with_capacity(PROBE as usize);
        for r in 0..PROBE {
            let seed = base + r * SEED_STRIDE;
            assert!(
                streams.insert(seed | 1),
                "replicates before {r} already used the stream of seed {seed:#x}: \
                 `seed | 1` collides, so two replicates share one dataset"
            );
        }
    }
}
