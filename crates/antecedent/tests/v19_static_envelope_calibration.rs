//! 1.9 repeated-sampling coverage for static envelope and tier cells (R-19).
//!
//! Every DGP is linear with Gaussian noise, so each completion's reported
//! functional has a closed-form population value (derived next to each
//! generator). The truth an interval is scored against is the population
//! value of the *reported* functional: the frozen-weight mixture over
//! identified completions, with unidentified mass excluded (it is reported,
//! never mixed in). Completions deliberately disagree, so a combiner that
//! drops between-completion covariance or scores against one completion
//! fails here.
//!
//! Ignored tests run via `scripts/gate_calibration.sh` (release build).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::similar_names
)]

mod common;

use antecedent::{BayesianConfig, EstimatorId, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ConditionalEffectQuery, ExecutionContext, VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_graph::{Cpdag, Dag, DenseNodeId, Pag, TieredBackground, WithinTier};

use common::calibration::{
    CoverageTally, Z90, gaussian, n_sim, normal_interval, quantile_interval,
};

const LEVEL: f64 = 0.9;
const DRAWS: usize = 400;

// Shared linear coefficients: t = ALPHA z + e_t, y = B t + ... ; Var(z) = Var(e_t) = 1.
const ALPHA: f64 = 0.8;
const B: f64 = 1.0;
/// Interaction of treatment and modifier in the ConditionalEffect DGPs.
const G: f64 = 0.8;
/// Population slope of `z` on `t`: Cov(z, t) / Var(t).
const KAPPA: f64 = ALPHA / (ALPHA * ALPHA + 1.0);

fn d(i: u32) -> DenseNodeId {
    DenseNodeId::from_raw(i)
}

fn uniform(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    move || {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (state >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn frequentist_interval(result: &antecedent::StudyResult) -> Option<(f64, f64)> {
    normal_interval(result.estimate.ate, Some(result.estimate.se_analytic), Z90)
}

fn bayes_interval(result: &antecedent::StudyResult) -> Option<(f64, f64)> {
    let posterior = result.posterior.as_ref()?;
    let col = posterior.effect_column()?;
    let draws = posterior.draws.column(col).ok()?;
    quantile_interval(draws, LEVEL)
}

fn bayes() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(DRAWS).prior_scale(10.0))
}

enum Structure {
    Dag(Dag),
    Cpdag(Cpdag),
    Pag(Pag),
}

fn run(
    data: TabularData,
    graph: &Structure,
    query: CausalQuery,
    inference: InferenceMode,
    seed: u64,
) -> Option<antecedent::StudyResult> {
    let builder = Study::tabular(data);
    let builder = match graph {
        Structure::Dag(g) => builder.graph(g.clone()),
        Structure::Cpdag(g) => builder.graph(g.clone()),
        Structure::Pag(g) => builder.graph(g.clone()),
    };
    builder
        .query(query)
        .inference(inference)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .ok()?
        .run(&ExecutionContext::for_tests(seed))
        .ok()
}

// ---------------------------------------------------------------- CPDAG ATE

/// `z — t`, `z -> y`, `t -> y`; data from `z -> t`:
/// `z ~ N(0,1)`, `t = 0.8 z + e`, `y = t + z + e` (columns `t, y, z`).
///
/// Completion `z -> t` adjusts `{z}`: population slope `B = 1`.
/// Completion `t -> z` adjusts nothing: slope of `y` on `t` is
/// `B + 1·KAPPA = 1.4878`. Equal weights: truth `1.2439`.
fn cpdag_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let (mut t, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        z[i] = g();
        t[i] = ALPHA * z[i] + g();
        y[i] = B * t[i] + z[i] + g();
    }
    TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())])
        .unwrap()
}

fn cpdag() -> Cpdag {
    let mut g = Cpdag::with_variables(3);
    g.insert_directed(d(2), d(1)).unwrap();
    g.insert_directed(d(0), d(1)).unwrap();
    g.insert_undirected(d(2), d(0)).unwrap();
    g
}

const CPDAG_TRUTH: f64 = 0.5 * B + 0.5 * (B + KAPPA);

fn ate_query() -> CausalQuery {
    CausalQuery::AverageEffect(AverageEffectQuery::with_levels(
        VariableId::from_raw(0),
        VariableId::from_raw(1),
        0.0,
        1.0,
    ))
}

fn run_ate_coverage(
    name: &str,
    graph: &Structure,
    generate: impl Fn(usize, u64) -> TabularData,
    n: usize,
    inference: impl Fn() -> InferenceMode,
    bayesian: bool,
    truth: f64,
    seed: u64,
) {
    let mut tally = CoverageTally::new(name, LEVEL);
    for rep in 0..u64::from(n_sim()) {
        let data = generate(n, seed + rep);
        let Some(result) = run(data, graph, ate_query(), inference(), seed + rep) else {
            tally.skip();
            continue;
        };
        let interval =
            if bayesian { bayes_interval(&result) } else { frequentist_interval(&result) };
        tally.record(interval, truth);
    }
    tally.assert();
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn static_cpdag_ate_envelope_frequentist_nominal_90_coverage() {
    run_ate_coverage(
        "static_cpdag_ate_envelope_frequentist",
        &Structure::Cpdag(cpdag()),
        cpdag_data,
        400,
        || InferenceMode::Frequentist,
        false,
        CPDAG_TRUTH,
        19_100,
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn static_cpdag_ate_envelope_bayesian_nominal_90_coverage() {
    run_ate_coverage(
        "static_cpdag_ate_envelope_bayesian",
        &Structure::Cpdag(cpdag()),
        cpdag_data,
        400,
        bayes,
        true,
        CPDAG_TRUTH,
        19_200,
    );
}

// ------------------------------------------------------------------ PAG ATE

/// The identified multi-completion PAG of
/// `conformance/estimate/pag_ate_envelope_identified`
/// (`v o-o t o-o z o-o m`, `t -> y`, `m -> y`, `x -> y`; columns
/// `t, y, z, m, v, x`) with continuous data from `z -> t -> v`, `z -> m -> y`:
/// `t = 0.8 z + e`, `v = 0.6 t + e`, `m = 0.9 z + e`, `x ~ N(0,1)`,
/// `y = t + m + 0.5 x + e`.
///
/// Four identified completions adjust `{z}` (population slope `B = 1`; `m` is
/// independent of `t` given `z`), two adjust nothing (slope
/// `B + 0.9·KAPPA = 1.4390`), one is unidentified and excluded from the
/// mixture. Truth `(4·1 + 2·1.4390)/6 = 1.1463`.
fn pag_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let cols: [Vec<f64>; 6] = std::array::from_fn(|_| vec![0.0; n]);
    let [mut t, mut y, mut z, mut m, mut v, mut x] = cols;
    for i in 0..n {
        z[i] = g();
        t[i] = ALPHA * z[i] + g();
        v[i] = 0.6 * t[i] + g();
        m[i] = 0.9 * z[i] + g();
        x[i] = g();
        y[i] = B * t[i] + m[i] + 0.5 * x[i] + g();
    }
    TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
        ("m", m.as_slice()),
        ("v", v.as_slice()),
        ("x", x.as_slice()),
    ])
    .unwrap()
}

fn pag() -> Pag {
    // t=0 y=1 z=2 m=3 v=4 x=5
    let mut g = Pag::with_variables(6);
    g.insert_circle_circle(d(4), d(0)).unwrap();
    g.insert_circle_circle(d(0), d(2)).unwrap();
    g.insert_circle_circle(d(2), d(3)).unwrap();
    g.insert_directed(d(0), d(1)).unwrap();
    g.insert_directed(d(3), d(1)).unwrap();
    g.insert_directed(d(5), d(1)).unwrap();
    g
}

const PAG_TOTAL_SHIFT: f64 = 0.9 * KAPPA;
const PAG_TRUTH: f64 = (4.0 * B + 2.0 * (B + PAG_TOTAL_SHIFT)) / 6.0;

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn static_pag_ate_envelope_frequentist_nominal_90_coverage() {
    run_ate_coverage(
        "static_pag_ate_envelope_frequentist",
        &Structure::Pag(pag()),
        pag_data,
        400,
        || InferenceMode::Frequentist,
        false,
        PAG_TRUTH,
        19_300,
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn static_pag_ate_envelope_bayesian_nominal_90_coverage() {
    run_ate_coverage(
        "static_pag_ate_envelope_bayesian",
        &Structure::Pag(pag()),
        pag_data,
        400,
        bayes,
        true,
        PAG_TRUTH,
        19_400,
    );
}

// ------------------------------------------------------- ConditionalEffect

/// ConditionalEffect DGPs add a binary modifier `x ~ Bern(p)` that affects
/// only `y`, with interaction `G·t·x`. The reported functional is
/// `β_t + β_tx·E[x]` from `y ~ 1 + t + x + t·x + Z_g`. With `x` independent of
/// `(t, z, m)`, the within-stratum projection of an omitted confounder on `t`
/// is the same in both strata, so each completion's population value is its
/// ATE slope plus `G·p`.
fn conditional_query(modifier: u32) -> CausalQuery {
    CausalQuery::ConditionalEffect(
        ConditionalEffectQuery::try_new(
            AverageEffectQuery::with_levels(
                VariableId::from_raw(0),
                VariableId::from_raw(1),
                0.0,
                1.0,
            )
            .with_effect_modifiers([VariableId::from_raw(modifier)]),
        )
        .unwrap(),
    )
}

/// Dag/Cpdag conditional data (columns `t, y, z, x`):
/// `t = 0.8 z + e`, `y = t + 0.8 t·x + z + 0.3 x + e`, `x ~ Bern(p)`.
fn conditional_dag_data(n: usize, seed: u64, p: f64) -> TabularData {
    let mut g = gaussian(seed);
    let mut u = uniform(seed ^ 0xC0FF_EE00);
    let (mut t, mut y, mut z, mut x) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        z[i] = g();
        x[i] = f64::from(u() < p);
        t[i] = ALPHA * z[i] + g();
        y[i] = B * t[i] + G * t[i] * x[i] + z[i] + 0.3 * x[i] + g();
    }
    TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
        ("x", x.as_slice()),
    ])
    .unwrap()
}

fn conditional_dag() -> Dag {
    let mut g = Dag::with_variables(4);
    g.insert_directed(d(2), d(0)).unwrap();
    g.insert_directed(d(2), d(1)).unwrap();
    g.insert_directed(d(0), d(1)).unwrap();
    g.insert_directed(d(3), d(1)).unwrap();
    g
}

fn conditional_cpdag() -> Cpdag {
    let mut g = Cpdag::with_variables(4);
    g.insert_undirected(d(2), d(0)).unwrap();
    g.insert_directed(d(2), d(1)).unwrap();
    g.insert_directed(d(0), d(1)).unwrap();
    g.insert_directed(d(3), d(1)).unwrap();
    g
}

/// PAG conditional data: the ATE PAG law with `x ~ Bern(p)` and
/// `y = t + 0.8 t·x + m + 0.3 x + e`.
fn conditional_pag_data(n: usize, seed: u64, p: f64) -> TabularData {
    let mut g = gaussian(seed);
    let mut u = uniform(seed ^ 0xC0FF_EE00);
    let cols: [Vec<f64>; 6] = std::array::from_fn(|_| vec![0.0; n]);
    let [mut t, mut y, mut z, mut m, mut v, mut x] = cols;
    for i in 0..n {
        z[i] = g();
        t[i] = ALPHA * z[i] + g();
        v[i] = 0.6 * t[i] + g();
        m[i] = 0.9 * z[i] + g();
        x[i] = f64::from(u() < p);
        y[i] = B * t[i] + G * t[i] * x[i] + m[i] + 0.3 * x[i] + g();
    }
    TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
        ("m", m.as_slice()),
        ("v", v.as_slice()),
        ("x", x.as_slice()),
    ])
    .unwrap()
}

#[allow(clippy::too_many_arguments)]
fn run_conditional_coverage(
    name: &str,
    graph: &Structure,
    generate: impl Fn(usize, u64) -> TabularData,
    modifier: u32,
    n: usize,
    bayesian: bool,
    truth: f64,
    seed: u64,
) {
    let mut tally = CoverageTally::new(name, LEVEL);
    for rep in 0..u64::from(n_sim()) {
        let data = generate(n, seed + rep);
        let inference = if bayesian { bayes() } else { InferenceMode::Frequentist };
        let Some(result) = run(data, graph, conditional_query(modifier), inference, seed + rep)
        else {
            tally.skip();
            continue;
        };
        let interval =
            if bayesian { bayes_interval(&result) } else { frequentist_interval(&result) };
        tally.record(interval, truth);
    }
    tally.assert();
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn conditional_effect_dag_frequentist_nominal_90_coverage() {
    run_conditional_coverage(
        "conditional_effect_dag_frequentist",
        &Structure::Dag(conditional_dag()),
        |n, s| conditional_dag_data(n, s, 0.5),
        3,
        400,
        false,
        B + G * 0.5,
        19_500,
    );
}

/// Small-subgroup regime: `P(x = 1) = 0.08` at `n = 300` (about 24 rows in the
/// modifier stratum), so the interaction coefficient is poorly determined.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn conditional_effect_dag_frequentist_small_subgroup_nominal_90_coverage() {
    run_conditional_coverage(
        "conditional_effect_dag_frequentist_small_subgroup",
        &Structure::Dag(conditional_dag()),
        |n, s| conditional_dag_data(n, s, 0.08),
        3,
        300,
        false,
        B + G * 0.08,
        19_600,
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn conditional_effect_dag_bayesian_nominal_90_coverage() {
    run_conditional_coverage(
        "conditional_effect_dag_bayesian",
        &Structure::Dag(conditional_dag()),
        |n, s| conditional_dag_data(n, s, 0.5),
        3,
        400,
        true,
        B + G * 0.5,
        19_700,
    );
}

/// Cpdag: completion `z -> t` adjusts `{z}` (`B + G p`); completion `t -> z`
/// adjusts nothing (`B + KAPPA + G p`).
const CONDITIONAL_CPDAG_TRUTH: f64 = B + G * 0.5 + 0.5 * KAPPA;

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn conditional_effect_cpdag_frequentist_nominal_90_coverage() {
    run_conditional_coverage(
        "conditional_effect_cpdag_frequentist",
        &Structure::Cpdag(conditional_cpdag()),
        |n, s| conditional_dag_data(n, s, 0.5),
        3,
        400,
        false,
        CONDITIONAL_CPDAG_TRUTH,
        19_800,
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn conditional_effect_cpdag_bayesian_nominal_90_coverage() {
    run_conditional_coverage(
        "conditional_effect_cpdag_bayesian",
        &Structure::Cpdag(conditional_cpdag()),
        |n, s| conditional_dag_data(n, s, 0.5),
        3,
        400,
        true,
        CONDITIONAL_CPDAG_TRUTH,
        19_900,
    );
}

/// Pag: four completions at `B + G p`, two at `B + 0.9·KAPPA + G p`.
const CONDITIONAL_PAG_TRUTH: f64 = B + G * 0.5 + 2.0 * PAG_TOTAL_SHIFT / 6.0;

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn conditional_effect_pag_frequentist_nominal_90_coverage() {
    run_conditional_coverage(
        "conditional_effect_pag_frequentist",
        &Structure::Pag(pag()),
        |n, s| conditional_pag_data(n, s, 0.5),
        5,
        400,
        false,
        CONDITIONAL_PAG_TRUTH,
        20_000,
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn conditional_effect_pag_bayesian_nominal_90_coverage() {
    run_conditional_coverage(
        "conditional_effect_pag_bayesian",
        &Structure::Pag(pag()),
        |n, s| conditional_pag_data(n, s, 0.5),
        5,
        400,
        true,
        CONDITIONAL_PAG_TRUTH,
        20_100,
    );
}

// ------------------------------------------------------ CoDetermined / Unknown

/// Same law as `v19_tiered_known_truth::codetermined_data`: tiers
/// `{z, u} | {t} | {y}`, `P(t=1) = logistic(0.6 z − 0.4 u)`,
/// `y = 2 t + z − 0.5 u + e`. The outcome model is linear, so the AIPW closure
/// functional is exactly 2.
fn codetermined_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let mut u01 = uniform(seed);
    let (mut z, mut u, mut t, mut y) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let common = g();
        z[i] = 0.7 * common + 0.7 * g();
        u[i] = 0.7 * common + 0.7 * g();
        let p = 1.0 / (1.0 + (-(0.6 * z[i] - 0.4 * u[i])).exp());
        t[i] = f64::from(u01() < p);
        y[i] = 2.0 * t[i] + z[i] - 0.5 * u[i] + g();
    }
    TabularData::from_f64_columns([
        ("z", z.as_slice()),
        ("u", u.as_slice()),
        ("t", t.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap()
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn codetermined_aipw_closure_nominal_90_coverage() {
    let mut tally = CoverageTally::new("codetermined_aipw_closure", LEVEL);
    for rep in 0..u64::from(n_sim()) {
        let data = codetermined_data(600, 20_200 + rep);
        let schema = data.schema().clone();
        let background = TieredBackground::from_named(
            &schema,
            &[vec!["z", "u"], vec!["t"], vec!["y"]],
            WithinTier::CoDetermined,
        )
        .unwrap();
        let query =
            AverageEffectQuery::binary_ate(schema.id_of("t").unwrap(), schema.id_of("y").unwrap());
        let result = Study::tabular(data)
            .tiered_background(background)
            .unwrap()
            .query(query)
            .estimator(EstimatorId::Aipw)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .run(&ExecutionContext::for_tests(20_200 + rep));
        match result {
            Ok(result) => tally.record(frequentist_interval(&result), 2.0),
            Err(_) => tally.skip(),
        }
    }
    tally.assert();
}

/// Same law as `v19_tiered_known_truth::unknown_data`: scenario truths
/// `[1, −1]` (pretreatment `{era}` total effect; closure `{era, m}` direct).
fn unknown_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let (mut era, mut t, mut m, mut y) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        era[i] = g();
        t[i] = 0.8 * era[i] + g();
        m[i] = t[i] + 0.2 * era[i] + 0.5 * g();
        y[i] = -t[i] + 2.0 * m[i] + 0.2 * era[i] + g();
    }
    TabularData::from_f64_columns([
        ("era", era.as_slice()),
        ("t", t.as_slice()),
        ("m", m.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap()
}

/// The Unknown cell publishes a 95% *simultaneous* (max-t) band over its two
/// canonical scenarios. Coverage is the joint event "both scenario truths lie
/// in their bands". `CoverageTally` scores one interval against one truth, so
/// each replicate records the unit interval `[0, 1]` against `0.5` when the
/// joint event holds and against `2.0` when it does not; the reported mean
/// length is therefore 1 by construction, and the summed band width is
/// printed separately.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn unknown_two_scenario_joint_band_nominal_95_coverage() {
    const TRUTH: [f64; 2] = [1.0, -1.0];
    let mut tally = CoverageTally::new("unknown_two_scenario_joint_band", 0.95);
    let mut width_sum = 0.0;
    let mut widths = 0u32;
    for rep in 0..u64::from(n_sim()) {
        let data = unknown_data(400, 20_300 + rep);
        let schema = data.schema().clone();
        let background = TieredBackground::from_named(
            &schema,
            &[vec!["era"], vec!["t", "m"], vec!["y"]],
            WithinTier::Unknown,
        )
        .unwrap();
        let query =
            AverageEffectQuery::binary_ate(schema.id_of("t").unwrap(), schema.id_of("y").unwrap());
        let result = Study::tabular(data)
            .tiered_background(background)
            .unwrap()
            .query(query)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .run(&ExecutionContext::for_tests(20_300 + rep));
        let Ok(result) = result else {
            tally.skip();
            continue;
        };
        let Some(bands) = result.estimate.scenario_intervals.as_ref() else {
            tally.record(None, 0.5);
            continue;
        };
        let covered = bands.len() == 2
            && bands.iter().zip(TRUTH).all(|(&(lo, hi), truth)| lo <= truth && truth <= hi);
        width_sum += bands.iter().map(|(lo, hi)| hi - lo).sum::<f64>();
        widths += 1;
        tally.record(Some((0.0, 1.0)), if covered { 0.5 } else { 2.0 });
    }
    eprintln!(
        "calibration unknown_two_scenario_joint_band: mean summed band width={:.4}",
        width_sum / f64::from(widths.max(1))
    );
    tally.assert();
}
