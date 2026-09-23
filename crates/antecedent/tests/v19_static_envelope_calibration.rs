//! Repeated-sampling coverage for static envelope and tier cells (R-19).
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
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::too_many_arguments,
    clippy::single_match_else,
    clippy::type_complexity
)]
#![allow(
    clippy::cast_possible_truncation,
    reason = "test scaffolding compares exact constants and indexes with small literals"
)]

mod common;

use antecedent::{BayesianConfig, EstimatorId, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ConditionalEffectQuery, ExecutionContext, VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_graph::{Cpdag, Dag, DenseNodeId, Pag, TieredBackground, WithinTier};

use common::calibration::{
    CoverageTally, REPORTED_LEVEL, RecordKey, Z90, Z95, gaussian, grid_n, map_replicates, n_sim,
    normal_interval, quantile_interval,
};
use common::calibration_bind::bind_all;
// The six-variable envelope PAG is measured by the Bayesian static suite too;
// one owner, so the two suites cannot enumerate different completions.
use common::static_dgp::envelope_pag as pag;

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

/// This suite's uniform stream: the harness's one LCG, conditioned the way the
/// recorded coverage of these cells was measured (a golden-ratio multiply
/// rather than `mix_seed`). The conditioning stays here rather than folding
/// into `calibration::uniform` because changing it would regenerate every
/// replicate and move the pinned rates.
fn uniform(seed: u64) -> impl FnMut() -> f64 {
    common::calibration::uniform_from_state(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
}

fn frequentist_interval(result: &antecedent::StudyResult) -> Option<(f64, f64)> {
    normal_interval(result.estimate.ate, Some(result.estimate.se_analytic), Z90)
}

fn bayes_interval(result: &antecedent::StudyResult) -> Option<(f64, f64)> {
    bayes_interval_at(result, LEVEL)
}

fn bayes_interval_at(result: &antecedent::StudyResult, level: f64) -> Option<(f64, f64)> {
    let posterior = result.posterior.as_ref()?;
    let col = posterior.effect_column()?;
    let draws = posterior.draws.column(col).ok()?;
    quantile_interval(draws, level)
}

/// Gated 90% interval and the runtime's reported 95% interval of the same replicate.
fn intervals(
    result: &antecedent::StudyResult,
    bayesian: bool,
) -> (Option<(f64, f64)>, Option<(f64, f64)>) {
    if bayesian {
        (bayes_interval(result), bayes_interval_at(result, REPORTED_LEVEL))
    } else {
        (
            frequentist_interval(result),
            normal_interval(result.estimate.ate, Some(result.estimate.se_analytic), Z95),
        )
    }
}

/// Record key of a scalar interval: the posterior quantile interval for a
/// Bayesian run, else the analytic-SE interval (no bootstrap is requested).
fn scalar_key(test: &'static str, dgp: &'static str, bayesian: bool) -> RecordKey {
    RecordKey { test, dgp, interval: if bayesian { "posterior_quantile" } else { "analytic_se" } }
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
) -> Option<(Study, antecedent::StudyResult)> {
    let builder = Study::tabular(data);
    let builder = match graph {
        Structure::Dag(g) => builder.graph(g.clone()),
        Structure::Cpdag(g) => builder.graph(g.clone()),
        Structure::Pag(g) => builder.graph(g.clone()),
    };
    let study = builder
        .query(query)
        .inference(inference)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .ok()?;
    let result = study.run(&ExecutionContext::for_tests(seed)).ok()?;
    Some((study, result))
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

/// Gated at 90%; the runtime's reported 95% interval is scored on the same
/// replicates and recorded.
fn run_ate_coverage(
    test: &'static str,
    dgp: &'static str,
    graph: &Structure,
    generate: impl Fn(usize, u64) -> TabularData + Sync,
    n: usize,
    inference: impl Fn() -> InferenceMode + Sync,
    bayesian: bool,
    truth: f64,
    seed: u64,
    measured: [Option<f64>; 3],
) {
    let key = scalar_key(test, dgp, bayesian);
    let mut tally = CoverageTally::for_record(key, LEVEL);
    let mut reported = CoverageTally::for_record(key, REPORTED_LEVEL).unasserted();
    let runs = map_replicates(n_sim(), |rep| {
        let data = generate(grid_n(n), seed + rep);
        run(data, graph, ate_query(), inference(), seed + rep)
    });
    for scored in &runs {
        let Some((study, result)) = scored else {
            tally.skip();
            reported.skip();
            continue;
        };
        let (interval, interval_95) = intervals(result, bayesian);
        bind_all(&mut [&mut tally, &mut reported], study, result);
        tally.record(interval, truth);
        reported.record(interval_95, truth);
    }
    tally.assert_boundary_at(measured);
    reported.emit();
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn static_cpdag_ate_envelope_frequentist_nominal_90_coverage() {
    run_ate_coverage(
        "static_cpdag_ate_envelope_frequentist_nominal_90_coverage",
        "cpdag_data",
        &Structure::Cpdag(cpdag()),
        cpdag_data,
        400,
        || InferenceMode::Frequentist,
        false,
        CPDAG_TRUTH,
        19_100,
        [Some(0.880), None, None],
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn static_cpdag_ate_envelope_bayesian_nominal_90_coverage() {
    run_ate_coverage(
        "static_cpdag_ate_envelope_bayesian_nominal_90_coverage",
        "cpdag_data",
        &Structure::Cpdag(cpdag()),
        cpdag_data,
        400,
        bayes,
        true,
        CPDAG_TRUTH,
        19_200,
        [None, None, None],
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

const PAG_TOTAL_SHIFT: f64 = 0.9 * KAPPA;
const PAG_TRUTH: f64 = (4.0 * B + 2.0 * (B + PAG_TOTAL_SHIFT)) / 6.0;

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn static_pag_ate_envelope_frequentist_nominal_90_coverage() {
    run_ate_coverage(
        "static_pag_ate_envelope_frequentist_nominal_90_coverage",
        "pag_data",
        &Structure::Pag(pag()),
        pag_data,
        400,
        || InferenceMode::Frequentist,
        false,
        PAG_TRUTH,
        19_300,
        [None, None, None],
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn static_pag_ate_envelope_bayesian_nominal_90_coverage() {
    run_ate_coverage(
        "static_pag_ate_envelope_bayesian_nominal_90_coverage",
        "pag_data",
        &Structure::Pag(pag()),
        pag_data,
        400,
        bayes,
        true,
        PAG_TRUTH,
        19_400,
        [None, None, None],
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

/// Gated at 90%; the runtime's reported 95% interval is scored on the same
/// replicates and recorded.
#[allow(clippy::too_many_arguments)]
fn run_conditional_coverage(
    test: &'static str,
    dgp: &'static str,
    graph: &Structure,
    generate: impl Fn(usize, u64) -> TabularData + Sync,
    modifier: u32,
    n: usize,
    bayesian: bool,
    truth: f64,
    seed: u64,
    measured: [Option<f64>; 3],
) {
    let key = scalar_key(test, dgp, bayesian);
    let mut tally = CoverageTally::for_record(key, LEVEL);
    let mut reported = CoverageTally::for_record(key, REPORTED_LEVEL).unasserted();
    let runs = map_replicates(n_sim(), |rep| {
        let data = generate(grid_n(n), seed + rep);
        let inference = if bayesian { bayes() } else { InferenceMode::Frequentist };
        run(data, graph, conditional_query(modifier), inference, seed + rep)
    });
    for scored in &runs {
        let Some((study, result)) = scored else {
            tally.skip();
            reported.skip();
            continue;
        };
        let (interval, interval_95) = intervals(result, bayesian);
        bind_all(&mut [&mut tally, &mut reported], study, result);
        tally.record(interval, truth);
        reported.record(interval_95, truth);
    }
    tally.assert_boundary_at(measured);
    reported.emit();
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn conditional_effect_dag_frequentist_nominal_90_coverage() {
    run_conditional_coverage(
        "conditional_effect_dag_frequentist_nominal_90_coverage",
        "conditional_dag_data",
        &Structure::Dag(conditional_dag()),
        |n, s| conditional_dag_data(n, s, 0.5),
        3,
        400,
        false,
        B + G * 0.5,
        19_500,
        [None, None, None],
    );
}

/// Small-subgroup regime: `P(x = 1) = 0.08` at `n = 300` (about 24 rows in the
/// modifier stratum), so the interaction coefficient is poorly determined.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn conditional_effect_dag_frequentist_small_subgroup_nominal_90_coverage() {
    run_conditional_coverage(
        "conditional_effect_dag_frequentist_small_subgroup_nominal_90_coverage",
        "conditional_dag_data",
        &Structure::Dag(conditional_dag()),
        |n, s| conditional_dag_data(n, s, 0.08),
        3,
        300,
        false,
        B + G * 0.08,
        19_600,
        [None, None, None],
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn conditional_effect_dag_bayesian_nominal_90_coverage() {
    run_conditional_coverage(
        "conditional_effect_dag_bayesian_nominal_90_coverage",
        "conditional_dag_data",
        &Structure::Dag(conditional_dag()),
        |n, s| conditional_dag_data(n, s, 0.5),
        3,
        400,
        true,
        B + G * 0.5,
        19_700,
        // Grid point 0 measures 0.878 at 2000 replicates (1756/2000), outside the
        // band [0.880, 0.920]: a named boundary, not a band failure.
        [Some(0.878), None, None],
    );
}

/// Cpdag: completion `z -> t` adjusts `{z}` (`B + G p`); completion `t -> z`
/// adjusts nothing (`B + KAPPA + G p`).
const CONDITIONAL_CPDAG_TRUTH: f64 = B + G * 0.5 + 0.5 * KAPPA;

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn conditional_effect_cpdag_frequentist_nominal_90_coverage() {
    run_conditional_coverage(
        "conditional_effect_cpdag_frequentist_nominal_90_coverage",
        "conditional_dag_data",
        &Structure::Cpdag(conditional_cpdag()),
        |n, s| conditional_dag_data(n, s, 0.5),
        3,
        400,
        false,
        CONDITIONAL_CPDAG_TRUTH,
        19_800,
        [Some(0.880), None, None],
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn conditional_effect_cpdag_bayesian_nominal_90_coverage() {
    run_conditional_coverage(
        "conditional_effect_cpdag_bayesian_nominal_90_coverage",
        "conditional_dag_data",
        &Structure::Cpdag(conditional_cpdag()),
        |n, s| conditional_dag_data(n, s, 0.5),
        3,
        400,
        true,
        CONDITIONAL_CPDAG_TRUTH,
        19_900,
        [None, None, None],
    );
}

/// Pag: four completions at `B + G p`, two at `B + 0.9·KAPPA + G p`.
const CONDITIONAL_PAG_TRUTH: f64 = B + G * 0.5 + 2.0 * PAG_TOTAL_SHIFT / 6.0;

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn conditional_effect_pag_frequentist_nominal_90_coverage() {
    run_conditional_coverage(
        "conditional_effect_pag_frequentist_nominal_90_coverage",
        "conditional_pag_data",
        &Structure::Pag(pag()),
        |n, s| conditional_pag_data(n, s, 0.5),
        5,
        400,
        false,
        CONDITIONAL_PAG_TRUTH,
        20_000,
        [None, None, None],
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn conditional_effect_pag_bayesian_nominal_90_coverage() {
    run_conditional_coverage(
        "conditional_effect_pag_bayesian_nominal_90_coverage",
        "conditional_pag_data",
        &Structure::Pag(pag()),
        |n, s| conditional_pag_data(n, s, 0.5),
        5,
        400,
        true,
        CONDITIONAL_PAG_TRUTH,
        20_100,
        [None, None, None],
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
    let key =
        scalar_key("codetermined_aipw_closure_nominal_90_coverage", "codetermined_data", false);
    let mut tally = CoverageTally::for_record(key, LEVEL);
    let mut reported = CoverageTally::for_record(key, REPORTED_LEVEL).unasserted();
    let runs = map_replicates(n_sim(), |rep| {
        let data = codetermined_data(grid_n(600), 20_200 + rep);
        let schema = data.schema().clone();
        let background = TieredBackground::from_named(
            &schema,
            &[vec!["z", "u"], vec!["t"], vec!["y"]],
            WithinTier::CoDetermined,
        )
        .unwrap();
        let query =
            AverageEffectQuery::binary_ate(schema.id_of("t").unwrap(), schema.id_of("y").unwrap());
        let study = Study::tabular(data)
            .tiered_background(background)
            .unwrap()
            .query(query)
            .estimator(EstimatorId::Aipw)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap();
        let result = study.run(&ExecutionContext::for_tests(20_200 + rep)).ok()?;
        Some((study, result))
    });
    for scored in &runs {
        match scored {
            Some((study, result)) => {
                let (interval, interval_95) = intervals(result, false);
                bind_all(&mut [&mut tally, &mut reported], study, result);
                tally.record(interval, 2.0);
                reported.record(interval_95, 2.0);
            }
            None => {
                tally.skip();
                reported.skip();
            }
        }
    }
    tally.assert();
    reported.emit();
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
    // No record: the runtime keys this execution's interval as `none`; the
    // scenario max-t band is not a reported primary/identified-set/simultaneous interval.
    let mut tally = CoverageTally::new("unknown_two_scenario_joint_band", 0.95);
    let mut width_sum = 0.0;
    let mut widths = 0u32;
    let runs = map_replicates(n_sim(), |rep| {
        let data = unknown_data(grid_n(400), 20_300 + rep);
        let schema = data.schema().clone();
        let background = TieredBackground::from_named(
            &schema,
            &[vec!["era"], vec!["t", "m"], vec!["y"]],
            WithinTier::Unknown,
        )
        .unwrap();
        let query =
            AverageEffectQuery::binary_ate(schema.id_of("t").unwrap(), schema.id_of("y").unwrap());
        Study::tabular(data)
            .tiered_background(background)
            .unwrap()
            .query(query)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .run(&ExecutionContext::for_tests(20_300 + rep))
            .ok()
    });
    for scored in &runs {
        let Some(result) = scored else {
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
    tally.assert_boundary_at([Some(0.940), None, None]);
}
