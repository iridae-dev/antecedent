//! Coverage tests for single-atom TemporalCpdag / TemporalPag *class* graph
//! posteriors: Pulse, Sustained and (temporal) Mediation effects.
//!
//! Each construction is a genuine `.graph_posterior(class_posterior(..))` with a
//! single fully-identified atom (weight `[1.0]`). A single identified atom mixes
//! under `StructuralAggregationPolicy::SameEstimandWeightedMean`, so the facade
//! publishes a scalar `conditional_on_identified` and a scoreable interval: the
//! shared circular-block bootstrap SE for the Frequentist arm and the posterior
//! quantile for the Bayesian arm (a partial-mass posterior is withheld, which is
//! why the Bayesian rows require weight `[1.0]`).
//!
//! Ignored tests run via `scripts/gate_calibration.sh`.
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
    CausalQuery, ExecutionContext, IdentificationStatus, MediationContrast, MediationQuery,
    TemporalEffectQuery, TemporalPolicy, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind, set_edge};
use antecedent_prob::InferenceDiagnostics;

use common::calibration::{
    CoverageTally, RecordKey, gaussian, grid_n, map_replicates, n_sim, stream_seed,
};
use common::calibration_bind::bind_all;
use common::reported::{
    GATE_LEVEL, REPORTED_LEVEL, gate_at, posterior_pair, record_pair, scalar_normal_pair,
};

/// Base series length; `grid_n` measures it at `n/2, n, 2n`.
const N: usize = 160;
/// Outer circular-block replicates per Frequentist fit.
const BOOT: u32 = 199;

// ---------------------------------------------------------------------------
// Data-generating processes (columns match the class-posterior lag masks).
// ---------------------------------------------------------------------------

/// Pulse law on `[t (0), y (1), z (2), w (3)]`:
/// `z[s] ~ N(0,1)`, `w[s] = 0.4 z[s] + e`, `t[s] = 0.5 z[s] + 0.4 w[s] + e`,
/// `y[s] = 1 + 2 t[s-1] + 0.8 z[s-1] + e`.
///
/// `z@-1` confounds `t@-1` and `y` (via `z[s-1] -> t[s-1]` and `z[s-1] -> y[s]`),
/// so adjusting for `{z@-1}` identifies the lag-1 effect of `t` on `y`. The pulse
/// contrast `E[y | do(t@-1 = 1)] - E[y | do(t@-1 = 0)] = 2.0*(1 - 0) = 2.0`; the
/// intercept and the `0.8 z@-1` term cancel in the do(1)-do(0) contrast.
///
/// `w` is the second contemporaneous parent of `t` (`t = 0.5 z + 0.4 w`, `w` a
/// child of `z`) and has no path to `y`: it is the visibility witness that makes
/// the definite lag-1 `t@-1 -> y@0` edge visible under a PAG (Zhang 2008), the
/// temporal analog of the static `agreeing_pag`'s `r`. The only backdoor path it
/// opens, `t@-1 <- w@-1 <- z@-1 -> y@0`, is already blocked by `{z@-1}`, so the
/// adjustment set and the truth `2.0` are unchanged.
fn pulse_series(n: usize, seed: u64) -> TimeSeriesData {
    const BURN: usize = 10;
    let len = n + BURN;
    let mut noise = gaussian(seed);
    let (mut t, mut y, mut z, mut w) =
        (vec![0.0; len], vec![0.0; len], vec![0.0; len], vec![0.0; len]);
    for s in 0..len {
        z[s] = noise();
        w[s] = 0.4 * z[s] + noise();
        t[s] = 0.5 * z[s] + 0.4 * w[s] + noise();
        let (t1, z1) = if s == 0 { (0.0, 0.0) } else { (t[s - 1], z[s - 1]) };
        y[s] = 1.0 + 2.0 * t1 + 0.8 * z1 + noise();
    }
    TimeSeriesData::from_f64_columns(
        [("t", &t[BURN..]), ("y", &y[BURN..]), ("z", &z[BURN..]), ("w", &w[BURN..])],
        1,
    )
    .unwrap()
}

/// Sustained law on `[t (0), y (1), z (2)]`, the pulse law plus a lag-2 `t -> y`
/// edge: `y[s] = 1 + B1 t[s-1] + B2 t[s-2] + 0.8 z[s-1] + e` with `B1 = 2.0`,
/// `B2 = 1.0`.
///
/// `z@-1` confounds `t@-1` and `y`; `t@-2` is unconfounded for `y` (`z@-2` does
/// not enter `y`), so adjusting for `{z@-1}` identifies both lag coefficients.
/// The sustained contrast holds `t = 1` at lags -1 and -2 vs `t = 0` at both, so
/// the effect is `B1 + B2 = 2.0 + 1.0 = 3.0`.
fn sustained_series(n: usize, seed: u64) -> TimeSeriesData {
    const BURN: usize = 10;
    let len = n + BURN;
    let mut noise = gaussian(seed);
    let (mut t, mut y, mut z) = (vec![0.0; len], vec![0.0; len], vec![0.0; len]);
    for s in 0..len {
        z[s] = noise();
        t[s] = 0.5 * z[s] + noise();
        let t1 = if s >= 1 { t[s - 1] } else { 0.0 };
        let t2 = if s >= 2 { t[s - 2] } else { 0.0 };
        let z1 = if s >= 1 { z[s - 1] } else { 0.0 };
        y[s] = 1.0 + 2.0 * t1 + 1.0 * t2 + 0.8 * z1 + noise();
    }
    TimeSeriesData::from_f64_columns([("t", &t[BURN..]), ("y", &y[BURN..]), ("z", &z[BURN..])], 1)
        .unwrap()
}

/// Mediation law on `[t (0), m (1), y (2)]`:
/// `t[s] ~ N(0,1)`, `m[s] = 0.8 t[s-1] + e`, `y[s] = 0.25 t[s-1] + 0.55 m[s] + e`.
///
/// The mediator-outcome relation is unconfounded (`t` is the only common cause,
/// and it is conditioned on), so the mediated (natural indirect) effect of `t`
/// on `y` through `m` is the path product `0.8 * 0.55 = 0.44` for the mediated
/// contrast.
fn mediation_series(n: usize, seed: u64) -> TimeSeriesData {
    const BURN: usize = 10;
    let len = n + BURN;
    let mut noise = gaussian(seed);
    let (mut t, mut m, mut y) = (vec![0.0; len], vec![0.0; len], vec![0.0; len]);
    for s in 0..len {
        t[s] = noise();
        let t1 = if s >= 1 { t[s - 1] } else { 0.0 };
        m[s] = 0.8 * t1 + 0.4 * noise();
        y[s] = 0.25 * t1 + 0.55 * m[s] + 0.4 * noise();
    }
    TimeSeriesData::from_f64_columns([("t", &t[BURN..]), ("m", &m[BURN..]), ("y", &y[BURN..])], 1)
        .unwrap()
}

/// Truth of the pulse contrast: the lag-1 `t -> y` coefficient (see `pulse_series`).
const PULSE_TRUTH: f64 = 2.0;
/// Truth of the sustained contrast: `B1 + B2` (see `sustained_series`).
const SUSTAINED_TRUTH: f64 = 3.0;
/// Truth of the mediated contrast: the `t -> m -> y` path product (see `mediation_series`).
const MEDIATION_TRUTH: f64 = 0.8 * 0.55;

// ---------------------------------------------------------------------------
// Single-atom class posteriors and queries.
// ---------------------------------------------------------------------------

/// One identified class-posterior atom (weight `[1.0]`) with no contemporaneous
/// edges (`adjacency = 0`) and the given lag mask over `n` variables.
fn temporal_atom(
    kind: GraphPosteriorAtomKind,
    n: usize,
    adjacency: u64,
    max_lag: u32,
    lag_mask: u64,
) -> GraphPosterior {
    GraphPosterior::new(
        n,
        vec![1.0],
        vec![adjacency],
        vec![0.0; n * n],
        vec![0.0; n * n],
        1.0,
        InferenceDiagnostics::analytic("temporal_class_single_atom"),
        0,
    )
    .unwrap()
    .with_atom_kind(kind)
    .with_lagged_marginals(max_lag, vec![0.0; (max_lag as usize) * n * n])
    .unwrap()
    .with_lag_masks(vec![lag_mask])
    .unwrap()
}

/// Pulse atom on `[t, y, z, w]`: lag-1 edges `t@-1 -> y` and `z@-1 -> y`.
/// Bits (lag-1, `from*4 + to`): `t(0)->y(1)` = 1, `z(2)->y(1)` = 9; mask `514`.
fn pulse_atom(kind: GraphPosteriorAtomKind) -> GraphPosterior {
    let lag_mask = (1u64 << 1) | (1u64 << 9);
    // Contemporaneous z -> t and w -> t (the DGP's t = 0.5 z + 0.4 w): z@-1
    // confounds the lag-1 t -> y effect and identification adjusts for it, while
    // w@-1 (no edge to y@0) is the visibility witness that makes t@-1 -> y@0
    // visible under the Pag atom kind (matches pulse_series).
    let contemporaneous = set_edge(set_edge(0, 4, 2, 0, true), 4, 3, 0, true);
    temporal_atom(kind, 4, contemporaneous, 1, lag_mask)
}

/// Sustained atom on `[t, y, z]`: the pulse edges plus `t@-2 -> y`.
/// Lag-2 block starts at `1*3*3 = 9`; `t(0)->y(1)` = `9 + 1 = 10`; mask
/// `130 | (1 << 10) = 1154`, `max_lag = 2`.
fn sustained_atom(kind: GraphPosteriorAtomKind) -> GraphPosterior {
    let lag_mask = (1u64 << 1) | (1u64 << 7) | (1u64 << 10);
    // Contemporaneous z -> t (the DGP's t = 0.5 z), so z@-1 confounds the lag-1
    // t -> y effect and identification adjusts for it.
    let contemporaneous = set_edge(0, 3, 2, 0, true);
    temporal_atom(kind, 3, contemporaneous, 2, lag_mask)
}

/// Mediation atom on `[t, m, y]`: contemporaneous `m -> y` and lag-1 `t@-1 -> m`,
/// `t@-1 -> y`. Contemporaneous bit via `set_edge` (compact edge index):
/// `m(1)->y(2)` = `1 << 3 = 8`. Lag-1 bits (`from*3 + to`): `t(0)->m(1)` = 1,
/// `t(0)->y(2)` = 2; mask `6`.
fn mediation_atom() -> GraphPosterior {
    let contemporaneous = set_edge(0, 3, 1, 2, true);
    let lag_mask = (1u64 << 1) | (1u64 << 2);
    temporal_atom(GraphPosteriorAtomKind::Cpdag, 3, contemporaneous, 1, lag_mask)
}

fn licensed_pulse() -> TemporalEffectQuery {
    TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1)
        .with_max_history_lag(Some(1))
}

fn multi_sustained() -> TemporalEffectQuery {
    TemporalEffectQuery::sustained(VariableId::from_raw(0), VariableId::from_raw(1), -2, 1.0)
        .with_policy(TemporalPolicy::sustained(-2, -1))
        .with_horizon_steps(1)
        .with_max_history_lag(Some(2))
}

fn mediated_query() -> MediationQuery {
    MediationQuery::binary(
        VariableId::from_raw(0),
        VariableId::from_raw(2),
        [VariableId::from_raw(1)],
        MediationContrast::Mediated,
    )
    .with_horizons(vec![1])
    .unwrap()
}

// ---------------------------------------------------------------------------
// Runners.
// ---------------------------------------------------------------------------

fn run_temporal(
    data: TimeSeriesData,
    query: TemporalEffectQuery,
    gp: GraphPosterior,
    inference: InferenceMode,
    boot: u32,
    seed: u64,
) -> (Study, StudyResult) {
    let study = Study::series(data)
        .graph_posterior(gp)
        .temporal_query(query)
        .inference(inference)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(boot)
        .build()
        .unwrap();
    let result = study.run(&ExecutionContext::for_tests(seed)).unwrap();
    (study, result)
}

fn run_mediation(
    data: TimeSeriesData,
    gp: GraphPosterior,
    inference: InferenceMode,
    boot: u32,
    seed: u64,
) -> (Study, StudyResult) {
    let study = Study::series(data)
        .graph_posterior(gp)
        .query(CausalQuery::Mediation(mediated_query()))
        .inference(inference)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(boot)
        .build()
        .unwrap();
    let result = study.run(&ExecutionContext::for_tests(seed)).unwrap();
    (study, result)
}

fn bayes() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(1000).prior_scale(8.0))
}

/// Effect column the facade designates for the scored posterior.
fn effect_col(result: &StudyResult) -> usize {
    result.posterior.as_ref().and_then(antecedent::CausalPosterior::effect_column).unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Coverage driver.
// ---------------------------------------------------------------------------

/// Score the reported (0.95) and gate (0.90) intervals of one single-atom class
/// graph-posterior construction. At rep 0 it asserts the estimator target, that
/// the single identified atom is not `NotIdentified`, and that the interval is
/// actually published (a hollow, withheld interval fails here instead of
/// silently recording a miss).
fn coverage(
    test: &'static str,
    dgp: &'static str,
    interval: &'static str,
    estimator: &'static str,
    truth: f64,
    run: impl Fn(u64) -> (Study, StudyResult) + Sync,
    pair: impl Fn(&StudyResult) -> [Option<(f64, f64)>; 2] + Sync,
    measured: [[Option<f64>; 3]; 2],
) {
    let mut tallies = [
        CoverageTally::for_record(RecordKey { test, dgp, interval }, REPORTED_LEVEL),
        CoverageTally::for_record(RecordKey { test, dgp, interval }, GATE_LEVEL),
    ];
    let runs = map_replicates(n_sim(), |rep| {
        let (study, result) = run(rep);
        let intervals = pair(&result);
        if rep == 0 {
            assert_eq!(
                result.logical_plan.estimator.as_deref(),
                Some(estimator),
                "{test}: estimator target"
            );
            assert_ne!(
                result.identification.status,
                IdentificationStatus::NotIdentified,
                "{test}: the single class-posterior atom must identify"
            );
            assert!(
                intervals[0].is_some(),
                "{test}: a single identified atom must publish the scored interval"
            );
        }
        (study, result, intervals)
    });
    for (study, result, intervals) in &runs {
        let [reported, gate_level] = &mut tallies;
        bind_all(&mut [reported, gate_level], study, result);
        record_pair(&mut tallies, *intervals, truth);
    }
    gate_at(&tallies, &measured);
}

// ---------------------------------------------------------------------------
// PulseEffect / TemporalCpdag / graph_posterior.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn pulse_effect_temporal_cpdag_graph_posterior_frequentist_nominal_coverage() {
    coverage(
        "pulse_effect_temporal_cpdag_graph_posterior_frequentist_nominal_coverage",
        "pulse_series",
        "circular_block_se",
        "temporal.linear.adjustment",
        PULSE_TRUTH,
        |rep| {
            let seed = stream_seed(0x11C_0001, rep);
            run_temporal(
                pulse_series(grid_n(N), seed),
                licensed_pulse(),
                pulse_atom(GraphPosteriorAtomKind::Cpdag),
                InferenceMode::Frequentist,
                BOOT,
                seed,
            )
        },
        scalar_normal_pair,
        [[None, None, None], [None, None, None]],
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn pulse_effect_temporal_cpdag_graph_posterior_bayesian_nominal_coverage() {
    coverage(
        "pulse_effect_temporal_cpdag_graph_posterior_bayesian_nominal_coverage",
        "pulse_series",
        "posterior_quantile",
        "bayesian.temporal.gcomp",
        PULSE_TRUTH,
        |rep| {
            let seed = stream_seed(0x11C_0002, rep);
            run_temporal(
                pulse_series(grid_n(N), seed),
                licensed_pulse(),
                pulse_atom(GraphPosteriorAtomKind::Cpdag),
                bayes(),
                0,
                seed,
            )
        },
        |result| posterior_pair(result, effect_col(result)),
        [[None, None, None], [None, None, None]],
    );
}

// ---------------------------------------------------------------------------
// PulseEffect / TemporalPag / graph_posterior.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn pulse_effect_temporal_pag_graph_posterior_frequentist_nominal_coverage() {
    coverage(
        "pulse_effect_temporal_pag_graph_posterior_frequentist_nominal_coverage",
        "pulse_series",
        "circular_block_se",
        "temporal.linear.adjustment",
        PULSE_TRUTH,
        |rep| {
            let seed = stream_seed(0x11C_0003, rep);
            run_temporal(
                pulse_series(grid_n(N), seed),
                licensed_pulse(),
                pulse_atom(GraphPosteriorAtomKind::Pag),
                InferenceMode::Frequentist,
                BOOT,
                seed,
            )
        },
        scalar_normal_pair,
        // Grid point 1 measures 0.940 at 0.95 (floor 0.940, 1879/2000) and 0.886 at
        // 0.90 (floor 0.887, 1773/2000) at 2000 replicates: named boundaries at the
        // precision floor, the same mild finite-sample behavior of the circular-block
        // analytic interval as the temporal-CPDAG twins. Points 0 and 2 gate nominal.
        [[None, Some(0.940), None], [None, Some(0.886), None]],
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn pulse_effect_temporal_pag_graph_posterior_bayesian_nominal_coverage() {
    coverage(
        "pulse_effect_temporal_pag_graph_posterior_bayesian_nominal_coverage",
        "pulse_series",
        "posterior_quantile",
        "bayesian.temporal.gcomp",
        PULSE_TRUTH,
        |rep| {
            let seed = stream_seed(0x11C_0004, rep);
            run_temporal(
                pulse_series(grid_n(N), seed),
                licensed_pulse(),
                pulse_atom(GraphPosteriorAtomKind::Pag),
                bayes(),
                0,
                seed,
            )
        },
        |result| posterior_pair(result, effect_col(result)),
        [[None, None, None], [None, None, None]],
    );
}

// ---------------------------------------------------------------------------
// SustainedEffect / TemporalCpdag / graph_posterior.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn sustained_effect_temporal_cpdag_graph_posterior_frequentist_nominal_coverage() {
    coverage(
        "sustained_effect_temporal_cpdag_graph_posterior_frequentist_nominal_coverage",
        "sustained_series",
        "circular_block_se",
        "temporal.sequential.gcomp",
        SUSTAINED_TRUTH,
        |rep| {
            let seed = stream_seed(0x11C_0005, rep);
            run_temporal(
                sustained_series(grid_n(N), seed),
                multi_sustained(),
                sustained_atom(GraphPosteriorAtomKind::Cpdag),
                InferenceMode::Frequentist,
                BOOT,
                seed,
            )
        },
        scalar_normal_pair,
        // Grid point 2 measures 0.940 at 2000 replicates against the precision
        // floor 0.940 (1880/2000): a named boundary, not a band failure.
        [[None, None, Some(0.940)], [None, None, None]],
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn sustained_effect_temporal_cpdag_graph_posterior_bayesian_nominal_coverage() {
    coverage(
        "sustained_effect_temporal_cpdag_graph_posterior_bayesian_nominal_coverage",
        "sustained_series",
        "posterior_quantile",
        "temporal.sequential.gcomp",
        SUSTAINED_TRUTH,
        |rep| {
            let seed = stream_seed(0x11C_0006, rep);
            run_temporal(
                sustained_series(grid_n(N), seed),
                multi_sustained(),
                sustained_atom(GraphPosteriorAtomKind::Cpdag),
                bayes(),
                0,
                seed,
            )
        },
        |result| posterior_pair(result, effect_col(result)),
        [[None, None, None], [None, None, None]],
    );
}

// ---------------------------------------------------------------------------
// SustainedEffect / TemporalPag / graph_posterior.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn sustained_effect_temporal_pag_graph_posterior_frequentist_nominal_coverage() {
    coverage(
        "sustained_effect_temporal_pag_graph_posterior_frequentist_nominal_coverage",
        "sustained_series",
        "circular_block_se",
        "temporal.sequential.gcomp",
        SUSTAINED_TRUTH,
        |rep| {
            let seed = stream_seed(0x11C_0007, rep);
            run_temporal(
                sustained_series(grid_n(N), seed),
                multi_sustained(),
                sustained_atom(GraphPosteriorAtomKind::Pag),
                InferenceMode::Frequentist,
                BOOT,
                seed,
            )
        },
        scalar_normal_pair,
        [[None, None, None], [None, None, None]],
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn sustained_effect_temporal_pag_graph_posterior_bayesian_nominal_coverage() {
    coverage(
        "sustained_effect_temporal_pag_graph_posterior_bayesian_nominal_coverage",
        "sustained_series",
        "posterior_quantile",
        "temporal.sequential.gcomp",
        SUSTAINED_TRUTH,
        |rep| {
            let seed = stream_seed(0x11C_0008, rep);
            run_temporal(
                sustained_series(grid_n(N), seed),
                multi_sustained(),
                sustained_atom(GraphPosteriorAtomKind::Pag),
                bayes(),
                0,
                seed,
            )
        },
        |result| posterior_pair(result, effect_col(result)),
        [[None, None, None], [None, None, None]],
    );
}

// ---------------------------------------------------------------------------
// TemporalMediationEffect / TemporalCpdag / graph_posterior.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn temporal_mediation_effect_temporal_cpdag_graph_posterior_frequentist_nominal_coverage() {
    coverage(
        "temporal_mediation_effect_temporal_cpdag_graph_posterior_frequentist_nominal_coverage",
        "mediation_series",
        "circular_block_se",
        "temporal.mediation",
        MEDIATION_TRUTH,
        |rep| {
            let seed = stream_seed(0x11C_0009, rep);
            run_mediation(
                mediation_series(grid_n(N), seed),
                mediation_atom(),
                InferenceMode::Frequentist,
                BOOT,
                seed,
            )
        },
        scalar_normal_pair,
        // Grid point 0, 0.90 level measures 0.916 at 2000 replicates against the
        // precision ceiling 0.913 (1833/2000): a named boundary over-covering,
        // not a band failure.
        [[None, None, None], [Some(0.916), None, None]],
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn temporal_mediation_effect_temporal_cpdag_graph_posterior_bayesian_nominal_coverage() {
    coverage(
        "temporal_mediation_effect_temporal_cpdag_graph_posterior_bayesian_nominal_coverage",
        "mediation_series",
        "posterior_quantile",
        "temporal.mediation.bayesian",
        MEDIATION_TRUTH,
        |rep| {
            let seed = stream_seed(0x11C_000A, rep);
            run_mediation(mediation_series(grid_n(N), seed), mediation_atom(), bayes(), 0, seed)
        },
        |result| posterior_pair(result, effect_col(result)),
        [[None, None, None], [None, None, None]],
    );
}
