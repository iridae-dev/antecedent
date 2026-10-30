//! Repeated-sampling coverage of the intervals a practitioner-sized study reports: 1,000 to
//! 100,000 independent rows with twenty adjustment covariates, and regular series of 320 to
//! 4,160 steps (so the 1,024- and 4,096-step series).
//!
//! The ordinary suites measure the same constructions at 150 to 1,200 rows or 80 to 320 steps,
//! and a record never extrapolates beyond the span of its sample-size grid. These designs
//! measure the construction at the scale itself, so a result at 10,000 or 100,000 rows or at
//! 1,024 or 4,096 steps matches a record of its own instead of reporting
//! `sample_size_outside_measured_range`. Each record is published with its measured range as
//! the limit: a row count above 100,000 or a series longer than 4,160 steps is outside every
//! record and is not assessed, however many grid points passed.
//!
//! Static designs (`practitioner_observational_study`): `z_0..z_19 ~ N(0, 1)`, a logistic
//! treatment on every covariate, and an outcome linear in the covariates with a constant
//! treatment effect of 0.3. The homoskedastic and clustered variants add a cluster-level
//! intercept and treatment-effect shift over 500 clusters; the heteroskedastic variant scales
//! the noise by the first covariate. Every interval is the facade's own: the construction a
//! record describes is read from the execution (`constructions`), never declared here.
//!
//! The replicate closures return only the construction and the interval, never the study
//! (a 100,000-row table), so memory stays at one table per worker thread.
//!
//! Temporal designs: the driven-treatment series of `common::driven_dgp` with AR(1) residuals
//! (`rho = 0.5`) on a `TemporalDag`, scored through the circular-block bootstrap interval the
//! runtime reports, at 320, 1,280 and 4,160 steps.
//!
//! Ignored tests run via `scripts/gate_calibration.sh` (release build). Wall-clock estimates
//! at 2,000 replicates on a 12-core machine are given at each design.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "test scaffolding builds row indices and covariate coefficients from small counts"
)]

mod common;

use antecedent::discovery::{GraphPosterior, SamplingSummary};
use antecedent::estimate::TemporalMediationUncertainty;
use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ExecutionContext, MediationContrast, VariableId,
};
use antecedent_data::TabularData;
use antecedent_discovery::GraphPosteriorAtomKind;
use antecedent_discovery::adjacency_mask::adjacency_mask_from_cpdag;
use antecedent_estimate::{AipwAte, AnalyticSeKind, LinearAdjustmentAte};
use antecedent_graph::{Cpdag, Dag, DenseNodeId};
use antecedent_prob::InferenceDiagnostics;
use common::calibration::{
    Construction, CoverageTally, RecordKey, SampleGrid, ScopeFacts, Z90, Z95, gaussian, grid_n,
    map_replicates, n_sim, normal_interval, quantile_interval, stream_seed, uniform_from_state,
};
use common::calibration_bind::constructions;
use common::driven_dgp::{
    A, B, BETA, C, Scenario, mediation_dag, mediation_query, mediation_series, pulse, pulse_dag,
    pulse_series, single_sustained,
};
use common::reported::{GATE_LEVEL, REPORTED_LEVEL, gate, posterior_pair, scalar_normal_pair};
use common::static_dgp::linear_ate_data;

/// 1,000, 10,000 and 100,000 rows: the practitioner sizes, one point per decade. The lowest
/// point lies inside the 150 to 1,200 rows the ordinary suites measure, so no row count between
/// those records and the practitioner sizes is left unassessed.
const PRACTITIONER_ROWS: SampleGrid =
    SampleGrid { name: "practitioner_rows", factors: [(1, 100), (1, 10), (1, 1)] };

/// 320, 1,280 and 4,160 steps: the longest series the ordinary suites measure, a 1,024-step
/// range inside the span and the 4,096-step series with the lags it consumes at its top.
const PRACTITIONER_STEPS: SampleGrid =
    SampleGrid { name: "practitioner_steps", factors: [(1, 13), (4, 13), (1, 1)] };

/// Rows of the largest grid point.
const MAX_ROWS: usize = 100_000;
/// Steps of the longest grid point.
const MAX_STEPS: usize = 4_160;

const COVARIATES: usize = 20;
const CLUSTERS: usize = 500;
const TRUTH: f64 = 0.3;
const T: u32 = 0;
const Y: u32 = 1;

fn rows() -> usize {
    PRACTITIONER_ROWS.n(MAX_ROWS)
}

/// The replicate's data and, for the clustered design, the cluster label of every row.
struct PractitionerStudy {
    data: TabularData,
    clusters: Vec<u32>,
}

#[derive(Clone, Copy)]
enum Noise {
    Homoskedastic,
    /// The noise scale grows with `|z_0|`: the variance the classical formula averages away.
    Heteroskedastic,
    /// A cluster-level intercept and treatment-effect shift over [`CLUSTERS`] clusters.
    Clustered,
}

/// `practitioner_observational_study` at `n` rows.
fn practitioner_study(n: usize, seed: u64, noise: Noise) -> PractitionerStudy {
    let mut normal = gaussian(seed);
    let mut uniform = uniform_from_state(seed ^ 0x5DEE_CE66_D1CE_4E5B);
    let treatment_slope =
        |j: usize| 0.3 * if j % 2 == 0 { 1.0 } else { -1.0 } / (1.0 + (j / 5) as f64);
    let outcome_slope =
        |j: usize| 0.5 * if j % 3 == 0 { 1.0 } else { -0.5 } / (1.0 + (j % 4) as f64);
    let clustered = matches!(noise, Noise::Clustered);
    let shifts: Vec<(f64, f64)> = if clustered {
        (0..CLUSTERS).map(|_| (0.4 * normal(), 0.3 * normal())).collect()
    } else {
        Vec::new()
    };
    let mut covariates = vec![vec![0.0; n]; COVARIATES];
    let (mut t, mut y) = (vec![0.0; n], vec![0.0; n]);
    let mut clusters = Vec::with_capacity(if clustered { n } else { 0 });
    for i in 0..n {
        let mut logit = 0.0;
        let mut mean = 0.0;
        for (j, column) in covariates.iter_mut().enumerate() {
            let z = normal();
            column[i] = z;
            logit += treatment_slope(j) * z;
            mean += outcome_slope(j) * z;
        }
        t[i] = f64::from(uniform() < 1.0 / (1.0 + (-logit).exp()));
        let (intercept, shift) = if clustered {
            let cluster = i * CLUSTERS / n;
            clusters.push(cluster as u32);
            shifts[cluster]
        } else {
            (0.0, 0.0)
        };
        let scale = if matches!(noise, Noise::Heteroskedastic) {
            0.6 + 0.5 * covariates[0][i].abs()
        } else {
            1.0
        };
        y[i] = (TRUTH + shift) * t[i] + mean + intercept + scale * normal();
    }
    let names: Vec<String> = (0..COVARIATES).map(|j| format!("z{j}")).collect();
    let mut columns: Vec<(&str, &[f64])> = vec![("t", t.as_slice()), ("y", y.as_slice())];
    columns.extend(names.iter().map(String::as_str).zip(covariates.iter().map(Vec::as_slice)));
    PractitionerStudy { data: TabularData::from_f64_columns(columns).expect("columns"), clusters }
}

/// `z_j -> t`, `z_j -> y` and `t -> y`.
fn practitioner_dag() -> Dag {
    let mut dag = Dag::with_variables(2 + COVARIATES as u32);
    let node = DenseNodeId::from_raw;
    for j in 0..COVARIATES as u32 {
        dag.insert_directed(node(2 + j), node(T)).unwrap();
        dag.insert_directed(node(2 + j), node(Y)).unwrap();
    }
    dag.insert_directed(node(T), node(Y)).unwrap();
    dag
}

fn binary_ate() -> AverageEffectQuery {
    AverageEffectQuery::binary_ate(VariableId::from_raw(T), VariableId::from_raw(Y))
}

/// One replicate's reported and gate-level intervals with the constructions the execution
/// reported. A replicate that failed to produce a result is `None`.
struct Scored {
    reported: Vec<(Construction, ScopeFacts, &'static str)>,
    intervals: [Option<(f64, f64)>; 2],
}

/// Run `study` once and read what the tally needs from it before the table is dropped.
fn scored(
    study: &Study,
    seed: u64,
    intervals: impl FnOnce(&StudyResult) -> [Option<(f64, f64)>; 2],
) -> Option<Scored> {
    let result = study.run(&ExecutionContext::production(seed, 1)).ok()?;
    let contract = study.inspect().ok()?;
    Some(Scored { reported: constructions(&contract, &result), intervals: intervals(&result) })
}

/// A pair of tallies, the facade's reported 0.95 interval and the same construction at the
/// 0.90 gate level, keyed by the emitting `test`.
fn tallies(test: &'static str, dgp: &'static str, interval: &'static str) -> [CoverageTally; 2] {
    let key = RecordKey { test, dgp, interval };
    [CoverageTally::for_record(key, REPORTED_LEVEL), CoverageTally::for_record(key, GATE_LEVEL)]
}

/// Score every replicate into both tallies, binding each to the construction whose interval
/// method it scores.
fn score(tallies: &mut [CoverageTally; 2], runs: &[Option<Scored>], truth: f64) {
    for run in runs {
        let Some(run) = run else {
            for tally in tallies.iter_mut() {
                tally.skip();
            }
            continue;
        };
        for (tally, interval) in tallies.iter_mut().zip(run.intervals) {
            let wanted = tally.record_interval().expect("record-keyed tally");
            let (construction, scope, validation) = run
                .reported
                .iter()
                .find(|(construction, _, _)| construction.interval_method == wanted)
                .unwrap_or_else(|| panic!("the execution reported no {wanted} interval"));
            tally.bind_validation(construction, *scope, validation);
            tally.record(interval, truth);
        }
    }
}

const DGP: &str = "crates/antecedent/tests/practitioner_scale_calibration.rs::practitioner_study";

/// Coverage of the interval a Frequentist `AverageEffect` reports with `estimator` (the
/// facade's own resolution of the study when `None`, at its default bootstrap replicates).
fn estimator_coverage(
    test: &'static str,
    interval: &'static str,
    noise: Noise,
    seed_base: u64,
    estimator: impl Fn(&PractitionerStudy) -> Option<antecedent::EstimatorSpec> + Sync,
) {
    let mut pair = tallies(test, DGP, interval);
    let runs = map_replicates(n_sim(), |rep| {
        let seed = stream_seed(seed_base, rep);
        let replicate = practitioner_study(rows(), seed, noise);
        let spec = estimator(&replicate);
        let mut builder = Study::tabular(replicate.data)
            .graph(practitioner_dag())
            .query(binary_ate())
            .refute(RefuteSuite::None);
        if let Some(spec) = spec {
            builder = builder.estimator(spec);
        }
        let study = builder.build().ok()?;
        scored(&study, seed, scalar_normal_pair)
    });
    score(&mut pair, &runs, TRUTH);
    gate(&pair, &[None, None]);
}

fn linear(se_kind: AnalyticSeKind) -> LinearAdjustmentAte {
    LinearAdjustmentAte::default().with_bootstrap_replicates(0).with_se_kind(se_kind)
}

fn aipw(se_kind: AnalyticSeKind) -> AipwAte {
    AipwAte::new().with_bootstrap_replicates(0).with_se_kind(se_kind)
}

// ================================================================== static estimators

/// Linear adjustment, classical standard error, 1,000 to 100,000 rows. About 4 minutes.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn practitioner_linear_adjustment_analytic_interval() {
    estimator_coverage(
        "practitioner_linear_adjustment_analytic_interval",
        "analytic_se",
        Noise::Homoskedastic,
        0x1700_0001,
        |_| Some(linear(AnalyticSeKind::Homoskedastic).into()),
    );
}

/// Linear adjustment, HC1 sandwich, heteroskedastic noise. About 4 minutes.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn practitioner_linear_adjustment_hc1_interval() {
    estimator_coverage(
        "practitioner_linear_adjustment_hc1_interval",
        "analytic_se",
        Noise::Heteroskedastic,
        0x1700_0002,
        |_| Some(linear(AnalyticSeKind::Hc1).into()),
    );
}

/// Linear adjustment, cluster-robust standard error over 500 clusters. About 5 minutes.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn practitioner_linear_adjustment_cluster_interval() {
    estimator_coverage(
        "practitioner_linear_adjustment_cluster_interval",
        "analytic_se",
        Noise::Clustered,
        0x1700_0003,
        |replicate| {
            Some(
                linear(AnalyticSeKind::Cluster).with_cluster_ids(replicate.clusters.clone()).into(),
            )
        },
    );
}

/// Linear adjustment at the facade default of 199 bootstrap replicates (the bootstrap-SE
/// interval of an `estimate` call that chose nothing). About 1.5 hours: 199 refits per replicate.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn practitioner_linear_adjustment_bootstrap_interval() {
    estimator_coverage(
        "practitioner_linear_adjustment_bootstrap_interval",
        "bootstrap_se",
        Noise::Homoskedastic,
        0x1700_0004,
        |_| None,
    );
}

#[test]
#[ignore = "full-size bootstrap solver regression"]
fn practitioner_bootstrap_reuses_prepared_solver() {
    let study =
        Study::tabular(practitioner_study(MAX_ROWS, 0x1700_0004, Noise::Homoskedastic).data)
            .graph(practitioner_dag())
            .query(binary_ate())
            .refute(RefuteSuite::None)
            .build()
            .unwrap();
    let (result, traffic) = antecedent_stats::faer_backend::observe_solver_traffic(|| {
        study.run(&ExecutionContext::production(17, 1))
    });
    result.expect("the full-size study must report its bootstrap interval");
    assert!(
        traffic.design_gathers < 8,
        "199 bootstrap refits should reuse one prepared basis: {traffic:?}"
    );
}

/// Cross-fitted AIPW, influence-function standard error. About 40 minutes.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn practitioner_aipw_analytic_interval() {
    estimator_coverage(
        "practitioner_aipw_analytic_interval",
        "analytic_se",
        Noise::Homoskedastic,
        0x1700_0005,
        |_| Some(aipw(AnalyticSeKind::Homoskedastic).into()),
    );
}

/// AIPW, HC1 influence-function variance, heteroskedastic noise. About 40 minutes.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn practitioner_aipw_hc1_interval() {
    estimator_coverage(
        "practitioner_aipw_hc1_interval",
        "analytic_se",
        Noise::Heteroskedastic,
        0x1700_0006,
        |_| Some(aipw(AnalyticSeKind::Hc1).into()),
    );
}

/// AIPW with the cluster-robust influence-function variance over 500 clusters. About 40 minutes.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn practitioner_aipw_cluster_interval() {
    estimator_coverage(
        "practitioner_aipw_cluster_interval",
        "analytic_se",
        Noise::Clustered,
        0x1700_0007,
        |replicate| {
            Some(aipw(AnalyticSeKind::Cluster).with_cluster_ids(replicate.clusters.clone()).into())
        },
    );
}

// ================================================================== Bayesian and class posterior

/// The facade's default Bayesian configuration (Laplace, 1,000 draws, prior scale 10) on the
/// practitioner study, with the type-7 quantile interval of its draws. A request for fewer draws is
/// raised to 1,000 (`LAPLACE_MIN_DRAWS`), so it is this construction at this draw count. About
/// 30 minutes.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn practitioner_bayesian_laplace_interval() {
    let mut pair = tallies("practitioner_bayesian_laplace_interval", DGP, "posterior_quantile");
    let runs = map_replicates(n_sim(), |rep| {
        let seed = stream_seed(0x1700_0008, rep);
        let replicate = practitioner_study(rows(), seed, Noise::Homoskedastic);
        let study = Study::tabular(replicate.data)
            .graph(practitioner_dag())
            .query(binary_ate())
            .inference(InferenceMode::Bayesian(BayesianConfig::laplace()))
            .refute(RefuteSuite::None)
            .build()
            .ok()?;
        scored(&study, seed, |result| {
            result
                .posterior
                .as_ref()
                .and_then(antecedent::CausalPosterior::effect_column)
                .map_or([None, None], |column| posterior_pair(result, column))
        })
    });
    score(&mut pair, &runs, TRUTH);
    gate(&pair, &[None, None]);
}

/// Equal-tailed exchangeable-rank (type-6) interval of `draws` at `level`.
fn rank_interval(draws: &[f64], level: f64) -> (f64, f64) {
    let mut sorted = draws.to_vec();
    sorted.sort_by(f64::total_cmp);
    antecedent_stats::equal_tail_interval_sorted(
        &sorted,
        level,
        antecedent_stats::QuantileRule::ExchangeableRank,
    )
}

/// Reported (0.95) and gate-level (0.90) intervals of a Laplace effect posterior below the default
/// draw count: the exchangeable-rank intervals of its draws (the published one is checked).
fn rank_pair(result: &StudyResult) -> [Option<(f64, f64)>; 2] {
    let Some(posterior) = result.posterior.as_ref() else { return [None, None] };
    let Some(column) = posterior.effect_column() else { return [None, None] };
    let Ok(draws) = posterior.draws.column(column) else { return [None, None] };
    let reported = (posterior.summaries.q025[column], posterior.summaries.q975[column]);
    let rebuilt = rank_interval(draws, REPORTED_LEVEL);
    assert!(
        (rebuilt.0 - reported.0).abs() <= 1e-9 * (1.0 + reported.0.abs())
            && (rebuilt.1 - reported.1).abs() <= 1e-9 * (1.0 + reported.1.abs()),
        "published {reported:?} is not the exchangeable-rank interval {rebuilt:?}"
    );
    [Some(reported), Some(rank_interval(draws, GATE_LEVEL))]
}

/// The Laplace construction at 64 draws: the exchangeable-rank interval of its draws
/// (`laplace.gaussian_identity.prior_scale=10.rank_quantile_interval`), whose expected posterior
/// mass is 0.95 at every count where the type-7 interval's is 0.921. Practitioner scale. About
/// 30 minutes.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn practitioner_bayesian_laplace_64_draw_interval() {
    let mut pair =
        tallies("practitioner_bayesian_laplace_64_draw_interval", DGP, "posterior_quantile");
    let runs = map_replicates(n_sim(), |rep| {
        let seed = stream_seed(0x1700_000A, rep);
        let replicate = practitioner_study(rows(), seed, Noise::Homoskedastic);
        let study = Study::tabular(replicate.data)
            .graph(practitioner_dag())
            .query(binary_ate())
            .inference(InferenceMode::Bayesian(BayesianConfig::laplace().n_draws(64)))
            .refute(RefuteSuite::None)
            .build()
            .ok()?;
        scored(&study, seed, rank_pair)
    });
    score(&mut pair, &runs, TRUTH);
    gate(&pair, &[None, None]);
}

/// The same 64-draw construction on the ordinary small-sample grid (250, 500 and 1,000 rows of
/// `static_dgp::linear_ate_data`, as the default Laplace record). About 3 minutes.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn small_sample_bayesian_laplace_64_draw_interval() {
    let mut pair = tallies(
        "small_sample_bayesian_laplace_64_draw_interval",
        "crates/antecedent/tests/common/static_dgp.rs::linear_ate_data",
        "posterior_quantile",
    );
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(node(2), node(0)).unwrap();
    dag.insert_directed(node(2), node(1)).unwrap();
    dag.insert_directed(node(0), node(1)).unwrap();
    let runs = map_replicates(n_sim(), |rep| {
        let seed = stream_seed(0x1700_000B, rep);
        let study = Study::tabular(linear_ate_data(grid_n(500), seed))
            .graph(dag.clone())
            .query(binary_ate())
            .inference(InferenceMode::Bayesian(BayesianConfig::laplace().n_draws(64)))
            .refute(RefuteSuite::None)
            .build()
            .ok()?;
        scored(&study, seed, rank_pair)
    });
    score(&mut pair, &runs, 2.0);
    gate(&pair, &[None, None]);
}

fn node(i: u32) -> DenseNodeId {
    DenseNodeId::from_raw(i)
}

fn oriented(from_z: bool) -> Cpdag {
    let mut g = Dag::with_variables(3);
    g.insert_directed(node(0), node(1)).unwrap();
    if from_z {
        g.insert_directed(node(2), node(0)).unwrap();
        g.insert_directed(node(2), node(1)).unwrap();
    }
    Cpdag::from_dag(&g)
}

fn outcome_parent_cpdag() -> Cpdag {
    let mut g = Dag::with_variables(3);
    g.insert_directed(node(0), node(1)).unwrap();
    g.insert_directed(node(2), node(1)).unwrap();
    Cpdag::from_dag(&g)
}

fn class_posterior() -> GraphPosterior {
    let masks = [
        adjacency_mask_from_cpdag(&oriented(false)).unwrap(),
        adjacency_mask_from_cpdag(&outcome_parent_cpdag()).unwrap(),
    ];
    GraphPosterior::from_atoms(
        3,
        vec![0.6, 0.4],
        masks.to_vec(),
        vec![0.0; 9],
        vec![0.0; 9],
        SamplingSummary {
            ess: 1.0,
            diagnostics: InferenceDiagnostics::analytic("class_posterior"),
            rejected_invalid: 0,
        },
    )
    .unwrap()
    .with_atom_kind(GraphPosteriorAtomKind::Cpdag)
}

/// The CPDAG class graph posterior of `class_posterior_calibration` (two identified atoms that
/// share an empty adjustment) at practitioner scale: independent fair `T` and `Z`,
/// `Y = 2T + 2Z ± 0.2`, joint influence-function standard error. About 6 minutes.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn practitioner_class_posterior_interval() {
    const CLASS_TRUTH: f64 = 2.0;
    let mut pair = tallies("practitioner_class_posterior_interval", DGP, "analytic_se");
    let runs = map_replicates(n_sim(), |rep| {
        let seed = stream_seed(0x1700_0009, rep);
        let n = rows();
        let mut unif = uniform_from_state(seed | 1);
        let (mut t, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        for i in 0..n {
            z[i] = f64::from(u8::from(unif() < 0.5));
            t[i] = f64::from(u8::from(unif() < 0.5));
            let epsilon = if unif() < 0.5 { -0.2 } else { 0.2 };
            y[i] = 2.0 * t[i] + 2.0 * z[i] + epsilon;
        }
        let data = TabularData::from_f64_columns([
            ("t", t.as_slice()),
            ("y", y.as_slice()),
            ("z", z.as_slice()),
        ])
        .ok()?;
        let study = Study::tabular(data)
            .graph_posterior(class_posterior())
            .query(binary_ate())
            .inference(InferenceMode::Frequentist)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .ok()?;
        scored(&study, seed, scalar_normal_pair)
    });
    score(&mut pair, &runs, CLASS_TRUTH);
    gate(&pair, &[None, None]);
}

// ================================================================== temporal

const AR_RESIDUAL: Scenario = Scenario {
    label: "AR(1) rho=0.5, 1024 to 4096 steps",
    rho: 0.5,
    n: MAX_STEPS,
    seed: 0x1700_0100,
};

/// Bootstrap replicates per fit (the neighbouring temporal designs').
const BOOT: u32 = 100;

fn steps() -> usize {
    PRACTITIONER_STEPS.n(MAX_STEPS)
}

/// Coverage of the circular-block interval `ate ± z·se_bootstrap` of a Frequentist
/// `TemporalDag` effect: the gated 0.90 interval and the reported 0.95 interval from the same
/// replicates, as the neighbouring temporal designs score them.
fn temporal_coverage(test: &'static str, query: &antecedent_core::TemporalEffectQuery, truth: f64) {
    let two_step = query.horizon_steps >= 2;
    let scenario = Scenario {
        n: steps(),
        seed: AR_RESIDUAL.seed + u64::from(query.horizon_steps) * 1_000_000,
        ..AR_RESIDUAL
    };
    let key = RecordKey {
        test,
        dgp: "crates/antecedent/tests/common/driven_dgp.rs::pulse_series",
        interval: "circular_block_se",
    };
    let mut gated = CoverageTally::for_record(key, 0.9);
    let mut reported = CoverageTally::for_record(key, REPORTED_LEVEL).unasserted();
    let runs = map_replicates(n_sim(), |rep| {
        let study = Study::series(pulse_series(scenario, u32::try_from(rep).ok()?, two_step))
            .graph(pulse_dag(two_step))
            .query(CausalQuery::TemporalEffect(query.clone()))
            .inference(InferenceMode::Frequentist)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(BOOT)
            .build()
            .ok()?;
        let result = study.run(&ExecutionContext::for_tests(scenario.seed + rep)).ok()?;
        let contract = study.inspect().ok()?;
        let effect = result.estimate().as_effect()?;
        let (ate, se) = (effect.ate, effect.se_bootstrap());
        Some(Scored {
            reported: constructions(&contract, &result),
            intervals: [normal_interval(ate, se, Z95), normal_interval(ate, se, Z90)],
        })
    });
    for run in &runs {
        let Some(run) = run else {
            gated.skip();
            reported.skip();
            continue;
        };
        let [at_reported, at_gate] = run.intervals;
        if at_gate.is_some() {
            let (construction, scope, validation) = run
                .reported
                .iter()
                .find(|(construction, _, _)| construction.interval_method == "circular_block_se")
                .expect("the execution reported its circular-block interval");
            gated.bind_validation(construction, *scope, validation);
            reported.bind_validation(construction, *scope, validation);
        }
        gated.record(at_gate, truth);
        reported.record(at_reported, truth);
    }
    gated.assert();
    reported.emit();
}

/// Pulse effect at horizon 1 on 320 to 4,160 steps. About 25 minutes.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn practitioner_temporal_pulse_h1_interval() {
    temporal_coverage("practitioner_temporal_pulse_h1_interval", &pulse(1), BETA);
}

/// Pulse effect at horizon 2 (propagated through an intermediate). About 25 minutes.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn practitioner_temporal_pulse_h2_interval() {
    temporal_coverage(
        "practitioner_temporal_pulse_h2_interval",
        &pulse(2),
        common::driven_dgp::ALPHA * common::driven_dgp::DELTA,
    );
}

/// Single-step sustained effect. About 25 minutes.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn practitioner_temporal_sustained_interval() {
    temporal_coverage("practitioner_temporal_sustained_interval", &single_sustained(), BETA);
}

/// Default Bayesian `PulseEffect` on a `TemporalDag` (Laplace, 1,000 draws, prior scale 10) at
/// 320 to 4,160 steps, iid residuals as `calibration_temporal`'s default design. About 20 minutes.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn practitioner_temporal_bayesian_pulse_interval() {
    let mut pair = tallies(
        "practitioner_temporal_bayesian_pulse_interval",
        "crates/antecedent/tests/common/driven_dgp.rs::pulse_series",
        "posterior_quantile",
    );
    let scenario =
        Scenario { label: "iid, 320 to 4160 steps", rho: 0.0, n: steps(), seed: 0x1700_0200 };
    let runs = map_replicates(n_sim(), |rep| {
        let study = Study::series(pulse_series(scenario, u32::try_from(rep).ok()?, false))
            .graph(pulse_dag(false))
            .query(pulse(1))
            .inference(InferenceMode::Bayesian(BayesianConfig::laplace()))
            .refute(RefuteSuite::None)
            .build()
            .ok()?;
        scored(&study, scenario.seed + rep, |result| {
            result
                .posterior
                .as_ref()
                .and_then(antecedent::CausalPosterior::effect_column)
                .map_or([None, None], |column| posterior_pair(result, column))
        })
    });
    score(&mut pair, &runs, BETA);
    gate(&pair, &[None, None]);
}

/// The mediation contrasts, each executed as its own query with its own labelled record, the
/// truth each is scored against (`C + A·B`, `C`, `A·B`).
const CONTRASTS: [(&str, MediationContrast, f64); 3] = [
    ("total", MediationContrast::Total, C + A * B),
    ("direct", MediationContrast::Direct, C),
    ("mediated", MediationContrast::Mediated, A * B),
];

/// Frequentist temporal mediation on a `TemporalDag` at 320 to 4,160 steps: Total, Direct and
/// Mediated are three queries, each bound to its own labelled record through the circular-block
/// interval that query reports. The three executions of a replicate share one simulated series
/// and one run seed, so they share the block replicate; each tally scores one functional against
/// its own truth and is judged on its own marginal coverage, never combined into a joint claim.
/// About 45 minutes.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn practitioner_temporal_mediation_interval() {
    let test = "practitioner_temporal_mediation_interval";
    let dgp = "crates/antecedent/tests/common/driven_dgp.rs::mediation_series";
    let key = RecordKey { test, dgp, interval: "circular_block_se" };
    let mut gated: Vec<CoverageTally> = CONTRASTS
        .iter()
        .map(|(name, ..)| CoverageTally::for_record(key, 0.9).labelled(*name))
        .collect();
    let mut reported: Vec<CoverageTally> = CONTRASTS
        .iter()
        .map(|(name, ..)| {
            CoverageTally::for_record(key, REPORTED_LEVEL).labelled(*name).unasserted()
        })
        .collect();
    let scenario = Scenario {
        label: "AR(1) rho=0.5, 320 to 4160 steps",
        rho: 0.5,
        n: steps(),
        seed: 0x1700_0300,
    };
    let runs = map_replicates(n_sim(), |rep| {
        let series = mediation_series(scenario, u32::try_from(rep).ok()?);
        let mut contrasts = Vec::with_capacity(CONTRASTS.len());
        for (_, contrast, _) in CONTRASTS {
            let study = Study::series(series.clone())
                .graph(mediation_dag())
                .query(mediation_query(contrast))
                .inference(InferenceMode::Frequentist)
                .refute(RefuteSuite::None)
                .bootstrap_replicates(BOOT)
                .build()
                .ok()?;
            let result = study.run(&ExecutionContext::for_tests(scenario.seed + rep)).ok()?;
            let contract = study.inspect().ok()?;
            let slice = result.mediation_grid()?.slices.first()?;
            let TemporalMediationUncertainty::FrequentistBlockBootstrap { block, .. } =
                &slice.uncertainty
            else {
                return None;
            };
            let (point, se) = match contrast {
                MediationContrast::Total => (slice.estimate.total, block.total),
                MediationContrast::Direct => (slice.estimate.direct, block.direct),
                MediationContrast::Mediated => (slice.estimate.mediated, block.mediated),
                MediationContrast::NaturalDirect | MediationContrast::NaturalIndirect => {
                    unreachable!("the design scores the Total, Direct and Mediated contrasts")
                }
            };
            let point = point?;
            contrasts.push(Scored {
                reported: constructions(&contract, &result),
                intervals: [normal_interval(point, se, Z95), normal_interval(point, se, Z90)],
            });
        }
        Some(contrasts)
    });
    for replicate in &runs {
        for (i, (_, _, truth)) in CONTRASTS.iter().enumerate() {
            let Some(contrasts) = replicate else {
                gated[i].skip();
                reported[i].skip();
                continue;
            };
            let [at_reported, at_gate] = contrasts[i].intervals;
            if at_gate.is_some() {
                let (construction, scope, validation) = contrasts[i]
                    .reported
                    .iter()
                    .find(|(construction, _, _)| {
                        construction.interval_method == "circular_block_se"
                    })
                    .expect("the execution reported its circular-block interval");
                gated[i].bind_validation(construction, *scope, validation);
                reported[i].bind_validation(construction, *scope, validation);
            }
            gated[i].record(at_gate, *truth);
            reported[i].record(at_reported, *truth);
        }
    }
    let failures: Vec<String> = gated
        .iter()
        .filter_map(|tally| {
            std::panic::catch_unwind(|| tally.assert()).err().map(|e| {
                e.downcast_ref::<String>().cloned().unwrap_or_else(|| "coverage failure".into())
            })
        })
        .collect();
    for tally in &reported {
        tally.emit();
    }
    assert!(failures.is_empty(), "{}", failures.join("; "));
}

/// Total, Direct and Mediated contrasts of the Bayesian temporal mediation posterior (default
/// Laplace, 1,000 draws) at 320 to 4,160 steps, iid residuals. Each contrast is its own query and
/// its own labelled record, scored against its own truth on its own marginal coverage; the three
/// executions of a replicate share one simulated series and are never combined into a joint
/// claim. About 40 minutes.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn practitioner_temporal_bayesian_mediation_interval() {
    let key = RecordKey {
        test: "practitioner_temporal_bayesian_mediation_interval",
        dgp: "crates/antecedent/tests/common/driven_dgp.rs::mediation_series",
        interval: "posterior_quantile",
    };
    let mut gated: Vec<CoverageTally> = CONTRASTS
        .iter()
        .map(|(name, ..)| CoverageTally::for_record(key, GATE_LEVEL).labelled(*name))
        .collect();
    let mut reported: Vec<CoverageTally> = CONTRASTS
        .iter()
        .map(|(name, ..)| {
            CoverageTally::for_record(key, REPORTED_LEVEL).labelled(*name).unasserted()
        })
        .collect();
    let scenario =
        Scenario { label: "iid, 320 to 4160 steps", rho: 0.0, n: steps(), seed: 0x1700_0400 };
    let runs = map_replicates(n_sim(), |rep| {
        let series = mediation_series(scenario, u32::try_from(rep).ok()?);
        let mut contrasts = Vec::with_capacity(CONTRASTS.len());
        for (_, contrast, _) in CONTRASTS {
            let study = Study::series(series.clone())
                .graph(mediation_dag())
                .query(mediation_query(contrast))
                .inference(InferenceMode::Bayesian(BayesianConfig::laplace()))
                .refute(RefuteSuite::None)
                .bootstrap_replicates(0)
                .build()
                .ok()?;
            let result = study.run(&ExecutionContext::for_tests(scenario.seed + rep)).ok()?;
            let contract = study.inspect().ok()?;
            let posterior = result.posterior.as_ref()?;
            let draws = posterior.draws.column(posterior.effect_column()?).ok()?;
            contrasts.push(Scored {
                reported: constructions(&contract, &result),
                intervals: [
                    quantile_interval(draws, REPORTED_LEVEL),
                    quantile_interval(draws, GATE_LEVEL),
                ],
            });
        }
        Some(contrasts)
    });
    for replicate in &runs {
        for (i, (_, _, truth)) in CONTRASTS.iter().enumerate() {
            let Some(contrasts) = replicate else {
                gated[i].skip();
                reported[i].skip();
                continue;
            };
            let [at_reported, at_gate] = contrasts[i].intervals;
            let (construction, scope, validation) = contrasts[i]
                .reported
                .iter()
                .find(|(construction, _, _)| construction.interval_method == "posterior_quantile")
                .expect("the execution reported its posterior interval");
            gated[i].bind_validation(construction, *scope, validation);
            reported[i].bind_validation(construction, *scope, validation);
            gated[i].record(at_gate, *truth);
            reported[i].record(at_reported, *truth);
        }
    }
    let failures: Vec<String> = gated
        .iter()
        .filter_map(|tally| {
            std::panic::catch_unwind(|| tally.assert()).err().map(|e| {
                e.downcast_ref::<String>().cloned().unwrap_or_else(|| "coverage failure".into())
            })
        })
        .collect();
    for tally in &reported {
        tally.emit();
    }
    assert!(failures.is_empty(), "{}", failures.join("; "));
}

// ================================================================== wiring (not ignored)

/// The grids span the published practitioner sizes, and the largest point is the published limit.
#[test]
fn the_grids_span_the_published_limits() {
    let rows: Vec<usize> = (0..3).map(|point| PRACTITIONER_ROWS.n_at(point, MAX_ROWS)).collect();
    assert_eq!(rows, [1_000, 10_000, 100_000]);
    let steps: Vec<usize> = (0..3).map(|point| PRACTITIONER_STEPS.n_at(point, MAX_STEPS)).collect();
    assert_eq!(steps, [320, 1_280, 4_160]);
    // A 1,024-step series with three lags has 1,021 analysis rows, a 4,096-step series 4,093:
    // both inside 319..=4,159, the range the temporal records can span.
    assert!(steps[0] <= 1_022 && 4_094 <= steps[2]);
}

/// The practitioner study generates the stated law: treated share between 0.3 and 0.7, the
/// clusters partition the rows evenly, and the effect is recoverable by regression.
#[test]
fn the_practitioner_study_is_a_well_posed_design() {
    let study = practitioner_study(20_000, 17, Noise::Clustered);
    assert_eq!(study.clusters.len(), 20_000);
    assert_eq!(study.clusters.iter().copied().max(), Some(CLUSTERS as u32 - 1));
    let result = Study::tabular(study.data)
        .graph(practitioner_dag())
        .query(binary_ate())
        .estimator(linear(AnalyticSeKind::Cluster).with_cluster_ids(study.clusters))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(17))
        .unwrap();
    let effect = result.estimate().as_effect().unwrap();
    assert!(effect.se_analytic() > 0.0);
    // The estimate sits within five analytic standard errors of the generating effect (the
    // cluster-level treatment shift has mean zero), not merely within a fixed 0.15.
    assert!(
        (effect.ate - TRUTH).abs() < 5.0 * effect.se_analytic(),
        "ate {} se {}",
        effect.ate,
        effect.se_analytic()
    );
}

/// A Laplace request for 64 draws keeps its draws, mean and sd; its effect interval is the
/// exchangeable-rank interval of those draws, not their type-7 interval, and a 1,000-draw run is
/// unchanged.
#[test]
fn a_64_draw_laplace_request_publishes_the_exchangeable_rank_interval() {
    let run = |draws: usize| {
        let replicate = practitioner_study(1_000, 3, Noise::Homoskedastic);
        let study = Study::tabular(replicate.data)
            .graph(practitioner_dag())
            .query(binary_ate())
            .inference(InferenceMode::Bayesian(BayesianConfig::laplace().n_draws(draws)))
            .refute(RefuteSuite::None)
            .build()
            .unwrap();
        study.run(&ExecutionContext::for_tests(3)).unwrap()
    };
    let small = run(64);
    let posterior = small.posterior.as_ref().unwrap();
    assert_eq!(posterior.draws.n_draws, 64);
    let column = posterior.effect_column().unwrap();
    let (lower, upper) = rank_interval(posterior.draws.column(column).unwrap(), 0.95);
    // Exchangeable-rank (type 6) by hand: position (n + 1) p = 65 * 0.025 = 1.625 and
    // 65 * 0.975 = 63.375 (1-based) interpolate the order statistics linearly.
    let mut sorted = posterior.draws.column(column).unwrap().to_vec();
    sorted.sort_by(f64::total_cmp);
    let by_hand = (
        sorted[0] + 0.625 * (sorted[1] - sorted[0]),
        sorted[62] + 0.375 * (sorted[63] - sorted[62]),
    );
    assert!((by_hand.0 - lower).abs() < 1e-12 && (by_hand.1 - upper).abs() < 1e-12);
    assert_eq!(posterior.summaries.q025[column].to_bits(), lower.to_bits());
    assert_eq!(posterior.summaries.q975[column].to_bits(), upper.to_bits());
    let own = posterior.draws.summarize();
    assert_eq!(posterior.summaries.mean[column].to_bits(), own.mean[column].to_bits());
    assert_ne!(posterior.summaries.q025[column].to_bits(), own.q025[column].to_bits());
    let full = run(1_000);
    let posterior = full.posterior.as_ref().unwrap();
    assert_eq!(posterior.summaries, posterior.draws.summarize());
}
