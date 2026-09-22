//! Repeated-sampling coverage of the ADMG front-door coordinates at the
//! facade level: the Frequentist and Bayesian `functional.effect` average
//! effect and the Bayesian `functional.distribution` interventional
//! distribution, on explicit and accepted structure.
//!
//! Binary front-door law with a latent `u` confounding `t` and `y`
//! (columns `t, m, y`; ADMG `t -> m -> y`, `t <-> y`):
//! `u ~ Bern(1/2)`, `t | u ~ Bern(0.3 + 0.4u)`, `m | t ~ Bern(0.2 + 0.6t)`,
//! `y | m, u ~ Bern(0.1 + 0.4m + 0.3u)`. Then `E[Y | do(M = m)] = 0.25 + 0.4m`,
//! `P(Y = 1 | do(T = t)) = 0.25 + 0.4·(0.2 + 0.6t)`, so
//! `P(Y = 1 | do(T = 1)) = 0.57` and the ATE is `0.24`.
//!
//! Every test scores exactly the interval the study reports by default (the
//! Study's default bootstrap for the Frequentist effect, the facade's default
//! Bayesian configuration) at 0.95, and the same construction at 0.90 from the
//! same replicates (`common::reported`). Replicate 0 checks that accepted
//! structure and the cheap / full validation suites report the identical
//! interval, so one measurement covers every licensed structure × validation
//! coordinate. Every tally is keyed through [`keyed`], this file's one
//! emission point: the key states only the emitting test, the DGP and the
//! scored interval method, and the construction each record describes is read
//! from the runtime by binding every scored execution
//! (`common::calibration_bind`).
//!
//! Ignored tests run via `scripts/gate_calibration.sh` (release build).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::doc_markdown, clippy::too_many_arguments)]
#![allow(
    clippy::float_cmp,
    reason = "test scaffolding compares exact constants and indexes with small literals"
)]

mod common;

use antecedent::{AcceptedGraph, BayesianConfig, InferenceMode, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ExecutionContext, Intervention,
    InterventionalDistributionQuery, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Admg, DenseNodeId};
use common::calibration::{
    CoverageTally, RecordKey, SampleGrid, map_replicates, n_sim, stream_seed,
};
use common::calibration_bind::bind_all;
use common::reported::{
    GATE_LEVEL, REPORTED_LEVEL, gate, gate_at, posterior_pair, record_pair, scalar_normal_pair,
    scalar_reported_se, skip_pair,
};

// ---------------------------------------------------------------- emission

/// The coordinate a test measures: what it asserts about the plan, and which
/// reported interval its records score. The construction itself is never
/// declared here — it is bound from each execution.
#[derive(Clone, Copy)]
struct Cell {
    query: &'static str,
    estimator: &'static str,
    interval_method: &'static str,
}

const N: usize = 1000;
const DGP: &str = "frontdoor_data";

/// This file's single coverage-record emission point: `test` is the emitting
/// `#[test] fn`'s own name, and the record's construction is read from the
/// runtime when the scored executions are bound.
fn keyed(test: &'static str, cell: Cell, level: f64) -> CoverageTally {
    CoverageTally::for_record(RecordKey { test, dgp: DGP, interval: cell.interval_method }, level)
}

fn keyed_pair(test: &'static str, cell: Cell) -> [CoverageTally; 2] {
    [keyed(test, cell, REPORTED_LEVEL), keyed(test, cell, GATE_LEVEL)]
}

// ---------------------------------------------------------------- DGP

const ATE_TRUTH: f64 = 0.24;
const P_Y1_DO_T1: f64 = 0.57;

/// This suite's uniform stream. The tag names the stream; the body is the
/// harness's one LCG, so the draws are unchanged.
fn uniform(seed: u64) -> impl FnMut() -> f64 {
    common::calibration::uniform(seed, 0xF0_0D00_0F0D)
}

fn frontdoor_data(n: usize, seed: u64) -> TabularData {
    let mut u01 = uniform(seed);
    let (mut t, mut m, mut y) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let u = f64::from(u01() < 0.5);
        t[i] = f64::from(u01() < 0.3 + 0.4 * u);
        m[i] = f64::from(u01() < 0.2 + 0.6 * t[i]);
        y[i] = f64::from(u01() < 0.1 + 0.4 * m[i] + 0.3 * u);
    }
    TabularData::from_f64_columns([("t", t.as_slice()), ("m", m.as_slice()), ("y", y.as_slice())])
        .unwrap()
}

fn frontdoor_admg() -> Admg {
    let mut g = Admg::with_variables(3);
    g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    g.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    g.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    g
}

/// The facade's default Bayesian configuration (Python `Bayesian()`).
fn bayes() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::laplace())
}

/// Build and run the study, returning it beside its result so the scored
/// execution can be bound to a record-keyed tally.
fn run(
    data: &TabularData,
    query: CausalQuery,
    inference: InferenceMode,
    accepted: bool,
    suite: RefuteSuite,
    seed: u64,
) -> Option<(Study, StudyResult)> {
    let builder = Study::tabular(data.clone());
    let builder = if accepted {
        builder.graph(AcceptedGraph::from(frontdoor_admg()))
    } else {
        builder.graph(frontdoor_admg())
    };
    let study = builder.query(query).inference(inference).refute(suite).build().ok()?;
    let result = study.run(&ExecutionContext::for_tests(seed)).ok()?;
    Some((study, result))
}

fn ate_query() -> CausalQuery {
    CausalQuery::AverageEffect(AverageEffectQuery::with_levels(
        VariableId::from_raw(0),
        VariableId::from_raw(2),
        0.0,
        1.0,
    ))
}

fn distribution_query() -> CausalQuery {
    CausalQuery::Distribution(InterventionalDistributionQuery::new(
        VariableId::from_raw(2),
        [Intervention::set(VariableId::from_raw(0), Value::f64(1.0))],
    ))
}

type Extract = fn(&StudyResult) -> [Option<(f64, f64)>; 2];

/// Score `extract(result)` against `truth`; replicate 0 checks the licensed
/// structure × validation coordinates report the same interval.
fn coverage(
    test: &'static str,
    cell: Cell,
    query: fn() -> CausalQuery,
    inference: fn() -> InferenceMode,
    suites: &[RefuteSuite],
    extract: Extract,
    truth: f64,
    family: u64,
) -> [CoverageTally; 2] {
    let mut tallies = keyed_pair(test, cell);
    let runs = map_replicates(n_sim(), |rep| {
        let seed = stream_seed(family, rep);
        let data = frontdoor_data(SampleGrid::HEAVY.n(N), seed);
        let (study, result) = run(&data, query(), inference(), false, RefuteSuite::None, seed)?;
        let pair = extract(&result);
        if rep == 0 {
            assert_eq!(result.logical_plan.estimator.as_deref(), Some(cell.estimator));
            assert!(pair[0].is_some(), "{}: the default run must report an interval", cell.query);
            for (accepted, suite) in
                std::iter::once((true, RefuteSuite::None)).chain(suites.iter().map(|&s| (false, s)))
            {
                let (_, other) = run(&data, query(), inference(), accepted, suite, seed)
                    .expect("licensed coordinate runs");
                assert_eq!(extract(&other), pair, "accepted={accepted} {suite:?}");
            }
        }
        Some((study, result, pair))
    });
    for scored in &runs {
        let Some((study, result, pair)) = scored else {
            skip_pair(&mut tallies);
            continue;
        };
        let [reported, gated] = &mut tallies;
        bind_all(&mut [reported, gated], study, result);
        record_pair(&mut tallies, *pair, truth);
    }
    tallies
}

/// The Frequentist `functional.effect` result carries only the bootstrap SE
/// (its analytic SE is NaN), so the reported interval is `ate ± z·se_boot`
/// at the Study's default 199 replicates.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn average_effect_admg_frontdoor_frequentist_nominal_coverage() {
    let cell = Cell {
        query: "AverageEffect",
        estimator: "functional.effect",
        interval_method: "bootstrap_se",
    };
    let tallies = coverage(
        "average_effect_admg_frontdoor_frequentist_nominal_coverage",
        cell,
        ate_query,
        || InferenceMode::Frequentist,
        &[RefuteSuite::Cheap, RefuteSuite::Full],
        |result| {
            assert_eq!(scalar_reported_se(result).1, "bootstrap_se", "bootstrap SE reported");
            scalar_normal_pair(result)
        },
        ATE_TRUTH,
        0x110_0101,
    );
    gate(&tallies, &[None, None]);
}

/// Bayesian `functional.effect` posterior (Bayesian-bootstrap Dirichlet row
/// law over the identified CPT factors): the published `q025` / `q975`.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn average_effect_admg_frontdoor_bayesian_nominal_coverage() {
    let cell = Cell {
        query: "AverageEffect",
        estimator: "functional.effect",
        interval_method: "posterior_quantile",
    };
    let tallies = coverage(
        "average_effect_admg_frontdoor_bayesian_nominal_coverage",
        cell,
        ate_query,
        bayes,
        &[RefuteSuite::Cheap, RefuteSuite::Full],
        |result| {
            let col =
                result.posterior.as_ref().and_then(antecedent::CausalPosterior::effect_column);
            col.map_or([None, None], |col| posterior_pair(result, col))
        },
        ATE_TRUTH,
        0x110_0102,
    );
    gate_at(&tallies, &[[Some(0.936), None, None], [Some(0.878), None, None]]);
}

/// Bayesian `functional.distribution` `P(Y = 1 | do(T = 1))`: the effect
/// column is the binary mean, the `Y = 1` atom (checked in replicate 0 of the
/// shape test).
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn interventional_distribution_admg_frontdoor_bayesian_nominal_coverage() {
    let cell = Cell {
        query: "InterventionalDistribution",
        estimator: "functional.distribution",
        interval_method: "posterior_quantile",
    };
    let tallies = coverage(
        "interventional_distribution_admg_frontdoor_bayesian_nominal_coverage",
        cell,
        distribution_query,
        bayes,
        &[],
        |result| {
            let col =
                result.posterior.as_ref().and_then(antecedent::CausalPosterior::effect_column);
            col.map_or([None, None], |col| posterior_pair(result, col))
        },
        P_Y1_DO_T1,
        0x110_0103,
    );
    gate(&tallies, &[None, None]);
}

/// Each coordinate publishes the interval its coverage test scores.
#[test]
fn admg_frontdoor_coordinates_publish_the_scored_interval() {
    let data = frontdoor_data(600, 3);
    let (_, freq) =
        run(&data, ate_query(), InferenceMode::Frequentist, false, RefuteSuite::None, 3)
            .expect("the Frequentist coordinate runs");
    assert!(freq.estimate.se_analytic.is_nan(), "functional.effect has no analytic SE");
    assert_eq!(freq.estimate.bootstrap_replicates_ok, Some(199), "Study default bootstrap");
    assert!(scalar_normal_pair(&freq)[0].is_some());
    let (_, dist) = run(&data, distribution_query(), bayes(), false, RefuteSuite::None, 3)
        .expect("the Bayesian distribution coordinate runs");
    let posterior = dist.posterior.as_ref().unwrap();
    let col = posterior.effect_column().unwrap();
    let atoms = &dist.distribution.as_ref().unwrap().atoms;
    let one = atoms.iter().position(|a| a.outcomes[0].1.as_f64() == Some(1.0)).unwrap();
    assert_eq!(
        posterior.draws.column(col).unwrap(),
        posterior.draws.column(col + 1 + one).unwrap(),
        "the effect column is the P(Y = 1) atom"
    );
    assert!(posterior_pair(&dist, col)[0].is_some());
}
