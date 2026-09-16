//! 1.10 repeated-sampling coverage of Frequentist `TemporalMediationEffect` on
//! a `TemporalCpdag` (explicit and accepted, validation none / cheap / full).
//!
//! The class of `common::fixtures::mediation_cpdag_two` has two completions
//! (`z@1 -> w@1`, `w@1 -> z@1`) that necessarily fit the same mediation design:
//! the same mediator and outcome parents and the same `S(h)`. When every
//! completion is identified and fits one design, the completion identified
//! set is a single contrast; at one horizon that contrast is the result's own
//! estimate under the design's circular-block SE (the `TemporalDag` route's
//! construction on the same stream), published beside the identified set. The
//! reported interval is `effect ± z·se_bootstrap` at
//! the Study's default 199 replicates; it is scored against the path-product
//! truth `fixtures::mediation_truth()` at 0.95 and at 0.90 from the same
//! replicates. Every tally is keyed through [`keyed`], this file's one
//! emission point; the construction each record describes is read from the
//! execution's own contract rather than declared there.
//!
//! The file also measures the interval an ordinary default Bayesian
//! `PulseEffect` on a `TemporalDag` reports: the facade's default Bayesian
//! configuration (Laplace backend, 1000 draws, prior scale 10 — Python
//! `Bayesian()`), whose posterior-quantile interval is scored at 0.95 and 0.90
//! on the driven-treatment Pulse law `common::driven_dgp::pulse_series` (iid
//! residuals, truth `BETA`).
//!
//! Ignored tests run via `scripts/gate_calibration.sh` (release build).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::doc_markdown, clippy::float_cmp)]

mod common;

use antecedent::{AcceptedGraph, BayesianConfig, InferenceMode, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    CausalQuery, ExecutionContext, MediationContrast, MediationQuery, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_estimate::{CircularBlockFamily, TemporalMediationUncertainty};
use common::calibration::{CoverageTally, RecordKey, grid_n, n_sim};
use common::calibration_bind::{bind_all, constructions};
use common::driven_dgp::{BETA, Scenario, pulse, pulse_dag, pulse_series};
use common::fixtures::{self, mediation_cpdag_two, mediation_series};
use common::reported::{
    GATE_LEVEL, REPORTED_LEVEL, gate, normal_at, posterior_pair, record_pair, skip_pair,
};

const N: usize = 160;

/// This file's single coverage-record emission point. The construction a
/// record describes is read from the execution the tally scores (`bind`), not
/// declared here; `label` separates the two mediator-confounding designs the
/// same test name would otherwise collide on.
fn keyed(test: &'static str, label: &str, level: f64) -> CoverageTally {
    keyed_record(RecordKey { test, dgp: "mediation_series", interval: "circular_block_se" }, level)
        .labelled(label.to_owned())
}

fn keyed_record(key: RecordKey, level: f64) -> CoverageTally {
    CoverageTally::for_record(key, level)
}

fn mediated_query() -> CausalQuery {
    CausalQuery::Mediation(
        MediationQuery::binary(
            VariableId::from_raw(0),
            VariableId::from_raw(2),
            [VariableId::from_raw(1)],
            MediationContrast::Mediated,
        )
        .with_horizons(vec![1])
        .unwrap(),
    )
}

/// The executed study beside its result: [`bind_all`] reads the construction
/// from the study's own contract, so a record cannot describe another one.
fn run(
    data: TimeSeriesData,
    accepted: bool,
    suite: RefuteSuite,
    seed: u64,
) -> Option<(Study, StudyResult)> {
    let builder = Study::series(data);
    let builder = if accepted {
        builder.graph(AcceptedGraph::temporal_cpdag(mediation_cpdag_two()).ok()?)
    } else {
        builder.graph(mediation_cpdag_two())
    };
    let study = builder.query(mediated_query()).refute(suite).build().ok()?;
    let result = study.run(&ExecutionContext::for_tests(seed)).ok()?;
    Some((study, result))
}

/// `(effect, block SE)` of the single class slice.
fn slice_effect(result: &StudyResult) -> Option<(f64, f64)> {
    let slice = result.mediation_grid.as_ref()?.slices.first()?;
    match slice.uncertainty {
        TemporalMediationUncertainty::FrequentistBlockBootstrap { requested: Some(se), .. } => {
            Some((slice.estimate.effect.ate, se))
        }
        _ => None,
    }
}

fn pair(result: &StudyResult) -> [Option<(f64, f64)>; 2] {
    slice_effect(result).map_or([None, None], |(effect, se)| {
        [normal_at(effect, se, REPORTED_LEVEL), normal_at(effect, se, GATE_LEVEL)]
    })
}

fn mediation_coverage(test: &'static str, label: &str, kappa: f64, seed_base: u64) {
    let truth = fixtures::mediation_truth();
    let mut tallies = [keyed(test, label, REPORTED_LEVEL), keyed(test, label, GATE_LEVEL)];
    for rep in 0..u64::from(n_sim()) {
        let seed = seed_base + rep;
        let data = mediation_series(grid_n(N), kappa, seed);
        let Some((study, result)) = run(data.clone(), false, RefuteSuite::None, rep) else {
            skip_pair(&mut tallies);
            continue;
        };
        let intervals = pair(&result);
        if rep == 0 {
            assert!(intervals[0].is_some(), "agreeing completions must publish the block interval");
            for (accepted, suite) in
                [(true, RefuteSuite::None), (false, RefuteSuite::Cheap), (false, RefuteSuite::Full)]
            {
                let (_, other) =
                    run(data.clone(), accepted, suite, rep).expect("licensed coordinate");
                assert_eq!(pair(&other), intervals, "accepted={accepted} {suite:?}");
            }
        }
        let [first, second] = &mut tallies;
        bind_all(&mut [first, second], &study, &result);
        record_pair(&mut tallies, intervals, truth);
    }
    gate(&tallies, &[None, None]);
}

/// Mediator–outcome confounding through `z[t-1] -> w[t-1] -> y`
/// (`kappa = MED_KAPPA`) that does not confound `t -> y`.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn temporal_cpdag_mediation_frequentist_confounded_nominal_coverage() {
    mediation_coverage(
        "temporal_cpdag_mediation_frequentist_confounded_nominal_coverage",
        "kappa=MED_KAPPA",
        fixtures::MED_KAPPA,
        0x110_0401,
    );
}

/// The same class without mediator–outcome confounding (`kappa = 0`).
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn temporal_cpdag_mediation_frequentist_unconfounded_nominal_coverage() {
    mediation_coverage(
        "temporal_cpdag_mediation_frequentist_unconfounded_nominal_coverage",
        "kappa=0",
        0.0,
        0x110_0402 << 20,
    );
}

/// Driven-treatment Pulse law, iid residuals, base length 160 (grid 80, 160, 320).
const DEFAULT_PULSE: Scenario = Scenario { label: "iid n=160", rho: 0.0, n: N, seed: 0x110_0450 };

/// The facade's default Bayesian `PulseEffect` on a `TemporalDag`.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn pulse_effect_temporal_dag_bayesian_default_nominal_coverage() {
    let key = RecordKey {
        test: "pulse_effect_temporal_dag_bayesian_default_nominal_coverage",
        dgp: "crates/antecedent/tests/common/driven_dgp.rs::pulse_series",
        interval: "posterior_quantile",
    };
    let mut tallies = [keyed_record(key, REPORTED_LEVEL), keyed_record(key, GATE_LEVEL)];
    let scenario = Scenario { n: grid_n(DEFAULT_PULSE.n), ..DEFAULT_PULSE };
    for rep in 0..n_sim() {
        let seed = scenario.seed + u64::from(rep);
        let study = Study::series(pulse_series(scenario, rep, false))
            .graph(pulse_dag(false))
            .query(pulse(1))
            .inference(InferenceMode::Bayesian(BayesianConfig::laplace()))
            .refute(RefuteSuite::None)
            .build()
            .expect("the default Bayesian study builds");
        let Ok(result) = study.run(&ExecutionContext::for_tests(seed)) else {
            skip_pair(&mut tallies);
            continue;
        };
        let Some(column) =
            result.posterior.as_ref().and_then(antecedent::CausalPosterior::effect_column)
        else {
            skip_pair(&mut tallies);
            continue;
        };
        let intervals = posterior_pair(&result, column);
        let [first, second] = &mut tallies;
        bind_all(&mut [first, second], &study, &result);
        record_pair(&mut tallies, intervals, BETA);
    }
    gate(&tallies, &[None, None]);
}

/// The class slice's interval is the `TemporalDag` route's interval on the
/// completions' shared design: same point, same circular-block SE.
#[test]
fn temporal_cpdag_mediation_slice_interval_is_the_shared_design_interval() {
    let data = mediation_series(N, fixtures::MED_KAPPA, 11);
    let (_, class) = run(data.clone(), false, RefuteSuite::None, 3).unwrap();
    let (effect, se) = slice_effect(&class).expect("block interval on the class slice");
    let dag = Study::series(data)
        .graph(fixtures::mediation_dag())
        .query(mediated_query())
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(3))
        .unwrap();
    assert_eq!(effect, dag.estimate.ate, "same contrast");
    assert_eq!(Some(se), dag.estimate.se_bootstrap, "same circular-block SE");
}

/// The shared contrast is the class result's **own** estimate, so the reported
/// interval is the mediation circular-block SE a consumer can read, and the
/// identified set stays beside it. A class whose completion enumeration was
/// capped is not one contrast and keeps the unbanded set.
#[test]
fn temporal_cpdag_mediation_class_reports_the_circular_block_interval() {
    let data = mediation_series(N, fixtures::MED_KAPPA, 11);
    let (study, result) = run(data.clone(), false, RefuteSuite::None, 3).unwrap();
    let (effect, se) = slice_effect(&result).expect("block interval on the class slice");
    assert_eq!(result.estimate.ate, effect, "the shared contrast is the result's estimate");
    assert_eq!(result.estimate.se_bootstrap, Some(se), "under the shared circular-block SE");
    assert_eq!(
        result.estimate.block_family,
        Some(CircularBlockFamily::Mediation),
        "keyed to the mediation block family"
    );
    assert!(result.mediation.is_some(), "the mediation record travels with the scalar");
    let contract = study.inspect().expect("inspect");
    let methods: Vec<String> = constructions(&contract, &result)
        .into_iter()
        .map(|(construction, _)| construction.interval_method)
        .collect();
    assert_eq!(methods, ["circular_block_se"], "the class publishes a readable interval");
    let set = result.mediation_grid.as_ref().expect("grid").slices[0]
        .identified_set
        .expect("identified set");
    assert_eq!((set.lower, set.upper), (effect, effect), "the set is that one contrast");

    // Capped enumeration: the examined completions are not the whole class, so
    // no single contrast is published and the set stays unbanded.
    let capped_study = Study::series(data)
        .graph(mediation_cpdag_two())
        .query(mediated_query())
        .refute(RefuteSuite::None)
        .max_completions(1)
        .build()
        .unwrap();
    let capped = capped_study.run(&ExecutionContext::for_tests(3)).unwrap();
    assert!(capped.estimate.ate.is_nan(), "a capped class publishes no scalar");
    assert!(capped.mediation.is_none());
    let capped_contract = capped_study.inspect().expect("inspect");
    let capped_methods: Vec<String> = constructions(&capped_contract, &capped)
        .into_iter()
        .map(|(construction, _)| construction.interval_method)
        .collect();
    assert_eq!(capped_methods, ["none"], "a capped class reports no interval");
    assert!(
        capped
            .structural_response
            .as_ref()
            .is_some_and(|structural| structural.identified_set.is_some()),
        "the unbanded identified set is still reported"
    );
}
