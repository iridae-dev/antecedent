//! Calibration evidence measured over a sample-size grid binds to ordinary
//! default executions.
//!
//! Each fixture record below has the shape `scripts/collect_coverage_records.py`
//! writes for a construction measured at the three points of its
//! `SampleGrid`: the construction key, one measurement per grid point, and the
//! measured range `n_min..n_max` spanning the points. The keys are the ones a
//! default study reports — a Frequentist DAG `AverageEffect` with no estimator
//! or bootstrap chosen, the same study with the facade's default Bayesian
//! configuration, and a Frequentist `TemporalDag` `PulseEffect` — so a
//! divergence between what these runs key under and what the grid measures
//! fails here. The coverages are the 0.95-level values a wiring smoke run
//! printed (100 replicates per point; `mcse` is a placeholder) and are not
//! registry evidence: the assertions depend only on the keys, the grid's row
//! counts and each point's boundary flag.
//!
//! Every study runs end to end (`Study::run` -> `StudyResult::calibration_bases`
//! -> `antecedent_io::calibration::calibration_slot_in`), the path a claim takes,
//! against this fixture registry instead of the generated one.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study, StudyResult};
use antecedent_core::{AverageEffectQuery, ExecutionContext, VariableId};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_io::calibration::{
    BOUNDARY_RECORD, CalibrationBasisWire, SAMPLE_SIZE_OUTSIDE_MEASURED_RANGE, calibration_slot_in,
};
use antecedent_io::contract_section::CalibrationSlotWire;
use antecedent_io::coverage_records_data::{CoverageGridPoint, CoverageRecord};
use common::driven_dgp::{Scenario, pulse, pulse_dag, pulse_series};
use common::static_dgp::linear_ate_data;

const fn point(k: u8, n: u64, observed: f64, boundary: bool) -> CoverageGridPoint {
    CoverageGridPoint {
        point: k,
        n_min: n,
        n_max: n,
        observed,
        mcse: 0.02,
        replicates: 100,
        boundary,
    }
}

/// Default Frequentist DAG `AverageEffect`: `linear_ate_data` at 250 / 500 / 1000.
static ATE_GRID: [CoverageGridPoint; 3] =
    [point(0, 250, 0.94, false), point(1, 500, 0.95, false), point(2, 1000, 0.94, false)];
/// The same construction with its smallest point made to under-cover
/// (constructed, not measured).
static ATE_GRID_FAILING_SMALL: [CoverageGridPoint; 3] =
    [point(0, 250, 0.87, true), point(1, 500, 0.95, false), point(2, 1000, 0.94, false)];
/// Default Bayesian DAG `AverageEffect` at 250 / 500 / 1000.
static BAYES_GRID: [CoverageGridPoint; 3] =
    [point(0, 250, 0.97, false), point(1, 500, 0.96, false), point(2, 1000, 0.93, false)];
/// Frequentist `TemporalDag` Pulse on the driven AR(1) ρ = 0.5 design at DGP
/// lengths 80 / 160 / 320. The slot binds on lag-aligned analysis n (79 / 159 / 319).
static PULSE_GRID: [CoverageGridPoint; 3] =
    [point(0, 79, 0.96, false), point(1, 159, 0.93, false), point(2, 319, 0.96, false)];

fn record(
    id: &'static str,
    key: [&'static str; 9],
    replicates_min: u32,
    posterior_draws_min: u32,
    grid: &'static [CoverageGridPoint],
    boundary: bool,
) -> CoverageRecord {
    let [
        query,
        graph_class,
        modality,
        inference,
        estimator,
        interval_method,
        dependence,
        posterior,
        functional,
    ] = key;
    let governing = grid.iter().min_by(|a, b| a.observed.total_cmp(&b.observed)).unwrap();
    CoverageRecord {
        id,
        query,
        graph_class,
        structure: "fixed",
        modality,
        inference,
        estimator,
        interval_method,
        se_kind: "",
        dependence,
        posterior,
        functional,
        identification: "point",
        nominal: 0.95,
        n_min: grid.iter().map(|p| p.n_min).min().unwrap(),
        n_max: grid.iter().map(|p| p.n_max).max().unwrap(),
        replicates_min,
        posterior_draws_min,
        unidentified_mass_max: 0.0,
        observed: governing.observed,
        mcse: governing.mcse,
        replicates: governing.replicates,
        boundary,
        grid,
        dgp: "crates/antecedent/tests/common/static_dgp.rs::linear_ate_data",
        test: "crates/antecedent/tests/calibration_grid_binding.rs::fixture",
        calibration_sha: "0123456789abcdef0123456789abcdef01234567",
    }
}

const FREQUENTIST_ATE: [&str; 9] = [
    "AverageEffect",
    "Dag",
    "tabular",
    "Frequentist",
    "linear.adjustment.ate",
    "bootstrap_se",
    "iid",
    "",
    "all_observed.mean",
];
const BAYESIAN_ATE: [&str; 9] = [
    "AverageEffect",
    "Dag",
    "tabular",
    "Bayesian",
    "bayesian.gcomp",
    "posterior_quantile",
    "iid",
    "laplace.gaussian_identity.prior_scale=10",
    "all_observed.mean",
];
const FREQUENTIST_PULSE: [&str; 9] = [
    "PulseEffect",
    "TemporalDag",
    "series",
    "Frequentist",
    "temporal.linear.adjustment",
    "circular_block_se",
    "circular_block:single_window",
    "",
    "pulse.h1.all_observed",
];

fn registry() -> Vec<CoverageRecord> {
    vec![
        record("cov.fixture.ate", FREQUENTIST_ATE, 199, 0, &ATE_GRID, false),
        record("cov.fixture.bayes", BAYESIAN_ATE, 0, 1000, &BAYES_GRID, false),
        record("cov.fixture.pulse", FREQUENTIST_PULSE, 100, 0, &PULSE_GRID, false),
    ]
}

fn dag() -> Dag {
    let mut g = Dag::with_variables(3);
    for (a, b) in [(2, 0), (2, 1), (0, 1)] {
        g.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    g
}

/// An ordinary study: only data, graph, query (and inference) — every other
/// option at its default. Validation is skipped for speed; it is not part of
/// the calibration key.
fn default_ate(n: usize, inference: InferenceMode) -> (Study, StudyResult) {
    let study = Study::tabular(linear_ate_data(n, 17))
        .graph(dag())
        .query(AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)))
        .inference(inference)
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let result = study.run(&ExecutionContext::for_tests(17)).unwrap();
    (study, result)
}

fn default_pulse(n: usize) -> (Study, StudyResult) {
    let scenario = Scenario { label: "fixture", rho: 0.5, n, seed: 23 };
    let study = Study::series(pulse_series(scenario, 0, false))
        .graph(pulse_dag(false))
        .query(pulse(1))
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let result = study.run(&ExecutionContext::for_tests(23)).unwrap();
    (study, result)
}

fn primary_basis((study, result): (Study, StudyResult)) -> CalibrationBasisWire {
    let contract = study.inspect().unwrap();
    result.calibration_bases(&contract).unwrap().remove(0)
}

fn slot(basis: &CalibrationBasisWire, records: &[CoverageRecord]) -> CalibrationSlotWire {
    calibration_slot_in(basis, records)
}

#[test]
fn a_default_frequentist_ate_is_calibrated_anywhere_inside_its_grid() {
    let records = registry();
    // The reviewer's row counts, and the grid's edges.
    for n in [250, 300, 301, 500, 777, 1000] {
        let basis = primary_basis(default_ate(n, InferenceMode::Frequentist));
        let slot = slot(&basis, &records);
        assert_eq!(slot.status, "calibrated", "n = {n}: {slot:#?}");
        assert_eq!(slot.record_id.as_deref(), Some("cov.fixture.ate"));
        assert_eq!((slot.scope_n, slot.scope_n_max), (Some(250), Some(1000)));
    }
}

#[test]
fn outside_the_grid_is_scope_not_assessed_never_extrapolated() {
    let records = registry();
    for n in [120, 249, 1001, 2000] {
        let basis = primary_basis(default_ate(n, InferenceMode::Frequentist));
        let slot = slot(&basis, &records);
        assert_eq!(slot.status, "scope_not_assessed", "n = {n}: {slot:#?}");
        assert_eq!(slot.reason.as_deref(), Some(SAMPLE_SIZE_OUTSIDE_MEASURED_RANGE));
        assert_eq!(slot.record_id.as_deref(), Some("cov.fixture.ate"), "the nearest record");
    }
}

#[test]
fn a_default_bayesian_ate_binds_inside_its_grid() {
    let records = registry();
    let inside =
        primary_basis(default_ate(500, InferenceMode::Bayesian(BayesianConfig::laplace())));
    let slot_inside = slot(&inside, &records);
    assert_eq!(slot_inside.status, "calibrated", "{slot_inside:#?}");
    assert_eq!(slot_inside.record_id.as_deref(), Some("cov.fixture.bayes"));
    let outside =
        primary_basis(default_ate(1500, InferenceMode::Bayesian(BayesianConfig::laplace())));
    let slot_outside = slot(&outside, &records);
    assert_eq!(slot_outside.status, "scope_not_assessed", "{slot_outside:#?}");
    assert_eq!(slot_outside.reason.as_deref(), Some(SAMPLE_SIZE_OUTSIDE_MEASURED_RANGE));
}

#[test]
fn a_default_temporal_pulse_binds_inside_its_grid() {
    let records = registry();
    for n in [80, 160, 240, 320] {
        let slot = slot(&primary_basis(default_pulse(n)), &records);
        assert_eq!(slot.status, "calibrated", "n = {n}: {slot:#?}");
        assert_eq!(slot.record_id.as_deref(), Some("cov.fixture.pulse"));
    }
    let slot_short = slot(&primary_basis(default_pulse(60)), &records);
    assert_eq!(slot_short.status, "scope_not_assessed", "{slot_short:#?}");
    assert_eq!(slot_short.reason.as_deref(), Some(SAMPLE_SIZE_OUTSIDE_MEASURED_RANGE));
}

#[test]
fn a_construction_that_failed_at_one_grid_point_is_a_named_boundary_over_its_range() {
    // The summary flag is deliberately left false: the matcher must read the
    // failing point itself rather than trust a row that averaged it away.
    let records = vec![record(
        "cov.fixture.ate_failing_small",
        FREQUENTIST_ATE,
        199,
        0,
        &ATE_GRID_FAILING_SMALL,
        false,
    )];
    for n in [250, 500, 1000] {
        let basis = primary_basis(default_ate(n, InferenceMode::Frequentist));
        let slot = slot(&basis, &records);
        assert_eq!(slot.status, "scope_not_assessed", "n = {n}: {slot:#?}");
        assert_eq!(slot.reason.as_deref(), Some(BOUNDARY_RECORD));
        assert_eq!(slot.observed, Some(0.87), "the failing point's coverage is reported");
    }
}

/// Leak a runtime key field into the `'static` shape of a generated record.
fn leak(text: &str) -> &'static str {
    Box::leak(text.to_owned().into_boxed_str())
}

/// A record whose construction is exactly the one `basis` reports, measured
/// over `grid` (the default Bayesian grid's row counts).
fn record_for(id: &'static str, basis: &CalibrationBasisWire) -> CoverageRecord {
    let key = &basis.key;
    CoverageRecord {
        se_kind: leak(&key.se_kind),
        identification: leak(&key.identification),
        structure: leak(&key.structure),
        ..record(
            id,
            [
                leak(&key.query),
                leak(&key.graph_class),
                leak(&key.modality),
                leak(&key.inference),
                leak(&key.estimator),
                leak(&key.interval_method),
                leak(&key.dependence),
                leak(&key.posterior),
                leak(&key.functional),
            ],
            0,
            0,
            &BAYES_GRID,
            false,
        )
    }
}

fn aipw_basis(estimator: antecedent_estimate::AipwAte) -> CalibrationBasisWire {
    let study = Study::tabular(linear_ate_data(500, 17))
        .graph(dag())
        .query(AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)))
        .estimator(estimator)
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let result = study.run(&ExecutionContext::for_tests(17)).unwrap();
    primary_basis((study, result))
}

#[test]
fn a_non_default_overlap_policy_does_not_bind_to_records_measured_at_the_default() {
    use antecedent_estimate::{AipwAte, OverlapPolicy};
    let default = aipw_basis(AipwAte::new().with_bootstrap_replicates(0));
    assert_eq!(default.key.functional, "all_observed.mean");
    let records = vec![record_for("cov.fixture.aipw_default_overlap", &default)];
    assert_eq!(slot(&default, &records).status, "calibrated");

    // Spelling the default policy out is the same construction.
    let spelled = aipw_basis(
        AipwAte::new()
            .with_bootstrap_replicates(0)
            .with_overlap(OverlapPolicy::RequireDiagnostics { clip: Some(0.01), trim: None }),
    );
    assert_eq!(spelled.key, default.key);

    for (policy, label) in [
        (
            OverlapPolicy::RequireDiagnostics { clip: Some(0.05), trim: None },
            "all_observed.mean+overlap=clip:0.05,trim:none",
        ),
        (
            OverlapPolicy::RequireDiagnostics { clip: Some(0.01), trim: Some(0.1) },
            "all_observed.mean+overlap=clip:0.01,trim:0.1",
        ),
    ] {
        let basis = aipw_basis(AipwAte::new().with_bootstrap_replicates(0).with_overlap(policy));
        assert_eq!(basis.key.functional, label);
        let slot = slot(&basis, &records);
        // No record measured this construction: unavailable, never borrowed
        // from the default-policy record.
        assert_eq!(slot.status, "unavailable", "{label}: {slot:#?}");
        assert_eq!(
            slot.reason.as_deref(),
            Some(antecedent_io::calibration::CONSTRUCTION_NOT_MEASURED),
            "{label}"
        );
        // A record measured under that policy binds it.
        let measured = vec![record_for("cov.fixture.aipw_overlap", &basis)];
        assert_eq!(calibration_slot_in(&basis, &measured).status, "calibrated", "{label}");
    }
}

#[test]
fn a_non_gaussian_likelihood_binds_only_to_records_of_that_likelihood() {
    use antecedent_prob::BayesLikelihood;
    let default =
        primary_basis(default_ate(500, InferenceMode::Bayesian(BayesianConfig::laplace())));
    let rows = 500;
    let mut treatment = vec![0.0; rows];
    let mut outcome = vec![0.0; rows];
    let mut confounder = vec![0.0; rows];
    for row in 0..rows {
        let k = u32::try_from(row).unwrap();
        let treated = (k * 13) % 7 < 3;
        let high = (k * 37) % 101 > 50;
        confounder[row] = f64::from((k * 37) % 101) / 50.0 - 1.0;
        treatment[row] = f64::from(treated);
        outcome[row] = f64::from((k * 29 + u32::from(treated) * 11) % 5 < 2 + u32::from(high));
    }
    let binary = antecedent_data::TabularData::from_f64_columns([
        ("t", treatment.as_slice()),
        ("y", &outcome),
        ("z", &confounder),
    ])
    .unwrap();
    let study = Study::tabular(binary)
        .graph(dag())
        .query(AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)))
        .inference(InferenceMode::Bayesian(
            BayesianConfig::laplace().likelihood(BayesLikelihood::BernoulliLogit),
        ))
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let result = study.run(&ExecutionContext::for_tests(17)).unwrap();
    let logit = primary_basis((study, result));
    assert_eq!(logit.key.posterior, "laplace.bernoulli_logit.prior_scale=10");
    let gaussian_records = registry();
    let slot = slot(&logit, &gaussian_records);
    assert_eq!(slot.status, "unavailable", "{slot:#?}");
    assert_eq!(slot.reason.as_deref(), Some(antecedent_io::calibration::CONSTRUCTION_NOT_MEASURED));
    let logit_records = vec![record_for("cov.fixture.logit", &logit)];
    assert_eq!(calibration_slot_in(&logit, &logit_records).status, "calibrated");
    assert_ne!(calibration_slot_in(&default, &logit_records).status, "calibrated");
}
