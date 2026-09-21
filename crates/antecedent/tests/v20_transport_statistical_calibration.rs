//! Coverage of the T6 empirical-table outer bootstrap against known binary SCMs.
//!
//! Ignored tests run via `scripts/gate_calibration.sh` (release build).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::many_single_char_names)]

mod common;

use std::collections::BTreeMap;
use std::sync::Arc;

use antecedent_core::{
    DependenceGroup, DistributionAvailability, Environment, EvidenceCatalog, EvidenceKind,
    EvidenceRegime, ExecutionContext, RegimeBinding, RegimeId, RegimeKind, SamplingDesign,
    TargetSampling, Value, VariableCoordinate, VariableDomain, VariableId,
};
use antecedent_estimate::{
    EmpiricalTableOptions, RegimeSample, StatisticalTransportInput, evaluate_statistical_transport,
};
use antecedent_expr::{Assignment, ExactEvaluationLimits};
use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram};
use antecedent_identify::{
    CatalogTransportResult, ClassicalTransportQuery, SidLimits, identify_catalog_transport,
    identify_classical_transport,
};
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, grid_n, map_replicates,
    n_sim, stream_seed, unit_uniform,
};

const INTERVAL: &str = "percentile_bootstrap";

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

fn keyed(test: &'static str, dgp: &'static str) -> CoverageTally {
    CoverageTally::for_record(RecordKey { test, dgp, interval: INTERVAL }, REPORTED_LEVEL)
}

fn construction() -> Construction {
    Construction {
        query: "ClassicalTransport".into(),
        graph_class: "Admg".into(),
        structure: "fixed".into(),
        modality: "tabular".into(),
        inference: "Frequentist".into(),
        estimator: "transport.empirical_table_plugin".into(),
        interval_method: INTERVAL.into(),
        se_kind: "percentile".into(),
        dependence: "iid".into(),
        posterior: String::new(),
        functional: "target_interventional_mean".into(),
        identification: "point".into(),
        reported_level: REPORTED_LEVEL,
    }
}

fn bern(seed: u64, p: f64) -> f64 {
    f64::from(unit_uniform(seed) < p)
}

fn options() -> EmpiricalTableOptions {
    EmpiricalTableOptions::default()
}

#[allow(clippy::cast_possible_truncation)] // This SCM helper emits binary 0/1 intervention codes.
fn sample(
    population: &str,
    regime: RegimeId,
    snapshot: &str,
    interventions: Vec<(VariableId, f64)>,
    columns: BTreeMap<VariableId, Vec<Option<f64>>>,
) -> RegimeSample {
    RegimeSample {
        population: Arc::from(population),
        regime,
        snapshot_identity: Arc::from(snapshot),
        interventions: interventions
            .into_iter()
            .map(|(variable, value)| antecedent_expr::InterventionAssignment {
                variable,
                value: Value::Int64(value as i64),
            })
            .collect(),
        columns,
    }
}

fn binary_env(identity: &str, vars: &[u32], selections: &[u32]) -> Environment {
    Environment::try_new(
        identity,
        vars.iter()
            .map(|&i| VariableCoordinate {
                variable: v(i),
                domain: VariableDomain::Binary,
                unit: None,
            })
            .collect::<Vec<_>>(),
        selections.iter().copied().map(v).collect::<Vec<_>>(),
    )
    .unwrap()
}

fn binding(regime: RegimeId, snapshot: &str) -> RegimeBinding {
    RegimeBinding {
        dataset_identity: None,
        regime,
        snapshot_identity: Arc::from(snapshot),
        schema_names: Arc::from([]),
        sampling: SamplingDesign::Independent,
        weights: None,
        dependence: DependenceGroup::IndependentStudies,
    }
}

#[allow(clippy::too_many_arguments)] // Keep the recorded construction, truth and scope explicit.
fn mean_interval(
    functional: &antecedent_identify::BoundTransportFunctional,
    input: &StatisticalTransportInput,
    request: Assignment,
    outcome: VariableId,
    seed: u64,
    n: u64,
    tally: &mut CoverageTally,
    truth: f64,
) {
    let ctx = ExecutionContext::for_tests(seed);
    match evaluate_statistical_transport(
        functional,
        input,
        request,
        ExactEvaluationLimits::default(),
        &options(),
        &ctx,
    ) {
        Ok(estimate) => {
            let ok = estimate.uncertainty.as_ref().map(|row| row.replicates_ok);
            tally.bind(
                &construction(),
                ScopeFacts {
                    row_count: n,
                    replicates_ok: ok,
                    posterior_draws: None,
                    unidentified_mass: 0.0,
                },
            );
            let interval = estimate.mean_intervals.as_ref().and_then(|rows| {
                rows.iter().find(|(var, _, _)| *var == outcome).map(|(_, lo, hi)| (*lo, *hi))
            });
            tally.record(interval, truth);
        }
        Err(_) => tally.skip(),
    }
}

fn target_observational_xy() -> (antecedent_identify::BoundTransportFunctional, Assignment) {
    let mut graph = Admg::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let diagram = SelectionDiagram::try_new(graph, [v(1)]).unwrap();
    let query = ClassicalTransportQuery {
        outcomes: Arc::from([v(1)]),
        treatments: Arc::from([v(0)]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    };
    let ctx = ExecutionContext::for_tests(1);
    let antecedent_identify::ClassicalTransportResult::Identified(proof) =
        identify_classical_transport(&diagram, &query, SidLimits::default(), &ctx).unwrap()
    else {
        panic!("identified");
    };
    let catalog = EvidenceCatalog::try_new(
        [binary_env("target", &[0, 1], &[1])],
        [EvidenceRegime::try_new(
            RegimeId::from_raw(0),
            RegimeKind::Observational,
            EvidenceKind::Available,
            [],
            [],
            [v(0), v(1)],
            "target",
            DistributionAvailability::Joint,
        )
        .unwrap()],
        [binding(RegimeId::from_raw(0), "obs")],
        Some(TargetSampling::RepresentativeSample),
    )
    .unwrap();
    (proof.bind_catalog(&catalog).unwrap(), Assignment::from_pairs([(v(0), Value::Int64(1))]))
}

fn xy_sample(n: usize, seed: u64, px: f64, py0: f64, py1: f64, snapshot: &str) -> RegimeSample {
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let xi = bern(stream_seed(seed, i as u64), px);
        let yi = bern(stream_seed(seed, 10_000 + i as u64), if xi > 0.5 { py1 } else { py0 });
        x.push(Some(xi));
        y.push(Some(yi));
    }
    sample(
        "target",
        RegimeId::from_raw(0),
        snapshot,
        Vec::new(),
        BTreeMap::from([(v(0), x), (v(1), y)]),
    )
}

/// Shared target joint: `P(X=1)=0.5`, `P(Y=1|X)=0.3 + 0.4 X`.
fn xy_shared_joint(n: usize, seed: u64) -> RegimeSample {
    xy_sample(n, seed, 0.5, 0.3, 0.7, "obs")
}

/// Rare-treatment boundary with the same outcome mechanism and `P(X=1)=0.04`.
fn xy_rare_treatment(n: usize, seed: u64) -> RegimeSample {
    xy_sample(n, seed, 0.04, 0.3, 0.7, "obs")
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn shared_factor_target_observational_nominal_coverage() {
    let n = grid_n(200);
    let (functional, request) = target_observational_xy();
    let mut tally = keyed("shared_factor_target_observational_nominal_coverage", "xy_shared_joint");
    for (rep, input) in map_replicates(n_sim(), |rep| StatisticalTransportInput {
        supplied: Vec::new(),
        samples: vec![xy_shared_joint(n, rep.wrapping_mul(1_000_003))],
    })
    .into_iter()
    .enumerate()
    {
        mean_interval(
            &functional,
            &input,
            request.clone(),
            v(1),
            rep as u64,
            n as u64,
            &mut tally,
            0.7,
        );
    }
    tally.assert();
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn weak_overlap_near_empty_conditioner_boundary() {
    let n = grid_n(200);
    let (functional, request) = target_observational_xy();
    let mut tally = keyed("weak_overlap_near_empty_conditioner_boundary", "xy_rare_treatment");
    for (rep, input) in map_replicates(n_sim(), |rep| StatisticalTransportInput {
        supplied: Vec::new(),
        samples: vec![xy_rare_treatment(n, rep.wrapping_mul(1_000_019))],
    })
    .into_iter()
    .enumerate()
    {
        mean_interval(
            &functional,
            &input,
            request.clone(),
            v(1),
            rep as u64,
            n as u64,
            &mut tally,
            0.7,
        );
    }
    tally.emit_named_boundary();
}

fn standardize_functional() -> (antecedent_identify::BoundTransportFunctional, Assignment) {
    let mut graph = Admg::with_variables(3);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    graph.insert_bidirected(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    let diagram = SelectionDiagram::try_new(graph, [v(0)]).unwrap();
    let query = ClassicalTransportQuery {
        outcomes: Arc::from([v(2)]),
        treatments: Arc::from([v(1)]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    };
    let ctx = ExecutionContext::for_tests(1);
    let catalog = EvidenceCatalog::try_new(
        [binary_env("source", &[0, 1, 2], &[0]), binary_env("target", &[0, 1, 2], &[0])],
        [
            EvidenceRegime::try_new(
                RegimeId::from_raw(0),
                RegimeKind::Experimental,
                EvidenceKind::Available,
                [v(1)],
                [],
                [v(0), v(2)],
                "source",
                DistributionAvailability::Joint,
            )
            .unwrap(),
            EvidenceRegime::try_new(
                RegimeId::from_raw(1),
                RegimeKind::Observational,
                EvidenceKind::Available,
                [],
                [],
                [v(0)],
                "target",
                DistributionAvailability::Joint,
            )
            .unwrap(),
        ],
        [binding(RegimeId::from_raw(0), "trial"), binding(RegimeId::from_raw(1), "covariates")],
        Some(TargetSampling::RepresentativeSample),
    )
    .unwrap();
    let CatalogTransportResult::Identified(bound) =
        identify_catalog_transport(&diagram, &query, &catalog, SidLimits::default(), &ctx).unwrap()
    else {
        panic!("catalog-identified");
    };
    (*bound, Assignment::from_pairs([(v(1), Value::Int64(1))]))
}

/// `Y | do(X=1), Z ~ Bern(0.25 + 0.4 Z)`, target `P(Z=1)=0.3` ⇒ `E[Y|do(X=1)]=0.37`.
fn standardize_imbalance(n_target: usize, seed: u64) -> StatisticalTransportInput {
    let n_source = (n_target / 4).max(20);
    let mut z_s = Vec::with_capacity(n_source);
    let mut y_s = Vec::with_capacity(n_source);
    for i in 0..n_source {
        let z = bern(stream_seed(seed, i as u64), 0.6);
        let y = bern(stream_seed(seed, 20_000 + i as u64), 0.25 + 0.4 * z);
        z_s.push(Some(z));
        y_s.push(Some(y));
    }
    let mut z_t = Vec::with_capacity(n_target);
    for i in 0..n_target {
        z_t.push(Some(bern(stream_seed(seed, 40_000 + i as u64), 0.3)));
    }
    StatisticalTransportInput {
        supplied: Vec::new(),
        samples: vec![
            sample(
                "source",
                RegimeId::from_raw(0),
                "trial",
                vec![(v(1), 1.0)],
                BTreeMap::from([(v(0), z_s), (v(2), y_s)]),
            ),
            sample(
                "target",
                RegimeId::from_raw(1),
                "covariates",
                Vec::new(),
                BTreeMap::from([(v(0), z_t)]),
            ),
        ],
    }
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn source_target_imbalance_standardize_nominal_coverage() {
    let n = grid_n(400);
    let (functional, request) = standardize_functional();
    let mut tally =
        keyed("source_target_imbalance_standardize_nominal_coverage", "standardize_imbalance");
    for (rep, input) in
        map_replicates(n_sim(), |rep| standardize_imbalance(n, rep.wrapping_mul(1_000_037)))
            .into_iter()
            .enumerate()
    {
        mean_interval(
            &functional,
            &input,
            request.clone(),
            v(2),
            rep as u64,
            n as u64,
            &mut tally,
            0.37,
        );
    }
    tally.assert();
}

fn frontdoor_functional() -> (antecedent_identify::BoundTransportFunctional, Assignment) {
    let mut graph = Admg::with_variables(3);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    graph.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    let diagram = SelectionDiagram::try_new(graph, []).unwrap();
    let query = ClassicalTransportQuery {
        outcomes: Arc::from([v(2)]),
        treatments: Arc::from([v(0)]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    };
    let ctx = ExecutionContext::for_tests(1);
    let antecedent_identify::ClassicalTransportResult::Identified(proof) =
        identify_classical_transport(&diagram, &query, SidLimits::default(), &ctx).unwrap()
    else {
        panic!("identified");
    };
    let catalog = EvidenceCatalog::try_new(
        [binary_env("target", &[0, 1, 2], &[])],
        [EvidenceRegime::try_new(
            RegimeId::from_raw(0),
            RegimeKind::Observational,
            EvidenceKind::Available,
            [],
            [],
            [v(0), v(1), v(2)],
            "target",
            DistributionAvailability::Joint,
        )
        .unwrap()],
        [binding(RegimeId::from_raw(0), "obs")],
        Some(TargetSampling::RepresentativeSample),
    )
    .unwrap();
    (proof.bind_catalog(&catalog).unwrap(), Assignment::from_pairs([(v(0), Value::Int64(1))]))
}

fn frontdoor_binary_scm(n: usize, seed: u64) -> RegimeSample {
    let mut x = Vec::with_capacity(n);
    let mut m = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let u = bern(stream_seed(seed, i as u64), 0.4);
        let xi = bern(stream_seed(seed, 50_000 + i as u64), 0.2 + 0.6 * u);
        let mi = bern(stream_seed(seed, 60_000 + i as u64), 0.15 + 0.55 * xi);
        let yi = bern(stream_seed(seed, 70_000 + i as u64), 0.1 + 0.45 * mi + 0.2 * u);
        x.push(Some(xi));
        m.push(Some(mi));
        y.push(Some(yi));
    }
    sample(
        "target",
        RegimeId::from_raw(0),
        "obs",
        Vec::new(),
        BTreeMap::from([(v(0), x), (v(1), m), (v(2), y)]),
    )
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn recursive_frontdoor_nominal_coverage() {
    let n = grid_n(300);
    let (functional, request) = frontdoor_functional();
    let mut tally = keyed("recursive_frontdoor_nominal_coverage", "frontdoor_binary_scm");
    for (rep, input) in map_replicates(n_sim(), |rep| StatisticalTransportInput {
        supplied: Vec::new(),
        samples: vec![frontdoor_binary_scm(n, rep.wrapping_mul(1_000_049))],
    })
    .into_iter()
    .enumerate()
    {
        mean_interval(
            &functional,
            &input,
            request.clone(),
            v(2),
            rep as u64,
            n as u64,
            &mut tally,
            0.495,
        );
    }
    tally.assert();
}
