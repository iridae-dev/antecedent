//! Coordinate-specific closure evidence for prepared finite-discrete distributions.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{
    AcceptedGraph, BayesianConfig, EstimatorId, IdentifierId, InferenceMode, RefuteSuite,
    StructureSource, Study,
};
use antecedent_core::{
    CausalQuery, ExecutionContext, Intervention, InterventionalDistributionQuery, SlotAvailability,
    Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Admg, Dag, DenseNodeId};
use antecedent_io::consume_analysis_result;

fn distribution_fixture() -> (TabularData, Dag, Admg, InterventionalDistributionQuery) {
    let mut treatment = Vec::new();
    let mut outcome = Vec::new();
    let mut modifier = Vec::new();
    // Each (z,t) stratum has 100 rows. Under intervention t=1, the outcome
    // probabilities are 0.8 for z=0 and 0.6 for z=1; P(z=1)=1/2, so the
    // independent g-formula truth is 0.7.
    for (z, t, y, count) in [
        (0.0, 0.0, 0.0, 80),
        (0.0, 0.0, 1.0, 20),
        (0.0, 1.0, 0.0, 20),
        (0.0, 1.0, 1.0, 80),
        (1.0, 0.0, 0.0, 60),
        (1.0, 0.0, 1.0, 40),
        (1.0, 1.0, 0.0, 40),
        (1.0, 1.0, 1.0, 60),
    ] {
        treatment.extend(std::iter::repeat_n(t, count));
        outcome.extend(std::iter::repeat_n(y, count));
        modifier.extend(std::iter::repeat_n(z, count));
    }
    let data = TabularData::from_f64_columns([
        ("t", treatment.as_slice()),
        ("y", outcome.as_slice()),
        ("z", modifier.as_slice()),
    ])
    .unwrap();
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    let mut admg = Admg::with_variables(3);
    admg.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    admg.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    admg.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    let query = InterventionalDistributionQuery::new(
        VariableId::from_raw(1),
        [Intervention::set(VariableId::from_raw(0), Value::f64(1.0))],
    );
    (data, dag, admg, query)
}

fn verify_distribution_coordinate(
    coordinate: &str,
    graph_kind: &str,
    accepted: bool,
    bayesian: bool,
    suite_name: &str,
) {
    let ctx = ExecutionContext::for_tests(2_101);
    let (data, dag, admg, query) = distribution_fixture();
    let suite = match suite_name {
        "none" => RefuteSuite::None,
        "cheap" => RefuteSuite::Cheap,
        "full" => RefuteSuite::Full,
        other => panic!("unexpected suite {other}"),
    };
    let base = Study::tabular(data.clone())
        .query(CausalQuery::Distribution(query.clone()))
        .identifier(IdentifierId::GeneralId)
        .estimator(EstimatorId::FunctionalDistribution)
        .inference(if bayesian {
            InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(256))
        } else {
            InferenceMode::Frequentist
        })
        .refute(suite)
        .bootstrap_replicates(0);
    let builder = match (graph_kind, accepted) {
        ("Dag", false) => base.graph(dag),
        ("Dag", true) => base.graph(AcceptedGraph::from(dag)),
        ("Admg", false) => base.graph(admg),
        ("Admg", true) => base.graph(AcceptedGraph::from(admg)),
        other => panic!("unexpected graph/source pair {other:?}"),
    };
    let study = builder.clone().build().unwrap();
    let mut prepared = study.prepare(&ctx).unwrap();
    drop(builder);
    drop(study);

    assert_eq!(prepared.query(), &CausalQuery::Distribution(query.clone()), "{coordinate}");
    assert_eq!(
        prepared.structure_source(),
        if accepted { StructureSource::Accepted } else { StructureSource::Explicit },
        "{coordinate} source"
    );
    assert_eq!(prepared.support_status(), Some(antecedent::CellStatus::Licensed), "{coordinate}");
    let contract = prepared.contract().unwrap();
    assert_eq!(contract.query_kind.as_ref(), "InterventionalDistribution", "{coordinate}");
    assert_eq!(contract.identifier.as_deref(), Some("general.id"), "{coordinate}");
    assert_eq!(contract.estimator.as_deref(), Some("functional.distribution"), "{coordinate}");
    assert_eq!(contract.inference.as_ref(), if bayesian { "bayesian" } else { "frequentist" });
    match &contract.reasoning.support {
        SlotAvailability::Available(slot) => assert_eq!(
            slot.matrix_coordinate.as_deref(),
            Some(coordinate),
            "prepared handle must bind the exact licensed coordinate"
        ),
        other => panic!("{coordinate} support slot unavailable: {other:?}"),
    }
    let program = prepared.checked_distribution_program().expect("retained checked program");
    assert_eq!(program.mapping().source, program.mapping().executable, "{coordinate}");
    assert!(!program.factor_requirements().is_empty(), "{coordinate}");
    assert_eq!(prepared.plan().logical.record.identifier.as_deref(), Some("general.id"));
    assert_eq!(
        prepared.plan().logical.record.estimator.as_deref(),
        Some("functional.distribution")
    );
    assert_eq!(
        prepared.plan().logical.record.validation_suite.as_deref(),
        match suite_name {
            "none" => None,
            "cheap" => Some("distribution.cheap"),
            "full" => Some("distribution.full"),
            _ => unreachable!(),
        },
        "{coordinate} procedure"
    );

    let result = prepared.estimate(&data, &ctx).unwrap();
    let distribution = result.distribution.as_ref().expect("distribution result");
    assert!(distribution.mean.is_finite(), "{coordinate}");
    assert!(
        (distribution.mean - 0.7).abs() < if bayesian { 0.08 } else { 1e-12 },
        "{coordinate}: expected independent g-formula truth 0.7, got {}",
        distribution.mean
    );
    if bayesian {
        let posterior = result.posterior.as_ref().expect("joint posterior draws");
        assert_eq!(posterior.draws.n_draws, 256, "{coordinate}");
        for draw in 0..posterior.draws.n_draws {
            let probability_sum: f64 = (0..distribution.atoms.len())
                .map(|atom| posterior.draws.column(atom + 1).unwrap()[draw])
                .sum();
            assert!((probability_sum - 1.0).abs() < 1e-12, "{coordinate} draw {draw}");
        }
    } else {
        assert!(result.posterior.is_none(), "{coordinate}: frequentist route has no posterior");
        assert_eq!(distribution.mean.to_bits(), 0.7_f64.to_bits(), "{coordinate}");
    }
    match suite_name {
        "none" => assert!(result.refutations.is_empty(), "{coordinate}"),
        "cheap" => assert_eq!(result.refutations.len(), 2, "{coordinate}"),
        "full" => assert_eq!(result.refutations.len(), 3, "{coordinate}"),
        _ => unreachable!(),
    }
    assert!(
        result.refutations.iter().all(|report| report.refuter.starts_with("distribution.")),
        "{coordinate}: {:?}",
        result.refutations.iter().map(|report| report.refuter.as_ref()).collect::<Vec<_>>()
    );

    let refreshed = prepared.refresh(data.clone(), &ctx).unwrap();
    let refreshed_distribution = refreshed.distribution.as_ref().expect("refreshed distribution");
    assert!(
        (refreshed_distribution.mean - 0.7).abs() < if bayesian { 0.08 } else { 1e-12 },
        "{coordinate}: refreshed mean {}",
        refreshed_distribution.mean
    );
    assert!(prepared.checked_distribution_program().is_some(), "{coordinate} retained program");

    let artifact = prepared
        .encode_contracted_result(&refreshed, &format!("distribution-{coordinate}"), &ctx)
        .unwrap();
    let consumed = consume_analysis_result(&artifact).unwrap();
    if bayesian {
        assert!(
            consumed
                .acceptance
                .unresolved
                .iter()
                .any(|reason| reason.as_ref() == "dependencies.distribution_posterior_factor_draws"),
            "{coordinate}: expected precise posterior-factor-draw replay dependency, got {:?}",
            consumed.acceptance.unresolved
        );
        assert!(
            !consumed.acceptance.accepts_as_verified_program(),
            "{coordinate}: posterior factor draws cannot yet be independently replayed"
        );
    } else {
        assert!(
            consumed.acceptance.accepts_as_verified_program(),
            "{coordinate}: {:?}",
            consumed.acceptance.unresolved
        );
        assert_eq!(consumed.body.estimate, Some(0.7), "{coordinate}");
    }
}

#[test]
fn dag_accepted_bayesian_cheap() {
    verify_distribution_coordinate(
        "InterventionalDistribution:Dag:accepted:Bayesian:cheap",
        "Dag",
        true,
        true,
        "cheap",
    );
}

#[test]
fn dag_accepted_bayesian_full() {
    verify_distribution_coordinate(
        "InterventionalDistribution:Dag:accepted:Bayesian:full",
        "Dag",
        true,
        true,
        "full",
    );
}

#[test]
fn dag_accepted_bayesian_none() {
    verify_distribution_coordinate(
        "InterventionalDistribution:Dag:accepted:Bayesian:none",
        "Dag",
        true,
        true,
        "none",
    );
}

#[test]
fn dag_accepted_frequentist_cheap() {
    verify_distribution_coordinate(
        "InterventionalDistribution:Dag:accepted:Frequentist:cheap",
        "Dag",
        true,
        false,
        "cheap",
    );
}

#[test]
fn dag_accepted_frequentist_full() {
    verify_distribution_coordinate(
        "InterventionalDistribution:Dag:accepted:Frequentist:full",
        "Dag",
        true,
        false,
        "full",
    );
}

#[test]
fn dag_accepted_frequentist_none() {
    verify_distribution_coordinate(
        "InterventionalDistribution:Dag:accepted:Frequentist:none",
        "Dag",
        true,
        false,
        "none",
    );
}

#[test]
fn dag_explicit_bayesian_cheap() {
    verify_distribution_coordinate(
        "InterventionalDistribution:Dag:explicit:Bayesian:cheap",
        "Dag",
        false,
        true,
        "cheap",
    );
}

#[test]
fn dag_explicit_bayesian_full() {
    verify_distribution_coordinate(
        "InterventionalDistribution:Dag:explicit:Bayesian:full",
        "Dag",
        false,
        true,
        "full",
    );
}

#[test]
fn dag_explicit_bayesian_none() {
    verify_distribution_coordinate(
        "InterventionalDistribution:Dag:explicit:Bayesian:none",
        "Dag",
        false,
        true,
        "none",
    );
}

#[test]
fn dag_explicit_frequentist_cheap() {
    verify_distribution_coordinate(
        "InterventionalDistribution:Dag:explicit:Frequentist:cheap",
        "Dag",
        false,
        false,
        "cheap",
    );
}

#[test]
fn dag_explicit_frequentist_full() {
    verify_distribution_coordinate(
        "InterventionalDistribution:Dag:explicit:Frequentist:full",
        "Dag",
        false,
        false,
        "full",
    );
}

#[test]
fn admg_accepted_bayesian_none() {
    verify_distribution_coordinate(
        "InterventionalDistribution:Admg:accepted:Bayesian:none",
        "Admg",
        true,
        true,
        "none",
    );
}

#[test]
fn admg_accepted_frequentist_none() {
    verify_distribution_coordinate(
        "InterventionalDistribution:Admg:accepted:Frequentist:none",
        "Admg",
        true,
        false,
        "none",
    );
}

#[test]
fn admg_explicit_bayesian_none() {
    verify_distribution_coordinate(
        "InterventionalDistribution:Admg:explicit:Bayesian:none",
        "Admg",
        false,
        true,
        "none",
    );
}

#[test]
fn admg_explicit_frequentist_none() {
    verify_distribution_coordinate(
        "InterventionalDistribution:Admg:explicit:Frequentist:none",
        "Admg",
        false,
        false,
        "none",
    );
}
