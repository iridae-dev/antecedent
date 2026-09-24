//! Coordinate-specific closure evidence for checked AIPW ATE execution.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{AcceptedGraph, EstimatorId, RefuteSuite, StructureSource, Study};
use antecedent_core::{ExecutionContext, SlotAvailability};
use antecedent_data::TableView;
use antecedent_io::consume_analysis_result;

mod common;

use common::fixtures::confounded_scm;

fn verify_aipw_coordinate(coordinate: &str, accepted: bool, validation: &str) {
    let ctx = ExecutionContext::for_tests(73);
    let (data, dag, query) = confounded_scm(512, 73);
    let suite = match validation {
        "none" => RefuteSuite::None,
        "cheap" => RefuteSuite::Cheap,
        "full" => RefuteSuite::Full,
        other => panic!("unexpected validation suite {other}"),
    };
    let builder = Study::tabular(data.clone())
        .query(query)
        .estimator(EstimatorId::Aipw)
        .refute(suite)
        .bootstrap_replicates(0);
    let builder =
        if accepted { builder.graph(AcceptedGraph::from(dag)) } else { builder.graph(dag) };
    let reference = builder.clone().build().unwrap().run(&ctx).unwrap();
    let study = builder.clone().build().unwrap();
    let mut prepared = study.prepare(&ctx).unwrap();
    drop(builder);
    drop(study);

    assert_eq!(
        prepared.structure_source(),
        if accepted { StructureSource::Accepted } else { StructureSource::Explicit },
        "{coordinate} structure source"
    );
    assert_eq!(prepared.support_status(), Some(antecedent::CellStatus::Licensed), "{coordinate}");
    let contract = prepared.contract().unwrap();
    assert_eq!(contract.query_kind.as_ref(), "AverageEffect", "{coordinate}");
    assert_eq!(contract.estimator.as_deref(), Some("aipw"), "{coordinate}");
    assert_eq!(contract.inference.as_ref(), "frequentist", "{coordinate}");
    match &contract.reasoning.support {
        SlotAvailability::Available(slot) => {
            assert_eq!(slot.matrix_status.as_ref(), "licensed", "{coordinate}");
            assert_eq!(
                slot.matrix_coordinate.as_deref(),
                coordinate.strip_suffix("::estimator=aipw"),
                "{coordinate} exact base support coordinate"
            );
        }
        other => panic!("{coordinate} support unavailable: {other:?}"),
    }
    assert_eq!(
        prepared.plan().logical.record.validation_suite.as_deref(),
        match validation {
            "none" => None,
            "cheap" => Some("overlap+evalue"),
            "full" => Some("validation.full"),
            _ => unreachable!(),
        },
        "{coordinate} validation choice"
    );
    assert_eq!(prepared.plan().logical.record.estimator.as_deref(), Some("aipw"));
    assert_eq!(prepared.plan().logical.record.identifier.as_deref(), Some("backdoor.adjustment"));

    let lowering = prepared.checked_aipw_ate().expect("retained checked AIPW operation");
    assert_eq!(lowering.program().mapping().source, lowering.target().functional, "{coordinate}");
    assert_eq!(
        lowering.lowering().procedure,
        antecedent_estimate::CheckedAipwProcedure::CrossFittedLogisticOls,
        "{coordinate} procedure"
    );
    assert_eq!(lowering.lowering().population, antecedent_core::TargetPopulation::AllObserved);
    assert_eq!(lowering.lowering().rows.len(), data.row_count(), "{coordinate} row binding");

    let result = prepared.estimate(&data, &ctx).unwrap();
    assert!((result.effect() - 2.0).abs() < 0.3, "{coordinate}: AIPW effect {}", result.effect());
    assert!((result.effect() - reference.effect()).abs() < 1e-12, "{coordinate} vs fresh route");
    assert_eq!(
        result.refutations.iter().map(|report| report.refuter.as_ref()).collect::<Vec<_>>(),
        reference.refutations.iter().map(|report| report.refuter.as_ref()).collect::<Vec<_>>(),
        "{coordinate} refutation methods"
    );
    assert_eq!(
        result.refutations.iter().map(|report| report.passed).collect::<Vec<_>>(),
        reference.refutations.iter().map(|report| report.passed).collect::<Vec<_>>(),
        "{coordinate} refutation decisions"
    );
    let refreshed = prepared.refresh(data.clone(), &ctx).unwrap();
    assert_eq!(refreshed.effect().to_bits(), result.effect().to_bits(), "{coordinate} refresh");
    assert!(prepared.checked_aipw_ate().is_some(), "{coordinate} retained operation after refresh");

    let artifact =
        prepared.encode_contracted_result(&refreshed, &format!("aipw-{coordinate}"), &ctx).unwrap();
    let consumed = consume_analysis_result(&artifact).unwrap();
    assert!(
        consumed.acceptance.accepts_as_verified_program(),
        "{coordinate} artifact dependencies: {:?}",
        consumed.acceptance.unresolved
    );
    assert_eq!(consumed.body.estimate, Some(refreshed.effect()), "{coordinate} artifact result");
}

#[test]
fn dag_accepted_frequentist_aipw_cheap() {
    verify_aipw_coordinate(
        "AverageEffect:Dag:accepted:Frequentist:cheap::estimator=aipw",
        true,
        "cheap",
    );
}

#[test]
fn dag_accepted_frequentist_aipw_full() {
    verify_aipw_coordinate(
        "AverageEffect:Dag:accepted:Frequentist:full::estimator=aipw",
        true,
        "full",
    );
}

#[test]
fn dag_accepted_frequentist_aipw_none() {
    verify_aipw_coordinate(
        "AverageEffect:Dag:accepted:Frequentist:none::estimator=aipw",
        true,
        "none",
    );
}

#[test]
fn dag_explicit_frequentist_aipw_cheap() {
    verify_aipw_coordinate(
        "AverageEffect:Dag:explicit:Frequentist:cheap::estimator=aipw",
        false,
        "cheap",
    );
}

#[test]
fn dag_explicit_frequentist_aipw_full() {
    verify_aipw_coordinate(
        "AverageEffect:Dag:explicit:Frequentist:full::estimator=aipw",
        false,
        "full",
    );
}

#[test]
fn dag_explicit_frequentist_aipw_none() {
    verify_aipw_coordinate(
        "AverageEffect:Dag:explicit:Frequentist:none::estimator=aipw",
        false,
        "none",
    );
}
