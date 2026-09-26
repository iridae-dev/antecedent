//! Coordinate-specific closure evidence for static linear adjustment.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{AcceptedGraph, EstimatorId, RefuteSuite, StructureSource, Study};
use antecedent_core::{ExecutionContext, SlotAvailability};
use antecedent_io::consume_analysis_result;

mod common;

use common::fixtures::confounded_scm;

#[allow(clippy::too_many_lines, reason = "one function walks a coordinate's full lifecycle")]
fn verify_linear_adjustment_coordinate(coordinate: &str, accepted: bool, validation: &str) {
    let ctx = ExecutionContext::for_tests(73);
    let (data, dag, query) = confounded_scm(512, 73);
    let suite = match validation {
        "none" => RefuteSuite::None,
        "cheap" => RefuteSuite::Cheap,
        "full" => RefuteSuite::Full,
        other => panic!("unexpected validation axis {other}"),
    };
    let base = Study::tabular(data.clone())
        .query(query)
        .estimator(EstimatorId::LinearAdjustmentAte)
        .refute(suite)
        .bootstrap_replicates(0);
    let builder = if accepted { base.graph(AcceptedGraph::from(dag)) } else { base.graph(dag) };
    let study = builder.clone().build().unwrap();
    let mut prepared = study.prepare(&ctx).unwrap();
    drop(builder);
    drop(study);

    assert_eq!(
        prepared.structure_source(),
        if accepted { StructureSource::Accepted } else { StructureSource::Explicit },
        "{coordinate} structure source"
    );
    assert_eq!(prepared.support_status(), Some(antecedent::CellStatus::Licensed));
    let contract = prepared.contract().unwrap();
    match &contract.reasoning.support {
        SlotAvailability::Available(slot) => assert_eq!(
            slot.matrix_coordinate.as_deref(),
            Some(coordinate),
            "prepared contract did not retain this exact coordinate"
        ),
        other => panic!("{coordinate} support slot unavailable: {other:?}"),
    }
    let expected_validation = match validation {
        "none" => None,
        "cheap" => Some("overlap+evalue"),
        "full" => Some("validation.full"),
        _ => unreachable!(),
    };
    assert_eq!(
        prepared.plan().logical.record.validation_suite.as_deref(),
        expected_validation,
        "{coordinate} validation choice"
    );

    let lowering = prepared
        .checked_linear_adjustment()
        .unwrap_or_else(|| panic!("{coordinate} omitted its checked linear-adjustment lowering"));
    assert_eq!(lowering.source_functional(), lowering.program().mapping().source);
    assert_eq!(lowering.executable_functional(), lowering.program().mapping().executable);
    assert_eq!(lowering.lowering().population, antecedent_core::TargetPopulation::AllObserved);
    assert_eq!(prepared.plan().logical.record.estimator.as_deref(), Some("linear.adjustment.ate"));

    let result = prepared.estimate(&data, &ctx).unwrap();
    // The fixture is generated from y = 2*t + z + noise and the graph supplies
    // the complete back-door adjustment set {z}; this is an independent truth.
    assert!((result.estimate.ate - 2.0).abs() < 0.2, "{coordinate}: ate={}", result.estimate.ate);
    let refreshed = prepared.refresh(data.clone(), &ctx).unwrap();
    assert!(
        (refreshed.estimate.ate - 2.0).abs() < 0.2,
        "{coordinate}: refreshed ate={}",
        refreshed.estimate.ate
    );
    assert_eq!(result.estimate.ate.to_bits(), refreshed.estimate.ate.to_bits());
    assert!(
        prepared.checked_linear_adjustment().is_some(),
        "{coordinate} lost its checked lowering after refresh"
    );

    let bytes = prepared
        .encode_contracted_result(
            &refreshed,
            &format!(
                "linear-adjustment-{validation}-{}",
                if accepted { "accepted" } else { "explicit" }
            ),
            &ctx,
        )
        .unwrap();
    let consumed = consume_analysis_result(&bytes).unwrap();
    assert!(
        consumed.acceptance.accepts_as_verified_program(),
        "{coordinate} artifact should replay from portable OLS moments: {:?}",
        consumed.acceptance.unresolved
    );
    assert_eq!(consumed.body.estimate, Some(refreshed.effect()));
    let consumed_contract = consumed.contract.as_ref().expect("contracted linear artifact");
    assert_eq!(consumed_contract.estimator.as_deref(), Some("linear.adjustment.ate"));
    assert!(
        consumed_contract
            .program
            .as_ref()
            .is_some_and(|program| { program.checked_linear_adjustment_lowering.is_some() })
    );
    assert!(consumed_contract.data_snapshot.as_ref().unwrap().linear_fit_moments.is_some());
    if coordinate == "AverageEffect:Dag:explicit:Frequentist:none" {
        let contract = consumed.contract.as_ref().expect("contracted linear artifact");
        let mut missing = contract.clone();
        missing.program.as_mut().unwrap().checked_linear_adjustment_lowering = None;
        let unresolved =
            antecedent_io::verify_contract_against_body(&consumed.header, &consumed.body, &missing);
        assert!(
            unresolved
                .iter()
                .any(|reason| { reason.as_ref() == "program.checked_linear_adjustment_lowering" }),
            "missing lowering refusal was not precise: {unresolved:?}"
        );

        let mut tampered = contract.clone();
        tampered
            .program
            .as_mut()
            .unwrap()
            .checked_linear_adjustment_lowering
            .as_mut()
            .unwrap()
            .functional ^= 1;
        let unresolved = antecedent_io::verify_contract_against_body(
            &consumed.header,
            &consumed.body,
            &tampered,
        );
        assert!(
            unresolved
                .iter()
                .any(|reason| { reason.as_ref() == "program.checked_linear_adjustment_binding" }),
            "semantic lowering tamper was not rejected: {unresolved:?}"
        );

        let mut missing_standard_error = consumed.body.clone();
        missing_standard_error.standard_error = None;
        let unresolved = antecedent_io::verify_contract_against_body(
            &consumed.header,
            &missing_standard_error,
            contract,
        );
        assert!(
            unresolved.iter().any(|reason| reason.as_ref() == "body.standard_error"),
            "an artifact missing the committed analytic standard error was accepted: {unresolved:?}"
        );

        let mut tampered_moments = contract.clone();
        tampered_moments
            .data_snapshot
            .as_mut()
            .unwrap()
            .linear_fit_moments
            .as_mut()
            .unwrap()
            .outcome_cross[1] += 1.0;
        let unresolved = antecedent_io::verify_contract_against_body(
            &consumed.header,
            &consumed.body,
            &tampered_moments,
        );
        assert!(!unresolved.is_empty(), "tampering with replay moments was accepted");
    }
}

#[test]
fn dag_explicit_frequentist_none_linear_adjustment_coordinate() {
    verify_linear_adjustment_coordinate(
        "AverageEffect:Dag:explicit:Frequentist:none",
        false,
        "none",
    );
}

#[test]
fn dag_explicit_frequentist_cheap_linear_adjustment_coordinate() {
    verify_linear_adjustment_coordinate(
        "AverageEffect:Dag:explicit:Frequentist:cheap",
        false,
        "cheap",
    );
}

#[test]
fn dag_explicit_frequentist_full_linear_adjustment_coordinate() {
    verify_linear_adjustment_coordinate(
        "AverageEffect:Dag:explicit:Frequentist:full",
        false,
        "full",
    );
}

#[test]
fn dag_accepted_frequentist_none_linear_adjustment_coordinate() {
    verify_linear_adjustment_coordinate(
        "AverageEffect:Dag:accepted:Frequentist:none",
        true,
        "none",
    );
}

#[test]
fn dag_accepted_frequentist_cheap_linear_adjustment_coordinate() {
    verify_linear_adjustment_coordinate(
        "AverageEffect:Dag:accepted:Frequentist:cheap",
        true,
        "cheap",
    );
}

#[test]
fn dag_accepted_frequentist_full_linear_adjustment_coordinate() {
    verify_linear_adjustment_coordinate(
        "AverageEffect:Dag:accepted:Frequentist:full",
        true,
        "full",
    );
}
