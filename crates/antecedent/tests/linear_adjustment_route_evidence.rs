//! Coordinate-specific closure evidence for static linear adjustment.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::RetainedOperation;
use antecedent::{AcceptedGraph, EstimatorId, RefuteSuite, StructureSource, Study};
use antecedent_contract::SlotAvailability;
use antecedent_core::ExecutionContext;
use antecedent_io::consume_analysis_result;

#[path = "fixtures_common/mod.rs"]
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

    let RetainedOperation::CheckedLinear(lowering) = prepared.retained_operation() else {
        panic!("{coordinate} omitted its checked linear-adjustment lowering")
    };
    assert_eq!(lowering.source_functional(), lowering.program().mapping().source);
    assert_eq!(lowering.executable_functional(), lowering.program().mapping().executable);
    assert_eq!(lowering.lowering().population, antecedent_core::TargetPopulation::AllObserved);
    assert_eq!(prepared.plan().logical.record.estimator.as_deref(), Some("linear.adjustment.ate"));

    let result = prepared.estimate(&data, &ctx).unwrap();
    // The fixture is generated from y = 2*t + z + noise and the graph supplies
    // the complete back-door adjustment set {z}; this is an independent truth.
    let reference = independent_adjusted_effect(&data);
    assert!(
        (result.estimate().as_effect().unwrap().ate - reference).abs() < 1e-9,
        "{coordinate}: ate={} against least squares {reference}",
        result.estimate().as_effect().unwrap().ate
    );
    // The sample estimate also sits within sampling error of the generating coefficient.
    assert!((reference - 2.0).abs() < 0.2, "{coordinate}: reference {reference}");
    let refreshed = rebind_estimate(&mut prepared, data.clone(), &ctx).unwrap();
    assert!(
        (refreshed.estimate().as_effect().unwrap().ate - reference).abs() < 1e-9,
        "{coordinate}: refreshed ate={}",
        refreshed.estimate().as_effect().unwrap().ate
    );
    assert_eq!(
        result.estimate().as_effect().unwrap().ate.to_bits(),
        refreshed.estimate().as_effect().unwrap().ate.to_bits()
    );
    assert!(
        matches!(prepared.retained_operation(), RetainedOperation::CheckedLinear(..)),
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

/// Ordinary least squares of `y` on `(1, t, z)` from the fixture's raw columns (ids 0, 1, 2 are
/// `t`, `y`, `z`), solved here by Gaussian elimination on the normal equations: the exact
/// back-door adjustment estimate for the adjustment set `{z}`.
fn independent_adjusted_effect(data: &antecedent_data::TabularData) -> f64 {
    let column = |id: usize| -> Vec<f64> {
        match &data.storage().columns()[id] {
            antecedent_data::OwnedColumn::Float64(c) => c.values.to_vec(),
            other => panic!("unexpected column {other:?}"),
        }
    };
    let (t, y, z) = (column(0), column(1), column(2));
    let mut a = [[0.0_f64; 4]; 3];
    for i in 0..t.len() {
        let row = [1.0, t[i], z[i]];
        for r in 0..3 {
            for c in 0..3 {
                a[r][c] += row[r] * row[c];
            }
            a[r][3] += row[r] * y[i];
        }
    }
    for pivot in 0..3 {
        let best = (pivot..3).max_by(|&p, &q| a[p][pivot].abs().total_cmp(&a[q][pivot].abs())).unwrap();
        a.swap(pivot, best);
        for r in pivot + 1..3 {
            let factor = a[r][pivot] / a[pivot][pivot];
            for c in pivot..4 {
                a[r][c] -= factor * a[pivot][c];
            }
        }
    }
    let mut beta = [0.0_f64; 3];
    for r in (0..3).rev() {
        let tail: f64 = (r + 1..3).map(|c| a[r][c] * beta[c]).sum();
        beta[r] = (a[r][3] - tail) / a[r][r];
    }
    beta[1]
}

/// Bind `data` on a new branch, materialize and estimate it, then adopt the branch as the handle.
/// A failure leaves the handle unchanged, as the immutable lifecycle does.
fn rebind_estimate(
    prepared: &mut antecedent::PreparedStudy,
    data: antecedent_data::TabularData,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<antecedent::StudyResult, antecedent::CausalError> {
    let branch = prepared.rebind_data(data, ctx)?.materialize(ctx)?;
    let result = branch.estimate_retained(ctx)?;
    *prepared = branch;
    Ok(result)
}
