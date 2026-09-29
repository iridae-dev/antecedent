//! Coordinate-specific closure evidence for ADMG functional-effect execution.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{
    AcceptedGraph, BayesianConfig, EstimatorId, IdentifierId, InferenceMode, RefuteSuite,
    StructureSource, Study,
};
use antecedent_core::{AverageEffectQuery, ExecutionContext, SlotAvailability};
use antecedent_data::TabularData;
use antecedent_graph::{Admg, DenseNodeId};
use antecedent_io::consume_analysis_result;

fn vid(raw: u32) -> antecedent_core::VariableId {
    antecedent_core::VariableId::from_raw(raw)
}

fn functional_admg_fixture() -> (TabularData, Admg, AverageEffectQuery) {
    let mut treatment = Vec::new();
    let mut outcome = Vec::new();
    let mut confounder = Vec::new();
    let mut latent_left = Vec::new();
    let mut latent_right = Vec::new();
    // P(z=1)=1/2. The outcome risks are .2/.8 when z=0 and
    // .4/.6 when z=1 for t=0/1 respectively, so the independent
    // adjustment formula gives E[Y|do(t=1)] - E[Y|do(t=0)] = .4.
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
        for _ in 0..count {
            for latent in [0.0, 1.0] {
                treatment.push(t);
                outcome.push(y);
                confounder.push(z);
                latent_left.push(latent);
                latent_right.push(latent);
            }
        }
    }
    let data = TabularData::from_f64_columns([
        ("t", treatment.as_slice()),
        ("y", outcome.as_slice()),
        ("z", confounder.as_slice()),
        ("u", latent_left.as_slice()),
        ("v", latent_right.as_slice()),
    ])
    .unwrap();
    let mut graph = Admg::with_variables(5);
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    // A disconnected latent component keeps this an ADMG while leaving the
    // target causal effect identified by adjustment on z.
    graph.insert_bidirected(DenseNodeId::from_raw(3), DenseNodeId::from_raw(4)).unwrap();
    (data, graph, AverageEffectQuery::binary_ate(vid(0), vid(1)))
}

#[allow(clippy::too_many_lines, reason = "one function walks a coordinate's full lifecycle")]
fn verify_functional_effect_coordinate(
    coordinate: &str,
    accepted: bool,
    bayesian: bool,
    validation: &str,
) {
    let ctx = ExecutionContext::for_tests(2_119);
    let (data, graph, query) = functional_admg_fixture();
    let suite = match validation {
        "none" => RefuteSuite::None,
        "cheap" => RefuteSuite::Cheap,
        "full" => RefuteSuite::Full,
        other => panic!("unexpected validation suite {other}"),
    };
    let base = Study::tabular(data.clone())
        .query(query.clone())
        .identifier(IdentifierId::GeneralId)
        .estimator(EstimatorId::FunctionalEffect)
        .inference(if bayesian {
            InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(256))
        } else {
            InferenceMode::Frequentist
        })
        .refute(suite)
        .bootstrap_replicates(0);
    let builder = if accepted { base.graph(AcceptedGraph::from(graph)) } else { base.graph(graph) };
    let study = builder.clone().build().unwrap_or_else(|error| panic!("{coordinate}: {error}"));
    let mut prepared = study.prepare(&ctx).unwrap_or_else(|error| panic!("{coordinate}: {error}"));
    drop(builder);
    drop(study);

    assert_eq!(prepared.query(), &antecedent_core::CausalQuery::AverageEffect(query));
    assert_eq!(
        prepared.structure_source(),
        if accepted { StructureSource::Accepted } else { StructureSource::Explicit },
        "{coordinate} structure source"
    );
    assert_eq!(prepared.support_status(), Some(antecedent::CellStatus::Licensed));
    let contract = prepared.contract().unwrap();
    assert_eq!(contract.query_kind.as_ref(), "AverageEffect", "{coordinate}");
    assert_eq!(contract.identifier.as_deref(), Some("general.id"), "{coordinate}");
    assert_eq!(contract.estimator.as_deref(), Some("functional.effect"), "{coordinate}");
    assert_eq!(contract.inference.as_ref(), if bayesian { "bayesian" } else { "frequentist" });
    match &contract.reasoning.support {
        SlotAvailability::Available(slot) => assert_eq!(
            slot.matrix_coordinate.as_deref(),
            Some(coordinate),
            "prepared handle must bind this exact coordinate"
        ),
        other => panic!("{coordinate} support unavailable: {other:?}"),
    }
    assert_eq!(prepared.plan().logical.record.identifier.as_deref(), Some("general.id"));
    assert_eq!(prepared.plan().logical.record.estimator.as_deref(), Some("functional.effect"));
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

    let program = prepared
        .checked_functional_effect_program()
        .unwrap_or_else(|| panic!("{coordinate} omitted checked functional-effect program"));
    assert_eq!(program.mapping().source, program.mapping().executable, "{coordinate}");

    let result =
        prepared.estimate(&data, &ctx).unwrap_or_else(|error| panic!("{coordinate}: {error}"));
    assert!(
        (result.effect() - 0.4).abs() < if bayesian { 0.08 } else { 1e-12 },
        "{coordinate}: {}",
        result.effect()
    );
    if bayesian {
        assert!(result.posterior.is_some(), "{coordinate} posterior missing");
    } else {
        assert!(result.posterior.is_none(), "{coordinate} unexpected posterior");
    }
    let refreshed = prepared.refresh(data.clone(), &ctx).unwrap();
    assert_eq!(refreshed.effect().to_bits(), result.effect().to_bits(), "{coordinate} refresh");
    assert!(
        prepared.checked_functional_effect_program().is_some(),
        "{coordinate} lost program on refresh"
    );

    let artifact = prepared
        .encode_contracted_result(&refreshed, &format!("functional-effect-{coordinate}"), &ctx)
        .unwrap();
    let consumed = consume_analysis_result(&artifact).unwrap();
    if bayesian {
        assert_eq!(
            consumed.acceptance.unresolved.len(),
            1,
            "{coordinate} should have one precise posterior replay dependency"
        );
        assert!(
            consumed.acceptance.unresolved.iter().any(|reason| {
                reason.as_ref() == "dependencies.functional_effect_posterior_draws"
            }),
            "{coordinate} artifact dependencies: {:?}",
            consumed.acceptance.unresolved
        );
        assert!(!consumed.acceptance.accepts_as_verified_program(), "{coordinate}");
    } else {
        assert!(
            consumed.acceptance.accepts_as_verified_program(),
            "{coordinate} artifact dependencies: {:?}",
            consumed.acceptance.unresolved
        );
    }
    assert_eq!(consumed.body.estimate, Some(refreshed.effect()), "{coordinate} artifact result");
}

#[test]
fn admg_default_estimator_replay_uses_resolved_commitment() {
    let (data, graph, query) = functional_admg_fixture();
    let ctx = ExecutionContext::for_tests(2_119);
    let builder = Study::tabular(data.clone())
        .graph(graph)
        .query(query)
        .identifier(IdentifierId::GeneralId)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0);
    let study = builder.clone().build().unwrap();
    let mut prepared = study.prepare(&ctx).unwrap();
    drop(builder);
    drop(study);

    assert_eq!(prepared.plan().logical.record.estimator.as_deref(), Some("functional.effect"));
    assert!(prepared.contract().unwrap().estimator.is_none());
    let result = prepared.estimate(&data, &ctx).unwrap();
    assert!((result.effect() - 0.4).abs() < 1e-12);
    let refreshed = prepared.refresh(data, &ctx).unwrap();
    assert_eq!(refreshed.effect().to_bits(), result.effect().to_bits());
    let bytes = prepared
        .encode_contracted_result(&refreshed, "admg-default-functional-effect", &ctx)
        .unwrap();
    let consumed = consume_analysis_result(&bytes).unwrap();
    assert!(
        consumed.acceptance.accepts_as_verified_program(),
        "{:?}",
        consumed.acceptance.unresolved
    );
    assert_eq!(consumed.body.estimate, Some(refreshed.effect()));
    let contract = consumed.contract.as_ref().expect("portable effect contract");
    assert!(contract.estimator.is_none(), "estimator was unexpectedly caller-selected");
    assert_eq!(
        contract
            .program
            .as_ref()
            .and_then(|program| program.commitments.resolved_estimator.as_deref()),
        Some("functional.effect"),
        "consumer must resolve the default estimator from program commitments"
    );
}

#[test]
fn admg_accepted_bayesian_cheap() {
    verify_functional_effect_coordinate(
        "AverageEffect:Admg:accepted:Bayesian:cheap",
        true,
        true,
        "cheap",
    );
}

#[test]
fn admg_accepted_bayesian_full() {
    verify_functional_effect_coordinate(
        "AverageEffect:Admg:accepted:Bayesian:full",
        true,
        true,
        "full",
    );
}

#[test]
fn admg_accepted_bayesian_none() {
    verify_functional_effect_coordinate(
        "AverageEffect:Admg:accepted:Bayesian:none",
        true,
        true,
        "none",
    );
}

#[test]
fn admg_accepted_frequentist_cheap() {
    verify_functional_effect_coordinate(
        "AverageEffect:Admg:accepted:Frequentist:cheap",
        true,
        false,
        "cheap",
    );
}

#[test]
fn admg_accepted_frequentist_full() {
    verify_functional_effect_coordinate(
        "AverageEffect:Admg:accepted:Frequentist:full",
        true,
        false,
        "full",
    );
}

#[test]
fn admg_accepted_frequentist_none() {
    verify_functional_effect_coordinate(
        "AverageEffect:Admg:accepted:Frequentist:none",
        true,
        false,
        "none",
    );
}

#[test]
fn admg_explicit_bayesian_cheap() {
    verify_functional_effect_coordinate(
        "AverageEffect:Admg:explicit:Bayesian:cheap",
        false,
        true,
        "cheap",
    );
}

#[test]
fn admg_explicit_bayesian_full() {
    verify_functional_effect_coordinate(
        "AverageEffect:Admg:explicit:Bayesian:full",
        false,
        true,
        "full",
    );
}

#[test]
fn admg_explicit_bayesian_none() {
    verify_functional_effect_coordinate(
        "AverageEffect:Admg:explicit:Bayesian:none",
        false,
        true,
        "none",
    );
}

#[test]
fn admg_explicit_frequentist_cheap() {
    verify_functional_effect_coordinate(
        "AverageEffect:Admg:explicit:Frequentist:cheap",
        false,
        false,
        "cheap",
    );
}

#[test]
fn admg_explicit_frequentist_full() {
    verify_functional_effect_coordinate(
        "AverageEffect:Admg:explicit:Frequentist:full",
        false,
        false,
        "full",
    );
}

#[test]
fn admg_explicit_frequentist_none() {
    verify_functional_effect_coordinate(
        "AverageEffect:Admg:explicit:Frequentist:none",
        false,
        false,
        "none",
    );
}
