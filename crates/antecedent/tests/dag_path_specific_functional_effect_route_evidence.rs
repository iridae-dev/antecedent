//! Coordinate-specific closure evidence for DAG path-specific execution.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{
    AcceptedGraph, BayesianConfig, EstimatorId, IdentifierId, InferenceMode, RefuteSuite,
    StructureSource, Study,
};
use antecedent_core::{
    CausalQuery, ExecutionContext, PathSpecificEffectQuery, SlotAvailability, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_io::consume_analysis_result;
use serde_json::Value;

fn pin() -> Value {
    serde_json::from_str(include_str!(
        "../../../conformance/estimate/path_specific_edge_gformula/expected.json"
    ))
    .unwrap()
}

fn expand(pin: &Value) -> TabularData {
    let columns: Vec<&str> =
        pin["columns"].as_array().unwrap().iter().map(|column| column.as_str().unwrap()).collect();
    let mut values: Vec<Vec<f64>> = vec![Vec::new(); columns.len()];
    for cell in pin["contingency_table"].as_array().unwrap() {
        let count = usize::try_from(cell["count"].as_u64().unwrap()).unwrap();
        for (index, name) in columns.iter().enumerate() {
            values[index].extend(std::iter::repeat_n(cell[name].as_f64().unwrap(), count));
        }
    }
    let pairs: Vec<(&str, &[f64])> =
        columns.iter().zip(&values).map(|(name, values)| (*name, values.as_slice())).collect();
    TabularData::from_f64_columns(pairs).unwrap()
}

fn variable(pin: &Value, name: &str) -> VariableId {
    let index = pin["columns"]
        .as_array()
        .unwrap()
        .iter()
        .position(|column| column.as_str() == Some(name))
        .unwrap();
    VariableId::from_raw(u32::try_from(index).unwrap())
}

fn graph(pin: &Value) -> Dag {
    let mut dag =
        Dag::with_variables(u32::try_from(pin["columns"].as_array().unwrap().len()).unwrap());
    for edge in pin["graph"]["directed_edges"].as_array().unwrap() {
        dag.insert_directed(
            DenseNodeId::from_raw(variable(pin, edge[0].as_str().unwrap()).raw()),
            DenseNodeId::from_raw(variable(pin, edge[1].as_str().unwrap()).raw()),
        )
        .unwrap();
    }
    dag
}

fn query(pin: &Value) -> PathSpecificEffectQuery {
    let query = &pin["query"];
    PathSpecificEffectQuery::binary(
        variable(pin, query["treatment"].as_str().unwrap()),
        variable(pin, query["outcome"].as_str().unwrap()),
    )
    .with_path_nodes(
        query["path_nodes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|name| variable(pin, name.as_str().unwrap()))
            .collect::<Vec<_>>(),
    )
}

fn verify_path_specific_coordinate(
    coordinate: &str,
    accepted: bool,
    bayesian: bool,
    validation: &str,
) {
    let fixture = pin();
    let data = expand(&fixture);
    let query = query(&fixture);
    let dag = graph(&fixture);
    let ctx = ExecutionContext::for_tests(2_123);
    let suite = match validation {
        "none" => RefuteSuite::None,
        "cheap" => RefuteSuite::Cheap,
        "full" => RefuteSuite::Full,
        other => panic!("unexpected validation suite {other}"),
    };
    let base = Study::tabular(data.clone())
        .query(CausalQuery::PathSpecific(query.clone()))
        .identifier(IdentifierId::PathSpecificNatural)
        .estimator(EstimatorId::FunctionalEffect)
        .inference(if bayesian {
            InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(100).prior_scale(10.0))
        } else {
            InferenceMode::Frequentist
        })
        .refute(suite)
        .bootstrap_replicates(0);
    let builder = if accepted { base.graph(AcceptedGraph::from(dag)) } else { base.graph(dag) };
    let study = builder.clone().build().unwrap_or_else(|error| panic!("{coordinate}: {error}"));
    let mut prepared = study.prepare(&ctx).unwrap_or_else(|error| panic!("{coordinate}: {error}"));
    drop(builder);
    drop(study);

    assert_eq!(prepared.query(), &CausalQuery::PathSpecific(query));
    assert_eq!(
        prepared.structure_source(),
        if accepted { StructureSource::Accepted } else { StructureSource::Explicit },
        "{coordinate} structure source"
    );
    assert_eq!(prepared.support_status(), Some(antecedent::CellStatus::Licensed));
    let contract = prepared.contract().unwrap();
    assert_eq!(contract.query_kind.as_ref(), "PathSpecificEffect", "{coordinate}");
    assert_eq!(contract.identifier.as_deref(), Some("path_specific.natural"), "{coordinate}");
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
    assert_eq!(prepared.plan().logical.record.identifier.as_deref(), Some("path_specific.natural"));
    assert_eq!(prepared.plan().logical.record.estimator.as_deref(), Some("functional.effect"));
    assert_eq!(
        prepared.plan().logical.record.validation_suite.as_deref(),
        match validation {
            "none" => None,
            "cheap" => Some("path.cheap"),
            "full" => Some("path.full"),
            _ => unreachable!(),
        },
        "{coordinate} validation choice"
    );
    let program = prepared
        .checked_functional_effect_program()
        .unwrap_or_else(|| panic!("{coordinate} omitted its retained checked effect program"));
    assert_eq!(program.mapping().source, program.mapping().executable, "{coordinate}");

    let result =
        prepared.estimate(&data, &ctx).unwrap_or_else(|error| panic!("{coordinate}: {error}"));
    let truth = fixture["truth"]["path_specific_effect"].as_f64().unwrap();
    let tolerance = if bayesian {
        fixture["bayesian"]["posterior_mean_tolerance"].as_f64().unwrap()
    } else {
        fixture["frequentist"]["absolute_tolerance"].as_f64().unwrap()
    };
    assert!(
        (result.effect() - truth).abs() < tolerance,
        "{coordinate}: {} vs {truth}",
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
        .encode_contracted_result(&refreshed, &format!("path-specific-{coordinate}"), &ctx)
        .unwrap();
    let consumed = consume_analysis_result(&artifact).unwrap();
    if bayesian {
        assert_eq!(consumed.acceptance.unresolved.len(), 1, "{coordinate} precise refusal");
        assert!(
            consumed.acceptance.unresolved.iter().any(|reason| {
                reason.as_ref() == "dependencies.functional_effect_posterior_draws"
            }),
            "{coordinate}: {:?}",
            consumed.acceptance.unresolved
        );
        assert!(!consumed.acceptance.accepts_as_verified_program(), "{coordinate}");
    } else {
        assert!(
            consumed.acceptance.accepts_as_verified_program(),
            "{coordinate}: {:?}",
            consumed.acceptance.unresolved
        );
    }
    assert_eq!(consumed.body.estimate, Some(refreshed.effect()), "{coordinate} artifact scalar");
}

#[test]
fn path_specific_dag_accepted_bayesian_cheap() {
    verify_path_specific_coordinate(
        "PathSpecificEffect:Dag:accepted:Bayesian:cheap",
        true,
        true,
        "cheap",
    );
}

#[test]
fn path_specific_dag_accepted_bayesian_full() {
    verify_path_specific_coordinate(
        "PathSpecificEffect:Dag:accepted:Bayesian:full",
        true,
        true,
        "full",
    );
}

#[test]
fn path_specific_dag_accepted_bayesian_none() {
    verify_path_specific_coordinate(
        "PathSpecificEffect:Dag:accepted:Bayesian:none",
        true,
        true,
        "none",
    );
}

#[test]
fn path_specific_dag_accepted_frequentist_cheap() {
    verify_path_specific_coordinate(
        "PathSpecificEffect:Dag:accepted:Frequentist:cheap",
        true,
        false,
        "cheap",
    );
}

#[test]
fn path_specific_dag_accepted_frequentist_full() {
    verify_path_specific_coordinate(
        "PathSpecificEffect:Dag:accepted:Frequentist:full",
        true,
        false,
        "full",
    );
}

#[test]
fn path_specific_dag_accepted_frequentist_none() {
    verify_path_specific_coordinate(
        "PathSpecificEffect:Dag:accepted:Frequentist:none",
        true,
        false,
        "none",
    );
}

#[test]
fn path_specific_dag_explicit_bayesian_cheap() {
    verify_path_specific_coordinate(
        "PathSpecificEffect:Dag:explicit:Bayesian:cheap",
        false,
        true,
        "cheap",
    );
}

#[test]
fn path_specific_dag_explicit_bayesian_full() {
    verify_path_specific_coordinate(
        "PathSpecificEffect:Dag:explicit:Bayesian:full",
        false,
        true,
        "full",
    );
}

#[test]
fn path_specific_dag_explicit_bayesian_none() {
    verify_path_specific_coordinate(
        "PathSpecificEffect:Dag:explicit:Bayesian:none",
        false,
        true,
        "none",
    );
}

#[test]
fn path_specific_dag_explicit_frequentist_cheap() {
    verify_path_specific_coordinate(
        "PathSpecificEffect:Dag:explicit:Frequentist:cheap",
        false,
        false,
        "cheap",
    );
}

#[test]
fn path_specific_dag_explicit_frequentist_full() {
    verify_path_specific_coordinate(
        "PathSpecificEffect:Dag:explicit:Frequentist:full",
        false,
        false,
        "full",
    );
}

#[test]
fn path_specific_dag_explicit_frequentist_none() {
    verify_path_specific_coordinate(
        "PathSpecificEffect:Dag:explicit:Frequentist:none",
        false,
        false,
        "none",
    );
}
