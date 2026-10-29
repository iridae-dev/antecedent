//! Coordinate-specific closure evidence for path-specific execution on a supplied ADMG, the
//! contemporaneous projection of a TemporalAdmg, a CPDAG and a PAG.
//!
//! Every expected value is read from `conformance/estimate/path_specific_mixed_graphs`, where an
//! independent generator enumerates the noise cells of the structural models and evaluates the
//! cross-world assignment on each (no identification formula is involved); the retained tables are
//! the exact observed law.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::RetainedOperation;
use antecedent::{
    AcceptedGraph, BayesianConfig, InferenceMode, RefuteSuite, StructureSource, Study,
};
use antecedent_contract::SlotAvailability;
use antecedent_core::{CausalQuery, ExecutionContext, PathSpecificEffectQuery, VariableId};
use antecedent_data::TabularData;
use antecedent_graph::{Admg, Cpdag, DenseNodeId, Pag, TemporalAdmg};
use antecedent_io::consume_analysis_result;
use serde_json::Value;

/// Posterior-mean tolerance around the enumerated truth on the retained tables.
const BAYESIAN_TOLERANCE: f64 = 2.0e-2;

fn pin() -> Value {
    serde_json::from_str(include_str!(
        "../../../conformance/estimate/path_specific_mixed_graphs/expected.json"
    ))
    .unwrap()
}

fn expand(case: &Value) -> TabularData {
    let columns: Vec<&str> =
        case["columns"].as_array().unwrap().iter().map(|column| column.as_str().unwrap()).collect();
    let mut values: Vec<Vec<f64>> = vec![Vec::new(); columns.len()];
    for cell in case["contingency_table"].as_array().unwrap() {
        let count = usize::try_from(cell["count"].as_u64().unwrap()).unwrap();
        for (index, name) in columns.iter().enumerate() {
            values[index].extend(std::iter::repeat_n(cell[name].as_f64().unwrap(), count));
        }
    }
    // The rows are read in a fixed interleaved order, so that consecutive rows (the blocks a
    // TemporalAdmg draw weights together) do not follow the sorted cells of the table.
    let rows = values[0].len();
    let order: Vec<usize> = (0..rows).map(|row| (row * 97) % rows).collect();
    assert!(order.iter().collect::<std::collections::BTreeSet<_>>().len() == rows);
    let shuffled: Vec<Vec<f64>> =
        values.iter().map(|column| order.iter().map(|&row| column[row]).collect()).collect();
    let pairs: Vec<(&str, &[f64])> =
        columns.iter().zip(&shuffled).map(|(name, values)| (*name, values.as_slice())).collect();
    TabularData::from_f64_columns(pairs).unwrap()
}

fn node(case: &Value, name: &str) -> u32 {
    let index = case["columns"]
        .as_array()
        .unwrap()
        .iter()
        .position(|column| column.as_str() == Some(name))
        .unwrap();
    u32::try_from(index).unwrap()
}

fn query(case: &Value) -> PathSpecificEffectQuery {
    let query = &case["query"];
    PathSpecificEffectQuery::binary(
        VariableId::from_raw(node(case, query["treatment"].as_str().unwrap())),
        VariableId::from_raw(node(case, query["outcome"].as_str().unwrap())),
    )
    .with_path_nodes(
        query["path_nodes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|name| VariableId::from_raw(node(case, name.as_str().unwrap())))
            .collect::<Vec<_>>(),
    )
    .with_max_len(4)
}

fn edges(case: &Value, graph: &Value, key: &str) -> Vec<(DenseNodeId, DenseNodeId)> {
    graph[key]
        .as_array()
        .map(|list| {
            list.iter()
                .map(|edge| {
                    (
                        DenseNodeId::from_raw(node(case, edge[0].as_str().unwrap())),
                        DenseNodeId::from_raw(node(case, edge[1].as_str().unwrap())),
                    )
                })
                .collect()
        })
        .unwrap_or_default()
}

fn admg(case: &Value) -> Admg {
    let graph = &case["graph"];
    let mut admg =
        Admg::with_variables(u32::try_from(case["columns"].as_array().unwrap().len()).unwrap());
    for (from, to) in edges(case, graph, "directed_edges") {
        admg.insert_directed(from, to).unwrap();
    }
    for (left, right) in edges(case, graph, "bidirected_edges") {
        admg.insert_bidirected(left, right).unwrap();
    }
    admg
}

/// The Markovian graph with every edge oriented, as a class with a single completion.
fn compelled_cpdag(case: &Value) -> Cpdag {
    let mut cpdag =
        Cpdag::with_variables(u32::try_from(case["columns"].as_array().unwrap().len()).unwrap());
    for (from, to) in edges(case, &case["graph"], "directed_edges") {
        cpdag.insert_directed(from, to).unwrap();
    }
    cpdag
}

/// The Markovian graph with every mark decided: each edge out of the treatment and into the
/// outcome is visible because the root `w` points into the treatment.
fn decided_pag(case: &Value) -> Pag {
    let mut pag =
        Pag::with_variables(u32::try_from(case["columns"].as_array().unwrap().len()).unwrap());
    for (from, to) in edges(case, &case["graph"], "directed_edges") {
        pag.insert_directed(from, to).unwrap();
    }
    pag
}

fn rebind_estimate(
    prepared: &mut antecedent::PreparedStudy,
    data: TabularData,
    ctx: &ExecutionContext,
) -> Result<antecedent::StudyResult, antecedent::CausalError> {
    let branch = prepared.rebind_data(data, ctx)?.materialize(ctx)?;
    let result = branch.estimate_retained(ctx)?;
    *prepared = branch;
    Ok(result)
}

#[allow(clippy::too_many_lines, reason = "one function walks a coordinate's full lifecycle")]
fn verify_coordinate(coordinate: &str, class: &str, accepted: bool, bayesian: bool) {
    let fixture = pin();
    let case = match class {
        "Admg" | "TemporalAdmg" => &fixture["district_admg"],
        "Cpdag" | "Pag" => &fixture["routed_markov"],
        other => panic!("unexpected graph class {other}"),
    };
    let data = expand(case);
    let query = query(case);
    let ctx = ExecutionContext::for_tests(2_124);
    let base = Study::tabular(data.clone())
        .query(CausalQuery::PathSpecific(query.clone()))
        .inference(if bayesian {
            InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(100).prior_scale(10.0))
        } else {
            InferenceMode::Frequentist
        })
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0);
    let builder = match (class, accepted) {
        ("Admg", false) => base.graph(admg(case)),
        ("Admg", true) => base.graph(AcceptedGraph::from(admg(case))),
        ("TemporalAdmg", false) => {
            base.graph(TemporalAdmg::from_static_contemporaneous(&admg(case)).unwrap())
        }
        ("TemporalAdmg", true) => base.graph(AcceptedGraph::from(
            TemporalAdmg::from_static_contemporaneous(&admg(case)).unwrap(),
        )),
        ("Cpdag", false) => base.graph(compelled_cpdag(case)),
        ("Cpdag", true) => base.graph(AcceptedGraph::from(compelled_cpdag(case))),
        ("Pag", false) => base.graph(decided_pag(case)),
        ("Pag", true) => base.graph(AcceptedGraph::from(decided_pag(case))),
        _ => unreachable!(),
    };
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
    assert_eq!(prepared.support_status(), Some(antecedent::CellStatus::Licensed), "{coordinate}");
    let (contract, inspection_work) =
        antecedent_expr::execution_counts::count_static_work(|| prepared.contract().unwrap());
    assert_eq!(inspection_work.factor_builds, 0, "{coordinate}: contract scanned factors");
    assert_eq!(inspection_work.provider_bindings, 0, "{coordinate}: contract rebound providers");
    assert_eq!(contract.query_kind.as_ref(), "PathSpecificEffect", "{coordinate}");
    assert_eq!(contract.inference.as_ref(), if bayesian { "bayesian" } else { "frequentist" });
    match &contract.reasoning.support {
        SlotAvailability::Available(slot) => assert_eq!(
            slot.matrix_coordinate.as_deref(),
            Some(coordinate),
            "prepared handle must bind this exact coordinate"
        ),
        other => panic!("{coordinate} support unavailable: {other:?}"),
    }
    assert!(
        matches!(prepared.retained_operation(), RetainedOperation::StaticCrossWorld),
        "{coordinate} retains its static cross-world operation"
    );

    let result =
        prepared.estimate(&data, &ctx).unwrap_or_else(|error| panic!("{coordinate}: {error}"));
    let truth = case["truth"]["path_specific_effect"].as_f64().unwrap();
    let tolerance = if bayesian {
        assert!(
            BAYESIAN_TOLERANCE <= fixture["bayesian"]["posterior_mean_tolerance"].as_f64().unwrap()
        );
        BAYESIAN_TOLERANCE
    } else {
        fixture["frequentist"]["absolute_tolerance"].as_f64().unwrap()
    };
    assert!(
        (result.effect() - truth).abs() < tolerance,
        "{coordinate}: {} vs {truth}",
        result.effect()
    );
    // A discriminator: neither the total effect nor the unselected-path effect is the answer.
    let total = case["truth"]["total_effect"].as_f64().unwrap();
    assert!((result.effect() - total).abs() > 0.05, "{coordinate}: read the total effect");
    let refreshed = rebind_estimate(&mut prepared, data.clone(), &ctx).unwrap();
    assert_eq!(refreshed.effect().to_bits(), result.effect().to_bits(), "{coordinate} refresh");
    assert!(
        matches!(prepared.retained_operation(), RetainedOperation::StaticCrossWorld),
        "{coordinate} lost its operation on refresh"
    );

    let artifact = prepared
        .encode_contracted_result(&refreshed, &format!("path-specific-{coordinate}"), &ctx)
        .unwrap();
    let consumed = consume_analysis_result(&artifact).unwrap();
    assert_eq!(consumed.body.estimate, Some(refreshed.effect()), "{coordinate} artifact scalar");
}

#[test]
fn path_specific_admg_explicit_frequentist_none() {
    verify_coordinate("PathSpecificEffect:Admg:explicit:Frequentist:none", "Admg", false, false);
}

#[test]
fn path_specific_admg_explicit_bayesian_none() {
    verify_coordinate("PathSpecificEffect:Admg:explicit:Bayesian:none", "Admg", false, true);
}

#[test]
fn path_specific_admg_accepted_frequentist_none() {
    verify_coordinate("PathSpecificEffect:Admg:accepted:Frequentist:none", "Admg", true, false);
}

#[test]
fn path_specific_admg_accepted_bayesian_none() {
    verify_coordinate("PathSpecificEffect:Admg:accepted:Bayesian:none", "Admg", true, true);
}

#[test]
fn path_specific_temporal_admg_explicit_frequentist_none() {
    verify_coordinate(
        "PathSpecificEffect:TemporalAdmg:explicit:Frequentist:none",
        "TemporalAdmg",
        false,
        false,
    );
}

#[test]
fn path_specific_temporal_admg_explicit_bayesian_none() {
    verify_coordinate(
        "PathSpecificEffect:TemporalAdmg:explicit:Bayesian:none",
        "TemporalAdmg",
        false,
        true,
    );
}

#[test]
fn path_specific_temporal_admg_accepted_frequentist_none() {
    verify_coordinate(
        "PathSpecificEffect:TemporalAdmg:accepted:Frequentist:none",
        "TemporalAdmg",
        true,
        false,
    );
}

#[test]
fn path_specific_temporal_admg_accepted_bayesian_none() {
    verify_coordinate(
        "PathSpecificEffect:TemporalAdmg:accepted:Bayesian:none",
        "TemporalAdmg",
        true,
        true,
    );
}

#[test]
fn path_specific_cpdag_explicit_frequentist_none() {
    verify_coordinate("PathSpecificEffect:Cpdag:explicit:Frequentist:none", "Cpdag", false, false);
}

#[test]
fn path_specific_cpdag_explicit_bayesian_none() {
    verify_coordinate("PathSpecificEffect:Cpdag:explicit:Bayesian:none", "Cpdag", false, true);
}

#[test]
fn path_specific_cpdag_accepted_frequentist_none() {
    verify_coordinate("PathSpecificEffect:Cpdag:accepted:Frequentist:none", "Cpdag", true, false);
}

#[test]
fn path_specific_cpdag_accepted_bayesian_none() {
    verify_coordinate("PathSpecificEffect:Cpdag:accepted:Bayesian:none", "Cpdag", true, true);
}

#[test]
fn path_specific_pag_explicit_frequentist_none() {
    verify_coordinate("PathSpecificEffect:Pag:explicit:Frequentist:none", "Pag", false, false);
}

#[test]
fn path_specific_pag_explicit_bayesian_none() {
    verify_coordinate("PathSpecificEffect:Pag:explicit:Bayesian:none", "Pag", false, true);
}

#[test]
fn path_specific_pag_accepted_frequentist_none() {
    verify_coordinate("PathSpecificEffect:Pag:accepted:Frequentist:none", "Pag", true, false);
}

#[test]
fn path_specific_pag_accepted_bayesian_none() {
    verify_coordinate("PathSpecificEffect:Pag:accepted:Bayesian:none", "Pag", true, true);
}

/// The class whose completions disagree publishes the identified set the generator derives: the
/// generating completion's effect below, the adjusted completion's above, and the completion with
/// no directed path through the selected child as unidentified mass.
#[test]
fn a_collider_class_publishes_the_enumerated_identified_set() {
    let fixture = pin();
    let case = &fixture["collider_class"];
    let mut cpdag = Cpdag::with_variables(4);
    for edge in case["cpdag"]["directed_edges"].as_array().unwrap() {
        cpdag
            .insert_directed(
                DenseNodeId::from_raw(node(case, edge[0].as_str().unwrap())),
                DenseNodeId::from_raw(node(case, edge[1].as_str().unwrap())),
            )
            .unwrap();
    }
    for edge in case["cpdag"]["undirected_edges"].as_array().unwrap() {
        cpdag
            .insert_undirected(
                DenseNodeId::from_raw(node(case, edge[0].as_str().unwrap())),
                DenseNodeId::from_raw(node(case, edge[1].as_str().unwrap())),
            )
            .unwrap();
    }
    let result = Study::tabular(expand(case))
        .graph(cpdag)
        .query(CausalQuery::PathSpecific(query(case)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(2_125))
        .unwrap();
    let mixture = result.structural_response.as_ref().expect("a partially identified class");
    let set = mixture.identified_set.as_ref().expect("an identified set over the completions");
    let truth = &case["truth"];
    assert!((set.lower[0] - truth["identified_set_lower"].as_f64().unwrap()).abs() < 1e-9);
    assert!((set.upper[0] - truth["identified_set_upper"].as_f64().unwrap()).abs() < 1e-9);
    assert!(
        (mixture.unidentified_mass - truth["unidentified_mass"].as_f64().unwrap()).abs() < 1e-12,
        "{}",
        mixture.unidentified_mass
    );
}

/// A latent common cause of the selected child and the complementary child puts both in one
/// district: the path is a recanting district and is refused, never estimated.
#[test]
fn a_recanting_district_is_refused_with_a_typed_reason() {
    let fixture = pin();
    let case = &fixture["district_admg"];
    let mut graph = admg(case);
    graph
        .insert_bidirected(
            DenseNodeId::from_raw(node(case, "a")),
            DenseNodeId::from_raw(node(case, "b")),
        )
        .unwrap();
    let study = Study::tabular(expand(case))
        .graph(graph)
        .query(CausalQuery::PathSpecific(query(case)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .expect("the study builds; identification refuses at run");
    let error = study.run(&ExecutionContext::for_tests(2_126)).unwrap_err();
    assert!(
        matches!(&error, antecedent::CausalError::NotIdentified { .. })
            && error.to_string().contains("reason=effect_not_identified"),
        "{error}"
    );
}

/// The case's table with every count multiplied by `scale`, the same law on more rows.
fn scaled(case: &Value, scale: u64) -> Value {
    let mut case = case.clone();
    for cell in case["contingency_table"].as_array_mut().unwrap() {
        let count = cell["count"].as_u64().unwrap();
        cell["count"] = Value::from(count * scale);
    }
    case
}

/// On the exact-law table the refuter states run and pass, and the cross-world placebo is
/// declared not run: the district formula sums the selected child out, so permuting its column
/// cannot move the effect. On the unscaled 256-row table a bootstrap resample can empty a cell
/// the functional reads; that refuter is refused with its reason instead of aborting the suite.
#[test]
fn the_validation_states_run_on_the_exact_law_and_decline_what_they_cannot_check() {
    let fixture = pin();
    let notes = |result: &antecedent::StudyResult| -> Vec<String> {
        result.diagnostics.iter().map(|d| d.message.to_string()).collect()
    };
    for (scale, suite, expected) in
        [(1, RefuteSuite::Full, 4), (40, RefuteSuite::Cheap, 3), (40, RefuteSuite::Full, 5)]
    {
        let case = scaled(&fixture["district_admg"], scale);
        let result = Study::tabular(expand(&case))
            .graph(admg(&case))
            .query(CausalQuery::PathSpecific(query(&case)))
            .refute(suite)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .run(&ExecutionContext::for_tests(5))
            .unwrap();
        assert_eq!(result.refutations.len(), expected, "scale {scale} {suite:?}");
        for report in &result.refutations {
            assert!(report.passed, "scale {scale} {suite:?}: {}", report.refuter);
            assert!((report.original_ate - 0.125).abs() < 1e-12);
        }
        assert!(
            notes(&result).iter().any(|note| note.contains("target.cross_world_placebo")
                && note.contains("does not read the permuted mediator column")),
            "scale {scale} {suite:?}"
        );
        if scale == 1 {
            assert!(
                notes(&result).iter().any(|note| note.contains("required probability cell empty")),
                "the refused bootstrap states its reason"
            );
        }
    }
}
