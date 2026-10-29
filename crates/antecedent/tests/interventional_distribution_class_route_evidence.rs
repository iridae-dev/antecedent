//! Coordinate-specific closure evidence for an interventional distribution on a supplied or
//! accepted CPDAG and PAG.
//!
//! Every expected value is read from `conformance/estimate/interventional_distribution_classes`,
//! where an independent generator enumerates the noise cells of the structural models and
//! evaluates the outcome equation under the intervention (no identification formula is involved);
//! the retained tables are the exact observed law.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::RetainedOperation;
use antecedent::{
    AcceptedGraph, BayesianConfig, EstimatorId, IdentifierId, InferenceMode, RefuteSuite,
    StructureSource, Study,
};
use antecedent_contract::SlotAvailability;
use antecedent_core::{
    CausalQuery, ExecutionContext, Intervention, InterventionalDistributionQuery, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Cpdag, DenseNodeId, Pag};
use antecedent_io::consume_analysis_result;
use serde_json::Value as Json;

/// Dirichlet-posterior-mean tolerance around the enumerated truth on the retained table.
const BAYESIAN_TOLERANCE: f64 = 3.0e-2;

fn pin() -> Json {
    serde_json::from_str(include_str!(
        "../../../conformance/estimate/interventional_distribution_classes/expected.json"
    ))
    .unwrap()
}

fn expand(case: &Json) -> TabularData {
    let columns: Vec<&str> =
        case["columns"].as_array().unwrap().iter().map(|column| column.as_str().unwrap()).collect();
    let mut values: Vec<Vec<f64>> = vec![Vec::new(); columns.len()];
    for cell in case["contingency_table"].as_array().unwrap() {
        let count = usize::try_from(cell["count"].as_u64().unwrap()).unwrap();
        for (index, name) in columns.iter().enumerate() {
            values[index].extend(std::iter::repeat_n(cell[name].as_f64().unwrap(), count));
        }
    }
    let pairs: Vec<(&str, &[f64])> =
        columns.iter().zip(&values).map(|(name, values)| (*name, values.as_slice())).collect();
    TabularData::from_f64_columns(pairs).unwrap()
}

fn node(case: &Json, name: &str) -> u32 {
    let index = case["columns"]
        .as_array()
        .unwrap()
        .iter()
        .position(|column| column.as_str() == Some(name))
        .unwrap();
    u32::try_from(index).unwrap()
}

fn query(case: &Json, pin: &Json) -> InterventionalDistributionQuery {
    InterventionalDistributionQuery::new(
        VariableId::from_raw(node(case, pin["query"]["outcome"].as_str().unwrap())),
        [Intervention::set(
            VariableId::from_raw(node(case, "t")),
            Value::f64(pin["query"]["do"]["t"].as_f64().unwrap()),
        )],
    )
}

fn edges(case: &Json, list: &Json) -> Vec<(DenseNodeId, DenseNodeId)> {
    list.as_array()
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

fn variable_count(case: &Json) -> u32 {
    u32::try_from(case["columns"].as_array().unwrap().len()).unwrap()
}

/// The Markovian graph with every edge oriented: a class with a single completion.
fn compelled_cpdag(case: &Json) -> Cpdag {
    let mut cpdag = Cpdag::with_variables(variable_count(case));
    for (from, to) in edges(case, &case["graph"]["directed_edges"]) {
        cpdag.insert_directed(from, to).unwrap();
    }
    cpdag
}

/// The Markovian graph with every mark decided; the root `w` makes each edge out of the
/// treatment visible.
fn decided_pag(case: &Json) -> Pag {
    let mut pag = Pag::with_variables(variable_count(case));
    for (from, to) in edges(case, &case["graph"]["directed_edges"]) {
        pag.insert_directed(from, to).unwrap();
    }
    pag
}

fn disagreeing_cpdag(case: &Json) -> Cpdag {
    let mut cpdag = Cpdag::with_variables(variable_count(case));
    for (from, to) in edges(case, &case["cpdag"]["directed_edges"]) {
        cpdag.insert_directed(from, to).unwrap();
    }
    for (left, right) in edges(case, &case["cpdag"]["undirected_edges"]) {
        cpdag.insert_undirected(left, right).unwrap();
    }
    cpdag
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

fn study(
    data: &TabularData,
    query: &InterventionalDistributionQuery,
    bayesian: bool,
) -> antecedent::analysis::StudyBuilder {
    Study::tabular(data.clone())
        .query(CausalQuery::Distribution(query.clone()))
        .identifier(IdentifierId::GeneralId)
        .estimator(EstimatorId::FunctionalDistribution)
        .inference(if bayesian {
            InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(256))
        } else {
            InferenceMode::Frequentist
        })
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
}

#[allow(clippy::too_many_lines, reason = "one function walks a coordinate's full lifecycle")]
fn verify_coordinate(coordinate: &str, class: &str, accepted: bool, bayesian: bool) {
    let fixture = pin();
    let case = &fixture["class_point"];
    let data = expand(case);
    let query = query(case, &fixture);
    let ctx = ExecutionContext::for_tests(2_127);
    let base = study(&data, &query, bayesian);
    let builder = match (class, accepted) {
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

    assert_eq!(prepared.query(), &CausalQuery::Distribution(query));
    assert_eq!(
        prepared.structure_source(),
        if accepted { StructureSource::Accepted } else { StructureSource::Explicit },
        "{coordinate} structure source"
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
            "prepared handle must bind this exact coordinate"
        ),
        other => panic!("{coordinate} support unavailable: {other:?}"),
    }

    let result =
        prepared.estimate(&data, &ctx).unwrap_or_else(|error| panic!("{coordinate}: {error}"));
    let distribution = result.distribution().expect("distribution result");
    let truth = &case["truth"];
    let tolerance = if bayesian {
        assert!(
            BAYESIAN_TOLERANCE <= fixture["bayesian"]["posterior_mean_tolerance"].as_f64().unwrap()
        );
        BAYESIAN_TOLERANCE
    } else {
        fixture["frequentist"]["absolute_tolerance"].as_f64().unwrap()
    };
    let probability_of = |level: f64| {
        distribution
            .atoms
            .iter()
            .find(|atom| atom.outcomes.iter().any(|(_, value)| *value == Value::f64(level)))
            .unwrap_or_else(|| panic!("{coordinate}: no atom at {level}"))
            .probability
    };
    for (key, level) in [("y0", 0.0), ("y1", 1.0)] {
        let want = truth[key].as_f64().unwrap();
        assert!(
            (probability_of(level) - want).abs() < tolerance,
            "{coordinate}: P({key}) {} vs {want}",
            probability_of(level)
        );
    }
    // One completion (or one functional over the PAG's candidates) is a point, not an envelope.
    assert!(
        result
            .diagnostics
            .iter()
            .all(|d| d.code.as_ref() != "estimate.distribution.class_envelope"),
        "{coordinate}: a point distribution published an envelope"
    );
    if bayesian {
        assert!(result.posterior.is_some(), "{coordinate} posterior missing");
    } else {
        assert!(result.posterior.is_none(), "{coordinate} unexpected posterior");
    }
    let refreshed = rebind_estimate(&mut prepared, data.clone(), &ctx).unwrap();
    assert_eq!(
        refreshed.distribution().unwrap().mean.to_bits(),
        distribution.mean.to_bits(),
        "{coordinate} refresh"
    );
    assert!(
        matches!(
            prepared.retained_operation(),
            RetainedOperation::Distribution(..) | RetainedOperation::EstimatorRoute
        ),
        "{coordinate} lost its retained program on refresh"
    );
    let artifact = prepared
        .encode_contracted_result(&refreshed, &format!("distribution-{coordinate}"), &ctx)
        .unwrap();
    consume_analysis_result(&artifact).unwrap();
}

#[test]
fn distribution_cpdag_explicit_frequentist_none() {
    verify_coordinate(
        "InterventionalDistribution:Cpdag:explicit:Frequentist:none",
        "Cpdag",
        false,
        false,
    );
}

#[test]
fn distribution_cpdag_explicit_bayesian_none() {
    verify_coordinate(
        "InterventionalDistribution:Cpdag:explicit:Bayesian:none",
        "Cpdag",
        false,
        true,
    );
}

#[test]
fn distribution_cpdag_accepted_frequentist_none() {
    verify_coordinate(
        "InterventionalDistribution:Cpdag:accepted:Frequentist:none",
        "Cpdag",
        true,
        false,
    );
}

#[test]
fn distribution_cpdag_accepted_bayesian_none() {
    verify_coordinate(
        "InterventionalDistribution:Cpdag:accepted:Bayesian:none",
        "Cpdag",
        true,
        true,
    );
}

#[test]
fn distribution_pag_explicit_frequentist_none() {
    verify_coordinate(
        "InterventionalDistribution:Pag:explicit:Frequentist:none",
        "Pag",
        false,
        false,
    );
}

#[test]
fn distribution_pag_explicit_bayesian_none() {
    verify_coordinate("InterventionalDistribution:Pag:explicit:Bayesian:none", "Pag", false, true);
}

#[test]
fn distribution_pag_accepted_frequentist_none() {
    verify_coordinate(
        "InterventionalDistribution:Pag:accepted:Frequentist:none",
        "Pag",
        true,
        false,
    );
}

#[test]
fn distribution_pag_accepted_bayesian_none() {
    verify_coordinate("InterventionalDistribution:Pag:accepted:Bayesian:none", "Pag", true, true);
}

fn field(diagnostic: &antecedent_core::Diagnostic, name: &str) -> f64 {
    diagnostic
        .fields
        .iter()
        .find(|(key, _)| key.as_ref() == name)
        .unwrap_or_else(|| panic!("envelope field {name}"))
        .1
        .parse()
        .unwrap()
}

/// A class whose completions disagree publishes, atom by atom, the lowest and highest
/// probability the generator derives for the two completions, with their equal-weight mean.
#[test]
fn a_cpdag_whose_completions_disagree_publishes_the_enumerated_envelope() {
    let fixture = pin();
    let case = &fixture["class_disagree"];
    let data = expand(case);
    let query = query(case, &fixture);
    let result = study(&data, &query, false)
        .graph(disagreeing_cpdag(case))
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(2_128))
        .unwrap();
    let envelope = result
        .diagnostics
        .iter()
        .find(|d| d.code.as_ref() == "estimate.distribution.class_envelope")
        .expect("the partially identified distribution discloses its envelope");
    assert_eq!(field(envelope, "members") as usize, case["completions"].as_u64().unwrap() as usize);
    assert!(field(envelope, "unidentified_mass").abs() < 1e-12);
    let truth = &case["truth"];
    let mut lowers = Vec::new();
    let mut uppers = Vec::new();
    let mut means = Vec::new();
    for index in 0..2 {
        lowers.push(field(envelope, &format!("atom_{index}_lower")));
        uppers.push(field(envelope, &format!("atom_{index}_upper")));
        means.push(field(envelope, &format!("atom_{index}_mean")));
    }
    for values in [&mut lowers, &mut uppers, &mut means] {
        values.sort_by(f64::total_cmp);
    }
    let expected = |statistic: &str| {
        let mut values: Vec<f64> =
            ["y0", "y1"].iter().map(|key| truth[key][statistic].as_f64().unwrap()).collect();
        values.sort_by(f64::total_cmp);
        values
    };
    for (got, want) in
        [(lowers, expected("lower")), (uppers, expected("upper")), (means, expected("mean"))]
    {
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-9, "{got:?} vs {want:?}");
        }
    }
    // The published atoms are the enumeration-weighted means, never a probability over members.
    let distribution = result.distribution().unwrap();
    let published = distribution
        .atoms
        .iter()
        .find(|atom| atom.outcomes.iter().any(|(_, value)| *value == Value::f64(1.0)))
        .unwrap()
        .probability;
    assert!((published - truth["y1"]["mean"].as_f64().unwrap()).abs() < 1e-9, "{published}");
}
