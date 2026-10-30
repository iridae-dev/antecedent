//! Coordinate-specific closure evidence for the mediation decomposition on a supplied ADMG, the
//! contemporaneous projection of a TemporalAdmg, a CPDAG and a PAG.
//!
//! Every expected value is read from `conformance/estimate/mediation_mixed_graphs`, where an
//! independent generator enumerates the noise cells of the structural models and evaluates the
//! outcome equation under each treatment and mediator-treatment argument (no identification
//! formula is involved); the retained tables are the exact observed law.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::RetainedOperation;
use antecedent::{
    AcceptedGraph, BayesianConfig, InferenceMode, RefuteSuite, StructureSource, Study,
};
use antecedent_contract::SlotAvailability;
use antecedent_core::{
    CausalQuery, ExecutionContext, MediationContrast, MediationQuery, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Admg, Cpdag, DenseNodeId, Pag, TemporalAdmg};
use antecedent_io::consume_analysis_result;
use serde_json::Value;

/// Posterior-mean tolerance around the enumerated truth on the retained tables.
const BAYESIAN_TOLERANCE: f64 = 3.0e-2;

fn pin() -> Value {
    serde_json::from_str(include_str!(
        "../../../conformance/estimate/mediation_mixed_graphs/expected.json"
    ))
    .unwrap()
}

/// The table's rows in a fixed interleaved order, so that consecutive rows (the blocks a
/// TemporalAdmg draw weights together) do not follow the sorted cells of the table.
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

fn query(case: &Value, pin: &Value, contrast: MediationContrast) -> MediationQuery {
    let query = &pin["query"];
    MediationQuery::binary(
        VariableId::from_raw(node(case, query["treatment"].as_str().unwrap())),
        VariableId::from_raw(node(case, query["outcome"].as_str().unwrap())),
        query["mediators"]
            .as_array()
            .unwrap()
            .iter()
            .map(|name| VariableId::from_raw(node(case, name.as_str().unwrap())))
            .collect::<Vec<_>>(),
        contrast,
    )
}

fn edges(case: &Value, key: &str) -> Vec<(DenseNodeId, DenseNodeId)> {
    case["graph"][key]
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

fn variable_count(case: &Value) -> u32 {
    u32::try_from(case["columns"].as_array().unwrap().len()).unwrap()
}

fn admg(case: &Value) -> Admg {
    let mut admg = Admg::with_variables(variable_count(case));
    for (from, to) in edges(case, "directed_edges") {
        admg.insert_directed(from, to).unwrap();
    }
    for (left, right) in edges(case, "bidirected_edges") {
        admg.insert_bidirected(left, right).unwrap();
    }
    admg
}

/// Every edge oriented: a class with a single completion.
fn compelled_cpdag(case: &Value) -> Cpdag {
    let mut cpdag = Cpdag::with_variables(variable_count(case));
    for (from, to) in edges(case, "directed_edges") {
        cpdag.insert_directed(from, to).unwrap();
    }
    cpdag
}

/// Every mark decided; `c -> t` and `w -> m` make the edges out of `t` and into `y` visible.
fn decided_pag(case: &Value) -> Pag {
    let mut pag = Pag::with_variables(variable_count(case));
    for (from, to) in edges(case, "directed_edges") {
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

fn builder_for(
    class: &str,
    accepted: bool,
    case: &Value,
    base: antecedent::analysis::StudyBuilder,
) -> antecedent::analysis::StudyBuilder {
    match (class, accepted) {
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
    }
}

fn case_of<'a>(fixture: &'a Value, class: &str) -> &'a Value {
    match class {
        "Admg" | "TemporalAdmg" => &fixture["covariate_latent"],
        "Cpdag" | "Pag" => &fixture["markov_parents"],
        other => panic!("unexpected graph class {other}"),
    }
}

fn base_study(
    data: &TabularData,
    query: MediationQuery,
    bayesian: bool,
    replicates: u32,
) -> antecedent::analysis::StudyBuilder {
    Study::tabular(data.clone())
        .query(CausalQuery::Mediation(query))
        .inference(if bayesian {
            InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(200).prior_scale(10.0))
        } else {
            InferenceMode::Frequentist
        })
        .refute(RefuteSuite::None)
        .bootstrap_replicates(replicates)
}

fn tolerance(fixture: &Value, bayesian: bool) -> f64 {
    if bayesian {
        assert!(
            BAYESIAN_TOLERANCE <= fixture["bayesian"]["posterior_mean_tolerance"].as_f64().unwrap()
        );
        BAYESIAN_TOLERANCE
    } else {
        fixture["frequentist"]["absolute_tolerance"].as_f64().unwrap()
    }
}

#[allow(clippy::too_many_lines, reason = "one function walks a coordinate's full lifecycle")]
fn verify_coordinate(
    coordinate: &str,
    class: &str,
    accepted: bool,
    bayesian: bool,
    validation: &str,
) {
    let fixture = pin();
    // The validation states resample the table, so they run on the same law at 40 times the rows.
    let case = &if validation == "none" {
        case_of(&fixture, class).clone()
    } else {
        scaled(case_of(&fixture, class), 40)
    };
    let data = expand(case);
    let query = query(case, &fixture, MediationContrast::NaturalDirect);
    let ctx = ExecutionContext::for_tests(2_129);
    let suite = match validation {
        "none" => RefuteSuite::None,
        "cheap" => RefuteSuite::Cheap,
        "full" => RefuteSuite::Full,
        other => panic!("unexpected validation {other}"),
    };
    let builder = builder_for(
        class,
        accepted,
        case,
        base_study(&data, query.clone(), bayesian, 0).refute(suite),
    );
    let study = builder.clone().build().unwrap_or_else(|error| panic!("{coordinate}: {error}"));
    let mut prepared = study.prepare(&ctx).unwrap_or_else(|error| panic!("{coordinate}: {error}"));
    drop(builder);
    drop(study);

    assert_eq!(prepared.query(), &CausalQuery::Mediation(query));
    assert_eq!(
        prepared.structure_source(),
        if accepted { StructureSource::Accepted } else { StructureSource::Explicit },
        "{coordinate} structure source"
    );
    assert_eq!(prepared.support_status(), Some(antecedent::CellStatus::Licensed), "{coordinate}");
    let contract = prepared.contract().unwrap();
    assert_eq!(contract.query_kind.as_ref(), "MediationEffect", "{coordinate}");
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
    let truth = &case["truth"];
    let tolerance = tolerance(&fixture, bayesian);
    let direct = truth["natural_direct_effect"].as_f64().unwrap();
    assert!(
        (result.effect() - direct).abs() < tolerance,
        "{coordinate}: requested contrast {} vs {direct}",
        result.effect()
    );
    let components = result.payload().mediation().expect("the decomposition");
    for (name, got, want) in [
        ("total", components.total, truth["total_effect"].as_f64().unwrap()),
        ("direct", components.direct, direct),
        ("indirect", components.mediated, truth["natural_indirect_effect"].as_f64().unwrap()),
    ] {
        let got = got.unwrap_or_else(|| panic!("{coordinate}: component {name} absent"));
        assert!((got - want).abs() < tolerance, "{coordinate}: {name} {got} vs {want}");
    }
    // The pure natural indirect effect is not the total effect minus the direct effect.
    let imputed =
        truth["total_effect"].as_f64().unwrap() - truth["natural_direct_effect"].as_f64().unwrap();
    let indirect = components.mediated.unwrap();
    assert!((indirect - imputed).abs() > 0.02, "{coordinate}: indirect imputed as total - direct");
    assert_validation_states(coordinate, &result, truth, validation);
    if bayesian && class != "Pag" {
        assert!(result.posterior.is_some(), "{coordinate} posterior missing");
    } else if !bayesian {
        assert!(result.posterior.is_none(), "{coordinate} unexpected posterior");
    }
    let refreshed = rebind_estimate(&mut prepared, data.clone(), &ctx).unwrap();
    assert_eq!(refreshed.effect().to_bits(), result.effect().to_bits(), "{coordinate} refresh");
    assert!(
        matches!(prepared.retained_operation(), RetainedOperation::StaticCrossWorld),
        "{coordinate} lost its operation on refresh"
    );
    let artifact = prepared
        .encode_contracted_result(&refreshed, &format!("mediation-{coordinate}"), &ctx)
        .unwrap();
    let consumed = consume_analysis_result(&artifact).unwrap();
    assert_eq!(consumed.body.estimate, Some(refreshed.effect()), "{coordinate} artifact scalar");
}

fn field(diagnostic: &antecedent_core::Diagnostic, key: &str) -> f64 {
    diagnostic
        .fields
        .iter()
        .find(|(k, _)| k.as_ref() == key)
        .unwrap_or_else(|| panic!("field {key}"))
        .1
        .parse()
        .unwrap()
}

/// Every requested contrast reads its own enumerated truth on each class of graph, and the
/// components evaluated at one resampled table are disclosed as a joint: the frequentist
/// bootstrap of a supplied mixed graph and the Bayesian-bootstrap draws of a CPDAG. (A class
/// resamples each frequentist component separately and a PAG discloses no joint.)
#[test]
fn every_contrast_equals_its_enumerated_truth_and_the_components_are_paired() {
    let fixture = pin();
    for (class, accepted, bayesian) in [
        ("Admg", false, false),
        ("TemporalAdmg", true, false),
        ("Cpdag", false, true),
        ("Pag", true, true),
    ] {
        let case = case_of(&fixture, class);
        let data = expand(case);
        for (contrast, key) in [
            (MediationContrast::Total, "total_effect"),
            (MediationContrast::NaturalDirect, "natural_direct_effect"),
            (MediationContrast::NaturalIndirect, "natural_indirect_effect"),
        ] {
            let result = builder_for(
                class,
                accepted,
                case,
                base_study(&data, query(case, &fixture, contrast), bayesian, 30),
            )
            .build()
            .unwrap()
            .run(&ExecutionContext::for_tests(2_130))
            .unwrap_or_else(|error| panic!("{class} {contrast:?}: {error}"));
            let want = case["truth"][key].as_f64().unwrap();
            assert!(
                (result.effect() - want).abs() < tolerance(&fixture, bayesian),
                "{class} {contrast:?}: {} vs {want}",
                result.effect()
            );
            let components = result.payload().mediation().expect("the decomposition");
            for (name, got, key) in [
                ("total", components.total, "total_effect"),
                ("direct", components.direct, "natural_direct_effect"),
                ("indirect", components.mediated, "natural_indirect_effect"),
            ] {
                let want = case["truth"][key].as_f64().unwrap();
                let got = got.unwrap_or_else(|| panic!("{class} {contrast:?}: {name} absent"));
                assert!(
                    (got - want).abs() < tolerance(&fixture, bayesian),
                    "{class} {contrast:?}: {name} {got} vs {want}"
                );
            }
            let joint =
                result.diagnostics.iter().find(|d| d.code.as_ref() == "estimate.mediation.joint");
            if class == "Pag" {
                // A PAG discloses the decomposition's components but no joint.
                assert!(joint.is_none(), "{class} {contrast:?}");
                continue;
            }
            let joint =
                joint.unwrap_or_else(|| panic!("{class} {contrast:?}: no joint disclosure"));
            assert!(field(joint, "draws") >= 20.0, "{class} {contrast:?}");
            // The paired replicates evaluate every component at one resampled table, so the
            // published component means stay near the enumerated components.
            for (published, truth_key) in [
                ("total_mean", "total_effect"),
                ("direct_mean", "natural_direct_effect"),
                ("indirect_mean", "natural_indirect_effect"),
            ] {
                let want = case["truth"][truth_key].as_f64().unwrap();
                assert!(
                    (field(joint, published) - want).abs() < 0.06,
                    "{class} {contrast:?}: {published} {} vs {want}",
                    field(joint, published)
                );
            }
        }
    }
}

/// A mediator-outcome confounder that the graph states (`m <-> y`) leaves the natural direct
/// effect unidentified: it is refused, never imputed.
#[test]
fn a_mediator_outcome_confounder_refuses_the_decomposition_it_breaks() {
    let fixture = pin();
    let case = &fixture["covariate_latent"];
    let mut graph = admg(case);
    graph
        .insert_bidirected(
            DenseNodeId::from_raw(node(case, "m")),
            DenseNodeId::from_raw(node(case, "y")),
        )
        .unwrap();
    let study = Study::tabular(expand(case))
        .graph(graph)
        .query(CausalQuery::Mediation(query(case, &fixture, MediationContrast::NaturalDirect)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .expect("the study builds; identification refuses at run");
    let error = study.run(&ExecutionContext::for_tests(2_131)).unwrap_err();
    assert!(matches!(&error, antecedent::CausalError::NotIdentified { .. }), "{error}");
}

#[test]
fn mediation_admg_explicit_frequentist_none() {
    verify_coordinate(
        "MediationEffect:Admg:explicit:Frequentist:none",
        "Admg",
        false,
        false,
        "none",
    );
}

#[test]
fn mediation_admg_explicit_bayesian_none() {
    verify_coordinate("MediationEffect:Admg:explicit:Bayesian:none", "Admg", false, true, "none");
}

#[test]
fn mediation_admg_accepted_frequentist_none() {
    verify_coordinate(
        "MediationEffect:Admg:accepted:Frequentist:none",
        "Admg",
        true,
        false,
        "none",
    );
}

#[test]
fn mediation_admg_accepted_bayesian_none() {
    verify_coordinate("MediationEffect:Admg:accepted:Bayesian:none", "Admg", true, true, "none");
}

#[test]
fn mediation_temporal_admg_explicit_frequentist_none() {
    verify_coordinate(
        "MediationEffect:TemporalAdmg:explicit:Frequentist:none",
        "TemporalAdmg",
        false,
        false,
        "none",
    );
}

#[test]
fn mediation_temporal_admg_explicit_bayesian_none() {
    verify_coordinate(
        "MediationEffect:TemporalAdmg:explicit:Bayesian:none",
        "TemporalAdmg",
        false,
        true,
        "none",
    );
}

#[test]
fn mediation_temporal_admg_accepted_frequentist_none() {
    verify_coordinate(
        "MediationEffect:TemporalAdmg:accepted:Frequentist:none",
        "TemporalAdmg",
        true,
        false,
        "none",
    );
}

#[test]
fn mediation_temporal_admg_accepted_bayesian_none() {
    verify_coordinate(
        "MediationEffect:TemporalAdmg:accepted:Bayesian:none",
        "TemporalAdmg",
        true,
        true,
        "none",
    );
}

#[test]
fn mediation_cpdag_explicit_frequentist_none() {
    verify_coordinate(
        "MediationEffect:Cpdag:explicit:Frequentist:none",
        "Cpdag",
        false,
        false,
        "none",
    );
}

#[test]
fn mediation_cpdag_explicit_bayesian_none() {
    verify_coordinate("MediationEffect:Cpdag:explicit:Bayesian:none", "Cpdag", false, true, "none");
}

#[test]
fn mediation_cpdag_accepted_frequentist_none() {
    verify_coordinate(
        "MediationEffect:Cpdag:accepted:Frequentist:none",
        "Cpdag",
        true,
        false,
        "none",
    );
}

#[test]
fn mediation_cpdag_accepted_bayesian_none() {
    verify_coordinate("MediationEffect:Cpdag:accepted:Bayesian:none", "Cpdag", true, true, "none");
}

#[test]
fn mediation_pag_explicit_frequentist_none() {
    verify_coordinate("MediationEffect:Pag:explicit:Frequentist:none", "Pag", false, false, "none");
}

#[test]
fn mediation_pag_explicit_bayesian_none() {
    verify_coordinate("MediationEffect:Pag:explicit:Bayesian:none", "Pag", false, true, "none");
}

#[test]
fn mediation_pag_accepted_frequentist_none() {
    verify_coordinate("MediationEffect:Pag:accepted:Frequentist:none", "Pag", true, false, "none");
}

#[test]
fn mediation_pag_accepted_bayesian_none() {
    verify_coordinate("MediationEffect:Pag:accepted:Bayesian:none", "Pag", true, true, "none");
}

/// A PAG whose ADMG candidates identify different functionals of the decomposition names one
/// functional of the combined proof; the functional used to be read from a member's own arena and
/// indexed past the combined one (a panic in the free-variable note). The table is uniform over
/// all 32 cells of `t, m, y, c, w`.
#[test]
fn a_pag_mediation_with_several_identified_functionals_runs_without_a_foreign_arena_index() {
    let mut columns: Vec<Vec<f64>> = vec![Vec::new(); 5];
    for bits in 0..32_usize {
        for _ in 0..(8 + bits % 5) {
            for (j, column) in columns.iter_mut().enumerate() {
                column.push(f64::from(u8::try_from((bits >> j) & 1).unwrap()));
            }
        }
    }
    let data = TabularData::from_f64_columns([
        ("t", columns[0].as_slice()),
        ("m", columns[1].as_slice()),
        ("y", columns[2].as_slice()),
        ("c", columns[3].as_slice()),
        ("w", columns[4].as_slice()),
    ])
    .unwrap();
    let mut pag = Pag::with_variables(5);
    for (from, to) in [(3, 0), (4, 1), (0, 1), (0, 2), (1, 2)] {
        pag.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    let query = MediationQuery::binary(
        VariableId::from_raw(0),
        VariableId::from_raw(2),
        [VariableId::from_raw(1)],
        MediationContrast::NaturalDirect,
    );
    let result = Study::tabular(data)
        .graph(pag)
        .query(CausalQuery::Mediation(query))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(2_132))
        .unwrap();
    // The cells are independent in every stratum, so the direct effect is zero.
    assert!(result.effect().abs() < 0.05, "{}", result.effect());
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

/// The mediation-native refuter states against the enumerated components: the mediator placebo
/// leaves the total effect in both the total and the direct component and removes the indirect
/// one, and every subset or bootstrap refit of a component returns that component. The
/// sequential-ignorability grid is a diagnostic the reference does not assert.
fn assert_validation_states(
    coordinate: &str,
    result: &antecedent::StudyResult,
    truth: &Value,
    validation: &str,
) {
    if validation == "none" {
        assert!(result.refutations.is_empty(), "{coordinate}");
        return;
    }
    let total = truth["total_effect"].as_f64().unwrap();
    let direct = truth["natural_direct_effect"].as_f64().unwrap();
    let indirect = truth["natural_indirect_effect"].as_f64().unwrap();
    let names: Vec<&str> = result.refutations.iter().map(|r| r.refuter.as_ref()).collect();
    for required in [
        "mediation.placebo_mediator_total",
        "mediation.placebo_mediator_indirect",
        "mediation.subset_direct",
        "mediation.joint_stability_subset",
    ] {
        assert!(names.contains(&required), "{coordinate}: {required} missing from {names:?}");
    }
    for report in &result.refutations {
        let name = report.refuter.as_ref();
        if name.contains("sequential_ignorability") {
            continue;
        }
        assert!(report.passed, "{coordinate}: {name} {:?}", report.failure_condition);
        let expected = if name.contains("placebo_mediator") {
            if name.ends_with("indirect") { 0.0 } else { total }
        } else if name.ends_with("indirect") {
            indirect
        } else if name.ends_with("total") {
            total
        } else if name.ends_with("direct") || name.contains("joint") {
            direct
        } else {
            panic!("{coordinate}: no enumerated value for refuter {name}")
        };
        assert!(
            (report.refuted_ate - expected).abs() < 0.03,
            "{coordinate}: {name} {} vs {expected}",
            report.refuted_ate
        );
    }
}

#[test]
fn mediation_admg_explicit_frequentist_cheap() {
    verify_coordinate(
        "MediationEffect:Admg:explicit:Frequentist:cheap",
        "Admg",
        false,
        false,
        "cheap",
    );
}

#[test]
fn mediation_admg_explicit_frequentist_full() {
    verify_coordinate(
        "MediationEffect:Admg:explicit:Frequentist:full",
        "Admg",
        false,
        false,
        "full",
    );
}

#[test]
fn mediation_admg_explicit_bayesian_cheap() {
    verify_coordinate("MediationEffect:Admg:explicit:Bayesian:cheap", "Admg", false, true, "cheap");
}

#[test]
fn mediation_admg_explicit_bayesian_full() {
    verify_coordinate("MediationEffect:Admg:explicit:Bayesian:full", "Admg", false, true, "full");
}

#[test]
fn mediation_admg_accepted_frequentist_cheap() {
    verify_coordinate(
        "MediationEffect:Admg:accepted:Frequentist:cheap",
        "Admg",
        true,
        false,
        "cheap",
    );
}

#[test]
fn mediation_admg_accepted_frequentist_full() {
    verify_coordinate(
        "MediationEffect:Admg:accepted:Frequentist:full",
        "Admg",
        true,
        false,
        "full",
    );
}

#[test]
fn mediation_admg_accepted_bayesian_cheap() {
    verify_coordinate("MediationEffect:Admg:accepted:Bayesian:cheap", "Admg", true, true, "cheap");
}

#[test]
fn mediation_admg_accepted_bayesian_full() {
    verify_coordinate("MediationEffect:Admg:accepted:Bayesian:full", "Admg", true, true, "full");
}

#[test]
fn mediation_temporal_admg_explicit_frequentist_cheap() {
    verify_coordinate(
        "MediationEffect:TemporalAdmg:explicit:Frequentist:cheap",
        "TemporalAdmg",
        false,
        false,
        "cheap",
    );
}

#[test]
fn mediation_temporal_admg_explicit_frequentist_full() {
    verify_coordinate(
        "MediationEffect:TemporalAdmg:explicit:Frequentist:full",
        "TemporalAdmg",
        false,
        false,
        "full",
    );
}

#[test]
fn mediation_temporal_admg_explicit_bayesian_cheap() {
    verify_coordinate(
        "MediationEffect:TemporalAdmg:explicit:Bayesian:cheap",
        "TemporalAdmg",
        false,
        true,
        "cheap",
    );
}

#[test]
fn mediation_temporal_admg_explicit_bayesian_full() {
    verify_coordinate(
        "MediationEffect:TemporalAdmg:explicit:Bayesian:full",
        "TemporalAdmg",
        false,
        true,
        "full",
    );
}

#[test]
fn mediation_temporal_admg_accepted_frequentist_cheap() {
    verify_coordinate(
        "MediationEffect:TemporalAdmg:accepted:Frequentist:cheap",
        "TemporalAdmg",
        true,
        false,
        "cheap",
    );
}

#[test]
fn mediation_temporal_admg_accepted_frequentist_full() {
    verify_coordinate(
        "MediationEffect:TemporalAdmg:accepted:Frequentist:full",
        "TemporalAdmg",
        true,
        false,
        "full",
    );
}

#[test]
fn mediation_temporal_admg_accepted_bayesian_cheap() {
    verify_coordinate(
        "MediationEffect:TemporalAdmg:accepted:Bayesian:cheap",
        "TemporalAdmg",
        true,
        true,
        "cheap",
    );
}

#[test]
fn mediation_temporal_admg_accepted_bayesian_full() {
    verify_coordinate(
        "MediationEffect:TemporalAdmg:accepted:Bayesian:full",
        "TemporalAdmg",
        true,
        true,
        "full",
    );
}

#[test]
fn mediation_cpdag_explicit_frequentist_cheap() {
    verify_coordinate(
        "MediationEffect:Cpdag:explicit:Frequentist:cheap",
        "Cpdag",
        false,
        false,
        "cheap",
    );
}

#[test]
fn mediation_cpdag_explicit_frequentist_full() {
    verify_coordinate(
        "MediationEffect:Cpdag:explicit:Frequentist:full",
        "Cpdag",
        false,
        false,
        "full",
    );
}

#[test]
fn mediation_cpdag_explicit_bayesian_cheap() {
    verify_coordinate(
        "MediationEffect:Cpdag:explicit:Bayesian:cheap",
        "Cpdag",
        false,
        true,
        "cheap",
    );
}

#[test]
fn mediation_cpdag_explicit_bayesian_full() {
    verify_coordinate("MediationEffect:Cpdag:explicit:Bayesian:full", "Cpdag", false, true, "full");
}

#[test]
fn mediation_cpdag_accepted_frequentist_cheap() {
    verify_coordinate(
        "MediationEffect:Cpdag:accepted:Frequentist:cheap",
        "Cpdag",
        true,
        false,
        "cheap",
    );
}

#[test]
fn mediation_cpdag_accepted_frequentist_full() {
    verify_coordinate(
        "MediationEffect:Cpdag:accepted:Frequentist:full",
        "Cpdag",
        true,
        false,
        "full",
    );
}

#[test]
fn mediation_cpdag_accepted_bayesian_cheap() {
    verify_coordinate(
        "MediationEffect:Cpdag:accepted:Bayesian:cheap",
        "Cpdag",
        true,
        true,
        "cheap",
    );
}

#[test]
fn mediation_cpdag_accepted_bayesian_full() {
    verify_coordinate("MediationEffect:Cpdag:accepted:Bayesian:full", "Cpdag", true, true, "full");
}

#[test]
fn mediation_pag_explicit_frequentist_cheap() {
    verify_coordinate(
        "MediationEffect:Pag:explicit:Frequentist:cheap",
        "Pag",
        false,
        false,
        "cheap",
    );
}

#[test]
fn mediation_pag_explicit_frequentist_full() {
    verify_coordinate("MediationEffect:Pag:explicit:Frequentist:full", "Pag", false, false, "full");
}

#[test]
fn mediation_pag_explicit_bayesian_cheap() {
    verify_coordinate("MediationEffect:Pag:explicit:Bayesian:cheap", "Pag", false, true, "cheap");
}

#[test]
fn mediation_pag_explicit_bayesian_full() {
    verify_coordinate("MediationEffect:Pag:explicit:Bayesian:full", "Pag", false, true, "full");
}

#[test]
fn mediation_pag_accepted_frequentist_cheap() {
    verify_coordinate(
        "MediationEffect:Pag:accepted:Frequentist:cheap",
        "Pag",
        true,
        false,
        "cheap",
    );
}

#[test]
fn mediation_pag_accepted_frequentist_full() {
    verify_coordinate("MediationEffect:Pag:accepted:Frequentist:full", "Pag", true, false, "full");
}

#[test]
fn mediation_pag_accepted_bayesian_cheap() {
    verify_coordinate("MediationEffect:Pag:accepted:Bayesian:cheap", "Pag", true, true, "cheap");
}

#[test]
fn mediation_pag_accepted_bayesian_full() {
    verify_coordinate("MediationEffect:Pag:accepted:Bayesian:full", "Pag", true, true, "full");
}
