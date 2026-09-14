//! Path-specific natural effect with a complementary path: the edge g-formula.
//!
//! `conformance/estimate/path_specific_edge_gformula` freezes the exact law of a
//! known binary SCM (`c -> t, c -> m, c -> y, t -> m, t -> y, m -> y`) whose
//! path-specific effect through `m` (0.12) differs from both the total effect
//! (0.356) and the natural direct effect (0.156). `reference.py` next to the
//! fixture derives the truth from the structural equations and re-derives the
//! edge g-formula from the table with numpy.
//!
//! The fixture's `shared_descendant` case pins a node on both a selected and an
//! unselected path that is not a recanting witness (`t -> a -> w -> y`
//! selected, `t -> b -> w -> y` not): path effect 0.125, total effect 0.0625.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use antecedent::{AcceptedGraph, BayesianConfig, InferenceMode, RefuteSuite, Study, StudyResult};
use antecedent_core::{CausalQuery, ExecutionContext, PathSpecificEffectQuery, VariableId};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};

fn pin() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../conformance/estimate/path_specific_edge_gformula/expected.json"
    ))
    .unwrap()
}

fn column_names(pin: &serde_json::Value) -> Vec<String> {
    pin["columns"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_owned()).collect()
}

fn expand(pin: &serde_json::Value) -> TabularData {
    let columns = column_names(pin);
    let mut values: Vec<Vec<f64>> = vec![Vec::new(); columns.len()];
    for cell in pin["contingency_table"].as_array().unwrap() {
        let count = usize::try_from(cell["count"].as_u64().unwrap()).unwrap();
        for (i, name) in columns.iter().enumerate() {
            values[i].extend(std::iter::repeat_n(cell[name.as_str()].as_f64().unwrap(), count));
        }
    }
    let pairs: Vec<(&str, &[f64])> =
        columns.iter().zip(values.iter()).map(|(n, c)| (n.as_str(), c.as_slice())).collect();
    TabularData::from_f64_columns(pairs).unwrap()
}

fn index(pin: &serde_json::Value, name: &str) -> u32 {
    u32::try_from(column_names(pin).iter().position(|c| c == name).unwrap()).unwrap()
}

fn graph(pin: &serde_json::Value) -> Dag {
    let mut dag = Dag::with_variables(u32::try_from(column_names(pin).len()).unwrap());
    for edge in pin["graph"]["directed_edges"].as_array().unwrap() {
        let from = index(pin, edge[0].as_str().unwrap());
        let to = index(pin, edge[1].as_str().unwrap());
        dag.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    dag
}

fn query(pin: &serde_json::Value) -> PathSpecificEffectQuery {
    let q = &pin["query"];
    let var = |name: &serde_json::Value| VariableId::from_raw(index(pin, name.as_str().unwrap()));
    PathSpecificEffectQuery::binary(var(&q["treatment"]), var(&q["outcome"]))
        .with_path_nodes(q["path_nodes"].as_array().unwrap().iter().map(var).collect::<Vec<_>>())
}

fn study(
    pin: &serde_json::Value,
    accepted: bool,
    inference: InferenceMode,
    suite: RefuteSuite,
) -> Study {
    let data = expand(pin);
    let builder = if accepted {
        Study::tabular(data).graph(AcceptedGraph::from(graph(pin)))
    } else {
        Study::tabular(data).graph(graph(pin))
    };
    builder
        .query(CausalQuery::PathSpecific(query(pin)))
        .inference(inference)
        .refute(suite)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
}

fn bayes() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(100).prior_scale(10.0))
}

fn assert_edge_g_formula_identification(result: &StudyResult, label: &str) {
    assert_eq!(result.logical_plan.identifier.as_deref(), Some("path_specific.natural"), "{label}");
    assert_eq!(result.logical_plan.estimator.as_deref(), Some("functional.effect"), "{label}");
    assert_eq!(result.estimand.method.as_ref(), "path_specific.natural", "{label}");
    assert!(
        result
            .identification
            .derivation
            .steps
            .iter()
            .any(|s| s.rule.as_ref() == "path_specific.edge_gformula"),
        "{label}: identification must use the edge g-formula, got {:?}",
        result.identification.derivation.steps
    );
}

#[test]
fn frequentist_edge_g_formula_matches_reference_fresh_and_prepared() {
    check_frequentist(&pin());
}

/// `w` sits on the selected `t -> a -> w -> y` and the unselected
/// `t -> b -> w -> y`; the paths leave `t` through different children, so `w`
/// is not a recanting witness and the edge g-formula recovers the structural
/// path effect.
#[test]
fn shared_descendant_through_distinct_children_matches_reference() {
    check_frequentist(&pin()["shared_descendant"]);
}

#[test]
fn shared_descendant_bayesian_posterior_centres_on_reference() {
    check_bayesian(&pin()["shared_descendant"]);
}

fn check_frequentist(pin: &serde_json::Value) {
    let truth = pin["frequentist"]["expected_effect"].as_f64().unwrap();
    let tol = pin["frequentist"]["absolute_tolerance"].as_f64().unwrap();
    let total = pin["truth"]["total_effect"].as_f64().unwrap();
    let ctx = ExecutionContext::for_tests(7);
    let data = expand(pin);
    for accepted in [false, true] {
        for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
            let built = study(pin, accepted, InferenceMode::Frequentist, suite);
            let fresh = built.clone().run(&ctx).unwrap();
            let click = built.prepare(&ctx).unwrap().estimate(&data, &ctx).unwrap();
            for (result, path) in [(&fresh, "fresh"), (&click, "prepared")] {
                let label = format!("accepted={accepted} suite={suite:?} {path}");
                assert_edge_g_formula_identification(result, &label);
                assert!(
                    (result.estimate.ate - truth).abs() < tol,
                    "{label}: effect {} vs edge g-formula {truth} (total effect {total})",
                    result.estimate.ate
                );
                match suite {
                    RefuteSuite::None => assert!(result.refutations.is_empty(), "{label}"),
                    _ => assert!(!result.refutations.is_empty(), "{label}"),
                }
            }
            assert!(
                click.diagnostics.iter().any(|d| d.code.as_ref() == "exec.identify.cached"),
                "prepared click must reuse identification"
            );
            assert_eq!(fresh.estimate.ate.to_bits(), click.estimate.ate.to_bits());
        }
    }
}

#[test]
fn bayesian_edge_g_formula_posterior_centres_on_reference_fresh_and_prepared() {
    check_bayesian(&pin());
}

fn check_bayesian(pin: &serde_json::Value) {
    let truth = pin["truth"]["path_specific_effect"].as_f64().unwrap();
    let tol = pin["bayesian"]["posterior_mean_tolerance"].as_f64().unwrap();
    let total = pin["truth"]["total_effect"].as_f64().unwrap();
    let ctx = ExecutionContext::for_tests(7);
    let data = expand(pin);
    for accepted in [false, true] {
        for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
            let built = study(pin, accepted, bayes(), suite);
            let fresh = built.clone().run(&ctx).unwrap();
            let click = built.prepare(&ctx).unwrap().estimate(&data, &ctx).unwrap();
            for (result, path) in [(&fresh, "fresh"), (&click, "prepared")] {
                let label = format!("accepted={accepted} suite={suite:?} {path}");
                assert_edge_g_formula_identification(result, &label);
                let posterior = result.posterior.as_ref().expect("functional posterior");
                assert_eq!(posterior.diagnostics.backend_id.as_ref(), "functional.dirichlet");
                assert!(
                    (result.estimate.ate - truth).abs() < tol,
                    "{label}: posterior mean {} vs edge g-formula {truth} (total effect {total})",
                    result.estimate.ate
                );
                let col = posterior.effect_column().expect("effect draws");
                let mut draws = posterior.draws.column(col).unwrap().to_vec();
                draws.sort_by(f64::total_cmp);
                let at = |q: f64| draws[((draws.len() - 1) as f64 * q).round() as usize];
                assert!(at(0.05) <= truth && truth <= at(0.95), "{label}: 90% interval");
                assert!(
                    !(at(0.05) <= total && total <= at(0.95)),
                    "{label}: interval must exclude the total effect"
                );
            }
            assert!((fresh.estimate.ate - click.estimate.ate).abs() < 1e-12);
        }
    }
}

/// A recanting witness (`w` on a selected and a complementary path) still
/// refuses: the edge g-formula does not apply there.
#[test]
fn recanting_witness_still_refuses() {
    // 0=t 1=w 2=a 3=b 4=c 5=y; t->c->y, t->w->b->y, t->w->a->y. Selected: through `a`.
    let n = 400;
    let cols: Vec<Vec<f64>> = (0..6)
        .map(|j| (0..n).map(|i| f64::from(u8::from((i * (j + 3) / 7 + j) % 2 == 0))).collect())
        .collect();
    let names = ["t", "w", "a", "b", "c", "y"];
    let pairs: Vec<(&str, &[f64])> =
        names.iter().zip(cols.iter()).map(|(n, c)| (*n, c.as_slice())).collect();
    let data = TabularData::from_f64_columns(pairs).unwrap();
    let mut dag = Dag::with_variables(6);
    for (u, v) in [(0, 4), (4, 5), (0, 1), (1, 3), (3, 5), (1, 2), (2, 5)] {
        dag.insert_directed(DenseNodeId::from_raw(u), DenseNodeId::from_raw(v)).unwrap();
    }
    let query = PathSpecificEffectQuery::binary(VariableId::from_raw(0), VariableId::from_raw(5))
        .with_path_nodes([VariableId::from_raw(2)]);
    for inference in [InferenceMode::Frequentist, bayes()] {
        let err = Study::tabular(data.clone())
            .graph(dag.clone())
            .query(CausalQuery::PathSpecific(query.clone()))
            .inference(inference)
            .refute(RefuteSuite::None)
            .build()
            .unwrap()
            .run(&ExecutionContext::for_tests(1))
            .expect_err("recanting witness must refuse");
        assert!(err.to_string().to_lowercase().contains("not identified"), "{err}");
    }
}
