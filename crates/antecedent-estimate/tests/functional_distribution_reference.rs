//! Empirical plug-in of identified functionals against exact rational evaluations of the identities.
//!
//! `oracle_data/functional_distribution_reference.py` (frozen in the JSON beside it) evaluates the
//! back-door adjustment (a ternary adjustment variable and ternary outcome with unbalanced and
//! singleton cells), the front-door formula (ternary mediator), the napkin ratio with a free
//! variable averaged over its empirical law, and a path-specific mediation functional from
//! contingency counts with `fractions.Fraction`, using no estimator code. The test identifies each
//! query with the ID machinery, builds the empirical tables from the expanded rows and compares
//! the distribution atoms, the interventional mean and the scalar effect. A table that lacks a
//! cell the identity needs must be refused for the contrast that reads it, never zero-filled.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "oracle_data/table_support.rs"]
mod table_support;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, CausalQuery, ExecutionContext, Intervention,
    InterventionalDistributionQuery, PathSpecificEffectQuery, RoleHint, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_estimate::{
    FunctionalDistribution, FunctionalDistributionWorkspace, FunctionalEffect,
};
use antecedent_graph::{Admg, Dag, DenseNodeId};
use antecedent_identify::{
    IdIdentifier, IdentificationResult, IdentificationWorkspace, PathSpecificIdentifier,
};
use serde_json::Value as Json;
use table_support::table;

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

fn cases() -> Vec<Json> {
    serde_json::from_str::<Json>(include_str!("oracle_data/functional_distribution_reference.json"))
        .unwrap()["cases"]
        .as_array()
        .unwrap()
        .clone()
}

fn case(name: &str) -> Json {
    cases().into_iter().find(|c| c["name"] == name).unwrap()
}

/// Expand the counted rows into a table whose column `k` holds variable `order[k]`.
fn expand(case: &Json, names: &[&str], order: &[&str]) -> TabularData {
    let mut columns: Vec<Vec<f64>> = vec![Vec::new(); order.len()];
    for row in case["rows"].as_array().unwrap() {
        let values = row[0].as_array().unwrap();
        for _ in 0..row[1].as_u64().unwrap() {
            for (k, name) in order.iter().enumerate() {
                let at = names.iter().position(|n| n == name).unwrap();
                columns[k].push(values[at].as_f64().unwrap());
            }
        }
    }
    let spec: Vec<(&str, RoleHint, Vec<f64>)> =
        order.iter().zip(columns).map(|(n, c)| (*n, RoleHint::Context, c)).collect();
    table(&spec)
}

fn close(got: f64, want: f64, what: &str) {
    assert!((got - want).abs() <= 1e-12, "{what}: got {got} want {want}");
}

fn distribution(
    id_result: &IdentificationResult,
    query: &InterventionalDistributionQuery,
    data: &TabularData,
) -> Result<
    antecedent_estimate::InterventionalDistributionEstimate,
    antecedent_estimate::EstimationError,
> {
    let estimator = FunctionalDistribution::new();
    let prepared = estimator.prepare(
        data,
        query,
        &id_result.estimands[0],
        &id_result.arena,
        id_result.required_assumptions.clone(),
    )?;
    estimator.estimate(
        &prepared,
        &[],
        &mut FunctionalDistributionWorkspace::default(),
        &ExecutionContext::for_tests(0),
    )
}

fn backdoor_dag() -> Dag {
    // t = 0, y = 1, z = 2
    let mut dag = Dag::with_variables(3);
    for (a, b) in [(2, 0), (2, 1), (0, 1)] {
        dag.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    dag
}

fn do_t(level: f64) -> (CausalQuery, InterventionalDistributionQuery) {
    let query =
        InterventionalDistributionQuery::new(v(1), [Intervention::set(v(0), Value::f64(level))]);
    (CausalQuery::Distribution(query.clone()), query)
}

#[test]
fn backdoor_distribution_atoms_mean_and_effect_match_the_exact_identity() {
    let case = case("backdoor_ternary_unbalanced");
    let data = expand(&case, &["z", "t", "y"], &["t", "y", "z"]);
    let id = IdIdentifier::new();
    let prep = id.prepare_dag(&backdoor_dag()).unwrap();
    let levels: Vec<f64> =
        case["y_levels"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
    for t in [0, 1] {
        let (causal, query) = do_t(f64::from(t));
        let identified =
            id.identify(&prep, &causal, &mut IdentificationWorkspace::default()).unwrap();
        let out = distribution(&identified, &query, &data).unwrap();
        let want = case[format!("dist_t{t}")].as_array().unwrap();
        assert_eq!(out.atoms.len(), levels.len());
        for (atom, (level, want)) in out.atoms.iter().zip(levels.iter().zip(want)) {
            let at = atom.outcomes[0].1.as_f64().unwrap();
            assert!((at - level).abs() < 1e-12, "atom order");
            close(atom.probability, want.as_f64().unwrap(), &format!("t={t} atom {level}"));
        }
        close(out.mean, case[format!("mean_t{t}")].as_f64().unwrap(), &format!("t={t} mean"));
    }
    // Scalar effect through the general ID functional.
    let identified = id
        .identify_ate(
            &prep,
            &AverageEffectQuery::binary_ate(v(0), v(1)),
            &mut IdentificationWorkspace::default(),
        )
        .unwrap();
    let estimator = FunctionalEffect::new();
    let prepared = estimator
        .prepare(&data, &identified.estimands[0], &identified.arena, AssumptionSet::new(), &[])
        .unwrap();
    let effect = estimator
        .estimate(
            &prepared,
            &mut FunctionalDistributionWorkspace::default(),
            &ExecutionContext::for_tests(0),
        )
        .unwrap();
    close(effect.ate, case["ate"].as_f64().unwrap(), "backdoor ate");
}

#[test]
fn a_cell_missing_under_one_arm_is_refused_for_that_contrast_only() {
    let case = case("backdoor_missing_control_cell");
    let data = expand(&case, &["z", "t", "y"], &["t", "y", "z"]);
    let id = IdIdentifier::new();
    let prep = id.prepare_dag(&backdoor_dag()).unwrap();
    let (causal, query) = do_t(1.0);
    let identified = id.identify(&prep, &causal, &mut IdentificationWorkspace::default()).unwrap();
    let out = distribution(&identified, &query, &data).unwrap();
    close(out.mean, case["mean_t1"].as_f64().unwrap(), "supported arm mean");
    let (causal, query) = do_t(0.0);
    let identified = id.identify(&prep, &causal, &mut IdentificationWorkspace::default()).unwrap();
    assert!(distribution(&identified, &query, &data).is_err(), "an unsupported cell must refuse");
    let identified = id
        .identify_ate(
            &prep,
            &AverageEffectQuery::binary_ate(v(0), v(1)),
            &mut IdentificationWorkspace::default(),
        )
        .unwrap();
    let estimator = FunctionalEffect::new();
    let prepared = estimator
        .prepare(&data, &identified.estimands[0], &identified.arena, AssumptionSet::new(), &[])
        .unwrap();
    assert!(
        estimator
            .estimate(
                &prepared,
                &mut FunctionalDistributionWorkspace::default(),
                &ExecutionContext::for_tests(0)
            )
            .is_err()
    );
}

#[test]
fn frontdoor_effect_matches_the_exact_formula() {
    let case = case("frontdoor_ternary_mediator");
    let data = expand(&case, &["t", "m", "y"], &["t", "y", "m"]);
    // t = 0, y = 1, m = 2; t -> m -> y and t <-> y.
    let mut admg = Admg::with_variables(3);
    admg.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    admg.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    admg.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let id = IdIdentifier::new();
    let prep = id.prepare(&admg).unwrap();
    let identified = id
        .identify_ate(
            &prep,
            &AverageEffectQuery::binary_ate(v(0), v(1)),
            &mut IdentificationWorkspace::default(),
        )
        .unwrap();
    let estimator = FunctionalEffect::new();
    let prepared = estimator
        .prepare(&data, &identified.estimands[0], &identified.arena, AssumptionSet::new(), &[])
        .unwrap();
    let effect = estimator
        .estimate(
            &prepared,
            &mut FunctionalDistributionWorkspace::default(),
            &ExecutionContext::for_tests(0),
        )
        .unwrap();
    close(effect.ate, case["ate"].as_f64().unwrap(), "front-door ate");
}

#[test]
fn napkin_free_variable_is_averaged_by_its_empirical_law() {
    let case = case("napkin_binary");
    // x = 0, y = 1, w1 = 2, w2 = 3; w2 -> w1 -> x -> y, w2 <-> x, w2 <-> y.
    let data = expand(&case, &["w2", "w1", "x", "y"], &["x", "y", "w1", "w2"]);
    let mut admg = Admg::with_variables(4);
    for (a, b) in [(3, 2), (2, 0), (0, 1)] {
        admg.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    admg.insert_bidirected(DenseNodeId::from_raw(3), DenseNodeId::from_raw(0)).unwrap();
    admg.insert_bidirected(DenseNodeId::from_raw(3), DenseNodeId::from_raw(1)).unwrap();
    let id = IdIdentifier::new();
    let prep = id.prepare(&admg).unwrap();
    for x in [0, 1] {
        let (causal, query) = {
            let query = InterventionalDistributionQuery::new(
                v(1),
                [Intervention::set(v(0), Value::f64(f64::from(x)))],
            );
            (CausalQuery::Distribution(query.clone()), query)
        };
        let identified =
            id.identify(&prep, &causal, &mut IdentificationWorkspace::default()).unwrap();
        let out = distribution(&identified, &query, &data).unwrap();
        let want = case[format!("dist_x{x}")].as_array().unwrap();
        for (atom, want) in out.atoms.iter().zip(want) {
            close(atom.probability, want.as_f64().unwrap(), &format!("napkin x={x}"));
        }
    }
}

#[test]
fn path_specific_mediation_functional_matches_the_exact_identity() {
    let case = case("mediation_edge");
    // t = 0, m = 1, y = 2, c = 3: c -> t, m, y; t -> m, y; m -> y; path through m.
    let data = expand(&case, &["c", "t", "m", "y"], &["t", "m", "y", "c"]);
    let mut dag = Dag::with_variables(4);
    for (a, b) in [(3, 0), (3, 1), (3, 2), (0, 1), (0, 2), (1, 2)] {
        dag.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    let id = PathSpecificIdentifier::new();
    let prep = id.prepare_dag(&dag).unwrap();
    let query = PathSpecificEffectQuery::binary(v(0), v(2)).with_path_nodes([v(1)]);
    let identified = id
        .identify(&prep, &CausalQuery::PathSpecific(query), &mut IdentificationWorkspace::default())
        .unwrap();
    let estimator = FunctionalEffect::new();
    let prepared = estimator
        .prepare(&data, &identified.estimands[0], &identified.arena, AssumptionSet::new(), &[])
        .unwrap();
    let effect = estimator
        .estimate(
            &prepared,
            &mut FunctionalDistributionWorkspace::default(),
            &ExecutionContext::for_tests(0),
        )
        .unwrap();
    close(effect.ate, case["nie"].as_f64().unwrap(), "path-specific effect");
}
