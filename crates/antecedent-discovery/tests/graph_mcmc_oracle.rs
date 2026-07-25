//! Exact exhaustive targets for order, structure, and CI-screened graph MCMC.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation)]

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use antecedent_core::{
    CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
    VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
};
use antecedent_discovery::{
    CiScreenedPosterior, DiscoveryConstraints, DiscoveryWorkspace, GraphPosterior, GraphPrior,
    OrderMcmc, StructureMcmc,
};
use antecedent_state::GraphScoreFamily;
use serde_json::Value as JsonValue;

fn fixture() -> JsonValue {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../conformance/discovery/graph_mcmc_exact/expected.json");
    serde_json::from_str(&fs::read_to_string(path).expect("graph MCMC fixture"))
        .expect("parse graph MCMC fixture")
}

fn case_data(case: &JsonValue) -> (TabularData, Vec<VariableId>, Vec<String>) {
    let names: Vec<String> = case["var_names"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect();
    let rows = case["data"].as_array().unwrap();
    let n = rows.len();
    let mut builder = CausalSchemaBuilder::new();
    for name in &names {
        builder
            .add_variable(
                name.as_str(),
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
    }
    let schema = builder.build().unwrap();
    let columns: Vec<OwnedColumn> = (0..names.len())
        .map(|column| {
            let values: Vec<f64> =
                rows.iter().map(|row| row.as_array().unwrap()[column].as_f64().unwrap()).collect();
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(column as u32),
                    Arc::from(values),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            )
        })
        .collect();
    let storage = OwnedColumnarStorage::try_new(schema, columns, None, None).unwrap();
    let variables = (0..names.len()).map(|i| VariableId::from_raw(i as u32)).collect();
    (TabularData::new(storage), variables, names)
}

fn named_marginals(post: &GraphPosterior, names: &[String]) -> BTreeMap<String, f64> {
    let mut out = BTreeMap::new();
    for from in 0..names.len() {
        for to in 0..names.len() {
            if from != to {
                out.insert(
                    format!("{}->{}", names[from], names[to]),
                    post.edge_marginals[from * names.len() + to],
                );
            }
        }
    }
    out
}

fn case_data_permuted(
    case: &JsonValue,
    perm: &[usize],
) -> (TabularData, Vec<VariableId>, Vec<String>) {
    let names: Vec<String> = case["var_names"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect();
    let names: Vec<String> = perm.iter().map(|&i| names[i].clone()).collect();
    let rows = case["data"].as_array().unwrap();
    let n = rows.len();
    let mut builder = CausalSchemaBuilder::new();
    for name in &names {
        builder
            .add_variable(
                name.as_str(),
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
    }
    let schema = builder.build().unwrap();
    let columns: Vec<OwnedColumn> = perm
        .iter()
        .enumerate()
        .map(|(out_col, &src_col)| {
            let values: Vec<f64> =
                rows.iter().map(|row| row.as_array().unwrap()[src_col].as_f64().unwrap()).collect();
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(out_col as u32),
                    Arc::from(values),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            )
        })
        .collect();
    let storage = OwnedColumnarStorage::try_new(schema, columns, None, None).unwrap();
    let variables = (0..names.len()).map(|i| VariableId::from_raw(i as u32)).collect();
    (TabularData::new(storage), variables, names)
}

fn expected(target: &JsonValue) -> BTreeMap<String, f64> {
    target["edge_marginals"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(key, value)| (key.clone(), value.as_f64().unwrap()))
        .collect()
}

fn schedule(fixture: &JsonValue) -> (u32, u32, u32, u32) {
    let value = &fixture["acceptance"]["schedule"];
    (
        value["chains"].as_u64().unwrap() as u32,
        value["warmup"].as_u64().unwrap() as u32,
        value["draws"].as_u64().unwrap() as u32,
        value["thin"].as_u64().unwrap() as u32,
    )
}

fn max_error(actual: &BTreeMap<String, f64>, target: &BTreeMap<String, f64>) -> f64 {
    actual.iter().map(|(key, value)| (value - target[key]).abs()).fold(0.0, f64::max)
}

#[test]
fn graph_mcmc_marginals_against_exact_exhaustive_targets() {
    // MM-011: Order MCMC subtracts log|#topo orders| so edge marginals match the
    // exhaustive uniform-DAG Gaussian-BIC target (Structure MCMC already did).
    let fixture = fixture();
    let (chains, warmup, draws, thin) = schedule(&fixture);
    let band = fixture["acceptance"]["absolute_edge_marginal_band"].as_f64().unwrap();
    let seeds: Vec<u64> = fixture["acceptance"]["seeds"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();
    for case in fixture["cases"].as_array().unwrap() {
        let (data, variables, names) = case_data(case);
        let target = expected(&case["targets"]["uniform_all_dags"]);
        for &seed in &seeds {
            let mut workspace = DiscoveryWorkspace::default();
            let structure = StructureMcmc::new()
                .with_schedule(chains, warmup, draws, thin)
                .with_diagnostics_gate(false);
            let structure_post = structure
                .run(
                    &data,
                    &variables,
                    &GraphPrior::uniform(),
                    GraphScoreFamily::GaussianBic,
                    &mut workspace,
                    &ExecutionContext::for_tests(seed),
                )
                .unwrap();
            let structure_error = max_error(&named_marginals(&structure_post, &names), &target);
            assert!(
                structure_error <= band,
                "{} seed={seed}: structure MCMC error {structure_error} > {band}",
                case["name"]
            );

            let mut workspace = DiscoveryWorkspace::default();
            let order = OrderMcmc::new()
                .with_schedule(chains, warmup, draws, thin)
                .with_diagnostics_gate(false);
            let order_post = order
                .run(
                    &data,
                    &variables,
                    &GraphPrior::uniform(),
                    GraphScoreFamily::GaussianBic,
                    &mut workspace,
                    &ExecutionContext::for_tests(seed),
                )
                .unwrap();
            let order_error = max_error(&named_marginals(&order_post, &names), &target);
            assert!(
                order_error <= band,
                "{} seed={seed}: order MCMC error {order_error} > {band}",
                case["name"]
            );
        }
    }
}

#[test]
fn order_mcmc_named_marginals_invariant_to_column_relabeling() {
    let fixture = fixture();
    let (chains, warmup, draws, thin) = schedule(&fixture);
    let band = fixture["acceptance"]["absolute_edge_marginal_band"].as_f64().unwrap();
    let case = &fixture["cases"][0];
    let target = expected(&case["targets"]["uniform_all_dags"]);
    let perm = [2usize, 0, 1];
    let (data, variables, names) = case_data_permuted(case, &perm);
    let mut workspace = DiscoveryWorkspace::default();
    let order =
        OrderMcmc::new().with_schedule(chains, warmup, draws, thin).with_diagnostics_gate(false);
    let post = order
        .run(
            &data,
            &variables,
            &GraphPrior::uniform(),
            GraphScoreFamily::GaussianBic,
            &mut workspace,
            &ExecutionContext::for_tests(401),
        )
        .unwrap();
    let error = max_error(&named_marginals(&post, &names), &target);
    assert!(error <= band, "relabeled order MCMC error {error} > {band} (names={names:?})");
}

#[test]
fn ci_screened_structure_mcmc_against_exact_restricted_targets() {
    let fixture = fixture();
    let (chains, warmup, draws, thin) = schedule(&fixture);
    let band = fixture["acceptance"]["absolute_edge_marginal_band"].as_f64().unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let (data, variables, names) = case_data(case);
        let target = expected(&case["targets"]["uniform_screened_pairs"]);
        let mcmc = StructureMcmc::new()
            .with_schedule(chains, warmup, draws, thin)
            .with_diagnostics_gate(false);
        let mut algorithm = CiScreenedPosterior::new()
            .with_constraints(DiscoveryConstraints {
                alpha: 0.01,
                max_cond_size: 3,
                ..Default::default()
            })
            .with_mcmc(mcmc);
        algorithm.fdr = None;
        let mut workspace = DiscoveryWorkspace::default();
        let post = algorithm
            .run(
                &data,
                &variables,
                &GraphPrior::uniform(),
                GraphScoreFamily::GaussianBic,
                &mut workspace,
                &ExecutionContext::for_tests(499),
            )
            .unwrap();
        let error = max_error(&named_marginals(&post, &names), &target);
        assert!(error <= band, "{} screened max error={error} > {band}", case["name"]);
    }
}

#[test]
fn graph_mcmc_fixture_records_exact_target_and_no_generator() {
    let fixture = fixture();
    assert_eq!(fixture["oracle"]["packages"]["numpy"]["version"], "2.1.3");
    assert_eq!(
        fixture["oracle"]["generation_location"],
        "temporary external harness; not retained"
    );
    assert_eq!(fixture["cases"][0]["targets"]["uniform_all_dags"]["n_dags"], 25);
    assert_eq!(
        fixture["oracle"]["packages"]["numpy"]["metadata_sha256"].as_str().unwrap().len(),
        64
    );
}
