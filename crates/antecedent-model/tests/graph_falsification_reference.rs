//! Graph permutation falsification against an exact reference.
//!
//! `oracle_data/graph_falsification_reference.py` (frozen in the JSON beside it, partial
//! correlations and Student-t tail probabilities at 60 digits with mpmath) enumerates the local
//! Markov implications of twelve designs by their definition, tests each with the residual
//! partial correlation, and counts violations for the graph and for every non-identity node
//! relabelling, the add-one p-value, the Markov-equivalent relabellings (equal skeleton and
//! unshielded colliders) and the verdict. The designs include data shifted by 1e8 and scaled by
//! 1e-6, a duplicated column to within 1e-4, seven rows with three residual degrees of freedom,
//! a p-value that sits exactly on the significance level, a complete graph (nothing testable)
//! and chains whose relabellings are largely equivalent. This test requires the product to
//! reproduce each count, the p-value, the statistic and p-value of every implication, and the
//! verdict; and requires a declared equivalence-share threshold to be applied rather than
//! ignored.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
    VariableId,
};
use antecedent_data::column::{Float64Column, ValidityBitmap};
use antecedent_data::{OwnedColumn, OwnedColumnarStorage, TabularData};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_model::falsify::{
    GraphFalsificationConfig, GraphFalsificationOutcome, GraphUninformativeReason, falsify_graph,
};
use serde_json::Value;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn table(columns: &[Vec<f64>]) -> TabularData {
    let mut b = CausalSchemaBuilder::new();
    for i in 0..columns.len() {
        b.add_variable(
            format!("v{i}"),
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let validity = ValidityBitmap::all_valid(columns[0].len());
    let cols = columns
        .iter()
        .enumerate()
        .map(|(i, v)| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(u32::try_from(i).unwrap()),
                    Arc::from(v.clone()),
                    validity.clone(),
                )
                .unwrap(),
            )
        })
        .collect();
    TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap())
}

#[test]
fn counts_p_values_equivalence_and_verdicts_match_the_definition() {
    let designs: Value =
        serde_json::from_str(include_str!("oracle_data/graph_falsification_reference.json"))
            .unwrap();
    let ctx = ExecutionContext::for_tests(3);
    let mut verdicts = std::collections::BTreeSet::new();
    for d in designs.as_array().unwrap() {
        let name = d["name"].as_str().unwrap();
        let n = d["n"].as_u64().unwrap() as u32;
        let mut dag = Dag::with_variables(n);
        for e in d["edges"].as_array().unwrap() {
            dag.insert_directed(
                DenseNodeId::from_raw(e[0].as_u64().unwrap() as u32),
                DenseNodeId::from_raw(e[1].as_u64().unwrap() as u32),
            )
            .unwrap();
        }
        let columns: Vec<Vec<f64>> = d["columns"].as_array().unwrap().iter().map(floats).collect();
        let data = table(&columns);
        let config = GraphFalsificationConfig {
            max_equivalent_share: d["max_equivalent_share"].as_f64(),
            seed: Some(17),
            ..GraphFalsificationConfig::default()
        };
        let report = falsify_graph(&dag, &data, &config, &ctx).unwrap();
        assert_eq!(report.n_implications as u64, d["n_implications"].as_u64().unwrap(), "{name}");
        let outcome = d["outcome"].as_str().unwrap();
        verdicts.insert(outcome.to_owned());
        match outcome {
            "uninformative" => {
                assert_eq!(report.outcome, GraphFalsificationOutcome::Uninformative, "{name}");
                let reason = match d["reason"].as_str().unwrap() {
                    "no_testable_implication" => GraphUninformativeReason::NoTestableImplication,
                    _ => GraphUninformativeReason::EquivalenceClassShare,
                };
                assert_eq!(report.uninformative_reason, Some(reason), "{name}");
                if let Some(share) = d["share"].as_f64() {
                    assert!((report.equivalent_share - share).abs() < 1e-15, "{name}");
                }
            }
            _ => {
                let want = if outcome == "rejected" {
                    GraphFalsificationOutcome::Rejected
                } else {
                    GraphFalsificationOutcome::NotRejected
                };
                assert_eq!(report.outcome, want, "{name}");
                assert!(report.exhaustive, "{name}");
                assert_eq!(
                    report.permutations_completed as u64,
                    d["permutations"].as_u64().unwrap(),
                    "{name}"
                );
                assert_eq!(
                    report.equivalent_permutations as u64,
                    d["equivalent"].as_u64().unwrap(),
                    "{name}"
                );
                assert_eq!(
                    report.n_violations.unwrap() as u64,
                    d["observed_violations"].as_u64().unwrap(),
                    "{name}"
                );
                let mut got: Vec<u32> = report.permutation_violations.to_vec();
                got.sort_unstable();
                let want_counts: Vec<u32> = d["permutation_violations"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|c| c.as_u64().unwrap() as u32)
                    .collect();
                assert_eq!(got, want_counts, "{name}");
                assert_eq!(report.p_value.unwrap(), d["p_value"].as_f64().unwrap(), "{name}");
                for (ev, want) in report.evidence.iter().zip(d["evidence"].as_array().unwrap()) {
                    assert_eq!(ev.node.raw() as u64, want["node"].as_u64().unwrap(), "{name}");
                    assert_eq!(ev.other.raw() as u64, want["other"].as_u64().unwrap(), "{name}");
                    let z: Vec<u64> = ev.conditioning.iter().map(|c| u64::from(c.raw())).collect();
                    let wz: Vec<u64> = want["conditioning"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|c| c.as_u64().unwrap())
                        .collect();
                    assert_eq!(z, wz, "{name}");
                    let wp = want["p_value"].as_f64().unwrap();
                    assert!(
                        (ev.p_value - wp).abs() <= 1e-6 * wp + 1e-300,
                        "{name}: p {} vs {wp}",
                        ev.p_value
                    );
                    let ws = want["statistic"].as_f64().unwrap();
                    assert!(
                        (ev.statistic - ws).abs() <= 1e-8,
                        "{name}: r {} vs {ws}",
                        ev.statistic
                    );
                    assert_eq!(ev.violated, wp < 0.05, "{name}");
                }
            }
        }
    }
    assert_eq!(verdicts.len(), 3, "{verdicts:?}");
}
