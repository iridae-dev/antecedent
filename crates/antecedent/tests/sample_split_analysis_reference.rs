//! The row split, weight carrying, Kish size, weighted HC3 effect and Student interval of the
//! sample-split analysis against exact references.
//!
//! `oracle_data/sample_split_analysis_reference.py` (frozen in the JSON beside it) filters a
//! 70-row table by the analysis mask and by completeness of the searched columns, reproduces the
//! seeded permutation from splitmix64 and Fisher-Yates, gathers each part's weights, computes the
//! Kish size of each part exactly, solves the weighted least-squares effect with its HC3 sandwich
//! over the rationals on the analysis rows of positive weight, and evaluates the Student
//! interval at the Kish size less the coefficients (the row count less the coefficients when
//! unweighted) at 60 digits. The product must reproduce all of it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::float_cmp, reason = "carried weights are compared bit for bit")]

use std::sync::Arc;

use antecedent::analysis::sample_split_analysis::{
    AnalysisEffectRequest, SplitRule, discover_on_split, split_rows,
};
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::{OwnedColumnarStorage, TableView, TabularData, ValidityBitmap};
use antecedent_discovery::selective::{CiTestSpec, StaticAlgorithm, StaticDiscoveryContract};
use serde_json::Value;

fn v(index: u32) -> VariableId {
    VariableId::from_raw(index)
}

fn doc() -> Value {
    serde_json::from_str(include_str!("oracle_data/sample_split_analysis_reference.json")).unwrap()
}

fn cells(column: &Value) -> Vec<f64> {
    column.as_array().unwrap().iter().map(|c| c.as_f64().unwrap_or(f64::NAN)).collect()
}

fn ids(value: &Value) -> Vec<usize> {
    value.as_array().unwrap().iter().map(|r| r.as_u64().unwrap() as usize).collect()
}

fn rel(got: f64, want: f64) -> f64 {
    (got - want).abs() / want.abs().max(1e-300)
}

fn weighted_masked_table(doc: &Value) -> TabularData {
    let columns: Vec<Vec<f64>> = doc["columns"].as_array().unwrap().iter().map(cells).collect();
    let names = ["a", "b", "c", "d"];
    let plain =
        TabularData::from_f64_columns(names.iter().copied().zip(columns.iter().map(Vec::as_slice)))
            .unwrap();
    let keep: Vec<bool> =
        doc["mask"].as_array().unwrap().iter().map(|b| b.as_bool().unwrap()).collect();
    let mut bytes = vec![0u8; keep.len().div_ceil(8)];
    for (i, kept) in keep.iter().enumerate() {
        if *kept {
            bytes[i / 8] |= 1 << (i % 8);
        }
    }
    let weights: Vec<f64> =
        doc["weights"].as_array().unwrap().iter().map(|w| w.as_f64().unwrap()).collect();
    let storage = plain.storage();
    TabularData::new(
        OwnedColumnarStorage::try_new(
            storage.schema().clone(),
            storage.columns().to_vec(),
            Some(ValidityBitmap::from_bytes(bytes, keep.len()).unwrap()),
            Some(Arc::from(weights)),
        )
        .unwrap(),
    )
}

#[test]
fn the_split_filters_assigns_and_carries_weights_as_the_definition_says() {
    let doc = doc();
    let data = weighted_masked_table(&doc);
    let searched: Vec<VariableId> = (0..3).map(v).collect();
    let split_ref = &doc["split"];
    let weights: Vec<f64> =
        doc["weights"].as_array().unwrap().iter().map(|w| w.as_f64().unwrap()).collect();
    let seed = split_ref["seed"].as_u64().unwrap();
    let fraction = split_ref["fraction"].as_f64().unwrap();
    for (name, rule) in [
        ("seeded", SplitRule::SeededPermutation { seed, discovery_fraction: fraction }),
        (
            "explicit",
            SplitRule::Explicit { discovery_rows: ids(&split_ref["explicit"]["discovery"]) },
        ),
    ] {
        let want = &split_ref[name];
        let split = split_rows(&data, &searched, rule).unwrap();
        assert_eq!(split.receipt.masked_out_rows, ids(&split_ref["masked_out"]), "{name}");
        assert_eq!(split.receipt.incomplete_rows, ids(&split_ref["incomplete"]), "{name}");
        assert_eq!(split.discovery_row_ids, ids(&want["discovery"]), "{name}: discovery rows");
        assert_eq!(split.analysis_row_ids, ids(&want["analysis"]), "{name}: analysis rows");
        let carried: Vec<f64> = split.analysis_row_ids.iter().map(|&r| weights[r]).collect();
        assert_eq!(
            split.analysis_weights().unwrap(),
            carried.as_slice(),
            "{name}: weights travel with rows"
        );
        let sw = split.receipt.weights.as_ref().unwrap();
        assert!(
            rel(sw.discovery_effective_sample_size, want["discovery_ess"].as_f64().unwrap())
                < 1e-12,
            "{name}"
        );
        assert!(
            rel(sw.analysis_effective_sample_size, want["analysis_ess"].as_f64().unwrap()) < 1e-12,
            "{name}"
        );
        // The unsearched fourth column travels with its rows, missing cell intact.
        let d_whole = cells(&doc["columns"][3]);
        let d_part = split.analysis.float64_values(v(3)).unwrap();
        for (value, &row) in d_part.iter().zip(&split.analysis_row_ids) {
            assert_eq!(
                value.to_bits(),
                d_whole[row].to_bits(),
                "{name}: unsearched column row {row}"
            );
        }
    }
}

fn contract(test: &str) -> StaticDiscoveryContract {
    let mut contract = StaticDiscoveryContract::new(
        StaticAlgorithm::Pc,
        (0..3).map(v).collect(),
        CiTestSpec::named(test),
        9,
    );
    contract.fdr = None;
    contract.constraints.alpha = 0.01;
    contract.constraints.max_cond_size = 2;
    contract
}

fn request() -> AnalysisEffectRequest {
    AnalysisEffectRequest {
        treatment: v(0),
        outcome: v(2),
        adjustment: vec![v(1)],
        levels: (0.0, 1.0),
        clusters: None,
    }
}

fn check_interval(
    effect: &antecedent::analysis::sample_split_analysis::SplitAnalysisEffect,
    intervals: &Value,
) {
    for interval in intervals.as_array().unwrap() {
        let got = effect.interval(interval["level"].as_f64().unwrap()).unwrap();
        assert!((got.degrees_of_freedom - interval["df"].as_f64().unwrap()).abs() < 1e-9);
        assert!(
            rel(got.lower, interval["lower"].as_f64().unwrap()) < 1e-7,
            "{interval}: {}",
            got.lower
        );
        assert!(
            rel(got.upper, interval["upper"].as_f64().unwrap()) < 1e-7,
            "{interval}: {}",
            got.upper
        );
    }
}

#[test]
fn the_weighted_hc3_effect_and_its_effective_size_interval_equal_the_exact_reference() {
    let doc = doc();
    let data = weighted_masked_table(&doc);
    let ctx = ExecutionContext::for_tests(9);
    let split = discover_on_split(
        &contract("weighted_parcorr"),
        &data,
        SplitRule::SeededPermutation { seed: 9, discovery_fraction: 0.5 },
        &ctx,
    )
    .unwrap();
    assert_eq!(split.analysis_row_ids(), ids(&doc["split"]["seeded"]["analysis"]).as_slice());
    let effect = split.estimate_analysis_effect(&request(), &ctx).unwrap();
    let want = &doc["weighted"];
    assert!(rel(effect.effect, want["point"].as_f64().unwrap()) < 1e-9, "point {}", effect.effect);
    assert!(
        rel(effect.standard_error, want["se"].as_f64().unwrap()) < 1e-9,
        "se {}",
        effect.standard_error
    );
    let record = effect.sampling_weights.expect("a weighted source reports its weights");
    assert_eq!(record.rows_used, want["rows_used"].as_u64().unwrap());
    assert_eq!(record.zero_weight_rows, want["zero_weight_rows"].as_u64().unwrap());
    assert!(rel(record.effective_sample_size, want["ess"].as_f64().unwrap()) < 1e-12);
    assert!(rel(record.weight_sum, want["weight_sum"].as_f64().unwrap()) < 1e-12);
    assert_eq!(effect.coefficients, 3);
    check_interval(&effect, &want["intervals"]);
}

#[test]
fn the_unweighted_hc3_effect_and_its_row_count_interval_equal_the_exact_reference() {
    let doc = doc();
    let want = &doc["unweighted"];
    let columns: Vec<Vec<f64>> = want["columns"].as_array().unwrap().iter().map(cells).collect();
    let data = TabularData::from_f64_columns(
        ["a", "b", "c"].into_iter().zip(columns.iter().map(Vec::as_slice)),
    )
    .unwrap();
    let ctx = ExecutionContext::for_tests(9);
    let split = discover_on_split(
        &contract("parcorr"),
        &data,
        SplitRule::SeededPermutation { seed: 9, discovery_fraction: 0.5 },
        &ctx,
    )
    .unwrap();
    assert_eq!(split.analysis_row_ids(), ids(&want["analysis_ids"]).as_slice());
    let effect = split.estimate_analysis_effect(&request(), &ctx).unwrap();
    assert!(effect.sampling_weights.is_none());
    assert_eq!(effect.rows, want["rows"].as_u64().unwrap());
    assert!(rel(effect.effect, want["point"].as_f64().unwrap()) < 1e-9, "point {}", effect.effect);
    assert!(
        rel(effect.standard_error, want["se"].as_f64().unwrap()) < 1e-9,
        "se {}",
        effect.standard_error
    );
    check_interval(&effect, &want["intervals"]);
}
