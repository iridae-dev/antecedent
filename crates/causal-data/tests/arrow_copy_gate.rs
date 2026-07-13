//! Cross-language Arrow load gate (Phase 0): measured copy behavior.
//!
//! Fixture: `conformance/gates/arrow_copy_fixture.json` (shared with Python).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(feature = "arrow")]

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use causal_data::{TableView, tabular_from_record_batch};
use serde_json::Value;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../conformance/gates/arrow_copy_fixture.json")
}

fn fixture_batch() -> RecordBatch {
    let raw = fs::read_to_string(fixture_path()).expect("arrow_copy_fixture.json");
    let v: Value = serde_json::from_str(&raw).expect("parse fixture");
    let names = v["column_names"].as_array().expect("column_names");
    let cols = v["columns"].as_object().expect("columns");
    let fields: Vec<Field> =
        names.iter().map(|n| Field::new(n.as_str().unwrap(), DataType::Float64, false)).collect();
    let arrays: Vec<ArrayRef> = names
        .iter()
        .map(|n| {
            let name = n.as_str().unwrap();
            let values: Vec<f64> =
                cols[name].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
            Arc::new(Float64Array::from(values)) as ArrayRef
        })
        .collect();
    RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap()
}

#[test]
fn rust_arrow_load_reports_measured_copy() {
    let batch = fixture_batch();
    let expected_rows = 8usize;
    let loaded = tabular_from_record_batch(&batch).unwrap();
    assert_eq!(loaded.data.row_count(), expected_rows);
    assert_eq!(loaded.data.schema().len(), 3);
    assert!(loaded.bytes_copied > 0);
    assert!(!loaded.diagnostics.is_empty());
    assert!(
        loaded.diagnostics.entries.iter().any(|d| d.code.contains("materialize")),
        "expected materialization diagnostics"
    );
}
