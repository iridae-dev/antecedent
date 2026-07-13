//! Cross-language Arrow load gate (Phase 0): measured copy behavior.
//!
//! The same float64 columns are loaded via the Arrow adapter. Copy bytes and
//! diagnostics are asserted so Python and Rust share one acceptance contract.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(feature = "arrow")]

use std::sync::Arc;

use arrow_array::{Float64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use causal_data::{TableView, tabular_from_record_batch};

/// Shared fixture values used by the Python gate (`python/tests/test_arrow_copy_gate.py`).
fn fixture_batch() -> RecordBatch {
    let schema = Schema::new(vec![
        Field::new("t", DataType::Float64, false),
        Field::new("y", DataType::Float64, false),
        Field::new("z", DataType::Float64, false),
    ]);
    let t = Float64Array::from(vec![0.0_f64, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
    let y = Float64Array::from(vec![1.0, 3.0, 1.5, 3.5, 2.0, 4.0, 2.5, 4.5]);
    let z = Float64Array::from(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
    RecordBatch::try_new(Arc::new(schema), vec![Arc::new(t), Arc::new(y), Arc::new(z)]).unwrap()
}

#[test]
fn rust_arrow_load_reports_measured_copy() {
    let batch = fixture_batch();
    let loaded = tabular_from_record_batch(&batch).unwrap();
    assert_eq!(loaded.data.row_count(), 8);
    assert_eq!(loaded.data.schema().len(), 3);
    // 8 rows × 3 cols × 8 bytes + validity bytes > 0
    assert!(loaded.bytes_copied > 0);
    assert!(!loaded.diagnostics.is_empty());
    assert!(
        loaded.diagnostics.entries.iter().any(|d| d.code.contains("materialize")),
        "expected materialization diagnostics"
    );
}
