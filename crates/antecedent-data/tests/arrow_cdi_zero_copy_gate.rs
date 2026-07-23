//! Arrow C Data Interface zero-copy acceptance gate.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(feature = "arrow")]

use antecedent_data::{ArrowCColumn, TableView, tabular_from_arrow_c_columns};
use arrow_array::ffi::to_ffi;
use arrow_array::{Array, Float64Array};

#[test]
fn rust_arrow_cdi_zero_copy_acceptance() {
    let x = Float64Array::from(vec![1.0_f64, 2.0, 3.0, 4.0]);
    let y = Float64Array::from(vec![10.0_f64, 20.0, 30.0, 40.0]);
    let (x_arr, x_sch) = to_ffi(&x.to_data()).unwrap();
    let (y_arr, y_sch) = to_ffi(&y.to_data()).unwrap();
    let loaded = tabular_from_arrow_c_columns(vec![
        ArrowCColumn { name: "x".into(), array: x_arr, schema: x_sch },
        ArrowCColumn { name: "y".into(), array: y_arr, schema: y_sch },
    ])
    .unwrap();
    assert_eq!(loaded.data.row_count(), 4);
    assert!(loaded.bytes_borrowed > 0, "expected zero-copy borrow of float64 value buffers");
    // Validity bitmaps may be copied; value buffers must not be.
    let value_bytes = (4 * 2 * core::mem::size_of::<f64>()) as u64;
    assert!(loaded.bytes_borrowed >= value_bytes);
    let col = loaded.data.column(antecedent_core::VariableId::from_raw(0)).unwrap();
    match col {
        antecedent_data::ColumnView::Float64(c) => {
            assert!(c.values.is_foreign());
            assert!((c.values[3] - 4.0).abs() < f64::EPSILON);
        }
        _ => panic!("expected float64"),
    }
}
