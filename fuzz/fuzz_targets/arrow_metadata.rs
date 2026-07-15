//! Fuzz Arrow RecordBatch → tabular load (malformed metadata / columns).
#![no_main]

use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use causal_data::tabular_from_record_batch;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    let n = (usize::from(data[0]) % 32).max(1);
    let use_bad_type = data.get(1).copied().unwrap_or(0) & 1 == 1;
    let values: Vec<f64> = (0..n)
        .map(|i| f64::from(data.get(2 + i % data.len().saturating_sub(2).max(1)).copied().unwrap_or(0)))
        .collect();

    let (schema, arrays): (Schema, Vec<ArrayRef>) = if use_bad_type {
        // Non-float64 column should be rejected without panic.
        let schema = Schema::new(vec![Field::new("x", DataType::Int32, true)]);
        let arr: ArrayRef = Arc::new(Int32Array::from(
            values.iter().map(|v| *v as i32).collect::<Vec<_>>(),
        ));
        (schema, vec![arr])
    } else {
        let mut fields = vec![Field::new("x", DataType::Float64, true)];
        let mut arrays: Vec<ArrayRef> = vec![Arc::new(Float64Array::from(values.clone()))];
        if data.get(2).copied().unwrap_or(0) & 1 == 1 {
            // Second float column with optional null bitmap noise via values.
            fields.push(Field::new("y", DataType::Float64, true));
            let y: Vec<Option<f64>> = values
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    if data.get(3 + i % 7).copied().unwrap_or(0) & 1 == 1 {
                        None
                    } else {
                        Some(*v)
                    }
                })
                .collect();
            arrays.push(Arc::new(Float64Array::from(y)));
        }
        // Attach opaque metadata keys from fuzzer bytes.
        let mut metadata = std::collections::HashMap::new();
        if let Ok(k) = std::str::from_utf8(&data[data.len().saturating_sub(8)..]) {
            metadata.insert("fuzz_meta".into(), k.to_string());
        }
        let schema = Schema::new_with_metadata(fields, metadata);
        (schema, arrays)
    };

    if let Ok(batch) = RecordBatch::try_new(Arc::new(schema), arrays) {
        let _ = tabular_from_record_batch(&batch);
    }
});
