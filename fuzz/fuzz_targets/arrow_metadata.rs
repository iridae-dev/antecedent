//! Fuzz Arrow RecordBatch load, C Data Interface lifetimes, and release on failure.
#![no_main]

use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use arrow_array::{ArrayRef, Float64Array, Int32Array, RecordBatch, StringArray};
use arrow_schema::{ArrowError, DataType, Field, Schema};
use antecedent_core::VariableId;
use antecedent_data::arrow_ffi::ArrowCStream;
use antecedent_data::{
    ArrowCColumn, ArrowFrameView, ColumnView, TableView, tabular_from_arrow_c_columns,
    tabular_from_record_batch,
};
use libfuzzer_sys::fuzz_target;

struct ReleaseFlag {
    flag: Arc<AtomicBool>,
    _data: Box<[f64]>,
}

impl Drop for ReleaseFlag {
    fn drop(&mut self) {
        self.flag.store(true, Ordering::SeqCst);
    }
}

fn foreign_array(flag: &Arc<AtomicBool>, mut data: Box<[f64]>) -> Float64Array {
    let len = data.len();
    let ptr = NonNull::new(data.as_mut_ptr().cast::<u8>()).expect("allocation pointer");
    let nbytes = std::mem::size_of_val(data.as_ref());
    let owner = Arc::new(ReleaseFlag { flag: Arc::clone(flag), _data: data });
    // SAFETY: `ptr` addresses `owner`'s heap buffer for `nbytes` bytes. `owner`
    // keeps that buffer alive and is the allocation the buffer releases.
    let buffer = unsafe { arrow_buffer::Buffer::from_custom_allocation(ptr, nbytes, owner) };
    Float64Array::new(arrow_buffer::ScalarBuffer::new(buffer, 0, len), None)
}

/// The exported column keeps the exporter alive after the source array is dropped.
fn column_outlives_the_source_array() {
    let flag = Arc::new(AtomicBool::new(false));
    let array = foreign_array(&flag, vec![1.0, 2.0, 3.0].into_boxed_slice());
    let column = ArrowCColumn::from_arrow_array("x", &array).expect("export");
    drop(array);
    assert!(!flag.load(Ordering::SeqCst), "source array dropped the exporter");
    let loaded = tabular_from_arrow_c_columns(vec![column]).expect("import");
    assert!(!flag.load(Ordering::SeqCst), "import dropped the exporter");
    let values = loaded.data.float64_slice(VariableId::from_raw(0)).expect("column");
    assert_eq!(values, &[1.0, 2.0, 3.0]);
    drop(loaded);
    assert!(flag.load(Ordering::SeqCst), "dropping the table did not release the exporter");
}

/// A later column that cannot be imported releases the column that already was.
fn failed_import_releases_the_first_column() {
    let flag = Arc::new(AtomicBool::new(false));
    let array = foreign_array(&flag, vec![1.0, 2.0].into_boxed_slice());
    let good = ArrowCColumn::from_arrow_array("x", &array).expect("export");
    drop(array);
    assert!(!flag.load(Ordering::SeqCst));
    let bad = ArrowCColumn::from_arrow_array("s", &StringArray::from(vec!["no"])).expect("export");
    let loaded = tabular_from_arrow_c_columns(vec![good, bad]);
    assert!(loaded.is_err());
    assert!(flag.load(Ordering::SeqCst), "failed import kept the first exporter");
}

/// Freeze copies the bytes. A later alias is refused and does not replace them.
fn frozen_study_refuses_a_foreign_alias() {
    let array = Float64Array::from(vec![1.0, 2.0, 3.0]);
    let frame = ArrowFrameView::from_c_columns(vec![
        ArrowCColumn::from_arrow_array("x", &array).expect("export"),
    ])
    .expect("frame");
    let frozen = frame.freeze().expect("freeze");
    drop(frame);
    drop(array);
    let alias =
        ArrowCColumn::from_arrow_array("x", &Float64Array::from(vec![9.0])).expect("alias");
    let refused = frozen.bind_foreign_alias(alias).expect_err("alias");
    assert!(refused.to_string().contains("refuses a mutable foreign alias"));
    let ColumnView::Float64(column) =
        frozen.data().column(VariableId::from_raw(0)).expect("column")
    else {
        panic!("expected a float column");
    };
    assert!(!column.values.is_foreign());
    assert_eq!(column.values.as_slice()[0].to_bits(), 1.0_f64.to_bits());
}

struct StreamRelease(Arc<AtomicBool>);

impl Drop for StreamRelease {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

struct FlakyStream {
    schema: Arc<Schema>,
    step: usize,
    _guard: StreamRelease,
}

impl Iterator for FlakyStream {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.step == 0 {
            self.step = 1;
            let batch = RecordBatch::try_new(
                Arc::clone(&self.schema),
                vec![Arc::new(Float64Array::from(vec![1.0, 2.0]))],
            )
            .expect("batch");
            Some(Ok(batch))
        } else {
            Some(Err(ArrowError::ComputeError("batch failed".into())))
        }
    }
}

impl arrow_array::RecordBatchReader for FlakyStream {
    fn schema(&self) -> arrow_schema::SchemaRef {
        Arc::clone(&self.schema)
    }
}

/// A stream that fails after its first batch still drops the reader.
fn stream_error_releases_the_reader() {
    let released = Arc::new(AtomicBool::new(false));
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, true)]));
    let reader = FlakyStream {
        schema,
        step: 0,
        _guard: StreamRelease(Arc::clone(&released)),
    };
    let loaded = ArrowFrameView::from_stream(ArrowCStream::from_reader(Box::new(reader)));
    assert!(loaded.is_err());
    assert!(released.load(Ordering::SeqCst), "stream error kept the reader");
}

fn same_bits(left: &[f64], right: &[f64]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(a, b)| a.to_bits() == b.to_bits())
}

fuzz_target!(|data: &[u8]| {
    column_outlives_the_source_array();
    failed_import_releases_the_first_column();
    frozen_study_refuses_a_foreign_alias();
    stream_error_releases_the_reader();
    if data.is_empty() {
        return;
    }
    let n = (usize::from(data[0]) % 32).max(1);
    let use_bad_type = data.get(1).copied().unwrap_or(0) & 1 == 1;
    let values: Vec<f64> = (0..n)
        .map(|i| {
            let index = 2 + i % data.len().saturating_sub(2).max(1);
            f64::from(data.get(index).copied().unwrap_or(0))
        })
        .collect();

    let (schema, arrays): (Schema, Vec<ArrayRef>) = if use_bad_type {
        let schema = Schema::new(vec![Field::new("x", DataType::Int32, true)]);
        let arr: ArrayRef =
            Arc::new(Int32Array::from(values.iter().map(|v| *v as i32).collect::<Vec<_>>()));
        (schema, vec![arr])
    } else {
        let mut fields = vec![Field::new("x", DataType::Float64, true)];
        let mut arrays: Vec<ArrayRef> = vec![Arc::new(Float64Array::from(values.clone()))];
        if data.get(2).copied().unwrap_or(0) & 1 == 1 {
            fields.push(Field::new("y", DataType::Float64, true));
            let y: Vec<Option<f64>> = values
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    if data.get(3 + i % 7).copied().unwrap_or(0) & 1 == 1 { None } else { Some(*v) }
                })
                .collect();
            arrays.push(Arc::new(Float64Array::from(y)));
        }
        let mut metadata = std::collections::HashMap::new();
        if let Ok(k) = std::str::from_utf8(&data[data.len().saturating_sub(8)..]) {
            metadata.insert("fuzz_meta".into(), k.to_string());
        }
        let schema = Schema::new_with_metadata(fields, metadata);
        (schema, arrays)
    };

    let Ok(batch) = RecordBatch::try_new(Arc::new(schema), arrays) else {
        return;
    };
    if use_bad_type {
        let _ = tabular_from_record_batch(&batch);
        return;
    }
    let Ok(loaded) = tabular_from_record_batch(&batch) else {
        return;
    };
    let column = Arc::clone(batch.column(0));
    let exported = ArrowCColumn::from_arrow_array("x", column.as_ref()).expect("export");
    drop(batch);
    drop(column);
    let again = tabular_from_arrow_c_columns(vec![exported]).expect("column outlives the batch");
    let left = loaded.data.float64_slice(VariableId::from_raw(0)).expect("loaded");
    let right = again.data.float64_slice(VariableId::from_raw(0)).expect("exported");
    assert!(same_bits(left, right));
});
