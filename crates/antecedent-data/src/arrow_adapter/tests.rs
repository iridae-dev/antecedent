//! Unit tests.

#![cfg_attr(test, allow(clippy::float_cmp, reason = "values are copied bit-for-bit"))]

use super::chunks::*;

use antecedent_core::VariableId;
use arrow_array::Float64Array;
use arrow_schema::{DataType, Field, Schema};

use super::*;
use crate::arrow_ffi::ArrowCColumn;
use crate::table::TableView;

#[test]
fn arrow_load_copies_and_exposes_table_view() {
    let schema = Schema::new(vec![
        Field::new("x", DataType::Float64, true),
        Field::new("y", DataType::Float64, true),
    ]);
    let x = Float64Array::from(vec![Some(1.0), None, Some(3.0)]);
    let y = Float64Array::from(vec![Some(4.0), Some(5.0), Some(6.0)]);
    let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(x), Arc::new(y)]).unwrap();

    let loaded = tabular_from_record_batch(&batch).unwrap();
    assert!(loaded.bytes_copied > 0);
    assert_eq!(loaded.bytes_borrowed, 0);
    assert!(!loaded.diagnostics.is_empty());
    assert_eq!(loaded.data.row_count(), 3);
    let col = loaded.data.column(VariableId::from_raw(0)).unwrap();
    match col {
        crate::column::ColumnView::Float64(c) => {
            assert!(c.validity.is_valid(0));
            assert!(!c.validity.is_valid(1));
            assert!((c.values[2] - 3.0).abs() < f64::EPSILON);
            assert!(!c.values.is_foreign());
        }
        _ => panic!("expected float"),
    }
}

#[test]
fn arrow_load_fixed_size_list_float64() {
    use arrow_array::FixedSizeListArray;
    use arrow_buffer::NullBuffer;

    let values = Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let list = FixedSizeListArray::new(
        Arc::new(Field::new("item", DataType::Float64, true)),
        2,
        Arc::new(values),
        Some(NullBuffer::from(vec![true, false, true])),
    );
    let schema = Schema::new(vec![Field::new(
        "v",
        DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float64, true)), 2),
        true,
    )]);
    let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(list)]).unwrap();
    let loaded = tabular_from_record_batch(&batch).unwrap();
    assert_eq!(loaded.data.row_count(), 3);
    match loaded.data.column(VariableId::from_raw(0)).unwrap() {
        crate::column::ColumnView::FixedVector(c) => {
            assert_eq!(c.dim, 2);
            assert!(c.validity.is_valid(0));
            assert!(!c.validity.is_valid(1));
            assert!(c.validity.is_valid(2));
            assert!((c.values[0] - 1.0).abs() < f64::EPSILON);
            assert!((c.values[1] - 2.0).abs() < f64::EPSILON);
            assert!((c.values[4] - 5.0).abs() < f64::EPSILON);
        }
        _ => panic!("expected FixedVector"),
    }
}

fn float_col(data: &TabularData, id: u32) -> &Float64Column {
    match data.column(VariableId::from_raw(id)).unwrap() {
        crate::column::ColumnView::Float64(c) => c,
        _ => panic!("expected float"),
    }
}

#[test]
fn arrow_null_float64_cell_reads_nan_on_every_path() {
    let schema = Schema::new(vec![Field::new("x", DataType::Float64, true)]);
    let x = Float64Array::from(vec![Some(1.0), None, Some(3.0)]);
    let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(x)]).unwrap();
    let loaded = tabular_from_record_batch(&batch).unwrap();
    let id = VariableId::from_raw(0);
    assert!(!float_col(&loaded.data, 0).validity.is_valid(1));
    let slice = loaded.data.float64_slice(id).unwrap();
    assert!(slice[1].is_nan(), "borrowed slice exposed {} for a null cell", slice[1]);
    assert!(loaded.data.float64_cow(id).unwrap()[1].is_nan());
    let values = loaded.data.float64_values(id).unwrap();
    assert!(values[1].is_nan());
    assert_eq!((values[0], values[2]), (1.0, 3.0));
}

#[test]
fn arrow_cdi_nulls_copy_with_nan_when_slots_hold_placeholders() {
    // arrow-rs writes 0.0 under null slots, like most exporters.
    let x = Float64Array::from(vec![Some(1.0), None, Some(3.0)]);
    assert_eq!(x.values()[1], 0.0);
    let loaded =
        tabular_from_arrow_c_columns(vec![ArrowCColumn::from_arrow_array("x", &x).unwrap()])
            .unwrap();
    assert_eq!(loaded.bytes_borrowed, 0);
    let c = float_col(&loaded.data, 0);
    assert!(!c.values.is_foreign());
    assert!(!c.validity.is_valid(1));
    assert!(c.values[1].is_nan());
    assert!(loaded.data.float64_slice(VariableId::from_raw(0)).unwrap()[1].is_nan());
    assert!((c.values[2] - 3.0).abs() < f64::EPSILON);
}

#[test]
fn arrow_cdi_nulls_stay_zero_copy_when_slots_hold_nan() {
    use arrow_buffer::{NullBuffer, ScalarBuffer};
    let x = Float64Array::new(
        ScalarBuffer::from(vec![1.0, f64::NAN, 3.0]),
        Some(NullBuffer::from(vec![true, false, true])),
    );
    let loaded =
        tabular_from_arrow_c_columns(vec![ArrowCColumn::from_arrow_array("x", &x).unwrap()])
            .unwrap();
    assert!(loaded.bytes_borrowed > 0);
    let c = float_col(&loaded.data, 0);
    assert!(c.values.is_foreign());
    assert!(!c.validity.is_valid(1));
    assert!(c.values[1].is_nan());
}

#[test]
fn arrow_fixed_size_list_null_component_invalidates_row() {
    use arrow_array::FixedSizeListArray;
    use arrow_buffer::NullBuffer;

    let values =
        Float64Array::from(vec![Some(1.0), Some(2.0), Some(3.0), None, Some(5.0), Some(6.0)]);
    let item = Arc::new(Field::new("item", DataType::Float64, true));
    let list = FixedSizeListArray::new(
        Arc::clone(&item),
        2,
        Arc::new(values),
        Some(NullBuffer::from(vec![true, true, false])),
    );
    let schema = Schema::new(vec![Field::new("v", DataType::FixedSizeList(item, 2), true)]);
    let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(list)]).unwrap();
    let loaded = tabular_from_record_batch(&batch).unwrap();
    match loaded.data.column(VariableId::from_raw(0)).unwrap() {
        crate::column::ColumnView::FixedVector(c) => {
            assert!(c.validity.is_valid(0));
            assert_eq!(&c.values[0..2], &[1.0, 2.0]);
            // Row 1 has a null component; row 2 is a null list slot.
            assert!(!c.validity.is_valid(1));
            assert!(!c.validity.is_valid(2));
            assert!(c.values[2..].iter().all(|v| v.is_nan()));
        }
        _ => panic!("expected FixedVector"),
    }
}

#[test]
fn sliced_fixed_size_list_reads_the_logical_window() {
    use arrow_array::FixedSizeListArray;
    use arrow_buffer::NullBuffer;

    let values = Float64Array::from(vec![
        Some(9.0),
        Some(8.0),
        Some(1.0),
        Some(2.0),
        Some(3.0),
        None,
        Some(5.0),
        Some(6.0),
    ]);
    let item = Arc::new(Field::new("item", DataType::Float64, true));
    let list = FixedSizeListArray::new(
        Arc::clone(&item),
        2,
        Arc::new(values),
        Some(NullBuffer::from(vec![true, true, true, false])),
    );
    let window = list.slice(1, 2);
    let schema = Schema::new(vec![Field::new("v", DataType::FixedSizeList(item, 2), true)]);
    let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(window)]).unwrap();
    let loaded = tabular_from_record_batch(&batch).unwrap();
    match loaded.data.column(VariableId::from_raw(0)).unwrap() {
        crate::column::ColumnView::FixedVector(c) => {
            assert_eq!(c.dim, 2);
            assert_eq!(c.len(), 2);
            assert!(c.validity.is_valid(0));
            assert_eq!(&c.values[0..2], &[1.0, 2.0]);
            assert!(!c.validity.is_valid(1));
            assert!(c.values[2].is_nan() && c.values[3].is_nan());
            assert!(c.values.iter().all(|v| *v != 9.0 && *v != 8.0));
        }
        _ => panic!("expected FixedVector"),
    }
}

#[test]
fn arrow_cdi_zero_copy_borrows_values() {
    let x = Float64Array::from(vec![1.0, 2.0, 3.0]);
    let y = Float64Array::from(vec![4.0, 5.0, 6.0]);
    let loaded = tabular_from_arrow_c_columns(vec![
        ArrowCColumn::from_arrow_array("x", &x).unwrap(),
        ArrowCColumn::from_arrow_array("y", &y).unwrap(),
    ])
    .unwrap();
    assert!(loaded.bytes_borrowed > 0);
    assert_eq!(loaded.data.row_count(), 3);
    let col = loaded.data.column(VariableId::from_raw(0)).unwrap();
    match col {
        crate::column::ColumnView::Float64(c) => {
            assert!(c.values.is_foreign());
            assert!((c.values[0] - 1.0).abs() < f64::EPSILON);
            assert!((c.values[2] - 3.0).abs() < f64::EPSILON);
        }
        _ => panic!("expected float"),
    }
}

#[test]
fn row_total_is_not_capped_at_u32_max() {
    let max_u32 = usize::try_from(u32::MAX).expect("row total");
    let wide = max_u32 + 2;
    assert_eq!(checked_row_total([wide]).unwrap(), wide);
    assert_eq!(checked_row_total([max_u32, 2]).unwrap(), wide);
    assert!(checked_row_total([usize::MAX, 1]).is_err());
}

#[test]
fn stream_keeps_chunks_until_contiguous() {
    use arrow_array::{Int64Array, RecordBatchIterator};
    use arrow_schema::ArrowError;

    let big = (1_i64 << 53) + 3;
    let schema = Arc::new(Schema::new(vec![Field::new("unit_id", DataType::Int64, true)]));
    let first = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from(vec![Some(1_i64), Some(big)]))],
    )
    .unwrap();
    let second = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from(vec![Some(4_i64)]))],
    )
    .unwrap();
    let empty = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from(Vec::<i64>::new()))],
    )
    .unwrap();
    let batches: Vec<Result<RecordBatch, ArrowError>> = vec![Ok(first), Ok(empty), Ok(second)];
    let reader = RecordBatchIterator::new(batches.into_iter(), Arc::clone(&schema));
    let frame =
        ArrowFrameView::from_stream(crate::arrow_ffi::ArrowCStream::from_reader(Box::new(reader)))
            .unwrap();
    assert_eq!(frame.n_chunks(), 3);
    let chunks = frame.column_chunks(0).unwrap();
    match &chunks[0] {
        crate::column::OwnedColumn::Int64(column) => {
            assert_eq!(column.values.as_ref(), &[1, big]);
            assert_eq!(column.len(), 2);
        }
        _ => panic!("expected int64 id"),
    }
    assert_eq!(chunks[1].len(), 0);
    let loaded = frame.contiguous().unwrap();
    assert_eq!(loaded.data.row_count(), 3);
    match loaded.data.column(VariableId::from_raw(0)).unwrap() {
        crate::column::ColumnView::Int64(column) => {
            assert_eq!(column.values.as_ref(), &[1, big, 4]);
        }
        _ => panic!("expected int64 id"),
    }
    let variable = &loaded.data.schema().variables()[0];
    assert!(variable.role_hints.contains(antecedent_core::RoleHint::UnitId));
}

#[test]
fn sliced_offset_reads_the_logical_window() {
    use arrow_array::{Int64Array, TimestampMicrosecondArray};

    let ids = Int64Array::from(vec![Some(9_i64), None, Some(7), Some(8)]);
    let window = ids.slice(1, 3);
    let loaded = tabular_from_arrow_c_columns(vec![
        ArrowCColumn::from_arrow_array("unit_id", &window).unwrap(),
    ])
    .unwrap();
    match loaded.data.column(VariableId::from_raw(0)).unwrap() {
        crate::column::ColumnView::Int64(column) => {
            assert!(!column.validity.is_valid(0));
            assert!(column.validity.is_valid(1));
            assert_eq!(column.values[1], 7);
            assert_eq!(column.values[2], 8);
            assert_ne!(column.values[0], 9);
        }
        _ => panic!("expected int64"),
    }

    let stamps = TimestampMicrosecondArray::from(vec![Some(9_i64), Some(5), None, Some(7)]);
    let window = stamps.slice(1, 3);
    let loaded = tabular_from_arrow_c_columns(vec![
        ArrowCColumn::from_arrow_array("when", &window).unwrap(),
    ])
    .unwrap();
    match loaded.data.column(VariableId::from_raw(0)).unwrap() {
        crate::column::ColumnView::Timestamp(column) => {
            assert_eq!(column.len(), 3);
            assert_eq!(column.values_ns[0], 5_000);
            assert!(!column.validity.is_valid(1));
            assert!(column.validity.is_valid(0));
            assert_eq!(column.values_ns[2], 7_000);
        }
        _ => panic!("expected timestamp"),
    }
    let variable = &loaded.data.schema().variables()[0];
    assert!(variable.role_hints.contains(antecedent_core::RoleHint::Time));
}

#[test]
fn dictionary_categories_keep_codes_and_labels() {
    use arrow_array::types::Int32Type;
    use arrow_array::{DictionaryArray, Int32Array, StringArray};

    let values = Arc::new(StringArray::from(vec!["red", "blue", "green"])) as arrow_array::ArrayRef;
    let keys = Int32Array::from(vec![Some(2), Some(0), None]);
    let dict = DictionaryArray::<Int32Type>::new(keys, Arc::clone(&values));
    let sliced = dict.slice(1, 2);
    let loaded =
        tabular_from_arrow_c_columns(vec![ArrowCColumn::from_arrow_array("arm", &sliced).unwrap()])
            .unwrap();
    match loaded.data.column(VariableId::from_raw(0)).unwrap() {
        crate::column::ColumnView::Categorical(column) => {
            assert_eq!(column.codes[0].raw(), 0);
            assert!(!column.validity.is_valid(1));
            assert_eq!(column.domain.levels.len(), 3);
            assert_eq!(column.domain.levels[0].label.as_ref(), "red");
            assert_eq!(column.domain.levels[2].label.as_ref(), "green");
            assert!(!column.domain.ordered);
        }
        _ => panic!("expected categorical"),
    }
}

#[test]
fn stream_error_releases_the_reader() {
    use std::sync::atomic::{AtomicBool, Ordering};

    use arrow_schema::ArrowError;

    struct ReleaseOnDrop(Arc<AtomicBool>);
    impl Drop for ReleaseOnDrop {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }
    struct Flaky {
        schema: Arc<Schema>,
        step: usize,
        _guard: ReleaseOnDrop,
    }
    impl Iterator for Flaky {
        type Item = Result<RecordBatch, ArrowError>;
        fn next(&mut self) -> Option<Self::Item> {
            if self.step == 0 {
                self.step = 1;
                let batch = RecordBatch::try_new(
                    Arc::clone(&self.schema),
                    vec![Arc::new(Float64Array::from(vec![1.0, 2.0]))],
                )
                .unwrap();
                Some(Ok(batch))
            } else {
                Some(Err(ArrowError::ComputeError("batch failed".into())))
            }
        }
    }
    impl arrow_array::RecordBatchReader for Flaky {
        fn schema(&self) -> arrow_schema::SchemaRef {
            Arc::clone(&self.schema)
        }
    }
    let released = Arc::new(AtomicBool::new(false));
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, true)]));
    let reader = Flaky { schema, step: 0, _guard: ReleaseOnDrop(Arc::clone(&released)) };
    let loaded =
        ArrowFrameView::from_stream(crate::arrow_ffi::ArrowCStream::from_reader(Box::new(reader)));
    assert!(loaded.is_err());
    assert!(released.load(Ordering::SeqCst));
}

#[test]
fn second_column_failure_drops_the_first_import() {
    use arrow_array::StringArray;

    let good = Float64Array::from(vec![1.0, 2.0, 3.0]);
    let bad = StringArray::from(vec!["no"]);
    let loaded = tabular_from_arrow_c_columns(vec![
        ArrowCColumn::from_arrow_array("x", &good).unwrap(),
        ArrowCColumn::from_arrow_array("s", &bad).unwrap(),
    ]);
    assert!(loaded.is_err());
}

#[test]
fn categorical_chunks_keep_the_reference_and_the_domain_id() {
    fn chunk(domain_id: u32, reference: u32) -> OwnedColumn {
        let domain = Arc::new(
            CategoryDomain::try_new(
                CategoryDomainId::from_raw(domain_id),
                vec![
                    CategoryLevel { label: Arc::<str>::from("a") },
                    CategoryLevel { label: Arc::<str>::from("b") },
                ],
                false,
                Some(CategoryCode::from_raw(reference)),
                UnknownCategoryPolicy::Fail,
            )
            .unwrap(),
        );
        OwnedColumn::Categorical(
            CategoricalColumn::try_new(
                VariableId::from_raw(1),
                vec![CategoryCode::from_raw(0)],
                ValidityBitmap::all_valid(1),
                domain,
            )
            .unwrap(),
        )
    }

    let matched = concat_chunks(&[chunk(7, 0), chunk(7, 0)]).unwrap();
    let OwnedColumn::Categorical(column) = matched else {
        panic!("categorical");
    };
    assert_eq!(column.domain.reference, Some(CategoryCode::from_raw(0)));
    assert!(concat_chunks(&[chunk(7, 0), chunk(7, 1)]).is_err());
    assert!(concat_chunks(&[chunk(7, 0), chunk(8, 0)]).is_err());
}

fn two_chunk_frame() -> ArrowFrameView {
    use arrow_array::RecordBatch;
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, true)]));
    let first = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Float64Array::from(vec![1.0, 2.0]))],
    )
    .unwrap();
    let second =
        RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Float64Array::from(vec![3.0]))])
            .unwrap();
    ArrowFrameView::from_record_batches(vec![first, second]).unwrap()
}

#[test]
fn an_estimator_that_cannot_stream_materializes_or_refuses() {
    use crate::materialize::{EstimatorStreamNeed, PreparationBudget, PreparationControls};
    use crate::sample_policy::MissingPolicy;

    let frame = two_chunk_frame();
    let mut controls = PreparationControls::finite_resident();
    controls.budget = Some(PreparationBudget::new(0));
    let refused = frame.open_for_estimator(&[], MissingPolicy::CompleteCase, &[], controls);
    let error = refused.expect_err("budget");
    assert!(
        matches!(error, crate::error::DataError::BudgetRefused { budget: Some(0), .. }),
        "{error:?}"
    );

    controls.budget = None;
    let delivered =
        frame.open_for_estimator(&[], MissingPolicy::CompleteCase, &[], controls).unwrap();
    let EstimatorFrame::Finite { table, product, diagnostics } = delivered else {
        panic!("cannot-stream delivery must be a finite table");
    };
    assert_eq!(table.row_count(), 3);
    assert_eq!(product.receipt.proved_rows(), Some(3));
    assert!(!product.receipt.estimator_sees_unbounded_stream());
    assert!(product.receipt.materialized_from_scan());
    // No budget was given, so the copy is recorded as unmetered rather than admitted.
    assert!(!product.receipt.is_bounded_end_to_end_route());
    assert!(
        diagnostics.entries.iter().any(|entry| { entry.code.contains("materialize.estimator") })
    );

    controls.stream = EstimatorStreamNeed::CanStream;
    let scan = frame.open_for_estimator(&[], MissingPolicy::CompleteCase, &[], controls).unwrap();
    let EstimatorFrame::Scan(scan) = scan else {
        panic!("a streaming estimator keeps the chunks");
    };
    assert!(scan.receipt().estimator_sees_unbounded_stream());
    assert!(!scan.receipt().is_bounded_end_to_end_route());
    assert!(!scan.receipt().missing().unwrap().population_bound);
}
