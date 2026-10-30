//! File-backed table source: Parquet and Arrow IPC files written in the test.
//!
//! Covers schema fidelity (numeric, boolean, timestamp, date, dictionary and
//! string categories), projection by the required-column closure, chunked
//! admission and its refusals, the streaming content identity, checked global
//! row ids past `u32::MAX` without allocating rows, null masks, datasets, the
//! writer, and Arrow C streams.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    reason = "test file: fixtures unwrap, and read-back values are compared exactly"
)]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use antecedent_core::ExecutionContext;
use antecedent_core::execution::AdmissionPool;
use antecedent_data::arrow_ffi::ArrowCStream;
use antecedent_data::column_closure::{ClosureInputs, ColumnRole};
use antecedent_data::file_source::{
    FileColumnKind, FileSourceError, MaterializationQuote, MemoryStage, ReadOptions, RowCursor,
    TableFile, TableFileFormat, TableStream, WriteOptions, chunk_local_index, write_table_file,
};
use antecedent_data::global_index::GlobalId;
use antecedent_data::{ColumnView, TableView, TabularData};
use arrow_array::types::{Int8Type, Int32Type};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Date32Array, DictionaryArray, Float32Array, Float64Array,
    Int8Array, Int32Array, ListArray, RecordBatch, RecordBatchIterator, StringArray,
    TimestampMillisecondArray, UInt8Array, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef, TimeUnit};

static NEXT: AtomicUsize = AtomicUsize::new(0);

/// A fresh directory under the system temporary directory, removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new(label: &str) -> Self {
        let n = NEXT.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir()
            .join(format!("antecedent-file-source-{}-{label}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }

    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn write_parquet(path: &Path, batch: &RecordBatch, rows_per_group: usize) {
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;

    let file = std::fs::File::create(path).unwrap();
    let props = WriterProperties::builder().set_max_row_group_size(rows_per_group).build();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props)).unwrap();
    let mut offset = 0;
    while offset < batch.num_rows() {
        let take = rows_per_group.min(batch.num_rows() - offset);
        writer.write(&batch.slice(offset, take)).unwrap();
        writer.flush().unwrap();
        offset += take;
    }
    writer.close().unwrap();
}

fn write_ipc_file(path: &Path, batch: &RecordBatch, rows_per_batch: usize) {
    let file = std::fs::File::create(path).unwrap();
    let mut writer = arrow_ipc::writer::FileWriter::try_new(file, &batch.schema()).unwrap();
    let mut offset = 0;
    while offset < batch.num_rows() {
        let take = rows_per_batch.min(batch.num_rows() - offset);
        writer.write(&batch.slice(offset, take)).unwrap();
        offset += take;
    }
    writer.finish().unwrap();
}

/// Twelve rows over every readable kind, with nulls in each nullable column.
#[allow(
    clippy::too_many_lines,
    reason = "a linear fixture spelling out one column per readable kind"
)]
fn mixed_batch() -> RecordBatch {
    let levels: ArrayRef = Arc::new(StringArray::from(vec!["low", "mid", "high"]));
    let keys = Int8Array::from(vec![
        Some(0),
        Some(1),
        None,
        Some(0),
        Some(1),
        Some(1),
        Some(0),
        Some(0),
        Some(1),
        None,
        Some(0),
        Some(1),
    ]);
    let grade = DictionaryArray::<Int8Type>::try_new(keys, levels).unwrap();
    let schema = Schema::new(vec![
        Field::new("y", DataType::Float64, true),
        Field::new("x32", DataType::Float32, false),
        Field::new("count", DataType::Int32, true),
        Field::new("small", DataType::UInt8, false),
        Field::new("flag", DataType::Boolean, true),
        Field::new("at", DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into())), true),
        Field::new("day", DataType::Date32, false),
        Field::new(
            "grade",
            DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
            true,
        ),
        Field::new("site", DataType::Utf8, true),
    ]);
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Float64Array::from(vec![
            Some(1.5),
            None,
            Some(f64::NAN),
            Some(-2.0),
            Some(0.0),
            Some(3.25),
            Some(4.0),
            Some(5.0),
            Some(6.0),
            Some(7.0),
            Some(8.0),
            Some(9.0),
        ])),
        Arc::new(Float32Array::from(vec![0.5_f32; 12])),
        Arc::new(Int32Array::from(vec![
            Some(1),
            Some(2),
            None,
            Some(4),
            Some(5),
            Some(6),
            Some(7),
            Some(8),
            Some(9),
            Some(10),
            Some(11),
            Some(-12),
        ])),
        Arc::new(UInt8Array::from(vec![200_u8; 12])),
        Arc::new(BooleanArray::from(vec![
            Some(true),
            Some(false),
            None,
            Some(true),
            Some(true),
            Some(false),
            Some(true),
            Some(false),
            Some(true),
            Some(false),
            Some(true),
            Some(false),
        ])),
        Arc::new(
            TimestampMillisecondArray::from(vec![
                Some(1_000),
                Some(2_000),
                None,
                Some(4_000),
                Some(5_000),
                Some(6_000),
                Some(7_000),
                Some(8_000),
                Some(9_000),
                Some(10_000),
                Some(11_000),
                Some(12_000),
            ])
            .with_timezone("UTC"),
        ),
        Arc::new(Date32Array::from(vec![19_000; 12])),
        Arc::new(grade),
        Arc::new(StringArray::from(vec![
            Some("north"),
            Some("south"),
            Some("north"),
            None,
            Some("east"),
            Some("south"),
            Some("north"),
            Some("east"),
            Some("east"),
            Some("north"),
            Some("south"),
            Some("north"),
        ])),
    ];
    RecordBatch::try_new(Arc::new(schema), columns).unwrap()
}

fn numeric_batch(rows: usize, bump: f64) -> RecordBatch {
    let x: Vec<f64> = (0..rows).map(|i| i as f64 * 0.5).collect();
    let y: Vec<f64> = (0..rows).map(|i| 1.0 + 2.0 * i as f64 + bump).collect();
    let z: Vec<i32> = (0..rows).map(|i| i32::try_from(i % 7).unwrap()).collect();
    let schema = Schema::new(vec![
        Field::new("x", DataType::Float64, false),
        Field::new("y", DataType::Float64, false),
        Field::new("z", DataType::Int32, false),
    ]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Float64Array::from(x)),
            Arc::new(Float64Array::from(y)),
            Arc::new(Int32Array::from(z)),
        ],
    )
    .unwrap()
}

fn column<'a>(data: &'a TabularData, name: &str) -> ColumnView<'a> {
    data.column(data.schema().id_of(name).unwrap()).unwrap()
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

#[test]
fn parquet_round_trip_keeps_numeric_boolean_time_and_category_kinds() {
    let scratch = Scratch::new("fidelity");
    let path = scratch.path("mixed.parquet");
    write_parquet(&path, &mixed_batch(), 5);
    let table = TableFile::open(&path).unwrap();
    assert_eq!(table.format(), TableFileFormat::Parquet);
    assert_eq!(table.declared_rows(), Some(GlobalId::new(12)));
    let kinds: Vec<(&str, FileColumnKind)> =
        table.columns().iter().map(|c| (&*c.name, c.kind.clone())).collect();
    assert_eq!(kinds[0], ("y", FileColumnKind::Float64));
    assert_eq!(kinds[1], ("x32", FileColumnKind::Float64));
    assert_eq!(kinds[2], ("count", FileColumnKind::Int64));
    assert_eq!(kinds[3], ("small", FileColumnKind::Int64));
    assert_eq!(kinds[4], ("flag", FileColumnKind::Boolean));
    assert!(matches!(kinds[5].1, FileColumnKind::Timestamp { .. }));
    assert_eq!(kinds[6], ("day", FileColumnKind::Timestamp { timezone: None }));
    assert_eq!(kinds[7], ("grade", FileColumnKind::Categorical { ordered: false }));
    assert_eq!(kinds[8], ("site", FileColumnKind::Categorical { ordered: false }));

    let plan = table.plan_all(ReadOptions::with_chunk_rows(4)).unwrap();
    let table = table.materialize(&plan, &ctx()).unwrap();
    let data = table.data();
    assert_eq!(data.row_count(), 12);

    let ColumnView::Float64(y) = column(data, "y") else { panic!("y is float64") };
    assert_eq!(y.values.as_slice()[0], 1.5);
    assert!(!y.validity.is_valid(1), "a null float is invalid");
    assert!(y.values.as_slice()[1].is_nan(), "an invalid float holds NaN");
    assert!(!y.validity.is_valid(2), "a NaN float is invalid");
    let ColumnView::Float64(x32) = column(data, "x32") else { panic!("x32 widens") };
    assert_eq!(x32.values.as_slice()[3], 0.5);
    let ColumnView::Int64(count) = column(data, "count") else { panic!("count is int64") };
    assert_eq!(count.values[11], -12);
    assert!(!count.validity.is_valid(2));
    let ColumnView::Int64(small) = column(data, "small") else { panic!("small is int64") };
    assert_eq!(small.values[0], 200);
    let ColumnView::Boolean(flag) = column(data, "flag") else { panic!("flag is boolean") };
    assert_eq!(flag.values[0], 1);
    assert!(!flag.validity.is_valid(2));
    let ColumnView::Timestamp(at) = column(data, "at") else { panic!("at is a timestamp") };
    assert_eq!(at.values_ns[0], 1_000_000_000);
    assert!(!at.validity.is_valid(2));
    let ColumnView::Timestamp(day) = column(data, "day") else { panic!("day is a timestamp") };
    assert_eq!(day.values_ns[0], 19_000 * 86_400_000_000_000);
    let ColumnView::Categorical(grade) = column(data, "grade") else { panic!("grade") };
    let label = |code: u32| grade.domain.levels[code as usize].label.to_string();
    assert_eq!(label(grade.codes[0].raw()), "low");
    assert_eq!(label(grade.codes[1].raw()), "mid");
    assert!(!grade.validity.is_valid(2));
    let ColumnView::Categorical(site) = column(data, "site") else { panic!("site") };
    let labels: Vec<&str> = site.domain.levels.iter().map(|level| &*level.label).collect();
    assert_eq!(labels, vec!["north", "south", "east"], "strings keep first-appearance order");
    assert!(!site.validity.is_valid(3));
}

#[test]
fn arrow_ipc_keeps_dictionary_order_unused_levels_order_flag_and_time_zone() {
    let scratch = Scratch::new("ipc-dictionary");
    let levels: ArrayRef = Arc::new(StringArray::from(vec!["low", "mid", "high"]));
    let keys = Int32Array::from(vec![Some(1), Some(0), None, Some(1)]);
    let grade = DictionaryArray::<Int32Type>::try_new(keys, levels).unwrap();
    let dictionary = DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));
    #[allow(deprecated)]
    let grade_field = Field::new_dict("grade", dictionary, true, 0, true);
    let schema = Schema::new(vec![
        grade_field,
        Field::new("at", DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into())), false),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(grade),
            Arc::new(TimestampMillisecondArray::from(vec![1, 2, 3, 4]).with_timezone("UTC")),
        ],
    )
    .unwrap();
    let path = scratch.path("ordered.arrow");
    write_ipc_file(&path, &batch, 2);
    let table = TableFile::open(&path).unwrap();
    assert_eq!(table.format(), TableFileFormat::ArrowIpcFile);
    assert_eq!(table.declared_rows(), None, "Arrow IPC records no row count in its footer");
    assert_eq!(table.columns()[0].kind, FileColumnKind::Categorical { ordered: true });
    assert_eq!(table.columns()[1].kind, FileColumnKind::Timestamp { timezone: Some("UTC".into()) });
    let plan = table.plan_all(ReadOptions::with_chunk_rows(3)).unwrap();
    let materialized = table.materialize(&plan, &ctx()).unwrap();
    let ColumnView::Categorical(grade) = column(materialized.data(), "grade") else {
        panic!("grade is categorical")
    };
    let labels: Vec<&str> = grade.domain.levels.iter().map(|level| &*level.label).collect();
    assert_eq!(labels, vec!["low", "mid", "high"], "dictionary order, unused level kept");
    assert!(grade.domain.ordered);
    assert_eq!(grade.codes[0].raw(), 1);
    assert!(!grade.validity.is_valid(2));
    assert_eq!(materialized.receipt().levels(0).len(), 3);
}

#[test]
fn projection_decodes_only_the_closure_and_refuses_what_it_cannot_read() {
    let scratch = Scratch::new("projection");
    let nested = ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
        Some(vec![Some(1)]),
        Some(vec![Some(2), Some(3)]),
        None,
    ]);
    let schema = Schema::new(vec![
        Field::new("a", DataType::Float64, false),
        Field::new("nested", nested.data_type().clone(), true),
        Field::new("b", DataType::Float64, false),
        Field::new("c", DataType::Float64, false),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            Arc::new(nested),
            Arc::new(Float64Array::from(vec![4.0, 5.0, 6.0])),
            Arc::new(Float64Array::from(vec![7.0, 8.0, 9.0])),
        ],
    )
    .unwrap();
    let path = scratch.path("wide.parquet");
    write_parquet(&path, &batch, 10);
    let table = TableFile::open(&path).unwrap();
    assert!(matches!(table.columns()[1].kind, FileColumnKind::Unsupported { .. }));

    let closure =
        ClosureInputs::new([ColumnRole::new("c")]).validator(ColumnRole::new("a")).close();
    let plan = table.plan(&closure, ReadOptions::default()).unwrap();
    assert_eq!(plan.projection(), &[0, 3], "only the closure is decoded, in file order");
    let names: Vec<String> = plan.names().iter().map(ToString::to_string).collect();
    assert_eq!(names, vec!["c", "a"], "columns come back in closure order");
    let full = table.plan_columns(&["a", "b", "c"], ReadOptions::default()).unwrap();
    assert!(plan.chunk_quote().unwrap().total_bytes() < full.chunk_quote().unwrap().total_bytes());

    let mut reader = table.reader(&plan, &ctx()).unwrap();
    let chunk = reader.next_chunk().unwrap().unwrap();
    assert_eq!(chunk.columns().len(), 2);
    let Some(antecedent_data::OwnedColumn::Float64(c)) = chunk.column("c") else {
        panic!("c is float64")
    };
    assert_eq!(c.values.as_slice(), &[7.0, 8.0, 9.0]);
    drop(chunk);
    assert!(reader.next_chunk().unwrap().is_none());

    let refused = table.plan_columns(&["a", "nested"], ReadOptions::default()).unwrap_err();
    let FileSourceError::UnsupportedColumn { column, next_attempt, .. } = &refused else {
        panic!("nested columns are refused: {refused}")
    };
    assert_eq!(column, "nested");
    assert!(next_attempt.contains("flatten"), "{next_attempt}");
    let unknown = table.plan_columns(&["missing"], ReadOptions::default()).unwrap_err();
    assert!(matches!(unknown, FileSourceError::UnknownColumn { .. }), "{unknown}");
}

#[test]
fn failed_later_column_cannot_advance_or_publish_a_corrupt_scan() {
    let scratch = Scratch::new("conversion-failure");
    let path = scratch.path("values.arrow");
    let schema = Arc::new(Schema::new(vec![
        Field::new("category", DataType::Utf8, false),
        Field::new("too_large", DataType::UInt64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec!["first", "second", "third"])),
            Arc::new(UInt64Array::from(vec![1, u64::MAX, 3])),
        ],
    )
    .unwrap();
    write_ipc_file(&path, &batch, 3);
    let table = TableFile::open(&path).unwrap();
    let plan =
        table.plan_columns(&["category", "too_large"], ReadOptions::with_chunk_rows(3)).unwrap();
    let mut reader = table.reader(&plan, &ctx()).unwrap();
    let first = reader.next_chunk().unwrap_err();
    assert!(matches!(first, FileSourceError::UnsupportedValue { .. }));
    assert_eq!(reader.position(), GlobalId::new(0));
    assert_eq!(reader.next_chunk().unwrap_err(), first);
    assert_eq!(reader.finish().unwrap_err(), first);
    reader.rewind().unwrap();
    assert_eq!(reader.position(), GlobalId::new(0));
    assert_eq!(reader.next_chunk().unwrap_err(), first);
}

#[test]
fn materialize_refuses_a_file_that_grew_after_its_quote() {
    let scratch = Scratch::new("changed-rows");
    let path = scratch.path("numbers.parquet");
    write_parquet(&path, &numeric_batch(2, 0.0), 2);
    let table = TableFile::open(&path).unwrap();
    let plan = table.plan_columns(&["x", "y"], ReadOptions::with_chunk_rows(3)).unwrap();
    assert_eq!(table.quote(&plan).unwrap().unwrap().rows(), GlobalId::new(2));
    write_parquet(&path, &numeric_batch(3, 0.0), 3);
    assert!(matches!(table.materialize(&plan, &ctx()), Err(FileSourceError::Data(_))));
    // A shorter replacement also cannot produce a complete receipt for the original quote.
    write_parquet(&path, &numeric_batch(1, 0.0), 1);
    assert!(matches!(table.materialize(&plan, &ctx()), Err(FileSourceError::Data(_))));
}

#[test]
fn chunked_admission_refuses_before_allocating_and_returns_the_pool() {
    let scratch = Scratch::new("admission");
    let path = scratch.path("numbers.parquet");
    write_parquet(&path, &numeric_batch(1_000, 0.0), 250);
    let table = TableFile::open(&path).unwrap();
    let plan = table.plan_columns(&["x", "y"], ReadOptions::with_chunk_rows(100)).unwrap();
    let quote = plan.chunk_quote().unwrap();

    let tiny = AdmissionPool::fixed(1, quote.decode_bytes - 1);
    let study = ctx().with_admission(tiny.clone());
    let mut reader = table.reader(&plan, &study).unwrap();
    let refused = reader.next_chunk().unwrap_err();
    let FileSourceError::MemoryRefused { stage, required_bytes, available_bytes, .. } = &refused
    else {
        panic!("a pool below one chunk is refused: {refused}")
    };
    assert_eq!(*stage, MemoryStage::ChunkDecode);
    assert_eq!(*required_bytes, quote.decode_bytes);
    assert_eq!(*available_bytes, quote.decode_bytes - 1);
    assert!(refused.next_attempt().unwrap().contains("chunk_rows"));
    assert_eq!(tiny.available().bytes(), quote.decode_bytes - 1, "nothing stays charged");
    drop(reader);

    let fits_decode = AdmissionPool::fixed(1, quote.decode_bytes + quote.owned_bytes / 2);
    let study = ctx().with_admission(fits_decode.clone());
    let mut reader = table.reader(&plan, &study).unwrap();
    let refused = reader.next_chunk().unwrap_err();
    assert!(
        matches!(
            refused,
            FileSourceError::MemoryRefused {
                stage: MemoryStage::ChunkColumns | MemoryStage::DecodedBatch,
                ..
            }
        ),
        "the owned columns of a chunk are charged before they are built: {refused}"
    );
    drop(reader);
    assert_eq!(fits_decode.available().bytes(), fits_decode.byte_budget(), "pool restored");

    // Slack for decoded Arrow buffers rounded up past the per-row decode quote.
    let one_chunk_budget = quote.total_bytes() + 65_536;
    let one_chunk = AdmissionPool::fixed(1, one_chunk_budget);
    let study = ctx().with_admission(one_chunk.clone());
    let mut reader = table.reader(&plan, &study).unwrap();
    let mut rows = 0;
    while let Some(chunk) = reader.next_chunk().unwrap() {
        assert!(chunk.rows() <= 100);
        assert!(chunk.held_bytes() > 0, "a chunk holds its lease");
        rows += chunk.rows();
    }
    assert_eq!(rows, 1_000, "one chunk at a time fits a pool of one chunk");
    assert!(reader.finish().unwrap().peak_chunk_bytes() <= one_chunk_budget);
    drop(reader);
    assert_eq!(one_chunk.available().bytes(), one_chunk_budget);

    let expected = table.quote(&plan).unwrap().unwrap();
    let short = AdmissionPool::fixed(1, expected.total_bytes() - 1);
    let refused = table.materialize(&plan, &ctx().with_admission(short.clone())).unwrap_err();
    let FileSourceError::MemoryRefused { stage, required_bytes, .. } = &refused else {
        panic!("materialization is quoted before reading: {refused}")
    };
    assert_eq!(*stage, MemoryStage::Materialization);
    assert_eq!(*required_bytes, expected.total_bytes());
    assert!(refused.next_attempt().unwrap().contains("summary route"));
    assert_eq!(short.available().bytes(), expected.total_bytes() - 1, "refused before reading");

    let enough_budget = expected.total_bytes() + 65_536;
    let enough = AdmissionPool::fixed(1, enough_budget);
    let materialized = table.materialize(&plan, &ctx().with_admission(enough.clone())).unwrap();
    assert_eq!(materialized.data().row_count(), 1_000);
    assert!(materialized.held_bytes() > 0);
    drop(materialized);
    assert_eq!(enough.available().bytes(), enough_budget);
}

#[test]
fn content_identity_is_chunk_invariant_and_changes_with_any_value() {
    let scratch = Scratch::new("identity");
    let base = numeric_batch(500, 0.0);
    let first = scratch.path("first.parquet");
    let regrouped = scratch.path("regrouped.parquet");
    let ipc = scratch.path("same.arrow");
    write_parquet(&first, &base, 64);
    write_parquet(&regrouped, &base, 333);
    write_ipc_file(&ipc, &base, 77);

    let identity = |path: &Path, chunk_rows: usize| {
        let table = TableFile::open(path).unwrap();
        let plan = table.plan_all(ReadOptions::with_chunk_rows(chunk_rows)).unwrap();
        table.scan_identity(&plan, &ctx()).unwrap().identity()
    };
    let reference = identity(&first, 1);
    assert_eq!(reference.rows(), GlobalId::new(500));
    for chunk_rows in [3, 64, 100, 499, 500, 10_000] {
        assert_eq!(identity(&first, chunk_rows), reference, "chunk rows {chunk_rows}");
    }
    assert_eq!(identity(&regrouped, 17), reference, "row-group layout is not content");
    assert_eq!(identity(&ipc, 50), reference, "the container is not content");

    let mut y: Vec<f64> = (0..500).map(|i| 1.0 + 2.0 * f64::from(i)).collect();
    y[499] += 1e-9;
    let changed = RecordBatch::try_new(
        base.schema(),
        vec![
            Arc::clone(base.column(0)),
            Arc::new(Float64Array::from(y)),
            Arc::clone(base.column(2)),
        ],
    )
    .unwrap();
    let edited = scratch.path("edited.parquet");
    write_parquet(&edited, &changed, 64);
    assert_ne!(identity(&edited, 64), reference, "one changed value changes the identity");

    let table = TableFile::open(&first).unwrap();
    let plan = table.plan_all(ReadOptions::with_chunk_rows(100)).unwrap();
    let mut reader = table.reader(&plan, &ctx()).unwrap();
    let _ = reader.next_chunk().unwrap();
    assert_eq!(reader.finish().unwrap_err(), FileSourceError::ScanIncomplete);
    while reader.next_chunk().unwrap().is_some() {}
    let first_pass = reader.finish().unwrap().identity();
    reader.rewind().unwrap();
    while reader.next_chunk().unwrap().is_some() {}
    assert_eq!(reader.finish().unwrap().identity(), first_pass, "a second pass agrees");
}

#[test]
fn row_ids_past_u32_max_are_checked_without_allocating_rows() {
    let mut cursor = RowCursor::new();
    let step = u64::from(u32::MAX);
    assert_eq!(cursor.advance(step).unwrap(), GlobalId::new(0));
    let start = cursor.advance(step).unwrap();
    assert_eq!(start.get(), step, "the second chunk starts at u32::MAX");
    assert_eq!(cursor.position().get(), 2 * step, "positions above u32::MAX are ordinary");
    let row = GlobalId::new(start.get() + 5);
    assert_eq!(chunk_local_index(start, row).unwrap().get(), 5);
    assert!(chunk_local_index(GlobalId::new(0), GlobalId::new(step + 1)).is_err());
    let mut end = RowCursor::starting_at(GlobalId::new(u64::MAX - 1));
    assert!(matches!(end.advance(2), Err(FileSourceError::CountOverflow { .. })));

    let scratch = Scratch::new("large-quote");
    let path = scratch.path("small.parquet");
    write_parquet(&path, &numeric_batch(10, 0.0), 10);
    let table = TableFile::open(&path).unwrap();
    let plan = table.plan_columns(&["x", "y"], ReadOptions::default()).unwrap();
    let rows = GlobalId::new(5_000_000_000);
    let quote = MaterializationQuote::for_rows(rows, &plan).unwrap();
    assert_eq!(quote.rows(), rows);
    assert_eq!(quote.values_bytes(), 5_000_000_000 * 16);
    assert_eq!(quote.mask_bytes(), 5_000_000_000_u64.div_ceil(8) * 2);
    let refused = quote.admit(1 << 30, plan.chunk_rows()).unwrap_err();
    let FileSourceError::MemoryRefused { required_bytes, stage, .. } = &refused else {
        panic!("a table above the pool is refused: {refused}")
    };
    assert_eq!(*stage, MemoryStage::Materialization);
    assert_eq!(*required_bytes, quote.total_bytes());
    assert!(MaterializationQuote::for_rows(GlobalId::new(u64::MAX), &plan).is_err());
}

#[test]
fn null_masks_follow_the_invalid_when_nan_rule_and_all_null_text_is_invalid() {
    let scratch = Scratch::new("nulls");
    let schema = Schema::new(vec![
        Field::new("v", DataType::Float64, true),
        Field::new("label", DataType::Utf8, true),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Float64Array::from(vec![None, Some(f64::NAN), Some(2.0)])),
            Arc::new(StringArray::from(vec![None::<&str>, None, None])),
        ],
    )
    .unwrap();
    let path = scratch.path("nulls.arrow");
    write_ipc_file(&path, &batch, 3);
    let table = TableFile::open(&path).unwrap();
    let plan = table.plan_all(ReadOptions::default()).unwrap();
    let materialized = table.materialize(&plan, &ctx()).unwrap();
    let ColumnView::Float64(v) = column(materialized.data(), "v") else { panic!("v") };
    assert!(!v.validity.is_valid(0) && !v.validity.is_valid(1) && v.validity.is_valid(2));
    let ColumnView::Categorical(label) = column(materialized.data(), "label") else {
        panic!("label")
    };
    assert!((0..3).all(|row| !label.validity.is_valid(row)));
}

#[test]
fn a_dataset_directory_reads_its_files_in_order_and_refuses_a_different_schema() {
    let scratch = Scratch::new("dataset");
    write_parquet(&scratch.path("part-0.parquet"), &numeric_batch(30, 0.0), 30);
    write_parquet(&scratch.path("part-1.parquet"), &numeric_batch(20, 100.0), 7);
    std::fs::write(scratch.path("_SUCCESS"), b"").unwrap();
    let table = TableFile::open(&scratch.0).unwrap();
    assert_eq!(table.paths().len(), 2);
    assert_eq!(table.declared_rows(), Some(GlobalId::new(50)));
    let plan = table.plan_columns(&["y"], ReadOptions::with_chunk_rows(8)).unwrap();
    let receipt = table.scan_identity(&plan, &ctx()).unwrap();
    assert_eq!(receipt.rows(), GlobalId::new(50));
    let materialized = table.materialize(&plan, &ctx()).unwrap();
    let ColumnView::Float64(y) = column(materialized.data(), "y") else { panic!("y") };
    assert_eq!(y.values.as_slice()[30], 101.0, "the second file follows the first");
    assert_eq!(y.values.as_slice()[29], 1.0 + 2.0 * 29.0, "the first file ends at row 29");
    assert_eq!(y.values.as_slice()[49], 1.0 + 2.0 * 19.0 + 100.0, "the second file ends last");

    let other = Schema::new(vec![Field::new("x", DataType::Int32, false)]);
    let odd =
        RecordBatch::try_new(Arc::new(other), vec![Arc::new(Int32Array::from(vec![1]))]).unwrap();
    write_parquet(&scratch.path("part-2.parquet"), &odd, 1);
    let refused = TableFile::open(&scratch.0).unwrap_err();
    assert!(matches!(refused, FileSourceError::SchemaMismatch { .. }), "{refused}");
}

#[test]
fn the_writer_round_trips_library_tables_through_every_format() {
    let scratch = Scratch::new("writer");
    let data = TabularData::from_f64_columns([
        ("t", &[0.0_f64, 1.0, 0.0, 1.0, f64::NAN][..]),
        ("y", &[1.0_f64, 3.0, 1.5, 2.5, 0.5][..]),
    ])
    .unwrap();
    for (name, format) in [
        ("table.parquet", TableFileFormat::Parquet),
        ("table.arrow", TableFileFormat::ArrowIpcFile),
        ("table.arrows", TableFileFormat::ArrowIpcStream),
    ] {
        let path = scratch.path(name);
        let rows =
            write_table_file(&path, format, &data, WriteOptions { rows_per_group: 2 }).unwrap();
        assert_eq!(rows, GlobalId::new(5));
        let table = TableFile::open(&path).unwrap();
        assert_eq!(table.format(), format);
        let plan = table.plan_all(ReadOptions::with_chunk_rows(2)).unwrap();
        let materialized = table.materialize(&plan, &ctx()).unwrap();
        assert_eq!(
            materialized.data().storage().content_digest(),
            data.storage().content_digest(),
            "{name} round-trips the stored columns and masks"
        );
        assert_eq!(materialized.snapshot_digest(), data.storage().content_digest());
        // Independent read-back of the written values, not only the digest.
        let ColumnView::Float64(t) = column(materialized.data(), "t") else { panic!("t") };
        assert_eq!(&t.values.as_slice()[..4], &[0.0, 1.0, 0.0, 1.0], "{name}");
        assert!(!t.validity.is_valid(4), "{name}: a NaN is written invalid");
        assert!((0..4).all(|row| t.validity.is_valid(row)), "{name}");
        let ColumnView::Float64(y) = column(materialized.data(), "y") else { panic!("y") };
        assert_eq!(y.values.as_slice(), &[1.0, 3.0, 1.5, 2.5, 0.5], "{name}");
        let left = materialized.row_range_identity(0..2).unwrap();
        assert_ne!(left, materialized.row_range_identity(2..5).unwrap());
        assert_ne!(left, materialized.snapshot_digest(), "a row range is not the full source");
    }
}

#[test]
fn an_arrow_c_stream_reads_once_in_bounded_chunks() {
    let batch = numeric_batch(10, 0.0);
    let schema: SchemaRef = batch.schema();
    let reader =
        RecordBatchIterator::new(vec![Ok(batch.slice(0, 6)), Ok(batch.slice(6, 4))], schema);
    let stream = ArrowCStream::from_reader(Box::new(reader));
    let table = TableStream::new(stream).unwrap();
    assert_eq!(table.columns().len(), 3);
    let closure = ClosureInputs::new([ColumnRole::new("y")]).close();
    let plan = table.plan(&closure, ReadOptions::with_chunk_rows(4)).unwrap();
    let mut reader = table.reader(&plan, &ctx()).unwrap();
    let mut sizes = Vec::new();
    while let Some(chunk) = reader.next_chunk().unwrap() {
        sizes.push(chunk.rows());
    }
    assert_eq!(sizes, vec![4, 2, 4]);
    assert_eq!(reader.finish().unwrap().rows(), GlobalId::new(10));
    assert_eq!(reader.rewind().unwrap_err(), FileSourceError::NotRewindable);
}

#[test]
fn cancellation_stops_the_read_and_returns_its_leases() {
    let scratch = Scratch::new("cancel");
    let path = scratch.path("numbers.parquet");
    write_parquet(&path, &numeric_batch(100, 0.0), 10);
    let table = TableFile::open(&path).unwrap();
    let plan = table.plan_all(ReadOptions::with_chunk_rows(10)).unwrap();
    let pool = AdmissionPool::fixed(1, 1 << 30);
    let study = ctx().with_admission(pool.clone());
    let mut reader = table.reader(&plan, &study).unwrap();
    let chunk = reader.next_chunk().unwrap().unwrap();
    study.cancellation.cancel();
    assert_eq!(reader.next_chunk().unwrap_err(), FileSourceError::Cancelled);
    drop(chunk);
    drop(reader);
    assert_eq!(pool.available().bytes(), 1 << 30);
}
