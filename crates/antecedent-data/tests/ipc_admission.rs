// SPDX-License-Identifier: MIT OR Apache-2.0
//! Writer-sized Arrow IPC batches are admitted before they are decoded.
//!
//! An IPC reader materializes a record batch at the size its writer chose. These tests
//! measure the live heap of the reading thread with a counting allocator, so a budget
//! violation is caught by what was actually allocated, not by a constant: a batch the pool
//! cannot hold is refused while the heap stays far below the batch, and a batch the pool
//! holds is read with a peak under the pool's byte budget. Every test takes the measuring
//! lock, because the counters are shared by the whole test binary.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicIsize, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use antecedent_core::ExecutionContext;
use antecedent_core::execution::AdmissionPool;
use antecedent_data::file_source::{FileSourceError, MemoryStage, ReadOptions, TableFile};
use arrow_array::{ArrayRef, Float64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};

thread_local! {
    static MEASURING: Cell<bool> = const { Cell::new(false) };
}

static LIVE: AtomicIsize = AtomicIsize::new(0);
static PEAK: AtomicIsize = AtomicIsize::new(0);

struct Counting;

// SAFETY: every method forwards to the system allocator with the layout it was given; the
// counters are updated only on the thread that opted in and never allocate.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller upholds the `GlobalAlloc::alloc` contract for `layout`.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && MEASURING.with(Cell::get) {
            let size = isize::try_from(layout.size()).unwrap_or(isize::MAX);
            let now = LIVE.fetch_add(size, Ordering::SeqCst) + size;
            PEAK.fetch_max(now, Ordering::SeqCst);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        if MEASURING.with(Cell::get) {
            LIVE.fetch_sub(isize::try_from(layout.size()).unwrap_or(isize::MAX), Ordering::SeqCst);
        }
        // SAFETY: `pointer` came from `System.alloc` with this `layout`.
        unsafe { System.dealloc(pointer, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

static SERIAL: Mutex<()> = Mutex::new(());

/// Holds the measuring lock; heap activity of this thread is counted until it drops.
struct Measure {
    _lock: MutexGuard<'static, ()>,
    start: isize,
}

impl Measure {
    fn begin() -> Self {
        let lock = SERIAL.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        LIVE.store(0, Ordering::SeqCst);
        PEAK.store(0, Ordering::SeqCst);
        MEASURING.with(|flag| flag.set(true));
        Self { _lock: lock, start: 0 }
    }

    /// Largest live heap above the start since `begin`, in bytes.
    fn peak(&self) -> usize {
        usize::try_from(PEAK.load(Ordering::SeqCst) - self.start).unwrap_or(0)
    }

    fn pause(&self) {
        MEASURING.with(|flag| flag.set(false));
    }

    fn resume(&self) {
        MEASURING.with(|flag| flag.set(true));
    }
}

impl Drop for Measure {
    fn drop(&mut self) {
        MEASURING.with(|flag| flag.set(false));
    }
}

static NEXT: AtomicUsize = AtomicUsize::new(0);

struct Scratch(PathBuf);

impl Scratch {
    fn new(label: &str) -> Self {
        let n = NEXT.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir()
            .join(format!("antecedent-ipc-admission-{}-{label}-{n}", std::process::id()));
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

fn write_ipc(path: &Path, batch: &RecordBatch, stream: bool) {
    let file = std::fs::File::create(path).unwrap();
    if stream {
        let mut writer = arrow_ipc::writer::StreamWriter::try_new(file, &batch.schema()).unwrap();
        writer.write(batch).unwrap();
        writer.finish().unwrap();
    } else {
        let mut writer = arrow_ipc::writer::FileWriter::try_new(file, &batch.schema()).unwrap();
        writer.write(batch).unwrap();
        writer.finish().unwrap();
    }
}

fn numeric(rows: usize) -> RecordBatch {
    let x: Vec<f64> = (0..rows).map(|i| i as f64).collect();
    let y: Vec<f64> = (0..rows).map(|i| 2.0 * i as f64 + 1.0).collect();
    let schema = Schema::new(vec![
        Field::new("x", DataType::Float64, false),
        Field::new("y", DataType::Float64, false),
    ]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(Float64Array::from(x)) as ArrayRef, Arc::new(Float64Array::from(y))],
    )
    .unwrap()
}

/// Rows of long strings: the body is dominated by variable-width data.
fn texts(rows: usize, width: usize) -> RecordBatch {
    let values: Vec<String> = (0..rows).map(|i| format!("{i:08}").repeat(width / 8)).collect();
    let schema = Schema::new(vec![
        Field::new("t", DataType::Utf8, false),
        Field::new("v", DataType::Float64, false),
    ]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(values)) as ArrayRef,
            Arc::new(Float64Array::from((0..rows).map(|i| i as f64).collect::<Vec<_>>())),
        ],
    )
    .unwrap()
}

fn study(pool: &AdmissionPool) -> ExecutionContext {
    ExecutionContext::for_tests(1).with_admission(pool.clone())
}

fn refusal(error: &FileSourceError) -> (MemoryStage, u64, u64) {
    let FileSourceError::MemoryRefused { stage, required_bytes, available_bytes, .. } = error
    else {
        panic!("expected a memory refusal, found: {error}")
    };
    (*stage, *required_bytes, *available_bytes)
}

#[test]
fn a_writer_sized_numeric_batch_is_refused_before_it_is_decoded() {
    const ROWS: usize = 1_000_000;
    for stream in [false, true] {
        let scratch = Scratch::new("numeric");
        let path = scratch.path(if stream { "n.arrows" } else { "n.arrow" });
        write_ipc(&path, &numeric(ROWS), stream);
        let table = TableFile::open(&path).unwrap();
        let plan = table.plan_columns(&["x"], ReadOptions::with_chunk_rows(1000)).unwrap();
        let quote = plan.chunk_quote().unwrap();
        let body = (ROWS * 16) as u64;

        let pool = AdmissionPool::fixed(1, 4 << 20);
        assert!(quote.total_bytes() < pool.byte_budget(), "one chunk fits; the batch does not");
        let ctx = study(&pool);
        let mut reader = table.reader(&plan, &ctx).unwrap();
        let measure = Measure::begin();
        let refused = reader.next_chunk().unwrap_err();
        let peak = measure.peak();
        drop(measure);
        let (stage, required, available) = refusal(&refused);
        assert_eq!(stage, MemoryStage::DecodedBatch);
        assert!(required >= body, "the quote covers the whole message body: {required}");
        assert_eq!(available, pool.byte_budget());
        assert!(
            peak < 1 << 20,
            "a refused batch must not be materialized: peak {peak} for a {body}-byte body"
        );
        assert!(refused.next_attempt().unwrap().contains("smaller record batches"));
        drop(reader);
        assert_eq!(pool.available().bytes(), pool.byte_budget(), "nothing stays charged");
    }
}

#[test]
fn a_variable_width_batch_is_charged_for_its_strings_before_decoding() {
    const ROWS: usize = 20_000;
    let scratch = Scratch::new("text");
    let path = scratch.path("t.arrow");
    write_ipc(&path, &texts(ROWS, 1024), false);
    let table = TableFile::open(&path).unwrap();
    let plan = table.plan_columns(&["t", "v"], ReadOptions::with_chunk_rows(500)).unwrap();
    // The default text quote is 64 bytes a row; the strings here are sixteen times wider.
    let assumed = plan.quote_rows(ROWS).unwrap().decode_bytes;
    let strings = (ROWS * 1024) as u64;
    assert!(assumed < strings / 4, "the per-row assumption underestimates this file");

    let pool = AdmissionPool::fixed(1, 6 << 20);
    let ctx = study(&pool);
    let mut reader = table.reader(&plan, &ctx).unwrap();
    let measure = Measure::begin();
    let refused = reader.next_chunk().unwrap_err();
    let peak = measure.peak();
    drop(measure);
    let (stage, required, _) = refusal(&refused);
    assert_eq!(stage, MemoryStage::DecodedBatch);
    assert!(required >= strings, "variable-width bytes are in the admission: {required}");
    assert!(peak < 1 << 20, "peak {peak} for a {strings}-byte string body");
    drop(reader);
    assert_eq!(pool.available().bytes(), pool.byte_budget());
}

#[test]
fn an_admitted_batch_is_read_within_the_pool_byte_budget() {
    const ROWS: usize = 400_000;
    let scratch = Scratch::new("fits");
    let path = scratch.path("f.arrow");
    write_ipc(&path, &numeric(ROWS), false);
    let table = TableFile::open(&path).unwrap();
    let plan = table.plan_columns(&["x", "y"], ReadOptions::with_chunk_rows(5000)).unwrap();
    let body = (ROWS * 16) as u64;
    let budget = body + body / 4 + (1 << 20);
    let pool = AdmissionPool::fixed(1, budget);
    let ctx = study(&pool);
    let mut reader = table.reader(&plan, &ctx).unwrap();
    let measure = Measure::begin();
    let mut rows = 0usize;
    while let Some(chunk) = reader.next_chunk().unwrap() {
        rows += chunk.rows();
        assert_eq!(chunk.column("x").unwrap().len(), chunk.rows());
    }
    let receipt = reader.finish().unwrap();
    let peak = measure.peak();
    measure.pause();
    assert_eq!(rows, ROWS);
    assert!(
        peak as u64 <= pool.byte_budget(),
        "live heap {peak} passed the pool budget {}",
        pool.byte_budget()
    );
    assert!(receipt.peak_chunk_bytes() <= pool.byte_budget());
    measure.resume();
    drop(reader);
    assert_eq!(pool.available().bytes(), budget);
}

#[test]
fn concurrent_readers_share_one_pool_and_none_decodes_a_batch_the_pool_cannot_hold() {
    const ROWS: usize = 300_000;
    let scratch = Scratch::new("shared");
    let path = scratch.path("s.arrow");
    write_ipc(&path, &numeric(ROWS), false);
    let table = TableFile::open(&path).unwrap();
    let plan = table.plan_columns(&["x", "y"], ReadOptions::with_chunk_rows(4000)).unwrap();
    let body = (ROWS * 16) as u64;
    // Room for one decoded batch and a little more, not for two.
    let pool = AdmissionPool::fixed(4, body + body / 2);
    let completed = AtomicUsize::new(0);
    let refused = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..6 {
            scope.spawn(|| {
                let ctx = study(&pool);
                let mut reader = table.reader(&plan, &ctx).unwrap();
                loop {
                    match reader.next_chunk() {
                        Ok(Some(_)) => {}
                        Ok(None) => {
                            completed.fetch_add(1, Ordering::SeqCst);
                            break;
                        }
                        Err(error) => {
                            assert!(matches!(
                                refusal(&error).0,
                                MemoryStage::DecodedBatch | MemoryStage::ChunkColumns
                            ));
                            refused.fetch_add(1, Ordering::SeqCst);
                            break;
                        }
                    }
                }
            });
        }
    });
    assert_eq!(completed.load(Ordering::SeqCst) + refused.load(Ordering::SeqCst), 6);
    assert!(completed.load(Ordering::SeqCst) >= 1, "at least one reader owns the batch");
    assert_eq!(pool.available().bytes(), pool.byte_budget(), "every refusal and finish returned");
}

#[test]
fn an_arrow_c_stream_batch_larger_than_the_pool_is_refused_with_the_producer_remedy() {
    use antecedent_data::arrow_ffi::ArrowCStream;
    use antecedent_data::column_closure::{ClosureInputs, ColumnRole};
    use antecedent_data::file_source::TableStream;
    use arrow_array::RecordBatchIterator;

    let closure = ClosureInputs::new([ColumnRole::new("x")]).close();

    let batch = numeric(200_000);
    let schema = batch.schema();
    let big = RecordBatchIterator::new(vec![Ok(batch.clone())], Arc::clone(&schema));
    let table = TableStream::new(ArrowCStream::from_reader(Box::new(big))).unwrap();
    let plan = table.plan(&closure, ReadOptions::with_chunk_rows(1000)).unwrap();
    let pool = AdmissionPool::fixed(1, 1 << 20);
    let mut reader = table.reader(&plan, &study(&pool)).unwrap();
    let refused = reader.next_chunk().unwrap_err();
    let (stage, required, _) = refusal(&refused);
    assert_eq!(stage, MemoryStage::DecodedBatch);
    assert!(required >= 1_500_000, "the projected column alone is 1.6 MB: {required}");
    let remedy = refused.next_attempt().unwrap();
    assert!(
        remedy.contains("from the producer") && remedy.contains("max_chunksize=1000"),
        "{remedy}"
    );
    drop(reader);
    assert_eq!(pool.available().bytes(), pool.byte_budget());

    // The same rows re-batched at the producer fit the same pool.
    let small: Vec<_> = (0..200).map(|i| Ok(batch.slice(i * 1000, 1000))).collect();
    let rebatched = RecordBatchIterator::new(small, schema);
    let table = TableStream::new(ArrowCStream::from_reader(Box::new(rebatched))).unwrap();
    let plan = table.plan(&closure, ReadOptions::with_chunk_rows(1000)).unwrap();
    let mut reader = table.reader(&plan, &study(&pool)).unwrap();
    let mut rows = 0;
    while let Some(chunk) = reader.next_chunk().unwrap() {
        rows += chunk.rows();
    }
    assert_eq!(rows, 200_000);
}
