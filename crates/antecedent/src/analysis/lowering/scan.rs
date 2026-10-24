//! Layer five's scan operators: the projection that selects the columns a route
//! reads and the scan that streams its rows.
//!
//! An in-memory route scans its table once as a single resident chunk. The plan
//! sizes that chunk from the rows, the projected source columns and the real
//! design width, with the dense planner's three copies (the design and two
//! workspaces). A file-backed route reads through a [`ReadPlan`], which already
//! fixes the projection and the rows per chunk; [`ScanPlan::from_read_plan`]
//! lifts that plan, its chunk quote and the accumulator the capacity route keeps
//! into the same typed scan, with the summary frame as the spilled output when
//! retention is disk-backed.
//!
//! [`ScanPlan::operators`] declares the projection and the scan as physical
//! operators. The scan carries the scan cancellation point, its scratch is one
//! chunk, its retained bytes are the accumulator, and its output is the spill.
//! [`install`](super::install) adds them to the program's physical layer for
//! every route that reads a table.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the plan view reads the scan through the declared operators; the accessors \
                  and the chunked mode serve the file-backed lowering and the tests"
    )
)]

use antecedent_core::execution_facts::OutputRetention;
use antecedent_core::{
    ByteEstimates, CancellationPoint, DesignDependencyId, PhysicalEffect, PhysicalOperator,
    SourceDependencyId,
};
use antecedent_data::file_source::ReadPlan;

use crate::error::CausalError;

/// Rows per batch a plan that streams its design reads from the source columns.
pub(crate) const STREAMED_BATCH_ROWS: usize = 8192;
/// Passes the streamed reduction makes over the rows: the cross-product pass and the
/// residual pass.
const STREAMED_PASSES: u64 = 2;

const SOURCE_DEPENDENCY: u64 = 1;
const DESIGN_DEPENDENCY: u64 = 1;
/// Bytes of one `f64`.
const WORD: u64 = 8;
/// Copies of the design the dense planner holds: the design and two workspaces.
const DENSE_COPIES: u64 = 3;

/// What one scan reads.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ScanDemand {
    /// Source rows scanned.
    pub(crate) rows: u64,
    /// Source columns the projection keeps.
    pub(crate) projected_columns: u64,
    /// Encoded design columns, when the route fits one design.
    pub(crate) design_columns: Option<u64>,
    /// Passes over the rows the fit makes: one, plus a rescan for a robust
    /// covariance or a diagnostic, plus one per resampling replicate.
    pub(crate) passes: u64,
}

/// How the scan holds the design.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ScanMode {
    /// One chunk of every row, resident.
    Resident,
    /// Rows stream in chunks and only the accumulator stays resident.
    Chunked,
}

/// A sized scan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ScanPlan {
    mode: ScanMode,
    rows: u64,
    projected_columns: u64,
    chunk_rows: u64,
    chunks: u64,
    passes: u64,
    projection_bytes: u64,
    chunk_bytes: u64,
    accumulator_bytes: u64,
    spill_bytes: u64,
}

/// Bytes of the Gram accumulator over a design of `width` columns: `X'X`, `X'y`,
/// the response sum of squares and the row count.
const fn accumulator_bytes(width: u64) -> Option<u64> {
    let Some(cells) = width.checked_mul(width) else { return None };
    let Some(cells) = cells.checked_add(2 * width + 2) else { return None };
    cells.checked_mul(WORD)
}

fn overflow() -> CausalError {
    CausalError::Resource { message: "scan size arithmetic overflow".into() }
}

impl ScanPlan {
    /// One resident scan of `demand`.
    ///
    /// # Errors
    ///
    /// The size arithmetic overflows.
    pub(crate) fn resident(demand: ScanDemand) -> Result<Self, CausalError> {
        let rows = demand.rows.max(1);
        let projected = demand.projected_columns.max(1);
        let width = demand.design_columns.unwrap_or(projected).max(1);
        let dense = rows
            .checked_mul(width)
            .and_then(|cells| cells.checked_mul(WORD))
            .and_then(|bytes| bytes.checked_mul(DENSE_COPIES))
            .ok_or_else(overflow)?;
        let projection_bytes = rows
            .checked_mul(projected)
            .and_then(|cells| cells.checked_mul(WORD))
            .ok_or_else(overflow)?;
        Ok(Self {
            mode: ScanMode::Resident,
            rows,
            projected_columns: projected,
            chunk_rows: rows,
            chunks: 1,
            passes: demand.passes.max(1),
            projection_bytes,
            chunk_bytes: dense,
            accumulator_bytes: 0,
            spill_bytes: 0,
        })
    }

    /// The scan of an in-memory table whose design is never built.
    ///
    /// Rows are read in batches of `batch_rows` straight from the borrowed source columns
    /// through the complete-case mask. A chunk holds one batch of the design row, the
    /// response and the weight; only the Gram accumulator of the real design width stays
    /// resident across chunks; nothing is spilled. The reduction reads the rows twice.
    ///
    /// # Errors
    ///
    /// The size arithmetic overflows.
    pub(crate) fn streamed(demand: ScanDemand, batch_rows: u64) -> Result<Self, CausalError> {
        let rows = demand.rows.max(1);
        let projected = demand.projected_columns.max(1);
        let width = demand.design_columns.unwrap_or(projected).max(1);
        let chunk_rows = batch_rows.min(rows).max(1);
        let chunk_bytes = width
            .checked_add(2)
            .and_then(|cells| cells.checked_mul(chunk_rows))
            .and_then(|cells| cells.checked_mul(WORD))
            .ok_or_else(overflow)?;
        Ok(Self {
            mode: ScanMode::Chunked,
            rows,
            projected_columns: projected,
            chunk_rows,
            chunks: rows.div_ceil(chunk_rows),
            passes: demand.passes.max(STREAMED_PASSES),
            projection_bytes: chunk_bytes,
            chunk_bytes,
            accumulator_bytes: accumulator_bytes(width).ok_or_else(overflow)?,
            spill_bytes: 0,
        })
    }

    /// The streaming scan a [`ReadPlan`] defines over `rows` rows and a design of
    /// `design_columns` columns.
    ///
    /// The projection is the plan's closure, a chunk is its chunk quote, and the
    /// accumulator is the Gram the capacity route keeps across chunks. A disk-backed
    /// retention spills the finished summary, which is the accumulator frame.
    ///
    /// # Errors
    ///
    /// The plan's chunk quote or the size arithmetic overflows.
    // The file-backed summary route lowers its read plan here; a table source has no
    // compile entry yet, so the only callers are the plan tests.
    pub(crate) fn from_read_plan(
        plan: &ReadPlan,
        rows: u64,
        design_columns: u64,
        passes: u64,
        retention: OutputRetention,
    ) -> Result<Self, CausalError> {
        let quote = plan
            .chunk_quote()
            .map_err(|error| CausalError::Resource { message: error.to_string() })?;
        let rows = rows.max(1);
        let chunk_rows = u64::try_from(plan.chunk_rows()).unwrap_or(u64::MAX).min(rows).max(1);
        let accumulator = accumulator_bytes(design_columns.max(1)).ok_or_else(overflow)?;
        Ok(Self {
            mode: if chunk_rows < rows { ScanMode::Chunked } else { ScanMode::Resident },
            rows,
            projected_columns: u64::try_from(plan.columns().len()).unwrap_or(u64::MAX).max(1),
            chunk_rows,
            chunks: rows.div_ceil(chunk_rows),
            passes: passes.max(1),
            projection_bytes: quote.decode_bytes,
            chunk_bytes: quote.total_bytes(),
            accumulator_bytes: accumulator,
            spill_bytes: if retention == OutputRetention::DiskBacked { accumulator } else { 0 },
        })
    }

    /// Resident or chunked.
    pub(crate) const fn mode(&self) -> ScanMode {
        self.mode
    }

    /// Rows in a full chunk.
    pub(crate) const fn chunk_rows(&self) -> u64 {
        self.chunk_rows
    }

    /// Chunks in one pass.
    pub(crate) const fn chunks(&self) -> u64 {
        self.chunks
    }

    /// Scans of the rows the fit makes.
    pub(crate) const fn passes(&self) -> u64 {
        self.passes
    }

    /// Source columns the projection keeps.
    pub(crate) const fn projected_columns(&self) -> u64 {
        self.projected_columns
    }

    /// Bytes one chunk of the scan holds.
    pub(crate) const fn chunk_bytes(&self) -> u64 {
        self.chunk_bytes
    }

    /// Bytes held across chunks.
    pub(crate) const fn accumulator_bytes(&self) -> u64 {
        self.accumulator_bytes
    }

    /// Bytes written out when retention is disk-backed.
    pub(crate) const fn spill_bytes(&self) -> u64 {
        self.spill_bytes
    }

    /// The projection and the scan as physical operators.
    ///
    /// # Errors
    ///
    /// An operator declaration is illegal.
    pub(crate) fn operators(&self) -> Result<Vec<PhysicalOperator>, CausalError> {
        let select = PhysicalOperator::declare(PhysicalEffect::Select)
            .source(SourceDependencyId::from_raw(SOURCE_DEPENDENCY))
            .estimates(ByteEstimates::declared(self.projection_bytes_per_chunk(), 0, 0))
            .build()
            .map_err(|error| CausalError::Compile { message: error.to_string() })?;
        let scan = PhysicalOperator::declare(PhysicalEffect::Scan)
            .source(SourceDependencyId::from_raw(SOURCE_DEPENDENCY))
            .design(DesignDependencyId::from_raw(DESIGN_DEPENDENCY))
            .estimates(ByteEstimates::declared(
                self.chunk_bytes,
                self.accumulator_bytes,
                self.spill_bytes,
            ))
            .cancellation_point(CancellationPoint::Scan)
            .build()
            .map_err(|error| CausalError::Compile { message: error.to_string() })?;
        Ok(vec![select, scan])
    }

    /// Bytes of the projected columns of one chunk. Resident, that is the whole table.
    fn projection_bytes_per_chunk(&self) -> u64 {
        match self.mode {
            ScanMode::Resident => self.projection_bytes,
            ScanMode::Chunked => self
                .projection_bytes
                .min(self.chunk_rows.saturating_mul(self.projected_columns).saturating_mul(WORD)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_data::TabularData;
    use antecedent_data::file_source::{
        ReadOptions, TableFile, TableFileFormat, WriteOptions, write_table_file,
    };

    fn demand(rows: u64, projected: u64, design: Option<u64>, passes: u64) -> ScanDemand {
        ScanDemand { rows, projected_columns: projected, design_columns: design, passes }
    }

    #[test]
    fn a_resident_scan_sizes_the_projection_and_three_copies_of_the_real_design() {
        let plan = ScanPlan::resident(demand(1000, 6, Some(4), 2)).unwrap();
        assert_eq!(plan.mode(), ScanMode::Resident);
        assert_eq!(plan.chunks(), 1);
        assert_eq!(plan.chunk_rows(), 1000);
        assert_eq!(plan.passes(), 2);
        assert_eq!(plan.projected_columns(), 6);
        assert_eq!(plan.chunk_bytes(), 3 * 1000 * 4 * 8);
        assert_eq!(plan.accumulator_bytes(), 0);
        assert_eq!(plan.spill_bytes(), 0);

        // A route with no design sizes the scan from the projected columns.
        let bare = ScanPlan::resident(demand(1000, 3, None, 1)).unwrap();
        assert_eq!(bare.chunk_bytes(), 3 * 1000 * 3 * 8);

        let overflowing = ScanPlan::resident(demand(u64::MAX, 8, Some(8), 1));
        assert!(matches!(overflowing, Err(CausalError::Resource { .. })));
    }

    #[test]
    fn a_streamed_scan_holds_one_batch_and_the_gram_of_the_real_design_width() {
        let plan = ScanPlan::streamed(demand(100_000, 6, Some(4), 1), 8192).unwrap();
        assert_eq!(plan.mode(), ScanMode::Chunked);
        assert_eq!(plan.chunk_rows(), 8192);
        assert_eq!(plan.chunks(), 13, "100000 rows in batches of 8192");
        assert_eq!(plan.passes(), 2, "the cross-product pass and the residual pass");
        // One batch: the design row, the response and the weight of 8192 rows.
        assert_eq!(plan.chunk_bytes(), (4 + 2) * 8192 * 8);
        assert_eq!(plan.accumulator_bytes(), (16 + 8 + 2) * 8);
        assert_eq!(plan.spill_bytes(), 0);
        // The resident plan of the same demand holds three copies of the whole design.
        let resident = ScanPlan::resident(demand(100_000, 6, Some(4), 1)).unwrap();
        assert!(plan.chunk_bytes() < resident.chunk_bytes());
        // A table smaller than a batch is one chunk and still streams.
        let small = ScanPlan::streamed(demand(500, 3, Some(3), 1), 8192).unwrap();
        assert_eq!(small.chunk_rows(), 500);
        assert_eq!(small.chunks(), 1);
        assert_eq!(small.mode(), ScanMode::Chunked);
        let operators = plan.operators().unwrap();
        assert_eq!(operators[1].estimates().scratch(), Some(plan.chunk_bytes()));
        assert_eq!(operators[1].estimates().retained(), Some(plan.accumulator_bytes()));
        for operator in &operators {
            assert!(operator.bounded_memory_admission().is_ok());
        }
        assert!(matches!(
            ScanPlan::streamed(demand(10, 1, Some(u64::MAX), 1), 8192),
            Err(CausalError::Resource { .. })
        ));
    }

    #[test]
    fn the_scan_declares_a_projection_and_a_cancellable_scan() {
        let plan = ScanPlan::resident(demand(500, 5, Some(3), 1)).unwrap();
        let operators = plan.operators().unwrap();
        assert_eq!(operators.len(), 2);
        assert_eq!(operators[0].effect(), PhysicalEffect::Select);
        assert_eq!(operators[0].estimates().scratch(), Some(500 * 5 * 8));
        assert_eq!(operators[1].effect(), PhysicalEffect::Scan);
        assert!(operators[1].cancellation_points().contains(CancellationPoint::Scan));
        assert_eq!(operators[1].estimates().scratch(), Some(3 * 500 * 3 * 8));
        assert_eq!(operators[1].estimates().output(), Some(0));
        for operator in &operators {
            assert!(operator.bounded_memory_admission().is_ok());
        }
    }

    #[test]
    fn a_read_plan_lifts_to_a_chunked_scan_with_its_chunk_quote_and_spill() {
        let directory = std::env::temp_dir().join(format!("scan-plan-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("table.parquet");
        let n = 1000_usize;
        let t: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| i as f64 * 0.5).collect();
        let z: Vec<f64> = (0..n).map(|i| (i % 7) as f64).collect();
        let unused: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let data = TabularData::from_f64_columns([
            ("t", &t[..]),
            ("y", &y[..]),
            ("z", &z[..]),
            ("unused", &unused[..]),
        ])
        .unwrap();
        write_table_file(
            &path,
            TableFileFormat::Parquet,
            &data,
            WriteOptions { rows_per_group: 256 },
        )
        .unwrap();
        let file = TableFile::open(&path).unwrap();
        let read = file.plan_columns(&["t", "y", "z"], ReadOptions::with_chunk_rows(128)).unwrap();

        let resident =
            ScanPlan::from_read_plan(&read, n as u64, 3, 2, OutputRetention::SummaryOnly).unwrap();
        assert_eq!(resident.mode(), ScanMode::Chunked);
        assert_eq!(resident.projected_columns(), 3, "the unused column is not projected");
        assert_eq!(resident.chunk_rows(), 128);
        assert_eq!(resident.chunks(), 8);
        assert_eq!(resident.chunk_bytes(), read.chunk_quote().unwrap().total_bytes());
        assert_eq!(resident.accumulator_bytes(), (9 + 6 + 2) * 8);
        assert_eq!(resident.spill_bytes(), 0);

        let spilled =
            ScanPlan::from_read_plan(&read, n as u64, 3, 2, OutputRetention::DiskBacked).unwrap();
        assert_eq!(spilled.spill_bytes(), spilled.accumulator_bytes());
        let scan = spilled.operators().unwrap().remove(1);
        assert_eq!(scan.estimates().output(), Some(spilled.spill_bytes()));
        assert_eq!(scan.estimates().retained(), Some(spilled.accumulator_bytes()));
        std::fs::remove_dir_all(&directory).ok();
    }
}
