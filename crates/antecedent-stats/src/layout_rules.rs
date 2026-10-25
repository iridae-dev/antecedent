//! Budget-aware product and worker choice from measured layout crossovers.
//!
//! The table below holds the winning configuration of every shape measured in
//! `parity/blas_layout/layout.toml`: the reference loop or `faer`, and how many workers. A
//! request is answered from the measured shape nearest to it in work (`rows * cols²`) and,
//! for sparse storage, in column density. The caller's budget then bounds the answer:
//!
//! - the worker count never exceeds the workers the caller has available;
//! - the scratch a factorization needs at that worker count must fit the caller's scratch
//!   budget, so a parallel scope is halved until it fits and a serial scope that still does
//!   not fit is refused rather than allocated;
//! - a shape or operation with no measurement runs the serial path, the reference product
//!   where one exists.
//!
//! Nothing here reads the process pool or spawns threads. A rule moves only when a new
//! measurement moves the matching `[[selection]]`; a test compares this table with that file.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use faer::Par;
use faer::Spec;
use faer::dyn_stack::StackReq;
use faer::linalg::evd::{ComputeEigenvectors, self_adjoint_evd_scratch};
use faer::linalg::qr::col_pivoting::factor::{qr_in_place_scratch, recommended_block_size};
use faer::linalg::qr::col_pivoting::solve::solve_lstsq_in_place_scratch;
use faer::linalg::svd::{ComputeSvdVectors, svd_scratch};

use crate::faer_scope::FaerScope;

/// Dense operation whose layout is chosen.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum LayoutOp {
    /// Gram product `XᵀX` of a `rows × cols` design.
    Gram,
    /// Column-pivoted QR with one right-hand-side solve.
    ColumnPivotedQr,
    /// Thin singular value decomposition.
    ThinSvd,
    /// Symmetric eigendecomposition of a `rows × rows` matrix with eigenvectors.
    SymmetricEigen,
}

impl LayoutOp {
    /// Name used in `layout.toml`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Gram => "gram",
            Self::ColumnPivotedQr => "qr",
            Self::ThinSvd => "svd",
            Self::SymmetricEigen => "eigen",
        }
    }

    const fn has_reference(self) -> bool {
        matches!(self, Self::Gram)
    }
}

/// How the design is stored.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum StorageLayout {
    /// Every entry is stored, column-major.
    Dense,
    /// Every entry is stored, row-major.
    DenseRowMajor,
    /// About `nonzeros_per_column` entries per column are nonzero.
    Sparse {
        /// Typical nonzero entries in one column.
        nonzeros_per_column: u32,
    },
}

/// Implementation that runs the operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Product {
    /// The scalar loop over column-major data. Needs no scratch and no workers.
    Reference,
    /// The `faer` kernel.
    Faer,
}

impl Product {
    /// Name used in `layout.toml`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Reference => "reference",
            Self::Faer => "faer",
        }
    }
}

/// What to run and how wide.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LayoutChoice {
    /// Implementation.
    pub product: Product,
    /// Workers the kernel may use, at least one and at most the caller's limit.
    pub workers: u32,
    /// Scratch bytes the choice needs, charged to the caller's scratch budget.
    pub scratch_bytes: u64,
}

/// One layout question under the caller's budget.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LayoutRequest {
    /// Operation.
    pub op: LayoutOp,
    /// Storage layout of the design.
    pub storage: StorageLayout,
    /// Rows of the design (the order, for an eigendecomposition).
    pub rows: usize,
    /// Columns of the design.
    pub cols: usize,
    /// Workers the caller can give this operation. Zero is refused.
    pub available_workers: u32,
    /// Scratch bytes the caller can lease for this operation.
    pub scratch_budget_bytes: u64,
}

/// Why no layout was chosen. Nothing was allocated.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum LayoutRefusal {
    /// The caller offered no workers.
    NoWorkers,
    /// A dimension is zero.
    EmptyShape,
    /// The cheapest configuration needs more scratch than the budget holds.
    ScratchOverBudget {
        /// Bytes the serial configuration needs.
        required_bytes: u64,
        /// Bytes the caller offered.
        budget_bytes: u64,
    },
}

impl core::fmt::Display for LayoutRefusal {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoWorkers => f.write_str("layout choice needs at least one worker"),
            Self::EmptyShape => f.write_str("layout choice needs a non-empty shape"),
            Self::ScratchOverBudget { required_bytes, budget_bytes } => write!(
                f,
                "serial factorization scratch needs {required_bytes} bytes, above the budget of \
                 {budget_bytes}; next attempt: raise the scratch budget"
            ),
        }
    }
}

impl std::error::Error for LayoutRefusal {}

/// One measured winner.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MeasuredPoint {
    pub(crate) op: LayoutOp,
    pub(crate) storage: StorageLayout,
    pub(crate) rows: u32,
    pub(crate) cols: u32,
    pub(crate) product: Product,
    pub(crate) workers: u32,
}

const fn dense(
    op: LayoutOp,
    rows: u32,
    cols: u32,
    product: Product,
    workers: u32,
) -> MeasuredPoint {
    MeasuredPoint { op, storage: StorageLayout::Dense, rows, cols, product, workers }
}

const fn dense_rowmajor(
    op: LayoutOp,
    rows: u32,
    cols: u32,
    product: Product,
    workers: u32,
) -> MeasuredPoint {
    MeasuredPoint { op, storage: StorageLayout::DenseRowMajor, rows, cols, product, workers }
}

const fn sparse(
    rows: u32,
    cols: u32,
    nonzeros_per_column: u32,
    product: Product,
    workers: u32,
) -> MeasuredPoint {
    MeasuredPoint {
        op: LayoutOp::Gram,
        storage: StorageLayout::Sparse { nonzeros_per_column },
        rows,
        cols,
        product,
        workers,
    }
}

// Winners of `parity/blas_layout/layout.toml`, one per measured shape. Generated from that
// record by `parity/blas_layout/derive_selections.py --write-rust`; do not edit by hand.
include!("layout_measured.rs");

fn work_of(rows: f64, cols: f64) -> f64 {
    rows * cols * cols
}

fn density_of(storage: StorageLayout, rows: f64) -> f64 {
    match storage {
        StorageLayout::Dense | StorageLayout::DenseRowMajor => 1.0,
        StorageLayout::Sparse { nonzeros_per_column } => {
            (f64::from(nonzeros_per_column.max(1)) / rows.max(1.0)).min(1.0)
        }
    }
}

fn same_storage(a: StorageLayout, b: StorageLayout) -> bool {
    matches!(
        (a, b),
        (StorageLayout::Dense, StorageLayout::Dense)
            | (StorageLayout::DenseRowMajor, StorageLayout::DenseRowMajor)
            | (StorageLayout::Sparse { .. }, StorageLayout::Sparse { .. })
    )
}

/// Measured winner nearest `request` in log work and, for sparse storage, log density.
fn nearest_measured(request: &LayoutRequest) -> Option<&'static MeasuredPoint> {
    let work = work_of(request.rows as f64, request.cols as f64).ln();
    let density = density_of(request.storage, request.rows as f64).ln();
    let mut best: Option<(&'static MeasuredPoint, f64)> = None;
    for point in MEASURED {
        if point.op != request.op || !same_storage(point.storage, request.storage) {
            continue;
        }
        let point_work = work_of(f64::from(point.rows), f64::from(point.cols)).ln();
        let point_density = density_of(point.storage, f64::from(point.rows)).ln();
        let distance = (work - point_work).abs() + (density - point_density).abs();
        // A tie keeps the earlier entry, which is the smaller measured shape.
        if best.is_none_or(|(_, kept)| distance < kept) {
            best = Some((point, distance));
        }
    }
    best.map(|(point, _)| point)
}

/// Scratch bytes `op` needs for a `rows × cols` design at `workers` workers.
///
/// The reference loop and the Gram product need none. The factorizations report the scratch
/// `faer` itself requires; a layout `faer` cannot express is `u64::MAX`, which no budget holds.
#[must_use]
pub fn scratch_requirement_bytes(op: LayoutOp, rows: usize, cols: usize, workers: u32) -> u64 {
    let par = FaerScope::try_new(workers.max(1)).map_or(Par::Seq, FaerScope::parallelism);
    let required: StackReq = match op {
        LayoutOp::Gram => return 0,
        LayoutOp::ColumnPivotedQr => {
            let block = recommended_block_size::<f64>(rows, cols);
            StackReq::and(
                qr_in_place_scratch::<usize, f64>(rows, cols, block, par, Spec::default()),
                solve_lstsq_in_place_scratch::<usize, f64>(rows, cols, block, 1, par),
            )
        }
        LayoutOp::ThinSvd => svd_scratch::<f64>(
            rows,
            cols,
            ComputeSvdVectors::Thin,
            ComputeSvdVectors::Thin,
            par,
            Spec::default(),
        ),
        LayoutOp::SymmetricEigen => {
            self_adjoint_evd_scratch::<f64>(rows, ComputeEigenvectors::Yes, par, Spec::default())
        }
    };
    if required.align_bytes() == 0 {
        return u64::MAX;
    }
    u64::try_from(required.size_bytes()).unwrap_or(u64::MAX)
}

/// Choose the product and worker count for `request` under its budget.
///
/// # Errors
///
/// [`LayoutRefusal::NoWorkers`] and [`LayoutRefusal::EmptyShape`] for an empty request, and
/// [`LayoutRefusal::ScratchOverBudget`] when even one worker needs more scratch than the
/// budget holds and no scratch-free reference product exists.
pub fn select_layout(request: &LayoutRequest) -> Result<LayoutChoice, LayoutRefusal> {
    if request.available_workers == 0 {
        return Err(LayoutRefusal::NoWorkers);
    }
    if request.rows == 0 || request.cols == 0 {
        return Err(LayoutRefusal::EmptyShape);
    }
    let (product, mut workers) = match nearest_measured(request) {
        Some(point) => (point.product, point.workers.min(request.available_workers)),
        None if request.op.has_reference() => (Product::Reference, 1),
        None => (Product::Faer, 1),
    };
    if product == Product::Reference {
        return Ok(LayoutChoice { product, workers: 1, scratch_bytes: 0 });
    }
    loop {
        let required = scratch_requirement_bytes(request.op, request.rows, request.cols, workers);
        if required <= request.scratch_budget_bytes {
            return Ok(LayoutChoice { product, workers, scratch_bytes: required });
        }
        if workers > 1 {
            workers = workers.div_ceil(2);
            continue;
        }
        if request.op.has_reference() {
            return Ok(LayoutChoice { product: Product::Reference, workers: 1, scratch_bytes: 0 });
        }
        return Err(LayoutRefusal::ScratchOverBudget {
            required_bytes: required,
            budget_bytes: request.scratch_budget_bytes,
        });
    }
}

/// Fill the symmetric row-major `ncols × ncols` Gram of a column-major design with the
/// measured product and at most `available_workers` workers, and return what ran.
///
/// The answer equals [`crate::gram::form_xtx`] up to summation order. It is exactly
/// symmetric. The Gram product needs no scratch, so the scratch budget cannot refuse it.
///
/// # Panics
///
/// When `x_colmajor` is shorter than `nrows * ncols` or `xtx` shorter than `ncols * ncols`.
pub fn form_xtx_measured(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    xtx: &mut [f64],
    available_workers: u32,
) -> LayoutChoice {
    let serial_reference =
        LayoutChoice { product: Product::Reference, workers: 1, scratch_bytes: 0 };
    let request = LayoutRequest {
        op: LayoutOp::Gram,
        storage: StorageLayout::Dense,
        rows: nrows,
        cols: ncols,
        available_workers: available_workers.max(1),
        scratch_budget_bytes: 0,
    };
    let choice = select_layout(&request).unwrap_or(serial_reference);
    if choice.product == Product::Reference {
        xtx[..ncols * ncols].fill(0.0);
        crate::gram::accumulate_xtx(x_colmajor, nrows, ncols, xtx);
        return choice;
    }
    let par = FaerScope::try_new(choice.workers).map_or(Par::Seq, FaerScope::parallelism);
    let x = faer::MatRef::from_column_major_slice(&x_colmajor[..nrows * ncols], nrows, ncols);
    let out = faer::MatMut::from_row_major_slice_mut(&mut xtx[..ncols * ncols], ncols, ncols);
    faer::linalg::matmul::matmul(out, faer::Accum::Replace, x.transpose(), x, 1.0, par);
    for i in 0..ncols {
        for j in i + 1..ncols {
            xtx[j * ncols + i] = xtx[i * ncols + j];
        }
    }
    choice
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense_request(op: LayoutOp, rows: usize, cols: usize, workers: u32) -> LayoutRequest {
        LayoutRequest {
            op,
            storage: StorageLayout::Dense,
            rows,
            cols,
            available_workers: workers,
            scratch_budget_bytes: u64::MAX,
        }
    }

    fn pick(request: &LayoutRequest) -> (Product, u32) {
        let choice = select_layout(request).unwrap();
        (choice.product, choice.workers)
    }

    #[test]
    fn measured_shapes_select_their_measured_winner_and_cap_at_the_callers_workers() {
        // Every measured dense shape selects its recorded winner when the caller has at least
        // its workers, never more than the caller has, and runs serial for a single worker.
        let mut checked = 0;
        for point in MEASURED {
            if !matches!(point.storage, StorageLayout::Dense) {
                continue;
            }
            let (rows, cols) = (point.rows as usize, point.cols as usize);
            let rich = pick(&dense_request(point.op, rows, cols, 64));
            assert_eq!(rich, (point.product, point.workers), "{:?} {rows}x{cols}", point.op);
            let capped = pick(&dense_request(point.op, rows, cols, 2));
            assert_eq!(capped.0, point.product, "{:?} {rows}x{cols}", point.op);
            assert_eq!(capped.1, point.workers.min(2), "{:?} {rows}x{cols}", point.op);
            let serial = pick(&dense_request(point.op, rows, cols, 1));
            assert_eq!(serial.1, 1, "{:?} {rows}x{cols}", point.op);
            checked += 1;
        }
        assert!(checked >= 10, "the record measures dense shapes of every operation");
        for point in MEASURED {
            if let StorageLayout::Sparse { nonzeros_per_column } = point.storage {
                let request = LayoutRequest {
                    storage: StorageLayout::Sparse { nonzeros_per_column },
                    ..dense_request(point.op, point.rows as usize, point.cols as usize, 64)
                };
                assert_eq!(
                    pick(&request),
                    (point.product, point.workers),
                    "{:?} sparse {}x{}",
                    point.op,
                    point.rows,
                    point.cols
                );
            }
        }
    }

    #[test]
    fn unmeasured_shapes_follow_the_nearest_measurement_and_unmeasured_operations_run_serial() {
        // Between the measured small and large dense Gram the nearer one in log work decides.
        let near_small = pick(&dense_request(LayoutOp::Gram, 128, 8, 64));
        let near_large = pick(&dense_request(LayoutOp::Gram, 8192, 128, 64));
        let nearest = |rows, cols| {
            let point = nearest_measured(&dense_request(LayoutOp::Gram, rows, cols, 64))
                .expect("a dense Gram is measured");
            (point.product, point.workers)
        };
        assert_eq!(near_small, nearest(128, 8));
        assert_eq!(near_large, nearest(8192, 128));
        assert_eq!(pick(&dense_request(LayoutOp::ThinSvd, 512, 64, 16)), (Product::Faer, 1));
        assert_eq!(pick(&dense_request(LayoutOp::SymmetricEigen, 64, 64, 16)), (Product::Faer, 1));
        let sparse_unmeasured = LayoutRequest {
            storage: StorageLayout::Sparse { nonzeros_per_column: 3 },
            ..dense_request(LayoutOp::ColumnPivotedQr, 100, 10, 16)
        };
        assert_eq!(pick(&sparse_unmeasured), (Product::Faer, 1));
    }

    #[test]
    fn scratch_budget_narrows_the_scope_or_refuses_before_any_allocation() {
        let need = |op, rows, cols, workers| scratch_requirement_bytes(op, rows, cols, workers);
        assert_eq!(need(LayoutOp::Gram, 4096, 128, 16), 0);
        let serial = need(LayoutOp::ThinSvd, 512, 64, 1);
        assert!(serial > 0 && serial < u64::MAX);
        let tight = LayoutRequest {
            scratch_budget_bytes: serial - 1,
            ..dense_request(LayoutOp::ThinSvd, 512, 64, 16)
        };
        assert_eq!(
            select_layout(&tight),
            Err(LayoutRefusal::ScratchOverBudget {
                required_bytes: serial,
                budget_bytes: serial - 1
            })
        );
        let exact = LayoutRequest { scratch_budget_bytes: serial, ..tight };
        let choice = select_layout(&exact).unwrap();
        assert_eq!((choice.workers, choice.scratch_bytes), (1, serial));
        // A parallel measurement is narrowed until its own scratch fits.
        let wide = need(LayoutOp::ColumnPivotedQr, 32, 8, 10);
        let narrow = need(LayoutOp::ColumnPivotedQr, 32, 8, 1);
        let budget = narrow.max(wide.saturating_sub(1));
        let request = LayoutRequest {
            scratch_budget_bytes: budget,
            ..dense_request(LayoutOp::ColumnPivotedQr, 32, 8, 10)
        };
        let choice = select_layout(&request).unwrap();
        assert!(choice.scratch_bytes <= budget);
        if wide > budget {
            assert!(choice.workers < 10);
        }
        assert_eq!(
            select_layout(&LayoutRequest { available_workers: 0, ..exact }),
            Err(LayoutRefusal::NoWorkers)
        );
        assert_eq!(
            select_layout(&LayoutRequest { rows: 0, ..exact }),
            Err(LayoutRefusal::EmptyShape)
        );
    }

    fn scalar_gram(x: &[f64], n: usize, p: usize) -> Vec<f64> {
        let mut out = vec![0.0; p * p];
        for i in 0..p {
            for j in 0..p {
                out[i * p + j] = (0..n).map(|r| x[i * n + r] * x[j * n + r]).sum();
            }
        }
        out
    }

    #[test]
    fn measured_gram_matches_the_scalar_gram_at_every_worker_count_and_is_symmetric() {
        for (n, p) in [(64usize, 8usize), (300, 20), (1024, 40), (4096, 130)] {
            let x: Vec<f64> = (0..n * p)
                .map(|i| (((i * 2_654_435_761) % 1_000_003) as f64) / 1_000_003.0 - 0.5)
                .collect();
            let expected = scalar_gram(&x, n, p);
            let scale = expected.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
            for workers in [1u32, 4, 16] {
                let mut actual = vec![f64::NAN; p * p];
                let choice = form_xtx_measured(&x, n, p, &mut actual, workers);
                assert!(choice.workers <= workers);
                for (a, e) in actual.iter().zip(&expected) {
                    assert!((a - e).abs() <= 1e-8 * (1.0 + scale), "{n}x{p} w{workers}");
                }
                for i in 0..p {
                    for j in 0..p {
                        assert_eq!(actual[i * p + j].to_bits(), actual[j * p + i].to_bits());
                    }
                }
            }
        }
    }

    /// Every `[[selection]]` of the measurement record is a row of the rule table, and the
    /// table has no row the record does not hold.
    #[test]
    fn rule_table_equals_the_selections_of_the_measurement_record() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../parity/blas_layout/layout.toml");
        let text = std::fs::read_to_string(path).expect("layout record is in the repository");
        let mut recorded: Vec<std::collections::BTreeMap<String, String>> = Vec::new();
        let mut in_selection = false;
        for line in text.lines() {
            let line = line.trim();
            if line.starts_with("[[") {
                in_selection = line == "[[selection]]";
                if in_selection {
                    recorded.push(std::collections::BTreeMap::new());
                }
                continue;
            }
            if !in_selection {
                continue;
            }
            if let Some((key, value)) = line.split_once('=') {
                if let Some(block) = recorded.last_mut() {
                    block
                        .insert(key.trim().to_string(), value.trim().trim_matches('"').to_string());
                }
            }
        }
        assert_eq!(recorded.len(), MEASURED.len());
        for point in MEASURED {
            let (layout, nonzeros) = match point.storage {
                StorageLayout::Dense => ("dense", None),
                StorageLayout::DenseRowMajor => ("dense_rowmajor", None),
                StorageLayout::Sparse { nonzeros_per_column } => {
                    ("sparse", Some(nonzeros_per_column.to_string()))
                }
            };
            let found = recorded.iter().any(|block| {
                block.get("op").map(String::as_str) == Some(point.op.as_str())
                    && block.get("layout").map(String::as_str) == Some(layout)
                    && block.get("rows") == Some(&point.rows.to_string())
                    && block.get("cols") == Some(&point.cols.to_string())
                    && block.get("nonzeros_per_column") == nonzeros.as_ref()
                    && block.get("product").map(String::as_str) == Some(point.product.as_str())
                    && block.get("workers") == Some(&point.workers.to_string())
            });
            assert!(found, "no selection for {point:?}");
        }
    }
}
