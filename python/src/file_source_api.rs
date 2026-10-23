//! Python binding for file-backed tables (`antecedent.data.from_parquet` and
//! `from_arrow_ipc`): open a Parquet file, a directory of Parquet files, or an
//! Arrow IPC file or stream; read projected columns in bounded chunks under the
//! admission pool; compute the streaming content identity; and stream the
//! weighted Gaussian capacity summary.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt::Write as _;
use std::path::PathBuf;

use antecedent::analysis::file_table::{
    CapacityControl, CapacityOptions, FileColumnKind, FileSourceError, FileSourceErrorKind,
    FileTableError, GaussianSummarySpec, ReadOptions, ReadPlan, SummaryMissing, TableFile,
    TableFileFormat, WeightedCapacityError, WeightedCapacityOutput, summarize_weighted_gaussian,
};
use antecedent_core::execution_facts::OutputRetention;
use antecedent_data::OwnedColumn;
use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::transport_common::execution_context;

fn with_remedy(err: PyErr, remedy: Option<String>) -> PyErr {
    if let Some(remedy) = remedy {
        Python::attach(|py| {
            let _ = err.value(py).setattr("remedy", remedy);
        });
    }
    err
}

fn source_err(error: &FileSourceError) -> PyErr {
    let message = error.to_string();
    let err = match error.kind() {
        FileSourceErrorKind::Data => crate::CausalDataError::new_err(message),
        FileSourceErrorKind::Resource => crate::CausalResourceError::new_err(message),
        FileSourceErrorKind::Cancelled => crate::CausalCancelledError::new_err(message),
        _ => crate::value_err_code("invalid_argument", message),
    };
    with_remedy(err, error.next_attempt())
}

fn table_err(error: &FileTableError) -> PyErr {
    match error {
        FileTableError::Source(source) => source_err(source),
        FileTableError::Capacity(WeightedCapacityError::Cancelled) => {
            crate::CausalCancelledError::new_err(error.to_string())
        }
        FileTableError::Capacity(WeightedCapacityError::OverCeiling { .. }) => with_remedy(
            crate::CausalResourceError::new_err(error.to_string()),
            Some("raise ceiling_bytes, or lower batch_rows".to_owned()),
        ),
        _ => crate::CausalEstimateError::new_err(error.to_string()),
    }
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

fn kind_fields(kind: &FileColumnKind) -> (&'static str, Option<String>, bool) {
    match kind {
        FileColumnKind::Timestamp { timezone } => {
            ("timestamp", timezone.as_ref().map(ToString::to_string), false)
        }
        FileColumnKind::Categorical { ordered } => ("categorical", None, *ordered),
        other => (other.as_str(), None, false),
    }
}

/// `(name, kind, nullable, arrow_type, timezone, ordered)` of one file column.
type ColumnInfo = (String, String, bool, String, Option<String>, bool);

/// Values of one read column, held without the GIL.
enum Values {
    Float(Vec<f64>),
    Int(Vec<i64>),
    Bool(Vec<bool>),
    Time(Vec<i64>, Option<String>),
    Codes(Vec<u32>, Vec<String>, bool),
}

struct ReadColumn {
    name: String,
    values: Values,
    valid: Option<Vec<bool>>,
}

struct ReadResult {
    rows: u64,
    chunks: u64,
    identity: String,
    snapshot_digest: String,
    peak_chunk_bytes: u64,
    columns: Vec<ReadColumn>,
}

fn plan(
    table: &TableFile,
    columns: Option<Vec<String>>,
    chunk_rows: Option<usize>,
) -> Result<ReadPlan, FileSourceError> {
    let options = chunk_rows.map_or_else(ReadOptions::default, ReadOptions::with_chunk_rows);
    match columns {
        Some(names) => table.plan_columns(&names, options),
        None => table.plan_all(options),
    }
}

fn read_columns(
    table: &TableFile,
    plan: &ReadPlan,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<ReadResult, FileSourceError> {
    let materialized = table.materialize(plan, ctx)?;
    let receipt = materialized.receipt().clone();
    let storage = materialized.data().storage();
    let mut columns = Vec::with_capacity(plan.columns().len());
    for (column, file_column) in storage.columns().iter().zip(plan.columns()) {
        let validity = column.as_view().validity();
        let valid = (!validity.is_all_valid())
            .then(|| (0..validity.len()).map(|row| validity.is_valid(row)).collect());
        let values = match column {
            OwnedColumn::Float64(column) => Values::Float(column.values.as_slice().to_vec()),
            OwnedColumn::Int64(column) => Values::Int(column.values.to_vec()),
            OwnedColumn::Boolean(column) => {
                Values::Bool(column.values.iter().map(|&value| value != 0).collect())
            }
            OwnedColumn::Timestamp(column) => {
                let (_, timezone, _) = kind_fields(&file_column.kind);
                Values::Time(column.values_ns.to_vec(), timezone)
            }
            OwnedColumn::Categorical(column) => Values::Codes(
                column.codes.iter().map(|code| code.raw()).collect(),
                column.domain.levels.iter().map(|level| level.label.to_string()).collect(),
                column.domain.ordered,
            ),
            OwnedColumn::FixedVector(_) => {
                return Err(FileSourceError::InvalidOptions {
                    detail: format!("column '{}' is a vector", file_column.name),
                });
            }
        };
        columns.push(ReadColumn { name: file_column.name.to_string(), values, valid });
    }
    Ok(ReadResult {
        rows: receipt.rows().get(),
        chunks: receipt.chunks(),
        identity: receipt.identity().to_hex(),
        snapshot_digest: hex(&materialized.snapshot_digest()),
        peak_chunk_bytes: receipt.peak_chunk_bytes(),
        columns,
    })
}

/// A Parquet file, a directory of Parquet files, or an Arrow IPC file or stream.
///
/// Opening reads the schema and footer only. Every read is planned from the
/// named columns, streams in bounded chunks charged to the admission pool, and
/// computes the full-content identity as it goes.
#[pyclass(name = "NativeTableFile", module = "antecedent._native", frozen, skip_from_py_object)]
pub(crate) struct PyTableFile {
    inner: TableFile,
}

#[pymethods]
impl PyTableFile {
    /// Open `path`. `format` is `parquet`, `arrow_ipc_file`, or `arrow_ipc_stream`;
    /// omitted, it is inferred from the extension (a directory is a Parquet dataset).
    #[staticmethod]
    #[pyo3(signature = (path, format=None))]
    fn open(py: Python<'_>, path: PathBuf, format: Option<String>) -> PyResult<Self> {
        let format = match format.as_deref() {
            None => None,
            Some(name) => Some(TableFileFormat::from_name(name).ok_or_else(|| {
                crate::value_err_code(
                    "invalid_argument",
                    format!(
                        "format must be parquet, arrow_ipc_file, or arrow_ipc_stream; got {name:?}"
                    ),
                )
            })?),
        };
        let inner = crate::detach_catch(py, move || {
            match format {
                None => TableFile::open(&path),
                Some(format) => TableFile::open_as(&path, format),
            }
            .map_err(|error| source_err(&error))
        })?;
        Ok(Self { inner })
    }

    /// Format name.
    #[getter]
    fn format(&self) -> &'static str {
        self.inner.format().as_str()
    }

    /// Files read, in order.
    #[getter]
    fn paths(&self) -> Vec<String> {
        self.inner.paths().iter().map(|path| path.display().to_string()).collect()
    }

    /// Row count the footers record, or `None` (Arrow IPC).
    #[getter]
    fn declared_rows(&self) -> Option<u64> {
        self.inner.declared_rows().map(antecedent_data::GlobalId::get)
    }

    /// `(name, kind, nullable, arrow_type, timezone, ordered)` for every column.
    fn columns(&self) -> Vec<ColumnInfo> {
        self.inner
            .columns()
            .iter()
            .map(|column| {
                let (kind, timezone, ordered) = kind_fields(&column.kind);
                (
                    column.name.to_string(),
                    kind.to_owned(),
                    column.nullable,
                    column.arrow_type.to_string(),
                    timezone,
                    ordered,
                )
            })
            .collect()
    }

    /// `(total_bytes or None, resident_bytes or None, chunk_bytes, chunk_rows)` of a read
    /// of `columns` (all readable columns when `None`).
    #[pyo3(signature = (columns=None, *, chunk_rows=None))]
    fn quote(
        &self,
        columns: Option<Vec<String>>,
        chunk_rows: Option<usize>,
    ) -> PyResult<(Option<u64>, Option<u64>, u64, usize)> {
        let plan = plan(&self.inner, columns, chunk_rows).map_err(|error| source_err(&error))?;
        let chunk = plan.chunk_quote().map_err(|error| source_err(&error))?;
        let quote = self.inner.quote(&plan).map_err(|error| source_err(&error))?;
        Ok((
            quote.map(|quote| quote.total_bytes()),
            quote.map(|quote| quote.resident_bytes()),
            chunk.total_bytes(),
            plan.chunk_rows(),
        ))
    }

    /// Read `columns` (all readable columns when `None`) into one resident table.
    ///
    /// Returns a dict with `rows`, `chunks`, `identity`, `snapshot_digest`,
    /// `peak_chunk_bytes`, and `columns`: one dict per column with `name`,
    /// `kind`, `values` (float64, int64, bool, int64 nanoseconds, or uint32
    /// codes), `valid` (bool mask or `None` when every cell is valid), and for
    /// categories `labels` and `ordered`, for timestamps `timezone`.
    #[pyo3(signature = (columns=None, *, chunk_rows=None, memory_limit_bytes=None, cancel=None))]
    fn read(
        &self,
        py: Python<'_>,
        columns: Option<Vec<String>>,
        chunk_rows: Option<usize>,
        memory_limit_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Py<PyDict>> {
        let plan = plan(&self.inner, columns, chunk_rows).map_err(|error| source_err(&error))?;
        let ctx = execution_context(0, memory_limit_bytes, cancel);
        let table = &self.inner;
        let read = crate::detach_catch(py, move || {
            read_columns(table, &plan, &ctx).map_err(|error| source_err(&error))
        })?;
        let out = PyDict::new(py);
        out.set_item("rows", read.rows)?;
        out.set_item("chunks", read.chunks)?;
        out.set_item("identity", read.identity)?;
        out.set_item("snapshot_digest", read.snapshot_digest)?;
        out.set_item("peak_chunk_bytes", read.peak_chunk_bytes)?;
        let columns = PyList::empty(py);
        for column in read.columns {
            let entry = PyDict::new(py);
            entry.set_item("name", column.name)?;
            match column.values {
                Values::Float(values) => {
                    entry.set_item("kind", "float64")?;
                    entry.set_item("values", PyArray1::from_vec(py, values))?;
                }
                Values::Int(values) => {
                    entry.set_item("kind", "int64")?;
                    entry.set_item("values", PyArray1::from_vec(py, values))?;
                }
                Values::Bool(values) => {
                    entry.set_item("kind", "boolean")?;
                    entry.set_item("values", PyArray1::from_vec(py, values))?;
                }
                Values::Time(values, timezone) => {
                    entry.set_item("kind", "timestamp")?;
                    entry.set_item("values", PyArray1::from_vec(py, values))?;
                    entry.set_item("timezone", timezone)?;
                }
                Values::Codes(codes, labels, ordered) => {
                    entry.set_item("kind", "categorical")?;
                    entry.set_item("values", PyArray1::from_vec(py, codes))?;
                    entry.set_item("labels", labels)?;
                    entry.set_item("ordered", ordered)?;
                }
            }
            match column.valid {
                Some(valid) => entry.set_item("valid", PyArray1::from_vec(py, valid))?,
                None => entry.set_item("valid", py.None())?,
            }
            columns.append(entry)?;
        }
        out.set_item("columns", columns)?;
        Ok(out.unbind())
    }

    /// Stream `columns` (all readable columns when `None`) once and return
    /// `(identity_hex, rows, chunks)`. Nothing is kept resident.
    #[pyo3(signature = (columns=None, *, chunk_rows=None, memory_limit_bytes=None, cancel=None))]
    fn content_identity(
        &self,
        py: Python<'_>,
        columns: Option<Vec<String>>,
        chunk_rows: Option<usize>,
        memory_limit_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<(String, u64, u64)> {
        let plan = plan(&self.inner, columns, chunk_rows).map_err(|error| source_err(&error))?;
        let ctx = execution_context(0, memory_limit_bytes, cancel);
        let table = &self.inner;
        let receipt = crate::detach_catch(py, move || {
            table.scan_identity(&plan, &ctx).map_err(|error| source_err(&error))
        })?;
        Ok((receipt.identity().to_hex(), receipt.rows().get(), receipt.chunks()))
    }

    /// Stream the weighted Gaussian / OLS capacity summary of `response` on
    /// `covariates` (with an intercept first when `intercept`), weighted by
    /// `weight` when given. Only one chunk of `batch_rows` rows is resident at a
    /// time. `missing` is `refuse` or `complete_case`; `control` is `glm` or
    /// `gaussian_bayes`.
    #[pyo3(signature = (
        covariates,
        response,
        *,
        weight=None,
        intercept=true,
        missing="refuse",
        batch_rows=65_536,
        ceiling_bytes=None,
        control="glm",
        memory_limit_bytes=None,
        cancel=None,
    ))]
    fn summarize_weighted_gaussian(
        &self,
        py: Python<'_>,
        covariates: Vec<String>,
        response: String,
        weight: Option<String>,
        intercept: bool,
        missing: &str,
        batch_rows: usize,
        ceiling_bytes: Option<u64>,
        control: &str,
        memory_limit_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Py<PyDict>> {
        let missing = SummaryMissing::from_name(missing).ok_or_else(|| {
            crate::value_err_code(
                "invalid_argument",
                format!("missing must be 'refuse' or 'complete_case'; got {missing:?}"),
            )
        })?;
        let control = match control {
            "glm" => CapacityControl::Glm,
            "gaussian_bayes" => CapacityControl::GaussianBayes,
            other => {
                return Err(crate::value_err_code(
                    "invalid_argument",
                    format!("control must be 'glm' or 'gaussian_bayes'; got {other:?}"),
                ));
            }
        };
        if batch_rows == 0 {
            return Err(crate::value_err_code("invalid_argument", "batch_rows must be positive"));
        }
        let spec = GaussianSummarySpec { covariates, response, weight, intercept, missing };
        let ctx = execution_context(0, memory_limit_bytes, cancel);
        let mut options = CapacityOptions::new(
            batch_rows,
            ceiling_bytes.unwrap_or(u64::MAX),
            OutputRetention::SummaryOnly,
        );
        options.cancellation = Some(ctx.cancellation.clone());
        let table = &self.inner;
        let summary = crate::detach_catch(py, move || {
            summarize_weighted_gaussian(table, &spec, control, &options, &ctx)
                .map_err(|error| table_err(&error))
        })?;
        let out = PyDict::new(py);
        out.set_item("design_columns", summary.design_columns)?;
        out.set_item("rows_read", summary.rows_read)?;
        out.set_item("rows_used", summary.rows_used)?;
        out.set_item("rows_dropped", summary.rows_dropped)?;
        out.set_item("file_identity", summary.file_identity.to_hex())?;
        let WeightedCapacityOutput::Resident(result) = summary.output else {
            return Err(crate::CausalEstimateError::new_err(
                "the summary-only capacity route returned a spill",
            ));
        };
        out.set_item("coefficients", result.coefficients)?;
        out.set_item("robust_covariance", result.robust_covariance)?;
        out.set_item("n_eff", result.cross_products.n_eff)?;
        out.set_item("xtwx", result.cross_products.xtwx)?;
        out.set_item("xtwy", result.cross_products.xtwy)?;
        out.set_item("ytwy", result.cross_products.ytwy)?;
        out.set_item("weighted_rss", result.residual.weighted_rss)?;
        out.set_item("max_abs_residual", result.residual.max_abs_residual)?;
        out.set_item("leverage_max", result.leverage.max)?;
        out.set_item("batches", result.batches)?;
        out.set_item("source_passes", result.source_passes)?;
        out.set_item("route_identity", hex(&result.content_identity))?;
        out.set_item("ceiling_total_bytes", result.charges.total_bytes)?;
        Ok(out.unbind())
    }

    fn __repr__(&self) -> String {
        format!(
            "NativeTableFile(format={}, files={}, columns={})",
            self.inner.format(),
            self.inner.paths().len(),
            self.inner.columns().len()
        )
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyTableFile>()
}
