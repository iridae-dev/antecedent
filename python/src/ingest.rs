//! NumPy ingestion: columns of any numeric dtype owned once as library tables, panel windows and the conversion probes.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[allow(
    clippy::wildcard_imports,
    reason = "the ingestion bindings share the native PyO3 prelude"
)]
use crate::*;

/// Result of the conversion probe (same Arrow→tabular path as analyze/discover).
#[pyclass]
pub(crate) struct ArrowLoadInfo {
    #[pyo3(get)]
    row_count: usize,
    #[pyo3(get)]
    column_count: usize,
    #[pyo3(get)]
    bytes_copied: u64,
    #[pyo3(get)]
    bytes_borrowed: u64,
    #[pyo3(get)]
    diagnostic_count: usize,
    /// Schema names after library-owned ingestion (proves the batch was parsed).
    #[pyo3(get)]
    column_names: Vec<String>,
}

/// NumPy columns owned once.
///
/// Each column is copied a single time, into the shared slice the table later keeps, so
/// there is no intermediate Arrow batch and no second copy into library storage. A NumPy
/// buffer is writable by Python, so it is not borrowed past the call.
#[derive(Clone, Debug)]
pub(crate) struct ColumnBatch {
    names: Vec<String>,
    columns: Vec<antecedent::data::F64Buffer>,
    receipt: NumpyIngestReceipt,
}

impl ColumnBatch {
    /// Rows in every column.
    pub(crate) fn num_rows(&self) -> usize {
        self.columns.first().map_or(0, antecedent::data::F64Buffer::len)
    }

    /// Bytes the owned columns hold.
    pub(crate) fn get_array_memory_size(&self) -> usize {
        self.columns.iter().map(|column| std::mem::size_of_val(&column[..])).sum()
    }

    /// Copies the ingestion made.
    pub(crate) fn receipt(&self) -> NumpyIngestReceipt {
        self.receipt
    }

    /// The table over the owned columns. The buffers are shared, not copied again.
    pub(crate) fn loaded(&self) -> PyResult<NumpyTable> {
        // The same schema the Arrow route builds: continuous columns with the context role hint.
        let mut builder = antecedent::prelude::CausalSchemaBuilder::new();
        for name in &self.names {
            builder
                .add_variable(
                    Arc::<str>::from(name.as_str()),
                    antecedent::schema::ValueType::Continuous,
                    antecedent::schema::SmallRoleSet::from_hint(
                        antecedent::schema::RoleHint::Context,
                    ),
                    None,
                    None,
                    antecedent::schema::MeasurementSpec::default(),
                )
                .map_err(|e| py_err(antecedent::data::DataError::from(e)))?;
        }
        let schema = builder.build().map_err(|e| py_err(antecedent::data::DataError::from(e)))?;
        let rows = self.num_rows();
        let mut columns = Vec::with_capacity(self.columns.len());
        for (index, values) in self.columns.iter().enumerate() {
            let id = antecedent::prelude::VariableId::from_raw(
                u32::try_from(index).map_err(|_| value_err("too many columns"))?,
            );
            let column = antecedent::data::Float64Column::new(
                id,
                values.clone(),
                antecedent::data::ValidityBitmap::all_valid(rows),
            )
            .map_err(py_err)?;
            columns.push(antecedent::data::OwnedColumn::Float64(column));
        }
        let storage = antecedent::data::OwnedColumnarStorage::try_new(schema, columns, None, None)
            .map_err(py_err)?;
        Ok(NumpyTable {
            data: antecedent::prelude::TabularData::new(storage),
            receipt: self.receipt,
        })
    }
}

/// Copies one NumPy ingestion made: one owned buffer per column.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct NumpyIngestReceipt {
    /// Owned numeric buffers made.
    pub(crate) owned_copies: u64,
    /// Bytes copied into those buffers.
    pub(crate) bytes_copied: u64,
    /// Columns that were not contiguous (gathered in the same single pass).
    pub(crate) noncontiguous_columns: u64,
}

/// The table a NumPy ingestion produced, with the copies it made.
pub(crate) struct NumpyTable {
    pub(crate) data: antecedent::prelude::TabularData,
    pub(crate) receipt: NumpyIngestReceipt,
}

/// Largest integer magnitude a float64 holds exactly.
pub(crate) const EXACT_INT_LIMIT: i64 = 1 << 53;

/// A NumPy column of any numeric dtype the ingestion widens to float64 exactly.
///
/// It is the argument type of every native entry point that takes table columns, so a caller
/// passes float, integer or boolean arrays alike. A float64 column is copied once into the
/// table's own buffer (or, when it is a view of a retained [`data_snapshot`] column, not copied
/// at all). An integer, boolean or float32 column is converted while it is copied, so widening
/// and ownership are one pass and one allocation: nothing is materialized as float64 in Python
/// first. A strided column is gathered in the same single pass. Integers beyond 2**53 do not
/// survive the widening and are refused.
pub(crate) enum NumericColumn<'py> {
    F64(PyReadonlyArray1<'py, f64>),
    F32(PyReadonlyArray1<'py, f32>),
    I64(PyReadonlyArray1<'py, i64>),
    I32(PyReadonlyArray1<'py, i32>),
    I16(PyReadonlyArray1<'py, i16>),
    I8(PyReadonlyArray1<'py, i8>),
    U64(PyReadonlyArray1<'py, u64>),
    U32(PyReadonlyArray1<'py, u32>),
    U16(PyReadonlyArray1<'py, u16>),
    U8(PyReadonlyArray1<'py, u8>),
    Bool(PyReadonlyArray1<'py, bool>),
}

impl<'a, 'py> FromPyObject<'a, 'py> for NumericColumn<'py> {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        Self::from_object(&obj)
    }
}

impl<'py> NumericColumn<'py> {
    /// The column behind `obj`, by its dtype.
    pub(crate) fn from_object(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        macro_rules! try_dtype {
            ($($variant:ident($ty:ty)),* $(,)?) => {
                $(if let Ok(array) = obj.extract::<PyReadonlyArray1<'py, $ty>>() {
                    return Ok(Self::$variant(array));
                })*
            };
        }
        try_dtype!(
            F64(f64),
            I64(i64),
            F32(f32),
            I32(i32),
            I16(i16),
            I8(i8),
            U64(u64),
            U32(u32),
            U16(u16),
            U8(u8),
            Bool(bool),
        );
        Err(value_err("a column must be a one-dimensional float, integer or boolean NumPy array"))
    }

    /// Rows in the column.
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::F64(a) => a.as_array().len(),
            Self::F32(a) => a.as_array().len(),
            Self::I64(a) => a.as_array().len(),
            Self::I32(a) => a.as_array().len(),
            Self::I16(a) => a.as_array().len(),
            Self::I8(a) => a.as_array().len(),
            Self::U64(a) => a.as_array().len(),
            Self::U32(a) => a.as_array().len(),
            Self::U16(a) => a.as_array().len(),
            Self::U8(a) => a.as_array().len(),
            Self::Bool(a) => a.as_array().len(),
        }
    }

    /// The column as a contiguous float64 slice, when it already is one (no copy needed to read
    /// it for the duration of the call).
    pub(crate) fn as_f64_slice(&self) -> Option<&[f64]> {
        match self {
            Self::F64(a) => a.as_slice().ok(),
            _ => None,
        }
    }

    /// The column's values as float64 for the duration of the call: borrowed when it is a
    /// contiguous float64 array, otherwise widened/gathered once.
    ///
    /// # Errors
    ///
    /// An integer column holding a value beyond 2**53.
    pub(crate) fn values(&self, name: &str) -> PyResult<std::borrow::Cow<'_, [f64]>> {
        match self.as_f64_slice() {
            Some(slice) => Ok(std::borrow::Cow::Borrowed(slice)),
            None => Ok(std::borrow::Cow::Owned(self.owned(name)?.0.to_vec())),
        }
    }

    /// The column's values as an owned float64 vector (one copy; widened when needed).
    ///
    /// # Errors
    ///
    /// An integer column holding a value beyond 2**53.
    pub(crate) fn to_vec(&self, name: &str) -> PyResult<Vec<f64>> {
        match self.as_f64_slice() {
            Some(slice) => Ok(slice.to_vec()),
            None => Ok(self.owned(name)?.0.to_vec()),
        }
    }

    /// The column's values as one owned float64 buffer, and whether the source was strided.
    ///
    /// # Errors
    ///
    /// An integer column holding a value beyond 2**53.
    pub(crate) fn owned(&self, name: &str) -> PyResult<(Arc<[f64]>, bool)> {
        self.owned_range(name, 0, self.len())
    }

    /// Rows `lo..hi` of the column as one owned float64 buffer (gathered in the same pass when
    /// the source is strided), and whether the source was strided.
    ///
    /// # Errors
    ///
    /// An integer column holding a value beyond 2**53 in the window.
    #[allow(
        clippy::cast_precision_loss,
        reason = "integers are range-checked to 2**53 before they are widened, so the cast is exact"
    )]
    pub(crate) fn owned_range(
        &self,
        name: &str,
        lo: usize,
        hi: usize,
    ) -> PyResult<(Arc<[f64]>, bool)> {
        macro_rules! widen {
            ($array:expr, $convert:expr) => {{
                let array = $array;
                let convert = $convert;
                match array.as_slice() {
                    Ok(slice) => {
                        (slice[lo..hi].iter().map(|v| convert(*v)).collect::<Arc<[f64]>>(), false)
                    }
                    // A strided column is gathered by index: the mapped range reports its exact
                    // length, so the shared slice is allocated and filled once.
                    Err(_) => {
                        let view = array.as_array();
                        ((lo..hi).map(|i| convert(view[i])).collect::<Arc<[f64]>>(), true)
                    }
                }
            }};
        }
        macro_rules! checked_int {
            ($array:expr, $ty:ty) => {{
                let array = $array;
                let window = array.as_array();
                let window = window.slice(numpy::ndarray::s![lo..hi]);
                if let (Some(low), Some(high)) = (window.iter().min(), window.iter().max()) {
                    if i128::from(*high) > i128::from(EXACT_INT_LIMIT)
                        || i128::from(*low) < -i128::from(EXACT_INT_LIMIT)
                    {
                        return Err(value_err(format!(
                            "integer column {name:?} spans [{low}, {high}], beyond 2**53, and \
                             would lose precision as float64"
                        )));
                    }
                }
                widen!(array, |v: $ty| v as f64)
            }};
        }
        Ok(match self {
            Self::F64(a) => widen!(a, |v: f64| v),
            Self::F32(a) => widen!(a, |v: f32| f64::from(v)),
            Self::I64(a) => checked_int!(a, i64),
            Self::I32(a) => widen!(a, |v: i32| f64::from(v)),
            Self::I16(a) => widen!(a, |v: i16| f64::from(v)),
            Self::I8(a) => widen!(a, |v: i8| f64::from(v)),
            Self::U64(a) => checked_int!(a, u64),
            Self::U32(a) => widen!(a, |v: u32| f64::from(v)),
            Self::U16(a) => widen!(a, |v: u16| f64::from(v)),
            Self::U8(a) => widen!(a, |v: u8| f64::from(v)),
            Self::Bool(a) => widen!(a, |v: bool| f64::from(u8::from(v))),
        })
    }
}

/// The shared length of `columns`, or the refusal naming the first column that differs.
fn common_length(names: &[String], columns: &[NumericColumn<'_>]) -> PyResult<usize> {
    if names.len() != columns.len() {
        return Err(value_err("names and columns must have the same length"));
    }
    let Some(first) = columns.first() else {
        return Err(value_err("at least one column required"));
    };
    let n = first.len();
    for (name, column) in names.iter().zip(columns) {
        if column.len() != n {
            return Err(value_err(format!(
                "column length mismatch: '{}' has length {}, '{}' has length {}",
                names[0],
                n,
                name,
                column.len()
            )));
        }
    }
    Ok(n)
}

/// Each column as an owned float64 vector under its name (one copy per column; widened when the
/// dtype is not float64). The pairing is positional.
///
/// # Errors
///
/// An integer column holding a value beyond 2**53.
pub(crate) fn named_f64_columns(
    names: impl IntoIterator<Item = String>,
    columns: &[NumericColumn<'_>],
) -> PyResult<Vec<(String, Vec<f64>)>> {
    names
        .into_iter()
        .zip(columns)
        .map(|(name, column)| column.to_vec(&name).map(|values| (name, values)))
        .collect()
}

/// Numeric NumPy columns of any supported dtype, each copied (and widened) once.
pub(crate) fn numeric_columns_to_batch(
    names: &[String],
    columns: &[NumericColumn<'_>],
) -> PyResult<ColumnBatch> {
    let n = common_length(names, columns)?;
    let mut noncontiguous = 0_u64;
    let mut copied = 0_u64;
    let mut owned = Vec::with_capacity(columns.len());
    for (name, column) in names.iter().zip(columns) {
        if let NumericColumn::F64(array) = column {
            if let Some(buffer) = data_snapshot::numpy_buffer(array)? {
                owned.push(buffer);
                continue;
            }
        }
        let (values, strided) = column.owned(name)?;
        noncontiguous += u64::from(strided);
        copied += 1;
        owned.push(antecedent::data::F64Buffer::owned(values));
    }
    Ok(ColumnBatch {
        names: names.to_vec(),
        columns: owned,
        receipt: NumpyIngestReceipt {
            owned_copies: copied,
            bytes_copied: copied
                .saturating_mul((n as u64).saturating_mul(std::mem::size_of::<f64>() as u64)),
            noncontiguous_columns: noncontiguous,
        },
    })
}

/// The columns of a series, or the per-unit windows of a panel, each element owned once.
///
/// A panel arrives as row-concatenated columns plus each unit's length. Each unit's window is
/// copied from the NumPy buffer straight into that unit's own storage, so there is no pooled
/// batch that a later per-unit slice would copy a second time.
pub(crate) enum IngestedColumns {
    /// One series (or one static table).
    Series(ColumnBatch),
    /// One batch per unit, in unit order.
    Panel(Vec<ColumnBatch>),
}

impl IngestedColumns {
    /// Copies the ingestion made, summed over every unit of a panel.
    pub(crate) fn receipt(&self) -> NumpyIngestReceipt {
        match self {
            Self::Series(batch) => batch.receipt(),
            Self::Panel(units) => units.iter().fold(
                NumpyIngestReceipt { owned_copies: 0, bytes_copied: 0, noncontiguous_columns: 0 },
                |total, unit| {
                    let each = unit.receipt();
                    NumpyIngestReceipt {
                        owned_copies: total.owned_copies.saturating_add(each.owned_copies),
                        bytes_copied: total.bytes_copied.saturating_add(each.bytes_copied),
                        noncontiguous_columns: total
                            .noncontiguous_columns
                            .saturating_add(each.noncontiguous_columns),
                    }
                },
            ),
        }
    }
}

/// NumPy columns owned once: one batch for a series, one per unit window for a panel
/// (`unit_lengths`, which must be positive and add up to the column length).
pub(crate) fn columns_to_ingested(
    names: &[String],
    columns: &[NumericColumn<'_>],
    unit_lengths: Option<&[usize]>,
) -> PyResult<IngestedColumns> {
    let Some(lengths) = unit_lengths else {
        return numeric_columns_to_batch(names, columns).map(IngestedColumns::Series);
    };
    let rows = common_length(names, columns)?;
    crate::discovery_api::check_unit_lengths(rows, lengths)?;
    let mut units = Vec::with_capacity(lengths.len());
    let mut offset = 0_usize;
    for &len in lengths {
        let mut noncontiguous = 0_u64;
        let mut owned = Vec::with_capacity(columns.len());
        for (name, column) in names.iter().zip(columns) {
            let (values, strided) = column.owned_range(name, offset, offset + len)?;
            noncontiguous = noncontiguous.saturating_add(u64::from(strided));
            owned.push(antecedent::data::F64Buffer::owned(values));
        }
        let ncols = u64::try_from(owned.len()).unwrap_or(u64::MAX);
        let unit_bytes = u64::try_from(len)
            .unwrap_or(u64::MAX)
            .saturating_mul(std::mem::size_of::<f64>() as u64);
        units.push(ColumnBatch {
            names: names.to_vec(),
            columns: owned,
            receipt: NumpyIngestReceipt {
                owned_copies: ncols,
                bytes_copied: ncols.saturating_mul(unit_bytes),
                noncontiguous_columns: noncontiguous,
            },
        });
        offset += len;
    }
    Ok(IngestedColumns::Panel(units))
}

/// NumPy columns into library-owned storage: one copy per column.
pub(crate) fn tabular_from_numpy_receipt(
    names: &[String],
    columns: &[NumericColumn<'_>],
) -> PyResult<NumpyTable> {
    numeric_columns_to_batch(names, columns)?.loaded()
}

pub(crate) fn tabular_from_numpy(
    names: &[String],
    columns: &[NumericColumn<'_>],
) -> PyResult<antecedent::prelude::TabularData> {
    tabular_from_numpy_receipt(names, columns).map(|table| table.data)
}

/// Conversion probe: NumPy → library-owned tabular storage, one copy per column.
///
/// Shares the same ingestion path as `analyze*` / `discover_*`. The loaded table is not
/// retained across the FFI boundary; call analysis APIs with the original NumPy columns.
#[pyfunction]
pub(crate) fn load_float64_columns(
    names: Vec<String>,
    columns: Vec<NumericColumn<'_>>,
) -> PyResult<ArrowLoadInfo> {
    catch_ffi(|| {
        let loaded = numeric_columns_to_batch(&names, &columns)?.loaded()?;
        let column_names: Vec<String> =
            loaded.data.schema().variables().iter().map(|v| v.name.to_string()).collect();
        Ok(ArrowLoadInfo {
            row_count: loaded.data.row_count(),
            column_count: loaded.data.schema().len(),
            bytes_copied: loaded.receipt.bytes_copied,
            bytes_borrowed: 0,
            // One materialization entry per owned buffer the ingestion made.
            diagnostic_count: usize::try_from(loaded.receipt.owned_copies).unwrap_or(usize::MAX),
            column_names,
        })
    })
}

/// Conversion probe for a panel: row-concatenated NumPy columns into per-unit library-owned
/// storage, each element copied once (the pooled columns are never copied as a whole first).
#[pyfunction]
pub(crate) fn load_float64_panel_columns(
    names: Vec<String>,
    columns: Vec<NumericColumn<'_>>,
    unit_lengths: Vec<usize>,
) -> PyResult<ArrowLoadInfo> {
    catch_ffi(|| {
        let ingested = columns_to_ingested(&names, &columns, Some(&unit_lengths))?;
        let receipt = ingested.receipt();
        let IngestedColumns::Panel(units) = ingested else {
            return Err(crate::value_err("a panel ingestion yields one batch per unit"));
        };
        let rows: usize = units.iter().map(ColumnBatch::num_rows).sum();
        Ok(ArrowLoadInfo {
            row_count: rows,
            column_count: names.len(),
            bytes_copied: receipt.bytes_copied,
            bytes_borrowed: 0,
            // One materialization entry per owned buffer the ingestion made.
            diagnostic_count: usize::try_from(receipt.owned_copies).unwrap_or(usize::MAX),
            column_names: names,
        })
    })
}

/// Load float64 columns from Arrow C Data Interface exporters (PyArrow / `__arrow_c_array__`).
///
/// Prefers zero-copy borrow of contiguous float64 value buffers. Borrowed buffers must remain
/// unchanged, with no writes through writable aliases, during this call. This conversion probe
/// returns only metadata and retains no data. A table a prepared study, a rebind or an estimate
/// click retains is never borrowed: [`tabular_from_py_columns`] copies Arrow buffers into
/// library-owned storage, so the identity recorded for the data stays the identity of the
/// bytes the study reads.
#[pyfunction]
pub(crate) fn load_float64_arrow_c_columns(
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
) -> PyResult<ArrowLoadInfo> {
    catch_ffi(|| {
        let cdi_cols = ffi::arrow_c_columns(names, columns)?;
        let loaded = antecedent::data::arrow_c::load_columns(cdi_cols).map_err(py_err)?;
        let column_names: Vec<String> =
            loaded.data.schema().variables().iter().map(|v| v.name.to_string()).collect();
        Ok(ArrowLoadInfo {
            row_count: loaded.data.row_count(),
            column_count: loaded.data.schema().len(),
            bytes_copied: loaded.bytes_copied,
            bytes_borrowed: loaded.bytes_borrowed,
            diagnostic_count: loaded.diagnostics.len(),
            column_names,
        })
    })
}

/// Arrow CDI columns borrowed without copying, for a call that does not retain them.
///
/// The returned table aliases the exporter's memory, which a writable alias (a NumPy array
/// behind a PyArrow array) can change at any time. It must not outlive the call: a handle that
/// keeps data uses [`tabular_from_py_columns`], which owns it.
pub(crate) fn tabular_from_arrow_c_objs(
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
) -> PyResult<(antecedent::prelude::TabularData, u64)> {
    let cdi_cols = ffi::arrow_c_columns(names, columns)?;
    let loaded = antecedent::data::arrow_c::load_columns(cdi_cols).map_err(py_err)?;
    Ok((loaded.data, loaded.bytes_borrowed))
}

/// Bytes the owned copy of these Arrow exporters retains, quoted from their lengths before
/// anything is imported or copied: every supported value type is at most eight bytes wide, plus
/// a validity bitmap. `None` when an exporter reports no length.
fn arrow_retention_bytes(columns: &[Bound<'_, PyAny>]) -> Option<u64> {
    columns.iter().try_fold(0_u64, |total, column| {
        let rows = u64::try_from(column.len().ok()?).ok()?;
        total.checked_add(rows.checked_mul(8)?.checked_add(rows.div_ceil(8))?)
    })
}

/// Fund `bytes` of retained table from the caller's execution pool, or refuse.
///
/// The lease is acquired before the copy it pays for and attached to the table itself
/// (`TabularData::with_retention`), whatever its column types, so it is released only when the
/// last holder of the table (or of a table derived from its columns) drops.
/// `CausalResourceError` when the pool cannot fund it; a cancelled run is cancelled.
pub(crate) fn admit_retention(bytes: u64) -> PyResult<Arc<antecedent::execution::Retention>> {
    let mut ctx = antecedent::prelude::ExecutionContext::production_default(1);
    crate::execution_api::apply(&mut ctx);
    antecedent::execution::retain_bounded(&ctx, bytes, std::time::Duration::ZERO)
        .map(Arc::new)
        .map_err(|error| match error {
            antecedent::execution::AdmissionRefusal::Cancelled => {
                py_err(antecedent::CausalError::Cancelled { stage: "ingest" })
            }
            error => crate::CausalResourceError::new_err(error.to_string()),
        })
}

/// Arrow CDI columns copied into library-owned storage.
///
/// Uses the data crate's freeze: every borrowed float64 buffer is copied once into an immutable
/// owned slice, so a later write through an exporter alias is not a write to the table. With
/// `admit`, the copy is funded from the caller's execution pool before it is made and the
/// lease is held by the table; a caller that already funded the same bytes (a
/// `TableSnapshot`) passes `false`.
pub(crate) fn owned_tabular_from_arrow_c_objs(
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
    admit: bool,
) -> PyResult<antecedent::prelude::TabularData> {
    let quoted = if admit { arrow_retention_bytes(&columns) } else { None };
    let mut lease = quoted.map(admit_retention).transpose()?;
    let cdi_cols = ffi::arrow_c_columns(names, columns)?;
    let frame = antecedent::data::arrow_adapter::ArrowFrameView::from_c_columns(cdi_cols)
        .map_err(py_err)?;
    if admit && lease.is_none() {
        // An exporter with no length is quoted from the imported frame, still before the copy.
        let rows = u64::try_from(frame.row_count().map_err(py_err)?).unwrap_or(u64::MAX);
        let cells = rows.saturating_mul(u64::try_from(frame.n_columns()).unwrap_or(u64::MAX));
        lease = Some(admit_retention(cells.saturating_mul(8).saturating_add(cells.div_ceil(8)))?);
    }
    let frozen = frame.freeze().map_err(py_err)?;
    let data = frozen.data().clone();
    Ok(match lease {
        Some(lease) => antecedent::data::with_retention(data, &lease),
        None => data,
    })
}

/// How [`py_columns`] treats Arrow value buffers.
#[derive(Clone, Copy, Eq, PartialEq)]
enum ArrowBuffers {
    /// Borrow contiguous float64 buffers; the table must not outlive the call.
    Borrow,
    /// Copy them into library-owned storage, funded from the caller's execution pool.
    Own,
    /// Copy them into library-owned storage the caller has already funded.
    OwnFunded,
}

fn py_columns(
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
    arrow: ArrowBuffers,
) -> PyResult<(antecedent::prelude::TabularData, Option<u64>)> {
    if names.len() != columns.len() {
        // Preserve the pre-unification ValueError for NumPy callers that catch it.
        return Err(crate::value_err("names and columns must have the same length"));
    }
    if let Some(snapshot) = data_snapshot::from_columns(&names, &columns)? {
        return Ok((snapshot, None));
    }
    let use_arrow = !columns.is_empty()
        && columns.iter().all(|c| c.hasattr("__arrow_c_array__").is_ok_and(|ok| ok));
    if use_arrow {
        return match arrow {
            ArrowBuffers::Borrow => {
                let (data, bytes_borrowed) = tabular_from_arrow_c_objs(names, columns)?;
                Ok((data, Some(bytes_borrowed)))
            }
            ArrowBuffers::Own => Ok((owned_tabular_from_arrow_c_objs(names, columns, true)?, None)),
            ArrowBuffers::OwnFunded => {
                Ok((owned_tabular_from_arrow_c_objs(names, columns, false)?, None))
            }
        };
    }
    let mut np = Vec::with_capacity(columns.len());
    let mut bytes = 0_u64;
    for c in &columns {
        np.push(NumericColumn::from_object(c)?);
        let rows = u64::try_from(c.len()?).unwrap_or(u64::MAX);
        let column = rows.saturating_mul(8).saturating_add(rows.div_ceil(8));
        bytes = bytes.saturating_add(column);
    }
    // A retained table's owned copy is funded before it is made, and the lease lives with the
    // table. A call-scoped borrow retains nothing; a snapshot funded its own copy.
    let lease = (arrow == ArrowBuffers::Own).then(|| admit_retention(bytes)).transpose()?;
    let data = numeric_columns_to_batch(&names, &np)?.loaded()?.data;
    Ok((
        match lease {
            Some(lease) => antecedent::data::with_retention(data, &lease),
            None => data,
        },
        None,
    ))
}

/// NumPy float64 columns or Arrow CDI exporters → library-owned [`TabularData`].
///
/// Every route owns its values, so the table is immutable for as long as anything keeps it: a
/// prepared study, a rebind, an estimate click or a retained result. NumPy columns are copied
/// (a NumPy buffer is writable); Arrow buffers, which can alias a writable NumPy array, are
/// copied too. Each copy is funded from the caller's execution pool before it is made
/// (`CausalResourceError` when `Execution(memory_bytes=...)` cannot fund it), and the lease is
/// held by the table for as long as any holder keeps it, whatever its column types. Only a
/// complete snapshot column, immutable by construction, is shared.
pub(crate) fn tabular_from_py_columns(
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
) -> PyResult<antecedent::prelude::TabularData> {
    py_columns(names, columns, ArrowBuffers::Own).map(|(data, _)| data)
}

/// Like [`tabular_from_py_columns`] for a caller that has already funded the table's retained
/// bytes from the execution pool (and holds that lease on the table), so the copy is not
/// charged twice.
pub(crate) fn funded_tabular_from_py_columns(
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
) -> PyResult<antecedent::prelude::TabularData> {
    py_columns(names, columns, ArrowBuffers::OwnFunded).map(|(data, _)| data)
}

/// Like [`tabular_from_py_columns`], but Arrow float64 buffers are borrowed under the caller's
/// explicit contract (`antecedent.data.table(..., borrow=True)`); the second value is the bytes
/// borrowed.
pub(crate) fn borrowing_tabular_from_py_columns(
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
) -> PyResult<(antecedent::prelude::TabularData, Option<u64>)> {
    py_columns(names, columns, ArrowBuffers::Borrow)
}
