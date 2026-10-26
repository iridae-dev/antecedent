//! Column conversion: Arrow arrays and record batches to owned library columns.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::chunks::*;
use super::schema::*;
use super::*;

/// Load float64 and fixed-size float64-list columns from an Arrow record batch.
///
/// Always copies into library-owned buffers (in-process `RecordBatch` path).
/// Prefer [`tabular_from_arrow_c_columns`] for Arrow C Data Interface zero-copy.
/// Null cells are invalid and hold `NaN`; a fixed-size-list row with a null
/// component is an invalid row.
///
/// # Errors
///
/// Unsupported column types, empty batches, schema construction failures, or more
/// than `u32::MAX` columns.
pub fn tabular_from_record_batch(batch: &RecordBatch) -> Result<ArrowLoadResult, DataError> {
    if batch.num_columns() == 0 {
        return Err(DataError::empty("record batch columns"));
    }
    let mut builder = CausalSchemaBuilder::new();
    let mut columns = Vec::with_capacity(batch.num_columns());
    let mut diagnostics = DiagnosticSet::new();
    let mut bytes_copied = 0u64;
    let n_rows = batch.num_rows();

    for (i, field) in batch.schema().fields().iter().enumerate() {
        push_record_column(
            &mut RecordImport {
                builder: &mut builder,
                columns: &mut columns,
                diagnostics: &mut diagnostics,
                bytes_copied: &mut bytes_copied,
            },
            i,
            field.name(),
            batch.column(i),
            n_rows,
        )?;
    }

    let schema: CausalSchema = builder.build().map_err(DataError::from)?;
    let storage = OwnedColumnarStorage::try_new(schema, columns, None, None)?;
    Ok(ArrowLoadResult {
        data: TabularData::new(storage),
        diagnostics,
        bytes_copied,
        bytes_borrowed: 0,
    })
}

pub(super) struct RecordImport<'a> {
    builder: &'a mut CausalSchemaBuilder,
    columns: &'a mut Vec<OwnedColumn>,
    diagnostics: &'a mut DiagnosticSet,
    bytes_copied: &'a mut u64,
}

pub(super) fn push_record_column(
    import: &mut RecordImport<'_>,
    index: usize,
    name: &str,
    array: &ArrayRef,
    n_rows: usize,
) -> Result<(), DataError> {
    let id = VariableId::from_raw(
        u32::try_from(index)
            .map_err(|_| DataError::overflow("Arrow column count as a VariableId"))?,
    );
    if let Some(floats) = array.as_any().downcast_ref::<Float64Array>() {
        import
            .builder
            .add_variable(
                Arc::<str>::from(name),
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .map_err(DataError::from)?;
        let (col, copied) = float64_owned_from_array(id, floats, n_rows)?;
        *import.bytes_copied += copied;
        import.diagnostics.push(materialization_diagnostic(
            MaterializationReason::ForeignBufferIncompatible,
            copied,
        ));
        import.columns.push(OwnedColumn::Float64(col));
        return Ok(());
    }
    if let Some(list) = array.as_any().downcast_ref::<FixedSizeListArray>() {
        let dim = usize::try_from(list.value_length())
            .map_err(|_| DataError::overflow("FixedSizeList width as usize"))?;
        if dim == 0 {
            return Err(DataError::invalid_value(
                "FixedSizeList width",
                "must be greater than zero",
            ));
        }
        let width = NonZeroU32::new(
            u32::try_from(dim).map_err(|_| DataError::overflow("FixedSizeList width as u32"))?,
        )
        .ok_or(DataError::invalid_value("FixedSizeList width", "must be greater than zero"))?;
        import
            .builder
            .add_variable(
                Arc::<str>::from(name),
                ValueType::Vector { width, element: ScalarType::Float64 },
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .map_err(DataError::from)?;
        let (col, copied) = fixed_vector_from_list(id, list, n_rows, dim)?;
        *import.bytes_copied += copied;
        import.diagnostics.push(materialization_diagnostic(
            MaterializationReason::ForeignBufferIncompatible,
            copied,
        ));
        import.columns.push(OwnedColumn::FixedVector(col));
        return Ok(());
    }
    let domain_id = CategoryDomainId::from_raw(id.raw());
    let Some(col) = typed_owned_column(id, domain_id, array)? else {
        return Err(DataError::TypeMismatch {
            id,
            expected: "float64, int, timestamp, dictionary, or FixedSizeList<float64>",
        });
    };
    let (value_type, roles, domain) = schema_binding(&col)?;
    import
        .builder
        .add_variable(
            Arc::<str>::from(name),
            value_type,
            roles,
            None,
            domain,
            MeasurementSpec::default(),
        )
        .map_err(DataError::from)?;
    let copied = owned_payload_bytes(&col)?;
    *import.bytes_copied += copied;
    import
        .diagnostics
        .push(materialization_diagnostic(MaterializationReason::ForeignBufferIncompatible, copied));
    import.columns.push(col);
    Ok(())
}

pub(super) fn float64_owned_from_array(
    id: VariableId,
    floats: &Float64Array,
    n_rows: usize,
) -> Result<(Float64Column, u64), DataError> {
    // One bulk copy of the value buffer straight into the shared slice; only a column that
    // actually has nulls pays a per-row pass, to write NaN under each missing cell (so
    // value-only readers never see an observation) and to build the bitmap.
    let mut validity_bytes = vec![0u8; n_rows.div_ceil(8)];
    let values: Arc<[f64]> = if floats.null_count() == 0 {
        let (full, tail) = (n_rows / 8, n_rows % 8);
        validity_bytes[..full].fill(0xFF);
        if tail > 0 {
            validity_bytes[full] = (1u8 << tail) - 1;
        }
        Arc::from(&floats.values()[..])
    } else {
        let mut patched = floats.values().to_vec();
        for row in 0..n_rows {
            if floats.is_null(row) {
                patched[row] = f64::NAN;
            } else {
                validity_bytes[row / 8] |= 1 << (row % 8);
            }
        }
        Arc::from(patched)
    };
    let copied = (values.len() * core::mem::size_of::<f64>() + validity_bytes.len()) as u64;
    let col = Float64Column::new(
        id,
        F64Buffer::owned(values),
        ValidityBitmap::from_bytes(validity_bytes, n_rows)?,
    )?;
    Ok((col, copied))
}

pub(super) fn fixed_vector_from_list(
    id: VariableId,
    list: &FixedSizeListArray,
    n_rows: usize,
    dim: usize,
) -> Result<(FixedVectorColumn, u64), DataError> {
    let values = list.values();
    let floats = values
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or(DataError::TypeMismatch { id, expected: "FixedSizeList<float64>" })?;
    let span = || DataError::overflow("FixedSizeList value span (child buffer)");
    let cells = n_rows.checked_mul(dim).ok_or_else(span)?;
    // Validity is per row, so a vector with any null component is a missing row;
    // every component of a missing row holds NaN.
    let mut flat = Vec::with_capacity(cells);
    let mut validity_bytes = vec![0u8; n_rows.div_ceil(8)];
    let raw = floats.values();
    let list_offset = list.offset();
    let child_offset = floats.offset();
    for row in 0..n_rows {
        let start = list_offset
            .checked_mul(dim)
            .and_then(|base| row.checked_mul(dim).and_then(|step| base.checked_add(step)))
            .ok_or_else(span)?;
        let end = start.checked_add(dim).ok_or_else(span)?;
        if end > floats.len() {
            return Err(span());
        }
        if list.is_null(row) || (start..end).any(|idx| floats.is_null(idx)) {
            flat.extend(std::iter::repeat_n(f64::NAN, dim));
            continue;
        }
        validity_bytes[row / 8] |= 1 << (row % 8);
        let phys = child_offset.checked_add(start).ok_or_else(span)?;
        let phys_end = phys.checked_add(dim).ok_or_else(span)?;
        if phys_end > raw.len() {
            return Err(span());
        }
        flat.extend_from_slice(&raw[phys..phys_end]);
    }
    let copied = (flat.len() * core::mem::size_of::<f64>() + validity_bytes.len()) as u64;
    let col = FixedVectorColumn::new(
        id,
        dim,
        Arc::from(flat),
        ValidityBitmap::from_bytes(validity_bytes, n_rows)?,
    )?;
    Ok((col, copied))
}

/// Load Arrow C Data Interface columns, preferring zero-copy for float64.
///
/// Consumes each [`ArrowCColumn`]'s FFI structs. One chunk of a contiguous
/// float64 value buffer is borrowed; validity bitmaps are copied. A float64
/// chunk with nulls is borrowed only when every null slot already holds `NaN`;
/// otherwise its values are copied with `NaN` written under each null.
/// Integer ids, timestamps, and dictionary categories are stored as
/// [`OwnedColumn::Int64`], [`OwnedColumn::Timestamp`], and
/// [`OwnedColumn::Categorical`]. Several chunks are concatenated because the
/// returned table is one buffer per column; use [`ArrowFrameView`] to keep them.
/// Row count is a `usize` sum of the chunks. It is not capped at `u32::MAX`.
///
/// # Errors
///
/// Empty input, an unsupported type, a CDI or schema failure, a category or
/// timestamp that cannot be kept exact, or more than `u32::MAX` columns.
pub fn tabular_from_arrow_c_columns(
    columns: Vec<ArrowCColumn>,
) -> Result<ArrowLoadResult, DataError> {
    let frame = ArrowFrameView::from_c_columns(columns)?;
    if frame.n_chunks() <= 1 { frame.load_single_chunk() } else { frame.contiguous() }
}

pub(crate) fn finish_load(
    names: &[String],
    columns: Vec<OwnedColumn>,
    diagnostics: DiagnosticSet,
    bytes_copied: u64,
    bytes_borrowed: u64,
) -> Result<ArrowLoadResult, DataError> {
    let mut builder = CausalSchemaBuilder::new();
    for (name, column) in names.iter().zip(columns.iter()) {
        let (value_type, roles, domain) = schema_binding(column)?;
        builder
            .add_variable(
                Arc::<str>::from(name.as_str()),
                value_type,
                roles,
                None,
                domain,
                MeasurementSpec::default(),
            )
            .map_err(DataError::from)?;
    }
    let schema = builder.build().map_err(DataError::from)?;
    let storage = OwnedColumnarStorage::try_new(schema, columns, None, None)?;
    Ok(ArrowLoadResult {
        data: TabularData::new(storage),
        diagnostics,
        bytes_copied,
        bytes_borrowed,
    })
}

pub(super) fn column_from_array(
    id: VariableId,
    domain_id: CategoryDomainId,
    array: ArrayRef,
) -> Result<(OwnedColumn, u64, u64, Option<antecedent_core::Diagnostic>), DataError> {
    if array.data_type() == &DataType::Float64 {
        return float64_column_from_array(id, array);
    }
    if let Some(list) = array.as_any().downcast_ref::<FixedSizeListArray>() {
        let dim = usize::try_from(list.value_length())
            .map_err(|_| DataError::overflow("FixedSizeList width as usize"))?;
        let (col, copied) = fixed_vector_from_list(id, list, list.len(), dim)?;
        let diag =
            materialization_diagnostic(MaterializationReason::ForeignBufferIncompatible, copied);
        return Ok((OwnedColumn::FixedVector(col), 0, copied, Some(diag)));
    }
    let Some(col) = typed_owned_column(id, domain_id, &array)? else {
        return Err(DataError::TypeMismatch {
            id,
            expected: "float64, int, timestamp, dictionary, or FixedSizeList<float64>",
        });
    };
    let copied = owned_payload_bytes(&col)?;
    let diag = materialization_diagnostic(MaterializationReason::ForeignBufferIncompatible, copied);
    Ok((col, 0, copied, Some(diag)))
}

pub(super) fn typed_owned_column(
    id: VariableId,
    domain_id: CategoryDomainId,
    array: &ArrayRef,
) -> Result<Option<OwnedColumn>, DataError> {
    let column = match array.data_type() {
        DataType::Int8 => int64_from_signed::<Int8Type>(id, array.as_primitive())?,
        DataType::Int16 => int64_from_signed::<Int16Type>(id, array.as_primitive())?,
        DataType::Int32 => int64_from_signed::<Int32Type>(id, array.as_primitive())?,
        DataType::Int64 => int64_from_signed::<Int64Type>(id, array.as_primitive())?,
        DataType::UInt8 => int64_from_unsigned::<UInt8Type>(id, array.as_primitive())?,
        DataType::UInt16 => int64_from_unsigned::<UInt16Type>(id, array.as_primitive())?,
        DataType::UInt32 => int64_from_unsigned::<UInt32Type>(id, array.as_primitive())?,
        DataType::UInt64 => int64_from_unsigned::<UInt64Type>(id, array.as_primitive())?,
        DataType::Timestamp(unit, _) => timestamp_column(id, array.as_ref(), *unit)?,
        DataType::Dictionary(_, _) => categorical_column(id, domain_id, array.as_ref())?,
        _ => return Ok(None),
    };
    Ok(Some(column))
}

pub(super) fn int64_from_signed<T>(
    id: VariableId,
    array: &PrimitiveArray<T>,
) -> Result<OwnedColumn, DataError>
where
    T: arrow_array::types::ArrowPrimitiveType,
    T::Native: Into<i64>,
{
    let (validity, _) = validity_from_array(array)?;
    let mut values = Vec::with_capacity(array.len());
    for row in 0..array.len() {
        let value = if array.is_null(row) { 0 } else { array.value(row).into() };
        values.push(value);
    }
    Ok(OwnedColumn::Int64(Int64Column::new(id, values, validity)?))
}

pub(super) fn int64_from_unsigned<T>(
    id: VariableId,
    array: &PrimitiveArray<T>,
) -> Result<OwnedColumn, DataError>
where
    T: arrow_array::types::ArrowPrimitiveType,
    T::Native: Into<u64>,
{
    let (validity, _) = validity_from_array(array)?;
    let mut values = Vec::with_capacity(array.len());
    for row in 0..array.len() {
        if array.is_null(row) {
            values.push(0);
            continue;
        }
        let raw = array.value(row).into();
        let value =
            i64::try_from(raw).map_err(|_| DataError::overflow("unsigned id (exact int64)"))?;
        values.push(value);
    }
    Ok(OwnedColumn::Int64(Int64Column::new(id, values, validity)?))
}

pub(super) fn timestamp_column(
    id: VariableId,
    array: &dyn Array,
    unit: TimeUnit,
) -> Result<OwnedColumn, DataError> {
    let (validity, _) = validity_from_array(array)?;
    let scale: i64 = match unit {
        TimeUnit::Nanosecond => 1,
        TimeUnit::Microsecond => 1_000,
        TimeUnit::Millisecond => 1_000_000,
        TimeUnit::Second => 1_000_000_000,
    };
    let mut values = Vec::with_capacity(array.len());
    for row in 0..array.len() {
        if array.is_null(row) {
            values.push(0);
            continue;
        }
        let raw = timestamp_raw(array, unit, row);
        let nanos = raw
            .checked_mul(scale)
            .ok_or_else(|| DataError::overflow("timestamp (exact nanoseconds)"))?;
        values.push(nanos);
    }
    Ok(OwnedColumn::Timestamp(TimestampColumn::new(id, values, validity)?))
}

pub(super) fn timestamp_raw(array: &dyn Array, unit: TimeUnit, row: usize) -> i64 {
    match unit {
        TimeUnit::Nanosecond => array.as_primitive::<TimestampNanosecondType>().value(row),
        TimeUnit::Microsecond => array.as_primitive::<TimestampMicrosecondType>().value(row),
        TimeUnit::Millisecond => array.as_primitive::<TimestampMillisecondType>().value(row),
        TimeUnit::Second => array.as_primitive::<TimestampSecondType>().value(row),
    }
}

pub(super) fn categorical_column(
    id: VariableId,
    domain_id: CategoryDomainId,
    array: &dyn Array,
) -> Result<OwnedColumn, DataError> {
    let DataType::Dictionary(key, _) = array.data_type() else {
        return Err(DataError::TypeMismatch { id, expected: "dictionary" });
    };
    match key.as_ref() {
        DataType::Int8 => dict_from::<Int8Type>(id, domain_id, array.as_dictionary()),
        DataType::Int16 => dict_from::<Int16Type>(id, domain_id, array.as_dictionary()),
        DataType::Int32 => dict_from::<Int32Type>(id, domain_id, array.as_dictionary()),
        DataType::Int64 => dict_from::<Int64Type>(id, domain_id, array.as_dictionary()),
        DataType::UInt8 => dict_from::<UInt8Type>(id, domain_id, array.as_dictionary()),
        DataType::UInt16 => dict_from::<UInt16Type>(id, domain_id, array.as_dictionary()),
        DataType::UInt32 => dict_from::<UInt32Type>(id, domain_id, array.as_dictionary()),
        other => Err(DataError::unsupported(
            "importing a dictionary column",
            format!("key type {other} is not an exact category code"),
        )),
    }
}

pub(super) fn dict_from<K>(
    id: VariableId,
    domain_id: CategoryDomainId,
    array: &DictionaryArray<K>,
) -> Result<OwnedColumn, DataError>
where
    K: ArrowDictionaryKeyType,
{
    let labels = dictionary_labels(array.values())?;
    let domain = Arc::new(CategoryDomain::try_new(
        domain_id,
        labels,
        array.is_ordered(),
        None,
        UnknownCategoryPolicy::Fail,
    )?);
    let (validity, _) = validity_from_array(array)?;
    let mut codes = Vec::with_capacity(array.len());
    for row in 0..array.len() {
        let code = match array.key(row) {
            None => 0,
            Some(key) => u32::try_from(key)
                .map_err(|_| DataError::overflow("dictionary key (category code)"))?,
        };
        codes.push(CategoryCode::from_raw(code));
    }
    Ok(OwnedColumn::Categorical(CategoricalColumn::try_new(id, codes, validity, domain)?))
}

pub(super) fn dictionary_labels(values: &ArrayRef) -> Result<Vec<CategoryLevel>, DataError> {
    if values.is_empty() {
        return Err(DataError::empty("dictionary category domain levels"));
    }
    let mut labels = Vec::with_capacity(values.len());
    for row in 0..values.len() {
        if values.is_null(row) {
            return Err(DataError::invalid_value("dictionary level", "is null"));
        }
        labels.push(CategoryLevel { label: Arc::<str>::from(level_text(values.as_ref(), row)?) });
    }
    Ok(labels)
}

pub(crate) fn level_text(values: &dyn Array, row: usize) -> Result<String, DataError> {
    if let Some(text) = values.as_any().downcast_ref::<StringArray>() {
        return Ok(text.value(row).to_owned());
    }
    if let Some(text) = values.as_any().downcast_ref::<LargeStringArray>() {
        return Ok(text.value(row).to_owned());
    }
    if let Some(text) = values.as_any().downcast_ref::<StringViewArray>() {
        return Ok(text.value(row).to_owned());
    }
    let rendered = match values.data_type() {
        DataType::Int8 => values.as_primitive::<Int8Type>().value(row).to_string(),
        DataType::Int16 => values.as_primitive::<Int16Type>().value(row).to_string(),
        DataType::Int32 => values.as_primitive::<Int32Type>().value(row).to_string(),
        DataType::Int64 => values.as_primitive::<Int64Type>().value(row).to_string(),
        DataType::UInt8 => values.as_primitive::<UInt8Type>().value(row).to_string(),
        DataType::UInt16 => values.as_primitive::<UInt16Type>().value(row).to_string(),
        DataType::UInt32 => values.as_primitive::<UInt32Type>().value(row).to_string(),
        DataType::UInt64 => values.as_primitive::<UInt64Type>().value(row).to_string(),
        other => {
            return Err(DataError::unsupported(
                "importing dictionary values",
                format!("values of type {other} are not category labels"),
            ));
        }
    };
    Ok(rendered)
}
