//! Schema binding and the scans a frame is converted under (schema, categories, missingness).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::chunks::*;
use super::*;

pub(super) fn schema_binding(
    column: &OwnedColumn,
) -> Result<(ValueType, SmallRoleSet, Option<CategoryDomainId>), DataError> {
    Ok(match column {
        OwnedColumn::Float64(_) => {
            (ValueType::Continuous, SmallRoleSet::from_hint(RoleHint::Context), None)
        }
        OwnedColumn::Int64(_) => {
            (ValueType::Count, SmallRoleSet::from_hint(RoleHint::UnitId), None)
        }
        OwnedColumn::Timestamp(_) => {
            (ValueType::Count, SmallRoleSet::from_hint(RoleHint::Time), None)
        }
        OwnedColumn::Boolean(_) => {
            (ValueType::Binary, SmallRoleSet::from_hint(RoleHint::Context), None)
        }
        OwnedColumn::Categorical(column) => {
            let value_type =
                if column.domain.ordered { ValueType::Ordinal } else { ValueType::Categorical };
            (value_type, SmallRoleSet::from_hint(RoleHint::Context), Some(column.domain.id))
        }
        OwnedColumn::FixedVector(column) => {
            let width = NonZeroU32::new(
                u32::try_from(column.dim)
                    .map_err(|_| DataError::overflow("FixedSizeList width as u32"))?,
            )
            .ok_or_else(|| {
                DataError::invalid_value("FixedSizeList width", "must be greater than zero")
            })?;
            (
                ValueType::Vector { width, element: ScalarType::Float64 },
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
            )
        }
    })
}

pub(super) fn resolve_order(n_columns: usize, order: &[usize]) -> Result<Vec<usize>, DataError> {
    let order = if order.is_empty() { (0..n_columns).collect::<Vec<_>>() } else { order.to_vec() };
    let mut seen = vec![false; n_columns];
    for &index in &order {
        if index >= n_columns || seen[index] {
            return Err(if index >= n_columns {
                DataError::out_of_range("preparation order index", index, n_columns)
            } else {
                DataError::duplicate("preparation order index")
            });
        }
        seen[index] = true;
    }
    Ok(order)
}

pub(super) fn frame_payload_bytes(frame: &ArrowFrameView) -> Result<u64, DataError> {
    let mut total = 0u64;
    for column in &frame.columns {
        for chunk in &column.chunks {
            total = total
                .checked_add(owned_payload_bytes(chunk)?)
                .ok_or_else(|| DataError::overflow("column payload size"))?;
        }
    }
    Ok(total)
}

pub(super) fn scan_frame(
    frame: &ArrowFrameView,
    order: &[usize],
    missing: MissingPolicy,
    categories: &[(VariableId, Contrast)],
    bytes: u64,
    budget: u64,
) -> Result<StreamingScan, DataError> {
    let schema = scan_schema(frame, order)?;
    let categories = scan_categories(frame, order, categories)?;
    let missing = scan_missing(frame, order, missing)?;
    let chunk_rows = chunk_row_counts(frame)?;
    finish_scan(schema, &categories, missing, &chunk_rows, bytes, budget)
}

pub(super) fn scan_categories(
    frame: &ArrowFrameView,
    order: &[usize],
    categories: &[(VariableId, Contrast)],
) -> Result<Vec<crate::categorical::CategoryConversion>, DataError> {
    let mut recorded = Vec::new();
    for &index in order {
        let Some(chunk) = frame.columns[index].chunks.first() else {
            continue;
        };
        let OwnedColumn::Categorical(column) = chunk else {
            continue;
        };
        let id = variable_id(index)?;
        let Some((_, contrast)) = categories.iter().find(|(variable, _)| *variable == id) else {
            return Err(DataError::invalid_column(
                id,
                "needs an explicit contrast for its category conversion",
            ));
        };
        recorded.push(crate::categorical::expand_contrast(column, contrast)?.conversion);
    }
    Ok(recorded)
}

pub(super) fn scan_schema(
    frame: &ArrowFrameView,
    order: &[usize],
) -> Result<SchemaConversion, DataError> {
    let mut columns = Vec::with_capacity(order.len());
    for &index in order {
        let column = &frame.columns[index];
        let kind = column.chunks.first().map_or(SchemaValueKind::Empty, storage_kind);
        columns.push(SchemaColumn {
            name: Arc::<str>::from(column.name.as_str()),
            id: variable_id(index)?,
            value_kind: kind,
        });
    }
    Ok(SchemaConversion { columns: Arc::from(columns) })
}

pub(super) const fn storage_kind(column: &OwnedColumn) -> SchemaValueKind {
    match column {
        OwnedColumn::Float64(_) => SchemaValueKind::Float64,
        OwnedColumn::Int64(_) => SchemaValueKind::Int64,
        OwnedColumn::Boolean(_) => SchemaValueKind::Boolean,
        OwnedColumn::Categorical(_) => SchemaValueKind::Categorical,
        OwnedColumn::Timestamp(_) => SchemaValueKind::Timestamp,
        OwnedColumn::FixedVector(_) => SchemaValueKind::Vector,
    }
}

pub(super) fn scan_missing(
    frame: &ArrowFrameView,
    order: &[usize],
    policy: MissingPolicy,
) -> Result<MissingConversion, DataError> {
    let mut chunks = Vec::with_capacity(frame.n_chunks());
    for index in 0..frame.n_chunks() {
        chunks.push(chunk_missing(frame, order, index)?);
    }
    MissingConversion::from_scan(policy, chunks)
}

pub(super) fn chunk_missing(
    frame: &ArrowFrameView,
    order: &[usize],
    index: usize,
) -> Result<ChunkMissing, DataError> {
    let rows = frame
        .columns
        .first()
        .map_or(0, |column| column.chunks.get(index).map_or(0, OwnedColumn::len));
    let mut dropped = Vec::new();
    for row in 0..rows {
        let invalid = order
            .iter()
            .any(|&column| !frame.columns[column].chunks[index].as_view().validity().is_valid(row));
        if invalid {
            let row = u64::try_from(row).map_err(|_| DataError::overflow("chunk row (u64)"))?;
            dropped.push(row);
        }
    }
    let rows_kept =
        rows.checked_sub(dropped.len()).ok_or_else(|| DataError::overflow("chunk kept rows"))?;
    Ok(ChunkMissing {
        rows_total: rows,
        rows_kept,
        dropped_digest: Arc::from(dropped_row_digest(rows, &dropped)),
    })
}

pub(super) fn chunk_row_counts(frame: &ArrowFrameView) -> Result<Vec<u64>, DataError> {
    let Some(column) = frame.columns.first() else {
        return Ok(Vec::new());
    };
    column
        .chunks
        .iter()
        .map(|chunk| u64::try_from(chunk.len()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| DataError::overflow("chunk row count (u64)"))
}

pub(super) fn materialize_frame(
    frame: &ArrowFrameView,
    order: &[usize],
    missing: MissingPolicy,
    categories: &[(VariableId, Contrast)],
    controls: PreparationControls,
    budget: u64,
) -> Result<EstimatorFrame, DataError> {
    let loaded = frame.contiguous()?;
    let ids = order.iter().copied().map(variable_id).collect::<Result<Vec<_>, _>>()?;
    let mut finite = controls;
    finite.stream = EstimatorStreamNeed::CannotStream;
    let mut product = prepare_table(&loaded.data, &ids, missing, categories, finite)?;
    let mut diagnostics = loaded.diagnostics;
    if frame.n_chunks() > 1 {
        product = product.into_materialized_from_scan()?;
        let copied = product.receipt.admission().materialized_bytes().unwrap_or(budget);
        diagnostics
            .push(materialization_diagnostic(MaterializationReason::EstimatorCannotStream, copied));
    }
    Ok(EstimatorFrame::Finite { table: loaded.data, product, diagnostics })
}

pub(super) fn detach_foreign(column: &OwnedColumn) -> Result<OwnedColumn, DataError> {
    match column {
        OwnedColumn::Float64(column) if column.values.is_foreign() => {
            let values = Arc::<[f64]>::from(column.values.as_slice());
            Ok(OwnedColumn::Float64(Float64Column::new(
                column.id,
                F64Buffer::owned(values),
                column.validity.clone(),
            )?))
        }
        other => Ok(other.clone()),
    }
}
