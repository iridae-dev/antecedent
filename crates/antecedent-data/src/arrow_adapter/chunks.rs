//! Chunk bookkeeping: alignment checks and per-column chunk concatenation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use super::convert::*;

/// Sum chunk lengths. A total above `u32::MAX` is a valid `usize` on a 64-bit
/// target; this function does not reject it.
pub(super) fn checked_row_total(chunk_rows: impl IntoIterator<Item = usize>) -> Result<usize, DataError> {
    let mut total = 0usize;
    for rows in chunk_rows {
        total = total.checked_add(rows).ok_or_else(|| DataError::InvalidArgument {
            message: "Arrow chunk row counts overflow usize".into(),
        })?;
    }
    Ok(total)
}

pub(super) fn remember_chunk_count(seen: &mut Option<usize>, count: usize) -> Result<(), DataError> {
    match *seen {
        None => *seen = Some(count),
        Some(expected) if expected == count => {}
        Some(expected) => {
            return Err(DataError::LengthMismatch {
                expected,
                actual: count,
                context: "Arrow CDI chunk counts",
            });
        }
    }
    Ok(())
}

pub(super) fn align_chunks(columns: &[FrameColumn]) -> Result<(), DataError> {
    let Some(first) = columns.first() else {
        return Ok(());
    };
    for column in columns.iter().skip(1) {
        if column.chunks.len() != first.chunks.len() {
            return Err(DataError::LengthMismatch {
                expected: first.chunks.len(),
                actual: column.chunks.len(),
                context: "Arrow CDI chunk counts",
            });
        }
        for (index, chunk) in column.chunks.iter().enumerate() {
            let expected = first.chunks[index].len();
            if chunk.len() != expected {
                return Err(DataError::LengthMismatch {
                    expected,
                    actual: chunk.len(),
                    context: "Arrow CDI column lengths",
                });
            }
        }
    }
    Ok(())
}

pub(super) fn variable_id(index: usize) -> Result<VariableId, DataError> {
    let raw = u32::try_from(index).map_err(|_| DataError::InvalidArgument {
        message: "too many Arrow columns for VariableId".into(),
    })?;
    Ok(VariableId::from_raw(raw))
}

pub(super) fn owned_payload_bytes(column: &OwnedColumn) -> Result<u64, DataError> {
    let (n_values, width) = match column {
        OwnedColumn::Float64(column) => (column.len(), core::mem::size_of::<f64>()),
        OwnedColumn::Int64(column) => (column.len(), core::mem::size_of::<i64>()),
        OwnedColumn::Timestamp(column) => (column.len(), core::mem::size_of::<i64>()),
        OwnedColumn::Boolean(column) => (column.len(), 1usize),
        OwnedColumn::Categorical(column) => (column.len(), core::mem::size_of::<CategoryCode>()),
        OwnedColumn::FixedVector(column) => (column.values.len(), core::mem::size_of::<f64>()),
    };
    let values = n_values.checked_mul(width).ok_or_else(|| DataError::InvalidArgument {
        message: "column payload size overflows usize".into(),
    })?;
    let validity = column.as_view().validity().len().div_ceil(8);
    let total = values.checked_add(validity).ok_or_else(|| DataError::InvalidArgument {
        message: "column payload size overflows usize".into(),
    })?;
    u64::try_from(total).map_err(|_| DataError::InvalidArgument {
        message: "column payload size does not fit u64".into(),
    })
}

pub(super) fn empty_column(
    id: VariableId,
    domain_id: CategoryDomainId,
    data_type: &DataType,
) -> Result<OwnedColumn, DataError> {
    let empty: ArrayRef = arrow_array::new_empty_array(data_type);
    let (column, _, _, _) = column_from_array(id, domain_id, empty)?;
    Ok(column)
}

pub(super) fn same_column_kind(previous: &OwnedColumn, next: &OwnedColumn) -> Result<(), DataError> {
    let same = match (previous, next) {
        (OwnedColumn::Float64(_), OwnedColumn::Float64(_))
        | (OwnedColumn::Int64(_), OwnedColumn::Int64(_))
        | (OwnedColumn::Timestamp(_), OwnedColumn::Timestamp(_))
        | (OwnedColumn::Boolean(_), OwnedColumn::Boolean(_)) => true,
        (OwnedColumn::FixedVector(left), OwnedColumn::FixedVector(right)) => left.dim == right.dim,
        (OwnedColumn::Categorical(left), OwnedColumn::Categorical(right)) => {
            left.domain == right.domain
        }
        _ => false,
    };
    if same {
        Ok(())
    } else {
        Err(DataError::InvalidArgument {
            message: "Arrow chunks of one column disagree in type or category domain".into(),
        })
    }
}

pub(super) fn concat_chunks(chunks: &[OwnedColumn]) -> Result<OwnedColumn, DataError> {
    let Some(first) = chunks.first() else {
        return Err(DataError::InvalidArgument {
            message: "Arrow column has no chunks to concatenate".into(),
        });
    };
    if chunks.len() == 1 {
        return Ok(first.clone());
    }
    for chunk in chunks.iter().skip(1) {
        same_column_kind(first, chunk)?;
    }
    match first {
        OwnedColumn::Float64(_) => concat_float(chunks),
        OwnedColumn::Int64(_) => concat_int(chunks),
        OwnedColumn::Timestamp(_) => concat_timestamp(chunks),
        OwnedColumn::Categorical(_) => concat_categorical(chunks),
        OwnedColumn::FixedVector(_) => concat_vectors(chunks),
        OwnedColumn::Boolean(_) => Err(DataError::InvalidArgument {
            message: "boolean Arrow chunks are not concatenated".into(),
        }),
    }
}

pub(super) fn concat_float(chunks: &[OwnedColumn]) -> Result<OwnedColumn, DataError> {
    let mut values = Vec::new();
    let mut validity = Vec::new();
    let mut len = 0usize;
    let mut id = None;
    for chunk in chunks {
        let OwnedColumn::Float64(column) = chunk else {
            return Err(DataError::TypeMismatch { id: chunk.id(), expected: "float64" });
        };
        id = Some(column.id);
        values.extend_from_slice(column.values.as_slice());
        len = append_validity(&mut validity, len, &column.validity)?;
    }
    let id = id.ok_or_else(|| DataError::InvalidArgument {
        message: "Arrow column has no chunks to concatenate".into(),
    })?;
    Ok(OwnedColumn::Float64(Float64Column::new(
        id,
        F64Buffer::owned(Arc::<[f64]>::from(values)),
        ValidityBitmap::from_bytes(validity, len)?,
    )?))
}

pub(super) fn concat_int(chunks: &[OwnedColumn]) -> Result<OwnedColumn, DataError> {
    let mut values = Vec::new();
    let mut validity = Vec::new();
    let mut len = 0usize;
    let mut id = None;
    for chunk in chunks {
        let OwnedColumn::Int64(column) = chunk else {
            return Err(DataError::TypeMismatch { id: chunk.id(), expected: "int64" });
        };
        id = Some(column.id);
        values.extend_from_slice(&column.values);
        len = append_validity(&mut validity, len, &column.validity)?;
    }
    let id = id.ok_or_else(|| DataError::InvalidArgument {
        message: "Arrow column has no chunks to concatenate".into(),
    })?;
    Ok(OwnedColumn::Int64(Int64Column::new(
        id,
        values,
        ValidityBitmap::from_bytes(validity, len)?,
    )?))
}

pub(super) fn concat_timestamp(chunks: &[OwnedColumn]) -> Result<OwnedColumn, DataError> {
    let mut values = Vec::new();
    let mut validity = Vec::new();
    let mut len = 0usize;
    let mut id = None;
    for chunk in chunks {
        let OwnedColumn::Timestamp(column) = chunk else {
            return Err(DataError::TypeMismatch { id: chunk.id(), expected: "timestamp" });
        };
        id = Some(column.id);
        values.extend_from_slice(&column.values_ns);
        len = append_validity(&mut validity, len, &column.validity)?;
    }
    let id = id.ok_or_else(|| DataError::InvalidArgument {
        message: "Arrow column has no chunks to concatenate".into(),
    })?;
    Ok(OwnedColumn::Timestamp(TimestampColumn::new(
        id,
        values,
        ValidityBitmap::from_bytes(validity, len)?,
    )?))
}

pub(super) fn concat_categorical(chunks: &[OwnedColumn]) -> Result<OwnedColumn, DataError> {
    let mut codes = Vec::new();
    let mut validity = Vec::new();
    let mut len = 0usize;
    let OwnedColumn::Categorical(first) = &chunks[0] else {
        return Err(DataError::TypeMismatch { id: chunks[0].id(), expected: "dictionary" });
    };
    let mut remapped = 0usize;
    for chunk in chunks {
        let OwnedColumn::Categorical(column) = chunk else {
            return Err(DataError::TypeMismatch { id: chunk.id(), expected: "dictionary" });
        };
        codes.extend_from_slice(&column.codes);
        remapped += column.n_remapped_unknown;
        len = append_validity(&mut validity, len, &column.validity)?;
    }
    let mut owned = CategoricalColumn::try_new(
        first.id,
        codes,
        ValidityBitmap::from_bytes(validity, len)?,
        Arc::clone(&first.domain),
    )?;
    owned.n_remapped_unknown = remapped;
    Ok(OwnedColumn::Categorical(owned))
}

pub(super) fn concat_vectors(chunks: &[OwnedColumn]) -> Result<OwnedColumn, DataError> {
    let OwnedColumn::FixedVector(first) = &chunks[0] else {
        return Err(DataError::TypeMismatch { id: chunks[0].id(), expected: "fixed vector" });
    };
    let mut values = Vec::new();
    let mut validity = Vec::new();
    let mut len = 0usize;
    for chunk in chunks {
        let OwnedColumn::FixedVector(column) = chunk else {
            return Err(DataError::TypeMismatch { id: chunk.id(), expected: "fixed vector" });
        };
        values.extend_from_slice(&column.values);
        len = append_validity(&mut validity, len, &column.validity)?;
    }
    Ok(OwnedColumn::FixedVector(FixedVectorColumn::new(
        first.id,
        first.dim,
        values,
        ValidityBitmap::from_bytes(validity, len)?,
    )?))
}

pub(super) fn append_validity(
    bytes: &mut Vec<u8>,
    len: usize,
    source: &ValidityBitmap,
) -> Result<usize, DataError> {
    let new_len = len.checked_add(source.len()).ok_or_else(|| DataError::InvalidArgument {
        message: "Arrow chunk row counts overflow usize".into(),
    })?;
    bytes.resize(new_len.div_ceil(8), 0);
    for row in 0..source.len() {
        if source.is_valid(row) {
            let bit = len + row;
            bytes[bit / 8] |= 1 << (bit % 8);
        }
    }
    Ok(new_len)
}
