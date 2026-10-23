//! Arrow record-batch columns to library-owned chunk columns, without lossy casts.
//!
//! Floats keep their IEEE-754 bits (a 32- or 16-bit float widens exactly);
//! integers become exact `i64` (an unsigned value above `i64::MAX` is refused);
//! dates and timestamps become exact nanoseconds (an overflow is refused);
//! dictionaries and strings become category codes. Nulls are validity masks:
//! a null float cell holds `NaN` under an invalid bit, so the column follows
//! the invalid-exactly-when-NaN rule the missing-data policies read.
//!
//! Category codes are stable for the whole read. The domain grows by
//! appending: a dictionary contributes its values in dictionary order, a
//! string column its labels in order of first appearance. A file whose
//! batches share one dictionary keeps that dictionary as its domain, unused
//! levels included. An ordered dictionary must be the same in every batch.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use antecedent_core::{CategoryDomainId, VariableId};
use arrow_array::cast::AsArray;
use arrow_array::types::{
    ArrowDictionaryKeyType, Date32Type, Date64Type, Float16Type, Float32Type, Float64Type,
    Int8Type, Int16Type, Int32Type, Int64Type, TimestampMicrosecondType, TimestampMillisecondType,
    TimestampNanosecondType, TimestampSecondType, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use arrow_array::{Array, ArrowPrimitiveType, DictionaryArray};
use arrow_schema::{DataType, TimeUnit};

use crate::arrow_ffi::validity_from_array;
use crate::categorical::{
    CategoricalColumn, CategoryCode, CategoryDomain, CategoryLevel, UnknownCategoryPolicy,
};
use crate::column::{BooleanColumn, Float64Column, Int64Column, OwnedColumn, TimestampColumn};
use crate::global_index::GlobalId;

use super::error::FileSourceError;
use super::identity::ColumnHasher;
use super::{FileColumn, FileColumnKind};

const NANOS_PER_DAY: i64 = 86_400_000_000_000;

/// Label of the one placeholder level of a category column with no observed level yet.
///
/// A category domain needs a level. Every row under the placeholder is invalid.
pub(crate) const NO_LEVEL_PLACEHOLDER: &str = "";

/// Conversion and hashing state of one projected column for one read.
pub(crate) struct ColumnState {
    pub(crate) column: FileColumn,
    id: VariableId,
    hasher: ColumnHasher,
    categories: Option<CategoryState>,
}

struct CategoryState {
    domain_id: CategoryDomainId,
    ordered: bool,
    labels: Vec<Arc<str>>,
    index: HashMap<Arc<str>, u32>,
    /// The dictionary an ordered column fixed at its first batch.
    fixed: Option<Vec<Option<Arc<str>>>>,
    snapshot: Option<Arc<CategoryDomain>>,
}

impl ColumnState {
    pub(crate) fn new(column: FileColumn, position: u32) -> Self {
        let categories = match column.kind {
            FileColumnKind::Categorical { ordered } => Some(CategoryState {
                domain_id: CategoryDomainId::from_raw(position),
                ordered,
                labels: Vec::new(),
                index: HashMap::new(),
                fixed: None,
                snapshot: None,
            }),
            _ => None,
        };
        Self { column, id: VariableId::from_raw(position), hasher: ColumnHasher::new(), categories }
    }

    /// Start a new pass. Category codes already assigned stay assigned.
    pub(crate) fn restart(&mut self) {
        self.hasher = ColumnHasher::new();
    }

    /// Finish this pass's digest.
    pub(crate) fn take_digest(&mut self) -> [u8; 32] {
        std::mem::replace(&mut self.hasher, ColumnHasher::new()).finish()
    }

    /// Category labels in domain order. Empty for other kinds.
    pub(crate) fn levels(&self) -> &[Arc<str>] {
        self.categories.as_ref().map_or(&[], |state| state.labels.as_slice())
    }

    /// Domain of every code assigned so far.
    pub(crate) fn domain(&mut self) -> Result<Option<Arc<CategoryDomain>>, FileSourceError> {
        match self.categories.as_mut() {
            None => Ok(None),
            Some(state) => state.domain().map(Some),
        }
    }

    /// Convert `array` (already sliced to the chunk) whose first row is global row `start`.
    #[allow(clippy::too_many_lines, reason = "one arm per exact Arrow source type")]
    pub(crate) fn convert(
        &mut self,
        array: &dyn Array,
        start: GlobalId,
    ) -> Result<OwnedColumn, FileSourceError> {
        let name = self.column.name.clone();
        let id = self.id;
        let column = match array.data_type() {
            DataType::Float64 => floats(
                id,
                array.as_primitive::<Float64Type>(),
                &mut self.hasher,
                std::convert::identity,
            )?,
            DataType::Float32 => {
                floats(id, array.as_primitive::<Float32Type>(), &mut self.hasher, f64::from)?
            }
            DataType::Float16 => {
                floats(id, array.as_primitive::<Float16Type>(), &mut self.hasher, f64::from)?
            }
            DataType::Int8 => signed(id, array.as_primitive::<Int8Type>(), &mut self.hasher)?,
            DataType::Int16 => signed(id, array.as_primitive::<Int16Type>(), &mut self.hasher)?,
            DataType::Int32 => signed(id, array.as_primitive::<Int32Type>(), &mut self.hasher)?,
            DataType::Int64 => signed(id, array.as_primitive::<Int64Type>(), &mut self.hasher)?,
            DataType::UInt8 => {
                unsigned(id, array.as_primitive::<UInt8Type>(), &mut self.hasher, &name, start)?
            }
            DataType::UInt16 => {
                unsigned(id, array.as_primitive::<UInt16Type>(), &mut self.hasher, &name, start)?
            }
            DataType::UInt32 => {
                unsigned(id, array.as_primitive::<UInt32Type>(), &mut self.hasher, &name, start)?
            }
            DataType::UInt64 => {
                unsigned(id, array.as_primitive::<UInt64Type>(), &mut self.hasher, &name, start)?
            }
            DataType::Boolean => booleans(id, array, &mut self.hasher)?,
            DataType::Timestamp(unit, _) => {
                let scale = match unit {
                    TimeUnit::Nanosecond => 1,
                    TimeUnit::Microsecond => 1_000,
                    TimeUnit::Millisecond => 1_000_000,
                    TimeUnit::Second => 1_000_000_000,
                };
                let raw: Vec<Option<i64>> = match unit {
                    TimeUnit::Nanosecond => {
                        raw_values(array.as_primitive::<TimestampNanosecondType>())
                    }
                    TimeUnit::Microsecond => {
                        raw_values(array.as_primitive::<TimestampMicrosecondType>())
                    }
                    TimeUnit::Millisecond => {
                        raw_values(array.as_primitive::<TimestampMillisecondType>())
                    }
                    TimeUnit::Second => raw_values(array.as_primitive::<TimestampSecondType>()),
                };
                timestamps(id, array, &raw, scale, &mut self.hasher, &name, start)?
            }
            DataType::Date32 => {
                let raw: Vec<Option<i64>> = array
                    .as_primitive::<Date32Type>()
                    .iter()
                    .map(|value| value.map(i64::from))
                    .collect();
                timestamps(id, array, &raw, NANOS_PER_DAY, &mut self.hasher, &name, start)?
            }
            DataType::Date64 => {
                let raw = raw_values(array.as_primitive::<Date64Type>());
                timestamps(id, array, &raw, 1_000_000, &mut self.hasher, &name, start)?
            }
            DataType::Utf8 => {
                let text = array.as_string::<i32>();
                let labels: Vec<Option<&str>> = text.iter().collect();
                self.text_column(array, &labels)?
            }
            DataType::LargeUtf8 => {
                let text = array.as_string::<i64>();
                let labels: Vec<Option<&str>> = text.iter().collect();
                self.text_column(array, &labels)?
            }
            DataType::Utf8View => {
                let text = array.as_string_view();
                let labels: Vec<Option<&str>> = text.iter().collect();
                self.text_column(array, &labels)?
            }
            DataType::Dictionary(key, _) => match key.as_ref() {
                DataType::Int8 => self.dictionary(array.as_dictionary::<Int8Type>(), start)?,
                DataType::Int16 => self.dictionary(array.as_dictionary::<Int16Type>(), start)?,
                DataType::Int32 => self.dictionary(array.as_dictionary::<Int32Type>(), start)?,
                DataType::Int64 => self.dictionary(array.as_dictionary::<Int64Type>(), start)?,
                DataType::UInt8 => self.dictionary(array.as_dictionary::<UInt8Type>(), start)?,
                DataType::UInt16 => self.dictionary(array.as_dictionary::<UInt16Type>(), start)?,
                DataType::UInt32 => self.dictionary(array.as_dictionary::<UInt32Type>(), start)?,
                other => return Err(unsupported_type(&name, other)),
            },
            other => return Err(unsupported_type(&name, other)),
        };
        Ok(column)
    }

    fn text_column(
        &mut self,
        array: &dyn Array,
        labels: &[Option<&str>],
    ) -> Result<OwnedColumn, FileSourceError> {
        let state = self.categories.as_mut().ok_or_else(|| not_categorical(&self.column))?;
        let mut codes = Vec::with_capacity(labels.len());
        for label in labels {
            match label {
                None => {
                    self.hasher.null();
                    codes.push(CategoryCode::from_raw(0));
                }
                Some(label) => {
                    self.hasher.label(label);
                    codes.push(CategoryCode::from_raw(state.admit(label)?));
                }
            }
        }
        let (validity, _) = validity_from_array(array)?;
        let domain = state.domain()?;
        Ok(OwnedColumn::Categorical(CategoricalColumn::try_new(self.id, codes, validity, domain)?))
    }

    fn dictionary<K: ArrowDictionaryKeyType>(
        &mut self,
        array: &DictionaryArray<K>,
        start: GlobalId,
    ) -> Result<OwnedColumn, FileSourceError> {
        let values = array.values();
        let mut dictionary: Vec<Option<Arc<str>>> = Vec::with_capacity(values.len());
        for row in 0..values.len() {
            if values.is_null(row) {
                dictionary.push(None);
            } else {
                let text = crate::arrow_adapter::level_text(values.as_ref(), row)?;
                dictionary.push(Some(Arc::from(text)));
            }
        }
        let name = self.column.name.clone();
        let state = self.categories.as_mut().ok_or_else(|| not_categorical(&self.column))?;
        let remap = state.admit_dictionary(&dictionary, &name, start)?;
        let mut codes = Vec::with_capacity(array.len());
        let mut validity = vec![0_u8; array.len().div_ceil(8)];
        for row in 0..array.len() {
            let local = array.key(row);
            let code = local.and_then(|key| remap.get(key).copied().flatten());
            if let (Some(key), Some(code)) = (local, code) {
                let label = dictionary[key].as_deref().unwrap_or(NO_LEVEL_PLACEHOLDER);
                self.hasher.label(label);
                validity[row / 8] |= 1 << (row % 8);
                codes.push(CategoryCode::from_raw(code));
            } else {
                self.hasher.null();
                codes.push(CategoryCode::from_raw(0));
            }
        }
        let validity = crate::column::ValidityBitmap::from_bytes(validity, array.len())?;
        let domain = state.domain()?;
        Ok(OwnedColumn::Categorical(CategoricalColumn::try_new(self.id, codes, validity, domain)?))
    }
}

impl CategoryState {
    fn admit(&mut self, label: &str) -> Result<u32, FileSourceError> {
        if let Some(&code) = self.index.get(label) {
            return Ok(code);
        }
        let code = u32::try_from(self.labels.len())
            .map_err(|_| FileSourceError::CountOverflow { context: "category level count" })?;
        let label: Arc<str> = Arc::from(label);
        self.labels.push(Arc::clone(&label));
        self.index.insert(label, code);
        self.snapshot = None;
        Ok(code)
    }

    fn admit_dictionary(
        &mut self,
        dictionary: &[Option<Arc<str>>],
        column: &str,
        start: GlobalId,
    ) -> Result<Vec<Option<u32>>, FileSourceError> {
        if self.ordered {
            if self.fixed.is_none() {
                self.fixed = Some(dictionary.to_vec());
            } else if self.fixed.as_deref() != Some(dictionary) {
                return Err(FileSourceError::UnsupportedValue {
                    column: column.to_owned(),
                    row: start,
                    detail: "the ordered dictionary changed between record batches, so its level \
                             order is not one order"
                        .to_owned(),
                    next_attempt: "write one dictionary for the column, or read it as an \
                                   unordered category",
                });
            }
        }
        dictionary
            .iter()
            .map(|label| label.as_deref().map(|label| self.admit(label)).transpose())
            .collect()
    }

    fn domain(&mut self) -> Result<Arc<CategoryDomain>, FileSourceError> {
        if let Some(domain) = &self.snapshot {
            return Ok(Arc::clone(domain));
        }
        let levels: Vec<CategoryLevel> = if self.labels.is_empty() {
            vec![CategoryLevel { label: Arc::from(NO_LEVEL_PLACEHOLDER) }]
        } else {
            self.labels.iter().map(|label| CategoryLevel { label: Arc::clone(label) }).collect()
        };
        let domain = Arc::new(CategoryDomain::try_new(
            self.domain_id,
            levels,
            self.ordered,
            None,
            UnknownCategoryPolicy::Fail,
        )?);
        self.snapshot = Some(Arc::clone(&domain));
        Ok(domain)
    }
}

fn raw_values<T>(array: &arrow_array::PrimitiveArray<T>) -> Vec<Option<i64>>
where
    T: ArrowPrimitiveType<Native = i64>,
{
    array.iter().collect()
}

fn floats<T>(
    id: VariableId,
    array: &arrow_array::PrimitiveArray<T>,
    hasher: &mut ColumnHasher,
    widen: impl Fn(T::Native) -> f64,
) -> Result<OwnedColumn, FileSourceError>
where
    T: ArrowPrimitiveType,
{
    let mut values = Vec::with_capacity(array.len());
    for value in array {
        match value {
            None => {
                hasher.null();
                values.push(f64::NAN);
            }
            Some(raw) => {
                let value = widen(raw);
                hasher.value(&value.to_bits().to_le_bytes());
                values.push(value);
            }
        }
    }
    let (validity, _) = validity_from_array(array)?;
    Ok(OwnedColumn::Float64(Float64Column::new(id, values, validity)?))
}

fn signed<T>(
    id: VariableId,
    array: &arrow_array::PrimitiveArray<T>,
    hasher: &mut ColumnHasher,
) -> Result<OwnedColumn, FileSourceError>
where
    T: ArrowPrimitiveType,
    T::Native: Into<i64>,
{
    let mut values = Vec::with_capacity(array.len());
    for value in array {
        match value {
            None => {
                hasher.null();
                values.push(0);
            }
            Some(raw) => {
                let value: i64 = raw.into();
                hasher.value(&value.to_le_bytes());
                values.push(value);
            }
        }
    }
    let (validity, _) = validity_from_array(array)?;
    Ok(OwnedColumn::Int64(Int64Column::new(id, values, validity)?))
}

fn unsigned<T>(
    id: VariableId,
    array: &arrow_array::PrimitiveArray<T>,
    hasher: &mut ColumnHasher,
    column: &str,
    start: GlobalId,
) -> Result<OwnedColumn, FileSourceError>
where
    T: ArrowPrimitiveType,
    T::Native: Into<u64>,
{
    let mut values = Vec::with_capacity(array.len());
    for (row, value) in array.iter().enumerate() {
        match value {
            None => {
                hasher.null();
                values.push(0);
            }
            Some(raw) => {
                let raw: u64 = raw.into();
                let value = i64::try_from(raw).map_err(|_| FileSourceError::UnsupportedValue {
                    column: column.to_owned(),
                    row: offset(start, row),
                    detail: format!("unsigned value {raw} does not fit an exact int64"),
                    next_attempt: "store the column as a signed 64-bit integer or as a category \
                                   label",
                })?;
                hasher.value(&value.to_le_bytes());
                values.push(value);
            }
        }
    }
    let (validity, _) = validity_from_array(array)?;
    Ok(OwnedColumn::Int64(Int64Column::new(id, values, validity)?))
}

fn booleans(
    id: VariableId,
    array: &dyn Array,
    hasher: &mut ColumnHasher,
) -> Result<OwnedColumn, FileSourceError> {
    let flags = array.as_boolean();
    let mut values = Vec::with_capacity(flags.len());
    for value in flags {
        match value {
            None => {
                hasher.null();
                values.push(0_u8);
            }
            Some(flag) => {
                let byte = u8::from(flag);
                hasher.value(&[byte]);
                values.push(byte);
            }
        }
    }
    let (validity, _) = validity_from_array(array)?;
    Ok(OwnedColumn::Boolean(BooleanColumn::new(id, values, validity)?))
}

fn timestamps(
    id: VariableId,
    array: &dyn Array,
    raw: &[Option<i64>],
    scale: i64,
    hasher: &mut ColumnHasher,
    column: &str,
    start: GlobalId,
) -> Result<OwnedColumn, FileSourceError> {
    let mut values = Vec::with_capacity(raw.len());
    for (row, value) in raw.iter().enumerate() {
        match value {
            None => {
                hasher.null();
                values.push(0);
            }
            Some(raw) => {
                let nanos =
                    raw.checked_mul(scale).ok_or_else(|| FileSourceError::UnsupportedValue {
                        column: column.to_owned(),
                        row: offset(start, row),
                        detail: format!("time value {raw} overflows exact int64 nanoseconds"),
                        next_attempt: "store the column at a coarser unit as an integer count",
                    })?;
                hasher.value(&nanos.to_le_bytes());
                values.push(nanos);
            }
        }
    }
    let (validity, _) = validity_from_array(array)?;
    Ok(OwnedColumn::Timestamp(TimestampColumn::new(id, values, validity)?))
}

fn offset(start: GlobalId, row: usize) -> GlobalId {
    GlobalId::new(start.get().saturating_add(row as u64))
}

fn not_categorical(column: &FileColumn) -> FileSourceError {
    FileSourceError::SchemaMismatch {
        path: std::path::PathBuf::new(),
        detail: format!("column '{}' changed from {} to a category", column.name, column.kind),
    }
}

/// Refusal for a column type the reader cannot keep exactly.
pub(crate) fn unsupported_type(column: &str, data_type: &DataType) -> FileSourceError {
    FileSourceError::UnsupportedColumn {
        column: column.to_owned(),
        arrow_type: data_type.to_string(),
        next_attempt: unsupported_next_attempt(data_type),
    }
}

/// What to do with a column type the reader refuses.
pub(crate) fn unsupported_next_attempt(data_type: &DataType) -> &'static str {
    match data_type {
        DataType::Decimal128(..) | DataType::Decimal256(..) => {
            "cast the decimal column explicitly: to float64, accepting the rounding, or to an \
             integer count of its smallest unit"
        }
        DataType::List(_)
        | DataType::LargeList(_)
        | DataType::ListView(_)
        | DataType::LargeListView(_)
        | DataType::FixedSizeList(..)
        | DataType::Struct(_)
        | DataType::Map(..)
        | DataType::Union(..)
        | DataType::RunEndEncoded(..) => {
            "flatten the nested column into scalar columns before reading, or leave it out of \
             the required columns"
        }
        DataType::Binary
        | DataType::LargeBinary
        | DataType::BinaryView
        | DataType::FixedSizeBinary(_) => {
            "decode the binary column into labels or numbers, or leave it out of the required \
             columns"
        }
        DataType::Null => "drop the all-null column, or leave it out of the required columns",
        DataType::Dictionary(..) => {
            "re-encode the dictionary with integer keys of at most 32 bits and string or integer \
             values"
        }
        _ => {
            "convert the column to a float, an integer, a boolean, a timestamp, or a category \
             label, or leave it out of the required columns"
        }
    }
}
