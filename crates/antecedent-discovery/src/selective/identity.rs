//! Canonical identities of the inputs a discovery stage depends on.
//!
//! Every digest is a length-delimited [`CanonicalWriter`] stream under a versioned domain
//! tag. Floats enter as bit patterns. Nothing is derived from `Debug` or from any text
//! formatting of a value. Stable caller labels and object addresses do not prove unchanged
//! content, so the data identity is the content of the columns the search reads.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{CanonicalWriter, ScalarType, ValueType, VariableId, to_hex};
use antecedent_data::{
    OwnedColumnarStorage, SamplingRegularity, TableView, TabularData, TimeSeriesData,
};

use crate::error::DiscoveryError;

/// Version word written into every selective-discovery identity.
pub const SELECTIVE_IDENTITY_VERSION: u64 = 1;

const COLUMN_TAG: &str = "antecedent.selective_discovery.column.v1";
const DATA_TAG: &str = "antecedent.selective_discovery.data.v1";

/// One 32-byte BLAKE3 identity.
pub type Digest = [u8; 32];

/// Digest of one column's values as bit patterns, in row order.
#[must_use]
pub fn column_digest(values: &[f64]) -> Digest {
    let mut bytes = Vec::with_capacity(values.len().saturating_mul(8));
    for value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    CanonicalWriter::digest_bytes(COLUMN_TAG, &bytes)
}

/// Lowercase hex of a digest.
#[must_use]
pub fn digest_hex(digest: &Digest) -> String {
    to_hex(digest)
}

/// Parse 64 lowercase hex characters.
#[must_use]
pub fn digest_from_hex(text: &str) -> Option<Digest> {
    let bytes = text.as_bytes();
    if bytes.len() != 64 {
        return None;
    }
    let nibble = |c: u8| match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        _ => None,
    };
    let mut out = [0u8; 32];
    for (slot, pair) in out.iter_mut().zip(bytes.chunks_exact(2)) {
        *slot = (nibble(pair[0])? << 4) | nibble(pair[1])?;
    }
    Some(out)
}

fn scalar_tag(scalar: ScalarType) -> u64 {
    match scalar {
        ScalarType::Float64 => 0,
        ScalarType::Float32 => 1,
        ScalarType::Int64 => 2,
        ScalarType::Int32 => 3,
    }
}

/// `(tag, width, element)` of a value type, for identity streams.
fn value_type_words(value_type: &ValueType) -> (u64, u64, u64) {
    match value_type {
        ValueType::Continuous => (0, 0, 0),
        ValueType::Count => (1, 0, 0),
        ValueType::Binary => (2, 0, 0),
        ValueType::Categorical => (3, 0, 0),
        ValueType::Ordinal => (4, 0, 0),
        ValueType::Unspecified => (5, 0, 0),
        ValueType::Vector { width, element } => (6, u64::from(width.get()), scalar_tag(*element)),
    }
}

/// What a stage depends on in the data: row count, the searched columns' content and types,
/// the analysis mask and weights, and (temporal) the sampling index.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DataIdentity {
    /// Rows (static) or time points (temporal).
    pub rows: u64,
    /// Searched variables in search order with the content digest of each column.
    pub columns: Vec<(VariableId, Digest)>,
    /// Digest of everything above plus types, names, mask, weights and sampling index.
    pub digest: Digest,
}

impl DataIdentity {
    /// Variables whose content differs from `other`, ascending; empty when the digests of
    /// every shared column agree. A variable present in only one side is reported too.
    #[must_use]
    pub fn changed_variables(&self, other: &Self) -> Vec<VariableId> {
        let mut changed = Vec::new();
        for (variable, digest) in &self.columns {
            match other.columns.iter().find(|(v, _)| v == variable) {
                Some((_, theirs)) if theirs == digest => {}
                _ => changed.push(*variable),
            }
        }
        for (variable, _) in &other.columns {
            if !self.columns.iter().any(|(v, _)| v == variable) {
                changed.push(*variable);
            }
        }
        changed.sort_unstable();
        changed.dedup();
        changed
    }
}

fn identity_of<T: TableView>(
    view: &T,
    storage: &OwnedColumnarStorage,
    variables: &[VariableId],
    sampling: Option<(u64, u64, u64)>,
) -> Result<DataIdentity, DiscoveryError> {
    let schema = view.schema();
    let rows = view.row_count();
    let mut columns = Vec::with_capacity(variables.len());
    let mut writer = CanonicalWriter::new(DATA_TAG);
    writer.word(SELECTIVE_IDENTITY_VERSION).word(rows as u64).word(variables.len() as u64);
    for &variable in variables {
        let spec = schema
            .variables()
            .get(variable.raw() as usize)
            .filter(|spec| spec.id == variable)
            .ok_or_else(|| DiscoveryError::data_msg("variable is not in the data schema"))?;
        let values = view.float64_values(variable).map_err(DiscoveryError::from)?;
        if values.len() != rows {
            return Err(DiscoveryError::data_msg("column length differs from the row count"));
        }
        let digest = column_digest(&values);
        let (tag, width, element) = value_type_words(&spec.value_type);
        writer
            .word(u64::from(variable.raw()))
            .text(&spec.name)
            .word(tag)
            .word(width)
            .word(element)
            .bytes(&digest);
        columns.push((variable, digest));
    }
    match storage.analysis_mask() {
        Some(mask) => {
            let bits: Vec<u8> = (0..mask.len()).map(|i| u8::from(mask.is_valid(i))).collect();
            writer.word(1).bytes(&bits);
        }
        None => {
            writer.word(0);
        }
    }
    match storage.weights() {
        Some(weights) => {
            writer.word(1).word(weights.len() as u64);
            writer.bytes(&column_digest(weights));
        }
        None => {
            writer.word(0);
        }
    }
    match sampling {
        Some((kind, interval, length)) => {
            writer.word(1).word(kind).word(interval).word(length);
        }
        None => {
            writer.word(0);
        }
    }
    Ok(DataIdentity { rows: rows as u64, columns, digest: writer.finish() })
}

/// Identity of a static table over `variables` (search order).
///
/// # Errors
///
/// A variable that is not in the schema or a column that is not numeric.
pub fn static_data_identity(
    data: &TabularData,
    variables: &[VariableId],
) -> Result<DataIdentity, DiscoveryError> {
    identity_of(data, data.storage(), variables, None)
}

/// Identity of a time series over `variables` (search order), including the sampling index.
///
/// # Errors
///
/// A variable that is not in the schema or a column that is not numeric.
pub fn temporal_data_identity(
    data: &TimeSeriesData,
    variables: &[VariableId],
) -> Result<DataIdentity, DiscoveryError> {
    let index = data.time_index();
    let (kind, interval) = match index.regularity {
        SamplingRegularity::Regular { interval_ns } => (1, interval_ns),
        SamplingRegularity::Irregular => (2, 0),
    };
    identity_of(data, data.storage(), variables, Some((kind, interval, index.length as u64)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn column_digest_is_bit_exact_and_order_sensitive() {
        let a = column_digest(&[1.0, 2.0, 3.0]);
        assert_eq!(a, column_digest(&[1.0, 2.0, 3.0]));
        assert_ne!(a, column_digest(&[3.0, 2.0, 1.0]));
        assert_ne!(column_digest(&[0.0]), column_digest(&[-0.0]));
        assert_ne!(column_digest(&[1.0]), column_digest(&[1.0, 1.0]));
    }

    #[test]
    fn hex_round_trips_and_refuses_uppercase_or_short_text() {
        let digest = column_digest(&[0.5]);
        let text = digest_hex(&digest);
        assert_eq!(digest_from_hex(&text), Some(digest));
        assert_eq!(digest_from_hex(&text.to_ascii_uppercase()), None);
        assert_eq!(digest_from_hex("abc"), None);
    }
}
