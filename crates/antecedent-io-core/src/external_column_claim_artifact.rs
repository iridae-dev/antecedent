//! Portable artifact for a row-level column bound to a checked contract.
//!
//! A response grid names one coordinate per value; a column of thousands of
//! rows (one conditional effect per unit) has one scientific quantity and a
//! value per row. The artifact keeps that quantity, the provider's identity,
//! lineage and trust, and the column's size, digest and support counts. It
//! does not carry the rows: the consumer holds the column and checks it against
//! the digest, so a changed value is a changed claim. Native estimation is
//! never claimed.

use antecedent_contract::{BoundExternalClaim, ColumnClaim};
use serde::{Deserialize, Serialize};

use crate::convert::{from_cbor, to_cbor};
use crate::error::IoError;
use crate::external_claim_artifact::{ExternalClaimIdentity, values_digest};

/// Maximum artifact bytes accepted before decode allocation.
pub const MAX_EXTERNAL_COLUMN_CLAIM_ARTIFACT_BYTES: usize = 1024 * 1024;
/// Maximum rows one column claim covers.
pub const MAX_EXTERNAL_COLUMN_ROWS: u64 = 16 * 1024 * 1024;
const ARTIFACT_KIND: &str = crate::artifact_kinds::EXTERNAL_COLUMN_CLAIM;
const META_SECTION: &str = "external_column_claim.meta";
const LABELS: [&str; 5] =
    ["supported", "weak_overlap", "extrapolative", "outside_empirical_support", "missing_evidence"];

/// Rows carrying one support label.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ColumnSupportCountWire {
    /// `SupportStatus` name.
    pub status: String,
    /// Rows with this label.
    pub rows: u64,
}

/// The column facts a consumer must agree on, beside the shared claim identity.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ColumnIdentityWire {
    /// Number of rows.
    pub rows: u64,
    /// Element type of the column; always `f64`.
    pub dtype: String,
    /// Rows per support label, weakest label last, labels with no rows omitted.
    pub support: Vec<ColumnSupportCountWire>,
}

/// Everything a consumer must independently agree on before trusting a column.
///
/// `claim.values_blake3` is the digest of the column's values, so the column
/// itself is part of the identity.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalColumnIdentity {
    /// Provider, contract, lineage, trust and the one row-level quantity.
    pub claim: ExternalClaimIdentity,
    /// Size, element type and support of the column.
    pub column: ColumnIdentityWire,
}

impl ExternalColumnIdentity {
    fn from_claim(claim: &BoundExternalClaim, causal_contract_id: &str) -> Result<Self, IoError> {
        let Some(column) = claim.column() else {
            return Err(IoError::Convert("claim is not a row-level column".into()));
        };
        let value = Self {
            claim: ExternalClaimIdentity::from_claim_with_digest(
                claim,
                causal_contract_id,
                column.values_blake3().to_owned(),
            )?,
            column: ColumnIdentityWire::from(column),
        };
        value.validate()?;
        Ok(value)
    }

    fn validate(&self) -> Result<(), IoError> {
        self.claim.validate()?;
        let column = &self.column;
        if self.claim.quantities.len() != 1 {
            return Err(IoError::Convert("a column claim has exactly one quantity".into()));
        }
        if column.dtype != ColumnClaim::DTYPE {
            return Err(IoError::Convert(format!("unsupported column dtype `{}`", column.dtype)));
        }
        if column.rows == 0 || column.rows > MAX_EXTERNAL_COLUMN_ROWS {
            return Err(IoError::TooLarge);
        }
        let positions: Option<Vec<usize>> = column
            .support
            .iter()
            .map(|count| LABELS.iter().position(|label| *label == count.status))
            .collect();
        let ordered = positions.is_some_and(|at| at.windows(2).all(|pair| pair[0] < pair[1]));
        let total = column.support.iter().try_fold(0_u64, |sum, count| sum.checked_add(count.rows));
        if !ordered
            || column.support.is_empty()
            || column.support.iter().any(|count| count.rows == 0)
            || total != Some(column.rows)
        {
            return Err(IoError::Convert("invalid column support counts".into()));
        }
        let worst = column.support.last().map(|count| count.status.as_str());
        if self.claim.point_status.len() != 1 || Some(self.claim.point_status[0].as_str()) != worst
        {
            return Err(IoError::Convert("column support label is not its worst row".into()));
        }
        Ok(())
    }
}

impl From<&ColumnClaim> for ColumnIdentityWire {
    fn from(column: &ColumnClaim) -> Self {
        Self {
            rows: column.rows() as u64,
            dtype: ColumnClaim::DTYPE.to_owned(),
            support: column
                .support_counts()
                .iter()
                .map(|(status, rows)| ColumnSupportCountWire {
                    status: status.as_str().to_owned(),
                    rows: *rows as u64,
                })
                .collect(),
        }
    }
}

/// Stored metadata of a column claim.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalColumnClaimMetadata {
    /// Metadata format version.
    pub version: u16,
    /// Complete scientific, provider and column identity.
    pub identity: ExternalColumnIdentity,
    /// Always false: the numbers were computed outside Antecedent.
    pub native_estimation: bool,
}

/// A validated column claim: identity and digest, never the rows.
#[derive(Clone, Debug, PartialEq)]
pub struct ExternalColumnClaimArtifact {
    metadata: ExternalColumnClaimMetadata,
}

fn validate(meta: &ExternalColumnClaimMetadata) -> Result<(), IoError> {
    if meta.version != 1 {
        return Err(IoError::UnsupportedVersion { version: u32::from(meta.version) });
    }
    if meta.native_estimation {
        return Err(IoError::Convert("external claim cannot be native estimation".into()));
    }
    meta.identity.validate()
}

impl ExternalColumnClaimArtifact {
    /// The column claim of a bound row-level result.
    ///
    /// # Errors
    /// A claim that is not a column, a blank contract id or an invalid
    /// coordinate refuses.
    pub fn from_bound_claim(
        claim: &BoundExternalClaim,
        causal_contract_id: &str,
    ) -> Result<Self, IoError> {
        let metadata = ExternalColumnClaimMetadata {
            version: 1,
            identity: ExternalColumnIdentity::from_claim(claim, causal_contract_id)?,
            native_estimation: false,
        };
        validate(&metadata)?;
        Ok(Self { metadata })
    }

    /// Bound metadata.
    #[must_use]
    pub fn metadata(&self) -> &ExternalColumnClaimMetadata {
        &self.metadata
    }

    /// Name of the external execution that produced the numbers.
    #[must_use]
    pub fn provenance_label(&self) -> String {
        let id = &self.metadata.identity.claim;
        format!("external:{}/{}@{}#{}", id.provider_id, id.object_id, id.version_id, id.snapshot_id)
    }

    /// Whether `values` is exactly the column this claim is bound to.
    #[must_use]
    pub fn covers(&self, values: &[f64]) -> bool {
        values.len() as u64 == self.metadata.identity.column.rows
            && values_digest(values) == self.metadata.identity.claim.values_blake3
    }

    /// Serialize through the checksummed sectioned container.
    ///
    /// # Errors
    /// A blank artifact id or an oversized payload refuses.
    pub fn to_bytes(&self, artifact_id: &str) -> Result<Vec<u8>, IoError> {
        validate(&self.metadata)?;
        if artifact_id.trim().is_empty() {
            return Err(IoError::Convert("missing artifact id".into()));
        }
        let meta_bytes = to_cbor(&self.metadata)?;
        crate::encode_container(
            &crate::ArtifactContainer {
                kind: ARTIFACT_KIND,
                note: "external_column_claim",
                max_bytes: MAX_EXTERNAL_COLUMN_CLAIM_ARTIFACT_BYTES,
                artifact_id,
                library_version: env!("CARGO_PKG_VERSION"),
            },
            &[crate::OutSection {
                id: META_SECTION,
                content_type: crate::CONTENT_CBOR,
                bytes: &meta_bytes,
            }],
        )
        .map_err(|error| error.into_io("unsupported external column claim layout"))
    }

    /// Load and check against an independently retained identity.
    ///
    /// # Errors
    /// Oversized, truncated, corrupt, incompatible or semantically changed
    /// artifacts refuse.
    pub fn from_bytes(bytes: &[u8], expected: &ExternalColumnIdentity) -> Result<Self, IoError> {
        Self::from_bytes_within(bytes, expected, MAX_EXTERNAL_COLUMN_CLAIM_ARTIFACT_BYTES)
    }

    /// [`Self::from_bytes`] under the caller's byte bound instead of [`MAX_EXTERNAL_COLUMN_CLAIM_ARTIFACT_BYTES`].
    ///
    /// # Errors
    /// As [`Self::from_bytes`]; a file over `max_bytes` is `TooLarge`.
    pub fn from_bytes_within(
        bytes: &[u8],
        expected: &ExternalColumnIdentity,
        max_bytes: usize,
    ) -> Result<Self, IoError> {
        if bytes.len() > max_bytes {
            return Err(IoError::TooLarge);
        }
        expected.validate()?;
        let parts = crate::decode_container(
            ARTIFACT_KIND,
            max_bytes,
            &[META_SECTION],
            bytes,
        )
        .map_err(|error| error.into_io("unsupported external column claim layout"))?;
        let Ok([meta_bytes]) = <[Vec<u8>; 1]>::try_from(parts) else {
            return Err(IoError::Convert("unsupported external column claim layout".into()));
        };
        let metadata: ExternalColumnClaimMetadata = from_cbor(&meta_bytes)?;
        if metadata.identity != *expected {
            return Err(IoError::Refused {
                code: antecedent_core::reason_code!("external_binding_mismatch"),
                message: "external_column_claim.identity_expected: claim identity differs from the consumer's checked contract".into(),
            });
        }
        validate(&metadata)?;
        Ok(Self { metadata })
    }
}
