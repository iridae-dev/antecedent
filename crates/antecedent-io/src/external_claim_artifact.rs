//! Portable artifact for a finite external response grid bound to a checked contract.
//!
//! The artifact states that the numbers were computed by a foreign provider,
//! names that execution, and carries the premises the binding stage checked.
//! A consumer supplies its own expected identity on load; a resealed change in
//! any scientific, provider, trust or premise field is refused. The artifact
//! never claims native estimation and grants no interval or calibration.

use antecedent_core::{
    BoundExternalClaim, BoundTrustLevel, ExternalUncertaintyMeaning, ScientificQuantity,
};
use serde::{Deserialize, Serialize};

use crate::container::{ArtifactManifest, EncodedArtifact, SectionBytes, section_descriptor};
use crate::convert::{from_cbor, to_cbor};
use crate::error::IoError;
use crate::quantity_wire::ScientificQuantityWire;
use crate::reader::ArtifactReader;
use crate::wire::{ArtifactKind, ProvenanceWire, SemanticVersion};

/// Maximum artifact bytes accepted before decode allocation.
pub const MAX_EXTERNAL_CLAIM_ARTIFACT_BYTES: usize = 4 * 1024 * 1024;
/// Maximum number of response coordinates.
pub const MAX_EXTERNAL_CLAIM_COORDINATES: usize = 100_000;
const ARTIFACT_KIND: &str = "external_response_claim_v1";
const META_SECTION: &str = "external_claim.meta";
const VALUES_SECTION: &str = "external_claim.values";

/// Trust label preserved without upgrading it during load.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExternalClaimTrust {
    /// Supplier assertion only.
    ExternallyAttested,
    /// The exact provider contract passed object-level verification.
    ExactRequestVerified,
}

impl From<BoundTrustLevel> for ExternalClaimTrust {
    fn from(level: BoundTrustLevel) -> Self {
        match level {
            BoundTrustLevel::ExternallyAttested => Self::ExternallyAttested,
            BoundTrustLevel::ExactRequestVerified => Self::ExactRequestVerified,
        }
    }
}

/// Everything a consumer must independently agree on before reading numbers.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalClaimIdentity {
    /// Independently checked causal-contract identity.
    pub causal_contract_id: String,
    /// Graph identity shared by contract and provider.
    pub graph_id: String,
    /// Structural identification status name.
    pub identification: String,
    /// Response coordinates in exact value order.
    pub quantities: Vec<ScientificQuantityWire>,
    /// Provider service identity.
    pub provider_id: String,
    /// Provider object identity.
    pub object_id: String,
    /// Provider version.
    pub version_id: String,
    /// Provider input or model snapshot.
    pub snapshot_id: String,
    /// Exact request fingerprint.
    pub request_id: String,
    /// Trust held by the provider object.
    pub trust: ExternalClaimTrust,
    /// Provider-declared uncertainty method; absent means no uncertainty claim.
    pub uncertainty_method: Option<String>,
    /// Evidence factors used.
    pub evidence_ids: Vec<String>,
    /// Assumptions relied on.
    pub assumption_ids: Vec<String>,
    /// Checked equivalences licensing an observational law.
    pub equivalence_ids: Vec<String>,
}

impl ExternalClaimIdentity {
    /// Identity of a bound claim under a named causal contract.
    ///
    /// # Errors
    /// Blank contract IDs, non-response claims and invalid coordinates refuse.
    pub fn from_claim(
        claim: &BoundExternalClaim,
        causal_contract_id: &str,
    ) -> Result<Self, IoError> {
        if claim.values().is_none() {
            return Err(IoError::Convert("claim is not a finite response grid".into()));
        }
        let execution = claim.execution();
        let value = Self {
            causal_contract_id: causal_contract_id.into(),
            graph_id: claim.graph_id().into(),
            identification: claim.identification().as_str().into(),
            quantities: claim.quantities().iter().map(ScientificQuantityWire::from).collect(),
            provider_id: execution.provider_id.clone(),
            object_id: execution.object_id.clone(),
            version_id: execution.version_id.clone(),
            snapshot_id: execution.snapshot_id.clone(),
            request_id: execution.request_id.clone(),
            trust: claim.trust().into(),
            uncertainty_method: match claim.uncertainty() {
                ExternalUncertaintyMeaning::None => None,
                ExternalUncertaintyMeaning::ProviderDeclared { method_id } => {
                    Some(method_id.clone())
                }
            },
            evidence_ids: claim.evidence_ids().to_vec(),
            assumption_ids: claim.assumption_ids().to_vec(),
            equivalence_ids: claim.equivalence_ids().to_vec(),
        };
        value.validate()?;
        Ok(value)
    }

    fn validate(&self) -> Result<(), IoError> {
        if self.quantities.is_empty() || self.quantities.len() > MAX_EXTERNAL_CLAIM_COORDINATES {
            return Err(IoError::TooLarge);
        }
        for quantity in &self.quantities {
            ScientificQuantity::try_from(quantity.clone())
                .map_err(|error| IoError::Convert(error.into()))?;
        }
        if [
            &self.causal_contract_id,
            &self.graph_id,
            &self.identification,
            &self.provider_id,
            &self.object_id,
            &self.version_id,
            &self.snapshot_id,
            &self.request_id,
        ]
        .into_iter()
        .any(|id| id.trim().is_empty())
        {
            return Err(IoError::Convert("missing external claim identity".into()));
        }
        Ok(())
    }
}

/// Stored metadata; values are a separate f64 LE section.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalClaimMetadata {
    /// Metadata format version.
    pub version: u16,
    /// Complete scientific, provider and premise identity.
    pub identity: ExternalClaimIdentity,
    /// Always false: the numbers were computed outside Antecedent.
    pub native_estimation: bool,
}

/// A validated finite response grid plus its bound identity.
#[derive(Clone, Debug, PartialEq)]
pub struct ExternalClaimArtifact {
    metadata: ExternalClaimMetadata,
    values: Vec<f64>,
}

fn validate(meta: &ExternalClaimMetadata, values: &[f64]) -> Result<(), IoError> {
    if meta.version != 1 {
        return Err(IoError::UnsupportedVersion { version: u32::from(meta.version) });
    }
    if meta.native_estimation {
        return Err(IoError::Convert("external claim cannot be native estimation".into()));
    }
    meta.identity.validate()?;
    if values.len() != meta.identity.quantities.len() || values.iter().any(|v| !v.is_finite()) {
        return Err(IoError::Convert("invalid external claim values or length".into()));
    }
    Ok(())
}

impl ExternalClaimArtifact {
    /// Own the bound claim's values under its identity.
    ///
    /// # Errors
    /// Non-response claims, blank IDs and invalid coordinates refuse.
    pub fn from_bound_claim(
        claim: &BoundExternalClaim,
        causal_contract_id: &str,
    ) -> Result<Self, IoError> {
        let values = claim
            .values()
            .ok_or_else(|| IoError::Convert("claim is not a finite response grid".into()))?
            .to_vec();
        let metadata = ExternalClaimMetadata {
            version: 1,
            identity: ExternalClaimIdentity::from_claim(claim, causal_contract_id)?,
            native_estimation: false,
        };
        validate(&metadata, &values)?;
        Ok(Self { metadata, values })
    }

    /// Validate and own metadata and values.
    ///
    /// # Errors
    /// Shape, finiteness, identity and the non-native flag are checked.
    pub fn new(metadata: ExternalClaimMetadata, values: Vec<f64>) -> Result<Self, IoError> {
        validate(&metadata, &values)?;
        Ok(Self { metadata, values })
    }

    /// Bound metadata.
    #[must_use]
    pub fn metadata(&self) -> &ExternalClaimMetadata {
        &self.metadata
    }

    /// Response values in coordinate order.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Name of the external execution that produced the numbers.
    #[must_use]
    pub fn provenance_label(&self) -> String {
        let id = &self.metadata.identity;
        format!("external:{}/{}@{}#{}", id.provider_id, id.object_id, id.version_id, id.snapshot_id)
    }

    /// Serialize through the checksummed sectioned container.
    ///
    /// # Errors
    /// Invalid content or an oversized payload refuses.
    pub fn to_bytes(&self, artifact_id: &str) -> Result<Vec<u8>, IoError> {
        validate(&self.metadata, &self.values)?;
        if artifact_id.trim().is_empty() {
            return Err(IoError::Convert("missing artifact id".into()));
        }
        let meta_bytes = to_cbor(&self.metadata)?;
        let mut value_bytes = Vec::with_capacity(self.values.len() * 8);
        for value in &self.values {
            value_bytes.extend_from_slice(&value.to_le_bytes());
        }
        let encoded = EncodedArtifact {
            manifest: ArtifactManifest {
                format_version: crate::migrate::STABLE_FORMAT,
                minimum_reader_version: crate::migrate::STABLE_FORMAT,
                artifact_kind: ArtifactKind::Other(ARTIFACT_KIND.into()),
                library_version: SemanticVersion::from_crate_version(env!("CARGO_PKG_VERSION"))?,
                artifact_id: artifact_id.into(),
                sections: vec![
                    section_descriptor(META_SECTION, "application/cbor", &meta_bytes),
                    section_descriptor(VALUES_SECTION, "application/octet-stream", &value_bytes),
                ],
                provenance: ProvenanceWire { note: "external_response_claim".into() },
            },
            sections: vec![
                SectionBytes::new(META_SECTION, meta_bytes),
                SectionBytes::new(VALUES_SECTION, value_bytes),
            ],
        };
        let mut bytes = Vec::new();
        encoded.write_to(&mut bytes)?;
        if bytes.len() > MAX_EXTERNAL_CLAIM_ARTIFACT_BYTES {
            return Err(IoError::TooLarge);
        }
        Ok(bytes)
    }

    /// Load and check against an independently retained identity.
    ///
    /// # Errors
    /// Oversized, truncated, corrupt, incompatible or semantically changed
    /// artifacts refuse before any value is exposed.
    pub fn from_bytes(bytes: &[u8], expected: &ExternalClaimIdentity) -> Result<Self, IoError> {
        if bytes.len() > MAX_EXTERNAL_CLAIM_ARTIFACT_BYTES {
            return Err(IoError::TooLarge);
        }
        expected.validate()?;
        let mut reader = ArtifactReader::open_seek(std::io::Cursor::new(bytes))?;
        let manifest = reader.manifest();
        if manifest.artifact_kind != ArtifactKind::Other(ARTIFACT_KIND.into())
            || manifest.sections.len() != 2
            || manifest.sections[0].id != META_SECTION
            || manifest.sections[1].id != VALUES_SECTION
        {
            return Err(IoError::Convert("unsupported external claim layout".into()));
        }
        let declared = manifest.sections.iter().try_fold(0u64, |total, section| {
            total.checked_add(section.uncompressed_size).ok_or(IoError::TooLarge)
        })?;
        if declared > MAX_EXTERNAL_CLAIM_ARTIFACT_BYTES as u64 {
            return Err(IoError::TooLarge);
        }
        let meta = reader.load_section(META_SECTION)?;
        let metadata: ExternalClaimMetadata = from_cbor(meta.as_bytes())?;
        if metadata.identity != *expected {
            return Err(IoError::Refused {
                code: antecedent_core::reason_code!("external_binding_mismatch"),
                message: "external_claim.identity_expected: claim identity differs from the consumer's checked contract".into(),
            });
        }
        let raw = reader.load_section(VALUES_SECTION)?;
        if raw.as_bytes().len() != metadata.identity.quantities.len() * 8 {
            return Err(IoError::Convert("external claim value payload length mismatch".into()));
        }
        let values = raw
            .as_bytes()
            .chunks_exact(8)
            .map(|chunk| f64::from_le_bytes(chunk.try_into().expect("eight-byte chunk")))
            .collect();
        Self::new(metadata, values)
    }
}
