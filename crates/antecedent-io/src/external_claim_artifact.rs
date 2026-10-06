//! Portable artifact for a finite external response grid bound to a checked contract.
//!
//! The artifact states that the numbers were computed by a foreign provider,
//! names that execution, and carries the premises the binding stage checked.
//! A consumer supplies its own expected identity on load; a resealed change in
//! any scientific, provider, trust or premise field is refused. The artifact
//! never claims native estimation and grants no interval or calibration.

use antecedent_core::{
    BoundExternalClaim, BoundTrustLevel, CompositionLink, CompositionStage,
    ExternalUncertaintyMeaning, ProvenanceChain, ScientificQuantity,
};
use serde::{Deserialize, Serialize};

use crate::container::{ArtifactManifest, EncodedArtifact, SectionBytes, section_descriptor};
use crate::convert::{from_cbor, to_cbor};
use crate::error::IoError;
use crate::quantity_wire::{DistributionMeaningWire, ScientificQuantityWire};
use crate::reader::ArtifactReader;
use crate::wire::{ArtifactKind, ProvenanceWire, SemanticVersion};

/// Maximum artifact bytes accepted before decode allocation.
pub const MAX_EXTERNAL_CLAIM_ARTIFACT_BYTES: usize = 4 * 1024 * 1024;
/// Maximum number of response coordinates.
pub const MAX_EXTERNAL_CLAIM_COORDINATES: usize = 100_000;
const ARTIFACT_KIND: &str = "external_response_claim_v1";
const META_SECTION: &str = "external_claim.meta";
const VALUES_SECTION: &str = "external_claim.values";
const LABELS: [&str; 5] =
    ["supported", "weak_overlap", "extrapolative", "outside_empirical_support", "missing_evidence"];

/// Trust label preserved without upgrading it during load.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExternalClaimTrust {
    /// Supplier assertion only.
    ExternallyAttested,
    /// The exact provider contract passed object-level verification. Written as
    /// `verified_extension`, the shared Rust/Python provider-trust vocabulary.
    #[serde(rename = "verified_extension")]
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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
    /// BLAKE3 digest of the ordered little-endian f64 value payload, retained
    /// independently by the consumer alongside the scientific identity.
    pub values_blake3: String,
    /// Trust held by the provider object.
    pub trust: ExternalClaimTrust,
    /// Support label per coordinate (`SupportStatus` names); `missing_evidence`
    /// where the provider declared none.
    pub point_status: Vec<String>,
    /// Provider-declared uncertainty method; absent means no uncertainty claim.
    pub uncertainty_method: Option<String>,
    /// Evidence factors used.
    pub evidence_ids: Vec<String>,
    /// Assumptions relied on.
    pub assumption_ids: Vec<String>,
    /// Checked equivalences licensing an observational law.
    pub equivalence_ids: Vec<String>,
    /// Derivation chain behind the values, parents before children.
    pub lineage: Vec<LineageLinkWire>,
    /// What one value of the provider's law means (`DistributionMeaning` wire name).
    pub provider_meaning: String,
    /// Operations the provider object declared at this request, sorted.
    pub capabilities: Vec<String>,
    /// BLAKE3 of the complete provider contract (identity, meaning, sorted
    /// capabilities, ordered coordinates); a different request or capability
    /// set has a different fingerprint.
    pub provider_fingerprint: String,
    /// Passed verification probes; present exactly when trust is
    /// `verified_extension`, and covering only `provider_fingerprint`.
    pub verification: Option<Vec<VerificationProbeWire>>,
}

/// One passed verification probe on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VerificationProbeWire {
    /// Property name (`normalization`, `moments`, ...).
    pub kind: String,
    /// Provider value.
    pub observed: f64,
    /// Independent expected value.
    pub expected: f64,
    /// Predeclared absolute tolerance.
    pub tolerance: f64,
}

/// One derivation step on the wire.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LineageLinkWire {
    /// Stable link identity.
    pub id: String,
    /// `CompositionStage` wire name.
    pub stage: String,
    /// Identities of earlier links this one was derived from.
    pub parents: Vec<String>,
}

impl ExternalClaimIdentity {
    /// The derivation chain, rebuilt and validated from the wire.
    ///
    /// # Errors
    /// An unknown stage, unresolved parent or duplicate link refuses.
    pub fn provenance_chain(&self) -> Result<ProvenanceChain, IoError> {
        let links = self
            .lineage
            .iter()
            .map(|link| {
                Ok(CompositionLink {
                    id: link.id.clone(),
                    stage: CompositionStage::from_name(&link.stage).ok_or_else(|| {
                        IoError::Convert(format!("unknown lineage stage `{}`", link.stage))
                    })?,
                    parents: link.parents.clone(),
                })
            })
            .collect::<Result<Vec<_>, IoError>>()?;
        ProvenanceChain::new(links).map_err(|error| IoError::Convert(format!("{error:?}")))
    }

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
            values_blake3: values_digest(claim.values().expect("response claim checked above")),
            trust: claim.trust().into(),
            point_status: claim
                .point_status()
                .unwrap_or_default()
                .iter()
                .map(|status| status.as_str().to_owned())
                .collect(),
            uncertainty_method: match claim.uncertainty() {
                ExternalUncertaintyMeaning::None => None,
                ExternalUncertaintyMeaning::ProviderDeclared { method_id } => {
                    Some(method_id.clone())
                }
            },
            evidence_ids: claim.evidence_ids().to_vec(),
            assumption_ids: claim.assumption_ids().to_vec(),
            equivalence_ids: claim.equivalence_ids().to_vec(),
            lineage: claim
                .provenance_chain(causal_contract_id)
                .map_err(|error| IoError::Convert(format!("{error:?}")))?
                .links()
                .iter()
                .map(|link| LineageLinkWire {
                    id: link.id.clone(),
                    stage: link.stage.as_str().to_owned(),
                    parents: link.parents.clone(),
                })
                .collect(),
            provider_meaning: claim
                .provider_meaning()
                .map(|meaning| {
                    serde_json::to_value(DistributionMeaningWire::from(meaning))
                        .ok()
                        .and_then(|v| v.as_str().map(str::to_owned))
                        .ok_or_else(|| IoError::Convert("unnamed provider meaning".into()))
                })
                .transpose()?
                .ok_or_else(|| IoError::Convert("claim has no provider law meaning".into()))?,
            capabilities: {
                let mut names: Vec<String> = claim
                    .capabilities()
                    .iter()
                    .map(|c| antecedent_core::capability_name(*c).to_owned())
                    .collect();
                names.sort();
                names
            },
            provider_fingerprint: String::new(),
            verification: match claim.trust() {
                BoundTrustLevel::ExactRequestVerified => Some(
                    claim
                        .verification()
                        .iter()
                        .map(|p| VerificationProbeWire {
                            kind: antecedent_core::probe_name(p.kind).to_owned(),
                            observed: p.observed,
                            expected: p.expected,
                            tolerance: p.tolerance,
                        })
                        .collect(),
                ),
                BoundTrustLevel::ExternallyAttested => None,
            },
        };
        let value = Self { provider_fingerprint: value.compute_fingerprint(), ..value };
        value.validate()?;
        Ok(value)
    }

    /// Recompute the complete provider-contract fingerprint from this identity.
    /// A consumer builds its expected identity from constants and sets this itself.
    #[must_use]
    pub fn compute_fingerprint(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        let mut put = |value: &str| {
            hasher.update(&(value.len() as u64).to_le_bytes());
            hasher.update(value.as_bytes());
        };
        for field in [
            &self.provider_id,
            &self.object_id,
            &self.version_id,
            &self.snapshot_id,
            &self.request_id,
            &self.provider_meaning,
        ] {
            put(field);
        }
        let mut capabilities = self.capabilities.clone();
        capabilities.sort();
        for capability in &capabilities {
            put(capability);
        }
        for quantity in &self.quantities {
            put(&format!("{quantity:?}"));
        }
        hasher.finalize().to_hex().to_string()
    }

    fn validate(&self) -> Result<(), IoError> {
        if self.quantities.is_empty() || self.quantities.len() > MAX_EXTERNAL_CLAIM_COORDINATES {
            return Err(IoError::TooLarge);
        }
        for quantity in &self.quantities {
            ScientificQuantity::try_from(quantity.clone())
                .map_err(|error| IoError::Convert(error.into()))?;
        }
        if self.point_status.len() != self.quantities.len()
            || self.point_status.iter().any(|label| !LABELS.contains(&label.as_str()))
        {
            return Err(IoError::Convert("invalid external claim support labels".into()));
        }
        if self.values_blake3.len() != 64
            || !self.values_blake3.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(IoError::Convert("invalid external claim value digest".into()));
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
        if self.provider_fingerprint != self.compute_fingerprint() {
            return Err(IoError::Convert("provider fingerprint differs from its contract".into()));
        }
        match (&self.verification, self.trust) {
            (None, ExternalClaimTrust::ExternallyAttested) => {}
            (Some(probes), ExternalClaimTrust::ExactRequestVerified) => {
                let passed = !probes.is_empty()
                    && probes.iter().all(|p| {
                        antecedent_core::probe_from_name(&p.kind).is_some()
                            && p.observed.is_finite()
                            && p.expected.is_finite()
                            && p.tolerance.is_finite()
                            && p.tolerance >= 0.0
                            && (p.observed - p.expected).abs() <= p.tolerance
                    });
                if !passed {
                    return Err(IoError::Convert(
                        "verification receipt has a failed or unknown probe".into(),
                    ));
                }
            }
            _ => {
                return Err(IoError::Convert(
                    "verification receipt must accompany exactly verified trust".into(),
                ));
            }
        }
        self.provenance_chain()?
            .require_stages(
                BoundExternalClaim::CLAIM_LINK_ID,
                &[
                    CompositionStage::CausalContract,
                    CompositionStage::ExternalProvider,
                    CompositionStage::Claim,
                ],
            )
            .map_err(|error| IoError::Convert(format!("incomplete lineage: {error:?}")))?;
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
    if values_digest(values) != meta.identity.values_blake3 {
        return Err(IoError::Convert("external claim values differ from retained identity".into()));
    }
    Ok(())
}

fn values_digest(values: &[f64]) -> String {
    let mut hasher = blake3::Hasher::new();
    for value in values {
        hasher.update(&value.to_le_bytes());
    }
    hasher.finalize().to_hex().to_string()
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
