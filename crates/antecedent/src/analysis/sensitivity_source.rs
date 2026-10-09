//! Original joint-sensitivity sources independently replayed into decision surfaces.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::JointSensitivityArtifactWire;
use antecedent_core::{
    CompositionLink, CompositionStage, ExecutionContext, ProvenanceChain, ScientificQuantity,
    reason_code,
};
use antecedent_io::sensitivity_artifact::SensitivityArtifact;
use antecedent_io::{IoError, from_cbor, to_cbor};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

const MAGIC: &[u8] = b"ANTE-SENSITIVITY-SOURCE-1\0";
/// Largest complete source-backed sensitivity envelope.
pub const MAX_BYTES: usize = 16 * 1024 * 1024;

fn sensitivity_invalid(detail: &str) -> IoError {
    IoError::Refused { code: reason_code!("invalid_argument"), message: detail.into() }
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Wire {
    source: Vec<u8>,
    surface: Vec<u8>,
}
/// Actual original source replay and its derived scientific surface.
#[derive(Clone, Debug)]
pub struct SourceBackedSensitivity {
    source: Arc<[u8]>,
    original: Arc<JointSensitivityArtifactWire>,
    surface: SensitivityArtifact,
}
impl SourceBackedSensitivity {
    /// Consume the original v3 source and rebuild the original surface adapter.
    /// # Errors
    /// Any original replay refusal, oversized source or changed derived surface.
    pub fn checked(
        source: &[u8],
        surface: &SensitivityArtifact,
        ctx: &ExecutionContext,
    ) -> Result<Self, IoError> {
        if source.len() > MAX_BYTES {
            return Err(sensitivity_invalid("sensitivity_source.limits_exceeded"));
        }
        let original = JointSensitivityArtifactWire::consume(source, ctx)?;
        let result = original.replay_result(ctx)?;
        let quantities = surface.quantities();
        if quantities.len() != 1 {
            return Err(sensitivity_invalid("sensitivity_source.surface_mismatch"));
        }
        let effect = ScientificQuantity::try_from(quantities[0].quantity.clone())
            .map_err(|_| sensitivity_invalid("sensitivity_source.surface_mismatch"))?;
        let actual = SensitivityArtifact::from_joint_result(
            &result,
            &effect,
            surface.grid().len(),
            surface.actions().to_vec(),
            &surface.provenance().causal_contract_id,
        )?;
        if &actual != surface {
            return Err(sensitivity_invalid("sensitivity_source.surface_mismatch"));
        }
        Ok(Self { source: Arc::from(source), original: Arc::new(original), surface: actual })
    }
    /// Original independently consumed v3 source bytes, including baseline laws and proof.
    #[must_use]
    pub fn original_bytes(&self) -> &[u8] {
        &self.source
    }
    /// Surface rebuilt by the original numerical engine and original adapter.
    #[must_use]
    pub fn surface(&self) -> &SensitivityArtifact {
        &self.surface
    }
    /// Typed ancestry from the baseline artifact and actual sensitivity premises.
    /// # Errors
    /// Invalid original identity (never a trust or calibration promotion).
    pub fn provenance_chain(&self) -> Result<ProvenanceChain, IoError> {
        let mut links = self.surface.provenance_chain()?.links().to_vec();
        let original = &self.original.body.provenance;
        let baseline = format!("baseline_artifact:{}", original.baseline_artifact_digest);
        let source = format!("joint_sensitivity_source:{}", blake3::hash(&self.source).to_hex());
        let evidence = format!("sensitivity_premises:{}", self.original.premises_digest);
        let link = |id, stage, parents| CompositionLink {
            id,
            stage,
            parents,
            declared_parent_digests: None,
        };
        let mut prefix = vec![
            link(baseline.clone(), CompositionStage::Evidence, vec![]),
            link(evidence.clone(), CompositionStage::Evidence, vec![baseline]),
            link(source.clone(), CompositionStage::SensitivityInput, vec![evidence]),
        ];
        if let Some(input) =
            links.iter_mut().find(|link| link.stage == CompositionStage::SensitivityInput)
        {
            input.parents.push(source);
        }
        prefix.extend(links);
        ProvenanceChain::new(prefix)
            .map_err(|_| sensitivity_invalid("sensitivity_source.surface_mismatch"))
    }
    /// Query source scientific standing and complete native derivation.
    /// # Errors
    /// Invalid original chain.
    /// # Panics
    /// Only if a validated native derivation chain loses one of its own links.
    pub fn summary(&self) -> Result<serde_json::Value, IoError> {
        let chain = self.provenance_chain()?;
        Ok(serde_json::json!({
            "source_artifact_digest":blake3::hash(&self.source).to_hex().to_string(),
            "baseline_artifact_digest":self.original.body.provenance.baseline_artifact_digest,
            "premises_digest":self.original.premises_digest,"data_digest":self.original.data_digest,
            "source_provenance":self.original.body.provenance,
            "source_verified":true,"verification_scope":"original stored assumptions and exact provider laws replayed; premises are declared",
            "native_authority_issued":false,"calibration_license_issued":false,
            "uncertainty":self.surface.uncertainty(),
            "coordinates":self.surface.quantities().iter().map(|q| &q.quantity).collect::<Vec<_>>(),
            "lineage":chain.links().iter().map(|link| serde_json::json!({
                "id":link.id,"stage":link.stage.as_str(),"parents":link.parents,
                "digest":chain.digest_of(&link.id).expect("validated link"),
                "parent_digests":link.parents.iter().map(|id| chain.digest_of(id).expect("validated parent")).collect::<Vec<_>>()
            })).collect::<Vec<_>>()
        }))
    }
    /// Export the complete source and original surface, preserving original formats.
    /// # Errors
    /// Original encoding refusal or oversized envelope.
    pub fn export(&self) -> Result<Vec<u8>, IoError> {
        let mut bytes = MAGIC.to_vec();
        bytes.extend(to_cbor(&Wire {
            source: self.source.to_vec(),
            surface: self.surface.to_bytes("source-backed-sensitivity")?,
        })?);
        if bytes.len() > MAX_BYTES {
            return Err(sensitivity_invalid("sensitivity_source.limits_exceeded"));
        }
        Ok(bytes)
    }
    /// Independently replay the original source and verify the entire scientific surface.
    /// # Errors
    /// Invalid, oversized or scientifically inconsistent source-backed envelope.
    pub fn consume(bytes: &[u8], ctx: &ExecutionContext) -> Result<Self, IoError> {
        if bytes.len() > MAX_BYTES {
            return Err(sensitivity_invalid("sensitivity_source.limits_exceeded"));
        }
        let payload = bytes
            .strip_prefix(MAGIC)
            .ok_or_else(|| sensitivity_invalid("sensitivity_source.invalid_artifact"))?;
        let wire: Wire = from_cbor(payload)
            .map_err(|_| sensitivity_invalid("sensitivity_source.invalid_artifact"))?;
        let surface = SensitivityArtifact::from_bytes(&wire.surface, None)?;
        Self::checked(&wire.source, &surface, ctx)
    }
}
