//! Original finite-law functional replay with complete source rows and scientific identity.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::decision_artifact::{contract_from_json_refusal, contract_to_json};
use crate::decision_contract::{DecisionContract, DecisionFunctional};
use crate::decision_eval::{FunctionalValue, evaluate_functional};
use antecedent_core::{CompositionLink, CompositionStage, ProvenanceChain, reason_code};
use antecedent_io::distribution_artifact::{DistributionArtifact, DistributionIdentity};
use antecedent_io::{IoError, from_cbor, to_cbor};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

const MAGIC: &[u8] = b"ANTE-LAW-FUNCTIONAL-1\0";
/// Largest complete law-functional envelope.
pub const MAX_BYTES: usize = 16 * 1024 * 1024;
fn functional_invalid(detail: &str) -> IoError {
    IoError::Refused { code: reason_code!("invalid_argument"), message: detail.into() }
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Wire {
    source: Vec<u8>,
    expected: DistributionIdentity,
    contract: String,
    action: String,
    functional: DecisionFunctional,
    value: f64,
    standard_error: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    source_evidence: Option<Vec<u8>>,
}
/// Complete original finite law and functional request, replayed by the original engine.
#[derive(Clone, Debug)]
pub struct LawFunctionalArtifact {
    source: Arc<DistributionArtifact>,
    contract: DecisionContract,
    action: String,
    functional: DecisionFunctional,
    result: FunctionalValue,
    source_evidence: Option<crate::source_evidence::SourceEvidence>,
}
impl LawFunctionalArtifact {
    /// Execute the original functional engine on an existing finite law.
    /// # Errors
    /// An original contract, coordinate, support, meaning or functional refusal.
    pub fn produce(
        source: DistributionArtifact,
        contract: DecisionContract,
        action: &str,
        functional: DecisionFunctional,
    ) -> Result<Self, IoError> {
        let result =
            evaluate_functional(&contract, action, functional, &source).map_err(|error| {
                IoError::Refused { code: error.reason_code(), message: format!("{error:?}") }
            })?;
        Ok(Self {
            source: Arc::new(source),
            contract,
            action: action.into(),
            functional,
            result,
            source_evidence: None,
        })
    }
    /// Retain original scoped source diagnostics, without claiming imported native row authority.
    /// # Errors
    /// A source coordinate/snapshot citation that does not match this actual law.
    pub fn with_source_evidence(
        mut self,
        evidence: &crate::source_evidence::SourceEvidence,
    ) -> Result<Self, IoError> {
        let quantities = self
            .source
            .quantities()
            .iter()
            .cloned()
            .map(antecedent_core::ScientificQuantity::try_from)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| functional_invalid("functional_source.invalid_source"))?;
        evidence
            .require_quantity_binding(&quantities, &self.source.metadata().identity.snapshot_id)
            .map_err(|_| functional_invalid("functional_source.invalid_source"))?;
        self.source_evidence = Some(
            evidence
                .project(
                    &contract_to_json(&self.contract)
                        .map_err(|_| functional_invalid("functional_source.invalid_contract"))?,
                    &[self.action.clone()],
                )
                .map_err(|_| functional_invalid("functional_source.invalid_source"))?,
        );
        Ok(self)
    }
    /// Complete original finite law, retained for an explicit actual-producer comparison.
    #[must_use]
    pub fn original_law(&self) -> &DistributionArtifact {
        &self.source
    }
    /// Retained diagnostic citation; original unresolved operations remain original.
    #[must_use]
    pub fn source_evidence(&self) -> Option<&crate::source_evidence::SourceEvidence> {
        self.source_evidence.as_ref()
    }
    /// Actual original-engine value, with sampled uncertainty left unmeasured.
    #[must_use]
    pub fn result(&self) -> &FunctionalValue {
        &self.result
    }
    /// Native typed original-law and transformation ancestry.
    /// # Errors
    /// Invalid original contract identity or derivation chain.
    pub fn provenance_chain(&self) -> Result<ProvenanceChain, IoError> {
        let metadata = self.source.metadata();
        let causal = format!("causal_contract:{}", metadata.identity.causal_contract_id);
        let data = format!("snapshot:{}", metadata.identity.snapshot_id);
        let law = format!("distribution:{}", crate::decision_artifact::source_digest(&self.source));
        let decision = format!(
            "decision_contract:{}",
            self.contract
                .identity()
                .map_err(|_| functional_invalid("functional_source.invalid_contract"))?
        );
        let transform = format!(
            "functional:{}",
            blake3::hash(&to_cbor(&(&self.action, &self.functional, &decision))?).to_hex()
        );
        let link = |id, stage, parents| CompositionLink {
            id,
            stage,
            parents,
            declared_parent_digests: None,
        };
        ProvenanceChain::new(vec![
            link(causal.clone(), CompositionStage::CausalContract, vec![]),
            link(data.clone(), CompositionStage::Data, vec![]),
            link(law.clone(), CompositionStage::DistributionArtifact, vec![causal, data]),
            link(decision.clone(), CompositionStage::DecisionContract, vec![]),
            link(transform.clone(), CompositionStage::Transformation, vec![law, decision]),
            link("claim".into(), CompositionStage::Claim, vec![transform]),
        ])
        .map_err(|_| functional_invalid("functional_source.invalid_source"))
    }
    /// Complete semantic source standing and original-law contributor coordinates.
    /// # Errors
    /// Invalid original chain.
    pub fn summary(&self) -> Result<serde_json::Value, IoError> {
        let chain = self.provenance_chain()?;
        let action = self
            .contract
            .actions
            .iter()
            .find(|action| action.id == self.action)
            .expect("evaluated action");
        Ok(serde_json::json!({
            "identity":blake3::hash(&self.export()?).to_hex().to_string(),
            "value":self.result.value,"standard_error":self.result.standard_error,"source_mode":format!("{:?}",self.result.source_mode),
            "source_metadata":self.source.metadata(),
            "source_evidence":self.source_evidence.as_ref().map(crate::source_evidence::SourceEvidence::summary),
            "native_row_origin_verified":false,"source_evidence_scope":"original scoped diagnostic citation; matching quantities and snapshot do not prove native row origin","action":self.action,"functional":self.functional,
            "contributors":action.utility.inputs_used().iter().filter_map(|index|action.inputs.get(*index)).map(antecedent_io::quantity_wire::ScientificQuantityWire::from).collect::<Vec<_>>(),
            "verification_scope":"original finite law and functional replay; source labels preserve declared standing",
            "native_authority_issued":false,"calibration_license_issued":false,
            "lineage":chain.links().iter().map(|link| serde_json::json!({
                "id":link.id,"stage":link.stage.as_str(),"parents":link.parents,"digest":chain.digest_of(&link.id).expect("validated link"),
                "parent_digests":link.parents.iter().map(|id|chain.digest_of(id).expect("validated parent")).collect::<Vec<_>>()
            })).collect::<Vec<_>>()
        }))
    }
    /// Export full original law rows, weights, axes, coupling and request for replay.
    /// # Errors
    /// Original encoding failure or oversized envelope.
    pub fn export(&self) -> Result<Vec<u8>, IoError> {
        if self.source.draws().len().checked_mul(8).is_none_or(|size| size > MAX_BYTES) {
            return Err(functional_invalid("functional_source.limits_exceeded"));
        }
        let mut bytes = MAGIC.to_vec();
        bytes.extend(to_cbor(&Wire {
            source: self.source.to_bytes("functional-original-law")?,
            expected: self.source.metadata().identity.clone(),
            contract: contract_to_json(&self.contract)
                .map_err(|_| functional_invalid("functional_source.invalid_contract"))?,
            action: self.action.clone(),
            functional: self.functional,
            value: self.result.value,
            standard_error: self.result.standard_error,
            source_evidence: self
                .source_evidence
                .as_ref()
                .map(crate::source_evidence::SourceEvidence::export)
                .transpose()
                .map_err(|_| functional_invalid("functional_source.invalid_source"))?,
        })?);
        if bytes.len() > MAX_BYTES {
            return Err(functional_invalid("functional_source.limits_exceeded"));
        }
        Ok(bytes)
    }
    /// Independently consume full original law and recompute the entire functional result.
    /// # Errors
    /// Invalid envelope, original source refusal or changed scientific output.
    pub fn consume(bytes: &[u8]) -> Result<Self, IoError> {
        Self::consume_expected(bytes, None)
    }
    /// Require an independently retained complete source-and-request identity before replay.
    /// # Errors
    /// Any original refusal or changed complete request identity.
    pub fn consume_expected(bytes: &[u8], expected: Option<&str>) -> Result<Self, IoError> {
        if bytes.len() > MAX_BYTES {
            return Err(functional_invalid("functional_source.limits_exceeded"));
        }
        if expected.is_some_and(|id| id != blake3::hash(bytes).to_hex().as_str()) {
            return Err(functional_invalid("functional_source.expected_identity_mismatch"));
        }
        let payload = bytes
            .strip_prefix(MAGIC)
            .ok_or_else(|| functional_invalid("functional_source.invalid_artifact"))?;
        let wire: Wire = from_cbor(payload)
            .map_err(|_| functional_invalid("functional_source.invalid_artifact"))?;
        let source = DistributionArtifact::from_bytes(&wire.source, &wire.expected)?;
        let contract = contract_from_json_refusal(&wire.contract)
            .map_err(|_| functional_invalid("functional_source.invalid_contract"))?;
        let mut result = Self::produce(source, contract, &wire.action, wire.functional)?;
        if let Some(source) = wire.source_evidence {
            result = result.with_source_evidence(
                &crate::source_evidence::SourceEvidence::consume(&source)
                    .map_err(|_| functional_invalid("functional_source.invalid_source"))?,
            )?;
        }
        if result.result.value.to_bits() != wire.value.to_bits()
            || result.result.standard_error.map(f64::to_bits)
                != wire.standard_error.map(f64::to_bits)
        {
            return Err(functional_invalid("functional_source.output_mismatch"));
        }
        Ok(result)
    }
}
