//! Independently consumed original source diagnostics and semantic contributor maps.
//!
//! Evidence preserves an existing scientific claim; importing it neither restores
//! executable state nor promotes provider trust or uncertainty calibration.

use crate::decision_artifact::{contract_from_json_refusal, contract_to_json};
use crate::decision_contract::DecisionContract;
use antecedent_core::{ScientificQuantity, reason_code};
use antecedent_io::quantity_wire::ScientificQuantityWire;
use antecedent_io::response_wire::{ResponseIdentificationWire, ResponseValueWire};
use antecedent_io::{AnalysisResultConsumption, consume_analysis_result};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::Arc;

/// Maximum independently retained source-evidence envelope size.
pub const MAX_SOURCE_EVIDENCE_BYTES: usize = 16 * 1024 * 1024;
const MAGIC: &[u8] = b"ANTE-SOURCE-EVIDENCE-1\0";

/// Original scientific evidence or bounded declaration failed validation.
#[derive(Clone, Copy, Debug)]
pub struct SourceEvidenceError {
    /// Stable registered broad reason.
    pub code: &'static str,
    /// Stable exact detail.
    pub detail: &'static str,
}
impl std::fmt::Display for SourceEvidenceError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}: {}", self.code, self.detail)
    }
}
impl std::error::Error for SourceEvidenceError {}
fn invalid(detail: &'static str) -> SourceEvidenceError {
    SourceEvidenceError { code: reason_code!("invalid_argument"), detail }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct EvidenceWire {
    source: Vec<u8>,
    contract: Option<String>,
    actions: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    external_identity: Option<antecedent_io::external_claim_artifact::ExternalClaimIdentity>,
}

/// Original recognized analysis result and an optional semantic functional mapping.
#[derive(Clone, Debug)]
pub struct SourceEvidence {
    source: Arc<[u8]>,
    original: Option<Arc<AnalysisResultConsumption>>,
    external: Option<Arc<antecedent_io::external_claim_artifact::ExternalClaimArtifact>>,
    manifest: Arc<antecedent_io::ArtifactManifest>,
    coordinates: Arc<[ScientificQuantity]>,
    contract: Option<DecisionContract>,
    actions: Vec<String>,
}
impl SourceEvidence {
    /// Consume the original contracted result; diagnostics always come from that result.
    /// # Errors
    /// Invalid bytes, absent original claim, absent response coordinates or invalid scopes.
    pub fn from_result(bytes: &[u8]) -> Result<Self, SourceEvidenceError> {
        if bytes.len() > MAX_SOURCE_EVIDENCE_BYTES {
            return Err(invalid("source_evidence.limits_exceeded"));
        }
        let manifest = antecedent_io::EncodedArtifact::read_selective(
            std::io::Cursor::new(bytes),
            &std::collections::HashSet::new(),
        )
        .map_err(|_| invalid("source_evidence.invalid_source_artifact"))?
        .manifest;
        let logical_bytes = manifest
            .sections
            .iter()
            .try_fold(0_u64, |total, section| total.checked_add(section.uncompressed_size))
            .ok_or_else(|| invalid("source_evidence.limits_exceeded"))?;
        if logical_bytes > 16 * 1024 * 1024 {
            return Err(invalid("source_evidence.limits_exceeded"));
        }
        let original = consume_analysis_result(bytes)
            .map_err(|_| invalid("source_evidence.invalid_source_artifact"))?;
        if !original.acceptance.recognized
            || !original.acceptance.claim_present
            || original.contract.is_none()
        {
            return Err(invalid("source_evidence.source_unverified"));
        }
        let response = original
            .body
            .response
            .as_ref()
            .ok_or_else(|| invalid("source_evidence.response_unavailable"))?;
        let coordinates = response
            .coordinates
            .as_ref()
            .filter(|coordinates| !coordinates.is_empty())
            .ok_or_else(|| invalid("source_evidence.coordinates_unavailable"))?
            .iter()
            .cloned()
            .map(ScientificQuantity::try_from)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| invalid("source_evidence.invalid_coordinates"))?;
        for diagnostic in &response.support.diagnostics {
            match diagnostic.scope.as_deref().unwrap_or("global") {
                "per_coordinate" if diagnostic.values.len() == coordinates.len() => {}
                "global" => {}
                "inapplicable" if diagnostic.values.is_empty() => {}
                _ => return Err(invalid("source_evidence.diagnostic_scope_mismatch")),
            }
        }
        Ok(Self {
            manifest: Arc::new(manifest),
            source: Arc::from(bytes),
            original: Some(Arc::new(original)),
            external: None,
            coordinates: coordinates.into(),
            contract: None,
            actions: Vec::new(),
        })
    }
    fn native_original(&self) -> &AnalysisResultConsumption {
        self.original.as_deref().expect("native original operation requires native evidence")
    }
    /// Original bound external artifact; independently validates full provider/request/quantity/value identity.
    /// Its declared support and ancestry are retained; absent numeric diagnostics remain explicitly unavailable.
    pub fn from_external(bytes: &[u8]) -> Result<Self, SourceEvidenceError> {
        use antecedent_io::external_claim_artifact::{
            ExternalClaimArtifact, ExternalClaimMetadata, MAX_EXTERNAL_CLAIM_ARTIFACT_BYTES,
        };
        if bytes.len() > MAX_EXTERNAL_CLAIM_ARTIFACT_BYTES {
            return Err(invalid("source_evidence.limits_exceeded"));
        }
        let mut reader = antecedent_io::ArtifactReader::open_seek(std::io::Cursor::new(bytes))
            .map_err(|_| invalid("source_evidence.invalid_source_artifact"))?;
        let manifest = reader.manifest().clone();
        if manifest
            .sections
            .iter()
            .try_fold(0_u64, |sum, section| sum.checked_add(section.uncompressed_size))
            .is_none_or(|size| size > MAX_EXTERNAL_CLAIM_ARTIFACT_BYTES as u64)
        {
            return Err(invalid("source_evidence.limits_exceeded"));
        }
        let metadata: ExternalClaimMetadata = antecedent_io::from_cbor(
            reader
                .load_section("external_claim.meta")
                .map_err(|_| invalid("source_evidence.invalid_source_artifact"))?
                .as_bytes(),
        )
        .map_err(|_| invalid("source_evidence.invalid_source_artifact"))?;
        let external = ExternalClaimArtifact::from_bytes(bytes, &metadata.identity)
            .map_err(|_| invalid("source_evidence.invalid_source_artifact"))?;
        let coordinates = metadata
            .identity
            .quantities
            .into_iter()
            .map(ScientificQuantity::try_from)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| invalid("source_evidence.invalid_coordinates"))?;
        Ok(Self {
            source: Arc::from(bytes),
            original: None,
            external: Some(Arc::new(external)),
            manifest: Arc::new(manifest),
            coordinates: coordinates.into(),
            contract: None,
            actions: Vec::new(),
        })
    }
    /// Whether this evidence describes an original checked native analysis, rather than external attestation.
    #[must_use]
    pub fn is_native_source(&self) -> bool {
        self.original.is_some()
    }
    /// Compare the full original scientific body, header and producing contract.
    /// This comparison alone is not execution evidence; only an opaque native issuer can resolve a dependency.
    pub fn require_same_source(&self, actual: &Self) -> Result<(), SourceEvidenceError> {
        if self.external.is_some() || actual.external.is_some() {
            return if self.external == actual.external {
                Ok(())
            } else {
                Err(invalid("source_evidence.source_binding_mismatch"))
            };
        }
        let encode = |source: &Self| {
            antecedent_io::to_cbor(&(
                source.manifest.as_ref(),
                &source.native_original().header,
                &source.native_original().body,
                &source.native_original().contract,
            ))
            .map_err(|_| invalid("source_evidence.invalid_source_artifact"))
        };
        if encode(self)? != encode(actual)? {
            return Err(invalid("source_evidence.source_binding_mismatch"));
        }
        Ok(())
    }
    /// Original dependencies still requiring original-operation replay.
    #[must_use]
    pub fn unresolved(&self) -> &[Arc<str>] {
        self.original.as_ref().map_or(&[], |original| original.acceptance.unresolved.as_ref())
    }

    /// Retained original consumer output for source-backed adapters; no re-decoding or authority upgrade.
    #[doc(hidden)]
    pub fn original_consumption(&self) -> Result<&AnalysisResultConsumption, SourceEvidenceError> {
        self.original
            .as_deref()
            .ok_or_else(|| invalid("source_evidence.resolution_scope_unsupported"))
    }

    /// Original producer artifact label, retained verbatim for native replay.
    #[must_use]
    pub fn artifact_id(&self) -> &str {
        &self.manifest.artifact_id
    }

    /// Scientific coordinates retained in original source order.
    #[must_use]
    pub fn coordinates(&self) -> &[ScientificQuantity] {
        &self.coordinates
    }
    /// Original independently consumed result bytes.
    #[must_use]
    pub fn original_bytes(&self) -> &[u8] {
        &self.source
    }
    /// Bind contributors to actual utility expressions; no local diagnostic is synthesized.
    /// # Errors
    /// Invalid contract, unknown/duplicate action or excessive declaration size.
    pub fn project(&self, text: &str, actions: &[String]) -> Result<Self, SourceEvidenceError> {
        if text.len() > 1024 * 1024 || actions.len() > 4096 {
            return Err(invalid("source_evidence.limits_exceeded"));
        }
        let contract = contract_from_json_refusal(text)
            .map_err(|_| invalid("source_evidence.invalid_contract"))?;
        if actions.iter().enumerate().any(|(index, id)| {
            actions[..index].contains(id) || !contract.actions.iter().any(|action| &action.id == id)
        }) {
            return Err(invalid("source_evidence.action_mismatch"));
        }
        let mut result = self.clone();
        result.actions = actions.to_vec();
        result.contract = Some(contract);
        Ok(result)
    }
    /// Verify a mean-grid point claim against the original source values and snapshot.
    pub fn require_mean_source(
        &self,
        source: &crate::decision_eval::MeanSource,
    ) -> Result<(), SourceEvidenceError> {
        if let Some(external) = &self.external {
            let identity = &external.metadata().identity;
            if identity.snapshot_id != source.snapshot_id
                || identity.provider_id != source.provider_id
                || identity.causal_contract_id != source.causal_contract_id
                || source.coordinates.len() != source.means.len()
            {
                return Err(invalid("source_evidence.point_binding_mismatch"));
            }
            for (quantity, value) in source.coordinates.iter().zip(&source.means) {
                let index = self
                    .coordinates
                    .iter()
                    .position(|q| q.require_same_coordinate(quantity).is_ok())
                    .ok_or_else(|| invalid("source_evidence.point_binding_mismatch"))?;
                if external.values()[index].to_bits() != value.to_bits() {
                    return Err(invalid("source_evidence.point_binding_mismatch"));
                }
            }
            return Ok(());
        }
        let response = self.native_original().body.response.as_ref().expect("validated response");
        let ResponseIdentificationWire::PointIdentified(ResponseValueWire::Surface {
            mean, ..
        }) = &response.estimate
        else {
            return Err(invalid("source_evidence.point_binding_mismatch"));
        };
        let snapshot = self
            .native_original()
            .contract
            .as_ref()
            .map(|contract| hex(&contract.identities.data_snapshot));
        if snapshot.as_deref() != Some(source.snapshot_id.as_str())
            || source.coordinates.len() != source.means.len()
        {
            return Err(invalid("source_evidence.point_binding_mismatch"));
        }
        for (quantity, value) in source.coordinates.iter().zip(&source.means) {
            let index = self
                .coordinates
                .iter()
                .position(|coordinate| coordinate.require_same_coordinate(quantity).is_ok())
                .ok_or_else(|| invalid("source_evidence.point_binding_mismatch"))?;
            if mean.get(index).is_none_or(|original| original.to_bits() != value.to_bits()) {
                return Err(invalid("source_evidence.point_binding_mismatch"));
            }
        }
        Ok(())
    }
    /// Require matching original coordinates and snapshot for a source-diagnostic citation.
    /// This citation does not verify numerical draw values or license native provider authority.
    pub fn require_quantity_binding(
        &self,
        quantities: &[ScientificQuantity],
        snapshot: &str,
    ) -> Result<(), SourceEvidenceError> {
        let expected = self
            .external
            .as_ref()
            .map(|external| external.metadata().identity.snapshot_id.clone())
            .or_else(|| {
                self.original
                    .as_ref()
                    .and_then(|original| original.contract.as_ref())
                    .map(|contract| hex(&contract.identities.data_snapshot))
            });
        if expected.as_deref() != Some(snapshot)
            || quantities.iter().any(|quantity| {
                !self
                    .coordinates
                    .iter()
                    .any(|coordinate| coordinate.require_same_coordinate(quantity).is_ok())
            })
        {
            return Err(invalid("source_evidence.point_binding_mismatch"));
        }
        Ok(())
    }

    /// Query original diagnostics for a semantic coordinate, never by vector length alone.
    /// # Errors
    /// The supplied semantic coordinate does not occur in the original result.
    pub fn diagnostics_at(
        &self,
        quantity: &ScientificQuantity,
    ) -> Result<Value, SourceEvidenceError> {
        let index = self
            .coordinates
            .iter()
            .position(|coordinate| coordinate.require_same_coordinate(quantity).is_ok())
            .ok_or_else(|| invalid("source_evidence.coordinate_mismatch"))?;
        if self.external.is_some() {
            return Ok(json!([]));
        }
        let response = self.native_original().body.response.as_ref().expect("validated response");
        Ok(Value::Array(response.support.diagnostics.iter().map(|diagnostic| {
            let scope = diagnostic.scope.as_deref().unwrap_or("global");
            json!({"id":diagnostic.id,"detail":diagnostic.detail,"scope":scope,
                "local_value": if scope == "per_coordinate" {Some(diagnostic.values[index])} else {None},
                "global_values": if scope == "global" {diagnostic.values.clone()} else {Vec::new()},
                "source_coordinate":ScientificQuantityWire::from(&self.coordinates[index])})
        }).collect()))
    }
    /// Typed ancestry derived from the independently consumed original contract and source.
    /// Recognition and derivation identity do not resolve the original numerical dependencies.
    pub fn provenance_chain(
        &self,
    ) -> Result<antecedent_core::ProvenanceChain, SourceEvidenceError> {
        use antecedent_core::{CompositionLink, CompositionStage, ProvenanceChain};
        if let Some(external) = &self.external {
            let chain = external
                .metadata()
                .identity
                .provenance_chain()
                .map_err(|_| invalid("source_evidence.invalid_source_artifact"))?;
            let mut links = chain.links().to_vec();
            let parent = links
                .last()
                .ok_or_else(|| invalid("source_evidence.invalid_source_artifact"))?
                .id
                .clone();
            let source = format!("source_artifact:{}", blake3::hash(&self.source).to_hex());
            links.push(CompositionLink {
                id: source.clone(),
                stage: CompositionStage::Evidence,
                parents: vec![parent],
                declared_parent_digests: None,
            });
            if let Some(contract) = &self.contract {
                let decision = format!(
                    "decision_contract:{}",
                    contract.identity().map_err(|_| invalid("source_evidence.invalid_contract"))?
                );
                links.push(CompositionLink {
                    id: decision.clone(),
                    stage: CompositionStage::DecisionContract,
                    parents: vec![],
                    declared_parent_digests: None,
                });
                links.push(CompositionLink {
                    id: "source_projection".into(),
                    stage: CompositionStage::Transformation,
                    parents: vec![source, decision],
                    declared_parent_digests: None,
                });
            }
            return ProvenanceChain::new(links)
                .map_err(|_| invalid("source_evidence.invalid_source_artifact"));
        }
        let original = self.native_original().contract.as_ref().expect("validated contract");
        let causal = format!("causal_contract:{}", hex(&original.seal));
        let data = format!("snapshot:{}", hex(&original.identities.data_snapshot));
        let program = format!("native_program:{}", hex(&original.identities.program));
        let source = format!("source_artifact:{}", hex(&self.native_original().artifact_digest));
        let link = |id: String, stage, parents: Vec<String>| CompositionLink {
            id,
            stage,
            parents,
            declared_parent_digests: None,
        };
        let mut links = vec![
            link(causal.clone(), CompositionStage::CausalContract, vec![]),
            link(data.clone(), CompositionStage::Data, vec![]),
            link(program.clone(), CompositionStage::Evidence, vec![causal, data]),
            link(source.clone(), CompositionStage::Evidence, vec![program]),
        ];
        let parent = if let Some(contract) = &self.contract {
            let decision = format!(
                "decision_contract:{}",
                contract.identity().map_err(|_| invalid("source_evidence.invalid_contract"))?
            );
            links.push(link(decision.clone(), CompositionStage::DecisionContract, vec![]));
            let transform = format!(
                "source_projection:{}",
                hex(blake3::hash(
                    &antecedent_io::to_cbor(&(&decision, &self.actions))
                        .map_err(|_| invalid("source_evidence.invalid_contract"))?
                )
                .as_bytes())
            );
            links.push(link(
                transform.clone(),
                CompositionStage::Transformation,
                vec![source, decision],
            ));
            transform
        } else {
            source
        };
        links.push(link("claim".into(), CompositionStage::Claim, vec![parent]));
        ProvenanceChain::new(links).map_err(|_| invalid("source_evidence.invalid_source_artifact"))
    }
    /// Original typed ancestry including verified Merkle parent links.
    pub fn lineage(&self) -> Result<Value, SourceEvidenceError> {
        let chain = self.provenance_chain()?;
        Ok(Value::Array(
            chain
                .links()
                .iter()
                .map(|link| {
                    json!({
                        "id":link.id,"stage":link.stage.as_str(),"parents":link.parents,
                        "digest":chain.digest_of(&link.id).expect("validated chain link"),
                        "parent_digests":link.parents.iter().map(|parent| chain.digest_of(parent)
                            .expect("validated chain parent")).collect::<Vec<_>>()
                    })
                })
                .collect(),
        ))
    }
    /// Queryable original identities, diagnostics, warnings and semantic action contributors.
    #[must_use]
    pub fn summary(&self) -> Value {
        let contributors = self
            .contract
            .as_ref()
            .map(|contract| {
                self.actions
                    .iter()
                    .map(|id| {
                        let action = contract
                            .actions
                            .iter()
                            .find(|action| &action.id == id)
                            .expect("validated action");
                        let coordinates = action
                            .utility
                            .inputs_used()
                            .iter()
                            .filter_map(|index| {
                                let quantity = action.inputs.get(*index)?;
                                self.coordinates
                                    .iter()
                                    .find(|source| source.require_same_coordinate(quantity).is_ok())
                                    .map(ScientificQuantityWire::from)
                            })
                            .collect::<Vec<_>>();
                        json!({"action_id":id,"source_coordinates":coordinates})
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        if let Some(external) = &self.external {
            return json!({"source_artifact_digest":blake3::hash(&self.source).to_hex().to_string(),
                "source_kind":"external_bound_claim", "provenance_id":external.metadata().identity.provider_fingerprint, "source_execution":null, "original_acceptance":{"recognized":true,"claim_present":true,"verified_references":false,"unresolved_dependencies":[],"restriction":"external attestation; original provider verification covers only its exact declared request"},
                "source_identities":external.metadata().identity, "lineage":self.lineage().expect("validated original source chain"),
                "coordinates":self.coordinates.iter().map(ScientificQuantityWire::from).collect::<Vec<_>>(),
                "diagnostics":[],"diagnostic_availability":"not_retained_by_original_external_claim_format", "warnings":[],
                "point_status":external.metadata().identity.point_status,"trust":external.metadata().identity.trust,
                "native_authority_issued":false,"calibration_license_issued":false,
                "contract_identity":self.contract.as_ref().and_then(|contract|contract.identity().ok()),"action_contributors":contributors,
                "scope":"original_external_source; no transformed numeric diagnostic or trust/calibration upgrade"});
        }
        let response = self.native_original().body.response.as_ref().expect("validated response");
        json!({"source_artifact_digest":hex(&self.native_original().artifact_digest),
            "original_acceptance": {"recognized":self.native_original().acceptance.recognized,"verified_references":self.native_original().acceptance.verified_references,"claim_present":self.native_original().acceptance.claim_present,"unresolved_dependencies":self.native_original().acceptance.unresolved.iter().map(AsRef::as_ref).collect::<Vec<_>>(),"restriction":self.native_original().acceptance.restriction.as_deref()},
            "source_identities":self.native_original().contract.as_ref().map(|contract| &contract.identities),
            "lineage":self.lineage().expect("validated original source chain"),
            "source_execution":self.native_original().contract.as_ref().and_then(|contract| contract.execution.as_ref()),
            "provenance_id":response.provenance_id,
            "coordinates":self.coordinates.iter().map(ScientificQuantityWire::from).collect::<Vec<_>>(),
            "diagnostics":response.support.diagnostics,"warnings":response.support.warnings,
            "point_status":response.support.point_status,"summary_support":response.support.status,
            "contract_identity":self.contract.as_ref().and_then(|contract| contract.identity().ok()),
            "action_contributors":contributors,
            "scope":"original_source_diagnostics; no transformed local diagnostic, executable state, trust upgrade or calibration license"})
    }
    /// Export original bytes and declarations; consumers rederive every diagnostic and mapping.
    pub fn export(&self) -> Result<Vec<u8>, SourceEvidenceError> {
        let mut bytes = MAGIC.to_vec();
        let wire = EvidenceWire {
            source: self.source.to_vec(),
            contract: self
                .contract
                .as_ref()
                .map(contract_to_json)
                .transpose()
                .map_err(|_| invalid("source_evidence.invalid_contract"))?,
            actions: self.actions.clone(),
            external_identity: self
                .external
                .as_ref()
                .map(|external| external.metadata().identity.clone()),
        };
        ciborium::into_writer(&wire, &mut bytes)
            .map_err(|_| invalid("source_evidence.invalid_artifact"))?;
        if bytes.len() > MAX_SOURCE_EVIDENCE_BYTES {
            return Err(invalid("source_evidence.limits_exceeded"));
        }
        Ok(bytes)
    }
    /// Independently consume original scientific result and rebuild semantic contributors.
    pub fn consume(bytes: &[u8]) -> Result<Self, SourceEvidenceError> {
        if bytes.len() > MAX_SOURCE_EVIDENCE_BYTES {
            return Err(invalid("source_evidence.limits_exceeded"));
        }
        if !bytes.starts_with(MAGIC) {
            return Err(invalid("source_evidence.invalid_artifact"));
        }
        let mut remainder = &bytes[MAGIC.len()..];
        let wire: EvidenceWire = ciborium::from_reader(&mut remainder)
            .map_err(|_| invalid("source_evidence.invalid_artifact"))?;
        if !remainder.is_empty() {
            return Err(invalid("source_evidence.invalid_artifact"));
        }
        let result = if let Some(expected) = wire.external_identity {
            let result = Self::from_external(&wire.source)?;
            if result.external.as_ref().expect("validated external source").metadata().identity
                != expected
            {
                return Err(invalid("source_evidence.source_binding_mismatch"));
            }
            result
        } else {
            Self::from_result(&wire.source)?
        };
        match wire.contract {
            Some(contract) => result.project(&contract, &wire.actions),
            None if wire.actions.is_empty() => Ok(result),
            None => Err(invalid("source_evidence.action_mismatch")),
        }
    }
}
fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes.iter().fold(String::with_capacity(bytes.len() * 2), |mut result, byte| {
        let _ = write!(result, "{byte:02x}");
        result
    })
}
