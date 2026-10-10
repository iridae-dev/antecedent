//! Finite source-law to terminal-decision and study-ranking binding.
//!
//! Consumption replays the original source and ranking consumers, projects one named
//! supported scalar coordinate, and checks the exact ordered prior and full decision
//! table. Source trust and calibration are retained; arithmetic replay grants no new
//! inference or provider-verification standing.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
use crate::decision_artifact::source_digest;
use crate::design_ranking_artifact::{
    ConsumeExpectation, DecisionWire, DesignRankingArtifactWire, PriorWire,
    consume_wire as consume_ranking,
};
use antecedent_io::container::{
    ArtifactManifest, EncodedArtifact, SectionBytes, section_descriptor,
};
use antecedent_io::convert::{from_cbor, to_cbor};
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionTrust,
    DrawAlignment,
};
use antecedent_io::error::IoError;
use antecedent_io::quantity_wire::{DistributionMeaningWire, ScientificQuantityWire};
use antecedent_io::reader::ArtifactReader;
use antecedent_io::wire::{ArtifactKind, ProvenanceWire, SemanticVersion};
use serde::{Deserialize, Serialize};

/// Container kind for a finite rollout/study-update binding.
pub const ROLLOUT_KIND: &str = "rollout_state_v1";
/// Byte bound checked before decoding.
pub const MAX_ROLLOUT_BYTES: usize = 32 * 1024 * 1024;
const SECTION: &str = "state_binding.body";

/// Explicit interpretation of the actual source scalar law as the decision state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateInterpretation {
    /// Actual parameter or causal-functional posterior realizations.
    PosteriorState,
    /// Actual interventional predictive realizations explicitly named `state`.
    InterventionalState,
}

/// Source identity and standing retained independently by a consumer.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RolloutSource {
    /// Complete scientific and provider coordinates.
    pub identity: DistributionIdentity,
    /// Original provider standing, never upgraded by replay.
    pub trust: DistributionTrust,
    /// Original inferential calibration label.
    pub calibration: DistributionCalibration,
}

/// Independently retained scientific inputs and terminal decision.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RolloutExpectation {
    /// Original source coordinates and standing.
    pub source: RolloutSource,
    /// Named scalar state coordinate.
    pub state: ScientificQuantityWire,
    /// Explicit scope of the state-law interpretation.
    pub interpretation: StateInterpretation,
    /// Digest of the complete original joint draws and coordinates.
    pub source_digest: String,
    /// Complete terminal actions, utilities, ordered prior and admissibility.
    pub decision: DecisionWire,
    /// Independently consumed study-ranking identity.
    pub ranking_identity: String,
}

/// Durable source, terminal-decision and ranking binding.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RolloutArtifact {
    /// Body schema version.
    pub version: u32,
    /// Scientific source and decision declaration.
    pub binding: RolloutExpectation,
    /// Original bounded source artifact; no provider is manufactured by consumption.
    pub source_artifact: Vec<u8>,
    /// Original ranking artifact, consumed through its existing verifier.
    pub ranking_artifact: Vec<u8>,
    /// Original scoped diagnostics cited by the finite law; this is not native row authority.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub source_evidence: Vec<u8>,
    /// Canonical digest of the full body.
    pub digest: String,
}

fn refused(detail: &'static str) -> IoError {
    IoError::Refused {
        code: antecedent_core::reason_code!("invalid_argument"),
        message: detail.into(),
    }
}
fn unsupported(detail: &'static str) -> IoError {
    IoError::Refused {
        code: antecedent_core::reason_code!("route_not_supported"),
        message: detail.into(),
    }
}

impl RolloutArtifact {
    /// Bind a supplied source and an actually consumed ranking to the declared decision.
    ///
    /// # Errors
    /// Unsupported state-law meanings, missing support, mismatched ordered prior,
    /// decision, source standing or ranking identity refuse.
    pub fn new(
        source_artifact: Vec<u8>,
        ranking_artifact: Vec<u8>,
        binding: RolloutExpectation,
    ) -> Result<Self, IoError> {
        let mut value = Self {
            version: 1,
            binding,
            source_artifact,
            ranking_artifact,
            source_evidence: Vec::new(),
            digest: String::new(),
        };
        value.seal()?;
        value.verify(&value.binding)?;
        Ok(value)
    }

    /// Attach original semantic diagnostics without asserting native law row origin.
    /// # Errors
    /// The original evidence must match the law coordinates and snapshot.
    pub fn with_source_evidence(mut self, bytes: Vec<u8>) -> Result<Self, IoError> {
        self.source_evidence = bytes;
        self.seal()?;
        self.verify(&self.binding)?;
        Ok(self)
    }

    /// Independently consume the original scoped diagnostic citation, when retained.
    /// # Errors
    /// Malformed or semantically incompatible original evidence refuses.
    pub fn diagnostic_evidence(
        &self,
    ) -> Result<Option<crate::source_evidence::SourceEvidence>, IoError> {
        if self.source_evidence.is_empty() {
            return Ok(None);
        }
        let evidence = crate::source_evidence::SourceEvidence::consume(&self.source_evidence)
            .map_err(|_| refused("rollout.source_evidence_mismatch"))?;
        let quantities = self
            .binding
            .source
            .identity
            .quantities
            .iter()
            .cloned()
            .map(antecedent_core::ScientificQuantity::try_from)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| refused("rollout.source_evidence_mismatch"))?;
        evidence
            .require_quantity_binding(&quantities, &self.binding.source.identity.snapshot_id)
            .map_err(|_| refused("rollout.source_evidence_mismatch"))?;
        Ok(Some(evidence))
    }

    /// Canonical body digest independent of container display identifiers.
    /// # Errors
    /// Wire encoding failure.
    pub fn recomputed_digest(&self) -> Result<String, IoError> {
        #[derive(Serialize)]
        struct DigestBody<'a> {
            version: u32,
            binding: &'a RolloutExpectation,
            source_artifact: &'a [u8],
            ranking_artifact: &'a [u8],
            #[serde(skip_serializing_if = "<[u8]>::is_empty")]
            source_evidence: &'a [u8],
            digest: &'static str,
        }
        self.check_bounds()?;
        let body = DigestBody {
            version: self.version,
            binding: &self.binding,
            source_artifact: &self.source_artifact,
            ranking_artifact: &self.ranking_artifact,
            source_evidence: &self.source_evidence,
            digest: "",
        };
        Ok(blake3::hash(&to_cbor(&body)?).to_hex().to_string())
    }

    fn check_bounds(&self) -> Result<(), IoError> {
        let oversized_table = match &self.binding.decision.utility {
            crate::design_ranking_artifact::UtilityWire::Table { rows } => {
                rows.len() > 65_536
                    || rows
                        .iter()
                        .try_fold(0_usize, |sum, row| sum.checked_add(row.len()))
                        .is_none_or(|cells| cells > 4 * 1024 * 1024)
            }
            crate::design_ranking_artifact::UtilityWire::Affine { coefficients } => {
                coefficients.len() > 65_536
            }
        };
        if self.source_evidence.len() > crate::source_evidence::MAX_SOURCE_EVIDENCE_BYTES
            || self.source_artifact.len()
                > antecedent_io::distribution_artifact::MAX_DISTRIBUTION_ARTIFACT_BYTES
            || self.ranking_artifact.len()
                > crate::design_ranking_artifact::MAX_DESIGN_RANKING_ARTIFACT_BYTES
            || self.binding.decision.action_ids.len() > 65_536
            || self.binding.source.identity.quantities.len() > 1_024
            || self.binding.decision.admissible.len() > 65_536
            || oversized_table
            || matches!(&self.binding.decision.prior, PriorWire::Draws { states } if states.len() > 65_536)
        {
            return Err(IoError::TooLarge);
        }
        Ok(())
    }
    /// Seal a body; sealing alone does not establish its scientific validity.
    /// # Errors
    /// Wire encoding failure.
    pub fn seal(&mut self) -> Result<(), IoError> {
        self.digest = self.recomputed_digest()?;
        Ok(())
    }
    /// Full terminal-decision fingerprint, also used across bundle ranking edges.
    /// # Errors
    /// Wire encoding failure.
    pub fn decision_digest(&self) -> Result<String, IoError> {
        decision_digest(&self.binding.decision)
    }

    /// Independently replay source projection and ranking under retained expectations.
    /// # Errors
    /// Malformed, unsupported, changed or semantically inconsistent inputs refuse.
    pub fn verify(&self, expected: &RolloutExpectation) -> Result<(), IoError> {
        self.diagnostic_evidence()?;
        if self.version != 1 || self.digest != self.recomputed_digest()? {
            return Err(refused("rollout.identity_mismatch"));
        }
        if to_cbor(&self.binding)? != to_cbor(expected)? {
            return Err(refused("rollout.expected_binding_mismatch"));
        }
        let source =
            DistributionArtifact::from_bytes(&self.source_artifact, &expected.source.identity)?;
        let metadata = source.metadata();
        if metadata.trust != expected.source.trust
            || metadata.calibration != expected.source.calibration
        {
            return Err(refused("rollout.source_standing_mismatch"));
        }
        if source_digest(&source) != expected.source_digest {
            return Err(refused("rollout.source_digest_mismatch"));
        }
        let coordinate = source
            .quantities()
            .iter()
            .position(|q| q == &expected.state)
            .ok_or_else(|| refused("rollout.state_coordinate_mismatch"))?;
        source.mean(coordinate)?; // invokes the existing actual support check
        if metadata.weights.is_some() || metadata.identity.alignment != DrawAlignment::Joint {
            return Err(unsupported("rollout.unweighted_joint_state_required"));
        }
        let authorized = match expected.interpretation {
            StateInterpretation::PosteriorState => matches!(
                source.semantic(),
                DistributionMeaningWire::ParameterPosterior
                    | DistributionMeaningWire::CausalFunctionalPosterior
            ),
            StateInterpretation::InterventionalState => {
                source.semantic() == DistributionMeaningWire::InterventionalPredictive
                    && expected.state.functional_id == "state"
            }
        };
        if !authorized {
            return Err(unsupported("rollout.state_interpretation_mismatch"));
        }
        let PriorWire::Draws { states } = &expected.decision.prior else {
            return Err(unsupported("rollout.finite_draw_state_required"));
        };
        if states.len() != source.n_draws()
            || states
                .iter()
                .zip(source.draws().chunks_exact(source.shape()[1]))
                .any(|(state, row)| state.to_bits() != row[coordinate].to_bits())
        {
            return Err(refused("rollout.ordered_prior_mismatch"));
        }
        let wire =
            DesignRankingArtifactWire::from_bytes(&self.ranking_artifact).map_err(|error| {
                IoError::Refused {
                    code: antecedent_core::reason_code!("invalid_argument"),
                    message: format!("rollout.ranking_invalid: {error}"),
                }
            })?;
        let ranking = consume_ranking(
            &wire,
            &ConsumeExpectation {
                artifact_identity: Some(expected.ranking_identity.clone()),
                decision_contract_identity: Some(expected.decision.contract_identity.clone()),
                ..ConsumeExpectation::default()
            },
        )
        .map_err(|error| IoError::Refused {
            code: antecedent_core::reason_code!("invalid_argument"),
            message: format!("rollout.ranking_invalid: {error}"),
        })?;
        if wire.decision != expected.decision || ranking.identity != expected.ranking_identity {
            return Err(refused("rollout.terminal_decision_mismatch"));
        }
        if !wire.source_digests.contains(&expected.source_digest) {
            return Err(refused("rollout.ranking_source_mismatch"));
        }
        Ok(())
    }

    /// Write a validated bounded checksummed container.
    /// # Errors
    /// Typed ancestry from the independently consumed original source and ranking.
    /// The finite state projection is bound to the original ordered ranking prior.
    /// # Errors
    /// Any original source/ranking refusal or invalid derivation chain.
    pub fn provenance_chain(&self) -> Result<antecedent_core::ProvenanceChain, IoError> {
        use antecedent_core::{CompositionLink, CompositionStage, ProvenanceChain};
        self.verify(&self.binding)?;
        let wire = DesignRankingArtifactWire::from_bytes(&self.ranking_artifact)
            .map_err(|_| refused("rollout.ranking_invalid"))?;
        let ranking = wire.provenance_chain().map_err(|_| refused("rollout.ranking_invalid"))?;
        let causal = format!("causal_contract:{}", self.binding.source.identity.causal_contract_id);
        let data = format!("snapshot:{}", self.binding.source.identity.snapshot_id);
        let law = format!("distribution:{}", self.binding.source_digest);
        let state =
            format!("state_projection:{}", blake3::hash(&to_cbor(&self.binding.state)?).to_hex());
        let link = |id, stage, parents| CompositionLink {
            id,
            stage,
            parents,
            declared_parent_digests: None,
        };
        let mut links = vec![
            link(causal.clone(), CompositionStage::CausalContract, vec![]),
            link(data.clone(), CompositionStage::Data, vec![]),
            link(law.clone(), CompositionStage::DistributionArtifact, vec![causal, data]),
            link(state.clone(), CompositionStage::Transformation, vec![law.clone()]),
        ];
        for original in ranking.links() {
            if original.id == law {
                continue;
            }
            let mut original = original.clone();
            if original.stage == CompositionStage::DecisionContract {
                original.parents.push(state.clone());
            }
            original.declared_parent_digests = None;
            links.push(original);
        }
        ProvenanceChain::new(links).map_err(|_| refused("rollout.ranking_invalid"))
    }

    /// Invalid body, empty display identity or oversized encoded artifact.
    pub fn to_bytes(&self, artifact_id: &str) -> Result<Vec<u8>, IoError> {
        self.verify(&self.binding)?;
        if artifact_id.trim().is_empty() {
            return Err(refused("rollout.artifact_id_missing"));
        }
        let body = to_cbor(self)?;
        let artifact = EncodedArtifact {
            manifest: ArtifactManifest {
                format_version: antecedent_io::migrate::STABLE_FORMAT,
                minimum_reader_version: antecedent_io::migrate::STABLE_FORMAT,
                artifact_kind: ArtifactKind::Other(ROLLOUT_KIND.into()),
                library_version: SemanticVersion::from_crate_version(env!("CARGO_PKG_VERSION"))?,
                artifact_id: artifact_id.into(),
                sections: vec![section_descriptor(SECTION, "application/cbor", &body)],
                provenance: ProvenanceWire {
                    note: "finite source-state terminal decision binding".into(),
                },
            },
            sections: vec![SectionBytes::new(SECTION, body)],
        };
        let mut bytes = Vec::new();
        artifact.write_to(&mut bytes)?;
        if bytes.len() > MAX_ROLLOUT_BYTES {
            return Err(IoError::TooLarge);
        }
        Ok(bytes)
    }

    /// Decode a bounded container; scientific verification remains explicit.
    /// # Errors
    /// Corruption, wrong kind/version or byte limits refuse before replay.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, IoError> {
        if bytes.len() > MAX_ROLLOUT_BYTES {
            return Err(IoError::TooLarge);
        }
        let mut reader = ArtifactReader::open_seek(std::io::Cursor::new(bytes))?;
        if reader.manifest().artifact_kind != ArtifactKind::Other(ROLLOUT_KIND.into())
            || reader.manifest().sections.len() != 1
            || reader.manifest().sections[0].id != SECTION
        {
            return Err(refused("rollout.artifact_kind_mismatch"));
        }
        if reader.manifest().sections[0].uncompressed_size > MAX_ROLLOUT_BYTES as u64 {
            return Err(IoError::TooLarge);
        }
        let body = reader.load_section(SECTION)?;
        let artifact: Self = from_cbor(body.as_bytes())?;
        artifact.check_bounds()?;
        Ok(artifact)
    }
}

/// Full prior/action/utility/admissibility fingerprint for a ranking edge.
/// # Errors
/// Wire encoding failure.
pub fn decision_digest(decision: &DecisionWire) -> Result<String, IoError> {
    Ok(blake3::hash(&to_cbor(decision)?).to_hex().to_string())
}

/// Independently consume source-backed finite rollout bytes.
/// # Errors
/// Changed expectations, unsupported source law or inconsistent decision/ranking.
pub fn consume(bytes: &[u8], expected: &RolloutExpectation) -> Result<RolloutArtifact, IoError> {
    let artifact = RolloutArtifact::from_bytes(bytes)?;
    artifact.verify(expected)?;
    Ok(artifact)
}
