//! Replayable exact images of identified sets and finite joint outcome laws.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::ScientificQuantity;
use antecedent_contract::bridge::{
    BridgeContract, BridgedIdentifiedSet, FixedBridgeDomain, IdentifiedSetSource, ValueInterval,
};
use antecedent_core::{CanonicalWriter, ExecutionContext, IdentifiedSet, SemanticDigest};
use serde::{Deserialize, Serialize};

use crate::bridge_artifact::{
    ContractWire, ScientificBridgeArtifactError, contract_from_wire, contract_to_wire,
};
use crate::bridge_joint_outcomes::{
    JointOutcomeBridgeError, JointOutcomeImage, JointOutcomeMap, JointOutcomeTerm,
};
use crate::distribution_artifact::{
    DistributionArtifact, DistributionIdentity, MAX_DISTRIBUTION_ARTIFACT_BYTES,
};
use crate::quantity_wire::ScientificQuantityWire;
use crate::{ArtifactContainer, IoError, OutSection};

/// The artifact feature used by the compiled exact-image routes.
pub const BRIDGE_IMAGE_ARTIFACT: &str = "scientific_bridge_exact_image_v1";
const META: &str = "image.declaration";
const INPUT: &str = "image.source";
const LIMIT: usize = 2 * MAX_DISTRIBUTION_ARTIFACT_BYTES;

/// The scientific inputs to one deterministic mapping.
#[derive(Clone, Debug)]
pub enum BridgeImageRequest {
    /// Closed identified source interval and its fixed bridge.
    IdentifiedSet {
        /// Source and target coordinates, mapping and identifying premises.
        bridge: Box<BridgeContract>,
        /// Original set and its evidence identity.
        source: IdentifiedSetSource,
        /// Valid mapping inputs.
        domain: FixedBridgeDomain,
    },
    /// Fixed vector mapping on the aligned source rows.
    JointOutcomes {
        /// Outputs and their coordinate-specific terms.
        map: JointOutcomeMap,
        /// Full source table.
        source: DistributionArtifact,
    },
}

/// The carrier produced by a checked scientific image operation.
#[derive(Clone, Debug)]
pub enum BridgeImageResult {
    /// Sharp interval of compatible target values.
    IdentifiedSet(BridgedIdentifiedSet),
    /// Weighted, paired target outcomes.
    JointOutcomes(JointOutcomeImage),
}

impl BridgeImageResult {
    /// Identity binding the map, source evidence and domain.
    pub const fn identity(&self) -> SemanticDigest {
        match self {
            Self::IdentifiedSet(v) => v.identity,
            Self::JointOutcomes(v) => v.identity,
        }
    }
    fn readout(&self) -> Result<[u8; 32], IoError> {
        let mut key = CanonicalWriter::new("antecedent.bridge.image.readout.v1");
        key.bytes(self.identity().as_bytes());
        match self {
            Self::IdentifiedSet(v) => {
                key.word(v.set.lower.to_bits()).word(v.set.upper.to_bits());
            }
            Self::JointOutcomes(v) => {
                key.bytes(&v.distribution.to_bytes("image")?);
            }
        }
        Ok(key.finish())
    }
}

/// A refused scientific image or invalid portable artifact.
#[derive(Debug, thiserror::Error)]
pub enum BridgeImageError {
    /// Scientific contract or alignment failure.
    #[error(transparent)]
    Mapping(#[from] JointOutcomeBridgeError),
    /// A bridge declaration could not be encoded or restored.
    #[error(transparent)]
    Declaration(#[from] ScientificBridgeArtifactError),
    /// Container, resource or numerical source failure.
    #[error(transparent)]
    Artifact(#[from] IoError),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TermWire {
    bridge: ContractWire,
    domain: [u64; 2],
}

impl TermWire {
    fn new(
        bridge: &BridgeContract,
        domain: FixedBridgeDomain,
    ) -> Result<Self, ScientificBridgeArtifactError> {
        let mut wire = contract_to_wire(bridge)?;
        wire.assumptions.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(Self { bridge: wire, domain: [domain.0.low.to_bits(), domain.0.high.to_bits()] })
    }
    fn restore(&self) -> Result<JointOutcomeTerm, ScientificBridgeArtifactError> {
        Ok(JointOutcomeTerm {
            contract: contract_from_wire(&self.bridge)?,
            domain: FixedBridgeDomain(ValueInterval {
                low: f64::from_bits(self.domain[0]),
                high: f64::from_bits(self.domain[1]),
            }),
        })
    }
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
enum InputWire {
    IdentifiedSet {
        term: Box<TermWire>,
        quantity: ScientificQuantityWire,
        bounds: [u64; 2],
        evidence: [u8; 32],
    },
    JointOutcomes {
        terms: Vec<Vec<TermWire>>,
        source: DistributionIdentity,
    },
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Meta {
    version: u16,
    program: [u8; 32],
    result: [u8; 32],
    readout: [u8; 32],
    input: InputWire,
}

impl BridgeImageRequest {
    fn wire(&self) -> Result<(InputWire, Vec<u8>), BridgeImageError> {
        Ok(match self {
            Self::IdentifiedSet { bridge, source, domain } => (
                InputWire::IdentifiedSet {
                    term: Box::new(TermWire::new(bridge, *domain)?),
                    quantity: (&source.quantity).into(),
                    bounds: [source.set.lower.to_bits(), source.set.upper.to_bits()],
                    evidence: *source.claim_identity.as_bytes(),
                },
                vec![],
            ),
            Self::JointOutcomes { map, source } => {
                let terms = map
                    .outputs
                    .iter()
                    .map(|terms| {
                        terms.iter().map(|t| TermWire::new(&t.contract, t.domain)).collect()
                    })
                    .collect::<Result<Vec<Vec<_>>, _>>()?;
                (
                    InputWire::JointOutcomes { terms, source: source.metadata().identity.clone() },
                    source.to_bytes("joint-outcome-source")?,
                )
            }
        })
    }
    /// Bind all scientific declarations and exact source bits at compile time.
    pub fn input_identity(&self) -> Result<SemanticDigest, BridgeImageError> {
        let (wire, data) = self.wire()?;
        let mut key = CanonicalWriter::new("antecedent.bridge.exact_image_request.v1");
        key.bytes(&crate::to_cbor(&wire)?).bytes(&data);
        Ok(SemanticDigest::from_bytes(key.finish()))
    }
    /// Execute the mapping under the caller's admission and cancellation controls.
    pub fn evaluate(&self, ctx: &ExecutionContext) -> Result<BridgeImageResult, BridgeImageError> {
        match self {
            Self::IdentifiedSet { bridge, source, domain } => {
                let _lease = ctx.acquire_serial_numerical_admission(128).map_err(IoError::from)?;
                Ok(BridgeImageResult::IdentifiedSet(
                    bridge
                        .identified_set_image(source, *domain)
                        .map_err(JointOutcomeBridgeError::from)?,
                ))
            }
            Self::JointOutcomes { map, source } => {
                Ok(BridgeImageResult::JointOutcomes(map.apply(source, ctx)?))
            }
        }
    }
    /// Retain the full source, declarations and the numerical readout of the compiled operation.
    pub fn export(
        &self,
        result: &BridgeImageResult,
        program: SemanticDigest,
        artifact_id: &str,
    ) -> Result<Vec<u8>, BridgeImageError> {
        let (input, bytes) = self.wire()?;
        let meta = Meta {
            version: 1,
            program: *program.as_bytes(),
            result: *result.identity().as_bytes(),
            readout: result.readout()?,
            input,
        };
        Ok(crate::artifact_codec::encode_sectioned_io(
            &ArtifactContainer {
                kind: BRIDGE_IMAGE_ARTIFACT,
                note: "scientific_bridge",
                max_bytes: LIMIT,
                artifact_id,
                library_version: env!("CARGO_PKG_VERSION"),
            },
            META,
            &meta,
            &[OutSection { id: INPUT, content_type: crate::CONTENT_BYTES, bytes: &bytes }],
            BRIDGE_IMAGE_ARTIFACT,
        )?)
    }
}

/// The inputs of an opened image, still requiring compilation and numerical replay.
pub struct StoredBridgeImage {
    /// Inputs reconstructed from the container.
    pub request: BridgeImageRequest,
    program: SemanticDigest,
    result: SemanticDigest,
    readout: [u8; 32],
}

impl StoredBridgeImage {
    /// Check container bounds, version and independent expectations before reconstructing inputs.
    pub fn open(
        bytes: &[u8],
        program: SemanticDigest,
        result: SemanticDigest,
    ) -> Result<Self, BridgeImageError> {
        let stored = Self::decode(bytes)?;
        if stored.program != program || stored.result != result {
            return Err(ScientificBridgeArtifactError::IdentityMismatch {
                field: "retained image program or result",
            }
            .into());
        }
        Ok(stored)
    }

    /// Open an embedded node before replay. A caller must check the result with
    /// [`Self::verify`] against a program compiled from its retained declaration.
    pub fn decode(bytes: &[u8]) -> Result<Self, BridgeImageError> {
        let (meta, [source_bytes]): (Meta, _) =
            crate::artifact_codec::decode_sectioned_io_versioned(
                BRIDGE_IMAGE_ARTIFACT,
                LIMIT,
                META,
                &[INPUT],
                BRIDGE_IMAGE_ARTIFACT,
                1,
                bytes,
            )?;
        let request = match meta.input {
            InputWire::IdentifiedSet { term, quantity, bounds, evidence } => {
                if !source_bytes.is_empty() {
                    return Err(ScientificBridgeArtifactError::ReplayMismatch(
                        "unexpected set payload",
                    )
                    .into());
                }
                let term = term.restore()?;
                let quantity = ScientificQuantity::try_from(quantity)
                    .map_err(|_| ScientificBridgeArtifactError::ReplayMismatch("set coordinate"))?;
                BridgeImageRequest::IdentifiedSet {
                    bridge: Box::new(term.contract),
                    domain: term.domain,
                    source: IdentifiedSetSource {
                        quantity,
                        set: IdentifiedSet::try_new(
                            f64::from_bits(bounds[0]),
                            f64::from_bits(bounds[1]),
                        )
                        .map_err(|_| ScientificBridgeArtifactError::ReplayMismatch("set bounds"))?,
                        claim_identity: SemanticDigest::from_bytes(evidence),
                    },
                }
            }
            InputWire::JointOutcomes { terms, source } => {
                let source = DistributionArtifact::from_bytes(&source_bytes, &source)?;
                let outputs = terms
                    .iter()
                    .map(|ts| ts.iter().map(TermWire::restore).collect())
                    .collect::<Result<Vec<Vec<_>>, _>>()?;
                BridgeImageRequest::JointOutcomes { map: JointOutcomeMap { outputs }, source }
            }
        };
        Ok(Self {
            request,
            program: SemanticDigest::from_bytes(meta.program),
            result: SemanticDigest::from_bytes(meta.result),
            readout: meta.readout,
        })
    }

    /// Compare compiler and result identities, and independently regenerated numerical bits.
    pub fn verify(
        &self,
        program: SemanticDigest,
        result: &BridgeImageResult,
    ) -> Result<(), BridgeImageError> {
        if self.program != program
            || self.result != result.identity()
            || self.readout != result.readout()?
        {
            return Err(
                ScientificBridgeArtifactError::ReplayMismatch("exact image or program").into()
            );
        }
        Ok(())
    }
}
