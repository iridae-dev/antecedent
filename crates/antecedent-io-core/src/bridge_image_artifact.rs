//! Replayable exact images of identified sets and finite joint outcome laws.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::io::Cursor;

use antecedent_contract::ScientificQuantity;
use antecedent_contract::bridge::{
    BridgeContract, BridgedIdentifiedSet, FixedBridgeDomain, IdentifiedSetSource, ValueInterval,
};
use antecedent_core::execution::{AdmissionError, AdmissionLease};
use antecedent_core::{CanonicalWriter, ExecutionContext, IdentifiedSet, SemanticDigest};
use serde::{Deserialize, Serialize};

use crate::bridge_artifact::{
    ContractWire, ScientificBridgeArtifactError, contract_from_wire, contract_to_wire,
};
pub use crate::bridge_interval_decision::BridgeIntervalDecision;
use crate::bridge_joint_outcomes::{
    JointOutcomeBridgeError, JointOutcomeImage, JointOutcomeMap, JointOutcomeTerm,
};
use crate::bridge_posterior_interval::{PosteriorAffineBridge, PosteriorBridgeInterval};
use crate::bridge_trial_interval::{
    TrialBridgeConfidenceInterval, TrialBridgeExperiment, TrialBridgeReleaseDesign, TrialMeanStudy,
};
use crate::distribution_artifact::{
    DistributionArtifact, DistributionIdentity, MAX_DISTRIBUTION_ARTIFACT_BYTES,
};
use crate::quantity_wire::ScientificQuantityWire;
use crate::{ArtifactContainer, ArtifactReader, IoError, OutSection};

/// The artifact feature used by the compiled exact-image routes.
pub const BRIDGE_IMAGE_ARTIFACT: &str = "scientific_bridge_exact_image_v1";
const META: &str = "image.declaration";
const INPUT: &str = "image.source";
const LIMIT: usize = 2 * MAX_DISTRIBUTION_ARTIFACT_BYTES;

/// The scientific inputs to one deterministic mapping.
#[derive(Clone, Debug)]
pub enum BridgeImageRequest {
    /// A three-way decision using a specified interval and a cutoff in its coordinate.
    IntervalThreshold {
        /// One joint posterior or trial confidence interval.
        interval: Box<BridgeImageRequest>,
        /// Target and utility cutoff in the declared measurement units.
        quantity: ScientificQuantity,
        /// Select a direction only when the entire interval lies strictly on one side.
        threshold: f64,
    },
    /// Confidence interval with validation-trial refits and a declared source study.
    TrialConfidence(Box<TrialBridgeExperiment>),
    /// Equal-tailed credible interval of an affine bridge under one joint posterior.
    JointPosterior {
        /// Scientific contract and coefficient shift coordinates.
        map: Box<PosteriorAffineBridge>,
        /// Joint posterior over the source and both mapping coefficients.
        source: DistributionArtifact,
        /// Requested probability of the equal-tailed interval.
        level: f64,
    },
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
    /// Decision induced by a declared cutoff on a specific bridged interval.
    IntervalThreshold(BridgeIntervalDecision),
    /// Whole-trial percentile confidence interval and its measured source design.
    TrialConfidence(TrialBridgeConfidenceInterval),
    /// Posterior interval and the mapped posterior law from which it was computed.
    JointPosterior(PosteriorBridgeInterval),
    /// Sharp interval of compatible target values.
    IdentifiedSet(BridgedIdentifiedSet),
    /// Weighted, paired target outcomes.
    JointOutcomes(JointOutcomeImage),
}

impl BridgeImageResult {
    /// Identity binding the map, source evidence and domain.
    pub const fn identity(&self) -> SemanticDigest {
        match self {
            Self::IntervalThreshold(v) => SemanticDigest::from_bytes(v.identity),
            Self::TrialConfidence(v) => v.interval.identity,
            Self::JointPosterior(v) => v.identity,
            Self::IdentifiedSet(v) => v.identity,
            Self::JointOutcomes(v) => v.identity,
        }
    }
    /// Reserve the simultaneously held result and its temporary encoding during replay.
    #[doc(hidden)]
    pub fn admit_readout(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<AdmissionLease, BridgeImageError> {
        let artifact = match self {
            Self::TrialConfidence(v) => Some(&v.interval.bootstrap),
            Self::JointPosterior(v) => Some(&v.posterior),
            Self::JointOutcomes(v) => Some(&v.distribution),
            _ => None,
        };
        let bytes = artifact.map_or(0, |artifact| {
            let weights = artifact.metadata().weights.as_ref().map_or(0, Vec::len);
            (artifact.draws().len().saturating_add(weights) as u64).saturating_mul(8)
        });
        Ok(admit_bytes(bytes, ctx)?)
    }

    fn readout(&self) -> Result<[u8; 32], IoError> {
        let mut key = CanonicalWriter::new("antecedent.bridge.image.readout.v1");
        key.bytes(self.identity().as_bytes());
        match self {
            Self::IntervalThreshold(v) => {
                // Calibration is looked up under the reader's current evidence. Its
                // record revision is not a different threshold, interval or verdict.
                let mut scientific = v.clone();
                scientific.interval_coverage_record = None;
                scientific.interval_calibration_sha = None;
                if matches!(
                    scientific.reading,
                    crate::bridge_interval_decision::BridgeIntervalReading::SamplingNominal
                ) {
                    // The v1 seal describes the original sampling construction, independent
                    // of whether its historical coverage is still current for the consumer.
                    scientific.reading =
                        crate::bridge_interval_decision::BridgeIntervalReading::SamplingConfidence;
                }
                key.bytes(&crate::to_cbor(&scientific)?);
            }
            Self::TrialConfidence(v) => {
                key.word(v.interval.point.to_bits())
                    .word(v.interval.endpoints[0].to_bits())
                    .word(v.interval.endpoints[1].to_bits())
                    .bytes(&v.interval.bootstrap.to_bytes("bootstrap")?);
            }
            Self::JointPosterior(v) => {
                key.word(v.level.to_bits())
                    .word(v.endpoints[0].to_bits())
                    .word(v.endpoints[1].to_bits())
                    .word(v.attained_mass.to_bits())
                    .bytes(&v.posterior.to_bytes("posterior-image")?);
            }
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

impl BridgeImageError {
    fn io_error(&self) -> Option<&IoError> {
        match self {
            Self::Artifact(error) | Self::Mapping(JointOutcomeBridgeError::Artifact(error)) => {
                Some(error)
            }
            _ => None,
        }
    }

    /// Whether the caller cancelled before or during replay.
    pub fn is_cancelled(&self) -> bool {
        matches!(
            self.io_error(),
            Some(IoError::Admission(AdmissionError::Cancelled) | IoError::StreamCancelled)
        )
    }

    /// Whether the image exceeded its structural or caller execution budget.
    pub fn is_resource_refusal(&self) -> bool {
        !self.is_cancelled()
            && (matches!(
                self.io_error(),
                Some(
                    IoError::Admission(_) | IoError::TooLarge | IoError::StreamBudgetRefused { .. }
                )
            ) || matches!(
                self,
                Self::Declaration(ScientificBridgeArtifactError::LimitsExceeded(_))
            ))
    }
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
    IntervalThreshold {
        interval_declaration: Vec<u8>,
        quantity: ScientificQuantityWire,
        threshold: u64,
    },
    TrialConfidence {
        contract: Box<ContractWire>,
        independent_study_id: Option<String>,
        level: u64,
        seed: u64,
        replicates: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        release_design: Option<TrialBridgeReleaseDesign>,
    },
    JointPosterior {
        term: Box<TermWire>,
        intercept_shift: ScientificQuantityWire,
        slope_shift: ScientificQuantityWire,
        level: u64,
        source: DistributionIdentity,
    },
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
    /// Validate the interval carrier before it is used by a downstream threshold.
    pub fn require_interval(&self) -> Result<(), BridgeImageError> {
        if matches!(self, Self::TrialConfidence(_) | Self::JointPosterior { .. }) {
            return Ok(());
        }
        Err(JointOutcomeBridgeError::from(antecedent_contract::bridge::draws_unusable(
            "threshold source",
            "supply a posterior or trial confidence interval",
        ))
        .into())
    }

    fn wire(&self) -> Result<(InputWire, Vec<u8>), BridgeImageError> {
        Ok(match self {
            Self::IntervalThreshold { interval, quantity, threshold } => {
                interval.require_interval()?;
                let (declaration, source) = interval.wire()?;
                (
                    InputWire::IntervalThreshold {
                        interval_declaration: crate::to_cbor(&declaration)?,
                        quantity: quantity.into(),
                        threshold: threshold.to_bits(),
                    },
                    source,
                )
            }
            Self::TrialConfidence(experiment) => {
                let mut wire = contract_to_wire(&experiment.contract)?;
                wire.assumptions.sort_by(|a, b| a.id.cmp(&b.id));
                let (independent_study_id, data) = match &experiment.study {
                    TrialMeanStudy::Shared => (None, Vec::new()),
                    TrialMeanStudy::Independent { study_id, surrogate_effects } => {
                        let mut ordered = surrogate_effects.clone();
                        ordered.sort_by(f64::total_cmp);
                        (
                            Some(study_id.clone()),
                            ordered.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect(),
                        )
                    }
                };
                (
                    InputWire::TrialConfidence {
                        contract: Box::new(wire),
                        independent_study_id,
                        level: experiment.level.to_bits(),
                        seed: experiment.seed,
                        replicates: experiment.replicates,
                        release_design: experiment.release_design,
                    },
                    data,
                )
            }
            Self::JointPosterior { map, source, level } => (
                InputWire::JointPosterior {
                    term: Box::new(TermWire::new(&map.contract, map.domain)?),
                    intercept_shift: (&map.intercept_shift).into(),
                    slope_shift: (&map.slope_shift).into(),
                    level: level.to_bits(),
                    source: source.metadata().identity.clone(),
                },
                source.to_bytes("joint-posterior")?,
            ),
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
        self.evaluate_image(ctx, false)
    }

    /// Recompute a stored image. Trial bounds without a current coverage receipt have a
    /// nominal reading after comparison to the retained numerical seal.
    #[doc(hidden)]
    pub fn replay_image(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<BridgeImageResult, BridgeImageError> {
        self.evaluate_image(ctx, true)
    }

    fn evaluate_image(
        &self,
        ctx: &ExecutionContext,
        historical_replay: bool,
    ) -> Result<BridgeImageResult, BridgeImageError> {
        match self {
            Self::IntervalThreshold { interval, quantity, threshold } => {
                interval.require_interval()?;
                let value = interval.evaluate_image(ctx, historical_replay)?;
                Ok(BridgeImageResult::IntervalThreshold(BridgeIntervalDecision::evaluate(
                    &value, quantity, *threshold,
                )?))
            }
            Self::TrialConfidence(experiment) => {
                let interval = if historical_replay && experiment.release_design.is_some() {
                    experiment.reconstruct_interval(ctx)?
                } else {
                    experiment.interval(ctx)?
                };
                Ok(BridgeImageResult::TrialConfidence(interval))
            }
            Self::JointPosterior { map, source, level } => {
                Ok(BridgeImageResult::JointPosterior(map.interval(source, *level, ctx)?))
            }
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
    binding: BridgeImageReplayBinding,
}

/// Retained expectations and the working-set lease of an opened image.
pub struct BridgeImageReplayBinding {
    program: SemanticDigest,
    result: SemanticDigest,
    readout: [u8; 32],
    _workspace: AdmissionLease,
    _source_expansion: Option<AdmissionLease>,
}

impl StoredBridgeImage {
    /// Open under the default execution limits and check the independent expectations.
    pub fn open(
        bytes: &[u8],
        program: SemanticDigest,
        result: SemanticDigest,
    ) -> Result<Self, BridgeImageError> {
        Self::open_with_context(bytes, program, result, &ExecutionContext::production_default(0))
    }

    /// Check the independently held identities under the caller's admission and cancellation policy.
    pub fn open_with_context(
        bytes: &[u8],
        program: SemanticDigest,
        result: SemanticDigest,
        ctx: &ExecutionContext,
    ) -> Result<Self, BridgeImageError> {
        Self::decode_admitted(bytes, Some((program, result)), ctx)
    }

    /// Open an embedded node under the default execution limits before numerical verification.
    pub fn decode(bytes: &[u8]) -> Result<Self, BridgeImageError> {
        Self::decode_with_context(bytes, &ExecutionContext::production_default(0))
    }

    /// Admit and decode an embedded node before numerical verification.
    pub fn decode_with_context(
        bytes: &[u8],
        ctx: &ExecutionContext,
    ) -> Result<Self, BridgeImageError> {
        Self::decode_admitted(bytes, None, ctx)
    }

    fn decode_admitted(
        bytes: &[u8],
        expected: Option<(SemanticDigest, SemanticDigest)>,
        ctx: &ExecutionContext,
    ) -> Result<Self, BridgeImageError> {
        let manifest_lease = admit_bytes(bytes.len() as u64, ctx)?;
        let decoded_bytes = decoded_size(bytes, LIMIT)?;
        drop(manifest_lease);
        // Account for the section buffers, restored inputs and their compilation encoding.
        let workspace = admit_bytes(decoded_bytes.max(bytes.len() as u64), ctx)?;
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
        let program = SemanticDigest::from_bytes(meta.program);
        let result = SemanticDigest::from_bytes(meta.result);
        if expected.is_some_and(|expected| expected != (program, result)) {
            return Err(ScientificBridgeArtifactError::IdentityMismatch {
                field: "retained image program or result",
            }
            .into());
        }
        let input = meta.input.unbox_threshold()?;
        let payload_size = source_bytes.len() as u64;
        let source_expansion = if input.has_distribution_payload() {
            let decoded_source = decoded_size(&source_bytes, MAX_DISTRIBUTION_ARTIFACT_BYTES)?;
            let expansion = decoded_source.saturating_sub(payload_size);
            (expansion != 0).then(|| admit_bytes(expansion, ctx)).transpose()?
        } else {
            None
        };
        check_replay_cancellation(ctx)?;
        let request = input.restore(&source_bytes)?;
        check_replay_cancellation(ctx)?;
        Ok(Self {
            request,
            binding: BridgeImageReplayBinding {
                program,
                result,
                readout: meta.readout,
                _workspace: workspace,
                _source_expansion: source_expansion,
            },
        })
    }

    /// Move the admitted input into the compiler while retaining its replay binding and lease.
    pub fn into_parts(self) -> (BridgeImageRequest, BridgeImageReplayBinding) {
        (self.request, self.binding)
    }

    /// Compare compiler and result identities, and independently regenerated numerical bits.
    pub fn verify(
        &self,
        program: SemanticDigest,
        result: &BridgeImageResult,
    ) -> Result<(), BridgeImageError> {
        self.binding.verify(program, result)
    }
}

impl BridgeImageReplayBinding {
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

fn admit_bytes(bytes: u64, ctx: &ExecutionContext) -> Result<AdmissionLease, IoError> {
    ctx.acquire_admission(0, bytes.saturating_mul(8).max(1)).map_err(IoError::from)
}

fn decoded_size(bytes: &[u8], limit: usize) -> Result<u64, IoError> {
    if bytes.len() > limit {
        return Err(IoError::TooLarge);
    }
    let reader = ArtifactReader::open_seek(Cursor::new(bytes))?;
    reader
        .manifest()
        .sections
        .iter()
        .try_fold(0_u64, |sum, section| sum.checked_add(section.uncompressed_size))
        .filter(|size| *size <= limit as u64)
        .ok_or(IoError::TooLarge)
}

fn check_replay_cancellation(ctx: &ExecutionContext) -> Result<(), IoError> {
    if ctx.cancellation.is_cancelled() {
        return Err(AdmissionError::Cancelled.into());
    }
    Ok(())
}

enum RestoreInput {
    Image(InputWire),
    Threshold { interval: InputWire, quantity: ScientificQuantityWire, threshold: u64 },
}

impl RestoreInput {
    fn has_distribution_payload(&self) -> bool {
        let wire = match self {
            Self::Image(wire) | Self::Threshold { interval: wire, .. } => wire,
        };
        matches!(wire, InputWire::JointPosterior { .. } | InputWire::JointOutcomes { .. })
    }

    fn restore(self, source_bytes: &[u8]) -> Result<BridgeImageRequest, BridgeImageError> {
        match self {
            Self::Image(wire) => wire.restore(source_bytes),
            Self::Threshold { interval, quantity, threshold } => {
                Ok(BridgeImageRequest::IntervalThreshold {
                    interval: Box::new(interval.restore(source_bytes)?),
                    quantity: ScientificQuantity::try_from(quantity).map_err(|_| {
                        ScientificBridgeArtifactError::ReplayMismatch("threshold coordinate")
                    })?,
                    threshold: f64::from_bits(threshold),
                })
            }
        }
    }
}

impl InputWire {
    fn unbox_threshold(self) -> Result<RestoreInput, BridgeImageError> {
        match self {
            Self::IntervalThreshold { interval_declaration, quantity, threshold } => {
                let interval = crate::from_cbor(&interval_declaration)?;
                if !matches!(interval, Self::JointPosterior { .. } | Self::TrialConfidence { .. }) {
                    return Err(ScientificBridgeArtifactError::ReplayMismatch(
                        "threshold requires a single interval declaration",
                    )
                    .into());
                }
                Ok(RestoreInput::Threshold { interval, quantity, threshold })
            }
            wire => Ok(RestoreInput::Image(wire)),
        }
    }

    fn restore(self, source_bytes: &[u8]) -> Result<BridgeImageRequest, BridgeImageError> {
        Ok(match self {
            InputWire::IntervalThreshold { .. } => {
                return Err(ScientificBridgeArtifactError::ReplayMismatch(
                    "unexpected nested threshold",
                )
                .into());
            }
            InputWire::TrialConfidence {
                contract,
                independent_study_id,
                level,
                seed,
                replicates,
                release_design,
            } => {
                let study = match independent_study_id {
                    Some(study_id) if source_bytes.len().is_multiple_of(8) => {
                        let surrogate_effects = source_bytes
                            .chunks_exact(8)
                            .map(|w| {
                                let mut arr = [0; 8];
                                arr.copy_from_slice(w);
                                f64::from_le_bytes(arr)
                            })
                            .collect();
                        TrialMeanStudy::Independent { study_id, surrogate_effects }
                    }
                    None if source_bytes.is_empty() => TrialMeanStudy::Shared,
                    _ => {
                        return Err(ScientificBridgeArtifactError::ReplayMismatch(
                            "trial source observations",
                        )
                        .into());
                    }
                };
                BridgeImageRequest::TrialConfidence(Box::new(TrialBridgeExperiment {
                    contract: contract_from_wire(&contract)?,
                    study,
                    level: f64::from_bits(level),
                    seed,
                    replicates,
                    release_design,
                }))
            }
            InputWire::JointPosterior { term, intercept_shift, slope_shift, level, source } => {
                let term = term.restore()?;
                let coordinate = |wire| {
                    ScientificQuantity::try_from(wire).map_err(|_| {
                        ScientificBridgeArtifactError::ReplayMismatch(
                            "posterior coefficient coordinate",
                        )
                    })
                };
                BridgeImageRequest::JointPosterior {
                    map: Box::new(PosteriorAffineBridge {
                        contract: term.contract,
                        domain: term.domain,
                        intercept_shift: coordinate(intercept_shift)?,
                        slope_shift: coordinate(slope_shift)?,
                    }),
                    source: DistributionArtifact::from_bytes(source_bytes, &source)?,
                    level: f64::from_bits(level),
                }
            }
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
                let source = DistributionArtifact::from_bytes(source_bytes, &source)?;
                let outputs = terms
                    .iter()
                    .map(|ts| ts.iter().map(TermWire::restore).collect())
                    .collect::<Result<Vec<Vec<_>>, _>>()?;
                BridgeImageRequest::JointOutcomes { map: JointOutcomeMap { outputs }, source }
            }
        })
    }
}

#[cfg(test)]
mod numerical_replay_tests {
    use super::*;
    use crate::bridge_interval_decision::{BridgeIntervalReading, ThresholdVerdict};

    #[test]
    fn decision_numerical_receipt_does_not_depend_on_current_coverage_registry() {
        let raw = BridgeIntervalDecision {
            identity: [1; 32],
            interval_identity: [2; 32],
            quantity: ScientificQuantityWire {
                version: 1,
                variable_id: "terminal".into(),
                variable_name: "terminal".into(),
                role: "outcome".into(),
                units: "score".into(),
                population_id: "population".into(),
                regime_id: "observed".into(),
                horizon: 0,
                functional_id: "mean".into(),
                conditioning: vec![],
                transform_id: "identity".into(),
            },
            threshold: 0.25,
            verdict: ThresholdVerdict::Above,
            endpoints: [0.5, 0.75],
            reading: BridgeIntervalReading::SamplingConfidence,
            level: 0.95,
            attained_mass: None,
            interval_coverage_record: Some("record-v1".into()),
            interval_calibration_sha: Some("revision-1".into()),
        };
        let first = BridgeImageResult::IntervalThreshold(raw.clone()).readout().unwrap();
        let mut updated = raw;
        updated.interval_coverage_record = Some("record-v2".into());
        updated.interval_calibration_sha = Some("revision-2".into());
        assert_eq!(first, BridgeImageResult::IntervalThreshold(updated.clone()).readout().unwrap());
        updated.reading = BridgeIntervalReading::SamplingNominal;
        updated.interval_coverage_record = None;
        updated.interval_calibration_sha = None;
        assert_eq!(first, BridgeImageResult::IntervalThreshold(updated.clone()).readout().unwrap());
        updated.endpoints[0] = 0.52;
        assert_ne!(first, BridgeImageResult::IntervalThreshold(updated).readout().unwrap());
    }
}
