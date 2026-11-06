//! The composition boundary: what a decision input carries and what may be
//! combined with what.
//!
//! A [`DecisionInput`] wraps one source (a mean grid, an aligned joint law or a
//! scalar claim) with its [`InputProvenance`] (who produced it, under which
//! trust, with which receipt, which calibration and capabilities) and a
//! per-coordinate [`SupportMap`]. Trust is never upgraded from artifact
//! metadata: an artifact labelled `native` or `exact` is treated as unverified
//! unless a matching native execution record or an exact-request verification
//! receipt is supplied, and a contract that requires native or verified inputs
//! refuses the label.
//!
//! [`evaluate_with_support`] decides support per action. An action that reads a
//! missing or unsupported coordinate, or that its source cannot answer, is
//! reported as `Unsupported` or `Unevaluated` with a reason while the supported
//! actions are still compared, unless the policy demands every action. When no
//! action is supported the result is a state, not an error.
//!
//! [`check_composition`] gates combining sources. Statistical pooling, Bayesian
//! borrowing, causal transport and evidence reuse are separate declared
//! operations; independent pooling or paired draws are refused when the inputs
//! share data, a prior or a fitted model, or when their dependence is unknown,
//! unless a covariance or joint-law route is declared and supported.
//! [`check_atom_combination`] never turns conflicting structural atoms into a
//! silent average.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod composition;

pub use composition::{
    AtomCombination, AtomPlan, CompositionOperation, CompositionReceipt, DependenceRoute,
    EvidenceRelation, PairRelation, RouteKind, check_atom_combination, check_composition,
    check_paired_draws,
};

use antecedent_contract::{
    ExternalCapability, ExternalRefusal, ExternalTrustState, ProviderTrust, ScientificQuantity,
};
use antecedent_core::{CanonicalWriter, SupportStatus};
use antecedent_io_core::IoError;
use antecedent_io_core::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionTrust,
};

use crate::signal::{RefusalBuilder, at_stage};

use crate::decision_contract::{
    DecisionAction, DecisionContract, DecisionContractError, DecisionCriterion, DecisionFunctional,
    SourceRepresentation,
};
use crate::decision_eval::{
    ActionOutcome, DecisionEvalError, DecisionResult, FunctionalValue, MeanSource, Verdict,
    evaluate_contract, evaluate_contract_on_means, evaluate_functional,
};

/// Who produced an input, as established by evidence and not by a label.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ProviderKind {
    /// Executed and supported natively, with a matching execution record.
    Native,
    /// Supplied by an external party, attested or unverified.
    ExternalAttested,
    /// An external object verified for its exact request.
    ExternalExactRequestVerified,
}

impl ProviderKind {
    const fn label(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::ExternalAttested => ProviderTrust::ExternallyAttested.as_wire(),
            Self::ExternalExactRequestVerified => ProviderTrust::ExactRequestVerified.as_wire(),
        }
    }
}

fn trust_label(trust: DistributionTrust) -> &'static str {
    match trust {
        DistributionTrust::NativeSupported => ProviderTrust::NativeSupported.as_wire(),
        DistributionTrust::ExternalAttested => ProviderTrust::ExternallyAttested.as_wire(),
        DistributionTrust::VerifiedExtension => ProviderTrust::ExactRequestVerified.as_wire(),
        DistributionTrust::Unverified => ProviderTrust::Unverified.as_wire(),
    }
}

/// Calibration standing of an input after trust resolution.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CalibrationStatus {
    /// Exact finite law from a native or verified provider.
    Exact,
    /// Descriptive point summaries only.
    PointOnly,
    /// A separately cited calibration record exists.
    Measured,
    /// Inferential calibration is absent.
    NotMeasured,
    /// The artifact said `exact` but no native record or verification receipt
    /// backs it, so the claim is not used.
    ExactClaimUnverified,
}

/// What a consumer requires of an input's provider.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum TrustRequirement {
    /// Any provider; the resolved trust is still recorded.
    Unrestricted,
    /// A native execution record is required.
    Native,
    /// A native record or an exact-request verification receipt is required.
    ExactRequestVerified,
}

/// A record that Antecedent itself executed the route that produced an input.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeExecutionRecord {
    /// Stable execution identity.
    pub execution_id: String,
    /// Provider object the execution produced.
    pub provider_id: String,
    /// Data snapshot the execution ran on.
    pub snapshot_id: String,
}

/// Evidence offered for an input's trust.
#[derive(Clone, Debug, PartialEq)]
pub enum TrustEvidence {
    /// Nothing beyond the artifact's own labels, which prove nothing.
    None,
    /// A native execution record.
    NativeExecution(NativeExecutionRecord),
    /// An external attestation or exact-request verification receipt.
    External(ExternalTrustState),
}

/// Reference to the receipt that backs an input's trust.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReceiptRef {
    /// Native execution record.
    NativeExecution {
        /// Execution identity.
        execution_id: String,
    },
    /// External attestation.
    Attestation {
        /// Party that asserted.
        attestor: String,
    },
    /// Exact-request verification receipt.
    Verification {
        /// Verified provider object.
        object_id: String,
        /// Exact request fingerprint.
        request_id: String,
    },
}

/// Provenance carried by every decision input.
#[derive(Clone, Debug, PartialEq)]
pub struct InputProvenance {
    /// Provider kind established by evidence.
    pub provider_kind: ProviderKind,
    /// Trust label after resolution; never above what the evidence supports.
    pub trust: DistributionTrust,
    /// Attestation or verification receipt reference, when one backs the trust.
    pub receipt: Option<ReceiptRef>,
    /// Calibration standing.
    pub calibration: CalibrationStatus,
    /// Operations the source's representation can answer.
    pub capabilities: Vec<ExternalCapability>,
    /// Provider object identity.
    pub provider_id: String,
    /// Data snapshot identity.
    pub snapshot_id: String,
    /// BLAKE3 digest of the source's content, identities, provider kind and trust.
    pub lineage_digest: String,
}

/// Support status per source coordinate.
#[derive(Clone, Debug, PartialEq)]
pub struct SupportMap {
    entries: Vec<(ScientificQuantity, SupportStatus)>,
}

impl SupportMap {
    /// Pair each coordinate with its status.
    ///
    /// A later coordinate that matches an earlier one is dropped. The first
    /// coordinate and its status are kept.
    ///
    /// # Errors
    /// The two slices must have the same length.
    pub fn new(
        coordinates: &[ScientificQuantity],
        statuses: &[SupportStatus],
    ) -> Result<Self, BoundaryError> {
        if coordinates.len() != statuses.len() {
            return Err(BoundaryError::InvalidInput("one support status per coordinate"));
        }
        let entries = antecedent_contract::quantity::deduped_coordinate_positions(coordinates)
            .into_iter()
            .map(|index| (coordinates[index].clone(), statuses[index]))
            .collect();
        Ok(Self { entries })
    }

    /// Status of `wanted`, or `None` when the source has no such coordinate.
    #[must_use]
    pub fn status_of(&self, wanted: &ScientificQuantity) -> Option<SupportStatus> {
        self.entries
            .iter()
            .find(|(candidate, _)| candidate.require_same_coordinate(wanted).is_ok())
            .map(|(_, status)| *status)
    }

    /// Every coordinate with its status, in source order.
    #[must_use]
    pub fn entries(&self) -> &[(ScientificQuantity, SupportStatus)] {
        &self.entries
    }
}

/// One scalar claim: a single value for a single coordinate.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarClaim {
    /// Coordinate the value is claimed for.
    pub coordinate: ScientificQuantity,
    /// Claimed value.
    pub value: f64,
    /// Provider object that supplied it.
    pub provider_id: String,
    /// Data snapshot or exact-request identity.
    pub snapshot_id: String,
    /// Causal-contract identity the claim was bound to.
    pub causal_contract_id: String,
}

impl ScalarClaim {
    fn to_mean_source(&self) -> MeanSource {
        MeanSource {
            coordinates: vec![self.coordinate.clone()],
            means: vec![self.value],
            provider_id: self.provider_id.clone(),
            snapshot_id: self.snapshot_id.clone(),
            causal_contract_id: self.causal_contract_id.clone(),
            rng_id: "none:scalar_claim".to_owned(),
            estimate: crate::decision_eval::MeanEstimate::Frequentist,
        }
    }
}

/// What an input supplies. A mean, a scalar and an aligned joint law are
/// different objects; none stands in for another.
#[derive(Clone, Debug, PartialEq)]
// The joint law is the large variant; boxing it would change the public pattern
// `InputSource::JointLaw(artifact)` used by callers and tests.
#[allow(clippy::large_enum_variant, reason = "built once per call and moved, never stored in bulk")]
pub enum InputSource {
    /// One mean per coordinate.
    Mean(MeanSource),
    /// Aligned joint draws.
    JointLaw(DistributionArtifact),
    /// One scalar claim.
    Scalar(ScalarClaim),
}

/// A decision input: a source, its provenance and its per-coordinate support.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionInput {
    id: String,
    source: InputSource,
    provenance: InputProvenance,
    support: SupportMap,
}

struct Resolved {
    kind: ProviderKind,
    trust: DistributionTrust,
    receipt: Option<ReceiptRef>,
}

fn resolve_trust(evidence: &TrustEvidence, provider_id: &str, snapshot_id: &str) -> Resolved {
    let unproven = Resolved {
        kind: ProviderKind::ExternalAttested,
        trust: DistributionTrust::Unverified,
        receipt: None,
    };
    match evidence {
        TrustEvidence::NativeExecution(record)
            if record.provider_id == provider_id
                && record.snapshot_id == snapshot_id
                && !record.execution_id.trim().is_empty() =>
        {
            Resolved {
                kind: ProviderKind::Native,
                trust: DistributionTrust::NativeSupported,
                receipt: Some(ReceiptRef::NativeExecution {
                    execution_id: record.execution_id.clone(),
                }),
            }
        }
        TrustEvidence::External(ExternalTrustState::ExternallyAttested { attestor }) => Resolved {
            kind: ProviderKind::ExternalAttested,
            trust: DistributionTrust::ExternalAttested,
            receipt: Some(ReceiptRef::Attestation { attestor: attestor.clone() }),
        },
        TrustEvidence::External(ExternalTrustState::ExactRequestVerified(receipt)) => {
            // A receipt may verify several exact requests of one provider; the
            // input is backed only by one that names this provider and snapshot.
            let verified =
                receipt.requests().iter().map(antecedent_contract::VerifiedRequest::identity).find(
                    |object| object.provider_id == provider_id && object.snapshot_id == snapshot_id,
                );
            match verified {
                Some(object) => Resolved {
                    kind: ProviderKind::ExternalExactRequestVerified,
                    trust: DistributionTrust::VerifiedExtension,
                    receipt: Some(ReceiptRef::Verification {
                        object_id: object.object_id.clone(),
                        request_id: object.request_id.clone(),
                    }),
                },
                None => unproven,
            }
        }
        // A missing record, a record for another provider or snapshot, and an
        // external object claiming native trust all prove nothing.
        _ => unproven,
    }
}

fn gate(
    requirement: TrustRequirement,
    resolved: &Resolved,
    input: &str,
    claims_native: bool,
) -> Result<(), BoundaryError> {
    let input = input.to_owned();
    match requirement {
        TrustRequirement::Unrestricted => Ok(()),
        TrustRequirement::Native => {
            if resolved.kind == ProviderKind::Native {
                Ok(())
            } else if claims_native {
                Err(BoundaryError::MetadataOnlyNativeClaim { input })
            } else {
                Err(BoundaryError::NativeRequired { input })
            }
        }
        TrustRequirement::ExactRequestVerified => {
            if resolved.kind == ProviderKind::ExternalAttested {
                Err(BoundaryError::VerificationReceiptMissing { input })
            } else {
                Ok(())
            }
        }
    }
}

const fn calibration_status(
    claimed: DistributionCalibration,
    kind: ProviderKind,
) -> CalibrationStatus {
    match claimed {
        DistributionCalibration::Exact => {
            if matches!(kind, ProviderKind::ExternalAttested) {
                CalibrationStatus::ExactClaimUnverified
            } else {
                CalibrationStatus::Exact
            }
        }
        DistributionCalibration::PointOnly => CalibrationStatus::PointOnly,
        DistributionCalibration::Measured => CalibrationStatus::Measured,
        DistributionCalibration::NotMeasured => CalibrationStatus::NotMeasured,
    }
}

fn lineage_digest(
    source: &InputSource,
    coordinates: &[(ScientificQuantity, SupportStatus)],
    kind: ProviderKind,
    trust: DistributionTrust,
) -> String {
    let mut hasher = CanonicalWriter::new("antecedent.composition_boundary.input.v1");
    match source {
        InputSource::Mean(mean) => {
            hasher.text("mean");
            for id in [&mean.provider_id, &mean.snapshot_id, &mean.causal_contract_id] {
                hasher.text(id);
            }
            hasher.reals(&mean.means);
        }
        InputSource::JointLaw(artifact) => {
            hasher.text("joint_law");
            let identity = &artifact.metadata().identity;
            for id in [
                &identity.source_id,
                &identity.provider_id,
                &identity.rng_id,
                &identity.snapshot_id,
                &identity.causal_contract_id,
            ] {
                hasher.text(id);
            }
            hasher.reals(artifact.draws());
        }
        InputSource::Scalar(claim) => {
            hasher.text("scalar");
            for id in [&claim.provider_id, &claim.snapshot_id, &claim.causal_contract_id] {
                hasher.text(id);
            }
            hasher.reals(&[claim.value]);
        }
    }
    for (quantity, status) in coordinates {
        quantity.write_canonical(&mut hasher);
        hasher.text(status.as_str());
    }
    hasher.text(kind.label());
    hasher.text(trust_label(trust));
    hasher.finish_hex()
}

fn check_id(id: &str) -> Result<(), BoundaryError> {
    if id.trim().is_empty() {
        return Err(BoundaryError::InvalidInput("input id"));
    }
    Ok(())
}

impl DecisionInput {
    fn assemble(
        id: &str,
        source: InputSource,
        resolved: Resolved,
        calibration: CalibrationStatus,
        capabilities: Vec<ExternalCapability>,
        support: SupportMap,
    ) -> Self {
        let (provider_id, snapshot_id) = match &source {
            InputSource::Mean(mean) => (mean.provider_id.clone(), mean.snapshot_id.clone()),
            InputSource::JointLaw(artifact) => {
                let identity = &artifact.metadata().identity;
                (identity.provider_id.clone(), identity.snapshot_id.clone())
            }
            InputSource::Scalar(claim) => (claim.provider_id.clone(), claim.snapshot_id.clone()),
        };
        let lineage_digest =
            lineage_digest(&source, support.entries(), resolved.kind, resolved.trust);
        Self {
            id: id.to_owned(),
            source,
            provenance: InputProvenance {
                provider_kind: resolved.kind,
                trust: resolved.trust,
                receipt: resolved.receipt,
                calibration,
                capabilities,
                provider_id,
                snapshot_id,
                lineage_digest,
            },
            support,
        }
    }

    /// Build an input from an aligned joint distribution artifact.
    ///
    /// The artifact's own `trust` and `calibration` labels are claims, not
    /// evidence. The provider kind comes only from `evidence`: a native
    /// execution record for the same provider and snapshot, an external
    /// attestation, or an exact-request verification receipt for the same
    /// provider and snapshot. An artifact labelled native or exact without that
    /// evidence is stored as `Unverified` with a `not_measured` calibration and its
    /// `exact` claim is not used. Coordinate support comes from the artifact's
    /// mask; an absent mask means every coordinate is supported.
    ///
    /// # Errors
    /// A blank id, an invalid coordinate, `MetadataOnlyNativeClaim` or
    /// `NativeRequired` when `requirement` is `Native` and no native record
    /// matches, and `VerificationReceiptMissing` when `requirement` is
    /// `ExactRequestVerified` and neither a native record nor a matching
    /// verification receipt is present.
    pub fn from_distribution_artifact(
        id: &str,
        artifact: &DistributionArtifact,
        evidence: &TrustEvidence,
        requirement: TrustRequirement,
    ) -> Result<Self, BoundaryError> {
        check_id(id)?;
        let meta = artifact.metadata();
        let resolved =
            resolve_trust(evidence, &meta.identity.provider_id, &meta.identity.snapshot_id);
        gate(requirement, &resolved, id, meta.trust == DistributionTrust::NativeSupported)?;
        let calibration = calibration_status(meta.calibration, resolved.kind);
        let mut coordinates = Vec::with_capacity(artifact.quantities().len());
        for wire in artifact.quantities() {
            coordinates.push(
                ScientificQuantity::try_from(wire.clone())
                    .map_err(|_| BoundaryError::InvalidInput("artifact quantity coordinate"))?,
            );
        }
        let mask = meta.supported.as_deref();
        let statuses: Vec<SupportStatus> = (0..coordinates.len())
            .map(|index| {
                if mask.and_then(|m| m.get(index)).copied().unwrap_or(true) {
                    SupportStatus::Supported
                } else {
                    SupportStatus::OutsideEmpiricalSupport
                }
            })
            .collect();
        let support = SupportMap::new(&coordinates, &statuses)?;
        let downgraded = calibration == CalibrationStatus::ExactClaimUnverified;
        let effective = if downgraded || meta.trust != resolved.trust {
            let mut metadata = meta.clone();
            if downgraded {
                metadata.calibration = DistributionCalibration::NotMeasured;
            }
            metadata.trust = resolved.trust;
            DistributionArtifact::new(metadata, artifact.draws().to_vec())
                .map_err(BoundaryError::Artifact)?
        } else {
            artifact.clone()
        };
        let mut capabilities = vec![
            ExternalCapability::Sample,
            ExternalCapability::Mean,
            ExternalCapability::Covariance,
        ];
        let meaning = antecedent_contract::DistributionMeaning::from(artifact.semantic());
        if meaning.answers_interventional_outcome_threshold() {
            capabilities.push(ExternalCapability::Intervention);
        }
        Ok(Self::assemble(
            id,
            InputSource::JointLaw(effective),
            resolved,
            calibration,
            capabilities,
            support,
        ))
    }

    /// Build an input from a mean source, with one support status per coordinate.
    ///
    /// # Errors
    /// A blank id, mismatched status count, or a failed trust requirement.
    pub fn from_mean_source(
        id: &str,
        source: MeanSource,
        statuses: &[SupportStatus],
        evidence: &TrustEvidence,
        requirement: TrustRequirement,
    ) -> Result<Self, BoundaryError> {
        check_id(id)?;
        let support = SupportMap::new(&source.coordinates, statuses)?;
        let resolved = resolve_trust(evidence, &source.provider_id, &source.snapshot_id);
        gate(requirement, &resolved, id, false)?;
        Ok(Self::assemble(
            id,
            InputSource::Mean(source),
            resolved,
            CalibrationStatus::PointOnly,
            vec![ExternalCapability::Mean],
            support,
        ))
    }

    /// Build an input from one scalar claim.
    ///
    /// # Errors
    /// A blank id, or a failed trust requirement.
    pub fn from_scalar_claim(
        id: &str,
        claim: ScalarClaim,
        status: SupportStatus,
        evidence: &TrustEvidence,
        requirement: TrustRequirement,
    ) -> Result<Self, BoundaryError> {
        check_id(id)?;
        let support = SupportMap::new(std::slice::from_ref(&claim.coordinate), &[status])?;
        let resolved = resolve_trust(evidence, &claim.provider_id, &claim.snapshot_id);
        gate(requirement, &resolved, id, false)?;
        Ok(Self::assemble(
            id,
            InputSource::Scalar(claim),
            resolved,
            CalibrationStatus::PointOnly,
            vec![ExternalCapability::Mean],
            support,
        ))
    }

    /// Input identity within one composition.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// The source.
    #[must_use]
    pub const fn source(&self) -> &InputSource {
        &self.source
    }

    /// Provenance.
    #[must_use]
    pub const fn provenance(&self) -> &InputProvenance {
        &self.provenance
    }

    /// Weaken every coordinate's support to at most `floor`: a coordinate
    /// keeps its own status when that is already weaker. A route that carries
    /// a source across populations uses this so the label of the carrying
    /// (its receipt) governs a law's coordinates, not the artifact's mask
    /// alone. Never strengthens a status.
    #[must_use]
    pub fn with_support_floor(mut self, floor: SupportStatus) -> Self {
        for (_, status) in &mut self.support.entries {
            if status.severity() < floor.severity() {
                *status = floor;
            }
        }
        self
    }

    /// Per-coordinate support.
    #[must_use]
    pub const fn support(&self) -> &SupportMap {
        &self.support
    }
}

/// Why a boundary operation refused.
#[derive(Clone, Debug, PartialEq)]
pub enum BoundaryError {
    /// A blank identity, mismatched lengths or an unusable declaration.
    InvalidInput(&'static str),
    /// Rebuilding an artifact with resolved trust failed.
    Artifact(IoError),
    /// The artifact labels itself native, but no native execution record matches.
    MetadataOnlyNativeClaim {
        /// Input whose label is unbacked.
        input: String,
    },
    /// A native provider is required and none is established.
    NativeRequired {
        /// Input that is not native.
        input: String,
    },
    /// An exact-request verification receipt is required and none matches.
    VerificationReceiptMissing {
        /// Input without a matching receipt.
        input: String,
    },
    /// The underlying evaluation refused.
    Eval(DecisionEvalError),
    /// The policy demands every action and these could not be evaluated.
    UnsupportedActions {
        /// Actions that were unsupported or unevaluated.
        actions: Vec<String>,
    },
    /// A mean or scalar was asked for something that needs a distribution.
    MeanIsNotADistribution {
        /// Action whose functional was requested.
        action: String,
    },
    /// Actions answered by different sources cannot be compared state by state.
    PairedDrawsAcrossSources,
    /// No composition operation was declared.
    OperationNotDeclared,
    /// Two inputs share evidence and the operation would treat them as independent.
    SharedEvidenceNotIndependent {
        /// First input.
        left: String,
        /// Second input.
        right: String,
    },
    /// Dependence between two inputs is unknown, and unknown is not independent.
    UnknownDependence {
        /// First input.
        left: String,
        /// Second input.
        right: String,
    },
    /// A declared covariance or joint-law route is not backed by an aligned law.
    DependenceRouteNotSupported {
        /// First input.
        left: String,
        /// Second input.
        right: String,
    },
    /// Conflicting structural atoms were to be averaged without declared probabilities.
    ConflictingAtomsNotAveraged,
    /// Atom probabilities are not finite, are negative or sum to more than one.
    InvalidAtomProbabilities,
    /// A relation names an input that was not supplied.
    UnknownInput(String),
}

fn refusal(code: &'static str, detail: &str) -> RefusalBuilder {
    RefusalBuilder::new(code, "compose", detail)
}

impl BoundaryError {
    /// Structured refusal with a registered code and a namespaced detail.
    #[must_use]
    pub fn to_refusal(&self) -> ExternalRefusal {
        let invalid = antecedent_core::reason_code!("invalid_argument");
        let unsatisfied = antecedent_core::reason_code!("decision_contract_unsatisfied");
        let unverifiable = antecedent_core::reason_code!("attested_not_reverifiable");
        let shared = antecedent_core::reason_code!("scenario_aggregate_not_supported");
        let dependence = antecedent_core::reason_code!("sampling_dependence_unknown");
        let pair = |left: &String, right: &String| format!("{left}~{right}");
        match self {
            Self::InvalidInput(why) => {
                refusal(invalid, antecedent_core::reason_details::composition_boundary::INVALID_INPUT).expected(*why).build()
            }
            Self::Artifact(error) => {
                refusal(invalid, antecedent_core::reason_details::composition_boundary::ARTIFACT_REBUILD_FAILED)
                    .supplied(error.to_string())
                    .build()
            }
            Self::MetadataOnlyNativeClaim { input } => {
                refusal(unverifiable, antecedent_core::reason_details::composition_boundary::METADATA_ONLY_NATIVE_CLAIM)
                    .offending(input.clone())
                    .remedy("supply a native execution record for this provider and snapshot")
                    .build()
            }
            Self::NativeRequired { input } => {
                refusal(unverifiable, antecedent_core::reason_details::composition_boundary::NATIVE_REQUIRED)
                    .offending(input.clone())
                    .remedy("supply a native execution record, or relax the requirement")
                    .build()
            }
            Self::VerificationReceiptMissing { input } => refusal(
                antecedent_core::reason_code!("external_verification_failed"),
                antecedent_core::reason_details::composition_boundary::VERIFICATION_RECEIPT_MISSING,
            )
            .offending(input.clone())
            .remedy("supply an exact-request verification receipt for this object")
            .build(),
            Self::Eval(error) => at_stage(error.to_refusal(), "compose"),
            Self::UnsupportedActions { actions } => {
                refusal(unsatisfied, antecedent_core::reason_details::composition_boundary::UNSUPPORTED_ACTION_NOT_COMPARABLE)
                    .offending(actions.join(","))
                    .remedy("compare supported actions, or supply the missing coordinates")
                    .build()
            }
            Self::MeanIsNotADistribution { action } => {
                refusal(unsatisfied, antecedent_core::reason_details::composition_boundary::MEAN_IS_NOT_A_DISTRIBUTION)
                    .offending(action.clone())
                    .remedy("supply an aligned joint law for this functional")
                    .build()
            }
            Self::PairedDrawsAcrossSources => refusal(
                antecedent_core::reason_code!("joint_law_required"),
                antecedent_core::reason_details::composition_boundary::PAIRED_DRAWS_ACROSS_SOURCES,
            )
            .remedy("supply one aligned joint law covering every compared action")
            .build(),
            Self::OperationNotDeclared => {
                refusal(invalid, antecedent_core::reason_details::composition_boundary::OPERATION_NOT_DECLARED)
                    .remedy("declare pooling, borrowing, transport or evidence reuse")
                    .build()
            }
            Self::SharedEvidenceNotIndependent { left, right } => {
                refusal(shared, antecedent_core::reason_details::composition_boundary::SHARED_EVIDENCE_NOT_INDEPENDENT)
                    .offending(pair(left, right))
                    .remedy("declare a covariance or joint-law route, or use another operation")
                    .build()
            }
            Self::UnknownDependence { left, right } => {
                refusal(dependence, antecedent_core::reason_details::composition_boundary::UNKNOWN_DEPENDENCE_IS_NOT_INDEPENDENCE)
                    .offending(pair(left, right))
                    .remedy("declare how the two inputs depend on each other")
                    .build()
            }
            Self::DependenceRouteNotSupported { left, right } => {
                refusal(dependence, antecedent_core::reason_details::composition_boundary::DEPENDENCE_ROUTE_NOT_SUPPORTED)
                    .offending(pair(left, right))
                    .remedy("name an aligned joint-law input that is one of the pair")
                    .build()
            }
            Self::ConflictingAtomsNotAveraged => {
                refusal(shared, antecedent_core::reason_details::composition_boundary::CONFLICTING_ATOMS_NOT_AVERAGED)
                    .remedy("declare a probability for every atom, or report each atom")
                    .build()
            }
            Self::InvalidAtomProbabilities => {
                refusal(unsatisfied, antecedent_core::reason_details::composition_boundary::ATOM_PROBABILITIES_INVALID).build()
            }
            Self::UnknownInput(id) => {
                refusal(invalid, antecedent_core::reason_details::composition_boundary::UNKNOWN_INPUT).offending(id.clone()).build()
            }
        }
    }

    /// Registered runtime reason code.
    #[must_use]
    pub fn reason_code(&self) -> &'static str {
        self.to_refusal().code
    }
}

/// What to do with actions that cannot be evaluated.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UnsupportedActionPolicy {
    /// Refuse unless every action is evaluated.
    RequireAllActions,
    /// Report unevaluated actions and compare the rest.
    CompareSupported,
}

/// Policy for per-action support evaluation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SupportPolicy {
    /// What to do with unsupported or unevaluated actions.
    pub unsupported: UnsupportedActionPolicy,
    /// Weakest support status a read coordinate may have.
    pub weakest_support: SupportStatus,
}

impl SupportPolicy {
    /// Compare supported actions, requiring `Supported` coordinates.
    #[must_use]
    pub const fn compare_supported() -> Self {
        Self {
            unsupported: UnsupportedActionPolicy::CompareSupported,
            weakest_support: SupportStatus::Supported,
        }
    }

    /// Require every action to be evaluated, with `Supported` coordinates.
    #[must_use]
    pub const fn require_all() -> Self {
        Self {
            unsupported: UnsupportedActionPolicy::RequireAllActions,
            weakest_support: SupportStatus::Supported,
        }
    }
}

/// Why a coordinate does not support its action.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CoordinateIssue {
    /// The source has no coordinate with this exact quantity.
    NotInSource,
    /// The coordinate's support is below the policy's weakest allowed status.
    BelowSupport(SupportStatus),
}

impl CoordinateIssue {
    /// Namespaced detail literal.
    #[must_use]
    pub const fn detail(self) -> &'static str {
        match self {
            Self::NotInSource => {
                antecedent_core::reason_details::composition_boundary::COORDINATE_MISSING
            }
            Self::BelowSupport(_) => {
                antecedent_core::reason_details::composition_boundary::COORDINATE_UNSUPPORTED
            }
        }
    }
}

/// One coordinate that blocked an action on one input.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UnsupportedReason {
    /// Input that was examined.
    pub input_id: String,
    /// Position of the action's input.
    pub coordinate: usize,
    /// What was wrong with it.
    pub issue: CoordinateIssue,
}

/// Why a supported action could not be evaluated from its source.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UnevaluatedReason {
    /// The utility is not affine, so a mean or scalar cannot answer it.
    NonAffineUtilityNeedsJointLaw,
    /// An input is an outcome law, which a mean or scalar is not.
    OutcomeLawNeedsJointLaw,
}

impl UnevaluatedReason {
    /// Namespaced detail literal.
    #[must_use]
    pub const fn detail(self) -> &'static str {
        match self {
            Self::NonAffineUtilityNeedsJointLaw => {
                antecedent_core::reason_details::composition_boundary::NON_AFFINE_NEEDS_JOINT_LAW
            }
            Self::OutcomeLawNeedsJointLaw => {
                antecedent_core::reason_details::composition_boundary::OUTCOME_LAW_NEEDS_JOINT_LAW
            }
        }
    }
}

/// What became of one action.
#[derive(Clone, Debug, PartialEq)]
pub enum ActionStatus {
    /// Evaluated on its input.
    Evaluated,
    /// A coordinate is missing or too weakly supported on every input.
    Unsupported {
        /// Blocking coordinates, per input examined.
        reasons: Vec<UnsupportedReason>,
    },
    /// Coordinates are supported but no input's representation can answer it.
    Unevaluated {
        /// Why.
        reason: UnevaluatedReason,
    },
}

/// One action's disposition.
#[derive(Clone, Debug, PartialEq)]
pub struct ActionDisposition {
    /// Semantic action identity.
    pub id: String,
    /// What became of it.
    pub status: ActionStatus,
    /// Input that answered it, when evaluated.
    pub input_id: Option<String>,
}

/// What the support-aware comparison can claim.
#[derive(Clone, Debug, PartialEq)]
pub enum SupportedVerdict {
    /// At least two actions were evaluated and compared.
    Compared(Verdict),
    /// Exactly one action was evaluated and admissible; nothing was compared.
    OnlyOneEvaluated(String),
    /// No action could be evaluated; this is a state, not an error.
    NoSupportedAction,
}

/// Result of a support-aware evaluation.
#[derive(Clone, Debug, PartialEq)]
pub struct SupportedDecision {
    /// Identity of the evaluated contract.
    pub contract_identity: String,
    /// Disposition of every action, in declaration order.
    pub dispositions: Vec<ActionDisposition>,
    /// Outcomes of the evaluated actions, in declaration order.
    pub outcomes: Vec<ActionOutcome>,
    /// The verdict over evaluated actions.
    pub verdict: SupportedVerdict,
    /// Expected value of perfect information, present only when every evaluated
    /// action was compared state by state on one aligned source.
    pub evpi: Option<f64>,
    /// Each source group's full result, for sources that compared two or more actions.
    pub results: Vec<DecisionResult>,
    /// Assumptions the result stands on.
    pub assumptions: Vec<String>,
}

const fn support_rank(status: SupportStatus) -> u8 {
    match status {
        SupportStatus::Supported => 5,
        SupportStatus::WeakOverlap => 4,
        SupportStatus::Extrapolative => 3,
        SupportStatus::OutsideEmpiricalSupport => 2,
        SupportStatus::MissingEvidence => 1,
    }
}

enum Issue {
    Unsupported(Vec<UnsupportedReason>),
    Unevaluated(UnevaluatedReason),
}

fn coordinate_reasons(
    action: &DecisionAction,
    input: &DecisionInput,
    weakest: SupportStatus,
) -> Vec<UnsupportedReason> {
    let mut reasons = Vec::new();
    for (position, quantity) in action.inputs.iter().enumerate() {
        let issue = match input.support.status_of(quantity) {
            None => Some(CoordinateIssue::NotInSource),
            Some(status) if support_rank(status) < support_rank(weakest) => {
                Some(CoordinateIssue::BelowSupport(status))
            }
            Some(_) => None,
        };
        if let Some(issue) = issue {
            reasons.push(UnsupportedReason {
                input_id: input.id.clone(),
                coordinate: position,
                issue,
            });
        }
    }
    reasons
}

fn assess(
    action: &DecisionAction,
    input: &DecisionInput,
    weakest: SupportStatus,
) -> Result<(), Issue> {
    let reasons = coordinate_reasons(action, input, weakest);
    if !reasons.is_empty() {
        return Err(Issue::Unsupported(reasons));
    }
    if !matches!(input.source, InputSource::JointLaw(_)) {
        if action.inputs.iter().any(|q| q.functional_id == "outcome") {
            return Err(Issue::Unevaluated(UnevaluatedReason::OutcomeLawNeedsJointLaw));
        }
        if !action.utility.is_affine() {
            return Err(Issue::Unevaluated(UnevaluatedReason::NonAffineUtilityNeedsJointLaw));
        }
    }
    Ok(())
}

fn classify(issues: Vec<Issue>) -> ActionStatus {
    let unevaluated = issues.iter().find_map(|issue| match issue {
        Issue::Unevaluated(reason) => Some(*reason),
        Issue::Unsupported(_) => None,
    });
    if let Some(reason) = unevaluated {
        return ActionStatus::Unevaluated { reason };
    }
    let reasons = issues
        .into_iter()
        .flat_map(|issue| match issue {
            Issue::Unsupported(reasons) => reasons,
            Issue::Unevaluated(_) => Vec::new(),
        })
        .collect();
    ActionStatus::Unsupported { reasons }
}

/// Assign each action to the first input that supports it and can answer it.
fn plan_actions(
    contract: &DecisionContract,
    inputs: &[DecisionInput],
    weakest: SupportStatus,
) -> (Vec<ActionDisposition>, Vec<Vec<usize>>) {
    let mut dispositions = Vec::with_capacity(contract.actions.len());
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); inputs.len()];
    for (index, action) in contract.actions.iter().enumerate() {
        let mut issues = Vec::new();
        let mut chosen = None;
        for (slot, input) in inputs.iter().enumerate() {
            match assess(action, input, weakest) {
                Ok(()) => {
                    chosen = Some(slot);
                    break;
                }
                Err(issue) => issues.push(issue),
            }
        }
        if let Some(slot) = chosen {
            groups[slot].push(index);
            dispositions.push(ActionDisposition {
                id: action.id.clone(),
                status: ActionStatus::Evaluated,
                input_id: Some(inputs[slot].id.clone()),
            });
        } else {
            dispositions.push(ActionDisposition {
                id: action.id.clone(),
                status: classify(issues),
                input_id: None,
            });
        }
    }
    (dispositions, groups)
}

/// The contract restricted to `group`, with a shadow copy when fewer than two
/// actions remain so the existing evaluator's two-action rule holds.
fn restrict(contract: &DecisionContract, group: &[usize]) -> (DecisionContract, bool) {
    let ids: Vec<&str> = group.iter().map(|i| contract.actions[*i].id.as_str()).collect();
    let mut actions: Vec<DecisionAction> =
        group.iter().map(|i| contract.actions[*i].clone()).collect();
    let shadowed = actions.len() < 2;
    if shadowed {
        let mut shadow = actions[0].clone();
        shadow.id = format!("{}#shadow", shadow.id);
        actions.push(shadow);
    }
    let constraints = contract
        .constraints
        .iter()
        .filter_map(|constraint| {
            if constraint.applies_to.is_empty() {
                return Some(constraint.clone());
            }
            let kept: Vec<String> = constraint
                .applies_to
                .iter()
                .filter(|action| ids.contains(&action.as_str()))
                .cloned()
                .collect();
            if kept.is_empty() {
                None
            } else {
                let mut narrowed = constraint.clone();
                narrowed.applies_to = kept;
                Some(narrowed)
            }
        })
        .collect();
    (DecisionContract { actions, constraints, ..contract.clone() }, shadowed)
}

fn run_source(
    contract: &DecisionContract,
    source: &InputSource,
) -> Result<DecisionResult, BoundaryError> {
    let result = match source {
        InputSource::JointLaw(artifact) => evaluate_contract(contract, artifact),
        InputSource::Mean(mean) => evaluate_contract_on_means(contract, mean),
        InputSource::Scalar(claim) => evaluate_contract_on_means(contract, &claim.to_mean_source()),
    };
    result.map_err(BoundaryError::Eval)
}

#[allow(clippy::type_complexity, reason = "one-off tuple returned from a single private helper")]
fn run_groups(
    contract: &DecisionContract,
    inputs: &[DecisionInput],
    groups: &[Vec<usize>],
) -> Result<(Vec<(usize, ActionOutcome)>, Vec<DecisionResult>), BoundaryError> {
    let mut outcomes = Vec::new();
    let mut results = Vec::new();
    for (slot, group) in groups.iter().enumerate().filter(|(_, g)| !g.is_empty()) {
        let (restricted, shadowed) = restrict(contract, group);
        let result = run_source(&restricted, &inputs[slot].source)?;
        for index in group {
            let id = &contract.actions[*index].id;
            if let Some(outcome) = result.actions.iter().find(|o| &o.id == id) {
                outcomes.push((*index, outcome.clone()));
            }
        }
        if !shadowed {
            results.push(result);
        }
    }
    outcomes.sort_by_key(|(index, _)| *index);
    Ok((outcomes, results))
}

fn derive_verdict(criterion: DecisionCriterion, outcomes: &[ActionOutcome]) -> SupportedVerdict {
    if outcomes.is_empty() {
        return SupportedVerdict::NoSupportedAction;
    }
    let admissible: Vec<&ActionOutcome> = outcomes.iter().filter(|o| o.admissible).collect();
    if admissible.is_empty() {
        return SupportedVerdict::Compared(Verdict::NoAdmissibleAction);
    }
    if outcomes.len() == 1 {
        return SupportedVerdict::OnlyOneEvaluated(admissible[0].id.clone());
    }
    let smaller_is_better = matches!(
        criterion,
        DecisionCriterion::PosteriorExpectedLoss
            | DecisionCriterion::Regret
            | DecisionCriterion::ExpectedRegret
    );
    let key = |o: &ActionOutcome| if smaller_is_better { -o.value } else { o.value };
    let mut best = admissible[0];
    for candidate in admissible[1..].iter().copied() {
        if key(candidate).total_cmp(&key(best)) == std::cmp::Ordering::Greater {
            best = candidate;
        }
    }
    let tied: Vec<String> = admissible
        .iter()
        .filter(|o| o.value.to_bits() == best.value.to_bits())
        .map(|o| o.id.clone())
        .collect();
    if tied.len() == 1 {
        SupportedVerdict::Compared(Verdict::UniquelyOptimal(best.id.clone()))
    } else {
        SupportedVerdict::Compared(Verdict::Indistinguishable(tied))
    }
}

/// Evaluate `contract` over `inputs`, deciding support per action.
///
/// Each action is assigned to the first input whose support map covers every
/// coordinate it reads at or above the policy's weakest status and whose
/// representation can answer it (a mean or scalar answers only an affine utility
/// of non-outcome inputs). The actions assigned to one input are evaluated
/// together by the existing evaluator ([`evaluate_contract`] or
/// [`evaluate_contract_on_means`]), with hard constraints narrowed to those
/// actions. Actions no input can answer are reported with a reason and left out
/// of the comparison under [`UnsupportedActionPolicy::CompareSupported`]. When
/// actions come from several inputs, values are compared by the criterion,
/// regret and perfect-information values are withheld, and a criterion that
/// needs state-by-state alignment refuses.
///
/// # Errors
/// An invalid contract or input set; any unevaluated action under
/// `RequireAllActions`; a state-aligned criterion across several sources; and
/// any refusal of the underlying evaluator (a mean source with a hard
/// constraint, a non-joint draw alignment, a non-outcome meaning).
pub fn evaluate_with_support(
    contract: &DecisionContract,
    inputs: &[DecisionInput],
    policy: &SupportPolicy,
) -> Result<SupportedDecision, BoundaryError> {
    antecedent_core::execution_attempt::run_operation(
        antecedent_core::execution_attempt::Operation::Decision,
        || {
            let contract_identity = contract
                .identity()
                .map_err(|error| BoundaryError::Eval(DecisionEvalError::Contract(error)))?;
            if inputs.is_empty() {
                return Err(BoundaryError::InvalidInput("at least one input"));
            }
            for (position, input) in inputs.iter().enumerate() {
                if inputs[..position].iter().any(|earlier| earlier.id == input.id) {
                    return Err(BoundaryError::InvalidInput("duplicate input id"));
                }
            }
            let (dispositions, groups) = plan_actions(contract, inputs, policy.weakest_support);
            let skipped: Vec<String> = dispositions
                .iter()
                .filter(|d| d.status != ActionStatus::Evaluated)
                .map(|d| d.id.clone())
                .collect();
            if !skipped.is_empty()
                && policy.unsupported == UnsupportedActionPolicy::RequireAllActions
            {
                return Err(BoundaryError::UnsupportedActions { actions: skipped });
            }
            let used = groups.iter().filter(|g| !g.is_empty()).count();
            if used > 1 && contract.criterion.needs_state_alignment() {
                return Err(BoundaryError::PairedDrawsAcrossSources);
            }
            let (indexed, results) = run_groups(contract, inputs, &groups)?;
            let mut outcomes: Vec<ActionOutcome> = indexed.into_iter().map(|(_, o)| o).collect();
            let evpi = match results.as_slice() {
                [only] if used == 1 && outcomes.len() == only.actions.len() => only.evpi,
                _ => None,
            };
            if used > 1 {
                for outcome in &mut outcomes {
                    outcome.expected_regret = None;
                    outcome.max_regret = None;
                }
            }
            let verdict = derive_verdict(contract.criterion, &outcomes);
            Ok(SupportedDecision {
                contract_identity,
                dispositions,
                outcomes,
                verdict,
                evpi,
                results,
                assumptions: vec![
                    "support is decided per action from each input's per-coordinate support map"
                        .into(),
                    "unsupported and unevaluated actions are reported and never scored".into(),
                    "actions answered by different sources are compared by criterion value only"
                        .into(),
                ],
            })
        },
    )
}

/// Compute `functional` of one action's utility from one input.
///
/// A joint law answers through [`evaluate_functional`]. A mean or scalar answers
/// only the expectation of an affine utility over non-outcome inputs; a
/// probability, quantile, variance, tail expectation, nonlinear utility or
/// outcome-law input is refused, never approximated from a mean.
///
/// # Errors
/// An unknown action, a coordinate below `weakest`, a functional a mean cannot
/// answer, and any refusal of the joint-law evaluator.
pub fn evaluate_functional_on_input(
    contract: &DecisionContract,
    action_id: &str,
    functional: DecisionFunctional,
    input: &DecisionInput,
    weakest: SupportStatus,
) -> Result<FunctionalValue, BoundaryError> {
    contract.validate().map_err(|error| BoundaryError::Eval(DecisionEvalError::Contract(error)))?;
    let action = contract.actions.iter().find(|a| a.id == action_id).ok_or_else(|| {
        BoundaryError::Eval(DecisionEvalError::Contract(DecisionContractError::UnknownAction(
            action_id.to_owned(),
        )))
    })?;
    if !coordinate_reasons(action, input, weakest).is_empty() {
        return Err(BoundaryError::UnsupportedActions { actions: vec![action.id.clone()] });
    }
    let mean_source = match &input.source {
        InputSource::JointLaw(artifact) => {
            return evaluate_functional(contract, action_id, functional, artifact)
                .map_err(BoundaryError::Eval);
        }
        InputSource::Mean(mean) => mean.clone(),
        InputSource::Scalar(claim) => claim.to_mean_source(),
    };
    let not_distribution = || BoundaryError::MeanIsNotADistribution { action: action.id.clone() };
    let mode = functional
        .requirement(&action.utility)
        .check(&[SourceRepresentation::Mean])
        .map_err(|error| match error {
            DecisionContractError::MissingSource { .. } => not_distribution(),
            other => BoundaryError::Eval(DecisionEvalError::Contract(other)),
        })?;
    if !action.utility.is_affine() {
        return Err(not_distribution());
    }
    let mut values = Vec::with_capacity(action.inputs.len());
    for (position, wanted) in action.inputs.iter().enumerate() {
        if wanted.functional_id == "outcome" {
            return Err(BoundaryError::Eval(DecisionEvalError::MeaningMismatch {
                action: action.id.clone(),
                input: position,
            }));
        }
        let column = mean_source
            .coordinates
            .iter()
            .position(|candidate| candidate.require_same_coordinate(wanted).is_ok())
            .filter(|&column| column < mean_source.means.len())
            .ok_or_else(|| {
                BoundaryError::Eval(DecisionEvalError::QuantityNotFound {
                    action: action.id.clone(),
                    input: position,
                })
            })?;
        values.push(mean_source.means[column]);
    }
    let value = action
        .utility
        .evaluate(&values)
        .map_err(|error| BoundaryError::Eval(DecisionEvalError::Contract(error)))?;
    if !value.is_finite() {
        return Err(BoundaryError::Eval(DecisionEvalError::NonFiniteUtility {
            action: action.id.clone(),
        }));
    }
    Ok(FunctionalValue { value, standard_error: None, source_mode: mode })
}
