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
//! unless a covariance or joint-law route is declared and licensed.
//! [`check_atom_combination`] never turns conflicting structural atoms into a
//! silent average.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    ExternalCapability, ExternalRefusal, ExternalTrustState, ScientificQuantity, SupportStatus,
};
use antecedent_io::IoError;
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionTrust, DrawAlignment,
};

use crate::decision_contract::{
    DecisionAction, DecisionContract, DecisionContractError, DecisionCriterion, DecisionFunctional,
    SourceRepresentation,
};
use crate::decision_eval::{
    ActionOutcome, DecisionEvalError, DecisionResult, FunctionalValue, MeanSource, Verdict,
    evaluate_contract, evaluate_contract_on_means, evaluate_functional,
};
use crate::decision_structural::StructuralAtom;

/// Who produced an input, as established by evidence and not by a label.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ProviderKind {
    /// Executed and licensed natively, with a matching execution record.
    Native,
    /// Supplied by an external party, attested or unverified.
    ExternalAttested,
    /// An external object verified for its exact request.
    ExternalExactRequestVerified,
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
    Unmeasured,
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
    /// # Errors
    /// The two slices must have the same length.
    pub fn new(
        coordinates: &[ScientificQuantity],
        statuses: &[SupportStatus],
    ) -> Result<Self, BoundaryError> {
        if coordinates.len() != statuses.len() {
            return Err(BoundaryError::InvalidInput("one support status per coordinate"));
        }
        Ok(Self { entries: coordinates.iter().cloned().zip(statuses.iter().copied()).collect() })
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
        }
    }
}

/// What an input supplies. A mean, a scalar and an aligned joint law are
/// different objects; none stands in for another.
#[derive(Clone, Debug, PartialEq)]
// The joint law is the large variant; boxing it would change the public pattern
// `InputSource::JointLaw(artifact)` used by callers and tests.
#[allow(clippy::large_enum_variant)]
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
                trust: DistributionTrust::NativeLicensed,
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
            let object = receipt.contract().identity();
            if object.provider_id == provider_id && object.snapshot_id == snapshot_id {
                Resolved {
                    kind: ProviderKind::ExternalExactRequestVerified,
                    trust: DistributionTrust::VerifiedExtension,
                    receipt: Some(ReceiptRef::Verification {
                        object_id: object.object_id.clone(),
                        request_id: object.request_id.clone(),
                    }),
                }
            } else {
                unproven
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
        DistributionCalibration::Unmeasured => CalibrationStatus::Unmeasured,
    }
}

fn absorb(hasher: &mut blake3::Hasher, part: &str) {
    hasher.update(&u64::try_from(part.len()).unwrap_or(u64::MAX).to_le_bytes());
    hasher.update(part.as_bytes());
}

fn absorb_quantity(hasher: &mut blake3::Hasher, quantity: &ScientificQuantity) {
    absorb(hasher, &quantity.variable_id);
    absorb(hasher, &quantity.population_id);
    absorb(hasher, &quantity.regime_id);
    absorb(hasher, &quantity.functional_id);
    absorb(hasher, &quantity.transform_id);
    absorb(hasher, &format!("{:?}", quantity.conditioning));
    hasher.update(&quantity.horizon.to_le_bytes());
}

fn absorb_values(hasher: &mut blake3::Hasher, values: &[f64]) {
    hasher.update(&u64::try_from(values.len()).unwrap_or(u64::MAX).to_le_bytes());
    for value in values {
        hasher.update(&value.to_le_bytes());
    }
}

fn lineage_digest(
    source: &InputSource,
    coordinates: &[(ScientificQuantity, SupportStatus)],
    kind: ProviderKind,
    trust: DistributionTrust,
) -> String {
    let mut hasher = blake3::Hasher::new();
    absorb(&mut hasher, "antecedent.composition_boundary.input.v1");
    match source {
        InputSource::Mean(mean) => {
            absorb(&mut hasher, "mean");
            for id in [&mean.provider_id, &mean.snapshot_id, &mean.causal_contract_id] {
                absorb(&mut hasher, id);
            }
            absorb_values(&mut hasher, &mean.means);
        }
        InputSource::JointLaw(artifact) => {
            absorb(&mut hasher, "joint_law");
            let identity = &artifact.metadata().identity;
            for id in [
                &identity.source_id,
                &identity.provider_id,
                &identity.rng_id,
                &identity.snapshot_id,
                &identity.causal_contract_id,
            ] {
                absorb(&mut hasher, id);
            }
            absorb_values(&mut hasher, artifact.draws());
        }
        InputSource::Scalar(claim) => {
            absorb(&mut hasher, "scalar");
            for id in [&claim.provider_id, &claim.snapshot_id, &claim.causal_contract_id] {
                absorb(&mut hasher, id);
            }
            absorb_values(&mut hasher, &[claim.value]);
        }
    }
    for (quantity, status) in coordinates {
        absorb_quantity(&mut hasher, quantity);
        absorb(&mut hasher, &format!("{status:?}"));
    }
    absorb(&mut hasher, &format!("{kind:?}"));
    absorb(&mut hasher, &format!("{trust:?}"));
    hasher.finalize().to_hex().to_string()
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
    /// evidence is stored as `Unverified` with an unmeasured calibration and its
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
        gate(requirement, &resolved, id, meta.trust == DistributionTrust::NativeLicensed)?;
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
                metadata.calibration = DistributionCalibration::Unmeasured;
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
        let meaning = antecedent_core::DistributionMeaning::from(artifact.semantic());
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
    DependenceRouteNotLicensed {
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

fn refusal(code: &'static str, detail: &str) -> ExternalRefusal {
    ExternalRefusal {
        code,
        stage: "compose",
        detail: detail.to_owned(),
        offending: None,
        expected: None,
        supplied: None,
        capability: None,
        remedy: None,
    }
}

impl BoundaryError {
    /// Structured refusal with a registered code and a namespaced detail.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn to_refusal(&self) -> ExternalRefusal {
        let invalid = antecedent_core::reason_code!("invalid_argument");
        let unsatisfied = antecedent_core::reason_code!("decision_contract_unsatisfied");
        let unverifiable = antecedent_core::reason_code!("attested_not_reverifiable");
        let shared = antecedent_core::reason_code!("scenario_aggregate_not_licensed");
        let dependence = antecedent_core::reason_code!("sampling_dependence_unknown");
        let pair = |left: &String, right: &String| Some(format!("{left}~{right}"));
        match self {
            Self::InvalidInput(why) => ExternalRefusal {
                expected: Some((*why).to_owned()),
                ..refusal(invalid, "composition_boundary.invalid_input")
            },
            Self::Artifact(error) => ExternalRefusal {
                supplied: Some(error.to_string()),
                ..refusal(invalid, "composition_boundary.artifact_rebuild_failed")
            },
            Self::MetadataOnlyNativeClaim { input } => ExternalRefusal {
                offending: Some(input.clone()),
                remedy: Some("supply a native execution record for this provider and snapshot"),
                ..refusal(unverifiable, "composition_boundary.metadata_only_native_claim")
            },
            Self::NativeRequired { input } => ExternalRefusal {
                offending: Some(input.clone()),
                remedy: Some("supply a native execution record, or relax the requirement"),
                ..refusal(unverifiable, "composition_boundary.native_required")
            },
            Self::VerificationReceiptMissing { input } => ExternalRefusal {
                offending: Some(input.clone()),
                remedy: Some("supply an exact-request verification receipt for this object"),
                ..refusal(
                    antecedent_core::reason_code!("external_verification_failed"),
                    "composition_boundary.verification_receipt_missing",
                )
            },
            Self::Eval(error) => {
                let mut value = error.to_refusal();
                value.stage = "compose";
                value
            }
            Self::UnsupportedActions { actions } => ExternalRefusal {
                offending: Some(actions.join(",")),
                remedy: Some("compare supported actions, or supply the missing coordinates"),
                ..refusal(unsatisfied, "composition_boundary.unsupported_action_not_comparable")
            },
            Self::MeanIsNotADistribution { action } => ExternalRefusal {
                offending: Some(action.clone()),
                remedy: Some("supply an aligned joint law for this functional"),
                ..refusal(unsatisfied, "composition_boundary.mean_is_not_a_distribution")
            },
            Self::PairedDrawsAcrossSources => ExternalRefusal {
                remedy: Some("supply one aligned joint law covering every compared action"),
                ..refusal(
                    antecedent_core::reason_code!("joint_law_required"),
                    "composition_boundary.paired_draws_across_sources",
                )
            },
            Self::OperationNotDeclared => ExternalRefusal {
                remedy: Some("declare pooling, borrowing, transport or evidence reuse"),
                ..refusal(invalid, "composition_boundary.operation_not_declared")
            },
            Self::SharedEvidenceNotIndependent { left, right } => ExternalRefusal {
                offending: pair(left, right),
                remedy: Some("declare a covariance or joint-law route, or use another operation"),
                ..refusal(shared, "composition_boundary.shared_evidence_not_independent")
            },
            Self::UnknownDependence { left, right } => ExternalRefusal {
                offending: pair(left, right),
                remedy: Some("declare how the two inputs depend on each other"),
                ..refusal(dependence, "composition_boundary.unknown_dependence_is_not_independence")
            },
            Self::DependenceRouteNotLicensed { left, right } => ExternalRefusal {
                offending: pair(left, right),
                remedy: Some("name an aligned joint-law input that is one of the pair"),
                ..refusal(dependence, "composition_boundary.dependence_route_not_licensed")
            },
            Self::ConflictingAtomsNotAveraged => ExternalRefusal {
                remedy: Some("declare a probability for every atom, or report each atom"),
                ..refusal(shared, "composition_boundary.conflicting_atoms_not_averaged")
            },
            Self::InvalidAtomProbabilities => {
                refusal(unsatisfied, "composition_boundary.atom_probabilities_invalid")
            }
            Self::UnknownInput(id) => ExternalRefusal {
                offending: Some(id.clone()),
                ..refusal(invalid, "composition_boundary.unknown_input")
            },
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
            Self::NotInSource => "composition_boundary.coordinate_missing",
            Self::BelowSupport(_) => "composition_boundary.coordinate_unsupported",
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
                "composition_boundary.non_affine_needs_joint_law"
            }
            Self::OutcomeLawNeedsJointLaw => "composition_boundary.outcome_law_needs_joint_law",
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
    if !skipped.is_empty() && policy.unsupported == UnsupportedActionPolicy::RequireAllActions {
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
            "support is decided per action from each input's per-coordinate support map".into(),
            "unsupported and unevaluated actions are reported and never scored".into(),
            "actions answered by different sources are compared by criterion value only".into(),
        ],
    })
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

/// How two inputs' evidence relates.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EvidenceRelation {
    /// Declared independent: no shared data, prior or fitted model.
    IndependentSources,
    /// Both use these data.
    SharedData {
        /// Shared data identities.
        ids: Vec<String>,
    },
    /// Both use this prior.
    SharedPrior {
        /// Shared prior identity.
        id: String,
    },
    /// Both use this fitted model.
    SharedFittedModel {
        /// Shared model identity.
        id: String,
    },
    /// Dependence was not assessed; this is never independence.
    UnknownDependence,
}

impl EvidenceRelation {
    fn shared_ids(&self) -> Vec<String> {
        match self {
            Self::SharedData { ids } => ids.clone(),
            Self::SharedPrior { id } | Self::SharedFittedModel { id } => vec![id.clone()],
            Self::IndependentSources | Self::UnknownDependence => Vec::new(),
        }
    }
}

/// Kind of licensed dependence route.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RouteKind {
    /// A covariance supplied by an aligned joint law.
    Covariance,
    /// An aligned joint law over both inputs' coordinates.
    JointLaw,
}

/// A declared route that accounts for dependence between two inputs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DependenceRoute {
    /// Route kind.
    pub kind: RouteKind,
    /// Route identity.
    pub id: String,
    /// Input that supplies the joint law; it must be one of the pair.
    pub source_input: String,
}

/// The declared relation between two inputs, with an optional dependence route.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PairRelation {
    /// First input identity.
    pub left: String,
    /// Second input identity.
    pub right: String,
    /// Declared evidence relation.
    pub relation: EvidenceRelation,
    /// Covariance or joint-law route, when one is declared.
    pub route: Option<DependenceRoute>,
}

/// The declared way inputs are combined. The operations are distinct and none
/// implies another.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CompositionOperation {
    /// Combine estimates as independent evidence, or pair their draws.
    StatisticalPooling,
    /// Use one source as a prior for another.
    BayesianBorrowing,
    /// Carry a result to a target population.
    CausalTransport,
    /// Reuse evidence in a second analysis, counted once.
    EvidenceReuse,
}

/// What a passed composition check records.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompositionReceipt {
    /// The declared operation.
    pub operation: CompositionOperation,
    /// Whether the combination assumes independent sources.
    pub independence_assumed: bool,
    /// Identities of the dependence routes used.
    pub routes: Vec<String>,
    /// Shared data, prior and model identities, sorted and distinct.
    pub shared_evidence: Vec<String>,
}

enum PairIssue {
    Shared,
    Unknown,
}

fn pair_policy(
    operation: CompositionOperation,
    relation: &EvidenceRelation,
) -> Result<(), PairIssue> {
    use CompositionOperation as Op;
    use EvidenceRelation as Rel;
    match (operation, relation) {
        (_, Rel::UnknownDependence) => Err(PairIssue::Unknown),
        (Op::StatisticalPooling, Rel::SharedData { .. } | Rel::SharedPrior { .. })
        | (Op::StatisticalPooling | Op::BayesianBorrowing, Rel::SharedFittedModel { .. })
        | (Op::BayesianBorrowing, Rel::SharedData { .. }) => Err(PairIssue::Shared),
        _ => Ok(()),
    }
}

fn validate_route(
    route: &DependenceRoute,
    left: &DecisionInput,
    right: &DecisionInput,
) -> Result<(), BoundaryError> {
    let unlicensed = || BoundaryError::DependenceRouteNotLicensed {
        left: left.id.clone(),
        right: right.id.clone(),
    };
    let source = [left, right].into_iter().find(|input| input.id == route.source_input);
    let Some(input) = source else {
        return Err(unlicensed());
    };
    let aligned = matches!(
        &input.source,
        InputSource::JointLaw(artifact)
            if artifact.metadata().identity.alignment == DrawAlignment::Joint
    );
    let capable = match route.kind {
        RouteKind::Covariance => {
            input.provenance.capabilities.contains(&ExternalCapability::Covariance)
        }
        RouteKind::JointLaw => input.provenance.capabilities.contains(&ExternalCapability::Sample),
    };
    if aligned && capable && !route.id.trim().is_empty() { Ok(()) } else { Err(unlicensed()) }
}

fn find_relation<'a>(
    relations: &'a [PairRelation],
    left: &str,
    right: &str,
) -> Option<&'a PairRelation> {
    relations
        .iter()
        .find(|r| (r.left == left && r.right == right) || (r.left == right && r.right == left))
}

fn validate_relations(
    inputs: &[DecisionInput],
    relations: &[PairRelation],
) -> Result<(), BoundaryError> {
    for (position, relation) in relations.iter().enumerate() {
        for id in [&relation.left, &relation.right] {
            if !inputs.iter().any(|input| &input.id == id) {
                return Err(BoundaryError::UnknownInput(id.clone()));
            }
        }
        if relation.left == relation.right {
            return Err(BoundaryError::InvalidInput("a relation names two different inputs"));
        }
        if find_relation(&relations[..position], &relation.left, &relation.right).is_some() {
            return Err(BoundaryError::InvalidInput("duplicate relation for one pair"));
        }
    }
    Ok(())
}

/// Check that `inputs` may be combined under the declared `operation`.
///
/// Every pair of inputs needs a declared [`PairRelation`]; a pair with none is
/// [`EvidenceRelation::UnknownDependence`]. A relation declared independent is
/// overridden to shared data when the two inputs carry the same snapshot
/// identity. Unknown dependence refuses every operation unless a route is
/// declared. Statistical pooling also refuses shared data, priors and fitted
/// models; Bayesian borrowing allows a shared prior but refuses shared data and
/// fitted models; transport and evidence reuse record shared evidence without
/// assuming independence. A declared covariance or joint-law route must name one
/// of the pair as an aligned joint law with the matching capability.
///
/// # Errors
/// No declared operation, fewer than two inputs, malformed relations, shared
/// evidence or unknown dependence without a licensed route.
pub fn check_composition(
    inputs: &[DecisionInput],
    relations: &[PairRelation],
    operation: Option<CompositionOperation>,
) -> Result<CompositionReceipt, BoundaryError> {
    let Some(operation) = operation else {
        return Err(BoundaryError::OperationNotDeclared);
    };
    if inputs.len() < 2 {
        return Err(BoundaryError::InvalidInput("composition needs at least two inputs"));
    }
    for (position, input) in inputs.iter().enumerate() {
        if inputs[..position].iter().any(|earlier| earlier.id == input.id) {
            return Err(BoundaryError::InvalidInput("duplicate input id"));
        }
    }
    validate_relations(inputs, relations)?;
    let mut routes = Vec::new();
    let mut shared = Vec::new();
    for (i, left) in inputs.iter().enumerate() {
        for right in &inputs[i + 1..] {
            let declared = find_relation(relations, &left.id, &right.id);
            let mut relation =
                declared.map_or(EvidenceRelation::UnknownDependence, |r| r.relation.clone());
            if relation == EvidenceRelation::IndependentSources
                && left.provenance.snapshot_id == right.provenance.snapshot_id
            {
                relation =
                    EvidenceRelation::SharedData { ids: vec![left.provenance.snapshot_id.clone()] };
            }
            let route = declared.and_then(|r| r.route.as_ref());
            if let Some(route) = route {
                validate_route(route, left, right)?;
            }
            if let Err(issue) = pair_policy(operation, &relation) {
                if let Some(route) = route {
                    routes.push(route.id.clone());
                } else {
                    let (l, r) = (left.id.clone(), right.id.clone());
                    return Err(match issue {
                        PairIssue::Shared => {
                            BoundaryError::SharedEvidenceNotIndependent { left: l, right: r }
                        }
                        PairIssue::Unknown => {
                            BoundaryError::UnknownDependence { left: l, right: r }
                        }
                    });
                }
            } else if let Some(route) = route {
                routes.push(route.id.clone());
            }
            shared.extend(relation.shared_ids());
        }
    }
    shared.sort();
    shared.dedup();
    routes.sort();
    routes.dedup();
    Ok(CompositionReceipt {
        operation,
        independence_assumed: operation == CompositionOperation::StatisticalPooling
            && routes.is_empty(),
        routes,
        shared_evidence: shared,
    })
}

/// Check that the draws of `inputs` may be paired row by row.
///
/// Pairing draws from separate sources treats them as independent, so this is
/// statistical pooling plus a requirement that every input is an aligned joint
/// law with the same draw count.
///
/// # Errors
/// Everything [`check_composition`] refuses under statistical pooling, and any
/// input that is not an aligned joint law, or a draw-count mismatch.
pub fn check_paired_draws(
    inputs: &[DecisionInput],
    relations: &[PairRelation],
) -> Result<CompositionReceipt, BoundaryError> {
    let mut draw_count = None;
    for input in inputs {
        let InputSource::JointLaw(artifact) = &input.source else {
            return Err(BoundaryError::Eval(DecisionEvalError::JointLawRequired {
                action: None,
                supplied_alignment: "not_a_joint_law",
            }));
        };
        if artifact.metadata().identity.alignment != DrawAlignment::Joint {
            return Err(BoundaryError::Eval(DecisionEvalError::JointLawRequired {
                action: None,
                supplied_alignment: "not_a_joint_law",
            }));
        }
        let n = artifact.metadata().shape[0];
        if draw_count.is_some_and(|previous| previous != n) {
            return Err(BoundaryError::InvalidInput("paired draw count mismatch"));
        }
        draw_count = Some(n);
    }
    check_composition(inputs, relations, Some(CompositionOperation::StatisticalPooling))
}

/// How structural atoms are to be combined.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AtomCombination {
    /// Report each atom's answer without combining.
    ReportEach,
    /// Rank by the worst case over atoms.
    WorstCase,
    /// Weight atoms by their declared probabilities.
    WeightedByDeclaredProbabilities,
}

/// A permitted way to combine structural atoms.
#[derive(Clone, Debug, PartialEq)]
pub enum AtomPlan {
    /// Report each atom.
    ReportEach,
    /// Worst case over atoms.
    WorstCase,
    /// Declared `(atom id, probability)` weights, not renormalized.
    Weighted(Vec<(String, f64)>),
}

/// Check that structural `atoms` may be combined as requested.
///
/// Two or more atoms are alternative structures; averaging them is licensed only
/// when every atom carries a declared probability. A completion count is not a
/// probability, and missing mass is retained, not renormalized.
///
/// # Errors
/// Blank or repeated atom identities, weighting without a probability on every
/// atom (`ConflictingAtomsNotAveraged`), and invalid probabilities.
pub fn check_atom_combination(
    atoms: &[StructuralAtom],
    combination: AtomCombination,
) -> Result<AtomPlan, BoundaryError> {
    if atoms.is_empty() {
        return Err(BoundaryError::InvalidInput("structural atoms"));
    }
    for (position, atom) in atoms.iter().enumerate() {
        if atom.id.trim().is_empty()
            || atoms[..position].iter().any(|earlier| earlier.id == atom.id)
        {
            return Err(BoundaryError::InvalidInput("structural atom identity"));
        }
    }
    match combination {
        AtomCombination::ReportEach => Ok(AtomPlan::ReportEach),
        AtomCombination::WorstCase => Ok(AtomPlan::WorstCase),
        AtomCombination::WeightedByDeclaredProbabilities => {
            if atoms.len() == 1 && atoms[0].probability.is_none() {
                return Ok(AtomPlan::Weighted(vec![(atoms[0].id.clone(), 1.0)]));
            }
            if atoms.iter().any(|atom| atom.probability.is_none()) {
                return Err(BoundaryError::ConflictingAtomsNotAveraged);
            }
            let weights: Vec<(String, f64)> = atoms
                .iter()
                .filter_map(|atom| atom.probability.map(|p| (atom.id.clone(), p)))
                .collect();
            let total: f64 = weights.iter().map(|(_, p)| *p).sum();
            if weights.iter().any(|(_, p)| !p.is_finite() || *p < 0.0) || total > 1.0 + 1e-9 {
                return Err(BoundaryError::InvalidAtomProbabilities);
            }
            Ok(AtomPlan::Weighted(weights))
        }
    }
}
