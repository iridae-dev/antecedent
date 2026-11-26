//! Portable identification of a patient-level binary principal-surrogate target.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use antecedent_contract::{QuantityCondition, QuantityRole, ScientificQuantity};
use antecedent_core::{CanonicalWriter, ExecutionContext, SemanticDigest};
use antecedent_identify::principal_surrogate::monotone_principal_surrogate_bounds;
pub use antecedent_identify::principal_surrogate::{
    PrincipalSurrogateBounds, PrincipalSurrogateError,
};
use serde::{Deserialize, Serialize};

use crate::{ArtifactContainer, IoError};

/// Portable artifact feature for principal-stratum identification and sample bounds.
pub const PATIENT_SURROGATE_ARTIFACT_KIND: &str = "patient_surrogate_identification_v1";
const METADATA: &str = "surrogate.identification";
const MAX_BYTES: usize = 16 * 1024 * 1024;
const MAX_PATIENTS: usize = 100_000;
const MAX_LABEL: usize = 128;

/// Premises and endpoint coordinates of a parallel patient-randomized trial.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PatientSurrogateDesign {
    /// Stable randomized study identity.
    pub trial_id: String,
    /// Target population common to both arm laws.
    pub population: String,
    /// Surrogate endpoint name, with one denoting improvement.
    pub surrogate: String,
    /// Terminal endpoint name, with one denoting benefit.
    pub outcome: String,
    /// Visit when the binary surrogate is measured.
    pub surrogate_visit: u32,
    /// Later visit when the terminal binary outcome is measured.
    pub outcome_visit: u32,
    /// Stable intervention label of the reference arm.
    pub control: String,
    /// Stable intervention label of the active arm.
    pub treatment: String,
    /// Actual observations equal the corresponding potential endpoints with no interference.
    pub consistency_and_no_interference: bool,
    /// Assignment is individually randomized, exchangeable and positive in both arms.
    pub randomized_exchangeability: bool,
    /// Both endpoints are jointly observed on each enrolled patient through outcome follow-up.
    pub complete_joint_follow_up: bool,
    /// No patient's surrogate decreases under assignment to the active arm.
    pub monotone_surrogate: bool,
}

/// Observed joint binary endpoints on one complete randomized patient.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservedSurrogatePatient {
    /// Stable patient identifier, unique across both arms.
    pub id: String,
    /// True for the declared active treatment.
    pub treated: bool,
    /// The patient's binary surrogate.
    pub surrogate: bool,
    /// The patient's later terminal endpoint.
    pub outcome: bool,
}

/// Joint endpoint evidence. A trial-level contrast cannot supply the joint cells.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PatientSurrogateEvidence {
    /// Exact population law of the joint individual-level endpoint pair in each randomized arm.
    PopulationJointLaw {
        /// Identity of the declared source of the joint population endpoint laws.
        evidence_id: String,
        /// Relative integer mass at (0,0), (0,1), (1,0), (1,1) in control.
        control_cells: [u32; 4],
        /// Relative integer mass in the same endpoint order under active treatment.
        treatment_cells: [u32; 4],
    },
    /// The observed randomized patient sample. Empirical bounds have no sampling coverage claim.
    CompletePatientSample {
        /// Original complete jointly observed records, one row per patient.
        patients: Vec<ObservedSurrogatePatient>,
    },
}

/// Requested patient-level identification operation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PatientSurrogateRequest {
    /// Population, assignment, endpoints and identification premises.
    pub design: PatientSurrogateDesign,
    /// The joint endpoint law or its empirical patient-sample counterpart.
    pub evidence: PatientSurrogateEvidence,
}

/// Reading appropriate to the evidence actually supplied.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PatientSurrogateReading {
    /// Sharp identified set under the exact joint population law and declared premises.
    PopulationIdentification,
    /// Plug-in calculation from empirical joint arm frequencies, without sampling limits.
    EmpiricalPlugin,
}

impl PatientSurrogateReading {
    /// Stable public and artifact label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PopulationIdentification => "population_identification",
            Self::EmpiricalPlugin => "empirical_plugin",
        }
    }
}

/// A typed identification, replay or decision refusal.
#[derive(Debug, thiserror::Error)]
pub enum PatientSurrogateError {
    /// A required trial coordinate, source identity or premise was missing.
    #[error("invalid patient-surrogate declaration: {0}")]
    Declaration(&'static str),
    /// The joint arm laws do not identify the declared principal stratum.
    #[error(transparent)]
    Identification(#[from] PrincipalSurrogateError),
    /// A threshold must give an exact probability between zero and one.
    #[error("threshold needs a positive denominator and a numerator no greater than it")]
    Threshold,
    /// An empirical plug-in bound has no standing as a population decision input.
    #[error(
        "the decision requires a population identified set; empirical plug-in bounds carry no sampling coverage"
    )]
    DecisionReading,
    /// The set contains several values and cannot supply a point.
    #[error("the verified population identified set does not identify one point")]
    PointNotIdentified,
    /// Recomputed declared evidence, program or result differs from retained expectations.
    #[error("patient-surrogate artifact identity or result differs from replay")]
    ReplayMismatch,
    /// The caller cancelled the evaluation.
    #[error("patient-surrogate identification was cancelled")]
    Cancelled,
    /// The caller's resource limit was too small for the patient source.
    #[error("patient-surrogate evidence exceeds the declared resource limit")]
    Resource,
    /// The artifact could not be read or encoded.
    #[error(transparent)]
    Artifact(#[from] IoError),
}

/// Terminal result of a comparison with a prespecified probability threshold.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SurrogateThresholdReading {
    /// Every compatible probability is at or above the threshold.
    AllAtOrAbove,
    /// Every compatible probability is strictly below the threshold.
    AllBelow,
    /// The exact interval contains values supporting both readings.
    Ambiguous,
}

/// A bound-aware threshold decision attached to the identified set it consumed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SurrogateThresholdDecision {
    /// Exact threshold numerator and denominator.
    pub threshold: [u64; 2],
    /// The interval's unanimous or ambiguous implication.
    pub reading: SurrogateThresholdReading,
    /// Identity of the checked population identified set.
    pub source: SemanticDigest,
    /// Identity of the bound threshold calculation.
    pub identity: SemanticDigest,
}

/// Checked principal-stratum bound. Only typed producers and artifact replay can construct it.
#[derive(Clone, Debug, PartialEq)]
pub struct PatientSurrogateResult {
    bounds: PrincipalSurrogateBounds,
    reading: PatientSurrogateReading,
    quantity: ScientificQuantity,
    source: SemanticDigest,
    identity: SemanticDigest,
}

impl PatientSurrogateResult {
    /// Exact lower and upper ratios in their common denominator.
    #[must_use]
    pub const fn bounds(&self) -> PrincipalSurrogateBounds {
        self.bounds
    }

    /// Reading appropriate to the retained joint endpoint evidence.
    #[must_use]
    pub const fn reading(&self) -> PatientSurrogateReading {
        self.reading
    }

    /// Stable identity of the verified request and result.
    #[must_use]
    pub const fn identity(&self) -> SemanticDigest {
        self.identity
    }

    /// Identity of the patient-level evidence and all declared premises.
    #[must_use]
    pub const fn source_identity(&self) -> SemanticDigest {
        self.source
    }

    /// The binary terminal-benefit probability in the surrogate-improver principal stratum.
    #[must_use]
    pub const fn quantity(&self) -> &ScientificQuantity {
        &self.quantity
    }

    /// Return the exact single value when the retained population set degenerates.
    ///
    /// # Errors
    /// Evidence supplied an empirical calculation or more than one compatible value.
    pub fn point(&self) -> Result<f64, PatientSurrogateError> {
        self.require_population()?;
        if !self.bounds.point_identified() {
            return Err(PatientSurrogateError::PointNotIdentified);
        }
        Ok(self.bounds.endpoints()[0])
    }

    /// Determine whether the whole population identified set supports a threshold decision.
    ///
    /// # Errors
    /// Invalid threshold or empirical evidence with no inference for the population.
    pub fn decide(
        &self,
        threshold_numerator: u64,
        threshold_denominator: u64,
    ) -> Result<SurrogateThresholdDecision, PatientSurrogateError> {
        self.require_population()?;
        if threshold_denominator == 0 || threshold_numerator > threshold_denominator {
            return Err(PatientSurrogateError::Threshold);
        }
        let cutoff = u128::from(threshold_numerator) * u128::from(self.bounds.improver_numerator);
        let scale = |value| u128::from(value) * u128::from(threshold_denominator);
        let reading = if scale(self.bounds.lower_numerator) >= cutoff {
            SurrogateThresholdReading::AllAtOrAbove
        } else if scale(self.bounds.upper_numerator) < cutoff {
            SurrogateThresholdReading::AllBelow
        } else {
            SurrogateThresholdReading::Ambiguous
        };
        let mut hash = CanonicalWriter::new("antecedent.patient_surrogate.threshold.v1");
        hash.bytes(self.identity.as_bytes()).word(threshold_numerator).word(threshold_denominator);
        Ok(SurrogateThresholdDecision {
            threshold: [threshold_numerator, threshold_denominator],
            reading,
            source: self.identity,
            identity: SemanticDigest::from_bytes(hash.finish()),
        })
    }

    fn require_population(&self) -> Result<(), PatientSurrogateError> {
        if self.reading == PatientSurrogateReading::PopulationIdentification {
            Ok(())
        } else {
            Err(PatientSurrogateError::DecisionReading)
        }
    }
}

fn valid_label(label: &str) -> bool {
    !label.trim().is_empty() && label.len() <= MAX_LABEL
}

impl PatientSurrogateRequest {
    /// Check that the declaration can identify a patient-level surrogate target.
    ///
    /// # Errors
    /// Inadequate endpoint coordinates, missing premises or an invalid evidence declaration.
    pub fn validate(&self) -> Result<(), PatientSurrogateError> {
        let design = &self.design;
        if [
            &design.trial_id,
            &design.population,
            &design.surrogate,
            &design.outcome,
            &design.control,
            &design.treatment,
        ]
        .iter()
        .any(|label| !valid_label(label))
            || design.surrogate == design.outcome
            || design.control == design.treatment
            || design.surrogate_visit == 0
            || design.surrogate_visit >= design.outcome_visit
        {
            return Err(PatientSurrogateError::Declaration(
                "name distinct interventions and the ordered joint endpoints",
            ));
        }
        if !design.consistency_and_no_interference || !design.randomized_exchangeability {
            return Err(PatientSurrogateError::Declaration(
                "declare consistency, no interference and positive individual random assignment",
            ));
        }
        if !design.complete_joint_follow_up || !design.monotone_surrogate {
            return Err(PatientSurrogateError::Declaration(
                "declare complete patient-level joint follow-up and monotone surrogate response",
            ));
        }
        match &self.evidence {
            PatientSurrogateEvidence::PopulationJointLaw { evidence_id, .. }
                if !valid_label(evidence_id) =>
            {
                return Err(PatientSurrogateError::Declaration(
                    "name the exact joint patient population law",
                ));
            }
            PatientSurrogateEvidence::CompletePatientSample { patients }
                if patients.is_empty() || patients.len() > MAX_PATIENTS =>
            {
                return Err(PatientSurrogateError::Resource);
            }
            _ => {}
        }
        Ok(())
    }

    /// Stable causal input identity including the joint endpoint evidence, not just its effects.
    ///
    /// # Errors
    /// The request violates its design contract, or evaluation was cancelled.
    pub fn identity(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<SemanticDigest, PatientSurrogateError> {
        self.validate()?;
        let design = &self.design;
        let mut key = CanonicalWriter::new("antecedent.patient_surrogate.request.v1");
        for label in [
            &design.trial_id,
            &design.population,
            &design.surrogate,
            &design.outcome,
            &design.control,
            &design.treatment,
        ] {
            key.text(label);
        }
        key.word(u64::from(design.surrogate_visit)).word(u64::from(design.outcome_visit));
        key.text("complete_individual_randomization_consistency_monotone_surrogate");
        match &self.evidence {
            PatientSurrogateEvidence::PopulationJointLaw {
                evidence_id,
                control_cells,
                treatment_cells,
            } => {
                key.word(1).text(evidence_id).u32s(control_cells).u32s(treatment_cells);
            }
            PatientSurrogateEvidence::CompletePatientSample { patients } => {
                key.word(2).word(patients.len() as u64);
                for patient in patients {
                    if ctx.cancellation.is_cancelled() {
                        return Err(PatientSurrogateError::Cancelled);
                    }
                    if !valid_label(&patient.id) {
                        return Err(PatientSurrogateError::Declaration(
                            "each observed patient needs one valid stable id",
                        ));
                    }
                    key.text(&patient.id)
                        .word(u64::from(patient.treated))
                        .word(u64::from(patient.surrogate))
                        .word(u64::from(patient.outcome));
                }
            }
        }
        if ctx.cancellation.is_cancelled() {
            return Err(PatientSurrogateError::Cancelled);
        }
        Ok(SemanticDigest::from_bytes(key.finish()))
    }

    /// Evaluate the closed-form sharp bound under the declared evidence reading.
    ///
    /// # Errors
    /// Unsupported or insufficient joint evidence, cancellation or a resource limit.
    pub fn evaluate(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<PatientSurrogateResult, PatientSurrogateError> {
        let identity = self.identity(ctx)?;
        let (cells, reading) = match &self.evidence {
            PatientSurrogateEvidence::PopulationJointLaw {
                control_cells, treatment_cells, ..
            } => (
                [*control_cells, *treatment_cells],
                PatientSurrogateReading::PopulationIdentification,
            ),
            PatientSurrogateEvidence::CompletePatientSample { patients } => {
                let _lease = ctx.admit_scratch(1, patients.len() as u64 * 64).map_err(|_| {
                    if ctx.cancellation.is_cancelled() {
                        PatientSurrogateError::Cancelled
                    } else {
                        PatientSurrogateError::Resource
                    }
                })?;
                let mut ids = BTreeSet::new();
                let mut cells = [[0_u32; 4]; 2];
                for patient in patients {
                    if ctx.cancellation.is_cancelled() {
                        return Err(PatientSurrogateError::Cancelled);
                    }
                    if !ids.insert(patient.id.as_str()) {
                        return Err(PatientSurrogateError::Declaration(
                            "the patient sample contains a repeated patient id",
                        ));
                    }
                    let cell = 2 * usize::from(patient.surrogate) + usize::from(patient.outcome);
                    cells[usize::from(patient.treated)][cell] += 1;
                }
                (cells, PatientSurrogateReading::EmpiricalPlugin)
            }
        };
        let bounds = monotone_principal_surrogate_bounds(cells[0], cells[1])?;
        let design = &self.design;
        let mut coordinate = CanonicalWriter::new("antecedent.patient_surrogate.coordinate.v1");
        coordinate
            .text(&design.surrogate)
            .text(&design.outcome)
            .word(u64::from(design.surrogate_visit))
            .word(u64::from(design.outcome_visit))
            .text(&design.control)
            .text(&design.treatment);
        let quantity = ScientificQuantity {
            variable_id: coordinate.finish_hex(),
            variable_name: design.outcome.clone(),
            role: QuantityRole::Outcome,
            units: "probability".into(),
            population_id: design.population.clone(),
            regime_id: format!("{}|{}|{}", design.trial_id, design.control, design.treatment),
            horizon: design.outcome_visit,
            functional_id: "principal_stratum_terminal_benefit".into(),
            conditioning: vec![QuantityCondition {
                variable_id: design.surrogate.clone(),
                value_id: "S(1)>S(0)".into(),
            }],
            transform_id: "identity".into(),
        };
        quantity.validate().map_err(|_| {
            PatientSurrogateError::Declaration("invalid principal surrogate quantity coordinate")
        })?;
        let mut readout = CanonicalWriter::new("antecedent.patient_surrogate.result.v1");
        readout.bytes(identity.as_bytes()).words(&[
            bounds.lower_numerator,
            bounds.upper_numerator,
            bounds.improver_numerator,
            bounds.law_denominator,
        ]);
        Ok(PatientSurrogateResult {
            bounds,
            reading,
            quantity,
            source: identity,
            identity: SemanticDigest::from_bytes(readout.finish()),
        })
    }

    /// Retain the joint evidence and independently verifiable result.
    ///
    /// # Errors
    /// Mismatched source or unsupported artifact encoding.
    pub fn export(
        &self,
        result: &PatientSurrogateResult,
        program: SemanticDigest,
        artifact_id: &str,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, PatientSurrogateError> {
        if self.identity(ctx)? != result.source {
            return Err(PatientSurrogateError::ReplayMismatch);
        }
        let meta = ArtifactMeta {
            version: 1,
            program: *program.as_bytes(),
            result: *result.identity.as_bytes(),
            readout: [
                result.bounds.lower_numerator,
                result.bounds.upper_numerator,
                result.bounds.improver_numerator,
                result.bounds.law_denominator,
            ],
            request: self,
        };
        Ok(crate::artifact_codec::encode_sectioned_io(
            &ArtifactContainer {
                kind: PATIENT_SURROGATE_ARTIFACT_KIND,
                note: "patient_surrogate",
                max_bytes: MAX_BYTES,
                artifact_id,
                library_version: env!("CARGO_PKG_VERSION"),
            },
            METADATA,
            &meta,
            &[],
            PATIENT_SURROGATE_ARTIFACT_KIND,
        )?)
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ArtifactMeta<Request> {
    version: u16,
    program: [u8; 32],
    result: [u8; 32],
    readout: [u64; 4],
    request: Request,
}

/// Retained request and expectations. The caller must replay the numerical identification.
pub struct StoredPatientSurrogate {
    /// Bound trial and patient-level law.
    pub request: PatientSurrogateRequest,
    /// Independently reproducible causal program identity.
    pub program: SemanticDigest,
    result: SemanticDigest,
    readout: [u64; 4],
}

impl StoredPatientSurrogate {
    /// Decode the versioned artifact within its byte bound.
    ///
    /// # Errors
    /// Invalid or unsupported artifact format.
    pub fn decode(bytes: &[u8]) -> Result<Self, PatientSurrogateError> {
        let (meta, []): (ArtifactMeta<PatientSurrogateRequest>, _) =
            crate::artifact_codec::decode_sectioned_io_versioned(
                PATIENT_SURROGATE_ARTIFACT_KIND,
                MAX_BYTES,
                METADATA,
                &[],
                PATIENT_SURROGATE_ARTIFACT_KIND,
                1,
                bytes,
            )?;
        Ok(Self {
            request: meta.request,
            program: SemanticDigest::from_bytes(meta.program),
            result: SemanticDigest::from_bytes(meta.result),
            readout: meta.readout,
        })
    }

    /// Verify independent expectations before re-running the causal compiler.
    ///
    /// # Errors
    /// Invalid format or mismatched expected identities.
    pub fn open(
        bytes: &[u8],
        program: SemanticDigest,
        result: SemanticDigest,
    ) -> Result<Self, PatientSurrogateError> {
        let stored = Self::decode(bytes)?;
        if stored.program != program || stored.result != result {
            return Err(PatientSurrogateError::ReplayMismatch);
        }
        Ok(stored)
    }

    /// Compare the computed request outcome to every retained result field.
    ///
    /// # Errors
    /// Recomputed numeric or causal program identity differs from the artifact.
    pub fn verify(
        &self,
        program: SemanticDigest,
        result: &PatientSurrogateResult,
    ) -> Result<(), PatientSurrogateError> {
        let bounds = result.bounds;
        if program != self.program
            || result.identity != self.result
            || self.readout
                != [
                    bounds.lower_numerator,
                    bounds.upper_numerator,
                    bounds.improver_numerator,
                    bounds.law_denominator,
                ]
        {
            return Err(PatientSurrogateError::ReplayMismatch);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patient_surrogate_artifact_rejects_a_resealed_edited_bound() {
        let request = PatientSurrogateRequest {
            design: PatientSurrogateDesign {
                trial_id: "trial-001".into(),
                population: "adults".into(),
                surrogate: "early".into(),
                outcome: "terminal".into(),
                surrogate_visit: 1,
                outcome_visit: 4,
                control: "usual-care".into(),
                treatment: "active".into(),
                consistency_and_no_interference: true,
                randomized_exchangeability: true,
                complete_joint_follow_up: true,
                monotone_surrogate: true,
            },
            evidence: PatientSurrogateEvidence::PopulationJointLaw {
                evidence_id: "joint-population-law".into(),
                control_cells: [3, 0, 1, 0],
                treatment_cells: [0, 0, 1, 3],
            },
        };
        let ctx = ExecutionContext::for_tests(16);
        let live = request.evaluate(&ctx).unwrap();
        let program = SemanticDigest::from_bytes([1; 32]);
        let artifact = request.export(&live, program, "patient-surrogate", &ctx).unwrap();
        let (mut meta, []): (ArtifactMeta<PatientSurrogateRequest>, _) =
            crate::artifact_codec::decode_sectioned_io_versioned(
                PATIENT_SURROGATE_ARTIFACT_KIND,
                MAX_BYTES,
                METADATA,
                &[],
                PATIENT_SURROGATE_ARTIFACT_KIND,
                1,
                &artifact,
            )
            .unwrap();
        meta.readout[0] = 1;
        let tampered = crate::artifact_codec::encode_sectioned_io(
            &ArtifactContainer {
                kind: PATIENT_SURROGATE_ARTIFACT_KIND,
                note: "tampered",
                max_bytes: MAX_BYTES,
                artifact_id: "patient-surrogate",
                library_version: env!("CARGO_PKG_VERSION"),
            },
            METADATA,
            &meta,
            &[],
            PATIENT_SURROGATE_ARTIFACT_KIND,
        )
        .unwrap();
        let stored = StoredPatientSurrogate::open(&tampered, program, live.identity()).unwrap();
        assert!(matches!(
            stored.verify(program, &live),
            Err(PatientSurrogateError::ReplayMismatch)
        ));
    }
}
