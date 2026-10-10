//! Checked single-source z-transport evidence repair through the generic search.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{EvidenceObligation, EvidenceRegime, ExecutionContext, reason_code};
use antecedent_identify::{
    IdentificationError, SidLimits, bind_z_transport_catalog, verify_z_transport_derivation,
};

use crate::obligation_adapters::z_transport_obligations;
use crate::repair::{RepairCancelled, RepairDerivation};
use crate::z_transport_planner::{ZTransportFailureSnapshot, ZTransportFailureStatus};
use crate::{DurableStudyCandidate, FamilyVerdict, RepairError, RepairFamily};

/// A missing-evidence z-transport proof, with its original leaf obligations.
///
/// Candidate evidence remains hypothetical. Every examined subset independently
/// re-verifies the frozen theorem derivation and binds its exact factors against
/// the preview catalog; matching variable names alone cannot repair a query.
#[derive(Clone, Debug)]
pub struct ZTransportRepairFamily {
    snapshot: ZTransportFailureSnapshot,
    limits: SidLimits,
    identity: String,
    obligations: Vec<EvidenceObligation>,
}

impl ZTransportRepairFamily {
    /// Consume a checked, possibly independently restored failure snapshot.
    ///
    /// # Errors
    /// A failure without a checked positive formula is not an evidence-repair
    /// contract. Obstructions and exhausted searches must retain their own status.
    pub fn try_new(
        snapshot: ZTransportFailureSnapshot,
        limits: SidLimits,
    ) -> Result<Self, RepairError> {
        let invalid = |message: String| RepairError {
            code: reason_code!("invalid_argument"),
            detail: "identification_repair.invalid_request",
            message,
            receipt: None,
        };
        if snapshot.status() != &ZTransportFailureStatus::MissingEvidence
            || snapshot.derivation().is_none()
        {
            return Err(invalid(
                "z-transport evidence repair requires a checked missing-evidence formula".into(),
            ));
        }
        let obligations = z_transport_obligations(&snapshot, snapshot.catalog())
            .map_err(|e| invalid(e.to_string()))?;
        let identity = snapshot.to_wire().map_err(|e| invalid(e.to_string()))?.snapshot_digest;
        Ok(Self { snapshot, limits, identity, obligations })
    }

    /// Identification limits used for independent theorem verification.
    #[must_use]
    pub const fn sid_limits(&self) -> SidLimits {
        self.limits
    }

    /// Proposed regimes delivered by these study declarations.
    ///
    /// # Errors
    /// Invalid study declarations.
    pub fn hypothetical_regimes(
        &self,
        studies: &[&DurableStudyCandidate],
    ) -> Result<Vec<EvidenceRegime>, Vec<String>> {
        crate::repair::hypothetical_regimes(self.snapshot.catalog(), studies)
    }

    /// Recheck a stored hypothetical delta against this frozen proof.
    ///
    /// # Errors
    /// Cooperative cancellation.
    pub fn check_regimes(
        &self,
        regimes: Vec<EvidenceRegime>,
        ctx: &ExecutionContext,
    ) -> Result<FamilyVerdict, RepairCancelled> {
        if ctx.cancellation.is_cancelled() {
            return Err(RepairCancelled);
        }
        let delta = match crate::repair::delta_of_regimes(self.snapshot.catalog(), regimes) {
            Ok(Some(delta)) => delta,
            Ok(None) => {
                return Ok(FamilyVerdict::Insufficient {
                    reasons: vec![
                        "no proposed regime supplies a missing z-transport factor".into(),
                    ],
                });
            }
            Err(reasons) => return Ok(FamilyVerdict::Invalid { reasons }),
        };
        let preview = match delta
            .preview_catalog(self.snapshot.catalog())
            .and_then(|preview| delta.with_placeholder_bindings(&preview))
        {
            Ok(preview) => preview,
            Err(error) => return Ok(FamilyVerdict::Invalid { reasons: vec![error.to_string()] }),
        };
        // Constructor guarantees a checked positive formula.
        let derivation = self.snapshot.derivation().expect("checked repair formula");
        let verification = verify_z_transport_derivation(
            self.snapshot.diagram(),
            self.snapshot.query(),
            derivation,
            self.limits,
            ctx,
        );
        if let Err(error) = verification {
            if matches!(error, IdentificationError::Cancelled) {
                return Err(RepairCancelled);
            }
            return Ok(FamilyVerdict::NotCertified { reasons: vec![error.to_string()] });
        }
        match bind_z_transport_catalog(
            self.snapshot.diagram(),
            self.snapshot.query(),
            derivation,
            &preview,
        ) {
            Ok(_) => Ok(FamilyVerdict::Sufficient(RepairDerivation {
                checker: "z_transport.catalog",
                steps: derivation
                    .inspect_proof(&preview)
                    .factors
                    .iter()
                    .map(|f| format!("leaf:{}", f.leaf))
                    .collect(),
                verified: true,
            })),
            Err(error @ IdentificationError::MissingEvidence { .. }) => {
                Ok(FamilyVerdict::Insufficient { reasons: vec![error.to_string()] })
            }
            Err(IdentificationError::Cancelled) => Err(RepairCancelled),
            Err(error) => Ok(FamilyVerdict::NotCertified { reasons: vec![error.to_string()] }),
        }
    }

    /// Frozen portable proof and evidence state of this repair contract.
    #[must_use]
    pub fn snapshot(&self) -> &ZTransportFailureSnapshot {
        &self.snapshot
    }
}

impl RepairFamily for ZTransportRepairFamily {
    fn family_id(&self) -> &'static str {
        "z_transport"
    }
    fn contract_id(&self) -> String {
        format!("z_transport:{}", self.identity)
    }
    fn unresolved_obligations(&self) -> Vec<EvidenceObligation> {
        self.obligations.clone()
    }

    fn check(
        &self,
        studies: &[&DurableStudyCandidate],
        ctx: &ExecutionContext,
    ) -> Result<FamilyVerdict, RepairCancelled> {
        if ctx.cancellation.is_cancelled() {
            return Err(RepairCancelled);
        }
        match self.hypothetical_regimes(studies) {
            Ok(regimes) => self.check_regimes(regimes, ctx),
            Err(reasons) => Ok(FamilyVerdict::Invalid { reasons }),
        }
    }
}
