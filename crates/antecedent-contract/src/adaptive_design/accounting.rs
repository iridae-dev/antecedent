//! Observation accounting across interim analyses.
//!
//! Every interim analysis of an adaptive experiment re-uses the observations
//! of the analyses before it. The evidence ledger is the one representation
//! of observation overlap; this module registers the data behind each look as
//! a ledger source (each look a subset of the next), then records the
//! repeated use as one ledger result. The ledger then counts each observation
//! once, and refuses to pool the looks as independent evidence.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::declaration::AdaptiveDesignContract;
use crate::ExternalRefusal;
use crate::evidence_ledger::{
    CombinationRequest, EvidenceLedger, LedgerError, Lineage, MassAccount, ObservationId,
    PoolingDependence, RefusalKind, ResultId, SourceDeclaration, SourceId, SourceKind,
    SynthesisOperation,
};

/// Stage name recorded when the looks re-use their observations.
pub const INTERIM_REUSE_STAGE: &str = "interim_analysis";

/// The ledger of the observations behind each interim analysis.
#[derive(Clone, Debug)]
pub struct InterimEvidence {
    /// The ledger, holding one source per look and the repeated-use result.
    pub ledger: EvidenceLedger,
    /// The source of each look, in look order.
    pub look_sources: Vec<SourceId>,
    /// The repeated-use result over every look.
    pub reuse: ResultId,
    /// Mass accounting of the looks: each observation counted once.
    pub mass: MassAccount,
}

fn ledger_refusal(error: &LedgerError) -> ExternalRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("invalid_argument"),
        "bind",
        "adaptive_inference.ledger_limit",
    )
    .with_supplied(error.detail)
    .with_remedy("reduce the number of units or interim analyses so the evidence ledger can account for them")
}

impl AdaptiveDesignContract {
    /// Identity of the observation behind `unit` in the evidence ledger.
    ///
    /// # Panics
    /// When `unit` does not fit in `u32`; a validated contract has at most
    /// `u32::MAX` units.
    #[must_use]
    pub fn observation_id(&self, unit: usize) -> ObservationId {
        let key = u32::try_from(unit).expect("a validated contract has at most u32::MAX units");
        ObservationId::from_key(&self.identity().to_hex(), &key.to_le_bytes())
    }

    /// Register the data behind each interim analysis and the repeated use.
    ///
    /// # Errors
    /// A ledger bound exceeded (too many units or looks).
    #[allow(
        clippy::result_large_err,
        reason = "a refusal is the cold path of a once-per-call check; boxing the error would not pay"
    )]
    pub fn interim_evidence(&self) -> Result<InterimEvidence, ExternalRefusal> {
        let looks = self.looks();
        let mut ledger = EvidenceLedger::new();
        let mut look_sources: Vec<Option<SourceId>> = vec![None; looks];
        // Registered from the last look backwards so that each look is a
        // declared subset of the one after it.
        for look in (0..looks).rev() {
            let mut rows: Vec<ObservationId> =
                self.analysis_units(look).into_iter().map(|u| self.observation_id(u)).collect();
            rows.sort_unstable();
            let lineage = match look_sources.get(look + 1).copied().flatten() {
                Some(of) => Lineage::Subset { of },
                None => Lineage::Original,
            };
            let declaration = SourceDeclaration::new(
                format!("interim_analysis_{look}"),
                SourceKind::ObservationSet,
            )
            .with_observations(rows)
            .with_lineage(lineage);
            let id = ledger.add_source(declaration).map_err(|e| ledger_refusal(&e))?;
            look_sources[look] = Some(id);
        }
        let look_sources: Vec<SourceId> = look_sources.into_iter().flatten().collect();
        let reuse = ledger
            .combine(CombinationRequest {
                label: "adaptive_interim_reuse".to_owned(),
                operation: SynthesisOperation::RepeatedObservationUse {
                    sources: look_sources.clone(),
                    stage: INTERIM_REUSE_STAGE.to_owned(),
                },
                replaces: None,
            })
            .map_err(|e| ledger_refusal(&e))?;
        let mass = ledger.mass_account(&look_sources).map_err(|e| ledger_refusal(&e))?;
        Ok(InterimEvidence { ledger, look_sources, reuse, mass })
    }

    /// Ask the ledger to pool the interim analyses as independent studies.
    ///
    /// The looks share observations, so the ledger refuses; this returns that
    /// refusal in the adaptive-inference vocabulary. `Ok` only when no look
    /// shares an observation with another (every earlier look empty).
    ///
    /// # Errors
    /// The refusal that a pooling of the looks as independent would double
    /// count their observations.
    #[allow(
        clippy::result_large_err,
        reason = "a refusal is the cold path of a once-per-call check; boxing the error would not pay"
    )]
    pub fn refuse_independent_pooling_of_looks(
        &self,
        evidence: &InterimEvidence,
    ) -> Result<(), ExternalRefusal> {
        if evidence.look_sources.len() < 2 {
            return Ok(());
        }
        let mut scratch = evidence.ledger.clone();
        let outcome = scratch.combine(CombinationRequest {
            label: "interim_looks_pooled_as_independent".to_owned(),
            operation: SynthesisOperation::StatisticalPooling {
                sources: evidence.look_sources.clone(),
                dependence: PoolingDependence::Independent,
            },
            replaces: None,
        });
        match outcome {
            Ok(_) => Ok(()),
            Err(error) => Err(match error.kind {
                RefusalKind::OverlapUnknown => ExternalRefusal::new(
                    antecedent_core::reason_code!("evidence_overlap_unknown"),
                    "bind",
                    "adaptive_inference.looks_overlap_unknown",
                )
                .with_remedy("declare the observations behind each interim analysis"),
                _ => ExternalRefusal::new(
                    antecedent_core::reason_code!("evidence_double_counted"),
                    "bind",
                    "adaptive_inference.looks_share_observations",
                )
                .with_supplied(error.detail)
                .with_remedy(
                    "treat the interim analyses as nested analyses of one growing sample: use the final analysis, or a method that accounts for the repeated looks",
                ),
            }),
        }
    }
}
