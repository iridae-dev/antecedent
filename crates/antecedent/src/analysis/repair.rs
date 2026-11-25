//! Identification-repair facade: which proposed studies make a failed causal
//! contract identify.
//!
//! A failed contract is a [`RepairFamily`]: a transport contract that misses
//! evidence ([`TransportRepairFamily`]) or a fixed-graph back-door contract
//! whose adjustment covariates were never measured jointly
//! ([`BackdoorRepairFamily`]), or a checked z-transport proof whose required
//! factors are missing ([`ZTransportRepairFamily`]). A failed identification
//! also yields one family per identifier: front-door ([`FrontDoorRepairFamily`]),
//! instrument ([`InstrumentRepairFamily`]), hedge ([`HedgeRepairFamily`]) and
//! sensitivity ([`SensitivityRepairFamily`]), and
//! [`identification_failure_obligations`] reads all of them from one failed
//! identification result. The family exposes its unresolved
//! [`EvidenceObligation`]s, each retaining its source proof step, and
//! [`repair_contract`] applies the hypothetical evidence of
//! [`DurableStudyCandidate`]s (one study, or a bounded subset) and re-runs the
//! family's own theorem checker on it. A candidate that only matches a variable
//! name never repairs a contract, and a search that a budget stopped keeps its
//! unevaluated subsets: exhaustion is never impossibility.
//!
//! [`RepairReportArtifact::build`] seals a finished report as a portable
//! `repair_search_receipt_v1` artifact that [`consume_repair_artifact`] replays
//! independently; a report is kept even when its artifact cannot be built. The specialized planners ([`crate::design::StudyPlanResult`]
//! and the transport and z-transport planners) are untouched.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_contract::{
    EvidenceObligation, EvidenceObligationKind, ObligationProvenance, ObligationRegime,
};
use antecedent_core::ExecutionContext;
pub use antecedent_design::{
    BackdoorRepairFamily, DurableStudyCandidate, ExpectedEvidence, FamilyVerdict,
    FrontDoorRepairFamily, HedgeRepairFamily, InstrumentRepairFamily, LatentCause,
    RepairArtifactError, RepairClassification, RepairConsumeLimits, RepairError, RepairFamily,
    RepairFamilyRef, RepairLimits, RepairObjective, RepairOutcome, RepairReceipt, RepairReport,
    RepairReportArtifact, SensitivityFinding, SensitivityMethod, SensitivityRepairFamily,
    StudyCandidateError, StudyCostDeclaration, StudyKind, TransportRepairFamily, UnitRules,
    ZTransportRepairFamily, latent_expansion,
};

/// The evidence that would resolve a failed identification, one verified
/// obligation per route, each retaining the proof step it was derived from.
///
/// `graph` is the structure the identification ran on (bidirected edges become
/// unmeasured common causes, one past the largest graph variable each), and
/// `observed` the variables measured jointly with the treatment and the outcome
/// in `population`. The routes are:
///
/// * back-door: one joint observational law over the treatment, the outcome and
///   the first admissible adjustment set (`BackdoorRepairFamily`);
/// * front-door: one joint law over the treatment, the outcome and a mediator
///   set the criterion accepts (`FrontDoorRepairFamily`);
/// * instrument: a joint law with an instrument, with its relevance, exclusion
///   and effect restriction as `EstablishAssumption` obligations that no study
///   satisfies, or the search for an instrument when the graph has none
///   (`InstrumentRepairFamily`);
/// * hedge: experiments on the treatment and on the hedge nodes that identify the
///   effect through a surrogate (`HedgeRepairFamily`);
/// * sensitivity: the confounder to measure when `sensitivity` reports a
///   conclusion an unmeasured confounder would overturn
///   (`SensitivityRepairFamily`);
/// * panel: repeated measurement, when back-door adjustment needs variables that
///   are not observed.
///
/// A route whose observed law already identifies the effect contributes nothing.
/// The query must be an average effect; other queries owe nothing here.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_design::obligation_adapters::identification_failure_obligations`].
///
/// # Errors
/// `identification_repair.not_a_failure` when `result` is identified, and
/// `identification_repair.invalid_request` for a graph or variables the
/// identifiers cannot read.
#[deprecated(
    since = "3.1.0",
    note = "use the compiled EvidenceObligations route, which calls this kernel function with the same inputs"
)]
pub fn identification_failure_obligations(
    result: &antecedent_identify::result::IdentificationResult,
    graph: &antecedent_graph::admg::Admg,
    observed: &[antecedent_core::ids::VariableId],
    population: &str,
    sensitivity: Option<&antecedent_design::SensitivityFinding>,
) -> Result<
    Vec<antecedent_contract::evidence_obligation::EvidenceObligation>,
    antecedent_design::RepairError,
> {
    antecedent_design::obligation_adapters::identification_failure_obligations(
        result,
        graph,
        observed,
        population,
        sensitivity,
    )
}

/// The unresolved evidence obligations of a failed contract, each with its
/// source proof step. A study never satisfies an
/// [`EvidenceObligationKind::EstablishAssumption`] obligation.
#[must_use]
pub fn unresolved_obligations(family: &dyn RepairFamily) -> Vec<EvidenceObligation> {
    family.unresolved_obligations()
}

/// Rank the candidates (and bounded subsets of them) that repair the contract.
///
/// # Errors
/// `identification_repair.cost_units_mismatch` when cost ranking
/// compares different unit labels and `identification_repair.budget` when the
/// budget is stopped before the search. A stop during the search is a report.
pub fn repair_contract(
    family: &dyn RepairFamily,
    candidates: &[DurableStudyCandidate],
    objective: RepairObjective,
    limits: RepairLimits,
    ctx: &ExecutionContext,
) -> Result<RepairReport, RepairError> {
    antecedent_design::repair(family, candidates, objective, limits, ctx)
}

/// Read a `repair_search_receipt_v1` container and replay it under the
/// consumer's `limits`: every stored classification is recomputed by the
/// family's theorem checker and the whole search must reproduce the stored
/// report. Returns the artifact and the replayed report.
///
/// # Errors
/// An unsupported version, stored limits beyond `limits`, a digest or replay
/// that does not match, or a replay that cancellation stopped.
pub fn consume_repair_artifact(
    bytes: &[u8],
    limits: RepairConsumeLimits,
    ctx: &ExecutionContext,
) -> Result<(RepairReportArtifact, RepairReport), RepairArtifactError> {
    let artifact = RepairReportArtifact::from_bytes(bytes)?;
    let report = artifact.consume(limits, ctx)?;
    Ok((artifact, report))
}
