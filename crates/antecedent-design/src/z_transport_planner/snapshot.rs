//! Frozen z-transport failure snapshots and independent replay.

use super::{
    Arc, ClassicalTransportQuery, ClassicalTransportResult, EvidenceCatalog, ExecutionContext,
    IdentificationBudget, IdentificationError, SelectionDiagram, SidLimits, ZTransportDecision,
    ZTransportDerivation, ZTransportLimitsReceipt, ZTransportObstruction,
    ZTransportObstructionRecord, ZTransportPlanningError, ZTransportProofInspection,
    ZTransportQuery, ZTransportQueryWire, ZTransportResult, bind_z_transport_catalog,
    catalog_digest, decide_z_transport_with_catalog, failure_snapshot_digest, graph_digest,
    identify_classical_transport, identify_z_transport_reporting, ids,
    target_joint_observational_law, validate_z_experiment_family,
};
use serde::{Deserialize, Serialize};

/// Why the original z-transport attempt did not yield an available result.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ZTransportFailureStatus {
    /// A checked formula exists, but one or more required joint laws are absent.
    MissingEvidence,
    /// The graph/query is outside the supported theorem input.
    UnsupportedInput,
    /// The complete theorem premises establish an obstruction.
    ProofObstruction,
    /// The bounded computation stopped before reaching a decision.
    ExhaustedComputation,
    /// The bounded search has neither a positive formula nor a checked obstruction.
    UnresolvedIdentification,
}

/// Portable record of one failed z-transport request and its exact evidence state.
#[derive(Clone, Debug)]
pub struct ZTransportFailureSnapshot {
    pub(super) diagram: SelectionDiagram,
    pub(super) query: ZTransportQuery,
    pub(super) catalog: EvidenceCatalog,
    pub(super) status: ZTransportFailureStatus,
    pub(super) obligations: Arc<[Arc<str>]>,
    pub(super) proof_graph: Option<ZTransportProofInspection>,
    pub(super) obstruction: Option<antecedent_identify::sid::SHedgeRecord>,
    pub(super) z_obstruction: Option<ZTransportObstructionRecord>,
    /// Limits in force and what the bounded search consumed, when the status is
    /// an exhausted computation. Absent for every other status.
    pub(super) limits_receipt: Option<ZTransportLimitsReceipt>,
    /// The checked positive formula of a missing-evidence failure, identified
    /// once at snapshot time and reused by every proposal.
    pub(super) derivation: Option<Arc<ZTransportDerivation>>,
    pub(super) catalog_digest: String,
    pub(super) graph_digest: String,
}

/// Serializable representation of a failure snapshot.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ZTransportFailureSnapshotWire {
    /// Schema version.
    pub version: u32,
    /// Query coordinates and concrete source assignment.
    pub query: ZTransportQueryWire,
    /// Shared source/target ADMG.
    pub graph: antecedent_io_core::wire::AdmgWire,
    /// Population-specific mechanisms.
    pub selection_targets: Vec<u32>,
    /// Exact catalog in canonical order, including currently available bindings.
    pub catalog: antecedent_io_core::transport_catalog_wire::EvidenceCatalogWire,
    /// Failure classification and missing factor obligations.
    pub status: ZTransportFailureStatus,
    /// Human-readable and machine-stable obligation records.
    pub obligations: Vec<String>,
    /// Reachable checked formula graph with factor-level binding failures.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proof_graph: Option<ZTransportProofInspection>,
    /// Checked stronger-family s-hedge, if one proves this z query impossible.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub obstruction: Option<antecedent_identify::sid::SHedgeRecord>,
    /// Checked bounded `TRz` line-11 obstruction, with complete-family evidence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub z_obstruction: Option<ZTransportObstructionRecord>,
    /// Limits in force and what the search consumed when the bounded computation
    /// was exhausted, so an exhausted status is inspectable, not opaque.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limits_receipt: Option<ZTransportLimitsReceipt>,
    /// Canonical digest of the frozen catalog.
    pub catalog_digest: String,
    /// Digest of the sorted graph wire and selection targets.
    pub graph_digest: String,
    /// Digest of every snapshot field, including query, status, and obligations.
    pub snapshot_digest: String,
}

/// Snapshot wire versions this reader accepts.
const SNAPSHOT_VERSIONS: [u32; 2] = [1, 2];

impl ZTransportFailureSnapshot {
    /// Frozen selection diagram used by the checked proof.
    #[must_use]
    pub fn diagram(&self) -> &SelectionDiagram {
        &self.diagram
    }

    /// Frozen query, including the controllable intervention set.
    #[must_use]
    pub fn query(&self) -> &ZTransportQuery {
        &self.query
    }

    /// Evidence state on which the failure was established.
    #[must_use]
    pub fn catalog(&self) -> &EvidenceCatalog {
        &self.catalog
    }

    /// Freeze an outcome, catalog, query, and graph as the planning anchor.
    fn new(
        diagram: &SelectionDiagram,
        query: &ZTransportQuery,
        catalog: &EvidenceCatalog,
        status: ZTransportFailureStatus,
        obligations: impl Into<Arc<[Arc<str>]>>,
    ) -> Result<Self, ZTransportPlanningError> {
        catalog.validate().map_err(|error| ZTransportPlanningError::Invalid(error.to_string()))?;
        Ok(Self {
            diagram: diagram.clone(),
            query: query.clone(),
            catalog: catalog.clone(),
            status,
            obligations: obligations.into(),
            proof_graph: None,
            obstruction: None,
            z_obstruction: None,
            limits_receipt: None,
            derivation: None,
            catalog_digest: catalog_digest(catalog)?,
            graph_digest: graph_digest(diagram)?,
        })
    }

    /// Snapshot status.
    #[must_use]
    pub const fn status(&self) -> &ZTransportFailureStatus {
        &self.status
    }

    /// Missing factor obligations from the original attempt.
    #[must_use]
    pub fn obligations(&self) -> &[Arc<str>] {
        &self.obligations
    }

    /// Frozen catalog identity.
    #[must_use]
    pub fn catalog_digest(&self) -> &str {
        &self.catalog_digest
    }

    /// The checked positive formula this failure could not bind, when the
    /// status is missing evidence.
    #[must_use]
    pub fn derivation(&self) -> Option<&ZTransportDerivation> {
        self.derivation.as_deref()
    }

    /// Limits in force and what the search consumed, when a budget stopped the
    /// bounded computation. Absent for every other status.
    #[must_use]
    pub const fn limits_receipt(&self) -> Option<&ZTransportLimitsReceipt> {
        self.limits_receipt.as_ref()
    }

    /// Encode all inputs needed to independently inspect this failure.
    ///
    /// # Errors
    /// Graph or catalog encoding failure.
    pub fn to_wire(&self) -> Result<ZTransportFailureSnapshotWire, ZTransportPlanningError> {
        let catalog = self
            .catalog
            .canonicalized()
            .map_err(|error| ZTransportPlanningError::Invalid(error.to_string()))?;
        let mut wire = ZTransportFailureSnapshotWire {
            version: if self.z_obstruction.is_some() { 2 } else { 1 },
            query: ZTransportQueryWire::from_query(&self.query),
            graph: antecedent_io_core::admg_to_wire(self.diagram.causal_graph())?,
            selection_targets: ids(self.diagram.selection_targets()),
            catalog: antecedent_io_core::transport_catalog_wire::EvidenceCatalogWire::from_catalog(
                &catalog,
            ),
            status: self.status.clone(),
            obligations: self.obligations.iter().map(ToString::to_string).collect(),
            proof_graph: self.proof_graph.clone(),
            obstruction: self.obstruction.clone(),
            z_obstruction: self.z_obstruction.clone(),
            limits_receipt: self.limits_receipt.clone(),
            catalog_digest: self.catalog_digest.clone(),
            graph_digest: self.graph_digest.clone(),
            snapshot_digest: String::new(),
        };
        wire.snapshot_digest = failure_snapshot_digest(&wire)?;
        Ok(wire)
    }

    /// Restore and verify a portable failure snapshot under the caller's
    /// identification limits and execution context.
    ///
    /// # Errors
    /// An unsupported version, a proof that does not replay, a digest that does
    /// not match, or a recomputed outcome that differs from the recorded one.
    // This is intentionally one validation transaction: decomposing it risks
    // obscuring the order in which nested proofs are checked against frozen inputs.
    pub fn from_wire(
        wire: &ZTransportFailureSnapshotWire,
        limits: SidLimits,
        ctx: &ExecutionContext,
    ) -> Result<Self, ZTransportPlanningError> {
        if !SNAPSHOT_VERSIONS.contains(&wire.version) {
            return Err(ZTransportPlanningError::UnsupportedVersion { version: wire.version });
        }
        if wire.version == 1 && wire.z_obstruction.is_some() {
            return Err(ZTransportPlanningError::Invalid(
                "a version 1 failure snapshot carries no line-11 obstruction".into(),
            ));
        }
        let graph = antecedent_io_core::admg_from_wire(&wire.graph)?;
        let diagram = SelectionDiagram::try_new(
            graph,
            wire.selection_targets
                .iter()
                .copied()
                .map(antecedent_core::VariableId::from_raw)
                .collect::<Vec<_>>(),
        )
        .map_err(|e| ZTransportPlanningError::Invalid(e.to_string()))?;
        let catalog = wire.catalog.to_catalog()?;
        let query = wire.query.to_query();
        let mut snapshot = Self::new(
            &diagram,
            &query,
            &catalog,
            wire.status.clone(),
            wire.obligations
                .iter()
                .map(|value| Arc::<str>::from(value.as_str()))
                .collect::<Vec<_>>(),
        )?;
        snapshot.proof_graph.clone_from(&wire.proof_graph);
        snapshot.limits_receipt.clone_from(&wire.limits_receipt);
        if let Some(record) = &wire.obstruction {
            let classical = ClassicalTransportQuery {
                outcomes: Arc::clone(&query.outcomes),
                treatments: Arc::clone(&query.treatments),
                source: Arc::clone(&query.source),
                target: Arc::clone(&query.target),
            };
            antecedent_identify::sid::SHedgeCertificate::from_record_checked(
                record.clone(),
                &diagram,
                &classical,
                ctx,
            )?;
            snapshot.obstruction = Some(record.clone());
        }
        if let Some(record) = &wire.z_obstruction {
            ZTransportObstruction::from_record_checked(
                record.clone(),
                &diagram,
                &query,
                limits,
                ctx,
            )?;
            snapshot.z_obstruction = Some(record.clone());
        }
        let recomputed = snapshot_z_transport_failure(&diagram, &query, &catalog, limits, ctx)?;
        if recomputed.status != snapshot.status
            || recomputed.obligations != snapshot.obligations
            || recomputed.obstruction != snapshot.obstruction
            || recomputed.z_obstruction != snapshot.z_obstruction
            || recomputed.proof_graph != snapshot.proof_graph
            || recomputed.limits_receipt != snapshot.limits_receipt
        {
            return Err(ZTransportPlanningError::Invalid(
                "failure snapshot outcome differs from current checked identification".into(),
            ));
        }
        if snapshot.catalog_digest != wire.catalog_digest
            || snapshot.graph_digest != wire.graph_digest
            || failure_snapshot_digest(wire)? != wire.snapshot_digest
        {
            return Err(ZTransportPlanningError::DigestMismatch);
        }
        snapshot.derivation = recomputed.derivation;
        Ok(snapshot)
    }
}

/// Capture an explicit failed outcome from the currently supported identifier
/// under the caller's identification limits and execution context.
///
/// # Errors
/// The query binds to available evidence (there is no failure to snapshot), an
/// invalid catalog, or an identification error that is not a classified
/// refusal (an unsupported input or exhausted budget is a snapshot status, not
/// an error).
// This routine classifies one bounded theorem/availability result and attaches
// the corresponding checked proof; keep that mapping visible in one place.
#[expect(
    clippy::too_many_lines,
    reason = "keeps failure classification and proof attachment together"
)]
pub fn snapshot_z_transport_failure(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    catalog: &EvidenceCatalog,
    limits: SidLimits,
    ctx: &ExecutionContext,
) -> Result<ZTransportFailureSnapshot, ZTransportPlanningError> {
    let mut receipt = None;
    let identification =
        match identify_z_transport_reporting(diagram, query, limits, ctx, &mut receipt) {
            Ok(result) => result,
            Err(error) => {
                let status = match &error {
                    IdentificationError::UnsupportedInput { .. } => {
                        Some(ZTransportFailureStatus::UnsupportedInput)
                    }
                    IdentificationError::Budget { budget: IdentificationBudget::ZTransport } => {
                        Some(ZTransportFailureStatus::ExhaustedComputation)
                    }
                    _ => None,
                };
                if let Some(status) = status {
                    let mut snapshot = ZTransportFailureSnapshot::new(
                        diagram,
                        query,
                        catalog,
                        status,
                        [Arc::from(error.to_string())],
                    )?;
                    snapshot.limits_receipt = receipt;
                    return Ok(snapshot);
                }
                return Err(error.into());
            }
        };
    match identification {
        ZTransportResult::Identified(derivation) => {
            match bind_z_transport_catalog(diagram, query, &derivation, catalog) {
                Ok(_) => Err(ZTransportPlanningError::Invalid(
                    "query already binds to available evidence".into(),
                )),
                Err(error @ IdentificationError::MissingEvidence { .. }) => {
                    let message = error.to_string();
                    let graph = derivation.inspect_proof(catalog);
                    let obligations = graph
                        .factors
                        .iter()
                        .filter(|factor| factor.failure.is_some())
                        .map(|factor| {
                            Arc::<str>::from(
                                serde_json::to_string(factor).unwrap_or_else(|_| message.clone()),
                            )
                        })
                        .collect::<Vec<_>>();
                    let mut snapshot = ZTransportFailureSnapshot::new(
                        diagram,
                        query,
                        catalog,
                        ZTransportFailureStatus::MissingEvidence,
                        if obligations.is_empty() { vec![Arc::from(message)] } else { obligations },
                    )?;
                    snapshot.proof_graph = Some(graph);
                    snapshot.derivation = Some(Arc::new(*derivation));
                    Ok(snapshot)
                }
                Err(error @ IdentificationError::UnsupportedInput { .. }) => {
                    ZTransportFailureSnapshot::new(
                        diagram,
                        query,
                        catalog,
                        ZTransportFailureStatus::UnsupportedInput,
                        [Arc::from(error.to_string())],
                    )
                }
                Err(error) => Err(error.into()),
            }
        }
        ZTransportResult::NotCertified { reason, .. } => {
            match decide_z_transport_with_catalog(diagram, query, catalog, limits, ctx)? {
                ZTransportDecision::ProvenNonTransportable(obstruction) => {
                    let mut snapshot = ZTransportFailureSnapshot::new(
                        diagram,
                        query,
                        catalog,
                        ZTransportFailureStatus::ProofObstruction,
                        [Arc::from("z_transport.checked_trz_line11_obstruction")],
                    )?;
                    snapshot.z_obstruction = Some(obstruction.to_record());
                    return Ok(snapshot);
                }
                ZTransportDecision::MissingEvidence { missing } => {
                    return ZTransportFailureSnapshot::new(
                        diagram,
                        query,
                        catalog,
                        ZTransportFailureStatus::MissingEvidence,
                        [Arc::from(format!("z_transport.experimental_family: {missing:?}"))],
                    );
                }
                ZTransportDecision::Identified(derivation) => {
                    return match bind_z_transport_catalog(
                        diagram,
                        derivation.query(),
                        &derivation,
                        catalog,
                    ) {
                        Ok(_) => Err(ZTransportPlanningError::Invalid(
                            "catalog-aware TRz decision found an available formula; prepare from the decided derivation".into(),
                        )),
                        Err(error) => ZTransportFailureSnapshot::new(
                            diagram,
                            query,
                            catalog,
                            ZTransportFailureStatus::MissingEvidence,
                            [Arc::from(error.to_string())],
                        ),
                    };
                }
                ZTransportDecision::NotCertified { .. } => {}
            }
            if let Err(family) = validate_z_experiment_family(diagram, query, catalog) {
                let status = match family {
                    antecedent_identify::ZExperimentFamilyError::UnsupportedDomain { .. } => {
                        ZTransportFailureStatus::UnsupportedInput
                    }
                    antecedent_identify::ZExperimentFamilyError::FamilyExceedsBudget { .. } => {
                        ZTransportFailureStatus::ExhaustedComputation
                    }
                    antecedent_identify::ZExperimentFamilyError::MissingJointLaw { .. } => {
                        ZTransportFailureStatus::MissingEvidence
                    }
                };
                return ZTransportFailureSnapshot::new(
                    diagram,
                    query,
                    catalog,
                    status,
                    [Arc::from(format!("z_transport.experimental_family: {family:?}"))],
                );
            }
            if !target_joint_observational_law(diagram, query, catalog) {
                return ZTransportFailureSnapshot::new(
                    diagram,
                    query,
                    catalog,
                    ZTransportFailureStatus::MissingEvidence,
                    [Arc::from("z_transport.target_joint_observational_law_required")],
                );
            }
            let classical = ClassicalTransportQuery {
                outcomes: Arc::clone(&query.outcomes),
                treatments: Arc::clone(&query.treatments),
                source: Arc::clone(&query.source),
                target: Arc::clone(&query.target),
            };
            if let ClassicalTransportResult::ProvenNonTransportable(witness) =
                identify_classical_transport(diagram, &classical, limits, ctx)?
            {
                let mut snapshot = ZTransportFailureSnapshot::new(
                    diagram,
                    query,
                    catalog,
                    ZTransportFailureStatus::ProofObstruction,
                    [Arc::from("z_transport.checked_stronger_family_s_hedge")],
                )?;
                snapshot.obstruction = Some(witness.to_record());
                return Ok(snapshot);
            }
            ZTransportFailureSnapshot::new(
                diagram,
                query,
                catalog,
                ZTransportFailureStatus::UnresolvedIdentification,
                [Arc::from(reason)],
            )
        }
    }
}
