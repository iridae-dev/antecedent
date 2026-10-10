//! Actual finite arrived-study estimation through the original checked transport evaluator.
//! The structural proposal receipt remains hypothetical; this result is empirical point only.
//! SPDX-License-Identifier: MIT OR Apache-2.0
use antecedent_core::{EvidenceCatalog, ExecutionContext, VariableId, reason_code};
use antecedent_design::design_ranking_artifact::{
    ConsumeExpectation, DesignRankingArtifactWire, consume_wire,
};
use antecedent_design::proposal_receipt::{
    Arrival, ArrivalVerdict, ArrivedEvidence, ProposalBundle, on_arrival,
};
use antecedent_design::{
    DurableStudyCandidate, RepairConsumeLimits, RepairFamilyRef, RepairReportArtifact,
    TransportRepairFamily, ZTransportFailureSnapshot, ZTransportRepairFamily,
};
use antecedent_expr::execution_counts::{StaticWorkCounts, count_static_work};
use antecedent_expr::{
    Assignment, ExactEvaluationLimits, ExactEvaluationPlan, ExactTransportData, LawTolerance,
};
use antecedent_graph::SelectionDiagram;
use antecedent_identify::{
    CatalogTransportResult, ClassicalTransportQuery, SidLimits, bind_z_transport_catalog,
    identify_catalog_transport, verify_z_transport_derivation,
};
use antecedent_io::convert::{from_cbor, to_cbor};
use antecedent_io::exact_law_wire::ExactLawWire;
use antecedent_io::expr_wire::{ExprArenaWire, expr_arena_to_wire};
use antecedent_io::mz_transport_artifact::MzTransportPointWire;
use antecedent_io::query_wire::ValueWire;
use antecedent_io::transport_catalog_wire::EvidenceCatalogWire;
use antecedent_io::{IoError, admg_from_wire};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, sync::Arc};

/// Maximum self-contained input/result body, checked before decoding.
pub const MAX_ARRIVAL_BYTES: usize = 32 * 1024 * 1024;
/// Actual arrived-study input, retaining the original proposal-producing artifacts.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArrivalRequest {
    /// Original independently consumed repair report.
    pub repair_artifact: Vec<u8>,
    /// Original independently consumed study ranking.
    pub ranking_artifact: Vec<u8>,
    /// Original ranked candidate semantic identity.
    pub candidate_id: String,
    /// Full original catalog plus the actual one delivered study regime and binding.
    pub catalog: EvidenceCatalogWire,
    /// Unchanged original source/target providers needed by the checked formula.
    pub base_laws: Vec<ExactLawWire>,
    /// Empirical cell-count tables for the delivered study's disjoint intervention worlds.
    pub arrived_laws: Vec<ExactLawWire>,
    /// Concrete original-query treatment assignment.
    pub assignment: Vec<(u32, ValueWire)>,
    /// Finite evaluation operation bound.
    pub operations: usize,
    /// Finite evaluation recursion bound.
    pub depth: usize,
    /// Finite support row bound.
    pub support_rows: usize,
}
/// Actual successful numerical engine work, measured in the engine itself.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArrivalWork {
    /// Original expression compilation count.
    pub program_compilations: u64,
    /// Original numerical provider construction count.
    pub provider_bindings: u64,
    /// Actual uncached factor evaluations.
    pub factor_evaluations: u64,
    /// Original distribution/mean integrations.
    pub integrations: u64,
    /// Actual factor-provider calls.
    pub provider_calls: u64,
}
impl From<StaticWorkCounts> for ArrivalWork {
    fn from(c: StaticWorkCounts) -> Self {
        Self {
            program_compilations: c.program_compilations,
            provider_bindings: c.provider_bindings,
            factor_evaluations: c.factor_evaluations,
            integrations: c.integrations,
            provider_calls: c.provider_calls,
        }
    }
}
/// Independently replayable arrived-study point result. It does not validate an interval.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArrivalPoint {
    /// Original bundle identity reconstructed from the retained artifacts.
    pub proposal_identity: String,
    /// Original candidate identity.
    pub candidate_id: String,
    /// Delivered snapshot identity.
    pub snapshot: String,
    /// Actual observed count total, reconciled with the original planned sample size.
    pub sample_size: u64,
    /// Original theorem record serialized without a caller-issued certificate.
    pub proof: Vec<u8>,
    /// Actual checked provider-bound original expression.
    pub program: ExprArenaWire,
    /// Checked root in the original expression.
    pub root: u32,
    /// Full evaluated atom probabilities.
    pub point: MzTransportPointWire,
    /// Scalar outcome mean, absent when the query has joint outcomes.
    pub mean: Option<f64>,
    /// Always empirical_point_only.
    pub inference: String,
    /// Always unmeasured.
    pub calibration: String,
    /// Engine-observed successful numerical operations.
    pub work: ArrivalWork,
}
/// Self-contained request and reproduced scientific result, sealed together.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArrivalArtifact {
    /// Current schema version.
    pub version: u32,
    /// Actual original artifacts and raw numerical providers.
    pub request: ArrivalRequest,
    /// Original-engine scientific output.
    pub result: ArrivalPoint,
    /// BLAKE3 digest of request and result, excluding this field.
    pub identity: String,
}
fn refused(detail: &'static str) -> IoError {
    IoError::Refused { code: reason_code!("invalid_argument"), message: detail.into() }
}
fn unsupported(detail: &'static str) -> IoError {
    IoError::Refused { code: reason_code!("route_not_supported"), message: detail.into() }
}
fn native_error(e: impl std::fmt::Display) -> IoError {
    IoError::Refused {
        code: reason_code!("invalid_argument"),
        message: format!("proposal_arrival.original_consumer_refused: {e}"),
    }
}
fn bounds(r: &ArrivalRequest) -> Result<(), IoError> {
    if r.repair_artifact.len() > 8 * 1024 * 1024
        || r.ranking_artifact.len() > 8 * 1024 * 1024
        || r.base_laws.len() > 64
        || r.arrived_laws.is_empty()
        || r.arrived_laws.len() > 64
        || r.catalog.regimes.len() > 128
        || r.catalog.environments.len() > 64
        || r.catalog.bindings.len() > 128
        || r.assignment.len() > 12
        || r.operations == 0
        || r.operations > 1_000_000
        || r.depth == 0
        || r.depth > 256
        || r.support_rows == 0
        || r.support_rows > 65_536
        || r.base_laws.iter().chain(&r.arrived_laws).any(|l| {
            l.axes.len() > 12
                || l.probabilities.len() > 65_536
                || l.empirical_counts.as_ref().is_some_and(|c| c.len() > 65_536)
                || l.axes.iter().any(|(_, v)| v.len() > 65_536)
        })
    {
        return Err(unsupported("proposal_arrival.bounds_exceeded"));
    }
    Ok(())
}
fn ids(raw: &[u32]) -> Arc<[VariableId]> {
    raw.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>().into()
}
enum Family {
    Transport(Box<TransportRepairFamily>),
    Z(Box<ZTransportRepairFamily>),
}
impl Family {
    fn reference(&self) -> RepairFamilyRef<'_> {
        match self {
            Self::Transport(f) => RepairFamilyRef::Transport(f),
            Self::Z(f) => RepairFamilyRef::ZTransport(f),
        }
    }
    fn base(&self) -> &EvidenceCatalog {
        match self {
            Self::Transport(f) => f.base_catalog(),
            Self::Z(f) => f.snapshot().catalog(),
        }
    }
}
fn family(a: &RepairReportArtifact, ctx: &ExecutionContext) -> Result<Family, IoError> {
    match (a.premises.family.as_str(), &a.premises.transport, &a.premises.z_transport) {
        ("transport", Some(t), None) => {
            let graph = admg_from_wire(&t.graph)?;
            let diagram = SelectionDiagram::try_new(graph, ids(&t.selection_targets).to_vec())
                .map_err(native_error)?;
            let query = ClassicalTransportQuery {
                outcomes: ids(&t.outcomes),
                treatments: ids(&t.treatments),
                source: t.source.as_str().into(),
                target: t.target.as_str().into(),
            };
            let catalog = a
                .data
                .catalog
                .as_ref()
                .ok_or_else(|| refused("proposal_arrival.base_catalog_missing"))?
                .to_catalog()?;
            Ok(Family::Transport(Box::new(
                TransportRepairFamily::try_new(
                    diagram,
                    query,
                    catalog,
                    SidLimits { steps: t.sid_steps, depth: t.sid_depth },
                    ctx,
                )
                .map_err(native_error)?,
            )))
        }
        ("z_transport", None, Some(z)) => {
            let limits = SidLimits { steps: z.sid_steps, depth: z.sid_depth };
            let snapshot = ZTransportFailureSnapshot::from_wire(&z.snapshot, limits, ctx)
                .map_err(native_error)?;
            Ok(Family::Z(Box::new(
                ZTransportRepairFamily::try_new(snapshot, limits).map_err(native_error)?,
            )))
        }
        _ => Err(unsupported("proposal_arrival.family_not_supported")),
    }
}
fn validate_base_providers(r: &ArrivalRequest, f: &Family) -> Result<(), IoError> {
    for l in &r.base_laws {
        let regime = f
            .base()
            .regimes
            .iter()
            .find(|x| x.id.raw() == l.regime && x.population.as_ref() == l.population)
            .ok_or_else(|| refused("proposal_arrival.base_provider_mismatch"))?;
        let b = f
            .base()
            .bindings
            .iter()
            .find(|b| b.regime == regime.id)
            .ok_or_else(|| refused("proposal_arrival.base_provider_mismatch"))?;
        if b.snapshot_identity.as_ref() != l.snapshot {
            return Err(refused("proposal_arrival.base_provider_mismatch"));
        }
    }
    Ok(())
}
fn validate_delivery(
    r: &ArrivalRequest,
    f: &Family,
    candidate: &DurableStudyCandidate,
) -> Result<(EvidenceCatalog, u64, String, Vec<antecedent_core::EvidenceRegime>), IoError> {
    let base = EvidenceCatalogWire::from_catalog(f.base());
    if base.environments != r.catalog.environments
        || base.target_sampling != r.catalog.target_sampling
        || base.regimes.iter().any(|x| !r.catalog.regimes.contains(x))
        || base.bindings.iter().any(|x| !r.catalog.bindings.contains(x))
        || r.catalog.bindings.len() != base.bindings.len() + 1
    {
        return Err(refused("proposal_arrival.base_catalog_changed"));
    }
    let catalog = r.catalog.to_catalog()?;
    let delivered: Vec<_> = catalog
        .regimes
        .iter()
        .filter(|x| !f.base().regimes.iter().any(|b| b.id == x.id))
        .cloned()
        .collect();
    if delivered.len() != 1
        || catalog.regimes.len() != f.base().regimes.len() + 1
        || candidate.unit_rules.cluster.is_some()
        || candidate.unit_rules.whole_cluster_sampling
        || !candidate.joint_measurement
    {
        return Err(unsupported("proposal_arrival.independent_joint_study_required"));
    }
    let regime = &delivered[0];
    let binding = catalog
        .bindings
        .iter()
        .find(|b| b.regime == regime.id)
        .ok_or_else(|| refused("proposal_arrival.binding_missing"))?;
    if regime.evidence_kind != antecedent_core::EvidenceKind::Available
        || !regime.conditioned_on.is_empty()
        || regime.selection != antecedent_core::SamplingSelection::Population
        || regime.origin != antecedent_core::LawOrigin::Measured
        || regime.distribution != antecedent_core::DistributionAvailability::Joint
        || binding.sampling != antecedent_core::SamplingDesign::Independent
        || binding.weights.is_some()
        || binding.dependence != antecedent_core::DependenceGroup::IndependentStudies
    {
        return Err(unsupported("proposal_arrival.independent_joint_study_required"));
    }
    let mut total = 0u64;
    let measured: BTreeSet<_> = regime.measured.iter().map(|v| v.raw()).collect();
    let interventions: BTreeSet<_> = regime.interventions.iter().map(|v| v.raw()).collect();
    for l in &r.arrived_laws {
        let counts = l
            .empirical_counts
            .as_ref()
            .ok_or_else(|| refused("proposal_arrival.raw_counts_required"))?;
        if l.population != regime.population.as_ref()
            || l.regime != regime.id.raw()
            || l.snapshot != binding.snapshot_identity.as_ref()
            || l.origin != "empirical_plugin"
            || l.axes.iter().map(|(v, _)| *v).collect::<BTreeSet<_>>() != measured
            || l.interventions.iter().map(|(v, _)| *v).collect::<BTreeSet<_>>() != interventions
        {
            return Err(refused("proposal_arrival.provider_binding_mismatch"));
        }
        for a in regime.intervention_values.iter() {
            if !l
                .interventions
                .iter()
                .any(|(v, value)| *v == a.variable.raw() && value.to_value() == a.value)
            {
                return Err(refused("proposal_arrival.provider_binding_mismatch"));
            }
        }
        for count in counts {
            total = total
                .checked_add(*count)
                .ok_or_else(|| refused("proposal_arrival.sample_size_mismatch"))?;
        }
    }
    if total != candidate.sample_size {
        return Err(refused("proposal_arrival.sample_size_mismatch"));
    }
    validate_base_providers(r, f)?;
    let snapshot = binding.snapshot_identity.to_string();
    Ok((catalog, total, snapshot, delivered))
}
struct BoundArrivalProgram {
    arena: antecedent_expr::CausalExprArena,
    root: antecedent_expr::ExprId,
    outcomes: Arc<[VariableId]>,
    treatments: Arc<[VariableId]>,
    proof: Vec<u8>,
    data: ExactTransportData,
}
fn bind_arrival_program(
    r: &ArrivalRequest,
    family: &Family,
    catalog: &EvidenceCatalog,
    ctx: &ExecutionContext,
) -> Result<BoundArrivalProgram, IoError> {
    let laws = r
        .base_laws
        .iter()
        .chain(&r.arrived_laws)
        .map(ExactLawWire::to_law)
        .collect::<Result<Vec<_>, _>>()?;
    let data = ExactTransportData::try_new(laws, r.support_rows).map_err(native_error)?;
    let (arena, root, outcomes, treatments, proof, data) = match family {
        Family::Transport(f) => {
            let CatalogTransportResult::Identified(bound) =
                identify_catalog_transport(f.diagram(), f.query(), catalog, f.sid_limits(), ctx)
                    .map_err(native_error)?
            else {
                return Err(refused("proposal_arrival.proof_not_bound"));
            };
            (
                bound.arena().clone(),
                bound.root(),
                f.query().outcomes.clone(),
                f.query().treatments.clone(),
                to_cbor(&bound.derivation().to_record())?,
                data,
            )
        }
        Family::Z(f) => {
            let s = f.snapshot();
            let derivation =
                s.derivation().ok_or_else(|| refused("proposal_arrival.proof_not_bound"))?;
            verify_z_transport_derivation(s.diagram(), s.query(), derivation, f.sid_limits(), ctx)
                .map_err(native_error)?;
            let bound = bind_z_transport_catalog(s.diagram(), s.query(), derivation, catalog)
                .map_err(native_error)?;
            let data = data.with_world_bound_leaves(bound.cited_regimes().to_vec());
            (
                bound.arena().clone(),
                bound.root(),
                s.query().outcomes.clone(),
                s.query().treatments.clone(),
                to_cbor(&derivation.to_record())?,
                data,
            )
        }
    };
    Ok(BoundArrivalProgram { arena, root, outcomes, treatments, proof, data })
}
/// Execute the original checked transport formula on actual delivered counts.
/// # Errors
/// Original consumer/proof failures, changed premises, sample/provider mismatch, support and bounds.
pub fn execute(r: &ArrivalRequest, ctx: &ExecutionContext) -> Result<ArrivalPoint, IoError> {
    bounds(r)?;
    if ctx.cancellation.is_cancelled() {
        return Err(unsupported("proposal_arrival.cancelled"));
    }
    let repair = RepairReportArtifact::from_bytes(&r.repair_artifact).map_err(native_error)?;
    repair.consume(RepairConsumeLimits::default(), ctx).map_err(native_error)?;
    let ranking =
        DesignRankingArtifactWire::from_bytes(&r.ranking_artifact).map_err(native_error)?;
    consume_wire(&ranking, &ConsumeExpectation::default()).map_err(native_error)?;
    let bundle = ProposalBundle::build(&repair, &ranking).map_err(native_error)?;
    let receipt = bundle
        .proposal(&r.candidate_id)
        .ok_or_else(|| refused("proposal_arrival.candidate_missing"))?;
    let candidate = repair
        .candidates()
        .map_err(native_error)?
        .into_iter()
        .find(|c| c.semantic_id().as_ref() == r.candidate_id)
        .ok_or_else(|| refused("proposal_arrival.candidate_missing"))?;
    let family = family(&repair, ctx)?;
    let (catalog, total, snapshot, delivered) = validate_delivery(r, &family, &candidate)?;
    let arrival = Arrival {
        snapshot_id: snapshot.clone(),
        sample_size: total,
        evidence: ArrivedEvidence::CatalogDelta(delivered),
    };
    match on_arrival(receipt, family.reference(), &candidate, &arrival, ctx)
        .map_err(native_error)?
    {
        ArrivalVerdict::Verified { .. } => {}
        _ => return Err(refused("proposal_arrival.structural_arrival_refused")),
    }
    let (evaluated, counts) = count_static_work(|| {
        let BoundArrivalProgram { arena, root, outcomes, treatments, proof, data } =
            bind_arrival_program(r, &family, &catalog, ctx)?;
        let assigned: BTreeSet<_> = r.assignment.iter().map(|(v, _)| *v).collect();
        if assigned.len() != r.assignment.len()
            || assigned != treatments.iter().map(|v| v.raw()).collect()
        {
            return Err(refused("proposal_arrival.assignment_mismatch"));
        }
        let request = Assignment::from_pairs(
            r.assignment.iter().map(|(v, x)| (VariableId::from_raw(*v), x.to_value())),
        );
        let plan = ExactEvaluationPlan::compile(
            &arena,
            root,
            data,
            outcomes.clone(),
            request,
            ExactEvaluationLimits { operations: r.operations, depth: r.depth },
            LawTolerance::default(),
            ctx,
        )
        .map_err(native_error)?;
        let point = plan.evaluate(ctx).map_err(native_error)?;
        let mean = if outcomes.len() == 1 {
            Some(point.mean(outcomes[0]).map_err(native_error)?)
        } else {
            None
        };
        Ok((
            proof,
            expr_arena_to_wire(&arena)?,
            root.raw(),
            MzTransportPointWire::from_distribution(&point),
            mean,
        ))
    });
    let (proof, program, root, point, mean) = evaluated?;
    Ok(ArrivalPoint {
        proposal_identity: bundle.identity,
        candidate_id: r.candidate_id.clone(),
        snapshot,
        sample_size: total,
        proof,
        program,
        root,
        point,
        mean,
        inference: "empirical_point_only".into(),
        calibration: "unmeasured".into(),
        work: counts.into(),
    })
}
impl ArrivalArtifact {
    /// Produce a sealed original-engine arrived-study result.
    /// # Errors
    /// The original execution or bounded serialization refuses.
    pub fn produce(request: ArrivalRequest, ctx: &ExecutionContext) -> Result<Self, IoError> {
        let result = execute(&request, ctx)?;
        let mut artifact = Self { version: 1, request, result, identity: String::new() };
        artifact.identity = artifact.digest()?;
        Ok(artifact)
    }
    fn digest(&self) -> Result<String, IoError> {
        Ok(blake3::hash(&to_cbor(&(self.version, &self.request, &self.result))?)
            .to_hex()
            .to_string())
    }
    /// Export bounded replay input and complete scientific output.
    /// # Errors
    /// Oversized/invalid artifacts refuse.
    pub fn to_bytes(&self) -> Result<Vec<u8>, IoError> {
        bounds(&self.request)?;
        let bytes = to_cbor(self)?;
        if bytes.len() > MAX_ARRIVAL_BYTES {
            return Err(IoError::TooLarge);
        }
        Ok(bytes)
    }
    /// Independently consume every original artifact, proof and raw numerical provider.
    /// # Errors
    /// Changed original proposal identity, corrupted/resealed result or unsupported input refuses.
    pub fn consume(
        bytes: &[u8],
        expected_proposal: &str,
        ctx: &ExecutionContext,
    ) -> Result<Self, IoError> {
        if bytes.len() > MAX_ARRIVAL_BYTES {
            return Err(IoError::TooLarge);
        }
        let artifact: Self = from_cbor(bytes)?;
        bounds(&artifact.request)?;
        if artifact.version != 1 || artifact.identity != artifact.digest()? {
            return Err(refused("proposal_arrival.identity_mismatch"));
        }
        if artifact.result.proposal_identity != expected_proposal {
            return Err(refused("proposal_arrival.expected_proposal_mismatch"));
        }
        let replay = execute(&artifact.request, ctx)?;
        if to_cbor(&artifact.result)? != to_cbor(&replay)? {
            return Err(refused("proposal_arrival.replay_mismatch"));
        }
        Ok(artifact)
    }
}
