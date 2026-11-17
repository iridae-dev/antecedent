//! Exact path-specific transport on an explicitly time-ordered finite graph.
//!
//! The two-slice temporal sequence route and static counterfactual transport
//! identify different functionals. This additive route time-checks the
//! unrolled graph, constructs the two edge-intervention worlds, and asks the
//! finite selection-diagram identifier for each target-population outcome
//! probability. It subtracts only two identified probabilities evaluated on
//! the same exact regime laws. Sampling intervals and a general continuous
//! longitudinal functional are outside this point-only method.

use std::collections::BTreeSet;
use std::sync::Arc;

use antecedent_contract::{ClaimKind, DataClassification};
use antecedent_core::query::{
    CounterfactualEvent, CounterfactualEventQuery, CrossWorldQuery,
    ExogenousCoupling, WorldId,
};
use antecedent_core::{
    ByteEstimates, CanonicalWriter, EvidenceCatalog, ExecutionContext,
    PhysicalEffect, PhysicalOperator, QueryError, SemanticDigest, VariableId,
};
use antecedent_expr::ExactTransportData;
use antecedent_identify::{
    CounterfactualTransportProblem, CounterfactualTransportQuery,
    CounterfactualTransportSource,
};
use antecedent_io::IoError;
use antecedent_io::counterfactual_transport_artifact::CounterfactualTransportConsumeLimits;
use antecedent_io::counterfactual_transport_wire::CounterfactualTransportExpectedIdentity;
use antecedent_io::exact_law_wire::ExactLawWire;
use antecedent_io::transport_catalog_wire::EvidenceCatalogWire;
use serde::{Deserialize, Serialize};

use super::counterfactual_transport::{
    CounterfactualTransportConsumption, CounterfactualTransportOptions,
    CounterfactualTransportPreparation, CounterfactualTransportResult,
    PreparedCounterfactualTransport, consume_counterfactual_transport_artifact,
    prepare_counterfactual_transport,
};
use crate::planner::lowered::{
    ClaimDemand, Composition, DesignShape, EvidenceDecisionProgram,
    EvidenceRole, EvidenceSourceRef, FunctionalKind, IdentificationLayer,
    LoweredProgram, NonGraphicalIdentification, PhysicalLayer, ProblemTarget,
    ProgramCell, ProgramIdentityScope, ScientificProblem,
    StandaloneCompositionLoweredParts, UncertaintyReading,
};
use crate::planner::statistical::{Realization, StatisticalProgram};
use crate::planner::{PlanStage, PlanView};

const FEATURE: &str = "temporal_path_transport_v1";
const MAX_ARTIFACT_BYTES: usize = 80 * 1024 * 1024;

/// The time-indexed edge intervention whose target outcome probability is
/// compared with the baseline world. Variable ids are raw, and their time
/// coordinates must name every graph node in ascending variable-id order.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TemporalPathSpec {
    /// Treatment variable id.
    pub treatment: u32,
    /// Final outcome variable id.
    pub outcome: u32,
    /// Baseline treatment level bits.
    pub control_bits: u64,
    /// Active treatment level bits.
    pub active_bits: u64,
    /// Outcome event level bits.
    pub outcome_level_bits: u64,
    /// Every observed variable and its nonnegative time index.
    pub times: Vec<(u32, u32)>,
    /// Directed edges that read the active world; all other edges read baseline.
    pub active_edges: Vec<(u32, u32)>,
}

/// Refusal of temporal order, path identification, exact evaluation or replay.
#[derive(Debug, thiserror::Error)]
pub enum TemporalPathTransportError {
    /// The temporal path specification is outside this checked method.
    #[error("temporal path transport: {0}")]
    Invalid(&'static str),
    /// The cross-world event was malformed.
    #[error("temporal path transport query: {0}")]
    Query(#[from] QueryError),
    /// An exact law, identification or artifact step refused.
    #[error("temporal path transport: {0}")]
    Io(#[from] IoError),
    /// A finite two-model witness proves the requested event is not transportable.
    #[error("the declared evidence does not transport a path world")]
    ProvenNonTransportable,
}

/// Both exact selection-diagram derivations and the complete compiler program.
#[derive(Clone, Debug)]
pub struct CompiledTemporalPathTransport {
    spec: TemporalPathSpec,
    options: TemporalPathOptionsWire,
    data: ExactTransportData,
    active: PreparedCounterfactualTransport,
    baseline: PreparedCounterfactualTransport,
    lowered: Arc<LoweredProgram>,
}

/// Two target-population path-world probabilities and their exact contrast.
#[derive(Clone, Debug)]
pub struct TemporalPathTransportResult {
    /// Target outcome probability under the active edge intervention.
    pub active: CounterfactualTransportResult,
    /// Target outcome probability in the baseline world.
    pub baseline: CounterfactualTransportResult,
    /// `active - baseline`, exact given the supplied finite laws.
    pub contrast: f64,
    /// Six-layer identity binding graph, selection, time, paths and laws.
    pub lowered_program_identity: SemanticDigest,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct InnerIdentityWire {
    premises_digest: Option<String>,
    data_digest: Option<String>,
}

impl From<CounterfactualTransportExpectedIdentity> for InnerIdentityWire {
    fn from(value: CounterfactualTransportExpectedIdentity) -> Self {
        Self { premises_digest: value.premises_digest, data_digest: value.data_digest }
    }
}

impl From<InnerIdentityWire> for CounterfactualTransportExpectedIdentity {
    fn from(value: InnerIdentityWire) -> Self {
        Self { premises_digest: value.premises_digest, data_digest: value.data_digest }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct TemporalPathTransportWire {
    version: u32,
    required_feature: String,
    spec: TemporalPathSpec,
    options: TemporalPathOptionsWire,
    program_identity: [u8; 32],
    active_identity: InnerIdentityWire,
    active_artifact: Vec<u8>,
    baseline_identity: InnerIdentityWire,
    baseline_artifact: Vec<u8>,
    contrast_bits: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct TemporalPathOptionsWire {
    search_operations: usize,
    search_depth: usize,
    evaluation_operations: usize,
    evaluation_depth: usize,
    variable_names: Vec<String>,
}

impl From<&CounterfactualTransportOptions> for TemporalPathOptionsWire {
    fn from(value: &CounterfactualTransportOptions) -> Self {
        Self {
            search_operations: value.search.operations,
            search_depth: value.search.depth,
            evaluation_operations: value.evaluation.operations,
            evaluation_depth: value.evaluation.depth,
            variable_names: value.variable_names.clone(),
        }
    }
}

fn variables_and_edges(problem: &CounterfactualTransportProblem)
    -> Vec<(VariableId, VariableId)> {
    problem.directed().iter().map(|&(from, to)| {
        (problem.variables()[from], problem.variables()[to])
    }).collect()
}

fn queries(
    problem: &CounterfactualTransportProblem,
    target: Arc<str>, sources: Arc<[CounterfactualTransportSource]>,
    spec: &TemporalPathSpec,
) -> Result<(CounterfactualTransportQuery, CounterfactualTransportQuery), TemporalPathTransportError> {
    if spec.times.len() != problem.variable_count()
        || spec.times.iter().zip(problem.variables()).any(|(&(variable, _), id)| variable != id.raw())
        || spec.active_edges.is_empty()
    {
        return Err(TemporalPathTransportError::Invalid("time labels must cover every graph node and select a path"));
    }
    let times: Vec<u32> = spec.times.iter().map(|(_, time)| *time).collect();
    let Some(treatment) = problem.index_of(VariableId::from_raw(spec.treatment)) else {
        return Err(TemporalPathTransportError::Invalid("treatment is absent from the graph"));
    };
    let Some(outcome) = problem.index_of(VariableId::from_raw(spec.outcome)) else {
        return Err(TemporalPathTransportError::Invalid("outcome is absent from the graph"));
    };
    if !problem.levels(treatment).contains(&f64::from_bits(spec.control_bits))
        || !problem.levels(treatment).contains(&f64::from_bits(spec.active_bits))
        || !problem.levels(outcome).contains(&f64::from_bits(spec.outcome_level_bits))
    {
        return Err(TemporalPathTransportError::Invalid("treatment or outcome event level is outside the finite model"));
    }
    if times[treatment] >= times[outcome]
        || times[outcome] != *times.iter().max().unwrap_or(&0)
        || problem.directed().iter().any(|&(from, to)| times[from] >= times[to])
    {
        return Err(TemporalPathTransportError::Invalid(
            "all graph arrows must advance time and the outcome must be at the final time",
        ));
    }
    let edges = variables_and_edges(problem);
    let active_edges: Vec<_> = spec.active_edges.iter().map(|&(from, to)| {
        (VariableId::from_raw(from), VariableId::from_raw(to))
    }).collect();
    if active_edges.iter().any(|&(from, _)| from != VariableId::from_raw(spec.treatment))
        || active_edges.iter().copied().collect::<BTreeSet<_>>().len() != active_edges.len()
    {
        return Err(TemporalPathTransportError::Invalid(
            "active path edges must be distinct outgoing treatment edges",
        ));
    }
    let contrast = CrossWorldQuery::path_specific(
        VariableId::from_raw(spec.treatment), VariableId::from_raw(spec.outcome),
        f64::from_bits(spec.control_bits), f64::from_bits(spec.active_bits),
        &edges, &active_edges,
    )?;
    let worlds = contrast.worlds().to_vec();
    let active_event = CounterfactualEventQuery::new(
        worlds.clone(), ExogenousCoupling::SharedLatentExogenous,
        [CounterfactualEvent::new(WorldId::new(1), VariableId::from_raw(spec.outcome),
            f64::from_bits(spec.outcome_level_bits))?], [],
    )?;
    let baseline_event = CounterfactualEventQuery::new(
        worlds, ExogenousCoupling::SharedLatentExogenous,
        [CounterfactualEvent::new(WorldId::new(0), VariableId::from_raw(spec.outcome),
            f64::from_bits(spec.outcome_level_bits))?], [],
    )?;
    let make = |event| CounterfactualTransportQuery {
        event, target: Arc::clone(&target), sources: Arc::clone(&sources),
    }.canonical();
    Ok((make(active_event), make(baseline_event)))
}

fn lower(
    problem: &CounterfactualTransportProblem,
    active: &CounterfactualTransportQuery,
    baseline: &CounterfactualTransportQuery,
    catalog: &EvidenceCatalog,
    data: &ExactTransportData,
    spec: &TemporalPathSpec,
    options: &TemporalPathOptionsWire,
) -> Result<Arc<LoweredProgram>, TemporalPathTransportError> {
    let mut law_bytes: Vec<Vec<u8>> = data.laws().iter().map(|law| {
        antecedent_io::to_cbor(&ExactLawWire::from_law(law))
    }).collect::<Result<_, _>>()?;
    law_bytes.sort();
    let mut source_writer = CanonicalWriter::new("antecedent.temporal_path_transport.exact_laws.v1");
    source_writer.word(law_bytes.len() as u64);
    for bytes in &law_bytes { source_writer.bytes(bytes); }
    let source_digest = source_writer.finish();
    let catalog_bytes = antecedent_io::to_cbor(&EvidenceCatalogWire::from_catalog(catalog))?;
    let spec_bytes = antecedent_io::to_cbor(spec)?;
    let options_bytes = antecedent_io::to_cbor(options)?;
    let mut writer = CanonicalWriter::new("antecedent.temporal_path_transport.question.v1");
    writer.text(&problem.canonical_text()).text(&active.canonical_text())
        .text(&baseline.canonical_text()).bytes(&catalog_bytes)
        .bytes(&spec_bytes).bytes(&options_bytes).bytes(&source_digest);
    let query = writer.finish();
    let design = DesignShape::quoted(data.laws().len() as u64, problem.variable_count() as u64, true);
    let scientific = ScientificProblem {
        runtime_query: None, identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::PathSpecific,
            outcome: Some(VariableId::from_raw(spec.outcome)), query,
        },
        plan_id: Arc::from("temporal_path_transport.v1"),
        data: DataClassification::Tabular,
        graph_class: "time_ordered_selection_admg",
        query_variables: Arc::from(problem.variables().to_vec()),
        source_digest: Some(source_digest),
        support_coordinate: None, supported: false, support_status: None,
    };
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SharedEvidenceContract { contract: query },
    ));
    let statistical = StatisticalProgram {
        identifier: None, estimator: None,
        realization: Realization::ExactPoint {
            evaluator: Arc::from("finite_counterfactual_transport.path_contrast"),
        },
        models: Arc::from([]), folds: None, resampling: None,
        posterior: None, design, operators: Arc::from([]),
    };
    let evidence = EvidenceDecisionProgram {
        sources: Arc::from([EvidenceSourceRef {
            role: EvidenceRole::TransportSource, digest: Some(source_digest),
        }]),
        composition: Composition::Chained { stages: 2 },
        decision: None, planning: None, chain: None,
    };
    let physical = PhysicalLayer {
        operators: Arc::from([
            PhysicalOperator::declare(PhysicalEffect::Read)
                .estimates(ByteEstimates::absent().with_output(
                    law_bytes.iter().map(|bytes| bytes.len() as u64).sum(),
                )).build().map_err(|_| TemporalPathTransportError::Invalid("exact law read cannot be lowered"))?,
            PhysicalOperator::declare(PhysicalEffect::Intervene)
                .build().map_err(|_| TemporalPathTransportError::Invalid("path intervention cannot be lowered"))?,
            PhysicalOperator::declare(PhysicalEffect::Transport)
                .build().map_err(|_| TemporalPathTransportError::Invalid("path transport cannot be lowered"))?,
        ]),
        design,
    };
    Ok(Arc::new(StandaloneCompositionLoweredParts {
        cell: ProgramCell::TemporalPathTransport,
        scientific, identification, statistical, evidence, physical,
        claim: ClaimDemand { uncertainty: UncertaintyReading::None,
            kind: Some(ClaimKind::Point) },
    }.finish()))
}

/// Compile a target-population path-specific event contrast after checking
/// time order and proving each world probability from the declared selection
/// diagram and source/target regime catalog. Exact laws are bound to the
/// compiler identity before either probability is evaluated.
///
/// # Errors
/// Invalid temporal order or edge routes, a missing identification proof,
/// proven nontransportability or a resource refusal.
#[allow(clippy::too_many_arguments, reason = "one declared time/path/selection problem with its exact evidence")]
pub fn compile_temporal_path_transport(
    problem: CounterfactualTransportProblem,
    target: Arc<str>, sources: Arc<[CounterfactualTransportSource]>,
    catalog: EvidenceCatalog, data: ExactTransportData,
    spec: TemporalPathSpec, options: CounterfactualTransportOptions,
    ctx: &ExecutionContext,
) -> Result<CompiledTemporalPathTransport, TemporalPathTransportError> {
    let catalog = catalog.canonicalized()?;
    let (active_query, baseline_query) = queries(&problem, target, sources, &spec)?;
    let options_wire = TemporalPathOptionsWire::from(&options);
    let lowered = lower(&problem, &active_query, &baseline_query, &catalog, &data, &spec, &options_wire)?;
    let scoped = ctx.enter_lowered_dispatch(lowered.identity());
    let prepare = |query| {
        prepare_counterfactual_transport(&problem, query, &catalog, None, &options, &scoped)
    };
    let active = match prepare(&active_query)? {
        CounterfactualTransportPreparation::Identified(value) => *value,
        CounterfactualTransportPreparation::ProvenNonTransportable(_) => {
            return Err(TemporalPathTransportError::ProvenNonTransportable);
        }
    };
    let baseline = match prepare(&baseline_query)? {
        CounterfactualTransportPreparation::Identified(value) => *value,
        CounterfactualTransportPreparation::ProvenNonTransportable(_) => {
            return Err(TemporalPathTransportError::ProvenNonTransportable);
        }
    };
    Ok(CompiledTemporalPathTransport { spec, options: options_wire, data, active, baseline, lowered })
}

impl CompiledTemporalPathTransport {
    /// All six compiler layers of the time/path/selection/law program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram { &self.lowered }

    /// Inspect the compiled program without evaluating its exact laws.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }

    /// Evaluate both identified target-population probabilities on the same
    /// exact laws and subtract them. There is no sampling interval.
    ///
    /// # Errors
    /// A cited regime law is missing, contradictory or outside exact support.
    pub fn execute(
        &self, ctx: &ExecutionContext,
    ) -> Result<TemporalPathTransportResult, TemporalPathTransportError> {
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        let active = self.active.evaluate(&self.data, &scoped)?;
        let baseline = self.baseline.evaluate(&self.data, &scoped)?;
        let contrast = active.probability() - baseline.probability();
        if !contrast.is_finite() {
            return Err(TemporalPathTransportError::Invalid("the path contrast is nonfinite"));
        }
        Ok(TemporalPathTransportResult {
            active, baseline, contrast,
            lowered_program_identity: self.lowered.identity(),
        })
    }

    /// Seal the two independently replayable identification/evaluation
    /// artifacts and the complete program identity after a fresh comparison.
    ///
    /// # Errors
    /// Changed exact result or artifact encoding failure.
    pub fn export(
        &self, result: &TemporalPathTransportResult, ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, TemporalPathTransportError> {
        let live = self.execute(ctx)?;
        if result.lowered_program_identity != self.lowered.identity()
            || result.active.probability().to_bits() != live.active.probability().to_bits()
            || result.baseline.probability().to_bits() != live.baseline.probability().to_bits()
            || result.contrast.to_bits() != live.contrast.to_bits()
        {
            return Err(TemporalPathTransportError::Invalid("path result differs on replay"));
        }
        let wire = TemporalPathTransportWire {
            version: 1, required_feature: FEATURE.into(), spec: self.spec.clone(),
            options: self.options.clone(),
            program_identity: *self.lowered.identity().as_bytes(),
            active_identity: live.active.identity()?.into(),
            active_artifact: live.active.export_artifact()?,
            baseline_identity: live.baseline.identity()?.into(),
            baseline_artifact: live.baseline.export_artifact()?,
            contrast_bits: live.contrast.to_bits(),
        };
        let bytes = antecedent_io::to_cbor(&wire)?;
        if bytes.len() > MAX_ARTIFACT_BYTES {
            return Err(TemporalPathTransportError::Invalid("path artifact exceeds its bound"));
        }
        Ok(bytes)
    }
}

/// Independently replay each exact transport derivation and reconstruct the
/// outer six-layer program from the verified graph, selection catalog, laws,
/// time labels and edge intervention.
///
/// # Errors
/// A foreign feature, altered path, changed law/derivation or mismatched point.
pub fn consume_temporal_path_transport(
    bytes: &[u8], expected_program: SemanticDigest, ctx: &ExecutionContext,
) -> Result<TemporalPathTransportResult, TemporalPathTransportError> {
    if bytes.len() > MAX_ARTIFACT_BYTES {
        return Err(TemporalPathTransportError::Invalid("path artifact exceeds its bound"));
    }
    let wire: TemporalPathTransportWire = antecedent_io::from_cbor(bytes)?;
    if wire.version != 1 || wire.required_feature != FEATURE
        || wire.program_identity != *expected_program.as_bytes()
    {
        return Err(TemporalPathTransportError::Invalid("path feature or program differs"));
    }
    let identified = |bytes: &[u8], identity: InnerIdentityWire| {
        match consume_counterfactual_transport_artifact(
            bytes, &identity.into(), CounterfactualTransportConsumeLimits::default(), ctx,
        )? {
            CounterfactualTransportConsumption::Identified(result) => Ok(*result),
            CounterfactualTransportConsumption::ProvenNonTransportable(_) => {
                Err(TemporalPathTransportError::ProvenNonTransportable)
            }
        }
    };
    let active = identified(&wire.active_artifact, wire.active_identity)?;
    let baseline = identified(&wire.baseline_artifact, wire.baseline_identity)?;
    let derivation = active.derivation();
    let problem = &derivation.problem;
    let catalog = &derivation.catalog;
    let data = active.data();
    let (active_query, baseline_query) = queries(
        problem, Arc::clone(&derivation.query.target),
        Arc::clone(&derivation.query.sources), &wire.spec,
    )?;
    if derivation.query.canonical_text() != active_query.canonical_text()
        || baseline.derivation().query.canonical_text() != baseline_query.canonical_text()
        || active.identity()?.data_digest != baseline.identity()?.data_digest
    {
        return Err(TemporalPathTransportError::Invalid("path worlds or source laws differ"));
    }
    let lowered = lower(problem, &active_query, &baseline_query,
        catalog, data, &wire.spec, &wire.options)?;
    let contrast = active.probability() - baseline.probability();
    if lowered.identity() != expected_program
        || contrast.to_bits() != wire.contrast_bits
    {
        return Err(TemporalPathTransportError::Invalid("recompiled path contrast differs"));
    }
    Ok(TemporalPathTransportResult {
        active, baseline, contrast, lowered_program_identity: lowered.identity(),
    })
}
