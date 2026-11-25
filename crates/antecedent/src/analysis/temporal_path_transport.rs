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
    CounterfactualEvent, CounterfactualEventQuery, CrossWorldQuery, ExogenousCoupling, WorldId,
};
use antecedent_core::{
    ByteEstimates, CalibrationBasis, CanonicalWriter, EvidenceCatalog, ExecutionContext,
    PhysicalEffect, PhysicalOperator, QueryError, ScoreFrequentist, ScoreModel, SearchLimits,
    SemanticDigest, VariableId,
};
use antecedent_estimate::EstimationError;
use antecedent_estimate::counterfactual_transport_sampled::{
    SampledRegimeCounts, SampledTransportContrastEstimate, SampledTransportRequest,
    estimate_sampled_counterfactual_transport_contrast,
};
use antecedent_expr::{ExactEvaluationLimits, ExactTransportData};
use antecedent_identify::{
    CounterfactualTransportProblem, CounterfactualTransportQuery, CounterfactualTransportSource,
};
use antecedent_io::IoError;
use antecedent_io::counterfactual_transport_artifact::CounterfactualTransportConsumeLimits;
use antecedent_io::counterfactual_transport_wire::CounterfactualTransportExpectedIdentity;
use antecedent_io::counterfactual_transport_wire::{
    CounterfactualTransportProblemWire, CounterfactualTransportQueryWire,
};
use antecedent_io::exact_law_wire::ExactLawWire;
use antecedent_io::sampled_counterfactual_transport_artifact::{
    SampledCountTableWire, SampledLimitsWire, SampledOverlapWire, SampledUnseenRuleWire,
};
use antecedent_io::transport_catalog_wire::EvidenceCatalogWire;
use serde::{Deserialize, Serialize};

use super::counterfactual_transport::{
    CounterfactualTransportConsumption, CounterfactualTransportOptions,
    CounterfactualTransportPreparation, CounterfactualTransportResult,
    PreparedCounterfactualTransport, consume_counterfactual_transport_artifact,
    prepare_counterfactual_transport,
};
use super::program_handle::{CompiledProgramHandle, gate_program};
use crate::planner::PlanView;
use crate::planner::lowered::{
    ClaimDemand, Composition, DesignShape, EvidenceDecisionProgram, EvidenceRole,
    EvidenceSourceRef, FunctionalKind, IdentificationLayer, LoweredProgram,
    NonGraphicalIdentification, PhysicalLayer, ProblemTarget, ProgramCell, ProgramIdentityScope,
    ScientificProblem, StandaloneCompositionLoweredParts, UncertaintyReading,
};
use crate::planner::statistical::{Realization, StatisticalProgram};

const FEATURE: &str = "temporal_path_transport_v1";
const SAMPLED_FEATURE: &str = "sampled_temporal_path_transport_v1";
const MAX_ARTIFACT_BYTES: usize = 80 * 1024 * 1024;

/// Whether `execute` passes the support gate of the unit the route is bound to
/// (`parity/compiled_units.toml`). It stays off until that unit's record has earned its support,
/// so no valid use refuses before then; turning it on makes a release build refuse the route
/// while the unit is unsupported and a development build record what it would refuse.
const GATE_ON_UNIT_SUPPORT: bool = false;

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
    /// Sampled count estimation or paired uncertainty refused.
    #[error("temporal path sampled estimate: {0}")]
    Estimate(#[from] EstimationError),
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

impl TemporalPathOptionsWire {
    fn to_options(&self) -> CounterfactualTransportOptions {
        CounterfactualTransportOptions {
            search: SearchLimits { operations: self.search_operations, depth: self.search_depth },
            evaluation: ExactEvaluationLimits {
                operations: self.evaluation_operations,
                depth: self.evaluation_depth,
            },
            variable_names: self.variable_names.clone(),
        }
    }
}

fn variables_and_edges(problem: &CounterfactualTransportProblem) -> Vec<(VariableId, VariableId)> {
    problem
        .directed()
        .iter()
        .map(|&(from, to)| (problem.variables()[from], problem.variables()[to]))
        .collect()
}

fn queries(
    problem: &CounterfactualTransportProblem,
    target: &Arc<str>,
    sources: &Arc<[CounterfactualTransportSource]>,
    spec: &TemporalPathSpec,
) -> Result<(CounterfactualTransportQuery, CounterfactualTransportQuery), TemporalPathTransportError>
{
    if spec.times.len() != problem.variable_count()
        || spec
            .times
            .iter()
            .zip(problem.variables())
            .any(|(&(variable, _), id)| variable != id.raw())
        || spec.active_edges.is_empty()
    {
        return Err(TemporalPathTransportError::Invalid(
            "time labels must cover every graph node and select a path",
        ));
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
        return Err(TemporalPathTransportError::Invalid(
            "treatment or outcome event level is outside the finite model",
        ));
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
    let active_edges: Vec<_> = spec
        .active_edges
        .iter()
        .map(|&(from, to)| (VariableId::from_raw(from), VariableId::from_raw(to)))
        .collect();
    if active_edges.iter().any(|&(from, _)| from != VariableId::from_raw(spec.treatment))
        || active_edges.iter().copied().collect::<BTreeSet<_>>().len() != active_edges.len()
    {
        return Err(TemporalPathTransportError::Invalid(
            "active path edges must be distinct outgoing treatment edges",
        ));
    }
    let contrast = CrossWorldQuery::path_specific(
        VariableId::from_raw(spec.treatment),
        VariableId::from_raw(spec.outcome),
        f64::from_bits(spec.control_bits),
        f64::from_bits(spec.active_bits),
        &edges,
        &active_edges,
    )?;
    let worlds = contrast.worlds().to_vec();
    let active_event = CounterfactualEventQuery::new(
        worlds.clone(),
        ExogenousCoupling::SharedLatentExogenous,
        [CounterfactualEvent::new(
            WorldId::new(1),
            VariableId::from_raw(spec.outcome),
            f64::from_bits(spec.outcome_level_bits),
        )?],
        [],
    )?;
    let baseline_event = CounterfactualEventQuery::new(
        worlds,
        ExogenousCoupling::SharedLatentExogenous,
        [CounterfactualEvent::new(
            WorldId::new(0),
            VariableId::from_raw(spec.outcome),
            f64::from_bits(spec.outcome_level_bits),
        )?],
        [],
    )?;
    let make = |event| {
        CounterfactualTransportQuery {
            event,
            target: Arc::clone(target),
            sources: Arc::clone(sources),
        }
        .canonical()
    };
    Ok((make(active_event), make(baseline_event)))
}

/// Which question digest a program binds: the current one, or the released one (active path
/// edges in declared order) a retained artifact was compiled under.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Question {
    Current,
    V1,
}

/// The path spec as the question writes it. The active path edges are a set (distinct
/// outgoing treatment edges; the path-specific contrast reads them by membership), so the
/// current question writes them sorted; the released one kept the declared order. Time labels
/// are already in ascending variable order (any other order is refused).
fn spec_bytes(spec: &TemporalPathSpec, version: Question) -> Result<Vec<u8>, IoError> {
    match version {
        Question::Current => {
            let mut canonical = spec.clone();
            canonical.active_edges.sort_unstable();
            antecedent_io::to_cbor(&canonical)
        }
        Question::V1 => antecedent_io::to_cbor(spec),
    }
}

#[allow(clippy::too_many_lines, reason = "the steps read in order in one place")]
#[allow(clippy::too_many_arguments, reason = "the declared path problem and its exact evidence")]
fn lower(
    problem: &CounterfactualTransportProblem,
    active: &CounterfactualTransportQuery,
    baseline: &CounterfactualTransportQuery,
    catalog: &EvidenceCatalog,
    data: &ExactTransportData,
    spec: &TemporalPathSpec,
    options: &TemporalPathOptionsWire,
    version: Question,
) -> Result<Arc<LoweredProgram>, TemporalPathTransportError> {
    let mut law_bytes: Vec<Vec<u8>> = data
        .laws()
        .iter()
        .map(|law| antecedent_io::to_cbor(&ExactLawWire::from_law(law)))
        .collect::<Result<_, _>>()?;
    law_bytes.sort();
    let mut source_writer =
        CanonicalWriter::new("antecedent.temporal_path_transport.exact_laws.v1");
    source_writer.word(law_bytes.len() as u64);
    for bytes in &law_bytes {
        source_writer.bytes(bytes);
    }
    let source_digest = source_writer.finish();
    let catalog_bytes = antecedent_io::to_cbor(&EvidenceCatalogWire::from_catalog(catalog))?;
    let spec_bytes = spec_bytes(spec, version)?;
    let options_bytes = antecedent_io::to_cbor(options)?;
    let mut writer = CanonicalWriter::new(match version {
        Question::Current => "antecedent.temporal_path_transport.question.v2",
        Question::V1 => "antecedent.temporal_path_transport.question.v1",
    });
    writer
        .text(&problem.canonical_text())
        .text(&active.canonical_text())
        .text(&baseline.canonical_text())
        .bytes(&catalog_bytes)
        .bytes(&spec_bytes)
        .bytes(&options_bytes)
        .bytes(&source_digest);
    let query = writer.finish();
    let design =
        DesignShape::quoted(data.laws().len() as u64, problem.variable_count() as u64, true);
    let scientific = ScientificProblem {
        runtime_query: None,
        identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::PathSpecific,
            outcome: Some(VariableId::from_raw(spec.outcome)),
            query,
        },
        plan_id: Arc::from("temporal_path_transport.v1"),
        data: DataClassification::Tabular,
        graph_class: "time_ordered_selection_admg",
        query_variables: Arc::from(problem.variables().to_vec()),
        source_digest: Some(source_digest),
        support_coordinate: None,
        supported: false,
        support_status: None,
    };
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SharedEvidenceContract { contract: query },
    ));
    let statistical = StatisticalProgram {
        identifier: None,
        estimator: None,
        realization: Realization::ExactPoint {
            evaluator: Arc::from("finite_counterfactual_transport.path_contrast"),
        },
        models: Arc::from([]),
        folds: None,
        resampling: None,
        posterior: None,
        design,
        operators: Arc::from([]),
    };
    let evidence = EvidenceDecisionProgram {
        sources: Arc::from([EvidenceSourceRef {
            role: EvidenceRole::TransportSource,
            digest: Some(source_digest),
        }]),
        composition: Composition::Chained { stages: 2 },
        decision: None,
        planning: None,
        chain: None,
    };
    let physical = PhysicalLayer {
        operators: Arc::from([
            PhysicalOperator::declare(PhysicalEffect::Read)
                .estimates(
                    ByteEstimates::absent()
                        .with_output(law_bytes.iter().map(|bytes| bytes.len() as u64).sum()),
                )
                .build()
                .map_err(|_| {
                    TemporalPathTransportError::Invalid("exact law read cannot be lowered")
                })?,
            PhysicalOperator::declare(PhysicalEffect::Intervene).build().map_err(|_| {
                TemporalPathTransportError::Invalid("path intervention cannot be lowered")
            })?,
            PhysicalOperator::declare(PhysicalEffect::Transport).build().map_err(|_| {
                TemporalPathTransportError::Invalid("path transport cannot be lowered")
            })?,
        ]),
        design,
    };
    Ok(Arc::new(
        StandaloneCompositionLoweredParts {
            cell: ProgramCell::TemporalPathTransport,
            scientific,
            identification,
            statistical,
            evidence,
            physical,
            claim: ClaimDemand {
                uncertainty: UncertaintyReading::None,
                kind: Some(ClaimKind::Point),
            },
        }
        .finish(),
    ))
}

/// Compile a target-population path-specific event contrast after checking
/// time order and proving each world probability from the declared selection
/// diagram and source/target regime catalog. Exact laws are bound to the
/// compiler identity before either probability is evaluated.
///
/// # Errors
/// Invalid temporal order or edge routes, a missing identification proof,
/// proven nontransportability or a resource refusal.
#[allow(
    clippy::too_many_arguments,
    reason = "one declared time/path/selection problem with its exact evidence"
)]
#[allow(
    clippy::needless_pass_by_value,
    reason = "the entry point takes ownership of the declared problem, evidence and options"
)]
pub fn compile_temporal_path_transport(
    problem: CounterfactualTransportProblem,
    target: Arc<str>,
    sources: Arc<[CounterfactualTransportSource]>,
    catalog: EvidenceCatalog,
    data: ExactTransportData,
    spec: TemporalPathSpec,
    options: CounterfactualTransportOptions,
    ctx: &ExecutionContext,
) -> Result<CompiledTemporalPathTransport, TemporalPathTransportError> {
    let catalog = catalog.canonicalized()?;
    let (active_query, baseline_query) = queries(&problem, &target, &sources, &spec)?;
    let options_wire = TemporalPathOptionsWire::from(&options);
    let lowered = lower(
        &problem,
        &active_query,
        &baseline_query,
        &catalog,
        &data,
        &spec,
        &options_wire,
        Question::Current,
    )?;
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
    Ok(CompiledTemporalPathTransport {
        spec,
        options: options_wire,
        data,
        active,
        baseline,
        lowered,
    })
}

impl CompiledProgramHandle for CompiledTemporalPathTransport {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.lowered
    }
}

impl CompiledTemporalPathTransport {
    /// All six compiler layers of the time/path/selection/law program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }

    /// Inspect the compiled program without evaluating its exact laws.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        CompiledProgramHandle::plan_view(self)
    }

    /// Evaluate both identified target-population probabilities on the same
    /// exact laws and subtract them. There is no sampling interval.
    ///
    /// # Errors
    /// A cited regime law is missing, contradictory or outside exact support.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<TemporalPathTransportResult, TemporalPathTransportError> {
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        if GATE_ON_UNIT_SUPPORT {
            gate_program(&scoped, self)
                .map_err(|refusal| TemporalPathTransportError::Invalid(refusal.message()))?;
        }
        let active = self.active.evaluate(&self.data, &scoped)?;
        let baseline = self.baseline.evaluate(&self.data, &scoped)?;
        let contrast = active.probability() - baseline.probability();
        if !contrast.is_finite() {
            return Err(TemporalPathTransportError::Invalid("the path contrast is nonfinite"));
        }
        Ok(TemporalPathTransportResult {
            active,
            baseline,
            contrast,
            lowered_program_identity: self.lowered.identity(),
        })
    }

    /// Seal the two independently replayable identification/evaluation
    /// artifacts and the complete program identity after a fresh comparison.
    ///
    /// # Errors
    /// Changed exact result or artifact encoding failure.
    pub fn export(
        &self,
        result: &TemporalPathTransportResult,
        ctx: &ExecutionContext,
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
            version: 1,
            required_feature: FEATURE.into(),
            spec: self.spec.clone(),
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
    bytes: &[u8],
    expected_program: SemanticDigest,
    ctx: &ExecutionContext,
) -> Result<TemporalPathTransportResult, TemporalPathTransportError> {
    if bytes.len() > MAX_ARTIFACT_BYTES {
        return Err(TemporalPathTransportError::Invalid("path artifact exceeds its bound"));
    }
    let wire: TemporalPathTransportWire = antecedent_io::from_cbor(bytes)?;
    if wire.version != 1
        || wire.required_feature != FEATURE
        || wire.program_identity != *expected_program.as_bytes()
    {
        return Err(TemporalPathTransportError::Invalid("path feature or program differs"));
    }
    let identified = |bytes: &[u8], identity: InnerIdentityWire| {
        match consume_counterfactual_transport_artifact(
            bytes,
            &identity.into(),
            CounterfactualTransportConsumeLimits::default(),
            ctx,
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
    let (active_query, baseline_query) =
        queries(problem, &derivation.query.target, &derivation.query.sources, &wire.spec)?;
    if derivation.query.canonical_text() != active_query.canonical_text()
        || baseline.derivation().query.canonical_text() != baseline_query.canonical_text()
        || active.identity()?.data_digest != baseline.identity()?.data_digest
    {
        return Err(TemporalPathTransportError::Invalid("path worlds or source laws differ"));
    }
    let lower_as = |version| {
        lower(
            problem,
            &active_query,
            &baseline_query,
            catalog,
            data,
            &wire.spec,
            &wire.options,
            version,
        )
    };
    let mut lowered = lower_as(Question::Current)?;
    if lowered.identity() != expected_program {
        // An artifact retained under the released question still verifies: the program is
        // rebuilt under that question and must match it exactly.
        lowered = lower_as(Question::V1)?;
    }
    let contrast = active.probability() - baseline.probability();
    if lowered.identity() != expected_program || contrast.to_bits() != wire.contrast_bits {
        return Err(TemporalPathTransportError::Invalid("recompiled path contrast differs"));
    }
    Ok(TemporalPathTransportResult {
        active,
        baseline,
        contrast,
        lowered_program_identity: lowered.identity(),
    })
}

/// The sampled evidence is embedded in the portable receipt. Its table order
/// is canonical; the two path derivations are independently reidentified by
/// the consumer before the paired estimator is rerun.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SampledPathRequestWire {
    tables: Vec<SampledCountTableWire>,
    overlap: SampledOverlapWire,
    unseen_rule: SampledUnseenRuleWire,
    limits: SampledLimitsWire,
}

impl SampledPathRequestWire {
    fn from_request(request: &SampledTransportRequest) -> Self {
        let mut tables = request.tables.clone();
        tables.sort_by_key(SampledRegimeCounts::key);
        Self {
            tables: tables.iter().map(SampledCountTableWire::from_table).collect(),
            overlap: SampledOverlapWire::from_declaration(&request.overlap),
            unseen_rule: SampledUnseenRuleWire::from_rule(request.unseen_rule),
            limits: SampledLimitsWire::from_limits(&request.limits),
        }
    }

    fn to_request(&self) -> Result<SampledTransportRequest, TemporalPathTransportError> {
        let tables = self
            .tables
            .iter()
            .map(SampledCountTableWire::to_table)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| IoError::Convert(error.to_string()))?;
        let overlap =
            self.overlap.to_declaration().map_err(|error| IoError::Convert(error.to_string()))?;
        let unseen_rule =
            self.unseen_rule.to_rule().map_err(|error| IoError::Convert(error.to_string()))?;
        let request = SampledTransportRequest {
            tables,
            overlap,
            unseen_rule,
            limits: self.limits.to_limits(),
        };
        if Self::from_request(&request).tables != self.tables {
            return Err(TemporalPathTransportError::Invalid(
                "sampled path tables are not canonical",
            ));
        }
        Ok(request)
    }
}

/// A compiler-owned sampled path contrast, including the two stored
/// identification derivations and one count/overlap schedule.
#[derive(Clone, Debug)]
pub struct CompiledSampledTemporalPathTransport {
    spec: TemporalPathSpec,
    options: TemporalPathOptionsWire,
    request: SampledTransportRequest,
    active: PreparedCounterfactualTransport,
    baseline: PreparedCounterfactualTransport,
    lowered: Arc<LoweredProgram>,
}

/// A paired sampled contrast with an uncalibrated shared-data standard error.
#[derive(Clone, Debug, PartialEq)]
pub struct SampledTemporalPathTransportResult {
    /// The paired estimate; no coverage claim has been earned.
    pub estimate: SampledTransportContrastEstimate,
    /// Six-layer identity binding path, selection, counts and sampling design.
    pub lowered_program_identity: SemanticDigest,
}

/// Unmeasured whole-path interval candidate and its exact method coordinate.
/// A prospective grid may measure coverage; this carries no earned record.
pub struct SampledTemporalPathCalibrationCandidate {
    /// Complete compiler execution.
    pub result: SampledTemporalPathTransportResult,
    /// Clamped paired-delta 95% Wald interval of the target path contrast.
    pub interval: (f64, f64),
    /// Source, selection, path, sampling-design and method coordinate.
    pub basis: CalibrationBasis,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SampledPathResultWire {
    active_bits: u64,
    baseline_bits: u64,
    contrast_bits: u64,
    standard_error_bits: Option<u64>,
    withheld_reason: Option<String>,
    evidence_identity: String,
    sampled_units: u64,
}

impl SampledPathResultWire {
    fn from_result(result: &SampledTemporalPathTransportResult) -> Self {
        let estimate = &result.estimate;
        Self {
            active_bits: estimate.active.to_bits(),
            baseline_bits: estimate.baseline.to_bits(),
            contrast_bits: estimate.contrast.to_bits(),
            standard_error_bits: estimate.standard_error.as_ref().map(|se| se.value.to_bits()),
            withheld_reason: estimate.standard_error_withheld_reason.map(str::to_owned),
            evidence_identity: estimate.evidence_identity.clone(),
            sampled_units: estimate.sampled_units,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SampledPathArtifactWire {
    version: u32,
    required_feature: String,
    program_identity: [u8; 32],
    problem: CounterfactualTransportProblemWire,
    active_query: CounterfactualTransportQueryWire,
    catalog: EvidenceCatalogWire,
    spec: TemporalPathSpec,
    options: TemporalPathOptionsWire,
    request: SampledPathRequestWire,
    result: SampledPathResultWire,
}

#[allow(clippy::too_many_arguments, reason = "the declared path problem and its sampled evidence")]
fn lower_sampled(
    problem: &CounterfactualTransportProblem,
    active: &CounterfactualTransportQuery,
    baseline: &CounterfactualTransportQuery,
    catalog: &EvidenceCatalog,
    request: &SampledPathRequestWire,
    spec: &TemporalPathSpec,
    options: &TemporalPathOptionsWire,
    version: Question,
) -> Result<Arc<LoweredProgram>, TemporalPathTransportError> {
    let request_bytes = antecedent_io::to_cbor(request)?;
    let catalog_bytes = antecedent_io::to_cbor(&EvidenceCatalogWire::from_catalog(catalog))?;
    let spec_bytes = spec_bytes(spec, version)?;
    let options_bytes = antecedent_io::to_cbor(options)?;
    let mut source_writer = CanonicalWriter::new("antecedent.sampled_temporal_path.evidence.v1");
    source_writer.bytes(&request_bytes);
    let source_digest = source_writer.finish();
    let mut writer = CanonicalWriter::new(match version {
        Question::Current => "antecedent.sampled_temporal_path.question.v2",
        Question::V1 => "antecedent.sampled_temporal_path.question.v1",
    });
    writer
        .text(&problem.canonical_text())
        .text(&active.canonical_text())
        .text(&baseline.canonical_text())
        .bytes(&catalog_bytes)
        .bytes(&spec_bytes)
        .bytes(&options_bytes)
        .bytes(&source_digest);
    let query = writer.finish();
    let design =
        DesignShape::quoted(request.tables.len() as u64, problem.variable_count() as u64, true);
    let scientific = ScientificProblem {
        runtime_query: None,
        identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::PathSpecific,
            outcome: Some(VariableId::from_raw(spec.outcome)),
            query,
        },
        plan_id: Arc::from("sampled_temporal_path_transport.v1"),
        data: DataClassification::Tabular,
        graph_class: "time_ordered_selection_admg",
        query_variables: Arc::from(problem.variables().to_vec()),
        source_digest: Some(source_digest),
        support_coordinate: None,
        supported: false,
        support_status: None,
    };
    let score = ScoreModel::new("paired_sampled_transport_delta").map_err(|_| {
        TemporalPathTransportError::Invalid("sampled path score identity is invalid")
    })?;
    let statistical = StatisticalProgram {
        identifier: None,
        estimator: None,
        realization: Realization::Frequentist(ScoreFrequentist::new(score)),
        models: Arc::from([]),
        folds: None,
        resampling: None,
        posterior: None,
        design,
        operators: Arc::from([]),
    };
    let evidence = EvidenceDecisionProgram {
        sources: Arc::from([EvidenceSourceRef {
            role: EvidenceRole::TransportSource,
            digest: Some(source_digest),
        }]),
        composition: Composition::Chained { stages: 3 },
        decision: None,
        planning: None,
        chain: None,
    };
    let mut operators = Vec::new();
    for effect in [
        PhysicalEffect::Read,
        PhysicalEffect::Intervene,
        PhysicalEffect::Transport,
        PhysicalEffect::Fit,
    ] {
        operators.push(PhysicalOperator::declare(effect).build().map_err(|_| {
            TemporalPathTransportError::Invalid("sampled path operator cannot be lowered")
        })?);
    }
    let physical = PhysicalLayer { operators: operators.into(), design };
    Ok(Arc::new(
        StandaloneCompositionLoweredParts {
            cell: ProgramCell::TemporalPathTransport,
            scientific,
            identification: IdentificationLayer::NonGraphical(Arc::new(
                NonGraphicalIdentification::SharedEvidenceContract { contract: query },
            )),
            statistical,
            evidence,
            physical,
            claim: ClaimDemand {
                uncertainty: UncertaintyReading::Sampling,
                kind: Some(ClaimKind::Point),
            },
        }
        .finish(),
    ))
}

/// Compile time-checked path worlds and a shared sampled count schedule before
/// numerical execution. The two derivations are identified under one selection
/// catalog. The new standard error carries no inherited calibration.
///
/// # Errors
/// Invalid time/path/evidence declarations or nontransportability.
#[allow(
    clippy::too_many_arguments,
    reason = "one path, selection and sampled-evidence declaration"
)]
pub fn compile_sampled_temporal_path_transport(
    problem: CounterfactualTransportProblem,
    target: Arc<str>,
    sources: Arc<[CounterfactualTransportSource]>,
    catalog: EvidenceCatalog,
    request: SampledTransportRequest,
    spec: TemporalPathSpec,
    options: CounterfactualTransportOptions,
    ctx: &ExecutionContext,
) -> Result<CompiledSampledTemporalPathTransport, TemporalPathTransportError> {
    compile_sampled_temporal_path_transport_as(
        problem,
        target,
        sources,
        catalog,
        request,
        spec,
        options,
        ctx,
        Question::Current,
    )
}

#[allow(
    clippy::too_many_arguments,
    reason = "one path, selection and sampled-evidence declaration"
)]
#[allow(
    clippy::needless_pass_by_value,
    reason = "the entry point takes ownership of the declared problem, evidence and options"
)]
fn compile_sampled_temporal_path_transport_as(
    problem: CounterfactualTransportProblem,
    target: Arc<str>,
    sources: Arc<[CounterfactualTransportSource]>,
    catalog: EvidenceCatalog,
    request: SampledTransportRequest,
    spec: TemporalPathSpec,
    options: CounterfactualTransportOptions,
    ctx: &ExecutionContext,
    version: Question,
) -> Result<CompiledSampledTemporalPathTransport, TemporalPathTransportError> {
    let catalog = catalog.canonicalized()?;
    let (active_query, baseline_query) = queries(&problem, &target, &sources, &spec)?;
    let options_wire = TemporalPathOptionsWire::from(&options);
    let request_wire = SampledPathRequestWire::from_request(&request);
    let lowered = lower_sampled(
        &problem,
        &active_query,
        &baseline_query,
        &catalog,
        &request_wire,
        &spec,
        &options_wire,
        version,
    )?;
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
    Ok(CompiledSampledTemporalPathTransport {
        spec,
        options: options_wire,
        request,
        active,
        baseline,
        lowered,
    })
}

impl CompiledProgramHandle for CompiledSampledTemporalPathTransport {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.lowered
    }
}

impl CompiledSampledTemporalPathTransport {
    /// The complete compiler program, including the sampled evidence identity.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }

    /// Inspect the compiled route without estimating the contrast.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        CompiledProgramHandle::plan_view(self)
    }

    /// Evaluate both stored derivations and their shared-data delta gradient.
    ///
    /// # Errors
    /// Missing or unsupported sampled evidence, budget or numerical refusal.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<SampledTemporalPathTransportResult, TemporalPathTransportError> {
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        if GATE_ON_UNIT_SUPPORT {
            gate_program(&scoped, self)
                .map_err(|refusal| TemporalPathTransportError::Invalid(refusal.message()))?;
        }
        let estimate = estimate_sampled_counterfactual_transport_contrast(
            self.active.derivation(),
            self.baseline.derivation(),
            &self.request,
            &scoped,
        )?;
        Ok(SampledTemporalPathTransportResult {
            estimate,
            lowered_program_identity: self.lowered.identity(),
        })
    }

    /// Expose a prospective whole-contrast coverage candidate at a stable
    /// construction coordinate. Replicate counts and snapshot labels are left
    /// out of the scope hash; table domains, structural zeros, graph, path,
    /// selection catalog, overlap and estimator limits remain bound.
    ///
    /// # Errors
    /// A missing paired standard error or sampled execution refusal.
    pub fn calibration_candidate(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<SampledTemporalPathCalibrationCandidate, TemporalPathTransportError> {
        let result = self.execute(ctx)?;
        let standard_error = result
            .estimate
            .standard_error
            .as_ref()
            .ok_or(TemporalPathTransportError::Invalid(
                "sampled path contrast has no standard error",
            ))?
            .value;
        let margin = 1.959_963_984_540_054 * standard_error;
        let interval = (
            (result.estimate.contrast - margin).max(-1.0),
            (result.estimate.contrast + margin).min(1.0),
        );
        let derivation = self.active.derivation();
        let mut shape = SampledPathRequestWire::from_request(&self.request);
        for table in &mut shape.tables {
            table.counts.clear();
            table.snapshot_identity.clear();
        }
        let mut writer =
            CanonicalWriter::new("antecedent.sampled_temporal_path.calibration_scope.v1");
        writer
            .text(&derivation.problem.canonical_text())
            .text(&derivation.query.canonical_text())
            .text(&self.baseline.derivation().query.canonical_text())
            .bytes(&antecedent_io::to_cbor(&EvidenceCatalogWire::from_catalog(
                &derivation.catalog,
            ))?)
            .bytes(&antecedent_io::to_cbor(&self.spec)?)
            .bytes(&antecedent_io::to_cbor(&self.options)?)
            .bytes(&antecedent_io::to_cbor(&shape)?);
        let scope = SemanticDigest::from_bytes(writer.finish()).to_hex();
        let functional =
            format!("target_path_contrast:outcome={}:scope={scope}", self.spec.outcome);
        let basis = CalibrationBasis::new(
            [
                "PathSpecific",
                "time_ordered_selection_admg",
                "fixed",
                "tabular",
                "Frequentist",
                SAMPLED_FEATURE,
                "paired_delta_wald_clamped_95",
                "paired_multinomial_delta",
                "declared_table_overlap",
                "",
                functional.as_str(),
            ]
            .map(Arc::from),
            0.95,
            Arc::from("point"),
            result.estimate.sampled_units,
            None,
            None,
            0.0,
        )
        .with_validation("none");
        Ok(SampledTemporalPathCalibrationCandidate { result, interval, basis })
    }

    /// Seal the sampled request and result after a fresh compiler execution.
    ///
    /// # Errors
    /// A changed result or artifact encoding refusal.
    pub fn export(
        &self,
        result: &SampledTemporalPathTransportResult,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, TemporalPathTransportError> {
        let live = self.execute(ctx)?;
        if live != *result {
            return Err(TemporalPathTransportError::Invalid(
                "sampled path result differs on replay",
            ));
        }
        let derivation = self.active.derivation();
        let wire = SampledPathArtifactWire {
            version: 1,
            required_feature: SAMPLED_FEATURE.into(),
            program_identity: *self.lowered.identity().as_bytes(),
            problem: CounterfactualTransportProblemWire::from_problem(&derivation.problem),
            active_query: CounterfactualTransportQueryWire::from_query(&derivation.query),
            catalog: EvidenceCatalogWire::from_catalog(&derivation.catalog),
            spec: self.spec.clone(),
            options: self.options.clone(),
            request: SampledPathRequestWire::from_request(&self.request),
            result: SampledPathResultWire::from_result(&live),
        };
        let bytes = antecedent_io::to_cbor(&wire)?;
        if bytes.len() > MAX_ARTIFACT_BYTES {
            return Err(TemporalPathTransportError::Invalid(
                "sampled path artifact exceeds its bound",
            ));
        }
        Ok(bytes)
    }
}

/// Recompile both sampled path derivations and rerun the paired estimator from
/// an embedded count schedule under an externally retained program identity.
///
/// # Errors
/// Foreign feature, changed path/evidence/derivation, or changed result.
pub fn consume_sampled_temporal_path_transport(
    bytes: &[u8],
    expected_program: SemanticDigest,
    ctx: &ExecutionContext,
) -> Result<SampledTemporalPathTransportResult, TemporalPathTransportError> {
    if bytes.len() > MAX_ARTIFACT_BYTES {
        return Err(TemporalPathTransportError::Invalid("sampled path artifact exceeds its bound"));
    }
    let wire: SampledPathArtifactWire = antecedent_io::from_cbor(bytes)?;
    if wire.version != 1
        || wire.required_feature != SAMPLED_FEATURE
        || wire.program_identity != *expected_program.as_bytes()
    {
        return Err(TemporalPathTransportError::Invalid("sampled path feature or program differs"));
    }
    let active_query =
        wire.active_query.to_query().map_err(|error| IoError::Convert(error.to_string()))?;
    let compile_as = |version| -> Result<_, TemporalPathTransportError> {
        let problem =
            wire.problem.to_problem().map_err(|error| IoError::Convert(error.to_string()))?;
        compile_sampled_temporal_path_transport_as(
            problem,
            Arc::clone(&active_query.target),
            Arc::clone(&active_query.sources),
            wire.catalog.to_catalog()?,
            wire.request.to_request()?,
            wire.spec.clone(),
            wire.options.to_options(),
            ctx,
            version,
        )
    };
    let mut compiled = compile_as(Question::Current)?;
    if compiled.lowered.identity() != expected_program {
        // An artifact retained under the released question still verifies: it is recompiled
        // under that question and must match it exactly.
        compiled = compile_as(Question::V1)?;
    }
    if compiled.lowered.identity() != expected_program
        || compiled.active.derivation().query.canonical_text() != active_query.canonical_text()
    {
        return Err(TemporalPathTransportError::Invalid("sampled path program does not recompile"));
    }
    let live = compiled.execute(ctx)?;
    let replayed = SampledPathResultWire::from_result(&live);
    if replayed.active_bits != wire.result.active_bits
        || replayed.baseline_bits != wire.result.baseline_bits
        || replayed.contrast_bits != wire.result.contrast_bits
        || replayed.standard_error_bits != wire.result.standard_error_bits
        || replayed.withheld_reason != wire.result.withheld_reason
        || replayed.evidence_identity != wire.result.evidence_identity
        || replayed.sampled_units != wire.result.sampled_units
    {
        return Err(TemporalPathTransportError::Invalid("sampled path estimate differs on replay"));
    }
    Ok(live)
}

/// One randomized source unit with baseline, sequential continuous mediators,
/// and terminal continuous outcome. All measurements belong to the same unit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContinuousChainSourceRow {
    /// Baseline covariate measured before treatment.
    pub baseline: f64,
    /// Randomized binary treatment.
    pub treated: bool,
    /// At least two continuous mediators in temporal order.
    pub mediators: Vec<f64>,
    /// Continuous terminal outcome.
    pub outcome: f64,
}

/// A continuous sequential linear structural model on a prespecified DAG.
/// Nodes 0..k are ordered mediators and node k is the final outcome. The
/// active world sets treatment to one on selected treatment edges and zero on
/// the others; the baseline world sets it to zero on all edges. Mediator
/// parents propagate recursively in both worlds. Only an observed DAG with
/// independent structural errors is represented; under that declared premise
/// every district is a singleton, excluding recanting districts structurally.
/// The compiler checks edge membership and topological order, not absence of
/// latent confounding. Identification additionally assumes source randomization,
/// consistency, no interference, complete observations, correct linear
/// conditional means, no unmeasured mediator or outcome confounding, invariant
/// source-to-target mechanisms, and positivity in both intervention worlds.
/// Baseline range checks do not establish mediator positivity.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContinuousChainTransportRequest {
    /// Source randomized unit records in common chronological measurement order.
    pub source: Vec<ContinuousChainSourceRow>,
    /// Target-population baseline covariate sample.
    pub target_baseline: Vec<f64>,
    /// Earlier mediator parents of each mediator and the final outcome.
    /// Each list is strictly increasing; every parent index must be less than its node index.
    pub mediator_parents: Vec<Vec<usize>>,
    /// Whether the treatment has an edge to each mediator and the final outcome.
    pub treatment_edges: Vec<bool>,
    /// Whether each corresponding treatment edge receives the active value.
    pub active_treatment_edges: Vec<bool>,
    /// Prespecified paired, arm-stratified source and target bootstrap replicates.
    pub bootstrap_replicates: usize,
}

/// A retained compiler plan for the restricted continuous chain.
#[derive(Clone, Debug)]
pub struct CompiledContinuousChainTransport {
    request: ContinuousChainTransportRequest,
    seed: u64,
    lowered: Arc<LoweredProgram>,
}

/// Point contrast and an unmeasured paired percentile candidate interval.
#[derive(Clone, Debug, PartialEq)]
pub struct ContinuousChainTransportResult {
    /// Target mean under the active treatment.
    pub active: f64,
    /// Target mean under the control treatment.
    pub baseline: f64,
    /// Active minus baseline under the declared edge intervention.
    pub contrast: f64,
    /// Percentile bootstrap interval; coverage is not measured.
    pub interval: (f64, f64),
    /// Always `not_measured`.
    pub calibration: &'static str,
    /// Compiler identity including source data and sampling schedule.
    pub lowered_program_identity: SemanticDigest,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ContinuousChainTransportWire {
    version: u32,
    feature: String,
    request: ContinuousChainTransportRequest,
    seed: u64,
    program_identity: [u8; 32],
    result_fingerprint: [u8; 32],
}

// OLS for one declared structural equation; partial pivoting rejects rank deficiency.
fn continuous_chain_fit(
    rows: &[&ContinuousChainSourceRow],
    equation: usize,
    parents: &[usize],
    treatment_edge: bool,
) -> Result<Vec<f64>, TemporalPathTransportError> {
    let p = 2 + usize::from(treatment_edge) + parents.len();
    let mut matrix = vec![vec![0.0; p + 1]; p];
    for row in rows {
        let mut x = Vec::with_capacity(p);
        x.push(1.0);
        x.push(row.baseline);
        if treatment_edge {
            x.push(if row.treated { 1.0 } else { 0.0 });
        }
        x.extend(parents.iter().map(|&parent| row.mediators[parent]));
        let y = if equation == row.mediators.len() { row.outcome } else { row.mediators[equation] };
        for i in 0..p {
            for j in 0..p {
                matrix[i][j] += x[i] * x[j];
            }
            matrix[i][p] += x[i] * y;
        }
    }
    for col in 0..p {
        let pivot = (col..p)
            .max_by(|&a, &b| matrix[a][col].abs().total_cmp(&matrix[b][col].abs()))
            .ok_or(TemporalPathTransportError::Invalid("continuous chain fit has no pivot"))?;
        if !matrix[pivot][col].is_finite() || matrix[pivot][col].abs() <= 1e-10 {
            return Err(TemporalPathTransportError::Invalid(
                "continuous chain structural fit is singular",
            ));
        }
        matrix.swap(col, pivot);
        let divisor = matrix[col][col];
        for j in col..=p {
            matrix[col][j] /= divisor;
        }
        let pivot_row = matrix[col].clone();
        for i in 0..p {
            if i != col {
                let factor = matrix[i][col];
                for j in col..=p {
                    matrix[i][j] -= factor * pivot_row[j];
                }
            }
        }
    }
    let coefficients: Vec<f64> = matrix.iter().map(|row| row[p]).collect();
    if coefficients.iter().any(|value| !value.is_finite()) {
        return Err(TemporalPathTransportError::Invalid(
            "continuous chain coefficients are nonfinite",
        ));
    }
    Ok(coefficients)
}

fn continuous_chain_worlds(
    rows: &[&ContinuousChainSourceRow],
    x: f64,
    request: &ContinuousChainTransportRequest,
) -> Result<(f64, f64), TemporalPathTransportError> {
    let k = rows[0].mediators.len();
    let coefficients = (0..=k)
        .map(|equation| {
            continuous_chain_fit(
                rows,
                equation,
                &request.mediator_parents[equation],
                request.treatment_edges[equation],
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let world = |active: bool| {
        let mut prior = Vec::with_capacity(k + 1);
        for (equation, beta) in coefficients.iter().enumerate() {
            let has_treatment = request.treatment_edges[equation];
            let treatment_input =
                if active && request.active_treatment_edges[equation] { 1.0 } else { 0.0 };
            let mut mean = beta[0] + beta[1] * x;
            let offset = if has_treatment {
                mean += beta[2] * treatment_input;
                3
            } else {
                2
            };
            mean += request.mediator_parents[equation]
                .iter()
                .enumerate()
                .map(|(i, &parent)| prior[parent] * beta[offset + i])
                .sum::<f64>();
            prior.push(mean);
        }
        prior[k]
    };
    let baseline = world(false);
    let active = world(true);
    if !baseline.is_finite() || !active.is_finite() {
        return Err(TemporalPathTransportError::Invalid("continuous chain world is nonfinite"));
    }
    Ok((baseline, active))
}

fn continuous_chain_fingerprint(result: &ContinuousChainTransportResult) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("antecedent.continuous_chain_transport.result.v1");
    writer
        .word(result.active.to_bits())
        .word(result.baseline.to_bits())
        .word(result.contrast.to_bits())
        .word(result.interval.0.to_bits())
        .word(result.interval.1.to_bits())
        .text(result.calibration)
        .bytes(result.lowered_program_identity.as_bytes());
    writer.finish()
}

/// Compile the restricted chain and a common paired resampling schedule.
/// The active edge intervention and baseline world are composed from fitted
/// sequential mediator and outcome mechanisms.
///
/// # Errors
/// Unsupported dimensions, nonfinite rows, no empirical arm overlap, singular
/// original fits, or resource bound.
#[allow(clippy::too_many_lines, reason = "the steps read in order in one place")]
pub fn compile_continuous_chain_transport(
    request: ContinuousChainTransportRequest,
    ctx: &ExecutionContext,
) -> Result<CompiledContinuousChainTransport, TemporalPathTransportError> {
    let n = request.source.len();
    let m = request.target_baseline.len();
    let k = request.source.first().map_or(0, |row| row.mediators.len());
    if n < 40
        || m < 20
        || !(2..=8).contains(&k)
        || request.bootstrap_replicates < 100
        || request.bootstrap_replicates > 4096
        || n.checked_add(m)
            .and_then(|v| v.checked_mul(request.bootstrap_replicates))
            .is_none_or(|work| work > 50_000_000)
        || request.source.iter().any(|row| {
            !row.baseline.is_finite()
                || !row.outcome.is_finite()
                || row.mediators.len() != k
                || row.mediators.iter().any(|x| !x.is_finite())
        })
        || request.target_baseline.iter().any(|x| !x.is_finite())
        || request.mediator_parents.len() != k + 1
        || request.treatment_edges.len() != k + 1
        || request.active_treatment_edges.len() != k + 1
        || !request.active_treatment_edges.iter().any(|active| *active)
        || request
            .active_treatment_edges
            .iter()
            .zip(&request.treatment_edges)
            .any(|(active, exists)| *active && !*exists)
        || request.mediator_parents.iter().enumerate().any(|(node, parents)| {
            parents.iter().any(|&parent| parent >= node)
                || parents.windows(2).any(|pair| pair[0] >= pair[1])
        })
    {
        return Err(TemporalPathTransportError::Invalid(
            "invalid continuous chain source or schedule",
        ));
    }
    let target_mean = request.target_baseline.iter().sum::<f64>() / m as f64;
    for arm in [false, true] {
        let rows: Vec<_> = request.source.iter().filter(|row| row.treated == arm).collect();
        if rows.len() < 20 {
            return Err(TemporalPathTransportError::Invalid(
                "continuous chain requires twenty source units per arm",
            ));
        }
        let min = rows.iter().map(|row| row.baseline).fold(f64::INFINITY, f64::min);
        let max = rows.iter().map(|row| row.baseline).fold(f64::NEG_INFINITY, f64::max);
        if request.target_baseline.iter().any(|x| *x < min || *x > max) {
            return Err(TemporalPathTransportError::Invalid(
                "target baseline is outside a source arm range",
            ));
        }
    }
    let all_rows: Vec<_> = request.source.iter().collect();
    continuous_chain_worlds(&all_rows, target_mean, &request)?;
    let bytes = antecedent_io::to_cbor(&request)?;
    if bytes.len() > MAX_ARTIFACT_BYTES {
        return Err(TemporalPathTransportError::Invalid(
            "continuous chain source exceeds its byte bound",
        ));
    }
    let source_digest =
        CanonicalWriter::digest_bytes("antecedent.continuous_chain_transport.source.v1", &bytes);
    let seed = ctx.rng.master_seed();
    let mut writer = CanonicalWriter::new("antecedent.continuous_chain_transport.question.v1");
    writer.bytes(&source_digest).word(seed);
    let query = writer.finish();
    let design = DesignShape::quoted(n as u64, (k + 3) as u64, true);
    let scientific = ScientificProblem {
        runtime_query: None,
        identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::PathSpecific,
            outcome: None,
            query,
        },
        plan_id: Arc::from("continuous_sequential_edge_dag_transport.v2"),
        data: DataClassification::Tabular,
        graph_class: "declared_randomized_sequential_edge_dag",
        query_variables: Arc::from([]),
        source_digest: Some(source_digest),
        support_coordinate: None,
        supported: false,
        support_status: None,
    };
    let score =
        ScoreModel::new("continuous_sequential_edge_gformula_paired_bootstrap").map_err(|_| {
            TemporalPathTransportError::Invalid("continuous chain score identity is invalid")
        })?;
    let statistical = StatisticalProgram {
        identifier: None,
        estimator: None,
        realization: Realization::Frequentist(ScoreFrequentist::new(score)),
        models: Arc::from([]),
        folds: None,
        resampling: None,
        posterior: None,
        design,
        operators: Arc::from([]),
    };
    let evidence = EvidenceDecisionProgram {
        sources: Arc::from([EvidenceSourceRef {
            role: EvidenceRole::TransportSource,
            digest: Some(source_digest),
        }]),
        composition: Composition::Chained { stages: request.bootstrap_replicates + 1 },
        decision: None,
        planning: None,
        chain: None,
    };
    let mut operators = Vec::new();
    for effect in [
        PhysicalEffect::Read,
        PhysicalEffect::Select,
        PhysicalEffect::Fit,
        PhysicalEffect::Transport,
    ] {
        operators.push(PhysicalOperator::declare(effect).build().map_err(|_| {
            TemporalPathTransportError::Invalid("continuous chain operator cannot be lowered")
        })?);
    }
    let lowered = Arc::new(
        StandaloneCompositionLoweredParts {
            cell: ProgramCell::TemporalPathTransport,
            scientific,
            identification: IdentificationLayer::NonGraphical(Arc::new(
                NonGraphicalIdentification::SharedEvidenceContract { contract: query },
            )),
            statistical,
            evidence,
            physical: PhysicalLayer { operators: operators.into(), design },
            claim: ClaimDemand {
                uncertainty: UncertaintyReading::Sampling,
                kind: Some(ClaimKind::Point),
            },
        }
        .finish(),
    );
    Ok(CompiledContinuousChainTransport { request, seed, lowered })
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "the remainder is below a bound that came from a usize"
)]
fn continuous_chain_draw(rng: &mut antecedent_core::CausalRng, n: usize) -> usize {
    let bound = n as u64;
    let threshold = bound.wrapping_neg() % bound;
    loop {
        let draw = rng.next_u64();
        if draw >= threshold {
            return (draw % bound) as usize;
        }
    }
}

impl CompiledProgramHandle for CompiledContinuousChainTransport {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.lowered
    }
}

impl CompiledContinuousChainTransport {
    /// Inspect the complete lowered transport and sampling program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }
    /// Inspect the compiled route before execution.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        CompiledProgramHandle::plan_view(self)
    }

    /// Refit both worlds on each shared source and target resample.
    ///
    /// # Errors
    /// Seed mismatch, cancellation or a singular bootstrap fit; failures are never discarded.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<ContinuousChainTransportResult, TemporalPathTransportError> {
        if ctx.rng.master_seed() != self.seed {
            return Err(TemporalPathTransportError::Invalid("continuous chain seed changed"));
        }
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        if GATE_ON_UNIT_SUPPORT {
            gate_program(&scoped, self)
                .map_err(|refusal| TemporalPathTransportError::Invalid(refusal.message()))?;
        }
        let arms: Vec<Vec<_>> = [false, true]
            .iter()
            .map(|arm| self.request.source.iter().filter(|row| row.treated == *arm).collect())
            .collect();
        let target_mean = self.request.target_baseline.iter().sum::<f64>()
            / self.request.target_baseline.len() as f64;
        let original: Vec<_> = self.request.source.iter().collect();
        let (baseline, active) = continuous_chain_worlds(&original, target_mean, &self.request)?;
        let contrast = active - baseline;
        let mut rng = scoped.rng.stream(0x434F_4E54_4348_4149);
        let mut replicates = Vec::with_capacity(self.request.bootstrap_replicates);
        for _ in 0..self.request.bootstrap_replicates {
            if scoped.cancellation.is_cancelled() {
                return Err(TemporalPathTransportError::Invalid(
                    "continuous chain bootstrap cancelled",
                ));
            }
            let sampled: Vec<Vec<_>> = arms
                .iter()
                .map(|arm| {
                    (0..arm.len())
                        .map(|_| arm[continuous_chain_draw(&mut rng, arm.len())])
                        .collect()
                })
                .collect();
            let mean = (0..self.request.target_baseline.len())
                .map(|_| {
                    self.request.target_baseline
                        [continuous_chain_draw(&mut rng, self.request.target_baseline.len())]
                })
                .sum::<f64>()
                / self.request.target_baseline.len() as f64;
            let rows: Vec<_> = sampled.iter().flat_map(|arm| arm.iter().copied()).collect();
            let (baseline, active) = continuous_chain_worlds(&rows, mean, &self.request)?;
            replicates.push(active - baseline);
        }
        replicates.sort_by(f64::total_cmp);
        let interval = (
            antecedent_kernels::quantile_type7_sorted(&replicates, 0.025),
            antecedent_kernels::quantile_type7_sorted(&replicates, 0.975),
        );
        if !contrast.is_finite() || !interval.0.is_finite() || !interval.1.is_finite() {
            return Err(TemporalPathTransportError::Invalid(
                "continuous chain contrast is nonfinite",
            ));
        }
        Ok(ContinuousChainTransportResult {
            active,
            baseline,
            contrast,
            interval,
            calibration: "not_measured",
            lowered_program_identity: self.lowered.identity(),
        })
    }

    /// Seal the complete raw schedule after fresh result verification.
    ///
    /// # Errors
    /// Changed result, execution failure or oversized artifact.
    pub fn export(
        &self,
        result: &ContinuousChainTransportResult,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, TemporalPathTransportError> {
        let fingerprint = continuous_chain_fingerprint(result);
        if fingerprint != continuous_chain_fingerprint(&self.execute(ctx)?) {
            return Err(TemporalPathTransportError::Invalid("continuous chain result differs"));
        }
        let bytes = antecedent_io::to_cbor(&ContinuousChainTransportWire {
            version: 1,
            feature: "continuous_sequential_edge_dag_transport_v2".into(),
            request: self.request.clone(),
            seed: self.seed,
            program_identity: *self.lowered.identity().as_bytes(),
            result_fingerprint: fingerprint,
        })?;
        if bytes.len() > MAX_ARTIFACT_BYTES {
            return Err(TemporalPathTransportError::Invalid(
                "continuous chain artifact exceeds its byte bound",
            ));
        }
        Ok(bytes)
    }
}

/// Independently recompile and rerun the continuous chain schedule.
///
/// # Errors
/// Changed identity, source, seed, feature, result or nested fit.
pub fn consume_continuous_chain_transport(
    bytes: &[u8],
    expected: SemanticDigest,
    ctx: &ExecutionContext,
) -> Result<ContinuousChainTransportResult, TemporalPathTransportError> {
    if bytes.len() > MAX_ARTIFACT_BYTES {
        return Err(TemporalPathTransportError::Invalid(
            "continuous chain artifact exceeds its byte bound",
        ));
    }
    let wire: ContinuousChainTransportWire = antecedent_io::from_cbor(bytes)?;
    if wire.version != 1
        || wire.feature != "continuous_sequential_edge_dag_transport_v2"
        || wire.seed != ctx.rng.master_seed()
        || wire.program_identity != *expected.as_bytes()
    {
        return Err(TemporalPathTransportError::Invalid(
            "continuous chain feature, seed or identity differs",
        ));
    }
    let plan = compile_continuous_chain_transport(wire.request, ctx)?;
    if plan.lowered.identity() != expected {
        return Err(TemporalPathTransportError::Invalid("continuous chain program differs"));
    }
    let result = plan.execute(ctx)?;
    if continuous_chain_fingerprint(&result) != wire.result_fingerprint {
        return Err(TemporalPathTransportError::Invalid(
            "continuous chain result differs on replay",
        ));
    }
    Ok(result)
}

/// Prospective calibration coordinate for the restricted paired bootstrap.
pub struct ContinuousChainCalibrationCandidate {
    /// Complete compiler execution with unmeasured interval.
    pub result: ContinuousChainTransportResult,
    /// Exact method and chain structure for a future calibration record; the
    /// source sample size is its row count.
    pub basis: CalibrationBasis,
}

impl CompiledContinuousChainTransport {
    /// Construct a prospective, unmeasured coverage candidate.
    ///
    /// # Errors
    /// Any complete-schedule execution or fitting failure.
    #[allow(clippy::cast_possible_truncation, reason = "the replicate count is at most 4096")]
    pub fn calibration_candidate(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<ContinuousChainCalibrationCandidate, TemporalPathTransportError> {
        let result = self.execute(ctx)?;
        // The edge structure is the construction; the arm and target sample
        // sizes are data, and the source size is the scope's row count.
        let mut writer =
            CanonicalWriter::new("antecedent.continuous_edge_dag.calibration_scope.v2");
        writer
            .bytes(&antecedent_io::to_cbor(&self.request.mediator_parents)?)
            .bytes(&antecedent_io::to_cbor(&self.request.treatment_edges)?)
            .bytes(&antecedent_io::to_cbor(&self.request.active_treatment_edges)?);
        let scope = SemanticDigest::from_bytes(writer.finish()).to_hex();
        let functional = format!("sequential_edge_dag:scope={scope}");
        let basis = CalibrationBasis::new(
            [
                "PathSpecific",
                "randomized_sequential_edge_dag",
                "fixed",
                "tabular",
                "Frequentist",
                "continuous_sequential_edge_dag_transport_v2",
                "paired_stratified_percentile_bootstrap_95",
                "full_refit",
                "independent_units_within_unit_longitudinal_dependence",
                "",
                functional.as_str(),
            ]
            .map(Arc::from),
            0.95,
            Arc::from("point"),
            self.request.source.len() as u64,
            Some(self.request.bootstrap_replicates as u32),
            None,
            0.0,
        )
        .with_validation("none");
        Ok(ContinuousChainCalibrationCandidate { result, basis })
    }
}

#[cfg(test)]
mod continuous_chain_tests {
    use super::*;

    #[test]
    fn continuous_chain_shared_schedule_replays() {
        let ctx = ExecutionContext::for_tests(991);
        let source = (0..200)
            .map(|i| {
                let x = f64::from(i % 25) / 25.0;
                let treated = i >= 100;
                let a = if treated { 1.0 } else { 0.0 };
                let noise1 = if (i / 25) % 2 == 0 { -1.0 } else { 1.0 };
                let noise2 = if (i / 50) % 2 == 0 { -1.0 } else { 1.0 };
                let m1 = x + a + noise1;
                let m2 = 2.0 * m1 + 3.0 * a + noise2;
                ContinuousChainSourceRow {
                    baseline: x,
                    treated,
                    mediators: vec![m1, m2],
                    outcome: 3.0 + x + m2 + 4.0 * a,
                }
            })
            .collect();
        let request = ContinuousChainTransportRequest {
            source,
            target_baseline: (0..25).map(|i| f64::from(i) / 25.0).collect(),
            mediator_parents: vec![vec![], vec![0], vec![0, 1]],
            treatment_edges: vec![true, true, true],
            active_treatment_edges: vec![true, false, false],
            bootstrap_replicates: 100,
        };
        let plan = compile_continuous_chain_transport(request.clone(), &ctx).unwrap();
        let candidate = plan.calibration_candidate(&ctx).unwrap();
        assert!((candidate.result.contrast - 2.0).abs() < 1e-10);
        assert_eq!(candidate.result.calibration, "not_measured");
        let artifact = plan.export(&candidate.result, &ctx).unwrap();
        let replay =
            consume_continuous_chain_transport(&artifact, plan.lowered().identity(), &ctx).unwrap();
        assert_eq!(candidate.result, replay);
        let mut alternate = request.clone();
        alternate.active_treatment_edges = vec![false, true, true];
        let alternate_plan = compile_continuous_chain_transport(alternate, &ctx).unwrap();
        assert_ne!(plan.lowered().identity(), alternate_plan.lowered().identity());
        let alternate_result = alternate_plan.execute(&ctx).unwrap();
        assert!((alternate_result.contrast - 7.0).abs() < 1e-10);
        assert!(
            consume_continuous_chain_transport(
                &artifact,
                alternate_plan.lowered().identity(),
                &ctx
            )
            .is_err()
        );
        let mut invalid = request;
        invalid.mediator_parents[1] = vec![1];
        assert!(compile_continuous_chain_transport(invalid, &ctx).is_err());
    }
}

#[cfg(test)]
mod site_tests {
    use super::*;
    use crate::identity_laws::{all_distinct, bit_exact, set_laws};
    use antecedent_core::{RegimeId, Value};
    use antecedent_estimate::counterfactual_transport_sampled::{
        OverlapDeclaration, SampledAxis, SampledRegimeCountsParts, SampledTransportLimits,
        SamplingDesign, UnseenCellRule,
    };
    use antecedent_expr::{DiscreteAxis, ExactDiscreteLaw, LawTolerance};

    fn query_of(lowered: &LoweredProgram) -> [u8; 32] {
        match lowered.scientific().target() {
            ProblemTarget::QueryFunctional { query, .. } => *query,
            _ => panic!("the program target is a query functional"),
        }
    }

    /// Binary `X -> M -> Y` with the direct `X -> Y` edge, at times 0, 1 and 2.
    fn problem() -> CounterfactualTransportProblem {
        let id = VariableId::from_raw;
        CounterfactualTransportProblem::new(
            &[(id(0), vec![0.0, 1.0]), (id(1), vec![0.0, 1.0]), (id(2), vec![0.0, 1.0])],
            &[(id(0), id(1)), (id(1), id(2)), (id(0), id(2))],
            &[],
        )
        .expect("a binary mediation graph")
    }

    fn path_spec(active_edges: Vec<(u32, u32)>) -> TemporalPathSpec {
        TemporalPathSpec {
            treatment: 0,
            outcome: 2,
            control_bits: 0.0_f64.to_bits(),
            active_bits: 1.0_f64.to_bits(),
            outcome_level_bits: 1.0_f64.to_bits(),
            times: vec![(0, 0), (1, 1), (2, 2)],
            active_edges,
        }
    }

    fn worlds(
        spec: &TemporalPathSpec,
    ) -> (CounterfactualTransportQuery, CounterfactualTransportQuery) {
        let target: Arc<str> = Arc::from("target");
        let sources: Arc<[CounterfactualTransportSource]> = Arc::from(Vec::new());
        queries(&problem(), &target, &sources, spec).expect("a time-ordered path query")
    }

    fn law(population: &str) -> ExactDiscreteLaw {
        ExactDiscreteLaw::try_new(
            population,
            RegimeId::from_raw(0),
            Vec::<antecedent_expr::InterventionAssignment>::new(),
            vec![DiscreteAxis {
                variable: VariableId::from_raw(2),
                values: Arc::from(vec![Value::f64(0.0), Value::f64(1.0)]),
            }],
            vec![0.4, 0.6],
            format!("snapshot:{population}"),
            LawTolerance::default(),
        )
        .expect("a normalized exact law")
    }

    #[test]
    fn site_temporal_path_transport_exact_laws_are_a_set() {
        let spec = path_spec(vec![(0, 2)]);
        let (active, baseline) = worlds(&spec);
        let options = TemporalPathOptionsWire::from(&CounterfactualTransportOptions::default());
        let catalog = EvidenceCatalog::empty();
        let identity = |populations: &[&str]| {
            let laws: Vec<ExactDiscreteLaw> = populations.iter().map(|name| law(name)).collect();
            let data = ExactTransportData::try_new(laws, 1 << 16).ok()?;
            lower(
                &problem(),
                &active,
                &baseline,
                &catalog,
                &data,
                &spec,
                &options,
                Question::Current,
            )
            .ok()
            .map(|lowered| query_of(&lowered))
        };
        // The laws are keyed by population, regime and world (a repeated binding is refused),
        // so the evidence is a set of laws.
        set_laws(&["target", "source-a", "source-b"], identity);
        let data = ExactTransportData::try_new(vec![law("target")], 1 << 16).expect("one law");
        let other_spec = path_spec(vec![(0, 1)]);
        let (other_active, other_baseline) = worlds(&other_spec);
        let named = TemporalPathOptionsWire::from(&CounterfactualTransportOptions {
            variable_names: vec!["x".into(), "m".into(), "y".into()],
            ..CounterfactualTransportOptions::default()
        });
        all_distinct(&[
            identity(&["target"]),
            lower(
                &problem(),
                &other_active,
                &other_baseline,
                &catalog,
                &data,
                &other_spec,
                &options,
                Question::Current,
            )
            .ok()
            .map(|lowered| query_of(&lowered)),
            lower(
                &problem(),
                &active,
                &baseline,
                &catalog,
                &data,
                &spec,
                &named,
                Question::Current,
            )
            .ok()
            .map(|lowered| query_of(&lowered)),
        ]);
    }

    fn table(population: &str, treated: u64) -> SampledRegimeCounts {
        SampledRegimeCounts::try_new(SampledRegimeCountsParts {
            population: Arc::from(population),
            regime: RegimeId::from_raw(0),
            interventions: Vec::new(),
            axes: vec![SampledAxis { variable: VariableId::from_raw(2), levels: vec![0.0, 1.0] }],
            counts: vec![100 - treated, treated],
            snapshot_identity: Arc::from(format!("snapshot:{population}")),
            design: SamplingDesign::IndependentUnits,
            structural_zeros: Vec::new(),
        })
        .expect("a sampled count table")
    }

    #[test]
    fn site_sampled_temporal_path_transport_tables_are_sorted_by_key() {
        let spec = path_spec(vec![(0, 2)]);
        let (active, baseline) = worlds(&spec);
        let options = TemporalPathOptionsWire::from(&CounterfactualTransportOptions::default());
        let catalog = EvidenceCatalog::empty();
        let identity = |tables: &[(&str, u64)]| {
            let request = SampledTransportRequest {
                tables: tables.iter().map(|(name, treated)| table(name, *treated)).collect(),
                overlap: OverlapDeclaration::all_independent(),
                unseen_rule: UnseenCellRule::Refuse,
                limits: SampledTransportLimits::default(),
            };
            let wire = SampledPathRequestWire::from_request(&request);
            lower_sampled(
                &problem(),
                &active,
                &baseline,
                &catalog,
                &wire,
                &spec,
                &options,
                Question::Current,
            )
            .ok()
            .map(|lowered| query_of(&lowered))
        };
        // Tables are keyed by population, regime and world and sorted by that key; the sampled
        // estimator refuses a repeated key when it runs, so only reorderings are compared here.
        let base = identity(&[("target", 60), ("source-a", 55), ("source-b", 50)]);
        assert_eq!(base, identity(&[("source-b", 50), ("target", 60), ("source-a", 55)]));
        assert_eq!(base, identity(&[("source-a", 55), ("source-b", 50), ("target", 60)]));
        all_distinct(&[
            base,
            identity(&[("target", 60), ("source-a", 55)]),
            identity(&[("target", 61), ("source-a", 55), ("source-b", 50)]),
        ]);
    }

    fn chain_request() -> ContinuousChainTransportRequest {
        let source = (0..200)
            .map(|i| {
                let x = f64::from(i % 25) / 25.0;
                let treated = i >= 100;
                let a = if treated { 1.0 } else { 0.0 };
                let noise1 = if (i / 25) % 2 == 0 { -1.0 } else { 1.0 };
                let noise2 = if (i / 50) % 2 == 0 { -1.0 } else { 1.0 };
                let m1 = x + a + noise1;
                let m2 = 2.0 * m1 + 3.0 * a + noise2;
                ContinuousChainSourceRow {
                    baseline: x,
                    treated,
                    mediators: vec![m1, m2],
                    outcome: 3.0 + x + m2 + 4.0 * a,
                }
            })
            .collect();
        ContinuousChainTransportRequest {
            source,
            target_baseline: (0..25).map(|i| f64::from(i) / 25.0).collect(),
            mediator_parents: vec![vec![], vec![0], vec![0, 1]],
            treatment_edges: vec![true, true, true],
            active_treatment_edges: vec![true, false, false],
            bootstrap_replicates: 100,
        }
    }

    #[test]
    fn site_continuous_chain_transport_question_binds_rows_schedule_and_seed() {
        let question = |request: ContinuousChainTransportRequest, seed: u64| {
            let ctx = ExecutionContext::for_tests(seed);
            compile_continuous_chain_transport(request, &ctx)
                .ok()
                .map(|plan| query_of(plan.lowered()))
        };
        // Source rows and target baselines are positional: the paired bootstrap draws them by
        // index from the seeded schedule.
        let mut source_rows = chain_request();
        source_rows.source.swap(0, 1);
        let mut target_rows = chain_request();
        target_rows.target_baseline.swap(0, 1);
        let mut edges = chain_request();
        edges.active_treatment_edges = vec![false, true, true];
        let mut replicates = chain_request();
        replicates.bootstrap_replicates = 101;
        all_distinct(&[
            question(chain_request(), 991),
            question(chain_request(), 992),
            question(source_rows, 991),
            question(target_rows, 991),
            question(edges, 991),
            question(replicates, 991),
        ]);
    }

    #[test]
    fn site_continuous_chain_transport_result_fingerprint_reads_every_field() {
        let base = ContinuousChainTransportResult {
            active: 1.0,
            baseline: 0.5,
            contrast: 0.5,
            interval: (0.25, 0.75),
            calibration: "not_measured",
            lowered_program_identity: SemanticDigest::from_bytes([3; 32]),
        };
        let mut variants = vec![base.clone()];
        let mut changed = base.clone();
        changed.active = 1.5;
        variants.push(changed);
        let mut changed = base.clone();
        changed.baseline = 0.6;
        variants.push(changed);
        let mut changed = base.clone();
        changed.contrast = 0.4;
        variants.push(changed);
        let mut changed = base.clone();
        changed.interval = (0.75, 0.25);
        variants.push(changed);
        let mut changed = base.clone();
        changed.calibration = "measured";
        variants.push(changed);
        let mut changed = base.clone();
        changed.lowered_program_identity = SemanticDigest::from_bytes([4; 32]);
        variants.push(changed);
        all_distinct(&variants.iter().map(continuous_chain_fingerprint).collect::<Vec<_>>());
        bit_exact(|contrast| {
            let mut changed = base.clone();
            changed.contrast = contrast;
            Some(continuous_chain_fingerprint(&changed))
        });
    }

    #[test]
    fn site_temporal_path_transport_active_edges_are_a_set() {
        let target: Arc<str> = Arc::from("target");
        let sources: Arc<[CounterfactualTransportSource]> = Arc::from(Vec::new());
        // The active edges are distinct outgoing treatment edges (a repeat is refused, and
        // skipped by the law) that the path contrast reads by membership.
        set_laws(&[(0_u32, 2_u32), (0, 1)], |edges: &[(u32, u32)]| {
            let spec = path_spec(edges.to_vec());
            queries(&problem(), &target, &sources, &spec).ok()?;
            spec_bytes(&spec, Question::Current).ok()
        });
        // The whole program agrees: the path worlds do not depend on the declared edge order.
        let options = TemporalPathOptionsWire::from(&CounterfactualTransportOptions::default());
        let catalog = EvidenceCatalog::empty();
        let data = ExactTransportData::try_new(vec![law("target")], 1 << 16).expect("one law");
        let program = |edges: Vec<(u32, u32)>, version: Question| {
            let spec = path_spec(edges);
            let (active, baseline) = worlds(&spec);
            lower(&problem(), &active, &baseline, &catalog, &data, &spec, &options, version)
                .ok()
                .map(|lowered| lowered.identity())
        };
        assert_eq!(
            program(vec![(0, 2), (0, 1)], Question::Current),
            program(vec![(0, 1), (0, 2)], Question::Current)
        );
        // The released question kept the declared order, so a retained artifact rebuilt under it
        // reproduces its own identity and no other.
        assert_ne!(
            program(vec![(0, 2), (0, 1)], Question::V1),
            program(vec![(0, 1), (0, 2)], Question::V1)
        );
        assert_ne!(
            program(vec![(0, 1), (0, 2)], Question::V1),
            program(vec![(0, 1), (0, 2)], Question::Current)
        );
    }

    #[test]
    fn site_temporal_path_transport_released_question_keeps_its_bytes() {
        // The released question wrote the spec as declared under its own tag; the rebuild a
        // retained artifact is verified against must write exactly those bytes.
        let spec = path_spec(vec![(0, 2), (0, 1)]);
        assert_eq!(
            spec_bytes(&spec, Question::V1).ok(),
            antecedent_io::to_cbor(&spec).ok(),
            "the released question writes the declared spec"
        );
        let mut sorted = spec.clone();
        sorted.active_edges.sort_unstable();
        assert_eq!(spec_bytes(&spec, Question::Current).ok(), antecedent_io::to_cbor(&sorted).ok());
    }
}
