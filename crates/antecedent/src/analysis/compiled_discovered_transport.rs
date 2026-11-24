//! Compiled transport across the class a compiled structure search returns.
//!
//! [`Study::compile_discovered_class_transport`] builds the study that searches the declared
//! table with the declared static selective contract (`DiscoverySpec::selective_static`) and
//! compiles it: that search is the `DiscoverStatic` program of its algorithm, PC or FCI. The
//! transport request over the class it returns (the declared selection targets, the transport
//! query, the member bound and what is asked of the class: its structural envelope, an estimate
//! from source and target evidence, a trial-compatible refinement and estimate, a
//! bounded-outcome confidence set or the replay of one) lowers to a program of the
//! `DiscoveredClassTransport` family composed over the search program: its identity is the
//! search program's identity, the contract identity, the selective data identity of the searched
//! table, the selection targets as a set, the query's canonical encoding, the member bound and
//! the stage with every declared setting and the data identity of every evidence table.
//!
//! [`CompiledDiscoveredClassTransport::execute`] runs the compiled search program on the study's
//! retained table (`CompiledExecution::run`, the adapter run the direct entry makes), then, inside
//! the transport program's own dispatch, reads the class with
//! `transport_discovered_structure_with_context` and runs the stage's kernel on the report:
//! `estimate_discovered_transport`, `estimate_trial_compatible_discovered_transport`,
//! `estimate_trial_compatible_transport_confidence_set` or
//! `verify_trial_compatible_transport_confidence_set_against`, with the same inputs and the
//! caller's execution context. These are the calls the deprecated direct entry
//! `antecedent::analysis::discovered_transport::discover_and_transport` and its Python bridges
//! make, so the outcome, the envelope and every estimate are the ones they return.
//!
//! As in the direct entry, a refusal of the transport request or of the stage does not discard
//! the discovery: both are returned. A refusal of the search itself is the compiled search
//! program's error.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::result_large_err,
    reason = "the run carries the shared discovered-transport refusal type unchanged"
)]

use std::fmt;
use std::sync::Arc;

use antecedent_contract::DataClassification;
use antecedent_core::{
    ByteEstimates, CanonicalEncode, CanonicalWriter, ExecutionContext, PhysicalEffect,
    PhysicalOperator, SemanticDigest, TargetSampling, TransportQuery, VariableId,
};
use antecedent_data::TabularData;
use antecedent_discovery::selective::identity::static_data_identity;
use antecedent_discovery::selective::{
    DiscoveryOutcome, PortableDiscoveryState, SelectiveRunReport, StaticDiscoveryContract,
};
use antecedent_expr::ExactEvaluationLimits;
use antecedent_identify::SidLimits;

use super::discovered_transport::{
    DEFAULT_ENVELOPE_MEMBER_BOUND, DiscoveredTransportError, DiscoveredTransportReport,
    transport_discovered_structure_with_context,
};
use super::discovered_transport_estimate::{
    DiscoveredTransportEstimate, TransportEvidence, discovered_transport_io_failure,
    estimate_discovered_transport,
};
use super::discovered_transport_trial::{
    TransportOutcomeSupport, TrialCompatibilityOptions, TrialCompatibleTransportConfidenceSet,
    TrialCompatibleTransportEstimate, TrialRandomization,
    estimate_trial_compatible_discovered_transport,
    estimate_trial_compatible_transport_confidence_set,
    verify_trial_compatible_transport_confidence_set_against,
};
use super::program_handle::{CompiledProgramHandle, gate_program};
use super::structure_route::{CompiledExecution, DiscoveredStructure, DiscoverySpec};
use super::supplied_identity::{Canonical, Declaration, SetOf, variant};
use crate::analysis::execute::Study;
use crate::error::CausalError;
use crate::planner::lowered::{
    ClaimDemand, Composition, DesignShape, EvidenceDecisionProgram, EvidenceRole,
    EvidenceSourceRef, FunctionalKind, IdentificationLayer, LoweredProgram, PhysicalLayer,
    ProblemTarget, ProgramCell, ProgramIdentityScope, ScientificProblem,
    StandaloneCompositionLoweredParts, UncertaintyReading,
};
use crate::planner::statistical::{
    Realization, ResampleUnit, ResamplingMethod, ResamplingPlan, StatisticalProgram,
};
use crate::planner::{PlanStage, PlanView};

const QUESTION_TAG: &str = "antecedent.discovered_class_transport.question.v1";
const ROWS_TAG: &str = "antecedent.discovered_class_transport.rows.v1";
/// Whether `execute` passes the support gate of the unit the route is bound to
/// (`parity/compiled_units.toml`). It stays off until that unit's record lists this compiled
/// route and is earned again with it, so no valid use refuses before then.
const GATE_ON_UNIT_SUPPORT: bool = false;

/// Source and target rows of an estimate across the class, with the settings of
/// [`TransportEvidence`].
///
/// [`Self::new`] has the defaults of [`TransportEvidence::new`]: independent rows from a
/// representative target sample, no interval, default limits. Both tables hold the searched
/// variables as integer-coded columns, named as in the searched table.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct DiscoveredClassEvidence {
    /// Rows of the source population. The treatment column holds the experimental assignment.
    pub source: TabularData,
    /// Rows of the target population, observational.
    pub target: TabularData,
    /// How the target rows relate to the target population.
    pub target_sampling: TargetSampling,
    /// Whether the rows of each table are independent draws.
    pub rows_independent: bool,
    /// Bootstrap replicates; zero asks for none.
    pub bootstrap_replicates: u32,
    /// Rows per block of a circular block bootstrap, for ordered rows.
    pub block_length: Option<u32>,
    /// Evaluation limits of the retained grid.
    pub limits: ExactEvaluationLimits,
    /// Search limits of the sID derivation run on each member.
    pub sid_limits: SidLimits,
}

impl DiscoveredClassEvidence {
    /// Source and target rows with the defaults of [`TransportEvidence::new`].
    #[must_use]
    pub fn new(source: TabularData, target: TabularData) -> Self {
        let defaults = TransportEvidence::new(&source, &target);
        let (target_sampling, rows_independent, bootstrap_replicates, block_length, limits) = (
            defaults.target_sampling,
            defaults.rows_independent,
            defaults.bootstrap_replicates,
            defaults.block_length,
            defaults.limits,
        );
        let sid_limits = defaults.sid_limits;
        Self {
            source,
            target,
            target_sampling,
            rows_independent,
            bootstrap_replicates,
            block_length,
            limits,
            sid_limits,
        }
    }

    /// The evidence the kernels read, borrowing the tables.
    fn borrowed(&self) -> TransportEvidence<'_> {
        let mut evidence = TransportEvidence::new(&self.source, &self.target);
        evidence.target_sampling = self.target_sampling;
        evidence.rows_independent = self.rows_independent;
        evidence.bootstrap_replicates = self.bootstrap_replicates;
        evidence.block_length = self.block_length;
        evidence.limits = self.limits;
        evidence.sid_limits = self.sid_limits;
        evidence
    }
}

/// What is asked of the class.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum DiscoveredClassStage {
    /// The structural envelope: each member's structural verdict, no number.
    Envelope,
    /// The transported response in every member, summarized as a point or a labelled
    /// structural range (`estimate_discovered_transport`).
    Estimate {
        /// Source and target rows.
        evidence: DiscoveredClassEvidence,
    },
    /// The class refined by the randomized binary source trial, then estimated in every
    /// retained member (`estimate_trial_compatible_discovered_transport`).
    TrialCompatibleEstimate {
        /// Source trial and target rows.
        evidence: DiscoveredClassEvidence,
        /// The level of the shared test and the asserted randomization.
        options: TrialCompatibilityOptions,
    },
    /// The bounded-outcome whole-procedure confidence set of the binary treatment mean
    /// contrast (`estimate_trial_compatible_transport_confidence_set`).
    TrialConfidenceSet {
        /// Source trial and target rows.
        evidence: DiscoveredClassEvidence,
        /// The level of the shared test and the asserted randomization.
        options: TrialCompatibilityOptions,
        /// The caller's support of every potential outcome.
        support: TransportOutcomeSupport,
    },
    /// The confidence set recomputed from the same inputs and checked against a receipt
    /// (`verify_trial_compatible_transport_confidence_set_against`).
    TrialConfidenceSetReplay {
        /// The exported receipt.
        receipt: Vec<u8>,
        /// Source trial and target rows.
        evidence: DiscoveredClassEvidence,
        /// The level of the shared test and the asserted randomization.
        options: TrialCompatibilityOptions,
        /// The caller's support of every potential outcome.
        support: TransportOutcomeSupport,
    },
}

impl DiscoveredClassStage {
    const fn label(&self) -> &'static str {
        match self {
            Self::Envelope => "envelope",
            Self::Estimate { .. } => "estimate",
            Self::TrialCompatibleEstimate { .. } => "trial_compatible_estimate",
            Self::TrialConfidenceSet { .. } => "trial_confidence_set",
            Self::TrialConfidenceSetReplay { .. } => "trial_confidence_set_replay",
        }
    }

    const fn evidence(&self) -> Option<&DiscoveredClassEvidence> {
        match self {
            Self::Envelope => None,
            Self::Estimate { evidence }
            | Self::TrialCompatibleEstimate { evidence, .. }
            | Self::TrialConfidenceSet { evidence, .. }
            | Self::TrialConfidenceSetReplay { evidence, .. } => Some(evidence),
        }
    }
}

/// A transport request over the class a static selective search of `data` returns.
#[derive(Clone, Debug)]
pub struct DiscoveredClassTransportRequest {
    contract: StaticDiscoveryContract,
    data: TabularData,
    selection_targets: Vec<VariableId>,
    query: TransportQuery,
    member_bound: usize,
    stage: DiscoveredClassStage,
}

impl DiscoveredClassTransportRequest {
    /// Search `data` under `contract` and ask `stage` of the class for `query`, with the
    /// caller's `selection_targets` and the default member bound
    /// ([`DEFAULT_ENVELOPE_MEMBER_BOUND`]).
    ///
    /// `data` holds the contract's variables, in its order; the context the program runs
    /// under carries the contract's seed.
    #[must_use]
    pub fn new(
        contract: StaticDiscoveryContract,
        data: TabularData,
        selection_targets: Vec<VariableId>,
        query: TransportQuery,
        stage: DiscoveredClassStage,
    ) -> Self {
        Self {
            contract,
            data,
            selection_targets,
            query,
            member_bound: DEFAULT_ENVELOPE_MEMBER_BOUND,
            stage,
        }
    }

    /// The same request with another bound on the enumeration of the class.
    #[must_use]
    pub fn with_member_bound(mut self, member_bound: usize) -> Self {
        self.member_bound = member_bound;
        self
    }

    /// The searched table.
    #[must_use]
    pub const fn data(&self) -> &TabularData {
        &self.data
    }

    /// The transport query.
    #[must_use]
    pub const fn query(&self) -> &TransportQuery {
        &self.query
    }

    /// What is asked of the class.
    #[must_use]
    pub const fn stage(&self) -> &DiscoveredClassStage {
        &self.stage
    }
}

/// A transport request over a discovered class, bound to its lowered program and to the
/// compiled structure search it is composed over.
#[derive(Clone, Debug)]
pub struct CompiledDiscoveredClassTransport {
    request: DiscoveredClassTransportRequest,
    search: Study,
    search_program: SemanticDigest,
    lowered: Arc<LoweredProgram>,
}

/// What the stage returned.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum DiscoveredClassStageResult {
    /// [`DiscoveredClassStage::Estimate`].
    Estimate(Box<DiscoveredTransportEstimate>),
    /// [`DiscoveredClassStage::TrialCompatibleEstimate`].
    TrialCompatibleEstimate(Box<TrialCompatibleTransportEstimate>),
    /// [`DiscoveredClassStage::TrialConfidenceSet`] and
    /// [`DiscoveredClassStage::TrialConfidenceSetReplay`].
    TrialConfidenceSet(Box<TrialCompatibleTransportConfidenceSet>),
}

/// The discovery and what the transport program returned over its class.
#[derive(Debug)]
#[non_exhaustive]
pub struct DiscoveredClassTransportRun {
    /// What the search did (stage plan, observed work).
    pub discovery: SelectiveRunReport,
    /// The retained discovery outcome the request was made over.
    pub outcome: DiscoveryOutcome,
    /// The portable state a later search resumes from, when the adapter exports one.
    pub state: Option<PortableDiscoveryState>,
    /// Identity of the search program that ran.
    pub discovery_program_identity: Option<SemanticDigest>,
    /// Identity of the transport program.
    pub lowered_program_identity: SemanticDigest,
    /// The envelope, or the typed refusal of the transport request. A refusal here leaves the
    /// discovery above intact.
    pub transport: Result<DiscoveredTransportReport, DiscoveredTransportError>,
    /// The stage's result or refusal; `None` for the envelope stage and when the transport
    /// request was refused.
    pub stage: Option<Result<DiscoveredClassStageResult, DiscoveredTransportError>>,
}

/// Why a compiled transport over a discovered class did not return a run.
#[derive(Debug)]
#[non_exhaustive]
pub enum DiscoveredClassTransportError {
    /// The program cannot run here: a reentry into another program's dispatch, or the support
    /// gate of its unit.
    Program(CausalError),
    /// The compiled structure search refused or failed, with the search program's error.
    Search(CausalError),
}

impl fmt::Display for DiscoveredClassTransportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Program(error) | Self::Search(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for DiscoveredClassTransportError {}

// -- identity -------------------------------------------------------------------------------------

/// A 32-byte digest written as length-prefixed bytes.
struct Digest32([u8; 32]);

impl Canonical for Digest32 {
    fn encode(&self, out: &mut CanonicalWriter) {
        out.bytes(&self.0);
    }
}

/// A value with its own canonical encoding in `antecedent_core`.
struct Encoded<'a, T: CanonicalEncode>(&'a T);

impl<T: CanonicalEncode> Canonical for Encoded<'_, T> {
    fn encode(&self, out: &mut CanonicalWriter) {
        CanonicalEncode::encode(self.0, out);
    }
}

/// The selective data identity of a table's searched columns (rows and digest), or the reason
/// it has none (the kernel that reads the table refuses it when the program runs).
struct RowSet(Result<(u64, [u8; 32]), String>);

impl RowSet {
    fn of(data: &TabularData, variables: &[VariableId]) -> Self {
        Self(
            static_data_identity(data, variables)
                .map(|identity| (identity.rows, identity.digest))
                .map_err(|error| error.to_string()),
        )
    }

    fn rows(&self) -> u64 {
        self.0.as_ref().map_or(0, |(rows, _)| *rows)
    }

    fn digest(&self) -> [u8; 32] {
        let mut writer = CanonicalWriter::new(ROWS_TAG);
        self.encode(&mut writer);
        writer.finish()
    }
}

impl Canonical for RowSet {
    fn encode(&self, out: &mut CanonicalWriter) {
        match &self.0 {
            Ok((rows, digest)) => {
                out.text("rows").word(*rows).bytes(digest);
            }
            Err(error) => {
                out.text("unreadable").text(error);
            }
        }
    }
}

/// Every declared setting of the evidence and the data identity of both tables.
struct EvidenceIdentity<'a> {
    evidence: &'a DiscoveredClassEvidence,
    source: RowSet,
    target: RowSet,
}

impl<'a> EvidenceIdentity<'a> {
    fn of(evidence: &'a DiscoveredClassEvidence, variables: &[VariableId]) -> Self {
        Self {
            evidence,
            source: RowSet::of(&evidence.source, variables),
            target: RowSet::of(&evidence.target, variables),
        }
    }
}

impl Canonical for EvidenceIdentity<'_> {
    fn encode(&self, out: &mut CanonicalWriter) {
        let evidence = self.evidence;
        self.source.encode(out);
        self.target.encode(out);
        Encoded(&evidence.target_sampling).encode(out);
        evidence.rows_independent.encode(out);
        evidence.bootstrap_replicates.encode(out);
        evidence.block_length.encode(out);
        out.word(evidence.limits.operations as u64).word(evidence.limits.depth as u64);
        out.word(evidence.sid_limits.steps as u64).word(evidence.sid_limits.depth as u64);
    }
}

struct Options(TrialCompatibilityOptions);

impl Canonical for Options {
    fn encode(&self, out: &mut CanonicalWriter) {
        out.real(self.0.alpha);
        match self.0.randomization {
            TrialRandomization::CompletelyRandomized => {
                variant(out, "completely_randomized");
            }
        }
    }
}

struct Support(TransportOutcomeSupport);

impl Canonical for Support {
    fn encode(&self, out: &mut CanonicalWriter) {
        out.real(self.0.lower).real(self.0.upper);
    }
}

/// The stage, its settings and the identity of its evidence.
struct StageIdentity<'a> {
    stage: &'a DiscoveredClassStage,
    evidence: Option<EvidenceIdentity<'a>>,
}

impl Canonical for StageIdentity<'_> {
    fn encode(&self, out: &mut CanonicalWriter) {
        variant(out, self.stage.label());
        if let Some(evidence) = &self.evidence {
            evidence.encode(out);
        }
        match self.stage {
            DiscoveredClassStage::Envelope | DiscoveredClassStage::Estimate { .. } => {}
            DiscoveredClassStage::TrialCompatibleEstimate { options, .. } => {
                Options(*options).encode(out);
            }
            DiscoveredClassStage::TrialConfidenceSet { options, support, .. } => {
                Options(*options).encode(out);
                Support(*support).encode(out);
            }
            DiscoveredClassStage::TrialConfidenceSetReplay {
                receipt, options, support, ..
            } => {
                Options(*options).encode(out);
                Support(*support).encode(out);
                out.bytes(receipt);
            }
        }
    }
}

/// A contract's identity, or the refusal that prevents one.
struct ContractIdentity<'a>(&'a StaticDiscoveryContract);

impl Canonical for ContractIdentity<'_> {
    fn encode(&self, out: &mut CanonicalWriter) {
        match self.0.identity() {
            Ok(digest) => {
                out.text("contract").bytes(&digest);
            }
            Err(error) => {
                out.text("unencodable").text(&error.to_string());
            }
        }
    }
}

// -- lowering -------------------------------------------------------------------------------------

fn plan_id(stage: &DiscoveredClassStage) -> String {
    format!("discovered_class_transport.{}.v1", stage.label())
}

fn evaluator(stage: &DiscoveredClassStage) -> &'static str {
    match stage {
        DiscoveredClassStage::Envelope => {
            "antecedent::analysis::discovered_transport::transport_discovered_structure_with_context"
        }
        DiscoveredClassStage::Estimate { .. } => {
            "antecedent::analysis::discovered_transport_estimate::estimate_discovered_transport"
        }
        DiscoveredClassStage::TrialCompatibleEstimate { .. } => {
            "antecedent::analysis::discovered_transport_trial::estimate_trial_compatible_discovered_transport"
        }
        DiscoveredClassStage::TrialConfidenceSet { .. } => {
            "antecedent::analysis::discovered_transport_trial::estimate_trial_compatible_transport_confidence_set"
        }
        DiscoveredClassStage::TrialConfidenceSetReplay { .. } => {
            "antecedent::analysis::discovered_transport_trial::verify_trial_compatible_transport_confidence_set_against"
        }
    }
}

/// The bootstrap the stage's kernel runs on its evidence, if it runs one.
fn resampling_plan(evidence: &DiscoveredClassEvidence) -> Option<ResamplingPlan> {
    let replicates = evidence.bootstrap_replicates;
    if replicates == 0 {
        return None;
    }
    Some(match evidence.block_length {
        Some(_) => ResamplingPlan {
            method: ResamplingMethod::BlockBootstrap,
            replicates,
            unit: ResampleUnit::CalendarBlock,
        },
        None => ResamplingPlan {
            method: ResamplingMethod::Bootstrap,
            replicates,
            unit: ResampleUnit::Row,
        },
    })
}

/// The read of the searched and evidence rows and the transport request over the class.
fn physical_layer(
    plan_id: &str,
    rows: u64,
    columns: u64,
    design: DesignShape,
) -> Result<PhysicalLayer, CausalError> {
    let operator = |effect: PhysicalEffect, bytes: Option<u64>| {
        let declared = PhysicalOperator::declare(effect);
        let declared = match bytes {
            Some(bytes) => declared.estimates(ByteEstimates::absent().with_output(bytes)),
            None => declared,
        };
        declared.build().map_err(|_| CausalError::Compile {
            message: format!("{plan_id}: a physical operator cannot be declared"),
        })
    };
    Ok(PhysicalLayer {
        operators: Arc::from([
            operator(PhysicalEffect::Read, Some(rows.saturating_mul(columns).saturating_mul(8)))?,
            operator(PhysicalEffect::Transport, None)?,
        ]),
        design,
    })
}

fn lower(
    request: &DiscoveredClassTransportRequest,
    search_program: SemanticDigest,
) -> Result<Arc<LoweredProgram>, CausalError> {
    let variables = request.contract.variables.as_slice();
    let searched = RowSet::of(&request.data, variables);
    let stage = StageIdentity {
        stage: &request.stage,
        evidence: request
            .stage
            .evidence()
            .map(|evidence| EvidenceIdentity::of(evidence, variables)),
    };
    let mut question = Declaration::new(QUESTION_TAG);
    question
        .field("search_program", &Digest32(*search_program.as_bytes()))
        .field("contract", &ContractIdentity(&request.contract))
        .field("searched_rows", &searched)
        .field("selection_targets", &SetOf(request.selection_targets.as_slice()))
        .field("query", &Encoded(&request.query))
        .field("member_bound", &request.member_bound)
        .field("stage", &stage);
    let question = question.finish();

    let columns = variables.len() as u64;
    let mut rows = searched.rows();
    let mut sources = vec![
        EvidenceSourceRef {
            role: EvidenceRole::DiscoveredStructure,
            digest: Some(*search_program.as_bytes()),
        },
        EvidenceSourceRef { role: EvidenceRole::TransportSource, digest: Some(question) },
    ];
    let mut resampling = None;
    if let Some(evidence) = &stage.evidence {
        rows = rows.saturating_add(evidence.source.rows()).saturating_add(evidence.target.rows());
        let source_role = match request.stage {
            DiscoveredClassStage::Estimate { .. } => EvidenceRole::Observations,
            _ => EvidenceRole::Trial,
        };
        sources
            .push(EvidenceSourceRef { role: source_role, digest: Some(evidence.source.digest()) });
        sources.push(EvidenceSourceRef {
            role: EvidenceRole::Observations,
            digest: Some(evidence.target.digest()),
        });
        resampling = resampling_plan(evidence.evidence);
    }
    let design = DesignShape::quoted(rows, columns, true);
    let plan_id = plan_id(&request.stage);
    let scientific = ScientificProblem {
        runtime_query: None,
        identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::InterventionalDistribution,
            outcome: None,
            query: question,
        },
        plan_id: Arc::from(plan_id.as_str()),
        data: DataClassification::Tabular,
        graph_class: "markov_equivalence_class_of_a_static_selective_search",
        query_variables: Arc::from(variables),
        source_digest: Some(searched.digest()),
        support_coordinate: None,
        supported: false,
        support_status: None,
    };
    let statistical = StatisticalProgram {
        identifier: None,
        estimator: None,
        realization: Realization::ExactPoint { evaluator: Arc::from(evaluator(&request.stage)) },
        models: Arc::from([]),
        folds: None,
        resampling,
        posterior: None,
        design,
        operators: Arc::from([]),
    };
    let stages = if matches!(request.stage, DiscoveredClassStage::Envelope) { 2 } else { 3 };
    let evidence = EvidenceDecisionProgram {
        sources: Arc::from(sources),
        composition: Composition::Chained { stages },
        decision: None,
        planning: None,
        chain: None,
    };
    let physical = physical_layer(&plan_id, rows, columns, design)?;
    let identification = IdentificationLayer::PerExecution {
        reason: "every member of the discovered class is identified each time the program runs",
    };
    let uncertainty =
        if resampling.is_some() { UncertaintyReading::Sampling } else { UncertaintyReading::None };
    Ok(Arc::new(
        StandaloneCompositionLoweredParts {
            cell: ProgramCell::DiscoveredClassTransport,
            scientific,
            identification,
            statistical,
            evidence,
            physical,
            claim: ClaimDemand { uncertainty, kind: None },
        }
        .finish(),
    ))
}

impl Study {
    /// Compile the structure search of `request` and bind the transport request over its class.
    ///
    /// # Errors
    ///
    /// `public_builder_reentry` inside another program's lowered dispatch; the refusals of the
    /// selective contract (`DiscoverySpec::selective_static`), of the study build and of
    /// [`Study::compile`] for the search; a physical operator that cannot be declared. The
    /// transport request and its evidence are checked by their kernels when the program runs.
    pub fn compile_discovered_class_transport(
        request: DiscoveredClassTransportRequest,
        ctx: &ExecutionContext,
    ) -> Result<CompiledDiscoveredClassTransport, CausalError> {
        crate::analysis::lowering::refuse_reentry(ctx)?;
        let spec = DiscoverySpec::selective_static(request.contract.clone())?;
        let search = Study::tabular(request.data.clone())
            .discovery(spec, request.contract.variables.clone())
            .build()?;
        let plan = search.compile(ctx)?;
        let search_program = plan.lowered().map(LoweredProgram::identity).ok_or_else(|| {
            crate::compile_reason!(
                "invalid_argument",
                "the compiled structure search carries no lowered program"
            )
        })?;
        let lowered = lower(&request, search_program)?;
        Ok(CompiledDiscoveredClassTransport { request, search, search_program, lowered })
    }
}

impl CompiledProgramHandle for CompiledDiscoveredClassTransport {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.lowered
    }
}

impl CompiledDiscoveredClassTransport {
    /// Inspect all six compiler layers.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }

    /// Inspect the plan, including the support decision of its unit, without evaluating.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }

    /// The bound request.
    #[must_use]
    pub const fn request(&self) -> &DiscoveredClassTransportRequest {
        &self.request
    }

    /// The study whose compiled structure search the program is composed over.
    #[must_use]
    pub const fn search(&self) -> &Study {
        &self.search
    }

    /// Identity of the compiled structure search program.
    #[must_use]
    pub const fn search_program_identity(&self) -> SemanticDigest {
        self.search_program
    }

    /// Run the structure search, then the transport request and its stage over the class.
    ///
    /// The context's master seed must be the contract's seed, as for the search itself.
    ///
    /// # Errors
    ///
    /// [`DiscoveredClassTransportError::Program`] inside another program's dispatch or, once it
    /// is on, from the support gate; [`DiscoveredClassTransportError::Search`] when the
    /// compiled search refuses, fails or is cancelled. A refusal of the transport request or
    /// of the stage is returned inside the run.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<DiscoveredClassTransportRun, DiscoveredClassTransportError> {
        crate::analysis::lowering::refuse_reentry(ctx)
            .map_err(DiscoveredClassTransportError::Program)?;
        // The search program runs first, outside this program's dispatch: it is a program of
        // its own, prepared and executed by the compiler on the study's retained table.
        let execution = CompiledExecution::run(&self.search, ctx)
            .map_err(DiscoveredClassTransportError::Search)?;
        let found = match execution.result().discovery().map(|payload| &payload.structure) {
            Some(DiscoveredStructure::Selective(found)) => found.as_ref().clone(),
            _ => {
                return Err(DiscoveredClassTransportError::Search(crate::compile_reason!(
                    "invalid_argument",
                    "the compiled structure search returned no selective outcome"
                )));
            }
        };
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        if GATE_ON_UNIT_SUPPORT {
            gate_program(&scoped, self).map_err(|refusal| {
                DiscoveredClassTransportError::Program(CausalError::from(refusal))
            })?;
        }
        let request = &self.request;
        let transport = transport_discovered_structure_with_context(
            &found.outcome,
            &request.selection_targets,
            &request.query,
            request.member_bound,
            &scoped,
        );
        let stage = transport.as_ref().ok().and_then(|report| self.run_stage(report, &scoped));
        Ok(DiscoveredClassTransportRun {
            discovery: found.report,
            outcome: found.outcome,
            state: found.state,
            discovery_program_identity: execution.lowered_identity(),
            lowered_program_identity: self.lowered.identity(),
            transport,
            stage,
        })
    }

    /// The stage's kernel on the report, with the same inputs the direct bridges pass.
    fn run_stage(
        &self,
        report: &DiscoveredTransportReport,
        ctx: &ExecutionContext,
    ) -> Option<Result<DiscoveredClassStageResult, DiscoveredTransportError>> {
        let query = &self.request.query;
        Some(match &self.request.stage {
            DiscoveredClassStage::Envelope => return None,
            DiscoveredClassStage::Estimate { evidence } => {
                estimate_discovered_transport(report, query, &evidence.borrowed(), ctx)
                    .map(|estimate| DiscoveredClassStageResult::Estimate(Box::new(estimate)))
            }
            DiscoveredClassStage::TrialCompatibleEstimate { evidence, options } => {
                estimate_trial_compatible_discovered_transport(
                    report,
                    query,
                    &evidence.borrowed(),
                    *options,
                    ctx,
                )
                .map(|estimate| {
                    DiscoveredClassStageResult::TrialCompatibleEstimate(Box::new(estimate))
                })
            }
            DiscoveredClassStage::TrialConfidenceSet { evidence, options, support } => {
                estimate_trial_compatible_transport_confidence_set(
                    report,
                    query,
                    &evidence.borrowed(),
                    *options,
                    *support,
                    ctx,
                )
                .map(|set| DiscoveredClassStageResult::TrialConfidenceSet(Box::new(set)))
            }
            DiscoveredClassStage::TrialConfidenceSetReplay {
                receipt,
                evidence,
                options,
                support,
            } => verify_trial_compatible_transport_confidence_set_against(
                receipt,
                report,
                query,
                &evidence.borrowed(),
                *options,
                *support,
                ctx,
            )
            .map_err(discovered_transport_io_failure)
            .map(|set| DiscoveredClassStageResult::TrialConfidenceSet(Box::new(set))),
        })
    }
}
