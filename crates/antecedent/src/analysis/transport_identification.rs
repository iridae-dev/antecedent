//! Compiled transport and recovery identification.
//!
//! Each theorem-scoped transport identifier (classical sID, the finite evidence-catalog search,
//! meta-transport, z-transport and its two-source combination, multi-source mz-transport, the
//! mixed-source proof search, conditional transport over a selection ADMG and general finite
//! counterfactual transport) and each observation-recovery decision (one partially observed
//! pattern law, an ordered response chain) is a compiled program: the request lowers to a
//! six-layer program whose cell is bound to the support unit of the capability, the retained
//! handle passes that unit's support gate inside its own dispatch, and execution calls the same
//! kernel function with the same inputs as the direct call. The kernel result is returned
//! unchanged. Binding a derivation to an evidence catalog is its own request, run by the
//! kernel's own binder, so a derivation decided once binds to a later catalog exactly as it does
//! through the direct binder.
//!
//! Identification computes no estimate: the identified functional is evaluated by the route that
//! consumes it. The program claims no uncertainty.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;
use std::sync::Arc;

use antecedent_contract::DataClassification;
use antecedent_core::{
    CanonicalEncode, CanonicalWriter, EvidenceCatalog, ExecutionContext, NotSupported,
    PhysicalEffect, PhysicalOperator, SearchLimits, VariableId,
};
use antecedent_graph::{Admg, SelectionDiagram};
use antecedent_identify::{
    BoundMixedSourceFunctional, BoundMzTransportFunctional, BoundTransportFunctional,
    BoundZTransportFunctional, CatalogTransportResult, ChainRecoveryDecision, ChainRecoveryError,
    ChainRecoveryQuery, ClassicalTransportDerivation, ClassicalTransportQuery,
    ClassicalTransportResult, ConditionalTransportDecision, ConditionalTransportQuery,
    CounterfactualTransportDecision, CounterfactualTransportProblem, CounterfactualTransportQuery,
    IdentificationError, MetaTransportQuery, MixedSourceDecision, MixedSourceDerivation,
    MixedSourceQuery, MzTransportDecision, MzTransportDerivation, MzTransportQuery,
    ObservationRecoveryQuery, RecoveredEffectQuery, RecoveryDecision, RecoveryError,
    RecoveryLimits, SidLimits, TwoSourceZTransportDecision, TwoSourceZTransportQuery,
    ZTransportDerivation, ZTransportOutcome, ZTransportQuery, ZTransportResult,
};
use antecedent_io::admg_conditional_transport_artifact::AdmgConditionalQueryWire;
use antecedent_io::recovery_artifact::RecoveryQueryWire;
use antecedent_io::recovery_chain_artifact::ChainQueryWire;
use antecedent_io::transport_query_wire::{
    MixedSourceQueryWire, MzTransportQueryWire, ZTransportQueryWire,
};
use antecedent_io_core::counterfactual_transport_wire::{
    CounterfactualTransportProblemWire, CounterfactualTransportQueryWire,
};

use super::program_handle::{CompiledProgramHandle, gate_program};
use super::supplied_identity::{Canonical, SetOf, Wire, encoded};
use crate::planner::PlanView;
use crate::planner::lowered::{
    ClaimDemand, Composition, DesignShape, EvidenceDecisionProgram, EvidenceRole,
    EvidenceSourceRef, FunctionalKind, IdentificationLayer, LoweredProgram, PhysicalLayer,
    ProblemTarget, ProgramCell, ProgramIdentityScope, ScientificProblem,
    StandaloneCompositionLoweredParts, UncertaintyReading,
};
use crate::planner::statistical::{Realization, StatisticalProgram};

/// One transport or observation-recovery identification question, with every input its
/// kernel identifier reads.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum TransportIdentificationRequest {
    /// Classical sID under the declared experimental information family
    /// (`antecedent_identify::identify_classical_transport`).
    Classical {
        /// Selection diagram.
        diagram: SelectionDiagram,
        /// Outcomes, treatments, source and target.
        query: ClassicalTransportQuery,
        /// Search limits.
        limits: SidLimits,
    },
    /// Classical sID searched against a finite evidence catalog
    /// (`antecedent_identify::identify_catalog_transport`).
    Catalog {
        /// Selection diagram.
        diagram: SelectionDiagram,
        /// Outcomes, treatments, source and target.
        query: ClassicalTransportQuery,
        /// Supplied evidence.
        catalog: EvidenceCatalog,
        /// Search limits.
        limits: SidLimits,
    },
    /// An sID derivation searched for its alternatives against a finite evidence catalog
    /// (`ClassicalTransportDerivation::search_catalog`).
    CatalogSearch {
        /// The derivation, from a classical decision.
        derivation: Box<ClassicalTransportDerivation>,
        /// Selection diagram the derivation was checked against.
        diagram: SelectionDiagram,
        /// Supplied evidence.
        catalog: EvidenceCatalog,
        /// Search limits.
        limits: SidLimits,
    },
    /// An sID derivation bound to a finite evidence catalog
    /// (`ClassicalTransportDerivation::bind_catalog_with_context`).
    CatalogBinding {
        /// The derivation, from a classical decision.
        derivation: Box<ClassicalTransportDerivation>,
        /// Supplied evidence.
        catalog: EvidenceCatalog,
        /// Search limits of the verification replay.
        limits: SidLimits,
    },
    /// Classical meta-transport from a collection of source populations
    /// (`antecedent_identify::identify_meta_transport`).
    Meta {
        /// Causal graph shared by the populations.
        graph: Admg,
        /// Outcomes, treatments, target and the declared sources.
        query: MetaTransportQuery,
        /// Search limits.
        limits: SidLimits,
    },
    /// Single-source z-transport (`antecedent_identify::identify_z_transport`).
    ZTransport {
        /// Selection diagram.
        diagram: SelectionDiagram,
        /// The z-transport query.
        query: ZTransportQuery,
        /// Search limits.
        limits: SidLimits,
    },
    /// A z-transport derivation bound to a catalog
    /// (`antecedent_identify::bind_z_transport_catalog`).
    ZTransportBinding {
        /// Selection diagram the derivation was checked against.
        diagram: SelectionDiagram,
        /// The z-transport query.
        query: ZTransportQuery,
        /// The derivation, from a z-transport decision.
        derivation: Box<ZTransportDerivation>,
        /// Supplied evidence.
        catalog: EvidenceCatalog,
    },
    /// Single-source z-transport decided against a catalog with its full inspection
    /// (`antecedent_identify::decide_z_transport_inspecting`).
    ZTransportInspecting {
        /// Selection diagram.
        diagram: SelectionDiagram,
        /// The z-transport query.
        query: ZTransportQuery,
        /// Supplied evidence.
        catalog: EvidenceCatalog,
        /// Search limits.
        limits: SidLimits,
    },
    /// Two-source z-transport and its complementary combination
    /// (`antecedent_identify::decide_two_source_z_transport`).
    TwoSourceZTransport {
        /// Causal graph shared by the populations.
        graph: Admg,
        /// The two-source query.
        query: TwoSourceZTransportQuery,
        /// One catalog per declared source, in source order.
        catalogs: [EvidenceCatalog; 2],
        /// Search limits.
        limits: SidLimits,
    },
    /// Multi-source mz-transport (`antecedent_identify::decide_mz_transport`).
    MzTransport {
        /// Causal graph shared by the populations.
        graph: Admg,
        /// The multi-source query.
        query: MzTransportQuery,
        /// Supplied evidence.
        catalog: EvidenceCatalog,
        /// Search limits.
        limits: SearchLimits,
    },
    /// An mz-transport derivation bound to a catalog
    /// (`antecedent_identify::bind_mz_transport_catalog`).
    MzTransportBinding {
        /// Causal graph the derivation was checked against.
        graph: Admg,
        /// The derivation, from an mz-transport decision.
        derivation: Box<MzTransportDerivation>,
        /// Supplied evidence.
        catalog: EvidenceCatalog,
    },
    /// Mixed-source proof search (`antecedent_identify::decide_mixed_source`).
    MixedSource {
        /// Causal graph of the target population.
        graph: Admg,
        /// The mixed-source query.
        query: MixedSourceQuery,
        /// Supplied evidence.
        catalog: EvidenceCatalog,
        /// Search limits.
        limits: SearchLimits,
    },
    /// A mixed-source derivation bound to a catalog
    /// (`antecedent_identify::bind_mixed_source_catalog`).
    MixedSourceBinding {
        /// Causal graph the derivation was checked against.
        graph: Admg,
        /// The derivation, from a mixed-source decision.
        derivation: Box<MixedSourceDerivation>,
        /// Supplied evidence.
        catalog: EvidenceCatalog,
    },
    /// Conditional transport over a latent-confounded selection ADMG
    /// (`antecedent_identify::decide_admg_conditional_transport`).
    AdmgConditional {
        /// Selection diagram.
        diagram: SelectionDiagram,
        /// The conditional query.
        query: ConditionalTransportQuery,
        /// Supplied evidence.
        catalog: EvidenceCatalog,
        /// Search limits.
        limits: SearchLimits,
    },
    /// General finite counterfactual transport
    /// (`antecedent_identify::identify_counterfactual_transport`).
    Counterfactual {
        /// The finite problem: variables, levels and edges.
        problem: CounterfactualTransportProblem,
        /// The counterfactual query and its sources.
        query: CounterfactualTransportQuery,
        /// Supplied evidence.
        catalog: EvidenceCatalog,
        /// Search limits.
        limits: SearchLimits,
    },
    /// Recovery of a law from one partially observed pattern law
    /// (`antecedent_identify::decide_observation_recovery`).
    ObservationRecovery {
        /// The m-graph.
        graph: Admg,
        /// The recovery query.
        query: ObservationRecoveryQuery,
        /// Supplied evidence.
        catalog: EvidenceCatalog,
        /// A downstream effect identified from the recovered law, when requested.
        effect: Option<RecoveredEffectQuery>,
        /// Recovery limits.
        limits: RecoveryLimits,
    },
    /// Recovery of an ordered response chain (`antecedent_identify::decide_chain_recovery`).
    ChainRecovery {
        /// The m-graph.
        graph: Admg,
        /// The chain query.
        query: ChainRecoveryQuery,
    },
}

/// What the kernel identifier returned, unchanged.
#[derive(Clone, Debug)]
#[non_exhaustive]
#[allow(
    clippy::large_enum_variant,
    reason = "each outcome is the kernel's own result, returned once and unchanged"
)]
pub enum TransportIdentificationOutcome {
    /// [`TransportIdentificationRequest::Classical`].
    Classical(ClassicalTransportResult),
    /// [`TransportIdentificationRequest::Catalog`].
    Catalog(CatalogTransportResult),
    /// [`TransportIdentificationRequest::CatalogSearch`].
    CatalogSearch(CatalogTransportResult),
    /// [`TransportIdentificationRequest::CatalogBinding`].
    CatalogBinding(Box<BoundTransportFunctional>),
    /// [`TransportIdentificationRequest::Meta`].
    Meta(ClassicalTransportResult),
    /// [`TransportIdentificationRequest::ZTransport`].
    ZTransport(ZTransportResult),
    /// [`TransportIdentificationRequest::ZTransportBinding`].
    ZTransportBinding(Box<BoundZTransportFunctional>),
    /// [`TransportIdentificationRequest::ZTransportInspecting`].
    ZTransportInspecting(ZTransportOutcome),
    /// [`TransportIdentificationRequest::TwoSourceZTransport`].
    TwoSourceZTransport(TwoSourceZTransportDecision),
    /// [`TransportIdentificationRequest::MzTransport`].
    MzTransport(MzTransportDecision),
    /// [`TransportIdentificationRequest::MzTransportBinding`].
    MzTransportBinding(Box<BoundMzTransportFunctional>),
    /// [`TransportIdentificationRequest::MixedSource`].
    MixedSource(MixedSourceDecision),
    /// [`TransportIdentificationRequest::MixedSourceBinding`].
    MixedSourceBinding(Box<BoundMixedSourceFunctional>),
    /// [`TransportIdentificationRequest::AdmgConditional`].
    AdmgConditional(ConditionalTransportDecision),
    /// [`TransportIdentificationRequest::Counterfactual`].
    Counterfactual(CounterfactualTransportDecision),
    /// [`TransportIdentificationRequest::ObservationRecovery`].
    ObservationRecovery(RecoveryDecision),
    /// [`TransportIdentificationRequest::ChainRecovery`].
    ChainRecovery(ChainRecoveryDecision),
}

/// Why a compiled identification did not return an outcome.
#[derive(Debug)]
#[non_exhaustive]
pub enum TransportIdentificationError {
    /// The release support gate of the route's unit refused it.
    NotSupported(NotSupported),
    /// The compiled program is inside another program's lowered dispatch.
    Reentry,
    /// A transport identifier refused, failed or was cancelled.
    Identification(IdentificationError),
    /// The observation-recovery decision refused or failed.
    Recovery(RecoveryError),
    /// The chain-recovery decision refused or failed.
    ChainRecovery(ChainRecoveryError),
}

/// Refusal detail of [`TransportIdentificationError::Reentry`] (reason code
/// `public_builder_reentry`).
pub const REENTRY_DETAIL: &str = antecedent_core::reason_details::transport_identification::REENTRY;

impl fmt::Display for TransportIdentificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotSupported(refusal) => write!(f, "{refusal}"),
            Self::Reentry => write!(
                f,
                "public_builder_reentry ({REENTRY_DETAIL}): a lowered dispatch executes its \
                 retained program; it does not compile or run another identification"
            ),
            Self::Identification(error) => write!(f, "{error}"),
            Self::Recovery(error) => write!(f, "{error}"),
            Self::ChainRecovery(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for TransportIdentificationError {}

impl From<IdentificationError> for TransportIdentificationError {
    fn from(error: IdentificationError) -> Self {
        Self::Identification(error)
    }
}

/// A compiled identification: the request and the lowered program it retains.
#[derive(Clone, Debug)]
pub struct CompiledTransportIdentification {
    request: TransportIdentificationRequest,
    lowered: Arc<LoweredProgram>,
}

impl CompiledProgramHandle for CompiledTransportIdentification {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.lowered
    }
}

/// Compile one identification question into its retained program.
///
/// Compiling checks nothing the kernel identifier does not check itself; a malformed request
/// is refused by the identifier when the program runs, with its own typed error.
///
/// # Errors
/// [`TransportIdentificationError::Reentry`] inside another program's lowered dispatch.
pub fn compile_transport_identification(
    request: TransportIdentificationRequest,
    ctx: &ExecutionContext,
) -> Result<CompiledTransportIdentification, TransportIdentificationError> {
    if ctx.lowered_dispatch().is_some() {
        return Err(TransportIdentificationError::Reentry);
    }
    let lowered = Arc::new(lower(&request)?);
    Ok(CompiledTransportIdentification { request, lowered })
}

impl CompiledTransportIdentification {
    /// The request the program was compiled from.
    #[must_use]
    pub const fn request(&self) -> &TransportIdentificationRequest {
        &self.request
    }

    /// Every layer of the retained program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }

    /// The plan view of the retained program, as compiled.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        CompiledProgramHandle::plan_view(self)
    }

    /// Run the kernel identifier of the request inside the program's dispatch, after the
    /// support gate of the route's unit.
    ///
    /// # Errors
    /// The gate's refusal in a release build, reentry, or the identifier's own error.
    pub fn identify(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<TransportIdentificationOutcome, TransportIdentificationError> {
        if ctx.lowered_dispatch().is_some() {
            return Err(TransportIdentificationError::Reentry);
        }
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        if gate_on_unit_support(cell_of(&self.request)) {
            gate_program(&scoped, self).map_err(TransportIdentificationError::NotSupported)?;
        }
        run(&self.request, &scoped)
    }
}

/// Whether `identify` passes the support gate of the unit the cell is bound to
/// (`parity/compiled_units.toml`). A cell whose unit has earned its support gates now; the two
/// cells whose records are new stay ungated until those records are earned, so no valid use
/// refuses before then. Turning a cell on makes a release build refuse it while its unit is
/// unsupported, and a development build record what it would refuse.
const fn gate_on_unit_support(cell: ProgramCell) -> bool {
    !matches!(
        cell,
        ProgramCell::ClassicalTransportIdentification | ProgramCell::ZTransportIdentification
    )
}

/// The program family a request lowers to; each is bound to one support unit.
const fn cell_of(request: &TransportIdentificationRequest) -> ProgramCell {
    use TransportIdentificationRequest as R;
    match request {
        R::Classical { .. }
        | R::Catalog { .. }
        | R::CatalogSearch { .. }
        | R::CatalogBinding { .. }
        | R::Meta { .. } => ProgramCell::ClassicalTransportIdentification,
        R::ZTransport { .. }
        | R::ZTransportBinding { .. }
        | R::ZTransportInspecting { .. }
        | R::TwoSourceZTransport { .. } => ProgramCell::ZTransportIdentification,
        R::MzTransport { .. } | R::MzTransportBinding { .. } => {
            ProgramCell::MzTransportIdentification
        }
        R::MixedSource { .. } | R::MixedSourceBinding { .. } => {
            ProgramCell::MixedSourceIdentification
        }
        R::AdmgConditional { .. } => ProgramCell::AdmgConditionalTransportIdentification,
        R::Counterfactual { .. } => ProgramCell::CounterfactualTransportIdentification,
        R::ObservationRecovery { .. } => ProgramCell::ObservationRecoveryIdentification,
        R::ChainRecovery { .. } => ProgramCell::ChainRecoveryIdentification,
    }
}

/// The kernel call of each request: the same function, with the same inputs, as the direct
/// path.
fn run(
    request: &TransportIdentificationRequest,
    ctx: &ExecutionContext,
) -> Result<TransportIdentificationOutcome, TransportIdentificationError> {
    use TransportIdentificationOutcome as O;
    use TransportIdentificationRequest as R;
    Ok(match request {
        R::Classical { diagram, query, limits } => O::Classical(
            antecedent_identify::identify_classical_transport(diagram, query, *limits, ctx)?,
        ),
        R::Catalog { diagram, query, catalog, limits } => O::Catalog(
            antecedent_identify::identify_catalog_transport(diagram, query, catalog, *limits, ctx)?,
        ),
        R::CatalogSearch { derivation, diagram, catalog, limits } => {
            O::CatalogSearch(derivation.search_catalog(diagram, catalog, *limits, ctx)?)
        }
        R::CatalogBinding { derivation, catalog, limits } => O::CatalogBinding(Box::new(
            derivation.bind_catalog_with_context(catalog, *limits, ctx)?,
        )),
        R::Meta { graph, query, limits } => {
            O::Meta(antecedent_identify::identify_meta_transport(graph, query, *limits, ctx)?)
        }
        R::ZTransport { diagram, query, limits } => {
            O::ZTransport(antecedent_identify::identify_z_transport(diagram, query, *limits, ctx)?)
        }
        R::ZTransportBinding { diagram, query, derivation, catalog } => {
            O::ZTransportBinding(Box::new(antecedent_identify::bind_z_transport_catalog(
                diagram, query, derivation, catalog,
            )?))
        }
        R::ZTransportInspecting { diagram, query, catalog, limits } => {
            O::ZTransportInspecting(antecedent_identify::decide_z_transport_inspecting(
                diagram, query, catalog, *limits, ctx,
            )?)
        }
        R::TwoSourceZTransport { graph, query, catalogs, limits } => {
            O::TwoSourceZTransport(antecedent_identify::decide_two_source_z_transport(
                graph,
                query,
                [&catalogs[0], &catalogs[1]],
                *limits,
                ctx,
            )?)
        }
        R::MzTransport { graph, query, catalog, limits } => O::MzTransport(
            antecedent_identify::decide_mz_transport(graph, query, catalog, *limits, ctx)?,
        ),
        R::MzTransportBinding { graph, derivation, catalog } => O::MzTransportBinding(Box::new(
            antecedent_identify::bind_mz_transport_catalog(graph, derivation, catalog)?,
        )),
        R::MixedSource { graph, query, catalog, limits } => O::MixedSource(
            antecedent_identify::decide_mixed_source(graph, query, catalog, *limits, ctx)?,
        ),
        R::MixedSourceBinding { graph, derivation, catalog } => O::MixedSourceBinding(Box::new(
            antecedent_identify::bind_mixed_source_catalog(graph, derivation, catalog)?,
        )),
        R::AdmgConditional { diagram, query, catalog, limits } => {
            O::AdmgConditional(antecedent_identify::decide_admg_conditional_transport(
                diagram, query, catalog, *limits, ctx,
            )?)
        }
        R::Counterfactual { problem, query, catalog, limits } => {
            O::Counterfactual(antecedent_identify::identify_counterfactual_transport(
                problem, query, catalog, *limits, ctx,
            )?)
        }
        R::ObservationRecovery { graph, query, catalog, effect, limits } => O::ObservationRecovery(
            antecedent_identify::decide_observation_recovery(
                graph,
                query,
                catalog,
                effect.as_ref(),
                *limits,
                ctx,
            )
            .map_err(TransportIdentificationError::Recovery)?,
        ),
        R::ChainRecovery { graph, query } => O::ChainRecovery(
            antecedent_identify::decide_chain_recovery(graph, query, ctx)
                .map_err(TransportIdentificationError::ChainRecovery)?,
        ),
    })
}

// --- lowering ------------------------------------------------------------------------------

/// An ADMG by its canonical identity: sorted directed and bidirected edge lists, then every
/// dense node's label (the queries name variables). The graph's own `Debug` is never encoded.
fn encode_graph(writer: &mut CanonicalWriter, graph: &Admg) {
    Canonical::encode(graph, writer);
}

/// A selection diagram: its graph, then its selection targets as a set (identification reads
/// them as a node set, so their order and repetitions do not enter the identity).
fn encode_diagram(writer: &mut CanonicalWriter, diagram: &SelectionDiagram) {
    encode_graph(writer, diagram.causal_graph());
    Canonical::encode(&SetOf(diagram.selection_targets()), writer);
}

fn encode_catalog(writer: &mut CanonicalWriter, catalog: &EvidenceCatalog) {
    CanonicalEncode::encode(catalog, writer);
}

/// A wire form (a query, a finite problem or a derivation record) by its CBOR encoding. Every
/// wire written here is a serde type of scalars and ordered lists with no hash-ordered map;
/// a derivation is encoded by its record, never by the arena it interns into.
fn encode_wire(writer: &mut CanonicalWriter, wire: &impl serde::Serialize) {
    Canonical::encode(&Wire(wire), writer);
}

/// Variable ids in declared order.
fn encode_ids(writer: &mut CanonicalWriter, ids: &[VariableId]) {
    writer.u32s(&ids.iter().map(|id| id.raw()).collect::<Vec<_>>());
}

/// A classical query: outcomes and treatments in declared order, source and target.
fn encode_classical(writer: &mut CanonicalWriter, query: &ClassicalTransportQuery) {
    encode_ids(writer, &query.outcomes);
    encode_ids(writer, &query.treatments);
    writer.text(&query.source).text(&query.target);
}

/// A meta-transport query: outcomes, treatments, target and every declared source in order.
fn encode_meta(writer: &mut CanonicalWriter, query: &MetaTransportQuery) {
    encode_ids(writer, &query.outcomes);
    encode_ids(writer, &query.treatments);
    writer.text(&query.target);
    encode_wire(writer, &query.sources);
}

/// A two-source query: outcomes, treatments, target, then each source in declared order: its
/// population, controllable coordinates, experiment assignment and selection targets (a set).
fn encode_two_source(writer: &mut CanonicalWriter, query: &TwoSourceZTransportQuery) {
    encode_ids(writer, &query.outcomes);
    encode_ids(writer, &query.treatments);
    writer.text(&query.target);
    for source in &query.sources {
        writer.text(&source.population);
        encode_ids(writer, &source.controllable);
        writer.word(source.experiment_assignment.len() as u64);
        for assignment in source.experiment_assignment.iter() {
            CanonicalEncode::encode(assignment, writer);
        }
        Canonical::encode(&SetOf(&source.selection_targets[..]), writer);
    }
}

fn sid_limits(writer: &mut CanonicalWriter, limits: SidLimits) {
    writer.word(limits.steps as u64).word(limits.depth as u64);
}

fn search_limits(writer: &mut CanonicalWriter, limits: SearchLimits) {
    writer.word(limits.operations as u64).word(limits.depth as u64);
}

/// The digest of the question: the request kind and every input the identifier reads.
#[allow(
    clippy::too_many_lines,
    reason = "one arm per identification request, each spelling its inputs"
)]
fn question(request: &TransportIdentificationRequest) -> [u8; 32] {
    use TransportIdentificationRequest as R;
    let mut w = CanonicalWriter::new("antecedent.transport_identification.question.v2");
    match request {
        R::Classical { diagram, query, limits } => {
            w.text("classical");
            encode_diagram(&mut w, diagram);
            encode_classical(&mut w, query);
            sid_limits(&mut w, *limits);
        }
        R::Catalog { diagram, query, catalog, limits } => {
            w.text("catalog");
            encode_diagram(&mut w, diagram);
            encode_classical(&mut w, query);
            encode_catalog(&mut w, catalog);
            sid_limits(&mut w, *limits);
        }
        R::CatalogSearch { derivation, diagram, catalog, limits } => {
            w.text("catalog_search");
            encode_wire(&mut w, &derivation.to_record());
            encode_diagram(&mut w, diagram);
            encode_catalog(&mut w, catalog);
            sid_limits(&mut w, *limits);
        }
        R::CatalogBinding { derivation, catalog, limits } => {
            w.text("catalog_binding");
            encode_wire(&mut w, &derivation.to_record());
            encode_catalog(&mut w, catalog);
            sid_limits(&mut w, *limits);
        }
        R::Meta { graph, query, limits } => {
            w.text("meta");
            encode_graph(&mut w, graph);
            encode_meta(&mut w, query);
            sid_limits(&mut w, *limits);
        }
        R::ZTransport { diagram, query, limits } => {
            w.text("z_transport");
            encode_diagram(&mut w, diagram);
            encode_wire(&mut w, &ZTransportQueryWire::from_query(query));
            sid_limits(&mut w, *limits);
        }
        R::ZTransportBinding { diagram, query, derivation, catalog } => {
            w.text("z_transport_binding");
            encode_diagram(&mut w, diagram);
            encode_wire(&mut w, &ZTransportQueryWire::from_query(query));
            encode_wire(&mut w, &derivation.to_record());
            encode_catalog(&mut w, catalog);
        }
        R::ZTransportInspecting { diagram, query, catalog, limits } => {
            w.text("z_transport_inspecting");
            encode_diagram(&mut w, diagram);
            encode_wire(&mut w, &ZTransportQueryWire::from_query(query));
            encode_catalog(&mut w, catalog);
            sid_limits(&mut w, *limits);
        }
        R::TwoSourceZTransport { graph, query, catalogs, limits } => {
            w.text("two_source_z_transport");
            encode_graph(&mut w, graph);
            encode_two_source(&mut w, query);
            encode_catalog(&mut w, &catalogs[0]);
            encode_catalog(&mut w, &catalogs[1]);
            sid_limits(&mut w, *limits);
        }
        R::MzTransport { graph, query, catalog, limits } => {
            w.text("mz_transport");
            encode_graph(&mut w, graph);
            encode_wire(&mut w, &MzTransportQueryWire::from_query(query));
            encode_catalog(&mut w, catalog);
            search_limits(&mut w, *limits);
        }
        R::MzTransportBinding { graph, derivation, catalog } => {
            w.text("mz_transport_binding");
            encode_graph(&mut w, graph);
            encode_wire(&mut w, &derivation.to_record());
            encode_catalog(&mut w, catalog);
        }
        R::MixedSource { graph, query, catalog, limits } => {
            w.text("mixed_source");
            encode_graph(&mut w, graph);
            encode_wire(&mut w, &MixedSourceQueryWire::from_query(query));
            encode_catalog(&mut w, catalog);
            search_limits(&mut w, *limits);
        }
        R::MixedSourceBinding { graph, derivation, catalog } => {
            w.text("mixed_source_binding");
            encode_graph(&mut w, graph);
            encode_wire(&mut w, &derivation.to_record());
            encode_catalog(&mut w, catalog);
        }
        R::AdmgConditional { diagram, query, catalog, limits } => {
            w.text("admg_conditional");
            encode_diagram(&mut w, diagram);
            encode_wire(&mut w, &AdmgConditionalQueryWire::from_query(query));
            encode_catalog(&mut w, catalog);
            search_limits(&mut w, *limits);
        }
        R::Counterfactual { problem, query, catalog, limits } => {
            w.text("counterfactual");
            encode_wire(&mut w, &CounterfactualTransportProblemWire::from_problem(problem));
            encode_wire(&mut w, &CounterfactualTransportQueryWire::from_query(query));
            encode_catalog(&mut w, catalog);
            search_limits(&mut w, *limits);
        }
        R::ObservationRecovery { graph, query, catalog, effect, limits } => {
            w.text("observation_recovery");
            encode_graph(&mut w, graph);
            encode_wire(&mut w, &RecoveryQueryWire::from_query(query));
            encode_catalog(&mut w, catalog);
            w.word(u64::from(effect.is_some()));
            if let Some(effect) = effect {
                encode_graph(&mut w, &effect.graph);
                encode_ids(&mut w, &effect.outcomes);
                encode_ids(&mut w, &effect.treatments);
            }
            search_limits(&mut w, limits.search);
            w.word(limits.memory_bytes);
        }
        R::ChainRecovery { graph, query } => {
            w.text("chain_recovery");
            encode_graph(&mut w, graph);
            encode_wire(&mut w, &ChainQueryWire::from_query(query));
        }
    }
    w.finish()
}

/// Plan id, graph class, functional kind and query variables of the request's family.
fn shape(
    request: &TransportIdentificationRequest,
) -> (&'static str, &'static str, FunctionalKind, Arc<[VariableId]>) {
    use TransportIdentificationRequest as R;
    let pair = |outcomes: &[VariableId], treatments: &[VariableId]| -> Arc<[VariableId]> {
        outcomes.iter().chain(treatments).copied().collect()
    };
    match request {
        R::Classical { query, .. } | R::Catalog { query, .. } => (
            "transport_identification.classical.v1",
            "selection_admg",
            FunctionalKind::InterventionalDistribution,
            pair(&query.outcomes, &query.treatments),
        ),
        R::CatalogSearch { derivation, .. } | R::CatalogBinding { derivation, .. } => (
            "transport_identification.classical.v1",
            "selection_admg",
            FunctionalKind::InterventionalDistribution,
            pair(&derivation.query().outcomes, &derivation.query().treatments),
        ),
        R::Meta { query, .. } => (
            "transport_identification.classical.v1",
            "admg_with_source_collection",
            FunctionalKind::InterventionalDistribution,
            pair(&query.outcomes, &query.treatments),
        ),
        R::ZTransport { query, .. }
        | R::ZTransportBinding { query, .. }
        | R::ZTransportInspecting { query, .. } => (
            "transport_identification.z.v1",
            "selection_admg",
            FunctionalKind::InterventionalDistribution,
            pair(&query.outcomes, &query.treatments),
        ),
        R::TwoSourceZTransport { query, .. } => (
            "transport_identification.z.v1",
            "admg_with_source_collection",
            FunctionalKind::InterventionalDistribution,
            pair(&query.outcomes, &query.treatments),
        ),
        R::MzTransport { .. } | R::MzTransportBinding { .. } => (
            "transport_identification.mz.v1",
            "admg_with_source_collection",
            FunctionalKind::InterventionalDistribution,
            Arc::from([]),
        ),
        R::MixedSource { .. } | R::MixedSourceBinding { .. } => (
            "transport_identification.mixed_source.v1",
            "admg",
            FunctionalKind::InterventionalDistribution,
            Arc::from([]),
        ),
        R::AdmgConditional { .. } => (
            "transport_identification.admg_conditional.v1",
            "selection_admg",
            FunctionalKind::InterventionalDistribution,
            Arc::from([]),
        ),
        R::Counterfactual { .. } => (
            "transport_identification.counterfactual.v1",
            "finite_selection_admg",
            FunctionalKind::Counterfactual,
            Arc::from([]),
        ),
        R::ObservationRecovery { .. } => (
            "observation_recovery_identification.v1",
            "m_graph",
            FunctionalKind::ConditionalObservation,
            Arc::from([]),
        ),
        R::ChainRecovery { .. } => (
            "chain_recovery_identification.v1",
            "m_graph",
            FunctionalKind::ConditionalObservation,
            Arc::from([]),
        ),
    }
}

/// The six layers of an identification program. The question digest is the problem's
/// identity; identification runs on each execution; nothing is estimated and no uncertainty
/// is claimed.
fn lower(
    request: &TransportIdentificationRequest,
) -> Result<LoweredProgram, TransportIdentificationError> {
    // A graph without a canonical identity refuses the compile rather than taking an identity
    // another request could share (`encoded`).
    let query = encoded("transport_identification", || question(request)).map_err(|_| {
        TransportIdentificationError::Identification(IdentificationError::UnsupportedQuery {
            message: "the declared graph has no canonical identity, so the compiled \
                      identification program cannot bind it",
        })
    })?;
    let (plan_id, graph_class, kind, query_variables) = shape(request);
    let cell = cell_of(request);
    let design = DesignShape::not_design_based("identification reads no rows");
    let scientific = ScientificProblem {
        runtime_query: None,
        identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional { kind, outcome: None, query },
        plan_id: Arc::from(plan_id),
        data: DataClassification::Tabular,
        graph_class,
        query_variables,
        source_digest: None,
        support_coordinate: None,
        supported: false,
        support_status: None,
    };
    let identification = IdentificationLayer::PerExecution {
        reason: "the theorem-scoped identifier decides the request each time the program runs",
    };
    let statistical = StatisticalProgram {
        identifier: None,
        estimator: None,
        realization: Realization::ExactPoint { evaluator: Arc::from(cell.as_str()) },
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
            digest: Some(query),
        }]),
        composition: Composition::Chained { stages: 1 },
        decision: None,
        planning: None,
        chain: None,
    };
    let effect = match cell {
        ProgramCell::ObservationRecoveryIdentification
        | ProgramCell::ChainRecoveryIdentification => PhysicalEffect::Condition,
        _ => PhysicalEffect::Transport,
    };
    let operators: Arc<[PhysicalOperator]> = PhysicalOperator::declare(effect)
        .build()
        .map_or_else(|_| Arc::from([]), |op| Arc::from([op]));
    let physical = PhysicalLayer { operators, design };
    Ok(StandaloneCompositionLoweredParts {
        cell,
        scientific,
        identification,
        statistical,
        evidence,
        physical,
        claim: ClaimDemand { uncertainty: UncertaintyReading::None, kind: None },
    }
    .finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_graph::DenseNodeId;

    fn v(raw: u32) -> VariableId {
        VariableId::from_raw(raw)
    }

    fn classical() -> TransportIdentificationRequest {
        let mut graph = Admg::with_variables(2);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        TransportIdentificationRequest::Classical {
            diagram: SelectionDiagram::try_new(graph, [v(1)]).unwrap(),
            query: ClassicalTransportQuery {
                outcomes: Arc::from([v(1)]),
                treatments: Arc::from([v(0)]),
                source: Arc::from("source"),
                target: Arc::from("target"),
            },
            limits: SidLimits::default(),
        }
    }

    #[test]
    fn the_program_identity_is_the_question_and_survives_recompilation() {
        let ctx = ExecutionContext::for_tests(0);
        let a = compile_transport_identification(classical(), &ctx).unwrap();
        let b = compile_transport_identification(classical(), &ctx).unwrap();
        assert_eq!(a.lowered().identity(), b.lowered().identity());
        assert_eq!(a.lowered().cell(), ProgramCell::ClassicalTransportIdentification);
    }

    fn classical_with(selections: &[u32], outcomes: &[u32]) -> TransportIdentificationRequest {
        let mut graph = Admg::with_variables(3);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
        let selections = selections.iter().copied().map(v).collect::<Vec<_>>();
        TransportIdentificationRequest::Classical {
            diagram: SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from(selections))
                .unwrap(),
            query: ClassicalTransportQuery {
                outcomes: outcomes.iter().copied().map(v).collect(),
                treatments: Arc::from([v(0)]),
                source: Arc::from("source"),
                target: Arc::from("target"),
            },
            limits: SidLimits::default(),
        }
    }

    fn two_source(level: f64) -> TransportIdentificationRequest {
        let mut graph = Admg::with_variables(3);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
        let source = |population: &str| antecedent_identify::ZTransportSourceSpec {
            population: Arc::from(population),
            controllable: Arc::from([v(1)]),
            experiment_assignment: Arc::from([antecedent_core::InterventionAssignment {
                variable: v(1),
                value: antecedent_core::Value::f64(level),
            }]),
            selection_targets: Arc::from([]),
        };
        TransportIdentificationRequest::TwoSourceZTransport {
            graph,
            query: TwoSourceZTransportQuery {
                outcomes: Arc::from([v(2)]),
                treatments: Arc::from([v(0)]),
                target: Arc::from("target"),
                sources: [source("a"), source("b")],
            },
            catalogs: [EvidenceCatalog::empty(), EvidenceCatalog::empty()],
            limits: SidLimits::default(),
        }
    }

    #[test]
    fn the_question_identity_is_canonical() {
        let base = question(&classical_with(&[1, 2], &[2, 1]));
        assert_eq!(base, question(&classical_with(&[1, 2], &[2, 1])), "repeatable");
        assert_eq!(base, question(&classical_with(&[2, 1], &[2, 1])), "selections are a set");
        assert_ne!(base, question(&classical_with(&[1, 2], &[1, 2])), "outcomes are ordered");
        assert_ne!(base, question(&classical_with(&[1], &[2, 1])), "a changed selection");
        assert_eq!(question(&two_source(1.0)), question(&two_source(1.0)));
        assert_ne!(question(&two_source(0.0)), question(&two_source(-0.0)), "reals by bits");
    }

    /// `question`: the selection targets are a set (identification reads a node set); outcomes
    /// and treatments are ordered (the identification record lists them in declared order); the
    /// population texts are framed; the request kind is tagged; the two sources of a two-source
    /// query are ordered (each binds its own catalog).
    #[test]
    fn site_transport_identification_question_selections_are_a_set() {
        use crate::identity_laws::{
            ADVERSARIAL_TEXT, all_distinct, injective, ordered_laws, set_laws,
        };

        let request = |selections: &[u32], outcomes: &[u32], source: &str, target: &str| {
            let mut graph = Admg::with_variables(4);
            for to in 1..4 {
                graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(to)).ok()?;
            }
            let selections = selections.iter().copied().map(v).collect::<Vec<_>>();
            let diagram =
                SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from(selections)).ok()?;
            Some(question(&TransportIdentificationRequest::Classical {
                diagram,
                query: ClassicalTransportQuery {
                    outcomes: outcomes.iter().copied().map(v).collect(),
                    treatments: Arc::from([v(0)]),
                    source: Arc::from(source),
                    target: Arc::from(target),
                },
                limits: SidLimits::default(),
            }))
        };
        set_laws(&[1, 2, 3], |selections| request(selections, &[2, 1], "s", "t"));
        ordered_laws(&[1, 2, 3], |outcomes| request(&[1], outcomes, "s", "t"));
        injective(
            &ADVERSARIAL_TEXT,
            |parts| *parts,
            |&(source, target)| request(&[1], &[2], source, target),
        );
        let mut swapped = two_source(1.0);
        if let TransportIdentificationRequest::TwoSourceZTransport { query, .. } = &mut swapped {
            query.sources.swap(0, 1);
        }
        let TransportIdentificationRequest::Classical { diagram, query, limits } = classical()
        else {
            unreachable!("classical() is a classical request")
        };
        let catalog = TransportIdentificationRequest::Catalog {
            diagram,
            query,
            catalog: EvidenceCatalog::empty(),
            limits,
        };
        all_distinct(&[
            question(&classical()),
            question(&catalog),
            question(&two_source(1.0)),
            question(&swapped),
        ]);
    }

    #[test]
    fn a_compiled_identification_refuses_reentry() {
        let ctx = ExecutionContext::for_tests(0);
        let compiled = compile_transport_identification(classical(), &ctx).unwrap();
        let inside = ctx.enter_lowered_dispatch(compiled.lowered().identity());
        assert!(matches!(compiled.identify(&inside), Err(TransportIdentificationError::Reentry)));
        assert!(matches!(
            compile_transport_identification(classical(), &inside),
            Err(TransportIdentificationError::Reentry)
        ));
    }
}
