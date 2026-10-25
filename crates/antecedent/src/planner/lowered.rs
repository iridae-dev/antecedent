//! The lowered program `Study::compile` attaches to every physical plan.
//!
//! One value per compiled route holds the six compiler layers: the scientific
//! problem, the identification product, the statistical program, the evidence and
//! decision program, the physical layer, and the claim the execution must publish.
//! Dispatch refuses a plan that carries none. The physical record itself stays on
//! [`crate::planner::PhysicalExecutionPlan`]; this value carries the operators and
//! design shape that record was planned from.
//!
//! The identity covers layers one to four and six. Layer five is execution
//! accounting: a data refresh may re-plan it without changing what was compiled.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod chain;
mod decision;
mod linear_gaussian;
mod planning;

use std::sync::Arc;

use antecedent_contract::{ClaimKind, DataClassification};
use antecedent_core::{
    CanonicalWriter, PhysicalOperator, ScientificTarget, SemanticDigest, VariableId,
};
use antecedent_expr::IdentifiedEstimand;
use antecedent_identify::{CompiledIdentificationProduct, IdentificationResult};

use crate::analysis::IdentificationCache;
use crate::analysis::lowering::RouteCell;
use crate::planner::statistical::StatisticalProgram;

pub use chain::ChainProgram;
pub use decision::{
    CarrierDemand, Composition, DecisionProgram, EvidenceDecisionProgram, EvidenceRole,
    EvidenceSourceRef, EvsiProgram,
};
pub(crate) use linear_gaussian::{
    linear_gaussian_physical_operators, linear_gaussian_program_operators,
    linear_gaussian_scientific_problem,
};
pub use planning::{PlanningProgram, SIMULATION_RECORD_BYTES, SimulationQuote};

const LOWERED_PROGRAM_TAG: &str = "antecedent.lowered_program";

/// How far a route's lowering has been carried. Every registered route cell lowers
/// each layer from its own scientific problem; there is no partial lowering.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum LoweringDepth {
    /// The route lowers each layer from its own scientific problem.
    Complete,
}

impl LoweringDepth {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Complete => "complete",
        }
    }
}

/// Non-world-indexed functional a scientific problem asks for.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum FunctionalKind {
    /// Law of an outcome under an intervention.
    InterventionalDistribution,
    /// Effect transmitted along a declared set of paths.
    PathSpecific,
    /// Unit-level counterfactual.
    Counterfactual,
    /// Attribution of an anomalous value to upstream mechanisms.
    AnomalyAttribution,
    /// Attribution of a distribution change to mechanism changes.
    ChangeAttribution,
    /// Detection of changed mechanisms.
    MechanismChange,
    /// Attribution of one unit's change.
    UnitChange,
    /// Matching on histories.
    HistoryMatch,
    /// Identified set rather than a point functional.
    IdentifiedSet,
    /// Contrast of randomized assignment arms (ITT, complier or treated-recipient effect).
    AssignmentContrast,
    /// Contrast of mean outcomes between two network exposure levels.
    ExposureContrast,
    /// Value of a fixed treatment policy against its reference policy.
    PolicyValue,
    /// Effect on the treated units or unit of a panel design.
    EffectOnTreated,
    /// Contrast of arm event-time laws: survival, restricted mean or cumulative incidence.
    SurvivalContrast,
    /// Value of a prescribed sequential treatment regime.
    RegimeValue,
    /// Derivative, average derivative, directional derivative or Jacobian of a response curve.
    ResponseDerivative,
    /// Contrast of an outcome functional between intervention worlds the world algebra
    /// cannot state (a callback action, or a schedule outside the horizon).
    InterventionContrast,
    /// A causal structure, structure class or structure posterior searched from the
    /// observational law.
    StructureSearch,
    /// An attribution computed under fitted mechanisms (intrinsic influence, root
    /// causes, distribution change, paths, feature relevance).
    Attribution,
    /// A simulation of fitted mechanisms (interventional draws, policy value, unit effects).
    Simulation,
    /// A quantity propagated through a chain of transformations: a measurement
    /// mapping, a transport, a pool, a derived quantity, a decision utility and a
    /// preposterior update.
    PropagatedChain,
    /// A noncausal association of an outcome with predictors in the observed law: a
    /// conditional law, regression coefficients, an association measure or a scored
    /// predictive model. It states no intervention.
    Association,
}

impl FunctionalKind {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InterventionalDistribution => "interventional_distribution",
            Self::PathSpecific => "path_specific",
            Self::Counterfactual => "counterfactual",
            Self::AnomalyAttribution => "anomaly_attribution",
            Self::ChangeAttribution => "change_attribution",
            Self::MechanismChange => "mechanism_change",
            Self::UnitChange => "unit_change",
            Self::HistoryMatch => "history_match",
            Self::IdentifiedSet => "identified_set",
            Self::AssignmentContrast => "assignment_contrast",
            Self::ExposureContrast => "exposure_contrast",
            Self::PolicyValue => "policy_value",
            Self::EffectOnTreated => "effect_on_treated",
            Self::SurvivalContrast => "survival_contrast",
            Self::RegimeValue => "regime_value",
            Self::ResponseDerivative => "response_derivative",
            Self::InterventionContrast => "intervention_contrast",
            Self::StructureSearch => "structure_search",
            Self::Attribution => "attribution",
            Self::Simulation => "simulation",
            Self::PropagatedChain => "propagated_chain",
            Self::Association => "association",
        }
    }
}

/// What the scientific problem asks for.
#[derive(Clone, Debug)]
pub enum ProblemTarget {
    /// World-indexed target: worlds, a contrast, a dose-time surface, or mediation.
    Target(ScientificTarget),
    /// One world-indexed target per requested horizon, in request order: a
    /// multi-horizon mediation or a per-horizon family the query answers jointly.
    Targets(Arc<[ScientificTarget]>),
    /// A functional that is not indexed by intervention worlds, with the digest of the query
    /// (interventions, levels, conditioning, evaluation points) that defines it, so two queries of
    /// one kind have different identities.
    QueryFunctional {
        /// Kind of functional.
        kind: FunctionalKind,
        /// Outcome the functional reads, when it has one.
        outcome: Option<VariableId>,
        /// Canonical digest of the query the functional answers.
        query: [u8; 32],
    },
}

/// Layer one: the scientific problem.
#[derive(Clone, Debug)]
pub struct ScientificProblem {
    pub(crate) target: ProblemTarget,
    pub(crate) plan_id: Arc<str>,
    pub(crate) data: DataClassification,
    pub(crate) graph_class: &'static str,
    pub(crate) query_variables: Arc<[VariableId]>,
    pub(crate) source_digest: Option<[u8; 32]>,
    pub(crate) support_coordinate: Option<Arc<str>>,
    pub(crate) licensed: bool,
    /// Evidence status of the coordinate, read from the study at lowering. Not part of the
    /// program identity: the identity binds the coordinate and whether it is licensed.
    pub(crate) support_status: Option<crate::support::CellStatus>,
}

impl ScientificProblem {
    /// What the problem asks for.
    #[must_use]
    pub const fn target(&self) -> &ProblemTarget {
        &self.target
    }

    /// Route record plan id.
    #[must_use]
    pub fn plan_id(&self) -> &str {
        &self.plan_id
    }

    /// Data modality the problem reads.
    #[must_use]
    pub const fn data(&self) -> DataClassification {
        self.data
    }

    /// Graph class of the supplied structure.
    #[must_use]
    pub const fn graph_class(&self) -> &'static str {
        self.graph_class
    }

    /// Variables the query names.
    #[must_use]
    pub fn query_variables(&self) -> &[VariableId] {
        &self.query_variables
    }

    /// Content digest of the source rows, when the lowering bound one.
    #[must_use]
    pub const fn source_digest(&self) -> Option<[u8; 32]> {
        self.source_digest
    }

    /// Support-matrix coordinate of the study, when the query is on the matrix axis.
    #[must_use]
    pub fn support_coordinate(&self) -> Option<&str> {
        self.support_coordinate.as_deref()
    }

    /// Whether the study's coordinate is a licensed cell.
    #[must_use]
    pub const fn licensed(&self) -> bool {
        self.licensed
    }

    /// Evidence status of the study's support-matrix cell.
    #[must_use]
    pub const fn support_status(&self) -> Option<crate::support::CellStatus> {
        self.support_status
    }

    fn write(&self, writer: &mut CanonicalWriter) {
        match &self.target {
            ProblemTarget::Target(target) => {
                writer.text("target").bytes(target.identity().as_bytes());
            }
            ProblemTarget::Targets(targets) => {
                writer.text("targets").word(targets.len() as u64);
                for target in targets.iter() {
                    writer.bytes(target.identity().as_bytes());
                }
            }
            ProblemTarget::QueryFunctional { kind, outcome, query } => {
                writer
                    .text("query_functional")
                    .text(kind.as_str())
                    .opt_word(outcome.map(|id| u64::from(id.raw())))
                    .bytes(query);
            }
        }
        writer
            .text(&self.plan_id)
            .text(data_classification_name(self.data))
            .text(self.graph_class)
            .word(self.query_variables.len() as u64);
        for variable in &*self.query_variables {
            writer.word(u64::from(variable.raw()));
        }
        writer
            .opt_text(self.source_digest.as_ref().map(hex).as_deref())
            .opt_text(self.support_coordinate.as_deref())
            .word(u64::from(self.licensed));
    }
}

/// Layer two: what identification the program carries.
#[derive(Clone, Debug)]
pub(crate) enum IdentificationLayer {
    /// Identification is left to preparation, which identifies and reports what it
    /// finds; the reason says why compile retained nothing.
    Deferred {
        /// Why compile retained no identification.
        reason: &'static str,
        /// The not-identified product the refusal came from, when the route reads one: its
        /// status, derivation and query. It is the record the evidence needs derive from;
        /// it is not an identification claim and preparation still raises the refusal.
        /// Not part of the program's identity: the declared graph and query already are.
        failure: Option<Arc<IdentificationResult>>,
    },
    /// Identified at compile and retained with the program; preparation installs
    /// these products and does not identify again.
    Retained(Arc<RetainedIdentification>),
    /// The route identifies on each execution.
    PerExecution {
        /// Why the route cannot retain one identification.
        reason: &'static str,
    },
    /// Identification of a feedback model that is not a graphical identification
    /// product of an acyclic model.
    NonGraphical(Arc<NonGraphicalIdentification>),
    /// The program claims no identification: it describes the observed law and states no
    /// causal contrast. When it follows a causal query that was refused, the refusal's
    /// reason code is retained.
    NotClaimed {
        /// Reason code of the refused causal query this program follows.
        refusal: Option<Arc<str>>,
    },
}

/// Identification a feedback program carries in place of a graphical product.
///
/// None of these is upgraded to graphical identification: a specified or fitted
/// model stays conditional on that model, a clamp-experiment recovery keeps the
/// directions its design leaves unresolved, and a cyclic input/output SCM
/// identification stays scoped to its theorem and premises.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum NonGraphicalIdentification {
    /// Conditional on the specified or fitted model whose digest is `model`.
    SpecifiedModelConditional {
        /// Digest of the model the answer is conditional on.
        model: [u8; 32],
    },
    /// Linear law recovered from declared clamp experiments.
    ExperimentalRecovery {
        /// Digest of the declared clamp design.
        design: [u8; 32],
        /// Directions the design leaves unresolved, as `(target, missing clamps)`.
        unresolved: Arc<[(VariableId, Arc<[VariableId]>)]>,
    },
    /// Theorem-scoped identification of an input/output SCM on a cyclic graph.
    CyclicIoScm {
        /// Digest of the graphical source and its premises.
        source: [u8; 32],
    },
    /// The overlap of the evidence a chain combines, from its dependence
    /// contract: which stages share rows, clusters, models, priors,
    /// disturbances or periods. Nothing is identified from observations.
    SharedEvidenceContract {
        /// Digest of the declared chain, which carries the declared factors.
        contract: [u8; 32],
    },
}

impl NonGraphicalIdentification {
    /// Stable `snake_case` name.
    pub(crate) const fn as_str(&self) -> &'static str {
        match self {
            Self::SpecifiedModelConditional { .. } => "specified_model_conditional",
            Self::ExperimentalRecovery { .. } => "experimental_recovery",
            Self::CyclicIoScm { .. } => "cyclic_ioscm",
            Self::SharedEvidenceContract { .. } => "shared_evidence_contract",
        }
    }

    fn write(&self, writer: &mut CanonicalWriter) {
        writer.text(self.as_str());
        match self {
            Self::SpecifiedModelConditional { model } => {
                writer.bytes(model);
            }
            Self::ExperimentalRecovery { design, unresolved } => {
                writer.bytes(design).word(unresolved.len() as u64);
                for (target, missing) in unresolved.iter() {
                    writer.word(u64::from(target.raw())).word(missing.len() as u64);
                    for source in missing.iter() {
                        writer.word(u64::from(source.raw()));
                    }
                }
            }
            Self::CyclicIoScm { source } => {
                writer.bytes(source);
            }
            Self::SharedEvidenceContract { contract } => {
                writer.bytes(contract);
            }
        }
    }
}

/// Identification products compiled with a lowered program.
#[derive(Clone, Debug)]
pub(crate) struct RetainedIdentification {
    /// Checked product bound to the scientific target, when the route builds one.
    pub(crate) product: Option<CompiledIdentificationProduct>,
    /// Products preparation installs, in install order.
    pub(crate) caches: Arc<[IdentificationCache]>,
    /// Digest of the study inputs these products were identified from (query,
    /// structure and the options identification reads). Dispatch installs a retained
    /// product only into a study with the same key; `None` (an input with no
    /// canonical encoding) matches no study.
    pub(crate) key: Option<[u8; 32]>,
    /// Seal of each product in [`Self::caches`], index-aligned, computed when compile
    /// retained it: a digest over the full identification result. A checked operation
    /// verifies the product it holds against this instead of identifying again.
    seals: Arc<[Option<[u8; 32]>]>,
    /// An identification no study slot holds, which the execution reads from the program: a
    /// mediation on a temporal window, or the probe of a temporal cross-world query.
    execution: Option<RetainedExecutionProof>,
}

/// Identification retained with the program for an execution that identifies on a window or
/// probe derived from its query and structure, so that it is identified once.
#[derive(Clone, Debug)]
pub(crate) enum RetainedExecutionProof {
    /// One proof per component (total, direct, indirect) on the unrolled ADMG window;
    /// `None` where a component the query did not request is not identified.
    Admg(Vec<Option<(IdentificationResult, IdentifiedEstimand)>>),
    /// One class certificate per component on the unfolded PAG window.
    Pag {
        /// Certificates in component order; `None` where not fully identified.
        envelopes: Vec<Option<antecedent_identify::TemporalClassEnvelope>>,
    },
    /// The one-step pulse probe of a temporal cross-world query on a temporal DAG.
    CrossWorld {
        /// The probe's identification.
        identification: IdentificationResult,
        /// The lag indexer of its unfolding.
        indexer: antecedent_core::TemporalIndexer,
    },
    /// The class certificate of the cross-world probe on a temporal CPDAG or PAG.
    CrossWorldClass {
        /// Completions of the class and their identification.
        envelope: Arc<antecedent_identify::TemporalClassEnvelope>,
    },
}

impl RetainedIdentification {
    /// Retain `caches`, sealing each product a checked operation proves from.
    ///
    /// # Errors
    ///
    /// A product cannot be encoded for its seal.
    pub(crate) fn new(
        product: Option<CompiledIdentificationProduct>,
        caches: Vec<IdentificationCache>,
        key: Option<[u8; 32]>,
    ) -> Result<Self, crate::error::CausalError> {
        let seals = caches
            .iter()
            .map(crate::analysis::identification_seal::seal_cache)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            product,
            caches: Arc::from(caches),
            key,
            seals: Arc::from(seals),
            execution: None,
        })
    }

    /// The same retained products with an execution proof.
    pub(crate) fn with_execution_proof(mut self, proof: RetainedExecutionProof) -> Self {
        self.execution = Some(proof);
        self
    }

    /// The execution proof, when the program retained one.
    pub(crate) const fn execution_proof(&self) -> Option<&RetainedExecutionProof> {
        self.execution.as_ref()
    }

    /// Products preparation installs.
    pub(crate) fn caches(&self) -> &[IdentificationCache] {
        &self.caches
    }

    /// The compile-time seal of the static identification, when one was retained.
    pub(crate) fn static_seal(&self) -> Option<[u8; 32]> {
        self.sealed_where(|cache| matches!(cache, IdentificationCache::Static(_)))
    }

    /// The compile-time seal of the temporal identification, when one was retained.
    pub(crate) fn temporal_seal(&self) -> Option<[u8; 32]> {
        self.sealed_where(|cache| matches!(cache, IdentificationCache::Temporal(_)))
    }

    fn sealed_where(&self, pick: impl Fn(&IdentificationCache) -> bool) -> Option<[u8; 32]> {
        self.caches
            .iter()
            .zip(self.seals.iter())
            .find(|(cache, _)| pick(cache))
            .and_then(|(_, seal)| *seal)
    }
}

impl IdentificationLayer {
    fn write(&self, writer: &mut CanonicalWriter) {
        match self {
            Self::Deferred { reason, .. } => {
                writer.text("deferred").text(reason);
            }
            Self::Retained(retained) => {
                writer.text("retained");
                match &retained.product {
                    Some(product) => writer.word(1).bytes(product.identity().as_bytes()),
                    None => writer.word(0),
                };
                writer.word(retained.caches.len() as u64);
                for cache in &*retained.caches {
                    writer.text(cache.kind());
                }
            }
            Self::PerExecution { reason } => {
                writer.text("per_execution").text(reason);
            }
            Self::NonGraphical(identification) => {
                writer.text("non_graphical");
                identification.write(writer);
            }
            Self::NotClaimed { refusal } => {
                writer.text("not_claimed").opt_text(refusal.as_deref());
            }
        }
    }
}

/// Where a physical design width came from.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum WidthSource {
    /// Encoded from the selected roles and their contrast codings.
    Quoted,
    /// The route fits no single quoted adjustment design; the reason says what it
    /// fits instead and when that is sized.
    NotDesignBased(&'static str),
}

impl WidthSource {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Quoted => "quoted",
            Self::NotDesignBased(_) => "not_design_based",
        }
    }
}

/// The design a program fits: the rows and encoded columns of a quoted design, or
/// the reason the route fits no single design. A route that is not design based
/// carries no rows or columns at all: the planner's sizing of the physical record
/// stays on that record and is not restated here as a design.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct DesignShape(DesignExtent);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
enum DesignExtent {
    Quoted { rows: u64, columns: u64, direct: bool },
    NotDesignBased(&'static str),
}

impl DesignShape {
    /// A quoted design of `rows` by `columns` encoded columns, including the intercept.
    pub(crate) const fn quoted(rows: u64, columns: u64, direct: bool) -> Self {
        Self(DesignExtent::Quoted { rows, columns, direct })
    }

    /// A route that fits no single quoted design; `reason` says what it fits instead.
    pub(crate) const fn not_design_based(reason: &'static str) -> Self {
        Self(DesignExtent::NotDesignBased(reason))
    }

    /// The same design gathered through a materialized contrast expansion.
    pub(crate) const fn materialized(self) -> Self {
        match self.0 {
            DesignExtent::Quoted { rows, columns, .. } => Self::quoted(rows, columns, false),
            DesignExtent::NotDesignBased(_) => self,
        }
    }

    /// Rows a quoted design was sized for. `None` when the route is not design based.
    #[must_use]
    pub const fn rows(&self) -> Option<u64> {
        match self.0 {
            DesignExtent::Quoted { rows, .. } => Some(rows),
            DesignExtent::NotDesignBased(_) => None,
        }
    }

    /// Encoded columns of a quoted design, including the intercept. `None` when the
    /// route is not design based.
    #[must_use]
    pub const fn columns(&self) -> Option<u64> {
        match self.0 {
            DesignExtent::Quoted { columns, .. } => Some(columns),
            DesignExtent::NotDesignBased(_) => None,
        }
    }

    /// Where the column count came from.
    #[must_use]
    pub const fn source(&self) -> WidthSource {
        match self.0 {
            DesignExtent::Quoted { .. } => WidthSource::Quoted,
            DesignExtent::NotDesignBased(reason) => WidthSource::NotDesignBased(reason),
        }
    }

    /// Whether every design column is numeric, so the design is gathered directly
    /// from borrowed source columns instead of a materialized contrast expansion.
    /// A route that is not design based gathers no design.
    #[must_use]
    pub const fn direct(&self) -> bool {
        match self.0 {
            DesignExtent::Quoted { direct, .. } => direct,
            DesignExtent::NotDesignBased(_) => false,
        }
    }
}

/// Layer five: the physical operators and design shape the plan record was sized from.
#[derive(Clone, Debug)]
pub struct PhysicalLayer {
    pub(crate) operators: Arc<[PhysicalOperator]>,
    pub(crate) design: DesignShape,
}

impl PhysicalLayer {
    /// Physical operator contracts.
    #[must_use]
    pub fn operators(&self) -> &[PhysicalOperator] {
        &self.operators
    }

    /// Design shape.
    #[must_use]
    pub const fn design(&self) -> DesignShape {
        self.design
    }
}

/// How the published uncertainty must be read.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum UncertaintyReading {
    /// Sampling law of a frequentist estimate.
    Sampling,
    /// Posterior law under a declared likelihood and prior.
    Posterior,
    /// Identified set or envelope.
    Envelope,
    /// Point claim with no uncertainty.
    None,
}

impl UncertaintyReading {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Sampling => "sampling",
            Self::Posterior => "posterior",
            Self::Envelope => "envelope",
            Self::None => "none",
        }
    }
}

/// Layer six: the claim an execution must publish. The measured receipt stays on the result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ClaimDemand {
    pub(crate) uncertainty: UncertaintyReading,
    pub(crate) kind: Option<ClaimKind>,
}

impl ClaimDemand {
    /// How the published uncertainty must be read.
    #[must_use]
    pub const fn uncertainty(&self) -> UncertaintyReading {
        self.uncertainty
    }

    /// Claim kind the route publishes, when its lowering states one.
    #[must_use]
    pub const fn kind(&self) -> Option<ClaimKind> {
        self.kind
    }
}

/// The layers a route lowering assembles. [`Self::finish`] computes the identity.
pub(crate) struct LoweredParts {
    pub(crate) cell: RouteCell,
    pub(crate) depth: LoweringDepth,
    pub(crate) scientific: ScientificProblem,
    pub(crate) identification: IdentificationLayer,
    pub(crate) statistical: StatisticalProgram,
    pub(crate) evidence: EvidenceDecisionProgram,
    pub(crate) physical: PhysicalLayer,
    pub(crate) claim: ClaimDemand,
}

impl LoweredParts {
    /// Seal the parts into a lowered program.
    pub(crate) fn finish(self) -> LoweredProgram {
        let mut writer = CanonicalWriter::new(LOWERED_PROGRAM_TAG);
        writer.text(self.cell.as_str()).text(self.depth.as_str());
        self.scientific.write(&mut writer);
        self.identification.write(&mut writer);
        writer.bytes(self.statistical.identity().as_bytes());
        writer.bytes(self.evidence.identity().as_bytes());
        writer.text(self.claim.uncertainty.as_str());
        let identity = SemanticDigest::from_bytes(writer.finish());
        LoweredProgram {
            cell: ProgramCell::Route(self.cell),
            depth: self.depth,
            scientific: self.scientific,
            identification: self.identification,
            statistical: Arc::new(self.statistical),
            evidence: self.evidence,
            physical: self.physical,
            claim: self.claim,
            identity,
        }
    }
}

/// The parts a cyclic family lowering assembles. The program is always
/// [`LoweringDepth::Complete`]; [`Self::finish`] computes the identity.
pub(crate) struct CyclicLoweredParts {
    /// Stable family name the program was lowered for.
    pub(crate) family: &'static str,
    pub(crate) scientific: ScientificProblem,
    pub(crate) identification: IdentificationLayer,
    pub(crate) statistical: StatisticalProgram,
    pub(crate) evidence: EvidenceDecisionProgram,
    pub(crate) physical: PhysicalLayer,
    pub(crate) claim: ClaimDemand,
}

impl CyclicLoweredParts {
    /// Seal the parts into a lowered program for the cyclic family.
    pub(crate) fn finish(self) -> LoweredProgram {
        let cell = ProgramCell::CyclicFamily(self.family);
        let depth = LoweringDepth::Complete;
        let mut writer = CanonicalWriter::new(LOWERED_PROGRAM_TAG);
        writer.text(cell.as_str()).text(self.family).text(depth.as_str());
        self.scientific.write(&mut writer);
        self.identification.write(&mut writer);
        writer.bytes(self.statistical.identity().as_bytes());
        writer.bytes(self.evidence.identity().as_bytes());
        writer.text(self.claim.uncertainty.as_str());
        let identity = SemanticDigest::from_bytes(writer.finish());
        LoweredProgram {
            cell,
            depth,
            scientific: self.scientific,
            identification: self.identification,
            statistical: Arc::new(self.statistical),
            evidence: self.evidence,
            physical: self.physical,
            claim: self.claim,
            identity,
        }
    }
}

/// The parts a standalone planning lowering assembles: a future trial of a
/// declared design, with no data bound. [`Self::finish`] computes the identity.
pub(crate) struct PlanningLoweredParts {
    /// Stable name of the planned design.
    pub(crate) design: &'static str,
    pub(crate) scientific: ScientificProblem,
    pub(crate) identification: IdentificationLayer,
    pub(crate) statistical: StatisticalProgram,
    pub(crate) evidence: EvidenceDecisionProgram,
    pub(crate) physical: PhysicalLayer,
    pub(crate) claim: ClaimDemand,
}

impl PlanningLoweredParts {
    /// Seal the parts into a lowered program for the planned design.
    pub(crate) fn finish(self) -> LoweredProgram {
        let cell = ProgramCell::Planning(self.design);
        let depth = LoweringDepth::Complete;
        let mut writer = CanonicalWriter::new(LOWERED_PROGRAM_TAG);
        writer.text(cell.as_str()).text(self.design).text(depth.as_str());
        self.scientific.write(&mut writer);
        self.identification.write(&mut writer);
        writer.bytes(self.statistical.identity().as_bytes());
        writer.bytes(self.evidence.identity().as_bytes());
        writer.text(self.claim.uncertainty.as_str());
        let identity = SemanticDigest::from_bytes(writer.finish());
        LoweredProgram {
            cell,
            depth,
            scientific: self.scientific,
            identification: self.identification,
            statistical: Arc::new(self.statistical),
            evidence: self.evidence,
            physical: self.physical,
            claim: self.claim,
            identity,
        }
    }
}

/// The parts a propagation-chain lowering assembles: a chain of transformations
/// over declared sources, with no study bound. [`Self::finish`] computes the
/// identity.
pub(crate) struct ChainLoweredParts {
    /// Stable name of the reading the chain answers (`spread`, `enclosure`, ...).
    pub(crate) reading: &'static str,
    pub(crate) scientific: ScientificProblem,
    pub(crate) identification: IdentificationLayer,
    pub(crate) statistical: StatisticalProgram,
    pub(crate) evidence: EvidenceDecisionProgram,
    pub(crate) physical: PhysicalLayer,
    pub(crate) claim: ClaimDemand,
}

impl ChainLoweredParts {
    /// Seal the parts into a lowered program for the chain.
    pub(crate) fn finish(self) -> LoweredProgram {
        let cell = ProgramCell::PropagationChain(self.reading);
        let depth = LoweringDepth::Complete;
        let mut writer = CanonicalWriter::new(LOWERED_PROGRAM_TAG);
        writer.text(cell.as_str()).text(self.reading).text(depth.as_str());
        self.scientific.write(&mut writer);
        self.identification.write(&mut writer);
        writer.bytes(self.statistical.identity().as_bytes());
        writer.bytes(self.evidence.identity().as_bytes());
        writer.text(self.claim.uncertainty.as_str());
        let identity = SemanticDigest::from_bytes(writer.finish());
        LoweredProgram {
            cell,
            depth,
            scientific: self.scientific,
            identification: self.identification,
            statistical: Arc::new(self.statistical),
            evidence: self.evidence,
            physical: self.physical,
            claim: self.claim,
            identity,
        }
    }
}

/// The compiled cell a lowered program was lowered for: a route cell of
/// `Study::compile`, a cyclic family that prepares through its own entry, a
/// standalone experiment plan, or a propagation chain.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) enum ProgramCell {
    /// A registered route cell.
    Route(RouteCell),
    /// A cyclic family, by its stable family name.
    CyclicFamily(&'static str),
    /// A standalone experiment plan, by its design name.
    Planning(&'static str),
    /// A chain of transformations, by the reading it answers.
    PropagationChain(&'static str),
}

impl ProgramCell {
    /// Stable name, written to the licensed-route record.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Route(cell) => cell.as_str(),
            Self::CyclicFamily(_) => "CyclicFamily",
            Self::Planning(_) => "ExperimentPlanning",
            Self::PropagationChain(_) => "PropagationChain",
        }
    }
}

impl PartialEq<ProgramCell> for RouteCell {
    fn eq(&self, other: &ProgramCell) -> bool {
        matches!(other, ProgramCell::Route(cell) if cell == self)
    }
}

impl PartialEq<RouteCell> for ProgramCell {
    fn eq(&self, other: &RouteCell) -> bool {
        matches!(self, ProgramCell::Route(cell) if cell == other)
    }
}

/// The program one route lowers to. Every plan `Study::compile` returns carries one.
#[derive(Clone, Debug)]
pub struct LoweredProgram {
    cell: ProgramCell,
    depth: LoweringDepth,
    scientific: ScientificProblem,
    identification: IdentificationLayer,
    statistical: Arc<StatisticalProgram>,
    evidence: EvidenceDecisionProgram,
    physical: PhysicalLayer,
    claim: ClaimDemand,
    identity: SemanticDigest,
}

const _: () = {
    const fn shared_across_workers<T: Send + Sync>() {}
    shared_across_workers::<LoweredProgram>();
};

impl LoweredProgram {
    /// Name of the compiled route cell this program was lowered for.
    #[must_use]
    pub const fn route(&self) -> &'static str {
        self.cell.as_str()
    }

    /// Stable family name when this program was lowered for a cyclic family.
    #[must_use]
    pub const fn cyclic_family(&self) -> Option<&'static str> {
        match self.cell {
            ProgramCell::CyclicFamily(family) => Some(family),
            ProgramCell::Route(_) | ProgramCell::Planning(_) | ProgramCell::PropagationChain(_) => {
                None
            }
        }
    }

    /// Stable design name when this program was lowered for a standalone experiment plan.
    #[must_use]
    pub const fn planned_design(&self) -> Option<&'static str> {
        match self.cell {
            ProgramCell::Planning(design) => Some(design),
            ProgramCell::Route(_)
            | ProgramCell::CyclicFamily(_)
            | ProgramCell::PropagationChain(_) => None,
        }
    }

    /// The reading a propagation chain answers when this program was lowered for one.
    #[must_use]
    pub const fn chain_reading(&self) -> Option<&'static str> {
        match self.cell {
            ProgramCell::PropagationChain(reading) => Some(reading),
            ProgramCell::Route(_) | ProgramCell::CyclicFamily(_) | ProgramCell::Planning(_) => None,
        }
    }

    /// The propagation chain this program was lowered from, when it was lowered
    /// for one.
    #[must_use]
    pub fn propagation_chain(&self) -> Option<&ChainProgram> {
        self.evidence.chain()
    }

    /// How far this route's lowering has been carried.
    #[must_use]
    pub const fn depth(&self) -> LoweringDepth {
        self.depth
    }

    /// Layer one.
    #[must_use]
    pub const fn scientific(&self) -> &ScientificProblem {
        &self.scientific
    }

    /// Whether layer two retains an identification product compiled with the program.
    #[must_use]
    pub const fn identification_retained(&self) -> bool {
        matches!(self.identification, IdentificationLayer::Retained(_))
    }

    /// Identification retained at compile, for preparation to install.
    pub(crate) fn retained_identification(&self) -> Option<&RetainedIdentification> {
        match &self.identification {
            IdentificationLayer::Retained(retained) => Some(retained),
            IdentificationLayer::Deferred { .. }
            | IdentificationLayer::PerExecution { .. }
            | IdentificationLayer::NotClaimed { .. }
            | IdentificationLayer::NonGraphical(_) => None,
        }
    }

    /// The not-identified product a deferred layer two retained, when the refusal came from
    /// a route that reads one.
    pub(crate) fn refused_identification(&self) -> Option<&IdentificationResult> {
        match &self.identification {
            IdentificationLayer::Deferred { failure, .. } => failure.as_deref(),
            IdentificationLayer::Retained(_)
            | IdentificationLayer::PerExecution { .. }
            | IdentificationLayer::NotClaimed { .. }
            | IdentificationLayer::NonGraphical(_) => None,
        }
    }

    /// Layer two as compile lowered it.
    pub(crate) const fn identification_layer(&self) -> &IdentificationLayer {
        &self.identification
    }

    /// Identification of a feedback program, when layer two carries one.
    pub(crate) fn non_graphical_identification(&self) -> Option<&NonGraphicalIdentification> {
        match &self.identification {
            IdentificationLayer::NonGraphical(identification) => Some(identification),
            IdentificationLayer::Deferred { .. }
            | IdentificationLayer::PerExecution { .. }
            | IdentificationLayer::NotClaimed { .. }
            | IdentificationLayer::Retained(_) => None,
        }
    }

    /// Layer three.
    #[must_use]
    pub fn statistical(&self) -> &StatisticalProgram {
        &self.statistical
    }

    /// Layer four.
    #[must_use]
    pub const fn evidence(&self) -> &EvidenceDecisionProgram {
        &self.evidence
    }

    /// Layer five.
    #[must_use]
    pub const fn physical(&self) -> &PhysicalLayer {
        &self.physical
    }

    /// The program with `operators` appended to its physical layer. The physical layer
    /// stays out of the identity, so the program's identity does not move.
    pub(crate) fn with_physical_operators(mut self, operators: Vec<PhysicalOperator>) -> Self {
        let mut all = self.physical.operators.to_vec();
        all.extend(operators);
        self.physical.operators = Arc::from(all);
        self
    }

    /// Layer six.
    #[must_use]
    pub const fn claim(&self) -> &ClaimDemand {
        &self.claim
    }

    /// Identity over layers one to four and six.
    #[must_use]
    pub const fn identity(&self) -> SemanticDigest {
        self.identity
    }

    /// Program cell, for dispatch verification. A cyclic family's program never
    /// equals a route cell, so route dispatch refuses it.
    pub(crate) const fn cell(&self) -> ProgramCell {
        self.cell
    }
}

/// Stable `snake_case` name of a data classification.
pub(crate) const fn data_classification_name(data: DataClassification) -> &'static str {
    match data {
        DataClassification::Tabular => "tabular",
        DataClassification::Temporal => "temporal",
        DataClassification::Panel => "panel",
        DataClassification::MultiEnvironment => "multi_environment",
        DataClassification::Event => "event",
    }
}

fn hex(digest: &[u8; 32]) -> String {
    SemanticDigest::from_bytes(*digest).to_hex()
}
