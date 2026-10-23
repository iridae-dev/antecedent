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

mod decision;
mod linear_gaussian;

use std::sync::Arc;

use antecedent_contract::{ClaimKind, DataClassification};
use antecedent_core::{
    CanonicalWriter, PhysicalOperator, ScientificTarget, SemanticDigest, VariableId,
};
use antecedent_identify::CompiledIdentificationProduct;

use crate::analysis::IdentificationCache;
use crate::analysis::lowering::RouteCell;
use crate::planner::statistical::StatisticalProgram;

pub use decision::{
    CarrierDemand, Composition, DecisionProgram, EvidenceDecisionProgram, EvidenceRole,
    EvidenceSourceRef, EvsiProgram,
};
pub(crate) use linear_gaussian::{
    linear_gaussian_physical_operators, linear_gaussian_program_operators,
    linear_gaussian_scientific_problem,
};

const LOWERED_PROGRAM_TAG: &str = "antecedent.lowered_program";

/// How far a route's lowering has been carried.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum LoweringDepth {
    /// Placeholder lowering shared by every route: the route record, the declared
    /// inference, and the planner's schema-bound physical estimate. Preparation
    /// still identifies on its own.
    Generic,
    /// The route lowers each layer from its own scientific problem.
    Complete,
}

impl LoweringDepth {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Generic => "generic",
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
        }
    }
}

/// What the scientific problem asks for.
#[derive(Clone, Debug)]
pub enum ProblemTarget {
    /// World-indexed target: worlds, a contrast, a dose-time surface, or mediation.
    Target(ScientificTarget),
    /// A functional that is not indexed by intervention worlds.
    Functional {
        /// Kind of functional.
        kind: FunctionalKind,
        /// Outcome the functional reads, when it has one.
        outcome: Option<VariableId>,
    },
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
    /// Generic lowering has not stated the target beyond the route record.
    Unlowered,
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

    fn write(&self, writer: &mut CanonicalWriter) {
        match &self.target {
            ProblemTarget::Target(target) => {
                writer.text("target").bytes(target.identity().as_bytes());
            }
            ProblemTarget::Functional { kind, outcome } => {
                writer
                    .text("functional")
                    .text(kind.as_str())
                    .opt_word(outcome.map(|id| u64::from(id.raw())));
            }
            ProblemTarget::QueryFunctional { kind, outcome, query } => {
                writer
                    .text("query_functional")
                    .text(kind.as_str())
                    .opt_word(outcome.map(|id| u64::from(id.raw())))
                    .bytes(query);
            }
            ProblemTarget::Unlowered => {
                writer.text("unlowered");
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
    /// The route has not lowered its identification: preparation identifies.
    Deferred,
    /// Identified at compile and retained with the program; preparation installs
    /// these products and does not identify again.
    Retained(Arc<RetainedIdentification>),
    /// The route identifies on each execution.
    PerExecution {
        /// Why the route cannot retain one identification.
        reason: &'static str,
    },
}

/// Identification products compiled with a lowered program.
#[derive(Clone, Debug)]
pub(crate) struct RetainedIdentification {
    /// Checked product bound to the scientific target, when the route builds one.
    pub(crate) product: Option<CompiledIdentificationProduct>,
    /// Products preparation installs, in install order.
    pub(crate) caches: Arc<[IdentificationCache]>,
}

impl RetainedIdentification {
    /// Products preparation installs.
    pub(crate) fn caches(&self) -> &[IdentificationCache] {
        &self.caches
    }
}

impl IdentificationLayer {
    fn write(&self, writer: &mut CanonicalWriter) {
        match self {
            Self::Deferred => {
                writer.text("deferred");
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
        }
    }
}

/// Where a physical design width came from.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum WidthSource {
    /// Encoded from the selected roles and their contrast codings.
    Quoted,
    /// The route does not fit a design matrix.
    NotDesignBased(&'static str),
    /// The planner's bound from the query's variable count.
    SchemaBound,
}

impl WidthSource {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Quoted => "quoted",
            Self::NotDesignBased(_) => "not_design_based",
            Self::SchemaBound => "schema_bound",
        }
    }
}

/// Rows and encoded columns a design-based program fits.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct DesignShape {
    pub(crate) rows: u64,
    pub(crate) columns: u64,
    pub(crate) source: WidthSource,
    pub(crate) direct: bool,
}

impl DesignShape {
    /// Rows the plan was sized for.
    #[must_use]
    pub const fn rows(&self) -> u64 {
        self.rows
    }

    /// Encoded columns, including the intercept.
    #[must_use]
    pub const fn columns(&self) -> u64 {
        self.columns
    }

    /// Where the column count came from.
    #[must_use]
    pub const fn source(&self) -> WidthSource {
        self.source
    }

    /// Whether every design column is numeric, so the design is gathered directly
    /// from borrowed source columns instead of a materialized contrast expansion.
    #[must_use]
    pub const fn direct(&self) -> bool {
        self.direct
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
            cell: self.cell,
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

/// The program one route lowers to. Every plan `Study::compile` returns carries one.
#[derive(Clone, Debug)]
pub struct LoweredProgram {
    cell: RouteCell,
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
            IdentificationLayer::Deferred | IdentificationLayer::PerExecution { .. } => None,
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

    /// Route cell, for dispatch verification.
    pub(crate) const fn cell(&self) -> RouteCell {
        self.cell
    }
}

const fn data_classification_name(data: DataClassification) -> &'static str {
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
