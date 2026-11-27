//! Compile a decision across a range of preferences through the causal compiler.
//!
//! A [`PreferenceContract`] values candidate actions through a value model over outcome
//! attributes whose trade-off parameters range over a declared or learned set. This module
//! makes it a compiled route of the study that estimates those attributes:
//!
//! * binding ([`Study::bind_preference_contract`]): every quantity an action reads must be a
//!   coordinate the study's query targets: its outcome, a regime on its dose grid, the
//!   contract's population, the static horizon, the mean on the identity scale, in the
//!   outcome's units (the schema's, when it declares one). A refusal names the action and the
//!   attribute;
//! * compile ([`Study::compile_for_preference_decision`], and the same on [`PreparedStudy`]):
//!   the demand (the quantities, the representation the contract needs and the carrier it asks
//!   of the statistical program) is lowered into layer four of the study's program, so the
//!   plan view shows the decision's quantities and carrier beside the identification before
//!   anything is fitted. The shared lowering checks the carrier against the route's
//!   statistical realization; identification is checked for every structure the study
//!   carries, each distinct structure of a graph posterior being compiled and prepared as its
//!   own study; and what cannot be answered is refused here;
//! * execute ([`CompiledPreferenceDecision::execute`]): each prepared structure is estimated
//!   once, its response becomes the source the demand named, and the evaluators of
//!   [`crate::analysis::preference_decision`] answer across the preference set and, for a
//!   graph posterior, across structures, with every structure's response retained and the
//!   unidentified and unevaluated mass reported, never renormalized.
//!
//! Several outcomes compile through machinery that already joins them: separately estimated
//! outcomes ([`compile_preference_decision_across_studies`]) meet as one mean source when
//! means suffice and otherwise through a propagation chain declared, checked and compiled at
//! compile with its family table bound at execution, under the declared distinct evidence of
//! the studies or, for studies the evidence ledger finds sharing observations (rows of one
//! source table, or units of one declared identifier space), under a declared cross-study
//! covariance ([`compile_preference_decision_with_shared_evidence`]); the outcomes of one compiled
//! static-regime, cluster-randomized or experiment design
//! ([`crate::analysis::joint_panel::designs::PreparedNativeJointDesign::compile_for_preference_decision`])
//! are read from its arm means or, for a static regime, from its aligned whole-subject
//! bootstrap refits, with the decision attached to layer four of the design's program.
//!
//! A graph posterior's structures are compiled per atom kind: DAG and ADMG atoms as studies
//! of their own structure, equivalence-class atoms per member, weighed by a declared class prior when one is given
//! (see `atoms`).
//!
//! An ordinary [`DecisionContract`], each of whose actions reads its own quantities through
//! its own utility, compiles on the same route ([`Study::compile_for_contract_decision`], the
//! same on [`PreparedStudy`], [`compile_contract_decision_across_studies`] and
//! [`crate::analysis::joint_panel::designs::PreparedNativeJointDesign::compile_for_contract_decision`]):
//! the same binding (a refusal names the action and the input position), demand, lowering and
//! checks, with the representation the contract's own source requirement accepts. Executed, it
//! is the contract's own evaluation ([`evaluate_contract`] on aligned joint draws,
//! [`evaluate_contract_on_means`] on means), so the result and its contract identity are the
//! ones the contract gives on the same source. Over the structures of a graph posterior an
//! ordinary contract is refused at compile (`preference_compile.ordinary_structures_not_compiled`).
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

mod atoms;
mod bind;
mod design;
mod evidence;
mod facts;
mod joint;
mod outcomes;
mod structures;

use std::fmt;
use std::sync::Arc;

use antecedent_contract::{ExternalRefusal, ScientificQuantity};
use antecedent_core::reason_details::{native_claims, preference_compile};
use antecedent_core::{CanonicalWriter, ExecutionContext, ResponseIdentification, SemanticDigest};
use antecedent_design::decision_contract::{
    DecisionContract, HardConstraint, SourceRepresentation, SourceRequirement, StructuralPolicy,
};
use antecedent_design::decision_eval::{
    DecisionEvalError, DecisionResult, evaluate_contract, evaluate_contract_on_means,
};
use antecedent_design::preference_contract::PreferenceContract;
use antecedent_design::preference_decision::{MeasureOrigin, PreferenceMeasure};
use antecedent_design::preference_evaluation::{
    PreferenceEvaluation, PreferenceEvaluationOptions, PreferenceRoute, RouteChoice,
    StructuralPreferenceEvaluation, preference_route,
};
use antecedent_io::method_provenance::MethodProvenanceWire;
use antecedent_io::{execution_digest, execution_identity_from_context};

pub use bind::BoundPreferenceContract;
pub use design::DesignBootstrap;
pub(crate) use evidence::EvidenceRows;
pub use evidence::{CrossStudyCovariance, RowProvenance, SharedEvidence, UnitKey};
pub use joint::PreferenceJointLaw;
pub use outcomes::{
    OutcomeDependence, compile_contract_decision_across_studies,
    compile_preference_decision_across_studies,
};
pub use outcomes::{
    compile_contract_decision_across_prepared_studies,
    compile_preference_decision_across_prepared_studies,
    compile_preference_decision_with_shared_evidence,
};
pub use structures::{CompiledStructure, StructureIdentification};

use super::Study;
use crate::analysis::attached_decision::{ParentDecisionProgram, ParentSpec, lower_parent};
use crate::analysis::contract::CausalContract;
use crate::analysis::decision_result::supplied_child_identity;
use crate::analysis::joint_panel::JointPanelError;
use crate::analysis::lowering::CompileGoal;
use crate::analysis::native_claims::{
    NativeDecisionInput, NativeDecisionSource, NativeResponseClaim, NativeResponseContext,
};
use crate::analysis::preference_decision::decide_with_context;
use crate::analysis::program_handle::CompiledProgramHandle;
use crate::analysis::supplied_identity::Declaration;
use crate::planner::PlanView;
use crate::planner::lowered::{
    CarrierDemand, Composition, DecisionProgram, PreferenceValueFacts, ProgramCell,
};
use crate::result::{StructuralResponseMixture, StructuralWeightBasis};
use crate::{CausalError, PreparedStudy};
use bind::{BoundContract, BoundStudy};
use structures::compile_structure;

const STAGE_BIND: &str = "bind";
const STAGE_COMPILE: &str = "compile";
const STAGE_EVALUATE: &str = "evaluate";
const CANCELLED_STAGE: &str = "compiled_preference_decision";
const BOUND_TAG: &str = "antecedent.preference_route.bound.v1";
const MEAN_FUNCTIONAL: &str = "mean";
const IDENTITY_TRANSFORM: &str = "identity";
const UNDECLARED_UNITS: &str = "undeclared";
/// Tag of a compiled decision's parent program over its child programs.
const PARENT_TAG: &str = "antecedent.attached_decision.preference_route.v1";
/// Tag of the decision's own declaration, which the parent's identity covers beside its
/// children.
const DECLARATION_TAG: &str = "antecedent.attached_decision.preference_route.declaration.v1";

// ---------------------------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------------------------

/// Why a compiled decision across preferences refused.
#[derive(Debug)]
pub enum PreferenceRouteError {
    /// The causal compiler refused: an identification refusal, a carrier the statistical
    /// realization cannot publish, a resource or cancellation refusal, or an estimation
    /// failure, each with its own typed cause and reason code.
    Compiler(Box<CausalError>),
    /// A compiled native joint design refused.
    Design(Box<JointPanelError>),
    /// A binding, source or decision refusal with its registered code, stage and detail.
    Refused(Box<ExternalRefusal>),
}

impl PreferenceRouteError {
    /// Registered reason code, when the refusal carries one. A compiled design's refusal
    /// carries the compiler's code, `route_not_supported` for a causal claim the design does
    /// not earn, `cancelled_no_claim` and `execution_resource_limit` for its engine's
    /// cancellation and resource refusals, and `invalid_argument` for a compilation refusal or
    /// an input or numerical failure of its engine.
    #[must_use]
    pub fn code(&self) -> Option<&str> {
        match self {
            Self::Compiler(error) => error.reason_code(),
            Self::Design(error) => design_code(error),
            Self::Refused(refusal) => Some(refusal.code),
        }
    }

    /// The structured refusal, when the route refused rather than the compiler.
    #[must_use]
    pub fn refusal(&self) -> Option<&ExternalRefusal> {
        match self {
            Self::Refused(refusal) => Some(refusal.as_ref()),
            Self::Compiler(_) | Self::Design(_) => None,
        }
    }

    /// The compiler's refusal, when the compiler refused.
    #[must_use]
    pub fn compiler(&self) -> Option<&CausalError> {
        match self {
            Self::Compiler(error) => Some(error.as_ref()),
            Self::Design(_) | Self::Refused(_) => None,
        }
    }
}

impl fmt::Display for PreferenceRouteError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compiler(error) => write!(formatter, "{error}"),
            Self::Design(error) => write!(formatter, "{error}"),
            Self::Refused(refusal) => {
                write!(
                    formatter,
                    "{}{}: {} at {}",
                    antecedent_core::reason_code::PREFIX,
                    refusal.code,
                    refusal.detail,
                    refusal.stage
                )?;
                if let Some(offending) = &refusal.offending {
                    write!(formatter, "; offending {offending}")?;
                }
                if let Some(expected) = &refusal.expected {
                    write!(formatter, "; expected {expected}")?;
                }
                if let Some(supplied) = &refusal.supplied {
                    write!(formatter, "; supplied {supplied}")?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for PreferenceRouteError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Compiler(error) => Some(error.as_ref()),
            Self::Design(error) => Some(error.as_ref()),
            Self::Refused(_) => None,
        }
    }
}

impl From<CausalError> for PreferenceRouteError {
    fn from(error: CausalError) -> Self {
        Self::Compiler(Box::new(error))
    }
}

impl From<ExternalRefusal> for PreferenceRouteError {
    fn from(refusal: ExternalRefusal) -> Self {
        Self::Refused(Box::new(refusal))
    }
}

impl From<JointPanelError> for PreferenceRouteError {
    fn from(error: JointPanelError) -> Self {
        Self::Design(Box::new(error))
    }
}

/// The registered reason code of a compiled design's refusal.
fn design_code(error: &JointPanelError) -> Option<&str> {
    use antecedent_estimate::joint_panel::PanelError;
    match error {
        JointPanelError::Compiler(error) => error.reason_code(),
        JointPanelError::Unsupported(_) => {
            Some(antecedent_core::reason_code!("route_not_supported"))
        }
        JointPanelError::Engine(PanelError::Cancelled) => {
            Some(antecedent_core::reason_code!("cancelled_no_claim"))
        }
        JointPanelError::Engine(PanelError::Resource(_)) => {
            Some(antecedent_core::reason_code!("execution_resource_limit"))
        }
        JointPanelError::Compile(_)
        | JointPanelError::Engine(PanelError::Invalid(_) | PanelError::Numerical(_)) => {
            Some(antecedent_core::reason_code!("invalid_argument"))
        }
    }
}

const MEASURE_TAG: &str = "antecedent.preference_route.measure.v1";
const OPTIONS_TAG: &str = "antecedent.preference_route.evaluation_options.v1";

/// The identity the executed `result` reports, checked against the plan compiled for it: the
/// decision program the plan states is the program that ran.
fn executed_identity(
    plan: Option<&PlanView>,
    result: &crate::StudyResult,
) -> Result<Option<SemanticDigest>, PreferenceRouteError> {
    checked_executed(plan.and_then(PlanView::semantic_identity), result.lowered_program_identity())
}

/// `executed`, the identity an executed result reports, when it is `compiled`, the identity of
/// the plan compiled for it; otherwise the refusal `preference_compile.executed_program_mismatch`.
pub(super) fn checked_executed(
    compiled: Option<SemanticDigest>,
    executed: Option<SemanticDigest>,
) -> Result<Option<SemanticDigest>, PreferenceRouteError> {
    if executed != compiled {
        let hex = |digest: Option<SemanticDigest>| {
            digest.map_or_else(|| "none".to_owned(), |digest| digest.to_hex())
        };
        return Err(executed_program_mismatch(
            format!("the compiled decision program {}", hex(compiled)),
            format!("a result of the lowered program {}", hex(executed)),
        ));
    }
    Ok(executed)
}

fn executed_program_mismatch(expected: String, supplied: String) -> PreferenceRouteError {
    ExternalRefusal::new(
        antecedent_core::reason_code!("route_not_lowered"),
        STAGE_EVALUATE,
        antecedent_core::reason_details::preference_compile::EXECUTED_PROGRAM_MISMATCH,
    )
    .with_expected(expected)
    .with_supplied(supplied)
    .with_remedy("compile the decision again from the study, and execute what it compiled")
    .into()
}

pub(super) fn child_count_mismatch(expected: usize, supplied: usize) -> PreferenceRouteError {
    executed_program_mismatch(
        format!("{expected} compiled children"),
        format!("{supplied} supplied results"),
    )
}

/// The compiled child's program, snapshot and execution, or the refusal
/// `preference_compile.executed_program_mismatch`. A mismatch does not adopt `result`.
pub(super) fn admit_supplied_child(
    plan: Option<&PlanView>,
    snapshot: &str,
    result: &crate::StudyResult,
    admission: &ExecutionContext,
) -> Result<Option<SemanticDigest>, PreferenceRouteError> {
    let compiled_program = plan.and_then(PlanView::semantic_identity);
    let result_program = result.lowered_program_identity();
    let result_snapshot = result
        .executed_contract()
        .map_or_else(|| "none".to_owned(), |contract| contract.identities.data_snapshot.to_hex());
    let compiled_execution = execution_hex(admission)?;
    let result_execution = digest_hex(result.execution_identity);
    let compiled =
        supplied_child_identity(&digest_hex(compiled_program), snapshot, &compiled_execution);
    let supplied_identity =
        supplied_child_identity(&digest_hex(result_program), &result_snapshot, &result_execution);
    if compiled != supplied_identity {
        return Err(executed_program_mismatch(
            format!(
                "the compiled child program {}, snapshot {snapshot}, execution {compiled_execution}",
                digest_hex(compiled_program)
            ),
            format!(
                "a result of program {}, snapshot {result_snapshot}, execution {result_execution}",
                digest_hex(result_program)
            ),
        ));
    }
    Ok(result_program)
}

fn digest_hex(digest: Option<SemanticDigest>) -> String {
    digest.map_or_else(|| "none".to_owned(), |digest| digest.to_hex())
}

fn execution_hex(ctx: &ExecutionContext) -> Result<String, PreferenceRouteError> {
    let digest = execution_digest(&execution_identity_from_context(ctx)).map_err(|error| {
        PreferenceRouteError::Compiler(Box::new(crate::CausalError::rejected(error)))
    })?;
    Ok(digest.to_hex())
}

/// Identity of the measure over the preference set an execution read.
fn measure_identity(measure: &PreferenceMeasure) -> String {
    let mut writer = CanonicalWriter::new(MEASURE_TAG);
    writer.word(measure.points.len() as u64);
    for point in &measure.points {
        writer.reals(point);
    }
    match &measure.weights {
        Some(weights) => {
            writer.word(1).reals(weights);
        }
        None => {
            writer.word(0);
        }
    }
    match &measure.labels {
        Some(labels) => {
            writer.word(1).texts(labels.as_slice());
        }
        None => {
            writer.word(0);
        }
    }
    match &measure.origin {
        MeasureOrigin::Declared { assumption_id } => {
            writer.text("declared").text(assumption_id);
        }
        MeasureOrigin::ChoiceEstimate { estimate_identity } => {
            writer.text("choice_estimate").text(estimate_identity);
        }
    }
    // The unresolved parameters are a set of indices (validation refuses a repeat): written
    // sorted, so the order a caller listed them in does not reach the identity.
    let mut unresolved = measure.unresolved.clone();
    unresolved.sort_unstable();
    writer.word(unresolved.len() as u64);
    for index in unresolved {
        writer.word(index as u64);
    }
    writer.finish_hex()
}

/// Identity of the options of the exact and certified routes an execution read. Every field
/// is destructured, so an option added to either route cannot be left out of the identity.
fn options_identity(options: &PreferenceEvaluationOptions) -> String {
    let PreferenceEvaluationOptions { exact, certified } = options;
    let antecedent_design::preference_decision::PreferenceDecisionOptions {
        tie_tolerance: exact_tie_tolerance,
        vertex_cap,
        lp,
    } = exact;
    let antecedent_stats::linear_program::LpOptions {
        feasibility_tolerance,
        optimality_tolerance,
        max_pivots,
    } = lp;
    let antecedent_design::preference_certified::CertifiedOptions {
        max_boxes,
        min_width_fraction,
        tie_tolerance: certified_tie_tolerance,
    } = certified;
    let mut writer = CanonicalWriter::new(OPTIONS_TAG);
    writer
        .real(*exact_tie_tolerance)
        .word(*vertex_cap as u64)
        .real(*feasibility_tolerance)
        .real(*optimality_tolerance)
        .word(*max_pivots as u64);
    writer.word(*max_boxes as u64).real(*min_width_fraction).real(*certified_tie_tolerance);
    writer.finish_hex()
}

fn at(action: &str, attribute: &str) -> String {
    format!("action {action} attribute {attribute}")
}

fn unsupported_query(supplied: &str) -> PreferenceRouteError {
    ExternalRefusal::new(
        antecedent_core::reason_code!("route_not_supported"),
        STAGE_BIND,
        native_claims::UNSUPPORTED_ESTIMAND,
    )
    .with_expected(
        "a static mean response curve of one outcome over a dose grid on the observed rows, \
         completely observed",
    )
    .with_supplied(supplied)
    .with_remedy("declare the quantities the actions read as the study's mean response curve")
    .into()
}

fn not_identified(quantities: &[DemandedQuantity], mass: f64, why: &str) -> PreferenceRouteError {
    let read = quantities
        .iter()
        .flat_map(|q| q.read_by.iter().map(|(action, attribute)| at(action, attribute)))
        .collect::<Vec<_>>()
        .join(", ");
    ExternalRefusal::new(
        antecedent_core::reason_code!("effect_not_identified"),
        STAGE_COMPILE,
        native_claims::RESPONSE_NOT_POINT_IDENTIFIED,
    )
    .with_offending(read)
    .with_expected("a point-identified response under some structure the study carries")
    .with_supplied(format!("unidentified mass {mass}: {why}"))
    .with_remedy(
        "declare a structure or design under which the quantities are identified; nothing was \
         fitted",
    )
    .into()
}

/// The contract a compiled decision answers.
#[derive(Clone, Debug)]
pub(super) enum CompiledContract {
    /// An ordinary decision contract: each action's own utility over its own inputs, decided
    /// as declared.
    Ordinary(DecisionContract),
    /// A value over outcome attributes whose trade-off parameters range over a declared or
    /// learned set, decided across that set.
    Tradeoff(PreferenceContract),
}

impl CompiledContract {
    /// Canonical identity of the contract.
    fn identity(&self) -> Result<String, PreferenceRouteError> {
        match self {
            Self::Ordinary(contract) => {
                contract.identity().map_err(|e| PreferenceRouteError::from(e.to_refusal()))
            }
            Self::Tradeoff(contract) => {
                contract.identity().map_err(|e| PreferenceRouteError::from(e.to_refusal()))
            }
        }
    }

    /// The trade-off contract, when the contract is one.
    const fn tradeoff(&self) -> Option<&PreferenceContract> {
        match self {
            Self::Tradeoff(contract) => Some(contract),
            Self::Ordinary(_) => None,
        }
    }

    /// The ordinary contract, when the contract is one.
    const fn ordinary(&self) -> Option<&DecisionContract> {
        match self {
            Self::Ordinary(contract) => Some(contract),
            Self::Tradeoff(_) => None,
        }
    }

    fn target_population(&self) -> &str {
        match self {
            Self::Ordinary(contract) => &contract.target_population,
            Self::Tradeoff(contract) => &contract.target_population,
        }
    }

    const fn structural_policy(&self) -> StructuralPolicy {
        match self {
            Self::Ordinary(contract) => contract.structural_policy,
            Self::Tradeoff(contract) => contract.structural_policy,
        }
    }

    fn actions(&self) -> usize {
        match self {
            Self::Ordinary(contract) => contract.actions.len(),
            Self::Tradeoff(contract) => contract.actions.len(),
        }
    }

    fn constraints(&self) -> &[HardConstraint] {
        match self {
            Self::Ordinary(contract) => &contract.constraints,
            Self::Tradeoff(contract) => &contract.constraints,
        }
    }

    fn utility_units(&self) -> &str {
        match self {
            Self::Ordinary(contract) => &contract.utility_units,
            Self::Tradeoff(contract) => &contract.model.utility_units,
        }
    }
}

/// The refusal of an ordinary contract over the structures of a graph posterior.
fn ordinary_structures_not_compiled() -> PreferenceRouteError {
    ExternalRefusal::new(
        antecedent_core::reason_code!("route_not_supported"),
        STAGE_COMPILE,
        preference_compile::ORDINARY_STRUCTURES_NOT_COMPILED,
    )
    .with_supplied("an ordinary decision contract and a study over a graph posterior")
    .with_expected("a study of one structure, or a contract that states its trade-offs as a value")
    .with_remedy(
        "compile the ordinary contract with a study of one structure, or decide across the \
         structures' responses with the structural decision evaluators; nothing was fitted",
    )
    .into()
}

/// The refusal of a measure over a preference set read by an ordinary contract, which
/// declares no trade-off parameters.
fn measure_not_read() -> PreferenceRouteError {
    ExternalRefusal::new(
        antecedent_core::reason_code!("option_not_applicable"),
        STAGE_EVALUATE,
        preference_compile::ORDINARY_MEASURE_NOT_READ,
    )
    .with_supplied("a measure over a preference set")
    .with_expected("no measure: an ordinary contract declares no trade-off parameters")
    .with_remedy("execute the ordinary contract without a measure; nothing was fitted")
    .into()
}

/// The source an ordinary contract needs, refusing a criterion that reads structure or
/// identified-set inputs, which one compiled source does not supply.
fn ordinary_requirement(
    contract: &DecisionContract,
) -> Result<SourceRequirement, PreferenceRouteError> {
    let structures = || {
        PreferenceRouteError::from(
            DecisionEvalError::StructureInputsRequired(
                "the criterion reads structure or identified-set inputs, which one compiled \
                 source does not supply",
            )
            .to_refusal(),
        )
    };
    let requirement = contract
        .source_requirement()
        .map_err(|e| PreferenceRouteError::from(e.to_refusal()))?
        .ok_or_else(structures)?;
    if requirement.check(&[SourceRepresentation::Mean, SourceRepresentation::JointDraws]).is_err() {
        return Err(structures());
    }
    Ok(requirement)
}

/// One scientific quantity the decision reads.
#[derive(Clone, Debug, PartialEq)]
pub struct DemandedQuantity {
    /// The coordinate.
    pub quantity: ScientificQuantity,
    /// Index of the source that targets it: the study in declaration order, or zero for a
    /// single study or design.
    pub source: usize,
    /// `(action, attribute)` of every input that reads it; for an ordinary contract the
    /// second element is the input's position, `input k`.
    pub read_by: Vec<(String, String)>,
}

// ---------------------------------------------------------------------------------------------
// Compile
// ---------------------------------------------------------------------------------------------

/// What the decision asks of the studies before anything is fitted.
#[derive(Clone, Debug, PartialEq)]
pub struct PreferenceDemand {
    /// The quantities the actions read.
    pub quantities: Vec<DemandedQuantity>,
    /// The route that answers the contract across its preference set, with its reason;
    /// `None` for an ordinary contract, which is evaluated as declared.
    pub route: Option<RouteChoice>,
    /// The representation the contract needs of an analysis
    /// ([`crate::analysis::preference_decision::source_requirement`]).
    pub requirement: SourceRequirement,
    /// The representation the route reads: means when they suffice, else aligned joint
    /// draws (always joint draws across the structures of a graph posterior).
    pub representation: SourceRepresentation,
    /// The carrier layer four of every study's program demands of its statistical program;
    /// `None` for a compiled design, whose own program is executed as compiled.
    pub carrier: Option<CarrierDemand>,
}

/// Where the compiled decision reads its quantities.
#[derive(Clone, Debug)]
enum Sources {
    /// Studies, each with one structure or the structures of a graph posterior.
    Studies {
        studies: Vec<BoundStudy>,
        structures: Vec<CompiledStructure>,
        posterior: bool,
        basis: StructuralWeightBasis,
    },
    /// A compiled native joint design of several outcomes.
    Design(Box<design::DesignSource>),
}

/// A decision across a range of preferences, or an ordinary decision contract, compiled with
/// the studies that estimate its quantities.
#[derive(Clone, Debug)]
pub struct CompiledPreferenceDecision {
    contract: CompiledContract,
    contract_identity: String,
    demand: PreferenceDemand,
    identity: String,
    causal: Vec<(String, String)>,
    dependence: OutcomeDependence,
    sources: Sources,
    /// Declared cross-study covariance blocks of studies that share observations.
    shared: Arc<[CrossStudyCovariance]>,
    /// The chain declared at compile that joins separately estimated outcomes.
    joint: Option<PreferenceJointLaw>,
    /// The decision's own program, composed over the child programs that estimate its
    /// quantities and gated on the decision's support unit.
    parent: ParentDecisionProgram,
}

/// A compiled decision: a decision across a range of preferences or an ordinary decision
/// contract, compiled with the studies that estimate its quantities.
pub type CompiledDecision = CompiledPreferenceDecision;

fn means_requirement(representation: SourceRepresentation) -> SourceRequirement {
    SourceRequirement { any_of: vec![representation], sampled_needs_error_receipt: false }
}

/// The decision program layer four carries for one study or design.
///
/// The utility is the value form that names every quantity the actions read, for an ordinary
/// contract as for a trade-off contract (an ordinary contract states no trade-off parameters),
/// and whether their means answer the decision.
fn decision_program(
    contract: &CompiledContract,
    identity: &str,
    stated: PreferenceValueFacts,
    (demand, carrier): (&PreferenceDemand, CarrierDemand),
) -> DecisionProgram {
    let read = stated.quantities().iter().map(|q| Arc::from(q.identity())).collect();
    DecisionProgram {
        carrier,
        utility: DecisionProgram::preference_value_form(
            read,
            demand.representation == SourceRepresentation::Mean,
            stated,
        ),
        actions: contract.actions(),
        constraints: contract.constraints().len(),
        chance_threshold: contract
            .constraints()
            .iter()
            .map(|constraint| constraint.min_probability)
            .fold(0.0_f64, f64::max),
        utility_unit: Some(Arc::from(contract.utility_units())),
        contract_identity: Some(Arc::from(identity)),
        information_value: None,
        scenario_source: None,
    }
}

/// The parent family, plan id and evaluator of `contract`'s decision: the exact or the
/// certified route across a preference set, or an ordinary contract's own evaluation. A family
/// binds one support unit, so the route the contract takes names the family.
fn parent_family(
    contract: &CompiledContract,
    demand: &PreferenceDemand,
    composition: Composition,
) -> (ProgramCell, &'static str, &'static str) {
    let structures = matches!(composition, Composition::AtomMixture { .. });
    let tradeoff = if structures {
        "preference_evaluation::decide_preferences_structural_with_context"
    } else {
        "preference_evaluation::decide_preferences_with_context"
    };
    match contract {
        CompiledContract::Ordinary(_) => (
            ProgramCell::ContractAttachedDecision,
            "attached.contract_decision.v1",
            if demand.representation == SourceRepresentation::Mean {
                "decision_eval::evaluate_contract_on_means"
            } else {
                "decision_eval::evaluate_contract"
            },
        ),
        CompiledContract::Tradeoff(_) => match demand.route.map(|choice| choice.route) {
            Some(PreferenceRoute::Certified) => (
                ProgramCell::CertifiedPreferenceAttachedDecision,
                "attached.certified_preference_decision.v1",
                tradeoff,
            ),
            Some(PreferenceRoute::Exact) | None => (
                ProgramCell::PreferenceAttachedDecision,
                "attached.preference_decision.v1",
                tradeoff,
            ),
        },
    }
}

/// The parent program of `contract`'s decision, composed over `children` in declared order:
/// the decision's own program, gated on the decision's unit. Its identity covers the
/// children's program identities and the decision's declaration: the bound identity (the
/// contract bound to the children's causal contracts), the contract identity and `weights`,
/// the structure weights a mixture over a graph posterior reads (empty otherwise).
fn parent_of(
    contract: &CompiledContract,
    (demand, program): (&PreferenceDemand, DecisionProgram),
    (bound_identity, contract_identity): (&str, &str),
    (children, weights): (&[Option<SemanticDigest>], &[f64]),
    composition: Composition,
) -> Result<ParentDecisionProgram, PreferenceRouteError> {
    let (cell, plan_id, evaluator) = parent_family(contract, demand, composition);
    let decision = Declaration::new(DECLARATION_TAG)
        .field("bound", bound_identity)
        .field("contract", contract_identity)
        .field("weights", weights)
        .finish();
    lower_parent(ParentSpec {
        cell,
        plan_id,
        tag: PARENT_TAG,
        children,
        decision,
        evaluator,
        composition,
        program,
        claim: Some(antecedent_contract::ClaimKind::PolicyValue),
    })
    .map_err(PreferenceRouteError::from)
}

/// One study entering the compile: its frozen causal contract and, for a prepared handle,
/// the handle that executes it.
pub(super) struct StudyEntry<'a> {
    pub(super) study: &'a Study,
    pub(super) causal: CausalContract,
    pub(super) prepared: Option<&'a PreparedStudy>,
}

/// The demand of `contract` over `quantities`, with the carrier a study's program demands:
/// means when the contract's requirement accepts them and no structure needs a law, aligned
/// joint draws of a posterior otherwise.
fn demand_of(
    contract: &CompiledContract,
    quantities: Vec<DemandedQuantity>,
    draws_required: bool,
) -> Result<(PreferenceDemand, CarrierDemand), PreferenceRouteError> {
    let (route, requirement) = match contract {
        CompiledContract::Tradeoff(contract) => (
            Some(preference_route(contract).map_err(|e| e.to_refusal())?),
            crate::analysis::preference_decision::source_requirement(contract)?,
        ),
        CompiledContract::Ordinary(contract) => (None, ordinary_requirement(contract)?),
    };
    let means = !draws_required && requirement.check(&[SourceRepresentation::Mean]).is_ok();
    let (representation, carrier) = if means {
        (SourceRepresentation::Mean, CarrierDemand::Mean)
    } else {
        (SourceRepresentation::JointDraws, CarrierDemand::ParameterPosterior)
    };
    let demand =
        PreferenceDemand { quantities, route, requirement, representation, carrier: Some(carrier) };
    Ok((demand, carrier))
}

/// Whether the entries decide over a graph posterior, refusing a posterior among several
/// separately fitted studies and an ordinary contract over posterior structures.
fn posterior_entries(
    entries: &[StudyEntry<'_>],
    contract: &CompiledContract,
) -> Result<bool, PreferenceRouteError> {
    let posterior = entries.iter().any(|entry| entry.study.graph_posterior.is_some());
    if posterior && entries.len() > 1 {
        return Err(unsupported_query("a graph posterior among several separately fitted studies"));
    }
    if posterior && contract.ordinary().is_some() {
        return Err(ordinary_structures_not_compiled());
    }
    Ok(posterior)
}

/// The parent program's children, weights and composition. The children, as declared: every
/// structure of every study in study order (`None` for a structure that compiled no program),
/// then the chain that joins separate studies.
fn parent_shape(
    structures: &[CompiledStructure],
    chain: Option<SemanticDigest>,
    posterior: bool,
    sources: usize,
) -> (Vec<Option<SemanticDigest>>, Vec<f64>, Composition) {
    let mut children: Vec<Option<SemanticDigest>> =
        structures.iter().map(|s| s.plan.as_ref().map(PlanView::identity)).collect();
    children.extend(chain.map(Some));
    let weights: Vec<f64> =
        if posterior { structures.iter().map(|s| s.weight).collect() } else { Vec::new() };
    let composition = if posterior {
        Composition::AtomMixture { atoms: structures.len() }
    } else if sources > 1 {
        Composition::Pooled { sources }
    } else {
        Composition::Single
    };
    (children, weights, composition)
}

/// Compile `entries` for `contract`: bind, derive the demand, check the declared dependence
/// (and, for studies that share observations, the declared cross-study covariance), declare
/// and check the chain that joins separately estimated outcomes, then compile and prepare
/// every structure of every study.
pub(super) fn compile_entries(
    entries: &[StudyEntry<'_>],
    contract: CompiledContract,
    (dependence, shared): (OutcomeDependence, &[CrossStudyCovariance]),
    ctx: &ExecutionContext,
) -> Result<CompiledPreferenceDecision, PreferenceRouteError> {
    crate::analysis::lowering::refuse_reentry(ctx)?;
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: CANCELLED_STAGE }.into());
    }
    let pairs: Vec<(&Study, &CausalContract)> =
        entries.iter().map(|entry| (entry.study, &entry.causal)).collect();
    let bound = BoundContract::bind(contract, &pairs)?;
    let contract = &bound.contract;
    let posterior = posterior_entries(entries, contract)?;
    let (demand, carrier) = demand_of(contract, bound.quantities.clone(), posterior)?;
    let overlaps = outcomes::check_dependence(&bound.studies, &demand, (dependence, shared), ctx)?;
    let declared = joint::declare(
        &bound.studies,
        &demand,
        (dependence, shared),
        overlaps.as_ref(),
        &bound.identity,
        ctx,
    )?;
    let mut structures = Vec::new();
    let mut basis = StructuralWeightBasis::PosteriorProbability;
    for (source, entry) in entries.iter().enumerate() {
        let read: Vec<&DemandedQuantity> =
            demand.quantities.iter().filter(|q| q.source == source).collect();
        let stated = facts::value_facts(
            contract,
            &demand,
            &read,
            bound.studies.get(source).map(|study| &study.schema),
            declared.as_ref().map(|law| law.fact().clone()),
            None,
        );
        let program = decision_program(contract, &bound.identity, stated, (&demand, carrier));
        let goal = CompileGoal::Decision(program);
        if entry.study.graph_posterior.is_some() {
            let (compiled, weights) = atoms::compile_posterior(
                entry.study,
                &goal,
                source,
                contract.structural_policy(),
                ctx,
            )?;
            structures.extend(compiled);
            basis = weights;
            continue;
        }
        match compile_structure(entry.study, &goal, entry.prepared, ctx)? {
            Ok(outcome) => structures.push(CompiledStructure::single(source, outcome)),
            Err(why) => {
                let read: Vec<DemandedQuantity> = read.into_iter().cloned().collect();
                return Err(not_identified(&read, 1.0, &why));
            }
        }
    }
    if posterior && structures.iter().all(|s| s.runner.is_none()) {
        return Err(not_identified(&demand.quantities, 1.0, "no structure identifies it"));
    }
    let (children, weights, composition) = parent_shape(
        &structures,
        declared.as_ref().map(|law| law.fact().program()),
        posterior,
        entries.len(),
    );
    let read: Vec<&DemandedQuantity> = demand.quantities.iter().collect();
    let schema = match bound.studies.as_slice() {
        [study] => Some(&study.schema),
        _ => None,
    };
    let stated = facts::value_facts(
        contract,
        &demand,
        &read,
        schema,
        declared.as_ref().map(|law| law.fact().clone()),
        None,
    );
    let program = decision_program(contract, &bound.identity, stated, (&demand, carrier));
    let parent = parent_of(
        contract,
        (&demand, program),
        (bound.identity.as_str(), bound.contract_identity.as_str()),
        (children.as_slice(), weights.as_slice()),
        composition,
    )?;
    let causal = bound.causal_bindings();
    Ok(CompiledPreferenceDecision {
        contract: bound.contract,
        contract_identity: bound.contract_identity,
        demand,
        identity: bound.identity,
        causal,
        dependence,
        sources: Sources::Studies { studies: bound.studies, structures, posterior, basis },
        shared: Arc::from(shared),
        joint: declared,
        parent,
    })
}

impl Study {
    /// Bind `contract` to this study: every quantity an action reads must be a coordinate of
    /// the study's mean response curve. Nothing is compiled or fitted.
    ///
    /// # Errors
    /// An invalid contract; a query that is not a static, completely observed mean response
    /// curve of tabular data on the observed rows (`native_claims.unsupported_estimand`); a
    /// quantity that names another outcome, another regime, other units, another population,
    /// horizon, functional or scale (each naming the action and the attribute).
    pub fn bind_preference_contract(
        &self,
        contract: &PreferenceContract,
    ) -> Result<BoundPreferenceContract, PreferenceRouteError> {
        let causal = self.inspect()?;
        BoundPreferenceContract::bind(contract, &[(self, &causal)])
    }

    /// Compile a decision across the preference set of `contract` with this study.
    ///
    /// The contract is bound to the study, its demand is lowered into layer four of the
    /// study's program (so the plan view shows the carrier and the quantities, by canonical
    /// identity, next to the identification), and every structure the study carries is
    /// compiled and prepared. A graph posterior's distinct structures are compiled one by
    /// one as studies of their own structure. Nothing is fitted.
    ///
    /// # Errors
    /// Every binding refusal of [`Self::bind_preference_contract`]; a carrier the route's
    /// statistical realization cannot publish (a frequentist realization asked for joint
    /// draws, which are required by a value not affine in the quantities, by a criterion
    /// other than an expectation, and by every graph posterior); a quantity not identified
    /// under the study's structure, or under any structure of its graph posterior
    /// (`effect_not_identified`, with the unidentified mass); graph-posterior atoms of a kind
    /// the single-graph compiler cannot compile (`preference_compile.atom_kind_not_compiled`),
    /// an equivalence class above the declared exact limit without a uniform declared measure
    /// that draws a sample (`preference_compile.class_completions_truncated`), a sample the
    /// budget cannot draw (`preference_compile.class_sampling_budget`), a keyed class prior
    /// with a missing or extra member key or ordered masses
    /// (`preference_compile.class_prior_keys_mismatch`), a sample under a measure that is not
    /// uniform (`preference_compile.class_prior_sampling_needs_uniform`), and
    /// `BayesOverStructures` over class atoms without a declared class prior; and every refusal
    /// of compile and preparation.
    pub fn compile_for_preference_decision(
        &self,
        contract: &PreferenceContract,
        ctx: &ExecutionContext,
    ) -> Result<CompiledPreferenceDecision, PreferenceRouteError> {
        let entry = StudyEntry { study: self, causal: self.inspect()?, prepared: None };
        compile_entries(
            &[entry],
            CompiledContract::Tradeoff(contract.clone()),
            (OutcomeDependence::Undeclared, &[]),
            ctx,
        )
    }

    /// Compile an ordinary decision contract with this study: each action reads its own
    /// quantities through its own utility, and every quantity must be a coordinate of the
    /// study's mean response curve.
    ///
    /// The same binding, demand, lowering and checks as
    /// [`Self::compile_for_preference_decision`]: the plan view shows the carrier and the
    /// quantities before anything is fitted, and the representation is the one the
    /// contract's own source requirement accepts (means when they answer it, else aligned
    /// joint draws). Executed, the decision is the contract's own evaluation on that source,
    /// with its result and contract identity. Nothing is fitted.
    ///
    /// # Errors
    /// Every binding refusal of [`Self::compile_for_preference_decision`], a quantity being
    /// named by its action and input position (`input k`); an invalid contract; a criterion
    /// that reads structure or identified-set inputs
    /// (`decision_evaluation.structure_inputs_required`); a study over a graph posterior
    /// (`preference_compile.ordinary_structures_not_compiled`); a carrier the route's
    /// statistical realization cannot publish; a quantity not identified under the study's
    /// structure; and every refusal of compile and preparation.
    pub fn compile_for_contract_decision(
        &self,
        contract: &DecisionContract,
        ctx: &ExecutionContext,
    ) -> Result<CompiledDecision, PreferenceRouteError> {
        let entry = StudyEntry { study: self, causal: self.inspect()?, prepared: None };
        compile_entries(
            &[entry],
            CompiledContract::Ordinary(contract.clone()),
            (OutcomeDependence::Undeclared, &[]),
            ctx,
        )
    }
}

impl PreparedStudy {
    /// Compile a decision across the preference set of `contract` with this prepared study:
    /// the same binding, demand, lowering and checks as
    /// [`Study::compile_for_preference_decision`], against the handle's frozen contract.
    /// The handle executes the study's single structure; the structures of a graph
    /// posterior are prepared as studies of their own structure.
    ///
    /// # Errors
    /// As [`Study::compile_for_preference_decision`].
    pub fn compile_for_preference_decision(
        &self,
        contract: &PreferenceContract,
        ctx: &ExecutionContext,
    ) -> Result<CompiledPreferenceDecision, PreferenceRouteError> {
        let study = self.study();
        let prepared = study.graph_posterior.is_none().then_some(self);
        let entry = StudyEntry { study, causal: self.contract()?, prepared };
        compile_entries(
            &[entry],
            CompiledContract::Tradeoff(contract.clone()),
            (OutcomeDependence::Undeclared, &[]),
            ctx,
        )
    }

    /// Compile an ordinary decision contract with this prepared study: the same binding,
    /// demand, lowering and checks as [`Study::compile_for_contract_decision`], against the
    /// handle's frozen contract. The handle executes the study's single structure.
    ///
    /// # Errors
    /// As [`Study::compile_for_contract_decision`].
    pub fn compile_for_contract_decision(
        &self,
        contract: &DecisionContract,
        ctx: &ExecutionContext,
    ) -> Result<CompiledDecision, PreferenceRouteError> {
        let study = self.study();
        let prepared = study.graph_posterior.is_none().then_some(self);
        let entry = StudyEntry { study, causal: self.contract()?, prepared };
        compile_entries(
            &[entry],
            CompiledContract::Ordinary(contract.clone()),
            (OutcomeDependence::Undeclared, &[]),
            ctx,
        )
    }
}

// ---------------------------------------------------------------------------------------------
// Execute
// ---------------------------------------------------------------------------------------------

/// The verdict of a compiled decision.
#[derive(Clone, Debug, PartialEq)]
pub enum CompiledPreferenceEvaluation {
    /// Across the preference set, on one aligned source.
    Single(PreferenceEvaluation),
    /// Across the structures of a graph posterior and the preference set together.
    Structures(StructuralPreferenceEvaluation),
    /// Across structures some of whose equivalence classes were sampled under a declared
    /// uniform measure: what a sample supports, never a verdict over every member.
    Sampled(Box<crate::analysis::preference_decision::SampledStructuralEvaluation>),
    /// An ordinary decision contract's own evaluation on one aligned source.
    Ordinary(Box<DecisionResult>),
}

impl CompiledPreferenceEvaluation {
    /// The verdict on one source, when the decision read one.
    #[must_use]
    pub const fn single(&self) -> Option<&PreferenceEvaluation> {
        match self {
            Self::Single(evaluation) => Some(evaluation),
            Self::Structures(_) | Self::Sampled(_) | Self::Ordinary(_) => None,
        }
    }

    /// The verdict across structures, when the decision read a graph posterior.
    #[must_use]
    pub const fn structures(&self) -> Option<&StructuralPreferenceEvaluation> {
        match self {
            Self::Structures(evaluation) => Some(evaluation),
            Self::Single(_) | Self::Sampled(_) | Self::Ordinary(_) => None,
        }
    }

    /// The decision on sampled class members, when some class was sampled.
    #[must_use]
    pub fn sampled(
        &self,
    ) -> Option<&crate::analysis::preference_decision::SampledStructuralEvaluation> {
        match self {
            Self::Sampled(evaluation) => Some(evaluation),
            Self::Single(_) | Self::Structures(_) | Self::Ordinary(_) => None,
        }
    }

    /// The ordinary contract's result, when the decision compiled an ordinary contract.
    #[must_use]
    pub fn ordinary(&self) -> Option<&DecisionResult> {
        match self {
            Self::Ordinary(result) => Some(result),
            Self::Single(_) | Self::Structures(_) | Self::Sampled(_) => None,
        }
    }
}

/// What ties an executed decision to what was compiled.
#[derive(Clone, Debug, PartialEq)]
pub struct PreferenceDecisionReceipt {
    /// Identity of the contract bound to the studies' causal contracts.
    pub bound_identity: String,
    /// Identity of the preference contract, or of the ordinary decision contract.
    pub contract_identity: String,
    /// `(causal target, causal identification)` of every study, or the program and
    /// dependency identities of a compiled design.
    pub causal: Vec<(String, String)>,
    /// Lowered program identity of every plan the execution ran, as compiled: each
    /// structure's decision plan, the design's program, and the chain that joined separately
    /// estimated outcomes.
    pub programs: Vec<SemanticDigest>,
    /// The lowered program identity each executed result reports, aligned with
    /// [`Self::programs`] position by position. Execution refuses a result whose identity is not
    /// its plan's (`preference_compile.executed_program_mismatch`), so a present entry equals the
    /// program beside it. An entry is `None` where no executed result reports an identity: a
    /// structure whose estimation failed without ending the run (it is left unevaluated), or a
    /// runtime-callback program, which has no semantic identity.
    pub executed: Vec<Option<SemanticDigest>>,
    /// Identity of the measure over the preference set the execution read; `None` without
    /// one.
    pub measure: Option<String>,
    /// Identity of the options of the exact and certified routes the execution was given;
    /// an ordinary contract reads none of them.
    pub options: String,
    /// Identity of the decision's own program, composed over [`Self::programs`] and gated on
    /// the decision's support unit ([`CompiledPreferenceDecision::plan_view`]).
    pub decision_program: SemanticDigest,
}

/// An executed compiled decision.
#[derive(Clone, Debug)]
pub struct PreferenceDecisionExecution {
    /// Method records for the decision and the evidence it read.
    pub method_provenance: MethodProvenanceWire,
    /// The verdict.
    pub evaluation: CompiledPreferenceEvaluation,
    /// The aligned source the decision read; `None` across structures, whose sources are the
    /// responses the mixture retains.
    pub source: Option<NativeDecisionSource>,
    /// The claim conversion of every study's response (support, trust, calibration and
    /// program identity); empty across structures and for a design.
    pub inputs: Vec<NativeDecisionInput>,
    /// Across structures: every structure with its weight, identification and its own
    /// response, and the identified, unidentified and unevaluated mass.
    pub mixture: Option<StructuralResponseMixture>,
    /// What ties the execution to the compile.
    pub receipt: PreferenceDecisionReceipt,
    /// What a release build would have refused and a development build ran under the
    /// execution's context: the decision's own unit and every child program's, as the
    /// context's shared log records them. Always empty in a release build, which refuses
    /// instead.
    pub would_refuse: Vec<antecedent_core::WouldRefuse>,
}

fn no_response() -> PreferenceRouteError {
    unsupported_query("an execution that published no response curve")
}

impl CompiledPreferenceDecision {
    /// The preference contract; `None` for an ordinary decision contract.
    #[must_use]
    pub const fn contract(&self) -> Option<&PreferenceContract> {
        self.contract.tradeoff()
    }

    /// The ordinary decision contract; `None` for a preference contract.
    #[must_use]
    pub const fn decision_contract(&self) -> Option<&DecisionContract> {
        self.contract.ordinary()
    }

    /// Identity of the preference contract, or of the ordinary decision contract.
    #[must_use]
    pub fn contract_identity(&self) -> &str {
        &self.contract_identity
    }

    /// What the decision asks of the studies.
    #[must_use]
    pub const fn demand(&self) -> &PreferenceDemand {
        &self.demand
    }

    /// Identity of the preference contract bound to the studies' causal contracts; layer
    /// four of every compiled plan carries it as the decision contract identity.
    #[must_use]
    pub fn bound_identity(&self) -> &str {
        &self.identity
    }

    /// `(causal target, causal identification)` of every study, or the program and
    /// dependency identities of a compiled design.
    #[must_use]
    pub fn causal_bindings(&self) -> &[(String, String)] {
        &self.causal
    }

    /// The declared dependence between separately estimated outcomes.
    #[must_use]
    pub const fn dependence(&self) -> OutcomeDependence {
        self.dependence
    }

    /// The declared cross-study covariance blocks of studies that share observations; empty
    /// unless the studies were compiled under shared evidence.
    #[must_use]
    pub fn cross_study_covariance(&self) -> &[CrossStudyCovariance] {
        &self.shared
    }

    /// The propagation chain declared, checked and compiled at compile that joins
    /// separately estimated outcomes into one law; `None` when means suffice, for one study
    /// and for a design.
    #[must_use]
    pub const fn joint_law(&self) -> Option<&PreferenceJointLaw> {
        self.joint.as_ref()
    }

    /// The plan view of the compiled chain that joins separately estimated outcomes, beside
    /// the decision plans of [`Self::plan_views`]; `None` when no chain is declared.
    #[must_use]
    pub fn chain_plan_view(&self) -> Option<PlanView> {
        self.joint.as_ref().map(|joint| joint.program().plan_view())
    }

    /// The plan view of the decision's own program: a parent program composed over the child
    /// programs of [`Self::plan_views`] (and the chain of [`Self::chain_plan_view`]), whose
    /// [`PlanView::unit_support`] is the support decision of the decision's unit
    /// (`preference_decision.exact` or `.certified` across a preference set, `decision.contract`
    /// for an ordinary contract). Each child program keeps its own unit.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        self.parent.plan_view()
    }

    /// Identity of the decision's own program: the children's program identities in the order
    /// they were declared and the decision's declaration.
    #[must_use]
    pub fn program_identity(&self) -> SemanticDigest {
        self.parent.identity()
    }

    /// The same decision, its own program gated by `decided` in place of its unit's status.
    #[cfg(test)]
    pub(crate) fn deciding(mut self, decided: crate::support::unit::UnitDecision) -> Self {
        self.parent = self.parent.deciding(decided);
        self
    }

    /// Every compiled structure, including the ones under which the quantities are not
    /// identified; empty for a design.
    #[must_use]
    pub fn structures(&self) -> &[CompiledStructure] {
        match &self.sources {
            Sources::Studies { structures, .. } => structures,
            Sources::Design(_) => &[],
        }
    }

    /// Share of the mass on structures under which the quantities are not identified.
    #[must_use]
    pub fn unidentified_mass(&self) -> f64 {
        self.structures()
            .iter()
            .filter(|s| matches!(s.identification, StructureIdentification::NotIdentified(_)))
            .map(CompiledStructure::weight)
            .sum()
    }

    /// The plan views of every compiled plan: each structure's plan with the decision in
    /// layer four, or the compiled design's program.
    #[must_use]
    pub fn plan_views(&self) -> Vec<PlanView> {
        match &self.sources {
            Sources::Studies { structures, .. } => {
                structures.iter().filter_map(|s| s.plan.clone()).collect()
            }
            Sources::Design(design) => vec![design.plan_view()],
        }
    }

    /// The response context the claim conversion of study `source` reads at an execution
    /// under `ctx`: the compiler-derived program binding, the data snapshot and the seed.
    #[must_use]
    pub fn response_context(
        &self,
        source: usize,
        ctx: &ExecutionContext,
    ) -> Option<NativeResponseContext> {
        match &self.sources {
            Sources::Studies { studies, .. } => studies.get(source).map(|s| s.context(ctx)),
            Sources::Design(_) => None,
        }
    }

    fn receipt(
        &self,
        (programs, executed): (Vec<SemanticDigest>, Vec<Option<SemanticDigest>>),
        measure: Option<&PreferenceMeasure>,
        options: &PreferenceEvaluationOptions,
    ) -> PreferenceDecisionReceipt {
        PreferenceDecisionReceipt {
            bound_identity: self.identity.clone(),
            contract_identity: self.contract_identity.clone(),
            causal: self.causal.clone(),
            programs,
            executed,
            measure: measure.map(measure_identity),
            options: options_identity(options),
            decision_program: self.parent.identity(),
        }
    }

    /// Run every compiled plan once and decide across the preference set (and across the
    /// structures of a graph posterior) on what it produced.
    ///
    /// `measure` is the measure over the preference set a weighted policy reads; `options`
    /// tune the exact and certified routes. An ordinary contract is the contract's own
    /// evaluation on the source ([`CompiledPreferenceEvaluation::Ordinary`]): it reads no
    /// measure and none of the options.
    ///
    /// The decision's own program ([`Self::plan_view`]) passes its support gate first: a
    /// release build refuses a decision whose unit is not supported (`not_supported`), before
    /// anything is fitted; a development build records it in
    /// [`PreferenceDecisionExecution::would_refuse`] and runs.
    ///
    /// # Errors
    /// A release build's refusal of the decision's unit; cancellation and resource refusals;
    /// an estimation failure of a single structure; a response the claim conversion refuses
    /// (no point identification, missing support, no supported coordinate); a failed
    /// bootstrap slot of a design; a refusal of the chain
    /// that joins separately estimated outcomes; a measure given with an ordinary contract
    /// (`preference_compile.ordinary_measure_not_read`), before anything is fitted; and every
    /// refusal of the evaluators.
    pub fn execute(
        &self,
        measure: Option<&PreferenceMeasure>,
        options: PreferenceEvaluationOptions,
        ctx: &ExecutionContext,
    ) -> Result<PreferenceDecisionExecution, PreferenceRouteError> {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: CANCELLED_STAGE }.into());
        }
        if measure.is_some() && self.contract.ordinary().is_some() {
            return Err(measure_not_read());
        }
        // The decision's own unit, before any child program runs; each child then passes its
        // own gate.
        self.parent.gate(ctx).map_err(CausalError::from)?;
        let mut execution = match &self.sources {
            Sources::Studies { studies, structures, posterior: true, basis } => self
                .execute_structures(
                    studies,
                    (structures.as_slice(), *basis),
                    measure,
                    options,
                    ctx,
                    None,
                ),
            Sources::Studies { studies, structures, posterior: false, .. } => {
                self.execute_studies(studies, structures, measure, options, ctx)
            }
            Sources::Design(design) => {
                let (source, programs, executed) =
                    design.execute(self.demand.representation, ctx)?;
                let mut evaluation = self.decide_source(&source, measure, options, ctx)?;
                design.state_premise(&mut evaluation);
                let methods = design.plan_view().method_provenance();
                Ok(PreferenceDecisionExecution {
                    method_provenance: self.parent.result_method_provenance([&methods]),
                    evaluation,
                    source: Some(source),
                    inputs: Vec::new(),
                    mixture: None,
                    receipt: self.receipt((programs, executed), measure, &options),
                    would_refuse: Vec::new(),
                })
            }
        }?;
        execution.would_refuse = ctx.would_refuse();
        if let CompiledPreferenceEvaluation::Ordinary(result) = &mut execution.evaluation {
            result.set_method_provenance(execution.method_provenance.clone());
        }
        Ok(execution)
    }

    /// Decide from `supplied` results instead of estimating the compiled children.
    ///
    /// One result per child that would have been estimated, in that order. A result is
    /// consumed when its lowered program, data snapshot and execution identity match the
    /// compiled child under `ctx`. The legacy producing-context argument is ignored because
    /// the result carries its own execution identity. A mismatch is
    /// `preference_compile.executed_program_mismatch` and the number is not adopted.
    ///
    /// # Errors
    /// The same refusals as [`Self::execute`], except a child is not estimated: a mismatch
    /// of program, snapshot or execution identity, or a count of results other than the
    /// number of compiled children, refuses before any child is fitted.
    pub fn execute_supplied(
        &self,
        supplied: &[crate::StudyResult],
        produced: &ExecutionContext,
        measure: Option<&PreferenceMeasure>,
        options: PreferenceEvaluationOptions,
        ctx: &ExecutionContext,
    ) -> Result<PreferenceDecisionExecution, PreferenceRouteError> {
        let _ = produced;
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: CANCELLED_STAGE }.into());
        }
        if measure.is_some() && self.contract.ordinary().is_some() {
            return Err(measure_not_read());
        }
        self.parent.gate(ctx).map_err(CausalError::from)?;
        let mut execution = match &self.sources {
            Sources::Studies { studies, structures, posterior: true, basis } => self
                .execute_structures(
                    studies,
                    (structures.as_slice(), *basis),
                    measure,
                    options,
                    ctx,
                    Some(supplied),
                ),
            Sources::Studies { studies, structures, posterior: false, .. } => {
                self.execute_studies_supplied(studies, structures, supplied, measure, options, ctx)
            }
            Sources::Design(design) => {
                if supplied.len() != 1 {
                    return Err(child_count_mismatch(1, supplied.len()));
                }
                // The design child is not fitted. A study result of another program is the
                // mismatch; a program that happens to match still has no design snapshot to
                // adopt under, so the number is not taken.
                let compiled = design.plan_view().semantic_identity();
                let executed = supplied[0].lowered_program_identity();
                Err(match checked_executed(compiled, executed) {
                    Err(refusal) => refusal,
                    Ok(_) => executed_program_mismatch(
                        "the compiled design program and its data snapshot".to_owned(),
                        "a study result with no design snapshot".to_owned(),
                    ),
                })
            }
        }?;
        execution.would_refuse = ctx.would_refuse();
        if let CompiledPreferenceEvaluation::Ordinary(result) = &mut execution.evaluation {
            result.set_method_provenance(execution.method_provenance.clone());
        }
        Ok(execution)
    }

    fn execute_studies(
        &self,
        studies: &[BoundStudy],
        structures: &[CompiledStructure],
        measure: Option<&PreferenceMeasure>,
        options: PreferenceEvaluationOptions,
        ctx: &ExecutionContext,
    ) -> Result<PreferenceDecisionExecution, PreferenceRouteError> {
        let requirement = means_requirement(self.demand.representation);
        let mut inputs = Vec::with_capacity(structures.len());
        let mut method_provenance = self.parent.result_method_provenance([]);
        let mut programs = Vec::with_capacity(structures.len() + 1);
        let mut executed = Vec::with_capacity(structures.len() + 1);
        for structure in structures {
            let (Some(runner), Some(study)) = (&structure.runner, studies.get(structure.source))
            else {
                return Err(not_identified(&self.demand.quantities, 1.0, "no prepared structure"));
            };
            let result = runner.estimate_retained(ctx)?;
            let ran = executed_identity(structure.plan.as_ref(), &result)?;
            if let Some(plan) = &structure.plan {
                programs.push(plan.identity());
                executed.push(ran);
            }
            let response = retained_response(&result).ok_or_else(no_response)?;
            let claim =
                NativeResponseClaim::from_response(&response, &study.schema, &study.context(ctx))?;
            inputs.push(claim.decision_source(&requirement)?);
            method_provenance.inherit([&result.method_provenance()]);
        }
        let source = if let [input] = inputs.as_slice() {
            input.source.clone()
        } else {
            let (source, chain) =
                outcomes::join_studies(&inputs, studies, self.joint.as_ref(), ctx)?;
            if let Some(chain) = chain {
                programs.extend(self.joint.as_ref().map(|joint| joint.fact().program()));
                executed.push(Some(chain));
            }
            source
        };
        let evaluation = self.decide_source(&source, measure, options, ctx)?;
        Ok(PreferenceDecisionExecution {
            method_provenance,
            evaluation,
            source: Some(source),
            inputs,
            mixture: None,
            receipt: self.receipt((programs, executed), measure, &options),
            would_refuse: Vec::new(),
        })
    }

    fn execute_studies_supplied(
        &self,
        studies: &[BoundStudy],
        structures: &[CompiledStructure],
        supplied: &[crate::StudyResult],
        measure: Option<&PreferenceMeasure>,
        options: PreferenceEvaluationOptions,
        ctx: &ExecutionContext,
    ) -> Result<PreferenceDecisionExecution, PreferenceRouteError> {
        if supplied.len() != structures.len() {
            return Err(child_count_mismatch(structures.len(), supplied.len()));
        }
        let requirement = means_requirement(self.demand.representation);
        let mut inputs = Vec::with_capacity(structures.len());
        let mut method_provenance = self.parent.result_method_provenance([]);
        let mut programs = Vec::with_capacity(structures.len() + 1);
        let mut executed = Vec::with_capacity(structures.len() + 1);
        for (structure, result) in structures.iter().zip(supplied) {
            let Some(study) = studies.get(structure.source) else {
                return Err(executed_program_mismatch(
                    "a prepared compiled child".to_owned(),
                    "no prepared structure".to_owned(),
                ));
            };
            let ran = admit_supplied_child(structure.plan.as_ref(), &study.snapshot, result, ctx)?;
            if let Some(plan) = &structure.plan {
                programs.push(plan.identity());
                executed.push(ran);
            }
            let response = retained_response(result).ok_or_else(no_response)?;
            let claim =
                NativeResponseClaim::from_response(&response, &study.schema, &study.context(ctx))?;
            inputs.push(claim.decision_source(&requirement)?);
            method_provenance.inherit([&result.method_provenance()]);
        }
        let source = if let [input] = inputs.as_slice() {
            input.source.clone()
        } else {
            let (source, chain) =
                outcomes::join_studies(&inputs, studies, self.joint.as_ref(), ctx)?;
            if let Some(chain) = chain {
                programs.extend(self.joint.as_ref().map(|joint| joint.fact().program()));
                executed.push(Some(chain));
            }
            source
        };
        let evaluation = self.decide_source(&source, measure, options, ctx)?;
        Ok(PreferenceDecisionExecution {
            method_provenance,
            evaluation,
            source: Some(source),
            inputs,
            mixture: None,
            receipt: self.receipt((programs, executed), measure, &options),
            would_refuse: Vec::new(),
        })
    }

    /// Decide on one aligned source: across the preference set for a trade-off contract, by
    /// the contract's own evaluation for an ordinary contract.
    fn decide_source(
        &self,
        source: &NativeDecisionSource,
        measure: Option<&PreferenceMeasure>,
        options: PreferenceEvaluationOptions,
        ctx: &ExecutionContext,
    ) -> Result<CompiledPreferenceEvaluation, PreferenceRouteError> {
        match &self.contract {
            CompiledContract::Tradeoff(contract) => Ok(CompiledPreferenceEvaluation::Single(
                decide_with_context(contract, source, measure, options, ctx)?,
            )),
            CompiledContract::Ordinary(contract) => {
                // The ordinary evaluators read no execution context, so cancellation is
                // checked once before they run.
                if ctx.cancellation.is_cancelled() {
                    return Err(CausalError::Cancelled { stage: CANCELLED_STAGE }.into());
                }
                let result = match source {
                    NativeDecisionSource::Mean(means) => {
                        evaluate_contract_on_means(contract, means)
                    }
                    NativeDecisionSource::JointLaw(law) => evaluate_contract(contract, law),
                }
                .map_err(|e| PreferenceRouteError::from(e.to_refusal()))?;
                Ok(CompiledPreferenceEvaluation::Ordinary(Box::new(result)))
            }
        }
    }
}

/// A structure with its own response, or one under which the quantity is not identified.
/// The response of `result` with the aligned draws its result retained beside it.
///
/// A mixed-graph Bayesian mean curve keeps its joint response-curve posterior (one
/// `response_at_<level>` column per grid point, draws aligned across the grid) on the result
/// and publishes a credible band without draws. The decision reads those columns as the
/// response's own credible draws; a response that already carries draws, or a posterior of
/// any other quantities, is returned as it is.
fn retained_response(result: &crate::StudyResult) -> Option<antecedent_core::CausalResponse> {
    let mut response = result.response()?.clone();
    let width = match &response.estimate {
        ResponseIdentification::PointIdentified(antecedent_core::ResponseValue::Surface {
            mean,
            ..
        }) => mean.len(),
        _ => return Some(response),
    };
    let antecedent_core::ResponseUncertainty::PointwiseBand {
        interpretation: antecedent_core::IntervalInterpretation::Credible,
        draws: draws @ None,
        ..
    } = &mut response.uncertainty
    else {
        return Some(response);
    };
    let Some(posterior) = result.posterior.as_ref() else {
        return Some(response);
    };
    let quantities = &posterior.draws.schema.quantities;
    let grid_columns = quantities.len() == width
        && quantities.iter().all(|quantity| {
            matches!(
                quantity,
                antecedent_prob::PosteriorQuantityKind::Scalar { name }
                    if name.starts_with("response_at_")
            )
        });
    if grid_columns {
        let columns: Option<Vec<Vec<f64>>> =
            (0..width).map(|j| posterior.draws.column(j).ok().map(<[f64]>::to_vec)).collect();
        if let Some(columns) = columns {
            *draws =
                Some(antecedent_core::CredibleDraws::columns(posterior.draws.n_draws, &columns));
        }
    }
    Some(response)
}

#[cfg(test)]
mod tests {
    use antecedent_design::decision_contract::DecisionCriterion;

    use super::*;

    #[test]
    fn an_executed_identity_other_than_the_compiled_one_is_refused() {
        let compiled = SemanticDigest::from_bytes([1; 32]);
        let other = SemanticDigest::from_bytes([2; 32]);
        assert_eq!(checked_executed(Some(compiled), Some(compiled)).unwrap(), Some(compiled));
        assert_eq!(checked_executed(None, None).unwrap(), None);
        for (stated, ran) in [(Some(compiled), Some(other)), (Some(compiled), None)] {
            let error = checked_executed(stated, ran).unwrap_err();
            let refusal = error.refusal().expect("a route refusal");
            assert_eq!(
                refusal.detail,
                antecedent_core::reason_details::preference_compile::EXECUTED_PROGRAM_MISMATCH
            );
        }
    }

    fn measure() -> PreferenceMeasure {
        PreferenceMeasure {
            points: vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
            weights: Some(vec![0.25, 0.75]),
            labels: Some(vec!["cautious".to_owned(), "bold".to_owned()]),
            origin: MeasureOrigin::Declared { assumption_id: "panel".to_owned() },
            unresolved: vec![0, 2],
        }
    }

    /// `measure_identity`: the unresolved indices are a set (written sorted); points, weights
    /// and labels are ordered and aligned; an absent part and its presence word are framed.
    #[test]
    fn site_measure_identity_sets_and_alignment() {
        let base = measure_identity(&measure());
        let mut reordered = measure();
        reordered.unresolved.reverse();
        assert_eq!(measure_identity(&reordered), base, "the unresolved indices are a set");
        let mut fewer = measure();
        fewer.unresolved.pop();
        assert_ne!(measure_identity(&fewer), base);

        let edits: [fn(&mut PreferenceMeasure); 7] = [
            |m| m.points.swap(0, 1),
            |m| m.weights = None,
            |m| m.weights.as_mut().unwrap().swap(0, 1),
            |m| m.labels.as_mut().unwrap().swap(0, 1),
            |m| m.labels = None,
            |m| m.origin = MeasureOrigin::ChoiceEstimate { estimate_identity: "panel".to_owned() },
            |m| m.points[0][0] += 1.0,
        ];
        let mut seen = std::collections::BTreeSet::from([base]);
        for edit in edits {
            let mut changed = measure();
            edit(&mut changed);
            assert!(seen.insert(measure_identity(&changed)), "{changed:?}");
        }
        // Reals enter by their bits.
        let mut zero = measure();
        zero.points[0][0] = 0.0;
        let mut negative_zero = measure();
        negative_zero.points[0][0] = -0.0;
        assert_ne!(measure_identity(&zero), measure_identity(&negative_zero));
    }

    /// `options_identity`: every option of both routes enters, each in its own position.
    #[test]
    fn site_options_identity_reads_every_option() {
        let base = PreferenceEvaluationOptions::default();
        let identity = options_identity(&base);
        let edits: [fn(&mut PreferenceEvaluationOptions); 8] = [
            |o| o.exact.tie_tolerance *= 2.0,
            |o| o.exact.vertex_cap = 10,
            |o| o.exact.lp.feasibility_tolerance *= 2.0,
            |o| o.exact.lp.optimality_tolerance *= 2.0,
            |o| o.exact.lp.max_pivots += 1,
            |o| o.certified.max_boxes = 10,
            |o| o.certified.min_width_fraction *= 2.0,
            |o| o.certified.tie_tolerance *= 2.0,
        ];
        let mut seen = std::collections::BTreeSet::from([identity]);
        for edit in edits {
            let mut changed = base;
            edit(&mut changed);
            assert!(seen.insert(options_identity(&changed)), "{changed:?}");
        }
    }

    /// `bind::bound_identity`: the studies' causal bindings are ordered (study order indexes
    /// the decision's sources) and each pair is framed.
    #[test]
    fn site_bound_identity_orders_and_frames_the_studies() {
        let pair =
            |target: &str, identification: &str| (target.to_owned(), identification.to_owned());
        let causal = vec![pair("t1", "i1"), pair("t2", "i2")];
        let base = bind::bound_identity("contract", &causal);
        let mut swapped = causal.clone();
        swapped.swap(0, 1);
        assert_ne!(bind::bound_identity("contract", &swapped), base, "study order is meaning");
        // Moving text across a field boundary is another binding.
        let shifted = vec![pair("t1i", "1"), pair("t2", "i2")];
        assert_ne!(bind::bound_identity("contract", &shifted), base);
        assert_ne!(bind::bound_identity("contract", &causal[..1]), base);
        assert_ne!(bind::bound_identity("contrac", &[pair("tt1", "i1"), pair("t2", "i2")]), base);
    }

    fn ordinary() -> CompiledContract {
        CompiledContract::Ordinary(DecisionContract {
            actions: Vec::new(),
            utility_units: "units".to_owned(),
            criterion: DecisionCriterion::PosteriorExpectedUtility,
            constraints: Vec::new(),
            target_population: "target".to_owned(),
            horizon: 0,
            structural_policy: StructuralPolicy::ReportOnly,
        })
    }

    fn mean_demand() -> PreferenceDemand {
        PreferenceDemand {
            quantities: Vec::new(),
            route: None,
            requirement: SourceRequirement {
                any_of: vec![SourceRepresentation::Mean],
                sampled_needs_error_receipt: false,
            },
            representation: SourceRepresentation::Mean,
            carrier: Some(CarrierDemand::Mean),
        }
    }

    fn mean_program() -> DecisionProgram {
        DecisionProgram {
            carrier: CarrierDemand::Mean,
            utility: DecisionProgram::utility_form(None),
            actions: 0,
            constraints: 0,
            chance_threshold: 0.0,
            utility_unit: None,
            contract_identity: None,
            information_value: None,
            scenario_source: None,
        }
    }

    /// `parent_of`: the decision's declaration frames the bound and contract identities, and
    /// its structure weights are ordered (aligned with the children, in posterior order);
    /// reals enter by their bits; the children are covered by `parent_identity`.
    #[test]
    fn site_compile_preference_parent_of_orders_weights_and_frames_identities() {
        let children = [
            Some(SemanticDigest::from_bytes([1; 32])),
            Some(SemanticDigest::from_bytes([2; 32])),
            Some(SemanticDigest::from_bytes([3; 32])),
        ];
        let parent = |identities: (&str, &str), weights: &[f64]| {
            parent_of(
                &ordinary(),
                (&mean_demand(), mean_program()),
                identities,
                (&children[..], weights),
                Composition::AtomMixture { atoms: children.len() },
            )
            .ok()
            .map(|program| program.identity())
        };
        crate::identity_laws::ordered_laws(&[0.5, 0.25, 0.125], |weights| {
            parent(("bound", "contract"), weights)
        });
        crate::identity_laws::bit_exact(|x| parent(("bound", "contract"), &[x, 0.25, 0.125]));
        crate::identity_laws::injective(
            &crate::identity_laws::ADVERSARIAL_TEXT,
            |parts| *parts,
            |&parts| parent(parts, &[0.5, 0.25, 0.125]),
        );
    }
}
