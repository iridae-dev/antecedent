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
use antecedent_core::reason_details::native_claims;
use antecedent_core::{CanonicalWriter, ExecutionContext, ResponseIdentification, SemanticDigest};
use antecedent_design::decision_contract::{SourceRepresentation, SourceRequirement};
use antecedent_design::preference_contract::PreferenceContract;
use antecedent_design::preference_decision::{MeasureOrigin, PreferenceMeasure};
use antecedent_design::preference_evaluation::{
    PreferenceEvaluation, PreferenceEvaluationOptions, RouteChoice, StructuralPreferenceEvaluation,
    preference_route,
};

pub use bind::BoundPreferenceContract;
pub use design::DesignBootstrap;
pub(crate) use evidence::EvidenceRows;
pub use evidence::{CrossStudyCovariance, RowProvenance, SharedEvidence, UnitKey};
pub use joint::PreferenceJointLaw;
pub use outcomes::{OutcomeDependence, compile_preference_decision_across_studies};
pub use outcomes::{
    compile_preference_decision_across_prepared_studies,
    compile_preference_decision_with_shared_evidence,
};
pub use structures::{CompiledStructure, StructureIdentification};

use super::Study;
use crate::analysis::contract::CausalContract;
use crate::analysis::joint_panel::JointPanelError;
use crate::analysis::lowering::CompileGoal;
use crate::analysis::native_claims::{
    NativeDecisionInput, NativeDecisionSource, NativeResponseClaim, NativeResponseContext,
};
use crate::analysis::preference_decision::decide_with_context;
use crate::planner::PlanView;
use crate::planner::lowered::{CarrierDemand, DecisionProgram, PreferenceValueFacts};
use crate::result::{StructuralResponseMixture, StructuralWeightBasis};
use crate::{CausalError, PreparedStudy};
use bind::BoundStudy;
use structures::compile_structure;

const STAGE_BIND: &str = "bind";
const STAGE_COMPILE: &str = "compile";
const STAGE_EVALUATE: &str = "evaluate";
const CANCELLED_STAGE: &str = "compiled_preference_decision";
const BOUND_TAG: &str = "antecedent.preference_route.bound.v1";
const MEAN_FUNCTIONAL: &str = "mean";
const IDENTITY_TRANSFORM: &str = "identity";
const UNDECLARED_UNITS: &str = "undeclared";

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

const MEASURE_TAG: &str = "antecedent.preference_route.measure";
const OPTIONS_TAG: &str = "antecedent.preference_route.evaluation_options";

/// The identity the executed `result` reports, checked against the plan compiled for it: the
/// decision program the plan states is the program that ran.
fn executed_identity(
    plan: Option<&PlanView>,
    result: &crate::StudyResult,
) -> Result<Option<SemanticDigest>, PreferenceRouteError> {
    let compiled = plan.and_then(PlanView::semantic_identity);
    let executed = result.lowered_program_identity();
    if executed != compiled {
        let hex = |digest: Option<SemanticDigest>| {
            digest.map_or_else(|| "none".to_owned(), |digest| digest.to_hex())
        };
        return Err(ExternalRefusal::new(
            antecedent_core::reason_code!("route_not_lowered"),
            STAGE_EVALUATE,
            antecedent_core::reason_details::preference_compile::EXECUTED_PROGRAM_MISMATCH,
        )
        .with_expected(format!("the compiled decision program {}", hex(compiled)))
        .with_supplied(format!("a result of the lowered program {}", hex(executed)))
        .with_remedy("compile the decision again from the study, and execute what it compiled")
        .into());
    }
    Ok(executed)
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
    writer.word(measure.unresolved.len() as u64);
    for &index in &measure.unresolved {
        writer.word(index as u64);
    }
    writer.finish_hex()
}

/// Identity of the options of the exact and certified routes an execution read.
fn options_identity(options: &PreferenceEvaluationOptions) -> String {
    let mut writer = CanonicalWriter::new(OPTIONS_TAG);
    let exact = &options.exact;
    writer
        .real(exact.tie_tolerance)
        .word(exact.vertex_cap as u64)
        .real(exact.lp.feasibility_tolerance)
        .real(exact.lp.optimality_tolerance)
        .word(exact.lp.max_pivots as u64);
    let certified = &options.certified;
    writer
        .word(certified.max_boxes as u64)
        .real(certified.min_width_fraction)
        .real(certified.tie_tolerance);
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

/// One scientific quantity the decision reads.
#[derive(Clone, Debug, PartialEq)]
pub struct DemandedQuantity {
    /// The coordinate.
    pub quantity: ScientificQuantity,
    /// Index of the source that targets it: the study in declaration order, or zero for a
    /// single study or design.
    pub source: usize,
    /// `(action, attribute)` of every input that reads it.
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
    /// The route that answers the contract, with its reason.
    pub route: RouteChoice,
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

/// A decision across a range of preferences compiled with the studies that estimate its
/// quantities.
#[derive(Clone, Debug)]
pub struct CompiledPreferenceDecision {
    contract: PreferenceContract,
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
}

fn means_requirement(representation: SourceRepresentation) -> SourceRequirement {
    SourceRequirement { any_of: vec![representation], sampled_needs_error_receipt: false }
}

/// The decision program layer four carries for one study or design.
fn decision_program(
    contract: &PreferenceContract,
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
        actions: contract.actions.len(),
        constraints: contract.constraints.len(),
        chance_threshold: contract
            .constraints
            .iter()
            .map(|constraint| constraint.min_probability)
            .fold(0.0_f64, f64::max),
        utility_unit: Some(Arc::from(contract.model.utility_units.as_str())),
        contract_identity: Some(Arc::from(identity)),
        information_value: None,
        scenario_source: None,
    }
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
    contract: &PreferenceContract,
    quantities: Vec<DemandedQuantity>,
    draws_required: bool,
) -> Result<(PreferenceDemand, CarrierDemand), PreferenceRouteError> {
    let route = preference_route(contract).map_err(|e| e.to_refusal())?;
    let requirement = crate::analysis::preference_decision::source_requirement(contract)?;
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

/// Compile `entries` for `contract`: bind, derive the demand, check the declared dependence
/// (and, for studies that share observations, the declared cross-study covariance), declare
/// and check the chain that joins separately estimated outcomes, then compile and prepare
/// every structure of every study.
pub(super) fn compile_entries(
    entries: &[StudyEntry<'_>],
    contract: &PreferenceContract,
    (dependence, shared): (OutcomeDependence, &[CrossStudyCovariance]),
    ctx: &ExecutionContext,
) -> Result<CompiledPreferenceDecision, PreferenceRouteError> {
    crate::analysis::lowering::refuse_reentry(ctx)?;
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: CANCELLED_STAGE }.into());
    }
    let pairs: Vec<(&Study, &CausalContract)> =
        entries.iter().map(|entry| (entry.study, &entry.causal)).collect();
    let bound = BoundPreferenceContract::bind(contract, &pairs)?;
    let posterior = entries.iter().any(|entry| entry.study.graph_posterior.is_some());
    if posterior && entries.len() > 1 {
        return Err(unsupported_query("a graph posterior among several separately fitted studies"));
    }
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
                contract.structural_policy,
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
        compile_entries(&[entry], contract, (OutcomeDependence::Undeclared, &[]), ctx)
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
        compile_entries(&[entry], contract, (OutcomeDependence::Undeclared, &[]), ctx)
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
}

impl CompiledPreferenceEvaluation {
    /// The verdict on one source, when the decision read one.
    #[must_use]
    pub const fn single(&self) -> Option<&PreferenceEvaluation> {
        match self {
            Self::Single(evaluation) => Some(evaluation),
            Self::Structures(_) | Self::Sampled(_) => None,
        }
    }

    /// The verdict across structures, when the decision read a graph posterior.
    #[must_use]
    pub const fn structures(&self) -> Option<&StructuralPreferenceEvaluation> {
        match self {
            Self::Structures(evaluation) => Some(evaluation),
            Self::Single(_) | Self::Sampled(_) => None,
        }
    }

    /// The decision on sampled class members, when some class was sampled.
    #[must_use]
    pub fn sampled(
        &self,
    ) -> Option<&crate::analysis::preference_decision::SampledStructuralEvaluation> {
        match self {
            Self::Sampled(evaluation) => Some(evaluation),
            Self::Single(_) | Self::Structures(_) => None,
        }
    }
}

/// What ties an executed decision to what was compiled.
#[derive(Clone, Debug, PartialEq)]
pub struct PreferenceDecisionReceipt {
    /// Identity of the preference contract bound to the studies' causal contracts.
    pub bound_identity: String,
    /// Identity of the preference contract.
    pub contract_identity: String,
    /// `(causal target, causal identification)` of every study, or the program and
    /// dependency identities of a compiled design.
    pub causal: Vec<(String, String)>,
    /// Lowered program identity of every plan the execution ran, as compiled: each
    /// structure's decision plan, the design's program, and the chain that joined separately
    /// estimated outcomes.
    pub programs: Vec<SemanticDigest>,
    /// The lowered program identity each executed result reports, in the order of
    /// [`Self::programs`]. Execution refuses a result whose identity is not its plan's
    /// (`preference_compile.executed_program_mismatch`), so the two agree; a runtime-callback
    /// program, which has no semantic identity, reports none and is not listed.
    pub executed: Vec<SemanticDigest>,
    /// Identity of the measure over the preference set the execution read; `None` without
    /// one.
    pub measure: Option<String>,
    /// Identity of the options of the exact and certified routes the execution read.
    pub options: String,
}

/// An executed compiled decision.
#[derive(Clone, Debug)]
pub struct PreferenceDecisionExecution {
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
}

fn no_response() -> PreferenceRouteError {
    unsupported_query("an execution that published no response curve")
}

impl CompiledPreferenceDecision {
    /// The preference contract.
    #[must_use]
    pub const fn contract(&self) -> &PreferenceContract {
        &self.contract
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
        (programs, executed): (Vec<SemanticDigest>, Vec<SemanticDigest>),
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
        }
    }

    /// Run every compiled plan once and decide across the preference set (and across the
    /// structures of a graph posterior) on what it produced.
    ///
    /// `measure` is the measure over the preference set a weighted policy reads; `options`
    /// tune the exact and certified routes.
    ///
    /// # Errors
    /// Cancellation and resource refusals; an estimation failure of a single structure; a
    /// response the claim conversion refuses (no point identification, missing support,
    /// no supported coordinate); a failed bootstrap slot of a design; a refusal of the chain
    /// that joins separately estimated outcomes; and every refusal of the evaluators.
    pub fn execute(
        &self,
        measure: Option<&PreferenceMeasure>,
        options: PreferenceEvaluationOptions,
        ctx: &ExecutionContext,
    ) -> Result<PreferenceDecisionExecution, PreferenceRouteError> {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: CANCELLED_STAGE }.into());
        }
        match &self.sources {
            Sources::Studies { studies, structures, posterior: true, basis } => self
                .execute_structures(
                    studies,
                    (structures.as_slice(), *basis),
                    measure,
                    options,
                    ctx,
                ),
            Sources::Studies { studies, structures, posterior: false, .. } => {
                self.execute_studies(studies, structures, measure, options, ctx)
            }
            Sources::Design(design) => {
                let (source, programs) = design.execute(self.demand.representation, ctx)?;
                let mut evaluation =
                    decide_with_context(&self.contract, &source, measure, options, ctx)?;
                design.state_premise(&mut evaluation);
                let executed = programs.clone();
                Ok(PreferenceDecisionExecution {
                    evaluation: CompiledPreferenceEvaluation::Single(evaluation),
                    source: Some(source),
                    inputs: Vec::new(),
                    mixture: None,
                    receipt: self.receipt((programs, executed), measure, &options),
                })
            }
        }
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
        let mut programs = Vec::with_capacity(structures.len() + 1);
        let mut executed = Vec::with_capacity(structures.len() + 1);
        for structure in structures {
            let (Some(runner), Some(study)) = (&structure.runner, studies.get(structure.source))
            else {
                return Err(not_identified(&self.demand.quantities, 1.0, "no prepared structure"));
            };
            let result = runner.estimate_retained(ctx)?;
            executed.extend(executed_identity(structure.plan.as_ref(), &result)?);
            let response = retained_response(&result).ok_or_else(no_response)?;
            let claim =
                NativeResponseClaim::from_response(&response, &study.schema, &study.context(ctx))?;
            inputs.push(claim.decision_source(&requirement)?);
            programs.extend(structure.plan.as_ref().map(PlanView::identity));
        }
        let source = if let [input] = inputs.as_slice() {
            input.source.clone()
        } else {
            let (source, chain) =
                outcomes::join_studies(&inputs, studies, self.joint.as_ref(), ctx)?;
            if let Some(chain) = chain {
                programs.extend(self.joint.as_ref().map(|joint| joint.fact().program()));
                executed.push(chain);
            }
            source
        };
        let evaluation = decide_with_context(&self.contract, &source, measure, options, ctx)?;
        Ok(PreferenceDecisionExecution {
            evaluation: CompiledPreferenceEvaluation::Single(evaluation),
            source: Some(source),
            inputs,
            mixture: None,
            receipt: self.receipt((programs, executed), measure, &options),
        })
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
