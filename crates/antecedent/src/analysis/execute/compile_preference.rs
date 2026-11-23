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
//! means suffice and otherwise through a propagation-chain family table under the declared
//! distinct evidence of the studies; the outcomes of one compiled static-regime design
//! ([`crate::analysis::joint_panel::designs::PreparedNativeJointDesign::compile_for_preference_decision`])
//! are read from its arm means or from its aligned whole-subject bootstrap refits.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

mod outcomes;

use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;

use antecedent_contract::{
    ExternalRefusal, ProgramBinding, QuantityMismatch, QuantityRole, ScientificQuantity,
};
use antecedent_core::reason_details::{
    decision_evaluation, decision_evidence, native_claims, program_binding,
};
use antecedent_core::{
    CanonicalWriter, CausalQuery, CausalSchema, ExecutionContext, IdentificationStatus,
    ObservationSpec, ResponseFunctional, ResponseIdentification, SemanticDigest, TargetPopulation,
};
use antecedent_data::TableView;
use antecedent_design::decision_contract::{SourceRepresentation, SourceRequirement};
use antecedent_design::preference_contract::{ActionInput, PreferenceContract};
use antecedent_design::preference_decision::PreferenceMeasure;
use antecedent_design::preference_evaluation::{
    PreferenceEvaluation, PreferenceEvaluationOptions, RouteChoice, StructuralPreferenceEvaluation,
    preference_route,
};
use antecedent_discovery::GraphPosteriorAtomKind;
use antecedent_discovery::adjacency_mask::dag_from_adjacency_mask;
use antecedent_io::distribution_artifact::DistributionCalibration;

pub use outcomes::{
    DesignBootstrap, OutcomeDependence, compile_preference_decision_across_studies,
};

use super::Study;
use crate::accepted::AcceptedGraph;
use crate::analysis::contract::CausalContract;
use crate::analysis::joint_panel::JointPanelError;
use crate::analysis::lowering::CompileGoal;
use crate::analysis::native_claims::{
    NativeDecisionInput, NativeDecisionSource, NativeResponseClaim, NativeResponseContext,
};
use crate::analysis::preference_decision::{
    decide_across_structures_with_context, decide_with_context, structure_claims,
};
use crate::planner::lowered::{CarrierDemand, DecisionProgram};
use crate::planner::plan_view::IdentificationKind;
use crate::planner::{PlanStage, PlanView};
use crate::result::{StructuralResponseAtom, StructuralResponseMixture, StructuralWeightBasis};
use crate::support::StructureSource;
use crate::{CausalError, PreparedStudy};

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
    /// Registered reason code, when the refusal carries one.
    #[must_use]
    pub fn code(&self) -> Option<&str> {
        match self {
            Self::Compiler(error) => error.reason_code(),
            Self::Design(_) => None,
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

// ---------------------------------------------------------------------------------------------
// Binding
// ---------------------------------------------------------------------------------------------

/// The mean response curve a study targets, read from its own query and schema.
#[derive(Clone, Debug)]
struct StudyQuery {
    outcome: String,
    treatment: String,
    outcome_units: Option<String>,
    dose_units: Option<String>,
    doses: Vec<f64>,
    schema: CausalSchema,
    causal_target: String,
    causal_identification: String,
    snapshot: String,
}

/// A study bound to the decision: its program binding and the coordinates it targets.
#[derive(Clone, Debug)]
struct BoundStudy {
    binding: ProgramBinding,
    coordinates: Vec<ScientificQuantity>,
    schema: CausalSchema,
    causal_target: String,
    causal_identification: String,
    snapshot: String,
}

impl BoundStudy {
    /// The response context the claim conversion reads: the compiler-derived binding, the
    /// data snapshot and the execution's seed.
    fn context(&self, ctx: &ExecutionContext) -> NativeResponseContext {
        NativeResponseContext {
            program: self.binding.clone(),
            snapshot_id: self.snapshot.clone(),
            rng_id: format!("antecedent.execution.master_seed:{}", ctx.rng.master_seed()),
            calibration: DistributionCalibration::NotMeasured,
        }
    }
}

fn study_query(study: &Study, causal: &CausalContract) -> Result<StudyQuery, PreferenceRouteError> {
    let CausalQuery::Response(query) = &study.query else {
        return Err(unsupported_query("a query that is not a response"));
    };
    let ResponseFunctional::MeanCurve { outcome, treatment } = &query.functional else {
        return Err(unsupported_query("a response functional other than a mean curve"));
    };
    if query.temporal.is_some()
        || query.observation != ObservationSpec::Complete
        || query.target_population != TargetPopulation::AllObserved
        || !query.outcome_functional.is_mean()
    {
        return Err(unsupported_query(
            "a temporal, partially observed, retargeted or non-mean response curve",
        ));
    }
    let doses = treatment.grid.values().map_err(|_| unsupported_query("an invalid dose grid"))?;
    let crate::analysis::builder::DataInput::Tabular(data) = &study.data else {
        return Err(unsupported_query("a mean response curve of non-tabular data"));
    };
    let schema = data.schema().clone();
    let variable = |id| {
        schema.get(id).map(|v| (v.name.to_string(), v.unit.as_deref().map(str::to_owned))).map_err(
            |_| {
                PreferenceRouteError::from(
                    ExternalRefusal::new(
                        antecedent_core::reason_code!("unknown_variable"),
                        STAGE_BIND,
                        native_claims::UNKNOWN_VARIABLE,
                    )
                    .with_supplied(format!("{id:?}")),
                )
            },
        )
    };
    let (outcome, outcome_units) = variable(*outcome)?;
    let (treatment, dose_units) = variable(treatment.variable)?;
    Ok(StudyQuery {
        outcome,
        treatment,
        outcome_units,
        dose_units,
        doses,
        schema,
        causal_target: causal.identities.target.to_hex(),
        causal_identification: causal.identities.identification.to_hex(),
        snapshot: causal.identities.data_snapshot.to_hex(),
    })
}

/// Every quantity input of `contract`: `(action, attribute, quantity)`.
fn quantity_inputs(
    contract: &PreferenceContract,
) -> impl Iterator<Item = (&str, &str, &ScientificQuantity)> {
    contract.actions.iter().flat_map(move |action| {
        action.inputs.iter().zip(&contract.model.attributes).filter_map(
            move |(input, attribute)| match input {
                ActionInput::Quantity(quantity) => {
                    Some((action.id.as_str(), attribute.id.as_str(), quantity))
                }
                ActionInput::Fixed { .. } => None,
            },
        )
    })
}

/// The study whose outcome `quantity` names.
fn study_of(
    studies: &[StudyQuery],
    (action, attribute): (&str, &str),
    quantity: &ScientificQuantity,
) -> Result<usize, PreferenceRouteError> {
    studies.iter().position(|study| study.outcome == quantity.variable_id).ok_or_else(|| {
        ExternalRefusal::new(
            antecedent_core::reason_code!("quantity_semantics_mismatch"),
            STAGE_BIND,
            program_binding::TREATMENT_OUTCOME_SUBSTITUTION,
        )
        .with_offending(at(action, attribute))
        .with_expected(
            studies.iter().map(|study| study.outcome.as_str()).collect::<Vec<_>>().join(", "),
        )
        .with_supplied(quantity.variable_id.clone())
        .with_remedy("read an outcome the compiled studies estimate")
        .into()
    })
}

/// The refusal of a quantity no coordinate of a study matches, naming the first dimension
/// that differs from the closest coordinate (the one at the same regime, else the first).
fn coordinate_refusal(
    coordinates: &[ScientificQuantity],
    (action, attribute): (&str, &str),
    quantity: &ScientificQuantity,
) -> PreferenceRouteError {
    let reference = coordinates
        .iter()
        .find(|coordinate| coordinate.regime_id == quantity.regime_id)
        .or_else(|| coordinates.first());
    let mismatch = reference.map_or(QuantityMismatch::InvalidCoordinate, |coordinate| {
        quantity.require_same_coordinate(coordinate).err().unwrap_or(QuantityMismatch::Regime)
    });
    let field_of =
        |field: fn(&ScientificQuantity) -> String| reference.map_or_else(String::new, field);
    let (detail, expected, supplied) = match mismatch {
        QuantityMismatch::Variable => (
            program_binding::TREATMENT_OUTCOME_SUBSTITUTION,
            field_of(|c| c.variable_id.clone()),
            quantity.variable_id.clone(),
        ),
        QuantityMismatch::Units => {
            (native_claims::UNITS_CONFLICT, field_of(|c| c.units.clone()), quantity.units.clone())
        }
        QuantityMismatch::Population => (
            program_binding::POPULATION_MISMATCH,
            field_of(|c| c.population_id.clone()),
            quantity.population_id.clone(),
        ),
        QuantityMismatch::Regime => (
            program_binding::DOSE_GRID_CHANGED,
            coordinates.iter().map(|c| c.regime_id.as_str()).collect::<Vec<_>>().join(", "),
            quantity.regime_id.clone(),
        ),
        QuantityMismatch::Horizon
        | QuantityMismatch::Functional
        | QuantityMismatch::Conditioning
        | QuantityMismatch::Transform
        | QuantityMismatch::InvalidCoordinate => (
            decision_evaluation::QUANTITY_NOT_FOUND,
            field_of(ScientificQuantity::canonical_identity),
            quantity.canonical_identity(),
        ),
    };
    ExternalRefusal::new(
        antecedent_core::reason_code!("quantity_semantics_mismatch"),
        STAGE_BIND,
        detail,
    )
    .with_offending(at(action, attribute))
    .with_expected(expected)
    .with_supplied(supplied)
    .with_remedy("read a coordinate the study's query targets; nothing is reinterpreted")
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

/// Bind every quantity input of `contract` to a coordinate of one study, resolving each
/// study's outcome units (the schema's, else the units every input reading it declares).
fn bind_studies(
    contract: &PreferenceContract,
    studies: Vec<StudyQuery>,
) -> Result<(Vec<BoundStudy>, Vec<DemandedQuantity>), PreferenceRouteError> {
    let mut seen = HashSet::new();
    if let Some(repeated) = studies.iter().find(|study| !seen.insert(study.outcome.as_str())) {
        return Err(ExternalRefusal::new(
            antecedent_core::reason_code!("invalid_argument"),
            STAGE_BIND,
            decision_evidence::INVALID_DECLARATION,
        )
        .with_offending(repeated.outcome.clone())
        .with_expected("one compiled study per outcome")
        .with_remedy("estimate each outcome once, or several outcomes of one design jointly")
        .into());
    }
    let mut units: Vec<Option<String>> =
        studies.iter().map(|study| study.outcome_units.clone()).collect();
    for (action, attribute, quantity) in quantity_inputs(contract) {
        let index = study_of(&studies, (action, attribute), quantity)?;
        if units[index].is_none() {
            units[index] = Some(quantity.units.clone());
        }
    }
    let bound: Vec<BoundStudy> = studies
        .into_iter()
        .zip(units)
        .map(|(study, outcome_units)| {
            let binding = ProgramBinding {
                graph_id: study.causal_identification.clone(),
                contract_id: study.causal_target.clone(),
                treatment_id: study.treatment,
                outcome_id: study.outcome,
                population_id: contract.target_population.clone(),
                intervention_kind: "do".to_owned(),
                horizon: 0,
                dose_grid: study.doses,
                dose_units: study.dose_units.unwrap_or_else(|| UNDECLARED_UNITS.to_owned()),
                outcome_units: outcome_units.unwrap_or_else(|| UNDECLARED_UNITS.to_owned()),
                functional_id: MEAN_FUNCTIONAL.to_owned(),
                transform_id: IDENTITY_TRANSFORM.to_owned(),
            };
            BoundStudy {
                coordinates: binding.expected_quantities(),
                binding,
                schema: study.schema,
                causal_target: study.causal_target,
                causal_identification: study.causal_identification,
                snapshot: study.snapshot,
            }
        })
        .collect();
    let mut demanded: Vec<DemandedQuantity> = Vec::new();
    for (action, attribute, quantity) in quantity_inputs(contract) {
        let source = bound
            .iter()
            .position(|study| study.binding.outcome_id == quantity.variable_id)
            .ok_or_else(|| coordinate_refusal(&[], (action, attribute), quantity))?;
        let coordinates = &bound[source].coordinates;
        let Some(coordinate) =
            coordinates.iter().find(|c| quantity.require_same_coordinate(c).is_ok())
        else {
            return Err(coordinate_refusal(coordinates, (action, attribute), quantity));
        };
        let read = (action.to_owned(), attribute.to_owned());
        match demanded.iter_mut().find(|d| d.quantity == *coordinate) {
            Some(entry) => entry.read_by.push(read),
            None => demanded.push(DemandedQuantity {
                quantity: coordinate.clone(),
                source,
                read_by: vec![read],
            }),
        }
    }
    Ok((bound, demanded))
}

fn bound_identity(contract_identity: &str, causal: &[(String, String)]) -> String {
    let mut writer = CanonicalWriter::new(BOUND_TAG);
    writer.text(contract_identity).word(causal.len() as u64);
    for (target, identification) in causal {
        writer.text(target).text(identification);
    }
    writer.finish_hex()
}

/// A preference contract bound to the causal contracts of the studies that estimate its
/// quantities.
#[derive(Clone, Debug)]
pub struct BoundPreferenceContract {
    contract: PreferenceContract,
    contract_identity: String,
    studies: Vec<BoundStudy>,
    quantities: Vec<DemandedQuantity>,
    identity: String,
}

impl BoundPreferenceContract {
    fn bind(
        contract: &PreferenceContract,
        studies: &[(&Study, &CausalContract)],
    ) -> Result<Self, PreferenceRouteError> {
        let contract_identity = contract.identity().map_err(|e| e.to_refusal())?;
        let queries = studies
            .iter()
            .map(|(study, causal)| study_query(study, causal))
            .collect::<Result<Vec<_>, _>>()?;
        let (studies, quantities) = bind_studies(contract, queries)?;
        let causal: Vec<(String, String)> = studies
            .iter()
            .map(|s| (s.causal_target.clone(), s.causal_identification.clone()))
            .collect();
        let identity = bound_identity(&contract_identity, &causal);
        Ok(Self { contract: contract.clone(), contract_identity, studies, quantities, identity })
    }

    /// The preference contract.
    #[must_use]
    pub const fn contract(&self) -> &PreferenceContract {
        &self.contract
    }

    /// The quantities the actions read, each with the study that targets it.
    #[must_use]
    pub fn quantities(&self) -> &[DemandedQuantity] {
        &self.quantities
    }

    /// The coordinates study `source` targets, one per dose of its grid.
    #[must_use]
    pub fn coordinates(&self, source: usize) -> Option<&[ScientificQuantity]> {
        self.studies.get(source).map(|study| &study.coordinates[..])
    }

    /// Canonical identity of the preference contract bound to every study's causal target
    /// and identification.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// `(causal target, causal identification)` of every study, lowercase hex.
    #[must_use]
    pub fn causal_bindings(&self) -> Vec<(String, String)> {
        self.studies
            .iter()
            .map(|s| (s.causal_target.clone(), s.causal_identification.clone()))
            .collect()
    }
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

/// How identification was established for one structure.
#[derive(Clone, Debug, PartialEq)]
pub enum StructureIdentification {
    /// Identified at compile with this status and retained with the program.
    Identified(IdentificationStatus),
    /// The program identifies on each execution; preparation at compile identified it.
    IdentifiedAtPreparation,
    /// Not identified under this structure, with the compiler's reason.
    NotIdentified(String),
}

/// One structure the decision is compiled under.
#[derive(Clone, Debug)]
pub struct CompiledStructure {
    key: Option<u64>,
    weight: f64,
    source: usize,
    identification: StructureIdentification,
    plan: Option<PlanView>,
    runner: Option<PreparedStudy>,
}

impl CompiledStructure {
    /// Graph key of a graph-posterior structure; `None` for a study's single structure.
    #[must_use]
    pub const fn key(&self) -> Option<u64> {
        self.key
    }

    /// Share of the posterior mass on this structure; one for a single structure.
    #[must_use]
    pub const fn weight(&self) -> f64 {
        self.weight
    }

    /// Index of the study the structure belongs to.
    #[must_use]
    pub const fn source(&self) -> usize {
        self.source
    }

    /// How identification was established.
    #[must_use]
    pub const fn identification(&self) -> &StructureIdentification {
        &self.identification
    }

    /// The compiled plan of this structure, with the decision in layer four; `None` when the
    /// structure did not compile because the quantity is not identified under it.
    #[must_use]
    pub const fn plan_view(&self) -> Option<&PlanView> {
        self.plan.as_ref()
    }
}

/// Where the compiled decision reads its quantities.
#[derive(Clone, Debug)]
enum Sources {
    /// Studies, each with one structure or the structures of a graph posterior.
    Studies { studies: Vec<BoundStudy>, structures: Vec<CompiledStructure>, posterior: bool },
    /// A compiled native joint design of several outcomes.
    Design(Box<outcomes::DesignSource>),
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
}

fn means_requirement(representation: SourceRepresentation) -> SourceRequirement {
    SourceRequirement { any_of: vec![representation], sampled_needs_error_receipt: false }
}

/// The decision program layer four carries for one study.
fn decision_program(
    contract: &PreferenceContract,
    identity: &str,
    read: Vec<Arc<str>>,
    (demand, carrier): (&PreferenceDemand, CarrierDemand),
) -> DecisionProgram {
    DecisionProgram {
        carrier,
        utility: DecisionProgram::preference_value_form(
            read,
            demand.representation == SourceRepresentation::Mean,
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

const fn point_identified(status: IdentificationStatus) -> bool {
    matches!(
        status,
        IdentificationStatus::NonparametricallyIdentified
            | IdentificationStatus::IdentifiedUnderParametricRestrictions
    )
}

/// A structure that compiled and prepared, or why the quantity is not identified under it.
type StructureOutcome = Result<(PlanView, StructureIdentification, PreparedStudy), String>;

/// Compile `study` with the decision in layer four, read its identification from the plan
/// and prepare it (or take the caller's prepared handle). An identification refusal is the
/// structure's outcome; any other refusal ends the compile.
fn compile_structure(
    study: &Study,
    goal: &CompileGoal,
    prepared: Option<&PreparedStudy>,
    ctx: &ExecutionContext,
) -> Result<StructureOutcome, PreferenceRouteError> {
    let plan = match study.compile_for_goal(goal, ctx) {
        Ok(plan) => plan,
        Err(error) if error.is_identification_refusal() => return Ok(Err(error.to_string())),
        Err(error) => return Err(error.into()),
    };
    let view =
        PlanView::of_plan(&plan, PlanStage::Compiled, Some(study), Some(ctx.kernel_policy), None)?;
    let identification = view.identification();
    let identified = match identification.kind() {
        IdentificationKind::Retained => match identification.status() {
            Some(status) if point_identified(status) => StructureIdentification::Identified(status),
            Some(status) => {
                return Ok(Err(format!("identification status {}", status.as_str())));
            }
            None => StructureIdentification::IdentifiedAtPreparation,
        },
        IdentificationKind::Deferred => {
            return Ok(Err(identification.reason().unwrap_or("deferred").to_owned()));
        }
        IdentificationKind::PerExecution => StructureIdentification::IdentifiedAtPreparation,
        IdentificationKind::NonGraphical | IdentificationKind::NotClaimed => {
            return Err(unsupported_query("a program without a graphical identification"));
        }
    };
    let runner = match prepared {
        Some(prepared) => prepared.clone(),
        None => match study.prepare(ctx) {
            Ok(prepared) => prepared,
            Err(error) if error.is_identification_refusal() => {
                return Ok(Err(error.to_string()));
            }
            Err(error) => return Err(error.into()),
        },
    };
    Ok(Ok((view, identified, runner)))
}

/// The study answering the query on graph-posterior structure `index`, when the structure is
/// a directed acyclic graph.
fn atom_study(study: &Study, index: usize) -> Option<Study> {
    let posterior = study.graph_posterior.as_ref()?;
    let mask = posterior.atoms().get(index)?;
    let dag = dag_from_adjacency_mask(mask, posterior.n_vars()).ok()?;
    let mut atom = study.clone();
    atom.graph_posterior = None;
    atom.graph = Arc::new(AcceptedGraph::dag(dag));
    atom.structure_source = StructureSource::Explicit;
    atom.clear_identification();
    Some(atom)
}

/// Compile every distinct structure of the study's graph posterior as its own study.
fn compile_atoms(
    study: &Study,
    goal: &CompileGoal,
    source: usize,
    ctx: &ExecutionContext,
) -> Result<Vec<CompiledStructure>, PreferenceRouteError> {
    let Some(posterior) = study.graph_posterior.as_ref() else {
        return Ok(Vec::new());
    };
    if !matches!(posterior.atom_kind(), GraphPosteriorAtomKind::Dag)
        || posterior.lag_masks().is_some()
        || posterior.max_lag().is_some()
    {
        return Err(unsupported_query(
            "a graph posterior whose atoms are not static directed acyclic graphs",
        ));
    }
    let total: f64 = posterior.weights().iter().sum();
    let mut structures: Vec<CompiledStructure> = Vec::new();
    for index in 0..posterior.n_graphs() {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: CANCELLED_STAGE }.into());
        }
        let key = posterior.graph_keys().get(index).copied();
        let weight = posterior.weights().get(index).copied().unwrap_or(0.0) / total;
        if let Some(existing) = structures.iter_mut().find(|s| s.key == key) {
            existing.weight += weight;
            continue;
        }
        let outcome = match atom_study(study, index) {
            Some(atom) => compile_structure(&atom, goal, None, ctx)?,
            None => Err("the structure is not a directed acyclic graph".to_owned()),
        };
        structures.push(match outcome {
            Ok((plan, identification, runner)) => CompiledStructure {
                key,
                weight,
                source,
                identification,
                plan: Some(plan),
                runner: Some(runner),
            },
            Err(why) => CompiledStructure {
                key,
                weight,
                source,
                identification: StructureIdentification::NotIdentified(why),
                plan: None,
                runner: None,
            },
        });
    }
    Ok(structures)
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

/// Compile `entries` for `contract`: bind, derive the demand, check the declared dependence,
/// then compile and prepare every structure of every study.
pub(super) fn compile_entries(
    entries: &[StudyEntry<'_>],
    contract: &PreferenceContract,
    dependence: OutcomeDependence,
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
    outcomes::check_dependence(&bound.studies, &demand, dependence)?;
    let mut structures = Vec::new();
    for (source, entry) in entries.iter().enumerate() {
        let read = demand
            .quantities
            .iter()
            .filter(|q| q.source == source)
            .map(|q| Arc::from(q.quantity.canonical_identity()))
            .collect();
        let program = decision_program(contract, &bound.identity, read, (&demand, carrier));
        let goal = CompileGoal::Decision(program);
        if entry.study.graph_posterior.is_some() {
            structures.extend(compile_atoms(entry.study, &goal, source, ctx)?);
            continue;
        }
        match compile_structure(entry.study, &goal, entry.prepared, ctx)? {
            Ok((plan, identification, runner)) => structures.push(CompiledStructure {
                key: None,
                weight: 1.0,
                source,
                identification,
                plan: Some(plan),
                runner: Some(runner),
            }),
            Err(why) => {
                let read: Vec<DemandedQuantity> =
                    demand.quantities.iter().filter(|q| q.source == source).cloned().collect();
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
        sources: Sources::Studies { studies: bound.studies, structures, posterior },
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
    /// (`effect_not_identified`, with the unidentified mass); a graph posterior whose atoms
    /// are not static directed acyclic graphs; and every refusal of compile and preparation.
    pub fn compile_for_preference_decision(
        &self,
        contract: &PreferenceContract,
        ctx: &ExecutionContext,
    ) -> Result<CompiledPreferenceDecision, PreferenceRouteError> {
        let entry = StudyEntry { study: self, causal: self.inspect()?, prepared: None };
        compile_entries(&[entry], contract, OutcomeDependence::Undeclared, ctx)
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
        compile_entries(&[entry], contract, OutcomeDependence::Undeclared, ctx)
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
}

impl CompiledPreferenceEvaluation {
    /// The verdict on one source, when the decision read one.
    #[must_use]
    pub const fn single(&self) -> Option<&PreferenceEvaluation> {
        match self {
            Self::Single(evaluation) => Some(evaluation),
            Self::Structures(_) => None,
        }
    }

    /// The verdict across structures, when the decision read a graph posterior.
    #[must_use]
    pub const fn structures(&self) -> Option<&StructuralPreferenceEvaluation> {
        match self {
            Self::Structures(evaluation) => Some(evaluation),
            Self::Single(_) => None,
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
    /// Lowered program identity of every plan the execution ran: each structure's decision
    /// plan, the design's program, and the chain that joined separately estimated outcomes.
    pub programs: Vec<SemanticDigest>,
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

/// A cancellation or resource refusal ends the run; any other estimation failure leaves a
/// structure unevaluated.
fn fatal(error: &CausalError) -> bool {
    matches!(error.peeled(), CausalError::Cancelled { .. } | CausalError::Resource { .. })
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

    fn receipt(&self, programs: Vec<SemanticDigest>) -> PreferenceDecisionReceipt {
        PreferenceDecisionReceipt {
            bound_identity: self.identity.clone(),
            contract_identity: self.contract_identity.clone(),
            causal: self.causal.clone(),
            programs,
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
            Sources::Studies { studies, structures, posterior: true } => {
                self.execute_structures(studies, structures, measure, options, ctx)
            }
            Sources::Studies { studies, structures, posterior: false } => {
                self.execute_studies(studies, structures, measure, options, ctx)
            }
            Sources::Design(design) => {
                let (source, programs) = design.execute(self.demand.representation, ctx)?;
                let evaluation =
                    decide_with_context(&self.contract, &source, measure, options, ctx)?;
                Ok(PreferenceDecisionExecution {
                    evaluation: CompiledPreferenceEvaluation::Single(evaluation),
                    source: Some(source),
                    inputs: Vec::new(),
                    mixture: None,
                    receipt: self.receipt(programs),
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
        for structure in structures {
            let (Some(runner), Some(study)) = (&structure.runner, studies.get(structure.source))
            else {
                return Err(not_identified(&self.demand.quantities, 1.0, "no prepared structure"));
            };
            let result = runner.estimate_retained(ctx)?;
            let response = result.response().ok_or_else(no_response)?;
            let claim =
                NativeResponseClaim::from_response(response, &study.schema, &study.context(ctx))?;
            inputs.push(claim.decision_source(&requirement)?);
            programs.extend(structure.plan.as_ref().map(PlanView::identity));
        }
        let source = if let [input] = inputs.as_slice() {
            input.source.clone()
        } else {
            let (source, chain) =
                outcomes::join_studies(&inputs, studies, self.dependence, &self.identity, ctx)?;
            programs.extend(chain);
            source
        };
        let evaluation = decide_with_context(&self.contract, &source, measure, options, ctx)?;
        Ok(PreferenceDecisionExecution {
            evaluation: CompiledPreferenceEvaluation::Single(evaluation),
            source: Some(source),
            inputs,
            mixture: None,
            receipt: self.receipt(programs),
        })
    }

    fn execute_structures(
        &self,
        studies: &[BoundStudy],
        structures: &[CompiledStructure],
        measure: Option<&PreferenceMeasure>,
        options: PreferenceEvaluationOptions,
        ctx: &ExecutionContext,
    ) -> Result<PreferenceDecisionExecution, PreferenceRouteError> {
        let study = studies.first().ok_or_else(no_response)?;
        let mut atoms = Vec::with_capacity(structures.len());
        let mut programs = Vec::with_capacity(structures.len());
        let (mut identified, mut unidentified, mut unevaluable) = (0.0, 0.0, 0.0);
        for structure in structures {
            let graph_key = structure.key.unwrap_or_default();
            let weight = structure.weight;
            let Some(runner) = &structure.runner else {
                unidentified += weight;
                atoms.push(structural_atom(graph_key, weight, None));
                continue;
            };
            programs.extend(structure.plan.as_ref().map(PlanView::identity));
            match runner.estimate_retained(ctx) {
                Ok(result) => {
                    if let Some(response) = result.response() {
                        identified += weight;
                        atoms.push(structural_atom(graph_key, weight, Some(response.clone())));
                    } else {
                        unevaluable += weight;
                        atoms.push(unevaluated_atom(graph_key, weight));
                    }
                }
                Err(error) if fatal(&error) => return Err(error.into()),
                Err(_) => {
                    unevaluable += weight;
                    atoms.push(unevaluated_atom(graph_key, weight));
                }
            }
        }
        let mixture = StructuralResponseMixture {
            weight_basis: StructuralWeightBasis::PosteriorProbability,
            atoms,
            identified_mass: identified,
            unidentified_mass: unidentified,
            unevaluable_mass: unevaluable,
            subsampled_out_mass: 0.0,
            identified_set: None,
            identified_set_interval: None,
            conditional_on_identified: None,
            full_mass_scope: true,
            truncated_atoms: 0,
            aggregation_policy: None,
        };
        let claims = structure_claims(
            &mixture,
            &[],
            &study.schema,
            &study.context(ctx),
            self.contract.structural_policy,
        )?;
        let evaluation =
            decide_across_structures_with_context(&self.contract, &claims, measure, options, ctx)?;
        Ok(PreferenceDecisionExecution {
            evaluation: CompiledPreferenceEvaluation::Structures(evaluation),
            source: None,
            inputs: Vec::new(),
            mixture: Some(mixture),
            receipt: self.receipt(programs),
        })
    }
}

/// A structure with its own response, or one under which the quantity is not identified.
fn structural_atom(
    graph_key: u64,
    weight: f64,
    response: Option<antecedent_core::CausalResponse>,
) -> StructuralResponseAtom {
    let Some(response) = response else {
        return StructuralResponseAtom {
            graph_key,
            weight,
            status: IdentificationStatus::NotIdentified,
            value: None,
            posterior: None,
            response: None,
        };
    };
    let value = match &response.estimate {
        ResponseIdentification::PointIdentified(value) => Some(value.clone()),
        _ => None,
    };
    StructuralResponseAtom {
        graph_key,
        weight,
        status: response.identification_status,
        value,
        posterior: None,
        response: Some(response),
    }
}

/// An identified structure whose estimate did not complete: it carries no response, so the
/// decision reports its mass as unevaluated.
const fn unevaluated_atom(graph_key: u64, weight: f64) -> StructuralResponseAtom {
    StructuralResponseAtom {
        graph_key,
        weight,
        status: IdentificationStatus::NonparametricallyIdentified,
        value: None,
        posterior: None,
        response: None,
    }
}

/// The coordinate of one outcome of a design under one arm.
fn arm_coordinate(
    variable_id: &str,
    name: &str,
    units: &str,
    population: &str,
    regime: &str,
    horizon: u32,
) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable_id.to_owned(),
        variable_name: name.to_owned(),
        role: QuantityRole::Outcome,
        units: units.to_owned(),
        population_id: population.to_owned(),
        regime_id: regime.to_owned(),
        horizon,
        functional_id: MEAN_FUNCTIONAL.to_owned(),
        conditioning: Vec::new(),
        transform_id: IDENTITY_TRANSFORM.to_owned(),
    }
}
