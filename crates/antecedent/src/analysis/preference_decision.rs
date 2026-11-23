//! Decide across a range of preferences from what an analysis produces.
//!
//! A value model weighs several outcome attributes through trade-off parameters whose values
//! are a declared or learned range ([`ValueModel`], [`PreferenceSet`]). A
//! [`PreferenceContract`] binds candidate actions to the model: each action reads a scientific
//! quantity an analysis estimates, or a value known without uncertainty, for every attribute.
//! The decision is answered across every admissible parameter value, exactly by linear
//! programs when the value is affine in the parameters and the criterion is an expectation,
//! by certified subdivision otherwise ([`preference_route`]).
//!
//! The way in is the causal compiler. [`crate::Study::compile_for_preference_decision`] (and
//! the same on [`crate::PreparedStudy`]) binds the contract to the study's query, lowers the
//! decision's quantities and carrier into the study's plan, checks identification under every
//! structure the study carries and refuses what cannot be answered before anything is fitted;
//! [`CompiledPreferenceDecision::execute`] runs the plan once and calls the functions below.
//! [`compile_preference_decision_across_studies`] and
//! [`crate::analysis::joint_panel::designs::PreparedNativeJointDesign::compile_for_preference_decision`]
//! compile several outcomes. The functions below are the lower layer that route calls; they
//! read sources a caller already holds.
//!
//! This module connects that capability to the analysis surface and owns no scientific rule
//! of its own:
//!
//! * [`decide`] reads a [`NativeDecisionSource`] (the mean source or aligned joint law a
//!   native response claim hands a decision); [`decide_claim`] first asks the claim for the
//!   representation [`source_requirement`] names, through the claim's own conversion;
//! * [`structure_claims`] reads the structures a study run over a graph posterior or a class
//!   of graphs examined ([`StructuralResponseMixture`]: weights and identification per
//!   structure), with each identified structure's own response, as structural atoms, keeping
//!   unidentified and unevaluated mass, and [`decide_across_structures`] decides across those
//!   structures and the preference set together; when some equivalence class was sampled
//!   under a declared uniform measure rather than enumerated, [`decide_on_sampled_structures`]
//!   reports what a sample supports (a Monte Carlo verdict on the sampled measure, or each
//!   sampled member's own verdict) and never a verdict over every member;
//! * [`analysis_options`] and [`analysis_belief_source`] state what was expected of the
//!   options of a recorded choice from an analysis result, so the compatible-set learner and
//!   the noisy-choice estimate learn from analysis-backed records;
//! * several estimated outcomes reach one source through routes that already put them
//!   together: [`experiment_mean_source`] reads the arm means of a several-outcome experiment
//!   ([`JointExperimentEvidence`]); [`chain_joint_law`] reads the aligned joint output law a
//!   propagation chain builds from separately fitted results under a declared dependence; and
//!   [`joint_law_of`] places laws side by side only when the joint-law rule says their rows
//!   pair, so separately produced draws refuse with that rule's own refusal;
//! * [`decide_accepted`] decides with reviewed, versioned [`AcceptedPreferences`];
//! * [`export_accepted`], [`consume_accepted`], [`record_decision`], [`export_decision`] and
//!   [`consume_decision`] move the two durable records through the checksummed container,
//!   recomputing on consumption.
//!
//! A result over a declared or learned preference set is an assumption range, never a
//! probability, unless a measure over the set was supplied or estimated. Every refusal is the
//! structured refusal of the underlying error, with its registered code and detail.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::result_large_err,
    reason = "DecisionFacadeRefusal carries the decision layer's structured refusal by value on a cold once-per-declaration path; boxing it would change the public error contract"
)]

use std::collections::BTreeMap;
use std::sync::Arc;

use antecedent_contract::propagation_chain::{ChainReading, TableConstruction};
use antecedent_contract::{
    InputAlignment, QuantityRole, ScientificQuantity, check_rows_pairable_across,
};
use antecedent_core::{
    CanonicalWriter, CausalResponse, CausalSchema, ExecutionContext, IdentificationStatus,
};
use antecedent_design::decision_adapters::{
    AdaptedClaims, AdapterError, ClaimProbability, SuppliedClaim, adapt_finite_scenarios,
    adapt_graph_dependent_claims, adapt_weighted_graph_atoms,
};
use antecedent_design::decision_artifact::source_digest;
use antecedent_design::decision_contract::{SourceRepresentation, SourceRequirement};
use antecedent_design::decision_eval::{
    DecisionEvalError, MeanEstimate, MeanSource, evaluate_contract_on_means, utility_draws,
};
use antecedent_design::decision_evidence::{Exposure, JointExperimentEvidence};
use antecedent_design::decision_robustness::AtomSupport;
use antecedent_design::decision_structural::AtomEvidence;
use antecedent_io::IoError;
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io::quantity_wire::DistributionMeaningWire;

pub use antecedent_design::accepted_preferences::{
    AcceptedPreferences, AcceptedPreferencesError, LearningOptions, PendingReview, PreferenceBasis,
    PreferenceContractParts, estimate_accepted, estimate_accepted_with_context,
    learn_compatible_set_accepted, learn_compatible_set_accepted_with_context,
};
pub use antecedent_design::choice_record::{
    BeliefSource, ChoiceFeatures, ChoiceOption, ChoiceRecord, ChoiceRecordError, ChoiceRecordSet,
    ChoiceSetFeatures, OptionBelief, Pooling, UnavailableOption, choice_features,
    choice_features_at,
};
pub use antecedent_design::decision_contract::{
    ActionKind, DecisionCriterion, HardConstraint, StructuralPolicy,
};
pub use antecedent_design::preference_artifact::{
    AcceptedPreferencesArtifact, PreferenceArtifactError, PreferenceDecisionArtifact,
    REPLAY_RELATIVE_TOLERANCE,
};
pub use antecedent_design::preference_certified::{
    AtomVerdict, CertifiedActionOutcome, CertifiedAtomStatus, CertifiedAtomSummary,
    CertifiedOptions, CertifiedPreferenceResult, CertifiedRegion, CertifiedVerdict,
    PreferenceCertifiedError, RegionStatus, StructuralCertifiedResult, StructuralCertifiedVerdict,
};
pub use antecedent_design::preference_choice_model::{
    ASSUMPTION_LOGIT_CHOICE, CALIBRATION_ESTIMATOR, CALIBRATION_INTERVAL_METHOD, CALIBRATION_QUERY,
    CALIBRATION_SHAPE_ESTIMATOR, CALIBRATION_SHAPE_INTERVAL_METHOD, ChoiceModelError,
    ChoiceModelEstimate, ChoiceModelOptions, ChoiceModelResult, CovarianceKind, EstimateOutcome,
    ParameterEstimate, ParameterIntervalMethod, calibration_basis, estimate_choice_model,
    estimate_choice_model_with_context, shape_calibration_basis,
};
pub use antecedent_design::preference_contract::{
    ActionInput, PreferenceAction, PreferenceContract, PreferenceContractError, PreferencePolicy,
    ProbeContract,
};
pub use antecedent_design::preference_decision::{
    AtomChoice, CertifiedRouteReason, MeasureOrigin, PreferenceActionOutcome, PreferenceAtomStatus,
    PreferenceAtomSummary, PreferenceDecisionError, PreferenceDecisionOptions,
    PreferenceDecisionResult, PreferenceMeasure, PreferenceSwitch, PreferenceVerdict,
    StructuralPreferenceResult, StructuralPreferenceVerdict, SwitchThreshold,
};
pub use antecedent_design::preference_evaluation::{
    PreferenceEvaluation, PreferenceEvaluationError, PreferenceEvaluationOptions, PreferenceRoute,
    ROUTE_AFFINE_EXPECTATION, ROUTE_CRITERION_NOT_AFFINE, ROUTE_PART_NOT_LOWERABLE,
    ROUTE_SHAPE_PARAMETERS, ROUTE_VALUE_RELATIONS, ROUTE_WORST_CASE_OF_EXPECTED_REGRET,
    RouteChoice, StructuralPreferenceEvaluation, decide_preferences, decide_preferences_on_means,
    decide_preferences_on_means_with_context, decide_preferences_structural,
    decide_preferences_structural_with_context, decide_preferences_with_context,
    decide_with_accepted, decide_with_accepted_with_context, preference_route,
};
pub use antecedent_design::preference_model::{
    Attribute, BeliefRows, Curve, LinearForm, LinearRelation, MAX_PREFERENCE_DEPTH, PreferenceExpr,
    PreferenceModelError, PreferenceParameter, PreferenceRows, PreferenceSet, RelationOrigin,
    RelationSense, ValueModel, ValueRelation, shape,
};
pub use antecedent_design::preference_set_learning::{
    ASSUMPTION_CHOICES_OPTIMAL, BindingChoice, CellStatus, ChoiceConflict, ConflictKind,
    DEFAULT_MAX_SHAPE_BOXES, DEFAULT_MIN_SHAPE_WIDTH_FRACTION, DeciderOverlap, DeciderSet,
    LearnedPreferenceSet, LearnedRegion, ParameterRange, RangeBasis, RegionCell, SetLearningError,
    SetLearningOptions, SetLearningResult, SetOutcome, UndeterminedSet, learn_compatible_set,
    learn_compatible_set_with_context,
};
pub use antecedent_stats::ShapeSearchOptions;

pub use super::execute::{
    BoundPreferenceContract, CompiledPreferenceDecision, CompiledPreferenceEvaluation,
    CompiledStructure, DemandedQuantity, DesignBootstrap, OutcomeDependence,
    PreferenceDecisionExecution, PreferenceDecisionReceipt, PreferenceDemand, PreferenceRouteError,
    StructureIdentification, compile_preference_decision_across_studies,
};
pub use super::execute::{
    CrossStudyCovariance, PreferenceJointLaw, RowProvenance, SharedEvidence, UnitKey,
    compile_preference_decision_across_prepared_studies,
    compile_preference_decision_with_shared_evidence,
};

use super::decision_contract_facade::DecisionFacadeRefusal;
use super::native_claims::{
    NativeDecisionInput, NativeDecisionSource, NativeResponseClaim, NativeResponseContext,
};
use super::propagation_chain::PropagationChain;
use crate::result::{StructuralResponseMixture, StructuralWeightBasis};

/// Atom identity prefix of a structure read from a [`StructuralResponseMixture`]; the graph
/// key follows it.
pub const GRAPH_ATOM_PREFIX: &str = "graph:";

const BELIEF_DOMAIN: &str = "antecedent.preference_decision.analysis_belief.v1";

/// A decision across a preference set together with the analysis input it read.
#[derive(Clone, Debug, PartialEq)]
pub struct AnalysisPreferenceDecision {
    /// What the analysis handed the decision: the mean source or aligned joint law, with its
    /// coordinates, support, trust, calibration and program identity.
    pub input: NativeDecisionInput,
    /// The verdict across the preference set, from whichever route answered.
    pub evaluation: PreferenceEvaluation,
}

/// The representation an analysis must supply for `contract`.
///
/// The certified route needs aligned joint draws. The exact route needs the expectation of
/// every part of the model's linear form; the requirement is the one the ordinary decision
/// contract of those parts ([`PreferenceContract::term_contract`]) states, so a part affine in
/// the attributes is answered by means and any other part needs draws.
///
/// # Errors
/// An invalid contract, and a refusal of the ordinary term contract.
pub fn source_requirement(
    contract: &PreferenceContract,
) -> Result<SourceRequirement, DecisionFacadeRefusal> {
    let choice = preference_route(contract).map_err(|e| e.to_refusal())?;
    if choice.route == PreferenceRoute::Certified {
        return Ok(joint_draws());
    }
    match contract.term_contract().map_err(|e| e.to_refusal())? {
        Some(terms) => {
            Ok(terms.source_requirement().map_err(|e| e.to_refusal())?.unwrap_or_else(joint_draws))
        }
        None => Ok(joint_draws()),
    }
}

fn joint_draws() -> SourceRequirement {
    SourceRequirement {
        any_of: vec![SourceRepresentation::JointDraws],
        sampled_needs_error_receipt: false,
    }
}

/// Decide `contract` across its preference set on what an analysis supplied.
///
/// A mean source is answered by [`decide_preferences_on_means`] (exact route only); an
/// aligned joint law by [`decide_preferences`] (either route).
///
/// # Errors
/// An invalid contract or measure, a contract that needs joint draws from a mean source
/// (`joint_law_required`), and every refusal of the route that answered.
pub fn decide(
    contract: &PreferenceContract,
    source: &NativeDecisionSource,
    measure: Option<&PreferenceMeasure>,
    options: PreferenceEvaluationOptions,
) -> Result<PreferenceEvaluation, DecisionFacadeRefusal> {
    decide_in(contract, source, measure, options, None)
}

/// [`decide`] under an execution context: the answering route reads the context's
/// cancellation token at its work boundaries.
///
/// # Errors
/// As [`decide`]; a cancelled run is the answering route's cancellation refusal.
pub fn decide_with_context(
    contract: &PreferenceContract,
    source: &NativeDecisionSource,
    measure: Option<&PreferenceMeasure>,
    options: PreferenceEvaluationOptions,
    ctx: &ExecutionContext,
) -> Result<PreferenceEvaluation, DecisionFacadeRefusal> {
    decide_in(contract, source, measure, options, Some(ctx))
}

fn decide_in(
    contract: &PreferenceContract,
    source: &NativeDecisionSource,
    measure: Option<&PreferenceMeasure>,
    options: PreferenceEvaluationOptions,
    ctx: Option<&ExecutionContext>,
) -> Result<PreferenceEvaluation, DecisionFacadeRefusal> {
    match (source, ctx) {
        (NativeDecisionSource::Mean(means), None) => {
            decide_preferences_on_means(contract, means, measure, options)
        }
        (NativeDecisionSource::Mean(means), Some(ctx)) => {
            decide_preferences_on_means_with_context(contract, means, measure, options, ctx)
        }
        (NativeDecisionSource::JointLaw(law), None) => {
            decide_preferences(contract, law, measure, options)
        }
        (NativeDecisionSource::JointLaw(law), Some(ctx)) => {
            decide_preferences_with_context(contract, law, measure, options, ctx)
        }
    }
    .map_err(|e| e.to_refusal())
}

/// Decide `contract` from a native response claim: the claim supplies the representation
/// [`source_requirement`] names (through [`NativeResponseClaim::decision_source`], which
/// withholds unsupported coordinates), and [`decide`] answers on it.
///
/// # Errors
/// The refusals of [`source_requirement`], of the claim's conversion (no representation that
/// satisfies the requirement, no supported coordinate, invalid retained draws) and of
/// [`decide`].
pub fn decide_claim(
    contract: &PreferenceContract,
    claim: &NativeResponseClaim,
    measure: Option<&PreferenceMeasure>,
    options: PreferenceEvaluationOptions,
) -> Result<AnalysisPreferenceDecision, DecisionFacadeRefusal> {
    decide_claim_in(contract, claim, measure, options, None)
}

/// [`decide_claim`] under an execution context, as [`decide_with_context`].
///
/// # Errors
/// As [`decide_claim`]; a cancelled run is the answering route's cancellation refusal.
pub fn decide_claim_with_context(
    contract: &PreferenceContract,
    claim: &NativeResponseClaim,
    measure: Option<&PreferenceMeasure>,
    options: PreferenceEvaluationOptions,
    ctx: &ExecutionContext,
) -> Result<AnalysisPreferenceDecision, DecisionFacadeRefusal> {
    decide_claim_in(contract, claim, measure, options, Some(ctx))
}

fn decide_claim_in(
    contract: &PreferenceContract,
    claim: &NativeResponseClaim,
    measure: Option<&PreferenceMeasure>,
    options: PreferenceEvaluationOptions,
    ctx: Option<&ExecutionContext>,
) -> Result<AnalysisPreferenceDecision, DecisionFacadeRefusal> {
    let requirement = source_requirement(contract)?;
    let input = claim.decision_source(&requirement)?;
    let evaluation = decide_in(contract, &input.source, measure, options, ctx)?;
    Ok(AnalysisPreferenceDecision { input, evaluation })
}

/// Decide with accepted preferences on what an analysis supplied: the contract is built over
/// the accepted model ([`AcceptedPreferences::contract`]) and the accepted measure (an
/// estimate's) is passed when there is one.
///
/// # Errors
/// As [`decide`].
pub fn decide_accepted(
    accepted: &AcceptedPreferences,
    parts: PreferenceContractParts,
    source: &NativeDecisionSource,
    options: PreferenceEvaluationOptions,
) -> Result<PreferenceEvaluation, DecisionFacadeRefusal> {
    let contract = accepted.contract(parts);
    decide(&contract, source, accepted.measure(), options)
}

/// [`decide_accepted`] under an execution context, as [`decide_with_context`].
///
/// # Errors
/// As [`decide_accepted`]; a cancelled run is the answering route's cancellation refusal.
pub fn decide_accepted_with_context(
    accepted: &AcceptedPreferences,
    parts: PreferenceContractParts,
    source: &NativeDecisionSource,
    options: PreferenceEvaluationOptions,
    ctx: &ExecutionContext,
) -> Result<PreferenceEvaluation, DecisionFacadeRefusal> {
    let contract = accepted.contract(parts);
    decide_with_context(&contract, source, accepted.measure(), options, ctx)
}

/// Read the structures a study run over a graph posterior or a class of graphs examined as
/// structural atoms under `policy`.
///
/// `mixture` supplies every examined structure, its weight and whether the quantity is
/// identified under it. Every structure becomes one atom named [`GRAPH_ATOM_PREFIX`] and its
/// graph key:
///
/// * a structure under which the quantity is not identified is an unidentified atom;
/// * an identified structure is evaluated from its own response: the per-structure response
///   the mixture retains, or else the response of `per_structure` with that graph key (an
///   analysis of the same table under that one structure). The response becomes an aligned
///   joint law through the claim conversion of [`NativeResponseClaim`] under `schema` and
///   `context`, with the response's summary support;
/// * an identified structure without a response, or one identified only partially or per
///   graph, is an unevaluated atom naming why.
///
/// No mass is dropped or renormalized: weight the mixture never assigned to an atom stays
/// unassigned, and the evaluators report it.
///
/// Posterior weights are genuine structure probabilities ([`adapt_weighted_graph_atoms`]),
/// and so are posterior atom weights times a declared conditional measure over the members of
/// each class atom ([`StructuralWeightBasis::PosteriorWithDeclaredClassPrior`]), under that
/// declaration; caller-supplied class priors are declared weights ([`adapt_finite_scenarios`]);
/// completion-enumeration weights are not probabilities, so completions are graph-dependent
/// claims without weights ([`adapt_graph_dependent_claims`]).
///
/// # Errors
/// A `per_structure` key that is repeated or names no identified structure of the mixture
/// (`decision_adapters.invalid_claims`); a response the claim conversion refuses (another
/// program, grid or population, missing support labels, no retained joint draws); every
/// refusal of the adapter (including `BayesOverStructures` without genuine probabilities).
pub fn structure_claims(
    mixture: &StructuralResponseMixture,
    per_structure: &[(u64, &CausalResponse)],
    schema: &CausalSchema,
    context: &NativeResponseContext,
    policy: StructuralPolicy,
) -> Result<AdaptedClaims, DecisionFacadeRefusal> {
    let identified = |status: IdentificationStatus| {
        matches!(
            status,
            IdentificationStatus::NonparametricallyIdentified
                | IdentificationStatus::IdentifiedUnderParametricRestrictions
        )
    };
    for (index, (key, _)) in per_structure.iter().enumerate() {
        let repeated = per_structure[..index].iter().any(|(other, _)| other == key);
        let known = mixture.atoms.iter().any(|a| a.graph_key == *key && identified(a.status));
        if repeated || !known {
            return Err(AdapterError::InvalidClaims(
                "each per-structure response names a distinct identified structure",
            )
            .to_refusal());
        }
    }
    let genuine = matches!(
        mixture.weight_basis,
        StructuralWeightBasis::PosteriorProbability
            | StructuralWeightBasis::CallerSuppliedClassPrior
            | StructuralWeightBasis::PosteriorWithDeclaredClassPrior
    );
    let requirement = joint_draws();
    let mut claims = Vec::with_capacity(mixture.atoms.len());
    for atom in &mixture.atoms {
        let response = atom.response.as_ref().or_else(|| {
            per_structure.iter().find(|(key, _)| *key == atom.graph_key).map(|(_, r)| *r)
        });
        let missing = || AtomSupport::uniform(antecedent_core::SupportStatus::MissingEvidence);
        let (evidence, support) = match (atom.status, response) {
            (IdentificationStatus::NotIdentified, _) => (AtomEvidence::Unidentified, missing()),
            (status, Some(response)) if identified(status) => {
                let claim = NativeResponseClaim::from_response(response, schema, context)?;
                let input = claim.decision_source(&requirement)?;
                let NativeDecisionSource::JointLaw(law) = input.source else {
                    return Err(AdapterError::InvalidClaims(
                        "a structure's evidence is an aligned joint law",
                    )
                    .to_refusal());
                };
                (AtomEvidence::Evaluated(law), AtomSupport::uniform(claim.support_status()))
            }
            (status, None) if identified(status) => (
                AtomEvidence::Unevaluated("no_response: the structure was not evaluated".into()),
                missing(),
            ),
            (status, _) => {
                (AtomEvidence::Unevaluated(format!("unevaluable: {}", status.as_str())), missing())
            }
        };
        claims.push(SuppliedClaim {
            id: format!("{GRAPH_ATOM_PREFIX}{}", atom.graph_key),
            probability: if genuine {
                ClaimProbability::Genuine(atom.weight)
            } else {
                ClaimProbability::Unspecified
            },
            evidence,
            support,
        });
    }
    match mixture.weight_basis {
        StructuralWeightBasis::PosteriorProbability
        | StructuralWeightBasis::PosteriorWithDeclaredClassPrior => {
            adapt_weighted_graph_atoms(claims, policy)
        }
        StructuralWeightBasis::CallerSuppliedClassPrior => adapt_finite_scenarios(claims, policy),
        StructuralWeightBasis::CompletionEnumeration => {
            adapt_graph_dependent_claims(claims, policy)
        }
    }
    .map_err(|e| e.to_refusal())
}

/// Decide `contract` across the adapted structures and its preference set together.
///
/// # Errors
/// Claims adapted under a structural policy other than the contract's
/// (`decision_adapters.policy_mismatch`), and every refusal of
/// [`decide_preferences_structural`].
pub fn decide_across_structures(
    contract: &PreferenceContract,
    claims: &AdaptedClaims,
    measure: Option<&PreferenceMeasure>,
    options: PreferenceEvaluationOptions,
) -> Result<StructuralPreferenceEvaluation, DecisionFacadeRefusal> {
    if claims.policy != contract.structural_policy {
        return Err(AdapterError::PolicyMismatch.to_refusal());
    }
    decide_preferences_structural(contract, &claims.atoms, measure, options)
        .map_err(|e| e.to_refusal())
}

/// [`decide_across_structures`] under an execution context: the answering route reads the
/// context's cancellation token at its work boundaries.
///
/// # Errors
/// As [`decide_across_structures`]; a cancelled run is the answering route's cancellation
/// refusal.
pub fn decide_across_structures_with_context(
    contract: &PreferenceContract,
    claims: &AdaptedClaims,
    measure: Option<&PreferenceMeasure>,
    options: PreferenceEvaluationOptions,
    ctx: &ExecutionContext,
) -> Result<StructuralPreferenceEvaluation, DecisionFacadeRefusal> {
    if claims.policy != contract.structural_policy {
        return Err(AdapterError::PolicyMismatch.to_refusal());
    }
    decide_preferences_structural_with_context(contract, &claims.atoms, measure, options, ctx)
        .map_err(|e| e.to_refusal())
}

// ---------------------------------------------------------------------------------------------
// A declared measure over class members, and decisions on a sample of members
// ---------------------------------------------------------------------------------------------

/// Two-sided level of every interval over sampled class masses, and the one-sided level of
/// every bound on the mass where another action is best.
pub const SAMPLED_MASS_LEVEL: f64 = 0.95;

/// How the members of a class atom were drawn.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClassSamplingScheme {
    /// A CPDAG: exactly uniform draws by clique-picking over its chain components, from an
    /// exact count of the class.
    CliquePicking,
    /// A PAG: every circle endpoint a tail or an arrowhead with probability one half, the draw
    /// accepted when it is a member of the class; exactly uniform over the maximal ancestral
    /// graphs the PAG represents, within a search budget.
    CircleEndpointRejection,
}

impl ClassSamplingScheme {
    /// Snake-case name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CliquePicking => "clique_picking",
            Self::CircleEndpointRejection => "circle_endpoint_rejection",
        }
    }
}

/// The number of members of a sampled class.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClassSize {
    /// Counted exactly.
    Exact(u128),
    /// Not counted; at least this many distinct members were seen.
    AtLeast(u128),
}

/// The draw of one class atom whose members were sampled rather than enumerated.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClassSampling {
    /// Graph key of the class atom.
    pub class_key: u64,
    /// How the members were drawn.
    pub scheme: ClassSamplingScheme,
    /// The class size, exact or a lower bound.
    pub class_size: ClassSize,
    /// Members drawn, with replacement.
    pub drawn: usize,
    /// Distinct members among the draws.
    pub distinct: usize,
    /// Seed of the draw; the class atom reads the stream `(seed, class_key)`.
    pub seed: u64,
    /// Endpoint assignments tried, accepted or not (PAG draws); `None` for an exact sampler.
    pub attempts: Option<u64>,
    /// The assumption the uniform measure was declared under.
    pub assumption: Option<String>,
}

/// One member of an equivalence-class atom among the structures of a decision.
#[derive(Clone, Debug, PartialEq)]
pub struct ClassMember {
    /// Content fingerprint of the member ([`antecedent_graph::dag_fingerprint`] of a DAG
    /// member, [`antecedent_graph::mag_fingerprint`] of a MAG member): the key a keyed class
    /// prior names.
    pub fingerprint: u64,
    /// Draws that produced the member; one when the class was enumerated.
    pub multiplicity: usize,
    /// The draw, when the class was sampled.
    pub sampling: Option<Arc<ClassSampling>>,
}

/// A proportion estimated from draws, scaled by the declared mass it is a share of, with an
/// exact Clopper–Pearson interval at [`SAMPLED_MASS_LEVEL`]. A Monte Carlo estimate of a mass
/// under the declared measure, never renormalized.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MassEstimate {
    /// Share of the draws times the class mass.
    pub estimate: f64,
    /// Lower end of the interval, times the class mass.
    pub lower: f64,
    /// Upper end of the interval, times the class mass.
    pub upper: f64,
}

/// What became of the draws of one sampled class.
#[derive(Clone, Debug, PartialEq)]
pub struct SampledClassMass {
    /// Graph key of the class atom.
    pub class_key: u64,
    /// The class atom's posterior mass.
    pub class_mass: f64,
    /// Members drawn.
    pub drawn: usize,
    /// Mass on members whose response was evaluated.
    pub identified: MassEstimate,
    /// Mass on members under which the quantities are not identified.
    pub unidentified: MassEstimate,
    /// Mass on identified members whose response was not evaluated.
    pub unevaluated: MassEstimate,
}

/// The Monte Carlo estimate of the expected-value margin of one action over another at one
/// vertex of the preference set, under the declared measure over structures, with its Monte
/// Carlo standard error. Margins are affine in the trade-off parameters, so the vertices
/// bound them over the whole set.
#[derive(Clone, Debug, PartialEq)]
pub struct SampledMargin {
    /// The earlier action in declaration order.
    pub first: String,
    /// The later action in declaration order.
    pub second: String,
    /// The vertex of the preference set, in parameter order.
    pub vertex: Vec<f64>,
    /// Mass-weighted expected value of `first` minus that of `second` over the evaluated
    /// structures, in the value model's utility units (for a loss criterion smaller values
    /// are better).
    pub margin: f64,
    /// Monte Carlo standard error from the sampled classes; `None` when a sampled class drew
    /// a single member.
    pub standard_error: Option<f64>,
}

/// A one-sided Clopper–Pearson bound on the declared mass of one sampled class on which an
/// action other than the reported one is best.
#[derive(Clone, Debug, PartialEq)]
pub struct OtherBestBound {
    /// Graph key of the class atom.
    pub class_key: u64,
    /// Draws whose member was evaluated with another action best.
    pub disagreeing: usize,
    /// Members drawn.
    pub drawn: usize,
    /// Upper bound at [`SAMPLED_MASS_LEVEL`], times the class mass.
    pub upper: f64,
}

/// What a decision on sampled class members can claim.
#[derive(Clone, Debug, PartialEq)]
pub enum SampledStructuralVerdict {
    /// `BayesOverStructures`: the verdict computed on the sampled measure, a Monte Carlo
    /// estimate of the declared one. No action is claimed best across the whole declared
    /// measure or for every member.
    OnSampledMeasure {
        /// The action the verdict on the sampled measure chooses, if it chooses one.
        best_on_sampled_measure: Option<String>,
        /// The actions still in contention on the sampled measure when none is chosen.
        candidates: Vec<String>,
    },
    /// `Maximin` and `RequireInvariantBestAction` need every member of a class, so no combined
    /// verdict is given from a sample: each sampled member's own verdict is reported.
    PerSampledMember {
        /// The action every evaluated structure (every sampled member included) chooses, if
        /// one does.
        best_on_every_sampled_member: Option<String>,
        /// For that action, per sampled class, a bound on the declared mass where another
        /// action is best.
        other_best: Vec<OtherBestBound>,
    },
    /// `ReportOnly`: per-structure results only.
    ReportOnly,
}

/// A decision across structures some of whose equivalence classes were sampled.
#[derive(Clone, Debug, PartialEq)]
pub struct SampledStructuralEvaluation {
    /// The evaluation on the sampled structures: under the contract's policy for
    /// `BayesOverStructures` and `ReportOnly`; each structure on its own (the report-only
    /// policy) for `Maximin` and `RequireInvariantBestAction`, whose minimum over a sample
    /// would overstate the worst case and is never computed.
    pub evaluation: StructuralPreferenceEvaluation,
    /// What may be claimed from the sample.
    pub verdict: SampledStructuralVerdict,
    /// The draw of every sampled class.
    pub sampling: Vec<ClassSampling>,
    /// The identified, unidentified and unevaluated mass of every sampled class.
    pub masses: Vec<SampledClassMass>,
    /// Pairwise margins at each vertex of the preference set with their Monte Carlo standard
    /// errors (`BayesOverStructures` on the exact route only; empty otherwise).
    pub margins: Vec<SampledMargin>,
    /// Level of the intervals and bounds.
    pub level: f64,
}

impl SampledStructuralEvaluation {
    /// The verdict in plain words, stating that it rests on sampled members.
    #[must_use]
    pub fn statement(&self) -> String {
        let draws: usize = self.sampling.iter().map(|s| s.drawn).sum();
        let scope = format!(
            "Some equivalence classes were sampled ({draws} member draw(s) under a declared \
             uniform measure)"
        );
        match &self.verdict {
            SampledStructuralVerdict::OnSampledMeasure {
                best_on_sampled_measure: Some(action),
                ..
            } => format!(
                "{scope}. On the sampled measure action {action} is preferred; this is a Monte \
                 Carlo estimate of the declared measure, not a claim that {action} is best for \
                 every member."
            ),
            SampledStructuralVerdict::OnSampledMeasure { candidates, .. } => format!(
                "{scope}. On the sampled measure no action is chosen{}.",
                if candidates.is_empty() {
                    String::new()
                } else {
                    format!("; still in contention: {}", candidates.join(", "))
                }
            ),
            SampledStructuralVerdict::PerSampledMember {
                best_on_every_sampled_member: Some(action),
                other_best,
            } => {
                let bound: f64 = other_best.iter().map(|b| b.upper).sum();
                format!(
                    "{scope}. Action {action} is best on every sampled member; the declared \
                     mass on which another action is best is at most {bound:.4} (one-sided, \
                     level {}, summed over the sampled classes). No verdict over every member \
                     is given.",
                    self.level
                )
            }
            SampledStructuralVerdict::PerSampledMember { .. } => format!(
                "{scope}. The sampled members do not all choose one action; no verdict over \
                 every member is given."
            ),
            SampledStructuralVerdict::ReportOnly => {
                format!("{scope}. No action is chosen; per-structure results are reported.")
            }
        }
    }
}

/// The exact two-sided Clopper–Pearson interval for a binomial proportion with `successes`
/// of `trials` at `level`: each end is a beta quantile. `None` for no trials, more successes
/// than trials, or a level outside `(0, 1)`.
#[must_use]
pub fn clopper_pearson(successes: u64, trials: u64, level: f64) -> Option<(f64, f64)> {
    if trials == 0 || successes > trials || !proper_level(level) {
        return None;
    }
    let tail = (1.0 - level) / 2.0;
    let (x, n) = (successes as f64, trials as f64);
    let lower = if successes == 0 { 0.0 } else { beta_quantile(tail, x, n - x + 1.0) };
    let upper = if successes == trials { 1.0 } else { beta_quantile(1.0 - tail, x + 1.0, n - x) };
    Some((lower, upper))
}

/// The exact one-sided Clopper–Pearson upper bound for a binomial proportion with
/// `successes` of `trials` at `level`; with no successes it is `1 - (1 - level)^(1/trials)`.
/// `None` as for [`clopper_pearson`].
#[must_use]
pub fn clopper_pearson_upper(successes: u64, trials: u64, level: f64) -> Option<f64> {
    if trials == 0 || successes > trials || !proper_level(level) {
        return None;
    }
    if successes == trials {
        return Some(1.0);
    }
    let (x, n) = (successes as f64, trials as f64);
    Some(beta_quantile(level, x + 1.0, n - x))
}

/// Whether `level` is a probability strictly between zero and one.
fn proper_level(level: f64) -> bool {
    level.is_finite() && level > 0.0 && level < 1.0
}

/// The `p` quantile of a Beta(`a`, `b`) law, by bisection on the regularized incomplete beta
/// function, which increases in its argument.
fn beta_quantile(p: f64, a: f64, b: f64) -> f64 {
    let (mut low, mut high) = (0.0_f64, 1.0_f64);
    for _ in 0..200 {
        let mid = 0.5 * (low + high);
        if antecedent_stats::special::regularized_incomplete_beta(mid, a, b) < p {
            low = mid;
        } else {
            high = mid;
        }
    }
    0.5 * (low + high)
}

/// The action a per-structure exact verdict chooses, if it chooses one.
fn exact_choice(verdict: &PreferenceVerdict) -> Option<&str> {
    match verdict {
        PreferenceVerdict::NecessarilyBest(action)
        | PreferenceVerdict::NeverBeaten { action, .. }
        | PreferenceVerdict::WorstCaseChoice(action)
        | PreferenceVerdict::MinimaxRegretChoice(action)
        | PreferenceVerdict::MeasureChoice { action, .. } => Some(action.as_str()),
        PreferenceVerdict::PreferenceDependent { .. }
        | PreferenceVerdict::Tied(_)
        | PreferenceVerdict::ReportOnly
        | PreferenceVerdict::NoAdmissibleAction => None,
    }
}

/// The action a per-structure certified verdict chooses, if it chooses one.
fn certified_choice(verdict: &CertifiedVerdict) -> Option<&str> {
    match verdict {
        CertifiedVerdict::NecessarilyBest(action)
        | CertifiedVerdict::WorstCaseChoice(action)
        | CertifiedVerdict::MinimaxRegretChoice(action)
        | CertifiedVerdict::MeasureChoice { action, .. } => Some(action.as_str()),
        CertifiedVerdict::PreferenceDependent { .. }
        | CertifiedVerdict::Undetermined { .. }
        | CertifiedVerdict::ReportOnly
        | CertifiedVerdict::NoAdmissibleAction => None,
    }
}

/// The verdict of a `BayesOverStructures` evaluation on the sampled measure.
fn on_sampled_measure(evaluation: &StructuralPreferenceEvaluation) -> SampledStructuralVerdict {
    let (best, candidates): (Option<&str>, Vec<String>) = match evaluation {
        StructuralPreferenceEvaluation::Exact(result) => match &result.verdict {
            StructuralPreferenceVerdict::Combined(verdict) => (
                exact_choice(verdict),
                match verdict {
                    PreferenceVerdict::PreferenceDependent { possibly_best } => {
                        possibly_best.clone()
                    }
                    PreferenceVerdict::Tied(actions) => actions.clone(),
                    _ => Vec::new(),
                },
            ),
            StructuralPreferenceVerdict::StructuresDisagree(choices) => (
                None,
                choices
                    .iter()
                    .filter_map(|choice| exact_choice(&choice.verdict).map(str::to_owned))
                    .collect(),
            ),
            StructuralPreferenceVerdict::InsufficientScience(_)
            | StructuralPreferenceVerdict::ReportOnly => (None, Vec::new()),
        },
        StructuralPreferenceEvaluation::Certified(result) => match &result.verdict {
            StructuralCertifiedVerdict::Combined(verdict) => (
                certified_choice(verdict),
                match verdict {
                    CertifiedVerdict::PreferenceDependent { certified_somewhere } => {
                        certified_somewhere.clone()
                    }
                    CertifiedVerdict::Undetermined { candidates, .. } => candidates.clone(),
                    _ => Vec::new(),
                },
            ),
            StructuralCertifiedVerdict::StructuresDisagree(choices) => (
                None,
                choices
                    .iter()
                    .filter_map(|choice| certified_choice(&choice.verdict).map(str::to_owned))
                    .collect(),
            ),
            StructuralCertifiedVerdict::InsufficientScience(_)
            | StructuralCertifiedVerdict::ReportOnly => (None, Vec::new()),
        },
    };
    let mut distinct: Vec<String> = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        if !distinct.contains(&candidate) {
            distinct.push(candidate);
        }
    }
    let candidates = distinct;
    SampledStructuralVerdict::OnSampledMeasure {
        best_on_sampled_measure: best.map(str::to_owned),
        candidates: if best.is_some() { Vec::new() } else { candidates },
    }
}

/// Each structure's id with the action its own verdict chooses, for the evaluated ones.
fn evaluated_choices(evaluation: &StructuralPreferenceEvaluation) -> Vec<(String, Option<String>)> {
    match evaluation {
        StructuralPreferenceEvaluation::Exact(result) => result
            .atoms
            .iter()
            .filter_map(|atom| match &atom.status {
                PreferenceAtomStatus::Evaluated(inner) => {
                    Some((atom.id.clone(), exact_choice(&inner.verdict).map(str::to_owned)))
                }
                PreferenceAtomStatus::Unidentified | PreferenceAtomStatus::Unevaluated(_) => None,
            })
            .collect(),
        StructuralPreferenceEvaluation::Certified(result) => result
            .atoms
            .iter()
            .filter_map(|atom| match &atom.status {
                CertifiedAtomStatus::Evaluated(inner) => {
                    Some((atom.id.clone(), certified_choice(&inner.verdict).map(str::to_owned)))
                }
                CertifiedAtomStatus::Unidentified | CertifiedAtomStatus::Unevaluated(_) => None,
            })
            .collect(),
    }
}

/// The structures of each sampled class: `(class mass, draw, [(atom index, multiplicity)])`.
type SampledClasses = BTreeMap<u64, (f64, Arc<ClassSampling>, Vec<(usize, usize)>)>;

fn sampled_classes(
    mixture: &StructuralResponseMixture,
    members: &[Option<ClassMember>],
) -> SampledClasses {
    let mut classes: SampledClasses = BTreeMap::new();
    for (index, (atom, member)) in mixture.atoms.iter().zip(members).enumerate() {
        let Some(member) = member else {
            continue;
        };
        let Some(sampling) = &member.sampling else {
            continue;
        };
        let entry = classes
            .entry(sampling.class_key)
            .or_insert_with(|| (0.0, Arc::clone(sampling), Vec::new()));
        entry.0 += atom.weight;
        entry.2.push((index, member.multiplicity));
    }
    classes
}

fn scaled(successes: usize, trials: usize, mass: f64) -> MassEstimate {
    let (x, n) = (successes as u64, trials as u64);
    let (lower, upper) = clopper_pearson(x, n, SAMPLED_MASS_LEVEL).unwrap_or((0.0, 1.0));
    let share = if trials == 0 { 0.0 } else { successes as f64 / trials as f64 };
    MassEstimate { estimate: share * mass, lower: lower * mass, upper: upper * mass }
}

/// Whether the mixture atom carries an evaluated, identified response.
fn evaluated_atom(atom: &crate::result::StructuralResponseAtom) -> bool {
    atom.response.is_some()
        && matches!(
            atom.status,
            IdentificationStatus::NonparametricallyIdentified
                | IdentificationStatus::IdentifiedUnderParametricRestrictions
        )
}

fn class_masses(
    mixture: &StructuralResponseMixture,
    classes: &SampledClasses,
) -> Vec<SampledClassMass> {
    classes
        .iter()
        .map(|(&class_key, (mass, sampling, atoms))| {
            let (mut identified, mut unidentified, mut unevaluated) = (0, 0, 0);
            for &(index, multiplicity) in atoms {
                match mixture.atoms.get(index) {
                    Some(atom) if atom.status == IdentificationStatus::NotIdentified => {
                        unidentified += multiplicity;
                    }
                    Some(atom) if evaluated_atom(atom) => identified += multiplicity,
                    _ => unevaluated += multiplicity,
                }
            }
            let drawn = sampling.drawn;
            SampledClassMass {
                class_key,
                class_mass: *mass,
                drawn,
                identified: scaled(identified, drawn, *mass),
                unidentified: scaled(unidentified, drawn, *mass),
                unevaluated: scaled(unevaluated, drawn, *mass),
            }
        })
        .collect()
}

/// Per sampled class, the bound on the declared mass where an action other than `action` is
/// best: draws whose member was evaluated and chose otherwise, of all draws.
fn other_best_bounds(
    action: &str,
    mixture: &StructuralResponseMixture,
    classes: &SampledClasses,
    choices: &[(String, Option<String>)],
) -> Vec<OtherBestBound> {
    classes
        .iter()
        .map(|(&class_key, (mass, sampling, atoms))| {
            let disagreeing: usize = atoms
                .iter()
                .filter(|(index, _)| {
                    mixture.atoms.get(*index).is_some_and(|atom| {
                        let id = format!("{GRAPH_ATOM_PREFIX}{}", atom.graph_key);
                        choices
                            .iter()
                            .any(|(seen, chosen)| *seen == id && chosen.as_deref() != Some(action))
                    })
                })
                .map(|(_, multiplicity)| *multiplicity)
                .sum();
            let upper = clopper_pearson_upper(
                disagreeing as u64,
                sampling.drawn as u64,
                SAMPLED_MASS_LEVEL,
            )
            .unwrap_or(1.0);
            OtherBestBound { class_key, disagreeing, drawn: sampling.drawn, upper: upper * mass }
        })
        .collect()
}

/// The expectation of every part of the linear form for every action under one structure's
/// response: `[action][part]`, part zero the fixed part. `None` when the response does not
/// convert to an aligned joint law the term contract reads.
fn part_expectations(
    terms: &antecedent_design::decision_contract::DecisionContract,
    (actions, parts): (usize, usize),
    response: &CausalResponse,
    (schema, context): (&CausalSchema, &NativeResponseContext),
) -> Option<Vec<Vec<f64>>> {
    let claim = NativeResponseClaim::from_response(response, schema, context).ok()?;
    let input = claim.decision_source(&joint_draws()).ok()?;
    let NativeDecisionSource::JointLaw(law) = input.source else {
        return None;
    };
    let draws = utility_draws(terms, &law).ok()?;
    let mean = |column: &[f64]| -> f64 {
        column.iter().zip(&draws.weights).map(|(value, weight)| value * weight).sum()
    };
    (0..actions)
        .map(|action| {
            (0..parts)
                .map(|part| draws.utilities.get(action * parts + part).map(|c| mean(c.as_slice())))
                .collect::<Option<Vec<f64>>>()
        })
        .collect()
}

/// The vertices of the contract's preference set; `None` when they cannot be enumerated.
fn preference_vertices(
    contract: &PreferenceContract,
    options: PreferenceEvaluationOptions,
) -> Option<Vec<Vec<f64>>> {
    let set = &contract.model.preferences;
    if set.dimension() == 0 {
        return Some(vec![Vec::new()]);
    }
    let rows = set.inequality_rows();
    let constraints: Vec<antecedent_stats::linear_program::LinearConstraint> = rows
        .coefficients
        .into_iter()
        .zip(rows.rhs)
        .map(|(coefficients, rhs)| antecedent_stats::linear_program::LinearConstraint {
            coefficients,
            sense: antecedent_stats::linear_program::ConstraintSense::AtMost,
            rhs,
        })
        .collect();
    let (lower, upper): (Vec<f64>, Vec<f64>) = set.bounds().into_iter().unzip();
    antecedent_stats::linear_program::vertices(
        &constraints,
        &lower,
        &upper,
        options.exact.vertex_cap,
        options.exact.lp,
    )
    .ok()
}

/// Every pairwise margin at every vertex, with the Monte Carlo standard error of the sampled
/// classes; empty when the model has no linear form, a vertex enumeration fails, or an
/// evaluated structure's response cannot be read.
fn sampled_margins(
    contract: &PreferenceContract,
    (mixture, members): (&StructuralResponseMixture, &[Option<ClassMember>]),
    (schema, context): (&CausalSchema, &NativeResponseContext),
    options: PreferenceEvaluationOptions,
) -> Vec<SampledMargin> {
    let Ok(Some(terms)) = contract.term_contract() else {
        return Vec::new();
    };
    let Some(vertices) = preference_vertices(contract, options) else {
        return Vec::new();
    };
    let actions = contract.actions.len();
    let parts = contract.model.preferences.dimension() + 1;
    // Expectations of every evaluated structure; an unread response leaves no margins.
    let mut expectations: Vec<Option<Vec<Vec<f64>>>> = Vec::with_capacity(mixture.atoms.len());
    for atom in &mixture.atoms {
        if !evaluated_atom(atom) {
            expectations.push(None);
            continue;
        }
        let Some(response) = atom.response.as_ref() else {
            return Vec::new();
        };
        let Some(read) = part_expectations(&terms, (actions, parts), response, (schema, context))
        else {
            return Vec::new();
        };
        expectations.push(Some(read));
    }
    let classes = sampled_classes(mixture, members);
    let margin_at = |index: usize, (first, second): (usize, usize), vertex: &[f64]| -> f64 {
        expectations.get(index).and_then(Option::as_ref).map_or(0.0, |e| {
            let difference = |part: usize| e[first][part] - e[second][part];
            difference(0)
                + vertex.iter().enumerate().map(|(j, theta)| theta * difference(j + 1)).sum::<f64>()
        })
    };
    let mut out = Vec::new();
    for vertex in &vertices {
        for first in 0..actions {
            for second in (first + 1)..actions {
                let margin: f64 = mixture
                    .atoms
                    .iter()
                    .enumerate()
                    .map(|(index, atom)| atom.weight * margin_at(index, (first, second), vertex))
                    .sum();
                let mut variance = 0.0;
                let mut defined = true;
                for (mass, sampling, atoms) in classes.values() {
                    let drawn = sampling.drawn;
                    if drawn < 2 {
                        defined = false;
                        continue;
                    }
                    let values: Vec<(f64, usize)> = atoms
                        .iter()
                        .map(|&(index, k)| (margin_at(index, (first, second), vertex), k))
                        .collect();
                    let n = drawn as f64;
                    let mean = values.iter().map(|(v, k)| v * *k as f64).sum::<f64>() / n;
                    let squares: f64 =
                        values.iter().map(|(v, k)| (v - mean).powi(2) * *k as f64).sum();
                    // Draws not among the listed members (none, when every draw compiled)
                    // contribute zero.
                    let listed: usize = values.iter().map(|(_, k)| *k).sum();
                    let rest = drawn.saturating_sub(listed) as f64 * mean.powi(2);
                    variance += mass * mass * (squares + rest) / (n - 1.0) / n;
                }
                out.push(SampledMargin {
                    first: contract.actions[first].id.clone(),
                    second: contract.actions[second].id.clone(),
                    vertex: vertex.clone(),
                    margin,
                    standard_error: defined.then(|| variance.sqrt()),
                });
            }
        }
    }
    out
}

/// Decide `contract` across structures some of whose equivalence classes were sampled.
///
/// `mixture` holds every structure with its weight (for a sampled member, its class mass
/// times its share of the draws) and `members` the class member each structure is, aligned
/// with `mixture.atoms`. The weights are read as probabilities under the declared class
/// measure. Under `BayesOverStructures` the verdict is computed on the sampled measure and
/// reported as a Monte Carlo estimate (never as best for every member), with every pairwise
/// expected-value margin at every vertex of the preference set and its Monte Carlo standard
/// error on the exact route. `Maximin` and `RequireInvariantBestAction` need every member, so
/// each structure is evaluated on its own and no combined verdict is given: the action best
/// on every sampled member, if any, is reported with a one-sided Clopper–Pearson bound on the
/// declared mass where another action is best. `ReportOnly` runs unchanged. The identified,
/// unidentified and unevaluated mass of every sampled class is estimated with an exact
/// Clopper–Pearson interval, scaled by the class mass and never renormalized.
///
/// # Errors
/// Every refusal of [`structure_claims`] and [`decide_across_structures_with_context`].
pub fn decide_on_sampled_structures(
    contract: &PreferenceContract,
    (mixture, members): (&StructuralResponseMixture, &[Option<ClassMember>]),
    (schema, context): (&CausalSchema, &NativeResponseContext),
    measure: Option<&PreferenceMeasure>,
    options: PreferenceEvaluationOptions,
    ctx: &ExecutionContext,
) -> Result<SampledStructuralEvaluation, DecisionFacadeRefusal> {
    let per_member = matches!(
        contract.structural_policy,
        StructuralPolicy::Maximin | StructuralPolicy::RequireInvariantBestAction
    );
    let mut evaluated = contract.clone();
    if per_member {
        evaluated.structural_policy = StructuralPolicy::ReportOnly;
    }
    let claims = structure_claims(mixture, &[], schema, context, evaluated.structural_policy)?;
    let evaluation =
        decide_across_structures_with_context(&evaluated, &claims, measure, options, ctx)?;
    let classes = sampled_classes(mixture, members);
    let (verdict, margins) = match contract.structural_policy {
        StructuralPolicy::BayesOverStructures => (
            on_sampled_measure(&evaluation),
            if evaluation.route() == PreferenceRoute::Exact {
                sampled_margins(contract, (mixture, members), (schema, context), options)
            } else {
                Vec::new()
            },
        ),
        StructuralPolicy::ReportOnly => (SampledStructuralVerdict::ReportOnly, Vec::new()),
        StructuralPolicy::Maximin | StructuralPolicy::RequireInvariantBestAction => {
            let choices = evaluated_choices(&evaluation);
            let first = choices.first().and_then(|(_, chosen)| chosen.clone());
            let common = first.filter(|action| {
                choices.iter().all(|(_, chosen)| chosen.as_deref() == Some(action.as_str()))
            });
            let other_best = common
                .as_deref()
                .map(|action| other_best_bounds(action, mixture, &classes, &choices))
                .unwrap_or_default();
            (
                SampledStructuralVerdict::PerSampledMember {
                    best_on_every_sampled_member: common,
                    other_best,
                },
                Vec::new(),
            )
        }
    };
    Ok(SampledStructuralEvaluation {
        evaluation,
        verdict,
        sampling: classes.values().map(|(_, sampling, _)| sampling.as_ref().clone()).collect(),
        masses: class_masses(mixture, &classes),
        margins,
        level: SAMPLED_MASS_LEVEL,
    })
}

/// What was expected of each action of `contract`, read from an analysis source, as the
/// options of a recorded choice (option id = action id, declaration order).
///
/// Each quantity input is located in the source by coordinate identity exactly as a decision
/// locates it (units, population, regime, horizon, functional), and fixed inputs are taken as
/// declared; the contract's validation checks every input's units against its attribute. A
/// mean source gives each option its attribute means; an aligned joint law gives each option
/// the joint draws of its attributes with the law's weights.
///
/// # Errors
/// An invalid contract, a quantity the source does not carry or does not support, and every
/// other refusal of the decision engine reading the source.
pub fn analysis_options(
    contract: &PreferenceContract,
    source: &NativeDecisionSource,
) -> Result<Vec<ChoiceOption>, DecisionFacadeRefusal> {
    let probe = contract.probe_contract().map_err(|e| e.to_refusal())?;
    match source {
        NativeDecisionSource::Mean(means) => {
            let read =
                evaluate_contract_on_means(&probe.contract, means).map_err(|e| e.to_refusal())?;
            let value = |p: usize| read.actions.get(p).map(|outcome| outcome.expected_utility);
            contract
                .actions
                .iter()
                .zip(&probe.probe_index)
                .map(|(action, row)| {
                    let values = attribute_values(action, row, &value)?;
                    Ok(ChoiceOption { id: action.id.clone(), belief: OptionBelief::Means(values) })
                })
                .collect()
        }
        NativeDecisionSource::JointLaw(law) => {
            let draws = utility_draws(&probe.contract, law).map_err(|e| e.to_refusal())?;
            let n = draws.weights.len();
            contract
                .actions
                .iter()
                .zip(&probe.probe_index)
                .map(|(action, row)| {
                    let mut rows = Vec::with_capacity(n * row.len());
                    for draw in 0..n {
                        let value = |p: usize| {
                            draws.utilities.get(p).and_then(|column| column.get(draw)).copied()
                        };
                        rows.extend(attribute_values(action, row, &value)?);
                    }
                    Ok(ChoiceOption {
                        id: action.id.clone(),
                        belief: OptionBelief::Draws { rows, weights: Some(draws.weights.clone()) },
                    })
                })
                .collect()
        }
    }
}

/// One action's attribute values: a probed quantity through `value`, a fixed input as declared.
fn attribute_values(
    action: &PreferenceAction,
    row: &[Option<usize>],
    value: &impl Fn(usize) -> Option<f64>,
) -> Result<Vec<f64>, DecisionFacadeRefusal> {
    action
        .inputs
        .iter()
        .zip(row)
        .map(|(input, probe)| match (input, probe) {
            (ActionInput::Fixed { value: fixed, .. }, _) => Ok(*fixed),
            (ActionInput::Quantity(_), Some(index)) => value(*index).ok_or_else(|| {
                AdapterError::InvalidClaims("every probed quantity has a value").to_refusal()
            }),
            (ActionInput::Quantity(_), None) => {
                Err(AdapterError::InvalidClaims("every quantity input has a probe").to_refusal())
            }
        })
        .collect()
}

/// Identity of an analysis input as a belief source reads it: the identified program and
/// request, the representation and the content of the source (every coordinate and mean of a
/// mean source; the draw digest of a joint law, as the decision artifacts compute it).
#[must_use]
pub fn analysis_result_identity(input: &NativeDecisionInput) -> String {
    let mut writer = CanonicalWriter::new(BELIEF_DOMAIN);
    writer.text(&input.program_identity);
    match &input.source {
        NativeDecisionSource::Mean(means) => {
            writer
                .word(1)
                .text(&means.provider_id)
                .text(&means.snapshot_id)
                .text(&means.causal_contract_id)
                .text(&means.rng_id)
                .word(means.coordinates.len() as u64);
            for coordinate in &means.coordinates {
                writer.text(&coordinate.canonical_identity());
            }
            writer.reals(&means.means);
        }
        NativeDecisionSource::JointLaw(law) => {
            writer.word(2).text(&source_digest(law));
        }
    }
    writer.finish_hex()
}

/// The belief source of a record whose options were read from `input`: the analysis result
/// ([`analysis_result_identity`]) as it stood at `as_of`.
#[must_use]
pub fn analysis_belief_source(input: &NativeDecisionInput, as_of: &str) -> BeliefSource {
    BeliefSource::AnalysisAsOf {
        result_identity: analysis_result_identity(input),
        as_of: as_of.to_owned(),
    }
}

/// Serialize accepted preferences through the checksummed container.
///
/// # Errors
/// A blank id, an encoding failure or an oversized payload.
pub fn export_accepted(
    accepted: &AcceptedPreferences,
    artifact_id: &str,
) -> Result<Vec<u8>, IoError> {
    Ok(AcceptedPreferencesArtifact::new(accepted.clone()).to_bytes(artifact_id)?)
}

/// Consume accepted preferences under an identity the consumer retained independently of the
/// bytes. The identity is recomputed from the decoded content, so a resealed edit is refused.
/// With `records`, a learned basis is relearned from them with the stored options and must
/// be reproduced; a declared basis replays without records.
///
/// # Errors
/// Corruption, another kind or version, a changed identity, records other than the ones the
/// basis was learned from, and a relearning that does not reproduce the stored basis.
pub fn consume_accepted(
    bytes: &[u8],
    expected_identity: &str,
    records: Option<&ChoiceRecordSet>,
) -> Result<AcceptedPreferences, IoError> {
    let artifact = AcceptedPreferencesArtifact::from_bytes(bytes, expected_identity)?;
    if let Some(records) = records {
        artifact.replay(records)?;
    }
    Ok(artifact.accepted().clone())
}

/// Decide `contract` on an aligned joint law and bind the result into a durable record.
///
/// # Errors
/// An invalid contract and every refusal of [`decide_preferences`].
pub fn record_decision(
    contract: PreferenceContract,
    source: &DistributionArtifact,
    measure: Option<PreferenceMeasure>,
    options: PreferenceEvaluationOptions,
) -> Result<PreferenceDecisionArtifact, DecisionFacadeRefusal> {
    PreferenceDecisionArtifact::evaluate(contract, source, measure, options)
        .map_err(|e| e.to_refusal())
}

/// [`record_decision`] under an execution context, as [`decide_with_context`].
///
/// # Errors
/// As [`record_decision`]; a cancelled run is the answering route's cancellation refusal.
pub fn record_decision_with_context(
    contract: PreferenceContract,
    source: &DistributionArtifact,
    measure: Option<PreferenceMeasure>,
    options: PreferenceEvaluationOptions,
    ctx: &ExecutionContext,
) -> Result<PreferenceDecisionArtifact, DecisionFacadeRefusal> {
    PreferenceDecisionArtifact::evaluate_with_context(contract, source, measure, options, ctx)
        .map_err(|e| e.to_refusal())
}

/// Serialize a decision across a preference set through the checksummed container.
///
/// # Errors
/// A blank id, an encoding failure or an oversized payload.
pub fn export_decision(
    artifact: &PreferenceDecisionArtifact,
    artifact_id: &str,
) -> Result<Vec<u8>, IoError> {
    Ok(artifact.to_bytes(artifact_id)?)
}

/// Consume a decision across a preference set by recomputation: `expected_contract_identity`
/// is retained by the consumer, `source` is the joint law the consumer holds; the stored
/// result must have been computed from exactly that law and must be reproduced by
/// re-evaluating the stored contract on it.
///
/// # Errors
/// Corruption, another kind or version, another contract or source than the retained ones, a
/// resealed edit, and a recomputation that does not reproduce the stored result.
pub fn consume_decision(
    bytes: &[u8],
    expected_contract_identity: &str,
    source: &DistributionArtifact,
) -> Result<PreferenceDecisionArtifact, IoError> {
    let artifact = PreferenceDecisionArtifact::from_bytes(
        bytes,
        expected_contract_identity,
        &source_digest(source),
    )?;
    artifact.replay(source)?;
    Ok(artifact)
}

// ---------------------------------------------------------------------------------------------
// Several estimated outcomes in one source
// ---------------------------------------------------------------------------------------------

fn invalid(why: &'static str) -> DecisionFacadeRefusal {
    AdapterError::InvalidClaims(why).to_refusal()
}

/// The coordinate of one outcome's arm mean in a several-outcome experiment: the component's
/// variable and units, the experiment's population and horizon, and the regime of the arm
/// (`treated` or control). A preference action reads the arm it would put in place.
///
/// # Errors
/// A component the evidence does not hold (`decision_adapters.invalid_claims`).
pub fn experiment_arm_quantity(
    evidence: &JointExperimentEvidence,
    component: &str,
    treated: bool,
) -> Result<ScientificQuantity, DecisionFacadeRefusal> {
    let outcome = evidence
        .components
        .iter()
        .find(|c| c.name == component)
        .ok_or_else(|| invalid("the component is one of the experiment's outcomes"))?;
    Ok(ScientificQuantity {
        variable_id: outcome.variable_id.clone(),
        variable_name: outcome.name.clone(),
        role: QuantityRole::Outcome,
        units: outcome.units.clone(),
        population_id: evidence.population_id.clone(),
        regime_id: if treated {
            evidence.treated_regime_id.clone()
        } else {
            evidence.control_regime_id.clone()
        },
        horizon: evidence.horizon,
        functional_id: "mean".to_owned(),
        conditioning: Vec::new(),
        transform_id: "identity".to_owned(),
    })
}

/// The arm means of every outcome of a several-outcome experiment as one mean source, with
/// the coordinates of [`experiment_arm_quantity`] (treated arm first, components in estimator
/// order). The expectation of a value affine in the attributes needs only these means, so the
/// exact route answers on it; anything that needs a joint law (a part not affine in the
/// attributes, a probability constraint, a quantile) is refused by the decision engine's own
/// mean-source refusal. The estimators' covariance is not a law and is not read.
///
/// # Errors
/// Evidence whose exposure is not declared free of between-unit interference: under bounded or
/// unspecified spillover the arm means are assumption ranges, not means.
pub fn experiment_mean_source(
    evidence: &JointExperimentEvidence,
) -> Result<MeanSource, DecisionFacadeRefusal> {
    if !matches!(evidence.exposure, Exposure::NoBetweenUnitInterference { .. }) {
        return Err(invalid(
            "arm means read as means need exposure declared free of between-unit interference",
        ));
    }
    let mut coordinates = Vec::with_capacity(2 * evidence.components.len());
    let mut means = Vec::with_capacity(2 * evidence.components.len());
    for (treated, arm) in [(true, &evidence.treated_means), (false, &evidence.control_means)] {
        for (component, mean) in evidence.components.iter().zip(arm) {
            coordinates.push(experiment_arm_quantity(evidence, &component.name, treated)?);
            means.push(*mean);
        }
    }
    Ok(MeanSource {
        coordinates,
        means,
        provider_id: format!("decision_evidence:{}", evidence.design.assignment_design),
        snapshot_id: evidence.evidence_id.clone(),
        causal_contract_id: evidence
            .design_contract_id
            .clone()
            .unwrap_or_else(|| "none_declared".to_owned()),
        rng_id: "none:arm_means".to_owned(),
        estimate: MeanEstimate::Frequentist,
    })
}

/// The several aligned output columns a propagation chain retains (its joint output law) as
/// one aligned joint law over the chain's declared output coordinates.
///
/// The chain is where separately fitted results meet under a declared dependence (shared
/// factors, a declared covariance, a ledger overlap or a declared independent product); its
/// rule has already refused any pairing it does not support. The law carries the chain's own
/// receipt (provider, chain identity, declared table construction) and is unverified and
/// uncalibrated.
///
/// # Errors
/// A chain without a retained joint output law (a point or spread reading, a withdrawn source):
/// `joint_law_required`; a chain read as neither a posterior nor a predictive law; an invalid
/// output coordinate.
pub fn chain_joint_law(
    chain: &PropagationChain,
) -> Result<DistributionArtifact, DecisionFacadeRefusal> {
    let unpaired = || {
        DecisionEvalError::JointLawRequired {
            action: None,
            supplied_alignment: "independent_marginals",
        }
        .to_refusal()
    };
    let (columns, weights) = chain.joint_output_law().ok_or_else(unpaired)?;
    let record = chain.record();
    let semantic = if record.reading == ChainReading::Posterior.as_str() {
        DistributionMeaningWire::CausalFunctionalPosterior
    } else if record.reading == ChainReading::Predictive.as_str() {
        // A predictive chain reads the outcome law under the chain's intervention, so its
        // `outcome` coordinates are answered as the chain decision answers them.
        DistributionMeaningWire::InterventionalPredictive
    } else {
        return Err(invalid("a chain law is read as a posterior or a predictive law"));
    };
    let quantities = record
        .joint_outputs
        .iter()
        .map(|output| ScientificQuantity::try_from(output.quantity.clone()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| invalid("every joint output names a valid coordinate"))?;
    let n = columns.first().map_or(0, Vec::len);
    if quantities.len() != columns.len() || columns.iter().any(|c| c.len() != n) {
        return Err(invalid("one aligned column per joint output coordinate"));
    }
    let mut rows = Vec::with_capacity(n * columns.len());
    for draw in 0..n {
        rows.extend(columns.iter().map(|column| f64::from_bits(column[draw])));
    }
    let rng_id = if record.construction == TableConstruction::IndependentProduct.as_str() {
        "declared_independent_product"
    } else {
        "declared_aligned_draws"
    };
    let identity = DistributionIdentity::new(
        semantic,
        &quantities,
        DrawAlignment::Joint,
        DistributionProvenance {
            source_id: chain.identity().to_owned(),
            provider_id: "antecedent.propagation_chain".to_owned(),
            rng_id: rng_id.to_owned(),
            snapshot_id: chain.identity().to_owned(),
            causal_contract_id: "none_declared".to_owned(),
        },
    )
    .map_err(|_| invalid("the chain's output coordinates form a joint identity"))?;
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".to_owned(), "quantity".to_owned()],
            shape: [n, columns.len()],
            weights: weights.map(|bits| bits.iter().map(|b| f64::from_bits(*b)).collect()),
            supported: None,
            calibration: DistributionCalibration::NotMeasured,
            trust: DistributionTrust::Unverified,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        rows,
    )
    .map_err(|_| invalid("the chain's joint output law forms a distribution artifact"))
}

/// One aligned joint law over the coordinates of several laws (a single law is returned
/// unchanged), when the joint-law rule says
/// their rows are realizations of one joint law (equal alignment identities,
/// `antecedent_contract::check_rows_pairable_across`); the columns are then placed side by
/// side. Rows of separately produced laws are never paired: they refuse with the rule's own
/// `joint_law_required` refusal (`alignment_mismatch`, or `index_pairing_refused` for
/// independent marginals).
///
/// # Errors
/// No laws; the rule's refusal; laws that share an alignment identity but differ in meaning,
/// weights or calibration.
pub fn joint_law_of(
    laws: &[&DistributionArtifact],
) -> Result<DistributionArtifact, DecisionFacadeRefusal> {
    match laws {
        [] => return Err(invalid("at least one law")),
        [only] => return Ok((*only).clone()),
        _ => {}
    }
    let first_refusal = |refusal: super::joint_law_propagation::JointLawPropagationRefusal| {
        refusal.refusals.into_iter().next().unwrap_or_else(|| invalid("the rule refused"))
    };
    let alignments = laws
        .iter()
        .map(|law| super::joint_law_propagation::input_alignment(law))
        .collect::<Result<Vec<_>, _>>()
        .map_err(first_refusal)?;
    let names: Vec<String> = (0..laws.len()).map(|i| format!("law:{i}")).collect();
    let inputs: Vec<(&str, &InputAlignment)> =
        names.iter().map(String::as_str).zip(alignments.iter()).collect();
    check_rows_pairable_across(&inputs)?;
    let first = laws[0];
    let head = first.metadata();
    if laws.iter().any(|law| {
        let m = law.metadata();
        m.identity.semantic != head.identity.semantic
            || m.weights != head.weights
            || m.calibration != head.calibration
            || m.trust != head.trust
    }) {
        return Err(invalid("paired laws share their meaning, weights, calibration and trust"));
    }
    let quantities = laws
        .iter()
        .flat_map(|law| law.quantities().iter().cloned())
        .map(ScientificQuantity::try_from)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| invalid("every paired column names a valid coordinate"))?;
    let n = first.n_draws();
    let width: usize = laws.iter().map(|law| law.shape()[1]).sum();
    let mut rows = Vec::with_capacity(n * width);
    for draw in 0..n {
        for law in laws {
            let [_, w] = law.shape();
            rows.extend_from_slice(&law.draws()[draw * w..(draw + 1) * w]);
        }
    }
    let supported = if laws.iter().all(|law| law.metadata().supported.is_none()) {
        None
    } else {
        Some(
            laws.iter()
                .flat_map(|law| {
                    let [_, w] = law.shape();
                    law.metadata().supported.clone().unwrap_or_else(|| vec![true; w])
                })
                .collect(),
        )
    };
    let identity = DistributionIdentity::new(
        head.identity.semantic,
        &quantities,
        DrawAlignment::Joint,
        DistributionProvenance {
            source_id: head.identity.source_id.clone(),
            provider_id: head.identity.provider_id.clone(),
            rng_id: head.identity.rng_id.clone(),
            snapshot_id: head.identity.snapshot_id.clone(),
            causal_contract_id: head.identity.causal_contract_id.clone(),
        },
    )
    .map_err(|_| invalid("the paired coordinates form a joint identity"))?;
    DistributionArtifact::new(
        DistributionMetadata {
            version: head.version,
            identity,
            axes: head.axes.clone(),
            shape: [n, width],
            weights: head.weights.clone(),
            supported,
            calibration: head.calibration,
            trust: head.trust,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        rows,
    )
    .map_err(|_| invalid("the paired columns form a distribution artifact"))
}
