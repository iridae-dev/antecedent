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
//!   structures and the preference set together;
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
    ChoiceModelError, ChoiceModelEstimate, ChoiceModelOptions, ChoiceModelResult, CovarianceKind,
    EstimateOutcome, ParameterEstimate, calibration_basis, estimate_choice_model,
    estimate_choice_model_with_context,
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
    ROUTE_SHAPE_PARAMETERS, ROUTE_WORST_CASE_OF_EXPECTED_REGRET, RouteChoice,
    StructuralPreferenceEvaluation, decide_preferences, decide_preferences_on_means,
    decide_preferences_on_means_with_context, decide_preferences_structural,
    decide_preferences_structural_with_context, decide_preferences_with_context,
    decide_with_accepted, decide_with_accepted_with_context, preference_route,
};
pub use antecedent_design::preference_model::{
    Attribute, Curve, LinearForm, LinearRelation, MAX_PREFERENCE_DEPTH, PreferenceExpr,
    PreferenceModelError, PreferenceParameter, PreferenceRows, PreferenceSet, RelationOrigin,
    RelationSense, ValueModel, shape,
};
pub use antecedent_design::preference_set_learning::{
    ASSUMPTION_CHOICES_OPTIMAL, BindingChoice, ChoiceConflict, ConflictKind, DeciderOverlap,
    DeciderSet, LearnedPreferenceSet, ParameterRange, SetLearningError, SetLearningOptions,
    SetLearningResult, SetOutcome, learn_compatible_set, learn_compatible_set_with_context,
};

pub use super::execute::{
    BoundPreferenceContract, CompiledPreferenceDecision, CompiledPreferenceEvaluation,
    CompiledStructure, DemandedQuantity, DesignBootstrap, OutcomeDependence,
    PreferenceDecisionExecution, PreferenceDecisionReceipt, PreferenceDemand, PreferenceRouteError,
    StructureIdentification, compile_preference_decision_across_studies,
};
pub use super::execute::{
    CrossStudyCovariance, PreferenceJointLaw, SharedEvidence,
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
/// Posterior weights are genuine structure probabilities ([`adapt_weighted_graph_atoms`]);
/// caller-supplied class priors are declared weights ([`adapt_finite_scenarios`]);
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
        StructuralWeightBasis::PosteriorProbability => adapt_weighted_graph_atoms(claims, policy),
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
