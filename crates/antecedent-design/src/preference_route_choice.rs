//! One entry point over the exact and the certified verdict across a preference set.
//!
//! Route rule. The exact route ([`crate::preference_decision`]) answers a contract whose value
//! model has a linear form, whose preference set holds no value relation and whose criterion is
//! posterior expected utility, posterior expected loss or expected regret; every other contract
//! goes to the certified route
//! ([`crate::preference_certified`]). When the exact evaluator itself reports that the certified
//! route is needed (a part of the linear form that cannot be lowered, or the worst case of
//! expected regret), the contract falls through to the certified route when the source has
//! joint draws. A mean-only source can use the exact route only: a contract that needs the
//! certified route on a mean source is refused, naming the joint draws it needs.
//!
//! [`PreferenceEvaluation`] carries the result of whichever route answered, with accessors for
//! the facts both routes share. Deciding with accepted (possibly learned) preferences lives with
//! the accepted preferences; this module reads no code that learns preferences from recorded
//! choices, so the explicit route chooser stands on its own.

// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::ExternalRefusal;
use antecedent_core::ExecutionContext;
use antecedent_io_core::distribution_artifact::DistributionArtifact;

use crate::decision_contract::DecisionCriterion;
use crate::decision_eval::MeanSource;
use crate::decision_structural::StructuralAtom;
use crate::preference_certified::{
    CertifiedOptions, CertifiedPreferenceResult, CertifiedVerdict, PreferenceCertifiedError,
    StructuralCertifiedResult, certified, certified_structural,
};
use crate::preference_contract::{PreferenceContract, PreferenceContractError};
use crate::preference_decision::{
    CertifiedRouteReason, NO_ADMISSIBLE_SENTENCE, PreferenceDecisionError,
    PreferenceDecisionOptions, PreferenceDecisionResult, PreferenceMeasure, PreferenceVerdict,
    REPORT_ONLY_SENTENCE, StructuralPreferenceResult, across_structures, join_ids,
    measure_choice_sentence, minimax_regret_sentence, on_draws, on_means, plain_number,
    preferred_everywhere, worst_case_sentence,
};
use crate::preference_refusal::{EVALUATE, refusal};
use crate::signal::at_stage;

/// Route label: the value is affine in the parameters and the criterion is an expectation.
pub const ROUTE_AFFINE_EXPECTATION: &str = "preference_route.affine_expectation";
/// Route label: the criterion is not affine in the parameters.
pub const ROUTE_CRITERION_NOT_AFFINE: &str = "preference_route.criterion_not_affine";
/// Route label: the value is not affine in some parameters.
pub const ROUTE_SHAPE_PARAMETERS: &str = "preference_route.shape_parameters";
/// Route label: a part of the linear form holds an exponential or logarithm of a quantity.
pub const ROUTE_PART_NOT_LOWERABLE: &str = "preference_route.part_not_lowerable";
/// Route label: the worst case of expected regret is not a linear program over the set.
pub const ROUTE_WORST_CASE_OF_EXPECTED_REGRET: &str =
    "preference_route.worst_case_of_expected_regret";
/// Route label: the preference set holds value relations (recorded choices under a value with
/// shape parameters), which are not linear in the parameters.
pub const ROUTE_VALUE_RELATIONS: &str = "preference_route.value_relations";

/// Which verdict answers a contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PreferenceRoute {
    /// Linear programs over the preference set.
    Exact,
    /// Certified interval subdivision of the preference set.
    Certified,
}

/// The route a contract takes, with the stable label naming why.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RouteChoice {
    /// The route.
    pub route: PreferenceRoute,
    /// One of the `ROUTE_*` labels of this module.
    pub reason: &'static str,
}

/// Options of both routes.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PreferenceEvaluationOptions {
    /// Options of the exact route.
    pub exact: PreferenceDecisionOptions,
    /// Options of the certified route.
    pub certified: CertifiedOptions,
}

/// The verdict across a preference set for one source, from whichever route answered.
#[derive(Clone, Debug, PartialEq)]
pub enum PreferenceEvaluation {
    /// Answered by linear programs.
    Exact(Box<PreferenceDecisionResult>),
    /// Answered by certified subdivision.
    Certified(Box<CertifiedPreferenceResult>),
}

/// The verdict across structures and a preference set, from whichever route answered.
#[derive(Clone, Debug, PartialEq)]
pub enum StructuralPreferenceEvaluation {
    /// Answered by linear programs.
    Exact(Box<StructuralPreferenceResult>),
    /// Answered by certified subdivision.
    Certified(Box<StructuralCertifiedResult>),
}

impl PreferenceEvaluation {
    /// The route that answered.
    #[must_use]
    pub fn route(&self) -> PreferenceRoute {
        match self {
            Self::Exact(_) => PreferenceRoute::Exact,
            Self::Certified(_) => PreferenceRoute::Certified,
        }
    }

    /// Identity of the evaluated preference contract.
    #[must_use]
    pub fn contract_identity(&self) -> &str {
        match self {
            Self::Exact(result) => &result.contract_identity,
            Self::Certified(result) => &result.contract_identity,
        }
    }

    /// The action the verdict chooses, when it chooses one: the unique winner, the one action
    /// never beaten, the worst-case choice, the minimax-regret choice or the measure choice.
    #[must_use]
    pub fn chosen_action(&self) -> Option<&str> {
        match self {
            Self::Exact(result) => match &result.verdict {
                PreferenceVerdict::NecessarilyBest(a)
                | PreferenceVerdict::NeverBeaten { action: a, .. }
                | PreferenceVerdict::WorstCaseChoice(a)
                | PreferenceVerdict::MinimaxRegretChoice(a)
                | PreferenceVerdict::MeasureChoice { action: a, .. } => Some(a.as_str()),
                PreferenceVerdict::PreferenceDependent { .. }
                | PreferenceVerdict::Tied(_)
                | PreferenceVerdict::ReportOnly
                | PreferenceVerdict::NoAdmissibleAction => None,
            },
            Self::Certified(result) => match &result.verdict {
                CertifiedVerdict::NecessarilyBest(a)
                | CertifiedVerdict::WorstCaseChoice(a)
                | CertifiedVerdict::MinimaxRegretChoice(a)
                | CertifiedVerdict::MeasureChoice { action: a, .. } => Some(a.as_str()),
                CertifiedVerdict::PreferenceDependent { .. }
                | CertifiedVerdict::Undetermined { .. }
                | CertifiedVerdict::ReportOnly
                | CertifiedVerdict::NoAdmissibleAction => None,
            },
        }
    }

    /// Whether the verdict is a unique winner: one action strictly ahead of every rival at every
    /// admissible parameter value.
    #[must_use]
    pub fn is_necessarily_best(&self) -> bool {
        match self {
            Self::Exact(result) => matches!(result.verdict, PreferenceVerdict::NecessarilyBest(_)),
            Self::Certified(result) => {
                matches!(result.verdict, CertifiedVerdict::NecessarilyBest(_))
            }
        }
    }

    /// Whether the verdict is that one action is never beaten at any admissible parameter
    /// value: a unique winner, or (on the exact route) one action that ties some rivals
    /// somewhere and leads them elsewhere. A certified box is decided only when one action is
    /// strictly ahead, so on the certified route this is the unique winner.
    #[must_use]
    pub fn is_never_beaten(&self) -> bool {
        match self {
            Self::Exact(result) => matches!(
                result.verdict,
                PreferenceVerdict::NecessarilyBest(_) | PreferenceVerdict::NeverBeaten { .. }
            ),
            Self::Certified(result) => {
                matches!(result.verdict, CertifiedVerdict::NecessarilyBest(_))
            }
        }
    }

    /// The verdict in one or two plain sentences, built only from the result's own facts. The
    /// exact route states [`PreferenceDecisionResult::statement`]; the certified route states its
    /// verdict and, when part of the set is left undecided, the undecided volume fraction (a
    /// share of the set's volume, not a probability), with numbers written by the same rule.
    #[must_use]
    pub fn statement(&self) -> String {
        match self {
            Self::Exact(result) => result.statement(),
            Self::Certified(result) => certified_statement(result),
        }
    }

    /// Volume fraction of the set left undecided; `None` on the exact route, which decides
    /// every point.
    #[must_use]
    pub fn undecided_fraction(&self) -> Option<f64> {
        match self {
            Self::Exact(_) => None,
            Self::Certified(result) => Some(result.undecided_fraction),
        }
    }

    /// Assumptions the verdict stands on, sorted and distinct.
    #[must_use]
    pub fn assumptions(&self) -> &[String] {
        match self {
            Self::Exact(result) => &result.assumptions,
            Self::Certified(result) => &result.assumptions,
        }
    }

    /// The exact result, when the exact route answered.
    #[must_use]
    pub fn exact(&self) -> Option<&PreferenceDecisionResult> {
        match self {
            Self::Exact(result) => Some(result.as_ref()),
            Self::Certified(_) => None,
        }
    }

    /// The certified result, when the certified route answered.
    #[must_use]
    pub fn certified(&self) -> Option<&CertifiedPreferenceResult> {
        match self {
            Self::Exact(_) => None,
            Self::Certified(result) => Some(result.as_ref()),
        }
    }
}

/// The plain statement of a certified result.
fn certified_statement(result: &CertifiedPreferenceResult) -> String {
    let has_parameters = !result.parameter_ids.is_empty();
    let fraction = plain_number(result.undecided_fraction);
    let undecided = if result.undecided_fraction > 0.0 {
        format!(" A volume fraction of {fraction} of the admissible set is left undecided.")
    } else {
        String::new()
    };
    match &result.verdict {
        CertifiedVerdict::NecessarilyBest(action) => {
            format!("{}{undecided}", preferred_everywhere(action, has_parameters, ""))
        }
        CertifiedVerdict::PreferenceDependent { certified_somewhere } => format!(
            "The preferred action depends on the declared trade-offs: {} {} certified preferred \
             on part of the admissible set.{undecided}",
            join_ids(certified_somewhere),
            if certified_somewhere.len() == 1 { "is" } else { "are each" }
        ),
        CertifiedVerdict::WorstCaseChoice(action) => {
            format!("{}{undecided}", worst_case_sentence(action, ""))
        }
        CertifiedVerdict::MinimaxRegretChoice(action) => {
            format!("{}{undecided}", minimax_regret_sentence(action, ""))
        }
        CertifiedVerdict::MeasureChoice { action, share } => {
            measure_choice_sentence(action, *share)
        }
        CertifiedVerdict::Undetermined { candidates, undecided_fraction } => format!(
            "The certified subdivision does not settle the choice among {}; a volume fraction of \
             {} of the admissible set is left undecided.",
            join_ids(candidates),
            plain_number(*undecided_fraction)
        ),
        CertifiedVerdict::ReportOnly => format!("{REPORT_ONLY_SENTENCE}{undecided}"),
        CertifiedVerdict::NoAdmissibleAction => NO_ADMISSIBLE_SENTENCE.to_owned(),
    }
}

impl StructuralPreferenceEvaluation {
    /// The route that answered.
    #[must_use]
    pub fn route(&self) -> PreferenceRoute {
        match self {
            Self::Exact(_) => PreferenceRoute::Exact,
            Self::Certified(_) => PreferenceRoute::Certified,
        }
    }
}

/// Why an evaluation across a preference set is refused.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum PreferenceEvaluationError {
    /// The preference contract is refused.
    #[error("preference contract: {0}")]
    Contract(PreferenceContractError),
    /// The exact route refused.
    #[error("exact route: {0}")]
    Exact(PreferenceDecisionError),
    /// The certified route refused.
    #[error("certified route: {0}")]
    Certified(PreferenceCertifiedError),
    /// The contract needs the certified route, which needs joint draws, but the source has
    /// means only.
    #[error("the certified route needs joint draws ({reason}); a mean source cannot answer")]
    JointDrawsRequired {
        /// The route label naming why the certified route is needed.
        reason: &'static str,
    },
}

impl PreferenceEvaluationError {
    /// Stable detail id, prefixed `preference_evaluation.`; a wrapped refusal keeps its own
    /// detail.
    #[must_use]
    pub fn detail(&self) -> &'static str {
        use antecedent_core::reason_details;
        match self {
            Self::Contract(inner) => inner.detail(),
            Self::Exact(inner) => inner.detail(),
            Self::Certified(inner) => inner.detail(),
            Self::JointDrawsRequired { .. } => {
                reason_details::preference_evaluation::JOINT_DRAWS_REQUIRED
            }
        }
    }
}

use PreferenceEvaluationError as E;

/// Which route answers `contract`, and why.
///
/// # Errors
/// An invalid contract.
pub fn preference_route(
    contract: &PreferenceContract,
) -> Result<RouteChoice, PreferenceEvaluationError> {
    contract.validate().map_err(E::Contract)?;
    let certified = |reason| RouteChoice { route: PreferenceRoute::Certified, reason };
    if contract.model.linear_form().is_none() {
        return Ok(certified(ROUTE_SHAPE_PARAMETERS));
    }
    if !contract.model.preferences.value_relations.is_empty() {
        return Ok(certified(ROUTE_VALUE_RELATIONS));
    }
    Ok(match contract.criterion {
        DecisionCriterion::PosteriorExpectedUtility
        | DecisionCriterion::PosteriorExpectedLoss
        | DecisionCriterion::ExpectedRegret => {
            RouteChoice { route: PreferenceRoute::Exact, reason: ROUTE_AFFINE_EXPECTATION }
        }
        DecisionCriterion::ThresholdProbability { .. }
        | DecisionCriterion::Quantile { .. }
        | DecisionCriterion::MinimaxOverIdentifiedSet
        | DecisionCriterion::MaximinOverStructures
        | DecisionCriterion::Regret => certified(ROUTE_CRITERION_NOT_AFFINE),
    })
}

/// The route label of an exact-route refusal that sends the contract to the certified route.
fn fall_through(error: &PreferenceDecisionError) -> Option<&'static str> {
    match error {
        PreferenceDecisionError::CertifiedRouteRequired { reason } => Some(match reason {
            CertifiedRouteReason::Criterion(_) => ROUTE_CRITERION_NOT_AFFINE,
            CertifiedRouteReason::ShapeParameters { .. } => ROUTE_SHAPE_PARAMETERS,
            CertifiedRouteReason::PartNotLowerable => ROUTE_PART_NOT_LOWERABLE,
            CertifiedRouteReason::WorstCaseOfExpectedRegret => ROUTE_WORST_CASE_OF_EXPECTED_REGRET,
            CertifiedRouteReason::ValueRelations { .. } => ROUTE_VALUE_RELATIONS,
        }),
        _ => None,
    }
}

/// Decide `contract` across its preference set on aligned joint draws, by the route rule.
///
/// # Errors
/// An invalid contract; a refusal of the route that answered.
pub fn decide_preferences(
    contract: &PreferenceContract,
    source: &DistributionArtifact,
    measure: Option<&PreferenceMeasure>,
    options: PreferenceEvaluationOptions,
) -> Result<PreferenceEvaluation, PreferenceEvaluationError> {
    decide_on_draws(contract, source, measure, options, None)
}

/// [`decide_preferences`] under an execution context: the route that answers reads the
/// context's cancellation token at its work boundaries (linear programs, vertex bases and draws
/// on the exact route, boxes and measure points on the certified route).
///
/// # Errors
/// As [`decide_preferences`]; a cancelled run is the answering route's `Cancelled` refusal.
pub fn decide_preferences_with_context(
    contract: &PreferenceContract,
    source: &DistributionArtifact,
    measure: Option<&PreferenceMeasure>,
    options: PreferenceEvaluationOptions,
    ctx: &ExecutionContext,
) -> Result<PreferenceEvaluation, PreferenceEvaluationError> {
    decide_on_draws(contract, source, measure, options, Some(ctx))
}

pub(crate) fn decide_on_draws(
    contract: &PreferenceContract,
    source: &DistributionArtifact,
    measure: Option<&PreferenceMeasure>,
    options: PreferenceEvaluationOptions,
    ctx: Option<&ExecutionContext>,
) -> Result<PreferenceEvaluation, PreferenceEvaluationError> {
    if preference_route(contract)?.route == PreferenceRoute::Exact {
        match on_draws(contract, source, measure, options.exact, ctx) {
            Ok(result) => return Ok(PreferenceEvaluation::Exact(Box::new(result))),
            Err(error) if fall_through(&error).is_some() => {}
            Err(error) => return Err(E::Exact(error)),
        }
    }
    certified(contract, source, measure, options.certified, ctx)
        .map(|result| PreferenceEvaluation::Certified(Box::new(result)))
        .map_err(E::Certified)
}

/// Decide `contract` across its preference set on a mean-only source. Only the exact route
/// can answer, and expected regret needs aligned draws, so a mean source refuses it as the
/// ordinary contract does.
///
/// # Errors
/// An invalid contract; a contract that needs the certified route
/// ([`PreferenceEvaluationError::JointDrawsRequired`]); an exact-route refusal.
pub fn decide_preferences_on_means(
    contract: &PreferenceContract,
    source: &MeanSource,
    measure: Option<&PreferenceMeasure>,
    options: PreferenceEvaluationOptions,
) -> Result<PreferenceEvaluation, PreferenceEvaluationError> {
    decide_on_means(contract, source, measure, options, None)
}

/// [`decide_preferences_on_means`] under an execution context: the exact route reads the
/// context's cancellation token before every linear program.
///
/// # Errors
/// As [`decide_preferences_on_means`]; a cancelled run is the exact route's `Cancelled`
/// refusal.
pub fn decide_preferences_on_means_with_context(
    contract: &PreferenceContract,
    source: &MeanSource,
    measure: Option<&PreferenceMeasure>,
    options: PreferenceEvaluationOptions,
    ctx: &ExecutionContext,
) -> Result<PreferenceEvaluation, PreferenceEvaluationError> {
    decide_on_means(contract, source, measure, options, Some(ctx))
}

fn decide_on_means(
    contract: &PreferenceContract,
    source: &MeanSource,
    measure: Option<&PreferenceMeasure>,
    options: PreferenceEvaluationOptions,
    ctx: Option<&ExecutionContext>,
) -> Result<PreferenceEvaluation, PreferenceEvaluationError> {
    let choice = preference_route(contract)?;
    if choice.route == PreferenceRoute::Certified {
        return Err(E::JointDrawsRequired { reason: choice.reason });
    }
    on_means(contract, source, measure, options.exact, ctx)
        .map(|result| PreferenceEvaluation::Exact(Box::new(result)))
        .map_err(|error| match fall_through(&error) {
            Some(reason) => E::JointDrawsRequired { reason },
            None => E::Exact(error),
        })
}

/// Decide `contract` across structures and its preference set, by the route rule.
///
/// # Errors
/// An invalid contract; a refusal of the route that answered.
pub fn decide_preferences_structural(
    contract: &PreferenceContract,
    atoms: &[StructuralAtom],
    measure: Option<&PreferenceMeasure>,
    options: PreferenceEvaluationOptions,
) -> Result<StructuralPreferenceEvaluation, PreferenceEvaluationError> {
    decide_structural(contract, atoms, measure, options, None)
}

/// [`decide_preferences_structural`] under an execution context: the route that answers reads
/// the context's cancellation token before every structure and at its work boundaries.
///
/// # Errors
/// As [`decide_preferences_structural`]; a cancelled run is the answering route's `Cancelled`
/// refusal.
pub fn decide_preferences_structural_with_context(
    contract: &PreferenceContract,
    atoms: &[StructuralAtom],
    measure: Option<&PreferenceMeasure>,
    options: PreferenceEvaluationOptions,
    ctx: &ExecutionContext,
) -> Result<StructuralPreferenceEvaluation, PreferenceEvaluationError> {
    decide_structural(contract, atoms, measure, options, Some(ctx))
}

fn decide_structural(
    contract: &PreferenceContract,
    atoms: &[StructuralAtom],
    measure: Option<&PreferenceMeasure>,
    options: PreferenceEvaluationOptions,
    ctx: Option<&ExecutionContext>,
) -> Result<StructuralPreferenceEvaluation, PreferenceEvaluationError> {
    if preference_route(contract)?.route == PreferenceRoute::Exact {
        match across_structures(contract, atoms, measure, options.exact, ctx) {
            Ok(result) => return Ok(StructuralPreferenceEvaluation::Exact(Box::new(result))),
            Err(error) if fall_through(&error).is_some() => {}
            Err(error) => return Err(E::Exact(error)),
        }
    }
    certified_structural(contract, atoms, measure, options.certified, ctx)
        .map(|result| StructuralPreferenceEvaluation::Certified(Box::new(result)))
        .map_err(E::Certified)
}

impl PreferenceEvaluationError {
    /// Structured refusal for a failure of the route-choosing evaluation across a preference
    /// set.
    ///
    /// A contract, exact-route or certified-route refusal keeps its own refusal.
    #[must_use]
    pub fn to_refusal(&self) -> ExternalRefusal {
        match self {
            Self::Contract(error) => at_stage(error.to_refusal(), EVALUATE),
            Self::Exact(error) => at_stage(error.to_refusal(), EVALUATE),
            Self::Certified(error) => at_stage(error.to_refusal(), EVALUATE),
            Self::JointDrawsRequired { reason } => refusal(
                antecedent_core::reason_code!("joint_law_required"),
                EVALUATE,
                self.detail(),
            )
            .offending(*reason)
            .expected("joint_draws")
            .supplied("mean")
            .remedy("supply aligned joint draws so the certified route can answer")
            .build(),
        }
    }
}
