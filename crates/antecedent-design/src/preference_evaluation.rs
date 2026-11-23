//! One entry point over the exact and the certified verdict across a preference set.
//!
//! Route rule. The exact route ([`crate::preference_decision`]) answers a contract whose value
//! model has a linear form and whose criterion is posterior expected utility, posterior expected
//! loss or expected regret; every other contract goes to the certified route
//! ([`crate::preference_certified`]). When the exact evaluator itself reports that the certified
//! route is needed (a part of the linear form that cannot be lowered, or the worst case of
//! expected regret), the contract falls through to the certified route when the source has
//! joint draws. A mean-only source can use the exact route only: a contract that needs the
//! certified route on a mean source is refused, naming the joint draws it needs.
//!
//! [`PreferenceEvaluation`] carries the result of whichever route answered, with accessors for
//! the facts both routes share. [`decide_with_accepted`] builds the contract over accepted
//! preferences and passes the accepted measure when there is one.

// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_io_core::distribution_artifact::DistributionArtifact;

use crate::accepted_preferences::{AcceptedPreferences, PreferenceContractParts};
use crate::decision_contract::DecisionCriterion;
use crate::decision_eval::MeanSource;
use crate::decision_structural::StructuralAtom;
use crate::preference_certified::{
    CertifiedOptions, CertifiedPreferenceResult, CertifiedVerdict, PreferenceCertifiedError,
    StructuralCertifiedResult, evaluate_preference_certified,
    evaluate_preference_certified_structural,
};
use crate::preference_contract::{PreferenceContract, PreferenceContractError};
use crate::preference_decision::{
    CertifiedRouteReason, PreferenceDecisionError, PreferenceDecisionOptions,
    PreferenceDecisionResult, PreferenceMeasure, PreferenceVerdict, StructuralPreferenceResult,
    evaluate_preference_contract, evaluate_preference_contract_on_means,
    evaluate_preference_structural,
};

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

    /// The action the verdict chooses, when it chooses one: necessarily best, the worst-case
    /// choice, the minimax-regret choice or the measure choice.
    #[must_use]
    pub fn chosen_action(&self) -> Option<&str> {
        match self {
            Self::Exact(result) => match &result.verdict {
                PreferenceVerdict::NecessarilyBest(a)
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

    /// Whether the verdict is that one action is best at every admissible parameter value.
    #[must_use]
    pub fn is_necessarily_best(&self) -> bool {
        match self {
            Self::Exact(result) => matches!(result.verdict, PreferenceVerdict::NecessarilyBest(_)),
            Self::Certified(result) => {
                matches!(result.verdict, CertifiedVerdict::NecessarilyBest(_))
            }
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
    if preference_route(contract)?.route == PreferenceRoute::Exact {
        match evaluate_preference_contract(contract, source, measure, options.exact) {
            Ok(result) => return Ok(PreferenceEvaluation::Exact(Box::new(result))),
            Err(error) if fall_through(&error).is_some() => {}
            Err(error) => return Err(E::Exact(error)),
        }
    }
    evaluate_preference_certified(contract, source, measure, options.certified)
        .map(|result| PreferenceEvaluation::Certified(Box::new(result)))
        .map_err(E::Certified)
}

/// Decide `contract` across its preference set on a mean-only source. Only the exact route
/// can answer.
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
    let choice = preference_route(contract)?;
    if choice.route == PreferenceRoute::Certified {
        return Err(E::JointDrawsRequired { reason: choice.reason });
    }
    evaluate_preference_contract_on_means(contract, source, measure, options.exact)
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
    if preference_route(contract)?.route == PreferenceRoute::Exact {
        match evaluate_preference_structural(contract, atoms, measure, options.exact) {
            Ok(result) => return Ok(StructuralPreferenceEvaluation::Exact(Box::new(result))),
            Err(error) if fall_through(&error).is_some() => {}
            Err(error) => return Err(E::Exact(error)),
        }
    }
    evaluate_preference_certified_structural(contract, atoms, measure, options.certified)
        .map(|result| StructuralPreferenceEvaluation::Certified(Box::new(result)))
        .map_err(E::Certified)
}

/// Decide with accepted preferences: the contract is built over the accepted model, and the
/// accepted measure (an estimate's) is passed when there is one.
///
/// # Errors
/// As [`decide_preferences`].
pub fn decide_with_accepted(
    accepted: &AcceptedPreferences,
    parts: PreferenceContractParts,
    source: &DistributionArtifact,
    options: PreferenceEvaluationOptions,
) -> Result<PreferenceEvaluation, PreferenceEvaluationError> {
    let contract = accepted.contract(parts);
    decide_preferences(&contract, source, accepted.measure(), options)
}
