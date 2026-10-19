//! Decision evaluation against the source demand a contract already declares.
//!
//! [`DecisionContract::source_requirement`] is that demand. This module evaluates
//! it on a mean source. The expectation of an affine utility is the utility of
//! the means, so that evaluation requests neither predictions nor draws. A
//! nonlinear utility, a threshold, a quantile, a hard constraint, or regret,
//! given only means, is refused, and the refusal names the missing predictions
//! or draws.
//!
//! [`BatchRecord`] and [`RetentionRecord`] are records of what was requested.
//! They are not a thread pool, not a scratch pool, and not a fit. Nothing here
//! forms a posterior: a mean is not a belief, and an identified set is not
//! given probabilities.
//!
//! Scoring these declared actions does not build a
//! [`crate::decision_contract::ProgramSearchContract`] and does not search.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::ScientificQuantity;
use antecedent_core::execution_facts::OutputRetention;
use thiserror::Error;

use crate::decision_contract::{
    DecisionContract, DecisionContractError, SourceRepresentation, SourceRequirement,
};
use crate::decision_eval::{
    DecisionEvalError, DecisionResult, MeanSource, evaluate_contract_on_means,
};

const ONLY_PREDICTIONS: &[MissingCarrier] = &[MissingCarrier::Predictions];
const ONLY_DRAWS: &[MissingCarrier] = &[MissingCarrier::Draws];
const PREDICTIONS_OR_DRAWS: &[MissingCarrier] =
    &[MissingCarrier::Predictions, MissingCarrier::Draws];

/// What an evaluation asked a source to supply.
///
/// A mean evaluation asks for [`Self::Means`] only.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestedCarrier {
    /// Means of the coordinates the utilities read.
    Means,
    /// A predictive law or predictive simulation. Not draws, and not a posterior.
    Predictions,
    /// Marginal or joint draws. Not a posterior.
    Draws,
}

impl RequestedCarrier {
    /// Stable name: `means`, `predictions`, or `draws`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Means => "means",
            Self::Predictions => "predictions",
            Self::Draws => "draws",
        }
    }
}

/// A carrier that means do not provide.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MissingCarrier {
    /// A predictive law or predictive simulation.
    Predictions,
    /// Marginal or joint draws.
    Draws,
}

impl MissingCarrier {
    /// Stable name: `predictions` or `draws`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Predictions => "predictions",
            Self::Draws => "draws",
        }
    }
}

/// The batch the evaluation asked for.
///
/// One record, one carrier. `batches` is how many batches that request is,
/// and `items` is how many distinct coordinates it holds. Shared coordinates
/// are one item. This admits no worker and allocates no scratch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BatchRecord {
    /// Carrier requested.
    pub carrier: RequestedCarrier,
    /// Batches in the request. A mean evaluation is one batch.
    pub batches: usize,
    /// Distinct coordinates in the batch.
    pub items: usize,
}

impl BatchRecord {
    /// One batch of `items` means.
    #[must_use]
    pub const fn means(items: usize) -> Self {
        Self { carrier: RequestedCarrier::Means, batches: 1, items }
    }
}

/// What the evaluation asked to keep.
///
/// [`Self::means_summary`] keeps the mean summary. `fits` is zero: retaining
/// that summary is not a fit, and it does not retain predictions or draws.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RetentionRecord {
    /// Retention mode of the request.
    pub mode: OutputRetention,
    /// Fits requested. Zero for a mean summary.
    pub fits: usize,
}

impl RetentionRecord {
    /// Summary of the means. Not a fit and not a draw store.
    #[must_use]
    pub const fn means_summary() -> Self {
        Self { mode: OutputRetention::SummaryOnly, fits: 0 }
    }
}

/// An affine decision scored from means, with the request that scoring made.
#[derive(Clone, Debug, PartialEq)]
pub struct DemandEvaluation {
    /// Decision under the mean source. No draws, no regret, no posterior.
    pub decision: DecisionResult,
    /// Batch that was requested.
    pub batching: BatchRecord,
    /// Retention that was requested.
    pub retention: RetentionRecord,
    /// Predictive simulations requested. Zero when means answer the utility.
    pub predictions_requested: usize,
    /// Draws requested. Zero when means answer the utility.
    pub draws_requested: usize,
}

/// Why a mean evaluation refused.
#[derive(Clone, Debug, PartialEq, Error)]
pub enum EvaluationError {
    /// The contract is invalid.
    #[error("decision contract refused: {0:?}")]
    Contract(DecisionContractError),
    /// The demand accepts means, and the mean source itself refused.
    #[error("decision evaluation refused: {0:?}")]
    Evaluation(DecisionEvalError),
    /// Means cannot answer the declared demand.
    ///
    /// `missing_label` is `predictions`, `draws`, or `predictions or draws`.
    #[error("means alone cannot answer this demand; missing {missing_label}")]
    MeansInsufficient {
        /// Carriers the demand still needs.
        missing: &'static [MissingCarrier],
        /// Source representations that would have sufficed.
        needed: Vec<SourceRepresentation>,
        /// Words the refusal uses for `missing`.
        missing_label: &'static str,
    },
    /// The demand is an identified set or a set of structures.
    ///
    /// Means are not that set, and this refusal does not put a probability on it.
    #[error("means alone are not an identified set or a set of structures")]
    IdentifiedSet,
}

/// Evaluate `contract` on `source` using the contract's declared source demand.
///
/// An affine expected utility or expected loss, with no hard constraint, is
/// scored from the means. The returned record requests no predictions and no
/// draws. Any other declared demand is refused before that scoring.
///
/// # Errors
///
/// [`EvaluationError::MeansInsufficient`] when the demand needs predictions or
/// draws. [`EvaluationError::IdentifiedSet`] when the demand is a set. Contract,
/// coordinate, meaning, and non-finite failures pass through after a demand that
/// does accept means.
#[must_use = "a demand that means cannot answer is refused"]
pub fn evaluate_on_means(
    contract: &DecisionContract,
    source: &MeanSource,
) -> Result<DemandEvaluation, EvaluationError> {
    let requirement = declared_requirement(contract)?;
    if requirement.is_identified_set() {
        return Err(EvaluationError::IdentifiedSet);
    }
    match requirement.check(&[SourceRepresentation::Mean]) {
        Ok(_) => {}
        Err(DecisionContractError::MissingSource { needed }) => {
            return Err(means_insufficient(needed));
        }
        Err(other) => return Err(EvaluationError::Contract(other)),
    }
    let decision = evaluate_contract_on_means(contract, source).map_err(lift_eval)?;
    Ok(DemandEvaluation {
        decision,
        batching: BatchRecord::means(distinct_inputs(contract)),
        retention: RetentionRecord::means_summary(),
        predictions_requested: 0,
        draws_requested: 0,
    })
}

fn declared_requirement(contract: &DecisionContract) -> Result<SourceRequirement, EvaluationError> {
    match contract.source_requirement().map_err(EvaluationError::Contract)? {
        Some(requirement) => Ok(requirement),
        None => Err(EvaluationError::Contract(DecisionContractError::InvalidDeclaration(
            "a decision compares at least two actions",
        ))),
    }
}

fn means_insufficient(needed: Vec<SourceRepresentation>) -> EvaluationError {
    let predictions = needed.iter().any(|representation| {
        matches!(representation, SourceRepresentation::Cdf | SourceRepresentation::QuantileFunction)
    });
    let draws = needed.iter().any(|representation| {
        matches!(
            representation,
            SourceRepresentation::MarginalDraws | SourceRepresentation::JointDraws
        )
    });
    let (missing, missing_label) = match (predictions, draws) {
        (true, true) => (PREDICTIONS_OR_DRAWS, "predictions or draws"),
        (true, false) => (ONLY_PREDICTIONS, "predictions"),
        (false, true) => (ONLY_DRAWS, "draws"),
        (false, false) => {
            return EvaluationError::MeansInsufficient {
                missing: &[],
                needed,
                missing_label: "a sufficient representation other than a mean",
            };
        }
    };
    EvaluationError::MeansInsufficient { missing, needed, missing_label }
}

fn lift_eval(error: DecisionEvalError) -> EvaluationError {
    match error {
        DecisionEvalError::Contract(error) => EvaluationError::Contract(error),
        DecisionEvalError::MeanSourceInsufficient { needed } => means_insufficient(needed),
        other => EvaluationError::Evaluation(other),
    }
}

fn distinct_inputs(contract: &DecisionContract) -> usize {
    let mut seen: Vec<&ScientificQuantity> = Vec::new();
    for action in &contract.actions {
        for quantity in &action.inputs {
            let already = seen.iter().any(|kept| kept.require_same_coordinate(quantity).is_ok());
            if !already {
                seen.push(quantity);
            }
        }
    }
    seen.len()
}

#[cfg(test)]
mod tests {
    use antecedent_contract::QuantityRole;

    use super::*;
    use crate::decision_contract::{
        ActionKind, DecisionAction, DecisionCriterion, HardConstraint, StructuralPolicy,
        UtilityExpr,
    };
    use crate::decision_eval::Verdict;

    fn quantity(regime: &str) -> ScientificQuantity {
        ScientificQuantity {
            variable_id: "y".into(),
            variable_name: "y".into(),
            role: QuantityRole::Outcome,
            units: "mmHg".into(),
            population_id: "target".into(),
            regime_id: regime.into(),
            horizon: 0,
            functional_id: "mean".into(),
            conditioning: vec![],
            transform_id: "identity".into(),
        }
    }

    /// `E[Y | do(a)] = 1 + 2a` on `a = 0, 1, 2`.
    fn source() -> MeanSource {
        MeanSource {
            coordinates: vec![quantity("do(a=0)"), quantity("do(a=1)"), quantity("do(a=2)")],
            means: vec![1.0, 3.0, 5.0],
            provider_id: "lab".into(),
            snapshot_id: "snap-9".into(),
            causal_contract_id: "checked-contract".into(),
            rng_id: "none:mean_grid".into(),
        }
    }

    fn affine() -> UtilityExpr {
        UtilityExpr::difference(
            UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Const(2.0)),
            UtilityExpr::Const(1.0),
        )
    }

    fn contract_with(utility: UtilityExpr) -> DecisionContract {
        DecisionContract {
            actions: vec![
                DecisionAction {
                    id: "wait".into(),
                    kind: ActionKind::Intervention,
                    inputs: vec![quantity("do(a=0)")],
                    utility: affine(),
                },
                DecisionAction {
                    id: "treat".into(),
                    kind: ActionKind::Intervention,
                    inputs: vec![quantity("do(a=2)")],
                    utility,
                },
            ],
            utility_units: "utility".into(),
            criterion: DecisionCriterion::PosteriorExpectedUtility,
            constraints: vec![],
            target_population: "target".into(),
            horizon: 0,
            structural_policy: StructuralPolicy::ReportOnly,
        }
    }

    fn contract() -> DecisionContract {
        contract_with(affine())
    }

    fn assert_missing(error: &EvaluationError, label: &str, carriers: &[MissingCarrier]) {
        let EvaluationError::MeansInsufficient { missing, missing_label, needed } = error else {
            panic!("expected MeansInsufficient, got {error:?}");
        };
        assert_eq!(*missing_label, label);
        assert_eq!(*missing, carriers);
        assert!(!needed.iter().any(|representation| *representation == SourceRepresentation::Mean));
        let text = error.to_string();
        assert!(text.contains(label), "{text}");
    }

    #[test]
    fn affine_utility_from_means_requests_neither_predictions_nor_draws() {
        let record = evaluate_on_means(&contract(), &source()).unwrap();
        // 2 * 1 - 1 = 1 and 2 * 5 - 1 = 9.
        assert!((record.decision.actions[0].expected_utility - 1.0).abs() < 1e-12);
        assert!((record.decision.actions[1].expected_utility - 9.0).abs() < 1e-12);
        assert_eq!(record.decision.verdict, Verdict::UniquelyOptimal("treat".into()));
        assert_eq!(record.predictions_requested, 0);
        assert_eq!(record.draws_requested, 0);
        assert_eq!(record.batching.carrier, RequestedCarrier::Means);
        assert_eq!(record.decision.n_draws, 0);
        assert!(record.decision.evpi.is_none());
        assert!(record.decision.actions.iter().all(|outcome| {
            outcome.standard_error.is_none()
                && outcome.expected_regret.is_none()
                && outcome.max_regret.is_none()
        }));
        assert!(record.decision.assumptions.iter().any(|note| note.contains("means only")));

        let mut loss = contract();
        loss.criterion = DecisionCriterion::PosteriorExpectedLoss;
        let loss = evaluate_on_means(&loss, &source()).unwrap();
        assert_eq!(loss.decision.verdict, Verdict::UniquelyOptimal("wait".into()));
        assert_eq!(loss.predictions_requested, 0);
        assert_eq!(loss.draws_requested, 0);
    }

    #[test]
    fn nonlinear_utility_given_only_means_names_the_missing_draws() {
        let squared = UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(0));
        let error = evaluate_on_means(&contract_with(squared), &source()).unwrap_err();
        assert_missing(&error, "draws", ONLY_DRAWS);
        let text = error.to_string();
        assert!(!text.contains("predictions"), "{text}");
        let EvaluationError::MeansInsufficient { needed, .. } = &error else {
            panic!("expected MeansInsufficient, got {error:?}");
        };
        assert!(needed.contains(&SourceRepresentation::JointDraws));
    }

    #[test]
    fn a_risk_demand_given_only_means_names_the_missing_predictions_or_draws() {
        let mut quantile = contract_with(UtilityExpr::Input(0));
        quantile.actions[0].utility = UtilityExpr::Input(0);
        quantile.criterion = DecisionCriterion::Quantile { p: 0.5 };
        let error = evaluate_on_means(&quantile, &source()).unwrap_err();
        assert_missing(&error, "predictions or draws", PREDICTIONS_OR_DRAWS);

        let mut constrained = contract();
        constrained.constraints.push(HardConstraint {
            id: "cap".into(),
            expr: UtilityExpr::Input(0),
            bound: 10.0,
            min_probability: 0.9,
            units: "mmHg".into(),
            applies_to: vec![],
        });
        let error = evaluate_on_means(&constrained, &source()).unwrap_err();
        assert_missing(&error, "predictions or draws", PREDICTIONS_OR_DRAWS);
        let text = error.to_string();
        assert!(text.contains("predictions"), "{text}");
        assert!(text.contains("draws"), "{text}");

        let mut regret = contract();
        regret.criterion = DecisionCriterion::ExpectedRegret;
        let error = evaluate_on_means(&regret, &source()).unwrap_err();
        assert_missing(&error, "draws", ONLY_DRAWS);
    }

    #[test]
    fn batching_and_retention_record_what_was_requested() {
        let record = evaluate_on_means(&contract(), &source()).unwrap();
        assert_eq!(record.batching, BatchRecord::means(2));
        assert_eq!(record.retention, RetentionRecord::means_summary());
        assert_eq!(record.retention.mode, OutputRetention::SummaryOnly);
        assert_eq!(record.retention.fits, 0);
        assert_eq!(record.predictions_requested, 0);
        assert_eq!(record.draws_requested, 0);

        let mut tied = contract();
        tied.actions[1].inputs = tied.actions[0].inputs.clone();
        let tied = evaluate_on_means(&tied, &source()).unwrap();
        assert_eq!(tied.batching.items, 1);
        assert_eq!(tied.batching.batches, 1);
        assert_eq!(tied.retention.fits, 0);
    }

    #[test]
    fn an_identified_set_is_not_formed_from_means() {
        for criterion in
            [DecisionCriterion::MinimaxOverIdentifiedSet, DecisionCriterion::MaximinOverStructures]
        {
            let mut robust = contract();
            robust.criterion = criterion;
            let error = evaluate_on_means(&robust, &source()).unwrap_err();
            assert_eq!(error, EvaluationError::IdentifiedSet);
            let text = error.to_string();
            assert!(!text.contains("predictions"), "{text}");
            assert!(!text.contains("draws"), "{text}");
            assert!(!text.contains("posterior"), "{text}");
        }
    }

    #[test]
    fn a_missing_mean_coordinate_is_not_a_request_for_draws() {
        let mut missing = source();
        missing.coordinates.pop();
        missing.means.pop();
        let error = evaluate_on_means(&contract(), &missing).unwrap_err();
        assert!(matches!(
            error,
            EvaluationError::Evaluation(DecisionEvalError::QuantityNotFound { .. })
        ));
    }
}
