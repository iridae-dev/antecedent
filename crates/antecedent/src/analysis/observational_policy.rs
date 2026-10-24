//! Observational policy value under a declared treatment support.
//!
//! The caller supplies the outcome, the observed treatment, the covariates they declare
//! sufficient for adjustment and the declared treatment support, and chooses how the policy
//! is evaluated; there is no default:
//!
//! * `prespecified`: the caller supplies the action the policy takes at each row, declares
//!   that the policy was fixed without these rows, and the value is scored on every row with
//!   cross-fitted nuisance models;
//! * `held_out`: a policy is learned on a seeded learning part and scored on the independent
//!   held-out rows;
//! * `cross_fitted_learning`: the policy is learned on the other folds for each fold and
//!   scored on the fold, a selection-aware value of the learning procedure.
//!
//! A policy declared as learned on the rows that score it is refused. The estimator is the
//! cross-fitted doubly robust score of [`antecedent_estimate::observational_policy`]; its
//! standard error is an influence-function standard error whose calibration is **unmeasured**,
//! so no interval is produced.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::ExecutionContext;
use antecedent_estimate::EstimationError;
use antecedent_estimate::observational_policy::{
    KnownPropensity, ObservationalPolicyValue as Estimator, PolicyData, PolicyEvaluation,
    PolicyObjective, PolicyOrigin,
};
use serde::{Deserialize, Serialize};

use super::multi_level_treatment::AdjustmentColumn;
use crate::error::CausalError;

/// Calibration standing of the reported standard errors.
pub const OBSERVATIONAL_POLICY_CALIBRATION: &str = "unmeasured";

/// The declared support, objective and evaluation.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationalPolicySpecWire {
    /// The declared treatment support; required.
    #[serde(default)]
    pub support: Option<Vec<f64>>,
    /// `maximize` or `minimize`; required.
    #[serde(default)]
    pub objective: Option<String>,
    /// `prespecified`, `held_out` or `cross_fitted_learning`; required.
    #[serde(default)]
    pub evaluation: Option<String>,
    /// The action the policy takes at each row (`prespecified`).
    #[serde(default)]
    pub policy_actions: Option<Vec<f64>>,
    /// `prespecified` or `learned_on_evaluation_rows` (`prespecified` evaluation); required
    /// there.
    #[serde(default)]
    pub policy_origin: Option<String>,
    /// Fraction of rows held out to score a learned policy (`held_out`).
    #[serde(default)]
    pub evaluation_fraction: Option<f64>,
    /// An action to compare the policy against.
    #[serde(default)]
    pub reference: Option<f64>,
    /// Cross-fit folds; `None` uses the library default.
    #[serde(default)]
    pub folds: Option<usize>,
    /// Propensity floor of every declared action; `None` uses the library default.
    #[serde(default)]
    pub min_propensity: Option<f64>,
    /// Known assignment probability of each declared action, in support order, for a
    /// randomized design; replaces the fitted propensity model.
    #[serde(default)]
    pub known_propensity: Option<Vec<f64>>,
    /// Known assignment probabilities per row: one column per declared action, in support
    /// order. Exclusive with `known_propensity`.
    #[serde(default)]
    pub known_propensity_columns: Option<Vec<Vec<f64>>>,
    /// Seed of the fold plan or held-out split.
    #[serde(default)]
    pub seed: u64,
}

/// A policy-value request.
#[derive(Clone, Debug, PartialEq)]
pub struct ObservationalPolicyRequest {
    /// Outcome of each row.
    pub outcome: Vec<f64>,
    /// Observed treatment level of each row.
    pub treatment: Vec<f64>,
    /// Declared adjustment columns.
    pub adjustment: Vec<AdjustmentColumn>,
    /// Support, objective and evaluation.
    pub spec: ObservationalPolicySpecWire,
}

/// One declared action's summaries, on the wire.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct ActionValueWire {
    /// Treatment level.
    pub level: f64,
    /// Doubly robust mean outcome if every scored unit were given this action.
    pub value: f64,
    /// Standard error of that mean.
    pub se: f64,
    /// Fraction of the scored rows to which the policy gives this action.
    pub policy_share: f64,
    /// Smallest out-of-sample propensity of the action.
    pub propensity_min: f64,
    /// Complete-case rows observed at the action.
    pub rows: u64,
}

/// The policy against a reference action, on the wire.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct ReferenceWire {
    /// The reference action.
    pub level: f64,
    /// Policy value minus the reference action's value.
    pub incremental_value: f64,
    /// Standard error of the paired difference.
    pub incremental_se: f64,
}

/// The self-describing result.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ObservationalPolicyReportWire {
    /// `prespecified`, `held_out` or `cross_fitted_learning`.
    pub evaluation: String,
    /// What the value is the value of.
    pub estimand: String,
    /// The assumptions the numbers rest on.
    pub assumptions: Vec<String>,
    /// `maximize` or `minimize`.
    pub objective: String,
    /// The declared treatment support.
    pub support: Vec<f64>,
    /// Doubly robust policy value.
    pub value: f64,
    /// Influence-function standard error.
    pub se: f64,
    /// Each declared action given to everyone, in declaration order.
    pub action_values: Vec<ActionValueWire>,
    /// The policy against the declared reference action.
    pub reference: Option<ReferenceWire>,
    /// `estimated` (multinomial logit) or `known` (declared assignment probabilities).
    pub propensity_source: String,
    /// Rows the value is scored on.
    pub evaluation_rows: u64,
    /// Rows the policy and nuisance models were fitted on.
    pub learning_rows: u64,
    /// Rows dropped for a missing cell.
    pub rows_dropped_missing: u64,
    /// Original row index of each scored row.
    pub row_index: Vec<u64>,
    /// The action the policy takes at each scored row.
    pub policy_actions: Vec<f64>,
    /// The doubly robust score of each scored row.
    pub scores: Vec<f64>,
    /// Calibration standing of the standard errors.
    pub calibration: String,
}

/// Observational policy value.
pub struct ObservationalPolicyValue;

impl ObservationalPolicyValue {
    /// Estimate several policies' values over one cache scope.
    ///
    /// The cross-fitted propensity and per-action outcome models do not depend on the policy
    /// being scored, so requests over the same rows, support and splits fit them once. The
    /// cache is charged to the admission pool of `ctx` and, when `retention` is not
    /// [`RetentionMode::FullReusable`], retains nothing. Each result equals the request's own
    /// [`Self::evaluate`]; one request's refusal does not affect another's.
    ///
    /// [`RetentionMode::FullReusable`]: antecedent_state::retention::RetentionMode::FullReusable
    #[must_use]
    pub fn evaluate_many(
        requests: &[ObservationalPolicyRequest],
        ctx: &ExecutionContext,
        retention: antecedent_state::retention::RetentionMode,
    ) -> Vec<Result<ObservationalPolicyReportWire, CausalError>> {
        let cache = super::batch::batch_nuisance_cache(ctx, retention);
        requests.iter().map(|request| cache.scope(|| Self::evaluate(request)).0).collect()
    }

    /// Estimate the declared policy's value.
    ///
    /// # Errors
    /// `required_option_missing` for an absent support, objective, evaluation or policy
    /// origin; `invalid_argument` for malformed columns or declarations;
    /// `policy_learned_on_evaluation_rows` for a policy learned on the scoring rows;
    /// `treatment_level_undeclared` for a row at no declared action; `positivity_violated`
    /// for an action outside the support or a propensity below the floor; `arm_not_populated`;
    /// and the estimator's GLM and regression refusals.
    #[expect(
        clippy::too_many_lines,
        reason = "one linear evaluation: resolve the request, run the estimator, then assemble the report wire"
    )]
    pub fn evaluate(
        request: &ObservationalPolicyRequest,
    ) -> Result<ObservationalPolicyReportWire, CausalError> {
        let refused = |code: &'static str, detail: &str, message: &str| {
            CausalError::Estimate(EstimationError::refused(code, format!("{detail}: {message}")))
        };
        let missing = |detail: &str, message: &str| {
            refused(antecedent_core::reason_code!("required_option_missing"), detail, message)
        };
        let invalid = |detail: &str, message: &str| {
            refused(antecedent_core::reason_code!("invalid_argument"), detail, message)
        };
        let spec = &request.spec;
        let Some(support) = spec.support.clone() else {
            return Err(missing(
                "policy_value.support_missing",
                "the treatment support must be declared: the actions the policy may take and whose propensity must be positive",
            ));
        };
        let objective = match spec.objective.as_deref() {
            None => {
                return Err(missing(
                    "policy_value.objective_missing",
                    "declare whether a larger (maximize) or smaller (minimize) outcome is better",
                ));
            }
            Some("maximize") => PolicyObjective::Maximize,
            Some("minimize") => PolicyObjective::Minimize,
            Some(other) => {
                return Err(invalid(
                    "policy_value.objective_unknown",
                    &format!("unknown objective {other:?}; use maximize or minimize"),
                ));
            }
        };
        let evaluation = match spec.evaluation.as_deref() {
            None => {
                return Err(missing(
                    "policy_value.evaluation_missing",
                    "declare the evaluation: prespecified, held_out or cross_fitted_learning",
                ));
            }
            Some("prespecified") => {
                let Some(actions) = spec.policy_actions.clone() else {
                    return Err(missing(
                        "policy_value.policy_actions_missing",
                        "a prespecified evaluation needs the action the policy takes at each row",
                    ));
                };
                let origin = match spec.policy_origin.as_deref() {
                    None => {
                        return Err(missing(
                            "policy_value.policy_origin_missing",
                            "declare where the policy came from: prespecified, or learned_on_evaluation_rows (which is refused)",
                        ));
                    }
                    Some("prespecified") => PolicyOrigin::Prespecified,
                    Some("learned_on_evaluation_rows") => PolicyOrigin::LearnedOnEvaluationRows,
                    Some(other) => {
                        return Err(invalid(
                            "policy_value.policy_origin_unknown",
                            &format!(
                                "unknown policy origin {other:?}; use prespecified or learned_on_evaluation_rows"
                            ),
                        ));
                    }
                };
                PolicyEvaluation::Prespecified { actions: Arc::from(actions), origin }
            }
            Some("held_out") => {
                let Some(evaluation_fraction) = spec.evaluation_fraction else {
                    return Err(missing(
                        "policy_value.evaluation_fraction_missing",
                        "a held-out evaluation needs the fraction of rows held out to score the learned policy",
                    ));
                };
                PolicyEvaluation::HeldOut { evaluation_fraction }
            }
            Some("cross_fitted_learning") => PolicyEvaluation::CrossFitted,
            Some(other) => {
                return Err(invalid(
                    "policy_value.evaluation_unknown",
                    &format!(
                        "unknown evaluation {other:?}; use prespecified, held_out or cross_fitted_learning"
                    ),
                ));
            }
        };
        let rows = request.outcome.len();
        let mut names: Vec<&str> = request.adjustment.iter().map(|c| c.name.as_str()).collect();
        names.sort_unstable();
        if request.adjustment.iter().any(|c| c.values.len() != rows)
            || names.iter().any(|n| n.is_empty())
            || names.windows(2).any(|pair| pair[0] == pair[1])
        {
            return Err(invalid(
                "policy_value.invalid_data",
                "adjustment columns must match the rows and have distinct, non-empty names",
            ));
        }
        let mut estimator = Estimator::new(Arc::from(support.clone()), objective, evaluation);
        estimator.reference = spec.reference;
        estimator.known_propensity = match (&spec.known_propensity, &spec.known_propensity_columns)
        {
            (None, None) => None,
            (Some(values), None) => Some(KnownPropensity::Constant(Arc::from(values.clone()))),
            (None, Some(columns)) => Some(KnownPropensity::PerRow(
                columns.iter().map(|c| Arc::from(c.clone())).collect(),
            )),
            (Some(_), Some(_)) => {
                return Err(invalid(
                    "policy_value.invalid_known_propensity",
                    "declare known propensities either as one constant per action or as per-row columns, not both",
                ));
            }
        };
        if let Some(folds) = spec.folds {
            estimator.folds = folds;
        }
        if let Some(floor) = spec.min_propensity {
            estimator.min_propensity = floor;
        }
        let covariates: Vec<&[f64]> =
            request.adjustment.iter().map(|c| c.values.as_slice()).collect();
        let ctx = ExecutionContext::production_default(spec.seed);
        let estimate = estimator
            .estimate(
                &PolicyData {
                    outcome: &request.outcome,
                    treatment: &request.treatment,
                    covariates: &covariates,
                },
                &ctx,
            )
            .map_err(CausalError::from)?;
        Ok(ObservationalPolicyReportWire {
            evaluation: estimate.evaluation.as_str().into(),
            estimand: estimand(&estimate.evaluation).into(),
            assumptions: assumptions(&estimate.evaluation),
            objective: objective.as_str().into(),
            support,
            value: estimate.value,
            se: estimate.se,
            action_values: estimate
                .action_values
                .iter()
                .map(|a| ActionValueWire {
                    level: a.level,
                    value: a.value,
                    se: a.se,
                    policy_share: a.policy_share,
                    propensity_min: a.propensity_min,
                    rows: a.rows,
                })
                .collect(),
            reference: estimate.reference.map(|r| ReferenceWire {
                level: r.level,
                incremental_value: r.incremental_value,
                incremental_se: r.incremental_se,
            }),
            propensity_source: if estimator.known_propensity.is_some() {
                "known"
            } else {
                "estimated"
            }
            .into(),
            evaluation_rows: estimate.evaluation_rows,
            learning_rows: estimate.learning_rows,
            rows_dropped_missing: estimate.rows_dropped_missing,
            row_index: estimate.row_index.iter().map(|id| id.get()).collect(),
            policy_actions: estimate.policy_actions,
            scores: estimate.scores,
            calibration: OBSERVATIONAL_POLICY_CALIBRATION.into(),
        })
    }
}

fn estimand(evaluation: &PolicyEvaluation) -> &'static str {
    match evaluation {
        PolicyEvaluation::Prespecified { .. } => {
            "mean outcome if every unit were given the action the prespecified policy picks, E[Y(pi(Z))]"
        }
        PolicyEvaluation::HeldOut { .. } => {
            "mean outcome under the one policy learned on the learning rows, E[Y(pi_hat(Z))], scored on independent held-out rows"
        }
        PolicyEvaluation::CrossFitted => {
            "average mean outcome of policies learned on the other folds, each scored on its own fold: the value of the learning procedure at (K - 1) / K of the data, not of one final policy"
        }
    }
}

fn assumptions(evaluation: &PolicyEvaluation) -> Vec<String> {
    let mut list = vec![
        "conditional exchangeability: the declared covariates are a sufficient adjustment set (declared, not checked)".to_owned(),
        "positivity: every action of the declared support has propensity above the floor at every scored row".to_owned(),
        "consistency and no interference between units".to_owned(),
        "the propensity model or the per-action outcome models are correctly specified (double robustness needs one side)".to_owned(),
    ];
    list.push(
        match evaluation {
            PolicyEvaluation::Prespecified { .. } => {
                "the policy was fixed without the outcomes or treatments of the scored rows (declared)"
            }
            PolicyEvaluation::HeldOut { .. } => {
                "the policy and nuisance models are fitted on rows disjoint from the scored rows, so the interval conditions on the learned policy"
            }
            PolicyEvaluation::CrossFitted => {
                "each row is scored with a policy and nuisance models that never saw it; the standard error treats the fold policies as fixed, which is valid when the learned policy stabilises across folds"
            }
        }
        .to_owned(),
    );
    list
}
