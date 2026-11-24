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
//! standard error is an influence-function standard error whose calibration is **`not_measured`**,
//! so no interval is produced.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use antecedent_contract::{ClaimKind, DataClassification};
use antecedent_core::{
    ByteEstimates, CanonicalWriter, ExecutionContext, PhysicalEffect, PhysicalOperator,
    ScoreFrequentist, ScoreModel, SemanticDigest,
};
use antecedent_estimate::EstimationError;
use antecedent_estimate::observational_policy::{
    DEFAULT_POLICY_FOLDS, KnownPropensity, ObservationalPolicyValue as Estimator, PolicyData,
    PolicyEvaluation, PolicyObjective, PolicyOrigin,
};
use antecedent_graph::{DSeparationWorkspace, Dag, DenseNodeId};
use antecedent_io::calibration::{
    CalibrationBasisWire, CalibrationKeyWire, CalibrationScopeWire, calibration_slot,
};
use antecedent_io::contract_section::CalibrationSlotWire;
use serde::{Deserialize, Serialize};

use super::multi_level_treatment::AdjustmentColumn;
use crate::analysis::program_handle::gate_program;
use crate::error::CausalError;
use crate::planner::lowered::{
    ClaimDemand, Composition, DesignShape, EvidenceDecisionProgram, EvidenceRole,
    EvidenceSourceRef, FunctionalKind, IdentificationLayer, LoweredProgram,
    NonGraphicalIdentification, PhysicalLayer, ProblemTarget, ProgramCell, ProgramIdentityScope,
    ScientificProblem, StandaloneCompositionLoweredParts, UncertaintyReading,
};
use crate::planner::statistical::{
    FoldPlan, ModelRole, ModelSlot, Realization, ResampleUnit, StatisticalProgram,
};
use crate::planner::{PlanStage, PlanView};
use crate::support::unit::{SupportKey, UnitId, require_program};
use crate::support::units_generated;

/// Calibration standing of the reported standard errors.
pub const OBSERVATIONAL_POLICY_CALIBRATION: &str = "not_measured";

/// A declared causally sufficient DAG for the policy's treatment and outcome.
/// The information set names columns available when the policy acts. Every
/// adjustment and information variable must precede treatment in this graph.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyGraphDeclaration {
    /// Distinct node names, including treatment, outcome and adjustment columns.
    pub nodes: Vec<String>,
    /// Directed edges in the declared DAG.
    pub edges: Vec<(String, String)>,
    /// Name of the treatment node.
    pub treatment: String,
    /// Name of the outcome node.
    pub outcome: String,
    /// Adjustment columns observable when the policy selects an action. A
    /// strict subset is allowed; nuisance models still fit on all columns.
    pub information_set: Vec<String>,
}

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
    /// Optional graph-checked exchangeability and policy information set.
    /// Absent requests keep the version-1 declared-adjustment route.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub causal_graph: Option<PolicyGraphDeclaration>,
    /// Seed of the fold plan or held-out split. `evaluate` runs under the caller's context and
    /// its seed; bindings that build the context from this declaration use this value.
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

/// A candidate Wald interval for a held-out learned policy. Its coverage is
/// reported separately from the point report and only through its exact
/// calibration slot; the point report keeps its original standing.
#[derive(Clone, Debug, PartialEq)]
pub struct HeldOutPolicyValueInterval {
    /// Lower endpoint of the two-sided interval.
    pub lower: f64,
    /// Upper endpoint of the two-sided interval.
    pub upper: f64,
    /// Nominal coverage level.
    pub level: f64,
    /// Registry standing of this exact construction.
    pub calibration: String,
}

/// One compiler execution and its separately keyed interval candidate.
#[derive(Clone, Debug)]
pub struct HeldOutPolicyValueCandidate {
    /// The ordinary policy report from the retained program.
    pub report: ObservationalPolicyReportWire,
    /// Wald interval of the learned policy's held-out value.
    pub interval: HeldOutPolicyValueInterval,
    /// Exact key and execution scope used for calibration matching.
    pub basis: CalibrationBasisWire,
    /// Matching record status, if a prospective record has been earned.
    pub slot: CalibrationSlotWire,
}

/// Separate unearned interval for the cross-fitted learning procedure's
/// fold-policy average. It does not describe the value of one final policy.
#[derive(Clone, Debug, PartialEq)]
pub struct CrossFittedPolicyValueInterval {
    /// Lower endpoint of the two-sided interval.
    pub lower: f64,
    /// Upper endpoint of the two-sided interval.
    pub upper: f64,
    /// Nominal level, without an earned coverage claim.
    pub level: f64,
    /// Standing of the exact cross-fitted construction.
    pub calibration: String,
}

/// One compiler execution and its separately keyed cross-fitted candidate.
#[derive(Clone, Debug)]
pub struct CrossFittedPolicyValueCandidate {
    /// Ordinary cross-fitted report from the retained program.
    pub report: ObservationalPolicyReportWire,
    /// Wald candidate for the fold-policy average.
    pub interval: CrossFittedPolicyValueInterval,
    /// Exact construction and runtime scope.
    pub basis: CalibrationBasisWire,
    /// Matching scientific record status, if ever earned.
    pub slot: CalibrationSlotWire,
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
    /// [`Self::evaluate`] under `ctx` (its seed, not each request's declared seed, drives the
    /// cross-fit); one request's refusal does not affect another's.
    ///
    /// [`RetentionMode::FullReusable`]: antecedent_state::retention::RetentionMode::FullReusable
    #[must_use]
    pub fn evaluate_many(
        requests: &[ObservationalPolicyRequest],
        ctx: &ExecutionContext,
        retention: antecedent_state::retention::RetentionMode,
    ) -> Vec<Result<ObservationalPolicyReportWire, CausalError>> {
        let cache = super::batch::batch_nuisance_cache(ctx, retention);
        requests.iter().map(|request| cache.scope(|| Self::evaluate(request, ctx)).0).collect()
    }

    fn policy_declarations(
        spec: &ObservationalPolicySpecWire,
    ) -> Result<(Vec<f64>, PolicyObjective, PolicyEvaluation), CausalError> {
        let refused = |code: &'static str, detail: &str, message: &str| {
            CausalError::Estimate(EstimationError::refused_detail(code, detail, message))
        };
        let missing = |detail: &str, message: &str| {
            refused(antecedent_core::reason_code!("required_option_missing"), detail, message)
        };
        let invalid = |detail: &str, message: &str| {
            refused(antecedent_core::reason_code!("invalid_argument"), detail, message)
        };
        let Some(support) = spec.support.clone() else {
            return Err(missing(
                antecedent_core::reason_details::policy_value::SUPPORT_MISSING,
                "the treatment support must be declared: the actions the policy may take and whose propensity must be positive",
            ));
        };
        let objective = match spec.objective.as_deref() {
            None => {
                return Err(missing(
                    antecedent_core::reason_details::policy_value::OBJECTIVE_MISSING,
                    "declare whether a larger (maximize) or smaller (minimize) outcome is better",
                ));
            }
            Some("maximize") => PolicyObjective::Maximize,
            Some("minimize") => PolicyObjective::Minimize,
            Some(other) => {
                return Err(invalid(
                    antecedent_core::reason_details::policy_value::OBJECTIVE_UNKNOWN,
                    &format!("unknown objective {other:?}; use maximize or minimize"),
                ));
            }
        };
        let evaluation = match spec.evaluation.as_deref() {
            None => {
                return Err(missing(
                    antecedent_core::reason_details::policy_value::EVALUATION_MISSING,
                    "declare the evaluation: prespecified, held_out or cross_fitted_learning",
                ));
            }
            Some("prespecified") => {
                let Some(actions) = spec.policy_actions.clone() else {
                    return Err(missing(
                        antecedent_core::reason_details::policy_value::POLICY_ACTIONS_MISSING,
                        "a prespecified evaluation needs the action the policy takes at each row",
                    ));
                };
                let origin = match spec.policy_origin.as_deref() {
                    None => {
                        return Err(missing(
                            antecedent_core::reason_details::policy_value::POLICY_ORIGIN_MISSING,
                            "declare where the policy came from: prespecified, or learned_on_evaluation_rows (which is refused)",
                        ));
                    }
                    Some("prespecified") => PolicyOrigin::Prespecified,
                    Some("learned_on_evaluation_rows") => PolicyOrigin::LearnedOnEvaluationRows,
                    Some(other) => {
                        return Err(invalid(
                            antecedent_core::reason_details::policy_value::POLICY_ORIGIN_UNKNOWN,
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
                        antecedent_core::reason_details::policy_value::EVALUATION_FRACTION_MISSING,
                        "a held-out evaluation needs the fraction of rows held out to score the learned policy",
                    ));
                };
                PolicyEvaluation::HeldOut { evaluation_fraction }
            }
            Some("cross_fitted_learning") => PolicyEvaluation::CrossFitted,
            Some(other) => {
                return Err(invalid(
                    antecedent_core::reason_details::policy_value::EVALUATION_UNKNOWN,
                    &format!(
                        "unknown evaluation {other:?}; use prespecified, held_out or cross_fitted_learning"
                    ),
                ));
            }
        };
        Ok((support, objective, evaluation))
    }

    /// Estimate the declared policy's value under the caller's `ctx`.
    ///
    /// A graph declaration enters the central compiler before fitting. The
    /// declaration-only route checks the
    /// declared support, objective, evaluation scheme and policy origin (a policy learned on
    /// the scoring rows is refused), the columns and the propensity floor. It does not check
    /// adjustment columns against a graph only when one is supplied. The
    /// graph check is conditional on its declared causal sufficiency. The
    /// cross-fit seed, cancellation and admission pool are `ctx`'s.
    ///
    /// # Errors
    /// `required_option_missing` for an absent support, objective, evaluation or policy
    /// origin; `invalid_argument` for malformed columns or declarations;
    /// `policy_learned_on_evaluation_rows` for a policy learned on the scoring rows;
    /// `treatment_level_undeclared` for a row at no declared action; `positivity_violated`
    /// for an action outside the support or a propensity below the floor; `arm_not_populated`;
    /// and the estimator's GLM and regression refusals.
    pub fn evaluate(
        request: &ObservationalPolicyRequest,
        ctx: &ExecutionContext,
    ) -> Result<ObservationalPolicyReportWire, CausalError> {
        if request.spec.causal_graph.is_some() && ctx.lowered_dispatch().is_none() {
            return compile_observational_policy(request.clone(), ctx)?.execute(ctx);
        }
        let spec = &request.spec;
        let (support, objective, evaluation) = Self::policy_declarations(spec)?;
        let invalid = |detail: &str, message: &str| {
            CausalError::Estimate(EstimationError::refused_detail(
                antecedent_core::reason_code!("invalid_argument"),
                detail,
                message,
            ))
        };
        let rows = request.outcome.len();
        let mut names: Vec<&str> = request.adjustment.iter().map(|c| c.name.as_str()).collect();
        names.sort_unstable();
        if request.adjustment.iter().any(|c| c.values.len() != rows)
            || names.iter().any(|n| n.is_empty())
            || names.windows(2).any(|pair| pair[0] == pair[1])
        {
            return Err(invalid(
                antecedent_core::reason_details::policy_value::INVALID_DATA,
                "adjustment columns must match the rows and have distinct, non-empty names",
            ));
        }
        let mut estimator = Estimator::new(Arc::from(support.clone()), objective, evaluation);
        estimator.policy_features = policy_feature_indices(request);
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
                    antecedent_core::reason_details::policy_value::INVALID_KNOWN_PROPENSITY,
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
        let estimate = estimator
            .estimate(
                &PolicyData {
                    outcome: &request.outcome,
                    treatment: &request.treatment,
                    covariates: &covariates,
                },
                ctx,
            )
            .map_err(CausalError::from)?;
        Ok(ObservationalPolicyReportWire {
            evaluation: estimate.evaluation.as_str().into(),
            estimand: estimand(&estimate.evaluation).into(),
            assumptions: assumptions(&estimate.evaluation, spec.causal_graph.is_some()),
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

const POLICY_PROGRAM_TAG: &str = "antecedent.observational_policy_program.v1";
const POLICY_ARTIFACT_FEATURE: &str = "observational_policy_compiled_v1";
const GRAPH_POLICY_ARTIFACT_FEATURE: &str = "observational_policy_graph_checked_v2";
const NARROW_POLICY_ARTIFACT_FEATURE: &str = "observational_policy_narrow_information_v3";

fn policy_feature_indices(request: &ObservationalPolicyRequest) -> Option<Vec<usize>> {
    let graph = request.spec.causal_graph.as_ref()?;
    if graph.information_set.len() == request.adjustment.len() {
        return None;
    }
    Some(
        request
            .adjustment
            .iter()
            .enumerate()
            .filter_map(|(index, column)| {
                graph.information_set.contains(&column.name).then_some(index)
            })
            .collect(),
    )
}
const MAX_POLICY_ARTIFACT_BYTES: usize = 80 * 1024 * 1024;

fn policy_compiler_refusal(message: &str) -> CausalError {
    CausalError::Estimate(EstimationError::refused_detail(
        antecedent_core::reason_code!("invalid_argument"),
        "observational_policy.compiler",
        message,
    ))
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "the node count is checked against the policy graph size bound above"
)]
fn check_policy_graph(
    request: &ObservationalPolicyRequest,
    declaration: &PolicyGraphDeclaration,
) -> Result<(), CausalError> {
    if declaration.nodes.len() < 2
        || declaration.nodes.len() > 128
        || declaration.edges.len() > 2048
    {
        return Err(policy_compiler_refusal("policy graph size is outside the checked bounds"));
    }
    let mut positions = BTreeMap::new();
    for (index, name) in declaration.nodes.iter().enumerate() {
        if name.trim().is_empty()
            || positions.insert(name.as_str(), DenseNodeId::from_raw(index as u32)).is_some()
        {
            return Err(policy_compiler_refusal("policy graph nodes need distinct nonempty names"));
        }
    }
    let treatment = *positions
        .get(declaration.treatment.as_str())
        .ok_or_else(|| policy_compiler_refusal("the graph has no treatment node"))?;
    let outcome = *positions
        .get(declaration.outcome.as_str())
        .ok_or_else(|| policy_compiler_refusal("the graph has no outcome node"))?;
    if treatment == outcome {
        return Err(policy_compiler_refusal("treatment and outcome must be distinct"));
    }
    let mut graph = Dag::with_variables(declaration.nodes.len() as u32);
    for (from, to) in &declaration.edges {
        let from = *positions
            .get(from.as_str())
            .ok_or_else(|| policy_compiler_refusal("an edge has an unknown parent"))?;
        let to = *positions
            .get(to.as_str())
            .ok_or_else(|| policy_compiler_refusal("an edge has an unknown child"))?;
        graph
            .insert_directed(from, to)
            .map_err(|_| policy_compiler_refusal("policy graph edges must form a DAG"))?;
    }
    let mut adjustment = BTreeSet::new();
    for column in &request.adjustment {
        let node = *positions.get(column.name.as_str()).ok_or_else(|| {
            policy_compiler_refusal("an adjustment column is absent from the graph")
        })?;
        if node == treatment || node == outcome || !adjustment.insert(node) {
            return Err(policy_compiler_refusal(
                "adjustment columns must be distinct pretreatment nodes",
            ));
        }
    }
    let mut information = BTreeSet::new();
    for name in &declaration.information_set {
        let node = *positions.get(name.as_str()).ok_or_else(|| {
            policy_compiler_refusal("a policy information variable is absent from the graph")
        })?;
        if !adjustment.contains(&node) || !information.insert(node) {
            return Err(policy_compiler_refusal(
                "policy information must be distinct adjustment columns",
            ));
        }
    }
    let mut descendants = BTreeSet::new();
    let mut frontier = vec![treatment];
    while let Some(parent) = frontier.pop() {
        for &child in graph.children(parent) {
            if descendants.insert(child) {
                frontier.push(child);
            }
        }
    }
    if adjustment.iter().any(|node| descendants.contains(node)) {
        return Err(policy_compiler_refusal(
            "post-treatment adjustment or policy information is forbidden",
        ));
    }
    let backdoor = graph
        .copy_filtered(|from, _| from != treatment)
        .map_err(|_| policy_compiler_refusal("the policy backdoor graph could not be formed"))?;
    let set: Vec<_> = adjustment.into_iter().collect();
    if !backdoor
        .is_d_separated(treatment, outcome, &set, &mut DSeparationWorkspace::default())
        .map_err(|_| policy_compiler_refusal("policy graph separation could not be checked"))?
    {
        return Err(policy_compiler_refusal(
            "the declared adjustment set leaves an open backdoor path",
        ));
    }
    Ok(())
}

fn policy_request_digest(
    request: &ObservationalPolicyRequest,
    seed: u64,
) -> Result<[u8; 32], CausalError> {
    let mut writer = CanonicalWriter::new(POLICY_PROGRAM_TAG);
    writer.word(seed).word(request.outcome.len() as u64);
    for value in &request.outcome {
        writer.word(value.to_bits());
    }
    writer.word(request.treatment.len() as u64);
    for value in &request.treatment {
        writer.word(value.to_bits());
    }
    writer.word(request.adjustment.len() as u64);
    for column in &request.adjustment {
        writer.text(&column.name).word(column.values.len() as u64);
        for value in &column.values {
            writer.word(value.to_bits());
        }
    }
    let spec = antecedent_io::to_cbor(&request.spec)?;
    writer.bytes(&spec);
    Ok(writer.finish())
}

/// The complete learning and evaluation schedule, bound to one six-layer
/// compiler program. A graph declaration is checked conditionally on its
/// causal sufficiency; otherwise adjustment sufficiency remains declared.
#[derive(Clone, Debug)]
pub struct CompiledObservationalPolicy {
    request: ObservationalPolicyRequest,
    seed: u64,
    lowered: Arc<LoweredProgram>,
}

impl CompiledObservationalPolicy {
    /// Every layer of the retained program, before fitting any nuisance model.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }

    /// View the retained policy program without scoring data.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }

    /// Fit and score only through the retained compiler dispatch. The context
    /// must carry the same split seed that compilation bound.
    ///
    /// # Errors
    /// A changed seed, unsupported split, positivity failure or estimator refusal.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<ObservationalPolicyReportWire, CausalError> {
        gate_program(ctx, self)?;
        if ctx.rng.master_seed() != self.seed {
            return Err(policy_compiler_refusal(
                "the execution split seed differs from the compiled policy program",
            ));
        }
        let expected = policy_request_digest(&self.request, self.seed)?;
        if self.lowered.scientific().source_digest() != Some(expected) {
            return Err(policy_compiler_refusal("the policy request changed after compilation"));
        }
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        ObservationalPolicyValue::evaluate(&self.request, &scoped)
    }

    /// Score the held-out learned policy once, then form a separate candidate
    /// interval whose calibration key describes the exact learner, split and
    /// narrower information route. No existing point-report claim is changed.
    ///
    /// # Errors
    /// A non-graph or non-held-out route, malformed score, mismatched execution
    /// context, or an estimator refusal.
    pub fn held_out_value_candidate(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<HeldOutPolicyValueCandidate, CausalError> {
        const LEVEL: f64 = 0.95;
        const Z: f64 = 1.959_963_984_540_054;
        gate_program(ctx, self)?;
        let graph = self.request.spec.causal_graph.as_ref().ok_or_else(|| {
            policy_compiler_refusal("a calibration candidate requires the graph-checked route")
        })?;
        let information = policy_feature_indices(&self.request).ok_or_else(|| {
            policy_compiler_refusal(
                "a calibration candidate requires a strict policy information subset",
            )
        })?;
        if self.request.spec.evaluation.as_deref() != Some("held_out") {
            return Err(policy_compiler_refusal(
                "a calibration candidate requires held-out policy learning",
            ));
        }
        let report = self.execute(ctx)?;
        if report.evaluation != "held_out"
            || report.evaluation_rows == 0
            || usize::try_from(report.evaluation_rows) != Ok(report.row_index.len())
            || report.row_index.len() != report.policy_actions.len()
            || report.row_index.len() != report.scores.len()
            || !report.value.is_finite()
            || !report.se.is_finite()
            || report.se <= 0.0
        {
            return Err(policy_compiler_refusal(
                "held-out policy score cannot form a candidate interval",
            ));
        }
        let mut writer =
            CanonicalWriter::new("antecedent.held_out_policy_value.calibration_coordinate.v1");
        writer
            .text(&graph.treatment)
            .text(&graph.outcome)
            .word(graph.nodes.len() as u64)
            .word(graph.edges.len() as u64);
        for node in &graph.nodes {
            writer.text(node);
        }
        for (parent, child) in &graph.edges {
            writer.text(parent).text(child);
        }
        for column in &self.request.adjustment {
            writer.text(&column.name);
        }
        for index in information {
            writer.word(index as u64);
        }
        writer
            .word(self.request.spec.folds.unwrap_or(DEFAULT_POLICY_FOLDS) as u64)
            .word(self.request.spec.evaluation_fraction.unwrap_or_default().to_bits())
            .text(self.request.spec.objective.as_deref().unwrap_or(""))
            .word(self.request.spec.min_propensity.unwrap_or_default().to_bits());
        for action in &report.support {
            writer.word(action.to_bits());
        }
        let graph_schedule = SemanticDigest::from_bytes(writer.finish()).to_hex();
        let assignment = if self.request.spec.known_propensity_columns.is_some() {
            "known_row_assignment"
        } else if self.request.spec.known_propensity.is_some() {
            "known_fixed_assignment"
        } else {
            "estimated_assignment"
        };
        let basis = CalibrationBasisWire {
            key: CalibrationKeyWire {
                validation: "not_applicable".into(),
                query: "held_out_learned_policy_value".into(),
                graph_class: "declared_dag_checked_backdoor".into(),
                structure: "fixed".into(),
                modality: "tabular".into(),
                inference: "frequentist".into(),
                estimator: "graph_narrow_policy_held_out_v3".into(),
                interval_method: "held_out_policy_score_wald_v1".into(),
                se_kind: "held_out_influence_score".into(),
                dependence: "independent_learning_evaluation_rows".into(),
                posterior: String::new(),
                functional: format!("held_out_policy_value.{assignment}.{graph_schedule}"),
                level: LEVEL,
                identification: "point_conditional_on_declared_dag".into(),
            },
            scope: CalibrationScopeWire {
                row_count: report.evaluation_rows,
                replicates_ok: None,
                posterior_draws: None,
                unidentified_mass: 0.0,
            },
        };
        let slot = calibration_slot(&basis);
        let interval = HeldOutPolicyValueInterval {
            lower: report.value - Z * report.se,
            upper: report.value + Z * report.se,
            level: LEVEL,
            calibration: slot.status.as_str().into(),
        };
        Ok(HeldOutPolicyValueCandidate { report, interval, basis, slot })
    }

    /// Score the graph-checked cross-fitted learner and form a separate
    /// candidate interval for the average value of its fold-specific policies:
    /// each realized fold policy's population mean outcome, weighted by its
    /// fold's share of the scored rows. The row-score SE includes the variation
    /// of the outcome means over the scored covariates, so it is not an interval
    /// for the scored rows' own conditional means. It assumes the policies
    /// stabilise across folds; coverage must be measured for this exact learner
    /// and graph schedule.
    ///
    /// # Errors
    /// A non-graph or non-cross-fitted route, malformed score or estimator
    /// refusal.
    pub fn cross_fitted_value_candidate(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<CrossFittedPolicyValueCandidate, CausalError> {
        const LEVEL: f64 = 0.95;
        const Z: f64 = 1.959_963_984_540_054;
        gate_program(ctx, self)?;
        let graph = self.request.spec.causal_graph.as_ref().ok_or_else(|| {
            policy_compiler_refusal("a cross-fit candidate requires the graph-checked route")
        })?;
        let information = policy_feature_indices(&self.request).ok_or_else(|| {
            policy_compiler_refusal(
                "a cross-fit candidate requires a strict policy information subset",
            )
        })?;
        if self.request.spec.evaluation.as_deref() != Some("cross_fitted_learning") {
            return Err(policy_compiler_refusal(
                "a cross-fit candidate requires cross-fitted learning",
            ));
        }
        let report = self.execute(ctx)?;
        if report.evaluation != "cross_fitted_learning"
            || report.evaluation_rows == 0
            || usize::try_from(report.evaluation_rows) != Ok(report.row_index.len())
            || report.row_index.len() != report.policy_actions.len()
            || report.row_index.len() != report.scores.len()
            || !report.value.is_finite()
            || !report.se.is_finite()
            || report.se <= 0.0
        {
            return Err(policy_compiler_refusal(
                "cross-fitted policy score cannot form a candidate interval",
            ));
        }
        let mut writer =
            CanonicalWriter::new("antecedent.cross_fitted_policy_value.calibration_coordinate.v1");
        writer
            .text(&graph.treatment)
            .text(&graph.outcome)
            .word(graph.nodes.len() as u64)
            .word(graph.edges.len() as u64);
        for node in &graph.nodes {
            writer.text(node);
        }
        for (parent, child) in &graph.edges {
            writer.text(parent).text(child);
        }
        for column in &self.request.adjustment {
            writer.text(&column.name);
        }
        for index in information {
            writer.word(index as u64);
        }
        writer
            .word(self.request.spec.folds.unwrap_or(DEFAULT_POLICY_FOLDS) as u64)
            .text(self.request.spec.objective.as_deref().unwrap_or(""))
            .word(self.request.spec.min_propensity.unwrap_or_default().to_bits());
        for action in &report.support {
            writer.word(action.to_bits());
        }
        let graph_schedule = SemanticDigest::from_bytes(writer.finish()).to_hex();
        let assignment = if self.request.spec.known_propensity_columns.is_some() {
            "known_row_assignment"
        } else if self.request.spec.known_propensity.is_some() {
            "known_fixed_assignment"
        } else {
            "estimated_assignment"
        };
        let basis = CalibrationBasisWire {
            key: CalibrationKeyWire {
                validation: "not_applicable".into(),
                query: "cross_fitted_learned_policy_value".into(),
                graph_class: "declared_dag_checked_backdoor".into(),
                structure: "fixed".into(),
                modality: "tabular".into(),
                inference: "frequentist".into(),
                estimator: "graph_narrow_policy_cross_fitted_v3".into(),
                interval_method: "cross_fitted_policy_score_wald_v1".into(),
                se_kind: "cross_fitted_influence_score".into(),
                dependence: "fold_policy_stability_assumed".into(),
                posterior: String::new(),
                functional: format!(
                    "empirical_frame_fold_policy_average_value.{assignment}.{graph_schedule}"
                ),
                level: LEVEL,
                identification: "point_conditional_on_declared_dag".into(),
            },
            scope: CalibrationScopeWire {
                row_count: report.evaluation_rows,
                replicates_ok: None,
                posterior_draws: None,
                unidentified_mass: 0.0,
            },
        };
        let slot = calibration_slot(&basis);
        let interval = CrossFittedPolicyValueInterval {
            lower: report.value - Z * report.se,
            upper: report.value + Z * report.se,
            level: LEVEL,
            calibration: slot.status.as_str().into(),
        };
        Ok(CrossFittedPolicyValueCandidate { report, interval, basis, slot })
    }

    /// Form a separately keyed, unmeasured m-out-of-n subsampling candidate.
    /// Each sample contains distinct original rows, and the compiler refits the
    /// learner, nuisance models and folds for every sample. The basic interval
    /// scales the subsample deviations by sqrt(m/(n-m)), the scaling of the
    /// delete-(n-m) jackknife histogram (Wu, 1990): a subsample drawn without
    /// replacement shares its rows with the full sample, so a mean-like
    /// estimate deviates from the full-sample one with variance
    /// sigma^2 (1/m - 1/n), not sigma^2/m. The factor tends to the m-out-of-n
    /// rate sqrt(m/n) as m/n -> 0. The deviation quantiles sit at the
    /// plotting positions (B + 1) p of the B replicates (see
    /// [`plotting_position_quantile`]). The interval targets the original
    /// empirical-frame fold-policy average: the population value of each
    /// realized fold policy, weighted by its fold's scored rows, not one final
    /// policy. Its validity depends on an appropriate subsampling limit and
    /// scaling; near decision ties or for unstable learners those conditions
    /// may fail. No stability or coverage is inferred from successful execution.
    ///
    /// # Errors
    /// Invalid schedule, missing cells, changed seed, cancellation, or any
    /// replicate compiler or estimator refusal. Failed replicates are not dropped.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the replicate count is validated to at most 4096 and the drawn index is below the row count"
    )]
    fn cross_fitted_subsampling_candidate(
        &self,
        subsample_rows: usize,
        replicates: usize,
        outer_identity: SemanticDigest,
        legacy_row_count_key: bool,
        ctx: &ExecutionContext,
    ) -> Result<CrossFittedPolicyValueCandidate, CausalError> {
        let n = self.request.outcome.len();
        if subsample_rows < 20
            || subsample_rows >= n
            || !(100..=4096).contains(&replicates)
            || n.checked_mul(replicates).is_none_or(|work| work > 50_000_000)
            || self.request.outcome.iter().chain(&self.request.treatment).any(|x| !x.is_finite())
            || self.request.adjustment.iter().flat_map(|c| &c.values).any(|x| !x.is_finite())
            || self.request.spec.known_propensity_columns.as_ref().is_some_and(|columns| {
                columns
                    .iter()
                    .any(|column| column.len() != n || column.iter().any(|x| !x.is_finite()))
            })
        {
            return Err(policy_compiler_refusal(
                "invalid complete-row policy subsampling schedule",
            ));
        }
        let mut candidate = self.cross_fitted_value_candidate(ctx)?;
        let scoped = ctx.enter_lowered_dispatch(outer_identity);
        let mut rng = scoped.rng.stream(0x504F_4C53_5542_5341);
        let mut deviations = Vec::with_capacity(replicates);
        for _ in 0..replicates {
            if scoped.cancellation.is_cancelled() {
                return Err(policy_compiler_refusal("policy subsampling was cancelled"));
            }
            let mut indices: Vec<usize> = (0..n).collect();
            for i in 0..subsample_rows {
                let bound = (n - i) as u64;
                let threshold = bound.wrapping_neg() % bound;
                let draw = loop {
                    let value = rng.next_u64();
                    if value >= threshold {
                        break (value % bound) as usize;
                    }
                };
                indices.swap(i, i + draw);
            }
            indices.truncate(subsample_rows);
            indices.sort_unstable();
            let mut sample = self.request.clone();
            sample.outcome = indices.iter().map(|&i| self.request.outcome[i]).collect();
            sample.treatment = indices.iter().map(|&i| self.request.treatment[i]).collect();
            for (column, original) in sample.adjustment.iter_mut().zip(&self.request.adjustment) {
                column.values = indices.iter().map(|&i| original.values[i]).collect();
            }
            if let Some(columns) = &self.request.spec.known_propensity_columns {
                sample.spec.known_propensity_columns = Some(
                    columns
                        .iter()
                        .map(|column| indices.iter().map(|&i| column[i]).collect())
                        .collect(),
                );
            }
            let report = compile_observational_policy(sample, ctx)?.execute(ctx)?;
            if report.evaluation_rows != subsample_rows as u64 || !report.value.is_finite() {
                return Err(policy_compiler_refusal(
                    "policy subsampling replicate lost rows or value",
                ));
            }
            deviations.push(report.value - candidate.report.value);
        }
        deviations.sort_by(f64::total_cmp);
        let scale = (subsample_rows as f64 / (n - subsample_rows) as f64).sqrt();
        let lower = candidate.report.value - scale * plotting_position_quantile(&deviations, 0.975);
        let upper = candidate.report.value - scale * plotting_position_quantile(&deviations, 0.025);
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return Err(policy_compiler_refusal(
                "policy subsampling interval is nonfinite or reversed",
            ));
        }
        candidate.basis.key.interval_method = "cross_fitted_policy_subsampling_basic_v2".into();
        candidate.basis.key.se_kind = "full_refit_subsample_distribution".into();
        candidate.basis.key.dependence = "iid_rows_subsampling_limit_assumed".into();
        let schedule = if legacy_row_count_key {
            format!("subsample_{subsample_rows}_of_{n}")
        } else {
            // The construction fixes m/n. The scope and compiled plan retain the row counts.
            let (mut divisor, mut remainder) = (n, subsample_rows);
            while remainder != 0 {
                (divisor, remainder) = (remainder, divisor % remainder);
            }
            format!("subsample_fraction_{}_of_{}", subsample_rows / divisor, n / divisor)
        };
        candidate.basis.key.functional =
            format!("{}.{schedule}.replicates_{replicates}", candidate.basis.key.functional);
        candidate.basis.scope.replicates_ok = Some(replicates as u32);
        candidate.slot = calibration_slot(&candidate.basis);
        candidate.interval = CrossFittedPolicyValueInterval {
            lower,
            upper,
            level: 0.95,
            calibration: candidate.slot.status.as_str().into(),
        };
        Ok(candidate)
    }

    /// Seal a report with the complete request for independent compiler replay.
    ///
    /// # Errors
    /// A report from a different program or an artifact over the byte bound.
    pub fn export(
        &self,
        report: &ObservationalPolicyReportWire,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, CausalError> {
        if self.execute(ctx)? != *report {
            return Err(policy_compiler_refusal(
                "the policy report differs from its compiled execution",
            ));
        }
        let report_bytes = antecedent_io::to_cbor(report)?;
        let checked = self.request.spec.causal_graph.is_some();
        let narrow = policy_feature_indices(&self.request).is_some();
        let wire = CompiledPolicyArtifactWire {
            version: if narrow {
                3
            } else if checked {
                2
            } else {
                1
            },
            required_feature: if narrow {
                NARROW_POLICY_ARTIFACT_FEATURE
            } else if checked {
                GRAPH_POLICY_ARTIFACT_FEATURE
            } else {
                POLICY_ARTIFACT_FEATURE
            }
            .into(),
            request: PolicyRequestWire::from_request(&self.request),
            seed: self.seed,
            lowered_program_identity: *self.lowered.identity().as_bytes(),
            report_fingerprint: CanonicalWriter::digest_bytes(
                "antecedent.observational_policy_report.v1",
                &report_bytes,
            ),
        };
        let bytes = antecedent_io::to_cbor(&wire)?;
        if bytes.len() > MAX_POLICY_ARTIFACT_BYTES {
            return Err(policy_compiler_refusal("compiled policy artifact exceeds its byte limit"));
        }
        Ok(bytes)
    }
}

/// Compile a policy learning and scoring request before fitting. The source
/// digest covers every row, adjustment column, split declaration and RNG seed;
/// the statistical layer retains the fold schedule and nuisance families.
///
/// # Errors
/// An incomplete or same-row learning declaration, malformed input shape or
/// physical operator refusal.
#[allow(clippy::too_many_lines, reason = "one linear sequence of steps that reads top to bottom")]
pub fn compile_observational_policy(
    request: ObservationalPolicyRequest,
    ctx: &ExecutionContext,
) -> Result<CompiledObservationalPolicy, CausalError> {
    let (support, objective, evaluation) =
        ObservationalPolicyValue::policy_declarations(&request.spec)?;
    if matches!(
        evaluation,
        PolicyEvaluation::Prespecified { origin: PolicyOrigin::LearnedOnEvaluationRows, .. }
    ) {
        return Err(policy_compiler_refusal(
            "a policy learned on scoring rows needs held-out or cross-fitted evaluation",
        ));
    }
    let rows = request.outcome.len();
    if rows == 0
        || request.treatment.len() != rows
        || request
            .adjustment
            .iter()
            .any(|column| column.name.trim().is_empty() || column.values.len() != rows)
    {
        return Err(policy_compiler_refusal(
            "the policy request needs aligned outcome, treatment and adjustment rows",
        ));
    }
    let mut names: Vec<&str> =
        request.adjustment.iter().map(|column| column.name.as_str()).collect();
    names.sort_unstable();
    if names.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(policy_compiler_refusal("policy adjustment column names must be distinct"));
    }
    if let Some(graph) = &request.spec.causal_graph {
        check_policy_graph(&request, graph)?;
    }
    let seed = ctx.rng.master_seed();
    let digest = policy_request_digest(&request, seed)?;
    let folds = request.spec.folds.unwrap_or(DEFAULT_POLICY_FOLDS);
    if folds < 2 {
        return Err(policy_compiler_refusal("policy cross-fitting requires at least two folds"));
    }
    let evaluation_name = request.spec.evaluation.as_deref().unwrap_or("invalid");
    let design = DesignShape::quoted(rows as u64, request.adjustment.len() as u64 + 1, true);
    let scientific = ScientificProblem {
        runtime_query: None,
        identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::PolicyValue,
            outcome: None,
            query: digest,
        },
        plan_id: Arc::from(format!("observational_policy.{evaluation_name}")),
        data: DataClassification::Tabular,
        graph_class: if request.spec.causal_graph.is_some() {
            "declared_dag_checked_backdoor"
        } else {
            "declared_adjustment"
        },
        query_variables: Arc::from(Vec::new()),
        source_digest: Some(digest),
        support_coordinate: None,
        supported: false,
        support_status: None,
    };
    let identification =
        IdentificationLayer::NonGraphical(Arc::new(if request.spec.causal_graph.is_some() {
            NonGraphicalIdentification::GraphCheckedPolicyEvaluation { request: digest }
        } else {
            NonGraphicalIdentification::DeclaredPolicyEvaluation { request: digest }
        }));
    let model = |role, name: String| ModelSlot {
        role,
        family: Arc::from(name),
        inputs: Arc::from(Vec::new()),
        output: None,
    };
    let mut models = vec![model(
        ModelRole::Outcome,
        format!("cross_fitted_per_action_outcome.{}", support.len()),
    )];
    models.push(model(
        ModelRole::Propensity,
        if request.spec.known_propensity.is_some()
            || request.spec.known_propensity_columns.is_some()
        {
            "declared_assignment_probabilities".into()
        } else {
            "cross_fitted_multinomial_logit".into()
        },
    ));
    if !matches!(evaluation, PolicyEvaluation::Prespecified { .. }) {
        models.push(model(
            ModelRole::Nuisance,
            format!("policy_learning.{evaluation_name}.{}", objective.as_str()),
        ));
    }
    let score_name = ScoreModel::new("observational_policy.cross_fitted_dr")
        .map_err(|error| policy_compiler_refusal(&error.to_string()))?;
    let statistical = StatisticalProgram {
        identifier: None,
        estimator: None,
        realization: Realization::Frequentist(ScoreFrequentist::new(score_name)),
        models: Arc::from(models),
        folds: Some(FoldPlan { folds, unit: ResampleUnit::Row, embargo: None }),
        resampling: None,
        posterior: None,
        design,
        operators: Arc::from(Vec::new()),
    };
    let evidence = EvidenceDecisionProgram {
        sources: Arc::from(vec![EvidenceSourceRef {
            role: EvidenceRole::Observations,
            digest: Some(digest),
        }]),
        composition: Composition::Chained { stages: 3 },
        decision: None,
        planning: None,
        chain: None,
    };
    let estimate = (rows as u64)
        .saturating_mul((request.adjustment.len() as u64).saturating_add(support.len() as u64 + 4))
        .saturating_mul(8);
    let physical = PhysicalLayer {
        operators: Arc::from([
            PhysicalOperator::declare(PhysicalEffect::Select)
                .estimates(ByteEstimates::absent().with_output(estimate))
                .build()
                .map_err(|error| policy_compiler_refusal(&error.to_string()))?,
            PhysicalOperator::declare(PhysicalEffect::Fit)
                .estimates(ByteEstimates::absent().with_output(estimate))
                .build()
                .map_err(|error| policy_compiler_refusal(&error.to_string()))?,
            PhysicalOperator::declare(PhysicalEffect::Condition)
                .estimates(ByteEstimates::absent().with_output(estimate))
                .build()
                .map_err(|error| policy_compiler_refusal(&error.to_string()))?,
        ]),
        design,
    };
    let claim = ClaimDemand {
        uncertainty: UncertaintyReading::Sampling,
        kind: Some(ClaimKind::PolicyValue),
    };
    let lowered = Arc::new(
        StandaloneCompositionLoweredParts {
            cell: ProgramCell::ObservationalPolicy,
            scientific,
            identification,
            statistical,
            evidence,
            physical,
            claim,
        }
        .finish(),
    );
    Ok(CompiledObservationalPolicy { request, seed, lowered })
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PolicyRequestWire {
    outcome: Vec<u64>,
    treatment: Vec<u64>,
    adjustment: Vec<(String, Vec<u64>)>,
    spec: ObservationalPolicySpecWire,
}

impl PolicyRequestWire {
    fn from_request(request: &ObservationalPolicyRequest) -> Self {
        Self {
            outcome: request.outcome.iter().map(|value| value.to_bits()).collect(),
            treatment: request.treatment.iter().map(|value| value.to_bits()).collect(),
            adjustment: request
                .adjustment
                .iter()
                .map(|column| {
                    (
                        column.name.clone(),
                        column.values.iter().map(|value| value.to_bits()).collect(),
                    )
                })
                .collect(),
            spec: request.spec.clone(),
        }
    }

    fn into_request(self) -> ObservationalPolicyRequest {
        ObservationalPolicyRequest {
            outcome: self.outcome.into_iter().map(f64::from_bits).collect(),
            treatment: self.treatment.into_iter().map(f64::from_bits).collect(),
            adjustment: self
                .adjustment
                .into_iter()
                .map(|(name, values)| AdjustmentColumn {
                    name,
                    values: values.into_iter().map(f64::from_bits).collect(),
                })
                .collect(),
            spec: self.spec,
        }
    }
}

// The capability unit whose support the policy subsampling routes run under; a release build
// refuses each entry point by its route id (`parity/capability_records.toml`) while the unit is
// not supported, and a development build runs it and records the refusal in its context.
const POLICY_UNIT: UnitId = units_generated::ESTIMATE_OBSERVATIONAL_POLICY_VALUE;
const SUBSAMPLING_COMPILE_ROUTE: &str = "antecedent.policy.compile_policy_subsampling";
const SUBSAMPLING_CONSUME_ROUTE: &str = "antecedent.policy.consume_policy_subsampling";

/// Quantile of ascending-sorted Monte Carlo draws at the plotting position
/// `(B + 1) p`, linearly interpolated between order statistics (Hyndman and
/// Fan type 6). When the `B` draws are exchangeable with the error they stand
/// in for, the order statistics at positions `(B + 1) a` and `(B + 1) b`
/// bracket it with probability `b - a` (Davison and Hinkley, 1997, section
/// 5.2); the type-7 positions `1 + (B - 1) p` bracket it with probability
/// `(b - a) (B - 1) / (B + 1)`, about 0.93 for a 95% interval from 100 draws.
/// Positions outside `[1, B]` read the extreme draw.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the position is clamped to [1, B], so its floor is a positive index within the draws"
)]
fn plotting_position_quantile(sorted: &[f64], p: f64) -> f64 {
    let draws = sorted.len();
    let position = ((draws + 1) as f64 * p).clamp(1.0, draws as f64);
    let below = position.floor();
    let index = below as usize - 1;
    let next = sorted[(index + 1).min(draws - 1)];
    sorted[index] + (position - below) * (next - sorted[index])
}

/// Compiler-owned m-out-of-n full-refit policy subsampling schedule.
#[derive(Clone, Debug)]
pub struct CompiledPolicySubsampling {
    policy: CompiledObservationalPolicy,
    legacy_row_count_key: bool,
    subsample_rows: usize,
    replicates: usize,
    seed: u64,
    lowered: Arc<LoweredProgram>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PolicySubsamplingArtifactWire {
    version: u32,
    feature: String,
    request: PolicyRequestWire,
    subsample_rows: usize,
    replicates: usize,
    seed: u64,
    program_identity: [u8; 32],
    result_fingerprint: [u8; 32],
}

/// Compile the original policy and the full-refit subsampling law before drawing rows.
/// The fixed schedule assumes IID rows and a valid m-out-of-n scaling limit.
/// The returned candidate has no earned coverage merely from compilation.
///
/// # Errors
/// Invalid policy, graph, schedule, input rows, or resource bound.
#[allow(clippy::too_many_lines, reason = "one linear sequence of steps that reads top to bottom")]
pub fn compile_policy_subsampling(
    request: ObservationalPolicyRequest,
    subsample_rows: usize,
    replicates: usize,
    ctx: &ExecutionContext,
) -> Result<CompiledPolicySubsampling, CausalError> {
    require_program(ctx, &SupportKey::Unit(POLICY_UNIT), SUBSAMPLING_COMPILE_ROUTE)?;
    let n = request.outcome.len();
    if subsample_rows < 20
        || subsample_rows >= n
        || !(100..=4096).contains(&replicates)
        || n.checked_mul(replicates).is_none_or(|work| work > 50_000_000)
        || request.spec.evaluation.as_deref() != Some("cross_fitted_learning")
        || request.spec.causal_graph.is_none()
        || policy_feature_indices(&request).is_none()
        || request.outcome.iter().chain(&request.treatment).any(|x| !x.is_finite())
        || request.adjustment.iter().flat_map(|c| &c.values).any(|x| !x.is_finite())
        || request.spec.known_propensity_columns.as_ref().is_some_and(|columns| {
            columns.iter().any(|column| column.len() != n || column.iter().any(|x| !x.is_finite()))
        })
    {
        return Err(policy_compiler_refusal("invalid complete-row policy subsampling declaration"));
    }
    if antecedent_io::to_cbor(&PolicyRequestWire::from_request(&request))?.len() > 80 * 1024 * 1024
    {
        return Err(policy_compiler_refusal("policy subsampling source exceeds its byte bound"));
    }
    let policy = compile_observational_policy(request, ctx)?;
    let seed = ctx.rng.master_seed();
    let mut writer = CanonicalWriter::new("antecedent.policy_subsampling.schedule.v1");
    writer
        .bytes(policy.lowered().identity().as_bytes())
        .word(seed)
        .word(subsample_rows as u64)
        .word(replicates as u64);
    let digest = writer.finish();
    let design = DesignShape::quoted(n as u64, policy.request.adjustment.len() as u64 + 1, true);
    let scientific = ScientificProblem {
        runtime_query: None,
        identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::PolicyValue,
            outcome: None,
            query: digest,
        },
        plan_id: Arc::from("cross_fitted_policy_subsampling.v1"),
        data: DataClassification::Tabular,
        graph_class: "declared_dag_checked_backdoor",
        query_variables: Arc::from([]),
        source_digest: Some(digest),
        support_coordinate: None,
        supported: false,
        support_status: None,
    };
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SharedEvidenceContract { contract: digest },
    ));
    let score = ScoreModel::new("cross_fitted_policy_full_refit_subsampling")
        .map_err(|error| policy_compiler_refusal(&error.to_string()))?;
    let statistical = StatisticalProgram {
        identifier: None,
        estimator: None,
        realization: Realization::Frequentist(ScoreFrequentist::new(score)),
        models: Arc::from([]),
        folds: None,
        resampling: None,
        posterior: None,
        design,
        operators: Arc::from([]),
    };
    let evidence = EvidenceDecisionProgram {
        sources: Arc::from([EvidenceSourceRef {
            role: EvidenceRole::Observations,
            digest: Some(*policy.lowered().identity().as_bytes()),
        }]),
        composition: Composition::Chained { stages: replicates + 1 },
        decision: None,
        planning: None,
        chain: None,
    };
    let physical = PhysicalLayer {
        operators: Arc::from([
            PhysicalOperator::declare(PhysicalEffect::Select)
                .build()
                .map_err(|error| policy_compiler_refusal(&error.to_string()))?,
            PhysicalOperator::declare(PhysicalEffect::Fit)
                .build()
                .map_err(|error| policy_compiler_refusal(&error.to_string()))?,
        ]),
        design,
    };
    let lowered = Arc::new(
        StandaloneCompositionLoweredParts {
            cell: ProgramCell::PolicySubsampling,
            scientific,
            identification,
            statistical,
            evidence,
            physical,
            claim: ClaimDemand {
                uncertainty: UncertaintyReading::Sampling,
                kind: Some(ClaimKind::PolicyValue),
            },
        }
        .finish(),
    );
    Ok(CompiledPolicySubsampling {
        policy,
        legacy_row_count_key: false,
        subsample_rows,
        replicates,
        seed,
        lowered,
    })
}

fn policy_subsampling_fingerprint(
    candidate: &CrossFittedPolicyValueCandidate,
) -> Result<[u8; 32], CausalError> {
    let mut writer = CanonicalWriter::new("antecedent.policy_subsampling.result.v1");
    writer
        .bytes(&antecedent_io::to_cbor(&candidate.report)?)
        .word(candidate.interval.lower.to_bits())
        .word(candidate.interval.upper.to_bits())
        .word(candidate.interval.level.to_bits())
        .text(&candidate.interval.calibration)
        .bytes(&antecedent_io::to_cbor(&candidate.basis)?);
    Ok(writer.finish())
}

impl CompiledPolicySubsampling {
    /// The retained six-layer resampling program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }

    /// Inspect the compiled resampling program.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }

    /// Execute all compiler-refitted subsamples; no coverage is presumed.
    ///
    /// # Errors
    /// Seed mismatch, cancellation, or any nested fitting refusal.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<CrossFittedPolicyValueCandidate, CausalError> {
        gate_program(ctx, self)?;
        if ctx.rng.master_seed() != self.seed {
            return Err(policy_compiler_refusal("policy subsampling execution seed changed"));
        }
        self.policy.cross_fitted_subsampling_candidate(
            self.subsample_rows,
            self.replicates,
            self.lowered.identity(),
            self.legacy_row_count_key,
            ctx,
        )
    }

    /// Seal source rows and a freshly verified result for independent replay.
    ///
    /// # Errors
    /// A changed result, execution refusal or oversized artifact.
    pub fn export(
        &self,
        result: &CrossFittedPolicyValueCandidate,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, CausalError> {
        let fingerprint = policy_subsampling_fingerprint(result)?;
        if fingerprint != policy_subsampling_fingerprint(&self.execute(ctx)?)? {
            return Err(policy_compiler_refusal("policy subsampling result changed"));
        }
        let bytes = antecedent_io::to_cbor(&PolicySubsamplingArtifactWire {
            version: 2,
            feature: "cross_fitted_policy_subsampling_v1".into(),
            request: PolicyRequestWire::from_request(&self.policy.request),
            subsample_rows: self.subsample_rows,
            replicates: self.replicates,
            seed: self.seed,
            program_identity: *self.lowered.identity().as_bytes(),
            result_fingerprint: fingerprint,
        })?;
        if bytes.len() > 80 * 1024 * 1024 {
            return Err(policy_compiler_refusal(
                "policy subsampling artifact exceeds its byte bound",
            ));
        }
        Ok(bytes)
    }
}

/// Independently recompile and rerun the retained policy subsampling schedule.
///
/// # Errors
/// Changed feature, seed, identity, source or result.
pub fn consume_policy_subsampling(
    bytes: &[u8],
    expected: SemanticDigest,
    ctx: &ExecutionContext,
) -> Result<CrossFittedPolicyValueCandidate, CausalError> {
    require_program(ctx, &SupportKey::Unit(POLICY_UNIT), SUBSAMPLING_CONSUME_ROUTE)?;
    if bytes.len() > 80 * 1024 * 1024 {
        return Err(policy_compiler_refusal("policy subsampling artifact exceeds its byte bound"));
    }
    let wire: PolicySubsamplingArtifactWire = antecedent_io::from_cbor(bytes)?;
    if !matches!(wire.version, 1 | 2)
        || wire.feature != "cross_fitted_policy_subsampling_v1"
        || wire.seed != ctx.rng.master_seed()
        || wire.program_identity != *expected.as_bytes()
    {
        return Err(policy_compiler_refusal(
            "policy subsampling feature, seed or identity differs",
        ));
    }
    let mut plan = compile_policy_subsampling(
        wire.request.into_request(),
        wire.subsample_rows,
        wire.replicates,
        ctx,
    )?;
    if plan.lowered().identity() != expected {
        return Err(policy_compiler_refusal("recompiled policy subsampling program differs"));
    }
    plan.legacy_row_count_key = wire.version == 1;
    let candidate = plan.execute(ctx)?;
    if policy_subsampling_fingerprint(&candidate)? != wire.result_fingerprint {
        return Err(policy_compiler_refusal("policy subsampling result differs on replay"));
    }
    Ok(candidate)
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CompiledPolicyArtifactWire {
    version: u32,
    required_feature: String,
    request: PolicyRequestWire,
    seed: u64,
    lowered_program_identity: [u8; 32],
    report_fingerprint: [u8; 32],
}

/// Independently recompile and execute a retained policy request. The caller's
/// context must use the artifact's split seed; every report value is recomputed.
///
/// # Errors
/// A foreign feature, changed compiler program, changed report or estimator refusal.
pub fn consume_compiled_observational_policy(
    bytes: &[u8],
    expected_program: Option<SemanticDigest>,
    ctx: &ExecutionContext,
) -> Result<ObservationalPolicyReportWire, CausalError> {
    if bytes.len() > MAX_POLICY_ARTIFACT_BYTES {
        return Err(policy_compiler_refusal("compiled policy artifact exceeds its byte limit"));
    }
    let wire: CompiledPolicyArtifactWire = antecedent_io::from_cbor(bytes)?;
    let checked = wire.request.spec.causal_graph.is_some();
    let narrow = policy_feature_indices(&wire.request.clone().into_request()).is_some();
    let expected_version = if narrow {
        3
    } else if checked {
        2
    } else {
        1
    };
    let expected_feature = if narrow {
        NARROW_POLICY_ARTIFACT_FEATURE
    } else if checked {
        GRAPH_POLICY_ARTIFACT_FEATURE
    } else {
        POLICY_ARTIFACT_FEATURE
    };
    if wire.version != expected_version
        || wire.required_feature != expected_feature
        || wire.seed != ctx.rng.master_seed()
    {
        return Err(policy_compiler_refusal(
            "compiled policy artifact version, feature or split seed differs",
        ));
    }
    let plan = compile_observational_policy(wire.request.into_request(), ctx)?;
    let identity = plan.lowered().identity();
    if wire.lowered_program_identity != *identity.as_bytes()
        || expected_program.is_some_and(|expected| expected != identity)
    {
        return Err(policy_compiler_refusal("compiled policy program identity differs"));
    }
    let report = plan.execute(ctx)?;
    let fingerprint = CanonicalWriter::digest_bytes(
        "antecedent.observational_policy_report.v1",
        &antecedent_io::to_cbor(&report)?,
    );
    if wire.report_fingerprint != fingerprint {
        return Err(policy_compiler_refusal(
            "compiled policy report differs on independent replay",
        ));
    }
    Ok(report)
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

fn assumptions(evaluation: &PolicyEvaluation, graph_checked: bool) -> Vec<String> {
    let mut list = vec![
        if graph_checked {
            "conditional exchangeability: the declared adjustment set blocks every backdoor path in the declared causally sufficient DAG; graph truth remains an assumption"
        } else {
            "conditional exchangeability: the declared covariates are a sufficient adjustment set (declared, not checked)"
        }.to_owned(),
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

impl crate::analysis::program_handle::CompiledProgramHandle for CompiledObservationalPolicy {
    fn program(&self) -> &std::sync::Arc<crate::planner::lowered::LoweredProgram> {
        &self.lowered
    }
}

impl crate::analysis::program_handle::CompiledProgramHandle for CompiledPolicySubsampling {
    fn program(&self) -> &std::sync::Arc<crate::planner::lowered::LoweredProgram> {
        &self.lowered
    }
}

#[cfg(test)]
mod site_tests {
    use super::*;
    use crate::identity_laws::{
        ADVERSARIAL_TEXT, all_distinct, bit_exact, injective, ordered_laws,
    };

    fn query_of(lowered: &LoweredProgram) -> [u8; 32] {
        match lowered.scientific().target() {
            ProblemTarget::QueryFunctional { query, .. } => *query,
            _ => panic!("the program target is a query functional"),
        }
    }

    fn request(outcome: &[f64], columns: &[(String, Vec<f64>)]) -> ObservationalPolicyRequest {
        ObservationalPolicyRequest {
            outcome: outcome.to_vec(),
            treatment: vec![0.0; outcome.len()],
            adjustment: columns
                .iter()
                .map(|(name, values)| AdjustmentColumn {
                    name: name.clone(),
                    values: values.clone(),
                })
                .collect(),
            spec: ObservationalPolicySpecWire::default(),
        }
    }

    fn digest(request: &ObservationalPolicyRequest) -> Option<[u8; 32]> {
        policy_request_digest(request, 7).ok()
    }

    #[test]
    fn site_observational_policy_request_digest_orders_rows_and_frames_columns() {
        // Rows are positional (row-aligned outcome, treatment and columns; the fold and
        // held-out schedules index them), and so are the columns of the design matrices.
        ordered_laws(&[1.0, 2.0, 3.0], |rows: &[f64]| digest(&request(rows, &[])));
        let columns = vec![
            ("x".to_owned(), vec![1.0]),
            ("w".to_owned(), vec![2.0]),
            ("v".to_owned(), vec![3.0]),
        ];
        ordered_laws(&columns, |columns: &[(String, Vec<f64>)]| digest(&request(&[1.0], columns)));
        injective(
            &ADVERSARIAL_TEXT,
            |parts| *parts,
            |(first, second)| {
                digest(&request(
                    &[1.0],
                    &[((*first).to_owned(), vec![1.0]), ((*second).to_owned(), vec![1.0])],
                ))
            },
        );
        // A value moved across a column boundary is another request.
        assert_ne!(
            digest(&request(&[1.0], &[("x".into(), vec![1.0, 2.0]), ("w".into(), vec![3.0])])),
            digest(&request(&[1.0], &[("x".into(), vec![1.0]), ("w".into(), vec![2.0, 3.0])]))
        );
        // Rows moved from the outcome to the treatment column are another request: each column
        // is count-prefixed (without the prefix these two write the same words).
        let split = |outcome: Vec<f64>, treatment: Vec<f64>| {
            let mut declared = request(&[], &[]);
            declared.outcome = outcome;
            declared.treatment = treatment;
            digest(&declared)
        };
        assert_ne!(split(vec![1.0, f64::from_bits(1)], Vec::new()), split(vec![1.0], vec![0.0]));
        bit_exact(|value| digest(&request(&[value], &[])));
        let base = request(&[1.0, 2.0], &[]);
        let mut treated = base.clone();
        treated.treatment[0] = 1.0;
        let mut declared = base.clone();
        declared.spec.support = Some(vec![0.0, 1.0]);
        all_distinct(&[
            policy_request_digest(&base, 7).ok(),
            policy_request_digest(&base, 8).ok(),
            digest(&treated),
            digest(&declared),
        ]);
    }

    fn graph_request(shift: f64) -> ObservationalPolicyRequest {
        let (mut outcome, mut treatment, mut x, mut w) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for i in 0_u32..80 {
            let xi = if i % 2 == 0 { -1.0 } else { 1.0 };
            let wi = if (i / 2) % 2 == 0 { -1.0 } else { 1.0 };
            let action = if (i / 4) % 2 == 0 { 0.0 } else { 1.0 };
            outcome.push(shift + 0.3 * xi + 0.1 * wi + 0.2 * action + 0.01 * f64::from(i % 7));
            treatment.push(action);
            x.push(xi);
            w.push(wi);
        }
        ObservationalPolicyRequest {
            outcome,
            treatment,
            adjustment: vec![
                AdjustmentColumn { name: "x".into(), values: x },
                AdjustmentColumn { name: "w".into(), values: w },
            ],
            spec: ObservationalPolicySpecWire {
                support: Some(vec![0.0, 1.0]),
                objective: Some("maximize".into()),
                evaluation: Some("cross_fitted_learning".into()),
                folds: Some(3),
                known_propensity_columns: Some(vec![vec![0.5; 80], vec![0.5; 80]]),
                causal_graph: Some(PolicyGraphDeclaration {
                    nodes: vec!["x".into(), "w".into(), "a".into(), "y".into()],
                    edges: vec![
                        ("x".into(), "a".into()),
                        ("w".into(), "a".into()),
                        ("x".into(), "y".into()),
                        ("w".into(), "y".into()),
                        ("a".into(), "y".into()),
                    ],
                    treatment: "a".into(),
                    outcome: "y".into(),
                    information_set: vec!["x".into()],
                }),
                seed: 5,
                ..ObservationalPolicySpecWire::default()
            },
        }
    }

    #[test]
    fn site_observational_policy_subsampling_binds_policy_and_schedule() {
        let query = |shift: f64, rows: usize, replicates: usize, seed: u64| {
            let ctx = ExecutionContext::for_tests(seed);
            let plan = compile_policy_subsampling(graph_request(shift), rows, replicates, &ctx)
                .expect("a valid subsampling schedule");
            query_of(plan.lowered())
        };
        all_distinct(&[
            query(1.0, 40, 100, 5),
            query(1.5, 40, 100, 5),
            query(1.0, 41, 100, 5),
            query(1.0, 40, 101, 5),
            query(1.0, 40, 100, 6),
        ]);
    }

    fn candidate() -> CrossFittedPolicyValueCandidate {
        CrossFittedPolicyValueCandidate {
            report: ObservationalPolicyReportWire {
                evaluation: "cross_fitted_learning".into(),
                estimand: "fold-policy average".into(),
                assumptions: vec!["declared".into()],
                objective: "maximize".into(),
                support: vec![0.0, 1.0],
                value: 1.0,
                se: 0.1,
                action_values: Vec::new(),
                reference: None,
                propensity_source: "known".into(),
                evaluation_rows: 3,
                learning_rows: 3,
                rows_dropped_missing: 0,
                row_index: vec![0, 1, 2],
                policy_actions: vec![0.0, 1.0, 1.0],
                scores: vec![1.0, 2.0, 3.0],
                calibration: "not_measured".into(),
            },
            interval: CrossFittedPolicyValueInterval {
                lower: 0.8,
                upper: 1.2,
                level: 0.95,
                calibration: "not_measured".into(),
            },
            basis: CalibrationBasisWire {
                key: CalibrationKeyWire {
                    validation: String::new(),
                    query: "q".into(),
                    graph_class: "g".into(),
                    structure: "fixed".into(),
                    modality: "tabular".into(),
                    inference: "frequentist".into(),
                    estimator: "e".into(),
                    interval_method: "m".into(),
                    se_kind: "s".into(),
                    dependence: "iid".into(),
                    posterior: String::new(),
                    functional: "f".into(),
                    level: 0.95,
                    identification: "point".into(),
                },
                scope: CalibrationScopeWire {
                    row_count: 3,
                    replicates_ok: Some(100),
                    posterior_draws: None,
                    unidentified_mass: 0.0,
                },
            },
            slot: CalibrationSlotWire::not_measured("site test"),
        }
    }

    #[test]
    fn site_observational_policy_subsampling_fingerprint_reads_every_result_field() {
        let fingerprint = |candidate: &CrossFittedPolicyValueCandidate| {
            policy_subsampling_fingerprint(candidate).expect("a result fingerprint")
        };
        let base = candidate();
        let mut variants = vec![base.clone()];
        let mut changed = base.clone();
        changed.report.scores.swap(0, 2);
        variants.push(changed);
        let mut changed = base.clone();
        changed.interval.lower = 0.7;
        variants.push(changed);
        let mut changed = base.clone();
        changed.interval.upper = 1.3;
        variants.push(changed);
        let mut changed = base.clone();
        changed.interval.level = 0.9;
        variants.push(changed);
        let mut changed = base.clone();
        changed.interval.calibration = "measured".into();
        variants.push(changed);
        let mut changed = base.clone();
        changed.basis.scope.row_count = 4;
        variants.push(changed);
        all_distinct(&variants.iter().map(fingerprint).collect::<Vec<_>>());
        bit_exact(|lower| {
            let mut changed = base.clone();
            changed.interval.lower = lower;
            policy_subsampling_fingerprint(&changed).ok()
        });
    }
}
