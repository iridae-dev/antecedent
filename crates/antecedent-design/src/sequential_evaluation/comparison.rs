//! Caller-supplied comparisons: act now, wait, collect evidence, or choose
//! differently after an observation.
//!
//! Each comparison is a fixed policy evaluated by the sequential g-formula.
//! The value of *collecting additional evidence* is not re-derived here: it is
//! scored by the existing design ranker (`DesignRanker` with the
//! `ReduceDecisionRegret` objective, whose exact preposterior engine is the
//! value-of-sample-information surface) over the finite measurement channel,
//! and checked against the enumerated value of the supplied collect-then-act
//! policy. A disagreement is refused, not reported.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{CausalRng, ExecutionContext};
use antecedent_prob::{GraphIdentFlag, WeightedGraphSamples};

use super::belief::{branches, predict};
use super::gformula::{EvaluationLimits, PolicyEvaluation, commit_tail, walk};
use super::model::{ActionRole, Compiled, dot};
use super::policy::{ContingentPolicy, HistoryKey, PolicySpec};
use super::{
    FiniteSequentialModel, SEQUENTIAL_EVALUATION_INFERENCE, SequentialRequest, SequentialResult,
    cancelled,
};
use crate::candidate::{CandidateDesign, DesignCost, SamplingPlan};
use crate::decision::{DecisionProblem, DecisionProblemId, Utility};
use crate::error::DesignError;
use crate::objective::DesignObjective;
use crate::preposterior::{DecisionPrior, DecisionSignal};
use crate::ranker::{DecisionRegistry, DesignEvaluationContext, DesignRankConfig, DesignRanker};
use crate::result::ScoreEvaluation;

/// What a comparison asks of its policy's first decision.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum ComparisonKind {
    /// Commit at the first decision time.
    ActNow,
    /// Continue without collecting anything and decide later.
    Wait,
    /// Collect an observation, then decide.
    CollectEvidence,
    /// Collect an observation and take different actions after different
    /// observations.
    ChooseAfterObservation,
}

impl ComparisonKind {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ActNow => "act_now",
            Self::Wait => "wait",
            Self::CollectEvidence => "collect_evidence",
            Self::ChooseAfterObservation => "choose_after_observation",
        }
    }

    /// Parse a stable name.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        [Self::ActNow, Self::Wait, Self::CollectEvidence, Self::ChooseAfterObservation]
            .into_iter()
            .find(|kind| kind.as_str() == name)
    }
}

/// One named comparison.
#[derive(Clone, Debug, PartialEq)]
pub struct ComparisonSpec {
    /// Unique label.
    pub label: String,
    /// What the comparison is.
    pub kind: ComparisonKind,
    /// The policy offered; only a fixed contingent policy is evaluated.
    pub policy: PolicySpec,
}

/// A comparison's evaluated policy.
#[derive(Clone, Debug, PartialEq)]
pub struct ComparisonRow {
    /// Comparison label.
    pub label: String,
    /// Comparison kind.
    pub kind: ComparisonKind,
    /// Policy version label.
    pub policy_version: String,
    /// Policy identity (hex).
    pub policy_identity: String,
    /// Value of the policy.
    pub value: f64,
    /// Value minus the first comparison's value.
    pub difference_from_first: f64,
    /// The full enumeration, with history and decision receipts.
    pub evaluation: PolicyEvaluation,
}

/// Which action collects evidence, and which commit actions follow it, for
/// the value-of-information cross-check.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VoiRequest {
    /// The continue action whose observation is collected at the first
    /// decision time. It must be pure information: no immediate or delayed
    /// utility.
    pub collect_action: usize,
    /// The commit actions available once the observation is in.
    pub commit_actions: Vec<usize>,
}

/// The value of collecting evidence, scored by the design ranker and checked
/// against the enumeration.
#[derive(Clone, Debug, PartialEq)]
pub struct VoiCheck {
    /// The action that collects the observation.
    pub collect_action: usize,
    /// Best expected utility of committing without the observation, under the
    /// belief after the collect action's transition.
    pub prior_value: f64,
    /// Expected value of the sample information, from the design ranker.
    pub expected_value_of_sample_information: f64,
    /// Utility cost of collecting.
    pub cost: f64,
    /// `evsi - cost`.
    pub net_value_of_collecting: f64,
    /// `prior_value + evsi - cost`.
    pub predicted_policy_value: f64,
    /// The g-formula value of collecting and then taking the Bayes action.
    pub enumerated_policy_value: f64,
    /// How the ranker scored it: `exact` or `monte_carlo`.
    pub ranking_evaluation: &'static str,
    /// The functional the ranker implemented.
    pub implemented_functional: String,
    /// The collect-then-Bayes-commit policy whose value was enumerated.
    pub bayes_after_observation: ContingentPolicy,
}

/// The evaluated comparisons.
#[derive(Clone, Debug, PartialEq)]
pub struct ComparisonReport {
    /// One row per comparison, in the order supplied.
    pub rows: Vec<ComparisonRow>,
    /// Index of the highest-valued comparison (the first among ties). A
    /// ranking of the supplied comparisons, not an optimum over policies.
    pub best: usize,
    /// The ranker-scored value of collecting evidence, when requested.
    pub value_of_information: Option<VoiCheck>,
    /// Model identity (hex).
    pub model_identity: String,
    /// The inferential claim: always `point_only`.
    pub inference: &'static str,
}

fn mismatch(label: &str, kind: ComparisonKind, why: &str) -> super::SequentialEvaluationError {
    refusal!(
        "invalid_argument",
        "sequential_evaluation.comparison_kind_mismatch",
        format!("comparison `{label}` is declared `{}` but its policy {why}", kind.as_str()),
        "declare the kind the policy implements, or supply a policy that implements the declared kind",
    )
}

fn check_kind(
    compiled: &Compiled,
    label: &str,
    kind: ComparisonKind,
    policy: &ContingentPolicy,
) -> SequentialResult<()> {
    let root = HistoryKey::root();
    let Some(first) = policy.distribution(&root) else {
        return Err(mismatch(label, kind, "declares no decision at the first decision time"));
    };
    let actions: Vec<_> = first.iter().filter_map(|(a, _)| compiled.actions.get(*a)).collect();
    if actions.len() != first.len() {
        return Err(mismatch(label, kind, "names an undeclared action at the first decision time"));
    }
    match kind {
        ComparisonKind::ActNow => {
            if actions.iter().any(|a| a.role != ActionRole::Commit) {
                return Err(mismatch(label, kind, "does not commit at the first decision time"));
            }
        }
        ComparisonKind::Wait => {
            if actions.iter().any(|a| a.role != ActionRole::Continue || a.has_channel) {
                return Err(mismatch(
                    label,
                    kind,
                    "does not continue without an observation at the first decision time",
                ));
            }
        }
        ComparisonKind::CollectEvidence | ComparisonKind::ChooseAfterObservation => {
            if actions.iter().any(|a| a.role != ActionRole::Continue || !a.has_channel) {
                return Err(mismatch(
                    label,
                    kind,
                    "does not collect an observation at the first decision time",
                ));
            }
        }
    }
    if kind == ComparisonKind::ChooseAfterObservation {
        let (action, _) = first[0];
        let symbols = compiled.actions[action].symbols;
        let mut seen: Vec<&[(usize, f64)]> = Vec::new();
        for symbol in 0..symbols {
            if let Some(next) = policy.distribution(&root.extended(action, symbol)) {
                if !seen.contains(&next) {
                    seen.push(next);
                }
            }
        }
        if seen.len() < 2 {
            return Err(mismatch(
                label,
                kind,
                "takes the same decision after every observation it declares",
            ));
        }
    }
    Ok(())
}

/// Evaluate caller-supplied comparisons on one model.
///
/// # Errors
/// A transported request without its contract; a static score table or a
/// policy learned from the data under evaluation; a policy that does not
/// implement its declared kind; any evaluation refusal; a value-of-information
/// request that the ranker and the enumeration disagree on.
pub fn compare_policies(
    model: &FiniteSequentialModel,
    specs: Vec<ComparisonSpec>,
    voi: Option<&VoiRequest>,
    request: &SequentialRequest,
    limits: EvaluationLimits,
    ctx: &ExecutionContext,
) -> SequentialResult<ComparisonReport> {
    request.check(model)?;
    if specs.is_empty() {
        return Err(refusal!(
            "invalid_argument",
            "sequential_evaluation.comparisons",
            "no comparison was supplied",
            "supply at least one named policy comparison",
        ));
    }
    let compiled = model.compile()?;
    let mut rows: Vec<ComparisonRow> = Vec::with_capacity(specs.len());
    for spec in specs {
        if rows.iter().any(|row| row.label == spec.label) {
            return Err(refusal!(
                "invalid_argument",
                "sequential_evaluation.comparison_label",
                format!("comparison label `{}` is not unique", spec.label),
                "give every comparison a distinct label",
            ));
        }
        let policy = spec.policy.into_contingent()?;
        check_kind(&compiled, &spec.label, spec.kind, &policy)?;
        let evaluation = walk(&compiled, &policy, None, limits, ctx)?;
        rows.push(ComparisonRow {
            label: spec.label,
            kind: spec.kind,
            policy_version: evaluation.policy_version.clone(),
            policy_identity: evaluation.policy_identity.clone(),
            value: evaluation.value,
            difference_from_first: 0.0,
            evaluation,
        });
    }
    let first = rows[0].value;
    for row in &mut rows {
        row.difference_from_first = row.value - first;
    }
    let mut best = 0;
    for (index, row) in rows.iter().enumerate() {
        if row.value > rows[best].value {
            best = index;
        }
    }
    let value_of_information =
        voi.map(|request| value_of_collecting(&compiled, request, limits, ctx)).transpose()?;
    Ok(ComparisonReport {
        rows,
        best,
        value_of_information,
        model_identity: compiled.identity.clone(),
        inference: SEQUENTIAL_EVALUATION_INFERENCE,
    })
}

/// A finite signal: one observation of a declared channel.
struct FiniteChannelSignal {
    likelihood: Vec<f64>,
    symbols: usize,
    states: usize,
}

impl FiniteChannelSignal {
    fn symbol_of(&self, statistic: f64) -> Option<usize> {
        (0..self.symbols).find(|&symbol| (symbol as f64 - statistic).abs() < 0.5)
    }
}

impl DecisionSignal<usize> for FiniteChannelSignal {
    fn name(&self) -> &'static str {
        "finite_measurement_channel"
    }

    fn finite_support(&self, n: u64) -> Option<Vec<f64>> {
        (n == 1).then(|| (0..self.symbols).map(|symbol| symbol as f64).collect())
    }

    fn log_likelihood(
        &self,
        statistic: f64,
        n: u64,
        states: &[usize],
        out: &mut [f64],
    ) -> Result<(), DesignError> {
        let symbol = self.symbol_of(statistic).filter(|_| n == 1);
        for (slot, &state) in out.iter_mut().zip(states) {
            *slot = match symbol {
                Some(symbol) if state < self.states => {
                    self.likelihood[symbol * self.states + state].ln()
                }
                _ => f64::NEG_INFINITY,
            };
        }
        Ok(())
    }

    fn sample_statistic(
        &self,
        state: &usize,
        n: u64,
        rng: &mut CausalRng,
    ) -> Result<f64, DesignError> {
        if n != 1 || *state >= self.states {
            return Err(DesignError::Shape(
                "the finite channel signal draws one observation of a declared state".into(),
            ));
        }
        let u = rng.next_f64();
        let mut cumulative = 0.0;
        for symbol in 0..self.symbols {
            cumulative += self.likelihood[symbol * self.states + state];
            if u < cumulative {
                return Ok(symbol as f64);
            }
        }
        Ok((self.symbols - 1) as f64)
    }
}

/// Terminal utilities already scaled by `states * belief(state)`, so that the
/// ranker's equally weighted draws of the state carry the belief's weights.
struct ScaledUtility {
    table: Vec<Vec<f64>>,
}

impl Utility<usize, usize> for ScaledUtility {
    fn evaluate_batch(
        &self,
        actions: &[usize],
        outcomes: &[usize],
        out: &mut [f64],
    ) -> Result<(), DesignError> {
        for (row, &action) in actions.iter().enumerate() {
            let utilities = self
                .table
                .get(action)
                .ok_or_else(|| DesignError::Shape(format!("no utility for action {action}")))?;
            for (column, &state) in outcomes.iter().enumerate() {
                out[row * outcomes.len() + column] = *utilities
                    .get(state)
                    .ok_or_else(|| DesignError::Shape(format!("no utility for state {state}")))?;
            }
        }
        Ok(())
    }
}

fn design_refusal(error: &DesignError) -> super::SequentialEvaluationError {
    match error {
        DesignError::Cancelled => cancelled(),
        other => refusal!(
            "invalid_argument",
            "sequential_evaluation.voi_ranking",
            format!("the design ranker refused the value-of-information request: {other}"),
            "check the collect action, the commit actions and the model's finite channel",
        ),
    }
}

/// Utility of committing `action` in latent state `state`, with the action's
/// own delayed outcome expected under its transition.
fn commit_utility(compiled: &Compiled, action: usize, state: usize) -> f64 {
    let spec = &compiled.actions[action];
    let mut unit = vec![0.0; compiled.states];
    unit[state] = 1.0;
    spec.immediate[state] - spec.utility_cost + commit_tail(compiled, action, &unit, 0, &[])
}

#[allow(
    clippy::too_many_lines,
    reason = "validate the request, score through the ranker, then enumerate and compare"
)]
fn value_of_collecting(
    compiled: &Compiled,
    request: &VoiRequest,
    limits: EvaluationLimits,
    ctx: &ExecutionContext,
) -> SequentialResult<VoiCheck> {
    let pure_information = |action: usize| {
        compiled.actions.get(action).filter(|spec| {
            spec.role == ActionRole::Continue
                && spec.has_channel
                && spec.delayed.is_none()
                && spec.immediate.iter().all(|u| *u == 0.0)
        })
    };
    let Some(collect) = pure_information(request.collect_action) else {
        return Err(refusal!(
            "invalid_argument",
            "sequential_evaluation.voi_collect_action",
            "the collect action must be a continue action with an observation channel and no immediate or delayed utility",
            "name a pure information-collecting action; evaluate other actions as policies",
        ));
    };
    if compiled.horizon < 2 || !compiled.boundaries.is_empty() || request.commit_actions.is_empty()
    {
        return Err(refusal!(
            "route_not_supported",
            "sequential_evaluation.voi_scope",
            "the value-of-information cross-check covers one collection followed by a commit, with a horizon of at least two decision times, no stopping boundary and at least one commit action",
            "evaluate the collect-then-act policy directly as a comparison",
        ));
    }
    if request
        .commit_actions
        .iter()
        .any(|&a| compiled.actions.get(a).is_none_or(|spec| spec.role != ActionRole::Commit))
    {
        return Err(refusal!(
            "invalid_argument",
            "sequential_evaluation.voi_commit_actions",
            "every follow-up action must be a declared commit action",
            "name declared commit actions",
        ));
    }
    let states = compiled.states;
    let believed = predict(compiled, &compiled.initial, request.collect_action);
    let table: Vec<Vec<f64>> = request
        .commit_actions
        .iter()
        .map(|&action| (0..states).map(|state| commit_utility(compiled, action, state)).collect())
        .collect();
    let prior_value =
        table.iter().map(|utilities| dot(&believed, utilities)).fold(f64::NEG_INFINITY, f64::max);
    let scaled: Vec<Vec<f64>> = table
        .iter()
        .map(|utilities| {
            utilities
                .iter()
                .zip(&believed)
                .map(|(utility, mass)| utility * mass * states as f64)
                .collect()
        })
        .collect();
    let problem = DecisionProblem::new(
        (0..request.commit_actions.len()).collect::<Vec<usize>>(),
        Arc::new(ScaledUtility { table: scaled }),
        Vec::new(),
    );
    let registry = DecisionRegistry {
        problems: vec![Some(problem)],
        prior: DecisionPrior::Draws((0..states).collect::<Vec<usize>>()),
        signal: Arc::new(FiniteChannelSignal {
            likelihood: collect.likelihood.clone(),
            symbols: collect.symbols,
            states,
        }),
    };
    let graphs =
        WeightedGraphSamples::new(vec![1.0], vec![GraphIdentFlag::Identified], vec![1_u64])
            .map_err(|error| {
                refusal!(
                    "invalid_argument",
                    "sequential_evaluation.voi_ranking",
                    error.to_string(),
                    "internal ranking context"
                )
            })?;
    let eval = DesignEvaluationContext {
        graphs: &graphs,
        effect_width: None,
        model_loglik: None,
        decisions: Some(&registry),
        query_id_unlock: None,
        env_id_unlock: None,
        identified_under_intervention: None,
        graph_features: None,
    };
    let candidates = [CandidateDesign::IncreaseSamplingRate(SamplingPlan {
        additional_samples: 1,
        cost: DesignCost::zero(),
        tag: 0,
    })];
    let ranking = DesignRanker::new()
        .with_config(DesignRankConfig {
            min_batches: 2,
            max_batches: 2,
            batch_size: 2,
            rank_uncertainty_threshold: 0.0,
        })
        .rank(
            &DesignObjective::ReduceDecisionRegret { decision: DecisionProblemId::from_raw(0) },
            &candidates,
            &eval,
            ctx,
        )
        .map_err(|error| design_refusal(&error))?;
    let scored = ranking.ranked.first().ok_or_else(|| {
        refusal!(
            "invalid_argument",
            "sequential_evaluation.voi_ranking",
            "the ranker scored no candidate",
            "internal ranking context"
        )
    })?;
    if scored.evaluation != ScoreEvaluation::Exact {
        return Err(refusal!(
            "invalid_argument",
            "sequential_evaluation.voi_ranking",
            "the ranker did not evaluate the finite channel exactly",
            "the cross-check requires the exact preposterior path",
        ));
    }
    let evsi = scored.score;
    // The Bayes action after each observation, from the same utilities.
    let root = HistoryKey::root();
    let mut policy = ContingentPolicy::new("bayes_after_observation")
        .decide(root.clone(), request.collect_action);
    for branch in branches(compiled, &compiled.initial, request.collect_action) {
        let mut best = 0;
        let mut best_value = f64::NEG_INFINITY;
        for (index, utilities) in table.iter().enumerate() {
            let value = dot(&branch.posterior, utilities);
            if value > best_value {
                best_value = value;
                best = index;
            }
        }
        policy = policy.decide(
            root.extended(request.collect_action, branch.symbol),
            request.commit_actions[best],
        );
    }
    let enumerated = walk(compiled, &policy, None, limits, ctx)?.value;
    let predicted = prior_value + evsi - collect.utility_cost;
    if (predicted - enumerated).abs() > 1e-9 * (1.0 + enumerated.abs()) {
        return Err(refusal!(
            "invalid_argument",
            "sequential_evaluation.voi_enumeration_mismatch",
            format!(
                "the ranker's value of information implies {predicted} for collecting then acting, but the enumeration gives {enumerated}"
            ),
            "the finite channel and the model disagree; revalidate the model declaration",
        ));
    }
    Ok(VoiCheck {
        collect_action: request.collect_action,
        prior_value,
        expected_value_of_sample_information: evsi,
        cost: collect.utility_cost,
        net_value_of_collecting: evsi - collect.utility_cost,
        predicted_policy_value: predicted,
        enumerated_policy_value: enumerated,
        ranking_evaluation: scored.evaluation.as_str(),
        implemented_functional: scored.implemented_functional.to_string(),
        bayes_after_observation: policy,
    })
}

/// Score the value of collecting one observation before committing, through
/// the design ranker, and check it against the enumerated policy value.
///
/// # Errors
/// A malformed model; a collect action that is not pure information; a model
/// outside the one-collection scope; a ranker refusal; or a disagreement
/// between the ranker and the enumeration.
pub fn value_of_collecting_evidence(
    model: &FiniteSequentialModel,
    request: &VoiRequest,
    limits: EvaluationLimits,
    ctx: &ExecutionContext,
) -> SequentialResult<VoiCheck> {
    let compiled = model.compile()?;
    value_of_collecting(&compiled, request, limits, ctx)
}
