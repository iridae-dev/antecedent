//! Exact decisions across a preference set when the value is affine in the parameters.
//!
//! For a [`PreferenceContract`] whose value model has a linear form, an action's criterion
//! value at a parameter point `theta` is the affine function
//! `v_a(theta) = c_a0 + sum_j theta_j c_aj`, where `c_aj` is the expectation of part `j` of the
//! linear form (part 0 is the fixed part) under the supplied source. Every question about the
//! admissible set (the declared ranges cut by the declared relations) is then a small linear
//! program: the range of each action's value, the range of each pairwise margin, whether an
//! action is best at every admissible point (*necessarily best*) or at some admissible point
//! (*possibly best*), each action's worst-case regret, and the pairs whose order changes inside
//! the set (*switches*), with one admissible point on each side and, when the order depends on a
//! single parameter, the value where it changes.
//!
//! Three situations are kept apart. An action strictly ahead of every rival everywhere is the
//! unique winner ([`PreferenceVerdict::NecessarilyBest`]). An action never beaten but tying some
//! rival somewhere, usually at the edge of the set, is reported as
//! [`PreferenceVerdict::NeverBeaten`] with the rivals it ties. An action strictly behind one
//! rival everywhere is dominated by it ([`PreferenceActionOutcome::dominated_by`]).
//!
//! Every result states its verdict in plain words ([`PreferenceDecisionResult::statement`]).
//!
//! A range over the set is a range over declared assumptions. It is not a confidence interval
//! and carries no probability. A share is reported only for a supplied [`PreferenceMeasure`],
//! and is the weight of that measure's points, not a probability of the preferences.
//!
//! Expected utility (larger is better), expected loss (smaller is better) and expected regret
//! are answered here. Expected regret ranks actions exactly as expected utility does at every
//! parameter point, because the best-action term is common to every action, so its verdict is
//! computed from expected-utility margins and no expected-regret amounts are reported. Shape
//! parameters, criteria that are not affine in the parameters and the worst case of expected
//! regret refuse with [`PreferenceDecisionError::CertifiedRouteRequired`].
//!
//! [`evaluate_preference_structural`] repeats the analysis across structures under the
//! contract's [`StructuralPolicy`]. Unidentified and unevaluated structure mass is reported and
//! never renormalized away; an action excluded by a hard constraint in any evaluated structure is
//! excluded in every structure.

// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashSet;

use antecedent_contract::ExternalRefusal;
use antecedent_core::{CancellationToken, ExecutionContext, is_blank};
use antecedent_io_core::distribution_artifact::DistributionArtifact;
use antecedent_stats::linear_program::{
    self, ConstraintSense, LinearConstraint, LinearProgram, LpError, LpOptions, LpOutcome,
    LpSolution,
};

use crate::decision_contract::{
    DecisionContract, DecisionContractError, DecisionCriterion, StructuralPolicy,
};
use crate::decision_eval::{
    ConstraintExclusion, DecisionEvalError, DecisionResult, MeanSource, SourceReceipt,
    evaluate_contract, evaluate_contract_on_means, utility_draws,
};
use crate::decision_structural::{AtomEvidence, StructuralAtom};
use crate::preference_contract::{PreferenceContract, PreferenceContractError, PreferencePolicy};
use crate::preference_model::{PreferenceSet, RelationOrigin};
use crate::preference_refusal::{CERTIFIED_ROUTE, EVALUATE, STRUCTURAL, refusal};
use crate::signal::at_stage;
use crate::structure_aggregator::{
    INVARIANCE_UNCHECKED, NO_EVALUATED_POSITIVE_PROBABILITY, NO_STRUCTURE_EVALUATED,
    UNRESOLVED_MASS_CAN_CHANGE_BAYES, WORST_CASE_NEEDS_EVERY_STRUCTURE, probability_sum,
    unassigned_probability,
};

/// Largest number of rival-structure choices enumerated to decide whether an action is
/// possibly best under the worst case over structures.
const MAX_STRUCTURE_CHOICES: usize = 4096;
/// Draws of the draw share examined between two reads of the cancellation token.
const DRAWS_PER_CHECK: usize = 1024;

/// A weighted set of parameter points: a measure over the preference set.
#[derive(Clone, Debug, PartialEq)]
pub struct PreferenceMeasure {
    /// The points, each with one finite coordinate per model parameter.
    pub points: Vec<Vec<f64>>,
    /// Non-negative weight of each point, normalized by their total; `None` gives every point
    /// the same weight.
    pub weights: Option<Vec<f64>>,
    /// Optional distinct, non-blank name of each point (a named viewpoint).
    pub labels: Option<Vec<String>>,
    /// Where the measure came from.
    pub origin: MeasureOrigin,
    /// Parameter indices the measure says nothing about. Their coordinates in `points` are
    /// ignored; the declared range and relations still apply to them.
    pub unresolved: Vec<usize>,
}

/// Where a measure over the preference set came from.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MeasureOrigin {
    /// Declared by the analyst as a judgment.
    Declared {
        /// Names the declared judgment; non-blank.
        assumption_id: String,
    },
    /// Estimated from recorded choices.
    ChoiceEstimate {
        /// Identity of the estimate; non-blank.
        estimate_identity: String,
    },
}

impl PreferenceMeasure {
    /// Check the measure against a preference set: at least one point, one finite coordinate
    /// per parameter, valid weights and labels, known and distinct unresolved indices, and every
    /// point inside the set within the default linear-program feasibility tolerance (for a
    /// point with unresolved parameters: some admissible value of those parameters puts it
    /// inside).
    ///
    /// # Errors
    /// An invalid set, an empty set, or the measure refusal naming the cause.
    pub fn validate(&self, set: &PreferenceSet) -> Result<(), PreferenceDecisionError> {
        set.validate().map_err(|error| {
            PreferenceDecisionError::Contract(PreferenceContractError::Model(error))
        })?;
        let lp = LpOptions::default();
        let region = Region::of(set);
        if region.feasible(&[], lp)?.is_none() {
            return Err(PreferenceDecisionError::EmptyPreferenceSet);
        }
        prepare_measure(self, set, &region, lp).map(|_| ())
    }
}

/// One action's outcome across the preference set.
#[derive(Clone, Debug, PartialEq)]
pub struct PreferenceActionOutcome {
    /// Semantic action identity.
    pub id: String,
    /// Whether the action takes part in the comparison: no hard constraint excluded it (in a
    /// structural analysis, in no evaluated structure).
    pub admissible: bool,
    /// Hard constraints the action failed under this source.
    pub exclusions: Vec<ConstraintExclusion>,
    /// `(c_a0, c_a1, ..., c_aK)`: expectations of the fixed part and of each parameter's part,
    /// in larger-is-better orientation (negated for a loss criterion). `None` when excluded or
    /// when the score is not one affine function (a worst case over structures, or a combination
    /// of per-structure results).
    pub coefficients: Option<Vec<f64>>,
    /// Lowest and highest criterion value over the set, in the criterion's own orientation. A
    /// range over declared assumptions, not a confidence interval. `None` when excluded and for
    /// expected regret.
    pub value_range: Option<(f64, f64)>,
    /// Never beaten: its margin over every admissible rival is at least minus the tie tolerance
    /// at every admissible parameter point. Ties are allowed, so an action can be necessarily
    /// best without being the unique winner. `false` when excluded.
    pub necessarily_best: bool,
    /// Strictly ahead everywhere: its margin over every admissible rival exceeds the tie
    /// tolerance at every admissible parameter point. Implies `necessarily_best`; at most one
    /// action has it. `false` when excluded.
    pub uniquely_best: bool,
    /// Admissible rivals strictly ahead of this action at every admissible parameter point
    /// (their margin over it exceeds the tie tolerance everywhere), in declaration order. An
    /// action can be best at no admissible point without being dominated by any single rival:
    /// different rivals can lead it in different parts of the set. Empty when excluded.
    pub dominated_by: Vec<String>,
    /// Best (ties allowed) at some admissible parameter point. `Some(false)` when excluded;
    /// `None` when the worst case over structures has too many rival-structure choices to
    /// enumerate.
    pub possibly_best: Option<bool>,
    /// Largest shortfall from the best rival over the set, in expected-utility units (expected
    /// loss units for a loss criterion), floored at zero. `None` when excluded.
    pub worst_case_regret: Option<f64>,
    /// Weight of the measure points at which this action leads, when a measure was supplied.
    /// A point where several actions are within the tie tolerance of the best is shared equally
    /// among them. Not a probability unless the measure is one.
    pub measure_share: Option<f64>,
    /// Measure-weighted criterion value, in the criterion's own orientation. `None` without a
    /// measure, with unresolved parameters, for expected regret and when excluded.
    pub measure_value: Option<f64>,
}

/// A pair of actions whose order changes inside the set.
#[derive(Clone, Debug, PartialEq)]
pub struct PreferenceSwitch {
    /// The earlier action in declaration order.
    pub first: String,
    /// The later action in declaration order.
    pub second: String,
    /// `(constant, per parameter)` of `v_first - v_second` in larger-is-better orientation;
    /// empty when the margin is not one affine function (structural combinations).
    pub margin_coefficients: Vec<f64>,
    /// Lowest and highest margin of `first` over `second` over the set, in larger-is-better
    /// orientation. A range over declared assumptions, not a confidence interval.
    pub margin_range: (f64, f64),
    /// An admissible point where `first` leads (where the margin is highest).
    pub favours_first_at: Vec<f64>,
    /// An admissible point where `second` leads (where the margin is lowest).
    pub favours_second_at: Vec<f64>,
    /// Where the order changes, when the margin depends on a single parameter; `None` when it
    /// depends on several (the boundary is then the hyperplane where the affine function with
    /// `margin_coefficients` is zero) and for structural combinations without coefficients.
    pub threshold: Option<SwitchThreshold>,
}

/// The value of a single parameter at which a switching pair changes order.
///
/// A margin depends on a single parameter when every other parameter's coefficient, times the
/// width of that parameter's declared range, is within the tie tolerance (a fixed parameter,
/// with a range of zero width, never counts). The other parameters are then held at the
/// midpoints of their declared ranges, which moves the margin by no more than the tie
/// tolerance.
#[derive(Clone, Debug, PartialEq)]
pub struct SwitchThreshold {
    /// The parameter the margin depends on.
    pub parameter_id: String,
    /// The parameter value at which the margin is zero.
    pub value: f64,
    /// Whether the first action of the switch leads above the value (and the second below it).
    pub first_leads_above: bool,
}

/// What can be claimed about the choice across the preference set.
#[derive(Clone, Debug, PartialEq)]
pub enum PreferenceVerdict {
    /// One admissible action is strictly ahead of every admissible rival, by more than the tie
    /// tolerance, at every admissible parameter point: the unique winner.
    NecessarilyBest(String),
    /// One admissible action is never beaten anywhere in the set but is not strictly ahead
    /// everywhere: it ties some rivals somewhere and leads them elsewhere.
    NeverBeaten {
        /// The action that is never beaten.
        action: String,
        /// Rivals whose margin range against it touches zero (within the tie tolerance), in
        /// declaration order.
        ties_with: Vec<String>,
    },
    /// No action is best everywhere; the actions best at some admissible point, in
    /// declaration order.
    PreferenceDependent {
        /// Possibly-best actions, in declaration order.
        possibly_best: Vec<String>,
    },
    /// The action with the best worst-case criterion value over the set.
    WorstCaseChoice(String),
    /// The action with the smallest worst-case regret over the set.
    MinimaxRegretChoice(String),
    /// The action with the best measure-weighted criterion value.
    MeasureChoice {
        /// The chosen action.
        action: String,
        /// Weight of the measure points at which it leads.
        share: f64,
    },
    /// Several actions share the best score within the tie tolerance, or (under the invariant
    /// policy) several actions are never beaten, which makes them equal within the tie tolerance
    /// at every admissible point; declaration order is not a choice.
    Tied(Vec<String>),
    /// Everything was computed; no choice is made.
    ReportOnly,
    /// Every action was excluded by a hard constraint.
    NoAdmissibleAction,
}

/// The decision across a preference set for one source.
#[derive(Clone, Debug, PartialEq)]
pub struct PreferenceDecisionResult {
    /// Identity of the preference contract.
    pub contract_identity: String,
    /// Preference policy that produced the verdict.
    pub policy: PreferencePolicy,
    /// Parameter ids, in index order; the coordinates of every point reported.
    pub parameter_ids: Vec<String>,
    /// Lowest and highest admissible value of each parameter, in index order: the declared
    /// range cut by the relations (for a learned set, the range the recorded choices leave).
    pub parameter_ranges: Vec<(f64, f64)>,
    /// Per-action outcomes, in declaration order.
    pub actions: Vec<PreferenceActionOutcome>,
    /// Switches, ordered by the declaration index of `(first, second)`.
    pub switches: Vec<PreferenceSwitch>,
    /// The verdict.
    pub verdict: PreferenceVerdict,
    /// Weight of draws in which the only never-beaten action (the unique winner, or the one
    /// action never beaten) also has a per-draw utility at least every rival's at every vertex
    /// of the set. A point diagnostic of the supplied law, not a calibrated probability. `None`
    /// for a mean source, unless exactly one admissible action is never beaten, when vertex
    /// enumeration exceeds its cap, and for structural combinations.
    pub draw_share_best_everywhere: Option<f64>,
    /// Weight of measure points whose leader changes over the unresolved parameters; `None`
    /// without a measure.
    pub unresolved_measure_share: Option<f64>,
    /// Parameter assumptions, relation origins and the measure origin; sorted, distinct.
    pub assumptions: Vec<String>,
    /// Source lineage. For a structural combination each field lists the distinct values of
    /// the evaluated structures, joined by `+`.
    pub source: SourceReceipt,
    /// Number of draws; 0 for a mean source; the sum over evaluated structures for a
    /// combination.
    pub n_draws: usize,
}

/// Tolerances and limits of the exact verdict.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PreferenceDecisionOptions {
    /// Margins within this distance of zero, in criterion units, count as ties; finite and
    /// non-negative.
    pub tie_tolerance: f64,
    /// Largest number of candidate bases vertex enumeration may solve.
    pub vertex_cap: usize,
    /// Linear-program tolerances and work limit.
    pub lp: LpOptions,
}

impl Default for PreferenceDecisionOptions {
    fn default() -> Self {
        Self { tie_tolerance: 1e-9, vertex_cap: 100_000, lp: LpOptions::default() }
    }
}

/// One structure in a structural result.
#[derive(Clone, Debug, PartialEq)]
pub struct PreferenceAtomSummary {
    /// Structure identity.
    pub id: String,
    /// Supplied structure probability.
    pub probability: Option<f64>,
    /// What became of the structure.
    pub status: PreferenceAtomStatus,
}

/// What became of one structure.
#[derive(Clone, Debug, PartialEq)]
pub enum PreferenceAtomStatus {
    /// Evaluated alone under the preference policy.
    Evaluated(Box<PreferenceDecisionResult>),
    /// Not identified.
    Unidentified,
    /// Not evaluated, with the supplied reason or the registered reason code.
    Unevaluated(String),
}

/// One structure's own verdict, listed when structures disagree.
#[derive(Clone, Debug, PartialEq)]
pub struct AtomChoice {
    /// Structure identity.
    pub atom: String,
    /// Its verdict under the preference policy.
    pub verdict: PreferenceVerdict,
}

/// What can be claimed across structures and preference parameters.
#[derive(Clone, Debug, PartialEq)]
pub enum StructuralPreferenceVerdict {
    /// The joint statement over structures and parameters.
    Combined(PreferenceVerdict),
    /// Evaluated structures give different verdicts; each structure's verdict is listed.
    StructuresDisagree(Vec<AtomChoice>),
    /// A claim would need a structure that is unidentified or unevaluated, or mass that is
    /// unresolved.
    InsufficientScience(String),
    /// Per-structure results only.
    ReportOnly,
}

/// The decision across structures and the preference set.
#[derive(Clone, Debug, PartialEq)]
pub struct StructuralPreferenceResult {
    /// Identity of the preference contract.
    pub contract_identity: String,
    /// Structural policy that combined the structures.
    pub structural_policy: StructuralPolicy,
    /// Preference policy applied within and across structures.
    pub preference_policy: PreferencePolicy,
    /// Per-structure outcomes, in the order supplied.
    pub atoms: Vec<PreferenceAtomSummary>,
    /// The joint table over structures and parameters; `None` for the report-only structural
    /// policy and when no structure was evaluated.
    pub combined: Option<PreferenceDecisionResult>,
    /// Probability mass of unidentified structures, when probabilities were supplied.
    pub unidentified_mass: Option<f64>,
    /// Probability mass of unevaluated structures plus undeclared mass, when probabilities
    /// were supplied.
    pub unevaluated_mass: Option<f64>,
    /// Probability mass of evaluated structures, when probabilities were supplied.
    pub evaluated_mass: Option<f64>,
    /// The verdict.
    pub verdict: StructuralPreferenceVerdict,
}

// ---------------------------------------------------------------------------------------------
// Plain statements
// ---------------------------------------------------------------------------------------------

impl PreferenceDecisionResult {
    /// The verdict in one or two plain sentences, built only from this result's own facts.
    ///
    /// Numbers are rounded to four significant digits (more when the integer part is longer),
    /// written in fixed notation for magnitudes from 0.0001 up to but excluding one million and
    /// in scientific notation (`1.5e7`) otherwise, with trailing zeros dropped. A range over the
    /// set is stated as a range and a measure share as a weight; neither is called a
    /// probability. The same result always gives the same statement.
    #[must_use]
    pub fn statement(&self) -> String {
        verdict_statement(&self.verdict, self.policy, Some(self), "")
    }
}

impl StructuralPreferenceResult {
    /// The verdict across structures in one or two plain sentences, built only from this
    /// result's own facts, with the same conventions as [`PreferenceDecisionResult::statement`].
    #[must_use]
    pub fn statement(&self) -> String {
        match &self.verdict {
            StructuralPreferenceVerdict::Combined(verdict) => {
                let scope = match self.structural_policy {
                    StructuralPolicy::BayesOverStructures => {
                        ", with the evaluated structures weighted by their probabilities"
                    }
                    StructuralPolicy::Maximin => {
                        ", scoring each action by its worst value over the evaluated structures"
                    }
                    StructuralPolicy::RequireInvariantBestAction | StructuralPolicy::ReportOnly => {
                        " and every evaluated structure"
                    }
                };
                verdict_statement(verdict, self.preference_policy, self.combined.as_ref(), scope)
            }
            StructuralPreferenceVerdict::StructuresDisagree(choices) => {
                let atoms: Vec<String> = choices.iter().map(|c| c.atom.clone()).collect();
                format!(
                    "The evaluated structures {} give different verdicts, so no action is \
                     preferred across structures.",
                    join_ids(&atoms)
                )
            }
            StructuralPreferenceVerdict::InsufficientScience(reason) => {
                format!("No action is chosen across structures: {reason}.")
            }
            StructuralPreferenceVerdict::ReportOnly => {
                "No action is chosen: the result of each structure is reported only.".to_owned()
            }
        }
    }
}

/// Write a number for a plain statement: rounded to four significant digits (more when the
/// integer part is longer), in fixed notation for magnitudes from 0.0001 up to but excluding
/// one million and in scientific notation (`1.5e7`) otherwise, with trailing zeros dropped.
/// Zero is written `0` whatever its sign; a non-finite value is written as Rust displays it.
pub(crate) fn plain_number(value: f64) -> String {
    fn trimmed(text: &str) -> String {
        let text = if text.contains('.') {
            text.trim_end_matches('0').trim_end_matches('.')
        } else {
            text
        };
        if text == "-0" { "0".to_owned() } else { text.to_owned() }
    }
    if !value.is_finite() {
        return format!("{value}");
    }
    let magnitude = value.abs();
    if magnitude < f64::MIN_POSITIVE {
        return "0".to_owned();
    }
    if !(1e-4..1e6).contains(&magnitude) {
        let text = format!("{value:.3e}");
        return match text.split_once('e') {
            Some((mantissa, power)) => format!("{}e{power}", trimmed(mantissa)),
            None => text,
        };
    }
    let decimals = [1e3, 1e2, 1e1, 1.0, 1e-1, 1e-2, 1e-3]
        .iter()
        .position(|&bound| magnitude >= bound)
        .unwrap_or(7);
    trimmed(&format!("{value:.decimals$}"))
}

/// `a`, `a and b`, `a, b and c`.
pub(crate) fn join_ids(ids: &[String]) -> String {
    match ids {
        [] => String::new(),
        [only] => only.clone(),
        [init @ .., last] => format!("{} and {last}", init.join(", ")),
    }
}

/// Where a claim over the whole set holds: every admissible parameter value, or the declared
/// value model alone when it has no parameter.
fn over_the_set(has_parameters: bool, quantifier: &str) -> String {
    if has_parameters {
        format!("for {quantifier} admissible value of the declared trade-offs")
    } else {
        "under the declared value model".to_owned()
    }
}

/// The unique-winner sentence shared with the certified route.
pub(crate) fn preferred_everywhere(action: &str, has_parameters: bool, scope: &str) -> String {
    format!("Action {action} is preferred {}{scope}.", over_the_set(has_parameters, "every"))
}

/// The worst-case choice sentence shared with the certified route.
pub(crate) fn worst_case_sentence(action: &str, scope: &str) -> String {
    format!(
        "Action {action} has the best worst-case value over the admissible values of the \
         declared trade-offs{scope}."
    )
}

/// The minimax-regret sentence shared with the certified route.
pub(crate) fn minimax_regret_sentence(action: &str, scope: &str) -> String {
    format!(
        "Action {action} has the smallest worst-case regret over the admissible values of the \
         declared trade-offs{scope}."
    )
}

/// The measure-choice sentence shared with the certified route.
pub(crate) fn measure_choice_sentence(action: &str, share: f64) -> String {
    format!(
        "Action {action} has the best value weighted by the supplied measure over the declared \
         trade-offs; it leads at measure points carrying a weight of {}.",
        plain_number(share)
    )
}

/// The report-only sentence shared with the certified route.
pub(crate) const REPORT_ONLY_SENTENCE: &str =
    "No action is chosen: the comparison across the declared trade-offs is reported only.";

/// The no-admissible-action sentence shared with the certified route.
pub(crate) const NO_ADMISSIBLE_SENTENCE: &str =
    "No action is admissible: every action fails a hard constraint.";

fn verdict_statement(
    verdict: &PreferenceVerdict,
    policy: PreferencePolicy,
    result: Option<&PreferenceDecisionResult>,
    scope: &str,
) -> String {
    let has_parameters = result.is_none_or(|r| !r.parameter_ids.is_empty());
    match verdict {
        PreferenceVerdict::NecessarilyBest(action) => {
            preferred_everywhere(action, has_parameters, scope)
        }
        PreferenceVerdict::NeverBeaten { action, ties_with } => {
            never_beaten_sentence(action, ties_with, has_parameters, result, scope)
        }
        PreferenceVerdict::PreferenceDependent { possibly_best } => {
            dependent_sentence(possibly_best, result)
        }
        PreferenceVerdict::WorstCaseChoice(action) => worst_case_sentence(action, scope),
        PreferenceVerdict::MinimaxRegretChoice(action) => minimax_regret_sentence(action, scope),
        PreferenceVerdict::MeasureChoice { action, share } => {
            measure_choice_sentence(action, *share)
        }
        PreferenceVerdict::Tied(ids) => {
            let subject = match ids.as_slice() {
                [only] => format!("Action {only}"),
                _ => format!("Actions {}", join_ids(ids)),
            };
            let (verb, plural) = match ids.as_slice() {
                [_] => ("shares", "is"),
                _ => ("share", "are"),
            };
            let shared = match policy {
                PreferencePolicy::Maximin => format!(
                    "{verb} the best worst-case value over the admissible values of the \
                     declared trade-offs{scope}"
                ),
                PreferencePolicy::MinimaxRegret => format!(
                    "{verb} the smallest worst-case regret over the admissible values of the \
                     declared trade-offs{scope}"
                ),
                PreferencePolicy::Weighted => {
                    format!("{verb} the best value weighted by the supplied measure")
                }
                PreferencePolicy::RequireInvariantBestAction | PreferencePolicy::ReportOnly => {
                    format!("{plural} equal {}{scope}", over_the_set(has_parameters, "every"))
                }
            };
            format!("{subject} {shared}; no single action is chosen.")
        }
        PreferenceVerdict::ReportOnly => REPORT_ONLY_SENTENCE.to_owned(),
        PreferenceVerdict::NoAdmissibleAction => NO_ADMISSIBLE_SENTENCE.to_owned(),
    }
}

fn never_beaten_sentence(
    action: &str,
    ties_with: &[String],
    has_parameters: bool,
    result: Option<&PreferenceDecisionResult>,
    scope: &str,
) -> String {
    // An affine margin that is never negative and not zero everywhere is zero only on the
    // boundary of the set, so with one affine score per action the ties lie on its edge.
    let affine = |id: &str| {
        result.is_some_and(|r| r.actions.iter().any(|o| o.id == id && o.coefficients.is_some()))
    };
    let place = if !has_parameters {
        ""
    } else if affine(action) && ties_with.iter().all(|b| affine(b)) {
        " at the edge of the admissible set"
    } else {
        " at some admissible values"
    };
    format!(
        "Action {action} is never beaten {}{scope}, and ties with {}{place}.",
        over_the_set(has_parameters, "any"),
        join_ids(ties_with)
    )
}

fn dependent_sentence(
    possibly_best: &[String],
    result: Option<&PreferenceDecisionResult>,
) -> String {
    const OPENING: &str = "The preferred action depends on the declared trade-offs";
    let each = match possibly_best {
        [] => return format!("{OPENING}."),
        [only] => format!("{only} is preferred somewhere in the admissible set"),
        _ => format!(
            "{} are each preferred somewhere in the admissible set",
            join_ids(possibly_best)
        ),
    };
    let Some(result) = result else {
        return format!("{OPENING}: {each}.");
    };
    let relevant: Vec<&PreferenceSwitch> = result
        .switches
        .iter()
        .filter(|s| {
            !s.margin_coefficients.is_empty()
                && possibly_best.contains(&s.first)
                && possibly_best.contains(&s.second)
        })
        .collect();
    if relevant.is_empty() {
        return format!("{OPENING}: {each}.");
    }
    let thresholds: Option<Vec<&SwitchThreshold>> =
        relevant.iter().map(|s| s.threshold.as_ref()).collect();
    let single = thresholds.as_ref().and_then(|list| {
        let first = list.first()?;
        list.iter().all(|t| t.parameter_id == first.parameter_id).then_some(first)
    });
    let Some(threshold) = single else {
        return format!(
            "{OPENING}: {each}, and the boundary between them depends on more than one trade-off."
        );
    };
    let parameter = &threshold.parameter_id;
    let range = result
        .parameter_ids
        .iter()
        .position(|id| id == parameter)
        .and_then(|j| result.parameter_ranges.get(j))
        .map(|(low, high)| {
            format!(
                "the admissible range of {parameter} is {} to {}",
                plain_number(*low),
                plain_number(*high)
            )
        });
    match (relevant.as_slice(), possibly_best.len(), range) {
        ([switch], 2, Some(range)) => {
            let (below, above) = if threshold.first_leads_above {
                (&switch.second, &switch.first)
            } else {
                (&switch.first, &switch.second)
            };
            format!(
                "{OPENING}: {below} leads when {parameter} is below {} and {above} leads above \
                 it; {range}.",
                plain_number(threshold.value)
            )
        }
        (_, _, Some(range)) => {
            format!("{OPENING}: {each}, changing order as {parameter} varies; {range}.")
        }
        (_, _, None) => format!("{OPENING}: {each}."),
    }
}

/// Why the exact route cannot answer and the certified route is needed.
#[derive(Clone, Debug, PartialEq)]
pub enum CertifiedRouteReason {
    /// The criterion is not affine in the parameters.
    Criterion(DecisionCriterion),
    /// The value is not affine in these parameters.
    ShapeParameters {
        /// Parameter ids.
        parameters: Vec<String>,
    },
    /// A part of the linear form holds an exponential or logarithm of a quantity attribute.
    PartNotLowerable,
    /// The worst case of expected regret is not a linear program over the set.
    WorstCaseOfExpectedRegret,
    /// The set holds value relations, which are not linear in the parameters.
    ValueRelations {
        /// Relation ids.
        relations: Vec<String>,
    },
}

impl std::fmt::Display for CertifiedRouteReason {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Criterion(criterion) => {
                write!(formatter, "criterion {criterion:?} is not affine in the parameters")
            }
            Self::ShapeParameters { parameters } => {
                write!(formatter, "the value is not affine in parameters {}", parameters.join(", "))
            }
            Self::PartNotLowerable => write!(
                formatter,
                "a part of the linear form holds an exponential or logarithm of a quantity"
            ),
            Self::WorstCaseOfExpectedRegret => {
                write!(formatter, "the worst case of expected regret is not a linear program")
            }
            Self::ValueRelations { relations } => write!(
                formatter,
                "the preference set holds value relations {}, which are not linear",
                relations.join(", ")
            ),
        }
    }
}

/// Why the exact verdict refused.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum PreferenceDecisionError {
    /// The preference contract or a contract derived from it is refused.
    #[error("preference contract: {0}")]
    Contract(PreferenceContractError),
    /// The source cannot answer the derived ordinary contract.
    #[error("source: {0}")]
    Eval(DecisionEvalError),
    /// A linear program refused.
    #[error("linear program: {0}")]
    Lp(LpError),
    /// The exact route does not apply.
    #[error("the certified route is needed: {reason}")]
    CertifiedRouteRequired {
        /// Why.
        reason: CertifiedRouteReason,
    },
    /// An option is invalid.
    #[error("option {option} is invalid")]
    InvalidOptions {
        /// Name of the option.
        option: &'static str,
    },
    /// The relations contradict the declared ranges: no parameter point is admissible.
    #[error("the preference set is empty")]
    EmptyPreferenceSet,
    /// The weighted policy needs a measure over the set.
    #[error("the weighted policy needs a measure over the preference set")]
    MeasureRequired,
    /// The weighted policy needs a measure that resolves every parameter.
    #[error("the measure leaves parameters {} unresolved", .parameters.join(", "))]
    MeasureIncomplete {
        /// Ids of the unresolved parameters.
        parameters: Vec<String>,
    },
    /// The measure has no point.
    #[error("the measure has no point")]
    MeasureEmpty,
    /// A measure point has the wrong number of coordinates.
    #[error("measure point {index} has {found} coordinates, expected {expected}")]
    MeasurePointDimension {
        /// Point index.
        index: usize,
        /// Model dimension.
        expected: usize,
        /// Coordinates given.
        found: usize,
    },
    /// A measure point has a non-finite coordinate.
    #[error("measure point {index} has a non-finite coordinate")]
    MeasurePointNonFinite {
        /// Point index.
        index: usize,
    },
    /// A measure point lies outside the preference set.
    #[error("measure point {index} lies outside the preference set")]
    MeasurePointOutsideSet {
        /// Point index.
        index: usize,
    },
    /// The weights do not give one weight per point.
    #[error("{found} measure weights for {expected} points")]
    MeasureWeightCount {
        /// Points.
        expected: usize,
        /// Weights given.
        found: usize,
    },
    /// A weight is negative or not finite.
    #[error("measure weight {index} is negative or not finite")]
    MeasureWeightInvalid {
        /// Weight index.
        index: usize,
    },
    /// The weights sum to zero.
    #[error("the measure weights sum to zero")]
    MeasureWeightTotalZero,
    /// The labels do not give one label per point.
    #[error("{found} measure labels for {expected} points")]
    MeasureLabelCount {
        /// Points.
        expected: usize,
        /// Labels given.
        found: usize,
    },
    /// A label is blank.
    #[error("measure label {index} is blank")]
    MeasureLabelBlank {
        /// Label index.
        index: usize,
    },
    /// Two points share a label.
    #[error("measure label {label} repeats")]
    MeasureLabelDuplicate {
        /// The repeated label.
        label: String,
    },
    /// The measure origin names no assumption or estimate.
    #[error("the measure origin is blank")]
    MeasureOriginBlank,
    /// An unresolved index names no parameter.
    #[error("unresolved parameter index {index} is not a parameter")]
    MeasureUnresolvedUnknown {
        /// The index.
        index: usize,
    },
    /// An unresolved index repeats.
    #[error("unresolved parameter index {index} repeats")]
    MeasureUnresolvedRepeated {
        /// The index.
        index: usize,
    },
    /// No structure was supplied.
    #[error("no structure was supplied")]
    NoStructures,
    /// A structure id is blank.
    #[error("structure at position {position} has a blank id")]
    StructureIdBlank {
        /// Position of the structure.
        position: usize,
    },
    /// Two structures share an id.
    #[error("duplicate structure {id}")]
    StructureDuplicate {
        /// The repeated id.
        id: String,
    },
    /// Probabilities were supplied for some structures but not all.
    #[error("structure probabilities are given for some structures but not all")]
    StructureProbabilitiesPartial,
    /// A structure probability is negative or not finite.
    #[error("structure {structure} has a negative or non-finite probability")]
    StructureProbabilityInvalid {
        /// Structure id.
        structure: String,
    },
    /// Structure probabilities sum to more than one.
    #[error("structure probabilities sum to more than one")]
    StructureProbabilitiesExceedOne,
    /// Weighting over structures needs a probability on every structure.
    #[error("weighting over structures needs structure probabilities")]
    StructureProbabilitiesRequired,
    /// The expectation of a part of an action's value is not finite.
    #[error("action {action} has a non-finite expected part")]
    NonFiniteCoefficient {
        /// Action id.
        action: String,
    },
    /// A linear program over the bounded set reported an unbounded objective, which valid
    /// input cannot produce.
    #[error("a linear program over the bounded preference set was unbounded")]
    UnboundedProgram,
    /// A linear program over a region already shown non-empty reported no feasible point.
    #[error("a linear program over a non-empty region was infeasible")]
    UnexpectedInfeasible,
    /// The execution context was cancelled between linear programs, vertex bases or draws.
    #[error("the evaluation across the preference set was cancelled")]
    Cancelled,
}

impl PreferenceDecisionError {
    /// Stable detail id, prefixed `preference_decision.`; a contract refusal keeps its own
    /// `preference_contract.`, `preference_model.` or `decision_contract.` detail and a source
    /// refusal its `decision_evaluation.` or `decision_contract.` detail, the detail the
    /// returned refusal carries.
    #[must_use]
    pub fn detail(&self) -> &'static str {
        use antecedent_core::reason_details;
        match self {
            Self::Contract(inner) => PreferenceContractError::detail(inner),
            Self::Eval(inner) => inner.detail(),
            Self::Lp(_) => reason_details::preference_decision::LINEAR_PROGRAM_REFUSED,
            Self::CertifiedRouteRequired { reason } => match reason {
                CertifiedRouteReason::Criterion(_) => {
                    reason_details::preference_decision::CRITERION_NEEDS_CERTIFIED_ROUTE
                }
                CertifiedRouteReason::ShapeParameters { .. } => {
                    reason_details::preference_decision::SHAPE_PARAMETERS_NEED_CERTIFIED_ROUTE
                }
                CertifiedRouteReason::PartNotLowerable => {
                    reason_details::preference_decision::PART_NOT_LOWERABLE
                }
                CertifiedRouteReason::WorstCaseOfExpectedRegret => {
                    reason_details::preference_decision::WORST_CASE_OF_EXPECTED_REGRET
                }
                CertifiedRouteReason::ValueRelations { .. } => {
                    reason_details::preference_decision::VALUE_RELATIONS_NEED_CERTIFIED_ROUTE
                }
            },
            Self::InvalidOptions { .. } => reason_details::preference_decision::OPTIONS_INVALID,
            Self::EmptyPreferenceSet => reason_details::preference_decision::PREFERENCE_SET_EMPTY,
            Self::MeasureRequired => reason_details::preference_decision::MEASURE_REQUIRED,
            Self::MeasureIncomplete { .. } => {
                reason_details::preference_decision::MEASURE_INCOMPLETE
            }
            Self::MeasureEmpty => reason_details::preference_decision::MEASURE_EMPTY,
            Self::MeasurePointDimension { .. } => {
                reason_details::preference_decision::MEASURE_POINT_DIMENSION
            }
            Self::MeasurePointNonFinite { .. } => {
                reason_details::preference_decision::MEASURE_POINT_NON_FINITE
            }
            Self::MeasurePointOutsideSet { .. } => {
                reason_details::preference_decision::MEASURE_POINT_OUTSIDE_SET
            }
            Self::MeasureWeightCount { .. } => {
                reason_details::preference_decision::MEASURE_WEIGHT_COUNT
            }
            Self::MeasureWeightInvalid { .. } => {
                reason_details::preference_decision::MEASURE_WEIGHT_INVALID
            }
            Self::MeasureWeightTotalZero => {
                reason_details::preference_decision::MEASURE_WEIGHT_TOTAL_ZERO
            }
            Self::MeasureLabelCount { .. } => {
                reason_details::preference_decision::MEASURE_LABEL_COUNT
            }
            Self::MeasureLabelBlank { .. } => {
                reason_details::preference_decision::MEASURE_LABEL_BLANK
            }
            Self::MeasureLabelDuplicate { .. } => {
                reason_details::preference_decision::MEASURE_LABEL_DUPLICATE
            }
            Self::MeasureOriginBlank => reason_details::preference_decision::MEASURE_ORIGIN_BLANK,
            Self::MeasureUnresolvedUnknown { .. } => {
                reason_details::preference_decision::MEASURE_UNRESOLVED_UNKNOWN
            }
            Self::MeasureUnresolvedRepeated { .. } => {
                reason_details::preference_decision::MEASURE_UNRESOLVED_REPEATED
            }
            Self::NoStructures => reason_details::preference_decision::STRUCTURES_EMPTY,
            Self::StructureIdBlank { .. } => {
                reason_details::preference_decision::STRUCTURE_ID_BLANK
            }
            Self::StructureDuplicate { .. } => {
                reason_details::preference_decision::STRUCTURE_DUPLICATE
            }
            Self::StructureProbabilitiesPartial => {
                reason_details::preference_decision::STRUCTURE_PROBABILITIES_PARTIAL
            }
            Self::StructureProbabilityInvalid { .. } => {
                reason_details::preference_decision::STRUCTURE_PROBABILITY_INVALID
            }
            Self::StructureProbabilitiesExceedOne => {
                reason_details::preference_decision::STRUCTURE_PROBABILITIES_EXCEED_ONE
            }
            Self::StructureProbabilitiesRequired => {
                reason_details::preference_decision::STRUCTURE_PROBABILITIES_REQUIRED
            }
            Self::NonFiniteCoefficient { .. } => {
                reason_details::preference_decision::COEFFICIENT_NON_FINITE
            }
            Self::UnboundedProgram => reason_details::preference_decision::LINEAR_PROGRAM_UNBOUNDED,
            Self::UnexpectedInfeasible => {
                reason_details::preference_decision::LINEAR_PROGRAM_INFEASIBLE
            }
            Self::Cancelled => reason_details::preference_decision::CANCELLED,
        }
    }
}

use PreferenceDecisionError as E;

/// Evaluate a preference contract across its preference set on aligned joint draws.
///
/// The expectation of every part of the linear form is read once per action through
/// [`PreferenceContract::term_contract`]; hard-constraint exclusions are read from the ordinary
/// contract bound at one admissible point (constraints never depend on the parameters).
///
/// # Errors
/// An invalid contract or options; a criterion, value or policy the exact route cannot answer
/// (`CertifiedRouteRequired`); an empty preference set; an invalid measure, or a missing or
/// incomplete one under the weighted policy; a source refusal; a linear-program refusal.
pub fn evaluate_preference_contract(
    contract: &PreferenceContract,
    source: &DistributionArtifact,
    measure: Option<&PreferenceMeasure>,
    options: PreferenceDecisionOptions,
) -> Result<PreferenceDecisionResult, PreferenceDecisionError> {
    on_draws(contract, source, measure, options, None)
}

/// [`evaluate_preference_contract`] under an execution context: the context's cancellation
/// token is read before every linear program, every candidate vertex basis and every 1024 draws
/// of the draw share.
///
/// # Errors
/// As [`evaluate_preference_contract`]; [`PreferenceDecisionError::Cancelled`] once the token
/// is set.
pub fn evaluate_preference_contract_with_context(
    contract: &PreferenceContract,
    source: &DistributionArtifact,
    measure: Option<&PreferenceMeasure>,
    options: PreferenceDecisionOptions,
    ctx: &ExecutionContext,
) -> Result<PreferenceDecisionResult, PreferenceDecisionError> {
    on_draws(contract, source, measure, options, Some(ctx))
}

pub(crate) fn on_draws(
    contract: &PreferenceContract,
    source: &DistributionArtifact,
    measure: Option<&PreferenceMeasure>,
    options: PreferenceDecisionOptions,
    ctx: Option<&ExecutionContext>,
) -> Result<PreferenceDecisionResult, PreferenceDecisionError> {
    let prepared = Prepared::new(contract, measure, options, ctx)?;
    let facts = prepared.draw_facts(source)?;
    prepared.single(&facts, &facts.admissible).map(|(result, _)| result)
}

/// Evaluate a preference contract across its preference set on a mean-only source.
///
/// Exact only when every part of the linear form is affine in the attributes; otherwise the
/// mean evaluator's refusal is surfaced. Expected regret needs aligned draws and is refused as
/// the ordinary contract refuses it on a mean source. A contract with hard constraints refuses as the mean
/// evaluator does, since a mean cannot answer a constraint probability.
///
/// # Errors
/// As [`evaluate_preference_contract`].
pub fn evaluate_preference_contract_on_means(
    contract: &PreferenceContract,
    source: &MeanSource,
    measure: Option<&PreferenceMeasure>,
    options: PreferenceDecisionOptions,
) -> Result<PreferenceDecisionResult, PreferenceDecisionError> {
    on_means(contract, source, measure, options, None)
}

/// [`evaluate_preference_contract_on_means`] under an execution context: the context's
/// cancellation token is read before every linear program.
///
/// # Errors
/// As [`evaluate_preference_contract_on_means`]; [`PreferenceDecisionError::Cancelled`] once
/// the token is set.
pub fn evaluate_preference_contract_on_means_with_context(
    contract: &PreferenceContract,
    source: &MeanSource,
    measure: Option<&PreferenceMeasure>,
    options: PreferenceDecisionOptions,
    ctx: &ExecutionContext,
) -> Result<PreferenceDecisionResult, PreferenceDecisionError> {
    on_means(contract, source, measure, options, Some(ctx))
}

pub(crate) fn on_means(
    contract: &PreferenceContract,
    source: &MeanSource,
    measure: Option<&PreferenceMeasure>,
    options: PreferenceDecisionOptions,
    ctx: Option<&ExecutionContext>,
) -> Result<PreferenceDecisionResult, PreferenceDecisionError> {
    let prepared = Prepared::new(contract, measure, options, ctx)?;
    let facts = prepared.mean_facts(source)?;
    prepared.single(&facts, &facts.admissible).map(|(result, _)| result)
}

/// Evaluate a preference contract across structures and its preference set.
///
/// Every evaluated structure is analysed alone under the preference policy (its result is in
/// `atoms`), then combined under the contract's structural policy:
///
/// - `BayesOverStructures` mixes the coefficients `sum_s p_s c_a(s)` over evaluated structures,
///   without renormalizing, and applies the single-source rules to the mixture; positive
///   unresolved mass leaves the choice unsupported.
/// - `Maximin` scores each action by its worst value over structures,
///   `g_a(theta) = min_s v_a(s, theta)`, and answers every question by linear programs over the
///   set; it needs every structure evaluated.
/// - `RequireInvariantBestAction` requires every structure evaluated and every structure's
///   verdict to agree; the combined table takes ranges over all structures.
/// - `ReportOnly` returns the per-structure results only.
///
/// A structure whose evidence cannot answer the contract keeps its mass as unevaluated, with
/// the registered reason code.
///
/// # Errors
/// As [`evaluate_preference_contract`]; no structures, blank or repeated ids, probabilities
/// for some structures only, invalid probabilities or probabilities above one in total, Bayes
/// weighting without probabilities, and the worst case over structures of expected regret.
pub fn evaluate_preference_structural(
    contract: &PreferenceContract,
    atoms: &[StructuralAtom],
    measure: Option<&PreferenceMeasure>,
    options: PreferenceDecisionOptions,
) -> Result<StructuralPreferenceResult, PreferenceDecisionError> {
    across_structures(contract, atoms, measure, options, None)
}

/// [`evaluate_preference_structural`] under an execution context: the context's cancellation
/// token is read before every structure, every linear program, every candidate vertex basis and
/// every 1024 draws of a draw share.
///
/// # Errors
/// As [`evaluate_preference_structural`]; [`PreferenceDecisionError::Cancelled`] once the token
/// is set.
pub fn evaluate_preference_structural_with_context(
    contract: &PreferenceContract,
    atoms: &[StructuralAtom],
    measure: Option<&PreferenceMeasure>,
    options: PreferenceDecisionOptions,
    ctx: &ExecutionContext,
) -> Result<StructuralPreferenceResult, PreferenceDecisionError> {
    across_structures(contract, atoms, measure, options, Some(ctx))
}

#[allow(
    clippy::too_many_lines,
    reason = "validate, analyse each structure, keep the masses, combine"
)]
pub(crate) fn across_structures(
    contract: &PreferenceContract,
    atoms: &[StructuralAtom],
    measure: Option<&PreferenceMeasure>,
    options: PreferenceDecisionOptions,
    ctx: Option<&ExecutionContext>,
) -> Result<StructuralPreferenceResult, PreferenceDecisionError> {
    let prepared = Prepared::new(contract, measure, options, ctx)?;
    let policy = contract.structural_policy;
    if policy == StructuralPolicy::Maximin && prepared.regret {
        return Err(E::CertifiedRouteRequired {
            reason: CertifiedRouteReason::WorstCaseOfExpectedRegret,
        });
    }
    let has_probabilities = check_structures(atoms)?;
    if policy == StructuralPolicy::BayesOverStructures && !has_probabilities {
        return Err(E::StructureProbabilitiesRequired);
    }
    let facts =
        atoms.iter().map(|atom| prepared.atom_facts(atom)).collect::<Result<Vec<_>, _>>()?;
    let n = contract.actions.len();
    let admissible: Vec<bool> = (0..n)
        .map(|a| {
            facts.iter().all(|fact| match fact {
                AtomFacts::Evaluated(f) => f.admissible[a],
                _ => true,
            })
        })
        .collect();
    let mut summaries = Vec::with_capacity(atoms.len());
    let mut evaluated: Vec<Evaluated<'_>> = Vec::new();
    for (atom, fact) in atoms.iter().zip(&facts) {
        prepared.observe()?;
        let status = match fact {
            AtomFacts::Evaluated(f) => {
                let (result, analysis) = prepared.single(f, &admissible)?;
                evaluated.push(Evaluated {
                    id: &atom.id,
                    probability: atom.probability,
                    facts: f,
                    analysis,
                    verdict: result.verdict.clone(),
                });
                PreferenceAtomStatus::Evaluated(Box::new(result))
            }
            AtomFacts::Unidentified => PreferenceAtomStatus::Unidentified,
            AtomFacts::Unevaluated(reason) => PreferenceAtomStatus::Unevaluated(reason.clone()),
        };
        summaries.push(PreferenceAtomSummary {
            id: atom.id.clone(),
            probability: atom.probability,
            status,
        });
    }
    let mass = |wanted: fn(&PreferenceAtomStatus) -> bool| -> Option<f64> {
        has_probabilities.then(|| {
            probability_sum(
                summaries.iter().filter(|s| wanted(&s.status)).filter_map(|s| s.probability),
            )
        })
    };
    let evaluated_mass = mass(|s| matches!(s, PreferenceAtomStatus::Evaluated(_)));
    let unidentified_mass = mass(|s| matches!(s, PreferenceAtomStatus::Unidentified));
    let unassigned = if has_probabilities {
        unassigned_probability(atoms.iter().filter_map(|a| a.probability))
    } else {
        0.0
    };
    let unevaluated_mass =
        mass(|s| matches!(s, PreferenceAtomStatus::Unevaluated(_))).map(|m| m + unassigned);
    let state = StructureState {
        all_evaluated: evaluated.len() == atoms.len(),
        unresolved_mass: unidentified_mass.unwrap_or(0.0) + unevaluated_mass.unwrap_or(0.0),
        evaluated_mass: evaluated_mass.unwrap_or(0.0),
    };
    let (combined, verdict) = prepared.combine(policy, &evaluated, &admissible, &state)?;
    Ok(StructuralPreferenceResult {
        contract_identity: prepared.identity.clone(),
        structural_policy: policy,
        preference_policy: contract.preference_policy,
        atoms: summaries,
        combined,
        unidentified_mass,
        unevaluated_mass,
        evaluated_mass,
        verdict,
    })
}

// ---------------------------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------------------------

fn check_structures(atoms: &[StructuralAtom]) -> Result<bool, PreferenceDecisionError> {
    if atoms.is_empty() {
        return Err(E::NoStructures);
    }
    let mut ids = HashSet::new();
    for (position, atom) in atoms.iter().enumerate() {
        if is_blank(&atom.id) {
            return Err(E::StructureIdBlank { position });
        }
        if !ids.insert(atom.id.as_str()) {
            return Err(E::StructureDuplicate { id: atom.id.clone() });
        }
    }
    let with = atoms.iter().filter(|a| a.probability.is_some()).count();
    if with == 0 {
        return Ok(false);
    }
    if with != atoms.len() {
        return Err(E::StructureProbabilitiesPartial);
    }
    if let Some(atom) =
        atoms.iter().find(|a| a.probability.is_some_and(|p| !p.is_finite() || p < 0.0))
    {
        return Err(E::StructureProbabilityInvalid { structure: atom.id.clone() });
    }
    if probability_sum(atoms.iter().filter_map(|a| a.probability)) > 1.0 + 1e-9 {
        return Err(E::StructureProbabilitiesExceedOne);
    }
    Ok(true)
}

/// A measure checked against the set, with normalized weights and, for each point with
/// unresolved parameters, the slice of the set through it.
#[derive(Clone, Debug)]
struct PreparedMeasure {
    points: Vec<Vec<f64>>,
    weights: Vec<f64>,
    unresolved: Vec<usize>,
    slices: Vec<Option<Region>>,
}

#[allow(clippy::too_many_lines, reason = "one check per declared field, in field order")]
fn prepare_measure(
    measure: &PreferenceMeasure,
    set: &PreferenceSet,
    region: &Region,
    lp: LpOptions,
) -> Result<PreparedMeasure, PreferenceDecisionError> {
    let dimension = set.dimension();
    let count = measure.points.len();
    if count == 0 {
        return Err(E::MeasureEmpty);
    }
    let origin_blank = match &measure.origin {
        MeasureOrigin::Declared { assumption_id } => is_blank(assumption_id),
        MeasureOrigin::ChoiceEstimate { estimate_identity } => is_blank(estimate_identity),
    };
    if origin_blank {
        return Err(E::MeasureOriginBlank);
    }
    for (index, point) in measure.points.iter().enumerate() {
        if point.len() != dimension {
            return Err(E::MeasurePointDimension {
                index,
                expected: dimension,
                found: point.len(),
            });
        }
        if point.iter().any(|value| !value.is_finite()) {
            return Err(E::MeasurePointNonFinite { index });
        }
    }
    let weights = match &measure.weights {
        None => vec![1.0 / count as f64; count],
        Some(weights) => {
            if weights.len() != count {
                return Err(E::MeasureWeightCount { expected: count, found: weights.len() });
            }
            if let Some(index) = weights.iter().position(|w| !w.is_finite() || *w < 0.0) {
                return Err(E::MeasureWeightInvalid { index });
            }
            let total: f64 = weights.iter().sum();
            if total <= 0.0 || !total.is_finite() {
                return Err(E::MeasureWeightTotalZero);
            }
            weights.iter().map(|w| w / total).collect()
        }
    };
    if let Some(labels) = &measure.labels {
        if labels.len() != count {
            return Err(E::MeasureLabelCount { expected: count, found: labels.len() });
        }
        let mut seen = HashSet::new();
        for (index, label) in labels.iter().enumerate() {
            if is_blank(label) {
                return Err(E::MeasureLabelBlank { index });
            }
            if !seen.insert(label.as_str()) {
                return Err(E::MeasureLabelDuplicate { label: label.clone() });
            }
        }
    }
    let mut unresolved = Vec::with_capacity(measure.unresolved.len());
    for &index in &measure.unresolved {
        if index >= dimension {
            return Err(E::MeasureUnresolvedUnknown { index });
        }
        if unresolved.contains(&index) {
            return Err(E::MeasureUnresolvedRepeated { index });
        }
        unresolved.push(index);
    }
    unresolved.sort_unstable();
    let mut slices = Vec::with_capacity(count);
    for (index, point) in measure.points.iter().enumerate() {
        let tolerance =
            lp.feasibility_tolerance * (1.0 + point.iter().fold(0.0_f64, |m, v| m.max(v.abs())));
        if unresolved.is_empty() {
            if !set.contains(point, tolerance) {
                return Err(E::MeasurePointOutsideSet { index });
            }
            slices.push(None);
            continue;
        }
        let mut lower = region.lower.clone();
        let mut upper = region.upper.clone();
        for j in (0..dimension).filter(|j| !unresolved.contains(j)) {
            if point[j] < region.lower[j] - tolerance || point[j] > region.upper[j] + tolerance {
                return Err(E::MeasurePointOutsideSet { index });
            }
            let fixed = point[j].clamp(region.lower[j], region.upper[j]);
            lower[j] = fixed;
            upper[j] = fixed;
        }
        let slice = Region {
            rows: region.rows.clone(),
            lower,
            upper,
            cancellation: region.cancellation.clone(),
        };
        if slice.feasible(&[], lp)?.is_none() {
            return Err(E::MeasurePointOutsideSet { index });
        }
        slices.push(Some(slice));
    }
    Ok(PreparedMeasure { points: measure.points.clone(), weights, unresolved, slices })
}

// ---------------------------------------------------------------------------------------------
// The admissible region and its linear programs
// ---------------------------------------------------------------------------------------------

/// `{theta : rows, lower <= theta <= upper}`, every row `coefficients . theta <= rhs`.
#[derive(Clone, Debug)]
struct Region {
    rows: Vec<LinearConstraint>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    /// Read before every linear program over the region.
    cancellation: Option<CancellationToken>,
}

/// An optimum of a function over a region and a point attaining it.
#[derive(Clone, Debug)]
struct Extreme {
    value: f64,
    point: Vec<f64>,
}

/// `f[0] + sum_j f[j + 1] theta_j`.
fn affine(f: &[f64], theta: &[f64]) -> f64 {
    f[0] + f[1..].iter().zip(theta).map(|(c, t)| c * t).sum::<f64>()
}

fn difference(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

fn negated(f: &[f64]) -> Vec<f64> {
    f.iter().map(|v| -v).collect()
}

/// The worst of several affine pieces at a point: `min_s f_s(theta)`.
fn worst_at(pieces: &[Vec<f64>], theta: &[f64]) -> f64 {
    pieces.iter().map(|f| affine(f, theta)).fold(f64::INFINITY, f64::min)
}

impl Region {
    /// The declared ranges as variable bounds and every relation row of the set.
    fn of(set: &PreferenceSet) -> Self {
        let rows = set.inequality_rows();
        let rows = rows
            .coefficients
            .into_iter()
            .zip(rows.rhs)
            .map(|(coefficients, rhs)| LinearConstraint {
                coefficients,
                sense: ConstraintSense::AtMost,
                rhs,
            })
            .collect();
        let (lower, upper) = set.bounds().into_iter().unzip();
        Self { rows, lower, upper, cancellation: None }
    }

    fn dimension(&self) -> usize {
        self.lower.len()
    }

    fn outcome(&self, program: &LinearProgram, lp: LpOptions) -> Result<Option<LpSolution>, E> {
        if self.cancellation.as_ref().is_some_and(CancellationToken::is_cancelled) {
            return Err(E::Cancelled);
        }
        match linear_program::solve(program, lp).map_err(E::Lp)? {
            LpOutcome::Optimal(solution) => Ok(Some(solution)),
            LpOutcome::Infeasible(_) => Ok(None),
            LpOutcome::Unbounded { .. } => Err(E::UnboundedProgram),
        }
    }

    /// A point of the region with `f(theta) <= 0` for each affine `f`, or `None`.
    fn feasible(&self, extra: &[Vec<f64>], lp: LpOptions) -> Result<Option<Vec<f64>>, E> {
        let constant_holds = |f: &Vec<f64>| f[0] <= lp.feasibility_tolerance * (1.0 + f[0].abs());
        let mut constraints = self.rows.clone();
        for f in extra {
            if f[1..].iter().all(|c| c.abs().total_cmp(&0.0).is_eq()) {
                if !constant_holds(f) {
                    return Ok(None);
                }
                continue;
            }
            constraints.push(LinearConstraint {
                coefficients: f[1..].to_vec(),
                sense: ConstraintSense::AtMost,
                rhs: -f[0],
            });
        }
        if self.dimension() == 0 {
            return Ok(Some(Vec::new()));
        }
        let program = LinearProgram {
            objective: vec![0.0; self.dimension()],
            constraints,
            lower: self.lower.clone(),
            upper: self.upper.clone(),
        };
        Ok(self.outcome(&program, lp)?.map(|solution| solution.x))
    }

    /// `min over the region of max_k f_k(theta)` and a minimising point.
    fn min_of_max(&self, pieces: &[Vec<f64>], lp: LpOptions) -> Result<Extreme, E> {
        let n = self.dimension();
        let point = if n == 0 {
            Vec::new()
        } else if let [f] = pieces {
            let program = LinearProgram {
                objective: f[1..].to_vec(),
                constraints: self.rows.clone(),
                lower: self.lower.clone(),
                upper: self.upper.clone(),
            };
            self.outcome(&program, lp)?.ok_or(E::UnexpectedInfeasible)?.x
        } else {
            // Epigraph form: minimise t subject to f_k(theta) <= t.
            let mut constraints: Vec<LinearConstraint> = self
                .rows
                .iter()
                .map(|row| {
                    let mut coefficients = row.coefficients.clone();
                    coefficients.push(0.0);
                    LinearConstraint { coefficients, sense: ConstraintSense::AtMost, rhs: row.rhs }
                })
                .collect();
            for f in pieces {
                let mut coefficients = f[1..].to_vec();
                coefficients.push(-1.0);
                constraints.push(LinearConstraint {
                    coefficients,
                    sense: ConstraintSense::AtMost,
                    rhs: -f[0],
                });
            }
            let mut objective = vec![0.0; n];
            objective.push(1.0);
            let mut lower = self.lower.clone();
            lower.push(f64::NEG_INFINITY);
            let mut upper = self.upper.clone();
            upper.push(f64::INFINITY);
            let program = LinearProgram { objective, constraints, lower, upper };
            let mut x = self.outcome(&program, lp)?.ok_or(E::UnexpectedInfeasible)?.x;
            x.truncate(n);
            x
        };
        let value = pieces.iter().map(|f| affine(f, &point)).fold(f64::NEG_INFINITY, f64::max);
        Ok(Extreme { value, point })
    }

    /// Lowest and highest value of each coordinate over the region: two linear programs per
    /// coordinate.
    fn parameter_ranges(&self, lp: LpOptions) -> Result<Vec<(f64, f64)>, E> {
        let n = self.dimension();
        (0..n)
            .map(|j| {
                let mut coordinate = vec![0.0; n + 1];
                coordinate[j + 1] = 1.0;
                let pieces = std::slice::from_ref(&coordinate);
                Ok((self.min_of_max(pieces, lp)?.value, self.max_of_min(pieces, lp)?.value))
            })
            .collect()
    }

    /// `max over the region of min_k f_k(theta)` and a maximising point.
    fn max_of_min(&self, pieces: &[Vec<f64>], lp: LpOptions) -> Result<Extreme, E> {
        let flipped: Vec<Vec<f64>> = pieces.iter().map(|f| negated(f)).collect();
        let extreme = self.min_of_max(&flipped, lp)?;
        Ok(Extreme { value: -extreme.value, point: extreme.point })
    }

    /// `min over the region of (g_a - g_b)` with `g = min over pieces`:
    /// `min_s min_theta max_s' (v_a(s) - v_b(s'))`.
    fn min_margin(&self, a: &[Vec<f64>], b: &[Vec<f64>], lp: LpOptions) -> Result<Extreme, E> {
        let mut best: Option<Extreme> = None;
        for va in a {
            let pieces: Vec<Vec<f64>> = b.iter().map(|vb| difference(va, vb)).collect();
            let extreme = self.min_of_max(&pieces, lp)?;
            if best.as_ref().is_none_or(|current| extreme.value < current.value) {
                best = Some(extreme);
            }
        }
        best.ok_or(E::UnexpectedInfeasible)
    }

    /// `max over the region of (g_a - g_b)`: `max_s' max_theta min_s (v_a(s) - v_b(s'))`.
    fn max_margin(&self, a: &[Vec<f64>], b: &[Vec<f64>], lp: LpOptions) -> Result<Extreme, E> {
        let mut best: Option<Extreme> = None;
        for vb in b {
            let pieces: Vec<Vec<f64>> = a.iter().map(|va| difference(va, vb)).collect();
            let extreme = self.max_of_min(&pieces, lp)?;
            if best.as_ref().is_none_or(|current| extreme.value > current.value) {
                best = Some(extreme);
            }
        }
        best.ok_or(E::UnexpectedInfeasible)
    }
}

// ---------------------------------------------------------------------------------------------
// Analysis of scores over the region
// ---------------------------------------------------------------------------------------------

/// Range of `g_a - g_b` over the region with the points attaining it.
#[derive(Clone, Debug)]
struct Margin {
    min: f64,
    min_at: Vec<f64>,
    max: f64,
    max_at: Vec<f64>,
}

impl Margin {
    fn flipped(&self) -> Self {
        Self {
            min: -self.max,
            min_at: self.max_at.clone(),
            max: -self.min,
            max_at: self.min_at.clone(),
        }
    }
}

/// Pairwise margins, stored for `a < b`.
#[derive(Clone, Debug)]
struct MarginTable {
    n: usize,
    entries: Vec<Option<Margin>>,
}

impl MarginTable {
    fn new(n: usize) -> Self {
        Self { n, entries: vec![None; n * n] }
    }

    fn get(&self, a: usize, b: usize) -> Option<Margin> {
        if a < b {
            self.entries[a * self.n + b].clone()
        } else {
            self.entries[b * self.n + a].as_ref().map(Margin::flipped)
        }
    }
}

/// Everything computed about a set of scores over the region, in larger-is-better
/// orientation.
#[derive(Clone, Debug)]
struct Analysis {
    value: Vec<Option<(f64, f64)>>,
    margins: MarginTable,
    standing: Standing,
    possibly: Vec<Option<bool>>,
    regret: Vec<Option<f64>>,
    share: Vec<Option<f64>>,
    measure_value: Vec<Option<f64>>,
    unresolved_share: Option<f64>,
}

/// How each admissible action stands against its rivals over the whole region, read from the
/// margin ranges.
#[derive(Clone, Debug)]
struct Standing {
    /// Margin over every rival at least `-tolerance` everywhere.
    never_beaten: Vec<bool>,
    /// Margin over every rival above `tolerance` everywhere.
    uniquely: Vec<bool>,
    /// Rivals whose margin over the action exceeds `tolerance` everywhere.
    dominated_by: Vec<Vec<usize>>,
    /// Rivals whose margin range against the action touches zero within `tolerance`.
    touching: Vec<Vec<usize>>,
}

impl Standing {
    /// Read from the margin table over the active actions. With no rival an action is
    /// vacuously never beaten and uniquely best.
    fn of(margins: &MarginTable, active: &[usize], n: usize, tolerance: f64) -> Self {
        let mut standing = Self {
            never_beaten: vec![false; n],
            uniquely: vec![false; n],
            dominated_by: vec![Vec::new(); n],
            touching: vec![Vec::new(); n],
        };
        for &a in active {
            let mut never_beaten = true;
            let mut uniquely = true;
            for &b in active.iter().filter(|&&b| b != a) {
                let Some(margin) = margins.get(a, b) else { continue };
                never_beaten &= margin.min >= -tolerance;
                uniquely &= margin.min > tolerance;
                if margin.max < -tolerance {
                    standing.dominated_by[a].push(b);
                }
                if margin.min <= tolerance && margin.max >= -tolerance {
                    standing.touching[a].push(b);
                }
            }
            standing.never_beaten[a] = never_beaten;
            standing.uniquely[a] = uniquely;
        }
        standing
    }

    fn empty(n: usize) -> Self {
        Self {
            never_beaten: vec![false; n],
            uniquely: vec![false; n],
            dominated_by: vec![Vec::new(); n],
            touching: vec![Vec::new(); n],
        }
    }
}

/// The unique best index by `score`, or every index within `tolerance` of the best.
fn pick(
    active: &[usize],
    score: impl Fn(usize) -> f64,
    tolerance: f64,
) -> Result<usize, Vec<usize>> {
    let best = active.iter().map(|&a| score(a)).fold(f64::NEG_INFINITY, f64::max);
    let tied: Vec<usize> =
        active.iter().copied().filter(|&a| score(a) >= best - tolerance).collect();
    match tied.as_slice() {
        [only] => Ok(*only),
        _ => Err(tied),
    }
}

// ---------------------------------------------------------------------------------------------
// Per-source facts
// ---------------------------------------------------------------------------------------------

/// Per-draw values of every part, oriented: `terms[a][j][i]`.
#[derive(Clone, Debug)]
struct DrawTerms {
    terms: Vec<Vec<Vec<f64>>>,
    weights: Vec<f64>,
}

/// What one source says about the actions.
#[derive(Clone, Debug)]
struct SourceFacts {
    coefficients: Vec<Vec<f64>>,
    per_draw: Option<DrawTerms>,
    admissible: Vec<bool>,
    exclusions: Vec<Vec<ConstraintExclusion>>,
    receipt: SourceReceipt,
    n_draws: usize,
}

enum AtomFacts {
    Evaluated(Box<SourceFacts>),
    Unidentified,
    Unevaluated(String),
}

/// One evaluated structure, with its own analysis under the overall exclusions.
struct Evaluated<'a> {
    id: &'a str,
    probability: Option<f64>,
    facts: &'a SourceFacts,
    analysis: Analysis,
    verdict: PreferenceVerdict,
}

struct StructureState {
    all_evaluated: bool,
    unresolved_mass: f64,
    evaluated_mass: f64,
}

fn read_exclusions(result: DecisionResult) -> (Vec<bool>, Vec<Vec<ConstraintExclusion>>) {
    result.actions.into_iter().map(|outcome| (outcome.admissible, outcome.exclusions)).unzip()
}

/// Each field lists the distinct values of the receipts, in order, joined by `+`.
fn joined(receipts: &[&SourceReceipt]) -> SourceReceipt {
    let join = |field: fn(&SourceReceipt) -> &str| {
        let mut values: Vec<&str> = Vec::new();
        for receipt in receipts {
            let value = field(receipt);
            if !values.contains(&value) {
                values.push(value);
            }
        }
        values.join("+")
    };
    SourceReceipt {
        provider_id: join(|r| r.provider_id.as_str()),
        snapshot_id: join(|r| r.snapshot_id.as_str()),
        rng_id: join(|r| r.rng_id.as_str()),
        causal_contract_id: join(|r| r.causal_contract_id.as_str()),
    }
}

/// Exclusions of each action across structures, first occurrence of each constraint kept.
fn merged_exclusions(evaluated: &[Evaluated<'_>], n: usize) -> Vec<Vec<ConstraintExclusion>> {
    let mut merged: Vec<Vec<ConstraintExclusion>> = vec![Vec::new(); n];
    for atom in evaluated {
        for (a, list) in atom.facts.exclusions.iter().enumerate() {
            for exclusion in list {
                if !merged[a].iter().any(|e| e.constraint_id == exclusion.constraint_id) {
                    merged[a].push(exclusion.clone());
                }
            }
        }
    }
    merged
}

/// The verdict shared by every structure, if they agree. Measure choices of one action agree
/// with the smallest share. A unique winner in some structures that is never beaten (with ties)
/// in the others is never beaten across structures, tying with every rival it ties anywhere,
/// listed in the declaration order `order`.
fn agreed(evaluated: &[Evaluated<'_>], order: &[String]) -> Option<PreferenceVerdict> {
    let first = &evaluated.first()?.verdict;
    if evaluated.iter().all(|atom| &atom.verdict == first) {
        return Some(first.clone());
    }
    let unbeaten = |verdict: &PreferenceVerdict| match verdict {
        PreferenceVerdict::NecessarilyBest(action) => Some((action.clone(), Vec::new())),
        PreferenceVerdict::NeverBeaten { action, ties_with } => {
            Some((action.clone(), ties_with.clone()))
        }
        _ => None,
    };
    if let Some((action, _)) = unbeaten(first) {
        let mut ties: Vec<String> = Vec::new();
        for atom in evaluated {
            let (other, with) = unbeaten(&atom.verdict)?;
            if other != action {
                return None;
            }
            ties.extend(with);
        }
        let ties_with: Vec<String> =
            order.iter().filter(|id| ties.contains(*id)).cloned().collect();
        return Some(PreferenceVerdict::NeverBeaten { action, ties_with });
    }
    if let PreferenceVerdict::MeasureChoice { action, .. } = first {
        let mut smallest = f64::INFINITY;
        for atom in evaluated {
            match &atom.verdict {
                PreferenceVerdict::MeasureChoice { action: other, share } if other == action => {
                    smallest = smallest.min(*share);
                }
                _ => return None,
            }
        }
        return Some(PreferenceVerdict::MeasureChoice { action: action.clone(), share: smallest });
    }
    None
}

/// Ranges over structures: lowest of the lows and highest of the highs. The standing of each
/// action is read from the combined margins, so an action is never beaten (uniquely best,
/// dominated by a rival) across structures exactly when it is in every structure.
fn combine_invariant(analyses: &[&Analysis], admissible: &[bool], tolerance: f64) -> Analysis {
    let n = admissible.len();
    let mut value = vec![None; n];
    let mut margins = MarginTable::new(n);
    let mut possibly = vec![Some(false); n];
    let mut regret: Vec<Option<f64>> = vec![None; n];
    for analysis in analyses {
        for a in 0..n {
            value[a] = match (value[a], analysis.value[a]) {
                (None, next) => next,
                (current, None) => current,
                (Some((lo, hi)), Some((l, h))) => Some((lo.min(l), hi.max(h))),
            };
            possibly[a] = match (possibly[a], analysis.possibly[a]) {
                (Some(true), _) | (_, Some(true)) => Some(true),
                (None, _) | (_, None) => None,
                _ => Some(false),
            };
            regret[a] = match (regret[a], analysis.regret[a]) {
                (None, next) => next,
                (current, None) => current,
                (Some(x), Some(y)) => Some(x.max(y)),
            };
            for b in (a + 1)..n {
                let Some(next) = analysis.margins.get(a, b) else { continue };
                let slot = &mut margins.entries[a * n + b];
                *slot = Some(match slot.take() {
                    None => next,
                    Some(mut current) => {
                        if next.min < current.min {
                            current.min = next.min;
                            current.min_at = next.min_at;
                        }
                        if next.max > current.max {
                            current.max = next.max;
                            current.max_at = next.max_at;
                        }
                        current
                    }
                });
            }
        }
    }
    let standing = if analyses.is_empty() {
        Standing::empty(n)
    } else {
        let active: Vec<usize> = (0..n).filter(|&a| admissible[a]).collect();
        Standing::of(&margins, &active, n, tolerance)
    };
    Analysis {
        value,
        margins,
        standing,
        possibly,
        regret,
        share: vec![None; n],
        measure_value: vec![None; n],
        unresolved_share: None,
    }
}

// ---------------------------------------------------------------------------------------------
// The evaluation
// ---------------------------------------------------------------------------------------------

/// A validated contract with its region, term contract and measure.
struct Prepared<'a> {
    contract: &'a PreferenceContract,
    identity: String,
    terms: DecisionContract,
    region: Region,
    interior: Vec<f64>,
    measure: Option<PreparedMeasure>,
    assumptions: Vec<String>,
    parameter_ids: Vec<String>,
    parameter_ranges: Vec<(f64, f64)>,
    sign: f64,
    regret: bool,
    loss: bool,
    options: PreferenceDecisionOptions,
    /// The caller's execution context, when the run is cancellable.
    ctx: Option<&'a ExecutionContext>,
}

/// The parts of a result that differ between a single source and a combination.
struct Assembly<'a> {
    analysis: &'a Analysis,
    admissible: &'a [bool],
    exclusions: Vec<Vec<ConstraintExclusion>>,
    coefficients: Vec<Option<Vec<f64>>>,
    pieces: Option<&'a [Vec<Vec<f64>>]>,
    source: SourceReceipt,
    n_draws: usize,
}

impl<'a> Prepared<'a> {
    #[allow(clippy::too_many_lines, reason = "one check per refusal, in the documented order")]
    fn new(
        contract: &'a PreferenceContract,
        measure: Option<&PreferenceMeasure>,
        options: PreferenceDecisionOptions,
        ctx: Option<&'a ExecutionContext>,
    ) -> Result<Self, E> {
        if !options.tie_tolerance.is_finite() || options.tie_tolerance < 0.0 {
            return Err(E::InvalidOptions { option: "tie_tolerance" });
        }
        let identity = contract.identity().map_err(E::Contract)?;
        let (sign, regret, loss) = match contract.criterion {
            DecisionCriterion::PosteriorExpectedUtility => (1.0, false, false),
            DecisionCriterion::PosteriorExpectedLoss => (-1.0, false, true),
            DecisionCriterion::ExpectedRegret => (1.0, true, false),
            other => {
                return Err(E::CertifiedRouteRequired {
                    reason: CertifiedRouteReason::Criterion(other),
                });
            }
        };
        let set = &contract.model.preferences;
        if contract.model.linear_form().is_none() {
            let parameters = contract
                .model
                .shape_parameters()
                .into_iter()
                .filter_map(|j| set.parameters.get(j).map(|p| p.id.clone()))
                .collect();
            return Err(E::CertifiedRouteRequired {
                reason: CertifiedRouteReason::ShapeParameters { parameters },
            });
        }
        if !set.value_relations.is_empty() {
            return Err(E::CertifiedRouteRequired {
                reason: CertifiedRouteReason::ValueRelations {
                    relations: set.value_relations.iter().map(|r| r.id.clone()).collect(),
                },
            });
        }
        let terms = contract
            .term_contract()
            .map_err(E::Contract)?
            .ok_or(E::CertifiedRouteRequired { reason: CertifiedRouteReason::PartNotLowerable })?;
        let lp = options.lp;
        let mut region = Region::of(set);
        region.cancellation = ctx.map(|c| c.cancellation.clone());
        let interior = region.feasible(&[], lp)?.ok_or(E::EmptyPreferenceSet)?;
        let measure_prepared = measure.map(|m| prepare_measure(m, set, &region, lp)).transpose()?;
        match contract.preference_policy {
            PreferencePolicy::Weighted => match &measure_prepared {
                None => return Err(E::MeasureRequired),
                Some(m) if !m.unresolved.is_empty() => {
                    return Err(E::MeasureIncomplete {
                        parameters: m
                            .unresolved
                            .iter()
                            .map(|&j| set.parameters[j].id.clone())
                            .collect(),
                    });
                }
                Some(_) => {}
            },
            PreferencePolicy::Maximin if regret => {
                return Err(E::CertifiedRouteRequired {
                    reason: CertifiedRouteReason::WorstCaseOfExpectedRegret,
                });
            }
            _ => {}
        }
        let mut assumptions: Vec<String> =
            set.parameters.iter().map(|p| p.assumption_id.clone()).collect();
        for relation in &set.relations {
            assumptions.push(match &relation.origin {
                RelationOrigin::Declared => format!("declared relation {}", relation.id),
                RelationOrigin::ShapeRule => format!("shape rule {}", relation.id),
                RelationOrigin::Choice { record_id, rejected_option } => {
                    format!("recorded choice {record_id} over {rejected_option}")
                }
            });
        }
        if let Some(m) = measure {
            assumptions.push(match &m.origin {
                MeasureOrigin::Declared { assumption_id } => assumption_id.clone(),
                MeasureOrigin::ChoiceEstimate { estimate_identity } => {
                    format!("choice estimate {estimate_identity}")
                }
            });
        }
        assumptions.sort();
        assumptions.dedup();
        let parameter_ranges = region.parameter_ranges(lp)?;
        Ok(Self {
            contract,
            identity,
            terms,
            region,
            interior,
            measure: measure_prepared,
            assumptions,
            parameter_ids: set.parameters.iter().map(|p| p.id.clone()).collect(),
            parameter_ranges,
            sign,
            regret,
            loss,
            options,
            ctx,
        })
    }

    /// `Err(Cancelled)` once the caller's token is set.
    fn observe(&self) -> Result<(), E> {
        if self.ctx.is_some_and(|ctx| ctx.cancellation.is_cancelled()) {
            return Err(E::Cancelled);
        }
        Ok(())
    }

    fn parts(&self) -> usize {
        1 + self.region.dimension()
    }

    fn draw_facts(&self, source: &DistributionArtifact) -> Result<SourceFacts, E> {
        let draws = utility_draws(&self.terms, source).map_err(E::Eval)?;
        let k = self.parts();
        let mut coefficients = Vec::with_capacity(self.contract.actions.len());
        let mut terms = Vec::with_capacity(self.contract.actions.len());
        for (a, action) in self.contract.actions.iter().enumerate() {
            let mut row = Vec::with_capacity(k);
            let mut parts = Vec::with_capacity(k);
            for j in 0..k {
                let values: Vec<f64> =
                    draws.utilities[a * k + j].iter().map(|v| self.sign * v).collect();
                let mean: f64 = values.iter().zip(&draws.weights).map(|(v, w)| v * w).sum();
                if !mean.is_finite() {
                    return Err(E::NonFiniteCoefficient { action: action.id.clone() });
                }
                row.push(mean);
                parts.push(values);
            }
            coefficients.push(row);
            terms.push(parts);
        }
        let (admissible, exclusions) = if self.contract.constraints.is_empty() {
            let n = self.contract.actions.len();
            (vec![true; n], vec![Vec::new(); n])
        } else {
            let bound = self.contract.bind(&self.interior).map_err(E::Contract)?;
            read_exclusions(evaluate_contract(&bound, source).map_err(E::Eval)?)
        };
        let identity = &source.metadata().identity;
        Ok(SourceFacts {
            coefficients,
            per_draw: Some(DrawTerms { terms, weights: draws.weights.clone() }),
            admissible,
            exclusions,
            receipt: SourceReceipt {
                provider_id: identity.provider_id.clone(),
                snapshot_id: identity.snapshot_id.clone(),
                rng_id: identity.rng_id.clone(),
                causal_contract_id: identity.causal_contract_id.clone(),
            },
            n_draws: draws.weights.len(),
        })
    }

    fn mean_facts(&self, source: &MeanSource) -> Result<SourceFacts, E> {
        if self.regret {
            // Expected regret compares actions draw by draw, so a mean cannot answer it: the
            // ordinary evaluator's refusal is surfaced.
            let mut ordinary = self.terms.clone();
            ordinary.criterion = DecisionCriterion::ExpectedRegret;
            evaluate_contract_on_means(&ordinary, source).map_err(E::Eval)?;
            return Err(E::Eval(DecisionEvalError::Contract(
                DecisionContractError::InvalidDeclaration("expected regret needs aligned draws"),
            )));
        }
        let result = evaluate_contract_on_means(&self.terms, source).map_err(E::Eval)?;
        let k = self.parts();
        let mut coefficients = Vec::with_capacity(self.contract.actions.len());
        for (a, action) in self.contract.actions.iter().enumerate() {
            let row: Vec<f64> =
                (0..k).map(|j| self.sign * result.actions[a * k + j].expected_utility).collect();
            if row.iter().any(|v| !v.is_finite()) {
                return Err(E::NonFiniteCoefficient { action: action.id.clone() });
            }
            coefficients.push(row);
        }
        let (admissible, exclusions) = if self.contract.constraints.is_empty() {
            let n = self.contract.actions.len();
            (vec![true; n], vec![Vec::new(); n])
        } else {
            let bound = self.contract.bind(&self.interior).map_err(E::Contract)?;
            read_exclusions(evaluate_contract_on_means(&bound, source).map_err(E::Eval)?)
        };
        Ok(SourceFacts {
            coefficients,
            per_draw: None,
            admissible,
            exclusions,
            receipt: result.source,
            n_draws: 0,
        })
    }

    fn atom_facts(&self, atom: &StructuralAtom) -> Result<AtomFacts, E> {
        Ok(match &atom.evidence {
            AtomEvidence::Unidentified => AtomFacts::Unidentified,
            AtomEvidence::Unevaluated(reason) => AtomFacts::Unevaluated(reason.clone()),
            AtomEvidence::Evaluated(source) => match self.draw_facts(source) {
                Ok(facts) => AtomFacts::Evaluated(Box::new(facts)),
                Err(E::Eval(DecisionEvalError::Contract(error))) => {
                    return Err(E::Eval(DecisionEvalError::Contract(error)));
                }
                // Evidence that cannot answer the contract keeps its mass as unevaluated.
                Err(E::Eval(other)) => AtomFacts::Unevaluated(other.reason_code().to_owned()),
                Err(other) => return Err(other),
            },
        })
    }

    /// The full analysis of one source under the given admissible set.
    fn single(
        &self,
        facts: &SourceFacts,
        admissible: &[bool],
    ) -> Result<(PreferenceDecisionResult, Analysis), E> {
        let pieces: Vec<Vec<Vec<f64>>> =
            facts.coefficients.iter().map(|c| vec![c.clone()]).collect();
        let analysis = self.analyse(&pieces, admissible)?;
        let mut result = self.assemble(Assembly {
            analysis: &analysis,
            admissible,
            exclusions: facts.exclusions.clone(),
            coefficients: facts.coefficients.iter().cloned().map(Some).collect(),
            pieces: Some(&pieces),
            source: facts.receipt.clone(),
            n_draws: facts.n_draws,
        });
        result.draw_share_best_everywhere = self.draw_share(facts, &analysis, admissible)?;
        Ok((result, analysis))
    }

    /// Analyse scores `g_a = min over pieces[a]` of affine functions over the region.
    fn analyse(&self, pieces: &[Vec<Vec<f64>>], admissible: &[bool]) -> Result<Analysis, E> {
        let n = pieces.len();
        let lp = self.options.lp;
        let tolerance = self.options.tie_tolerance;
        let active: Vec<usize> = (0..n).filter(|&a| admissible[a]).collect();
        let mut value = vec![None; n];
        for &a in &active {
            let mut lowest = f64::INFINITY;
            for piece in &pieces[a] {
                lowest = lowest.min(self.region.min_of_max(std::slice::from_ref(piece), lp)?.value);
            }
            let highest = self.region.max_of_min(&pieces[a], lp)?.value;
            value[a] = Some((lowest, highest));
        }
        let mut margins = MarginTable::new(n);
        for (i, &a) in active.iter().enumerate() {
            for &b in &active[i + 1..] {
                let low = self.region.min_margin(&pieces[a], &pieces[b], lp)?;
                let high = self.region.max_margin(&pieces[a], &pieces[b], lp)?;
                margins.entries[a * n + b] = Some(Margin {
                    min: low.value,
                    min_at: low.point,
                    max: high.value,
                    max_at: high.point,
                });
            }
        }
        let standing = Standing::of(&margins, &active, n, tolerance);
        let mut regret = vec![None; n];
        let mut possibly = vec![Some(false); n];
        for &a in &active {
            let mut worst = 0.0_f64;
            for &b in active.iter().filter(|&&b| b != a) {
                if let Some(margin) = margins.get(a, b) {
                    worst = worst.max(-margin.min);
                }
            }
            regret[a] = Some(worst);
            possibly[a] = self.possibly_best(pieces, &active, a)?;
        }
        let (share, measure_value, unresolved_share) = match &self.measure {
            None => (vec![None; n], vec![None; n], None),
            Some(measure) => self.measure_summary(measure, pieces, &active)?,
        };
        Ok(Analysis {
            value,
            margins,
            standing,
            possibly,
            regret,
            share,
            measure_value,
            unresolved_share,
        })
    }

    /// Whether some admissible point makes `a` at least as good as every rival. Under a worst
    /// case over pieces this needs, for each rival, one piece `s'` with
    /// `v_a(s) >= v_b(s')` for every piece `s` of `a`; every choice of pieces is tried.
    fn possibly_best(
        &self,
        pieces: &[Vec<Vec<f64>>],
        active: &[usize],
        a: usize,
    ) -> Result<Option<bool>, E> {
        let rivals: Vec<usize> = active.iter().copied().filter(|&b| b != a).collect();
        let choices =
            rivals.iter().try_fold(1_usize, |count, &b| count.checked_mul(pieces[b].len()));
        if choices.is_none_or(|count| count > MAX_STRUCTURE_CHOICES) {
            return Ok(None);
        }
        let mut choice = vec![0_usize; rivals.len()];
        loop {
            let mut rows = Vec::new();
            for (r, &b) in rivals.iter().enumerate() {
                for va in &pieces[a] {
                    rows.push(difference(&pieces[b][choice[r]], va));
                }
            }
            if self.region.feasible(&rows, self.options.lp)?.is_some() {
                return Ok(Some(true));
            }
            let mut position = 0;
            loop {
                if position == rivals.len() {
                    return Ok(Some(false));
                }
                choice[position] += 1;
                if choice[position] < pieces[rivals[position]].len() {
                    break;
                }
                choice[position] = 0;
                position += 1;
            }
        }
    }

    /// Shares and weighted values of the measure. A fully resolved point goes to its leader
    /// (shared equally among actions within the tie tolerance of the best); a point with
    /// unresolved parameters goes to an action only if it leads by more than the tolerance over
    /// the whole slice, otherwise to the unresolved share.
    #[allow(clippy::type_complexity, reason = "three parallel per-action columns")]
    fn measure_summary(
        &self,
        measure: &PreparedMeasure,
        pieces: &[Vec<Vec<f64>>],
        active: &[usize],
    ) -> Result<(Vec<Option<f64>>, Vec<Option<f64>>, Option<f64>), E> {
        let n = pieces.len();
        let tolerance = self.options.tie_tolerance;
        let mut share = vec![0.0; n];
        let mut totals = vec![0.0; n];
        let mut unresolved = 0.0;
        for ((point, weight), slice) in
            measure.points.iter().zip(&measure.weights).zip(&measure.slices)
        {
            match slice {
                None => {
                    let scores: Vec<(usize, f64)> =
                        active.iter().map(|&a| (a, worst_at(&pieces[a], point))).collect();
                    let best = scores.iter().map(|s| s.1).fold(f64::NEG_INFINITY, f64::max);
                    let leaders: Vec<usize> =
                        scores.iter().filter(|s| s.1 >= best - tolerance).map(|s| s.0).collect();
                    for &leader in &leaders {
                        share[leader] += weight / leaders.len() as f64;
                    }
                    for (a, score) in scores {
                        totals[a] += weight * score;
                    }
                }
                Some(slice) => {
                    let mut leader = None;
                    'candidates: for &a in active {
                        for &b in active.iter().filter(|&&b| b != a) {
                            let low = slice.min_margin(&pieces[a], &pieces[b], self.options.lp)?;
                            if low.value <= tolerance {
                                continue 'candidates;
                            }
                        }
                        leader = Some(a);
                        break;
                    }
                    match leader {
                        Some(a) => share[a] += weight,
                        None => unresolved += weight,
                    }
                }
            }
        }
        let complete = measure.unresolved.is_empty();
        let mut shares = vec![None; n];
        let mut values = vec![None; n];
        for &a in active {
            shares[a] = Some(share[a]);
            values[a] = complete.then_some(totals[a]);
        }
        Ok((shares, values, Some(unresolved)))
    }

    /// Weight of draws where the only never-beaten action has a per-draw utility at least every
    /// rival's at every vertex of the set. Per-draw utility is affine in the parameters, so the
    /// vertices decide the whole set.
    fn draw_share(
        &self,
        facts: &SourceFacts,
        analysis: &Analysis,
        admissible: &[bool],
    ) -> Result<Option<f64>, E> {
        let Some(per_draw) = &facts.per_draw else {
            return Ok(None);
        };
        let n = admissible.len();
        let unbeaten: Vec<usize> =
            (0..n).filter(|&a| admissible[a] && analysis.standing.never_beaten[a]).collect();
        let [best] = unbeaten.as_slice() else {
            return Ok(None);
        };
        let best = *best;
        let vertices = if self.region.dimension() == 0 {
            vec![Vec::new()]
        } else {
            let (rows, lower, upper) = (&self.region.rows, &self.region.lower, &self.region.upper);
            let (cap, lp) = (self.options.vertex_cap, self.options.lp);
            let enumerated = match self.ctx {
                Some(ctx) => {
                    linear_program::vertices_with_context(rows, lower, upper, cap, lp, ctx)
                }
                None => linear_program::vertices(rows, lower, upper, cap, lp),
            };
            match enumerated {
                Ok(vertices) => vertices,
                Err(LpError::VertexCapExceeded { .. }) => return Ok(None),
                Err(LpError::Cancelled) => return Err(E::Cancelled),
                Err(error) => return Err(E::Lp(error)),
            }
        };
        let rivals: Vec<usize> = (0..n).filter(|&b| admissible[b] && b != best).collect();
        let mut share = 0.0;
        for (i, weight) in per_draw.weights.iter().enumerate() {
            if i % DRAWS_PER_CHECK == 0 {
                self.observe()?;
            }
            let utility = |a: usize, theta: &[f64]| -> f64 {
                let parts = &per_draw.terms[a];
                parts[0][i]
                    + theta.iter().zip(&parts[1..]).map(|(t, part)| t * part[i]).sum::<f64>()
            };
            let everywhere = vertices.iter().all(|theta| {
                let own = utility(best, theta);
                rivals.iter().all(|&b| own >= utility(b, theta))
            });
            if everywhere {
                share += weight;
            }
        }
        Ok(Some(share))
    }

    fn decide(&self, analysis: &Analysis, admissible: &[bool]) -> PreferenceVerdict {
        let ids = |list: Vec<usize>| -> Vec<String> {
            list.into_iter().map(|a| self.contract.actions[a].id.clone()).collect()
        };
        let id = |a: usize| self.contract.actions[a].id.clone();
        let tolerance = self.options.tie_tolerance;
        let active: Vec<usize> = (0..admissible.len()).filter(|&a| admissible[a]).collect();
        if active.is_empty() {
            return PreferenceVerdict::NoAdmissibleAction;
        }
        match self.contract.preference_policy {
            PreferencePolicy::ReportOnly => PreferenceVerdict::ReportOnly,
            PreferencePolicy::RequireInvariantBestAction => {
                self.invariant_verdict(analysis, &active)
            }
            PreferencePolicy::Maximin => {
                let score = |a: usize| analysis.value[a].map_or(f64::NEG_INFINITY, |v| v.0);
                match pick(&active, score, tolerance) {
                    Ok(a) => PreferenceVerdict::WorstCaseChoice(id(a)),
                    Err(tied) => PreferenceVerdict::Tied(ids(tied)),
                }
            }
            PreferencePolicy::MinimaxRegret => {
                let score = |a: usize| -analysis.regret[a].unwrap_or(f64::INFINITY);
                match pick(&active, score, tolerance) {
                    Ok(a) => PreferenceVerdict::MinimaxRegretChoice(id(a)),
                    Err(tied) => PreferenceVerdict::Tied(ids(tied)),
                }
            }
            PreferencePolicy::Weighted => {
                let score = |a: usize| analysis.measure_value[a].unwrap_or(f64::NEG_INFINITY);
                match pick(&active, score, tolerance) {
                    Ok(a) => PreferenceVerdict::MeasureChoice {
                        action: id(a),
                        share: analysis.share[a].unwrap_or(0.0),
                    },
                    Err(tied) => PreferenceVerdict::Tied(ids(tied)),
                }
            }
        }
    }

    /// `NecessarilyBest` for the action that is uniquely best (ahead of every rival by more
    /// than the tolerance everywhere); `NeverBeaten` when exactly one action is never beaten
    /// but it is not uniquely best, naming the rivals whose margin touches zero; `Tied` when
    /// several actions are never beaten; otherwise the possibly-best actions (an action whose
    /// status could not be enumerated is listed).
    ///
    /// Several never-beaten actions are always equal everywhere, so no further case exists:
    /// when `a` and `b` are both never beaten, the margin of `a` over `b` is at least
    /// `-tolerance` everywhere (`a` is never beaten) and at most `tolerance` everywhere (`b` is
    /// never beaten, and its margin over `a` is the negation), so it lies within the tolerance of
    /// zero at every admissible point. A uniquely best action excludes every other never-beaten
    /// action, since its margin over each rival exceeds the tolerance somewhere.
    fn invariant_verdict(&self, analysis: &Analysis, active: &[usize]) -> PreferenceVerdict {
        let id = |a: usize| self.contract.actions[a].id.clone();
        let standing = &analysis.standing;
        if let Some(&a) = active.iter().find(|&&a| standing.uniquely[a]) {
            return PreferenceVerdict::NecessarilyBest(id(a));
        }
        let unbeaten: Vec<usize> =
            active.iter().copied().filter(|&a| standing.never_beaten[a]).collect();
        match unbeaten.as_slice() {
            [] => {}
            [a] => {
                return PreferenceVerdict::NeverBeaten {
                    action: id(*a),
                    ties_with: standing.touching[*a].iter().map(|&b| id(b)).collect(),
                };
            }
            several => return PreferenceVerdict::Tied(several.iter().map(|&a| id(a)).collect()),
        }
        PreferenceVerdict::PreferenceDependent {
            possibly_best: active
                .iter()
                .filter(|&&a| analysis.possibly[a] != Some(false))
                .map(|&a| self.contract.actions[a].id.clone())
                .collect(),
        }
    }

    fn criterion_range(&self, range: (f64, f64)) -> Option<(f64, f64)> {
        if self.regret {
            None
        } else if self.loss {
            Some((-range.1, -range.0))
        } else {
            Some(range)
        }
    }

    /// The single parameter a margin `(constant, per parameter)` depends on and where it is
    /// zero, as [`SwitchThreshold`] documents; `None` for an empty margin, a margin that
    /// depends on several parameters or on none.
    fn threshold(&self, margin: &[f64]) -> Option<SwitchThreshold> {
        let n = self.region.dimension();
        if margin.len() != n + 1 {
            return None;
        }
        let (lower, upper) = (&self.region.lower, &self.region.upper);
        let tolerance = self.options.tie_tolerance;
        let varying: Vec<usize> =
            (0..n).filter(|&j| margin[j + 1].abs() * (upper[j] - lower[j]) > tolerance).collect();
        let [p] = varying.as_slice() else {
            return None;
        };
        let p = *p;
        let rest: f64 = margin[0]
            + (0..n)
                .filter(|&j| j != p)
                .map(|j| margin[j + 1] * 0.5 * (lower[j] + upper[j]))
                .sum::<f64>();
        let slope = margin[p + 1];
        let value = -rest / slope;
        value.is_finite().then(|| SwitchThreshold {
            parameter_id: self.parameter_ids[p].clone(),
            value,
            first_leads_above: slope > 0.0,
        })
    }

    fn assemble(&self, parts: Assembly<'_>) -> PreferenceDecisionResult {
        let Assembly { analysis, admissible, exclusions, coefficients, pieces, source, n_draws } =
            parts;
        let tolerance = self.options.tie_tolerance;
        let n = admissible.len();
        let actions = self
            .contract
            .actions
            .iter()
            .zip(exclusions)
            .zip(coefficients)
            .enumerate()
            .map(|(a, ((action, exclusions), coefficients))| {
                let admitted = admissible[a];
                PreferenceActionOutcome {
                    id: action.id.clone(),
                    admissible: admitted,
                    exclusions,
                    coefficients: coefficients.filter(|_| admitted),
                    value_range: analysis.value[a].and_then(|range| self.criterion_range(range)),
                    necessarily_best: admitted && analysis.standing.never_beaten[a],
                    uniquely_best: admitted && analysis.standing.uniquely[a],
                    dominated_by: if admitted {
                        analysis.standing.dominated_by[a]
                            .iter()
                            .map(|&b| self.contract.actions[b].id.clone())
                            .collect()
                    } else {
                        Vec::new()
                    },
                    possibly_best: if admitted { analysis.possibly[a] } else { Some(false) },
                    worst_case_regret: analysis.regret[a],
                    measure_share: analysis.share[a],
                    measure_value: analysis.measure_value[a]
                        .filter(|_| !self.regret)
                        .map(|value| self.sign * value),
                }
            })
            .collect();
        let mut switches = Vec::new();
        for a in 0..n {
            for b in (a + 1)..n {
                let Some(margin) = analysis.margins.get(a, b) else { continue };
                if margin.min < -tolerance && margin.max > tolerance {
                    let margin_coefficients = match pieces {
                        Some(pieces) => match (pieces[a].as_slice(), pieces[b].as_slice()) {
                            ([va], [vb]) => difference(va, vb),
                            _ => Vec::new(),
                        },
                        None => Vec::new(),
                    };
                    let threshold = self.threshold(&margin_coefficients);
                    switches.push(PreferenceSwitch {
                        first: self.contract.actions[a].id.clone(),
                        second: self.contract.actions[b].id.clone(),
                        margin_coefficients,
                        margin_range: (margin.min, margin.max),
                        favours_first_at: margin.max_at,
                        favours_second_at: margin.min_at,
                        threshold,
                    });
                }
            }
        }
        PreferenceDecisionResult {
            contract_identity: self.identity.clone(),
            policy: self.contract.preference_policy,
            parameter_ids: self.parameter_ids.clone(),
            parameter_ranges: self.parameter_ranges.clone(),
            actions,
            switches,
            verdict: self.decide(analysis, admissible),
            draw_share_best_everywhere: None,
            unresolved_measure_share: analysis.unresolved_share,
            assumptions: self.assumptions.clone(),
            source,
            n_draws,
        }
    }

    /// The combined table and the structural verdict.
    #[allow(clippy::too_many_lines, reason = "one branch per structural policy")]
    fn combine(
        &self,
        policy: StructuralPolicy,
        evaluated: &[Evaluated<'_>],
        admissible: &[bool],
        state: &StructureState,
    ) -> Result<(Option<PreferenceDecisionResult>, StructuralPreferenceVerdict), E> {
        if policy == StructuralPolicy::ReportOnly {
            return Ok((None, StructuralPreferenceVerdict::ReportOnly));
        }
        if evaluated.is_empty() {
            let verdict = if self.contract.preference_policy == PreferencePolicy::ReportOnly {
                StructuralPreferenceVerdict::Combined(PreferenceVerdict::ReportOnly)
            } else {
                StructuralPreferenceVerdict::InsufficientScience(NO_STRUCTURE_EVALUATED.into())
            };
            return Ok((None, verdict));
        }
        let n = admissible.len();
        let receipts: Vec<&SourceReceipt> = evaluated.iter().map(|e| &e.facts.receipt).collect();
        let source = joined(&receipts);
        let n_draws = evaluated.iter().map(|e| e.facts.n_draws).sum();
        let exclusions = merged_exclusions(evaluated, n);
        let report = self.contract.preference_policy == PreferencePolicy::ReportOnly;
        let combined = match policy {
            StructuralPolicy::BayesOverStructures => {
                let mut mixture = vec![vec![0.0; self.parts()]; n];
                for atom in evaluated {
                    let p = atom.probability.unwrap_or(0.0);
                    for (row, own) in mixture.iter_mut().zip(&atom.facts.coefficients) {
                        for (slot, c) in row.iter_mut().zip(own) {
                            *slot += p * c;
                        }
                    }
                }
                let pieces: Vec<Vec<Vec<f64>>> = mixture.iter().map(|c| vec![c.clone()]).collect();
                let analysis = self.analyse(&pieces, admissible)?;
                let result = self.assemble(Assembly {
                    analysis: &analysis,
                    admissible,
                    exclusions,
                    coefficients: mixture.into_iter().map(Some).collect(),
                    pieces: Some(&pieces),
                    source,
                    n_draws,
                });
                let verdict = if report {
                    StructuralPreferenceVerdict::Combined(PreferenceVerdict::ReportOnly)
                } else if state.unresolved_mass > 0.0 {
                    StructuralPreferenceVerdict::InsufficientScience(
                        UNRESOLVED_MASS_CAN_CHANGE_BAYES.into(),
                    )
                } else if state.evaluated_mass <= 0.0 {
                    StructuralPreferenceVerdict::InsufficientScience(
                        NO_EVALUATED_POSITIVE_PROBABILITY.into(),
                    )
                } else {
                    StructuralPreferenceVerdict::Combined(result.verdict.clone())
                };
                (result, verdict)
            }
            StructuralPolicy::Maximin => {
                let pieces: Vec<Vec<Vec<f64>>> = (0..n)
                    .map(|a| {
                        evaluated
                            .iter()
                            .map(|e| e.facts.coefficients[a].clone())
                            .collect::<Vec<_>>()
                    })
                    .collect();
                let analysis = self.analyse(&pieces, admissible)?;
                let result = self.assemble(Assembly {
                    analysis: &analysis,
                    admissible,
                    exclusions,
                    coefficients: vec![None; n],
                    pieces: Some(&pieces),
                    source,
                    n_draws,
                });
                let verdict = if report {
                    StructuralPreferenceVerdict::Combined(PreferenceVerdict::ReportOnly)
                } else if !state.all_evaluated {
                    StructuralPreferenceVerdict::InsufficientScience(
                        WORST_CASE_NEEDS_EVERY_STRUCTURE.into(),
                    )
                } else {
                    StructuralPreferenceVerdict::Combined(result.verdict.clone())
                };
                (result, verdict)
            }
            _ => self.invariant_combined(evaluated, admissible, state, exclusions, source, n_draws),
        };
        Ok((Some(combined.0), combined.1))
    }

    fn invariant_combined(
        &self,
        evaluated: &[Evaluated<'_>],
        admissible: &[bool],
        state: &StructureState,
        exclusions: Vec<Vec<ConstraintExclusion>>,
        source: SourceReceipt,
        n_draws: usize,
    ) -> (PreferenceDecisionResult, StructuralPreferenceVerdict) {
        let n = admissible.len();
        let analyses: Vec<&Analysis> = evaluated.iter().map(|e| &e.analysis).collect();
        let analysis = combine_invariant(&analyses, admissible, self.options.tie_tolerance);
        let mut result = self.assemble(Assembly {
            analysis: &analysis,
            admissible,
            exclusions,
            coefficients: vec![None; n],
            pieces: None,
            source,
            n_draws,
        });
        let order: Vec<String> = self.contract.actions.iter().map(|a| a.id.clone()).collect();
        let agreement = agreed(evaluated, &order);
        result.verdict = agreement.clone().unwrap_or_else(|| {
            if admissible.iter().any(|a| *a) {
                PreferenceVerdict::PreferenceDependent {
                    possibly_best: (0..n)
                        .filter(|&a| admissible[a] && analysis.possibly[a] != Some(false))
                        .map(|a| self.contract.actions[a].id.clone())
                        .collect(),
                }
            } else {
                PreferenceVerdict::NoAdmissibleAction
            }
        });
        let verdict = if self.contract.preference_policy == PreferencePolicy::ReportOnly {
            StructuralPreferenceVerdict::Combined(PreferenceVerdict::ReportOnly)
        } else if !state.all_evaluated {
            StructuralPreferenceVerdict::InsufficientScience(INVARIANCE_UNCHECKED.into())
        } else if let Some(verdict) = agreement {
            StructuralPreferenceVerdict::Combined(verdict)
        } else {
            StructuralPreferenceVerdict::StructuresDisagree(
                evaluated
                    .iter()
                    .map(|e| AtomChoice { atom: e.id.to_owned(), verdict: e.verdict.clone() })
                    .collect(),
            )
        };
        (result, verdict)
    }
}

impl PreferenceDecisionError {
    /// Structured refusal for a failure of the exact evaluation across a preference set.
    ///
    /// A contract or source refusal keeps its own refusal; a linear-program refusal is
    /// reported under `preference_decision.linear_program_refused` with the linear program's
    /// detail and message as the supplied text.
    #[must_use]
    #[allow(clippy::too_many_lines, reason = "one arm per exact-evaluation failure")]
    pub fn to_refusal(&self) -> ExternalRefusal {
        let invalid = |remedy| {
            refusal(antecedent_core::reason_code!("invalid_argument"), EVALUATE, self.detail())
                .remedy(remedy)
        };
        let unsatisfied = |stage, remedy| {
            refusal(
                antecedent_core::reason_code!("decision_contract_unsatisfied"),
                stage,
                self.detail(),
            )
            .remedy(remedy)
        };
        let structural = |remedy| {
            refusal(antecedent_core::reason_code!("invalid_argument"), STRUCTURAL, self.detail())
                .remedy(remedy)
        };
        match self {
            Self::Contract(error) => at_stage(error.to_refusal(), EVALUATE),
            Self::Eval(error) => at_stage(error.to_refusal(), EVALUATE),
            Self::Lp(error) => refusal(
                antecedent_core::reason_code!("invalid_argument"),
                EVALUATE,
                antecedent_core::reason_details::preference_decision::LINEAR_PROGRAM_REFUSED,
            )
            .supplied(format!("{}: {error}", error.detail()))
            .remedy(
                "relax the linear-program options, or rescale the declared ranges and relations",
            )
            .build(),
            Self::CertifiedRouteRequired { reason } => {
                let builder = refusal(
                    antecedent_core::reason_code!("route_not_supported"),
                    EVALUATE,
                    self.detail(),
                )
                .remedy(CERTIFIED_ROUTE);
                match reason {
                    CertifiedRouteReason::Criterion(criterion) => {
                        builder.offending(format!("{criterion:?}")).build()
                    }
                    CertifiedRouteReason::ShapeParameters { parameters } => {
                        builder.offending(parameters.join(",")).build()
                    }
                    CertifiedRouteReason::ValueRelations { relations } => {
                        builder.offending(relations.join(",")).build()
                    }
                    CertifiedRouteReason::PartNotLowerable
                    | CertifiedRouteReason::WorstCaseOfExpectedRegret => builder.build(),
                }
            }
            Self::InvalidOptions { option } => {
                invalid("supply the named option inside its documented range")
                    .offending(*option)
                    .build()
            }
            Self::EmptyPreferenceSet => {
                invalid("widen the declared ranges or remove a relation").build()
            }
            Self::MeasureRequired => unsatisfied(
                EVALUATE,
                "supply a measure over the preference set, or choose another preference policy",
            )
            .build(),
            Self::MeasureIncomplete { parameters } => {
                unsatisfied(EVALUATE, "supply a measure that resolves every parameter")
                    .offending(parameters.join(","))
                    .build()
            }
            Self::MeasureEmpty => invalid("supply at least one measure point").build(),
            Self::MeasurePointDimension { index, expected, found } => {
                invalid("supply one value per preference parameter")
                    .offending(index.to_string())
                    .expected(expected.to_string())
                    .supplied(found.to_string())
                    .build()
            }
            Self::MeasurePointNonFinite { index } => {
                invalid("supply finite measure points").offending(index.to_string()).build()
            }
            Self::MeasurePointOutsideSet { index } => {
                invalid("supply measure points inside the declared ranges and relations")
                    .offending(index.to_string())
                    .build()
            }
            Self::MeasureWeightCount { expected, found } => {
                invalid("supply one weight per measure point")
                    .expected(expected.to_string())
                    .supplied(found.to_string())
                    .build()
            }
            Self::MeasureWeightInvalid { index } => {
                invalid("supply finite, non-negative measure weights")
                    .offending(index.to_string())
                    .build()
            }
            Self::MeasureWeightTotalZero => {
                invalid("give at least one measure point a positive weight").build()
            }
            Self::MeasureLabelCount { expected, found } => {
                invalid("supply one label per measure point")
                    .expected(expected.to_string())
                    .supplied(found.to_string())
                    .build()
            }
            Self::MeasureLabelBlank { index } => {
                invalid("give every measure point a non-blank label")
                    .offending(index.to_string())
                    .build()
            }
            Self::MeasureLabelDuplicate { label } => {
                invalid("give every measure point a distinct label")
                    .offending(label.clone())
                    .build()
            }
            Self::MeasureOriginBlank => {
                invalid("name the assumption or estimate the measure comes from").build()
            }
            Self::MeasureUnresolvedUnknown { index } => {
                invalid("name only declared parameter positions as unresolved")
                    .offending(index.to_string())
                    .build()
            }
            Self::MeasureUnresolvedRepeated { index } => {
                invalid("name each unresolved parameter once").offending(index.to_string()).build()
            }
            Self::NoStructures => structural("supply at least one structure").build(),
            Self::StructureIdBlank { position } => {
                structural("give every structure a non-blank id")
                    .offending(position.to_string())
                    .build()
            }
            Self::StructureDuplicate { id } => {
                structural("give every structure a distinct id").offending(id.clone()).build()
            }
            Self::StructureProbabilitiesPartial => {
                unsatisfied(STRUCTURAL, "give every structure a probability, or none").build()
            }
            Self::StructureProbabilityInvalid { structure } => {
                unsatisfied(STRUCTURAL, "supply finite, non-negative structure probabilities")
                    .offending(structure.clone())
                    .build()
            }
            Self::StructureProbabilitiesExceedOne => {
                unsatisfied(STRUCTURAL, "supply structure probabilities whose total is at most one")
                    .build()
            }
            Self::StructureProbabilitiesRequired => unsatisfied(
                STRUCTURAL,
                "supply genuine structure probabilities, or choose another structural policy",
            )
            .build(),
            Self::NonFiniteCoefficient { action } => {
                unsatisfied(EVALUATE, "check the source draws and the domain of the value")
                    .offending(action.clone())
                    .build()
            }
            Self::UnboundedProgram | Self::UnexpectedInfeasible => {
                refusal(antecedent_core::reason_code!("invalid_argument"), EVALUATE, self.detail())
                    .build()
            }
            Self::Cancelled => refusal(
                antecedent_core::reason_code!("cancelled_no_claim"),
                EVALUATE,
                self.detail(),
            )
            .build(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_square() -> Region {
        Region {
            rows: Vec::new(),
            lower: vec![0.0, 0.0],
            upper: vec![1.0, 1.0],
            cancellation: None,
        }
    }

    #[test]
    fn epigraph_programs_give_the_worst_and_best_of_several_pieces() {
        let region = unit_square();
        let lp = LpOptions::default();
        // max(x, y) is smallest at the origin; min(x, 1 - x) is largest at x = 1/2.
        let low = region.min_of_max(&[vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]], lp).unwrap();
        assert!(low.value.abs() < 1e-9, "{low:?}");
        let high = region.max_of_min(&[vec![0.0, 1.0, 0.0], vec![1.0, -1.0, 0.0]], lp).unwrap();
        assert!((high.value - 0.5).abs() < 1e-9, "{high:?}");
        assert!((high.point[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn margins_flip_with_their_witnesses() {
        let margin = Margin { min: -1.0, min_at: vec![0.0], max: 2.0, max_at: vec![1.0] };
        let flipped = margin.flipped();
        assert!((flipped.min + 2.0).abs() < f64::EPSILON);
        assert_eq!(flipped.min_at, vec![1.0]);
        assert!((flipped.max - 1.0).abs() < f64::EPSILON);
        assert_eq!(flipped.max_at, vec![0.0]);
        let mut table = MarginTable::new(2);
        table.entries[1] = Some(margin);
        assert!((table.get(1, 0).unwrap().max - 1.0).abs() < f64::EPSILON);
        assert!(table.get(0, 0).is_none());
    }

    #[test]
    fn pick_reports_ties_within_the_tolerance() {
        let scores = [1.0, 3.0, 3.0 - 1e-12, 2.0];
        assert_eq!(pick(&[0, 1, 2, 3], |a| scores[a], 1e-9), Err(vec![1, 2]));
        assert_eq!(pick(&[0, 1, 3], |a| scores[a], 1e-9), Ok(1));
    }

    #[test]
    fn a_zero_row_is_decided_by_its_constant() {
        let region = unit_square();
        let lp = LpOptions::default();
        assert!(region.feasible(&[vec![0.0, 0.0, 0.0]], lp).unwrap().is_some());
        assert!(region.feasible(&[vec![1.0, 0.0, 0.0]], lp).unwrap().is_none());
    }

    #[test]
    fn statement_numbers_have_four_significant_digits() {
        assert_eq!(plain_number(6300.0), "6300");
        assert_eq!(plain_number(6_300.000_000_1), "6300");
        assert_eq!(plain_number(123_456.0), "123456");
        assert_eq!(plain_number(1.5), "1.5");
        assert_eq!(plain_number(-0.123_456), "-0.1235");
        assert_eq!(plain_number(0.000_123_456), "0.0001235");
        assert_eq!(plain_number(0.0), "0");
        assert_eq!(plain_number(-0.0), "0");
        assert_eq!(plain_number(-1e-300), "-1e-300");
        assert_eq!(plain_number(15_000_000.0), "1.5e7");
        assert_eq!(plain_number(2.0e-5), "2e-5");
        assert_eq!(plain_number(f64::INFINITY), "inf");
    }

    #[test]
    fn identifiers_are_joined_as_a_list() {
        let ids = |list: &[&str]| -> Vec<String> { list.iter().map(|s| (*s).to_owned()).collect() };
        assert_eq!(join_ids(&ids(&[])), "");
        assert_eq!(join_ids(&ids(&["a"])), "a");
        assert_eq!(join_ids(&ids(&["a", "b"])), "a and b");
        assert_eq!(join_ids(&ids(&["a", "b", "c"])), "a, b and c");
    }

    #[test]
    fn standing_separates_unique_tied_and_dominated_actions() {
        // a - b in [0, 2] (a never beaten, ties b at an edge); a - c in [1, 3] (c dominated by
        // a); b - c in [-1, 1] (b and c switch).
        let mut table = MarginTable::new(3);
        let margin = |min: f64, max: f64| Margin { min, min_at: vec![], max, max_at: vec![] };
        table.entries[1] = Some(margin(0.0, 2.0));
        table.entries[2] = Some(margin(1.0, 3.0));
        table.entries[5] = Some(margin(-1.0, 1.0));
        let standing = Standing::of(&table, &[0, 1, 2], 3, 1e-9);
        assert_eq!(standing.never_beaten, vec![true, false, false]);
        assert_eq!(standing.uniquely, vec![false, false, false]);
        assert_eq!(standing.dominated_by, vec![vec![], vec![], vec![0]]);
        assert_eq!(standing.touching[0], vec![1]);
        // Alone, an action is vacuously uniquely best.
        let alone = Standing::of(&MarginTable::new(1), &[0], 1, 1e-9);
        assert_eq!((alone.never_beaten[0], alone.uniquely[0]), (true, true));
    }

    #[test]
    fn receipts_join_distinct_values_in_order() {
        let receipt = |p: &str| SourceReceipt {
            provider_id: p.into(),
            snapshot_id: "s".into(),
            rng_id: "r".into(),
            causal_contract_id: "c".into(),
        };
        let (a, b) = (receipt("x"), receipt("y"));
        let joined = joined(&[&a, &b, &a]);
        assert_eq!(joined.provider_id, "x+y");
        assert_eq!(joined.snapshot_id, "s");
    }
}
