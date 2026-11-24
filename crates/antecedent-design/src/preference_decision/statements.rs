//! Plain statements of decisions across a preference set: numbers written to four significant
//! digits and the sentence of each verdict, built only from a result's own facts.

// SPDX-License-Identifier: MIT OR Apache-2.0

#[allow(clippy::wildcard_imports, reason = "a part of its parent module")]
use super::*;

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
pub(super) fn over_the_set(has_parameters: bool, quantifier: &str) -> String {
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

pub(super) fn verdict_statement(
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

pub(super) fn never_beaten_sentence(
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

pub(super) fn dependent_sentence(
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
