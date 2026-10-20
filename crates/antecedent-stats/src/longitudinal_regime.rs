//! Prespecified binary longitudinal regime value with supplied sequential probabilities.
//!
//! Probabilities and subject-level fold ownership
//! are inputs, not nuisance models fitted or verified by this kernel.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::diagnostic::{RouteCheck, RouteKind, RouteMass, RouteNativeDiagnostic};

/// A regime-kernel refusal.
///
/// Positivity and censoring failures carry a route record. Another message does
/// not, so it is not relabeled as one of those checks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegimeKernelError {
    message: &'static str,
    route: Option<RouteNativeDiagnostic>,
}

impl RegimeKernelError {
    /// Classify `message`.
    ///
    /// The positivity and censoring refusals are matched in full. Any other
    /// message stays unclassified.
    #[must_use = "an unclassified message is dropped if the error is discarded"]
    pub fn from_message(message: &'static str) -> Self {
        let failed = RouteCheck::new(true, RouteMass::unreported());
        let route = match message {
            "sequential treatment positivity is violated at the declared probability floor"
            | "sequential treatment positivity fails at the declared floor"
            | "sequential treatment positivity is violated" => {
                Some(RouteNativeDiagnostic::check(RouteKind::Positivity, failed))
            }
            "sequential censoring positivity is violated at the declared probability floor"
            | "sequential censoring positivity fails at the declared floor"
            | "sequential censoring positivity is violated" => {
                Some(RouteNativeDiagnostic::check(RouteKind::Censoring, failed))
            }
            _ => None,
        };
        Self { message, route }
    }

    /// Refusal text.
    #[must_use]
    pub const fn message(self) -> &'static str {
        self.message
    }

    /// Route record, when this refusal is a positivity or censoring check.
    #[must_use = "a failed positivity or censoring check is dropped if the record is discarded"]
    pub const fn route(self) -> Option<RouteNativeDiagnostic> {
        self.route
    }
}

impl std::fmt::Display for RegimeKernelError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.message)
    }
}

impl std::error::Error for RegimeKernelError {}

impl AsRef<str> for RegimeKernelError {
    fn as_ref(&self) -> &str {
        self.message
    }
}

impl std::ops::Deref for RegimeKernelError {
    type Target = str;

    fn deref(&self) -> &str {
        self.message
    }
}

/// Subject-major longitudinal design shared by the regime-value kernels and the marginal
/// structural model: the realized treatment and observation history with the supplied
/// sequential probabilities and the positivity floor they are checked against.
///
/// Every per-period slice is flattened subject-major (`n * periods` cells);
/// `outcome_observed` has one entry per subject.
#[derive(Clone, Copy, Debug)]
pub struct SubjectHistories<'a> {
    /// Realized treatment at each subject-period.
    pub treatment: &'a [bool],
    /// Conditional probability of action 1 at each subject-period.
    pub treatment_probability: &'a [f64],
    /// Whether each subject's terminal outcome is observed.
    pub outcome_observed: &'a [bool],
    /// Conditional probability of remaining observed at each decision.
    pub censoring_probability: &'a [f64],
    /// Decision periods per subject.
    pub periods: usize,
    /// Positivity floor in `(0, 0.5]` for treatment and censoring probabilities.
    pub minimum_probability: f64,
}

/// Point summary for one prescribed longitudinal regime.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegimeValueSummary {
    /// Horvitz--Thompson mean over all enrolled subjects.
    pub value: f64,
    /// Kish effective sample size of the matched-trajectory weights for inverse-probability
    /// summaries; the enrolled subject count for g-formula and sequential doubly robust ones.
    pub effective_sample_size: f64,
    /// Fraction of subjects observed through the requested trajectory.
    pub matched_observed_fraction: f64,
    /// Largest cumulative treatment and censoring weight.
    pub maximum_weight: f64,
    /// Smallest conditional probability of the prescribed action.
    pub minimum_action_probability: f64,
    /// Smallest conditional probability of remaining uncensored.
    pub minimum_censoring_probability: f64,
    /// Independent-subject standard error of the Horvitz--Thompson scores,
    /// when the method supplies observed subject scores.
    pub score_standard_error: Option<f64>,
}

/// Pointwise 95% regime-value interval for independent randomized subjects.
/// A finite support floor protects the normal approximation from rare
/// matched trajectories. The caller must establish known probabilities and
/// independent subject histories before publishing this interval.
#[must_use]
pub fn ipw_pointwise_interval_95(
    summary: &RegimeValueSummary,
    subjects: usize,
) -> Option<[f64; 2]> {
    let se = summary.score_standard_error?;
    let matched = summary.matched_observed_fraction * subjects as f64;
    if subjects < 500
        || matched < 50.0
        || summary.effective_sample_size < 50.0
        || !se.is_finite()
        || se <= 0.0
    {
        return None;
    }
    let span = crate::NORMAL_Q975 * se;
    let bounds = [summary.value - span, summary.value + span];
    bounds.iter().all(|value| value.is_finite()).then_some(bounds)
}

/// Backward-recursive sequential doubly robust score. All histories are
/// subject-major; a censored subject contributes its last available Q value.
pub fn evaluate_sequential_dr_value(
    outcomes: &[f64],
    histories: &SubjectHistories<'_>,
    sequential: &SequentialDrInputs<'_>,
) -> Result<RegimeValueSummary, RegimeKernelError> {
    evaluate_sequential_dr_value_with_subject_scores(outcomes, histories, sequential)
        .map(|(summary, _)| summary)
}

/// The prescribed regime and the frozen nuisance predictions a sequential doubly robust
/// value is formed from, subject-major like [`SubjectHistories`].
#[derive(Clone, Copy, Debug)]
pub struct SequentialDrInputs<'a> {
    /// Prescribed regime action at each subject-period.
    pub actions: &'a [bool],
    /// Whether the subject is still observed at each subject-period.
    pub observation_history: &'a [bool],
    /// Frozen, subject-excluded Q predictions at each subject-period.
    pub q_predictions: &'a [f64],
}

/// Return one full-history augmentation score per subject for calibration.
/// Scores alone do not establish independent nuisance fitting or an interval
/// claim; callers must keep the Q ownership contract separate.
#[allow(
    clippy::too_many_lines,
    reason = "backward sequential regression over time points, then the value and subject scores; the stages share the fitted nuisances"
)]
pub fn evaluate_sequential_dr_value_with_subject_scores(
    outcomes: &[f64],
    histories: &SubjectHistories<'_>,
    sequential: &SequentialDrInputs<'_>,
) -> Result<(RegimeValueSummary, Vec<f64>), RegimeKernelError> {
    (|| -> Result<(RegimeValueSummary, Vec<f64>), &'static str> {
    let SubjectHistories {
        treatment,
        treatment_probability,
        outcome_observed,
        censoring_probability,
        periods,
        minimum_probability,
    } = *histories;
    let SequentialDrInputs { actions, observation_history, q_predictions } = *sequential;
    let n = outcomes.len();
    let cells = n.checked_mul(periods).ok_or("longitudinal dimensions overflow")?;
    if n == 0
        || periods == 0
        || outcome_observed.len() != n
        || [
            observation_history.len(),
            treatment.len(),
            actions.len(),
            q_predictions.len(),
            treatment_probability.len(),
            censoring_probability.len(),
        ]
        .iter()
        .any(|&len| len != cells)
    {
        return Err("sequential DR arrays must align by subject and period");
    }
    if !(minimum_probability.is_finite() && 0.0 < minimum_probability && minimum_probability <= 0.5)
    {
        return Err("minimum probability must be finite and in (0, 0.5]");
    }
    if !outcome_observed.iter().any(|&observed| observed) {
        return Err("at least one terminal outcome must be observed for sequential augmentation");
    }
    let mut total = 0.0;
    let mut min_action: f64 = 1.0;
    let mut min_censor: f64 = 1.0;
    let mut maximum_weight: f64 = 1.0;
    let mut scores = Vec::with_capacity(n);
    for i in 0..n {
        if outcome_observed[i] && !outcomes[i].is_finite() {
            return Err("observed terminal outcomes must be finite");
        }
        if observation_history[i * periods + periods - 1] != outcome_observed[i] {
            return Err(
                "terminal outcome observation must agree with the final observation-history period",
            );
        }
        let mut next = if outcome_observed[i] { outcomes[i] } else { 0.0 };
        for t in (0..periods).rev() {
            let j = i * periods + t;
            if t > 0 && !observation_history[j - 1] && observation_history[j] {
                return Err("observation_history must be monotone after censoring or dropout");
            }
            let q = q_predictions[j];
            let p = treatment_probability[j];
            let c = censoring_probability[j];
            if !q.is_finite() {
                return Err("q_prediction values must be finite");
            }
            if !p.is_finite() || p < minimum_probability || p > 1.0 - minimum_probability {
                return Err(
                    "sequential treatment positivity is violated at the declared probability floor",
                );
            }
            if !c.is_finite() || c < minimum_probability || c > 1.0 {
                return Err(
                    "sequential censoring positivity is violated at the declared probability floor",
                );
            }
            let action_p = if actions[j] { p } else { 1.0 - p };
            min_action = min_action.min(action_p);
            min_censor = min_censor.min(c);
            next = if observation_history[j] && treatment[j] == actions[j] {
                q + (next - q) / (action_p * c)
            } else {
                q
            };
            if !next.is_finite() {
                return Err(
                    "sequential augmentation overflowed; raise the positivity floor or shorten the horizon",
                );
            }
        }
        let mut weight = 1.0;
        for t in 0..periods {
            let j = i * periods + t;
            if !observation_history[j] || treatment[j] != actions[j] {
                break;
            }
            let ap =
                if actions[j] { treatment_probability[j] } else { 1.0 - treatment_probability[j] };
            weight /= ap * censoring_probability[j];
            if !weight.is_finite() {
                return Err("sequential weight overflowed");
            }
            maximum_weight = maximum_weight.max(weight);
        }
        total += next;
        scores.push(next);
        if !total.is_finite() {
            return Err("sequential augmented values overflowed across subjects");
        }
    }
    Ok((
        RegimeValueSummary {
            value: total / n as f64,
            effective_sample_size: n as f64,
            matched_observed_fraction: outcome_observed.iter().filter(|&&x| x).count() as f64
                / n as f64,
            maximum_weight,
            minimum_action_probability: min_action,
            minimum_censoring_probability: min_censor,
            score_standard_error: None,
        },
        scores,
    ))
    })()
    .map_err(RegimeKernelError::from_message)
}

/// Calibrated two- or three-period pointwise interval for a prespecified regime under
/// known randomization and genuinely subject-excluded Q predictions. The
/// caller must establish those ownership conditions; this kernel can check
/// only the score and support inputs. The support floor is deliberately
/// limited to the randomized designs tested in repeated samples.
#[must_use]
pub fn sequential_dr_pointwise_interval_95(
    summary: &RegimeValueSummary,
    scores: &[f64],
    periods: usize,
    observed_subjects: usize,
    observed_matching_subjects: usize,
) -> Option<(f64, [f64; 2])> {
    let n = scores.len();
    let supported = match periods {
        2 => {
            n >= 500
                && observed_subjects >= 200
                && observed_matching_subjects >= 50
                && summary.minimum_action_probability >= 0.4
                && summary.minimum_censoring_probability >= 0.85
        }
        3 => {
            n >= 800
                && observed_subjects >= 600
                && observed_matching_subjects >= 60
                && summary.minimum_action_probability >= 0.5
                && summary.minimum_censoring_probability >= 0.95
        }
        _ => false,
    };
    if !supported || scores.iter().any(|score| !score.is_finite()) {
        return None;
    }
    let se = (scores.iter().map(|score| (score - summary.value).powi(2)).sum::<f64>()
        / (n * (n - 1)) as f64)
        .sqrt();
    if !se.is_finite() || se <= 0.0 {
        return None;
    }
    let span = crate::NORMAL_Q975 * se;
    let bounds = [summary.value - span, summary.value + span];
    bounds.iter().all(|bound| bound.is_finite()).then_some((se, bounds))
}

/// Evaluate supplied conditional period rewards under a prescribed regime.
/// Predictions are subject-major and must be produced without reusing the
/// subject's outcome when excluded-fold ownership is declared upstream.
pub fn evaluate_g_formula_value(
    predictions: &[f64],
    actions: &[bool],
    treatment_probability: &[f64],
    censoring_probability: &[f64],
    subjects: usize,
    periods: usize,
    minimum_probability: f64,
) -> Result<RegimeValueSummary, RegimeKernelError> {
    (|| -> Result<RegimeValueSummary, &'static str> {
        let cells = subjects.checked_mul(periods).ok_or("longitudinal dimensions overflow")?;
        if subjects == 0
            || periods == 0
            || predictions.len() != cells
            || actions.len() != cells
            || treatment_probability.len() != cells
            || censoring_probability.len() != cells
        {
            return Err("g-formula arrays must align by subject and period");
        }
        if !(minimum_probability.is_finite()
            && 0.0 < minimum_probability
            && minimum_probability <= 0.5)
        {
            return Err("minimum probability must be finite and in (0, 0.5]");
        }
        let mut total = 0.0;
        let mut minimum_action_probability: f64 = 1.0;
        let mut minimum_censoring_probability: f64 = 1.0;
        for j in 0..cells {
            let q = predictions[j];
            let p = treatment_probability[j];
            let c = censoring_probability[j];
            if !q.is_finite() {
                return Err("period outcome predictions must be finite");
            }
            if !p.is_finite() || p < minimum_probability || p > 1.0 - minimum_probability {
                return Err("sequential treatment positivity fails at the declared floor");
            }
            if !c.is_finite() || c < minimum_probability || c > 1.0 {
                return Err("sequential censoring positivity fails at the declared floor");
            }
            minimum_action_probability =
                minimum_action_probability.min(if actions[j] { p } else { 1.0 - p });
            minimum_censoring_probability = minimum_censoring_probability.min(c);
            total += q;
            if !total.is_finite() {
                return Err("summed outcome predictions overflowed");
            }
        }
        Ok(RegimeValueSummary {
            value: total / subjects as f64,
            effective_sample_size: subjects as f64,
            matched_observed_fraction: 1.0,
            maximum_weight: 1.0,
            minimum_action_probability,
            minimum_censoring_probability,
            score_standard_error: None,
        })
    })()
    .map_err(RegimeKernelError::from_message)
}

/// Conditional-on-fixed-known-Q pointwise interval for a two-period g-formula
/// value over independently sampled subject histories. Each subject contributes
/// the sum of its two supplied period rewards. This interval excludes any
/// uncertainty from fitting or selecting the Q law, including cross-fitting.
/// A retained caller must separately attest that the Q law is fixed and known.
#[must_use]
pub fn g_formula_fixed_q_pointwise_interval_95(
    summary: &RegimeValueSummary,
    predictions: &[f64],
    subjects: usize,
    periods: usize,
) -> Option<(f64, [f64; 2])> {
    if periods != 2
        || subjects < 300
        || predictions.len() != subjects.checked_mul(periods)?
        || summary.minimum_action_probability < 0.2
        || summary.minimum_censoring_probability < 0.8
        || !summary.value.is_finite()
    {
        return None;
    }
    let mut centered_sum = 0.0;
    let mut total = 0.0;
    for subject in 0..subjects {
        let score = predictions[subject * periods] + predictions[subject * periods + 1];
        if !score.is_finite() {
            return None;
        }
        total += score;
        centered_sum += (score - summary.value).powi(2);
    }
    if !total.is_finite()
        || (total / subjects as f64 - summary.value).abs() > 1e-10 * (1.0 + summary.value.abs())
    {
        return None;
    }
    let se = (centered_sum / (subjects * (subjects - 1)) as f64).sqrt();
    if !se.is_finite() || se <= 0.0 {
        return None;
    }
    let span = crate::NORMAL_Q975 * se;
    let bounds = [summary.value - span, summary.value + span];
    bounds.iter().all(|bound| bound.is_finite()).then_some((se, bounds))
}

/// Evaluate a caller-prescribed static or history-dependent regime on one row
/// per subject. All two-dimensional inputs are flattened in subject-major order.
///
/// `treatment_probability` is the conditional probability of action 1, and
/// `censoring_probability` is the conditional probability of remaining observed
/// at that decision. Values are unconditional Horvitz--Thompson averages, so
/// unmatched or censored subjects contribute zero to the numerator.
pub fn evaluate_regime_value(
    outcomes: &[f64],
    histories: &SubjectHistories<'_>,
    actions: &[bool],
) -> Result<RegimeValueSummary, RegimeKernelError> {
    (|| -> Result<RegimeValueSummary, &'static str> {
        let SubjectHistories {
            treatment,
            treatment_probability,
            outcome_observed,
            censoring_probability,
            periods,
            minimum_probability,
        } = *histories;
        let n = outcomes.len();
        let cells = n.checked_mul(periods).ok_or("longitudinal dimensions overflow")?;
        if n == 0
            || periods == 0
            || treatment.len() != cells
            || actions.len() != cells
            || treatment_probability.len() != cells
            || censoring_probability.len() != cells
            || outcome_observed.len() != n
        {
            return Err("regime arrays must align by subject and period");
        }
        if !(minimum_probability.is_finite()
            && 0.0 < minimum_probability
            && minimum_probability <= 0.5)
        {
            return Err("minimum probability must be finite and in (0, 0.5]");
        }
        let mut total = 0.0;
        let mut matched_weights = Vec::new();
        let mut maximum_weight: f64 = 0.0;
        let mut minimum_action_probability: f64 = 1.0;
        let mut minimum_censoring_probability: f64 = 1.0;
        let mut matched = 0usize;
        let mut scores = Vec::with_capacity(n);
        for i in 0..n {
            if !outcomes[i].is_finite() {
                return Err("longitudinal outcomes must be finite");
            }
            let mut follows = true;
            let mut weight = 1.0;
            for t in 0..periods {
                let j = i * periods + t;
                let p = treatment_probability[j];
                let c = censoring_probability[j];
                if !p.is_finite() || p < minimum_probability || p > 1.0 - minimum_probability {
                    return Err("sequential treatment positivity fails at the declared floor");
                }
                if !c.is_finite() || c < minimum_probability || c > 1.0 {
                    return Err("sequential censoring positivity fails at the declared floor");
                }
                let action_probability = if actions[j] { p } else { 1.0 - p };
                minimum_action_probability = minimum_action_probability.min(action_probability);
                minimum_censoring_probability = minimum_censoring_probability.min(c);
                follows &= treatment[j] == actions[j];
                if follows {
                    weight /= action_probability * c;
                    if !weight.is_finite() {
                        return Err("sequential weight overflowed");
                    }
                }
            }
            if follows && outcome_observed[i] {
                matched += 1;
                total += weight * outcomes[i];
                matched_weights.push(weight);
                maximum_weight = maximum_weight.max(weight);
                scores.push(weight * outcomes[i]);
            } else {
                scores.push(0.0);
            }
        }
        if matched == 0 {
            return Err("no observed subjects followed the requested regime");
        }
        if !total.is_finite() {
            return Err("longitudinal weighted score overflowed");
        }
        let value = total / n as f64;
        let score_standard_error = if n > 1 {
            let se = (scores.iter().map(|score| (score - value).powi(2)).sum::<f64>()
                / (n * (n - 1)) as f64)
                .sqrt();
            if !se.is_finite() {
                return Err("longitudinal subject-score variance overflowed");
            }
            Some(se)
        } else {
            None
        };
        Ok(RegimeValueSummary {
            value,
            effective_sample_size: antecedent_kernels::kish_ess(&matched_weights),
            matched_observed_fraction: matched as f64 / n as f64,
            maximum_weight,
            minimum_action_probability,
            minimum_censoring_probability,
            score_standard_error,
        })
    })()
    .map_err(RegimeKernelError::from_message)
}

#[cfg(test)]
mod tests {
    #![cfg_attr(
        test,
        allow(
            clippy::float_cmp,
            reason = "tests assert exact deterministic estimates and coverage arithmetic"
        )
    )]
    use antecedent_core::splitmix64;

    use super::*;

    /// A uniform draw in `[0, 1)` from the next `SplitMix64` output.
    fn splitmix64_unit(state: &mut u64) -> f64 {
        (splitmix64(state) >> 11) as f64 / (1_u64 << 53) as f64
    }

    #[test]
    fn three_period_sequential_dr_interval_covers_randomized_truth() {
        let n = 800;
        let simulations = 2_000;
        let mut state = 0x87A4_12CD_95EF_3B60_u64;
        let mut uniform = || splitmix64_unit(&mut state);
        let actions = vec![true; n * 3];
        let q = (0..n).flat_map(|_| [4.4, 4.7, 5.1]).collect::<Vec<_>>();
        let p = vec![0.5; n * 3];
        let c = vec![0.95; n * 3];
        let mut accepted = 0;
        let mut covered = 0;
        for _ in 0..simulations {
            let mut treatment = Vec::with_capacity(n * 3);
            let mut history = Vec::with_capacity(n * 3);
            let mut observed = Vec::with_capacity(n);
            let mut outcomes = Vec::with_capacity(n);
            let mut matched = 0;
            let mut observed_count = 0;
            for _ in 0..n {
                let a = [uniform() < 0.5, uniform() < 0.5, uniform() < 0.5];
                let o0 = uniform() < 0.95;
                let o1 = o0 && uniform() < 0.95;
                let o2 = o1 && uniform() < 0.95;
                treatment.extend(a);
                history.extend([o0, o1, o2]);
                observed.push(o2);
                observed_count += usize::from(o2);
                matched += usize::from(o2 && a.iter().all(|&action| action));
                outcomes.push(
                    2.0 + a.iter().map(|&action| f64::from(action)).sum::<f64>()
                        + 2.0 * (uniform() - 0.5),
                );
            }
            let (summary, scores) = evaluate_sequential_dr_value_with_subject_scores(
                &outcomes,
                &SubjectHistories {
                    treatment: &treatment,
                    treatment_probability: &p,
                    outcome_observed: &observed,
                    censoring_probability: &c,
                    periods: 3,
                    minimum_probability: 0.01,
                },
                &SequentialDrInputs {
                    actions: &actions,
                    observation_history: &history,
                    q_predictions: &q,
                },
            )
            .unwrap();
            if let Some((_, bounds)) =
                sequential_dr_pointwise_interval_95(&summary, &scores, 3, observed_count, matched)
            {
                accepted += 1;
                covered += usize::from(bounds[0] <= 5.0 && 5.0 <= bounds[1]);
            }
        }
        assert!(accepted >= 1_800, "three-period support: {accepted}/{simulations}");
        let coverage = covered as f64 / f64::from(accepted);
        println!("three-period sequential DR: accepted={accepted}, coverage={coverage}");
        assert!((coverage - 0.95).abs() <= 3.0 * (0.95_f64 * 0.05 / f64::from(accepted)).sqrt());
    }

    #[test]
    fn three_period_excluded_fold_q_interval_covers_randomized_truth() {
        let n = 800;
        let simulations = 2_000;
        let truth = 5.0;
        let mut state = 0xA83D_91E4_57C0_2FB6_u64;
        let mut uniform = || splitmix64_unit(&mut state);
        let actions = vec![true; n * 3];
        let p = vec![0.5; n * 3];
        let c = vec![0.95; n * 3];
        let mut accepted = 0;
        let mut covered = 0;
        for _ in 0..simulations {
            let mut treatment = Vec::with_capacity(n * 3);
            let mut history = Vec::with_capacity(n * 3);
            let mut observed = Vec::with_capacity(n);
            let mut outcomes = Vec::with_capacity(n);
            for _ in 0..n {
                let a = [uniform() < 0.5, uniform() < 0.5, uniform() < 0.5];
                let o0 = uniform() < 0.95;
                let o1 = o0 && uniform() < 0.95;
                let o2 = o1 && uniform() < 0.95;
                treatment.extend(a);
                history.extend([o0, o1, o2]);
                observed.push(o2);
                outcomes.push(
                    2.0 + a.iter().map(|&action| f64::from(action)).sum::<f64>()
                        + 2.0 * (uniform() - 0.5),
                );
            }
            let mut q = vec![0.0; n * 3];
            for fold in 0..5 {
                let (sum, count) = (0..n)
                    .filter(|&i| {
                        i % 5 != fold
                            && observed[i]
                            && treatment[3 * i..3 * i + 3].iter().all(|&a| a)
                    })
                    .fold((0.0, 0_usize), |(sum, count), i| (sum + outcomes[i], count + 1));
                assert!(count >= 40, "excluded-fold training support");
                let prediction = sum / count as f64;
                for i in (fold..n).step_by(5) {
                    q[3 * i..3 * i + 3].fill(prediction);
                }
            }
            let (summary, scores) = evaluate_sequential_dr_value_with_subject_scores(
                &outcomes,
                &SubjectHistories {
                    treatment: &treatment,
                    treatment_probability: &p,
                    outcome_observed: &observed,
                    censoring_probability: &c,
                    periods: 3,
                    minimum_probability: 0.01,
                },
                &SequentialDrInputs {
                    actions: &actions,
                    observation_history: &history,
                    q_predictions: &q,
                },
            )
            .unwrap();
            let matched = (0..n)
                .filter(|&i| observed[i] && treatment[3 * i..3 * i + 3].iter().all(|&a| a))
                .count();
            if let Some((_, bounds)) = sequential_dr_pointwise_interval_95(
                &summary,
                &scores,
                3,
                observed.iter().filter(|&&x| x).count(),
                matched,
            ) {
                accepted += 1;
                covered += usize::from(bounds[0] <= truth && truth <= bounds[1]);
            }
        }
        assert!(accepted >= 1_800, "three-period excluded-fold support: {accepted}/{simulations}");
        let coverage = covered as f64 / f64::from(accepted);
        println!(
            "three-period excluded-fold Q sequential DR: accepted={accepted}, coverage={coverage}"
        );
        assert!((coverage - 0.95).abs() <= 3.0 * (0.95_f64 * 0.05 / f64::from(accepted)).sqrt());
    }

    #[test]
    fn fixed_exogenous_q_sequential_dr_subject_scores_cover_randomized_truth() {
        let n = 300;
        let simulations = 2_000;
        let mut state = 0xD73A_94E1_2B6C_850F_u64;
        let mut uniform = || splitmix64_unit(&mut state);
        let actions = vec![true; n * 2];
        let q = (0..n).flat_map(|_| [3.5, 4.2]).collect::<Vec<_>>();
        let p = vec![0.5; n * 2];
        let c = vec![0.9; n * 2];
        let mut covered_90 = 0;
        let mut covered_95 = 0;
        let mut gated_total = 0;
        let mut gated_covered = 0;
        for _ in 0..simulations {
            let mut treatment = Vec::with_capacity(n * 2);
            let mut history = Vec::with_capacity(n * 2);
            let mut observed = Vec::with_capacity(n);
            let mut outcomes = Vec::with_capacity(n);
            for _ in 0..n {
                let a0 = uniform() < 0.5;
                let a1 = uniform() < 0.5;
                let o0 = uniform() < 0.9;
                let o1 = o0 && uniform() < 0.9;
                treatment.extend([a0, a1]);
                history.extend([o0, o1]);
                observed.push(o1);
                outcomes.push(2.0 + f64::from(a0) + f64::from(a1) + 2.0 * (uniform() - 0.5));
            }
            let (summary, scores) = evaluate_sequential_dr_value_with_subject_scores(
                &outcomes,
                &SubjectHistories {
                    treatment: &treatment,
                    treatment_probability: &p,
                    outcome_observed: &observed,
                    censoring_probability: &c,
                    periods: 2,
                    minimum_probability: 0.01,
                },
                &SequentialDrInputs {
                    actions: &actions,
                    observation_history: &history,
                    q_predictions: &q,
                },
            )
            .unwrap();
            let matched =
                (0..n).filter(|&i| observed[i] && treatment[2 * i] && treatment[2 * i + 1]).count();
            if let Some((_, bounds)) = sequential_dr_pointwise_interval_95(
                &summary,
                &scores,
                2,
                observed.iter().filter(|&&x| x).count(),
                matched,
            ) {
                gated_total += 1;
                gated_covered += usize::from(bounds[0] <= 4.0 && 4.0 <= bounds[1]);
            }
            assert_eq!(scores.len(), n);
            let se = (scores.iter().map(|score| (score - summary.value).powi(2)).sum::<f64>()
                / (n * (n - 1)) as f64)
                .sqrt();
            assert!(se.is_finite() && se > 0.0);
            covered_90 += usize::from((summary.value - 4.0).abs() <= crate::normal_ppf(0.95) * se);
            covered_95 += usize::from((summary.value - 4.0).abs() <= crate::NORMAL_Q975 * se);
        }
        for (nominal, hits) in [(0.90_f64, covered_90), (0.95, covered_95)] {
            let coverage = hits as f64 / f64::from(simulations);
            let mcse = (nominal * (1.0 - nominal) / f64::from(simulations)).sqrt();
            println!("fixed exogenous Q sequential DR: nominal={nominal}, coverage={coverage}");
            assert!(
                (coverage - nominal).abs() <= 3.0 * mcse,
                "fixed-Q sequential DR coverage {coverage} at {nominal}"
            );
        }
        assert_eq!(
            gated_total, 0,
            "300-subject fixed Q has no retained DR interval without a distinct fixed-Q contract"
        );
        assert_eq!(gated_covered, 0);
    }

    #[test]
    fn subject_excluded_fold_q_sequential_dr_scores_cover_randomized_truth() {
        let n = 500;
        let simulations = 2_000;
        let truth = 4.0;
        let mut state = 0x149C_6F82_D30A_5BE7_u64;
        let mut uniform = || splitmix64_unit(&mut state);
        let actions = vec![true; n * 2];
        let p = vec![0.4; n * 2];
        let c = vec![0.85; n * 2];
        let mut covered_90 = 0;
        let mut covered_95 = 0;
        let mut gated_total = 0;
        let mut gated_covered = 0;
        for _ in 0..simulations {
            let mut treatment = Vec::with_capacity(n * 2);
            let mut history = Vec::with_capacity(n * 2);
            let mut observed = Vec::with_capacity(n);
            let mut outcomes = Vec::with_capacity(n);
            for _ in 0..n {
                let a0 = uniform() < 0.4;
                let a1 = uniform() < 0.4;
                let o0 = uniform() < 0.85;
                let o1 = o0 && uniform() < 0.85;
                treatment.extend([a0, a1]);
                history.extend([o0, o1]);
                observed.push(o1);
                outcomes.push(2.0 + f64::from(a0) + f64::from(a1) + 2.0 * (uniform() - 0.5));
            }
            let mut q = vec![0.0; n * 2];
            for fold in 0..5 {
                let training = (0..n).filter(|&i| {
                    i % 5 != fold && observed[i] && treatment[2 * i] && treatment[2 * i + 1]
                });
                let (sum, count) =
                    training.fold((0.0, 0_usize), |(sum, count), i| (sum + outcomes[i], count + 1));
                assert!(count >= 20, "training support in excluded folds");
                let prediction = sum / count as f64;
                for i in (fold..n).step_by(5) {
                    q[2 * i] = prediction;
                    q[2 * i + 1] = prediction;
                }
            }
            let (summary, scores) = evaluate_sequential_dr_value_with_subject_scores(
                &outcomes,
                &SubjectHistories {
                    treatment: &treatment,
                    treatment_probability: &p,
                    outcome_observed: &observed,
                    censoring_probability: &c,
                    periods: 2,
                    minimum_probability: 0.01,
                },
                &SequentialDrInputs {
                    actions: &actions,
                    observation_history: &history,
                    q_predictions: &q,
                },
            )
            .unwrap();
            let matched =
                (0..n).filter(|&i| observed[i] && treatment[2 * i] && treatment[2 * i + 1]).count();
            if let Some((_, bounds)) = sequential_dr_pointwise_interval_95(
                &summary,
                &scores,
                2,
                observed.iter().filter(|&&x| x).count(),
                matched,
            ) {
                gated_total += 1;
                gated_covered += usize::from(bounds[0] <= truth && truth <= bounds[1]);
            }
            let se = (scores.iter().map(|score| (score - summary.value).powi(2)).sum::<f64>()
                / (n * (n - 1)) as f64)
                .sqrt();
            assert!(se.is_finite() && se > 0.0);
            covered_90 += usize::from((summary.value - 4.0).abs() <= crate::normal_ppf(0.95) * se);
            covered_95 += usize::from((summary.value - 4.0).abs() <= crate::NORMAL_Q975 * se);
        }
        for (nominal, hits) in [(0.90_f64, covered_90), (0.95, covered_95)] {
            let coverage = hits as f64 / f64::from(simulations);
            let mcse = (nominal * (1.0 - nominal) / f64::from(simulations)).sqrt();
            println!(
                "subject-excluded fold Q sequential DR: nominal={nominal}, coverage={coverage}"
            );
            assert!(
                (coverage - nominal).abs() <= 3.0 * mcse,
                "subject-excluded Q sequential DR coverage {coverage} at {nominal}"
            );
        }
        assert!(gated_total >= 1_400);
        let coverage = gated_covered as f64 / f64::from(gated_total);
        println!(
            "excluded-fold Q supported DR interval: accepted={gated_total}, coverage={coverage}"
        );
        assert!((coverage - 0.95).abs() <= 3.0 * (0.95_f64 * 0.05 / f64::from(gated_total)).sqrt());
    }

    #[test]
    fn subject_ipw_interval_recovers_randomized_censored_truth_over_repeated_samples() {
        let n = 500;
        let simulations = 2_000;
        let mut state = 0x72B4_07D1_9C38_EF65_u64;
        let mut uniform = || splitmix64_unit(&mut state);
        let actions = vec![true; n * 2];
        let treatment_probabilities = vec![0.5; n * 2];
        let censoring_probabilities = vec![0.9; n * 2];
        let mut covered = 0;
        let mut skipped = 0;
        for _ in 0..simulations {
            let mut treatment = Vec::with_capacity(n * 2);
            let mut observed = Vec::with_capacity(n);
            let mut outcomes = Vec::with_capacity(n);
            for _ in 0..n {
                let a0 = uniform() < 0.5;
                let a1 = uniform() < 0.5;
                treatment.extend([a0, a1]);
                observed.push(uniform() < 0.9 && uniform() < 0.9);
                outcomes.push(2.0 + f64::from(a0) + f64::from(a1) + (uniform() - 0.5) * 2.0);
            }
            let summary = evaluate_regime_value(
                &outcomes,
                &SubjectHistories {
                    treatment: &treatment,
                    treatment_probability: &treatment_probabilities,
                    outcome_observed: &observed,
                    censoring_probability: &censoring_probabilities,
                    periods: 2,
                    minimum_probability: 0.01,
                },
                &actions,
            )
            .unwrap();
            if let Some(interval) = ipw_pointwise_interval_95(&summary, n) {
                covered += usize::from(interval[0] <= 4.0 && 4.0 <= interval[1]);
            } else {
                skipped += 1;
            }
        }
        let coverage = covered as f64 / f64::from(simulations);
        let mcse = (0.95_f64 * 0.05 / f64::from(simulations)).sqrt();
        assert!(skipped <= simulations / 100, "weak-support skips {skipped}");
        assert!((coverage - 0.95).abs() <= 3.0 * mcse, "longitudinal IPW coverage {coverage}");
    }

    #[test]
    fn fixed_known_q_g_formula_interval_covers_regime_truth() {
        // Two-period g-formula value under a fixed, known conditional reward law.
        // Each subject's supplied period rewards are a fixed measurement, so the
        // regime value is the sample mean of the summed per-subject rewards and
        // the pointwise interval is its independent-subject normal CI. The truth
        // is the population mean of the summed rewards; coverage confirms the
        // conditional-on-fixed-Q claim holds over repeated subject samples.
        let subjects = 400;
        let simulations = 2_000;
        let truth = 5.0;
        let mut state = 0x51ED_270B_C7A9_D341_u64;
        let mut uniform = || splitmix64_unit(&mut state);
        let actions = vec![true; subjects * 2];
        let treatment_probabilities = vec![0.5; subjects * 2];
        let censoring_probabilities = vec![0.9; subjects * 2];
        let mut covered = 0;
        let mut skipped = 0;
        for _ in 0..simulations {
            let mut predictions = Vec::with_capacity(subjects * 2);
            for _ in 0..subjects {
                let q0 = 2.0 + (uniform() - 0.5) * 2.0;
                let q1 = 3.0 + (uniform() - 0.5) * 2.0;
                predictions.push(q0);
                predictions.push(q1);
            }
            let summary = evaluate_g_formula_value(
                &predictions,
                &actions,
                &treatment_probabilities,
                &censoring_probabilities,
                subjects,
                2,
                0.01,
            )
            .unwrap();
            if let Some((_, bounds)) =
                g_formula_fixed_q_pointwise_interval_95(&summary, &predictions, subjects, 2)
            {
                covered += usize::from(bounds[0] <= truth && truth <= bounds[1]);
            } else {
                skipped += 1;
            }
        }
        assert!(skipped <= simulations / 100, "fixed-known-Q weak-support skips {skipped}");
        let coverage = covered as f64 / f64::from(simulations - skipped);
        let mcse = (0.95_f64 * 0.05 / f64::from(simulations - skipped)).sqrt();
        println!("fixed-known-Q g-formula: covered={covered}, coverage={coverage}");
        assert!(
            (coverage - 0.95).abs() <= 3.0 * mcse,
            "fixed-known-Q g-formula coverage {coverage}"
        );
    }

    #[test]
    fn known_truth_two_period_bernoulli() {
        let summary = evaluate_regime_value(
            &[4.0, 0.0, 0.0, 0.0],
            &SubjectHistories {
                treatment: &[true, true, true, false, false, true, false, false],
                treatment_probability: &[0.5; 8],
                outcome_observed: &[true; 4],
                censoring_probability: &[1.0; 8],
                periods: 2,
                minimum_probability: 0.01,
            },
            &[true, true, true, true, true, true, true, true],
        )
        .unwrap();
        assert_eq!(summary.value, 4.0);
        assert_eq!(summary.matched_observed_fraction, 0.25);
        assert_eq!(summary.effective_sample_size, 1.0);
    }

    #[test]
    fn refuses_missing_support_and_positivity() {
        let mut p = [0.5; 4];
        assert!(
            evaluate_regime_value(
                &[1.0, 2.0],
                &SubjectHistories {
                    treatment: &[false; 4],
                    treatment_probability: &p,
                    outcome_observed: &[true; 2],
                    censoring_probability: &[1.0; 4],
                    periods: 2,
                    minimum_probability: 0.01
                },
                &[true; 4]
            )
            .unwrap_err()
            .contains("no observed")
        );
        p[0] = 0.0;
        assert!(
            evaluate_regime_value(
                &[1.0, 2.0],
                &SubjectHistories {
                    treatment: &[false; 4],
                    treatment_probability: &p,
                    outcome_observed: &[true; 2],
                    censoring_probability: &[1.0; 4],
                    periods: 2,
                    minimum_probability: 0.01
                },
                &[true; 4]
            )
            .unwrap_err()
            .contains("positivity")
        );
    }

    #[test]
    fn treatment_and_censoring_floor_failures_are_different_checks() {
        fn histories<'a>(
            treatment_probability: &'a [f64],
            censoring_probability: &'a [f64],
        ) -> SubjectHistories<'a> {
            SubjectHistories {
                treatment: &[true, true],
                treatment_probability,
                outcome_observed: &[true],
                censoring_probability,
                periods: 2,
                minimum_probability: 0.1,
            }
        }
        let actions = [true, true];
        let positivity =
            evaluate_regime_value(&[1.0], &histories(&[0.01, 0.5], &[1.0, 1.0]), &actions)
                .expect_err("treatment probability is below the floor");
        let censoring =
            evaluate_regime_value(&[1.0], &histories(&[0.5, 0.5], &[0.01, 1.0]), &actions)
                .expect_err("censoring probability is below the floor");
        let positivity = positivity.route().expect("positivity record");
        let censoring = censoring.route().expect("censoring record");
        assert!(positivity.failed());
        assert!(censoring.failed());
        assert_eq!(positivity.route_name(), "positivity");
        assert_eq!(censoring.route_name(), "censoring");
        assert_ne!(positivity, censoring);
        assert!(!positivity.is_unsupported_inference());
        assert!(!censoring.is_model_restriction());
        assert!(!censoring.is_unsupported_inference());
    }
}
