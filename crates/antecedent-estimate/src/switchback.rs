//! Sequence-level inference for randomized switchback experiments.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

/// Finite assignment distribution for one independent switchback sequence.
/// Each arm labels a complete simultaneous multichannel assignment at one period.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HistoryAssignment {
    /// All possible schedules, including schedules not realized in this sample.
    pub schedules: Vec<Vec<usize>>,
    /// Known probabilities in schedule order; strictly positive and summing to one.
    pub probabilities: Vec<f64>,
    /// Index of the schedule actually assigned.
    pub realized: usize,
    /// Observed outcomes for the complete realized schedule.
    pub outcomes: Vec<f64>,
}

/// History-aware Horvitz--Thompson response means from independent sequences.
/// Uncertainty is an uncalibrated sequence-cluster sandwich approximation.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HistorySwitchbackFit {
    /// One mean per requested history, in caller order.
    pub means: Vec<f64>,
    /// Row-major joint covariance, including cross-history covariance.
    pub covariance: Vec<f64>,
    /// Sequence-level contributions, retained for joint contrasts and resampling.
    pub sequence_scores: Vec<Vec<f64>>,
    /// Number of observed matches for each requested history.
    pub observed_support: Vec<usize>,
    /// Number of outcome periods included across all independent sequences.
    pub eligible_periods: usize,
}

/// Estimate joint-history means under a known finite schedule randomization.
///
/// A history of length `L+1` identifies the response at its final period under
/// consistency, no anticipation, no interference between independent sequences,
/// and no carryover beyond `L`. These are design premises, not tests on the data.
/// Inclusion probabilities are computed from *whole schedules*, never multiplied
/// from marginal arm probabilities. Correlation within a sequence is unrestricted.
/// The first `L` periods have no observed prehistory and are excluded explicitly.
///
/// # Errors
/// Refuses malformed/duplicate schedules, impossible realizations, nonfinite inputs,
/// histories without design or observed support, fewer than two independent sequences,
/// and numerical overflow. It never silently substitutes a shorter carryover order.
pub fn switchback_history_means(
    sequences: &[HistoryAssignment],
    histories: &[Vec<usize>],
) -> Result<HistorySwitchbackFit, &'static str> {
    let width = histories.first().ok_or("at least one history is required")?.len();
    if width == 0 || histories.iter().any(|h| h.len() != width) {
        return Err("histories must have one common positive length");
    }
    if sequences.len() < 2 {
        return Err("at least two independent assignment sequences are required");
    }
    for (i, h) in histories.iter().enumerate() {
        if histories[..i].contains(h) {
            return Err("duplicate requested history");
        }
    }
    let mut scores = Vec::with_capacity(sequences.len());
    let mut support = vec![0; histories.len()];
    let mut eligible = 0_usize;
    for sequence in sequences {
        let n = sequence.outcomes.len();
        if n < width
            || sequence.schedules.is_empty()
            || sequence.schedules.len() != sequence.probabilities.len()
            || sequence.realized >= sequence.schedules.len()
            || sequence.schedules.iter().any(|s| s.len() != n)
            || sequence.outcomes.iter().any(|y| !y.is_finite())
            || sequence.probabilities.iter().any(|p| !p.is_finite() || *p <= 0.0)
            || (sequence.probabilities.iter().sum::<f64>() - 1.0).abs() > 1e-12
        {
            return Err("invalid finite schedule distribution or observed sequence");
        }
        for (i, schedule) in sequence.schedules.iter().enumerate() {
            if sequence.schedules[..i].contains(schedule) {
                return Err("duplicate assignment schedule");
            }
        }
        eligible = eligible.checked_add(n - width + 1).ok_or("period count overflow")?;
        let observed = &sequence.schedules[sequence.realized];
        let mut score = vec![0.0; histories.len()];
        for time in width - 1..n {
            for (j, history) in histories.iter().enumerate() {
                let range = time + 1 - width..=time;
                let probability = sequence
                    .schedules
                    .iter()
                    .zip(&sequence.probabilities)
                    .filter(|(schedule, _)| schedule[range.clone()] == history[..])
                    .map(|(_, probability)| probability)
                    .sum::<f64>();
                if probability <= 0.0 {
                    return Err("joint-history positivity failure at an eligible period");
                }
                if observed[range] == history[..] {
                    score[j] += sequence.outcomes[time] / probability;
                    support[j] += 1;
                }
            }
        }
        scores.push(score);
    }
    if support.contains(&0) {
        return Err("a requested history has no observed support");
    }
    let count = sequences.len() as f64;
    let total = eligible as f64;
    let means = (0..histories.len())
        .map(|j| scores.iter().map(|s| s[j]).sum::<f64>() / total)
        .collect::<Vec<_>>();
    // Center the cluster estimating equations, accounting for unequal sequence lengths.
    let mut covariance =
        vec![0.0; histories.len().checked_mul(histories.len()).ok_or("covariance size overflow")?];
    for (sequence, score) in sequences.iter().zip(&scores) {
        let periods = (sequence.outcomes.len() - width + 1) as f64;
        for j in 0..histories.len() {
            for k in 0..histories.len() {
                covariance[j * histories.len() + k] += count / (count - 1.0)
                    * (score[j] - periods * means[j])
                    * (score[k] - periods * means[k])
                    / total.powi(2);
            }
        }
    }
    if means.iter().chain(&covariance).chain(scores.iter().flatten()).any(|v| !v.is_finite()) {
        return Err("history-weighted response overflow");
    }
    Ok(HistorySwitchbackFit {
        means,
        covariance,
        sequence_scores: scores,
        observed_support: support,
        eligible_periods: eligible,
    })
}

#[cfg(test)]
mod history_tests {
    use super::*;

    fn fixture() -> Vec<HistoryAssignment> {
        let schedules = vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]];
        schedules
            .iter()
            .enumerate()
            .map(|(realized, schedule)| {
                let a = schedule[0] as f64;
                let b = schedule[1] as f64;
                HistoryAssignment {
                    schedules: schedules.clone(),
                    probabilities: vec![0.25; 4],
                    realized,
                    outcomes: vec![0.0, 2.0 * a + 3.0 * b + 5.0 * a * b],
                }
            })
            .collect()
    }

    #[test]
    fn finite_history_randomization_recovers_carryover_and_interaction() {
        let histories = vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]];
        let fit = switchback_history_means(&fixture(), &histories).unwrap();
        assert_eq!(fit.means, vec![0.0, 3.0, 2.0, 10.0]);
        assert_eq!(fit.means[3] - fit.means[2] - fit.means[1] + fit.means[0], 5.0);
        assert_eq!(fit.eligible_periods, 4);
        assert_eq!(fit.observed_support, vec![1; 4]);
        assert!(fit.covariance[1 * 4 + 2] < 0.0);
        for j in 0..4 {
            for k in 0..4 {
                assert_eq!(fit.covariance[j * 4 + k], fit.covariance[k * 4 + j]);
            }
        }
    }

    #[test]
    fn blocked_assignment_cannot_be_replaced_with_marginal_probabilities() {
        let schedules = vec![vec![0, 0], vec![1, 1]];
        let sequences = (0..2)
            .map(|realized| HistoryAssignment {
                schedules: schedules.clone(),
                probabilities: vec![0.5, 0.5],
                realized,
                outcomes: vec![0.0, realized as f64 * 4.0],
            })
            .collect::<Vec<_>>();
        let fit = switchback_history_means(&sequences, &[vec![0, 0], vec![1, 1]]).unwrap();
        assert_eq!(fit.means, vec![0.0, 4.0]);
        assert!(switchback_history_means(&sequences, &[vec![0, 1]]).is_err());
    }

    #[test]
    fn malformed_design_and_unsupported_observations_are_refused() {
        let mut sequences = fixture();
        assert!(switchback_history_means(&sequences, &[vec![0], vec![0, 1]]).is_err());
        sequences[0].schedules[1] = sequences[0].schedules[0].clone();
        assert!(switchback_history_means(&sequences, &[vec![0, 0]]).is_err());
        let mut sequences = fixture();
        sequences[0].probabilities[0] = f64::NAN;
        assert!(switchback_history_means(&sequences, &[vec![0, 0]]).is_err());
        let mut sequences = fixture();
        for sequence in &mut sequences {
            sequence.realized = 0;
        }
        assert!(switchback_history_means(&sequences, &[vec![1, 1]]).is_err());
    }
}

/// Minimum independent assignment sequences for the pointwise interval.
pub(crate) const MIN_SEQUENCES_FOR_INTERVAL: usize = 30;
/// Minimum known marginal assignment probability for either arm.
pub(crate) const MIN_ARM_PROBABILITY_FOR_INTERVAL: f64 = 0.2;

/// Unit-period Horvitz–Thompson ITT with sequence-level uncertainty.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SwitchbackItt {
    /// Average effect over the observed unit-period schedule.
    pub effect: f64,
    /// Sandwich variance treating whole assignment sequences as independent.
    pub variance: f64,
    /// Pointwise 95% Student interval when the measured support gate is met.
    pub interval_95: Option<[f64; 2]>,
    /// Number of independently randomized sequences.
    pub sequences: usize,
}

/// Estimate a switchback ITT from known marginal randomization probabilities.
///
/// Returns `None` on invalid rows, fewer than two independent sequences, or
/// an unobserved treatment arm across the entire schedule. Both arms need not
/// appear within each sequence.
/// Within-sequence outcomes may depend arbitrarily on one another. Identification
/// additionally requires no treatment carryover and no between-sequence
/// interference; neither can be established from these observations. Equal
/// sequence lengths, at least 30 sequences, and 30 observed periods per arm
/// are required for an interval so that no sequence dominates the
/// period-weighted mean and both arm contributions are supported.
#[must_use]
pub fn switchback_itt(
    outcomes: &[f64],
    assignment: &[bool],
    probabilities: &[f64],
    sequence_ids: &[&str],
) -> Option<SwitchbackItt> {
    let n = outcomes.len();
    if n == 0 || assignment.len() != n || probabilities.len() != n || sequence_ids.len() != n {
        return None;
    }
    let mut grouped = BTreeMap::<&str, (f64, usize)>::new();
    let mut arm_counts = [0_usize; 2];
    let mut overlap = true;
    for i in 0..n {
        let (y, p) = (outcomes[i], probabilities[i]);
        if !y.is_finite()
            || !p.is_finite()
            || !(0.0..1.0).contains(&p)
            || p == 0.0
            || sequence_ids[i].is_empty()
        {
            return None;
        }
        overlap &= p + 1e-12 >= MIN_ARM_PROBABILITY_FOR_INTERVAL
            && 1.0 - p + 1e-12 >= MIN_ARM_PROBABILITY_FOR_INTERVAL;
        let group = grouped.entry(sequence_ids[i]).or_default();
        group.0 += if assignment[i] { y / p } else { -y / (1.0 - p) };
        group.1 += 1;
        arm_counts[usize::from(assignment[i])] += 1;
    }
    let sequences = grouped.len();
    if sequences < 2 || arm_counts.contains(&0) {
        return None;
    }
    let mean_total = grouped.values().map(|(score, _)| score).sum::<f64>() / sequences as f64;
    let effect = mean_total * sequences as f64 / n as f64;
    let variance = sequences as f64 / (sequences - 1) as f64
        * grouped.values().map(|(score, _)| (score - mean_total).powi(2)).sum::<f64>()
        / (n * n) as f64;
    if !effect.is_finite() || !variance.is_finite() || variance < 0.0 {
        return None;
    }
    let period_count = grouped.values().next().map_or(0, |(_, count)| *count);
    let balanced = period_count > 0 && grouped.values().all(|(_, count)| *count == period_count);
    let interval_95 = (sequences >= MIN_SEQUENCES_FOR_INTERVAL
        && balanced
        && overlap
        && arm_counts.iter().all(|count| *count >= 30)
        && variance > 0.0)
        .then(|| {
            let critical = antecedent_stats::student_t_ppf(0.975, (sequences - 1) as f64);
            let radius = critical * variance.sqrt();
            [effect - radius, effect + radius]
        });
    Some(SwitchbackItt { effect, variance, interval_95, sequences })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_truth_and_sparse_refusal() {
        let n = 32 * 12;
        let y = (0..n).map(|i| if i % 2 == 0 { 2.0 } else { 1.0 }).collect::<Vec<_>>();
        let a = (0..n).map(|i| i % 2 == 0).collect::<Vec<_>>();
        let ids = (0..n).map(|i| format!("s{}", i / 12)).collect::<Vec<_>>();
        let borrowed = ids.iter().map(String::as_str).collect::<Vec<_>>();
        let fit = switchback_itt(&y, &a, &vec![0.5; n], &borrowed).unwrap();
        assert!((fit.effect - 1.0).abs() < 1e-12);
        // Degenerate observed variance cannot support a calibrated interval.
        assert_eq!(fit.interval_95, None);
        assert!(switchback_itt(&y[..12], &a[..12], &[0.5; 12], &borrowed[..12]).is_none());
        let mut unequal = ids.clone();
        unequal[0] = "s1".into();
        let unequal_refs = unequal.iter().map(String::as_str).collect::<Vec<_>>();
        assert_eq!(switchback_itt(&y, &a, &vec![0.5; n], &unequal_refs).unwrap().interval_95, None);
        let weak = switchback_itt(&y, &a, &vec![0.199; n], &borrowed).unwrap();
        assert_eq!(weak.interval_95, None);
        assert!(switchback_itt(&y, &vec![true; n], &vec![0.5; n], &borrowed).is_none());
        assert!(switchback_itt(&y, &a, &vec![0.0; n], &borrowed).is_none());
    }

    #[test]
    fn one_arm_within_a_sequence_is_admissible_for_ht() {
        let n = 30 * 12;
        let assignment = (0..n).map(|i| i / 12 % 2 == 0).collect::<Vec<_>>();
        let outcomes = assignment.iter().map(|a| if *a { 1.0 } else { 0.0 }).collect::<Vec<_>>();
        let ids = (0..n).map(|i| format!("s{}", i / 12)).collect::<Vec<_>>();
        let refs = ids.iter().map(String::as_str).collect::<Vec<_>>();
        let fit = switchback_itt(&outcomes, &assignment, &vec![0.5; n], &refs).unwrap();
        assert_eq!(fit.sequences, 30);
        assert!(fit.interval_95.is_some());
    }
}
