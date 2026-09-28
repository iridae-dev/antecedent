//! Sequence-level inference for randomized switchback experiments.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

/// Minimum independent assignment sequences for the pointwise interval.
pub const MIN_SEQUENCES_FOR_INTERVAL: usize = 30;
/// Minimum known marginal assignment probability for either arm.
pub const MIN_ARM_PROBABILITY_FOR_INTERVAL: f64 = 0.2;

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
