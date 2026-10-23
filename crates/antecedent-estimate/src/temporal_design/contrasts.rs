//! Horvitz-Thompson excursion scores of the arm designs: two arms, several arms, categorical
//! levels, a two-factor factorial and within-cluster interference.
//!
//! Every function pairs decisions with outcomes `lag` periods later ([`super::lagged_pairs`])
//! and returns one score per pair and coordinate whose cluster ratio is the coordinate. The
//! arm-mean coordinate of arm `a` is `1{A_t = a} Y_{t+lag} / P(A_t = a | history)`; a contrast
//! is a difference of arm means, so its score is the difference of their scores.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{LaggedPairs, invalid};
use crate::error::EstimationError;
use crate::util::refuse;

/// Pair scores of one design, before they are summed within clusters.
#[derive(Clone, Debug, PartialEq)]
pub struct DesignScores {
    /// Coordinate names, in score order.
    pub coordinates: Vec<String>,
    /// Pair-major scores: `scores[p * coordinates.len() + k]`.
    pub scores: Vec<f64>,
    /// Independent cluster of each pair.
    pub cluster_of_pair: Vec<usize>,
    /// Contrast coordinates tested as one family (Holm), possibly empty.
    pub family: Vec<usize>,
    /// Coordinates a simultaneous band covers, possibly empty.
    pub band: Vec<usize>,
    /// Smallest probability of a realized assignment (or exposure) among the paired decisions.
    pub minimum_probability: f64,
    /// Number of pairs.
    pub pairs: usize,
}

/// Validate one known assignment probability against the positivity floor.
fn checked_probability(p: f64, floor: f64) -> Result<f64, EstimationError> {
    if !p.is_finite() || p <= 0.0 || p >= 1.0 {
        return Err(invalid("every assignment probability must be in (0, 1)"));
    }
    if p < floor {
        return Err(refuse(
            antecedent_core::reason_code!("positivity_violated"),
            "temporal_design.positivity",
            &format!(
                "an arm's assignment probability {p} is below the declared floor {floor}. Next \
                 attempt: a design whose every arm keeps at least the floor at every decision, \
                 or a lower declared floor"
            ),
        ));
    }
    Ok(p)
}

fn validate_floor(floor: f64) -> Result<(), EstimationError> {
    if floor.is_finite() && floor > 0.0 && floor <= 0.5 {
        Ok(())
    } else {
        Err(invalid("the positivity floor must be in (0, 0.5]"))
    }
}

fn require_aligned(rows: usize, lengths: &[usize]) -> Result<(), EstimationError> {
    if lengths.iter().any(|len| *len != rows) {
        return Err(invalid("design arrays must be aligned with the table rows"));
    }
    Ok(())
}

fn require_populated(counts: &[usize], labels: &[String]) -> Result<(), EstimationError> {
    if let Some(index) = counts.iter().position(|count| *count == 0) {
        return Err(refuse(
            antecedent_core::reason_code!("arm_not_populated"),
            "temporal_design.arm_not_populated",
            &format!("no paired decision realized arm {}", labels[index]),
        ));
    }
    Ok(())
}

/// Two-arm excursion scores: `[effect, treated mean, control mean]`.
///
/// # Errors
/// Misaligned arrays, a probability outside `(0, 1)` or below the floor, a non-finite outcome,
/// or an arm no paired decision realized.
pub fn two_arm_scores(
    pairs: &LaggedPairs,
    outcome: &[f64],
    assignment: &[bool],
    probability: &[f64],
    cluster: &[usize],
    floor: f64,
) -> Result<DesignScores, EstimationError> {
    validate_floor(floor)?;
    let rows = outcome.len();
    require_aligned(rows, &[assignment.len(), probability.len(), cluster.len()])?;
    let mut scores = Vec::with_capacity(3 * pairs.decision.len());
    let mut cluster_of_pair = Vec::with_capacity(pairs.decision.len());
    let mut arm_counts = [0_usize; 2];
    let mut minimum = 1.0_f64;
    for (&t, &o) in pairs.decision.iter().zip(&pairs.outcome) {
        let y = outcome[o];
        if !y.is_finite() {
            return Err(invalid("every paired outcome must be finite"));
        }
        let p = probability[t];
        let realized = if assignment[t] { p } else { 1.0 - p };
        // Positivity is a property of the design at this decision: every arm, not only the
        // realized one, must keep the floor.
        checked_probability(p, floor)?;
        checked_probability(1.0 - p, floor)?;
        minimum = minimum.min(realized);
        let treated = if assignment[t] { y / p } else { 0.0 };
        let control = if assignment[t] { 0.0 } else { y / (1.0 - p) };
        scores.extend_from_slice(&[treated - control, treated, control]);
        cluster_of_pair.push(cluster[t]);
        arm_counts[usize::from(assignment[t])] += 1;
    }
    let labels = vec!["control".to_owned(), "treated".to_owned()];
    require_populated(&arm_counts, &labels)?;
    Ok(DesignScores {
        coordinates: vec![
            "excursion_effect".to_owned(),
            "treated_mean".to_owned(),
            "control_mean".to_owned(),
        ],
        scores,
        cluster_of_pair,
        family: vec![0],
        band: Vec::new(),
        minimum_probability: minimum,
        pairs: pairs.decision.len(),
    })
}

/// Multi-arm (or categorical-level) excursion scores.
///
/// Coordinates: the contrast of every arm `k >= 1` against the reference arm 0, then, when
/// `ordered`, the adjacent steps `k - (k - 1)` for `k >= 2`, then every arm mean. The contrasts
/// against the reference form the Holm family; a simultaneous band covers the same contrasts.
///
/// # Errors
/// Fewer than three arms, misaligned arrays, a probability row that is not a distribution or
/// whose realized arm is below the floor, a non-finite outcome, or an arm no decision realized.
#[allow(
    clippy::too_many_arguments,
    reason = "one design: rows, pairs, outcome, assignment, probabilities, arms, clusters, floor"
)]
pub fn multi_arm_scores(
    pairs: &LaggedPairs,
    outcome: &[f64],
    assignment: &[usize],
    probabilities: &[Vec<f64>],
    arms: &[String],
    cluster: &[usize],
    ordered: bool,
    floor: f64,
) -> Result<DesignScores, EstimationError> {
    validate_floor(floor)?;
    let k = arms.len();
    if k < 3 {
        return Err(invalid("a multi-arm design declares at least three arms"));
    }
    let rows = outcome.len();
    require_aligned(rows, &[assignment.len(), probabilities.len(), cluster.len()])?;
    let steps = if ordered { k - 2 } else { 0 };
    let width = (k - 1) + steps + k;
    let mut scores = Vec::with_capacity(width * pairs.decision.len());
    let mut cluster_of_pair = Vec::with_capacity(pairs.decision.len());
    let mut arm_counts = vec![0_usize; k];
    let mut minimum = 1.0_f64;
    let mut means = vec![0.0; k];
    for (&t, &o) in pairs.decision.iter().zip(&pairs.outcome) {
        let y = outcome[o];
        if !y.is_finite() {
            return Err(invalid("every paired outcome must be finite"));
        }
        let row = &probabilities[t];
        let arm = assignment[t];
        if arm >= k
            || row.len() != k
            || row.iter().any(|p| !p.is_finite() || *p <= 0.0 || *p >= 1.0)
            || (row.iter().sum::<f64>() - 1.0).abs() > 1e-8
        {
            return Err(invalid(
                "each decision needs a probability row over every arm and a realized arm in range",
            ));
        }
        for &q in row {
            checked_probability(q, floor)?;
        }
        let p = row[arm];
        minimum = minimum.min(p);
        means.iter_mut().for_each(|value| *value = 0.0);
        means[arm] = y / p;
        for j in 1..k {
            scores.push(means[j] - means[0]);
        }
        for j in 2..(2 + steps) {
            scores.push(means[j] - means[j - 1]);
        }
        scores.extend_from_slice(&means);
        cluster_of_pair.push(cluster[t]);
        arm_counts[arm] += 1;
    }
    require_populated(&arm_counts, arms)?;
    let mut coordinates: Vec<String> =
        (1..k).map(|j| format!("contrast:{}-{}", arms[j], arms[0])).collect();
    coordinates.extend((2..(2 + steps)).map(|j| format!("step:{}-{}", arms[j], arms[j - 1])));
    coordinates.extend(arms.iter().map(|arm| format!("mean:{arm}")));
    let family: Vec<usize> = (0..(k - 1)).collect();
    Ok(DesignScores {
        coordinates,
        scores,
        cluster_of_pair,
        band: family.clone(),
        family,
        minimum_probability: minimum,
        pairs: pairs.decision.len(),
    })
}

/// Two-factor factorial excursion scores, the factors assigned independently at each decision.
///
/// Coordinates: the main effect of each factor (averaged over the other factor's levels), the
/// interaction `mu11 - mu10 - mu01 + mu00`, then the four cell means in the order `00, 10, 01,
/// 11` (first factor, then second). The two main effects and the interaction form the family.
///
/// # Errors
/// Misaligned arrays, a probability outside `(0, 1)` or a realized cell below the floor, a
/// non-finite outcome, or a cell no decision realized.
#[allow(
    clippy::too_many_arguments,
    reason = "one design: pairs, outcome, both factors with their probabilities, clusters, floor"
)]
pub fn factorial_scores(
    pairs: &LaggedPairs,
    outcome: &[f64],
    first: &[bool],
    first_probability: &[f64],
    second: &[bool],
    second_probability: &[f64],
    cluster: &[usize],
    floor: f64,
) -> Result<DesignScores, EstimationError> {
    validate_floor(floor)?;
    let rows = outcome.len();
    require_aligned(
        rows,
        &[
            first.len(),
            first_probability.len(),
            second.len(),
            second_probability.len(),
            cluster.len(),
        ],
    )?;
    let mut scores = Vec::with_capacity(7 * pairs.decision.len());
    let mut cluster_of_pair = Vec::with_capacity(pairs.decision.len());
    let mut cell_counts = [0_usize; 4];
    let mut minimum = 1.0_f64;
    for (&t, &o) in pairs.decision.iter().zip(&pairs.outcome) {
        let y = outcome[o];
        if !y.is_finite() {
            return Err(invalid("every paired outcome must be finite"));
        }
        let (pa, pb) = (first_probability[t], second_probability[t]);
        for p in [pa, pb] {
            if !p.is_finite() || p <= 0.0 || p >= 1.0 {
                return Err(invalid("every factor probability must be in (0, 1)"));
            }
        }
        let realized_a = if first[t] { pa } else { 1.0 - pa };
        let realized_b = if second[t] { pb } else { 1.0 - pb };
        for (qa, qb) in [(pa, pb), (pa, 1.0 - pb), (1.0 - pa, pb), (1.0 - pa, 1.0 - pb)] {
            checked_probability(qa * qb, floor)?;
        }
        let cell_probability = realized_a * realized_b;
        minimum = minimum.min(cell_probability);
        let cell = usize::from(first[t]) + 2 * usize::from(second[t]);
        let mut mu = [0.0; 4];
        mu[cell] = y / cell_probability;
        let main_first = 0.5 * ((mu[1] - mu[0]) + (mu[3] - mu[2]));
        let main_second = 0.5 * ((mu[2] - mu[0]) + (mu[3] - mu[1]));
        let interaction = mu[3] - mu[1] - mu[2] + mu[0];
        scores.extend_from_slice(&[main_first, main_second, interaction]);
        scores.extend_from_slice(&mu);
        cluster_of_pair.push(cluster[t]);
        cell_counts[cell] += 1;
    }
    let cells: Vec<String> = ["00", "10", "01", "11"].iter().map(|c| format!("cell:{c}")).collect();
    require_populated(&cell_counts, &cells)?;
    let mut coordinates = vec![
        "main_effect:first".to_owned(),
        "main_effect:second".to_owned(),
        "interaction".to_owned(),
    ];
    coordinates.extend(cells.iter().map(|cell| format!("mean:{cell}")));
    Ok(DesignScores {
        coordinates,
        scores,
        cluster_of_pair,
        family: vec![0, 1, 2],
        band: vec![0, 1, 2],
        minimum_probability: minimum,
        pairs: pairs.decision.len(),
    })
}

/// Within-cluster interference excursion scores.
///
/// At each decision period a unit's exposure is its own assignment and whether any other unit
/// of its interference cluster was treated in the same period. Assignments are independent
/// across units given the recorded probabilities, so the probability of exposure `(a, g)` is
/// `P(A = a) * P(G = g)` with `P(G = 0) = prod_j (1 - p_j)` over the other members observed that
/// period. Coordinates: the direct effect `mu(1,0) - mu(0,0)`, the spillover effect
/// `mu(0,1) - mu(0,0)`, the total effect `mu(1,1) - mu(0,0)`, then the four exposure means in
/// the order `(0,0), (1,0), (0,1), (1,1)`. Inference clusters are the interference clusters.
///
/// # Errors
/// Misaligned arrays, a decision with no other cluster member that period, a probability
/// outside `(0, 1)` or a realized exposure below the floor, a non-finite outcome, or an
/// exposure no decision realized.
#[allow(
    clippy::too_many_arguments,
    reason = "one design: pairs, outcome, assignment, probabilities, clusters, periods, floor"
)]
pub fn interference_scores(
    pairs: &LaggedPairs,
    outcome: &[f64],
    assignment: &[bool],
    probability: &[f64],
    group: &[usize],
    period: &[i64],
    floor: f64,
) -> Result<DesignScores, EstimationError> {
    validate_floor(floor)?;
    let rows = outcome.len();
    require_aligned(rows, &[assignment.len(), probability.len(), group.len(), period.len()])?;
    if probability.iter().any(|p| !p.is_finite() || *p <= 0.0 || *p >= 1.0) {
        return Err(invalid("every assignment probability must be in (0, 1)"));
    }
    // Members of each (cluster, period), by row.
    let mut members: std::collections::HashMap<(usize, i64), Vec<usize>> =
        std::collections::HashMap::new();
    for row in 0..rows {
        members.entry((group[row], period[row])).or_default().push(row);
    }
    let mut scores = Vec::with_capacity(7 * pairs.decision.len());
    let mut cluster_of_pair = Vec::with_capacity(pairs.decision.len());
    let mut exposure_counts = [0_usize; 4];
    let mut minimum = 1.0_f64;
    for (&t, &o) in pairs.decision.iter().zip(&pairs.outcome) {
        let y = outcome[o];
        if !y.is_finite() {
            return Err(invalid("every paired outcome must be finite"));
        }
        let peers = &members[&(group[t], period[t])];
        if peers.len() < 2 {
            return Err(refuse(
                antecedent_core::reason_code!("invalid_argument"),
                "temporal_design.isolated_unit",
                "a decision has no other member of its interference cluster in that period, so \
                 its spillover exposure is not defined",
            ));
        }
        let mut none_treated = 1.0;
        let mut any_treated = false;
        for &peer in peers.iter().filter(|peer| **peer != t) {
            none_treated *= 1.0 - probability[peer];
            any_treated |= assignment[peer];
        }
        let p_own = if assignment[t] { probability[t] } else { 1.0 - probability[t] };
        let p_peers = if any_treated { 1.0 - none_treated } else { none_treated };
        let own = probability[t];
        for (q_own, q_peers) in [
            (own, none_treated),
            (own, 1.0 - none_treated),
            (1.0 - own, none_treated),
            (1.0 - own, 1.0 - none_treated),
        ] {
            checked_probability(q_own * q_peers, floor)?;
        }
        let exposure_probability = p_own * p_peers;
        minimum = minimum.min(exposure_probability);
        let exposure = usize::from(assignment[t]) + 2 * usize::from(any_treated);
        let mut mu = [0.0; 4];
        mu[exposure] = y / exposure_probability;
        scores.extend_from_slice(&[mu[1] - mu[0], mu[2] - mu[0], mu[3] - mu[0]]);
        scores.extend_from_slice(&mu);
        cluster_of_pair.push(group[t]);
        exposure_counts[exposure] += 1;
    }
    let exposures: Vec<String> = ["own0_peers0", "own1_peers0", "own0_peers1", "own1_peers1"]
        .iter()
        .map(|e| (*e).to_owned())
        .collect();
    require_populated(&exposure_counts, &exposures)?;
    let mut coordinates =
        vec!["direct_effect".to_owned(), "spillover_effect".to_owned(), "total_effect".to_owned()];
    coordinates.extend(exposures.iter().map(|exposure| format!("mean:{exposure}")));
    Ok(DesignScores {
        coordinates,
        scores,
        cluster_of_pair,
        family: vec![0, 1, 2],
        band: vec![0, 1, 2],
        minimum_probability: minimum,
        pairs: pairs.decision.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::super::{
        ClusterScores, PanelRows, bayesian_bootstrap, lagged_pairs, sampling_summary,
    };
    use super::*;
    use antecedent_core::ExecutionContext;

    /// A deterministic uniform in `[0, 1)` from an integer.
    fn uniform(seed: u64) -> f64 {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
        state ^= state >> 31;
        state = state.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        state ^= state >> 29;
        (state >> 11) as f64 / (1_u64 << 53) as f64
    }

    type Panel = (Vec<usize>, Vec<i64>, Vec<bool>, Vec<f64>, Vec<f64>);

    /// Units over periods; assignment probability depends on the previous outcome; outcome at
    /// `t + 1` carries `effect * A_t` plus carryover from `A_{t-1}` and unit noise.
    fn panel(units: u64, periods: u64, effect: f64) -> Panel {
        let mut unit = Vec::new();
        let mut period = Vec::new();
        let mut assignment = Vec::new();
        let mut probability = Vec::new();
        let mut outcome = Vec::new();
        for i in 0..units {
            let level = uniform(i * 7919) - 0.5;
            let mut previous_y = 0.0;
            let mut previous_a = 0.0;
            let mut before_previous_a = 0.0;
            for t in 0..periods {
                let p = if previous_y > 0.0 { 0.6 } else { 0.3 };
                let a = uniform(i * 1000 + t + 17) < p;
                let noise = uniform(i * 1000 + t + 99_991) - 0.5;
                let y = level + effect * previous_a + 0.25 * before_previous_a + noise;
                unit.push(usize::try_from(i).unwrap());
                period.push(i64::try_from(t).unwrap());
                assignment.push(a);
                probability.push(p);
                outcome.push(y);
                previous_y = y;
                before_previous_a = previous_a;
                previous_a = f64::from(u8::from(a));
            }
        }
        (unit, period, assignment, probability, outcome)
    }

    #[test]
    fn the_lag_one_excursion_recovers_the_simulated_effect() {
        let (unit, period, assignment, probability, outcome) = panel(400, 12, 1.5);
        let pairs = lagged_pairs(PanelRows { unit: &unit, period: &period }, 1).unwrap();
        let design =
            two_arm_scores(&pairs, &outcome, &assignment, &probability, &unit, 0.05).unwrap();
        let accumulated =
            ClusterScores::accumulate(&design.scores, 3, &design.cluster_of_pair).unwrap();
        let summary = sampling_summary(&accumulated).unwrap();
        let effect = summary.estimate[0];
        assert!((effect - 1.5).abs() < 4.0 * summary.standard_error[0], "{effect}");
        assert!(summary.standard_error[0] > 0.0 && summary.standard_error[0] < 0.3);
        let ctx = ExecutionContext::for_tests(3);
        let posterior = bayesian_bootstrap(&accumulated, 400, &ctx).unwrap();
        assert!((posterior.mean[0] - effect).abs() < 0.1);
        let relative = posterior.standard_deviation[0] / summary.standard_error[0];
        assert!((0.7..1.3).contains(&relative), "{relative}");
    }

    #[test]
    fn a_probability_below_the_floor_is_refused() {
        let (unit, period, assignment, mut probability, outcome) = panel(20, 4, 1.0);
        probability[0] = 0.01;
        let pairs = lagged_pairs(PanelRows { unit: &unit, period: &period }, 0).unwrap();
        let result = two_arm_scores(&pairs, &outcome, &assignment, &probability, &unit, 0.05);
        assert!(matches!(result, Err(EstimationError::Refused { .. })));
    }

    #[test]
    fn interference_needs_a_peer_and_separates_direct_and_spillover() {
        // Clusters of two units over many periods; spillover adds 0.5 when the peer is treated.
        let mut unit = Vec::new();
        let mut period = Vec::new();
        let mut group = Vec::new();
        let mut assignment = Vec::new();
        let mut probability = Vec::new();
        for c in 0..300_u64 {
            for t in 0..6_u64 {
                for member in 0..2_u64 {
                    let i = c * 2 + member;
                    unit.push(usize::try_from(i).unwrap());
                    period.push(i64::try_from(t).unwrap());
                    group.push(usize::try_from(c).unwrap());
                    assignment.push(uniform(i * 100 + t + 5) < 0.5);
                    probability.push(0.5);
                }
            }
        }
        let rows = unit.len();
        let mut outcome = vec![0.0; rows];
        for row in 0..rows {
            let peer = if row % 2 == 0 { row + 1 } else { row - 1 };
            let noise = uniform(u64::try_from(row).unwrap() + 4242) - 0.5;
            outcome[row] = 1.0 * f64::from(u8::from(assignment[row]))
                + 0.5 * f64::from(u8::from(assignment[peer]))
                + noise;
        }
        let pairs = lagged_pairs(PanelRows { unit: &unit, period: &period }, 0).unwrap();
        let design =
            interference_scores(&pairs, &outcome, &assignment, &probability, &group, &period, 0.05)
                .unwrap();
        let accumulated =
            ClusterScores::accumulate(&design.scores, 7, &design.cluster_of_pair).unwrap();
        let summary = sampling_summary(&accumulated).unwrap();
        assert!((summary.estimate[0] - 1.0).abs() < 4.0 * summary.standard_error[0]);
        assert!((summary.estimate[1] - 0.5).abs() < 4.0 * summary.standard_error[1]);
        assert!((summary.estimate[2] - 1.5).abs() < 4.0 * summary.standard_error[2]);
    }
}
