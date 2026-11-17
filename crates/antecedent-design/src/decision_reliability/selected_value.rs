//! A whole-procedure interval for the value of the decision a search selected.
//!
//! The finite routes of this module report the selected value *under the supplied law* and
//! keep the law's own spread as a separate component. A search over a continuous program
//! space evaluates many proposals on the same aligned draws of one fitted response law and
//! reports the best one found, so the statement a reader needs is about the whole
//! procedure: does the reported interval contain the value of the program the search
//! returned, when the law is the fit's posterior or estimator-sampling law and the program
//! was chosen from every proposal the search looked at?
//!
//! The interval is simultaneous over every evaluated proposal that was retained, so it
//! holds for whichever of them the search ended on, whatever the dependence between the
//! proposals: with `u_ij` the utility of proposal `j` at aligned draw `i`, `m_j` and `s_j`
//! its mean and standard deviation over the draws, the critical value is the `level`
//! quantile over draws of `max_j |u_ij - m_j| / s_j` (the sup-t band of the shared joint
//! law), and the interval of proposal `j` is `m_j +- c s_j`. Pairing by draw index is what
//! makes this dependence-aware: neighbouring proposals of a pattern search are almost
//! perfectly correlated, so the band costs little beyond the ordinary interval, while a
//! Bonferroni correction over the same count would. Weighted laws use the weighted mean,
//! standard deviation and quantile.
//!
//! Like every route here it is published with a calibration annotation and never withheld:
//! [`selected_value_calibration`] resolves the coverage record of the construction at the
//! coordinate (candidate-set size, draw budget, selection rule, coupling, dependence and
//! the adaptive origin of the candidates).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "bounded draw counts and an order-statistic rank"
)]

use antecedent_io_core::contract_section::CalibrationSlotWire;

use super::ReliabilityError;
use super::calibration::{CalibrationCoordinate, selected_value_calibration};
use super::draws::{CandidateOrigin, DecisionDraws, SelectionRule};
use super::interval::{MAX_INTERVAL_LEVEL, MIN_INTERVAL_LEVEL};

/// Interval for the value of the decision a search selected, with its calibration.
#[derive(Clone, Debug, PartialEq)]
pub struct SelectedValueInterval {
    /// The selected candidate.
    pub selected_id: String,
    /// Mean utility of the selected candidate over the draws of the supplied law.
    pub value: f64,
    /// Standard deviation of its utility over the draws.
    pub source_standard_deviation: f64,
    /// Candidates the search considered (at least the retained set).
    pub n_considered: u64,
    /// Candidates the band holds over.
    pub n_retained: usize,
    /// The sup-t critical value, in standard deviations.
    pub critical_value: f64,
    /// Lower end of the simultaneous interval.
    pub lower: f64,
    /// Upper end of the simultaneous interval.
    pub upper: f64,
    /// Nominal level.
    pub level: f64,
    /// The coordinate a coverage record must measure.
    pub coordinate: CalibrationCoordinate,
    /// Calibration of this construction at the coordinate; an annotation, never a gate.
    pub calibration: CalibrationSlotWire,
}

/// Weighted mean and standard deviation of one candidate's draws. Unweighted draws use the
/// `n - 1` divisor.
fn moments(values: &[f64], weights: Option<&[f64]>) -> (f64, f64) {
    let n = values.len();
    match weights {
        None => {
            let mean = values.iter().sum::<f64>() / n as f64;
            let variance =
                values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n as f64 - 1.0);
            (mean, variance.max(0.0).sqrt())
        }
        Some(weights) => {
            let total: f64 = weights.iter().sum();
            let mean = values.iter().zip(weights).map(|(v, w)| v * w).sum::<f64>() / total;
            let variance =
                values.iter().zip(weights).map(|(v, w)| w * (v - mean) * (v - mean)).sum::<f64>()
                    / total;
            (mean, variance.max(0.0).sqrt())
        }
    }
}

/// The `level` quantile of `scores` (weighted when `weights` is given).
fn critical_quantile(scores: &[f64], weights: Option<&[f64]>, level: f64) -> f64 {
    let n = scores.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| scores[a].total_cmp(&scores[b]));
    match weights {
        None => {
            // The order statistic that holds the level with the finite-sample margin of one
            // draw: rank ceil((n + 1) level), capped at the maximum.
            let rank = (((n as f64 + 1.0) * level).ceil() as usize).clamp(1, n);
            scores[order[rank - 1]]
        }
        Some(weights) => {
            let total: f64 = weights.iter().sum();
            let mut cumulative = 0.0;
            for &i in &order {
                cumulative += weights[i];
                if cumulative >= level * total {
                    return scores[i];
                }
            }
            scores[order[n - 1]]
        }
    }
}

/// The simultaneous interval for the value of `selected_id` among the candidates of `draws`.
///
/// `draws` must carry the candidates on one shared simulation (aligned draws), with the
/// declarations the carrier always requires; an adaptively generated set needs its search
/// record, and `n_considered` of the coordinate counts every candidate the search looked at.
///
/// # Errors
/// An invalid carrier, an unaligned (independently drawn) candidate set with more than one
/// candidate, an unknown `selected_id`, or a level outside `[0.5, 0.9999]`.
pub fn selected_value_interval(
    draws: &DecisionDraws,
    selected_id: &str,
    level: f64,
) -> Result<SelectedValueInterval, ReliabilityError> {
    draws.validate()?;
    if !(level.is_finite() && (MIN_INTERVAL_LEVEL..=MAX_INTERVAL_LEVEL).contains(&level)) {
        return Err(ReliabilityError::invalid("interval_level", "level must lie in [0.5, 0.9999]"));
    }
    if draws.candidates.len() > 1
        && draws.coupling != super::draws::CandidateCoupling::SharedSimulation
    {
        return Err(ReliabilityError::JointLawRequired {
            reason: "a simultaneous band over candidates needs draws of one shared law",
        });
    }
    let selected = draws.candidate(selected_id).ok_or_else(|| {
        ReliabilityError::invalid("selected_id", "the selected id is not a candidate")
    })?;
    let weights = draws.weights.as_deref();
    let n = draws.n_draws();
    let stats: Vec<(f64, f64)> =
        draws.candidates.iter().map(|c| moments(&c.utilities, weights)).collect();
    let scores: Vec<f64> = (0..n)
        .map(|i| {
            draws
                .candidates
                .iter()
                .zip(&stats)
                .filter(|(_, (_, sd))| *sd > 0.0)
                .map(|(c, (mean, sd))| ((c.utilities[i] - mean) / sd).abs())
                .fold(0.0_f64, f64::max)
        })
        .collect();
    let critical = critical_quantile(&scores, weights, level);
    let index = draws.candidates.iter().position(|c| c.id == selected.id).ok_or_else(|| {
        ReliabilityError::invalid("selected_id", "the selected id is not a candidate")
    })?;
    let (value, sd) = stats[index];
    let half = critical * sd;
    let coordinate = CalibrationCoordinate {
        candidate_set_size: draws.n_considered(),
        mc_budget: n as u64,
        selection_rule: draws
            .search_record
            .as_ref()
            .map_or(SelectionRule::MaximizeExpectedUtility, |r| r.selection_rule.clone())
            .kind(),
        functional: "expected_utility".to_owned(),
        coupling: draws.coupling,
        paradigm: draws.source,
        within_stream: draws.within_stream,
        importance_weighted: draws.weights.is_some(),
        adaptively_generated: draws.origin == CandidateOrigin::AdaptivelyGenerated,
    };
    let calibration = selected_value_calibration(&coordinate, level);
    Ok(SelectedValueInterval {
        selected_id: selected.id.clone(),
        value,
        source_standard_deviation: sd,
        n_considered: coordinate.candidate_set_size,
        n_retained: draws.candidates.len(),
        critical_value: critical,
        lower: value - half,
        upper: value + half,
        level,
        coordinate,
        calibration,
    })
}
