//! Nearest-neighbor contrast on observed histories.
//!
//! Each unit has a time-ordered history and one later outcome. The match score is the mean of
//! that history. A slope fitted to the history, or a coefficient from a contemporaneous
//! regression, is not the score and is not computed. The contrast is the sampling-weight mean of
//! treated-minus-matched-control outcome differences, with no outcome regression.
//!
//! Equal scores collapse when the donor index is built. The lowest input index is the only donor
//! a later query can meet, and a donor may be reused. A caliper is a maximum absolute distance on
//! the score: finite and non-negative, and a distance equal to the caliper is a match.
//!
//! [`FrequentistHistoryContrast`] and [`BayesianHistoryContrast`] are different objects. A
//! [`ContrastPrior`] is not a sampling weight and does not enter the weight total. No posterior
//! for this contrast is constructed: [`BayesianHistoryContrast::refuse_posterior`] names the
//! missing premise and the next attempt (see [`crate::matching_posterior_boundary`]).
//!
//! [`PeriodHistoryContrast`] is the temporal matching route. Units carry a calendar period
//! ([`HistoryUnit::in_period`]); a treated unit is matched only to a control of its own period,
//! so a match never crosses a unit or a calendar boundary, and a period with no control leaves
//! its treated units unmatched instead of borrowing a donor from another period. The estimand is
//! the sampling-weighted effect on the matched treated units, conditional on the period and on
//! the history-mean score. It is identified under unconfoundedness given that score within a
//! period, overlap of the scores within the period, a stable unit and no interference. A mean
//! history is a coarse sufficient statistic: it is the caller's assumption that it carries the
//! confounding. Its variance is the Abadie-Imbens (2006) conditional form for one match with
//! replacement and a known scalar score, with each unit's outcome variance estimated from its
//! nearest same-arm neighbour in the period; for a scalar score the matching discrepancy bias is
//! of smaller order than `n^-1/2`, so no bias adjustment is applied. A time-series regression is
//! not a substitute.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::cmp::Ordering;
use std::collections::BTreeMap;

use crate::EstimationError;
use crate::temporal_design::ClusterScores;
use crate::temporal_design::serial::{newey_west_bandwidth, serial_sampling_summary};
use crate::matching_posterior_boundary::{MatchingPosteriorRequest, refuse_matching_posterior};

/// One unit: a history, the outcome contrasted after it, and a sampling weight.
#[derive(Clone, Debug)]
pub struct HistoryUnit {
    history: Vec<f64>,
    outcome: f64,
    treated: bool,
    weight: f64,
    period: i64,
}

impl HistoryUnit {
    /// Build a unit.
    ///
    /// `weight` is a sampling weight. It may be zero or negative; the contrast later rejects a
    /// non-positive total on the matched treated units. A prior does not belong in `weight`.
    ///
    /// # Errors
    /// The history is empty or non-finite, its mean is non-finite, or the outcome or weight is
    /// non-finite.
    pub fn try_new(
        history: &[f64],
        outcome: f64,
        treated: bool,
        weight: f64,
    ) -> Result<Self, EstimationError> {
        if history.is_empty() || history.iter().any(|value| !value.is_finite()) {
            return Err(EstimationError::data_msg(
                "history match requires a non-empty finite history",
            ));
        }
        let unit = Self { history: history.to_vec(), outcome, treated, weight, period: 0 };
        if !unit.score().is_finite() {
            return Err(EstimationError::data_msg("history match score must be finite"));
        }
        if !outcome.is_finite() {
            return Err(EstimationError::data_msg("history match outcome must be finite"));
        }
        if !weight.is_finite() {
            return Err(EstimationError::data_msg("history match sampling weight must be finite"));
        }
        Ok(unit)
    }

    /// Calendar period (or treatment cohort) the unit is observed in. A match inside a period
    /// never reaches a donor of another period. Units start in period `0`.
    #[must_use]
    pub const fn in_period(mut self, period: i64) -> Self {
        self.period = period;
        self
    }

    /// Mean of the observed history.
    ///
    /// Every time point enters with equal weight. This is not a regression coefficient.
    #[must_use]
    pub fn score(&self) -> f64 {
        self.history.iter().sum::<f64>() / self.history.len() as f64
    }

    /// Number of history time points.
    pub(crate) fn history_len(&self) -> usize {
        self.history.len()
    }

    /// The outcome contrasted after the history.
    pub(crate) const fn outcome(&self) -> f64 {
        self.outcome
    }

    /// Whether the unit is in the treated arm.
    pub(crate) const fn is_treated(&self) -> bool {
        self.treated
    }

    /// Sampling weight.
    pub(crate) const fn weight(&self) -> f64 {
        self.weight
    }

    /// Calendar period.
    pub(crate) const fn period(&self) -> i64 {
        self.period
    }
}

/// Normal prior for a history-matched contrast.
///
/// The mean and variance are prior parameters. They are not sampling weights and they are not
/// added into a weight total.
#[derive(Clone, Copy, Debug)]
pub struct ContrastPrior {
    mean: f64,
    variance: f64,
}

impl ContrastPrior {
    /// Build a prior.
    ///
    /// # Errors
    /// `mean` is non-finite, or `variance` is non-finite or not strictly positive.
    pub fn try_new(mean: f64, variance: f64) -> Result<Self, EstimationError> {
        if !mean.is_finite() {
            return Err(EstimationError::data_msg("contrast prior mean must be finite"));
        }
        if !variance.is_finite() || variance <= 0.0 {
            return Err(EstimationError::data_msg(
                "contrast prior variance must be finite and positive",
            ));
        }
        Ok(Self { mean, variance })
    }

    /// Prior mean.
    #[must_use]
    pub const fn mean(self) -> f64 {
        self.mean
    }

    /// Prior variance.
    #[must_use]
    pub const fn variance(self) -> f64 {
        self.variance
    }
}

/// Frequentist nearest-neighbor contrast on history scores.
#[derive(Clone, Debug)]
pub struct FrequentistHistoryContrast {
    caliper: Option<f64>,
}

impl Default for FrequentistHistoryContrast {
    fn default() -> Self {
        Self::new()
    }
}

impl FrequentistHistoryContrast {
    /// Match with no caliper. Every treated unit uses its nearest control history.
    #[must_use]
    pub const fn new() -> Self {
        Self { caliper: None }
    }

    /// Match only when the absolute score distance is at most `caliper`.
    ///
    /// # Errors
    /// `caliper` is non-finite or negative.
    pub fn try_with_caliper(caliper: f64) -> Result<Self, EstimationError> {
        validate_caliper(Some(caliper))?;
        Ok(Self { caliper: Some(caliper) })
    }

    /// Weighted mean of treated-minus-matched-control outcomes.
    ///
    /// # Errors
    /// The histories cannot be matched, no treated unit falls inside the caliper, or the
    /// sampling weights on the matched treated units have a non-positive total.
    pub fn contrast(
        &self,
        units: &[HistoryUnit],
    ) -> Result<FrequentistHistoryEstimate, EstimationError> {
        let matched = match_histories(self.caliper, units)?;
        Ok(FrequentistHistoryEstimate {
            contrast: matched.contrast,
            weight_total: matched.weight_total,
            collapsed_donors: matched.collapsed_donors,
            matched_queries: matched.queries,
            matched_donors: matched.donors,
        })
    }
}

/// Frequentist history-matched contrast.
#[must_use]
#[derive(Clone, Debug)]
pub struct FrequentistHistoryEstimate {
    /// Sampling-weighted mean of treated-minus-matched-control outcomes.
    pub contrast: f64,
    /// Sum of sampling weights on matched treated units.
    pub weight_total: f64,
    /// Control histories left after equal scores were collapsed at index build.
    pub collapsed_donors: usize,
    /// Input indices of matched treated units, in input order.
    pub matched_queries: Vec<usize>,
    /// Input index of the control history each matched query used.
    pub matched_donors: Vec<usize>,
}

/// Bayesian history match. The prior stays a prior.
///
/// Calling [`Self::contrast`] reports the same matched contrast as
/// [`FrequentistHistoryContrast`] and carries the prior beside it. [`Self::refuse_posterior`]
/// does not turn that prior into a weight or into a posterior.
#[derive(Clone, Debug)]
pub struct BayesianHistoryContrast {
    caliper: Option<f64>,
    prior: ContrastPrior,
}

impl BayesianHistoryContrast {
    /// Match with `prior` held out of the sampling weights.
    ///
    /// # Errors
    /// `caliper` is `Some` and is non-finite or negative.
    pub fn try_new(prior: ContrastPrior, caliper: Option<f64>) -> Result<Self, EstimationError> {
        validate_caliper(caliper)?;
        Ok(Self { caliper, prior })
    }

    /// History-matched contrast, with the prior unchanged.
    ///
    /// # Errors
    /// The histories cannot be matched, no treated unit falls inside the caliper, or the
    /// sampling weights on the matched treated units have a non-positive total.
    pub fn contrast(
        &self,
        units: &[HistoryUnit],
    ) -> Result<BayesianHistoryEstimate, EstimationError> {
        let matched = match_histories(self.caliper, units)?;
        Ok(BayesianHistoryEstimate {
            contrast: matched.contrast,
            weight_total: matched.weight_total,
            collapsed_donors: matched.collapsed_donors,
            matched_queries: matched.queries,
            matched_donors: matched.donors,
            prior: self.prior,
        })
    }

    /// Match on history, then refuse a posterior.
    ///
    /// A successful match still returns an error. The prior is not written into the sampling
    /// weights, and a time-series regression is not returned in its place. A match failure is
    /// returned as that failure. The refusal names the missing premise and the next attempt.
    #[must_use]
    pub fn refuse_posterior(&self, units: &[HistoryUnit]) -> EstimationError {
        if let Err(error) = match_histories(self.caliper, units) {
            return error;
        }
        refuse_matching_posterior(MatchingPosteriorRequest::WithinPeriodHistory)
    }
}

/// History-matched contrast with the prior kept separate from the weight total.
#[must_use]
#[derive(Clone, Debug)]
pub struct BayesianHistoryEstimate {
    /// Sampling-weighted mean of treated-minus-matched-control outcomes.
    pub contrast: f64,
    /// Sum of sampling weights on matched treated units. The prior is not included.
    pub weight_total: f64,
    /// Control histories left after equal scores were collapsed at index build.
    pub collapsed_donors: usize,
    /// Input indices of matched treated units, in input order.
    pub matched_queries: Vec<usize>,
    /// Input index of the control history each matched query used.
    pub matched_donors: Vec<usize>,
    /// Prior supplied to the Bayesian object. It did not enter `weight_total`.
    pub prior: ContrastPrior,
}

/// Frequentist nearest-neighbor contrast on history scores, matched within calendar periods.
#[derive(Clone, Debug)]
pub struct PeriodHistoryContrast {
    caliper: Option<f64>,
}

impl PeriodHistoryContrast {
    /// Match within each period, with `caliper` as the maximum absolute score distance (a
    /// distance equal to it is a match) or no caliper.
    ///
    /// # Errors
    /// `caliper` is `Some` and is non-finite or negative.
    pub fn try_new(caliper: Option<f64>) -> Result<Self, EstimationError> {
        validate_caliper(caliper)?;
        Ok(Self { caliper })
    }

    /// Weighted mean of treated-minus-matched-control outcomes, each treated unit matched to the
    /// nearest control history of its own period (lowest input index on a tie, donors reused).
    ///
    /// # Errors
    /// The histories differ in length, no treated unit has a control within its period and the
    /// caliper, or the sampling weights on the matched treated units have a non-positive total.
    pub fn contrast(
        &self,
        units: &[HistoryUnit],
    ) -> Result<PeriodHistoryEstimate, EstimationError> {
        validate_caliper(self.caliper)?;
        let Some(first) = units.first() else {
            return Err(EstimationError::data_msg(
                "history match requires treated and control histories",
            ));
        };
        if units.iter().any(|unit| unit.history.len() != first.history.len()) {
            return Err(EstimationError::data_msg(
                "history match requires every history to have the same length",
            ));
        }
        let mut periods: BTreeMap<i64, (Vec<Row>, Vec<Row>)> = BTreeMap::new();
        for (index, unit) in units.iter().enumerate() {
            let row =
                Row { index, outcome: unit.outcome, weight: unit.weight, score: unit.score() };
            let (treated, control) = periods.entry(unit.period).or_default();
            if unit.treated {
                treated.push(row);
            } else {
                control.push(row);
            }
        }
        let mut pairs = Vec::new();
        let mut periods_matched = 0_usize;
        let mut noise: BTreeMap<usize, f64> = BTreeMap::new();
        for (treated, control) in periods.values() {
            if treated.is_empty() || control.is_empty() {
                continue;
            }
            let (found, _) = match_rows(self.caliper, treated, control);
            periods_matched += usize::from(!found.is_empty());
            if !found.is_empty() {
                within_arm_noise(treated, &mut noise);
                within_arm_noise(control, &mut noise);
            }
            pairs.extend(found);
        }
        pairs.sort_by_key(|pair| pair.query);
        if pairs.is_empty() {
            return Err(EstimationError::data_msg("no matched units within caliper and period"));
        }
        let weight_total: f64 = pairs.iter().map(|pair| pair.weight).sum();
        if weight_total.partial_cmp(&0.0_f64) != Some(Ordering::Greater) {
            return Err(EstimationError::data_msg(
                "history-matched contrast weights left no positive mass on matched units",
            ));
        }
        let contrast =
            pairs.iter().map(|pair| pair.weight * pair.difference).sum::<f64>() / weight_total;
        let noise_of = |index: usize| noise.get(&index).copied();
        let query_variances: Option<Vec<f64>> =
            pairs.iter().map(|pair| noise_of(pair.query)).collect();
        let donor_variances: Option<Vec<f64>> =
            pairs.iter().map(|pair| noise_of(pair.donor)).collect();
        let (standard_error, query_variances, donor_variances) =
            match (query_variances, donor_variances) {
                (Some(query), Some(donor)) => {
                    let weights: Vec<f64> = pairs.iter().map(|pair| pair.weight).collect();
                    let donors: Vec<usize> = pairs.iter().map(|pair| pair.donor).collect();
                    match abadie_imbens_standard_error(&weights, &donors, &query, &donor) {
                        Some(error) => (Some(error), query, donor),
                        None => (None, Vec::new(), Vec::new()),
                    }
                }
                _ => (None, Vec::new(), Vec::new()),
            };
        let matched_queries: Vec<usize> = pairs.iter().map(|pair| pair.query).collect();
        let matched_donors: Vec<usize> = pairs.iter().map(|pair| pair.donor).collect();
        let unmatched_treated: Vec<usize> = units
            .iter()
            .enumerate()
            .filter(|(index, unit)| unit.treated && matched_queries.binary_search(index).is_err())
            .map(|(index, _)| index)
            .collect();
        let matched_periods: Vec<i64> = matched_queries.iter().map(|q| units[*q].period).collect();
        let differences: Vec<f64> = pairs.iter().map(|pair| pair.difference).collect();
        let pair_weights: Vec<f64> = pairs.iter().map(|pair| pair.weight).collect();
        let serial = serial_history_interval(&matched_periods, &differences, &pair_weights).ok();
        Ok(PeriodHistoryEstimate {
            contrast,
            matched_periods,
            differences,
            pair_weights,
            serial,
            standard_error,
            query_variances,
            donor_variances,
            weight_total,
            matched_queries,
            matched_donors,
            unmatched_treated,
            periods_matched,
        })
    }
}

/// Period-matched history contrast with its analytic standard error.
#[must_use]
#[derive(Clone, Debug)]
pub struct PeriodHistoryEstimate {
    /// Sampling-weighted mean of treated-minus-matched-control outcomes.
    pub contrast: f64,
    /// Abadie-Imbens standard error for one match with replacement and a known scalar score,
    /// conditional on the scores and the treatment, with within-arm nearest-neighbour outcome
    /// variances. `None` when a matched treated unit or a used donor has no other unit of its arm
    /// in its period, or a matched weight is not positive. Its repeated-sampling calibration is
    /// unmeasured.
    pub standard_error: Option<f64>,
    /// Within-arm outcome variance estimate of each matched treated unit, in matched order;
    /// empty when some matched unit or donor has no same-arm neighbour.
    pub query_variances: Vec<f64>,
    /// Within-arm outcome variance estimate of the control each matched query used, in matched
    /// order; empty exactly when [`Self::query_variances`] is.
    pub donor_variances: Vec<f64>,
    /// Sum of sampling weights on matched treated units.
    pub weight_total: f64,
    /// Input indices of matched treated units, in input order.
    pub matched_queries: Vec<usize>,
    /// Input index of the control each matched query used.
    pub matched_donors: Vec<usize>,
    /// Treated units with no control inside their period and the caliper, in input order.
    pub unmatched_treated: Vec<usize>,
    /// Periods in which at least one treated unit matched.
    pub periods_matched: usize,
    /// Calendar period of each matched treated unit, in matched order.
    pub matched_periods: Vec<i64>,
    /// Outcome difference of each matched pair, in matched order.
    pub differences: Vec<f64>,
    /// Sampling weight of each matched treated unit, in matched order.
    pub pair_weights: Vec<f64>,
    /// Serial-dependence-aware standard error and interval over the matched periods. `None`
    /// when fewer than two periods hold a match (see [`serial_history_interval`]).
    pub serial: Option<SerialHistoryInterval>,
}

/// Newey-West standard error and normal interval of a period-matched contrast.
///
/// Matched pairs are aggregated by calendar period, the weighted-mean linearization residual of
/// each period is `sum_q w_q (d_q - contrast)`, and the long-run variance of that period series
/// is the Bartlett-kernel Newey-West estimator at the truncation lag
/// `floor(4 (T / 100)^(2/9))` (periods enter by their labels, so a missing period invents no
/// adjacency), scaled by `T / (T - 1) / (sum of weights)^2`. This is the estimator the arm
/// designs use for serially dependent unit-period data. The interval is normal at the 97.5%
/// point and is conditional on the periods that hold a match; its calibration is unmeasured.
#[derive(Clone, Debug, PartialEq)]
pub struct SerialHistoryInterval {
    /// Newey-West standard error.
    pub standard_error: f64,
    /// Lower endpoint of the 95% normal interval.
    pub lower: f64,
    /// Upper endpoint of the 95% normal interval.
    pub upper: f64,
    /// Bartlett truncation lag used.
    pub bandwidth: usize,
    /// Matched periods in ascending order.
    pub periods: Vec<i64>,
}

/// The serial-dependence-aware interval of a weighted mean of matched differences.
///
/// `periods`, `differences` and `weights` are aligned per matched pair.
///
/// # Errors
/// Misaligned inputs, a non-finite value, a non-positive weight, or fewer than two distinct
/// periods (serial dependence cannot be estimated from one period).
pub fn serial_history_interval(
    periods: &[i64],
    differences: &[f64],
    weights: &[f64],
) -> Result<SerialHistoryInterval, EstimationError> {
    if periods.len() != differences.len() || periods.len() != weights.len() {
        return Err(EstimationError::data_msg("matched periods, differences and weights differ in length"));
    }
    let mut labels: Vec<i64> = periods.to_vec();
    labels.sort_unstable();
    labels.dedup();
    if labels.len() < 2 {
        return Err(EstimationError::unsupported_coded(
            antecedent_core::reason_code!("route_not_supported"),
            "a serial-dependence-aware interval needs matches in at least two calendar periods",
        ));
    }
    let cluster: Vec<usize> = periods
        .iter()
        .map(|period| labels.binary_search(period).unwrap_or_default())
        .collect();
    let scores = ClusterScores::accumulate_weighted(differences, 1, &cluster, Some(weights))?;
    let summary = serial_sampling_summary(&scores, &labels)?;
    Ok(SerialHistoryInterval {
        standard_error: summary.standard_error[0],
        lower: summary.lower[0],
        upper: summary.upper[0],
        bandwidth: newey_west_bandwidth(labels.len()),
        periods: labels,
    })
}

/// Abadie-Imbens (2006) standard error of a weighted matched mean with one match, conditional on
/// the scores and the treatment.
///
/// With shares `w_q = weight_q / total` and `c_j` the share of the queries that control `j`
/// donated to, the variance of the contrast given the matches is
/// `sum_q w_q^2 s_q^2 + sum_j c_j^2 s_j^2`, where `s_i^2` is the conditional outcome variance of
/// unit `i`. Each is estimated by the Abadie-Imbens within-arm nearest-neighbour form
/// `(1/2) (y_i - y_l(i))^2`, `l(i)` the nearest other unit of the same arm and period on the
/// score ([`within_arm_noise`]); the pair differences are not used, because every query that
/// shares a donor shares its noise and the differences understate the spread. `query_noise` and
/// `donor_noise` hold those variances for the query and the donor of each pair. `None` when a
/// matched weight is not positive.
#[must_use]
pub fn abadie_imbens_standard_error(
    weights: &[f64],
    donors: &[usize],
    query_noise: &[f64],
    donor_noise: &[f64],
) -> Option<f64> {
    let pairs = weights.len();
    if pairs == 0
        || donors.len() != pairs
        || query_noise.len() != pairs
        || donor_noise.len() != pairs
        || weights.iter().any(|weight| weight.partial_cmp(&0.0_f64) != Some(Ordering::Greater))
    {
        return None;
    }
    let total: f64 = weights.iter().sum();
    let mut variance = 0.0_f64;
    let mut donated: BTreeMap<usize, (f64, f64)> = BTreeMap::new();
    for k in 0..pairs {
        let share = weights[k] / total;
        variance += share * share * query_noise[k];
        donated.entry(donors[k]).or_insert((0.0, donor_noise[k])).0 += share;
    }
    for (share, noise) in donated.values() {
        variance += share * share * noise;
    }
    (variance.is_finite() && variance >= 0.0).then_some(variance.sqrt())
}

/// Within-arm conditional outcome variance of each row, `(1/2) (y_i - y_l(i))^2`, where `l(i)` is
/// the nearest other row on the score (equal scores are at distance zero; the lowest input index
/// breaks a tie in distance). A row alone in its arm gets no entry.
fn within_arm_noise(rows: &[Row], out: &mut BTreeMap<usize, f64>) {
    if rows.len() < 2 {
        return;
    }
    let mut sorted: Vec<&Row> = rows.iter().collect();
    sorted.sort_by(|a, b| a.score.total_cmp(&b.score).then(a.index.cmp(&b.index)));
    // Runs of equal score, each in ascending input index.
    let mut runs: Vec<&[&Row]> = Vec::new();
    let mut start = 0;
    for k in 1..=sorted.len() {
        if k == sorted.len() || sorted[k].score.total_cmp(&sorted[start].score).is_ne() {
            runs.push(&sorted[start..k]);
            start = k;
        }
    }
    for (g, run) in runs.iter().enumerate() {
        for row in *run {
            let neighbour = if run.len() > 1 {
                run.iter().find(|other| other.index != row.index).copied()
            } else {
                let left = g.checked_sub(1).map(|h| runs[h][0]);
                let right = runs.get(g + 1).map(|r| r[0]);
                match (left, right) {
                    (Some(l), Some(r)) => {
                        let (dl, dr) = ((row.score - l.score).abs(), (r.score - row.score).abs());
                        match dl.total_cmp(&dr) {
                            Ordering::Less => Some(l),
                            Ordering::Greater => Some(r),
                            Ordering::Equal => Some(if l.index < r.index { l } else { r }),
                        }
                    }
                    (Some(l), None) => Some(l),
                    (None, other) => other,
                }
            };
            if let Some(other) = neighbour {
                out.insert(row.index, 0.5 * (row.outcome - other.outcome).powi(2));
            }
        }
    }
}

struct Row {
    index: usize,
    outcome: f64,
    weight: f64,
    score: f64,
}

struct MatchedHistories {
    contrast: f64,
    weight_total: f64,
    collapsed_donors: usize,
    queries: Vec<usize>,
    donors: Vec<usize>,
}

/// A caliper is a maximum distance: finite and non-negative.
fn validate_caliper(caliper: Option<f64>) -> Result<(), EstimationError> {
    match caliper {
        Some(limit) if !(limit.is_finite() && limit >= 0.0) => {
            Err(EstimationError::data_msg("matching caliper must be finite and non-negative"))
        }
        _ => Ok(()),
    }
}

fn match_histories(
    caliper: Option<f64>,
    units: &[HistoryUnit],
) -> Result<MatchedHistories, EstimationError> {
    validate_caliper(caliper)?;
    if units.is_empty() {
        return Err(EstimationError::data_msg(
            "history match requires treated and control histories",
        ));
    }
    let length = units[0].history.len();
    if units.iter().any(|unit| unit.history.len() != length) {
        return Err(EstimationError::data_msg(
            "history match requires every history to have the same length",
        ));
    }
    let mut treated = Vec::new();
    let mut control = Vec::new();
    for (index, unit) in units.iter().enumerate() {
        let row = Row { index, outcome: unit.outcome, weight: unit.weight, score: unit.score() };
        if unit.treated {
            treated.push(row);
        } else {
            control.push(row);
        }
    }
    if treated.is_empty() || control.is_empty() {
        return Err(EstimationError::data_msg(
            "history match requires treated and control histories",
        ));
    }
    let (pairs, collapsed_donors) = match_rows(caliper, &treated, &control);
    let mut numerator = 0.0_f64;
    let mut weight_total = 0.0_f64;
    let mut queries = Vec::new();
    let mut donors = Vec::new();
    for pair in &pairs {
        numerator += pair.weight * pair.difference;
        weight_total += pair.weight;
        queries.push(pair.query);
        donors.push(pair.donor);
    }
    if queries.is_empty() {
        return Err(EstimationError::data_msg("no matched units within caliper"));
    }
    if weight_total.partial_cmp(&0.0_f64) != Some(Ordering::Greater) {
        return Err(EstimationError::data_msg(
            "history-matched contrast weights left no positive mass on matched units",
        ));
    }
    Ok(MatchedHistories {
        contrast: numerator / weight_total,
        weight_total,
        collapsed_donors,
        queries,
        donors,
    })
}

/// One treated unit and the control it matched, by input index.
struct Pair {
    query: usize,
    donor: usize,
    weight: f64,
    difference: f64,
}

/// Match every treated row to its nearest control row, with replacement. Returns the pairs in
/// treated order and the number of control scores left after equal scores collapsed.
fn match_rows(caliper: Option<f64>, treated: &[Row], control: &[Row]) -> (Vec<Pair>, usize) {
    let scores: Vec<f64> = control.iter().map(|row| row.score).collect();
    let index = CollapsedHistoryIndex::from_donors(&scores);
    let mut pairs = Vec::new();
    for query in treated {
        let Some((donor_slot, _)) = index.nearest(query.score, caliper) else {
            continue;
        };
        let Some(donor) = control.get(donor_slot) else {
            continue;
        };
        pairs.push(Pair {
            query: query.index,
            donor: donor.index,
            weight: query.weight,
            difference: query.outcome - donor.outcome,
        });
    }
    (pairs, index.by_score.len())
}

/// Donor index after equal history scores collapse.
///
/// Donors stay in the index after a query matches them, so a later query can use the same donor.
/// Equal absolute differences resolve to the lowest donor index.
struct CollapsedHistoryIndex {
    by_score: Vec<(f64, usize)>,
}

impl CollapsedHistoryIndex {
    #[must_use]
    fn from_donors(donors: &[f64]) -> Self {
        let mut by_score: Vec<(f64, usize)> =
            donors.iter().copied().enumerate().map(|(index, score)| (score, index)).collect();
        by_score.sort_by(|left, right| left.0.total_cmp(&right.0).then(left.1.cmp(&right.1)));
        // The first entry of an equal-score run is its lowest donor index, which is the only
        // one a query can resolve to; the rest never change a result.
        by_score.dedup_by(|later, kept| later.0.total_cmp(&kept.0).is_eq());
        Self { by_score }
    }

    /// Nearest donor, or `None` when that distance is non-finite or past `caliper`.
    ///
    /// A distance equal to the caliper is a match. `caliper == None` accepts every finite
    /// nearest donor.
    #[must_use]
    fn nearest(&self, query: f64, caliper: Option<f64>) -> Option<(usize, f64)> {
        let split = self.by_score.partition_point(|&(score, _)| score < query);
        let left = split.checked_sub(1).map(|slot| (query - self.by_score[slot].0).abs());
        let right = self.by_score.get(split).map(|&(score, _)| (query - score).abs());
        let best = match (left, right) {
            (Some(left_distance), Some(right_distance)) => left_distance.min(right_distance),
            (Some(distance), None) | (None, Some(distance)) => distance,
            (None, None) => return None,
        };
        if !best.is_finite() {
            return None;
        }
        // `|query - score|` is non-decreasing as `score` moves away from `query`, so every
        // donor tied with `best` is contiguous around the two neighbours.
        let mut position = usize::MAX;
        let mut slot = split;
        while slot > 0 {
            slot -= 1;
            let (score, donor) = self.by_score[slot];
            if (query - score).abs().total_cmp(&best).is_ne() {
                break;
            }
            position = position.min(donor);
        }
        for &(score, donor) in &self.by_score[split..] {
            if (query - score).abs().total_cmp(&best).is_ne() {
                break;
            }
            position = position.min(donor);
        }
        if position == usize::MAX {
            return None;
        }
        if let Some(limit) = caliper {
            // Negated so that a NaN limit refuses instead of accepting every distance.
            if !matches!(best.partial_cmp(&limit), Some(Ordering::Less | Ordering::Equal)) {
                return None;
            }
        }
        Some((position, best))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BayesianHistoryContrast, ContrastPrior, FrequentistHistoryContrast, HistoryUnit,
        PeriodHistoryContrast,
    };

    fn unit(history: &[f64], outcome: f64, treated: bool, weight: f64) -> HistoryUnit {
        HistoryUnit::try_new(history, outcome, treated, weight).unwrap()
    }

    /// Controls at scores 1, 1, and 4, then treated units at 1, 4, and 10.
    ///
    /// The two controls at score 1 have slopes of opposite sign on time. Control weights are
    /// large so a total that included them would not equal the treated mass.
    fn hand_sample() -> Vec<HistoryUnit> {
        vec![
            unit(&[0.0, 2.0], 10.0, false, 100.0),
            unit(&[2.0, 0.0], 100.0, false, 100.0),
            unit(&[4.0, 4.0], 1.0, false, 100.0),
            unit(&[1.0, 1.0], 14.0, true, 3.0),
            unit(&[4.0, 4.0], 5.0, true, 1.0),
            unit(&[10.0, 10.0], 0.0, true, 9.0),
        ]
    }

    #[test]
    fn weighted_contrast_matches_the_hand_calculation() {
        let estimate = FrequentistHistoryContrast::try_with_caliper(0.5)
            .unwrap()
            .contrast(&hand_sample())
            .unwrap();
        // Query scores 1 and 4 match controls 0 and 2. Effects are both 4. Weights 3 and 1.
        // The treated unit at score 10 is outside the caliper. (3*4 + 1*4) / 4 = 4.
        assert_eq!(estimate.contrast.to_bits(), 4.0_f64.to_bits());
        assert_eq!(estimate.weight_total.to_bits(), 4.0_f64.to_bits());
        assert_eq!(estimate.matched_queries, vec![3, 4]);
        assert_eq!(estimate.matched_donors, vec![0, 2]);
        assert_eq!(estimate.collapsed_donors, 2);
    }

    #[test]
    fn tied_history_scores_collapse_at_build() {
        // The tied score 0 is never queried. It still collapses, so three controls become two
        // index entries before any query is matched.
        let units = vec![
            unit(&[0.0, 0.0], 1.0, false, 1.0),
            unit(&[0.0, 0.0], 9.0, false, 1.0),
            unit(&[5.0, 5.0], 4.0, false, 1.0),
            unit(&[5.0, 5.0], 7.0, true, 1.0),
        ];
        let estimate = FrequentistHistoryContrast::new().contrast(&units).unwrap();
        assert_eq!(estimate.collapsed_donors, 2);
        assert_eq!(estimate.matched_donors, vec![2]);
        assert_eq!(estimate.contrast.to_bits(), 3.0_f64.to_bits());
    }

    #[test]
    fn opposite_history_slopes_stay_tied() {
        let sample = hand_sample();
        assert_eq!(sample[0].score().to_bits(), sample[1].score().to_bits());
        // A regression of history on time would give slopes +2 and -2. The kept donor is the
        // lower index, whose outcome is 10, not the later slope's outcome of 100.
        let estimate = FrequentistHistoryContrast::new().contrast(&sample).unwrap();
        assert_eq!(estimate.matched_donors[0], 0);
        assert_eq!(estimate.collapsed_donors, 2);
    }

    #[test]
    fn equidistant_histories_keep_the_lower_index() {
        // Query score 1 is equidistant from donor 0 (score 2) and donor 1 (score 0). Donor 0 is
        // the right-hand neighbour and still wins, because its index is lower.
        let units = vec![
            unit(&[2.0], 7.0, false, 1.0),
            unit(&[0.0], 9.0, false, 1.0),
            unit(&[1.0], 10.0, true, 1.0),
        ];
        let estimate = FrequentistHistoryContrast::new().contrast(&units).unwrap();
        assert_eq!(estimate.matched_donors, vec![0]);
        assert_eq!(estimate.contrast.to_bits(), 3.0_f64.to_bits());
        assert_eq!(estimate.collapsed_donors, 2);
    }

    #[test]
    fn caliper_accepts_an_equal_distance_and_drops_a_farther_history() {
        let units = vec![
            unit(&[0.0], 1.0, false, 1.0),
            unit(&[2.0], 5.0, false, 1.0),
            unit(&[1.0], 4.0, true, 1.0),
        ];
        let on_the_boundary =
            FrequentistHistoryContrast::try_with_caliper(1.0).unwrap().contrast(&units).unwrap();
        assert_eq!(on_the_boundary.matched_donors, vec![0]);
        assert_eq!(on_the_boundary.contrast.to_bits(), 3.0_f64.to_bits());
        let outside = FrequentistHistoryContrast::try_with_caliper(0.5).unwrap().contrast(&units);
        assert!(outside.unwrap_err().to_string().contains("no matched units within caliper"));

        let farther = FrequentistHistoryContrast::try_with_caliper(0.5)
            .unwrap()
            .contrast(&hand_sample())
            .unwrap();
        assert_eq!(farther.matched_queries, vec![3, 4]);
    }

    #[test]
    fn repeated_query_reuses_the_collapsed_donor() {
        let units = vec![
            unit(&[0.0, 2.0], 2.0, false, 1.0),
            unit(&[2.0, 0.0], 8.0, false, 1.0),
            unit(&[1.0, 1.0], 5.0, true, 1.0),
            unit(&[1.0, 1.0], 6.0, true, 1.0),
        ];
        let estimate = FrequentistHistoryContrast::new().contrast(&units).unwrap();
        assert_eq!(estimate.collapsed_donors, 1);
        assert_eq!(estimate.matched_donors, vec![0, 0]);
    }

    #[test]
    fn caliper_must_be_finite_and_non_negative() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.1] {
            let frequentist = FrequentistHistoryContrast::try_with_caliper(bad).unwrap_err();
            assert!(
                frequentist.to_string().contains("finite and non-negative"),
                "{bad}: {frequentist}"
            );
            let prior = ContrastPrior::try_new(0.0, 1.0).unwrap();
            let bayesian = BayesianHistoryContrast::try_new(prior, Some(bad)).unwrap_err();
            assert!(bayesian.to_string().contains("finite and non-negative"), "{bad}: {bayesian}");
        }
        assert!(FrequentistHistoryContrast::try_with_caliper(0.0).is_ok());
    }

    #[test]
    fn non_positive_weight_total_is_refused() {
        let zero = vec![unit(&[0.0], 1.0, false, 1.0), unit(&[0.0], 2.0, true, 0.0)];
        let zero_total = FrequentistHistoryContrast::new().contrast(&zero).unwrap_err();
        assert!(zero_total.to_string().contains("no positive mass"), "{zero_total}");

        let cancelling = vec![
            unit(&[0.0], 1.0, false, 5.0),
            unit(&[0.0], 2.0, true, 2.0),
            unit(&[0.0], 4.0, true, -2.0),
        ];
        let cancelled = FrequentistHistoryContrast::new().contrast(&cancelling).unwrap_err();
        assert!(cancelled.to_string().contains("no positive mass"), "{cancelled}");

        let negative = vec![
            unit(&[1.0], 1.0, false, 1.0),
            unit(&[1.0], 3.0, true, -1.0),
            unit(&[1.0], 3.0, true, -1.0),
        ];
        let negative_total = FrequentistHistoryContrast::new().contrast(&negative).unwrap_err();
        assert!(negative_total.to_string().contains("no positive mass"), "{negative_total}");

        let nan_weight = HistoryUnit::try_new(&[0.0], 1.0, true, f64::NAN).unwrap_err();
        assert!(nan_weight.to_string().contains("sampling weight must be finite"));
    }

    #[test]
    fn prior_does_not_enter_the_weight_total() {
        let prior = ContrastPrior::try_new(20.0, 100.0).unwrap();
        let bayesian = BayesianHistoryContrast::try_new(prior, Some(0.5)).unwrap();
        let frequentist = FrequentistHistoryContrast::try_with_caliper(0.5)
            .unwrap()
            .contrast(&hand_sample())
            .unwrap();
        let estimate = bayesian.contrast(&hand_sample()).unwrap();
        assert_eq!(estimate.contrast.to_bits(), frequentist.contrast.to_bits());
        assert_eq!(estimate.weight_total.to_bits(), frequentist.weight_total.to_bits());
        assert_eq!(estimate.weight_total.to_bits(), 4.0_f64.to_bits());
        assert_eq!(estimate.prior.variance().to_bits(), 100.0_f64.to_bits());
        assert_ne!(
            (estimate.weight_total + estimate.prior.variance()).to_bits(),
            estimate.weight_total.to_bits()
        );
        assert_eq!(estimate.matched_donors, frequentist.matched_donors);
        assert_eq!(estimate.collapsed_donors, frequentist.collapsed_donors);
    }

    #[test]
    fn bayesian_contrast_refuses_a_posterior() {
        let prior = ContrastPrior::try_new(0.0, 1.0).unwrap();
        let bayesian = BayesianHistoryContrast::try_new(prior, Some(0.5)).unwrap();
        let refused = bayesian.refuse_posterior(&hand_sample());
        assert!(refused.to_string().contains("a prior is not a sampling weight"), "{refused}");
        assert!(!refused.to_string().contains("no positive mass"));

        let empty = bayesian.refuse_posterior(&[]);
        assert!(empty.to_string().contains("treated and control histories"), "{empty}");
        assert!(!empty.to_string().contains("a prior is not a sampling weight"));
    }

    #[test]
    fn refuses_histories_that_cannot_be_matched() {
        let estimator = FrequentistHistoryContrast::new();
        let empty = estimator.contrast(&[]).unwrap_err();
        assert!(empty.to_string().contains("treated and control"));

        let treated_only = vec![unit(&[1.0, 1.0], 1.0, true, 1.0)];
        let treated = estimator.contrast(&treated_only).unwrap_err();
        assert!(treated.to_string().contains("treated and control"), "{treated}");

        let mixed_length = vec![unit(&[1.0], 1.0, false, 1.0), unit(&[1.0, 1.0], 1.0, true, 1.0)];
        let lengths = estimator.contrast(&mixed_length).unwrap_err();
        assert!(lengths.to_string().contains("same length"), "{lengths}");

        let empty_history = HistoryUnit::try_new(&[], 1.0, true, 1.0).unwrap_err();
        assert!(empty_history.to_string().contains("non-empty finite history"));
        let non_finite = HistoryUnit::try_new(&[f64::NAN], 1.0, true, 1.0).unwrap_err();
        assert!(non_finite.to_string().contains("non-empty finite history"));

        let bad_prior = ContrastPrior::try_new(0.0, 0.0).unwrap_err();
        assert!(bad_prior.to_string().contains("variance must be finite and positive"));
        let flat = ContrastPrior::try_new(0.0, f64::INFINITY).unwrap_err();
        assert!(flat.to_string().contains("variance must be finite and positive"));
    }

    fn periodic(
        period: i64,
        history: &[f64],
        outcome: f64,
        treated: bool,
        weight: f64,
    ) -> HistoryUnit {
        unit(history, outcome, treated, weight).in_period(period)
    }

    /// Nine units over three periods. Period 1 has an equidistant control pair, period 2 has a
    /// treated unit far from its only control, and period 3 has a treated unit and no control.
    /// Unit 0 has the same score as the period-2 control, unit 4, which it must not meet.
    fn periodic_sample() -> Vec<HistoryUnit> {
        vec![
            periodic(1, &[1.0, 3.0], 10.0, true, 1.0),
            periodic(1, &[1.0, 1.0], 4.0, false, 1.0),
            periodic(1, &[3.0, 3.0], 6.0, false, 1.0),
            periodic(1, &[2.8, 3.0], 9.0, true, 2.0),
            periodic(2, &[2.0, 2.0], 5.0, false, 1.0),
            periodic(2, &[6.0, 6.0], 7.0, false, 1.0),
            periodic(2, &[5.0, 6.0], 12.0, true, 1.5),
            periodic(2, &[9.0, 9.0], 20.0, true, 1.0),
            periodic(3, &[1.0, 1.0], 3.0, true, 1.0),
        ]
    }

    fn close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1.0e-12, "actual {actual} expected {expected}");
    }

    #[test]
    fn period_match_stays_inside_its_period_and_matches_the_reference() {
        // Reference: brute-force nearest control inside each period (lowest index on a tie) and
        // the Abadie-Imbens conditional variance with within-arm nearest-neighbour outcome
        // variances, in exact rational arithmetic.
        let estimate = PeriodHistoryContrast::try_new(Some(2.0))
            .unwrap()
            .contrast(&periodic_sample())
            .unwrap();
        assert_eq!(estimate.matched_queries, vec![0, 3, 6]);
        assert_eq!(estimate.matched_donors, vec![1, 2, 5]);
        assert_eq!(estimate.unmatched_treated, vec![7, 8]);
        assert_eq!(estimate.periods_matched, 2);
        close(estimate.weight_total, 4.5);
        close(estimate.contrast, 4.333_333_333_333_333);
        close(estimate.standard_error.unwrap(), 2.096_440_251_568_134);
    }

    #[test]
    fn period_match_without_a_caliper_reuses_a_donor() {
        let estimate =
            PeriodHistoryContrast::try_new(None).unwrap().contrast(&periodic_sample()).unwrap();
        assert_eq!(estimate.matched_queries, vec![0, 3, 6, 7]);
        assert_eq!(estimate.matched_donors, vec![1, 2, 5, 5]);
        assert_eq!(estimate.unmatched_treated, vec![8]);
        close(estimate.weight_total, 5.5);
        close(estimate.contrast, 5.909_090_909_090_908);
        close(estimate.standard_error.unwrap(), 2.065_057_580_291_008_5);
    }

    #[test]
    fn one_period_reduces_to_the_cross_sectional_history_match() {
        let units = hand_sample();
        let cross = FrequentistHistoryContrast::new().contrast(&units).unwrap();
        let periodic = PeriodHistoryContrast::try_new(None).unwrap().contrast(&units).unwrap();
        close(periodic.contrast, cross.contrast);
        assert_eq!(periodic.matched_queries, cross.matched_queries);
        assert_eq!(periodic.matched_donors, cross.matched_donors);
    }

    #[test]
    fn a_control_in_another_period_is_not_a_donor() {
        let units = vec![periodic(1, &[1.0], 5.0, true, 1.0), periodic(2, &[1.0], 2.0, false, 1.0)];
        let error = PeriodHistoryContrast::try_new(None).unwrap().contrast(&units).unwrap_err();
        assert!(
            error.to_string().contains("no matched units within caliper and period"),
            "{error}"
        );
    }

    #[test]
    fn a_single_matched_unit_has_no_standard_error() {
        let units = vec![periodic(1, &[1.0], 5.0, true, 1.0), periodic(1, &[1.5], 2.0, false, 1.0)];
        let estimate = PeriodHistoryContrast::try_new(None).unwrap().contrast(&units).unwrap();
        close(estimate.contrast, 3.0);
        assert!(estimate.standard_error.is_none());
    }

    #[test]
    fn the_bayesian_refusal_names_the_premise_and_the_next_attempt() {
        let prior = ContrastPrior::try_new(0.0, 1.0).unwrap();
        let bayesian = BayesianHistoryContrast::try_new(prior, None).unwrap();
        let message = bayesian.refuse_posterior(&periodic_sample()).to_string();
        assert!(message.contains("no likelihood"), "{message}");
        assert!(message.contains("next attempt: declare a Bayesian outcome model"), "{message}");
    }
}
