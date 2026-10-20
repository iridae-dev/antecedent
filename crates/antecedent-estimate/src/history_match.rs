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
//! for this contrast is constructed. This file is not a temporal matching route, and a
//! time-series regression is not a substitute.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::cmp::Ordering;

use crate::EstimationError;

/// One unit: a history, the outcome contrasted after it, and a sampling weight.
#[derive(Clone, Debug)]
pub struct HistoryUnit {
    history: Vec<f64>,
    outcome: f64,
    treated: bool,
    weight: f64,
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
        let unit = Self { history: history.to_vec(), outcome, treated, weight };
        if !unit.score().is_finite() {
            return Err(EstimationError::data_msg("history match score must be finite"));
        }
        if !outcome.is_finite() {
            return Err(EstimationError::data_msg("history match outcome must be finite"));
        }
        if !weight.is_finite() {
            return Err(EstimationError::data_msg(
                "history match sampling weight must be finite",
            ));
        }
        Ok(unit)
    }

    /// Mean of the observed history.
    ///
    /// Every time point enters with equal weight. This is not a regression coefficient.
    #[must_use]
    pub fn score(&self) -> f64 {
        self.history.iter().sum::<f64>() / self.history.len() as f64
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
    /// returned as that failure.
    #[must_use]
    pub fn refuse_posterior(&self, units: &[HistoryUnit]) -> EstimationError {
        if let Err(error) = match_histories(self.caliper, units) {
            return error;
        }
        EstimationError::unsupported(
            "no posterior for the history-matched contrast; a prior is not a sampling weight",
        )
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
        Some(limit) if !(limit.is_finite() && limit >= 0.0) => Err(EstimationError::data_msg(
            "matching caliper must be finite and non-negative",
        )),
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
    let scores: Vec<f64> = control.iter().map(|row| row.score).collect();
    let index = CollapsedHistoryIndex::from_donors(&scores);
    let collapsed_donors = index.by_score.len();
    let mut numerator = 0.0_f64;
    let mut weight_total = 0.0_f64;
    let mut queries = Vec::new();
    let mut donors = Vec::new();
    for query in &treated {
        let Some((donor_slot, _)) = index.nearest(query.score, caliper) else {
            continue;
        };
        let donor = &control[donor_slot];
        numerator += query.weight * (query.outcome - donor.outcome);
        weight_total += query.weight;
        queries.push(query.index);
        donors.push(donor.index);
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
    use super::{BayesianHistoryContrast, ContrastPrior, FrequentistHistoryContrast, HistoryUnit};

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
            assert!(
                bayesian.to_string().contains("finite and non-negative"),
                "{bad}: {bayesian}"
            );
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

        let mixed_length = vec![
            unit(&[1.0], 1.0, false, 1.0),
            unit(&[1.0, 1.0], 1.0, true, 1.0),
        ];
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
}
