//! Declared sampling weights.
//!
//! A sampling weight `s_i` is the number of population units row `i` stands for (an inverse
//! inclusion probability, possibly after calibration). Declaring weights changes the
//! estimand and the uncertainty, so every estimator that reads them follows the same
//! contract.
//!
//! **Estimand.** The target is the *weighted population*: the population the declared weights
//! represent. An average treatment effect under weights `s` is the effect in that population,
//! `Σ s_i τ_i / Σ s_i` for the finite weighted population, and the corresponding
//! super-population functional when the rows are independent draws from a design with
//! inclusion probabilities `1/s_i`. A target-population declaration composes with the weights
//! by product: rows of a custom distribution or row-weight population carry
//! `s_i · target_i`, so the estimand is the target law reweighted *within the weighted
//! population*.
//!
//! **Normalization.** Estimates, standard errors and intervals depend on the weights only up
//! to scale. Weights are normalized to mean one over the rows an estimator analyses, which
//! keeps working weights at the scale of unweighted rows. The sum of the declared weights is
//! reported but never read as a population size: no finite-population correction is made.
//!
//! **Zero weights.** A row with weight zero stands for no population unit. It is kept in
//! the accounting (counted on the record) and contributes nothing: every estimate equals the
//! one computed on the table with those rows deleted.
//!
//! **Variance.** The standard error is the linearization (Taylor-series) estimator of the
//! with-replacement design, i.e. the sandwich over weighted estimating equations. It treats
//! rows, or declared clusters when a cluster column is supplied, as the independent primary
//! sampling units. A model-based variance that treats the weights as precision weights is not
//! offered: it would understate the variability of unequal weights.
//!
//! **Effective sample size.** The Kish effective sample size `(Σ s)² / Σ s²` of the analysed
//! rows is reported beside the row count.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::error::EstimationError;

/// What a declared set of sampling weights did to one analysis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingWeightRecord {
    /// Rows with positive weight that the estimate was computed on.
    pub rows_used: u64,
    /// Rows that passed every other selection but have weight zero. They are excluded from
    /// the computation and counted here.
    pub zero_weight_rows: u64,
    /// Kish effective sample size `(Σ s)² / Σ s²` of the rows used.
    pub effective_sample_size: f64,
    /// Sum of the declared weights over the rows used, before normalization. It is the
    /// population size only when the weights are inverse inclusion probabilities of a
    /// without-replacement design, and no estimator here reads it as one.
    pub weight_sum: f64,
}

impl SamplingWeightRecord {
    /// The population an estimate under sampling weights speaks for.
    pub const TARGET_POPULATION: &'static str =
        "the weighted population the declared sampling weights represent";

    /// Kish design effect `rows_used / effective_sample_size`: how much the unequal weights
    /// inflate the variance of a mean relative to equal weights.
    #[must_use]
    pub fn design_effect(&self) -> f64 {
        self.rows_used as f64 / self.effective_sample_size
    }
}

/// Sampling weights declared for every row of a source table.
///
/// Construction validates the declaration; [`Self::restrict`] aligns it to the rows an
/// estimator analyses after its complete-case and population selections.
#[derive(Clone, Debug, PartialEq)]
pub struct SamplingWeights {
    source: Arc<[f64]>,
}

/// The sampling weights of the rows one estimator analyses.
#[derive(Clone, Debug, PartialEq)]
pub struct RetainedSamplingWeights {
    /// Weights of the retained rows (all positive), normalized to mean one.
    pub weights: Arc<[f64]>,
    /// What the weights did to this analysis.
    pub record: SamplingWeightRecord,
}

impl SamplingWeights {
    /// Declare weights for every source row.
    ///
    /// # Errors
    ///
    /// `sampling_weights_invalid` (detail `sampling_weights.invalid`) naming the first
    /// negative or non-finite weight, or when no weight is positive.
    pub fn new(weights: impl Into<Arc<[f64]>>) -> Result<Self, EstimationError> {
        let source: Arc<[f64]> = weights.into();
        if let Some((row, value)) =
            source.iter().copied().enumerate().find(|(_, w)| !w.is_finite() || *w < 0.0)
        {
            return Err(invalid(&format!(
                "sampling weight {value} at source row {row} is not a finite non-negative \
                 number; weights count population units and cannot be negative, NaN or \
                 infinite"
            )));
        }
        if !source.iter().any(|w| *w > 0.0) {
            return Err(invalid(
                "every sampling weight is zero; the weights represent no population",
            ));
        }
        Ok(Self { source })
    }

    /// The declared weight of every source row.
    #[must_use]
    pub fn source(&self) -> &[f64] {
        &self.source
    }

    /// Check the declaration covers exactly `source_rows` rows.
    ///
    /// # Errors
    ///
    /// `sampling_weights_invalid` when the counts differ.
    pub fn require_rows(&self, source_rows: usize) -> Result<(), EstimationError> {
        if self.source.len() != source_rows {
            return Err(invalid(&format!(
                "{} sampling weights were declared for a table of {source_rows} rows",
                self.source.len()
            )));
        }
        Ok(())
    }

    /// Remove the rows of weight zero from `row_mask` (indexed by source row) and return how
    /// many were removed.
    ///
    /// # Errors
    ///
    /// `sampling_weights_invalid` when the mask is not the declared length, or when no
    /// retained row has a positive weight.
    pub fn drop_zero_weight_rows(&self, row_mask: &mut [bool]) -> Result<u64, EstimationError> {
        self.require_rows(row_mask.len())?;
        let mut zero_weight_rows = 0u64;
        let mut positive = false;
        for (keep, &weight) in row_mask.iter_mut().zip(self.source.iter()) {
            if !*keep {
                continue;
            }
            if weight > 0.0 {
                positive = true;
            } else {
                zero_weight_rows += 1;
                *keep = false;
            }
        }
        if !positive {
            return Err(invalid(
                "no analysed row has a positive sampling weight; the weighted population is empty",
            ));
        }
        Ok(zero_weight_rows)
    }

    /// The weights of the rows `row_mask` retains, normalized to mean one, in source-row
    /// order. `zero_weight_rows` is the count [`Self::drop_zero_weight_rows`] returned; the
    /// mask must already exclude those rows.
    ///
    /// # Errors
    ///
    /// `sampling_weights_invalid` when the mask is not the declared length or retains a
    /// zero-weight row or no row.
    pub fn retained(
        &self,
        row_mask: &[bool],
        zero_weight_rows: u64,
    ) -> Result<RetainedSamplingWeights, EstimationError> {
        self.require_rows(row_mask.len())?;
        let kept: Vec<f64> = row_mask
            .iter()
            .zip(self.source.iter())
            .filter_map(|(&keep, &weight)| keep.then_some(weight))
            .collect();
        if kept.is_empty() || kept.iter().any(|w| *w <= 0.0) {
            return Err(invalid(
                "the analysed rows must all have a positive sampling weight and there must be \
                 at least one",
            ));
        }
        Ok(RetainedSamplingWeights::from_positive(&kept, zero_weight_rows))
    }

    /// Restrict to the rows `row_mask` retains: [`Self::drop_zero_weight_rows`] then
    /// [`Self::retained`].
    ///
    /// # Errors
    ///
    /// As those two.
    pub fn restrict(
        &self,
        row_mask: &mut [bool],
    ) -> Result<RetainedSamplingWeights, EstimationError> {
        let zero_weight_rows = self.drop_zero_weight_rows(row_mask)?;
        self.retained(row_mask, zero_weight_rows)
    }

    /// Fingerprint of the declaration, for estimator identities that carry it.
    #[must_use]
    pub fn digest(&self) -> u64 {
        let mut hash = antecedent_core::Fnv1a64::new();
        hash.word(self.source.len() as u64);
        for &w in self.source.iter() {
            hash.word(w.to_bits());
        }
        hash.finish()
    }
}

impl RetainedSamplingWeights {
    /// Normalize positive `weights` to mean one and record them.
    pub(crate) fn from_positive(weights: &[f64], zero_weight_rows: u64) -> Self {
        let n = weights.len() as f64;
        let weight_sum = weights.iter().sum::<f64>();
        let scale = n / weight_sum;
        let normalized: Arc<[f64]> = if weight_sum.is_finite() && scale.is_finite() {
            weights.iter().map(|w| w * scale).collect()
        } else {
            // The weights are individually finite but their sum overflows (a scale of 1e308)
            // or underflows into a reciprocal that overflows (a scale of 1e-320). Normalize
            // relative to the largest weight, which only changes the order of the rounding.
            let largest = weights.iter().copied().fold(0.0_f64, f64::max);
            let unit_sum: f64 = weights.iter().map(|w| w / largest).sum();
            weights.iter().map(|w| (w / largest) * (n / unit_sum)).collect()
        };
        Self {
            weights: normalized,
            record: SamplingWeightRecord {
                rows_used: weights.len() as u64,
                zero_weight_rows,
                effective_sample_size: antecedent_kernels::kish_ess(weights),
                weight_sum,
            },
        }
    }

    /// Gather the weights of `rows` (indices into the retained rows) and renormalize them to
    /// mean one, recording the subset. Used where an estimator restricts further (trimming).
    #[must_use]
    pub fn subset(&self, rows: &[usize]) -> Self {
        let gathered: Vec<f64> = rows.iter().map(|&i| self.weights[i]).collect();
        Self::from_positive(&gathered, self.record.zero_weight_rows)
    }
}

/// The weighted-least-squares transform of a design: every row of the matrix and the outcome
/// multiplied by the square root of its weight. Ordinary least squares and the sandwich
/// formulas on the result are the weighted-fit and sampling-weighted-score formulas of the
/// original rows.
pub(crate) fn root_weight_design(
    mut design: antecedent_stats::CompiledDesign,
    weights: &[f64],
) -> antecedent_stats::CompiledDesign {
    let n = design.nrows;
    let root: Vec<f64> = weights.iter().map(|w| w.sqrt()).collect();
    let mut matrix = design.matrix.to_vec();
    for column in matrix.chunks_exact_mut(n) {
        for (value, scale) in column.iter_mut().zip(&root) {
            *value *= scale;
        }
    }
    design.outcome = design.outcome.iter().zip(&root).map(|(y, scale)| y * scale).collect();
    design.matrix = matrix.into();
    design
}

/// The refusal for a declaration that is not a valid set of sampling weights
/// (`sampling_weights_invalid`, detail `sampling_weights.invalid`).
#[must_use]
pub fn invalid(message: &str) -> EstimationError {
    crate::util::refuse(
        antecedent_core::reason_code!("sampling_weights_invalid"),
        "sampling_weights.invalid",
        message,
    )
}

/// The refusal for a route that has no construction with sampling weights
/// (`option_not_applicable`, detail `sampling_weights.route_unsupported`).
#[must_use]
pub fn unsupported_route(message: &str) -> EstimationError {
    crate::util::refuse(
        antecedent_core::reason_code!("option_not_applicable"),
        "sampling_weights.route_unsupported",
        message,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_weights_are_refused_naming_the_first_offender() {
        let negative = SamplingWeights::new(vec![1.0, 2.0, -0.5, f64::NAN]).unwrap_err();
        let text = negative.to_string();
        assert!(text.starts_with("reason=sampling_weights_invalid:"), "{text}");
        assert!(text.contains("-0.5") && text.contains("source row 2"), "{text}");
        let nan = SamplingWeights::new(vec![1.0, f64::NAN]).unwrap_err().to_string();
        assert!(nan.contains("NaN") && nan.contains("source row 1"), "{nan}");
        let infinite = SamplingWeights::new(vec![f64::INFINITY]).unwrap_err().to_string();
        assert!(infinite.contains("source row 0"), "{infinite}");
        let zero = SamplingWeights::new(vec![0.0, 0.0]).unwrap_err().to_string();
        assert!(zero.contains("every sampling weight is zero"), "{zero}");
    }

    #[test]
    fn zero_weight_rows_leave_the_mask_and_are_counted() {
        let weights = SamplingWeights::new(vec![2.0, 0.0, 4.0, 6.0, 0.0]).unwrap();
        let mut mask = vec![true, true, true, false, true];
        let kept = weights.restrict(&mut mask).unwrap();
        assert_eq!(mask, [true, false, true, false, false]);
        assert_eq!(kept.record.rows_used, 2);
        assert_eq!(kept.record.zero_weight_rows, 2);
        assert!((kept.record.weight_sum - 6.0).abs() < 1e-12);
        // Mean one: (2, 4) -> (2/3, 4/3) * ... = 2 * (2, 4) / 6.
        assert!((kept.weights[0] - 2.0 / 3.0).abs() < 1e-15);
        assert!((kept.weights[1] - 4.0 / 3.0).abs() < 1e-15);
        assert!((kept.record.effective_sample_size - 36.0 / 20.0).abs() < 1e-12);
    }

    #[test]
    fn normalization_and_effective_sample_size_are_scale_free() {
        let base = vec![1.0, 3.0, 0.5, 8.0];
        let scaled: Vec<f64> = base.iter().map(|w| w * 1.0e6).collect();
        let (mut a, mut b) = (vec![true; 4], vec![true; 4]);
        let one = SamplingWeights::new(base).unwrap().restrict(&mut a).unwrap();
        let two = SamplingWeights::new(scaled).unwrap().restrict(&mut b).unwrap();
        for (x, y) in one.weights.iter().zip(two.weights.iter()) {
            assert!((x - y).abs() < 1e-12);
        }
        assert!((one.record.effective_sample_size - two.record.effective_sample_size).abs() < 1e-9);
        assert!((one.weights.iter().sum::<f64>() - 4.0).abs() < 1e-12);
    }

    #[test]
    fn weights_at_the_edge_of_the_float_range_normalize_to_mean_one() {
        for scale in [1.0e308, 5.0e-320, 1.0e-300] {
            let base = [1.0, 0.25, 0.5, 0.25];
            let declared: Vec<f64> = base.iter().map(|w| w * scale).collect();
            let mut mask = vec![true; 4];
            let kept = SamplingWeights::new(declared).unwrap().restrict(&mut mask).unwrap();
            for (got, want) in kept.weights.iter().zip([2.0, 0.5, 1.0, 0.5]) {
                assert!((got - want).abs() < 1e-3 * want, "scale {scale}: {got} vs {want}");
            }
            assert!((kept.record.effective_sample_size - 4.0 / 1.375).abs() < 1e-2);
        }
    }

    #[test]
    fn a_mask_that_keeps_only_zero_weight_rows_is_refused() {
        let weights = SamplingWeights::new(vec![0.0, 1.0]).unwrap();
        let mut mask = vec![true, false];
        let error = weights.restrict(&mut mask).unwrap_err().to_string();
        assert!(error.contains("no analysed row has a positive sampling weight"), "{error}");
        let mut short = vec![true];
        assert!(weights.restrict(&mut short).unwrap_err().to_string().contains("1 rows"));
    }
}
