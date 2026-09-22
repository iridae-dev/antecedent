//! Cross-fitted AIPW score tables for retargeting and joint inference.
//!
//! The table is a stable artifact payload: per-row scores `φ_i^a` for every
//! arm (and every exceedance threshold), fold ids, row index, nuisance
//! provenance, and the certified adjustment set. Cross-language round-trip
//! is through [`ScoreTableWire`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::VariableId;

use crate::error::EstimationError;
use crate::joint_if::{JointCovariance, joint_influence_covariance, kish_n_eff, weighted_mean};

/// Column key: one interventional arm, optionally at an exceedance threshold.
#[derive(Clone, Debug, PartialEq)]
pub struct ScoreColumn {
    /// Arm label (`0`/`1` for binary, or a cell mask for joint treatments).
    pub arm: u32,
    /// Exceedance threshold. `None` is the mean functional.
    pub threshold: Option<f64>,
}

/// Per-row cross-fitted AIPW scores.
#[derive(Clone, Debug, PartialEq)]
pub struct ScoreTable {
    /// Observed cell on each retained row; empty on legacy score artifacts.
    pub observed_arm: Arc<[u32]>,
    /// Out-of-fold raw propensity for each score column, column-major.
    pub propensities: Arc<[f64]>,
    /// Observed original outcome for per-threshold support.
    pub observed_outcome: Arc<[f64]>,
    /// Complete-case row count.
    pub n_rows: usize,
    /// Original data-frame row index of each complete-case row.
    pub row_index: Arc<[u32]>,
    /// Fold assignment (seeded, arm-stratified plan over distinct units; or a shared plan).
    pub fold_ids: Arc<[u32]>,
    /// Number of folds used to fit nuisances.
    pub n_folds: u32,
    /// Column-major scores: `scores[col * n_rows + row]`.
    pub scores: Arc<[f64]>,
    /// Column keys, in the same order as the score columns.
    pub columns: Arc<[ScoreColumn]>,
    /// Certified adjustment set the nuisances conditioned on.
    pub adjustment_set: Arc<[VariableId]>,
    /// Nuisance provenance tag (cross-fit rule + model family).
    pub nuisance_provenance: Arc<str>,
    /// Propensity clip the estimator applied to the held-out propensities before forming
    /// inverse-probability weights (`None`: unclipped). The scores already embody it, and
    /// the raw [`Self::propensities`] are what a retarget's overlap gate reads, so the gate
    /// measures the extreme-propensity share against this applied band rather than a
    /// library default the fit never used.
    pub propensity_clip: Option<f64>,
    /// Treatment variable.
    pub treatment: VariableId,
    /// Additional intervened coordinates (joint cells). Empty for binary ATE.
    pub intervened: Arc<[VariableId]>,
}

impl ScoreTable {
    /// Number of score columns.
    #[must_use]
    pub fn n_columns(&self) -> usize {
        self.columns.len()
    }

    /// Number of distinct outcome thresholds across the columns.
    ///
    /// Thresholds are equal only when bit-identical under `total_cmp`: a threshold is a
    /// data value or a grid point, so an absolute tolerance would merge distinct
    /// thresholds of small-unit outcomes and equal exact ones at large magnitudes. The one
    /// definition every scalar-vs-grid decision uses.
    #[must_use]
    pub fn distinct_threshold_count(&self) -> usize {
        let mut thresholds: Vec<f64> = self.columns.iter().filter_map(|c| c.threshold).collect();
        thresholds.sort_by(f64::total_cmp);
        thresholds.dedup_by(|a, b| a.total_cmp(b).is_eq());
        thresholds.len()
    }

    /// Borrow score column `j`.
    ///
    /// # Errors
    ///
    /// Column index out of range.
    pub fn column(&self, j: usize) -> Result<&[f64], EstimationError> {
        if j >= self.n_columns() {
            return Err(EstimationError::data_msg("score column index out of range"));
        }
        let start = j
            .checked_mul(self.n_rows)
            .ok_or_else(|| EstimationError::data_msg("score table size overflow"))?;
        let end = start
            .checked_add(self.n_rows)
            .ok_or_else(|| EstimationError::data_msg("score table size overflow"))?;
        self.scores
            .get(start..end)
            .ok_or_else(|| EstimationError::data_msg("score table shape mismatch"))
    }

    /// Weighted means, contrasts, and joint IF covariance across columns.
    ///
    /// # Errors
    ///
    /// Weight length mismatch or empty mass.
    pub fn summarize(&self, weights: Option<&[f64]>) -> Result<ScoreSummary, EstimationError> {
        let mut means = Vec::with_capacity(self.n_columns());
        let mut cols: Vec<&[f64]> = Vec::with_capacity(self.n_columns());
        for j in 0..self.n_columns() {
            let col = self.column(j)?;
            cols.push(col);
            means.push(weighted_mean(col, weights)?);
        }
        let covariance =
            crate::joint_if::joint_influence_covariance_with_means(&cols, weights, &means)?;
        let n_eff = match weights {
            Some(w) => kish_n_eff(w),
            None => self.n_rows as f64,
        };
        Ok(ScoreSummary { means: Arc::from(means), covariance, n_eff })
    }

    /// Contrast `Σ c_j θ_j` with IF variance from the joint covariance.
    ///
    /// # Errors
    ///
    /// Contrast length mismatch.
    pub fn linear_contrast(
        &self,
        summary: &ScoreSummary,
        coefficients: &[f64],
    ) -> Result<LinearContrast, EstimationError> {
        if coefficients.len() != summary.means.len()
            || coefficients.len() != self.n_columns()
            || summary.covariance.dim != coefficients.len()
            || summary.covariance.values.len()
                != coefficients.len().saturating_mul(coefficients.len())
            || coefficients.iter().any(|v| !v.is_finite())
            || summary.means.iter().any(|v| !v.is_finite())
            || summary.covariance.values.iter().any(|v| !v.is_finite())
        {
            return Err(EstimationError::data_msg("contrast length must match score columns"));
        }
        let mut value = 0.0;
        for (c, m) in coefficients.iter().zip(summary.means.iter()) {
            value += c * m;
        }
        let dim = summary.covariance.dim;
        let mut var = 0.0;
        let mut absolute_terms = 0.0;
        for j in 0..dim {
            for i in 0..dim {
                let term = coefficients[i] * coefficients[j] * summary.covariance.get(i, j);
                var += term;
                absolute_terms += term.abs();
            }
        }
        if !value.is_finite() || !var.is_finite() || !absolute_terms.is_finite() {
            return Err(EstimationError::data_msg("contrast value or variance is non-finite"));
        }
        if var < -1e-12 * absolute_terms {
            return Err(EstimationError::data_msg("contrast variance is negative"));
        }
        Ok(LinearContrast { value, se: var.max(0.0).sqrt() })
    }

    /// Per-row scores of `Σ c_j φ_j` (same coefficients as [`Self::linear_contrast`]).
    ///
    /// # Errors
    ///
    /// Contrast length mismatch or a non-finite coefficient.
    pub fn combine_scores(&self, coefficients: &[f64]) -> Result<Vec<f64>, EstimationError> {
        if coefficients.len() != self.n_columns() || coefficients.iter().any(|v| !v.is_finite()) {
            return Err(EstimationError::data_msg("contrast length must match score columns"));
        }
        let mut out = vec![0.0; self.n_rows];
        for (j, &c) in coefficients.iter().enumerate() {
            if c == 0.0 {
                continue;
            }
            let col = self.column(j)?;
            for (dst, &src) in out.iter_mut().zip(col) {
                *dst += c * src;
            }
        }
        Ok(out)
    }
}

/// Weighted column means and their joint covariance.
#[derive(Clone, Debug, PartialEq)]
pub struct ScoreSummary {
    /// `θ_j` for each score column.
    pub means: Arc<[f64]>,
    /// Joint IF covariance of the means.
    pub covariance: JointCovariance,
    /// Kish effective sample size of the weights (or `n` if unweighted).
    pub n_eff: f64,
}

/// Simultaneous inference and empirical support in score-column order.
#[derive(Clone, Debug, PartialEq)]
pub struct ScoreInference {
    /// Raw AIPW means (before bounded isotonic projection).
    pub raw_means: Vec<f64>,
    /// Lower simultaneous endpoints for raw means.
    pub lower: Vec<f64>,
    /// Upper simultaneous endpoints for raw means.
    pub upper: Vec<f64>,
    /// Simultaneous confidence level.
    pub level: f64,
    /// Gaussian max-t critical value.
    pub critical_value: f64,
    /// Weighted effective sample size of observed rows in each arm/threshold event.
    pub event_n_eff: Vec<f64>,
    /// Whether at least ten effective events and non-events exist at each threshold.
    pub threshold_supported: Vec<bool>,
    /// Target-local joint-cell overlap.
    pub support: crate::crossfit_aipw::WeightedSupport,
}

/// Minimum Kish/count events and non-events required to support a tail probability.
pub const MIN_THRESHOLD_EVENTS: f64 = 10.0;

fn kish_threshold_support(
    arms: &[u32],
    outcome: &[f64],
    weights: &[f64],
    arm: u32,
    threshold: Option<f64>,
) -> (f64, f64) {
    let scale = weights.iter().copied().fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return (0.0, 0.0);
    }
    let mut e_sum = 0.0;
    let mut e_sq = 0.0;
    let mut n_sum = 0.0;
    let mut n_sq = 0.0;
    for ((&a, &y), &weight) in arms.iter().zip(outcome).zip(weights) {
        if a != arm || !(weight > 0.0 && weight.is_finite()) {
            continue;
        }
        let weight = weight / scale;
        if threshold.is_none_or(|c| y > c) {
            e_sum += weight;
            e_sq += weight * weight;
        } else {
            n_sum += weight;
            n_sq += weight * weight;
        }
    }
    (
        if e_sq <= 0.0 { 0.0 } else { (e_sum * e_sum) / e_sq },
        if n_sq <= 0.0 { 0.0 } else { (n_sum * n_sum) / n_sq },
    )
}

/// Simultaneous bands from aligned influence columns (threshold-major, two arms).
///
/// Unsupported tails receive non-finite band endpoints rather than a silent
/// empty-cell or first-threshold standard error. Covariance is the raw-score
/// joint IF; rearranged CDF values must not be mixed with these intervals.
///
/// # Errors
///
/// Shape mismatch, empty family, or invalid covariance.
pub fn inference_from_influence_columns(
    raw_means: &[f64],
    columns: &[&[f64]],
    event_n_eff: &[f64],
    threshold_supported: &[bool],
    support: crate::crossfit_aipw::WeightedSupport,
) -> Result<ScoreInference, EstimationError> {
    if raw_means.len() != columns.len()
        || event_n_eff.len() != columns.len()
        || threshold_supported.len() != columns.len()
        || columns.is_empty()
        || raw_means.iter().any(|v| !v.is_finite())
        || event_n_eff.iter().any(|v| !v.is_finite() || *v < 0.0)
    {
        return Err(EstimationError::data_msg(
            "per-arm CDF inference requires aligned finite means, IF columns, and tail counts",
        ));
    }
    let covariance = joint_influence_covariance(columns, None)?;
    let (critical_value, lower, upper) =
        simultaneous_bands(raw_means, &covariance, threshold_supported)?;
    Ok(ScoreInference {
        raw_means: raw_means.to_vec(),
        lower,
        upper,
        level: 0.95,
        critical_value,
        event_n_eff: event_n_eff.to_vec(),
        threshold_supported: threshold_supported.to_vec(),
        support,
    })
}

/// Max-t critical value and simultaneous endpoints over the supported columns.
///
/// The family is the supported columns with a positive plug-in SE: unsupported
/// columns publish no band, so they must not widen the others', and a degenerate
/// coordinate has a zero-width plug-in band. The support flags prevent reading an
/// empty observed tail as established zero risk. Unsupported columns get `NaN`
/// endpoints.
fn simultaneous_bands(
    means: &[f64],
    covariance: &JointCovariance,
    supported: &[bool],
) -> Result<(f64, Vec<f64>, Vec<f64>), EstimationError> {
    let active: Vec<_> =
        (0..means.len()).filter(|&j| supported[j] && covariance.se(j) > 0.0).collect();
    let critical_value = if active.is_empty() {
        0.0
    } else {
        let mut values = Vec::with_capacity(active.len() * active.len());
        for &j in &active {
            for &i in &active {
                values.push(covariance.get(i, j));
            }
        }
        crate::joint_if::max_t_critical(
            &JointCovariance { dim: active.len(), values: values.into() },
            0.95,
            4096,
            0x15,
        )?
    };
    let mut lower = Vec::with_capacity(means.len());
    let mut upper = Vec::with_capacity(means.len());
    for (j, &is_supported) in supported.iter().enumerate() {
        if is_supported {
            let radius = critical_value * covariance.se(j);
            lower.push(means[j] - radius);
            upper.push(means[j] + radius);
        } else {
            lower.push(f64::NAN);
            upper.push(f64::NAN);
        }
    }
    Ok((critical_value, lower, upper))
}

/// Empirical support of a score table under a target weighting.
#[derive(Clone, Debug, PartialEq)]
pub struct ScoreSupport {
    /// Weighted effective sample size of observed rows in each arm/threshold event.
    pub event_n_eff: Vec<f64>,
    /// Whether at least ten effective events and non-events exist at each threshold.
    pub threshold_supported: Vec<bool>,
    /// Target-local joint-cell overlap.
    pub overlap: crate::crossfit_aipw::WeightedSupport,
}

impl ScoreTable {
    /// Weighted column means only (no covariance, no simultaneous critical value).
    ///
    /// # Errors
    ///
    /// Weight length mismatch or empty mass.
    pub fn weighted_means(&self, weights: Option<&[f64]>) -> Result<Vec<f64>, EstimationError> {
        (0..self.n_columns()).map(|j| weighted_mean(self.column(j)?, weights)).collect()
    }

    /// Kish event counts, threshold support flags, and target overlap.
    #[must_use]
    pub fn support(&self, weights: Option<&[f64]>) -> ScoreSupport {
        let ones;
        let w: &[f64] = if let Some(weights) = weights {
            weights
        } else {
            ones = vec![1.0; self.n_rows];
            &ones
        };
        let overlap = crate::retarget::score_weighted_support(self, w);
        let mut event_n_eff = Vec::with_capacity(self.n_columns());
        let mut threshold_supported = Vec::with_capacity(self.n_columns());
        for col in self.columns.iter() {
            let (ne, n_non) = kish_threshold_support(
                &self.observed_arm,
                &self.observed_outcome,
                w,
                col.arm,
                col.threshold,
            );
            event_n_eff.push(ne);
            threshold_supported.push(
                ne >= MIN_THRESHOLD_EVENTS
                    && (col.threshold.is_none() || n_non >= MIN_THRESHOLD_EVENTS),
            );
        }
        ScoreSupport { event_n_eff, threshold_supported, overlap }
    }

    /// Simultaneous bands for the fixed declared family; does not cover data-driven selection.
    pub fn inference(&self, weights: Option<&[f64]>) -> Result<ScoreInference, EstimationError> {
        let summary = self.summarize(weights)?;
        let support = self.support(weights);
        let (critical_value, lower, upper) =
            simultaneous_bands(&summary.means, &summary.covariance, &support.threshold_supported)?;
        Ok(ScoreInference {
            raw_means: summary.means.to_vec(),
            lower,
            upper,
            level: 0.95,
            critical_value,
            event_n_eff: support.event_n_eff,
            threshold_supported: support.threshold_supported,
            support: support.overlap,
        })
    }
}

/// Scalar linear contrast of score-column means.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinearContrast {
    /// Point estimate.
    pub value: f64,
    /// Analytic IF standard error.
    pub se: f64,
}

/// Wire form for cross-language round-trip.
#[derive(Clone, Debug, PartialEq)]
pub struct ScoreTableWire {
    /// Observed cell on each retained row.
    pub observed_arm: Vec<u32>,
    /// Out-of-fold raw propensity for each score column.
    pub propensities: Vec<f64>,
    /// Observed original outcome.
    pub observed_outcome: Vec<f64>,
    /// Complete-case row count.
    pub n_rows: u64,
    /// Original row indexes.
    pub row_index: Vec<u32>,
    /// Fold ids.
    pub fold_ids: Vec<u32>,
    /// Fold count.
    pub n_folds: u32,
    /// Column-major scores.
    pub scores: Vec<f64>,
    /// Column keys.
    pub columns: Vec<ScoreColumn>,
    /// Adjustment variable raw ids.
    pub adjustment_set: Vec<u32>,
    /// Nuisance provenance.
    pub nuisance_provenance: String,
    /// Applied propensity clip (`None`: unclipped).
    pub propensity_clip: Option<f64>,
    /// Treatment raw id.
    pub treatment: u32,
    /// Extra intervened raw ids.
    pub intervened: Vec<u32>,
}

impl ScoreTable {
    /// Encode for artifacts / Python.
    #[must_use]
    pub fn to_wire(&self) -> ScoreTableWire {
        ScoreTableWire {
            observed_arm: self.observed_arm.to_vec(),
            propensities: self.propensities.to_vec(),
            observed_outcome: self.observed_outcome.to_vec(),
            n_rows: self.n_rows as u64,
            row_index: self.row_index.to_vec(),
            fold_ids: self.fold_ids.to_vec(),
            n_folds: self.n_folds,
            scores: self.scores.to_vec(),
            columns: self.columns.to_vec(),
            adjustment_set: self.adjustment_set.iter().map(|v| v.raw()).collect(),
            nuisance_provenance: self.nuisance_provenance.to_string(),
            propensity_clip: self.propensity_clip,
            treatment: self.treatment.raw(),
            intervened: self.intervened.iter().map(|v| v.raw()).collect(),
        }
    }

    /// Decode a wire payload.
    ///
    /// # Errors
    ///
    /// Shape mismatch.
    pub fn from_wire(wire: ScoreTableWire) -> Result<Self, EstimationError> {
        let n = usize::try_from(wire.n_rows).unwrap_or(usize::MAX);
        if wire.row_index.len() != n
            || wire.fold_ids.len() != n
            || wire.scores.len() != n.saturating_mul(wire.columns.len())
        {
            return Err(EstimationError::data_msg("score table wire shape mismatch"));
        }
        if (!wire.observed_arm.is_empty() && wire.observed_arm.len() != n)
            || (!wire.observed_outcome.is_empty() && wire.observed_outcome.len() != n)
            || (!wire.propensities.is_empty() && wire.propensities.len() != wire.scores.len())
            || wire.scores.iter().any(|v| !v.is_finite())
            || wire.propensities.iter().any(|v| !v.is_finite() || *v < 0.0 || *v > 1.0)
            || wire.observed_outcome.iter().any(|v| !v.is_finite())
            || wire.n_folds < 2
            || wire.fold_ids.iter().any(|&f| f >= wire.n_folds)
            || wire.columns.iter().any(|c| c.threshold.is_some_and(|v| !v.is_finite()))
            || wire.propensity_clip.is_some_and(|c| !(c > 0.0 && c < 0.5))
        {
            return Err(EstimationError::data_msg("invalid score table values or support shape"));
        }
        Ok(Self {
            observed_arm: wire.observed_arm.into(),
            propensities: wire.propensities.into(),
            observed_outcome: wire.observed_outcome.into(),
            n_rows: n,
            row_index: Arc::from(wire.row_index),
            fold_ids: Arc::from(wire.fold_ids),
            n_folds: wire.n_folds,
            scores: Arc::from(wire.scores),
            columns: Arc::from(wire.columns),
            adjustment_set: wire
                .adjustment_set
                .into_iter()
                .map(VariableId::from_raw)
                .collect::<Vec<_>>()
                .into(),
            nuisance_provenance: Arc::from(wire.nuisance_provenance),
            propensity_clip: wire.propensity_clip,
            treatment: VariableId::from_raw(wire.treatment),
            intervened: wire
                .intervened
                .into_iter()
                .map(VariableId::from_raw)
                .collect::<Vec<_>>()
                .into(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Thirty rows, one arm; column 1 is an exceedance at 0.5 with only five events.
    fn thin_tail_table() -> ScoreTable {
        let n = 30;
        let mut scores: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
        scores.extend((0..n).map(|i| f64::from(u32::try_from((i * 7) % 5).unwrap()) * 0.2));
        ScoreTable {
            observed_arm: vec![0; n].into(),
            propensities: vec![0.5; 2 * n].into(),
            observed_outcome: (0..n).map(|i| if i < 5 { 1.0 } else { 0.0 }).collect(),
            n_rows: n,
            row_index: (0..n).map(|i| u32::try_from(i).unwrap()).collect(),
            fold_ids: (0..n).map(|i| u32::try_from(i % 2).unwrap()).collect(),
            n_folds: 2,
            scores: scores.into(),
            columns: Arc::from([
                ScoreColumn { arm: 0, threshold: None },
                ScoreColumn { arm: 0, threshold: Some(0.5) },
            ]),
            adjustment_set: Arc::from([]),
            nuisance_provenance: Arc::from("test"),
            propensity_clip: None,
            treatment: VariableId::from_raw(0),
            intervened: Arc::from([]),
        }
    }

    #[test]
    fn means_and_support_match_the_full_summary_and_inference() {
        let table = thin_tail_table();
        let weights: Vec<f64> = (0..30).map(|i| 1.0 + f64::from(i % 3)).collect();
        for w in [None, Some(weights.as_slice())] {
            let summary = table.summarize(w).unwrap();
            let inference = table.inference(w).unwrap();
            let support = table.support(w);
            assert_eq!(table.weighted_means(w).unwrap(), summary.means.to_vec());
            assert_eq!(support.event_n_eff, inference.event_n_eff);
            assert_eq!(support.threshold_supported, inference.threshold_supported);
            assert_eq!(support.overlap, inference.support);
        }
        // Five events (< 10) leave the tail unsupported; the mean column stays supported.
        assert_eq!(table.support(None).threshold_supported, vec![true, false]);
    }

    #[test]
    fn max_t_family_is_the_supported_columns_only() {
        let table = thin_tail_table();
        let inference = table.inference(None).unwrap();
        let summary = table.summarize(None).unwrap();
        let only_supported = crate::joint_if::max_t_critical(
            &JointCovariance { dim: 1, values: Arc::from([summary.covariance.get(0, 0)]) },
            0.95,
            4096,
            0x15,
        )
        .unwrap();
        assert!(summary.covariance.se(1) > 0.0, "the unsupported column must not be degenerate");
        assert_eq!(inference.critical_value, only_supported);
        assert!(inference.lower[1].is_nan() && inference.upper[1].is_nan());
    }

    #[test]
    fn simultaneous_bands_ignore_unsupported_columns_and_blank_their_endpoints() {
        let covariance = JointCovariance { dim: 2, values: Arc::from([1.0, 0.0, 0.0, 100.0]) };
        let (both, ..) = simultaneous_bands(&[0.0, 0.0], &covariance, &[true, true]).unwrap();
        let (one, lower, upper) =
            simultaneous_bands(&[3.0, 0.0], &covariance, &[true, false]).unwrap();
        // Two independent standardized coordinates need a wider max-t than one.
        assert!(both > one + 0.1, "{both} vs {one}");
        assert!((lower[0] - (3.0 - one)).abs() < 1e-12 && (upper[0] - (3.0 + one)).abs() < 1e-12);
        assert!(lower[1].is_nan() && upper[1].is_nan());
    }

    #[test]
    fn review_invalid_contrast_variance_is_not_zero_uncertainty() {
        let table = ScoreTable {
            observed_arm: Arc::from([]),
            propensities: Arc::from([]),
            observed_outcome: Arc::from([]),
            n_rows: 0,
            row_index: Arc::from([]),
            fold_ids: Arc::from([]),
            n_folds: 2,
            scores: Arc::from([]),
            columns: Arc::from([ScoreColumn { arm: 0, threshold: None }]),
            adjustment_set: Arc::from([]),
            nuisance_provenance: Arc::from("test"),
            propensity_clip: None,
            treatment: VariableId::from_raw(0),
            intervened: Arc::from([]),
        };
        for variance in [f64::NAN, f64::INFINITY, -1.0, -1e-20] {
            let summary = ScoreSummary {
                means: Arc::from([1.0]),
                covariance: JointCovariance { dim: 1, values: Arc::from([variance]) },
                n_eff: 10.0,
            };
            assert!(table.linear_contrast(&summary, &[1.0]).is_err());
        }
    }

    #[test]
    fn review_threshold_effective_counts_are_weight_scale_invariant() {
        for scale in [1e-200, 1.0, 1e200] {
            let (events, non_events) = kish_threshold_support(
                &[0, 0, 0, 0],
                &[0.0, 0.0, 1.0, 1.0],
                &[scale, scale, scale, scale],
                0,
                Some(0.5),
            );
            assert!((events - 2.0).abs() < 1e-12);
            assert!((non_events - 2.0).abs() < 1e-12);
        }
    }

    #[test]
    fn distinct_thresholds_are_bitwise_not_absolute_tolerance() {
        let table_with = |thresholds: &[Option<f64>]| ScoreTable {
            observed_arm: Arc::from([]),
            propensities: Arc::from([]),
            observed_outcome: Arc::from([]),
            n_rows: 0,
            row_index: Arc::from([]),
            fold_ids: Arc::from([]),
            n_folds: 2,
            scores: Arc::from([]),
            columns: thresholds
                .iter()
                .enumerate()
                .map(|(i, &threshold)| ScoreColumn { arm: (i % 2) as u32, threshold })
                .collect(),
            adjustment_set: Arc::from([]),
            nuisance_provenance: Arc::from("test"),
            propensity_clip: None,
            treatment: VariableId::from_raw(0),
            intervened: Arc::from([]),
        };
        // Two arms at one threshold are one threshold; the mean functional has none.
        assert_eq!(table_with(&[Some(0.5), Some(0.5)]).distinct_threshold_count(), 1);
        assert_eq!(table_with(&[None, None]).distinct_threshold_count(), 0);
        // Outcomes in tiny units: 1e-17 and 3e-17 differ by less than f64::EPSILON but are
        // different thresholds, and 1e30 and its next float are different too.
        assert_eq!(table_with(&[Some(1e-17), Some(3e-17)]).distinct_threshold_count(), 2);
        let next = f64::from_bits(1e30_f64.to_bits() + 1);
        assert_eq!(table_with(&[Some(1e30), Some(next)]).distinct_threshold_count(), 2);
    }

    #[test]
    fn wire_round_trip_preserves_scores() {
        let table = ScoreTable {
            observed_arm: Arc::from([0, 1]),
            propensities: Arc::from([0.5; 4]),
            observed_outcome: Arc::from([1.0, 2.0]),
            n_rows: 2,
            row_index: Arc::from([0, 1]),
            fold_ids: Arc::from([0, 1]),
            n_folds: 2,
            scores: Arc::from([1.0, 2.0, 3.0, 4.0]),
            columns: Arc::from([
                ScoreColumn { arm: 0, threshold: None },
                ScoreColumn { arm: 1, threshold: None },
            ]),
            adjustment_set: Arc::from([VariableId::from_raw(2)]),
            nuisance_provenance: Arc::from("aipw.crossfit.v1"),
            propensity_clip: Some(0.02),
            treatment: VariableId::from_raw(0),
            intervened: Arc::from([]),
        };
        let restored = ScoreTable::from_wire(table.to_wire()).unwrap();
        assert_eq!(restored.scores.as_ref(), table.scores.as_ref());
        assert_eq!(restored.adjustment_set.as_ref(), table.adjustment_set.as_ref());
        assert_eq!(restored.propensity_clip, Some(0.02));
        let mut wire = table.to_wire();
        wire.propensity_clip = Some(0.7);
        assert!(ScoreTable::from_wire(wire).is_err(), "a clip outside (0, 0.5) is not a clip");
    }
}
