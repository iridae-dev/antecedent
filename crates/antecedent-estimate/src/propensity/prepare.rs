//! Propensity problem preparation and shared workspace.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::too_many_lines)]

use std::sync::Arc;

use antecedent_core::{AverageEffectQuery, PopulationRegistry, TargetPopulation, VariableId};
use antecedent_data::{TableView, TabularData};
use antecedent_expr::IdentifiedEstimand;
use antecedent_stats::{
    FaerBackend, GlmOptions, MatchingDistance, MatchingIndex, PropensityFit, PropensityWorkspace,
    fit_propensity,
};

use crate::adjustment::intervention_f64;
use crate::error::EstimationError;
use crate::overlap::OverlapPolicy;
use crate::util::stats_err;

/// Prepared covariate design + treatment/outcome shared by every propensity estimator.
///
/// Built once from `(data, estimand, query)`; reused across point estimate and bootstrap.
#[derive(Clone, Debug)]
pub struct PreparedPropensityProblem {
    /// Shared learner nuisance cache; changing any fit input invalidates reuse.
    pub(crate) learner_cache:
        Arc<std::sync::Mutex<Vec<Arc<crate::learn_nuisance::AipwCacheEntry>>>>,
    /// Column-major `[1 | Z…]` design used to fit the propensity model.
    pub design_matrix: Arc<[f64]>,
    /// Number of design columns (`1 + adjustment_set.len()`).
    pub design_ncols: usize,
    /// Number of complete-case rows.
    pub nrows: usize,
    /// Binary treatment indicator (0/1), length `nrows`.
    pub treatment: Arc<[f64]>,
    /// Outcome, length `nrows`.
    pub outcome: Arc<[f64]>,
    /// Raw adjustment covariate columns, in `adjustment_set` order (excludes intercept).
    pub covariates: Arc<[Arc<[f64]>]>,
    /// Estimand method tag.
    pub method: Arc<str>,
    /// Adjustment set.
    pub adjustment_set: Arc<[VariableId]>,
    /// Overlap policy applied.
    pub overlap: OverlapPolicy,
    /// Target population requested by the query.
    pub target_population: TargetPopulation,
    /// Complete-case–aligned observation weights from [`TargetPopulation::CustomDistribution`].
    ///
    /// `None` for all other target populations. Length equals [`Self::nrows`] when present.
    pub target_weights: Option<Arc<[f64]>>,
    /// Original data-frame row index of each complete-case row.
    pub row_index: Arc<[u32]>,
    /// Treatment variable id (for score-table provenance).
    pub treatment_id: VariableId,
    /// Optional complete-case fold ids. `None` draws a seeded, arm-stratified, unit-level
    /// plan from [`Self::fold_seed`] (see `learn_nuisance::crossfit_fold_plan`).
    pub fold_assignment: Option<Arc<[u32]>>,
    /// Seed of the cross-fit fold plan used when [`Self::fold_assignment`] is `None`.
    /// Callers holding an execution context set it to the master seed so the recorded
    /// cross-fit seed controls the folds as well as the learners.
    pub fold_seed: u64,
}

/// Fitted propensity model shared by weighting, stratification, and matching estimators.
///
/// Retains the raw [`PropensityFit`] (coefficients, scores, GLM diagnostics) plus the
/// clip-adjusted scores actually used for weighting/matching/distance calculations.
#[derive(Clone, Debug)]
pub struct PropensityModel {
    /// Raw logistic fit (pre-clip scores in `fit.scores`).
    pub fit: PropensityFit,
    /// Clip threshold applied to `clipped_scores`, taken from the overlap policy.
    pub clip: Option<f64>,
    /// Propensity scores after clipping into `[clip, 1 - clip]` (identical to `fit.scores`
    /// when `clip` is `None`).
    pub clipped_scores: Vec<f64>,
}

impl PropensityModel {
    /// Fit the logistic propensity model on `problem`'s design, applying the clip threshold
    /// from `problem.overlap` when present.
    ///
    /// # Errors
    ///
    /// Propagates GLM/backend failures.
    pub fn fit(
        problem: &PreparedPropensityProblem,
        backend: &FaerBackend,
        workspace: &mut PropensityWorkspace,
        options: &GlmOptions,
    ) -> Result<Self, EstimationError> {
        let fit = fit_propensity(
            &problem.design_matrix,
            problem.nrows,
            problem.design_ncols,
            &problem.treatment,
            backend,
            workspace,
            options,
        )
        .map_err(stats_err)?;
        fit.glm.require_ok().map_err(stats_err)?;
        let clip = clip_of(problem.overlap);
        let mut clipped_scores = fit.scores.clone();
        if let Some(c) = clip {
            clamp_scores(&mut clipped_scores, c);
        }
        Ok(Self { fit, clip, clipped_scores })
    }

    /// `logit(ê)` per row computed from the fitted linear predictor `η̂ = Xγ̂`, so it stays
    /// finite when `ê` rounds to 0 or 1 (`|η̂| ≳ 37`). Under a clip the predictor is clamped
    /// to `±logit(1 − clip)`, which is exactly the logit of the clipped score.
    pub(crate) fn logit_scores(&self, problem: &PreparedPropensityProblem) -> Vec<f64> {
        let n = problem.nrows;
        let mut eta = vec![0.0; n];
        for (c, &coef) in self.fit.coefficients.iter().enumerate().take(problem.design_ncols) {
            let column = &problem.design_matrix[c * n..(c + 1) * n];
            for (slot, &x) in eta.iter_mut().zip(column) {
                *slot += coef * x;
            }
        }
        eta.into_iter().map(|e| clamp_linear_predictor(e, self.clip)).collect()
    }
}

/// Clamp a logistic linear predictor to the band whose logit-inverse is `[clip, 1 − clip]`.
pub(crate) fn clamp_linear_predictor(eta: f64, clip: Option<f64>) -> f64 {
    match clip {
        Some(c) => {
            let bound = ((1.0 - c) / c).ln();
            eta.clamp(-bound, bound)
        }
        None => eta,
    }
}

/// Refuse propensities that are exactly 0 or 1 (or not finite): they carry an infinite
/// inverse-probability weight. With a clip they cannot occur; without one the user asked for
/// no floor, so the fit is refused instead of silently flooring at a hidden constant.
pub(crate) fn require_interior_propensities(scores: &[f64]) -> Result<(), EstimationError> {
    if scores.iter().any(|&e| !(e > 0.0 && e < 1.0)) {
        return Err(EstimationError::Overlap {
            message: "a fitted propensity is 0 or 1 (or not finite) and no clip is set;                       the inverse-probability weight is infinite — set an overlap clip",
        });
    }
    Ok(())
}

/// Geometry key for a cached [`MatchingIndex`] (rebuild when this changes).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MatchingIndexKey {
    dim: usize,
    n_donors: usize,
    distance: MatchingDistance,
    /// FNV-1a over donor feature bytes (stable across identical layouts).
    features_hash: u64,
}

impl Default for MatchingIndexKey {
    fn default() -> Self {
        Self { dim: 0, n_donors: 0, distance: MatchingDistance::Euclidean, features_hash: 0 }
    }
}

/// Reusable scratch for propensity estimators.
///
/// Point estimates retain a [`MatchingIndex`] across compatible donor geometries.
/// Bootstrap replicates rebuild the index whenever resampled donors change the geometry key.
#[derive(Clone, Debug, Default)]
pub struct PropensityEstimationWorkspace {
    /// Logistic IRLS scratch reused across point-estimate and bootstrap propensity refits.
    pub propensity: PropensityWorkspace,
    /// Matching output buffer: matched donor row per query row.
    pub matching_donor_rows: Vec<usize>,
    /// Matching output buffer: distance to the matched donor per query row.
    pub matching_distances: Vec<f64>,
    /// Cached nearest-neighbor index for the current donor geometry.
    pub(crate) matching_index: Option<MatchingIndex>,
    /// Key of [`Self::matching_index`].
    matching_index_key: MatchingIndexKey,
    /// Number of times a new [`MatchingIndex`] was constructed (reuse diagnostics / benches).
    pub matching_index_builds: u32,
    /// Clipped propensity scores for matching bootstrap (raw scores stay on `propensity`).
    pub clip_scratch: Vec<f64>,
}

impl PropensityEstimationWorkspace {
    /// Ensure a matching index for `donor_features`, rebuilding only when geometry changes.
    pub(crate) fn ensure_matching_index(
        &mut self,
        donor_features: &[f64],
        dim: usize,
        distance: MatchingDistance,
    ) -> Result<(), EstimationError> {
        let n_donors = donor_features.len() / dim.max(1);
        let key =
            MatchingIndexKey { dim, n_donors, distance, features_hash: fnv1a64(donor_features) };
        let needs_rebuild = self.matching_index.is_none() || self.matching_index_key != key;
        if needs_rebuild {
            let donor_ids: Vec<usize> = (0..n_donors).collect();
            let index = MatchingIndex::exact(donor_features, dim, &donor_ids, distance)
                .map_err(stats_err)?;
            self.matching_index = Some(index);
            self.matching_index_key = key;
            self.matching_index_builds = self.matching_index_builds.saturating_add(1);
        }
        Ok(())
    }

    /// Estimated retained bytes for propensity + matching scratch.
    #[must_use]
    pub fn retained_memory_bytes(&self) -> u64 {
        let mut bytes = 0u64;
        bytes += (self.propensity.scores.capacity() * std::mem::size_of::<f64>()) as u64;
        bytes += (self.propensity.ols.scratch.capacity() * std::mem::size_of::<f64>()) as u64;
        bytes += (self.propensity.ols.rhs.capacity() * std::mem::size_of::<f64>()) as u64;
        bytes += (self.propensity.ols.residuals.capacity() * std::mem::size_of::<f64>()) as u64;
        bytes += (self.matching_donor_rows.capacity() * std::mem::size_of::<usize>()) as u64;
        bytes += (self.matching_distances.capacity() * std::mem::size_of::<f64>()) as u64;
        if let Some(ref idx) = self.matching_index {
            bytes += idx.retained_memory_bytes();
        }
        bytes
    }
}

pub(crate) fn fnv1a64(bytes_as_f64: &[f64]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0100_0000_01b3;
    let mut hash = OFFSET;
    for &v in bytes_as_f64 {
        for b in v.to_bits().to_le_bytes() {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(PRIME);
        }
    }
    hash
}

/// Default overlap policy for all propensity estimators: diagnostics mandatory,
/// propensities clipped into `[0.01, 0.99]`, no trimming.
#[must_use]
pub const fn default_propensity_overlap() -> OverlapPolicy {
    OverlapPolicy::RequireDiagnostics {
        clip: Some(crate::overlap::DEFAULT_PROPENSITY_CLIP),
        trim: None,
    }
}

// ---------------------------------------------------------------------------------------------
// Shared prepare / small helpers
// ---------------------------------------------------------------------------------------------

/// Prepare propensity inputs with optional named-predicate / distribution bindings.
pub(crate) fn prepare_propensity_problem_with_registry(
    data: &TabularData,
    estimand: &IdentifiedEstimand,
    query: &AverageEffectQuery,
    overlap: OverlapPolicy,
    registry: Option<&PopulationRegistry>,
) -> Result<PreparedPropensityProblem, EstimationError> {
    crate::util::refuse_explicit_override(
        overlap,
        "propensity estimators require RequireDiagnostics overlap policy; positivity is mandatory",
    )?;
    overlap.validate()?;
    if !estimand.is_adjustment_shaped() {
        return Err(EstimationError::IncompatibleEstimand {
            message: "propensity estimators expect an adjustment-shaped estimand",
        });
    }
    query.validate()?;
    if !query.effect_modifiers.is_empty() {
        return Err(EstimationError::unsupported(
            "propensity estimators do not support effect modifiers",
        ));
    }
    let active = intervention_f64(&query.active)?;
    let control = intervention_f64(&query.control)?;
    if (active - 1.0).abs() > 1e-12 || control.abs() > 1e-12 {
        return Err(EstimationError::unsupported(
            "propensity estimators require binary treatment coded active=1.0, control=0.0",
        ));
    }

    let treatment = query.treatment;
    let outcome = query.outcome;
    let mut ids = Vec::with_capacity(2 + estimand.adjustment_set.len());
    ids.push(treatment);
    ids.push(outcome);
    ids.extend_from_slice(&estimand.adjustment_set);
    let mut row_mask = data.complete_case_mask(&ids).map_err(EstimationError::from)?;
    let n_full = data.row_count();
    let mut full_weights: Option<Arc<[f64]>> = None;
    match &query.target_population {
        TargetPopulation::Predicate(_) => {
            crate::prepare::intersect_predicate_mask(
                &mut row_mask,
                &query.target_population,
                n_full,
                registry,
            )?;
        }
        TargetPopulation::CustomDistribution(_) => {
            let sel = query
                .target_population
                .resolve(n_full, None, registry)
                .map_err(|e| EstimationError::data_msg(e.to_string()))?;
            // Keep complete-case rows; weights apply at estimation time (aligned below).
            full_weights = sel.weights;
        }
        _ => {}
    }
    let t = data.float64_masked(treatment, &row_mask).map_err(EstimationError::from)?;
    let y = data.float64_masked(outcome, &row_mask).map_err(EstimationError::from)?;
    let nrows = t.len();
    if nrows == 0 {
        return Err(EstimationError::data_msg("no complete-case rows for propensity design"));
    }
    let target_weights = if let Some(w) = full_weights {
        let mut aligned = Vec::with_capacity(nrows);
        for (i, &keep) in row_mask.iter().enumerate() {
            if keep {
                aligned.push(w.get(i).copied().unwrap_or(0.0));
            }
        }
        if aligned.len() != nrows {
            return Err(EstimationError::data_msg(
                "CustomDistribution weights failed to align with complete-case rows",
            ));
        }
        if aligned.iter().all(|&x| x <= 0.0) {
            return Err(EstimationError::data_msg(
                "CustomDistribution weights are all zero on complete-case rows",
            ));
        }
        Some(Arc::from(aligned))
    } else {
        None
    };
    // The query levels are already validated to be exactly 0.0/1.0 above; the treatment
    // column must match them, otherwise a {1,2}-coded or continuous treatment would be
    // silently dichotomized at t > 0.5 and fed to the logistic fit as-is.
    for &ti in &t {
        if ti.abs() > 1e-12 && (ti - 1.0).abs() > 1e-12 {
            return Err(EstimationError::data_msg(format!(
                "propensity estimators require a binary treatment column matching the query \
                 levels (0.0/1.0); found treatment value {ti}"
            )));
        }
    }

    let ncols = 1 + estimand.adjustment_set.len();
    let mut design = vec![0.0; nrows * ncols];
    for r in 0..nrows {
        design[r] = 1.0;
    }
    let mut covariate_cols: Vec<Arc<[f64]>> = Vec::with_capacity(estimand.adjustment_set.len());
    for (i, &z) in estimand.adjustment_set.iter().enumerate() {
        let col = data.float64_masked(z, &row_mask).map_err(EstimationError::from)?;
        let base = (1 + i) * nrows;
        for r in 0..nrows {
            design[base + r] = col[r];
        }
        covariate_cols.push(Arc::from(col));
    }

    Ok(PreparedPropensityProblem {
        learner_cache: Arc::default(),
        design_matrix: Arc::from(design),
        design_ncols: ncols,
        nrows,
        treatment: Arc::from(t),
        outcome: Arc::from(y),
        covariates: Arc::from(covariate_cols),
        method: Arc::clone(&estimand.method),
        adjustment_set: Arc::clone(&estimand.adjustment_set),
        overlap,
        target_population: query.target_population.clone(),
        target_weights,
        row_index: {
            let mut idx = Vec::with_capacity(nrows);
            for (i, &keep) in row_mask.iter().enumerate() {
                if keep {
                    idx.push(u32::try_from(i).map_err(|_| {
                        EstimationError::data_msg("row index exceeds the u32 row-id capacity")
                    })?);
                }
            }
            Arc::from(idx)
        },
        treatment_id: treatment,
        fold_assignment: None,
        fold_seed: 0,
    })
}

pub(crate) fn clip_of(overlap: OverlapPolicy) -> Option<f64> {
    overlap_clip_trim(overlap).0
}

pub(crate) fn trim_of(overlap: OverlapPolicy) -> Option<f64> {
    overlap_clip_trim(overlap).1
}

/// Indices of rows whose **raw** (pre-clip) propensity lies inside the `[trim, 1 - trim]`
/// common-support band. Returns `None` when no trim is configured (all rows retained).
///
/// Trimming redefines the estimand to the common-support population — exactly what the
/// [`OverlapReport`] built from the same raw scores claims via `excluded_fraction` /
/// `excluded_regions`.
///
/// # Errors
///
/// Every row falls outside the band.
pub(crate) fn trim_retained_rows(
    raw_scores: &[f64],
    trim: Option<f64>,
) -> Result<Option<Vec<usize>>, EstimationError> {
    let Some(t) = trim else { return Ok(None) };
    let retained: Vec<usize> = raw_scores
        .iter()
        .enumerate()
        .filter_map(|(i, &p)| (p >= t && p <= 1.0 - t).then_some(i))
        .collect();
    if retained.is_empty() {
        return Err(EstimationError::data_msg(
            "overlap trim removed every row; no common-support units remain",
        ));
    }
    Ok(Some(retained))
}

/// Restrict `(treatment, outcome, features)` to `retained` rows (`features` row-major with
/// `dim` columns); clones the full slices when `retained` is `None` (no trim configured).
pub(crate) fn restrict_to_rows(
    treatment: &[f64],
    outcome: &[f64],
    features: &[f64],
    dim: usize,
    retained: Option<&[usize]>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    match retained {
        Some(idx) => {
            (gather(treatment, idx), gather(outcome, idx), gather_rowmajor(features, dim, idx))
        }
        None => (treatment.to_vec(), outcome.to_vec(), features.to_vec()),
    }
}

/// Optionally gather row labels aligned to prepared `nrows`, then restrict to `retained`.
pub(crate) fn gather_optional_row_labels<T: Copy>(
    labels: Option<&[T]>,
    nrows: usize,
    retained: Option<&[usize]>,
    name: &str,
) -> Result<Option<Vec<T>>, EstimationError> {
    let Some(labels) = labels else {
        return Ok(None);
    };
    if labels.len() != nrows {
        return Err(EstimationError::data_msg(format!(
            "{name} length {} != nrows {nrows}",
            labels.len()
        )));
    }
    Ok(Some(match retained {
        Some(idx) => idx.iter().map(|&i| labels[i]).collect(),
        None => labels.to_vec(),
    }))
}

/// Gather each multiway cluster dimension with the same retained-row map as treatment.
pub(crate) fn gather_optional_multiway(
    dims: Option<&[Vec<u32>]>,
    nrows: usize,
    retained: Option<&[usize]>,
) -> Result<Option<Vec<Vec<u32>>>, EstimationError> {
    let Some(dims) = dims else {
        return Ok(None);
    };
    let mut out = Vec::with_capacity(dims.len());
    for (i, d) in dims.iter().enumerate() {
        let name = format!("multiway_ids[{i}]");
        let Some(gathered) =
            gather_optional_row_labels(Some(d.as_slice()), nrows, retained, &name)?
        else {
            return Ok(None);
        };
        out.push(gathered);
    }
    Ok(Some(out))
}

pub(crate) fn overlap_clip_trim(overlap: OverlapPolicy) -> (Option<f64>, Option<f64>) {
    match overlap {
        OverlapPolicy::RequireDiagnostics { clip, trim, .. } => (clip, trim),
        OverlapPolicy::ExplicitOverride => (None, None),
    }
}

pub(crate) fn clamp_scores(scores: &mut [f64], clip: f64) {
    for s in scores.iter_mut() {
        *s = s.clamp(clip, 1.0 - clip);
    }
}

pub(crate) fn to_row_major(cols: &[Arc<[f64]>], nrows: usize) -> Vec<f64> {
    let dim = cols.len().max(1);
    let mut out = vec![0.0; nrows * dim];
    for (c, col) in cols.iter().enumerate() {
        for r in 0..nrows {
            out[r * dim + c] = col[r];
        }
    }
    out
}

pub(crate) fn split_by_treatment(treatment: &[f64]) -> (Vec<usize>, Vec<usize>) {
    let mut treated = Vec::new();
    let mut control = Vec::new();
    for (i, &t) in treatment.iter().enumerate() {
        if t > 0.5 {
            treated.push(i);
        } else {
            control.push(i);
        }
    }
    (treated, control)
}

pub(crate) fn gather(values: &[f64], idx: &[usize]) -> Vec<f64> {
    idx.iter().map(|&i| values[i]).collect()
}

/// [`gather`] into a reused buffer for replicate loops.
pub(crate) fn gather_into(out: &mut Vec<f64>, values: &[f64], idx: &[usize]) {
    out.clear();
    out.extend(idx.iter().map(|&i| values[i]));
}

pub(crate) fn gather_rowmajor(matrix: &[f64], dim: usize, idx: &[usize]) -> Vec<f64> {
    let mut out = Vec::with_capacity(idx.len() * dim);
    for &i in idx {
        out.extend_from_slice(&matrix[i * dim..(i + 1) * dim]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logit_from_the_linear_predictor_stays_finite_where_the_score_rounds_to_one() {
        // At η = 40 the fitted probability is exactly 1.0 in f64, so ln(e/(1−e)) is +∞.
        let e = 1.0 / (1.0 + (-40.0_f64).exp());
        assert!(e == 1.0);
        assert!(((e / (1.0 - e)).ln()).is_infinite());
        assert_eq!(clamp_linear_predictor(40.0, None), 40.0);
        assert_eq!(clamp_linear_predictor(-40.0, None), -40.0);
        // Under a clip the predictor is clamped to logit(1 − clip), the logit of the clipped score.
        let bound = (0.99_f64 / 0.01).ln();
        assert!((clamp_linear_predictor(40.0, Some(0.01)) - bound).abs() < 1e-12);
        assert!((clamp_linear_predictor(-40.0, Some(0.01)) + bound).abs() < 1e-12);
        assert_eq!(clamp_linear_predictor(0.3, Some(0.01)), 0.3);
    }

    #[test]
    fn exact_zero_or_one_propensities_are_refused_not_floored() {
        assert!(require_interior_propensities(&[0.2, 1e-12, 1.0 - 1e-12]).is_ok());
        for bad in [0.0, 1.0, f64::NAN] {
            assert!(require_interior_propensities(&[0.5, bad]).is_err(), "{bad}");
        }
    }
}
