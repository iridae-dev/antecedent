
/// Prepared covariate design + treatment/outcome shared by every propensity estimator.
///
/// Built once from `(data, estimand, query)`; reused across point estimate and bootstrap.
#[derive(Clone, Debug)]
pub struct PreparedPropensityProblem {
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
/// Point estimates retain a [`MatchingIndex`] across compatible donor geometries (DESIGN §14.6).
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
    matching_index: Option<MatchingIndex>,
    /// Key of [`Self::matching_index`].
    matching_index_key: MatchingIndexKey,
    /// Number of times a new [`MatchingIndex`] was constructed (reuse diagnostics / benches).
    pub matching_index_builds: u32,
}

impl PropensityEstimationWorkspace {
    /// Ensure a matching index for `donor_features`, rebuilding only when geometry changes.
    fn ensure_matching_index(
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

    /// Estimated retained bytes for propensity + matching scratch (DESIGN §14.6).
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

fn fnv1a64(bytes_as_f64: &[f64]) -> u64 {
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
    OverlapPolicy::RequireDiagnostics { clip: Some(0.01), trim: None }
}

// ---------------------------------------------------------------------------------------------
// Shared prepare / small helpers
// ---------------------------------------------------------------------------------------------

pub(crate) fn prepare_propensity_problem(
    data: &TabularData,
    estimand: &IdentifiedEstimand,
    query: &AverageEffectQuery,
    overlap: OverlapPolicy,
) -> Result<PreparedPropensityProblem, EstimationError> {
    crate::util::refuse_explicit_override(
        overlap,
        "propensity estimators require RequireDiagnostics overlap policy; positivity is mandatory",
    )?;
    if !matches!(estimand.method_kind().ok(), Some(causal_expr::EstimandMethod::BackdoorAdjustment | causal_expr::EstimandMethod::BackdoorEfficient)) {
        return Err(EstimationError::IncompatibleEstimand {
            message: "propensity estimators expect backdoor.adjustment or backdoor.efficient",
        });
    }
    query.validate().map_err(|e| EstimationError::UnsupportedQuery(e.to_string()))?;
    if !query.effect_modifiers.is_empty() {
        return Err(EstimationError::UnsupportedQuery(
            "propensity estimators do not support effect modifiers".into(),
        ));
    }
    let active = intervention_f64(&query.active)?;
    let control = intervention_f64(&query.control)?;
    if (active - 1.0).abs() > 1e-12 || control.abs() > 1e-12 {
        return Err(EstimationError::UnsupportedQuery(
            "propensity estimators require binary treatment coded active=1.0, control=0.0".into(),
        ));
    }

    let treatment = query.treatment;
    let outcome = query.outcome;
    let mut ids = Vec::with_capacity(2 + estimand.adjustment_set.len());
    ids.push(treatment);
    ids.push(outcome);
    ids.extend_from_slice(&estimand.adjustment_set);
    let row_mask =
        data.complete_case_mask(&ids).map_err(EstimationError::from)?;
    let t = data
        .float64_masked(treatment, &row_mask)
        .map_err(EstimationError::from)?;
    let y = data
        .float64_masked(outcome, &row_mask)
        .map_err(EstimationError::from)?;
    let nrows = t.len();
    if nrows == 0 {
        return Err(EstimationError::data_msg("no complete-case rows for propensity design"));
    }
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
        let col =
            data.float64_masked(z, &row_mask).map_err(EstimationError::from)?;
        let base = (1 + i) * nrows;
        for r in 0..nrows {
            design[base + r] = col[r];
        }
        covariate_cols.push(Arc::from(col));
    }

    Ok(PreparedPropensityProblem {
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
        return Err(EstimationError::data_msg("overlap trim removed every row; no common-support units remain"));
    }
    Ok(Some(retained))
}

/// Restrict `(treatment, outcome, features)` to `retained` rows (`features` row-major with
/// `dim` columns); clones the full slices when `retained` is `None` (no trim configured).
fn restrict_to_rows(
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

fn overlap_clip_trim(overlap: OverlapPolicy) -> (Option<f64>, Option<f64>) {
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

fn to_row_major(cols: &[Arc<[f64]>], nrows: usize) -> Vec<f64> {
    let dim = cols.len().max(1);
    let mut out = vec![0.0; nrows * dim];
    for (c, col) in cols.iter().enumerate() {
        for r in 0..nrows {
            out[r * dim + c] = col[r];
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------
// IPW weights + Hajek estimator (shared by `PropensityWeighting`)
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum IpwTarget {
    Ate,
    Att,
    Atc,
}

impl IpwTarget {
    fn from_population(pop: &TargetPopulation) -> Result<Self, EstimationError> {
        match pop {
            TargetPopulation::AllObserved => Ok(Self::Ate),
            TargetPopulation::Treated => Ok(Self::Att),
            TargetPopulation::Untreated => Ok(Self::Atc),
            _ => Err(EstimationError::UnsupportedQuery(
                "propensity weighting supports AllObserved, Treated, or Untreated target populations".into(),
            )),
        }
    }

    fn weight(self, t: f64, e: f64) -> f64 {
        match self {
            Self::Ate => {
                if t > 0.5 {
                    1.0 / e
                } else {
                    1.0 / (1.0 - e)
                }
            }
            Self::Att => {
                if t > 0.5 {
                    1.0
                } else {
                    e / (1.0 - e)
                }
            }
            Self::Atc => {
                if t > 0.5 {
                    (1.0 - e) / e
                } else {
                    1.0
                }
            }
        }
    }
}

/// `scores_for_weight` feeds the weight formula (typically clipped); `scores_for_trim` feeds
/// the trim decision (typically the raw, pre-clip scores) — they may be the same slice.
fn compute_ipw_weights(
    treatment: &[f64],
    scores_for_weight: &[f64],
    scores_for_trim: &[f64],
    target: IpwTarget,
    trim: Option<f64>,
) -> Vec<f64> {
    treatment
        .iter()
        .zip(scores_for_weight)
        .zip(scores_for_trim)
        .map(|((&t, &e), &raw)| {
            if let Some(tr) = trim {
                if raw < tr || raw > 1.0 - tr {
                    return 0.0;
                }
            }
            target.weight(t, e)
        })
        .collect()
}

fn hajek_difference(
    treatment: &[f64],
    outcome: &[f64],
    weights: &[f64],
) -> Result<f64, EstimationError> {
    let (mut num1, mut den1, mut num0, mut den0) = (0.0, 0.0, 0.0, 0.0);
    for ((&t, &y), &w) in treatment.iter().zip(outcome).zip(weights) {
        if t > 0.5 {
            num1 += w * y;
            den1 += w;
        } else {
            num0 += w * y;
            den0 += w;
        }
    }
    if den1 <= 0.0 || den0 <= 0.0 {
        return Err(EstimationError::data_msg("IPW weighting left an arm with zero total weight (trimming/clipping removed all treated or all control units)"));
    }
    Ok(num1 / den1 - num0 / den0)
}

fn hajek_weighted_mean(
    treatment: &[f64],
    outcome: &[f64],
    weights: &[f64],
    want_treated: bool,
) -> f64 {
    let (mut num, mut den) = (0.0, 0.0);
    for i in 0..treatment.len() {
        if (treatment[i] > 0.5) == want_treated {
            num += weights[i] * outcome[i];
            den += weights[i];
        }
    }
    if den > 0.0 { num / den } else { f64::NAN }
}

fn hajek_group_variance(
    treatment: &[f64],
    outcome: &[f64],
    weights: &[f64],
    want_treated: bool,
    mu: f64,
) -> f64 {
    let (mut num, mut den) = (0.0, 0.0);
    for i in 0..treatment.len() {
        if (treatment[i] > 0.5) == want_treated {
            let w = weights[i];
            num += w * w * (outcome[i] - mu).powi(2);
            den += w;
        }
    }
    if den > 0.0 { num / (den * den) } else { f64::NAN }
}

/// Linearized (ratio-estimator) analytic SE of the Hajek ATE/ATT/ATC difference.
fn hajek_analytic_se(treatment: &[f64], outcome: &[f64], weights: &[f64]) -> f64 {
    let mu1 = hajek_weighted_mean(treatment, outcome, weights, true);
    let mu0 = hajek_weighted_mean(treatment, outcome, weights, false);
    let v1 = hajek_group_variance(treatment, outcome, weights, true, mu1);
    let v0 = hajek_group_variance(treatment, outcome, weights, false, mu0);
    (v1 + v0).sqrt()
}
