//! Conditional logit for one choice among a set of options.
//!
//! Each choice set holds two or more options; exactly one option of each set is chosen; features
//! vary by option. The model is
//!
//! `P(chosen = i | set) = exp(x_i . beta) / sum_{j in set} exp(x_j . beta)`.
//!
//! There is no intercept: adding a constant vector to every option of one set leaves every
//! probability unchanged, so only within-set feature differences carry information. The fit works
//! on the differences `z_j = x_j - x_chosen`, which makes it exactly invariant to such shifts up to
//! the rounding of one subtraction.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use crate::gram::{invert_square, solve_spd};
use thiserror::Error;

/// Choice sets for a conditional logit fit.
#[derive(Clone, Copy, Debug)]
pub struct ConditionalLogitDesign<'a> {
    /// Row-major features, one row per option across all sets, `n_features` columns.
    pub features: &'a [f64],
    /// Number of feature columns (`p >= 1`).
    pub n_features: usize,
    /// Set boundaries, `n_sets + 1` entries: set `s` owns rows `set_starts[s]..set_starts[s + 1]`.
    pub set_starts: &'a [usize],
    /// Chosen option of each set, as an index within the set (`n_sets` entries).
    pub chosen: &'a [usize],
    /// Cluster label of each set (`n_sets` entries); sets of one decision-maker share a label.
    pub clusters: Option<&'a [u32]>,
}

/// Iteration controls for [`fit_conditional_logit`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConditionalLogitOptions {
    /// Largest number of accepted Newton steps.
    pub max_iterations: usize,
    /// Convergence threshold on the gradient, relative to the gradient scale (see
    /// [`fit_conditional_logit`]). Finite and positive.
    pub gradient_tolerance: f64,
    /// Largest within-set linear-predictor difference `max |z_j . beta|` beyond which the
    /// coefficient direction is tested as a separating direction. Finite and positive.
    pub separation_bound: f64,
}

impl Default for ConditionalLogitOptions {
    fn default() -> Self {
        Self { max_iterations: 200, gradient_tolerance: 1e-12, separation_bound: 20.0 }
    }
}

/// Maximum-likelihood conditional logit fit.
#[derive(Clone, Debug, PartialEq)]
pub struct ConditionalLogitFit {
    /// Coefficients `beta`, one per feature column.
    pub coefficients: Vec<f64>,
    /// Log-likelihood at `coefficients`.
    pub log_likelihood: f64,
    /// Log-likelihood with every coefficient zero: `-sum_s ln(n_options_s)`.
    pub null_log_likelihood: f64,
    /// Observed information (negative Hessian of the log-likelihood) at `coefficients`,
    /// row-major `p x p`.
    pub information: Vec<f64>,
    /// Inverse of `information`, row-major `p x p`.
    pub covariance: Vec<f64>,
    /// Cluster-robust sandwich covariance, row-major `p x p`, present when clusters were given.
    pub cluster_covariance: Option<Vec<f64>>,
    /// Accepted Newton steps.
    pub iterations: usize,
    /// Largest absolute gradient entry at `coefficients`.
    pub gradient_norm: f64,
}

/// Why a conditional logit fit or evaluation was refused.
#[derive(Clone, Debug, PartialEq, Error)]
pub enum ConditionalLogitError {
    /// `n_features` is zero.
    #[error("conditional logit needs at least one feature")]
    NoFeatures,
    /// The feature slice length is not a multiple of `n_features`.
    #[error("feature length {len} is not a multiple of {n_features} features")]
    FeatureLength {
        /// Length of the feature slice.
        len: usize,
        /// Declared number of features.
        n_features: usize,
    },
    /// `set_starts` describes no set (fewer than two entries).
    #[error("conditional logit needs at least one choice set")]
    NoSets,
    /// `set_starts` does not start at zero.
    #[error("set boundaries start at {first}, not 0")]
    SetStartsNotZero {
        /// First entry of `set_starts`.
        first: usize,
    },
    /// A set holds fewer than two options (boundaries not increasing by at least two).
    #[error("choice set {set} holds {options} options; at least two are needed")]
    SetTooSmall {
        /// Set index.
        set: usize,
        /// Number of options (zero when the boundaries decrease).
        options: usize,
    },
    /// The last set boundary is not the number of feature rows.
    #[error("set boundaries end at {last}, but the features hold {rows} rows")]
    SetStartsRowMismatch {
        /// Last entry of `set_starts`.
        last: usize,
        /// Feature rows.
        rows: usize,
    },
    /// `chosen` does not hold one entry per set.
    #[error("{len} chosen indices for {n_sets} sets")]
    ChosenLength {
        /// Length of `chosen`.
        len: usize,
        /// Number of sets.
        n_sets: usize,
    },
    /// A chosen index is not an option of its set.
    #[error("chosen index {index} is outside set {set} of {options} options")]
    ChosenOutOfRange {
        /// Set index.
        set: usize,
        /// The chosen index given.
        index: usize,
        /// Options in the set.
        options: usize,
    },
    /// `clusters` does not hold one entry per set.
    #[error("{len} cluster labels for {n_sets} sets")]
    ClusterLength {
        /// Length of `clusters`.
        len: usize,
        /// Number of sets.
        n_sets: usize,
    },
    /// A feature, or a within-set feature difference, is not finite.
    #[error("non-finite feature at row {row}, column {column}")]
    NonFiniteFeature {
        /// Option row.
        row: usize,
        /// Feature column.
        column: usize,
    },
    /// The coefficient vector does not hold one entry per feature.
    #[error("{len} coefficients for {n_features} features")]
    CoefficientLength {
        /// Length of the coefficient vector.
        len: usize,
        /// Number of features.
        n_features: usize,
    },
    /// A coefficient is not finite.
    #[error("non-finite coefficient {index}")]
    NonFiniteCoefficient {
        /// Coefficient index.
        index: usize,
    },
    /// An iteration control is out of range.
    #[error("gradient tolerance and separation bound must be finite and positive")]
    InvalidOptions,
    /// The within-set centred design has a null direction; `columns` are linearly dependent on
    /// earlier columns (a column that is zero after centring is one of them).
    #[error("rank-deficient within-set design; dependent columns {columns:?}")]
    RankDeficient {
        /// Dependent column indices, ascending.
        columns: Vec<usize>,
    },
    /// The likelihood has no finite maximum: `direction` (unit Euclidean norm) weakly prefers the
    /// chosen option in every set and strictly in at least one.
    #[error("choices are separated; no finite maximum-likelihood estimate")]
    Separated {
        /// Separating direction, unit Euclidean norm.
        direction: Vec<f64>,
    },
    /// The observed information could not be factored or inverted at a point where no separating
    /// direction was found.
    #[error("observed information is numerically singular")]
    SingularInformation,
    /// The iteration budget ran out, or step halving could not keep the log-likelihood from
    /// decreasing, before the gradient met the tolerance.
    #[error("no convergence after {iterations} iterations; gradient norm {gradient_norm}")]
    NotConverged {
        /// Accepted Newton steps.
        iterations: usize,
        /// Largest absolute gradient entry at the last iterate.
        gradient_norm: f64,
    },
}

impl ConditionalLogitError {
    /// Stable reason id in the `conditional_logit` family.
    #[must_use]
    pub fn detail(&self) -> &'static str {
        match self {
            Self::NoFeatures => "conditional_logit.no_features",
            Self::FeatureLength { .. } => "conditional_logit.feature_length",
            Self::NoSets => "conditional_logit.no_sets",
            Self::SetStartsNotZero { .. } => "conditional_logit.set_starts_not_zero",
            Self::SetTooSmall { .. } => "conditional_logit.set_too_small",
            Self::SetStartsRowMismatch { .. } => "conditional_logit.set_starts_row_mismatch",
            Self::ChosenLength { .. } => "conditional_logit.chosen_length",
            Self::ChosenOutOfRange { .. } => "conditional_logit.chosen_out_of_range",
            Self::ClusterLength { .. } => "conditional_logit.cluster_length",
            Self::NonFiniteFeature { .. } => "conditional_logit.non_finite_feature",
            Self::CoefficientLength { .. } => "conditional_logit.coefficient_length",
            Self::NonFiniteCoefficient { .. } => "conditional_logit.non_finite_coefficient",
            Self::InvalidOptions => "conditional_logit.invalid_options",
            Self::RankDeficient { .. } => "conditional_logit.rank_deficient",
            Self::Separated { .. } => "conditional_logit.separated",
            Self::SingularInformation => "conditional_logit.singular_information",
            Self::NotConverged { .. } => "conditional_logit.not_converged",
        }
    }
}

/// Relative tolerance of the column-dependence test on the within-set centred design.
const RANK_TOLERANCE: f64 = 1e-9;
/// A direction may exceed a tie with a non-chosen option by at most this fraction of `|z_j|`.
const SEPARATION_TIE: f64 = 1e-8;
/// A direction strictly prefers the chosen option when `z_j . d < -SEPARATION_STRICT |z_j|`.
const SEPARATION_STRICT: f64 = 1e-6;
/// Step halvings tried before a Newton step is given up.
const MAX_HALVINGS: usize = 60;

/// Validated design with the within-set differences `z_r = x_r - x_chosen(set of r)`.
struct Prepared<'a> {
    p: usize,
    n_sets: usize,
    set_starts: &'a [usize],
    chosen: &'a [usize],
    clusters: Option<&'a [u32]>,
    /// Row-major `rows x p`; the chosen row of every set is exactly zero.
    diffs: Vec<f64>,
}

impl Prepared<'_> {
    fn row(&self, r: usize) -> &[f64] {
        &self.diffs[r * self.p..(r + 1) * self.p]
    }

    fn set_rows(&self, s: usize) -> std::ops::Range<usize> {
        self.set_starts[s]..self.set_starts[s + 1]
    }
}

fn prepare<'a>(design: &ConditionalLogitDesign<'a>) -> Result<Prepared<'a>, ConditionalLogitError> {
    let p = design.n_features;
    if p == 0 {
        return Err(ConditionalLogitError::NoFeatures);
    }
    let len = design.features.len();
    if !len.is_multiple_of(p) {
        return Err(ConditionalLogitError::FeatureLength { len, n_features: p });
    }
    let rows = len / p;
    let starts = design.set_starts;
    if starts.len() < 2 {
        return Err(ConditionalLogitError::NoSets);
    }
    let n_sets = starts.len() - 1;
    if starts[0] != 0 {
        return Err(ConditionalLogitError::SetStartsNotZero { first: starts[0] });
    }
    for s in 0..n_sets {
        let (a, b) = (starts[s], starts[s + 1]);
        if b < a.saturating_add(2) {
            return Err(ConditionalLogitError::SetTooSmall {
                set: s,
                options: b.saturating_sub(a),
            });
        }
    }
    if starts[n_sets] != rows {
        return Err(ConditionalLogitError::SetStartsRowMismatch { last: starts[n_sets], rows });
    }
    if design.chosen.len() != n_sets {
        return Err(ConditionalLogitError::ChosenLength { len: design.chosen.len(), n_sets });
    }
    for s in 0..n_sets {
        let options = starts[s + 1] - starts[s];
        if design.chosen[s] >= options {
            return Err(ConditionalLogitError::ChosenOutOfRange {
                set: s,
                index: design.chosen[s],
                options,
            });
        }
    }
    if let Some(clusters) = design.clusters
        && clusters.len() != n_sets
    {
        return Err(ConditionalLogitError::ClusterLength { len: clusters.len(), n_sets });
    }
    if let Some(i) = design.features.iter().position(|v| !v.is_finite()) {
        return Err(ConditionalLogitError::NonFiniteFeature { row: i / p, column: i % p });
    }
    let x = design.features;
    let mut diffs = vec![0.0; len];
    for s in 0..n_sets {
        let c = starts[s] + design.chosen[s];
        for r in starts[s]..starts[s + 1] {
            for k in 0..p {
                let d = x[r * p + k] - x[c * p + k];
                if !d.is_finite() {
                    return Err(ConditionalLogitError::NonFiniteFeature { row: r, column: k });
                }
                diffs[r * p + k] = d;
            }
        }
    }
    Ok(Prepared {
        p,
        n_sets,
        set_starts: starts,
        chosen: design.chosen,
        clusters: design.clusters,
        diffs,
    })
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn euclidean(a: &[f64]) -> f64 {
    a.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn max_abs(a: &[f64]) -> f64 {
    a.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}

/// Columns of the within-set centred design that lie in the span of earlier columns.
///
/// Gram-Schmidt in column order with one re-orthogonalisation pass: column `k` is dependent when
/// its residual after projecting out the accepted earlier columns has Euclidean norm at most
/// [`RANK_TOLERANCE`] times its own norm, or when it is zero. Centring on the chosen option and
/// centring on the set mean give the same null space, since both vanish exactly on directions
/// along which every option of every set has the same linear predictor.
fn dependent_columns(prep: &Prepared<'_>) -> Vec<usize> {
    let p = prep.p;
    let rows = prep.diffs.len() / p;
    let mut basis: Vec<Vec<f64>> = Vec::new();
    let mut dependent = Vec::new();
    for k in 0..p {
        let mut v: Vec<f64> = (0..rows).map(|r| prep.diffs[r * p + k]).collect();
        let original = euclidean(&v);
        if original <= 0.0 {
            dependent.push(k);
            continue;
        }
        for _ in 0..2 {
            for q in &basis {
                let proj = dot(q, &v);
                for (vi, qi) in v.iter_mut().zip(q) {
                    *vi -= proj * qi;
                }
            }
        }
        let residual = euclidean(&v);
        if residual <= RANK_TOLERANCE * original {
            dependent.push(k);
        } else {
            for vi in &mut v {
                *vi /= residual;
            }
            basis.push(v);
        }
    }
    dependent
}

/// Log-likelihood and derivatives at one coefficient vector.
struct Evaluation {
    log_likelihood: f64,
    gradient: Vec<f64>,
    information: Vec<f64>,
    /// `max_{s, j} |z_j . beta|`.
    largest_gap: f64,
    /// Per-set score vectors, row-major `n_sets x p`, when requested.
    set_scores: Option<Vec<f64>>,
}

/// Log-sum-exp evaluation per set on the differences `z_j`, so the chosen option has linear
/// predictor exactly zero: `l_s = -ln sum_j exp(z_j . beta)`, score `-zbar_s` with
/// `zbar_s = sum_j p_j z_j`, information `sum_j p_j (z_j - zbar_s)(z_j - zbar_s)'`.
fn evaluate(prep: &Prepared<'_>, beta: &[f64], derivatives: bool, keep_scores: bool) -> Evaluation {
    let p = prep.p;
    let mut log_likelihood = 0.0;
    let mut gradient = vec![0.0; if derivatives { p } else { 0 }];
    let mut information = vec![0.0; if derivatives { p * p } else { 0 }];
    let mut set_scores = keep_scores.then(|| vec![0.0; prep.n_sets * p]);
    let mut largest_gap = 0.0_f64;
    let mut weights = Vec::new();
    let mut zbar = vec![0.0; p];
    let mut centred = vec![0.0; p];
    for s in 0..prep.n_sets {
        let rows = prep.set_rows(s);
        weights.clear();
        weights.extend(rows.clone().map(|r| dot(prep.row(r), beta)));
        let top = weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        largest_gap = largest_gap.max(max_abs(&weights));
        let mut total = 0.0;
        for w in &mut weights {
            *w = (*w - top).exp();
            total += *w;
        }
        log_likelihood -= top + total.ln();
        if !derivatives {
            continue;
        }
        zbar.fill(0.0);
        for (w, r) in weights.iter_mut().zip(rows.clone()) {
            *w /= total;
            for (zb, z) in zbar.iter_mut().zip(prep.row(r)) {
                *zb += *w * z;
            }
        }
        for (g, zb) in gradient.iter_mut().zip(&zbar) {
            *g -= zb;
        }
        if let Some(scores) = set_scores.as_mut() {
            for (sc, zb) in scores[s * p..(s + 1) * p].iter_mut().zip(&zbar) {
                *sc = -zb;
            }
        }
        for (w, r) in weights.iter().zip(rows) {
            for ((c, z), zb) in centred.iter_mut().zip(prep.row(r)).zip(&zbar) {
                *c = z - zb;
            }
            for i in 0..p {
                let wi = w * centred[i];
                for j in i..p {
                    information[i * p + j] += wi * centred[j];
                }
            }
        }
    }
    if derivatives {
        for i in 0..p {
            for j in 0..i {
                information[i * p + j] = information[j * p + i];
            }
        }
    }
    Evaluation { log_likelihood, gradient, information, largest_gap, set_scores }
}

/// `sum_s max_j |z_j|_inf`: a bound on `|gradient|_inf` at any coefficients, used to scale the
/// convergence threshold so that it does not depend on feature units or the number of sets.
fn gradient_scale(prep: &Prepared<'_>) -> f64 {
    (0..prep.n_sets)
        .map(|s| prep.set_rows(s).fold(0.0_f64, |m, r| m.max(max_abs(prep.row(r)))))
        .sum()
}

/// The unit vector along `v` when it certifies separation, `None` otherwise.
///
/// With `d = v / |v|`, separation holds when every non-chosen option `j` of every set satisfies
/// `z_j . d <= SEPARATION_TIE |z_j|` (the chosen option is weakly preferred) and at least one
/// satisfies `z_j . d < -SEPARATION_STRICT |z_j|` (strictly preferred). Options identical to the
/// chosen one (`z_j = 0`) are ties in every direction and are skipped.
fn separation_direction(prep: &Prepared<'_>, v: &[f64]) -> Option<Vec<f64>> {
    let norm = euclidean(v);
    if !(norm.is_finite() && norm > 0.0) {
        return None;
    }
    let d: Vec<f64> = v.iter().map(|x| x / norm).collect();
    let mut strict = false;
    for s in 0..prep.n_sets {
        let chosen_row = prep.set_starts[s] + prep.chosen[s];
        for r in prep.set_rows(s) {
            if r == chosen_row {
                continue;
            }
            let z = prep.row(r);
            let size = euclidean(z);
            if size <= 0.0 {
                continue;
            }
            let along = dot(z, &d);
            if along > SEPARATION_TIE * size {
                return None;
            }
            if along < -SEPARATION_STRICT * size {
                strict = true;
            }
        }
    }
    strict.then_some(d)
}

fn symmetrise(m: &mut [f64], p: usize) {
    for i in 0..p {
        for j in 0..i {
            let mean = 0.5 * (m[i * p + j] + m[j * p + i]);
            m[i * p + j] = mean;
            m[j * p + i] = mean;
        }
    }
}

/// `A B` for row-major `p x p` matrices.
fn matmul(a: &[f64], b: &[f64], p: usize) -> Vec<f64> {
    let mut out = vec![0.0; p * p];
    for i in 0..p {
        for k in 0..p {
            let aik = a[i * p + k];
            for j in 0..p {
                out[i * p + j] += aik * b[k * p + j];
            }
        }
    }
    out
}

/// `H^-1 (sum_c s_c s_c') H^-1`, with `s_c` the sum of the per-set scores of cluster `c`.
/// Clusters are visited in ascending label order.
fn cluster_sandwich(
    covariance: &[f64],
    set_scores: &[f64],
    clusters: &[u32],
    p: usize,
) -> Vec<f64> {
    let mut sums: BTreeMap<u32, Vec<f64>> = BTreeMap::new();
    for (s, &label) in clusters.iter().enumerate() {
        let entry = sums.entry(label).or_insert_with(|| vec![0.0; p]);
        for (e, v) in entry.iter_mut().zip(&set_scores[s * p..(s + 1) * p]) {
            *e += v;
        }
    }
    let mut meat = vec![0.0; p * p];
    for score in sums.values() {
        for i in 0..p {
            for j in 0..p {
                meat[i * p + j] += score[i] * score[j];
            }
        }
    }
    let mut sandwich = matmul(&matmul(covariance, &meat, p), covariance, p);
    symmetrise(&mut sandwich, p);
    sandwich
}

/// Fit a conditional logit by Newton's method on the concave log-likelihood.
///
/// Starting from `beta = 0`, each iteration solves `I(beta) step = g(beta)` (observed information
/// and gradient, both from a log-sum-exp per set) by Cholesky and halves the step until the
/// log-likelihood does not decrease beyond its rounding allowance `64 eps (|l| + n_sets)`. The fit
/// has converged when `max_k |g_k| <= gradient_tolerance * sum_s max_j |z_j|_inf`, where
/// `z_j = x_j - x_chosen`; the right-hand side bounds `|g|_inf` at every `beta`, so the test does
/// not depend on feature units.
///
/// Before iterating, the within-set centred design is checked for full column rank (Gram-Schmidt
/// in column order, relative tolerance `1e-9`).
///
/// Separation. The likelihood has no finite maximum exactly when some direction `d` has
/// `(x_chosen - x_j) . d >= 0` for every set and every non-chosen option `j`, with strict
/// inequality somewhere; along `d` the log-likelihood rises towards its supremum (zero when every
/// inequality is strict) while the coefficients diverge. A unit direction `d` is accepted as such
/// a certificate when `z_j . d <= 1e-8 |z_j|` for every non-chosen option and
/// `z_j . d < -1e-6 |z_j|` for at least one. Two candidates are tested: the direction of every
/// accepted Newton step (which aligns with `d` while the remaining coefficients settle), and,
/// once the largest within-set linear-predictor difference `max |z_j . beta|` exceeds
/// `separation_bound`, the direction of `beta` itself. A Cholesky failure is also tested against
/// the direction of `beta`.
///
/// `covariance` is the inverse observed information at the maximum. `cluster_covariance` is the
/// sandwich `I^-1 (sum_c s_c s_c') I^-1` with `s_c` the sum of the per-set score vectors of
/// cluster `c`; no small-sample multiplier (such as `C / (C - 1)`) is applied.
///
/// # Errors
///
/// Every shape refusal of [`ConditionalLogitError`]; [`ConditionalLogitError::InvalidOptions`];
/// [`ConditionalLogitError::RankDeficient`]; [`ConditionalLogitError::Separated`];
/// [`ConditionalLogitError::SingularInformation`]; [`ConditionalLogitError::NotConverged`].
pub fn fit_conditional_logit(
    design: &ConditionalLogitDesign<'_>,
    options: ConditionalLogitOptions,
) -> Result<ConditionalLogitFit, ConditionalLogitError> {
    if !(options.gradient_tolerance.is_finite()
        && options.gradient_tolerance > 0.0
        && options.separation_bound.is_finite()
        && options.separation_bound > 0.0)
    {
        return Err(ConditionalLogitError::InvalidOptions);
    }
    let prep = prepare(design)?;
    let columns = dependent_columns(&prep);
    if !columns.is_empty() {
        return Err(ConditionalLogitError::RankDeficient { columns });
    }
    let p = prep.p;
    let threshold = options.gradient_tolerance * gradient_scale(&prep);
    let mut beta = vec![0.0; p];
    let mut current = evaluate(&prep, &beta, true, false);
    let null_log_likelihood = current.log_likelihood;
    let mut iterations = 0;
    loop {
        let gradient_norm = max_abs(&current.gradient);
        if gradient_norm <= threshold {
            break;
        }
        if iterations >= options.max_iterations {
            return Err(ConditionalLogitError::NotConverged { iterations, gradient_norm });
        }
        let Some(step) = solve_spd(&current.information, p, &current.gradient) else {
            return Err(separation_direction(&prep, &beta)
                .map_or(ConditionalLogitError::SingularInformation, |direction| {
                    ConditionalLogitError::Separated { direction }
                }));
        };
        let allowance = 64.0 * f64::EPSILON * (current.log_likelihood.abs() + prep.n_sets as f64);
        let floor = current.log_likelihood - allowance;
        let mut scale = 1.0;
        let mut accepted = None;
        for _ in 0..=MAX_HALVINGS {
            let candidate: Vec<f64> = beta.iter().zip(&step).map(|(b, d)| b + scale * d).collect();
            let value = evaluate(&prep, &candidate, false, false).log_likelihood;
            if value >= floor {
                accepted = Some(candidate);
                break;
            }
            scale *= 0.5;
        }
        let Some(next) = accepted else {
            return Err(ConditionalLogitError::NotConverged { iterations, gradient_norm });
        };
        beta = next;
        iterations += 1;
        current = evaluate(&prep, &beta, true, false);
        if let Some(direction) = separation_direction(&prep, &step) {
            return Err(ConditionalLogitError::Separated { direction });
        }
        if current.largest_gap > options.separation_bound
            && let Some(direction) = separation_direction(&prep, &beta)
        {
            return Err(ConditionalLogitError::Separated { direction });
        }
    }
    let mut covariance =
        invert_square(&current.information, p).ok_or(ConditionalLogitError::SingularInformation)?;
    symmetrise(&mut covariance, p);
    let cluster_covariance = prep.clusters.map(|clusters| {
        let scores = evaluate(&prep, &beta, true, true).set_scores.unwrap_or_default();
        cluster_sandwich(&covariance, &scores, clusters, p)
    });
    Ok(ConditionalLogitFit {
        gradient_norm: max_abs(&current.gradient),
        coefficients: beta,
        log_likelihood: current.log_likelihood,
        null_log_likelihood,
        information: current.information,
        covariance,
        cluster_covariance,
        iterations,
    })
}

/// Conditional logit log-likelihood `sum_s (x_chosen . beta - ln sum_{j in s} exp(x_j . beta))`.
///
/// Evaluated with a log-sum-exp on the within-set differences, as in [`fit_conditional_logit`].
/// No rank or separation check is made.
///
/// # Errors
///
/// Every shape refusal of [`ConditionalLogitError`],
/// [`ConditionalLogitError::CoefficientLength`] and
/// [`ConditionalLogitError::NonFiniteCoefficient`].
pub fn conditional_logit_log_likelihood(
    design: &ConditionalLogitDesign<'_>,
    coefficients: &[f64],
) -> Result<f64, ConditionalLogitError> {
    let prep = prepare(design)?;
    if coefficients.len() != prep.p {
        return Err(ConditionalLogitError::CoefficientLength {
            len: coefficients.len(),
            n_features: prep.p,
        });
    }
    if let Some(index) = coefficients.iter().position(|v| !v.is_finite()) {
        return Err(ConditionalLogitError::NonFiniteCoefficient { index });
    }
    Ok(evaluate(&prep, coefficients, false, false).log_likelihood)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prepared<'a>(
        features: &'a [f64],
        p: usize,
        starts: &'a [usize],
        chosen: &'a [usize],
    ) -> Prepared<'a> {
        let design = ConditionalLogitDesign {
            features,
            n_features: p,
            set_starts: starts,
            chosen,
            clusters: None,
        };
        prepare(&design).unwrap_or_else(|e| unreachable!("valid fixture refused: {e}"))
    }

    #[test]
    fn dependent_columns_names_later_copies_and_set_constants() {
        // Column 1 is twice column 0; column 2 is constant within each set; column 3 is free.
        let features = [
            0.0, 0.0, 5.0, 1.0, //
            1.0, 2.0, 5.0, 0.0, //
            2.0, 4.0, -1.0, 3.0, //
            0.5, 1.0, -1.0, 1.0, //
            3.0, 6.0, -1.0, 2.0,
        ];
        let starts = [0, 2, 5];
        let chosen = [0, 1];
        let prep = prepared(&features, 4, &starts, &chosen);
        assert_eq!(dependent_columns(&prep), vec![1, 2]);
    }

    #[test]
    fn separation_certificate_needs_a_strict_preference() {
        let features = [0.0, 1.0, 0.0, 2.0];
        let starts = [0, 2, 4];
        let chosen = [1, 1];
        let prep = prepared(&features, 1, &starts, &chosen);
        assert_eq!(separation_direction(&prep, &[3.0]), Some(vec![1.0]));
        assert_eq!(separation_direction(&prep, &[-1.0]), None);
        assert_eq!(separation_direction(&prep, &[0.0]), None);
    }

    #[test]
    fn evaluation_information_is_symmetric() {
        let features = [0.0, 1.0, 1.0, 0.5, -1.0, 2.0, 0.3, 0.3, 1.0, -0.2];
        let starts = [0, 3, 5];
        let chosen = [2, 0];
        let prep = prepared(&features, 2, &starts, &chosen);
        let e = evaluate(&prep, &[0.4, -0.7], true, true);
        assert!((e.information[1] - e.information[2]).abs() <= 1e-15);
        let scores = e.set_scores.unwrap_or_default();
        let summed = [scores[0] + scores[2], scores[1] + scores[3]];
        assert!((summed[0] - e.gradient[0]).abs() <= 1e-15);
        assert!((summed[1] - e.gradient[1]).abs() <= 1e-15);
    }
}
