//! Power-prior discount `α` chosen from how well the source predicts the target.
//!
//! A power prior discounts a source by an exponent `α ∈ [0, 1]`: the source's
//! contribution to the prior is raised to the power `α`, so `α = 0` ignores the
//! source and `α = 1` takes it at face value. A hand-set `α` is an assumption.
//! This module chooses `α` from the target data instead, with one of two rules:
//!
//! * [`AlphaCriterion::EmpiricalBayes`]: the `α` on the grid that maximizes the
//!   marginal likelihood `p(target data | α)` (type-II maximum likelihood).
//! * [`AlphaCriterion::CrossValidated`]: the `α` that maximizes the `K`-fold
//!   log predictive density of held-out target rows. For a conjugate model the
//!   held-out block density is `ML(all rows) / ML(training rows)` (up to a
//!   constant that does not depend on `α`), so no fold is ever fitted by sampling.
//!
//! Three conjugate target models are provided: a Gaussian linear model whose
//! prior is composed from external sources by
//! [`crate::external_prior::compose_external_priors_with_alphas`] (known or
//! Inv-Gamma residual variance), a Beta-Binomial proportion with source counts,
//! and a Gamma-Poisson rate with source counts.
//!
//! ## The interval accounts for the selection
//!
//! A posterior interval computed at the selected `α̂` treats `α̂` as if it had
//! been fixed in advance. `α̂` is a function of the data, so that interval
//! understates uncertainty whenever the data cannot pin `α` down. The reported
//! interval is instead the equal-tailed interval of the **select-then-estimate
//! procedure**: for each of `B` bootstrap resamples of the target rows the whole
//! selection is rerun, `α̂_b` is read off, and the posterior of the target
//! functional given the *original* data at `α̂_b` is one component of an
//! equal-weight mixture. The interval is the mixture's quantiles, so the spread
//! of `α̂` across resamples widens it. That construction is a procedure, not a
//! guarantee: no data-driven discount covers uniformly across every
//! source/target discrepancy, and the coverage of this one is measured, by
//! repeated sampling, over a stated range of discrepancies.
//!
//! The report also carries the conditional interval at `α̂` and a sensitivity
//! table: the criterion's score, the estimate and the conditional interval at
//! every grid `α`. A prior is an assumption and never upgrades identification.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::needless_range_loop,
    reason = "dense small-matrix and index arithmetic over bounded grids and row counts"
)]

use antecedent_core::{CausalRng, derive_stream};
use antecedent_kernels::norm_inv;
use antecedent_stats::{ln_gamma, student_t_ppf, student_t_sf};

use crate::conjugate_counts::{BetaBinomialPosterior, GammaPoissonPosterior};
use crate::conjugate_moment_match::{BetaHyperparameters, GammaHyperparameters};
use crate::error::ProbError;
use crate::external_prior::{ExternalPriorSource, compose_external_priors_with_alphas};
use crate::linalg::{cholesky_spd, solve_chol_into};
use crate::prior::{GaussianVarianceModel, PriorSet};

/// Fewest bootstrap resamples a selection-aware interval accepts.
pub const MIN_SELECTION_BOOTSTRAP: u32 = 50;
/// Largest number of bootstrap resamples.
pub const MAX_SELECTION_BOOTSTRAP: u32 = 20_000;
/// Largest number of grid points.
pub const MAX_ALPHA_GRID: usize = 1_000;
/// Relative margin by which a larger `α` must beat the best score so far.
const TIE_MARGIN: f64 = 1e-12;
/// Stream used to assign rows to cross-validation folds.
const FOLD_STREAM: u64 = u64::MAX;

/// How `α` is chosen from the target data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AlphaCriterion {
    /// Maximize the marginal likelihood of the target data over the grid.
    EmpiricalBayes,
    /// Maximize the `folds`-fold log predictive density of held-out target rows.
    CrossValidated {
        /// Number of folds, `2 ..= rows`.
        folds: usize,
    },
}

impl AlphaCriterion {
    /// Stable name of the criterion.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::EmpiricalBayes => "empirical_bayes",
            Self::CrossValidated { .. } => "cross_validated",
        }
    }
}

/// The `α` grid: 21 equally spaced points on `[0, 1]`.
#[must_use]
pub fn default_alpha_grid() -> Vec<f64> {
    (0..=20).map(|i| f64::from(i) / 20.0).collect()
}

/// Options of a select-then-estimate run.
#[derive(Clone, Debug, PartialEq)]
pub struct AlphaSelectionOptions {
    /// Selection rule.
    pub criterion: AlphaCriterion,
    /// Strictly increasing `α` values in `[0, 1]`.
    pub grid: Vec<f64>,
    /// Credible level of the reported intervals, in `(0, 1)`.
    pub level: f64,
    /// Bootstrap resamples of the whole selection, at least [`MIN_SELECTION_BOOTSTRAP`].
    pub n_bootstrap: u32,
    /// Seed of the resampling and of the fold assignment.
    pub seed: u64,
}

impl AlphaSelectionOptions {
    /// Defaults: [`default_alpha_grid`], level 0.95, 200 resamples, seed 0.
    #[must_use]
    pub fn new(criterion: AlphaCriterion) -> Self {
        Self { criterion, grid: default_alpha_grid(), level: 0.95, n_bootstrap: 200, seed: 0 }
    }

    fn validate(&self, n_rows: usize) -> Result<(), ProbError> {
        if self.grid.is_empty() || self.grid.len() > MAX_ALPHA_GRID {
            return Err(ProbError::InvalidPrior {
                message: "alpha grid must hold 1..=1000 points",
            });
        }
        let mut previous = f64::NEG_INFINITY;
        for &a in &self.grid {
            if !a.is_finite() || !(0.0..=1.0).contains(&a) || a <= previous {
                return Err(ProbError::InvalidPrior {
                    message: "alpha grid must be strictly increasing and inside [0, 1]",
                });
            }
            previous = a;
        }
        if !(self.level.is_finite() && self.level > 0.0 && self.level < 1.0) {
            return Err(ProbError::InvalidPrior {
                message: "credible level must be finite and in (0, 1)",
            });
        }
        if !(MIN_SELECTION_BOOTSTRAP..=MAX_SELECTION_BOOTSTRAP).contains(&self.n_bootstrap) {
            return Err(ProbError::InvalidPrior {
                message: "a selection-aware interval needs 50..=20000 bootstrap resamples",
            });
        }
        if let AlphaCriterion::CrossValidated { folds } = self.criterion
            && (folds < 2 || folds > n_rows)
        {
            return Err(ProbError::InvalidPrior {
                message: "cross-validation folds must be in 2..=rows",
            });
        }
        Ok(())
    }
}

/// One row of the sensitivity table: the target estimate at a fixed `α`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AlphaSensitivityRow {
    /// Discount.
    pub alpha: f64,
    /// Criterion score on the original data (log marginal likelihood, or `K`-fold
    /// log predictive density).
    pub score: f64,
    /// Posterior mean of the target functional at this `α`.
    pub estimate: f64,
    /// Lower end of the equal-tailed credible interval at this `α`.
    pub lower: f64,
    /// Upper end of the equal-tailed credible interval at this `α`.
    pub upper: f64,
}

/// Result of choosing `α` and estimating the target functional.
#[derive(Clone, Debug, PartialEq)]
pub struct SelectedAlphaReport {
    /// Rule that chose `α`.
    pub criterion: AlphaCriterion,
    /// Chosen `α`.
    pub selected_alpha: f64,
    /// Index of the chosen `α` in [`Self::sensitivity`].
    pub selected_index: usize,
    /// Posterior mean of the functional at the chosen `α`.
    pub estimate: f64,
    /// Credible level of both intervals.
    pub level: f64,
    /// Equal-tailed interval at the chosen `α`, treating it as fixed in advance.
    pub conditional_interval: (f64, f64),
    /// Interval of the whole select-then-estimate procedure (see the module docs).
    pub interval: (f64, f64),
    /// Estimate and conditional interval at every grid `α`.
    pub sensitivity: Vec<AlphaSensitivityRow>,
    /// Number of bootstrap resamples that chose each grid point.
    pub bootstrap_alpha_counts: Vec<u32>,
    /// Number of bootstrap resamples.
    pub n_bootstrap: u32,
}

/// Posterior law of the target functional at one `α`.
#[derive(Clone, Copy, Debug)]
enum Component {
    Normal { loc: f64, scale: f64 },
    StudentT { loc: f64, scale: f64, df: f64 },
    Beta(BetaHyperparameters),
    Gamma(GammaHyperparameters),
}

impl Component {
    fn mean(self) -> f64 {
        match self {
            Self::Normal { loc, .. } | Self::StudentT { loc, .. } => loc,
            Self::Beta(p) => p.mean(),
            Self::Gamma(p) => p.mean(),
        }
    }

    fn cdf(self, x: f64) -> f64 {
        match self {
            Self::Normal { loc, scale } => antecedent_kernels::norm_cdf((x - loc) / scale),
            Self::StudentT { loc, scale, df } => student_t_sf(-(x - loc) / scale, df),
            Self::Beta(p) => BetaBinomialPosterior::new(p, 0, 0).map_or(f64::NAN, |b| b.cdf(x)),
            Self::Gamma(p) => {
                GammaPoissonPosterior::from_totals(p, 0, 0.0).map_or(f64::NAN, |g| g.cdf(x))
            }
        }
    }

    fn quantile(self, p: f64) -> Result<f64, ProbError> {
        Ok(match self {
            Self::Normal { loc, scale } => loc + scale * norm_inv(p),
            Self::StudentT { loc, scale, df } => loc + scale * student_t_ppf(p, df),
            Self::Beta(h) => BetaBinomialPosterior::new(h, 0, 0)?.quantile(p)?,
            Self::Gamma(h) => GammaPoissonPosterior::from_totals(h, 0, 0.0)?.quantile(p)?,
        })
    }

    fn interval(self, level: f64) -> Result<(f64, f64), ProbError> {
        let tail = 0.5 * (1.0 - level);
        Ok((self.quantile(tail)?, self.quantile(1.0 - tail)?))
    }
}

/// Equal-weight mixture `Σ_g w_g F_g` quantile by bisection.
fn mixture_interval(parts: &[(Component, f64)], level: f64) -> Result<(f64, f64), ProbError> {
    let total: f64 = parts.iter().map(|&(_, w)| w).sum();
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &(c, w) in parts {
        if w > 0.0 {
            lo = lo.min(c.quantile(1e-10)?);
            hi = hi.max(c.quantile(1.0 - 1e-10)?);
        }
    }
    let cdf = |x: f64| parts.iter().map(|&(c, w)| w * c.cdf(x)).sum::<f64>() / total;
    let invert = |p: f64| {
        let (mut a, mut b) = (lo, hi);
        for _ in 0..200 {
            let mid = 0.5 * (a + b);
            if mid <= a || mid >= b {
                break;
            }
            if cdf(mid) < p {
                a = mid;
            } else {
                b = mid;
            }
        }
        0.5 * (a + b)
    };
    let tail = 0.5 * (1.0 - level);
    Ok((invert(tail), invert(1.0 - tail)))
}

/// A conjugate target model scored over the `α` grid.
trait AlphaModel {
    /// Number of target rows.
    fn n_rows(&self) -> usize;
    /// Log marginal likelihood at grid point `i` of the target rows `rows`
    /// (indices into the original rows, repeats allowed).
    fn log_marginal(&self, i: usize, rows: &[usize]) -> Result<f64, ProbError>;
    /// Posterior of the functional at grid point `i` given all original rows.
    fn component(&self, i: usize) -> Result<Component, ProbError>;
}

/// Score of every grid point on `rows`: `ML` for empirical Bayes, the summed held-out
/// block log density for cross-validation. `fold_of` maps an original row to its fold.
fn scores_on(
    model: &dyn AlphaModel,
    grid_len: usize,
    criterion: AlphaCriterion,
    rows: &[usize],
    fold_of: &[usize],
) -> Result<Vec<f64>, ProbError> {
    let mut scores = Vec::with_capacity(grid_len);
    for i in 0..grid_len {
        let whole = model.log_marginal(i, rows)?;
        let score = match criterion {
            AlphaCriterion::EmpiricalBayes => whole,
            AlphaCriterion::CrossValidated { folds } => {
                let mut total = 0.0;
                for fold in 0..folds {
                    let train: Vec<usize> =
                        rows.iter().copied().filter(|&r| fold_of[r] != fold).collect();
                    if train.len() == rows.len() {
                        continue;
                    }
                    total += whole - model.log_marginal(i, &train)?;
                }
                total
            }
        };
        if score.is_nan() {
            return Err(ProbError::Numerical { message: "alpha score is not a number".into() });
        }
        scores.push(score);
    }
    Ok(scores)
}

/// Index of the best score; a larger `α` must beat the best so far by a relative margin.
fn best_index(scores: &[f64]) -> usize {
    let mut best = 0;
    for i in 1..scores.len() {
        if scores[i] > scores[best] + TIE_MARGIN * scores[best].abs().max(1.0) {
            best = i;
        }
    }
    best
}

/// Assign every original row to a fold by a seeded shuffle.
fn assign_folds(n: usize, folds: usize, seed: u64) -> Vec<usize> {
    let mut rng = CausalRng::from_seed(derive_stream(seed, FOLD_STREAM));
    let mut order: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = (rng.next_f64() * (i + 1) as f64) as usize;
        order.swap(i, j.min(i));
    }
    let mut fold_of = vec![0; n];
    for (rank, &row) in order.iter().enumerate() {
        fold_of[row] = rank % folds.max(1);
    }
    fold_of
}

fn select_and_estimate(
    model: &dyn AlphaModel,
    options: &AlphaSelectionOptions,
) -> Result<SelectedAlphaReport, ProbError> {
    let n = model.n_rows();
    options.validate(n)?;
    let grid = &options.grid;
    let folds = match options.criterion {
        AlphaCriterion::CrossValidated { folds } => folds,
        AlphaCriterion::EmpiricalBayes => 1,
    };
    let fold_of = assign_folds(n, folds, options.seed);
    let original: Vec<usize> = (0..n).collect();
    let scores = scores_on(model, grid.len(), options.criterion, &original, &fold_of)?;
    let selected_index = best_index(&scores);

    let mut components = Vec::with_capacity(grid.len());
    let mut sensitivity = Vec::with_capacity(grid.len());
    for (i, &alpha) in grid.iter().enumerate() {
        let component = model.component(i)?;
        let (lower, upper) = component.interval(options.level)?;
        sensitivity.push(AlphaSensitivityRow {
            alpha,
            score: scores[i],
            estimate: component.mean(),
            lower,
            upper,
        });
        components.push(component);
    }

    let mut counts = vec![0u32; grid.len()];
    let mut rows = vec![0usize; n];
    for b in 0..options.n_bootstrap {
        let mut rng = CausalRng::from_seed(derive_stream(options.seed, u64::from(b)));
        for slot in &mut rows {
            *slot = ((rng.next_f64() * n as f64) as usize).min(n - 1);
        }
        let resampled = scores_on(model, grid.len(), options.criterion, &rows, &fold_of)?;
        counts[best_index(&resampled)] += 1;
    }
    let parts: Vec<(Component, f64)> = components
        .iter()
        .zip(&counts)
        .filter(|&(_, &c)| c > 0)
        .map(|(&component, &c)| (component, f64::from(c)))
        .collect();
    let interval = mixture_interval(&parts, options.level)?;

    let row = sensitivity[selected_index];
    Ok(SelectedAlphaReport {
        criterion: options.criterion,
        selected_alpha: row.alpha,
        selected_index,
        estimate: row.estimate,
        level: options.level,
        conditional_interval: (row.lower, row.upper),
        interval,
        sensitivity,
        bootstrap_alpha_counts: counts,
        n_bootstrap: options.n_bootstrap,
    })
}

// -- Gaussian linear target -------------------------------------------------------------------

/// A Gaussian linear target: rows of `x` (row-major `n_rows × n_cols`), outcomes `y`, and
/// the linear functional `c'β` whose posterior is reported.
#[derive(Clone, Copy, Debug)]
pub struct LinearTarget<'a> {
    /// Design, row-major.
    pub x: &'a [f64],
    /// Outcomes.
    pub y: &'a [f64],
    /// Number of rows.
    pub n_rows: usize,
    /// Number of coefficients.
    pub n_cols: usize,
    /// Functional weights `c` (length `n_cols`), e.g. a unit vector for one coefficient.
    pub functional: &'a [f64],
}

enum Residual {
    Known { sigma2: f64 },
    InvGamma { shape: f64, scale: f64 },
}

/// Composed conjugate prior at one grid `α`.
struct ConjugatePrior {
    mean: Vec<f64>,
    precision: Vec<f64>,
    log_det_precision: f64,
    residual: Residual,
}

struct Fit {
    log_ml: f64,
    mean: Vec<f64>,
    chol: Vec<f64>,
    quad: f64,
}

struct LinearModel<'a> {
    target: LinearTarget<'a>,
    priors: Vec<ConjugatePrior>,
}

fn log_det_from_chol(chol: &[f64], n: usize) -> f64 {
    (0..n).map(|i| 2.0 * chol[i * n + i].ln()).sum()
}

impl LinearModel<'_> {
    fn fit(&self, i: usize, rows: &[usize]) -> Result<Fit, ProbError> {
        let prior = &self.priors[i];
        let p = self.target.n_cols;
        let mut lam = prior.precision.clone();
        let mut rhs = vec![0.0; p];
        for r in 0..p {
            for c in 0..p {
                rhs[r] += prior.precision[r * p + c] * prior.mean[c];
            }
        }
        for &row in rows {
            let xi = &self.target.x[row * p..(row + 1) * p];
            let yi = self.target.y[row];
            for r in 0..p {
                rhs[r] += xi[r] * yi;
                for c in 0..p {
                    lam[r * p + c] += xi[r] * xi[c];
                }
            }
        }
        let chol = cholesky_spd(&lam, p)?;
        let mut scratch = vec![0.0; p];
        let mut mean = vec![0.0; p];
        solve_chol_into(&chol, p, &rhs, &mut scratch, &mut mean);

        let mut resid = 0.0;
        for &row in rows {
            let xi = &self.target.x[row * p..(row + 1) * p];
            let fitted: f64 = xi.iter().zip(&mean).map(|(a, b)| a * b).sum();
            let e = self.target.y[row] - fitted;
            resid += e * e;
        }
        let mut quad = 0.0;
        for r in 0..p {
            for c in 0..p {
                quad += (mean[r] - prior.mean[r])
                    * prior.precision[r * p + c]
                    * (mean[c] - prior.mean[c]);
            }
        }
        let q = resid + quad;
        let n = rows.len() as f64;
        let half_log_ratio = 0.5 * (prior.log_det_precision - log_det_from_chol(&chol, p));
        let log_ml = match prior.residual {
            Residual::Known { sigma2 } => {
                -0.5 * n * (2.0 * std::f64::consts::PI * sigma2).ln() + half_log_ratio
                    - q / (2.0 * sigma2)
            }
            Residual::InvGamma { shape, scale } => {
                let shape_n = shape + 0.5 * n;
                let scale_n = scale + 0.5 * q;
                -0.5 * n * (2.0 * std::f64::consts::PI).ln() + half_log_ratio + shape * scale.ln()
                    - shape_n * scale_n.ln()
                    + ln_gamma(shape_n)
                    - ln_gamma(shape)
            }
        };
        Ok(Fit { log_ml, mean, chol, quad })
    }
}

impl AlphaModel for LinearModel<'_> {
    fn n_rows(&self) -> usize {
        self.target.n_rows
    }

    fn log_marginal(&self, i: usize, rows: &[usize]) -> Result<f64, ProbError> {
        Ok(self.fit(i, rows)?.log_ml)
    }

    fn component(&self, i: usize) -> Result<Component, ProbError> {
        let all: Vec<usize> = (0..self.target.n_rows).collect();
        let fit = self.fit(i, &all)?;
        let p = self.target.n_cols;
        let c = self.target.functional;
        let loc: f64 = c.iter().zip(&fit.mean).map(|(a, b)| a * b).sum();
        let mut scratch = vec![0.0; p];
        let mut z = vec![0.0; p];
        solve_chol_into(&fit.chol, p, c, &mut scratch, &mut z);
        let v: f64 = c.iter().zip(&z).map(|(a, b)| a * b).sum();
        let n = self.target.n_rows as f64;
        match self.priors[i].residual {
            Residual::Known { sigma2 } => Ok(Component::Normal { loc, scale: (sigma2 * v).sqrt() }),
            Residual::InvGamma { shape, scale } => {
                let resid = {
                    let mut s = 0.0;
                    for row in 0..self.target.n_rows {
                        let xi = &self.target.x[row * p..(row + 1) * p];
                        let fitted: f64 = xi.iter().zip(&fit.mean).map(|(a, b)| a * b).sum();
                        let e = self.target.y[row] - fitted;
                        s += e * e;
                    }
                    s
                };
                let shape_n = shape + 0.5 * n;
                let scale_n = scale + 0.5 * (resid + fit.quad);
                Ok(Component::StudentT {
                    loc,
                    scale: (scale_n / shape_n * v).sqrt(),
                    df: 2.0 * shape_n,
                })
            }
        }
    }
}

fn conjugate_prior_from(prior: &PriorSet, n_cols: usize) -> Result<ConjugatePrior, ProbError> {
    let coef = prior
        .gaussian_coefficients()
        .ok_or(ProbError::InvalidPrior { message: "composed prior has no coefficient prior" })?;
    if coef.len() != n_cols {
        return Err(ProbError::Shape { message: "coefficient prior length != design columns" });
    }
    let precision = prior.coefficient_precision(coef)?;
    let mut dense = vec![0.0; n_cols * n_cols];
    for r in 0..n_cols {
        for c in 0..n_cols {
            dense[r * n_cols + c] = precision.get(r, c);
        }
    }
    let chol = cholesky_spd(&dense, n_cols)?;
    let residual = match GaussianVarianceModel::from_prior_set(prior)? {
        GaussianVarianceModel::Known { sigma2 } => Residual::Known { sigma2 },
        GaussianVarianceModel::InvGamma { shape, scale } => Residual::InvGamma { shape, scale },
    };
    Ok(ConjugatePrior {
        mean: coef.mean.to_vec(),
        log_det_precision: log_det_from_chol(&chol, n_cols),
        precision: dense,
        residual,
    })
}

/// Choose the power-prior `α` of external Gaussian sources for a Gaussian linear target,
/// and report the target functional's posterior with a selection-aware interval.
///
/// Every source takes the same `α` from the grid (the power path of
/// [`compose_external_priors_with_alphas`]); a source with a mixture weight is refused.
/// The residual variance is the baseline's: known, or Inv-Gamma. Rows are unweighted.
///
/// # Errors
///
/// A design or functional of the wrong shape or with non-finite entries, a mixture-path
/// source, an invalid composition, invalid options, or a numerically singular posterior.
pub fn select_linear_power_prior(
    sources: &[ExternalPriorSource],
    baseline: &PriorSet,
    target: LinearTarget<'_>,
    options: &AlphaSelectionOptions,
) -> Result<SelectedAlphaReport, ProbError> {
    let LinearTarget { x, y, n_rows, n_cols, functional } = target;
    if n_rows == 0
        || n_cols == 0
        || x.len() != n_rows.saturating_mul(n_cols)
        || y.len() != n_rows
        || functional.len() != n_cols
    {
        return Err(ProbError::Shape {
            message: "linear target design, outcome or functional shape",
        });
    }
    if n_rows <= n_cols {
        return Err(ProbError::Shape {
            message: "alpha selection needs more rows than coefficients",
        });
    }
    if x.iter().chain(y).chain(functional).any(|v| !v.is_finite()) {
        return Err(ProbError::Numerical {
            message: "linear target holds a non-finite value".into(),
        });
    }
    if sources.is_empty() {
        return Err(ProbError::InvalidPrior {
            message: "alpha selection needs at least one source",
        });
    }
    if sources.iter().any(|s| s.weight.mixture_weight.is_some()) {
        return Err(ProbError::InvalidPrior {
            message: "alpha selection is defined on the power path; remove mixture weights",
        });
    }
    options.validate(n_rows)?;
    let mut priors = Vec::with_capacity(options.grid.len());
    for &alpha in &options.grid {
        let alphas = vec![alpha; sources.len()];
        let composed = compose_external_priors_with_alphas(sources, &alphas, &alphas, baseline)?;
        priors.push(conjugate_prior_from(&composed.prior, n_cols)?);
    }
    select_and_estimate(&LinearModel { target, priors }, options)
}

// -- Beta-Binomial and Gamma-Poisson targets ---------------------------------------------------

/// Source counts of a Beta-Binomial power prior: `Beta(a + α k₀, b + α (n₀ − k₀))`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BetaSource {
    /// Base prior before the source.
    pub base: BetaHyperparameters,
    /// Source successes `k₀`.
    pub successes: u64,
    /// Source trials `n₀`.
    pub trials: u64,
}

impl BetaSource {
    /// The power prior at discount `alpha`.
    ///
    /// # Errors
    ///
    /// An invalid base, `successes > trials`, or `alpha` outside `[0, 1]`.
    pub fn power_prior(&self, alpha: f64) -> Result<BetaHyperparameters, ProbError> {
        self.base.validate()?;
        if self.successes > self.trials {
            return Err(ProbError::Shape { message: "source successes must not exceed trials" });
        }
        if !(alpha.is_finite() && (0.0..=1.0).contains(&alpha)) {
            return Err(ProbError::InvalidPrior { message: "alpha must be finite and in [0, 1]" });
        }
        Ok(BetaHyperparameters {
            alpha: self.base.alpha + alpha * self.successes as f64,
            beta: self.base.beta + alpha * (self.trials - self.successes) as f64,
        })
    }
}

struct BetaModel<'a> {
    source: BetaSource,
    grid: &'a [f64],
    outcomes: &'a [bool],
}

impl AlphaModel for BetaModel<'_> {
    fn n_rows(&self) -> usize {
        self.outcomes.len()
    }

    fn log_marginal(&self, i: usize, rows: &[usize]) -> Result<f64, ProbError> {
        let prior = self.source.power_prior(self.grid[i])?;
        let k = rows.iter().filter(|&&r| self.outcomes[r]).count() as u64;
        Ok(BetaBinomialPosterior::new(prior, k, rows.len() as u64)?.log_marginal_likelihood())
    }

    fn component(&self, i: usize) -> Result<Component, ProbError> {
        let prior = self.source.power_prior(self.grid[i])?;
        let k = self.outcomes.iter().filter(|&&o| o).count() as u64;
        Ok(Component::Beta(
            BetaBinomialPosterior::new(prior, k, self.outcomes.len() as u64)?.posterior(),
        ))
    }
}

/// Choose the power-prior `α` of a source proportion for a Beta-Binomial target.
///
/// `outcomes[i]` is whether target trial `i` was a success (for the exceedance query,
/// whether the outcome exceeded the threshold). The reported functional is the proportion
/// `θ`.
///
/// # Errors
///
/// An invalid source or options, or fewer than two target trials.
pub fn select_beta_power_prior(
    source: BetaSource,
    outcomes: &[bool],
    options: &AlphaSelectionOptions,
) -> Result<SelectedAlphaReport, ProbError> {
    if outcomes.len() < 2 {
        return Err(ProbError::Shape {
            message: "alpha selection needs at least two target trials",
        });
    }
    options.validate(outcomes.len())?;
    select_and_estimate(&BetaModel { source, grid: &options.grid, outcomes }, options)
}

/// Source counts of a Gamma-Poisson power prior: `Gamma(a + α S₀, b + α E₀)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GammaSource {
    /// Base prior before the source.
    pub base: GammaHyperparameters,
    /// Source total count `S₀ = Σ y`.
    pub total_count: u64,
    /// Source total exposure `E₀` (the number of observations for unit exposures).
    pub exposure: f64,
}

impl GammaSource {
    /// The power prior at discount `alpha`.
    ///
    /// # Errors
    ///
    /// An invalid base, a negative or non-finite exposure, or `alpha` outside `[0, 1]`.
    pub fn power_prior(&self, alpha: f64) -> Result<GammaHyperparameters, ProbError> {
        self.base.validate()?;
        if !(self.exposure.is_finite() && self.exposure >= 0.0) {
            return Err(ProbError::InvalidPrior {
                message: "source exposure must be finite and >= 0",
            });
        }
        if !(alpha.is_finite() && (0.0..=1.0).contains(&alpha)) {
            return Err(ProbError::InvalidPrior { message: "alpha must be finite and in [0, 1]" });
        }
        Ok(GammaHyperparameters {
            shape: self.base.shape + alpha * self.total_count as f64,
            rate: self.base.rate + alpha * self.exposure,
        })
    }
}

struct GammaModel<'a> {
    source: GammaSource,
    grid: &'a [f64],
    counts: &'a [u64],
}

impl AlphaModel for GammaModel<'_> {
    fn n_rows(&self) -> usize {
        self.counts.len()
    }

    fn log_marginal(&self, i: usize, rows: &[usize]) -> Result<f64, ProbError> {
        let prior = self.source.power_prior(self.grid[i])?;
        let counts: Vec<u64> = rows.iter().map(|&r| self.counts[r]).collect();
        Ok(GammaPoissonPosterior::from_counts(prior, &counts, None)?.log_marginal_likelihood())
    }

    fn component(&self, i: usize) -> Result<Component, ProbError> {
        let prior = self.source.power_prior(self.grid[i])?;
        Ok(Component::Gamma(
            GammaPoissonPosterior::from_counts(prior, self.counts, None)?.posterior(),
        ))
    }
}

/// Choose the power-prior `α` of a source rate for a Gamma-Poisson target.
///
/// `counts[i]` is target observation `i`'s count over unit exposure. The reported
/// functional is the rate `λ`.
///
/// # Errors
///
/// An invalid source or options, or fewer than two target observations.
pub fn select_gamma_power_prior(
    source: GammaSource,
    counts: &[u64],
    options: &AlphaSelectionOptions,
) -> Result<SelectedAlphaReport, ProbError> {
    if counts.len() < 2 {
        return Err(ProbError::Shape {
            message: "alpha selection needs at least two target observations",
        });
    }
    options.validate(counts.len())?;
    select_and_estimate(&GammaModel { source, grid: &options.grid, counts }, options)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empirical_bayes_picks_the_marginal_likelihood_maximizer_on_a_beta_case() {
        // Source 30/100 against a target of 9/30: the data sit at the source's proportion.
        let source = BetaSource {
            base: BetaHyperparameters { alpha: 1.0, beta: 1.0 },
            successes: 30,
            trials: 100,
        };
        let outcomes: Vec<bool> = (0..30).map(|i| i < 9).collect();
        let options = AlphaSelectionOptions::new(AlphaCriterion::EmpiricalBayes);
        let report = select_beta_power_prior(source, &outcomes, &options).unwrap();
        let best = report
            .sensitivity
            .iter()
            .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
            .unwrap();
        assert_eq!(report.selected_alpha, best.alpha);
        assert!(report.interval.0 <= report.estimate && report.estimate <= report.interval.1);
    }

    #[test]
    fn a_discrepant_source_is_discounted() {
        let source = BetaSource {
            base: BetaHyperparameters { alpha: 1.0, beta: 1.0 },
            successes: 90,
            trials: 100,
        };
        let outcomes: Vec<bool> = (0..60).map(|i| i < 12).collect();
        let options = AlphaSelectionOptions::new(AlphaCriterion::EmpiricalBayes);
        let report = select_beta_power_prior(source, &outcomes, &options).unwrap();
        assert!(report.selected_alpha < 0.2);
    }

    #[test]
    fn cross_validation_needs_valid_folds() {
        let source = GammaSource {
            base: GammaHyperparameters { shape: 1.0, rate: 1.0 },
            total_count: 50,
            exposure: 25.0,
        };
        let counts = [2u64, 1, 3, 2, 2, 1];
        let bad = AlphaSelectionOptions::new(AlphaCriterion::CrossValidated { folds: 1 });
        assert!(select_gamma_power_prior(source, &counts, &bad).is_err());
        let good = AlphaSelectionOptions::new(AlphaCriterion::CrossValidated { folds: 3 });
        let report = select_gamma_power_prior(source, &counts, &good).unwrap();
        assert!(report.interval.0 < report.interval.1);
    }
}
