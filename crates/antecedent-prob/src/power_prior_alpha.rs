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
//! ## Intervals that stay valid under the selection
//!
//! A posterior interval at the selected `α̂` treats `α̂` as fixed in advance and is
//! reported only as `conditional_interval`. Two intervals that account for the
//! selection are reported:
//!
//! * `uniform_interval`: the interval of the target data alone (`α = 0`) widened to
//!   contain the estimate. It does not depend on the source, so it covers at the
//!   credible level whatever the source/target discrepancy. It cannot be narrower
//!   than the target-only interval: no interval that covers at the nominal level
//!   for **every** discrepancy can be, because the data that would tell a matching
//!   source from a discrepant one are the target data themselves.
//! * `interval` with a declared `discrepancy_bound τ`: a bias-aware interval around
//!   the estimate at the selected `α̂`. The source's prior means are assumed to lie
//!   within `τ` of the true coefficients (the rate or proportion, for the count
//!   models). The worst-case bias of the pooled estimator under that bound, added to
//!   the sampling error's critical value taken as the maximum over the whole `α`
//!   grid (so it holds whichever `α̂` the data select), covers at the credible
//!   level for every discrepancy up to `τ`, and is narrower than the target-only
//!   interval when `τ` is small. `τ` is an assumption, recorded in the report, like
//!   a hand-set `α`; beyond it no coverage is claimed. Without `τ`, `interval` is
//!   the uniform interval.
//!
//! For the linear target the coverage is exact for a known residual variance; with
//! an Inv-Gamma residual variance the OLS estimate is plugged in. The count models
//! use the normal approximation of the sufficient statistic for the bias-aware width.
//!
//! The report also carries a sensitivity table: the criterion's score, the estimate
//! and the conditional interval at every grid `α`. A prior is an assumption and
//! never upgrades identification.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    reason = "dense small-matrix and index arithmetic over bounded grids and row counts"
)]

use std::sync::Arc;

use antecedent_core::{CausalRng, PriorAssumption, derive_stream};
use antecedent_kernels::{norm_inv, standard_normal};
use antecedent_stats::{ln_gamma, student_t_ppf};

use crate::conjugate_counts::{BetaBinomialPosterior, GammaPoissonPosterior};
use crate::conjugate_moment_match::{BetaHyperparameters, GammaHyperparameters};
use crate::error::ProbError;
use crate::external_prior::{
    ComposedPrior, ExternalPriorSource, compose_external_priors_with_alphas,
};
use crate::linalg::{cholesky_spd, solve_chol_into};
use crate::prior::{GaussianVarianceModel, PriorSet};

/// Largest number of grid points.
pub const MAX_ALPHA_GRID: usize = 1_000;
/// Relative margin by which a larger `α` must beat the best score so far.
const TIE_MARGIN: f64 = 1e-12;
/// Stream used to assign rows to cross-validation folds.
const FOLD_STREAM: u64 = u64::MAX;
/// Stream of the draws that fix the selection-robust critical value.
const CRITICAL_STREAM: u64 = u64::MAX - 1;
/// Draws behind the maximum-over-the-grid critical value.
const CRITICAL_DRAWS: usize = 20_000;
/// Standard errors of the OLS estimate allowed for the (negligible) baseline-prior term.
const BASELINE_SE: f64 = 3.0;

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
    /// Strictly increasing `α` values in `[0, 1]`, starting at `0` (the target-only fit).
    pub grid: Vec<f64>,
    /// Credible level of the reported intervals, in `(0, 1)`.
    pub level: f64,
    /// Declared bound `τ` on how far the source's prior means lie from the truth
    /// (coefficient units; for the count models, proportion or rate units). With it,
    /// `interval` is the bias-aware interval; without it, the uniform interval.
    pub discrepancy_bound: Option<f64>,
    /// Seed of the fold assignment and of the critical-value draws.
    pub seed: u64,
}

impl AlphaSelectionOptions {
    /// Defaults: [`default_alpha_grid`], level 0.95, no declared bound, seed 0.
    #[must_use]
    pub fn new(criterion: AlphaCriterion) -> Self {
        Self {
            criterion,
            grid: default_alpha_grid(),
            level: 0.95,
            discrepancy_bound: None,
            seed: 0,
        }
    }

    fn validate(&self, n_rows: usize) -> Result<(), ProbError> {
        if self.grid.is_empty() || self.grid.len() > MAX_ALPHA_GRID {
            return Err(ProbError::InvalidPrior {
                message: "alpha grid must hold 1..=1000 points",
            });
        }
        if self.grid[0].abs() > 0.0 {
            return Err(ProbError::InvalidPrior {
                message: "alpha grid must start at 0, the target-only fit",
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
        if let Some(tau) = self.discrepancy_bound
            && !(tau.is_finite() && tau >= 0.0)
        {
            return Err(ProbError::InvalidPrior {
                message: "discrepancy bound must be finite and >= 0",
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
    /// The interval to report: the bias-aware interval when a discrepancy bound was
    /// declared (valid up to that bound), else [`Self::uniform_interval`].
    pub interval: (f64, f64),
    /// Target-only interval widened to contain the estimate; covers at the level for
    /// every source/target discrepancy.
    pub uniform_interval: (f64, f64),
    /// The declared bound `τ` behind [`Self::interval`], when one was declared.
    pub discrepancy_bound: Option<f64>,
    /// Estimate and conditional interval at every grid `α`.
    pub sensitivity: Vec<AlphaSensitivityRow>,
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

/// A conjugate target model scored over the `α` grid.
trait AlphaModel {
    /// Number of target rows.
    fn n_rows(&self) -> usize;
    /// Log marginal likelihood at grid point `i` of the target rows `rows`
    /// (indices into the original rows, repeats allowed).
    fn log_marginal(&self, i: usize, rows: &[usize]) -> Result<f64, ProbError>;
    /// Posterior of the functional at grid point `i` given all original rows.
    fn component(&self, i: usize) -> Result<Component, ProbError>;
    /// Worst-case absolute bias of the estimate at grid point `i` when the source's prior
    /// means lie within `tau` of the truth, and the standard deviation of its sampling
    /// error.
    fn bias_and_sd(&self, i: usize, tau: f64) -> Result<(f64, f64), ProbError>;
    /// Row-major correlation of the standardized sampling errors across the grid, or
    /// `None` when they are perfectly correlated.
    fn noise_correlation(&self) -> Result<Option<Vec<f64>>, ProbError>;
    /// A target-only interval whose coverage is at least `level` by construction (exact
    /// binomial or Poisson), when the model has one; otherwise the `α = 0` posterior
    /// interval stands in.
    fn exact_target_only(&self, _level: f64) -> Result<Option<(f64, f64)>, ProbError> {
        Ok(None)
    }
}

/// Quantile at `level` of `max_i |Z_i|` for `Z ~ N(0, corr)`; the one-dimensional normal
/// quantile when the errors are perfectly correlated.
fn selection_critical_value(
    corr: Option<&[f64]>,
    k: usize,
    level: f64,
    seed: u64,
) -> Result<f64, ProbError> {
    let Some(corr) = corr else {
        return Ok(norm_inv(0.5 + 0.5 * level));
    };
    let mut jittered = corr.to_vec();
    for i in 0..k {
        jittered[i * k + i] += 1e-10;
    }
    let chol = cholesky_spd(&jittered, k)?;
    let mut rng = CausalRng::from_seed(derive_stream(seed, CRITICAL_STREAM));
    let mut maxima = Vec::with_capacity(CRITICAL_DRAWS);
    let mut z = vec![0.0; k];
    for _ in 0..CRITICAL_DRAWS {
        for slot in &mut z {
            *slot = standard_normal(&mut rng);
        }
        let mut worst = 0.0_f64;
        for i in 0..k {
            let mut v = 0.0;
            for j in 0..=i {
                v += chol[i * k + j] * z[j];
            }
            worst = worst.max(v.abs());
        }
        maxima.push(worst);
    }
    maxima.sort_by(f64::total_cmp);
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "level is in (0, 1), so the rank is in 1..=CRITICAL_DRAWS"
    )]
    let index = ((level * CRITICAL_DRAWS as f64).ceil() as usize).clamp(1, CRITICAL_DRAWS) - 1;
    Ok(maxima[index])
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
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a uniform draw in [0, 1) scaled by i + 1 is a nonnegative index at most i"
        )]
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
    }

    let row = sensitivity[selected_index];
    let (target_lo, target_hi) = model
        .exact_target_only(options.level)?
        .unwrap_or((sensitivity[0].lower, sensitivity[0].upper));
    let uniform_interval = (target_lo.min(row.estimate), target_hi.max(row.estimate));
    let interval = match options.discrepancy_bound {
        None => uniform_interval,
        Some(tau) => {
            let corr = model.noise_correlation()?;
            let critical =
                selection_critical_value(corr.as_deref(), grid.len(), options.level, options.seed)?;
            let (bias, sd) = model.bias_and_sd(selected_index, tau)?;
            let half = bias + critical * sd;
            (row.estimate - half, row.estimate + half)
        }
    };
    Ok(SelectedAlphaReport {
        criterion: options.criterion,
        selected_alpha: row.alpha,
        selected_index,
        estimate: row.estimate,
        level: options.level,
        conditional_interval: (row.lower, row.upper),
        interval,
        uniform_interval,
        discrepancy_bound: options.discrepancy_bound,
        sensitivity,
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
    grid: Vec<f64>,
    /// Dense precision `Λ_k` of every source's coefficient prior.
    source_precisions: Vec<Vec<f64>>,
    /// Dense precision and mean of the baseline coefficient prior.
    baseline_precision: Vec<f64>,
    baseline_mean: Vec<f64>,
}

fn mat_vec(m: &[f64], p: usize, v: &[f64]) -> Vec<f64> {
    (0..p).map(|r| (0..p).map(|c| m[r * p + c] * v[c]).sum()).collect()
}

fn dense_precision(prior: &PriorSet, n_cols: usize) -> Result<Vec<f64>, ProbError> {
    let coef = prior
        .gaussian_coefficients()
        .ok_or(ProbError::InvalidPrior { message: "prior has no coefficient prior" })?;
    let precision = prior.coefficient_precision(coef)?;
    let mut dense = vec![0.0; n_cols * n_cols];
    for r in 0..n_cols {
        for c in 0..n_cols {
            dense[r * n_cols + c] = precision.get(r, c);
        }
    }
    Ok(dense)
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

impl LinearModel<'_> {
    /// `w = Λₙ⁻¹ c` at grid point `i` and the noise vector `X w` over the rows.
    fn weights(&self, i: usize) -> Result<(Vec<f64>, Vec<f64>), ProbError> {
        let p = self.target.n_cols;
        let all: Vec<usize> = (0..self.target.n_rows).collect();
        let fit = self.fit(i, &all)?;
        let mut scratch = vec![0.0; p];
        let mut w = vec![0.0; p];
        solve_chol_into(&fit.chol, p, self.target.functional, &mut scratch, &mut w);
        let a: Vec<f64> = (0..self.target.n_rows)
            .map(|r| self.target.x[r * p..(r + 1) * p].iter().zip(&w).map(|(x, w)| x * w).sum())
            .collect();
        Ok((w, a))
    }

    /// OLS coefficients, residual variance (`n − p` degrees of freedom) and the diagonal of
    /// `(X'X)⁻¹`.
    fn ols(&self) -> Result<(Vec<f64>, f64, Vec<f64>), ProbError> {
        let p = self.target.n_cols;
        let mut xtx = vec![0.0; p * p];
        let mut xty = vec![0.0; p];
        for r in 0..self.target.n_rows {
            let xi = &self.target.x[r * p..(r + 1) * p];
            for a in 0..p {
                xty[a] += xi[a] * self.target.y[r];
                for b in 0..p {
                    xtx[a * p + b] += xi[a] * xi[b];
                }
            }
        }
        let chol = cholesky_spd(&xtx, p)?;
        let mut scratch = vec![0.0; p];
        let mut beta = vec![0.0; p];
        solve_chol_into(&chol, p, &xty, &mut scratch, &mut beta);
        let mut rss = 0.0;
        for r in 0..self.target.n_rows {
            let xi = &self.target.x[r * p..(r + 1) * p];
            let e = self.target.y[r] - xi.iter().zip(&beta).map(|(a, b)| a * b).sum::<f64>();
            rss += e * e;
        }
        let sigma2 = rss / (self.target.n_rows - p) as f64;
        let mut diag = vec![0.0; p];
        for j in 0..p {
            let mut unit = vec![0.0; p];
            unit[j] = 1.0;
            let mut col = vec![0.0; p];
            solve_chol_into(&chol, p, &unit, &mut scratch, &mut col);
            diag[j] = col[j];
        }
        Ok((beta, sigma2, diag))
    }
}

impl AlphaModel for LinearModel<'_> {
    fn n_rows(&self) -> usize {
        self.target.n_rows
    }

    fn bias_and_sd(&self, i: usize, tau: f64) -> Result<(f64, f64), ProbError> {
        let p = self.target.n_cols;
        let (w, a) = self.weights(i)?;
        let (beta, ols_sigma2, xtx_inv_diag) = self.ols()?;
        let sigma2 = match self.priors[i].residual {
            Residual::Known { sigma2 } => sigma2,
            Residual::InvGamma { .. } => ols_sigma2,
        };
        // bias = c'Λₙ⁻¹ [Λ_b (m_b − β) + α Σ_k Λ_k (m_k − β)], |m_k − β| <= τ componentwise.
        let alpha = self.grid[i];
        let mut bound = 0.0;
        for lam in &self.source_precisions {
            bound += alpha * mat_vec(lam, p, &w).iter().map(|v| v.abs()).sum::<f64>() * tau;
        }
        // The baseline term is `w'Λ_b (m_b − β)`; the baseline is nearly flat, so it is
        // bounded at the OLS estimate plus a few standard errors.
        let lam_b_w = mat_vec(&self.baseline_precision, p, &w);
        for j in 0..p {
            let reach = (self.baseline_mean[j] - beta[j]).abs()
                + BASELINE_SE * (sigma2 * xtx_inv_diag[j]).sqrt();
            bound += lam_b_w[j].abs() * reach;
        }
        let sd = sigma2.sqrt() * a.iter().map(|v| v * v).sum::<f64>().sqrt();
        Ok((bound, sd))
    }

    fn noise_correlation(&self) -> Result<Option<Vec<f64>>, ProbError> {
        let k = self.grid.len();
        let vectors: Vec<Vec<f64>> =
            (0..k).map(|i| self.weights(i).map(|(_, a)| a)).collect::<Result<_, _>>()?;
        let norms: Vec<f64> =
            vectors.iter().map(|v| v.iter().map(|x| x * x).sum::<f64>().sqrt()).collect();
        let mut corr = vec![0.0; k * k];
        for i in 0..k {
            for j in 0..k {
                let dot: f64 = vectors[i].iter().zip(&vectors[j]).map(|(a, b)| a * b).sum();
                corr[i * k + j] = dot / (norms[i] * norms[j]);
            }
        }
        Ok(Some(corr))
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
/// and report the target functional's posterior with intervals that stay valid under the selection.
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
    let source_precisions =
        sources.iter().map(|s| dense_precision(&s.prior, n_cols)).collect::<Result<Vec<_>, _>>()?;
    let baseline_precision = dense_precision(baseline, n_cols)?;
    let baseline_mean = baseline
        .gaussian_coefficients()
        .ok_or(ProbError::InvalidPrior { message: "baseline has no coefficient prior" })?
        .mean
        .to_vec();
    select_and_estimate(
        &LinearModel {
            target,
            priors,
            grid: options.grid.clone(),
            source_precisions,
            baseline_precision,
            baseline_mean,
        },
        options,
    )
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

    fn bias_and_sd(&self, i: usize, tau: f64) -> Result<(f64, f64), ProbError> {
        let n0 = self.source.trials as f64;
        if self.source.trials == 0 {
            return Err(ProbError::Shape { message: "the source needs at least one trial" });
        }
        let source_rate = self.source.successes as f64 / n0;
        let lo = (source_rate - tau).max(0.0);
        let hi = (source_rate + tau).min(1.0);
        let alpha = self.grid[i];
        let c = self.source.base.alpha + alpha * self.source.successes as f64;
        let d = self.source.base.alpha + self.source.base.beta + alpha * n0;
        let n = self.outcomes.len() as f64;
        let bias = (c - d * lo).abs().max((c - d * hi).abs()) / (d + n);
        let theta = 0.5_f64.clamp(lo, hi);
        Ok((bias, (n * theta * (1.0 - theta)).sqrt() / (d + n)))
    }

    fn noise_correlation(&self) -> Result<Option<Vec<f64>>, ProbError> {
        Ok(None)
    }

    /// Clopper-Pearson interval: coverage at least `level` for every proportion.
    fn exact_target_only(&self, level: f64) -> Result<Option<(f64, f64)>, ProbError> {
        let n = self.outcomes.len() as u64;
        let k = self.outcomes.iter().filter(|&&o| o).count() as u64;
        let tail = 0.5 * (1.0 - level);
        let edge = |a: f64, b: f64, p: f64| -> Result<f64, ProbError> {
            BetaBinomialPosterior::new(BetaHyperparameters { alpha: a, beta: b }, 0, 0)?.quantile(p)
        };
        let lower = if k == 0 { 0.0 } else { edge(k as f64, (n - k + 1) as f64, tail)? };
        let upper = if k == n { 1.0 } else { edge((k + 1) as f64, (n - k) as f64, 1.0 - tail)? };
        Ok(Some((lower, upper)))
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

    fn bias_and_sd(&self, i: usize, tau: f64) -> Result<(f64, f64), ProbError> {
        if !self.source.exposure.is_finite() || self.source.exposure <= 0.0 {
            return Err(ProbError::Shape { message: "the source needs a positive exposure" });
        }
        let source_rate = self.source.total_count as f64 / self.source.exposure;
        let lo = (source_rate - tau).max(0.0);
        let hi = source_rate + tau;
        let alpha = self.grid[i];
        let c = self.source.base.shape + alpha * self.source.total_count as f64;
        let d = self.source.base.rate + alpha * self.source.exposure;
        let n = self.counts.len() as f64;
        let bias = (c - d * lo).abs().max((c - d * hi).abs()) / (d + n);
        Ok((bias, (n * hi).sqrt() / (d + n)))
    }

    fn noise_correlation(&self) -> Result<Option<Vec<f64>>, ProbError> {
        Ok(None)
    }

    /// Garwood interval: coverage at least `level` for every rate.
    fn exact_target_only(&self, level: f64) -> Result<Option<(f64, f64)>, ProbError> {
        let n = self.counts.len() as f64;
        let total: u64 = self.counts.iter().sum();
        let tail = 0.5 * (1.0 - level);
        let edge = |shape: f64, p: f64| -> Result<f64, ProbError> {
            GammaPoissonPosterior::from_totals(GammaHyperparameters { shape, rate: n }, 0, 0.0)?
                .quantile(p)
        };
        let lower = if total == 0 { 0.0 } else { edge(total as f64, tail)? };
        let upper = edge(total as f64 + 1.0, 1.0 - tail)?;
        Ok(Some((lower, upper)))
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

impl SelectedAlphaReport {
    /// Which selection-valid interval [`Self::interval`] is: `bias_aware` under a declared
    /// discrepancy bound, else `uniform`.
    #[must_use]
    pub const fn interval_semantics(&self) -> &'static str {
        if self.discrepancy_bound.is_some() { "bias_aware" } else { "uniform" }
    }

    /// Identity of the selection inside the prior composed at its `α`: the criterion, the
    /// selected discount, the credible level and the interval semantics (with the declared
    /// bound), so a prior selected under another rule or bound is a different prior.
    #[must_use]
    pub fn provenance_id(&self) -> String {
        let bound = self.discrepancy_bound.map_or_else(|| "none".to_owned(), |b| format!("{b:?}"));
        format!(
            "power_prior_alpha_selected:criterion={}:alpha={:?}:level={:?}:interval={}:\
             discrepancy_bound={bound}",
            self.criterion.name(),
            self.selected_alpha,
            self.level,
            self.interval_semantics(),
        )
    }

    fn provenance_assumption(&self) -> PriorAssumption {
        PriorAssumption {
            id: Arc::from(self.provenance_id()),
            description: Arc::from(format!(
                "power-prior discount alpha={:?} selected from the target data by {}; the \
                 reported interval is the {} interval at level {:?}; the selection is part of \
                 the prior and never upgrades identification",
                self.selected_alpha,
                self.criterion.name(),
                self.interval_semantics(),
                self.level,
            )),
        }
    }
}

/// The Beta power prior at the discount chosen from the target trials, with the selection report.
///
/// # Errors
///
/// As [`select_beta_power_prior`].
pub fn select_beta_prior(
    source: BetaSource,
    outcomes: &[bool],
    options: &AlphaSelectionOptions,
) -> Result<(BetaHyperparameters, SelectedAlphaReport), ProbError> {
    let report = select_beta_power_prior(source, outcomes, options)?;
    let prior = source.power_prior(report.selected_alpha)?;
    Ok((prior, report))
}

/// The Gamma power prior at the discount chosen from the target counts, with the report.
///
/// # Errors
///
/// As [`select_gamma_power_prior`].
pub fn select_gamma_prior(
    source: GammaSource,
    counts: &[u64],
    options: &AlphaSelectionOptions,
) -> Result<(GammaHyperparameters, SelectedAlphaReport), ProbError> {
    let report = select_gamma_power_prior(source, counts, options)?;
    let prior = source.power_prior(report.selected_alpha)?;
    Ok((prior, report))
}

/// The Gaussian prior composed from `sources` at the discount chosen from the target data, with
/// the selection report. Every source takes the selected `α`; the selection (criterion, `α`,
/// level, interval semantics) is a restriction of the composed prior, so it enters its identity.
///
/// # Errors
///
/// As [`select_linear_power_prior`], or a composition failure.
pub fn select_linear_prior(
    sources: &[ExternalPriorSource],
    baseline: &PriorSet,
    target: LinearTarget<'_>,
    options: &AlphaSelectionOptions,
) -> Result<(ComposedPrior, SelectedAlphaReport), ProbError> {
    let report = select_linear_power_prior(sources, baseline, target, options)?;
    let alphas = vec![report.selected_alpha; sources.len()];
    let mut composed = compose_external_priors_with_alphas(sources, &alphas, &alphas, baseline)?;
    composed.prior.restrictions.push(report.provenance_assumption());
    Ok((composed, report))
}

/// Compose Beta-Binomial count sources `(successes, trials, α)` into one Beta prior:
/// `Beta(a + Σ α k, b + Σ α (n − k))`.
///
/// # Errors
///
/// An invalid base, `successes > trials`, or `α` outside `[0, 1]`.
pub fn compose_beta_power_priors(
    base: BetaHyperparameters,
    sources: &[(u64, u64, f64)],
) -> Result<BetaHyperparameters, ProbError> {
    base.validate()?;
    let (mut alpha, mut beta) = (base.alpha, base.beta);
    for &(successes, trials, discount) in sources {
        let source = BetaSource { base, successes, trials };
        source.power_prior(discount)?;
        alpha += discount * successes as f64;
        beta += discount * (trials - successes) as f64;
    }
    Ok(BetaHyperparameters { alpha, beta })
}

/// Compose Gamma-Poisson count sources `(total count, exposure, α)` into one Gamma prior:
/// `Gamma(s + Σ α S, r + Σ α E)`.
///
/// # Errors
///
/// An invalid base, a negative exposure, or `α` outside `[0, 1]`.
pub fn compose_gamma_power_priors(
    base: GammaHyperparameters,
    sources: &[(u64, f64, f64)],
) -> Result<GammaHyperparameters, ProbError> {
    base.validate()?;
    let (mut shape, mut rate) = (base.shape, base.rate);
    for &(total_count, exposure, discount) in sources {
        let source = GammaSource { base, total_count, exposure };
        source.power_prior(discount)?;
        shape += discount * total_count as f64;
        rate += discount * exposure;
    }
    Ok(GammaHyperparameters { shape, rate })
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
    fn a_declared_bound_gives_a_bias_aware_interval_around_the_estimate() {
        let source = BetaSource {
            base: BetaHyperparameters { alpha: 1.0, beta: 1.0 },
            successes: 300,
            trials: 1_000,
        };
        let outcomes: Vec<bool> = (0..100).map(|i| i < 30).collect();
        let mut options = AlphaSelectionOptions::new(AlphaCriterion::EmpiricalBayes);
        options.discrepancy_bound = Some(0.02);
        let report = select_beta_power_prior(source, &outcomes, &options).unwrap();
        let mid = 0.5 * (report.interval.0 + report.interval.1);
        assert!((mid - report.estimate).abs() < 1e-12);
        assert_eq!(report.discrepancy_bound, Some(0.02));
        // A matching source with a small declared bound narrows the interval.
        let width = |i: (f64, f64)| i.1 - i.0;
        assert!(width(report.interval) < width(report.uniform_interval));
        // Without a bound the reported interval is the uniform one.
        options.discrepancy_bound = None;
        let plain = select_beta_power_prior(source, &outcomes, &options).unwrap();
        assert_eq!(plain.interval, plain.uniform_interval);
    }

    #[test]
    fn a_grid_without_the_target_only_fit_is_refused() {
        let source = BetaSource {
            base: BetaHyperparameters { alpha: 1.0, beta: 1.0 },
            successes: 3,
            trials: 10,
        };
        let mut options = AlphaSelectionOptions::new(AlphaCriterion::EmpiricalBayes);
        options.grid = vec![0.5, 1.0];
        assert!(select_beta_power_prior(source, &[true, false, true], &options).is_err());
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

    #[test]
    fn composed_count_priors_add_the_discounted_source_counts() {
        let beta = compose_beta_power_priors(
            BetaHyperparameters { alpha: 1.0, beta: 2.0 },
            &[(10, 40, 1.0), (6, 10, 0.5)],
        )
        .unwrap();
        assert_eq!((beta.alpha, beta.beta), (1.0 + 10.0 + 3.0, 2.0 + 30.0 + 2.0));
        let gamma = compose_gamma_power_priors(
            GammaHyperparameters { shape: 1.0, rate: 1.0 },
            &[(12, 4.0, 0.5)],
        )
        .unwrap();
        assert_eq!((gamma.shape, gamma.rate), (7.0, 3.0));
        assert!(
            compose_beta_power_priors(
                BetaHyperparameters { alpha: 1.0, beta: 1.0 },
                &[(1, 2, 1.5)]
            )
            .is_err()
        );
    }

    #[test]
    fn the_selected_beta_prior_is_the_power_prior_at_the_selected_alpha() {
        let base = BetaHyperparameters { alpha: 1.0, beta: 1.0 };
        let source = BetaSource { base, successes: 30, trials: 50 };
        let outcomes: Vec<bool> = (0..40).map(|i| i % 5 < 3).collect();
        let mut options = AlphaSelectionOptions::new(AlphaCriterion::EmpiricalBayes);
        options.grid = vec![0.0, 0.5, 1.0];
        options.discrepancy_bound = Some(0.2);
        let (prior, report) = select_beta_prior(source, &outcomes, &options).unwrap();
        let alpha = report.selected_alpha;
        assert_eq!((prior.alpha, prior.beta), (1.0 + alpha * 30.0, 1.0 + alpha * 20.0));
        let id = report.provenance_id();
        assert!(id.contains("interval=bias_aware") && id.contains("discrepancy_bound=0.2"), "{id}");
    }
}
