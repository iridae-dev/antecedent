//! Lag-aligned linear slope designs of the unfolded temporal point routes.
//!
//! The temporal cross-world, derivative and one-modifier conditional routes
//! publish functionals of one or more least-squares fits on the lag-aligned
//! back-door design: a slope times a level gap, an intercept plus a slope, a
//! product of two edge slopes, a local derivative, or the interaction-averaged
//! conditional slope. A [`TemporalSlopeDesign`] is one such fit prepared once on
//! the original series. It keeps every design row's lag window from the series,
//! so the whole route can be refit on resampled series times
//! ([`TemporalSlopeDesign::refit_rows`], called by a shared circular-block
//! bootstrap over every design the route reads) and its Gaussian-likelihood
//! posterior can be drawn ([`TemporalSlopeDesign::posterior`]).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{CausalRng, KernelPolicy, TargetPopulation, TemporalEffectQuery, VariableId};
use antecedent_data::{LaggedColumn, TemporalIndexer, TimeSeriesData};
use antecedent_expr::IdentifiedEstimand;
use antecedent_model::history_operators::{HistoryScan, SequentialRegression};
use antecedent_stats::LeastSquaresWorkspace;

use crate::adjustment::intervention_f64;
use crate::error::EstimationError;
use crate::temporal_adjustment::{TemporalSlopeSpec, offset_to_lag};
use crate::temporal_block::{AlignedRows, GappedAlignedRows};

/// Fewest aligned rows a slope design is fit on, the floor of the point routes.
const MIN_SLOPE_ROWS: usize = 8;

/// What a design's columns hold beyond the intercept.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SlopeLayout {
    /// `[1, treatments…, adjustment…]`: slopes are columns `1..=treatments`.
    Slopes { treatments: usize },
    /// `[1, T, W, T·W, adjustment…]`: the one-modifier conditional regression.
    Conditional,
}

/// One lag-aligned least-squares design of a temporal point route.
///
/// Column-major `rows × cols` with an intercept in column 0. Design row `r` is
/// the intact lag window `windows` lists as its `r`-th intact window, read at
/// its original series time.
#[derive(Clone, Debug)]
pub struct TemporalSlopeDesign {
    matrix: Vec<f64>,
    outcome: Vec<f64>,
    rows: usize,
    cols: usize,
    layout: SlopeLayout,
    windows: GappedAlignedRows,
}

/// The coefficients of one fit of a [`TemporalSlopeDesign`] and, for the
/// conditional layout, the modifier mean of the rows it was fit on.
#[derive(Clone, Debug, PartialEq)]
pub struct SlopeFit {
    /// Every coefficient, intercept first.
    pub coefficients: Vec<f64>,
    /// Mean of the modifier column over the fitted rows (conditional layout);
    /// `NaN` for a slope layout.
    pub modifier_mean: f64,
}

impl SlopeFit {
    /// The intercept.
    #[must_use]
    pub fn intercept(&self) -> f64 {
        self.coefficients[0]
    }

    /// The slopes of the treatments, in the order the design named them (slope layout),
    /// or `[β_T, β_W, β_TW]` (conditional layout).
    #[must_use]
    pub fn slopes(&self, count: usize) -> Vec<f64> {
        self.coefficients[1..=count].to_vec()
    }

    /// The one-modifier marginal conditional slope `β_T + β_TW · mean(W)` (conditional layout).
    #[must_use]
    pub fn conditional_slope(&self) -> f64 {
        self.coefficients[1] + self.coefficients[3] * self.modifier_mean
    }
}

/// Reusable buffers for [`TemporalSlopeDesign::refit_rows`].
#[derive(Debug, Default)]
pub struct SlopeRefitScratch {
    matrix: Vec<f64>,
    outcome: Vec<f64>,
}

/// Gaussian-likelihood posterior draws of one design's coefficients.
#[derive(Clone, Debug)]
pub struct SlopePosterior {
    /// Posterior mean of every coefficient (the ridge point the route publishes).
    pub mean: Vec<f64>,
    /// `draws[d]` is draw `d` of every coefficient.
    pub draws: Vec<Vec<f64>>,
    /// Factor the Gaussian-likelihood covariance was widened by for serial
    /// dependence: the aligned rows over the effective rows of the slope
    /// columns' normal-equation scores, at least one.
    pub tempering: f64,
    /// Smallest effective-row count of the slope columns' normal-equation scores.
    pub effective_rows: f64,
    /// Modifier mean of the full sample (conditional layout), else `NaN`.
    pub modifier_mean: f64,
}

/// The inputs of [`TemporalSlopeDesign::posterior`].
#[derive(Clone, Copy, Debug)]
pub struct SlopePosteriorSpec {
    /// Prior standard deviation of every slope (the intercept is unpenalized).
    pub prior_scale: f64,
    /// Draws to retain.
    pub draws: usize,
    /// Structural lag span of the design, the floor of the dependence block length.
    pub structural_span: usize,
}

impl TemporalSlopeDesign {
    /// The design [`crate::temporal_linear_slopes`] fits: intercept, each treatment at
    /// the treatment offset (or at the outcome's own lag when listed as
    /// contemporaneous), then the identified adjustment nodes, on the lag-aligned rows.
    ///
    /// # Errors
    ///
    /// The refusals of [`crate::temporal_linear_slopes`].
    pub fn slopes(
        data: &TimeSeriesData,
        estimand: &IdentifiedEstimand,
        indexer: &TemporalIndexer,
        spec: &TemporalSlopeSpec<'_>,
        policy: &KernelPolicy,
    ) -> Result<Self, EstimationError> {
        let TemporalSlopeSpec {
            outcome,
            treatments,
            treatment_offset,
            outcome_offset,
            contemporaneous,
        } = *spec;
        if treatments.is_empty() || treatments.contains(&outcome) {
            return Err(EstimationError::unsupported(
                "temporal linear slope needs treatments distinct from the outcome",
            ));
        }
        let sample_anchor = outcome_offset.max(0);
        let t_lag = offset_to_lag(treatment_offset - sample_anchor)?;
        let y_lag = offset_to_lag(outcome_offset - sample_anchor)?;
        let mut cols = Vec::with_capacity(treatments.len() + 1);
        for &treatment in treatments {
            let lag = if contemporaneous.contains(&treatment) { y_lag } else { t_lag };
            cols.push(LaggedColumn { variable: treatment, lag });
        }
        cols.push(LaggedColumn { variable: outcome, lag: y_lag });
        let treatment_count = treatments.len();
        let adjustment = adjustment_columns(
            estimand,
            indexer,
            sample_anchor,
            |variable| variable == outcome || treatments.contains(&variable),
            &mut cols,
        )?;
        let max_lag = cols.iter().map(|c| c.lag.raw()).max().unwrap_or(0);
        let mut scan = HistoryScan::plan(data, max_lag, Arc::<[LaggedColumn]>::from(cols))
            .map_err(EstimationError::from)?;
        let prep = scan.complete(data, policy).map_err(EstimationError::from)?;
        let n = prep.n;
        if n < MIN_SLOPE_ROWS {
            return Err(EstimationError::data_msg(
                "too few aligned rows for a temporal linear slope",
            ));
        }
        let ncols = 1 + treatment_count + adjustment;
        let mut matrix = vec![0.0; n * ncols];
        matrix[..n].fill(1.0);
        for k in 0..treatment_count {
            matrix[(1 + k) * n..(2 + k) * n].copy_from_slice(prep.column(k));
        }
        for k in 0..adjustment {
            let base = (1 + treatment_count + k) * n;
            matrix[base..base + n].copy_from_slice(prep.column(treatment_count + 1 + k));
        }
        let outcome_column = prep.column(treatment_count).to_vec();
        Ok(Self {
            matrix,
            outcome: outcome_column,
            rows: n,
            cols: ncols,
            layout: SlopeLayout::Slopes { treatments: treatment_count },
            windows: GappedAlignedRows { first_time: max_lag as usize, complete: vec![true; n] },
        })
    }

    /// The one-modifier conditional design `Y ~ 1 + T + W + T×W + Z` that
    /// [`crate::temporal_conditional_marginal_effect_with_gaps`] fits, with the
    /// treatment contrast `active − control` it scales the slope by. With
    /// `allow_gaps` the windows that read a missing value are dropped and the
    /// survivors keep their own times.
    ///
    /// # Errors
    ///
    /// The refusals of [`crate::temporal_conditional_marginal_effect_with_gaps`].
    pub fn conditional(
        data: &TimeSeriesData,
        estimand: &IdentifiedEstimand,
        query: &TemporalEffectQuery,
        indexer: &TemporalIndexer,
        policy: &KernelPolicy,
        allow_gaps: bool,
    ) -> Result<(Self, f64), EstimationError> {
        if query.effect_modifiers.len() != 1 {
            return Err(EstimationError::unsupported(
                "temporal conditional effect fits exactly one modifier",
            ));
        }
        query.validate()?;
        if query.target_population != TargetPopulation::AllObserved {
            return Err(EstimationError::TargetPopulation);
        }
        let active = intervention_f64(&query.active)?;
        let control = intervention_f64(&query.control)?;
        let delta = active - control;
        if delta == 0.0 {
            return Err(EstimationError::unsupported(
                "active and control treatment levels must differ",
            ));
        }
        let sample_anchor = query.outcome_offset().max(0);
        let t_lag = offset_to_lag(query.try_treatment_offset()? - sample_anchor)?;
        let y_lag = offset_to_lag(query.outcome_offset() - sample_anchor)?;
        let modifier = query.effect_modifiers[0];
        let mut cols = vec![
            LaggedColumn { variable: query.treatment, lag: t_lag },
            LaggedColumn { variable: query.outcome, lag: y_lag },
            LaggedColumn { variable: modifier, lag: y_lag },
        ];
        let adjustment = adjustment_columns(
            estimand,
            indexer,
            sample_anchor,
            |variable| {
                variable == modifier || variable == query.treatment || variable == query.outcome
            },
            &mut cols,
        )?;
        let max_lag = cols.iter().map(|c| c.lag.raw()).max().unwrap_or(0);
        let mut scan = HistoryScan::plan(data, max_lag, Arc::<[LaggedColumn]>::from(cols))
            .map_err(EstimationError::from)?;
        let (prep, complete) =
            scan.windows(data, policy, allow_gaps).map_err(EstimationError::from)?;
        let complete = complete.unwrap_or_else(|| vec![true; prep.n]);
        let n = prep.n;
        if n < MIN_SLOPE_ROWS {
            return Err(EstimationError::data_msg(
                "too few aligned rows for a temporal conditional effect",
            ));
        }
        let t = prep.column(0);
        let w = prep.column(2);
        let ncols = 4 + adjustment;
        let mut matrix = vec![0.0; n * ncols];
        for i in 0..n {
            matrix[i] = 1.0;
            matrix[n + i] = t[i];
            matrix[2 * n + i] = w[i];
            matrix[3 * n + i] = t[i] * w[i];
        }
        for k in 0..adjustment {
            let base = (4 + k) * n;
            matrix[base..base + n].copy_from_slice(prep.column(3 + k));
        }
        let outcome = prep.column(1).to_vec();
        Ok((
            Self {
                matrix,
                outcome,
                rows: n,
                cols: ncols,
                layout: SlopeLayout::Conditional,
                windows: GappedAlignedRows { first_time: max_lag as usize, complete },
            },
            delta,
        ))
    }

    /// Aligned design rows.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Design columns, intercept included.
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// The design's lag windows on the original series axis with their intactness.
    #[must_use]
    pub const fn windows(&self) -> &GappedAlignedRows {
        &self.windows
    }

    /// Whether a window was dropped for a missing value.
    #[must_use]
    pub fn has_gaps(&self) -> bool {
        self.windows.complete.iter().any(|intact| !intact)
    }

    /// The contiguous aligned rows of a design without gaps.
    #[must_use]
    pub const fn aligned_rows(&self) -> AlignedRows {
        AlignedRows { first_time: self.windows.first_time, rows: self.rows }
    }

    /// The modifier column's mean over `rows` (conditional layout), else `NaN`.
    fn modifier_mean_over(&self, rows: impl Iterator<Item = usize>) -> f64 {
        if self.layout != SlopeLayout::Conditional {
            return f64::NAN;
        }
        let column = &self.matrix[2 * self.rows..3 * self.rows];
        let (mut sum, mut count) = (0.0, 0usize);
        for row in rows {
            sum += column[row];
            count += 1;
        }
        if count == 0 { f64::NAN } else { sum / count as f64 }
    }

    /// Full-sample fit: least squares, or with `prior_scale` the posterior mean of the
    /// slopes under independent `N(0, scale²)` priors and the least-squares residual
    /// variance (the intercept unpenalized), as the point routes publish.
    ///
    /// # Errors
    ///
    /// A singular design or a non-positive prior scale.
    pub fn fit(&self, prior_scale: Option<f64>) -> Result<SlopeFit, EstimationError> {
        let mut ws = LeastSquaresWorkspace::default();
        let fit = SequentialRegression::solve_design(
            &self.matrix,
            self.rows,
            self.cols,
            &self.outcome,
            None,
            &mut ws,
        )
        .map_err(crate::util::stats_err)?;
        let coefficients = match prior_scale {
            Some(scale) => self.ridge_mean(&fit.coefficients, scale)?.0,
            None => fit.coefficients,
        };
        Ok(SlopeFit { coefficients, modifier_mean: self.modifier_mean_over(0..self.rows) })
    }

    /// Least-squares refit on design rows `rows` (a resampled index set, repeats
    /// allowed); `None` when the replicate design is singular or too small.
    #[must_use]
    pub fn refit_rows(&self, rows: &[usize], scratch: &mut SlopeRefitScratch) -> Option<SlopeFit> {
        let m = rows.len();
        if m < self.cols.max(MIN_SLOPE_ROWS) || rows.iter().any(|&row| row >= self.rows) {
            return None;
        }
        scratch.matrix.clear();
        scratch.matrix.resize(m * self.cols, 0.0);
        scratch.outcome.clear();
        scratch.outcome.extend(rows.iter().map(|&row| self.outcome[row]));
        for col in 0..self.cols {
            let source = &self.matrix[col * self.rows..(col + 1) * self.rows];
            let target = &mut scratch.matrix[col * m..(col + 1) * m];
            for (slot, &row) in target.iter_mut().zip(rows) {
                *slot = source[row];
            }
        }
        let coefficients = SequentialRegression::solve_coefficients(
            &scratch.matrix,
            m,
            self.cols,
            &scratch.outcome,
        )
        .ok()?;
        if !coefficients.iter().all(|value| value.is_finite()) {
            return None;
        }
        Some(SlopeFit {
            coefficients,
            modifier_mean: self.modifier_mean_over(rows.iter().copied()),
        })
    }

    /// Normal-equation scores `x_tj · ê_t` of every column at the least-squares fit,
    /// one series per column over the design rows; empty when the fit fails.
    #[must_use]
    pub fn normal_scores(&self) -> Vec<Vec<f64>> {
        crate::temporal_block::normal_equation_scores(
            &self.matrix,
            self.rows,
            self.cols,
            &self.outcome,
        )
        .unwrap_or_default()
    }

    /// The slope-penalized normal equations `X'X + λD` (row-major) and the
    /// posterior mean `(X'X + λD)⁻¹X'y`, `λ = σ̂² / scale²`, with `σ̂²` the
    /// least-squares residual variance; returns the mean, the inverse of the
    /// penalized Gram and `σ̂²`.
    fn ridge_mean(
        &self,
        ols: &[f64],
        scale: f64,
    ) -> Result<(Vec<f64>, Vec<f64>, f64), EstimationError> {
        if !(scale.is_finite() && scale > 0.0) {
            return Err(EstimationError::unsupported(
                "temporal slope prior scale must be finite and positive",
            ));
        }
        let (n, p) = (self.rows, self.cols);
        let sigma2 = crate::util::ols_sigma2(&self.matrix, n, p, &self.outcome, ols).max(1e-12);
        let penalty = sigma2 / (scale * scale);
        let mut gram = vec![0.0; p * p];
        let mut cross = vec![0.0; p];
        for i in 0..p {
            let left = &self.matrix[i * n..(i + 1) * n];
            cross[i] = left.iter().zip(&self.outcome).map(|(x, y)| x * y).sum();
            for j in i..p {
                let right = &self.matrix[j * n..(j + 1) * n];
                let value: f64 = left.iter().zip(right).map(|(a, b)| a * b).sum();
                gram[i * p + j] = value;
                gram[j * p + i] = value;
            }
        }
        for j in 1..p {
            gram[j * p + j] += penalty;
        }
        let inverse = antecedent_stats::invert_square(&gram, p).ok_or_else(|| {
            EstimationError::stats_msg("the slope-penalized normal equations are singular")
        })?;
        let mean = (0..p).map(|i| (0..p).map(|j| inverse[i * p + j] * cross[j]).sum()).collect();
        Ok((mean, inverse, sigma2))
    }

    /// Draws of the coefficients from the Gaussian-likelihood posterior under
    /// independent `N(0, scale²)` slope priors and a flat intercept, with the
    /// least-squares residual variance plugged in: mean `(X'X + λD)⁻¹X'y`,
    /// covariance `τ σ̂² (X'X + λD)⁻¹`.
    ///
    /// `τ ≥ 1` tempers the likelihood for serial dependence: the aligned rows over
    /// the smallest effective-row count of the slope columns' normal-equation
    /// scores at the dependence block length of those scores
    /// ([`crate::temporal_block::score_effective_rows`]). Draw `d` reads the
    /// semantic stream `stream_for(Bayesian, derive_stream(stream, d))`, so the
    /// draws do not depend on the thread count.
    ///
    /// # Errors
    ///
    /// A singular design, a non-positive prior scale, fewer than two draws, or a
    /// covariance that is not positive definite.
    pub fn posterior(
        &self,
        spec: SlopePosteriorSpec,
        stream: u64,
        rng: &antecedent_core::RngFactory,
    ) -> Result<SlopePosterior, EstimationError> {
        let SlopePosteriorSpec { prior_scale, draws, structural_span } = spec;
        if draws < 2 {
            return Err(EstimationError::stats_msg(
                "a temporal slope posterior needs at least two draws",
            ));
        }
        let mut ws = LeastSquaresWorkspace::default();
        let fit = SequentialRegression::solve_design(
            &self.matrix,
            self.rows,
            self.cols,
            &self.outcome,
            None,
            &mut ws,
        )
        .map_err(crate::util::stats_err)?;
        let (mean, inverse, sigma2) = self.ridge_mean(&fit.coefficients, prior_scale)?;
        let scores = crate::temporal_block::normal_equation_scores_of_residuals(
            &self.matrix,
            self.rows,
            self.cols,
            &fit.residuals,
        );
        let slope_scores: Vec<&[f64]> = scores.iter().skip(1).map(Vec::as_slice).collect();
        let all_scores: Vec<&[f64]> = scores.iter().map(Vec::as_slice).collect();
        let block =
            crate::temporal_block::dependence_block_length(structural_span, self.rows, &all_scores);
        let effective_rows = crate::temporal_block::score_effective_rows(&slope_scores, block);
        let tempering = if effective_rows.is_finite() && effective_rows > 0.0 {
            (self.rows as f64 / effective_rows).max(1.0)
        } else {
            1.0
        };
        let p = self.cols;
        let covariance: Vec<f64> = inverse.iter().map(|v| v * sigma2 * tempering).collect();
        let factor = antecedent_stats::cholesky_spd(&covariance, p).ok_or_else(|| {
            EstimationError::stats_msg(
                "the temporal slope posterior covariance is not positive definite",
            )
        })?;
        let mut out = Vec::with_capacity(draws);
        let mut z = vec![0.0; p];
        for d in 0..draws {
            let mut stream_rng: CausalRng = rng.stream_for(
                antecedent_core::StreamDomain::Bayesian,
                antecedent_core::derive_stream(stream, d as u64),
            );
            for slot in &mut z {
                *slot = antecedent_kernels::standard_normal(&mut stream_rng);
            }
            let draw: Vec<f64> = (0..p)
                .map(|i| mean[i] + (0..=i).map(|j| factor[i * p + j] * z[j]).sum::<f64>())
                .collect();
            out.push(draw);
        }
        Ok(SlopePosterior {
            mean,
            draws: out,
            tempering,
            effective_rows,
            modifier_mean: self.modifier_mean_over(0..self.rows),
        })
    }

    /// Series time of each design row (the intact windows in time order).
    fn row_times(&self) -> Vec<usize> {
        self.windows
            .complete
            .iter()
            .enumerate()
            .filter(|(_, intact)| **intact)
            .map(|(w, _)| self.windows.first_time + w)
            .collect()
    }

    /// One coherent joint draw of the coefficients of several designs read from the same series.
    ///
    /// Each design keeps exactly the tempered Gaussian marginal of [`Self::posterior`]. The
    /// designs are tied together by the dependence of their estimating equations: the
    /// normal-equation scores of every column of every design, placed on the series time axis
    /// (zero where a design has no intact window), give the Bartlett-weighted long-run
    /// covariance `C` of the stacked scores at the dependence block length. The sandwich
    /// `A C A'` with `A = diag(G_k^{-1})` is the joint covariance of the coefficients. Each
    /// block is whitened by the Cholesky factor of its own sandwich diagonal, which leaves the
    /// cross-design correlation of the whitened coordinates; a draw is one standard-normal
    /// vector for the whole stack, correlated by that structure and then mapped back through
    /// each design's own marginal factor. Shared noise, shared regressors and shared parameters
    /// of the regressions are therefore one dependence structure, not independent per-design
    /// draws. Draw `d` reads the semantic stream `derive_stream(stream, d)`.
    ///
    /// Returns one [`SlopePosterior`] per design (draw `d` of every design is the same joint
    /// draw) and the dependence block length of the stacked scores.
    ///
    /// # Errors
    ///
    /// A singular design, a non-positive prior scale, fewer than two draws, or a joint
    /// dependence structure that is not positive definite after a bounded ridge.
    #[allow(
        clippy::too_many_lines,
        reason = "fit each design, stack the scores on the series axis, estimate the joint dependence, then draw"
    )]
    pub fn joint_posterior(
        designs: &[&Self],
        spec: SlopePosteriorSpec,
        stream: u64,
        rng: &antecedent_core::RngFactory,
    ) -> Result<(Vec<SlopePosterior>, usize), EstimationError> {
        struct Part {
            mean: Vec<f64>,
            inverse: Vec<f64>,
            marginal_factor: Vec<f64>,
            tempering: f64,
            effective_rows: f64,
            scores: Vec<Vec<f64>>,
            times: Vec<usize>,
        }
        let SlopePosteriorSpec { prior_scale, draws, structural_span } = spec;
        if draws < 2 {
            return Err(EstimationError::stats_msg(
                "a temporal slope posterior needs at least two draws",
            ));
        }
        if designs.is_empty() {
            return Err(EstimationError::unsupported("a joint posterior needs a design"));
        }
        let mut parts = Vec::with_capacity(designs.len());
        for design in designs {
            let mut ws = LeastSquaresWorkspace::default();
            let fit = SequentialRegression::solve_design(
                &design.matrix,
                design.rows,
                design.cols,
                &design.outcome,
                None,
                &mut ws,
            )
            .map_err(crate::util::stats_err)?;
            let (mean, inverse, sigma2) = design.ridge_mean(&fit.coefficients, prior_scale)?;
            let scores = crate::temporal_block::normal_equation_scores_of_residuals(
                &design.matrix,
                design.rows,
                design.cols,
                &fit.residuals,
            );
            let slope_scores: Vec<&[f64]> = scores.iter().skip(1).map(Vec::as_slice).collect();
            let all_scores: Vec<&[f64]> = scores.iter().map(Vec::as_slice).collect();
            let block = crate::temporal_block::dependence_block_length(
                structural_span,
                design.rows,
                &all_scores,
            );
            let effective_rows = crate::temporal_block::score_effective_rows(&slope_scores, block);
            let tempering = if effective_rows.is_finite() && effective_rows > 0.0 {
                (design.rows as f64 / effective_rows).max(1.0)
            } else {
                1.0
            };
            let covariance: Vec<f64> = inverse.iter().map(|v| v * sigma2 * tempering).collect();
            let marginal_factor = antecedent_stats::cholesky_spd(&covariance, design.cols)
                .ok_or_else(|| {
                    EstimationError::stats_msg(
                        "the temporal slope posterior covariance is not positive definite",
                    )
                })?;
            parts.push(Part {
                mean,
                inverse,
                marginal_factor,
                tempering,
                effective_rows,
                scores,
                times: design.row_times(),
            });
        }
        let widths: Vec<usize> = designs.iter().map(|d| d.cols).collect();
        let total: usize = widths.iter().sum();
        let offsets: Vec<usize> = widths
            .iter()
            .scan(0usize, |acc, w| {
                let start = *acc;
                *acc += w;
                Some(start)
            })
            .collect();
        let dependence: Vec<DependencePart<'_>> = parts
            .iter()
            .map(|part| DependencePart {
                inverse: &part.inverse,
                scores: &part.scores,
                times: &part.times,
            })
            .collect();
        let (joint_factor, block) = joint_dependence_factor(&dependence, structural_span)?;
        let mut out: Vec<Vec<Vec<f64>>> =
            designs.iter().map(|_| Vec::with_capacity(draws)).collect();
        let mut z = vec![0.0; total];
        for d in 0..draws {
            let mut stream_rng: CausalRng = rng.stream_for(
                antecedent_core::StreamDomain::Bayesian,
                antecedent_core::derive_stream(stream, d as u64),
            );
            for slot in &mut z {
                *slot = antecedent_kernels::standard_normal(&mut stream_rng);
            }
            let v: Vec<f64> = (0..total)
                .map(|i| (0..=i).map(|j| joint_factor[i * total + j] * z[j]).sum())
                .collect();
            for (k, part) in parts.iter().enumerate() {
                let (offset, width) = (offsets[k], widths[k]);
                let draw: Vec<f64> = (0..width)
                    .map(|i| {
                        part.mean[i]
                            + (0..=i)
                                .map(|j| part.marginal_factor[i * width + j] * v[offset + j])
                                .sum::<f64>()
                    })
                    .collect();
                out[k].push(draw);
            }
        }
        let posteriors = parts
            .into_iter()
            .zip(out)
            .zip(designs)
            .map(|((part, draws), design)| SlopePosterior {
                mean: part.mean,
                draws,
                tempering: part.tempering,
                effective_rows: part.effective_rows,
                modifier_mean: design.modifier_mean_over(0..design.rows),
            })
            .collect();
        Ok((posteriors, block))
    }
}

/// One regression's contribution to a joint dependence structure: the matrix `A` that maps the
/// regression's summed estimating-equation scores to its coefficient error (`G⁻¹`, or any
/// positive multiple of the coefficient covariance), those scores column by column, and the
/// series time of each score.
pub(crate) struct DependencePart<'a> {
    /// Row-major `width × width` map from score sums to coefficient error, up to a scalar.
    pub inverse: &'a [f64],
    /// Normal-equation scores, one series per coefficient column, over the regression's rows.
    pub scores: &'a [Vec<f64>],
    /// Series time of each row.
    pub times: &'a [usize],
}

/// Lower Cholesky factor (row-major `total × total`, `total` the summed widths) of the
/// cross-regression correlation of whitened coefficient errors, and the block length used.
///
/// The scores of every regression are placed on the series time axis (zero where a regression
/// has no row) and stacked. Their Bartlett-weighted long-run covariance at the dependence
/// block length, pushed through `A`, is the joint sandwich covariance of the coefficient
/// errors. Each block is whitened by the Cholesky factor of its own sandwich diagonal, so the
/// diagonal blocks of the result are the identity and the off-diagonal blocks are the
/// dependence that shared innovations, shared regressors and a shared series put between the
/// regressions. A cross block that is not jointly positive definite is shrunk toward zero.
///
/// # Errors
///
/// A singular sandwich block, or a result that no bounded shrinkage makes positive definite.
#[allow(
    clippy::too_many_lines,
    reason = "stack the scores, estimate the long-run covariance, push through the sandwich, whiten and factor"
)]
pub(crate) fn joint_dependence_factor(
    parts: &[DependencePart<'_>],
    structural_span: usize,
) -> Result<(Vec<f64>, usize), EstimationError> {
    let widths: Vec<usize> = parts.iter().map(|part| part.scores.len()).collect();
    let first = parts.iter().filter_map(|p| p.times.first().copied()).min().unwrap_or(0);
    let last = parts.iter().filter_map(|p| p.times.last().copied()).max().unwrap_or(0);
    let span = last + 1 - first;
    let total: usize = widths.iter().sum();
    // Stacked score series on the series time axis.
    let mut stacked: Vec<Vec<f64>> = Vec::with_capacity(total);
    for part in parts {
        for score in part.scores {
            let mut axis = vec![0.0; span];
            for (&time, &value) in part.times.iter().zip(score) {
                axis[time - first] = value;
            }
            stacked.push(axis);
        }
    }
    let views: Vec<&[f64]> = stacked.iter().map(Vec::as_slice).collect();
    let block = crate::temporal_block::dependence_block_length(structural_span, span, &views)
        .max(1)
        .min(span);
    // Bartlett long-run covariance of the stacked scores.
    let mut long_run = vec![0.0; total * total];
    for lag in 0..block {
        let weight = if lag == 0 { 1.0 } else { 1.0 - lag as f64 / block as f64 };
        for i in 0..total {
            for j in 0..total {
                let gamma: f64 = (lag..span).map(|t| stacked[i][t] * stacked[j][t - lag]).sum();
                if lag == 0 {
                    long_run[i * total + j] += gamma;
                } else {
                    long_run[i * total + j] += weight * gamma;
                    long_run[j * total + i] += weight * gamma;
                }
            }
        }
    }
    // Sandwich A C A' with A = blockdiag(inverse_k).
    let offsets: Vec<usize> = widths
        .iter()
        .scan(0usize, |acc, w| {
            let start = *acc;
            *acc += w;
            Some(start)
        })
        .collect();
    let mut sandwich = vec![0.0; total * total];
    for (a, (&oa, &wa)) in offsets.iter().zip(&widths).enumerate() {
        for (b, (&ob, &wb)) in offsets.iter().zip(&widths).enumerate() {
            for i in 0..wa {
                for j in 0..wb {
                    let mut value = 0.0;
                    for k in 0..wa {
                        for l in 0..wb {
                            value += parts[a].inverse[i * wa + k]
                                * long_run[(oa + k) * total + ob + l]
                                * parts[b].inverse[j * wb + l];
                        }
                    }
                    sandwich[(oa + i) * total + ob + j] = value;
                }
            }
        }
    }
    // Whiten each block by the Cholesky factor of its own sandwich diagonal.
    let mut sandwich_factors = Vec::with_capacity(parts.len());
    for (&offset, &width) in offsets.iter().zip(&widths) {
        let mut diagonal = vec![0.0; width * width];
        for i in 0..width {
            for j in 0..width {
                diagonal[i * width + j] = sandwich[(offset + i) * total + offset + j];
            }
        }
        let scale = (0..width).map(|i| diagonal[i * width + i]).fold(0.0_f64, f64::max);
        let mut factor = None;
        for ridge in [0.0, 1e-10, 1e-8, 1e-6] {
            let mut jittered = diagonal.clone();
            for i in 0..width {
                jittered[i * width + i] += ridge * scale.max(f64::MIN_POSITIVE);
            }
            if let Some(found) = antecedent_stats::cholesky_spd(&jittered, width) {
                factor = Some(found);
                break;
            }
        }
        sandwich_factors.push(factor.ok_or_else(|| {
            EstimationError::stats_msg(
                "the joint temporal slope dependence has a singular design block",
            )
        })?);
    }
    // V = W_a Σ_ab W_b' with W = factor^{-1}; diagonal blocks are the identity.
    let mut correlation = vec![0.0; total * total];
    for a in 0..parts.len() {
        for b in 0..parts.len() {
            let (oa, wa, ob, wb) = (offsets[a], widths[a], offsets[b], widths[b]);
            if a == b {
                for i in 0..wa {
                    correlation[(oa + i) * total + oa + i] = 1.0;
                }
                continue;
            }
            // left = La^{-1} Σ_ab  (wa x wb), then block = left Lb^{-T}.
            let mut left = vec![0.0; wa * wb];
            for col in 0..wb {
                for i in 0..wa {
                    let mut value = sandwich[(oa + i) * total + ob + col];
                    for k in 0..i {
                        value -= sandwich_factors[a][i * wa + k] * left[k * wb + col];
                    }
                    left[i * wb + col] = value / sandwich_factors[a][i * wa + i];
                }
            }
            for row in 0..wa {
                for j in 0..wb {
                    let mut value = left[row * wb + j];
                    for k in 0..j {
                        value -= sandwich_factors[b][j * wb + k]
                            * correlation[(oa + row) * total + ob + k];
                    }
                    correlation[(oa + row) * total + ob + j] =
                        value / sandwich_factors[b][j * wb + j];
                }
            }
        }
    }
    // Make the whole matrix symmetric from the upper blocks and factor it.
    for i in 0..total {
        for j in 0..i {
            let upper = correlation[j * total + i];
            let lower = correlation[i * total + j];
            let same_block = offsets
                .iter()
                .zip(&widths)
                .any(|(&o, &w)| i >= o && i < o + w && j >= o && j < o + w);
            if !same_block {
                let value = 0.5 * (upper + lower);
                correlation[j * total + i] = value;
                correlation[i * total + j] = value;
            }
        }
    }
    let mut joint_factor = None;
    for ridge in [0.0_f64, 1e-10, 1e-8, 1e-6, 1e-4] {
        let mut jittered = correlation.clone();
        // Shrink the cross blocks toward zero rather than widening the marginals.
        if ridge > 0.0 {
            for i in 0..total {
                for j in 0..total {
                    let same_block = offsets
                        .iter()
                        .zip(&widths)
                        .any(|(&o, &w)| i >= o && i < o + w && j >= o && j < o + w);
                    if !same_block {
                        jittered[i * total + j] *= 1.0 - ridge.sqrt();
                    }
                }
            }
        }
        if let Some(found) = antecedent_stats::cholesky_spd(&jittered, total) {
            joint_factor = Some(found);
            break;
        }
    }
    let joint_factor = joint_factor.ok_or_else(|| {
        EstimationError::stats_msg("the joint temporal slope dependence is not positive definite")
    })?;
    Ok((joint_factor, block))
}

/// Append the identified adjustment nodes of `estimand` at their lag from the
/// sample anchor, skipping the variables `skip` names; returns how many were added.
fn adjustment_columns(
    estimand: &IdentifiedEstimand,
    indexer: &TemporalIndexer,
    sample_anchor: i32,
    skip: impl Fn(VariableId) -> bool,
    cols: &mut Vec<LaggedColumn>,
) -> Result<usize, EstimationError> {
    let mut added = 0;
    for &dense_var in estimand.adjustment_set.iter() {
        let key = indexer
            .key_of(dense_var.raw())
            .map_err(|e| EstimationError::data_msg(e.to_string()))?;
        if skip(key.variable) {
            continue;
        }
        let lag = offset_to_lag(key.offset - sample_anchor).map_err(|_| {
            EstimationError::unsupported(
                "adjustment node sits after this horizon's sample anchor and cannot be lag-aligned",
            )
        })?;
        cols.push(LaggedColumn { variable: key.variable, lag });
        added += 1;
    }
    Ok(added)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn design(rows: usize) -> TemporalSlopeDesign {
        // y = 1 + 2 x with a deterministic wiggle; x periodic.
        let x: Vec<f64> = (0..rows).map(|t| ((t as f64) * 0.37).sin()).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(t, v)| 1.0 + 2.0 * v + if t % 2 == 0 { 0.1 } else { -0.1 })
            .collect();
        let mut matrix = vec![1.0; rows * 2];
        matrix[rows..].copy_from_slice(&x);
        TemporalSlopeDesign {
            matrix,
            outcome: y,
            rows,
            cols: 2,
            layout: SlopeLayout::Slopes { treatments: 1 },
            windows: GappedAlignedRows { first_time: 1, complete: vec![true; rows] },
        }
    }

    #[test]
    fn identity_refit_reproduces_the_full_sample_fit() {
        let design = design(60);
        let full = design.fit(None).unwrap();
        let rows: Vec<usize> = (0..60).collect();
        let refit = design.refit_rows(&rows, &mut SlopeRefitScratch::default()).unwrap();
        for (a, b) in full.coefficients.iter().zip(&refit.coefficients) {
            assert!((a - b).abs() < 1e-9);
        }
        assert!((full.slopes(1)[0] - 2.0).abs() < 0.05);
    }

    #[test]
    fn too_few_rows_refuse_a_replicate() {
        let design = design(60);
        assert!(design.refit_rows(&[0, 1, 2], &mut SlopeRefitScratch::default()).is_none());
        assert!(
            design
                .refit_rows(&[0, 1, 2, 3, 4, 5, 6, 600], &mut SlopeRefitScratch::default())
                .is_none()
        );
    }

    #[test]
    fn posterior_draws_centre_on_the_ridge_point_and_are_reproducible() {
        let design = design(80);
        let rng = antecedent_core::RngFactory::from_seed(7);
        let spec = SlopePosteriorSpec { prior_scale: 1_000.0, draws: 400, structural_span: 1 };
        let first = design.posterior(spec, 11, &rng).unwrap();
        let second = design.posterior(spec, 11, &rng).unwrap();
        assert_eq!(first.draws, second.draws);
        let ridge = design.fit(Some(1_000.0)).unwrap();
        for (a, b) in first.mean.iter().zip(&ridge.coefficients) {
            assert!((a - b).abs() < 1e-8);
        }
        let mean_slope: f64 = first.draws.iter().map(|d| d[1]).sum::<f64>() / 400.0;
        assert!((mean_slope - ridge.coefficients[1]).abs() < 0.05);
        assert!(first.tempering >= 1.0);
    }

    /// Deterministic standard-normal-ish noise (sum of uniforms) from a seeded LCG.
    fn noise(seed: u64, n: usize) -> Vec<f64> {
        let mut state = seed;
        let mut uniform = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) as f64) / ((1_u64 << 53) as f64)
        };
        (0..n).map(|_| (0..12).map(|_| uniform()).sum::<f64>() - 6.0).collect()
    }

    fn slope_design(x: &[f64], y: Vec<f64>) -> TemporalSlopeDesign {
        let rows = x.len();
        let mut matrix = vec![1.0; rows * 2];
        matrix[rows..].copy_from_slice(x);
        TemporalSlopeDesign {
            matrix,
            outcome: y,
            rows,
            cols: 2,
            layout: SlopeLayout::Slopes { treatments: 1 },
            windows: GappedAlignedRows { first_time: 2, complete: vec![true; rows] },
        }
    }

    fn correlation(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len() as f64;
        let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
        let cov: f64 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum();
        let va: f64 = a.iter().map(|x| (x - ma) * (x - ma)).sum();
        let vb: f64 = b.iter().map(|y| (y - mb) * (y - mb)).sum();
        cov / (va * vb).sqrt()
    }

    fn joint_slope_columns(shared_noise: bool) -> (Vec<f64>, Vec<f64>, f64, f64) {
        let n = 400;
        let x1 = noise(1, n);
        let u = noise(2, n);
        let x2: Vec<f64> = x1.iter().zip(&u).map(|(a, b)| 0.9 * a + 0.44 * b).collect();
        let e1 = noise(3, n);
        let e2 = if shared_noise { e1.clone() } else { noise(4, n) };
        let y1: Vec<f64> = x1.iter().zip(&e1).map(|(x, e)| 1.0 + 2.0 * x + e).collect();
        let y2: Vec<f64> = x2.iter().zip(&e2).map(|(x, e)| 0.5 - 1.0 * x + e).collect();
        let (d1, d2) = (slope_design(&x1, y1), slope_design(&x2, y2));
        let rng = antecedent_core::RngFactory::from_seed(5);
        let spec = SlopePosteriorSpec { prior_scale: 1_000.0, draws: 6000, structural_span: 1 };
        let (joint, _) = TemporalSlopeDesign::joint_posterior(&[&d1, &d2], spec, 9, &rng).unwrap();
        let marginal = d1.posterior(spec, 9, &rng).unwrap();
        let sd = |draws: &[Vec<f64>]| {
            let v: Vec<f64> = draws.iter().map(|d| d[1]).collect();
            let m = v.iter().sum::<f64>() / v.len() as f64;
            (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
        };
        let a: Vec<f64> = joint[0].draws.iter().map(|d| d[1]).collect();
        let b: Vec<f64> = joint[1].draws.iter().map(|d| d[1]).collect();
        (a, b, sd(&joint[0].draws), sd(&marginal.draws))
    }

    #[test]
    fn joint_draws_carry_shared_noise_dependence_and_keep_each_marginal() {
        // Same innovations drive both regressions and the regressors are correlated, so the
        // two slope errors are strongly correlated; independent innovations give none. Both
        // are checked against the exact finite-sample value (the regressor correlation for
        // shared noise, zero for independent noise), not against the paired-marginal draws.
        let (a, b, joint_sd, marginal_sd) = joint_slope_columns(true);
        let shared = correlation(&a, &b);
        assert!(shared > 0.7, "shared-noise slope correlation {shared}");
        assert!((joint_sd / marginal_sd - 1.0).abs() < 0.08, "{joint_sd} vs {marginal_sd}");
        let (a, b, _, _) = joint_slope_columns(false);
        let independent = correlation(&a, &b);
        assert!(independent.abs() < 0.25, "independent-noise slope correlation {independent}");
    }

    #[test]
    fn posterior_refuses_one_draw() {
        let design = design(40);
        let rng = antecedent_core::RngFactory::from_seed(7);
        let spec = SlopePosteriorSpec { prior_scale: 10.0, draws: 1, structural_span: 1 };
        assert!(design.posterior(spec, 0, &rng).is_err());
    }
}
