//! One retained factor for a gathered Gaussian design.
//!
//! Gaussian estimation, a flat-prior Laplace step, an identity-mass HMC step,
//! and a repeated contrast all read this summary. The design is gathered and
//! factored once, under the one-worker scope. A short rank drops the Gram
//! factor and keeps the minimum-norm SVD solution of that same gather. Which
//! scope is faster is already recorded for the larger QR designs on this host;
//! this path does not measure that crossover again.
//!
//! What is retained is the Gram of the weighted rows and, when the pivot test
//! accepts it, the Cholesky factor of that Gram. The reference residual at the
//! weighted least-squares solution is stored beside it: the anchor
//! coefficients, `X' W r₀`, and `r₀' W r₀`. A later residual sum of squares
//! expands about that anchor. The raw moment `y' W y` is not stored, and
//! neither is an inverse.
//!
//! Per-row residuals, leverage, and Cook's distance are not determined by that
//! summary. Each one is an extra pass over the original rows. Leverage solves
//! the Gram factor one row at a time.
//!
//! [`FactorCondition::NumericallySingular`] and [`FactorCondition::IllConditioned`]
//! are different outcomes. A short rank, or a Gram pivot the Cholesky test
//! refuses, is singular. A full-rank factor whose reciprocal condition sits
//! below [`ILL_CONDITIONED_RCOND`] is ill-conditioned, and the factor is kept.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_kernels::cholesky::{CholeskyError, cholesky_spd_pivots_into};

use crate::error::StatsError;
use crate::faer_scope::FaerScope;
use crate::gram::{chol_solve, form_xtx};
use crate::solver_workspace::SolverWorkspace;

/// Reciprocal-condition floor of the gathered QR.
///
/// A full-rank factor at or above this floor is [`FactorCondition::Stable`].
/// Below it, the factor is [`FactorCondition::IllConditioned`] and is still
/// retained. Rank and a refused Gram pivot do not use this floor.
pub const ILL_CONDITIONED_RCOND: f64 = 1e-3;

/// A negative residual sum of squares inside this fraction of its scale is rounding.
const RESIDUAL_SUM_ROUNDING: f64 = 1e-9;

/// Condition of the retained Gram factor.
///
/// Singular and ill-conditioned are separate. An ill-conditioned factor is
/// full rank and still used. A singular factor is not an extreme case of the
/// same status.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FactorCondition {
    /// Full rank, reciprocal condition at or above the floor.
    Stable {
        /// `min|R_ii| / max|R_ii|` of the gathered QR.
        rcond: f64,
    },
    /// Full rank, reciprocal condition below the floor. The Gram factor is kept.
    IllConditioned {
        /// `min|R_ii| / max|R_ii|` of the gathered QR.
        rcond: f64,
    },
    /// Short QR rank, or a Gram pivot the Cholesky test refused.
    NumericallySingular {
        /// Accepted rank: the QR rank, or the first refused Gram pivot.
        rank: usize,
        /// Columns of the gathered design.
        ncols: usize,
    },
}

/// Column-major sample a retained factor is built from, or scanned again.
///
/// `weights` of `None` weighs every row by one. Rows not listed in `selected`
/// are not part of the gather.
#[derive(Clone, Copy, Debug)]
pub struct SelectedDesign<'a> {
    /// Column-major source. Column `c` starts at `c * source_rows`.
    pub x_colmajor: &'a [f64],
    /// Rows in the source, which is also the length of `outcome`.
    pub source_rows: usize,
    /// Columns.
    pub ncols: usize,
    /// Outcome aligned with `source_rows`.
    pub outcome: &'a [f64],
    /// Non-negative row weights aligned with `source_rows`.
    pub weights: Option<&'a [f64]>,
    /// Source rows copied into the gather.
    pub selected: &'a [usize],
}

/// Gaussian coefficients from the one gathered solve.
#[derive(Clone, Debug)]
pub struct GaussianEstimate {
    /// Weighted least-squares coefficients, or the minimum-norm solution.
    pub coefficients: Vec<f64>,
    /// Rank reported by that solve.
    pub rank: usize,
    /// `r₀' W r₀` at those coefficients. This is not `y' W y`.
    pub reference_rss: f64,
    /// Condition of the Gram factor that accompanies the solve.
    pub condition: FactorCondition,
}

/// One Newton step of the flat-prior Gaussian log-likelihood.
///
/// The step solves the retained Gram factor. A positive `sigma_squared` makes
/// the target a defined log-density; the Gram step itself does not scale with it.
#[derive(Clone, Debug)]
pub struct LaplaceStep {
    /// Coefficients after the step.
    pub coefficients: Vec<f64>,
    /// Condition of the factor the step solved.
    pub condition: FactorCondition,
}

/// Position and identity-mass momentum of one HMC state.
#[derive(Clone, Debug)]
pub struct HmcState {
    /// Unconstrained coefficients.
    pub position: Vec<f64>,
    /// Momentum in the Euclidean metric.
    pub momentum: Vec<f64>,
}

/// One leapfrog step. The force is the retained score; the mass is identity.
#[derive(Clone, Debug)]
pub struct HmcStep {
    /// State after the step.
    pub state: HmcState,
    /// Condition of the factor that made the step eligible.
    pub condition: FactorCondition,
}

/// Contrast of the retained Gaussian coefficients.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RepeatedQuery {
    /// `contrast · β` at the retained solution.
    pub value: f64,
    /// Condition of the factor the solution was taken from.
    pub condition: FactorCondition,
}

/// Stable Gram factor, or the minimum-norm solution when that factor is refused.
///
/// The gathered design lives in the solver workspace. Queries after [`Self::retain`]
/// do not gather again and do not build another factor.
#[must_use = "queries read this factor; dropping it discards the gather"]
pub struct RetainedFactor {
    workspace: SolverWorkspace,
    /// Row-major `X' W X`.
    gram: Vec<f64>,
    /// Lower-triangular Cholesky of `gram` when the pivot test accepts it.
    /// Absent for a numerically singular design. Not an inverse.
    gram_factor: Option<Vec<f64>>,
    /// Weighted least-squares anchor `β₀`, or the minimum-norm anchor.
    anchor: Vec<f64>,
    /// `X' W r₀` at the anchor.
    anchor_score: Vec<f64>,
    /// `r₀' W r₀` at the anchor. Not `y' W y`.
    anchor_rss: f64,
    solver_rank: usize,
    condition: FactorCondition,
    /// Builds of this summary. A query does not increment it.
    factor_builds: u64,
    nrows: usize,
    ncols: usize,
}

impl core::fmt::Debug for RetainedFactor {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("RetainedFactor")
            .field("workspace", &self.workspace.layout_copies())
            .field("gram", &self.gram)
            .field("gram_factor", &self.gram_factor.as_ref().map(std::vec::Vec::len))
            .field("anchor", &self.anchor)
            .field("anchor_score", &self.anchor_score)
            .field("anchor_rss", &self.anchor_rss)
            .field("solver_rank", &self.solver_rank)
            .field("condition", &self.condition)
            .field("factor_builds", &self.factor_builds)
            .field("nrows", &self.nrows)
            .field("ncols", &self.ncols)
            .finish()
    }
}

impl RetainedFactor {
    /// Gather `design` once and retain its factor or its minimum-norm solution.
    ///
    /// The scope is one worker. A second query does not gather or factor again.
    ///
    /// # Errors
    ///
    /// The selection is empty, a row is out of range, a weight is negative, a
    /// value is non-finite, the effective sample size is not positive, or the
    /// gathered shape is not a least-squares shape (`nrows < ncols`).
    #[must_use = "a refused gather does not retain a factor"]
    pub fn retain(design: SelectedDesign<'_>) -> Result<Self, StatsError> {
        let ncols = design.ncols;
        let (scaled, outcome) = materialize(design)?;
        let nrows = outcome.len();
        let mut workspace = SolverWorkspace::new(one_worker_scope());
        let rows: Vec<usize> = (0..nrows).collect();
        workspace.gather_selected(&scaled, nrows, ncols, &outcome, &rows)?;
        let fit = workspace.solve()?;
        if fit.coefficients.len() != ncols || fit.residuals.len() != nrows {
            return Err(StatsError::Backend("solver returned a different shape".into()));
        }
        if fit.coefficients.iter().any(|value| !value.is_finite()) {
            return Err(StatsError::Backend("non-finite least-squares solution".into()));
        }
        let gram = weighted_gram(&scaled, nrows, ncols)?;
        let anchor_score = cross_score(&scaled, nrows, ncols, &fit.residuals);
        if !fit.rss.is_finite() || anchor_score.iter().any(|value| !value.is_finite()) {
            return Err(StatsError::Backend("non-finite reference residual".into()));
        }
        let (gram_factor, condition) = classify(ncols, fit.rank, fit.diagnostics.rcond, &gram);
        Ok(Self {
            workspace,
            gram,
            gram_factor,
            anchor: fit.coefficients,
            anchor_score,
            anchor_rss: fit.rss,
            solver_rank: fit.rank,
            condition,
            factor_builds: 1,
            nrows,
            ncols,
        })
    }

    /// Coefficients and reference residual from the gathered solve.
    ///
    /// A second call copies those values. It does not factor the design again.
    #[must_use]
    pub fn gaussian_estimate(&self) -> GaussianEstimate {
        GaussianEstimate {
            coefficients: self.anchor.clone(),
            rank: self.solver_rank,
            reference_rss: self.anchor_rss,
            condition: self.condition,
        }
    }

    /// Newton step from `beta` through the retained Gram factor.
    ///
    /// Eligible when the Gram factor was retained: full rank, including an
    /// ill-conditioned factor. A numerically singular design has no such step.
    ///
    /// # Errors
    ///
    /// `sigma_squared` is not positive and finite, `beta` has the wrong length
    /// or a non-finite entry, the Gram factor is absent, or the step is non-finite.
    #[must_use = "a refused Laplace step is not a coefficient vector"]
    pub fn laplace_step(
        &self,
        beta: &[f64],
        sigma_squared: f64,
    ) -> Result<LaplaceStep, StatsError> {
        require_sigma(sigma_squared)?;
        let factor = self.require_gram_factor()?;
        let score = self.score(beta)?;
        let step = chol_solve(factor, self.ncols, &score).ok_or_else(|| {
            StatsError::Backend("retained gram factor rejected the Laplace step".into())
        })?;
        let mut coefficients = Vec::with_capacity(self.ncols);
        for (value, increment) in beta.iter().zip(&step) {
            let next = value + increment;
            if !next.is_finite() {
                return Err(StatsError::Backend("non-finite Laplace step".into()));
            }
            coefficients.push(next);
        }
        Ok(LaplaceStep { coefficients, condition: self.condition })
    }

    /// One identity-mass leapfrog step. The force is the retained score.
    ///
    /// The design is not rescanned. A numerically singular factor is not a
    /// target for this step.
    ///
    /// # Errors
    ///
    /// `sigma_squared` is not positive and finite, the step size or the state
    /// is non-finite, the state length differs from the column count, or the
    /// Gram factor is absent.
    #[must_use = "a refused HMC step is not a state"]
    pub fn hmc_step(
        &self,
        state: &HmcState,
        step_size: f64,
        sigma_squared: f64,
    ) -> Result<HmcStep, StatsError> {
        require_sigma(sigma_squared)?;
        self.require_gram_factor()?;
        if !step_size.is_finite() {
            return Err(StatsError::Shape { message: "HMC step size must be finite" });
        }
        if state.position.len() != self.ncols || state.momentum.len() != self.ncols {
            return Err(StatsError::Shape { message: "HMC state length != ncols" });
        }
        if !finite_slice(&state.position) || !finite_slice(&state.momentum) {
            return Err(StatsError::Shape { message: "non-finite HMC state" });
        }
        let half = 0.5 * step_size;
        let mut position = state.position.clone();
        let mut momentum = state.momentum.clone();
        self.kick(&mut momentum, &position, sigma_squared, half)?;
        for (coordinate, impulse) in position.iter_mut().zip(&momentum) {
            *coordinate += step_size * impulse;
        }
        self.kick(&mut momentum, &position, sigma_squared, half)?;
        if !finite_slice(&position) || !finite_slice(&momentum) {
            return Err(StatsError::Backend("non-finite HMC step".into()));
        }
        Ok(HmcStep { state: HmcState { position, momentum }, condition: self.condition })
    }

    /// `contrast · β` at the retained solution.
    ///
    /// This reads the gathered coefficients. A singular solution still answers,
    /// and the status on the result stays numerically singular.
    ///
    /// # Errors
    ///
    /// `contrast` has the wrong length, a non-finite entry, or the dot product
    /// is non-finite.
    #[must_use = "a refused contrast is not a scalar"]
    pub fn repeated_query(&self, contrast: &[f64]) -> Result<RepeatedQuery, StatsError> {
        if contrast.len() != self.ncols {
            return Err(StatsError::Shape { message: "contrast length != ncols" });
        }
        if !finite_slice(contrast) {
            return Err(StatsError::Shape { message: "non-finite contrast" });
        }
        let mut value = 0.0;
        for (weight, coefficient) in contrast.iter().zip(&self.anchor) {
            value += weight * coefficient;
        }
        if !value.is_finite() {
            return Err(StatsError::Backend("non-finite repeated query".into()));
        }
        Ok(RepeatedQuery { value, condition: self.condition })
    }

    /// Residual sum of squares at `beta`, expanded about the anchor residual.
    ///
    /// No row is read. A negative value inside rounding of zero is clamped to
    /// zero; a larger one is refused.
    ///
    /// # Errors
    ///
    /// `beta` has the wrong length or a non-finite entry, or the expansion is
    /// non-finite or negative beyond rounding.
    #[must_use = "a refused expansion is not a residual sum of squares"]
    pub fn residual_sum_of_squares(&self, beta: &[f64]) -> Result<f64, StatsError> {
        let score = self.score(beta)?;
        let mut rss = self.anchor_rss;
        let mut scale = self.anchor_rss.abs();
        for index in 0..self.ncols {
            let delta = beta[index] - self.anchor[index];
            let term = delta * (self.anchor_score[index] + score[index]);
            rss -= term;
            scale += term.abs();
        }
        if !rss.is_finite() {
            return Err(StatsError::Backend(
                "non-finite residual sum of squares from the retained factor".into(),
            ));
        }
        if rss < 0.0 {
            if rss < -RESIDUAL_SUM_ROUNDING * scale.max(1.0) {
                return Err(StatsError::Backend(
                    "negative residual sum of squares from the retained factor".into(),
                ));
            }
            return Ok(0.0);
        }
        Ok(rss)
    }

    /// Per-row residuals `y − xβ` at the retained solution.
    ///
    /// The summary stores only the anchor residual products, so this reads the
    /// rows again. It does not update the factor.
    ///
    /// # Errors
    ///
    /// The pass does not match the retained sample, or a row is non-finite.
    #[must_use = "a refused residual pass is not a residual vector"]
    pub fn residual_pass(&self, design: SelectedDesign<'_>) -> Result<Vec<f64>, StatsError> {
        self.same_sample(design)?;
        validate(design)?;
        let mut residuals = vec![0.0; self.nrows];
        for (slot, &row) in design.selected.iter().enumerate() {
            residuals[slot] = design.outcome[row] - self.predict_row(design, row)?;
        }
        Ok(residuals)
    }

    /// Diagonal of the weighted hat matrix.
    ///
    /// Each entry solves the retained Gram factor against that row. The inverse
    /// is not formed or stored. A numerically singular factor has no diagonal.
    ///
    /// # Errors
    ///
    /// The pass does not match the retained sample, a row is non-finite, or the
    /// Gram factor is absent.
    #[must_use = "a refused leverage pass is not a hat diagonal"]
    pub fn leverage_pass(&self, design: SelectedDesign<'_>) -> Result<Vec<f64>, StatsError> {
        self.same_sample(design)?;
        validate(design)?;
        let factor = self.require_gram_factor()?;
        let mut row = vec![0.0; self.ncols];
        let mut leverage = Vec::with_capacity(self.nrows);
        for &index in design.selected {
            copy_row(design, index, &mut row);
            let solved = chol_solve(factor, self.ncols, &row).ok_or_else(|| {
                StatsError::Backend("retained gram factor rejected a leverage row".into())
            })?;
            let mut quad = 0.0;
            for (value, solved_value) in row.iter().zip(&solved) {
                quad += value * solved_value;
            }
            let hat = row_weight(design, index)? * quad;
            if !hat.is_finite() {
                return Err(StatsError::Backend("non-finite leverage".into()));
            }
            leverage.push(hat);
        }
        Ok(leverage)
    }

    /// Cook's distance from a residual pass and a leverage pass.
    ///
    /// Both scans run here. The distances are not stored on the factor. The
    /// scale is the anchor residual sum of squares divided by `nrows − rank`.
    ///
    /// # Errors
    ///
    /// The residual or leverage pass fails, the rank leaves no residual degree
    /// of freedom, or a leverage is saturated at one.
    #[must_use = "a refused influence pass is not a Cook distance"]
    pub fn influence_pass(&self, design: SelectedDesign<'_>) -> Result<Vec<f64>, StatsError> {
        let residuals = self.residual_pass(design)?;
        let leverage = self.leverage_pass(design)?;
        let df = self.nrows.saturating_sub(self.solver_rank);
        if df == 0 {
            return Err(StatsError::Shape { message: "influence needs more rows than the rank" });
        }
        let scale = self.anchor_rss / (df as f64);
        if scale <= 0.0 || !scale.is_finite() {
            if residuals.iter().all(|residual| residual.abs() <= f64::EPSILON) {
                return Ok(vec![0.0; residuals.len()]);
            }
            return Err(StatsError::Shape { message: "influence scale is not positive" });
        }
        let mut cook = Vec::with_capacity(residuals.len());
        for (slot, (residual, hat)) in residuals.iter().zip(&leverage).enumerate() {
            if !hat.is_finite() || *hat > 1.0 || (1.0 - hat) <= 64.0 * f64::EPSILON {
                return Err(StatsError::Shape { message: "influence is undefined at leverage 1" });
            }
            let sqrt_w = match design.weights {
                Some(weights) => row_weight_sqrt(weights, design.selected[slot])?,
                None => 1.0,
            };
            let weighted = residual * sqrt_w;
            let distance = weighted * weighted / (self.ncols as f64 * scale) * hat
                / ((1.0 - hat) * (1.0 - hat));
            if !distance.is_finite() {
                return Err(StatsError::Backend("non-finite influence".into()));
            }
            cook.push(distance);
        }
        Ok(cook)
    }

    /// Condition of the retained factor.
    #[must_use]
    pub const fn condition(&self) -> FactorCondition {
        self.condition
    }

    /// Whether the Gram Cholesky is retained. False when the design is numerically singular.
    #[must_use]
    pub const fn gram_factor_retained(&self) -> bool {
        self.gram_factor.is_some()
    }

    /// Builds of this summary. Queries leave it unchanged.
    #[must_use]
    pub const fn factor_builds(&self) -> u64 {
        self.factor_builds
    }

    /// Times the solver wrote the gathered design.
    #[must_use]
    pub const fn gather_writes(&self) -> u64 {
        self.workspace.gather_writes()
    }

    /// Times the solver replaced its scratch buffer.
    #[must_use]
    pub const fn scratch_allocations(&self) -> u64 {
        self.workspace.scratch_allocations()
    }

    /// Owned column-major layouts inside the solver. One after a successful retain.
    #[must_use]
    pub const fn layout_copies(&self) -> u64 {
        self.workspace.layout_copies()
    }

    /// Whether the gathered solve used the minimum-norm SVD of that design.
    #[must_use]
    pub const fn used_min_norm_fallback(&self) -> bool {
        self.workspace.used_min_norm_fallback()
    }

    fn require_gram_factor(&self) -> Result<&[f64], StatsError> {
        self.gram_factor.as_deref().ok_or(match self.condition {
            FactorCondition::NumericallySingular { rank, ncols } => {
                StatsError::RankDeficient { rank, ncols }
            }
            FactorCondition::Stable { .. } | FactorCondition::IllConditioned { .. } => {
                StatsError::Backend("gram factor was not retained".into())
            }
        })
    }

    fn same_sample(&self, design: SelectedDesign<'_>) -> Result<(), StatsError> {
        if design.ncols == self.ncols && design.selected.len() == self.nrows {
            Ok(())
        } else {
            Err(StatsError::Shape { message: "diagnostic pass does not match the retained sample" })
        }
    }

    fn score(&self, beta: &[f64]) -> Result<Vec<f64>, StatsError> {
        if beta.len() != self.ncols {
            return Err(StatsError::Shape { message: "beta length != ncols" });
        }
        if !finite_slice(beta) {
            return Err(StatsError::Shape { message: "non-finite coefficient" });
        }
        let p = self.ncols;
        let mut score = vec![0.0; p];
        for (row, score_row) in score.iter_mut().enumerate() {
            let mut acc = self.anchor_score[row];
            for (col, &gram) in self.gram[row * p..(row + 1) * p].iter().enumerate() {
                acc -= gram * (beta[col] - self.anchor[col]);
            }
            *score_row = acc;
        }
        if !finite_slice(&score) {
            return Err(StatsError::Backend("non-finite score from the retained factor".into()));
        }
        Ok(score)
    }

    fn kick(
        &self,
        momentum: &mut [f64],
        position: &[f64],
        sigma_squared: f64,
        half_step: f64,
    ) -> Result<(), StatsError> {
        let score = self.score(position)?;
        for (impulse, force) in momentum.iter_mut().zip(&score) {
            *impulse += half_step * (force / sigma_squared);
        }
        Ok(())
    }

    fn predict_row(&self, design: SelectedDesign<'_>, row: usize) -> Result<f64, StatsError> {
        let mut predicted = 0.0;
        for (column, coefficient) in self.anchor.iter().enumerate() {
            predicted += design.x_colmajor[column * design.source_rows + row] * coefficient;
        }
        if predicted.is_finite() {
            Ok(predicted)
        } else {
            Err(StatsError::Backend("non-finite fitted value".into()))
        }
    }
}

/// One worker and an unbounded scratch budget.
///
/// Larger QR designs on this host are faster with one worker. That record is
/// not recomputed here.
fn one_worker_scope() -> FaerScope {
    FaerScope::try_new(1).unwrap_or_else(|_| unreachable!("one worker is a scope"))
}

fn require_sigma(sigma_squared: f64) -> Result<(), StatsError> {
    if sigma_squared.is_finite() && sigma_squared > 0.0 {
        Ok(())
    } else {
        Err(StatsError::Shape { message: "sigma^2 must be positive and finite" })
    }
}

fn finite_slice(values: &[f64]) -> bool {
    values.iter().all(|value| value.is_finite())
}

fn validate(design: SelectedDesign<'_>) -> Result<(), StatsError> {
    if design.selected.is_empty() || design.ncols == 0 {
        return Err(StatsError::Shape { message: "empty selected design" });
    }
    if design.outcome.len() != design.source_rows {
        return Err(StatsError::Shape { message: "y length != source rows" });
    }
    let Some(cells) = design.source_rows.checked_mul(design.ncols) else {
        return Err(StatsError::Shape { message: "X buffer too short" });
    };
    if design.x_colmajor.len() < cells {
        return Err(StatsError::Shape { message: "X buffer too short" });
    }
    if let Some(weights) = design.weights {
        if weights.len() != design.source_rows {
            return Err(StatsError::Shape { message: "weights length != source rows" });
        }
    }
    let mut n_eff = 0.0;
    for &row in design.selected {
        if row >= design.source_rows {
            return Err(StatsError::Shape { message: "selected row out of range" });
        }
        let weight = row_weight(design, row)?;
        if !design.outcome[row].is_finite() {
            return Err(StatsError::Shape { message: "non-finite design row" });
        }
        for column in 0..design.ncols {
            if !design.x_colmajor[column * design.source_rows + row].is_finite() {
                return Err(StatsError::Shape { message: "non-finite design row" });
            }
        }
        n_eff += weight;
    }
    if n_eff <= 0.0 || !n_eff.is_finite() {
        return Err(StatsError::Shape { message: "non-positive effective sample size" });
    }
    Ok(())
}

fn row_weight(design: SelectedDesign<'_>, row: usize) -> Result<f64, StatsError> {
    let weight = design.weights.map_or(1.0, |weights| weights[row]);
    if !weight.is_finite() {
        return Err(StatsError::Shape { message: "non-finite design row" });
    }
    if weight < 0.0 {
        return Err(StatsError::Shape { message: "negative weight" });
    }
    Ok(weight)
}

fn row_weight_sqrt(weights: &[f64], row: usize) -> Result<f64, StatsError> {
    let weight = weights[row];
    if !weight.is_finite() {
        return Err(StatsError::Shape { message: "non-finite design row" });
    }
    if weight < 0.0 {
        return Err(StatsError::Shape { message: "negative weight" });
    }
    Ok(weight.sqrt())
}

fn materialize(design: SelectedDesign<'_>) -> Result<(Vec<f64>, Vec<f64>), StatsError> {
    validate(design)?;
    let nrows = design.selected.len();
    let ncols = design.ncols;
    let Some(cells) = nrows.checked_mul(ncols) else {
        return Err(StatsError::Shape { message: "selected design overflows" });
    };
    let mut scaled = vec![0.0; cells];
    let mut outcome = vec![0.0; nrows];
    for (slot, &row) in design.selected.iter().enumerate() {
        let sqrt_w = design.weights.map_or(1.0, |weights| weights[row].sqrt());
        outcome[slot] = design.outcome[row] * sqrt_w;
        for column in 0..ncols {
            let value = design.x_colmajor[column * design.source_rows + row];
            scaled[column * nrows + slot] = value * sqrt_w;
        }
    }
    Ok((scaled, outcome))
}

fn weighted_gram(scaled: &[f64], nrows: usize, ncols: usize) -> Result<Vec<f64>, StatsError> {
    let Some(gram_len) = ncols.checked_mul(ncols) else {
        return Err(StatsError::Shape { message: "selected design overflows" });
    };
    let mut gram = vec![0.0; gram_len];
    form_xtx(scaled, nrows, ncols, &mut gram);
    if finite_slice(&gram) { Ok(gram) } else { Err(StatsError::Backend("non-finite gram".into())) }
}

fn copy_row(design: SelectedDesign<'_>, row: usize, into: &mut [f64]) {
    for (column, value) in into.iter_mut().enumerate() {
        *value = design.x_colmajor[column * design.source_rows + row];
    }
}

fn cross_score(design: &[f64], nrows: usize, ncols: usize, residuals: &[f64]) -> Vec<f64> {
    debug_assert_eq!(residuals.len(), nrows);
    debug_assert!(design.len() >= nrows.saturating_mul(ncols));
    let mut score = vec![0.0; ncols];
    for (column, score_col) in score.iter_mut().enumerate() {
        let mut acc = 0.0;
        let column_values = &design[column * nrows..(column + 1) * nrows];
        for (value, residual) in column_values.iter().zip(residuals) {
            acc += value * residual;
        }
        *score_col = acc;
    }
    score
}

fn classify(
    ncols: usize,
    rank: usize,
    rcond: Option<f64>,
    gram: &[f64],
) -> (Option<Vec<f64>>, FactorCondition) {
    if rank < ncols {
        return (None, FactorCondition::NumericallySingular { rank, ncols });
    }
    let mut factor = vec![0.0; ncols.saturating_mul(ncols)];
    let refused_pivot = match cholesky_spd_pivots_into(gram, ncols, &mut factor) {
        Ok(_) => None,
        Err(refusal) => Some(match refusal.error {
            CholeskyError::NotPositiveDefinite { index } => index,
            CholeskyError::BufferTooShort => 0,
        }),
    };
    if let Some(pivot) = refused_pivot {
        return (None, FactorCondition::NumericallySingular { rank: pivot, ncols });
    }
    match rcond {
        Some(rcond) if rcond.is_finite() && rcond >= ILL_CONDITIONED_RCOND => {
            (Some(factor), FactorCondition::Stable { rcond })
        }
        Some(rcond) if rcond.is_finite() && rcond > 0.0 => {
            (Some(factor), FactorCondition::IllConditioned { rcond })
        }
        _ => (None, FactorCondition::NumericallySingular { rank: 0, ncols }),
    }
}

#[cfg(test)]
mod tests {
    use super::{FactorCondition, HmcState, ILL_CONDITIONED_RCOND, RetainedFactor, SelectedDesign};
    use crate::error::StatsError;
    use crate::faer_backend::FaerBackend;
    use crate::faer_scope::FaerScope;

    fn rows(n: usize) -> Vec<usize> {
        (0..n).collect()
    }

    fn design<'a>(
        x: &'a [f64],
        n: usize,
        p: usize,
        y: &'a [f64],
        selected: &'a [usize],
    ) -> SelectedDesign<'a> {
        SelectedDesign {
            x_colmajor: x,
            source_rows: n,
            ncols: p,
            outcome: y,
            weights: None,
            selected,
        }
    }

    fn line(n: usize) -> (Vec<f64>, Vec<f64>) {
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for row in 0..n {
            x[row] = 1.0;
            x[n + row] = row as f64;
            y[row] = 3.0 + 4.0 * (row as f64);
        }
        (x, y)
    }

    fn close(left: f64, right: f64) {
        assert!((left - right).abs() < 1e-8, "{left} vs {right}");
    }

    #[test]
    fn retained_factor_does_not_default_its_scope() {
        assert_no_default::<RetainedFactor>();
        assert_no_default::<FaerScope>();
    }

    #[test]
    fn known_line_serves_every_retained_query_from_one_factor() {
        let (x, y) = line(6);
        let selected = rows(6);
        let factor = RetainedFactor::retain(design(&x, 6, 2, &y, &selected)).unwrap();
        assert_eq!(factor.factor_builds(), 1);
        assert_eq!(factor.gather_writes(), 1);
        assert_eq!(factor.layout_copies(), 1);
        assert_eq!(factor.scratch_allocations(), 1);
        assert!(factor.gram_factor_retained());
        assert!(!factor.used_min_norm_fallback());
        assert!(matches!(factor.condition(), FactorCondition::Stable { .. }));

        let fit = factor.gaussian_estimate();
        close(fit.coefficients[0], 3.0);
        close(fit.coefficients[1], 4.0);
        assert_eq!(fit.rank, 2);
        assert!(fit.reference_rss < 1e-16);

        let stepped = factor.laplace_step(&[0.0, 0.0], 1.0).unwrap();
        close(stepped.coefficients[0], 3.0);
        close(stepped.coefficients[1], 4.0);
        assert!(matches!(stepped.condition, FactorCondition::Stable { .. }));

        let slope = factor.repeated_query(&[0.0, 1.0]).unwrap();
        close(slope.value, 4.0);
        let again = factor.gaussian_estimate();
        close(again.coefficients[0], 3.0);
        close(again.coefficients[1], 4.0);
        assert!(factor.residual_sum_of_squares(&fit.coefficients).unwrap() < 1e-16);

        assert_eq!(factor.factor_builds(), 1);
        assert_eq!(factor.gather_writes(), 1);
        assert_eq!(factor.scratch_allocations(), 1);
        assert_eq!(factor.layout_copies(), 1);
    }

    #[test]
    fn hand_leapfrog_matches_the_retained_hmc_step() {
        // One column of ones, outcome ones: β = 1, Gram = [2], score(0) = 2.
        // ε = 1/2, σ² = 1, identity mass, start at rest at 0:
        // p ← 1/2, q ← 1/4, force ← 2(1 − 1/4) = 3/2, p ← 1/2 + 3/8 = 7/8.
        let x = [1.0, 1.0];
        let y = [1.0, 1.0];
        let selected = rows(2);
        let factor = RetainedFactor::retain(design(&x, 2, 1, &y, &selected)).unwrap();
        let builds = factor.factor_builds();
        let writes = factor.gather_writes();
        let scratch = factor.scratch_allocations();
        let state = HmcState { position: vec![0.0], momentum: vec![0.0] };
        let stepped = factor.hmc_step(&state, 0.5, 1.0).unwrap();
        close(stepped.state.position[0], 0.25);
        close(stepped.state.momentum[0], 0.875);
        assert!(matches!(stepped.condition, FactorCondition::Stable { .. }));
        assert_eq!(factor.factor_builds(), builds);
        assert_eq!(factor.gather_writes(), writes);
        assert_eq!(factor.scratch_allocations(), scratch);
    }

    #[test]
    fn short_rank_uses_minimum_norm_of_the_gathered_design() {
        let n = 8usize;
        let column: Vec<f64> = (0..n).map(|index| (index as f64 * 0.7).sin()).collect();
        let mut x = column.clone();
        x.extend(column.iter().map(|value| 2.0 * value));
        let y: Vec<f64> = column.iter().map(|value| 3.0 * value).collect();
        let selected = rows(n);
        let factor = RetainedFactor::retain(design(&x, n, 2, &y, &selected)).unwrap();
        let backend = FaerBackend.min_norm_least_squares(&x, n, 2, &y).unwrap();
        assert!(factor.used_min_norm_fallback());
        assert!(!factor.gram_factor_retained());
        assert_eq!(factor.condition(), FactorCondition::NumericallySingular { rank: 1, ncols: 2 });
        assert_eq!(factor.gather_writes(), 1);
        assert_eq!(factor.layout_copies(), 1);
        let fit = factor.gaussian_estimate();
        assert_eq!(fit.rank, 1);
        close(fit.coefficients[0], 0.6);
        close(fit.coefficients[1], 1.2);
        for (left, right) in fit.coefficients.iter().zip(&backend.coefficients) {
            close(*left, *right);
        }
        let query = factor.repeated_query(&[1.0, 2.0]).unwrap();
        close(query.value, 3.0);
        assert_eq!(query.condition, FactorCondition::NumericallySingular { rank: 1, ncols: 2 });
        let refused = factor.laplace_step(&[0.0, 0.0], 1.0).unwrap_err();
        assert_eq!(refused, StatsError::RankDeficient { rank: 1, ncols: 2 });
        let state = HmcState { position: vec![0.0, 0.0], momentum: vec![0.0, 0.0] };
        assert_eq!(
            factor.hmc_step(&state, 0.1, 1.0).unwrap_err(),
            StatsError::RankDeficient { rank: 1, ncols: 2 }
        );
        assert!(factor.leverage_pass(design(&x, n, 2, &y, &selected)).is_err());
        let residuals = factor.residual_pass(design(&x, n, 2, &y, &selected)).unwrap();
        assert_eq!(residuals.len(), n);
        assert!(residuals.iter().all(|residual| residual.abs() < 1e-8));
        assert_eq!(factor.factor_builds(), 1);
    }

    #[test]
    fn ill_conditioned_stays_distinct_from_numerically_singular() {
        let n = 4usize;
        let mut x = vec![1.0; n * 2];
        x[n * 2 - 1] = 1.0 + 1e-5;
        let y = vec![1.0; n];
        let selected = rows(n);
        let ill = RetainedFactor::retain(design(&x, n, 2, &y, &selected)).unwrap();
        match ill.condition() {
            FactorCondition::IllConditioned { rcond } => {
                assert!(rcond > 0.0 && rcond < ILL_CONDITIONED_RCOND, "{rcond}");
            }
            other => panic!("expected an ill-conditioned factor, got {other:?}"),
        }
        assert!(ill.gram_factor_retained());
        assert!(!ill.used_min_norm_fallback());
        assert!(ill.laplace_step(&[0.0, 0.0], 1.0).is_ok());

        let column = [0.2, -0.4, 0.6, 0.1];
        let mut singular_x = column.to_vec();
        singular_x.extend(column.map(|value| 2.0 * value));
        let singular_y: Vec<f64> = column.map(|value| 3.0 * value).to_vec();
        let singular =
            RetainedFactor::retain(design(&singular_x, n, 2, &singular_y, &selected)).unwrap();
        assert_eq!(
            singular.condition(),
            FactorCondition::NumericallySingular { rank: 1, ncols: 2 }
        );
        assert!(!singular.gram_factor_retained());
        assert_ne!(ill.condition(), singular.condition());
        assert!(matches!(
            singular.laplace_step(&[0.0, 0.0], 1.0).unwrap_err(),
            StatsError::RankDeficient { .. }
        ));
    }

    #[test]
    fn reference_residual_holds_when_the_outcome_is_far_from_zero() {
        let n = 8usize;
        let (x, mut y) = line(n);
        // 2^40 is exact. Its square sits far enough above the residual that y'y cannot keep it.
        let offset = 2.0_f64.powi(40);
        for value in &mut y {
            *value += offset;
        }
        // A quarter-unit bump is a real residual. y'y at this offset cannot keep it.
        y[3] += 0.25;
        let selected = rows(n);
        let factor = RetainedFactor::retain(design(&x, n, 2, &y, &selected)).unwrap();
        let fit = factor.gaussian_estimate();
        let intercept_error = (fit.coefficients[0] - (offset + 3.0)).abs();
        assert!(intercept_error < 0.1, "{intercept_error}");
        let slope_error = (fit.coefficients[1] - 4.0).abs();
        assert!(slope_error < 1e-2, "{slope_error}");
        let reference_rss = fit.reference_rss;
        assert!(reference_rss < 1.0, "{reference_rss}");
        let intercept = fit.coefficients[0] - offset;
        let slope = fit.coefficients[1];
        let stable = |intercept_shift: f64, slope_shift: f64| {
            let mut sum = 0.0_f64;
            for row in 0..n {
                let covariate = row as f64;
                let bump = if row == 3 { 0.25 } else { 0.0 };
                let residual = (3.0 + 4.0 * covariate + bump)
                    - (intercept + intercept_shift)
                    - (slope + slope_shift) * covariate;
                sum += residual * residual;
            }
            sum
        };
        let at_anchor = factor.residual_sum_of_squares(&fit.coefficients).unwrap();
        let anchor_gap = (at_anchor - stable(0.0, 0.0)).abs();
        assert!(anchor_gap < 1e-4, "{at_anchor} gap {anchor_gap}");
        let mut perturbed = fit.coefficients.clone();
        perturbed[0] += 1e-3;
        perturbed[1] -= 1e-3;
        let expanded = factor.residual_sum_of_squares(&perturbed).unwrap();
        let expanded_gap = (expanded - stable(1e-3, -1e-3)).abs();
        assert!(expanded_gap < 1e-4, "{expanded} gap {expanded_gap}");
        let cancelled = cancelling_rss(&x, n, 2, &y, &fit.coefficients);
        assert!(reference_rss > 1e-2, "{reference_rss}");
        assert!(
            cancelled.abs() < reference_rss * 1e-3,
            "cancelling moment {cancelled} still tracked the residual {reference_rss}",
        );
    }

    #[test]
    fn leverage_and_influence_match_a_direct_hat_diagonal() {
        // x = [0, 1, 2], y = [0, 0, 3]. Hat diagonal 5/6, 1/3, 5/6.
        // Cook's distance 5/2, 1/4, 5/2.
        let n = 3usize;
        let x = [1.0, 1.0, 1.0, 0.0, 1.0, 2.0];
        let y = [0.0, 0.0, 3.0];
        let selected = rows(n);
        let sample = design(&x, n, 2, &y, &selected);
        let factor = RetainedFactor::retain(sample).unwrap();
        let fit = factor.gaussian_estimate();
        close(fit.coefficients[0], -0.5);
        close(fit.coefficients[1], 1.5);
        let residuals = factor.residual_pass(sample).unwrap();
        close(residuals[0], 0.5);
        close(residuals[1], -1.0);
        close(residuals[2], 0.5);
        let leverage = factor.leverage_pass(sample).unwrap();
        close(leverage[0], 5.0 / 6.0);
        close(leverage[1], 1.0 / 3.0);
        close(leverage[2], 5.0 / 6.0);
        let cook = factor.influence_pass(sample).unwrap();
        close(cook[0], 2.5);
        close(cook[1], 0.25);
        close(cook[2], 2.5);
        assert_eq!(factor.factor_builds(), 1);
        assert_eq!(factor.gather_writes(), 1);
    }

    #[test]
    fn weighted_column_matches_the_normal_equation() {
        let x = [1.0, 1.0, 1.0];
        let y = [0.0, 0.0, 3.0];
        let weights = [1.0, 1.0, 100.0];
        let selected = rows(3);
        let sample = SelectedDesign {
            x_colmajor: &x,
            source_rows: 3,
            ncols: 1,
            outcome: &y,
            weights: Some(&weights),
            selected: &selected,
        };
        let factor = RetainedFactor::retain(sample).unwrap();
        let fit = factor.gaussian_estimate();
        close(fit.coefficients[0], 300.0 / 102.0);
        let leverage = factor.leverage_pass(sample).unwrap();
        close(leverage[0], 1.0 / 102.0);
        close(leverage[1], 1.0 / 102.0);
        close(leverage[2], 100.0 / 102.0);
        assert_eq!(factor.gather_writes(), 1);
        assert!(factor.gram_factor_retained());
    }

    #[test]
    fn selection_keeps_the_line_and_drops_the_other_row() {
        let n = 5usize;
        let (x, mut y) = line(n);
        y[4] = 1_000.0;
        let selected = [0usize, 1, 2, 3];
        let factor = RetainedFactor::retain(design(&x, n, 2, &y, &selected)).unwrap();
        let fit = factor.gaussian_estimate();
        close(fit.coefficients[0], 3.0);
        close(fit.coefficients[1], 4.0);
        assert_eq!(factor.gather_writes(), 1);
        assert_eq!(factor.layout_copies(), 1);
    }

    #[test]
    fn refused_designs_do_not_retain_a_factor() {
        let x = [1.0, 1.0, 0.0, 1.0];
        let y = [1.0, 2.0];
        let selected = rows(2);
        let empty = design(&x, 2, 2, &y, &[]);
        assert_eq!(
            RetainedFactor::retain(empty).unwrap_err(),
            StatsError::Shape { message: "empty selected design" }
        );
        let weights = [1.0, -1.0];
        let negative = SelectedDesign {
            x_colmajor: &x,
            source_rows: 2,
            ncols: 2,
            outcome: &y,
            weights: Some(&weights),
            selected: &selected,
        };
        assert_eq!(
            RetainedFactor::retain(negative).unwrap_err(),
            StatsError::Shape { message: "negative weight" }
        );
        let non_finite = [1.0, f64::NAN];
        let broken = SelectedDesign {
            x_colmajor: &x,
            source_rows: 2,
            ncols: 2,
            outcome: &non_finite,
            weights: None,
            selected: &selected,
        };
        assert_eq!(
            RetainedFactor::retain(broken).unwrap_err(),
            StatsError::Shape { message: "non-finite design row" }
        );
        let wide = [1.0, 1.0, 0.0, 1.0, 2.0, 3.0];
        let under = design(&wide, 2, 3, &y, &selected);
        assert_eq!(
            RetainedFactor::retain(under).unwrap_err(),
            StatsError::Shape { message: "nrows < ncols" }
        );
    }

    fn cancelling_rss(x: &[f64], nrows: usize, ncols: usize, y: &[f64], beta: &[f64]) -> f64 {
        let mut yty = 0.0;
        for value in y {
            yty += value * value;
        }
        let mut xtx = vec![0.0; ncols * ncols];
        let mut xty = vec![0.0; ncols];
        for column in 0..ncols {
            for row in 0..nrows {
                xty[column] += x[column * nrows + row] * y[row];
            }
            for other in 0..ncols {
                let mut acc = 0.0;
                for row in 0..nrows {
                    acc += x[column * nrows + row] * x[other * nrows + row];
                }
                xtx[column * ncols + other] = acc;
            }
        }
        let mut quad = 0.0;
        let mut cross = 0.0;
        for left in 0..ncols {
            cross += beta[left] * xty[left];
            for right in 0..ncols {
                quad += beta[left] * xtx[left * ncols + right] * beta[right];
            }
        }
        yty - 2.0 * cross + quad
    }

    fn assert_no_default<T: ?Sized>() {
        struct Marker;
        trait Probe<A> {
            fn probe() {}
        }
        impl<T: ?Sized> Probe<()> for T {}
        #[allow(dead_code)]
        impl<T: Default> Probe<Marker> for T {}
        <T as Probe<_>>::probe();
    }
}
