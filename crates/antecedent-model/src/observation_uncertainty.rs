//! Parameter uncertainty of an observation-conditioned query by retained-data
//! redraw.
//!
//! A conditional probability or conditional mean computed from fitted
//! mechanisms is a function of the fitted parameters, and the parameters were
//! estimated from a sample. This module keeps the sufficient statistics the
//! fit was made from and redraws the parameters from their conjugate
//! posterior, re-executes the composed model under each draw and returns the
//! queried quantities as aligned draws: every row is one joint realization of
//! all quantities, because one parameter draw produced them all.
//!
//! - **Linear-Gaussian mechanisms**: the retained statistics are `X'X`, `X'y`,
//!   `y'y` and the row count of the regression of the mechanism's variable on
//!   its declared parents (with an intercept). The posterior is the conjugate
//!   Normal-Inverse-Gamma: `sigma2 ~ InvGamma(a_n, b_n)` and
//!   `beta | sigma2 ~ N(m_n, sigma2 Lambda_n^-1)`, so every linear functional
//!   of the coefficients has a Student-t marginal with `2 a_n` degrees of
//!   freedom.
//! - **Finite tables**: the retained statistics are the counts of each parent
//!   configuration. The posterior of each conditional row is an independent
//!   Dirichlet over the counts plus the prior concentration.
//!
//! Mechanisms fitted on the same complete rows have posteriors that factor over
//! the mechanisms (the likelihood is the Markov factorization and the priors
//! are independent), so the draws of different mechanisms are independent. A
//! mechanism with no retained data is kept at its declared numbers and is not
//! redrawn; a mechanism with declared shared latents cannot be redrawn from a
//! regression on its parents and is refused.
//!
//! # Boundary
//!
//! Draws are exact posterior draws given the stated priors and likelihood, and
//! they carry no coverage statement: a spread of the draws is labelled
//! `uncalibrated_spread` by the caller unless a measured coverage record exists
//! at the coordinate. Draws are a deterministic function of the retained
//! statistics, the priors, the seed and the draw count.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::needless_range_loop,
    reason = "dense regression statistics index several row-major buffers by the same position"
)]

use antecedent_core::{CausalRng, ExecutionContext, mix64};
use antecedent_kernels::{sample_gamma, sample_inv_gamma, standard_normal};
use antecedent_stats::invert_square;

use crate::compose::gaussian::cholesky_lower;
use crate::compose::{
    ComposedFinite, ComposedLinearGaussian, ComposedModelSpec, MechanismForm, TabularRow,
};
use crate::observation_query::{
    ObservationQueryError, condition_after_intervention, condition_gaussian,
};

/// Fewest posterior draws a redraw may take.
pub const MIN_REDRAWS: u32 = 2;
/// Most posterior draws a redraw may take.
pub const MAX_REDRAWS: u32 = 20_000;
/// Most coefficients (including the intercept) one retained regression may have.
pub const MAX_RETAINED_REGRESSORS: usize = 32;
/// Most quantities one redraw may return.
pub const MAX_REDRAW_QUANTITIES: usize = 256;

/// Functional id of the conditional mean of a continuous outcome.
pub const CONDITIONAL_MEAN: &str = "conditional_mean";
/// Functional id of the conditional variance of a continuous outcome.
pub const CONDITIONAL_VARIANCE: &str = "conditional_variance";

/// Priors of the redraw.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RedrawPrior {
    /// Dirichlet concentration added to every cell count; finite and positive.
    pub dirichlet_alpha: f64,
    /// Prior precision of every regression coefficient relative to the error
    /// variance; finite and positive.
    pub coefficient_precision: f64,
    /// Inverse-Gamma shape of the error variance; finite and positive.
    pub variance_shape: f64,
    /// Inverse-Gamma scale of the error variance; finite and positive.
    pub variance_scale: f64,
}

impl Default for RedrawPrior {
    /// A weak prior: uniform Dirichlet, near-flat coefficients and a
    /// near-Jeffreys variance.
    fn default() -> Self {
        Self {
            dirichlet_alpha: 1.0,
            coefficient_precision: 1e-6,
            variance_shape: 1e-3,
            variance_scale: 1e-3,
        }
    }
}

impl RedrawPrior {
    fn check(&self) -> Result<(), ObservationQueryError> {
        let positive = |v: f64| v.is_finite() && v > 0.0;
        if positive(self.dirichlet_alpha)
            && positive(self.coefficient_precision)
            && positive(self.variance_shape)
            && positive(self.variance_scale)
        {
            Ok(())
        } else {
            Err(ObservationQueryError::Invalid(
                "every redraw prior parameter must be finite and positive",
            ))
        }
    }
}

/// Sufficient statistics of the regression of one variable on its parents.
#[derive(Clone, Debug, PartialEq)]
pub struct GaussianSufficient {
    /// The variable the mechanism produces.
    pub variable: String,
    /// The regressors after the intercept, in the order of the statistics.
    pub parents: Vec<String>,
    /// Row-major `(1 + parents) x (1 + parents)` Gram matrix `X'X` with the
    /// intercept column first.
    pub gram: Vec<f64>,
    /// `X'y`, intercept first.
    pub xty: Vec<f64>,
    /// `y'y`.
    pub yty: f64,
    /// Rows the statistics were computed from.
    pub rows: u64,
}

impl GaussianSufficient {
    /// The statistics of `y` regressed on an intercept and the named columns.
    ///
    /// # Errors
    /// No rows, columns of unequal length, a non-finite value or too many
    /// regressors.
    pub fn from_columns(
        variable: &str,
        parents: &[(&str, &[f64])],
        y: &[f64],
    ) -> Result<Self, ObservationQueryError> {
        let n = y.len();
        let p = parents.len() + 1;
        if n == 0
            || p > MAX_RETAINED_REGRESSORS
            || parents.iter().any(|(_, c)| c.len() != n)
            || y.iter().chain(parents.iter().flat_map(|(_, c)| c.iter())).any(|v| !v.is_finite())
        {
            return Err(ObservationQueryError::Invalid(
                "regression columns must be finite, equal in length and non-empty",
            ));
        }
        let column = |j: usize, r: usize| if j == 0 { 1.0 } else { parents[j - 1].1[r] };
        let mut gram = vec![0.0_f64; p * p];
        let mut xty = vec![0.0_f64; p];
        for r in 0..n {
            for a in 0..p {
                let xa = column(a, r);
                xty[a] += xa * y[r];
                for b in 0..p {
                    gram[a * p + b] += xa * column(b, r);
                }
            }
        }
        Ok(Self {
            variable: variable.to_owned(),
            parents: parents.iter().map(|(name, _)| (*name).to_owned()).collect(),
            gram,
            xty,
            yty: y.iter().map(|v| v * v).sum(),
            rows: n as u64,
        })
    }
}

/// Counts of one parent configuration of a finite mechanism.
#[derive(Clone, Debug, PartialEq)]
pub struct TableCountRow {
    /// `(parent variable, level)` for every parent.
    pub parent_levels: Vec<(String, String)>,
    /// `(level, count)` for every level of the produced variable.
    pub counts: Vec<(String, u64)>,
}

/// The counts a finite mechanism was fitted from.
#[derive(Clone, Debug, PartialEq)]
pub struct TableCounts {
    /// The variable the mechanism produces.
    pub variable: String,
    /// One row per parent configuration of the mechanism.
    pub rows: Vec<TableCountRow>,
}

/// The data a mechanism's numbers were fitted from, retained for redraw.
#[derive(Clone, Debug, PartialEq)]
pub enum RetainedMechanism {
    /// A linear-Gaussian regression.
    Gaussian(GaussianSufficient),
    /// A finite conditional table.
    Table(TableCounts),
}

impl RetainedMechanism {
    /// The variable whose mechanism is redrawn.
    #[must_use]
    pub fn variable(&self) -> &str {
        match self {
            Self::Gaussian(s) => &s.variable,
            Self::Table(t) => &t.variable,
        }
    }
}

/// The quantity redrawn: a conditional functional of the composed model.
#[derive(Clone, Debug, PartialEq)]
pub enum RedrawTarget {
    /// Conditional mean and variance of a continuous outcome.
    Gaussian {
        /// Outcome variable.
        outcome: String,
        /// Hard interventions `(variable, value)`.
        interventions: Vec<(String, f64)>,
        /// Observed `(variable, value)`.
        observed: Vec<(String, f64)>,
    },
    /// The conditional probability of every level of a finite outcome.
    Finite {
        /// Outcome variable.
        outcome: String,
        /// Interventions `(variable, level)`.
        interventions: Vec<(String, String)>,
        /// Observed `(variable, level)`.
        observed: Vec<(String, String)>,
    },
}

/// Aligned posterior draws of the queried quantities.
#[derive(Clone, Debug, PartialEq)]
pub struct ParameterDraws {
    /// Functional ids of the quantities, in column order.
    pub quantities: Vec<String>,
    /// Number of draws.
    pub draws: usize,
    /// Draw-major values: draw `i` occupies `i * quantities.len()..`.
    pub values: Vec<f64>,
    /// The quantities under the composition's own (fitted) numbers.
    pub point: Vec<f64>,
}

impl ParameterDraws {
    /// Number of quantities per draw.
    #[must_use]
    pub fn width(&self) -> usize {
        self.quantities.len()
    }

    /// The draws of one quantity.
    #[must_use]
    pub fn column(&self, quantity: usize) -> Vec<f64> {
        let w = self.width();
        (0..self.draws).map(|d| self.values[d * w + quantity]).collect()
    }

    /// Mean of one quantity over the draws.
    #[must_use]
    pub fn mean(&self, quantity: usize) -> f64 {
        self.column(quantity).iter().sum::<f64>() / self.draws as f64
    }

    /// Sample standard deviation of one quantity over the draws.
    #[must_use]
    pub fn standard_deviation(&self, quantity: usize) -> f64 {
        let column = self.column(quantity);
        let m = column.iter().sum::<f64>() / column.len() as f64;
        let ss: f64 = column.iter().map(|v| (v - m) * (v - m)).sum();
        (ss / (column.len() as f64 - 1.0)).sqrt()
    }
}

struct NigPosterior {
    mean: Vec<f64>,
    factor: Vec<f64>,
    shape: f64,
    scale: f64,
}

fn nig(
    stats: &GaussianSufficient,
    prior: &RedrawPrior,
) -> Result<NigPosterior, ObservationQueryError> {
    let p = stats.parents.len() + 1;
    if p > MAX_RETAINED_REGRESSORS
        || stats.gram.len() != p * p
        || stats.xty.len() != p
        || stats.rows == 0
        || !stats.yty.is_finite()
        || !stats.gram.iter().chain(&stats.xty).all(|v| v.is_finite())
    {
        return Err(ObservationQueryError::Invalid(
            "retained regression statistics must be finite with a (1 + parents) square Gram matrix and at least one row",
        ));
    }
    let mut precision = stats.gram.clone();
    for j in 0..p {
        precision[j * p + j] += prior.coefficient_precision;
    }
    let inverse = invert_square(&precision, p).ok_or(ObservationQueryError::Invalid(
        "the posterior precision of a retained regression is singular",
    ))?;
    let mean: Vec<f64> =
        (0..p).map(|i| (0..p).map(|j| inverse[i * p + j] * stats.xty[j]).sum::<f64>()).collect();
    let explained: f64 = mean.iter().zip(&stats.xty).map(|(m, c)| m * c).sum();
    let factor = cholesky_lower(&inverse, p).ok_or(ObservationQueryError::Invalid(
        "the posterior covariance of a retained regression is not positive definite",
    ))?;
    Ok(NigPosterior {
        mean,
        factor,
        shape: prior.variance_shape + stats.rows as f64 / 2.0,
        scale: prior.variance_scale + 0.5 * (stats.yty - explained).max(0.0),
    })
}

impl NigPosterior {
    /// One draw `(coefficients, sigma)`, intercept first.
    fn draw(&self, rng: &mut CausalRng) -> Result<(Vec<f64>, f64), ObservationQueryError> {
        let p = self.mean.len();
        let sigma2 = sample_inv_gamma(self.shape, self.scale, rng);
        if !(sigma2.is_finite() && sigma2 > 0.0) {
            return Err(ObservationQueryError::Invalid(
                "a variance draw underflowed; the retained regression is degenerate",
            ));
        }
        let sigma = sigma2.sqrt();
        let z: Vec<f64> = (0..p).map(|_| standard_normal(rng)).collect();
        let beta: Vec<f64> = (0..p)
            .map(|i| {
                let shock: f64 = (0..=i).map(|k| self.factor[i * p + k] * z[k]).sum();
                self.mean[i] + sigma * shock
            })
            .collect();
        Ok((beta, sigma))
    }
}

/// A redrawable mechanism, resolved against the specification.
enum Plan {
    Gaussian { index: usize, parents: Vec<String>, posterior: NigPosterior },
    Table { index: usize, rows: Vec<(usize, Vec<(String, f64)>)> },
}

fn plan(
    spec: &ComposedModelSpec,
    retained: &[RetainedMechanism],
    prior: &RedrawPrior,
) -> Result<Vec<Plan>, ObservationQueryError> {
    prior.check()?;
    if retained.is_empty() {
        return Err(ObservationQueryError::Invalid(
            "a redraw needs the retained data of at least one mechanism",
        ));
    }
    let mut plans = Vec::with_capacity(retained.len());
    let mut seen = Vec::new();
    for item in retained {
        let variable = item.variable();
        if seen.contains(&variable) {
            return Err(ObservationQueryError::RepeatedVariable(variable.to_owned()));
        }
        seen.push(variable);
        let Some(index) = spec.mechanisms.iter().position(|m| m.variable == variable) else {
            return Err(ObservationQueryError::UnknownVariable(variable.to_owned()));
        };
        let mechanism = &spec.mechanisms[index];
        match (item, &mechanism.form) {
            (
                RetainedMechanism::Gaussian(stats),
                MechanismForm::LinearGaussian { coefficients, .. },
            ) => {
                if !mechanism.latents.is_empty() {
                    return Err(ObservationQueryError::Invalid(
                        "a mechanism with declared shared latents cannot be redrawn from a regression on its parents",
                    ));
                }
                let mut declared: Vec<&str> =
                    coefficients.iter().map(|(n, _)| n.as_str()).collect();
                let mut retained_names: Vec<&str> =
                    stats.parents.iter().map(String::as_str).collect();
                declared.sort_unstable();
                retained_names.sort_unstable();
                if declared != retained_names {
                    return Err(ObservationQueryError::Invalid(
                        "the retained regressors must be exactly the declared parents of the mechanism",
                    ));
                }
                plans.push(Plan::Gaussian {
                    index,
                    parents: stats.parents.clone(),
                    posterior: nig(stats, prior)?,
                });
            }
            (RetainedMechanism::Table(counts), MechanismForm::Tabular { levels, rows }) => {
                let mut resolved = Vec::with_capacity(rows.len());
                for (row_index, row) in rows.iter().enumerate() {
                    let key = sorted_key(&row.parent_levels);
                    let Some(found) =
                        counts.rows.iter().find(|c| sorted_key(&c.parent_levels) == key)
                    else {
                        return Err(ObservationQueryError::Invalid(
                            "retained counts are missing a parent configuration of the table",
                        ));
                    };
                    let mut cells = Vec::with_capacity(levels.len());
                    for level in levels {
                        let Some((_, count)) = found.counts.iter().find(|(l, _)| l == level) else {
                            return Err(ObservationQueryError::Invalid(
                                "retained counts are missing a level of the table",
                            ));
                        };
                        cells.push((level.clone(), *count as f64 + prior.dirichlet_alpha));
                    }
                    resolved.push((row_index, cells));
                }
                plans.push(Plan::Table { index, rows: resolved });
            }
            _ => {
                return Err(ObservationQueryError::Invalid(
                    "the retained data is of a different family than the mechanism",
                ));
            }
        }
    }
    Ok(plans)
}

fn sorted_key(levels: &[(String, String)]) -> Vec<(String, String)> {
    let mut key = levels.to_vec();
    key.sort();
    key
}

fn draw_spec(
    base: &ComposedModelSpec,
    plans: &[Plan],
    rng: &mut CausalRng,
) -> Result<ComposedModelSpec, ObservationQueryError> {
    let mut spec = base.clone();
    for plan in plans {
        match plan {
            Plan::Gaussian { index, parents, posterior } => {
                let (beta, sigma) = posterior.draw(rng)?;
                let mechanism = &mut spec.mechanisms[*index];
                if let MechanismForm::LinearGaussian { intercept, sigma: s, coefficients } =
                    &mut mechanism.form
                {
                    *intercept = beta[0];
                    *s = sigma;
                    for (name, value) in coefficients.iter_mut() {
                        let position = parents
                            .iter()
                            .position(|p| p == name)
                            .ok_or(ObservationQueryError::Invalid("retained regressor missing"))?;
                        *value = beta[position + 1];
                    }
                }
            }
            Plan::Table { index, rows } => {
                let mechanism = &mut spec.mechanisms[*index];
                if let MechanismForm::Tabular { rows: table, .. } = &mut mechanism.form {
                    for (row_index, cells) in rows {
                        let gammas: Vec<f64> =
                            cells.iter().map(|(_, shape)| sample_gamma(*shape, 1.0, rng)).collect();
                        let total: f64 = gammas.iter().sum();
                        if !(total.is_finite() && total > 0.0) {
                            return Err(ObservationQueryError::Invalid(
                                "a Dirichlet draw underflowed; the retained counts are degenerate",
                            ));
                        }
                        // The probabilities are assigned by level name: the order of a row's
                        // cells is not part of the declaration.
                        let row: &mut TabularRow = &mut table[*row_index];
                        for (level, p) in row.probabilities.iter_mut() {
                            let Some(position) = cells.iter().position(|(l, _)| l == level) else {
                                return Err(ObservationQueryError::Invalid(
                                    "a table row names a level the retained counts do not",
                                ));
                            };
                            *p = gammas[position] / total;
                        }
                    }
                }
            }
        }
    }
    Ok(spec)
}

fn refs_real(pairs: &[(String, f64)]) -> Vec<(&str, f64)> {
    pairs.iter().map(|(n, v)| (n.as_str(), *v)).collect()
}

fn refs_level(pairs: &[(String, String)]) -> Vec<(&str, &str)> {
    pairs.iter().map(|(n, v)| (n.as_str(), v.as_str())).collect()
}

/// The quantity ids a target reports, in column order.
///
/// # Errors
/// An unknown outcome, or a specification of the other family.
pub fn target_quantities(
    spec: &ComposedModelSpec,
    target: &RedrawTarget,
) -> Result<Vec<String>, ObservationQueryError> {
    match target {
        RedrawTarget::Gaussian { outcome, .. } => {
            if spec.mechanisms.iter().any(|m| m.variable == *outcome) {
                Ok(vec![CONDITIONAL_MEAN.to_owned(), CONDITIONAL_VARIANCE.to_owned()])
            } else {
                Err(ObservationQueryError::UnknownVariable(outcome.clone()))
            }
        }
        RedrawTarget::Finite { outcome, .. } => {
            let Some(mechanism) = spec.mechanisms.iter().find(|m| m.variable == *outcome) else {
                return Err(ObservationQueryError::UnknownVariable(outcome.clone()));
            };
            let MechanismForm::Tabular { levels, .. } = &mechanism.form else {
                return Err(ObservationQueryError::Invalid(
                    "a finite target needs a finite outcome mechanism",
                ));
            };
            let mut sorted = levels.clone();
            sorted.sort();
            Ok(sorted.iter().map(|l| format!("probability:{l}")).collect())
        }
    }
}

fn evaluate(
    spec: &ComposedModelSpec,
    target: &RedrawTarget,
    ctx: &ExecutionContext,
) -> Result<Vec<f64>, ObservationQueryError> {
    let composition = |error: crate::compose::CompositionError| match error {
        crate::compose::CompositionError::Cancelled => ObservationQueryError::Cancelled,
        other => ObservationQueryError::Composition(other),
    };
    match target {
        RedrawTarget::Gaussian { outcome, interventions, observed } => {
            let model = ComposedLinearGaussian::compile(spec).map_err(composition)?;
            let answer = condition_gaussian(
                &model,
                outcome,
                &refs_real(interventions),
                &refs_real(observed),
                ctx,
            )?;
            Ok(vec![answer.mean, answer.variance])
        }
        RedrawTarget::Finite { outcome, interventions, observed } => {
            let model = ComposedFinite::compile(spec).map_err(composition)?;
            let answer = condition_after_intervention(
                &model,
                outcome,
                &refs_level(interventions),
                &refs_level(observed),
                ctx,
            )?;
            Ok(answer.levels.iter().map(|(_, p)| *p).collect())
        }
    }
}

/// Redraw the parameters of the retained mechanisms `draws` times and return
/// the queried conditional functional under each draw, as aligned draws.
///
/// The stream of draw `i` is a function of `seed` and `i` alone, so the result
/// is a deterministic function of the retained data, the priors, the seed and
/// the draw count, independent of how many threads or contexts execute it.
///
/// # Errors
/// A prior outside its range, retained data that does not match the
/// composition, a draw count outside `MIN_REDRAWS..=MAX_REDRAWS`, evidence of
/// probability zero under a draw, or cancellation.
pub fn redraw_conditional(
    spec: &ComposedModelSpec,
    target: &RedrawTarget,
    retained: &[RetainedMechanism],
    prior: &RedrawPrior,
    draws: u32,
    seed: u64,
    ctx: &ExecutionContext,
) -> Result<ParameterDraws, ObservationQueryError> {
    if !(MIN_REDRAWS..=MAX_REDRAWS).contains(&draws) {
        return Err(ObservationQueryError::Invalid(
            "the draw count must lie between the supported minimum and maximum",
        ));
    }
    let quantities = target_quantities(spec, target)?;
    if quantities.len() > MAX_REDRAW_QUANTITIES {
        return Err(ObservationQueryError::Invalid("too many quantities for one redraw"));
    }
    let plans = plan(spec, retained, prior)?;
    let point = evaluate(spec, target, ctx)?;
    let width = quantities.len();
    let mut values = Vec::with_capacity(draws as usize * width);
    for draw in 0..u64::from(draws) {
        if ctx.cancellation.is_cancelled() {
            return Err(ObservationQueryError::Cancelled);
        }
        let mut rng = CausalRng::from_seed(mix64(mix64(seed) ^ draw.wrapping_add(1)));
        let drawn = draw_spec(spec, &plans, &mut rng)?;
        let row = evaluate(&drawn, target, ctx)?;
        if row.len() != width || row.iter().any(|v| !v.is_finite()) {
            return Err(ObservationQueryError::Invalid(
                "a parameter draw produced a non-finite or misaligned quantity",
            ));
        }
        values.extend(row);
    }
    Ok(ParameterDraws { quantities, draws: draws as usize, values, point })
}
