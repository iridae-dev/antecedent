//! Plug-in estimator of the nonparametric front-door functional.
//!
//! For a mediator set `M` that satisfies the front-door criterion relative to `(T, Y)`
//! (see `antecedent_identify::frontdoor`), the interventional mean is
//!
//! `E[Y | do(T=a)] = Σ_m P(m | a) · Σ_{t'} E[Y | m, t'] · P(t')`,
//!
//! and the reported effect is the contrast `E[Y | do(T=active)] − E[Y | do(T=control)]`.
//! Writing `g(m) = Σ_{t'} μ(m, t') P(t')` with `μ(m, t') = E[Y | M=m, T=t']`, the outer sum
//! is `E[g(M) | T=a]`, so no mediator density is ever estimated: `g` is averaged over the
//! empirical mediator law of arm `a`.
//!
//! The treatment must be discrete (its observed levels are the arms, and both contrasted
//! levels must be observed). Two outcome models `μ` are available:
//!
//! - [`FrontDoorOutcomeModel::Saturated`]: every observed joint mediator value is a cell and
//!   `μ(m, t')` is the cell mean. Nothing is assumed about functional form. Each mediator
//!   value that occurs under a contrasted arm must occur in **every** arm, otherwise
//!   `μ(m, t')` has no data and the fit is refused rather than extrapolated.
//! - [`FrontDoorOutcomeModel::ArmLinear`]: within each arm `t'`, `μ(m, t') = β₀(t') + β(t')ᵀm`
//!   by OLS. Saturated in the treatment (so treatment–mediator interaction is free), linear
//!   and additive in the mediators within an arm, and extrapolated to the mediator mean of
//!   the contrasted arms. This restriction is recorded on the result.
//!
//! [`FrontDoorOutcomeModel::Auto`] picks `Saturated` when every mediator has at most
//! [`MAX_SATURATED_MEDIATOR_LEVELS`] distinct values and `ArmLinear` otherwise.
//!
//! # Standard error
//!
//! Both variants are smooth functions of arm frequencies, arm-wise mediator means and
//! arm-wise regression coefficients, so the plug-in is asymptotically linear. With
//! `ψ_a = E[Y | do(a)]`, `h_a(t') = E[μ(M, t') | T=a]` and `p_t = P(T=t)`, the influence
//! function of `ψ̂_a` is
//!
//! `φ_a = [h_a(T) − ψ_a] + 1{T=a}/p_a · [g(M) − ψ_a] + r_a`,
//!
//! where the outcome-model term is `r_a = P(M | a)/P(M | T) · (Y − μ(M, T))` for the
//! saturated model and `r_a = n_T · x̄_aᵀ (X_TᵀX_T)⁻¹ x · (Y − μ(M, T))` for the arm-linear
//! model (`x = [1, M]`, `x̄_a` its arm-`a` mean, `X_T` the design of the row's own arm). The
//! analytic SE is `sqrt(Σ_i (φ_active,i − φ_control,i)²) / n`; it is heteroskedasticity-robust
//! and assumes independent rows.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::similar_names)]

use std::collections::BTreeMap;
use std::sync::Arc;

use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
    AssumptionStatus, AverageEffectQuery, ExecutionContext, ParametricAssumption,
};
use antecedent_data::TabularData;
use antecedent_expr::IdentifiedEstimand;
use antecedent_stats::{
    DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace, form_xtx, invert_square,
};

use crate::adjustment::EffectEstimate;
use crate::error::EstimationError;
use crate::frontdoor::{PreparedFrontDoorProblem, prepare_frontdoor_problem};
use crate::overlap::OverlapPolicy;
use crate::util::{BootstrapSeResult, stats_err};

/// Largest number of distinct treatment values treated as arms.
pub const MAX_TREATMENT_LEVELS: usize = 32;
/// Largest per-mediator distinct-value count for which [`FrontDoorOutcomeModel::Auto`]
/// selects the saturated cell-mean model.
pub const MAX_SATURATED_MEDIATOR_LEVELS: usize = 16;

/// Stable id of the positivity requirement checked by the saturated model.
pub const SATURATED_ASSUMPTION_ID: &str = "frontdoor.functional.saturated_cells";
/// Stable id of the arm-linear outcome-model restriction.
pub const ARM_LINEAR_ASSUMPTION_ID: &str = "frontdoor.functional.arm_linear_outcome";

const ALGORITHM: &str = "frontdoor.functional";

/// Model for `μ(m, t') = E[Y | M=m, T=t']` inside the front-door functional.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum FrontDoorOutcomeModel {
    /// Saturated when every mediator is discrete, arm-linear otherwise.
    #[default]
    Auto,
    /// Cell means over every observed joint mediator value.
    Saturated,
    /// Per-arm OLS of `Y` on `[1, M₁…Mₖ]`.
    ArmLinear,
}

/// Plug-in estimator of the front-door functional. See the module docs.
#[derive(Clone, Debug)]
pub struct FrontDoorFunctional {
    /// Dense linear-algebra backend for the arm-linear outcome regressions.
    pub backend: FaerBackend,
    /// Bootstrap replicates (0 = skip bootstrap).
    pub bootstrap_replicates: u32,
    /// Overlap policy (must be [`OverlapPolicy::ExplicitOverride`]; there is no propensity).
    pub overlap: OverlapPolicy,
    /// Outcome model inside the functional.
    pub outcome_model: FrontDoorOutcomeModel,
}

impl Default for FrontDoorFunctional {
    fn default() -> Self {
        Self::new()
    }
}

impl FrontDoorFunctional {
    /// Default: 200 bootstrap replicates, explicit overlap override, automatic outcome model.
    #[must_use]
    pub fn new() -> Self {
        Self {
            backend: FaerBackend,
            bootstrap_replicates: 200,
            overlap: OverlapPolicy::ExplicitOverride,
            outcome_model: FrontDoorOutcomeModel::Auto,
        }
    }

    /// Set the number of bootstrap replicates (`0` reports only the influence-function SE).
    #[must_use]
    pub const fn with_bootstrap_replicates(mut self, replicates: u32) -> Self {
        self.bootstrap_replicates = replicates;
        self
    }

    /// Set the outcome model.
    #[must_use]
    pub const fn with_outcome_model(mut self, outcome_model: FrontDoorOutcomeModel) -> Self {
        self.outcome_model = outcome_model;
        self
    }

    /// Prepare the treatment/mediator/outcome columns (complete cases).
    ///
    /// # Errors
    ///
    /// Incompatible estimand or unsupported query options.
    pub fn prepare(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
    ) -> Result<PreparedFrontDoorProblem, EstimationError> {
        prepare_frontdoor_problem(data, estimand, query, self.overlap)
    }

    /// Fit the plug-in contrast with its influence-function SE and optional bootstrap.
    ///
    /// # Errors
    ///
    /// Non-discrete treatment, a contrasted level that is not observed, a mediator value
    /// without data in some arm (saturated), or a rank-deficient arm regression (arm-linear).
    pub fn fit(
        &self,
        problem: &PreparedFrontDoorProblem,
        ctx: &ExecutionContext,
        mut assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        let model = resolve_model(self.outcome_model, &problem.mediators);
        let mediators: Vec<&[f64]> = problem.mediators.iter().map(AsRef::as_ref).collect();
        let sample = Sample {
            treatment: &problem.treatment,
            mediators: &mediators,
            outcome: &problem.outcome,
        };
        let mut ols = LeastSquaresWorkspace::default();
        let fitted = contrast(
            &sample,
            problem.active,
            problem.control,
            model,
            &self.backend,
            &mut ols,
            true,
        )?;
        let influence = fitted.influence.unwrap_or_default();
        let n = problem.nrows as f64;
        let se_analytic = influence.iter().map(|d| d * d).sum::<f64>().sqrt() / n;

        assumptions.push(model_assumption(model));

        let boot = if self.bootstrap_replicates == 0 {
            None
        } else {
            Some(self.bootstrap_se(problem, model, ctx)?)
        };
        Ok(EffectEstimate::new(fitted.ate, se_analytic, assumptions, problem.overlap)
            .with_n_obs(problem.nrows as u64)
            .with_bootstrap(boot))
    }

    fn bootstrap_se(
        &self,
        problem: &PreparedFrontDoorProblem,
        model: ResolvedModel,
        ctx: &ExecutionContext,
    ) -> Result<BootstrapSeResult, EstimationError> {
        let n = problem.nrows;
        let k = problem.mediators.len();
        crate::util::bootstrap_se_with_scratch(
            self.bootstrap_replicates,
            ctx,
            0xF80E_u64,
            n,
            || {
                (
                    LeastSquaresWorkspace::default(),
                    vec![0.0; n],
                    (0..k).map(|_| vec![0.0; n]).collect::<Vec<_>>(),
                    vec![0.0; n],
                )
            },
            |(ols, t_boot, m_boot, y_boot), idx| {
                for (r, &src) in idx.iter().enumerate() {
                    t_boot[r] = problem.treatment[src];
                    y_boot[r] = problem.outcome[src];
                    for (j, mcol) in problem.mediators.iter().enumerate() {
                        m_boot[j][r] = mcol[src];
                    }
                }
                let mediators: Vec<&[f64]> = m_boot.iter().map(Vec::as_slice).collect();
                let sample = Sample { treatment: t_boot, mediators: &mediators, outcome: y_boot };
                // A resample that loses an arm or a mediator cell has no plug-in value; it is
                // counted as a failed replicate, never replaced by an extrapolated one.
                Ok(contrast(
                    &sample,
                    problem.active,
                    problem.control,
                    model,
                    &self.backend,
                    ols,
                    false,
                )
                .ok()
                .map(|fit| fit.ate))
            },
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ResolvedModel {
    Saturated,
    ArmLinear,
}

fn resolve_model(requested: FrontDoorOutcomeModel, mediators: &[Arc<[f64]>]) -> ResolvedModel {
    match requested {
        FrontDoorOutcomeModel::Saturated => ResolvedModel::Saturated,
        FrontDoorOutcomeModel::ArmLinear => ResolvedModel::ArmLinear,
        FrontDoorOutcomeModel::Auto => {
            let discrete = mediators.iter().all(|m| {
                level_codes(m)
                    .is_some_and(|(_, levels)| levels.len() <= MAX_SATURATED_MEDIATOR_LEVELS)
            });
            if discrete { ResolvedModel::Saturated } else { ResolvedModel::ArmLinear }
        }
    }
}

fn model_assumption(model: ResolvedModel) -> AssumptionRecord {
    let (assumption, status) = match model {
        ResolvedModel::Saturated => (
            Assumption::Custom {
                id: Arc::from(SATURATED_ASSUMPTION_ID),
                description: Arc::from(
                    "The front-door functional is evaluated by empirical arm frequencies and saturated (treatment, joint mediator value) cell means of the outcome: treatment and mediators are treated as discrete with their observed values as categories, and no functional form is imposed. Every mediator value observed under a contrasted treatment level was checked to occur under every observed treatment level; the fit is refused otherwise.",
                ),
            },
            AssumptionStatus::Supported,
        ),
        ResolvedModel::ArmLinear => (
            Assumption::ParametricRestriction(ParametricAssumption {
                id: Arc::from(ARM_LINEAR_ASSUMPTION_ID),
                description: Arc::from(
                    "The front-door functional is evaluated with E[Y | M, T=t'] modelled separately within each observed treatment level as linear and additive in the mediators (treatment-mediator interaction is unrestricted; mediator-mediator interaction and curvature in a mediator are excluded), extrapolated to the mediator means of the contrasted levels. The mediator law given treatment is empirical and unrestricted. Under a nonlinear within-arm outcome regression the estimate is not the identified functional.",
                ),
            }),
            AssumptionStatus::Declared,
        ),
    };
    AssumptionRecord {
        assumption,
        source: AssumptionSource::AlgorithmDefault { algorithm: Arc::from(ALGORITHM) },
        scope: AssumptionScope::Estimation,
        status,
    }
}

/// Borrowed complete-case columns.
struct Sample<'a> {
    treatment: &'a [f64],
    mediators: &'a [&'a [f64]],
    outcome: &'a [f64],
}

struct Contrast {
    ate: f64,
    /// Per-row `φ_active − φ_control` when requested.
    influence: Option<Vec<f64>>,
}

/// Code a column by its sorted distinct values. `None` when a value is not finite.
fn level_codes(values: &[f64]) -> Option<(Vec<u32>, Vec<f64>)> {
    if values.iter().any(|v| !v.is_finite()) {
        return None;
    }
    // `+ 0.0` folds `-0.0` into `0.0` so the two do not become separate levels.
    let mut levels: Vec<f64> = values.iter().map(|v| v + 0.0).collect();
    levels.sort_by(f64::total_cmp);
    levels.dedup();
    let codes = values
        .iter()
        .map(|v| levels.binary_search_by(|probe| probe.total_cmp(&(v + 0.0))).unwrap_or(0) as u32)
        .collect();
    Some((codes, levels))
}

fn level_index(levels: &[f64], value: f64) -> Option<usize> {
    levels.binary_search_by(|probe| probe.total_cmp(&(value + 0.0))).ok()
}

#[allow(clippy::too_many_arguments)]
fn contrast(
    sample: &Sample<'_>,
    active: f64,
    control: f64,
    model: ResolvedModel,
    backend: &FaerBackend,
    ols: &mut LeastSquaresWorkspace,
    want_influence: bool,
) -> Result<Contrast, EstimationError> {
    let n = sample.treatment.len();
    if n == 0 {
        return Err(EstimationError::data_msg("front-door functional has no complete rows"));
    }
    if sample.outcome.iter().any(|y| !y.is_finite()) {
        return Err(EstimationError::data_msg("front-door functional needs a finite outcome"));
    }
    let (arm_of, arms) = level_codes(sample.treatment).ok_or_else(|| {
        EstimationError::data_msg("front-door functional needs a finite treatment")
    })?;
    if arms.len() > MAX_TREATMENT_LEVELS {
        return Err(EstimationError::unsupported(
            "frontdoor.functional needs a discrete treatment (its observed values are the arms of the functional); use frontdoor.linear_two_stage for a continuous treatment under its recorded linearity restriction",
        ));
    }
    let (Some(arm_a), Some(arm_c)) = (level_index(&arms, active), level_index(&arms, control))
    else {
        return Err(EstimationError::unsupported(
            "frontdoor.functional needs both contrasted treatment levels to be observed: P(M | T=t) is estimated from the rows at that level",
        ));
    };
    let mut arm_n = vec![0.0_f64; arms.len()];
    for &a in &arm_of {
        arm_n[a as usize] += 1.0;
    }
    let outcome_model = match model {
        ResolvedModel::Saturated => OutcomeFit::saturated(sample, &arm_of, &arm_n, [arm_a, arm_c])?,
        ResolvedModel::ArmLinear => {
            OutcomeFit::arm_linear(sample, &arm_of, &arm_n, [arm_a, arm_c], backend, ols)?
        }
    };
    let nf = n as f64;
    let arm_p: Vec<f64> = arm_n.iter().map(|c| c / nf).collect();

    // μ(M_i, t') for every row and arm, then g(M_i) = Σ_t' p_t' μ(M_i, t').
    let n_arms = arms.len();
    let mut psi = [0.0_f64; 2];
    let mut h = [vec![0.0_f64; n_arms], vec![0.0_f64; n_arms]];
    let mut g = vec![0.0_f64; n];
    for i in 0..n {
        let own = arm_of[i] as usize;
        let slot = match own {
            a if a == arm_a => Some(0),
            a if a == arm_c => Some(1),
            _ => None,
        };
        // Only rows of a contrasted arm enter E[· | T=a]; their μ is defined in every arm.
        let Some(slot) = slot else { continue };
        for (arm, &p) in arm_p.iter().enumerate() {
            let mu = outcome_model.mean(sample, i, arm);
            g[i] += p * mu;
            h[slot][arm] += mu / arm_n[own];
        }
        psi[slot] += g[i] / arm_n[own];
    }
    let ate = psi[0] - psi[1];
    if !ate.is_finite() {
        return Err(EstimationError::stats_msg("front-door functional is not finite"));
    }
    if !want_influence {
        return Ok(Contrast { ate, influence: None });
    }

    let mut influence = vec![0.0_f64; n];
    for i in 0..n {
        let own = arm_of[i] as usize;
        let mut phi = [0.0_f64; 2];
        for (slot, &arm) in [arm_a, arm_c].iter().enumerate() {
            phi[slot] = h[slot][own] - psi[slot];
            if own == arm {
                phi[slot] += (g[i] - psi[slot]) / arm_p[arm];
            }
            phi[slot] +=
                outcome_model.residual_weight(sample, i, own, slot) * outcome_model.residual(i);
        }
        influence[i] = phi[0] - phi[1];
    }
    Ok(Contrast { ate, influence: Some(influence) })
}

/// Fitted `μ(m, t')` with what its influence term needs.
enum OutcomeFit {
    Saturated {
        /// Row → joint mediator cell.
        cell_of: Vec<usize>,
        /// `[cell][arm]` outcome means (`NaN` where the cell has no rows in the arm).
        mean: Vec<Vec<f64>>,
        /// `[cell][arm]` row counts.
        count: Vec<Vec<f64>>,
        /// Arm sizes.
        arm_n: Vec<f64>,
        /// Contrasted arms `[active, control]`.
        contrasted: [usize; 2],
        residual: Vec<f64>,
    },
    ArmLinear {
        /// Per-arm `[β₀, β₁…βₖ]`.
        coefficients: Vec<Vec<f64>>,
        /// Per-arm `n_arm · (XᵀX)⁻¹` (row-major, `(1+k)²`).
        scaled_gram_inverse: Vec<Vec<f64>>,
        /// `[active, control]` arm means of `x = [1, M]`.
        contrasted_mean: [Vec<f64>; 2],
        residual: Vec<f64>,
    },
}

impl OutcomeFit {
    fn saturated(
        sample: &Sample<'_>,
        arm_of: &[u32],
        arm_n: &[f64],
        contrasted: [usize; 2],
    ) -> Result<Self, EstimationError> {
        let n = sample.treatment.len();
        let mut coded = Vec::with_capacity(sample.mediators.len());
        for m in sample.mediators {
            let (codes, _) = level_codes(m).ok_or_else(|| {
                EstimationError::data_msg("front-door functional needs finite mediators")
            })?;
            coded.push(codes);
        }
        let mut cells: BTreeMap<Vec<u32>, usize> = BTreeMap::new();
        let mut cell_of = Vec::with_capacity(n);
        for i in 0..n {
            let key: Vec<u32> = coded.iter().map(|codes| codes[i]).collect();
            let next = cells.len();
            cell_of.push(*cells.entry(key).or_insert(next));
        }
        let n_arms = arm_n.len();
        let mut count = vec![vec![0.0_f64; n_arms]; cells.len()];
        let mut mean = vec![vec![0.0_f64; n_arms]; cells.len()];
        for i in 0..n {
            count[cell_of[i]][arm_of[i] as usize] += 1.0;
            mean[cell_of[i]][arm_of[i] as usize] += sample.outcome[i];
        }
        for (cell_mean, cell_count) in mean.iter_mut().zip(&count) {
            let needed = contrasted.iter().any(|&arm| cell_count[arm] > 0.0);
            for (m, &c) in cell_mean.iter_mut().zip(cell_count) {
                if c > 0.0 {
                    *m /= c;
                } else if needed {
                    return Err(EstimationError::Overlap {
                        message: "frontdoor.functional (saturated): a mediator value observed under a contrasted treatment level never occurs under another treatment level, so E[Y | M, T] has no data there; coarsen the mediator or request the arm-linear outcome model",
                    });
                } else {
                    *m = f64::NAN;
                }
            }
        }
        let residual =
            (0..n).map(|i| sample.outcome[i] - mean[cell_of[i]][arm_of[i] as usize]).collect();
        Ok(Self::Saturated { cell_of, mean, count, arm_n: arm_n.to_vec(), contrasted, residual })
    }

    fn arm_linear(
        sample: &Sample<'_>,
        arm_of: &[u32],
        arm_n: &[f64],
        contrasted: [usize; 2],
        backend: &FaerBackend,
        ols: &mut LeastSquaresWorkspace,
    ) -> Result<Self, EstimationError> {
        let n = sample.treatment.len();
        let k = sample.mediators.len();
        let ncols = 1 + k;
        if sample.mediators.iter().any(|m| m.iter().any(|v| !v.is_finite())) {
            return Err(EstimationError::data_msg("front-door functional needs finite mediators"));
        }
        let mut coefficients = Vec::with_capacity(arm_n.len());
        let mut scaled_gram_inverse = Vec::with_capacity(arm_n.len());
        let mut residual = vec![0.0_f64; n];
        for arm in 0..arm_n.len() {
            let rows: Vec<usize> = (0..n).filter(|&i| arm_of[i] as usize == arm).collect();
            let n_arm = rows.len();
            let mut x = vec![0.0_f64; n_arm * ncols];
            x[..n_arm].fill(1.0);
            for (j, m) in sample.mediators.iter().enumerate() {
                for (r, &i) in rows.iter().enumerate() {
                    x[(1 + j) * n_arm + r] = m[i];
                }
            }
            let y: Vec<f64> = rows.iter().map(|&i| sample.outcome[i]).collect();
            let rank_deficient = EstimationError::stats_msg(
                "frontdoor.functional (arm-linear): the outcome regression on the mediators is rank deficient within a treatment level",
            );
            if n_arm < ncols {
                return Err(rank_deficient);
            }
            let fit = backend.least_squares(&x, n_arm, ncols, &y, ols).map_err(stats_err)?;
            if fit.rank < ncols {
                return Err(rank_deficient);
            }
            let mut gram = vec![0.0_f64; ncols * ncols];
            form_xtx(&x, n_arm, ncols, &mut gram);
            let mut inverse = invert_square(&gram, ncols).ok_or(rank_deficient)?;
            for v in &mut inverse {
                *v *= n_arm as f64;
            }
            for (r, &i) in rows.iter().enumerate() {
                residual[i] = fit.residuals[r];
            }
            coefficients.push(fit.coefficients);
            scaled_gram_inverse.push(inverse);
        }
        let contrasted_mean = contrasted.map(|arm| {
            let mut xbar = vec![0.0_f64; ncols];
            xbar[0] = 1.0;
            for i in (0..n).filter(|&i| arm_of[i] as usize == arm) {
                for (j, m) in sample.mediators.iter().enumerate() {
                    xbar[1 + j] += m[i] / arm_n[arm];
                }
            }
            xbar
        });
        Ok(Self::ArmLinear { coefficients, scaled_gram_inverse, contrasted_mean, residual })
    }

    /// `μ(M_i, arm)`.
    fn mean(&self, sample: &Sample<'_>, row: usize, arm: usize) -> f64 {
        match self {
            Self::Saturated { cell_of, mean, .. } => mean[cell_of[row]][arm],
            Self::ArmLinear { coefficients, .. } => {
                let beta = &coefficients[arm];
                sample
                    .mediators
                    .iter()
                    .enumerate()
                    .fold(beta[0], |acc, (j, m)| acc + beta[1 + j] * m[row])
            }
        }
    }

    fn residual(&self, row: usize) -> f64 {
        match self {
            Self::Saturated { residual, .. } | Self::ArmLinear { residual, .. } => residual[row],
        }
    }

    /// Weight on the row's own residual in `φ` of contrasted arm `slot`.
    fn residual_weight(&self, sample: &Sample<'_>, row: usize, own: usize, slot: usize) -> f64 {
        match self {
            Self::Saturated { cell_of, count, arm_n, contrasted, .. } => {
                let cell = &count[cell_of[row]];
                let arm = contrasted[slot];
                (cell[arm] / arm_n[arm]) / (cell[own] / arm_n[own])
            }
            Self::ArmLinear { scaled_gram_inverse, contrasted_mean, .. } => {
                let ncols = 1 + sample.mediators.len();
                let inverse = &scaled_gram_inverse[own];
                let xbar = &contrasted_mean[slot];
                let mut weight = 0.0;
                for r in 0..ncols {
                    let mut acc = inverse[r * ncols];
                    for (j, m) in sample.mediators.iter().enumerate() {
                        acc += inverse[r * ncols + 1 + j] * m[row];
                    }
                    weight += xbar[r] * acc;
                }
                weight
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::many_single_char_names, clippy::float_cmp)]
mod tests {
    use antecedent_core::{
        CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType, VariableId,
    };
    use antecedent_data::{Float64Column, OwnedColumn, OwnedColumnarStorage, ValidityBitmap};
    use antecedent_expr::ExprId;
    use antecedent_kernels::standard_normal;

    use super::*;
    use crate::frontdoor::tests::{
        build_frontdoor_data, ctx, frontdoor_estimand, interaction_scm_exact_table, query,
    };
    use crate::frontdoor::{FrontDoorTwoStage, FrontDoorWorkspace};

    fn no_bootstrap(model: FrontDoorOutcomeModel) -> FrontDoorFunctional {
        FrontDoorFunctional::new().with_bootstrap_replicates(0).with_outcome_model(model)
    }

    fn fit(
        est: &FrontDoorFunctional,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
    ) -> Result<EffectEstimate, EstimationError> {
        let prep = est.prepare(data, estimand, &query())?;
        est.fit(&prep, &ctx(), AssumptionSet::new())
    }

    fn records(effect: &EffectEstimate, id: &str) -> bool {
        effect.assumptions.entries.iter().any(|r| match &r.assumption {
            Assumption::ParametricRestriction(p) => p.id.as_ref() == id,
            Assumption::Custom { id: custom, .. } => custom.as_ref() == id,
            _ => false,
        })
    }

    #[test]
    fn saturated_plug_in_matches_enumerated_truth_on_exact_table() {
        // Arm frequencies are 0.35 / 0.65 and the within-arm mediator slopes differ
        // (0.771 vs 0.277), so weighting Σ_t' uniformly gives 0.367 and the linear shortcut
        // gives 0.363; only P(t') weights reach the enumerated 0.315.
        let (data, truth) = interaction_scm_exact_table();
        for model in [FrontDoorOutcomeModel::Auto, FrontDoorOutcomeModel::Saturated] {
            let effect = fit(&no_bootstrap(model), &data, &frontdoor_estimand()).unwrap();
            assert!((effect.ate - truth).abs() < 1e-12, "ate={} truth={truth}", effect.ate);
            assert!(records(&effect, SATURATED_ASSUMPTION_ID));
            assert!(!records(&effect, ARM_LINEAR_ASSUMPTION_ID));
        }
        // With one binary mediator the arm-linear model is saturated too.
        let effect =
            fit(&no_bootstrap(FrontDoorOutcomeModel::ArmLinear), &data, &frontdoor_estimand())
                .unwrap();
        assert!((effect.ate - truth).abs() < 1e-10, "arm-linear ate={}", effect.ate);
        assert!(records(&effect, ARM_LINEAR_ASSUMPTION_ID));
    }

    /// Columns in id order `t, y, m…`.
    fn build_columns(columns: Vec<Vec<f64>>) -> TabularData {
        let n = columns[0].len();
        let mut b = CausalSchemaBuilder::new();
        for idx in 0..columns.len() {
            let role = match idx {
                0 => RoleHint::TreatmentCandidate,
                1 => RoleHint::OutcomeCandidate,
                _ => RoleHint::Context,
            };
            b.add_variable(
                format!("v{idx}"),
                ValueType::Continuous,
                SmallRoleSet::from_hint(role),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        }
        let cols = columns
            .into_iter()
            .enumerate()
            .map(|(idx, col)| {
                OwnedColumn::Float64(
                    Float64Column::new(
                        VariableId::from_raw(idx as u32),
                        Arc::from(col),
                        ValidityBitmap::all_valid(n),
                    )
                    .unwrap(),
                )
            })
            .collect();
        TabularData::new(
            OwnedColumnarStorage::try_new(b.build().unwrap(), cols, None, None).unwrap(),
        )
    }

    /// Exact table (N = 2000) of a three-arm, two-chained-mediator SCM with a latent `U`
    /// that shifts treatment and interacts with both mediators in a deterministic outcome:
    /// `P(U=1) = .5`, `P(T|U)`, `P(M1=1|T)`, `P(M2=1|T,M1)` in tenths, `Y = f(M1, M2, U)`.
    fn three_arm_two_mediator_table() -> (TabularData, IdentifiedEstimand, [f64; 3]) {
        let p_t = [[0.5, 0.3, 0.2], [0.2, 0.2, 0.6]];
        let p_m1 = [0.2, 0.5, 0.9];
        let p_m2 = |t: usize, m1: usize| 0.1 + 0.2 * t as f64 + 0.3 * m1 as f64;
        let f = |m1: usize, m2: usize, u: usize| {
            1.0 + 2.0 * m1 as f64 - 1.5 * m2 as f64
                + u as f64 * (0.5 + 3.0 * (m1 * m2) as f64 + m2 as f64)
        };
        let bern = |p: f64, v: usize| if v == 1 { p } else { 1.0 - p };
        let mut truth = [0.0_f64; 3];
        let (mut t, mut m1c, mut m2c, mut y) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for u in 0..2 {
            for m1 in 0..2 {
                for m2 in 0..2 {
                    for (arm, slot) in truth.iter_mut().enumerate() {
                        *slot += 0.5 * bern(p_m1[arm], m1) * bern(p_m2(arm, m1), m2) * f(m1, m2, u);
                    }
                    for ti in 0..3 {
                        let mass =
                            0.5 * p_t[u][ti] * bern(p_m1[ti], m1) * bern(p_m2(ti, m1), m2) * 2000.0;
                        let count = mass.round() as usize;
                        assert!((mass - count as f64).abs() < 1e-9, "mass={mass}");
                        for _ in 0..count {
                            t.push(ti as f64);
                            m1c.push(m1 as f64);
                            m2c.push(m2 as f64);
                            y.push(f(m1, m2, u));
                        }
                    }
                }
            }
        }
        let data = build_columns(vec![t, y, m1c, m2c]);
        let estimand = IdentifiedEstimand::frontdoor(
            "frontdoor",
            Arc::from([VariableId::from_raw(2), VariableId::from_raw(3)]),
            ExprId::from_raw(0),
        );
        (data, estimand, truth)
    }

    #[test]
    fn saturated_plug_in_handles_three_arms_and_a_joint_mediator_set() {
        let (data, estimand, truth) = three_arm_two_mediator_table();
        let est = no_bootstrap(FrontDoorOutcomeModel::Saturated);
        for (active, control) in [(1.0, 0.0), (2.0, 0.0), (2.0, 1.0)] {
            let q = AverageEffectQuery::with_levels(
                VariableId::from_raw(0),
                VariableId::from_raw(1),
                control,
                active,
            );
            let prep = est.prepare(&data, &estimand, &q).unwrap();
            let effect = est.fit(&prep, &ctx(), AssumptionSet::new()).unwrap();
            let expected = truth[active as usize] - truth[control as usize];
            assert!(expected.abs() > 0.1, "fixture contrast must be non-trivial");
            assert!(
                (effect.ate - expected).abs() < 1e-12,
                "ate={} expected={expected}",
                effect.ate
            );
        }
    }

    /// One draw of the binary interaction SCM behind `interaction_scm_exact_table`.
    fn binary_scm_draw(n: usize, seed: u64) -> TabularData {
        let mut rng = ExecutionContext::for_tests(seed).rng.stream(0xF410_u64);
        let (mut t, mut m, mut y) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        for i in 0..n {
            let u = f64::from(rng.next_f64() < 0.5);
            t[i] = f64::from(rng.next_f64() < 0.1 + 0.5 * u);
            m[i] = f64::from(rng.next_f64() < 0.1 + 0.7 * t[i]);
            y[i] = f64::from(rng.next_f64() < 0.05 + 0.9 * m[i] * u);
        }
        build_frontdoor_data(n, t, y, m)
    }

    /// Binary `T`, continuous heteroskedastic `M = 1 + 0.4T + (1 + T)ε` (uncentred, slope
    /// 0.4), `Y = 5·M·(0.5 + U) + ε`. `E[Y|M,T]` is linear in `M` within an arm with slope
    /// `5(0.5 + E[U|T])`, and `E[Y|do(t)] = 5·E[M|t]·(0.5 + E[U])`, so the effect is
    /// `5 · 0.4 · 1 = 2`. The product of coefficients weights the arm slopes by
    /// `P(t)·Var(M|t)` instead of `P(t)` and converges to about 2.37.
    fn arm_linear_scm_draw(n: usize, seed: u64) -> TabularData {
        let mut rng = ExecutionContext::for_tests(seed).rng.stream(0xF411_u64);
        let (mut t, mut m, mut y) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        for i in 0..n {
            let u = f64::from(rng.next_f64() < 0.5);
            t[i] = f64::from(rng.next_f64() < 0.1 + 0.5 * u);
            m[i] = 1.0 + 0.4 * t[i] + (1.0 + t[i]) * standard_normal(&mut rng);
            y[i] = 5.0 * m[i] * (0.5 + u) + standard_normal(&mut rng);
        }
        build_frontdoor_data(n, t, y, m)
    }

    #[test]
    fn arm_linear_recovers_truth_where_the_path_product_does_not() {
        let data = arm_linear_scm_draw(200_000, 5);
        let effect =
            fit(&no_bootstrap(FrontDoorOutcomeModel::Auto), &data, &frontdoor_estimand()).unwrap();
        assert!(records(&effect, ARM_LINEAR_ASSUMPTION_ID), "continuous M must select arm-linear");
        assert!(effect.se_analytic < 0.05, "se={}", effect.se_analytic);
        assert!((effect.ate - 2.0).abs() < 4.0 * effect.se_analytic, "ate={}", effect.ate);

        let linear = FrontDoorTwoStage { bootstrap_replicates: 0, ..FrontDoorTwoStage::new() };
        let prep = linear.prepare(&data, &frontdoor_estimand(), &query()).unwrap();
        let shortcut = linear
            .fit(&prep, &mut FrontDoorWorkspace::default(), &ctx(), AssumptionSet::new())
            .unwrap();
        assert!((shortcut.ate - 2.37).abs() < 0.08, "path product ate={}", shortcut.ate);
    }

    /// Empirical SD of the estimator vs mean analytic SE, and 95% Wald coverage, over seeded
    /// replicates. With `R` replicates the coverage band is `0.95 ± 3·sqrt(.95·.05/R)`; the SD
    /// of a sample SD over `R` near-normal draws is about `1/sqrt(2R)` relative, so a 12% band
    /// on the SE ratio is over three of those at `R = 400`.
    fn assert_se_calibrated(
        model: FrontDoorOutcomeModel,
        truth: f64,
        draw: impl Fn(u64) -> TabularData,
    ) {
        let reps = 400_u64;
        let est = no_bootstrap(model);
        let (mut estimates, mut mean_se, mut covered) = (Vec::new(), 0.0, 0.0);
        for seed in 0..reps {
            let effect = fit(&est, &draw(1000 + seed), &frontdoor_estimand()).unwrap();
            estimates.push(effect.ate);
            mean_se += effect.se_analytic / reps as f64;
            if (effect.ate - truth).abs() <= 1.959_963_984_540_054 * effect.se_analytic {
                covered += 1.0 / reps as f64;
            }
        }
        let sd = antecedent_stats::sample_std(&estimates);
        let ratio = mean_se / sd;
        assert!((ratio - 1.0).abs() < 0.12, "mean se={mean_se} empirical sd={sd}");
        let band = 3.0 * (0.95 * 0.05 / reps as f64).sqrt();
        assert!((covered - 0.95).abs() <= band, "coverage={covered} band={band}");
    }

    #[test]
    fn saturated_influence_se_matches_sampling_sd_and_covers() {
        assert_se_calibrated(FrontDoorOutcomeModel::Saturated, 0.315, |seed| {
            binary_scm_draw(1500, seed)
        });
    }

    #[test]
    fn arm_linear_influence_se_matches_sampling_sd_and_covers() {
        assert_se_calibrated(FrontDoorOutcomeModel::ArmLinear, 2.0, |seed| {
            arm_linear_scm_draw(1500, seed)
        });
    }

    #[test]
    fn refuses_a_continuous_treatment_and_an_unobserved_level() {
        let n = 400;
        let mut rng = ExecutionContext::for_tests(3).rng.stream(0xF412_u64);
        let t: Vec<f64> = (0..n).map(|_| standard_normal(&mut rng)).collect();
        let m: Vec<f64> = t.iter().map(|ti| 0.4 * ti).collect();
        let y: Vec<f64> = m.iter().map(|mi| 5.0 * mi).collect();
        let data = build_frontdoor_data(n, t, y, m);
        let err = fit(&no_bootstrap(FrontDoorOutcomeModel::Auto), &data, &frontdoor_estimand())
            .unwrap_err();
        assert!(err.to_string().contains("discrete treatment"), "err={err}");

        // Levels {0, 1} are observed; contrast 3 vs 0.
        let data = binary_scm_draw(400, 9);
        let mut prep =
            FrontDoorFunctional::new().prepare(&data, &frontdoor_estimand(), &query()).unwrap();
        prep.active = 3.0;
        let err = no_bootstrap(FrontDoorOutcomeModel::Auto)
            .fit(&prep, &ctx(), AssumptionSet::new())
            .unwrap_err();
        assert!(err.to_string().contains("observed"), "err={err}");
    }

    #[test]
    fn saturated_refuses_a_mediator_value_missing_from_an_arm() {
        // M = 2 occurs only under T = 1, so E[Y | M=2, T=0] has no data.
        let t = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let m = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 2.0];
        let y = vec![0.0, 1.0, 0.5, 1.5, 0.2, 1.2, 2.0, 2.4];
        let data = build_frontdoor_data(8, t, y, m);
        let err =
            fit(&no_bootstrap(FrontDoorOutcomeModel::Saturated), &data, &frontdoor_estimand())
                .unwrap_err();
        assert!(matches!(err, EstimationError::Overlap { .. }), "err={err}");
    }

    #[test]
    fn bootstrap_se_agrees_with_influence_se() {
        let data = binary_scm_draw(3000, 77);
        let est = FrontDoorFunctional::new().with_bootstrap_replicates(400);
        let effect = fit(&est, &data, &frontdoor_estimand()).unwrap();
        let boot = effect.se_bootstrap.expect("bootstrap SE");
        let rel = (effect.se_analytic - boot).abs() / boot;
        assert!(rel < 0.15, "analytic={} boot={boot}", effect.se_analytic);
    }
}
