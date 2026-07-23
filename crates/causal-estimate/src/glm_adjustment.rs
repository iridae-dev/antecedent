//! Generalized linear model (logistic) adjustment ATE estimator for binary outcomes .
//!
//! Fits a logistic GLM `Y ~ T + Z` and recovers the ATE by finite-difference g-computation:
//! the fitted model is evaluated at `T = active` and `T = control` for every row (holding `Z`
//! fixed), and the ATE is the mean of the per-row predicted-probability contrast. This is the
//! standard g-computation contrast for a non-identity link, since the coefficient on `T` alone
//! is a log-odds-ratio, not a probability-scale effect.
//!
//! Positivity is handled the same way as [`crate::adjustment::LinearAdjustmentAte`]:
//! [`OverlapPolicy::ExplicitOverride`] is the only supported policy, since this is a regression
//! (not propensity-based) path.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::manual_map,
    clippy::similar_names,
    clippy::too_many_arguments
)]

use std::sync::Arc;

use causal_core::{
    AssumptionSet, AverageEffectQuery, ExecutionContext, TargetPopulation, VariableId,
};
use causal_data::TabularData;
use causal_expr::IdentifiedEstimand;
use causal_stats::{
    CompiledDesign, FaerBackend, GlmDesignRef, GlmFamily, GlmOptions, LeastSquaresWorkspace,
    fit_glm, form_xtx, invert_square, score_coefficient_covariance,
};

use crate::adjustment::{EffectEstimate, intervention_f64};
use crate::error::EstimationError;
use crate::gcomp::gcomp_diffs;
use crate::overlap::OverlapPolicy;
use crate::se::AnalyticSeKind;
use crate::util::{BootstrapSeResult, bootstrap_se, stats_err};

/// Prepared GLM adjustment problem (compiled design retained).
#[derive(Clone, Debug)]
pub struct PreparedGlmProblem {
    /// Compiled `[1 | T | Z…]` design; outcome must be binary (0/1).
    pub design: CompiledDesign,
    /// Estimand method tag.
    pub method: Arc<str>,
    /// Adjustment set.
    pub adjustment_set: Arc<[VariableId]>,
    /// Overlap policy applied.
    pub overlap: OverlapPolicy,
    /// Active treatment level used for the g-computation contrast.
    pub active: f64,
    /// Control treatment level used for the g-computation contrast.
    pub control: f64,
    /// GLM family used for this problem.
    pub family: GlmFamily,
    /// Target population for g-computation averaging.
    pub target_population: TargetPopulation,
    /// Complete-case treatment values.
    pub treatment: Arc<[f64]>,
}

/// Estimation workspace (reusable across bootstrap replicates).
#[derive(Clone, Debug, Default)]
pub struct GlmAdjustmentWorkspace {
    /// IRLS least-squares scratch.
    pub ols: LeastSquaresWorkspace,
}

/// Logistic GLM adjustment estimator for binary-outcome backdoor ATE.
///
/// ATE is estimated by finite-difference g-computation: `mean(μ(T=active, Z) − μ(T=control, Z))`
/// over the complete-case sample, where `μ` is the fitted logistic mean function.
#[derive(Clone, Debug)]
pub struct GlmAdjustmentAte {
    /// Dense linear-algebra backend used by the IRLS inner loop.
    pub backend: FaerBackend,
    /// Bootstrap replicates (0 = skip bootstrap).
    pub bootstrap_replicates: u32,
    /// Overlap policy (must be [`OverlapPolicy::ExplicitOverride`]).
    pub overlap: OverlapPolicy,
    /// GLM fitting options (max iterations, convergence tolerance).
    pub glm_options: GlmOptions,
    /// Outcome family / link.
    pub family: GlmFamily,
    /// Analytic SE kind (default Homoskedastic → Fisher delta-method).
    pub se_kind: AnalyticSeKind,
    /// Optional cluster ids for cluster / panel sandwich SE.
    pub cluster_ids: Option<Vec<u32>>,
    /// Optional multiway cluster ids.
    pub multiway_ids: Option<Vec<Vec<u32>>>,
}

impl Default for GlmAdjustmentAte {
    fn default() -> Self {
        Self::new()
    }
}

impl GlmAdjustmentAte {
    /// Default: 200 bootstrap replicates, explicit overlap override, ridge-on-separation.
    #[must_use]
    pub fn new() -> Self {
        Self {
            backend: FaerBackend,
            bootstrap_replicates: 200,
            overlap: OverlapPolicy::ExplicitOverride,
            glm_options: GlmOptions::default(),
            family: GlmFamily::BinomialLogit,
            se_kind: AnalyticSeKind::Homoskedastic,
            cluster_ids: None,
            multiway_ids: None,
        }
    }

    /// Prepare design from tabular data, identified estimand, and query levels.
    ///
    /// Accepts `backdoor.adjustment` / `backdoor.efficient` estimands.
    ///
    /// # Errors
    ///
    /// Overlap policy is not `ExplicitOverride`, incompatible estimand, unsupported query, or
    /// missing/invalid data columns.
    pub fn prepare(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
    ) -> Result<PreparedGlmProblem, EstimationError> {
        crate::util::require_explicit_override(
            self.overlap,
            "GlmAdjustmentAte requires ExplicitOverride overlap policy",
        )?;
        if !matches!(
            estimand.method_kind().ok(),
            Some(
                causal_expr::EstimandMethod::BackdoorAdjustment
                    | causal_expr::EstimandMethod::BackdoorEfficient
            )
        ) {
            return Err(EstimationError::IncompatibleEstimand {
                message: "GlmAdjustmentAte expects backdoor.adjustment or backdoor.efficient",
            });
        }
        query.validate()?;
        if !query.effect_modifiers.is_empty() {
            return Err(EstimationError::unsupported(
                "GLM adjustment does not support effect modifiers",
            ));
        }
        if !matches!(
            query.target_population,
            TargetPopulation::AllObserved
                | TargetPopulation::Treated
                | TargetPopulation::Untreated
                | TargetPopulation::Predicate(_)
        ) {
            return Err(EstimationError::unsupported(
                "GLM adjustment supports AllObserved, Treated, Untreated, or Predicate",
            ));
        }
        let treatment = query.treatment;
        let outcome = query.outcome;
        let active = intervention_f64(&query.active)?;
        let control = intervention_f64(&query.control)?;
        if active == control {
            return Err(EstimationError::unsupported(
                "active and control treatment levels must differ",
            ));
        }

        let mut ids = Vec::with_capacity(2 + estimand.adjustment_set.len());
        ids.push(treatment);
        ids.push(outcome);
        ids.extend_from_slice(&estimand.adjustment_set);
        let row_mask = data.complete_case_mask(&ids).map_err(EstimationError::from)?;
        let t = data.float64_masked(treatment, &row_mask).map_err(EstimationError::from)?;
        let y = data.float64_masked(outcome, &row_mask).map_err(EstimationError::from)?;
        match self.family {
            GlmFamily::BinomialLogit | GlmFamily::BinomialProbit => {
                for &yi in &y {
                    if !(yi == 0.0 || yi == 1.0) {
                        return Err(EstimationError::unsupported(
                            "Binomial GlmAdjustmentAte requires a binary (0/1) outcome",
                        ));
                    }
                }
            }
            GlmFamily::PoissonLog | GlmFamily::NegativeBinomial => {
                for &yi in &y {
                    if !(yi.is_finite() && yi >= 0.0) {
                        return Err(EstimationError::unsupported(
                            "Poisson/NB GlmAdjustmentAte requires non-negative outcomes",
                        ));
                    }
                }
            }
            GlmFamily::GaussianIdentity => {}
        }
        let mut covs: Vec<(VariableId, Vec<f64>)> = Vec::new();
        for &z in estimand.adjustment_set.iter() {
            covs.push((z, data.float64_masked(z, &row_mask).map_err(EstimationError::from)?));
        }
        let cov_refs: Vec<(VariableId, &[f64])> =
            covs.iter().map(|(id, v)| (*id, v.as_slice())).collect();
        let selected_rows: Vec<usize> =
            row_mask.iter().enumerate().filter_map(|(i, keep)| keep.then_some(i)).collect();
        let design = CompiledDesign::linear_adjustment(&t, &cov_refs, &y, &selected_rows)
            .map_err(EstimationError::from)?;
        Ok(PreparedGlmProblem {
            design,
            method: Arc::clone(&estimand.method),
            adjustment_set: Arc::clone(&estimand.adjustment_set),
            overlap: self.overlap,
            active,
            control,
            family: self.family,
            target_population: query.target_population.clone(),
            treatment: Arc::from(t),
        })
    }

    /// Fit the logistic GLM and compute the g-computation ATE, with optional bootstrap.
    ///
    /// # Errors
    ///
    /// GLM/backend failure.
    pub fn fit(
        &self,
        problem: &PreparedGlmProblem,
        workspace: &mut GlmAdjustmentWorkspace,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        let t_col = problem
            .design
            .treatment_column()
            .ok_or_else(|| EstimationError::stats_msg("missing treatment column"))?;
        let glm_fit = fit_glm(
            problem.family,
            GlmDesignRef {
                x_colmajor: &problem.design.matrix,
                nrows: problem.design.nrows,
                ncols: problem.design.ncols,
                y: &problem.design.outcome,
            },
            &self.backend,
            &mut workspace.ols,
            &self.glm_options,
        )
        .map_err(stats_err)?;
        glm_fit.require_ok().map_err(stats_err)?;

        let diffs = gcomp_diffs(
            problem.family,
            &problem.design.matrix,
            problem.design.nrows,
            problem.design.ncols,
            t_col,
            &glm_fit.coefficients,
            problem.active,
            problem.control,
        );
        let ate = average_gcomp_for_target(&diffs, &problem.treatment, &problem.target_population)?;
        let se_analytic = match self.se_kind {
            AnalyticSeKind::Homoskedastic => gcomp_delta_method_se(
                problem.family,
                &problem.design.matrix,
                problem.design.nrows,
                problem.design.ncols,
                t_col,
                &glm_fit.coefficients,
                problem.active,
                problem.control,
                glm_fit.deviance,
            ),
            other => gcomp_sandwich_se(
                other,
                problem.family,
                &problem.design.matrix,
                problem.design.nrows,
                problem.design.ncols,
                t_col,
                &glm_fit.coefficients,
                &problem.design.outcome,
                problem.active,
                problem.control,
                glm_fit.nb_alpha.unwrap_or(0.0),
                self.cluster_ids.as_deref(),
                self.multiway_ids.as_deref(),
            )?,
        };

        let boot = if self.bootstrap_replicates == 0 {
            None
        } else {
            Some(self.bootstrap_se(problem, workspace, ctx, t_col)?)
        };

        Ok(EffectEstimate {
            ate,
            se_analytic,
            se_bootstrap: None,
            bootstrap_replicates_ok: None,
            bootstrap_replicates_failed: None,
            bootstrap_cancelled: false,
            bootstrap_early_stopped: false,
            assumptions,
            overlap: problem.overlap,
            overlap_report: None,
            retained_memory_bytes: None,
        }
        .with_bootstrap(boot))
    }

    fn bootstrap_se(
        &self,
        problem: &PreparedGlmProblem,
        workspace: &mut GlmAdjustmentWorkspace,
        ctx: &ExecutionContext,
        t_col: usize,
    ) -> Result<BootstrapSeResult, EstimationError> {
        let n = problem.design.nrows;
        let p = problem.design.ncols;
        let mut x_boot = vec![0.0; n * p];
        let mut y_boot = vec![0.0; n];
        bootstrap_se(self.bootstrap_replicates, ctx, 0xC17A_u64, n, |idx| {
            for (r, &src) in idx.iter().enumerate() {
                y_boot[r] = problem.design.outcome[src];
                for c in 0..p {
                    x_boot[c * n + r] = problem.design.matrix[c * n + src];
                }
            }
            let Ok(fit) = fit_glm(
                problem.family,
                GlmDesignRef { x_colmajor: &x_boot, nrows: n, ncols: p, y: &y_boot },
                &self.backend,
                &mut workspace.ols,
                &self.glm_options,
            ) else {
                return Ok(None);
            };
            if fit.require_ok().is_err() {
                return Ok(None);
            };
            let diffs = gcomp_diffs(
                problem.family,
                &x_boot,
                n,
                p,
                t_col,
                &fit.coefficients,
                problem.active,
                problem.control,
            );
            let t_boot: Vec<f64> = idx.iter().map(|&src| problem.treatment[src]).collect();
            match average_gcomp_for_target(&diffs, &t_boot, &problem.target_population) {
                Ok(ate) => Ok(Some(ate)),
                Err(_) => Ok(None),
            }
        })
    }
}

fn average_gcomp_for_target(
    diffs: &[f64],
    treatment: &[f64],
    target: &TargetPopulation,
) -> Result<f64, EstimationError> {
    let mut sum = 0.0;
    let mut count = 0usize;
    for (i, &d) in diffs.iter().enumerate() {
        let include = match target {
            TargetPopulation::Treated => treatment.get(i).copied().unwrap_or(0.0) > 0.5,
            TargetPopulation::Untreated => treatment.get(i).copied().unwrap_or(1.0) <= 0.5,
            _ => true,
        };
        if include {
            sum += d;
            count += 1;
        }
    }
    if count == 0 {
        return Err(EstimationError::data_msg(
            "target population left no rows for GLM g-computation",
        ));
    }
    Ok(sum / count as f64)
}

/// Response-scale derivative `dμ/dη` at `eta`.
fn mean_derivative(family: GlmFamily, eta: f64) -> f64 {
    match family {
        GlmFamily::BinomialLogit => {
            let mu = 1.0 / (1.0 + (-eta).exp());
            mu * (1.0 - mu)
        }
        GlmFamily::BinomialProbit => {
            // φ(η) = dμ/dη for probit.
            causal_kernels::norm_pdf(eta)
        }
        GlmFamily::GaussianIdentity => 1.0,
        GlmFamily::PoissonLog | GlmFamily::NegativeBinomial => eta.exp(),
    }
}

/// IRLS / Fisher information weight `W_ii` at `eta`.
///
/// For canonical Bernoulli/logit and Poisson this equals `dμ/dη`. For probit the
/// Bernoulli Fisher weight is `φ(η)² / (μ(1−μ))`, not `φ(η)`.
fn fisher_weight(family: GlmFamily, eta: f64) -> f64 {
    match family {
        GlmFamily::BinomialProbit => {
            let phi = mean_derivative(GlmFamily::BinomialProbit, eta);
            // Φ(η) via erf; clamp away from {0,1} for numerical stability.
            let mu = (0.5 * (1.0 + causal_kernels::erf(eta / std::f64::consts::SQRT_2)))
                .clamp(1e-12, 1.0 - 1e-12);
            (phi * phi) / (mu * (1.0 - mu))
        }
        other => mean_derivative(other, eta),
    }
}

/// Delta-method standard error for the g-computation ATE, **conditional on the observed
/// covariate rows** (standard g-computation practice).
///
/// With gradient `g = (1/n) Σ_i [μ'(η_i1)·x_i1 − μ'(η_i0)·x_i0]` over the coefficient vector
/// (where `x_i1`/`x_i0` are row `i` with the treatment column set to `active`/`control`) and
/// `Cov(β̂) = φ·(XᵀWX)⁻¹` — the inverse Fisher information at the fit, `W = diag(w(η_i))`
/// with Bernoulli/logit / Poisson `w = μ'` and probit `w = φ²/(μ(1−μ))`, dispersion
/// `φ = RSS/(n−p)` for Gaussian and `1` otherwise — the SE is `sqrt(gᵀ Cov(β̂) g)`.
/// Returns `NaN` when the information matrix is singular.
#[allow(clippy::too_many_arguments)]
fn gcomp_delta_method_se(
    family: GlmFamily,
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    t_col: usize,
    coefficients: &[f64],
    active: f64,
    control: f64,
    deviance: f64,
) -> f64 {
    // Fisher information XᵀWX at the fitted coefficients, via a √W-scaled design copy.
    let mut x_w = vec![0.0; nrows * ncols];
    for r in 0..nrows {
        let mut eta = 0.0;
        for c in 0..ncols {
            eta += x_colmajor[c * nrows + r] * coefficients[c];
        }
        let sqrt_w = fisher_weight(family, eta).max(0.0).sqrt();
        for c in 0..ncols {
            x_w[c * nrows + r] = x_colmajor[c * nrows + r] * sqrt_w;
        }
    }
    let mut info = vec![0.0; ncols * ncols];
    form_xtx(&x_w, nrows, ncols, &mut info);
    let Some(cov_unscaled) = invert_square(&info, ncols) else {
        return f64::NAN;
    };
    let n = nrows as f64;
    let dispersion = match family {
        // For Gaussian/identity the fit's deviance is the RSS.
        GlmFamily::GaussianIdentity => deviance / (n - ncols as f64).max(1.0),
        GlmFamily::BinomialLogit
        | GlmFamily::BinomialProbit
        | GlmFamily::PoissonLog
        | GlmFamily::NegativeBinomial => 1.0,
    };

    // Gradient of the mean g-computation contrast w.r.t. the coefficient vector.
    let mut grad = vec![0.0; ncols];
    for r in 0..nrows {
        let mut eta_active = 0.0;
        let mut eta_control = 0.0;
        for c in 0..ncols {
            let coef = coefficients[c];
            if c == t_col {
                eta_active += active * coef;
                eta_control += control * coef;
            } else {
                let val = x_colmajor[c * nrows + r];
                eta_active += val * coef;
                eta_control += val * coef;
            }
        }
        let d1 = mean_derivative(family, eta_active);
        let d0 = mean_derivative(family, eta_control);
        for c in 0..ncols {
            let (x1, x0) = if c == t_col {
                (active, control)
            } else {
                let val = x_colmajor[c * nrows + r];
                (val, val)
            };
            grad[c] += d1 * x1 - d0 * x0;
        }
    }
    for g in &mut grad {
        *g /= n;
    }

    let mut quad = 0.0;
    for i in 0..ncols {
        for j in 0..ncols {
            quad += grad[i] * cov_unscaled[i * ncols + j] * grad[j];
        }
    }
    (dispersion * quad.max(0.0)).sqrt()
}

/// G-computation SE using score-exact GLM sandwich Cov(β̂) then the same
/// mean-contrast gradient as the Fisher delta-method.
///
/// Meat uses score contributions `s_i = u_i x_i` with
/// `u_i = (y_i−μ_i) V(μ_i)⁻¹ μ'(η_i)`; bread is Fisher `(XᵀWX)⁻¹`.
#[allow(clippy::too_many_arguments)]
fn gcomp_sandwich_se(
    kind: AnalyticSeKind,
    family: GlmFamily,
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    t_col: usize,
    coefficients: &[f64],
    y: &[f64],
    active: f64,
    control: f64,
    nb_alpha: f64,
    cluster_ids: Option<&[u32]>,
    multiway_ids: Option<&[Vec<u32>]>,
) -> Result<f64, EstimationError> {
    let (score_u, fisher_w) =
        glm_score_components(family, x_colmajor, nrows, ncols, coefficients, y, nb_alpha);
    let cov = sandwich_cov_matrix(
        kind,
        x_colmajor,
        nrows,
        ncols,
        &score_u,
        &fisher_w,
        cluster_ids,
        multiway_ids,
    )?;
    let Some(cov) = cov else {
        return Ok(gcomp_delta_method_se(
            family,
            x_colmajor,
            nrows,
            ncols,
            t_col,
            coefficients,
            active,
            control,
            0.0,
        ));
    };
    let grad =
        gcomp_gradient(family, x_colmajor, nrows, ncols, t_col, coefficients, active, control);
    let mut quad = 0.0;
    for i in 0..ncols {
        for j in 0..ncols {
            quad += grad[i] * cov[i * ncols + j] * grad[j];
        }
    }
    Ok(quad.max(0.0).sqrt())
}

fn sandwich_cov_matrix(
    kind: AnalyticSeKind,
    x: &[f64],
    nrows: usize,
    ncols: usize,
    score_u: &[f64],
    fisher_w: &[f64],
    cluster_ids: Option<&[u32]>,
    multiway_ids: Option<&[Vec<u32>]>,
) -> Result<Option<Vec<f64>>, EstimationError> {
    use crate::se::{require_clusters, require_multiway};
    use causal_stats::SandwichKind;
    if matches!(kind, AnalyticSeKind::Homoskedastic) {
        return Ok(None);
    }
    let cov = match kind {
        AnalyticSeKind::Homoskedastic => unreachable!(),
        AnalyticSeKind::Hc0 => {
            score_coefficient_covariance(x, nrows, ncols, score_u, fisher_w, SandwichKind::Hc0)
        }
        AnalyticSeKind::Hc1 => {
            score_coefficient_covariance(x, nrows, ncols, score_u, fisher_w, SandwichKind::Hc1)
        }
        AnalyticSeKind::Hc2 => {
            score_coefficient_covariance(x, nrows, ncols, score_u, fisher_w, SandwichKind::Hc2)
        }
        AnalyticSeKind::Hc3 => {
            score_coefficient_covariance(x, nrows, ncols, score_u, fisher_w, SandwichKind::Hc3)
        }
        AnalyticSeKind::Cluster => {
            let groups = require_clusters(cluster_ids, nrows)?;
            score_coefficient_covariance(
                x,
                nrows,
                ncols,
                score_u,
                fisher_w,
                SandwichKind::Cluster { groups },
            )
        }
        AnalyticSeKind::Multiway => {
            let dims = require_multiway(multiway_ids, nrows)?;
            let refs: Vec<&[u32]> = dims.iter().map(Vec::as_slice).collect();
            score_coefficient_covariance(
                x,
                nrows,
                ncols,
                score_u,
                fisher_w,
                SandwichKind::Multiway { dimensions: &refs },
            )
        }
        AnalyticSeKind::NeweyWest { lag } => score_coefficient_covariance(
            x,
            nrows,
            ncols,
            score_u,
            fisher_w,
            SandwichKind::NeweyWest { lag },
        ),
        AnalyticSeKind::PanelClusterHac { lag } => {
            let groups = require_clusters(cluster_ids, nrows)?;
            score_coefficient_covariance(
                x,
                nrows,
                ncols,
                score_u,
                fisher_w,
                SandwichKind::PanelClusterHac { groups, lag },
            )
        }
    };
    Ok(Some(cov.unwrap_or_else(|_| vec![f64::NAN; ncols * ncols])))
}

/// Per-row GLM score multiplier `u_i` and Fisher weight `w_i`.
///
/// Score contribution is `s_i = u_i x_i` with
/// `u_i = (y−μ) V(μ)⁻¹ μ'(η)`; `w_i` is the IRLS Fisher weight for bread `XᵀWX`.
fn glm_score_components(
    family: GlmFamily,
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    coefficients: &[f64],
    y: &[f64],
    nb_alpha: f64,
) -> (Vec<f64>, Vec<f64>) {
    use causal_kernels::norm_pdf;
    let mut score_u = vec![0.0; nrows];
    let mut fisher_w = vec![0.0; nrows];
    let alpha = nb_alpha.max(0.0);
    for r in 0..nrows {
        let mut eta = 0.0;
        for c in 0..ncols {
            eta += x_colmajor[c * nrows + r] * coefficients[c];
        }
        let mu = family.mean_from_eta(eta);
        let (u, w) = match family {
            GlmFamily::GaussianIdentity => {
                // V=1, μ'=1 → u = y−μ, w = 1
                (y[r] - mu, 1.0)
            }
            GlmFamily::BinomialLogit => {
                let mu = mu.clamp(1e-9, 1.0 - 1e-9);
                let var = (mu * (1.0 - mu)).max(1e-12);
                // μ' = var, u = (y−μ)/var * var = y−μ; w = var
                (y[r] - mu, var)
            }
            GlmFamily::BinomialProbit => {
                let mu = mu.clamp(1e-9, 1.0 - 1e-9);
                let phi = norm_pdf(eta).max(1e-12);
                let var = (mu * (1.0 - mu)).max(1e-12);
                // u = (y−μ) φ / V; w = φ² / V
                ((y[r] - mu) * phi / var, (phi * phi) / var)
            }
            GlmFamily::PoissonLog => {
                let mu = mu.max(1e-12);
                // V=μ, μ'=μ → u = y−μ; w = μ
                (y[r] - mu, mu)
            }
            GlmFamily::NegativeBinomial => {
                let mu = mu.max(1e-12);
                let var = (mu + alpha * mu * mu).max(1e-12);
                // μ'=μ, u = (y−μ) μ / V; w = μ² / V
                ((y[r] - mu) * mu / var, (mu * mu) / var)
            }
        };
        score_u[r] = u;
        fisher_w[r] = w;
    }
    (score_u, fisher_w)
}

fn gcomp_gradient(
    family: GlmFamily,
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    t_col: usize,
    coefficients: &[f64],
    active: f64,
    control: f64,
) -> Vec<f64> {
    let n = nrows as f64;
    let mut grad = vec![0.0; ncols];
    for r in 0..nrows {
        let mut eta_active = 0.0;
        let mut eta_control = 0.0;
        for c in 0..ncols {
            let coef = coefficients[c];
            if c == t_col {
                eta_active += active * coef;
                eta_control += control * coef;
            } else {
                let val = x_colmajor[c * nrows + r];
                eta_active += val * coef;
                eta_control += val * coef;
            }
        }
        let d1 = mean_derivative(family, eta_active);
        let d0 = mean_derivative(family, eta_control);
        for c in 0..ncols {
            let (x1, x0) = if c == t_col {
                (active, control)
            } else {
                let val = x_colmajor[c * nrows + r];
                (val, val)
            };
            grad[c] += d1 * x1 - d0 * x0;
        }
    }
    for g in &mut grad {
        *g /= n;
    }
    grad
}

#[cfg(test)]
#[allow(clippy::many_single_char_names, clippy::float_cmp)]
mod tests {
    use std::sync::Arc;

    use causal_core::{
        CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint, SmallRoleSet,
        TargetPopulation, ValueType, VariableId,
    };
    use causal_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
    };
    use causal_expr::ExprId;
    use causal_expr::IdentifiedEstimand;

    use super::*;
    use crate::overlap::OverlapPolicy;

    /// Binary-outcome SCM: `Z ~ U(-0.5, 0.5)`, `T ∈ {0,1}`, `logit(Y=1) = -0.5 + 2T + Z`.
    fn binary_scm(n: usize, seed: u64) -> (TabularData, IdentifiedEstimand) {
        let mut rng = ExecutionContext::for_tests(seed).rng.stream(0xABCD_u64);
        let mut t = vec![0.0; n];
        let mut z = vec![0.0; n];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let ti = (i % 2) as f64;
            let zi = (i as f64) / (n as f64) - 0.5;
            let logit = -0.5 + 2.0 * ti + zi;
            let p = 1.0 / (1.0 + (-logit).exp());
            let yi = if rng.next_f64() < p { 1.0 } else { 0.0 };
            t[i] = ti;
            z[i] = zi;
            y[i] = yi;
        }

        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "t",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "z",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(t),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(y),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(2),
                    Arc::from(z),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let estimand = IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from([VariableId::from_raw(2)]),
            ExprId::from_raw(0),
        );
        (TabularData::new(storage), estimand)
    }

    fn ctx() -> ExecutionContext {
        ExecutionContext::for_tests(11)
    }

    #[test]
    fn recovers_positive_ate_on_binary_outcome() {
        let (data, estimand) = binary_scm(4000, 1);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let est = GlmAdjustmentAte { bootstrap_replicates: 30, ..GlmAdjustmentAte::new() };
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = GlmAdjustmentWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        assert!(effect.ate > 0.0, "ate={}", effect.ate);
        assert!(effect.ate < 1.0, "ate={}", effect.ate);
        assert!(effect.se_bootstrap.is_some());
    }

    #[test]
    fn works_with_efficient_backdoor_estimand() {
        let (data, mut estimand) = binary_scm(2000, 2);
        estimand.method = Arc::from("backdoor.efficient");
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let est = GlmAdjustmentAte { bootstrap_replicates: 0, ..GlmAdjustmentAte::new() };
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = GlmAdjustmentWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        assert!(effect.ate > 0.0, "ate={}", effect.ate);
    }

    #[test]
    fn rejects_require_diagnostics_overlap() {
        let (data, estimand) = binary_scm(200, 3);
        let est = GlmAdjustmentAte {
            overlap: OverlapPolicy::require_diagnostics(),
            ..GlmAdjustmentAte::new()
        };
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let err = est.prepare(&data, &estimand, &query).unwrap_err();
        assert!(matches!(err, EstimationError::Overlap { .. }));
    }

    /// Gaussian SCM with homogeneous contrasts: `Y = 1 + 2T + Z + noise` (no interactions).
    fn gaussian_scm(n: usize, seed: u64) -> (TabularData, IdentifiedEstimand) {
        let mut rng = ExecutionContext::for_tests(seed).rng.stream(0xFEED_u64);
        let mut t = vec![0.0; n];
        let mut z = vec![0.0; n];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let ti = (i % 2) as f64;
            let zi = (i as f64) / (n as f64) - 0.5;
            let noise = rng.next_f64() - 0.5;
            t[i] = ti;
            z[i] = zi;
            y[i] = 1.0 + 2.0 * ti + zi + noise;
        }

        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "t",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "z",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(t),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(y),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(2),
                    Arc::from(z),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let estimand = IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from([VariableId::from_raw(2)]),
            ExprId::from_raw(0),
        );
        (TabularData::new(storage), estimand)
    }

    #[test]
    fn gaussian_delta_method_se_positive_and_near_bootstrap() {
        // Homogeneous contrasts: every per-row contrast equals β_T exactly, so the old
        // spread-based formula (sample_std(diffs)/√n) returned ≈0 regardless of coefficient
        // noise. The delta-method SE must be positive and on the bootstrap's scale.
        let (data, estimand) = gaussian_scm(400, 6);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let est = GlmAdjustmentAte {
            bootstrap_replicates: 200,
            family: GlmFamily::GaussianIdentity,
            ..GlmAdjustmentAte::new()
        };
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = GlmAdjustmentWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        assert!(effect.se_analytic > 0.0, "se_analytic={}", effect.se_analytic);
        let boot = effect.se_bootstrap.unwrap();
        assert!(
            effect.se_analytic < 3.0 * boot && effect.se_analytic > boot / 3.0,
            "se_analytic={} se_bootstrap={boot}",
            effect.se_analytic
        );
    }

    #[test]
    fn probit_adjustment_fits_binary_outcome() {
        let (data, estimand) = binary_scm(200, 7);
        let est = GlmAdjustmentAte {
            family: GlmFamily::BinomialProbit,
            bootstrap_replicates: 0,
            ..GlmAdjustmentAte::new()
        };
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = GlmAdjustmentWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        assert!(effect.ate.is_finite());
        assert!(effect.se_analytic.is_finite() && effect.se_analytic > 0.0);
    }

    #[test]
    fn rejects_non_binary_outcome() {
        let (data, estimand) = binary_scm(200, 4);
        // Replace outcome with a non-binary value to trigger the validation path.
        let (data, _) = data.with_appended_float("dummy", Arc::from(vec![0.0; 200])).unwrap();
        let bad_y = (0..200).map(f64::from).collect::<Vec<_>>();
        let data = data.with_replaced_float(VariableId::from_raw(1), Arc::from(bad_y)).unwrap();
        let est = GlmAdjustmentAte::new();
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let err = est.prepare(&data, &estimand, &query).unwrap_err();
        assert!(matches!(err, EstimationError::Unsupported { .. }));
    }

    #[test]
    fn recovers_att_via_gcomp() {
        let (data, estimand) = binary_scm(800, 5);
        let est = GlmAdjustmentAte { bootstrap_replicates: 0, ..GlmAdjustmentAte::new() };
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
                .with_target_population(TargetPopulation::Treated);
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = GlmAdjustmentWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        assert!(effect.ate.is_finite());
    }

    #[test]
    fn negbin_gcomp_accepts_nonnegative_counts() {
        let n = 200usize;
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "t",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let t: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 3.0 }).collect();
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(t),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(y),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let data = TabularData::new(storage);
        let estimand =
            IdentifiedEstimand::backdoor("backdoor.adjustment", Arc::from([]), ExprId::from_raw(0));
        let est = GlmAdjustmentAte {
            family: GlmFamily::NegativeBinomial,
            bootstrap_replicates: 0,
            ..GlmAdjustmentAte::new()
        };
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = GlmAdjustmentWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        assert!(effect.ate.is_finite() && effect.ate > 0.0);
    }

    #[test]
    fn hc1_sandwich_se_finite_without_bootstrap() {
        let (data, estimand) = binary_scm(800, 9);
        let est = GlmAdjustmentAte {
            bootstrap_replicates: 0,
            se_kind: AnalyticSeKind::Hc1,
            ..GlmAdjustmentAte::new()
        };
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = GlmAdjustmentWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        assert!(effect.se_analytic.is_finite() && effect.se_analytic > 0.0);
    }
}
