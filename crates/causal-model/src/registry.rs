//! Mechanism registry and auto-assignment.
//!
//! Assignment returns candidates and scores; there is no silent default family.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::needless_range_loop
)]

use std::sync::Arc;

use causal_core::{RoleHint, VariableId};
use causal_data::{TableView, TabularData};
use causal_graph::DenseNodeId;
use causal_stats::{
    DenseLinearAlgebra, FaerBackend, GlmDesignRef, GlmFamily, GlmOptions, LeastSquaresWorkspace,
    MultinomialDesignRef, fit_glm, fit_multinomial_logit,
};

use crate::batch::ParentBatch;
use crate::compile::{
    CompiledCausalModel, CompiledMechanismStore, MechanismSlot, ParentGatherPlan,
};
use crate::error::ModelError;
use crate::mechanism::log_prob_column;

/// Candidate mechanism family known to the registry.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum MechanismFamily {
    /// Linear Gaussian additive noise (invertible).
    LinearGaussian,
    /// Constant (root or intercept-only).
    Constant,
    /// Discrete categorical (unconditional root or parent-conditional softmax).
    Discrete,
    /// Hierarchical linear Gaussian (EB / group partial pooling).
    HierarchicalLinear,
    /// Hierarchical Bernoulli-logit GLM (EB / group shrinkage) → [`MechanismSlot::Discrete`].
    HierarchicalGlm,
    /// Single-equation Bayesian VAR (Minnesota prior).
    Bvar,
    /// Linear Gaussian state-space observation mechanism (Kalman EM fit).
    LinearGaussianStateSpace,
    /// Gaussian-process regression mechanism (feature `gaussian-process`).
    GaussianProcess,
}

impl MechanismFamily {
    /// Registry id string.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::LinearGaussian => "linear_gaussian",
            Self::Constant => "constant",
            Self::Discrete => "discrete",
            Self::HierarchicalLinear => "hierarchical_linear",
            Self::HierarchicalGlm => "hierarchical_glm",
            Self::Bvar => "bvar",
            Self::LinearGaussianStateSpace => "lgssm",
            Self::GaussianProcess => "gaussian_process",
        }
    }
}

/// Scored candidate for one node.
#[derive(Clone, Debug)]
pub struct MechanismCandidate {
    /// Family.
    pub family: MechanismFamily,
    /// Validation score (higher is better; e.g. negative MSE or log-lik).
    pub score: f64,
    /// Estimated fit cost (relative).
    pub fit_cost: f64,
    /// Estimated evaluation cost (relative).
    pub eval_cost: f64,
}

/// Result of auto-assignment for one node.
#[derive(Clone, Debug)]
pub struct MechanismAssignment {
    /// Dense node.
    pub node: DenseNodeId,
    /// Variable.
    pub variable: VariableId,
    /// All scored candidates (sorted descending by score).
    pub candidates: Arc<[MechanismCandidate]>,
    /// Selected family (must be chosen explicitly from candidates).
    pub selected: MechanismFamily,
    /// Fitted slot.
    pub fitted: MechanismSlot,
    /// Families that failed to score/fit, with error messages.
    pub failed_families: Arc<[(MechanismFamily, String)]>,
}

/// Registry of mechanism families.
#[derive(Clone, Debug)]
pub struct MechanismRegistry {
    /// Families considered for continuous nodes.
    pub continuous: Arc<[MechanismFamily]>,
    /// Families considered for discrete / low-cardinality nodes.
    pub discrete: Arc<[MechanismFamily]>,
}

impl Default for MechanismRegistry {
    fn default() -> Self {
        Self::standard()
    }
}

impl MechanismRegistry {
    /// Standard registry (core families).
    #[must_use]
    pub fn standard() -> Self {
        Self {
            continuous: Arc::from(vec![MechanismFamily::LinearGaussian, MechanismFamily::Constant]),
            discrete: Arc::from(vec![MechanismFamily::Discrete, MechanismFamily::Constant]),
        }
    }

    /// Extended continuous registry including hierarchical / BVAR / LGSSM / GP.
    #[must_use]
    pub fn with_bayesian_families() -> Self {
        #[cfg(feature = "gaussian-process")]
        let continuous = {
            let mut continuous = vec![
                MechanismFamily::LinearGaussian,
                MechanismFamily::HierarchicalLinear,
                MechanismFamily::Bvar,
                MechanismFamily::LinearGaussianStateSpace,
                MechanismFamily::Constant,
            ];
            continuous.insert(continuous.len() - 1, MechanismFamily::GaussianProcess);
            continuous
        };
        #[cfg(not(feature = "gaussian-process"))]
        let continuous = vec![
            MechanismFamily::LinearGaussian,
            MechanismFamily::HierarchicalLinear,
            MechanismFamily::Bvar,
            MechanismFamily::LinearGaussianStateSpace,
            MechanismFamily::Constant,
        ];
        let discrete = vec![
            MechanismFamily::Discrete,
            MechanismFamily::HierarchicalGlm,
            MechanismFamily::Constant,
        ];
        Self { continuous: Arc::from(continuous), discrete: Arc::from(discrete) }
    }

    /// Assign and fit all nodes. Requires an explicit selection policy.
    ///
    /// # Errors
    ///
    /// Data / fit failures, or empty candidate sets.
    pub fn assign_and_fit(
        &self,
        model: &CompiledCausalModel,
        data: &TabularData,
        policy: SelectionPolicy,
    ) -> Result<(CompiledMechanismStore, Vec<MechanismAssignment>), ModelError> {
        let n = model.n_nodes();
        let nrows = data.row_count();
        if nrows == 0 {
            return Err(ModelError::Shape { message: "empty data for mechanism fit".into() });
        }
        let mut slots = vec![MechanismSlot::Vacant; n];
        let mut assignments = Vec::with_capacity(n);
        let backend = FaerBackend;
        let mut ls_ws = LeastSquaresWorkspace::default();

        for gather in model.parent_gathers.iter() {
            let node = gather.child;
            let var = model.output_layout.variables[node.as_usize()];
            let y = data.float64_values(var).map_err(ModelError::from)?;
            let is_discrete = is_low_cardinality(&y, 8);
            let families: &[MechanismFamily] =
                if is_discrete { &self.discrete } else { &self.continuous };

            let mut candidates = Vec::new();
            let mut failed = Vec::new();
            for &family in families {
                match score_family(family, gather, model, data, &y, backend, &mut ls_ws) {
                    Ok(c) => candidates.push(c),
                    Err(e) => failed.push((family, e.to_string())),
                }
            }
            if candidates.is_empty() {
                let detail = failed
                    .iter()
                    .map(|(f, e)| format!("{f:?}: {e}"))
                    .collect::<Vec<_>>()
                    .join("; ");
                return Err(ModelError::Unsupported {
                    message: format!(
                        "no mechanism candidates for variable {var} (failures: {detail})"
                    ),
                });
            }
            candidates
                .sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
            let selected = policy.select(&candidates).ok_or_else(|| ModelError::Unsupported {
                message: "selection policy produced no family".into(),
            })?;
            let fitted = fit_family(selected, gather, model, data, &y, backend, &mut ls_ws)?;
            slots[node.as_usize()] = fitted.clone();
            assignments.push(MechanismAssignment {
                node,
                variable: var,
                candidates: Arc::from(candidates),
                selected,
                fitted,
                failed_families: Arc::from(failed),
            });
        }

        Ok((CompiledMechanismStore { slots: Arc::from(slots) }, assignments))
    }
}

/// How to pick among scored candidates (no silent fallback).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum SelectionPolicy {
    /// Highest validation score.
    BestScore,
    /// Require the named family to appear; error if missing.
    RequireFamily(MechanismFamily),
}

impl SelectionPolicy {
    /// Select a family.
    #[must_use]
    pub fn select(self, candidates: &[MechanismCandidate]) -> Option<MechanismFamily> {
        match self {
            Self::BestScore => candidates.first().map(|c| c.family),
            Self::RequireFamily(fam) => {
                candidates.iter().find(|c| c.family == fam).map(|c| c.family)
            }
        }
    }
}

fn is_low_cardinality(y: &[f64], max_levels: usize) -> bool {
    let mut vals: Vec<i64> =
        y.iter().filter(|v| v.is_finite()).map(|v| (v * 1e6).round() as i64).collect();
    vals.sort_unstable();
    vals.dedup();
    !vals.is_empty() && vals.len() <= max_levels
}

fn score_family(
    family: MechanismFamily,
    gather: &ParentGatherPlan,
    model: &CompiledCausalModel,
    data: &TabularData,
    y: &[f64],
    backend: FaerBackend,
    ls_ws: &mut LeastSquaresWorkspace,
) -> Result<MechanismCandidate, ModelError> {
    let fitted = fit_family(family, gather, model, data, y, backend, ls_ws)?;
    let score = match &fitted {
        MechanismSlot::LinearGaussian { intercept, coeffs, sigma }
        | MechanismSlot::HierarchicalLinear { intercept, coeffs, sigma, .. }
        | MechanismSlot::Bvar { intercept, coeffs, sigma } => {
            let mse = residual_mse(gather, model, data, y, *intercept, coeffs)?;
            -mse - sigma.ln().abs() * 0.01
        }
        MechanismSlot::Constant { value } => {
            let mse = y.iter().map(|yi| (yi - value).powi(2)).sum::<f64>() / y.len().max(1) as f64;
            -mse
        }
        MechanismSlot::Discrete { support, probs, logit_coeffs } => match logit_coeffs {
            None => {
                let ent: f64 =
                    probs.iter().map(|p| if *p > 0.0 { -p * p.ln() } else { 0.0 }).sum();
                -ent
            }
            Some(logits) => {
                discrete_mean_loglik(gather, model, data, y, support, logits)?
            }
        },
        MechanismSlot::LinearGaussianStateSpace { process_std, obs_std, .. } => {
            let mse = y.iter().map(|yi| yi.powi(2)).sum::<f64>() / y.len().max(1) as f64;
            -mse - (process_std + obs_std).ln().abs() * 0.01
        }
        MechanismSlot::GaussianProcess { noise_std, .. } => {
            let mse = y.iter().map(|yi| yi.powi(2)).sum::<f64>() / y.len().max(1) as f64;
            -mse - noise_std.ln().abs() * 0.01
        }
        _ => f64::NEG_INFINITY,
    };
    Ok(MechanismCandidate {
        family,
        score,
        fit_cost: 1.0 + gather.n_parents() as f64,
        eval_cost: 1.0 + gather.n_parents() as f64,
    })
}

fn fit_family(
    family: MechanismFamily,
    gather: &ParentGatherPlan,
    model: &CompiledCausalModel,
    data: &TabularData,
    y: &[f64],
    backend: FaerBackend,
    ls_ws: &mut LeastSquaresWorkspace,
) -> Result<MechanismSlot, ModelError> {
    let n = y.len();
    match family {
        MechanismFamily::Constant => {
            let mean = y.iter().sum::<f64>() / n.max(1) as f64;
            Ok(MechanismSlot::Constant { value: mean })
        }
        MechanismFamily::Discrete => {
            let mut pairs: Vec<(i64, f64, usize)> = Vec::new();
            for &yi in y {
                if !yi.is_finite() {
                    continue;
                }
                let key = (yi * 1e6).round() as i64;
                if let Some(e) = pairs.iter_mut().find(|(k, _, _)| *k == key) {
                    e.2 += 1;
                } else {
                    pairs.push((key, yi, 1));
                }
            }
            if pairs.is_empty() {
                return Err(ModelError::Shape {
                    message: "no finite values for discrete fit".into(),
                });
            }
            // Stable support order → stable baseline-category reference (index 0).
            pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let total = pairs.iter().map(|(_, _, c)| *c).sum::<usize>() as f64;
            let support: Vec<f64> = pairs.iter().map(|(_, v, _)| *v).collect();
            let probs: Vec<f64> = pairs.iter().map(|(_, _, c)| *c as f64 / total).collect();
            let k = support.len();
            let p = gather.n_parents();
            if p == 0 {
                return Ok(MechanismSlot::Discrete {
                    support: Arc::from(support),
                    probs: Arc::from(probs),
                    logit_coeffs: None,
                });
            }
            // Parent-conditional: baseline-category multinomial logit MLE (Fisher / IRLS).
            // Coefficients are true softmax logits; category 0 is the reference (zeros).
            let ncols = 1 + p;
            let mut x = vec![0.0; n * ncols];
            for r in 0..n {
                x[r] = 1.0;
            }
            for (pi, &parent) in gather.parents.iter().enumerate() {
                let var = model.output_layout.variables[parent.as_usize()];
                let col = data.float64_values(var).map_err(ModelError::from)?;
                let base = (1 + pi) * n;
                x[base..base + n].copy_from_slice(&col[..n]);
            }
            let mut y_cat = vec![0u32; n];
            for (r, &yi) in y.iter().enumerate() {
                let Some(idx) = support.iter().position(|&s| (s - yi).abs() < 1e-12) else {
                    return Err(ModelError::Shape {
                        message: "discrete outcome not in fitted support".into(),
                    });
                };
                y_cat[r] = u32::try_from(idx).map_err(|_| ModelError::Shape {
                    message: "too many discrete categories".into(),
                })?;
            }
            let fit = fit_multinomial_logit(
                MultinomialDesignRef {
                    x_colmajor: &x,
                    nrows: n,
                    ncols,
                    y_category: &y_cat,
                    n_categories: k,
                },
                &backend,
                ls_ws,
                &GlmOptions::default(),
            )?;
            // Refuse non-converged fits; separation is allowed (near-deterministic
            // conditionals → large logits; softmax evaluation remains well-defined).
            if !fit.converged {
                return Err(ModelError::Numerical {
                    message: format!(
                        "multinomial logit did not converge (iters={}, deviance={})",
                        fit.iterations, fit.deviance
                    ),
                });
            }
            Ok(MechanismSlot::Discrete {
                support: Arc::from(support),
                probs: Arc::from(probs),
                logit_coeffs: Some(Arc::from(fit.coefficients)),
            })
        }
        MechanismFamily::LinearGaussian => {
            fit_linear_gaussian(gather, model, data, y, backend, ls_ws, 0.0)
        }
        MechanismFamily::HierarchicalLinear => {
            fit_hierarchical_linear(gather, model, data, y, backend, ls_ws)
        }
        MechanismFamily::HierarchicalGlm => fit_hierarchical_glm(gather, model, data, y, backend, ls_ws),
        MechanismFamily::Bvar => fit_bvar_minnesota(gather, model, data, y, backend, ls_ws),
        MechanismFamily::LinearGaussianStateSpace => {
            fit_lgssm_kalman_em(gather, model, data, y, backend, ls_ws)
        }
        MechanismFamily::GaussianProcess => {
            #[cfg(feature = "gaussian-process")]
            {
                fit_gaussian_process(gather, model, data, y)
            }
            #[cfg(not(feature = "gaussian-process"))]
            {
                let _ = (gather, model, data, y, backend, ls_ws);
                Err(ModelError::Unsupported {
                    message: "GaussianProcess requires feature `gaussian-process`".into(),
                })
            }
        }
    }
}

fn gather_parent_cols(
    gather: &ParentGatherPlan,
    model: &CompiledCausalModel,
    data: &TabularData,
) -> Result<Vec<Vec<f64>>, ModelError> {
    let mut parent_cols: Vec<Vec<f64>> = Vec::with_capacity(gather.n_parents());
    for &parent in gather.parents.iter() {
        let var = model.output_layout.variables[parent.as_usize()];
        parent_cols.push(data.float64_values(var).map_err(ModelError::from)?);
    }
    Ok(parent_cols)
}

/// Empirical-Bayes hierarchical linear: estimate τ² / λ from OLS, optional UnitId
/// random-intercept demeaning for partial pooling.
fn fit_hierarchical_linear(
    gather: &ParentGatherPlan,
    model: &CompiledCausalModel,
    data: &TabularData,
    y: &[f64],
    backend: FaerBackend,
    ls_ws: &mut LeastSquaresWorkspace,
) -> Result<MechanismSlot, ModelError> {
    let n = y.len();
    let ols = fit_linear_gaussian(gather, model, data, y, backend, ls_ws, 0.0)?;
    let MechanismSlot::LinearGaussian { intercept: ols_int, coeffs: ols_coeffs, sigma: ols_sigma } =
        ols
    else {
        return Err(ModelError::Unsupported { message: "hierarchical base fit failed".into() });
    };
    let p = ols_coeffs.len();
    // Method-of-moments EB: τ² ≈ mean(β̂²) − σ²·mean(diag((X'X)^{-1})) proxy;
    // use simplified τ² = mean(β̂²) clamped, λ = σ² / τ².
    let mean_b2 = if p == 0 {
        0.0
    } else {
        ols_coeffs.iter().map(|b| b * b).sum::<f64>() / p as f64
    };
    let tau2 = (mean_b2 - ols_sigma * ols_sigma / n.max(1) as f64).max(1e-8);
    let mut lambda = (ols_sigma * ols_sigma / tau2).clamp(1e-6, 1e6);

    // Optional UnitId random-intercept: demean within groups, then refit EB ridge.
    let mut y_work = y.to_vec();
    let mut group_tau = 0.0;
    if let Some(groups) = unit_id_groups(data, y.len()) {
        let (demeaned, tau) = demean_by_group(&y_work, &groups);
        y_work = demeaned;
        group_tau = tau;
        // Re-estimate λ on demeaned series.
        let ols2 = fit_linear_gaussian(gather, model, data, &y_work, backend, ls_ws, 0.0)?;
        if let MechanismSlot::LinearGaussian { coeffs, sigma, .. } = ols2 {
            let mean_b2 = if coeffs.is_empty() {
                0.0
            } else {
                coeffs.iter().map(|b| b * b).sum::<f64>() / coeffs.len() as f64
            };
            let tau2 = (mean_b2 - sigma * sigma / n.max(1) as f64).max(1e-8);
            lambda = (sigma * sigma / tau2).clamp(1e-6, 1e6);
        }
    }

    let slot = fit_linear_gaussian(gather, model, data, &y_work, backend, ls_ws, lambda)?;
    match slot {
        MechanismSlot::LinearGaussian { intercept, coeffs, sigma } => {
            // Restore population intercept when we demeaned.
            let intercept = if group_tau > 0.0 {
                y.iter().sum::<f64>() / n.max(1) as f64
                    - coeffs.iter().enumerate().try_fold(0.0, |acc, (i, c)| {
                        let var = model.output_layout.variables[gather.parents[i].as_usize()];
                        let col = data.float64_values(var).map_err(ModelError::from)?;
                        let mean = col.iter().sum::<f64>() / n.max(1) as f64;
                        Ok::<_, ModelError>(acc + c * mean)
                    })?
            } else {
                intercept
            };
            let _ = (ols_int, group_tau);
            Ok(MechanismSlot::HierarchicalLinear {
                intercept,
                coeffs,
                sigma,
                shrinkage: lambda,
            })
        }
        other => Ok(other),
    }
}

/// Hierarchical Bernoulli logit with EB ridge (and optional UnitId demeaning of the
/// linear predictor target via frequency offsets — here: ridge λ from OLS proxy on
/// working residuals).
fn fit_hierarchical_glm(
    gather: &ParentGatherPlan,
    model: &CompiledCausalModel,
    data: &TabularData,
    y: &[f64],
    backend: FaerBackend,
    ls_ws: &mut LeastSquaresWorkspace,
) -> Result<MechanismSlot, ModelError> {
    let n = y.len();
    let binary = y.iter().all(|&yi| yi == 0.0 || yi == 1.0);
    if !binary {
        return Err(ModelError::Unsupported {
            message: "HierarchicalGlm requires binary {0,1} outcomes".into(),
        });
    }
    // EB λ from linear-probability OLS moments.
    let ols = fit_linear_gaussian(gather, model, data, y, backend, ls_ws, 0.0)?;
    let lambda = match &ols {
        MechanismSlot::LinearGaussian { coeffs, sigma, .. } => {
            let p = coeffs.len().max(1);
            let mean_b2 = coeffs.iter().map(|b| b * b).sum::<f64>() / p as f64;
            let tau2 = (mean_b2 - sigma * sigma / n.max(1) as f64).max(1e-8);
            (sigma * sigma / tau2).clamp(1e-4, 1e3)
        }
        _ => 1.0,
    };
    let p = gather.n_parents();
    let ncols = 1 + p;
    let mut x = vec![0.0; n * ncols];
    for r in 0..n {
        x[r] = 1.0;
    }
    for (pi, &parent) in gather.parents.iter().enumerate() {
        let var = model.output_layout.variables[parent.as_usize()];
        let col = data.float64_values(var).map_err(ModelError::from)?;
        let base = (1 + pi) * n;
        x[base..base + n].copy_from_slice(&col[..n]);
    }
    let mut opts = GlmOptions::default();
    opts.ridge_on_separation = Some(lambda);
    let fit = fit_glm(
        GlmFamily::BinomialLogit,
        GlmDesignRef { x_colmajor: &x, nrows: n, ncols, y },
        &backend,
        ls_ws,
        &opts,
    )
    .map_err(|e| ModelError::Numerical { message: e.to_string() })?;
    if !fit.converged {
        return Err(ModelError::Numerical {
            message: "hierarchical GLM logit did not converge".into(),
        });
    }
    // Encode as 2-category Discrete with baseline-category logits (cat0 = 0, cat1 = β).
    let mut logit_coeffs = vec![0.0; 2 * ncols];
    logit_coeffs[ncols..].copy_from_slice(&fit.coefficients[..ncols]);
    let n1 = y.iter().filter(|&&yi| yi == 1.0).count() as f64;
    let p1 = n1 / n.max(1) as f64;
    Ok(MechanismSlot::Discrete {
        support: Arc::from([0.0, 1.0]),
        probs: Arc::from([1.0 - p1, p1]),
        logit_coeffs: Some(Arc::from(logit_coeffs)),
    })
}

/// Minnesota-prior single-equation BVAR: prior variance φ/(ℓ+1)² on coefficient ℓ.
fn fit_bvar_minnesota(
    gather: &ParentGatherPlan,
    model: &CompiledCausalModel,
    data: &TabularData,
    y: &[f64],
    backend: FaerBackend,
    ls_ws: &mut LeastSquaresWorkspace,
) -> Result<MechanismSlot, ModelError> {
    let n = y.len();
    let p = gather.n_parents();
    let ncols = 1 + p;
    let phi: f64 = 0.2; // overall tightness
    let mut x = vec![0.0; n * ncols];
    for r in 0..n {
        x[r] = 1.0;
    }
    for (pi, &parent) in gather.parents.iter().enumerate() {
        let var = model.output_layout.variables[parent.as_usize()];
        let col = data.float64_values(var).map_err(ModelError::from)?;
        let base = (1 + pi) * n;
        x[base..base + n].copy_from_slice(&col[..n]);
    }
    // Augment with Minnesota prior pseudo-observations: √(1/v_j) * e_j → 0.
    let extra = ncols; // intercept + each lag coeff
    let mut x2 = vec![0.0; (n + extra) * ncols];
    let mut y2 = vec![0.0; n + extra];
    for c in 0..ncols {
        for r in 0..n {
            x2[c * (n + extra) + r] = x[c * n + r];
        }
    }
    y2[..n].copy_from_slice(y);
    // Intercept prior: loose (v = 100 * φ)
    let v0: f64 = (100.0 * phi).max(1e-6);
    x2[0 * (n + extra) + n] = (1.0 / v0).sqrt();
    for j in 0..p {
        let lag = (j + 1) as f64;
        let v: f64 = (phi / (lag * lag)).max(1e-8);
        x2[(1 + j) * (n + extra) + (n + 1 + j)] = (1.0 / v).sqrt();
    }
    let fit =
        backend.least_squares(&x2, n + extra, ncols, &y2, ls_ws).map_err(ModelError::from)?;
    let intercept = fit.coefficients[0];
    let coeffs: Arc<[f64]> = Arc::from(fit.coefficients[1..].to_vec());
    let sigma = (fit.rss / (n.saturating_sub(ncols)).max(1) as f64).sqrt().max(1e-8);
    Ok(MechanismSlot::Bvar { intercept, coeffs, sigma })
}

/// Scalar LGSSM on parent-adjusted residuals via EM (Kalman filter/smoother).
fn fit_lgssm_kalman_em(
    gather: &ParentGatherPlan,
    model: &CompiledCausalModel,
    data: &TabularData,
    y: &[f64],
    backend: FaerBackend,
    ls_ws: &mut LeastSquaresWorkspace,
) -> Result<MechanismSlot, ModelError> {
    let lg = fit_linear_gaussian(gather, model, data, y, backend, ls_ws, 0.0)?;
    let (intercept, coeffs, sigma) = match lg {
        MechanismSlot::LinearGaussian { intercept, coeffs, sigma } => (intercept, coeffs, sigma),
        _ => {
            return Err(ModelError::Unsupported {
                message: "lgssm fit requires linear base".into(),
            });
        }
    };
    let parent_cols = gather_parent_cols(gather, model, data)?;
    let mut resid = vec![0.0; y.len()];
    for r in 0..y.len() {
        let mut pred = intercept;
        for (p, col) in parent_cols.iter().enumerate() {
            pred += coeffs[p] * col[r];
        }
        resid[r] = y[r] - pred;
    }
    let (a, process_std, obs_std, initial_mean) = lgssm_em(&resid, 25);
    let _ = sigma;
    Ok(MechanismSlot::LinearGaussianStateSpace {
        a,
        process_std: process_std.max(1e-8),
        obs_std: obs_std.max(1e-8),
        initial_mean,
    })
}

/// EM for scalar LGSSM: x_t = a x_{t-1} + q ε, y_t = x_t + r η.
fn lgssm_em(y: &[f64], max_iters: usize) -> (f64, f64, f64, f64) {
    let n = y.len();
    if n < 3 {
        let (a, q) = fit_ar1(y);
        return (a, q, q.max(1e-8), y.first().copied().unwrap_or(0.0));
    }
    let mut a = 0.8;
    let mut q = 1.0; // process variance
    let mut r = 1.0; // obs variance
    let mut x0 = y[0];
    let p0 = 1.0;
    for _ in 0..max_iters {
        let (x_f, p_f, x_pred, p_pred) = crate::lgssm::kalman_filter(y, a, q, r, x0, p0);
        let (x_s, p_s, p_lag) = crate::lgssm::rts_smooth(a, &x_f, &p_f, &x_pred, &p_pred);
        // M-step
        let mut num = 0.0;
        let mut den = 0.0;
        for t in 1..n {
            num += p_lag[t] + x_s[t] * x_s[t - 1];
            den += p_s[t - 1] + x_s[t - 1] * x_s[t - 1];
        }
        a = if den > 1e-12 { (num / den).clamp(-0.999, 0.999) } else { a };
        let mut q_acc = 0.0;
        for t in 1..n {
            q_acc += p_s[t]
                + x_s[t] * x_s[t]
                + a * a * (p_s[t - 1] + x_s[t - 1] * x_s[t - 1])
                - 2.0 * a * (p_lag[t] + x_s[t] * x_s[t - 1]);
        }
        q = (q_acc / (n - 1) as f64).max(1e-8);
        let mut r_acc = 0.0;
        for t in 0..n {
            r_acc += p_s[t] + (y[t] - x_s[t]).powi(2);
        }
        r = (r_acc / n as f64).max(1e-8);
        x0 = x_s[0];
    }
    (a, q.sqrt(), r.sqrt(), x0)
}

fn unit_id_groups(data: &TabularData, n: usize) -> Option<Vec<u32>> {
    let schema = data.schema();
    for var in schema.variables() {
        if !var.role_hints.contains(RoleHint::UnitId) {
            continue;
        }
        let Ok(col) = data.float64_values(var.id) else {
            continue;
        };
        if col.len() != n {
            continue;
        }
        let mut groups = Vec::with_capacity(n);
        let mut ok = true;
        for &v in col.iter() {
            if !v.is_finite() {
                ok = false;
                break;
            }
            groups.push(v.round() as u32);
        }
        if ok {
            let mut uniq = groups.clone();
            uniq.sort_unstable();
            uniq.dedup();
            if uniq.len() >= 2 && uniq.len() < n {
                return Some(groups);
            }
        }
    }
    None
}

fn demean_by_group(y: &[f64], groups: &[u32]) -> (Vec<f64>, f64) {
    let mut sums = std::collections::HashMap::<u32, (f64, usize)>::new();
    for (&g, &yi) in groups.iter().zip(y.iter()) {
        let e = sums.entry(g).or_insert((0.0, 0));
        e.0 += yi;
        e.1 += 1;
    }
    let grand = y.iter().sum::<f64>() / y.len().max(1) as f64;
    let mut tau2 = 0.0;
    let mut gcount = 0usize;
    for (s, c) in sums.values() {
        let m = s / (*c).max(1) as f64;
        tau2 += (m - grand).powi(2);
        gcount += 1;
    }
    let tau = (tau2 / gcount.max(1) as f64).sqrt();
    let out: Vec<f64> = groups
        .iter()
        .zip(y.iter())
        .map(|(&g, &yi)| {
            let (s, c) = sums[&g];
            yi - s / c.max(1) as f64
        })
        .collect();
    (out, tau)
}

fn fit_ar1(series: &[f64]) -> (f64, f64) {
    let n = series.len();
    if n < 3 {
        return (0.0, series.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-8));
    }
    let mut num = 0.0;
    let mut den = 0.0;
    for t in 1..n {
        num += series[t] * series[t - 1];
        den += series[t - 1] * series[t - 1];
    }
    let a = if den > 1e-12 { (num / den).clamp(-0.999, 0.999) } else { 0.0 };
    let mut rss = 0.0;
    for t in 1..n {
        let e = series[t] - a * series[t - 1];
        rss += e * e;
    }
    let process_std = (rss / (n - 1) as f64).sqrt().max(1e-8);
    (a, process_std)
}

fn fit_linear_gaussian(
    gather: &ParentGatherPlan,
    model: &CompiledCausalModel,
    data: &TabularData,
    y: &[f64],
    backend: FaerBackend,
    ls_ws: &mut LeastSquaresWorkspace,
    ridge: f64,
) -> Result<MechanismSlot, ModelError> {
    let n = y.len();
    let p = gather.n_parents();
    let ncols = 1 + p;
    let mut x = vec![0.0; n * ncols];
    for r in 0..n {
        x[r] = 1.0;
    }
    for (pi, &parent) in gather.parents.iter().enumerate() {
        let var = model.output_layout.variables[parent.as_usize()];
        let col = data.float64_values(var).map_err(ModelError::from)?;
        let base = (1 + pi) * n;
        x[base..base + n].copy_from_slice(&col[..n]);
    }
    if ridge > 0.0 {
        // Augment with ridge rows for coefficients (not intercept).
        let extra = p;
        let mut x2 = vec![0.0; (n + extra) * ncols];
        let mut y2 = vec![0.0; n + extra];
        for c in 0..ncols {
            for r in 0..n {
                x2[c * (n + extra) + r] = x[c * n + r];
            }
        }
        y2[..n].copy_from_slice(y);
        let sqrt_r = ridge.sqrt();
        for j in 0..p {
            x2[(1 + j) * (n + extra) + (n + j)] = sqrt_r;
        }
        let fit =
            backend.least_squares(&x2, n + extra, ncols, &y2, ls_ws).map_err(ModelError::from)?;
        let intercept = fit.coefficients[0];
        let coeffs: Arc<[f64]> = Arc::from(fit.coefficients[1..].to_vec());
        let sigma = (fit.rss / (n.saturating_sub(ncols)).max(1) as f64).sqrt().max(1e-8);
        return Ok(MechanismSlot::LinearGaussian { intercept, coeffs, sigma });
    }
    let fit = backend.least_squares(&x, n, ncols, y, ls_ws).map_err(ModelError::from)?;
    let intercept = fit.coefficients[0];
    let coeffs: Arc<[f64]> = Arc::from(fit.coefficients[1..].to_vec());
    let sigma = (fit.rss / (n.saturating_sub(ncols)).max(1) as f64).sqrt().max(1e-8);
    Ok(MechanismSlot::LinearGaussian { intercept, coeffs, sigma })
}

#[cfg(feature = "gaussian-process")]
fn fit_gaussian_process(
    gather: &ParentGatherPlan,
    model: &CompiledCausalModel,
    data: &TabularData,
    y: &[f64],
) -> Result<MechanismSlot, ModelError> {
    let n = y.len();
    let p = gather.n_parents();
    if p == 0 {
        return Err(ModelError::Unsupported {
            message: "GaussianProcess requires at least one parent".into(),
        });
    }
    let parent_cols = gather_parent_cols(gather, model, data)?;
    let mut x_train = vec![0.0; n * p];
    for r in 0..n {
        for c in 0..p {
            x_train[r * p + c] = parent_cols[c][r];
        }
    }
    // Grid-search length_scale and noise on log marginal likelihood (variance fixed at 1).
    let variance = 1.0;
    let length_scales = [0.25, 0.5, 1.0, 2.0, 4.0];
    let noise_stds = [0.05, 0.1, 0.2, 0.5];
    let mut best = None::<(f64, f64, f64, Vec<f64>)>; // (nlml, ℓ, σ, α)
    for &length_scale in &length_scales {
        for &noise_std in &noise_stds {
            let mut k = vec![0.0; n * n];
            let inv_l2 = 1.0 / (length_scale * length_scale);
            for i in 0..n {
                for j in i..n {
                    let mut d2 = 0.0;
                    for c in 0..p {
                        let d = x_train[i * p + c] - x_train[j * p + c];
                        d2 += d * d;
                    }
                    let kij = variance * (-0.5 * d2 * inv_l2).exp();
                    k[i * n + j] = kij;
                    k[j * n + i] = kij;
                }
                k[i * n + i] += noise_std * noise_std;
            }
            let Ok(alpha) = solve_dense(&k, n, y) else {
                continue;
            };
            // Approximate NLML ∝ y'α + log|K| via diagonal of Cholesky-free proxy: sum log diag after GE.
            let mut y_alpha = 0.0;
            for i in 0..n {
                y_alpha += y[i] * alpha[i];
            }
            let logdet_proxy: f64 = (0..n).map(|i| k[i * n + i].abs().max(1e-12).ln()).sum();
            let nlml = 0.5 * y_alpha + 0.5 * logdet_proxy;
            match &best {
                Some((best_nlml, ..)) if nlml >= *best_nlml => {}
                _ => best = Some((nlml, length_scale, noise_std, alpha)),
            }
        }
    }
    let (_nlml, length_scale, noise_std, alpha) = best.ok_or_else(|| ModelError::Numerical {
        message: "GP hyperparameter search failed".into(),
    })?;
    Ok(MechanismSlot::GaussianProcess {
        length_scale,
        variance,
        noise_std,
        x_train: Arc::from(x_train),
        n_train: n,
        n_parents: p,
        alpha: Arc::from(alpha),
    })
}

#[cfg(feature = "gaussian-process")]
fn solve_dense(a: &[f64], n: usize, b: &[f64]) -> Result<Vec<f64>, ModelError> {
    let mut m = a.to_vec();
    let mut x = b.to_vec();
    for col in 0..n {
        let mut piv = col;
        for r in (col + 1)..n {
            if m[r * n + col].abs() > m[piv * n + col].abs() {
                piv = r;
            }
        }
        if m[piv * n + col].abs() < 1e-12 {
            return Err(ModelError::Numerical { message: "singular GP kernel".into() });
        }
        if piv != col {
            for c in 0..n {
                m.swap(col * n + c, piv * n + c);
            }
            x.swap(col, piv);
        }
        let diag = m[col * n + col];
        for r in (col + 1)..n {
            let f = m[r * n + col] / diag;
            for c in col..n {
                m[r * n + c] -= f * m[col * n + c];
            }
            x[r] -= f * x[col];
        }
    }
    for col in (0..n).rev() {
        let mut acc = x[col];
        for c in (col + 1)..n {
            acc -= m[col * n + c] * x[c];
        }
        x[col] = acc / m[col * n + col];
    }
    Ok(x)
}

// Keep the old LinearGaussian arm body removed — already handled above.

fn residual_mse(
    gather: &ParentGatherPlan,
    model: &CompiledCausalModel,
    data: &TabularData,
    y: &[f64],
    intercept: f64,
    coeffs: &[f64],
) -> Result<f64, ModelError> {
    let n = y.len();
    let mut sse = 0.0;
    let mut parent_cols: Vec<Vec<f64>> = Vec::with_capacity(gather.n_parents());
    for &parent in gather.parents.iter() {
        let var = model.output_layout.variables[parent.as_usize()];
        parent_cols.push(data.float64_values(var).map_err(ModelError::from)?);
    }
    for r in 0..n {
        let mut pred = intercept;
        for (p, col) in parent_cols.iter().enumerate() {
            pred += coeffs[p] * col[r];
        }
        let e = y[r] - pred;
        sse += e * e;
    }
    Ok(sse / n.max(1) as f64)
}

fn discrete_mean_loglik(
    gather: &ParentGatherPlan,
    model: &CompiledCausalModel,
    data: &TabularData,
    y: &[f64],
    support: &[f64],
    logits: &[f64],
) -> Result<f64, ModelError> {
    let n = y.len();
    let p = gather.n_parents();
    let mut parent_mat = vec![0.0; n * p.max(1)];
    for (pi, &parent) in gather.parents.iter().enumerate() {
        let var = model.output_layout.variables[parent.as_usize()];
        let col = data.float64_values(var).map_err(ModelError::from)?;
        let base = pi * n;
        parent_mat[base..base + n].copy_from_slice(&col[..n]);
    }
    let parents = ParentBatch { n_rows: n, n_parents: p, values: &parent_mat[..n * p] };
    let slot = MechanismSlot::Discrete {
        support: Arc::from(support.to_vec()),
        probs: Arc::from(vec![0.0; support.len()]),
        logit_coeffs: Some(Arc::from(logits.to_vec())),
    };
    let mut lp = vec![0.0; n];
    log_prob_column(&slot, y, parents, &mut lp)?;
    Ok(lp.iter().sum::<f64>() / n.max(1) as f64)
}

/// Collection of fitted models weighted by graph posterior mass.
#[derive(Clone, Debug)]
pub struct ModelCollection {
    /// Per-graph compiled models.
    pub models: Arc<[CompiledCausalModel]>,
    /// Graph keys aligned with `models`.
    pub graph_keys: Arc<[u64]>,
    /// Normalized weights (sum to 1 over identified graphs).
    pub weights: Arc<[f64]>,
}

impl ModelCollection {
    /// Build from parallel arrays.
    ///
    /// # Errors
    ///
    /// Length mismatch or non-positive weight sum.
    pub fn new(
        models: impl Into<Arc<[CompiledCausalModel]>>,
        graph_keys: impl Into<Arc<[u64]>>,
        weights: impl Into<Arc<[f64]>>,
    ) -> Result<Self, ModelError> {
        let models = models.into();
        let graph_keys = graph_keys.into();
        let weights = weights.into();
        if models.len() != graph_keys.len() || models.len() != weights.len() {
            return Err(ModelError::Shape { message: "ModelCollection length mismatch".into() });
        }
        let sum: f64 = weights.iter().sum();
        if sum.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
            return Err(ModelError::Shape {
                message: "ModelCollection weights non-positive".into(),
            });
        }
        let weights: Arc<[f64]> = Arc::from(weights.iter().map(|w| w / sum).collect::<Vec<_>>());
        Ok(Self { models, graph_keys, weights })
    }

    /// Number of graphs.
    #[must_use]
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Empty check.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use causal_core::{
        CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType, VariableId,
    };
    use causal_data::column::{Float64Column, ValidityBitmap};
    use causal_data::{OwnedColumn, OwnedColumnarStorage, TabularData};
    use causal_graph::{Dag, DenseNodeId};

    fn toy_data() -> (TabularData, Dag) {
        let n = 40usize;
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "x",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
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
        let mut xv = vec![0.0; n];
        let mut yv = vec![0.0; n];
        for i in 0..n {
            xv[i] = i as f64 * 0.1;
            yv[i] = 1.0 + 2.0 * xv[i];
        }
        let validity = ValidityBitmap::all_valid(n);
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(0), Arc::from(xv), validity.clone())
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(1), Arc::from(yv), validity).unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let mut g = Dag::with_variables(2);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        (TabularData::new(storage), g)
    }

    #[test]
    fn auto_assign_linear_chain() {
        let (data, g) = toy_data();
        let compiled = CompiledCausalModel::compile(g).unwrap();
        let reg = MechanismRegistry::standard();
        let (store, assigns) =
            reg.assign_and_fit(&compiled, &data, SelectionPolicy::BestScore).unwrap();
        assert_eq!(assigns.len(), 2);
        assert!(matches!(
            store.get(DenseNodeId::from_raw(1)),
            MechanismSlot::LinearGaussian { .. }
        ));
    }

    #[test]
    fn bayesian_families_fit_hierarchical_and_bvar() {
        let (data, g) = toy_data();
        let compiled = CompiledCausalModel::compile(g).unwrap();
        let reg = MechanismRegistry::with_bayesian_families();
        let (store, _) = reg
            .assign_and_fit(
                &compiled,
                &data,
                SelectionPolicy::RequireFamily(MechanismFamily::HierarchicalLinear),
            )
            .unwrap();
        assert!(matches!(
            store.get(DenseNodeId::from_raw(1)),
            MechanismSlot::HierarchicalLinear { .. }
        ));
        let (store2, _) = reg
            .assign_and_fit(
                &compiled,
                &data,
                SelectionPolicy::RequireFamily(MechanismFamily::Bvar),
            )
            .unwrap();
        assert!(matches!(store2.get(DenseNodeId::from_raw(1)), MechanismSlot::Bvar { .. }));
        let (store3, _) = reg
            .assign_and_fit(
                &compiled,
                &data,
                SelectionPolicy::RequireFamily(MechanismFamily::LinearGaussianStateSpace),
            )
            .unwrap();
        assert!(matches!(
            store3.get(DenseNodeId::from_raw(1)),
            MechanismSlot::LinearGaussianStateSpace { .. }
        ));
    }

    #[test]
    fn discrete_conditional_multinomial_logit_mle() {
        let n = 120usize;
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "x",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
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
        let mut xv = vec![0.0; n];
        let mut yv = vec![0.0; n];
        for i in 0..n {
            let t = if i < n / 2 { 0.0 } else { 1.0 };
            xv[i] = t;
            // Soft association: mostly Y=t, occasional flips (avoids complete separation).
            yv[i] = if i % 8 == 0 { 1.0 - t } else { t };
        }
        let validity = ValidityBitmap::all_valid(n);
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(0), Arc::from(xv), validity.clone())
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(1), Arc::from(yv), validity).unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let data = TabularData::new(storage);
        let mut g = Dag::with_variables(2);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let compiled = CompiledCausalModel::compile(g).unwrap();
        let reg = MechanismRegistry::standard();
        let (store, _) = reg
            .assign_and_fit(
                &compiled,
                &data,
                SelectionPolicy::RequireFamily(MechanismFamily::Discrete),
            )
            .unwrap();
        let MechanismSlot::Discrete { support, logit_coeffs, .. } =
            store.get(DenseNodeId::from_raw(1))
        else {
            panic!("expected discrete mechanism");
        };
        let logits = logit_coeffs.as_ref().expect("parent-conditional logits");
        assert_eq!(support.len(), 2);
        assert_eq!(logits.len(), 2 * 2); // K * (1 + p)
        // Reference category pinned to zero.
        assert!(logits[0].abs() < 1e-12 && logits[1].abs() < 1e-12);
        // Positive slope for the higher class vs reference.
        assert!(logits[3] > 0.5, "slope={}", logits[3]);
    }
}
