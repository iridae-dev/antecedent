//! Mechanism family fitters.

use std::borrow::Cow;
use std::sync::Arc;

use antecedent_core::RoleHint;
use antecedent_data::{TableView, TabularData};
use antecedent_stats::{FaerBackend, GlmOptions, LeastSquaresWorkspace, MultinomialDesignRef};

use crate::basis::column_moments;

use super::MechanismFamily;
use super::cv::FitFrame;
use crate::compile::{CompiledCausalModel, MechanismSlot, ParentGatherPlan};
use crate::error::ModelError;

mod basis;
mod gp;
mod hierarchical;
mod linear;
mod state_space;

/// Family-specific fitters used by [`fit_family_weighted`].
pub(super) use basis::*;
#[cfg(feature = "gaussian-process")]
pub(super) use gp::{GpGrid, fit_gaussian_process, gp_grid};
use hierarchical::*;
use linear::*;
/// State-space fitters used by [`fit_family_weighted`].
pub(super) use state_space::*;

/// Fit `family` with unit weights.
#[cfg(test)]
pub(super) fn fit_family(
    family: MechanismFamily,
    gather: &ParentGatherPlan,
    model: &CompiledCausalModel,
    data: &TabularData,
    y: &[f64],
    backend: FaerBackend,
    ls_ws: &mut LeastSquaresWorkspace,
) -> Result<MechanismSlot, ModelError> {
    fit_family_weighted(family, FitSource { gather, model, data }, y, backend, ls_ws, None)
}

/// Gather plan, model and data one mechanism fit reads its parents from.
#[derive(Clone, Copy)]
pub(super) struct FitSource<'a> {
    /// Parent column plan for the child being fit.
    pub(super) gather: &'a ParentGatherPlan,
    /// Compiled model the plan was taken from.
    pub(super) model: &'a CompiledCausalModel,
    /// Table the parent columns are read from.
    pub(super) data: &'a TabularData,
}

/// Fit `family`, optionally with per-row weights.
pub(super) fn fit_family_weighted(
    family: MechanismFamily,
    source: FitSource<'_>,
    y: &[f64],
    backend: FaerBackend,
    ls_ws: &mut LeastSquaresWorkspace,
    weights: Option<&[f64]>,
) -> Result<MechanismSlot, ModelError> {
    let FitSource { gather, model, data } = source;
    let frame = FitFrame::from_data(gather, model, data, y.len())?;
    fit_frame_family(family, &frame, y, backend, ls_ws, weights)
}

/// Fit `family` on a frame that is already built.
pub(super) fn fit_frame_family(
    family: MechanismFamily,
    frame: &FitFrame<'_>,
    y: &[f64],
    backend: FaerBackend,
    ls_ws: &mut LeastSquaresWorkspace,
    weights: Option<&[f64]>,
) -> Result<MechanismSlot, ModelError> {
    let n = y.len();
    match family {
        MechanismFamily::Constant => {
            let mean =
                y.iter().enumerate().map(|(i, v)| v * weights.map_or(1.0, |w| w[i])).sum::<f64>()
                    / weights.map_or(n.max(1) as f64, |w| w.iter().sum());
            Ok(MechanismSlot::Constant { value: mean })
        }
        MechanismFamily::Discrete => {
            let (support, probs) = discrete_support(y, weights)?;
            let k = support.len();
            let p = frame.n_parents();
            if p == 0 {
                return Ok(MechanismSlot::Discrete {
                    support: Arc::from(support),
                    probs: Arc::from(probs),
                    logit_coeffs: None,
                });
            }
            // Parent-conditional: baseline-category multinomial logit MLE (Fisher / IRLS).
            // Coefficients are true softmax logits; category 0 is the reference (zeros).
            //
            // The IRLS is run on **standardized** parent columns and the fitted
            // logits are transformed back to the raw scale afterwards. A
            // multinomial logit is equivariant under an affine change of the
            // design, so the fitted conditional law is identical either way —
            // but its *numerics* are not: on columns that differ in scale by
            // orders of magnitude (a raw year column beside a 0–1 axis) the
            // Fisher step is ill-conditioned and the iteration stalls short of
            // the tolerance. Standardizing removes a numerical property of the
            // optimizer from the list of things that decide whether a licensed
            // cell can run at all. The stored coefficients are on the original
            // parent scale, so nothing downstream changes shape.
            let refs: Vec<&[f64]> = frame.parents.iter().map(|c| &c[..n]).collect();
            let (centers, scales) = column_moments(&refs, n);
            let ncols = 1 + p;
            let mut x = vec![0.0; n * ncols];
            x[..n].fill(1.0);
            for (pi, col) in refs.iter().enumerate() {
                let base = (1 + pi) * n;
                for r in 0..n {
                    x[base + r] = (col[r] - centers[pi]) / scales[pi];
                }
            }
            let y_cat = discrete_categories(y, &support)?;
            let fit = antecedent_stats::fit_multinomial_logit_weighted(
                MultinomialDesignRef {
                    x_colmajor: &x,
                    nrows: n,
                    ncols,
                    y_category: &y_cat,
                    n_categories: k,
                },
                weights,
                &backend,
                ls_ws,
                &GlmOptions::default(),
            )?;
            // Refuse non-converged fits; separation is allowed (near-deterministic
            // conditionals → large logits; softmax evaluation remains well-defined).
            if !fit.converged {
                return Err(ModelError::NotConverged {
                    message: format!(
                        "multinomial logit did not converge on standardized parents \
                         (iters={}, deviance={})",
                        fit.iterations, fit.deviance
                    ),
                });
            }
            // Back-transform: η = b₀ + Σ bⱼ (xⱼ − cⱼ)/sⱼ
            //                   = (b₀ − Σ bⱼ cⱼ/sⱼ) + Σ (bⱼ/sⱼ) xⱼ.
            let mut coefficients = fit.coefficients;
            for cat in 0..k {
                let base = cat * ncols;
                let mut shift = 0.0;
                for pi in 0..p {
                    let raw = coefficients[base + 1 + pi] / scales[pi];
                    shift += raw * centers[pi];
                    coefficients[base + 1 + pi] = raw;
                }
                coefficients[base] -= shift;
            }
            Ok(MechanismSlot::Discrete {
                support: Arc::from(support),
                probs: Arc::from(probs),
                logit_coeffs: Some(Arc::from(coefficients)),
            })
        }
        MechanismFamily::LinearGaussian => {
            fit_linear_gaussian_weighted(frame, y, backend, ls_ws, 0.0, weights)
        }
        MechanismFamily::LinearInteractions | MechanismFamily::LinearSpline => {
            let basis = build_basis(family, frame, n)?;
            fit_linear_basis(&basis, frame, y, backend, ls_ws, weights)
        }
        MechanismFamily::DiscreteInteractions | MechanismFamily::DiscreteSpline => {
            let basis = build_basis(family, frame, n)?;
            fit_discrete_basis(&basis, frame, y, backend, ls_ws, weights)
        }
        MechanismFamily::HierarchicalLinear => fit_hierarchical_linear(frame, y, backend, ls_ws),
        MechanismFamily::HierarchicalGlm => fit_hierarchical_glm(frame, y, backend, ls_ws),
        MechanismFamily::Bvar => fit_bvar_minnesota(frame, y, backend, ls_ws),
        MechanismFamily::LinearGaussianStateSpace => fit_lgssm_kalman_em(frame, y, backend, ls_ws),
        MechanismFamily::GaussianProcess => {
            #[cfg(feature = "gaussian-process")]
            {
                fit_gaussian_process(frame, y)
            }
            #[cfg(not(feature = "gaussian-process"))]
            {
                let _ = (frame, y, backend, ls_ws);
                Err(ModelError::Unsupported {
                    message: "GaussianProcess requires feature `gaussian-process`".into(),
                })
            }
        }
    }
}

/// Parent columns of `node`, borrowed from `data`.
pub(super) fn gather_parent_cols<'d>(
    gather: &ParentGatherPlan,
    model: &CompiledCausalModel,
    data: &'d TabularData,
) -> Result<Vec<Cow<'d, [f64]>>, ModelError> {
    let mut parent_cols = Vec::with_capacity(gather.n_parents());
    for &parent in gather.parents.iter() {
        let var = model.output_layout.variables[parent.as_usize()];
        parent_cols.push(data.float64_cow(var).map_err(ModelError::from)?);
    }
    Ok(parent_cols)
}

/// Relative floor of a fitted residual standard deviation: a fraction of the
/// outcome's own RMS, so a deterministic relation measured in small units keeps a
/// σ at the data's scale instead of an absolute `1e-8` that may exceed it.
pub(super) fn sigma_floor(y: &[f64]) -> f64 {
    let rms = (y.iter().map(|v| v * v).sum::<f64>() / y.len().max(1) as f64).sqrt();
    (1e-8 * rms).max(f64::MIN_POSITIVE)
}

/// Residual standard deviation from a data-row residual sum of squares.
pub(super) fn residual_sigma(rss: f64, n: usize, ncols: usize, y: &[f64]) -> f64 {
    (rss / (n.saturating_sub(ncols)).max(1) as f64).sqrt().max(sigma_floor(y))
}

/// Residual sum of squares of `[1 | parents]·(intercept, coeffs)` over the data rows
/// only (no pseudo-observations), weighted when `weights` are given.
pub(super) fn data_rss(
    frame: &FitFrame<'_>,
    y: &[f64],
    intercept: f64,
    coeffs: &[f64],
    weights: Option<&[f64]>,
) -> f64 {
    let mut rss = 0.0;
    for r in 0..y.len() {
        let mut pred = intercept;
        for (c, col) in frame.parents.iter().enumerate() {
            pred += coeffs[c] * col[r];
        }
        rss += weights.map_or(1.0, |w| w[r]) * (y[r] - pred).powi(2);
    }
    rss
}

/// Whether raw `UnitId` labels define a usable grouping: at least two groups, and
/// fewer groups than rows (otherwise every group mean is its own observation).
pub(super) fn group_structure_usable(groups: &[u32]) -> bool {
    let mut uniq = groups.to_vec();
    uniq.sort_unstable();
    uniq.dedup();
    uniq.len() >= 2 && uniq.len() < groups.len()
}

/// Raw `UnitId` labels per row, when a finite `UnitId` column of length `n` exists.
pub(super) fn unit_id_groups(data: &TabularData, n: usize) -> Option<Vec<u32>> {
    let schema = data.schema();
    for var in schema.variables() {
        if !var.role_hints.contains(RoleHint::UnitId) {
            continue;
        }
        let Ok(col) = data.float64_cow(var.id) else {
            continue;
        };
        if col.len() != n {
            continue;
        }
        let mut groups = Vec::with_capacity(n);
        let mut ok = true;
        for &v in col.iter() {
            let id = v.round();
            if !v.is_finite() || id < 0.0 || id > f64::from(u32::MAX) {
                ok = false;
                break;
            }
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "id was just checked to lie in [0, u32::MAX] and is already rounded to an integer"
            )]
            groups.push(id as u32);
        }
        if ok {
            return Some(groups);
        }
    }
    None
}

/// Subtract each unit's mean from its rows.
pub(super) fn demean_by_group(values: &[f64], groups: &[u32]) -> Vec<f64> {
    let mut sums = std::collections::HashMap::<u32, (f64, usize)>::new();
    for (&g, &v) in groups.iter().zip(values.iter()) {
        let e = sums.entry(g).or_insert((0.0, 0));
        e.0 += v;
        e.1 += 1;
    }
    groups
        .iter()
        .zip(values.iter())
        .map(|(&g, &v)| {
            let (s, c) = sums[&g];
            v - s / c.max(1) as f64
        })
        .collect()
}

/// Gaussian AR(1) coefficient and innovation variance.
pub(super) fn fit_ar1(series: &[f64]) -> (f64, f64) {
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

#[cfg(feature = "gaussian-process")]
pub use gp::GP_FAMILY_MAX_ROWS;
