//! Built-in PCM/SCM/invertible mechanism kernels.
//!
//! Hot paths dispatch on [`MechanismSlot`] (enum), not trait objects per scalar.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::needless_range_loop, clippy::too_many_lines)]
#![cfg_attr(
    test,
    allow(
        clippy::float_cmp,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

use antecedent_core::CausalRng;
use antecedent_kernels::{categorical_from_u, standard_normal};

use crate::basis::ParentBasis;
use crate::batch::{MechanismWorkspace, NoiseBatchMut, ParentBatch, ValueBatchMut};
use crate::compile::MechanismSlot;
use crate::error::ModelError;
use crate::lgssm::{kalman_filter, sample_lgssm_noise};

/// How noise was recovered for a mechanism family during abduction.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum NoiseInferenceMode {
    /// Unique structural residual (`y − f(pa)` or equivalent), including the
    /// residual path of a state-space mechanism.
    Invertible,
    /// Sampled posterior noise: the map from noise to value is many-to-one
    /// (categorical CDF bin), so abduction draws from the noise's posterior.
    /// Counterfactuals then rest on the assumed coupling of that noise across
    /// parent settings (rank preservation for a categorical node).
    Posterior,
}

/// The reference ("out-of-coalition") noise value that reconstructs a node at its
/// typical / median output, for point-mass Shapley references
/// (e.g. [`antecedent_attribution`](../antecedent_attribution/index.html)'s
/// ancestor-noise anomaly attribution).
///
/// `0` is the median for every additive-noise family here (`y = f(pa) + ε` with a
/// symmetric, zero-centred `ε`), which is why callers have historically hard-coded
/// `0`. [`MechanismSlot::Discrete`] and [`MechanismSlot::DiscreteBasis`] are the
/// exception: their noise is `u ~ U(0,1)` read through a CDF bin lookup, not an
/// additive residual, so their median is `0.5`, not `0`; feeding them the additive
/// families' `0` reference is an invalid draw ([`categorical_noise`] refuses it).
#[must_use]
pub const fn reference_noise(slot: &MechanismSlot) -> f64 {
    match slot {
        MechanismSlot::Discrete { .. } | MechanismSlot::DiscreteBasis { .. } => 0.5,
        _ => 0.0,
    }
}

/// Validate the uniform noise driving a categorical draw.
///
/// The noise of a categorical mechanism is a `U(0,1)` draw; `NaN`, `0`, `1` or a
/// value outside the interval is not such a draw (for instance an additive
/// residual carried over from another family, or a hand-built posterior). It is
/// refused rather than read as the median category.
fn categorical_noise(noise: f64) -> Result<f64, ModelError> {
    if noise > 0.0 && noise < 1.0 {
        Ok(noise)
    } else {
        Err(ModelError::Numerical {
            message: format!(
                "categorical mechanism noise must lie strictly inside (0, 1), got {noise}"
            ),
        })
    }
}

/// Sample structural noise for a mechanism into `output` (one column).
///
/// # Errors
///
/// Unsupported / vacant slot.
pub fn sample_noise_column(
    slot: &MechanismSlot,
    n_rows: usize,
    rng: &mut CausalRng,
    output: &mut [f64],
) -> Result<(), ModelError> {
    if output.len() < n_rows {
        return Err(ModelError::Shape { message: "noise output too short".into() });
    }
    match slot {
        MechanismSlot::Vacant | MechanismSlot::Pending { .. } => {
            Err(ModelError::Unsupported { message: "mechanism not fitted".into() })
        }
        MechanismSlot::Constant { .. } => {
            output[..n_rows].fill(0.0);
            Ok(())
        }
        MechanismSlot::LinearGaussian { sigma, .. }
        | MechanismSlot::HierarchicalLinear { sigma, .. }
        | MechanismSlot::Bvar { sigma, .. }
        | MechanismSlot::LinearBasis { sigma, .. }
        | MechanismSlot::GaussianProcess { noise_std: sigma, .. } => {
            for i in 0..n_rows {
                output[i] = *sigma * standard_normal(rng);
            }
            Ok(())
        }
        MechanismSlot::LinearGaussianStateSpace { a, process_std, obs_std, initial_mean }
        | MechanismSlot::ConditionalLinearGaussianStateSpace {
            a,
            process_std,
            obs_std,
            initial_mean,
            ..
        } => sample_lgssm_noise(n_rows, *a, *process_std, *obs_std, *initial_mean, rng, output),
        MechanismSlot::Discrete { .. } | MechanismSlot::DiscreteBasis { .. } => {
            // Uniform(0,1) drives categorical draws in evaluate / sample_column.
            for i in 0..n_rows {
                output[i] = rng.next_f64().clamp(f64::EPSILON, 1.0 - f64::EPSILON);
            }
            Ok(())
        }
        MechanismSlot::Dynamic { mechanism, .. } => {
            mechanism.sample_noise_column(n_rows, rng, output)
        }
    }
}

/// Evaluate structural assignment `x = f(parents, noise)` into `output`.
///
/// # Errors
///
/// Shape / vacant.
pub fn evaluate_column(
    slot: &MechanismSlot,
    parents: ParentBatch<'_>,
    noise: &[f64],
    output: &mut [f64],
    ws: &mut MechanismWorkspace,
) -> Result<(), ModelError> {
    let n = parents.n_rows;
    if output.len() < n || noise.len() < n {
        return Err(ModelError::Shape { message: "evaluate buffers too short".into() });
    }
    match slot {
        MechanismSlot::Vacant | MechanismSlot::Pending { .. } => {
            Err(ModelError::Unsupported { message: "mechanism not fitted".into() })
        }
        MechanismSlot::Constant { value } => {
            output[..n].fill(*value);
            Ok(())
        }
        MechanismSlot::LinearGaussian { intercept, coeffs, .. }
        | MechanismSlot::HierarchicalLinear { intercept, coeffs, .. }
        | MechanismSlot::Bvar { intercept, coeffs, .. } => {
            if coeffs.len() != parents.n_parents {
                return Err(ModelError::Shape {
                    message: "linear gaussian coeff length != n_parents".into(),
                });
            }
            for r in 0..n {
                let mut eta = *intercept + noise[r];
                for p in 0..parents.n_parents {
                    eta += coeffs[p] * parents.column(p)?[r];
                }
                output[r] = eta;
            }
            Ok(())
        }
        MechanismSlot::LinearBasis { intercept, basis, coeffs, .. } => {
            basis_mean_column(*intercept, basis, coeffs, parents, &mut output[..n])?;
            for r in 0..n {
                output[r] += noise[r];
            }
            Ok(())
        }
        MechanismSlot::DiscreteBasis { support, basis, logit_coeffs, .. } => {
            if support.is_empty() {
                return Err(ModelError::Shape { message: "empty discrete support".into() });
            }
            let k = support.len();
            let mut row_probs = vec![0.0; k];
            let mut expanded = vec![0.0; basis.n_terms()];
            let mut row = vec![0.0; basis.n_parents()];
            for r in 0..n {
                basis_softmax_row_probs(
                    basis,
                    logit_coeffs,
                    k,
                    parents,
                    r,
                    &mut expanded,
                    &mut row,
                    &mut row_probs,
                )?;
                let u = categorical_noise(noise[r])?;
                output[r] = categorical_draw(support, &row_probs, u);
            }
            Ok(())
        }
        MechanismSlot::ConditionalLinearGaussianStateSpace { intercept, coeffs, .. } => {
            // `noise` is the residual path `r_t`; parents enter through the additive mean.
            output[..n].copy_from_slice(&noise[..n]);
            add_linear_mean(*intercept, coeffs, parents, &mut output[..n], 1.0)
        }
        MechanismSlot::LinearGaussianStateSpace { .. } => {
            output[..n].copy_from_slice(&noise[..n]);
            Ok(())
        }
        MechanismSlot::GaussianProcess {
            length_scale,
            variance,
            mean,
            x_train,
            n_train,
            n_parents,
            alpha,
            ..
        } => {
            if *n_parents != parents.n_parents {
                return Err(ModelError::Shape { message: "GP n_parents mismatch".into() });
            }
            gp_predictive_mean_column(
                *length_scale,
                *variance,
                *mean,
                x_train,
                *n_train,
                *n_parents,
                alpha,
                parents,
                &mut output[..n],
            )?;
            for r in 0..n {
                output[r] += noise[r];
            }
            Ok(())
        }
        MechanismSlot::Discrete { support, probs, logit_coeffs } => {
            if support.is_empty() {
                return Err(ModelError::Shape { message: "empty discrete support".into() });
            }
            match logit_coeffs {
                None => {
                    if support.len() != probs.len() {
                        return Err(ModelError::Shape {
                            message: "discrete support/probs mismatch".into(),
                        });
                    }
                    for r in 0..n {
                        let u = categorical_noise(noise[r])?;
                        output[r] = categorical_draw(support, probs, u);
                    }
                }
                Some(logits) => {
                    let k = support.len();
                    let width = 1 + parents.n_parents;
                    if logits.len() != k * width {
                        return Err(ModelError::Shape {
                            message: "discrete logit_coeffs length mismatch".into(),
                        });
                    }
                    let mut row_probs = vec![0.0; k];
                    for r in 0..n {
                        softmax_row_probs(logits, k, width, parents, r, &mut row_probs)?;
                        let u = categorical_noise(noise[r])?;
                        output[r] = categorical_draw(support, &row_probs, u);
                    }
                }
            }
            Ok(())
        }
        MechanismSlot::Dynamic { mechanism, .. } => {
            mechanism.evaluate_column(parents, noise, output, ws)
        }
    }
}

/// Conditional mean of a [`MechanismSlot::LinearBasis`] at every row.
///
/// Shared by evaluation, abduction, log-density and the registry's scoring so
/// all four read the same surface.
///
/// # Errors
///
/// Coefficient / parent arity mismatch, or out-of-range parent access.
pub(crate) fn basis_mean_column(
    intercept: f64,
    basis: &ParentBasis,
    coeffs: &[f64],
    parents: ParentBatch<'_>,
    output: &mut [f64],
) -> Result<(), ModelError> {
    if basis.n_parents() != parents.n_parents || coeffs.len() != basis.n_terms() {
        return Err(ModelError::Shape {
            message: "basis mechanism arity does not match the parent batch".into(),
        });
    }
    let n = parents.n_rows;
    let mut row = vec![0.0; basis.n_parents()];
    let mut expanded = vec![0.0; basis.n_terms()];
    for r in 0..n {
        for (p, slot) in row.iter_mut().enumerate() {
            *slot = basis.standardize(p, parents.column(p)?[r]);
        }
        basis.expand_standardized(&row, &mut expanded)?;
        let mut eta = intercept;
        for (c, x) in coeffs.iter().zip(&expanded) {
            eta += c * x;
        }
        output[r] = eta;
    }
    Ok(())
}

/// Softmax probabilities of a [`MechanismSlot::DiscreteBasis`] at one row.
#[allow(clippy::too_many_arguments)] // row index plus three reused scratch buffers
fn basis_softmax_row_probs(
    basis: &ParentBasis,
    logits: &[f64],
    k: usize,
    parents: ParentBatch<'_>,
    row_index: usize,
    expanded: &mut [f64],
    row: &mut [f64],
    out: &mut [f64],
) -> Result<(), ModelError> {
    let width = 1 + basis.n_terms();
    if logits.len() != k * width || basis.n_parents() != parents.n_parents {
        return Err(ModelError::Shape {
            message: "discrete basis logit_coeffs length mismatch".into(),
        });
    }
    for (p, slot) in row.iter_mut().enumerate() {
        *slot = basis.standardize(p, parents.column(p)?[row_index]);
    }
    basis.expand_standardized(row, expanded)?;
    for cat in 0..k {
        let base = cat * width;
        let mut pred = logits[base];
        for (t, x) in expanded.iter().enumerate() {
            pred += logits[base + 1 + t] * x;
        }
        out[cat] = pred;
    }
    softmax_in_place(&mut out[..k]);
    Ok(())
}

/// GP dual-form predictive mean at each row:
/// `mean + Σᵢ αᵢ · variance · exp(-0.5 d(x_row, xᵢ)² / ℓ²)`.
///
/// Shared by [`evaluate_column`], [`infer_noise_column_rng`], `log_prob_column`, and the
/// registry's residual-MSE scoring so all four use an identical dual-form prediction.
///
/// # Errors
///
/// Out-of-range parent column access.
#[allow(clippy::too_many_arguments)] // GP dual-form params, threaded through from the fitted slot.
pub(crate) fn gp_predictive_mean_column(
    length_scale: f64,
    variance: f64,
    prior_mean: f64,
    x_train: &[f64],
    n_train: usize,
    n_parents: usize,
    alpha: &[f64],
    parents: ParentBatch<'_>,
    output: &mut [f64],
) -> Result<(), ModelError> {
    let n = parents.n_rows;
    let inv_l2 = 1.0 / (length_scale * length_scale);
    for r in 0..n {
        let mut mean = prior_mean;
        for i in 0..n_train {
            let mut d2 = 0.0;
            for p in 0..parents.n_parents {
                let d = parents.column(p)?[r] - x_train[i * n_parents + p];
                d2 += d * d;
            }
            mean += alpha[i] * variance * (-0.5 * d2 * inv_l2).exp();
        }
        output[r] = mean;
    }
    Ok(())
}

/// Infer exogenous noise from observed value and parents (invertible path).
///
/// Families whose abduction is a posterior draw (categorical mechanisms) need an
/// RNG and refuse here — use [`infer_noise_column_rng`] for those.
///
/// # Errors
///
/// Non-invertible family ([`ModelError::Unsupported`]) or shape.
pub fn infer_noise_column(
    slot: &MechanismSlot,
    value: &[f64],
    parents: ParentBatch<'_>,
    output: &mut [f64],
) -> Result<(), ModelError> {
    // A fixed seed can only be reached by a family that never consults it; a
    // posterior family is refused below before any draw is used.
    let mut unused = CausalRng::from_seed(0);
    let mode = infer_noise_column_rng(slot, value, parents, output, &mut unused)?;
    if mode == NoiseInferenceMode::Posterior {
        return Err(ModelError::Unsupported {
            message: "noise inference for this mechanism is a posterior draw, not an inversion; \
                      call infer_noise_column_rng with an explicit RNG stream"
                .into(),
        });
    }
    Ok(())
}

/// Infer exogenous noise, sampling posterior noise when the map is many-to-one.
///
/// Returns whether the draw was invertible or posterior.
///
/// # Errors
///
/// Unsupported family or shape.
pub fn infer_noise_column_rng(
    slot: &MechanismSlot,
    value: &[f64],
    parents: ParentBatch<'_>,
    output: &mut [f64],
    rng: &mut CausalRng,
) -> Result<NoiseInferenceMode, ModelError> {
    let n = parents.n_rows;
    if value.len() < n || output.len() < n {
        return Err(ModelError::Shape { message: "infer_noise buffers too short".into() });
    }
    match slot {
        MechanismSlot::LinearGaussian { intercept, coeffs, .. }
        | MechanismSlot::HierarchicalLinear { intercept, coeffs, .. }
        | MechanismSlot::Bvar { intercept, coeffs, .. } => {
            if coeffs.len() != parents.n_parents {
                return Err(ModelError::Shape {
                    message: "linear gaussian coeff length != n_parents".into(),
                });
            }
            for r in 0..n {
                let mut eta = *intercept;
                for p in 0..parents.n_parents {
                    eta += coeffs[p] * parents.column(p)?[r];
                }
                output[r] = value[r] - eta;
            }
            Ok(NoiseInferenceMode::Invertible)
        }
        MechanismSlot::Constant { value: c } => {
            // `evaluate_column` ignores the noise of a constant, so a replay reproduces `c`
            // whatever was abduced. Abduction is therefore only faithful on rows that equal
            // the constant; any other row is refused instead of reported as "invertible".
            let tol = 1e-9 * c.abs().max(1.0);
            for r in 0..n {
                let deviation = (value[r] - *c).abs();
                if deviation.is_nan() || deviation > tol {
                    return Err(ModelError::Numerical {
                        message: format!(
                            "constant mechanism ({c}) cannot reproduce the observed value {} at \
                             row {r}; the observation is not a draw from a deterministic node",
                            value[r]
                        ),
                    });
                }
                output[r] = value[r] - *c;
            }
            Ok(NoiseInferenceMode::Invertible)
        }
        MechanismSlot::GaussianProcess {
            length_scale,
            variance,
            mean: prior_mean,
            x_train,
            n_train,
            n_parents,
            alpha,
            ..
        } => {
            if *n_parents != parents.n_parents {
                return Err(ModelError::Shape { message: "GP n_parents mismatch".into() });
            }
            let mut mean = vec![0.0; n];
            gp_predictive_mean_column(
                *length_scale,
                *variance,
                *prior_mean,
                x_train,
                *n_train,
                *n_parents,
                alpha,
                parents,
                &mut mean,
            )?;
            for r in 0..n {
                output[r] = value[r] - mean[r];
            }
            Ok(NoiseInferenceMode::Invertible)
        }
        MechanismSlot::LinearBasis { intercept, basis, coeffs, .. } => {
            // Additive disturbance: the expansion moves the conditional mean
            // only, so the structural residual is recovered exactly.
            let mut mean = vec![0.0; n];
            basis_mean_column(*intercept, basis, coeffs, parents, &mut mean)?;
            for r in 0..n {
                output[r] = value[r] - mean[r];
            }
            Ok(NoiseInferenceMode::Invertible)
        }
        MechanismSlot::Discrete { support, probs, logit_coeffs } => {
            infer_discrete_posterior_noise(
                support,
                probs,
                logit_coeffs.as_deref(),
                value,
                parents,
                output,
                rng,
            )?;
            Ok(NoiseInferenceMode::Posterior)
        }
        MechanismSlot::DiscreteBasis { support, basis, logit_coeffs, .. } => {
            let k = support.len();
            if k == 0 {
                return Err(ModelError::Shape { message: "empty discrete support".into() });
            }
            let mut row_probs = vec![0.0; k];
            let mut expanded = vec![0.0; basis.n_terms()];
            let mut row = vec![0.0; basis.n_parents()];
            for r in 0..n {
                basis_softmax_row_probs(
                    basis,
                    logit_coeffs,
                    k,
                    parents,
                    r,
                    &mut expanded,
                    &mut row,
                    &mut row_probs,
                )?;
                output[r] = categorical_inverse_cdf_draw(support, &row_probs, value[r], rng)?;
            }
            Ok(NoiseInferenceMode::Posterior)
        }
        MechanismSlot::ConditionalLinearGaussianStateSpace { intercept, coeffs, .. } => {
            // The noise is the residual path itself: exact, lossless inversion.
            output[..n].copy_from_slice(&value[..n]);
            add_linear_mean(*intercept, coeffs, parents, &mut output[..n], -1.0)?;
            Ok(NoiseInferenceMode::Invertible)
        }
        MechanismSlot::LinearGaussianStateSpace { .. } => {
            output[..n].copy_from_slice(&value[..n]);
            Ok(NoiseInferenceMode::Invertible)
        }
        MechanismSlot::Dynamic { mechanism, .. } => {
            mechanism.infer_noise_column(value, parents, output)?;
            Ok(mechanism.noise_inference_mode())
        }
        _ => Err(ModelError::Unsupported {
            message: "noise inference unsupported for this mechanism family".into(),
        }),
    }
}

fn infer_discrete_posterior_noise(
    support: &[f64],
    probs: &[f64],
    logit_coeffs: Option<&[f64]>,
    value: &[f64],
    parents: ParentBatch<'_>,
    output: &mut [f64],
    rng: &mut CausalRng,
) -> Result<(), ModelError> {
    let n = parents.n_rows;
    if support.is_empty() {
        return Err(ModelError::Shape { message: "empty discrete support".into() });
    }
    let k = support.len();
    let mut row_probs = vec![0.0; k];
    for r in 0..n {
        match logit_coeffs {
            None => {
                if support.len() != probs.len() {
                    return Err(ModelError::Shape {
                        message: "discrete support/probs mismatch".into(),
                    });
                }
                row_probs.copy_from_slice(probs);
                let sum: f64 = row_probs.iter().sum::<f64>().max(f64::EPSILON);
                for p in &mut row_probs {
                    *p /= sum;
                }
            }
            Some(logits) => {
                let width = 1 + parents.n_parents;
                if logits.len() != k * width {
                    return Err(ModelError::Shape {
                        message: "discrete logit_coeffs length mismatch".into(),
                    });
                }
                softmax_row_probs(logits, k, width, parents, r, &mut row_probs)?;
            }
        }
        output[r] = categorical_inverse_cdf_draw(support, &row_probs, value[r], rng)?;
    }
    Ok(())
}

/// Draw the uniform noise of a categorical mechanism that produced `value`.
///
/// The map from `u ~ U(0,1)` to a category is many-to-one, so abduction draws
/// uniformly from the CDF bin of the observed category — the posterior of the
/// noise given the value. One owner, shared by the parent-conditional
/// [`MechanismSlot::Discrete`] and [`MechanismSlot::DiscreteBasis`] paths.
fn categorical_inverse_cdf_draw(
    support: &[f64],
    row_probs: &[f64],
    value: f64,
    rng: &mut CausalRng,
) -> Result<f64, ModelError> {
    let cat = support.iter().position(|&s| (value - s).abs() < 1e-12).ok_or_else(|| {
        ModelError::Unsupported { message: format!("discrete value {value} not in support") }
    })?;
    let mut lo = 0.0;
    for i in 0..cat {
        lo += row_probs[i];
    }
    let hi = lo + row_probs[cat];
    // The draw must land inside the observed category's own bin so that replaying it
    // returns the observed value. Widening a vanishing bin (as an epsilon floor does)
    // moves the draw into the neighbouring category and silently breaks factual
    // consistency, so a bin narrower than the resolution of an `f64` uniform is refused.
    let mut u = lo + (hi - lo) * rng.next_f64();
    if categorical_from_u(u, row_probs) != Some(cat) {
        u = 0.5 * (lo + hi);
    }
    if !(u > 0.0 && u < 1.0) || categorical_from_u(u, row_probs) != Some(cat) {
        return Err(ModelError::Numerical {
            message: format!(
                "observed category {value} has model probability {:e}, below the resolution of \
                 a uniform draw; its noise cannot be abduced faithfully",
                row_probs[cat]
            ),
        });
    }
    Ok(u)
}

/// Log-density of observed values under the mechanism (PCM path).
///
/// # Errors
///
/// Shape / vacant.
pub fn log_prob_column(
    slot: &MechanismSlot,
    values: &[f64],
    parents: ParentBatch<'_>,
    output: &mut [f64],
) -> Result<(), ModelError> {
    let n = parents.n_rows;
    if values.len() < n || output.len() < n {
        return Err(ModelError::Shape { message: "log_prob buffers too short".into() });
    }
    match slot {
        MechanismSlot::LinearGaussian { intercept, coeffs, sigma }
        | MechanismSlot::HierarchicalLinear { intercept, coeffs, sigma, .. }
        | MechanismSlot::Bvar { intercept, coeffs, sigma } => {
            if !(sigma.is_finite() && *sigma > 0.0) {
                return Err(ModelError::Numerical { message: "sigma must be > 0".into() });
            }
            if coeffs.len() != parents.n_parents {
                return Err(ModelError::Shape {
                    message: "linear gaussian coeff length != n_parents".into(),
                });
            }
            let inv_s = 1.0 / sigma;
            let log_norm = -0.5 * (2.0 * std::f64::consts::PI).ln() - sigma.ln();
            for r in 0..n {
                let mut eta = *intercept;
                for p in 0..parents.n_parents {
                    eta += coeffs[p] * parents.column(p)?[r];
                }
                let z = (values[r] - eta) * inv_s;
                output[r] = log_norm - 0.5 * z * z;
            }
            Ok(())
        }
        MechanismSlot::Constant { value } => {
            for r in 0..n {
                output[r] =
                    if (values[r] - *value).abs() < 1e-12 { 0.0 } else { f64::NEG_INFINITY };
            }
            Ok(())
        }
        MechanismSlot::Discrete { support, probs, logit_coeffs } => {
            for r in 0..n {
                let lp = match logit_coeffs {
                    None => {
                        let sum: f64 = probs.iter().sum();
                        let mut found = f64::NEG_INFINITY;
                        for (i, &s) in support.iter().enumerate() {
                            if (values[r] - s).abs() < 1e-12 {
                                found = (probs[i] / sum.max(f64::EPSILON)).ln();
                                break;
                            }
                        }
                        found
                    }
                    Some(logits) => {
                        let k = support.len();
                        let width = 1 + parents.n_parents;
                        if logits.len() != k * width {
                            return Err(ModelError::Shape {
                                message: "discrete logit_coeffs length mismatch".into(),
                            });
                        }
                        let mut row_probs = vec![0.0; k];
                        softmax_row_probs(logits, k, width, parents, r, &mut row_probs)?;
                        let mut found = f64::NEG_INFINITY;
                        for (i, &s) in support.iter().enumerate() {
                            if (values[r] - s).abs() < 1e-12 {
                                found = row_probs[i].max(f64::EPSILON).ln();
                                break;
                            }
                        }
                        found
                    }
                };
                output[r] = lp;
            }
            Ok(())
        }
        MechanismSlot::LinearBasis { intercept, basis, coeffs, sigma } => {
            if !(sigma.is_finite() && *sigma > 0.0) {
                return Err(ModelError::Numerical { message: "sigma must be > 0".into() });
            }
            let mut mean = vec![0.0; n];
            basis_mean_column(*intercept, basis, coeffs, parents, &mut mean)?;
            let inv_s = 1.0 / sigma;
            let log_norm = -0.5 * (2.0 * std::f64::consts::PI).ln() - sigma.ln();
            for r in 0..n {
                let z = (values[r] - mean[r]) * inv_s;
                output[r] = log_norm - 0.5 * z * z;
            }
            Ok(())
        }
        MechanismSlot::DiscreteBasis { support, basis, logit_coeffs, .. } => {
            let k = support.len();
            let mut row_probs = vec![0.0; k];
            let mut expanded = vec![0.0; basis.n_terms()];
            let mut row = vec![0.0; basis.n_parents()];
            for r in 0..n {
                basis_softmax_row_probs(
                    basis,
                    logit_coeffs,
                    k,
                    parents,
                    r,
                    &mut expanded,
                    &mut row,
                    &mut row_probs,
                )?;
                let mut found = f64::NEG_INFINITY;
                for (i, &s) in support.iter().enumerate() {
                    if (values[r] - s).abs() < 1e-12 {
                        found = row_probs[i].max(f64::EPSILON).ln();
                        break;
                    }
                }
                output[r] = found;
            }
            Ok(())
        }
        MechanismSlot::Vacant | MechanismSlot::Pending { .. } => {
            Err(ModelError::Unsupported { message: "mechanism not fitted".into() })
        }
        MechanismSlot::ConditionalLinearGaussianStateSpace {
            intercept,
            coeffs,
            a,
            process_std,
            obs_std,
            initial_mean,
        } => {
            let mut residual_values = values[..n].to_vec();
            add_linear_mean(*intercept, coeffs, parents, &mut residual_values, -1.0)?;
            let residual = MechanismSlot::LinearGaussianStateSpace {
                a: *a,
                process_std: *process_std,
                obs_std: *obs_std,
                initial_mean: *initial_mean,
            };
            log_prob_column(&residual, &residual_values, parents, output)
        }
        MechanismSlot::LinearGaussianStateSpace { a, process_std, obs_std, initial_mean } => {
            let scaled = crate::lgssm::scaled_lgssm(
                &values[..n],
                *a,
                *process_std,
                *obs_std,
                *initial_mean,
            )?;
            let (_, _, x_pred, p_pred) = kalman_filter(
                &scaled.values,
                *a,
                scaled.process_var,
                scaled.obs_var,
                scaled.initial_mean,
                scaled.process_var,
            );
            for t in 0..n {
                let var = p_pred[t] + scaled.obs_var;
                let z = scaled.values[t] - x_pred[t];
                output[t] = -0.5 * ((2.0 * std::f64::consts::PI).ln() + var.ln())
                    - scaled.scale.ln()
                    - 0.5 * z * z / var;
            }
            Ok(())
        }
        MechanismSlot::GaussianProcess {
            length_scale,
            variance,
            noise_std,
            mean: prior_mean,
            x_train,
            n_train,
            n_parents,
            alpha,
        } => {
            if !(noise_std.is_finite() && *noise_std > 0.0) {
                return Err(ModelError::Numerical { message: "noise_std must be > 0".into() });
            }
            if *n_parents != parents.n_parents {
                return Err(ModelError::Shape { message: "GP n_parents mismatch".into() });
            }
            // Score against the GP predictive mean, not the raw observed value.
            let mut mean = vec![0.0; n];
            gp_predictive_mean_column(
                *length_scale,
                *variance,
                *prior_mean,
                x_train,
                *n_train,
                *n_parents,
                alpha,
                parents,
                &mut mean,
            )?;
            let inv_s = 1.0 / noise_std;
            let log_norm = -0.5 * (2.0 * std::f64::consts::PI).ln() - noise_std.ln();
            for r in 0..n {
                let z = (values[r] - mean[r]) * inv_s;
                output[r] = log_norm - 0.5 * z * z;
            }
            Ok(())
        }
        MechanismSlot::Dynamic { mechanism, .. } => {
            mechanism.log_prob_column(values, parents, output)
        }
    }
}

/// Sample a full column from parents (PCM sample path combining noise+evaluate).
///
/// # Errors
///
/// Mechanism errors.
pub fn sample_column(
    slot: &MechanismSlot,
    parents: ParentBatch<'_>,
    rng: &mut CausalRng,
    output: &mut [f64],
    ws: &mut MechanismWorkspace,
) -> Result<(), ModelError> {
    let n = parents.n_rows;
    ws.prepare(n, parents.n_parents.max(1));
    if let MechanismSlot::Discrete { support, probs, logit_coeffs } = slot {
        for r in 0..n {
            let u = rng.next_f64().max(f64::EPSILON);
            match logit_coeffs {
                None => {
                    output[r] = categorical_draw(support, probs, u);
                }
                Some(logits) => {
                    let k = support.len();
                    let width = 1 + parents.n_parents;
                    if logits.len() != k * width {
                        return Err(ModelError::Shape {
                            message: "discrete logit_coeffs length mismatch".into(),
                        });
                    }
                    let mut row_probs = vec![0.0; k];
                    softmax_row_probs(logits, k, width, parents, r, &mut row_probs)?;
                    output[r] = categorical_draw(support, &row_probs, u);
                }
            }
        }
        Ok(())
    } else {
        let mut noise = vec![0.0; n];
        sample_noise_column(slot, n, rng, &mut noise)?;
        evaluate_column(slot, parents, &noise, output, ws)
    }
}

fn softmax_row_probs(
    logits: &[f64],
    k: usize,
    width: usize,
    parents: ParentBatch<'_>,
    row: usize,
    out: &mut [f64],
) -> Result<(), ModelError> {
    // True multinomial-logit coefficients from Fisher/IRLS (`fit_multinomial_logit`).
    for cat in 0..k {
        let base = cat * width;
        let mut pred = logits[base];
        for p in 0..parents.n_parents {
            pred += logits[base + 1 + p] * parents.column(p)?[row];
        }
        out[cat] = pred;
    }
    softmax_in_place(&mut out[..k]);
    Ok(())
}

/// Max-shifted softmax of the logits in `values`, written back in place. The one
/// owner of the normalization every categorical mechanism uses.
pub(crate) fn softmax_in_place(values: &mut [f64]) {
    let max_eta = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0;
    for v in values.iter_mut() {
        *v = (*v - max_eta).exp();
        sum += *v;
    }
    let inv = 1.0 / sum.max(f64::EPSILON);
    for v in values.iter_mut() {
        *v *= inv;
    }
}

fn categorical_draw(support: &[f64], probs: &[f64], u: f64) -> f64 {
    // No probability mass has no category: NaN, not an arbitrary support point.
    categorical_from_u(u, probs).map_or(f64::NAN, |idx| support.get(idx).copied().unwrap_or(0.0))
}

/// Fill an entire noise batch for all nodes (structural path).
///
/// # Errors
///
/// Mechanism errors.
pub fn sample_noise_batch(
    slots: &[MechanismSlot],
    n_rows: usize,
    rng: &mut CausalRng,
    noise: &mut NoiseBatchMut<'_>,
) -> Result<(), ModelError> {
    for (node, slot) in slots.iter().enumerate() {
        let col = noise.column_mut(node)?;
        sample_noise_column(slot, n_rows, rng, col)?;
    }
    Ok(())
}

/// Evaluate all nodes in topological order given parents already in `values`
/// for upstream nodes. Writes into `values` columns for each node in `order`.
///
/// # Errors
///
/// Mechanism / shape.
pub fn evaluate_batch_topo(
    order: &[antecedent_graph::DenseNodeId],
    gathers: &[crate::compile::ParentGatherPlan],
    slots: &[MechanismSlot],
    noise: &NoiseBatchMut<'_>,
    values: &mut ValueBatchMut<'_>,
    ws: &mut MechanismWorkspace,
) -> Result<(), ModelError> {
    let n_rows = values.n_rows;
    for (gi, &node) in order.iter().enumerate() {
        let gather = &gathers[gi];
        debug_assert_eq!(gather.child, node);
        ws.prepare(n_rows, gather.n_parents().max(1));
        gather.gather(values.values, n_rows, &mut ws.parents);
        let parent_owned = ws.parents[..gather.n_parents().saturating_mul(n_rows)].to_vec();
        let parents = ParentBatch { n_rows, n_parents: gather.n_parents(), values: &parent_owned };
        let noise_slice = noise.column(node.as_usize())?;
        let noise_owned = noise_slice.to_vec();
        let out = values.column_mut(node.as_usize())?;
        evaluate_column(&slots[node.as_usize()], parents, &noise_owned, out, ws)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::MechanismSlot;
    use antecedent_core::CausalRng;
    use std::sync::Arc;

    #[test]
    fn linear_gaussian_round_trip_noise() {
        let slot = MechanismSlot::LinearGaussian {
            intercept: 1.0,
            coeffs: Arc::from(vec![2.0]),
            sigma: 1.0,
        };
        let parent_vals = [0.5_f64, 1.0];
        let parents = ParentBatch { n_rows: 2, n_parents: 1, values: &parent_vals };
        let noise = [0.1, -0.2];
        let mut out = [0.0; 2];
        let mut ws = MechanismWorkspace::default();
        evaluate_column(&slot, parents, &noise, &mut out, &mut ws).unwrap();
        assert!((out[0] - (1.0 + 2.0 * 0.5 + 0.1)).abs() < 1e-12);
        let mut inferred = [0.0; 2];
        infer_noise_column(&slot, &out, parents, &mut inferred).unwrap();
        assert!((inferred[0] - 0.1).abs() < 1e-12);
        assert!((inferred[1] - (-0.2)).abs() < 1e-12);
    }

    #[test]
    fn discrete_posterior_noise_recovers_category() {
        let slot = MechanismSlot::Discrete {
            support: Arc::from(vec![0.0, 1.0, 2.0]),
            probs: Arc::from(vec![0.2, 0.5, 0.3]),
            logit_coeffs: None,
        };
        let parents = ParentBatch { n_rows: 3, n_parents: 0, values: &[] };
        let value = [1.0, 0.0, 2.0];
        let mut noise = [0.0; 3];
        let mut rng = CausalRng::from_seed(42);
        let mode = infer_noise_column_rng(&slot, &value, parents, &mut noise, &mut rng).unwrap();
        assert_eq!(mode, NoiseInferenceMode::Posterior);
        let mut out = [0.0; 3];
        let mut ws = MechanismWorkspace::default();
        evaluate_column(&slot, parents, &noise, &mut out, &mut ws).unwrap();
        assert_eq!(out, value);
    }

    #[test]
    fn gp_invertible_noise_round_trip() {
        let slot = MechanismSlot::GaussianProcess {
            length_scale: 1.0,
            variance: 1.0,
            noise_std: 0.1,
            mean: 0.0,
            x_train: Arc::from(vec![0.0, 1.0]),
            n_train: 2,
            n_parents: 1,
            alpha: Arc::from(vec![0.5, -0.25]),
        };
        let parent_vals = [0.0_f64, 1.0];
        let parents = ParentBatch { n_rows: 2, n_parents: 1, values: &parent_vals };
        let noise = [0.05, -0.02];
        let mut out = [0.0; 2];
        let mut ws = MechanismWorkspace::default();
        evaluate_column(&slot, parents, &noise, &mut out, &mut ws).unwrap();
        let mut inferred = [0.0; 2];
        let mode = infer_noise_column_rng(
            &slot,
            &out,
            parents,
            &mut inferred,
            &mut CausalRng::from_seed(1),
        )
        .unwrap();
        assert_eq!(mode, NoiseInferenceMode::Invertible);
        assert!((inferred[0] - noise[0]).abs() < 1e-10);
        assert!((inferred[1] - noise[1]).abs() < 1e-10);
    }

    /// `Σ αᵢ k(x, xᵢ) + mean` with one training point: at `x = x₁` the kernel is
    /// `variance`, so the prediction is `mean + α·variance` exactly, and far away it
    /// reverts to `mean`, not to zero.
    #[test]
    fn gp_prediction_reverts_to_the_prior_mean_off_support() {
        let slot = MechanismSlot::GaussianProcess {
            length_scale: 1.0,
            variance: 4.0,
            noise_std: 0.1,
            mean: 1000.0,
            x_train: Arc::from(vec![0.0]),
            n_train: 1,
            n_parents: 1,
            alpha: Arc::from(vec![0.5]),
        };
        let parent_vals = [0.0_f64, 50.0];
        let parents = ParentBatch { n_rows: 2, n_parents: 1, values: &parent_vals };
        let mut out = [0.0; 2];
        evaluate_column(&slot, parents, &[0.0, 0.0], &mut out, &mut MechanismWorkspace::default())
            .unwrap();
        assert!((out[0] - (1000.0 + 0.5 * 4.0)).abs() < 1e-12);
        assert!((out[1] - 1000.0).abs() < 1e-12);
    }

    /// Noise outside (0, 1) is not a categorical draw; it must be refused, not read as
    /// the median category.
    #[test]
    fn categorical_evaluation_refuses_invalid_noise() {
        let slot = MechanismSlot::Discrete {
            support: Arc::from(vec![0.0, 1.0, 2.0]),
            probs: Arc::from(vec![0.2, 0.5, 0.3]),
            logit_coeffs: None,
        };
        let parents = ParentBatch { n_rows: 1, n_parents: 0, values: &[] };
        let mut ws = MechanismWorkspace::default();
        for bad in [f64::NAN, 0.0, 1.0, -0.3, 1.7] {
            let mut out = [0.0];
            let err = evaluate_column(&slot, parents, &[bad], &mut out, &mut ws).unwrap_err();
            assert!(matches!(err, ModelError::Numerical { .. }), "noise {bad}: {err:?}");
        }
        let mut out = [0.0];
        evaluate_column(&slot, parents, &[0.6], &mut out, &mut ws).unwrap();
        assert_eq!(out, [1.0]); // cdf = 0.2, 0.7, 1.0: u = 0.6 lies in the middle bin
    }

    /// A posterior family has no deterministic inversion: the RNG-free entry point
    /// must say so instead of returning seed-0 draws.
    #[test]
    fn rng_free_noise_inference_refuses_posterior_families() {
        let slot = MechanismSlot::Discrete {
            support: Arc::from(vec![0.0, 1.0]),
            probs: Arc::from(vec![0.5, 0.5]),
            logit_coeffs: None,
        };
        let parents = ParentBatch { n_rows: 2, n_parents: 0, values: &[] };
        let mut noise = [0.0; 2];
        let err = infer_noise_column(&slot, &[0.0, 1.0], parents, &mut noise).unwrap_err();
        assert!(matches!(err, ModelError::Unsupported { .. }), "{err:?}");
    }

    /// A constant's replay ignores noise, so abducing a row that is not the constant
    /// cannot reproduce it and must not be reported as an inversion.
    #[test]
    fn constant_abduction_refuses_rows_that_are_not_the_constant() {
        let slot = MechanismSlot::Constant { value: 3.0 };
        let parents = ParentBatch { n_rows: 2, n_parents: 0, values: &[] };
        let mut noise = [0.0; 2];
        infer_noise_column(&slot, &[3.0, 3.0], parents, &mut noise).unwrap();
        assert_eq!(noise, [0.0, 0.0]);
        let err = infer_noise_column(&slot, &[3.0, 3.5], parents, &mut noise).unwrap_err();
        assert!(matches!(err, ModelError::Numerical { .. }), "{err:?}");
    }

    /// Category probabilities of 1e-20 sit below the resolution of an `f64` uniform: the
    /// abduced draw must either replay the observed category or be refused — never the
    /// neighbouring category.
    #[test]
    fn vanishing_category_bin_is_abduced_faithfully_or_refused() {
        let slot = MechanismSlot::Discrete {
            support: Arc::from(vec![0.0, 1.0, 2.0]),
            probs: Arc::from(vec![0.5, 1e-20, 0.5 - 1e-20]),
            logit_coeffs: None,
        };
        let parents = ParentBatch { n_rows: 1, n_parents: 0, values: &[] };
        let mut ws = MechanismWorkspace::default();
        let mut rng = CausalRng::from_seed(3);
        for observed in [0.0, 1.0, 2.0] {
            let mut noise = [0.0];
            match infer_noise_column_rng(&slot, &[observed], parents, &mut noise, &mut rng) {
                Ok(_) => {
                    let mut out = [f64::NAN];
                    evaluate_column(&slot, parents, &noise, &mut out, &mut ws).unwrap();
                    assert_eq!(out[0], observed);
                }
                Err(e) => assert!(matches!(e, ModelError::Numerical { .. }), "{e:?}"),
            }
        }
    }

    #[test]
    fn dynamic_mechanism_dispatch() {
        struct ConstMech(f64);
        impl crate::compile::DynamicMechanism for ConstMech {
            fn sample_noise_column(
                &self,
                n_rows: usize,
                _rng: &mut antecedent_core::CausalRng,
                output: &mut [f64],
            ) -> Result<(), ModelError> {
                output[..n_rows].fill(0.0);
                Ok(())
            }
            fn evaluate_column(
                &self,
                parents: ParentBatch<'_>,
                _noise: &[f64],
                output: &mut [f64],
                _ws: &mut MechanismWorkspace,
            ) -> Result<(), ModelError> {
                output[..parents.n_rows].fill(self.0);
                Ok(())
            }
        }
        let slot =
            MechanismSlot::Dynamic { id: Arc::from("y"), mechanism: Arc::new(ConstMech(7.0)) };
        let parents = ParentBatch { n_rows: 3, n_parents: 0, values: &[] };
        let noise = [0.0; 3];
        let mut out = [0.0; 3];
        let mut ws = MechanismWorkspace::default();
        evaluate_column(&slot, parents, &noise, &mut out, &mut ws).unwrap();
        assert_eq!(out, [7.0, 7.0, 7.0]);
    }
}

/// Add or subtract the conditional mean without changing the latent noise model.
fn add_linear_mean(
    intercept: f64,
    coeffs: &[f64],
    parents: ParentBatch<'_>,
    values: &mut [f64],
    sign: f64,
) -> Result<(), ModelError> {
    if coeffs.len() != parents.n_parents {
        return Err(ModelError::Shape {
            message: "conditional LGSSM coefficient count differs from parent count".into(),
        });
    }
    for value in values.iter_mut() {
        *value += sign * intercept;
    }
    for (p, &coefficient) in coeffs.iter().enumerate() {
        for (value, &parent) in values.iter_mut().zip(parents.column(p)?) {
            *value += sign * coefficient * parent;
        }
    }
    Ok(())
}

#[cfg(test)]
mod conditional_lgssm_tests {
    use super::*;

    #[test]
    fn conditional_lgssm_density_abduction_and_counterfactual_agree() {
        let conditional = MechanismSlot::ConditionalLinearGaussianStateSpace {
            intercept: 4.0,
            coeffs: std::sync::Arc::from([2.0]),
            a: 0.7,
            process_std: 0.3,
            obs_std: 0.2,
            initial_mean: 0.1,
        };
        let plain = MechanismSlot::LinearGaussianStateSpace {
            a: 0.7,
            process_std: 0.3,
            obs_std: 0.2,
            initial_mean: 0.1,
        };
        let parents = [0.0, 1.0, 2.0, 3.0];
        let residual = [0.4, -0.3, 0.7, 0.2];
        let values: Vec<f64> =
            parents.iter().zip(residual).map(|(p, e)| 4.0 + 2.0 * p + e).collect();
        let batch = ParentBatch { n_rows: 4, n_parents: 1, values: &parents };
        let mut conditional_lp = [0.0; 4];
        let mut plain_lp = [0.0; 4];
        log_prob_column(&conditional, &values, batch, &mut conditional_lp).unwrap();
        log_prob_column(&plain, &residual, ParentBatch::empty(4), &mut plain_lp).unwrap();
        for (a, b) in conditional_lp.iter().zip(plain_lp) {
            assert!((a - b).abs() < 1e-12);
        }
        let mut noise = [0.0; 4];
        infer_noise_column_rng(
            &conditional,
            &values,
            batch,
            &mut noise,
            &mut CausalRng::from_seed(42),
        )
        .unwrap();
        let mut reconstructed = [0.0; 4];
        let mut workspace = MechanismWorkspace::default();
        evaluate_column(&conditional, batch, &noise, &mut reconstructed, &mut workspace).unwrap();
        for (actual, expected) in reconstructed.iter().zip(&values) {
            assert!((actual - expected).abs() < 1e-6);
        }
        let shifted = [1.0, 2.0, 3.0, 4.0];
        let shifted_batch = ParentBatch { values: &shifted, ..batch };
        let mut counterfactual = [0.0; 4];
        evaluate_column(&conditional, shifted_batch, &noise, &mut counterfactual, &mut workspace)
            .unwrap();
        for (cf, factual) in counterfactual.iter().zip(reconstructed) {
            assert!((cf - factual - 2.0).abs() < 1e-12);
        }
    }
}
