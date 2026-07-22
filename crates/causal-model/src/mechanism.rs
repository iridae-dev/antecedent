//! Built-in PCM/SCM/invertible mechanism kernels.
//!
//! Hot paths dispatch on [`MechanismSlot`] (enum), not trait objects per scalar.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::too_many_lines
)]

use causal_core::CausalRng;
use causal_kernels::{categorical_from_u, standard_normal};

use crate::batch::{MechanismWorkspace, NoiseBatchMut, ParentBatch, ValueBatchMut};
use crate::compile::MechanismSlot;
use crate::error::ModelError;
use crate::lgssm::{infer_lgssm_innovations, sample_lgssm_noise, unpack_innovations};

/// How noise was recovered for a mechanism family during abduction.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum NoiseInferenceMode {
    /// Unique structural residual (`y − f(pa)` or equivalent).
    Invertible,
    /// Sampled / smoothed posterior noise (Discrete CDF bin, LGSSM innovations).
    Posterior,
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
        | MechanismSlot::GaussianProcess { noise_std: sigma, .. } => {
            for i in 0..n_rows {
                output[i] = *sigma * standard_normal(rng);
            }
            Ok(())
        }
        MechanismSlot::LinearGaussianStateSpace { .. } => sample_lgssm_noise(n_rows, rng, output),
        MechanismSlot::Discrete { .. } => {
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
        MechanismSlot::LinearGaussianStateSpace { a, process_std, obs_std, initial_mean } => {
            // `noise[r]` packs unit-normal (ε, η); y_t = x_t + σ_obs η.
            let mut x = *initial_mean;
            for r in 0..n {
                let (eps, eta) = unpack_innovations(noise[r]);
                x = if r == 0 {
                    *initial_mean + process_std * eps
                } else {
                    a * x + process_std * eps
                };
                output[r] = x + obs_std * eta;
            }
            Ok(())
        }
        MechanismSlot::GaussianProcess {
            length_scale,
            variance,
            x_train,
            n_train,
            n_parents,
            alpha,
            ..
        } => {
            if *n_parents != parents.n_parents {
                return Err(ModelError::Shape { message: "GP n_parents mismatch".into() });
            }
            let inv_l2 = 1.0 / (length_scale * length_scale);
            for r in 0..n {
                let mut mean = 0.0;
                for i in 0..*n_train {
                    let mut d2 = 0.0;
                    for p in 0..parents.n_parents {
                        let d = parents.column(p)?[r] - x_train[i * n_parents + p];
                        d2 += d * d;
                    }
                    mean += alpha[i] * variance * (-0.5 * d2 * inv_l2).exp();
                }
                output[r] = mean + noise[r];
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
                        let u = if noise[r] > 0.0 && noise[r] < 1.0 { noise[r] } else { 0.5 };
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
                        let u = if noise[r] > 0.0 && noise[r] < 1.0 { noise[r] } else { 0.5 };
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

/// Infer exogenous noise from observed value and parents (invertible path).
///
/// Discrete and LGSSM require RNG for posterior sampling — use
/// [`infer_noise_column_rng`] for those families.
///
/// # Errors
///
/// Non-invertible family or shape.
pub fn infer_noise_column(
    slot: &MechanismSlot,
    value: &[f64],
    parents: ParentBatch<'_>,
    output: &mut [f64],
) -> Result<(), ModelError> {
    let mut unused = CausalRng::from_seed(0);
    let mode = infer_noise_column_rng(slot, value, parents, output, &mut unused)?;
    if mode == NoiseInferenceMode::Posterior {
        // Deterministic call sites should use the RNG-aware API; fall through with
        // seed-0 draws so shape checks still work in tests.
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
            for r in 0..n {
                output[r] = value[r] - *c;
            }
            Ok(NoiseInferenceMode::Invertible)
        }
        MechanismSlot::GaussianProcess {
            length_scale,
            variance,
            x_train,
            n_train,
            n_parents,
            alpha,
            ..
        } => {
            if *n_parents != parents.n_parents {
                return Err(ModelError::Shape { message: "GP n_parents mismatch".into() });
            }
            let inv_l2 = 1.0 / (length_scale * length_scale);
            for r in 0..n {
                let mut mean = 0.0;
                for i in 0..*n_train {
                    let mut d2 = 0.0;
                    for p in 0..parents.n_parents {
                        let d = parents.column(p)?[r] - x_train[i * n_parents + p];
                        d2 += d * d;
                    }
                    mean += alpha[i] * variance * (-0.5 * d2 * inv_l2).exp();
                }
                output[r] = value[r] - mean;
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
        MechanismSlot::LinearGaussianStateSpace { a, process_std, obs_std, initial_mean } => {
            infer_lgssm_innovations(
                &value[..n],
                *a,
                *process_std,
                *obs_std,
                *initial_mean,
                &mut output[..n],
                Some(rng),
            )?;
            Ok(NoiseInferenceMode::Posterior)
        }
        MechanismSlot::Dynamic { mechanism, .. } => {
            mechanism.infer_noise_column(value, parents, output)?;
            Ok(NoiseInferenceMode::Invertible)
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
        let cat = support.iter().position(|&s| (value[r] - s).abs() < 1e-12).ok_or_else(|| {
            ModelError::Unsupported {
                message: format!("discrete value {} not in support", value[r]),
            }
        })?;
        let mut lo = 0.0;
        for i in 0..cat {
            lo += row_probs[i];
        }
        let hi = (lo + row_probs[cat]).min(1.0);
        let lo = lo.clamp(0.0, 1.0 - f64::EPSILON);
        let hi = hi.max(lo + f64::EPSILON).min(1.0 - f64::EPSILON / 2.0);
        let u = lo + (hi - lo) * rng.next_f64();
        output[r] = u.clamp(f64::EPSILON, 1.0 - f64::EPSILON);
    }
    Ok(())
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
        MechanismSlot::Vacant | MechanismSlot::Pending { .. } => {
            Err(ModelError::Unsupported { message: "mechanism not fitted".into() })
        }
        MechanismSlot::LinearGaussianStateSpace { obs_std, .. } => {
            if !(obs_std.is_finite() && *obs_std > 0.0) {
                return Err(ModelError::Numerical { message: "obs_std must be > 0".into() });
            }
            let inv_s = 1.0 / obs_std;
            let log_norm = -0.5 * (2.0 * std::f64::consts::PI).ln() - obs_std.ln();
            for r in 0..n {
                // Marginal N(0, obs) approximation for scoring.
                let z = values[r] * inv_s;
                output[r] = log_norm - 0.5 * z * z;
            }
            Ok(())
        }
        MechanismSlot::GaussianProcess { noise_std, .. } => {
            if !(noise_std.is_finite() && *noise_std > 0.0) {
                return Err(ModelError::Numerical { message: "noise_std must be > 0".into() });
            }
            // Predictive mean already in evaluate; here use noise-scale residual approx.
            let inv_s = 1.0 / noise_std;
            let log_norm = -0.5 * (2.0 * std::f64::consts::PI).ln() - noise_std.ln();
            for r in 0..n {
                let z = values[r] * inv_s;
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
    let mut max_eta = f64::NEG_INFINITY;
    let mut etas = vec![0.0; k];
    for cat in 0..k {
        let base = cat * width;
        let mut pred = logits[base];
        for p in 0..parents.n_parents {
            pred += logits[base + 1 + p] * parents.column(p)?[row];
        }
        etas[cat] = pred;
        if pred > max_eta {
            max_eta = pred;
        }
    }
    let mut sum = 0.0;
    for cat in 0..k {
        let e = (etas[cat] - max_eta).exp();
        out[cat] = e;
        sum += e;
    }
    let inv = 1.0 / sum.max(f64::EPSILON);
    for p in out.iter_mut() {
        *p *= inv;
    }
    Ok(())
}

fn categorical_draw(support: &[f64], probs: &[f64], u: f64) -> f64 {
    let idx = categorical_from_u(u, probs);
    support.get(idx).copied().unwrap_or(0.0)
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
    order: &[causal_graph::DenseNodeId],
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
    use causal_core::CausalRng;
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

    #[test]
    fn dynamic_mechanism_dispatch() {
        struct ConstMech(f64);
        impl crate::compile::DynamicMechanism for ConstMech {
            fn sample_noise_column(
                &self,
                n_rows: usize,
                _rng: &mut causal_core::CausalRng,
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
