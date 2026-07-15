//! Built-in PCM/SCM/invertible mechanism kernels (DESIGN.md §15.2).
//!
//! Hot paths dispatch on [`MechanismSlot`] (enum), not trait objects per scalar.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::needless_range_loop, clippy::many_single_char_names)]

use causal_core::CausalRng;
use causal_kernels::standard_normal;

use crate::batch::{MechanismWorkspace, NoiseBatchMut, ParentBatch, ValueBatchMut};
use crate::compile::MechanismSlot;
use crate::error::ModelError;

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
        MechanismSlot::LinearGaussian { sigma, .. } => {
            for i in 0..n_rows {
                output[i] = *sigma * standard_normal(rng);
            }
            Ok(())
        }
        MechanismSlot::Discrete { .. } => {
            // Uniform(0,1) drives categorical draws in evaluate / sample_column.
            for i in 0..n_rows {
                output[i] = rng.next_f64().clamp(f64::EPSILON, 1.0 - f64::EPSILON);
            }
            Ok(())
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
    _ws: &mut MechanismWorkspace,
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
        MechanismSlot::LinearGaussian { intercept, coeffs, .. } => {
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
    }
}

/// Infer exogenous noise from observed value and parents (invertible path).
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
    let n = parents.n_rows;
    if value.len() < n || output.len() < n {
        return Err(ModelError::Shape { message: "infer_noise buffers too short".into() });
    }
    match slot {
        MechanismSlot::LinearGaussian { intercept, coeffs, .. } => {
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
            Ok(())
        }
        MechanismSlot::Constant { value: c } => {
            for r in 0..n {
                output[r] = value[r] - *c;
            }
            Ok(())
        }
        _ => Err(ModelError::Unsupported {
            message: "noise inference requires invertible linear/constant mechanism".into(),
        }),
    }
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
        MechanismSlot::LinearGaussian { intercept, coeffs, sigma } => {
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
    // Interim until multinomial IRLS (P5): registry stores one-vs-rest linear-probability
    // coefficients. Applying ln(clip(π)) before softmax recovers ≈π (softmax(π) ≠ π).
    const EPS: f64 = 1e-6;
    let mut max_eta = f64::NEG_INFINITY;
    let mut etas = vec![0.0; k];
    for cat in 0..k {
        let base = cat * width;
        let mut pred = logits[base];
        for p in 0..parents.n_parents {
            pred += logits[base + 1 + p] * parents.column(p)?[row];
        }
        let eta = pred.clamp(EPS, 1.0 - EPS).ln();
        etas[cat] = eta;
        if eta > max_eta {
            max_eta = eta;
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
    let sum: f64 = probs.iter().sum::<f64>().max(f64::EPSILON);
    let mut acc = 0.0;
    let target = u * sum;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if target <= acc {
            return support[i];
        }
    }
    *support.last().unwrap_or(&0.0)
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
}
