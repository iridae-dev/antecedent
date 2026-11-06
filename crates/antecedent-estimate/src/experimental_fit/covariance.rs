//! Joint covariance assembly for experimentally recovered coefficients.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    BlockSolution, BlockSystem, CoefficientCovarianceEstimator, ConstraintWeighting,
    ExperimentalFitError, F64_BYTES, RegimeLayout, RowGroup, dot, gram_of_upper_inverse,
};

pub(super) struct CovarianceInputs<'s, 'a> {
    pub(super) systems: &'s [BlockSystem],
    pub(super) solutions: &'s [BlockSolution],
    pub(super) offsets: &'s [Option<usize>],
    pub(super) vectors: &'s [Option<Vec<f64>>],
    pub(super) variances: &'s [Option<f64>],
    pub(super) regimes: &'s [RegimeLayout],
    pub(super) columns: &'s [Vec<&'a [f64]>],
    pub(super) weighting: ConstraintWeighting,
    pub(super) max_bytes: u64,
}

/// Joint covariance `Σ_r G_{s,r} S_{st,r} G_{t,r}'` of the resolved coefficients.
///
/// `G_{s,r} = B_s⁻¹ A_{s,r}' W_r Σ_JJ⁻¹ / n_r` is the influence of regime `r` on
/// block `s`. The sandwich uses `S_{st,r} = Σ_rows z z' ê_s ê_t`; the model-based
/// form uses `S_{ss,r} = σ̂_s² n_r Σ_JJ` and zero across blocks.
pub(super) fn joint_covariance(
    inputs: &CovarianceInputs<'_, '_>,
    p: usize,
    estimator: CoefficientCovarianceEstimator,
) -> Result<Vec<f64>, ExperimentalFitError> {
    if p == 0 {
        return Ok(Vec::new());
    }
    check_covariance_budget(inputs, p)?;
    let mut covariance = vec![0.0; p * p];
    for (regime, layout) in inputs.regimes.iter().enumerate() {
        let j = layout.clamped.len();
        if j == 0 {
            continue;
        }
        let members: Vec<(usize, RowGroup)> = inputs
            .systems
            .iter()
            .enumerate()
            .filter(|(index, _)| inputs.offsets[*index].is_some())
            .filter_map(|(index, system)| {
                system.groups.iter().find(|group| group.regime == regime).map(|g| (index, *g))
            })
            .collect();
        if members.is_empty() {
            continue;
        }
        let gains = members
            .iter()
            .map(|(index, group)| {
                regime_gain(
                    &inputs.systems[*index],
                    *group,
                    &inputs.solutions[*index],
                    layout,
                    inputs.weighting,
                )
            })
            .collect::<Option<Vec<_>>>()
            .ok_or(ExperimentalFitError::CovarianceNotPositiveSemidefinite)?;
        let scatter = match estimator {
            CoefficientCovarianceEstimator::Sandwich => {
                sandwich_scatter(inputs, regime, layout, &members)
            }
            _ => model_scatter(inputs, layout, &members),
        };
        let t = members.len();
        for left in 0..t {
            for right in left..t {
                let Some(s) = &scatter[left * t + right] else {
                    continue;
                };
                add_gain_product(inputs, &members, &gains, (left, right), s, j, &mut covariance, p);
            }
        }
    }
    for row in 0..p {
        for col in (row + 1)..p {
            let mean = 0.5 * (covariance[row * p + col] + covariance[col * p + row]);
            covariance[row * p + col] = mean;
            covariance[col * p + row] = mean;
        }
    }
    Ok(covariance)
}

fn check_covariance_budget(
    inputs: &CovarianceInputs<'_, '_>,
    p: usize,
) -> Result<(), ExperimentalFitError> {
    let word =
        |value: usize| u64::try_from(value).map_err(|_| ExperimentalFitError::ResourceOverflow);
    let p64 = word(p)?;
    let mut cells = p64.checked_mul(p64).ok_or(ExperimentalFitError::ResourceOverflow)?;
    let mut largest = 0_u64;
    for (regime, layout) in inputs.regimes.iter().enumerate() {
        let members = inputs
            .systems
            .iter()
            .filter(|system| system.groups.iter().any(|group| group.regime == regime))
            .count();
        let j = word(layout.clamped.len())?;
        let t = word(members)?;
        let scatter = t
            .checked_mul(t)
            .and_then(|pairs| pairs.checked_mul(j))
            .and_then(|pairs| pairs.checked_mul(j))
            .ok_or(ExperimentalFitError::ResourceOverflow)?;
        largest = largest.max(scatter);
    }
    cells = cells.checked_add(largest).ok_or(ExperimentalFitError::ResourceOverflow)?;
    let requested = cells.checked_mul(F64_BYTES).ok_or(ExperimentalFitError::ResourceOverflow)?;
    if requested > inputs.max_bytes {
        return Err(ExperimentalFitError::BudgetExceeded { requested, limit: inputs.max_bytes });
    }
    Ok(())
}

/// `G = B⁻¹ M`, `k × j`, with `M = A_r'` (efficient) or `A_r' Σ_JJ⁻¹ / n_r` (unweighted).
fn regime_gain(
    system: &BlockSystem,
    group: RowGroup,
    solution: &BlockSolution,
    layout: &RegimeLayout,
    weighting: ConstraintWeighting,
) -> Option<Vec<f64>> {
    let k = system.sources.len();
    let j = group.len;
    let mut m = vec![0.0; k * j];
    match weighting {
        ConstraintWeighting::Efficient => {
            for c in 0..k {
                for p in 0..j {
                    m[c * j + p] = system.a[(group.start + p) * k + c];
                }
            }
        }
        ConstraintWeighting::Unweighted => {
            let inverse = layout.clamped_inverse.as_ref()?;
            let rows = layout.rows as f64;
            for c in 0..k {
                for p in 0..j {
                    m[c * j + p] = (0..j)
                        .map(|q| system.a[(group.start + q) * k + c] * inverse[q * j + p])
                        .sum::<f64>()
                        / rows;
                }
            }
        }
    }
    let b_inverse = gram_of_upper_inverse(&solution.r_inverse, k);
    let mut gain = vec![0.0; k * j];
    for row in 0..k {
        for p in 0..j {
            gain[row * j + p] = (0..k).map(|c| b_inverse[row * k + c] * m[c * j + p]).sum();
        }
    }
    Some(gain)
}

fn sandwich_scatter(
    inputs: &CovarianceInputs<'_, '_>,
    regime: usize,
    layout: &RegimeLayout,
    members: &[(usize, RowGroup)],
) -> Vec<Option<Vec<f64>>> {
    let t = members.len();
    let j = layout.clamped.len();
    let columns = &inputs.columns[regime];
    let n = columns.len();
    let vectors: Vec<&[f64]> = members
        .iter()
        .map(|(index, _)| inputs.vectors[inputs.systems[*index].target].as_deref().unwrap_or(&[]))
        .collect();
    let mut scatter: Vec<Option<Vec<f64>>> = vec![None; t * t];
    for left in 0..t {
        for right in left..t {
            scatter[left * t + right] = Some(vec![0.0; j * j]);
        }
    }
    let mut dx = vec![0.0; n];
    let mut z = vec![0.0; j];
    let mut residual = vec![0.0; t];
    for row in 0..layout.rows {
        for (c, column) in columns.iter().enumerate() {
            dx[c] = column[row] - layout.mean[c];
        }
        for (p, &clamp) in layout.clamped.iter().enumerate() {
            z[p] = dx[clamp];
        }
        for (slot, v) in residual.iter_mut().zip(&vectors) {
            *slot = dot(v, &dx);
        }
        for left in 0..t {
            for right in left..t {
                let weight = residual[left] * residual[right];
                if let Some(cell) = scatter[left * t + right].as_mut() {
                    for p in 0..j {
                        for q in 0..j {
                            cell[p * j + q] += weight * z[p] * z[q];
                        }
                    }
                }
            }
        }
    }
    scatter
}

fn model_scatter(
    inputs: &CovarianceInputs<'_, '_>,
    layout: &RegimeLayout,
    members: &[(usize, RowGroup)],
) -> Vec<Option<Vec<f64>>> {
    let t = members.len();
    let rows = layout.rows as f64;
    let mut scatter: Vec<Option<Vec<f64>>> = vec![None; t * t];
    for (slot, (index, _)) in members.iter().enumerate() {
        let variance = inputs.variances[inputs.systems[*index].target].unwrap_or(f64::NAN);
        scatter[slot * t + slot] =
            Some(layout.clamped_covariance.iter().map(|value| variance * rows * value).collect());
    }
    scatter
}

#[allow(clippy::too_many_arguments, reason = "one accumulation step over named pieces")]
fn add_gain_product(
    inputs: &CovarianceInputs<'_, '_>,
    members: &[(usize, RowGroup)],
    gains: &[Vec<f64>],
    (left, right): (usize, usize),
    scatter: &[f64],
    j: usize,
    covariance: &mut [f64],
    p: usize,
) {
    let (left_system, right_system) = (members[left].0, members[right].0);
    let (Some(left_offset), Some(right_offset)) =
        (inputs.offsets[left_system], inputs.offsets[right_system])
    else {
        return;
    };
    let kl = inputs.systems[left_system].sources.len();
    let kr = inputs.systems[right_system].sources.len();
    let gl = &gains[left];
    let gr = &gains[right];
    // middle = S G_right'  (j × kr)
    let mut middle = vec![0.0; j * kr];
    for a in 0..j {
        for c in 0..kr {
            middle[a * kr + c] = (0..j).map(|b| scatter[a * j + b] * gr[c * j + b]).sum();
        }
    }
    for row in 0..kl {
        for col in 0..kr {
            let value: f64 = (0..j).map(|a| gl[row * j + a] * middle[a * kr + col]).sum();
            covariance[(left_offset + row) * p + right_offset + col] += value;
            if left != right {
                covariance[(right_offset + col) * p + left_offset + row] += value;
            }
        }
    }
}
