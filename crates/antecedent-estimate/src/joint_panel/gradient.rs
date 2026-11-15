// SPDX-License-Identifier: MIT OR Apache-2.0
//! Exact analytic score of the independent-group marginal log likelihood.
//!
//! The objective is the one `likelihood::marginal` evaluates, with every transform of the
//! unconstrained coordinates differentiated by the chain rule:
//!
//! * Per group, with analysed periods `t_1 < ... < t_m`, residuals `r`, `S` the stationary
//!   AR(1) covariance at the true time distances and `mu = tau^2 b / (1 + tau^2 ones)` the
//!   conditional intercept mean (`ones = 1'S^-1 1`, `b = 1'S^-1 r`, `Q = r'S^-1 r`,
//!   `D = log|S|`):
//!   `-2 l = m log 2 pi + D + log(1 + tau^2 ones) + Q - tau^2 b^2 / (1 + tau^2 ones)`.
//!   The residual derivative is `dl/dr = -S^-1 (r - mu 1)`, so the derivative with respect to
//!   the mean response of period `t` is `+[S^-1 (r - mu 1)]_t`; the noise derivatives follow
//!   from the partials of `ones`, `b`, `Q` and `D` in `rho` and `log sigma`.
//! * The mean response is `intercept + sum_c beta_c S_c + sum_j gamma_j F_j(S) +
//!   confounder z + season`, with `S_c = saturate(E_c, k, shape_c)` and the lag-weighted
//!   exposure `E_c = sum_l w_cl v_(t-l)`. The lag weights are a softmax of two logits or a
//!   normalized geometric sequence of a logistic decay, the half-saturation, shape and scales
//!   are exponentials of their coordinates, and `rho = tanh` of its coordinate.
//! * Interaction features are differentiated exactly through their portable expression
//!   (constants, variables, negation, the six arithmetic operators and `clip`); any other
//!   expression node falls back to a central difference of the compiled program for that
//!   feature alone.
//!
//! A spatially dependent residual law couples the groups through a Cholesky factor and keeps
//! the numerical score.
use super::likelihood::{
    CompiledPanel, ResponseScratch, decode, exposure, layout, observed_response_with,
    positive_exp_log_slope, saturate,
};
use super::{PanelGroup, PanelParameters};
use antecedent_core::{ArithOp, ValueExpr, VariableId, clip_value};

/// Derivatives of one specification's log likelihood in the unconstrained coordinates.
pub(super) struct Score {
    /// Derivatives through the residual law (`rho`, `log sigma`, `log tau`); zero elsewhere.
    pub shared: Vec<f64>,
    /// Derivatives through the mean mechanism, one full-length vector per mechanism slot.
    pub slots: Vec<Vec<f64>>,
}

/// Exact derivatives of one group's marginal log likelihood (unweighted).
pub(super) struct GroupScore {
    /// `d l / d rho`.
    pub rho: f64,
    /// `d l / d log sigma`.
    pub log_sigma: f64,
    /// `d l / d log tau`.
    pub log_tau: f64,
    /// `d l / d mean response` at every analysed period: `[S^-1 (r - mu 1)]_t`.
    pub response: Vec<f64>,
}

/// Gradient of `ar1_random_intercept_at` over analysed `(period, residual)` pairs; contiguous
/// periods reduce to the closed form of `ar1_random_intercept`.
#[allow(clippy::many_single_char_names, reason = "the notation of the module derivation")]
pub(super) fn group_score(
    observed: &[(usize, f64)],
    rho: f64,
    sigma: f64,
    tau: f64,
) -> Option<GroupScore> {
    if observed.is_empty()
        || !rho.is_finite()
        || rho.abs() >= 1.0
        || !sigma.is_finite()
        || sigma <= 0.0
        || !tau.is_finite()
        || tau <= 0.0
        || observed.iter().any(|(_, value)| !value.is_finite())
    {
        return None;
    }
    let m = observed.len();
    let stationary = 1.0 - rho * rho;
    let variance = sigma * sigma;
    let first = stationary.sqrt() / sigma;
    let r0 = observed[0].1;
    let (mut ones, mut cross, mut quad) = (first * first, first * first * r0, (first * r0).powi(2));
    // Derivatives in rho; the first period contributes `(1 - rho^2) / sigma^2` times 1, r, r^2.
    let (mut d_ones, mut d_cross, mut d_quad) =
        (-2.0 * rho / variance, -2.0 * rho * r0 / variance, -2.0 * rho * r0 * r0 / variance);
    let mut d_log_det = 2.0 * rho / stationary;
    let mut steps: Vec<(f64, f64)> = Vec::with_capacity(m - 1);
    for pair in observed.windows(2) {
        let distance = i32::try_from(pair[1].0 - pair[0].0).ok()?;
        let phi = rho.powi(distance);
        let q = (1.0 - rho.powi(2 * distance.min(1_000_000))) / stationary;
        // q = sum_{i<d} rho^(2i), so dq = sum_{i=1}^{d-1} 2 i rho^(2i-1).
        let mut d_q = 0.0;
        if distance > 1 {
            let mut power = rho;
            for i in 1..distance.min(1_000_000) {
                d_q += 2.0 * f64::from(i) * power;
                power *= rho * rho;
            }
        }
        let step_variance = variance * q;
        let inverse = 1.0 / step_variance.sqrt();
        let d_step_variance = variance * d_q;
        let d_inverse = -inverse * d_step_variance / (2.0 * step_variance);
        let d_phi = f64::from(distance) * rho.powi(distance - 1);
        let (numerator_one, numerator_r) = (1.0 - phi, pair[1].1 - phi * pair[0].1);
        let (u, z) = (numerator_one * inverse, numerator_r * inverse);
        let d_u = -d_phi * inverse + numerator_one * d_inverse;
        let d_z = -d_phi * pair[0].1 * inverse + numerator_r * d_inverse;
        ones += u * u;
        cross += u * z;
        quad += z * z;
        d_ones += 2.0 * u * d_u;
        d_cross += d_u * z + u * d_z;
        d_quad += 2.0 * z * d_z;
        d_log_det += d_step_variance / step_variance;
        steps.push((inverse, phi));
    }
    let tau2 = tau * tau;
    let shrinkage = 1.0 + tau2 * ones;
    let mean = tau2 * cross / shrinkage;
    // F = -2 l - m log 2 pi = D + log s + Q - tau^2 b^2 / s.
    let d_f_ones = tau2 / shrinkage + (tau2 * cross / shrinkage).powi(2);
    let d_f_cross = -2.0 * mean;
    let d_rho = d_log_det + d_f_ones * d_ones + d_f_cross * d_cross + d_quad;
    let d_log_sigma =
        d_f_ones * (-2.0 * ones) + d_f_cross * (-2.0 * cross) - 2.0 * quad + 2.0 * m as f64;
    let d_log_tau = 2.0 * tau2 * (ones / shrinkage - (cross / shrinkage).powi(2));
    // S^-1 x = A'A x with A the whitening of the stationary AR(1) process.
    let centred: Vec<f64> = observed.iter().map(|(_, value)| value - mean).collect();
    let mut whitened = Vec::with_capacity(m);
    whitened.push(first * centred[0]);
    for (k, &(inverse, phi)) in steps.iter().enumerate() {
        whitened.push((centred[k + 1] - phi * centred[k]) * inverse);
    }
    let mut response = Vec::with_capacity(m);
    for k in 0..m {
        let own = if k == 0 { first } else { steps[k - 1].0 };
        let next = if k + 1 < m { -steps[k].1 * steps[k].0 * whitened[k + 1] } else { 0.0 };
        response.push(own * whitened[k] + next);
    }
    let score = GroupScore {
        rho: -0.5 * d_rho,
        log_sigma: -0.5 * d_log_sigma,
        log_tau: -0.5 * d_log_tau,
        response,
    };
    (score.rho.is_finite()
        && score.log_sigma.is_finite()
        && score.log_tau.is_finite()
        && score.response.iter().all(|value| value.is_finite()))
    .then_some(score)
}

fn zip_with(a: &[f64], b: &[f64], f: impl Fn(f64, f64) -> f64) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| f(*x, *y)).collect()
}
/// Value and partial derivatives of an expression in its channel variables.
fn dual(expression: &ValueExpr, axis: &[VariableId], saturated: &[f64]) -> Option<(f64, Vec<f64>)> {
    let width = saturated.len();
    match expression {
        ValueExpr::Const(number) => Some((number.get(), vec![0.0; width])),
        ValueExpr::Var(variable) => {
            let index = axis.iter().position(|candidate| candidate == variable)?;
            let mut gradient = vec![0.0; width];
            gradient[index] = 1.0;
            Some((saturated[index], gradient))
        }
        ValueExpr::Neg(inner) => {
            let (value, gradient) = dual(inner, axis, saturated)?;
            Some((-value, gradient.into_iter().map(|g| -g).collect()))
        }
        ValueExpr::Arith { op, lhs, rhs } => {
            let (a, ga) = dual(lhs, axis, saturated)?;
            let (b, gb) = dual(rhs, axis, saturated)?;
            let gradient = match op {
                ArithOp::Add => zip_with(&ga, &gb, |x, y| x + y),
                ArithOp::Sub => zip_with(&ga, &gb, |x, y| x - y),
                ArithOp::Mul => zip_with(&ga, &gb, |x, y| x * b + y * a),
                ArithOp::Div => zip_with(&ga, &gb, |x, y| (x * b - a * y) / (b * b)),
                ArithOp::Min => {
                    if b < a {
                        gb
                    } else {
                        ga
                    }
                }
                ArithOp::Max => {
                    if b > a {
                        gb
                    } else {
                        ga
                    }
                }
                _ => return None,
            };
            Some((op.apply(a, b), gradient))
        }
        ValueExpr::Clip { value, lower, upper } => {
            let (v, gv) = dual(value, axis, saturated)?;
            let (l, gl) = dual(lower, axis, saturated)?;
            let (u, gu) = dual(upper, axis, saturated)?;
            let gradient = if v < l {
                gl
            } else if v > u {
                gu
            } else {
                gv
            };
            Some((clip_value(v, l, u), gradient))
        }
        _ => None,
    }
}

/// `dF_j / dS_c` for every channel `c`.
fn feature_partials(
    data: &CompiledPanel<'_>,
    index: usize,
    saturated: &[f64],
    out: &mut Vec<f64>,
    buffers: (&mut Vec<f64>, &mut Vec<f64>),
) {
    out.clear();
    if let Some((_, gradient)) = dual(&data.expressions[index], &data.axis, saturated) {
        out.extend(gradient);
        return;
    }
    // An expression node outside the exact set: central difference of the compiled program.
    let (values, scratch) = buffers;
    let mut work = saturated.to_vec();
    for channel in 0..saturated.len() {
        let original = saturated[channel];
        let step = 1e-6 * (1.0 + original.abs());
        work[channel] = original + step;
        let high = data.feature_value(index, &work, values, scratch);
        work[channel] = original - step;
        let low = data.feature_value(index, &work, values, scratch);
        work[channel] = original;
        out.push((high - low) / (2.0 * step));
    }
}

/// `(dS/dE, dS/dlog k, dS/dlog shape)` of `saturate(E, k, shape)`.
fn saturation_partials(exposure: f64, k: f64, shape: Option<f64>) -> (f64, f64, f64) {
    match shape {
        None => {
            let denominator = exposure + k;
            let square = denominator * denominator;
            (k / square, -exposure * k / square, 0.0)
        }
        Some(shape) => {
            if exposure <= 0.0 {
                return (0.0, 0.0, 0.0);
            }
            let z = (exposure / k).powf(shape);
            if z.is_infinite() {
                return (0.0, 0.0, 0.0);
            }
            let slope = 1.0 / ((1.0 + z) * (1.0 + z));
            (
                slope * shape * z / exposure,
                -slope * shape * z,
                slope * shape * z * (exposure / k).ln(),
            )
        }
    }
}

#[derive(Default)]
struct Buffers {
    saturated: Vec<f64>,
    partials: Vec<f64>,
    heights: Vec<f64>,
    values: Vec<f64>,
    scratch: Vec<f64>,
}

/// Back-propagate `scale * dl/df` of one `(group, period)` mean response into the
/// unconstrained mechanism coordinates `x` of its slot.
#[allow(
    clippy::too_many_lines,
    clippy::too_many_arguments,
    reason = "one chain rule through every mechanism transform"
)]
fn accumulate_mechanism(
    data: &CompiledPanel<'_>,
    p: &PanelParameters,
    x: &[f64],
    group: &PanelGroup,
    group_index: usize,
    time: usize,
    scale: f64,
    gradient: &mut [f64],
    buffers: &mut Buffers,
) {
    let layout = layout(data);
    let d = data.channels.len();
    let i = data.interactions.len();
    let class = data.family.class_of(group_index);
    let k = p.half_saturation_of(class);
    let k_index = if class == 0 { layout.k0 } else { layout.class + class - 1 };
    let Buffers { saturated, partials, heights, values, scratch } = buffers;
    let exposures: Vec<f64> = group
        .channels
        .iter()
        .zip(&group.history)
        .zip(&p.channels)
        .map(|((series, history), channel)| exposure(series, *history, channel.lag, time))
        .collect();
    saturated.clear();
    saturated.extend(
        exposures
            .iter()
            .enumerate()
            .map(|(c, e)| saturate(*e, k, p.family.hill_shape.get(c).copied())),
    );
    // Height of every saturated value: its own coefficient plus the feature terms.
    heights.clear();
    heights.extend(p.channels.iter().map(|channel| channel.coefficient));
    gradient[0] += scale;
    for (c, value) in saturated.iter().enumerate() {
        gradient[1 + c] += scale * value;
    }
    for j in 0..i {
        let value = data.feature_value(j, &saturated[..], values, scratch);
        gradient[1 + d + j] += scale * value;
        feature_partials(data, j, &saturated[..], partials, (&mut *values, &mut *scratch));
        let coefficient = p.interactions[j].coefficient;
        for (height, partial) in heights.iter_mut().zip(partials.iter()) {
            *height += coefficient * partial;
        }
    }
    gradient[1 + d + i] += scale * group.confounder[time];
    let season = group.season[time];
    if season > 0 {
        gradient[1 + d + i + season] += scale;
    }
    for c in 0..d {
        let shape = p.family.hill_shape.get(c).copied();
        let (d_exposure, d_log_k, d_log_shape) = saturation_partials(exposures[c], k, shape);
        let through = scale * heights[c];
        gradient[k_index] += through * d_log_k * positive_exp_log_slope(x[k_index]);
        if shape.is_some() {
            let index = layout.shape + c;
            gradient[index] += through * d_log_shape * positive_exp_log_slope(x[index]);
        }
        if data.lag_order != 2 {
            continue;
        }
        let at = |lag: usize| {
            if lag <= time {
                group.channels[c][time - lag]
            } else {
                group.history[c][2 + time - lag]
            }
        };
        let weights = p.channels[c].lag;
        let through_weight = [0, 1, 2].map(|lag| through * d_exposure * at(lag));
        if data.family.is_geometric() {
            let theta = crate::util::sigmoid(x[layout.lag + c]);
            let (sum, d_sum) = (1.0 + theta + theta * theta, 1.0 + 2.0 * theta);
            let d_u = [0.0, 1.0, 2.0 * theta];
            let d_theta: f64 = (0..3)
                .map(|lag| through_weight[lag] * (d_u[lag] - weights[lag] * d_sum) / sum)
                .sum();
            gradient[layout.lag + c] += d_theta * theta * (1.0 - theta);
        } else {
            let mean: f64 = (0..3).map(|lag| through_weight[lag] * weights[lag]).sum();
            for coordinate in 0..2 {
                gradient[layout.lag + 2 * c + coordinate] +=
                    weights[coordinate] * (through_weight[coordinate] - mean);
            }
        }
    }
}

/// Exact score of the independent-group likelihood. `shared_x` holds the residual-law
/// coordinates; period `t` uses the mechanism coordinates `slot_x[slot_of(t)]`. `None` for a
/// spatial residual law, an invalid decoded parameter or a numerical failure.
pub(super) fn independent_score(
    data: &CompiledPanel<'_>,
    shared_x: &[f64],
    slot_x: &[Vec<f64>],
    slot_of: &dyn Fn(usize) -> usize,
) -> Option<Score> {
    if data.family.spatial.is_some() {
        return None;
    }
    let layout = layout(data);
    let base = decode(data, shared_x)?;
    let slot_p: Vec<PanelParameters> =
        slot_x.iter().map(|x| decode(data, x)).collect::<Option<_>>()?;
    let mut shared = vec![0.0; shared_x.len()];
    let mut slots = vec![vec![0.0; shared_x.len()]; slot_x.len()];
    let mut responses = ResponseScratch::default();
    let mut buffers = Buffers::default();
    for (group_index, group) in data.groups.iter().enumerate() {
        let mut observed = Vec::with_capacity(group.outcome.len());
        for time in 0..group.outcome.len() {
            if group.outcome[time].is_finite() {
                let fitted = observed_response_with(
                    data,
                    &slot_p[slot_of(time)],
                    group,
                    group_index,
                    time,
                    &mut responses,
                );
                observed.push((time, group.outcome[time] - fitted));
            }
        }
        let score = group_score(&observed, base.rho, base.innovation_sd, base.group_sd)?;
        let weight = group.weight;
        if let Some(index) = layout.rho {
            shared[index] += weight * score.rho * (1.0 - base.rho * base.rho);
        }
        shared[layout.sigma] +=
            weight * score.log_sigma * positive_exp_log_slope(shared_x[layout.sigma]);
        shared[layout.tau] += weight * score.log_tau * positive_exp_log_slope(shared_x[layout.tau]);
        for (&(time, _), response) in observed.iter().zip(&score.response) {
            let slot = slot_of(time);
            accumulate_mechanism(
                data,
                &slot_p[slot],
                &slot_x[slot],
                group,
                group_index,
                time,
                weight * response,
                &mut slots[slot],
                &mut buffers,
            );
        }
    }
    let finite = shared.iter().chain(slots.iter().flatten()).all(|value| value.is_finite());
    finite.then_some(Score { shared, slots })
}
