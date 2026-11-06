//! Gap-aware segmentation and long-run tempering.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use super::{
    CompiledDesign, DependenceScope, EstimationError, MIN_ROWS, TemperingFactor, dot,
    newey_west_bandwidth, scope_directions,
};

/// Segment ids of the intact windows of a series with gaps: windows whose positions on the
/// original axis are adjacent share a segment, and a gap starts a new one.
#[must_use]
pub fn window_segments(complete: &[bool]) -> Arc<[u32]> {
    let mut ids = Vec::new();
    let mut segment = 0u32;
    let mut previous: Option<usize> = None;
    for (position, intact) in complete.iter().enumerate() {
        if !*intact {
            continue;
        }
        if previous.is_some_and(|last| last + 1 != position) {
            segment += 1;
        }
        ids.push(segment);
        previous = Some(position);
    }
    Arc::from(ids)
}

/// Bartlett long-run variance of `u` summed over autocovariances within segments only,
/// divided by the number of rows (uncentred). Two rows in different segments never
/// contribute a cross product, so no pair that straddles a gap enters.
pub(super) fn segmented_long_run_variance(u: &[f64], segments: &[u32], bandwidth: usize) -> f64 {
    let n = u.len();
    let mut total = 0.0;
    let mut start = 0;
    while start < n {
        let mut end = start + 1;
        while end < n && segments[end] == segments[start] {
            end += 1;
        }
        let part = &u[start..end];
        let m = part.len();
        let lags = bandwidth.min(m.saturating_sub(1));
        total += dot(part, part);
        for lag in 1..=lags {
            let weight = 1.0 - lag as f64 / (lags + 1) as f64;
            total += 2.0 * weight * dot(&part[lag..], &part[..m - lag]);
        }
        start = end;
    }
    (total / n as f64).max(0.0)
}

/// The tempering factor of a design whose rows are the intact windows of a series with gaps.
///
/// For each targeted combination `c'β`, the row weights `w = X (X'X)⁻¹ c` and the OLS residual
/// `e` give the estimating score `s_i = w_i e_i`, whose segment-wise Bartlett long-run variance
/// over its iid variance `Σ s_i²/n` is the factor (floored at 1, capped to leave `p + 2`
/// effective rows); the largest over the combinations drives `κ̂`. No autocovariance crosses a
/// gap. This is the HAC ratio only: the REML autoregression and its small-sample scale
/// correction are defined for consecutive rows and are not applied.
///
/// # Errors
///
/// A wrong-width direction, a missing treatment column, or a singular `X'X`.
pub(crate) fn segmented_tempering_factor(
    design: &CompiledDesign,
    scope: &DependenceScope,
    segments: &[u32],
) -> Result<TemperingFactor, EstimationError> {
    let n = design.nrows;
    let p = design.ncols;
    let bandwidth = newey_west_bandwidth(n);
    let floor = TemperingFactor {
        kappa: 1.0,
        raw_ratio: 1.0,
        ar_ratio: 1.0,
        df_loss: p as f64,
        kappa_log_sd: 0.0,
        hac_ratio: 1.0,
        fixed_b: 1.0,
        score_ar_order: 0,
        residual_ar_order: 0,
        bandwidth,
        nrows: n,
        bounded: false,
        capped: false,
        inestimable: false,
        scope: scope.label(),
    };
    if segments.len() != n {
        return Err(EstimationError::stats_msg(
            "segmented tempering needs one segment id per design row",
        ));
    }
    let directions: Vec<Vec<f64>> = scope_directions(design, scope)?
        .into_iter()
        .filter(|c| c.iter().all(|v| v.is_finite()) && c.iter().any(|v| *v != 0.0))
        .collect();
    if n < MIN_ROWS.max(p + 2) {
        return Ok(TemperingFactor { inestimable: true, ..floor });
    }
    if directions.is_empty() {
        return Ok(floor);
    }
    let x = &design.matrix[..n * p];
    let y = &design.outcome[..n];
    let mut xtx = vec![0.0; p * p];
    let mut xty = vec![0.0; p];
    for a in 0..p {
        xty[a] = dot(&x[a * n..(a + 1) * n], y);
        for b in 0..p {
            xtx[a * p + b] = dot(&x[a * n..(a + 1) * n], &x[b * n..(b + 1) * n]);
        }
    }
    let beta = crate::util::solve_spd(&xtx, &xty, p)
        .ok_or_else(|| EstimationError::stats_msg("segmented tempering: singular X'X"))?;
    let residuals: Vec<f64> =
        (0..n).map(|t| y[t] - (0..p).map(|k| x[k * n + t] * beta[k]).sum::<f64>()).collect();
    let mut ratio = 1.0_f64;
    for c in &directions {
        let v = crate::util::solve_spd(&xtx, c, p)
            .ok_or_else(|| EstimationError::stats_msg("segmented tempering: singular X'X"))?;
        let score: Vec<f64> = (0..n)
            .map(|t| (0..p).map(|k| x[k * n + t] * v[k]).sum::<f64>() * residuals[t])
            .collect();
        let gamma0 = dot(&score, &score) / n as f64;
        if gamma0 <= 0.0 || !gamma0.is_finite() {
            continue;
        }
        ratio = ratio.max(segmented_long_run_variance(&score, segments, bandwidth) / gamma0);
    }
    let cap = (n as f64 / (p + 2) as f64).max(1.0);
    Ok(TemperingFactor {
        kappa: ratio.clamp(1.0, cap),
        raw_ratio: ratio,
        hac_ratio: ratio,
        capped: ratio > cap,
        ..floor
    })
}
