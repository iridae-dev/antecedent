//! G-squared and regression CI tests (Phase 5).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::all)]

use std::collections::HashMap;

use causal_core::ExecutionContext;

use super::analytic::normal_ppf;
use super::parcorr::PartialCorrelation;
use super::types::{
    CiBatchRequest, CiBatchResult, CiResult, CiWorkspace, ConditionalIndependenceTest,
    ConfidenceMethod,
};

#[cfg(test)]
use super::types::{CiQuery, SignificanceMethod};
use crate::error::StatsError;

/// G-squared conditional independence for discrete (integer-coded) columns.
#[derive(Clone, Debug, Default)]
pub struct GSquared;

impl GSquared {
    /// Construct.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl ConditionalIndependenceTest for GSquared {
    fn test_batch(
        &self,
        request: &CiBatchRequest<'_>,
        workspace: &mut CiWorkspace,
        _ctx: &ExecutionContext,
    ) -> Result<CiBatchResult, StatsError> {
        let n = request.columns.first().map(|c| c.len()).unwrap_or(0);
        if n == 0 {
            return Err(StatsError::Shape { message: "no columns" });
        }
        let mut results = Vec::with_capacity(request.queries.len());
        for q in request.queries {
            let z = &request.z_flat[q.z_start..q.z_start + q.z_len];
            let (g, df) = g_squared_statistic(request.columns, q.x, q.y, z, n, workspace)?;
            let p = chi2_sf(g, df);
            let ci = Some(analytic_gsquared_ci(g, df, 0.95));
            results.push(CiResult { statistic: g, p_value: p, df, ci });
        }
        Ok(CiBatchResult { results })
    }
}

/// Asymptotic 100`level`% Wald interval for the G² statistic under a central-χ²
/// variance proxy (`Var(G²) ≈ 2·df`). Lower bound is clipped at 0.
fn analytic_gsquared_ci(g: f64, df: f64, level: f64) -> (f64, f64) {
    let z = normal_ppf(0.5 + 0.5 * level.clamp(0.0, 1.0));
    let se = (2.0 * df.max(1.0)).sqrt();
    ((g - z * se).max(0.0), g + z * se)
}

fn g_squared_statistic(
    columns: &[&[f64]],
    x: usize,
    y: usize,
    z: &[usize],
    n: usize,
    workspace: &mut CiWorkspace,
) -> Result<(f64, f64), StatsError> {
    let xi: Vec<i32> = columns[x].iter().map(|v| v.round() as i32).collect();
    let yi: Vec<i32> = columns[y].iter().map(|v| v.round() as i32).collect();
    let mut strata: HashMap<u64, Vec<usize>> = HashMap::new();
    for r in 0..n {
        let key = if z.is_empty() {
            0u64
        } else {
            let mut h = 0xcbf29ce484222325u64;
            for &zc in z {
                let v = columns[zc][r].round() as i32;
                h ^= v as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
            h
        };
        strata.entry(key).or_default().push(r);
    }
    let mut g_total = 0.0;
    let mut df_total = 0.0;
    for rows in strata.values() {
        if rows.len() < 2 {
            continue;
        }
        let (g, df) = g_squared_on_rows(&xi, &yi, rows, workspace)?;
        g_total += g;
        df_total += df;
    }
    if df_total <= 0.0 {
        return Err(StatsError::Shape { message: "empty stratified contingency" });
    }
    Ok((g_total, df_total))
}

fn g_squared_on_rows(
    xi: &[i32],
    yi: &[i32],
    rows: &[usize],
    workspace: &mut CiWorkspace,
) -> Result<(f64, f64), StatsError> {
    let mut levels_x: Vec<i32> = rows.iter().map(|&r| xi[r]).collect();
    levels_x.sort_unstable();
    levels_x.dedup();
    let mut levels_y: Vec<i32> = rows.iter().map(|&r| yi[r]).collect();
    levels_y.sort_unstable();
    levels_y.dedup();
    let lx = levels_x.len().max(1);
    let ly = levels_y.len().max(1);
    let need = lx * ly;
    if workspace.shuffled.len() < need {
        workspace.shuffled.resize(need, 0.0);
    }
    for v in &mut workspace.shuffled[..need] {
        *v = 0.0;
    }
    for &r in rows {
        let ix = levels_x.binary_search(&xi[r]).unwrap_or(0);
        let iy = levels_y.binary_search(&yi[r]).unwrap_or(0);
        workspace.shuffled[ix * ly + iy] += 1.0;
    }
    let mut row_sum = vec![0.0; lx];
    let mut col_sum = vec![0.0; ly];
    let mut total = 0.0;
    for i in 0..lx {
        for j in 0..ly {
            let o = workspace.shuffled[i * ly + j];
            row_sum[i] += o;
            col_sum[j] += o;
            total += o;
        }
    }
    if total < 1.0 {
        return Ok((0.0, 0.0));
    }
    let mut g = 0.0;
    for i in 0..lx {
        for j in 0..ly {
            let o = workspace.shuffled[i * ly + j];
            let e = row_sum[i] * col_sum[j] / total;
            if o > 0.0 && e > 0.0 {
                g += 2.0 * o * (o / e).ln();
            }
        }
    }
    let df = ((lx - 1) * (ly - 1)) as f64;
    Ok((g, df.max(1.0)))
}

/// Upper survival function approximation for chi-squared via Wilson–Hilferty + normal SF.
fn chi2_sf(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    if df <= 0.0 {
        return 0.0;
    }
    let h = 2.0 / (9.0 * df);
    let z = (x / df).powf(1.0 / 3.0) - (1.0 - h);
    let z = z / h.sqrt();
    norm_sf(z)
}

/// Standard normal survival function Φ̄(z).
fn norm_sf(z: f64) -> f64 {
    // Abramowitz–Stegun 7.1.26 via erf approximation on |z|/√2
    0.5 * erfc_approx(z / std::f64::consts::SQRT_2)
}

fn erfc_approx(x: f64) -> f64 {
    // erfc(x) = 1 - erf(x); erf via A&S 7.1.26
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let erf = sign * (1.0 - poly * (-ax * ax).exp());
    1.0 - erf
}

/// Regression CI: residualize X,Y on Z via OLS then correlate residuals (same as ParCorr).
#[derive(Clone, Debug, Default)]
pub struct RegressionCi {
    inner: PartialCorrelation,
}

impl RegressionCi {
    /// Construct.
    #[must_use]
    pub fn new() -> Self {
        Self { inner: PartialCorrelation::new() }
    }
}

impl ConditionalIndependenceTest for RegressionCi {
    fn test_batch(
        &self,
        request: &CiBatchRequest<'_>,
        workspace: &mut CiWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<CiBatchResult, StatsError> {
        self.inner.test_batch(request, workspace, ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gsq_independent_high_p() {
        let n = 400usize;
        let x: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| ((i / 2) % 2) as f64).collect();
        let cols: [&[f64]; 2] = [&x, &y];
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 0 }];
        let req = CiBatchRequest {
            columns: &cols,
            queries: &queries,
            z_flat: &[],
            significance: SignificanceMethod::Analytic,
            confidence: ConfidenceMethod::default(),
        };
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let out = GSquared::new().test_batch(&req, &mut ws, &ctx).unwrap();
        assert!(out.results[0].p_value > 0.01, "p={}", out.results[0].p_value);
    }

    #[test]
    fn gsq_dependent_low_p() {
        let n = 400usize;
        let x: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
        let y: Vec<f64> = x.clone();
        let cols: [&[f64]; 2] = [&x, &y];
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 0 }];
        let req = CiBatchRequest {
            columns: &cols,
            queries: &queries,
            z_flat: &[],
            significance: SignificanceMethod::Analytic,
            confidence: ConfidenceMethod::default(),
        };
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(2);
        let out = GSquared::new().test_batch(&req, &mut ws, &ctx).unwrap();
        assert!(out.results[0].p_value < 1e-6);
        let ci = out.results[0].ci.expect("G² analytic CI");
        assert!(ci.0 <= out.results[0].statistic && out.results[0].statistic <= ci.1);
        assert!(ci.0 >= 0.0);
    }
}
