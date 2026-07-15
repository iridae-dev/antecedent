//! G-squared and regression CI tests .
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::doc_markdown
)]

use std::collections::HashMap;

use causal_core::{CausalRng, ExecutionContext};

use super::analytic::{ln_gamma, normal_ppf};
use super::parcorr::PartialCorrelation;
use super::types::{
    CiBatchRequest, CiBatchResult, CiResult, CiWorkspace, ConditionalIndependenceTest,
    analytic_confidence_level,
};

#[cfg(test)]
use super::types::{CiQuery, ConfidenceMethod, SignificanceMethod};
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
        ctx: &ExecutionContext,
    ) -> Result<CiBatchResult, StatsError> {
        let n = request.columns.first().map_or(0, |c| c.len());
        if n == 0 {
            return Err(StatsError::Shape { message: "no columns" });
        }
        let level = analytic_confidence_level(request.confidence);
        let mut results = Vec::with_capacity(request.queries.len());
        for (qi, q) in request.queries.iter().enumerate() {
            let z = &request.z_flat[q.z_start..q.z_start + q.z_len];
            let (g, df) = g_squared_statistic(request.columns, q.x, q.y, z, n, workspace)?;
            let (p, ci) = match request.significance {
                super::types::SignificanceMethod::Analytic => {
                    let p = chi2_sf(g, df);
                    let ci = level.map(|lv| analytic_gsquared_ci(g, df, lv));
                    (p, ci)
                }
                super::types::SignificanceMethod::BlockShuffle { replicates, block_size } => {
                    let n_perm = replicates.max(1) as usize;
                    let strata = gsq_strata(request.columns, z, n);
                    let mut y_perm = request.columns[q.y].to_vec();
                    let mut rng = ctx.rng.stream(0x65C0_u64.wrapping_add(qi as u64));
                    let mut null_ge = 0u32;
                    for _ in 0..n_perm {
                        if block_size > 1 && z.is_empty() {
                            block_shuffle_y(&mut y_perm, block_size, &mut rng);
                        } else {
                            for rows in &strata {
                                for i in (1..rows.len()).rev() {
                                    let j = (rng.next_u64() as usize) % (i + 1);
                                    y_perm.swap(rows[i], rows[j]);
                                }
                            }
                        }
                        let mut cols: Vec<&[f64]> = request.columns.to_vec();
                        cols[q.y] = &y_perm;
                        let (g_null, _) = g_squared_statistic(&cols, q.x, q.y, z, n, workspace)?;
                        if g_null >= g {
                            null_ge = null_ge.saturating_add(1);
                        }
                    }
                    let p = (1.0 + f64::from(null_ge)) / (1.0 + n_perm as f64);
                    (p, None)
                }
            };
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

fn gsq_strata(columns: &[&[f64]], z: &[usize], n: usize) -> Vec<Vec<usize>> {
    let mut strata: HashMap<u64, Vec<usize>> = HashMap::new();
    for r in 0..n {
        let key = if z.is_empty() {
            0u64
        } else {
            let mut h = 0xcbf2_9ce4_8422_2325_u64;
            for &zc in z {
                let v = columns[zc][r].round() as i32;
                h ^= u64::from(v as u32);
                h = h.wrapping_mul(0x0100_0000_01b3);
            }
            h
        };
        strata.entry(key).or_default().push(r);
    }
    let mut keys: Vec<u64> = strata.keys().copied().collect();
    keys.sort_unstable();
    keys.into_iter().filter_map(|k| strata.remove(&k)).collect()
}

fn block_shuffle_y(y: &mut [f64], block_size: usize, rng: &mut CausalRng) {
    let n = y.len();
    let bs = block_size.max(1).min(n);
    let n_blocks = n.div_ceil(bs);
    let mut order: Vec<usize> = (0..n_blocks).collect();
    for i in (1..order.len()).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        order.swap(i, j);
    }
    let original = y.to_vec();
    let mut dest = 0;
    for &bi in &order {
        let start = bi * bs;
        let end = (start + bs).min(n);
        let len = end - start;
        y[dest..dest + len].copy_from_slice(&original[start..end]);
        dest += len;
    }
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
            let mut h = 0xcbf2_9ce4_8422_2325_u64;
            for &zc in z {
                let v = columns[zc][r].round() as i32;
                h ^= u64::from(v as u32);
                h = h.wrapping_mul(0x0100_0000_01b3);
            }
            h
        };
        strata.entry(key).or_default().push(r);
    }
    let mut g_total = 0.0;
    let mut df_total = 0.0;
    let mut any = false;
    for rows in strata.values() {
        if rows.len() < 2 {
            continue;
        }
        let (g, df) = g_squared_on_rows(&xi, &yi, rows, workspace);
        g_total += g;
        df_total += df;
        any = true;
    }
    if !any {
        return Err(StatsError::Shape { message: "empty stratified contingency" });
    }
    // Per-stratum dof sums (rows-1)(cols-1) over nonempty strata (pgmpy/DoWhy-style);
    // a stratum with constant X or Y contributes 0. Floor the TOTAL at 1 to avoid a
    // df=0 chi-square.
    Ok((g_total, df_total.max(1.0)))
}

fn g_squared_on_rows(
    xi: &[i32],
    yi: &[i32],
    rows: &[usize],
    workspace: &mut CiWorkspace,
) -> (f64, f64) {
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
        return (0.0, 0.0);
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
    (g, df)
}

/// Exact chi-squared survival function via the regularized upper incomplete gamma
/// `Q(df/2, x/2)`.
fn chi2_sf(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    if df <= 0.0 {
        return 0.0;
    }
    gamma_q(df * 0.5, x * 0.5)
}

/// Regularized upper incomplete gamma `Q(a, x)`: series for `x < a + 1`, continued
/// fraction otherwise (Numerical Recipes `gammq` style).
fn gamma_q(a: f64, x: f64) -> f64 {
    if x < a + 1.0 {
        (1.0 - gamma_p_series(a, x)).clamp(0.0, 1.0)
    } else {
        gamma_q_cf(a, x).clamp(0.0, 1.0)
    }
}

/// Lower regularized incomplete gamma `P(a, x)` by series expansion.
fn gamma_p_series(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut del = sum;
    for _ in 0..500 {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * 1e-15 {
            break;
        }
    }
    sum * (-x + a * x.ln() - ln_gamma(a)).exp()
}

/// Upper regularized incomplete gamma `Q(a, x)` by Lentz continued fraction.
fn gamma_q_cf(a: f64, x: f64) -> f64 {
    const TINY: f64 = 1e-300;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / TINY;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..500 {
        let an = -f64::from(i) * (f64::from(i) - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < TINY {
            d = TINY;
        }
        c = b + an / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-15 {
            break;
        }
    }
    (-x + a * x.ln() - ln_gamma(a)).exp() * h
}

/// Residualize-then-correlate CI — an explicit alias of [`PartialCorrelation`].
///
/// Kept as a named entry point for callers who think in regression terms; the
/// statistic is identical to ParCorr (DESIGN.md §12).
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

    #[test]
    fn chi2_sf_pins_known_values() {
        assert!((chi2_sf(3.841_459, 1.0) - 0.05).abs() < 1e-5);
        assert!((chi2_sf(5.991_465, 2.0) - 0.05).abs() < 1e-5);
        assert!((chi2_sf(10.0, 1.0) - 0.001_565).abs() < 1e-5);
    }

    #[test]
    fn constant_strata_contribute_zero_dof() {
        // Z has two strata; in each stratum X is constant, so per-stratum dof is 0 and
        // the total floors at 1 (not the old per-stratum max which summed to 2).
        let n = 80usize;
        let z: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
        let x = z.clone(); // constant within each Z stratum
        let y: Vec<f64> = (0..n).map(|i| ((i / 2) % 2) as f64).collect();
        let cols: [&[f64]; 3] = [&x, &y, &z];
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 1 }];
        let z_flat = [2usize];
        let req = CiBatchRequest {
            columns: &cols,
            queries: &queries,
            z_flat: &z_flat,
            significance: SignificanceMethod::Analytic,
            confidence: ConfidenceMethod::default(),
        };
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(3);
        let out = GSquared::new().test_batch(&req, &mut ws, &ctx).unwrap();
        assert!((out.results[0].df - 1.0).abs() < 1e-12, "df={}", out.results[0].df);
        assert!((out.results[0].p_value - 1.0).abs() < 1e-9);
    }
}
