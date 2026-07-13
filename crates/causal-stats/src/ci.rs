//! Conditional independence tests (DESIGN.md §12).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::trivially_copy_pass_by_ref,
    clippy::unnecessary_wraps
)]

use causal_core::{ExecutionContext, KernelPolicy};
use causal_kernels::{
    ParCorrQuery, ParCorrWorkspace, partial_correlation, partial_correlation_batch,
};

use crate::error::StatsError;

/// Significance method for a CI statistic.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum SignificanceMethod {
    /// Analytic Fisher-z / Student-t for partial correlation.
    Analytic,
    /// Block-shuffle null distribution.
    BlockShuffle {
        /// Number of null replicates.
        replicates: u32,
        /// Block length for shuffling.
        block_size: usize,
    },
}

/// One CI query over column indexes into a shared matrix.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct CiQuery {
    /// X column index.
    pub x: usize,
    /// Y column index.
    pub y: usize,
    /// Start into flat conditioning indexes.
    pub z_start: usize,
    /// Conditioning arity.
    pub z_len: usize,
}

/// Batch of CI queries (deterministic output order).
#[derive(Clone, Debug)]
pub struct CiBatchRequest<'a> {
    /// Column-major / list of equal-length float columns.
    pub columns: &'a [&'a [f64]],
    /// Queries.
    pub queries: &'a [CiQuery],
    /// Flat conditioning column indexes.
    pub z_flat: &'a [usize],
    /// Significance.
    pub significance: SignificanceMethod,
}

/// One CI result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CiResult {
    /// Test statistic (partial correlation for partial-correlation CI).
    pub statistic: f64,
    /// Two-sided p-value.
    pub p_value: f64,
    /// Residual degrees of freedom (analytic path).
    pub df: f64,
}

/// Batch results aligned with request queries.
#[derive(Clone, Debug, Default)]
pub struct CiBatchResult {
    /// Per-query results.
    pub results: Vec<CiResult>,
}

/// Conditional independence test.
pub trait ConditionalIndependence {
    /// Evaluate a batch of queries.
    ///
    /// # Errors
    ///
    /// Shape / numerical failures.
    fn test_batch(
        &self,
        request: &CiBatchRequest<'_>,
        workspace: &mut CiWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<CiBatchResult, StatsError>;
}

/// Shared scratch for CI batches.
#[derive(Clone, Debug, Default)]
pub struct CiWorkspace {
    /// Partial-correlation residualization workspace.
    pub parcorr: ParCorrWorkspace,
    /// Temporary statistic buffer.
    pub stats: Vec<Option<f64>>,
    /// Block-shuffle column scratch.
    pub shuffled: Vec<f64>,
    /// Block starts for shuffle.
    pub block_perm: Vec<usize>,
}

impl CiWorkspace {
    /// Prepare for `n_queries` results.
    pub fn prepare_queries(&mut self, n_queries: usize) {
        if self.stats.len() < n_queries {
            self.stats.resize(n_queries, None);
        }
    }
}

/// Partial-correlation CI test.
#[derive(Clone, Debug)]
pub struct PartialCorrelation {
    /// Kernel policy.
    pub policy: KernelPolicy,
}

impl Default for PartialCorrelation {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialCorrelation {
    /// Default policy.
    #[must_use]
    pub fn new() -> Self {
        Self { policy: KernelPolicy::default_policy() }
    }
}

impl ConditionalIndependence for PartialCorrelation {
    fn test_batch(
        &self,
        request: &CiBatchRequest<'_>,
        workspace: &mut CiWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<CiBatchResult, StatsError> {
        if request.columns.is_empty() {
            return Err(StatsError::Shape { message: "no columns" });
        }
        let n = request.columns[0].len();
        for col in request.columns {
            if col.len() != n {
                return Err(StatsError::Shape { message: "column length mismatch" });
            }
        }
        let nq = request.queries.len();
        workspace.prepare_queries(nq);
        let queries: Vec<ParCorrQuery> = request
            .queries
            .iter()
            .map(|q| ParCorrQuery {
                x: q.x,
                y: q.y,
                z_start: q.z_start,
                z_len: q.z_len,
            })
            .collect();
        let portable = !self.policy.force_scalar;
        partial_correlation_batch(
            request.columns,
            &queries,
            request.z_flat,
            &mut workspace.stats[..nq],
            &mut workspace.parcorr,
            portable,
        );

        let mut results = Vec::with_capacity(nq);
        match request.significance {
            SignificanceMethod::Analytic => {
                for (i, q) in request.queries.iter().enumerate() {
                    let r = workspace.stats[i].ok_or(StatsError::Shape {
                        message: "partial correlation failed",
                    })?;
                    let qcond = q.z_len;
                    let df = (n as f64) - 2.0 - (qcond as f64);
                    if df <= 0.0 {
                        return Err(StatsError::Shape { message: "non-positive residual df" });
                    }
                    let p = analytic_parcorr_pvalue(r, df);
                    results.push(CiResult { statistic: r, p_value: p, df });
                }
            }
            SignificanceMethod::BlockShuffle { replicates, block_size } => {
                if block_size == 0 || replicates == 0 {
                    return Err(StatsError::Shape {
                        message: "block shuffle needs positive block_size and replicates",
                    });
                }
                for (i, q) in request.queries.iter().enumerate() {
                    let observed = workspace.stats[i].ok_or(StatsError::Shape {
                        message: "partial correlation failed",
                    })?;
                    let p = block_shuffle_pvalue(
                        &self.policy,
                        request.columns,
                        *q,
                        request.z_flat,
                        observed,
                        replicates,
                        block_size,
                        workspace,
                        ctx,
                        i as u64,
                    )?;
                    let df = (n as f64) - 2.0 - (q.z_len as f64);
                    results.push(CiResult { statistic: observed, p_value: p, df });
                }
            }
        }
        Ok(CiBatchResult { results })
    }
}

fn analytic_parcorr_pvalue(r: f64, df: f64) -> f64 {
    let r = r.clamp(-1.0 + 1e-15, 1.0 - 1e-15);
    let t = r * (df / (1.0 - r * r)).sqrt();
    2.0 * student_t_sf(t.abs(), df)
}

/// Survival function P(T > t) for Student-t with `df` degrees of freedom.
fn student_t_sf(t: f64, df: f64) -> f64 {
    // Regularized incomplete beta relation:
    // P(|T| > t) = I_{df/(df+t^2)}(df/2, 1/2)
    // Use a continued-fraction incomplete-beta approximation.
    let x = df / (df + t * t);
    0.5 * regularized_incomplete_beta(x, df * 0.5, 0.5)
}

fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    // Continued fraction (Lentz) for Ix(a,b) when x < (a+1)/(a+b+2)
    let ln_beta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    let front = (x.ln() * a + (1.0 - x).ln() * b - ln_beta).exp() / a;
    let mut c = 1.0;
    let mut d = 1.0 - (a + b) * x / (a + 1.0);
    if d.abs() < 1e-30 {
        d = 1e-30;
    }
    d = 1.0 / d;
    let mut f = d;
    for m in 1..200 {
        let m_f = m as f64;
        // even step
        let num = m_f * (b - m_f) * x / ((a + 2.0 * m_f - 1.0) * (a + 2.0 * m_f));
        d = 1.0 + num * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + num / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        f *= d * c;
        // odd step
        let num = -(a + m_f) * (a + b + m_f) * x / ((a + 2.0 * m_f) * (a + 2.0 * m_f + 1.0));
        d = 1.0 + num * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + num / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let delta = d * c;
        f *= delta;
        if (delta - 1.0).abs() < 1e-10 {
            break;
        }
    }
    (front * f).clamp(0.0, 1.0)
}

fn ln_gamma(z: f64) -> f64 {
    // Lanczos approximation
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_654_078_675e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if z < 0.5 {
        return std::f64::consts::PI.ln()
            - (std::f64::consts::PI * z).sin().ln()
            - ln_gamma(1.0 - z);
    }
    let z = z - 1.0;
    let mut x = C[0];
    for (i, &c) in C.iter().enumerate().skip(1) {
        x += c / (z + i as f64);
    }
    let t = z + G + 0.5;
    (2.0 * std::f64::consts::PI).sqrt().ln() + (z + 0.5) * t.ln() - t + x.ln()
}

#[allow(clippy::too_many_arguments)]
fn block_shuffle_pvalue(
    policy: &KernelPolicy,
    columns: &[&[f64]],
    query: CiQuery,
    z_flat: &[usize],
    observed: f64,
    replicates: u32,
    block_size: usize,
    workspace: &mut CiWorkspace,
    ctx: &ExecutionContext,
    stream_salt: u64,
) -> Result<f64, StatsError> {
    let n = columns[0].len();
    let x = columns[query.x];
    let y = columns[query.y];
    let z_idxs = &z_flat[query.z_start..query.z_start + query.z_len];
    if workspace.shuffled.len() < n {
        workspace.shuffled.resize(n, 0.0);
    }
    let n_blocks = n.div_ceil(block_size);
    if workspace.block_perm.len() < n_blocks {
        workspace.block_perm.resize(n_blocks, 0);
    }
    for (i, slot) in workspace.block_perm.iter_mut().enumerate().take(n_blocks) {
        *slot = i;
    }
    let mut rng = ctx.rng.stream(0xC1_u64.wrapping_add(stream_salt));
    let mut extreme = 0u32;
    let abs_obs = observed.abs();
    for _ in 0..replicates {
        // Fisher–Yates on blocks
        for i in (1..n_blocks).rev() {
            let j = (rng.next_u64() as usize) % (i + 1);
            workspace.block_perm.swap(i, j);
        }
        let mut dst = 0usize;
        for &b in workspace.block_perm.iter().take(n_blocks) {
            let start = b * block_size;
            let end = (start + block_size).min(n);
            let len = end - start;
            workspace.shuffled[dst..dst + len].copy_from_slice(&x[start..end]);
            dst += len;
        }
        let mut z_refs: Vec<&[f64]> = z_idxs.iter().map(|&i| columns[i]).collect();
        let r = partial_correlation(
            policy,
            &workspace.shuffled[..n],
            y,
            &z_refs,
            &mut workspace.parcorr,
        )
        .unwrap_or(0.0);
        let _ = &mut z_refs; // keep binding clear for next iter
        if r.abs() >= abs_obs {
            extreme += 1;
        }
    }
    Ok(((extreme as f64) + 1.0) / ((replicates as f64) + 1.0))
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
mod tests {
    use causal_core::ExecutionContext;

    use super::*;

    #[test]
    fn independent_noise_high_pvalue() {
        let n = 300usize;
        let x: Vec<f64> = (0..n).map(|i| (i % 7) as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| (i % 11) as f64).collect();
        let cols: [&[f64]; 2] = [&x, &y];
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 0 }];
        let req = CiBatchRequest {
            columns: &cols,
            queries: &queries,
            z_flat: &[],
            significance: SignificanceMethod::Analytic,
        };
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let out = PartialCorrelation::new().test_batch(&req, &mut ws, &ctx).unwrap();
        assert!(out.results[0].p_value > 0.01, "p={}", out.results[0].p_value);
    }

    #[test]
    fn dependent_low_pvalue() {
        let n = 200usize;
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| 2.0 * i as f64 + 0.01).collect();
        let cols: [&[f64]; 2] = [&x, &y];
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 0 }];
        let req = CiBatchRequest {
            columns: &cols,
            queries: &queries,
            z_flat: &[],
            significance: SignificanceMethod::Analytic,
        };
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(2);
        let out = PartialCorrelation::new().test_batch(&req, &mut ws, &ctx).unwrap();
        assert!(out.results[0].p_value < 1e-6);
        assert!(out.results[0].statistic > 0.99);
    }

    #[test]
    fn block_shuffle_runs() {
        let n = 120usize;
        let x: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let y: Vec<f64> = (0..n).map(|i| (i as f64).cos()).collect();
        let cols: [&[f64]; 2] = [&x, &y];
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 0 }];
        let req = CiBatchRequest {
            columns: &cols,
            queries: &queries,
            z_flat: &[],
            significance: SignificanceMethod::BlockShuffle { replicates: 50, block_size: 10 },
        };
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(3);
        let out = PartialCorrelation::new().test_batch(&req, &mut ws, &ctx).unwrap();
        assert!((0.0..=1.0).contains(&out.results[0].p_value));
    }
}
