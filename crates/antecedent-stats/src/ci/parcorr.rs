//! Partial-correlation conditional independence test.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::trivially_copy_pass_by_ref,
    clippy::unused_self
)]

use antecedent_core::{ExecutionContext, KernelPolicy, StreamDomain};
use antecedent_kernels::{
    KernelImpl, ParCorrMode, ParCorrQuery, partial_correlation_batch, select_impl,
};

use super::analytic::{analytic_parcorr_ci, analytic_parcorr_pvalue};
use super::block_shuffle::{query_stream_salt, residual_null_pvalue};
use super::residualize::ZDesign;
use super::types::{
    CiBatchRequest, CiBatchResult, CiQuery, CiResult, CiWorkspace, ConditionalIndependenceTest,
    ConfidenceMethod, PreparedCiTest, SignificanceMethod, permutation_min_p,
};
use crate::error::StatsError;

/// Map [`KernelPolicy`] to the `ParCorr` batch mode (no arch-SIMD path).
#[must_use]
pub(crate) fn parcorr_mode(policy: &KernelPolicy) -> ParCorrMode {
    match select_impl(policy) {
        KernelImpl::Scalar => ParCorrMode::Native,
        KernelImpl::PortableOptimized | KernelImpl::ArchSimd => ParCorrMode::Portable,
    }
}

/// Partial-correlation CI test.
///
/// Kernel path selection comes from [`ExecutionContext::kernel_policy`] at call time
///, not from state on this type.
#[derive(Clone, Debug, Default)]
pub struct PartialCorrelation;

impl PartialCorrelation {
    /// Construct.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Single CI query without allocating request/result vectors.
    ///
    /// `columns[0]` is X, `columns[1]` is Y, and `z_flat` indexes conditioning
    /// columns into `columns` (typically `2..`).
    ///
    /// # Errors
    ///
    /// Shape / numerical failures.
    pub fn test_one(
        &self,
        columns: &[&[f64]],
        z_flat: &[usize],
        significance: SignificanceMethod,
        workspace: &mut CiWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<CiResult, StatsError> {
        if columns.len() < 2 {
            return Err(StatsError::Shape { message: "need X and Y columns" });
        }
        let n = columns[0].len();
        for col in columns {
            if col.len() != n {
                return Err(StatsError::Shape { message: "column length mismatch" });
            }
        }
        workspace.prepare_queries(1);
        let query = ParCorrQuery { x: 0, y: 1, z_start: 0, z_len: z_flat.len() };
        let mode = parcorr_mode(&ctx.kernel_policy);
        partial_correlation_batch(
            columns,
            &[query],
            z_flat,
            &mut workspace.stats[..1],
            &mut workspace.parcorr,
            mode,
        );
        let r = workspace.stats[0]
            .ok_or(StatsError::Shape { message: "partial correlation failed" })?;
        let ci_query = CiQuery { x: 0, y: 1, z_start: 0, z_len: z_flat.len() };
        self.interpret(
            r,
            n,
            ci_query,
            significance,
            ConfidenceMethod::default(),
            columns,
            z_flat,
            ctx,
        )
    }

    /// Map a partial-correlation statistic to a [`CiResult`] under `significance`.
    #[allow(clippy::too_many_arguments)]
    fn interpret(
        &self,
        r: f64,
        n: usize,
        query: CiQuery,
        significance: SignificanceMethod,
        confidence: ConfidenceMethod,
        columns: &[&[f64]],
        z_flat: &[usize],
        ctx: &ExecutionContext,
    ) -> Result<CiResult, StatsError> {
        let df = (n as f64) - 2.0 - (query.z_len as f64);
        match significance {
            SignificanceMethod::Analytic => {
                if df <= 0.0 {
                    return Err(StatsError::Shape { message: "non-positive residual df" });
                }
                let p = analytic_parcorr_pvalue(r, df);
                let ci = match confidence {
                    ConfidenceMethod::None => None,
                    ConfidenceMethod::Analytic { level } => Some(analytic_parcorr_ci(r, df, level)),
                };
                Ok(CiResult { statistic: r, p_value: p, df, ci })
            }
            SignificanceMethod::BlockShuffle { replicates, block_size } => {
                if block_size == 0 || replicates == 0 {
                    return Err(StatsError::Shape {
                        message: "block shuffle needs positive block_size and replicates",
                    });
                }
                // Residualise X and Y on Z once, then block-permute the X residual: the same
                // null for every partial-correlation variant, valid with a non-empty Z.
                let z = &z_flat[query.z_start..query.z_start + query.z_len];
                let design = ZDesign::fit(columns, z, None, n)?;
                let rx = design.residuals(columns[query.x])?;
                let ry = design.residuals(columns[query.y])?;
                let mut rng = ctx.rng.stream_for(
                    StreamDomain::StatsCi,
                    0xC1_u64 ^ query_stream_salt(&[query.x], &[query.y], z),
                );
                let p = residual_null_pvalue(
                    &ctx.kernel_policy,
                    &rx,
                    &ry,
                    None,
                    replicates,
                    block_size,
                    &mut rng,
                )?;
                Ok(CiResult { statistic: r, p_value: p, df, ci: None })
            }
        }
    }
}

impl ConditionalIndependenceTest for PartialCorrelation {
    fn test_batch(
        &self,
        prepared: &PreparedCiTest,
        request: &CiBatchRequest<'_>,
        workspace: &mut CiWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<CiBatchResult, StatsError> {
        prepared.ensure_compatible(request)?;
        let request = &prepared.bind_request(request);
        let n = request.nrows()?;
        let nq = request.queries.len();
        workspace.prepare_queries(nq);
        let queries: Vec<ParCorrQuery> = request
            .queries
            .iter()
            .map(|q| ParCorrQuery { x: q.x, y: q.y, z_start: q.z_start, z_len: q.z_len })
            .collect();
        let mode = parcorr_mode(&ctx.kernel_policy);
        partial_correlation_batch(
            request.columns,
            &queries,
            request.z_flat,
            &mut workspace.stats[..nq],
            &mut workspace.parcorr,
            mode,
        );

        let mut results = Vec::with_capacity(nq);
        for (i, q) in request.queries.iter().enumerate() {
            let r = workspace.stats[i]
                .ok_or(StatsError::Shape { message: "partial correlation failed" })?;
            results.push(self.interpret(
                r,
                n,
                *q,
                request.significance,
                request.confidence,
                request.columns,
                request.z_flat,
                ctx,
            )?);
        }
        Ok(CiBatchResult { results })
    }

    fn min_attainable_p(&self, significance: SignificanceMethod) -> f64 {
        block_shuffle_min_p(significance)
    }
}

/// Smallest attainable p-value of the block-permutation null (`0` under the analytic path).
pub(crate) fn block_shuffle_min_p(significance: SignificanceMethod) -> f64 {
    match significance {
        SignificanceMethod::Analytic => 0.0,
        SignificanceMethod::BlockShuffle { replicates, .. } => {
            permutation_min_p(replicates as usize)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analytic_result_is_invariant_to_conditioning_offset() {
        let n = 80usize;
        let z: Vec<_> = (0..n).map(|i| (i as f64 * 0.19).sin() - 2.0).collect();
        let z_shifted: Vec<_> = z.iter().map(|v| v + 1_000.0).collect();
        let x: Vec<_> = (0..n).map(|i| 1.2 * z[i] + (i as f64 * 0.37).cos()).collect();
        let y: Vec<_> = (0..n).map(|i| -0.8 * z[i] + (i as f64 * 0.29).sin()).collect();
        let test = PartialCorrelation::new();
        let ctx = ExecutionContext::for_tests(17);
        let mut workspace = CiWorkspace::default();
        let base = test
            .test_one(&[&x, &y, &z], &[2], SignificanceMethod::Analytic, &mut workspace, &ctx)
            .unwrap();
        let shifted = test
            .test_one(
                &[&x, &y, &z_shifted],
                &[2],
                SignificanceMethod::Analytic,
                &mut workspace,
                &ctx,
            )
            .unwrap();
        assert!((base.statistic - shifted.statistic).abs() <= 1e-12);
        assert!((base.p_value - shifted.p_value).abs() <= 1e-12);
    }

    fn simple_residuals(t: &[f64], z: &[f64]) -> Vec<f64> {
        let n = t.len() as f64;
        let mt = t.iter().sum::<f64>() / n;
        let mz = z.iter().sum::<f64>() / n;
        let szz: f64 = z.iter().map(|v| (v - mz) * (v - mz)).sum();
        let szt: f64 = z.iter().zip(t).map(|(a, b)| (a - mz) * (b - mt)).sum();
        t.iter().zip(z).map(|(tv, zv)| (tv - mt) - (szt / szz) * (zv - mz)).collect()
    }

    fn corr(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len() as f64;
        let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
        let sab: f64 = a.iter().zip(b).map(|(p, q)| (p - ma) * (q - mb)).sum();
        let saa: f64 = a.iter().map(|p| (p - ma) * (p - ma)).sum();
        let sbb: f64 = b.iter().map(|q| (q - mb) * (q - mb)).sum();
        sab / (saa * sbb).sqrt()
    }

    /// A conditioned block-preserving request used to be refused ("with Z the permutation of X
    /// is not a valid null"). The null block-permutes the X residual after removing Z, so its
    /// p-value must equal the exact fraction of block arrangements of that residual whose
    /// `|r|` reaches the observed one.
    #[test]
    fn block_shuffle_with_conditioning_set_permutes_residuals() {
        let z = [0.3, -1.1, 0.8, 1.9, -0.7, 0.2];
        let x = [3.1, -2.9, 2.6, 6.1, -2.0, 1.1];
        let y = [0.5, -1.3, 1.0, 1.6, -0.2, 1.4];
        let rx = simple_residuals(&x, &z);
        let ry = simple_residuals(&y, &z);
        let observed = corr(&rx, &ry).abs();
        let mut extreme = 0usize;
        for order in [[0usize, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]] {
            let mut px: Vec<f64> = Vec::new();
            for b in order {
                px.extend_from_slice(&rx[2 * b..2 * b + 2]);
            }
            if corr(&px, &ry).abs() >= observed * (1.0 - 1e-12) {
                extreme += 1;
            }
        }
        let exact = extreme as f64 / 6.0;
        let cols: [&[f64]; 3] = [&x, &y, &z];
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(21);
        let out = PartialCorrelation::new()
            .test_one(
                &cols,
                &[2],
                SignificanceMethod::BlockShuffle { replicates: 6000, block_size: 2 },
                &mut ws,
                &ctx,
            )
            .expect("block shuffle with a conditioning set is supported");
        assert!((out.p_value - exact).abs() < 0.03, "p={} exact={exact}", out.p_value);
    }

    /// The permutation stream is keyed by the query, not its position in the batch.
    #[test]
    fn block_shuffle_p_value_is_independent_of_batch_position() {
        let n = 40usize;
        let x: Vec<f64> =
            (0..n).map(|i| ((i * 7 + 3) % 11) as f64 + (i as f64 * 0.31).sin()).collect();
        let y: Vec<f64> =
            (0..n).map(|i| ((i * 5 + 1) % 13) as f64 + (i as f64 * 0.17).cos()).collect();
        let w: Vec<f64> = (0..n).map(|i| ((i * 3 + 2) % 9) as f64).collect();
        let cols: [&[f64]; 3] = [&x, &y, &w];
        let target = CiQuery { x: 0, y: 1, z_start: 0, z_len: 0 };
        let other = CiQuery { x: 0, y: 2, z_start: 0, z_len: 0 };
        let sig = SignificanceMethod::BlockShuffle { replicates: 99, block_size: 4 };
        let run = |queries: &[CiQuery]| {
            let req = CiBatchRequest {
                columns: &cols,
                queries,
                z_flat: &[],
                significance: sig,
                confidence: ConfidenceMethod::None,
            };
            let mut ws = CiWorkspace::default();
            let ctx = ExecutionContext::for_tests(33);
            PartialCorrelation::new().test_batch_adhoc(&req, &mut ws, &ctx).unwrap()
        };
        let alone = run(&[target]);
        let second = run(&[other, target]);
        assert_eq!(alone.results[0].p_value.to_bits(), second.results[1].p_value.to_bits());
    }

    #[test]
    fn min_attainable_p_reports_permutation_resolution() {
        let pc = PartialCorrelation::new();
        assert_eq!(pc.min_attainable_p(SignificanceMethod::Analytic), 0.0);
        let sig = SignificanceMethod::BlockShuffle { replicates: 99, block_size: 1 };
        assert!((pc.min_attainable_p(sig) - 0.01).abs() < 1e-15);
    }
}
