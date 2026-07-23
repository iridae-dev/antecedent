//! Partial-correlation conditional independence test.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::trivially_copy_pass_by_ref,
    clippy::unused_self
)]

use antecedent_core::{ExecutionContext, KernelPolicy};
use antecedent_kernels::{
    KernelImpl, ParCorrMode, ParCorrQuery, partial_correlation_batch, select_impl,
};

use super::analytic::{analytic_parcorr_ci, analytic_parcorr_pvalue};
use super::block_shuffle::block_shuffle_pvalue;
use super::types::{
    CiBatchRequest, CiBatchResult, CiQuery, CiResult, CiWorkspace, ConditionalIndependenceTest,
    ConfidenceMethod, PreparedCiTest, SignificanceMethod,
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
            workspace,
            ctx,
            0,
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
        workspace: &mut CiWorkspace,
        ctx: &ExecutionContext,
        stream_id: u64,
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
                let p = block_shuffle_pvalue(
                    &ctx.kernel_policy,
                    columns,
                    query,
                    z_flat,
                    r,
                    replicates,
                    block_size,
                    workspace,
                    ctx,
                    stream_id,
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
                workspace,
                ctx,
                i as u64,
            )?);
        }
        Ok(CiBatchResult { results })
    }
}
