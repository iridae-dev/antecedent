//! Partial-correlation conditional independence test.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_lossless
)]

use causal_core::{ExecutionContext, KernelPolicy};
use causal_kernels::{ParCorrQuery, partial_correlation_batch};

use super::analytic::analytic_parcorr_pvalue;
use super::block_shuffle::block_shuffle_pvalue;
use super::types::{
    CiBatchRequest, CiBatchResult, CiResult, CiWorkspace, ConditionalIndependence,
    SignificanceMethod,
};
use crate::error::StatsError;

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
                    );
                    let df = (n as f64) - 2.0 - (q.z_len as f64);
                    results.push(CiResult { statistic: observed, p_value: p, df });
                }
            }
        }
        Ok(CiBatchResult { results })
    }
}
