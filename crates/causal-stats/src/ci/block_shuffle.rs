//! Block-shuffle nulls for `ParCorr` CI.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::too_many_arguments,
    clippy::trivially_copy_pass_by_ref
)]

use causal_core::{ExecutionContext, KernelPolicy};
use causal_kernels::partial_correlation;

use super::types::{CiQuery, CiWorkspace};
use crate::error::StatsError;

pub(crate) fn block_shuffle_pvalue(
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
    let z_refs: Vec<&[f64]> = z_idxs.iter().map(|&i| columns[i]).collect();
    for _ in 0..replicates {
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
        let r = partial_correlation(
            policy,
            &workspace.shuffled[..n],
            y,
            &z_refs,
            &mut workspace.parcorr,
        )
        .ok_or(StatsError::Shape {
            message: "block-shuffle replicate: partial correlation failed",
        })?;
        if r.abs() >= abs_obs {
            extreme += 1;
        }
    }
    Ok(((extreme as f64) + 1.0) / ((replicates as f64) + 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use causal_core::ExecutionContext;

    #[test]
    fn replicate_failure_propagates() {
        // n < 3 → partial_correlation returns None; must not be treated as r=0.
        let x = [1.0, 2.0];
        let y = [3.0, 4.0];
        let cols: [&[f64]; 2] = [&x, &y];
        let query = CiQuery { x: 0, y: 1, z_start: 0, z_len: 0 };
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let err = block_shuffle_pvalue(
            &ctx.kernel_policy,
            &cols,
            query,
            &[],
            0.5,
            5,
            1,
            &mut ws,
            &ctx,
            0,
        );
        assert!(err.is_err());
    }
}
