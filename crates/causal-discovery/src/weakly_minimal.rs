//! Weakly-minimal separating-set refinement (Gerhardus & Runge 2020 Def. 1).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::too_many_arguments)]

use std::sync::Arc;

use causal_core::{ExecutionContext, Lag, VariableId};
use causal_data::LaggedFrame;
use causal_graph::DenseNodeId;

use crate::engine::{DiscoveryWorkspace, PcmciEngine};
use crate::error::DiscoveryError;
use crate::orientation::OrientationState;

/// Refine `Z` to a weakly-minimal sepset of `X ⫫ Y | ·` relative to known ancestors `ancs`.
///
/// Keeps the ancestor part of `Z` fixed and drops non-ancestors while independence
/// holds (order-independent: among removable nodes that preserve independence, prefer
/// larger |statistic|, recurse).
pub fn make_sepset_weakly_minimal(
    engine: &PcmciEngine,
    frame: &LaggedFrame,
    x: VariableId,
    x_lag: Lag,
    y: VariableId,
    y_lag: Lag,
    z: &[(VariableId, Lag)],
    ancs: &[(VariableId, Lag)],
    workspace: &mut DiscoveryWorkspace,
    ctx: &ExecutionContext,
) -> Result<Vec<(VariableId, Lag)>, DiscoveryError> {
    let ancs_set: std::collections::HashSet<(u32, u32)> =
        ancs.iter().map(|&(v, l)| (v.raw(), l.raw())).collect();

    if z.len() <= 1 || z.iter().all(|&(v, l)| ancs_set.contains(&(v.raw(), l.raw()))) {
        return Ok(z.to_vec());
    }

    let removable: Vec<(VariableId, Lag)> =
        z.iter().copied().filter(|&(v, l)| !ancs_set.contains(&(v.raw(), l.raw()))).collect();

    let mut best_sepsets: Vec<(f64, Vec<(VariableId, Lag)>)> = Vec::new();
    for &drop in &removable {
        let z_a: Vec<_> = z.iter().copied().filter(|&p| p != drop).collect();
        let (stat, p) = engine.ci_statistic(frame, x, x_lag, y, y_lag, &z_a, workspace, ctx)?;
        if p > engine.constraints.alpha {
            best_sepsets.push((stat.abs(), z_a));
        }
    }

    if best_sepsets.is_empty() {
        return Ok(z.to_vec());
    }

    best_sepsets.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let top = best_sepsets[0].0;
    if let Some((_, cand)) = best_sepsets.iter().find(|(s, _)| (*s - top).abs() <= 1e-12) {
        return make_sepset_weakly_minimal(
            engine, frame, x, x_lag, y, y_lag, cand, ancs, workspace, ctx,
        );
    }
    Ok(z.to_vec())
}

/// Store a weakly-minimal sepset on [`OrientationState`].
pub fn store_weakly_minimal_sepset(
    state: &mut OrientationState,
    a: DenseNodeId,
    b: DenseNodeId,
    sep_nodes: Arc<[DenseNodeId]>,
) {
    state.set_sepset(a, b, sep_nodes);
    state.mark_weakly_minimal(a, b);
}
