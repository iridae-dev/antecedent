//! Weakly-minimal separating-set refinement (Gerhardus & Runge 2020 Def. 1).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::ops::ControlFlow;
use std::sync::Arc;

use antecedent_core::{ExecutionContext, Lag, VariableId};
use antecedent_data::LaggedFrame;
use antecedent_graph::DenseNodeId;

use crate::engine::{CiMemo, CiProbe, DiscoveryWorkspace, PcmciEngine};
use crate::error::DiscoveryError;
use crate::orientation::OrientationState;
use crate::result::LaggedLink;

/// Refine `Z` to a weakly-minimal sepset of `X ⫫ Y | ·` relative to known ancestors `ancs`.
///
/// Keeps the ancestor part of `Z` fixed and drops non-ancestors while independence
/// holds (order-independent: among removable nodes that preserve independence, prefer
/// larger |statistic|, recurse). `link` names the pair `X ⫫ Y` by its lagged endpoints.
pub fn make_sepset_weakly_minimal(
    engine: &PcmciEngine,
    frame: &LaggedFrame,
    link: LaggedLink,
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
    let sets = removable
        .iter()
        .map(|&drop| Ok(z.iter().copied().filter(|&p| p != drop).collect::<Vec<_>>()));
    let alpha = engine.constraints.alpha;
    engine.scan_ci_sets(
        CiProbe { frame, link },
        sets,
        &mut CiMemo::default(),
        workspace,
        ctx,
        |z_a, stat, p| {
            if p > alpha {
                best_sepsets.push((stat.abs(), z_a));
            }
            ControlFlow::<()>::Continue(())
        },
    )?;

    if best_sepsets.is_empty() {
        return Ok(z.to_vec());
    }

    best_sepsets.sort_by(|a, b| b.0.total_cmp(&a.0));
    let top = best_sepsets[0].0;
    if let Some((_, cand)) = best_sepsets.iter().find(|(s, _)| (*s - top).abs() <= 1e-12) {
        return make_sepset_weakly_minimal(engine, frame, link, cand, ancs, workspace, ctx);
    }
    Ok(z.to_vec())
}

/// Store a weakly-minimal sepset on [`OrientationState`].
pub(crate) fn store_weakly_minimal_sepset(
    state: &mut OrientationState,
    a: DenseNodeId,
    b: DenseNodeId,
    sep_nodes: Arc<[DenseNodeId]>,
) {
    state.set_sepset(a, b, sep_nodes);
    state.mark_weakly_minimal(a, b);
}
