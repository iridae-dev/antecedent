//! One finite unfolding for a temporal query window.
//!
//! History grows by walking template ancestry. The static DAG is materialised
//! once, at the history that closes the truncated boundary.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{TemporalIndexer, TemporalNodeKey};
use antecedent_graph::algo::implicit_closure;
#[cfg(debug_assertions)]
use antecedent_graph::{BitSet, DenseNodeId, GraphWorkspace};
use antecedent_graph::{TemporalDag, UnfoldedTemporalGraph};

use crate::error::IdentificationError;

/// The unfolded window and the dense ids of the seeds it was grown for.
pub(crate) struct GrownWindow {
    /// History depth of [`Self::unfolded`].
    pub history: u32,
    /// Static DAG for that history. Dense ids match [`UnfoldedTemporalGraph::indexer`].
    pub unfolded: UnfoldedTemporalGraph,
    /// Dense id of each seed, in seed order.
    pub dense: Vec<u32>,
}

/// Grow history until ancestral closure of `seeds` no longer sits on the
/// truncated boundary, then unfold that window once.
///
/// `history_cap` stops growth. Reaching it while the boundary is still open
/// is [`IdentificationError::NotCertified`], with the same refusal
/// `identify_temporal` used when each candidate history was unfolded from scratch.
///
/// # Errors
///
/// Indexer or unfold failures, a seed outside the window, or the history-cap refusal.
#[must_use = "the grown window is the only unfolding of this query"]
pub(crate) fn grow_window(
    template: &TemporalDag,
    seeds: &[TemporalNodeKey],
    variable_count: u32,
    horizon: u32,
    base_history: u32,
    history_cap: u32,
    chain_cap: u32,
) -> Result<GrownWindow, IdentificationError> {
    let edges = StationaryEdges::from_template(template, variable_count);
    let mut depth = base_history;
    let history = loop {
        let indexer = TemporalIndexer::new(variable_count, depth, horizon)
            .map_err(|e| IdentificationError::construction(e))?;
        for seed in seeds {
            indexer
                .dense_id(*seed)
                .map_err(|_| IdentificationError::UnknownVariable { id: seed.variable })?;
        }
        if !edges.ancestry_touches_boundary(seeds, depth, horizon) {
            break depth;
        }
        if depth >= history_cap {
            return Err(history_cap_refusal(history_cap, chain_cap));
        }
        depth += 1;
    };

    let indexer = TemporalIndexer::new(variable_count, history, horizon)
        .map_err(|e| IdentificationError::construction(e))?;
    let unfolded = template.unfold(indexer).map_err(|e| IdentificationError::construction(e))?;
    let mut dense = Vec::with_capacity(seeds.len());
    for seed in seeds {
        dense.push(
            unfolded
                .indexer
                .dense_id(*seed)
                .map_err(|_| IdentificationError::UnknownVariable { id: seed.variable })?,
        );
    }
    #[cfg(debug_assertions)]
    assert_stopped_at_fixed_point(
        template,
        &edges,
        seeds,
        &unfolded,
        &dense,
        base_history,
    );
    Ok(GrownWindow { history, unfolded, dense })
}

/// Whether `from` reaches `to` in the stationary unfolding, including `from == to`.
///
/// `history` and `horizon` are the same window [`TemporalIndexer::new`] would build.
/// Callers still resolve dense ids themselves so an offset outside that window
/// keeps the indexer's error.
#[must_use]
pub(crate) fn reaches_in_window(
    template: &TemporalDag,
    from: TemporalNodeKey,
    to: TemporalNodeKey,
    variable_count: u32,
    history: u32,
    horizon: u32,
) -> bool {
    if from == to {
        return true;
    }
    let edges = StationaryEdges::from_template(template, variable_count);
    let min_off = -i64::from(history);
    let max_off = i64::from(horizon) - 1;
    let target = (to.variable.raw(), i64::from(to.offset));
    implicit_closure([(from.variable.raw(), i64::from(from.offset))], |&(var, offset)| {
        edges.children(var, offset, min_off, max_off)
    })
    .contains(&target)
}

/// The refusal when window growth stops at `history_cap` with ancestry still
/// crossing the truncated boundary.
///
/// A confounder chain that repeats no variable reaches at most `chain_cap`
/// slices back (`variable_count * template_max_lag + |min_offset|`). Reaching
/// that depth therefore means an ancestor lies on a lagged cycle, such as an
/// autoregressive edge `x(t-1) -> x(t)`, whose ancestry never ends: no finite
/// window certifies it, and a larger `max_history_lag` cannot help. Only a
/// caller-set `max_history_lag` below `chain_cap` can stop growth early, and
/// then raising it is the remedy.
fn history_cap_refusal(history_cap: u32, chain_cap: u32) -> IdentificationError {
    let message = if history_cap < chain_cap {
        "temporal unfolding reached the query's max_history_lag while confounder ancestry \
         still crossed the truncated boundary; cannot certify backdoor identification over \
         the finite window (raise max_history_lag)"
    } else {
        "temporal unfolding reached its history cap while confounder ancestry still crossed \
         the truncated boundary: an ancestor of the treatment or outcome lies on a lagged \
         cycle (for example an autoregressive edge x(t-1) -> x(t)), so no finite window \
         certifies backdoor identification; review whether that lagged edge belongs in the \
         graph"
    };
    IdentificationError::NotCertified { message }
}

/// Whether any ancestor of `nodes` in the unfolded graph has a template parent
/// that the truncation to `history` slices cut off.
#[cfg(debug_assertions)]
fn ancestry_touches_unfolded(
    unfolded: &UnfoldedTemporalGraph,
    edges: &StationaryEdges,
    nodes: &[u32],
    history: u32,
) -> bool {
    let dag = &unfolded.dag;
    let mut ancestors = BitSet::with_len(dag.node_count());
    let mut gws = GraphWorkspace::default();
    let dense: Vec<DenseNodeId> = nodes.iter().copied().map(DenseNodeId::from_raw).collect();
    dag.ancestors_of(&dense, &mut ancestors, &mut gws);
    let min_off = -i64::from(history);
    for i in 0..dag.node_count() {
        let id = DenseNodeId::from_index(i);
        if !ancestors.contains(id) {
            continue;
        }
        let Ok(key) = unfolded.indexer.key_of(id.raw()) else {
            continue;
        };
        if edges.has_cut_parent(key.variable.raw(), i64::from(key.offset), min_off) {
            return true;
        }
    }
    false
}

/// Template edges replicated by the same shift rule as [`TemporalDag::unfold`].
struct StationaryEdges {
    /// `incoming[child_var] = (parent_var, parent_template_offset, child_template_offset)`.
    incoming: Vec<Vec<(u32, i32, i32)>>,
    /// `outgoing[parent_var] = (child_var, parent_template_offset, child_template_offset)`.
    outgoing: Vec<Vec<(u32, i32, i32)>>,
}

impl StationaryEdges {
    fn from_template(template: &TemporalDag, variable_count: u32) -> Self {
        let n = variable_count as usize;
        let mut incoming = vec![Vec::new(); n];
        let mut outgoing = vec![Vec::new(); n];
        for edge in template.edges() {
            let Some((from, to)) = edge.parent_child() else {
                continue;
            };
            let (Some(from_key), Some(to_key)) =
                (template.temporal_key(from), template.temporal_key(to))
            else {
                continue;
            };
            let parent = from_key.variable.raw() as usize;
            let child = to_key.variable.raw() as usize;
            if parent >= n || child >= n {
                continue;
            }
            incoming[child].push((from_key.variable.raw(), from_key.offset, to_key.offset));
            outgoing[parent].push((to_key.variable.raw(), from_key.offset, to_key.offset));
        }
        Self { incoming, outgoing }
    }

    fn ancestry_touches_boundary(
        &self,
        seeds: &[TemporalNodeKey],
        history: u32,
        horizon: u32,
    ) -> bool {
        let min_off = -i64::from(history);
        let max_off = i64::from(horizon) - 1;
        let reached = implicit_closure(
            seeds.iter().map(|seed| (seed.variable.raw(), i64::from(seed.offset))),
            |&(var, offset)| self.parents(var, offset, min_off, max_off),
        );
        reached.into_iter().any(|(var, offset)| self.has_cut_parent(var, offset, min_off))
    }

    /// Whether the node `(var, offset)` has a template parent earlier than `min_off`, which
    /// the window therefore omits. Any such node, at any slice, leaves the window's ancestry
    /// open: with template lag `k`, nodes within `k` slices of the boundary lose parents,
    /// not just the nodes on the boundary slice.
    fn has_cut_parent(&self, var: u32, offset: i64, min_off: i64) -> bool {
        let Some(edges) = self.incoming.get(var as usize) else {
            return false;
        };
        edges.iter().any(|&(parent_var, parent_off, child_off)| {
            let Some(delta) = offset.checked_sub(i64::from(child_off)) else {
                return false;
            };
            let Ok(delta) = i32::try_from(delta) else {
                return false;
            };
            let parent = i64::from(parent_off.saturating_add(delta));
            parent < min_off && !(parent_var == var && parent == offset)
        })
    }

    fn parents(&self, var: u32, offset: i64, min_off: i64, max_off: i64) -> Vec<(u32, i64)> {
        Self::shift(&self.incoming, var, offset, min_off, max_off, true)
    }

    fn children(&self, var: u32, offset: i64, min_off: i64, max_off: i64) -> Vec<(u32, i64)> {
        Self::shift(&self.outgoing, var, offset, min_off, max_off, false)
    }

    /// Replicate template edges onto one concrete endpoint.
    ///
    /// `incoming` anchors the shift on the child template offset and emits parents.
    /// `outgoing` anchors it on the parent template offset and emits children.
    /// A shift that saturates, leaves the window, or lands on the same node is
    /// dropped, matching `TemporalDag::unfold`.
    fn shift(
        table: &[Vec<(u32, i32, i32)>],
        var: u32,
        offset: i64,
        min_off: i64,
        max_off: i64,
        incoming: bool,
    ) -> Vec<(u32, i64)> {
        let Some(edges) = table.get(var as usize) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for &(other_var, parent_off, child_off) in edges {
            let anchor = if incoming { child_off } else { parent_off };
            let Some(delta) = offset.checked_sub(i64::from(anchor)) else {
                continue;
            };
            let Ok(delta) = i32::try_from(delta) else {
                continue;
            };
            let parent_concrete = i64::from(parent_off.saturating_add(delta));
            let child_concrete = i64::from(child_off.saturating_add(delta));
            let (anchored, other) = if incoming {
                (child_concrete, parent_concrete)
            } else {
                (parent_concrete, child_concrete)
            };
            if anchored != offset || other < min_off || other > max_off {
                continue;
            }
            if other_var == var && other == offset {
                continue;
            }
            out.push((other_var, other));
        }
        out
    }
}

#[cfg(debug_assertions)]
fn assert_stopped_at_fixed_point(
    template: &TemporalDag,
    edges: &StationaryEdges,
    seeds: &[TemporalNodeKey],
    unfolded: &UnfoldedTemporalGraph,
    dense: &[u32],
    base_history: u32,
) {
    let history = unfolded.indexer.history();
    assert!(
        !ancestry_touches_unfolded(unfolded, edges, dense, history),
        "grow_window certified a window whose ancestry still meets the truncated boundary"
    );
    if history == base_history {
        return;
    }
    let indexer = TemporalIndexer::new(
        unfolded.indexer.variable_count(),
        history - 1,
        unfolded.indexer.horizon(),
    )
    .expect("previous history indexed");
    let previous = template.unfold(indexer).expect("previous history unfolded");
    let previous_dense: Vec<u32> = seeds
        .iter()
        .map(|seed| previous.indexer.dense_id(*seed).expect("seed stays in the previous window"))
        .collect();
    assert!(
        ancestry_touches_unfolded(&previous, edges, &previous_dense, history - 1),
        "grow_window grew past a history that already closed the truncated boundary"
    );
}
