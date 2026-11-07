// SPDX-License-Identifier: MIT OR Apache-2.0
//! Graph-prior and lag-aware edge-constraint helpers.

use super::{
    AdjacencyMask, DiscoveryConstraints, GraphPrior, GraphPriorTable, Lag, LaggedLink, VariableId,
};

/// Contemporaneous `LaggedLink` for dense indices into `variables`.
#[must_use]
pub(crate) fn static_link(variables: &[VariableId], from: usize, to: usize) -> LaggedLink {
    LaggedLink {
        source: variables[from],
        source_lag: Lag::CONTEMPORANEOUS,
        target: variables[to],
        target_lag: Lag::CONTEMPORANEOUS,
    }
}

/// Whether constraints forbid directed edge `from → to`.
#[must_use]
#[allow(
    dead_code,
    reason = "single-edge predicate; mask scans use ConstraintIndex, which matches this boolean"
)]
pub(crate) fn edge_forbidden(
    constraints: &DiscoveryConstraints,
    variables: &[VariableId],
    from: usize,
    to: usize,
) -> bool {
    let link = static_link(variables, from, to);
    // `is_forbidden` already applies `tier_forbids`.
    constraints.is_forbidden(link)
}

/// Lagged link `from` at `lag` → `to` contemporaneous.
#[must_use]
pub fn lagged_link(variables: &[VariableId], from: usize, lag: u32, to: usize) -> LaggedLink {
    LaggedLink {
        source: variables[from],
        source_lag: Lag::from_raw(lag),
        target: variables[to],
        target_lag: Lag::CONTEMPORANEOUS,
    }
}

/// Whether constraints forbid the lagged edge `from_{t−lag} → to_t`.
///
/// [`edge_forbidden`] pins both endpoints to [`Lag::CONTEMPORANEOUS`] via [`static_link`],
/// so it structurally cannot express a constraint on a lag > 0 edge. `DiscoveryConstraints`
/// stores `LaggedLink`s precisely so lag-specific constraints can be written, and callers
/// that score lagged structure need this variant to honor them.
#[must_use]
#[allow(
    dead_code,
    reason = "single-edge predicate; lag-mask scans use ConstraintIndex, which matches this boolean"
)]
pub(crate) fn lagged_edge_forbidden(
    constraints: &DiscoveryConstraints,
    variables: &[VariableId],
    from: usize,
    lag: u32,
    to: usize,
) -> bool {
    let link = lagged_link(variables, from, lag, to);
    // `is_forbidden` already applies `tier_forbids`.
    constraints.is_forbidden(link)
}

/// Whether constraints require the lagged edge `from_{t−lag} → to_t`.
#[must_use]
pub(crate) fn lagged_edge_required(
    constraints: &DiscoveryConstraints,
    variables: &[VariableId],
    from: usize,
    lag: u32,
    to: usize,
) -> bool {
    constraints.is_required(lagged_link(variables, from, lag, to))
}

/// Whether constraints require directed edge `from → to`.
#[must_use]
pub(crate) fn edge_required(
    constraints: &DiscoveryConstraints,
    variables: &[VariableId],
    from: usize,
    to: usize,
) -> bool {
    constraints.is_required(static_link(variables, from, to))
}

/// Log prior for a constraint-valid DAG mask (`−∞` encoded as `None` if invalid).
#[must_use]
pub(crate) fn log_prior_mask(
    mask: &AdjacencyMask,
    n: usize,
    prior: &GraphPrior,
    variables: &[VariableId],
) -> Option<f64> {
    GraphPriorTable::new(n, prior, variables).log_prior(mask)
}
