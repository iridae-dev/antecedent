//! The declared unrolled point graph of a panel program, as a Python declaration.
// SPDX-License-Identifier: MIT OR Apache-2.0
use antecedent::execution::{AdmissionRefusal, Retention};
use antecedent::identify::IdentificationError;
use antecedent::identify::panel_unrolled::{PanelNode, UnrolledPanelGraph, UnrolledPanelSpec};
use antecedent::prelude::ExecutionContext;
use pyo3::PyResult;
use serde::Deserialize;

/// A node of the unrolled graph: `{"context": i}`, `{"channel": {"channel": c, "period": t}}`
/// or `{"outcome": t}`.
#[derive(Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
enum Node {
    Context(usize),
    Channel { channel: usize, period: usize },
    Outcome(usize),
}

impl Node {
    const fn native(&self) -> PanelNode {
        match *self {
            Self::Context(index) => PanelNode::Context(index),
            Self::Channel { channel, period } => PanelNode::Channel { channel, period },
            Self::Outcome(period) => PanelNode::Outcome(period),
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Declaration {
    periods: usize,
    channels: usize,
    #[serde(default)]
    contexts: usize,
    #[serde(default)]
    carryover_reach: usize,
    #[serde(default)]
    outcome_persistence: bool,
    #[serde(default)]
    channel_persistence: bool,
    #[serde(default)]
    history_feedback: bool,
    #[serde(default)]
    latent_confounding: Vec<(Node, Node)>,
    outcome_periods: Vec<usize>,
}

/// Most nodes (contexts plus, per period, the channels and the outcome) a declared unrolled
/// graph may have. The graph is dense and the general ID run grows faster than its size, so
/// a larger declaration is a resource refusal, never an allocation sized by a few digits.
pub(super) const MAX_UNROLLED_NODES: usize = 1024;

/// The unrolled graph a declaration names, with the outcome periods the verdict is for and
/// the admission that funds it (hold it while the graph lives).
///
/// # Errors
/// A malformed declaration, or a graph that cannot be built (no periods or channels, or a
/// latent edge naming a node outside the program or joining a node to itself); a graph
/// above [`MAX_UNROLLED_NODES`] or that the caller's admission cannot fund
/// (`CausalResourceError`); a cancelled run (`CausalCancelledError`).
pub(super) fn graph_from_json(
    text: &str,
    ctx: &ExecutionContext,
) -> PyResult<(UnrolledPanelGraph, Vec<usize>, Retention)> {
    let declaration: Declaration =
        serde_json::from_str(text).map_err(|error| crate::value_err(error.to_string()))?;
    let nodes = declaration
        .channels
        .checked_add(1)
        .and_then(|per_period| declaration.periods.checked_mul(per_period))
        .and_then(|inner| inner.checked_add(declaration.contexts))
        .filter(|&nodes| nodes <= MAX_UNROLLED_NODES)
        .ok_or_else(|| {
            crate::CausalResourceError::new_err(format!(
                "the declared unrolled graph exceeds {MAX_UNROLLED_NODES} nodes"
            ))
        })?;
    // Dense adjacency of both edge kinds, charged before the graph is built.
    let bytes = u64::try_from(nodes.saturating_mul(nodes).saturating_mul(2)).unwrap_or(u64::MAX);
    let lease = antecedent::execution::retain(&ctx, bytes.saturating_add(4096)).map_err(
        |error| match error {
            AdmissionRefusal::Cancelled => crate::CausalCancelledError::new_err(error.to_string()),
            _ => crate::CausalResourceError::new_err(error.to_string()),
        },
    )?;
    let spec = UnrolledPanelSpec {
        periods: declaration.periods,
        channels: declaration.channels,
        contexts: declaration.contexts,
        carryover_reach: declaration.carryover_reach,
        outcome_persistence: declaration.outcome_persistence,
        channel_persistence: declaration.channel_persistence,
        history_feedback: declaration.history_feedback,
        latent_confounding: declaration
            .latent_confounding
            .iter()
            .map(|(left, right)| (left.native(), right.native()))
            .collect(),
    };
    let graph = UnrolledPanelGraph::build(spec).map_err(|error| match error {
        IdentificationError::Cancelled => crate::CausalCancelledError::new_err(error.to_string()),
        IdentificationError::Budget { .. } => {
            crate::CausalResourceError::new_err(error.to_string())
        }
        other => crate::value_err(other.to_string()),
    })?;
    Ok((graph, declaration.outcome_periods, lease))
}
