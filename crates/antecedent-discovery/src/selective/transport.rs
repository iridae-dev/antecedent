//! What a transport request over a *discovered* structure returns.
//!
//! A selection diagram is a causal graph the caller asserts, plus the variables whose
//! mechanisms may differ between populations. A discovered structure is an estimate: an
//! equivalence class (CPDAG, PAG), possibly produced under an exhausted search bound, under
//! faithfulness and sufficiency assumptions the data cannot confirm. It is not an assertion
//! of one graph, so it cannot supply a selection diagram.
//!
//! The decision recorded here:
//!
//! * A discovered-structure request is **refused**, and the refusal names every unresolved
//!   edge (undirected edge of the class, circle endpoint, conflict, bidirected edge, edge
//!   whose search state is not definite) and any exhausted search bound.
//! * An envelope of transport answers across the members of the equivalence class is **not
//!   offered**: enumerating members and transporting through each states nothing about which
//!   member is true, multiplies the identification work by the class size, and would be read
//!   as a single answer. It is closed, not deferred silently.
//! * A fully directed, search-complete structure is *still* refused as a discovered
//!   structure: the caller declares the graph (and takes responsibility for the assumptions)
//!   by reviewing and accepting it, and the transport request is made with that declared
//!   graph and a declared selection diagram.
//! * Multi-population discovery (a search that pools several source datasets) is **out**:
//!   see [`super::contract`]. A pooled search carries its own invariance and identification
//!   assumptions, which belong to the transport contract and are not inferable from data.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::convert::Infallible;

use antecedent_core::VariableId;

use super::outcome::DiscoveryOutcome;
use super::structure::UnresolvedEdge;

/// Refusal of a transport request that rests on a discovered structure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiscoveredStructureRefusal {
    /// Registered reason code.
    pub code: &'static str,
    /// Namespaced detail: `selective_discovery.search_incomplete`,
    /// `selective_discovery.unresolved_structure` or
    /// `selective_discovery.unaccepted_discovery`.
    pub detail: &'static str,
    /// Every unresolved edge of the structure with its reason.
    pub unresolved_edges: Vec<UnresolvedEdge>,
    /// `(code, message)` of the exhausted search bounds.
    pub bounded_search_receipts: Vec<(String, String)>,
    /// What would license the request.
    pub remedy: String,
}

impl std::fmt::Display for DiscoveredStructureRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.detail, self.remedy)
    }
}

/// Request a selection diagram from a discovered structure.
///
/// Always refuses; the type says so. The refusal distinguishes why, so a caller can act on
/// it. `selection_targets` are the variables whose mechanisms the caller wants to differ
/// between populations; a target the structure does not contain is named in the remedy.
///
/// # Errors
///
/// Always [`DiscoveredStructureRefusal`].
pub fn request_selection_diagram(
    outcome: &DiscoveryOutcome,
    selection_targets: &[VariableId],
) -> Result<Infallible, DiscoveredStructureRefusal> {
    let code = antecedent_core::reason_code!("discovered_structure_unresolved");
    let unresolved = outcome.structure.unresolved_edges();
    let absent: Vec<String> = selection_targets
        .iter()
        .filter(|target| !outcome.structure.nodes().iter().any(|n| n.variable_id() == **target))
        .map(|target| format!("v{}", target.raw()))
        .collect();
    let absent_note = if absent.is_empty() {
        String::new()
    } else {
        format!(" Selection targets not in the discovered structure: {}.", absent.join(", "))
    };
    let (detail, remedy) = if outcome.search_incomplete {
        (
            "selective_discovery.search_incomplete",
            format!(
                "the search exhausted {} bound(s), so absent edges are not established. Rerun \
                 with larger bounds, then declare and review the graph before requesting \
                 transport.{absent_note}",
                outcome.bounded_search_receipts.len()
            ),
        )
    } else if !unresolved.is_empty() {
        (
            "selective_discovery.unresolved_structure",
            format!(
                "{} edge(s) are not resolved to one orientation within the equivalence class. \
                 Orient them with background knowledge or an intervention, or declare the \
                 graph; transport is not computed across the class.{absent_note}",
                unresolved.len()
            ),
        )
    } else {
        (
            "selective_discovery.unaccepted_discovery",
            format!(
                "a discovered graph is an estimate under faithfulness and sufficiency \
                 assumptions, not a declared causal graph. Review and accept it as the causal \
                 graph, then declare the selection diagram for the transport request.\
                 {absent_note}"
            ),
        )
    };
    Err(DiscoveredStructureRefusal {
        code,
        detail,
        unresolved_edges: unresolved,
        bounded_search_receipts: outcome.bounded_search_receipts.clone(),
        remedy,
    })
}
