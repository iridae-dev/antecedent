//! Per-population discovery in the analysis facade: one declared contract run on each
//! population's rows separately, with one retained statistic store, one data identity and one
//! result identity per population, and an edge-by-edge comparison.
//!
//! The driver itself ([`PerPopulationDiscovery`]) lives in `antecedent-discovery`; this module
//! is where its results enter the rest of the library:
//!
//! * [`add_per_population_claims`] places every population's discovery stages in a
//!   [`CompositionGraph`] under `{prefix}.{population}`, each node carrying the population as
//!   its coordinate, so a change of one population's rows recomputes that population's claims
//!   and leaves the other populations' claims valid.
//! * [`request_selection_targets_from_differences`] is the typed refusal of the one thing this
//!   route must not do: read an edge that differs between two discovered structures as a
//!   mechanism that differs between the populations.
//!
//! # What is not claimed
//!
//! Nothing is pooled and nothing is tested. Agreement or difference of two structures is a
//! comparison of two estimates; no error rate, invariance statement or selection diagram is
//! derived from it. A pooled search over stacked rows, or content-identical populations, are
//! refused by the driver (`selective_discovery.pooled_population_search`,
//! `selective_discovery.pooled_population_data`).
//!
//! # Refresh with outcome-only data
//!
//! There is no in-place refresh: new rows for one population are a new request, and that
//! population's run, identity and claims are recomputed while every other population's are
//! kept. Relabelling rows as another population is refused, not absorbed.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;
use std::convert::Infallible;

use antecedent_contract::composition_verification::{CompositionGraph, GraphError};
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_discovery::selective::{
    DiscoveredStructureRefusal, SelectiveDiscoveryError, StaticDiscoveryContract,
};

pub use antecedent_discovery::selective::populations::{
    PerPopulationDiscovery, PerPopulationReport, PopulationDiscovery, PopulationEdgeComparison,
    PopulationRows,
};

use super::selective_discovery::{DiscoveryClaimKeys, DiscoverySnapshot, add_discovery_claims};

/// Run `contract` on each population's rows with a fresh driver.
///
/// The driver is returned so its retained per-population stores can serve a later request
/// ([`PerPopulationDiscovery::run`] on the same driver).
///
/// # Errors
///
/// As [`PerPopulationDiscovery::run`].
pub fn discover_per_population(
    contract: &StaticDiscoveryContract,
    populations: &[PopulationRows<'_>],
    ctx: &ExecutionContext,
) -> Result<PerPopulationDiscovery, SelectiveDiscoveryError> {
    let mut driver = PerPopulationDiscovery::new();
    driver.run(contract, populations, ctx)?;
    Ok(driver)
}

/// Add every population's discovery stages of the driver's last committed report to `graph`.
///
/// Each population is added with [`add_discovery_claims`] under the prefix
/// `{prefix}.{population}`; no completion is declared, so a downstream claim anchors on the
/// discovered structure of its own population. The population label is part of every node's
/// coordinate. Returns the node keys by population.
///
/// # Errors
///
/// A key that already exists in `graph`, or a bound of the graph exceeded. A driver with no
/// committed report adds nothing and returns an empty map.
pub fn add_per_population_claims(
    graph: &mut CompositionGraph,
    prefix: &str,
    driver: &PerPopulationDiscovery,
) -> Result<BTreeMap<String, DiscoveryClaimKeys>, GraphError> {
    let mut keys = BTreeMap::new();
    let Some(report) = driver.report() else {
        return Ok(keys);
    };
    for population in &report.populations {
        let label = population.population.as_str();
        let Some(adapter) = driver.adapter(label) else { continue };
        let (Some(inputs), Some(data), Some(outcome)) =
            (adapter.inputs(), adapter.data_identity(), adapter.outcome())
        else {
            continue;
        };
        let snapshot = DiscoverySnapshot { inputs, data, outcome, completion: None };
        let added = add_discovery_claims(graph, &format!("{prefix}.{label}"), &snapshot)?;
        keys.insert(label.to_owned(), added);
    }
    Ok(keys)
}

/// Request selection targets from the edges that differ between populations.
///
/// Always refuses: a difference between two discovered structures is a difference between two
/// estimates, not a statement that a mechanism differs, and no error rate is attached to it.
/// The refusal names the number of differing edges and what would license a selection
/// diagram: targets the caller declares.
///
/// # Errors
///
/// Always [`DiscoveredStructureRefusal`] with detail
/// `selective_discovery.differences_are_not_selection_targets`.
pub fn request_selection_targets_from_differences(
    report: &PerPopulationReport,
    _candidate_targets: &[VariableId],
) -> Result<Infallible, DiscoveredStructureRefusal> {
    let differing = report.differing_edges().len();
    Err(DiscoveredStructureRefusal {
        code: antecedent_core::reason_code!("discovered_structure_unresolved"),
        detail: "selective_discovery.differences_are_not_selection_targets",
        unresolved_edges: Vec::new(),
        bounded_search_receipts: Vec::new(),
        remedy: format!(
            "{differing} edge(s) differ between the populations' discovered structures. A \
             difference between two estimates is not a statement that a mechanism differs and \
             no invariance test or error rate is attached to it. Declare the selection \
             targets from subject-matter knowledge or an experiment, then request transport \
             with a declared selection diagram"
        ),
    })
}
