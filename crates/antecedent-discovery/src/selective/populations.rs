//! Discovery over several source populations, one population at a time.
//!
//! # What this route is
//!
//! One declared contract is run separately on each population's rows. Every population has
//! its own retained statistic store, its own data identity, its own stage inputs and its own
//! result identity. Nothing is pooled: no row of one population is ever an input of another
//! population's test, and no statistic is shared between populations. The report lines the
//! per-population structures up edge by edge so a reader can see where they agree and where
//! they differ.
//!
//! # What this route is not
//!
//! * **Not a pooled search.** A search over rows stacked from several populations (with or
//!   without context or dataset-dummy variables) carries its own invariance and
//!   faithfulness-across-populations assumptions. It is refused here
//!   (`selective_discovery.pooled_population_search`): use the pooled algorithm directly and
//!   declare the cross-population assumptions under the transport contract.
//! * **Not a selection diagram.** A difference between two discovered structures is a
//!   difference between two estimates. It is not a statement that a mechanism differs, and no
//!   function here turns an edge difference into a set of selection targets. A selection
//!   diagram is declared by the caller; see [`super::transport`].
//! * **Not an invariance test.** Agreement between populations is reported, never tested; no
//!   error rate is attached to an agreement or a difference.
//!
//! # Pooled-data confusion
//!
//! Two populations whose searched columns, row counts, mask, weights and column types are
//! content-identical are the same rows presented under two labels, which is the signature of
//! a pooled table split by label, or of a table relabelled. That request is refused
//! (`selective_discovery.pooled_population_data`) rather than reported as two independent
//! populations that happen to agree. Overlap that is not an exact duplicate cannot be seen
//! from content identities: the caller declares disjoint populations.
//!
//! # Whole-method calibration
//!
//! The recorded design in [`super::claims`] and [`super::reuse`] applies per population.
//! This route does not state an error rate for agreement between populations.
//!
//! # Refresh with outcome-only data
//!
//! There is no in-place refresh. New rows for one population are new rows with a new data
//! identity: that population's run and the comparison report are recomputed, and the other
//! populations' results are kept untouched because nothing is pooled. Rows relabelled as
//! another population, or content-identical to another population's, are refused rather
//! than absorbed. Outcome-only rows leave the declared contract and the structure edge
//! alignment rule unchanged.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use antecedent_core::{CanonicalWriter, ExecutionContext};
use antecedent_data::TabularData;

use super::adapter::{SelectiveRunReport, StaticSelectiveDiscovery};
use super::contract::StaticDiscoveryContract;
use super::identity::{DataIdentity, Digest, SELECTIVE_IDENTITY_VERSION, static_data_identity};
use super::refusal::SelectiveDiscoveryError;
use super::structure::{Mark, StructureNode};

const POPULATIONS_TAG: &str = "antecedent.selective_discovery.populations.v1";

fn invalid(detail: &'static str, remedy: impl Into<String>) -> SelectiveDiscoveryError {
    SelectiveDiscoveryError::refused(
        antecedent_core::reason_code!("invalid_argument"),
        detail,
        remedy,
    )
}

/// One population's rows, labelled.
#[derive(Clone, Copy, Debug)]
pub struct PopulationRows<'a> {
    /// The population the rows were drawn from. Distinct across a request.
    pub population: &'a str,
    /// The population's own rows.
    pub data: &'a TabularData,
}

/// What discovery on one population returned and what it retained.
#[derive(Clone, Debug)]
pub struct PopulationDiscovery {
    /// Population label.
    pub population: String,
    /// Identity of this population's searched columns.
    pub data_identity: DataIdentity,
    /// Reuse decisions and observed work of this population's run.
    pub report: SelectiveRunReport,
    /// Identity of this population's structure.
    pub structure_identity: Digest,
    /// Identity of everything this population's run returned except execution detail.
    pub result_identity: Digest,
    /// Statistics retained for this population (its receipts; never shared).
    pub retained_statistics: usize,
    /// Identity of this population's receipt set; `None` when the retained statistics exceed
    /// what a portable state carries.
    pub receipts_identity: Option<Digest>,
    /// Whether this population's search exhausted a bound.
    pub search_incomplete: bool,
    /// Edges of this population's structure that are not resolved.
    pub unresolved_edges: usize,
}

/// One edge compared across populations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PopulationEdgeComparison {
    /// First endpoint (the smaller node).
    pub a: StructureNode,
    /// Second endpoint.
    pub b: StructureNode,
    /// Per population, in label order: the marks at `(a, b)`, or `None` when the population's
    /// structure has no such edge.
    pub marks: Vec<(String, Option<(Mark, Mark)>)>,
}

impl PopulationEdgeComparison {
    /// Whether every population has this edge with the same marks.
    #[must_use]
    pub fn agrees(&self) -> bool {
        self.marks.windows(2).all(|pair| pair[0].1 == pair[1].1)
    }
}

/// The result of one per-population request.
///
/// Descriptive only: the comparison is not a selection diagram and carries no error rate.
#[derive(Clone, Debug)]
pub struct PerPopulationReport {
    /// Per-population results, in label order.
    pub populations: Vec<PopulationDiscovery>,
    /// Every edge present in any population, in canonical order.
    pub edges: Vec<PopulationEdgeComparison>,
    /// Identity of the set: labels, data identities, result identities and receipt
    /// identities in label order.
    pub identity: Digest,
}

impl PerPopulationReport {
    /// Edges on which the populations do not all agree (present in some, or different marks).
    #[must_use]
    pub fn differing_edges(&self) -> Vec<&PopulationEdgeComparison> {
        self.edges.iter().filter(|edge| !edge.agrees()).collect()
    }
}

/// Per-population discovery with one retained statistic store per population.
///
/// Each population's adapter is atomic on its own: a refused or failed request leaves that
/// population's retained state, outcome and identities as they were. The set-level report is
/// replaced only by a request in which every population succeeded.
#[derive(Debug)]
pub struct PerPopulationDiscovery {
    adapters: BTreeMap<String, StaticSelectiveDiscovery>,
    committed: Option<PerPopulationReport>,
}

impl Default for PerPopulationDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

impl PerPopulationDiscovery {
    /// A driver with no retained state.
    #[must_use]
    pub fn new() -> Self {
        Self { adapters: BTreeMap::new(), committed: None }
    }

    /// The last report in which every population succeeded.
    #[must_use]
    pub const fn report(&self) -> Option<&PerPopulationReport> {
        self.committed.as_ref()
    }

    /// The retained adapter of one population.
    #[must_use]
    pub fn adapter(&self, population: &str) -> Option<&StaticSelectiveDiscovery> {
        self.adapters.get(population)
    }

    /// Run `contract` on each population's rows, separately.
    ///
    /// `contract.population` is replaced by each population's label; the contract must
    /// declare no further populations and no multi-dataset constraints. The context's seed
    /// must be the contract's, as for a single population.
    ///
    /// All declarations and data are checked before any population is executed, so a pooled
    /// or malformed request refuses without work.
    ///
    /// # Errors
    ///
    /// Fewer than two populations, a blank or repeated label, a pooled search declaration,
    /// populations whose rows are content-identical (pooled-data confusion), a contract or
    /// data a single run refuses, or an algorithm failure or cancellation in one population.
    pub fn run(
        &mut self,
        contract: &StaticDiscoveryContract,
        populations: &[PopulationRows<'_>],
        ctx: &ExecutionContext,
    ) -> Result<&PerPopulationReport, SelectiveDiscoveryError> {
        validate_request(contract, populations)?;

        // Check every population before executing any: contract, data and plan.
        let mut identities: Vec<(&str, DataIdentity)> = Vec::with_capacity(populations.len());
        for rows in populations {
            let per_population = for_population(contract, rows.population);
            per_population.validate()?;
            let adapter = self.adapters.get(rows.population);
            match adapter {
                Some(adapter) => {
                    adapter.plan(&per_population, rows.data, ctx)?;
                }
                None => {
                    StaticSelectiveDiscovery::new().plan(&per_population, rows.data, ctx)?;
                }
            }
            let identity = static_data_identity(rows.data, &contract.variables)
                .map_err(SelectiveDiscoveryError::from)?;
            identities.push((rows.population, identity));
        }
        refuse_duplicate_rows(&identities)?;

        // Execute. Each adapter replaces its own state atomically; the set-level report is
        // committed only when all of them succeed.
        let mut results: Vec<PopulationDiscovery> = Vec::with_capacity(populations.len());
        let mut structures: Vec<(String, Vec<super::structure::StructureEdge>)> = Vec::new();
        for (rows, (_, identity)) in populations.iter().zip(identities) {
            let per_population = for_population(contract, rows.population);
            let adapter = self.adapters.entry(rows.population.to_owned()).or_default();
            let report = adapter.run(&per_population, rows.data, ctx)?;
            let outcome = adapter.outcome().ok_or_else(|| {
                invalid(
                    "selective_discovery.nothing_retained",
                    "the population run retained no outcome",
                )
            })?;
            structures.push((rows.population.to_owned(), outcome.structure.edges().to_vec()));
            results.push(PopulationDiscovery {
                population: rows.population.to_owned(),
                data_identity: identity,
                report,
                structure_identity: outcome.structure_identity,
                result_identity: outcome.result_identity,
                retained_statistics: adapter.retained_statistics(),
                receipts_identity: adapter.export_state().ok().map(|s| s.receipts_identity),
                search_incomplete: outcome.search_incomplete,
                unresolved_edges: outcome.structure.unresolved_edges().len(),
            });
        }
        results.sort_by(|a, b| a.population.cmp(&b.population));
        structures.sort_by(|a, b| a.0.cmp(&b.0));

        let report = PerPopulationReport {
            edges: compare(&structures),
            identity: set_identity(&results),
            populations: results,
        };
        Ok(self.committed.insert(report))
    }
}

fn for_population(contract: &StaticDiscoveryContract, population: &str) -> StaticDiscoveryContract {
    let mut per_population = contract.clone();
    population.clone_into(&mut per_population.population);
    per_population.additional_populations.clear();
    per_population
}

fn validate_request(
    contract: &StaticDiscoveryContract,
    populations: &[PopulationRows<'_>],
) -> Result<(), SelectiveDiscoveryError> {
    let md = &contract.constraints.multi_dataset;
    if !contract.additional_populations.is_empty()
        || !md.context_variables.is_empty()
        || !md.space_dummy_variables.is_empty()
        || !md.time_dummy_variables.is_empty()
        || md.include_time_dummy
    {
        return Err(SelectiveDiscoveryError::refused(
            antecedent_core::reason_code!("route_not_supported"),
            "selective_discovery.pooled_population_search",
            "per-population discovery runs the contract on each population separately and \
             pools nothing. A pooled or context-variable search has its own cross-population \
             invariance assumptions: use the pooled algorithm directly and declare those \
             assumptions under the transport contract",
        ));
    }
    if populations.len() < 2 {
        return Err(invalid(
            "selective_discovery.populations_missing",
            "declare at least two populations; a single population is the single-population \
             route",
        ));
    }
    let mut seen = BTreeSet::new();
    for rows in populations {
        if rows.population.trim().is_empty() {
            return Err(invalid(
                "selective_discovery.population_missing",
                "every table is declared with the population it was drawn from",
            ));
        }
        if !seen.insert(rows.population) {
            return Err(invalid(
                "selective_discovery.population_repeated",
                format!(
                    "population `{}` is declared twice; give each table its own population \
                     label, or merge them into one declared population",
                    rows.population
                ),
            ));
        }
    }
    Ok(())
}

fn refuse_duplicate_rows(
    identities: &[(&str, DataIdentity)],
) -> Result<(), SelectiveDiscoveryError> {
    for (i, (first, a)) in identities.iter().enumerate() {
        for (second, b) in &identities[i + 1..] {
            if a.digest == b.digest {
                return Err(invalid(
                    "selective_discovery.pooled_population_data",
                    format!(
                        "populations `{first}` and `{second}` hold content-identical rows, so \
                         they are one dataset under two labels (a pooled table split by label, \
                         or a relabelled copy), not two independent populations. Supply each \
                         population's own rows, or declare one population"
                    ),
                ));
            }
        }
    }
    Ok(())
}

fn compare(
    structures: &[(String, Vec<super::structure::StructureEdge>)],
) -> Vec<PopulationEdgeComparison> {
    let mut keys: BTreeSet<(StructureNode, StructureNode)> = BTreeSet::new();
    for (_, edges) in structures {
        keys.extend(edges.iter().map(|e| (e.a, e.b)));
    }
    keys.into_iter()
        .map(|(a, b)| {
            let marks = structures
                .iter()
                .map(|(population, edges)| {
                    let found =
                        edges.iter().find(|e| e.a == a && e.b == b).map(|e| (e.at_a, e.at_b));
                    (population.clone(), found)
                })
                .collect();
            PopulationEdgeComparison { a, b, marks }
        })
        .collect()
}

fn set_identity(results: &[PopulationDiscovery]) -> Digest {
    let mut writer = CanonicalWriter::new(POPULATIONS_TAG);
    writer.word(SELECTIVE_IDENTITY_VERSION).word(results.len() as u64);
    for result in results {
        writer
            .text(&result.population)
            .bytes(&result.data_identity.digest)
            .bytes(&result.result_identity);
        match &result.receipts_identity {
            Some(identity) => writer.word(1).bytes(identity),
            None => writer.word(0),
        };
    }
    writer.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::selective::structure::StructureEdge;

    fn node(variable: u32) -> StructureNode {
        StructureNode { kind: 0, variable, aux: 0 }
    }

    fn edge(a: u32, b: u32, at_a: Mark, at_b: Mark) -> StructureEdge {
        StructureEdge { a: node(a), b: node(b), at_a, at_b, middle: 4 }
    }

    #[test]
    fn comparison_reports_absent_and_remarked_edges_as_differences() {
        let structures = vec![
            ("north".to_owned(), vec![edge(0, 1, Mark::Tail, Mark::Arrow)]),
            (
                "south".to_owned(),
                vec![edge(0, 1, Mark::Tail, Mark::Arrow), edge(1, 2, Mark::Tail, Mark::Arrow)],
            ),
            ("west".to_owned(), vec![edge(0, 1, Mark::Arrow, Mark::Tail)]),
        ];
        let compared = compare(&structures);
        assert_eq!(compared.len(), 2);
        // 0 -> 1 in two populations, reversed in the third.
        assert!(!compared[0].agrees());
        assert_eq!(compared[0].marks[2].1, Some((Mark::Arrow, Mark::Tail)));
        // 1 -> 2 only in `south`.
        assert_eq!(compared[1].marks[0].1, None);
        assert_eq!(compared[1].marks[1].1, Some((Mark::Tail, Mark::Arrow)));
        assert!(!compared[1].agrees());
    }

    #[test]
    fn identical_structures_agree_on_every_edge() {
        let same = vec![edge(0, 1, Mark::Tail, Mark::Arrow)];
        let compared = compare(&[("a".to_owned(), same.clone()), ("b".to_owned(), same)]);
        assert_eq!(compared.len(), 1);
        assert!(compared[0].agrees());
    }
}
