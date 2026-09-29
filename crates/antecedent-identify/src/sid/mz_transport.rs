//! The bounded multi-source limited-experiment (mz-transportability) contract.
//!
//! The theorem is `TR^mz` (Bareinboim, Lee, Honavar & Pearl, `NeurIPS` 2013), whose
//! completeness Bareinboim & Pearl prove in "Transportability from Multiple
//! Environments with Limited Experiments: Completeness Results" (`NeurIPS` 2014,
//! UCLA R-443, Theorems 4–5). It is `TR^z` with a loop over source domains at
//! line 10: once a subcall exchanges into one source's experiment, the rest of
//! that subcall stays in that source. Separate c-factors of the line-4
//! factorization may therefore come from different sources; no single c-factor
//! ever mixes sources, so a joint over several sources' interventions is never
//! fabricated.
//!
//! Completeness holds for the theorem's own information family: for every
//! source, experiments on every subset of its declared controllable set with all
//! observed variables measured. A formula bound to a supplied catalog is sound
//! but incomplete; a failure is an obstruction only when it is evaluated
//! structurally over the declared controllable sets, never because a catalog
//! lacks a regime. When several sources certify the same factor, the paper
//! returns a weighted combination; for exact laws each is a valid formula, so
//! this contract selects one in canonical source order and keeps the others as
//! alternatives, which makes the result invariant to source declaration order.
//!
//! This module fixes the query, its bounds, and input validation. Identification
//! and execution are separate stages built on it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;
use std::sync::Arc;

use antecedent_core::{EvidenceCatalog, RegimeKind, SearchLimits, VariableId};
use antecedent_graph::SelectionDiagram;

use super::IdentificationError;
use super::z_transport::{
    Z_TRANSPORT_MAX_CONTROLLABLE, Z_TRANSPORT_MAX_OBSERVED, ZTransportQuery, ZTransportSourceSpec,
    validate_z_transport_query,
};

/// Observed variables in the shared graph.
pub const MZ_TRANSPORT_MAX_OBSERVED: usize = Z_TRANSPORT_MAX_OBSERVED;
/// Source populations in one query. One source is the single-source z route.
pub const MZ_TRANSPORT_MAX_SOURCES: usize = 4;
/// Controllable variables declared per source.
pub const MZ_TRANSPORT_MAX_CONTROLLABLE_PER_SOURCE: usize = Z_TRANSPORT_MAX_CONTROLLABLE;
/// Available source regimes the search may consider as candidate evidence.
pub const MZ_TRANSPORT_MAX_CANDIDATE_REGIMES: usize = 64;
/// Default search limits: states charged and recursion depth. Memory and
/// cancellation come from the execution context on every charge.
pub const MZ_TRANSPORT_DEFAULT_LIMITS: SearchLimits = SearchLimits { operations: 4096, depth: 24 };

/// One target interventional query answered from several limited-experiment sources.
///
/// Each source keeps its own population, selection targets on the shared causal
/// graph, declared controllable set (theoretical experimental availability) and
/// concrete experiment assignment. The regimes actually supplied live in the
/// evidence catalog; declaring a controllable set never implies results exist.
#[derive(Clone, Debug, PartialEq)]
pub struct MzTransportQuery {
    /// Joint outcomes in the target population.
    pub outcomes: Arc<[VariableId]>,
    /// Treatments whose target interventional response is queried.
    pub treatments: Arc<[VariableId]>,
    /// Target population; it supplies observational evidence only.
    pub target: Arc<str>,
    /// Two to [`MZ_TRANSPORT_MAX_SOURCES`] sources.
    pub sources: Arc<[ZTransportSourceSpec]>,
}

impl MzTransportQuery {
    /// Canonical form: sources ordered by population identity and every
    /// variable set sorted. Two queries that differ only in declaration order
    /// have equal canonical forms.
    #[must_use]
    pub fn canonical(&self) -> Self {
        let sorted = |variables: &[VariableId]| {
            let mut out = variables.to_vec();
            out.sort_unstable();
            Arc::<[VariableId]>::from(out)
        };
        let mut sources = self
            .sources
            .iter()
            .map(|source| {
                let mut assignment = source.experiment_assignment.to_vec();
                assignment.sort_by_key(|a| a.variable);
                ZTransportSourceSpec {
                    population: Arc::clone(&source.population),
                    controllable: sorted(&source.controllable),
                    experiment_assignment: assignment.into(),
                    selection_targets: sorted(&source.selection_targets),
                }
            })
            .collect::<Vec<_>>();
        sources.sort_by(|a, b| a.population.cmp(&b.population));
        Self {
            outcomes: sorted(&self.outcomes),
            treatments: sorted(&self.treatments),
            target: Arc::clone(&self.target),
            sources: sources.into(),
        }
    }

    /// The single-source z-transport query of one source, sharing the target question.
    #[must_use]
    pub fn source_query(&self, source: &ZTransportSourceSpec) -> ZTransportQuery {
        ZTransportQuery {
            outcomes: Arc::clone(&self.outcomes),
            treatments: Arc::clone(&self.treatments),
            controllable: Arc::clone(&source.controllable),
            experiment_assignment: Arc::clone(&source.experiment_assignment),
            source: Arc::clone(&source.population),
            target: Arc::clone(&self.target),
        }
    }
}

/// A query that passed the contract: canonical, with one selection diagram per
/// source in canonical source order.
#[derive(Clone, Debug)]
pub struct ValidatedMzTransportQuery {
    /// Canonical query.
    pub query: MzTransportQuery,
    /// Each source's selection diagram, aligned with `query.sources`.
    pub diagrams: Vec<SelectionDiagram>,
    /// Available source regimes the search may cite, counted against
    /// [`MZ_TRANSPORT_MAX_CANDIDATE_REGIMES`].
    pub candidate_regimes: usize,
}

/// Validate a multi-source query against the shared graph and supplied catalog.
///
/// Each source must pass the single-source z-transport contract on its own
/// selection diagram. Across sources: two to four distinct populations, none the
/// target; every available source experiment intervenes only on a subset of that
/// source's declared controllable set; the target supplies no experiments; and
/// at most [`MZ_TRANSPORT_MAX_CANDIDATE_REGIMES`] available source regimes.
///
/// # Errors
///
/// [`IdentificationError::UnsupportedInput`] with an `mz_transport.*` code for an
/// exceeded bound; [`IdentificationError::InvalidInput`] for a malformed query or
/// a catalog that contradicts its declarations.
pub fn validate_mz_transport_query(
    graph: &antecedent_graph::Admg,
    query: &MzTransportQuery,
    catalog: &EvidenceCatalog,
) -> Result<ValidatedMzTransportQuery, IdentificationError> {
    let query = query.canonical();
    if query.sources.len() < 2 {
        return Err(IdentificationError::invalid_input(
            "mz_transport.single_source_uses_z_transport",
        ));
    }
    if query.sources.len() > MZ_TRANSPORT_MAX_SOURCES {
        return Err(IdentificationError::UnsupportedInput { code: "mz_transport.source_count" });
    }
    let populations = query.sources.iter().map(|s| s.population.as_ref()).collect::<BTreeSet<_>>();
    if populations.len() != query.sources.len() || populations.contains(query.target.as_ref()) {
        return Err(IdentificationError::invalid_input("mz_transport.population_collision"));
    }
    let mut diagrams = Vec::with_capacity(query.sources.len());
    for source in query.sources.iter() {
        let diagram =
            SelectionDiagram::try_new(graph.clone(), Arc::clone(&source.selection_targets))
                .map_err(|error| IdentificationError::invalid_input(error.to_string()))?;
        validate_z_transport_query(&diagram, &query.source_query(source))?;
        diagrams.push(diagram);
    }
    let mut candidate_regimes = 0usize;
    for regime in catalog.regimes.iter().filter(|r| r.supplies_population_law()) {
        if regime.population == query.target {
            if regime.kind == RegimeKind::Experimental {
                return Err(IdentificationError::invalid_input(
                    "mz_transport.target_experiments_unsupported",
                ));
            }
            continue;
        }
        let Some(source) = query.sources.iter().find(|s| s.population == regime.population) else {
            continue;
        };
        if regime.interventions.iter().any(|v| !source.controllable.contains(v)) {
            return Err(IdentificationError::invalid_input(
                "mz_transport.regime_outside_controllable",
            ));
        }
        candidate_regimes += 1;
    }
    if candidate_regimes > MZ_TRANSPORT_MAX_CANDIDATE_REGIMES {
        return Err(IdentificationError::UnsupportedInput {
            code: "mz_transport.candidate_regime_count",
        });
    }
    Ok(ValidatedMzTransportQuery { query, diagrams, candidate_regimes })
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{
        DistributionAvailability, EvidenceKind, EvidenceRegime, InterventionAssignment, RegimeId,
        Value,
    };
    use antecedent_graph::{Admg, DenseNodeId};

    fn v(i: u32) -> VariableId {
        VariableId::from_raw(i)
    }

    /// A four-node chain Z1 -> X -> Z2 -> Y with Z1 <-> X and X <-> Y.
    fn graph() -> Admg {
        let mut g = Admg::with_variables(4);
        for (a, b) in [(0, 1), (1, 2), (2, 3)] {
            g.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        for (a, b) in [(0, 1), (1, 3)] {
            g.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        g
    }

    fn source(population: &str, controllable: &[u32], selection: &[u32]) -> ZTransportSourceSpec {
        ZTransportSourceSpec {
            population: Arc::from(population),
            controllable: controllable.iter().copied().map(v).collect::<Vec<_>>().into(),
            experiment_assignment: Arc::from([]),
            selection_targets: selection.iter().copied().map(v).collect::<Vec<_>>().into(),
        }
    }

    fn query(sources: Vec<ZTransportSourceSpec>) -> MzTransportQuery {
        MzTransportQuery {
            outcomes: Arc::from([v(3)]),
            treatments: Arc::from([v(1)]),
            target: Arc::from("target"),
            sources: sources.into(),
        }
    }

    fn experiment(id: u32, population: &str, on: &[u32]) -> EvidenceRegime {
        EvidenceRegime::try_new(
            RegimeId::from_raw(id),
            RegimeKind::Experimental,
            EvidenceKind::Available,
            on.iter().copied().map(v).collect::<Vec<_>>(),
            [],
            [v(0), v(1), v(2), v(3)]
                .into_iter()
                .filter(|x| !on.contains(&x.raw()))
                .collect::<Vec<_>>(),
            population,
            DistributionAvailability::Joint,
        )
        .unwrap()
    }

    fn catalog(regimes: Vec<EvidenceRegime>) -> EvidenceCatalog {
        EvidenceCatalog::try_new([], regimes, [], None).unwrap()
    }

    #[test]
    fn source_order_does_not_change_the_validated_query() {
        let a = source("a", &[2], &[0]);
        let b = source("b", &[0], &[2]);
        let evidence = catalog(vec![experiment(1, "a", &[2]), experiment(2, "b", &[0])]);
        let forward =
            validate_mz_transport_query(&graph(), &query(vec![a.clone(), b.clone()]), &evidence)
                .unwrap();
        let reverse = validate_mz_transport_query(&graph(), &query(vec![b, a]), &evidence).unwrap();
        assert_eq!(forward.query, reverse.query);
        assert_eq!(&*forward.query.sources[0].population, "a");
        assert_eq!(forward.candidate_regimes, 2);
        assert_eq!(forward.diagrams.len(), 2);
    }

    #[test]
    fn source_count_and_population_bounds_refuse_by_code() {
        let evidence = catalog(vec![]);
        let one =
            validate_mz_transport_query(&graph(), &query(vec![source("a", &[2], &[])]), &evidence);
        assert!(
            matches!(one, Err(IdentificationError::InvalidInput { message }) if message == "mz_transport.single_source_uses_z_transport")
        );
        let five = (0..5).map(|i| source(&format!("s{i}"), &[2], &[])).collect();
        assert!(matches!(
            validate_mz_transport_query(&graph(), &query(five), &evidence),
            Err(IdentificationError::UnsupportedInput { code: "mz_transport.source_count" })
        ));
        let duplicate = query(vec![source("a", &[2], &[]), source("a", &[0], &[])]);
        assert!(
            matches!(validate_mz_transport_query(&graph(), &duplicate, &evidence), Err(IdentificationError::InvalidInput { message }) if message == "mz_transport.population_collision")
        );
        let as_target = query(vec![source("a", &[2], &[]), source("target", &[0], &[])]);
        assert!(validate_mz_transport_query(&graph(), &as_target, &evidence).is_err());
        // Each source still passes the single-source contract.
        let no_controllable = query(vec![source("a", &[], &[]), source("b", &[0], &[])]);
        assert!(validate_mz_transport_query(&graph(), &no_controllable, &evidence).is_err());
    }

    #[test]
    fn supplied_regimes_must_respect_declared_availability() {
        let sources = query(vec![source("a", &[2], &[0]), source("b", &[0], &[2])]);
        // Source a declared only {Z2} controllable but supplies do(Z1).
        let outside = catalog(vec![experiment(1, "a", &[0])]);
        assert!(
            matches!(validate_mz_transport_query(&graph(), &sources, &outside), Err(IdentificationError::InvalidInput { message }) if message == "mz_transport.regime_outside_controllable")
        );
        let target_experiment = catalog(vec![experiment(1, "target", &[1])]);
        assert!(
            matches!(validate_mz_transport_query(&graph(), &sources, &target_experiment), Err(IdentificationError::InvalidInput { message }) if message == "mz_transport.target_experiments_unsupported")
        );
        // A proposed regime is not supplied evidence and is not counted.
        let mut proposed = experiment(1, "a", &[0]);
        proposed.evidence_kind = EvidenceKind::Proposed;
        let validated =
            validate_mz_transport_query(&graph(), &sources, &catalog(vec![proposed])).unwrap();
        assert_eq!(validated.candidate_regimes, 0);
    }

    #[test]
    fn candidate_regime_bound_refuses_rather_than_truncating() {
        let sources = query(vec![source("a", &[2], &[]), source("b", &[0], &[])]);
        let regimes = (0..=u32::try_from(MZ_TRANSPORT_MAX_CANDIDATE_REGIMES).unwrap())
            .map(|id| {
                let mut regime = experiment(id, "a", &[2]);
                regime.intervention_values = Arc::from([InterventionAssignment {
                    variable: v(2),
                    value: Value::f64(f64::from(id)),
                }]);
                regime
            })
            .collect();
        assert!(matches!(
            validate_mz_transport_query(&graph(), &sources, &catalog(regimes)),
            Err(IdentificationError::UnsupportedInput {
                code: "mz_transport.candidate_regime_count"
            })
        ));
    }
}
