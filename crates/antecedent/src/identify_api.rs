//! Free `identify()`: identification as a function of structure and query.
//!
//! `CausalAnalysis::identify_only()` requires a caller-supplied `TabularData` it never
//! reads (see `examples/rust/identify_only.rs`, which apologises for this in a
//! comment). Identification does not need data — only a graph and a query — and this
//! module says so in its signature: [`identify`] and [`identify_with`] take an
//! [`AcceptedGraph`] and a [`CausalQuery`], nothing else.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::CausalQuery;
use antecedent_graph::Pag;
use antecedent_identify::{IdentificationEnvelope, IdentificationResult, IdentifiedEstimand};

use crate::accepted::{AcceptedGraph, GraphClass};
use crate::error::CausalError;
use crate::strategy_table::{
    self, DEFAULT_ADMG_IDENTIFIER_ID, DEFAULT_IDENTIFIER_ID, DEFAULT_PAG_IDENTIFIER_ID,
    IdentifierId, identify_admg, identify_pag, identify_static_query,
};

/// Identification outcome.
///
/// Point and envelope stay distinct: uncertainty sources are not collapsed into one
/// number. A [`Self::Point`] result comes from a single-graph identifier (DAG, ADMG,
/// or a fully-oriented CPDAG reinterpreted as a DAG). A [`Self::Envelope`] result comes
/// from class-aware identification over a PAG's equivalence class, where some MAG
/// completions may identify and others may not — that split is preserved rather than
/// averaged away.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum Identification {
    /// Single-graph identification result.
    Point {
        /// Underlying identification result.
        result: IdentificationResult,
        /// Identifier strategy that produced this result.
        strategy: IdentifierId,
        /// [`AcceptedGraph::version`] of the structure identification ran against.
        structure_version: u32,
    },
    /// Class-aware (PAG equivalence-class) identification envelope.
    Envelope {
        /// Underlying identification envelope.
        envelope: IdentificationEnvelope<Pag>,
        /// Identifier strategy that produced this result.
        strategy: IdentifierId,
        /// [`AcceptedGraph::version`] of the structure identification ran against.
        structure_version: u32,
    },
}

impl Identification {
    /// Aggregate identification status.
    #[must_use]
    pub fn status(&self) -> antecedent_core::IdentificationStatus {
        match self {
            Self::Point { result, .. } => result.status,
            Self::Envelope { envelope, .. } => envelope.status,
        }
    }

    /// Identifier strategy that produced this result.
    #[must_use]
    pub fn strategy(&self) -> IdentifierId {
        match self {
            Self::Point { strategy, .. } | Self::Envelope { strategy, .. } => *strategy,
        }
    }

    /// [`AcceptedGraph::version`] of the structure identification ran against.
    #[must_use]
    pub fn structure_version(&self) -> u32 {
        match self {
            Self::Point { structure_version, .. } | Self::Envelope { structure_version, .. } => {
                *structure_version
            }
        }
    }

    /// Whether the aggregate status is acceptable for estimation.
    #[must_use]
    pub fn is_identified(&self) -> bool {
        strategy_table::identification_status_acceptable(self.status())
    }

    /// Identified estimands.
    ///
    /// For [`Self::Point`], this is the identifier's full estimand list. For
    /// [`Self::Envelope`], an envelope has no single estimand list by construction — it
    /// returns the shared invariant estimand as a one-element slice when all identified
    /// cases agree on it, and an empty slice otherwise (including when nothing in the
    /// equivalence class identifies).
    #[must_use]
    pub fn estimands(&self) -> &[IdentifiedEstimand] {
        match self {
            Self::Point { result, .. } => &result.estimands,
            Self::Envelope { envelope, .. } => match &envelope.invariant {
                Some(estimand) => std::slice::from_ref(estimand),
                None => &[],
            },
        }
    }
}

/// Identify `query` against `structure` using the class-appropriate default identifier.
///
/// Takes no data: identification is a function of structure and query.
///
/// # Errors
///
/// Unsupported graph-class/query pair, or identification failure.
pub fn identify(
    structure: &AcceptedGraph,
    query: &CausalQuery,
) -> Result<Identification, CausalError> {
    let strategy = default_strategy(structure.class());
    identify_with(structure, query, strategy)
}

/// Identify `query` against `structure` using an explicit identifier strategy.
///
/// # Errors
///
/// Strategy incompatible with the graph class, or identification failure.
///
/// # Panics
///
/// Never in practice. Each `expect` asserts an invariant that [`AcceptedGraph::class`]
/// itself establishes — the class tag is derived from the stored graph, so the matching
/// accessor is always `Some`. A panic here would mean `AcceptedGraph`'s internal
/// representation had drifted from its class tag.
pub fn identify_with(
    structure: &AcceptedGraph,
    query: &CausalQuery,
    strategy: IdentifierId,
) -> Result<Identification, CausalError> {
    let structure_version = structure.version();
    match structure.class() {
        GraphClass::Dag => {
            let dag = structure.as_dag().expect("class() == Dag implies as_dag() is Some");
            let result = identify_static_query(strategy, dag, query)?;
            Ok(Identification::Point { result, strategy, structure_version })
        }
        GraphClass::Cpdag => {
            let cpdag = structure.as_cpdag().expect("class() == Cpdag implies as_cpdag() is Some");
            // `AcceptedGraph::cpdag` / the CpdagReview review gate already refused any
            // Cpdag carrying undirected or conflict marks, so `try_into_dag` here is a
            // lossless structural reinterpretation of already-fully-oriented data —
            // never a choice among members of an otherwise-unresolved equivalence class.
            let dag = cpdag.try_into_dag()?;
            let result = identify_static_query(strategy, &dag, query)?;
            Ok(Identification::Point { result, strategy, structure_version })
        }
        GraphClass::Pag => {
            let pag = structure.as_pag().expect("class() == Pag implies as_pag() is Some");
            let CausalQuery::AverageEffect(average_effect) = query else {
                return Err(CausalError::Unsupported {
                    message: "PAG identification supports only CausalQuery::AverageEffect",
                });
            };
            let envelope = identify_pag(strategy, pag, average_effect)?;
            Ok(Identification::Envelope { envelope, strategy, structure_version })
        }
        GraphClass::Admg => {
            let admg = structure.as_admg().expect("class() == Admg implies as_admg() is Some");
            let CausalQuery::AverageEffect(average_effect) = query else {
                return Err(CausalError::Unsupported {
                    message: "ADMG identification supports only CausalQuery::AverageEffect",
                });
            };
            let result = identify_admg(strategy, admg, average_effect)?;
            Ok(Identification::Point { result, strategy, structure_version })
        }
        GraphClass::TemporalDag | GraphClass::TemporalCpdag | GraphClass::TemporalPag => {
            // `strategy_table` exposes no temporal-graph identify entry point among the
            // three this module is allowed to call (`identify_static_query` takes
            // `&Dag`, `identify_pag` takes `&Pag`, `identify_admg` takes `&Admg`; none
            // accept a temporal graph type). Refusing here — rather than guessing a
            // conversion — keeps this free function from silently picking a resolution
            // strategy for temporal structure it was never wired to identify.
            Err(CausalError::Unsupported {
                message: "temporal graph classes have no wired free-identify() entry point; \
                          use CausalAnalysis for the temporal path",
            })
        }
    }
}

/// Class-appropriate default identifier, reusing the existing `DEFAULT_*_IDENTIFIER_ID`
/// constants rather than hardcoding wire strings.
fn default_strategy(class: GraphClass) -> IdentifierId {
    match class {
        GraphClass::Dag | GraphClass::Cpdag => DEFAULT_IDENTIFIER_ID,
        GraphClass::Pag => DEFAULT_PAG_IDENTIFIER_ID,
        GraphClass::Admg => DEFAULT_ADMG_IDENTIFIER_ID,
        // No supported temporal path; any value is fine here since `identify_with`
        // refuses temporal classes before consulting `strategy`.
        GraphClass::TemporalDag | GraphClass::TemporalCpdag | GraphClass::TemporalPag => {
            DEFAULT_IDENTIFIER_ID
        }
    }
}

#[cfg(test)]
mod tests {
    use antecedent_core::AverageEffectQuery;
    use antecedent_graph::{Dag, DenseNodeId};

    use super::*;

    fn toy_dag() -> Dag {
        let mut g = Dag::with_variables(2);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        g
    }

    #[test]
    fn identify_uses_default_strategy_and_needs_no_data() {
        let structure = AcceptedGraph::from(toy_dag());
        let query = CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(
            antecedent_core::VariableId::from_raw(0),
            antecedent_core::VariableId::from_raw(1),
        ));
        let identification = identify(&structure, &query).unwrap();
        match &identification {
            Identification::Point { .. } => {}
            Identification::Envelope { .. } => panic!("expected Point for a Dag structure"),
        }
        assert!(identification.is_identified());
        assert_eq!(identification.structure_version(), 1);
        assert_eq!(identification.strategy(), DEFAULT_IDENTIFIER_ID);
    }
}
