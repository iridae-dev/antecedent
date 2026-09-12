//! Free `identify()`: identification as a function of structure and query.
//!
//! `Study::identify_only()` requires a caller-supplied `TabularData` it never
//! reads (see `examples/rust/identify_only.rs`, which apologises for this in a
//! comment). Identification does not need data — only a graph and a query — and this
//! module says so in its signature: [`identify`] and [`identify_with`] take an
//! [`AcceptedGraph`] and a [`CausalQuery`], nothing else.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
use std::sync::Arc;

use antecedent_core::{
    AverageEffectQuery, CausalQuery, Intervention, TemporalEffectQuery, TemporalIndexer, Value,
};
use antecedent_graph::Pag;
use antecedent_identify::{
    IdentificationEnvelope, IdentificationResult, IdentifiedEstimand, TemporalBackdoorIdentifier,
    TemporalClassEnvelope,
};

use crate::accepted::{AcceptedGraph, GraphClass};
use crate::error::CausalError;
use crate::strategy_table::{
    self, DEFAULT_ADMG_IDENTIFIER_ID, DEFAULT_IDENTIFIER_ID, DEFAULT_PAG_IDENTIFIER_ID,
    IdentifierId, identify_admg, identify_cpdag, identify_pag, identify_static_query,
    identify_temporal_cpdag, identify_temporal_pag,
};

/// Identification outcome.
///
/// Point and envelope stay distinct: uncertainty sources are not collapsed into one
/// number. A [`Self::Point`] result comes from a single-graph identifier (DAG or
/// ADMG). A [`Self::Envelope`] / [`Self::CpdagEnvelope`] result comes from
/// class-aware identification over a PAG or CPDAG equivalence class, where some
/// completions may identify and others may not — that split is preserved rather
/// than averaged away.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum Identification {
    /// Single-graph identification result.
    Point {
        /// Underlying identification result.
        result: IdentificationResult,
        /// Unfolding coordinates for a temporal point result. Estimand variable
        /// ids remain in the expression arena's coordinate system.
        temporal_indexer: Option<TemporalIndexer>,
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
    /// Class-aware (CPDAG MEC) identification envelope. Completions are DAGs.
    CpdagEnvelope {
        /// Underlying identification envelope over MEC DAG completions.
        envelope: IdentificationEnvelope<antecedent_graph::Dag>,
        /// Identifier strategy that produced this result.
        strategy: IdentifierId,
        /// [`AcceptedGraph::version`] of the structure identification ran against.
        structure_version: u32,
    },
    /// Class-aware temporal envelope over `TemporalDag` completions.
    TemporalEnvelope {
        /// Underlying identification envelope plus unfold indexers.
        envelope: TemporalClassEnvelope,
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
            Self::CpdagEnvelope { envelope, .. } => envelope.status,
            Self::TemporalEnvelope { envelope, .. } => envelope.envelope.status,
        }
    }

    /// Identifier strategy that produced this result.
    #[must_use]
    pub fn strategy(&self) -> IdentifierId {
        match self {
            Self::Point { strategy, .. }
            | Self::Envelope { strategy, .. }
            | Self::CpdagEnvelope { strategy, .. }
            | Self::TemporalEnvelope { strategy, .. } => *strategy,
        }
    }

    /// [`AcceptedGraph::version`] of the structure identification ran against.
    #[must_use]
    pub fn structure_version(&self) -> u32 {
        match self {
            Self::Point { structure_version, .. }
            | Self::Envelope { structure_version, .. }
            | Self::CpdagEnvelope { structure_version, .. }
            | Self::TemporalEnvelope { structure_version, .. } => *structure_version,
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
            Self::CpdagEnvelope { envelope, .. } => match &envelope.invariant {
                Some(estimand) => std::slice::from_ref(estimand),
                None => &[],
            },
            Self::TemporalEnvelope { envelope, .. } => match &envelope.envelope.invariant {
                Some(estimand) => std::slice::from_ref(estimand),
                None => &[],
            },
        }
    }

    /// Stable completion fingerprints for a temporal class envelope.
    ///
    /// Empty for point or static envelopes. Use these keys with [`crate::ClassPrior`].
    #[must_use]
    pub fn completion_keys(&self) -> Vec<u64> {
        match self {
            Self::TemporalEnvelope { envelope, .. } => {
                envelope.envelope.cases.iter().map(|case| case.graph.fingerprint()).collect()
            }
            Self::Point { .. } | Self::Envelope { .. } | Self::CpdagEnvelope { .. } => Vec::new(),
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
    let strategy = match (structure.class(), query) {
        (GraphClass::Dag, CausalQuery::Counterfactual(_)) => IdentifierId::GcmParametric,
        (GraphClass::Dag, CausalQuery::Response(_)) => IdentifierId::ResponseBackdoor,
        (GraphClass::Dag, CausalQuery::Mediation(_) | CausalQuery::PathSpecific(_)) => {
            IdentifierId::PathSpecificNatural
        }
        (GraphClass::Dag, CausalQuery::Distribution(_)) => IdentifierId::GeneralId,
        _ => default_strategy(structure.class()),
    };
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
    identify_with_source(structure, query, strategy, crate::support::StructureSource::Accepted)
}

/// Identify a supplied explicit DAG without relabeling it as an accepted-graph cell.
///
/// # Errors
/// Unsupported query coordinate or identification failure.
pub fn identify_dag(
    graph: &antecedent_graph::Dag,
    query: &CausalQuery,
) -> Result<Identification, CausalError> {
    let strategy = match query {
        CausalQuery::Response(_) => IdentifierId::ResponseBackdoor,
        CausalQuery::Mediation(_) | CausalQuery::PathSpecific(_) => {
            IdentifierId::PathSpecificNatural
        }
        CausalQuery::Counterfactual(_) => IdentifierId::GcmParametric,
        CausalQuery::Distribution(_) => IdentifierId::GeneralId,
        _ => DEFAULT_IDENTIFIER_ID,
    };
    identify_with_source(
        &AcceptedGraph::from(graph.clone()),
        query,
        strategy,
        crate::support::StructureSource::Explicit,
    )
}

#[allow(clippy::too_many_lines)]
fn identify_with_source(
    structure: &AcceptedGraph,
    query: &CausalQuery,
    strategy: IdentifierId,
    source: crate::support::StructureSource,
) -> Result<Identification, CausalError> {
    let structure_version = structure.version();
    if let Some(cell) = crate::support::support_cell(
        query,
        crate::support::effective_graph_class(structure, query),
        source,
        &crate::inference::InferenceMode::Frequentist,
        crate::analysis::RefuteSuite::None,
    ) {
        crate::support::refuse_if_not_applicable(cell)?;
    }
    match structure.class() {
        GraphClass::Dag => {
            let dag = structure.as_dag().expect("class() == Dag implies as_dag() is Some");
            let id_query = static_identify_query(query);
            let result = identify_static_query(strategy, dag, &id_query)?;
            Ok(Identification::Point {
                result,
                temporal_indexer: None,
                strategy,
                structure_version,
            })
        }
        GraphClass::Cpdag => {
            let cpdag = structure.as_cpdag().expect("class() == Cpdag implies as_cpdag() is Some");
            let envelope = if let CausalQuery::Response(response) = query {
                strategy_table::identify_cpdag_response(strategy, cpdag, response)?
            } else {
                identify_cpdag(strategy, cpdag, &class_ate_witness(query, "CPDAG")?)?
            };
            Ok(Identification::CpdagEnvelope { envelope, strategy, structure_version })
        }
        GraphClass::Pag => {
            let pag = structure.as_pag().expect("class() == Pag implies as_pag() is Some");
            let envelope = if let CausalQuery::Response(response) = query {
                strategy_table::identify_pag_response(strategy, pag, response)?
            } else {
                identify_pag(strategy, pag, &class_ate_witness(query, "PAG")?)?
            };
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
            Ok(Identification::Point {
                result,
                temporal_indexer: None,
                strategy,
                structure_version,
            })
        }
        GraphClass::TemporalDag => {
            let dag = structure
                .as_temporal_dag()
                .expect("class() == TemporalDag implies as_temporal_dag() is Some");
            let CausalQuery::TemporalEffect(q) = query else {
                return Err(CausalError::Unsupported {
                    message: "TemporalDag identification supports only CausalQuery::TemporalEffect",
                });
            };
            let identified = TemporalBackdoorIdentifier::new()
                .identify_temporal(dag, q)
                .map_err(CausalError::from)?;
            Ok(Identification::Point {
                result: identified.result,
                temporal_indexer: Some(identified.indexer),
                strategy,
                structure_version,
            })
        }
        GraphClass::TemporalCpdag => {
            let cpdag = structure
                .as_temporal_cpdag()
                .expect("class() == TemporalCpdag implies as_temporal_cpdag() is Some");
            let q = temporal_class_identify_query(query, "TemporalCpdag")?;
            let mut envelope = identify_temporal_cpdag(strategy, cpdag, &q)?;
            refine_temporal_class_identification(&mut envelope, query, q.horizon_steps)?;
            audit_temporal_class_horizons(&mut envelope, structure, query, strategy, &q)?;
            Ok(Identification::TemporalEnvelope { envelope, strategy, structure_version })
        }
        GraphClass::TemporalPag => {
            let pag = structure
                .as_temporal_pag()
                .expect("class() == TemporalPag implies as_temporal_pag() is Some");
            if matches!(query, CausalQuery::Mediation(_)) {
                return Err(CausalError::Unsupported {
                    message: "Latent-confounded TemporalPag natural mediation is outside the \
                              existing cross-world identification theory",
                });
            }
            let q = temporal_class_identify_query(query, "TemporalPag")?;
            let mut envelope = identify_temporal_pag(strategy, pag, &q)?;
            refine_temporal_class_identification(&mut envelope, query, q.horizon_steps)?;
            audit_temporal_class_horizons(&mut envelope, structure, query, strategy, &q)?;
            Ok(Identification::TemporalEnvelope { envelope, strategy, structure_version })
        }
    }
}

fn static_identify_query(query: &CausalQuery) -> CausalQuery {
    match query {
        CausalQuery::ConditionalEffect(q) => CausalQuery::AverageEffect(q.inner.clone()),
        other => other.clone(),
    }
}

fn temporal_class_identify_query(
    query: &CausalQuery,
    class_tag: &str,
) -> Result<TemporalEffectQuery, CausalError> {
    match query {
        CausalQuery::TemporalEffect(q) => Ok(q.clone()),
        CausalQuery::Response(q) => {
            let temporal = q.temporal.as_ref().ok_or(CausalError::Unsupported {
                message: "temporal class response identification requires TemporalResponseSpec",
            })?;
            let (treatment, outcome) =
                q.functional.primary_pair().ok_or_else(|| CausalError::Compile {
                    message: format!("{class_tag} response has no treatment/outcome pair"),
                })?;
            Ok(TemporalEffectQuery {
                treatment,
                outcome,
                policy: temporal.policy.clone(),
                control: Intervention::set(treatment, Value::f64(0.0)),
                active: Intervention::set(treatment, Value::f64(1.0)),
                horizon_steps: temporal.horizons.first().copied().unwrap_or(1),
                max_history_lag: temporal.max_history_lag,
                target_population: q.target_population.clone(),
            })
        }
        CausalQuery::Mediation(q) if class_tag == "TemporalCpdag" => {
            let mut witness = TemporalEffectQuery::pulse(q.treatment, q.outcome, 1.0);
            witness.horizon_steps = q.horizons.first().copied().unwrap_or(1);
            Ok(witness)
        }
        _ => Err(CausalError::Unsupported {
            message: "temporal class identification supports TemporalEffect, temporal Response, \
                      and TemporalCpdag TemporalMediationEffect",
        }),
    }
}

fn class_ate_witness(
    query: &CausalQuery,
    class_tag: &str,
) -> Result<AverageEffectQuery, CausalError> {
    match query {
        CausalQuery::AverageEffect(q) => Ok(q.clone()),
        CausalQuery::Response(q) => crate::analysis::response_witness_ate(q),
        CausalQuery::ConditionalEffect(q) => Ok(q.inner.clone()),
        _ => Err(CausalError::Compile {
            message: format!(
                "{class_tag} identification supports AverageEffect, static Response, \
                 and ConditionalEffect"
            ),
        }),
    }
}

/// Class-appropriate default identifier, reusing the existing `DEFAULT_*_IDENTIFIER_ID`
/// constants rather than hardcoding wire strings.
fn default_strategy(class: GraphClass) -> IdentifierId {
    match class {
        GraphClass::Dag => DEFAULT_IDENTIFIER_ID,

        GraphClass::Admg => DEFAULT_ADMG_IDENTIFIER_ID,
        GraphClass::TemporalDag => IdentifierId::TemporalBackdoorUnfolded,
        GraphClass::Cpdag
        | GraphClass::Pag
        | GraphClass::TemporalCpdag
        | GraphClass::TemporalPag => DEFAULT_PAG_IDENTIFIER_ID,
    }
}

fn audit_temporal_class_horizons(
    primary: &mut TemporalClassEnvelope,
    structure: &AcceptedGraph,
    query: &CausalQuery,
    strategy: IdentifierId,
    witness: &TemporalEffectQuery,
) -> Result<(), CausalError> {
    let horizons: &[u32] = match query {
        CausalQuery::Response(response) => {
            response.temporal.as_ref().map_or(&[], |temporal| temporal.horizons.as_ref())
        }
        CausalQuery::Mediation(mediation) => &mediation.horizons,
        _ => return Ok(()),
    };
    for &horizon in horizons {
        if horizon == witness.horizon_steps {
            continue;
        }
        let mut qh = witness.clone();
        qh.horizon_steps = horizon;
        let mut local = if let Some(graph) = structure.as_temporal_cpdag() {
            identify_temporal_cpdag(strategy, graph, &qh)?
        } else {
            identify_temporal_pag(
                strategy,
                structure.as_temporal_pag().expect("temporal class"),
                &qh,
            )?
        };
        refine_temporal_class_identification(&mut local, query, horizon)?;
        primary.envelope.truncated_completions =
            primary.envelope.truncated_completions.max(local.envelope.truncated_completions);
        for case in &mut primary.envelope.cases {
            let local_case = local
                .envelope
                .cases
                .iter()
                .find(|other| other.graph.fingerprint() == case.graph.fingerprint());
            if local_case.is_none_or(|other| {
                !strategy_table::identification_status_acceptable(other.result.status)
            }) {
                case.result.status = antecedent_core::IdentificationStatus::GraphDependent;
                case.result.estimands.clear();
            }
        }
    }
    let mut rebuilt = IdentificationEnvelope::from_cases(primary.envelope.cases.clone());
    rebuilt.truncated_completions =
        rebuilt.truncated_completions.max(primary.envelope.truncated_completions);
    rebuilt.push_features(primary.envelope.critical_graph_features.clone());
    if rebuilt.truncated_completions > 0 {
        rebuilt.invariant = None;
        if rebuilt.status == antecedent_core::IdentificationStatus::NonparametricallyIdentified {
            rebuilt.status = antecedent_core::IdentificationStatus::PartiallyIdentified;
        }
    }
    primary.envelope = rebuilt;
    Ok(())
}

/// Replace the enumeration witness with the actual requested functional.
/// Enumeration supplies structures, never an identification certificate for a
/// different intervention schedule or mediation contrast.
pub(crate) fn refine_temporal_class_identification(
    envelope: &mut TemporalClassEnvelope,
    query: &CausalQuery,
    horizon: u32,
) -> Result<(), CausalError> {
    let schedule = if let CausalQuery::Response(response) = query {
        antecedent_estimate::plan_from_response_query(response)
            .map_err(CausalError::from)?
            .and_then(|plan| plan.mechanism_overlays())
            .map(|overlays| {
                overlays
                    .iter()
                    .map(|overlay| (overlay.node.variable, overlay.node.offset))
                    .collect::<Vec<_>>()
            })
    } else {
        None
    };
    if schedule.is_none() && !matches!(query, CausalQuery::Mediation(_)) {
        return Ok(());
    }
    for (case, indexer) in envelope.envelope.cases.iter_mut().zip(&mut envelope.indexers) {
        let Some(dag) = case.graph.sequential_dag() else {
            continue;
        };
        let identified = match query {
            CausalQuery::Response(response) => {
                let temporal = response.temporal.as_ref().expect("temporal response");
                let (_, outcome) = response.functional.primary_pair().ok_or_else(|| {
                    CausalError::Compile { message: "response requires primary pair".into() }
                })?;
                TemporalBackdoorIdentifier::new()
                    .identify_temporal_schedule(
                        &dag,
                        outcome,
                        i32::try_from(horizon.saturating_sub(1)).unwrap_or(i32::MAX),
                        schedule.as_deref().expect("schedule"),
                        temporal.max_history_lag,
                        response.target_population.clone(),
                    )
                    .map(|identified| (identified.result, identified.indexer))
            }
            CausalQuery::Mediation(mediation) => antecedent_identify::TemporalMediationIdentifier {
                allow_natural_controlled_alias: true,
                ..antecedent_identify::TemporalMediationIdentifier::new()
            }
            .identify_with_horizon(&dag, mediation, horizon)
            .map(|(result, temporal)| (result, temporal.indexer)),
            _ => unreachable!(),
        };
        match identified {
            Ok((result, coordinates)) => {
                case.result = result;
                *indexer = coordinates;
            }
            Err(antecedent_identify::IdentificationError::UnknownVariable { .. }) => {
                case.result.status = antecedent_core::IdentificationStatus::NotIdentified;
                case.result.estimands.clear();
            }
            Err(error) => return Err(error.into()),
        }
    }
    let old = &envelope.envelope;
    let mut rebuilt = IdentificationEnvelope::from_cases(old.cases.clone());
    rebuilt.truncated_completions = rebuilt.truncated_completions.max(old.truncated_completions);
    rebuilt.push_features(old.critical_graph_features.clone());
    if rebuilt.truncated_completions > 0 {
        rebuilt.invariant = None;
        if rebuilt.status == antecedent_core::IdentificationStatus::NonparametricallyIdentified {
            rebuilt.status = antecedent_core::IdentificationStatus::PartiallyIdentified;
        }
    }
    envelope.envelope = rebuilt;
    Ok(())
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
            Identification::Envelope { .. }
            | Identification::CpdagEnvelope { .. }
            | Identification::TemporalEnvelope { .. } => {
                panic!("expected Point for a Dag structure")
            }
        }
        assert!(identification.is_identified());
        assert_eq!(identification.structure_version(), 1);
        assert_eq!(identification.strategy(), DEFAULT_IDENTIFIER_ID);
    }

    #[test]
    fn identify_refuses_pulse_on_static_dag() {
        let structure = AcceptedGraph::from(toy_dag());
        let query = CausalQuery::TemporalEffect(antecedent_core::TemporalEffectQuery::pulse(
            antecedent_core::VariableId::from_raw(0),
            antecedent_core::VariableId::from_raw(1),
            1.0,
        ));
        let err = identify(&structure, &query).unwrap_err();
        assert!(
            matches!(
                err,
                CausalError::Support { id: crate::support::SupportRefusal::NotApplicable, .. }
            ),
            "{err}"
        );
    }

    #[test]
    fn identify_cpdag_average_effect_returns_envelope() {
        let mut cpdag = antecedent_graph::Cpdag::with_variables(3);
        cpdag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        cpdag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
        cpdag.insert_undirected(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
        let structure = AcceptedGraph::from(cpdag);
        let query = CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(
            antecedent_core::VariableId::from_raw(1),
            antecedent_core::VariableId::from_raw(2),
        ));
        let identification = identify(&structure, &query).unwrap();
        match identification {
            Identification::CpdagEnvelope { .. } => {}
            other => panic!("expected CpdagEnvelope, got {other:?}"),
        }
        assert_eq!(identification.strategy(), DEFAULT_PAG_IDENTIFIER_ID);
    }

    #[test]
    fn identify_cpdag_response_uses_ate_witness() {
        use antecedent_core::{ContinuousDomain, GridSpec, ResponseFunctional, ResponseQuery};
        let mut cpdag = antecedent_graph::Cpdag::with_variables(3);
        cpdag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        cpdag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
        cpdag.insert_undirected(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
        let structure = AcceptedGraph::from(cpdag);
        let query = CausalQuery::Response(ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: antecedent_core::VariableId::from_raw(2),
            treatment: ContinuousDomain::new(
                antecedent_core::VariableId::from_raw(1),
                GridSpec::Values(Arc::from([0.0, 1.0])),
            ),
        }));
        let identification = identify(&structure, &query).unwrap();
        match identification {
            Identification::CpdagEnvelope { .. } => {}
            other => panic!("expected CpdagEnvelope, got {other:?}"),
        }
    }

    #[test]
    fn identify_temporal_dag_pulse_is_point() {
        use antecedent_core::Lag;
        use antecedent_graph::TemporalDag;
        let mut dag = TemporalDag::empty();
        let t_lag =
            dag.add_lagged(antecedent_core::VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let y_now =
            dag.add_lagged(antecedent_core::VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        dag.insert_directed(t_lag, y_now).unwrap();
        let structure = AcceptedGraph::from(dag);
        let query = CausalQuery::TemporalEffect(antecedent_core::TemporalEffectQuery::pulse(
            antecedent_core::VariableId::from_raw(0),
            antecedent_core::VariableId::from_raw(1),
            1.0,
        ));
        let identification = identify(&structure, &query).unwrap();
        match identification {
            Identification::Point { .. } => {}
            other => panic!("expected Point for TemporalDag, got {other:?}"),
        }
        assert_eq!(identification.strategy(), IdentifierId::TemporalBackdoorUnfolded);
    }
}
