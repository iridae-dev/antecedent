//! Structure-and-query identification without data, for staged tooling.
//!
//! [`Identification`] is what a compiled study derives for a structure and a query. It is
//! reached through [`crate::StudyBuilder::identify_structure`], which builds the study over a
//! table with no rows (identification reads structure and query, never data); an analysis goes
//! through `Study::compile`, `Study::prepare` or `Study::run`, which retain the identification
//! product. No raw identification entry is public.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
use std::sync::Arc;

use antecedent_core::{
    AverageEffectQuery, CausalQuery, Diagnostic, DiagnosticKind, DiagnosticSeverity, Intervention,
    TemporalEffectQuery, TemporalIndexer, Value,
};
use antecedent_graph::Pag;
use antecedent_identify::{
    IdentificationEnvelope, IdentificationResult, IdentifiedEstimand, TemporalBackdoorIdentifier,
    TemporalClassEnvelope,
};

use crate::accepted::{AcceptedGraph, GraphClass, GraphRef};
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

/// Identify `query` against `structure` for a compiled study that carries no rows.
///
/// `identifier` is the caller's explicit choice; without one the class-appropriate default
/// applies. `source` is how the study received its structure (accepted graph or explicit
/// graph). Reached through [`crate::StudyBuilder::identify_structure`] and
/// [`crate::Study::identify_structure`]: identification is a function of structure and query,
/// and the compiled study is the entry, so no raw entry is public.
///
/// # Errors
///
/// Unsupported graph-class/query pair, strategy incompatible with the graph class, or
/// identification failure.
pub(crate) fn identify_study(
    structure: &AcceptedGraph,
    query: &CausalQuery,
    identifier: Option<IdentifierId>,
    source: crate::support::StructureSource,
) -> Result<Identification, CausalError> {
    let strategy = identifier.unwrap_or_else(|| match structure.class() {
        GraphClass::Dag => dag_default_strategy(query),
        class => default_strategy(class),
    });
    identify_with_source(structure, query, strategy, source)
}

/// The default-identifier identification of an accepted graph, for the module's own tests.
#[cfg(test)]
fn identify(structure: &AcceptedGraph, query: &CausalQuery) -> Result<Identification, CausalError> {
    identify_study(structure, query, None, crate::support::StructureSource::Accepted)
}

fn identify_with_source(
    structure: &AcceptedGraph,
    query: &CausalQuery,
    strategy: IdentifierId,
    source: crate::support::StructureSource,
) -> Result<Identification, CausalError> {
    let structure_version = structure.version();
    crate::support::refuse_undeclared_off_axis(query)?;
    if let Some(cell) = crate::support::support_cell(
        query,
        crate::support::effective_graph_class(structure, query),
        source,
        &crate::inference::InferenceMode::Frequentist,
        crate::analysis::RefuteSuite::None,
    ) {
        crate::support::refuse_if_not_applicable(cell)?;
    }
    match structure.graph_ref() {
        GraphRef::Dag(dag) => identify_source_dag(dag, query, strategy, structure_version),
        GraphRef::Cpdag(cpdag) => identify_source_cpdag(cpdag, query, strategy, structure_version),
        GraphRef::Pag(pag) => identify_source_pag(pag, query, strategy, structure_version),
        GraphRef::Admg(admg) => identify_source_admg(admg, query, strategy, structure_version),
        GraphRef::TemporalAdmg(temporal) => {
            identify_source_temporal_admg(temporal, query, strategy, structure_version)
        }
        GraphRef::TemporalDag(dag) => {
            identify_source_temporal_dag(dag, query, strategy, structure_version)
        }
        GraphRef::TemporalCpdag(cpdag) => {
            identify_source_temporal_cpdag(structure, cpdag, query, strategy, structure_version)
        }
        GraphRef::TemporalPag(pag) => {
            identify_source_temporal_pag(structure, pag, query, strategy, structure_version)
        }
        GraphRef::RandomizedTrial => Err(CausalError::Unsupported {
            message: "randomized trial queries are identified by the dedicated experiment route",
        }),
    }
}

fn identify_source_dag(
    dag: &antecedent_graph::Dag,
    query: &CausalQuery,
    strategy: IdentifierId,
    structure_version: u32,
) -> Result<Identification, CausalError> {
    let id_query = static_identify_query(query);
    let result = identify_static_query(strategy, dag, &id_query)?;
    Ok(Identification::Point { result, temporal_indexer: None, strategy, structure_version })
}

fn identify_source_cpdag(
    cpdag: &antecedent_graph::Cpdag,
    query: &CausalQuery,
    strategy: IdentifierId,
    structure_version: u32,
) -> Result<Identification, CausalError> {
    let envelope = if let CausalQuery::Response(response) = query {
        strategy_table::identify_cpdag_response(strategy, cpdag, response)?
    } else {
        identify_cpdag(strategy, cpdag, &class_ate_witness(query, "CPDAG")?)?
    };
    Ok(Identification::CpdagEnvelope { envelope, strategy, structure_version })
}

fn identify_source_pag(
    pag: &antecedent_graph::Pag,
    query: &CausalQuery,
    strategy: IdentifierId,
    structure_version: u32,
) -> Result<Identification, CausalError> {
    let envelope = if let CausalQuery::Response(response) = query {
        strategy_table::identify_pag_response(strategy, pag, response)?
    } else {
        identify_pag(strategy, pag, &class_ate_witness(query, "PAG")?)?
    };
    Ok(Identification::Envelope { envelope, strategy, structure_version })
}

fn identify_source_admg(
    admg: &antecedent_graph::Admg,
    query: &CausalQuery,
    strategy: IdentifierId,
    structure_version: u32,
) -> Result<Identification, CausalError> {
    let CausalQuery::AverageEffect(average_effect) = query else {
        return Err(CausalError::Unsupported {
            message: "ADMG identification supports only CausalQuery::AverageEffect",
        });
    };
    let result = identify_admg(strategy, admg, average_effect)?;
    Ok(Identification::Point { result, temporal_indexer: None, strategy, structure_version })
}

fn identify_source_temporal_admg(
    temporal: &antecedent_graph::TemporalAdmg,
    query: &CausalQuery,
    strategy: IdentifierId,
    structure_version: u32,
) -> Result<Identification, CausalError> {
    let admg = temporal.project_preserving_functional().map_err(CausalError::rejected)?;
    let CausalQuery::AverageEffect(average_effect) = query else {
        return Err(CausalError::Unsupported {
            message: "TemporalAdmg identification supports only CausalQuery::AverageEffect \
                      through the static ADMG projection",
        });
    };
    let result = identify_admg(strategy, &admg, average_effect)?;
    Ok(Identification::Point { result, temporal_indexer: None, strategy, structure_version })
}

fn identify_source_temporal_dag(
    dag: &antecedent_graph::TemporalDag,
    query: &CausalQuery,
    strategy: IdentifierId,
    structure_version: u32,
) -> Result<Identification, CausalError> {
    let CausalQuery::TemporalEffect(q) = query else {
        return Err(CausalError::Unsupported {
            message: "TemporalDag identification supports only CausalQuery::TemporalEffect",
        });
    };
    let identified = TemporalBackdoorIdentifier::new()
        .with_parent_adjustment_fallback()
        .identify_temporal(dag, q)
        .map_err(CausalError::from)?;
    Ok(Identification::Point {
        result: identified.result,
        temporal_indexer: Some(identified.indexer),
        strategy,
        structure_version,
    })
}

fn identify_source_temporal_cpdag(
    structure: &AcceptedGraph,
    cpdag: &antecedent_graph::TemporalCpdag,
    query: &CausalQuery,
    strategy: IdentifierId,
    structure_version: u32,
) -> Result<Identification, CausalError> {
    let q = temporal_class_identify_query(query, "TemporalCpdag")?;
    let mut envelope = identify_temporal_cpdag(strategy, cpdag, &q)?;
    refine_temporal_class_identification(&mut envelope, query, q.horizon_steps)?;
    audit_temporal_class_horizons(&mut envelope, structure, query, strategy, &q)?;
    Ok(Identification::TemporalEnvelope { envelope, strategy, structure_version })
}

fn identify_source_temporal_pag(
    structure: &AcceptedGraph,
    pag: &antecedent_graph::TemporalPag,
    query: &CausalQuery,
    strategy: IdentifierId,
    structure_version: u32,
) -> Result<Identification, CausalError> {
    if let CausalQuery::Mediation(mediation) = query {
        return identify_temporal_pag_mediation(pag, mediation, strategy, structure_version);
    }
    let q = temporal_class_identify_query(query, "TemporalPag")?;
    let mut envelope = identify_temporal_pag(strategy, pag, &q)?;
    refine_temporal_class_identification(&mut envelope, query, q.horizon_steps)?;
    audit_temporal_class_horizons(&mut envelope, structure, query, strategy, &q)?;
    Ok(Identification::TemporalEnvelope { envelope, strategy, structure_version })
}

/// Natural mediation under a temporal PAG: the envelope of the window's completions, each
/// read through the ADMGs sharing its MAG. The window is numbered over the variables the
/// template and the query name, the schema a study with rows takes from its series.
fn identify_temporal_pag_mediation(
    pag: &antecedent_graph::TemporalPag,
    mediation: &antecedent_core::MediationQuery,
    strategy: IdentifierId,
    structure_version: u32,
) -> Result<Identification, CausalError> {
    mediation.validate()?;
    let [horizon] = mediation.horizons.as_ref() else {
        return Err(CausalError::Unsupported {
            message: "mediation under a temporal PAG reads one outcome horizon; next attempt: \
                      one study per horizon",
        });
    };
    let mut effect =
        antecedent_core::TemporalEffectQuery::pulse(mediation.treatment, mediation.outcome, 1.0)
            .with_horizon_steps(*horizon);
    effect.control = mediation.control.clone();
    effect.active = mediation.active.clone();
    effect.target_population = mediation.target_population.clone();
    let largest = pag
        .nodes()
        .iter()
        .filter_map(|node| match node {
            antecedent_graph::NodeRef::Lagged { variable, .. } => Some(variable.raw()),
            _ => None,
        })
        .chain([mediation.treatment.raw(), mediation.outcome.raw()])
        .chain(mediation.mediators.iter().map(|mediator| mediator.raw()))
        .max()
        .unwrap_or(0);
    let window =
        antecedent_identify::temporal_pag_mediation_window(pag, &effect, mediation, largest + 1)?;
    let envelope = antecedent_identify::GeneralizedAdjustmentIdentifier::new()
        .identify_temporal_pag_path_specific_envelope(pag, &window)?;
    Ok(Identification::TemporalEnvelope { envelope, strategy, structure_version })
}

fn static_identify_query(query: &CausalQuery) -> CausalQuery {
    match query {
        CausalQuery::ConditionalEffect(q) => CausalQuery::AverageEffect(q.inner.clone()),
        CausalQuery::NestedCounterfactual(q) => CausalQuery::Mediation(q.as_mediation_query()),
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
                crate::analysis::proof_support::require_primary_pair(&q.functional)?;
            Ok(TemporalEffectQuery {
                treatment,
                outcome,
                policy: temporal.policy.clone(),
                control: Intervention::set(treatment, Value::f64(0.0)),
                active: Intervention::set(treatment, Value::f64(1.0)),
                horizon_steps: temporal.horizons.first().copied().unwrap_or(1),
                max_history_lag: temporal.max_history_lag,
                target_population: q.target_population.clone(),
                effect_modifiers: std::sync::Arc::from([]),
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

/// Default identifier for `query` on a DAG: the one table behind [`identify`] and
/// [`identify_dag`].
fn dag_default_strategy(query: &CausalQuery) -> IdentifierId {
    match query {
        CausalQuery::Counterfactual(_) => IdentifierId::GcmParametric,
        CausalQuery::Response(_) => IdentifierId::ResponseBackdoor,
        CausalQuery::Mediation(_) | CausalQuery::PathSpecific(_) => {
            IdentifierId::PathSpecificNatural
        }
        CausalQuery::NestedCounterfactual(_) => IdentifierId::PathSpecificNatural,
        CausalQuery::Distribution(_) => IdentifierId::GeneralId,
        _ => default_strategy(GraphClass::Dag),
    }
}

/// Class-appropriate default identifier, reusing the existing `DEFAULT_*_IDENTIFIER_ID`
/// constants rather than hardcoding wire strings.
fn default_strategy(class: GraphClass) -> IdentifierId {
    match class {
        GraphClass::Dag => DEFAULT_IDENTIFIER_ID,

        GraphClass::Admg | GraphClass::TemporalAdmg => DEFAULT_ADMG_IDENTIFIER_ID,
        GraphClass::TemporalDag => IdentifierId::TemporalBackdoorUnfolded,
        GraphClass::Cpdag
        | GraphClass::Pag
        | GraphClass::TemporalCpdag
        | GraphClass::TemporalPag => DEFAULT_PAG_IDENTIFIER_ID,
        GraphClass::RandomizedTrial => IdentifierId::RandomizedDesign,
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
        let mut local = match structure.graph_ref() {
            GraphRef::TemporalCpdag(graph) => identify_temporal_cpdag(strategy, graph, &qh)?,
            GraphRef::TemporalPag(graph) => identify_temporal_pag(strategy, graph, &qh)?,
            _ => {
                return Err(CausalError::Compile {
                    message: "temporal class horizon audit requires a TemporalCpdag or \
                              TemporalPag"
                        .into(),
                });
            }
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
        match antecedent_estimate::plan_from_response_query(response).map_err(CausalError::from)? {
            Some(plan) if plan.mechanism_overlays().is_some() => {
                let temporal = response.temporal.as_ref().ok_or_else(|| CausalError::Compile {
                    message: "sequence schedule requires TemporalResponseSpec".into(),
                })?;
                Some(plan.identification_schedule(temporal))
            }
            _ => None,
        }
    } else {
        None
    };
    if schedule.is_none() && !matches!(query, CausalQuery::Mediation(_)) {
        return Ok(());
    }
    for (case, indexer) in envelope.envelope.cases.iter_mut().zip(&mut envelope.indexers) {
        let Some(dag) = case.graph.sequential_dag() else {
            // The enumeration witness certified Pulse/Sustained adjustment, not the
            // requested schedule or mediation functional. Those are certified only
            // on directed completions; a bidirected member keeps its mass without
            // inheriting a certificate for a functional nobody examined.
            case.result.status = antecedent_core::IdentificationStatus::GraphDependent;
            case.result.estimands.clear();
            case.result.diagnostics.push(Diagnostic::new(
                "identify.temporal_class.bidirected_completion_uncertified",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "completion has bidirected edges; the requested sequential schedule or \
                 mediation functional is certified only on directed completions",
            ));
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
