//! Conservative structural transportability identification.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{ResponseFunctional, TransportOutcome, TransportQuery, VariableId};
use antecedent_graph::{Admg, DSeparationWorkspace, DenseNodeId, SelectionDiagram};

use crate::IdentificationError;

/// Population-labelled symbolic distribution required by a transport formula.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PopulationFactor {
    /// Population key.
    pub population: Arc<str>,
    /// Supplied evidence regime used by this factor, when explicitly catalogued.
    pub regime: Option<antecedent_core::RegimeId>,
    /// Variables whose conditional law is required.
    pub variables: Arc<[VariableId]>,
    /// Conditioning variables.
    pub conditioned_on: Arc<[VariableId]>,
    /// Variables experimentally intervened on in this population.
    pub interventions: Arc<[VariableId]>,
}

/// A sound, population-labelled transport formula.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TransportFormula {
    /// The source experimental response is invariant and can be used directly.
    Direct(PopulationFactor),
    /// Standardize a source conditional experimental response over a target covariate law.
    Standardize {
        /// Variables marginalized after multiplying the two factors.
        over: Arc<[VariableId]>,
        /// Source conditional experimental response.
        source_response: PopulationFactor,
        /// Target standardization law.
        target_law: PopulationFactor,
    },
    /// Truncated recursive factorization for a singleton-district (causally sufficient) graph.
    RecursiveFactorization {
        /// Variables marginalized from the product.
        sum_out: Arc<[VariableId]>,
        /// Topologically ordered population-labelled factors.
        factors: Arc<[PopulationFactor]>,
    },
}

/// Positive certificate explaining why the emitted formula is valid.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransportCertificate {
    /// Stable rule id.
    pub rule: Arc<str>,
    /// Selection variables used by the rule.
    pub selection_targets: Arc<[VariableId]>,
    /// Human-readable premises checked by the implementation.
    pub premises: Arc<[Arc<str>]>,
}

/// Inconclusive certificate: no implemented derivation was certified.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NotCertifiedCertificate {
    /// Stable failure id.
    pub reason: Arc<str>,
    /// Variables witnessing the failed criterion.
    pub witness: Arc<[VariableId]>,
    /// Scope note: refusal is not a completeness claim outside implemented rules.
    pub message: Arc<str>,
}

/// Required available evidence was absent. Distinct from [`NotCertifiedCertificate`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MissingEvidenceCertificate {
    /// Stable reason id.
    pub reason: Arc<str>,
    /// Variables or regimes that were required and missing.
    pub missing: Arc<[VariableId]>,
    /// Scope note. Callers must not parse this prose.
    pub message: Arc<str>,
}

/// Result of structural transport identification.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TransportIdentification {
    /// A sound symbolic formula and its checked premises.
    Transportable {
        /// Formula.
        formula: TransportFormula,
        /// Positive certificate.
        certificate: TransportCertificate,
    },
    /// No implemented sound rule applies. Historical meaning preserved.
    NotCertified(NotCertifiedCertificate),
    /// A required available source regime is absent.
    MissingEvidence(MissingEvidenceCertificate),
}

impl TransportIdentification {
    /// Typed outcome. Callers match [`TransportOutcome::kind`]; they must not parse prose.
    #[must_use]
    pub fn outcome(&self) -> TransportOutcome {
        match self {
            Self::Transportable { certificate, .. } => {
                TransportOutcome::identified(Arc::clone(&certificate.rule))
            }
            Self::NotCertified(certificate) => TransportOutcome::not_certified(
                Arc::clone(&certificate.reason),
                Arc::clone(&certificate.witness),
            ),
            Self::MissingEvidence(certificate) => TransportOutcome::missing_evidence(
                Arc::clone(&certificate.reason),
                Arc::clone(&certificate.missing),
            ),
        }
    }

    /// Whether a sound formula was certified.
    #[must_use]
    pub const fn is_transportable(&self) -> bool {
        matches!(self, Self::Transportable { .. })
    }
}

/// Conservative sID-style identifier covering direct transport, general pre-treatment
/// S-admissible standardization, and recursive singleton c-components.
#[derive(Clone, Debug, Default)]
pub struct TransportIdentifier;

impl TransportIdentifier {
    /// Construct an identifier.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Identify a structural transport formula.
    ///
    /// Target-only identification is attempted before any source experiment is
    /// demanded. An empty source catalog on a causally sufficient graph emits
    /// the target observational truncated factorization. A required missing source regime is
    /// [`TransportIdentification::MissingEvidence`], not [`TransportIdentification::NotCertified`].
    ///
    /// The implementation deliberately returns [`Self::NotCertified`] rather than claiming
    /// non-transportability when general multi-node c-component recursion is required.
    #[allow(clippy::too_many_lines)]
    pub fn identify(
        &self,
        diagram: &SelectionDiagram,
        query: &TransportQuery,
    ) -> Result<TransportIdentification, IdentificationError> {
        let result = Self::identify_structural(diagram, query)?;
        let bound = bind_catalog(result, query);
        if matches!(bound, TransportIdentification::MissingEvidence(_))
            && diagram.causal_graph().district_count() == diagram.causal_graph().node_count()
        {
            let mut target_query = query.clone();
            target_query.source_experiments = Arc::from([]);
            target_query.catalog = None;
            return Ok(bind_catalog(Self::identify_structural(diagram, &target_query)?, query));
        }
        Ok(bound)
    }

    #[allow(clippy::too_many_lines)]
    fn identify_structural(
        diagram: &SelectionDiagram,
        query: &TransportQuery,
    ) -> Result<TransportIdentification, IdentificationError> {
        query.validate().map_err(|error| IdentificationError::msg(error.to_string()))?;
        let graph = diagram.causal_graph();
        let (outcomes, treatments) = response_variables(query)?;
        let mut coordinates = outcomes
            .iter()
            .chain(treatments.iter())
            .chain(query.source_experiments.iter())
            .copied()
            .collect::<Vec<_>>();
        if let Some(catalog) = &query.catalog {
            for environment in catalog.environments.iter() {
                coordinates.extend(environment.variables.iter().map(|v| v.variable));
                coordinates.extend(environment.selection_targets.iter().copied());
            }
            for regime in catalog.regimes.iter() {
                coordinates
                    .extend(regime.measured.iter().chain(regime.interventions.iter()).copied());
            }
        }
        for variable in &coordinates {
            if variable.as_usize() >= graph.node_count() {
                return Err(IdentificationError::UnknownVariable { id: *variable });
            }
        }

        // One reachability workspace for every ancestry probe in this identify
        // call; `reaches()` would otherwise allocate a fresh workspace per pair.
        let mut reach_ws = antecedent_graph::GraphWorkspace::default();
        let mut outcome_ancestors = std::collections::BTreeSet::new();
        let mut pending = outcomes.to_vec();
        while let Some(variable) = pending.pop() {
            if treatments.contains(&variable) || !outcome_ancestors.insert(variable) {
                continue;
            }
            pending.extend(
                graph.parents(dense(variable)).iter().map(|p| VariableId::from_raw(p.raw())),
            );
        }
        let relevant_selection = diagram
            .selection_targets()
            .iter()
            .copied()
            .filter(|v| outcome_ancestors.contains(v))
            .collect::<Vec<_>>();
        if relevant_selection.is_empty() && source_experiments_cover(query, &treatments) {
            let population = Arc::clone(&query.source_population);
            let factor = response_factor(population, &outcomes, &[], &treatments);
            return Ok(TransportIdentification::Transportable {
                formula: TransportFormula::Direct(factor),
                certificate: TransportCertificate {
                    rule: Arc::from("transport.sid.direct"),
                    selection_targets: Arc::from([]),
                    premises: Arc::from([Arc::from(
                        "no population-varying mechanism is an ancestor of a requested outcome",
                    )]),
                },
            });
        }
        // In a causally sufficient graph, target observational factors identify
        // the intervention by the truncated factorization, regardless of selection.
        if !source_experiments_cover(query, &treatments)
            && graph.district_count() == graph.node_count()
        {
            let order = topological_order(diagram)?
                .into_iter()
                .filter(|v| outcome_ancestors.contains(v))
                .collect::<Vec<_>>();
            let factors = order
                .iter()
                .copied()
                .filter(|v| !treatments.contains(v))
                .map(|v| {
                    let parents = graph
                        .parents(dense(v))
                        .iter()
                        .map(|p| VariableId::from_raw(p.raw()))
                        .collect::<Vec<_>>();
                    response_factor(Arc::clone(&query.target_population), &[v], &parents, &[])
                })
                .collect::<Vec<_>>();
            let sum_out = order
                .iter()
                .copied()
                .filter(|v| !outcomes.contains(v) && !treatments.contains(v))
                .collect::<Vec<_>>();
            return Ok(TransportIdentification::Transportable {
                formula: TransportFormula::RecursiveFactorization {
                    sum_out: sum_out.into(),
                    factors: factors.into(),
                },
                certificate: TransportCertificate {
                    rule: Arc::from("transport.sid.target_g_formula"),
                    selection_targets: diagram.selection_targets().to_vec().into(),
                    premises: Arc::from([Arc::from(
                        "causal sufficiency licenses the target observational truncated factorization",
                    )]),
                },
            });
        }
        if !source_experiments_cover(query, &treatments) {
            return Ok(TransportIdentification::MissingEvidence(MissingEvidenceCertificate {
                reason: Arc::from("transport.source_experiment_missing"),
                missing: treatments,
                message: Arc::from(
                    "the source experiment required by this transport query is unavailable",
                ),
            }));
        }

        // This is intentionally stronger than general S-admissibility. Requiring selected
        // standardizers to be causally prior, exogenous, and district-singleton prevents
        // conditioning from opening S -> Z <- U -> Y paths.
        //
        // A requested outcome can never be a standardizer: `reaches` is reflexive, so an
        // outcome that is itself a selection target lands in `relevant_selection`, and
        // standardizing over it would emit P(y | do(x), y) and marginalize away the very
        // quantity the query asks for. Such a diagram falls through to the S-admissible
        // search and then to `NotCertified`.
        let safe_standardizers = !relevant_selection.iter().any(|z| outcomes.contains(z))
            && relevant_selection.iter().all(|z| {
                let dz = dense(*z);
                !treatments.iter().any(|x| graph.reaches_with(dense(*x), dz, &mut reach_ws))
                    && graph.parents(dz).is_empty()
                    && graph.bidirected_neighbors(dz).is_empty()
            });
        if safe_standardizers {
            let over: Arc<[VariableId]> = relevant_selection.clone().into();
            return Ok(TransportIdentification::Transportable {
                formula: TransportFormula::Standardize {
                    over: Arc::clone(&over),
                    source_response: response_factor(
                        Arc::clone(&query.source_population),
                        &outcomes,
                        &over,
                        &treatments,
                    ),
                    target_law: response_factor(
                        Arc::clone(&query.target_population),
                        &over,
                        &[],
                        &[],
                    ),
                },
                certificate: TransportCertificate {
                    rule: Arc::from("transport.sid.standardize"),
                    selection_targets: over,
                    premises: Arc::from([
                        Arc::from("standardizers are pre-treatment"),
                        Arc::from("standardizers are exogenous singleton districts"),
                    ]),
                },
            });
        }

        // Pearl & Bareinboim (2014), Theorem 2: if Z is S-admissible, then
        // P*(y | do(x)) = sum_z P(y | do(x), z) P*(z). We search all pre-treatment
        // observed subsets so that the target factor is observational P*(z), rather
        // than silently treating a post-treatment law as observational.
        if let Some(over) = s_admissible_standardizers(diagram, &outcomes, &treatments)? {
            return Ok(TransportIdentification::Transportable {
                formula: TransportFormula::Standardize {
                    over: Arc::clone(&over),
                    source_response: response_factor(
                        Arc::clone(&query.source_population),
                        &outcomes,
                        &over,
                        &treatments,
                    ),
                    target_law: response_factor(
                        Arc::clone(&query.target_population),
                        &over,
                        &[],
                        &[],
                    ),
                },
                certificate: TransportCertificate {
                    rule: Arc::from("transport.sid.s_admissible"),
                    selection_targets: diagram.selection_targets().to_vec().into(),
                    premises: Arc::from([
                        Arc::from(
                            "outcomes are m-separated from every selection node given the standardizers in the treatment-mutilated selection diagram",
                        ),
                        Arc::from("standardizers are observed non-descendants of every treatment"),
                    ]),
                },
            });
        }

        if graph.district_count() == graph.node_count() {
            let order = topological_order(diagram)?;
            let mut factors = Vec::new();
            for (position, variable) in order.iter().copied().enumerate() {
                if treatments.contains(&variable) {
                    continue;
                }
                let population = if diagram.mechanism_may_differ(variable) {
                    Arc::clone(&query.target_population)
                } else {
                    Arc::clone(&query.source_population)
                };
                factors.push(response_factor(population, &[variable], &order[..position], &[]));
            }
            let sum_out = order
                .iter()
                .copied()
                .filter(|v| !outcomes.contains(v) && !treatments.contains(v))
                .collect::<Vec<_>>();
            return Ok(TransportIdentification::Transportable {
                formula: TransportFormula::RecursiveFactorization {
                    sum_out: sum_out.into(),
                    factors: factors.into(),
                },
                certificate: TransportCertificate {
                    rule: Arc::from("transport.sid.singleton_c_components"),
                    selection_targets: diagram.selection_targets().to_vec().into(),
                    premises: Arc::from([
                        Arc::from("all c-components are singleton"),
                        Arc::from(
                            "population labels follow mechanism invariance in topological order",
                        ),
                    ]),
                },
            });
        }

        Ok(TransportIdentification::NotCertified(NotCertifiedCertificate {
            reason: Arc::from("transport.sid.multinode_c_component_not_implemented"),
            witness: relevant_selection.into(),
            message: Arc::from(
                "a population-varying outcome ancestor lies in a graph requiring general c-component recursion; no non-transportability claim is made",
            ),
        }))
    }
}

fn bind_catalog(
    mut result: TransportIdentification,
    query: &TransportQuery,
) -> TransportIdentification {
    let Some(catalog) = &query.catalog else {
        return result;
    };
    let TransportIdentification::Transportable { formula, .. } = &mut result else {
        return result;
    };
    let factors: Vec<&mut PopulationFactor> = match formula {
        TransportFormula::Direct(factor) => vec![factor],
        TransportFormula::Standardize { source_response, target_law, .. } => {
            vec![source_response, target_law]
        }
        TransportFormula::RecursiveFactorization { factors, .. } => {
            Arc::make_mut(factors).iter_mut().collect()
        }
    };
    let mut missing = std::collections::BTreeSet::new();
    for factor in factors {
        let target = factor.population == query.target_population;
        let sampling_valid = !target
            || catalog
                .target_sampling
                .is_none_or(antecedent_core::TargetSampling::represents_target_law);
        let needed = factor
            .variables
            .iter()
            .chain(factor.conditioned_on.iter())
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        let regime = catalog.regimes.iter().find(|r| {
            r.population == factor.population && r.evidence_kind.can_satisfy_factor()
                && r.conditioned_on.iter().all(|v| factor.conditioned_on.contains(v) && !factor.variables.contains(v))
                && r.interventions.len() == factor.interventions.len()
                && r.interventions.iter().all(|v| factor.interventions.contains(v))
                && needed.iter().all(|v| r.measured.contains(v) || r.interventions.contains(v))
                && (matches!(r.distribution, antecedent_core::DistributionAvailability::Joint)
                    || needed.len() <= 1)
                // Restricted assignments cannot supply an unrestricted symbolic response.
                && r.intervention_values.is_empty()
        });
        if sampling_valid {
            if let Some(regime) = regime {
                factor.regime = Some(regime.id);
                continue;
            }
            // An empty source catalog does not withdraw the target observational law.
            if target
                && catalog.target_sampling
                    != Some(antecedent_core::TargetSampling::LicensedWeightedDesign)
                && factor.interventions.is_empty()
                && !catalog.regimes.iter().any(|r| r.population == factor.population)
            {
                continue;
            }
        }
        missing.extend(needed);
    }
    if missing.is_empty() {
        result
    } else {
        TransportIdentification::MissingEvidence(MissingEvidenceCertificate {
            reason: Arc::from("transport.missing_evidence"),
            missing: missing.into_iter().collect::<Vec<_>>().into(),
            message: Arc::from(
                "required joint measured law, intervention regime, or representative target law is unavailable",
            ),
        })
    }
}

fn s_admissible_standardizers(
    diagram: &SelectionDiagram,
    outcomes: &[VariableId],
    treatments: &[VariableId],
) -> Result<Option<Arc<[VariableId]>>, IdentificationError> {
    let graph = diagram.causal_graph();
    let mut reach_ws = antecedent_graph::GraphWorkspace::default();
    let candidates = (0..graph.node_count())
        .filter_map(|i| u32::try_from(i).ok().map(VariableId::from_raw))
        .filter(|candidate| !outcomes.contains(candidate) && !treatments.contains(candidate))
        .filter(|candidate| {
            !treatments.iter().any(|treatment| {
                graph.reaches_with(dense(*treatment), dense(*candidate), &mut reach_ws)
            })
        })
        .collect::<Vec<_>>();
    // Exhaustive subset search is intentionally bounded. Larger diagrams retain the
    // already checked sufficient rules and fail closed instead of using a heuristic.
    if candidates.len() > 20 {
        return Ok(None);
    }
    let augmented = treatment_mutilated_selection_graph(diagram, treatments)?;
    let selection_offset = graph.node_count();
    let mut workspace = DSeparationWorkspace::default();
    // Selection-node ids are loop-invariant; precompute once.
    let selection_nodes: Vec<DenseNodeId> = (0..diagram.selection_targets().len())
        .map(|selection_index| {
            u32::try_from(selection_offset + selection_index).map(DenseNodeId::from_raw).map_err(
                |_| IdentificationError::msg("selection diagram exceeds u32 node capacity"),
            )
        })
        .collect::<Result<_, _>>()?;
    let k = candidates.len();
    let mut standardizers: Vec<VariableId> = Vec::with_capacity(k);
    let mut conditioned: Vec<DenseNodeId> = Vec::with_capacity(k);
    // Size-ascending, then numerically ascending within a size — the same
    // visiting order as the historical full 2^k scan with a popcount filter,
    // but enumerating only the C(k, size) masks per size (Gosper's hack).
    for size in 0..=k {
        let mut mask: u64 = if size == 0 { 0 } else { (1_u64 << size) - 1 };
        loop {
            standardizers.clear();
            conditioned.clear();
            for (i, variable) in candidates.iter().enumerate() {
                if (mask >> i) & 1 == 1 {
                    standardizers.push(*variable);
                    conditioned.push(dense(*variable));
                }
            }
            let mut separated = true;
            'outcomes: for outcome in outcomes {
                for &selection_node in &selection_nodes {
                    if !augmented
                        .is_m_separated(
                            dense(*outcome),
                            selection_node,
                            &conditioned,
                            &mut workspace,
                        )
                        .map_err(IdentificationError::from)?
                    {
                        separated = false;
                        break 'outcomes;
                    }
                }
            }
            if separated {
                return Ok(Some(standardizers.into()));
            }
            if size == 0 {
                break;
            }
            // Next-higher mask with the same popcount.
            let low = mask & mask.wrapping_neg();
            let ripple = mask + low;
            let next = (((ripple ^ mask) >> 2) / low) | ripple;
            if next >= (1_u64 << k) {
                break;
            }
            mask = next;
        }
    }
    Ok(None)
}

fn treatment_mutilated_selection_graph(
    diagram: &SelectionDiagram,
    treatments: &[VariableId],
) -> Result<Admg, IdentificationError> {
    let graph = diagram.causal_graph();
    let total = graph
        .node_count()
        .checked_add(diagram.selection_targets().len())
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| IdentificationError::msg("selection diagram exceeds u32 node capacity"))?;
    let mut out = Admg::with_variables(total);
    for i in 0..graph.node_count() {
        let from = DenseNodeId::from_raw(u32::try_from(i).expect("validated node capacity"));
        for &to in graph.children(from) {
            if !treatments.iter().any(|treatment| dense(*treatment) == to) {
                out.insert_directed(from, to).map_err(IdentificationError::from)?;
            }
        }
        for &other in graph.bidirected_neighbors(from) {
            if from.raw() < other.raw()
                && !treatments.iter().any(|t| dense(*t) == from || dense(*t) == other)
            {
                out.insert_bidirected(from, other).map_err(IdentificationError::from)?;
            }
        }
    }
    for (index, target) in diagram.selection_targets().iter().copied().enumerate() {
        if treatments.contains(&target) {
            continue;
        }
        let raw = u32::try_from(graph.node_count() + index)
            .map_err(|_| IdentificationError::msg("selection diagram exceeds u32 node capacity"))?;
        out.insert_directed(DenseNodeId::from_raw(raw), dense(target))
            .map_err(IdentificationError::from)?;
    }
    Ok(out)
}

fn source_experiments_cover(query: &TransportQuery, treatments: &[VariableId]) -> bool {
    if let Some(catalog) = &query.catalog {
        catalog.has_available_experiment(&query.source_population, treatments)
    } else {
        treatments.len() == 1 && query.source_experiments.contains(&treatments[0])
    }
}

fn response_factor(
    population: Arc<str>,
    variables: &[VariableId],
    conditioned_on: &[VariableId],
    interventions: &[VariableId],
) -> PopulationFactor {
    PopulationFactor {
        population,
        regime: None,
        variables: variables.to_vec().into(),
        conditioned_on: conditioned_on.to_vec().into(),
        interventions: interventions.to_vec().into(),
    }
}

type ResponseVariables = (Arc<[VariableId]>, Arc<[VariableId]>);

fn response_variables(query: &TransportQuery) -> Result<ResponseVariables, IdentificationError> {
    let (outcomes, treatments): (Vec<_>, Vec<_>) = match &query.response.functional {
        ResponseFunctional::MeanCurve { outcome, treatment } => {
            (vec![*outcome], vec![treatment.variable])
        }
        ResponseFunctional::AverageDerivative { outcome, treatment, .. }
        | ResponseFunctional::PointDerivative { outcome, treatment, .. } => {
            (vec![*outcome], vec![*treatment])
        }
        ResponseFunctional::DirectionalDerivative { outcomes, treatments, .. }
        | ResponseFunctional::Jacobian { outcomes, treatments, .. } => {
            (outcomes.to_vec(), treatments.to_vec())
        }
        ResponseFunctional::InterventionResponse { outcome, interventions } => {
            let treatments = interventions
                .iter()
                .filter_map(antecedent_core::Intervention::primary_variable)
                .collect::<Vec<_>>();
            (vec![*outcome], treatments)
        }
    };
    if treatments.is_empty() {
        return Err(IdentificationError::unsupported(
            "transport response has no statically resolvable intervention variables",
        ));
    }
    Ok((outcomes.into(), treatments.into()))
}

const fn dense(variable: VariableId) -> DenseNodeId {
    DenseNodeId::from_raw(variable.raw())
}

fn topological_order(diagram: &SelectionDiagram) -> Result<Vec<VariableId>, IdentificationError> {
    let graph = diagram.causal_graph();
    let mut indegree = Vec::with_capacity(graph.node_count());
    for i in 0..graph.node_count() {
        let raw = u32::try_from(i)
            .map_err(|_| IdentificationError::msg("selection diagram exceeds u32 node capacity"))?;
        indegree.push(graph.parents(DenseNodeId::from_raw(raw)).len());
    }
    let mut ready = indegree
        .iter()
        .enumerate()
        .filter_map(|(i, &degree)| (degree == 0).then_some(i))
        .collect::<Vec<_>>();
    let mut out = Vec::with_capacity(graph.node_count());
    while let Some(i) = ready.pop() {
        let raw = u32::try_from(i)
            .map_err(|_| IdentificationError::msg("selection diagram exceeds u32 node capacity"))?;
        let node = DenseNodeId::from_raw(raw);
        out.push(VariableId::from_raw(raw));
        for child in graph.children(node) {
            indegree[child.as_usize()] -= 1;
            if indegree[child.as_usize()] == 0 {
                ready.push(child.as_usize());
            }
        }
    }
    if out.len() == graph.node_count() {
        Ok(out)
    } else {
        Err(IdentificationError::msg("selection diagram contains a directed cycle"))
    }
}

#[cfg(test)]
mod tests {
    use antecedent_core::{ContinuousDomain, GridSpec, ResponseQuery, TransportOutcomeKind};
    use antecedent_graph::Admg;

    use super::*;

    fn query() -> TransportQuery {
        TransportQuery::new(
            ResponseQuery::new(ResponseFunctional::MeanCurve {
                outcome: VariableId::from_raw(2),
                treatment: ContinuousDomain::new(
                    VariableId::from_raw(0),
                    GridSpec::Values(Arc::from([0.0, 1.0])),
                ),
            }),
            "trial",
            "target",
            [VariableId::from_raw(0)],
        )
    }

    #[test]
    fn direct_transport_when_selection_is_irrelevant() {
        let mut graph = Admg::with_variables(3);
        graph
            .insert_directed(dense(VariableId::from_raw(0)), dense(VariableId::from_raw(2)))
            .unwrap();
        let diagram = SelectionDiagram::try_new(graph, [VariableId::from_raw(1)]).unwrap();
        let result = TransportIdentifier::new().identify(&diagram, &query()).unwrap();
        assert!(matches!(
            result,
            TransportIdentification::Transportable { formula: TransportFormula::Direct(_), .. }
        ));
    }

    #[test]
    fn intervening_on_selected_treatment_removes_population_difference() {
        let mut graph = Admg::with_variables(3);
        graph
            .insert_directed(dense(VariableId::from_raw(0)), dense(VariableId::from_raw(2)))
            .unwrap();
        graph
            .insert_bidirected(dense(VariableId::from_raw(0)), dense(VariableId::from_raw(2)))
            .unwrap();
        let diagram = SelectionDiagram::try_new(graph, [VariableId::from_raw(0)]).unwrap();
        let result = TransportIdentifier::new().identify(&diagram, &query()).unwrap();
        assert!(matches!(
            result,
            TransportIdentification::Transportable { formula: TransportFormula::Direct(_), .. }
        ));
    }

    #[test]
    fn exogenous_pre_treatment_selection_standardizes() {
        let mut graph = Admg::with_variables(3);
        graph
            .insert_directed(dense(VariableId::from_raw(1)), dense(VariableId::from_raw(2)))
            .unwrap();
        graph
            .insert_directed(dense(VariableId::from_raw(0)), dense(VariableId::from_raw(2)))
            .unwrap();
        let diagram = SelectionDiagram::try_new(graph, [VariableId::from_raw(1)]).unwrap();
        let result = TransportIdentifier::new().identify(&diagram, &query()).unwrap();
        assert!(matches!(
            result,
            TransportIdentification::Transportable {
                formula: TransportFormula::Standardize { .. },
                ..
            }
        ));
    }

    #[test]
    fn singleton_c_components_use_recursive_population_factors() {
        let mut graph = Admg::with_variables(3);
        graph
            .insert_directed(dense(VariableId::from_raw(0)), dense(VariableId::from_raw(1)))
            .unwrap();
        graph
            .insert_directed(dense(VariableId::from_raw(1)), dense(VariableId::from_raw(2)))
            .unwrap();
        let diagram = SelectionDiagram::try_new(graph, [VariableId::from_raw(1)]).unwrap();
        let result = TransportIdentifier::new().identify(&diagram, &query()).unwrap();
        assert!(matches!(
            result,
            TransportIdentification::Transportable {
                formula: TransportFormula::RecursiveFactorization { .. },
                ..
            }
        ));
    }

    #[test]
    fn general_s_admissibility_accepts_nonexogenous_pre_treatment_standardizer() {
        let mut graph = Admg::with_variables(4);
        graph
            .insert_directed(dense(VariableId::from_raw(3)), dense(VariableId::from_raw(1)))
            .unwrap();
        graph
            .insert_directed(dense(VariableId::from_raw(1)), dense(VariableId::from_raw(2)))
            .unwrap();
        graph
            .insert_directed(dense(VariableId::from_raw(0)), dense(VariableId::from_raw(2)))
            .unwrap();
        let diagram = SelectionDiagram::try_new(graph, [VariableId::from_raw(1)]).unwrap();
        let result = TransportIdentifier::new().identify(&diagram, &query()).unwrap();
        let TransportIdentification::Transportable { formula, certificate } = result else {
            panic!("S-admissible standardization should be certified");
        };
        assert_eq!(&*certificate.rule, "transport.sid.s_admissible");
        assert!(matches!(
            formula,
            TransportFormula::Standardize { over, .. }
                if *over == [VariableId::from_raw(1)]
        ));
    }

    #[test]
    fn outcome_as_selection_target_is_never_standardized_over() {
        // Y itself is the population-varying mechanism. Reachability is reflexive, so Y
        // lands in the relevant-selection set; the standardize tier used to accept it and
        // emit P_trial(y | do(x), y) summed over y — a degenerate factor that marginalizes
        // away the requested outcome. The correct reading of this diagram is the
        // c-component factorization, which takes Y's factor from the target population.
        let outcome = VariableId::from_raw(2);
        let graph = Admg::with_variables(3);
        let diagram = SelectionDiagram::try_new(graph, [outcome]).unwrap();
        let result = TransportIdentifier::new().identify(&diagram, &query()).unwrap();
        let TransportIdentification::Transportable { formula, certificate } = &result else {
            return; // Refusal is also an acceptable answer; the degenerate formula is not.
        };
        assert_eq!(&*certificate.rule, "transport.sid.singleton_c_components");
        match formula {
            TransportFormula::Standardize { .. } => {
                panic!("a varying outcome mechanism must not use the standardize formula")
            }
            TransportFormula::RecursiveFactorization { sum_out, .. } => {
                assert!(!sum_out.contains(&outcome), "the requested outcome was marginalized away");
            }
            TransportFormula::Direct(_) => panic!("a varying outcome mechanism is not direct"),
        }
    }

    #[test]
    fn post_treatment_selected_mediator_is_never_a_standardizer() {
        // X -> M -> Y with M's mechanism varying. Σ_m P_trial(y | do(x), m) P_target(m) is
        // wrong: the target's observational law of M is not its law under do(x).
        let (x, m, y) = (VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2));
        let mut graph = Admg::with_variables(3);
        graph.insert_directed(dense(x), dense(m)).unwrap();
        graph.insert_directed(dense(m), dense(y)).unwrap();
        let diagram = SelectionDiagram::try_new(graph, [m]).unwrap();
        let result = TransportIdentifier::new().identify(&diagram, &query()).unwrap();
        let TransportIdentification::Transportable { formula, certificate } = &result else {
            panic!("singleton districts identify this diagram: {result:?}");
        };
        assert_eq!(&*certificate.rule, "transport.sid.singleton_c_components");
        let TransportFormula::RecursiveFactorization { factors, .. } = formula else {
            panic!("a post-treatment selection target must not be standardized over: {formula:?}");
        };
        let mediator = factors.iter().find(|f| f.variables.contains(&m)).unwrap();
        assert_eq!(&*mediator.population, "target");
    }

    #[test]
    fn multi_node_district_returns_scoped_negative_certificate() {
        let mut graph = Admg::with_variables(3);
        graph
            .insert_directed(dense(VariableId::from_raw(1)), dense(VariableId::from_raw(2)))
            .unwrap();
        graph
            .insert_bidirected(dense(VariableId::from_raw(1)), dense(VariableId::from_raw(2)))
            .unwrap();
        let diagram = SelectionDiagram::try_new(graph, [VariableId::from_raw(1)]).unwrap();
        let result = TransportIdentifier::new().identify(&diagram, &query()).unwrap();
        assert!(matches!(result, TransportIdentification::NotCertified(_)));
        assert_eq!(result.outcome().kind, TransportOutcomeKind::NotCertified);
    }

    #[test]
    fn selection_upstream_of_intervened_treatment_needs_no_target_covariate_law() {
        let mut graph = Admg::with_variables(3);
        graph
            .insert_directed(dense(VariableId::from_raw(1)), dense(VariableId::from_raw(0)))
            .unwrap();
        graph
            .insert_directed(dense(VariableId::from_raw(0)), dense(VariableId::from_raw(2)))
            .unwrap();
        graph
            .insert_bidirected(dense(VariableId::from_raw(0)), dense(VariableId::from_raw(2)))
            .unwrap();
        let diagram = SelectionDiagram::try_new(graph, [VariableId::from_raw(1)]).unwrap();
        let result = TransportIdentifier::new().identify(&diagram, &query()).unwrap();
        assert!(matches!(
            result,
            TransportIdentification::Transportable { formula: TransportFormula::Direct(_), .. }
        ));
        let mutilated =
            treatment_mutilated_selection_graph(&diagram, &[VariableId::from_raw(0)]).unwrap();
        assert!(mutilated.bidirected_neighbors(dense(VariableId::from_raw(0))).is_empty());
    }

    fn empty_catalog_query() -> TransportQuery {
        TransportQuery::new(query().response, "trial", "target", Vec::<VariableId>::new())
            .with_catalog(antecedent_core::EvidenceCatalog::empty())
            .unwrap()
    }

    #[test]
    fn empty_catalog_and_invariant_mechanisms_identify_on_the_target() {
        let graph = Admg::with_variables(3);
        let diagram = SelectionDiagram::try_new(graph, []).unwrap();
        let result = TransportIdentifier::new().identify(&diagram, &empty_catalog_query()).unwrap();
        let TransportIdentification::Transportable { formula, .. } = &result else {
            panic!("target-only identify must succeed with an empty source catalog");
        };
        match formula {
            TransportFormula::RecursiveFactorization { factors, .. } => {
                assert!(
                    factors
                        .iter()
                        .all(|f| f.population.as_ref() == "target" && f.interventions.is_empty())
                );
            }
            other => panic!("expected observational target factorization, got {other:?}"),
        }
        assert_eq!(result.outcome().kind, TransportOutcomeKind::Identified);
    }

    #[test]
    fn missing_source_experiment_is_missing_evidence_not_not_certified() {
        let mut graph = Admg::with_variables(3);
        graph
            .insert_directed(dense(VariableId::from_raw(1)), dense(VariableId::from_raw(2)))
            .unwrap();
        graph
            .insert_directed(dense(VariableId::from_raw(0)), dense(VariableId::from_raw(2)))
            .unwrap();
        graph
            .insert_bidirected(dense(VariableId::from_raw(0)), dense(VariableId::from_raw(2)))
            .unwrap();
        let diagram = SelectionDiagram::try_new(graph, [VariableId::from_raw(1)]).unwrap();
        let result = TransportIdentifier::new().identify(&diagram, &empty_catalog_query()).unwrap();
        assert!(matches!(result, TransportIdentification::MissingEvidence(_)));
        assert_eq!(result.outcome().kind, TransportOutcomeKind::MissingEvidence);
        assert_ne!(result.outcome().kind, TransportOutcomeKind::NotCertified);
        assert_ne!(result.outcome().kind, TransportOutcomeKind::ProvenNonTransportable);
    }
}
