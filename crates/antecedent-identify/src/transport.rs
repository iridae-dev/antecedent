//! Conservative structural transportability identification.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{ResponseFunctional, TransportOutcome, TransportQuery, VariableId};
use antecedent_graph::{BitSet, DenseNodeId, SelectionDiagram};

use crate::selection_separation::{
    MutilatedSelection, SubsetSearchEnd, for_each_admissible_subset,
};
use crate::{IdentificationError, PreparedAdmg};

/// Most treatment-mutilated subsets the structural standardizer search will
/// separation-test. Exhausting it is a typed inconclusive outcome, not a
/// non-existence claim.
const STANDARDIZER_SUBSET_BUDGET: usize = 1 << 20;

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
    /// A catalog that declares no regime at all for the target does not withdraw the target
    /// observational law: the formula is certified with an explicit "target observational law
    /// assumed available" premise and its target factors stay unbound. The bound-evaluation
    /// path (`identify_catalog_transport`) requires a declared target regime instead.
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

    fn identify_structural(
        diagram: &SelectionDiagram,
        query: &TransportQuery,
    ) -> Result<TransportIdentification, IdentificationError> {
        Self::identify_structural_within(diagram, query, STANDARDIZER_SUBSET_BUDGET)
    }

    #[allow(clippy::too_many_lines)]
    fn identify_structural_within(
        diagram: &SelectionDiagram,
        query: &TransportQuery,
        subset_budget: usize,
    ) -> Result<TransportIdentification, IdentificationError> {
        query.validate().map_err(|error| IdentificationError::msg(error.to_string()))?;
        let graph = diagram.causal_graph();
        // Every coordinate goes through the graph's own node table, so a graph
        // whose nodes are not numbered like their variable ids is read correctly.
        let prepared = PreparedAdmg::new(graph.clone())?;
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
            prepared.var_to_dense(*variable)?;
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
            for parent in graph.parents(prepared.var_to_dense(variable)?) {
                pending.push(prepared.dense_to_var(*parent)?);
            }
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
            let order = topological_variables(&prepared)?
                .into_iter()
                .filter(|v| outcome_ancestors.contains(v))
                .collect::<Vec<_>>();
            let mut factors = Vec::new();
            for v in order.iter().copied().filter(|v| !treatments.contains(v)) {
                let parents = graph
                    .parents(prepared.var_to_dense(v)?)
                    .iter()
                    .map(|p| prepared.dense_to_var(*p))
                    .collect::<Result<Vec<_>, _>>()?;
                factors.push(response_factor(
                    Arc::clone(&query.target_population),
                    &[v],
                    &parents,
                    &[],
                ));
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
        let mut safe_standardizers = !relevant_selection.iter().any(|z| outcomes.contains(z));
        for z in &relevant_selection {
            let dz = prepared.var_to_dense(*z)?;
            let mut downstream = false;
            for x in treatments.iter() {
                downstream |= graph.reaches_with(prepared.var_to_dense(*x)?, dz, &mut reach_ws);
            }
            safe_standardizers &= !downstream
                && graph.parents(dz).is_empty()
                && graph.bidirected_neighbors(dz).is_empty();
        }
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
        let search =
            s_admissible_standardizers(diagram, &prepared, &outcomes, &treatments, subset_budget)?;
        if let StandardizerSearch::Found(over) = &search {
            let over = Arc::clone(over);
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
            let order = topological_variables(&prepared)?;
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

        if matches!(search, StandardizerSearch::Capped) {
            return Ok(TransportIdentification::NotCertified(NotCertifiedCertificate {
                reason: Arc::from("transport.sid.standardizer_search_capped"),
                witness: relevant_selection.into(),
                message: Arc::from(
                    "the S-admissible standardizer search reached its subset budget before examining every pretreatment subset; no non-transportability claim is made",
                ),
            }));
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
    let TransportIdentification::Transportable { formula, certificate } = &mut result else {
        return result;
    };
    let mut assumed_target_law = false;
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
        let need = antecedent_core::FactorNeed {
            population: &factor.population,
            variables: &factor.variables,
            conditioned_on: &factor.conditioned_on,
            interventions: &factor.interventions,
        };
        let needed = need.needed_variables();
        let regime = catalog.satisfying_regime(&need).map(|regime| regime.id);
        if sampling_valid {
            if let Some(regime) = regime {
                factor.regime = Some(regime);
                continue;
            }
            // A catalog that says nothing about the target does not withdraw its
            // observational law, but the formula is then unbound: it is certified
            // on that assumption, recorded below, and an evaluator still needs a
            // target regime before it can consume the factor.
            if target
                && catalog.target_sampling
                    != Some(antecedent_core::TargetSampling::LicensedWeightedDesign)
                && factor.interventions.is_empty()
                && !catalog.regimes.iter().any(|r| r.population == factor.population)
            {
                assumed_target_law = true;
                continue;
            }
        }
        missing.extend(needed);
    }
    if missing.is_empty() {
        if assumed_target_law {
            let mut premises = certificate.premises.to_vec();
            premises.push(Arc::from(
                "target observational law assumed available: the catalog declares no target regime",
            ));
            certificate.premises = premises.into();
        }
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

/// Outcome of the pretreatment standardizer search.
enum StandardizerSearch {
    Found(Arc<[VariableId]>),
    /// Every pretreatment subset was examined; none is S-admissible.
    NotFound,
    /// The subset budget ran out first. Inconclusive.
    Capped,
}

fn s_admissible_standardizers(
    diagram: &SelectionDiagram,
    prepared: &PreparedAdmg,
    outcomes: &[VariableId],
    treatments: &[VariableId],
    subset_budget: usize,
) -> Result<StandardizerSearch, IdentificationError> {
    let graph = diagram.causal_graph();
    let n = graph.node_count();
    let mut reach_ws = antecedent_graph::GraphWorkspace::default();
    let outcome_nodes = dense_all(prepared, outcomes)?;
    let treatment_nodes = dense_all(prepared, treatments)?;
    let mut candidates = Vec::new();
    for i in 0..n {
        let node = DenseNodeId::from_raw(u32::try_from(i).map_err(|_| {
            IdentificationError::msg("selection diagram exceeds u32 node capacity")
        })?);
        if outcome_nodes.contains(&node)
            || treatment_nodes.contains(&node)
            || treatment_nodes.iter().any(|t| graph.reaches_with(*t, node, &mut reach_ws))
        {
            continue;
        }
        candidates.push(node);
    }
    let mut all = BitSet::with_len(n);
    for node in prepared.topo() {
        all.insert(*node);
    }
    let mut treated = BitSet::with_len(n);
    for node in &treatment_nodes {
        treated.insert(*node);
    }
    let targets = dense_all(prepared, diagram.selection_targets())?;
    let selection = MutilatedSelection::build(graph, &all, &treated, &targets)?;
    let mut found = None;
    let end = for_each_admissible_subset(
        &selection,
        &outcome_nodes,
        &treatment_nodes,
        &candidates,
        subset_budget,
        || Ok(()),
        |subset| {
            found = Some(
                subset
                    .iter()
                    .map(|node| prepared.dense_to_var(*node))
                    .collect::<Result<Vec<_>, _>>()?,
            );
            Ok(true)
        },
    )?;
    Ok(match (found, end) {
        (Some(over), _) => StandardizerSearch::Found(over.into()),
        (None, SubsetSearchEnd::Capped) => StandardizerSearch::Capped,
        (None, _) => StandardizerSearch::NotFound,
    })
}

fn dense_all(
    prepared: &PreparedAdmg,
    variables: &[VariableId],
) -> Result<Vec<DenseNodeId>, IdentificationError> {
    variables.iter().map(|v| prepared.var_to_dense(*v)).collect()
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

fn topological_variables(prepared: &PreparedAdmg) -> Result<Vec<VariableId>, IdentificationError> {
    prepared.topo().iter().map(|node| prepared.dense_to_var(*node)).collect()
}

#[cfg(test)]
mod tests {
    use antecedent_core::{ContinuousDomain, GridSpec, ResponseQuery, TransportOutcomeKind};
    use antecedent_graph::Admg;

    use super::*;

    const fn dense(variable: VariableId) -> DenseNodeId {
        DenseNodeId::from_raw(variable.raw())
    }

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
        // The c-component factorization identifies this diagram, so a refusal would be a
        // regression too: a refuse-everything identifier must not pass.
        let TransportIdentification::Transportable { formula, certificate } = &result else {
            panic!("the c-component factorization identifies this diagram: {result:?}");
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
        // The catalog is silent about the target, so its observational law is assumed;
        // the certificate must say so rather than look like a bound regime.
        let TransportIdentification::Transportable { certificate, .. } = &result else {
            unreachable!();
        };
        assert!(
            certificate.premises.iter().any(|p| p.contains("target observational law assumed"))
        );
    }

    fn query_over(outcome: u32, treatment: u32) -> TransportQuery {
        TransportQuery::new(
            ResponseQuery::new(ResponseFunctional::MeanCurve {
                outcome: VariableId::from_raw(outcome),
                treatment: ContinuousDomain::new(
                    VariableId::from_raw(treatment),
                    GridSpec::Values(Arc::from([0.0, 1.0])),
                ),
            }),
            "trial",
            "target",
            [VariableId::from_raw(treatment)],
        )
    }

    fn graph_with_ids(ids: &[u32]) -> Admg {
        let mut graph = Admg::empty();
        for id in ids {
            graph.add_node(antecedent_graph::NodeRef::Static(VariableId::from_raw(*id))).unwrap();
        }
        graph
    }

    #[test]
    fn graph_nodes_numbered_unlike_their_variable_ids_are_read_through_the_node_table() {
        // Variables 17 -> 41 -> 99 occupy dense nodes 0 -> 1 -> 2. Selection on 17 is
        // blocked from 99 by the treatment 41, so the source experiment transports directly.
        let mut graph = graph_with_ids(&[17, 41, 99]);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        graph.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
        let diagram = SelectionDiagram::try_new(graph, [VariableId::from_raw(17)]).unwrap();
        let result = TransportIdentifier::new().identify(&diagram, &query_over(99, 41)).unwrap();
        let TransportIdentification::Transportable { formula, certificate } = result else {
            panic!("direct transport expected: {result:?}");
        };
        assert_eq!(&*certificate.rule, "transport.sid.direct");
        let TransportFormula::Direct(factor) = formula else { panic!("direct formula") };
        assert_eq!(*factor.variables, [VariableId::from_raw(99)]);
        assert_eq!(*factor.interventions, [VariableId::from_raw(41)]);
    }

    #[test]
    fn standardizer_is_reported_in_variable_ids_not_dense_positions() {
        // Z=10 -> Y=30 <- X=20 with Z selected: standardize over Z, whose dense node is 0.
        let mut graph = graph_with_ids(&[10, 20, 30]);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
        graph.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
        let diagram = SelectionDiagram::try_new(graph, [VariableId::from_raw(10)]).unwrap();
        let result = TransportIdentifier::new().identify(&diagram, &query_over(30, 20)).unwrap();
        let TransportIdentification::Transportable {
            formula: TransportFormula::Standardize { over, .. },
            ..
        } = result
        else {
            panic!("standardization expected: {result:?}");
        };
        assert_eq!(*over, [VariableId::from_raw(10)]);
    }

    fn selected_collider_graph(nodes: u32) -> SelectionDiagram {
        // Z <-> Y with Z -> Y and Z selected has no S-admissible standardizer:
        // conditioning on the collider Z opens S -> Z <-> Y. Z, Y and X are nodes 0, 1, 2;
        // any further nodes are isolated pretreatment variables.
        let mut graph = Admg::with_variables(nodes);
        let (z, y) = (DenseNodeId::from_raw(0), DenseNodeId::from_raw(1));
        graph.insert_directed(z, y).unwrap();
        graph.insert_bidirected(z, y).unwrap();
        SelectionDiagram::try_new(graph, [VariableId::from_raw(0)]).unwrap()
    }

    #[test]
    fn exhausted_standardizer_search_is_named_and_not_blamed_on_c_component_recursion() {
        // 22 isolated pretreatment variables make the subset space larger than the budget.
        let query = query_over(1, 2);
        let capped = TransportIdentifier::identify_structural_within(
            &selected_collider_graph(25),
            &query,
            64,
        )
        .unwrap();
        let TransportIdentification::NotCertified(refusal) = capped else {
            panic!("expected an inconclusive refusal: {capped:?}");
        };
        assert_eq!(&*refusal.reason, "transport.sid.standardizer_search_capped");
        // A search that finishes without a standardizer is refused for the genuine reason.
        let exhaustive = TransportIdentifier::identify_structural_within(
            &selected_collider_graph(3),
            &query,
            usize::MAX,
        )
        .unwrap();
        let TransportIdentification::NotCertified(refusal) = exhaustive else {
            panic!("expected a refusal: {exhaustive:?}");
        };
        assert_eq!(&*refusal.reason, "transport.sid.multinode_c_component_not_implemented");
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
