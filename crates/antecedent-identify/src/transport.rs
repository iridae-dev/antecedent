//! Conservative structural transportability identification.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{ResponseFunctional, TransportQuery, VariableId};
use antecedent_graph::{Admg, DSeparationWorkspace, DenseNodeId, SelectionDiagram};

use crate::IdentificationError;

/// Population-labelled symbolic distribution required by a transport formula.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PopulationFactor {
    /// Population key.
    pub population: Arc<str>,
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

/// Explicit negative certificate for the conservative identifier.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NonTransportableCertificate {
    /// Stable failure id.
    pub reason: Arc<str>,
    /// Variables witnessing the failed criterion.
    pub witness: Arc<[VariableId]>,
    /// Scope note: refusal is not a completeness claim outside implemented rules.
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
    /// No implemented sound rule applies.
    NotCertified(NonTransportableCertificate),
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
    /// The implementation deliberately returns `NotCertified` rather than claiming
    /// non-transportability when general multi-node c-component recursion is required.
    #[allow(clippy::too_many_lines)]
    pub fn identify(
        &self,
        diagram: &SelectionDiagram,
        query: &TransportQuery,
    ) -> Result<TransportIdentification, IdentificationError> {
        query.validate().map_err(|error| IdentificationError::msg(error.to_string()))?;
        let graph = diagram.causal_graph();
        let (outcomes, treatments) = response_variables(query)?;
        for variable in outcomes.iter().chain(treatments.iter()) {
            if variable.as_usize() >= graph.node_count() {
                return Err(IdentificationError::UnknownVariable { id: *variable });
            }
        }
        if treatments.iter().any(|x| !query.source_experiments.contains(x)) {
            return Ok(TransportIdentification::NotCertified(NonTransportableCertificate {
                reason: Arc::from("transport.source_experiment_missing"),
                witness: treatments,
                message: Arc::from(
                    "the source experiment required by this transport query is unavailable",
                ),
            }));
        }

        let relevant_selection = diagram
            .selection_targets()
            .iter()
            .copied()
            .filter(|selection| {
                !treatments.contains(selection)
                    && outcomes
                        .iter()
                        .any(|outcome| graph.reaches(dense(*selection), dense(*outcome)))
            })
            .collect::<Vec<_>>();
        if relevant_selection.is_empty() {
            let factor =
                response_factor(Arc::clone(&query.source_population), &outcomes, &[], &treatments);
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
                !treatments.iter().any(|x| graph.reaches(dense(*x), dz))
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

        Ok(TransportIdentification::NotCertified(NonTransportableCertificate {
            reason: Arc::from("transport.sid.multinode_c_component_not_implemented"),
            witness: relevant_selection.into(),
            message: Arc::from(
                "a population-varying outcome ancestor lies in a graph requiring general c-component recursion; no non-transportability claim is made",
            ),
        }))
    }
}

fn s_admissible_standardizers(
    diagram: &SelectionDiagram,
    outcomes: &[VariableId],
    treatments: &[VariableId],
) -> Result<Option<Arc<[VariableId]>>, IdentificationError> {
    let graph = diagram.causal_graph();
    let candidates = (0..graph.node_count())
        .filter_map(|i| u32::try_from(i).ok().map(VariableId::from_raw))
        .filter(|candidate| !outcomes.contains(candidate) && !treatments.contains(candidate))
        .filter(|candidate| {
            !treatments.iter().any(|treatment| graph.reaches(dense(*treatment), dense(*candidate)))
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
    for size in 0..=candidates.len() {
        for mask in 0_u64..(1_u64 << candidates.len()) {
            if usize::try_from(mask.count_ones()).expect("u32 fits usize") != size {
                continue;
            }
            let standardizers = candidates
                .iter()
                .enumerate()
                .filter_map(|(i, variable)| ((mask >> i) & 1 == 1).then_some(*variable))
                .collect::<Vec<_>>();
            let conditioned = standardizers.iter().copied().map(dense).collect::<Vec<_>>();
            let mut separated = true;
            for outcome in outcomes {
                for selection_index in 0..diagram.selection_targets().len() {
                    let raw = u32::try_from(selection_offset + selection_index).map_err(|_| {
                        IdentificationError::msg("selection diagram exceeds u32 node capacity")
                    })?;
                    if !augmented
                        .is_m_separated(
                            dense(*outcome),
                            DenseNodeId::from_raw(raw),
                            &conditioned,
                            &mut workspace,
                        )
                        .map_err(IdentificationError::from)?
                    {
                        separated = false;
                        break;
                    }
                }
                if !separated {
                    break;
                }
            }
            if separated {
                return Ok(Some(standardizers.into()));
            }
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
            if from.raw() < other.raw() {
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

fn response_factor(
    population: Arc<str>,
    variables: &[VariableId],
    conditioned_on: &[VariableId],
    interventions: &[VariableId],
) -> PopulationFactor {
    PopulationFactor {
        population,
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
    use antecedent_core::{ContinuousDomain, GridSpec, ResponseQuery};
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
    }
}
