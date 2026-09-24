//! Contract validation for the bounded single-source z-transport setting.
//!
//! The contract is kept distinct from classical sID: a source that can intervene
//! only on `controllable` does not satisfy the unrestricted source-experiment
//! premise of the classical theorem.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::IdentificationError;
use antecedent_core::{
    DistributionAvailability, EvidenceCatalog, EvidenceKind,
    InterventionAssignment as CatalogInterventionAssignment, RegimeKind, VariableDomain,
    VariableId,
};
use antecedent_expr::{CausalExprArena, DomainRef, ExprId, ExprNode};
use antecedent_graph::{NodeRef, SelectionDiagram};
use std::sync::Arc;

/// Fixed graph and population contract for a single-source z-transport query.
#[derive(Clone, Debug, PartialEq)]
pub struct ZTransportQuery {
    /// Joint outcomes, in the target population.
    pub outcomes: Arc<[VariableId]>,
    /// Treatment coordinates whose target interventional response is queried.
    pub treatments: Arc<[VariableId]>,
    /// Variables the source can manipulate. The experimental family consists of
    /// joint regimes for subsets of this set; actual available regimes belong in
    /// the evidence catalog and are not implied by this declaration.
    pub controllable: Arc<[VariableId]>,
    /// Concrete source experiment assignment used by an identified formula.
    /// This is distinct from controllability: its presence does not claim results
    /// exist; those must be found in the evidence catalog.
    pub experiment_assignment: Arc<[CatalogInterventionAssignment]>,
    /// One source population supplying the declared experimental family.
    pub source: Arc<str>,
    /// Target population supplying its observational law.
    pub target: Arc<str>,
}

/// Current explicit bound for the 2.1 single-source z-transport contract.
pub const Z_TRANSPORT_MAX_OBSERVED: usize = 6;
/// Current explicit bound for controllable variables.
pub const Z_TRANSPORT_MAX_CONTROLLABLE: usize = 2;

/// A missing or unsupported part of the declared full-law experiment family.
#[derive(Clone, Debug, PartialEq)]
pub enum ZExperimentFamilyError {
    /// A finite discrete family cannot be enumerated from this variable domain.
    UnsupportedDomain {
        /// Controllable variable with an unsupported or undeclared finite domain.
        variable: VariableId,
    },
    /// One required intervention joint law is absent or has only marginal data.
    MissingJointLaw {
        /// Variables intervened on.
        interventions: Arc<[VariableId]>,
        /// Concrete assignment required for this regime (empty for observational).
        values: Arc<[(VariableId, f64)]>,
    },
}

/// A checked derivation for the registered four-variable surrogate experiment.
#[derive(Clone, Debug)]
pub struct ZTransportDerivation {
    query: ZTransportQuery,
    graph_signature: String,
    surrogate: VariableId,
    confounder: VariableId,
    arena: CausalExprArena,
    root: ExprId,
}

/// Catalog-bound form of the registered zTR formula.
#[derive(Clone, Debug)]
pub struct BoundZTransportFunctional {
    derivation: ZTransportDerivation,
    arena: CausalExprArena,
    root: ExprId,
    catalog: EvidenceCatalog,
    regime: antecedent_core::RegimeId,
}

impl BoundZTransportFunctional {
    /// Checked symbolic derivation.
    #[must_use]
    pub const fn derivation(&self) -> &ZTransportDerivation {
        &self.derivation
    }

    /// Provider-bound formula arena.
    #[must_use]
    pub const fn arena(&self) -> &CausalExprArena {
        &self.arena
    }

    /// Provider-bound formula root.
    #[must_use]
    pub const fn root(&self) -> ExprId {
        self.root
    }

    /// Regime supplying the formula's two source-law factors.
    #[must_use]
    pub const fn regime(&self) -> antecedent_core::RegimeId {
        self.regime
    }

    /// Frozen catalog to which factor leaves are bound.
    #[must_use]
    pub const fn catalog(&self) -> &EvidenceCatalog {
        &self.catalog
    }
}

impl ZTransportDerivation {
    /// Original typed z-transport query.
    #[must_use]
    pub const fn query(&self) -> &ZTransportQuery {
        &self.query
    }

    /// Surrogate variable manipulated in the source experiment.
    #[must_use]
    pub const fn surrogate(&self) -> VariableId {
        self.surrogate
    }

    /// Exact source intervention assignment cited by the checked formula.
    #[must_use]
    pub fn experiment_assignment(&self) -> &[CatalogInterventionAssignment] {
        &self.query.experiment_assignment
    }

    /// Shared confounder marginalized by the formula.
    #[must_use]
    pub const fn confounder(&self) -> VariableId {
        self.confounder
    }

    /// Formula arena for `Σ_w P_s(y | w,x,do(z)) P_s(w | do(z))`.
    #[must_use]
    pub const fn arena(&self) -> &CausalExprArena {
        &self.arena
    }

    /// Root of the checked surrogate formula.
    #[must_use]
    pub const fn root(&self) -> ExprId {
        self.root
    }
}

/// Bounded outcome of the currently licensed sIDz special case.
#[derive(Clone, Debug)]
pub enum ZTransportResult {
    /// Formula and theorem premises were checked for the registered graph family.
    Identified(Box<ZTransportDerivation>),
    /// This bounded implementation did not find a licensed case. This is not
    /// a theorem-specific impossibility result.
    NotCertified {
        /// Stable scope note.
        reason: &'static str,
    },
}

/// Validate the graph/query portion of the bounded z-transport contract.
///
/// This check does not imply that experiments exist. Callers must bind every
/// formula factor to an available exact joint regime with matching population,
/// measurement scope, intervention set, and intervention values.
///
/// # Errors
/// Returns a stable validation error when the graph, query, population identities,
/// or declared computational bound is invalid.
pub fn validate_z_transport_query(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
) -> Result<(), IdentificationError> {
    if diagram.causal_graph().node_count() > Z_TRANSPORT_MAX_OBSERVED {
        return Err(IdentificationError::msg("z_transport.unsupported_observed_count"));
    }
    if query.controllable.is_empty() || query.controllable.len() > Z_TRANSPORT_MAX_CONTROLLABLE {
        return Err(IdentificationError::msg("z_transport.unsupported_controllable_count"));
    }
    if query.outcomes.is_empty()
        || query.treatments.is_empty()
        || query.source.trim().is_empty()
        || query.target.trim().is_empty()
        || query.source == query.target
    {
        return Err(IdentificationError::msg("z_transport.invalid_query"));
    }

    for group in [&query.outcomes, &query.treatments, &query.controllable] {
        let mut seen = std::collections::BTreeSet::new();
        for variable in group.iter().copied() {
            if !diagram.causal_graph().nodes().contains(&NodeRef::Static(variable)) {
                return Err(IdentificationError::msg("z_transport.unknown_variable"));
            }
            if !seen.insert(variable.raw()) {
                return Err(IdentificationError::msg("z_transport.duplicate_variable"));
            }
        }
    }
    if query.outcomes.iter().any(|v| query.treatments.contains(v)) {
        return Err(IdentificationError::msg("z_transport.outcomes_overlap_treatments"));
    }
    let mut assigned = std::collections::BTreeSet::new();
    for assignment in query.experiment_assignment.iter() {
        if !query.controllable.contains(&assignment.variable)
            || !assigned.insert(assignment.variable.raw())
            || assignment.value.validate_concrete_intervention_level().is_err()
        {
            return Err(IdentificationError::msg("z_transport.invalid_experiment_assignment"));
        }
    }
    Ok(())
}

/// Check for the finite full-law experiment family declared by the zTR contract.
///
/// The theorem input family includes a joint source law for every joint assignment
/// to each non-empty subset of controllable variables. An observational law is not
/// part of this family unless a separate theorem premise requires it.
/// Results here are catalog facts only; a successful check does not identify a
/// query or establish that one supplied regime can substitute for another.
pub fn validate_z_experiment_family(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    catalog: &EvidenceCatalog,
) -> Result<Vec<antecedent_core::RegimeId>, ZExperimentFamilyError> {
    let source =
        catalog.environments.iter().find(|environment| environment.identity == query.source);
    let domains = query
        .controllable
        .iter()
        .map(|variable| {
            let coordinate = source.and_then(|environment| {
                environment.variables.iter().find(|c| c.variable == *variable)
            });
            let domain = coordinate.map(|coordinate| &coordinate.domain);
            let values = match domain {
                Some(VariableDomain::Binary) => vec![0.0, 1.0],
                Some(VariableDomain::Categorical { cardinality }) => {
                    (0..*cardinality).map(f64::from).collect()
                }
                _ => return Err(ZExperimentFamilyError::UnsupportedDomain { variable: *variable }),
            };
            Ok((*variable, values))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let observed = diagram
        .causal_graph()
        .nodes()
        .iter()
        .filter_map(|node| match node {
            NodeRef::Static(variable) => Some(*variable),
            _ => None,
        })
        .collect::<Vec<_>>();
    let mut required = Vec::new();
    let k = query.controllable.len();
    for mask in 1..(1usize << k) {
        let intervened = (0..k)
            .filter(|bit| mask & (1 << bit) != 0)
            .map(|bit| query.controllable[bit])
            .collect::<Vec<_>>();
        let mut assignments = vec![Vec::<(VariableId, f64)>::new()];
        for variable in &intervened {
            let levels = &domains
                .iter()
                .find(|(candidate, _)| candidate == variable)
                .expect("controllable coordinate was enumerated")
                .1;
            assignments = assignments
                .into_iter()
                .flat_map(|prefix| {
                    levels.iter().map(move |value| {
                        let mut assignment = prefix.clone();
                        assignment.push((*variable, *value));
                        assignment
                    })
                })
                .collect();
        }
        for assignment in assignments {
            let supplied = catalog.regimes.iter().find(|regime| {
                let kind_matches = if intervened.is_empty() {
                    regime.kind == RegimeKind::Observational
                } else {
                    regime.kind == RegimeKind::Experimental
                };
                let regime_values = regime
                    .intervention_values
                    .iter()
                    .filter_map(|value| value.value.as_f64().map(|v| (value.variable, v)))
                    .collect::<Vec<_>>();
                kind_matches
                    && regime.evidence_kind == EvidenceKind::Available
                    && regime.population.as_ref() == query.source.as_ref()
                    && same_variable_set(&regime.interventions, &intervened)
                    && (regime_values.len() == assignment.len()
                        && assignment.iter().all(|(variable, value)| {
                            regime_values.iter().any(|(v, actual)| v == variable && actual == value)
                        }))
                    && same_variable_set(&regime.measured, &observed)
                    && regime.conditioned_on.is_empty()
                    && regime.distribution == DistributionAvailability::Joint
                    && catalog.bindings.iter().any(|binding| binding.regime == regime.id)
            });
            if let Some(regime) = supplied {
                required.push(regime.id);
            } else {
                return Err(ZExperimentFamilyError::MissingJointLaw {
                    interventions: intervened.into(),
                    values: assignment.into(),
                });
            }
        }
    }
    Ok(required)
}

/// Identify the registered surrogate zTR case from Lee & Honavar's distinct
/// z-transport setting, Figure 5-style support contract. This is a sound,
/// deliberately incomplete route: other diagrams return `NotCertified`.
///
/// The licensed graph has four observed variables with edges `W→Z→X→Y`,
/// `W→Y`, and bidirected `W↔Y`, `Z↔Y`, `Z↔X`. The only controllable variable is
/// `Z`; the query is `P(Y | do(X))`; source and target share the fixed graph
/// with no selection targets. The formula is checked by rebuilding its exact
/// expression and graph premises before publication.
///
/// # Errors
/// Invalid query coordinates or declared bounds.
pub fn identify_z_transport_surrogate(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
) -> Result<ZTransportResult, IdentificationError> {
    validate_z_transport_query(diagram, query)?;
    if diagram.causal_graph().node_count() != 4 || !diagram.selection_targets().is_empty() {
        return Ok(ZTransportResult::NotCertified {
            reason: "z_transport.outside_registered_surrogate_graph",
        });
    }
    if query.outcomes.len() != 1 || query.treatments.len() != 1 || query.controllable.len() != 1 {
        return Ok(ZTransportResult::NotCertified {
            reason: "z_transport.outside_registered_surrogate_query",
        });
    }
    if query.experiment_assignment.len() != 1
        || query.experiment_assignment[0].variable != query.controllable[0]
    {
        return Ok(ZTransportResult::NotCertified {
            reason: "z_transport.experiment_assignment_required",
        });
    }
    let outcome = query.outcomes[0];
    let treatment = query.treatments[0];
    let surrogate = query.controllable[0];
    let Some(confounder) = diagram
        .causal_graph()
        .nodes()
        .iter()
        .filter_map(|node| match node {
            NodeRef::Static(v) => Some(*v),
            _ => None,
        })
        .find(|v| *v != outcome && *v != treatment && *v != surrogate)
    else {
        return Ok(ZTransportResult::NotCertified {
            reason: "z_transport.outside_registered_surrogate_graph",
        });
    };
    if !matches_registered_surrogate_graph(diagram, confounder, surrogate, treatment, outcome) {
        return Ok(ZTransportResult::NotCertified {
            reason: "z_transport.outside_registered_surrogate_graph",
        });
    }

    let mut arena = CausalExprArena::new();
    let y_set = arena.intern_var_set([outcome]);
    let w_set = arena.intern_var_set([confounder]);
    let wx_set = arena.intern_var_set([confounder, treatment]);
    let z_do =
        arena.intern_intervention_assignments(query.experiment_assignment.iter().map(|a| {
            antecedent_expr::InterventionAssignment::concrete(a.variable, a.value.clone())
        }));
    let source = arena.intern_population(Arc::clone(&query.source));
    let conditional = arena.intern(ExprNode::Distribution {
        variables: y_set,
        conditioned_on: wx_set,
        intervention: z_do,
        domain: DomainRef::Interventional,
        population: source,
        regime: None,
    });
    let empty = arena.empty_var_set();
    let marginal = arena.intern(ExprNode::Distribution {
        variables: w_set,
        conditioned_on: empty,
        intervention: z_do,
        domain: DomainRef::Interventional,
        population: source,
        regime: None,
    });
    let factors = arena.intern_list([conditional, marginal]);
    let product = arena.intern(ExprNode::Product(factors));
    let root = arena.intern(ExprNode::SumOut { variables: w_set, expr: product });
    let derivation = ZTransportDerivation {
        query: query.clone(),
        graph_signature: super::graph_signature(diagram),
        surrogate,
        confounder,
        arena,
        root,
    };
    verify_z_transport_derivation(diagram, query, &derivation)?;
    Ok(ZTransportResult::Identified(Box::new(derivation)))
}

/// Independently recheck the graph and formula premises of a zTR derivation.
///
/// # Errors
/// A changed graph, query, or formula.
pub fn verify_z_transport_derivation(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    derivation: &ZTransportDerivation,
) -> Result<(), IdentificationError> {
    validate_z_transport_query(diagram, query)?;
    if derivation.query != *query
        || derivation.graph_signature != super::graph_signature(diagram)
        || diagram.causal_graph().node_count() != 4
        || !diagram.selection_targets().is_empty()
        || query.outcomes.len() != 1
        || query.treatments.len() != 1
        || query.controllable.as_ref() != [derivation.surrogate]
        || query.experiment_assignment.len() != 1
        || query.experiment_assignment[0].variable != derivation.surrogate
    {
        return Err(IdentificationError::msg("z_transport.proof_input_mismatch"));
    }
    if !matches_registered_surrogate_graph(
        diagram,
        derivation.confounder,
        derivation.surrogate,
        query.treatments[0],
        query.outcomes[0],
    ) {
        return Err(IdentificationError::msg("z_transport.proof_graph_mismatch"));
    }
    let expected =
        match identify_z_transport_surrogate_unchecked(diagram, query, derivation.confounder) {
            Some(expected) => expected,
            None => return Err(IdentificationError::msg("z_transport.proof_formula_mismatch")),
        };
    let same_nodes = derivation.arena.len() == expected.0.len()
        && (0..derivation.arena.len()).all(|index| {
            let id = ExprId::from_raw(index as u32);
            derivation.arena.node(id) == expected.0.node(id)
        })
        && derivation.arena.var_set_count() == expected.0.var_set_count()
        && (0..derivation.arena.var_set_count()).all(|index| {
            let id = antecedent_expr::VarSetId::from_raw(index as u32);
            derivation.arena.var_set(id) == expected.0.var_set(id)
        })
        && derivation.arena.intervention_set_count() == expected.0.intervention_set_count()
        && (0..derivation.arena.intervention_set_count()).all(|index| {
            let id = antecedent_expr::InterventionSetId::from_raw(index as u32);
            derivation.arena.intervention_assignments(id) == expected.0.intervention_assignments(id)
        })
        && derivation.arena.population_count() == expected.0.population_count()
        && (0..derivation.arena.population_count()).all(|index| {
            let id = antecedent_expr::PopulationKeyId::from_raw(index as u32);
            derivation.arena.population(id) == expected.0.population(id)
        })
        && derivation.arena.list_count() == expected.0.list_count()
        && (0..derivation.arena.list_count()).all(|index| {
            let id = antecedent_expr::ExprListId::from_raw(index as u32);
            derivation.arena.list(id) == expected.0.list(id)
        });
    if derivation.root != expected.1 || !same_nodes {
        return Err(IdentificationError::msg("z_transport.proof_formula_mismatch"));
    }
    Ok(())
}

/// Bind the checked surrogate formula to the available full-joint source
/// intervention law named by the query's concrete assignment.
///
/// # Errors
/// Invalid catalog, an incomplete experiment family, or a missing exact joint
/// regime for the formula's selected source intervention.
pub fn bind_z_transport_catalog(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    derivation: &ZTransportDerivation,
    catalog: &EvidenceCatalog,
) -> Result<BoundZTransportFunctional, IdentificationError> {
    verify_z_transport_derivation(diagram, query, derivation)?;
    catalog.validate().map_err(|error| IdentificationError::msg(error.to_string()))?;
    let required = validate_z_experiment_family(diagram, query, catalog).map_err(|error| {
        IdentificationError::msg(format!("z_transport.missing_evidence: {error:?}"))
    })?;
    let assignments = &query.experiment_assignment;
    let mut intervention_variables = assignments.iter().map(|a| a.variable).collect::<Vec<_>>();
    intervention_variables.sort_unstable();
    let measured = diagram
        .causal_graph()
        .nodes()
        .iter()
        .filter_map(|node| match node {
            NodeRef::Static(v) => Some(*v),
            _ => None,
        })
        .collect::<Vec<_>>();
    let selected = catalog
        .regimes
        .iter()
        .find(|regime| {
            required.contains(&regime.id)
                && regime.population.as_ref() == query.source.as_ref()
                && regime.evidence_kind == EvidenceKind::Available
                && regime.kind == RegimeKind::Experimental
                && same_variable_set(&regime.interventions, &intervention_variables)
                && regime.intervention_values.len() == assignments.len()
                && assignments.iter().all(|expected| {
                    regime.intervention_values.iter().any(|actual| {
                        actual.variable == expected.variable
                            && actual.value.as_f64() == expected.value.as_f64()
                    })
                })
                && same_variable_set(&regime.measured, &measured)
                && regime.conditioned_on.is_empty()
                && regime.distribution == DistributionAvailability::Joint
        })
        .ok_or_else(|| IdentificationError::msg("z_transport.missing_selected_joint_regime"))?;

    let mut arena = derivation.arena.clone();
    let ExprNode::SumOut { variables, expr } = derivation.arena.node(derivation.root).clone()
    else {
        return Err(IdentificationError::msg("z_transport.invalid_formula_root"));
    };
    let ExprNode::Product(factors) = derivation.arena.node(expr).clone() else {
        return Err(IdentificationError::msg("z_transport.invalid_formula_product"));
    };
    let mut bound_factors = Vec::new();
    for factor in derivation.arena.list(factors) {
        let ExprNode::Distribution {
            variables,
            conditioned_on,
            intervention,
            domain,
            population,
            ..
        } = derivation.arena.node(*factor).clone()
        else {
            return Err(IdentificationError::msg("z_transport.invalid_formula_leaf"));
        };
        bound_factors.push(arena.intern(ExprNode::Distribution {
            variables,
            conditioned_on,
            intervention,
            domain,
            population,
            regime: Some(selected.id),
        }));
    }
    let bound_list = arena.intern_list(bound_factors);
    let bound_product = arena.intern(ExprNode::Product(bound_list));
    let root = arena.intern(ExprNode::SumOut { variables, expr: bound_product });
    Ok(BoundZTransportFunctional {
        derivation: derivation.clone(),
        arena,
        root,
        catalog: catalog.clone(),
        regime: selected.id,
    })
}

fn identify_z_transport_surrogate_unchecked(
    _diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    confounder: VariableId,
) -> Option<(CausalExprArena, ExprId)> {
    let mut arena = CausalExprArena::new();
    let y_set = arena.intern_var_set([query.outcomes[0]]);
    let w_set = arena.intern_var_set([confounder]);
    let wx_set = arena.intern_var_set([confounder, query.treatments[0]]);
    let z_do =
        arena.intern_intervention_assignments(query.experiment_assignment.iter().map(|a| {
            antecedent_expr::InterventionAssignment::concrete(a.variable, a.value.clone())
        }));
    let source = arena.intern_population(Arc::clone(&query.source));
    let conditional = arena.intern(ExprNode::Distribution {
        variables: y_set,
        conditioned_on: wx_set,
        intervention: z_do,
        domain: DomainRef::Interventional,
        population: source,
        regime: None,
    });
    let empty = arena.empty_var_set();
    let marginal = arena.intern(ExprNode::Distribution {
        variables: w_set,
        conditioned_on: empty,
        intervention: z_do,
        domain: DomainRef::Interventional,
        population: source,
        regime: None,
    });
    let factors = arena.intern_list([conditional, marginal]);
    let product = arena.intern(ExprNode::Product(factors));
    let root = arena.intern(ExprNode::SumOut { variables: w_set, expr: product });
    Some((arena, root))
}

fn matches_registered_surrogate_graph(
    diagram: &SelectionDiagram,
    w: VariableId,
    z: VariableId,
    x: VariableId,
    y: VariableId,
) -> bool {
    let graph = diagram.causal_graph();
    let dense = |variable: VariableId| {
        graph.nodes().iter().position(|node| *node == NodeRef::Static(variable))
    };
    let Some(wi) = dense(w) else { return false };
    let Some(zi) = dense(z) else { return false };
    let Some(xi) = dense(x) else { return false };
    let Some(yi) = dense(y) else { return false };
    let directed = graph
        .nodes()
        .iter()
        .enumerate()
        .flat_map(|(from, _)| {
            graph
                .children(antecedent_graph::DenseNodeId::from_raw(from as u32))
                .iter()
                .map(move |to| (from, to.as_usize()))
        })
        .collect::<std::collections::BTreeSet<_>>();
    let expected_directed = [(wi, zi), (zi, xi), (xi, yi), (wi, yi)]
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    let bidirected = graph
        .nodes()
        .iter()
        .enumerate()
        .flat_map(|(a, _)| {
            graph
                .bidirected_neighbors(antecedent_graph::DenseNodeId::from_raw(a as u32))
                .iter()
                .filter(move |b| a < b.as_usize())
                .map(move |b| (a, b.as_usize()))
        })
        .collect::<std::collections::BTreeSet<_>>();
    let expected_bidirected =
        [(wi.min(yi), wi.max(yi)), (zi.min(yi), zi.max(yi)), (zi.min(xi), zi.max(xi))]
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>();
    directed == expected_directed && bidirected == expected_bidirected
}

fn same_variable_set(left: &[VariableId], right: &[VariableId]) -> bool {
    left.len() == right.len() && left.iter().all(|variable| right.contains(variable))
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{
        DependenceGroup, Environment, RegimeBinding, RegimeId, SamplingDesign, Value,
        VariableCoordinate,
    };
    use antecedent_graph::{Admg, DenseNodeId};

    fn query(graph_size: usize, controllable: &[u32]) -> (SelectionDiagram, ZTransportQuery) {
        let diagram = SelectionDiagram::try_new(
            Admg::with_variables(graph_size as u32),
            Arc::<[VariableId]>::from([VariableId::from_raw(2)]),
        )
        .unwrap();
        let query = ZTransportQuery {
            outcomes: Arc::from([VariableId::from_raw(2)]),
            treatments: Arc::from([VariableId::from_raw(0)]),
            controllable: controllable
                .iter()
                .copied()
                .map(VariableId::from_raw)
                .collect::<Vec<_>>()
                .into(),
            experiment_assignment: controllable
                .first()
                .map(|variable| {
                    Arc::from([CatalogInterventionAssignment {
                        variable: VariableId::from_raw(*variable),
                        value: Value::Bool(false),
                    }])
                })
                .unwrap_or_else(|| Arc::from([])),
            source: Arc::from("source"),
            target: Arc::from("target"),
        };
        (diagram, query)
    }

    #[test]
    fn bounded_query_contract_accepts_declared_controllable_set() {
        let (diagram, query) = query(6, &[1, 2]);
        validate_z_transport_query(&diagram, &query).unwrap();
    }

    #[test]
    fn bounded_query_contract_refuses_oversized_graph_and_controllable_set() {
        let (diagram, oversized_graph_query) = query(7, &[1]);
        assert_eq!(
            validate_z_transport_query(&diagram, &oversized_graph_query).unwrap_err().to_string(),
            "z_transport.unsupported_observed_count"
        );
        let (diagram, oversized_control_query) = query(6, &[0, 1, 2]);
        assert_eq!(
            validate_z_transport_query(&diagram, &oversized_control_query).unwrap_err().to_string(),
            "z_transport.unsupported_controllable_count"
        );
    }

    #[test]
    fn contract_rejects_unknown_and_duplicate_coordinates() {
        let (diagram, mut query) = query(3, &[1]);
        query.controllable = Arc::from([VariableId::from_raw(9)]);
        assert_eq!(
            validate_z_transport_query(&diagram, &query).unwrap_err().to_string(),
            "z_transport.unknown_variable"
        );
        query.controllable = Arc::from([VariableId::from_raw(1), VariableId::from_raw(1)]);
        assert_eq!(
            validate_z_transport_query(&diagram, &query).unwrap_err().to_string(),
            "z_transport.duplicate_variable"
        );
    }

    fn surrogate_fixture() -> (SelectionDiagram, ZTransportQuery) {
        let mut graph = Admg::with_variables(4);
        for (from, to) in [(0, 1), (1, 2), (2, 3), (0, 3)] {
            graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
        }
        for (a, b) in [(0, 3), (1, 3), (1, 2)] {
            graph.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        let diagram = SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from([])).unwrap();
        let query = ZTransportQuery {
            outcomes: Arc::from([VariableId::from_raw(3)]),
            treatments: Arc::from([VariableId::from_raw(2)]),
            controllable: Arc::from([VariableId::from_raw(1)]),
            experiment_assignment: Arc::from([CatalogInterventionAssignment {
                variable: VariableId::from_raw(1),
                value: Value::Bool(false),
            }]),
            source: Arc::from("source"),
            target: Arc::from("target"),
        };
        (diagram, query)
    }

    #[test]
    fn registered_surrogate_query_has_checked_formula_and_rejects_substitution() {
        let (diagram, query) = surrogate_fixture();
        let ZTransportResult::Identified(proof) =
            identify_z_transport_surrogate(&diagram, &query).unwrap()
        else {
            panic!("registered surrogate graph should be identified");
        };
        assert_eq!(proof.surrogate(), VariableId::from_raw(1));
        assert_eq!(proof.confounder(), VariableId::from_raw(0));
        let ExprNode::SumOut { variables, expr } = proof.arena().node(proof.root()) else {
            panic!("formula root must sum out W");
        };
        assert_eq!(proof.arena().var_set(*variables), &[VariableId::from_raw(0)]);
        let ExprNode::Product(factors) = proof.arena().node(*expr) else {
            panic!("formula must multiply two source factors");
        };
        assert_eq!(proof.arena().list(*factors).len(), 2);
        verify_z_transport_derivation(&diagram, &query, &proof).unwrap();
        let mut altered = query.clone();
        altered.controllable = Arc::from([VariableId::from_raw(0)]);
        assert!(verify_z_transport_derivation(&diagram, &altered, &proof).is_err());
    }

    #[test]
    fn formula_matches_independent_binary_scm_truth() {
        // Independent binary exogenous causes: A→W,Y; B→Z,Y; C→Z,X.
        // The three shared exogenous variables induce precisely the fixture's
        // bidirected edges. Non-symmetric probabilities make the check numerical.
        let p_a = 0.25;
        let p_b = 0.20;
        let p_c = 0.35;
        let mut target = [0.0_f64; 2];
        let mut experiment = [[[0.0_f64; 2]; 2]; 2]; // w,x,y under do(Z)
        for a in 0..=1 {
            for b in 0..=1 {
                for c in 0..=1 {
                    let mass = if a == 1 { p_a } else { 1.0 - p_a }
                        * if b == 1 { p_b } else { 1.0 - p_b }
                        * if c == 1 { p_c } else { 1.0 - p_c };
                    let w = a;
                    // Effect truth under do(X=0), marginalizing the graph's
                    // exogenous variables independently of the intervention.
                    let y_do_x = w ^ a ^ b;
                    target[y_do_x] += mass;
                    // The source experiment is do(Z=0); all W, X, Y are jointly
                    // measured and X still shares C with Z.
                    let z = 0;
                    let x = z ^ c;
                    let y = x ^ w ^ a ^ b;
                    experiment[w][x][y] += mass;
                }
            }
        }
        let mut formula = 0.0;
        for w in 0..=1 {
            let pw = experiment[w].iter().flatten().sum::<f64>();
            let y_given_w_x = experiment[w][0][1] / (experiment[w][0][0] + experiment[w][0][1]);
            formula += pw * y_given_w_x;
        }
        assert!((formula - target[1]).abs() < 1e-12, "formula={formula}, truth={}", target[1]);
    }

    #[test]
    fn family_validation_requires_each_joint_intervention_law() {
        let (diagram, query) = surrogate_fixture();
        let coords = (0..4)
            .map(|raw| VariableCoordinate {
                variable: VariableId::from_raw(raw),
                domain: if raw == 1 { VariableDomain::Binary } else { VariableDomain::Binary },
                unit: None,
            })
            .collect::<Vec<_>>();
        let environment =
            Environment::try_new("source", coords, Arc::<[VariableId]>::from([])).unwrap();
        let measured: Arc<[VariableId]> = (0..4).map(VariableId::from_raw).collect();
        let regimes = (0..2)
            .map(|level| {
                antecedent_core::EvidenceRegime::try_new(
                    RegimeId::from_raw(level),
                    RegimeKind::Experimental,
                    EvidenceKind::Available,
                    [VariableId::from_raw(1)],
                    [CatalogInterventionAssignment {
                        variable: VariableId::from_raw(1),
                        value: Value::Bool(level == 1),
                    }],
                    Arc::clone(&measured),
                    "source",
                    DistributionAvailability::Joint,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let bindings = (0..2)
            .map(|level| RegimeBinding {
                dataset_identity: None,
                regime: RegimeId::from_raw(level),
                snapshot_identity: Arc::from(format!("source-do-z-{level}")),
                schema_names: Arc::from([]),
                sampling: SamplingDesign::Independent,
                weights: None,
                dependence: DependenceGroup::IndependentStudies,
            })
            .collect::<Vec<_>>();
        let catalog = EvidenceCatalog {
            environments: Arc::from([environment]),
            regimes: regimes.into(),
            bindings: bindings.into(),
            target_sampling: None,
        };
        assert_eq!(validate_z_experiment_family(&diagram, &query, &catalog).unwrap().len(), 2);

        let mut missing_joint = catalog.clone();
        let mut altered = missing_joint.regimes.to_vec();
        altered.pop();
        missing_joint.regimes = altered.into();
        assert!(matches!(
            validate_z_experiment_family(&diagram, &query, &missing_joint),
            Err(ZExperimentFamilyError::MissingJointLaw { .. })
        ));
        let mut marginal_only = catalog.clone();
        let mut altered = marginal_only.regimes.to_vec();
        altered[0].distribution = DistributionAvailability::SeparateMarginals {
            variables: Arc::from([VariableId::from_raw(0), VariableId::from_raw(3)]),
        };
        marginal_only.regimes = altered.into();
        assert!(matches!(
            validate_z_experiment_family(&diagram, &query, &marginal_only),
            Err(ZExperimentFamilyError::MissingJointLaw { .. })
        ));

        let mut insufficient_measurement = catalog;
        let mut altered = insufficient_measurement.regimes.to_vec();
        altered[0].measured =
            Arc::from([VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(3)]);
        insufficient_measurement.regimes = altered.into();
        assert!(matches!(
            validate_z_experiment_family(&diagram, &query, &insufficient_measurement),
            Err(ZExperimentFamilyError::MissingJointLaw { .. })
        ));
    }
}
