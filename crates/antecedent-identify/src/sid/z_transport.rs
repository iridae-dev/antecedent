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
use antecedent_graph::{BitSet, DenseNodeId, NodeRef, SelectionDiagram};
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
    kind: ZFormulaKind,
    rules: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ZFormulaKind {
    Surrogate,
    DirectJoint,
    Recursive,
}

/// Untrusted portable premises of a restricted-experiment derivation. The
/// expression arena is transported separately and checked against the graph.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ZTransportDerivationRecord {
    /// Stable graph identity, including selection targets.
    pub graph_signature: String,
    /// Surrogate intervention coordinate used by this derivation.
    pub surrogate: u32,
    /// Shared covariate eliminated by the formula.
    pub confounder: u32,
    /// Root expression identifier in the accompanying arena.
    pub root: u32,
    /// Checked rule names, in derivation order.
    pub rules: Vec<String>,
}

/// One reachable operation in a compact, inspectable formula DAG.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ZProofOperation {
    /// Expression node ID.
    pub id: u32,
    /// Operation name.
    pub operation: String,
    /// Child expression node IDs.
    pub children: Vec<u32>,
}

/// A required joint-law factor and the actual catalog entries that can supply it.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ZFactorObligation {
    /// Expression leaf ID.
    pub leaf: u32,
    /// Population that must supply the factor.
    pub population: String,
    /// Joint variable margin required by the formula.
    pub variables: Vec<u32>,
    /// Conditioning coordinates retained by the factor.
    pub conditioned_on: Vec<u32>,
    /// Concrete intervention coordinates and numeric levels.
    pub intervention: Vec<(u32, Option<f64>)>,
    /// Available, bound catalog regime IDs satisfying this factor.
    pub supplied_by: Vec<u32>,
    /// Precise first binding failure when no catalog entry supplies the factor.
    pub failure: Option<String>,
}

/// Formula graph and evidence obligations for a checked z derivation.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ZTransportProofInspection {
    /// Root expression node.
    pub root: u32,
    /// Checked high-level TRz rule scope.
    pub rules: Vec<String>,
    /// Reachable expression DAG in postorder.
    pub operations: Vec<ZProofOperation>,
    /// Every reachable required factor, including unsatisfied ones.
    pub factors: Vec<ZFactorObligation>,
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
    /// Inspect the checked expression and match each required factor to actual
    /// available catalog entries. This read-only view never promotes proposed
    /// evidence into an available binding.
    #[must_use]
    pub fn inspect_proof(&self, catalog: &EvidenceCatalog) -> ZTransportProofInspection {
        let mut inspection = ZTransportProofInspection {
            root: self.root.raw(),
            rules: self.to_record().rules,
            operations: Vec::new(),
            factors: Vec::new(),
        };
        let mut seen = std::collections::BTreeSet::new();
        inspect_z_expression(self.root, &self.arena, catalog, &mut seen, &mut inspection);
        inspection
    }
    /// Export proof premises for an independently checked artifact.
    #[must_use]
    pub fn to_record(&self) -> ZTransportDerivationRecord {
        ZTransportDerivationRecord {
            graph_signature: self.graph_signature.clone(),
            surrogate: self.surrogate.raw(),
            confounder: self.confounder.raw(),
            root: self.root.raw(),
            rules: self.rules.clone(),
        }
    }

    /// Reconstruct authority only after the graph, query, and every expression
    /// node have been checked against the registered derivation.
    ///
    /// # Errors
    /// Returns an error if any recorded premise or expression differs.
    pub fn from_record_checked(
        diagram: &SelectionDiagram,
        query: &ZTransportQuery,
        record: &ZTransportDerivationRecord,
        arena: CausalExprArena,
    ) -> Result<Self, IdentificationError> {
        let candidate = Self {
            query: query.clone(),
            graph_signature: record.graph_signature.clone(),
            surrogate: VariableId::from_raw(record.surrogate),
            confounder: VariableId::from_raw(record.confounder),
            arena,
            root: ExprId::from_raw(record.root),
            kind: match record.rules.first().map(String::as_str) {
                Some("ztr.surrogate_factorization") => ZFormulaKind::Surrogate,
                Some("ztr.source_exchange_joint") => ZFormulaKind::DirectJoint,
                Some("ztr.recursive_reduction") => ZFormulaKind::Recursive,
                _ => return Err(IdentificationError::msg("z_transport.proof_rule_mismatch")),
            },
            rules: record.rules.clone(),
        };
        verify_z_transport_derivation(diagram, query, &candidate)?;
        Ok(candidate)
    }
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

/// Identify a bounded single-source restricted-experiment query.
///
/// The registered surrogate formula and direct joint exchange have dedicated
/// local checkers. Other positive formulas follow the recursive TRz reduction
/// and are rederived on replay. The bounded search does not yet issue a
/// theorem-level negative certificate; a failed branch is `NotCertified`.
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
    identify_z_transport_with_limits(
        diagram,
        query,
        super::SidLimits::default(),
        &antecedent_core::ExecutionContext::for_tests(0),
    )
}

/// Run bounded restricted-experiment identification with explicit resource limits.
///
/// # Errors
/// Invalid query coordinates, cancellation, or exhausted computation.
pub fn identify_z_transport_with_limits(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    limits: super::SidLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<ZTransportResult, IdentificationError> {
    validate_z_transport_query(diagram, query)?;
    if limits.steps == 0 || limits.depth == 0 {
        return Err(IdentificationError::msg("z_transport.exhausted_computation"));
    }
    if ctx.cancellation.is_cancelled() {
        return Err(IdentificationError::msg("z_transport.cancelled"));
    }
    if direct_joint_admissible(diagram, query)? {
        let (arena, root) = direct_joint_formula(query);
        let derivation = ZTransportDerivation {
            query: query.clone(),
            graph_signature: super::graph_signature(diagram),
            surrogate: query.experiment_assignment[0].variable,
            confounder: query.experiment_assignment[0].variable,
            arena,
            root,
            kind: ZFormulaKind::DirectJoint,
            rules: vec!["ztr.source_exchange_joint".into()],
        };
        verify_z_transport_derivation(diagram, query, &derivation)?;
        return Ok(ZTransportResult::Identified(Box::new(derivation)));
    }
    if diagram.causal_graph().node_count() != 4 || !diagram.selection_targets().is_empty() {
        return identify_recursive_or_not_certified(
            diagram,
            query,
            limits,
            ctx,
            "z_transport.no_checked_recursive_formula",
        );
    }
    if query.outcomes.len() != 1 || query.treatments.len() != 1 || query.controllable.len() != 1 {
        return identify_recursive_or_not_certified(
            diagram,
            query,
            limits,
            ctx,
            "z_transport.no_checked_recursive_formula",
        );
    }
    if query.experiment_assignment.len() != 1
        || query.experiment_assignment[0].variable != query.controllable[0]
    {
        return identify_recursive_or_not_certified(
            diagram,
            query,
            limits,
            ctx,
            "z_transport.experiment_assignment_required",
        );
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
        return identify_recursive_or_not_certified(
            diagram,
            query,
            limits,
            ctx,
            "z_transport.no_checked_recursive_formula",
        );
    };
    if !matches_registered_surrogate_graph(diagram, confounder, surrogate, treatment, outcome) {
        return identify_recursive_or_not_certified(
            diagram,
            query,
            limits,
            ctx,
            "z_transport.no_checked_recursive_formula",
        );
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
        kind: ZFormulaKind::Surrogate,
        rules: vec!["ztr.surrogate_factorization".into()],
    };
    verify_z_transport_derivation(diagram, query, &derivation)?;
    Ok(ZTransportResult::Identified(Box::new(derivation)))
}

/// Preferred entry point for the bounded restricted-experiment identifier.
///
/// # Errors
/// Invalid query coordinates or exhausted computation.
pub fn identify_z_transport(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
) -> Result<ZTransportResult, IdentificationError> {
    identify_z_transport_surrogate(diagram, query)
}

fn identify_recursive_or_not_certified(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    limits: super::SidLimits,
    ctx: &antecedent_core::ExecutionContext,
    reason: &'static str,
) -> Result<ZTransportResult, IdentificationError> {
    let searched = search_trz(diagram, query, limits, ctx).map_err(|error| {
        if error.to_string() == "transport.identification_budget" {
            IdentificationError::msg("z_transport.exhausted_computation")
        } else {
            error
        }
    })?;
    let Some((arena, root, trace)) = searched else {
        return Ok(ZTransportResult::NotCertified { reason });
    };
    let derivation = ZTransportDerivation {
        query: query.clone(),
        graph_signature: super::graph_signature(diagram),
        surrogate: query.controllable[0],
        confounder: query.outcomes[0],
        arena,
        root,
        kind: ZFormulaKind::Recursive,
        rules: std::iter::once("ztr.recursive_reduction".to_owned()).chain(trace).collect(),
    };
    verify_z_transport_derivation(diagram, query, &derivation)?;
    Ok(ZTransportResult::Identified(Box::new(derivation)))
}

// A direct TRz source exchange is legal when the requested intervention is a
// concrete subset of the controllable family and the population selectors are
// separated from the outcome in the graph with incoming treatment arrows cut.
// The source experiment must still be bound to real evidence by the caller.
fn direct_joint_admissible(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
) -> Result<bool, IdentificationError> {
    if query.experiment_assignment.len() != query.treatments.len()
        || !query.treatments.iter().all(|x| {
            query.controllable.contains(x)
                && query.experiment_assignment.iter().any(|a| a.variable == *x)
        })
    {
        return Ok(false);
    }
    let graph = diagram.causal_graph();
    let dense = |v: VariableId| {
        graph
            .nodes()
            .iter()
            .position(|node| *node == NodeRef::Static(v))
            .map(|i| DenseNodeId::from_raw(i as u32))
            .ok_or_else(|| IdentificationError::msg("z_transport.unknown_variable"))
    };
    let mut all = BitSet::with_len(graph.node_count());
    for index in 0..graph.node_count() {
        all.insert(DenseNodeId::from_raw(index as u32));
    }
    let mut intervened = BitSet::with_len(graph.node_count());
    for x in query.treatments.iter().copied() {
        intervened.insert(dense(x)?);
    }
    let selectors =
        diagram.selection_targets().iter().copied().map(dense).collect::<Result<Vec<_>, _>>()?;
    let outcomes = query.outcomes.iter().copied().map(dense).collect::<Result<Vec<_>, _>>()?;
    let graph = super::MutilatedSelection::build(graph, &all, &intervened, &selectors)?;
    graph.separates(
        &outcomes,
        &intervened.to_dense_ids(),
        &mut antecedent_graph::DSeparationWorkspace::default(),
    )
}

fn direct_joint_formula(query: &ZTransportQuery) -> (CausalExprArena, ExprId) {
    let mut arena = CausalExprArena::new();
    let variables = arena.intern_var_set(query.outcomes.iter().copied());
    let conditioned_on = arena.empty_var_set();
    let intervention =
        arena.intern_intervention_assignments(query.experiment_assignment.iter().map(|a| {
            antecedent_expr::InterventionAssignment::concrete(a.variable, a.value.clone())
        }));
    let population = arena.intern_population(Arc::clone(&query.source));
    let root = arena.intern(ExprNode::Distribution {
        variables,
        conditioned_on,
        intervention,
        domain: DomainRef::Interventional,
        population,
        regime: None,
    });
    (arena, root)
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
    if derivation.kind == ZFormulaKind::Recursive {
        if derivation.query != *query
            || derivation.graph_signature != super::graph_signature(diagram)
            || derivation.surrogate != query.controllable[0]
            || derivation.confounder != query.outcomes[0]
        {
            return Err(IdentificationError::msg("z_transport.proof_input_mismatch"));
        }
        let Some((expected, root, trace)) = search_trz(
            diagram,
            query,
            super::SidLimits::default(),
            &antecedent_core::ExecutionContext::for_tests(0),
        )?
        else {
            return Err(IdentificationError::msg("z_transport.proof_rule_mismatch"));
        };
        let expected_rules =
            std::iter::once("ztr.recursive_reduction".to_owned()).chain(trace).collect::<Vec<_>>();
        if derivation.root != root
            || !same_arena(&derivation.arena, &expected)
            || derivation.rules != expected_rules
        {
            return Err(IdentificationError::msg("z_transport.proof_formula_mismatch"));
        }
        return Ok(());
    }
    if derivation.kind == ZFormulaKind::DirectJoint {
        if derivation.query != *query
            || derivation.graph_signature != super::graph_signature(diagram)
            || !direct_joint_admissible(diagram, query)?
            || derivation.surrogate != query.experiment_assignment[0].variable
            || derivation.confounder != derivation.surrogate
            || derivation.rules != ["ztr.source_exchange_joint"]
        {
            return Err(IdentificationError::msg("z_transport.proof_input_mismatch"));
        }
        let (expected, root) = direct_joint_formula(query);
        if derivation.root != root || !same_arena(&derivation.arena, &expected) {
            return Err(IdentificationError::msg("z_transport.proof_formula_mismatch"));
        }
        return Ok(());
    }
    if derivation.query != *query
        || derivation.graph_signature != super::graph_signature(diagram)
        || diagram.causal_graph().node_count() != 4
        || !diagram.selection_targets().is_empty()
        || query.outcomes.len() != 1
        || query.treatments.len() != 1
        || query.controllable.as_ref() != [derivation.surrogate]
        || query.experiment_assignment.len() != 1
        || query.experiment_assignment[0].variable != derivation.surrogate
        || derivation.rules != ["ztr.surrogate_factorization"]
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

/// Bind the checked surrogate formula to a sufficient source joint margin.
///
/// # Errors
/// Invalid catalog or a missing joint margin covering the formula variables.
pub fn bind_z_transport_catalog(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    derivation: &ZTransportDerivation,
    catalog: &EvidenceCatalog,
) -> Result<BoundZTransportFunctional, IdentificationError> {
    verify_z_transport_derivation(diagram, query, derivation)?;
    catalog.validate().map_err(|error| IdentificationError::msg(error.to_string()))?;
    if derivation.kind == ZFormulaKind::Recursive {
        let mut arena = derivation.arena.clone();
        let mut memo = std::collections::HashMap::new();
        let mut cited = Vec::new();
        let root =
            bind_recursive_expression(derivation.root, &mut arena, catalog, &mut memo, &mut cited)?;
        let regime = cited.iter().copied().next().ok_or_else(|| {
            IdentificationError::msg("z_transport.recursive_formula_has_no_factor")
        })?;
        return Ok(BoundZTransportFunctional {
            derivation: derivation.clone(),
            arena,
            root,
            catalog: catalog.clone(),
            regime,
        });
    }
    if derivation.kind == ZFormulaKind::DirectJoint {
        let assignment_variables =
            query.experiment_assignment.iter().map(|a| a.variable).collect::<Vec<_>>();
        let selected = catalog
            .regimes
            .iter()
            .find(|regime| {
                regime.population.as_ref() == query.source.as_ref()
                    && regime.evidence_kind == EvidenceKind::Available
                    && regime.kind == RegimeKind::Experimental
                    && same_variable_set(&regime.interventions, &assignment_variables)
                    && regime.intervention_values.len() == query.experiment_assignment.len()
                    && query.experiment_assignment.iter().all(|expected| {
                        regime.intervention_values.iter().any(|actual| {
                            actual.variable == expected.variable
                                && actual.value.as_f64() == expected.value.as_f64()
                        })
                    })
                    && query.outcomes.iter().all(|y| regime.measured.contains(y))
                    && regime.conditioned_on.is_empty()
                    && regime.distribution == DistributionAvailability::Joint
                    && catalog.bindings.iter().any(|binding| binding.regime == regime.id)
            })
            .ok_or_else(|| {
                IdentificationError::msg(format!(
                    "z_transport.missing_evidence: {} joint law under do({:?}) measuring {:?}",
                    query.source, query.experiment_assignment, query.outcomes
                ))
            })?;
        let mut arena = derivation.arena.clone();
        let ExprNode::Distribution {
            variables,
            conditioned_on,
            intervention,
            domain,
            population,
            ..
        } = arena.node(derivation.root).clone()
        else {
            return Err(IdentificationError::msg("z_transport.invalid_formula_root"));
        };
        let root = arena.intern(ExprNode::Distribution {
            variables,
            conditioned_on,
            intervention,
            domain,
            population,
            regime: Some(selected.id),
        });
        return Ok(BoundZTransportFunctional {
            derivation: derivation.clone(),
            arena,
            root,
            catalog: catalog.clone(),
            regime: selected.id,
        });
    }
    let assignments = &query.experiment_assignment;
    let mut intervention_variables = assignments.iter().map(|a| a.variable).collect::<Vec<_>>();
    intervention_variables.sort_unstable();
    // Both source leaves can be obtained by marginalizing a joint law on
    // {Y, W, X}. Requiring every observed coordinate (or the complete
    // experimental family) here would turn a sufficient positive certificate
    // into a spurious missing-evidence result.
    let required_margin = [query.outcomes[0], derivation.confounder, query.treatments[0]];
    let selected = catalog
        .regimes
        .iter()
        .find(|regime| {
            regime.population.as_ref() == query.source.as_ref()
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
                && required_margin.iter().all(|variable| regime.measured.contains(variable))
                && regime.conditioned_on.is_empty()
                && regime.distribution == DistributionAvailability::Joint
                && catalog.bindings.iter().any(|binding| binding.regime == regime.id)
        })
        .ok_or_else(|| {
            IdentificationError::msg(format!(
                "z_transport.missing_evidence: {} joint law under do({:?}) measuring {:?}",
                query.source, query.experiment_assignment, required_margin
            ))
        })?;

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

fn bind_recursive_expression(
    id: ExprId,
    arena: &mut CausalExprArena,
    catalog: &EvidenceCatalog,
    memo: &mut std::collections::HashMap<ExprId, ExprId>,
    cited: &mut Vec<antecedent_core::RegimeId>,
) -> Result<ExprId, IdentificationError> {
    if let Some(hit) = memo.get(&id) {
        return Ok(*hit);
    }
    let node = arena.node(id).clone();
    let bound =
        match node {
            ExprNode::Distribution {
                variables,
                conditioned_on,
                intervention,
                domain,
                population,
                ..
            } => {
                let name = arena.population(population);
                let interventions = arena.intervention_assignments(intervention);
                let intervention_vars =
                    interventions.iter().map(|a| a.variable).collect::<Vec<_>>();
                let regime = catalog.regimes.iter().filter(|regime| {
                regime.population.as_ref() == name
                    && regime.evidence_kind == EvidenceKind::Available
                    && regime.kind == if interventions.is_empty() {
                        RegimeKind::Observational
                    } else { RegimeKind::Experimental }
                    && same_variable_set(&regime.interventions, &intervention_vars)
                    && regime.intervention_values.len() == interventions.len()
                    && interventions.iter().all(|expected| regime.intervention_values.iter()
                        .any(|actual| actual.variable == expected.variable
                            && actual.value == expected.value))
                    && arena.var_set(variables).iter().chain(arena.var_set(conditioned_on))
                        .all(|v| regime.measured.contains(v))
                    && regime.conditioned_on.is_empty()
                    && regime.distribution == DistributionAvailability::Joint
                    && catalog.bindings.iter().any(|binding| binding.regime == regime.id)
            }).min_by_key(|regime| regime.id.raw())
                .ok_or_else(|| IdentificationError::msg(format!(
                    "z_transport.missing_evidence: {name} joint factor under {interventions:?}"
                )))?;
                cited.push(regime.id);
                arena.intern(ExprNode::Distribution {
                    variables,
                    conditioned_on,
                    intervention,
                    domain,
                    population,
                    regime: Some(regime.id),
                })
            }
            ExprNode::Product(list) => {
                let children = arena.list(list).to_vec();
                let bound_children = children
                    .into_iter()
                    .map(|child| bind_recursive_expression(child, arena, catalog, memo, cited))
                    .collect::<Result<Vec<_>, _>>()?;
                let list = arena.intern_list(bound_children);
                arena.intern(ExprNode::Product(list))
            }
            ExprNode::SumOut { variables, expr } => {
                let expr = bind_recursive_expression(expr, arena, catalog, memo, cited)?;
                arena.intern(ExprNode::SumOut { variables, expr })
            }
            ExprNode::Ratio { numerator, denominator } => {
                let numerator = bind_recursive_expression(numerator, arena, catalog, memo, cited)?;
                let denominator =
                    bind_recursive_expression(denominator, arena, catalog, memo, cited)?;
                arena.intern(ExprNode::Ratio { numerator, denominator })
            }
            _ => return Err(IdentificationError::msg("z_transport.unsupported_recursive_node")),
        };
    memo.insert(id, bound);
    Ok(bound)
}

fn same_arena(left: &CausalExprArena, right: &CausalExprArena) -> bool {
    left.len() == right.len()
        && (0..left.len()).all(|i| {
            left.node(ExprId::from_raw(i as u32)) == right.node(ExprId::from_raw(i as u32))
        })
        && left.var_set_count() == right.var_set_count()
        && (0..left.var_set_count()).all(|i| {
            left.var_set(antecedent_expr::VarSetId::from_raw(i as u32))
                == right.var_set(antecedent_expr::VarSetId::from_raw(i as u32))
        })
        && left.intervention_set_count() == right.intervention_set_count()
        && (0..left.intervention_set_count()).all(|i| {
            left.intervention_assignments(antecedent_expr::InterventionSetId::from_raw(i as u32))
                == right.intervention_assignments(antecedent_expr::InterventionSetId::from_raw(
                    i as u32,
                ))
        })
        && left.population_count() == right.population_count()
        && (0..left.population_count()).all(|i| {
            left.population(antecedent_expr::PopulationKeyId::from_raw(i as u32))
                == right.population(antecedent_expr::PopulationKeyId::from_raw(i as u32))
        })
        && left.list_count() == right.list_count()
        && (0..left.list_count()).all(|i| {
            left.list(antecedent_expr::ExprListId::from_raw(i as u32))
                == right.list(antecedent_expr::ExprListId::from_raw(i as u32))
        })
}

fn inspect_z_expression(
    id: ExprId,
    arena: &CausalExprArena,
    catalog: &EvidenceCatalog,
    seen: &mut std::collections::BTreeSet<u32>,
    result: &mut ZTransportProofInspection,
) {
    if !seen.insert(id.raw()) {
        return;
    }
    let node = arena.node(id);
    let (operation, children): (&str, Vec<ExprId>) = match node {
        ExprNode::Distribution { .. } => ("distribution", Vec::new()),
        ExprNode::Product(list) => ("product", arena.list(*list).to_vec()),
        ExprNode::SumOut { expr, .. } => ("sum_out", vec![*expr]),
        ExprNode::Ratio { numerator, denominator } => ("ratio", vec![*numerator, *denominator]),
        ExprNode::Kernel { body, .. } => ("kernel", vec![*body]),
        ExprNode::IntegralOut { expr, .. } => ("integral_out", vec![*expr]),
        ExprNode::Expectation { distribution, .. } => ("expectation", vec![*distribution]),
        ExprNode::Contrast { left, right, .. } => ("contrast", vec![*left, *right]),
    };
    for child in &children {
        inspect_z_expression(*child, arena, catalog, seen, result);
    }
    result.operations.push(ZProofOperation {
        id: id.raw(),
        operation: operation.to_owned(),
        children: children.iter().map(|child| child.raw()).collect(),
    });
    let ExprNode::Distribution { variables, conditioned_on, intervention, population, .. } = node
    else {
        return;
    };
    let name = arena.population(*population);
    let vars = arena.var_set(*variables);
    let conditions = arena.var_set(*conditioned_on);
    let assignments = arena.intervention_assignments(*intervention);
    let intervention_vars = assignments.iter().map(|a| a.variable).collect::<Vec<_>>();
    let mut exact_regime = false;
    let mut missing_joint = false;
    let mut missing_margin = false;
    let mut missing_binding = false;
    let mut supplied_by = Vec::new();
    for regime in catalog.regimes.iter().filter(|regime| {
        regime.population.as_ref() == name
            && regime.evidence_kind == EvidenceKind::Available
            && regime.kind
                == if assignments.is_empty() {
                    RegimeKind::Observational
                } else {
                    RegimeKind::Experimental
                }
            && same_variable_set(&regime.interventions, &intervention_vars)
            && regime.intervention_values.len() == assignments.len()
            && assignments.iter().all(|expected| {
                regime.intervention_values.iter().any(|actual| {
                    actual.variable == expected.variable && actual.value == expected.value
                })
            })
    }) {
        exact_regime = true;
        if regime.distribution != DistributionAvailability::Joint
            || !regime.conditioned_on.is_empty()
        {
            missing_joint = true;
        } else if !vars.iter().chain(conditions).all(|v| regime.measured.contains(v)) {
            missing_margin = true;
        } else if !catalog.bindings.iter().any(|binding| binding.regime == regime.id) {
            missing_binding = true;
        } else {
            supplied_by.push(regime.id.raw());
        }
    }
    let failure = if !supplied_by.is_empty() {
        None
    } else if !exact_regime {
        Some("missing_matching_population_regime_or_intervention_values")
    } else if missing_joint {
        Some("joint_law_required")
    } else if missing_margin {
        Some("measured_joint_margin_insufficient")
    } else if missing_binding {
        Some("provider_binding_missing")
    } else {
        Some("factor_binding_failed")
    };
    result.factors.push(ZFactorObligation {
        leaf: id.raw(),
        population: name.to_owned(),
        variables: vars.iter().map(|v| v.raw()).collect(),
        conditioned_on: conditions.iter().map(|v| v.raw()).collect(),
        intervention: assignments.iter().map(|a| (a.variable.raw(), a.value.as_f64())).collect(),
        supplied_by,
        failure: failure.map(str::to_owned),
    });
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

// TRz recursion uses the ordinary ID kernel operations for lines 1–8. The
// restricted source exchange at line 10 changes the carried law to a *joint*
// source experiment on the currently relevant controllable coordinates, then
// runs ordinary ID on that experimental kernel. Keep this search separate from
// the published derivation until its broader proof/evidence binding is checked.
#[allow(dead_code)]
fn search_trz(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    limits: super::SidLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<Option<(CausalExprArena, ExprId, Vec<String>)>, IdentificationError> {
    let classical = super::ClassicalTransportQuery {
        outcomes: Arc::clone(&query.outcomes),
        treatments: Arc::clone(&query.treatments),
        source: Arc::clone(&query.source),
        target: Arc::clone(&query.target),
    };
    let mut engine = super::Engine::new(diagram, &classical, limits, ctx)?;
    let initial = engine.initial()?;
    let mut trace = Vec::new();
    let result = search_trz_state(&mut engine, initial, query, 0, &mut trace)?;
    Ok(result.map(|root| (engine.arena, root, trace)))
}

#[allow(dead_code)]
fn search_trz_state(
    engine: &mut super::Engine<'_>,
    state: super::State,
    query: &ZTransportQuery,
    depth: usize,
    trace: &mut Vec<String>,
) -> Result<Option<ExprId>, IdentificationError> {
    engine.charge(depth)?;
    let mut ws = antecedent_graph::GraphWorkspace::default();
    if !state.x.any() {
        trace.push("ztr.line1.marginal".into());
        return Ok(Some(engine.marginal(state.kernel, &super::difference(&state.v, &state.y))?));
    }
    let ancestors = engine.prepared.ancestors_within(&state.y, &state.v, &mut ws);
    if !ancestors.equal_set(&state.v) {
        trace.push("ztr.line2.ancestors".into());
        let next = super::State {
            y: state.y.clone(),
            x: super::intersection(&state.x, &ancestors),
            v: ancestors.clone(),
            kernel: engine.marginal(state.kernel, &super::difference(&state.v, &ancestors))?,
        };
        return search_trz_state(engine, next, query, depth + 1, trace);
    }
    let mut irrelevant = super::difference(&state.v, &state.x);
    let bar = engine.prepared.ancestors_bar_x(&state.y, &state.v, &state.x, &mut ws);
    irrelevant.difference_with(&bar);
    if irrelevant.any() {
        trace.push("ztr.line3.enlarge".into());
        let mut next = state.clone();
        next.x.union_with(&irrelevant);
        let Some(child) = search_trz_state(engine, next, query, depth + 1, trace)? else {
            return Ok(None);
        };
        return Ok(Some(engine.enlarge_output(&state, &irrelevant, child)?));
    }
    let districts = engine.prepared.c_components(&super::difference(&state.v, &state.x));
    if districts.len() > 1 {
        trace.push("ztr.line4.districts".into());
        let mut expressions = Vec::with_capacity(districts.len());
        for district in districts {
            let next = super::State {
                y: district.clone(),
                x: super::difference(&state.v, &district),
                ..state.clone()
            };
            let Some(child) = search_trz_state(engine, next, query, depth + 1, trace)? else {
                return Ok(None);
            };
            expressions.push(child);
        }
        let product = engine.product(expressions);
        return Ok(Some(engine.marginal(
            product,
            &super::difference(&super::difference(&state.v, &state.x), &state.y),
        )?));
    }
    let district =
        districts.first().ok_or_else(|| IdentificationError::msg("z_transport.empty_district"))?;
    let containing = engine.prepared.c_components(&state.v);
    if containing.len() > 1 {
        if containing.iter().any(|d| d.equal_set(district)) {
            trace.push("ztr.line7.factor".into());
            let kernel = engine.factor(&state, district)?;
            return Ok(Some(engine.marginal(kernel, &super::difference(district, &state.y))?));
        }
        let larger = containing
            .iter()
            .find(|d| district.is_subset_of(d))
            .ok_or_else(|| IdentificationError::msg("z_transport.missing_containing_district"))?;
        let next = super::State {
            y: state.y.clone(),
            x: super::intersection(&state.x, larger),
            v: larger.clone(),
            kernel: engine.factor(&state, larger)?,
        };
        trace.push("ztr.line8.recurse".into());
        return search_trz_state(engine, next, query, depth + 1, trace);
    }

    let mut activated = BitSet::with_len(engine.diagram.causal_graph().node_count());
    for variable in query.controllable.iter().copied() {
        let dense = engine.prepared.var_to_dense(variable)?;
        if state.x.contains(dense) {
            activated.insert(dense);
        }
    }
    if !activated.any() || !engine.source_admissible(&state)? {
        return Ok(None);
    }
    let active_vars = engine.vars(&activated)?;
    let Some(assignments) = active_vars
        .iter()
        .map(|v| query.experiment_assignment.iter().find(|a| a.variable == *v))
        .collect::<Option<Vec<_>>>()
    else {
        return Ok(None);
    };
    trace.push(format!(
        "ztr.line10.source_exchange:{:?}",
        assignments.iter().map(|a| (a.variable.raw(), a.value.as_f64())).collect::<Vec<_>>()
    ));
    let remaining = super::difference(&state.v, &activated);
    let variables = engine.arena.intern_var_set(engine.vars(&remaining)?);
    // A district kernel can retain external parent coordinates after line 8.
    // Its source replacement must keep those coordinates as parameters; a
    // marginal source law would silently use the source parent distribution.
    let mut external_parents = BitSet::with_len(engine.diagram.causal_graph().node_count());
    for node in remaining.to_dense_ids() {
        for parent in engine.diagram.causal_graph().parents(node) {
            if !state.v.contains(*parent) && !activated.contains(*parent) {
                external_parents.insert(*parent);
            }
        }
    }
    let conditioned_on = engine.arena.intern_var_set(engine.vars(&external_parents)?);
    let intervention =
        engine.arena.intern_intervention_assignments(assignments.iter().map(|a| {
            antecedent_expr::InterventionAssignment::concrete(a.variable, a.value.clone())
        }));
    let population = engine.arena.intern_population(Arc::clone(&query.source));
    let kernel = engine.arena.intern(ExprNode::Distribution {
        variables,
        conditioned_on,
        intervention,
        domain: DomainRef::Interventional,
        population,
        regime: None,
    });
    let next = super::State {
        y: state.y,
        x: super::difference(&state.x, &activated),
        v: remaining,
        kernel,
    };
    let proof_start = engine.proof.len();
    let Some(step) = engine.solve(next, false, depth + 1)? else { return Ok(None) };
    trace.extend(
        engine.proof[proof_start..]
            .iter()
            .map(|step| format!("ztr.source_id.{}", step.rule.name())),
    );
    Ok(Some(engine.proof[step].output))
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
    fn direct_exchange_requires_concrete_joint_intervention_and_checks_proof() {
        let mut graph = Admg::with_variables(3);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
        graph.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
        let diagram = SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from([])).unwrap();
        let query = ZTransportQuery {
            outcomes: Arc::from([VariableId::from_raw(2)]),
            treatments: Arc::from([VariableId::from_raw(0), VariableId::from_raw(1)]),
            controllable: Arc::from([VariableId::from_raw(0), VariableId::from_raw(1)]),
            experiment_assignment: Arc::from([
                CatalogInterventionAssignment {
                    variable: VariableId::from_raw(0),
                    value: Value::Bool(true),
                },
                CatalogInterventionAssignment {
                    variable: VariableId::from_raw(1),
                    value: Value::Bool(false),
                },
            ]),
            source: Arc::from("source"),
            target: Arc::from("target"),
        };
        let ZTransportResult::Identified(proof) =
            identify_z_transport_surrogate(&diagram, &query).unwrap()
        else {
            panic!("direct joint source exchange must be identified")
        };
        assert_eq!(proof.to_record().rules, ["ztr.source_exchange_joint"]);
        assert!(matches!(proof.arena().node(proof.root()), ExprNode::Distribution { .. }));
        let reconstructed = ZTransportDerivation::from_record_checked(
            &diagram,
            &query,
            &proof.to_record(),
            proof.arena().clone(),
        )
        .unwrap();
        verify_z_transport_derivation(&diagram, &query, &reconstructed).unwrap();
        let mut changed = query;
        changed.experiment_assignment = Arc::from([
            CatalogInterventionAssignment {
                variable: VariableId::from_raw(0),
                value: Value::Bool(false),
            },
            CatalogInterventionAssignment {
                variable: VariableId::from_raw(1),
                value: Value::Bool(false),
            },
        ]);
        assert!(verify_z_transport_derivation(&diagram, &changed, &proof).is_err());
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
    fn recursive_trz_search_reaches_registered_surrogate() {
        let (diagram, query) = surrogate_fixture();
        let answer = search_trz(
            &diagram,
            &query,
            super::super::SidLimits::default(),
            &antecedent_core::ExecutionContext::for_tests(7),
        )
        .unwrap();
        assert!(answer.is_some(), "TRz recursion should find the surrogate case");
    }

    #[test]
    fn recursive_trz_surrogate_formula_matches_independent_scm() {
        use antecedent_expr::{
            Assignment, DiscreteAxis, EvalContext, ExactDiscreteLaw, ExactTransportData,
            InterventionAssignment as ExprAssignment, LawTolerance,
        };
        let (diagram, query) = surrogate_fixture();
        let (arena, root, trace) = search_trz(
            &diagram,
            &query,
            super::super::SidLimits::default(),
            &antecedent_core::ExecutionContext::for_tests(7),
        )
        .unwrap()
        .unwrap();
        assert!(trace.iter().any(|rule| rule.starts_with("ztr.line10.source_exchange:")));
        let mut bound = arena.clone();
        let mut remap = Vec::<ExprId>::new();
        for index in 0..arena.len() {
            let node = match arena.node(ExprId::from_raw(index as u32)).clone() {
                ExprNode::Distribution {
                    variables,
                    conditioned_on,
                    intervention,
                    domain,
                    population,
                    ..
                } => ExprNode::Distribution {
                    variables,
                    conditioned_on,
                    intervention,
                    domain,
                    population,
                    regime: Some(if arena.population(population) == "source" {
                        RegimeId::from_raw(1)
                    } else {
                        RegimeId::from_raw(0)
                    }),
                },
                ExprNode::SumOut { variables, expr } => {
                    ExprNode::SumOut { variables, expr: remap[expr.raw() as usize] }
                }
                ExprNode::Ratio { numerator, denominator } => ExprNode::Ratio {
                    numerator: remap[numerator.raw() as usize],
                    denominator: remap[denominator.raw() as usize],
                },
                ExprNode::Product(list) => {
                    let children = arena
                        .list(list)
                        .iter()
                        .map(|id| remap[id.raw() as usize])
                        .collect::<Vec<_>>();
                    ExprNode::Product(bound.intern_list(children))
                }
                other => panic!("unexpected recursive node: {other:?}"),
            };
            remap.push(bound.intern(node));
        }
        let root = remap[root.raw() as usize];
        let mut target = vec![0.0; 16];
        let mut source = vec![0.0; 8];
        for a in 0..=1usize {
            for b in 0..=1usize {
                for c in 0..=1usize {
                    let mass = (if a == 1 { 0.25 } else { 0.75 })
                        * (if b == 1 { 0.20 } else { 0.80 })
                        * (if c == 1 { 0.35 } else { 0.65 });
                    let w = a;
                    let z = w ^ b ^ c;
                    let x = z ^ c;
                    let y = x ^ (w & b);
                    target[w * 8 + z * 4 + x * 2 + y] += mass;
                    let sx = c;
                    let sy = sx ^ (w & b);
                    source[w * 4 + sx * 2 + sy] += mass;
                }
            }
        }
        let axis = |raw| DiscreteAxis {
            variable: VariableId::from_raw(raw),
            values: Arc::from([Value::Bool(false), Value::Bool(true)]),
        };
        let target_law = ExactDiscreteLaw::try_new(
            "target",
            RegimeId::from_raw(0),
            [],
            [axis(0), axis(1), axis(2), axis(3)],
            target,
            "target-obs",
            LawTolerance::default(),
        )
        .unwrap();
        let source_law = ExactDiscreteLaw::try_new(
            "source",
            RegimeId::from_raw(1),
            [ExprAssignment::concrete(VariableId::from_raw(1), Value::Bool(false))],
            [axis(0), axis(2), axis(3)],
            source,
            "source-do-z0",
            LawTolerance::default(),
        )
        .unwrap();
        let provider = ExactTransportData::try_new([target_law, source_law], 128).unwrap();
        let assignment = Assignment::from_pairs([
            (VariableId::from_raw(2), Value::Bool(false)),
            (VariableId::from_raw(3), Value::Bool(true)),
        ]);
        let answer = bound
            .compile(root)
            .unwrap()
            .evaluate_with(&bound, &provider, &EvalContext::default(), &assignment)
            .unwrap();
        assert!((answer - 0.05).abs() < 1e-12, "recursive answer={answer}");
    }

    #[test]
    fn recursive_route_binds_nonregistered_graph_using_actual_margins() {
        let mut graph = Admg::with_variables(5);
        for (a, b) in [(0, 1), (1, 2), (2, 3), (0, 3)] {
            graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        for (a, b) in [(0, 3), (1, 3), (1, 2)] {
            graph.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        let diagram = SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from([])).unwrap();
        let (_, query) = surrogate_fixture();
        let ZTransportResult::Identified(proof) =
            identify_z_transport_surrogate(&diagram, &query).unwrap()
        else {
            panic!("recursive TRz should identify the graph with an isolated extra node")
        };
        assert_eq!(
            proof.to_record().rules.first().map(String::as_str),
            Some("ztr.recursive_reduction")
        );
        let coords = (0..5)
            .map(|raw| VariableCoordinate {
                variable: VariableId::from_raw(raw),
                domain: VariableDomain::Binary,
                unit: None,
            })
            .collect::<Vec<_>>();
        let source =
            Environment::try_new("source", coords.clone(), Arc::<[VariableId]>::from([])).unwrap();
        let target = Environment::try_new("target", coords, Arc::<[VariableId]>::from([])).unwrap();
        let obs = antecedent_core::EvidenceRegime::try_new(
            RegimeId::from_raw(0),
            RegimeKind::Observational,
            EvidenceKind::Available,
            [],
            [],
            (0..5).map(VariableId::from_raw).collect::<Vec<_>>(),
            "target",
            DistributionAvailability::Joint,
        )
        .unwrap();
        let experiment = antecedent_core::EvidenceRegime::try_new(
            RegimeId::from_raw(1),
            RegimeKind::Experimental,
            EvidenceKind::Available,
            [VariableId::from_raw(1)],
            [CatalogInterventionAssignment {
                variable: VariableId::from_raw(1),
                value: Value::Bool(false),
            }],
            [VariableId::from_raw(0), VariableId::from_raw(2), VariableId::from_raw(3)],
            "source",
            DistributionAvailability::Joint,
        )
        .unwrap();
        let bindings = [0, 1].map(|raw| RegimeBinding {
            dataset_identity: None,
            regime: RegimeId::from_raw(raw),
            snapshot_identity: Arc::from(format!("snapshot-{raw}")),
            schema_names: Arc::from([]),
            sampling: SamplingDesign::Independent,
            weights: None,
            dependence: DependenceGroup::IndependentStudies,
        });
        let catalog =
            EvidenceCatalog::try_new([source, target], [obs, experiment], bindings, None).unwrap();
        let bound = bind_z_transport_catalog(&diagram, &query, &proof, &catalog).unwrap();
        assert_eq!(bound.catalog().regimes.len(), 2);
        ZTransportDerivation::from_record_checked(
            &diagram,
            &query,
            &proof.to_record(),
            proof.arena().clone(),
        )
        .unwrap();
    }

    #[test]
    fn bounded_search_reports_exhaustion_separately_from_noncertification() {
        let (diagram, query) = surrogate_fixture();
        let exhausted = identify_z_transport_with_limits(
            &diagram,
            &query,
            super::super::SidLimits { steps: 0, depth: 1 },
            &antecedent_core::ExecutionContext::for_tests(1),
        )
        .unwrap_err();
        assert_eq!(exhausted.to_string(), "z_transport.exhausted_computation");
    }

    #[test]
    fn recursive_search_handles_selected_mediator_surrogate_graph() {
        use antecedent_expr::{
            Assignment, DiscreteAxis, EvalContext, ExactDiscreteLaw, ExactTransportData,
            InterventionAssignment as ExprAssignment, LawTolerance,
        };
        // A source Z experiment combines with target W distribution. This is
        // structurally distinct from the registered W→Z surrogate graph.
        let mut graph = Admg::with_variables(4);
        for (a, b) in [(0, 1), (1, 2), (1, 3), (2, 3)] {
            graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        for (a, b) in [(0, 1), (0, 3)] {
            graph.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        let diagram =
            SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from([VariableId::from_raw(2)]))
                .unwrap();
        let query = ZTransportQuery {
            outcomes: Arc::from([VariableId::from_raw(3)]),
            treatments: Arc::from([VariableId::from_raw(1)]),
            controllable: Arc::from([VariableId::from_raw(0)]),
            experiment_assignment: Arc::from([CatalogInterventionAssignment {
                variable: VariableId::from_raw(0),
                value: Value::Bool(false),
            }]),
            source: Arc::from("source"),
            target: Arc::from("target"),
        };
        let result = identify_z_transport(&diagram, &query).unwrap();
        let ZTransportResult::Identified(proof) = result else { panic!("{result:?}") };
        let coords = (0..4)
            .map(|raw| VariableCoordinate {
                variable: VariableId::from_raw(raw),
                domain: VariableDomain::Binary,
                unit: None,
            })
            .collect::<Vec<_>>();
        let source_env =
            Environment::try_new("source", coords.clone(), Arc::<[VariableId]>::from([])).unwrap();
        let target_env =
            Environment::try_new("target", coords, Arc::<[VariableId]>::from([])).unwrap();
        let target_regime = antecedent_core::EvidenceRegime::try_new(
            RegimeId::from_raw(0),
            RegimeKind::Observational,
            EvidenceKind::Available,
            [],
            [],
            (0..4).map(VariableId::from_raw).collect::<Vec<_>>(),
            "target",
            DistributionAvailability::Joint,
        )
        .unwrap();
        let source_regime = antecedent_core::EvidenceRegime::try_new(
            RegimeId::from_raw(1),
            RegimeKind::Experimental,
            EvidenceKind::Available,
            [VariableId::from_raw(0)],
            [CatalogInterventionAssignment {
                variable: VariableId::from_raw(0),
                value: Value::Bool(false),
            }],
            [VariableId::from_raw(1), VariableId::from_raw(2), VariableId::from_raw(3)],
            "source",
            DistributionAvailability::Joint,
        )
        .unwrap();
        let bindings = [0, 1].map(|raw| RegimeBinding {
            dataset_identity: None,
            regime: RegimeId::from_raw(raw),
            snapshot_identity: Arc::from(format!("selected-{raw}")),
            schema_names: Arc::from([]),
            sampling: SamplingDesign::Independent,
            weights: None,
            dependence: DependenceGroup::IndependentStudies,
        });
        let catalog = EvidenceCatalog::try_new(
            [source_env, target_env],
            [target_regime, source_regime],
            bindings,
            None,
        )
        .unwrap();
        let functional = bind_z_transport_catalog(&diagram, &query, &proof, &catalog).unwrap();
        let mut target = vec![0.0; 16];
        let mut source = vec![0.0; 8];
        let mass = |a: usize, b: usize, e: usize, p_e: f64| {
            (if a == 1 { 0.35 } else { 0.65 })
                * (if b == 1 { 0.20 } else { 0.80 })
                * (if e == 1 { p_e } else { 1.0 - p_e })
        };
        for a in 0..=1usize {
            for b in 0..=1usize {
                for e in 0..=1usize {
                    let z = a ^ b;
                    let x = z ^ a;
                    let w = x ^ e;
                    let y = x ^ w ^ b;
                    target[z * 8 + x * 4 + w * 2 + y] += mass(a, b, e, 0.25);
                    let sx = a;
                    let sw = sx ^ e;
                    let sy = sx ^ sw ^ b;
                    source[sx * 4 + sw * 2 + sy] += mass(a, b, e, 0.65);
                }
            }
        }
        let axis = |raw| DiscreteAxis {
            variable: VariableId::from_raw(raw),
            values: Arc::from([Value::Bool(false), Value::Bool(true)]),
        };
        let target_law = ExactDiscreteLaw::try_new(
            "target",
            RegimeId::from_raw(0),
            [],
            [axis(0), axis(1), axis(2), axis(3)],
            target,
            "selected-0",
            LawTolerance::default(),
        )
        .unwrap();
        let source_law = ExactDiscreteLaw::try_new(
            "source",
            RegimeId::from_raw(1),
            [ExprAssignment::concrete(VariableId::from_raw(0), Value::Bool(false))],
            [axis(1), axis(2), axis(3)],
            source,
            "selected-1",
            LawTolerance::default(),
        )
        .unwrap();
        let data = ExactTransportData::try_new([target_law, source_law], 128).unwrap();
        for treatment in [false, true] {
            let request = Assignment::from_pairs([
                (VariableId::from_raw(1), Value::Bool(treatment)),
                (VariableId::from_raw(3), Value::Bool(true)),
            ]);
            let value = functional
                .arena()
                .compile(functional.root())
                .unwrap()
                .evaluate_with(functional.arena(), &data, &EvalContext::default(), &request)
                .unwrap();
            assert!((value - 0.35).abs() < 1e-12, "x={treatment}, formula={value}");
        }
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

        // The positive formula uses only the selected do(Z=0) margin on
        // (W,X,Y). The unselected level and the measured Z coordinate are not
        // premises of this certificate.
        let ZTransportResult::Identified(proof) =
            identify_z_transport_surrogate(&diagram, &query).unwrap()
        else {
            panic!()
        };
        let mut sufficient = catalog.clone();
        let mut regimes = sufficient.regimes.to_vec();
        regimes.truncate(1);
        regimes[0].measured =
            Arc::from([VariableId::from_raw(0), VariableId::from_raw(2), VariableId::from_raw(3)]);
        sufficient.regimes = regimes.into();
        sufficient.bindings = Arc::from([sufficient.bindings[0].clone()]);
        bind_z_transport_catalog(&diagram, &query, &proof, &sufficient).unwrap();
        let inspection = proof.inspect_proof(&sufficient);
        assert_eq!(inspection.factors.len(), 2);
        assert!(inspection.factors.iter().all(|factor| factor.failure.is_none()));
        assert!(inspection.factors.iter().all(|factor| factor.supplied_by == [0]));

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
        let inspection = proof.inspect_proof(&marginal_only);
        assert!(
            inspection
                .factors
                .iter()
                .any(|factor| factor.failure.as_deref() == Some("joint_law_required"))
        );
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
