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
    InterventionAssignment as CatalogInterventionAssignment, RegimeKind, Value, VariableDomain,
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
        /// Observed variable with an unsupported or undeclared finite domain.
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

/// Why complete theorem inputs are not yet available for a negative decision.
#[derive(Clone, Debug, PartialEq)]
pub enum ZTransportMissingEvidence {
    /// No source environment declaration is available.
    SourceEnvironment,
    /// The target observational joint law over all observed variables is absent.
    TargetObservationalJoint,
    /// A required source experiment is missing or is not a joint law.
    SourceExperiment(ZExperimentFamilyError),
}

/// Result of the bounded TRz decision when its declared data family is checked.
#[derive(Clone, Debug)]
pub enum ZTransportDecision {
    /// A positive TRz formula was derived.
    Identified(Box<ZTransportDerivation>),
    /// A checked TRz line-11 obstruction was reached with the complete data family.
    ProvenNonTransportable(Box<ZTransportObstruction>),
    /// A theorem input law is not present in the catalog.
    MissingEvidence {
        /// Exact missing evidence requirement.
        missing: ZTransportMissingEvidence,
    },
    /// Search completed without a replayable line-11 terminal state.
    NotCertified {
        /// Stable scope note.
        reason: &'static str,
    },
}

/// A checked bounded TRz failure tied to the complete experimental family.
#[derive(Clone, Debug)]
pub struct ZTransportObstruction {
    query: ZTransportQuery,
    graph_signature: String,
    selected_assignment: Vec<(u32, f64)>,
    family_regimes: Vec<u32>,
    terminal: TrzTerminalFailure,
}

/// Portable line-11 premises. Query and catalog are supplied separately and
/// must be identical in meaning when this record is checked.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ZTransportObstructionRecord {
    /// Stable graph identity, including selection targets.
    pub graph_signature: String,
    /// Canonical finite assignment used while replaying TRz.
    pub selected_assignment: Vec<(u32, f64)>,
    /// Regime IDs supplying the complete source experimental family.
    pub family_regimes: Vec<u32>,
    /// Reduced terminal subproblem at TRz line 11.
    pub terminal: ZTransportTerminalRecord,
}

/// Portable reduced TRz state for the terminal failure rule.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ZTransportTerminalRecord {
    /// Outcome coordinates in the reduced problem.
    pub outcomes: Vec<u32>,
    /// Intervention coordinates in the reduced problem.
    pub treatments: Vec<u32>,
    /// Vertices in the reduced selection diagram.
    pub vertices: Vec<u32>,
    /// The sole c-component of the reduced graph after removing treatments (C0).
    #[serde(default)]
    pub c0: Vec<u32>,
    /// Controllable coordinates remaining after previous exchanges.
    pub remaining_controllable: Vec<u32>,
    /// Remaining controllables overlapping the reduced treatment set (Z ∩ X).
    #[serde(default)]
    pub candidate_active: Vec<u32>,
    /// Previously activated intervention coordinates and values.
    pub active_interventions: Vec<(u32, f64)>,
    /// Whether selection nodes are separated from outcomes given treatments.
    pub selection_separated: bool,
    /// Recursive TRz rules used to reach this terminal state.
    pub rules: Vec<String>,
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
    let target =
        catalog.environments.iter().find(|environment| environment.identity == query.target);
    let observed = diagram
        .causal_graph()
        .nodes()
        .iter()
        .filter_map(|node| match node {
            NodeRef::Static(variable) => Some(*variable),
            _ => None,
        })
        .collect::<Vec<_>>();
    for variable in &observed {
        for environment in [source, target].into_iter().flatten() {
            let domain = environment
                .variables
                .iter()
                .find(|coordinate| coordinate.variable == *variable)
                .map(|coordinate| &coordinate.domain);
            if !matches!(domain, Some(VariableDomain::Binary | VariableDomain::Categorical { .. }))
            {
                return Err(ZExperimentFamilyError::UnsupportedDomain { variable: *variable });
            }
        }
    }
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

fn has_target_observational_joint(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    catalog: &EvidenceCatalog,
) -> bool {
    let observed = diagram
        .causal_graph()
        .nodes()
        .iter()
        .filter_map(|node| match node {
            NodeRef::Static(variable) => Some(*variable),
            _ => None,
        })
        .collect::<Vec<_>>();
    catalog.regimes.iter().any(|regime| {
        regime.kind == RegimeKind::Observational
            && regime.evidence_kind == EvidenceKind::Available
            && regime.population.as_ref() == query.target.as_ref()
            && regime.interventions.is_empty()
            && regime.intervention_values.is_empty()
            && same_variable_set(&regime.measured, &observed)
            && regime.conditioned_on.is_empty()
            && regime.distribution == DistributionAvailability::Joint
            && catalog.bindings.iter().any(|binding| binding.regime == regime.id)
    })
}

fn query_with_complete_assignment(
    query: &ZTransportQuery,
    catalog: &EvidenceCatalog,
) -> Result<ZTransportQuery, IdentificationError> {
    if query.controllable.iter().all(|variable| {
        query.experiment_assignment.iter().any(|assignment| assignment.variable == *variable)
    }) {
        return Ok(query.clone());
    }
    let source = catalog
        .environments
        .iter()
        .find(|environment| environment.identity == query.source)
        .ok_or_else(|| IdentificationError::msg("z_transport.source_environment_missing"))?;
    let mut assignments = query.experiment_assignment.to_vec();
    for variable in query.controllable.iter().copied() {
        let coordinate = source
            .variables
            .iter()
            .find(|coordinate| coordinate.variable == variable)
            .ok_or_else(|| IdentificationError::msg("z_transport.unsupported_domain"))?;
        let (levels, default_value) = match coordinate.domain {
            VariableDomain::Binary => (vec![0.0, 1.0], Value::Bool(false)),
            VariableDomain::Categorical { cardinality } => {
                ((0..cardinality).map(f64::from).collect(), Value::Category(0))
            }
            _ => return Err(IdentificationError::msg("z_transport.unsupported_domain")),
        };
        if let Some(assignment) = assignments.iter().find(|a| a.variable == variable) {
            let selected = assignment.value.as_f64().ok_or_else(|| {
                IdentificationError::msg("z_transport.invalid_experiment_assignment")
            })?;
            if !levels.contains(&selected) {
                return Err(IdentificationError::msg("z_transport.invalid_experiment_assignment"));
            }
        } else {
            assignments.push(CatalogInterventionAssignment { variable, value: default_value });
        }
    }
    Ok(ZTransportQuery {
        outcomes: Arc::clone(&query.outcomes),
        treatments: Arc::clone(&query.treatments),
        controllable: Arc::clone(&query.controllable),
        experiment_assignment: assignments.into(),
        source: Arc::clone(&query.source),
        target: Arc::clone(&query.target),
    })
}

/// Decide the bounded single-source z-transport problem under the complete
/// discrete source experiment family and the target observational joint law.
///
/// Positive formulas follow the recursive TRz rules. A negative result is
/// certified only when the recursive search reaches TRz line 11, the complete
/// source experiment family and target observational joint are available, and
/// an independent checker confirms the reduced graph, C0, Z∩X, and failed
/// line-10 separation premise. Missing evidence and exhausted search remain
/// distinct from theorem obstruction.
///
/// # Errors
/// Invalid or unsupported query, cancellation, or exhausted search.
pub fn decide_z_transport_with_catalog(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    catalog: &EvidenceCatalog,
    limits: super::SidLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<ZTransportDecision, IdentificationError> {
    validate_z_transport_query(diagram, query)?;
    catalog.validate().map_err(|_| IdentificationError::msg("z_transport.invalid_catalog"))?;
    if limits.steps == 0 || limits.depth == 0 {
        return Err(IdentificationError::msg("z_transport.exhausted_computation"));
    }
    if ctx.cancellation.is_cancelled() {
        return Err(IdentificationError::msg("z_transport.cancelled"));
    }
    let search_query = match query_with_complete_assignment(query, catalog) {
        Ok(query) => query,
        Err(error) if error.to_string() == "z_transport.source_environment_missing" => {
            return Ok(ZTransportDecision::MissingEvidence {
                missing: ZTransportMissingEvidence::SourceEnvironment,
            });
        }
        Err(error) => return Err(error),
    };
    if direct_joint_admissible(diagram, &search_query)? {
        let (arena, root) = direct_joint_formula(&search_query);
        let derivation = ZTransportDerivation {
            query: search_query.clone(),
            graph_signature: super::graph_signature(diagram),
            surrogate: search_query.experiment_assignment[0].variable,
            confounder: search_query.experiment_assignment[0].variable,
            arena,
            root,
            kind: ZFormulaKind::DirectJoint,
            rules: vec!["ztr.source_exchange_joint".into()],
        };
        verify_z_transport_derivation(diagram, &search_query, &derivation)?;
        return Ok(ZTransportDecision::Identified(Box::new(derivation)));
    }
    if !has_target_observational_joint(diagram, query, catalog) {
        return Ok(ZTransportDecision::MissingEvidence {
            missing: ZTransportMissingEvidence::TargetObservationalJoint,
        });
    }
    if !catalog.environments.iter().any(|environment| environment.identity == query.source) {
        return Ok(ZTransportDecision::MissingEvidence {
            missing: ZTransportMissingEvidence::SourceEnvironment,
        });
    }
    let family = match validate_z_experiment_family(diagram, query, catalog) {
        Ok(family) => family,
        Err(missing @ ZExperimentFamilyError::MissingJointLaw { .. }) => {
            return Ok(ZTransportDecision::MissingEvidence {
                missing: ZTransportMissingEvidence::SourceExperiment(missing),
            });
        }
        Err(ZExperimentFamilyError::UnsupportedDomain { variable }) => {
            return Err(IdentificationError::msg(format!(
                "z_transport.unsupported_domain: {variable}"
            )));
        }
    };
    let searched = search_trz_detailed(diagram, &search_query, limits, ctx).map_err(|error| {
        if error.to_string() == "transport.identification_budget" {
            IdentificationError::msg("z_transport.exhausted_computation")
        } else {
            error
        }
    })?;
    if let Some((arena, root, trace)) = searched.identified {
        let derivation = ZTransportDerivation {
            query: search_query.clone(),
            graph_signature: super::graph_signature(diagram),
            surrogate: search_query.controllable[0],
            confounder: search_query.outcomes[0],
            arena,
            root,
            kind: ZFormulaKind::Recursive,
            rules: std::iter::once("ztr.recursive_reduction".to_owned()).chain(trace).collect(),
        };
        verify_z_transport_derivation(diagram, &search_query, &derivation)?;
        return Ok(ZTransportDecision::Identified(Box::new(derivation)));
    }
    let Some(terminal) = searched.terminal_failure else {
        return Ok(ZTransportDecision::NotCertified {
            reason: "z_transport.no_replayable_line11_failure",
        });
    };
    let obstruction = ZTransportObstruction {
        query: query.clone(),
        graph_signature: super::graph_signature(diagram),
        selected_assignment: search_query
            .experiment_assignment
            .iter()
            .filter_map(|assignment| {
                assignment.value.as_f64().map(|value| (assignment.variable.raw(), value))
            })
            .collect(),
        family_regimes: family.iter().map(|regime| regime.raw()).collect(),
        terminal,
    };
    verify_z_transport_obstruction(diagram, query, catalog, &obstruction, limits, ctx)?;
    Ok(ZTransportDecision::ProvenNonTransportable(Box::new(obstruction)))
}

impl ZTransportObstruction {
    /// The original structural query.
    #[must_use]
    pub const fn query(&self) -> &ZTransportQuery {
        &self.query
    }

    /// Export the reduced terminal state and theorem evidence identities.
    #[must_use]
    pub fn to_record(&self) -> ZTransportObstructionRecord {
        ZTransportObstructionRecord {
            graph_signature: self.graph_signature.clone(),
            selected_assignment: self.selected_assignment.clone(),
            family_regimes: self.family_regimes.clone(),
            terminal: ZTransportTerminalRecord {
                outcomes: self.terminal.outcomes.clone(),
                treatments: self.terminal.treatments.clone(),
                vertices: self.terminal.vertices.clone(),
                c0: self.terminal.c0.clone(),
                remaining_controllable: self.terminal.remaining_controllable.clone(),
                candidate_active: self.terminal.candidate_active.clone(),
                active_interventions: self.terminal.active_interventions.clone(),
                selection_separated: self.terminal.selection_separated,
                rules: self.terminal.rules.clone(),
            },
        }
    }

    /// Reconstruct an obstruction only after full-family and line-11 replay.
    ///
    /// # Errors
    /// Changed graph, query, source family, target law, or terminal search state.
    pub fn from_record_checked(
        record: ZTransportObstructionRecord,
        diagram: &SelectionDiagram,
        query: &ZTransportQuery,
        catalog: &EvidenceCatalog,
        limits: super::SidLimits,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<Self, IdentificationError> {
        let checked = match decide_z_transport_with_catalog(diagram, query, catalog, limits, ctx)? {
            ZTransportDecision::ProvenNonTransportable(obstruction) => *obstruction,
            ZTransportDecision::MissingEvidence { .. } => {
                return Err(IdentificationError::msg("z_transport.obstruction_missing_evidence"));
            }
            ZTransportDecision::Identified(_) => {
                return Err(IdentificationError::msg("z_transport.obstruction_not_reproduced"));
            }
            ZTransportDecision::NotCertified { .. } => {
                return Err(IdentificationError::msg("z_transport.obstruction_not_reproduced"));
            }
        };
        if checked.to_record() != record {
            return Err(IdentificationError::msg("z_transport.obstruction_record_mismatch"));
        }
        Ok(checked)
    }
}

/// Independently replay a bounded negative z-transport certificate.
///
/// # Errors
/// The complete data family is absent, the query differs, or TRz does not
/// reach the recorded line-11 state.
pub fn verify_z_transport_obstruction(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    catalog: &EvidenceCatalog,
    obstruction: &ZTransportObstruction,
    limits: super::SidLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<(), IdentificationError> {
    validate_z_transport_query(diagram, query)?;
    if obstruction.query != *query || obstruction.graph_signature != super::graph_signature(diagram)
    {
        return Err(IdentificationError::msg("z_transport.obstruction_input_mismatch"));
    }
    catalog
        .validate()
        .map_err(|_| IdentificationError::msg("z_transport.obstruction_catalog_invalid"))?;
    if !has_target_observational_joint(diagram, query, catalog) {
        return Err(IdentificationError::msg("z_transport.obstruction_target_law_missing"));
    }
    let family = validate_z_experiment_family(diagram, query, catalog)
        .map_err(|_| IdentificationError::msg("z_transport.obstruction_family_incomplete"))?;
    if family.iter().map(|regime| regime.raw()).collect::<Vec<_>>() != obstruction.family_regimes {
        return Err(IdentificationError::msg("z_transport.obstruction_family_mismatch"));
    }
    verify_trz_line11_terminal(diagram, query, &obstruction.terminal, ctx)?;
    let search_query = query_with_complete_assignment(query, catalog)?;
    let assignment = search_query
        .experiment_assignment
        .iter()
        .filter_map(|intervention| {
            intervention.value.as_f64().map(|value| (intervention.variable.raw(), value))
        })
        .collect::<Vec<_>>();
    if assignment != obstruction.selected_assignment {
        return Err(IdentificationError::msg("z_transport.obstruction_assignment_mismatch"));
    }
    let replay = search_trz_detailed(diagram, &search_query, limits, ctx)?;
    if replay.identified.is_some()
        || replay.terminal_failure.as_ref() != Some(&obstruction.terminal)
    {
        return Err(IdentificationError::msg("z_transport.obstruction_replay_mismatch"));
    }
    Ok(())
}

/// Check the local graph premises of Bareinboim–Pearl TRz Figure 4 line 11
/// independently of the recursive search's terminal flag and separation
/// helper. Their Theorem 5 maps this failed line-11 state `(D, C0)` to a
/// zs-hedge; `verify_z_transport_obstruction` separately replays the prefix.
fn verify_trz_line11_terminal(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    terminal: &TrzTerminalFailure,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<(), IdentificationError> {
    let bad = || IdentificationError::msg("z_transport.invalid_line11_terminal");
    let classical_query = super::ClassicalTransportQuery {
        outcomes: Arc::clone(&query.outcomes),
        treatments: Arc::clone(&query.treatments),
        source: Arc::clone(&query.source),
        target: Arc::clone(&query.target),
    };
    let engine = super::Engine::new(diagram, &classical_query, super::SidLimits::default(), ctx)?;
    let set = |raws: &[u32]| -> Result<BitSet, IdentificationError> {
        let variables = raws.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>();
        if variables.iter().copied().collect::<std::collections::BTreeSet<_>>().len()
            != variables.len()
        {
            return Err(bad());
        }
        engine.set(&variables).map_err(|_| bad())
    };
    let y = set(&terminal.outcomes)?;
    let x = set(&terminal.treatments)?;
    let v = set(&terminal.vertices)?;
    if !y.any()
        || !x.any()
        || y.to_dense_ids().iter().any(|node| !v.contains(*node))
        || x.to_dense_ids().iter().any(|node| !v.contains(*node))
    {
        return Err(bad());
    }
    let mut candidate_active = Vec::new();
    for raw in &terminal.remaining_controllable {
        let variable = VariableId::from_raw(*raw);
        let dense = engine.prepared.var_to_dense(variable).map_err(|_| bad())?;
        if x.contains(dense) {
            candidate_active.push(*raw);
        }
    }
    candidate_active.sort_unstable();
    if candidate_active != terminal.candidate_active {
        return Err(bad());
    }
    let c0 = super::difference(&v, &x);
    let districts = engine.prepared.c_components(&c0);
    if districts.len() != 1 || engine.prepared.c_components(&v).len() != 1 {
        return Err(bad());
    }
    let mut c0_variables = engine.vars(&districts[0])?.iter().map(|v| v.raw()).collect::<Vec<_>>();
    c0_variables.sort_unstable();
    if c0_variables != terminal.c0 {
        return Err(bad());
    }
    let state = super::State { y, x, v, kernel: ExprId::from_raw(0) };
    let separation = engine.independently_admissible(&state, &[], diagram.selection_targets())?;
    if separation != terminal.selection_separated
        || (!candidate_active.is_empty() && separation)
        || terminal.rules.last().map(String::as_str) != Some("ztr.line11.fail")
    {
        return Err(bad());
    }
    Ok(())
}

/// Identify a bounded single-source restricted-experiment query.
///
/// The registered surrogate formula and direct joint exchange have dedicated
/// local checkers. Other positive formulas follow the recursive TRz reduction
/// and are rederived on replay. Use [`decide_z_transport_with_catalog`] when a
/// negative result must be distinguished from incomplete catalog evidence.
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

// Bareinboim–Pearl (AAAI 2013, Figure 4) rules 1–4 map to the marginal,
// ancestor, enlargement, and district branches below; rules 5–8 map to the
// C-component, factor, and subdistrict branches. Rule 10 exchanges the active
// controllables in X and recurses on the reduced graph; rule 11 is recorded
// only when that exchange's separation premise fails or Z∩X is empty.
#[allow(dead_code)]
fn search_trz(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    limits: super::SidLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<Option<(CausalExprArena, ExprId, Vec<String>)>, IdentificationError> {
    Ok(search_trz_detailed(diagram, query, limits, ctx)?.identified)
}

#[derive(Clone, Debug, PartialEq)]
struct TrzTerminalFailure {
    outcomes: Vec<u32>,
    treatments: Vec<u32>,
    vertices: Vec<u32>,
    c0: Vec<u32>,
    remaining_controllable: Vec<u32>,
    candidate_active: Vec<u32>,
    active_interventions: Vec<(u32, f64)>,
    selection_separated: bool,
    rules: Vec<String>,
}

#[derive(Debug)]
struct TrzSearchResult {
    identified: Option<(CausalExprArena, ExprId, Vec<String>)>,
    terminal_failure: Option<TrzTerminalFailure>,
}

fn search_trz_detailed(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    limits: super::SidLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<TrzSearchResult, IdentificationError> {
    let classical = super::ClassicalTransportQuery {
        outcomes: Arc::clone(&query.outcomes),
        treatments: Arc::clone(&query.treatments),
        source: Arc::clone(&query.source),
        target: Arc::clone(&query.target),
    };
    let mut engine = super::Engine::new(diagram, &classical, limits, ctx)?;
    let initial = engine.initial()?;
    let mut trace = Vec::new();
    let mut terminal_failure = None;
    let result = search_trz_state(
        &mut engine,
        initial,
        query,
        &query.controllable,
        &[],
        0,
        &mut trace,
        &mut terminal_failure,
    )?;
    Ok(TrzSearchResult {
        identified: result.map(|root| (engine.arena, root, trace)),
        terminal_failure,
    })
}

#[allow(dead_code)]
fn search_trz_state(
    engine: &mut super::Engine<'_>,
    state: super::State,
    query: &ZTransportQuery,
    remaining_controllable: &[VariableId],
    active_interventions: &[CatalogInterventionAssignment],
    depth: usize,
    trace: &mut Vec<String>,
    terminal_failure: &mut Option<TrzTerminalFailure>,
) -> Result<Option<ExprId>, IdentificationError> {
    engine.charge(depth)?;
    let mut ws = antecedent_graph::GraphWorkspace::default();
    // TRz rule 1: no active treatment coordinates remain.
    if !state.x.any() {
        trace.push("ztr.line1.marginal".into());
        return Ok(Some(engine.marginal(state.kernel, &super::difference(&state.v, &state.y))?));
    }
    let ancestors = engine.prepared.ancestors_within(&state.y, &state.v, &mut ws);
    // TRz rule 2: restrict to ancestors of the current outcomes.
    if !ancestors.equal_set(&state.v) {
        trace.push("ztr.line2.ancestors".into());
        let next = super::State {
            y: state.y.clone(),
            x: super::intersection(&state.x, &ancestors),
            v: ancestors.clone(),
            kernel: engine.marginal(state.kernel, &super::difference(&state.v, &ancestors))?,
        };
        return search_trz_state(
            engine,
            next,
            query,
            remaining_controllable,
            active_interventions,
            depth + 1,
            trace,
            terminal_failure,
        );
    }
    let mut irrelevant = super::difference(&state.v, &state.x);
    let bar = engine.prepared.ancestors_bar_x(&state.y, &state.v, &state.x, &mut ws);
    irrelevant.difference_with(&bar);
    // TRz rule 3: add non-treatment vertices outside An(Y) in D_X.
    if irrelevant.any() {
        trace.push("ztr.line3.enlarge".into());
        let mut next = state.clone();
        next.x.union_with(&irrelevant);
        let Some(child) = search_trz_state(
            engine,
            next,
            query,
            remaining_controllable,
            active_interventions,
            depth + 1,
            trace,
            terminal_failure,
        )?
        else {
            return Ok(None);
        };
        return Ok(Some(engine.enlarge_output(&state, &irrelevant, child)?));
    }
    let districts = engine.prepared.c_components(&super::difference(&state.v, &state.x));
    // TRz rule 4: factor over multiple districts in D \ X.
    if districts.len() > 1 {
        trace.push("ztr.line4.districts".into());
        let mut expressions = Vec::with_capacity(districts.len());
        for district in districts {
            let next = super::State {
                y: district.clone(),
                x: super::difference(&state.v, &district),
                ..state.clone()
            };
            let Some(child) = search_trz_state(
                engine,
                next,
                query,
                remaining_controllable,
                active_interventions,
                depth + 1,
                trace,
                terminal_failure,
            )?
            else {
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
    // TRz rules 5–6 establish C0 and whether D is a single c-component.
    if containing.len() > 1 {
        if containing.iter().any(|d| d.equal_set(district)) {
            // TRz rule 7: C0 is a c-component of D, so its kernel is available.
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
        // TRz rule 8: recurse into the containing c-component of D.
        trace.push("ztr.line8.recurse".into());
        return search_trz_state(
            engine,
            next,
            query,
            remaining_controllable,
            active_interventions,
            depth + 1,
            trace,
            terminal_failure,
        );
    }

    let mut activated = BitSet::with_len(engine.diagram.causal_graph().node_count());
    for variable in remaining_controllable.iter().copied() {
        let dense = engine.prepared.var_to_dense(variable)?;
        if state.x.contains(dense) {
            activated.insert(dense);
        }
    }
    let selection_separated = engine.source_admissible(&state)?;
    if !activated.any() || !selection_separated {
        // TRz rule 11: rule 10 cannot exchange an active experiment.
        if terminal_failure.is_none() {
            let mut c0 = engine.vars(district)?.iter().map(|v| v.raw()).collect::<Vec<_>>();
            c0.sort_unstable();
            let mut candidate_active =
                engine.vars(&activated)?.iter().map(|v| v.raw()).collect::<Vec<_>>();
            candidate_active.sort_unstable();
            trace.push("ztr.line11.fail".into());
            *terminal_failure = Some(TrzTerminalFailure {
                outcomes: engine.vars(&state.y)?.iter().map(|v| v.raw()).collect(),
                treatments: engine.vars(&state.x)?.iter().map(|v| v.raw()).collect(),
                vertices: engine.vars(&state.v)?.iter().map(|v| v.raw()).collect(),
                c0,
                remaining_controllable: remaining_controllable.iter().map(|v| v.raw()).collect(),
                candidate_active,
                active_interventions: active_interventions
                    .iter()
                    .filter_map(|a| a.value.as_f64().map(|value| (a.variable.raw(), value)))
                    .collect(),
                selection_separated,
                rules: trace.clone(),
            });
        }
        return Ok(None);
    }
    let active_vars = engine.vars(&activated)?;
    let Some(assignments) = active_vars
        .iter()
        .map(|v| query.experiment_assignment.iter().find(|a| a.variable == *v))
        .collect::<Option<Vec<_>>>()
    else {
        return Err(IdentificationError::msg("z_transport.experiment_assignment_required"));
    };
    trace.push(format!(
        "ztr.line10.source_exchange:{:?}",
        assignments.iter().map(|a| (a.variable.raw(), a.value.as_f64())).collect::<Vec<_>>()
    ));
    // TRz rule 10: exchange Z∩X and recurse with Z\X and active set I=Z∩X.
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
    let mut cumulative = active_interventions.to_vec();
    for assignment in assignments {
        if !cumulative.iter().any(|a| a.variable == assignment.variable) {
            cumulative.push(assignment.clone());
        }
    }
    let intervention =
        engine.arena.intern_intervention_assignments(cumulative.iter().map(|a| {
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
    // Fig. 4 line 10 recurses into TRz on the reduced graph and retains
    // experiments over the remaining controllable variables.
    let remaining = remaining_controllable
        .iter()
        .copied()
        .filter(|variable| !active_vars.contains(variable))
        .collect::<Vec<_>>();
    search_trz_state(
        engine,
        next,
        query,
        &remaining,
        &cumulative,
        depth + 1,
        trace,
        terminal_failure,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{
        DependenceGroup, Environment, RegimeBinding, RegimeId, SamplingDesign, Value,
        VariableCoordinate,
    };
    use antecedent_expr::{Assignment, DistributionProvider, EvalContext, EvalError, FactorSpec};
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

    struct SequentialExchangeScm;

    impl DistributionProvider for SequentialExchangeScm {
        fn probability(
            &self,
            spec: &FactorSpec<'_>,
            assignment: &Assignment,
            _ctx: &EvalContext,
        ) -> Result<f64, EvalError> {
            let mut intervention = [None; 4];
            for item in spec.intervention {
                let value = item.value.as_f64().ok_or(EvalError::MissingBinding(item.variable))?;
                intervention[item.variable.raw() as usize] = Some(usize::from(value == 1.0));
            }
            let mut denominator = 0.0;
            let mut numerator = 0.0;
            for exogenous in 0..16 {
                let a = exogenous & 1;
                let b = (exogenous >> 1) & 1;
                let noise_z = (exogenous >> 2) & 1;
                let noise_t = (exogenous >> 3) & 1;
                let x = intervention[0].unwrap_or(a ^ b);
                let t = intervention[1].unwrap_or(noise_t);
                let z = intervention[2].unwrap_or(x ^ a ^ noise_z);
                let y = intervention[3].unwrap_or(z ^ b);
                let state = [x, t, z, y];
                let matches = |variables: &[VariableId]| {
                    variables.iter().all(|variable| {
                        assignment.get(*variable).and_then(Value::as_f64).is_some_and(|value| {
                            usize::from(value == 1.0) == state[variable.raw() as usize]
                        })
                    })
                };
                if matches(spec.conditioned_on) {
                    denominator += 1.0 / 16.0;
                    if matches(spec.variables) {
                        numerator += 1.0 / 16.0;
                    }
                }
            }
            if denominator == 0.0 {
                return Err(EvalError::DivisionByZero);
            }
            Ok(numerator / denominator)
        }

        fn support(
            &self,
            variables: &[VariableId],
            _ctx: &EvalContext,
        ) -> Result<Arc<[Arc<[Value]>]>, EvalError> {
            Ok((0..(1_usize << variables.len()))
                .map(|mask| {
                    variables
                        .iter()
                        .enumerate()
                        .map(|(index, _)| Value::f64(((mask >> index) & 1) as f64))
                        .collect::<Arc<[_]>>()
                })
                .collect::<Vec<_>>()
                .into())
        }

        fn outcome(
            &self,
            variable: VariableId,
            assignment: &Assignment,
            _ctx: &EvalContext,
        ) -> Result<f64, EvalError> {
            assignment
                .get(variable)
                .and_then(Value::as_f64)
                .ok_or(EvalError::MissingBinding(variable))
        }

        fn n_draws(&self) -> Option<usize> {
            None
        }
    }

    #[test]
    fn recursive_line10_keeps_remaining_controllables_and_matches_exact_scm() {
        // SCM: A and B are independent binary causes of X; A also causes Z,
        // B also causes Y, and the directed chain is X -> Z -> Y. T is
        // independent. The latent projection has X<->Z and X<->Y.
        let mut graph = Admg::with_variables(4);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
        graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(3)).unwrap();
        graph.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
        graph.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(3)).unwrap();
        let diagram = SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from([])).unwrap();
        let query = ZTransportQuery {
            outcomes: Arc::from([VariableId::from_raw(3)]),
            treatments: Arc::from([VariableId::from_raw(0), VariableId::from_raw(1)]),
            controllable: Arc::from([VariableId::from_raw(0), VariableId::from_raw(2)]),
            experiment_assignment: Arc::from([
                CatalogInterventionAssignment {
                    variable: VariableId::from_raw(0),
                    value: Value::Bool(false),
                },
                CatalogInterventionAssignment {
                    variable: VariableId::from_raw(2),
                    value: Value::Bool(false),
                },
            ]),
            source: Arc::from("source"),
            target: Arc::from("target"),
        };
        let Some((arena, root, trace)) = search_trz(
            &diagram,
            &query,
            super::super::SidLimits::default(),
            &antecedent_core::ExecutionContext::for_tests(0),
        )
        .unwrap() else {
            panic!("two-stage TRz exchange should identify this query");
        };
        assert_eq!(
            trace.iter().filter(|rule| rule.starts_with("ztr.line10.source_exchange")).count(),
            2,
            "the second controllable must remain available after the first exchange: {trace:?}"
        );

        let evaluator = arena.compile(root).unwrap();
        let provider = SequentialExchangeScm;
        for treatment in [0.0, 1.0] {
            for outcome in [0.0, 1.0] {
                let assignment = Assignment::from_pairs([
                    (VariableId::from_raw(0), Value::f64(treatment)),
                    (VariableId::from_raw(1), Value::f64(0.0)),
                    (VariableId::from_raw(3), Value::f64(outcome)),
                ]);
                let actual = evaluator
                    .evaluate_with(&arena, &provider, &EvalContext::default(), &assignment)
                    .unwrap();
                // Under do(X=x,T=0), Z has fair A/E noise and Y adds
                // independent fair B, so P(Y=y)=1/2 for both levels.
                assert!((actual - 0.5).abs() < 1e-12, "x={treatment}, y={outcome}: {actual}");
            }
        }
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
