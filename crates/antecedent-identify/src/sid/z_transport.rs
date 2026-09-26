//! Contract validation for the bounded single-source z-transport setting.
//!
//! The contract is kept distinct from classical sID: a source that can intervene
//! only on `controllable` does not satisfy the unrestricted source-experiment
//! premise of the classical theorem.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{IdentificationBudget, IdentificationError};
use antecedent_core::{
    DistributionAvailability, EvidenceCatalog, EvidenceKind,
    InterventionAssignment as CatalogInterventionAssignment, RegimeKind, VariableDomain,
    VariableId,
};
use antecedent_expr::{CausalExprArena, DomainRef, ExprId, ExprNode};
use antecedent_graph::{BitSet, DenseNodeId, NodeRef, SelectionDiagram};
use std::sync::Arc;

fn index_u32(index: usize) -> u32 {
    u32::try_from(index).expect("bounded graph or expression arena index fits u32")
}

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
    /// Concrete source experiment levels a formula may cite: the direct joint
    /// exchange and the registered surrogate cite exactly these, and the
    /// recursive route fixes a coordinate at its declared level only where the
    /// derivation proved the answer constant in it. A queried treatment and a
    /// summed coordinate are never fixed; the request and the summation bind
    /// them at evaluation. This is distinct from controllability: its presence
    /// does not claim results exist; those must be found in the evidence catalog.
    pub experiment_assignment: Arc<[CatalogInterventionAssignment]>,
    /// One source population supplying the declared experimental family.
    pub source: Arc<str>,
    /// Target population supplying its observational law.
    pub target: Arc<str>,
}

/// Explicit observed-variable bound for the single-source z-transport contract.
pub const Z_TRANSPORT_MAX_OBSERVED: usize = 12;
/// Explicit bound for controllable variables.
pub const Z_TRANSPORT_MAX_CONTROLLABLE: usize = 4;
/// Largest source-experiment family enumerated for a negative certificate.
pub const Z_TRANSPORT_MAX_FAMILY_REGIMES: usize = 256;

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
    /// The cartesian experiment family is larger than the negative-certificate budget.
    FamilyExceedsBudget {
        /// Regimes the family would enumerate.
        regimes: usize,
    },
}

/// Why a positive z-transport formula is not yet bound to the catalog.
#[derive(Clone, Debug, PartialEq)]
pub enum ZTransportMissingEvidence {
    /// Line 10 would exchange this controllable, but the query names no level.
    UnassignedControllable {
        /// Controllable coordinate activated by the reduction.
        variable: VariableId,
    },
    /// A derived formula cites a joint law that the catalog does not supply.
    CitedFactor {
        /// Population, intervention, and measured margin of the missing factor.
        detail: String,
    },
}

/// Result of the bounded `TRz` decision.
#[derive(Clone, Debug)]
pub enum ZTransportDecision {
    /// A positive `TRz` formula was derived and its cited factors bind.
    Identified(Box<ZTransportDerivation>),
    /// A checked `TRz` line-11 obstruction for the declared controllable set.
    /// This is a structural claim about that set, not about which tables are present.
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
        /// Structured record of what the bounded search explored and why it
        /// stopped short of a formula or a checked obstruction. The search
        /// produced no expression, so there is no proof graph to expose; this
        /// carries the explored region instead of a fabricated one.
        inspection: ZTransportNotCertifiedInspection,
    },
}

/// Why the bounded `TRz` search stopped without an identifying formula or a
/// checked line-11 obstruction. Neither variant is an impossibility claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ZTransportNotCertifiedKind {
    /// Search reached a `TRz` line-10 exchange but the source c-factor could not
    /// be identified from the cited source law, so no formula was produced.
    SourceCFactorUnidentified,
    /// The bounded recursive search exhausted its licensed rules without reaching
    /// an identifying formula or a checked line-11 terminal.
    NoRecursiveFormula,
    /// Search reached a checked `TRz` line-11 terminal. Deciding the same query
    /// against a catalog certifies this terminal as a structural obstruction;
    /// without a catalog the bounded identifier reports it as not certified.
    CheckedLine11Terminal,
    /// A `TRz` line-10 exchange would activate a controllable the query named no
    /// experiment level for. Supplying that level, or the cited joint law,
    /// resolves the exchange.
    ExperimentAssignmentRequired,
}

/// What the bounded search actually visited on a not-certified outcome.
///
/// The search built no expression, so no proof-graph prefix exists; the honest
/// record is the region it explored: the recursive rules it applied (in order),
/// the recursive subproblems it charged, and the deepest level it reached.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ZTransportNotCertifiedInspection {
    /// Typed classification of the dead end the search reached.
    pub kind: ZTransportNotCertifiedKind,
    /// Recursive-rule applications explored before the search stopped, in the
    /// order they fired. Empty when the search stopped before any rule applied.
    pub explored_rules: Vec<String>,
    /// Recursive subproblems the search charged.
    pub steps_explored: usize,
    /// Deepest `TRz` recursion level the search reached.
    pub depth_reached: usize,
}

/// Which budget stopped a bounded z-transport search. Exhaustion is a resource
/// outcome, never an impossibility or not-certified claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ZTransportBudgetKind {
    /// The recursive-step limit was exceeded.
    Steps,
    /// The recursion-depth limit was reached.
    Depth,
    /// The execution memory budget was exceeded.
    Memory,
    /// Cooperative cancellation was observed.
    Cancelled,
}

/// What a bounded z-transport search consumed against the limits it ran under,
/// captured when a budget or cancellation stopped it. Every count is the
/// engine's own accounting at the point it stopped, never a scientific claim.
///
/// `steps_consumed` and `depth_reached` are `None` when the budget tripped
/// before the recursive search was entered (for example a zero limit, or a
/// memory or cancellation check at engine construction), so no search
/// accounting was taken; they are never fabricated.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ZTransportLimitsReceipt {
    /// Which budget stopped the search.
    pub budget: ZTransportBudgetKind,
    /// Recursive-step limit in force.
    pub steps_limit: usize,
    /// Recursion-depth limit in force.
    pub depth_limit: usize,
    /// Recursive subproblems charged when the search stopped, if measured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steps_consumed: Option<usize>,
    /// Deepest recursion level reached when the search stopped, if measured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub depth_reached: Option<usize>,
    /// Recursive-rule applications explored before the search stopped, in order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub explored_rules: Vec<String>,
}

/// A bounded single-source decision, or a limits receipt when a budget or
/// cancellation stopped the search before it could decide.
#[derive(Clone, Debug)]
pub enum ZTransportOutcome {
    /// The search reached a decision.
    Decided(ZTransportDecision),
    /// A budget or cancellation stopped the search; the receipt records the
    /// limits in force and what was consumed, rather than an opaque error.
    Exhausted(ZTransportLimitsReceipt),
}

/// One source in a two-source z-transport query.
///
/// Each source keeps its own selection diagram, controllable set, and catalog.
/// Factors are never combined across sources.
#[derive(Clone, Debug, PartialEq)]
pub struct ZTransportSourceSpec {
    /// Source population identity.
    pub population: Arc<str>,
    /// Variables this source can manipulate.
    pub controllable: Arc<[VariableId]>,
    /// Concrete experiment assignment for a positive formula from this source.
    pub experiment_assignment: Arc<[CatalogInterventionAssignment]>,
    /// Selection targets on this source's diagram.
    pub selection_targets: Arc<[VariableId]>,
}

/// Shared target query searched once per source, with no cross-source combination.
#[derive(Clone, Debug, PartialEq)]
pub struct TwoSourceZTransportQuery {
    /// Joint outcomes in the target population.
    pub outcomes: Arc<[VariableId]>,
    /// Treatment coordinates of the target query.
    pub treatments: Arc<[VariableId]>,
    /// Target population.
    pub target: Arc<str>,
    /// Exactly two sources. A longer list is outside this query.
    pub sources: [ZTransportSourceSpec; 2],
}

/// One checked proof for a target outcome factor supplied by one source.
#[derive(Clone, Debug)]
pub struct TwoSourceZTransportComponent {
    /// Source population that supplies this component's joint outcome law.
    pub source: Arc<str>,
    /// Outcomes in this factor.
    pub outcomes: Arc<[VariableId]>,
    /// Treatments whose intervention this factor's marginal effect retains.
    pub treatments: Arc<[VariableId]>,
    /// Checked derivation, bound only to this source's catalog.
    pub derivation: Box<ZTransportDerivation>,
}

/// Why a combined two-source result is a sound product of one factor per source.
///
/// Both variants keep the joint-regime rule: each component's cited factors bind
/// to exactly one source's regimes, and no factor is a fabricated joint over both
/// sources' interventions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComponentFactorization {
    /// The queried outcomes and treatments split into two disconnected static
    /// graph components; the joint target law is the product of the components.
    DisconnectedGraphComponents,
    /// A single connected graph whose intervened outcomes m-separate into two
    /// groups given the treatments, so `P*_x(y)` factorizes into one marginal
    /// interventional effect per group and each group is transported from one
    /// source. A single connected c-factor spanning both sources is never
    /// fabricated.
    InterventionSeparatedGroups,
}

/// Result of searching two sources separately, then the bounded disconnected case.
#[allow(clippy::large_enum_variant)] // Public result keeps both source certificates directly inspectable.
#[derive(Clone, Debug)]
pub enum TwoSourceZTransportDecision {
    /// One source's single-source derivation identifies and its cited factors bind.
    Identified {
        /// Population that supplied the formula.
        source: Arc<str>,
        /// Checked derivation for that source alone.
        derivation: Box<ZTransportDerivation>,
    },
    /// Complementary sources each identify one factor of the target law, which
    /// is their product. The factors are either two disconnected graph
    /// components or two intervention-separated outcome groups of one connected
    /// graph; `factorization` says which, and each factor still binds to exactly
    /// one source's regimes.
    CombinedIdentified {
        /// Component proofs in factor order.
        components: [TwoSourceZTransportComponent; 2],
        /// Why the product of the two component laws equals the target law.
        factorization: ComponentFactorization,
    },
    /// Both sources reach a checked line-11 terminal.
    ProvenNonTransportable {
        /// Line-11 obstructions in source order.
        obstructions: [ZTransportObstruction; 2],
    },
    /// Neither source identifies on its own, and the sources are not both line 11.
    /// Cross-source factor combination is not searched.
    NotCertified {
        /// Stable reason. Combination is refused by `z_transport.multi_source_combination_not_searched`.
        reason: &'static str,
    },
}

/// A checked `TRz` line-11 failure for the declared controllable set.
#[derive(Clone, Debug)]
pub struct ZTransportObstruction {
    query: ZTransportQuery,
    graph_signature: String,
    selected_assignment: Vec<(u32, f64)>,
    terminal: TrzTerminalFailure,
}

/// Portable line-11 premises. The query is supplied separately and must match.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ZTransportObstructionRecord {
    /// Stable graph identity, including selection targets.
    pub graph_signature: String,
    /// Declared experiment assignment used while replaying `TRz`.
    pub selected_assignment: Vec<(u32, f64)>,
    /// Reduced terminal subproblem at `TRz` line 11.
    pub terminal: ZTransportTerminalRecord,
}

/// Portable reduced `TRz` state for the terminal failure rule.
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
    /// Previously activated intervention coordinates and their concrete levels.
    /// `None` marks a coordinate that stays symbolic because an enclosing
    /// summation binds it at evaluation.
    pub active_interventions: Vec<(u32, Option<f64>)>,
    /// Whether selection nodes are separated from outcomes given treatments.
    pub selection_separated: bool,
    /// Recursive `TRz` rules used to reach this terminal state.
    pub rules: Vec<String>,
}

/// A checked z-transport derivation: the registered four-variable surrogate
/// factorization, a direct joint source exchange, or a recursive `TRz` formula.
///
/// Every constructor checks the derivation against its diagram and query, so a
/// value of this type is verified; only an untrusted portable record is
/// re-verified, by [`Self::from_record_checked`].
#[derive(Clone, Debug)]
pub struct ZTransportDerivation {
    query: ZTransportQuery,
    graph_signature: String,
    /// The exchanged source coordinate of a surrogate or direct-joint formula.
    surrogate: Option<VariableId>,
    /// The covariate the registered surrogate formula marginalizes.
    confounder: Option<VariableId>,
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
    /// Surrogate intervention coordinate of a surrogate or direct-joint
    /// formula; absent on a recursive derivation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub surrogate: Option<u32>,
    /// Shared covariate the registered surrogate formula eliminates; absent
    /// on the other kinds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confounder: Option<u32>,
    /// Root expression identifier in the accompanying arena.
    pub root: u32,
    /// Checked rule names, in derivation order.
    pub rules: Vec<String>,
}

/// One reachable operation in a compact, inspectable formula DAG.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ZProofOperation {
    /// Expression node ID.
    pub id: u32,
    /// Operation name.
    pub operation: String,
    /// Child expression node IDs.
    pub children: Vec<u32>,
}

/// A required joint-law factor and the actual catalog entries that can supply it.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
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
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ZTransportProofInspection {
    /// Root expression node.
    pub root: u32,
    /// Checked high-level `TRz` rule scope.
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
    cited: Arc<[antecedent_core::RegimeId]>,
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

    /// Every catalog regime a bound factor cites, sorted and without repeats.
    ///
    /// The registered surrogate and direct-exchange formulas cite exactly one
    /// regime; a recursive formula cites one regime per source factor, and a
    /// factor whose exchanged coordinate is bound at evaluation cites one regime
    /// per world of that coordinate.
    #[must_use]
    pub fn cited_regimes(&self) -> &[antecedent_core::RegimeId] {
        &self.cited
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
            surrogate: self.surrogate.map(VariableId::raw),
            confounder: self.confounder.map(VariableId::raw),
            root: self.root.raw(),
            rules: self.rules.clone(),
        }
    }

    /// Reconstruct authority only after the graph, query, and every expression
    /// node have been checked against the registered derivation, under the
    /// caller's search limits and execution context.
    ///
    /// # Errors
    /// Returns an error if any recorded premise or expression differs, or the
    /// replay exhausts its budget or is cancelled.
    pub fn from_record_checked(
        diagram: &SelectionDiagram,
        query: &ZTransportQuery,
        record: &ZTransportDerivationRecord,
        arena: CausalExprArena,
        limits: super::SidLimits,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<Self, IdentificationError> {
        let candidate = Self {
            query: query.clone(),
            graph_signature: record.graph_signature.clone(),
            surrogate: record.surrogate.map(VariableId::from_raw),
            confounder: record.confounder.map(VariableId::from_raw),
            arena,
            root: ExprId::from_raw(record.root),
            kind: match record.rules.first().map(String::as_str) {
                Some("ztr.surrogate_factorization") => ZFormulaKind::Surrogate,
                Some("ztr.source_exchange_joint") => ZFormulaKind::DirectJoint,
                Some("ztr.recursive_reduction") => ZFormulaKind::Recursive,
                _ => {
                    return Err(IdentificationError::invalid_derivation(
                        "z_transport.proof_rule_mismatch",
                    ));
                }
            },
            rules: record.rules.clone(),
        };
        verify_z_transport_derivation(diagram, query, &candidate, limits, ctx)?;
        Ok(candidate)
    }

    /// Check that this derivation was derived for exactly `diagram` and
    /// `query`. A derivation is verified when constructed, so binding needs no
    /// second search; this is the input identity it still has to prove.
    ///
    /// # Errors
    /// A different query or graph.
    pub fn check_inputs(
        &self,
        diagram: &SelectionDiagram,
        query: &ZTransportQuery,
    ) -> Result<(), IdentificationError> {
        validate_z_transport_query(diagram, query)?;
        if self.query != *query || self.graph_signature != super::graph_signature(diagram) {
            return Err(IdentificationError::invalid_derivation(
                "z_transport.proof_input_mismatch",
            ));
        }
        Ok(())
    }
    /// Original typed z-transport query.
    #[must_use]
    pub const fn query(&self) -> &ZTransportQuery {
        &self.query
    }

    /// Surrogate variable manipulated in the source experiment of a surrogate
    /// or direct-joint formula. A recursive derivation exchanges its
    /// coordinates inside the formula and names none here.
    #[must_use]
    pub const fn surrogate(&self) -> Option<VariableId> {
        self.surrogate
    }

    /// Exact source intervention assignment cited by the checked formula.
    #[must_use]
    pub fn experiment_assignment(&self) -> &[CatalogInterventionAssignment] {
        &self.query.experiment_assignment
    }

    /// Shared confounder the registered surrogate formula marginalizes; the
    /// other kinds have none.
    #[must_use]
    pub const fn confounder(&self) -> Option<VariableId> {
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
        /// Structured record of the region the bounded search explored and the
        /// typed reason it stopped short of a formula. When the search reached a
        /// checked line-11 terminal, `decide` against a catalog certifies that
        /// terminal as a structural obstruction; here it is reported as not
        /// certified because no catalog was consulted.
        inspection: ZTransportNotCertifiedInspection,
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
        return Err(IdentificationError::UnsupportedInput {
            code: "z_transport.unsupported_observed_count",
        });
    }
    if query.controllable.is_empty() || query.controllable.len() > Z_TRANSPORT_MAX_CONTROLLABLE {
        return Err(IdentificationError::UnsupportedInput {
            code: "z_transport.unsupported_controllable_count",
        });
    }
    if query.outcomes.is_empty()
        || query.treatments.is_empty()
        || query.source.trim().is_empty()
        || query.target.trim().is_empty()
        || query.source == query.target
    {
        return Err(IdentificationError::invalid_input("z_transport.invalid_query"));
    }

    for group in [&query.outcomes, &query.treatments, &query.controllable] {
        let mut seen = std::collections::BTreeSet::new();
        for variable in group.iter().copied() {
            if !diagram.causal_graph().nodes().contains(&NodeRef::Static(variable)) {
                return Err(IdentificationError::invalid_input("z_transport.unknown_variable"));
            }
            if !seen.insert(variable.raw()) {
                return Err(IdentificationError::invalid_input("z_transport.duplicate_variable"));
            }
        }
    }
    if query.outcomes.iter().any(|v| query.treatments.contains(v)) {
        return Err(IdentificationError::invalid_input("z_transport.outcomes_overlap_treatments"));
    }
    let mut assigned = std::collections::BTreeSet::new();
    for assignment in query.experiment_assignment.iter() {
        if !query.controllable.contains(&assignment.variable)
            || !assigned.insert(assignment.variable.raw())
            || assignment.value.validate_concrete_intervention_level().is_err()
        {
            return Err(IdentificationError::invalid_input(
                "z_transport.invalid_experiment_assignment",
            ));
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
#[allow(clippy::too_many_lines)]
#[allow(clippy::float_cmp)] // Categorical intervention labels are finite exact integers encoded as f64.
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
    let mut family_size = 0usize;
    let k = domains.len();
    for mask in 1..(1usize << k) {
        let mut block = 1usize;
        for (bit, (_, levels)) in domains.iter().enumerate() {
            if mask & (1usize << bit) != 0 {
                block = block.saturating_mul(levels.len());
            }
        }
        family_size = family_size.saturating_add(block);
        if family_size > Z_TRANSPORT_MAX_FAMILY_REGIMES {
            return Err(ZExperimentFamilyError::FamilyExceedsBudget { regimes: family_size });
        }
    }

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

/// Decide a bounded single-source z-transport query against the supplied catalog.
///
/// A positive result is a checked `TRz` formula whose cited joints are present.
/// Unused experiments and uncited variables are not required. A negative result
/// is certified when search reaches `TRz` line 11 and an independent checker
/// confirms the reduced graph, C0, Z∩X, and the failed line-10 separation
/// premise. That obstruction is a fact about the graph and the declared
/// controllable set; an incomplete catalog is not the obstruction. A missing
/// cited factor, or a line-10 exchange whose controllable has no declared
/// level, remains missing evidence. Exhausted search remains distinct from
/// theorem obstruction.
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
    decide_z_transport_reporting(diagram, query, catalog, limits, ctx, &mut None)
}

/// Decide a bounded single-source z-transport query, and on a budget or
/// cancellation return the limits receipt rather than an opaque exhaustion
/// error, so callers can inspect the limits in force and what was consumed.
///
/// Every non-exhaustion outcome is a [`ZTransportOutcome::Decided`]; only a
/// budget or cooperative cancellation becomes [`ZTransportOutcome::Exhausted`].
/// Malformed inputs and invalid catalogs are still returned as errors.
///
/// # Errors
/// Invalid or unsupported query, or an invalid catalog.
pub fn decide_z_transport_inspecting(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    catalog: &EvidenceCatalog,
    limits: super::SidLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<ZTransportOutcome, IdentificationError> {
    let mut receipt = None;
    match decide_z_transport_reporting(diagram, query, catalog, limits, ctx, &mut receipt) {
        Ok(decision) => Ok(ZTransportOutcome::Decided(decision)),
        Err(error) if error.is_budget_or_cancel() => {
            let receipt = receipt.unwrap_or_else(|| ZTransportLimitsReceipt {
                budget: pre_search_budget_kind(&error),
                steps_limit: limits.steps,
                depth_limit: limits.depth,
                steps_consumed: None,
                depth_reached: None,
                explored_rules: Vec::new(),
            });
            Ok(ZTransportOutcome::Exhausted(receipt))
        }
        Err(error) => Err(error),
    }
}

fn decide_z_transport_reporting(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    catalog: &EvidenceCatalog,
    limits: super::SidLimits,
    ctx: &antecedent_core::ExecutionContext,
    receipt: &mut Option<ZTransportLimitsReceipt>,
) -> Result<ZTransportDecision, IdentificationError> {
    catalog.validate().map_err(|error| {
        IdentificationError::invalid_catalog(format!("z_transport.invalid_catalog: {error}"))
    })?;
    match derive_z_transport(diagram, query, limits, ctx, receipt)? {
        ZDerivation::Formula(derivation) => {
            match bind_z_transport_catalog(diagram, derivation.query(), &derivation, catalog) {
                Ok(_) => Ok(ZTransportDecision::Identified(Box::new(derivation))),
                Err(error @ IdentificationError::MissingEvidence { .. }) => {
                    Ok(ZTransportDecision::MissingEvidence {
                        missing: ZTransportMissingEvidence::CitedFactor {
                            detail: error.to_string(),
                        },
                    })
                }
                Err(error) => Err(error),
            }
        }
        ZDerivation::Unassigned { variable, .. } => Ok(ZTransportDecision::MissingEvidence {
            missing: ZTransportMissingEvidence::UnassignedControllable { variable },
        }),
        ZDerivation::Line11 { terminal, .. } => {
            certify_line11_obstruction(diagram, query, terminal, limits, ctx)
        }
        ZDerivation::NotCertified { reason, inspection } => {
            Ok(ZTransportDecision::NotCertified { reason, inspection })
        }
    }
}

/// Search two sources separately on a shared causal graph.
///
/// One identifying source is enough, and the other source's catalog is not
/// required: both sources are searched, and a failure of one source does not
/// hide an identification by the other. Two line-11 terminals become one
/// obstruction only for one experiment family. A result that would mix a
/// factor from each source is refused by name. This does not call classical
/// meta-transport, which assumes every source can experiment on every variable.
///
/// # Errors
/// A source population collides with the other source or the target, a
/// selection diagram is invalid, or neither source identifies and one search
/// failed (the first failure is reported).
#[allow(clippy::too_many_lines)] // Mirrors the single-source, disconnected, and connected branches explicitly.
pub fn decide_two_source_z_transport(
    graph: &antecedent_graph::Admg,
    query: &TwoSourceZTransportQuery,
    catalogs: [&EvidenceCatalog; 2],
    limits: super::SidLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<TwoSourceZTransportDecision, IdentificationError> {
    if query.sources[0].population == query.sources[1].population
        || query.sources.iter().any(|source| source.population == query.target)
    {
        return Err(IdentificationError::invalid_input(
            "z_transport.two_source_population_collision",
        ));
    }
    let mut decisions = Vec::with_capacity(2);
    for (source, catalog) in query.sources.iter().zip(catalogs) {
        let diagram =
            SelectionDiagram::try_new(graph.clone(), Arc::clone(&source.selection_targets))
                .map_err(|error| IdentificationError::invalid_input(error.to_string()))?;
        let single = ZTransportQuery {
            outcomes: Arc::clone(&query.outcomes),
            treatments: Arc::clone(&query.treatments),
            controllable: Arc::clone(&source.controllable),
            experiment_assignment: Arc::clone(&source.experiment_assignment),
            source: Arc::clone(&source.population),
            target: Arc::clone(&query.target),
        };
        decisions.push(decide_z_transport_with_catalog(&diagram, &single, catalog, limits, ctx));
    }
    for (source, decision) in query.sources.iter().zip(&decisions) {
        if let Ok(ZTransportDecision::Identified(derivation)) = decision {
            return Ok(TwoSourceZTransportDecision::Identified {
                source: Arc::clone(&source.population),
                derivation: derivation.clone(),
            });
        }
    }
    if let Some(graph_components) = two_disconnected_query_components(graph, query) {
        for source_order in [[0usize, 1usize], [1usize, 0usize]] {
            let mut component_proofs = Vec::with_capacity(2);
            let mut complete = true;
            for (component_index, source_index) in source_order.into_iter().enumerate() {
                let nodes = &graph_components[component_index];
                let source = &query.sources[source_index];
                let outcomes = query
                    .outcomes
                    .iter()
                    .copied()
                    .filter(|variable| nodes.contains(variable))
                    .collect::<Vec<_>>();
                let treatments = query
                    .treatments
                    .iter()
                    .copied()
                    .filter(|variable| nodes.contains(variable))
                    .collect::<Vec<_>>();
                let assignment = source
                    .experiment_assignment
                    .iter()
                    .filter(|assignment| nodes.contains(&assignment.variable))
                    .cloned()
                    .collect::<Vec<_>>();
                if outcomes.is_empty()
                    || treatments.is_empty()
                    || treatments.iter().any(|variable| !source.controllable.contains(variable))
                    || !assignment.iter().all(|a| treatments.contains(&a.variable))
                    || !treatments.iter().all(|variable| {
                        assignment.iter().any(|assignment| assignment.variable == *variable)
                    })
                    || source.selection_targets.iter().any(|variable| !nodes.contains(variable))
                    || source.controllable.iter().any(|variable| !nodes.contains(variable))
                {
                    complete = false;
                    break;
                }
                let component_query = ZTransportQuery {
                    outcomes: outcomes.clone().into(),
                    treatments: treatments.clone().into(),
                    controllable: source.controllable.clone(),
                    experiment_assignment: assignment.into(),
                    source: Arc::clone(&source.population),
                    target: Arc::clone(&query.target),
                };
                let diagram =
                    SelectionDiagram::try_new(graph.clone(), Arc::clone(&source.selection_targets))
                        .map_err(|error| IdentificationError::invalid_input(error.to_string()))?;
                let ZTransportDecision::Identified(derivation) = decide_z_transport_with_catalog(
                    &diagram,
                    &component_query,
                    catalogs[source_index],
                    limits,
                    ctx,
                )?
                else {
                    complete = false;
                    break;
                };
                component_proofs.push(TwoSourceZTransportComponent {
                    source: Arc::clone(&source.population),
                    outcomes: outcomes.into(),
                    treatments: treatments.into(),
                    derivation,
                });
            }
            if complete {
                let [first, second] = component_proofs.try_into().expect("two source components");
                return Ok(TwoSourceZTransportDecision::CombinedIdentified {
                    components: [first, second],
                    factorization: ComponentFactorization::DisconnectedGraphComponents,
                });
            }
        }
    }
    // Connected complementary case: one connected graph whose intervened
    // outcomes m-separate into two groups given the treatments. Each group's
    // marginal interventional effect is transported from one source, and the
    // target law is their product. A single connected c-factor that would need
    // a fabricated joint over both sources' interventions is never combined.
    if let Some(decision) = connected_complementary_decision(graph, query, catalogs, limits, ctx)? {
        return Ok(decision);
    }
    let decisions = decisions.into_iter().collect::<Result<Vec<_>, _>>()?;
    // A line-11 terminal is an obstruction relative to one source's experiment
    // family. Two such terminals rule out the union of the families only when
    // the union is one of them: the sources declare the same controllable set
    // on the same selection diagram. Otherwise a factor from each source might
    // still combine, which this decision does not search.
    if let [
        ZTransportDecision::ProvenNonTransportable(left),
        ZTransportDecision::ProvenNonTransportable(right),
    ] = decisions.as_slice()
    {
        let [first, second] = &query.sources;
        let same_family = same_variable_set(&first.controllable, &second.controllable)
            && same_variable_set(&first.selection_targets, &second.selection_targets);
        if same_family {
            return Ok(TwoSourceZTransportDecision::ProvenNonTransportable {
                obstructions: [*left.clone(), *right.clone()],
            });
        }
    }
    Ok(TwoSourceZTransportDecision::NotCertified {
        reason: "z_transport.multi_source_combination_not_searched",
    })
}

/// Return the two graph components only for a query whose outcomes and
/// treatments partition into exactly two disconnected static components.
fn two_disconnected_query_components(
    graph: &antecedent_graph::Admg,
    query: &TwoSourceZTransportQuery,
) -> Option<[Vec<VariableId>; 2]> {
    if graph.nodes().iter().any(|node| !matches!(node, NodeRef::Static(_))) {
        return None;
    }
    let mut labels = vec![usize::MAX; graph.node_count()];
    let mut components = Vec::<Vec<VariableId>>::new();
    for root in 0..graph.node_count() {
        if labels[root] != usize::MAX {
            continue;
        }
        let label = components.len();
        let mut stack = vec![DenseNodeId::from_raw(u32::try_from(root).ok()?)];
        labels[root] = label;
        let mut variables = Vec::new();
        while let Some(node) = stack.pop() {
            let NodeRef::Static(variable) = graph.nodes()[node.as_usize()] else {
                return None;
            };
            variables.push(variable);
            for neighbor in graph
                .children(node)
                .iter()
                .chain(graph.parents(node))
                .chain(graph.bidirected_neighbors(node))
            {
                if labels[neighbor.as_usize()] == usize::MAX {
                    labels[neighbor.as_usize()] = label;
                    stack.push(*neighbor);
                }
            }
        }
        components.push(variables);
    }
    if components.len() != 2
        || query.outcomes.iter().any(|v| !components.iter().any(|component| component.contains(v)))
        || query
            .treatments
            .iter()
            .any(|v| !components.iter().any(|component| component.contains(v)))
        || components.iter().any(|component| {
            !query.outcomes.iter().any(|v| component.contains(v))
                || !query.treatments.iter().any(|v| component.contains(v))
        })
    {
        return None;
    }
    let [first, second] = components.try_into().ok()?;
    Some([first, second])
}

/// Dense id of `variable` in `graph`'s node order, or `None` if absent.
fn dense_of(graph: &antecedent_graph::Admg, variable: VariableId) -> Option<DenseNodeId> {
    graph
        .nodes()
        .iter()
        .position(|node| matches!(node, NodeRef::Static(candidate) if *candidate == variable))
        .map(|index| DenseNodeId::from_raw(index_u32(index)))
}

/// Whether the shared static graph is one connected component under directed,
/// bidirected, and parent adjacency. The disconnected case is decided
/// separately, so the connected route only fires on a single component.
fn graph_is_connected(graph: &antecedent_graph::Admg) -> bool {
    if graph.node_count() == 0
        || graph.nodes().iter().any(|node| !matches!(node, NodeRef::Static(_)))
    {
        return false;
    }
    let mut seen = vec![false; graph.node_count()];
    let mut stack = vec![DenseNodeId::from_raw(0)];
    seen[0] = true;
    let mut count = 1usize;
    while let Some(node) = stack.pop() {
        for neighbor in graph
            .children(node)
            .iter()
            .chain(graph.parents(node))
            .chain(graph.bidirected_neighbors(node))
        {
            if !seen[neighbor.as_usize()] {
                seen[neighbor.as_usize()] = true;
                count += 1;
                stack.push(*neighbor);
            }
        }
    }
    count == graph.node_count()
}

/// The `do(treatments)`-mutilated ADMG on the same dense node layout: every
/// directed edge into a treatment and every bidirected edge incident to a
/// treatment is dropped, so the treatments are exogenous roots. m-separation on
/// this graph is m-separation in the intervened law `P*_x(v)`.
fn mutilated_admg(
    graph: &antecedent_graph::Admg,
    treatments: &BitSet,
) -> Result<antecedent_graph::Admg, IdentificationError> {
    let n = graph.node_count();
    let mut mutilated = antecedent_graph::Admg::with_variables(index_u32(n));
    let build = || IdentificationError::invalid_input("z_transport.mutilated_graph");
    for from_index in 0..n {
        let from = DenseNodeId::from_raw(index_u32(from_index));
        for to in graph.children(from) {
            if !treatments.contains(*to) {
                mutilated.insert_directed(from, *to).map_err(|_| build())?;
            }
        }
        for other in graph.bidirected_neighbors(from) {
            if other.as_usize() > from_index
                && !treatments.contains(from)
                && !treatments.contains(*other)
            {
                mutilated.insert_bidirected(from, *other).map_err(|_| build())?;
            }
        }
    }
    Ok(mutilated)
}

/// Directed-ancestor closure of `seeds` in `graph` (seeds included).
fn directed_ancestors(graph: &antecedent_graph::Admg, seeds: &[DenseNodeId]) -> BitSet {
    let mut closure = BitSet::with_len(graph.node_count());
    let mut stack = seeds.to_vec();
    for seed in seeds {
        closure.insert(*seed);
    }
    while let Some(node) = stack.pop() {
        for parent in graph.parents(node) {
            if !closure.contains(*parent) {
                closure.insert(*parent);
                stack.push(*parent);
            }
        }
    }
    closure
}

/// Partition the queried outcomes into exactly two groups that are m-separated
/// from each other given the treatments in the `do(X)`-mutilated graph, so
/// `P*_x(y)` factorizes as the product of the two groups' marginal effects.
/// `None` when the graph is not all-static, the outcomes stay m-connected (one
/// group, whose single c-factor would need a fabricated cross-source joint), or
/// they split into more than two groups (outside the two-source contract).
fn intervention_separated_outcome_groups(
    graph: &antecedent_graph::Admg,
    query: &TwoSourceZTransportQuery,
) -> Result<Option<[Vec<VariableId>; 2]>, IdentificationError> {
    if graph.nodes().iter().any(|node| !matches!(node, NodeRef::Static(_))) {
        return Ok(None);
    }
    let mut treatments = BitSet::with_len(graph.node_count());
    for treatment in query.treatments.iter().copied() {
        let Some(dense) = dense_of(graph, treatment) else { return Ok(None) };
        treatments.insert(dense);
    }
    let mutilated = mutilated_admg(graph, &treatments)?;
    let treatment_ids = treatments.to_dense_ids();
    let outcomes = query.outcomes.to_vec();
    let outcome_dense = outcomes
        .iter()
        .map(|variable| dense_of(graph, *variable).ok_or_else(build_unknown))
        .collect::<Result<Vec<_>, _>>()?;
    // Connect two outcomes that are m-connected given the treatments; the groups
    // are the connected components of that relation. Cross-group pairs are all
    // m-separated, so the groups are jointly independent given the intervention.
    let mut ws = antecedent_graph::DSeparationWorkspace::default();
    let mut labels = vec![usize::MAX; outcomes.len()];
    let mut group_count = 0usize;
    for start in 0..outcomes.len() {
        if labels[start] != usize::MAX {
            continue;
        }
        let label = group_count;
        group_count += 1;
        let mut stack = vec![start];
        labels[start] = label;
        while let Some(current) = stack.pop() {
            for other in 0..outcomes.len() {
                if labels[other] != usize::MAX {
                    continue;
                }
                let connected = !mutilated
                    .is_m_separated(
                        outcome_dense[current],
                        outcome_dense[other],
                        &treatment_ids,
                        &mut ws,
                    )
                    .map_err(|_| build_unknown())?;
                if connected {
                    labels[other] = label;
                    stack.push(other);
                }
            }
        }
    }
    if group_count != 2 {
        return Ok(None);
    }
    let mut groups: [Vec<VariableId>; 2] = [Vec::new(), Vec::new()];
    for (index, variable) in outcomes.into_iter().enumerate() {
        groups[labels[index]].push(variable);
    }
    Ok(Some(groups))
}

fn build_unknown() -> IdentificationError {
    IdentificationError::invalid_input("z_transport.unknown_variable")
}

/// Decide the connected complementary case: one connected graph whose intervened
/// outcomes factorize into two groups, each transported from one distinct
/// source. Each factor binds only to its own source's regimes (the joint-regime
/// rule); a c-factor that would span both sources is never fabricated.
fn connected_complementary_decision(
    graph: &antecedent_graph::Admg,
    query: &TwoSourceZTransportQuery,
    catalogs: [&EvidenceCatalog; 2],
    limits: super::SidLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<Option<TwoSourceZTransportDecision>, IdentificationError> {
    if !graph_is_connected(graph) {
        return Ok(None);
    }
    let Some(groups) = intervention_separated_outcome_groups(graph, query)? else {
        return Ok(None);
    };
    let mut treatments = BitSet::with_len(graph.node_count());
    for treatment in query.treatments.iter().copied() {
        let Some(dense) = dense_of(graph, treatment) else { return Ok(None) };
        treatments.insert(dense);
    }
    let mutilated = mutilated_admg(graph, &treatments)?;
    // Each group's marginal effect P*_x(y_g) equals P*_{x_g}(y_g), where x_g are
    // the treatments that remain ancestors of the group after intervention; the
    // other treatments do not change the group's law.
    let relevant_treatments =
        |group: &[VariableId]| -> Result<Vec<VariableId>, IdentificationError> {
            let seeds = group
                .iter()
                .map(|variable| dense_of(graph, *variable).ok_or_else(build_unknown))
                .collect::<Result<Vec<_>, _>>()?;
            let ancestors = directed_ancestors(&mutilated, &seeds);
            let mut retained = query
                .treatments
                .iter()
                .copied()
                .filter(|treatment| {
                    dense_of(graph, *treatment).is_some_and(|dense| ancestors.contains(dense))
                })
                .collect::<Vec<_>>();
            retained.sort_unstable_by_key(|variable| variable.raw());
            Ok(retained)
        };
    for order in [[0usize, 1usize], [1usize, 0usize]] {
        let mut proofs = Vec::with_capacity(2);
        let mut complete = true;
        for (group_index, source_index) in order.into_iter().enumerate() {
            let outcomes = &groups[group_index];
            let source = &query.sources[source_index];
            let group_treatments = relevant_treatments(outcomes)?;
            let assignment = source
                .experiment_assignment
                .iter()
                .filter(|assignment| group_treatments.contains(&assignment.variable))
                .cloned()
                .collect::<Vec<_>>();
            if group_treatments.is_empty()
                || group_treatments.iter().any(|variable| !source.controllable.contains(variable))
                || !group_treatments.iter().all(|variable| {
                    assignment.iter().any(|assignment| assignment.variable == *variable)
                })
            {
                complete = false;
                break;
            }
            let component_query = ZTransportQuery {
                outcomes: outcomes.clone().into(),
                treatments: group_treatments.clone().into(),
                controllable: Arc::clone(&source.controllable),
                experiment_assignment: assignment.into(),
                source: Arc::clone(&source.population),
                target: Arc::clone(&query.target),
            };
            let diagram =
                SelectionDiagram::try_new(graph.clone(), Arc::clone(&source.selection_targets))
                    .map_err(|error| IdentificationError::invalid_input(error.to_string()))?;
            let ZTransportDecision::Identified(derivation) = decide_z_transport_with_catalog(
                &diagram,
                &component_query,
                catalogs[source_index],
                limits,
                ctx,
            )?
            else {
                complete = false;
                break;
            };
            proofs.push(TwoSourceZTransportComponent {
                source: Arc::clone(&source.population),
                outcomes: outcomes.clone().into(),
                treatments: group_treatments.into(),
                derivation,
            });
        }
        if complete {
            let [first, second] = proofs.try_into().expect("two connected components");
            return Ok(Some(TwoSourceZTransportDecision::CombinedIdentified {
                components: [first, second],
                factorization: ComponentFactorization::InterventionSeparatedGroups,
            }));
        }
    }
    Ok(None)
}

fn certify_line11_obstruction(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    terminal: TrzTerminalFailure,
    limits: super::SidLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<ZTransportDecision, IdentificationError> {
    let obstruction = ZTransportObstruction {
        query: query.clone(),
        graph_signature: super::graph_signature(diagram),
        selected_assignment: query
            .experiment_assignment
            .iter()
            .filter_map(|assignment| {
                assignment.value.as_f64().map(|value| (assignment.variable.raw(), value))
            })
            .collect(),
        terminal,
    };
    verify_z_transport_obstruction(diagram, query, &obstruction, limits, ctx)?;
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

    /// Reconstruct an obstruction only after line-11 replay.
    ///
    /// The obstruction is structural: the search is replayed on the diagram
    /// and query alone, and no catalog is consulted.
    ///
    /// # Errors
    /// Changed graph, query, controllable set, or terminal search state.
    #[allow(clippy::needless_pass_by_value)] // Public checked-import API consumes the untrusted record.
    pub fn from_record_checked(
        record: ZTransportObstructionRecord,
        diagram: &SelectionDiagram,
        query: &ZTransportQuery,
        limits: super::SidLimits,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<Self, IdentificationError> {
        let ZDerivation::Line11 { terminal, .. } =
            derive_z_transport(diagram, query, limits, ctx, &mut None)?
        else {
            return Err(IdentificationError::invalid_derivation(
                "z_transport.obstruction_not_reproduced",
            ));
        };
        let ZTransportDecision::ProvenNonTransportable(checked) =
            certify_line11_obstruction(diagram, query, terminal, limits, ctx)?
        else {
            return Err(IdentificationError::invalid_derivation(
                "z_transport.obstruction_not_reproduced",
            ));
        };
        let checked = *checked;
        if checked.to_record() != record {
            return Err(IdentificationError::invalid_derivation(
                "z_transport.obstruction_record_mismatch",
            ));
        }
        Ok(checked)
    }
}

/// Independently replay a structural negative z-transport certificate.
///
/// # Errors
/// The query differs, or `TRz` does not reach the recorded line-11 state.
pub fn verify_z_transport_obstruction(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    obstruction: &ZTransportObstruction,
    limits: super::SidLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<(), IdentificationError> {
    validate_z_transport_query(diagram, query)?;
    if obstruction.query != *query || obstruction.graph_signature != super::graph_signature(diagram)
    {
        return Err(IdentificationError::invalid_derivation(
            "z_transport.obstruction_input_mismatch",
        ));
    }
    verify_trz_line11_terminal(diagram, query, &obstruction.terminal, ctx)?;
    let assignment = query
        .experiment_assignment
        .iter()
        .filter_map(|intervention| {
            intervention.value.as_f64().map(|value| (intervention.variable.raw(), value))
        })
        .collect::<Vec<_>>();
    if assignment != obstruction.selected_assignment {
        return Err(IdentificationError::invalid_derivation(
            "z_transport.obstruction_assignment_mismatch",
        ));
    }
    let replay = z_search(diagram, query, limits, ctx, &mut None)?;
    if replay.identified.is_some()
        || replay.terminal_failure.as_ref() != Some(&obstruction.terminal)
    {
        return Err(IdentificationError::invalid_derivation(
            "z_transport.obstruction_replay_mismatch",
        ));
    }
    Ok(())
}

/// Check the local graph premises of Bareinboim–Pearl `TRz` Figure 4 line 11
/// independently of the recursive search's terminal flag and separation
/// helper. Their Theorem 5 maps this failed line-11 state `(D, C0)` to a
/// zs-hedge; `verify_z_transport_obstruction` separately replays the prefix.
fn verify_trz_line11_terminal(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    terminal: &TrzTerminalFailure,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<(), IdentificationError> {
    let bad = || IdentificationError::invalid_derivation("z_transport.invalid_line11_terminal");
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

/// Identify a bounded single-source restricted-experiment query under the
/// caller's search limits and execution context.
///
/// The registered surrogate formula and direct joint exchange have dedicated
/// local checkers. Other in-bound graphs, including admissible selection
/// diagrams, follow the recursive `TRz` reduction and are rederived on replay.
/// Use [`decide_z_transport_with_catalog`] when a negative result must be
/// distinguished from incomplete catalog evidence: a line-11 failure without a
/// catalog is [`ZTransportResult::NotCertified`], not an obstruction.
///
/// The registered factorization is the four-variable graph `W→Z→X→Y`, `W→Y`,
/// bidirected `W↔Y`, `Z↔Y`, `Z↔X`, controllable `Z`, query `P(Y | do(X))`, and
/// no selection targets. Other graphs up to 12 observed and 4 controllable
/// variables publish the recursive derivation when search succeeds.
///
/// # Errors
/// Invalid query coordinates, cancellation, or exhausted computation.
pub fn identify_z_transport(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    limits: super::SidLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<ZTransportResult, IdentificationError> {
    identify_z_transport_reporting(diagram, query, limits, ctx, &mut None)
}

/// Identify a bounded z-transport formula, and on a budget or cancellation fill
/// `receipt` with the limits in force and what the search consumed before
/// returning the exhaustion error, so callers that snapshot a failure can
/// expose a limits receipt instead of an opaque status.
///
/// # Errors
/// Invalid query coordinates, cancellation, or exhausted computation.
pub fn identify_z_transport_reporting(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    limits: super::SidLimits,
    ctx: &antecedent_core::ExecutionContext,
    receipt: &mut Option<ZTransportLimitsReceipt>,
) -> Result<ZTransportResult, IdentificationError> {
    Ok(match derive_z_transport(diagram, query, limits, ctx, receipt)? {
        ZDerivation::Formula(derivation) => ZTransportResult::Identified(Box::new(derivation)),
        ZDerivation::Unassigned { inspection, .. } => ZTransportResult::NotCertified {
            reason: "z_transport.experiment_assignment_required",
            inspection,
        },
        ZDerivation::Line11 { inspection, .. } | ZDerivation::NotCertified { inspection, .. } => {
            ZTransportResult::NotCertified {
                reason: "z_transport.no_checked_recursive_formula",
                inspection,
            }
        }
    })
}

#[allow(clippy::large_enum_variant)] // Private short-lived state; boxing would add allocation to recursive search.
enum ZDerivation {
    Formula(ZTransportDerivation),
    Line11 { terminal: TrzTerminalFailure, inspection: ZTransportNotCertifiedInspection },
    Unassigned { variable: VariableId, inspection: ZTransportNotCertifiedInspection },
    NotCertified { reason: &'static str, inspection: ZTransportNotCertifiedInspection },
}

fn derive_z_transport(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    limits: super::SidLimits,
    ctx: &antecedent_core::ExecutionContext,
    receipt: &mut Option<ZTransportLimitsReceipt>,
) -> Result<ZDerivation, IdentificationError> {
    validate_z_transport_query(diagram, query)?;
    if limits.steps == 0 || limits.depth == 0 {
        *receipt = Some(ZTransportLimitsReceipt {
            budget: if limits.depth == 0 {
                ZTransportBudgetKind::Depth
            } else {
                ZTransportBudgetKind::Steps
            },
            steps_limit: limits.steps,
            depth_limit: limits.depth,
            steps_consumed: Some(0),
            depth_reached: Some(0),
            explored_rules: Vec::new(),
        });
        return Err(IdentificationError::budget(IdentificationBudget::ZTransport));
    }
    if ctx.cancellation.is_cancelled() {
        *receipt = Some(ZTransportLimitsReceipt {
            budget: ZTransportBudgetKind::Cancelled,
            steps_limit: limits.steps,
            depth_limit: limits.depth,
            steps_consumed: Some(0),
            depth_reached: Some(0),
            explored_rules: Vec::new(),
        });
        return Err(IdentificationError::Cancelled);
    }
    if direct_joint_admissible(diagram, query)? {
        let (arena, root) = direct_joint_formula(query);
        let derivation = ZTransportDerivation {
            query: query.clone(),
            graph_signature: super::graph_signature(diagram),
            surrogate: Some(query.experiment_assignment[0].variable),
            confounder: None,
            arena,
            root,
            kind: ZFormulaKind::DirectJoint,
            rules: vec!["ztr.source_exchange_joint".into()],
        };
        verify_z_transport_derivation(diagram, query, &derivation, limits, ctx)?;
        return Ok(ZDerivation::Formula(derivation));
    }
    if let Some(derivation) = registered_surrogate_derivation(diagram, query, limits, ctx)? {
        return Ok(ZDerivation::Formula(derivation));
    }
    let searched = z_search(diagram, query, limits, ctx, receipt)?;
    let explored_rules = searched.explored_rules.clone();
    let steps_explored = searched.steps_explored;
    let depth_reached = searched.depth_reached;
    let inspection = |kind: ZTransportNotCertifiedKind| ZTransportNotCertifiedInspection {
        kind,
        explored_rules: explored_rules.clone(),
        steps_explored,
        depth_reached,
    };
    if let Some(variable) = searched.unassigned {
        return Ok(ZDerivation::Unassigned {
            variable,
            inspection: inspection(ZTransportNotCertifiedKind::ExperimentAssignmentRequired),
        });
    }
    if let Some((arena, root, trace)) = searched.identified {
        // The recursive formula is the search's own output: re-running the
        // search would recheck nothing. Replay verifies untrusted records.
        return Ok(ZDerivation::Formula(ZTransportDerivation {
            query: query.clone(),
            graph_signature: super::graph_signature(diagram),
            surrogate: None,
            confounder: None,
            arena,
            root,
            kind: ZFormulaKind::Recursive,
            rules: std::iter::once("ztr.recursive_reduction".to_owned()).chain(trace).collect(),
        }));
    }
    if let Some(terminal) = searched.terminal_failure {
        return Ok(ZDerivation::Line11 {
            terminal,
            inspection: inspection(ZTransportNotCertifiedKind::CheckedLine11Terminal),
        });
    }
    let kind = if explored_rules.iter().any(|rule| rule == "ztr.line10.c_factor_not_identified") {
        ZTransportNotCertifiedKind::SourceCFactorUnidentified
    } else {
        ZTransportNotCertifiedKind::NoRecursiveFormula
    };
    Ok(ZDerivation::NotCertified {
        reason: "z_transport.no_checked_recursive_formula",
        inspection: inspection(kind),
    })
}

fn registered_surrogate_derivation(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    limits: super::SidLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<Option<ZTransportDerivation>, IdentificationError> {
    if diagram.causal_graph().node_count() != 4
        || !diagram.selection_targets().is_empty()
        || query.outcomes.len() != 1
        || query.treatments.len() != 1
        || query.controllable.len() != 1
        || query.experiment_assignment.len() != 1
        || query.experiment_assignment[0].variable != query.controllable[0]
    {
        return Ok(None);
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
        return Ok(None);
    };
    if !matches_registered_surrogate_graph(diagram, confounder, surrogate, treatment, outcome) {
        return Ok(None);
    }
    let (arena, root) = surrogate_formula(query, confounder);
    let derivation = ZTransportDerivation {
        query: query.clone(),
        graph_signature: super::graph_signature(diagram),
        surrogate: Some(surrogate),
        confounder: Some(confounder),
        arena,
        root,
        kind: ZFormulaKind::Surrogate,
        rules: vec!["ztr.surrogate_factorization".into()],
    };
    verify_z_transport_derivation(diagram, query, &derivation, limits, ctx)?;
    Ok(Some(derivation))
}

// A direct `TRz` source exchange is legal when the requested intervention is a
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
            .map(|i| DenseNodeId::from_raw(index_u32(i)))
            .ok_or_else(|| IdentificationError::invalid_input("z_transport.unknown_variable"))
    };
    let mut all = BitSet::with_len(graph.node_count());
    for index in 0..graph.node_count() {
        all.insert(DenseNodeId::from_raw(index_u32(index)));
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

/// Independently recheck the graph and formula premises of a zTR derivation
/// under the caller's search limits and execution context.
///
/// # Errors
/// A changed graph, query, or formula; an exhausted replay budget; or
/// cancellation.
pub fn verify_z_transport_derivation(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    derivation: &ZTransportDerivation,
    limits: super::SidLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<(), IdentificationError> {
    derivation.check_inputs(diagram, query)?;
    let input_mismatch =
        || IdentificationError::invalid_derivation("z_transport.proof_input_mismatch");
    let formula_mismatch =
        || IdentificationError::invalid_derivation("z_transport.proof_formula_mismatch");
    match derivation.kind {
        ZFormulaKind::Recursive => {
            if derivation.surrogate.is_some() || derivation.confounder.is_some() {
                return Err(input_mismatch());
            }
            let Some((expected, root, trace)) =
                z_search(diagram, query, limits, ctx, &mut None)?.identified
            else {
                return Err(IdentificationError::invalid_derivation(
                    "z_transport.proof_rule_mismatch",
                ));
            };
            let expected_rules = std::iter::once("ztr.recursive_reduction".to_owned())
                .chain(trace)
                .collect::<Vec<_>>();
            if derivation.root != root
                || derivation.arena != expected
                || derivation.rules != expected_rules
            {
                return Err(formula_mismatch());
            }
        }
        ZFormulaKind::DirectJoint => {
            if !direct_joint_admissible(diagram, query)?
                || derivation.surrogate != Some(query.experiment_assignment[0].variable)
                || derivation.confounder.is_some()
                || derivation.rules != ["ztr.source_exchange_joint"]
            {
                return Err(input_mismatch());
            }
            let (expected, root) = direct_joint_formula(query);
            if derivation.root != root || derivation.arena != expected {
                return Err(formula_mismatch());
            }
        }
        ZFormulaKind::Surrogate => {
            let (Some(surrogate), Some(confounder)) = (derivation.surrogate, derivation.confounder)
            else {
                return Err(input_mismatch());
            };
            if diagram.causal_graph().node_count() != 4
                || !diagram.selection_targets().is_empty()
                || query.outcomes.len() != 1
                || query.treatments.len() != 1
                || query.controllable.as_ref() != [surrogate]
                || query.experiment_assignment.len() != 1
                || query.experiment_assignment[0].variable != surrogate
                || derivation.rules != ["ztr.surrogate_factorization"]
            {
                return Err(input_mismatch());
            }
            if !matches_registered_surrogate_graph(
                diagram,
                confounder,
                surrogate,
                query.treatments[0],
                query.outcomes[0],
            ) {
                return Err(IdentificationError::invalid_derivation(
                    "z_transport.proof_graph_mismatch",
                ));
            }
            let (expected, root) = surrogate_formula(query, confounder);
            if derivation.root != root || derivation.arena != expected {
                return Err(formula_mismatch());
            }
        }
    }
    Ok(())
}

/// Run the `TRz` search, reporting an exhausted engine budget as the
/// z-transport budget. This is the one place that remap happens.
fn z_search(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    limits: super::SidLimits,
    ctx: &antecedent_core::ExecutionContext,
    receipt: &mut Option<ZTransportLimitsReceipt>,
) -> Result<TrzSearchResult, IdentificationError> {
    search_trz_detailed(diagram, query, limits, ctx, receipt).map_err(|error| match error {
        IdentificationError::Budget { budget: IdentificationBudget::Steps } => {
            IdentificationError::budget(IdentificationBudget::ZTransport)
        }
        other => other,
    })
}

/// Bind the checked surrogate formula to a sufficient source joint margin.
///
/// # Errors
/// Invalid catalog or a missing joint margin covering the formula variables.
#[allow(clippy::too_many_lines)]
pub fn bind_z_transport_catalog(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    derivation: &ZTransportDerivation,
    catalog: &EvidenceCatalog,
) -> Result<BoundZTransportFunctional, IdentificationError> {
    derivation.check_inputs(diagram, query)?;
    catalog.validate().map_err(|error| {
        IdentificationError::invalid_catalog(format!("z_transport.invalid_catalog: {error}"))
    })?;
    if derivation.kind == ZFormulaKind::Recursive {
        let mut arena = derivation.arena.clone();
        let mut memo = std::collections::HashMap::new();
        let mut cited = Vec::new();
        let root = bind_recursive_expression(
            derivation.root,
            &mut arena,
            catalog,
            &query.treatments,
            &mut memo,
            &mut cited,
        )?;
        if cited.is_empty() {
            return Err(IdentificationError::InvariantViolated {
                message: "z_transport.recursive_formula_has_no_factor",
            });
        }
        return Ok(BoundZTransportFunctional {
            derivation: derivation.clone(),
            arena,
            root,
            catalog: catalog.clone(),
            cited: cited_regimes(cited),
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
                                && antecedent_core::same_intervention_level(
                                    &actual.value,
                                    &expected.value,
                                )
                        })
                    })
                    && query.outcomes.iter().all(|y| regime.measured.contains(y))
                    && regime.conditioned_on.is_empty()
                    && regime.distribution == DistributionAvailability::Joint
                    && catalog.bindings.iter().any(|binding| binding.regime == regime.id)
            })
            .ok_or_else(|| {
                IdentificationError::missing_evidence(
                    "z_transport.missing_evidence",
                    format!(
                        "{} joint law under do({:?}) measuring {:?}",
                        query.source, query.experiment_assignment, query.outcomes
                    ),
                )
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
            return Err(IdentificationError::InvariantViolated {
                message: "z_transport.invalid_formula_root",
            });
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
            cited: cited_regimes(vec![selected.id]),
        });
    }
    let assignments = &query.experiment_assignment;
    let mut intervention_variables = assignments.iter().map(|a| a.variable).collect::<Vec<_>>();
    intervention_variables.sort_unstable();
    // Both source leaves can be obtained by marginalizing a joint law on
    // {Y, W, X}. Requiring every observed coordinate (or the complete
    // experimental family) here would turn a sufficient positive certificate
    // into a spurious missing-evidence result.
    let Some(confounder) = derivation.confounder else {
        return Err(IdentificationError::InvariantViolated {
            message: "z_transport.surrogate_formula_names_no_confounder",
        });
    };
    let required_margin = [query.outcomes[0], confounder, query.treatments[0]];
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
                            && antecedent_core::same_intervention_level(
                                &actual.value,
                                &expected.value,
                            )
                    })
                })
                && required_margin.iter().all(|variable| regime.measured.contains(variable))
                && regime.conditioned_on.is_empty()
                && regime.distribution == DistributionAvailability::Joint
                && catalog.bindings.iter().any(|binding| binding.regime == regime.id)
        })
        .ok_or_else(|| {
            IdentificationError::missing_evidence(
                "z_transport.missing_evidence",
                format!(
                    "{} joint law under do({:?}) measuring {:?}",
                    query.source, query.experiment_assignment, required_margin
                ),
            )
        })?;

    let mut arena = derivation.arena.clone();
    let ExprNode::SumOut { variables, expr } = derivation.arena.node(derivation.root).clone()
    else {
        return Err(IdentificationError::InvariantViolated {
            message: "z_transport.invalid_formula_root",
        });
    };
    let ExprNode::Product(factors) = derivation.arena.node(expr).clone() else {
        return Err(IdentificationError::InvariantViolated {
            message: "z_transport.invalid_formula_product",
        });
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
            return Err(IdentificationError::InvariantViolated {
                message: "z_transport.invalid_formula_leaf",
            });
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
        cited: cited_regimes(vec![selected.id]),
    })
}

fn cited_regimes(mut cited: Vec<antecedent_core::RegimeId>) -> Arc<[antecedent_core::RegimeId]> {
    cited.sort_unstable_by_key(|regime| regime.raw());
    cited.dedup();
    cited.into()
}

/// How one bound factor selects its law at evaluation.
enum LeafBinding {
    /// One regime supplies every world the factor asks for.
    Single(antecedent_core::RegimeId),
    /// One regime per world of the factor's symbolic coordinates; the concrete
    /// world selects the law at evaluation, so the leaf carries no regime.
    PerWorld(Vec<antecedent_core::RegimeId>),
}

impl LeafBinding {
    fn regimes(&self) -> Vec<antecedent_core::RegimeId> {
        match self {
            Self::Single(regime) => vec![*regime],
            Self::PerWorld(regimes) => regimes.clone(),
        }
    }
}

/// Whether `regime` is an available experiment on exactly the factor's
/// intervention set whose concrete levels agree with the factor's concrete
/// coordinates. A regime declaring no levels stands for the whole family and
/// agrees with every concrete coordinate; a symbolic coordinate agrees with
/// every level.
fn regime_matches_world(
    regime: &antecedent_core::EvidenceRegime,
    interventions: &[antecedent_expr::InterventionAssignment],
) -> bool {
    let intervention_vars = interventions.iter().map(|a| a.variable).collect::<Vec<_>>();
    regime.kind
        == if interventions.is_empty() {
            RegimeKind::Observational
        } else {
            RegimeKind::Experimental
        }
        && same_variable_set(&regime.interventions, &intervention_vars)
        && (regime.intervention_values.is_empty()
            || (regime.intervention_values.len() == interventions.len()
                && interventions.iter().all(|expected| {
                    expected.is_symbolic()
                        || regime.intervention_values.iter().any(|actual| {
                            actual.variable == expected.variable
                                && antecedent_core::same_intervention_level(
                                    &actual.value,
                                    &expected.value,
                                )
                        })
                })))
}

/// Whether `regime` supplies the joint margin a factor reads.
fn regime_supplies_margin(
    catalog: &EvidenceCatalog,
    regime: &antecedent_core::EvidenceRegime,
    needed: &[VariableId],
) -> bool {
    regime.distribution == DistributionAvailability::Joint
        && regime.conditioned_on.is_empty()
        && needed.iter().all(|v| regime.measured.contains(v))
        && catalog.bindings.iter().any(|binding| binding.regime == regime.id)
}

/// Declared finite levels of `variable` in `population`, as the numeric codes a
/// per-world regime names them by.
fn declared_levels(
    catalog: &EvidenceCatalog,
    population: &str,
    variable: VariableId,
) -> Option<Vec<f64>> {
    let coordinate = catalog
        .environments
        .iter()
        .find(|environment| environment.identity.as_ref() == population)?
        .variables
        .iter()
        .find(|coordinate| coordinate.variable == variable)?;
    match coordinate.domain {
        VariableDomain::Binary => Some(vec![0.0, 1.0]),
        VariableDomain::Categorical { cardinality } => {
            Some((0..cardinality).map(f64::from).collect())
        }
        _ => None,
    }
}

/// Bind one factor to the available catalog regimes that supply it.
///
/// A symbolic coordinate the evaluation request binds (a queried treatment)
/// cites every level the catalog supplies, and a request at an unsupplied level
/// is refused at evaluation. A symbolic coordinate an enclosing summation binds
/// needs a regime for every declared level, since every summand is evaluated.
fn bind_leaf(
    catalog: &EvidenceCatalog,
    population: &str,
    interventions: &[antecedent_expr::InterventionAssignment],
    needed: &[VariableId],
    request_bound: &[VariableId],
) -> Result<LeafBinding, IdentificationError> {
    let missing = || {
        IdentificationError::missing_evidence(
            "z_transport.missing_evidence",
            format!("{population} joint factor under {interventions:?}"),
        )
    };
    let candidates = catalog
        .regimes
        .iter()
        .filter(|regime| {
            regime.population.as_ref() == population
                && regime.evidence_kind == EvidenceKind::Available
                && regime_matches_world(regime, interventions)
                && regime_supplies_margin(catalog, regime, needed)
        })
        .collect::<Vec<_>>();
    let symbolic = interventions.iter().filter(|a| a.is_symbolic()).collect::<Vec<_>>();
    if symbolic.is_empty() {
        return candidates
            .iter()
            .map(|regime| regime.id)
            .min_by_key(|regime| regime.raw())
            .map(LeafBinding::Single)
            .ok_or_else(missing);
    }
    // A family regime (no declared levels) supplies every world at once.
    if let Some(family) = candidates
        .iter()
        .filter(|regime| regime.intervention_values.is_empty())
        .map(|regime| regime.id)
        .min_by_key(|regime| regime.raw())
    {
        return Ok(LeafBinding::Single(family));
    }
    // Otherwise every world of the symbolic coordinates needs its own regime.
    let mut worlds = vec![Vec::new()];
    for assignment in &symbolic {
        let levels = if request_bound.contains(&assignment.variable) {
            let mut supplied = candidates
                .iter()
                .flat_map(|regime| regime.intervention_values.iter())
                .filter(|actual| actual.variable == assignment.variable)
                .filter_map(|actual| actual.value.as_f64())
                .collect::<Vec<_>>();
            supplied.sort_by(f64::total_cmp);
            supplied.dedup();
            if supplied.is_empty() {
                return Err(missing());
            }
            supplied
        } else {
            declared_levels(catalog, population, assignment.variable).ok_or_else(|| {
                IdentificationError::missing_evidence(
                    "z_transport.missing_evidence",
                    format!("{population} declares no finite domain for {:?}", assignment.variable),
                )
            })?
        };
        worlds = worlds
            .into_iter()
            .flat_map(|prefix| {
                levels.iter().map(move |level| {
                    let mut world = prefix.clone();
                    world.push(*level);
                    world
                })
            })
            .collect();
    }
    let mut regimes = Vec::with_capacity(worlds.len());
    for world in worlds {
        let mut supplying = candidates.iter().filter(|regime| {
            symbolic.iter().zip(&world).all(|(assignment, level)| {
                regime.intervention_values.iter().any(|actual| {
                    actual.variable == assignment.variable && actual.value.as_f64() == Some(*level)
                })
            })
        });
        let Some(regime) = supplying.next() else {
            return Err(IdentificationError::missing_evidence(
                "z_transport.missing_evidence",
                format!("{population} joint factor under {interventions:?} at world {world:?}"),
            ));
        };
        if supplying.next().is_some() {
            return Err(IdentificationError::invalid_catalog(format!(
                "z_transport.ambiguous_evidence: {population} supplies world {world:?} under {interventions:?} more than once"
            )));
        }
        regimes.push(regime.id);
    }
    Ok(LeafBinding::PerWorld(regimes))
}

fn bind_recursive_expression(
    id: ExprId,
    arena: &mut CausalExprArena,
    catalog: &EvidenceCatalog,
    treatments: &[VariableId],
    memo: &mut std::collections::HashMap<ExprId, ExprId>,
    cited: &mut Vec<antecedent_core::RegimeId>,
) -> Result<ExprId, IdentificationError> {
    if let Some(hit) = memo.get(&id) {
        return Ok(*hit);
    }
    let node = arena.node(id).clone();
    let bound = match node {
        ExprNode::Distribution {
            variables,
            conditioned_on,
            intervention,
            domain,
            population,
            ..
        } => {
            let needed = arena
                .var_set(variables)
                .iter()
                .chain(arena.var_set(conditioned_on))
                .copied()
                .collect::<Vec<_>>();
            let binding = bind_leaf(
                catalog,
                arena.population(population),
                arena.intervention_assignments(intervention),
                &needed,
                treatments,
            )?;
            let regime = match &binding {
                LeafBinding::Single(regime) => Some(*regime),
                LeafBinding::PerWorld(_) => None,
            };
            cited.extend(binding.regimes());
            arena.intern(ExprNode::Distribution {
                variables,
                conditioned_on,
                intervention,
                domain,
                population,
                regime,
            })
        }
        ExprNode::Product(list) => {
            let children = arena.list(list).to_vec();
            let bound_children = children
                .into_iter()
                .map(|child| {
                    bind_recursive_expression(child, arena, catalog, treatments, memo, cited)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let list = arena.intern_list(bound_children);
            arena.intern(ExprNode::Product(list))
        }
        ExprNode::SumOut { variables, expr } => {
            let expr = bind_recursive_expression(expr, arena, catalog, treatments, memo, cited)?;
            arena.intern(ExprNode::SumOut { variables, expr })
        }
        ExprNode::Ratio { numerator, denominator } => {
            let numerator =
                bind_recursive_expression(numerator, arena, catalog, treatments, memo, cited)?;
            let denominator =
                bind_recursive_expression(denominator, arena, catalog, treatments, memo, cited)?;
            arena.intern(ExprNode::Ratio { numerator, denominator })
        }
        _ => {
            return Err(IdentificationError::InvariantViolated {
                message: "z_transport.unsupported_recursive_node",
            });
        }
    };
    memo.insert(id, bound);
    Ok(bound)
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
    let mut exact_regime = false;
    let mut missing_joint = false;
    let mut missing_margin = false;
    let mut missing_binding = false;
    let mut supplied_by = Vec::new();
    for regime in catalog.regimes.iter().filter(|regime| {
        regime.population.as_ref() == name
            && regime.evidence_kind == EvidenceKind::Available
            && regime_matches_world(regime, assignments)
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

/// The registered surrogate formula `Σ_w P_s(y | w, x, do(z)) P_s(w | do(z))`.
fn surrogate_formula(query: &ZTransportQuery, confounder: VariableId) -> (CausalExprArena, ExprId) {
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
    (arena, root)
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
                .children(antecedent_graph::DenseNodeId::from_raw(index_u32(from)))
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
                .bidirected_neighbors(antecedent_graph::DenseNodeId::from_raw(index_u32(a)))
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

/// Set equality of two coordinate lists: repeats and order do not matter.
fn same_variable_set(left: &[VariableId], right: &[VariableId]) -> bool {
    left.iter().collect::<std::collections::BTreeSet<_>>()
        == right.iter().collect::<std::collections::BTreeSet<_>>()
}

// Bareinboim–Pearl (AAAI 2013, Figure 4) rules 1–4 map to the marginal,
// ancestor, enlargement, and district branches below; rules 5–8 map to the
// C-component, factor, and subdistrict branches. Rule 10 exchanges the active
// controllables in X and recurses on the reduced graph; rule 11 is recorded
// only when that exchange's separation premise fails or Z∩X is empty.
#[cfg(test)]
fn search_trz(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    limits: super::SidLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<Option<(CausalExprArena, ExprId, Vec<String>)>, IdentificationError> {
    Ok(search_trz_detailed(diagram, query, limits, ctx, &mut None)?.identified)
}

#[derive(Clone, Debug, PartialEq)]
struct TrzTerminalFailure {
    outcomes: Vec<u32>,
    treatments: Vec<u32>,
    vertices: Vec<u32>,
    c0: Vec<u32>,
    remaining_controllable: Vec<u32>,
    candidate_active: Vec<u32>,
    active_interventions: Vec<(u32, Option<f64>)>,
    selection_separated: bool,
    rules: Vec<String>,
}

#[derive(Debug)]
struct TrzSearchResult {
    identified: Option<(CausalExprArena, ExprId, Vec<String>)>,
    terminal_failure: Option<TrzTerminalFailure>,
    unassigned: Option<VariableId>,
    /// Every recursive rule the search applied, in order, whatever the outcome.
    explored_rules: Vec<String>,
    /// Recursive subproblems the search charged.
    steps_explored: usize,
    /// Deepest recursion level the search reached.
    depth_reached: usize,
}

fn search_trz_detailed(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    limits: super::SidLimits,
    ctx: &antecedent_core::ExecutionContext,
    receipt: &mut Option<ZTransportLimitsReceipt>,
) -> Result<TrzSearchResult, IdentificationError> {
    let classical = super::ClassicalTransportQuery {
        outcomes: Arc::clone(&query.outcomes),
        treatments: Arc::clone(&query.treatments),
        source: Arc::clone(&query.source),
        target: Arc::clone(&query.target),
    };
    // A budget or cancellation observed at engine construction stops the search
    // before any recursive accounting is taken, so the receipt reports the
    // limits in force with no consumed counts rather than fabricating them.
    let mut engine = match super::Engine::new(diagram, &classical, limits, ctx) {
        Ok(engine) => engine,
        Err(error) if error.is_budget_or_cancel() => {
            *receipt = Some(ZTransportLimitsReceipt {
                budget: pre_search_budget_kind(&error),
                steps_limit: limits.steps,
                depth_limit: limits.depth,
                steps_consumed: None,
                depth_reached: None,
                explored_rules: Vec::new(),
            });
            return Err(error);
        }
        Err(error) => return Err(error),
    };
    let initial = match engine.initial() {
        Ok(initial) => initial,
        Err(error) if error.is_budget_or_cancel() => {
            *receipt = Some(search_limits_receipt(&error, &engine, limits, Vec::new()));
            return Err(error);
        }
        Err(error) => return Err(error),
    };
    let mut trace = Vec::new();
    let mut terminal_failure = None;
    let mut unassigned = None;
    let irrelevant = BitSet::with_len(diagram.causal_graph().node_count());
    let outcome = search_trz_state(
        &mut engine,
        initial,
        query,
        &query.controllable,
        &[],
        &irrelevant,
        0,
        &mut trace,
        &mut terminal_failure,
        &mut unassigned,
    );
    match outcome {
        Ok(result) => {
            let steps_explored = engine.steps;
            let depth_reached = engine.max_depth;
            Ok(TrzSearchResult {
                identified: result.map(|root| (engine.arena, root, trace.clone())),
                terminal_failure,
                unassigned,
                explored_rules: trace,
                steps_explored,
                depth_reached,
            })
        }
        Err(error) if error.is_budget_or_cancel() => {
            *receipt = Some(search_limits_receipt(&error, &engine, limits, trace));
            Err(error)
        }
        Err(error) => Err(error),
    }
}

/// Classify a budget or cancellation observed before the recursive search runs.
fn pre_search_budget_kind(error: &IdentificationError) -> ZTransportBudgetKind {
    match error {
        IdentificationError::Cancelled => ZTransportBudgetKind::Cancelled,
        IdentificationError::Budget { budget: IdentificationBudget::Memory } => {
            ZTransportBudgetKind::Memory
        }
        _ => ZTransportBudgetKind::Steps,
    }
}

/// Build a receipt from the engine's own accounting when a budget or
/// cancellation stopped the recursive search. The step-versus-depth
/// distinction is read from the accounting, never guessed.
fn search_limits_receipt(
    error: &IdentificationError,
    engine: &super::Engine<'_>,
    limits: super::SidLimits,
    explored_rules: Vec<String>,
) -> ZTransportLimitsReceipt {
    let budget = match error {
        IdentificationError::Cancelled => ZTransportBudgetKind::Cancelled,
        IdentificationError::Budget { budget: IdentificationBudget::Memory } => {
            ZTransportBudgetKind::Memory
        }
        _ if engine.max_depth >= limits.depth => ZTransportBudgetKind::Depth,
        _ => ZTransportBudgetKind::Steps,
    };
    ZTransportLimitsReceipt {
        budget,
        steps_limit: limits.steps,
        depth_limit: limits.depth,
        steps_consumed: Some(engine.steps),
        depth_reached: Some(engine.max_depth),
        explored_rules,
    }
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)] // Mirrors the paper rule state explicitly.
fn search_trz_state(
    engine: &mut super::Engine<'_>,
    state: super::State,
    query: &ZTransportQuery,
    remaining_controllable: &[VariableId],
    active_interventions: &[antecedent_expr::InterventionAssignment],
    irrelevant: &BitSet,
    depth: usize,
    trace: &mut Vec<String>,
    terminal_failure: &mut Option<TrzTerminalFailure>,
    unassigned: &mut Option<VariableId>,
) -> Result<Option<ExprId>, IdentificationError> {
    engine.charge(depth)?;
    let mut ws = antecedent_graph::GraphWorkspace::default();
    // `TRz` rule 1: no active treatment coordinates remain.
    if !state.x.any() {
        trace.push("ztr.line1.marginal".into());
        return Ok(Some(engine.marginal(state.kernel, &super::difference(&state.v, &state.y))?));
    }
    let ancestors = engine.prepared.ancestors_within(&state.y, &state.v, &mut ws);
    // `TRz` rule 2: restrict to ancestors of the current outcomes.
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
            irrelevant,
            depth + 1,
            trace,
            terminal_failure,
            unassigned,
        );
    }
    let mut enlarged = super::difference(&state.v, &state.x);
    let bar = engine.prepared.ancestors_bar_x(&state.y, &state.v, &state.x, &mut ws);
    enlarged.difference_with(&bar);
    // `TRz` rule 3: add non-treatment vertices outside An(Y) in D_X. The child
    // is constant in them, so a later exchange may fix them at a declared level.
    if enlarged.any() {
        trace.push("ztr.line3.enlarge".into());
        let mut next = state.clone();
        next.x.union_with(&enlarged);
        let mut fixed = irrelevant.clone();
        fixed.union_with(&enlarged);
        let Some(child) = search_trz_state(
            engine,
            next,
            query,
            remaining_controllable,
            active_interventions,
            &fixed,
            depth + 1,
            trace,
            terminal_failure,
            unassigned,
        )?
        else {
            return Ok(None);
        };
        // Rule 3 is the identity: P_x(y) = P_{x,w}(y). The child names a
        // coordinate of `w` only where a later exchange fixed it at its
        // declared level, so it is returned unchanged; a child that still
        // reads `w` is averaged over the kernel's own law of `w` given `x`,
        // which is zero exactly where the child conditions on a null event.
        let enlarged_vars = engine.vars(&enlarged)?;
        let free = engine.arena.free_variables(child);
        if enlarged_vars.iter().any(|variable| free.contains(variable)) {
            trace.push("ztr.line3.kernel_weighted".into());
            return Ok(Some(engine.enlarge_output(&state, &enlarged, child)?));
        }
        return Ok(Some(child));
    }
    let districts = engine.prepared.c_components(&super::difference(&state.v, &state.x));
    // `TRz` rule 4: factor over multiple districts in D \ X.
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
                irrelevant,
                depth + 1,
                trace,
                terminal_failure,
                unassigned,
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
    let district = districts
        .first()
        .ok_or(IdentificationError::InvariantViolated { message: "z_transport.empty_district" })?;
    let containing = engine.prepared.c_components(&state.v);
    // `TRz` rules 5–6 establish C0 and whether D is a single c-component.
    if containing.len() > 1 {
        if containing.iter().any(|d| d.equal_set(district)) {
            // `TRz` rule 7: C0 is a c-component of D, so its kernel is available.
            trace.push("ztr.line7.factor".into());
            let kernel = engine.factor(&state, district)?;
            return Ok(Some(engine.marginal(kernel, &super::difference(district, &state.y))?));
        }
        let larger = containing.iter().find(|d| district.is_subset_of(d)).ok_or(
            IdentificationError::InvariantViolated {
                message: "z_transport.missing_containing_district",
            },
        )?;
        let next = super::State {
            y: state.y.clone(),
            x: super::intersection(&state.x, larger),
            v: larger.clone(),
            kernel: engine.factor(&state, larger)?,
        };
        // `TRz` rule 8: recurse into the containing c-component of D.
        trace.push("ztr.line8.recurse".into());
        return search_trz_state(
            engine,
            next,
            query,
            remaining_controllable,
            active_interventions,
            irrelevant,
            depth + 1,
            trace,
            terminal_failure,
            unassigned,
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
        // `TRz` rule 11: rule 10 cannot exchange an active experiment.
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
                    .map(|a| (a.variable.raw(), a.value.as_f64()))
                    .collect(),
                selection_separated,
                rules: trace.clone(),
            });
        }
        return Ok(None);
    }
    let active_vars = engine.vars(&activated)?;
    // The level each exchanged coordinate takes in the cited source experiment.
    // A queried treatment and a coordinate that rule 4 handed to this district
    // stay symbolic: the evaluation request binds the former and the enclosing
    // summation binds the latter, so one formula answers every requested level
    // and every summand at its own world. A coordinate that rule 3 proved
    // irrelevant to the child may be fixed at its declared level, since the
    // child is constant in it.
    let mut exchanged = Vec::with_capacity(active_vars.len());
    for variable in active_vars.iter().copied() {
        let dense = engine.prepared.var_to_dense(variable)?;
        if irrelevant.contains(dense) && !query.treatments.contains(&variable) {
            let Some(assignment) = query
                .experiment_assignment
                .iter()
                .find(|assignment| assignment.variable == variable)
            else {
                *unassigned = Some(variable);
                return Ok(None);
            };
            exchanged.push(antecedent_expr::InterventionAssignment::concrete(
                variable,
                assignment.value.clone(),
            ));
        } else {
            exchanged.push(antecedent_expr::InterventionAssignment::symbolic(variable));
        }
    }
    trace.push(format!(
        "ztr.line10.source_exchange:{:?}",
        exchanged.iter().map(|a| (a.variable.raw(), a.value.as_f64())).collect::<Vec<_>>()
    ));
    let mut cumulative = active_interventions.to_vec();
    for assignment in exchanged {
        if !cumulative.iter().any(|a| a.variable == assignment.variable) {
            cumulative.push(assignment);
        }
    }
    // `TRz` rule 10: exchange Z∩X and recurse with Z\X and active set I=Z∩X.
    // The new kernel is the source c-factor Q^s_z[V\Z] = P^s_{z ∪ (V_all\V)}(V\Z):
    // every coordinate outside the current vertex set is intervened, not
    // conditioned on. When the state still spans the whole graph that is the
    // cited source law itself; otherwise the c-factor is identified from that
    // law by the interventional-distribution recursion on the graph with the
    // exchanged coordinates removed (sID^z line 7).
    let remaining = super::difference(&state.v, &activated);
    let Some(kernel) = source_c_factor(engine, &remaining, &cumulative, query, depth, trace)?
    else {
        return Ok(None);
    };
    let next = super::State {
        y: state.y,
        x: super::difference(&state.x, &activated),
        v: remaining,
        kernel,
    };
    // Fig. 4 line 10 recurses into `TRz` on the reduced graph and retains
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
        irrelevant,
        depth + 1,
        trace,
        terminal_failure,
        unassigned,
    )
}

/// The source c-factor `Q^s_z[remaining]` after exchanging `cumulative`, in the
/// z engine's arena. `None` when the interventional-distribution recursion cannot
/// identify it from the cited source law; that is not an obstruction claim.
fn source_c_factor(
    engine: &mut super::Engine<'_>,
    remaining: &BitSet,
    cumulative: &[antecedent_expr::InterventionAssignment],
    query: &ZTransportQuery,
    depth: usize,
    trace: &mut Vec<String>,
) -> Result<Option<ExprId>, IdentificationError> {
    let graph = engine.diagram.causal_graph();
    let exchanged = cumulative.iter().map(|a| a.variable).collect::<Vec<_>>();
    let remaining_vars = engine.vars(remaining)?;
    // An outside coordinate that is not an ancestor of the remaining set in the
    // graph without the exchanged coordinates neither changes the c-factor
    // when intervened on nor needs to be measured: the source law over the
    // ancestral closure is the marginal the theorem reads. Only the ancestral
    // outside coordinates are intervened on and identified away.
    let mut closure = BitSet::with_len(graph.node_count());
    let mut pending = remaining.to_dense_ids();
    while let Some(node) = pending.pop() {
        if closure.contains(node) {
            continue;
        }
        closure.insert(node);
        for parent in graph.parents(node) {
            let parent_var = engine.prepared.dense_to_var(*parent)?;
            if !exchanged.contains(&parent_var) {
                pending.push(*parent);
            }
        }
    }
    let kept = engine.vars(&closure)?;
    let outside = kept.iter().copied().filter(|v| !remaining_vars.contains(v)).collect::<Vec<_>>();
    let population = engine.arena.intern_population(Arc::clone(&query.source));
    if outside.is_empty() {
        let variables = engine.arena.intern_var_set(remaining_vars);
        let conditioned_on = engine.arena.empty_var_set();
        let intervention = engine.arena.intern_intervention_assignments(cumulative.iter().cloned());
        return Ok(Some(engine.arena.intern(ExprNode::Distribution {
            variables,
            conditioned_on,
            intervention,
            domain: DomainRef::Interventional,
            population,
            regime: None,
        })));
    }
    // Identify P^s_{z ∪ outside}(remaining) from P^s_z(kept) on the graph
    // without the exchanged coordinates.
    let mut reduced = antecedent_graph::Admg::empty();
    for variable in &kept {
        reduced.add_node(NodeRef::Static(*variable))?;
    }
    let dense = |variable: VariableId| {
        kept.iter().position(|v| *v == variable).map(|i| DenseNodeId::from_raw(index_u32(i)))
    };
    for (from_index, node) in graph.nodes().iter().enumerate() {
        let NodeRef::Static(from) = node else { continue };
        let Some(reduced_from) = dense(*from) else { continue };
        let from_dense = DenseNodeId::from_raw(index_u32(from_index));
        for to in graph.children(from_dense) {
            if let Some(reduced_to) = engine.prepared.dense_to_var(*to).ok().and_then(dense) {
                reduced.insert_directed(reduced_from, reduced_to)?;
            }
        }
        for other in graph.bidirected_neighbors(from_dense) {
            if other.as_usize() > from_index {
                if let Some(reduced_other) =
                    engine.prepared.dense_to_var(*other).ok().and_then(dense)
                {
                    reduced.insert_bidirected(reduced_from, reduced_other)?;
                }
            }
        }
    }
    let diagram = SelectionDiagram::try_new(reduced, Arc::<[VariableId]>::from([]))?;
    let classical = super::ClassicalTransportQuery {
        outcomes: remaining_vars.clone().into(),
        treatments: outside.into(),
        source: Arc::clone(&query.source),
        target: Arc::clone(&query.target),
    };
    let limits = super::SidLimits {
        steps: engine.limits.steps.saturating_sub(engine.steps).max(1),
        depth: engine.limits.depth.saturating_sub(depth).max(1),
    };
    let mut sub = super::Engine::new(&diagram, &classical, limits, engine.ctx)?;
    let initial = sub.initial_source_kernel(&query.source, cumulative)?;
    let solved = sub.solve(initial, false, 0);
    engine.steps = engine.steps.saturating_add(sub.steps);
    let Some(root_step) = solved? else {
        trace.push("ztr.line10.c_factor_not_identified".into());
        return Ok(None);
    };
    trace.push(format!("ztr.line10.c_factor:{:?}", sub.reachable_rules(root_step)));
    let output = sub.proof[root_step].output;
    let mut memo = std::collections::HashMap::new();
    Ok(Some(transplant(&sub.arena, output, &mut engine.arena, &mut memo)?))
}

/// Copy the expression `id` of `from` into `into`, re-interning every table.
fn transplant(
    from: &CausalExprArena,
    id: ExprId,
    into: &mut CausalExprArena,
    memo: &mut std::collections::HashMap<ExprId, ExprId>,
) -> Result<ExprId, IdentificationError> {
    if let Some(copied) = memo.get(&id) {
        return Ok(*copied);
    }
    let copied = match from.node(id).clone() {
        ExprNode::Distribution {
            variables,
            conditioned_on,
            intervention,
            domain,
            population,
            regime,
        } => {
            let variables = into.intern_var_set(from.var_set(variables).iter().copied());
            let conditioned_on = into.intern_var_set(from.var_set(conditioned_on).iter().copied());
            let intervention = into.intern_intervention_assignments(
                from.intervention_assignments(intervention).iter().cloned(),
            );
            let population = into.intern_population(Arc::from(from.population(population)));
            into.intern(ExprNode::Distribution {
                variables,
                conditioned_on,
                intervention,
                domain,
                population,
                regime,
            })
        }
        ExprNode::Product(list) => {
            let children = from
                .list(list)
                .iter()
                .map(|child| transplant(from, *child, into, memo))
                .collect::<Result<Vec<_>, _>>()?;
            let list = into.intern_list(children);
            into.intern(ExprNode::Product(list))
        }
        ExprNode::SumOut { variables, expr } => {
            let expr = transplant(from, expr, into, memo)?;
            let variables = into.intern_var_set(from.var_set(variables).iter().copied());
            into.intern(ExprNode::SumOut { variables, expr })
        }
        ExprNode::Ratio { numerator, denominator } => {
            let numerator = transplant(from, numerator, into, memo)?;
            let denominator = transplant(from, denominator, into, memo)?;
            into.intern(ExprNode::Ratio { numerator, denominator })
        }
        _ => {
            return Err(IdentificationError::InvariantViolated {
                message: "source c-factor identification produced an unexpected node",
            });
        }
    };
    memo.insert(id, copied);
    Ok(copied)
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
            Admg::with_variables(index_u32(graph_size)),
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
            experiment_assignment: controllable.first().map_or_else(
                || Arc::from([]),
                |variable| {
                    Arc::from([CatalogInterventionAssignment {
                        variable: VariableId::from_raw(*variable),
                        value: Value::Bool(false),
                    }])
                },
            ),
            source: Arc::from("source"),
            target: Arc::from("target"),
        };
        (diagram, query)
    }

    #[test]
    fn bounded_query_contract_accepts_declared_controllable_set() {
        let (diagram, query) = query(12, &[1, 2, 3, 4]);
        validate_z_transport_query(&diagram, &query).unwrap();
    }

    #[test]
    fn bounded_query_contract_refuses_oversized_graph_and_controllable_set() {
        let (diagram, oversized_graph_query) = query(13, &[1]);
        assert_eq!(
            validate_z_transport_query(&diagram, &oversized_graph_query).unwrap_err().to_string(),
            "z_transport.unsupported_observed_count"
        );
        let (diagram, oversized_control_query) = query(12, &[0, 1, 2, 3, 4]);
        assert_eq!(
            validate_z_transport_query(&diagram, &oversized_control_query).unwrap_err().to_string(),
            "z_transport.unsupported_controllable_count"
        );
    }

    #[test]
    fn negative_family_enumeration_stops_at_the_regime_budget() {
        let (diagram, query) = query(4, &[0, 1, 2, 3]);
        let coordinates = (0..4)
            .map(|raw| VariableCoordinate {
                variable: VariableId::from_raw(raw),
                domain: VariableDomain::Categorical { cardinality: 10 },
                unit: None,
            })
            .collect::<Vec<_>>();
        let source = Environment::try_new("source", coordinates.clone(), []).unwrap();
        let target = Environment::try_new("target", coordinates, []).unwrap();
        let catalog = EvidenceCatalog::try_new([source, target], [], [], None).unwrap();
        assert!(matches!(
            validate_z_experiment_family(&diagram, &query, &catalog),
            Err(ZExperimentFamilyError::FamilyExceedsBudget { regimes }) if regimes > Z_TRANSPORT_MAX_FAMILY_REGIMES
        ));
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
        let ZTransportResult::Identified(proof) = identify_z_transport(
            &diagram,
            &query,
            super::super::SidLimits::default(),
            &antecedent_core::ExecutionContext::for_tests(0),
        )
        .unwrap() else {
            panic!("registered surrogate graph should be identified");
        };
        assert_eq!(proof.surrogate(), Some(VariableId::from_raw(1)));
        assert_eq!(proof.confounder(), Some(VariableId::from_raw(0)));
        let ExprNode::SumOut { variables, expr } = proof.arena().node(proof.root()) else {
            panic!("formula root must sum out W");
        };
        assert_eq!(proof.arena().var_set(*variables), &[VariableId::from_raw(0)]);
        let ExprNode::Product(factors) = proof.arena().node(*expr) else {
            panic!("formula must multiply two source factors");
        };
        assert_eq!(proof.arena().list(*factors).len(), 2);
        verify_z_transport_derivation(
            &diagram,
            &query,
            &proof,
            super::super::SidLimits::default(),
            &antecedent_core::ExecutionContext::for_tests(0),
        )
        .unwrap();
        let mut altered = query.clone();
        altered.controllable = Arc::from([VariableId::from_raw(0)]);
        assert!(
            verify_z_transport_derivation(
                &diagram,
                &altered,
                &proof,
                super::super::SidLimits::default(),
                &antecedent_core::ExecutionContext::for_tests(0)
            )
            .is_err()
        );
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
        let ZTransportResult::Identified(proof) = identify_z_transport(
            &diagram,
            &query,
            super::super::SidLimits::default(),
            &antecedent_core::ExecutionContext::for_tests(0),
        )
        .unwrap() else {
            panic!("direct joint source exchange must be identified")
        };
        assert_eq!(proof.to_record().rules, ["ztr.source_exchange_joint"]);
        assert!(matches!(proof.arena().node(proof.root()), ExprNode::Distribution { .. }));
        let reconstructed = ZTransportDerivation::from_record_checked(
            &diagram,
            &query,
            &proof.to_record(),
            proof.arena().clone(),
            super::super::SidLimits::default(),
            &antecedent_core::ExecutionContext::for_tests(0),
        )
        .unwrap();
        verify_z_transport_derivation(
            &diagram,
            &query,
            &reconstructed,
            super::super::SidLimits::default(),
            &antecedent_core::ExecutionContext::for_tests(0),
        )
        .unwrap();
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
        assert!(
            verify_z_transport_derivation(
                &diagram,
                &changed,
                &proof,
                super::super::SidLimits::default(),
                &antecedent_core::ExecutionContext::for_tests(0)
            )
            .is_err()
        );
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
        for stratum in &experiment {
            let pw = stratum.iter().flatten().sum::<f64>();
            let y_given_w_x = stratum[0][1] / (stratum[0][0] + stratum[0][1]);
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
    #[allow(clippy::too_many_lines)]
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
            let node = match arena.node(ExprId::from_raw(index_u32(index))).clone() {
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
        let ZTransportResult::Identified(proof) = identify_z_transport(
            &diagram,
            &query,
            super::super::SidLimits::default(),
            &antecedent_core::ExecutionContext::for_tests(0),
        )
        .unwrap() else {
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
            super::super::SidLimits::default(),
            &antecedent_core::ExecutionContext::for_tests(0),
        )
        .unwrap();
    }

    #[test]
    fn bounded_search_reports_exhaustion_separately_from_noncertification() {
        let (diagram, query) = surrogate_fixture();
        let exhausted = identify_z_transport(
            &diagram,
            &query,
            super::super::SidLimits { steps: 0, depth: 1 },
            &antecedent_core::ExecutionContext::for_tests(1),
        )
        .unwrap_err();
        assert_eq!(exhausted.to_string(), "z_transport.exhausted_computation");
    }

    struct SequentialExchangeScm;

    #[allow(clippy::float_cmp)] // The provider accepts only binary values 0.0 and 1.0.
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
    #[allow(clippy::too_many_lines)]
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
        let result = identify_z_transport(
            &diagram,
            &query,
            super::super::SidLimits::default(),
            &antecedent_core::ExecutionContext::for_tests(0),
        )
        .unwrap();
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
    fn a_starved_search_reports_the_z_transport_budget_never_the_sid_step_budget() {
        // The five-node graph is not a registered surrogate, so every level
        // goes through the recursion whose step budget is the SID engine's.
        let mut graph = Admg::with_variables(5);
        for (a, b) in [(0, 1), (1, 2), (2, 3), (0, 3)] {
            graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        for (a, b) in [(0, 3), (1, 3), (1, 2)] {
            graph.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        let diagram = SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from([])).unwrap();
        let (_, query) = surrogate_fixture();
        let ctx = antecedent_core::ExecutionContext::for_tests(0);
        for steps in [1, 2, 3] {
            let error = identify_z_transport(
                &diagram,
                &query,
                super::super::SidLimits { steps, depth: 8 },
                &ctx,
            )
            .expect_err("a starved search has no result");
            assert!(
                matches!(
                    error,
                    IdentificationError::Budget { budget: IdentificationBudget::ZTransport }
                ),
                "steps={steps}: {error}"
            );
            assert_eq!(error.to_string(), "z_transport.exhausted_computation");
        }
        assert!(matches!(
            identify_z_transport(&diagram, &query, super::super::SidLimits::default(), &ctx)
                .unwrap(),
            ZTransportResult::Identified(_)
        ));
    }

    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "one fixture exercised under every family-validation failure"
    )]
    fn family_validation_requires_each_joint_intervention_law() {
        let (diagram, query) = surrogate_fixture();
        let coords = (0..4)
            .map(|raw| VariableCoordinate {
                variable: VariableId::from_raw(raw),
                domain: VariableDomain::Binary,
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
        let ZTransportResult::Identified(proof) = identify_z_transport(
            &diagram,
            &query,
            super::super::SidLimits::default(),
            &antecedent_core::ExecutionContext::for_tests(0),
        )
        .unwrap() else {
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
