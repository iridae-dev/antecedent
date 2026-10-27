//! Contract validation for the bounded single-source z-transport setting.
//!
//! The contract is kept distinct from classical sID: a source that can intervene
//! only on `controllable` does not satisfy the unrestricted source-experiment
//! premise of the classical theorem.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
use super::IdentificationBudget;
use super::IdentificationError;

use antecedent_core::{
    DistributionAvailability, EvidenceCatalog,
    InterventionAssignment as CatalogInterventionAssignment, RegimeKind, VariableDomain,
    VariableId,
};

#[cfg(test)]
use antecedent_expr::ExprNode;
use antecedent_expr::{CausalExprArena, ExprId};

use antecedent_graph::{NodeRef, SelectionDiagram};

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
    /// Retained for compatibility: the family is no longer sized against a fixed
    /// budget. It is enumerated lazily against the catalog, and a family larger than
    /// the catalog's regimes is reported as the first [`Self::MissingJointLaw`].
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

impl ZTransportDecision {
    /// Identification status in the shared transport vocabulary
    /// ([`TransportOutcomeKind::IDENTIFICATION`](antecedent_core::TransportOutcomeKind::IDENTIFICATION)).
    /// A budget or cancellation stop is not a decision; its limits receipt reports
    /// `budget_cancel` (legacy Python `outcome` spelling `exhausted`).
    #[must_use]
    pub const fn identification_status(&self) -> antecedent_core::TransportOutcomeKind {
        match self {
            Self::Identified(_) => antecedent_core::TransportOutcomeKind::Identified,
            Self::ProvenNonTransportable(_) => {
                antecedent_core::TransportOutcomeKind::ProvenNonTransportable
            }
            Self::MissingEvidence { .. } => antecedent_core::TransportOutcomeKind::MissingEvidence,
            Self::NotCertified { .. } => antecedent_core::TransportOutcomeKind::NotCertified,
        }
    }
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
#[allow(
    clippy::large_enum_variant,
    reason = "Public result keeps both source certificates directly inspectable"
)]
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

impl TwoSourceZTransportDecision {
    /// Identification status in the shared transport vocabulary
    /// ([`TransportOutcomeKind::IDENTIFICATION`](antecedent_core::TransportOutcomeKind::IDENTIFICATION)).
    ///
    /// Its legacy Python `outcome` spelling `combined_identified` reads as `identified`.
    #[must_use]
    pub const fn identification_status(&self) -> antecedent_core::TransportOutcomeKind {
        match self {
            Self::Identified { .. } | Self::CombinedIdentified { .. } => {
                antecedent_core::TransportOutcomeKind::Identified
            }
            Self::ProvenNonTransportable { .. } => {
                antecedent_core::TransportOutcomeKind::ProvenNonTransportable
            }
            Self::NotCertified { .. } => antecedent_core::TransportOutcomeKind::NotCertified,
        }
    }
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

impl ZTransportResult {
    /// Identification status in the shared transport vocabulary
    /// ([`TransportOutcomeKind::IDENTIFICATION`](antecedent_core::TransportOutcomeKind::IDENTIFICATION)).
    #[must_use]
    pub const fn identification_status(&self) -> antecedent_core::TransportOutcomeKind {
        match self {
            Self::Identified(_) => antecedent_core::TransportOutcomeKind::Identified,
            Self::NotCertified { .. } => antecedent_core::TransportOutcomeKind::NotCertified,
        }
    }
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
    #[allow(
        clippy::needless_pass_by_value,
        reason = "Public checked-import API consumes the untrusted record"
    )]
    pub fn from_record_checked(
        record: ZTransportObstructionRecord,
        diagram: &SelectionDiagram,
        query: &ZTransportQuery,
        limits: super::SidLimits,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<Self, IdentificationError> {
        let ZDerivation::Line11 { terminal, .. } =
            derive_z_transport(diagram, query, super::SidMeter::Limits(limits), ctx, &mut None)?
        else {
            return Err(IdentificationError::invalid_derivation(
                "z_transport.obstruction_not_reproduced",
            ));
        };
        let ZTransportDecision::ProvenNonTransportable(checked) = certify_line11_obstruction(
            diagram,
            query,
            terminal,
            super::SidMeter::Limits(limits),
            ctx,
        )?
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

mod search;

pub use search::bind_z_transport_catalog;

#[cfg(test)]
use search::search_trz;

pub(crate) use search::{
    TrzCall, TrzDomain, TrzTerminalFailure, bind_recursive_expression, cited_regimes,
    search_trz_call,
};

use search::inspect_z_expression;

mod decide;
mod derive;
mod validate;
use decide::certify_line11_obstruction;
pub(super) use decide::decide_z_transport_reporting;
pub use decide::{
    decide_two_source_z_transport, decide_z_transport_inspecting, decide_z_transport_with_catalog,
    intervention_ancestral_treatments, intervention_levels_conflict, intervention_mutilated_admg,
};
use derive::{ZDerivation, derive_z_transport, verify_z_obstruction_metered};
pub use derive::{
    identify_z_transport, identify_z_transport_reporting, verify_z_transport_derivation,
};
pub use validate::{validate_z_experiment_family, validate_z_transport_query};

#[cfg(test)]
mod tests;
