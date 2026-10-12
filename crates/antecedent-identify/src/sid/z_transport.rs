//! Contract validation for the bounded single-source z-transport setting.
//!
//! The contract is kept distinct from classical sID: a source that can intervene
//! only on `controllable` does not satisfy the unrestricted source-experiment
//! premise of the classical theorem.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{IdentificationBudget, IdentificationError};
use antecedent_core::{
    DistributionAvailability, EvidenceCatalog,
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
pub(crate) const Z_TRANSPORT_MAX_OBSERVED: usize = 12;
/// Explicit bound for controllable variables.
pub(crate) const Z_TRANSPORT_MAX_CONTROLLABLE: usize = 4;
/// Largest source-experiment family enumerated for a negative certificate.
pub(crate) const Z_TRANSPORT_MAX_FAMILY_REGIMES: usize = 256;

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
    /// The cartesian experiment family is larger than the negative-certificate
    /// budget, or declares more controllables than
    /// [`Z_TRANSPORT_MAX_CONTROLLABLE`].
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
#[allow(
    clippy::float_cmp,
    reason = "Categorical intervention labels are finite exact integers encoded as f64"
)]
#[allow(
    clippy::too_many_lines,
    reason = "one linear identification pipeline whose stages share the prepared graph and workspace"
)]
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
    // Every non-empty subset at every joint level: prod(1 + levels) - 1
    // regimes. The saturating product is computed without enumerating subsets,
    // so a controllable set of any width is sized before a mask is shifted.
    let family_size = domains
        .iter()
        .fold(1usize, |size, (_, levels)| size.saturating_mul(levels.len().saturating_add(1)))
        - 1;
    if family_size > Z_TRANSPORT_MAX_FAMILY_REGIMES || domains.len() > Z_TRANSPORT_MAX_CONTROLLABLE
    {
        return Err(ZExperimentFamilyError::FamilyExceedsBudget { regimes: family_size });
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
                    && regime.supplies_population_law()
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
    let meter = super::SidMeter::Limits(limits);
    decide_z_transport_reporting(diagram, query, catalog, meter, ctx, &mut None)
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
    let meter = super::SidMeter::Limits(limits);
    match decide_z_transport_reporting(diagram, query, catalog, meter, ctx, &mut receipt) {
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

/// Decide a single-source query charging `meter`: its own limits for the
/// public entry points, or the shared budget of an enclosing mz decision.
pub(super) fn decide_z_transport_reporting(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    catalog: &EvidenceCatalog,
    mut meter: super::SidMeter<'_>,
    ctx: &antecedent_core::ExecutionContext,
    receipt: &mut Option<ZTransportLimitsReceipt>,
) -> Result<ZTransportDecision, IdentificationError> {
    catalog.validate().map_err(|error| {
        IdentificationError::invalid_catalog(format!("z_transport.invalid_catalog: {error}"))
    })?;
    match derive_z_transport(diagram, query, meter.reborrow(), ctx, receipt)? {
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
            certify_line11_obstruction(diagram, query, terminal, meter, ctx)
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
#[allow(
    clippy::too_many_lines,
    reason = "Mirrors the single-source, disconnected, and connected branches explicitly"
)]
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
    let components = graph
        .weak_components()
        .into_iter()
        .map(|component| {
            component
                .into_iter()
                .map(|node| graph.static_variable(node))
                .collect::<Option<Vec<_>>>()
        })
        .collect::<Option<Vec<_>>>()?;
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

/// Whether the shared static graph is one connected component under directed,
/// bidirected, and parent adjacency. The disconnected case is decided
/// separately, so the connected route only fires on a single component.
fn graph_is_connected(graph: &antecedent_graph::Admg) -> bool {
    if graph.node_count() == 0
        || graph.nodes().iter().any(|node| !matches!(node, NodeRef::Static(_)))
    {
        return false;
    }
    graph.weak_components().len() == 1
}

/// The `do(treatments)`-mutilated ADMG on the same dense node layout: every
/// directed edge into a treatment and every bidirected edge incident to a
/// treatment is dropped, so the treatments are exogenous roots. m-separation on
/// this graph is m-separation in the intervened law `P*_x(v)`.
fn mutilated_admg(
    graph: &antecedent_graph::Admg,
    treatments: &BitSet,
) -> Result<antecedent_graph::Admg, IdentificationError> {
    let build = || IdentificationError::invalid_input("z_transport.mutilated_graph");
    graph
        .copy_filtered(
            antecedent_graph::CopyLabels::StaticIndex { extra: 0 },
            |_from, to| !treatments.contains(to),
            // do(X) severs latent confounding of X.
            |a, b| !treatments.contains(a) && !treatments.contains(b),
        )
        .map_err(|_| build())
}

/// The `do(treatments)`-mutilated form of a static `graph`: directed edges into
/// a treatment and bidirected edges incident to one are dropped, so the
/// treatments are exogenous roots.
///
/// # Errors
/// A treatment that is not a static node of `graph`.
pub fn intervention_mutilated_admg(
    graph: &antecedent_graph::Admg,
    treatments: &[VariableId],
) -> Result<antecedent_graph::Admg, IdentificationError> {
    let mut set = BitSet::with_len(graph.node_count());
    for treatment in treatments {
        set.insert(graph.dense_of_static(*treatment).ok_or_else(build_unknown)?);
    }
    mutilated_admg(graph, &set)
}

/// The `treatments` that remain directed ancestors of `outcomes` in the
/// `do(treatments)`-mutilated graph `mutilated` of `graph`, sorted by id: the
/// treatments whose level still changes the outcomes' interventional law.
///
/// # Errors
/// An outcome that is not a static node of `graph`.
pub fn intervention_ancestral_treatments(
    graph: &antecedent_graph::Admg,
    mutilated: &antecedent_graph::Admg,
    treatments: &[VariableId],
    outcomes: &[VariableId],
) -> Result<Vec<VariableId>, IdentificationError> {
    let seeds = outcomes
        .iter()
        .map(|variable| graph.dense_of_static(*variable).ok_or_else(build_unknown))
        .collect::<Result<Vec<_>, _>>()?;
    let mut ancestors = BitSet::with_len(mutilated.node_count());
    mutilated.ancestors_of(
        &seeds,
        &mut ancestors,
        &mut antecedent_graph::GraphWorkspace::default(),
    );
    let mut retained = treatments
        .iter()
        .copied()
        .filter(|treatment| {
            graph.dense_of_static(*treatment).is_some_and(|dense| ancestors.contains(dense))
        })
        .collect::<Vec<_>>();
    retained.sort_unstable_by_key(|variable| variable.raw());
    Ok(retained)
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
        let Some(dense) = graph.dense_of_static(treatment) else { return Ok(None) };
        treatments.insert(dense);
    }
    let mutilated = mutilated_admg(graph, &treatments)?;
    let treatment_ids = treatments.to_dense_ids();
    let outcomes = query.outcomes.to_vec();
    let outcome_dense = outcomes
        .iter()
        .map(|variable| graph.dense_of_static(*variable).ok_or_else(build_unknown))
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

/// Whether `current` fixes a treatment that an earlier group already fixed at a
/// different level. Both factors of a product must describe one intervention:
/// a treatment that drives both groups is set to the same level in each source,
/// or the product is not the law of any single target regime.
#[must_use]
pub fn intervention_levels_conflict(
    earlier: &[Vec<CatalogInterventionAssignment>],
    current: &[CatalogInterventionAssignment],
) -> bool {
    earlier.iter().flatten().any(|previous| {
        current.iter().any(|assignment| {
            assignment.variable == previous.variable
                && !antecedent_core::same_intervention_level(&assignment.value, &previous.value)
        })
    })
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
        let Some(dense) = graph.dense_of_static(treatment) else { return Ok(None) };
        treatments.insert(dense);
    }
    let mutilated = mutilated_admg(graph, &treatments)?;
    // Each group's marginal effect P*_x(y_g) equals P*_{x_g}(y_g), where x_g are
    // the treatments that remain ancestors of the group after intervention; the
    // other treatments do not change the group's law.
    let relevant_treatments = |group: &[VariableId]| {
        intervention_ancestral_treatments(graph, &mutilated, &query.treatments, group)
    };
    for order in [[0usize, 1usize], [1usize, 0usize]] {
        let mut proofs = Vec::with_capacity(2);
        let mut group_assignments: Vec<Vec<CatalogInterventionAssignment>> = Vec::with_capacity(2);
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
                || intervention_levels_conflict(&group_assignments, &assignment)
            {
                complete = false;
                break;
            }
            group_assignments.push(assignment.clone());
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
    meter: super::SidMeter<'_>,
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
    verify_z_obstruction_metered(diagram, query, &obstruction, meter, ctx)?;
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

/// [`verify_z_transport_obstruction`] with its replay charged to `meter`.
fn verify_z_obstruction_metered(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    obstruction: &ZTransportObstruction,
    meter: super::SidMeter<'_>,
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
    let replay = z_search(diagram, query, meter, ctx, &mut None)?;
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
        || y.iter().any(|node| !v.contains(node))
        || x.iter().any(|node| !v.contains(node))
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
    Ok(match derive_z_transport(diagram, query, super::SidMeter::Limits(limits), ctx, receipt)? {
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

#[allow(
    clippy::large_enum_variant,
    reason = "Private short-lived state; boxing would add allocation to recursive search"
)]
enum ZDerivation {
    Formula(ZTransportDerivation),
    Line11 { terminal: TrzTerminalFailure, inspection: ZTransportNotCertifiedInspection },
    Unassigned { variable: VariableId, inspection: ZTransportNotCertifiedInspection },
    NotCertified { reason: &'static str, inspection: ZTransportNotCertifiedInspection },
}

fn derive_z_transport(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    meter: super::SidMeter<'_>,
    ctx: &antecedent_core::ExecutionContext,
    receipt: &mut Option<ZTransportLimitsReceipt>,
) -> Result<ZDerivation, IdentificationError> {
    validate_z_transport_query(diagram, query)?;
    let limits = meter.limits();
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
        let (arena, root) = direct_joint_formula(query)?;
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
    let searched = z_search(diagram, query, meter, ctx, receipt)?;
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
    let (arena, root) = surrogate_formula(query, confounder)?;
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

fn direct_joint_formula(
    query: &ZTransportQuery,
) -> Result<(CausalExprArena, ExprId), IdentificationError> {
    let mut arena = CausalExprArena::new();
    let variables = arena.intern_var_set(query.outcomes.iter().copied());
    let conditioned_on = arena.empty_var_set();
    let intervention = crate::intervention_support::intern_assignments(
        &mut arena,
        query.experiment_assignment.iter().map(|a| {
            antecedent_expr::InterventionAssignment::concrete(a.variable, a.value.clone())
        }),
    )?;
    let population = arena.intern_population(Arc::clone(&query.source));
    let root = arena.intern(ExprNode::Distribution {
        variables,
        conditioned_on,
        intervention,
        domain: DomainRef::Interventional,
        population,
        regime: None,
    });
    Ok((arena, root))
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
                z_search(diagram, query, super::SidMeter::Limits(limits), ctx, &mut None)?
                    .identified
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
            let (expected, root) = direct_joint_formula(query)?;
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
            let (expected, root) = surrogate_formula(query, confounder)?;
            if derivation.root != root || derivation.arena != expected {
                return Err(formula_mismatch());
            }
        }
    }
    Ok(())
}

mod search;
pub use search::bind_z_transport_catalog;
#[cfg(test)]
use search::search_trz;
use search::z_search;
pub(crate) use search::{
    TrzCall, TrzDomain, TrzTerminalFailure, bind_recursive_expression, cited_regimes,
    search_trz_call,
};
use search::{
    inspect_z_expression, matches_registered_surrogate_graph, pre_search_budget_kind,
    same_variable_set, surrogate_formula,
};

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::EvidenceKind;
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
    fn family_check_refuses_a_controllable_set_too_wide_to_enumerate() {
        // 64 binary controllables: the family has 3^64 - 1 regimes, far past
        // the budget. It must be refused, never an empty "complete" family.
        let variables = (0..64).map(VariableId::from_raw).collect::<Vec<_>>();
        let coordinates = variables
            .iter()
            .map(|&variable| VariableCoordinate {
                variable,
                domain: VariableDomain::Binary,
                unit: None,
            })
            .collect::<Vec<_>>();
        let catalog = EvidenceCatalog::try_new(
            [
                Environment::try_new("source", coordinates.clone(), []).unwrap(),
                Environment::try_new("target", coordinates, []).unwrap(),
            ],
            [],
            [],
            None,
        )
        .unwrap();
        let diagram =
            SelectionDiagram::try_new(Admg::with_variables(64), Arc::<[VariableId]>::from([]))
                .unwrap();
        let query = ZTransportQuery {
            outcomes: Arc::from([variables[0]]),
            treatments: Arc::from([variables[1]]),
            controllable: variables.clone().into(),
            experiment_assignment: Arc::from([]),
            source: Arc::from("source"),
            target: Arc::from("target"),
        };
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
    #[allow(
        clippy::too_many_lines,
        reason = "one fixture exercised end to end; later assertions read locals built above them"
    )]
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

    /// An available, bound joint regime for the preference tests. `levels`
    /// `None` declares the whole intervention family.
    fn preference_regime(
        raw: u32,
        population: &str,
        interventions: &[u32],
        levels: Option<&[bool]>,
        measured: &[u32],
    ) -> antecedent_core::EvidenceRegime {
        antecedent_core::EvidenceRegime::try_new(
            RegimeId::from_raw(raw),
            if interventions.is_empty() {
                RegimeKind::Observational
            } else {
                RegimeKind::Experimental
            },
            EvidenceKind::Available,
            interventions.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>(),
            levels.map_or_else(Vec::new, |levels| {
                interventions
                    .iter()
                    .zip(levels)
                    .map(|(variable, level)| CatalogInterventionAssignment {
                        variable: VariableId::from_raw(*variable),
                        value: Value::Bool(*level),
                    })
                    .collect()
            }),
            measured.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>(),
            population,
            DistributionAvailability::Joint,
        )
        .unwrap()
    }

    /// A binary source/target catalog over `n` variables binding every regime.
    fn preference_catalog(
        n: u32,
        regimes: Vec<antecedent_core::EvidenceRegime>,
    ) -> EvidenceCatalog {
        let coords = (0..n)
            .map(|raw| VariableCoordinate {
                variable: VariableId::from_raw(raw),
                domain: VariableDomain::Binary,
                unit: None,
            })
            .collect::<Vec<_>>();
        let bindings = regimes
            .iter()
            .map(|regime| RegimeBinding {
                dataset_identity: None,
                regime: regime.id,
                snapshot_identity: Arc::from(format!("snapshot-{}", regime.id.raw())),
                schema_names: Arc::from([]),
                sampling: SamplingDesign::Independent,
                weights: None,
                dependence: DependenceGroup::IndependentStudies,
            })
            .collect::<Vec<_>>();
        EvidenceCatalog::try_new(
            [
                Environment::try_new("source", coords.clone(), Arc::<[VariableId]>::from([]))
                    .unwrap(),
                Environment::try_new("target", coords, Arc::<[VariableId]>::from([])).unwrap(),
            ],
            regimes,
            bindings,
            None,
        )
        .unwrap()
    }

    fn cited(bound: &BoundZTransportFunctional) -> Vec<u32> {
        bound.cited_regimes().iter().map(|regime| regime.raw()).collect()
    }

    /// Z→X→Y with an observed confounder W→X, W→Y: the recursive formula cites
    /// the target's observational joint. Two target regimes that both supply
    /// it bind the lowest id, whatever the catalog order.
    #[test]
    fn recursive_concrete_factor_binds_the_lowest_of_two_supplying_regimes() {
        let mut graph = Admg::with_variables(5);
        for (a, b) in [(1, 2), (2, 3), (0, 2), (0, 3)] {
            graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
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
            panic!("the observed-confounder graph identifies recursively")
        };
        let all = [0, 1, 2, 3, 4];
        for order in [[1, 4], [4, 1]] {
            let catalog = preference_catalog(
                5,
                order
                    .iter()
                    .map(|raw| preference_regime(*raw, "target", &[], None, &all))
                    .collect(),
            );
            let bound = bind_z_transport_catalog(&diagram, &query, &proof, &catalog).unwrap();
            assert_eq!(cited(&bound), [1], "catalog order {order:?}");
        }
    }

    /// The recursive route's concrete source experiment do(X1 = 0): a family
    /// regime on X1 (no declared levels) supplies it too and is preferred over
    /// a concrete regime with a lower id; two concrete regimes bind the lowest.
    #[test]
    fn recursive_concrete_experiment_prefers_a_family_regime_then_the_lowest_id() {
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
        let observational = preference_regime(0, "target", &[], None, &[0, 1, 2, 3, 4]);
        let measured = [0, 2, 3];
        let concrete = |raw| preference_regime(raw, "source", &[1], Some(&[false]), &measured);
        let family = |raw| preference_regime(raw, "source", &[1], None, &measured);
        let catalog = preference_catalog(5, vec![observational.clone(), concrete(1), family(7)]);
        let bound = bind_z_transport_catalog(&diagram, &query, &proof, &catalog).unwrap();
        assert_eq!(cited(&bound), [7]);
        let catalog = preference_catalog(5, vec![observational, concrete(6), concrete(2)]);
        let bound = bind_z_transport_catalog(&diagram, &query, &proof, &catalog).unwrap();
        assert_eq!(cited(&bound), [2]);
    }

    /// Bow graph W→Y, X→Y, X↔Y with selection on W: the exchanged treatment
    /// stays symbolic and binds one regime per world. A world supplied twice
    /// binds the lowest id; a family regime supplies every world and wins.
    #[test]
    fn per_world_binding_prefers_a_family_regime_then_the_lowest_id_per_world() {
        let mut graph = Admg::with_variables(3);
        for (a, b) in [(0, 2), (1, 2)] {
            graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        graph.insert_bidirected(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
        let diagram =
            SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from([VariableId::from_raw(0)]))
                .unwrap();
        let query = ZTransportQuery {
            outcomes: Arc::from([VariableId::from_raw(2)]),
            treatments: Arc::from([VariableId::from_raw(1)]),
            controllable: Arc::from([VariableId::from_raw(1)]),
            experiment_assignment: Arc::from([CatalogInterventionAssignment {
                variable: VariableId::from_raw(1),
                value: Value::Bool(false),
            }]),
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
            panic!("the bow graph identifies")
        };
        let observational = preference_regime(0, "target", &[], None, &[0, 1, 2]);
        let at = |raw, level| preference_regime(raw, "source", &[1], Some(&[level]), &[0, 2]);
        let catalog = preference_catalog(
            3,
            vec![observational.clone(), at(5, false), at(2, true), at(3, false)],
        );
        let bound = bind_z_transport_catalog(&diagram, &query, &proof, &catalog).unwrap();
        assert_eq!(cited(&bound), [0, 2, 3]);
        let family = preference_regime(9, "source", &[1], None, &[0, 2]);
        let catalog = preference_catalog(3, vec![observational, at(1, false), at(2, true), family]);
        let bound = bind_z_transport_catalog(&diagram, &query, &proof, &catalog).unwrap();
        assert_eq!(cited(&bound), [0, 9]);
    }

    /// The registered surrogate formula binds one source joint; two supplying
    /// experiments bind the lowest id, whatever the catalog order.
    #[test]
    fn surrogate_formula_binds_the_lowest_of_two_supplying_experiments() {
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
        assert_eq!(proof.kind, ZFormulaKind::Surrogate);
        let experiment = |raw| preference_regime(raw, "source", &[1], Some(&[false]), &[0, 2, 3]);
        for order in [[3, 1], [1, 3]] {
            let catalog = preference_catalog(4, order.iter().map(|raw| experiment(*raw)).collect());
            let bound = bind_z_transport_catalog(&diagram, &query, &proof, &catalog).unwrap();
            assert_eq!(cited(&bound), [1], "catalog order {order:?}");
        }
    }

    /// The direct joint exchange binds one source joint; two supplying
    /// experiments bind the lowest id, whatever the catalog order.
    #[test]
    fn direct_joint_formula_binds_the_lowest_of_two_supplying_experiments() {
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
        assert_eq!(proof.kind, ZFormulaKind::DirectJoint);
        let experiment =
            |raw| preference_regime(raw, "source", &[0, 1], Some(&[true, false]), &[2]);
        for order in [[4, 2], [2, 4]] {
            let catalog = preference_catalog(3, order.iter().map(|raw| experiment(*raw)).collect());
            let bound = bind_z_transport_catalog(&diagram, &query, &proof, &catalog).unwrap();
            assert_eq!(cited(&bound), [2], "catalog order {order:?}");
        }
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

    /// Dense slot of `variable` in the `Admg::with_variables(4)` fixture this SCM evaluates.
    fn exchange_slot(variable: VariableId) -> usize {
        Admg::with_variables(4)
            .dense_of_static(variable)
            .unwrap_or_else(|| panic!("sequential exchange fixture has no node {}", variable.raw()))
            .as_usize()
    }

    struct SequentialExchangeScm;

    #[allow(clippy::float_cmp, reason = "The provider accepts only binary values 0.0 and 1.0")]
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
                intervention[exchange_slot(item.variable)] = Some(usize::from(value == 1.0));
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
                            usize::from(value == 1.0) == state[exchange_slot(*variable)]
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
    #[allow(
        clippy::too_many_lines,
        reason = "one fixture exercised end to end; later assertions read locals built above them"
    )]
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
