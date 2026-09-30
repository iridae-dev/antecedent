//! Exact binary causal observation recovery (2.2B X10): graph-licensed recovery
//! of the full law of partially observed binary variables, not MAR/IPCW.
//!
//! # Model class
//!
//! An m-graph is a DAG over four roles: fully observed binary variables `O`,
//! partially observed binary variables `X(1)`, one response indicator `R_i` per
//! partially observed variable and one proxy `X*_i` with domain `{0, 1, ?}`,
//! deterministic: `X*_i = X_i` when `R_i = 1`, `?` when `R_i = 0`. The bounded
//! class this module decides is: response indicators whose parents are
//! substantive variables (`X`, `O`) only, no `R -> R` edge, proxies with exactly
//! the parents `X_i`, `R_i` and no children, substantive variables with
//! substantive parents only, at most three partially observed and two fully
//! observed variables. Selection nodes (a selected-sample law), bidirected edges,
//! non-binary variables, `R -> R` edges and noisy proxies are refused as
//! unsupported mechanisms, never decided.
//!
//! # The decision and why it is complete in the class
//!
//! The target is the full law `P(X(1), O)`, recovered from one named observed
//! pattern law `P(R, X*, O)`. Within the class:
//!
//! - **Sufficiency (derived here).** The m-graph factorizes
//!   `p(x, o, r) = p(x, o) prod_i p(r_i | pa(R_i))` because response indicators
//!   have substantive parents only and no edges between them. Hence
//!   `P(R = 1, X* = x, O = o) = p(x, o) prod_i p(R_i = 1 | pa(R_i))`. When no
//!   `X_i -> R_i` edge exists, each propensity is observed: with `S` the partially
//!   observed parents of `R_i`, the local Markov property gives
//!   `R_i _||_ R_S | pa(R_i)` (every `R_j` is a non-descendant of `R_i`, whose only
//!   child is its proxy), and on `R_S = 1` the proxies equal the variables, so
//!   `p(R_i = 1 | pa(R_i)) = P(R_i = 1, R_S = 1, X*_S = x_S, O_pa)
//!   / P(R_S = 1, X*_S = x_S, O_pa)`, a ratio of two observed pattern margins.
//!   The recovered law is the complete-case cell divided by that product.
//!   Positivity (every complete-case cell has mass) makes every ratio defined.
//! - **Necessity (verified here).** When some `X_i -> R_i` edge exists, two
//!   explicit positive models Markov to the m-graph agree on the whole observed
//!   law and differ on `P(X_i)`: every mechanism is a fair coin except
//!   `X_i ~ Bern(1/2), P(R_i = 1 | X_i) = (2/5, 3/5)` against
//!   `X_i ~ Bern(2/5), P(R_i = 1 | X_i) = (1/3, 3/4)`. Both give
//!   `P(R_i = 1, X_i = 1) = 3/10` and `P(R_i = 1, X_i = 0) = 1/5`. The witness is
//!   verified by exact integer enumeration of both joints (every mechanism
//!   probability is `k / 60`), so it is self-certifying and no theorem is trusted.
//!
//! **What "nonrecoverable" means.** The witness shows the target is not
//! recoverable *for every model Markov to the m-graph*: no function of the
//! observed law returns `P(X(1), O)` on all of them. The witness models are
//! degenerate (every other mechanism is an independent fair coin, so they are
//! not faithful to the m-graph); it says nothing about faithful or generic
//! parameters. Under such restrictions some self-censoring graphs are
//! generically identified, for example a shadow-variable graph
//! `Z -> X -> R_X` with `Z` not a parent of `R_X`. This route makes no
//! faithfulness or generic-parameter assumption, so it refuses there.
//!
//! So in this bounded class the graphical criterion "no self-censoring edge" is
//! decided exactly (over all models Markov to the m-graph), by in-repo proof. Nabi, Bhattacharya and Shpitser (ICML
//! 2020) state necessity and sufficiency of "no self-censoring and no colluder"
//! for the full law including the response mechanism in general missing-data
//! DAGs; that general statement is paper-inherited (only the abstract was read).
//! A colluder `X_j -> R_i <- R_j` needs an `R -> R` edge, which this class
//! refuses; for the target `P(X(1), O)` alone the colluder archetype is not an
//! obstruction (the target is recoverable there), so the refusal is
//! "unsupported mechanism", never "nonrecoverable".
//!
//! # What is not done
//!
//! No automatic schema alignment (proxies are declared per variable and must be
//! wired to it), no complete-case fallback, no MAR or inverse-probability
//! weighting substitution: the assumption lives in the graph and is checked. A
//! recovered law is a derived law with [`LawOrigin::Recovered`] provenance and
//! is fed to the ordinary target ID stage of sID only after its population,
//! variables, factorization and support are matched.
//!
//! Every stage of a decision (class check, formula lowering and its independent
//! check, witness enumeration, downstream identification) charges one
//! [`SearchBudget`]; a stop is a receipt naming the stages completed
//! (`explored`) and the stage it stopped in followed by the ones left
//! (`unevaluated`), never a verdict. The standalone [`verify_recovery_witness`]
//! is not charged: the class bounds cap it at eight non-proxy nodes, so it
//! enumerates at most `2 * 2^8` configurations.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use antecedent_core::{
    CatalogDistribution, DistributionAvailability, EvidenceCatalog, EvidenceProjection,
    ExecutionContext, LawOrigin, RegimeId, RegimeKind, SamplingSelection, SearchBudget,
    SearchLimits, SearchReceipt, SearchStop, VariableDomain, VariableId,
};
use antecedent_expr::{CausalExprArena, DomainRef, ExprId, ExprNode};
use antecedent_graph::{Admg, DSeparationWorkspace, DenseNodeId, NodeRef, SelectionDiagram};

use crate::sid::{
    ClassicalTransportQuery, ClassicalTransportResult, SearchCharge, SharedSearch,
    identify_classical_transport_metered,
};

/// Most partially observed variables one query may declare.
pub const RECOVERY_MAX_PARTIALLY_OBSERVED: usize = 3;
/// Most fully observed variables one query may declare.
pub const RECOVERY_MAX_FULLY_OBSERVED: usize = 2;
/// Most cells the observed pattern law may carry (`6^3 * 2^2`).
pub const RECOVERY_MAX_OBSERVED_CELLS: usize = 864;
/// Default search limits of one decision; they are also the maxima.
pub const RECOVERY_DEFAULT_LIMITS: RecoveryLimits = RecoveryLimits {
    search: SearchLimits { operations: 50_000, depth: 32 },
    memory_bytes: antecedent_core::DEFAULT_SEARCH_MEMORY_BYTES,
};
/// Version of the recovery rule, bound into every derivation.
pub const RECOVERY_RULE_VERSION: &str = "x10.recovery.v1";
/// Every witness mechanism probability is `k / WITNESS_SCALE`.
pub const WITNESS_SCALE: u32 = 60;

/// The label of the missing level of a proxy.
pub const MISSING_LEVEL: &str = "?";

/// Why a recovery request was refused. The detail and reason code are fixed per kind.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RecoveryDetail {
    /// A self-censoring edge with a verified witness: not recoverable for every
    /// model Markov to the m-graph.
    NonrecoverableWitness,
    /// A mechanism outside the bounded class.
    UnsupportedMechanism,
    /// The named observed pattern law does not supply a required margin.
    MissingMargin,
    /// The shared budget stopped (operations, depth, memory or cancellation).
    Budget,
    /// A violation without a verified witness: nothing is claimed.
    WitnessUnavailable,
    /// A malformed query or role declaration.
    InvalidQuery,
    /// A stored derivation, margin binding, witness or artifact does not verify.
    InvalidDerivation,
    /// A declared bound is exceeded.
    BoundsExceeded,
    /// A complete-case cell of the observed law has no mass.
    Positivity,
    /// The exact table disagrees with its catalog distribution or the proxy model.
    InvalidObservedLaw,
    /// The downstream effect does not match the recovered law.
    HandoffMismatch,
    /// A sampled or empirical provider was offered.
    EmpiricalNotLicensed,
}

impl RecoveryDetail {
    /// Stable namespaced detail.
    #[must_use]
    pub const fn detail(self) -> &'static str {
        match self {
            Self::NonrecoverableWitness => "recovery.nonrecoverable_witness",
            Self::UnsupportedMechanism => "recovery.unsupported_mechanism",
            Self::MissingMargin => "recovery.missing_margin",
            Self::Budget => "recovery.budget",
            Self::WitnessUnavailable => "recovery.witness_unavailable",
            Self::InvalidQuery => "recovery.invalid_query",
            Self::InvalidDerivation => "recovery.invalid_derivation",
            Self::BoundsExceeded => "recovery.bounds_exceeded",
            Self::Positivity => "recovery.positivity",
            Self::InvalidObservedLaw => "recovery.invalid_observed_law",
            Self::HandoffMismatch => "recovery.handoff_mismatch",
            Self::EmpiricalNotLicensed => "recovery.empirical_not_licensed",
        }
    }

    /// Registered top-level reason code.
    #[must_use]
    pub const fn reason_code(self) -> &'static str {
        match self {
            Self::NonrecoverableWitness => {
                antecedent_core::reason_code!("transport_proven_non_transportable")
            }
            Self::UnsupportedMechanism | Self::BoundsExceeded => {
                antecedent_core::reason_code!("route_not_supported")
            }
            Self::MissingMargin => antecedent_core::reason_code!("transport_missing_evidence"),
            Self::Budget => antecedent_core::reason_code!("transport_budget_cancel"),
            Self::WitnessUnavailable | Self::InvalidDerivation => {
                antecedent_core::reason_code!("transport_not_certified")
            }
            Self::InvalidQuery | Self::InvalidObservedLaw | Self::HandoffMismatch => {
                antecedent_core::reason_code!("invalid_argument")
            }
            Self::Positivity => antecedent_core::reason_code!("transport_support_failure"),
            Self::EmpiricalNotLicensed => antecedent_core::reason_code!("cell_not_licensed"),
        }
    }
}

/// A typed recovery refusal: detail, reason code, explanation and, for a budget
/// stop, the receipt.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecoveryError {
    /// Why.
    pub detail: RecoveryDetail,
    /// Human explanation.
    pub message: String,
    /// The receipt of a budget stop.
    pub receipt: Option<Box<SearchReceipt>>,
}

impl RecoveryError {
    /// A refusal of `detail` with an explanation.
    #[must_use]
    pub fn new(detail: RecoveryDetail, message: impl Into<String>) -> Self {
        Self { detail, message: message.into(), receipt: None }
    }

    /// Registered reason code.
    #[must_use]
    pub const fn reason_code(&self) -> &'static str {
        self.detail.reason_code()
    }
}

impl fmt::Display for RecoveryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.detail.detail(), self.message)
    }
}

impl std::error::Error for RecoveryError {}

fn refuse(detail: RecoveryDetail, message: impl Into<String>) -> RecoveryError {
    RecoveryError::new(detail, message)
}

/// One partially observed variable with its response indicator and proxy.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct PartiallyObserved {
    /// The substantive variable `X_i`.
    pub variable: VariableId,
    /// Its response indicator `R_i` (1 observed, 0 missing).
    pub response: VariableId,
    /// Its proxy `X*_i` with domain `{0, 1, ?}`.
    pub proxy: VariableId,
}

/// Recovery of `P(X(1), O)` from one named observed pattern law.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObservationRecoveryQuery {
    /// Population the observed pattern law describes.
    pub population: Arc<str>,
    /// Catalog regime of the observed pattern law.
    pub observed_regime: RegimeId,
    /// Partially observed variables with their roles.
    pub partially_observed: Arc<[PartiallyObserved]>,
    /// Fully observed variables.
    pub fully_observed: Arc<[VariableId]>,
}

impl ObservationRecoveryQuery {
    /// The query with its role lists in canonical (variable id) order: the
    /// decision depends on the roles, never on their declaration order.
    #[must_use]
    pub fn canonical(&self) -> Self {
        let mut partially = self.partially_observed.to_vec();
        partially.sort_by_key(|p| p.variable);
        let mut fully = self.fully_observed.to_vec();
        fully.sort_unstable();
        Self {
            population: Arc::clone(&self.population),
            observed_regime: self.observed_regime,
            partially_observed: partially.into(),
            fully_observed: fully.into(),
        }
    }

    /// Substantive variables `X ∪ O`, sorted.
    #[must_use]
    pub fn substantive(&self) -> Vec<VariableId> {
        let mut out: Vec<_> = self
            .partially_observed
            .iter()
            .map(|p| p.variable)
            .chain(self.fully_observed.iter().copied())
            .collect();
        out.sort_unstable();
        out
    }

    /// Observed coordinates `R ∪ X* ∪ O`, sorted.
    #[must_use]
    pub fn observed(&self) -> Vec<VariableId> {
        let mut out: Vec<_> = self
            .partially_observed
            .iter()
            .flat_map(|p| [p.response, p.proxy])
            .chain(self.fully_observed.iter().copied())
            .collect();
        out.sort_unstable();
        out
    }
}

/// A downstream causal effect `P(outcomes | do(treatments))` to identify from
/// the recovered law with the ordinary target ID stage of sID.
#[derive(Clone, Debug)]
pub struct RecoveredEffectQuery {
    /// Causal graph over exactly the substantive variables `X ∪ O`; it must equal
    /// the m-graph's restriction to them.
    pub graph: Admg,
    /// Outcomes.
    pub outcomes: Arc<[VariableId]>,
    /// Treatments.
    pub treatments: Arc<[VariableId]>,
}

/// Search limits and memory cap of one decision.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RecoveryLimits {
    /// Operation and depth limits.
    pub search: SearchLimits,
    /// Requested memory cap in bytes (the effective cap is also bounded by the
    /// context's hard limit).
    pub memory_bytes: u64,
}

impl Default for RecoveryLimits {
    fn default() -> Self {
        RECOVERY_DEFAULT_LIMITS
    }
}

/// One propensity factor of the recovery formula.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveryFactorRecord {
    /// Response indicator `R_i`.
    pub response: u32,
    /// Its graph parents, sorted.
    pub parents: Vec<u32>,
    /// Conditioning coordinates `R_S ∪ X*_S ∪ O_pa`, sorted.
    pub conditioning: Vec<u32>,
    /// Expression of the factor.
    pub expression: u32,
}

/// One leaf of the formula bound to a margin of the named catalog distribution.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveryMarginRecord {
    /// Leaf expression.
    pub expression: u32,
    /// Variables of the margin, sorted.
    pub variables: Vec<u32>,
    /// Canonical identity of the projected catalog distribution.
    pub identity: String,
}

/// The downstream effect derivation.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveredEffectRecord {
    /// Outcomes, sorted.
    pub outcomes: Vec<u32>,
    /// Treatments, sorted.
    pub treatments: Vec<u32>,
    /// Directed edges of the causal graph, sorted.
    pub edges: Vec<(u32, u32)>,
    /// Root of the identified expression.
    pub root: u32,
    /// sID branches the derivation used.
    pub rules: Vec<String>,
}

/// What the one shared budget consumed. Limits are the requested ones.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveryReceiptRecord {
    /// Operation limit.
    pub operations_limit: usize,
    /// Depth limit.
    pub depth_limit: usize,
    /// Requested memory cap.
    pub memory_bytes: u64,
    /// Operations charged across every stage.
    pub operations_consumed: usize,
    /// Deepest level charged.
    pub depth_reached: usize,
}

/// Portable record of a checked recovery derivation. A consumer re-decides from
/// its inputs and accepts only an identical record.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveryDerivationRecord {
    /// [`RECOVERY_RULE_VERSION`].
    pub rule_version: String,
    /// m-graph identity.
    pub graph_signature: String,
    /// Population.
    pub population: String,
    /// Observed regime.
    pub observed_regime: u32,
    /// Canonical identity of the observed pattern law's catalog distribution.
    pub observed_identity: String,
    /// `(variable, response, proxy)` in canonical order.
    pub partially_observed: Vec<(u32, u32, u32)>,
    /// Fully observed, sorted.
    pub fully_observed: Vec<u32>,
    /// Checked graphical premises, in order.
    pub premises: Vec<String>,
    /// Numerator leaf (the complete-case cell).
    pub numerator: u32,
    /// Propensity factors in canonical order.
    pub factors: Vec<RecoveryFactorRecord>,
    /// Root of the recovery formula.
    pub root: u32,
    /// Every leaf with its bound margin.
    pub margins: Vec<RecoveryMarginRecord>,
    /// Downstream effect, when requested.
    pub effect: Option<RecoveredEffectRecord>,
    /// Budget consumption.
    pub receipt: RecoveryReceiptRecord,
}

/// A checked recovery derivation. Fields are private execution authority.
#[derive(Clone, Debug)]
pub struct RecoveryDerivation {
    query: ObservationRecoveryQuery,
    record: RecoveryDerivationRecord,
    arena: CausalExprArena,
    root: ExprId,
    observed: CatalogDistribution,
    effect: Option<RecoveredEffect>,
}

/// The downstream effect identified from the recovered law.
#[derive(Clone, Debug)]
pub struct RecoveredEffect {
    arena: CausalExprArena,
    root: ExprId,
    outcomes: Arc<[VariableId]>,
    treatments: Arc<[VariableId]>,
}

impl RecoveredEffect {
    /// Identified expression arena (target-population observational leaves).
    #[must_use]
    pub const fn arena(&self) -> &CausalExprArena {
        &self.arena
    }
    /// Distribution root.
    #[must_use]
    pub const fn root(&self) -> ExprId {
        self.root
    }
    /// Outcomes.
    #[must_use]
    pub fn outcomes(&self) -> &[VariableId] {
        &self.outcomes
    }
    /// Treatments.
    #[must_use]
    pub fn treatments(&self) -> &[VariableId] {
        &self.treatments
    }
}

impl RecoveryDerivation {
    /// Canonical query.
    #[must_use]
    pub const fn query(&self) -> &ObservationRecoveryQuery {
        &self.query
    }
    /// Portable record.
    #[must_use]
    pub const fn record(&self) -> &RecoveryDerivationRecord {
        &self.record
    }
    /// Formula arena.
    #[must_use]
    pub const fn arena(&self) -> &CausalExprArena {
        &self.arena
    }
    /// Formula root: the recovered cell `P(x, o)` as a function of the observed
    /// coordinates at `R = 1, X* = x, O = o`.
    #[must_use]
    pub const fn root(&self) -> ExprId {
        self.root
    }
    /// The named observed pattern law's catalog distribution.
    #[must_use]
    pub const fn observed(&self) -> &CatalogDistribution {
        &self.observed
    }
    /// The downstream effect, when one was requested.
    #[must_use]
    pub const fn effect(&self) -> Option<&RecoveredEffect> {
        self.effect.as_ref()
    }
    /// Canonical text identity of this derivation (rule, roles, graph, observed
    /// distribution, formula, margins and effect). A recovered law carries it as
    /// its [`LawOrigin::Recovered`] provenance.
    #[must_use]
    pub fn identity(&self) -> String {
        derivation_identity(&self.record)
    }
    /// The catalog descriptor of the recovered law: a derived observational joint
    /// law of the same population over `X ∪ O` with [`LawOrigin::Recovered`]
    /// provenance, traced to the observed pattern law's regime.
    #[must_use]
    pub fn recovered_descriptor(&self) -> CatalogDistribution {
        let mut descriptor = self.observed.clone();
        descriptor.origin = LawOrigin::Recovered { derivation: Arc::from(self.identity()) };
        descriptor.measured = self.query.substantive().into();
        descriptor.availability = DistributionAvailability::Joint;
        descriptor.conditioned_on = Arc::from([]);
        descriptor.projections = Arc::from([]);
        descriptor
    }
}

fn derivation_identity(record: &RecoveryDerivationRecord) -> String {
    // Every field except the budget accounting: the provenance names what was
    // derived, not how much work the producer's budget recorded.
    let factors = record
        .factors
        .iter()
        .map(|f| format!("{}<{:?}|{:?}@{}", f.response, f.parents, f.conditioning, f.expression))
        .collect::<Vec<_>>()
        .join(";");
    let margins = record
        .margins
        .iter()
        .map(|m| format!("{}:{:?}:{}", m.expression, m.variables, m.identity))
        .collect::<Vec<_>>()
        .join(";");
    let effect = record.effect.as_ref().map_or_else(String::new, |e| {
        format!("{:?}|{:?}|{:?}|{}|{:?}", e.outcomes, e.treatments, e.edges, e.root, e.rules)
    });
    format!(
        "{}|graph={}|population={}|regime={}|observed={}|roles={:?}|fully={:?}|premises={:?}|numerator={}|factors={factors}|root={}|margins={margins}|effect={effect}",
        record.rule_version,
        record.graph_signature,
        record.population,
        record.observed_regime,
        record.observed_identity,
        record.partially_observed,
        record.fully_observed,
        record.premises,
        record.numerator,
        record.root,
    )
}

/// The outcome of a decision.
#[derive(Clone, Debug)]
pub enum RecoveryDecision {
    /// A checked derivation (and downstream effect, when requested).
    Recovered(Box<RecoveryDerivation>),
    /// A self-censoring edge with a verified witness: not recoverable for every
    /// model Markov to the m-graph (the witness is degenerate; nothing is claimed
    /// under faithful or generic parameters).
    NonRecoverable(Box<RecoveryWitness>),
}

/// One mechanism of a witness model: `P(node = 1 | parents) = numerator / 60`,
/// indexed by parent configuration (first parent most significant).
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WitnessMechanism {
    /// Node.
    pub node: u32,
    /// Its graph parents, in graph order.
    pub parents: Vec<u32>,
    /// `P(node = 1 | configuration) * 60` per parent configuration.
    pub numerators: Vec<u32>,
}

/// Two explicit models that agree on the observed law and differ on the target.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveryWitness {
    /// The self-censoring edge `(X_i, R_i)`.
    pub edge: (u32, u32),
    /// Mechanisms of the first model (every non-proxy node).
    pub first: Vec<WitnessMechanism>,
    /// Mechanisms of the second model.
    pub second: Vec<WitnessMechanism>,
}

/// What a verified witness shows, in exact integer masses over `60^n`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WitnessCheck {
    /// Number of observed-law cells compared (all equal).
    pub observed_cells: usize,
    /// A target cell `(variable, level)` where the two models differ.
    pub differing_cell: Vec<(VariableId, u8)>,
    /// The two target masses at that cell, over [`Self::denominator`].
    pub masses: (u128, u128),
    /// Common denominator `60^n`.
    pub denominator: u128,
}

/// Role of one m-graph node.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Role {
    Partial(usize),
    Response(usize),
    Proxy(usize),
    Full,
}

/// The m-graph with its roles resolved.
struct MGraph<'g> {
    graph: &'g Admg,
    dense: BTreeMap<VariableId, DenseNodeId>,
    roles: BTreeMap<VariableId, Role>,
}

impl MGraph<'_> {
    fn parents(&self, v: VariableId) -> Vec<VariableId> {
        let mut out: Vec<_> =
            self.graph.parents(self.dense[&v]).iter().map(|p| var(self.graph, *p)).collect();
        out.sort_unstable();
        out
    }
    fn children(&self, v: VariableId) -> Vec<VariableId> {
        let mut out: Vec<_> =
            self.graph.children(self.dense[&v]).iter().map(|c| var(self.graph, *c)).collect();
        out.sort_unstable();
        out
    }
}

fn var(graph: &Admg, node: DenseNodeId) -> VariableId {
    match graph.nodes()[node.as_usize()] {
        NodeRef::Static(v) => v,
        // Non-static nodes are refused before any lookup.
        _ => VariableId::from_raw(u32::MAX),
    }
}

fn ids(raw: &[VariableId]) -> Vec<u32> {
    raw.iter().map(|v| v.raw()).collect()
}

/// Resolve the roles of every node: bounds, role shape and proxy wiring.
fn resolve_roles<'g>(
    graph: &'g Admg,
    query: &ObservationRecoveryQuery,
) -> Result<MGraph<'g>, RecoveryError> {
    use RecoveryDetail::{BoundsExceeded, InvalidQuery};
    let k = query.partially_observed.len();
    if k > RECOVERY_MAX_PARTIALLY_OBSERVED {
        return Err(refuse(
            BoundsExceeded,
            format!("{k} partially observed variables; at most {RECOVERY_MAX_PARTIALLY_OBSERVED}"),
        ));
    }
    if query.fully_observed.len() > RECOVERY_MAX_FULLY_OBSERVED {
        return Err(refuse(
            BoundsExceeded,
            format!(
                "{} fully observed variables; at most {RECOVERY_MAX_FULLY_OBSERVED}",
                query.fully_observed.len()
            ),
        ));
    }
    if query.population.trim().is_empty() {
        return Err(refuse(InvalidQuery, "the population must be named"));
    }
    if k == 0 {
        return Err(refuse(InvalidQuery, "declare at least one partially observed variable"));
    }
    let mut dense = BTreeMap::new();
    for (i, node) in graph.nodes().iter().enumerate() {
        let NodeRef::Static(v) = node else {
            return Err(refuse(InvalidQuery, "the m-graph must have static nodes only"));
        };
        let id = DenseNodeId::from_raw(
            u32::try_from(i).map_err(|_| refuse(InvalidQuery, "graph too large"))?,
        );
        if dense.insert(*v, id).is_some() {
            return Err(refuse(InvalidQuery, "duplicate m-graph node"));
        }
    }
    let mut roles = BTreeMap::new();
    let mut assign = |v: VariableId, role: Role| -> Result<(), RecoveryError> {
        if !dense.contains_key(&v) {
            return Err(refuse(InvalidQuery, format!("{v:?} is not an m-graph node")));
        }
        if roles.insert(v, role).is_some() {
            return Err(refuse(InvalidQuery, format!("{v:?} is declared in two roles")));
        }
        Ok(())
    };
    for (i, p) in query.partially_observed.iter().enumerate() {
        assign(p.variable, Role::Partial(i))?;
        assign(p.response, Role::Response(i))?;
        assign(p.proxy, Role::Proxy(i))?;
    }
    for v in query.fully_observed.iter() {
        assign(*v, Role::Full)?;
    }
    if roles.len() != dense.len() {
        return Err(refuse(InvalidQuery, "every m-graph node must have exactly one declared role"));
    }
    Ok(MGraph { graph, dense, roles })
}

/// Refuse every mechanism outside the bounded class, charging one operation per
/// node inspected. Returns the self-censoring edges found.
#[allow(clippy::too_many_lines)] // One linear pass per stage; splitting it would scatter the checks.
fn check_class(
    m: &MGraph<'_>,
    query: &ObservationRecoveryQuery,
    catalog: &EvidenceCatalog,
    search: &mut SharedSearch<'_>,
    at: At,
    bytes: u64,
) -> Result<Vec<(VariableId, VariableId)>, RecoveryError> {
    use RecoveryDetail::{InvalidQuery, UnsupportedMechanism};
    if m.graph.has_bidirected() {
        return Err(refuse(
            UnsupportedMechanism,
            "bidirected edges (unmeasured confounding) are outside the recovery class",
        ));
    }
    let mut self_censoring = Vec::new();
    for (v, role) in &m.roles {
        charge(search, at, 1, bytes)?;
        let parents = m.parents(*v);
        let children = m.children(*v);
        match *role {
            Role::Proxy(i) => {
                let own = query.partially_observed[i];
                if !children.is_empty() {
                    return Err(refuse(UnsupportedMechanism, "a proxy may have no children"));
                }
                let foreign = parents
                    .iter()
                    .any(|p| matches!(m.roles[p], Role::Partial(j) | Role::Response(j) if j != i));
                if foreign || !parents.contains(&own.variable) || !parents.contains(&own.response) {
                    return Err(refuse(
                        InvalidQuery,
                        "a proxy must be wired to exactly its declared variable and response \
                         (relabelled proxies are refused, never aligned)",
                    ));
                }
                if parents.len() != 2 {
                    return Err(refuse(
                        UnsupportedMechanism,
                        "a proxy with parents other than its variable and response (proxy noise)",
                    ));
                }
            }
            Role::Response(i) => {
                if children.iter().chain(&parents).any(|c| matches!(m.roles[c], Role::Response(_)))
                {
                    return Err(refuse(
                        UnsupportedMechanism,
                        "R -> R edges are outside the first recovery cut",
                    ));
                }
                if children.iter().any(|c| matches!(m.roles[c], Role::Proxy(j) if j != i)) {
                    return Err(refuse(
                        InvalidQuery,
                        "a response indicator drives another variable's proxy (relabelled \
                         responses are refused, never aligned)",
                    ));
                }
                if children != [query.partially_observed[i].proxy] {
                    return Err(refuse(
                        UnsupportedMechanism,
                        "a response indicator may have only its own proxy as a child",
                    ));
                }
                if parents.iter().any(|p| matches!(m.roles[p], Role::Proxy(_))) {
                    return Err(refuse(UnsupportedMechanism, "a proxy may have no children"));
                }
                if parents.contains(&query.partially_observed[i].variable) {
                    self_censoring.push((query.partially_observed[i].variable, *v));
                }
            }
            Role::Partial(_) | Role::Full => {
                if parents.iter().any(|p| matches!(m.roles[p], Role::Response(_) | Role::Proxy(_)))
                {
                    return Err(refuse(
                        UnsupportedMechanism,
                        "a response indicator or proxy may not cause a substantive variable",
                    ));
                }
            }
        }
    }
    for environment in catalog.environments.iter().filter(|e| e.identity == query.population) {
        charge(search, at, 1, bytes)?;
        if !environment.selection_targets.is_empty() {
            return Err(refuse(
                UnsupportedMechanism,
                "selection nodes are outside the recovery class",
            ));
        }
        for coordinate in environment.variables.iter() {
            let Some(role) = m.roles.get(&coordinate.variable) else { continue };
            let binary =
                matches!(coordinate.domain, VariableDomain::Unspecified | VariableDomain::Binary)
                    || coordinate.domain == (VariableDomain::Categorical { cardinality: 2 });
            let proxy_domain = matches!(coordinate.domain, VariableDomain::Unspecified)
                || coordinate.domain == (VariableDomain::Categorical { cardinality: 3 });
            let fine = if matches!(role, Role::Proxy(_)) { proxy_domain } else { binary };
            if !fine {
                return Err(refuse(
                    UnsupportedMechanism,
                    format!(
                        "{:?} is declared non-binary; only binary variables are recovered",
                        coordinate.variable
                    ),
                ));
            }
        }
    }
    Ok(self_censoring)
}

/// The named observed pattern law's catalog distribution, or the missing margin.
fn observed_distribution(
    query: &ObservationRecoveryQuery,
    catalog: &EvidenceCatalog,
) -> Result<CatalogDistribution, RecoveryError> {
    use RecoveryDetail::{InvalidQuery, MissingMargin, UnsupportedMechanism};
    catalog.validate().map_err(|e| refuse(InvalidQuery, format!("invalid catalog: {e}")))?;
    let Some(d) = catalog.distribution(query.observed_regime) else {
        return Err(refuse(MissingMargin, "the named observed regime is not in the catalog"));
    };
    if d.population != query.population {
        return Err(refuse(
            MissingMargin,
            "the named observed regime describes another population",
        ));
    }
    if d.selection != SamplingSelection::Population {
        return Err(refuse(
            UnsupportedMechanism,
            "a selected-sample law (selection node) is outside the recovery class",
        ));
    }
    if !d.evidence_kind.can_satisfy_factor() || d.origin != LawOrigin::Measured {
        return Err(refuse(
            MissingMargin,
            "the named regime is not an available measured law (a model artifact or a recovered law is not observed)",
        ));
    }
    if d.kind != RegimeKind::Observational || !d.conditioned_on.is_empty() {
        return Err(refuse(
            MissingMargin,
            "the observed pattern law must be an unconditioned observational law",
        ));
    }
    if !d.is_joint() {
        return Err(refuse(
            MissingMargin,
            "the observed pattern law is supplied only as separate marginals; a joint over R, X* and O is required",
        ));
    }
    let measured: BTreeSet<_> = d.measured.iter().copied().collect();
    let needed: BTreeSet<_> = query.observed().into_iter().collect();
    if let Some(missing) = needed.difference(&measured).next() {
        return Err(refuse(
            MissingMargin,
            format!("the observed pattern law does not measure {missing:?}"),
        ));
    }
    if measured != needed {
        return Err(refuse(
            InvalidQuery,
            "the observed pattern law measures a coordinate outside R, X* and O (a partially observed variable is never observed directly)",
        ));
    }
    Ok(d)
}

/// The stage of one decision a charge belongs to.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Stage {
    /// Role, class and observed-law checks.
    ClassCheck,
    /// Formula lowering and its independent check.
    Formula,
    /// Witness construction and exact enumeration.
    Witness,
    /// Downstream target ID of the effect.
    Effect,
}

/// Where a decision is: the stage being charged and whether an effect was requested.
#[derive(Clone, Copy, Debug)]
struct At {
    stage: Stage,
    effect: bool,
}

impl At {
    const fn with(self, stage: Stage) -> Self {
        Self { stage, effect: self.effect }
    }

    /// `(explored, unevaluated)` stage names of a stop in this stage: explored
    /// lists the stages completed before the stop; unevaluated lists the stopped
    /// stage, then every stage that would have followed. A stop in the class check
    /// does not yet know whether the formula or the witness follows.
    fn regions(self) -> (Vec<String>, Vec<String>) {
        let effect = self.effect.then_some("downstream_effect");
        let (explored, unevaluated): (&[&str], Vec<&str>) = match self.stage {
            Stage::ClassCheck => {
                (&[], [Some("class_check"), Some("formula_or_witness"), effect].into_iter().flatten().collect())
            }
            Stage::Formula => (&["class_check"], [Some("formula"), effect].into_iter().flatten().collect()),
            // A verified witness ends the decision: no effect stage follows.
            Stage::Witness => (&["class_check"], vec!["witness"]),
            Stage::Effect => (&["class_check", "formula"], vec!["downstream_effect"]),
        };
        (
            explored.iter().map(|s| (*s).to_owned()).collect(),
            unevaluated.into_iter().map(str::to_owned).collect(),
        )
    }
}

fn charge(
    search: &mut SharedSearch<'_>,
    at: At,
    depth: usize,
    bytes: u64,
) -> Result<(), RecoveryError> {
    search.charge(depth, bytes).map_err(|stop| budget_error(search, at, stop))
}

fn budget_error(search: &SharedSearch<'_>, at: At, stop: SearchStop) -> RecoveryError {
    let (explored, unevaluated) = at.regions();
    let receipt = search.receipt(stop, explored, unevaluated);
    RecoveryError {
        detail: RecoveryDetail::Budget,
        message: format!("the shared recovery budget stopped ({})", stop.code()),
        receipt: Some(Box::new(receipt)),
    }
}

/// Decide whether `P(X(1), O)` is recoverable from the named observed pattern law,
/// and, when `effect` is given, identify it downstream, all under one budget.
///
/// # Errors
///
/// A typed [`RecoveryError`]: bounds, invalid query, unsupported mechanism,
/// missing margin, witness unavailable, handoff mismatch, effect not identified,
/// or a budget stop with its receipt.
pub fn decide_observation_recovery(
    graph: &Admg,
    query: &ObservationRecoveryQuery,
    catalog: &EvidenceCatalog,
    effect: Option<&RecoveredEffectQuery>,
    limits: RecoveryLimits,
    ctx: &ExecutionContext,
) -> Result<RecoveryDecision, RecoveryError> {
    let max = RECOVERY_DEFAULT_LIMITS;
    if limits.search.operations > max.search.operations
        || limits.search.depth > max.search.depth
        || limits.memory_bytes > max.memory_bytes
    {
        return Err(refuse(
            RecoveryDetail::BoundsExceeded,
            "search limits above 50000 operations / depth 32 / the default memory cap",
        ));
    }
    let query = query.canonical();
    let m = resolve_roles(graph, &query)?;
    let budget =
        SearchBudget::with_memory(limits.search, limits.memory_bytes, ctx).map_err(|receipt| {
            RecoveryError {
                detail: RecoveryDetail::Budget,
                message: format!(
                    "the recovery budget stopped before entry ({})",
                    receipt.stop.code()
                ),
                receipt: Some(Box::new(receipt)),
            }
        })?;
    let mut search = SharedSearch::new(budget);
    search.mark_decision();
    let base = graph_bytes(graph);
    let at = At { stage: Stage::ClassCheck, effect: effect.is_some() };
    let violations = check_class(&m, &query, catalog, &mut search, at, base)?;
    let observed = observed_distribution(&query, catalog)?;
    charge(&mut search, at, 1, base)?;
    if let Some(&(x, r)) = violations.first() {
        let witness = self_censoring_witness(&m, x, r);
        return nonrecoverable(&m, witness, &mut search, at.with(Stage::Witness), base);
    }
    let formula = at.with(Stage::Formula);
    let (arena, record) = build_formula(&m, &query, &observed, graph, &mut search, formula, base)?;
    let root = ExprId::from_raw(record.root);
    check_formula(graph, &query, &observed, &record, &arena, &mut |bytes| {
        charge(&mut search, formula, 2, base.saturating_add(bytes))
    })?;
    let (effect_derivation, effect_record) = match effect {
        None => (None, None),
        Some(effect) => {
            let (derivation, record) =
                identify_effect(&m, &query, effect, &mut search, at.with(Stage::Effect), ctx)?;
            (Some(derivation), Some(record))
        }
    };
    let mut record = record;
    record.effect = effect_record;
    record.receipt = RecoveryReceiptRecord {
        operations_limit: limits.search.operations,
        depth_limit: limits.search.depth,
        memory_bytes: limits.memory_bytes,
        operations_consumed: search.operations(),
        depth_reached: search.decision_depth(),
    };
    Ok(RecoveryDecision::Recovered(Box::new(RecoveryDerivation {
        query,
        record,
        arena,
        root,
        observed,
        effect: effect_derivation,
    })))
}

/// A violation is a nonrecoverability claim only with a verified witness;
/// otherwise nothing is claimed (bounded-incomplete, never a proof).
fn nonrecoverable(
    m: &MGraph<'_>,
    witness: RecoveryWitness,
    search: &mut SharedSearch<'_>,
    at: At,
    base: u64,
) -> Result<RecoveryDecision, RecoveryError> {
    match verify_witness_charged(m, &witness, &mut |bytes| {
        charge(search, at, 3, base.saturating_add(bytes))
    }) {
        Ok(_) => Ok(RecoveryDecision::NonRecoverable(Box::new(witness))),
        Err(error) if error.detail == RecoveryDetail::Budget => Err(error),
        Err(error) => Err(refuse(
            RecoveryDetail::WitnessUnavailable,
            format!("a self-censoring edge was found but no witness verified: {error}"),
        )),
    }
}

fn graph_bytes(graph: &Admg) -> u64 {
    u64::try_from(graph.node_count()).unwrap_or(u64::MAX).saturating_mul(256)
}

fn signature(graph: &Admg) -> String {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    for i in 0..graph.node_count() {
        let from = DenseNodeId::from_raw(u32::try_from(i).unwrap_or(u32::MAX));
        nodes.push(var(graph, from).raw());
        for child in graph.children(from) {
            edges.push((var(graph, from).raw(), var(graph, *child).raw()));
        }
    }
    nodes.sort_unstable();
    edges.sort_unstable();
    format!("mgraph.v1;nodes={nodes:?};directed={edges:?}")
}

fn margin_identity(
    observed: &CatalogDistribution,
    variables: &[VariableId],
) -> Result<String, RecoveryError> {
    let drop: Vec<_> =
        observed.measured.iter().copied().filter(|v| !variables.contains(v)).collect();
    if drop.is_empty() {
        return Ok(observed.canonical_identity());
    }
    observed
        .project(EvidenceProjection::Marginalize { drop: drop.into() })
        .map(|d| d.canonical_identity())
        .map_err(|e| refuse(RecoveryDetail::MissingMargin, format!("margin not licensed: {e}")))
}

fn leaf(
    arena: &mut CausalExprArena,
    variables: &[VariableId],
    query: &ObservationRecoveryQuery,
) -> ExprId {
    let vars = arena.intern_var_set(variables.iter().copied());
    let empty = arena.empty_var_set();
    let none = arena.empty_intervention_set();
    let population = arena.intern_population(Arc::clone(&query.population));
    arena.intern(ExprNode::Distribution {
        variables: vars,
        conditioned_on: empty,
        intervention: none,
        domain: DomainRef::Observational,
        population,
        regime: Some(query.observed_regime),
    })
}

/// Conditioning coordinates of `R_i`'s propensity: `R_S ∪ X*_S ∪ O_pa`.
fn conditioning(
    m: &MGraph<'_>,
    query: &ObservationRecoveryQuery,
    parents: &[VariableId],
) -> Vec<VariableId> {
    let mut out = Vec::new();
    for p in parents {
        match m.roles[p] {
            Role::Partial(j) => {
                out.push(query.partially_observed[j].response);
                out.push(query.partially_observed[j].proxy);
            }
            Role::Full => out.push(*p),
            Role::Response(_) | Role::Proxy(_) => {}
        }
    }
    out.sort_unstable();
    out
}

fn build_formula(
    m: &MGraph<'_>,
    query: &ObservationRecoveryQuery,
    observed: &CatalogDistribution,
    graph: &Admg,
    search: &mut SharedSearch<'_>,
    at: At,
    base: u64,
) -> Result<(CausalExprArena, RecoveryDerivationRecord), RecoveryError> {
    let mut arena = CausalExprArena::new();
    let all = query.observed();
    let numerator = leaf(&mut arena, &all, query);
    let mut margins = vec![(numerator, all.clone())];
    let mut factors = Vec::new();
    let mut premises = vec!["m_graph_dag_no_bidirected".to_owned()];
    for p in query.partially_observed.iter() {
        charge(search, at, 2, base.saturating_add(arena_bytes(&arena)))?;
        let parents = m.parents(p.response);
        let cond = conditioning(m, query, &parents);
        premises.push(format!("no_self_censoring:{}", p.response.raw()));
        premises.push(format!("response_parents:{}<-{:?}", p.response.raw(), ids(&parents)));
        let expression = if cond.is_empty() {
            let own = leaf(&mut arena, &[p.response], query);
            margins.push((own, vec![p.response]));
            own
        } else {
            let mut joint = cond.clone();
            joint.push(p.response);
            joint.sort_unstable();
            let top = leaf(&mut arena, &joint, query);
            let bottom = leaf(&mut arena, &cond, query);
            margins.push((top, joint));
            margins.push((bottom, cond.clone()));
            arena.intern(ExprNode::Ratio { numerator: top, denominator: bottom })
        };
        factors.push(RecoveryFactorRecord {
            response: p.response.raw(),
            parents: ids(&parents),
            conditioning: ids(&cond),
            expression: expression.raw(),
        });
    }
    let list = arena.intern_list(factors.iter().map(|f| ExprId::from_raw(f.expression)));
    let product = arena.intern(ExprNode::Product(list));
    let root = arena.intern(ExprNode::Ratio { numerator, denominator: product });
    let mut seen = BTreeSet::new();
    let mut margin_records = Vec::new();
    for (expression, variables) in margins {
        if !seen.insert(expression) {
            continue;
        }
        charge(search, at, 2, base.saturating_add(arena_bytes(&arena)))?;
        margin_records.push(RecoveryMarginRecord {
            expression: expression.raw(),
            variables: ids(&variables),
            identity: margin_identity(observed, &variables)?,
        });
    }
    margin_records.sort_by_key(|m| m.expression);
    let record = RecoveryDerivationRecord {
        rule_version: RECOVERY_RULE_VERSION.to_owned(),
        graph_signature: signature(graph),
        population: query.population.to_string(),
        observed_regime: query.observed_regime.raw(),
        observed_identity: observed.canonical_identity(),
        partially_observed: query
            .partially_observed
            .iter()
            .map(|p| (p.variable.raw(), p.response.raw(), p.proxy.raw()))
            .collect(),
        fully_observed: ids(&query.fully_observed),
        premises,
        numerator: numerator.raw(),
        factors,
        root: root.raw(),
        margins: margin_records,
        effect: None,
        receipt: RecoveryReceiptRecord {
            operations_limit: 0,
            depth_limit: 0,
            memory_bytes: 0,
            operations_consumed: 0,
            depth_reached: 0,
        },
    };
    Ok((arena, record))
}

fn arena_bytes(arena: &CausalExprArena) -> u64 {
    u64::try_from(arena.len().saturating_add(arena.table_entry_count()))
        .unwrap_or(u64::MAX)
        .saturating_mul(64)
}

/// Independent checker of a recovery formula. It does not reuse the builder: it
/// re-derives every propensity's conditioning set from the graph, verifies the
/// local-Markov premise `R_i _||_ R_S | pa(R_i)` by m-separation in the m-graph,
/// pattern-matches the arena from the root, and recomputes every margin's
/// catalog identity.
#[allow(clippy::too_many_lines)] // One linear pass per stage; splitting it would scatter the checks.
fn check_formula(
    graph: &Admg,
    query: &ObservationRecoveryQuery,
    observed: &CatalogDistribution,
    record: &RecoveryDerivationRecord,
    arena: &CausalExprArena,
    charge_step: &mut dyn FnMut(u64) -> Result<(), RecoveryError>,
) -> Result<(), RecoveryError> {
    let bad = |why: &str| {
        refuse(RecoveryDetail::InvalidDerivation, format!("recovery formula check: {why}"))
    };
    let root = ExprId::from_raw(record.root);
    if root.raw() as usize >= arena.len() {
        return Err(bad("root outside the arena"));
    }
    let node_of: BTreeMap<VariableId, DenseNodeId> = (0..graph.node_count())
        .map(|i| {
            let d = DenseNodeId::from_raw(u32::try_from(i).unwrap_or(u32::MAX));
            (var(graph, d), d)
        })
        .collect();
    let responses: BTreeMap<VariableId, usize> =
        query.partially_observed.iter().enumerate().map(|(i, p)| (p.response, i)).collect();
    let variables: BTreeMap<VariableId, usize> =
        query.partially_observed.iter().enumerate().map(|(i, p)| (p.variable, i)).collect();
    let fully: BTreeSet<VariableId> = query.fully_observed.iter().copied().collect();
    let expect_leaf = |id: ExprId, vars: &[VariableId]| -> Result<(), RecoveryError> {
        let ExprNode::Distribution {
            variables,
            conditioned_on,
            intervention,
            domain,
            population,
            regime,
        } = arena.node(id)
        else {
            return Err(bad("a factor is not an observed margin"));
        };
        let mut want = vars.to_vec();
        want.sort_unstable();
        if arena.var_set(*variables) != want.as_slice()
            || !arena.var_set(*conditioned_on).is_empty()
            || !arena.intervention_assignments(*intervention).is_empty()
            || *domain != DomainRef::Observational
            || arena.population(*population) != query.population.as_ref()
            || *regime != Some(query.observed_regime)
        {
            return Err(bad("a leaf is not the named observed margin"));
        }
        let identity = margin_identity(observed, &want)?;
        if !record.margins.iter().any(|m| {
            m.expression == id.raw() && m.variables == ids(&want) && m.identity == identity
        }) {
            return Err(bad("a leaf's margin binding disagrees with the catalog distribution"));
        }
        Ok(())
    };
    let ExprNode::Ratio { numerator, denominator } = arena.node(root) else {
        return Err(bad("the root is not complete case over propensities"));
    };
    expect_leaf(*numerator, &query.observed())?;
    let ExprNode::Product(list) = arena.node(*denominator) else {
        return Err(bad("the denominator is not a product of propensities"));
    };
    let list = arena.list(*list);
    if list.len() != query.partially_observed.len() {
        return Err(bad("one propensity per response indicator is required"));
    }
    let mut ws = DSeparationWorkspace::default();
    let mut used = 1 + list.len();
    for (p, factor) in query.partially_observed.iter().zip(list) {
        charge_step(64)?;
        let r = *node_of.get(&p.response).ok_or_else(|| bad("unknown response"))?;
        let parents: Vec<VariableId> = graph.parents(r).iter().map(|d| var(graph, *d)).collect();
        let mut cond = Vec::new();
        let mut other_responses = Vec::new();
        for parent in &parents {
            if let Some(&j) = variables.get(parent) {
                if j == responses[&p.response] {
                    return Err(bad("self-censoring edge: the propensity is not observed"));
                }
                cond.push(query.partially_observed[j].response);
                cond.push(query.partially_observed[j].proxy);
                other_responses.push(query.partially_observed[j].response);
            } else if fully.contains(parent) {
                cond.push(*parent);
            } else {
                return Err(bad("a response indicator has a parent outside X and O"));
            }
        }
        let parent_nodes: Vec<DenseNodeId> = parents.iter().map(|v| node_of[v]).collect();
        for other in &other_responses {
            charge_step(64)?;
            let separated = graph
                .is_m_separated(r, node_of[other], &parent_nodes, &mut ws)
                .map_err(|e| bad(&e.to_string()))?;
            if !separated {
                return Err(bad("the local-Markov premise R_i _||_ R_S | pa(R_i) fails"));
            }
        }
        if cond.is_empty() {
            expect_leaf(*factor, &[p.response])?;
        } else {
            let ExprNode::Ratio { numerator, denominator } = arena.node(*factor) else {
                return Err(bad("a propensity is not a ratio of observed margins"));
            };
            let mut joint = cond.clone();
            joint.push(p.response);
            expect_leaf(*numerator, &joint)?;
            expect_leaf(*denominator, &cond)?;
            used += 2;
        }
    }
    // Every recorded margin is a leaf the formula uses: nothing extra is bound.
    let leaves: BTreeSet<u32> = arena.distribution_leaves(root).iter().map(|e| e.raw()).collect();
    if record.margins.iter().any(|m| !leaves.contains(&m.expression))
        || record.margins.len() != leaves.len()
    {
        return Err(bad("a recorded margin is not a leaf of the formula"));
    }
    let _ = used;
    Ok(())
}

/// The closed-form self-censoring witness for `x -> r`: every mechanism a fair
/// coin except `x` and `r`, which differ between the two models while
/// `P(r = 1, x)` is equal.
fn self_censoring_witness(m: &MGraph<'_>, x: VariableId, r: VariableId) -> RecoveryWitness {
    let half = WITNESS_SCALE / 2;
    let mut first = Vec::new();
    let mut second = Vec::new();
    for (v, role) in &m.roles {
        if matches!(role, Role::Proxy(_)) {
            continue;
        }
        let parents =
            m.graph.parents(m.dense[v]).iter().map(|p| var(m.graph, *p)).collect::<Vec<_>>();
        let configs = 1usize << parents.len();
        let at = parents.iter().position(|p| *p == x);
        let table = |x1: u32, x0: u32| -> Vec<u32> {
            (0..configs)
                .map(|c| {
                    let bit = at.map(|i| (c >> (parents.len() - 1 - i)) & 1);
                    if bit == Some(1) { x1 } else { x0 }
                })
                .collect()
        };
        let (a, b) = if *v == x {
            (vec![half; configs], vec![24; configs])
        } else if *v == r {
            (table(36, 24), table(45, 20))
        } else {
            (vec![half; configs], vec![half; configs])
        };
        let raw = parents.iter().map(|p| p.raw()).collect::<Vec<_>>();
        first.push(WitnessMechanism { node: v.raw(), parents: raw.clone(), numerators: a });
        second.push(WitnessMechanism { node: v.raw(), parents: raw, numerators: b });
    }
    RecoveryWitness { edge: (x.raw(), r.raw()), first, second }
}

/// Verify a witness against an m-graph by exact enumeration: both models are
/// positive and Markov to the graph, agree on every observed-law cell, and
/// differ on the target `P(X, O)`. No theorem is trusted. A verified witness
/// shows non-recoverability over every model Markov to the m-graph, not under
/// faithful or generic parameters.
///
/// Not charged to a budget: the role bounds (at most three partially and two
/// fully observed variables) cap the enumeration at eight non-proxy nodes,
/// `2 * 2^8` configurations.
///
/// # Errors
///
/// [`RecoveryDetail::InvalidQuery`] for a malformed role declaration, or
/// [`RecoveryDetail::InvalidDerivation`] when the witness does not verify.
pub fn verify_recovery_witness(
    graph: &Admg,
    query: &ObservationRecoveryQuery,
    witness: &RecoveryWitness,
) -> Result<WitnessCheck, RecoveryError> {
    let query = query.canonical();
    let m = resolve_roles(graph, &query)?;
    if graph.has_bidirected() {
        return Err(refuse(RecoveryDetail::InvalidDerivation, "a witness is checked on a DAG"));
    }
    verify_witness_charged(&m, witness, &mut |_| Ok(()))
}

#[allow(clippy::too_many_lines)] // One linear pass per stage; splitting it would scatter the checks.
fn verify_witness_charged(
    m: &MGraph<'_>,
    witness: &RecoveryWitness,
    charge_step: &mut dyn FnMut(u64) -> Result<(), RecoveryError>,
) -> Result<WitnessCheck, RecoveryError> {
    let bad = |why: &str| refuse(RecoveryDetail::InvalidDerivation, format!("witness: {why}"));
    // Nodes with a mechanism: every non-proxy node, in variable order.
    let nodes: Vec<VariableId> =
        m.roles.iter().filter(|(_, r)| !matches!(r, Role::Proxy(_))).map(|(v, _)| *v).collect();
    let index: BTreeMap<VariableId, usize> =
        nodes.iter().enumerate().map(|(i, v)| (*v, i)).collect();
    let n = nodes.len();
    if n > 16 {
        return Err(bad("too many nodes to enumerate"));
    }
    let check_model = |model: &[WitnessMechanism]| -> Result<(), RecoveryError> {
        if model.len() != n {
            return Err(bad("a model must give one mechanism per non-proxy node"));
        }
        for mechanism in model {
            let v = VariableId::from_raw(mechanism.node);
            if !index.contains_key(&v) {
                return Err(bad("a mechanism names a proxy or an unknown node"));
            }
            let graph_parents = m
                .graph
                .parents(m.dense[&v])
                .iter()
                .map(|p| var(m.graph, *p).raw())
                .collect::<Vec<_>>();
            if mechanism.parents != graph_parents {
                return Err(bad(
                    "a mechanism's parents are not the node's graph parents (not Markov to the m-graph)",
                ));
            }
            if mechanism.numerators.len() != 1 << graph_parents.len() {
                return Err(bad("a mechanism must give one probability per parent configuration"));
            }
            if mechanism.numerators.iter().any(|k| *k == 0 || *k >= WITNESS_SCALE) {
                return Err(bad(
                    "a mechanism probability is not strictly between 0 and 1 (positivity)",
                ));
            }
        }
        let distinct: BTreeSet<u32> = model.iter().map(|m| m.node).collect();
        if distinct.len() != n {
            return Err(bad("a node has two mechanisms"));
        }
        Ok(())
    };
    check_model(&witness.first)?;
    check_model(&witness.second)?;
    let substantive: Vec<VariableId> = m
        .roles
        .iter()
        .filter(|(_, r)| matches!(r, Role::Partial(_) | Role::Full))
        .map(|(v, _)| *v)
        .collect();
    // Observed coordinates: each response and fully observed value, and each
    // proxy's value in {0, 1, ?} (encoded 0, 1, 2).
    let pairs = witness_roles(m);
    let enumerate = |model: &[WitnessMechanism],
                     charge_step: &mut dyn FnMut(u64) -> Result<(), RecoveryError>|
     -> Result<(Masses, Masses), RecoveryError> {
        let by_node: BTreeMap<u32, &WitnessMechanism> = model.iter().map(|m| (m.node, m)).collect();
        let mut observed = BTreeMap::new();
        let mut target = BTreeMap::new();
        for config in 0u32..(1u32 << n) {
            charge_step(
                u64::try_from(observed.len() + target.len()).unwrap_or(u64::MAX).saturating_mul(48),
            )?;
            let value = |v: VariableId| u8::from((config >> index[&v]) & 1 == 1);
            let mut mass: u128 = 1;
            for v in &nodes {
                let mechanism = by_node[&v.raw()];
                let parent_config = mechanism.parents.iter().fold(0usize, |acc, p| {
                    (acc << 1) | usize::from(value(VariableId::from_raw(*p)))
                });
                let k = u128::from(mechanism.numerators[parent_config]);
                mass *= if value(*v) == 1 { k } else { u128::from(WITNESS_SCALE) - k };
            }
            let mut cell = Vec::new();
            for (v, role) in &m.roles {
                cell.push(match role {
                    Role::Partial(_) => continue,
                    Role::Proxy(i) => {
                        let p = &pairs[*i];
                        if value(p.1) == 1 { value(p.0) } else { 2 }
                    }
                    Role::Response(_) | Role::Full => value(*v),
                });
            }
            *observed.entry(cell).or_insert(0) += mass;
            let target_cell: Vec<u8> = substantive.iter().map(|v| value(*v)).collect();
            *target.entry(target_cell).or_insert(0) += mass;
        }
        Ok((observed, target))
    };
    let (observed_a, target_a) = enumerate(&witness.first, charge_step)?;
    let (observed_b, target_b) = enumerate(&witness.second, charge_step)?;
    if observed_a != observed_b {
        return Err(bad("the two models disagree on the observed law"));
    }
    let Some((cell, mass_a)) =
        target_a.iter().find(|(cell, mass)| target_b.get(*cell) != Some(mass))
    else {
        return Err(bad("the two models agree on the target"));
    };
    let denominator = u128::from(WITNESS_SCALE).pow(u32::try_from(n).unwrap_or(u32::MAX));
    Ok(WitnessCheck {
        observed_cells: observed_a.len(),
        differing_cell: substantive.iter().copied().zip(cell.iter().copied()).collect(),
        masses: (*mass_a, target_b.get(cell).copied().unwrap_or(0)),
        denominator,
    })
}

/// Exact integer masses per cell (observed or target levels).
type Masses = BTreeMap<Vec<u8>, u128>;

/// `(variable, response)` per partially observed index.
fn witness_roles(m: &MGraph<'_>) -> Vec<(VariableId, VariableId)> {
    let mut out = BTreeMap::new();
    for (v, role) in &m.roles {
        match role {
            Role::Partial(i) => out.entry(*i).or_insert((*v, *v)).0 = *v,
            Role::Response(i) => out.entry(*i).or_insert((*v, *v)).1 = *v,
            _ => {}
        }
    }
    out.into_values().collect()
}

/// Identify the downstream effect with the ordinary target ID stage of sID on
/// the causal restriction of the m-graph, charged to the shared budget.
#[allow(clippy::too_many_lines)] // One linear pass per stage; splitting it would scatter the checks.
fn identify_effect(
    m: &MGraph<'_>,
    query: &ObservationRecoveryQuery,
    effect: &RecoveredEffectQuery,
    search: &mut SharedSearch<'_>,
    at: At,
    ctx: &ExecutionContext,
) -> Result<(RecoveredEffect, RecoveredEffectRecord), RecoveryError> {
    use RecoveryDetail::{HandoffMismatch, InvalidDerivation, InvalidQuery};
    let substantive: BTreeSet<VariableId> = query.substantive().into_iter().collect();
    let mut nodes = BTreeSet::new();
    for node in effect.graph.nodes() {
        let NodeRef::Static(v) = node else {
            return Err(refuse(HandoffMismatch, "the effect graph must have static nodes only"));
        };
        nodes.insert(*v);
    }
    if nodes != substantive {
        return Err(refuse(
            HandoffMismatch,
            "the effect graph's variables are not exactly the recovered law's variables X and O",
        ));
    }
    if effect.graph.has_bidirected() {
        return Err(refuse(
            HandoffMismatch,
            "the recovered law factorizes by a DAG; a bidirected edge changes the factorization",
        ));
    }
    let mut edges = Vec::new();
    for i in 0..effect.graph.node_count() {
        let from = DenseNodeId::from_raw(u32::try_from(i).unwrap_or(u32::MAX));
        for child in effect.graph.children(from) {
            edges.push((var(&effect.graph, from).raw(), var(&effect.graph, *child).raw()));
        }
    }
    edges.sort_unstable();
    let mut restriction = Vec::new();
    for v in &substantive {
        for child in m.children(*v) {
            if substantive.contains(&child) {
                restriction.push((v.raw(), child.raw()));
            }
        }
    }
    restriction.sort_unstable();
    if edges != restriction {
        return Err(refuse(
            HandoffMismatch,
            "the effect graph is not the m-graph's causal restriction (factorization mismatch)",
        ));
    }
    let mut outcomes = effect.outcomes.to_vec();
    outcomes.sort_unstable();
    let mut treatments = effect.treatments.to_vec();
    treatments.sort_unstable();
    if outcomes.is_empty()
        || outcomes.windows(2).any(|w| w[0] == w[1])
        || treatments.windows(2).any(|w| w[0] == w[1])
        || outcomes.iter().any(|o| treatments.contains(o))
    {
        return Err(refuse(
            InvalidQuery,
            "effect outcomes must be non-empty, distinct and disjoint from treatments",
        ));
    }
    if outcomes.iter().chain(&treatments).any(|v| !substantive.contains(v)) {
        return Err(refuse(
            HandoffMismatch,
            "effect outcomes and treatments must be variables of the recovered law",
        ));
    }
    let diagram = SelectionDiagram::try_new(effect.graph.clone(), Arc::<[VariableId]>::from([]))
        .map_err(|e| refuse(HandoffMismatch, e.to_string()))?;
    let target: Arc<str> = Arc::clone(&query.population);
    let sid_query = ClassicalTransportQuery {
        outcomes: outcomes.clone().into(),
        treatments: treatments.clone().into(),
        // The source is a label only: a derivation citing any leaf other than the
        // target's observational law is refused below, so no source law is used.
        source: Arc::from(format!("{target}#recovery-no-source")),
        target: Arc::clone(&target),
    };
    search.begin_stage();
    let result = identify_classical_transport_metered(&diagram, &sid_query, search.meter(), ctx)
        .map_err(|error| {
            if error.is_budget_or_cancel() {
                budget_error(search, at, search.stop_of(&error))
            } else {
                refuse(InvalidDerivation, format!("ordinary target ID failed: {error}"))
            }
        })?;
    search.end_stage();
    // Every effect on a DAG whose variables the recovered law all covers is
    // identified by ordinary target ID (the g-formula); anything else is a defect,
    // reported as a failed derivation, never as a claim about the effect.
    let ClassicalTransportResult::Identified(derivation) = result else {
        return Err(refuse(
            InvalidDerivation,
            "ordinary target ID did not identify an effect on a fully recovered DAG",
        ));
    };
    let arena = derivation.arena().clone();
    let root = derivation.root();
    // Defense in depth, unreachable through the public API: sID on a selection
    // diagram without selection nodes identifies every DAG effect by target ID,
    // whose leaves are all target observational laws (audit mutant A7 survives
    // every public-route test for that reason). The predicate is unit tested
    // directly (`the_effect_leaf_predicate_refuses_foreign_or_interventional_leaves`).
    if !cites_only_target_observational(&arena, root, &target) {
        return Err(refuse(
            InvalidDerivation,
            "the effect derivation cites a law other than the recovered target law",
        ));
    }
    let record = RecoveredEffectRecord {
        outcomes: ids(&outcomes),
        treatments: ids(&treatments),
        edges,
        root: root.raw(),
        rules: derivation.rules().into_iter().map(str::to_owned).collect(),
    };
    Ok((
        RecoveredEffect { arena, root, outcomes: outcomes.into(), treatments: treatments.into() },
        record,
    ))
}

/// Whether every distribution leaf under `root` is an observational,
/// non-interventional law of `target`: the only law a recovered law supplies.
fn cites_only_target_observational(arena: &CausalExprArena, root: ExprId, target: &str) -> bool {
    arena.distribution_leaves(root).into_iter().all(|leaf| {
        let ExprNode::Distribution { intervention, domain, population, .. } = arena.node(leaf)
        else {
            return true;
        };
        arena.population(*population) == target
            && *domain == DomainRef::Observational
            && arena.intervention_assignments(*intervention).is_empty()
    })
}

/// Re-check a stored derivation: re-decide from the stored inputs under the
/// stored limits (refused when above `max`), accept only an identical record and
/// arena, and run the independent formula checker on the stored record.
///
/// # Errors
///
/// [`RecoveryDetail::BoundsExceeded`] for stored limits above `max`;
/// [`RecoveryDetail::InvalidDerivation`] when anything differs; the decision's
/// own refusal when the inputs no longer recover.
#[allow(clippy::too_many_arguments)]
pub fn verify_observation_recovery(
    graph: &Admg,
    query: &ObservationRecoveryQuery,
    catalog: &EvidenceCatalog,
    effect: Option<&RecoveredEffectQuery>,
    record: &RecoveryDerivationRecord,
    arena: &CausalExprArena,
    effect_arena: Option<&CausalExprArena>,
    max: RecoveryLimits,
    ctx: &ExecutionContext,
) -> Result<RecoveryDerivation, RecoveryError> {
    let stored = RecoveryLimits {
        search: SearchLimits {
            operations: record.receipt.operations_limit,
            depth: record.receipt.depth_limit,
        },
        memory_bytes: record.receipt.memory_bytes,
    };
    if stored.search.operations > max.search.operations
        || stored.search.depth > max.search.depth
        || stored.memory_bytes > max.memory_bytes
    {
        return Err(refuse(
            RecoveryDetail::BoundsExceeded,
            "the stored search limits exceed the consumer's maxima",
        ));
    }
    let invalid = |why: &str| refuse(RecoveryDetail::InvalidDerivation, why.to_owned());
    let canonical = query.canonical();
    let observed = observed_distribution(&canonical, catalog)?;
    // The independent checker runs on the stored record and arena first, so a
    // tampered formula is refused for what it is, not only for differing.
    check_formula(graph, &canonical, &observed, record, arena, &mut |_| Ok(()))?;
    let decision = decide_observation_recovery(graph, query, catalog, effect, stored, ctx)?;
    let RecoveryDecision::Recovered(fresh) = decision else {
        return Err(invalid("the stored inputs are nonrecoverable"));
    };
    if fresh.record != *record {
        return Err(invalid("the stored derivation record is not reproduced"));
    }
    if fresh.arena != *arena {
        return Err(invalid("the stored formula expression is not reproduced"));
    }
    match (&fresh.effect, effect_arena) {
        (None, None) => {}
        (Some(e), Some(stored)) if e.arena == *stored => {}
        _ => return Err(invalid("the stored effect expression is not reproduced")),
    }
    Ok(*fresh)
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{
        DistributionAvailability, EvidenceKind, EvidenceRegime, RegimeKind, SearchStop,
    };

    fn v(raw: u32) -> VariableId {
        VariableId::from_raw(raw)
    }

    const WITNESS_AT: At = At { stage: Stage::Witness, effect: false };

    /// X=0, R=1, X*=2, O=3; O -> X, O -> R, X -> X*, R -> X*.
    fn k1() -> (Admg, ObservationRecoveryQuery, EvidenceCatalog) {
        let mut g = Admg::with_variables(4);
        for (a, b) in [(3, 0), (3, 1), (0, 2), (1, 2)] {
            g.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        let q = ObservationRecoveryQuery {
            population: Arc::from("p"),
            observed_regime: RegimeId::from_raw(0),
            partially_observed: Arc::from([PartiallyObserved {
                variable: v(0),
                response: v(1),
                proxy: v(2),
            }]),
            fully_observed: Arc::from([v(3)]),
        };
        let regime = EvidenceRegime::try_new(
            RegimeId::from_raw(0),
            RegimeKind::Observational,
            EvidenceKind::Available,
            [],
            [],
            [v(1), v(2), v(3)],
            "p",
            DistributionAvailability::Joint,
        )
        .unwrap();
        let catalog = EvidenceCatalog {
            environments: Arc::from([]),
            regimes: Arc::from([regime]),
            bindings: Arc::from([]),
            target_sampling: None,
        };
        (g, q, catalog)
    }

    #[test]
    fn the_checker_rejects_a_formula_whose_propensity_lost_its_conditioning() {
        let (g, q, c) = k1();
        let ctx = ExecutionContext::for_tests(1);
        let RecoveryDecision::Recovered(d) =
            decide_observation_recovery(&g, &q, &c, None, RecoveryLimits::default(), &ctx).unwrap()
        else {
            panic!("recoverable");
        };
        let observed = d.observed().clone();
        // Rebuild the formula with P(R) in place of P(R | O): the mutated factor.
        let mut arena = CausalExprArena::new();
        let canonical = q.canonical();
        let numerator = leaf(&mut arena, &canonical.observed(), &canonical);
        let bare = leaf(&mut arena, &[v(1)], &canonical);
        let list = arena.intern_list([bare]);
        let product = arena.intern(ExprNode::Product(list));
        let root = arena.intern(ExprNode::Ratio { numerator, denominator: product });
        let mut record = d.record().clone();
        record.root = root.raw();
        record.margins = vec![
            RecoveryMarginRecord {
                expression: numerator.raw(),
                variables: vec![1, 2, 3],
                identity: margin_identity(&observed, &canonical.observed()).unwrap(),
            },
            RecoveryMarginRecord {
                expression: bare.raw(),
                variables: vec![1],
                identity: margin_identity(&observed, &[v(1)]).unwrap(),
            },
        ];
        let error =
            check_formula(&g, &canonical, &observed, &record, &arena, &mut |_| Ok(())).unwrap_err();
        assert_eq!(error.detail, RecoveryDetail::InvalidDerivation);
        assert!(error.message.contains("not a ratio"), "{error}");
        // The unmutated formula passes the same checker.
        check_formula(&g, &canonical, &observed, d.record(), d.arena(), &mut |_| Ok(())).unwrap();
    }

    #[test]
    fn the_checker_refuses_a_stored_self_censoring_factor_by_itself() {
        // A formula lowered on the compliant k1 graph, then checked against the
        // same graph plus the self-censoring edge X -> R: the independent checker
        // must refuse it for the self-censoring edge itself (a stored record is
        // checked before any replay, so it cannot lean on the class check).
        let (g, q, c) = k1();
        let ctx = ExecutionContext::for_tests(1);
        let RecoveryDecision::Recovered(d) =
            decide_observation_recovery(&g, &q, &c, None, RecoveryLimits::default(), &ctx).unwrap()
        else {
            panic!("recoverable");
        };
        let mut censored = g.clone();
        censored.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let canonical = q.canonical();
        let error =
            check_formula(&censored, &canonical, d.observed(), d.record(), d.arena(), &mut |_| {
                Ok(())
            })
            .unwrap_err();
        assert_eq!(error.detail, RecoveryDetail::InvalidDerivation);
        assert!(error.message.contains("self-censoring edge"), "{error}");
        // The same record passes on the graph it was decided on.
        check_formula(&g, &canonical, d.observed(), d.record(), d.arena(), &mut |_| Ok(()))
            .unwrap();
    }

    #[test]
    fn the_effect_leaf_predicate_refuses_foreign_or_interventional_leaves() {
        // Pure-function test of the defense-in-depth predicate; the public route
        // never reaches its refusal (see identify_effect).
        let mut arena = CausalExprArena::new();
        let mut leaf_of = |population: &str, domain: DomainRef, intervened: &[VariableId]| {
            let variables = arena.intern_var_set([v(0)]);
            let conditioned_on = arena.empty_var_set();
            let intervention = arena.intern_intervention_set(intervened.iter().copied());
            let population = arena.intern_population(Arc::from(population));
            arena.intern(ExprNode::Distribution {
                variables,
                conditioned_on,
                intervention,
                domain,
                population,
                regime: None,
            })
        };
        let good = leaf_of("p", DomainRef::Observational, &[]);
        let foreign = leaf_of("q", DomainRef::Observational, &[]);
        let interventional = leaf_of("p", DomainRef::Interventional, &[v(1)]);
        let assigned = leaf_of("p", DomainRef::Observational, &[v(1)]);
        assert!(cites_only_target_observational(&arena, good, "p"));
        assert!(!cites_only_target_observational(&arena, good, "q"));
        for bad in [foreign, interventional, assigned] {
            assert!(!cites_only_target_observational(&arena, bad, "p"));
            let list = arena.intern_list([good, bad]);
            let product = arena.intern(ExprNode::Product(list));
            assert!(!cites_only_target_observational(&arena, product, "p"));
        }
    }

    #[test]
    fn the_checker_refuses_a_propensity_whose_local_markov_premise_fails() {
        // X0=0, X1=1, O=2, R0=3, R1=4, X*0=5, X*1=6 with X1 -> R0, R0 -> O -> R1: R1
        // is a descendant of R0, so R0 is not separated from R1 given pa(R0) = {X1}
        // and p(R0 = 1 | X1) is not P(R0 = 1 | R1 = 1, X*1). The class check refuses
        // R0 -> O, but the independent checker must refuse the formula on its own
        // (a stored record is checked before the decision is replayed).
        let mut g = Admg::with_variables(7);
        for (a, b) in [(1, 3), (3, 2), (2, 4), (0, 5), (3, 5), (1, 6), (4, 6)] {
            g.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        let q = ObservationRecoveryQuery {
            population: Arc::from("p"),
            observed_regime: RegimeId::from_raw(0),
            partially_observed: Arc::from([
                PartiallyObserved { variable: v(0), response: v(3), proxy: v(5) },
                PartiallyObserved { variable: v(1), response: v(4), proxy: v(6) },
            ]),
            fully_observed: Arc::from([v(2)]),
        };
        let regime = EvidenceRegime::try_new(
            RegimeId::from_raw(0),
            RegimeKind::Observational,
            EvidenceKind::Available,
            [],
            [],
            [v(2), v(3), v(4), v(5), v(6)],
            "p",
            DistributionAvailability::Joint,
        )
        .unwrap();
        let catalog = EvidenceCatalog {
            environments: Arc::from([]),
            regimes: Arc::from([regime]),
            bindings: Arc::from([]),
            target_sampling: None,
        };
        let canonical = q.canonical();
        let m = resolve_roles(&g, &canonical).unwrap();
        let observed = observed_distribution(&canonical, &catalog).unwrap();
        let ctx = ExecutionContext::for_tests(1);
        let mut search =
            SharedSearch::new(SearchBudget::new(RECOVERY_DEFAULT_LIMITS.search, &ctx).unwrap());
        let at = At { stage: Stage::Formula, effect: false };
        let (arena, record) =
            build_formula(&m, &canonical, &observed, &g, &mut search, at, 0).unwrap();
        let error =
            check_formula(&g, &canonical, &observed, &record, &arena, &mut |_| Ok(())).unwrap_err();
        assert_eq!(error.detail, RecoveryDetail::InvalidDerivation);
        assert!(error.message.contains("local-Markov"), "{error}");
        // The decision itself refuses the mechanism (a response causing O).
        let decided =
            decide_observation_recovery(&g, &q, &catalog, None, RecoveryLimits::default(), &ctx)
                .unwrap_err();
        assert_eq!(decided.detail, RecoveryDetail::UnsupportedMechanism);
    }

    #[test]
    fn a_violation_without_a_verified_witness_is_never_a_nonrecoverability_claim() {
        // X -> R self-censoring: the constructed witness verifies; a corrupted one
        // (the second model's response mechanism no longer compensates) does not,
        // and the outcome is witness_unavailable (not certified), never a proof.
        let (mut g, q, _) = k1();
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let canonical = q.canonical();
        let m = resolve_roles(&g, &canonical).unwrap();
        let ctx = ExecutionContext::for_tests(1);
        let witness = self_censoring_witness(&m, v(0), v(1));
        let mut search =
            SharedSearch::new(SearchBudget::new(RECOVERY_DEFAULT_LIMITS.search, &ctx).unwrap());
        assert!(matches!(
            nonrecoverable(&m, witness.clone(), &mut search, WITNESS_AT, 0).unwrap(),
            RecoveryDecision::NonRecoverable(_)
        ));
        let mut broken = witness;
        let r = broken.second.iter_mut().find(|m| m.node == 1).unwrap();
        r.numerators = r.numerators.iter().map(|k| if *k == 45 { 44 } else { *k }).collect();
        let mut search =
            SharedSearch::new(SearchBudget::new(RECOVERY_DEFAULT_LIMITS.search, &ctx).unwrap());
        let error = nonrecoverable(&m, broken, &mut search, WITNESS_AT, 0).unwrap_err();
        assert_eq!(error.detail, RecoveryDetail::WitnessUnavailable);
        assert_eq!(error.reason_code(), "transport_not_certified");
    }

    #[test]
    fn the_shared_budget_is_charged_by_every_stage() {
        let (g, q, c) = k1();
        let ctx = ExecutionContext::for_tests(1);
        let RecoveryDecision::Recovered(d) =
            decide_observation_recovery(&g, &q, &c, None, RecoveryLimits::default(), &ctx).unwrap()
        else {
            panic!("recoverable");
        };
        let used = d.record().receipt.operations_consumed;
        assert!(used >= 6, "class check, formula and checker all charge: {used}");
        for limit in 1..used {
            let limits = RecoveryLimits {
                search: SearchLimits { operations: limit, depth: 32 },
                memory_bytes: RecoveryLimits::default().memory_bytes,
            };
            let error = decide_observation_recovery(&g, &q, &c, None, limits, &ctx).unwrap_err();
            assert_eq!(error.detail, RecoveryDetail::Budget);
            assert_eq!(error.receipt.as_ref().unwrap().stop, SearchStop::Operations);
        }
    }
}
