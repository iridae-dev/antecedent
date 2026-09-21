//! Arena-backed causal-functional IR.
//!
//! # Modules
//!
//! - [`estimand`] — identified estimand + method tags
//! - [`eval`] — compiled evaluators over providers
//! - [`simplify`] — algebraic simplification
//! - [`pretty`] / [`latex`] — display helpers
//! - [`provider`] — distribution / table / posterior providers
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod estimand;
pub mod eval;
pub mod exact;
mod exact_engine;
pub mod exact_plan;
pub use exact::{
    DiscreteAxis, ExactDiscreteLaw, ExactLawError, ExactTransportData, LawOrigin, LawTolerance,
};
pub use exact_plan::{
    ExactDistribution, ExactEvaluationLimits, ExactEvaluationPlan, ExactSupportRecord,
};
pub mod latex;
pub mod pretty;
pub mod provider;
pub mod simplify;

pub use estimand::{EstimandMethod, IdentifiedEstimand, RdDesignParams};
pub use eval::CompiledEvaluator;
pub use provider::{
    Assignment, DistributionProvider, EmpiricalTableProvider, EvalContext, EvalError, FactorSpec,
    GaussianDensityProvider, PosteriorDrawProvider, QuadratureNodes,
};
pub use simplify::SimplifyError;

mod scope;
pub use scope::LeafBinding;

use latex::latex_expr;
use pretty::pretty_expr;

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use antecedent_core::{RegimeId, Value, VariableId};

/// Opaque expression node id.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ExprId(u32);

impl ExprId {
    /// Create from a raw index (tests / deserialization).
    #[must_use]
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Raw index.
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

/// Interned sorted variable set id.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct VarSetId(u32);

impl VarSetId {
    /// Create from a raw index (deserialization).
    #[must_use]
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Raw index.
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

/// Interned intervention-set id (hard assignments `do(V := value)`).
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct InterventionSetId(u32);

impl InterventionSetId {
    /// Create from a raw index (deserialization).
    #[must_use]
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Raw index.
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

/// One hard intervention assignment in an interned set.
///
/// Symbolic coordinates (`do(V)` with unspecified level) use
/// [`Value::symbolic_intervention`] in [`Self::value`], matching the wire
/// form's `symbolic: bool`. A genuine non-finite float is never that mark.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct InterventionAssignment {
    /// Target variable.
    pub variable: VariableId,
    /// Assigned value under `do(·)`, or the symbolic marker.
    pub value: Value,
}

impl InterventionAssignment {
    /// Concrete hard intervention `do(variable = value)`.
    #[must_use]
    pub const fn concrete(variable: VariableId, value: Value) -> Self {
        Self { variable, value }
    }

    /// Symbolic intervention coordinate `do(variable)` (level unspecified).
    #[must_use]
    pub fn symbolic(variable: VariableId) -> Self {
        Self { variable, value: Value::symbolic_intervention() }
    }

    /// Whether this assignment is a symbolic (unspecified-level) coordinate.
    #[must_use]
    pub fn is_symbolic(&self) -> bool {
        self.value.is_symbolic_intervention()
    }
}

/// Contrast operator between two expressions.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ContrastOp {
    /// Left − right.
    Difference,
}

/// Domain reference for a distribution .
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum DomainRef {
    /// Observational P(·).
    Observational,
    /// Interventional P(· | do(·)).
    Interventional,
}

/// Outcome function id .
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct OutcomeExprId(VariableId);

impl OutcomeExprId {
    /// Identity outcome Y.
    #[must_use]
    pub const fn identity(variable: VariableId) -> Self {
        Self(variable)
    }

    /// Underlying variable.
    #[must_use]
    pub const fn variable(self) -> VariableId {
        self.0
    }
}

/// Interned population key id.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct PopulationKeyId(u32);

impl PopulationKeyId {
    /// Create from a raw index (deserialization).
    #[must_use]
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Raw index.
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

/// Expression list id (product children).
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ExprListId(u32);

impl ExprListId {
    /// Create from a raw index (deserialization).
    #[must_use]
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Raw index.
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

/// Semantic expression node (no derivation metadata).
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum ExprNode {
    /// Joint / conditional distribution factor.
    Distribution {
        /// Variables in the factor.
        variables: VarSetId,
        /// Conditioning set.
        conditioned_on: VarSetId,
        /// Intervention set (empty for observational).
        intervention: InterventionSetId,
        /// Domain.
        domain: DomainRef,
        /// Population this factor is labelled with. Empty is the default (single-study).
        population: PopulationKeyId,
        /// Catalog regime this leaf cites. `None` is an anonymous / single-study factor.
        regime: Option<RegimeId>,
    },
    /// Intermediate kernel: a nested subexpression, not a supplied observational law.
    Kernel {
        /// Kernel body.
        body: ExprId,
        /// Kernel parameter coordinates. They remain free until explicitly marginalized.
        bound: VarSetId,
        /// Population this kernel is labelled with.
        population: PopulationKeyId,
        /// Catalog regime, if this kernel is bound to one.
        regime: Option<RegimeId>,
    },
    /// Product of factors.
    Product(ExprListId),
    /// Discrete marginalization.
    SumOut {
        /// Variables summed out.
        variables: VarSetId,
        /// Body.
        expr: ExprId,
    },
    /// Continuous marginalization.
    IntegralOut {
        /// Variables integrated out.
        variables: VarSetId,
        /// Body.
        expr: ExprId,
    },
    /// Ratio of expressions.
    Ratio {
        /// Numerator.
        numerator: ExprId,
        /// Denominator.
        denominator: ExprId,
    },
    /// Expectation of an outcome under a distribution.
    Expectation {
        /// Outcome function.
        function: OutcomeExprId,
        /// Distribution expression.
        distribution: ExprId,
    },
    /// Contrast of two expectations / functionals.
    Contrast {
        /// Left side.
        left: ExprId,
        /// Right side.
        right: ExprId,
        /// Operator.
        op: ContrastOp,
    },
}

/// Errors from constructing a tagged expression leaf.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ExprError {
    /// A catalog regime's observational/experimental kind disagrees with [`DomainRef`].
    RegimeDomainMismatch,
    /// Substitution would capture a bound variable or conflict with an assignment.
    CaptureOrConflict,
    /// A required binding is absent.
    MissingBinding,
    /// Free variables of the expression disagree with the certified target.
    FreeVariableMismatch,
    /// Certificate leaf set does not match the lowered expression.
    CertificateBindFailed,
}

impl fmt::Display for ExprError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RegimeDomainMismatch => {
                write!(f, "regime kind disagrees with the distribution domain")
            }
            Self::CaptureOrConflict => {
                write!(f, "substitution would capture a bound variable or conflict")
            }
            Self::MissingBinding => write!(f, "required binding is absent"),
            Self::FreeVariableMismatch => {
                write!(f, "expression free variables disagree with the certified target")
            }
            Self::CertificateBindFailed => {
                write!(f, "certificate does not bind the lowered expression")
            }
        }
    }
}

impl std::error::Error for ExprError {}

/// Separate derivation metadata keyed by expression id.
///
/// Off the semantic hash: the same [`ExprId`] may carry different traces.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DerivationMeta {
    /// Named rule (e.g. `backdoor.adjustment`, `transport.sid.direct`).
    pub rule: Arc<str>,
    /// Optional display note. Projection of the typed record, not an equality key.
    pub note: Option<Arc<str>>,
    /// Input subproblem, when this node was derived from one.
    pub input: Option<ExprId>,
    /// Output subproblem (usually the node this metadata is attached to).
    pub output: Option<ExprId>,
    /// Optional graph operation tag (`mutilate`, `ancestry`, …).
    pub graph_operation: Option<Arc<str>>,
    /// Checked premises.
    pub premises: Arc<[Arc<str>]>,
    /// Evidence regimes this step depends on.
    pub evidence: Arc<[RegimeId]>,
    /// Parent derivation nodes.
    pub parents: Arc<[ExprId]>,
}

impl DerivationMeta {
    /// Rule-only metadata (existing builders).
    #[must_use]
    pub fn rule(rule: impl Into<Arc<str>>, note: Option<Arc<str>>) -> Self {
        Self { rule: rule.into(), note, ..Self::default() }
    }

    /// Display projection of the typed record (not an equality key).
    #[must_use]
    pub fn pretty(&self) -> String {
        match &self.note {
            Some(note) => format!("{}: {note}", self.rule),
            None => self.rule.to_string(),
        }
    }
}

/// Arena for causal expressions with interned variable sets.
#[derive(Clone, Debug, Default)]
pub struct CausalExprArena {
    nodes: Vec<ExprNode>,
    var_sets: Vec<Arc<[VariableId]>>,
    var_set_index: HashMap<Arc<[VariableId]>, VarSetId>,
    interventions: Vec<Arc<[InterventionAssignment]>>,
    intervention_index: HashMap<Arc<[InterventionAssignment]>, InterventionSetId>,
    lists: Vec<Arc<[ExprId]>>,
    list_index: HashMap<Arc<[ExprId]>, ExprListId>,
    /// Hash-cons map from node → id.
    node_index: HashMap<ExprNode, ExprId>,
    /// Derivation metadata (optional; not part of semantic equality).
    derivation: HashMap<u32, DerivationMeta>,
    /// Cached id of the interned empty variable set. The empty sets are
    /// requested by nearly every builder and evaluator; caching skips the
    /// intern-table lookup on repeat calls.
    empty_var_set_id: Option<VarSetId>,
    /// Cached id of the interned empty intervention set (see above).
    empty_intervention_set_id: Option<InterventionSetId>,
    populations: Vec<Arc<str>>,
    population_index: HashMap<Arc<str>, PopulationKeyId>,
    empty_population_id: Option<PopulationKeyId>,
}

impl CausalExprArena {
    /// Empty arena.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern a sorted variable set (sorts and dedups input).
    pub fn intern_var_set(&mut self, vars: impl IntoIterator<Item = VariableId>) -> VarSetId {
        let mut v: Vec<VariableId> = vars.into_iter().collect();
        v.sort_unstable();
        v.dedup();
        // Borrow-based lookup: allocate the Arc key only on a cache miss.
        if let Some(id) = self.var_set_index.get(v.as_slice()) {
            return *id;
        }
        let key: Arc<[VariableId]> = Arc::from(v);
        let id = VarSetId(u32::try_from(self.var_sets.len()).expect("var set id"));
        self.var_sets.push(Arc::clone(&key));
        self.var_set_index.insert(key, id);
        id
    }

    /// Intern a hard-intervention assignment set (sorted by variable id).
    pub fn intern_intervention_assignments(
        &mut self,
        assignments: impl IntoIterator<Item = InterventionAssignment>,
    ) -> InterventionSetId {
        let mut v: Vec<InterventionAssignment> = assignments.into_iter().collect();
        v.sort_by_key(|a| a.variable.raw());
        v.dedup_by_key(|a| a.variable.raw());
        if let Some(id) = self.intervention_index.get(v.as_slice()) {
            return *id;
        }
        let key: Arc<[InterventionAssignment]> = Arc::from(v);
        let id = InterventionSetId(u32::try_from(self.interventions.len()).expect("id"));
        self.interventions.push(Arc::clone(&key));
        self.intervention_index.insert(key, id);
        id
    }

    /// Intern an intervention over variables only (symbolic / unspecified level).
    pub fn intern_intervention_set(
        &mut self,
        vars: impl IntoIterator<Item = VariableId>,
    ) -> InterventionSetId {
        self.intern_intervention_assignments(vars.into_iter().map(InterventionAssignment::symbolic))
    }

    /// Empty var set.
    pub fn empty_var_set(&mut self) -> VarSetId {
        if let Some(id) = self.empty_var_set_id {
            return id;
        }
        let id = self.intern_var_set([]);
        self.empty_var_set_id = Some(id);
        id
    }

    /// Intern a population key. The empty string is the default single-study label.
    pub fn intern_population(&mut self, key: impl Into<Arc<str>>) -> PopulationKeyId {
        let key = key.into();
        if let Some(id) = self.population_index.get(&key) {
            return *id;
        }
        let id = PopulationKeyId(u32::try_from(self.populations.len()).expect("population id"));
        self.populations.push(Arc::clone(&key));
        self.population_index.insert(key, id);
        id
    }

    /// Empty / default population key.
    pub fn empty_population(&mut self) -> PopulationKeyId {
        if let Some(id) = self.empty_population_id {
            return id;
        }
        let id = self.intern_population("");
        self.empty_population_id = Some(id);
        id
    }

    /// Borrow an interned population key.
    #[must_use]
    pub fn population(&self, id: PopulationKeyId) -> &str {
        &self.populations[id.0 as usize]
    }

    /// Number of interned population keys (for serialization).
    #[must_use]
    pub fn population_count(&self) -> usize {
        self.populations.len()
    }

    /// Hash-cons a default (single-study) distribution leaf.
    pub fn intern_distribution(
        &mut self,
        variables: VarSetId,
        conditioned_on: VarSetId,
        intervention: InterventionSetId,
        domain: DomainRef,
    ) -> ExprId {
        let population = self.empty_population();
        self.intern(ExprNode::Distribution {
            variables,
            conditioned_on,
            intervention,
            domain,
            population,
            regime: None,
        })
    }

    /// Hash-cons a population- and regime-tagged distribution leaf.
    ///
    /// # Errors
    ///
    /// [`ExprError::RegimeDomainMismatch`] when `regime_kind` disagrees with `domain`.
    #[allow(clippy::too_many_arguments)]
    pub fn intern_distribution_tagged(
        &mut self,
        variables: VarSetId,
        conditioned_on: VarSetId,
        intervention: InterventionSetId,
        domain: DomainRef,
        population: impl Into<Arc<str>>,
        regime: Option<RegimeId>,
        regime_kind: Option<DomainRef>,
    ) -> Result<ExprId, ExprError> {
        if let (Some(_), Some(kind)) = (regime, regime_kind) {
            if kind != domain {
                return Err(ExprError::RegimeDomainMismatch);
            }
        }
        let population = self.intern_population(population);
        Ok(self.intern(ExprNode::Distribution {
            variables,
            conditioned_on,
            intervention,
            domain,
            population,
            regime,
        }))
    }

    /// Hash-cons an intermediate kernel.
    pub fn intern_kernel(
        &mut self,
        body: ExprId,
        bound: VarSetId,
        population: impl Into<Arc<str>>,
        regime: Option<RegimeId>,
    ) -> ExprId {
        let population = self.intern_population(population);
        self.intern(ExprNode::Kernel { body, bound, population, regime })
    }

    /// Empty intervention set.
    pub fn empty_intervention_set(&mut self) -> InterventionSetId {
        if let Some(id) = self.empty_intervention_set_id {
            return id;
        }
        let id = self.intern_intervention_assignments([]);
        self.empty_intervention_set_id = Some(id);
        id
    }

    /// Look up a var set.
    #[must_use]
    pub fn var_set(&self, id: VarSetId) -> &[VariableId] {
        &self.var_sets[id.0 as usize]
    }

    /// Look up intervention assignments.
    #[must_use]
    pub fn intervention_assignments(&self, id: InterventionSetId) -> &[InterventionAssignment] {
        &self.interventions[id.0 as usize]
    }

    /// Variables appearing in an intervention set (legacy helper).
    #[must_use]
    pub fn intervention_set(&self, id: InterventionSetId) -> Vec<VariableId> {
        self.intervention_assignments(id).iter().map(|a| a.variable).collect()
    }

    /// Intern an expression list.
    pub fn intern_list(&mut self, exprs: impl IntoIterator<Item = ExprId>) -> ExprListId {
        let v: Vec<ExprId> = exprs.into_iter().collect();
        if let Some(id) = self.list_index.get(v.as_slice()) {
            return *id;
        }
        let key: Arc<[ExprId]> = Arc::from(v);
        let id = ExprListId(u32::try_from(self.lists.len()).expect("list id"));
        self.lists.push(Arc::clone(&key));
        self.list_index.insert(key, id);
        id
    }

    /// Borrow an interned expression list.
    #[must_use]
    pub fn list(&self, id: ExprListId) -> &[ExprId] {
        &self.lists[id.0 as usize]
    }

    /// Hash-cons an expression node.
    pub fn intern(&mut self, node: ExprNode) -> ExprId {
        if let Some(id) = self.node_index.get(&node) {
            return *id;
        }
        let id = ExprId(u32::try_from(self.nodes.len()).expect("expr id"));
        self.nodes.push(node.clone());
        self.node_index.insert(node, id);
        id
    }

    /// Attach derivation metadata (does not affect semantic equality).
    pub fn set_derivation(&mut self, id: ExprId, meta: DerivationMeta) {
        self.derivation.insert(id.0, meta);
    }

    /// Attach derivation metadata only when absent (never overwrites ID rules).
    pub fn set_derivation_if_absent(&mut self, id: ExprId, meta: DerivationMeta) {
        self.derivation.entry(id.0).or_insert(meta);
    }

    /// Simplify `root` with worklist-style bottom-up rewrite + memoization.
    ///
    /// # Errors
    ///
    /// [`SimplifyError`] if a `SumOut`/`IntegralOut` binds a variable absent from its
    /// body's free variables — an ill-formed estimand. See [`SimplifyError`] docs.
    pub fn simplify(&mut self, root: ExprId) -> Result<ExprId, SimplifyError> {
        simplify::simplify(self, root)
    }

    /// Borrow derivation metadata.
    #[must_use]
    pub fn derivation(&self, id: ExprId) -> Option<&DerivationMeta> {
        self.derivation.get(&id.0)
    }

    /// Borrow a node.
    #[must_use]
    pub fn node(&self, id: ExprId) -> &ExprNode {
        &self.nodes[id.0 as usize]
    }

    /// Number of nodes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Number of interned variable sets (for serialization).
    #[must_use]
    pub fn var_set_count(&self) -> usize {
        self.var_sets.len()
    }

    /// Number of interned intervention sets (for serialization).
    #[must_use]
    pub fn intervention_set_count(&self) -> usize {
        self.interventions.len()
    }

    /// Number of interned expression lists (for serialization).
    #[must_use]
    pub fn list_count(&self) -> usize {
        self.lists.len()
    }

    /// Build the backdoor adjustment functional for ATE:
    /// `E[Y | do(T=active)] − E[Y | do(T=control)]` under adjustment by Z.
    pub fn backdoor_ate(
        &mut self,
        treatment: VariableId,
        outcome: VariableId,
        adjustment: &[VariableId],
        active: Value,
        control: Value,
    ) -> ExprId {
        let left = self.backdoor_potential_outcome(treatment, outcome, adjustment, active);
        let right = self.backdoor_potential_outcome(treatment, outcome, adjustment, control);
        let contrast = self.intern(ExprNode::Contrast { left, right, op: ContrastOp::Difference });
        self.set_derivation(
            contrast,
            DerivationMeta::rule(
                "backdoor.adjustment",
                Some(Arc::from(format!("ATE adjustment set size {}", adjustment.len()))),
            ),
        );
        contrast
    }

    /// Build the backdoor adjustment functional for a single-arm intervention mean:
    /// `E[Y | do(T=level)]` under adjustment by Z.
    ///
    /// This is the staged object for a requested [`antecedent_core::Intervention::Set`],
    /// not an active−control contrast with the query label swapped.
    pub fn backdoor_mean(
        &mut self,
        treatment: VariableId,
        outcome: VariableId,
        adjustment: &[VariableId],
        level: Value,
    ) -> ExprId {
        let mean = self.backdoor_potential_outcome(treatment, outcome, adjustment, level);
        self.set_derivation(
            mean,
            DerivationMeta::rule(
                "backdoor.adjustment",
                Some(Arc::from(format!(
                    "single-arm intervention mean, adjustment set size {}",
                    adjustment.len()
                ))),
            ),
        );
        mean
    }

    fn backdoor_potential_outcome(
        &mut self,
        treatment: VariableId,
        outcome: VariableId,
        adjustment: &[VariableId],
        level: Value,
    ) -> ExprId {
        let z = self.intern_var_set(adjustment.iter().copied());
        let y = self.intern_var_set([outcome]);
        let empty = self.empty_var_set();
        let empty_i = self.empty_intervention_set();
        let do_t = self.intern_intervention_assignments([InterventionAssignment {
            variable: treatment,
            value: level,
        }]);

        let dist_body = self.intern_distribution(y, z, do_t, DomainRef::Interventional);
        let z_marg = self.intern_distribution(z, empty, empty_i, DomainRef::Observational);
        let product = {
            let list = self.intern_list([dist_body, z_marg]);
            self.intern(ExprNode::Product(list))
        };
        let summed = self.intern(ExprNode::SumOut { variables: z, expr: product });
        self.intern(ExprNode::Expectation {
            function: OutcomeExprId::identity(outcome),
            distribution: summed,
        })
    }

    /// Build the front-door functional for ATE:
    /// `E[Y | do(T=active)] − E[Y | do(T=control)]`, mediated through `M` via
    /// `sum_m P(m | t) * sum_t' P(y | m, t') P(t')` (FD condition 2 reduces
    /// `P(m | do(t))` to the observational `P(m | t)`).
    pub fn frontdoor_ate(
        &mut self,
        treatment: VariableId,
        outcome: VariableId,
        mediators: &[VariableId],
        active: Value,
        control: Value,
    ) -> ExprId {
        let left = self.frontdoor_potential_outcome(treatment, outcome, mediators, active);
        let right = self.frontdoor_potential_outcome(treatment, outcome, mediators, control);
        let contrast = self.intern(ExprNode::Contrast { left, right, op: ContrastOp::Difference });
        self.set_derivation(
            contrast,
            DerivationMeta::rule(
                "frontdoor",
                Some(Arc::from(format!("front-door mediator set size {}", mediators.len()))),
            ),
        );
        contrast
    }

    /// Linear temporal-mediation path-product ATE contrast (same product-of-coefficients
    /// geometry as front-door under a linear SEM, tagged `temporal_mediation` — not front-door).
    pub fn temporal_mediation_ate(
        &mut self,
        treatment: VariableId,
        outcome: VariableId,
        mediators: &[VariableId],
        active: Value,
        control: Value,
    ) -> ExprId {
        let left = self.frontdoor_potential_outcome(treatment, outcome, mediators, active);
        let right = self.frontdoor_potential_outcome(treatment, outcome, mediators, control);
        let contrast = self.intern(ExprNode::Contrast { left, right, op: ContrastOp::Difference });
        self.set_derivation(
            contrast,
            DerivationMeta::rule(
                "temporal_mediation",
                Some(Arc::from(format!(
                    "linear temporal mediation path-product; mediator set size {}",
                    mediators.len()
                ))),
            ),
        );
        contrast
    }

    fn frontdoor_potential_outcome(
        &mut self,
        treatment: VariableId,
        outcome: VariableId,
        mediators: &[VariableId],
        level: Value,
    ) -> ExprId {
        let m = self.intern_var_set(mediators.iter().copied());
        let y = self.intern_var_set([outcome]);
        let t = self.intern_var_set([treatment]);
        let m_and_t = self.intern_var_set(mediators.iter().copied().chain([treatment]));
        let empty = self.empty_var_set();
        let empty_i = self.empty_intervention_set();
        let do_t = self.intern_intervention_assignments([InterventionAssignment {
            variable: treatment,
            value: level,
        }]);

        // P(m | t): observational under FD condition 2; treatment level bound so
        // the evaluator treats it as fixed (not free).
        let m_given_t = self.intern_distribution(m, t, do_t, DomainRef::Observational);
        // P(y | m, t').
        let y_given_m_t = self.intern_distribution(y, m_and_t, empty_i, DomainRef::Observational);
        // P(t').
        let t_marginal = self.intern_distribution(t, empty, empty_i, DomainRef::Observational);
        let inner_product = {
            let list = self.intern_list([y_given_m_t, t_marginal]);
            self.intern(ExprNode::Product(list))
        };
        let inner_summed = self.intern(ExprNode::SumOut { variables: t, expr: inner_product });
        let outer_product = {
            let list = self.intern_list([m_given_t, inner_summed]);
            self.intern(ExprNode::Product(list))
        };
        let outer_summed = self.intern(ExprNode::SumOut { variables: m, expr: outer_product });
        self.intern(ExprNode::Expectation {
            function: OutcomeExprId::identity(outcome),
            distribution: outer_summed,
        })
    }

    /// Build the Wald IV functional for binary instrument `Z`:
    /// `(E[Y|Z=1] − E[Y|Z=0]) / (E[T|Z=1] − E[T|Z=0])`.
    ///
    /// `active` / `control` are recorded in derivation metadata (treatment contrast
    /// scaling); the ratio itself conditions on instrument levels 1 and 0.
    pub fn iv_wald(
        &mut self,
        treatment: VariableId,
        outcome: VariableId,
        instruments: &[VariableId],
        active: &Value,
        control: &Value,
    ) -> ExprId {
        let z = instruments.first().copied().unwrap_or(treatment);
        let z1 = Value::f64(1.0);
        let z0 = Value::f64(0.0);
        let outcome_given_z1 = self.observational_conditional_mean(outcome, z, z1.clone());
        let outcome_given_z0 = self.observational_conditional_mean(outcome, z, z0.clone());
        let treatment_given_z1 = self.observational_conditional_mean(treatment, z, z1);
        let treatment_given_z0 = self.observational_conditional_mean(treatment, z, z0);
        let num = self.intern(ExprNode::Contrast {
            left: outcome_given_z1,
            right: outcome_given_z0,
            op: ContrastOp::Difference,
        });
        let den = self.intern(ExprNode::Contrast {
            left: treatment_given_z1,
            right: treatment_given_z0,
            op: ContrastOp::Difference,
        });
        let ratio = self.intern(ExprNode::Ratio { numerator: num, denominator: den });
        self.set_derivation(
            ratio,
            DerivationMeta::rule(
                "iv.wald",
                Some(Arc::from(format!(
                    "Wald IV ratio using {} instrument(s); treatment contrast [{active:?}, {control:?}]",
                    instruments.len()
                ))),
            ),
        );
        ratio
    }

    /// Sharp regression-discontinuity functional: the effect for units at the cutoff,
    /// `lim_{r↓c} E[Y | R = r] − lim_{r↑c} E[Y | R = r]`.
    ///
    /// With `T = 1{R ≥ c}` each one-sided limit is the boundary value at `R = c` of that
    /// side's observational regression, so the contrast is written
    /// `E[Y | T = active, R = c] − E[Y | T = control, R = c]`. The control-side cell has no
    /// support at `R = c`; it denotes the continuous extension of `E[Y | T = control, R = r]`
    /// to the cutoff. Nothing here is interventional and nothing averages over `R`: this is
    /// not the unadjusted contrast `E[Y | T = active] − E[Y | T = control]`.
    pub fn rd_sharp_local_effect(
        &mut self,
        treatment: VariableId,
        outcome: VariableId,
        running: VariableId,
        cutoff: f64,
        active: Value,
        control: Value,
    ) -> ExprId {
        let above = self.boundary_conditional_mean(outcome, treatment, active, running, cutoff);
        let below = self.boundary_conditional_mean(outcome, treatment, control, running, cutoff);
        let contrast = self.intern(ExprNode::Contrast {
            left: above,
            right: below,
            op: ContrastOp::Difference,
        });
        self.set_derivation(
            contrast,
            DerivationMeta::rule(
                "rd.sharp",
                Some(Arc::from(format!(
                    "difference of the one-sided limits of E[Y | R = r] at r = {cutoff}; each \
                     side is the boundary value of that treatment arm's regression on R"
                ))),
            ),
        );
        contrast
    }

    /// Observational `E[outcome | arm = level, running = cutoff]`, both levels bound.
    fn boundary_conditional_mean(
        &mut self,
        outcome: VariableId,
        arm: VariableId,
        level: Value,
        running: VariableId,
        cutoff: f64,
    ) -> ExprId {
        let y = self.intern_var_set([outcome]);
        let given = self.intern_var_set([arm, running]);
        let bind = self.intern_intervention_assignments([
            InterventionAssignment { variable: arm, value: level },
            InterventionAssignment { variable: running, value: Value::f64(cutoff) },
        ]);
        let dist = self.intern_distribution(y, given, bind, DomainRef::Observational);
        self.intern(ExprNode::Expectation {
            function: OutcomeExprId::identity(outcome),
            distribution: dist,
        })
    }

    /// Observational `E[outcome | conditioner = level]`.
    ///
    /// The conditioning level is bound via an intervention assignment so the
    /// evaluator treats it as fixed (not free), while the factor remains
    /// observational `P(outcome | conditioner)`.
    fn observational_conditional_mean(
        &mut self,
        outcome: VariableId,
        conditioner: VariableId,
        level: Value,
    ) -> ExprId {
        let y = self.intern_var_set([outcome]);
        let z = self.intern_var_set([conditioner]);
        let bind = self.intern_intervention_assignments([InterventionAssignment {
            variable: conditioner,
            value: level,
        }]);
        let dist = self.intern_distribution(y, z, bind, DomainRef::Observational);
        self.intern(ExprNode::Expectation {
            function: OutcomeExprId::identity(outcome),
            distribution: dist,
        })
    }

    /// Pretty-print an expression (diagnostics only; not an equality key).
    #[must_use]
    pub fn pretty(&self, id: ExprId) -> String {
        pretty_expr(self, id)
    }

    /// Render an expression as LaTeX (diagnostics only; not an equality key).
    #[must_use]
    pub fn latex(&self, id: ExprId) -> String {
        latex_expr(self, id)
    }
}

impl fmt::Display for ExprId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "E{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rd_functional_is_the_boundary_contrast_not_the_unadjusted_one() {
        let (t, y, r) = (VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2));
        let mut a = CausalExprArena::new();
        let rd = a.rd_sharp_local_effect(t, y, r, 0.5, Value::f64(1.0), Value::f64(0.0));
        let naive = a.backdoor_ate(t, y, &[], Value::f64(1.0), Value::f64(0.0));
        assert_ne!(rd, naive);
        let ExprNode::Contrast { left, right, op: ContrastOp::Difference } = a.node(rd).clone()
        else {
            panic!("RD functional must be a difference");
        };
        assert_ne!(left, right);
        for (side, level) in [(left, 1.0), (right, 0.0)] {
            let ExprNode::Expectation { distribution, .. } = a.node(side).clone() else {
                panic!("each side is a conditional mean");
            };
            let ExprNode::Distribution { variables, conditioned_on, intervention, domain, .. } =
                a.node(distribution).clone()
            else {
                panic!("each side is one observational factor");
            };
            assert_eq!(domain, DomainRef::Observational);
            assert_eq!(a.var_set(variables), &[y]);
            // Conditions on the arm AND on the running variable at the cutoff.
            assert_eq!(a.var_set(conditioned_on), &[t, r]);
            let bound: Vec<(VariableId, Option<f64>)> = a
                .intervention_assignments(intervention)
                .iter()
                .map(|b| (b.variable, b.value.as_f64()))
                .collect();
            assert!(bound.contains(&(t, Some(level))), "{bound:?}");
            assert!(bound.contains(&(r, Some(0.5))), "{bound:?}");
        }
    }

    #[test]
    fn var_sets_are_sorted_and_interned() {
        let mut a = CausalExprArena::new();
        let s1 = a.intern_var_set([VariableId::from_raw(2), VariableId::from_raw(1)]);
        let s2 = a.intern_var_set([VariableId::from_raw(1), VariableId::from_raw(2)]);
        assert_eq!(s1, s2);
        assert_eq!(a.var_set(s1), &[VariableId::from_raw(1), VariableId::from_raw(2)]);
    }

    #[test]
    fn hash_cons_reuses_nodes() {
        let mut a = CausalExprArena::new();
        let empty = a.empty_var_set();
        let empty_i = a.empty_intervention_set();
        let n1 = a.intern_distribution(empty, empty, empty_i, DomainRef::Observational);
        let n2 = a.intern_distribution(empty, empty, empty_i, DomainRef::Observational);
        assert_eq!(n1, n2);
        assert_eq!(a.len(), 1);
    }

    #[test]
    fn backdoor_ate_contrasts_distinct_levels() {
        let mut a = CausalExprArena::new();
        let id = a.backdoor_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            &[VariableId::from_raw(2)],
            Value::f64(1.0),
            Value::f64(0.0),
        );
        let meta = a.derivation(id).unwrap();
        assert_eq!(&*meta.rule, "backdoor.adjustment");
        let ExprNode::Contrast { left, right, .. } = a.node(id) else {
            panic!("expected contrast");
        };
        assert_ne!(left, right);
        let pretty = a.pretty(id);
        assert!(pretty.contains('−') || pretty.contains("E["));
        let latex = a.latex(id);
        assert!(latex.contains("\\mathbb{E}") || latex.contains("\\mathrm{do}"));
        assert!(latex.contains('-'));
    }

    #[test]
    fn backdoor_mean_is_expectation_not_contrast() {
        let mut a = CausalExprArena::new();
        let id = a.backdoor_mean(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            &[VariableId::from_raw(2)],
            Value::f64(0.0),
        );
        assert!(
            !matches!(a.node(id), ExprNode::Contrast { .. }),
            "single-arm mean must not persist as an ATE contrast"
        );
        assert!(matches!(a.node(id), ExprNode::Expectation { .. }));
        let pretty = a.pretty(id);
        assert!(pretty.contains("do(") && pretty.contains('0'), "{pretty}");
        assert!(!pretty.contains('−'), "{pretty}");
    }

    #[test]
    fn frontdoor_ate_contrasts_distinct_levels() {
        let mut a = CausalExprArena::new();
        let id = a.frontdoor_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            &[VariableId::from_raw(2)],
            Value::f64(1.0),
            Value::f64(0.0),
        );
        let meta = a.derivation(id).unwrap();
        assert_eq!(&*meta.rule, "frontdoor");
        let ExprNode::Contrast { left, right, .. } = a.node(id) else {
            panic!("expected contrast");
        };
        assert_ne!(left, right);
    }

    #[test]
    fn iv_wald_is_ratio_of_instrument_contrasts() {
        let mut a = CausalExprArena::new();
        let id = a.iv_wald(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            &[VariableId::from_raw(2)],
            &Value::f64(1.0),
            &Value::f64(0.0),
        );
        let meta = a.derivation(id).unwrap();
        assert_eq!(&*meta.rule, "iv.wald");
        let ExprNode::Ratio { numerator, denominator } = a.node(id) else {
            panic!("expected Wald ratio");
        };
        assert!(matches!(a.node(*numerator), ExprNode::Contrast { .. }));
        assert!(matches!(a.node(*denominator), ExprNode::Contrast { .. }));
        assert_ne!(*numerator, *denominator);
    }

    #[test]
    fn existing_builders_keep_empty_population_identity() {
        let mut a = CausalExprArena::new();
        let left = a.backdoor_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            &[VariableId::from_raw(2)],
            Value::f64(1.0),
            Value::f64(0.0),
        );
        let right = a.backdoor_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            &[VariableId::from_raw(2)],
            Value::f64(1.0),
            Value::f64(0.0),
        );
        assert_eq!(left, right);
        assert!(a.leaf_bindings(left).iter().all(|b| b.population.as_ref().is_empty()));
    }

    #[test]
    fn substitute_rejects_missing_binding() {
        let mut a = CausalExprArena::new();
        let y = a.intern_var_set([VariableId::from_raw(1)]);
        let empty = a.empty_var_set();
        let empty_i = a.empty_intervention_set();
        let id = a.intern_distribution(y, empty, empty_i, DomainRef::Observational);
        let err =
            a.substitute(id, &[(VariableId::from_raw(9), VariableId::from_raw(8))]).unwrap_err();
        assert_eq!(err, ExprError::MissingBinding);
    }

    #[test]
    fn identical_symbolic_intervention_sets_intern_to_same_id() {
        let mut arena = CausalExprArena::new();
        let t = VariableId::from_raw(0);
        let z = VariableId::from_raw(2);
        let a = arena.intern_intervention_set([t, z]);
        let b = arena.intern_intervention_set([z, t]);
        assert_eq!(a, b, "symbolic sets must hash-cons (NaN sentinel never did)");
        let assignments = arena.intervention_assignments(a);
        assert_eq!(assignments.len(), 2);
        assert!(assignments.iter().all(InterventionAssignment::is_symbolic));
        assert!(!assignments.iter().any(|x| matches!(
            x.value,
            Value::Float64(v) if v.is_nan()
        )));
    }
}
