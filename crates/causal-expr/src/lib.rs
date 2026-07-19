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
pub mod latex;
pub mod pretty;
pub mod provider;
pub mod simplify;

pub use estimand::{EstimandMethod, IdentifiedEstimand, RdDesignParams};
pub use eval::CompiledEvaluator;
pub use provider::{
    Assignment, DistributionProvider, EmpiricalTableProvider, EvalContext, EvalError, FactorSpec,
    GaussianDensityProvider, PosteriorDrawProvider,
};

use latex::latex_expr;
use pretty::pretty_expr;

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use causal_core::{Value, VariableId};

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
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct InterventionAssignment {
    /// Target variable.
    pub variable: VariableId,
    /// Assigned value under `do(·)`.
    pub value: Value,
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

/// Separate derivation metadata keyed by expression id.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DerivationMeta {
    /// Human-readable rule tag (e.g. `backdoor.adjustment`).
    pub rule: Arc<str>,
    /// Optional note.
    pub note: Option<Arc<str>>,
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
        let key: Arc<[VariableId]> = Arc::from(v);
        if let Some(id) = self.var_set_index.get(&key) {
            return *id;
        }
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
        let key: Arc<[InterventionAssignment]> = Arc::from(v);
        if let Some(id) = self.intervention_index.get(&key) {
            return *id;
        }
        let id = InterventionSetId(u32::try_from(self.interventions.len()).expect("id"));
        self.interventions.push(Arc::clone(&key));
        self.intervention_index.insert(key, id);
        id
    }

    /// Intern an intervention over variables only (value unspecified / placeholder).
    pub fn intern_intervention_set(
        &mut self,
        vars: impl IntoIterator<Item = VariableId>,
    ) -> InterventionSetId {
        self.intern_intervention_assignments(
            vars.into_iter()
                .map(|variable| InterventionAssignment { variable, value: Value::f64(f64::NAN) }),
        )
    }

    /// Empty var set.
    pub fn empty_var_set(&mut self) -> VarSetId {
        self.intern_var_set([])
    }

    /// Empty intervention set.
    pub fn empty_intervention_set(&mut self) -> InterventionSetId {
        self.intern_intervention_assignments([])
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
        let key: Arc<[ExprId]> = Arc::from(exprs.into_iter().collect::<Vec<_>>());
        if let Some(id) = self.list_index.get(&key) {
            return *id;
        }
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
    pub fn simplify(&mut self, root: ExprId) -> ExprId {
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
            DerivationMeta {
                rule: Arc::from("backdoor.adjustment"),
                note: Some(Arc::from(format!("ATE adjustment set size {}", adjustment.len()))),
            },
        );
        contrast
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

        let dist_body = self.intern(ExprNode::Distribution {
            variables: y,
            conditioned_on: z,
            intervention: do_t,
            domain: DomainRef::Interventional,
        });
        let z_marg = self.intern(ExprNode::Distribution {
            variables: z,
            conditioned_on: empty,
            intervention: empty_i,
            domain: DomainRef::Observational,
        });
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
    /// `sum_m P(m | do(t)) * sum_t' P(y | m, t') P(t')`.
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
            DerivationMeta {
                rule: Arc::from("frontdoor"),
                note: Some(Arc::from(format!("front-door mediator set size {}", mediators.len()))),
            },
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
            DerivationMeta {
                rule: Arc::from("temporal_mediation"),
                note: Some(Arc::from(format!(
                    "linear temporal mediation path-product; mediator set size {}",
                    mediators.len()
                ))),
            },
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

        // P(m | do(t)).
        let m_given_do_t = self.intern(ExprNode::Distribution {
            variables: m,
            conditioned_on: empty,
            intervention: do_t,
            domain: DomainRef::Interventional,
        });
        // P(y | m, t').
        let y_given_m_t = self.intern(ExprNode::Distribution {
            variables: y,
            conditioned_on: m_and_t,
            intervention: empty_i,
            domain: DomainRef::Observational,
        });
        // P(t').
        let t_marginal = self.intern(ExprNode::Distribution {
            variables: t,
            conditioned_on: empty,
            intervention: empty_i,
            domain: DomainRef::Observational,
        });
        let inner_product = {
            let list = self.intern_list([y_given_m_t, t_marginal]);
            self.intern(ExprNode::Product(list))
        };
        let inner_summed = self.intern(ExprNode::SumOut { variables: t, expr: inner_product });
        let outer_product = {
            let list = self.intern_list([m_given_do_t, inner_summed]);
            self.intern(ExprNode::Product(list))
        };
        let outer_summed = self.intern(ExprNode::SumOut { variables: m, expr: outer_product });
        self.intern(ExprNode::Expectation {
            function: OutcomeExprId::identity(outcome),
            distribution: outer_summed,
        })
    }

    /// Build the Wald IV functional for ATE as a contrast of potential
    /// outcomes with an empty adjustment set; the instrument set is recorded
    /// only in derivation metadata .
    pub fn iv_wald(
        &mut self,
        treatment: VariableId,
        outcome: VariableId,
        instruments: &[VariableId],
        active: Value,
        control: Value,
    ) -> ExprId {
        let left = self.backdoor_potential_outcome(treatment, outcome, &[], active);
        let right = self.backdoor_potential_outcome(treatment, outcome, &[], control);
        let contrast = self.intern(ExprNode::Contrast { left, right, op: ContrastOp::Difference });
        self.set_derivation(
            contrast,
            DerivationMeta {
                rule: Arc::from("iv.wald"),
                note: Some(Arc::from(format!(
                    "Wald IV ratio using {} instrument(s)",
                    instruments.len()
                ))),
            },
        );
        contrast
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
        let n1 = a.intern(ExprNode::Distribution {
            variables: empty,
            conditioned_on: empty,
            intervention: empty_i,
            domain: DomainRef::Observational,
        });
        let n2 = a.intern(ExprNode::Distribution {
            variables: empty,
            conditioned_on: empty,
            intervention: empty_i,
            domain: DomainRef::Observational,
        });
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
    fn iv_wald_contrasts_distinct_levels() {
        let mut a = CausalExprArena::new();
        let id = a.iv_wald(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            &[VariableId::from_raw(2)],
            Value::f64(1.0),
            Value::f64(0.0),
        );
        let meta = a.derivation(id).unwrap();
        assert_eq!(&*meta.rule, "iv.wald");
        let ExprNode::Contrast { left, right, .. } = a.node(id) else {
            panic!("expected contrast");
        };
        assert_ne!(left, right);
    }
}
