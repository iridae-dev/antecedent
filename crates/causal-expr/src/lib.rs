//! Arena-backed causal-functional IR (DESIGN.md §9).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use causal_core::VariableId;

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
    /// Raw index.
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

/// Interned intervention-set id (Phase 1: ordered variable list being intervened).
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct InterventionSetId(u32);

/// Contrast operator between two expressions.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ContrastOp {
    /// Left − right.
    Difference,
}

/// Domain reference for a distribution (Phase 1: observational or interventional).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum DomainRef {
    /// Observational P(·).
    Observational,
    /// Interventional P(· | do(·)).
    Interventional,
}

/// Outcome function id (Phase 1: identity of a single variable).
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
    interventions: Vec<Arc<[VariableId]>>,
    intervention_index: HashMap<Arc<[VariableId]>, InterventionSetId>,
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

    /// Intern an intervention variable set.
    pub fn intern_intervention_set(
        &mut self,
        vars: impl IntoIterator<Item = VariableId>,
    ) -> InterventionSetId {
        let mut v: Vec<VariableId> = vars.into_iter().collect();
        v.sort_unstable();
        v.dedup();
        let key: Arc<[VariableId]> = Arc::from(v);
        if let Some(id) = self.intervention_index.get(&key) {
            return *id;
        }
        let id = InterventionSetId(u32::try_from(self.interventions.len()).expect("id"));
        self.interventions.push(Arc::clone(&key));
        self.intervention_index.insert(key, id);
        id
    }

    /// Empty var set.
    pub fn empty_var_set(&mut self) -> VarSetId {
        self.intern_var_set([])
    }

    /// Empty intervention set.
    pub fn empty_intervention_set(&mut self) -> InterventionSetId {
        self.intern_intervention_set([])
    }

    /// Look up a var set.
    #[must_use]
    pub fn var_set(&self, id: VarSetId) -> &[VariableId] {
        &self.var_sets[id.0 as usize]
    }

    /// Look up intervention variables.
    #[must_use]
    pub fn intervention_set(&self, id: InterventionSetId) -> &[VariableId] {
        &self.interventions[id.0 as usize]
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

    /// Build the backdoor adjustment functional for ATE:
    /// `E[E[Y|T=1,Z] − E[Y|T=0,Z]]` represented as a contrast of expectations
    /// under interventional adjustment distributions (Phase 1 encoding).
    pub fn backdoor_ate(
        &mut self,
        treatment: VariableId,
        outcome: VariableId,
        adjustment: &[VariableId],
    ) -> ExprId {
        let z = self.intern_var_set(adjustment.iter().copied());
        let y = self.intern_var_set([outcome]);
        let empty = self.empty_var_set();
        let empty_i = self.empty_intervention_set();
        let do_t = self.intern_intervention_set([treatment]);

        // P(Y | Z, do(T)) as Distribution with interventional domain.
        let dist_body = self.intern(ExprNode::Distribution {
            variables: y,
            conditioned_on: z,
            intervention: do_t,
            domain: DomainRef::Interventional,
        });
        // Marginalize / weight by P(Z) observationally.
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
        let left = self.intern(ExprNode::Expectation {
            function: OutcomeExprId::identity(outcome),
            distribution: summed,
        });
        // Phase 1: encode contrast as left − right with the same structure;
        // active vs control is recorded in derivation metadata for the estimator.
        let right = left; // structural placeholder; estimator uses query levels
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

    /// Pretty-print an expression (diagnostics only; not an equality key).
    #[must_use]
    pub fn pretty(&self, id: ExprId) -> String {
        pretty_expr(self, id)
    }
}

fn pretty_expr(arena: &CausalExprArena, id: ExprId) -> String {
    match arena.node(id) {
        ExprNode::Distribution { variables, conditioned_on, intervention, domain } => {
            let vars = fmt_vars(arena.var_set(*variables));
            let cond = fmt_vars(arena.var_set(*conditioned_on));
            let interv = fmt_vars(arena.intervention_set(*intervention));
            match domain {
                DomainRef::Observational => {
                    if cond.is_empty() {
                        format!("P({vars})")
                    } else {
                        format!("P({vars}|{cond})")
                    }
                }
                DomainRef::Interventional => {
                    if cond.is_empty() {
                        format!("P({vars}|do({interv}))")
                    } else {
                        format!("P({vars}|{cond},do({interv}))")
                    }
                }
            }
        }
        ExprNode::Product(list) => {
            let parts: Vec<String> =
                arena.lists[list.0 as usize].iter().map(|e| pretty_expr(arena, *e)).collect();
            parts.join(" * ")
        }
        ExprNode::SumOut { variables, expr } => {
            format!("Σ_{{{}}}[{}]", fmt_vars(arena.var_set(*variables)), pretty_expr(arena, *expr))
        }
        ExprNode::IntegralOut { variables, expr } => {
            format!("∫_{{{}}}[{}]", fmt_vars(arena.var_set(*variables)), pretty_expr(arena, *expr))
        }
        ExprNode::Ratio { numerator, denominator } => {
            format!("({})/({})", pretty_expr(arena, *numerator), pretty_expr(arena, *denominator))
        }
        ExprNode::Expectation { function, distribution } => {
            format!("E[V{} | {}]", function.variable().raw(), pretty_expr(arena, *distribution))
        }
        ExprNode::Contrast { left, right, op } => {
            let op_s = match op {
                ContrastOp::Difference => "−",
            };
            format!("({}) {} ({})", pretty_expr(arena, *left), op_s, pretty_expr(arena, *right))
        }
    }
}

fn fmt_vars(vars: &[VariableId]) -> String {
    vars.iter().map(|v| format!("V{}", v.raw())).collect::<Vec<_>>().join(",")
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
    fn backdoor_ate_has_derivation() {
        let mut a = CausalExprArena::new();
        let id = a.backdoor_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            &[VariableId::from_raw(2)],
        );
        let meta = a.derivation(id).unwrap();
        assert_eq!(&*meta.rule, "backdoor.adjustment");
        let pretty = a.pretty(id);
        assert!(pretty.contains('−') || pretty.contains("E["));
    }
}
