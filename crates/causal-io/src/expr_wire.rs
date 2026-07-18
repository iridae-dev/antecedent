//! Causal expression arena wire types.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use causal_core::VariableId;
use causal_expr::{
    CausalExprArena, ContrastOp, DomainRef, ExprId, ExprListId, ExprNode, InterventionAssignment,
    InterventionSetId, OutcomeExprId, VarSetId,
};
use serde::{Deserialize, Serialize};

use crate::error::IoError;
use crate::query_wire::ValueWire;

/// Expr arena wire (tables only; hash indexes rebuilt on load).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ExprArenaWire {
    /// Variable sets: list of variable raw ids.
    pub var_sets: Vec<Vec<u32>>,
    /// Intervention assignment sets.
    pub interventions: Vec<Vec<InterventionAssignmentWire>>,
    /// Expression lists (raw ExprIds).
    pub lists: Vec<Vec<u32>>,
    /// Expression nodes in id order.
    pub nodes: Vec<ExprNodeWire>,
}

/// Intervention assignment wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct InterventionAssignmentWire {
    /// Variable.
    pub variable: u32,
    /// Value.
    pub value: ValueWire,
}

/// Expression node wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ExprNodeWire {
    /// Distribution factor.
    Distribution {
        /// Variables.
        variables: u32,
        /// Conditioning set.
        conditioned_on: u32,
        /// Intervention set.
        intervention: u32,
        /// Domain tag.
        domain: String,
    },
    /// Product.
    Product(u32),
    /// Sum-out.
    SumOut {
        /// Variables.
        variables: u32,
        /// Body.
        expr: u32,
    },
    /// Integral-out.
    IntegralOut {
        /// Variables.
        variables: u32,
        /// Body.
        expr: u32,
    },
    /// Ratio.
    Ratio {
        /// Numerator.
        numerator: u32,
        /// Denominator.
        denominator: u32,
    },
    /// Expectation.
    Expectation {
        /// Outcome variable.
        function: u32,
        /// Distribution.
        distribution: u32,
    },
    /// Contrast.
    Contrast {
        /// Left.
        left: u32,
        /// Right.
        right: u32,
        /// Op.
        op: String,
    },
}

/// Encode arena.
///
/// # Errors
///
/// Arena indexes that do not fit in `u32`.
pub fn expr_arena_to_wire(arena: &CausalExprArena) -> Result<ExprArenaWire, IoError> {
    let mut var_sets = Vec::with_capacity(arena.var_set_count());
    for i in 0..arena.var_set_count() {
        let id = VarSetId::from_raw(u32::try_from(i).map_err(|_| IoError::TooLarge)?);
        var_sets.push(arena.var_set(id).iter().map(|v| v.raw()).collect());
    }
    let mut interventions = Vec::with_capacity(arena.intervention_set_count());
    for i in 0..arena.intervention_set_count() {
        let id = InterventionSetId::from_raw(u32::try_from(i).map_err(|_| IoError::TooLarge)?);
        interventions.push(
            arena
                .intervention_assignments(id)
                .iter()
                .map(|a| InterventionAssignmentWire {
                    variable: a.variable.raw(),
                    value: ValueWire::from_value(&a.value),
                })
                .collect(),
        );
    }
    let mut lists = Vec::with_capacity(arena.list_count());
    for i in 0..arena.list_count() {
        let id = ExprListId::from_raw(u32::try_from(i).map_err(|_| IoError::TooLarge)?);
        lists.push(arena.list(id).iter().map(|e| e.raw()).collect());
    }
    let mut nodes = Vec::with_capacity(arena.len());
    for i in 0..arena.len() {
        let id = ExprId::from_raw(u32::try_from(i).map_err(|_| IoError::TooLarge)?);
        nodes.push(node_to_wire(arena.node(id)));
    }
    Ok(ExprArenaWire { var_sets, interventions, lists, nodes })
}

/// Decode arena by re-interning in order.
///
/// # Errors
///
/// Unknown tags or malformed indexes.
pub fn expr_arena_from_wire(w: &ExprArenaWire) -> Result<CausalExprArena, IoError> {
    let mut arena = CausalExprArena::new();
    for vs in &w.var_sets {
        let _ = arena.intern_var_set(vs.iter().copied().map(VariableId::from_raw));
    }
    for iv in &w.interventions {
        let _ = arena.intern_intervention_assignments(iv.iter().map(|a| InterventionAssignment {
            variable: VariableId::from_raw(a.variable),
            value: a.value.to_value(),
        }));
    }
    for list in &w.lists {
        let _ = arena.intern_list(list.iter().copied().map(ExprId::from_raw));
    }
    for node in &w.nodes {
        let _ = arena.intern(node_from_wire(node)?);
    }
    Ok(arena)
}

fn node_to_wire(n: &ExprNode) -> ExprNodeWire {
    match n {
        ExprNode::Distribution { variables, conditioned_on, intervention, domain } => {
            ExprNodeWire::Distribution {
                variables: variables.raw(),
                conditioned_on: conditioned_on.raw(),
                intervention: intervention.raw(),
                domain: match domain {
                    DomainRef::Observational => "observational".into(),
                    DomainRef::Interventional => "interventional".into(),
                },
            }
        }
        ExprNode::Product(list) => ExprNodeWire::Product(list.raw()),
        ExprNode::SumOut { variables, expr } => {
            ExprNodeWire::SumOut { variables: variables.raw(), expr: expr.raw() }
        }
        ExprNode::IntegralOut { variables, expr } => {
            ExprNodeWire::IntegralOut { variables: variables.raw(), expr: expr.raw() }
        }
        ExprNode::Ratio { numerator, denominator } => ExprNodeWire::Ratio {
            numerator: numerator.raw(),
            denominator: denominator.raw(),
        },
        ExprNode::Expectation { function, distribution } => ExprNodeWire::Expectation {
            function: function.variable().raw(),
            distribution: distribution.raw(),
        },
        ExprNode::Contrast { left, right, op } => ExprNodeWire::Contrast {
            left: left.raw(),
            right: right.raw(),
            op: match op {
                ContrastOp::Difference => "difference".into(),
            },
        },
    }
}

fn node_from_wire(n: &ExprNodeWire) -> Result<ExprNode, IoError> {
    Ok(match n {
        ExprNodeWire::Distribution { variables, conditioned_on, intervention, domain } => {
            ExprNode::Distribution {
                variables: VarSetId::from_raw(*variables),
                conditioned_on: VarSetId::from_raw(*conditioned_on),
                intervention: InterventionSetId::from_raw(*intervention),
                domain: match domain.as_str() {
                    "observational" => DomainRef::Observational,
                    "interventional" => DomainRef::Interventional,
                    other => {
                        return Err(IoError::Convert(format!("unknown DomainRef `{other}`")));
                    }
                },
            }
        }
        ExprNodeWire::Product(list) => ExprNode::Product(ExprListId::from_raw(*list)),
        ExprNodeWire::SumOut { variables, expr } => ExprNode::SumOut {
            variables: VarSetId::from_raw(*variables),
            expr: ExprId::from_raw(*expr),
        },
        ExprNodeWire::IntegralOut { variables, expr } => ExprNode::IntegralOut {
            variables: VarSetId::from_raw(*variables),
            expr: ExprId::from_raw(*expr),
        },
        ExprNodeWire::Ratio { numerator, denominator } => ExprNode::Ratio {
            numerator: ExprId::from_raw(*numerator),
            denominator: ExprId::from_raw(*denominator),
        },
        ExprNodeWire::Expectation { function, distribution } => ExprNode::Expectation {
            function: OutcomeExprId::identity(VariableId::from_raw(*function)),
            distribution: ExprId::from_raw(*distribution),
        },
        ExprNodeWire::Contrast { left, right, op } => ExprNode::Contrast {
            left: ExprId::from_raw(*left),
            right: ExprId::from_raw(*right),
            op: match op.as_str() {
                "difference" => ContrastOp::Difference,
                other => return Err(IoError::Convert(format!("unknown ContrastOp `{other}`"))),
            },
        },
    })
}
