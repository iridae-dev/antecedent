//! Pretty-printing for diagnostics (not equality keys).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use causal_core::VariableId;

use crate::{CausalExprArena, ContrastOp, DomainRef, ExprId, ExprNode};

pub(crate) fn pretty_expr(arena: &CausalExprArena, id: ExprId) -> String {
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

