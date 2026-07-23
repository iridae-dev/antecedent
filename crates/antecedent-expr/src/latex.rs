//! LaTeX rendering for diagnostics (not equality keys).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{Value, VariableId};

use crate::{CausalExprArena, ContrastOp, DomainRef, ExprId, ExprNode, InterventionAssignment};

pub(crate) fn latex_expr(arena: &CausalExprArena, id: ExprId) -> String {
    match arena.node(id) {
        ExprNode::Distribution { variables, conditioned_on, intervention, domain } => {
            let vars = fmt_vars(arena.var_set(*variables));
            let cond = fmt_vars(arena.var_set(*conditioned_on));
            let interv = fmt_assignments(arena.intervention_assignments(*intervention));
            match domain {
                DomainRef::Observational => {
                    if cond.is_empty() {
                        format!("P({vars})")
                    } else {
                        format!("P({vars}\\mid {cond})")
                    }
                }
                DomainRef::Interventional => {
                    if cond.is_empty() {
                        format!("P({vars}\\mid \\mathrm{{do}}({interv}))")
                    } else {
                        format!("P({vars}\\mid {cond},\\mathrm{{do}}({interv}))")
                    }
                }
            }
        }
        ExprNode::Product(list) => {
            let parts: Vec<String> =
                arena.lists[list.0 as usize].iter().map(|e| latex_expr(arena, *e)).collect();
            parts.join(" \\cdot ")
        }
        ExprNode::SumOut { variables, expr } => {
            format!(
                "\\sum_{{{}}}\\left[{}\\right]",
                fmt_vars(arena.var_set(*variables)),
                latex_expr(arena, *expr)
            )
        }
        ExprNode::IntegralOut { variables, expr } => {
            format!(
                "\\int_{{{}}}\\left[{}\\right]",
                fmt_vars(arena.var_set(*variables)),
                latex_expr(arena, *expr)
            )
        }
        ExprNode::Ratio { numerator, denominator } => {
            format!(
                "\\frac{{{}}}{{{}}}",
                latex_expr(arena, *numerator),
                latex_expr(arena, *denominator)
            )
        }
        ExprNode::Expectation { function, distribution } => {
            format!(
                "\\mathbb{{E}}\\left[V{} \\mid {}\\right]",
                function.variable().raw(),
                latex_expr(arena, *distribution)
            )
        }
        ExprNode::Contrast { left, right, op } => {
            let op_s = match op {
                ContrastOp::Difference => "-",
            };
            format!(
                "\\left({}\\right) {} \\left({}\\right)",
                latex_expr(arena, *left),
                op_s,
                latex_expr(arena, *right)
            )
        }
    }
}

fn fmt_vars(vars: &[VariableId]) -> String {
    vars.iter().map(|v| format!("V{}", v.raw())).collect::<Vec<_>>().join(",")
}

fn fmt_assignments(assignments: &[InterventionAssignment]) -> String {
    assignments
        .iter()
        .map(|a| format!("V{}:={}", a.variable.raw(), fmt_value(&a.value)))
        .collect::<Vec<_>>()
        .join(",")
}

fn fmt_value(v: &Value) -> String {
    match v {
        Value::Float64(x) => format!("{x}"),
        Value::Int64(x) => format!("{x}"),
        Value::Bool(x) => format!("{x}"),
        Value::Category(x) => format!("c{x}"),
        Value::Label(x) => x.to_string(),
    }
}
