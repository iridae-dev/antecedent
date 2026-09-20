//! Pretty-printing for diagnostics (not equality keys).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{Value, VariableId};

use crate::{CausalExprArena, ContrastOp, DomainRef, ExprId, ExprNode, InterventionAssignment};

pub(crate) fn pretty_expr(arena: &CausalExprArena, id: ExprId) -> String {
    match arena.node(id) {
        ExprNode::Distribution {
            variables,
            conditioned_on,
            intervention,
            domain,
            population,
            regime,
        } => {
            let vars = fmt_vars(arena.var_set(*variables));
            let cond = fmt_vars(arena.var_set(*conditioned_on));
            let interv = fmt_assignments(arena.intervention_assignments(*intervention));
            let head = fmt_p_head(arena, *population);
            let tail = fmt_regime(*regime);
            match domain {
                DomainRef::Observational => {
                    if cond.is_empty() {
                        format!("{head}({vars}{tail})")
                    } else {
                        format!("{head}({vars}|{cond}{tail})")
                    }
                }
                DomainRef::Interventional => {
                    if cond.is_empty() {
                        format!("{head}({vars}|do({interv}){tail})")
                    } else {
                        format!("{head}({vars}|{cond},do({interv}){tail})")
                    }
                }
            }
        }
        ExprNode::Kernel { body, bound, population, regime } => {
            let pop = arena.population(*population);
            let pop = if pop.is_empty() { String::new() } else { format!("_{pop}") };
            format!(
                "K{pop}_{{{}}}[{}{}]",
                fmt_vars(arena.var_set(*bound)),
                pretty_expr(arena, *body),
                fmt_regime(*regime)
            )
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

fn fmt_p_head(arena: &CausalExprArena, population: crate::PopulationKeyId) -> String {
    let name = arena.population(population);
    if name.is_empty() { "P".into() } else { format!("P_{name}") }
}

fn fmt_regime(regime: Option<antecedent_core::RegimeId>) -> String {
    match regime {
        Some(regime) => format!("; regime=R{}", regime.raw()),
        None => String::new(),
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
