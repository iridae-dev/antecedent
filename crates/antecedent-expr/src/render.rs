//! Shared expression renderer for diagnostics (not equality keys).
//!
//! LaTeX and plain-text rendering share one arena traversal; they differ only in
//! the literal fragments captured by [`Style`]. This is diagnostic output, never
//! an equality key.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{Value, VariableId};

use crate::{CausalExprArena, ContrastOp, DomainRef, ExprId, ExprNode, InterventionAssignment};

/// The style-specific literal fragments that separate LaTeX from plain text.
pub(crate) struct Style {
    /// Separator before a conditioning list (`\mid ` vs `|`).
    cond_mid: &'static str,
    /// Opening of a `do(...)` intervention (`\mathrm{do}(` vs `do(`).
    do_open: &'static str,
    /// Subscript delimiters for a population / kernel label (`_{` and `}` in LaTeX;
    /// a bare `_` and nothing in plain text).
    sub_open: &'static str,
    sub_close: &'static str,
    /// Regime tail prefix, before the raw regime id.
    regime_prefix: &'static str,
    /// Bracket delimiters around a kernel / sum / integral body.
    bracket_open: &'static str,
    bracket_close: &'static str,
    /// Product join.
    product_join: &'static str,
    /// Sum / integral binder symbols.
    sum_sym: &'static str,
    int_sym: &'static str,
    /// Ratio delimiters (`\frac{` `}{` `}` vs `(` `)/(` `)`).
    ratio_open: &'static str,
    ratio_mid: &'static str,
    ratio_close: &'static str,
    /// Expectation prefix (through the leading `V`) and its conditioning separator.
    exp_open: &'static str,
    exp_mid: &'static str,
    /// Parenthesization of the two sides of a contrast, and the difference operator.
    paren_open: &'static str,
    paren_close: &'static str,
    diff_op: &'static str,
}

pub(crate) const LATEX: Style = Style {
    cond_mid: "\\mid ",
    do_open: "\\mathrm{do}(",
    sub_open: "_{",
    sub_close: "}",
    regime_prefix: ";\\,\\mathrm{regime}=R",
    bracket_open: "\\left[",
    bracket_close: "\\right]",
    product_join: " \\cdot ",
    sum_sym: "\\sum",
    int_sym: "\\int",
    ratio_open: "\\frac{",
    ratio_mid: "}{",
    ratio_close: "}",
    exp_open: "\\mathbb{E}\\left[V",
    exp_mid: " \\mid ",
    paren_open: "\\left(",
    paren_close: "\\right)",
    diff_op: "-",
};

pub(crate) const PRETTY: Style = Style {
    cond_mid: "|",
    do_open: "do(",
    sub_open: "_",
    sub_close: "",
    regime_prefix: "; regime=R",
    bracket_open: "[",
    bracket_close: "]",
    product_join: " * ",
    sum_sym: "\u{03a3}",
    int_sym: "\u{222b}",
    ratio_open: "(",
    ratio_mid: ")/(",
    ratio_close: ")",
    exp_open: "E[V",
    exp_mid: " | ",
    paren_open: "(",
    paren_close: ")",
    diff_op: "\u{2212}",
};

pub(crate) fn render_expr(arena: &CausalExprArena, id: ExprId, s: &Style) -> String {
    let mut memo = std::collections::HashMap::new();
    crate::walk::fold_expr(arena, id, &mut memo, &mut |arena, id, children| {
        render_step(arena, id, s, children)
    })
}

fn render_step(arena: &CausalExprArena, id: ExprId, s: &Style, children: &[String]) -> String {
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
            let conditioned = arena.var_set(*conditioned_on);
            let assignments = arena.intervention_assignments(*intervention);
            let head = fmt_p_head(arena, *population, s);
            let tail = fmt_regime(*regime, s);
            match domain {
                DomainRef::Observational => {
                    let observed_cond = fmt_conditioning(conditioned, assignments);
                    if observed_cond.is_empty() {
                        format!("{head}({vars}{tail})")
                    } else {
                        format!("{head}({vars}{}{observed_cond}{tail})", s.cond_mid)
                    }
                }
                DomainRef::Interventional => {
                    let cond = fmt_vars(conditioned);
                    let interv = fmt_assignments(assignments);
                    if cond.is_empty() {
                        format!("{head}({vars}{}{}{interv}){tail})", s.cond_mid, s.do_open)
                    } else {
                        format!("{head}({vars}{}{cond},{}{interv}){tail})", s.cond_mid, s.do_open)
                    }
                }
            }
        }
        ExprNode::Kernel { bound, population, regime, .. } => {
            let pop = arena.population(*population);
            let pop = if pop.is_empty() {
                String::new()
            } else {
                format!("{}{pop}{}", s.sub_open, s.sub_close)
            };
            format!(
                "K{pop}_{{{}}}{}{}{}{}",
                fmt_vars(arena.var_set(*bound)),
                s.bracket_open,
                children[0],
                fmt_regime(*regime, s),
                s.bracket_close,
            )
        }
        ExprNode::Product(_) => children.join(s.product_join),
        ExprNode::SumOut { variables, .. } => {
            render_marginal(arena, s.sum_sym, *variables, &children[0], s)
        }
        ExprNode::IntegralOut { variables, .. } => {
            render_marginal(arena, s.int_sym, *variables, &children[0], s)
        }
        ExprNode::Ratio { .. } => {
            format!(
                "{}{}{}{}{}",
                s.ratio_open, children[0], s.ratio_mid, children[1], s.ratio_close,
            )
        }
        ExprNode::Expectation { function, .. } => {
            format!(
                "{}{}{}{}{}",
                s.exp_open,
                function.variable().raw(),
                s.exp_mid,
                children[0],
                s.bracket_close,
            )
        }
        ExprNode::Contrast { op, .. } => {
            let op_s = match op {
                ContrastOp::Difference => s.diff_op,
            };
            format!(
                "{}{}{} {op_s} {}{}{}",
                s.paren_open, children[0], s.paren_close, s.paren_open, children[1], s.paren_close,
            )
        }
    }
}

/// Direct recursion, kept so a test can require the shared fold to print the same text.
///
/// Child ids are named in this match. It does not call [`crate::walk::child_ids`].
#[cfg(test)]
pub(crate) fn render_expr_recursive(arena: &CausalExprArena, id: ExprId, s: &Style) -> String {
    let rendered: Vec<String> = match arena.node(id) {
        ExprNode::Distribution { .. } => Vec::new(),
        ExprNode::Kernel { body, .. } => vec![render_expr_recursive(arena, *body, s)],
        ExprNode::Product(list) => {
            arena.list(*list).iter().map(|child| render_expr_recursive(arena, *child, s)).collect()
        }
        ExprNode::SumOut { expr, .. } | ExprNode::IntegralOut { expr, .. } => {
            vec![render_expr_recursive(arena, *expr, s)]
        }
        ExprNode::Ratio { numerator, denominator } => {
            vec![
                render_expr_recursive(arena, *numerator, s),
                render_expr_recursive(arena, *denominator, s),
            ]
        }
        ExprNode::Expectation { distribution, .. } => {
            vec![render_expr_recursive(arena, *distribution, s)]
        }
        ExprNode::Contrast { left, right, .. } => {
            vec![render_expr_recursive(arena, *left, s), render_expr_recursive(arena, *right, s)]
        }
    };
    render_step(arena, id, s, &rendered)
}

/// A sum or integral over `variables`: the operator symbol, its bound subscript, and the body.
fn render_marginal(
    arena: &CausalExprArena,
    symbol: &str,
    variables: crate::VarSetId,
    body: &str,
    s: &Style,
) -> String {
    format!(
        "{symbol}_{{{}}}{}{}{}",
        fmt_vars(arena.var_set(variables)),
        s.bracket_open,
        body,
        s.bracket_close,
    )
}

fn fmt_p_head(arena: &CausalExprArena, population: crate::PopulationKeyId, s: &Style) -> String {
    let name = arena.population(population);
    if name.is_empty() { "P".into() } else { format!("P{}{name}{}", s.sub_open, s.sub_close) }
}

fn fmt_regime(regime: Option<antecedent_core::RegimeId>, s: &Style) -> String {
    match regime {
        Some(regime) => format!("{}{}", s.regime_prefix, regime.raw()),
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

/// Conditioning list of an observational leaf. A conditioner bound by the leaf's assignment set
/// prints its level (`V2=1`), as does a binder outside the conditioners, so leaves that differ
/// only in the level they are bound to print differently.
fn fmt_conditioning(
    conditioned_on: &[VariableId],
    assignments: &[InterventionAssignment],
) -> String {
    let item = |variable: VariableId| match assignments.iter().find(|a| a.variable == variable) {
        Some(a) if !a.is_symbolic() => format!("V{}={}", variable.raw(), fmt_value(&a.value)),
        _ => format!("V{}", variable.raw()),
    };
    let mut items: Vec<String> = conditioned_on.iter().map(|v| item(*v)).collect();
    items.extend(
        assignments
            .iter()
            .filter(|a| !conditioned_on.contains(&a.variable))
            .map(|a| item(a.variable)),
    );
    items.join(",")
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
