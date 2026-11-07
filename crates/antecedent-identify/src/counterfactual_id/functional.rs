//! Counterfactual functional expressions and their rendering.

use antecedent_expr::{CausalExprArena, ExprId};
use serde::{Deserialize, Serialize};

/// A level in the counterfactual functional: a literal level (IEEE bits) or a
/// summation symbol bound by an enclosing [`CounterfactualFunctional::Sum`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum CfValue {
    /// A literal level.
    Level(u64),
    /// Summation symbol id.
    Symbol(u32),
}

impl CfValue {
    pub(super) fn render(self) -> String {
        match self {
            Self::Level(bits) => format!("{}", f64::from_bits(bits)),
            Self::Symbol(id) => format!("s{id}"),
        }
    }
}

/// One summation symbol and the variable whose levels it ranges over.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CfSymbol {
    /// Symbol id, unique within a derivation.
    pub id: u32,
    /// Variable whose levels the symbol ranges over.
    pub variable: u32,
}

/// One district term `P_{intervention}(event)`, identified from `P(V)` by ID.
#[derive(Clone, Debug, PartialEq)]
pub struct CfTerm {
    /// `(variable, level)` set by the intervention, sorted.
    pub intervention: Vec<(u32, CfValue)>,
    /// `(variable, level)` of the event, sorted.
    pub event: Vec<(u32, CfValue)>,
    /// Arena holding the ID functional; the intervened variables are left free.
    pub arena: CausalExprArena,
    /// Root of the ID functional.
    pub expression: ExprId,
}

/// The identified counterfactual functional.
#[derive(Clone, Debug, PartialEq)]
pub enum CounterfactualFunctional {
    /// Probability one (an empty conjunction).
    One,
    /// Probability zero (a contradictory conjunction).
    Zero,
    /// Sum over the symbols' levels of the product of the factors.
    Sum {
        /// Symbols summed.
        symbols: Vec<CfSymbol>,
        /// Factors multiplied.
        factors: Vec<CounterfactualFunctional>,
    },
    /// A district term.
    Term(Box<CfTerm>),
    /// A binary treatment's consistency complement,
    /// `P(Y_x = y, X = x') = P_x(Y = y) - P(Y = y, X = x)`: consistency gives
    /// `P(Y_x = y, X = x) = P(Y = y, X = x)` and `X = x'` is the complement of
    /// `X = x`. Used only when ID* does not identify the conjunction and ID
    /// identifies `P_x(Y)`.
    ConsistencyComplement {
        /// `P_x(Y = y)`, identified from `P(V)` by ID.
        interventional: Box<CfTerm>,
        /// `(variable, level)` of the observed subtrahend `P(Y = y, X = x)`,
        /// sorted.
        observed: Vec<(u32, CfValue)>,
    },
}

fn render_term(term: &CfTerm, out: &mut String) {
    out.push_str("P[do(");
    for (v, x) in &term.intervention {
        out.push_str(&format!("v{v}={},", x.render()));
    }
    out.push_str(")](");
    for (v, x) in &term.event {
        out.push_str(&format!("v{v}={},", x.render()));
    }
    out.push_str(") := ");
    out.push_str(&term.arena.pretty(term.expression));
}

impl CounterfactualFunctional {
    pub(super) fn render(&self, out: &mut String) {
        match self {
            Self::One => out.push('1'),
            Self::Zero => out.push('0'),
            Self::Sum { symbols, factors } => {
                out.push_str("sum[");
                for s in symbols {
                    out.push_str(&format!("s{}:v{},", s.id, s.variable));
                }
                out.push_str("](");
                for f in factors {
                    f.render(out);
                    out.push_str(" * ");
                }
                out.push(')');
            }
            Self::Term(term) => render_term(term, out),
            Self::ConsistencyComplement { interventional, observed } => {
                out.push_str("complement(");
                render_term(interventional, out);
                out.push_str(" - P(");
                for (v, x) in observed {
                    out.push_str(&format!("v{v}={},", x.render()));
                }
                out.push_str("))");
            }
        }
    }

    pub(super) fn terms<'a>(&'a self, out: &mut Vec<&'a CfTerm>) {
        match self {
            Self::One | Self::Zero => {}
            Self::Sum { factors, .. } => factors.iter().for_each(|f| f.terms(out)),
            Self::Term(term) => out.push(term),
            Self::ConsistencyComplement { interventional, .. } => out.push(interventional),
        }
    }

    /// Whether the functional answers through the binary consistency
    /// complement rather than ID* alone.
    #[must_use]
    pub fn uses_consistency_complement(&self) -> bool {
        match self {
            Self::One | Self::Zero | Self::Term(_) => false,
            Self::Sum { factors, .. } => factors.iter().any(Self::uses_consistency_complement),
            Self::ConsistencyComplement { .. } => true,
        }
    }
}
