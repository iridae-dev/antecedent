//! Typed value expressions for programmable interventions.
//!
//! A [`ValueExpr`] is a small, closed expression tree over numbers and booleans:
//! arithmetic, `min`/`max`/`clip`, comparisons, boolean connectives and
//! `if/then/else`, with explicit reads of other variables ([`ValueExpr::Var`]),
//! of lagged history ([`ValueExpr::Lag`]) and of the node's own generated value
//! ([`ValueExpr::Generated`]). It is a *value* language for interventions, not the
//! identified-functional IR (`antecedent-expr`), and it has no randomness: the same
//! reads always give the same value.
//!
//! Booleans are carried as `0.0` / `1.0` when evaluated, so one column kernel serves
//! both types; the type checker keeps the two apart so a comparison cannot be written
//! into a numeric node without an explicit [`ValueExpr::indicator`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;
use std::sync::Arc;

use crate::callback::{CallbackCall, CallbackOutput, Portability};
use crate::canonical_encode::{CanonicalEncode, count, encode_variables};
use crate::identity::CanonicalWriter;
use crate::ids::VariableId;

/// Largest number of nodes an expression may hold.
pub const MAX_EXPR_NODES: usize = 512;
/// Deepest nesting an expression may have.
pub const MAX_EXPR_DEPTH: usize = 64;

/// Static type of an expression.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ExprType {
    /// A real number.
    Number,
    /// A truth value.
    Boolean,
}

impl fmt::Display for ExprType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Number => "number",
            Self::Boolean => "boolean",
        })
    }
}

/// A finite or non-finite `f64` constant compared and hashed by its bits, so that
/// [`ValueExpr`] can be `Eq + Hash` like the other intervention parts.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ExprNumber(u64);

impl ExprNumber {
    /// Wrap a number.
    #[must_use]
    pub fn new(value: f64) -> Self {
        Self(value.to_bits())
    }

    /// The number.
    #[must_use]
    pub fn get(self) -> f64 {
        f64::from_bits(self.0)
    }
}

/// Binary numeric operator.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum ArithOp {
    /// `a + b`.
    Add,
    /// `a - b`.
    Sub,
    /// `a * b`.
    Mul,
    /// `a / b` (a non-finite quotient makes the whole value non-finite and is refused
    /// at execution rather than propagated).
    Div,
    /// Smaller of the two; NaN if either is NaN.
    Min,
    /// Larger of the two; NaN if either is NaN.
    Max,
}

impl ArithOp {
    /// Canonical tag. Exhaustive, so a new operator must be numbered here.
    const fn canonical_tag(self) -> u64 {
        match self {
            Self::Add => 1,
            Self::Sub => 2,
            Self::Mul => 3,
            Self::Div => 4,
            Self::Min => 5,
            Self::Max => 6,
        }
    }
}

impl ArithOp {
    /// Apply to one pair. The batch kernel calls exactly this, so batch and scalar
    /// evaluation are bit-identical.
    #[must_use]
    #[inline]
    pub fn apply(self, a: f64, b: f64) -> f64 {
        match self {
            Self::Add => a + b,
            Self::Sub => a - b,
            Self::Mul => a * b,
            Self::Div => a / b,
            // Written as selects (non-short-circuit `|`) so a column loop vectorizes; the
            // values are those of the branching form, signed zeros included.
            Self::Min => {
                let m = if b < a { b } else { a };
                if a.is_nan() | b.is_nan() { f64::NAN } else { m }
            }
            Self::Max => {
                let m = if b > a { b } else { a };
                if a.is_nan() | b.is_nan() { f64::NAN } else { m }
            }
        }
    }
}

/// `value` limited to `[lower, upper]`; NaN when any input is NaN or the bounds are
/// inverted.
#[must_use]
#[inline]
pub fn clip_value(value: f64, lower: f64, upper: f64) -> f64 {
    if value.is_nan() || lower.is_nan() || upper.is_nan() || lower > upper {
        f64::NAN
    } else {
        value.clamp(lower, upper)
    }
}

/// Numeric comparison.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum CompareOp {
    /// `a < b`.
    Lt,
    /// `a <= b`.
    Le,
    /// `a > b`.
    Gt,
    /// `a >= b`.
    Ge,
    /// `a == b`.
    Eq,
    /// `a != b`.
    Ne,
}

impl CompareOp {
    /// Canonical tag. Exhaustive, so a new operator must be numbered here.
    const fn canonical_tag(self) -> u64 {
        match self {
            Self::Lt => 1,
            Self::Le => 2,
            Self::Gt => 3,
            Self::Ge => 4,
            Self::Eq => 5,
            Self::Ne => 6,
        }
    }
}

impl CompareOp {
    /// Apply to one pair (false when either side is NaN, except `Ne`).
    #[must_use]
    #[allow(
        clippy::float_cmp,
        reason = "an explicit equality comparison is the operator's meaning"
    )]
    #[inline]
    pub fn apply(self, a: f64, b: f64) -> bool {
        match self {
            Self::Lt => a < b,
            Self::Le => a <= b,
            Self::Gt => a > b,
            Self::Ge => a >= b,
            Self::Eq => a == b,
            Self::Ne => a != b,
        }
    }
}

/// Boolean connective.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum LogicOp {
    /// Both true.
    And,
    /// Either true.
    Or,
}

impl LogicOp {
    /// Canonical tag. Exhaustive, so a new connective must be numbered here.
    const fn canonical_tag(self) -> u64 {
        match self {
            Self::And => 1,
            Self::Or => 2,
        }
    }
}

/// A typed value expression.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum ValueExpr {
    /// A numeric constant.
    Const(ExprNumber),
    /// The value the node's own mechanism generated in this draw. Readable by a
    /// transform only.
    Generated,
    /// The current-draw value of another variable, after any intervention on it.
    Var(VariableId),
    /// The value of a variable `lag` steps before the current time. Only temporal
    /// routes with a declared history window can supply it.
    Lag {
        /// Variable read.
        variable: VariableId,
        /// Steps back (at least one).
        lag: u32,
    },
    /// Arithmetic negation.
    Neg(Box<ValueExpr>),
    /// Binary numeric operator.
    Arith {
        /// Operator.
        op: ArithOp,
        /// Left operand.
        lhs: Box<ValueExpr>,
        /// Right operand.
        rhs: Box<ValueExpr>,
    },
    /// `clip(value, lower, upper)`.
    Clip {
        /// Clipped value.
        value: Box<ValueExpr>,
        /// Lower bound.
        lower: Box<ValueExpr>,
        /// Upper bound.
        upper: Box<ValueExpr>,
    },
    /// Numeric comparison (boolean result).
    Compare {
        /// Operator.
        op: CompareOp,
        /// Left operand.
        lhs: Box<ValueExpr>,
        /// Right operand.
        rhs: Box<ValueExpr>,
    },
    /// Boolean negation.
    Not(Box<ValueExpr>),
    /// Boolean connective.
    Logic {
        /// Connective.
        op: LogicOp,
        /// Left operand.
        lhs: Box<ValueExpr>,
        /// Right operand.
        rhs: Box<ValueExpr>,
    },
    /// `if condition then then_branch else else_branch`, selected per row. Both
    /// branches are well-typed and evaluated; only the selected value is used.
    If {
        /// Boolean condition.
        condition: Box<ValueExpr>,
        /// Value when true.
        then_branch: Box<ValueExpr>,
        /// Value when false.
        else_branch: Box<ValueExpr>,
    },
    /// Explicit boolean-to-number conversion (`true` is `1`, `false` is `0`).
    Indicator(Box<ValueExpr>),
    /// A caller-supplied batch callback applied to numeric argument expressions. A
    /// runtime binding: it executes in this process only and is not portable (see
    /// [`ValueExpr::portability`]).
    Callback(CallbackCall),
}

/// Why an expression is not a valid node program.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ExpressionFault {
    /// An operand has the wrong type.
    #[error("operand has type {found} where {expected} is required")]
    TypeMismatch {
        /// Required type.
        expected: ExprType,
        /// Type found.
        found: ExprType,
    },
    /// The two branches of a conditional have different types.
    #[error("conditional branches have different types ({then_type} and {else_type})")]
    BranchTypes {
        /// Type of the `then` branch.
        then_type: ExprType,
        /// Type of the `else` branch.
        else_type: ExprType,
    },
    /// The expression does not produce a number.
    #[error(
        "expression produces {found}; a node value must be a number (convert a condition with indicator)"
    )]
    OutputType {
        /// Type produced.
        found: ExprType,
    },
    /// A constant is NaN or infinite.
    #[error("expression constant is not finite")]
    NonFiniteConstant,
    /// The expression exceeds the node or depth bound.
    #[error("expression exceeds the {}-node or {}-level bound", MAX_EXPR_NODES, MAX_EXPR_DEPTH)]
    TooLarge,
    /// A lagged read with lag zero (use a current read).
    #[error("a lagged read must look back at least one step")]
    ZeroLag,
    /// An assignment reads the generated value it replaces.
    #[error("an assignment replaces the node's generated value and cannot read it")]
    GeneratedInAssignment,
    /// The declared read set differs from the variables the expression references.
    #[error("declared reads differ from the variables the expression references")]
    DeclaredReadsMismatch,
    /// A callback is applied to a different number of arguments than it declares inputs.
    #[error("a callback declares {declared} inputs but is applied to {supplied} arguments")]
    CallbackArity {
        /// Inputs the callback declares.
        declared: usize,
        /// Arguments supplied.
        supplied: usize,
    },
    /// A callback's declaration is malformed (name, input labels, output or recovery data).
    #[error("a callback declaration is malformed")]
    CallbackDeclaration,
}

/// The variables a program reads, as declared by its author.
///
/// Validation requires the declaration to equal what the expression references
/// ([`ValueExpr::reads`]), so a policy cannot read anything it did not declare.
#[derive(Clone, Debug, Default, Eq, Hash, PartialEq)]
pub struct PolicyReads {
    /// Variables read at the current time, sorted and unique.
    pub current: Arc<[VariableId]>,
    /// `(variable, lag)` history reads, sorted and unique.
    pub history: Arc<[(VariableId, u32)]>,
}

impl CanonicalEncode for PolicyReads {
    fn encode(&self, writer: &mut CanonicalWriter) {
        encode_variables(writer, &self.current);
        writer.word(count(self.history.len()));
        for &(variable, lag) in self.history.as_ref() {
            variable.encode(writer);
            writer.word(u64::from(lag));
        }
    }
}

impl PolicyReads {
    /// Build from unordered reads (sorted and de-duplicated).
    #[must_use]
    pub fn new(
        current: impl IntoIterator<Item = VariableId>,
        history: impl IntoIterator<Item = (VariableId, u32)>,
    ) -> Self {
        let mut current: Vec<_> = current.into_iter().collect();
        current.sort_unstable();
        current.dedup();
        let mut history: Vec<_> = history.into_iter().collect();
        history.sort_unstable();
        history.dedup();
        Self { current: current.into(), history: history.into() }
    }

    /// Whether nothing is read.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.current.is_empty() && self.history.is_empty()
    }
}

impl CanonicalEncode for ValueExpr {
    fn encode(&self, writer: &mut CanonicalWriter) {
        match self {
            Self::Const(number) => {
                writer.word(1).real(number.get());
            }
            Self::Generated => {
                writer.word(2);
            }
            Self::Var(variable) => {
                writer.word(3);
                variable.encode(writer);
            }
            Self::Lag { variable, lag } => {
                writer.word(4);
                variable.encode(writer);
                writer.word(u64::from(*lag));
            }
            Self::Neg(inner) => {
                writer.word(5);
                inner.encode(writer);
            }
            Self::Arith { op, lhs, rhs } => {
                writer.word(6).word(op.canonical_tag());
                lhs.encode(writer);
                rhs.encode(writer);
            }
            Self::Clip { value, lower, upper } => {
                writer.word(7);
                value.encode(writer);
                lower.encode(writer);
                upper.encode(writer);
            }
            Self::Compare { op, lhs, rhs } => {
                writer.word(8).word(op.canonical_tag());
                lhs.encode(writer);
                rhs.encode(writer);
            }
            Self::Not(inner) => {
                writer.word(9);
                inner.encode(writer);
            }
            Self::Logic { op, lhs, rhs } => {
                writer.word(10).word(op.canonical_tag());
                lhs.encode(writer);
                rhs.encode(writer);
            }
            Self::If { condition, then_branch, else_branch } => {
                writer.word(11);
                condition.encode(writer);
                then_branch.encode(writer);
                else_branch.encode(writer);
            }
            Self::Indicator(inner) => {
                writer.word(12);
                inner.encode(writer);
            }
            Self::Callback(call) => {
                writer.word(13);
                call.encode(writer);
            }
        }
    }
}

impl ValueExpr {
    /// Numeric constant.
    #[must_use]
    pub fn constant(value: f64) -> Self {
        Self::Const(ExprNumber::new(value))
    }

    /// The node's own generated value.
    #[must_use]
    pub const fn generated() -> Self {
        Self::Generated
    }

    /// Current-draw read of a variable.
    #[must_use]
    pub const fn var(variable: VariableId) -> Self {
        Self::Var(variable)
    }

    /// Read of a variable `lag` steps back.
    #[must_use]
    pub const fn lag(variable: VariableId, lag: u32) -> Self {
        Self::Lag { variable, lag }
    }

    fn arith(self, op: ArithOp, rhs: Self) -> Self {
        Self::Arith { op, lhs: Box::new(self), rhs: Box::new(rhs) }
    }

    fn compare(self, op: CompareOp, rhs: Self) -> Self {
        Self::Compare { op, lhs: Box::new(self), rhs: Box::new(rhs) }
    }

    /// `self + rhs`.
    #[must_use]
    #[allow(clippy::should_implement_trait, reason = "builds a tree; not an operator on values")]
    pub fn add(self, rhs: Self) -> Self {
        self.arith(ArithOp::Add, rhs)
    }

    /// `self - rhs`.
    #[must_use]
    #[allow(clippy::should_implement_trait, reason = "builds a tree; not an operator on values")]
    pub fn sub(self, rhs: Self) -> Self {
        self.arith(ArithOp::Sub, rhs)
    }

    /// `self * rhs`.
    #[must_use]
    #[allow(clippy::should_implement_trait, reason = "builds a tree; not an operator on values")]
    pub fn mul(self, rhs: Self) -> Self {
        self.arith(ArithOp::Mul, rhs)
    }

    /// `self / rhs`.
    #[must_use]
    #[allow(clippy::should_implement_trait, reason = "builds a tree; not an operator on values")]
    pub fn div(self, rhs: Self) -> Self {
        self.arith(ArithOp::Div, rhs)
    }

    /// Smaller of `self` and `rhs`.
    #[must_use]
    pub fn min(self, rhs: Self) -> Self {
        self.arith(ArithOp::Min, rhs)
    }

    /// Larger of `self` and `rhs`.
    #[must_use]
    pub fn max(self, rhs: Self) -> Self {
        self.arith(ArithOp::Max, rhs)
    }

    /// `self` limited to `[lower, upper]`.
    #[must_use]
    pub fn clip(self, lower: Self, upper: Self) -> Self {
        Self::Clip { value: Box::new(self), lower: Box::new(lower), upper: Box::new(upper) }
    }

    /// Arithmetic negation.
    #[must_use]
    #[allow(clippy::should_implement_trait, reason = "builds a tree; not an operator on values")]
    pub fn neg(self) -> Self {
        Self::Neg(Box::new(self))
    }

    /// `self < rhs`.
    #[must_use]
    pub fn lt(self, rhs: Self) -> Self {
        self.compare(CompareOp::Lt, rhs)
    }

    /// `self <= rhs`.
    #[must_use]
    pub fn le(self, rhs: Self) -> Self {
        self.compare(CompareOp::Le, rhs)
    }

    /// `self > rhs`.
    #[must_use]
    pub fn gt(self, rhs: Self) -> Self {
        self.compare(CompareOp::Gt, rhs)
    }

    /// `self >= rhs`.
    #[must_use]
    pub fn ge(self, rhs: Self) -> Self {
        self.compare(CompareOp::Ge, rhs)
    }

    /// `self == rhs`.
    #[must_use]
    pub fn equals(self, rhs: Self) -> Self {
        self.compare(CompareOp::Eq, rhs)
    }

    /// `self != rhs`.
    #[must_use]
    pub fn not_equals(self, rhs: Self) -> Self {
        self.compare(CompareOp::Ne, rhs)
    }

    /// Boolean negation.
    #[must_use]
    #[allow(clippy::should_implement_trait, reason = "builds a tree; not an operator on values")]
    pub fn not(self) -> Self {
        Self::Not(Box::new(self))
    }

    /// `self && rhs`.
    #[must_use]
    pub fn and(self, rhs: Self) -> Self {
        Self::Logic { op: LogicOp::And, lhs: Box::new(self), rhs: Box::new(rhs) }
    }

    /// `self || rhs`.
    #[must_use]
    pub fn or(self, rhs: Self) -> Self {
        Self::Logic { op: LogicOp::Or, lhs: Box::new(self), rhs: Box::new(rhs) }
    }

    /// `if self then then_branch else else_branch`.
    #[must_use]
    pub fn if_then_else(self, then_branch: Self, else_branch: Self) -> Self {
        Self::If {
            condition: Box::new(self),
            then_branch: Box::new(then_branch),
            else_branch: Box::new(else_branch),
        }
    }

    /// Convert a boolean to `1` / `0`.
    #[must_use]
    pub fn indicator(self) -> Self {
        Self::Indicator(Box::new(self))
    }

    /// A batch callback applied to `args`, one numeric expression per declared input.
    #[must_use]
    pub fn callback(
        callback: std::sync::Arc<dyn crate::callback::BatchCallback>,
        args: impl Into<Arc<[Self]>>,
    ) -> Self {
        Self::Callback(CallbackCall::new(callback, args))
    }

    /// Whether the expression reads the node's own generated value.
    #[must_use]
    pub fn reads_generated(&self) -> bool {
        match self {
            Self::Generated => true,
            Self::Const(_) | Self::Var(_) | Self::Lag { .. } => false,
            Self::Neg(a) | Self::Not(a) | Self::Indicator(a) => a.reads_generated(),
            Self::Arith { lhs, rhs, .. }
            | Self::Compare { lhs, rhs, .. }
            | Self::Logic { lhs, rhs, .. } => lhs.reads_generated() || rhs.reads_generated(),
            Self::Clip { value, lower, upper } => {
                value.reads_generated() || lower.reads_generated() || upper.reads_generated()
            }
            Self::If { condition, then_branch, else_branch } => {
                condition.reads_generated()
                    || then_branch.reads_generated()
                    || else_branch.reads_generated()
            }
            Self::Callback(call) => call.args().iter().any(Self::reads_generated),
        }
    }

    /// The same expression with every variable reference replaced by `map`'s image
    /// (current and lagged reads alike).
    ///
    /// # Errors
    ///
    /// The first error `map` returns.
    pub fn try_map_variables<E>(
        &self,
        map: &mut dyn FnMut(VariableId) -> Result<VariableId, E>,
    ) -> Result<Self, E> {
        let mut go = |e: &Self| e.try_map_variables(&mut *map).map(Box::new);
        Ok(match self {
            Self::Const(_) | Self::Generated => self.clone(),
            Self::Var(v) => Self::Var(map(*v)?),
            Self::Lag { variable, lag } => Self::Lag { variable: map(*variable)?, lag: *lag },
            Self::Neg(a) => Self::Neg(go(a)?),
            Self::Not(a) => Self::Not(go(a)?),
            Self::Indicator(a) => Self::Indicator(go(a)?),
            Self::Arith { op, lhs, rhs } => Self::Arith { op: *op, lhs: go(lhs)?, rhs: go(rhs)? },
            Self::Compare { op, lhs, rhs } => {
                Self::Compare { op: *op, lhs: go(lhs)?, rhs: go(rhs)? }
            }
            Self::Logic { op, lhs, rhs } => Self::Logic { op: *op, lhs: go(lhs)?, rhs: go(rhs)? },
            Self::Clip { value, lower, upper } => {
                Self::Clip { value: go(value)?, lower: go(lower)?, upper: go(upper)? }
            }
            Self::If { condition, then_branch, else_branch } => Self::If {
                condition: go(condition)?,
                then_branch: go(then_branch)?,
                else_branch: go(else_branch)?,
            },
            Self::Callback(call) => {
                let args = call
                    .args()
                    .iter()
                    .map(|a| a.try_map_variables(&mut *map))
                    .collect::<Result<Vec<_>, E>>()?;
                Self::Callback(call.with_args(args))
            }
        })
    }

    /// The variables the expression references, sorted and unique.
    #[must_use]
    pub fn reads(&self) -> PolicyReads {
        let mut current = Vec::new();
        let mut history = Vec::new();
        self.collect_reads(&mut current, &mut history);
        PolicyReads::new(current, history)
    }

    fn collect_reads(&self, current: &mut Vec<VariableId>, history: &mut Vec<(VariableId, u32)>) {
        match self {
            Self::Const(_) | Self::Generated => {}
            Self::Var(v) => current.push(*v),
            Self::Lag { variable, lag } => history.push((*variable, *lag)),
            Self::Neg(a) | Self::Not(a) | Self::Indicator(a) => a.collect_reads(current, history),
            Self::Arith { lhs, rhs, .. }
            | Self::Compare { lhs, rhs, .. }
            | Self::Logic { lhs, rhs, .. } => {
                lhs.collect_reads(current, history);
                rhs.collect_reads(current, history);
            }
            Self::Clip { value, lower, upper } => {
                value.collect_reads(current, history);
                lower.collect_reads(current, history);
                upper.collect_reads(current, history);
            }
            Self::If { condition, then_branch, else_branch } => {
                condition.collect_reads(current, history);
                then_branch.collect_reads(current, history);
                else_branch.collect_reads(current, history);
            }
            Self::Callback(call) => {
                for arg in call.args() {
                    arg.collect_reads(current, history);
                }
            }
        }
    }

    /// Type of the expression, checking every operand, constant and bound.
    ///
    /// # Errors
    ///
    /// [`ExpressionFault`] naming the first violation.
    pub fn value_type(&self) -> Result<ExprType, ExpressionFault> {
        let mut count = 0usize;
        self.check(1, &mut count)
    }

    fn check(&self, depth: usize, count: &mut usize) -> Result<ExprType, ExpressionFault> {
        *count += 1;
        if depth > MAX_EXPR_DEPTH || *count > MAX_EXPR_NODES {
            return Err(ExpressionFault::TooLarge);
        }
        let want = |expr: &Self, expected: ExprType, count: &mut usize| {
            let found = expr.check(depth + 1, count)?;
            if found == expected {
                Ok(())
            } else {
                Err(ExpressionFault::TypeMismatch { expected, found })
            }
        };
        match self {
            Self::Const(c) => {
                if c.get().is_finite() {
                    Ok(ExprType::Number)
                } else {
                    Err(ExpressionFault::NonFiniteConstant)
                }
            }
            Self::Generated | Self::Var(_) => Ok(ExprType::Number),
            Self::Lag { lag, .. } => {
                if *lag == 0 {
                    Err(ExpressionFault::ZeroLag)
                } else {
                    Ok(ExprType::Number)
                }
            }
            Self::Neg(a) => {
                want(a, ExprType::Number, count)?;
                Ok(ExprType::Number)
            }
            Self::Arith { lhs, rhs, .. } => {
                want(lhs, ExprType::Number, count)?;
                want(rhs, ExprType::Number, count)?;
                Ok(ExprType::Number)
            }
            Self::Clip { value, lower, upper } => {
                want(value, ExprType::Number, count)?;
                want(lower, ExprType::Number, count)?;
                want(upper, ExprType::Number, count)?;
                Ok(ExprType::Number)
            }
            Self::Compare { lhs, rhs, .. } => {
                want(lhs, ExprType::Number, count)?;
                want(rhs, ExprType::Number, count)?;
                Ok(ExprType::Boolean)
            }
            Self::Not(a) => {
                want(a, ExprType::Boolean, count)?;
                Ok(ExprType::Boolean)
            }
            Self::Logic { lhs, rhs, .. } => {
                want(lhs, ExprType::Boolean, count)?;
                want(rhs, ExprType::Boolean, count)?;
                Ok(ExprType::Boolean)
            }
            Self::If { condition, then_branch, else_branch } => {
                want(condition, ExprType::Boolean, count)?;
                let then_type = then_branch.check(depth + 1, count)?;
                let else_type = else_branch.check(depth + 1, count)?;
                if then_type == else_type {
                    Ok(then_type)
                } else {
                    Err(ExpressionFault::BranchTypes { then_type, else_type })
                }
            }
            Self::Indicator(a) => {
                want(a, ExprType::Boolean, count)?;
                Ok(ExprType::Number)
            }
            Self::Callback(call) => {
                if !call.spec().is_well_formed() {
                    return Err(ExpressionFault::CallbackDeclaration);
                }
                let declared = call.spec().inputs.len();
                if declared != call.args().len() {
                    return Err(ExpressionFault::CallbackArity {
                        declared,
                        supplied: call.args().len(),
                    });
                }
                for arg in call.args() {
                    want(arg, ExprType::Number, count)?;
                }
                Ok(ExprType::Number)
            }
        }
    }

    /// Validate as the program of one node: well typed, numeric output, finite
    /// constants, declared reads equal to the referenced variables, and no read of the
    /// generated value unless `allow_generated`.
    ///
    /// # Errors
    ///
    /// [`ExpressionFault`] naming the first violation.
    pub fn validate_program(
        &self,
        allow_generated: bool,
        declared: &PolicyReads,
    ) -> Result<(), ExpressionFault> {
        let found = self.value_type()?;
        if found != ExprType::Number {
            return Err(ExpressionFault::OutputType { found });
        }
        if !allow_generated && self.reads_generated() {
            return Err(ExpressionFault::GeneratedInAssignment);
        }
        if &self.reads() != declared {
            return Err(ExpressionFault::DeclaredReadsMismatch);
        }
        Ok(())
    }

    /// Whether every value the expression can take is `0` or `1`: a constant 0 or 1, an
    /// [`Self::indicator`], or a conditional whose branches both are.
    #[must_use]
    #[allow(clippy::float_cmp, reason = "membership of exactly the values 0 and 1")]
    pub fn is_binary_valued(&self) -> bool {
        match self {
            Self::Const(c) => c.get() == 0.0 || c.get() == 1.0,
            Self::Indicator(_) => true,
            Self::If { then_branch, else_branch, .. } => {
                then_branch.is_binary_valued() && else_branch.is_binary_valued()
            }
            Self::Callback(call) => matches!(
                &call.spec().output,
                CallbackOutput::Levels(levels) if levels.iter().all(|l| *l == 0.0 || *l == 1.0)
            ),
            _ => false,
        }
    }

    /// The declared levels when the expression is a bare callback with a level output;
    /// execution validates every returned value against them.
    #[must_use]
    pub fn callback_levels(&self) -> Option<&[f64]> {
        match self {
            Self::Callback(call) => match &call.spec().output {
                CallbackOutput::Levels(levels) => Some(levels),
                _ => None,
            },
            _ => None,
        }
    }

    /// Every callback call in the expression, in left-to-right leaf order (the order the
    /// program compiler emits them in).
    #[must_use]
    pub fn callbacks(&self) -> Vec<&CallbackCall> {
        let mut found = Vec::new();
        self.collect_callbacks(&mut found);
        found
    }

    fn collect_callbacks<'a>(&'a self, found: &mut Vec<&'a CallbackCall>) {
        match self {
            Self::Const(_) | Self::Generated | Self::Var(_) | Self::Lag { .. } => {}
            Self::Neg(a) | Self::Not(a) | Self::Indicator(a) => a.collect_callbacks(found),
            Self::Arith { lhs, rhs, .. }
            | Self::Compare { lhs, rhs, .. }
            | Self::Logic { lhs, rhs, .. } => {
                lhs.collect_callbacks(found);
                rhs.collect_callbacks(found);
            }
            Self::Clip { value, lower, upper } => {
                value.collect_callbacks(found);
                lower.collect_callbacks(found);
                upper.collect_callbacks(found);
            }
            Self::If { condition, then_branch, else_branch } => {
                condition.collect_callbacks(found);
                then_branch.collect_callbacks(found);
                else_branch.collect_callbacks(found);
            }
            Self::Callback(call) => {
                for arg in call.args() {
                    arg.collect_callbacks(found);
                }
                found.push(call);
            }
        }
    }

    /// Whether the expression can leave the process. Closed-form expressions are
    /// portable; a callback is portable only as a verifiable recovery record.
    #[must_use]
    pub fn portability(&self) -> Portability {
        let mut records = Vec::new();
        let mut bound = Vec::new();
        for call in self.callbacks() {
            match call.spec().recovery_record() {
                Some(record) => records.push(record),
                None => bound.push(call.spec().name.clone()),
            }
        }
        if !bound.is_empty() {
            Portability::RuntimeBound { callbacks: bound }
        } else if records.is_empty() {
            Portability::Portable
        } else {
            Portability::Recoverable(records)
        }
    }

    /// The constant the expression folds to when it reads nothing.
    #[must_use]
    pub fn as_constant(&self) -> Option<f64> {
        self.evaluate(None, &|_| None).filter(|v| v.is_finite())
    }

    /// Reference scalar interpreter: evaluate with `generated` and a current-value
    /// lookup. Booleans evaluate to `0.0` / `1.0`. `None` when a read is missing or the
    /// expression reads history (no history is available to a scalar lookup).
    ///
    /// The batch kernel applies the same operator functions row by row, so it agrees
    /// with this interpreter bit for bit.
    #[must_use]
    #[allow(clippy::float_cmp, reason = "a boolean is stored as exactly 0.0 or 1.0")]
    pub fn evaluate(
        &self,
        generated: Option<f64>,
        current: &dyn Fn(VariableId) -> Option<f64>,
    ) -> Option<f64> {
        let go = |e: &Self| e.evaluate(generated, current);
        let truth = |x: f64| x != 0.0;
        Some(match self {
            Self::Const(c) => c.get(),
            Self::Generated => generated?,
            Self::Var(v) => current(*v)?,
            // A history read has no value in a scalar lookup, and a callback is a batch
            // function with possible side effects that the reference interpreter never calls.
            Self::Lag { .. } | Self::Callback(_) => return None,
            Self::Neg(a) => -go(a)?,
            Self::Arith { op, lhs, rhs } => op.apply(go(lhs)?, go(rhs)?),
            Self::Clip { value, lower, upper } => clip_value(go(value)?, go(lower)?, go(upper)?),
            Self::Compare { op, lhs, rhs } => f64::from(u8::from(op.apply(go(lhs)?, go(rhs)?))),
            Self::Not(a) => f64::from(u8::from(!truth(go(a)?))),
            Self::Logic { op, lhs, rhs } => {
                let (a, b) = (truth(go(lhs)?), truth(go(rhs)?));
                f64::from(u8::from(match op {
                    LogicOp::And => a && b,
                    LogicOp::Or => a || b,
                }))
            }
            Self::If { condition, then_branch, else_branch } => {
                let (t, e) = (go(then_branch)?, go(else_branch)?);
                if truth(go(condition)?) { t } else { e }
            }
            Self::Indicator(a) => go(a)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(raw: u32) -> VariableId {
        VariableId::from_raw(raw)
    }

    #[test]
    fn price_floor_types_as_a_number_and_reads_only_the_generated_value() {
        let floor =
            ValueExpr::constant(10.0).max(ValueExpr::constant(0.85).mul(ValueExpr::generated()));
        assert_eq!(floor.value_type(), Ok(ExprType::Number));
        assert!(floor.reads_generated());
        assert!(floor.reads().is_empty());
        floor.validate_program(true, &PolicyReads::default()).unwrap();
        assert_eq!(
            floor.validate_program(false, &PolicyReads::default()),
            Err(ExpressionFault::GeneratedInAssignment)
        );
    }

    #[test]
    fn a_comparison_needs_an_explicit_indicator_to_be_a_node_value() {
        let flag = ValueExpr::var(v(1)).gt(ValueExpr::constant(0.0));
        assert_eq!(flag.value_type(), Ok(ExprType::Boolean));
        assert_eq!(
            flag.validate_program(true, &flag.reads()),
            Err(ExpressionFault::OutputType { found: ExprType::Boolean })
        );
        let numeric = flag.clone().indicator();
        numeric.validate_program(true, &numeric.reads()).unwrap();
    }

    #[test]
    fn operand_and_branch_types_are_checked() {
        let mixed = ValueExpr::constant(1.0).add(ValueExpr::var(v(0)).gt(ValueExpr::constant(0.0)));
        assert_eq!(
            mixed.value_type(),
            Err(ExpressionFault::TypeMismatch {
                expected: ExprType::Number,
                found: ExprType::Boolean
            })
        );
        let branches = ValueExpr::var(v(0)).gt(ValueExpr::constant(0.0)).if_then_else(
            ValueExpr::constant(1.0),
            ValueExpr::var(v(0)).lt(ValueExpr::constant(0.0)),
        );
        assert_eq!(
            branches.value_type(),
            Err(ExpressionFault::BranchTypes {
                then_type: ExprType::Number,
                else_type: ExprType::Boolean
            })
        );
        let not_number = ValueExpr::constant(1.0).not();
        assert!(matches!(not_number.value_type(), Err(ExpressionFault::TypeMismatch { .. })));
    }

    #[test]
    fn declared_reads_must_equal_the_referenced_variables() {
        let e = ValueExpr::constant(0.5).mul(ValueExpr::var(v(2))).add(ValueExpr::var(v(2)));
        assert_eq!(e.reads(), PolicyReads::new([v(2)], []));
        e.validate_program(true, &PolicyReads::new([v(2)], [])).unwrap();
        for wrong in [
            PolicyReads::default(),
            PolicyReads::new([v(2), v(3)], []),
            PolicyReads::new([v(3)], []),
            PolicyReads::new([v(2)], [(v(2), 1)]),
        ] {
            assert_eq!(
                e.validate_program(true, &wrong),
                Err(ExpressionFault::DeclaredReadsMismatch)
            );
        }
    }

    #[test]
    fn constants_lags_and_size_are_bounded() {
        assert_eq!(
            ValueExpr::constant(f64::NAN).value_type(),
            Err(ExpressionFault::NonFiniteConstant)
        );
        assert_eq!(ValueExpr::lag(v(0), 0).value_type(), Err(ExpressionFault::ZeroLag));
        let mut deep = ValueExpr::constant(1.0);
        for _ in 0..MAX_EXPR_DEPTH {
            deep = deep.neg();
        }
        assert_eq!(deep.value_type(), Err(ExpressionFault::TooLarge));
        let mut wide = ValueExpr::constant(1.0);
        for _ in 0..MAX_EXPR_NODES {
            wide = wide.add(ValueExpr::constant(1.0));
        }
        assert_eq!(wide.value_type(), Err(ExpressionFault::TooLarge));
    }

    #[test]
    fn the_scalar_interpreter_matches_the_closed_forms() {
        let x = v(0);
        let z = v(1);
        let env = |id: VariableId| (id == z).then_some(-0.5);
        let branch = ValueExpr::var(z).gt(ValueExpr::constant(0.0)).if_then_else(
            ValueExpr::generated().add(ValueExpr::constant(2.0)),
            ValueExpr::generated(),
        );
        assert_eq!(branch.evaluate(Some(3.0), &env), Some(3.0));
        let clipped =
            ValueExpr::generated().clip(ValueExpr::constant(-1.0), ValueExpr::constant(1.0));
        assert_eq!(clipped.evaluate(Some(7.0), &env), Some(1.0));
        assert_eq!(clipped.evaluate(Some(-7.0), &env), Some(-1.0));
        assert_eq!(clipped.evaluate(Some(0.25), &env), Some(0.25));
        assert_eq!(ValueExpr::var(x).evaluate(Some(1.0), &env), None);
        assert_eq!(ValueExpr::lag(x, 1).evaluate(Some(1.0), &env), None);
        let inverted =
            ValueExpr::generated().clip(ValueExpr::constant(1.0), ValueExpr::constant(-1.0));
        assert!(inverted.evaluate(Some(0.0), &env).is_some_and(f64::is_nan));
    }

    #[test]
    fn binary_valued_expressions_are_recognized() {
        let flag = ValueExpr::var(v(0)).gt(ValueExpr::constant(0.0));
        assert!(flag.clone().indicator().is_binary_valued());
        assert!(ValueExpr::constant(1.0).is_binary_valued());
        assert!(!ValueExpr::constant(2.0).is_binary_valued());
        assert!(!ValueExpr::var(v(0)).mul(ValueExpr::constant(2.0)).is_binary_valued());
        assert!(
            flag.if_then_else(ValueExpr::constant(1.0), ValueExpr::constant(0.0))
                .is_binary_valued()
        );
    }

    #[test]
    fn constant_folding_needs_no_reads_and_a_finite_result() {
        let six = ValueExpr::constant(2.0).mul(ValueExpr::constant(3.0));
        assert_eq!(six.as_constant(), Some(6.0));
        assert_eq!(ValueExpr::var(v(0)).as_constant(), None);
        assert_eq!(ValueExpr::generated().as_constant(), None);
        let by_zero = ValueExpr::constant(1.0).div(ValueExpr::constant(0.0));
        assert_eq!(by_zero.as_constant(), None);
    }

    #[test]
    fn equality_and_hash_agree_on_constants() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let hash = |e: &ValueExpr| {
            let mut h = DefaultHasher::new();
            e.hash(&mut h);
            h.finish()
        };
        let nan = ValueExpr::constant(f64::NAN);
        assert_eq!(nan, nan.clone());
        assert_eq!(hash(&nan), hash(&nan.clone()));
        let (a, b) = (ValueExpr::constant(0.0), ValueExpr::constant(-0.0));
        assert_eq!(a == b, hash(&a) == hash(&b));
    }
}
