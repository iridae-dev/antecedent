//! Value-expression and callback recovery wire forms.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{Arc, Deserialize, IoError, Serialize, ValueExpr, VariableId};

/// Binary numeric operator wire code.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ArithOpWire {
    /// Addition.
    Add,
    /// Subtraction.
    Sub,
    /// Multiplication.
    Mul,
    /// Division.
    Div,
    /// Minimum.
    Min,
    /// Maximum.
    Max,
}

/// Comparison wire code.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CompareOpWire {
    /// Less than.
    Lt,
    /// Less than or equal.
    Le,
    /// Greater than.
    Gt,
    /// Greater than or equal.
    Ge,
    /// Equal.
    Eq,
    /// Not equal.
    Ne,
}

/// Boolean connective wire code.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LogicOpWire {
    /// Conjunction.
    And,
    /// Disjunction.
    Or,
}

/// Value expression wire: the canonical, ordered encoding of the expression tree.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "snake_case")]
pub enum ValueExprWire {
    /// Numeric constant.
    Const {
        /// Value.
        value: f64,
    },
    /// The node's own generated value.
    Generated,
    /// Current-draw read of a variable.
    Var {
        /// Variable.
        variable: u32,
    },
    /// Read of a variable `lag` steps back.
    Lag {
        /// Variable.
        variable: u32,
        /// Steps back.
        lag: u32,
    },
    /// Negation.
    Neg {
        /// Operand.
        operand: Box<ValueExprWire>,
    },
    /// Binary numeric operation.
    Arith {
        /// Operator.
        op: ArithOpWire,
        /// Left operand.
        lhs: Box<ValueExprWire>,
        /// Right operand.
        rhs: Box<ValueExprWire>,
    },
    /// Clip to bounds.
    Clip {
        /// Clipped value.
        value: Box<ValueExprWire>,
        /// Lower bound.
        lower: Box<ValueExprWire>,
        /// Upper bound.
        upper: Box<ValueExprWire>,
    },
    /// Comparison.
    Compare {
        /// Operator.
        op: CompareOpWire,
        /// Left operand.
        lhs: Box<ValueExprWire>,
        /// Right operand.
        rhs: Box<ValueExprWire>,
    },
    /// Boolean negation.
    Not {
        /// Operand.
        operand: Box<ValueExprWire>,
    },
    /// Boolean connective.
    Logic {
        /// Connective.
        op: LogicOpWire,
        /// Left operand.
        lhs: Box<ValueExprWire>,
        /// Right operand.
        rhs: Box<ValueExprWire>,
    },
    /// Conditional.
    If {
        /// Boolean condition.
        condition: Box<ValueExprWire>,
        /// Value when true.
        then_branch: Box<ValueExprWire>,
        /// Value when false.
        else_branch: Box<ValueExprWire>,
    },
    /// Boolean to number.
    Indicator {
        /// Operand.
        operand: Box<ValueExprWire>,
    },
    /// A recoverable batch callback applied to argument expressions: the callback's
    /// recovery record (identity, never code) and one argument per declared input. Only
    /// the policy-program encoding writes this arm, and only for a program whose callbacks
    /// are all recoverable; an expression without a callback never carries it.
    Callback {
        /// What a consumer verifies against the implementation it supplies.
        record: CallbackRecordWire,
        /// Argument expressions in call order.
        args: Vec<ValueExprWire>,
    },
}

/// Declared output of a recoverable callback on the wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "snake_case")]
pub enum CallbackOutputWire {
    /// Any finite real.
    Real,
    /// Finite reals in `[lower, upper]`.
    Range {
        /// Smallest admitted value.
        lower: f64,
        /// Largest admitted value.
        upper: f64,
    },
    /// Exactly these levels.
    Levels {
        /// Sorted, unique, finite levels.
        #[serde(deserialize_with = "canonical_levels")]
        levels: Vec<f64>,
    },
}

/// Levels exactly as the domain stores them: strictly ascending, so two orderings of one set are
/// not two encodings of the same value.
fn canonical_levels<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Vec<f64>, D::Error> {
    let levels = Vec::<f64>::deserialize(deserializer)?;
    if levels.windows(2).all(|pair| pair[0] < pair[1]) {
        Ok(levels)
    } else {
        Err(serde::de::Error::custom("callback levels must be strictly ascending"))
    }
}

/// Recovery record of a callback on the wire: the content digest of its source, its entry
/// point and its pinned dependency versions.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CallbackRecordWire {
    /// Callback name.
    pub name: String,
    /// Input labels in call order.
    pub inputs: Vec<String>,
    /// Declared output.
    pub output: CallbackOutputWire,
    /// Lower-case hex SHA-256 of the source the callback ran.
    pub source_sha256: String,
    /// Entry point within the source.
    pub entry_point: String,
    /// Pinned dependencies as `(distribution name, exact version)`.
    pub dependencies: Vec<(String, String)>,
}

impl CallbackRecordWire {
    fn from_record(record: &antecedent_core::RecoveryRecord) -> Result<Self, IoError> {
        use antecedent_core::CallbackOutput;
        let output = match &record.output {
            CallbackOutput::Real => CallbackOutputWire::Real,
            CallbackOutput::Range { lower, upper } => {
                CallbackOutputWire::Range { lower: *lower, upper: *upper }
            }
            CallbackOutput::Levels(levels) => {
                CallbackOutputWire::Levels { levels: levels.to_vec() }
            }
            other => {
                return Err(IoError::Convert(format!("unsupported callback output: {other:?}")));
            }
        };
        Ok(Self {
            name: record.name.clone(),
            inputs: record.inputs.to_vec(),
            output,
            source_sha256: record.source_sha256.clone(),
            entry_point: record.entry_point.clone(),
            dependencies: record
                .dependencies
                .iter()
                .map(|pin| (pin.name.clone(), pin.version.clone()))
                .collect(),
        })
    }

    /// The record this wire form stands for.
    #[must_use]
    pub fn to_record(&self) -> antecedent_core::RecoveryRecord {
        use antecedent_core::{CallbackOutput, PinnedDependency};
        antecedent_core::RecoveryRecord {
            name: self.name.clone(),
            inputs: Arc::from(self.inputs.as_slice()),
            output: match &self.output {
                CallbackOutputWire::Real => CallbackOutput::Real,
                CallbackOutputWire::Range { lower, upper } => {
                    CallbackOutput::Range { lower: *lower, upper: *upper }
                }
                CallbackOutputWire::Levels { levels } => CallbackOutput::levels(levels.clone()),
            },
            source_sha256: self.source_sha256.clone(),
            entry_point: self.entry_point.clone(),
            dependencies: self
                .dependencies
                .iter()
                .map(|(name, version)| PinnedDependency {
                    name: name.clone(),
                    version: version.clone(),
                })
                .collect(),
        }
    }
}

/// Stand-in for a callback whose implementation a decoder was not given: it carries the
/// recovery record's declaration and refuses to run.
#[derive(Debug)]
struct UnresolvedCallback(antecedent_core::CallbackSpec);

impl antecedent_core::BatchCallback for UnresolvedCallback {
    fn spec(&self) -> &antecedent_core::CallbackSpec {
        &self.0
    }

    fn call(
        &self,
        _inputs: &[&[f64]],
        _out: &mut [f64],
    ) -> Result<(), antecedent_core::CallbackError> {
        Err(antecedent_core::CallbackError::Raised {
            message: format!(
                "callback {} has no implementation: a decoded recovery record is a declaration; \
                 supply the implementation it identifies when decoding",
                self.0.name
            ),
        })
    }
}

/// The unresolved stand-in for the callback a record identifies.
#[must_use]
pub fn unresolved_callback(
    record: &antecedent_core::RecoveryRecord,
) -> Arc<dyn antecedent_core::BatchCallback> {
    Arc::new(UnresolvedCallback(antecedent_core::CallbackSpec {
        name: record.name.clone(),
        inputs: Arc::clone(&record.inputs),
        output: record.output.clone(),
        dependencies: Arc::clone(&record.dependencies),
        implementation: antecedent_core::CallbackImplementation::Recoverable {
            source_sha256: record.source_sha256.clone(),
            entry_point: record.entry_point.clone(),
        },
    }))
}

/// Refuse a program no durable encoding can carry: one that contains a callback bound to
/// the running process.
pub(super) fn refuse_runtime_bound(
    portability: &antecedent_core::Portability,
) -> Result<(), IoError> {
    match portability.export_refusal() {
        Some(refusal) => Err(IoError::Refused { code: refusal.code, message: refusal.message }),
        None => Ok(()),
    }
}

/// Supplies the implementation a recoverable callback record identifies, or refuses.
pub type ResolveCallback<'r, E> = dyn FnMut(&antecedent_core::RecoveryRecord) -> Result<Arc<dyn antecedent_core::BatchCallback>, E>
    + 'r;

impl ValueExprWire {
    /// Encode.
    ///
    /// # Errors
    ///
    /// A program that cannot leave the process (a runtime-bound callback:
    /// `policy_callback_not_portable`), a recoverable callback (this form carries
    /// closed-form expressions only; see [`Self::from_domain_recoverable`]), or unknown
    /// expression or operator variants.
    pub fn from_domain(expr: &ValueExpr) -> Result<Self, IoError> {
        refuse_runtime_bound(&expr.portability())?;
        Self::encode(expr, false)
    }

    /// Encode an expression whose callbacks may be recoverable: each becomes a
    /// [`Self::Callback`] arm carrying its recovery record.
    ///
    /// # Errors
    ///
    /// As [`Self::from_domain`], except that a recoverable callback is encoded.
    pub fn from_domain_recoverable(expr: &ValueExpr) -> Result<Self, IoError> {
        refuse_runtime_bound(&expr.portability())?;
        Self::encode(expr, true)
    }

    fn encode(expr: &ValueExpr, recoverable: bool) -> Result<Self, IoError> {
        use antecedent_core::{ArithOp, CompareOp, LogicOp};
        let boxed = |e: &ValueExpr| Self::encode(e, recoverable).map(Box::new);
        Ok(match expr {
            ValueExpr::Const(c) => Self::Const { value: c.get() },
            ValueExpr::Generated => Self::Generated,
            ValueExpr::Var(v) => Self::Var { variable: v.raw() },
            ValueExpr::Lag { variable, lag } => Self::Lag { variable: variable.raw(), lag: *lag },
            ValueExpr::Neg(a) => Self::Neg { operand: boxed(a)? },
            ValueExpr::Not(a) => Self::Not { operand: boxed(a)? },
            ValueExpr::Indicator(a) => Self::Indicator { operand: boxed(a)? },
            ValueExpr::Arith { op, lhs, rhs } => Self::Arith {
                op: match op {
                    ArithOp::Add => ArithOpWire::Add,
                    ArithOp::Sub => ArithOpWire::Sub,
                    ArithOp::Mul => ArithOpWire::Mul,
                    ArithOp::Div => ArithOpWire::Div,
                    ArithOp::Min => ArithOpWire::Min,
                    ArithOp::Max => ArithOpWire::Max,
                    other => {
                        return Err(IoError::Convert(format!("unsupported ArithOp: {other:?}")));
                    }
                },
                lhs: boxed(lhs)?,
                rhs: boxed(rhs)?,
            },
            ValueExpr::Compare { op, lhs, rhs } => Self::Compare {
                op: match op {
                    CompareOp::Lt => CompareOpWire::Lt,
                    CompareOp::Le => CompareOpWire::Le,
                    CompareOp::Gt => CompareOpWire::Gt,
                    CompareOp::Ge => CompareOpWire::Ge,
                    CompareOp::Eq => CompareOpWire::Eq,
                    CompareOp::Ne => CompareOpWire::Ne,
                    other => {
                        return Err(IoError::Convert(format!("unsupported CompareOp: {other:?}")));
                    }
                },
                lhs: boxed(lhs)?,
                rhs: boxed(rhs)?,
            },
            ValueExpr::Logic { op, lhs, rhs } => Self::Logic {
                op: match op {
                    LogicOp::And => LogicOpWire::And,
                    LogicOp::Or => LogicOpWire::Or,
                    other => {
                        return Err(IoError::Convert(format!("unsupported LogicOp: {other:?}")));
                    }
                },
                lhs: boxed(lhs)?,
                rhs: boxed(rhs)?,
            },
            ValueExpr::Clip { value, lower, upper } => {
                Self::Clip { value: boxed(value)?, lower: boxed(lower)?, upper: boxed(upper)? }
            }
            ValueExpr::If { condition, then_branch, else_branch } => Self::If {
                condition: boxed(condition)?,
                then_branch: boxed(then_branch)?,
                else_branch: boxed(else_branch)?,
            },
            ValueExpr::Callback(call) if recoverable => {
                let record = call.spec().recovery_record().ok_or_else(|| {
                    IoError::Convert(format!(
                        "callback {} has no recovery record",
                        call.spec().name
                    ))
                })?;
                let mut args = Vec::with_capacity(call.args().len());
                for arg in call.args() {
                    args.push(Self::encode(arg, true)?);
                }
                Self::Callback { record: CallbackRecordWire::from_record(&record)?, args }
            }
            ValueExpr::Callback(call) => {
                return Err(IoError::Refused {
                    code: antecedent_core::reason_code!("policy_callback_not_portable"),
                    message: format!(
                        "policy_callback.wire_form_unavailable: callback {} is recoverable, but \
                         the expression wire carries closed-form expressions only",
                        call.spec().name
                    ),
                });
            }
            other => {
                return Err(IoError::Convert(format!("unsupported ValueExpr: {other:?}")));
            }
        })
    }

    /// Decode. A [`Self::Callback`] arm decodes to an unresolved stand-in that declares
    /// the record and refuses to run; use [`Self::to_domain_with`] to bind implementations.
    #[must_use]
    pub fn to_domain(&self) -> ValueExpr {
        self.to_domain_with(&mut |record| {
            Ok::<_, std::convert::Infallible>(unresolved_callback(record))
        })
        .unwrap_or_else(|never| match never {})
    }

    /// Decode, binding each [`Self::Callback`] arm to the implementation `resolve` returns
    /// for its recovery record.
    ///
    /// # Errors
    ///
    /// Whatever `resolve` refuses with.
    pub fn to_domain_with<E>(&self, resolve: &mut ResolveCallback<'_, E>) -> Result<ValueExpr, E> {
        use antecedent_core::{ArithOp, CompareOp, LogicOp};
        let mut un = |e: &Self| e.to_domain_with(&mut *resolve).map(Box::new);
        Ok(match self {
            Self::Const { value } => ValueExpr::constant(*value),
            Self::Generated => ValueExpr::Generated,
            Self::Var { variable } => ValueExpr::Var(VariableId::from_raw(*variable)),
            Self::Lag { variable, lag } => {
                ValueExpr::Lag { variable: VariableId::from_raw(*variable), lag: *lag }
            }
            Self::Neg { operand } => ValueExpr::Neg(un(operand)?),
            Self::Not { operand } => ValueExpr::Not(un(operand)?),
            Self::Indicator { operand } => ValueExpr::Indicator(un(operand)?),
            Self::Arith { op, lhs, rhs } => ValueExpr::Arith {
                op: match op {
                    ArithOpWire::Add => ArithOp::Add,
                    ArithOpWire::Sub => ArithOp::Sub,
                    ArithOpWire::Mul => ArithOp::Mul,
                    ArithOpWire::Div => ArithOp::Div,
                    ArithOpWire::Min => ArithOp::Min,
                    ArithOpWire::Max => ArithOp::Max,
                },
                lhs: un(lhs)?,
                rhs: un(rhs)?,
            },
            Self::Compare { op, lhs, rhs } => ValueExpr::Compare {
                op: match op {
                    CompareOpWire::Lt => CompareOp::Lt,
                    CompareOpWire::Le => CompareOp::Le,
                    CompareOpWire::Gt => CompareOp::Gt,
                    CompareOpWire::Ge => CompareOp::Ge,
                    CompareOpWire::Eq => CompareOp::Eq,
                    CompareOpWire::Ne => CompareOp::Ne,
                },
                lhs: un(lhs)?,
                rhs: un(rhs)?,
            },
            Self::Logic { op, lhs, rhs } => ValueExpr::Logic {
                op: match op {
                    LogicOpWire::And => LogicOp::And,
                    LogicOpWire::Or => LogicOp::Or,
                },
                lhs: un(lhs)?,
                rhs: un(rhs)?,
            },
            Self::Clip { value, lower, upper } => {
                ValueExpr::Clip { value: un(value)?, lower: un(lower)?, upper: un(upper)? }
            }
            Self::If { condition, then_branch, else_branch } => ValueExpr::If {
                condition: un(condition)?,
                then_branch: un(then_branch)?,
                else_branch: un(else_branch)?,
            },
            Self::Callback { record, args } => {
                let callback = resolve(&record.to_record())?;
                let mut decoded = Vec::with_capacity(args.len());
                for arg in args {
                    decoded.push(arg.to_domain_with(&mut *resolve)?);
                }
                ValueExpr::callback(callback, decoded)
            }
        })
    }
}
