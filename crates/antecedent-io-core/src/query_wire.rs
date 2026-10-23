//! Full `CausalQuery` and Intervention wire forms.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AllocationMethod, AnomalyAttributionQuery, AnomalyReference, AssignmentDesign,
    AttributionComponents, AverageEffectQuery, CausalQuery, ChangeAttributionQuery,
    ConditionalEffectQuery, CounterfactualQuery, DistributionRef, DynamicRuleId, EnvironmentId,
    ExposureLevel, ExposureMapping, InterferenceFunctional, InterferenceQuery, Intervention,
    InterventionSequence, InterventionalDistributionQuery, LocalPolynomialRatioQuery,
    MechanismChangeQuery, MechanismOverride, MediationContrast, MediationQuery, OrderedFloatBits,
    OutcomeFunctional, PanelDidQuery, PathSpecificEffectQuery, PolicyValueQuery,
    PopulationRegistry, PopulationSelector, PredicateExpr, RandomizationDesign,
    RandomizedEffectQuery, SequencedIntervention, ShapleyConfig, ShapleyMode, StochasticPolicy,
    SyntheticControlQuery, TargetPopulation, TemporalEffectQuery, TemporalPolicy, TransportQuery,
    UnitChangeQuery, Value, ValueExpr, VariableId,
};
use serde::{Deserialize, Serialize};

use crate::convert::{vars_from_raw, vars_to_raw};
use crate::error::IoError;

#[allow(
    clippy::trivially_copy_pass_by_ref,
    reason = "serde skip_serializing_if requires an fn(&T) -> bool signature"
)]
fn is_false(value: &bool) -> bool {
    !*value
}

/// Deserialize a field that was present on the wire, keeping an explicit
/// `null` distinct from an absent key (which `#[serde(default)]` maps to
/// `None`).
fn present<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    T::deserialize(deserializer).map(Some)
}
use crate::response_wire::{ResponseQueryWire, response_query_from_wire, response_query_to_wire};

/// Wire scalar value.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "snake_case")]
pub enum ValueWire {
    /// Float64.
    Float64(f64),
    /// Int64.
    Int64(i64),
    /// Bool.
    Bool(bool),
    /// Category code.
    Category(u32),
    /// Diagnostic label.
    Label(String),
}

impl ValueWire {
    /// Encode.
    #[must_use]
    pub fn from_value(v: &Value) -> Self {
        match v {
            Value::Float64(x) => Self::Float64(*x),
            Value::Int64(x) => Self::Int64(*x),
            Value::Bool(x) => Self::Bool(*x),
            Value::Category(x) => Self::Category(*x),
            Value::Label(s) => Self::Label(s.to_string()),
        }
    }

    /// Decode.
    #[must_use]
    pub fn to_value(&self) -> Value {
        match self {
            Self::Float64(x) => Value::Float64(*x),
            Self::Int64(x) => Value::Int64(*x),
            Self::Bool(x) => Value::Bool(*x),
            Self::Category(x) => Value::Category(*x),
            Self::Label(s) => Value::Label(Arc::from(s.as_str())),
        }
    }
}

/// Hard set intervention on the wire (kept for posterior/distribution helpers).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SetInterventionWire {
    /// Target variable raw id.
    pub variable: u32,
    /// Assigned value.
    pub value: ValueWire,
}

/// Outcome functional on the wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeFunctionalWire {
    /// Mean (default).
    Mean,
    /// Single exceedance threshold.
    Exceedance(f64),
    /// Exceedance grid.
    ExceedanceGrid(Vec<f64>),
    /// Quantile level τ.
    Quantile(f64),
}

impl Default for OutcomeFunctionalWire {
    fn default() -> Self {
        Self::Mean
    }
}

impl OutcomeFunctionalWire {
    /// Whether this is the omitted backward-compatible default.
    #[must_use]
    pub const fn is_mean(&self) -> bool {
        matches!(self, Self::Mean)
    }
    /// Encode.
    #[must_use]
    pub fn from_domain(f: &OutcomeFunctional) -> Self {
        match f {
            OutcomeFunctional::Exceedance(c) => Self::Exceedance(c.to_f64()),
            OutcomeFunctional::ExceedanceGrid(grid) => {
                Self::ExceedanceGrid(grid.iter().map(|c| c.to_f64()).collect())
            }
            OutcomeFunctional::Quantile(tau) => Self::Quantile(tau.to_f64()),
            _ => Self::Mean,
        }
    }

    /// Decode.
    #[must_use]
    pub fn to_domain(&self) -> OutcomeFunctional {
        match self {
            Self::Mean => OutcomeFunctional::Mean,
            Self::Exceedance(c) => OutcomeFunctional::exceedance(*c),
            Self::ExceedanceGrid(grid) => OutcomeFunctional::exceedance_grid(grid.clone()),
            Self::Quantile(tau) => OutcomeFunctional::quantile(*tau),
        }
    }
}

/// Target population on the wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "snake_case")]
pub enum TargetPopulationWire {
    /// All observed units.
    AllObserved,
    /// Treated units.
    Treated,
    /// Untreated units.
    Untreated,
    /// Environment-restricted.
    Environment(u32),
    /// Named registry predicate with resolved-row digest.
    PredicateNamed {
        /// Registry name.
        name: String,
        /// Digest of resolved row indices as little-endian `u64`.
        rows: [u8; 32],
    },
    /// Explicit row indices.
    PredicateRows(Vec<u64>),
    /// Custom distribution handle with bound weights.
    CustomDistribution {
        /// Registry handle.
        handle: u32,
        /// Weight digest.
        weights: [u8; 32],
        /// Declared parents.
        depends_on: Vec<u32>,
    },
    /// Row-weight retarget.
    RowWeights {
        /// Weight digest.
        weights: [u8; 32],
        /// Declared parents.
        depends_on: Vec<u32>,
    },
    /// Units at a running-variable cutoff (the sharp-RD limit population).
    LocalAtCutoff {
        /// Running variable.
        running: u32,
        /// Cutoff.
        cutoff: f64,
    },
}

impl TargetPopulationWire {
    /// Encode.
    ///
    /// # Errors
    ///
    /// Unknown variants or row indices that do not fit `u64`.
    pub fn from_domain(p: &TargetPopulation) -> Result<Self, IoError> {
        Self::from_domain_with_registry(p, None)
    }

    /// Encode a population, resolving registry-backed variants.
    ///
    /// # Errors
    ///
    /// Unknown variants, missing registry entries, or row indices that do not fit `u64`.
    pub fn from_domain_with_registry(
        p: &TargetPopulation,
        registry: Option<&PopulationRegistry>,
    ) -> Result<Self, IoError> {
        Ok(match p {
            TargetPopulation::AllObserved => Self::AllObserved,
            TargetPopulation::Treated => Self::Treated,
            TargetPopulation::Untreated => Self::Untreated,
            TargetPopulation::Environment(id) => Self::Environment(id.raw()),
            TargetPopulation::Predicate(PredicateExpr::Named(name)) => {
                let registry = registry.ok_or_else(|| {
                    IoError::Convert("population registry required for named predicate".into())
                })?;
                let rows = registry
                    .predicate(name)
                    .ok_or_else(|| IoError::Convert(format!("unknown predicate {name}")))?;
                let mut bytes = Vec::with_capacity(rows.len() * 8);
                for &row in rows {
                    bytes.extend_from_slice(
                        &u64::try_from(row).map_err(|_| IoError::TooLarge)?.to_le_bytes(),
                    );
                }
                Self::PredicateNamed {
                    name: name.to_string(),
                    rows: crate::payload_digest("population.predicate_rows", &bytes),
                }
            }
            TargetPopulation::Predicate(PredicateExpr::Rows(rows)) => Self::PredicateRows(
                rows.iter()
                    .map(|&r| u64::try_from(r).map_err(|_| IoError::TooLarge))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            TargetPopulation::Predicate(other) => {
                return Err(IoError::Convert(format!(
                    "unsupported PredicateExpr for query wire: {other:?}"
                )));
            }
            TargetPopulation::CustomDistribution(r) => {
                let registry = registry.ok_or_else(|| {
                    IoError::Convert("population registry required for custom distribution".into())
                })?;
                let weights = registry.distribution(*r).ok_or_else(|| {
                    IoError::Convert(format!("unknown distribution handle {}", r.raw()))
                })?;
                let mut bytes = Vec::with_capacity(weights.len() * 8);
                for &weight in weights {
                    bytes.extend_from_slice(&weight.to_bits().to_le_bytes());
                }
                Self::CustomDistribution {
                    handle: r.raw(),
                    weights: crate::payload_digest("population.distribution_weights", &bytes),
                    depends_on: registry
                        .distribution_dependencies(*r)
                        .unwrap_or(&[])
                        .iter()
                        .map(|id| id.raw())
                        .collect(),
                }
            }
            TargetPopulation::RowWeights { weights, depends_on } => Self::RowWeights {
                weights: *weights,
                depends_on: depends_on.iter().map(|id| id.raw()).collect(),
            },
            TargetPopulation::LocalAtCutoff { running, cutoff } => {
                Self::LocalAtCutoff { running: running.raw(), cutoff: cutoff.to_f64() }
            }
            other => {
                return Err(IoError::Convert(format!(
                    "unsupported TargetPopulation for query wire: {other:?}"
                )));
            }
        })
    }

    /// Decode.
    ///
    /// # Errors
    ///
    /// Row indices that do not fit `usize`.
    pub fn to_domain(&self) -> Result<TargetPopulation, IoError> {
        Ok(match self {
            Self::AllObserved => TargetPopulation::AllObserved,
            Self::Treated => TargetPopulation::Treated,
            Self::Untreated => TargetPopulation::Untreated,
            Self::Environment(raw) => TargetPopulation::Environment(EnvironmentId::from_raw(*raw)),
            Self::PredicateNamed { name, .. } => {
                TargetPopulation::Predicate(PredicateExpr::named(name.as_str()))
            }
            Self::PredicateRows(rows) => {
                let idxs = rows
                    .iter()
                    .map(|&r| usize::try_from(r).map_err(|_| IoError::TooLarge))
                    .collect::<Result<Vec<_>, _>>()?;
                TargetPopulation::Predicate(PredicateExpr::rows(idxs))
            }
            Self::CustomDistribution { handle, .. } => {
                TargetPopulation::CustomDistribution(DistributionRef::from_raw(*handle))
            }
            Self::RowWeights { weights, depends_on } => TargetPopulation::RowWeights {
                weights: *weights,
                depends_on: depends_on.iter().copied().map(VariableId::from_raw).collect(),
            },
            Self::LocalAtCutoff { running, cutoff } => {
                TargetPopulation::local_at_cutoff(VariableId::from_raw(*running), *cutoff)
            }
        })
    }
}

/// Temporal policy wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "snake_case")]
pub enum TemporalPolicyWire {
    /// Pulse.
    Pulse {
        /// Offset.
        at: i32,
    },
    /// Sustained inclusive range.
    Sustained {
        /// From.
        from: i32,
        /// Until.
        until: i32,
    },
    /// Dynamic rule handle with explicit active offsets.
    Dynamic {
        /// Rule id.
        rule: u32,
        /// Active step offsets (sorted unique).
        #[serde(default)]
        active_at: Vec<i32>,
    },
}

impl TemporalPolicyWire {
    /// Encode a domain temporal policy.
    ///
    /// # Errors
    ///
    /// Unknown policy variants.
    pub fn from_domain(p: &TemporalPolicy) -> Result<Self, IoError> {
        Ok(match p {
            TemporalPolicy::Pulse { at } => Self::Pulse { at: *at },
            TemporalPolicy::Sustained { from, until } => {
                Self::Sustained { from: *from, until: *until }
            }
            TemporalPolicy::Dynamic { rule, active_at } => {
                Self::Dynamic { rule: rule.raw(), active_at: active_at.as_ref().to_vec() }
            }
            other => {
                return Err(IoError::Convert(format!("unsupported TemporalPolicy: {other:?}")));
            }
        })
    }

    /// Decode a domain temporal policy.
    #[must_use]
    pub fn to_domain(&self) -> TemporalPolicy {
        match self {
            Self::Pulse { at } => TemporalPolicy::Pulse { at: *at },
            Self::Sustained { from, until } => {
                TemporalPolicy::Sustained { from: *from, until: *until }
            }
            Self::Dynamic { rule, active_at } => {
                TemporalPolicy::dynamic(DynamicRuleId::from_raw(*rule), active_at.as_slice())
            }
        }
    }
}

/// Stochastic policy wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "snake_case")]
pub enum StochasticPolicyWire {
    /// Bernoulli.
    Bernoulli {
        /// p.
        p: f64,
    },
    /// Gaussian.
    Gaussian {
        /// Mean.
        mean: f64,
        /// Variance.
        variance: f64,
    },
    /// Categorical.
    Categorical {
        /// Probabilities.
        probs: Vec<f64>,
    },
}

impl StochasticPolicyWire {
    fn from_domain(p: &StochasticPolicy) -> Result<Self, IoError> {
        Ok(match p {
            StochasticPolicy::Bernoulli { p } => Self::Bernoulli { p: *p },
            StochasticPolicy::Gaussian { mean, variance } => {
                Self::Gaussian { mean: *mean, variance: *variance }
            }
            StochasticPolicy::Categorical { probs } => Self::Categorical { probs: probs.to_vec() },
            other => {
                return Err(IoError::Convert(format!("unsupported StochasticPolicy: {other:?}")));
            }
        })
    }

    #[must_use]
    fn to_domain(&self) -> StochasticPolicy {
        match self {
            Self::Bernoulli { p } => StochasticPolicy::Bernoulli { p: *p },
            Self::Gaussian { mean, variance } => {
                StochasticPolicy::Gaussian { mean: *mean, variance: *variance }
            }
            Self::Categorical { probs } => {
                StochasticPolicy::Categorical { probs: Arc::from(probs.as_slice()) }
            }
        }
    }
}

/// Soft mechanism override wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MechanismOverrideWire {
    /// Family id.
    pub family_id: String,
    /// Parameters.
    pub parameters: Vec<f64>,
}

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
        levels: Vec<f64>,
    },
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
fn refuse_runtime_bound(portability: &antecedent_core::Portability) -> Result<(), IoError> {
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

/// Declared reads of a value program, as written (decoded verbatim so validation can
/// compare them with what the expression references).
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PolicyReadsWire {
    /// Current-draw reads (raw variable ids).
    pub current: Vec<u32>,
    /// History reads as `(raw variable id, lag)`.
    pub history: Vec<(u32, u32)>,
}

impl PolicyReadsWire {
    fn from_domain(reads: &antecedent_core::PolicyReads) -> Self {
        Self {
            current: reads.current.iter().map(|v| v.raw()).collect(),
            history: reads.history.iter().map(|(v, lag)| (v.raw(), *lag)).collect(),
        }
    }

    fn to_domain(&self) -> antecedent_core::PolicyReads {
        antecedent_core::PolicyReads::new(
            self.current.iter().map(|v| VariableId::from_raw(*v)),
            self.history.iter().map(|(v, lag)| (VariableId::from_raw(*v), *lag)),
        )
    }
}

/// Full intervention wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "snake_case")]
pub enum InterventionWire {
    /// Hard set.
    Set {
        /// Variable.
        variable: u32,
        /// Value.
        value: ValueWire,
    },
    /// Shift.
    Shift {
        /// Variable.
        variable: u32,
        /// Delta.
        delta: ValueWire,
    },
    /// Stochastic.
    Stochastic {
        /// Variable.
        variable: u32,
        /// Policy.
        policy: StochasticPolicyWire,
    },
    /// Soft.
    Soft {
        /// Variable.
        variable: u32,
        /// Mechanism.
        mechanism: MechanismOverrideWire,
    },
    /// Transform of the generated value by an expression.
    Transform {
        /// Variable.
        variable: u32,
        /// Expression.
        expr: ValueExprWire,
        /// Declared reads.
        reads: PolicyReadsWire,
    },
    /// Assignment from an expression of other variables.
    Assign {
        /// Variable.
        variable: u32,
        /// Expression.
        expr: ValueExprWire,
        /// Declared reads.
        reads: PolicyReadsWire,
    },
    /// Sequence.
    Sequence {
        /// Steps.
        steps: Vec<SequencedInterventionWire>,
    },
}

/// Sequenced intervention step.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SequencedInterventionWire {
    /// Nested intervention.
    pub intervention: Box<InterventionWire>,
    /// Temporal policy.
    pub temporal: TemporalPolicyWire,
}

impl InterventionWire {
    /// Encode.
    ///
    /// # Errors
    ///
    /// A program holding a callback (`policy_callback_not_portable`; see
    /// [`Self::from_domain_recoverable`] for recoverable ones), or unknown intervention
    /// variants.
    pub fn from_domain(iv: &Intervention) -> Result<Self, IoError> {
        Self::encode(iv, false)
    }

    /// Encode an intervention whose callbacks may be recoverable: each is written as its
    /// recovery record (see [`ValueExprWire::Callback`]).
    ///
    /// # Errors
    ///
    /// A runtime-bound callback (`policy_callback_not_portable`), or unknown intervention
    /// variants.
    pub fn from_domain_recoverable(iv: &Intervention) -> Result<Self, IoError> {
        Self::encode(iv, true)
    }

    fn encode(iv: &Intervention, recoverable: bool) -> Result<Self, IoError> {
        refuse_runtime_bound(&iv.portability())?;
        let expr = |expr: &ValueExpr| {
            if recoverable {
                ValueExprWire::from_domain_recoverable(expr)
            } else {
                ValueExprWire::from_domain(expr)
            }
        };
        Ok(match iv {
            Intervention::Set { variable, value } => {
                Self::Set { variable: variable.raw(), value: ValueWire::from_value(value) }
            }
            Intervention::Shift { variable, delta } => {
                Self::Shift { variable: variable.raw(), delta: ValueWire::from_value(delta) }
            }
            Intervention::Stochastic { variable, policy } => Self::Stochastic {
                variable: variable.raw(),
                policy: StochasticPolicyWire::from_domain(policy)?,
            },
            Intervention::Soft { variable, mechanism } => Self::Soft {
                variable: variable.raw(),
                mechanism: MechanismOverrideWire {
                    family_id: mechanism.family_id.to_string(),
                    parameters: mechanism.parameters.to_vec(),
                },
            },
            Intervention::Transform { variable, expr: value, reads } => Self::Transform {
                variable: variable.raw(),
                expr: expr(value)?,
                reads: PolicyReadsWire::from_domain(reads),
            },
            Intervention::Assign { variable, expr: value, reads } => Self::Assign {
                variable: variable.raw(),
                expr: expr(value)?,
                reads: PolicyReadsWire::from_domain(reads),
            },
            Intervention::Sequence(seq) => Self::Sequence {
                steps: seq
                    .steps
                    .iter()
                    .map(|s| {
                        Ok(SequencedInterventionWire {
                            intervention: Box::new(Self::encode(&s.intervention, recoverable)?),
                            temporal: TemporalPolicyWire::from_domain(&s.temporal)?,
                        })
                    })
                    .collect::<Result<Vec<_>, IoError>>()?,
            },
            other => {
                return Err(IoError::Convert(format!("unsupported Intervention: {other:?}")));
            }
        })
    }

    /// Decode. A callback decodes to an unresolved stand-in (see
    /// [`ValueExprWire::to_domain`]).
    #[must_use]
    pub fn to_domain(&self) -> Intervention {
        self.to_domain_with(&mut |record| {
            Ok::<_, std::convert::Infallible>(unresolved_callback(record))
        })
        .unwrap_or_else(|never| match never {})
    }

    /// Decode, binding each callback to the implementation `resolve` returns for its
    /// recovery record.
    ///
    /// # Errors
    ///
    /// Whatever `resolve` refuses with.
    pub fn to_domain_with<E>(
        &self,
        resolve: &mut ResolveCallback<'_, E>,
    ) -> Result<Intervention, E> {
        Ok(match self {
            Self::Set { variable, value } => {
                Intervention::set(VariableId::from_raw(*variable), value.to_value())
            }
            Self::Shift { variable, delta } => {
                Intervention::shift(VariableId::from_raw(*variable), delta.to_value())
            }
            Self::Stochastic { variable, policy } => {
                Intervention::stochastic(VariableId::from_raw(*variable), policy.to_domain())
            }
            Self::Soft { variable, mechanism } => Intervention::soft(
                VariableId::from_raw(*variable),
                MechanismOverride {
                    family_id: Arc::from(mechanism.family_id.as_str()),
                    parameters: Arc::from(mechanism.parameters.as_slice()),
                },
            ),
            Self::Transform { variable, expr, reads } => Intervention::Transform {
                variable: VariableId::from_raw(*variable),
                expr: expr.to_domain_with(resolve)?,
                reads: reads.to_domain(),
            },
            Self::Assign { variable, expr, reads } => Intervention::Assign {
                variable: VariableId::from_raw(*variable),
                expr: expr.to_domain_with(resolve)?,
                reads: reads.to_domain(),
            },
            Self::Sequence { steps } => {
                let mut decoded = Vec::with_capacity(steps.len());
                for step in steps {
                    decoded.push(SequencedIntervention {
                        intervention: step.intervention.to_domain_with(&mut *resolve)?,
                        temporal: step.temporal.to_domain(),
                    });
                }
                Intervention::sequence(InterventionSequence::new(decoded))
            }
        })
    }
}

/// Wire form of [`InterventionalDistributionQuery`].
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct InterventionalDistributionQueryWire {
    /// Outcome variable raw ids.
    pub outcomes: Vec<u32>,
    /// Interventions (full).
    pub interventions: Vec<InterventionWire>,
    /// Observational conditioning raw ids (empty = unconditional / ID).
    #[serde(default)]
    pub conditioning: Vec<u32>,
    /// Target population.
    pub target_population: TargetPopulationWire,
}

/// Wire form of [`PathSpecificEffectQuery`].
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PathSpecificEffectQueryWire {
    /// Treatment raw id.
    pub treatment: u32,
    /// Outcome raw id.
    pub outcome: u32,
    /// Intermediate path-node raw ids.
    pub path_nodes: Vec<u32>,
    /// Control.
    pub control: InterventionWire,
    /// Active.
    pub active: InterventionWire,
    /// Target population.
    pub target_population: TargetPopulationWire,
    /// Max paths.
    pub max_paths: u64,
    /// Max path length.
    pub max_len: u64,
}

/// Population selector wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "snake_case")]
pub enum PopulationSelectorWire {
    /// All.
    All,
    /// Explicit rows.
    Rows(Vec<u64>),
    /// Environment index.
    Environment {
        /// Index.
        env_index: u64,
    },
    /// Time range.
    TimeRange {
        /// Start.
        start: u64,
        /// End.
        end: u64,
    },
}

/// Full causal query wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "snake_case")]
pub enum CausalQueryWire {
    /// Average effect.
    AverageEffect {
        /// Treatment.
        treatment: u32,
        /// Outcome.
        outcome: u32,
        /// Modifiers.
        effect_modifiers: Vec<u32>,
        /// Control.
        control: InterventionWire,
        /// Active.
        active: InterventionWire,
        /// Population.
        target_population: TargetPopulationWire,
        /// Outcome functional. Absent field decodes as Mean.
        #[serde(default, skip_serializing_if = "OutcomeFunctionalWire::is_mean")]
        outcome_functional: OutcomeFunctionalWire,
    },
    /// Temporal effect.
    TemporalEffect {
        /// Treatment.
        treatment: u32,
        /// Outcome.
        outcome: u32,
        /// Policy.
        policy: TemporalPolicyWire,
        /// Control.
        control: InterventionWire,
        /// Active.
        active: InterventionWire,
        /// Horizon.
        horizon_steps: u32,
        /// Max history lag.
        max_history_lag: Option<u32>,
        /// Population.
        target_population: TargetPopulationWire,
        /// Effect modifiers. Absent on pulse and sustained artifacts that predate
        /// the temporal conditional effect.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        effect_modifiers: Vec<u32>,
    },
    /// Counterfactual.
    Counterfactual {
        /// Outcomes.
        outcomes: Vec<u32>,
        /// Interventions defining the active world.
        interventions: Vec<InterventionWire>,
        /// Control world. Absent on format-0.4 artifacts written before this field; decode as 0.
        #[serde(default)]
        control: Option<InterventionWire>,
        /// Nested flag.
        allow_nested: bool,
    },
    /// Fixed-contract natural direct effect.
    NestedCounterfactual {
        /// Treatment node.
        treatment: u32,
        /// Mediator node.
        mediator: u32,
        /// Outcome node.
        outcome: u32,
        /// Control level IEEE-754 bits.
        control_bits: u64,
        /// Active level IEEE-754 bits.
        active_bits: u64,
    },
    /// Anomaly attribution.
    AnomalyAttribution {
        /// Targets.
        targets: Vec<u32>,
        /// Optional rows.
        unit_rows: Option<Vec<u64>>,
        /// Cap.
        max_units: u64,
        /// Fixed IT-score reference `(center, scale)`; an inner `None` is the
        /// empirical default. The outer `None` records a payload with
        /// no `reference` key so it re-encodes to its original bytes; current
        /// producers write `Some(None)` (an explicit `null`) for empirical.
        #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "present")]
        reference: Option<Option<(f64, f64)>>,
    },
    /// Change attribution.
    ChangeAttribution {
        /// Outcome.
        outcome: u32,
        /// Baseline.
        baseline: PopulationSelectorWire,
        /// Comparison.
        comparison: PopulationSelectorWire,
        /// Components.
        components: AttributionComponentsWire,
        /// Allocation.
        allocation: AllocationMethodWire,
        /// Cap.
        max_components: u64,
    },
    /// Mechanism change.
    MechanismChange {
        /// Targets.
        targets: Vec<u32>,
        /// Baseline.
        baseline: PopulationSelectorWire,
        /// Comparison.
        comparison: PopulationSelectorWire,
        /// Alpha bits.
        significance_level_bits: u64,
        /// Cap.
        max_targets: u64,
    },
    /// Unit change.
    UnitChange {
        /// Outcome.
        outcome: u32,
        /// Rows.
        unit_rows: Option<Vec<u64>>,
        /// Components.
        components: AttributionComponentsWire,
        /// Allocation.
        allocation: AllocationMethodWire,
        /// Cap.
        max_units: u64,
    },
    /// Mediation.
    Mediation {
        /// Treatment.
        treatment: u32,
        /// Outcome.
        outcome: u32,
        /// Mediators.
        mediators: Vec<u32>,
        /// Contrast.
        contrast: MediationContrastWire,
        /// Control.
        control: InterventionWire,
        /// Active.
        active: InterventionWire,
        /// Population.
        target_population: TargetPopulationWire,
        /// Outcome horizons; omitted older payloads default to `[1]`.
        #[serde(default = "default_mediation_horizons")]
        horizons: Vec<u32>,
    },
    /// Conditional effect.
    ConditionalEffect {
        /// Inner average-effect query.
        inner: Box<CausalQueryWire>,
    },
    /// Interventional distribution.
    Distribution(InterventionalDistributionQueryWire),
    /// Path-specific.
    PathSpecific(PathSpecificEffectQueryWire),
    /// Continuous causal response.
    Response(ResponseQueryWire),
    /// Structural transportability query.
    Transport(TransportQueryWire),
    /// Randomized interference query.
    Interference(InterferenceQueryWire),
    /// Graphless Bernoulli randomized ITT query.
    RandomizedEffect(RandomizedEffectQueryWire),
    /// Held-out doubly robust policy value.
    PolicyValue(PolicyValueQueryWire),
    /// Fixed conditional continuous-dose response grid.
    ContinuousDoseResponse(ContinuousDoseResponseQueryWire),
    /// Balanced two-period panel difference-in-differences.
    PanelDid(PanelDidQueryWire),
    /// Balanced-panel synthetic-control design.
    SyntheticControl(SyntheticControlQueryWire),
    /// Fixed-window local fuzzy discontinuity or regression kink.
    LocalPolynomialRatio(LocalPolynomialRatioQueryWire),
    /// Randomized survival or competing-risk query.
    Survival(SurvivalQueryWire),
    /// Prespecified longitudinal treatment regime value.
    LongitudinalRegime(LongitudinalRegimeQueryWire),
    /// Observational two-decision contrast of two static regimes.
    SequentialContrast(SequentialContrastQueryWire),
    /// Nearest-neighbor contrast on histories, matched within calendar periods.
    HistoryMatch(HistoryMatchQueryWire),
    /// Structure discovery.
    Discover(crate::structure_query_wire::DiscoverQueryWire),
    /// Attribution on a fitted structural model.
    Attribute(crate::structure_query_wire::AttributeQueryWire),
    /// Simulation on a fitted structural model.
    Simulate(crate::structure_query_wire::SimulateQueryWire),
    /// Sequentially randomized unit-period design or regime survival.
    TemporalDesign(crate::temporal_design_query_wire::TemporalDesignQueryWire),
}

/// Declared columns and caliper of a within-period history match.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct HistoryMatchQueryWire {
    /// Outcome column.
    pub outcome: u32,
    /// Treatment indicator column.
    pub treated: u32,
    /// Time-ordered history columns.
    pub history: Vec<u32>,
    /// Calendar period column.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub period: Option<u32>,
    /// Sampling-weight column.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight: Option<u32>,
    /// Maximum absolute score distance of a match.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub caliper: Option<f64>,
}

/// Declared columns, regimes and cross-fit folds of an observational sequential contrast.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SequentialContrastQueryWire {
    /// Terminal outcome column.
    pub outcome: u32,
    /// First-decision action column.
    pub first_treatment: u32,
    /// Second-decision action column.
    pub second_treatment: u32,
    /// Baseline covariate columns.
    pub baseline: Vec<u32>,
    /// Covariates measured between the two decisions.
    pub intermediate: Vec<u32>,
    /// Active regime actions.
    pub active: [u8; 2],
    /// Control regime actions.
    pub control: [u8; 2],
    /// Cross-fit fold of each row.
    pub fold_of_row: Vec<u32>,
    /// Number of folds.
    pub fold_count: u32,
    /// Positivity floor.
    pub min_probability: f64,
}

/// Subject-owned longitudinal histories and supplied sequential probabilities.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LongitudinalRegimeQueryWire {
    /// Endpoint outcome variable.
    pub outcome: u32,
    /// Point estimator. Defaults to IPW for older artifacts.
    #[serde(default = "default_longitudinal_method")]
    pub method: String,
    /// Caller-supplied conditional period rewards for g-formula.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub period_outcome_predictions: Vec<f64>,
    /// Caller declaration that g-formula predictions come from a fixed known law.
    #[serde(default)]
    pub known_fixed_outcome_predictions: bool,
    /// Stabilizing treatment-one probabilities, one per period for MSM.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stabilizing_numerator_probabilities: Vec<f64>,
    /// Subject-major conditional Q scores for sequential augmentation.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub q_predictions: Vec<f64>,
    /// Subject-major monotone observation history.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub observation_history: Vec<bool>,
    /// Fold ownership of each subject's Q trajectory.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub prediction_fold_ids: Vec<u32>,
    /// Decisions per subject.
    pub periods: usize,
    /// Observed treatment history, subject-major.
    pub treatment_history: Vec<bool>,
    /// Prescribed regime actions, subject-major.
    pub regime_actions: Vec<bool>,
    /// Conditional treatment-one probabilities, subject-major.
    pub treatment_probabilities: Vec<f64>,
    /// Conditional uncensored probabilities, subject-major.
    pub censoring_probabilities: Vec<f64>,
    /// Whether endpoint was observed, one per subject.
    pub outcome_observed: Vec<bool>,
    /// Unique subject identifiers.
    pub subject_ids: Vec<String>,
    /// Subject-level excluded fold identifiers.
    pub fold_ids: Vec<u32>,
    /// Caller-declared excluded-fold prediction ownership.
    pub excluded_fold_predictions: bool,
    /// Probabilities fixed by a known sequential randomization mechanism.
    pub probabilities_known_by_design: bool,
    /// Sequential positivity floor.
    pub minimum_probability: f64,
    /// Caller supplied identity for a materialized history-adaptive rule.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rule_id: Option<String>,
    /// Stable caller-declared rule version.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rule_version: Option<String>,
    /// Caller-declared source or provenance of the rule implementation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rule_provenance: Option<String>,
}

fn default_longitudinal_method() -> String {
    "ipw".into()
}

/// Right-censored randomized survival query with explicit marginal observation assumption.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SurvivalQueryWire {
    /// Observed follow-up duration column.
    pub duration: u32,
    /// Binary event or coded cause column.
    pub event: u32,
    /// Randomized treatment assignment column.
    pub treatment: u32,
    /// Restriction horizon.
    pub tau: f64,
    /// Optional delayed entry column.
    pub delayed_entry: Option<u32>,
    /// Target cause; absent for survival/RMST.
    pub target_cause: Option<i64>,
    /// Caller declaration of marginal independent censoring/entry.
    pub independent_observation: bool,
    /// Variables conditioning the independent censoring claim.
    #[serde(default)]
    pub independent_given: Vec<u32>,
    /// Caller-supplied censoring survival time grid.
    #[serde(default)]
    pub censoring_times: Vec<f64>,
    /// One row-aligned censoring survival column per grid time.
    #[serde(default)]
    pub censoring_columns: Vec<u32>,
    /// Minimum accepted censoring survival, when a grid is present.
    #[serde(default)]
    pub censoring_probability_floor: Option<f64>,
}

/// Frozen design vectors for a retained panel `DiD` query.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PanelDidQueryWire {
    /// Outcome column id.
    pub outcome: u32,
    /// Pre outcome, propensity, untreated-change prediction, and cross-fit declaration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub augmented: Option<(u32, u32, u32, bool)>,
    /// True for repeated cross sections; absent in older balanced-panel artifacts.
    #[serde(default)]
    pub repeated_cross_section: bool,
    /// Selected staggered cohort-period comparison, when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub staggered_target: Option<(i64, i64)>,
    /// All cohort-specific event-time comparisons, including descriptive preperiods.
    #[serde(default)]
    pub staggered_event_study: bool,
    /// Calendar periods for a staggered panel.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub periods: Vec<i64>,
    /// Adoption cohorts; zero denotes never treated.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cohorts: Vec<i64>,
    /// Row-aligned treatment and period indicators.
    pub treated: Vec<bool>,
    /// Row-aligned pre/post indicator.
    pub post: Vec<bool>,
    /// Subject identity by row.
    pub subjects: Vec<String>,
    /// Cluster identity by row.
    pub clusters: Vec<String>,
    /// Heterogeneity-robust rollout specification; absent for the other designs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rollout: Option<crate::rollout_query_wire::RolloutDidWire>,
    /// Whole-procedure resample of a rollout or augmented panel; absent when none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resampling: Option<crate::rollout_query_wire::DidResamplingWire>,
}

/// Row-aligned synthetic-control design frozen in query identity.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SyntheticControlQueryWire {
    /// Continuous outcome variable.
    pub outcome: u32,
    /// Unit labels in row order.
    pub units: Vec<String>,
    /// Calendar periods in row order.
    pub periods: Vec<i64>,
    /// Unit receiving treatment.
    pub treated_unit: String,
    /// First treated period.
    pub intervention_period: i64,
    /// Whether the panel contrast uses convex unit and pre-period weights.
    #[serde(default)]
    pub difference_in_differences: bool,
    /// Uniformly randomized choice of one treated unit for exact sharp-null inference.
    #[serde(default, skip_serializing_if = "is_false")]
    pub uniform_unit_randomization: bool,
    /// Prespecified constant additive effect under the uniform-unit sharp null.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sharp_null_effect: Option<f64>,
    /// Positive donor outcome-model ridge penalty for augmented synthetic control.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub augmentation_ridge: Option<f64>,
    /// Adopting units and interval construction; absent for a point-only panel.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rollout: Option<crate::rollout_query_wire::SyntheticRolloutWire>,
}

/// Fixed-cutoff local ratio design, including whether the contrast is a kink.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LocalPolynomialRatioQueryWire {
    /// Continuous outcome variable.
    pub outcome: u32,
    /// Observed treatment receipt or dose.
    pub treatment: u32,
    /// Running variable.
    pub running: u32,
    /// Prespecified cutoff.
    pub cutoff: f64,
    /// Prespecified fitting bandwidth.
    pub bandwidth: f64,
    /// Whether to use the slope-kink contrast.
    pub kink: bool,
    /// Bandwidth selection, calendar-time running variable and diagnostics; absent when none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub options: Option<crate::rollout_query_wire::LocalRatioOptionsWire>,
}

/// Frozen policy value inputs retained as part of query identity.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PolicyValueQueryWire {
    /// Outcome variable id.
    pub outcome: u32,
    /// Assignment, propensity, policy, and reference row vectors.
    pub assignment: Vec<bool>,
    /// Known randomized propensity, scalar or row-aligned.
    pub propensity: Vec<f64>,
    /// Policy and reference recommendations.
    pub actions: Vec<bool>,
    /// Reference recommendations.
    pub reference: Vec<bool>,
    /// Frozen nuisance outcome predictions.
    pub mu0: Vec<f64>,
    /// Frozen treated-outcome predictions.
    pub mu1: Vec<f64>,
    /// Policy and reference costs.
    pub costs: Vec<f64>,
    /// Reference costs.
    pub reference_costs: Vec<f64>,
    /// Evaluation subject identity.
    pub evaluation_subject_ids: Vec<String>,
    /// Disjoint-training ownership declaration.
    pub disjoint_training_subjects: bool,
    /// Excluded-fold ownership declaration.
    pub crossfit_fold_ownership_valid: bool,
    /// Whether globally coupled capacity or budget selection was used.
    #[serde(default, skip_serializing_if = "is_false")]
    pub global_constraints_present: bool,
    /// Multi-action randomized IPW inputs, absent for binary policies.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub multi_action: Option<MultiActionPolicyInputsWire>,
    /// Frozen descending-score bin for each binary evaluation row.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub uplift_bins: Vec<usize>,
    /// Number of nonempty uplift bins; zero when no ranked view is requested.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub uplift_bin_count: usize,
    /// Caller-declared ranking training subjects, disjoint from evaluation subjects.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub uplift_training_subject_ids: Vec<String>,
    /// Prespecified fixed candidate class and disjoint selection subjects.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub regret: Option<FixedCandidateRegretWire>,
}

/// Portable fixed-class regret ownership and action vectors.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FixedCandidateRegretWire {
    /// Candidate recommendations in input order.
    pub candidates: Vec<Vec<bool>>,
    /// Selected candidate index.
    pub selected_index: usize,
    /// Disjoint construction and selection subject IDs.
    pub training_subject_ids: Vec<String>,
}

/// Frozen policy and reference group-dose maps for a continuous-dose query.
type FixedPolicyMaps = (Vec<(String, f64)>, Vec<(String, f64)>);

/// Portable caller-supplied conditional continuous-dose response design.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ContinuousDoseResponseQueryWire {
    /// Outcome variable id.
    pub outcome: u32,
    /// Observed dose variable id.
    pub dose: u32,
    /// Supplied dose-density variable id.
    pub dose_density: u32,
    /// Pre-treatment group labels aligned by row.
    pub baseline_groups: Vec<String>,
    /// Prespecified target dose grid.
    pub target_doses: Vec<f64>,
    /// Triangular-kernel bandwidth.
    pub bandwidth: f64,
    /// Minimum local rows per group-target cell.
    pub min_local_support: usize,
    /// Known or externally estimated density.
    pub density_provenance: String,
    /// Frozen policy and reference group-dose maps, when policy value is requested.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fixed_policy: Option<FixedPolicyMaps>,
}

#[allow(
    clippy::trivially_copy_pass_by_ref,
    reason = "serde skip_serializing_if requires an fn(&T) -> bool signature"
)]
fn is_zero(value: &usize) -> bool {
    *value == 0
}

/// Portable row-major multi-action policy inputs.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MultiActionPolicyInputsWire {
    /// Ordered action labels with control first.
    pub action_labels: Vec<String>,
    /// Realized action index per row.
    pub assignment: Vec<usize>,
    /// Policy action index per row.
    pub actions: Vec<usize>,
    /// Reference action index per row.
    pub reference: Vec<usize>,
    /// Row-major assignment probabilities.
    pub propensities: Vec<f64>,
    /// Row-major action availability.
    pub available: Vec<bool>,
    /// Per-action policy costs.
    pub costs: Vec<f64>,
    /// Per-action reference costs.
    pub reference_costs: Vec<f64>,
    /// Per-action policy capacities.
    pub capacities: Vec<usize>,
    /// Per-action reference capacities.
    pub reference_capacities: Vec<usize>,
    /// Policy budget, if constrained.
    pub budget: Option<f64>,
    /// Reference budget, if constrained.
    pub reference_budget: Option<f64>,
    /// Frozen baseline strata for conditional action effects.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cate_groups: Vec<String>,
}

impl From<&antecedent_core::MultiActionPolicyInputs> for MultiActionPolicyInputsWire {
    fn from(value: &antecedent_core::MultiActionPolicyInputs) -> Self {
        Self {
            action_labels: value.action_labels.iter().map(ToString::to_string).collect(),
            assignment: value.assignment.to_vec(),
            actions: value.actions.to_vec(),
            reference: value.reference.to_vec(),
            propensities: value.propensities.to_vec(),
            available: value.available.to_vec(),
            costs: value.costs.to_vec(),
            reference_costs: value.reference_costs.to_vec(),
            capacities: value.capacities.to_vec(),
            reference_capacities: value.reference_capacities.to_vec(),
            budget: value.budget,
            reference_budget: value.reference_budget,
            cate_groups: value.cate_groups.iter().map(ToString::to_string).collect(),
        }
    }
}

impl From<&MultiActionPolicyInputsWire> for antecedent_core::MultiActionPolicyInputs {
    fn from(value: &MultiActionPolicyInputsWire) -> Self {
        Self {
            action_labels: value
                .action_labels
                .iter()
                .map(|v| Arc::<str>::from(v.as_str()))
                .collect::<Vec<_>>()
                .into(),
            assignment: value.assignment.clone().into(),
            actions: value.actions.clone().into(),
            reference: value.reference.clone().into(),
            propensities: value.propensities.clone().into(),
            available: value.available.clone().into(),
            costs: value.costs.clone().into(),
            reference_costs: value.reference_costs.clone().into(),
            capacities: value.capacities.clone().into(),
            reference_capacities: value.reference_capacities.clone().into(),
            budget: value.budget,
            reference_budget: value.reference_budget,
            cate_groups: value
                .cate_groups
                .iter()
                .map(|group| Arc::<str>::from(group.as_str()))
                .collect::<Vec<_>>()
                .into(),
        }
    }
}

/// Randomized ITT design metadata wire form.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RandomizedEffectQueryWire {
    /// Assignment ITT or randomized-encouragement CACE/LATE.
    #[serde(default)]
    pub estimand: RandomizedEstimandWire,
    /// Declared assignment mechanism.
    #[serde(default)]
    pub design: RandomizationDesignWire,
    /// Outcome variable id.
    pub outcome: u32,
    /// Realized assignment in row order.
    pub realized_assignment: Vec<bool>,
    /// Known Bernoulli probabilities in row order.
    pub assignment_probabilities: Vec<f64>,
    /// Assignment unit labels in row order.
    pub assignment_units: Vec<String>,
    /// Outcome unit labels in row order.
    pub outcome_units: Vec<String>,
    /// Control and treatment labels.
    pub treatment_arms: (String, String),
    /// Block labels in row order (stratified designs only).
    #[serde(default)]
    pub blocks: Vec<String>,
    /// Declared treated count for each row's block (stratified designs only).
    #[serde(default)]
    pub treated_per_row: Vec<usize>,
    /// Row-aligned switchback period labels.
    #[serde(default)]
    pub periods: Vec<String>,
    /// Optional pre-assignment covariate and externally fixed CUPED coefficient.
    #[serde(default)]
    pub fixed_cuped: Option<(u32, f64)>,
    /// Pre-assignment covariates jointly fitted by ANCOVA.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ancova_covariates: Vec<u32>,
    /// Row-aligned observed receipt for CACE/LATE.
    #[serde(default)]
    pub received_treatment: Option<Vec<bool>>,
    /// Exhaustive two-sided Fisher sharp-null test for complete randomization.
    #[serde(default)]
    pub exact_randomization_test: bool,
    /// Row-aligned assignment to the second factor in a 2×2 design.
    #[serde(default)]
    pub second_factor_assignment: Vec<bool>,
    /// Fixed cell counts in 00, 10, 01, 11 order.
    #[serde(default)]
    pub factorial_cell_counts: Option<[usize; 4]>,
    /// Labels for the second factor's two levels.
    #[serde(default)]
    pub second_factor_arms: Option<(String, String)>,
    /// Ordered multi-arm labels, first being the reference action.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub multi_arm_labels: Vec<String>,
    /// Observed multi-arm action indices in row order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub multi_arm_assignment: Vec<usize>,
    /// Known action probabilities in declared label order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub multi_arm_probabilities: Vec<Vec<f64>>,
}

/// Target estimand serialized with a randomized query.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RandomizedEstimandWire {
    /// Assignment intention-to-treat contrast.
    #[default]
    Itt,
    /// Wald complier effect under exclusion and monotonicity.
    CaceLate,
    /// Effect among treatment recipients under one-sided noncompliance.
    TreatmentOnTreated,
}

/// Assignment mechanism serialized with a randomized ITT query.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum RandomizationDesignWire {
    /// Independent Bernoulli assignment.
    #[default]
    Bernoulli,
    /// Unit-period switching within independent sequences.
    Switchback,
    /// Complete randomization with a fixed treated count.
    Complete {
        /// Declared number assigned to treatment.
        treated_units: usize,
    },
    /// Complete randomization of assignment clusters.
    Cluster {
        /// Declared number of treated clusters.
        treated_clusters: usize,
    },
    /// Independent complete randomization within blocks.
    Stratified,
    /// Joint complete randomization to four fixed 2×2 cells.
    Factorial2x2,
    /// Independent assignment among three or more actions.
    MultiArm,
}

impl CausalQueryWire {
    /// Mutable target population of a population-scoped query.
    ///
    /// Mirrors [`antecedent_core::CausalQuery::target_population_mut`] variant
    /// for variant. The wire enum is a distinct type with distinct variant
    /// shapes and a distinct population payload
    /// ([`TargetPopulationWire`]), so the table cannot be shared with the core
    /// owner; what keeps the two from drifting is that both matches here and
    /// both matches there are exhaustive, so a new [`CausalQuery`] variant is
    /// a compile error in all four. Which kinds are population-scoped is
    /// decided by [`antecedent_core::CausalQuery::target_population`]; change
    /// it there first.
    pub fn target_population_mut(&mut self) -> Option<&mut TargetPopulationWire> {
        match self {
            Self::AverageEffect { target_population, .. }
            | Self::TemporalEffect { target_population, .. }
            | Self::Mediation { target_population, .. } => Some(target_population),
            Self::Distribution(inner) => Some(&mut inner.target_population),
            Self::PathSpecific(inner) => Some(&mut inner.target_population),
            Self::Response(inner) => Some(&mut inner.target_population),
            Self::ConditionalEffect { inner } => inner.target_population_mut(),
            Self::Counterfactual { .. }
            | Self::NestedCounterfactual { .. }
            | Self::AnomalyAttribution { .. }
            | Self::ChangeAttribution { .. }
            | Self::MechanismChange { .. }
            | Self::UnitChange { .. }
            | Self::Transport(_)
            | Self::Interference(_)
            | Self::RandomizedEffect(_)
            | Self::PolicyValue(_)
            | Self::ContinuousDoseResponse(_)
            | Self::PanelDid(_)
            | Self::SyntheticControl(_)
            | Self::LocalPolynomialRatio(_)
            | Self::Survival(_)
            | Self::LongitudinalRegime(_)
            | Self::SequentialContrast(_)
            | Self::HistoryMatch(_)
            | Self::Discover(_)
            | Self::Attribute(_)
            | Self::Simulate(_)
            | Self::TemporalDesign(_) => None,
        }
    }

    /// Target population of a population-scoped query.
    ///
    /// The wire mirror of [`antecedent_core::CausalQuery::target_population`],
    /// which owns the population-scoped / population-free split; see
    /// [`Self::target_population_mut`] for why the table is mirrored rather
    /// than shared.
    #[must_use]
    pub fn target_population(&self) -> Option<&TargetPopulationWire> {
        match self {
            Self::AverageEffect { target_population, .. }
            | Self::TemporalEffect { target_population, .. }
            | Self::Mediation { target_population, .. } => Some(target_population),
            Self::Distribution(inner) => Some(&inner.target_population),
            Self::PathSpecific(inner) => Some(&inner.target_population),
            Self::Response(inner) => Some(&inner.target_population),
            Self::ConditionalEffect { inner } => inner.target_population(),
            Self::Counterfactual { .. }
            | Self::NestedCounterfactual { .. }
            | Self::AnomalyAttribution { .. }
            | Self::ChangeAttribution { .. }
            | Self::MechanismChange { .. }
            | Self::UnitChange { .. }
            | Self::Transport(_)
            | Self::Interference(_)
            | Self::RandomizedEffect(_)
            | Self::PolicyValue(_)
            | Self::ContinuousDoseResponse(_)
            | Self::PanelDid(_)
            | Self::SyntheticControl(_)
            | Self::LocalPolynomialRatio(_)
            | Self::Survival(_)
            | Self::LongitudinalRegime(_)
            | Self::SequentialContrast(_)
            | Self::HistoryMatch(_)
            | Self::Discover(_)
            | Self::Attribute(_)
            | Self::Simulate(_)
            | Self::TemporalDesign(_) => None,
        }
    }
}

/// Structural transportability query wire form.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TransportQueryWire {
    /// Supplied catalog; omission preserves the legacy experimental contract.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub catalog: Option<crate::transport_catalog_wire::EvidenceCatalogWire>,
    /// Target response query.
    pub response: ResponseQueryWire,
    /// Source population key.
    pub source_population: String,
    /// Target population key.
    pub target_population: String,
    /// Source experimental variables.
    pub source_experiments: Vec<u32>,
}

/// Known assignment design wire form.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "snake_case")]
pub enum AssignmentDesignWire {
    /// Independent Bernoulli assignment.
    Bernoulli {
        /// Scalar or unit-specific assignment probabilities.
        probabilities: Vec<f64>,
    },
    /// Uniform complete randomization.
    CompleteRandomization {
        /// Number of treated units.
        treated: u64,
    },
    /// Uniform cluster randomization.
    ClusterRandomization {
        /// Cluster id per unit.
        clusters: Vec<u32>,
        /// Number of treated clusters.
        treated_clusters: u64,
    },
    /// Complete cluster allocation followed by within-cluster Bernoulli assignment.
    TwoStageSaturation {
        /// Cluster id per unit.
        clusters: Vec<u32>,
        /// Lower treatment probability.
        low_probability: f64,
        /// Higher treatment probability.
        high_probability: f64,
        /// Number of clusters assigned the higher probability.
        high_clusters: u64,
        /// Realized cluster probability repeated per unit.
        realized_saturation: Vec<f64>,
    },
    /// Observed network exposure with supplied probabilities and declared exchangeability.
    ObservedExposure {
        /// Cluster id per unit.
        clusters: Vec<u32>,
        /// Baseline exposure probability per unit.
        propensity_from: Vec<f64>,
        /// Active exposure probability per unit.
        propensity_to: Vec<f64>,
        /// `known` or `externally_estimated`.
        provenance: String,
        /// Caller declaration of network exposure exchangeability.
        assume_network_exchangeability: bool,
    },
}

/// Built-in exposure mapping wire form.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "snake_case")]
pub enum ExposureMappingWire {
    /// Own treatment only.
    OwnTreatment,
    /// Treated-neighbor count.
    NeighborCount,
    /// Treated-neighbor fraction.
    NeighborFraction,
    /// Weighted neighbor mean.
    WeightedNeighborExposure,
    /// Caller registry id.
    Custom(String),
}

/// Exposure level wire form.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ExposureLevelWire {
    /// Own treatment.
    pub own: f64,
    /// Neighbor summary.
    pub neighbors: f64,
}

/// Interference functional wire form.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "snake_case")]
pub enum InterferenceFunctionalWire {
    /// Exposure mean contrast.
    ExposureContrast {
        /// Outcome variable.
        outcome: u32,
        /// Baseline exposure.
        from: ExposureLevelWire,
        /// Active exposure.
        to: ExposureLevelWire,
    },
}

const fn default_probability_draws() -> u32 {
    10_000
}

fn default_mediation_horizons() -> Vec<u32> {
    vec![1]
}

/// Randomized interference query wire form.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct InterferenceQueryWire {
    /// Known assignment design.
    pub assignment: AssignmentDesignWire,
    /// Exposure mapping.
    pub exposure: ExposureMappingWire,
    /// Requested functional.
    pub functional: InterferenceFunctionalWire,
    /// Monte Carlo assignment count; defaults for older payloads which omitted it.
    #[serde(default = "default_probability_draws")]
    pub probability_draws: u32,
}

/// Shapley mode tag, stored as the historical `snake_case` label.
///
/// The sample count stays on [`AllocationMethodWire::Shapley::n`].
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ShapleyModeWire {
    /// Exact enumeration.
    Exact,
    /// Monte Carlo coalition sampling.
    MonteCarlo,
    /// Random permutation sampling.
    Permutation,
}

/// Attribution component set, stored as the historical `snake_case` label.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AttributionComponentsWire {
    /// Input changes only.
    Inputs,
    /// Mechanism changes only.
    Mechanisms,
    /// Graph-structure changes only.
    Structure,
    /// Inputs and mechanisms jointly.
    InputsAndMechanisms,
    /// Full component set.
    All,
}

impl TryFrom<AttributionComponents> for AttributionComponentsWire {
    type Error = IoError;

    fn try_from(components: AttributionComponents) -> Result<Self, Self::Error> {
        Ok(match components {
            AttributionComponents::Inputs => Self::Inputs,
            AttributionComponents::Mechanisms => Self::Mechanisms,
            AttributionComponents::Structure => Self::Structure,
            AttributionComponents::InputsAndMechanisms => Self::InputsAndMechanisms,
            AttributionComponents::All => Self::All,
            _ => return Err(IoError::Convert("unsupported AttributionComponents".into())),
        })
    }
}

impl From<AttributionComponentsWire> for AttributionComponents {
    fn from(components: AttributionComponentsWire) -> Self {
        match components {
            AttributionComponentsWire::Inputs => Self::Inputs,
            AttributionComponentsWire::Mechanisms => Self::Mechanisms,
            AttributionComponentsWire::Structure => Self::Structure,
            AttributionComponentsWire::InputsAndMechanisms => Self::InputsAndMechanisms,
            AttributionComponentsWire::All => Self::All,
        }
    }
}

/// Mediation contrast, stored as the historical `snake_case` label.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MediationContrastWire {
    /// Total effect.
    Total,
    /// Direct effect.
    Direct,
    /// Mediated effect.
    Mediated,
    /// Natural direct effect.
    NaturalDirect,
    /// Natural indirect effect.
    NaturalIndirect,
}

impl From<MediationContrast> for MediationContrastWire {
    fn from(contrast: MediationContrast) -> Self {
        match contrast {
            MediationContrast::Total => Self::Total,
            MediationContrast::Direct => Self::Direct,
            MediationContrast::Mediated => Self::Mediated,
            MediationContrast::NaturalDirect => Self::NaturalDirect,
            MediationContrast::NaturalIndirect => Self::NaturalIndirect,
        }
    }
}

impl From<MediationContrastWire> for MediationContrast {
    fn from(contrast: MediationContrastWire) -> Self {
        match contrast {
            MediationContrastWire::Total => Self::Total,
            MediationContrastWire::Direct => Self::Direct,
            MediationContrastWire::Mediated => Self::Mediated,
            MediationContrastWire::NaturalDirect => Self::NaturalDirect,
            MediationContrastWire::NaturalIndirect => Self::NaturalIndirect,
        }
    }
}

/// Allocation method wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "snake_case")]
pub enum AllocationMethodWire {
    /// Sequential.
    Sequential {
        /// Component raw ids.
        order: Vec<u32>,
    },
    /// Shapley.
    Shapley {
        /// Mode tag.
        mode: ShapleyModeWire,
        /// Mode parameter (samples / permutations).
        n: u64,
        /// Exact component cap.
        max_exact_components: u64,
        /// Override flag.
        allow_exact_override: bool,
        /// Seed.
        seed: u64,
    },
    /// Path-based.
    PathBased,
}

fn population_to_wire(p: &PopulationSelector) -> Result<PopulationSelectorWire, IoError> {
    Ok(match p {
        PopulationSelector::All => PopulationSelectorWire::All,
        PopulationSelector::Rows(rows) => PopulationSelectorWire::Rows(
            rows.iter()
                .map(|&r| u64::try_from(r).map_err(|_| IoError::TooLarge))
                .collect::<Result<Vec<_>, _>>()?,
        ),
        PopulationSelector::Environment { env_index } => PopulationSelectorWire::Environment {
            env_index: u64::try_from(*env_index).map_err(|_| IoError::TooLarge)?,
        },
        PopulationSelector::TimeRange { start, end } => PopulationSelectorWire::TimeRange {
            start: u64::try_from(*start).map_err(|_| IoError::TooLarge)?,
            end: u64::try_from(*end).map_err(|_| IoError::TooLarge)?,
        },
        other => {
            return Err(IoError::Convert(format!("unsupported PopulationSelector: {other:?}")));
        }
    })
}

fn population_from_wire(p: &PopulationSelectorWire) -> Result<PopulationSelector, IoError> {
    Ok(match p {
        PopulationSelectorWire::All => PopulationSelector::All,
        PopulationSelectorWire::Rows(rows) => PopulationSelector::Rows(
            rows.iter()
                .map(|&r| usize::try_from(r).map_err(|_| IoError::TooLarge))
                .collect::<Result<Vec<_>, _>>()?
                .into(),
        ),
        PopulationSelectorWire::Environment { env_index } => PopulationSelector::Environment {
            env_index: usize::try_from(*env_index).map_err(|_| IoError::TooLarge)?,
        },
        PopulationSelectorWire::TimeRange { start, end } => PopulationSelector::TimeRange {
            start: usize::try_from(*start).map_err(|_| IoError::TooLarge)?,
            end: usize::try_from(*end).map_err(|_| IoError::TooLarge)?,
        },
    })
}

fn allocation_to_wire(a: &AllocationMethod) -> Result<AllocationMethodWire, IoError> {
    Ok(match a {
        AllocationMethod::Sequential { order } => {
            AllocationMethodWire::Sequential { order: order.iter().map(|c| c.raw()).collect() }
        }
        AllocationMethod::PathBased => AllocationMethodWire::PathBased,
        AllocationMethod::Shapley { approximation } => {
            let (mode, n) = match approximation.mode {
                ShapleyMode::Exact => (ShapleyModeWire::Exact, 0u64),
                ShapleyMode::MonteCarlo { n_samples } => {
                    (ShapleyModeWire::MonteCarlo, crate::error::fit_u64(n_samples)?)
                }
                ShapleyMode::Permutation { n_permutations } => {
                    (ShapleyModeWire::Permutation, crate::error::fit_u64(n_permutations)?)
                }
                _ => {
                    return Err(IoError::Convert("unsupported ShapleyMode".into()));
                }
            };
            AllocationMethodWire::Shapley {
                mode,
                n,
                max_exact_components: crate::error::fit_u64(approximation.max_exact_components)?,
                allow_exact_override: approximation.allow_exact_override,
                seed: approximation.seed,
            }
        }
        _ => return Err(IoError::Convert("unsupported AllocationMethod".into())),
    })
}

fn allocation_from_wire(a: &AllocationMethodWire) -> Result<AllocationMethod, IoError> {
    Ok(match a {
        AllocationMethodWire::Sequential { order } => AllocationMethod::Sequential {
            order: order
                .iter()
                .copied()
                .map(antecedent_core::ComponentId::from_raw)
                .collect::<Vec<_>>()
                .into(),
        },
        AllocationMethodWire::PathBased => AllocationMethod::PathBased,
        AllocationMethodWire::Shapley {
            mode,
            n,
            max_exact_components,
            allow_exact_override,
            seed,
        } => {
            let n_usize = usize::try_from(*n).map_err(|_| IoError::TooLarge)?;
            let mode = match mode {
                ShapleyModeWire::Exact => ShapleyMode::Exact,
                ShapleyModeWire::MonteCarlo => ShapleyMode::MonteCarlo { n_samples: n_usize },
                ShapleyModeWire::Permutation => {
                    ShapleyMode::Permutation { n_permutations: n_usize }
                }
            };
            AllocationMethod::Shapley {
                approximation: ShapleyConfig {
                    mode,
                    max_exact_components: usize::try_from(*max_exact_components)
                        .map_err(|_| IoError::TooLarge)?,
                    allow_exact_override: *allow_exact_override,
                    seed: *seed,
                },
            }
        }
    })
}

/// Encode any [`CausalQuery`].
///
/// # Errors
///
/// Unsupported nested fields.
pub fn causal_query_to_wire(q: &CausalQuery) -> Result<CausalQueryWire, IoError> {
    causal_query_to_wire_with_registry(q, None)
}

/// Encode any [`CausalQuery`], resolving registry-backed populations.
///
/// # Errors
///
/// Unsupported nested fields or missing registry entries.
#[allow(
    clippy::too_many_lines,
    reason = "one arm per query kind, each converting that kind fields"
)]
pub fn causal_query_to_wire_with_registry(
    q: &CausalQuery,
    registry: Option<&PopulationRegistry>,
) -> Result<CausalQueryWire, IoError> {
    Ok(match q {
        CausalQuery::AverageEffect(q) => CausalQueryWire::AverageEffect {
            treatment: q.treatment.raw(),
            outcome: q.outcome.raw(),
            effect_modifiers: vars_to_raw(&q.effect_modifiers),
            control: InterventionWire::from_domain(&q.control)?,
            active: InterventionWire::from_domain(&q.active)?,
            target_population: TargetPopulationWire::from_domain_with_registry(
                &q.target_population,
                registry,
            )?,
            outcome_functional: OutcomeFunctionalWire::from_domain(&q.outcome_functional),
        },
        CausalQuery::TemporalEffect(q) => CausalQueryWire::TemporalEffect {
            treatment: q.treatment.raw(),
            outcome: q.outcome.raw(),
            policy: TemporalPolicyWire::from_domain(&q.policy)?,
            control: InterventionWire::from_domain(&q.control)?,
            active: InterventionWire::from_domain(&q.active)?,
            horizon_steps: q.horizon_steps,
            max_history_lag: q.max_history_lag,
            target_population: TargetPopulationWire::from_domain_with_registry(
                &q.target_population,
                registry,
            )?,
            effect_modifiers: q.effect_modifiers.iter().map(|id| id.raw()).collect(),
        },
        CausalQuery::Counterfactual(q) => CausalQueryWire::Counterfactual {
            outcomes: vars_to_raw(&q.outcomes),
            interventions: q
                .interventions
                .iter()
                .map(InterventionWire::from_domain)
                .collect::<Result<Vec<_>, _>>()?,
            control: Some(InterventionWire::from_domain(&q.control)?),
            allow_nested: q.allow_nested,
        },
        CausalQuery::NestedCounterfactual(q) => CausalQueryWire::NestedCounterfactual {
            treatment: q.treatment.raw(),
            mediator: q.mediator.raw(),
            outcome: q.outcome.raw(),
            control_bits: q.control_value().to_bits(),
            active_bits: q.active_value().to_bits(),
        },
        CausalQuery::AnomalyAttribution(q) => CausalQueryWire::AnomalyAttribution {
            targets: vars_to_raw(&q.targets),
            unit_rows: q
                .unit_rows
                .as_ref()
                .map(|rows| {
                    rows.iter()
                        .map(|&r| u64::try_from(r).map_err(|_| IoError::TooLarge))
                        .collect::<Result<Vec<_>, _>>()
                })
                .transpose()?,
            max_units: crate::error::fit_u64(q.max_units)?,
            reference: Some(match q.reference {
                AnomalyReference::Empirical => None,
                AnomalyReference::Fixed { center, scale } => {
                    Some((center.to_f64(), scale.to_f64()))
                }
            }),
        },
        CausalQuery::ChangeAttribution(q) => CausalQueryWire::ChangeAttribution {
            outcome: q.outcome.raw(),
            baseline: population_to_wire(&q.baseline)?,
            comparison: population_to_wire(&q.comparison)?,
            components: q.components.try_into()?,
            allocation: allocation_to_wire(&q.allocation)?,
            max_components: crate::error::fit_u64(q.max_components)?,
        },
        CausalQuery::MechanismChange(q) => CausalQueryWire::MechanismChange {
            targets: vars_to_raw(&q.targets),
            baseline: population_to_wire(&q.baseline)?,
            comparison: population_to_wire(&q.comparison)?,
            significance_level_bits: q.significance_level.to_f64().to_bits(),
            max_targets: crate::error::fit_u64(q.max_targets)?,
        },
        CausalQuery::UnitChange(q) => CausalQueryWire::UnitChange {
            outcome: q.outcome.raw(),
            unit_rows: q
                .unit_rows
                .as_ref()
                .map(|rows| {
                    rows.iter()
                        .map(|&r| u64::try_from(r).map_err(|_| IoError::TooLarge))
                        .collect::<Result<Vec<_>, _>>()
                })
                .transpose()?,
            components: q.components.try_into()?,
            allocation: allocation_to_wire(&q.allocation)?,
            max_units: crate::error::fit_u64(q.max_units)?,
        },
        CausalQuery::Mediation(q) => CausalQueryWire::Mediation {
            treatment: q.treatment.raw(),
            outcome: q.outcome.raw(),
            mediators: vars_to_raw(&q.mediators),
            contrast: q.contrast.into(),
            control: InterventionWire::from_domain(&q.control)?,
            active: InterventionWire::from_domain(&q.active)?,
            target_population: TargetPopulationWire::from_domain_with_registry(
                &q.target_population,
                registry,
            )?,
            horizons: q.horizons.iter().copied().collect(),
        },
        CausalQuery::ConditionalEffect(q) => CausalQueryWire::ConditionalEffect {
            inner: Box::new(causal_query_to_wire_with_registry(
                &CausalQuery::AverageEffect(q.inner.clone()),
                registry,
            )?),
        },
        CausalQuery::Distribution(q) => {
            CausalQueryWire::Distribution(interventional_distribution_to_wire(q)?)
        }
        CausalQuery::PathSpecific(q) => CausalQueryWire::PathSpecific(path_specific_to_wire(q)?),
        CausalQuery::Response(q) => CausalQueryWire::Response(response_query_to_wire(q)?),
        CausalQuery::Transport(q) => CausalQueryWire::Transport(transport_query_to_wire(q)?),
        CausalQuery::Interference(q) => {
            CausalQueryWire::Interference(interference_query_to_wire(q)?)
        }
        CausalQuery::RandomizedEffect(q) => {
            CausalQueryWire::RandomizedEffect(RandomizedEffectQueryWire {
                estimand: match q.estimand {
                    antecedent_core::RandomizedEstimand::IntentionToTreat => {
                        RandomizedEstimandWire::Itt
                    }
                    antecedent_core::RandomizedEstimand::ComplierAverageCausalEffect => {
                        RandomizedEstimandWire::CaceLate
                    }
                    antecedent_core::RandomizedEstimand::TreatmentOnTreated => {
                        RandomizedEstimandWire::TreatmentOnTreated
                    }
                },
                design: match &q.design {
                    RandomizationDesign::Bernoulli => RandomizationDesignWire::Bernoulli,
                    RandomizationDesign::Complete { treated_units } => {
                        RandomizationDesignWire::Complete { treated_units: *treated_units }
                    }
                    RandomizationDesign::Cluster { treated_clusters } => {
                        RandomizationDesignWire::Cluster { treated_clusters: *treated_clusters }
                    }
                    RandomizationDesign::Stratified { .. } => RandomizationDesignWire::Stratified,
                    RandomizationDesign::Factorial2x2 { .. } => {
                        RandomizationDesignWire::Factorial2x2
                    }
                    RandomizationDesign::MultiArm { .. } => RandomizationDesignWire::MultiArm,
                    RandomizationDesign::Switchback { .. } => RandomizationDesignWire::Switchback,
                },
                outcome: q.outcome.raw(),
                realized_assignment: q.realized_assignment.to_vec(),
                assignment_probabilities: q.assignment_probabilities.to_vec(),
                assignment_units: q.assignment_units.iter().map(ToString::to_string).collect(),
                outcome_units: q.outcome_units.iter().map(ToString::to_string).collect(),
                treatment_arms: (q.treatment_arms.0.to_string(), q.treatment_arms.1.to_string()),
                blocks: match &q.design {
                    RandomizationDesign::Stratified { blocks, .. } => {
                        blocks.iter().map(ToString::to_string).collect()
                    }
                    _ => Vec::new(),
                },
                treated_per_row: match &q.design {
                    RandomizationDesign::Stratified { treated_per_row, .. } => {
                        treated_per_row.to_vec()
                    }
                    _ => Vec::new(),
                },
                fixed_cuped: q.fixed_cuped.map(|(id, coefficient)| (id.raw(), coefficient)),
                ancova_covariates: q.ancova_covariates.iter().map(|id| id.raw()).collect(),
                received_treatment: q.received_treatment.as_ref().map(|receipt| receipt.to_vec()),
                exact_randomization_test: q.exact_randomization_test,
                second_factor_assignment: match &q.design {
                    RandomizationDesign::Factorial2x2 { second_factor_assignment, .. } => {
                        second_factor_assignment.to_vec()
                    }
                    _ => Vec::new(),
                },
                factorial_cell_counts: match &q.design {
                    RandomizationDesign::Factorial2x2 { cell_counts, .. } => Some(*cell_counts),
                    _ => None,
                },
                second_factor_arms: match &q.design {
                    RandomizationDesign::Factorial2x2 { second_factor_arms, .. } => {
                        Some((second_factor_arms.0.to_string(), second_factor_arms.1.to_string()))
                    }
                    _ => None,
                },
                multi_arm_labels: match &q.design {
                    RandomizationDesign::MultiArm { arms, .. } => {
                        arms.iter().map(ToString::to_string).collect()
                    }
                    _ => Vec::new(),
                },
                multi_arm_assignment: match &q.design {
                    RandomizationDesign::MultiArm { assignment, .. } => assignment.to_vec(),
                    _ => Vec::new(),
                },
                multi_arm_probabilities: match &q.design {
                    RandomizationDesign::MultiArm { probabilities, .. } => probabilities.to_vec(),
                    _ => Vec::new(),
                },
                periods: match &q.design {
                    RandomizationDesign::Switchback { periods } => {
                        periods.iter().map(ToString::to_string).collect()
                    }
                    _ => Vec::new(),
                },
            })
        }
        CausalQuery::PolicyValue(q) => CausalQueryWire::PolicyValue(PolicyValueQueryWire {
            outcome: q.outcome.raw(),
            assignment: q.assignment.to_vec(),
            propensity: q.propensity.to_vec(),
            actions: q.actions.to_vec(),
            reference: q.reference.to_vec(),
            mu0: q.mu0.to_vec(),
            mu1: q.mu1.to_vec(),
            costs: q.costs.to_vec(),
            reference_costs: q.reference_costs.to_vec(),
            evaluation_subject_ids: q
                .evaluation_subject_ids
                .iter()
                .map(ToString::to_string)
                .collect(),
            disjoint_training_subjects: q.disjoint_training_subjects,
            crossfit_fold_ownership_valid: q.crossfit_fold_ownership_valid,
            global_constraints_present: q.global_constraints_present,
            multi_action: q.multi_action.as_ref().map(Into::into),
            uplift_bins: q.uplift_bins.to_vec(),
            uplift_bin_count: q.uplift_bin_count,
            uplift_training_subject_ids: q
                .uplift_training_subject_ids
                .iter()
                .map(ToString::to_string)
                .collect(),
            regret: q.regret.as_ref().map(|regret| FixedCandidateRegretWire {
                candidates: regret.candidates.iter().map(|candidate| candidate.to_vec()).collect(),
                selected_index: regret.selected_index,
                training_subject_ids: regret
                    .training_subject_ids
                    .iter()
                    .map(ToString::to_string)
                    .collect(),
            }),
        }),
        CausalQuery::ContinuousDoseResponse(q) => {
            CausalQueryWire::ContinuousDoseResponse(ContinuousDoseResponseQueryWire {
                outcome: q.outcome.raw(),
                dose: q.dose.raw(),
                dose_density: q.dose_density.raw(),
                baseline_groups: q.baseline_groups.iter().map(ToString::to_string).collect(),
                target_doses: q.target_doses.to_vec(),
                bandwidth: q.bandwidth,
                min_local_support: q.min_local_support,
                density_provenance: q.density_provenance.to_string(),
                fixed_policy: q.fixed_policy.as_ref().map(|rule| {
                    (
                        rule.policy_doses
                            .iter()
                            .map(|(group, dose)| (group.to_string(), *dose))
                            .collect(),
                        rule.reference_doses
                            .iter()
                            .map(|(group, dose)| (group.to_string(), *dose))
                            .collect(),
                    )
                }),
            })
        }
        CausalQuery::PanelDid(q) => CausalQueryWire::PanelDid(PanelDidQueryWire {
            outcome: q.outcome.raw(),
            augmented: q.augmented.as_ref().map(|n| {
                (
                    n.outcome_pre.raw(),
                    n.propensity.raw(),
                    n.untreated_change_prediction.raw(),
                    n.predictions_cross_fitted,
                )
            }),
            repeated_cross_section: q.design
                == antecedent_core::DidSamplingDesign::RepeatedCrossSection,
            staggered_target: q.target,
            staggered_event_study: q.design
                == antecedent_core::DidSamplingDesign::StaggeredEventStudy,
            periods: q.periods.to_vec(),
            cohorts: q.cohorts.to_vec(),
            treated: q.treated.to_vec(),
            post: q.post.to_vec(),
            subjects: q.subjects.iter().map(ToString::to_string).collect(),
            clusters: q.clusters.iter().map(ToString::to_string).collect(),
            rollout: q
                .robust
                .as_ref()
                .map(|robust| crate::rollout_query_wire::rollout_to_wire(q.design, robust)),
            resampling: q.resampling.as_ref().map(crate::rollout_query_wire::resampling_to_wire),
        }),
        CausalQuery::SyntheticControl(q) => {
            CausalQueryWire::SyntheticControl(SyntheticControlQueryWire {
                outcome: q.outcome.raw(),
                units: q.units.iter().map(ToString::to_string).collect(),
                periods: q.periods.to_vec(),
                treated_unit: q.treated_unit.to_string(),
                intervention_period: q.intervention_period,
                difference_in_differences: q.method
                    == antecedent_core::SyntheticPanelMethod::DifferenceInDifferences,
                uniform_unit_randomization: q.uniform_unit_randomization,
                sharp_null_effect: q.sharp_null_effect,
                augmentation_ridge: q.augmentation_ridge,
                rollout: q
                    .rollout
                    .as_ref()
                    .map(crate::rollout_query_wire::synthetic_rollout_to_wire),
            })
        }
        CausalQuery::LocalPolynomialRatio(q) => {
            CausalQueryWire::LocalPolynomialRatio(LocalPolynomialRatioQueryWire {
                outcome: q.outcome.raw(),
                treatment: q.treatment.raw(),
                running: q.running.raw(),
                cutoff: q.cutoff,
                bandwidth: q.bandwidth,
                kink: q.kink,
                options: q
                    .options
                    .as_ref()
                    .map(crate::rollout_query_wire::local_ratio_options_to_wire),
            })
        }
        CausalQuery::Survival(q) => CausalQueryWire::Survival(SurvivalQueryWire {
            duration: q.duration.raw(),
            event: q.event.raw(),
            treatment: q.treatment.raw(),
            tau: q.tau,
            delayed_entry: q.delayed_entry.map(antecedent_core::VariableId::raw),
            target_cause: match q.functional {
                antecedent_core::SurvivalFunctional::SurvivalAndRmst => None,
                antecedent_core::SurvivalFunctional::CumulativeIncidence { target_cause } => {
                    Some(target_cause)
                }
            },
            independent_observation: matches!(
                &q.observation_assumption,
                antecedent_core::ObservationAssumption::IndependentGiven(_)
            ),
            independent_given: match &q.observation_assumption {
                antecedent_core::ObservationAssumption::IndependentGiven(vars) => {
                    vars.iter().map(|id| id.raw()).collect()
                }
                _ => Vec::new(),
            },
            censoring_times: q
                .known_censoring
                .as_ref()
                .map_or_else(Vec::new, |known| known.times.to_vec()),
            censoring_columns: q
                .known_censoring
                .as_ref()
                .map_or_else(Vec::new, |known| known.columns.iter().map(|id| id.raw()).collect()),
            censoring_probability_floor: q
                .known_censoring
                .as_ref()
                .map(|known| known.minimum_probability),
        }),
        CausalQuery::LongitudinalRegime(q) => {
            CausalQueryWire::LongitudinalRegime(LongitudinalRegimeQueryWire {
                outcome: q.outcome.raw(),
                method: match q.method {
                    antecedent_core::LongitudinalRegimeMethod::Ipw => "ipw",
                    antecedent_core::LongitudinalRegimeMethod::GFormula => "g_formula",
                    antecedent_core::LongitudinalRegimeMethod::SequentialDoublyRobust => {
                        "sequential_dr"
                    }
                    antecedent_core::LongitudinalRegimeMethod::MarginalStructuralModel => {
                        "marginal_structural_model"
                    }
                }
                .into(),
                period_outcome_predictions: q
                    .period_outcome_predictions
                    .as_ref()
                    .map_or_else(Vec::new, |q| q.to_vec()),
                known_fixed_outcome_predictions: q.known_fixed_outcome_predictions,
                stabilizing_numerator_probabilities: q
                    .stabilizing_numerator_probabilities
                    .as_ref()
                    .map_or_else(Vec::new, |p| p.to_vec()),
                q_predictions: q.q_predictions.as_ref().map_or_else(Vec::new, |q| q.to_vec()),
                observation_history: q
                    .observation_history
                    .as_ref()
                    .map_or_else(Vec::new, |o| o.to_vec()),
                prediction_fold_ids: q
                    .prediction_fold_ids
                    .as_ref()
                    .map_or_else(Vec::new, |f| f.to_vec()),
                periods: q.periods,
                treatment_history: q.treatment_history.to_vec(),
                regime_actions: q.regime_actions.to_vec(),
                treatment_probabilities: q.treatment_probabilities.to_vec(),
                censoring_probabilities: q.censoring_probabilities.to_vec(),
                outcome_observed: q.outcome_observed.to_vec(),
                subject_ids: q.subject_ids.iter().map(ToString::to_string).collect(),
                fold_ids: q.fold_ids.to_vec(),
                excluded_fold_predictions: q.excluded_fold_predictions,
                probabilities_known_by_design: q.probabilities_known_by_design,
                minimum_probability: q.minimum_probability,
                rule_id: q.rule_id.as_ref().map(ToString::to_string),
                rule_version: q.rule_version.as_ref().map(ToString::to_string),
                rule_provenance: q.rule_provenance.as_ref().map(ToString::to_string),
            })
        }
        CausalQuery::SequentialContrast(q) => {
            CausalQueryWire::SequentialContrast(SequentialContrastQueryWire {
                outcome: q.outcome.raw(),
                first_treatment: q.first_treatment.raw(),
                second_treatment: q.second_treatment.raw(),
                baseline: q.baseline.iter().map(|id| id.raw()).collect(),
                intermediate: q.intermediate.iter().map(|id| id.raw()).collect(),
                active: q.active,
                control: q.control,
                fold_of_row: q.fold_of_row.to_vec(),
                fold_count: q.fold_count,
                min_probability: q.min_probability,
            })
        }
        CausalQuery::HistoryMatch(q) => CausalQueryWire::HistoryMatch(HistoryMatchQueryWire {
            outcome: q.outcome.raw(),
            treated: q.treated.raw(),
            history: q.history.iter().map(|id| id.raw()).collect(),
            period: q.period.map(antecedent_core::VariableId::raw),
            weight: q.weight.map(antecedent_core::VariableId::raw),
            caliper: q.caliper,
        }),
        CausalQuery::Discover(q) => CausalQueryWire::Discover(
            crate::structure_query_wire::DiscoverQueryWire::from_domain(q),
        ),
        CausalQuery::Attribute(q) => CausalQueryWire::Attribute(
            crate::structure_query_wire::AttributeQueryWire::from_domain(q)?,
        ),
        CausalQuery::Simulate(q) => CausalQueryWire::Simulate(
            crate::structure_query_wire::SimulateQueryWire::from_domain(q)?,
        ),
        CausalQuery::TemporalDesign(q) => CausalQueryWire::TemporalDesign(
            crate::temporal_design_query_wire::TemporalDesignQueryWire::from_domain(q)?,
        ),
        _ => return Err(IoError::Convert("unsupported CausalQuery variant".into())),
    })
}

/// Decode [`CausalQueryWire`].
///
/// # Errors
///
/// Unknown tags or size overflows.
#[allow(
    clippy::too_many_lines,
    reason = "one arm per query kind, each converting that kind fields"
)]
pub fn causal_query_from_wire(w: &CausalQueryWire) -> Result<CausalQuery, IoError> {
    Ok(match w {
        CausalQueryWire::AverageEffect {
            treatment,
            outcome,
            effect_modifiers,
            control,
            active,
            target_population,
            outcome_functional,
        } => CausalQuery::AverageEffect(
            AverageEffectQuery::new(
                VariableId::from_raw(*treatment),
                VariableId::from_raw(*outcome),
                vars_from_raw(effect_modifiers),
                control.to_domain(),
                active.to_domain(),
                target_population.to_domain()?,
            )
            .with_outcome_functional(outcome_functional.to_domain()),
        ),
        CausalQueryWire::TemporalEffect {
            treatment,
            outcome,
            policy,
            control,
            active,
            horizon_steps,
            max_history_lag,
            target_population,
            effect_modifiers,
        } => CausalQuery::TemporalEffect(TemporalEffectQuery {
            treatment: VariableId::from_raw(*treatment),
            outcome: VariableId::from_raw(*outcome),
            policy: policy.to_domain(),
            control: control.to_domain(),
            active: active.to_domain(),
            horizon_steps: *horizon_steps,
            max_history_lag: *max_history_lag,
            target_population: target_population.to_domain()?,
            effect_modifiers: effect_modifiers.iter().copied().map(VariableId::from_raw).collect(),
        }),
        CausalQueryWire::Counterfactual { outcomes, interventions, control, allow_nested } => {
            let interventions: Arc<[Intervention]> =
                interventions.iter().map(InterventionWire::to_domain).collect::<Vec<_>>().into();
            let control = match control {
                Some(c) => c.to_domain(),
                None => CounterfactualQuery::default_control(&interventions),
            };
            CausalQuery::Counterfactual(CounterfactualQuery {
                outcomes: vars_from_raw(outcomes),
                interventions,
                control,
                allow_nested: *allow_nested,
            })
        }
        CausalQueryWire::NestedCounterfactual {
            treatment,
            mediator,
            outcome,
            control_bits,
            active_bits,
        } => {
            let query = antecedent_core::NestedCounterfactualQuery::with_levels(
                antecedent_core::VariableId::from_raw(*treatment),
                antecedent_core::VariableId::from_raw(*mediator),
                antecedent_core::VariableId::from_raw(*outcome),
                f64::from_bits(*control_bits),
                f64::from_bits(*active_bits),
            )
            .map_err(|e| IoError::Convert(e.to_string()))?;
            CausalQuery::NestedCounterfactual(query)
        }
        CausalQueryWire::AnomalyAttribution { targets, unit_rows, max_units, reference } => {
            CausalQuery::AnomalyAttribution(AnomalyAttributionQuery {
                targets: vars_from_raw(targets),
                unit_rows: unit_rows
                    .as_ref()
                    .map(|rows| {
                        rows.iter()
                            .map(|&r| usize::try_from(r).map_err(|_| IoError::TooLarge))
                            .collect::<Result<Vec<_>, _>>()
                    })
                    .transpose()?
                    .map(Arc::from),
                max_units: usize::try_from(*max_units).map_err(|_| IoError::TooLarge)?,
                reference: match reference.flatten() {
                    None => AnomalyReference::Empirical,
                    Some((center, scale)) => AnomalyReference::fixed(center, scale),
                },
            })
        }
        CausalQueryWire::ChangeAttribution {
            outcome,
            baseline,
            comparison,
            components,
            allocation,
            max_components,
        } => CausalQuery::ChangeAttribution(ChangeAttributionQuery {
            outcome: VariableId::from_raw(*outcome),
            baseline: population_from_wire(baseline)?,
            comparison: population_from_wire(comparison)?,
            components: (*components).into(),
            allocation: allocation_from_wire(allocation)?,
            max_components: usize::try_from(*max_components).map_err(|_| IoError::TooLarge)?,
        }),
        CausalQueryWire::MechanismChange {
            targets,
            baseline,
            comparison,
            significance_level_bits,
            max_targets,
        } => CausalQuery::MechanismChange(MechanismChangeQuery {
            targets: vars_from_raw(targets),
            baseline: population_from_wire(baseline)?,
            comparison: population_from_wire(comparison)?,
            significance_level: OrderedFloatBits::from_f64(f64::from_bits(
                *significance_level_bits,
            )),
            max_targets: usize::try_from(*max_targets).map_err(|_| IoError::TooLarge)?,
        }),
        CausalQueryWire::UnitChange { outcome, unit_rows, components, allocation, max_units } => {
            CausalQuery::UnitChange(UnitChangeQuery {
                outcome: VariableId::from_raw(*outcome),
                unit_rows: unit_rows
                    .as_ref()
                    .map(|rows| {
                        rows.iter()
                            .map(|&r| usize::try_from(r).map_err(|_| IoError::TooLarge))
                            .collect::<Result<Vec<_>, _>>()
                    })
                    .transpose()?
                    .map(Arc::from),
                components: (*components).into(),
                allocation: allocation_from_wire(allocation)?,
                max_units: usize::try_from(*max_units).map_err(|_| IoError::TooLarge)?,
            })
        }
        CausalQueryWire::Mediation {
            treatment,
            outcome,
            mediators,
            contrast,
            control,
            active,
            target_population,
            horizons,
        } => {
            let query = MediationQuery {
                treatment: VariableId::from_raw(*treatment),
                outcome: VariableId::from_raw(*outcome),
                mediators: vars_from_raw(mediators),
                contrast: (*contrast).into(),
                control: control.to_domain(),
                active: active.to_domain(),
                target_population: target_population.to_domain()?,
                horizons: Arc::from(horizons.as_slice()),
            };
            query.validate().map_err(|e| IoError::Convert(e.to_string()))?;
            CausalQuery::Mediation(query)
        }
        CausalQueryWire::ConditionalEffect { inner } => {
            let CausalQuery::AverageEffect(inner_q) = causal_query_from_wire(inner)? else {
                return Err(IoError::Convert(
                    "ConditionalEffect.inner must be AverageEffect".into(),
                ));
            };
            CausalQuery::ConditionalEffect(ConditionalEffectQuery { inner: inner_q })
        }
        CausalQueryWire::Distribution(w) => {
            CausalQuery::Distribution(interventional_distribution_from_wire(w)?)
        }
        CausalQueryWire::PathSpecific(w) => CausalQuery::PathSpecific(path_specific_from_wire(w)?),
        CausalQueryWire::Response(w) => CausalQuery::Response(response_query_from_wire(w)?),
        CausalQueryWire::Transport(w) => CausalQuery::Transport(transport_query_from_wire(w)?),
        CausalQueryWire::Interference(w) => {
            CausalQuery::Interference(interference_query_from_wire(w)?)
        }
        CausalQueryWire::RandomizedEffect(w) => {
            if !matches!(w.design, RandomizationDesignWire::MultiArm)
                && (!w.multi_arm_labels.is_empty()
                    || !w.multi_arm_assignment.is_empty()
                    || !w.multi_arm_probabilities.is_empty())
            {
                return Err(IoError::Convert(
                    "multi-arm metadata requires a multi-arm design".into(),
                ));
            }
            if !matches!(w.design, RandomizationDesignWire::Factorial2x2)
                && (!w.second_factor_assignment.is_empty()
                    || w.factorial_cell_counts.is_some()
                    || w.second_factor_arms.is_some())
            {
                return Err(IoError::Convert(
                    "factorial metadata requires a factorial design".into(),
                ));
            }
            let design = match &w.design {
                RandomizationDesignWire::Bernoulli => RandomizationDesign::Bernoulli,
                RandomizationDesignWire::Complete { treated_units } => {
                    RandomizationDesign::Complete { treated_units: *treated_units }
                }
                RandomizationDesignWire::Cluster { treated_clusters } => {
                    RandomizationDesign::Cluster { treated_clusters: *treated_clusters }
                }
                RandomizationDesignWire::Stratified => RandomizationDesign::Stratified {
                    blocks: w
                        .blocks
                        .iter()
                        .map(|x| Arc::<str>::from(x.as_str()))
                        .collect::<Vec<_>>()
                        .into(),
                    treated_per_row: w.treated_per_row.clone().into(),
                },
                RandomizationDesignWire::Factorial2x2 => RandomizationDesign::Factorial2x2 {
                    second_factor_assignment: w.second_factor_assignment.clone().into(),
                    cell_counts: w.factorial_cell_counts.ok_or_else(|| {
                        IoError::Convert("factorial cell counts are missing".into())
                    })?,
                    second_factor_arms: {
                        let arms = w.second_factor_arms.as_ref().ok_or_else(|| {
                            IoError::Convert("second-factor labels are missing".into())
                        })?;
                        (Arc::<str>::from(arms.0.as_str()), Arc::<str>::from(arms.1.as_str()))
                    },
                },
                RandomizationDesignWire::MultiArm => RandomizationDesign::MultiArm {
                    assignment: w.multi_arm_assignment.clone().into(),
                    probabilities: w.multi_arm_probabilities.clone().into(),
                    arms: w
                        .multi_arm_labels
                        .iter()
                        .map(|arm| Arc::<str>::from(arm.as_str()))
                        .collect::<Vec<_>>()
                        .into(),
                },
                RandomizationDesignWire::Switchback => RandomizationDesign::Switchback {
                    periods: w
                        .periods
                        .iter()
                        .map(|period| Arc::<str>::from(period.as_str()))
                        .collect::<Vec<_>>()
                        .into(),
                },
            };
            let mut query = RandomizedEffectQuery::with_design(
                design,
                VariableId::from_raw(w.outcome),
                Arc::<[bool]>::from(w.realized_assignment.clone()),
                Arc::<[f64]>::from(w.assignment_probabilities.clone()),
                w.assignment_units.iter().map(|x| Arc::<str>::from(x.as_str())).collect::<Vec<_>>(),
                w.outcome_units.iter().map(|x| Arc::<str>::from(x.as_str())).collect::<Vec<_>>(),
                (
                    Arc::<str>::from(w.treatment_arms.0.as_str()),
                    Arc::<str>::from(w.treatment_arms.1.as_str()),
                ),
            );
            if let Some((id, coefficient)) = w.fixed_cuped {
                query = query.with_fixed_cuped(VariableId::from_raw(id), coefficient);
            }
            if !w.ancova_covariates.is_empty() {
                query = query.with_ancova(
                    w.ancova_covariates
                        .iter()
                        .copied()
                        .map(VariableId::from_raw)
                        .collect::<Vec<_>>(),
                );
            }
            if let Some(receipt) = &w.received_treatment {
                query = if w.estimand == RandomizedEstimandWire::TreatmentOnTreated {
                    query.with_treatment_on_treated(receipt.clone())
                } else {
                    query.with_received_treatment(receipt.clone())
                };
            }
            if w.exact_randomization_test {
                query = query.with_exact_randomization_test();
            }
            if (w.estimand != RandomizedEstimandWire::Itt) != w.received_treatment.is_some() {
                return Err(IoError::Convert(
                    "receipt-adjusted estimand and treatment receipt must agree".into(),
                ));
            }
            query.validate().map_err(|e| IoError::Convert(e.to_string()))?;
            CausalQuery::RandomizedEffect(query)
        }
        CausalQueryWire::PolicyValue(w) => {
            let q = PolicyValueQuery {
                outcome: VariableId::from_raw(w.outcome),
                assignment: w.assignment.clone().into(),
                propensity: w.propensity.clone().into(),
                actions: w.actions.clone().into(),
                reference: w.reference.clone().into(),
                mu0: w.mu0.clone().into(),
                mu1: w.mu1.clone().into(),
                costs: w.costs.clone().into(),
                reference_costs: w.reference_costs.clone().into(),
                evaluation_subject_ids: w
                    .evaluation_subject_ids
                    .iter()
                    .map(|x| Arc::<str>::from(x.as_str()))
                    .collect::<Vec<_>>()
                    .into(),
                disjoint_training_subjects: w.disjoint_training_subjects,
                crossfit_fold_ownership_valid: w.crossfit_fold_ownership_valid,
                global_constraints_present: w.global_constraints_present,
                multi_action: w.multi_action.as_ref().map(Into::into),
                uplift_bins: w.uplift_bins.clone().into(),
                uplift_bin_count: w.uplift_bin_count,
                uplift_training_subject_ids: w
                    .uplift_training_subject_ids
                    .iter()
                    .map(|x| Arc::<str>::from(x.as_str()))
                    .collect::<Vec<_>>()
                    .into(),
                regret: w.regret.as_ref().map(|regret| {
                    antecedent_core::FixedCandidateRegretInputs {
                        candidates: regret
                            .candidates
                            .iter()
                            .map(|actions| Arc::<[bool]>::from(actions.clone()))
                            .collect::<Vec<_>>()
                            .into(),
                        selected_index: regret.selected_index,
                        training_subject_ids: regret
                            .training_subject_ids
                            .iter()
                            .map(|id| Arc::<str>::from(id.as_str()))
                            .collect::<Vec<_>>()
                            .into(),
                    }
                }),
            };
            q.validate().map_err(|e| IoError::Convert(e.to_string()))?;
            CausalQuery::PolicyValue(q)
        }
        CausalQueryWire::ContinuousDoseResponse(w) => {
            let q = antecedent_core::ContinuousDoseResponseQuery {
                outcome: VariableId::from_raw(w.outcome),
                dose: VariableId::from_raw(w.dose),
                dose_density: VariableId::from_raw(w.dose_density),
                baseline_groups: w
                    .baseline_groups
                    .iter()
                    .map(|value| Arc::<str>::from(value.as_str()))
                    .collect::<Vec<_>>()
                    .into(),
                target_doses: w.target_doses.clone().into(),
                bandwidth: w.bandwidth,
                min_local_support: w.min_local_support,
                density_provenance: Arc::from(w.density_provenance.as_str()),
                fixed_policy: w.fixed_policy.as_ref().map(|(policy, reference)| {
                    antecedent_core::FixedGroupDosePolicy {
                        policy_doses: policy
                            .iter()
                            .map(|(group, dose)| (Arc::from(group.as_str()), *dose))
                            .collect::<Vec<_>>()
                            .into(),
                        reference_doses: reference
                            .iter()
                            .map(|(group, dose)| (Arc::from(group.as_str()), *dose))
                            .collect::<Vec<_>>()
                            .into(),
                    }
                }),
            };
            q.validate().map_err(|error| IoError::Convert(error.to_string()))?;
            CausalQuery::ContinuousDoseResponse(q)
        }
        CausalQueryWire::PanelDid(w) if w.rollout.is_some() => {
            let rollout = w.rollout.as_ref().ok_or_else(|| IoError::Convert("rollout".into()))?;
            if w.augmented.is_some()
                || w.staggered_event_study
                || w.staggered_target.is_some()
                || w.repeated_cross_section
            {
                return Err(IoError::Convert(
                    "a heterogeneity-robust rollout cannot also select another sampling design"
                        .into(),
                ));
            }
            let (design, robust) = crate::rollout_query_wire::rollout_from_wire(rollout)?;
            let subjects =
                w.subjects.iter().map(|x| Arc::<str>::from(x.as_str())).collect::<Vec<_>>();
            let clusters =
                w.clusters.iter().map(|x| Arc::<str>::from(x.as_str())).collect::<Vec<_>>();
            let mut q = if design == antecedent_core::DidSamplingDesign::CovariateAdjustedTwoPeriod
            {
                PanelDidQuery::covariate_adjusted_two_period(
                    VariableId::from_raw(w.outcome),
                    w.treated.clone(),
                    w.post.clone(),
                    subjects,
                    clusters,
                    Arc::clone(&robust.covariates),
                    robust.nuisance,
                )
            } else {
                PanelDidQuery::heterogeneity_robust(
                    VariableId::from_raw(w.outcome),
                    subjects,
                    clusters,
                    w.periods.clone(),
                    w.cohorts.clone(),
                    robust.clone(),
                )
            };
            q.robust = Some(robust);
            q.resampling = w
                .resampling
                .as_ref()
                .map(crate::rollout_query_wire::resampling_from_wire)
                .transpose()?;
            if q.treated.as_ref() != w.treated.as_slice() || q.post.as_ref() != w.post.as_slice() {
                return Err(IoError::Convert(
                    "rollout treatment and post indicators disagree with cohort and period metadata".into(),
                ));
            }
            q.validate().map_err(|e| IoError::Convert(e.to_string()))?;
            CausalQuery::PanelDid(q)
        }
        CausalQueryWire::PanelDid(w) => {
            let q = if let Some((pre, propensity, prediction, cross_fitted)) = w.augmented {
                if w.staggered_event_study
                    || w.staggered_target.is_some()
                    || w.repeated_cross_section
                    || !w.periods.is_empty()
                    || !w.cohorts.is_empty()
                {
                    return Err(IoError::Convert(
                        "augmented panel DiD cannot also select another sampling design".into(),
                    ));
                }
                PanelDidQuery::augmented_panel(
                    VariableId::from_raw(w.outcome),
                    VariableId::from_raw(pre),
                    VariableId::from_raw(propensity),
                    VariableId::from_raw(prediction),
                    w.treated.clone(),
                    w.subjects.iter().map(|x| Arc::<str>::from(x.as_str())).collect::<Vec<_>>(),
                    w.clusters.iter().map(|x| Arc::<str>::from(x.as_str())).collect::<Vec<_>>(),
                    cross_fitted,
                )
            } else if w.staggered_event_study {
                if w.repeated_cross_section || w.staggered_target.is_some() {
                    return Err(IoError::Convert("event study cannot also select a group-time target or repeated cross section".into()));
                }
                PanelDidQuery::staggered_event_study(
                    VariableId::from_raw(w.outcome),
                    w.subjects.iter().map(|x| Arc::<str>::from(x.as_str())).collect::<Vec<_>>(),
                    w.clusters.iter().map(|x| Arc::<str>::from(x.as_str())).collect::<Vec<_>>(),
                    w.periods.clone(),
                    w.cohorts.clone(),
                )
            } else if let Some((cohort, period)) = w.staggered_target {
                if w.repeated_cross_section {
                    return Err(IoError::Convert(
                        "staggered DiD cannot also be a repeated cross section".into(),
                    ));
                }
                PanelDidQuery::staggered_group_time(
                    VariableId::from_raw(w.outcome),
                    w.subjects.iter().map(|x| Arc::<str>::from(x.as_str())).collect::<Vec<_>>(),
                    w.clusters.iter().map(|x| Arc::<str>::from(x.as_str())).collect::<Vec<_>>(),
                    w.periods.clone(),
                    w.cohorts.clone(),
                    cohort,
                    period,
                )
            } else {
                let constructor = if w.repeated_cross_section {
                    PanelDidQuery::repeated_cross_section
                } else {
                    PanelDidQuery::new
                };
                constructor(
                    VariableId::from_raw(w.outcome),
                    w.treated.clone(),
                    w.post.clone(),
                    w.subjects.iter().map(|x| Arc::<str>::from(x.as_str())).collect::<Vec<_>>(),
                    w.clusters.iter().map(|x| Arc::<str>::from(x.as_str())).collect::<Vec<_>>(),
                )
            };
            if (w.staggered_target.is_some() || w.staggered_event_study || w.augmented.is_some())
                && (q.treated.as_ref() != w.treated.as_slice()
                    || q.post.as_ref() != w.post.as_slice())
            {
                return Err(IoError::Convert(
                    "staggered DiD treatment and post indicators disagree with cohort and period metadata".into(),
                ));
            }
            let mut q = q;
            q.resampling = w
                .resampling
                .as_ref()
                .map(crate::rollout_query_wire::resampling_from_wire)
                .transpose()?;
            q.validate().map_err(|e| IoError::Convert(e.to_string()))?;
            CausalQuery::PanelDid(q)
        }
        CausalQueryWire::SyntheticControl(w) => {
            let q = SyntheticControlQuery::new(
                VariableId::from_raw(w.outcome),
                w.units.iter().map(|unit| Arc::<str>::from(unit.as_str())).collect::<Vec<_>>(),
                w.periods.clone(),
                Arc::<str>::from(w.treated_unit.as_str()),
                w.intervention_period,
            );
            let q = if w.difference_in_differences { q.difference_in_differences() } else { q };
            let q =
                if w.uniform_unit_randomization { q.with_uniform_unit_randomization() } else { q };
            let q = if let Some(effect) = w.sharp_null_effect {
                q.with_sharp_null_effect(effect)
            } else {
                q
            };
            let mut q =
                if let Some(ridge) = w.augmentation_ridge { q.with_augmentation(ridge) } else { q };
            q.rollout = w
                .rollout
                .as_ref()
                .map(crate::rollout_query_wire::synthetic_rollout_from_wire)
                .transpose()?;
            q.validate().map_err(|e| IoError::Convert(e.to_string()))?;
            CausalQuery::SyntheticControl(q)
        }
        CausalQueryWire::LocalPolynomialRatio(w) => {
            let q = LocalPolynomialRatioQuery {
                outcome: VariableId::from_raw(w.outcome),
                treatment: VariableId::from_raw(w.treatment),
                running: VariableId::from_raw(w.running),
                cutoff: w.cutoff,
                bandwidth: w.bandwidth,
                kink: w.kink,
                options: w
                    .options
                    .as_ref()
                    .map(crate::rollout_query_wire::local_ratio_options_from_wire)
                    .transpose()?,
            };
            q.validate().map_err(|e| IoError::Convert(e.to_string()))?;
            CausalQuery::LocalPolynomialRatio(q)
        }
        CausalQueryWire::Survival(w) => {
            if w.censoring_probability_floor.is_none()
                && (!w.censoring_times.is_empty() || !w.censoring_columns.is_empty())
            {
                return Err(IoError::Convert(
                    "censoring grid requires its positivity floor".into(),
                ));
            }
            let q = antecedent_core::SurvivalQuery {
                duration: VariableId::from_raw(w.duration),
                event: VariableId::from_raw(w.event),
                treatment: VariableId::from_raw(w.treatment),
                tau: w.tau,
                delayed_entry: w.delayed_entry.map(VariableId::from_raw),
                known_censoring: w.censoring_probability_floor.map(|minimum_probability| {
                    antecedent_core::KnownCensoringSurvival {
                        times: Arc::from(w.censoring_times.clone()),
                        columns: Arc::from(
                            w.censoring_columns
                                .iter()
                                .copied()
                                .map(VariableId::from_raw)
                                .collect::<Vec<_>>(),
                        ),
                        minimum_probability,
                    }
                }),
                observation_assumption: if w.independent_observation {
                    antecedent_core::ObservationAssumption::IndependentGiven(Arc::from(
                        w.independent_given
                            .iter()
                            .copied()
                            .map(VariableId::from_raw)
                            .collect::<Vec<_>>(),
                    ))
                } else {
                    return Err(IoError::Convert(
                        "survival requires marginal independent observation declaration".into(),
                    ));
                },
                functional: w.target_cause.map_or(
                    antecedent_core::SurvivalFunctional::SurvivalAndRmst,
                    |target_cause| antecedent_core::SurvivalFunctional::CumulativeIncidence {
                        target_cause,
                    },
                ),
            };
            q.validate().map_err(|e| IoError::Convert(e.to_string()))?;
            CausalQuery::Survival(q)
        }
        CausalQueryWire::LongitudinalRegime(w) => {
            let method = match w.method.as_str() {
                "ipw" => antecedent_core::LongitudinalRegimeMethod::Ipw,
                "g_formula" => antecedent_core::LongitudinalRegimeMethod::GFormula,
                "sequential_dr" => {
                    antecedent_core::LongitudinalRegimeMethod::SequentialDoublyRobust
                }
                "marginal_structural_model" => {
                    antecedent_core::LongitudinalRegimeMethod::MarginalStructuralModel
                }
                _ => return Err(IoError::Convert("unknown longitudinal regime method".into())),
            };
            let q = antecedent_core::LongitudinalRegimeQuery {
                outcome: VariableId::from_raw(w.outcome),
                periods: w.periods,
                method,
                period_outcome_predictions: if w.period_outcome_predictions.is_empty() {
                    None
                } else {
                    Some(w.period_outcome_predictions.clone().into())
                },
                known_fixed_outcome_predictions: w.known_fixed_outcome_predictions,
                stabilizing_numerator_probabilities: if w
                    .stabilizing_numerator_probabilities
                    .is_empty()
                {
                    None
                } else {
                    Some(w.stabilizing_numerator_probabilities.clone().into())
                },
                q_predictions: if w.q_predictions.is_empty() {
                    None
                } else {
                    Some(w.q_predictions.clone().into())
                },
                observation_history: if w.observation_history.is_empty() {
                    None
                } else {
                    Some(w.observation_history.clone().into())
                },
                prediction_fold_ids: if w.prediction_fold_ids.is_empty() {
                    None
                } else {
                    Some(w.prediction_fold_ids.clone().into())
                },
                treatment_history: w.treatment_history.clone().into(),
                regime_actions: w.regime_actions.clone().into(),
                treatment_probabilities: w.treatment_probabilities.clone().into(),
                censoring_probabilities: w.censoring_probabilities.clone().into(),
                outcome_observed: w.outcome_observed.clone().into(),
                subject_ids: w
                    .subject_ids
                    .iter()
                    .map(|s| Arc::<str>::from(s.as_str()))
                    .collect::<Vec<_>>()
                    .into(),
                fold_ids: w.fold_ids.clone().into(),
                excluded_fold_predictions: w.excluded_fold_predictions,
                probabilities_known_by_design: w.probabilities_known_by_design,
                minimum_probability: w.minimum_probability,
                rule_id: w.rule_id.as_deref().map(Arc::<str>::from),
                rule_version: w.rule_version.as_deref().map(Arc::<str>::from),
                rule_provenance: w.rule_provenance.as_deref().map(Arc::<str>::from),
            };
            q.validate().map_err(|e| IoError::Convert(e.to_string()))?;
            CausalQuery::LongitudinalRegime(q)
        }
        CausalQueryWire::SequentialContrast(w) => {
            let q = antecedent_core::SequentialContrastQuery {
                outcome: VariableId::from_raw(w.outcome),
                first_treatment: VariableId::from_raw(w.first_treatment),
                second_treatment: VariableId::from_raw(w.second_treatment),
                baseline: w.baseline.iter().map(|id| VariableId::from_raw(*id)).collect(),
                intermediate: w.intermediate.iter().map(|id| VariableId::from_raw(*id)).collect(),
                active: w.active,
                control: w.control,
                fold_of_row: w.fold_of_row.clone().into(),
                fold_count: w.fold_count,
                min_probability: w.min_probability,
            };
            q.validate().map_err(|e| IoError::Convert(e.to_string()))?;
            CausalQuery::SequentialContrast(q)
        }
        CausalQueryWire::HistoryMatch(w) => {
            let q = antecedent_core::HistoryMatchQuery {
                outcome: VariableId::from_raw(w.outcome),
                treated: VariableId::from_raw(w.treated),
                history: w.history.iter().map(|id| VariableId::from_raw(*id)).collect(),
                period: w.period.map(VariableId::from_raw),
                weight: w.weight.map(VariableId::from_raw),
                caliper: w.caliper,
            };
            q.validate().map_err(|e| IoError::Convert(e.to_string()))?;
            CausalQuery::HistoryMatch(q)
        }
        CausalQueryWire::Discover(w) => CausalQuery::Discover(w.to_domain()?),
        CausalQueryWire::Attribute(w) => CausalQuery::Attribute(w.to_domain()?),
        CausalQueryWire::Simulate(w) => CausalQuery::Simulate(w.to_domain()?),
        CausalQueryWire::TemporalDesign(w) => CausalQuery::TemporalDesign(w.to_domain()?),
    })
}

/// Encode a structural transport query.
///
/// # Errors
///
/// Unsupported response-query fields.
pub fn transport_query_to_wire(q: &TransportQuery) -> Result<TransportQueryWire, IoError> {
    Ok(TransportQueryWire {
        catalog: q
            .catalog
            .as_ref()
            .map(crate::transport_catalog_wire::EvidenceCatalogWire::from_catalog),
        response: response_query_to_wire(&q.response)?,
        source_population: q.source_population.to_string(),
        target_population: q.target_population.to_string(),
        source_experiments: vars_to_raw(&q.source_experiments),
    })
}

/// Decode and validate a structural transport query.
///
/// # Errors
///
/// Invalid response or transport semantics.
pub fn transport_query_from_wire(w: &TransportQueryWire) -> Result<TransportQuery, IoError> {
    let mut query = TransportQuery::new(
        response_query_from_wire(&w.response)?,
        w.source_population.as_str(),
        w.target_population.as_str(),
        vars_from_raw(&w.source_experiments),
    );
    if let Some(catalog) = &w.catalog {
        query = query
            .with_catalog(catalog.to_catalog()?)
            .map_err(|e| IoError::Convert(e.to_string()))?;
    }
    query.validate().map_err(|error| IoError::Convert(error.to_string()))?;
    Ok(query)
}

/// Encode a randomized interference query.
///
/// # Errors
///
/// Counts that do not fit the portable wire representation.
pub fn interference_query_to_wire(q: &InterferenceQuery) -> Result<InterferenceQueryWire, IoError> {
    let assignment = match &q.assignment {
        AssignmentDesign::Bernoulli { probabilities } => {
            AssignmentDesignWire::Bernoulli { probabilities: probabilities.to_vec() }
        }
        AssignmentDesign::CompleteRandomization { treated } => {
            AssignmentDesignWire::CompleteRandomization {
                treated: u64::try_from(*treated).map_err(|_| IoError::TooLarge)?,
            }
        }
        AssignmentDesign::ClusterRandomization { clusters, treated_clusters } => {
            AssignmentDesignWire::ClusterRandomization {
                clusters: clusters.to_vec(),
                treated_clusters: u64::try_from(*treated_clusters)
                    .map_err(|_| IoError::TooLarge)?,
            }
        }
        AssignmentDesign::TwoStageSaturation {
            clusters,
            low_probability,
            high_probability,
            high_clusters,
            realized_saturation,
        } => AssignmentDesignWire::TwoStageSaturation {
            clusters: clusters.to_vec(),
            low_probability: *low_probability,
            high_probability: *high_probability,
            high_clusters: u64::try_from(*high_clusters).map_err(|_| IoError::TooLarge)?,
            realized_saturation: realized_saturation.to_vec(),
        },
        AssignmentDesign::ObservedExposure {
            clusters,
            propensity_from,
            propensity_to,
            provenance,
            assume_network_exchangeability,
        } => AssignmentDesignWire::ObservedExposure {
            clusters: clusters.to_vec(),
            propensity_from: propensity_from.to_vec(),
            propensity_to: propensity_to.to_vec(),
            provenance: match provenance {
                antecedent_core::ExposurePropensityProvenance::Known => "known",
                antecedent_core::ExposurePropensityProvenance::ExternallyEstimated => {
                    "externally_estimated"
                }
            }
            .into(),
            assume_network_exchangeability: *assume_network_exchangeability,
        },
    };
    let exposure = match &q.exposure {
        ExposureMapping::OwnTreatment => ExposureMappingWire::OwnTreatment,
        ExposureMapping::NeighborCount => ExposureMappingWire::NeighborCount,
        ExposureMapping::NeighborFraction => ExposureMappingWire::NeighborFraction,
        ExposureMapping::WeightedNeighborExposure => ExposureMappingWire::WeightedNeighborExposure,
        ExposureMapping::Custom(id) => ExposureMappingWire::Custom(id.to_string()),
    };
    let InterferenceFunctional::ExposureContrast { outcome, from, to } = &q.functional;
    Ok(InterferenceQueryWire {
        assignment,
        exposure,
        functional: InterferenceFunctionalWire::ExposureContrast {
            outcome: outcome.raw(),
            from: ExposureLevelWire { own: from.own, neighbors: from.neighbors },
            to: ExposureLevelWire { own: to.own, neighbors: to.neighbors },
        },
        probability_draws: q.probability_draws,
    })
}

/// Decode and validate a randomized interference query.
///
/// # Errors
///
/// Counts that do not fit `usize` or invalid query semantics.
pub fn interference_query_from_wire(
    w: &InterferenceQueryWire,
) -> Result<InterferenceQuery, IoError> {
    let assignment = match &w.assignment {
        AssignmentDesignWire::Bernoulli { probabilities } => {
            AssignmentDesign::Bernoulli { probabilities: probabilities.clone().into() }
        }
        AssignmentDesignWire::CompleteRandomization { treated } => {
            AssignmentDesign::CompleteRandomization {
                treated: usize::try_from(*treated).map_err(|_| IoError::TooLarge)?,
            }
        }
        AssignmentDesignWire::ClusterRandomization { clusters, treated_clusters } => {
            AssignmentDesign::ClusterRandomization {
                clusters: clusters.clone().into(),
                treated_clusters: usize::try_from(*treated_clusters)
                    .map_err(|_| IoError::TooLarge)?,
            }
        }
        AssignmentDesignWire::TwoStageSaturation {
            clusters,
            low_probability,
            high_probability,
            high_clusters,
            realized_saturation,
        } => AssignmentDesign::TwoStageSaturation {
            clusters: clusters.clone().into(),
            low_probability: *low_probability,
            high_probability: *high_probability,
            high_clusters: usize::try_from(*high_clusters).map_err(|_| IoError::TooLarge)?,
            realized_saturation: realized_saturation.clone().into(),
        },
        AssignmentDesignWire::ObservedExposure {
            clusters,
            propensity_from,
            propensity_to,
            provenance,
            assume_network_exchangeability,
        } => AssignmentDesign::ObservedExposure {
            clusters: clusters.clone().into(),
            propensity_from: propensity_from.clone().into(),
            propensity_to: propensity_to.clone().into(),
            provenance: match provenance.as_str() {
                "known" => antecedent_core::ExposurePropensityProvenance::Known,
                "externally_estimated" => {
                    antecedent_core::ExposurePropensityProvenance::ExternallyEstimated
                }
                _ => {
                    return Err(IoError::Convert(
                        "unknown observational propensity provenance".into(),
                    ));
                }
            },
            assume_network_exchangeability: *assume_network_exchangeability,
        },
    };
    let exposure = match &w.exposure {
        ExposureMappingWire::OwnTreatment => ExposureMapping::OwnTreatment,
        ExposureMappingWire::NeighborCount => ExposureMapping::NeighborCount,
        ExposureMappingWire::NeighborFraction => ExposureMapping::NeighborFraction,
        ExposureMappingWire::WeightedNeighborExposure => ExposureMapping::WeightedNeighborExposure,
        ExposureMappingWire::Custom(id) => ExposureMapping::Custom(Arc::from(id.as_str())),
    };
    let InterferenceFunctionalWire::ExposureContrast { outcome, from, to } = &w.functional;
    let query = InterferenceQuery {
        assignment,
        exposure,
        functional: InterferenceFunctional::ExposureContrast {
            outcome: VariableId::from_raw(*outcome),
            from: ExposureLevel { own: from.own, neighbors: from.neighbors },
            to: ExposureLevel { own: to.own, neighbors: to.neighbors },
        },
        probability_draws: w.probability_draws,
    };
    query.validate().map_err(|error| IoError::Convert(error.to_string()))?;
    Ok(query)
}

/// Encode an interventional distribution query.
///
/// # Errors
///
/// Unsupported target population.
pub fn interventional_distribution_to_wire(
    q: &InterventionalDistributionQuery,
) -> Result<InterventionalDistributionQueryWire, IoError> {
    Ok(InterventionalDistributionQueryWire {
        outcomes: vars_to_raw(&q.outcomes),
        interventions: q
            .interventions
            .iter()
            .map(InterventionWire::from_domain)
            .collect::<Result<Vec<_>, _>>()?,
        conditioning: vars_to_raw(&q.conditioning),
        target_population: TargetPopulationWire::from_domain(&q.target_population)?,
    })
}

/// Decode an interventional distribution query.
///
/// # Errors
///
/// Row indices that do not fit `usize`.
pub fn interventional_distribution_from_wire(
    w: &InterventionalDistributionQueryWire,
) -> Result<InterventionalDistributionQuery, IoError> {
    Ok(InterventionalDistributionQuery {
        outcomes: vars_from_raw(&w.outcomes),
        interventions: w
            .interventions
            .iter()
            .map(InterventionWire::to_domain)
            .collect::<Vec<_>>()
            .into(),
        conditioning: vars_from_raw(&w.conditioning),
        target_population: w.target_population.to_domain()?,
    })
}

/// Encode a path-specific effect query.
///
/// # Errors
///
/// Unsupported target population.
pub fn path_specific_to_wire(
    q: &PathSpecificEffectQuery,
) -> Result<PathSpecificEffectQueryWire, IoError> {
    Ok(PathSpecificEffectQueryWire {
        treatment: q.treatment.raw(),
        outcome: q.outcome.raw(),
        path_nodes: vars_to_raw(&q.path_nodes),
        control: InterventionWire::from_domain(&q.control)?,
        active: InterventionWire::from_domain(&q.active)?,
        target_population: TargetPopulationWire::from_domain(&q.target_population)?,
        max_paths: crate::error::fit_u64(q.max_paths)?,
        max_len: crate::error::fit_u64(q.max_len)?,
    })
}

/// Decode a path-specific effect query.
///
/// # Errors
///
/// Limits that do not fit `usize`.
pub fn path_specific_from_wire(
    w: &PathSpecificEffectQueryWire,
) -> Result<PathSpecificEffectQuery, IoError> {
    Ok(PathSpecificEffectQuery {
        treatment: VariableId::from_raw(w.treatment),
        outcome: VariableId::from_raw(w.outcome),
        path_nodes: vars_from_raw(&w.path_nodes),
        control: w.control.to_domain(),
        active: w.active.to_domain(),
        target_population: w.target_population.to_domain()?,
        max_paths: usize::try_from(w.max_paths)
            .map_err(|_| IoError::Convert("max_paths does not fit usize".into()))?,
        max_len: usize::try_from(w.max_len)
            .map_err(|_| IoError::Convert("max_len does not fit usize".into()))?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convert::{from_cbor, to_cbor};
    use antecedent_core::{ComponentId, PopulationRegistry};

    #[test]
    fn path_limits_encode_as_the_exact_count() {
        let mut query = antecedent_core::PathSpecificEffectQuery::binary(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        );
        query.max_paths = 7;
        query.max_len = 3;
        let wire = path_specific_to_wire(&query).unwrap();
        assert_eq!(wire.max_paths, 7);
        assert_eq!(wire.max_len, 3);
        let back = path_specific_from_wire(&wire).unwrap();
        assert_eq!(back.max_paths, 7);
        assert_eq!(back.max_len, 3);
    }

    #[test]
    fn closed_query_codes_encode_as_their_historical_strings() {
        use crate::convert::assert_cbor_text_code;
        for (text, value) in [
            ("exact", ShapleyModeWire::Exact),
            ("monte_carlo", ShapleyModeWire::MonteCarlo),
            ("permutation", ShapleyModeWire::Permutation),
        ] {
            assert_cbor_text_code(text, &value);
        }
        for (text, value) in [
            ("inputs", AttributionComponentsWire::Inputs),
            ("mechanisms", AttributionComponentsWire::Mechanisms),
            ("structure", AttributionComponentsWire::Structure),
            ("inputs_and_mechanisms", AttributionComponentsWire::InputsAndMechanisms),
            ("all", AttributionComponentsWire::All),
        ] {
            assert_cbor_text_code(text, &value);
        }
        for (text, value) in [
            ("total", MediationContrastWire::Total),
            ("direct", MediationContrastWire::Direct),
            ("mediated", MediationContrastWire::Mediated),
            ("natural_direct", MediationContrastWire::NaturalDirect),
            ("natural_indirect", MediationContrastWire::NaturalIndirect),
        ] {
            assert_cbor_text_code(text, &value);
        }
    }

    #[test]
    fn randomized_itt_design_round_trips_as_its_own_query_kind() {
        let query = CausalQuery::RandomizedEffect(RandomizedEffectQuery::bernoulli_itt(
            VariableId::from_raw(0),
            [true, false],
            [0.5, 0.5],
            [Arc::<str>::from("a"), Arc::<str>::from("b")],
            [Arc::<str>::from("r0"), Arc::<str>::from("r1")],
            ("control", "treated"),
        ));
        let wire = causal_query_to_wire(&query).unwrap();
        assert!(matches!(wire, CausalQueryWire::RandomizedEffect(_)));
        assert_eq!(causal_query_from_wire(&wire).unwrap(), query);

        let cuped = CausalQuery::RandomizedEffect(
            RandomizedEffectQuery::bernoulli_itt(
                VariableId::from_raw(0),
                [true, false],
                [0.5, 0.5],
                [Arc::<str>::from("a"), Arc::<str>::from("b")],
                [Arc::<str>::from("r0"), Arc::<str>::from("r1")],
                ("control", "treated"),
            )
            .with_fixed_cuped(VariableId::from_raw(1), 4.0),
        );
        let cuped_wire = causal_query_to_wire(&cuped).unwrap();
        assert_eq!(causal_query_from_wire(&cuped_wire).unwrap(), cuped);

        let switchback = CausalQuery::RandomizedEffect(RandomizedEffectQuery::with_design(
            RandomizationDesign::Switchback {
                periods: ["p0", "p1", "p0", "p1"].map(Arc::<str>::from).into(),
            },
            VariableId::from_raw(0),
            [true, false, true, false],
            [0.5; 4],
            ["s0", "s0", "s1", "s1"].map(Arc::<str>::from),
            ["r0", "r1", "r2", "r3"].map(Arc::<str>::from),
            ("off", "on"),
        ));
        let switchback_wire = causal_query_to_wire(&switchback).unwrap();
        assert_eq!(causal_query_from_wire(&switchback_wire).unwrap(), switchback);

        let cace = CausalQuery::RandomizedEffect(
            RandomizedEffectQuery::bernoulli_itt(
                VariableId::from_raw(0),
                [true, false, true, false],
                [0.5; 4],
                ["u0", "u1", "u2", "u3"].map(Arc::<str>::from),
                ["y0", "y1", "y2", "y3"].map(Arc::<str>::from),
                ("control", "encouraged"),
            )
            .with_received_treatment([true, false, false, false]),
        );
        let cace_wire = causal_query_to_wire(&cace).unwrap();
        assert_eq!(causal_query_from_wire(&cace_wire).unwrap(), cace);

        let fisher = CausalQuery::RandomizedEffect(
            RandomizedEffectQuery::with_design(
                RandomizationDesign::Complete { treated_units: 2 },
                VariableId::from_raw(0),
                [true, true, false, false],
                [0.5; 4],
                ["u0", "u1", "u2", "u3"].map(Arc::<str>::from),
                ["y0", "y1", "y2", "y3"].map(Arc::<str>::from),
                ("control", "treated"),
            )
            .with_exact_randomization_test(),
        );
        let fisher_wire = causal_query_to_wire(&fisher).unwrap();
        assert_eq!(causal_query_from_wire(&fisher_wire).unwrap(), fisher);
        let factorial = CausalQuery::RandomizedEffect(RandomizedEffectQuery::with_design(
            RandomizationDesign::Factorial2x2 {
                second_factor_assignment: [false, false, false, false, true, true, true, true]
                    .into(),
                cell_counts: [2; 4],
                second_factor_arms: (Arc::from("off"), Arc::from("on")),
            },
            VariableId::from_raw(0),
            [false, false, true, true, false, false, true, true],
            [0.5; 8],
            (0..8).map(|i| Arc::<str>::from(format!("u{i}"))).collect::<Vec<_>>(),
            (0..8).map(|i| Arc::<str>::from(format!("y{i}"))).collect::<Vec<_>>(),
            ("control", "treated"),
        ));
        let factorial_wire = causal_query_to_wire(&factorial).unwrap();
        assert_eq!(causal_query_from_wire(&factorial_wire).unwrap(), factorial);

        let stratified = CausalQuery::RandomizedEffect(RandomizedEffectQuery::with_design(
            RandomizationDesign::Stratified {
                blocks: ["north", "north", "north", "north", "south", "south", "south", "south"]
                    .map(Arc::<str>::from)
                    .into(),
                treated_per_row: Arc::from([2; 8]),
            },
            VariableId::from_raw(0),
            [true, false, true, false, true, false, true, false],
            [0.5; 8],
            (0..8).map(|i| Arc::<str>::from(format!("u{i}"))).collect::<Vec<_>>(),
            (0..8).map(|i| Arc::<str>::from(format!("y{i}"))).collect::<Vec<_>>(),
            ("control", "treated"),
        ));
        assert_eq!(
            causal_query_from_wire(&causal_query_to_wire(&stratified).unwrap()).unwrap(),
            stratified
        );
    }

    #[test]
    fn cluster_randomized_itt_round_trips_repeated_assignment_units() {
        let query = CausalQuery::RandomizedEffect(RandomizedEffectQuery::with_design(
            RandomizationDesign::Cluster { treated_clusters: 2 },
            VariableId::from_raw(0),
            [true, true, true, false, false, false],
            [0.5; 6],
            ["a", "a", "b", "c", "c", "d"].map(Arc::<str>::from),
            ["r0", "r1", "r2", "r3", "r4", "r5"].map(Arc::<str>::from),
            ("control", "treated"),
        ));
        let wire = causal_query_to_wire(&query).unwrap();
        let bytes = to_cbor(&wire).unwrap();
        let decoded: CausalQueryWire = from_cbor(&bytes).unwrap();
        assert_eq!(causal_query_from_wire(&decoded).unwrap(), query);
    }

    #[test]
    fn multi_arm_randomized_query_round_trips_with_all_probability_rows() {
        let assignment = [0_usize, 1, 2, 0, 1, 2];
        let query = CausalQuery::RandomizedEffect(RandomizedEffectQuery::with_design(
            RandomizationDesign::MultiArm {
                assignment: assignment.into(),
                probabilities: vec![vec![1.0 / 3.0; 3]; 6].into(),
                arms: ["control", "low", "high"].map(Arc::<str>::from).into(),
            },
            VariableId::from_raw(0),
            assignment.map(|arm| arm != 0),
            [1.0 / 3.0; 6],
            (0..6).map(|i| Arc::<str>::from(format!("u{i}"))).collect::<Vec<_>>(),
            (0..6).map(|i| Arc::<str>::from(format!("r{i}"))).collect::<Vec<_>>(),
            ("control", "low"),
        ));
        let wire = causal_query_to_wire(&query).unwrap();
        assert_eq!(causal_query_from_wire(&wire).unwrap(), query);
    }

    #[test]
    fn panel_did_design_round_trips_with_subject_and_cluster_ownership() {
        let ids = ["a", "a", "b", "b", "c", "c", "d", "d"];
        let query = CausalQuery::PanelDid(PanelDidQuery::new(
            VariableId::from_raw(0),
            [true, true, true, true, false, false, false, false],
            [false, true, false, true, false, true, false, true],
            ids.map(Arc::<str>::from),
            ["x", "x", "y", "y", "z", "z", "w", "w"].map(Arc::<str>::from),
        ));
        let wire = causal_query_to_wire(&query).unwrap();
        assert!(matches!(wire, CausalQueryWire::PanelDid(_)));
        assert_eq!(causal_query_from_wire(&wire).unwrap(), query);
        let bytes = to_cbor(&wire).unwrap();
        let decoded: CausalQueryWire = from_cbor(&bytes).unwrap();
        assert_eq!(causal_query_from_wire(&decoded).unwrap(), query);
    }

    #[test]
    fn repeated_cross_section_design_round_trips_and_differs_from_panel() {
        let subjects = ["a", "b", "c", "d", "e", "f", "g", "h"].map(Arc::<str>::from);
        let clusters = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"].map(Arc::<str>::from);
        let panel = PanelDidQuery::new(
            VariableId::from_raw(0),
            [false, false, false, false, true, true, true, true],
            [false, false, true, true, false, false, true, true],
            subjects.clone(),
            clusters.clone(),
        );
        let rcs = PanelDidQuery::repeated_cross_section(
            VariableId::from_raw(0),
            [false, false, false, false, true, true, true, true],
            [false, false, true, true, false, false, true, true],
            subjects,
            clusters,
        );
        let wire = causal_query_to_wire(&CausalQuery::PanelDid(rcs.clone())).unwrap();
        assert_ne!(wire, causal_query_to_wire(&CausalQuery::PanelDid(panel)).unwrap());
        let bytes = to_cbor(&wire).unwrap();
        let decoded: CausalQueryWire = from_cbor(&bytes).unwrap();
        assert_eq!(causal_query_from_wire(&decoded).unwrap(), CausalQuery::PanelDid(rcs));
    }

    #[test]
    fn staggered_group_time_design_round_trips_with_target_and_histories() {
        let query = CausalQuery::PanelDid(PanelDidQuery::staggered_group_time(
            VariableId::from_raw(0),
            ["a", "a", "b", "b", "c", "c", "d", "d"].map(Arc::<str>::from),
            ["ca", "ca", "cb", "cb", "cc", "cc", "cd", "cd"].map(Arc::<str>::from),
            [2, 3, 2, 3, 2, 3, 2, 3],
            [0, 0, 0, 0, 3, 3, 3, 3],
            3,
            3,
        ));
        let wire = causal_query_to_wire(&query).unwrap();
        let bytes = to_cbor(&wire).unwrap();
        let decoded: CausalQueryWire = from_cbor(&bytes).unwrap();
        assert_eq!(causal_query_from_wire(&decoded).unwrap(), query);
    }

    #[test]
    fn policy_value_query_round_trips_frozen_inputs_and_ownership() {
        let query = CausalQuery::PolicyValue(PolicyValueQuery {
            outcome: VariableId::from_raw(2),
            assignment: Arc::from([false, true]),
            propensity: Arc::from([0.5]),
            actions: Arc::from([true, false]),
            reference: Arc::from([false, false]),
            mu0: Arc::from([1.0, 2.0]),
            mu1: Arc::from([3.0, 4.0]),
            costs: Arc::from([0.1]),
            reference_costs: Arc::from([0.0]),
            evaluation_subject_ids: Arc::from([Arc::<str>::from("s0"), Arc::<str>::from("s1")]),
            disjoint_training_subjects: true,
            crossfit_fold_ownership_valid: false,
            global_constraints_present: false,
            multi_action: None,
            uplift_bins: Arc::from([]),
            uplift_bin_count: 0,
            uplift_training_subject_ids: Arc::from([]),
            regret: None,
        });
        let wire = causal_query_to_wire(&query).unwrap();
        let bytes = to_cbor(&wire).unwrap();
        let restored: CausalQueryWire = from_cbor(&bytes).unwrap();
        assert_eq!(causal_query_from_wire(&restored).unwrap(), query);
    }

    #[test]
    fn misspelled_query_keys_are_rejected_not_defaulted() {
        let query = antecedent_core::CausalQuery::AverageEffect(
            antecedent_core::AverageEffectQuery::binary_ate(
                VariableId::from_raw(0),
                VariableId::from_raw(1),
            ),
        );
        let wire = causal_query_to_wire(&query).unwrap();
        let value = serde_json::to_value(&wire).unwrap();
        assert_eq!(serde_json::from_value::<CausalQueryWire>(value.clone()).unwrap(), wire);
        for (misspelled, correct) in
            [("outcome_functionl", "outcome_functional"), ("effect_modifer", "effect_modifiers")]
        {
            let mut tampered = value.clone();
            let body = tampered.get_mut("average_effect").unwrap().as_object_mut().unwrap();
            let moved = body.remove(correct).unwrap_or(serde_json::Value::Null);
            body.insert(misspelled.into(), moved);
            let error = serde_json::from_value::<CausalQueryWire>(tampered).unwrap_err();
            assert!(error.to_string().contains("unknown field"), "{misspelled}: {error}");
        }
    }

    fn registry_for_wire_tests() -> PopulationRegistry {
        let mut registry = PopulationRegistry::new();
        registry.insert_predicate("cohort", [0usize, 2, 5]);
        registry.insert_distribution_with_dependence(
            DistributionRef::from_raw(9),
            [1.0, 0.5],
            [VariableId::from_raw(2)],
        );
        registry.insert_distribution_with_dependence(
            DistributionRef::from_raw(11),
            [1.0, 2.0],
            [VariableId::from_raw(2)],
        );
        registry
    }

    #[test]
    fn average_effect_and_distribution_round_trip() {
        let q = CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ));
        let wire = causal_query_to_wire(&q).unwrap();
        let bytes = to_cbor(&wire).unwrap();
        let decoded: CausalQueryWire = from_cbor(&bytes).unwrap();
        let back = causal_query_from_wire(&decoded).unwrap();
        assert!(matches!(back, CausalQuery::AverageEffect(_)));
    }

    #[test]
    fn interventional_distribution_cbor_round_trip() {
        let q = InterventionalDistributionQuery::new(
            VariableId::from_raw(1),
            [Intervention::set(VariableId::from_raw(0), Value::f64(3.0))],
        )
        .with_conditioning([VariableId::from_raw(2)]);
        let wire = interventional_distribution_to_wire(&q).unwrap();
        let bytes = to_cbor(&wire).unwrap();
        let decoded: InterventionalDistributionQueryWire = from_cbor(&bytes).unwrap();
        let back = interventional_distribution_from_wire(&decoded).unwrap();
        assert_eq!(back.outcomes.as_ref(), q.outcomes.as_ref());
        assert_eq!(back.conditioning.as_ref(), q.conditioning.as_ref());
        assert_eq!(back.interventions.len(), 1);
        back.validate().unwrap();
    }

    #[test]
    fn path_specific_cbor_round_trip() {
        let q = PathSpecificEffectQuery::binary(VariableId::from_raw(0), VariableId::from_raw(2))
            .with_path_nodes([VariableId::from_raw(1)])
            .with_max_paths(32)
            .with_max_len(8);
        let wire = path_specific_to_wire(&q).unwrap();
        let bytes = to_cbor(&wire).unwrap();
        let decoded: PathSpecificEffectQueryWire = from_cbor(&bytes).unwrap();
        let back = path_specific_from_wire(&decoded).unwrap();
        assert_eq!(back.max_paths, 32);
        back.validate().unwrap();
    }

    #[test]
    fn transport_and_interference_queries_round_trip() {
        let response =
            antecedent_core::ResponseQuery::new(antecedent_core::ResponseFunctional::MeanCurve {
                outcome: VariableId::from_raw(1),
                treatment: antecedent_core::ContinuousDomain::new(
                    VariableId::from_raw(0),
                    antecedent_core::GridSpec::Values(Arc::from([0.0, 1.0])),
                ),
            });
        let transport = CausalQuery::Transport(TransportQuery::new(
            response,
            "trial",
            "target",
            [VariableId::from_raw(0)],
        ));
        assert_rt(&transport);

        let interference = CausalQuery::Interference(InterferenceQuery::new(
            AssignmentDesign::CompleteRandomization { treated: 2 },
            ExposureMapping::NeighborFraction,
            InterferenceFunctional::ExposureContrast {
                outcome: VariableId::from_raw(1),
                from: ExposureLevel { own: 0.0, neighbors: 0.0 },
                to: ExposureLevel { own: 1.0, neighbors: 0.5 },
            },
        ));
        assert_rt(&interference);
    }

    #[test]
    fn interference_wire_missing_draw_budget_uses_legacy_default() {
        let wire = InterferenceQueryWire {
            assignment: AssignmentDesignWire::Bernoulli { probabilities: vec![0.5] },
            exposure: ExposureMappingWire::OwnTreatment,
            functional: InterferenceFunctionalWire::ExposureContrast {
                outcome: 1,
                from: ExposureLevelWire { own: 0.0, neighbors: 0.0 },
                to: ExposureLevelWire { own: 1.0, neighbors: 0.0 },
            },
            probability_draws: 10_000,
        };
        let mut json = serde_json::to_value(&wire).unwrap();
        json.as_object_mut().unwrap().remove("probability_draws");
        let decoded: InterferenceQueryWire = serde_json::from_value(json).unwrap();
        assert_eq!(decoded.probability_draws, 10_000);
        interference_query_from_wire(&decoded).unwrap();
    }

    #[test]
    fn planned_variants_cbor_round_trip() {
        let ate = CausalQuery::AverageEffect(
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
                .with_target_population(TargetPopulation::Predicate(PredicateExpr::named(
                    "cohort_a",
                ))),
        );
        let mut registry = PopulationRegistry::new();
        registry.insert_predicate("cohort_a", [0usize, 1]);
        let wire = causal_query_to_wire_with_registry(&ate, Some(&registry)).unwrap();
        let bytes = to_cbor(&wire).unwrap();
        let decoded: CausalQueryWire = from_cbor(&bytes).unwrap();
        let back = causal_query_from_wire(&decoded).unwrap();
        match back {
            CausalQuery::AverageEffect(q) => match q.target_population {
                TargetPopulation::Predicate(PredicateExpr::Named(name)) => {
                    assert_eq!(&*name, "cohort_a");
                }
                other => panic!("expected PredicateNamed, got {other:?}"),
            },
            other => panic!("expected AverageEffect, got {other:?}"),
        }

        let rows_q = CausalQuery::AverageEffect(
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
                .with_target_population(TargetPopulation::Predicate(PredicateExpr::rows([
                    1usize, 3,
                ]))),
        );
        let back = causal_query_from_wire(&causal_query_to_wire(&rows_q).unwrap()).unwrap();
        match back {
            CausalQuery::AverageEffect(q) => match q.target_population {
                TargetPopulation::Predicate(PredicateExpr::Rows(rows)) => {
                    assert_eq!(rows.as_ref(), &[1, 3]);
                }
                other => panic!("expected PredicateRows, got {other:?}"),
            },
            other => panic!("expected AverageEffect, got {other:?}"),
        }

        let dist_q = CausalQuery::AverageEffect(
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
                .with_target_population(TargetPopulation::CustomDistribution(
                    DistributionRef::from_raw(9),
                )),
        );
        let back = causal_query_from_wire(
            &causal_query_to_wire_with_registry(&dist_q, Some(&registry_for_wire_tests())).unwrap(),
        )
        .unwrap();
        match back {
            CausalQuery::AverageEffect(q) => {
                assert_eq!(
                    q.target_population,
                    TargetPopulation::CustomDistribution(DistributionRef::from_raw(9))
                );
            }
            other => panic!("expected AverageEffect, got {other:?}"),
        }

        let temporal = CausalQuery::TemporalEffect(
            TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
                .with_policy(TemporalPolicy::dynamic(DynamicRuleId::from_raw(4), [0, 2]))
                .with_horizon_steps(2),
        );
        let back = causal_query_from_wire(&causal_query_to_wire(&temporal).unwrap()).unwrap();
        match back {
            CausalQuery::TemporalEffect(q) => {
                assert_eq!(q.policy, TemporalPolicy::dynamic(DynamicRuleId::from_raw(4), [0, 2]));
            }
            other => panic!("expected TemporalEffect, got {other:?}"),
        }
    }

    fn assert_rt(q: &CausalQuery) {
        let back = causal_query_from_wire(&causal_query_to_wire(q).unwrap()).unwrap();
        let again = causal_query_from_wire(&causal_query_to_wire(&back).unwrap()).unwrap();
        // CBOR + domain round-trip is stable under a second encode.
        let w1 = to_cbor(&causal_query_to_wire(q).unwrap()).unwrap();
        let w2 = to_cbor(&causal_query_to_wire(&again).unwrap()).unwrap();
        assert_eq!(w1, w2, "wire bytes drifted for {q:?}");
    }

    #[test]
    fn all_intervention_variants_round_trip() {
        let t = VariableId::from_raw(0);
        let y = VariableId::from_raw(1);
        let interventions = [
            Intervention::set(t, Value::f64(1.0)),
            Intervention::shift(t, Value::f64(0.5)),
            Intervention::stochastic(t, StochasticPolicy::bernoulli(0.4)),
            Intervention::stochastic(t, StochasticPolicy::gaussian(0.0, 1.0)),
            Intervention::stochastic(
                t,
                StochasticPolicy::Categorical { probs: Arc::from([0.2, 0.3, 0.5]) },
            ),
            Intervention::soft(t, MechanismOverride::constant(2.0)),
            Intervention::soft(t, MechanismOverride::named("linear_gaussian", [0.1, 0.2, 1.0])),
            Intervention::sequence(InterventionSequence::new(vec![
                SequencedIntervention::new(
                    Intervention::set(t, Value::f64(0.0)),
                    TemporalPolicy::pulse(0),
                ),
                SequencedIntervention::new(
                    Intervention::shift(t, Value::f64(1.0)),
                    TemporalPolicy::sustained(1, 3),
                ),
                SequencedIntervention::new(
                    Intervention::stochastic(t, StochasticPolicy::bernoulli(0.5)),
                    TemporalPolicy::dynamic(DynamicRuleId::from_raw(9), [0, 4, 8]),
                ),
            ])),
        ];
        for iv in interventions {
            let q = CausalQuery::Counterfactual(
                CounterfactualQuery::new(y, [iv.clone()]).with_nested(true),
            );
            assert_rt(&q);
            let q = CausalQuery::Distribution(InterventionalDistributionQuery::new(y, [iv]));
            assert_rt(&q);
        }
    }

    #[test]
    fn all_target_populations_round_trip() {
        let ate = |pop| {
            CausalQuery::AverageEffect(
                AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
                    .with_target_population(pop),
            )
        };
        for pop in [
            TargetPopulation::AllObserved,
            TargetPopulation::Treated,
            TargetPopulation::Untreated,
            TargetPopulation::Environment(EnvironmentId::from_raw(3)),
            TargetPopulation::Predicate(PredicateExpr::rows([0usize, 2, 5])),
            TargetPopulation::local_at_cutoff(VariableId::from_raw(2), -1.25),
        ] {
            assert_rt(&ate(pop));
        }
        let local = ate(TargetPopulation::local_at_cutoff(VariableId::from_raw(2), -1.25));
        let back = causal_query_from_wire(&causal_query_to_wire(&local).unwrap()).unwrap();
        assert_eq!(back, local);
        let registry = registry_for_wire_tests();
        for pop in [
            TargetPopulation::Predicate(PredicateExpr::named("cohort")),
            TargetPopulation::CustomDistribution(DistributionRef::from_raw(11)),
        ] {
            let q = ate(pop);
            let wire = causal_query_to_wire_with_registry(&q, Some(&registry)).unwrap();
            let back = causal_query_from_wire(&wire).unwrap();
            let again = causal_query_to_wire_with_registry(&back, Some(&registry)).unwrap();
            assert_eq!(to_cbor(&wire).unwrap(), to_cbor(&again).unwrap());
        }
    }

    #[test]
    fn all_population_selectors_and_attribution_components_round_trip() {
        let selectors = [
            PopulationSelector::All,
            PopulationSelector::Rows(Arc::from([0usize, 1, 4])),
            PopulationSelector::Environment { env_index: 2 },
            PopulationSelector::TimeRange { start: 10, end: 20 },
        ];
        let components = [
            AttributionComponents::Inputs,
            AttributionComponents::Mechanisms,
            AttributionComponents::Structure,
            AttributionComponents::InputsAndMechanisms,
            AttributionComponents::All,
        ];
        let allocations = [
            AllocationMethod::PathBased,
            AllocationMethod::Sequential {
                order: Arc::from([
                    ComponentId::from(VariableId::from_raw(0)),
                    ComponentId::from(VariableId::from_raw(1)),
                ]),
            },
            AllocationMethod::Shapley { approximation: ShapleyConfig::exact() },
            AllocationMethod::Shapley { approximation: ShapleyConfig::monte_carlo(500) },
            AllocationMethod::Shapley { approximation: ShapleyConfig::permutation(100) },
        ];
        for base in &selectors {
            for comp in &selectors {
                for components in &components {
                    for allocation in &allocations {
                        let q = CausalQuery::ChangeAttribution(
                            ChangeAttributionQuery::new(
                                VariableId::from_raw(2),
                                base.clone(),
                                comp.clone(),
                            )
                            .with_components(*components)
                            .with_allocation(allocation.clone())
                            .with_max_components(16),
                        );
                        assert_rt(&q);
                    }
                }
            }
        }
    }

    #[test]
    fn anomaly_attribution_reference_presence_survives_re_encoding() {
        // Earlier writers emitted no `reference` key; later ones wrote `reference: null` for an
        // empirical reference. Both shapes must re-encode to their own bytes,
        // or the digests they were sealed under stop matching.
        #[derive(Serialize)]
        #[serde(rename_all = "snake_case")]
        enum V200 {
            AnomalyAttribution { targets: Vec<u32>, unit_rows: Option<Vec<u64>>, max_units: u64 },
        }
        #[derive(Serialize)]
        #[serde(rename_all = "snake_case")]
        enum V210 {
            AnomalyAttribution {
                targets: Vec<u32>,
                unit_rows: Option<Vec<u64>>,
                max_units: u64,
                reference: Option<(f64, f64)>,
            },
        }
        let v200 = to_cbor(&V200::AnomalyAttribution {
            targets: vec![1, 2],
            unit_rows: Some(vec![0, 3]),
            max_units: 64,
        })
        .unwrap();
        let v210 = to_cbor(&V210::AnomalyAttribution {
            targets: vec![1, 2],
            unit_rows: Some(vec![0, 3]),
            max_units: 64,
            reference: None,
        })
        .unwrap();
        for legacy in [v200, v210] {
            let decoded: CausalQueryWire = from_cbor(&legacy).unwrap();
            assert_eq!(to_cbor(&decoded).unwrap(), legacy, "re-encoding changed {decoded:?}");
            let query = causal_query_from_wire(&decoded).unwrap();
            let CausalQuery::AnomalyAttribution(query) = query else { panic!("{query:?}") };
            assert_eq!(query.reference, AnomalyReference::Empirical);
        }
    }

    #[test]
    fn remaining_causal_query_variants_round_trip() {
        let t = VariableId::from_raw(0);
        let y = VariableId::from_raw(1);
        let m = VariableId::from_raw(2);
        let z = VariableId::from_raw(3);

        assert_rt(&CausalQuery::AnomalyAttribution(
            AnomalyAttributionQuery::new([y, m], 64).with_unit_rows([0usize, 1, 2]),
        ));
        assert_rt(&CausalQuery::AnomalyAttribution(
            AnomalyAttributionQuery::new([y], 64)
                .with_reference(AnomalyReference::fixed(1.5, 2.25)),
        ));
        assert_rt(&CausalQuery::MechanismChange(MechanismChangeQuery::new(
            [t, y],
            PopulationSelector::All,
            PopulationSelector::Environment { env_index: 1 },
            0.05,
            16,
        )));
        assert_rt(&CausalQuery::UnitChange(
            UnitChangeQuery::new(y, 32).with_unit_rows([3usize, 4]),
        ));
        assert_rt(&CausalQuery::Mediation(MediationQuery::binary(
            t,
            y,
            [m],
            MediationContrast::Direct,
        )));
        assert_rt(&CausalQuery::Mediation(MediationQuery::binary(
            t,
            y,
            [m],
            MediationContrast::NaturalIndirect,
        )));
        assert_rt(&CausalQuery::NestedCounterfactual(
            antecedent_core::NestedCounterfactualQuery::with_levels(t, m, y, -0.5, 1.5).unwrap(),
        ));
        let conditional = ConditionalEffectQuery::try_new(
            AverageEffectQuery::binary_ate(t, y).with_effect_modifiers([z]),
        )
        .unwrap();
        assert_rt(&CausalQuery::ConditionalEffect(conditional));
        assert_rt(&CausalQuery::PathSpecific(
            PathSpecificEffectQuery::binary(t, y)
                .with_path_nodes([m])
                .with_max_paths(8)
                .with_max_len(4),
        ));
        assert_rt(&CausalQuery::TemporalEffect(
            TemporalEffectQuery::pulse(t, y, 1.0)
                .with_policy(TemporalPolicy::sustained(0, 5))
                .with_horizon_steps(6),
        ));
        assert_rt(&CausalQuery::Counterfactual(
            CounterfactualQuery::new(
                y,
                [
                    Intervention::set(t, Value::f64(1.0)),
                    Intervention::soft(m, MechanismOverride::additive_shift(0.25)),
                ],
            )
            .with_nested(false),
        ));
        assert_rt(&CausalQuery::Distribution(
            InterventionalDistributionQuery::new(
                y,
                [Intervention::stochastic(t, StochasticPolicy::gaussian(0.0, 2.0))],
            )
            .with_conditioning([z]),
        ));
    }

    #[test]
    fn counterfactual_wire_omitted_control_defaults_to_zero() {
        let t = VariableId::from_raw(0);
        let y = VariableId::from_raw(1);
        let q = CausalQuery::Counterfactual(CounterfactualQuery::new(
            y,
            [Intervention::set(t, Value::f64(1.0))],
        ));
        let mut json = serde_json::to_value(causal_query_to_wire(&q).unwrap()).unwrap();
        json.as_object_mut()
            .and_then(|root| root.get_mut("counterfactual"))
            .and_then(serde_json::Value::as_object_mut)
            .expect("externally tagged counterfactual")
            .remove("control");
        let decoded: CausalQueryWire = serde_json::from_value(json).unwrap();
        let back = causal_query_from_wire(&decoded).unwrap();
        match back {
            CausalQuery::Counterfactual(q) => {
                assert_eq!(q.control, Intervention::set(t, Value::f64(0.0)));
                assert_eq!(q.interventions.as_ref(), &[Intervention::set(t, Value::f64(1.0))]);
            }
            other => panic!("expected Counterfactual, got {other:?}"),
        }
    }

    fn every_expression_form() -> ValueExpr {
        let x = VariableId::from_raw(0);
        let z = VariableId::from_raw(1);
        let branch = ValueExpr::var(z).gt(ValueExpr::constant(0.0)).and(
            ValueExpr::lag(z, 2)
                .le(ValueExpr::constant(1.0))
                .or(ValueExpr::var(x).not_equals(ValueExpr::constant(3.0)).not()),
        );
        branch
            .if_then_else(
                ValueExpr::generated().sub(ValueExpr::constant(2.0)).neg(),
                ValueExpr::generated()
                    .clip(ValueExpr::constant(-1.0), ValueExpr::constant(1.0))
                    .div(ValueExpr::constant(4.0))
                    .min(ValueExpr::constant(9.0))
                    .max(ValueExpr::constant(-9.0)),
            )
            .add(ValueExpr::var(z).equals(ValueExpr::constant(0.5)).indicator())
            .mul(
                ValueExpr::var(z)
                    .ge(ValueExpr::constant(0.0))
                    .indicator()
                    .add(ValueExpr::constant(1.0)),
            )
            .add(ValueExpr::var(x).lt(ValueExpr::constant(1.0)).indicator())
    }

    #[test]
    fn programmed_interventions_round_trip_through_cbor() {
        let x = VariableId::from_raw(0);
        let expr = every_expression_form();
        for iv in [Intervention::transform(x, expr.clone()), Intervention::assign(x, expr)] {
            let wire = InterventionWire::from_domain(&iv).unwrap();
            let back: InterventionWire = from_cbor(&to_cbor(&wire).unwrap()).unwrap();
            assert_eq!(back, wire);
            assert_eq!(back.to_domain(), iv);
            let again = InterventionWire::from_domain(&back.to_domain()).unwrap();
            assert_eq!(to_cbor(&again).unwrap(), to_cbor(&wire).unwrap(), "encoding is canonical");
        }
    }

    #[test]
    fn released_interventions_keep_their_encoding_and_programs_use_new_tags() {
        let x = VariableId::from_raw(0);
        let set = InterventionWire::from_domain(&Intervention::set(x, Value::f64(1.0))).unwrap();
        let set_json = serde_json::to_string(&set).unwrap();
        assert_eq!(set_json, r#"{"set":{"variable":0,"value":{"float64":1.0}}}"#);
        let program = InterventionWire::from_domain(&Intervention::transform(
            x,
            ValueExpr::generated().add(ValueExpr::constant(1.0)),
        ))
        .unwrap();
        let json = serde_json::to_value(&program).unwrap();
        assert!(json.get("transform").is_some(), "{json}");
        // A wire that edits the declared reads decodes verbatim; validation, not the
        // codec, is what refuses the mismatch.
        let mut tampered = program.clone();
        if let InterventionWire::Transform { reads, .. } = &mut tampered {
            reads.current.push(7);
        }
        assert!(tampered.to_domain().validate().is_err());
    }
}
