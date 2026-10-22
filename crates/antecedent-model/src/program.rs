//! Compiled intervention programs: value expressions run column-wise over draws.
//!
//! A [`ValueExpr`] attached to a node by `Intervention::Transform` or
//! `Intervention::Assign` is validated against the compiled model and flattened to a
//! postfix program. [`apply_node_program`] is the one executor every sampler shares:
//! forward ancestral sampling, conditional interventional sampling, the do-samplers and
//! counterfactual world evaluation all call it, so a policy means the same thing on
//! every route.
//!
//! A temporal route supplies history by laying its value batch out with one extra block of
//! columns per step back ([`history_slot`]) and compiling with [`NodeProgram::compile_windowed`];
//! a lagged read is then a load like any other, so the executor is unchanged.
//!
//! Two semantics, kept distinct:
//!
//! * **Transform** runs after the node's mechanism produced its value (parents, noise,
//!   soft override and stochastic policy all flow through) and replaces that value with
//!   the expression, which may read it as `Generated`. The node keeps its incoming edges
//!   and its noise.
//! * **Assign** discards the mechanism; the node's value is the expression of other
//!   variables. Incoming edges are severed.
//!
//! Reads are of other nodes' *current-draw* values, after any intervention on them. The
//! reads a program declares, and the graph's own parent edges, form one dependency
//! graph: a read that closes a cycle is refused, and otherwise nodes are evaluated in a
//! topological order of that combined graph, so a policy always sees its inputs computed.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Arc;

use antecedent_core::{
    ArithOp, CallbackCall, CallbackError, CompareOp, ExecutionContext, ExpressionFault,
    Intervention, LogicOp, PolicyReads, ValueExpr, VariableId, clip_value, reason_code,
};
use antecedent_graph::DenseNodeId;

use crate::batch::ValueBatchMut;
use crate::compile::{CompiledCausalModel, MechanismSlot};
use crate::error::ModelError;

/// Which part of a node's evaluation a program replaces.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ProgramKind {
    /// Replaces the generated value after the mechanism ran.
    Transform,
    /// Replaces the assignment; the mechanism is not evaluated.
    Assign,
}

/// A typed refusal of a policy: reason code, namespaced detail, subject and remedy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PolicyRefusal {
    /// Registered reason code.
    pub code: &'static str,
    /// Namespaced detail naming the specific failure.
    pub detail: &'static str,
    /// What was refused (variables and policies involved).
    pub subject: String,
    /// What would make the request valid.
    pub remedy: &'static str,
}

impl PolicyRefusal {
    /// Refusal as a model error whose message carries the reason code, so the facade
    /// classifies it like every other coded refusal.
    #[must_use]
    pub fn into_error(self) -> ModelError {
        ModelError::Unsupported {
            message: format!(
                "{}{}: {}: {}; {}",
                antecedent_core::reason_code::PREFIX,
                self.code,
                self.detail,
                self.subject,
                self.remedy
            ),
        }
    }
}

impl From<PolicyRefusal> for ModelError {
    fn from(refusal: PolicyRefusal) -> Self {
        refusal.into_error()
    }
}

fn refusal_circular(subject: String) -> PolicyRefusal {
    let _refusal = antecedent_graph::error::CyclicDagRefusal::new(
        antecedent_graph::error::DagOperation::Simulation,
        antecedent_graph::error::CyclicNextAttempt::Unlicensed {
            consumer: "this topological executor",
            attempt: "schedule the dependency feedback with CyclicDigraph::condensation, which \
                      does not execute the policies",
        },
    );
    PolicyRefusal {
        code: reason_code!("policy_dependency_circular"),
        detail: "intervention_program.dependency_circular",
        subject,
        remedy: "schedule the dependency feedback with CyclicDigraph::condensation, which does \
                 not execute the policies",
    }
}

fn refusal_unknown_variable(subject: String) -> PolicyRefusal {
    PolicyRefusal {
        code: reason_code!("policy_dependency_unavailable"),
        detail: "intervention_program.variable_not_in_model",
        subject,
        remedy: "read only variables of the compiled model",
    }
}

fn refusal_generated_unavailable(subject: String) -> PolicyRefusal {
    PolicyRefusal {
        code: reason_code!("policy_dependency_unavailable"),
        detail: "intervention_program.generated_value_unavailable",
        subject,
        remedy: "use a transform to read the generated value, or drop the read from the \
                 assignment",
    }
}

fn refusal_undeclared_read(subject: String) -> PolicyRefusal {
    PolicyRefusal {
        code: reason_code!("policy_dependency_unavailable"),
        detail: "intervention_program.undeclared_read",
        subject,
        remedy: "declare exactly the variables the expression references",
    }
}

fn refusal_history_unavailable(subject: String) -> PolicyRefusal {
    PolicyRefusal {
        code: reason_code!("policy_history_unavailable"),
        detail: "intervention_program.history_unavailable",
        subject,
        remedy: "read current values here, or run the policy on a temporal route that declares \
                 a history window",
    }
}

fn refusal_history_not_past(subject: String) -> PolicyRefusal {
    PolicyRefusal {
        code: reason_code!("policy_history_unavailable"),
        detail: "intervention_program.history_not_past",
        subject,
        remedy: "a history read must look back at least one step; future values do not exist \
                 when the policy is applied",
    }
}

fn refusal_history_beyond_window(subject: String) -> PolicyRefusal {
    PolicyRefusal {
        code: reason_code!("policy_history_unavailable"),
        detail: "intervention_program.history_beyond_window",
        subject,
        remedy: "read no further back than the route's declared history window, or widen the \
                 window",
    }
}

pub(crate) fn refusal_history_not_yet_observed(subject: String) -> PolicyRefusal {
    PolicyRefusal {
        code: reason_code!("policy_history_unavailable"),
        detail: "intervention_program.history_not_yet_observed",
        subject,
        remedy: "start the policy once enough steps exist for its deepest look-back, or supply \
                 initial history",
    }
}

pub(crate) fn refusal_directive_conflict(subject: String) -> PolicyRefusal {
    PolicyRefusal {
        code: reason_code!("policy_directive_conflict"),
        detail: "intervention_program.directive_conflict",
        subject,
        remedy: "state one directive for the node, or make the repeated directive identical",
    }
}

fn refusal_output_type(subject: String) -> PolicyRefusal {
    PolicyRefusal {
        code: reason_code!("policy_output_type_invalid"),
        detail: "intervention_program.output_type",
        subject,
        remedy: "make the expression numeric (convert a condition with indicator)",
    }
}

fn refusal_categorical_output(subject: String) -> PolicyRefusal {
    PolicyRefusal {
        code: reason_code!("policy_output_type_invalid"),
        detail: "intervention_program.categorical_output",
        subject,
        remedy: "write a level of the node's support, or an indicator when the support holds 0 \
                 and 1",
    }
}

fn refusal_expression_invalid(detail: &'static str, subject: String) -> PolicyRefusal {
    PolicyRefusal {
        code: reason_code!("invalid_argument"),
        detail,
        subject,
        remedy: "use a well-typed expression of finite constants within the size bound",
    }
}

fn refusal_budget(subject: String) -> PolicyRefusal {
    PolicyRefusal {
        code: reason_code!("invalid_argument"),
        detail: "intervention_program.memory_budget_exceeded",
        subject,
        remedy: "evaluate fewer rows at a time or raise the hard memory limit",
    }
}

fn refusal_route(subject: String) -> PolicyRefusal {
    PolicyRefusal {
        code: reason_code!("policy_route_not_licensed"),
        detail: "intervention_program.identification_requires_hard_level",
        subject,
        remedy: "simulate or execute the counterfactual, or state a hard level (a constant \
                 assignment) for observational identification",
    }
}

fn refusal_callback_declaration(subject: String) -> PolicyRefusal {
    PolicyRefusal {
        code: reason_code!("invalid_argument"),
        detail: "intervention_program.callback_declaration",
        subject,
        remedy: "declare a named callback with distinct input labels, one argument per input, \
                 and a well-formed output",
    }
}

/// Refusal for a callback that raised or returned an output violating its declaration.
fn refusal_callback(call: &CallbackCall, node: usize, error: &CallbackError) -> PolicyRefusal {
    let subject = format!("callback {} on dense node {node}: {error}", call.spec().name);
    if error.is_raised() {
        PolicyRefusal {
            code: reason_code!("policy_callback_failed"),
            detail: "intervention_program.callback_failed",
            subject,
            remedy: "fix the callback; its failure is reported as raised and never replaced by \
                     a default value",
        }
    } else {
        PolicyRefusal {
            code: reason_code!("policy_callback_output_invalid"),
            detail: "intervention_program.callback_output",
            subject,
            remedy: "return one finite real value per row inside the declared output, or widen \
                     the declaration",
        }
    }
}

/// Map an expression fault to the policy refusal it is.
fn refusal_for_fault(fault: ExpressionFault, subject: String) -> PolicyRefusal {
    match fault {
        ExpressionFault::OutputType { .. } => refusal_output_type(subject),
        ExpressionFault::TypeMismatch { .. } | ExpressionFault::BranchTypes { .. } => {
            refusal_expression_invalid("intervention_program.operand_type", subject)
        }
        ExpressionFault::ZeroLag => refusal_history_not_past(subject),
        ExpressionFault::GeneratedInAssignment => refusal_generated_unavailable(subject),
        ExpressionFault::DeclaredReadsMismatch => refusal_undeclared_read(subject),
        ExpressionFault::CallbackArity { .. } | ExpressionFault::CallbackDeclaration => {
            refusal_callback_declaration(subject)
        }
        // Remaining faults (non-finite constant, size bound) describe a malformed tree.
        _ => refusal_expression_invalid("intervention_program.expression_invalid", subject),
    }
}

#[derive(Clone, Copy, Debug)]
enum Op {
    Const(f64),
    Generated,
    Read(usize),
    Neg,
    Arith(ArithOp),
    /// `op(constant, x)` or `op(x, constant)` on the top column.
    ArithConst {
        op: ArithOp,
        constant: f64,
        constant_left: bool,
    },
    Clip,
    Compare(CompareOp),
    Not,
    Logic(LogicOp),
    Select,
    /// The `k`-th callback of the program, applied to the argument columns on the stack.
    Callback(usize),
}

/// A validated, flattened program for one node.
#[derive(Clone, Debug)]
pub struct NodeProgram {
    kind: ProgramKind,
    expr: ValueExpr,
    ops: Arc<[Op]>,
    reads: Arc<[DenseNodeId]>,
    /// Columns of the value batch holding lagged copies (see [`history_slot`]); empty
    /// unless the program reads history.
    lag_slots: Arc<[usize]>,
    /// Deepest look-back, in steps (zero when the program reads no history).
    max_lag: u32,
    /// Callback calls in emission order; `Op::Callback(k)` runs `callbacks[k]`.
    callbacks: Arc<[CallbackCall]>,
    /// Work columns the executor needs, including one for a callback's output.
    max_stack: usize,
    /// Present only for an explicitly column-bound component program.
    input_axis: Option<Arc<[VariableId]>>,
}

impl PartialEq for NodeProgram {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind && self.expr == other.expr && self.input_axis == other.input_axis
    }
}

impl NodeProgram {
    /// Validate `expr` as the program of `target` against `model` and flatten it.
    ///
    /// # Errors
    ///
    /// A coded policy refusal: invalid or mistyped expression, an assignment that reads
    /// the generated value, undeclared or unknown reads, a history read, or a
    /// non-category output onto a categorical node.
    pub fn compile(
        model: &CompiledCausalModel,
        target: VariableId,
        kind: ProgramKind,
        expr: &ValueExpr,
        declared: &PolicyReads,
    ) -> Result<Self, PolicyRefusal> {
        Self::compile_windowed(model, target, kind, expr, declared, 0)
    }

    /// [`Self::compile`] for a route that keeps `window` steps of history, so the program
    /// may read a variable up to `window` steps back. A window of zero is the static case.
    ///
    /// The lagged reads are loads from the history columns of a batch laid out by
    /// [`history_slot`]; the route fills those columns before each step and the one
    /// executor ([`apply_node_program`]) reads them like any other column.
    ///
    /// # Errors
    ///
    /// As [`Self::compile`]; a read deeper than `window` is a history refusal.
    pub fn compile_windowed(
        model: &CompiledCausalModel,
        target: VariableId,
        kind: ProgramKind,
        expr: &ValueExpr,
        declared: &PolicyReads,
        window: u32,
    ) -> Result<Self, PolicyRefusal> {
        let verb = match kind {
            ProgramKind::Transform => "transform",
            ProgramKind::Assign => "assign",
        };
        let label = |what: &str| format!("{verb} of {target}: {what}");
        expr.validate_program(kind == ProgramKind::Transform, declared)
            .map_err(|fault| refusal_for_fault(fault, label(&fault.to_string())))?;
        let mut lag_slots = Vec::with_capacity(declared.history.len());
        for (variable, lag) in declared.history.iter() {
            if window == 0 {
                return Err(refusal_history_unavailable(label(&format!(
                    "reads {variable} at lag {lag}, but a static model has no history window"
                ))));
            }
            if *lag > window {
                return Err(refusal_history_beyond_window(label(&format!(
                    "reads {variable} at lag {lag}, beyond the history window of {window} steps"
                ))));
            }
            let dense = model.dense_of(*variable).ok_or_else(|| {
                refusal_unknown_variable(label(&format!("reads {variable}, absent from the model")))
            })?;
            lag_slots.push(history_slot(model.n_nodes(), dense.as_usize(), *lag));
        }
        let max_lag = declared.history.iter().map(|(_, lag)| *lag).max().unwrap_or(0);
        let mut reads = Vec::with_capacity(declared.current.len());
        for variable in declared.current.iter() {
            let dense = model.dense_of(*variable).ok_or_else(|| {
                refusal_unknown_variable(label(&format!("reads {variable}, absent from the model")))
            })?;
            reads.push(dense);
        }
        if let Some(slot_support) = categorical_support(model, target) {
            let admitted = match expr.as_constant() {
                Some(level) => slot_support.contains(&level),
                None => match expr.callback_levels() {
                    // Every returned value is checked against these levels at execution.
                    Some(levels) => levels.iter().all(|level| slot_support.contains(level)),
                    None => {
                        expr.is_binary_valued()
                            && slot_support.contains(&0.0)
                            && slot_support.contains(&1.0)
                    }
                },
            };
            if !admitted {
                return Err(refusal_categorical_output(label(
                    "the node is categorical and the expression is not a level of its support",
                )));
            }
        }
        let (ops, callbacks, max_stack) =
            lowered_ops(model.n_nodes(), &|variable| model.dense_of(variable), expr);
        Ok(Self {
            kind,
            expr: expr.clone(),
            ops,
            reads: reads.into(),
            lag_slots: lag_slots.into(),
            max_lag,
            callbacks,
            max_stack,
            input_axis: None,
        })
    }

    /// Bind the existing portable expression language to an explicit numeric column axis.
    ///
    /// This component compiler grants no graph, identification or intervention authority.
    /// Input columns already carry their caller-declared scale. It uses the same validated
    /// postfix lowering and executor as causal-model intervention programs.
    ///
    /// # Errors
    /// Refuses duplicate/unknown columns, generated/history reads, callbacks and invalid expressions.
    pub fn compile_columns(axis: &[VariableId], expr: &ValueExpr) -> Result<Self, PolicyRefusal> {
        if axis.iter().copied().collect::<std::collections::BTreeSet<_>>().len() != axis.len()
            || axis.len() >= u32::MAX as usize
        {
            return Err(refusal_unknown_variable(
                "column expression requires a unique bounded variable axis".into(),
            ));
        }
        let declared = expr.reads();
        if !declared.history.is_empty() {
            return Err(refusal_history_unavailable(
                "column expressions cannot read implicit history".into(),
            ));
        }
        if expr.reads_generated() {
            return Err(refusal_generated_unavailable(
                "column expressions have no generated node value".into(),
            ));
        }
        if !expr.callbacks().is_empty() {
            return Err(refusal_unknown_variable(
                "column expressions require portable callback-free operators".into(),
            ));
        }
        expr.validate_program(false, &declared)
            .map_err(|fault| refusal_for_fault(fault, "invalid column expression".into()))?;
        let resolve = |variable| {
            axis.iter().position(|&candidate| candidate == variable).map(DenseNodeId::from_index)
        };
        let reads = declared
            .current
            .iter()
            .map(|&variable| {
                resolve(variable).ok_or_else(|| {
                    refusal_unknown_variable(format!(
                        "column expression reads absent variable {variable}"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (ops, callbacks, max_stack) = lowered_ops(axis.len() + 1, &resolve, expr);
        Ok(Self {
            kind: ProgramKind::Assign,
            expr: expr.clone(),
            ops,
            reads: reads.into(),
            lag_slots: Arc::from([]),
            max_lag: 0,
            callbacks,
            max_stack,
            input_axis: Some(Arc::from(axis)),
        })
    }

    /// Conservative compilation/storage quote from an already bounded expression node count.
    /// Includes temporary expression copies, resolved reads, postfix emission and validation work.
    /// The enclosing owner separately counts its portable wire representation.
    ///
    /// # Errors
    /// Invalid dimensions or overflowing allocation arithmetic.
    pub fn column_compile_bytes(
        axis_width: usize,
        expression_nodes: usize,
    ) -> Result<u64, ModelError> {
        if expression_nodes == 0
            || expression_nodes > antecedent_core::value_expr::MAX_EXPR_NODES
            || axis_width >= u32::MAX as usize
        {
            return Err(
                refusal_budget("invalid bounded column compilation dimensions".into()).into_error()
            );
        }
        let per_node = 2 * std::mem::size_of::<ValueExpr>() + std::mem::size_of::<Op>() + 256;
        expression_nodes
            .checked_mul(per_node)
            .and_then(|n| n.checked_add(axis_width.checked_mul(32)?))
            .and_then(|n| n.checked_add(std::mem::size_of::<Self>()))
            .and_then(|n| u64::try_from(n).ok())
            .ok_or_else(|| refusal_budget("column compilation quote overflow".into()).into_error())
    }

    /// Conservative retained payload quote of a compiled callback-free column plan.
    /// Includes its expression tree, postfix operations, resolved read axis and shared-array headers.
    ///
    /// # Errors
    /// Refuses non-column programs or an overflowing retained size.
    pub fn column_plan_bytes(&self) -> Result<u64, ModelError> {
        let axis = self.input_axis.as_ref().ok_or_else(|| ModelError::Shape {
            message: "column plan quote requires a column-bound program".into(),
        })?;
        let parts = [
            std::mem::size_of::<Self>(),
            self.ops.len().checked_mul(std::mem::size_of::<Op>()).unwrap_or(usize::MAX),
            self.reads.len().checked_mul(std::mem::size_of::<DenseNodeId>()).unwrap_or(usize::MAX),
            axis.len().checked_mul(std::mem::size_of::<VariableId>()).unwrap_or(usize::MAX),
            column_expression_nodes(&self.expr)
                .and_then(|n| n.checked_mul(std::mem::size_of::<ValueExpr>()))
                .unwrap_or(usize::MAX),
            8 * std::mem::size_of::<usize>(),
        ];
        parts
            .into_iter()
            .try_fold(0_usize, usize::checked_add)
            .and_then(|n| u64::try_from(n).ok())
            .ok_or_else(|| refusal_budget("column plan size overflow".into()).into_error())
    }

    /// Exact value/output/postfix buffer payload needed by scalar column evaluation.
    /// This is a declared allocation quote, not process memory or allocator measurement.
    ///
    /// # Errors
    /// Refuses non-column programs or an overflowing work size.
    pub fn column_workspace_bytes(&self) -> Result<u64, ModelError> {
        let width = self.input_axis.as_ref().map(|axis| axis.len()).ok_or_else(|| {
            ModelError::Shape { message: "column workspace requires a column-bound program".into() }
        })?;
        width
            .checked_add(1)
            .and_then(|n| n.checked_add(self.max_stack))
            .and_then(|n| n.checked_mul(8))
            .and_then(|n| u64::try_from(n).ok())
            .ok_or_else(|| {
                refusal_budget("column program scratch size overflow".into()).into_error()
            })
    }

    /// Evaluate one scalar row through the existing batched postfix executor.
    ///
    /// # Errors
    /// Refuses a non-column program, wrong input width, nonfinite values, cancellation or resource failure.
    pub fn evaluate_columns(
        &self,
        inputs: &[f64],
        ctx: Option<&ExecutionContext>,
    ) -> Result<f64, ModelError> {
        self.evaluate_columns_with_scratch(inputs, &mut Vec::new(), &mut Vec::new(), ctx)
    }

    /// Evaluate a scalar row with reusable value and postfix-stack buffers.
    ///
    /// # Errors
    /// The same input, expression, cancellation and memory failures as [`Self::evaluate_columns`].
    pub fn evaluate_columns_with_scratch(
        &self,
        inputs: &[f64],
        values: &mut Vec<f64>,
        scratch: &mut Vec<f64>,
        ctx: Option<&ExecutionContext>,
    ) -> Result<f64, ModelError> {
        let width =
            self.input_axis.as_ref().map(|axis| axis.len()).ok_or_else(|| ModelError::Shape {
                message: "scalar column evaluation requires a column-bound program".into(),
            })?;
        if inputs.len() != width || inputs.iter().any(|value| !value.is_finite()) {
            return Err(ModelError::Shape {
                message: "column expression input width or finite values differ".into(),
            });
        }
        if ctx.is_some_and(|ctx| ctx.cancellation.is_cancelled()) {
            return Err(ModelError::Cancelled);
        }
        let bytes = self.column_workspace_bytes()?;
        if ctx.and_then(|ctx| ctx.memory_limit_bytes()).is_some_and(|limit| bytes > limit) {
            return Err(refusal_budget(
                "column program scratch exceeds declared hard memory limit".into(),
            )
            .into_error());
        }
        let _lease = ctx
            .map(|ctx| {
                ctx.acquire_admission(0, bytes)
                    .map_err(|error| refusal_budget(error.to_string()).into_error())
            })
            .transpose()?;
        values.clear();
        values.extend_from_slice(inputs);
        values.push(0.0);
        let mut batch = ValueBatchMut::new(1, width + 1, values)?;
        apply_node_program(self, &mut batch, width, scratch, ctx)?;
        Ok(values[width])
    }

    /// Whether this program transforms the generated value or assigns.
    #[must_use]
    pub fn kind(&self) -> ProgramKind {
        self.kind
    }

    /// The source expression.
    #[must_use]
    pub fn expression(&self) -> &ValueExpr {
        &self.expr
    }

    /// Dense nodes the program reads.
    #[must_use]
    pub fn reads(&self) -> &[DenseNodeId] {
        &self.reads
    }

    /// Deepest look-back in steps; zero when the program reads no history.
    #[must_use]
    pub fn max_lag(&self) -> u32 {
        self.max_lag
    }
}

/// Column of a history-carrying value batch that holds `dense` node's value `lag` steps back.
///
/// The batch has `n_nodes * (window + 1)` columns: the current step's columns first, then
/// one block of `n_nodes` columns per step back. Static batches have only the first block.
#[must_use]
pub fn history_slot(n_nodes: usize, dense: usize, lag: u32) -> usize {
    n_nodes.saturating_mul(lag as usize).saturating_add(dense)
}

fn categorical_support(model: &CompiledCausalModel, target: VariableId) -> Option<Arc<[f64]>> {
    let dense = model.dense_of(target)?;
    match model.mechanisms.get(dense) {
        MechanismSlot::Discrete { support, .. } | MechanismSlot::DiscreteBasis { support, .. } => {
            Some(Arc::clone(support))
        }
        _ => None,
    }
}

/// Postfix emission. `depth` tracks the live stack height; `max_stack` its maximum.
fn column_expression_nodes(expr: &ValueExpr) -> Option<usize> {
    let children = match expr {
        ValueExpr::Const(_) | ValueExpr::Var(_) | ValueExpr::Generated | ValueExpr::Lag { .. } => 0,
        ValueExpr::Neg(a) | ValueExpr::Not(a) | ValueExpr::Indicator(a) => {
            column_expression_nodes(a)?
        }
        ValueExpr::Arith { lhs, rhs, .. }
        | ValueExpr::Compare { lhs, rhs, .. }
        | ValueExpr::Logic { lhs, rhs, .. } => {
            column_expression_nodes(lhs)?.checked_add(column_expression_nodes(rhs)?)?
        }
        ValueExpr::Clip { value, lower, upper } => column_expression_nodes(value)?
            .checked_add(column_expression_nodes(lower)?)?
            .checked_add(column_expression_nodes(upper)?)?,
        ValueExpr::If { condition, then_branch, else_branch } => {
            column_expression_nodes(condition)?
                .checked_add(column_expression_nodes(then_branch)?)?
                .checked_add(column_expression_nodes(else_branch)?)?
        }
        _ => return None,
    };
    children.checked_add(1)
}

fn lowered_ops(
    n_nodes: usize,
    dense_of: &impl Fn(VariableId) -> Option<DenseNodeId>,
    expr: &ValueExpr,
) -> (Arc<[Op]>, Arc<[CallbackCall]>, usize) {
    let mut ops = Vec::new();
    let mut depth = 0;
    let mut max_stack = 0;
    emit(n_nodes, dense_of, expr, &mut ops, &mut depth, &mut max_stack);
    let callbacks = expr.callbacks().into_iter().cloned().collect::<Vec<_>>();
    if !callbacks.is_empty() {
        max_stack += 1;
    }
    (ops.into(), callbacks.into(), max_stack)
}

fn emit(
    n_nodes: usize,
    dense_of: &impl Fn(VariableId) -> Option<DenseNodeId>,
    expr: &ValueExpr,
    ops: &mut Vec<Op>,
    depth: &mut usize,
    max_stack: &mut usize,
) {
    let push = |ops: &mut Vec<Op>, op: Op, depth: &mut usize, max_stack: &mut usize| {
        ops.push(op);
        *depth += 1;
        *max_stack = (*max_stack).max(*depth);
    };
    match expr {
        ValueExpr::Const(c) => push(ops, Op::Const(c.get()), depth, max_stack),
        ValueExpr::Generated => push(ops, Op::Generated, depth, max_stack),
        ValueExpr::Var(v) => {
            // Reads were resolved before emission; an unresolved one cannot reach here.
            let dense = dense_of(*v).map_or(0, DenseNodeId::as_usize);
            push(ops, Op::Read(dense), depth, max_stack);
        }
        ValueExpr::Lag { variable, lag } => {
            // The window check ran before emission, so the variable resolves.
            let dense = dense_of(*variable).map_or(0, DenseNodeId::as_usize);
            push(ops, Op::Read(history_slot(n_nodes, dense, *lag)), depth, max_stack);
        }
        ValueExpr::Neg(a) => {
            emit(n_nodes, dense_of, a, ops, depth, max_stack);
            ops.push(Op::Neg);
        }
        // A constant operand is applied as a scalar rather than materialized as a column.
        ValueExpr::Arith { op, lhs, rhs } => match (lhs.as_ref(), rhs.as_ref()) {
            (ValueExpr::Const(c), other) => {
                emit(n_nodes, dense_of, other, ops, depth, max_stack);
                ops.push(Op::ArithConst { op: *op, constant: c.get(), constant_left: true });
            }
            (other, ValueExpr::Const(c)) => {
                emit(n_nodes, dense_of, other, ops, depth, max_stack);
                ops.push(Op::ArithConst { op: *op, constant: c.get(), constant_left: false });
            }
            _ => {
                emit(n_nodes, dense_of, lhs, ops, depth, max_stack);
                emit(n_nodes, dense_of, rhs, ops, depth, max_stack);
                ops.push(Op::Arith(*op));
                *depth -= 1;
            }
        },
        ValueExpr::Clip { value, lower, upper } => {
            emit(n_nodes, dense_of, value, ops, depth, max_stack);
            emit(n_nodes, dense_of, lower, ops, depth, max_stack);
            emit(n_nodes, dense_of, upper, ops, depth, max_stack);
            ops.push(Op::Clip);
            *depth -= 2;
        }
        ValueExpr::Compare { op, lhs, rhs } => {
            emit(n_nodes, dense_of, lhs, ops, depth, max_stack);
            emit(n_nodes, dense_of, rhs, ops, depth, max_stack);
            ops.push(Op::Compare(*op));
            *depth -= 1;
        }
        ValueExpr::Not(a) => {
            emit(n_nodes, dense_of, a, ops, depth, max_stack);
            ops.push(Op::Not);
        }
        ValueExpr::Logic { op, lhs, rhs } => {
            emit(n_nodes, dense_of, lhs, ops, depth, max_stack);
            emit(n_nodes, dense_of, rhs, ops, depth, max_stack);
            ops.push(Op::Logic(*op));
            *depth -= 1;
        }
        ValueExpr::If { condition, then_branch, else_branch } => {
            emit(n_nodes, dense_of, condition, ops, depth, max_stack);
            emit(n_nodes, dense_of, then_branch, ops, depth, max_stack);
            emit(n_nodes, dense_of, else_branch, ops, depth, max_stack);
            ops.push(Op::Select);
            *depth -= 2;
        }
        // A boolean is already 0/1 in a column; the conversion is a type-level act only.
        ValueExpr::Indicator(a) => emit(n_nodes, dense_of, a, ops, depth, max_stack),
        ValueExpr::Callback(call) => {
            for arg in call.args() {
                emit(n_nodes, dense_of, arg, ops, depth, max_stack);
            }
            let k = ops.iter().filter(|op| matches!(op, Op::Callback(_))).count();
            ops.push(Op::Callback(k));
            match call.args().len() {
                0 => {
                    *depth += 1;
                    *max_stack = (*max_stack).max(*depth);
                }
                arity => *depth -= arity - 1,
            }
        }
        // `ValueExpr` is non-exhaustive across crates, so an unknown node cannot validate and
        // never produces a usable slot.
        _ => push(ops, Op::Const(f64::NAN), depth, max_stack),
    }
}

/// `a[i] = f(a[i], b[i])`, dispatched on the operator once per column so that each loop is
/// the operator's own (inlined, vectorizable) arithmetic. The function applied is
/// [`ArithOp::apply`] itself, so the result stays bit-identical to scalar evaluation.
fn arith_columns(f: ArithOp, a: &mut [f64], b: &[f64]) {
    macro_rules! each {
        ($($variant:ident),*) => {
            match f {
                $(ArithOp::$variant => {
                    for (x, y) in a.iter_mut().zip(b) {
                        *x = ArithOp::$variant.apply(*x, *y);
                    }
                })*
                // `ArithOp` is non-exhaustive across crates.
                _ => {
                    for (x, y) in a.iter_mut().zip(b) {
                        *x = f.apply(*x, *y);
                    }
                }
            }
        };
    }
    each!(Add, Sub, Mul, Div, Min, Max);
}

/// `x = f(c, x)` (`constant_left`) or `x = f(x, c)` over a column, dispatched once per
/// column like [`arith_columns`], with the same [`ArithOp::apply`].
fn arith_scalar(f: ArithOp, a: &mut [f64], c: f64, constant_left: bool) {
    macro_rules! each {
        ($($variant:ident),*) => {
            match (f, constant_left) {
                $((ArithOp::$variant, true) => {
                    for x in a.iter_mut() {
                        *x = ArithOp::$variant.apply(c, *x);
                    }
                }
                (ArithOp::$variant, false) => {
                    for x in a.iter_mut() {
                        *x = ArithOp::$variant.apply(*x, c);
                    }
                })*
                // `ArithOp` is non-exhaustive across crates.
                (_, true) => a.iter_mut().for_each(|x| *x = f.apply(c, *x)),
                (_, false) => a.iter_mut().for_each(|x| *x = f.apply(*x, c)),
            }
        };
    }
    each!(Add, Sub, Mul, Div, Min, Max);
}

/// Apply a column-local operator to the top of the work stack; returns the new height.
#[allow(
    clippy::float_cmp,
    reason = "a boolean column holds exactly 0.0 or 1.0, so truth is an exact comparison"
)]
fn step_columns(op: Op, scratch: &mut [f64], mut sp: usize, n: usize) -> usize {
    match op {
        Op::Neg => {
            for x in &mut scratch[(sp - 1) * n..sp * n] {
                *x = -*x;
            }
        }
        Op::Not => {
            for x in &mut scratch[(sp - 1) * n..sp * n] {
                *x = f64::from(u8::from(*x == 0.0));
            }
        }
        Op::Arith(f) => {
            let (head, tail) = scratch.split_at_mut((sp - 1) * n);
            arith_columns(f, &mut head[(sp - 2) * n..], &tail[..n]);
            sp -= 1;
        }
        Op::ArithConst { op, constant, constant_left } => {
            arith_scalar(op, &mut scratch[(sp - 1) * n..sp * n], constant, constant_left);
        }
        Op::Compare(f) => {
            let (head, tail) = scratch.split_at_mut((sp - 1) * n);
            let a = &mut head[(sp - 2) * n..];
            for (x, y) in a.iter_mut().zip(&tail[..n]) {
                *x = f64::from(u8::from(f.apply(*x, *y)));
            }
            sp -= 1;
        }
        Op::Logic(f) => {
            let (head, tail) = scratch.split_at_mut((sp - 1) * n);
            let a = &mut head[(sp - 2) * n..];
            for (x, y) in a.iter_mut().zip(&tail[..n]) {
                let (p, q) = (*x != 0.0, *y != 0.0);
                *x = f64::from(u8::from(match f {
                    LogicOp::And => p && q,
                    LogicOp::Or => p || q,
                    // `LogicOp` is non-exhaustive across crates; the closed set is
                    // compiled from the two connectives above.
                    _ => false,
                }));
            }
            sp -= 1;
        }
        Op::Clip => {
            let (head, tail) = scratch.split_at_mut((sp - 2) * n);
            let value = &mut head[(sp - 3) * n..];
            let (lower, upper) = tail.split_at(n);
            for ((x, lo), hi) in value.iter_mut().zip(lower).zip(&upper[..n]) {
                *x = clip_value(*x, *lo, *hi);
            }
            sp -= 2;
        }
        Op::Select => {
            let (head, tail) = scratch.split_at_mut((sp - 2) * n);
            let condition = &mut head[(sp - 3) * n..];
            let (then_branch, else_branch) = tail.split_at(n);
            for ((c, t), e) in condition.iter_mut().zip(then_branch).zip(&else_branch[..n]) {
                *c = if *c == 0.0 { *e } else { *t };
            }
            sp -= 2;
        }
        // Loads are handled by the caller, which owns the value batch.
        Op::Const(_) | Op::Generated | Op::Read(_) | Op::Callback(_) => {}
    }
    sp
}

/// Run one callback once over the whole batch: its argument columns are the top
/// `arity` stack entries, its output is validated against its declaration, and it
/// replaces them. Returns the new stack height.
fn step_callback(
    call: &CallbackCall,
    node: usize,
    scratch: &mut [f64],
    sp: usize,
    n: usize,
    work: usize,
) -> Result<usize, ModelError> {
    let arity = call.args().len();
    let base = sp - arity;
    let (stack, tail) = scratch.split_at_mut(work * n);
    let out = &mut tail[..n];
    let inputs: Vec<&[f64]> = (base..sp).map(|slot| &stack[slot * n..(slot + 1) * n]).collect();
    call.callback()
        .call(&inputs, out)
        .and_then(|()| call.spec().output.validate(out))
        .map_err(|error| refusal_callback(call, node, &error).into_error())?;
    stack[base * n..(base + 1) * n].copy_from_slice(out);
    Ok(base + 1)
}

/// Run `program` for `node` over every row of `values`, writing the node's column.
///
/// This is the single executor behind every sampler. `Generated` reads the node's own
/// column as it stands, so a transform is called after the mechanism (and any shift) has
/// filled it; an assignment never reads it. Reads see the other columns as they stand,
/// which the dependency order guarantees are already final. All operators are the
/// functions of `antecedent-core`, applied row by row, so the result is bit-identical to
/// [`ValueExpr::evaluate`] on each row. A non-finite result refuses rather than reaching
/// the draws.
///
/// `scratch` is a reusable work buffer (grown, never shrunk). `ctx` supplies the
/// cancellation token and the hard memory limit; callers without a context pass `None`.
///
/// # Errors
///
/// Shape errors, cancellation, a memory limit breach, or a non-finite result.
#[allow(
    clippy::float_cmp,
    reason = "a boolean column holds exactly 0.0 or 1.0, so truth is an exact comparison"
)]
pub fn apply_node_program(
    program: &NodeProgram,
    values: &mut ValueBatchMut<'_>,
    node: usize,
    scratch: &mut Vec<f64>,
    ctx: Option<&ExecutionContext>,
) -> Result<(), ModelError> {
    let n = values.n_rows;
    let n_nodes = values.n_nodes;
    if node >= n_nodes
        || program.reads.iter().any(|d| d.as_usize() >= n_nodes)
        || program.lag_slots.iter().any(|slot| *slot >= n_nodes)
    {
        return Err(ModelError::Shape {
            message: "intervention program refers to a node outside the value batch".into(),
        });
    }
    // A program without callbacks is evaluated in row blocks, so its stack stays in cache;
    // every operator is row-wise, so the result does not depend on the block size. A callback
    // is called once on the whole batch, so a program with one runs on every row at once.
    let block = if program.callbacks.is_empty() { n.min(PROGRAM_BLOCK_ROWS) } else { n };
    let need = program.max_stack.saturating_mul(block);
    if let Some(limit) = ctx.and_then(|c| c.memory_limit_bytes()) {
        let bytes = (need as u64).saturating_mul(8);
        if bytes > limit {
            return Err(refusal_budget(format!(
                "program for dense node {node} needs {bytes} bytes of work space over a hard \
                 limit of {limit}"
            ))
            .into_error());
        }
    }
    if scratch.len() < need {
        scratch.resize(need, 0.0);
    }
    let mut start = 0usize;
    while start < n {
        if ctx.is_some_and(|c| c.cancellation.is_cancelled()) {
            return Err(ModelError::Cancelled);
        }
        let rows = block.min(n - start);
        run_block(program, values, node, start, rows, scratch)?;
        start += rows;
    }
    Ok(())
}

/// Rows of one evaluation block of a callback-free program: 4,096 rows keep a stack of a
/// few columns within a core's L2 cache.
const PROGRAM_BLOCK_ROWS: usize = 4096;

/// [`run_block`] for a program whose stack never holds more than one column: the node's own
/// column is that slot, so the block is computed where it is stored, with no copy in or out.
/// The operators are the same functions in the same order, so the values are identical.
fn run_block_in_place(
    program: &NodeProgram,
    values: &mut ValueBatchMut<'_>,
    node: usize,
    start: usize,
    m: usize,
) -> Result<(), ModelError> {
    let n = values.n_rows;
    let at = node * n + start;
    for op in program.ops.iter() {
        match *op {
            Op::Const(c) => values.values[at..at + m].fill(c),
            // The column already holds the generated value.
            Op::Generated => {}
            Op::Read(dense) => {
                let from = dense * n + start;
                values.values.copy_within(from..from + m, at);
            }
            op => {
                step_columns(op, &mut values.values[at..at + m], 1, m);
            }
        }
    }
    let result = &values.values[at..at + m];
    let largest = result.iter().map(|v| v.to_bits() & !(1_u64 << 63)).max().unwrap_or(0);
    if largest >= f64::INFINITY.to_bits() {
        if let Some(row) = result.iter().position(|v| !v.is_finite()) {
            return Err(ModelError::Numerical {
                message: format!(
                    "intervention program for dense node {node} is not finite at row {} ({}); \
                     guard a division or restrict the clip bounds",
                    start + row,
                    result[row]
                ),
            });
        }
    }
    Ok(())
}

/// Evaluate `program` for rows `start..start + rows` and write them to the node's column.
fn run_block(
    program: &NodeProgram,
    values: &mut ValueBatchMut<'_>,
    node: usize,
    start: usize,
    rows: usize,
    scratch: &mut [f64],
) -> Result<(), ModelError> {
    let (n, m) = (values.n_rows, rows);
    let column = |dense: usize| dense * n + start..dense * n + start + m;
    if program.max_stack == 1 && program.callbacks.is_empty() {
        return run_block_in_place(program, values, node, start, m);
    }
    let mut sp = 0usize;
    for op in program.ops.iter() {
        match *op {
            Op::Const(c) => {
                scratch[sp * m..(sp + 1) * m].fill(c);
                sp += 1;
            }
            Op::Generated => {
                scratch[sp * m..(sp + 1) * m].copy_from_slice(&values.values[column(node)]);
                sp += 1;
            }
            Op::Read(dense) => {
                scratch[sp * m..(sp + 1) * m].copy_from_slice(&values.values[column(dense)]);
                sp += 1;
            }
            Op::Callback(k) => {
                sp = step_callback(
                    &program.callbacks[k],
                    node,
                    scratch,
                    sp,
                    m,
                    program.max_stack - 1,
                )?;
            }
            op => sp = step_columns(op, scratch, sp, m),
        }
    }
    let result = &scratch[..m];
    // A vectorizable check first: the largest magnitude bit pattern reaches the all-ones
    // exponent exactly when some row is infinite or NaN (a max reduction cannot stop early,
    // so it compiles to vector code); the failing row is located only when one fails.
    let largest = result.iter().map(|v| v.to_bits() & !(1_u64 << 63)).max().unwrap_or(0);
    let any_non_finite = largest >= f64::INFINITY.to_bits();
    let failed = if any_non_finite { result.iter().position(|v| !v.is_finite()) } else { None };
    if let Some(row) = failed {
        return Err(ModelError::Numerical {
            message: format!(
                "intervention program for dense node {node} is not finite at row {} ({}); \
                 guard a division or restrict the clip bounds",
                start + row,
                result[row]
            ),
        });
    }
    values.values[column(node)].copy_from_slice(result);
    Ok(())
}

/// Dense program positions are graph node indexes. A model that compiled already
/// stored them as `u32`.
#[allow(
    clippy::cast_possible_truncation,
    reason = "program node indices are bounded by the node limit, far below u32::MAX"
)]
fn dense_id(index: usize) -> u32 {
    index as u32
}

/// Evaluation order of a model once program reads are added to its parent edges.
///
/// Returns indices into `model.parent_gathers`. `None` when no program reads another
/// node (the model's own topological order already serves). Ties keep the model's order.
///
/// # Errors
///
/// A circular-dependency refusal naming the cycle.
pub(crate) fn execution_order(
    model: &CompiledCausalModel,
    programs: &[Option<NodeProgram>],
) -> Result<Option<Arc<[u32]>>, PolicyRefusal> {
    if programs.iter().flatten().all(|p| p.reads.is_empty()) {
        return Ok(None);
    }
    let n = model.n_nodes();
    let mut position = vec![0usize; n];
    for (gi, gather) in model.parent_gathers.iter().enumerate() {
        position[gather.child.as_usize()] = gi;
    }
    // `deps[node]` = nodes that must be computed before `node`.
    let mut deps: Vec<Vec<usize>> = vec![Vec::new(); n];
    for gather in model.parent_gathers.iter() {
        let child = gather.child.as_usize();
        deps[child].extend(gather.parents.iter().map(|p| p.as_usize()));
    }
    for (node, program) in programs.iter().enumerate() {
        if let Some(program) = program {
            deps[node].extend(program.reads.iter().map(|d| d.as_usize()));
        }
    }
    let mut waiting: Vec<usize> = deps.iter().map(Vec::len).collect();
    let mut dependants: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (node, ds) in deps.iter().enumerate() {
        for &d in ds {
            dependants[d].push(node);
        }
    }
    let dependency_edges: Vec<(u32, u32)> = deps
        .iter()
        .enumerate()
        .flat_map(|(node, predecessors)| {
            predecessors.iter().map(move |&predecessor| (dense_id(predecessor), dense_id(node)))
        })
        .collect();
    if antecedent_graph::error::directed_edges_contain_cycle(n, &dependency_edges) {
        let pending = nodes_in_a_cycle(n, &dependency_edges);
        return Err(refusal_circular(describe_cycle(model, &deps, &pending, programs)));
    }
    let mut ready: BinaryHeap<Reverse<(usize, usize)>> = BinaryHeap::new();
    for node in 0..n {
        if waiting[node] == 0 {
            ready.push(Reverse((position[node], node)));
        }
    }
    let mut order = Vec::with_capacity(n);
    let mut processed = vec![false; n];
    while let Some(Reverse((pos, node))) = ready.pop() {
        processed[node] = true;
        order.push(u32::try_from(pos).map_err(|_| {
            refusal_expression_invalid(
                "intervention_program.expression_invalid",
                "node count exceeds u32 capacity".into(),
            )
        })?);
        for &next in &dependants[node] {
            waiting[next] -= 1;
            if waiting[next] == 0 {
                ready.push(Reverse((position[next], next)));
            }
        }
    }
    if order.len() == n {
        return Ok(Some(order.into()));
    }
    let pending: Vec<bool> = processed.iter().map(|done| !done).collect();
    Err(refusal_circular(describe_cycle(model, &deps, &pending, programs)))
}

/// Nodes on a directed cycle. A self-loop is a cycle. Downstream nodes that are
/// not in the cycle are omitted: the refusal names the cycle, not an order.
fn nodes_in_a_cycle(n: usize, edges: &[(u32, u32)]) -> Vec<bool> {
    let mut pending = vec![false; n];
    let Ok(count) = u32::try_from(n) else {
        return pending;
    };
    let mut graph = antecedent_graph::CyclicDigraph::with_variables(count);
    for &(from, to) in edges {
        let from_id = DenseNodeId::from_raw(from);
        let to_id = DenseNodeId::from_raw(to);
        if from_id.as_usize() >= n || to_id.as_usize() >= n {
            continue;
        }
        if from == to {
            pending[from_id.as_usize()] = true;
            continue;
        }
        let inserted = graph.insert_directed(from_id, to_id);
        if let Err(antecedent_graph::error::GraphError::DuplicateEdge { .. }) = inserted {
            continue;
        }
    }
    for component in graph.condensation().components() {
        if component.len() > 1 {
            for id in component {
                pending[id.as_usize()] = true;
            }
        }
    }
    pending
}

/// Walk unmet dependencies from any unprocessed node until a node repeats.
fn describe_cycle(
    model: &CompiledCausalModel,
    deps: &[Vec<usize>],
    pending: &[bool],
    programs: &[Option<NodeProgram>],
) -> String {
    let name = |dense: usize| {
        model
            .output_layout
            .variables
            .get(dense)
            .map_or_else(|| format!("node {dense}"), std::string::ToString::to_string)
    };
    let Some(start) = pending.iter().position(|p| *p) else {
        return "a dependency cycle".into();
    };
    let mut path = vec![start];
    let mut seen = vec![usize::MAX; pending.len()];
    seen[start] = 0;
    let mut current = start;
    loop {
        let Some(&next) = deps[current].iter().find(|d| pending[**d]) else {
            return "a dependency cycle".into();
        };
        if seen[next] != usize::MAX {
            let cycle = &path[seen[next]..];
            let mut text: Vec<String> = cycle.iter().map(|d| name(*d)).collect();
            text.push(name(next));
            let carriers: Vec<String> = cycle
                .iter()
                .filter(|d| programs.get(**d).is_some_and(Option::is_some))
                .map(|d| name(*d))
                .collect();
            return format!(
                "{} (each depends on the next; policies on {})",
                text.join(" -> "),
                carriers.join(", ")
            );
        }
        seen[next] = path.len();
        path.push(next);
        current = next;
    }
}

/// An execution route a policy can be licensed for.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PolicyRoute {
    /// Forward simulation of the intervened model.
    Simulation,
    /// Counterfactual execution under abduced noise.
    CounterfactualExecution,
    /// Identification of an interventional effect from observational data.
    ObservationalIdentification,
}

/// Licensing decision for one route.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RouteDecision {
    /// The route may run the policy.
    Licensed,
    /// The route may run the policy provided a stated assumption holds; the assumption
    /// is part of the licence and is reported with every result.
    LicensedUnder {
        /// The assumption the licence rests on.
        assumption: String,
    },
    /// The route refuses the policy.
    Refused {
        /// Registered reason code.
        code: String,
        /// Namespaced detail.
        detail: String,
        /// Full refusal text.
        message: String,
    },
}

impl RouteDecision {
    /// Whether the route is licensed.
    #[must_use]
    pub fn is_licensed(&self) -> bool {
        matches!(self, Self::Licensed | Self::LicensedUnder { .. })
    }

    /// The assumption a conditional licence rests on.
    #[must_use]
    pub fn assumption(&self) -> Option<&str> {
        match self {
            Self::LicensedUnder { assumption } => Some(assumption),
            _ => None,
        }
    }

    fn from_error(error: &ModelError) -> Self {
        let message = error.to_string();
        let body = message.split_once("unsupported: ").map_or(message.as_str(), |(_, rest)| rest);
        let parsed = antecedent_core::reason_code::split_prefix(body).map(|(code, tail)| {
            (code.to_string(), tail.split(':').next().unwrap_or_default().trim().to_string())
        });
        let (code, detail) = parsed.unwrap_or_else(|| {
            (
                reason_code!("invalid_argument").to_string(),
                "intervention_program.overlay_invalid".to_string(),
            )
        });
        Self::Refused { code, detail, message }
    }
}

/// A separate licensing decision per route for one set of interventions.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PolicySupport {
    /// Forward simulation.
    pub simulation: RouteDecision,
    /// Counterfactual execution.
    pub counterfactual: RouteDecision,
    /// Observational identification.
    pub identification: RouteDecision,
}

impl PolicySupport {
    /// Decision for one route.
    #[must_use]
    pub fn decision(&self, route: PolicyRoute) -> &RouteDecision {
        match route {
            PolicyRoute::Simulation => &self.simulation,
            PolicyRoute::CounterfactualExecution => &self.counterfactual,
            PolicyRoute::ObservationalIdentification => &self.identification,
        }
    }
}

/// License a set of interventions for each execution route.
///
/// *Simulation* and *counterfactual execution* are licensed when the interventions
/// compile into a valid overlay: a program is deterministic in its reads, so replaying
/// abduced noise through it is as well defined as drawing fresh noise. *Observational
/// identification* is licensed only for interventions that reduce to the identification
/// layer's hard levels: a `Set`, or an `Assign` that folds to a constant. A transform of
/// the generated value depends on each unit's own noise, and an assignment from other
/// variables is a function, not a level, so both stay simulation-only.
///
/// A batch callback is licensed for simulation. Counterfactual execution replays abduced
/// noise through the callback, which is well defined only if its output depends on its
/// declared inputs alone (no hidden state, clock or randomness), so that licence is
/// conditional and says so. A callback is a function, not a level, so it is never
/// identified observationally.
#[must_use]
pub fn policy_support(
    model: &CompiledCausalModel,
    interventions: &[Intervention],
) -> PolicySupport {
    let overlay = crate::overlay::InterventionOverlay::from_interventions(model, interventions);
    let execution = match &overlay {
        Ok(_) => RouteDecision::Licensed,
        Err(error) => RouteDecision::from_error(error),
    };
    let identification = if let Err(error) = &overlay {
        RouteDecision::from_error(error)
    } else {
        identification_decision(interventions)
    };
    let counterfactual = match (&execution, callback_names(interventions)) {
        (RouteDecision::Licensed, names) if !names.is_empty() => RouteDecision::LicensedUnder {
            assumption: format!(
                "callback {} returns a function of its declared inputs alone: replaying abduced \
                 noise through it is counterfactual only if it holds no hidden state, clock or \
                 randomness (not verified)",
                names.join(", ")
            ),
        },
        _ => execution.clone(),
    };
    PolicySupport { simulation: execution, counterfactual, identification }
}

/// Names of the callbacks any intervention (or sequence step) applies.
fn callback_names(interventions: &[Intervention]) -> Vec<String> {
    let mut names = Vec::new();
    for iv in interventions {
        match iv {
            Intervention::Transform { expr, .. } | Intervention::Assign { expr, .. } => {
                names.extend(expr.callbacks().iter().map(|c| c.spec().name.clone()));
            }
            Intervention::Sequence(seq) => {
                let steps: Vec<Intervention> =
                    seq.steps.iter().map(|step| step.intervention.clone()).collect();
                names.extend(callback_names(&steps));
            }
            _ => {}
        }
    }
    names.sort();
    names.dedup();
    names
}

fn identification_decision(interventions: &[Intervention]) -> RouteDecision {
    for iv in interventions {
        let reducible = match iv {
            Intervention::Set { .. } => true,
            Intervention::Assign { expr, .. } => expr.as_constant().is_some(),
            _ => false,
        };
        if !reducible {
            let variable = iv
                .primary_variable()
                .map_or_else(|| "a multi-variable intervention".to_string(), |v| v.to_string());
            let refusal = refusal_route(format!(
                "the intervention on {variable} is not a hard level, so it has no observational \
                 identification here"
            ));
            return RouteDecision::from_error(&refusal.into_error());
        }
    }
    RouteDecision::Licensed
}

#[cfg(test)]
mod column_expression_tests {
    use super::*;

    #[test]
    fn explicit_column_bindings_reuse_full_portable_expression_semantics() {
        let a = VariableId::from_raw(7);
        let b = VariableId::from_raw(2);
        let c = VariableId::from_raw(19);
        let product = ValueExpr::var(a).mul(ValueExpr::var(b)).mul(ValueExpr::var(c));
        let expression = ValueExpr::var(a).gt(ValueExpr::constant(0.25)).if_then_else(
            product.clip(ValueExpr::constant(0.0), ValueExpr::constant(10.0)),
            ValueExpr::constant(-1.0),
        );
        let program = NodeProgram::compile_columns(&[c, a, b], &expression).unwrap();
        let mut values = Vec::new();
        let mut scratch = Vec::new();
        for row in [[3.0, 2.0, 4.0], [3.0, 0.1, 4.0], [-2.0, 2.0, 4.0]] {
            let reference = expression
                .evaluate(None, &|variable| {
                    [c, a, b].iter().position(|&v| v == variable).map(|index| row[index])
                })
                .unwrap();
            assert_eq!(
                program
                    .evaluate_columns_with_scratch(&row, &mut values, &mut scratch, None)
                    .unwrap()
                    .to_bits(),
                reference.to_bits()
            );
        }
        let reordered = NodeProgram::compile_columns(&[a, b, c], &expression).unwrap();
        assert_ne!(program, reordered);
    }

    #[test]
    fn column_program_refuses_unknown_generated_history_and_bad_runtime_axes() {
        let a = VariableId::from_raw(7);
        let unknown = VariableId::from_raw(8);
        assert!(NodeProgram::compile_columns(&[a], &ValueExpr::var(unknown)).is_err());
        assert!(NodeProgram::compile_columns(&[a, a], &ValueExpr::var(a)).is_err());
        assert!(NodeProgram::compile_columns(&[a], &ValueExpr::Generated).is_err());
        assert!(NodeProgram::compile_columns(&[a], &ValueExpr::lag(a, 1)).is_err());
        let program = NodeProgram::compile_columns(&[a], &ValueExpr::var(a)).unwrap();
        assert!(program.evaluate_columns(&[], None).is_err());
        assert!(program.evaluate_columns(&[f64::NAN], None).is_err());
        let mut ctx = ExecutionContext::for_tests(1);
        ctx.set_memory_limit(Some(1));
        assert!(program.evaluate_columns(&[1.0], Some(&ctx)).is_err());
        ctx.set_memory_limit(None);
        ctx.cancellation.cancel();
        assert!(matches!(program.evaluate_columns(&[1.0], Some(&ctx)), Err(ModelError::Cancelled)));
    }
}
