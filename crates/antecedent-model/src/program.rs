//! Compiled intervention programs: value expressions run column-wise over draws.
//!
//! A [`ValueExpr`] attached to a node by `Intervention::Transform` or
//! `Intervention::Assign` is validated against the compiled model and flattened to a
//! postfix program. [`apply_node_program`] is the one executor every sampler shares:
//! forward ancestral sampling, conditional interventional sampling, the do-samplers and
//! counterfactual world evaluation all call it, so a policy means the same thing on
//! every route.
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
    ArithOp, CompareOp, ExecutionContext, ExpressionFault, Intervention, LogicOp, PolicyReads,
    ValueExpr, VariableId, clip_value, reason_code,
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
    PolicyRefusal {
        code: reason_code!("policy_dependency_circular"),
        detail: "intervention_program.dependency_circular",
        subject,
        remedy: "remove a read so that no policy depends on its own output, directly or through \
                 the graph",
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
    Clip,
    Compare(CompareOp),
    Not,
    Logic(LogicOp),
    Select,
}

/// A validated, flattened program for one node.
#[derive(Clone, Debug)]
pub struct NodeProgram {
    kind: ProgramKind,
    expr: ValueExpr,
    ops: Arc<[Op]>,
    reads: Arc<[DenseNodeId]>,
    max_stack: usize,
}

impl PartialEq for NodeProgram {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind && self.expr == other.expr
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
        let verb = match kind {
            ProgramKind::Transform => "transform",
            ProgramKind::Assign => "assign",
        };
        let label = |what: &str| format!("{verb} of {target}: {what}");
        expr.validate_program(kind == ProgramKind::Transform, declared)
            .map_err(|fault| refusal_for_fault(fault, label(&fault.to_string())))?;
        if let Some((variable, lag)) = declared.history.first() {
            return Err(refusal_history_unavailable(label(&format!(
                "reads {variable} at lag {lag}, but a static model has no history window"
            ))));
        }
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
                None => {
                    expr.is_binary_valued()
                        && slot_support.contains(&0.0)
                        && slot_support.contains(&1.0)
                }
            };
            if !admitted {
                return Err(refusal_categorical_output(label(
                    "the node is categorical and the expression is not a level of its support",
                )));
            }
        }
        let mut ops = Vec::new();
        let mut depth = 0usize;
        let mut max_stack = 0usize;
        emit(model, expr, &mut ops, &mut depth, &mut max_stack);
        Ok(Self { kind, expr: expr.clone(), ops: ops.into(), reads: reads.into(), max_stack })
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
fn emit(
    model: &CompiledCausalModel,
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
            let dense = model.dense_of(*v).map_or(0, DenseNodeId::as_usize);
            push(ops, Op::Read(dense), depth, max_stack);
        }
        ValueExpr::Neg(a) => {
            emit(model, a, ops, depth, max_stack);
            ops.push(Op::Neg);
        }
        ValueExpr::Arith { op, lhs, rhs } => {
            emit(model, lhs, ops, depth, max_stack);
            emit(model, rhs, ops, depth, max_stack);
            ops.push(Op::Arith(*op));
            *depth -= 1;
        }
        ValueExpr::Clip { value, lower, upper } => {
            emit(model, value, ops, depth, max_stack);
            emit(model, lower, ops, depth, max_stack);
            emit(model, upper, ops, depth, max_stack);
            ops.push(Op::Clip);
            *depth -= 2;
        }
        ValueExpr::Compare { op, lhs, rhs } => {
            emit(model, lhs, ops, depth, max_stack);
            emit(model, rhs, ops, depth, max_stack);
            ops.push(Op::Compare(*op));
            *depth -= 1;
        }
        ValueExpr::Not(a) => {
            emit(model, a, ops, depth, max_stack);
            ops.push(Op::Not);
        }
        ValueExpr::Logic { op, lhs, rhs } => {
            emit(model, lhs, ops, depth, max_stack);
            emit(model, rhs, ops, depth, max_stack);
            ops.push(Op::Logic(*op));
            *depth -= 1;
        }
        ValueExpr::If { condition, then_branch, else_branch } => {
            emit(model, condition, ops, depth, max_stack);
            emit(model, then_branch, ops, depth, max_stack);
            emit(model, else_branch, ops, depth, max_stack);
            ops.push(Op::Select);
            *depth -= 2;
        }
        // A boolean is already 0/1 in a column; the conversion is a type-level act only.
        ValueExpr::Indicator(a) => emit(model, a, ops, depth, max_stack),
        // A history read is refused before emission, and `ValueExpr` is non-exhaustive across
        // crates, so an unknown node cannot validate; neither produces a usable slot.
        _ => push(ops, Op::Const(f64::NAN), depth, max_stack),
    }
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
            let a = &mut head[(sp - 2) * n..];
            for (x, y) in a.iter_mut().zip(&tail[..n]) {
                *x = f.apply(*x, *y);
            }
            sp -= 1;
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
        Op::Const(_) | Op::Generated | Op::Read(_) => {}
    }
    sp
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
    if node >= n_nodes || program.reads.iter().any(|d| d.as_usize() >= n_nodes) {
        return Err(ModelError::Shape {
            message: "intervention program refers to a node outside the value batch".into(),
        });
    }
    let need = program.max_stack.saturating_mul(n);
    if let Some(limit) = ctx.and_then(|c| c.memory.hard_limit_bytes) {
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
    let mut sp = 0usize;
    for op in program.ops.iter() {
        if ctx.is_some_and(|c| c.cancellation.is_cancelled()) {
            return Err(ModelError::Cancelled);
        }
        match *op {
            Op::Const(c) => {
                scratch[sp * n..(sp + 1) * n].fill(c);
                sp += 1;
            }
            Op::Generated => {
                scratch[sp * n..(sp + 1) * n]
                    .copy_from_slice(&values.values[node * n..(node + 1) * n]);
                sp += 1;
            }
            Op::Read(dense) => {
                scratch[sp * n..(sp + 1) * n]
                    .copy_from_slice(&values.values[dense * n..(dense + 1) * n]);
                sp += 1;
            }
            op => sp = step_columns(op, scratch, sp, n),
        }
    }
    let result = &scratch[..n];
    if let Some(row) = result.iter().position(|v| !v.is_finite()) {
        return Err(ModelError::Numerical {
            message: format!(
                "intervention program for dense node {node} is not finite at row {row} ({}); \
                 guard a division or restrict the clip bounds",
                result[row]
            ),
        });
    }
    values.column_mut(node)?.copy_from_slice(result);
    Ok(())
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
        matches!(self, Self::Licensed)
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
    PolicySupport { simulation: execution.clone(), counterfactual: execution, identification }
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
