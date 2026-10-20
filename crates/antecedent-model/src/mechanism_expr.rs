// SPDX-License-Identifier: MIT OR Apache-2.0
//! Serializable compositional mechanisms and differentiable execution tapes.
//! Reverse propagation through value/input-gradient pairs supplies mixed
//! derivatives for the equilibrium Jacobian likelihood without finite-difference
//! parameter scores. Bounds are domain-wide on R^n, not sampled derivatives.
use crate::nonlinear_equilibrium::EquilibriumError;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Portable scalar structural mechanism. Parameters are local to a stable equation id.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum MechanismExpression {
    /// An exact represented finite constant.
    Constant {
        /// Constant value.
        value: f64,
    },
    /// Read an endogenous coordinate, including within a feedback block.
    Variable {
        /// Dense endogenous coordinate.
        variable: usize,
    },
    /// Read a fitted local mechanism parameter.
    Parameter {
        /// Local parameter coordinate.
        parameter: usize,
    },
    /// Add one or more scalar terms.
    Add {
        /// Ordered additive terms.
        terms: Vec<Self>,
    },
    /// Multiply one or more scalar factors.
    Multiply {
        /// Ordered multiplicative factors.
        factors: Vec<Self>,
    },
    /// Smooth bounded saturation.
    Tanh {
        /// Scalar argument.
        argument: Box<Self>,
    },
    /// Smooth non-monotone response.
    Sin {
        /// Scalar argument.
        argument: Box<Self>,
    },
    /// Stable smooth rectifier, log(1+exp(x)), with derivative bounded by one.
    Softplus {
        /// Scalar argument.
        argument: Box<Self>,
    },
}
#[derive(Clone, Copy, Debug)]
enum Op {
    Constant(f64),
    Variable(usize),
    Parameter(usize),
    Add(usize, usize),
    Multiply(usize, usize),
    Tanh(usize),
    Sin(usize),
    Softplus(usize),
}
/// Checked, bounded postfix execution tape shared by values, derivatives and scores.
#[derive(Clone, Debug)]
pub struct MechanismTape {
    nodes: Vec<Op>,
    reads: Vec<usize>,
    variables: usize,
    parameters: usize,
}
/// Values and forward endogenous derivatives retained for a reverse score pass.
#[derive(Clone, Debug)]
pub struct MechanismEvaluation {
    values: Vec<f64>,
    gradients: Vec<f64>,
    variables: usize,
}
fn invalid(message: &str) -> EquilibriumError {
    EquilibriumError::Invalid(message.into())
}
fn push(nodes: &mut Vec<Op>, op: Op) -> Result<usize, EquilibriumError> {
    if nodes.len() >= 4096 {
        return Err(invalid("mechanism expression node limit"));
    }
    let index = nodes.len();
    nodes.push(op);
    Ok(index)
}
fn compile(
    expr: &MechanismExpression,
    n: usize,
    p: usize,
    depth: usize,
    nodes: &mut Vec<Op>,
    reads: &mut BTreeSet<usize>,
) -> Result<usize, EquilibriumError> {
    if depth > 48 {
        return Err(invalid("mechanism expression depth limit"));
    }
    match expr {
        MechanismExpression::Constant { value } => {
            if !value.is_finite() {
                return Err(invalid("nonfinite expression constant"));
            }
            push(nodes, Op::Constant(*value))
        }
        MechanismExpression::Variable { variable } => {
            if *variable >= n {
                return Err(invalid("expression variable binding"));
            }
            reads.insert(*variable);
            push(nodes, Op::Variable(*variable))
        }
        MechanismExpression::Parameter { parameter } => {
            if *parameter >= p {
                return Err(invalid("expression parameter binding"));
            }
            push(nodes, Op::Parameter(*parameter))
        }
        MechanismExpression::Add { terms } | MechanismExpression::Multiply { factors: terms } => {
            if terms.is_empty() || terms.len() > 4096 {
                return Err(invalid("nonempty bounded expression arguments required"));
            }
            let product = matches!(expr, MechanismExpression::Multiply { .. });
            let mut iter = terms.iter();
            let first = iter.next().expect("checked nonempty");
            let mut result = compile(first, n, p, depth + 1, nodes, reads)?;
            for term in iter {
                let rhs = compile(term, n, p, depth + 1, nodes, reads)?;
                result = push(
                    nodes,
                    if product { Op::Multiply(result, rhs) } else { Op::Add(result, rhs) },
                )?;
            }
            Ok(result)
        }
        MechanismExpression::Tanh { argument }
        | MechanismExpression::Sin { argument }
        | MechanismExpression::Softplus { argument } => {
            let argument = compile(argument, n, p, depth + 1, nodes, reads)?;
            push(
                nodes,
                match expr {
                    MechanismExpression::Tanh { .. } => Op::Tanh(argument),
                    MechanismExpression::Sin { .. } => Op::Sin(argument),
                    _ => Op::Softplus(argument),
                },
            )
        }
    }
}
fn softplus(x: f64) -> f64 {
    if x > 0.0 { x + (-x).exp().ln_1p() } else { x.exp().ln_1p() }
}
fn logistic(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}
impl MechanismExpression {
    /// Compile a bounded expression after checking all variable/parameter reads.
    pub fn compile(
        &self,
        variables: usize,
        parameters: usize,
    ) -> Result<MechanismTape, EquilibriumError> {
        if variables == 0 || variables > 128 || parameters > 4096 {
            return Err(invalid("mechanism expression dimensions"));
        }
        let mut nodes = Vec::new();
        let mut reads = BTreeSet::new();
        compile(self, variables, parameters, 0, &mut nodes, &mut reads)?;
        Ok(MechanismTape { nodes, reads: reads.into_iter().collect(), variables, parameters })
    }
}
impl MechanismTape {
    /// Original endogenous coordinates actually read by this expression.
    #[must_use]
    pub fn reads(&self) -> &[usize] {
        &self.reads
    }
    /// Admitted postfix node count.
    #[must_use]
    pub fn nodes(&self) -> usize {
        self.nodes.len()
    }
    /// Evaluate value and exact analytic input derivatives in one forward pass.
    pub fn evaluate(
        &self,
        state: &[f64],
        parameters: &[f64],
    ) -> Result<MechanismEvaluation, EquilibriumError> {
        if state.len() != self.variables
            || parameters.len() != self.parameters
            || state.iter().chain(parameters).any(|v| !v.is_finite())
        {
            return Err(invalid("expression evaluation bindings"));
        }
        let n = self.variables;
        let mut values = vec![0.0; self.nodes.len()];
        let mut gradients = vec![0.0; self.nodes.len() * n];
        for (i, op) in self.nodes.iter().enumerate() {
            values[i] = match *op {
                Op::Constant(v) => v,
                Op::Variable(v) => state[v],
                Op::Parameter(p) => parameters[p],
                Op::Add(a, b) => values[a] + values[b],
                Op::Multiply(a, b) => values[a] * values[b],
                Op::Tanh(a) => values[a].tanh(),
                Op::Sin(a) => values[a].sin(),
                Op::Softplus(a) => softplus(values[a]),
            };
            for j in 0..n {
                gradients[i * n + j] = match *op {
                    Op::Constant(_) | Op::Parameter(_) => 0.0,
                    Op::Variable(v) => {
                        if v == j {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    Op::Add(a, b) => gradients[a * n + j] + gradients[b * n + j],
                    Op::Multiply(a, b) => {
                        gradients[a * n + j] * values[b] + values[a] * gradients[b * n + j]
                    }
                    Op::Tanh(a) => (1.0 - values[i] * values[i]) * gradients[a * n + j],
                    Op::Sin(a) => values[a].cos() * gradients[a * n + j],
                    Op::Softplus(a) => logistic(values[a]) * gradients[a * n + j],
                };
            }
        }
        if values.iter().chain(&gradients).any(|v| !v.is_finite()) {
            return Err(invalid("nonfinite mechanism expression/derivative"));
        }
        Ok(MechanismEvaluation { values, gradients, variables: n })
    }
    /// Differentiate `value_adjoint*f + sum_j gradient_adjoint[j]*df/dx_j`
    /// with respect to local parameters. This includes the mixed derivatives
    /// needed by log|det(I-J)| rather than omitting that likelihood term.
    pub fn parameter_score(
        &self,
        evaluation: &MechanismEvaluation,
        value_adjoint: f64,
        gradient_adjoint: &[f64],
    ) -> Result<Vec<f64>, EquilibriumError> {
        let n = self.variables;
        let count = self.nodes.len();
        if evaluation.variables != n
            || evaluation.values.len() != count
            || gradient_adjoint.len() != n
            || !value_adjoint.is_finite()
            || gradient_adjoint.iter().any(|v| !v.is_finite())
        {
            return Err(invalid("expression score dimensions/adjoints"));
        }
        let mut value_score = vec![0.0; count];
        let mut gradient_score = vec![0.0; count * n];
        let mut parameters = vec![0.0; self.parameters];
        value_score[count - 1] = value_adjoint;
        gradient_score[(count - 1) * n..count * n].copy_from_slice(gradient_adjoint);
        for (i, operation) in self.nodes.iter().copied().enumerate().rev() {
            match operation {
                Op::Constant(_) | Op::Variable(_) => {}
                Op::Parameter(p) => parameters[p] += value_score[i],
                Op::Add(a, b) => {
                    value_score[a] += value_score[i];
                    value_score[b] += value_score[i];
                    for j in 0..n {
                        gradient_score[a * n + j] += gradient_score[i * n + j];
                        gradient_score[b * n + j] += gradient_score[i * n + j];
                    }
                }
                Op::Multiply(a, b) => {
                    value_score[a] += value_score[i] * evaluation.values[b];
                    value_score[b] += value_score[i] * evaluation.values[a];
                    for j in 0..n {
                        let adj = gradient_score[i * n + j];
                        value_score[a] += adj * evaluation.gradients[b * n + j];
                        value_score[b] += adj * evaluation.gradients[a * n + j];
                        gradient_score[a * n + j] += adj * evaluation.values[b];
                        gradient_score[b * n + j] += adj * evaluation.values[a];
                    }
                }
                op @ (Op::Tanh(_) | Op::Sin(_) | Op::Softplus(_)) => {
                    let (a, first, second) = match op {
                        Op::Tanh(a) => {
                            let d = 1.0 - evaluation.values[i].powi(2);
                            (a, d, -2.0 * evaluation.values[i] * d)
                        }
                        Op::Sin(a) => (a, evaluation.values[a].cos(), -evaluation.values[i]),
                        Op::Softplus(a) => {
                            let d = logistic(evaluation.values[a]);
                            (a, d, d * (1.0 - d))
                        }
                        _ => unreachable!(),
                    };
                    value_score[a] += value_score[i] * first;
                    for j in 0..n {
                        let adj = gradient_score[i * n + j];
                        value_score[a] += second * adj * evaluation.gradients[a * n + j];
                        gradient_score[a * n + j] += first * adj;
                    }
                }
            }
        }
        if parameters.iter().any(|v| !v.is_finite()) {
            return Err(invalid("nonfinite expression score"));
        }
        Ok(parameters)
    }
    /// Conservative domain-wide absolute derivative bounds on all of R^n.
    /// Infinity means that this sufficient certificate cannot license a density.
    pub fn derivative_bounds(&self, parameters: &[f64]) -> Result<Vec<f64>, EquilibriumError> {
        if parameters.len() != self.parameters || parameters.iter().any(|v| !v.is_finite()) {
            return Err(invalid("expression bound parameters"));
        }
        let n = self.variables;
        let mut magnitude = vec![0.0; self.nodes.len()];
        let mut gradients = vec![0.0; self.nodes.len() * n];
        for (i, op) in self.nodes.iter().enumerate() {
            magnitude[i] = match *op {
                Op::Constant(v) => v.abs(),
                Op::Parameter(p) => parameters[p].abs(),
                Op::Variable(_) | Op::Softplus(_) => f64::INFINITY,
                Op::Tanh(_) | Op::Sin(_) => 1.0,
                Op::Add(a, b) => bound_add(magnitude[a], magnitude[b]),
                Op::Multiply(a, b) => bound_mul(magnitude[a], magnitude[b]),
            };
            for j in 0..n {
                gradients[i * n + j] = match *op {
                    Op::Constant(_) | Op::Parameter(_) => 0.0,
                    Op::Variable(v) => {
                        if v == j {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    Op::Add(a, b) => bound_add(gradients[a * n + j], gradients[b * n + j]),
                    Op::Multiply(a, b) => bound_add(
                        bound_mul(gradients[a * n + j], magnitude[b]),
                        bound_mul(magnitude[a], gradients[b * n + j]),
                    ),
                    Op::Tanh(a) | Op::Sin(a) | Op::Softplus(a) => gradients[a * n + j],
                };
            }
        }
        Ok(gradients[(self.nodes.len() - 1) * n..].to_vec())
    }
}
fn bound_add(a: f64, b: f64) -> f64 {
    if a == 0.0 {
        b
    } else if b == 0.0 {
        a
    } else {
        next_up(a + b)
    }
}
fn bound_mul(a: f64, b: f64) -> f64 {
    if a == 0.0 || b == 0.0 { 0.0 } else { next_up(a * b) }
}
fn next_up(value: f64) -> f64 {
    if value.is_infinite() { value } else { f64::from_bits(value.to_bits() + 1) }
}
impl MechanismEvaluation {
    /// Final mechanism value.
    #[must_use]
    pub fn value(&self) -> f64 {
        *self.values.last().expect("compiled nonempty tape")
    }
    /// Derivatives with respect to endogenous coordinates.
    #[must_use]
    pub fn gradient(&self) -> &[f64] {
        &self.gradients[(self.values.len() - 1) * self.variables..]
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn parameter(p: usize) -> MechanismExpression {
        MechanismExpression::Parameter { parameter: p }
    }
    fn variable(v: usize) -> MechanismExpression {
        MechanismExpression::Variable { variable: v }
    }
    fn product(a: MechanismExpression, b: MechanismExpression) -> MechanismExpression {
        MechanismExpression::Multiply { factors: vec![a, b] }
    }
    #[test]
    fn compositional_input_and_mixed_parameter_derivatives_match_independent_differences() {
        let expression = product(
            MechanismExpression::Add {
                terms: vec![
                    product(parameter(0), variable(0)),
                    MechanismExpression::Tanh {
                        argument: Box::new(product(parameter(1), variable(1))),
                    },
                ],
            },
            MechanismExpression::Sin { argument: Box::new(variable(0)) },
        );
        let tape = expression.compile(2, 2).unwrap();
        let state = [0.4, -0.8];
        let parameters = [0.3, 0.5];
        let evaluated = tape.evaluate(&state, &parameters).unwrap();
        let h = 1e-5;
        for j in 0..2 {
            let mut plus = state;
            let mut minus = state;
            plus[j] += h;
            minus[j] -= h;
            let difference = (tape.evaluate(&plus, &parameters).unwrap().value()
                - tape.evaluate(&minus, &parameters).unwrap().value())
                / (2.0 * h);
            assert!((difference - evaluated.gradient()[j]).abs() < 1e-9);
        }
        let score = tape.parameter_score(&evaluated, 0.6, &[0.2, -0.4]).unwrap();
        let objective = |p: &[f64]| {
            let e = tape.evaluate(&state, p).unwrap();
            0.6 * e.value() + 0.2 * e.gradient()[0] - 0.4 * e.gradient()[1]
        };
        for j in 0..2 {
            let mut plus = parameters;
            let mut minus = parameters;
            plus[j] += h;
            minus[j] -= h;
            assert!((score[j] - (objective(&plus) - objective(&minus)) / (2.0 * h)).abs() < 1e-9);
        }
    }
    #[test]
    fn global_bounds_do_not_turn_sampled_derivatives_into_certificates() {
        let expression = product(
            parameter(0),
            MechanismExpression::Tanh {
                argument: Box::new(MechanismExpression::Add {
                    terms: vec![
                        product(parameter(1), variable(0)),
                        product(parameter(2), variable(1)),
                    ],
                }),
            },
        );
        let bounds =
            expression.compile(2, 3).unwrap().derivative_bounds(&[0.2, 0.3, -0.6]).unwrap();
        assert!(bounds[0] >= 0.06 && bounds[0] < 0.061 && bounds[1] >= 0.12 && bounds[1] < 0.121);
        let unbounded = product(variable(0), variable(1))
            .compile(2, 0)
            .unwrap()
            .derivative_bounds(&[])
            .unwrap();
        assert!(unbounded.iter().all(|v| v.is_infinite()));
        for x in [-700.0, 700.0] {
            let e = MechanismExpression::Softplus { argument: Box::new(variable(0)) }
                .compile(1, 0)
                .unwrap()
                .evaluate(&[x], &[])
                .unwrap();
            assert!(e.value().is_finite() && e.gradient()[0] >= 0.0 && e.gradient()[0] <= 1.0);
        }
    }
}
