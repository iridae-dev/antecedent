//! Shape proofs are computed from the expression; callers cannot assert them.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{
    ArithOp, Deserialize, ExecutionContext, ObjectiveInterval, PreparedCertifiedSearch,
    ProgramSearchError, Serialize, ValueExpr, cancelled, invalid,
};

/// A box reduction justified by an enclosed derivative and constraint signs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MonotoneReduction {
    /// Axis in the declared control order.
    pub axis: usize,
    /// Derivative enclosure over the complete original box.
    pub derivative: ObjectiveInterval,
    /// Endpoint that preserves every hard operational restriction.
    pub fixed_value: f64,
}
/// A supporting-plane Lagrangian upper bound for a proven concave objective.
/// Box support functions are evaluated with interval arithmetic. Every multiplier
/// is nonnegative, so the bound is valid without assuming optimizer convergence.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConcaveDualBound {
    /// Feasible point at which the supporting plane was formed.
    pub at: Vec<f64>,
    /// Enclosed gradient or supergradient at that point.
    pub gradient: Vec<ObjectiveInterval>,
    /// Nonnegative multipliers in original restriction order.
    pub multipliers: Vec<f64>,
    /// Upper bound on the constrained objective over the complete original box.
    pub upper: f64,
}
#[derive(Clone, Copy, PartialEq)]
enum Shape {
    Affine,
    Concave,
    Convex,
    Unknown,
}
impl Shape {
    fn negative(self) -> Self {
        match self {
            Self::Concave => Self::Convex,
            Self::Convex => Self::Concave,
            x => x,
        }
    }
    fn sum(self, rhs: Self) -> Self {
        match (self, rhs) {
            (Self::Affine, s) | (s, Self::Affine) => s,
            (Self::Concave, Self::Concave) => Self::Concave,
            (Self::Convex, Self::Convex) => Self::Convex,
            _ => Self::Unknown,
        }
    }
}
fn saturation<'a>(lhs: &'a ValueExpr, rhs: &'a ValueExpr) -> Option<f64> {
    let ValueExpr::Arith { op: ArithOp::Add, lhs: base, rhs: k } = rhs else { return None };
    let ValueExpr::Const(k) = k.as_ref() else { return None };
    (base.as_ref() == lhs).then_some(k.get())
}
fn constant(expression: &ValueExpr) -> Option<f64> {
    match expression {
        ValueExpr::Const(n) => Some(n.get()),
        ValueExpr::Neg(v) => constant(v).map(|x| -x),
        _ => None,
    }
}
impl PreparedCertifiedSearch {
    fn shape(&self, expression: &ValueExpr) -> Shape {
        match expression {
            ValueExpr::Const(_) | ValueExpr::Var(_) => Shape::Affine,
            ValueExpr::Neg(v) => self.shape(v).negative(),
            ValueExpr::Arith { op: ArithOp::Add, lhs, rhs } => self.shape(lhs).sum(self.shape(rhs)),
            ValueExpr::Arith { op: ArithOp::Sub, lhs, rhs } => {
                self.shape(lhs).sum(self.shape(rhs).negative())
            }
            ValueExpr::Arith { op: ArithOp::Mul, lhs, rhs } => {
                if let Some(c) = constant(lhs) {
                    if c >= 0.0 { self.shape(rhs) } else { self.shape(rhs).negative() }
                } else if let Some(c) = constant(rhs) {
                    if c >= 0.0 { self.shape(lhs) } else { self.shape(lhs).negative() }
                } else if lhs == rhs && self.shape(lhs) == Shape::Affine {
                    Shape::Convex
                } else {
                    Shape::Unknown
                }
            }
            ValueExpr::Arith { op: ArithOp::Div, lhs, rhs } => {
                if let Some(c) = constant(rhs) {
                    return if c > 0.0 { self.shape(lhs) } else { self.shape(lhs).negative() };
                }
                if self.shape(lhs) == Shape::Affine {
                    if let Some(k) = saturation(lhs, rhs) {
                        if self.enclose(rhs, &self.root(), 0, &mut 0).is_ok_and(|v| v.lower > 0.0) {
                            return if k > 0.0 {
                                Shape::Concave
                            } else if k < 0.0 {
                                Shape::Convex
                            } else {
                                Shape::Affine
                            };
                        }
                    }
                }
                Shape::Unknown
            }
            ValueExpr::Arith { op: ArithOp::Min, lhs, rhs } => {
                if matches!(self.shape(lhs), Shape::Affine | Shape::Concave)
                    && matches!(self.shape(rhs), Shape::Affine | Shape::Concave)
                {
                    Shape::Concave
                } else {
                    Shape::Unknown
                }
            }
            ValueExpr::Arith { op: ArithOp::Max, lhs, rhs } => {
                if matches!(self.shape(lhs), Shape::Affine | Shape::Convex)
                    && matches!(self.shape(rhs), Shape::Affine | Shape::Convex)
                {
                    Shape::Convex
                } else {
                    Shape::Unknown
                }
            }
            _ => Shape::Unknown,
        }
    }
    fn derivative(
        &self,
        expression: &ValueExpr,
        bounds: &[ObjectiveInterval],
        axis: usize,
    ) -> Result<(ObjectiveInterval, ObjectiveInterval), ProgramSearchError> {
        let zero = ObjectiveInterval::exact(0.0);
        match expression {
            ValueExpr::Const(n) => Ok((ObjectiveInterval::exact(n.get()), zero)),
            ValueExpr::Var(v) => {
                let index = self
                    .controls
                    .iter()
                    .position(|c| c.variable == *v)
                    .ok_or_else(|| invalid("unbound derivative variable"))?;
                Ok((bounds[index], ObjectiveInterval::exact(if index == axis { 1.0 } else { 0.0 })))
            }
            ValueExpr::Neg(v) => {
                let (value, d) = self.derivative(v, bounds, axis)?;
                Ok((
                    ObjectiveInterval { lower: -value.upper, upper: -value.lower },
                    ObjectiveInterval { lower: -d.upper, upper: -d.lower },
                ))
            }
            ValueExpr::Arith { op, lhs, rhs } => {
                let (a, da) = self.derivative(lhs, bounds, axis)?;
                let (b, db) = self.derivative(rhs, bounds, axis)?;
                let value = a.op(*op, b)?;
                let derivative = match op {
                    ArithOp::Add | ArithOp::Sub => da.op(*op, db)?,
                    ArithOp::Mul => {
                        da.op(ArithOp::Mul, b)?.op(ArithOp::Add, a.op(ArithOp::Mul, db)?)?
                    }
                    ArithOp::Div => {
                        let numerator = if let Some(k) = saturation(lhs, rhs) {
                            ObjectiveInterval::exact(k).op(ArithOp::Mul, da)?
                        } else {
                            da.op(ArithOp::Mul, b)?.op(ArithOp::Sub, a.op(ArithOp::Mul, db)?)?
                        };
                        numerator.op(ArithOp::Div, b.op(ArithOp::Mul, b)?)?
                    }
                    ArithOp::Min | ArithOp::Max => ObjectiveInterval {
                        lower: da.lower.min(db.lower),
                        upper: da.upper.max(db.upper),
                    },
                    _ => return Err(invalid("unsupported derivative operator")),
                };
                Ok((value, derivative))
            }
            _ => Err(invalid("unsupported derivative expression")),
        }
    }
    pub(super) fn reduce_monotone(
        &self,
        root: &mut [ObjectiveInterval],
        point: &mut Vec<f64>,
        ctx: &ExecutionContext,
    ) -> Result<Vec<MonotoneReduction>, ProgramSearchError> {
        let original = self.root();
        let mut reductions = Vec::new();
        for axis in 0..self.controls.len() {
            cancelled(ctx)?;
            let Ok((_, derivative)) = self.derivative(&self.expression, &original, axis) else {
                continue;
            };
            let down = derivative.upper <= 0.0
                && self.restrictions.iter().all(|r| r.coefficients[axis] >= 0.0);
            let up = derivative.lower >= 0.0
                && self.restrictions.iter().all(|r| r.coefficients[axis] <= 0.0);
            let value = if down {
                root[axis].lower
            } else if up {
                root[axis].upper
            } else {
                continue;
            };
            let mut candidate = point.clone();
            candidate[axis] = value;
            if !self.feasible(&candidate)? {
                continue;
            }
            *point = candidate;
            root[axis] = ObjectiveInterval::exact(value);
            reductions.push(MonotoneReduction { axis, derivative, fixed_value: value });
        }
        Ok(reductions)
    }
    fn dual_value(
        &self,
        at: &[f64],
        gradient: &[ObjectiveInterval],
        lambda: &[f64],
    ) -> Result<f64, ProgramSearchError> {
        let point = at.iter().map(|x| ObjectiveInterval::exact(*x)).collect::<Vec<_>>();
        let mut value = self.bounds(&point)?;
        for (r, multiplier) in self.restrictions.iter().zip(lambda) {
            value = value.op(
                ArithOp::Add,
                ObjectiveInterval::exact(*multiplier)
                    .op(ArithOp::Mul, ObjectiveInterval::exact(r.bound))?,
            )?;
        }
        for (axis, g) in gradient.iter().enumerate() {
            value = value.op(ArithOp::Sub, g.op(ArithOp::Mul, point[axis])?)?;
            let mut residual = *g;
            for (r, multiplier) in self.restrictions.iter().zip(lambda) {
                residual = residual.op(
                    ArithOp::Sub,
                    ObjectiveInterval::exact(*multiplier)
                        .op(ArithOp::Mul, ObjectiveInterval::exact(r.coefficients[axis]))?,
                )?;
            }
            let support = residual.op(ArithOp::Mul, self.controls[axis].bounds)?;
            value = value.op(ArithOp::Add, ObjectiveInterval::exact(support.upper))?;
        }
        Ok(value.upper)
    }
    /// Form a valid concave-objective Lagrange dual bound at a feasible point.
    /// Concavity is proved structurally from the actual expression; unsupported
    /// expressions return `None`, retaining their interval branch-and-bound path.
    ///
    /// # Errors
    /// Invalid primal point, numerical overflow or cancellation.
    pub fn concave_dual_bound(
        &self,
        at: &[f64],
        ctx: &ExecutionContext,
    ) -> Result<Option<ConcaveDualBound>, ProgramSearchError> {
        cancelled(ctx)?;
        if !self.feasible(at)? {
            return Err(invalid("dual bound requires a proved feasible primal point"));
        }
        if !matches!(self.shape(&self.expression), Shape::Affine | Shape::Concave) {
            return Ok(None);
        }
        let point = at.iter().map(|x| ObjectiveInterval::exact(*x)).collect::<Vec<_>>();
        let gradient = (0..at.len())
            .map(|axis| self.derivative(&self.expression, &point, axis).map(|(_, d)| d))
            .collect::<Result<Vec<_>, _>>()?;
        let mut multipliers = vec![0.0; self.restrictions.len()];
        let mut upper = self.dual_value(at, &gradient, &multipliers)?;
        for _ in 0..4 {
            for row in 0..multipliers.len() {
                cancelled(ctx)?;
                let mut trials = vec![0.0, multipliers[row]];
                for (axis, g) in gradient.iter().enumerate() {
                    let a = self.restrictions[row].coefficients[axis];
                    if a == 0.0 {
                        continue;
                    }
                    let residual = g.lower / 2.0 + g.upper / 2.0
                        - self
                            .restrictions
                            .iter()
                            .zip(&multipliers)
                            .enumerate()
                            .filter(|(i, _)| *i != row)
                            .map(|(_, (r, m))| r.coefficients[axis] * m)
                            .sum::<f64>();
                    let candidate = residual / a;
                    if candidate.is_finite() && candidate >= 0.0 {
                        trials.push(candidate);
                    }
                }
                let previous = multipliers[row];
                let mut best = previous;
                for trial in trials {
                    multipliers[row] = trial;
                    if let Ok(value) = self.dual_value(at, &gradient, &multipliers) {
                        if value < upper {
                            upper = value;
                            best = trial;
                        }
                    }
                }
                multipliers[row] = best;
            }
        }
        Ok(Some(ConcaveDualBound { at: at.to_vec(), gradient, multipliers, upper }))
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        CertificateOptions, CertificateStop, CertifiedControl, CertifiedRestriction, VariableId,
    };
    use super::*;
    fn saturation_expr(variable: VariableId) -> ValueExpr {
        let x = ValueExpr::var(variable);
        x.clone().div(x.add(ValueExpr::constant(1.0)))
    }
    fn controls() -> Vec<CertifiedControl> {
        (0..2)
            .map(|i| CertifiedControl {
                variable: VariableId::from_raw(i),
                bounds: ObjectiveInterval { lower: 0.0, upper: 1.0 },
            })
            .collect()
    }
    #[test]
    fn shape_proof_reduces_separate_monotone_saturating_coordinates() {
        let expr =
            saturation_expr(VariableId::from_raw(0)).add(saturation_expr(VariableId::from_raw(1)));
        let plan = PreparedCertifiedSearch::compile(expr, controls(), vec![]).unwrap();
        let result = plan
            .execute(
                vec![0.0, 0.0],
                CertificateOptions { absolute_gap: 1e-10, max_splits: 0 },
                &ExecutionContext::for_tests(11),
            )
            .unwrap();
        assert_eq!(result.stop, CertificateStop::Certified);
        assert_eq!(result.monotone_reductions.len(), 2);
        assert_eq!(result.controls, vec![1.0, 1.0]);
        assert!(result.global_upper >= 1.0);
        assert!(result.gap < 1e-10);
    }
    #[test]
    fn primal_dual_bound_certifies_a_coupled_budget_without_a_shape_flag() {
        let expr =
            saturation_expr(VariableId::from_raw(0)).add(saturation_expr(VariableId::from_raw(1)));
        let plan = PreparedCertifiedSearch::compile(
            expr,
            controls(),
            vec![CertifiedRestriction { coefficients: vec![1.0, 1.0], bound: 1.0 }],
        )
        .unwrap();
        let result = plan
            .execute(
                vec![0.499_999_999, 0.499_999_999],
                CertificateOptions { absolute_gap: 1e-7, max_splits: 0 },
                &ExecutionContext::for_tests(11),
            )
            .unwrap();
        assert_eq!(result.stop, CertificateStop::Certified);
        let dual = result.dual_bound.as_ref().unwrap();
        assert!(dual.multipliers.iter().all(|lambda| *lambda >= 0.0));
        assert!(dual.upper >= 2.0 / 3.0);
        assert!(result.global_upper >= 2.0 / 3.0);
        assert!(result.gap < 1e-7);
        assert!(result.monotone_reductions.is_empty());
    }
    #[test]
    fn positive_product_is_not_mislabeled_concave() {
        let plan = PreparedCertifiedSearch::compile(
            ValueExpr::var(VariableId::from_raw(0)).mul(ValueExpr::var(VariableId::from_raw(1))),
            controls(),
            vec![],
        )
        .unwrap();
        assert!(
            plan.concave_dual_bound(&[0.5, 0.5], &ExecutionContext::for_tests(11))
                .unwrap()
                .is_none()
        );
    }
}
