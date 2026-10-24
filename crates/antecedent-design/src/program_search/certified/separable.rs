//! Additive coordinate factorization proved from the actual expression tree.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{
    ArithOp, CertificateOptions, CertifiedRestriction, Deserialize, ExecutionContext,
    ObjectiveInterval, PreparedCertifiedSearch, ProgramSearchError, Serialize, ValueExpr,
    VariableId, cancelled, invalid,
};

/// Bound for one independently optimized coordinate factor.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SeparableBlockBound {
    /// Original coordinate index.
    pub axis: usize,
    /// Exact subproblem identity including its operational restrictions.
    pub identity: String,
    /// Feasible incumbent coordinate.
    pub control: f64,
    /// Outward enclosure of its evaluated contribution.
    pub incumbent: ObjectiveInterval,
    /// Valid global upper bound for this factor.
    pub upper: f64,
    /// Subdivisions actually executed in this factor.
    pub splits: usize,
}
/// Independent factor bounds, combined with outward rounding.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SeparableBound {
    /// Coordinate factors; no operational constraint connects two factors.
    pub blocks: Vec<SeparableBlockBound>,
    /// Control-independent contribution.
    pub constant: ObjectiveInterval,
    /// Global upper bound on the original objective.
    pub upper: f64,
}
fn variables(e: &ValueExpr, out: &mut std::collections::BTreeSet<VariableId>) {
    match e {
        ValueExpr::Var(v) => {
            out.insert(*v);
        }
        ValueExpr::Neg(v) => variables(v, out),
        ValueExpr::Arith { lhs, rhs, .. } => {
            variables(lhs, out);
            variables(rhs, out);
        }
        _ => {}
    }
}
fn terms(e: &ValueExpr, negative: bool, out: &mut Vec<ValueExpr>) {
    match e {
        ValueExpr::Neg(v) => terms(v, !negative, out),
        ValueExpr::Arith { op: ArithOp::Add, lhs, rhs } => {
            terms(lhs, negative, out);
            terms(rhs, negative, out);
        }
        ValueExpr::Arith { op: ArithOp::Sub, lhs, rhs } => {
            terms(lhs, negative, out);
            terms(rhs, !negative, out);
        }
        _ => out.push(if negative { ValueExpr::Neg(Box::new(e.clone())) } else { e.clone() }),
    }
}
fn sum(mut expressions: Vec<ValueExpr>) -> ValueExpr {
    if expressions.is_empty() {
        return ValueExpr::constant(0.0);
    }
    while expressions.len() > 1 {
        let mut next = Vec::with_capacity(expressions.len().div_ceil(2));
        let mut iter = expressions.into_iter();
        while let Some(lhs) = iter.next() {
            next.push(if let Some(rhs) = iter.next() { lhs.add(rhs) } else { lhs });
        }
        expressions = next;
    }
    expressions.pop().unwrap_or_else(|| ValueExpr::constant(0.0))
}
impl PreparedCertifiedSearch {
    pub(super) fn separable_bound(
        &self,
        point: &mut Vec<f64>,
        options: CertificateOptions,
        ctx: &ExecutionContext,
    ) -> Result<Option<SeparableBound>, ProgramSearchError> {
        if self.controls.len() < 2
            || self
                .restrictions
                .iter()
                .any(|r| r.coefficients.iter().filter(|c| **c != 0.0).count() > 1)
        {
            return Ok(None);
        }
        let mut expanded = Vec::new();
        terms(&self.expression, false, &mut expanded);
        let mut factors = vec![Vec::new(); self.controls.len()];
        let mut constant = Vec::new();
        for term in expanded {
            let mut read = std::collections::BTreeSet::new();
            variables(&term, &mut read);
            if read.len() > 1 {
                return Ok(None);
            }
            if let Some(variable) = read.first() {
                let axis = self
                    .controls
                    .iter()
                    .position(|c| c.variable == *variable)
                    .ok_or_else(|| invalid("unbound separable coordinate"))?;
                factors[axis].push(term);
            } else {
                constant.push(term);
            }
        }
        let constant = self.enclose(&sum(constant), &self.root(), 0, &mut 0)?;
        let mut upper = constant;
        let mut candidate = point.clone();
        let mut blocks = Vec::new();
        for (axis, expression) in factors.into_iter().enumerate() {
            cancelled(ctx)?;
            let restrictions = self
                .restrictions
                .iter()
                .filter(|r| r.coefficients[axis] != 0.0)
                .map(|r| CertifiedRestriction {
                    coefficients: vec![r.coefficients[axis]],
                    bound: r.bound,
                })
                .collect();
            let plan =
                Self::compile(sum(expression), vec![self.controls[axis].clone()], restrictions)?;
            let result = plan.execute(
                vec![point[axis]],
                CertificateOptions {
                    absolute_gap: options.absolute_gap / (self.controls.len() as f64),
                    max_splits: options.max_splits / self.controls.len(),
                },
                ctx,
            )?;
            candidate[axis] = result.controls[0];
            upper = upper.op(ArithOp::Add, ObjectiveInterval::exact(result.global_upper))?;
            blocks.push(SeparableBlockBound {
                axis,
                identity: result.identity.to_hex(),
                control: result.controls[0],
                incumbent: result.incumbent,
                upper: result.global_upper,
                splits: result.splits,
            });
        }
        // Original outward feasibility is authoritative even when the algebraic
        // operational factorization appears independent.
        if !self.feasible(&candidate)? {
            return Ok(None);
        }
        *point = candidate;
        Ok(Some(SeparableBound { blocks, constant, upper: upper.upper }))
    }
}
#[cfg(test)]
mod tests {
    use super::super::{CertificateStop, CertifiedControl};
    use super::*;
    #[test]
    fn two_nonmonotone_factors_have_an_independently_checked_global_bound() {
        let x = ValueExpr::var(VariableId::from_raw(0));
        let y = ValueExpr::var(VariableId::from_raw(1));
        let expression = x
            .clone()
            .mul(ValueExpr::constant(1.0).sub(x))
            .add(y.clone().mul(ValueExpr::constant(1.0).sub(y)));
        let plan = PreparedCertifiedSearch::compile(
            expression,
            (0..2)
                .map(|v| CertifiedControl {
                    variable: VariableId::from_raw(v),
                    bounds: ObjectiveInterval { lower: 0.0, upper: 1.0 },
                })
                .collect(),
            vec![],
        )
        .unwrap();
        let c = plan
            .execute(
                vec![0.0, 0.0],
                CertificateOptions { absolute_gap: 1e-3, max_splits: 2000 },
                &ExecutionContext::for_tests(1),
            )
            .unwrap();
        assert_eq!(c.stop, CertificateStop::Certified);
        assert!(c.global_upper >= 0.5 && c.gap <= 1e-3);
        let proof = c.separable_bound.unwrap();
        assert_eq!(proof.blocks.len(), 2);
        assert!(proof.blocks.iter().all(|b| b.upper >= 0.25));
        assert_eq!(c.splits, proof.blocks.iter().map(|b| b.splits).sum::<usize>());
    }
}
