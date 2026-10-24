//! Global bounds derived from expressions, with outward-rounded arithmetic.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{ProgramSearchError, cancelled, invalid};
use antecedent_core::{
    ArithOp, CanonicalEncode, CanonicalWriter, ExecutionContext, SemanticDigest, ValueExpr,
    VariableId,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub mod artifact;
mod separable;
mod shape;
pub use separable::{SeparableBlockBound, SeparableBound};
pub use shape::{ConcaveDualBound, MonotoneReduction};

/// Closed interval enclosing every represented value.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObjectiveInterval {
    /// Lower bound.
    pub lower: f64,
    /// Upper bound.
    pub upper: f64,
}
fn adjacent(x: f64, up: bool) -> f64 {
    if x.is_nan() || (up && x == f64::INFINITY) || (!up && x == f64::NEG_INFINITY) {
        return x;
    }
    if x == 0.0 {
        return if up { f64::from_bits(1) } else { -f64::from_bits(1) };
    }
    f64::from_bits(if (x > 0.0) == up { x.to_bits() + 1 } else { x.to_bits() - 1 })
}
fn objective_gap(upper: f64, lower: f64) -> f64 {
    if upper <= lower { 0.0 } else { adjacent(upper - lower, true) }
}
impl ObjectiveInterval {
    fn exact(value: f64) -> Self {
        Self { lower: value, upper: value }
    }
    fn rounded(lower: f64, upper: f64) -> Result<Self, ProgramSearchError> {
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return Err(invalid("nonfinite or inverted objective enclosure"));
        }
        let interval = Self { lower: adjacent(lower, false), upper: adjacent(upper, true) };
        if !interval.lower.is_finite() || !interval.upper.is_finite() {
            return Err(invalid("outward enclosure exceeds finite floating-point range"));
        }
        Ok(interval)
    }
    fn op(self, op: ArithOp, rhs: Self) -> Result<Self, ProgramSearchError> {
        // These identities are exact, including zero products. Keeping them exact
        // also permits proving feasibility of a genuinely zero-cost boundary.
        if matches!(op, ArithOp::Mul) && (self == Self::exact(0.0) || rhs == Self::exact(0.0)) {
            return Ok(Self::exact(0.0));
        }
        if matches!(op, ArithOp::Add) && self == Self::exact(0.0) {
            return Ok(rhs);
        }
        if matches!(op, ArithOp::Add | ArithOp::Sub) && rhs == Self::exact(0.0) {
            return Ok(self);
        }
        match op {
            ArithOp::Add => Self::rounded(self.lower + rhs.lower, self.upper + rhs.upper),
            ArithOp::Sub => Self::rounded(self.lower - rhs.upper, self.upper - rhs.lower),
            ArithOp::Mul => {
                let values = [
                    self.lower * rhs.lower,
                    self.lower * rhs.upper,
                    self.upper * rhs.lower,
                    self.upper * rhs.upper,
                ];
                Self::rounded(
                    values.into_iter().fold(f64::INFINITY, f64::min),
                    values.into_iter().fold(f64::NEG_INFINITY, f64::max),
                )
            }
            ArithOp::Div => {
                if rhs.lower <= 0.0 && rhs.upper >= 0.0 {
                    return Err(invalid("objective denominator interval contains zero"));
                }
                self.op(ArithOp::Mul, Self::rounded(1.0 / rhs.upper, 1.0 / rhs.lower)?)
            }
            ArithOp::Min => Self::rounded(self.lower.min(rhs.lower), self.upper.min(rhs.upper)),
            ArithOp::Max => Self::rounded(self.lower.max(rhs.lower), self.upper.max(rhs.upper)),
            _ => Err(invalid("unsupported objective arithmetic")),
        }
    }
}

/// One bounded real control. Scheduling is bound by the producing compiler,
/// which assigns distinct expression variables to distinct scheduled controls.
#[derive(Clone, Debug, PartialEq)]
pub struct CertifiedControl {
    /// Unique expression variable.
    pub variable: VariableId,
    /// Closed finite bounds, possibly a fixed value.
    pub bounds: ObjectiveInterval,
}
/// Linear hard constraint `dot(coefficients, controls) <= bound`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CertifiedRestriction {
    /// Coefficients in control order.
    pub coefficients: Vec<f64>,
    /// Finite upper bound.
    pub bound: f64,
}
/// Explicit finite work and optimality tolerance.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CertificateOptions {
    /// Absolute objective gap required for certification.
    pub absolute_gap: f64,
    /// Maximum number of region subdivisions, zero permits only the root bound.
    pub max_splits: usize,
}
/// Actual termination of global bounded search.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CertificateStop {
    /// Global bound and feasible incumbent differ by at most the requested tolerance.
    Certified,
    /// Valid global bound retained after the subdivision budget was exhausted.
    SplitLimit,
    /// No floating-point subdivision remained; the remaining bound is still retained.
    ResolutionLimit,
}
/// A live region and its conservative objective enclosure.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CertifiedRegion {
    /// Control intervals in declared order.
    pub bounds: Vec<ObjectiveInterval>,
    /// Maximum objective bound over this region, irrespective of feasibility.
    pub upper: f64,
}
/// A numerical global certificate for the compiled objective, not a statistical license.
#[derive(Clone, Debug)]
pub struct GlobalProgramCertificate {
    /// Objective/domain/restriction identity.
    pub identity: SemanticDigest,
    /// Feasible controls actually evaluated.
    pub controls: Vec<f64>,
    /// Outward-rounded objective value at the incumbent.
    pub incumbent: ObjectiveInterval,
    /// Upper bound on the true global optimum of this numerical objective.
    pub global_upper: f64,
    /// Conservative upper bound on suboptimality.
    pub gap: f64,
    /// Why search stopped.
    pub stop: CertificateStop,
    /// Unresolved regions retained for independent verification.
    pub frontier: Vec<CertifiedRegion>,
    /// Number of subdivisions actually executed.
    pub splits: usize,
    /// Regions excluded by outward-rounded operational feasibility bounds.
    pub infeasible_regions: usize,
    /// Regions excluded because they cannot improve the feasible incumbent.
    pub dominated_regions: usize,
    /// Compiler-derived derivative proofs used to reduce the original box.
    pub monotone_reductions: Vec<MonotoneReduction>,
    /// Feasible nonnegative Lagrange multipliers and their concave-objective bound.
    pub dual_bound: Option<ConcaveDualBound>,
    /// Independent coordinate bounds when the expression and restrictions separate.
    pub separable_bound: Option<SeparableBound>,
    retention: Arc<antecedent_core::execution::AdmissionLease>,
}
impl GlobalProgramCertificate {
    /// Conservative retained storage admission, not process RSS.
    #[must_use]
    pub fn admitted_bytes(&self) -> u64 {
        self.retention.remainder().bytes()
    }
}

/// A compiled scalar objective supporting addition, subtraction, multiplication,
/// division with a separated denominator, negation, min and max. No callback or
/// caller-supplied shape declaration can create a certificate.
#[derive(Clone, Debug)]
pub struct PreparedCertifiedSearch {
    expression: ValueExpr,
    controls: Vec<CertifiedControl>,
    restrictions: Vec<CertifiedRestriction>,
    identity: SemanticDigest,
}
impl PreparedCertifiedSearch {
    /// Bind an actual expression and its complete bounded operational domain.
    ///
    /// # Errors
    /// Invalid domains/restrictions, foreign variables, unsupported operators,
    /// excessive expression size/depth and singular denominator enclosures.
    pub fn compile(
        expression: ValueExpr,
        controls: Vec<CertifiedControl>,
        restrictions: Vec<CertifiedRestriction>,
    ) -> Result<Self, ProgramSearchError> {
        if controls.is_empty() || controls.len() > 4096 {
            return Err(invalid("certificate requires 1..4096 bounded controls"));
        }
        let mut writer = CanonicalWriter::new("antecedent.certified_objective.v1");
        let mut seen = std::collections::HashSet::new();
        for control in &controls {
            let b = control.bounds;
            if !seen.insert(control.variable)
                || !b.lower.is_finite()
                || !b.upper.is_finite()
                || b.lower > b.upper
            {
                return Err(invalid("invalid or duplicate certified control"));
            }
            writer
                .word(u64::from(control.variable.raw()))
                .word(b.lower.to_bits())
                .word(b.upper.to_bits());
        }
        writer.word(restrictions.len() as u64);
        for restriction in &restrictions {
            if restriction.coefficients.len() != controls.len()
                || !restriction.bound.is_finite()
                || restriction.coefficients.iter().any(|x| !x.is_finite())
            {
                return Err(invalid("invalid certified linear restriction"));
            }
            for coefficient in &restriction.coefficients {
                writer.word(coefficient.to_bits());
            }
            writer.word(restriction.bound.to_bits());
        }
        let mut plan = Self {
            expression,
            controls,
            restrictions,
            identity: SemanticDigest::from_bytes([0; 32]),
        };
        let root = plan.root();
        let mut nodes = 0;
        plan.enclose(&plan.expression, &root, 0, &mut nodes)?;
        plan.expression.encode(&mut writer);
        plan.identity = SemanticDigest::from_bytes(writer.finish());
        Ok(plan)
    }
    fn root(&self) -> Vec<ObjectiveInterval> {
        self.controls.iter().map(|c| c.bounds).collect()
    }
    fn enclose(
        &self,
        expression: &ValueExpr,
        bounds: &[ObjectiveInterval],
        depth: usize,
        nodes: &mut usize,
    ) -> Result<ObjectiveInterval, ProgramSearchError> {
        *nodes += 1;
        if depth > 64 || *nodes > 65_536 {
            return Err(invalid("certified expression size/depth limit"));
        }
        match expression {
            ValueExpr::Const(n) if n.get().is_finite() => Ok(ObjectiveInterval::exact(n.get())),
            ValueExpr::Var(variable) => self
                .controls
                .iter()
                .position(|c| c.variable == *variable)
                .map(|i| bounds[i])
                .ok_or_else(|| invalid("unbound certified expression variable")),
            ValueExpr::Neg(value) => {
                let b = self.enclose(value, bounds, depth + 1, nodes)?;
                Ok(ObjectiveInterval { lower: -b.upper, upper: -b.lower })
            }
            ValueExpr::Arith { op, lhs, rhs } => self
                .enclose(lhs, bounds, depth + 1, nodes)?
                .op(*op, self.enclose(rhs, bounds, depth + 1, nodes)?),
            _ => Err(invalid("objective expression has no implemented certified enclosure")),
        }
    }
    /// Recompute a conservative expression enclosure for a subdomain.
    ///
    /// # Errors
    /// Invalid subdomains, numerical overflow or unresolved division singularities.
    pub fn bounds(
        &self,
        bounds: &[ObjectiveInterval],
    ) -> Result<ObjectiveInterval, ProgramSearchError> {
        if bounds.len() != self.controls.len()
            || bounds.iter().zip(&self.controls).any(|(b, c)| {
                !b.lower.is_finite()
                    || !b.upper.is_finite()
                    || b.lower > b.upper
                    || b.lower < c.bounds.lower
                    || b.upper > c.bounds.upper
            })
        {
            return Err(invalid("certificate subdomain is outside compiled bounds"));
        }
        self.enclose(&self.expression, bounds, 0, &mut 0)
    }
    fn constraint_bounds(
        coefficients: &[f64],
        bounds: &[ObjectiveInterval],
    ) -> Result<ObjectiveInterval, ProgramSearchError> {
        coefficients.iter().zip(bounds).try_fold(ObjectiveInterval::exact(0.0), |sum, (a, b)| {
            sum.op(ArithOp::Add, ObjectiveInterval::exact(*a).op(ArithOp::Mul, *b)?)
        })
    }
    fn feasible(&self, point: &[f64]) -> Result<bool, ProgramSearchError> {
        if point.len() != self.controls.len()
            || point
                .iter()
                .zip(&self.controls)
                .any(|(x, c)| !x.is_finite() || *x < c.bounds.lower || *x > c.bounds.upper)
        {
            return Ok(false);
        }
        let bounds = point.iter().map(|x| ObjectiveInterval::exact(*x)).collect::<Vec<_>>();
        // Feasibility must be proved, rather than accepted by an unrounded dot product.
        for r in &self.restrictions {
            if Self::constraint_bounds(&r.coefficients, &bounds)?.upper > r.bound {
                return Ok(false);
            }
        }
        Ok(true)
    }
    fn possible(&self, bounds: &[ObjectiveInterval]) -> Result<bool, ProgramSearchError> {
        for r in &self.restrictions {
            if Self::constraint_bounds(&r.coefficients, bounds)?.lower > r.bound {
                return Ok(false);
            }
        }
        Ok(true)
    }
    fn reserve_frontier(
        &self,
        options: CertificateOptions,
        ctx: &ExecutionContext,
    ) -> Result<Arc<antecedent_core::execution::AdmissionLease>, ProgramSearchError> {
        let nodes = options
            .max_splits
            .checked_add(1)
            .ok_or_else(|| invalid("certificate region count overflow"))?;
        let quote = self
            .controls
            .len()
            .checked_mul(std::mem::size_of::<ObjectiveInterval>())
            .and_then(|n| n.checked_add(std::mem::size_of::<CertifiedRegion>()))
            .and_then(|n| n.checked_mul(nodes))
            .and_then(|n| n.checked_mul(2))
            .and_then(|n| n.checked_add(self.controls.len().checked_mul(256)?))
            .and_then(|n| n.checked_add(self.restrictions.len().checked_mul(32)?))
            .and_then(|n| n.checked_add(1024))
            .ok_or_else(|| invalid("certificate storage overflow"))?;
        if ctx.memory_limit_bytes().is_some_and(|limit| quote as u64 > limit) {
            return Err(ProgramSearchError::Resource(
                "certificate frontier exceeds memory limit".into(),
            ));
        }
        Ok(Arc::new(
            ctx.acquire_admission(0, quote as u64)
                .map_err(|e| ProgramSearchError::Resource(e.to_string()))?,
        ))
    }
    /// Maximize the actual expression by deterministic best-bound subdivision.
    /// The caller supplies one proved-feasible initial point; no feasibility solver
    /// or shape assumption is silently substituted. Resource exhaustion keeps a
    /// conservative bound, while cancellation refuses execution.
    ///
    /// # Errors
    /// Invalid tolerance or initial point, numerical failure, cancellation or admission refusal.
    #[expect(
        clippy::too_many_lines,
        reason = "keeps the deterministic region traversal and its stopping proof together"
    )]
    pub fn execute(
        &self,
        initial: Vec<f64>,
        options: CertificateOptions,
        ctx: &ExecutionContext,
    ) -> Result<GlobalProgramCertificate, ProgramSearchError> {
        cancelled(ctx)?;
        if !options.absolute_gap.is_finite()
            || options.absolute_gap < 0.0
            || !self.feasible(&initial)?
        {
            return Err(invalid("invalid tolerance or unproved feasible initial point"));
        }
        let retention = self.reserve_frontier(options, ctx)?;
        let mut controls = initial;
        let separable_bound = self.separable_bound(&mut controls, options, ctx)?;
        let mut root = self.root();
        let monotone_reductions = self.reduce_monotone(&mut root, &mut controls, ctx)?;
        let mut incumbent = self
            .bounds(&controls.iter().map(|x| ObjectiveInterval::exact(*x)).collect::<Vec<_>>())?;
        let mut frontier = vec![CertifiedRegion { upper: self.bounds(&root)?.upper, bounds: root }];
        let mut splits =
            separable_bound.as_ref().map_or(0, |b| b.blocks.iter().map(|b| b.splits).sum());
        let mut infeasible_regions = 0;
        let mut dominated_regions = 0;
        let stop = loop {
            cancelled(ctx)?;
            let upper = frontier
                .iter()
                .map(|r| r.upper)
                .fold(incumbent.upper, f64::max)
                .min(separable_bound.as_ref().map_or(f64::INFINITY, |b| b.upper));
            let gap = objective_gap(upper, incumbent.lower);
            if gap <= options.absolute_gap {
                break CertificateStop::Certified;
            }
            if splits == options.max_splits {
                break CertificateStop::SplitLimit;
            }
            let index = frontier
                .iter()
                .enumerate()
                .max_by(|(i, a), (j, b)| a.upper.total_cmp(&b.upper).then_with(|| j.cmp(i)))
                .map(|(i, _)| i);
            let Some(index) = index else {
                break CertificateStop::ResolutionLimit;
            };
            let region = frontier.swap_remove(index);
            let axis = region
                .bounds
                .iter()
                .enumerate()
                .max_by(|(i, a), (j, b)| {
                    (a.upper / 2.0 - a.lower / 2.0)
                        .total_cmp(&(b.upper / 2.0 - b.lower / 2.0))
                        .then_with(|| j.cmp(i))
                })
                .map(|(i, _)| i)
                .ok_or_else(|| invalid("empty certificate region"))?;
            let midpoint = region.bounds[axis].lower / 2.0 + region.bounds[axis].upper / 2.0;
            if midpoint <= region.bounds[axis].lower || midpoint >= region.bounds[axis].upper {
                frontier.push(region);
                break CertificateStop::ResolutionLimit;
            }
            splits += 1;
            for upper_half in [false, true] {
                let mut bounds = region.bounds.clone();
                if upper_half {
                    bounds[axis].lower = midpoint;
                } else {
                    bounds[axis].upper = midpoint;
                }
                if !self.possible(&bounds)? {
                    infeasible_regions += 1;
                    continue;
                }
                let candidate =
                    bounds.iter().map(|b| b.lower / 2.0 + b.upper / 2.0).collect::<Vec<_>>();
                if self.feasible(&candidate)? {
                    let value = self.bounds(
                        &candidate.iter().map(|x| ObjectiveInterval::exact(*x)).collect::<Vec<_>>(),
                    )?;
                    if value.lower > incumbent.lower {
                        controls = candidate;
                        incumbent = value;
                    }
                }
                let upper = self.bounds(&bounds)?.upper;
                if upper <= incumbent.lower {
                    dominated_regions += 1;
                } else {
                    frontier.push(CertifiedRegion { bounds, upper });
                }
            }
        };
        let dual_bound = match self.concave_dual_bound(&controls, ctx) {
            Ok(bound) => bound,
            Err(ProgramSearchError::Cancelled) => return Err(ProgramSearchError::Cancelled),
            Err(_) => None,
        };
        let global_upper = frontier
            .iter()
            .map(|r| r.upper)
            .fold(incumbent.upper, f64::max)
            .min(dual_bound.as_ref().map_or(f64::INFINITY, |bound| bound.upper));
        let global_upper =
            global_upper.min(separable_bound.as_ref().map_or(f64::INFINITY, |b| b.upper));
        let gap = objective_gap(global_upper, incumbent.lower);
        let stop = if gap <= options.absolute_gap { CertificateStop::Certified } else { stop };
        Ok(GlobalProgramCertificate {
            identity: self.identity,
            controls,
            incumbent,
            global_upper,
            gap,
            stop,
            frontier,
            splits,
            infeasible_regions,
            dominated_regions,
            monotone_reductions,
            dual_bound,
            separable_bound,
            retention,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn variable() -> ValueExpr {
        ValueExpr::var(VariableId::from_raw(0))
    }
    fn compile(expression: ValueExpr) -> PreparedCertifiedSearch {
        PreparedCertifiedSearch::compile(
            expression,
            vec![CertifiedControl {
                variable: VariableId::from_raw(0),
                bounds: ObjectiveInterval { lower: 0.0, upper: 1.0 },
            }],
            vec![],
        )
        .unwrap()
    }
    #[test]
    fn certified_interior_optimum_and_retained_global_bound() {
        let x = variable();
        let plan = compile(x.clone().mul(ValueExpr::constant(1.0).sub(x)));
        let result = plan
            .execute(
                vec![0.0],
                CertificateOptions { absolute_gap: 1e-4, max_splits: 100_000 },
                &ExecutionContext::for_tests(1),
            )
            .unwrap();
        assert_eq!(result.stop, CertificateStop::Certified);
        assert!(result.global_upper >= 0.25);
        assert!(result.incumbent.lower <= 0.25);
        assert!(result.gap <= 1e-4);
        for region in &result.frontier {
            assert!(plan.bounds(&region.bounds).unwrap().upper <= region.upper);
        }
    }
    #[test]
    fn resource_termination_never_claims_certification() {
        let x = variable();
        let plan = compile(x.clone().mul(ValueExpr::constant(1.0).sub(x)));
        let result = plan
            .execute(
                vec![0.0],
                CertificateOptions { absolute_gap: 1e-8, max_splits: 0 },
                &ExecutionContext::for_tests(1),
            )
            .unwrap();
        assert_eq!(result.stop, CertificateStop::SplitLimit);
        assert!(result.global_upper >= 1.0);
        let ctx = ExecutionContext::for_tests(1);
        ctx.cancellation.cancel();
        assert!(matches!(
            plan.execute(vec![0.0], CertificateOptions { absolute_gap: 1e-8, max_splits: 0 }, &ctx),
            Err(ProgramSearchError::Cancelled)
        ));
    }

    #[test]
    fn exactly_constant_objective_can_certify_a_zero_gap_without_subdivision() {
        let result = compile(ValueExpr::constant(5.0))
            .execute(
                vec![0.0],
                CertificateOptions { absolute_gap: 0.0, max_splits: 0 },
                &ExecutionContext::for_tests(1),
            )
            .unwrap();
        assert_eq!(result.stop, CertificateStop::Certified);
        assert_eq!(result.gap, 0.0);
    }
    #[test]
    fn interval_contains_dense_numerical_truth_and_refuses_singular_division() {
        let x = variable();
        let plan = compile(x.clone().mul(ValueExpr::constant(1.0).sub(x.clone())));
        let bound = plan.bounds(&[ObjectiveInterval { lower: 0.2, upper: 0.7 }]).unwrap();
        for i in 200..=700 {
            let x = f64::from(i) / 1000.0;
            assert!(bound.lower <= x * (1.0 - x) && bound.upper >= x * (1.0 - x));
        }
        assert!(
            PreparedCertifiedSearch::compile(
                ValueExpr::constant(1.0).div(x),
                plan.controls.clone(),
                vec![]
            )
            .is_err()
        );
    }

    #[test]
    fn nonconcave_competing_peaks_do_not_create_a_local_optimum_certificate() {
        let peak = |at, height| {
            let delta = variable().sub(ValueExpr::constant(at));
            ValueExpr::constant(height).sub(delta.clone().mul(delta))
        };
        let expression = ValueExpr::Arith {
            op: ArithOp::Max,
            lhs: Box::new(peak(0.2, 3.0)),
            rhs: Box::new(peak(0.8, 4.0)),
        };
        let plan = compile(expression);
        let result = plan
            .execute(
                vec![0.2],
                CertificateOptions { absolute_gap: 1e-4, max_splits: 10_000 },
                &ExecutionContext::for_tests(1),
            )
            .unwrap();
        assert_eq!(result.stop, CertificateStop::Certified);
        assert!(result.global_upper >= 4.0);
        assert!((result.controls[0] - 0.8).abs() < 0.02);
    }

    #[test]
    fn binding_linear_budget_prunes_regions_and_keeps_a_sound_bound() {
        let plan = PreparedCertifiedSearch::compile(
            variable(),
            compile(variable()).controls,
            vec![CertifiedRestriction { coefficients: vec![1.0], bound: 0.6 }],
        )
        .unwrap();
        let result = plan
            .execute(
                vec![0.0],
                CertificateOptions { absolute_gap: 1e-3, max_splits: 1000 },
                &ExecutionContext::for_tests(1),
            )
            .unwrap();
        assert_eq!(result.stop, CertificateStop::Certified);
        assert!(result.controls[0] <= 0.6);
        assert!(result.global_upper >= 0.6);
        assert!(result.gap <= 1e-3);
        assert!(result.infeasible_regions > 0);
    }
}
