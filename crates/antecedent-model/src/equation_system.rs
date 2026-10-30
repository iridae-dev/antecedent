// SPDX-License-Identifier: MIT OR Apache-2.0
//! Bounded polynomial equation systems and equation-specific interventions.
//! Interval branch-and-bound uses outward-rounded arithmetic and a Krawczyk
//! enclosure. A box is certified only when the enclosure maps strictly inside
//! the box and its interval derivative is a contraction. Unprocessed or small
//! undecided boxes remain explicit; no solver seed supplies a selection law.

use crate::linear_algebra::equilibrated_general_solve;
use crate::nonlinear_equilibrium::{EquilibriumError, check_cancel};
use antecedent_core::ExecutionContext;
use serde::{Deserialize, Serialize};

/// Closed finite interval, validated at the public boundary.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Interval {
    /// Lower endpoint.
    pub lower: f64,
    /// Upper endpoint.
    pub upper: f64,
}
fn up(x: f64) -> f64 {
    if x.is_nan() || x == f64::INFINITY {
        return x;
    }
    if x == 0.0 {
        return f64::from_bits(1);
    }
    f64::from_bits(if x > 0.0 { x.to_bits() + 1 } else { x.to_bits() - 1 })
}
fn down(x: f64) -> f64 {
    -up(-x)
}
impl Interval {
    /// Outward-rounded addition of two interval enclosures.
    #[must_use]
    pub fn add_outward(self, other: Self) -> Self {
        self.add(other)
    }

    /// Multiply represented closed intervals with outward rounding.
    #[must_use]
    pub fn multiply_outward(self, other: Self) -> Self {
        self.mul(other)
    }
    /// Outward-rounded multiplication by an exactly represented scalar.
    #[must_use]
    pub fn scale_outward(self, scalar: f64) -> Self {
        self.mul(Self::point(scalar))
    }

    /// Enclose `(observed-x)^2 / variance` for every `x` in this interval.
    /// Requires a finite interval/observation and positive finite variance.
    pub fn standardized_squared_distance(
        self,
        observed: f64,
        variance: f64,
    ) -> Result<Self, EquilibriumError> {
        if !self.lower.is_finite()
            || !self.upper.is_finite()
            || self.lower > self.upper
            || !observed.is_finite()
            || !variance.is_finite()
            || variance <= 0.0
        {
            return Err(invalid("squared-distance interval/variance"));
        }
        let minimum = if observed < self.lower {
            down(self.lower - observed).max(0.0)
        } else if observed > self.upper {
            down(observed - self.upper).max(0.0)
        } else {
            0.0
        };
        let maximum = up((observed - self.lower).abs()).max(up((observed - self.upper).abs()));
        Ok(Self {
            lower: down(down(minimum * minimum).max(0.0) / variance).max(0.0),
            upper: up(up(maximum * maximum) / variance),
        })
    }

    fn point(x: f64) -> Self {
        Self { lower: x, upper: x }
    }
    fn add(self, b: Self) -> Self {
        Self { lower: down(self.lower + b.lower), upper: up(self.upper + b.upper) }
    }
    fn neg(self) -> Self {
        Self { lower: -self.upper, upper: -self.lower }
    }
    fn sub(self, b: Self) -> Self {
        self.add(b.neg())
    }
    fn mul(self, b: Self) -> Self {
        let v = [
            self.lower * b.lower,
            self.lower * b.upper,
            self.upper * b.lower,
            self.upper * b.upper,
        ];
        if v.iter().any(|x| x.is_nan()) {
            return Self { lower: f64::NEG_INFINITY, upper: f64::INFINITY };
        }
        Self {
            lower: down(v.iter().copied().fold(f64::INFINITY, f64::min)),
            upper: up(v.iter().copied().fold(f64::NEG_INFINITY, f64::max)),
        }
    }
    fn pow(self, k: u32) -> Self {
        (0..k).fold(Self::point(1.0), |a, _| a.mul(self))
    }
    fn midpoint(self) -> f64 {
        0.5 * self.lower + 0.5 * self.upper
    }
    fn magnitude(self) -> f64 {
        self.lower.abs().max(self.upper.abs())
    }
    fn excludes_zero(self) -> bool {
        self.lower > 0.0 || self.upper < 0.0
    }
}

/// One polynomial term with bounded nonnegative integer powers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Monomial {
    /// Floating coefficient, treated as its exact represented value.
    pub coefficient: f64,
    /// One power per variable, each <= 8.
    pub powers: Vec<u32>,
}

/// A named residual equation `sum(terms) = 0`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolynomialEquation {
    /// Stable equation identity.
    pub id: String,
    /// Polynomial residual terms.
    pub terms: Vec<Monomial>,
}

/// Square bounded equation system. Equation ids are distinct from variable ids.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolynomialSystem {
    /// Dense variable names.
    pub variables: Vec<String>,
    /// Residual equations, in dense equation order.
    pub equations: Vec<PolynomialEquation>,
    /// Declared finite search domain; no claim outside this domain.
    pub domain: Vec<Interval>,
}

/// Work and unresolved-box policy.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IsolationPolicy {
    /// Maximum processed boxes; remaining boxes are retained as unresolved.
    pub max_boxes: usize,
    /// Minimum box width before declaring that box unresolved.
    pub minimum_width: f64,
}
impl Default for IsolationPolicy {
    fn default() -> Self {
        Self { max_boxes: 10_000, minimum_width: 1e-8 }
    }
}

/// Bounded solution-set certificate and explicit unresolved regions.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IsolatedSolutions {
    /// Each enclosure contains exactly one root of the declared equations.
    pub roots: Vec<Vec<Interval>>,
    /// Regions not proved empty or uniquely solved (including exhausted work).
    pub unresolved: Vec<Vec<Interval>>,
    /// Processed boxes, including exclusions and bisections.
    pub visited: usize,
    /// Boxes certified to contain no root.
    pub excluded: usize,
}

/// Declared probability weights over certified root enclosures.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EquilibriumSelection {
    /// Certified roots, in the original solution-set order.
    pub roots: Vec<Vec<Interval>>,
    /// Explicit normalized weights. No implicit uniform law is supplied.
    pub probabilities: Vec<f64>,
}

fn invalid(s: &str) -> EquilibriumError {
    EquilibriumError::Invalid(s.into())
}
fn evaluate(e: &PolynomialEquation, x: &[Interval], derivative: Option<usize>) -> Interval {
    e.terms.iter().fold(Interval::point(0.0), |sum, term| {
        let mut value = Interval::point(term.coefficient);
        for (j, (&p, &xi)) in term.powers.iter().zip(x).enumerate() {
            let power = if derivative == Some(j) {
                if p == 0 {
                    return sum;
                }
                value = value.mul(Interval::point(f64::from(p)));
                p - 1
            } else {
                p
            };
            value = value.mul(xi.pow(power));
        }
        sum.add(value)
    })
}

impl PolynomialSystem {
    /// Check named equations/variables, degree, and finite bounded domains.
    pub fn validate(&self) -> Result<(), EquilibriumError> {
        let n = self.variables.len();
        if n == 0 || n > 8 || self.equations.len() != n || self.domain.len() != n {
            return Err(invalid("polynomial system order (1..8)"));
        }
        for names in [self.variables.clone(), self.equations.iter().map(|e| e.id.clone()).collect()]
        {
            let mut seen = std::collections::BTreeSet::new();
            if names.iter().any(|s| s.trim().is_empty() || s.len() > 256 || !seen.insert(s)) {
                return Err(invalid("equation/variable identities"));
            }
        }
        if self
            .domain
            .iter()
            .any(|b| !b.lower.is_finite() || !b.upper.is_finite() || b.lower >= b.upper)
        {
            return Err(invalid("finite nonempty domain"));
        }
        for e in &self.equations {
            if e.terms.len() > 256
                || e.terms.iter().any(|t| {
                    !t.coefficient.is_finite()
                        || t.powers.len() != n
                        || t.powers.iter().any(|&p| p > 8)
                })
            {
                return Err(invalid("polynomial terms"));
            }
        }
        Ok(())
    }

    /// Replace the named equation by `variable=value`, preserving all other
    /// equations. Different equation choices may give different interventions.
    pub fn replace_equation(
        &self,
        equation: &str,
        variable: usize,
        value: f64,
    ) -> Result<Self, EquilibriumError> {
        self.validate()?;
        if variable >= self.variables.len() || !value.is_finite() {
            return Err(invalid("equation intervention"));
        }
        let index = self
            .equations
            .iter()
            .position(|e| e.id == equation)
            .ok_or_else(|| invalid("unknown equation"))?;
        let n = self.variables.len();
        let mut powers = vec![0; n];
        powers[variable] = 1;
        let mut next = self.clone();
        next.equations[index].terms = vec![
            Monomial { coefficient: 1.0, powers },
            Monomial { coefficient: -value, powers: vec![0; n] },
        ];
        Ok(next)
    }

    /// Enumerate disjoint search boxes with outward-rounded exclusions and
    /// Krawczyk uniqueness. Resource exhaustion retains unresolved regions.
    pub fn isolate(
        &self,
        policy: IsolationPolicy,
        ctx: &ExecutionContext,
    ) -> Result<IsolatedSolutions, EquilibriumError> {
        self.validate()?;
        if policy.max_boxes == 0
            || policy.max_boxes > 100_000
            || !policy.minimum_width.is_finite()
            || policy.minimum_width <= 0.0
        {
            return Err(invalid("isolation limits"));
        }
        let plan = crate::compile::compile_constraints(self)?;
        let n = plan.blocks[0].variables().len();
        let bytes = policy
            .max_boxes
            .checked_mul(n * 64 + 128)
            .ok_or_else(|| invalid("isolation memory overflow"))?;
        let _lease = ctx
            .acquire_serial_numerical_admission(bytes as u64)
            .map_err(|e| EquilibriumError::Resource(e.to_string()))?;
        let mut pending = vec![self.domain.clone()];
        let mut result = IsolatedSolutions {
            roots: Vec::new(),
            unresolved: Vec::new(),
            visited: 0,
            excluded: 0,
        };
        while let Some(region) = pending.pop() {
            check_cancel(ctx)?;
            if result.visited >= policy.max_boxes {
                result.unresolved.push(region);
                result.unresolved.extend(pending);
                break;
            }
            result.visited += 1;
            if self.equations.iter().any(|e| evaluate(e, &region, None).excludes_zero()) {
                result.excluded += 1;
                continue;
            }
            let (excluded, certified, _) = self.krawczyk(&region);
            if excluded {
                result.excluded += 1;
                continue;
            }
            if certified {
                let mut refined = region;
                for _ in 0..64 {
                    let (_, _, enclosure) = self.krawczyk(&refined);
                    if enclosure.len() != n {
                        break;
                    }
                    let mut changed = false;
                    for (value, next) in refined.iter_mut().zip(enclosure) {
                        let lower = value.lower.max(next.lower);
                        let upper = value.upper.min(next.upper);
                        if lower > upper {
                            return Err(invalid("inconsistent root enclosure"));
                        }
                        changed |= lower > value.lower || upper < value.upper;
                        *value = Interval { lower, upper };
                    }
                    if !changed || refined.iter().all(|b| b.upper - b.lower <= policy.minimum_width)
                    {
                        break;
                    }
                }
                result.roots.push(refined);
                continue;
            }
            let split = (0..n)
                .max_by(|&i, &j| {
                    (region[i].upper - region[i].lower)
                        .total_cmp(&(region[j].upper - region[j].lower))
                })
                .ok_or_else(|| invalid("domain order"))?;
            let width = region[split].upper - region[split].lower;
            let midpoint = 0.527 * region[split].lower + 0.473 * region[split].upper;
            if width <= policy.minimum_width
                || midpoint <= region[split].lower
                || midpoint >= region[split].upper
            {
                result.unresolved.push(region);
                continue;
            }
            let mut left = region.clone();
            let mut right = region;
            left[split].upper = midpoint;
            right[split].lower = midpoint;
            pending.push(right);
            pending.push(left);
        }
        Ok(result)
    }

    fn krawczyk(&self, region: &[Interval]) -> (bool, bool, Vec<Interval>) {
        let n = region.len();
        let mid: Vec<_> = region.iter().map(|b| Interval::point(b.midpoint())).collect();
        let jac: Vec<_> = self
            .equations
            .iter()
            .flat_map(|e| (0..n).map(|j| evaluate(e, &mid, Some(j)).midpoint()))
            .collect();
        let mut matrix = jac;
        let mut inverse = vec![0.0; n * n];
        // Kernel RHS storage is column-major; transpose while reading below.
        for i in 0..n {
            inverse[i * n + i] = 1.0;
        }
        if !equilibrated_general_solve(&mut matrix, &mut inverse, n, n) {
            return (false, false, Vec::new());
        }
        if inverse.iter().any(|v| !v.is_finite()) {
            return (false, false, Vec::new());
        }
        let f: Vec<_> = self.equations.iter().map(|e| evaluate(e, &mid, None)).collect();
        let mut enclosure = Vec::new();
        let mut contractive = true;
        for i in 0..n {
            let mut center = mid[i];
            for a in 0..n {
                center = center.sub(Interval::point(inverse[a * n + i]).mul(f[a]));
            }
            let mut lip = 0.0;
            for j in 0..n {
                let mut error = Interval::point(if i == j { 1.0 } else { 0.0 });
                for a in 0..n {
                    error = error.sub(Interval::point(inverse[a * n + i]).mul(evaluate(
                        &self.equations[a],
                        region,
                        Some(j),
                    )));
                }
                lip = up(lip + error.magnitude());
                center = center.add(error.mul(region[j].sub(mid[j])));
            }
            contractive &= lip < 1.0;
            enclosure.push(center);
        }
        let excluded =
            enclosure.iter().zip(region).any(|(k, x)| k.upper < x.lower || k.lower > x.upper);
        let certified = contractive
            && enclosure.iter().zip(region).all(|(k, x)| k.lower > x.lower && k.upper < x.upper);
        (excluded, certified, enclosure)
    }
}

impl IsolatedSolutions {
    /// Bind a caller-declared selection distribution. Unresolved mass prevents
    /// claiming an exhaustive selection law; weights must explicitly sum to one.
    pub fn select(&self, probabilities: &[f64]) -> Result<EquilibriumSelection, EquilibriumError> {
        if !self.unresolved.is_empty()
            || self.roots.is_empty()
            || probabilities.len() != self.roots.len()
            || probabilities.iter().any(|p| !p.is_finite() || *p < 0.0)
            || (probabilities.iter().sum::<f64>() - 1.0).abs() > 1e-12
        {
            return Err(invalid(
                "selection needs complete roots and explicit normalized probabilities",
            ));
        }
        Ok(EquilibriumSelection {
            roots: self.roots.clone(),
            probabilities: probabilities.to_vec(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn one_dim() -> PolynomialSystem {
        PolynomialSystem {
            variables: vec!["x".into()],
            equations: vec![PolynomialEquation {
                id: "cubic".into(),
                terms: vec![
                    Monomial { coefficient: 1.0, powers: vec![3] },
                    Monomial { coefficient: -1.0, powers: vec![1] },
                ],
            }],
            domain: vec![Interval { lower: -2.0, upper: 2.0 }],
        }
    }
    #[test]
    fn krawczyk_accepts_mixed_scale_nonsingular_jacobian() {
        let region = vec![Interval { lower: -1.0, upper: 1.0 }; 2];
        let system = PolynomialSystem {
            variables: vec!["x".into(), "y".into()],
            equations: vec![
                PolynomialEquation {
                    id: "x".into(),
                    terms: vec![Monomial { coefficient: 1e-100, powers: vec![1, 0] }],
                },
                PolynomialEquation {
                    id: "y".into(),
                    terms: vec![Monomial { coefficient: 1e100, powers: vec![0, 1] }],
                },
            ],
            domain: region.clone(),
        };
        let (excluded, certified, enclosure) = system.krawczyk(&region);
        assert!(!excluded && certified, "{enclosure:?}");
        assert!(enclosure.iter().all(|x| x.lower <= 0.0 && x.upper >= 0.0));
    }

    #[test]
    fn cubic_roots_are_isolated_and_selection_is_explicit() {
        let ctx = ExecutionContext::for_tests(0);
        let solutions = one_dim().isolate(IsolationPolicy::default(), &ctx).unwrap();
        assert!(solutions.unresolved.is_empty(), "{solutions:?}");
        assert_eq!(solutions.roots.len(), 3);
        for expected in [-1.0, 0.0, 1.0] {
            assert!(
                solutions.roots.iter().any(|b| b[0].lower <= expected && b[0].upper >= expected)
            );
        }
        assert!(solutions.select(&[0.2, 0.3, 0.5]).is_ok());
        assert!(solutions.select(&[1.0, 1.0, 1.0]).is_err());
        let short = one_dim()
            .isolate(IsolationPolicy { max_boxes: 1, ..Default::default() }, &ctx)
            .unwrap();
        assert!(!short.unresolved.is_empty());
        assert!(short.select(&[]).is_err());
    }
    #[test]
    fn replacing_distinct_equations_at_same_variable_changes_the_solution() {
        let system = PolynomialSystem {
            variables: vec!["x".into(), "y".into()],
            equations: vec![
                PolynomialEquation {
                    id: "sum".into(),
                    terms: vec![
                        Monomial { coefficient: 1.0, powers: vec![1, 0] },
                        Monomial { coefficient: 1.0, powers: vec![0, 1] },
                        Monomial { coefficient: -3.0, powers: vec![0, 0] },
                    ],
                },
                PolynomialEquation {
                    id: "difference".into(),
                    terms: vec![
                        Monomial { coefficient: 1.0, powers: vec![1, 0] },
                        Monomial { coefficient: -1.0, powers: vec![0, 1] },
                        Monomial { coefficient: -1.0, powers: vec![0, 0] },
                    ],
                },
            ],
            domain: vec![Interval { lower: -5.0, upper: 5.0 }; 2],
        };
        let ctx = ExecutionContext::for_tests(1);
        let a = system
            .replace_equation("sum", 0, 0.0)
            .unwrap()
            .isolate(IsolationPolicy::default(), &ctx)
            .unwrap();
        let b = system
            .replace_equation("difference", 0, 0.0)
            .unwrap()
            .isolate(IsolationPolicy::default(), &ctx)
            .unwrap();
        assert_eq!(a.roots.len(), 1);
        assert_eq!(b.roots.len(), 1);
        assert!(a.unresolved.is_empty() && b.unresolved.is_empty());
        // Initial boxes may be broad; use narrow declared y domains to certify the distinct truth.
        let mut sa = system.replace_equation("sum", 0, 0.0).unwrap();
        sa.domain[1] = Interval { lower: -1.1, upper: -0.9 };
        let mut sb = system.replace_equation("difference", 0, 0.0).unwrap();
        sb.domain[1] = Interval { lower: 2.9, upper: 3.1 };
        assert_eq!(sa.isolate(IsolationPolicy::default(), &ctx).unwrap().roots.len(), 1);
        assert_eq!(sb.isolate(IsolationPolicy::default(), &ctx).unwrap().roots.len(), 1);
    }
}
