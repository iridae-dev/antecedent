//! Outward-rounded interval enclosures for certified decisions across a preference set.
//!
//! An [`Interval`] is a closed interval of reals with finite ends. Every floating operation
//! moves the lower end of its result down and the upper end up by one representable step (two
//! for `exp` and `ln`), so each enclosure contains the exact real range. [`enclose`] evaluates
//! a model expression with the attributes at their drawn values and every parameter ranging
//! over its interval; [`weighted_sum`] widens a weighted sum by `2 (n + 1) eps` times the sum of
//! magnitudes so it also contains any floating summation order. [`Queued`] orders boxes
//! waiting to be split: the box of largest volume first, ties broken by the lexicographically
//! smallest lower corner. [`measure_box`] and [`split_box`] measure and halve a box relative to
//! the declared ranges; [`linear_relation_fit`] and [`value_relation_fit`] place a box relative to
//! the linear and value relations of a preference set.

// SPDX-License-Identifier: MIT OR Apache-2.0

use std::cmp::Ordering;

use crate::preference_model::{
    BeliefRows, LinearRelation, PreferenceExpr, RelationSense, ValueRelation,
};

/// Why an enclosure could not be formed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum EncloseFailure {
    /// A logarithm's argument reaches zero or below.
    LnDomain,
    /// A bound is not finite (or an index is unknown, which a validated contract excludes).
    NonFinite,
}

/// A closed interval `[lo, hi]` of reals with finite ends.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Interval {
    pub(crate) lo: f64,
    pub(crate) hi: f64,
}

impl Interval {
    /// The degenerate interval of an exactly known value.
    pub(crate) const fn point(value: f64) -> Self {
        Self { lo: value, hi: value }
    }

    /// Finite ends only.
    pub(crate) fn checked(lo: f64, hi: f64) -> Result<Self, EncloseFailure> {
        if lo.is_finite() && hi.is_finite() && lo <= hi {
            Ok(Self { lo, hi })
        } else {
            Err(EncloseFailure::NonFinite)
        }
    }

    /// Ends computed with round-to-nearest, widened outward by one representable step.
    pub(crate) fn rounded(lo: f64, hi: f64) -> Result<Self, EncloseFailure> {
        Self::checked(lo.next_down(), hi.next_up())
    }

    pub(crate) fn plus(self, other: Self) -> Result<Self, EncloseFailure> {
        Self::rounded(self.lo + other.lo, self.hi + other.hi)
    }

    pub(crate) fn minus(self, other: Self) -> Result<Self, EncloseFailure> {
        Self::rounded(self.lo - other.hi, self.hi - other.lo)
    }

    pub(crate) const fn negated(self) -> Self {
        Self { lo: -self.hi, hi: -self.lo }
    }

    pub(crate) fn times(self, other: Self) -> Result<Self, EncloseFailure> {
        let products =
            [self.lo * other.lo, self.lo * other.hi, self.hi * other.lo, self.hi * other.hi];
        if products.iter().any(|value| value.is_nan()) {
            return Err(EncloseFailure::NonFinite);
        }
        let lo = products.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = products.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Self::rounded(lo, hi)
    }

    pub(crate) fn scale(self, factor: f64) -> Result<Self, EncloseFailure> {
        self.times(Self::point(factor))
    }

    /// Pointwise minimum; exact, since `min` is monotone in both arguments.
    pub(crate) const fn min(self, other: Self) -> Self {
        Self { lo: self.lo.min(other.lo), hi: self.hi.min(other.hi) }
    }

    /// Pointwise maximum; exact, since `max` is monotone in both arguments.
    pub(crate) const fn max(self, other: Self) -> Self {
        Self { lo: self.lo.max(other.lo), hi: self.hi.max(other.hi) }
    }

    /// `exp` is increasing; the library function is not correctly rounded, so each end moves
    /// two steps. The lower end is never below zero.
    pub(crate) fn exp(self) -> Result<Self, EncloseFailure> {
        let lo = self.lo.exp().next_down().next_down().max(0.0);
        let hi = self.hi.exp().next_up().next_up();
        Self::checked(lo, hi)
    }

    /// `ln` is increasing on `(0, inf)`; an argument reaching zero or below is outside the
    /// domain.
    pub(crate) fn ln(self) -> Result<Self, EncloseFailure> {
        if self.lo <= 0.0 {
            return Err(EncloseFailure::LnDomain);
        }
        Self::checked(self.lo.ln().next_down().next_down(), self.hi.ln().next_up().next_up())
    }

    /// Exactly zero at both ends.
    pub(crate) fn is_zero(self) -> bool {
        self.lo >= 0.0 && self.hi <= 0.0
    }
}

/// Enclose `expr` with the attributes at the given values and every parameter ranging over its
/// interval.
pub(crate) fn enclose(
    expr: &PreferenceExpr,
    attributes: &[f64],
    parameters: &[Interval],
) -> Result<Interval, EncloseFailure> {
    use PreferenceExpr as P;
    let go = |inner: &PreferenceExpr| enclose(inner, attributes, parameters);
    let value = match expr {
        P::Const(value) => Interval::point(*value),
        P::Attribute(k) => Interval::point(*attributes.get(*k).ok_or(EncloseFailure::NonFinite)?),
        P::Parameter(j) => *parameters.get(*j).ok_or(EncloseFailure::NonFinite)?,
        P::Neg(a) => go(a)?.negated(),
        P::Add(a, b) => go(a)?.plus(go(b)?)?,
        P::Sub(a, b) => go(a)?.minus(go(b)?)?,
        P::Mul(a, b) => go(a)?.times(go(b)?)?,
        P::Min(a, b) => go(a)?.min(go(b)?),
        P::Max(a, b) => go(a)?.max(go(b)?),
        P::Exp(a) => go(a)?.exp()?,
        P::Ln(a) => go(a)?.ln()?,
    };
    Interval::checked(value.lo, value.hi)
}

/// Enclosure of `sum w_i x_i` over terms of positive weight, widened by `2 (n + 1) eps` times
/// the sum of magnitudes so that any floating summation order is also contained.
pub(crate) fn weighted_sum(
    terms: impl IntoIterator<Item = (f64, Interval)>,
) -> Result<Interval, EncloseFailure> {
    let mut total = Interval::point(0.0);
    let mut magnitude = 0.0_f64;
    let mut count = 0.0_f64;
    for (weight, value) in terms {
        if weight > 0.0 {
            total = total.plus(value.scale(weight)?)?;
            magnitude += weight * value.lo.abs().max(value.hi.abs());
            count += 1.0;
        }
    }
    let slack = 2.0 * (count + 1.0) * f64::EPSILON * magnitude;
    Interval::rounded(total.lo - slack, total.hi + slack)
}

/// `constant + sum_j parameters_j * coefficients_j`, skipping exactly-zero coefficients.
pub(crate) fn affine(
    constant: Interval,
    coefficients: &[Interval],
    parameters: &[Interval],
) -> Result<Interval, EncloseFailure> {
    let mut total = constant;
    for (coefficient, parameter) in coefficients.iter().zip(parameters) {
        if !coefficient.is_zero() {
            total = total.plus(parameter.times(*coefficient)?)?;
        }
    }
    Ok(total)
}

/// A box waiting to be split. Larger volume first, then the smaller lower corner.
pub(crate) struct Queued {
    pub(crate) volume: f64,
    pub(crate) lower: Vec<f64>,
    pub(crate) slot: usize,
}

pub(crate) fn lexicographic(left: &[f64], right: &[f64]) -> Ordering {
    left.iter()
        .zip(right)
        .map(|(l, r)| l.total_cmp(r))
        .find(|order| order.is_ne())
        .unwrap_or(Ordering::Equal)
}

impl Ord for Queued {
    fn cmp(&self, other: &Self) -> Ordering {
        self.volume
            .total_cmp(&other.volume)
            .then_with(|| lexicographic(&other.lower, &self.lower))
            .then_with(|| other.slot.cmp(&self.slot))
    }
}

impl PartialOrd for Queued {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Queued {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl Eq for Queued {}

/// Where a box sits relative to the relations.
pub(crate) enum Fit {
    Inside,
    Boundary,
    Outside,
}

/// Volume relative to the declared box and the largest relative width, over parameters of
/// positive declared width.
pub(crate) fn measure_box(declared: &[(f64, f64)], lower: &[f64], upper: &[f64]) -> (f64, f64) {
    let mut volume = 1.0;
    let mut widest = 0.0_f64;
    for ((l, u), (dl, du)) in lower.iter().zip(upper).zip(declared) {
        let range = du - dl;
        if range > 0.0 {
            let fraction = (u - l) / range;
            volume *= fraction;
            widest = widest.max(fraction);
        }
    }
    (volume, widest)
}

/// The two halves of a box, split at the midpoint of the parameter whose width is largest
/// relative to its declared range (the first such parameter on ties), or `None` when every
/// relative width is at most `min_width_fraction` or the midpoint is not interior.
pub(crate) fn split_box(
    declared: &[(f64, f64)],
    min_width_fraction: f64,
    lower: &[f64],
    upper: &[f64],
) -> Option<[(Vec<f64>, Vec<f64>); 2]> {
    let (_, widest) = measure_box(declared, lower, upper);
    if widest <= min_width_fraction {
        return None;
    }
    let mut chosen: Option<(usize, f64)> = None;
    for (j, ((l, u), (dl, du))) in lower.iter().zip(upper).zip(declared).enumerate() {
        let range = du - dl;
        if range > 0.0 {
            let fraction = (u - l) / range;
            if chosen.is_none_or(|(_, best)| fraction > best) {
                chosen = Some((j, fraction));
            }
        }
    }
    let (j, _) = chosen?;
    let (l, u) = (lower[j], upper[j]);
    let mid = 0.5 * l + 0.5 * u;
    let interior = l < mid && mid < u;
    if !interior {
        return None;
    }
    let mut left_upper = upper.to_vec();
    left_upper[j] = mid;
    let mut right_lower = lower.to_vec();
    right_lower[j] = mid;
    Some([(lower.to_vec(), left_upper), (right_lower, upper.to_vec())])
}

/// Place a box relative to linear relations: `Outside` when some relation is proven violated
/// everywhere in it, `Boundary` when some relation is neither proven satisfied nor proven
/// violated, `Inside` otherwise. An enclosure failure names the position of its relation.
pub(crate) fn linear_relation_fit(
    relations: &[LinearRelation],
    params: &[Interval],
) -> Result<Fit, (EncloseFailure, usize)> {
    let mut boundary = false;
    for (position, relation) in relations.iter().enumerate() {
        let lhs = relation
            .terms
            .iter()
            .try_fold(Interval::point(0.0), |sum, (j, coefficient)| {
                let parameter = params.get(*j).copied().ok_or(EncloseFailure::NonFinite)?;
                sum.plus(parameter.scale(*coefficient)?)
            })
            .map_err(|failure| (failure, position))?;
        let bound = relation.bound;
        let (satisfied, violated) = match relation.sense {
            RelationSense::AtLeast => (lhs.lo >= bound, lhs.hi < bound),
            RelationSense::AtMost => (lhs.hi <= bound, lhs.lo > bound),
            RelationSense::Equal => {
                (lhs.lo >= bound && lhs.hi <= bound, lhs.hi < bound || lhs.lo > bound)
            }
        };
        if violated {
            return Ok(Fit::Outside);
        }
        boundary |= !satisfied;
    }
    Ok(if boundary { Fit::Boundary } else { Fit::Inside })
}

/// Enclosure `(lower, upper)` of the exact real `sum + error`, where `error` is the exact
/// rounding error of a round-to-nearest operation whose result is `value`: the end on the side
/// of a non-zero error moves one representable step outward; an exact result stays a point.
fn residual_enclosure(value: f64, error: f64) -> (f64, f64) {
    if error > 0.0 {
        (value, value.next_up())
    } else if error < 0.0 {
        (value.next_down(), value)
    } else {
        (value, value)
    }
}

/// Enclosure of the exact sum `a + b`: the rounding error is recovered exactly by the two-sum
/// transformation, which holds for finite operands whose sum does not overflow.
fn exact_sum(a: f64, b: f64) -> Result<(f64, f64), EncloseFailure> {
    let sum = a + b;
    if !sum.is_finite() {
        return Err(EncloseFailure::NonFinite);
    }
    let b_part = sum - a;
    let error = (a - (sum - b_part)) + (b - b_part);
    Ok(residual_enclosure(sum, error))
}

/// Enclosure of the exact product `a * b`: the rounding error is recovered exactly by a fused
/// multiply-add, which holds when the product neither overflows nor comes within `2^106` of the
/// subnormal range (below that the error itself may not be representable); such a small product
/// is widened by one step on each side unless a factor is exactly zero.
fn exact_product(a: f64, b: f64) -> Result<(f64, f64), EncloseFailure> {
    let product = a * b;
    if !product.is_finite() {
        return Err(EncloseFailure::NonFinite);
    }
    if a == 0.0 || b == 0.0 {
        return Ok((0.0, 0.0));
    }
    let representable_error = f64::MIN_POSITIVE * 2.0_f64.powi(106);
    if product.abs() < representable_error {
        return Ok((product.next_down(), product.next_up()));
    }
    Ok(residual_enclosure(product, a.mul_add(b, -product)))
}

/// Interval sum whose ends move outward only when the floating result of an end is inexact.
fn tight_plus(x: Interval, y: Interval) -> Result<Interval, EncloseFailure> {
    let (lo, _) = exact_sum(x.lo, y.lo)?;
    let (_, hi) = exact_sum(x.hi, y.hi)?;
    Interval::checked(lo, hi)
}

/// Interval product whose ends move outward only when a floating corner product is inexact.
fn tight_times(x: Interval, y: Interval) -> Result<Interval, EncloseFailure> {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for (a, b) in [(x.lo, y.lo), (x.lo, y.hi), (x.hi, y.lo), (x.hi, y.hi)] {
        let (low, high) = exact_product(a, b)?;
        lo = lo.min(low);
        hi = hi.max(high);
    }
    Interval::checked(lo, hi)
}

/// Whether every linear relation is proven satisfied on the box with no tolerance. Unlike
/// [`linear_relation_fit`], an end of an intermediate result moves outward only when its floating
/// operation is inexact (the exact rounding error, recovered by error-free transformations, is
/// non-zero), so a relation that holds with equality at a point whose sums and products are
/// representable, such as `t >= 0.5` at `t = 0.5`, is proven satisfied there.
pub(crate) fn linear_relations_hold(
    relations: &[LinearRelation],
    params: &[Interval],
) -> Result<bool, EncloseFailure> {
    for relation in relations {
        let mut lhs = Interval::point(0.0);
        for (j, coefficient) in &relation.terms {
            let parameter = params.get(*j).copied().ok_or(EncloseFailure::NonFinite)?;
            lhs = tight_plus(lhs, tight_times(parameter, Interval::point(*coefficient))?)?;
        }
        let bound = relation.bound;
        let satisfied = match relation.sense {
            RelationSense::AtLeast => lhs.lo >= bound,
            RelationSense::AtMost => lhs.hi <= bound,
            RelationSense::Equal => lhs.lo >= bound && lhs.hi <= bound,
        };
        if !satisfied {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Enclosure of `sum_i w_i v(row_i)` over the rows of one belief, with every parameter ranging
/// over its interval.
pub(crate) fn expectation(
    value: &PreferenceExpr,
    belief: &BeliefRows,
    params: &[Interval],
) -> Result<Interval, EncloseFailure> {
    let width = belief.rows.len().checked_div(belief.weights.len()).unwrap_or(0);
    if width == 0 {
        return Err(EncloseFailure::NonFinite);
    }
    let values = belief
        .rows
        .chunks_exact(width)
        .map(|row| enclose(value, row, params))
        .collect::<Result<Vec<_>, _>>()?;
    weighted_sum(belief.weights.iter().copied().zip(values))
}

/// Enclosure of the gap `E_chosen[v] - E_rejected[v]` of a value relation over a box.
pub(crate) fn value_relation_gap(
    value: &PreferenceExpr,
    relation: &ValueRelation,
    params: &[Interval],
) -> Result<Interval, EncloseFailure> {
    expectation(value, &relation.chosen, params)?.minus(expectation(
        value,
        &relation.rejected,
        params,
    )?)
}

/// Place a box relative to one value relation: `Outside` when its gap is proven below
/// `-tolerance` everywhere in the box, `Inside` when it is proven at least `-tolerance`,
/// `Boundary` otherwise.
pub(crate) fn value_relation_fit(
    value: &PreferenceExpr,
    relation: &ValueRelation,
    params: &[Interval],
) -> Result<Fit, EncloseFailure> {
    let gap = value_relation_gap(value, relation, params)?;
    let floor = -relation.tolerance;
    Ok(if gap.hi < floor {
        Fit::Outside
    } else if gap.lo >= floor {
        Fit::Inside
    } else {
        Fit::Boundary
    })
}
