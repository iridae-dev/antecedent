//! Evaluation of a decision [`UtilityExpr`] as a derived-quantity function: value, gradient,
//! interval enclosure and structure checks.
//!
//! The expression language is the one the decision contract already owns. This module adds no
//! operators; it reads the same tree for the three things a derived experimental quantity needs:
//! its value at a point, its gradient for propagating covariance (refused where the tree is not
//! differentiable), and an enclosure over a box of assumed values.
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::decision_contract::UtilityExpr;

/// Whether the tree contains a pointwise minimum or maximum, where the gradient is not defined.
pub(super) fn has_kink(expr: &UtilityExpr) -> bool {
    match expr {
        UtilityExpr::Const(_) | UtilityExpr::Input(_) => false,
        UtilityExpr::Neg(a) => has_kink(a),
        UtilityExpr::Add(a, b) | UtilityExpr::Sub(a, b) | UtilityExpr::Mul(a, b) => {
            has_kink(a) || has_kink(b)
        }
        UtilityExpr::Min(..) | UtilityExpr::Max(..) => true,
    }
}

/// Conservative structural check for an affine expression.
pub(super) fn is_affine(expr: &UtilityExpr) -> bool {
    fn degree(e: &UtilityExpr) -> usize {
        match e {
            UtilityExpr::Const(_) => 0,
            UtilityExpr::Input(_) => 1,
            UtilityExpr::Neg(a) => degree(a),
            UtilityExpr::Add(a, b) | UtilityExpr::Sub(a, b) => degree(a).max(degree(b)),
            UtilityExpr::Mul(a, b) => degree(a).saturating_add(degree(b)),
            UtilityExpr::Min(..) | UtilityExpr::Max(..) => 2,
        }
    }
    degree(expr) <= 1
}

/// Whether every input is read at most once and the tree has no kink, so the function is
/// multilinear in its inputs and its extremes over a box sit at the box's vertices.
pub(super) fn is_multilinear(expr: &UtilityExpr) -> bool {
    fn reads(expr: &UtilityExpr, counts: &mut Vec<usize>) {
        match expr {
            UtilityExpr::Const(_) => {}
            UtilityExpr::Input(k) => {
                if counts.len() <= *k {
                    counts.resize(*k + 1, 0);
                }
                counts[*k] += 1;
            }
            UtilityExpr::Neg(a) => reads(a, counts),
            UtilityExpr::Add(a, b)
            | UtilityExpr::Sub(a, b)
            | UtilityExpr::Mul(a, b)
            | UtilityExpr::Min(a, b)
            | UtilityExpr::Max(a, b) => {
                reads(a, counts);
                reads(b, counts);
            }
        }
    }
    let mut counts = Vec::new();
    reads(expr, &mut counts);
    !has_kink(expr) && counts.iter().all(|c| *c <= 1)
}

/// Value of the expression at `x`; `None` when an input is out of range or the value is not
/// finite.
pub(super) fn value(expr: &UtilityExpr, x: &[f64]) -> Option<f64> {
    let v = match expr {
        UtilityExpr::Const(c) => *c,
        UtilityExpr::Input(k) => *x.get(*k)?,
        UtilityExpr::Neg(a) => -value(a, x)?,
        UtilityExpr::Add(a, b) => value(a, x)? + value(b, x)?,
        UtilityExpr::Sub(a, b) => value(a, x)? - value(b, x)?,
        UtilityExpr::Mul(a, b) => value(a, x)? * value(b, x)?,
        UtilityExpr::Min(a, b) => value(a, x)?.min(value(b, x)?),
        UtilityExpr::Max(a, b) => value(a, x)?.max(value(b, x)?),
    };
    v.is_finite().then_some(v)
}

/// Value and gradient by forward accumulation; `None` for a kinked tree or a bad input.
pub(super) fn value_and_gradient(expr: &UtilityExpr, x: &[f64]) -> Option<(f64, Vec<f64>)> {
    if has_kink(expr) {
        return None;
    }
    fn dual(expr: &UtilityExpr, x: &[f64]) -> Option<(f64, Vec<f64>)> {
        let n = x.len();
        Some(match expr {
            UtilityExpr::Const(c) => (*c, vec![0.0; n]),
            UtilityExpr::Input(k) => {
                let mut g = vec![0.0; n];
                *g.get_mut(*k)? = 1.0;
                (*x.get(*k)?, g)
            }
            UtilityExpr::Neg(a) => {
                let (v, g) = dual(a, x)?;
                (-v, g.into_iter().map(|d| -d).collect())
            }
            UtilityExpr::Add(a, b) => {
                let ((va, ga), (vb, gb)) = (dual(a, x)?, dual(b, x)?);
                (va + vb, ga.iter().zip(&gb).map(|(p, q)| p + q).collect())
            }
            UtilityExpr::Sub(a, b) => {
                let ((va, ga), (vb, gb)) = (dual(a, x)?, dual(b, x)?);
                (va - vb, ga.iter().zip(&gb).map(|(p, q)| p - q).collect())
            }
            UtilityExpr::Mul(a, b) => {
                let ((va, ga), (vb, gb)) = (dual(a, x)?, dual(b, x)?);
                (va * vb, ga.iter().zip(&gb).map(|(p, q)| p * vb + va * q).collect())
            }
            UtilityExpr::Min(..) | UtilityExpr::Max(..) => return None,
        })
    }
    let (v, g) = dual(expr, x)?;
    (v.is_finite() && g.iter().all(|d| d.is_finite())).then_some((v, g))
}

/// A guaranteed enclosure of the expression over the box `lower <= x <= upper`.
///
/// Interval arithmetic: exact when every input is read once, otherwise a possibly loose outer
/// bound that never excludes an attainable value.
pub(super) fn enclosure(expr: &UtilityExpr, lower: &[f64], upper: &[f64]) -> Option<(f64, f64)> {
    let (lo, hi) = match expr {
        UtilityExpr::Const(c) => (*c, *c),
        UtilityExpr::Input(k) => (*lower.get(*k)?, *upper.get(*k)?),
        UtilityExpr::Neg(a) => {
            let (l, h) = enclosure(a, lower, upper)?;
            (-h, -l)
        }
        UtilityExpr::Add(a, b) => {
            let ((la, ha), (lb, hb)) = (enclosure(a, lower, upper)?, enclosure(b, lower, upper)?);
            (la + lb, ha + hb)
        }
        UtilityExpr::Sub(a, b) => {
            let ((la, ha), (lb, hb)) = (enclosure(a, lower, upper)?, enclosure(b, lower, upper)?);
            (la - hb, ha - lb)
        }
        UtilityExpr::Mul(a, b) => {
            let ((la, ha), (lb, hb)) = (enclosure(a, lower, upper)?, enclosure(b, lower, upper)?);
            let p = [la * lb, la * hb, ha * lb, ha * hb];
            (
                p.iter().copied().fold(f64::INFINITY, f64::min),
                p.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            )
        }
        UtilityExpr::Min(a, b) => {
            let ((la, ha), (lb, hb)) = (enclosure(a, lower, upper)?, enclosure(b, lower, upper)?);
            (la.min(lb), ha.min(hb))
        }
        UtilityExpr::Max(a, b) => {
            let ((la, ha), (lb, hb)) = (enclosure(a, lower, upper)?, enclosure(b, lower, upper)?);
            (la.max(lb), ha.max(hb))
        }
    };
    (lo.is_finite() && hi.is_finite()).then_some((lo, hi))
}

#[cfg(test)]
mod tests {
    use super::*;
    use UtilityExpr::{Const, Input};

    /// margin = frequency * (value - cost)
    fn margin() -> UtilityExpr {
        UtilityExpr::product(Input(0), UtilityExpr::difference(Input(1), Input(2)))
    }

    #[test]
    fn value_gradient_and_structure_of_a_margin() {
        let x = [2.0, 10.0, 4.0];
        assert_eq!(value(&margin(), &x), Some(12.0));
        let (v, g) = value_and_gradient(&margin(), &x).unwrap();
        assert_eq!(v, 12.0);
        assert_eq!(g, vec![6.0, 2.0, -2.0]);
        assert!(is_multilinear(&margin()));
        assert!(!is_multilinear(&UtilityExpr::product(Input(0), Input(0))));
        assert!(!is_multilinear(&UtilityExpr::maximum(Input(0), Const(0.0))));
        assert!(value_and_gradient(&UtilityExpr::maximum(Input(0), Const(0.0)), &x).is_none());
    }

    #[test]
    fn enclosure_is_exact_for_read_once_and_outer_otherwise() {
        let (lo, hi) = enclosure(&margin(), &[2.0, 10.0, 3.0], &[2.0, 10.0, 5.0]).unwrap();
        assert_eq!((lo, hi), (10.0, 14.0));
        // x * x over [-1, 2]: attainable [0, 4]; interval arithmetic encloses it with [-2, 4].
        let square = UtilityExpr::product(Input(0), Input(0));
        let (lo, hi) = enclosure(&square, &[-1.0], &[2.0]).unwrap();
        assert!(lo <= 0.0 && hi >= 4.0);
    }
}
