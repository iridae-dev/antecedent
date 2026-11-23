//! Interval enclosure of SDE expressions and their derivatives.

use crate::mechanism_expr::MechanismExpression;

use super::{SdeError, bad};

#[derive(Clone, Copy, Debug)]
pub(super) struct Iv {
    pub(super) lo: f64,
    pub(super) hi: f64,
}
const ZERO: Iv = Iv { lo: 0.0, hi: 0.0 };

fn fmul(a: f64, b: f64) -> f64 {
    if a == 0.0 || b == 0.0 { 0.0 } else { a * b }
}
fn sane(v: Iv) -> Iv {
    if v.lo.is_nan() || v.hi.is_nan() { Iv { lo: f64::NEG_INFINITY, hi: f64::INFINITY } } else { v }
}
fn iadd(a: Iv, b: Iv) -> Iv {
    sane(Iv { lo: a.lo + b.lo, hi: a.hi + b.hi })
}
fn imul(a: Iv, b: Iv) -> Iv {
    let p = [fmul(a.lo, b.lo), fmul(a.lo, b.hi), fmul(a.hi, b.lo), fmul(a.hi, b.hi)];
    sane(Iv {
        lo: p.iter().copied().fold(f64::INFINITY, f64::min),
        hi: p.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    })
}
pub(super) fn imax_abs(a: Iv) -> f64 {
    a.lo.abs().max(a.hi.abs())
}

fn softplus(x: f64) -> f64 {
    if x > 30.0 { x } else { x.exp().ln_1p() }
}
fn logistic(x: f64) -> f64 {
    if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { x.exp() / (1.0 + x.exp()) }
}

/// Interval enclosure of an expression and of its gradient over a coordinate box.
pub(super) fn enclose(
    expr: &MechanismExpression,
    boxes: &[Iv],
    theta: &[f64],
    depth: usize,
) -> Result<(Iv, Vec<Iv>), SdeError> {
    if depth > 48 {
        return Err(bad("SDE expression depth"));
    }
    let n = boxes.len();
    match expr {
        MechanismExpression::Constant { value } => {
            Ok((Iv { lo: *value, hi: *value }, vec![ZERO; n]))
        }
        MechanismExpression::Parameter { parameter } => {
            let v = *theta.get(*parameter).ok_or_else(|| bad("SDE parameter binding"))?;
            Ok((Iv { lo: v, hi: v }, vec![ZERO; n]))
        }
        MechanismExpression::Variable { variable } => {
            let b = *boxes.get(*variable).ok_or_else(|| bad("SDE variable binding"))?;
            let mut g = vec![ZERO; n];
            g[*variable] = Iv { lo: 1.0, hi: 1.0 };
            Ok((b, g))
        }
        MechanismExpression::Add { terms } => {
            let mut acc = (ZERO, vec![ZERO; n]);
            for term in terms {
                let (v, g) = enclose(term, boxes, theta, depth + 1)?;
                acc.0 = iadd(acc.0, v);
                for (a, b) in acc.1.iter_mut().zip(g) {
                    *a = iadd(*a, b);
                }
            }
            Ok(acc)
        }
        MechanismExpression::Multiply { factors } => {
            let mut acc = (Iv { lo: 1.0, hi: 1.0 }, vec![ZERO; n]);
            for factor in factors {
                let (v, g) = enclose(factor, boxes, theta, depth + 1)?;
                let mut next = vec![ZERO; n];
                for j in 0..n {
                    next[j] = iadd(imul(acc.1[j], v), imul(acc.0, g[j]));
                }
                acc = (imul(acc.0, v), next);
            }
            Ok(acc)
        }
        MechanismExpression::Tanh { argument } => {
            let (v, g) = enclose(argument, boxes, theta, depth + 1)?;
            let t = Iv { lo: v.lo.tanh(), hi: v.hi.tanh() };
            let (tt_lo, tt_hi) = if t.lo >= 0.0 {
                (t.lo * t.lo, t.hi * t.hi)
            } else if t.hi <= 0.0 {
                (t.hi * t.hi, t.lo * t.lo)
            } else {
                (0.0, (t.lo * t.lo).max(t.hi * t.hi))
            };
            let d = Iv { lo: 1.0 - tt_hi, hi: 1.0 - tt_lo };
            Ok((t, g.into_iter().map(|x| imul(d, x)).collect()))
        }
        MechanismExpression::Sin { argument } => {
            let (_, g) = enclose(argument, boxes, theta, depth + 1)?;
            let unit = Iv { lo: -1.0, hi: 1.0 };
            Ok((unit, g.into_iter().map(|x| imul(unit, x)).collect()))
        }
        MechanismExpression::Softplus { argument } => {
            let (v, g) = enclose(argument, boxes, theta, depth + 1)?;
            let value = Iv { lo: softplus(v.lo), hi: softplus(v.hi) };
            let d = Iv { lo: logistic(v.lo), hi: logistic(v.hi) };
            Ok((value, g.into_iter().map(|x| imul(d, x)).collect()))
        }
    }
}
