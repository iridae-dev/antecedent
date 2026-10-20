// SPDX-License-Identifier: MIT OR Apache-2.0
//! Verified invertibility of represented affine residual systems.
//! A numerical inverse is only a candidate. Outward interval arithmetic checks
//! ||I - R A||_infinity < 1 for the exact represented A = I-W; the Neumann
//! argument then proves invertibility independently of any numerical root.
use crate::equation_system::Interval;
use antecedent_kernels::gauss_jordan_solve;
/// Prove invertibility of I-W using an outward-rounded inverse residual.
/// Returns false when the bounded certificate fails, including ill conditioning.
/// This is no assertion about dynamic stability or contraction of W.
#[must_use]
pub fn certifies_affine_bijection(weights: &[f64], order: usize) -> bool {
    if order == 0 {
        return weights.is_empty();
    }
    if order > 128 || weights.len() != order * order || weights.iter().any(|x| !x.is_finite()) {
        return false;
    }
    let mut matrix: Vec<_> = (0..order)
        .flat_map(|i| {
            (0..order).map(move |j| {
                if i == j { 1.0 - weights[i * order + j] } else { -weights[i * order + j] }
            })
        })
        .collect();
    let mut inverse = vec![0.0; order * order];
    for i in 0..order {
        inverse[i * order + i] = 1.0;
    }
    if !gauss_jordan_solve(&mut matrix, &mut inverse, order, order)
        || inverse.iter().any(|x| !x.is_finite())
    {
        return false;
    }
    for i in 0..order {
        let mut norm = Interval { lower: 0.0, upper: 0.0 };
        for j in 0..order {
            let mut residual = Interval {
                lower: if i == j { 1.0 } else { 0.0 },
                upper: if i == j { 1.0 } else { 0.0 },
            };
            for k in 0..order {
                let a = Interval {
                    lower: if k == j { 1.0 } else { 0.0 },
                    upper: if k == j { 1.0 } else { 0.0 },
                }
                .add_outward(Interval {
                    lower: -weights[k * order + j],
                    upper: -weights[k * order + j],
                });
                residual = residual.add_outward(a.scale_outward(-inverse[k * order + i]));
            }
            let magnitude = residual.lower.abs().max(residual.upper.abs());
            norm = norm.add_outward(Interval { lower: 0.0, upper: magnitude });
        }
        if !norm.upper.is_finite() || norm.upper >= 1.0 {
            return false;
        }
    }
    true
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn affine_bijection_is_separate_from_contraction_stability_and_numeric_success() {
        assert!(certifies_affine_bijection(&[0.0, 2.0, -2.0, 0.0], 2));
        assert!(!certifies_affine_bijection(&[0.0, 1.0, 1.0, 0.0], 2));
        assert!(!certifies_affine_bijection(&[f64::NAN], 1));
        assert!(certifies_affine_bijection(&[2.0], 1));
        assert!(!certifies_affine_bijection(&[1.0], 1));
    }
}
