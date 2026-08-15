//! Sharp finite binary-IV bounds by latent response-type enumeration.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::needless_range_loop, clippy::cast_precision_loss)]

use antecedent_core::IdentifiedSet;

use crate::error::IdentificationError;

const TYPES: usize = 16;
const RANK: usize = 7;
const TOL: f64 = 1e-9;

/// Observed binary-IV law `P(Y=y,D=d | Z=z)`.
///
/// Cell order within each instrument arm is `(y,d) = (0,0), (1,0), (0,1), (1,1)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BinaryIvLaw {
    /// Conditional cell probabilities by instrument arm.
    pub cells: [[f64; 4]; 2],
}

impl BinaryIvLaw {
    /// Validate finite non-negative cells summing to one within each instrument arm.
    pub fn validate(&self) -> Result<(), IdentificationError> {
        for arm in self.cells {
            if arm.iter().any(|p| !p.is_finite() || *p < 0.0)
                || (arm.iter().sum::<f64>() - 1.0).abs() > TOL
            {
                return Err(IdentificationError::unsupported(
                    "binary-IV cells must be finite, non-negative, and sum to one by Z arm",
                ));
            }
        }
        Ok(())
    }
}

/// Compute sharp Balke–Pearl-style bounds on `E[Y(1)-Y(0)]` without monotonicity.
///
/// The implementation solves the exact 16-type latent response-program. It assumes randomized
/// instrument assignment/exogeneity and exclusion through the response-type model; those claims
/// must be recorded by the caller in the enclosing identification artifact.
///
/// Returns an error naming the Balke–Pearl instrumental inequality specifically when the
/// response-type program is infeasible, which is distinct from (and reported separately from)
/// malformed input — the latter is rejected earlier by [`BinaryIvLaw::validate`].
pub fn binary_iv_ate_bounds(law: BinaryIvLaw) -> Result<IdentifiedSet<f64>, IdentificationError> {
    law.validate()?;
    let Some((lower, upper)) = feasible_ate_bounds(law) else {
        return Err(IdentificationError::unsupported(
            "binary-IV observed law violates the Balke\u{2013}Pearl instrumental inequality: no \
             response-type distribution over the 16 latent types reproduces the observed cells",
        ));
    };
    IdentifiedSet::try_new(lower.max(-1.0), upper.min(1.0))
        .map_err(IdentificationError::unsupported)
}

/// Test the Balke–Pearl instrumental inequality for a binary instrument/treatment/outcome law.
///
/// For binary `Z`, `D`, `Y`, the observed law is consistent with *some* distribution over the 16
/// latent response types (equivalently, satisfies the instrumental inequality) if and only if the
/// same feasibility search used by [`binary_iv_ate_bounds`] finds at least one feasible vertex.
/// This predicate shares that search rather than re-deriving it, so the two entry points cannot
/// drift apart.
pub fn binary_iv_inequality_satisfied(law: BinaryIvLaw) -> Result<bool, IdentificationError> {
    law.validate()?;
    Ok(feasible_ate_bounds(law).is_some())
}

/// Enumerate response-type basic feasible solutions and return the sharp `E[Y(1)-Y(0)]` range,
/// or `None` if the observed law admits no feasible response-type distribution at all (i.e. the
/// Balke–Pearl instrumental inequality is violated).
fn feasible_ate_bounds(law: BinaryIvLaw) -> Option<(f64, f64)> {
    let (a, b) = constraint_system(law);
    let objective: [f64; TYPES] = std::array::from_fn(|kind| {
        let (y0, y1, _, _) = decode_type(kind);
        f64::from(y1) - f64::from(y0)
    });

    let mut lower = f64::INFINITY;
    let mut upper = f64::NEG_INFINITY;
    let mut basis = [0usize; RANK];
    enumerate_bases(0, 0, &mut basis, &mut |cols| {
        let mut square = [[0.0; RANK]; RANK];
        for row in 0..RANK {
            for j in 0..RANK {
                square[row][j] = a[row][cols[j]];
            }
        }
        let Some(solution) = solve(square, b) else {
            return;
        };
        if solution.iter().any(|q| *q < -TOL || !q.is_finite()) {
            return;
        }
        let mut q = [0.0; TYPES];
        for (j, &col) in cols.iter().enumerate() {
            q[col] = solution[j].max(0.0);
        }
        if !matches_observed(&q, law) {
            return;
        }
        let value: f64 = q.iter().zip(objective).map(|(weight, effect)| weight * effect).sum();
        lower = lower.min(value);
        upper = upper.max(value);
    });

    if !lower.is_finite() || !upper.is_finite() {
        return None;
    }
    Some((lower, upper))
}

fn constraint_system(law: BinaryIvLaw) -> ([[f64; TYPES]; RANK], [f64; RANK]) {
    let mut a = [[0.0; TYPES]; RANK];
    let mut b = [0.0; RANK];
    // Three cells per Z arm plus normalization are an independent representation.
    for z in 0..2 {
        for cell in 0..3 {
            let row = z * 3 + cell;
            b[row] = law.cells[z][cell];
            for kind in 0..TYPES {
                let (y0, y1, d0, d1) = decode_type(kind);
                let d = if z == 0 { d0 } else { d1 };
                let y = if d == 0 { y0 } else { y1 };
                if cell_index(y, d) == cell {
                    a[row][kind] = 1.0;
                }
            }
        }
    }
    for kind in 0..TYPES {
        a[6][kind] = 1.0;
    }
    b[6] = 1.0;
    (a, b)
}

fn decode_type(kind: usize) -> (u8, u8, u8, u8) {
    (
        u8::from(kind & 1 != 0),
        u8::from((kind >> 1) & 1 != 0),
        u8::from((kind >> 2) & 1 != 0),
        u8::from((kind >> 3) & 1 != 0),
    )
}

fn cell_index(y: u8, d: u8) -> usize {
    usize::from(d) * 2 + usize::from(y)
}

fn matches_observed(q: &[f64; TYPES], law: BinaryIvLaw) -> bool {
    for z in 0..2 {
        let mut got = [0.0; 4];
        for kind in 0..TYPES {
            let (y0, y1, d0, d1) = decode_type(kind);
            let d = if z == 0 { d0 } else { d1 };
            let y = if d == 0 { y0 } else { y1 };
            got[cell_index(y, d)] += q[kind];
        }
        if got.iter().zip(law.cells[z]).any(|(x, y)| (*x - y).abs() > 1e-7) {
            return false;
        }
    }
    true
}

fn enumerate_bases(
    depth: usize,
    start: usize,
    basis: &mut [usize; RANK],
    visit: &mut impl FnMut(&[usize; RANK]),
) {
    if depth == RANK {
        visit(basis);
        return;
    }
    let remaining = RANK - depth;
    for col in start..=TYPES - remaining {
        basis[depth] = col;
        enumerate_bases(depth + 1, col + 1, basis, visit);
    }
}

fn solve(mut a: [[f64; RANK]; RANK], mut b: [f64; RANK]) -> Option<[f64; RANK]> {
    for col in 0..RANK {
        let pivot = (col..RANK).max_by(|&i, &j| {
            a[i][col].abs().partial_cmp(&a[j][col].abs()).unwrap_or(std::cmp::Ordering::Equal)
        })?;
        if a[pivot][col].abs() <= TOL {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        let scale = a[col][col];
        for j in col..RANK {
            a[col][j] /= scale;
        }
        b[col] /= scale;
        for row in 0..RANK {
            if row == col {
                continue;
            }
            let factor = a[row][col];
            for j in col..RANK {
                a[row][j] -= factor * a[col][j];
            }
            b[row] -= factor * b[col];
        }
    }
    Some(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_frozen_bpbounds_table1_oracle() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../conformance/response/binary_iv_balke_pearl_table1/expected.json"
        ))
        .unwrap();
        let arms = fixture["cells"].as_array().unwrap();
        let mut cells = [[0.0; 4]; 2];
        for (z, arm) in arms.iter().enumerate() {
            for (cell, value) in arm.as_array().unwrap().iter().enumerate() {
                cells[z][cell] = value.as_f64().unwrap();
            }
        }
        let bounds = binary_iv_ate_bounds(BinaryIvLaw { cells }).unwrap();
        let expected = &fixture["expected"];
        let atol = fixture["tolerance"]["atol"].as_f64().unwrap();
        assert!((bounds.lower - expected["lower"].as_f64().unwrap()).abs() <= atol);
        assert!((bounds.upper - expected["upper"].as_f64().unwrap()).abs() <= atol);
        assert_eq!(
            binary_iv_inequality_satisfied(BinaryIvLaw { cells }).unwrap(),
            expected["iv_inequality_satisfied"].as_bool().unwrap()
        );
    }

    #[test]
    fn perfect_compliance_deterministic_effect_is_point_identified() {
        let law = BinaryIvLaw { cells: [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]] };
        let bounds = binary_iv_ate_bounds(law).unwrap();
        assert!((bounds.lower - 1.0).abs() < 1e-9);
        assert!((bounds.upper - 1.0).abs() < 1e-9);
    }

    #[test]
    fn never_takers_leave_treated_potential_outcome_unknown() {
        let law = BinaryIvLaw { cells: [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]] };
        let bounds = binary_iv_ate_bounds(law).unwrap();
        assert!((bounds.lower - 0.0).abs() < 1e-9);
        assert!((bounds.upper - 1.0).abs() < 1e-9);
    }

    #[test]
    fn satisfied_law_is_reported_by_the_shared_predicate() {
        // Perfect compliance with a deterministic effect: every individual has D = Z and
        // Y = D, so the degenerate response-type distribution (all mass on the single type
        // y0=0, y1=1, d0=0, d1=1) reproduces both arms exactly. The inequality holds.
        let law = BinaryIvLaw { cells: [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]] };
        assert!(binary_iv_inequality_satisfied(law).unwrap());
        // The bounds entry point must agree, since it shares the same feasibility search.
        assert!(binary_iv_ate_bounds(law).is_ok());
    }

    #[test]
    fn violated_law_is_rejected_by_bounds_and_the_shared_predicate() {
        // Z=0 arm puts all mass on (Y=1,D=0): every observed type must have d0=0 and y0=1.
        // Z=1 arm puts all mass on (Y=0,D=0): every observed type must have d1=0 and y0=0.
        // Since exogeneity requires a single population-wide response-type distribution to
        // explain both arms, and the same y0 bit cannot be both 1 (forced by the Z=0 arm)
        // and 0 (forced by the Z=1 arm), no response-type distribution over the 16 latent
        // types can reproduce this law: the remaining probability mass has nowhere to go
        // that satisfies both constraints simultaneously. This is a hand-verified violation
        // of the Balke-Pearl instrumental inequality, not a malformed-input case (the law
        // is a valid probability distribution within each Z arm).
        let law = BinaryIvLaw { cells: [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]] };
        law.validate().expect("law is a well-formed probability distribution per arm");

        assert!(!binary_iv_inequality_satisfied(law).unwrap());

        let err = binary_iv_ate_bounds(law).unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("instrumental inequality"),
            "error should name the instrumental inequality distinctly from malformed input: {message}"
        );
    }
}
