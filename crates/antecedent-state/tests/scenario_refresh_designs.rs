//! The least-squares refresh on a dummy-coded categorical design and on a degenerate design.
//!
//! The refresh keeps the Gram matrix, cross-product and residual variance of `y = X b`. A
//! three-level treatment enters as an intercept and two dummy columns, so the coefficients are
//! the reference mean and the two level-minus-reference mean differences, which are written out
//! by hand below. A constant column (the same as the intercept) or a repeated covariate makes the
//! Gram singular: the coefficients are withheld (`None`) rather than returned from a
//! pseudo-inverse, and the refresh agrees with a fresh rebuild.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_state::{OlsRefresh, RefreshRow};

fn fit(design: &[(Vec<f64>, f64)]) -> OlsRefresh {
    let rows: Vec<RefreshRow> = design
        .iter()
        .enumerate()
        .map(|(id, (x, y))| RefreshRow::new(id as u64, x.clone(), *y))
        .collect();
    OlsRefresh::from_rows("scenario", design[0].0.len(), rows).unwrap()
}

#[test]
fn dummy_coded_levels_give_the_reference_mean_and_the_level_differences() {
    // Group means 1 (reference a), 4 (b) and 9 (c): coefficients [1, 3, 8].
    let groups: [(&[f64], [f64; 2]); 3] =
        [(&[0.0, 2.0], [0.0, 0.0]), (&[3.0, 5.0], [1.0, 0.0]), (&[8.0, 10.0], [0.0, 1.0])];
    let mut design = Vec::new();
    for (outcomes, dummies) in groups {
        for &y in outcomes {
            design.push((vec![1.0, dummies[0], dummies[1]], y));
        }
    }
    let refresh = fit(&design);
    let snapshot = refresh.snapshot();
    let beta = snapshot.beta.as_ref().expect("a full-rank dummy design has coefficients");
    for (got, want) in beta.iter().zip([1.0, 3.0, 8.0]) {
        assert!((got - want).abs() < 1e-9, "{beta:?}");
    }
    // Within-group variance: each group has deviations of one, so the residual sum of squares is
    // 6 over n - p = 3 degrees of freedom.
    assert!((snapshot.residual_variance.unwrap() - 2.0).abs() < 1e-9);
    refresh.check_against_fresh().unwrap();
}

#[test]
fn a_constant_or_repeated_column_withholds_the_coefficients_instead_of_inventing_them() {
    let outcome = |i: usize| 1.0 + 2.0 * (i % 5) as f64 + 0.1 * (i % 3) as f64;
    let constant: Vec<(Vec<f64>, f64)> = (0..20).map(|i| (vec![1.0, 2.0], outcome(i))).collect();
    let repeated: Vec<(Vec<f64>, f64)> = (0..20)
        .map(|i| {
            let x = (i % 5) as f64;
            (vec![1.0, x, x], outcome(i))
        })
        .collect();
    for (label, design) in [("constant column", constant), ("repeated covariate", repeated)] {
        let refresh = fit(&design);
        let snapshot = refresh.snapshot();
        assert!(snapshot.beta.is_none(), "{label}: coefficients from a singular Gram");
        refresh.check_against_fresh().unwrap();
    }
}
