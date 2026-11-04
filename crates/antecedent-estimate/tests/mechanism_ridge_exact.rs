//! Posterior mean of a population's mechanism coefficients, against the exact ridge solution
//! computed here from the raw rows.
//!
//! The route under test is `test_mechanism_discrepancy` (which builds the source summary from the
//! rows) followed by `mechanism_coefficient_posterior`. The model is a normal prior of scale `s`
//! on every parent slope, an unpenalized intercept, and the OLS residual variance
//! `sigma2 = RSS / (n - p)` plugged in. The conjugate posterior mean is the ridge solution
//! `(X'X + diag(0, lambda, ..., lambda)) beta = X'y` with `lambda = sigma2 / s^2`.
//!
//! The expected value is computed without the production summary or solver, by data
//! augmentation: one pseudo-observation `y = 0` with regressor `sqrt(lambda)` on each slope
//! column is appended to the rows, and ordinary least squares is solved on the augmented rows
//! (Gaussian elimination on the augmented normal equations). The one-parent case also has a
//! closed form written out by hand: with `x = 0..3`, `y = 1, 3, 2, 4`, `X'X = [[4, 6], [6, 14]]`,
//! `X'y = [10, 19]`, `beta_ols = (1.3, 0.8)`, `RSS = 1.8`, `sigma2 = 0.9`, the ridge solution is
//! `beta_1 = 16 / (20 + 4 lambda)` and `beta_0 = (26 + 10 lambda) / (20 + 4 lambda)`.
//!
//! The prior scales are moderate (0.5 to 4), so the answer sits visibly between the OLS fit and
//! the prior mean. The tolerance 1e-10 is the rounding of two small elimination solves.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::mechanism_discrepancy::{
    DiscrepancyOptions, MechanismMeasurement, ParentSpec, PopulationSample, SampleDependence,
    test_mechanism_discrepancy,
};
use antecedent_estimate::posterior_mean::{PosteriorRequest, mechanism_coefficient_posterior};

const TOLERANCE: f64 = 1e-10;

fn close(actual: f64, expected: f64, what: &str) {
    assert!((actual - expected).abs() <= TOLERANCE, "{what}: {actual} vs {expected}");
}

fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&i, &j| a[i][col].abs().partial_cmp(&a[j][col].abs()).unwrap())
            .unwrap();
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in col + 1..n {
            let factor = a[row][col] / a[col][col];
            for k in col..n {
                let delta = factor * a[col][k];
                a[row][k] -= delta;
            }
            let delta = factor * b[col];
            b[row] -= delta;
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let tail: f64 = (row + 1..n).map(|k| a[row][k] * x[k]).sum();
        x[row] = (b[row] - tail) / a[row][row];
    }
    x
}

/// Ordinary least squares on explicit design rows (intercept column included).
fn ols(rows: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
    let p = rows[0].len();
    let mut gram = vec![vec![0.0; p]; p];
    let mut rhs = vec![0.0; p];
    for (row, target) in rows.iter().zip(y) {
        for i in 0..p {
            rhs[i] += row[i] * target;
            for j in 0..p {
                gram[i][j] += row[i] * row[j];
            }
        }
    }
    solve(gram, rhs)
}

/// Ridge posterior mean by data augmentation, with the OLS residual variance plugged in.
fn augmented_ridge(parents: &[&[f64]], y: &[f64], scale: f64) -> Vec<f64> {
    let n = y.len();
    let p = parents.len() + 1;
    let rows: Vec<Vec<f64>> = (0..n)
        .map(|i| std::iter::once(1.0).chain(parents.iter().map(|column| column[i])).collect())
        .collect();
    let fit = ols(&rows, y);
    let rss: f64 = rows
        .iter()
        .zip(y)
        .map(|(row, target)| {
            let predicted: f64 = row.iter().zip(&fit).map(|(x, b)| x * b).sum();
            (target - predicted).powi(2)
        })
        .sum();
    let sigma2 = rss / (n - p) as f64;
    let lambda = sigma2 / (scale * scale);
    let mut augmented = rows;
    let mut targets = y.to_vec();
    for slope in 1..p {
        let mut pseudo = vec![0.0; p];
        pseudo[slope] = lambda.sqrt();
        augmented.push(pseudo);
        targets.push(0.0);
    }
    ols(&augmented, &targets)
}

fn measurement(parents: &[&str]) -> MechanismMeasurement {
    MechanismMeasurement {
        node: "V".into(),
        node_unit: "mg".into(),
        parents: parents
            .iter()
            .map(|name| ParentSpec { name: (*name).into(), unit: "u".into() })
            .collect(),
        protocol_id: "protocol-1".into(),
    }
}

fn sample(label: &str, y: &[f64], parents: &[(&str, &[f64])]) -> PopulationSample {
    let names: Vec<&str> = parents.iter().map(|(name, _)| *name).collect();
    PopulationSample {
        label: label.into(),
        measurement: measurement(&names),
        outcome: y.to_vec(),
        parent_values: parents.iter().map(|(_, values)| values.to_vec()).collect(),
        unit_ids: vec![],
    }
}

fn options() -> DiscrepancyOptions {
    DiscrepancyOptions {
        compare_intercept: true,
        alpha: 0.05,
        power: 0.8,
        dependence: SampleDependence::Independent,
    }
}

/// Source posterior mean through the discrepancy route, for each prior scale.
fn route_posterior(source: &PopulationSample, target: &PopulationSample, scale: f64) -> Vec<f64> {
    let result = test_mechanism_discrepancy(source, target, &options()).unwrap();
    mechanism_coefficient_posterior(
        &result.source_summary,
        PosteriorRequest::IndependentNormal { scale },
    )
    .unwrap()
}

#[test]
fn one_parent_mechanism_posterior_is_the_exact_ridge_solution() {
    let x = [0.0, 1.0, 2.0, 3.0];
    let ys = [1.0, 3.0, 2.0, 4.0];
    let yt = [1.5, 4.0, 3.5, 6.0];
    let source = sample("source", &ys, &[("x", &x)]);
    let target = sample("target", &yt, &[("x", &x)]);
    let mut previous_slope = 0.0;
    for scale in [0.5, 1.0, 2.0, 4.0] {
        let posterior = route_posterior(&source, &target, scale);
        // Hand closed form: lambda = sigma2 / s^2 with sigma2 = 1.8 / 2.
        let lambda = 0.9 / (scale * scale);
        let det = 20.0 + 4.0 * lambda;
        close(posterior[0], (26.0 + 10.0 * lambda) / det, &format!("intercept, s = {scale}"));
        close(posterior[1], 16.0 / det, &format!("slope, s = {scale}"));
        // The independent data-augmentation solve agrees.
        let augmented = augmented_ridge(&[&x], &ys, scale);
        close(posterior[0], augmented[0], "intercept vs augmented rows");
        close(posterior[1], augmented[1], "slope vs augmented rows");
        // The slope 16 / (20 + 4 lambda) lies strictly between 0 and the OLS slope 0.8, and the
        // shrinkage grows as the prior scale falls (the scales run upward).
        assert!(posterior[1] > 0.0 && posterior[1] < 0.8, "{posterior:?}");
        assert!(posterior[1] > previous_slope, "slope rises towards OLS with the scale");
        previous_slope = posterior[1];
    }
}

#[test]
fn two_parent_mechanism_posterior_is_the_exact_ridge_solution() {
    let a = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0];
    let b = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
    let ys = [3.0, 2.0, 5.0, 1.0, 0.0, 4.5];
    let yt = [4.5, 2.5, 6.5, 1.0, 0.0, 5.0];
    let source = sample("source", &ys, &[("a", &a), ("b", &b)]);
    let target = sample("target", &yt, &[("a", &a), ("b", &b)]);
    for scale in [0.5, 1.0, 3.0] {
        let posterior = route_posterior(&source, &target, scale);
        let augmented = augmented_ridge(&[&a, &b], &ys, scale);
        assert_eq!(posterior.len(), 3);
        for (index, (got, want)) in posterior.iter().zip(&augmented).enumerate() {
            close(*got, *want, &format!("coefficient {index}, s = {scale}"));
        }
    }
    // The slope vector at a small prior scale is strictly shorter than at a large one.
    let norm = |v: &[f64]| v[1] * v[1] + v[2] * v[2];
    let tight = route_posterior(&source, &target, 0.5);
    let loose = route_posterior(&source, &target, 3.0);
    assert!(norm(&tight) < norm(&loose));
}
