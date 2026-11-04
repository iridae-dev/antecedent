//! Posterior means published by the joint-treatment, categorical, local-quadratic and
//! effect-constancy routes, against closed forms computed in this file from the raw rows.
//!
//! Each route is run through its own public entry point (design summary, kernel weights,
//! contrast construction included). The expected value is an independent ridge solution built
//! here by Gaussian elimination on the rows (or the hand normal-normal update), not a call to
//! the production kernel, at a prior scale that moves the answer visibly off the least-squares
//! fit.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::categorical_treatment::{
    CategoricalTreatmentInput, CategoricalTreatmentSpec, LevelScale,
};
use antecedent_estimate::effect_constancy::{
    ContrastFamily, EffectEstimandIdentity, PartitionDependence, PartitionEstimate,
    PartitionSupport, test_effect_constancy,
};
use antecedent_estimate::posterior_mean::{
    LocalQuadraticNamespace, PosteriorRequest, ScalarNamespace, categorical_treatment_posterior,
    known_variance_posterior, local_quadratic_posterior, vector_treatment_posterior,
};
use antecedent_estimate::vector_treatment::{
    TreatmentColumn, VectorCovariance, VectorTreatmentInput,
};

/// Solve `a x = b` by Gaussian elimination with partial pivoting.
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

/// Weighted ridge with the first `unpenalized` columns free: the residual variance is the
/// weighted residual sum of squares of the unpenalized fit over `denominator`, and the penalty
/// is that variance over `scale^2`.
fn ridge(
    rows: &[Vec<f64>],
    y: &[f64],
    weights: &[f64],
    unpenalized: usize,
    denominator: f64,
    scale: f64,
) -> Vec<f64> {
    let p = rows[0].len();
    let mut gram = vec![vec![0.0; p]; p];
    let mut rhs = vec![0.0; p];
    for ((row, &yi), &w) in rows.iter().zip(y).zip(weights) {
        for j in 0..p {
            rhs[j] += w * row[j] * yi;
            for k in 0..p {
                gram[j][k] += w * row[j] * row[k];
            }
        }
    }
    let ols = solve(gram.clone(), rhs.clone());
    let rss: f64 = rows
        .iter()
        .zip(y)
        .zip(weights)
        .map(|((row, &yi), &w)| {
            let fitted: f64 = row.iter().zip(&ols).map(|(a, b)| a * b).sum();
            w * (yi - fitted) * (yi - fitted)
        })
        .sum();
    let penalty = rss / denominator / (scale * scale);
    for (j, row) in gram.iter_mut().enumerate().skip(unpenalized) {
        row[j] += penalty;
    }
    solve(gram, rhs)
}

fn treatment(name: &str, values: &[f64]) -> TreatmentColumn<'static> {
    TreatmentColumn {
        name: name.to_owned().into(),
        values: values.to_vec().into(),
        adjustment_set: vec![],
        row_snapshot: "snap".into(),
    }
}

#[test]
fn vector_treatment_posterior_equals_the_ridge_solution_built_from_the_rows() {
    let y = [3.0, 2.0, 5.0, 1.0, 0.0, 4.0];
    let t1 = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0];
    let t2 = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
    let input = VectorTreatmentInput {
        outcome: y.to_vec().into(),
        row_snapshot: "snap".into(),
        adjustment: vec![],
        treatments: vec![treatment("t1", &t1), treatment("t2", &t2)],
    };
    let rows: Vec<Vec<f64>> = (0..6).map(|i| vec![1.0, t1[i], t2[i]]).collect();
    let ones = vec![1.0; 6];
    for scale in [0.5, 1.0, 3.0] {
        let got = vector_treatment_posterior(&input, PosteriorRequest::IndependentNormal { scale })
            .unwrap();
        // The route returns the treatment coefficients; the intercept is unpenalized.
        let want = ridge(&rows, &y, &ones, 1, 3.0, scale);
        assert_eq!(got.len(), 2);
        for (g, w) in got.iter().zip(&want[1..]) {
            assert!((g - w).abs() < 1e-9, "scale {scale}: {got:?} vs {want:?}");
        }
    }
    // The prior moves the answer: the shrunk fit is not the least-squares fit (2.75, 0.75).
    let shrunk =
        vector_treatment_posterior(&input, PosteriorRequest::IndependentNormal { scale: 0.5 })
            .unwrap();
    assert!((shrunk[0] - 2.75).abs() > 0.05, "{shrunk:?}");
}

#[test]
fn categorical_treatment_posterior_equals_the_dummy_ridge_solution_built_from_the_rows() {
    let data = [
        ("c", 9.0),
        ("a", 1.0),
        ("b", 4.0),
        ("c", 7.0),
        ("b", 6.0),
        ("a", 3.0),
        ("c", 11.0),
        ("b", 5.0),
        ("c", 9.0),
    ];
    let input = CategoricalTreatmentInput {
        outcome: data.iter().map(|row| row.1).collect(),
        row_snapshot: "snap".into(),
        adjustment: vec![],
        levels: data.iter().map(|row| row.0.to_string()).collect(),
    };
    let spec = CategoricalTreatmentSpec {
        declared_levels: ["a", "b", "c"].into_iter().map(str::to_string).collect(),
        scale: LevelScale::Unordered,
        reference: "a".into(),
        min_level_rows: 2,
        pairwise: vec![],
        monotonicity: None,
        covariance: VectorCovariance::ModelBased,
    };
    let y: Vec<f64> = data.iter().map(|row| row.1).collect();
    let rows: Vec<Vec<f64>> = data
        .iter()
        .map(|row| vec![1.0, f64::from(u8::from(row.0 == "b")), f64::from(u8::from(row.0 == "c"))])
        .collect();
    let ones = vec![1.0; rows.len()];
    for scale in [0.5, 2.0] {
        let got = categorical_treatment_posterior(
            &input,
            &spec,
            PosteriorRequest::IndependentNormal { scale },
        )
        .unwrap();
        let want = ridge(&rows, &y, &ones, 1, 6.0, scale);
        assert_eq!(got.len(), 2);
        for (g, w) in got.iter().zip(&want[1..]) {
            assert!((g - w).abs() < 1e-9, "scale {scale}: {got:?} vs {want:?}");
        }
    }
    // The least-squares level differences are the group-mean differences (3, 7); a prior of
    // scale 0.5 shrinks them.
    let shrunk = categorical_treatment_posterior(
        &input,
        &spec,
        PosteriorRequest::IndependentNormal { scale: 0.5 },
    )
    .unwrap();
    assert!(shrunk[0] < 3.0 - 0.1 && shrunk[1] < 7.0 - 0.1, "{shrunk:?}");
}

fn local_quadratic_expected(
    dose: &[f64],
    outcome: &[f64],
    at: f64,
    bandwidth: f64,
    scale: f64,
) -> (f64, f64) {
    let rows: Vec<Vec<f64>> = dose
        .iter()
        .map(|&x| {
            let dx = x - at;
            vec![1.0, dx, dx * dx]
        })
        .collect();
    let weights: Vec<f64> =
        dose.iter().map(|&x| (-0.5 * ((x - at) / bandwidth).powi(2)).exp()).collect();
    let denominator = weights.iter().sum::<f64>() - 3.0;
    let beta = ridge(&rows, outcome, &weights, 1, denominator, scale);
    (beta[0], beta[1])
}

#[test]
fn local_quadratic_posterior_equals_the_weighted_ridge_solution_for_both_routes() {
    let dose = [0.0, 0.7, 1.1, 1.9, 2.4, 3.0, 3.8, 4.5, 5.1, 6.0];
    let outcome = [0.4, 1.3, 1.1, 2.9, 3.4, 5.2, 8.0, 9.9, 12.8, 17.5];
    for namespace in [LocalQuadraticNamespace::QuadraticDose, LocalQuadraticNamespace::DoseGrid] {
        for (at, bandwidth, scale) in [(3.0, 1.5, 0.7), (2.0, 2.5, 3.0)] {
            let (level, derivative) = local_quadratic_posterior(
                &dose,
                &outcome,
                at,
                bandwidth,
                PosteriorRequest::IndependentNormal { scale },
                namespace,
            )
            .unwrap();
            let (want_level, want_derivative) =
                local_quadratic_expected(&dose, &outcome, at, bandwidth, scale);
            assert!(
                (level - want_level).abs() < 1e-9 && (derivative - want_derivative).abs() < 1e-9,
                "{namespace:?} at {at}: ({level}, {derivative}) vs ({want_level}, {want_derivative})"
            );
        }
    }
}

fn part(label: &str, effect: f64, se: f64) -> PartitionEstimate {
    PartitionEstimate {
        label: label.into(),
        coordinate: format!("period:{label}"),
        support: PartitionSupport::Supported,
        estimand: EffectEstimandIdentity {
            estimand: "ate_difference".into(),
            units: "outcome_units".into(),
            regime: "treat_vs_control".into(),
            population: "all_observed_h2".into(),
        },
        effect,
        standard_error: se,
    }
}

#[test]
fn effect_constancy_posterior_equals_the_hand_normal_update_of_the_route_contrast() {
    // Independent partitions: Var(e1 - e2) = se1^2 + se2^2 by hand.
    for (e1, s1, e2, s2, scale) in [(1.0, 0.5, 2.0, 0.5, 1.0), (4.0, 0.5, 1.5, 0.3, 2.0)] {
        let result = test_effect_constancy(
            &[part("p1", e1, s1), part("p2", e2, s2)],
            &PartitionDependence::Independent,
            &ContrastFamily::AllPairs,
            0.05,
        )
        .unwrap();
        let contrast = &result.contrasts[0];
        assert!((contrast.difference.abs() - (e1 - e2).abs()).abs() < 1e-12);
        let variance = s1 * s1 + s2 * s2;
        let precision = 1.0 / variance;
        let prior = 1.0 / (scale * scale);
        let want = precision / (precision + prior) * contrast.difference;
        let got = known_variance_posterior(
            contrast.difference,
            variance,
            PosteriorRequest::IndependentNormal { scale },
            ScalarNamespace::Constancy,
        )
        .unwrap();
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        assert!(got.abs() < contrast.difference.abs() - 0.01, "the prior must shrink the contrast");
    }
}
