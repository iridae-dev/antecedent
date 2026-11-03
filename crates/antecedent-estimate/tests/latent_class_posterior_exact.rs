//! The latent-class effect posterior means, from the route's retained per-class designs,
//! against a ridge solution built here from the raw class rows.
//!
//! The data are the exactly balanced, perfectly separated two-class design of
//! `latent_class_effects.rs`, so every responsibility is 0 or 1 and each class's weighted normal
//! equations are those of its own rows. The expected treatment coefficient of each class is the
//! ridge solution of `y ~ 1 + a + x` on that class's raw rows (Gaussian elimination, intercept
//! free, residual variance RSS/(n-3), penalty variance over scale squared).

use antecedent_core::ExecutionContext;
use antecedent_estimate::EstimationError;
use antecedent_estimate::latent_class_effects::{
    LatentClassConfig, LatentClassData, LatentClassResult, fit_latent_class_effects,
};
use antecedent_estimate::posterior_mean::{PosteriorRequest, latent_class_effect_posterior};

const NOISE: [f64; 4] = [0.5, -0.5, 0.25, -0.25];
const NOISE_VARIANCE: f64 = 0.15625;

struct Data {
    y: Vec<f64>,
    a: Vec<f64>,
    x: Vec<Vec<f64>>,
    /// 0 for class `P`, 1 for class `Q`.
    class: Vec<usize>,
}

fn push(data: &mut Data, a: f64, x: f64, mean: f64, class: usize) {
    for e in NOISE {
        data.y.push(mean + e);
        data.a.push(a);
        data.x[0].push(x);
        data.class.push(class);
    }
}

fn build() -> Data {
    let mut data = Data { y: vec![], a: vec![], x: vec![vec![]], class: vec![] };
    for a in [0.0, 1.0] {
        for x in [-1.0, 0.0, 1.0] {
            push(&mut data, a, x, a + 0.5 * x, 0);
        }
    }
    for a in [0.0, 1.0] {
        for x in [-1.0, 1.0] {
            push(&mut data, a, x, 20.0 - 2.0 * a - 0.5 * x, 1);
        }
    }
    data
}

fn config(seed: u64) -> LatentClassConfig {
    let mut config = LatentClassConfig::new(2, seed).declare_conditional_randomization();
    config.bootstrap_replicates = 0;
    config
}

fn fit(data: &Data, config: &LatentClassConfig) -> Result<LatentClassResult, EstimationError> {
    fit_latent_class_effects(
        &LatentClassData { outcome: &data.y, treatment: &data.a, covariates: &data.x },
        config,
        &ExecutionContext::for_tests(1),
    )
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


/// Treatment coefficient of the ridge fit `y ~ 1 + a + x` on the rows of one class.
fn class_ridge_effect(data: &Data, class: usize, scale: f64) -> f64 {
    let idx: Vec<usize> = (0..data.y.len()).filter(|&i| data.class[i] == class).collect();
    let rows: Vec<Vec<f64>> = idx.iter().map(|&i| vec![1.0, data.a[i], data.x[0][i]]).collect();
    let y: Vec<f64> = idx.iter().map(|&i| data.y[i]).collect();
    let p = 3;
    let mut gram = vec![vec![0.0; p]; p];
    let mut rhs = vec![0.0; p];
    for (row, &yi) in rows.iter().zip(&y) {
        for j in 0..p {
            rhs[j] += row[j] * yi;
            for k in 0..p {
                gram[j][k] += row[j] * row[k];
            }
        }
    }
    let ols = solve(gram.clone(), rhs.clone());
    let rss: f64 = rows
        .iter()
        .zip(&y)
        .map(|(row, &yi)| {
            let fitted: f64 = row.iter().zip(&ols).map(|(a, b)| a * b).sum();
            (yi - fitted) * (yi - fitted)
        })
        .sum();
    let penalty = rss / (rows.len() - p) as f64 / (scale * scale);
    for (j, row) in gram.iter_mut().enumerate().skip(1) {
        row[j] += penalty;
    }
    solve(gram, rhs)[1]
}

#[test]
fn latent_class_posterior_means_equal_ridge_solutions_built_from_the_class_rows() {
    let data = build();
    let result = fit(&data, &config(11)).expect("well-separated fit");
    let designs: Vec<_> = result.classes.iter().map(|class| class.design.clone()).collect();
    for scale in [0.5, 2.0] {
        let got = latent_class_effect_posterior(
            &designs,
            PosteriorRequest::IndependentNormal { scale },
        )
        .unwrap();
        // Canonical order is ascending effect: class Q (data class 1) then class P (class 0).
        let want = [class_ridge_effect(&data, 1, scale), class_ridge_effect(&data, 0, scale)];
        assert_eq!(got.len(), 2);
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-8, "scale {scale}: {got:?} vs {want:?}");
        }
        // The prior shrinks both effects (-2 and 1) toward zero.
        assert!(got[0] > -2.0 + 1e-3 && got[1] < 1.0 - 1e-3, "{got:?}");
    }
}
