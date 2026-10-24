//! Joint normal tables for family estimates: the declared moments are matched
//! exactly, the dependence that overlapping rows imply agrees with a direct
//! simulation of the overlapping estimators, and impossible requests refuse.

use super::*;

const COVARIANCE: [f64; 9] = [0.25, 0.06, -0.03, 0.06, 0.16, 0.02, -0.03, 0.02, 0.09];

fn sample_moments(table: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
    let k = table.len();
    let n = table[0].len() as f64;
    let means: Vec<f64> = table.iter().map(|column| column.iter().sum::<f64>() / n).collect();
    let mut covariance = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..k {
            covariance[i * k + j] = table[i]
                .iter()
                .zip(&table[j])
                .map(|(a, b)| (a - means[i]) * (b - means[j]))
                .sum::<f64>()
                / (n - 1.0);
        }
    }
    (means, covariance)
}

#[test]
fn the_table_carries_the_declared_mean_and_covariance_exactly() {
    let means = [1.0, -2.0, 0.5];
    let table = moment_matched_normal_table(&means, &COVARIANCE, 4001, 11).expect("table");
    assert_eq!(table.len(), 3);
    assert!(table.iter().all(|column| column.len() == 4001));
    let (sample_mean, sample_covariance) = sample_moments(&table);
    for i in 0..3 {
        assert!((sample_mean[i] - means[i]).abs() < 1e-10, "mean {i}");
    }
    for i in 0..9 {
        assert!((sample_covariance[i] - COVARIANCE[i]).abs() < 1e-10, "covariance {i}");
    }
    // The marginals are normal, not merely moment-matched: the share above
    // one standard deviation is the normal tail within Monte Carlo error.
    let share = table[0].iter().filter(|value| **value > 1.0 + 0.5).count() as f64 / 4001.0;
    assert!((share - 0.158_655).abs() < 0.03, "{share}");
}

#[test]
fn a_nonlinear_functional_sees_the_declared_dependence() {
    // E[x0 x1] = mean0 mean1 + cov01 exactly for the sample covariance used.
    let table =
        moment_matched_normal_table(&[1.0, 2.0], &[0.25, 0.1, 0.1, 0.16], 2001, 5).expect("table");
    let product: f64 = table[0].iter().zip(&table[1]).map(|(a, b)| a * b).sum::<f64>() / 2001.0;
    let expected = 1.0 * 2.0 + 0.1 * 2000.0 / 2001.0;
    assert!((product - expected).abs() < 1e-10, "{product} vs {expected}");
}

#[test]
fn perfectly_correlated_variables_are_the_same_draws() {
    let table = moment_matched_normal_table(&[0.0, 3.0], &[1.0, 2.0, 2.0, 4.0], 500, 2)
        .expect("singular covariance is allowed");
    for (a, b) in table[0].iter().zip(&table[1]) {
        assert!((b - (3.0 + 2.0 * a)).abs() < 1e-9);
    }
}

#[test]
fn a_seed_reproduces_the_table_and_another_seed_changes_it() {
    let draw = |seed| moment_matched_normal_table(&[0.0, 0.0], &[1.0, 0.3, 0.3, 1.0], 300, seed);
    assert_eq!(draw(9).unwrap(), draw(9).unwrap());
    assert_ne!(draw(9).unwrap(), draw(10).unwrap());
}

#[test]
fn overlapping_rows_imply_the_correlation_a_direct_simulation_shows() {
    // Estimator A averages rows 0..200 and B rows 100..300 of one pool.
    let predicted = shared_row_correlation(200, 200, 100).expect("correlation");
    assert!((predicted - 0.5).abs() < 1e-15);
    let mut stream = SeedStream(77);
    let replicates = 4000;
    let (mut a_values, mut b_values) = (Vec::new(), Vec::new());
    for _ in 0..replicates {
        let mut pool = Vec::with_capacity(300);
        while pool.len() < 300 {
            let (x, y) = stream.normal_pair();
            pool.push(x);
            pool.push(y);
        }
        a_values.push(pool[..200].iter().sum::<f64>() / 200.0);
        b_values.push(pool[100..300].iter().sum::<f64>() / 200.0);
    }
    let (_, covariance) = sample_moments(&[a_values, b_values]);
    let simulated = covariance[1] / (covariance[0] * covariance[3]).sqrt();
    assert!((simulated - predicted).abs() < 0.06, "{simulated} vs {predicted}");
}

#[test]
fn impossible_tables_and_overlaps_are_refused() {
    let ok = [1.0, 0.0, 0.0, 1.0];
    assert!(matches!(
        moment_matched_normal_table(&[0.0, 0.0], &[1.0, 2.0, 2.0, 1.0], 100, 1),
        Err(ChainError::Operator(_))
    ));
    assert!(matches!(
        moment_matched_normal_table(&[0.0, 0.0], &[1.0, 0.5, 0.0, 1.0], 100, 1),
        Err(ChainError::Operator(_))
    ));
    assert!(matches!(
        moment_matched_normal_table(&[0.0, 0.0], &ok, 3, 1),
        Err(ChainError::Shape(_))
    ));
    assert!(matches!(
        moment_matched_normal_table(&[0.0, f64::NAN], &ok, 100, 1),
        Err(ChainError::NonFinite(_))
    ));
    assert!(matches!(moment_matched_normal_table(&[0.0], &ok, 100, 1), Err(ChainError::Shape(_))));
    assert!(matches!(
        moment_matched_normal_table(&[0.0, 0.0], &ok, MAX_CHAIN_ROWS + 1, 1),
        Err(ChainError::Budget(_))
    ));
    assert!(shared_row_correlation(10, 10, 11).is_err());
    assert!(shared_row_correlation(0, 10, 0).is_err());
}
