//! Design-based Neyman estimators against exact enumeration of the randomization law.
//!
//! A finite population has fixed potential outcomes (skewed, with unit-level effect
//! heterogeneity). Every complete randomization is enumerated; the estimator and its variance
//! bound are evaluated on each, and compared with the finite-population moments of Neyman's
//! analysis, computed here from the potential outcomes:
//!
//! * the estimator is exactly unbiased for the finite-population average effect;
//! * its exact variance over the enumerated assignments equals `S1^2/n1 + S0^2/n0 - S_tau^2/N`;
//! * the mean of the reported variance bound over the assignments equals
//!   `S1^2/n1 + S0^2/n0`, so the bound exceeds the true variance by exactly `S_tau^2/N`.
//!
//! The same three identities are checked for blocks (independent complete randomizations),
//! for cluster totals, for the 2x2 factorial (four fixed cell sizes) and for the Hajek
//! estimator with known arm probabilities.

use antecedent_estimate::randomized_neyman::{
    blocked_unit_itt, complete_cluster_itt, complete_unit_itt, factorial_2x2, weighted_itt,
};

fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    fn rec(start: usize, n: usize, k: usize, cur: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if cur.len() == k {
            out.push(cur.clone());
            return;
        }
        for i in start..n {
            cur.push(i);
            rec(i + 1, n, k, cur, out);
            cur.pop();
        }
    }
    let mut out = Vec::new();
    rec(0, n, k, &mut Vec::new(), &mut out);
    out
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// Finite-population variance with divisor `N - 1`.
fn pop_var(values: &[f64]) -> f64 {
    let m = mean(values);
    values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (values.len() - 1) as f64
}

struct Law {
    mean_estimate: f64,
    variance_estimate: f64,
    mean_bound: f64,
}

fn close(a: f64, b: f64, what: &str) {
    assert!((a - b).abs() <= 1e-11 * (1.0 + a.abs().max(b.abs())), "{what}: {a} vs {b}");
}

const Y0: [f64; 9] = [0.0, 1.0, 1.0, 2.0, 2.5, 3.0, 7.0, 0.5, 12.0];
const TAU: [f64; 9] = [1.0, -2.0, 0.5, 4.0, 0.0, 3.0, -1.5, 6.0, -4.0];

fn y1() -> Vec<f64> {
    Y0.iter().zip(&TAU).map(|(a, b)| a + b).collect()
}

fn complete_law(y0: &[f64], y1: &[f64], n1: usize) -> Law {
    let n = y0.len();
    let (mut sum, mut sum_sq, mut bound) = (0.0, 0.0, 0.0);
    let assignments = combinations(n, n1);
    for treated in &assignments {
        let t: Vec<f64> = treated.iter().map(|&i| y1[i]).collect();
        let c: Vec<f64> = (0..n).filter(|i| !treated.contains(i)).map(|i| y0[i]).collect();
        let fit = complete_unit_itt(&t, &c).expect("fit");
        sum += fit.effect;
        sum_sq += fit.effect * fit.effect;
        bound += fit.variance_upper_bound;
    }
    let m = assignments.len() as f64;
    Law { mean_estimate: sum / m, variance_estimate: sum_sq / m - (sum / m).powi(2), mean_bound: bound / m }
}

#[test]
fn complete_randomization_is_unbiased_with_the_neyman_variance_and_exact_slack() {
    let y1 = y1();
    let ate = mean(&TAU);
    for n1 in [2_usize, 3, 4, 7] {
        let n0 = 9 - n1;
        let law = complete_law(&Y0, &y1, n1);
        close(law.mean_estimate, ate, "unbiased");
        let (s1, s0, st) = (pop_var(&y1), pop_var(&Y0), pop_var(&TAU));
        let true_var = s1 / n1 as f64 + s0 / n0 as f64 - st / 9.0;
        close(law.variance_estimate, true_var, "Neyman variance");
        close(law.mean_bound, s1 / n1 as f64 + s0 / n0 as f64, "mean of the bound");
        assert!(law.mean_bound >= law.variance_estimate - 1e-12);
    }
}

#[test]
fn hajek_contrast_with_constant_arm_probabilities_is_the_complete_randomization_contrast() {
    let y1 = y1();
    let n1 = 4;
    let (p1, p0) = (n1 as f64 / 9.0, 1.0 - n1 as f64 / 9.0);
    for treated in combinations(9, n1) {
        let mut outcomes = Vec::new();
        let mut flags = Vec::new();
        let mut probs = Vec::new();
        for i in 0..9 {
            let t = treated.contains(&i);
            outcomes.push(if t { y1[i] } else { Y0[i] });
            flags.push(t);
            probs.push(if t { p1 } else { p0 });
        }
        let weighted = weighted_itt(&outcomes, &flags, &probs, &[1.0; 9]).expect("weighted");
        let t: Vec<f64> = treated.iter().map(|&i| y1[i]).collect();
        let c: Vec<f64> = (0..9).filter(|i| !treated.contains(i)).map(|i| Y0[i]).collect();
        let plain = complete_unit_itt(&t, &c).expect("plain");
        close(weighted.effect, plain.effect, "effect");
        close(weighted.variance, plain.variance_upper_bound, "variance");
    }
}

#[test]
fn blocked_randomization_matches_the_independent_block_law() {
    // Blocks of 5 (2 treated) and 6 (3 treated), randomized independently.
    let y0 = [0.0, 1.0, 4.0, 2.0, 9.0, 1.0, 3.0, 3.5, 0.0, 6.0, 2.0];
    let tau = [2.0, -1.0, 0.5, 3.0, -3.0, 1.0, 1.0, 0.0, 5.0, -2.0, 2.5];
    let y1: Vec<f64> = y0.iter().zip(&tau).map(|(a, b)| a + b).collect();
    let blocks = [(0_usize..5, 2_usize), (5..11, 3)];
    let n = 11.0;
    let (first, second) = (combinations(5, 2), combinations(6, 3));
    let (mut sum, mut sum_sq, mut bound, mut count) = (0.0, 0.0, 0.0, 0.0);
    for a in &first {
        for b in &second {
            let mut parts: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
            for (range, picked) in [(&blocks[0].0, a), (&blocks[1].0, b)] {
                let ids: Vec<usize> = range.clone().collect();
                let t: Vec<f64> = picked.iter().map(|&k| y1[ids[k]]).collect();
                let c: Vec<f64> =
                    (0..ids.len()).filter(|k| !picked.contains(k)).map(|k| y0[ids[k]]).collect();
                parts.push((t, c));
            }
            let refs: Vec<(&[f64], &[f64])> =
                parts.iter().map(|(t, c)| (t.as_slice(), c.as_slice())).collect();
            let fit = blocked_unit_itt(&refs).expect("blocked");
            sum += fit.effect;
            sum_sq += fit.effect * fit.effect;
            bound += fit.variance_upper_bound;
            count += 1.0;
        }
    }
    close(sum / count, mean(&tau), "blocked unbiased");
    let mut true_var = 0.0;
    let mut bound_mean = 0.0;
    for (range, treated) in &blocks {
        let size = range.len() as f64;
        let w = size / n;
        let (b1, b0, bt) = (&y1[range.clone()], &y0[range.clone()], &tau[range.clone()]);
        let (n1, n0) = (*treated as f64, size - *treated as f64);
        true_var += w * w * (pop_var(b1) / n1 + pop_var(b0) / n0 - pop_var(bt) / size);
        bound_mean += w * w * (pop_var(b1) / n1 + pop_var(b0) / n0);
    }
    close(sum_sq / count - (sum / count).powi(2), true_var, "blocked variance");
    close(bound / count, bound_mean, "blocked mean bound");
}

#[test]
fn cluster_totals_estimate_the_per_row_effect_with_the_cluster_neyman_variance() {
    // Seven clusters of unequal size; the potential outcome of a cluster is its row total.
    let rows = [3.0, 1.0, 4.0, 2.0, 6.0, 2.0, 5.0];
    let t0 = [4.0, 1.5, 7.0, 2.0, 11.0, 2.5, 9.0];
    let tau = [1.0, 2.0, -3.0, 0.5, 4.0, 1.0, -2.0];
    let t1: Vec<f64> = t0.iter().zip(&tau).map(|(a, b)| a + b).collect();
    let total_rows: f64 = rows.iter().sum();
    let (j, n1) = (7_usize, 3_usize);
    let scale = j as f64 / total_rows;
    let (mut sum, mut sum_sq, mut bound, mut count) = (0.0, 0.0, 0.0, 0.0);
    for treated in combinations(j, n1) {
        let t: Vec<f64> = treated.iter().map(|&i| t1[i]).collect();
        let c: Vec<f64> = (0..j).filter(|i| !treated.contains(i)).map(|i| t0[i]).collect();
        let fit = complete_cluster_itt(&t, &c, total_rows as usize).expect("cluster");
        sum += fit.effect;
        sum_sq += fit.effect * fit.effect;
        bound += fit.variance_upper_bound;
        count += 1.0;
    }
    let ate_per_row = tau.iter().sum::<f64>() / total_rows;
    close(sum / count, ate_per_row, "cluster unbiased");
    let (s1, s0, st) = (pop_var(&t1), pop_var(&t0), pop_var(&tau));
    let n0 = (j - n1) as f64;
    let true_var = scale * scale * (s1 / n1 as f64 + s0 / n0 - st / j as f64);
    close(sum_sq / count - (sum / count).powi(2), true_var, "cluster variance");
    close(bound / count, scale * scale * (s1 / n1 as f64 + s0 / n0), "cluster mean bound");
}

#[test]
fn factorial_main_effects_and_interaction_match_the_four_cell_randomization_law() {
    // Eight units, four potential outcomes each (cells 00, 10, 01, 11), two units per cell.
    let y: [[f64; 4]; 8] = [
        [0.0, 1.0, 2.0, 5.0],
        [1.0, 1.5, 0.0, 2.0],
        [2.0, 4.0, 3.0, 3.0],
        [0.5, 0.0, 1.0, 6.5],
        [3.0, 3.5, 4.0, 9.0],
        [1.0, 2.0, 1.0, 1.0],
        [0.0, 0.5, 2.5, 3.0],
        [2.0, 2.0, 5.0, 4.0],
    ];
    let col = |c: usize| -> Vec<f64> { y.iter().map(|row| row[c]).collect() };
    let primary_truth = 0.5 * (mean(&col(1)) - mean(&col(0)) + mean(&col(3)) - mean(&col(2)));
    let secondary_truth = 0.5 * (mean(&col(2)) - mean(&col(0)) + mean(&col(3)) - mean(&col(1)));
    let interaction_truth = mean(&col(3)) - mean(&col(2)) - mean(&col(1)) + mean(&col(0));
    let mut stats = [(0.0_f64, 0.0_f64); 3];
    let (mut main_bound, mut inter_bound, mut count) = (0.0, 0.0, 0.0);
    for c00 in combinations(8, 2) {
        let rest0: Vec<usize> = (0..8).filter(|i| !c00.contains(i)).collect();
        for p10 in combinations(6, 2) {
            let c10: Vec<usize> = p10.iter().map(|&k| rest0[k]).collect();
            let rest1: Vec<usize> = (0..6).filter(|k| !p10.contains(k)).map(|k| rest0[k]).collect();
            for p01 in combinations(4, 2) {
                let c01: Vec<usize> = p01.iter().map(|&k| rest1[k]).collect();
                let c11: Vec<usize> =
                    (0..4).filter(|k| !p01.contains(k)).map(|k| rest1[k]).collect();
                let cell = |ids: &[usize], c: usize| -> Vec<f64> {
                    ids.iter().map(|&i| y[i][c]).collect()
                };
                let (a, b, c, d) = (cell(&c00, 0), cell(&c10, 1), cell(&c01, 2), cell(&c11, 3));
                let fit = factorial_2x2([&a, &b, &c, &d]).expect("factorial");
                for (slot, value) in [fit.primary, fit.secondary, fit.interaction].iter().enumerate()
                {
                    stats[slot].0 += value;
                    stats[slot].1 += value * value;
                }
                main_bound += fit.main_effect_variance_upper_bound;
                inter_bound += fit.interaction_variance_upper_bound;
                count += 1.0;
            }
        }
    }
    assert_eq!(count, 2520.0);
    for (slot, truth, name) in [
        (0, primary_truth, "primary"),
        (1, secondary_truth, "secondary"),
        (2, interaction_truth, "interaction"),
    ] {
        close(stats[slot].0 / count, truth, name);
    }
    // E[bound] = sum_c S_c^2 / n_c (n_c = 2), divided by 4 for a main effect.
    let cell_sum: f64 = (0..4).map(|c| pop_var(&col(c)) / 2.0).sum();
    close(inter_bound / count, cell_sum, "interaction mean bound");
    close(main_bound / count, cell_sum / 4.0, "main-effect mean bound");
    // The bound is conservative for the exact randomization variance of each contrast.
    let variance = |slot: usize| stats[slot].1 / count - (stats[slot].0 / count).powi(2);
    assert!(variance(2) <= inter_bound / count + 1e-12);
    assert!(variance(0) <= main_bound / count + 1e-12);
    assert!(variance(1) <= main_bound / count + 1e-12);
}
