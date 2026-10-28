//! Exhaustive-enumeration checks of the randomization-based exposure kernels.
//!
//! A six-unit directed network with unequal in-degrees, a potential-outcome table that depends on
//! own treatment and on the number of treated in-neighbours, and two designs whose assignments
//! are enumerated here independently of the product enumerator (complete randomization with
//! three treated, which induces negative dependence between units, and Bernoulli with
//! unit-specific probabilities). Over the whole assignment support the Horvitz-Thompson mean must
//! be exactly unbiased for the population exposure mean, the diagonal variance estimate must have
//! expectation `sum_i Var(X_i) / n`, and that expectation must bound the true variance of the
//! estimator (the Young-inequality property that the product documents). Exposure probabilities
//! returned by the product must equal the enumerated ones.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{AssignmentDesign, ExposureLevel, ExposureMapping};
use antecedent_stats::{
    exposure_probabilities, exposures, randomization_contrast, randomization_mean,
};

const N: usize = 6;

fn network() -> Vec<Vec<(usize, f64)>> {
    vec![
        vec![(1, 1.0), (2, 1.0)],
        vec![(0, 1.0), (3, 1.0), (4, 1.0)],
        vec![(5, 1.0)],
        vec![(1, 1.0), (2, 1.0), (4, 1.0), (5, 1.0)],
        vec![(3, 1.0), (0, 1.0)],
        vec![(2, 1.0), (4, 1.0)],
    ]
}

/// Potential outcome of unit `i` at own treatment `own` and `k` treated in-neighbours.
fn potential(i: usize, own: f64, k: f64) -> f64 {
    1.0 + 0.5 * i as f64 + 2.0 * own + 1.5 * k + 0.3 * own * k + 0.1 * k * k
}

/// Independent enumeration of `(assignment, probability)` for the two designs.
fn support(design: &str) -> Vec<(Vec<bool>, f64)> {
    let per_unit = [0.5, 0.4, 0.6, 0.5, 0.3, 0.7];
    let mut out = Vec::new();
    for mask in 0..(1_u32 << N) {
        let a: Vec<bool> = (0..N).map(|i| mask >> i & 1 == 1).collect();
        match design {
            "complete" => {
                if mask.count_ones() == 3 {
                    out.push((a, 1.0 / 20.0));
                }
            }
            _ => {
                let p: f64 =
                    (0..N).map(|i| if a[i] { per_unit[i] } else { 1.0 - per_unit[i] }).product();
                out.push((a, p));
            }
        }
    }
    out
}

fn design(name: &str) -> AssignmentDesign {
    match name {
        "complete" => AssignmentDesign::CompleteRandomization { treated: 3 },
        _ => AssignmentDesign::Bernoulli {
            probabilities: Arc::from(vec![0.5, 0.4, 0.6, 0.5, 0.3, 0.7]),
        },
    }
}

fn own_k(a: &[bool], i: usize, net: &[Vec<(usize, f64)>]) -> (f64, f64) {
    (f64::from(a[i]), net[i].iter().filter(|(s, _)| a[*s]).count() as f64)
}

#[test]
fn exposure_probabilities_equal_independent_enumeration() {
    let net = network();
    for name in ["complete", "bernoulli"] {
        for level in [(1.0, 1.0), (0.0, 2.0), (1.0, 0.0)] {
            let expected: Vec<f64> = (0..N)
                .map(|i| {
                    support(name)
                        .iter()
                        .filter(|(a, _)| own_k(a, i, &net) == level)
                        .map(|(_, p)| p)
                        .sum()
                })
                .collect();
            let got = exposure_probabilities(
                &design(name),
                &net,
                &ExposureMapping::NeighborCount,
                ExposureLevel { own: level.0, neighbors: level.1 },
                1,
                0,
            )
            .unwrap();
            for i in 0..N {
                assert!(
                    (got.probabilities[i] - expected[i]).abs() <= 1e-12,
                    "{name} level {level:?} unit {i}: {} vs {}",
                    got.probabilities[i],
                    expected[i]
                );
            }
        }
    }
}

#[test]
fn horvitz_thompson_is_unbiased_and_its_variance_bound_dominates_over_the_whole_support() {
    let net = network();
    for name in ["complete", "bernoulli"] {
        let level = ExposureLevel { own: 1.0, neighbors: 1.0 };
        let pi = exposure_probabilities(
            &design(name),
            &net,
            &ExposureMapping::NeighborCount,
            level,
            1,
            0,
        )
        .unwrap()
        .probabilities;
        assert!(pi.iter().all(|&p| p > 0.0), "{name}: positivity {pi:?}");
        let y_level: Vec<f64> = (0..N).map(|i| potential(i, 1.0, 1.0)).collect();
        let truth: f64 = y_level.iter().sum::<f64>() / N as f64;

        let (mut e_ht, mut e_ht2, mut e_var, mut e_hajek_num, mut mass_present) =
            (0.0, 0.0, 0.0, 0.0, 0.0);
        for (a, p) in support(name) {
            let observed = exposures(&a, &net, &ExposureMapping::NeighborCount).unwrap();
            let outcomes: Vec<f64> =
                (0..N).map(|i| potential(i, observed[i].own, observed[i].neighbors)).collect();
            // Own implementation of the estimator on this realization.
            let exposed: Vec<usize> = (0..N)
                .filter(|&i| (observed[i].own, observed[i].neighbors) == (1.0, 1.0))
                .collect();
            let (ht, var) = if exposed.is_empty() {
                (0.0, 0.0)
            } else {
                let m = randomization_mean(&outcomes, &observed, &pi, level).unwrap();
                assert_eq!(m.exposed_units, exposed.len());
                let direct: f64 =
                    exposed.iter().map(|&i| outcomes[i] / pi[i]).sum::<f64>() / N as f64;
                assert!((m.horvitz_thompson - direct).abs() <= 1e-12);
                let hajek = exposed.iter().map(|&i| outcomes[i] / pi[i]).sum::<f64>()
                    / exposed.iter().map(|&i| 1.0 / pi[i]).sum::<f64>();
                assert!((m.hajek - hajek).abs() <= 1e-12);
                e_hajek_num += p * m.hajek;
                mass_present += p;
                (m.horvitz_thompson, m.conservative_variance)
            };
            e_ht += p * ht;
            e_ht2 += p * ht * ht;
            e_var += p * var;
        }
        assert!((e_ht - truth).abs() <= 1e-10, "{name}: E[HT] {e_ht} vs {truth}");
        let true_var = e_ht2 - e_ht * e_ht;
        let diagonal: f64 =
            (0..N).map(|i| (1.0 - pi[i]) / pi[i] * y_level[i].powi(2)).sum::<f64>() / N as f64;
        assert!(
            (e_var - diagonal).abs() <= 1e-9 * diagonal,
            "{name}: E[variance estimate] {e_var} vs sum Var(X_i)/n {diagonal}"
        );
        assert!(
            e_var >= true_var - 1e-12,
            "{name}: Young bound {e_var} must dominate the true variance {true_var}"
        );
        // The Hajek ratio is only consistent, not unbiased, but it must lie in the outcome range.
        let hajek_mean = e_hajek_num / mass_present;
        let (lo, hi) = (
            y_level.iter().cloned().fold(f64::INFINITY, f64::min),
            y_level.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        );
        assert!(hajek_mean > lo - 5.0 && hajek_mean < hi + 5.0);
    }
}

#[test]
fn contrast_combines_means_and_doubles_the_summed_variance_bound() {
    let net = network();
    let d = design("bernoulli");
    let a = [false, false, false, false, true, true];
    let observed = exposures(&a, &net, &ExposureMapping::NeighborCount).unwrap();
    let outcomes: Vec<f64> =
        (0..N).map(|i| potential(i, observed[i].own, observed[i].neighbors)).collect();
    let levels =
        [ExposureLevel { own: 1.0, neighbors: 1.0 }, ExposureLevel { own: 0.0, neighbors: 1.0 }];
    let mut means = Vec::new();
    for level in levels {
        let pi = exposure_probabilities(&d, &net, &ExposureMapping::NeighborCount, level, 1, 0)
            .unwrap()
            .probabilities;
        means.push(randomization_mean(&outcomes, &observed, &pi, level).unwrap());
    }
    let c = randomization_contrast(means[0], means[1]);
    assert!(
        (c.horvitz_thompson - (means[1].horvitz_thompson - means[0].horvitz_thompson)).abs()
            < 1e-15
    );
    assert!((c.hajek - (means[1].hajek - means[0].hajek)).abs() < 1e-15);
    // Var(B - A) <= 2 (Var A + Var B) because Cov(A, B) >= -sqrt(Var A Var B) and
    // 2 sqrt(ab) <= a + b.
    assert!(
        (c.conservative_variance
            - 2.0 * (means[0].conservative_variance + means[1].conservative_variance))
            .abs()
            < 1e-12
    );
}

#[test]
fn unreachable_exposure_for_some_unit_is_refused_not_averaged_in() {
    // Unit 2 has a single in-neighbour, so two treated in-neighbours is impossible for it.
    let net = network();
    let level = ExposureLevel { own: 1.0, neighbors: 2.0 };
    let pi = exposure_probabilities(
        &design("bernoulli"),
        &net,
        &ExposureMapping::NeighborCount,
        level,
        1,
        0,
    )
    .unwrap()
    .probabilities;
    assert_eq!(pi[2], 0.0);
    let a = [true, true, true, true, true, true];
    let observed = exposures(&a, &net, &ExposureMapping::NeighborCount).unwrap();
    let outcomes = vec![1.0; N];
    let err = randomization_mean(&outcomes, &observed, &pi, level).unwrap_err();
    assert!(err.to_string().contains("positivity"), "{err}");
}
