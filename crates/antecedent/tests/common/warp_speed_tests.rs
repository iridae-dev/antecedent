//! Hand cases of the pooled-draw intervals; no coverage is measured.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    reason = "the sample sizes are far below the integer range of f64"
)]

use super::warp_speed::*;

const TYPE7: QuantileRule = QuantileRule::Interpolated;
const TYPE6: QuantileRule = QuantileRule::ExchangeableRank;

fn near(left: f64, right: f64) {
    assert!((left - right).abs() < 1e-9, "{left} != {right}");
}

fn pair(interval: Option<(f64, f64)>) -> (f64, f64) {
    interval.expect("an interval")
}

/// Replicates with estimate `estimate` whose single draws sit at `estimate + root`.
fn rooted(estimate: f64, roots: &[f64]) -> Vec<Option<Replicate>> {
    roots.iter().map(|root| Some(Replicate::single(estimate, estimate + root))).collect()
}

#[test]
fn the_two_quantile_rules_match_their_hand_positions() {
    let sample = [1.0, 2.0, 4.0, 8.0, 16.0];
    near(quantile_sorted(&sample, 0.25, TYPE7), 2.0);
    near(quantile_sorted(&sample, 0.6, TYPE7), 5.6);
    near(quantile_sorted(&sample, 0.0, TYPE7), 1.0);
    near(quantile_sorted(&sample, 1.0, TYPE7), 16.0);
    let nine: Vec<f64> = (1..=9).map(f64::from).collect();
    near(quantile_sorted(&nine, 0.2, TYPE6), 2.0);
    near(quantile_sorted(&nine, 0.25, TYPE6), 2.5);
    near(quantile_sorted(&nine, 0.001, TYPE6), 1.0);
    near(quantile_sorted(&nine, 0.999, TYPE6), 9.0);
    assert!(quantile_sorted(&[], 0.5, TYPE7).is_nan());
}

#[test]
fn a_percentile_interval_takes_the_pooled_root_quantiles_about_each_estimate() {
    // Roots -4..=4: the type 7 quartiles of nine values are -2 and 2.
    let roots: Vec<f64> = (-4..=4).map(f64::from).collect();
    let mut replicates = rooted(0.0, &roots);
    replicates[0] = Some(Replicate::single(10.0, 10.0 + roots[0]));
    let found = pooled_intervals(
        &replicates,
        Construction::Percentile { rule: TYPE7 },
        Pooling::whole_pool(0.5),
    );
    let (lower, upper) = pair(found[0]);
    near(lower, 8.0);
    near(upper, 12.0);
    let (lower, upper) = pair(found[4]);
    near(lower, -2.0);
    near(upper, 2.0);
}

#[test]
fn the_scale_and_the_product_factor_stretch_the_distance_from_the_estimate() {
    // The scored replicate has scale 2 and factor 1.5, so its root -4 enters the pool as -2 and
    // its interval spreads the pooled quartiles by 3 about its estimate.
    let roots: Vec<f64> = (-4..=4).map(f64::from).collect();
    let mut replicates = rooted(0.0, &roots);
    replicates[0] = Some(Replicate::single(10.0, 10.0 - 4.0).scaled(2.0).corrected(1.5));
    // The pool the helper sees: roots -3 ..= 4 of the other replicates and -2 of this one.
    let mut pool: Vec<f64> = vec![-2.0];
    pool.extend((-3..=4).map(f64::from));
    pool.sort_by(f64::total_cmp);
    let (low, high) = (quantile_sorted(&pool, 0.25, TYPE7), quantile_sorted(&pool, 0.75, TYPE7));
    let found = pooled_intervals(
        &replicates,
        Construction::Percentile { rule: TYPE7 },
        Pooling::whole_pool(0.5),
    );
    let (lower, upper) = pair(found[0]);
    near(lower, 10.0 + 3.0 * low);
    near(upper, 10.0 + 3.0 * high);
}

#[test]
fn a_basic_interval_reflects_the_percentile_quantiles_about_the_estimate() {
    // Roots 0..=4: quartiles 1 and 3.
    let replicates = rooted(10.0, &[0.0, 1.0, 2.0, 3.0, 4.0]);
    let level = Pooling::whole_pool(0.5);
    let percentile = pooled_intervals(&replicates, Construction::Percentile { rule: TYPE7 }, level);
    let basic = pooled_intervals(&replicates, Construction::Basic { rule: TYPE7 }, level);
    let (lower, upper) = pair(percentile[0]);
    near(lower, 11.0);
    near(upper, 13.0);
    let (lower, upper) = pair(basic[0]);
    near(lower, 7.0);
    near(upper, 9.0);
}

#[test]
fn a_studentized_interval_scales_the_pooled_pivots_by_the_replicate_error() {
    // Roots 0..=4 over draw errors 2 give pivots 0, .5, 1, 1.5, 2.
    let replicates: Vec<Option<Replicate>> = (0..5)
        .map(|k| Some(Replicate::single(10.0, 10.0 + f64::from(k)).studentized(vec![2.0], 4.0)))
        .collect();
    let found = pooled_intervals(
        &replicates,
        Construction::Studentized { rule: TYPE6 },
        Pooling::whole_pool(0.5),
    );
    // Type 6 on five pivots: ranks .25 * 6 = 1.5 and .75 * 6 = 4.5, so .25 and 1.75.
    let (lower, upper) = pair(found[0]);
    near(lower, 10.0 - 4.0 * 1.75);
    near(upper, 10.0 - 4.0 * 0.25);
}

#[test]
fn a_studentized_replicate_without_an_error_has_no_interval() {
    let mut replicates = rooted(0.0, &[0.0, 1.0, 2.0]);
    for slot in replicates.iter_mut().flatten() {
        slot.draw_errors = vec![1.0];
    }
    let found = pooled_intervals(
        &replicates,
        Construction::Studentized { rule: TYPE7 },
        Pooling::whole_pool(0.95),
    );
    assert!(found.iter().all(Option::is_none), "no replicate error, no studentized interval");
}

#[test]
fn a_standard_error_interval_uses_the_pooled_sd_and_the_product_critical_value() {
    // Roots -2..=2 have sample sd sqrt(10 / 4).
    let replicates = rooted(1.0, &[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let found = pooled_intervals(
        &replicates,
        Construction::NormalSd { critical: 2.0 },
        Pooling::whole_pool(0.95),
    );
    let half = 2.0 * (2.5_f64).sqrt();
    let (lower, upper) = pair(found[2]);
    near(lower, 1.0 - half);
    near(upper, 1.0 + half);
}

#[test]
fn several_draws_per_replicate_all_enter_the_pool() {
    let replicates = vec![
        Some(Replicate::with_draws(0.0, vec![-2.0, -1.0, 0.0])),
        Some(Replicate::with_draws(0.0, vec![1.0, 2.0])),
    ];
    let found = pooled_intervals(
        &replicates,
        Construction::Percentile { rule: TYPE7 },
        Pooling::whole_pool(0.5),
    );
    // Pool -2, -1, 0, 1, 2: quartiles -1 and 1.
    assert_eq!(found[0], found[1]);
    let (lower, upper) = pair(found[0]);
    near(lower, -1.0);
    near(upper, 1.0);
}

#[test]
fn missing_unusable_and_non_finite_inputs_are_misses_not_skips() {
    let mut replicates = rooted(0.0, &[-1.0, 0.0, 1.0]);
    replicates.push(None);
    replicates.push(Some(Replicate::single(f64::NAN, 0.0)));
    replicates.push(Some(Replicate::single(0.0, f64::INFINITY)));
    replicates.push(Some(Replicate::single(0.0, 0.0).scaled(0.0)));
    let found = pooled_intervals(
        &replicates,
        Construction::Percentile { rule: TYPE7 },
        Pooling::whole_pool(0.5),
    );
    assert_eq!(found.len(), replicates.len());
    assert!(found[..3].iter().all(Option::is_some));
    // The infinite draw is dropped from the pool, and the replicate that ran still scores.
    assert!(found[3].is_none() && found[4].is_none() && found[6].is_none());
    assert!(found[5].is_some());
}

#[test]
fn a_pool_below_the_minimum_forms_no_interval() {
    let replicates = rooted(0.0, &[-1.0, 1.0]);
    let pooling = Pooling { min_pool: 3, ..Pooling::whole_pool(0.95) };
    let found = pooled_intervals(&replicates, Construction::Percentile { rule: TYPE7 }, pooling);
    assert!(found.iter().all(Option::is_none));
}

#[test]
fn a_subset_of_the_pool_is_deterministic_and_inside_the_pool() {
    let roots: Vec<f64> = (0..200).map(f64::from).collect();
    let replicates = rooted(0.0, &roots);
    let construction = Construction::Percentile { rule: TYPE7 };
    let subset = Pooling { inner: Some(20), seed: 7, ..Pooling::whole_pool(0.9) };
    let first = pooled_intervals(&replicates, construction, subset);
    let again = pooled_intervals(&replicates, construction, subset);
    assert_eq!(first, again, "the same seed chooses the same subsets");
    let other = pooled_intervals(&replicates, construction, Pooling { seed: 8, ..subset });
    assert_ne!(first, other, "another seed chooses other subsets");
    for (lower, upper) in first.iter().flatten() {
        assert!((0.0..=199.0).contains(lower) && (0.0..=199.0).contains(upper));
        assert!(lower < upper);
    }
    // Different replicates draw different subsets, so their endpoints differ.
    assert_ne!(first[0], first[1]);
}

#[test]
fn a_subset_as_large_as_the_pool_is_the_whole_pool() {
    let roots: Vec<f64> = (0..50).map(f64::from).collect();
    let replicates = rooted(0.0, &roots);
    let construction = Construction::Percentile { rule: TYPE6 };
    let whole = pooled_intervals(&replicates, construction, Pooling::whole_pool(0.9));
    let same = Pooling { inner: Some(50), seed: 3, ..Pooling::whole_pool(0.9) };
    assert_eq!(whole, pooled_intervals(&replicates, construction, same));
}

/// `SplitMix64`-driven standard normals for the coverage sanity check.
fn normals(seed: u64, n: usize) -> Vec<f64> {
    let mut state = seed;
    let mut uniform = || ((next_u64(&mut state) >> 11) as f64 + 0.5) / (1_u64 << 53) as f64;
    (0..n)
        .map(|_| {
            let (a, b) = (uniform(), uniform());
            (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
        })
        .collect()
}

#[test]
fn pooled_single_draws_recover_the_coverage_of_a_pivotal_resampling_law() {
    // Estimates N(0, 1) about the truth 0; one draw per replicate N(estimate, 1): the pooled
    // percentile interval is estimate +- 1.96, which covers at the nominal level.
    let count = 3_000;
    let noise = normals(12_345, 2 * count);
    let replicates: Vec<Option<Replicate>> = (0..count)
        .map(|r| Some(Replicate::single(noise[r], noise[r] + noise[count + r])))
        .collect();
    let found = pooled_intervals(
        &replicates,
        Construction::Percentile { rule: TYPE7 },
        Pooling::whole_pool(0.95),
    );
    let covered = found.iter().flatten().filter(|(lo, hi)| *lo <= 0.0 && 0.0 <= *hi).count();
    let rate = covered as f64 / count as f64;
    assert!((rate - 0.95).abs() < 0.02, "pooled percentile coverage {rate}");
}

#[test]
fn composed_draw_sets_reproduce_the_pooled_percentile_interval() {
    let roots: Vec<f64> = (0..40).map(|k| f64::from(k) - 19.5).collect();
    let mut replicates = rooted(10.0, &roots);
    replicates[3] = Some(Replicate::single(2.0, 2.0 + roots[3]).scaled(2.0));
    replicates[7] = None;
    let pooling = Pooling::whole_pool(0.9);
    let sets = pooled_draw_sets(&replicates, TYPE7, pooling);
    let intervals =
        pooled_intervals(&replicates, Construction::Percentile { rule: TYPE7 }, pooling);
    assert_eq!(sets.len(), replicates.len());
    assert!(sets[7].is_none());
    for (set, interval) in sets.iter().zip(&intervals) {
        let (Some(set), Some((low, high))) = (set, interval) else { continue };
        near(quantile_sorted(set, 0.05, TYPE7), *low);
        near(quantile_sorted(set, 0.95, TYPE7), *high);
    }
}

#[test]
fn composed_draw_sets_take_the_inner_count_from_a_larger_pool() {
    let roots: Vec<f64> = (0..50).map(|k| f64::from(k) * 0.1).collect();
    let replicates = rooted(1.0, &roots);
    let pooling = Pooling { level: 0.95, min_pool: 2, inner: Some(20), seed: 9 };
    let sets = pooled_draw_sets(&replicates, TYPE7, pooling);
    assert!(sets.iter().all(|set| set.as_ref().is_some_and(|set| set.len() == 20)));
    let again = pooled_draw_sets(&replicates, TYPE7, pooling);
    assert_eq!(sets, again);
}
