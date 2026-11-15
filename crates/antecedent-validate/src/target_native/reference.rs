//! Finite-resampling reference for a replicate refuter's joint discrepancy.
//!
//! A replicate refuter forms `D² = (e − m)ᵀ S⁻¹ (e − m)` from the mean `m` and covariance `S` of
//! `R` replicate target vectors. The `χ²_k` tail treats `S` as the exact covariance of one
//! replicate. It is not: `S` is itself estimated, and two independent sources of noise enter it.
//!
//! * **Monte-Carlo noise.** `S` is a covariance of `R` draws, so it carries `R − 1` degrees of
//!   freedom. For a vector with the covariance of one replicate (the standardised estimate of
//!   a resampling refuter), `D²` is then a Hotelling statistic,
//!   `D² = ν k / (ν − k + 1) · F(k, ν − k + 1)` with `ν = R − 1`, whose tail exceeds the `χ²_k`
//!   tail (at `k = 2`, `R = 40` the nominal 5% level is 6.6%).
//! * **Resampling noise.** The law the replicates are drawn from is the data's empirical law, and
//!   its covariance differs from the true sampling covariance of the estimate by an amount that
//!   shrinks only with the number of independent units the resample is built from: `g` rows
//!   for an iid row bootstrap, about `n / b` blocks for a circular block bootstrap with blocks of
//!   `b` periods (the fixed-block regime, where a studentised statistic has a Student-t / `F`
//!   reference with degrees of freedom tied to the block count rather than a normal one).
//!   A variance estimate with `ν` degrees of freedom has relative variance `2 / ν`; the two
//!   sources are independent, so their relative variances add, and the effective degrees of
//!   freedom are `1 / (1 / (R − 1) + 1 / (g − 1))`.
//!
//! The tail below uses those effective degrees of freedom. It is exact for a vector that has the
//! covariance of one replicate and conservative for one with less (placebo and stability
//! refuters, whose replicate mean is centred on the expectation).
//!
//! # Subset stability: separation of the refits
//!
//! A mean-type statistic cannot see one deciding unit through the replicate mean: when a subset
//! drops the unit the refit moves far, when it keeps it the refit stays, and the mixture mean is
//! about the original while the replicate spread (the standardiser) grows with the very shift
//! to be detected. What a deciding unit does leave is structure: the refits fall into two
//! groups, those that kept the unit and those that dropped it, separated by many within-group
//! spreads, with the full-data estimate in neither group. [`subset_separation`] measures it on
//! each coordinate: over the splits of the sorted refits into a lower and an upper group of at
//! least a tenth of the replicates each, whose gap contains the original, the gap in units of
//! the pooled within-group standard deviation; the largest value over coordinates and splits is
//! the separation. The original is required to lie inside the gap because refits on most of the
//! data that reproduce the full-data estimate in neither group state that the estimate rests on
//! the dropped units; a skewed but unimodal refit distribution has its original inside a group.
//!
//! The null law of the separation is not tied to one refit distribution. Simulating 30 to 60
//! refits per coordinate (3 to 10 coordinates, the original at the centre or drawn like a
//! refit) gives a separation below 3 for normal and `t₃` refits and a rate of at most 0.3%
//! above 8 for refits as skewed as a log-normal with scale 1.5; [`SEPARATION_THRESHOLD`] sits
//! there, so the rejection rate stays far below any usual tolerance, while one deciding unit
//! separates the groups by more than ten within-group spreads. [`separation_tail`] maps a
//! separation above the threshold to a value below the tolerance, decreasing in the
//! separation; it is a monotone yardstick, not a p-value.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// Effective degrees of freedom of a replicate covariance built from `replicates` draws of a
/// resample over `independent_units` independent units (`None`: the resample is not limited by
/// the unit count, so only the Monte-Carlo term applies).
#[must_use]
pub fn effective_covariance_df(replicates: usize, independent_units: Option<usize>) -> f64 {
    let monte_carlo = replicates.saturating_sub(1) as f64;
    match independent_units {
        Some(units) if units >= 2 => 1.0 / (1.0 / monte_carlo + 1.0 / (units - 1) as f64),
        _ => monte_carlo,
    }
}

/// Tail probability of `value = D²` on a `k`-coordinate target under the Hotelling reference with
/// [`effective_covariance_df`] degrees of freedom; `None` when there are too few effective
/// degrees of freedom to estimate a `k × k` covariance (`ν − k + 1 < 1`).
#[must_use]
pub fn replicate_reference_tail(
    value: f64,
    k: usize,
    replicates: usize,
    independent_units: Option<usize>,
) -> Option<f64> {
    let nu = effective_covariance_df(replicates, independent_units);
    let k = k as f64;
    let denominator = nu - k + 1.0;
    if denominator.is_nan() || denominator < 1.0 - 1e-9 || value.is_nan() {
        return None;
    }
    if value <= 0.0 {
        return Some(1.0);
    }
    if value.is_infinite() {
        return Some(0.0);
    }
    // D² = ν k / (ν − k + 1) · F(k, ν − k + 1); the F tail is a regularised incomplete beta.
    let f = value * denominator / (nu * k);
    let x = denominator / (denominator + k * f);
    antecedent_stats::adaptive_inference::regularized_incomplete_beta(x, 0.5 * denominator, 0.5 * k)
        .ok()
        .map(|tail| tail.clamp(0.0, 1.0))
}

/// Separation (in pooled within-group standard deviations) at which refits that split into two
/// groups around the original are taken as a deciding unit; see [`subset_separation`].
pub const SEPARATION_THRESHOLD: f64 = 8.0;

/// Separation of the replicate refits of `k`-coordinate rows around `expected` (the original).
///
/// For each coordinate the refits are sorted and every split into a lower and an upper group of
/// at least `max(3, ⌈R / 10⌉)` refits each is considered whose gap strictly contains
/// `expected`; the score is the gap divided by `sqrt(SS_within / (R − 2))`, the pooled
/// within-group standard deviation. The result is the largest score over coordinates and splits,
/// `0` when no split qualifies (too few refits, or the original lies inside a group). A split
/// whose groups have zero within-group spread carries no scale and is skipped.
#[must_use]
pub fn subset_separation(expected: &[f64], rows: &[f64], k: usize) -> f64 {
    if k == 0 || expected.len() != k || !rows.len().is_multiple_of(k) {
        return 0.0;
    }
    let r = rows.len() / k;
    let min_group = r.div_ceil(10).max(3);
    if r < 2 * min_group {
        return 0.0;
    }
    let mut best = 0.0_f64;
    let mut column = vec![0.0; r];
    for (coordinate, &e) in expected.iter().enumerate() {
        for (slot, row) in column.iter_mut().zip(rows.chunks_exact(k)) {
            *slot = row[coordinate];
        }
        column.sort_by(f64::total_cmp);
        for split in min_group..=(r - min_group) {
            let (lower, upper) = column.split_at(split);
            let (below, above) = (lower[lower.len() - 1], upper[0]);
            if !(below < e && e < above) {
                continue;
            }
            let within = |group: &[f64]| {
                let mean = group.iter().sum::<f64>() / group.len() as f64;
                group.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>()
            };
            let sd = ((within(lower) + within(upper)) / (r - 2) as f64).sqrt();
            if sd > 0.0 {
                best = best.max((above - below) / sd);
            }
        }
    }
    best
}

/// Yardstick tail of a [`subset_separation`]: `1` up to [`SEPARATION_THRESHOLD`], then
/// `alpha · (threshold / separation)²`, below `alpha` and decreasing.
#[must_use]
pub fn separation_tail(separation: f64, alpha: f64) -> f64 {
    if separation.is_nan() || separation <= SEPARATION_THRESHOLD {
        1.0
    } else {
        (alpha * (SEPARATION_THRESHOLD / separation).powi(2)).min(1.0)
    }
}

/// The replicate count a refuter draws so that [`replicate_reference_tail`] is defined.
///
/// The reference needs effective degrees of freedom `ν ≥ k`, with
/// `1/ν = 1/(R − 1) + 1/(g − 1)`, which holds exactly when `R − 1 ≥ k (g − 1) / (g − 1 − k)`.
/// That has a solution only when `g − 1 > k`: the replicate vectors of a resample with `g`
/// independent units span at most `g − 1` directions, so with `g − 1 ≤ k` the `k × k`
/// resampling covariance is singular however many replicates are drawn, and the discrepancy
/// is undefined. Then (and when no units are declared) `requested` is returned unchanged and
/// the tail is `None`. Never fewer than `requested`.
#[must_use]
pub fn replicates_for_defined_reference(
    k: usize,
    requested: u32,
    independent_units: Option<usize>,
) -> u32 {
    let Some(units) = independent_units else {
        return requested;
    };
    let g1 = units.saturating_sub(1);
    if g1 <= k {
        return requested;
    }
    let needed = (k * g1).div_ceil(g1 - k) + 1;
    requested.max(u32::try_from(needed).unwrap_or(u32::MAX))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effective_df_combines_monte_carlo_and_resampling_noise() {
        assert!((effective_covariance_df(40, None) - 39.0).abs() < 1e-12);
        let nu = effective_covariance_df(40, Some(12));
        assert!((nu - 1.0 / (1.0 / 39.0 + 1.0 / 11.0)).abs() < 1e-12);
        assert!(nu < 11.0);
        // One unit carries no information about the resampling law: only Monte-Carlo noise.
        assert!((effective_covariance_df(40, Some(1)) - 39.0).abs() < 1e-12);
    }

    #[test]
    fn two_coordinate_tail_matches_the_closed_form() {
        // For k = 2 the F(2, d) tail is (1 + 2 f / d)^(-d / 2) and D² = ν · 2 / d · f.
        for (replicates, units) in [(40, None), (40, Some(12)), (200, Some(100))] {
            let nu = effective_covariance_df(replicates, units);
            let d = nu - 1.0;
            for value in [0.5, 2.0, 5.99, 12.0] {
                let f = value * d / (2.0 * nu);
                let want = (1.0 + 2.0 * f / d).powf(-d / 2.0);
                let got = replicate_reference_tail(value, 2, replicates, units).unwrap();
                assert!((got - want).abs() < 1e-9, "{replicates} {units:?} {value}: {got} {want}");
            }
        }
    }

    #[test]
    fn the_finite_replicate_tail_is_heavier_than_the_chi_square_tail() {
        // chi-square(2) tail at 5.99 is 0.0500; forty replicates leave a heavier tail.
        let tail = replicate_reference_tail(5.991_464_547_107_98, 2, 40, None).unwrap();
        assert!(tail > 0.06 && tail < 0.07, "{tail}");
        let blocks = replicate_reference_tail(5.991_464_547_107_98, 2, 40, Some(12)).unwrap();
        assert!(blocks > tail);
    }

    #[test]
    fn replicates_are_raised_until_the_reference_is_defined() {
        // Ten coordinates from thirteen units: forty replicates leave nu below ten, so more are
        // drawn, and with them the tail is defined.
        assert!(replicate_reference_tail(1.0, 10, 30, Some(13)).is_none());
        let needed = replicates_for_defined_reference(10, 30, Some(13));
        assert!(needed > 30);
        assert!(replicate_reference_tail(1.0, 10, needed as usize, Some(13)).is_some());
        assert!(replicate_reference_tail(1.0, 10, needed as usize - 1, Some(13)).is_none());
        // Enough already, or no declared units: unchanged.
        assert_eq!(replicates_for_defined_reference(2, 30, Some(500)), 30);
        assert_eq!(replicates_for_defined_reference(2, 30, None), 30);
        // As many units as coordinates plus one: singular resampling covariance, not raised.
        assert_eq!(replicates_for_defined_reference(10, 30, Some(11)), 30);
        assert!(replicate_reference_tail(1.0, 10, 100_000, Some(11)).is_none());
    }

    #[test]
    fn degenerate_inputs_are_handled() {
        assert_eq!(replicate_reference_tail(0.0, 2, 40, None), Some(1.0));
        assert_eq!(replicate_reference_tail(f64::INFINITY, 2, 40, None), Some(0.0));
        assert_eq!(replicate_reference_tail(f64::NAN, 2, 40, None), None);
        // Three effective units cannot estimate a three-coordinate covariance.
        assert_eq!(replicate_reference_tail(1.0, 3, 40, Some(3)), None);
    }

    /// Thirty circular-window refits (80 percent of a 100-period series) of a three-coordinate
    /// target with no deciding period.
    #[allow(clippy::unreadable_literal, reason = "refit values copied as measured")]
    const CLEAN_REFITS: &[f64] = &[
        0.791696, 0.582906, 0.00580361, 0.786842, 0.612423, 0.00554714, 0.7966, 0.649485,
        0.00518934, 0.76738, 0.576477, 0.00451624, 0.823012, 0.756871, 0.0311789, 0.80622,
        0.700626, 0.0107384, 0.797117, 0.646161, 0.00542165, 0.79969, 0.743621, 0.0342633,
        0.790447, 0.60255, 0.00481456, 0.760192, 0.584959, 0.00460625, 0.746438, 0.572357,
        0.00503567, 0.704195, 0.613775, 0.00845854, 0.721882, 0.665733, 0.025906, 0.803114,
        0.613175, 0.00323924, 0.864285, 0.740672, 0.00679534, 0.758066, 0.573479, 0.00536506,
        0.789439, 0.73168, 0.0349845, 0.831432, 0.677935, 0.00460154, 0.713299, 0.550099,
        0.0015609, 0.765712, 0.637441, 0.00620369, 0.743071, 0.565724, 0.0051204, 0.710923,
        0.610208, 0.00627445, 0.83927, 0.70986, 0.00613849, 0.786842, 0.612423, 0.00554714,
        0.838495, 0.730011, 0.0106024, 0.713299, 0.550099, 0.0015609, 0.835792, 0.689634,
        0.00512518, 0.81188, 0.746865, 0.0285001, 0.758523, 0.589389, 0.00255864, 0.765712,
        0.637441, 0.00620369,
    ];
    #[allow(clippy::unreadable_literal, reason = "refit values copied as measured")]
    const CLEAN_ORIGINAL: [f64; 3] = [0.823002, 0.705772, 0.00800196];
    /// The same series with one period planted at 40 (and its outcome raised by 60): the refits
    /// that keep it sit near 30, those that drop it near 0.7, the original (25) between them.
    #[allow(clippy::unreadable_literal, reason = "refit values copied as measured")]
    const PLANTED_REFITS: &[f64] = &[
        30.5994, 29.9539, 0.0419792, 0.786842, 0.612423, 0.00554714, 33.747, 32.5021, 0.0331808,
        32.1424, 31.4457, 0.0468436, 30.4484, 29.6505, 0.0296071, 33.7738, 32.6987, 0.0309709,
        33.0957, 31.8595, 0.0325753, 26.7042, 25.9799, 0.0161469, 32.6154, 31.3224, 0.030849,
        32.2467, 31.5286, 0.0370918, 32.1055, 31.3904, 0.0375895, 30.1437, 29.4346, 0.0291226,
        0.721882, 0.665733, 0.025906, 0.803114, 0.613175, 0.00323924, 32.1738, 31.1017, 0.0308179,
        30.3808, 29.7043, 0.039203, 0.789439, 0.73168, 0.0349845, 33.0935, 31.8612, 0.0321012,
        0.713299, 0.550099, 0.0015609, 29.7308, 29.1015, 0.0361879, 32.2582, 31.5453, 0.0414805,
        29.1848, 28.5064, 0.0328772, 30.1621, 29.5049, 0.0436327, 0.786842, 0.612423, 0.00554714,
        33.2225, 32.1604, 0.0293121, 0.713299, 0.550099, 0.0015609, 0.835792, 0.689634, 0.00512518,
        29.0579, 28.2665, 0.0230903, 26.9007, 25.7468, 0.012757, 29.7308, 29.1015, 0.0361879,
    ];
    #[allow(clippy::unreadable_literal, reason = "refit values copied as measured")]
    const PLANTED_ORIGINAL: [f64; 3] = [25.1151, 24.4067, 0.0426127];

    #[test]
    fn a_deciding_unit_splits_the_refits_around_the_original() {
        let separation = subset_separation(&PLANTED_ORIGINAL, PLANTED_REFITS, 3);
        assert!(separation > 12.0, "{separation}");
        let tail = separation_tail(separation, 0.05);
        assert!(tail < 0.05, "{tail}");
    }

    #[test]
    #[allow(clippy::float_cmp, reason = "the tail below the threshold is exactly one")]
    fn refits_without_a_deciding_unit_are_not_separated() {
        let separation = subset_separation(&CLEAN_ORIGINAL, CLEAN_REFITS, 3);
        assert!(separation < SEPARATION_THRESHOLD / 4.0, "{separation}");
        assert_eq!(separation_tail(separation, 0.05), 1.0);
    }

    #[test]
    fn the_original_inside_a_group_is_not_a_split() {
        // The same two-group refits, but the original sits in the upper group.
        let separation = subset_separation(&[31.0, 30.0, 0.03], PLANTED_REFITS, 3);
        assert!(separation < SEPARATION_THRESHOLD, "{separation}");
    }

    /// Deterministic normal refits and skewed ones (a log-normal coordinate) stay below the
    /// threshold.
    #[test]
    fn smooth_and_skewed_null_refits_stay_below_the_threshold() {
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut uniform = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1_u64 << 53) as f64
        };
        let mut worst = 0.0_f64;
        for _ in 0..400 {
            let mut rows = Vec::new();
            for _ in 0..30 {
                let mut normal = || {
                    let (a, b) = (uniform().max(1e-300), uniform());
                    (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
                };
                let (a, b, c) = (normal(), normal(), normal());
                rows.extend([a, 0.5 * a + b, (0.7 * c).exp()]);
            }
            let original = [0.0, 0.0, (0.245_f64).exp()];
            worst = worst.max(subset_separation(&original, &rows, 3));
        }
        assert!(worst < SEPARATION_THRESHOLD, "{worst}");
    }

    #[test]
    #[allow(clippy::float_cmp, reason = "degenerate inputs return exact sentinels")]
    fn separation_degenerate_inputs_are_handled() {
        assert_eq!(subset_separation(&[0.0], &[0.0, 1.0, 2.0], 1), 0.0);
        assert_eq!(separation_tail(f64::NAN, 0.05), 1.0);
        assert!(separation_tail(16.0, 0.05) < 0.05);
        assert!(separation_tail(32.0, 0.05) < separation_tail(16.0, 0.05));
    }
}
