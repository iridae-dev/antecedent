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
    if denominator.is_nan() || denominator < 1.0 || value.is_nan() {
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
    fn degenerate_inputs_are_handled() {
        assert_eq!(replicate_reference_tail(0.0, 2, 40, None), Some(1.0));
        assert_eq!(replicate_reference_tail(f64::INFINITY, 2, 40, None), Some(0.0));
        assert_eq!(replicate_reference_tail(f64::NAN, 2, 40, None), None);
        // Three effective units cannot estimate a three-coordinate covariance.
        assert_eq!(replicate_reference_tail(1.0, 3, 40, Some(3)), None);
    }
}
