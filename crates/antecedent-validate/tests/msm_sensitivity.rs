//! 2.3 B3: marginal sensitivity model bounds and tipping point (hand-derived oracles).
//!
//! Two-stratum example, binary outcome, exact inputs:
//!
//! | stratum | mass | e(x) | treated rate r1 | control rate r0 |
//! |---------|------|------|-----------------|-----------------|
//! | A       | 0.5  | 0.50 | 0.8             | 0.4             |
//! | B       | 0.5  | 0.25 | 0.5             | 0.3             |
//!
//! Identified (`Lambda = 1`): `ATE_A = 0.8 - 0.4 = 0.4`, `ATE_B = 0.5 - 0.3 = 0.2`, so
//! `ATE = 0.5 * 0.4 + 0.5 * 0.2 = 0.3`.
//!
//! For a binary outcome with rate r, the sharp shifts of E[Y | x] at Lambda are
//! (hidden = unobserved fraction of the arm: 1 - e for the treated arm, e for control)
//!
//! * up   S(r) = (1 - 1/L)(1 - r) if r >= 1/(L + 1), else (L - 1) r;
//! * down D(r) = (1 - 1/L) r      if r <= L/(L + 1), else (L - 1)(1 - r);
//!
//! and `ATE lower = identified - sum_x p [(1 - e) D(r1) + e S(r0)]`,
//! `ATE upper = identified + sum_x p [(1 - e) S(r1) + e D(r0)]`.
//!
//! At `Lambda = 2` (`tau = 1/3`, threshold `L/(L+1) = 2/3`):
//!
//! * A, treated `r1 = 0.8 > 2/3`: `D = (2 - 1)(0.2) = 0.2`; `S`: 0.8 >= 1/3 so `S = 0.5 * 0.2 = 0.1`.
//!   Control `r0 = 0.4`: `S`: 0.4 >= 1/3 so `S = 0.5 * 0.6 = 0.3`; `D`: 0.4 <= 2/3 so `D = 0.5 * 0.4 = 0.2`.
//!   `lower_A = 0.4 - [0.5 * 0.2 + 0.5 * 0.3] = 0.15`; `upper_A = 0.4 + [0.5 * 0.1 + 0.5 * 0.2] = 0.55`.
//! * B, treated `r1 = 0.5`: `D = 0.5 * 0.5 = 0.25` (0.5 <= 2/3); `S = 0.5 * 0.5 = 0.25`.
//!   Control `r0 = 0.3`: `S`: 0.3 < 1/3 so `S = (2 - 1) 0.3 = 0.3`; `D = 0.5 * 0.3 = 0.15`.
//!   `lower_B = 0.2 - [0.75 * 0.25 + 0.25 * 0.3] = 0.2 - 0.2625 = -0.0625`;
//!   `upper_B = 0.2 + [0.75 * 0.25 + 0.25 * 0.15] = 0.2 + 0.225 = 0.425`.
//! * `ATE lower = 0.5 * 0.15 + 0.5 * (-0.0625) = 0.04375`; `upper = 0.5 * 0.55 + 0.5 * 0.425 = 0.4875`.
//!
//! Tipping against 0 (lower bound): for 1.5 <= L < 7/3 (A: D(0.8) = L - 1 since L < 4,
//! S(0.4) = 1 - 1/L since L >= 1.5; B: D(0.5) = (1 - 1/L)/2, S(0.3) = L - 1 since L < 7/3)
//!
//! loss(L) = 0.5 [0.5 (0.2)(L - 1) + 0.5 (0.6)(1 - 1/L)] + 0.5 [0.75 (0.5)(1 - 1/L) + 0.25 (0.3)(L - 1)]
//!         = 0.0875 (L - 1) + 0.3375 (1 - 1/L).
//!
//! loss = 0.3 with u = L - 1 gives 0.0875 u (1 + u) + 0.3375 u = 0.3 (1 + u), i.e.
//! 7 u^2 + 10 u - 24 = 0, u = (-5 + sqrt(193)) / 7, so L* = (2 + sqrt(193)) / 7 = 2.27035...
//! which lies in [1.5, 7/3), consistent with the regime used.
//!
//! Three-atom example (one stratum, e = 1/2, both laws on {0, 1, 2} with probabilities
//! {1/4, 1/2, 1/4}, mean 1): identified ATE 0; at Lambda = 2 (tau = 1/3) the top third
//! is 1/4 of y = 2 plus 1/12 of y = 1, T = 7/12, T - tau m = 1/4, spread 3/2, so
//! U = 1 + (1/2)(3/2)(1/4) = 1.1875 and L = 0.8125 for both arms, giving ATE bounds
//! [-0.375, 0.375]. At Lambda = 3 (tau = 1/4, spread 8/3): U = 4/3, L = 2/3, ATE [-2/3, 2/3].

#![allow(clippy::float_cmp, reason = "exact grid endpoints and identical computations")]

use antecedent_validate::msm_sensitivity::{
    MSM_SAMPLING_STATEMENT, MsmOutcomeLaw, MsmStratum, MsmTippingDirection, msm_ate_bounds_at,
    msm_ate_sensitivity,
};
use antecedent_validate::{MsmSensitivityError, MsmSensitivitySpec, TippingStatus};

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!((actual - expected).abs() <= tolerance, "{actual} vs {expected}");
}

fn binary(rate: f64) -> MsmOutcomeLaw {
    MsmOutcomeLaw { values: vec![0.0, 1.0], probabilities: vec![1.0 - rate, rate] }
}

fn stratum(mass: f64, propensity: f64, r1: f64, r0: f64) -> MsmStratum {
    MsmStratum { mass, propensity, treated: binary(r1), control: binary(r0) }
}

fn two_strata() -> Vec<MsmStratum> {
    vec![stratum(0.5, 0.5, 0.8, 0.4), stratum(0.5, 0.25, 0.5, 0.3)]
}

fn refusal(error: &MsmSensitivityError) -> (&'static str, &'static str) {
    (error.reason_code(), error.detail())
}

#[test]
fn b3_msm_lambda_one_is_the_stratified_identified_value() {
    let result = msm_ate_sensitivity(&two_strata(), &MsmSensitivitySpec::new(3.0)).unwrap();
    close(result.identified, 0.3, 1e-12);
    let first = result.grid[0];
    assert_eq!(first.lambda, 1.0);
    assert_eq!(first.lower, first.upper);
    assert_eq!(first.lower, result.identified);
    close(first.lower, 0.5 * 0.4 + 0.5 * 0.2, 1e-12);
    let last = result.grid.last().unwrap();
    assert_eq!(last.lambda, 3.0);
    assert_eq!(result.grid.len(), 17);
}

#[test]
fn b3_msm_sharp_bounds_at_lambda_two_match_the_hand_derivation() {
    let point = msm_ate_bounds_at(&two_strata(), 2.0).unwrap();
    close(point.lower, 0.04375, 1e-12);
    close(point.upper, 0.4875, 1e-12);
}

#[test]
fn b3_msm_three_atom_fractional_split_matches_the_hand_derivation() {
    let law =
        || MsmOutcomeLaw { values: vec![2.0, 0.0, 1.0], probabilities: vec![0.25, 0.25, 0.5] };
    let strata = vec![MsmStratum { mass: 1.0, propensity: 0.5, treated: law(), control: law() }];
    let two = msm_ate_bounds_at(&strata, 2.0).unwrap();
    close(two.lower, -0.375, 1e-12);
    close(two.upper, 0.375, 1e-12);
    let three = msm_ate_bounds_at(&strata, 3.0).unwrap();
    close(three.lower, -2.0 / 3.0, 1e-12);
    close(three.upper, 2.0 / 3.0, 1e-12);
    let one = msm_ate_bounds_at(&strata, 1.0).unwrap();
    close(one.lower, 0.0, 1e-12);
    close(one.upper, 0.0, 1e-12);
}

#[test]
fn b3_msm_tipping_lambda_for_a_zero_threshold_matches_the_closed_form() {
    let mut spec = MsmSensitivitySpec::new(4.0).with_threshold(0.0);
    spec.tolerance = 1e-12;
    let result = msm_ate_sensitivity(&two_strata(), &spec).unwrap();
    let tipping = result.tipping.unwrap();
    assert_eq!(tipping.direction, MsmTippingDirection::LowerBoundFalls);
    assert_eq!(tipping.status, TippingStatus::Bracketed);
    assert!(tipping.bracketed);
    assert_eq!(tipping.tolerance, 1e-12);
    let bracket = tipping.bracket.unwrap();
    let star = (2.0 + 193.0_f64.sqrt()) / 7.0;
    assert!(bracket.upper - bracket.lower <= 1e-12);
    assert!(bracket.lower - 1e-9 <= star && star <= bracket.upper + 1e-9);
    // The bracket is certified: not reached at its lower end, reached at its upper end.
    assert!(msm_ate_bounds_at(&two_strata(), bracket.lower).unwrap().lower > 0.0);
    assert!(msm_ate_bounds_at(&two_strata(), bracket.upper).unwrap().lower <= 0.0);
}

#[test]
fn b3_msm_upper_tipping_origin_and_unreached_statuses() {
    // Identified 0.3 below 0.5: the upper bound rises to it.
    let spec = MsmSensitivitySpec::new(4.0).with_threshold(0.5);
    let tipping = msm_ate_sensitivity(&two_strata(), &spec).unwrap().tipping.unwrap();
    assert_eq!(tipping.direction, MsmTippingDirection::UpperBoundRises);
    assert!(tipping.bracketed);
    let bracket = tipping.bracket.unwrap();
    assert!(msm_ate_bounds_at(&two_strata(), bracket.lower).unwrap().upper < 0.5);
    assert!(msm_ate_bounds_at(&two_strata(), bracket.upper).unwrap().upper >= 0.5);
    // Threshold equal to the identified value is reached at the origin.
    let identified = msm_ate_bounds_at(&two_strata(), 1.0).unwrap().lower;
    close(identified, 0.3, 1e-12);
    let origin = MsmSensitivitySpec::new(2.0).with_threshold(identified);
    let tipping = msm_ate_sensitivity(&two_strata(), &origin).unwrap().tipping.unwrap();
    assert_eq!(tipping.status, TippingStatus::ReachedAtOrigin);
    assert!(!tipping.bracketed);
    // At Lambda 2 the lower bound is 0.04375, so -10 is never reached.
    let far = MsmSensitivitySpec::new(2.0).with_threshold(-10.0);
    let tipping = msm_ate_sensitivity(&two_strata(), &far).unwrap().tipping.unwrap();
    assert_eq!(tipping.status, TippingStatus::NotReachedInBox);
    assert!(!tipping.bracketed);
    assert!(tipping.bracket.is_none());
    // No threshold, no tipping point.
    assert!(
        msm_ate_sensitivity(&two_strata(), &MsmSensitivitySpec::new(2.0))
            .unwrap()
            .tipping
            .is_none()
    );
}

#[test]
fn b3_msm_bounds_widen_with_lambda() {
    let mut spec = MsmSensitivitySpec::new(8.0);
    spec.grid_points = 33;
    let result = msm_ate_sensitivity(&two_strata(), &spec).unwrap();
    for pair in result.grid.windows(2) {
        assert!(pair[1].lambda > pair[0].lambda);
        assert!(pair[1].lower <= pair[0].lower + 1e-12, "lower must not rise");
        assert!(pair[1].upper >= pair[0].upper - 1e-12, "upper must not fall");
    }
    for point in &result.grid {
        assert!(
            point.lower <= result.identified + 1e-12 && result.identified <= point.upper + 1e-12
        );
    }
}

#[test]
fn b3_msm_refuses_lambda_below_one() {
    let error = msm_ate_bounds_at(&two_strata(), 0.5).unwrap_err();
    assert_eq!(refusal(&error), ("invalid_argument", "msm_sensitivity.lambda_below_one"));
    let error = msm_ate_sensitivity(&two_strata(), &MsmSensitivitySpec::new(0.9)).unwrap_err();
    assert_eq!(refusal(&error), ("invalid_argument", "msm_sensitivity.lambda_below_one"));
    let error = msm_ate_sensitivity(&two_strata(), &MsmSensitivitySpec::new(1.0)).unwrap_err();
    assert_eq!(refusal(&error), ("invalid_argument", "msm_sensitivity.lambda_range_empty"));
    let error = msm_ate_bounds_at(&two_strata(), f64::NAN).unwrap_err();
    assert_eq!(error.detail(), "msm_sensitivity.lambda_below_one");
}

#[test]
fn b3_msm_refuses_propensities_on_the_boundary() {
    for propensity in [0.0, 1.0, -0.1, 1.2, f64::NAN] {
        let strata = vec![stratum(0.5, 0.5, 0.8, 0.4), stratum(0.5, propensity, 0.5, 0.3)];
        let error = msm_ate_bounds_at(&strata, 2.0).unwrap_err();
        assert_eq!(refusal(&error), ("route_not_supported", "msm_sensitivity.positivity"));
    }
}

#[test]
fn b3_msm_refuses_malformed_inputs_and_sampling_composition() {
    let mut strata = two_strata();
    strata[0].mass = 0.6;
    let error = msm_ate_bounds_at(&strata, 2.0).unwrap_err();
    assert_eq!(refusal(&error), ("invalid_argument", "msm_sensitivity.stratum_mass"));
    let mut strata = two_strata();
    strata[1].treated.probabilities = vec![0.5, 0.6];
    let error = msm_ate_bounds_at(&strata, 2.0).unwrap_err();
    assert_eq!(refusal(&error), ("invalid_argument", "msm_sensitivity.outcome_law"));
    let mut strata = two_strata();
    strata[1].control.values = vec![f64::INFINITY, 1.0];
    assert_eq!(
        msm_ate_bounds_at(&strata, 2.0).unwrap_err().detail(),
        "msm_sensitivity.outcome_law"
    );
    let error = msm_ate_bounds_at(&[], 2.0).unwrap_err();
    assert_eq!(refusal(&error), ("route_not_supported", "msm_sensitivity.bounds_exceeded"));

    let mut spec = MsmSensitivitySpec::new(2.0);
    spec.sampling_composition = Some("percentile_bootstrap_of_bounds".into());
    let error = msm_ate_sensitivity(&two_strata(), &spec).unwrap_err();
    assert_eq!(refusal(&error), ("cell_not_licensed", "msm_sensitivity.composition_not_licensed"));
    let mut spec = MsmSensitivitySpec::new(2.0);
    spec.grid_points = 1;
    assert_eq!(
        msm_ate_sensitivity(&two_strata(), &spec).unwrap_err().detail(),
        "msm_sensitivity.bounds_exceeded"
    );
    let mut spec = MsmSensitivitySpec::new(2.0);
    spec.tolerance = 0.5;
    assert_eq!(
        msm_ate_sensitivity(&two_strata(), &spec).unwrap_err().detail(),
        "msm_sensitivity.invalid_tolerance"
    );
    let spec = MsmSensitivitySpec::new(2.0).with_threshold(f64::NAN);
    assert_eq!(
        msm_ate_sensitivity(&two_strata(), &spec).unwrap_err().detail(),
        "msm_sensitivity.invalid_threshold"
    );
}

#[test]
fn b3_msm_is_invariant_to_stratum_order() {
    let forward = two_strata();
    let mut reversed = two_strata();
    reversed.reverse();
    let spec = MsmSensitivitySpec::new(4.0).with_threshold(0.0);
    let a = msm_ate_sensitivity(&forward, &spec).unwrap();
    let b = msm_ate_sensitivity(&reversed, &spec).unwrap();
    close(a.identified, b.identified, 1e-12);
    for (x, y) in a.grid.iter().zip(&b.grid) {
        close(x.lower, y.lower, 1e-12);
        close(x.upper, y.upper, 1e-12);
    }
    let (ta, tb) = (a.tipping.unwrap().bracket.unwrap(), b.tipping.unwrap().bracket.unwrap());
    close(ta.lower, tb.lower, 1e-8);
}

#[test]
fn b3_msm_keeps_the_assumption_range_distinct_from_sampling() {
    let result = msm_ate_sensitivity(&two_strata(), &MsmSensitivitySpec::new(2.0)).unwrap();
    assert_eq!(result.inference_claim, "assumption_range");
    assert_eq!(result.uncertainty.sampling_interval, "sampling interval: not reported");
    assert_eq!(result.uncertainty.sampling_interval, MSM_SAMPLING_STATEMENT);
    assert_eq!(result.uncertainty.reason_code, "cell_not_licensed");
    assert_eq!(result.uncertainty.detail, "msm_sensitivity.interval_withheld");
    assert!(result.normalization.contains("not Hajek"));
}
