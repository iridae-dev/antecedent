//! Marginal sensitivity model bounds and tipping point against exhaustive vertex enumeration.
//!
//! `oracle_data/msm_sensitivity_reference.py` (frozen in the JSON beside it) takes the feasible
//! set `{ rho : 1/L <= rho <= L, E_f[rho] = 1 }` of each stratum and arm as a polytope, enumerates
//! every vertex (all coordinates on a bound but at most one, fixed by the equality) in exact
//! rational arithmetic and takes the extreme values of the linear objective, with no knapsack or
//! quantile shortcut. It combines them as `sum_x P(x)(L_1 - U_0)` and `sum_x P(x)(U_1 - L_0)` and
//! finds the tipping `Lambda` by exact bisection to 1e-30. Designs cover the hand-derived binary
//! example, a zero-probability atom with tied and unsorted values, five atoms at an outcome offset
//! of 1e6 with propensities 0.001 and 0.999 and `Lambda` up to 1e6, and a single-atom design that
//! no `Lambda` moves. The test also checks the refusals of the contract, including a requested
//! sampling composition.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_validate::msm_sensitivity::{
    MsmOutcomeLaw, MsmStratum, MsmTippingDirection, msm_ate_bounds_at, msm_ate_sensitivity,
};
use antecedent_validate::{MsmSensitivitySpec, TippingStatus};
use serde_json::Value;

fn designs() -> Vec<Value> {
    serde_json::from_str::<Value>(include_str!("oracle_data/msm_sensitivity_reference.json"))
        .unwrap()["designs"]
        .as_array()
        .unwrap()
        .clone()
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn law(v: &Value) -> MsmOutcomeLaw {
    MsmOutcomeLaw { values: floats(&v["values"]), probabilities: floats(&v["probabilities"]) }
}

fn strata(design: &Value) -> Vec<MsmStratum> {
    design["strata"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| MsmStratum {
            mass: s["mass"].as_f64().unwrap(),
            propensity: s["propensity"].as_f64().unwrap(),
            treated: law(&s["treated"]),
            control: law(&s["control"]),
        })
        .collect()
}

/// Bounds are sums of products of O(1e6) outcomes with O(1) masses, so the absolute scale is the
/// offset times the machine epsilon.
fn close(got: f64, want: f64, scale: f64, what: &str) {
    assert!((got - want).abs() <= 1e-12 * (want.abs() + scale), "{what}: got {got} want {want}");
}

#[test]
fn sharp_bounds_equal_the_vertex_enumeration_at_every_lambda() {
    for design in designs() {
        let name = design["name"].as_str().unwrap();
        let strata = strata(&design);
        let scale = if name.contains("offset") { 1e6 } else { 1.0 };
        for point in design["points"].as_array().unwrap() {
            let lambda = point["lambda"].as_f64().unwrap();
            let got = msm_ate_bounds_at(&strata, lambda).unwrap();
            close(
                got.lower,
                point["lower"].as_f64().unwrap(),
                scale,
                &format!("{name} L={lambda}"),
            );
            close(
                got.upper,
                point["upper"].as_f64().unwrap(),
                scale,
                &format!("{name} U={lambda}"),
            );
        }
    }
}

#[test]
fn the_grid_and_the_identified_value_are_the_same_bounds() {
    for design in designs() {
        let name = design["name"].as_str().unwrap();
        let strata = strata(&design);
        let lambda_max = design["lambda_max"].as_f64().unwrap();
        let mut spec = MsmSensitivitySpec::new(lambda_max);
        spec.grid_points = 5;
        let result = msm_ate_sensitivity(&strata, &spec).unwrap();
        assert_eq!(result.grid.len(), 5, "{name}");
        for point in &result.grid {
            let direct = msm_ate_bounds_at(&strata, point.lambda).unwrap();
            assert_eq!((point.lower, point.upper), (direct.lower, direct.upper), "{name}");
        }
        assert_eq!(result.grid[0].lambda, 1.0);
        assert_eq!(result.grid[4].lambda, lambda_max);
        let first = design["points"][0].clone();
        if first["lambda"].as_f64() == Some(1.0) {
            let scale = if name.contains("offset") { 1e6 } else { 1.0 };
            close(result.identified, first["lower"].as_f64().unwrap(), scale, name);
        }
    }
}

#[test]
fn the_tipping_bracket_contains_the_exact_crossing() {
    let mut bracketed = 0;
    for design in designs() {
        let name = design["name"].as_str().unwrap();
        let strata = strata(&design);
        for tip in design["tipping"].as_array().unwrap() {
            let threshold = tip["threshold"].as_f64().unwrap();
            let mut spec = MsmSensitivitySpec::new(design["lambda_max"].as_f64().unwrap())
                .with_threshold(threshold);
            spec.tolerance = 1e-10;
            let got = msm_ate_sensitivity(&strata, &spec).unwrap().tipping.unwrap();
            let label = format!("{name} threshold {threshold}");
            match tip["status"].as_str().unwrap() {
                "origin" => assert_eq!(got.status, TippingStatus::ReachedAtOrigin, "{label}"),
                "not_reached" => assert_eq!(got.status, TippingStatus::NotReachedInBox, "{label}"),
                _ => {
                    bracketed += 1;
                    assert_eq!(got.status, TippingStatus::Bracketed, "{label}");
                    let want = tip["lambda"].as_f64().unwrap();
                    let bracket = got.bracket.unwrap();
                    assert!(bracket.upper - bracket.lower <= 1e-10 * (1.0 + want), "{label}");
                    // the closed form is evaluated in binary64, so allow its rounding
                    let slack = 1e-9 * want;
                    assert!(
                        bracket.lower - slack <= want && want <= bracket.upper + slack,
                        "{label}: [{}, {}] vs exact {want}",
                        bracket.lower,
                        bracket.upper
                    );
                    let downward = tip["down"].as_bool().unwrap();
                    assert_eq!(
                        got.direction,
                        if downward {
                            MsmTippingDirection::LowerBoundFalls
                        } else {
                            MsmTippingDirection::UpperBoundRises
                        },
                        "{label}"
                    );
                }
            }
        }
    }
    assert_eq!(bracketed, 3);
}

#[test]
fn declarations_the_range_cannot_honour_are_refused_not_ignored() {
    let design = &designs()[0];
    let strata = strata(design);
    let mut spec = MsmSensitivitySpec::new(3.0);
    spec.sampling_composition = Some("percentile bootstrap".to_owned());
    assert_eq!(
        msm_ate_sensitivity(&strata, &spec).unwrap_err().detail(),
        "msm_sensitivity.composition_not_licensed"
    );
    assert!(msm_ate_bounds_at(&strata, 0.99).is_err(), "Lambda below one");
    assert!(msm_ate_bounds_at(&strata, f64::NAN).is_err());
    let mut bad = strata.clone();
    bad[0].propensity = 1.0;
    assert!(msm_ate_bounds_at(&bad, 2.0).is_err(), "propensity must be inside (0, 1)");
    let mut bad = strata.clone();
    bad[0].mass = 0.6;
    assert!(msm_ate_bounds_at(&bad, 2.0).is_err(), "masses must sum to one");
    let mut bad = strata.clone();
    bad[1].treated.probabilities = vec![0.5, 0.6];
    assert!(msm_ate_bounds_at(&bad, 2.0).is_err(), "laws must sum to one");
    let mut spec = MsmSensitivitySpec::new(3.0);
    spec.tolerance = 1e-14;
    assert!(msm_ate_sensitivity(&strata, &spec).is_err());
    assert!(msm_ate_sensitivity(&strata, &MsmSensitivitySpec::new(1.0)).is_err());
}
