//! Overlap diagnostics against exact rational references.
//!
//! `oracle_data/overlap_reference.py` (frozen in the JSON beside it) rebuilds the clipped
//! estimand-specific inverse-probability weights from the definitions of the ATE, ATT, ATC and
//! observation-weighted targets, and from them the Kish effective size (all rows, treated rows,
//! control rows), the extreme-weight count over the clip-sensitivity grid, the common-support
//! share and the trimmed share, in `Fraction` arithmetic. The product report must agree.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::{IpwTarget, OverlapPolicy, OverlapReport};
use serde_json::Value;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn close(got: f64, want: f64, what: &str) {
    assert!((got - want).abs() <= 1e-12 * want.abs().max(1.0), "{what}: {got} vs {want}");
}

#[test]
fn the_overlap_report_matches_the_exact_diagnostics() {
    let cases: Vec<Value> =
        serde_json::from_str(include_str!("oracle_data/overlap_reference.json")).unwrap();
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let (p, t) = (floats(&case["propensities"]), floats(&case["treatment"]));
        let target = match case["target"].as_str().unwrap() {
            "ate" => IpwTarget::Ate,
            "att" => IpwTarget::Att,
            "atc" => IpwTarget::Atc,
            _ => IpwTarget::Custom,
        };
        let policy = OverlapPolicy::RequireDiagnostics {
            clip: case["clip"].as_f64(),
            trim: case["trim"].as_f64(),
        };
        let observation =
            case["observation_weights"].as_array().map(|_| floats(&case["observation_weights"]));
        let report = OverlapReport::from_propensities(
            &p,
            None,
            policy,
            Some(&t),
            Some(target),
            observation.as_deref(),
        );
        close(report.propensity_min, case["min"].as_f64().unwrap(), name);
        close(report.propensity_max, case["max"].as_f64().unwrap(), name);
        close(report.target_population_support, case["support"].as_f64().unwrap(), name);
        close(report.excluded_fraction, case["excluded"].as_f64().unwrap(), name);
        match case.get("grid") {
            Some(grid) => {
                let sensitivity = report.clip_sensitivity.as_ref().expect(name);
                let thresholds = floats(&grid["thresholds"]);
                assert_eq!(sensitivity.thresholds.len(), thresholds.len(), "{name} grid");
                for (k, &c) in thresholds.iter().enumerate() {
                    close(sensitivity.thresholds[k], c, name);
                    close(sensitivity.ess[k], floats(&grid["ess"])[k], &format!("{name} ess {k}"));
                    close(
                        sensitivity.treated_ess[k],
                        floats(&grid["treated_ess"])[k],
                        &format!("{name} treated ess {k}"),
                    );
                    close(
                        sensitivity.control_ess[k],
                        floats(&grid["control_ess"])[k],
                        &format!("{name} control ess {k}"),
                    );
                    assert_eq!(
                        f64::from(sensitivity.extreme_weight_counts[k]),
                        floats(&grid["extreme"])[k],
                        "{name} extreme {k}"
                    );
                }
            }
            None => assert!(report.clip_sensitivity.is_none(), "{name}: no clip, no grid"),
        }
    }
}

#[test]
fn the_undeclared_default_computes_as_the_declared_one_percent_clip() {
    let cases: Vec<Value> =
        serde_json::from_str(include_str!("oracle_data/overlap_reference.json")).unwrap();
    let case = &cases[0];
    assert_eq!(case["clip"].as_f64(), Some(0.01));
    let (p, t) = (floats(&case["propensities"]), floats(&case["treatment"]));
    let report = OverlapReport::from_propensities(
        &p,
        None,
        OverlapPolicy::Default,
        Some(&t),
        Some(IpwTarget::Ate),
        None,
    );
    let sensitivity = report.clip_sensitivity.as_ref().unwrap();
    let want = floats(&case["grid"]["ess"]);
    for (got, want) in sensitivity.ess.iter().zip(want) {
        close(*got, want, "default policy ess");
    }
    close(report.target_population_support, case["support"].as_f64().unwrap(), "default support");
}
