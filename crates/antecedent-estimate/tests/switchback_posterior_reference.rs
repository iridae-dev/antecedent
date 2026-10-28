//! The switchback posterior against an exact dense Normal-Inverse-Gamma reference.
//!
//! `oracle_data/switchback_posterior_reference.py` (frozen in the JSON beside it) builds the
//! block-diagonal covariance of the declared dependence, inverts it densely and forms the
//! posterior from the residual rather than from the sufficient statistics, in 60-digit
//! arithmetic with the Student-t quantile and sign probability from the incomplete beta
//! function. The cases include compound symmetry at 1e-3 above its admissible lower bound,
//! autoregressive correlation 0.999, unequal sequence lengths, negligible and overwhelming
//! priors and outcomes offset by 1e3 to 1e7 under a negligible intercept prior.

use antecedent_estimate::switchback_posterior::{
    CarryoverDeclaration, PeriodAdjustment, SequenceDependence, SwitchbackModel, SwitchbackPrior,
    SwitchbackRows, posterior_switchback,
};
use serde_json::Value;

fn reference() -> Vec<Value> {
    serde_json::from_str::<Value>(include_str!("oracle_data/switchback_posterior_reference.json"))
        .expect("frozen reference parses")
        .as_array()
        .cloned()
        .expect("array")
}

fn f(v: &Value, key: &str) -> f64 {
    v[key].as_f64().unwrap_or_else(|| panic!("{key}"))
}

fn close(got: f64, want: f64, rel: f64, what: &str) {
    assert!((got - want).abs() <= rel * want.abs().max(1e-300), "{what}: {got} vs {want}");
}

fn check(name: &str, part: &str, summary: &antecedent_estimate::gaussian_linear_posterior::PosteriorSummary, want: &Value, level_rel: f64) {
    let what = format!("{name}/{part}");
    close(summary.mean, f(want, "location"), level_rel, &what);
    if let Some(sd) = want["sd"].as_f64() {
        close(summary.standard_deviation, sd, level_rel, &what);
    }
    close(summary.degrees_of_freedom, f(want, "df"), 1e-12, &what);
    let radius = f(want, "interval_radius");
    close((summary.interval[1] - summary.interval[0]) / 2.0, radius, level_rel.max(1e-8), &what);
    close(summary.probability_positive, f(want, "probability_positive"), level_rel.max(1e-8), &what);
}

fn run(case: &Value) -> antecedent_estimate::switchback_posterior::SwitchbackPosterior {
    let mut outcomes = Vec::new();
    let mut assignment = Vec::new();
    let mut ids: Vec<String> = Vec::new();
    for s in case["sequences"].as_array().unwrap() {
        for (y, a) in s["y"].as_array().unwrap().iter().zip(s["assign"].as_array().unwrap()) {
            outcomes.push(y.as_f64().unwrap());
            assignment.push(a.as_bool().unwrap());
            ids.push(s["id"].as_str().unwrap().to_owned());
        }
    }
    let id_refs: Vec<&str> = ids.iter().map(String::as_str).collect();
    let rho = f(case, "rho");
    let dependence = match case["kind"].as_str().unwrap() {
        "independent" => SequenceDependence::Independent,
        "compound" => SequenceDependence::CompoundSymmetric { rho },
        _ => SequenceDependence::Autoregressive { rho },
    };
    let model = SwitchbackModel {
        carryover: if case["carryover"].as_bool().unwrap() {
            CarryoverDeclaration::FirstOrder
        } else {
            CarryoverDeclaration::None
        },
        dependence,
        periods: if case["period_effects"].as_bool().unwrap() {
            PeriodAdjustment::FixedEffects
        } else {
            PeriodAdjustment::None
        },
        prior: SwitchbackPrior {
            effect_precision: f(case, "effect_precision"),
            nuisance_precision: f(case, "nuisance_precision"),
            variance_shape: f(case, "variance_shape"),
            variance_scale: f(case, "variance_scale"),
        },
    };
    posterior_switchback(
        &SwitchbackRows { outcomes: &outcomes, assignment: &assignment, sequence_ids: &id_refs },
        &model,
        f(case, "level"),
    )
    .unwrap_or_else(|e| panic!("{}: {e}", case["name"]))
}

#[test]
fn every_effect_matches_the_exact_posterior() {
    for case in reference() {
        let name = case["name"].as_str().unwrap();
        let post = run(&case);
        let want = &case["result"];
        // Strict agreement is claimed up to the conditioning of the data and the declared
        // correlation; the near-singular correlations are held to a looser, stated bound.
        // The coefficients come from the normal equations, whose absolute error scales with the
        // outcome offset (about 1e-9 at 1e7 against an effect of order one).
        let rel = if name.contains("near_unit") || name.contains("near_admissible") {
            1e-6
        } else if name.contains("offset_1e+07") {
            1e-7
        } else {
            1e-9
        };
        let (shape, scale) = post.variance_shape_and_scale;
        close(shape, f(&case, "shape"), 1e-12, name);
        close(scale, f(&case, "scale"), rel, &format!("{name}/scale"));
        check(name, "direct", &post.direct_effect, &want["direct"], rel);
        if let Some(c) = &post.carryover_effect {
            check(name, "carryover", c, &want["carryover"], rel);
            check(name, "sustained", post.sustained_effect.as_ref().unwrap(), &want["sustained"], rel);
        } else {
            assert!(want.get("carryover").is_none(), "{name}");
        }
        let periods = want["periods"].as_array().unwrap();
        assert_eq!(post.period_contrasts.len(), periods.len(), "{name}");
        for (j, (got, w)) in post.period_contrasts.iter().zip(periods).enumerate() {
            check(name, &format!("period{j}"), got, w, rel);
        }
    }
}
