//! Gaussian mechanism fits and the Wald invariance comparison against a 50-digit reference.
//!
//! `oracle_data/typed_mechanism_transport_reference.py` (frozen in the JSON beside it) fits a
//! Gaussian degree-1 mechanism by ordinary least squares on parents standardized with the source
//! sample, refits the target on the source standardization, and computes the Wald statistic
//! d'(S_source + S_target)^-1 d and its chi-square upper tail from the definitions. The test fits
//! the same samples through the product, requires the parameter vectors, covariances, statistic
//! and p-value to agree, and checks the rejection decision on either side of the p-value, the
//! equal-sample case (statistic zero) and the refusal when the target is not on the source design.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_model::compose::{MechanismContext, MechanismOrigin, MechanismProvenance};
use antecedent_model::typed_mechanism_transport::{
    DETAIL_DESIGN_NOT_SHARED, FittedMechanism, MeanSpec, MechanismSpec, ResponseFamily,
    fit_mechanism, invariance_check, refit_in_target,
};
use serde_json::Value;

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap()
}

fn vec_of(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(num).collect()
}

fn columns(sample: &Value, parents: usize) -> BTreeMap<String, Vec<f64>> {
    let mut out = BTreeMap::new();
    for (j, col) in sample["columns"].as_array().unwrap().iter().enumerate() {
        out.insert(format!("p{j}"), vec_of(col));
    }
    assert_eq!(out.len(), parents);
    out.insert("y".to_owned(), vec_of(&sample["y"]));
    out
}

fn context(population: &str) -> MechanismContext {
    MechanismContext { population: population.into(), regime: "observational".into() }
}

fn provenance(population: &str) -> MechanismProvenance {
    MechanismProvenance {
        origin: MechanismOrigin::Fitted,
        artifact_id: format!("fit:{population}"),
        evidence_id: format!("evidence:{population}"),
    }
}

fn near(got: f64, want: f64, scale: f64, tolerance: f64, what: &str) {
    assert!((got - want).abs() <= tolerance * want.abs().max(scale), "{what}: {got} vs {want}");
}

fn check_fit(fit: &FittedMechanism, theta: &Value, cov: &Value, what: &str) {
    let want = vec_of(theta);
    let d = want.len();
    assert_eq!(fit.parameters().len(), d, "{what}");
    for i in 0..d {
        let scale = num(&cov[i][i]).sqrt();
        near(fit.parameters()[i], want[i], scale, 1e-9, &format!("{what} theta {i}"));
        for j in 0..d {
            near(
                fit.covariance()[i * d + j],
                num(&cov[i][j]),
                scale * num(&cov[j][j]).sqrt(),
                1e-9,
                &format!("{what} cov {i},{j}"),
            );
        }
    }
}

#[test]
fn fits_and_wald_comparison_match_the_fifty_digit_reference() {
    let reference: Value =
        serde_json::from_str(include_str!("oracle_data/typed_mechanism_transport_reference.json"))
            .unwrap();
    let cases = reference["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 5);
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let parents = case["parents"].as_u64().unwrap() as usize;
        let spec = MechanismSpec {
            variable: "y".into(),
            parents: (0..parents).map(|j| format!("p{j}")).collect(),
            mean: MeanSpec::Polynomial { degree: 1, interactions: false },
            family: ResponseFamily::Gaussian,
        };
        let source = fit_mechanism(
            &spec,
            &columns(&case["source"], parents),
            context("source"),
            provenance("source"),
            None,
        )
        .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        let target = refit_in_target(
            &source,
            &columns(&case["target"], parents),
            context("target"),
            provenance("target"),
        )
        .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        check_fit(&source, &case["source_theta"], &case["source_covariance"], name);
        check_fit(&target, &case["target_theta"], &case["target_covariance"], name);

        let p = num(&case["p_value"]);
        let check = invariance_check(&source, &target, 0.05).unwrap();
        assert_eq!(check.degrees_of_freedom, case["degrees_of_freedom"].as_u64().unwrap() as usize);
        near(check.statistic, num(&case["statistic"]), 1e-9, 1e-8, &format!("{name} statistic"));
        if p > 1e-250 {
            near(check.p_value, p, 0.0, 1e-6, &format!("{name} p-value"));
        } else {
            assert!(check.p_value < 1e-200, "{name}: {}", check.p_value);
        }
        assert_eq!(check.rejected, p < 0.05, "{name}");
        if (1e-250..0.5).contains(&p) {
            // The decision flips across the p-value and nowhere else.
            assert!(!invariance_check(&source, &target, p * 0.5).unwrap().rejected, "{name}");
            assert!(invariance_check(&source, &target, p * 2.0).unwrap().rejected, "{name}");
        }
    }
}

#[test]
fn the_same_sample_is_exactly_invariant_and_a_foreign_design_is_refused() {
    let reference: Value =
        serde_json::from_str(include_str!("oracle_data/typed_mechanism_transport_reference.json"))
            .unwrap();
    let case = &reference["cases"].as_array().unwrap()[4];
    assert_eq!(case["name"], "identical_samples");
    let spec = MechanismSpec {
        variable: "y".into(),
        parents: vec!["p0".into()],
        mean: MeanSpec::Polynomial { degree: 1, interactions: false },
        family: ResponseFamily::Gaussian,
    };
    let data = columns(&case["source"], 1);
    let a = fit_mechanism(&spec, &data, context("source"), provenance("source"), None).unwrap();
    let b = refit_in_target(&a, &data, context("target"), provenance("target")).unwrap();
    let check = invariance_check(&a, &b, 0.05).unwrap();
    assert!(check.statistic.abs() < 1e-18, "{}", check.statistic);
    assert!((check.p_value - 1.0).abs() < 1e-12);
    assert!(!check.rejected);
    // A target fitted on its own standardization is not on the source design: refused.
    let own = fit_mechanism(&spec, &data, context("target"), provenance("target"), None).unwrap();
    // Same sample, so the standardization coincides and the comparison is allowed; shift the
    // target sample so its own scaling differs and the comparison must be refused.
    let mut shifted = data.clone();
    for v in shifted.get_mut("p0").unwrap() {
        *v += 3.0;
    }
    let other =
        fit_mechanism(&spec, &shifted, context("target"), provenance("target"), None).unwrap();
    assert!(invariance_check(&a, &own, 0.05).is_ok());
    let err = invariance_check(&a, &other, 0.05).unwrap_err();
    assert!(format!("{err:?}").contains(DETAIL_DESIGN_NOT_SHARED), "{err:?}");
}
