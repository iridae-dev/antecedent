//! The linear interaction route of the conditional effect against a 50-digit least-squares
//! reference.
//!
//! `oracle_data/conditional_reference.py` (frozen in the JSON beside it) fits
//! `y ~ 1 + t + w_1..w_m + t w_1..t w_m + z..` by the normal equations at 50 digits and forms the
//! effect at the modifier means `delta (b_t + sum_k b_tw_k wbar_k)`, the standard error that adds
//! the sampling variance of the modifier means to the delta-method variance of the coefficients,
//! the two arm means, and the arm and contrast influence functions. The cases cover one to three
//! modifiers, adjustment covariates, non-binary and negative treatment levels with the contrast
//! scaled by `active - control`, a 12 percent treated arm, a modifier offset by 1e4 at scale
//! 0.05, an adjustment covariate within 1e-3 of the modifier, heteroskedastic noise and the
//! smallest admitted sample of 8 rows.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AverageEffectQuery, ConditionalEffectQuery, ExecutionContext, OutcomeFunctional,
    TargetPopulation, VariableId,
};
use antecedent_data::TabularData;
use antecedent_estimate::ConditionalLinearAdjustment;
use antecedent_expr::{ExprId, IdentifiedEstimand};
use serde_json::Value;

fn reference() -> Vec<Value> {
    serde_json::from_str::<Value>(include_str!("oracle_data/conditional_reference.json"))
        .expect("frozen reference parses")
        .as_array()
        .cloned()
        .expect("array")
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().expect("array").iter().map(|x| x.as_f64().expect("number")).collect()
}

struct Case {
    name: String,
    data: TabularData,
    estimand: IdentifiedEstimand,
    query: AverageEffectQuery,
    expected: Value,
}

fn case(json: &Value) -> Case {
    let t = floats(&json["t"]);
    let y = floats(&json["y"]);
    let w: Vec<Vec<f64>> = json["w"].as_array().unwrap().iter().map(floats).collect();
    let z: Vec<Vec<f64>> = json["z"].as_array().unwrap().iter().map(floats).collect();
    let mut columns: Vec<(String, &[f64])> =
        vec![("t".to_owned(), t.as_slice()), ("y".to_owned(), y.as_slice())];
    for (k, column) in w.iter().enumerate() {
        columns.push((format!("w{k}"), column.as_slice()));
    }
    for (k, column) in z.iter().enumerate() {
        columns.push((format!("z{k}"), column.as_slice()));
    }
    let data =
        TabularData::from_f64_columns(columns.iter().map(|(n, c)| (n.as_str(), *c))).unwrap();
    let modifiers: Vec<VariableId> =
        (0..w.len()).map(|k| VariableId::from_raw(2 + k as u32)).collect();
    let adjustment: Vec<VariableId> =
        (0..z.len()).map(|k| VariableId::from_raw((2 + w.len() + k) as u32)).collect();
    let estimand = IdentifiedEstimand::backdoor(
        "backdoor.adjustment",
        Arc::from(adjustment),
        ExprId::from_raw(0),
    );
    let query = AverageEffectQuery::with_levels(
        VariableId::from_raw(0),
        VariableId::from_raw(1),
        json["control"].as_f64().unwrap(),
        json["active"].as_f64().unwrap(),
    )
    .with_effect_modifiers(modifiers);
    Case {
        name: json["name"].as_str().unwrap().to_owned(),
        data,
        estimand,
        query,
        expected: json["expected"].clone(),
    }
}

fn near(got: f64, want: f64, rel: f64, scale: f64, what: &str) {
    assert!((got - want).abs() <= rel * scale.max(1e-12), "{what}: {got} vs {want}");
}

fn near_vec(got: &[f64], want: &[f64], rel: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    let scale = want.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        near(*g, *w, rel, scale, &format!("{what}[{i}]"));
    }
}

#[test]
fn effect_standard_error_arm_means_and_influences_equal_the_multiprecision_fit() {
    for json in reference() {
        let c = case(&json);
        // The offset and near-collinear designs lose digits to conditioning, not to the identity.
        let rel = match c.name.as_str() {
            "offset_modifier" => 1e-4,
            "near_collinear_adjustment" => 1e-6,
            _ => 1e-9,
        };
        let (estimate, means, scores) = ConditionalLinearAdjustment::new()
            .estimate_ate(&c.data, &c.estimand, &c.query)
            .unwrap_or_else(|e| panic!("{}: {e:?}", c.name));
        let want = |key: &str| c.expected[key].as_f64().unwrap();
        near(estimate.ate, want("ate"), rel, want("ate").abs(), &format!("{} ate", c.name));
        near(estimate.se_analytic(), want("se"), rel, want("se"), &format!("{} se", c.name));
        near(means[0], want("mu0"), rel, want("mu0").abs().max(1.0), &format!("{} mu0", c.name));
        near(means[1], want("mu1"), rel, want("mu1").abs().max(1.0), &format!("{} mu1", c.name));
        near(
            scores.means[1] - scores.means[0],
            estimate.ate,
            1e-9,
            estimate.ate.abs().max(1.0),
            &c.name,
        );
        near_vec(
            &scores.influence[0],
            &floats(&c.expected["if0"]),
            rel,
            &format!("{} if0", c.name),
        );
        near_vec(
            &scores.influence[1],
            &floats(&c.expected["if1"]),
            rel,
            &format!("{} if1", c.name),
        );
        let contrast = estimate.scores.influence.as_ref().expect("contrast influence");
        near_vec(
            contrast,
            &floats(&c.expected["contrast_influence"]),
            rel,
            &format!("{} contrast", c.name),
        );
        // The two public entry points agree with the direct call.
        let query = ConditionalEffectQuery::try_new(c.query.clone()).unwrap();
        let ctx = ExecutionContext::for_tests(2);
        let via = ConditionalLinearAdjustment::new()
            .estimate(&c.data, &c.estimand, &query, &ctx)
            .unwrap();
        assert_eq!(via.ate, estimate.ate, "{}", c.name);
        assert_eq!(via.se_analytic(), estimate.se_analytic(), "{}", c.name);
    }
}

#[test]
fn declared_levels_modifiers_and_population_are_used_or_refused() {
    let json = reference().into_iter().find(|c| c["name"] == "one_modifier_adjustment").unwrap();
    let c = case(&json);
    let fit = |query: &AverageEffectQuery| {
        ConditionalLinearAdjustment::new().estimate_ate(&c.data, &c.estimand, query)
    };
    // A different declared modifier set changes the answer.
    let (base, _, _) = fit(&c.query).unwrap();
    let other = c.query.clone().with_effect_modifiers([VariableId::from_raw(3)]);
    let (changed, _, _) = fit(&other).unwrap();
    assert_ne!(changed.ate, base.ate);
    // Equal levels, a non-AllObserved population and a quantile functional are refused.
    let mut same = c.query.clone();
    same.active = same.control.clone();
    assert_eq!(
        fit(&same).unwrap_err().to_string(),
        "active and control treatment levels must differ"
    );
    let treated = c.query.clone().with_target_population(TargetPopulation::Treated);
    assert_eq!(
        fit(&treated).unwrap_err().to_string(),
        "reason=population_not_estimable: ConditionalLinearAdjustment only supports AllObserved"
    );
    let quantile = c.query.clone().with_outcome_functional(OutcomeFunctional::Quantile(
        antecedent_core::OrderedFloatBits::from_f64(0.5),
    ));
    assert_eq!(
        fit(&quantile).unwrap_err().to_string(),
        "conditional quantiles require CDF-grid inversion through Study; a mean regression cannot estimate a quantile"
    );
    // A third treatment level the contrast does not name is refused rather than dropped.
    let mut third = json["t"].as_array().unwrap().clone();
    third[0] = serde_json::json!(0.5);
    let mut altered = json.clone();
    altered["t"] = serde_json::Value::Array(third);
    let c3 = case(&altered);
    let e3 = ConditionalLinearAdjustment::new().estimate_ate(&c3.data, &c3.estimand, &c3.query);
    let message = e3.unwrap_err().to_string();
    assert!(message.starts_with("reason=treatment_level_undeclared:"), "{message}");
    assert!(message.contains("first found 0.5"), "{message}");
    // Fewer than eight complete rows are refused.
    let mut short = json.clone();
    for key in ["t", "y"] {
        short[key] = serde_json::Value::Array(json[key].as_array().unwrap()[..7].to_vec());
    }
    for key in ["w", "z"] {
        short[key] = serde_json::Value::Array(
            json[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|column| serde_json::Value::Array(column.as_array().unwrap()[..7].to_vec()))
                .collect(),
        );
    }
    let c7 = case(&short);
    let e7 = ConditionalLinearAdjustment::new().estimate_ate(&c7.data, &c7.estimand, &c7.query);
    assert_eq!(
        e7.unwrap_err().to_string(),
        "invalid argument: too few complete rows for conditional ATE"
    );
}

#[test]
fn a_threshold_functional_is_refused_not_read_as_the_mean_contrast() {
    let json = reference().into_iter().find(|c| c["name"] == "one_modifier").unwrap();
    let c = case(&json);
    let exceedance = c.query.clone().with_outcome_functional(OutcomeFunctional::Exceedance(
        antecedent_core::OrderedFloatBits::from_f64(1.0),
    ));
    let refusal = ConditionalLinearAdjustment::new()
        .estimate_ate(&c.data, &c.estimand, &exceedance)
        .unwrap_err();
    assert_eq!(
        refusal.to_string(),
        "the linear interaction fit estimates a mean; an exceedance functional requires the outcome transformed to its indicator and the Mean functional"
    );
}
