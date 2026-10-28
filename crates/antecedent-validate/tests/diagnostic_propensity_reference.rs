//! The overlap diagnostics and the Riesz-representer sensitivity against 60-digit evaluation of
//! their definitions (`oracle_data/diagnostic_propensity_reference.py`).
//!
//! One synthetic study (160 rows, two covariates, a moderately strong propensity that reaches
//! 0.07 and 0.95) is fitted by exact Newton maximum likelihood in the reference. The test
//! drives `OverlapRefuter`, `OverlapRuleRefuter` and `RieszSensitivity` through their diagnostic
//! propensity path and compares the reported polarity, the verdicts around the declared bars, the
//! tipping grid value and the bound with the reference, and does the same for the continuous
//! treatment support diagnostic. The product's IRLS tolerance is 1e-8, so numbers agree to 1e-6.

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec,
    RoleHint, SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
};
use antecedent_estimate::{EffectEstimate, EstimationWorkspace, OverlapPolicy};
use antecedent_expr::ExprId;
use antecedent_identify::IdentifiedEstimand;
use antecedent_validate::{
    OverlapRefuter, OverlapRuleRefuter, RefutationProblem, RieszSensitivity,
};
use serde_json::Value;

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/diagnostic_propensity_reference.json"))
        .expect("frozen reference")
}

fn f(value: &Value) -> f64 {
    value.as_f64().expect("number")
}

fn column(r: &Value, name: &str) -> Vec<f64> {
    r[name].as_array().expect("column").iter().map(f).collect()
}

fn study(r: &Value) -> (TabularData, IdentifiedEstimand) {
    let names = ["t", "y", "z1", "z2", "tc"];
    let mut b = CausalSchemaBuilder::new();
    for name in names {
        b.add_variable(
            name,
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let n = r["n"].as_u64().unwrap() as usize;
    let cols = names
        .iter()
        .enumerate()
        .map(|(k, name)| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(k as u32),
                    Arc::from(column(r, name)),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            )
        })
        .collect();
    let storage = OwnedColumnarStorage::try_new(b.build().unwrap(), cols, None, None).unwrap();
    let estimand = IdentifiedEstimand::backdoor(
        "backdoor.adjustment",
        Arc::from([VariableId::from_raw(2), VariableId::from_raw(3)]),
        ExprId::from_raw(0),
    );
    (TabularData::new(storage), estimand)
}

fn estimate(ate: f64) -> EffectEstimate {
    EffectEstimate::new(ate, 0.1, AssumptionSet::new(), OverlapPolicy::ExplicitOverride)
}

fn near(got: f64, want: f64, tol: f64, what: &str) {
    assert!((got - want).abs() <= tol * (1.0 + want.abs()), "{what}: {got} vs {want}");
}

#[test]
fn overlap_assessment_reports_unsupported_ess_mass_and_judges_range_and_ess() {
    let r = reference();
    let (data, estimand) = study(&r);
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let original = estimate(2.0);
    let problem = RefutationProblem::new(&data, &estimand, &query, &original, None, None);
    for case in r["overlap"].as_array().unwrap() {
        let refuter = OverlapRefuter {
            eps: f(&case["eps"]),
            min_ess_fraction: f(&case["min_ess"]),
            ..OverlapRefuter::new()
        };
        let report = refuter.refute(&problem).unwrap();
        near(report.comparison, f(&case["comparison"]), 1e-6, &format!("{case}"));
        assert_eq!(report.passed, case["passed"].as_bool().unwrap(), "{case}");
        assert_eq!(report.passed, report.failure_condition.is_none());
    }
}

#[test]
fn overlap_rule_retains_the_mass_inside_the_declared_band() {
    let r = reference();
    let (data, estimand) = study(&r);
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let original = estimate(2.0);
    let problem = RefutationProblem::new(&data, &estimand, &query, &original, None, None);
    for case in r["rule"].as_array().unwrap() {
        let refuter = OverlapRuleRefuter {
            rule_eps: f(&case["eps"]),
            min_retained_fraction: f(&case["min_retained"]),
            ..OverlapRuleRefuter::new()
        };
        let report = refuter.refute(&problem).unwrap();
        near(report.comparison, f(&case["comparison"]), 1e-9, &format!("{case}"));
        assert_eq!(report.passed, case["passed"].as_bool().unwrap(), "{case}");
    }
}

#[test]
fn continuous_treatment_support_counts_rows_with_both_doses_inside_the_residual_band() {
    let r = reference();
    let (data, estimand) = study(&r);
    let original = estimate(0.8);
    for case in r["continuous"].as_array().unwrap() {
        let query = AverageEffectQuery::with_levels(
            VariableId::from_raw(4),
            VariableId::from_raw(1),
            f(&case["control"]),
            f(&case["active"]),
        );
        let problem = RefutationProblem::new(&data, &estimand, &query, &original, None, None);
        let refuter = OverlapRefuter {
            eps: f(&case["eps"]),
            min_ess_fraction: f(&case["min_frac"]),
            ..OverlapRefuter::new()
        };
        let report = refuter.refute(&problem).unwrap();
        assert_eq!(report.refuter.as_ref(), "overlap.continuous_support");
        near(report.comparison, f(&case["comparison"]), 1e-9, &format!("{case}"));
        assert_eq!(report.passed, case["passed"].as_bool().unwrap(), "{case}");
        // The rule refuter reaches the same support computation with its band as eps.
        let rule = OverlapRuleRefuter {
            rule_eps: f(&case["eps"]),
            min_retained_fraction: f(&case["min_frac"]),
            ..OverlapRuleRefuter::new()
        };
        let report = rule.refute(&problem).unwrap();
        near(report.comparison, f(&case["comparison"]), 1e-9, &format!("rule {case}"));
    }
}

#[test]
fn riesz_tips_at_the_first_grid_delta_whose_worst_case_bias_reaches_the_estimate() {
    let r = reference();
    let (data, estimand) = study(&r);
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let mut ws = EstimationWorkspace::default();
    let ctx = ExecutionContext::for_tests(1);
    let (mut refused, mut tipped, mut untipped) = (0, 0, 0);
    for case in r["riesz"].as_array().unwrap() {
        let original = estimate(f(&case["published"]));
        let problem = RefutationProblem::new(&data, &estimand, &query, &original, None, None);
        let riesz = RieszSensitivity {
            clip: f(&case["clip"]),
            pass_threshold: case["threshold"].as_f64().unwrap_or(0.1),
            ..RieszSensitivity::new()
        };
        let result = riesz.refute(&problem, &mut ws, &ctx);
        if case["refused"].as_bool().unwrap() {
            // An IPW contrast of the opposite sign means the diagnostic fit does not describe
            // the published estimate: refused, never recentred on the diagnostic number.
            assert!(result.is_err(), "{case}");
            refused += 1;
            continue;
        }
        let report = result.unwrap();
        match case["comparison"].as_f64() {
            Some(delta) => {
                assert_eq!(report.comparison, delta, "{case}");
                tipped += 1;
            }
            None => {
                assert!(report.comparison.is_infinite(), "{case}");
                untipped += 1;
            }
        }
        near(report.refuted_ate, f(&case["refuted"]), 1e-6, &format!("bound {case}"));
        assert_eq!(report.passed, case["passed"].as_bool().unwrap(), "{case}");
        assert_eq!(report.original_ate, f(&case["published"]));
    }
    assert!(refused >= 2 && tipped >= 8 && untipped >= 4, "{refused} {tipped} {untipped}");
}
