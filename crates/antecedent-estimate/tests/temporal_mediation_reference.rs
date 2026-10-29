//! Linear temporal mediation against an exact rational reference.
//!
//! `oracle_data/temporal_mediation_reference.py` (frozen in the JSON beside it) solves, in
//! `fractions.Fraction` arithmetic, the mediator regressions `M_j ~ [1, T, adjustment]`, the outcome
//! regression `Y ~ [1, T, M, adjustment]` and the reduced form `Y ~ [1, T, adjustment]` over the
//! intact lag windows (treatment read `h` steps back, mediators and outcome now, adjustment at its
//! own lag), and forms the total, direct and mediated effects scaled by the active-minus-control
//! level and the iid standard errors the product publishes on request. The product builds the same
//! windows from a lagged scan, so this checks the alignment, the gap handling, two mediators with a
//! shared outcome regression, a non-unit treatment contrast, large offsets and a tiny sample.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    CausalSchemaBuilder, ExecutionContext, Lag, MeasurementSpec, MediationContrast, MediationQuery,
    RoleHint, SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, LaggedColumn, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
    TimeSeriesData, ValidityBitmap,
};
use antecedent_estimate::TemporalMediationEstimator;
use antecedent_expr::{CausalExprArena, IdentifiedEstimand};
use serde_json::Value;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn close(actual: f64, expected: f64, tol: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= tol * expected.abs().max(1e-3),
        "{what}: {actual} vs {expected}"
    );
}

#[test]
fn mediation_effects_and_standard_errors_match_the_exact_reference() {
    let reference: Value =
        serde_json::from_str(include_str!("oracle_data/temporal_mediation_reference.json"))
            .unwrap();
    for case in reference["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let q = usize::try_from(case["q"].as_u64().unwrap()).unwrap();
        let h = u32::try_from(case["h"].as_u64().unwrap()).unwrap();
        let delta = case["delta"].as_f64().unwrap();
        let missing: Vec<bool> =
            case["missing"].as_array().unwrap().iter().map(|v| v.as_bool().unwrap()).collect();
        let valid: Vec<bool> = missing.iter().map(|m| !m).collect();
        let n = missing.len();

        // Variables: 0 treatment, 1..=q mediators, q + 1 outcome, q + 2 adjustment covariate.
        let mut builder = CausalSchemaBuilder::new();
        for index in 0..(q + 3) {
            builder
                .add_variable(
                    format!("v{index}"),
                    ValueType::Continuous,
                    SmallRoleSet::from_hint(RoleHint::Context),
                    None,
                    None,
                    MeasurementSpec::default(),
                )
                .unwrap();
        }
        let schema = builder.build().unwrap();
        let mut series = vec![floats(&case["t"])];
        for m in case["mediators"].as_array().unwrap() {
            series.push(floats(m));
        }
        series.push(floats(&case["y"]));
        series.push(floats(&case["z"]));
        let cols: Vec<OwnedColumn> = series
            .into_iter()
            .enumerate()
            .map(|(i, values)| {
                OwnedColumn::Float64(
                    Float64Column::new(
                        VariableId::from_raw(u32::try_from(i).unwrap()),
                        Arc::from(values),
                        ValidityBitmap::from_bools(&valid),
                    )
                    .unwrap(),
                )
            })
            .collect();
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let data = TimeSeriesData::try_new(
            storage,
            TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
        )
        .unwrap();

        let mediators: Vec<VariableId> =
            (1..=q).map(|i| VariableId::from_raw(u32::try_from(i).unwrap())).collect();
        let outcome = VariableId::from_raw(u32::try_from(q + 1).unwrap());
        let covariate = VariableId::from_raw(u32::try_from(q + 2).unwrap());
        let adjustment: Vec<LaggedColumn> = case["adj_lags"]
            .as_array()
            .unwrap()
            .iter()
            .map(|l| LaggedColumn {
                variable: covariate,
                lag: Lag::from_raw(u32::try_from(l.as_u64().unwrap()).unwrap()),
            })
            .collect();
        let treatment = VariableId::from_raw(0);
        let ctx = ExecutionContext::for_tests(1);
        let gapped = missing.iter().any(|m| *m);
        let estimator =
            TemporalMediationEstimator::new().with_allow_iid_sobel_se(true).with_allow_gaps(gapped);

        for (contrast, key) in [
            (MediationContrast::Total, "total"),
            (MediationContrast::Direct, "direct"),
            (MediationContrast::Mediated, "mediated"),
        ] {
            let mut query = MediationQuery::binary(treatment, outcome, mediators.clone(), contrast)
                .with_horizons([h])
                .unwrap();
            query.active =
                antecedent_core::Intervention::set(treatment, antecedent_core::Value::f64(delta));
            let mut arena = CausalExprArena::new();
            let functional = arena.temporal_mediation_ate(
                query.treatment,
                query.outcome,
                &query.mediators,
                antecedent_core::Value::f64(delta),
                antecedent_core::Value::f64(0.0),
            );
            let estimand = IdentifiedEstimand::temporal_mediation(
                "temporal_mediation.mediated",
                Arc::clone(&query.mediators),
                functional,
            );
            let got = estimator
                .estimate_with_adjustment(&data, &estimand, &query, &adjustment, &[], &ctx)
                .unwrap();
            close(got.effect.ate, case[key].as_f64().unwrap(), 1e-8, &format!("{name} {key}"));
            close(
                got.effect.se_analytic(),
                case[format!("se_{key}").as_str()].as_f64().unwrap(),
                1e-7,
                &format!("{name} se {key}"),
            );
            // The three contrasts are one decomposition of the same fit.
            close(
                got.total.unwrap(),
                case["total"].as_f64().unwrap(),
                1e-8,
                &format!("{name} decomposition total"),
            );
            close(
                got.total.unwrap(),
                got.direct.unwrap() + got.mediated.unwrap(),
                1e-6,
                &format!("{name} total = direct + mediated"),
            );
        }
    }
}

/// A declared target population is part of the question; an estimator that fits every observed
/// window with equal weight cannot answer it, so it must refuse rather than answer the default
/// question under the declared one's name.
#[test]
fn a_declared_target_population_is_refused_and_extra_horizons_read_the_first() {
    let reference: Value =
        serde_json::from_str(include_str!("oracle_data/temporal_mediation_reference.json"))
            .unwrap();
    let case = &reference["cases"][0];
    let n = case["missing"].as_array().unwrap().len();
    let mut builder = CausalSchemaBuilder::new();
    for index in 0..4 {
        builder
            .add_variable(
                format!("v{index}"),
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
    }
    let schema = builder.build().unwrap();
    let series =
        [floats(&case["t"]), floats(&case["mediators"][0]), floats(&case["y"]), floats(&case["z"])];
    let cols: Vec<OwnedColumn> = series
        .into_iter()
        .enumerate()
        .map(|(i, values)| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(u32::try_from(i).unwrap()),
                    Arc::from(values),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            )
        })
        .collect();
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let data = TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap();
    let estimand_for = |query: &MediationQuery| {
        let mut arena = CausalExprArena::new();
        let functional = arena.temporal_mediation_ate(
            query.treatment,
            query.outcome,
            &query.mediators,
            antecedent_core::Value::f64(1.0),
            antecedent_core::Value::f64(0.0),
        );
        IdentifiedEstimand::temporal_mediation(
            "temporal_mediation.mediated",
            Arc::clone(&query.mediators),
            functional,
        )
    };
    let base = MediationQuery::binary(
        VariableId::from_raw(0),
        VariableId::from_raw(2),
        [VariableId::from_raw(1)],
        MediationContrast::Mediated,
    );
    let ctx = ExecutionContext::for_tests(1);
    let estimator = TemporalMediationEstimator::new();
    estimator
        .estimate(&data, &estimand_for(&base), &base, &ctx)
        .expect("the default population and one horizon are fitted");

    let mut treated = base.clone();
    treated.target_population = antecedent_core::TargetPopulation::Treated;
    let refused = estimator
        .estimate(&data, &estimand_for(&treated), &treated, &ctx)
        .err()
        .expect("a declared target population was fitted as if it were all observed windows");
    assert!(matches!(refused, antecedent_estimate::EstimationError::TargetPopulation));
    assert_eq!(
        refused.to_string(),
        "reason=population_not_estimable: only TargetPopulation::AllObserved is supported on \
         this estimator path"
    );

    // Several horizons on one query are fitted at the first horizon only (the facade slices
    // the grid into one query per horizon); the result is exactly the first horizon's fit.
    let two_horizons = base.clone().with_horizons([1, 2]).unwrap();
    let first = estimator.estimate(&data, &estimand_for(&base), &base, &ctx).unwrap();
    let sliced =
        estimator.estimate(&data, &estimand_for(&two_horizons), &two_horizons, &ctx).unwrap();
    assert_eq!(first.mediated.unwrap().to_bits(), sliced.mediated.unwrap().to_bits());
}
