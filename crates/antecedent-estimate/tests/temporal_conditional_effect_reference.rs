//! The temporal conditional effect over declared modifiers against an exact rational reference.
//!
//! `oracle_data/temporal_conditional_effect_reference.py` (frozen in the JSON beside it) regresses
//! the outcome on an intercept, the treatment, the modifiers, their products with the treatment
//! and the lagged adjustment values over the intact windows, in `fractions.Fraction` arithmetic,
//! and derives the standard error of `b_T + b_{TW}' mean(W)` from the influence function of the
//! regression and the modifier means in the uncentred parametrization. The product centres the
//! modifiers at their window means and reads the treatment coefficient, so matching the
//! reference checks the centring identity, the lag alignment (the modifier is its own column and
//! the same variable at other lags stays an adjustment node), the gap handling, the contrast
//! scaling and the modifier-mean term of the variance.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, CausalSchemaBuilder, Lag, MeasurementSpec, RoleHint, SmallRoleSet,
    TemporalEffectQuery, TemporalPolicy, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
    TimeSeriesData, ValidityBitmap,
};
use antecedent_estimate::{
    AnalyticSeKind, EstimationWorkspace, TemporalLinearAdjustment,
    temporal_conditional_marginal_effect_with_gaps, temporal_conditional_marginal_fit,
};
use antecedent_graph::{TemporalDag, ensure_lagged};
use antecedent_identify::TemporalBackdoorIdentifier;
use serde_json::Value;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn series(case: &Value, weights: Option<Vec<f64>>) -> TimeSeriesData {
    let mut b = CausalSchemaBuilder::new();
    for (name, hint) in [
        ("x", RoleHint::TreatmentCandidate),
        ("y", RoleHint::OutcomeCandidate),
        ("z", RoleHint::Context),
        ("w", RoleHint::Context),
    ] {
        b.add_variable(
            name,
            ValueType::Continuous,
            SmallRoleSet::from_hint(hint),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let valid: Vec<bool> =
        case["missing"].as_array().unwrap().iter().map(|v| !v.as_bool().unwrap()).collect();
    let n = valid.len();
    let column = |id: u32, key: &str| {
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(id),
                Arc::from(floats(&case[key])),
                ValidityBitmap::from_bools(&valid),
            )
            .unwrap(),
        )
    };
    let cols = vec![column(0, "x"), column(1, "y"), column(2, "z"), column(3, "w")];
    let storage =
        OwnedColumnarStorage::try_new(schema, cols, None, weights.map(Arc::from)).unwrap();
    TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap()
}

fn graph() -> TemporalDag {
    let (x, y, z) = (VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2));
    let mut g = TemporalDag::empty();
    let x1 = ensure_lagged(&mut g, x, Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut g, y, Lag::CONTEMPORANEOUS).unwrap();
    let z1 = ensure_lagged(&mut g, z, Lag::from_raw(1)).unwrap();
    let z2 = ensure_lagged(&mut g, z, Lag::from_raw(2)).unwrap();
    for (from, to) in [(x1, y0), (z1, x1), (z1, y0), (z2, x1), (z2, y0)] {
        g.insert_directed(from, to).unwrap();
    }
    g
}

fn modifier_ids(case: &Value) -> Vec<VariableId> {
    case["modifiers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|name| VariableId::from_raw(if name.as_str().unwrap() == "z" { 2 } else { 3 }))
        .collect()
}

fn query_for(case: &Value) -> TemporalEffectQuery {
    TemporalEffectQuery::pulse(
        VariableId::from_raw(0),
        VariableId::from_raw(1),
        case["delta"].as_f64().unwrap(),
    )
    .with_policy(TemporalPolicy::pulse(-1))
    .with_horizon_steps(1)
    .with_max_history_lag(Some(2))
    .with_effect_modifiers(modifier_ids(case))
}

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/temporal_conditional_effect_reference.json"))
        .unwrap()
}

fn close(actual: f64, expected: f64, tol: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= tol * expected.abs().max(1e-3),
        "{what}: {actual} vs {expected}"
    );
}

#[test]
fn conditional_effect_and_standard_errors_match_the_exact_reference() {
    let g = graph();
    for case in reference()["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let data = series(case, None);
        let query = query_for(case);
        let id_res = TemporalBackdoorIdentifier::new().identify_temporal(&g, &query).unwrap();
        let estimand = id_res.result.estimands.first().unwrap();
        let policy = antecedent_core::ExecutionContext::for_tests(1).kernel_policy;
        let gapped = case["missing"].as_array().unwrap().iter().any(|v| v.as_bool().unwrap());
        for (kind, key) in [
            (AnalyticSeKind::Homoskedastic, "se_homoskedastic"),
            (AnalyticSeKind::Hc0, "se_hc0"),
            (AnalyticSeKind::Hc1, "se_hc1"),
        ] {
            let fit = temporal_conditional_marginal_fit(
                &data,
                estimand,
                &query,
                &id_res.indexer,
                &policy,
                gapped,
                kind,
            )
            .unwrap();
            assert_eq!(fit.rows as u64, case["rows"].as_u64().unwrap(), "{name}: rows");
            close(fit.effect, case["effect"].as_f64().unwrap(), 1e-9, &format!("{name} effect"));
            close(fit.standard_error, case[key].as_f64().unwrap(), 1e-8, &format!("{name} {key}"));
            assert_eq!(fit.selection.is_some(), gapped, "{name}: gap selection");
            if let Some(selection) = &fit.selection {
                let kept: Vec<u64> = selection
                    .complete
                    .iter()
                    .enumerate()
                    .filter(|(_, ok)| **ok)
                    .map(|(r, _)| (selection.first_time + r) as u64)
                    .collect();
                let want: Vec<u64> =
                    case["times"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
                assert_eq!(kept, want, "{name}: intact windows");
            }
        }
    }
}

#[test]
fn the_prepared_problem_fits_to_the_same_conditional_effect_and_the_float_function_agrees() {
    let g = graph();
    for case in reference()["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let data = series(case, None);
        let query = query_for(case);
        let id_res = TemporalBackdoorIdentifier::new().identify_temporal(&g, &query).unwrap();
        let estimand = id_res.result.estimands.first().unwrap();
        let ctx = antecedent_core::ExecutionContext::for_tests(1);
        let gapped = case["missing"].as_array().unwrap().iter().any(|v| v.as_bool().unwrap());
        let mut est = TemporalLinearAdjustment::new();
        est.inner.inference.bootstrap_replicates = 0;
        let prep = if gapped {
            est.prepare_with_gaps(&data, estimand, &query, &id_res.indexer, &ctx.kernel_policy)
                .unwrap()
                .0
        } else {
            est.prepare(&data, estimand, &query, &id_res.indexer, None, &ctx.kernel_policy).unwrap()
        };
        assert!(prep.method.starts_with("temporal.linear.adjustment.conditional"), "{name}");
        let effect = est
            .fit(&prep, &mut EstimationWorkspace::default(), &ctx, AssumptionSet::default())
            .unwrap();
        close(effect.ate, case["effect"].as_f64().unwrap(), 1e-9, &format!("{name} effect"));
        close(
            effect.se_analytic(),
            case["se_homoskedastic"].as_f64().unwrap(),
            1e-8,
            &format!("{name} se"),
        );
        // The estimate-only function reads the same design: one point, no ridge prior.
        let (point, _) = temporal_conditional_marginal_effect_with_gaps(
            &data,
            estimand,
            &query,
            &id_res.indexer,
            &ctx.kernel_policy,
            None,
            gapped,
        )
        .unwrap();
        close(point, case["effect"].as_f64().unwrap(), 1e-9, &format!("{name} float effect"));
    }
}

#[test]
fn a_repeated_treatment_or_outcome_modifier_a_weighted_series_and_a_bootstrap_are_refused() {
    let g = graph();
    let reference = reference();
    let case = &reference["cases"].as_array().unwrap()[0];
    let data = series(case, None);
    let ctx = antecedent_core::ExecutionContext::for_tests(1);
    let prepare = |data: &TimeSeriesData, query: &TemporalEffectQuery| {
        let id_res = TemporalBackdoorIdentifier::new().identify_temporal(&g, query).unwrap();
        let estimand = id_res.result.estimands.first().unwrap();
        TemporalLinearAdjustment::new().prepare(
            data,
            estimand,
            query,
            &id_res.indexer,
            None,
            &ctx.kernel_policy,
        )
    };
    let base = query_for(case);
    // A modifier that is the treatment or the outcome is rejected by the query itself.
    for modifier in [VariableId::from_raw(0), VariableId::from_raw(1)] {
        assert!(base.clone().with_effect_modifiers([modifier]).validate().is_err());
    }
    // A modifier named twice would be the same column twice.
    let id_res = TemporalBackdoorIdentifier::new().identify_temporal(&g, &base).unwrap();
    let estimand = id_res.result.estimands.first().unwrap();
    let twice =
        base.clone().with_effect_modifiers([VariableId::from_raw(2), VariableId::from_raw(2)]);
    let error = TemporalLinearAdjustment::new()
        .prepare(&data, estimand, &twice, &id_res.indexer, None, &ctx.kernel_policy)
        .unwrap_err();
    let (code, message) = error.coded_refusal().expect("a coded refusal");
    assert_eq!(code, "route_not_supported");
    assert_eq!(message.detail(), Some("temporal_adjustment.effect_modifiers_invalid"));
    let weights: Vec<f64> = (0..data.time_index().length).map(|i| 1.0 + (i % 3) as f64).collect();
    let error = prepare(&series(case, Some(weights)), &base).unwrap_err();
    let (_, message) = error.coded_refusal().expect("a coded refusal");
    assert_eq!(message.detail(), Some("sampling_weights.route_unsupported"));

    // The iid row bootstrap and the dependence-honest fit would hold the centring fixed.
    let id_res = TemporalBackdoorIdentifier::new().identify_temporal(&g, &base).unwrap();
    let estimand = id_res.result.estimands.first().unwrap();
    let mut est = TemporalLinearAdjustment::new();
    let prep =
        est.prepare(&data, estimand, &base, &id_res.indexer, None, &ctx.kernel_policy).unwrap();
    est.inner.inference.bootstrap_replicates = 20;
    let error = est
        .fit(&prep, &mut EstimationWorkspace::default(), &ctx, AssumptionSet::default())
        .unwrap_err();
    let (_, message) = error.coded_refusal().unwrap();
    assert_eq!(message.detail(), Some("temporal_adjustment.conditional_bootstrap"));
    let error = est
        .fit_dependence_honest(&prep, &id_res.indexer, &ctx, AssumptionSet::default())
        .unwrap_err();
    let (_, message) = error.coded_refusal().unwrap();
    assert_eq!(message.detail(), Some("temporal_adjustment.conditional_bootstrap"));
    // A covariance kind the delta-method variance does not read is refused, not substituted.
    est.inner.inference.bootstrap_replicates = 0;
    est.inner.inference.se_kind = AnalyticSeKind::Hc3;
    let error = est
        .fit(&prep, &mut EstimationWorkspace::default(), &ctx, AssumptionSet::default())
        .unwrap_err();
    let (_, message) = error.coded_refusal().unwrap();
    assert_eq!(message.detail(), Some("temporal_adjustment.conditional_covariance_kind"));
}

/// The float function reads no sampling weights, so a weighted series is refused with the same
/// coded refusal as the prepared problem, never answered unweighted: a circular-block bootstrap
/// carried as row multiplicities would otherwise return the unweighted point for every replicate.
#[test]
fn the_float_conditional_function_refuses_a_weighted_series_like_the_prepared_problem() {
    let g = graph();
    let reference = reference();
    let case = &reference["cases"].as_array().unwrap()[0];
    let base = query_for(case);
    let id_res = TemporalBackdoorIdentifier::new().identify_temporal(&g, &base).unwrap();
    let estimand = id_res.result.estimands.first().unwrap();
    let ctx = antecedent_core::ExecutionContext::for_tests(1);
    let plain = series(case, None);
    let weights: Vec<f64> = (0..plain.time_index().length).map(|i| 1.0 + (i % 3) as f64).collect();
    let weighted = series(case, Some(weights));
    assert!(
        temporal_conditional_marginal_effect_with_gaps(
            &plain,
            estimand,
            &base,
            &id_res.indexer,
            &ctx.kernel_policy,
            None,
            false,
        )
        .is_ok()
    );
    let error = temporal_conditional_marginal_effect_with_gaps(
        &weighted,
        estimand,
        &base,
        &id_res.indexer,
        &ctx.kernel_policy,
        None,
        false,
    )
    .unwrap_err();
    let (_, message) = error.coded_refusal().expect("a coded refusal");
    assert_eq!(message.detail(), Some("sampling_weights.route_unsupported"));
}
