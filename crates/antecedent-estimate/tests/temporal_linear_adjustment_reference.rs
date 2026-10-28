//! The lag-aligned temporal linear adjustment against an exact rational reference.
//!
//! `oracle_data/temporal_linear_adjustment_reference.py` (frozen in the JSON beside it) regresses
//! the outcome at time t on an intercept, the treatment at t - 1 and a confounder at t - 1 and
//! t - 2, over exactly the windows whose every read value is observed, in `fractions.Fraction`
//! arithmetic. The product builds the same design from a temporal graph, an identified adjustment
//! set and a query, so this checks the alignment (which row reads which lag), the gap handling
//! (a missing value removes the windows that read it and shifts none), the effect scaling by the
//! active level and the homoskedastic standard error, under large offsets, nearly collinear lags
//! and a tiny sample.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    CausalSchemaBuilder, ExecutionContext, Lag, MeasurementSpec, RoleHint, SmallRoleSet,
    TemporalEffectQuery, TemporalPolicy, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
    TimeSeriesData, ValidityBitmap,
};
use antecedent_estimate::{EstimationWorkspace, TemporalLinearAdjustment};
use antecedent_graph::{TemporalDag, ensure_lagged};
use antecedent_identify::TemporalBackdoorIdentifier;
use serde_json::Value;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn series(case: &Value) -> TimeSeriesData {
    let mut b = CausalSchemaBuilder::new();
    for (name, hint) in [
        ("x", RoleHint::TreatmentCandidate),
        ("y", RoleHint::OutcomeCandidate),
        ("z", RoleHint::Context),
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
    let missing: Vec<bool> =
        case["missing"].as_array().unwrap().iter().map(|v| v.as_bool().unwrap()).collect();
    let valid: Vec<bool> = missing.iter().map(|m| !m).collect();
    let n = missing.len();
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
    let cols = vec![column(0, "x"), column(1, "y"), column(2, "z")];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
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

fn close(actual: f64, expected: f64, tol: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= tol * expected.abs().max(1e-3),
        "{what}: {actual} vs {expected}"
    );
}

#[test]
fn aligned_windows_effect_and_standard_error_match_the_exact_reference() {
    let reference: Value =
        serde_json::from_str(include_str!("oracle_data/temporal_linear_adjustment_reference.json"))
            .unwrap();
    let g = graph();
    for case in reference["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let data = series(case);
        let query = TemporalEffectQuery::pulse(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            case["delta"].as_f64().unwrap(),
        )
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1)
        .with_max_history_lag(Some(2));
        let id_res = TemporalBackdoorIdentifier::new().identify_temporal(&g, &query).unwrap();
        let estimand = id_res.result.estimands.first().unwrap();
        let ctx = ExecutionContext::for_tests(1);
        let mut est = TemporalLinearAdjustment::new();
        est.inner.inference.bootstrap_replicates = 0;
        let gapped = case["missing"].as_array().unwrap().iter().any(|v| v.as_bool().unwrap());
        let prep = if gapped {
            let (prep, selection) = est
                .prepare_with_gaps(&data, estimand, &query, &id_res.indexer, &ctx.kernel_policy)
                .unwrap();
            // The windows kept are exactly the reference windows, labelled by their series time.
            let kept: Vec<usize> = selection
                .complete
                .iter()
                .enumerate()
                .filter(|(_, ok)| **ok)
                .map(|(r, _)| selection.first_time + r)
                .collect();
            let want: Vec<usize> = case["times"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| usize::try_from(v.as_u64().unwrap()).unwrap())
                .collect();
            assert_eq!(kept, want, "{name}: intact windows");
            prep
        } else {
            est.prepare(&data, estimand, &query, &id_res.indexer, None, &ctx.kernel_policy).unwrap()
        };
        let mut ws = EstimationWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx, id_res.result.required_assumptions).unwrap();
        close(effect.ate, case["effect"].as_f64().unwrap(), 1e-9, &format!("{name} effect"));
        close(effect.se_analytic(), case["se"].as_f64().unwrap(), 1e-8, &format!("{name} se"));
        assert_eq!(
            effect.provenance.n_obs,
            Some(case["rows"].as_u64().unwrap()),
            "{name}: analysed rows"
        );
    }
}

#[test]
fn a_query_declaring_effect_modifiers_is_refused_not_answered_with_the_average_effect() {
    let reference: Value =
        serde_json::from_str(include_str!("oracle_data/temporal_linear_adjustment_reference.json"))
            .unwrap();
    let case = &reference["cases"].as_array().unwrap()[0];
    let data = series(case);
    let g = graph();
    let query = TemporalEffectQuery::pulse(
        VariableId::from_raw(0),
        VariableId::from_raw(1),
        case["delta"].as_f64().unwrap(),
    )
    .with_policy(TemporalPolicy::pulse(-1))
    .with_horizon_steps(1)
    .with_max_history_lag(Some(2))
    .with_effect_modifiers([VariableId::from_raw(2)]);
    let id_res = TemporalBackdoorIdentifier::new().identify_temporal(&g, &query).unwrap();
    let estimand = id_res.result.estimands.first().unwrap();
    let ctx = ExecutionContext::for_tests(1);
    let est = TemporalLinearAdjustment::new();
    let error = est
        .prepare(&data, estimand, &query, &id_res.indexer, None, &ctx.kernel_policy)
        .unwrap_err();
    let (code, message) = error.coded_refusal().expect("a coded refusal");
    assert_eq!(code, "route_not_supported");
    assert_eq!(message.detail(), Some("temporal_adjustment.effect_modifiers_declared"));
    let gapped = est
        .prepare_with_gaps(&data, estimand, &query, &id_res.indexer, &ctx.kernel_policy)
        .unwrap_err();
    assert!(gapped.coded_refusal().is_some());
}
