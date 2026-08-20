//! Supporting temporal-response fixtures: confounding and horizon-varying support.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss)]

use std::sync::Arc;

use antecedent::{RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, CausalSchemaBuilder, ContinuousDomain, ExecutionContext, GridSpec, Intervention,
    Lag, MeasurementSpec, ResponseFunctional, ResponseIdentification, ResponseQuery, ResponseValue,
    RoleHint, SmallRoleSet, SupportStatus, TemporalEffectQuery, TemporalNodeKey, TemporalPolicy,
    TemporalResponseSpec, Value, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
    TimeSeriesData, ValidityBitmap,
};
use antecedent_graph::{TemporalDag, ensure_lagged};
use antecedent_identify::{
    IdentificationStatus, TemporalBackdoorIdentifier, TemporalIdentificationResult,
};

fn confounded_fixture() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../conformance/response/temporal_confounded_pulse/expected.json"
    ))
    .unwrap()
}

fn horizon_support_fixture() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../conformance/response/temporal_horizon_support/expected.json"
    ))
    .unwrap()
}

fn f64s(value: &serde_json::Value) -> Vec<f64> {
    value.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()
}

fn series(
    names: &[&str],
    columns: Vec<Vec<f64>>,
    edges: &[(u32, u32, u32, u32)],
) -> (TimeSeriesData, TemporalDag) {
    let n = columns[0].len();
    let mut builder = CausalSchemaBuilder::new();
    for (index, name) in names.iter().enumerate() {
        let hint = match *name {
            "t" => RoleHint::TreatmentCandidate,
            "y" => RoleHint::OutcomeCandidate,
            _ => RoleHint::Context,
        };
        builder
            .add_variable(
                *name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(hint),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        assert_eq!(columns[index].len(), n);
    }
    let schema = builder.build().unwrap();
    let owned: Vec<OwnedColumn> = columns
        .into_iter()
        .enumerate()
        .map(|(index, values)| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(u32::try_from(index).unwrap()),
                    Arc::from(values),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            )
        })
        .collect();
    let storage = OwnedColumnarStorage::try_new(schema, owned, None, None).unwrap();
    let data = TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap();
    let mut graph = TemporalDag::empty();
    for &(from, from_lag, to, to_lag) in edges {
        let src =
            ensure_lagged(&mut graph, VariableId::from_raw(from), Lag::from_raw(from_lag)).unwrap();
        let dst =
            ensure_lagged(&mut graph, VariableId::from_raw(to), Lag::from_raw(to_lag)).unwrap();
        graph.insert_directed(src, dst).unwrap();
    }
    (data, graph)
}

fn mean_curve_query(doses: &[f64], fixture: &serde_json::Value) -> ResponseQuery {
    mean_curve_query_horizons(doses, fixture, None)
}

fn mean_curve_query_horizons(
    doses: &[f64],
    fixture: &serde_json::Value,
    horizons: Option<Vec<u32>>,
) -> ResponseQuery {
    let horizons = horizons.unwrap_or_else(|| {
        fixture["contract"]["horizons"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| u32::try_from(value.as_u64().unwrap()).unwrap())
            .collect()
    });
    let at = i32::try_from(fixture["contract"]["policy"]["at"].as_i64().unwrap()).unwrap();
    ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from(doses.to_vec())),
        ),
    })
    .with_temporal(TemporalResponseSpec::new(horizons, TemporalPolicy::pulse(at), None).unwrap())
}

fn surface_mean(result: &antecedent::result::StudyResult) -> Arc<[f64]> {
    let response = result.response.as_ref().expect("temporal response payload");
    let ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) =
        &response.estimate
    else {
        panic!("expected point-identified surface");
    };
    Arc::clone(mean)
}

fn confounded_columns(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let z: Vec<f64> = (0..n).map(|i| if i % 4 == 0 || i % 4 == 1 { 1.0 } else { -1.0 }).collect();
    let t: Vec<f64> =
        (0..n).map(|i| z[i] + if i % 4 == 0 || i % 4 == 2 { 1.0 } else { -1.0 }).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            1.0 + 2.0 * i.checked_sub(1).map_or(0.0, |j| t[j])
                + 5.0 * i.checked_sub(1).map_or(0.0, |j| z[j])
        })
        .collect();
    (t, y, z)
}

fn names(fixture: &serde_json::Value) -> Vec<&str> {
    fixture["contract"]["variables"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_str().unwrap())
        .collect()
}

fn var_id(names: &[&str], name: &str) -> VariableId {
    let index = names.iter().position(|n| *n == name).unwrap_or_else(|| panic!("unknown {name}"));
    VariableId::from_raw(u32::try_from(index).unwrap())
}

fn ident_key(value: &serde_json::Value) -> (String, i32) {
    (
        value["variable"].as_str().unwrap().to_string(),
        i32::try_from(value["offset"].as_i64().unwrap()).unwrap(),
    )
}

fn named_adjustment(id: &TemporalIdentificationResult, names: &[&str]) -> Vec<(String, i32)> {
    id.result.estimands[0]
        .adjustment_set
        .iter()
        .map(|&dense| {
            let key = id.indexer.key_of(dense.raw()).expect("dense adjustment id");
            (names[key.variable.as_usize()].to_string(), key.offset)
        })
        .collect()
}

fn expected_named_adjustment(value: &serde_json::Value) -> Vec<(String, i32)> {
    value.as_array().unwrap().iter().map(ident_key).collect()
}

fn parse_id_status(value: &str) -> IdentificationStatus {
    match value {
        "NonparametricallyIdentified" => IdentificationStatus::NonparametricallyIdentified,
        other => panic!("unknown identification status {other}"),
    }
}

fn assert_identified_estimand(
    id: &TemporalIdentificationResult,
    spec: &serde_json::Value,
    names: &[&str],
    label: &str,
) {
    let expected_status = parse_id_status(spec["status"].as_str().unwrap());
    let expected_method = spec["method"].as_str().unwrap();
    assert_eq!(id.result.status, expected_status, "{label}: identification status");
    assert_eq!(id.result.estimands[0].method.as_ref(), expected_method, "{label}: method");
    if spec.get("treatment").is_some() {
        let (treatment, offset) = ident_key(&spec["treatment"]);
        assert_eq!(
            id.treatment_key,
            TemporalNodeKey { variable: var_id(names, &treatment), offset },
            "{label}: treatment key"
        );
    }
    if spec.get("outcome").is_some() {
        let (outcome, offset) = ident_key(&spec["outcome"]);
        assert_eq!(
            id.outcome_key,
            TemporalNodeKey { variable: var_id(names, &outcome), offset },
            "{label}: outcome key"
        );
    }
    let expected_z = expected_named_adjustment(&spec["adjustment_set"]);
    assert_eq!(named_adjustment(id, names), expected_z, "{label}: adjustment set");
    if !expected_z.is_empty() {
        assert!(
            !id.result.estimands[0].adjustment_set.is_empty(),
            "{label}: empty Z with method {expected_method} is the schedule-ID relabel bug"
        );
    }
}

fn assert_study_used_ident(
    result: &antecedent::result::StudyResult,
    id: &TemporalIdentificationResult,
    spec: &serde_json::Value,
    estimator: &str,
    label: &str,
) {
    assert_eq!(result.identification.status, id.result.status, "{label}: study status");
    assert_eq!(
        result.estimand.method.as_ref(),
        spec["method"].as_str().unwrap(),
        "{label}: study method"
    );
    assert_eq!(
        result.estimand.adjustment_set.as_ref(),
        id.result.estimands[0].adjustment_set.as_ref(),
        "{label}: study used the identified Z, not a relabeled empty set"
    );
    assert_eq!(result.logical_plan.estimator.as_deref(), Some(estimator), "{label}: estimator");
}

fn run_temporal(
    data: &TimeSeriesData,
    graph: &TemporalDag,
    query: TemporalEffectQuery,
    ctx: &ExecutionContext,
) -> antecedent::result::StudyResult {
    Study::series(data.clone())
        .graph(graph.clone())
        .temporal_query(query)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(ctx)
        .unwrap()
}

fn prepare_then_estimate(
    data: &TimeSeriesData,
    graph: &TemporalDag,
    query: TemporalEffectQuery,
    ctx: &ExecutionContext,
) -> antecedent::result::StudyResult {
    Study::series(data.clone())
        .graph(graph.clone())
        .temporal_query(query)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(ctx)
        .unwrap()
        .estimate_series(data, ctx)
        .unwrap()
}

fn assert_ate(result: &antecedent::result::StudyResult, expected: f64, atol: f64, label: &str) {
    assert!(
        (result.estimate.ate - expected).abs() <= atol,
        "{label}: ate={} expected={expected}",
        result.estimate.ate
    );
}

#[test]
fn confounded_pulse_recovers_structural_surface_and_unadjusted_does_not() {
    let fixture = confounded_fixture();
    let n = usize::try_from(fixture["generation"]["n"].as_u64().unwrap()).unwrap();
    let names = names(&fixture);
    let ident = &fixture["contract"]["identification"];
    let (t, y, z) = confounded_columns(n);
    let (data, graph) = series(
        &names,
        vec![t.clone(), y.clone(), z.clone()],
        &[(2, 0, 0, 0), (2, 1, 1, 0), (0, 1, 1, 0)],
    );
    let doses = f64s(&fixture["contract"]["dose_grid"]);
    let expected = f64s(&fixture["contract"]["surface"]["mean"]);
    let unadjusted_expected = f64s(&fixture["contract"]["unadjusted"]["surface"]["mean"]);
    let atol = fixture["tolerance"]["atol"].as_f64().unwrap();
    let query = mean_curve_query(&doses, &fixture);
    let ctx = ExecutionContext::for_tests(21);
    let at = i32::try_from(fixture["contract"]["policy"]["at"].as_i64().unwrap()).unwrap();
    let horizon = u32::try_from(fixture["contract"]["horizons"][0].as_u64().unwrap()).unwrap();
    let id_query = TemporalEffectQuery::pulse(var_id(&names, "t"), var_id(&names, "y"), 1.0)
        .with_policy(TemporalPolicy::pulse(at))
        .with_horizon_steps(horizon);
    let identifier = TemporalBackdoorIdentifier::new();
    let id = identifier.identify_temporal(&graph, &id_query).unwrap();
    assert_identified_estimand(&id, ident, &names, "response");

    let adjusted = Study::series(data.clone())
        .graph(graph.clone())
        .query(CausalQuery::Response(query.clone()))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    assert_study_used_ident(
        &adjusted,
        &id,
        ident,
        fixture["contract"]["estimators"]["response"].as_str().unwrap(),
        "response",
    );
    let mean = surface_mean(&adjusted);
    assert_eq!(mean.len(), expected.len());
    for (index, (&actual, &truth)) in mean.iter().zip(&expected).enumerate() {
        assert!((actual - truth).abs() <= atol, "adjusted[{index}]={actual} truth={truth}");
    }
    assert_eq!(adjusted.response.as_ref().unwrap().support.status, SupportStatus::Supported);

    let prepared = Study::series(data.clone())
        .graph(graph.clone())
        .query(CausalQuery::Response(query.clone()))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let click = prepared.estimate_series(&data, &ctx).unwrap();
    assert_study_used_ident(
        &click,
        &id,
        ident,
        fixture["contract"]["estimators"]["response"].as_str().unwrap(),
        "prepared response",
    );
    let click_mean = surface_mean(&click);
    for (index, (&actual, &truth)) in click_mean.iter().zip(&expected).enumerate() {
        assert!((actual - truth).abs() <= atol, "prepared[{index}]={actual} truth={truth}");
    }

    let (unadj_data, unadj_graph) = series(&names, vec![t, y, z], &[(0, 1, 1, 0)]);
    let unadj_id = identifier.identify_temporal(&unadj_graph, &id_query).unwrap();
    assert_identified_estimand(
        &unadj_id,
        &fixture["contract"]["unadjusted"]["identification"],
        &names,
        "unadjusted response",
    );
    let unadjusted = Study::series(unadj_data)
        .graph(unadj_graph)
        .query(CausalQuery::Response(query))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    assert_study_used_ident(
        &unadjusted,
        &unadj_id,
        &fixture["contract"]["unadjusted"]["identification"],
        fixture["contract"]["estimators"]["response"].as_str().unwrap(),
        "unadjusted response",
    );
    let un_mean = surface_mean(&unadjusted);
    for (index, (&actual, &truth)) in un_mean.iter().zip(&unadjusted_expected).enumerate() {
        assert!((actual - truth).abs() <= atol, "unadjusted[{index}]={actual} truth={truth}");
    }
    assert!((un_mean[1] - un_mean[0] - (mean[1] - mean[0])).abs() > 1.0);
}

#[test]
fn confounded_pulse_and_sustained_match_structural_contrast() {
    let fixture = confounded_fixture();
    let n = usize::try_from(fixture["generation"]["n"].as_u64().unwrap()).unwrap();
    let names = names(&fixture);
    let ident = &fixture["contract"]["identification"];
    let (t, y, z) = confounded_columns(n);
    let (data, graph) = series(
        &names,
        vec![t.clone(), y.clone(), z.clone()],
        &[(2, 0, 0, 0), (2, 1, 1, 0), (0, 1, 1, 0)],
    );
    let atol = fixture["tolerance"]["atol"].as_f64().unwrap();
    let expected = fixture["contract"]["pulse_effect_projection"]["contrast"].as_f64().unwrap();
    let unadjusted_expected = fixture["contract"]["unadjusted"]["pulse_contrast"].as_f64().unwrap();
    let ctx = ExecutionContext::for_tests(23);
    let at = i32::try_from(fixture["contract"]["policy"]["at"].as_i64().unwrap()).unwrap();
    let horizon = u32::try_from(fixture["contract"]["horizons"][0].as_u64().unwrap()).unwrap();
    let pulse = TemporalEffectQuery::pulse(var_id(&names, "t"), var_id(&names, "y"), 1.0)
        .with_policy(TemporalPolicy::pulse(at))
        .with_horizon_steps(horizon);
    let sustained =
        TemporalEffectQuery::sustained(var_id(&names, "t"), var_id(&names, "y"), 0, 1.0)
            .with_policy(TemporalPolicy::sustained(at, at))
            .with_horizon_steps(horizon);
    let identifier = TemporalBackdoorIdentifier::new();
    let pulse_id = identifier.identify_temporal(&graph, &pulse).unwrap();
    let sustained_id = identifier.identify_temporal(&graph, &sustained).unwrap();
    assert_identified_estimand(&pulse_id, ident, &names, "pulse");
    assert_identified_estimand(&sustained_id, ident, &names, "sustained");
    assert_eq!(
        pulse_id.result.estimands[0].adjustment_set.as_ref(),
        sustained_id.result.estimands[0].adjustment_set.as_ref(),
        "single-step Sustained must reuse the Pulse backdoor set, not empty schedule ID"
    );

    for (label, query, id) in
        [("pulse", pulse.clone(), &pulse_id), ("sustained", sustained, &sustained_id)]
    {
        let estimator = fixture["contract"]["estimators"][label].as_str().unwrap();
        let result = run_temporal(&data, &graph, query, &ctx);
        assert_study_used_ident(&result, id, ident, estimator, label);
        assert_ate(&result, expected, atol, label);
    }

    let click = prepare_then_estimate(&data, &graph, pulse.clone(), &ctx);
    assert_study_used_ident(
        &click,
        &pulse_id,
        ident,
        fixture["contract"]["estimators"]["pulse"].as_str().unwrap(),
        "prepared pulse",
    );
    assert_ate(&click, expected, atol, "prepared pulse");

    let (unadj_data, unadj_graph) = series(&names, vec![t, y, z], &[(0, 1, 1, 0)]);
    let unadj_id = identifier.identify_temporal(&unadj_graph, &pulse).unwrap();
    assert_identified_estimand(
        &unadj_id,
        &fixture["contract"]["unadjusted"]["identification"],
        &names,
        "unadjusted pulse",
    );
    let unadjusted = run_temporal(&unadj_data, &unadj_graph, pulse, &ctx);
    assert_study_used_ident(
        &unadjusted,
        &unadj_id,
        &fixture["contract"]["unadjusted"]["identification"],
        fixture["contract"]["estimators"]["pulse"].as_str().unwrap(),
        "unadjusted pulse",
    );
    assert_ate(&unadjusted, unadjusted_expected, atol, "unadjusted pulse");
}

#[test]
fn confounded_multi_horizon_identifies_per_horizon_not_at_max() {
    let fixture = confounded_fixture();
    let n = usize::try_from(fixture["generation"]["n"].as_u64().unwrap()).unwrap();
    let names = names(&fixture);
    let spec = &fixture["contract"]["multi_horizon"];
    let (t, y, z) = confounded_columns(n);
    let (data, graph) = series(&names, vec![t, y, z], &[(2, 0, 0, 0), (2, 1, 1, 0), (0, 1, 1, 0)]);
    let doses = f64s(&fixture["contract"]["dose_grid"]);
    let horizons: Vec<u32> = spec["horizons"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| u32::try_from(value.as_u64().unwrap()).unwrap())
        .collect();
    let expected = f64s(&spec["surface"]["mean"]);
    let atol = fixture["tolerance"]["atol"].as_f64().unwrap();
    let h2_atol = spec["horizon_2_atol"].as_f64().unwrap();
    let query = mean_curve_query_horizons(&doses, &fixture, Some(horizons.clone()));
    let ctx = ExecutionContext::for_tests(29);
    let at = i32::try_from(fixture["contract"]["policy"]["at"].as_i64().unwrap()).unwrap();
    let identifier = TemporalBackdoorIdentifier::new();
    let mut per_horizon_id = Vec::new();
    for (index, &horizon) in horizons.iter().enumerate() {
        let id_query = TemporalEffectQuery::pulse(var_id(&names, "t"), var_id(&names, "y"), 1.0)
            .with_policy(TemporalPolicy::pulse(at))
            .with_horizon_steps(horizon);
        let id = identifier.identify_temporal(&graph, &id_query).unwrap();
        assert_identified_estimand(
            &id,
            &spec["identification"][index],
            &names,
            &format!("h={horizon}"),
        );
        per_horizon_id.push(id);
    }
    assert!(
        !per_horizon_id[0].result.estimands[0].adjustment_set.is_empty(),
        "horizon 1 must keep Z"
    );
    assert!(
        per_horizon_id[1].result.estimands[0].adjustment_set.is_empty(),
        "horizon 2 has no backdoor through Z"
    );

    let adjusted = Study::series(data.clone())
        .graph(graph.clone())
        .query(CausalQuery::Response(query.clone()))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    assert_study_used_ident(
        &adjusted,
        &per_horizon_id[0],
        &spec["identification"][0],
        fixture["contract"]["estimators"]["response"].as_str().unwrap(),
        "multi-horizon primary I(h=1)",
    );
    assert!(
        adjusted
            .diagnostics
            .iter()
            .any(|d| d.code.as_ref() == "identify.temporal_response.horizon_dependent"),
        "mixed I(h) must be observable"
    );
    let response = adjusted.response.as_ref().expect("temporal response payload");
    let horizon_id = response.horizon_identification.as_ref().expect("I(h) on the surface");
    assert_eq!(horizon_id.len(), 2);
    assert_eq!(horizon_id[0].horizon, 1);
    assert_eq!(horizon_id[1].horizon, 2);
    assert_eq!(
        horizon_id[0].adjustment.as_ref(),
        &[TemporalNodeKey { variable: var_id(&names, "z"), offset: -1 }]
    );
    assert!(horizon_id[1].adjustment.is_empty());
    let mean = surface_mean(&adjusted);
    assert_eq!(mean.len(), expected.len());
    // dose-major: (0,1), (0,2), (1,1), (1,2)
    assert!((mean[0] - expected[0]).abs() <= atol, "R(0,1)={} expected={}", mean[0], expected[0]);
    assert!((mean[2] - expected[2]).abs() <= atol, "R(1,1)={} expected={}", mean[2], expected[2]);
    assert!(
        (mean[1] - expected[1]).abs() <= h2_atol,
        "R(0,2)={} expected={}",
        mean[1],
        expected[1]
    );
    assert!(
        (mean[3] - expected[3]).abs() <= h2_atol,
        "R(1,2)={} expected={}",
        mean[3],
        expected[3]
    );
    assert!(
        (mean[2] - mean[0] - 2.0).abs() <= atol,
        "horizon 1 contrast must stay the structural pulse, not the confounded 4.5"
    );

    let prepared = Study::series(data.clone())
        .graph(graph)
        .query(CausalQuery::Response(query))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let click = prepared.estimate_series(&data, &ctx).unwrap();
    assert!(
        click.diagnostics.iter().any(|d| d.code.as_ref() == "exec.identify.cached"),
        "estimate click must not re-identify"
    );
    assert!(
        click
            .diagnostics
            .iter()
            .any(|d| d.code.as_ref() == "identify.temporal_response.horizon_dependent")
    );
    let click_mean = surface_mean(&click);
    for (index, (&actual, &truth)) in click_mean.iter().zip(mean.iter()).enumerate() {
        assert!((actual - truth).abs() <= atol, "prepared[{index}]={actual} truth={truth}");
    }
}

fn parse_status(value: &str) -> SupportStatus {
    match value {
        "supported" => SupportStatus::Supported,
        "weak_overlap" => SupportStatus::WeakOverlap,
        "extrapolative" => SupportStatus::Extrapolative,
        "outside_empirical_support" => SupportStatus::OutsideEmpiricalSupport,
        other => panic!("unknown support status {other}"),
    }
}

#[test]
fn horizon_support_is_per_cell_not_union() {
    let fixture = horizon_support_fixture();
    let n = usize::try_from(fixture["generation"]["n"].as_u64().unwrap()).unwrap();
    let t: Vec<f64> = (0..n)
        .map(|i| if i >= n.saturating_sub(7) { 10.0 } else { 0.05 * (i as f64).sin() })
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            1.0 + 2.0 * i.checked_sub(1).map_or(0.0, |j| t[j])
                + 3.0 * i.checked_sub(2).map_or(0.0, |j| t[j])
        })
        .collect();
    let (data, graph) = series(&["t", "y"], vec![t, y], &[(0, 1, 1, 0), (0, 2, 1, 0)]);
    let doses = f64s(&fixture["contract"]["dose_grid"]);
    let atol = fixture["tolerance"]["atol"].as_f64().unwrap();
    let query = mean_curve_query(&doses, &fixture);
    let result = Study::series(data.clone())
        .graph(graph.clone())
        .query(CausalQuery::Response(query))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(11))
        .unwrap();
    let support = &result.response.as_ref().unwrap().support;
    assert_eq!(
        support.status,
        parse_status(fixture["contract"]["support"]["status"].as_str().unwrap())
    );
    let expected: Vec<SupportStatus> = fixture["contract"]["support"]["point_status"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| parse_status(value.as_str().unwrap()))
        .collect();
    assert_eq!(support.point_status.as_ref().map(AsRef::as_ref), Some(expected.as_slice()));
    let ranges = support
        .diagnostics
        .iter()
        .find(|d| d.id.as_ref() == "response.temporal.horizon_treatment_range")
        .expect("horizon treatment range");
    let expected_ranges = f64s(&fixture["contract"]["support"]["horizon_treatment_range"]);
    assert_eq!(ranges.values.len(), expected_ranges.len());
    for (actual, truth) in ranges.values.iter().zip(&expected_ranges) {
        assert!((actual - truth).abs() <= atol, "range {actual} vs {truth}");
    }
    let union_max = ranges.values[1].max(ranges.values[3]);
    let union_min = ranges.values[0].min(ranges.values[2]);
    assert!(doses[1] >= union_min && doses[1] <= union_max);

    let warning = fixture["contract"]["support"]["warning_code"].as_str().unwrap();
    assert!(support.warnings.iter().any(|w| w.code.as_ref() == warning));

    let at = i32::try_from(fixture["contract"]["policy"]["at"].as_i64().unwrap()).unwrap();
    let horizons: Vec<u32> = fixture["contract"]["horizons"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| u32::try_from(value.as_u64().unwrap()).unwrap())
        .collect();
    let path_query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(1),
        interventions: Arc::from(vec![Intervention::set(
            VariableId::from_raw(0),
            Value::f64(doses[1]),
        )]),
    })
    .with_temporal(TemporalResponseSpec::new(horizons, TemporalPolicy::pulse(at), None).unwrap());
    let path = Study::series(data)
        .graph(graph)
        .query(CausalQuery::Response(path_query))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(11))
        .unwrap();
    let path_support = &path.response.as_ref().unwrap().support;
    assert_eq!(path_support.status, SupportStatus::Extrapolative);
    assert_eq!(
        path_support.point_status.as_ref().map(AsRef::as_ref),
        Some([SupportStatus::Supported, SupportStatus::OutsideEmpiricalSupport].as_slice())
    );
}
