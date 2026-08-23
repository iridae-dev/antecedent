//! End-to-end temporal-response facade conformance.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(clippy::cast_precision_loss, clippy::many_single_char_names)]

use std::sync::Arc;

use antecedent::{RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, CausalSchemaBuilder, ContinuousDomain, ExecutionContext, GridSpec, Intervention,
    InterventionSequence, Lag, MeasurementSpec, MechanismOverride, ResponseFunctional,
    ResponseIdentification, ResponseQuery, ResponseUncertainty, ResponseValue, RoleHint,
    SequencedIntervention, SmallRoleSet, SupportStatus, TargetPopulation, TemporalEffectQuery,
    TemporalPolicy, TemporalResponseSpec, Value, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
    TimeSeriesData, ValidityBitmap,
};
use antecedent_graph::{TemporalDag, ensure_lagged};

fn fixture() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../conformance/response/temporal_dose_horizon/expected.json"
    ))
    .unwrap()
}

fn temporal_fixture_series() -> (TimeSeriesData, TemporalDag) {
    let fixture = fixture();
    let n = usize::try_from(fixture["generation"]["n"].as_u64().unwrap()).unwrap();
    let t: Vec<f64> = (0..n)
        .map(|i| match i % 4 {
            0 | 2 => 0.0,
            1 => 1.0,
            3 => -1.0,
            _ => unreachable!(),
        })
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            1.0 + 2.0 * i.checked_sub(1).map_or(0.0, |j| t[j])
                + 3.0 * i.checked_sub(2).map_or(0.0, |j| t[j])
        })
        .collect();

    let mut builder = CausalSchemaBuilder::new();
    builder
        .add_variable(
            "t",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    builder
        .add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    let schema = builder.build().unwrap();
    let columns = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(t), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(y), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, columns, None, None).unwrap();
    let series = TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap();

    let mut graph = TemporalDag::empty();
    let t1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let t2 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(2)).unwrap();
    let y0 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(t1, y0).unwrap();
    graph.insert_directed(t2, y0).unwrap();
    (series, graph)
}

fn temporal_spec(fixture: &serde_json::Value) -> TemporalResponseSpec {
    let horizons: Vec<u32> = fixture["contract"]["horizons"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| u32::try_from(value.as_u64().unwrap()).unwrap())
        .collect();
    let at = i32::try_from(fixture["contract"]["policy"]["at"].as_i64().unwrap()).unwrap();
    TemporalResponseSpec::new(horizons, TemporalPolicy::pulse(at), None).unwrap()
}

fn assert_surface(
    result: &antecedent::result::StudyResult,
    expected: &[f64],
    atol: f64,
    expected_provenance: &str,
) {
    let response = result.response.as_ref().expect("temporal response payload");
    assert_eq!(response.provenance_id.as_ref(), expected_provenance);
    let ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) =
        &response.estimate
    else {
        panic!("expected point-identified temporal response surface");
    };
    assert_eq!(mean.len(), expected.len());
    for (index, (&actual, &truth)) in mean.iter().zip(expected).enumerate() {
        assert!(
            (actual - truth).abs() <= atol,
            "surface[{index}]={actual}, truth={truth}, atol={atol}"
        );
    }
}

#[test]
fn temporal_dose_horizon_surface_matches_fixture_and_prepared_path() {
    let fixture = fixture();
    let (series, graph) = temporal_fixture_series();
    let doses: Vec<f64> = fixture["contract"]["dose_grid"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap())
        .collect();
    let expected: Vec<f64> = fixture["contract"]["surface"]["mean"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap())
        .collect();
    let atol = fixture["tolerance"]["atol"].as_f64().unwrap();
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from(doses)),
        ),
    })
    .with_temporal(temporal_spec(&fixture));
    let study = Study::series(series.clone())
        .graph(graph)
        .query(CausalQuery::Response(query))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(21);

    let direct = study.run(&ctx).unwrap();
    let prepared = study.prepare(&ctx).unwrap();
    let click = prepared.estimate_series(&series, &ctx).unwrap();

    assert_surface(&direct, &expected, atol, "estimate.temporal_response.gcomp");
    assert_surface(&click, &expected, atol, "estimate.temporal_response.gcomp");
    let expected_grid: Vec<f64> = fixture["contract"]["surface"]["grid_pairs"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|pair| pair.as_array().unwrap().iter().map(|value| value.as_f64().unwrap()))
        .collect();
    let ResponseIdentification::PointIdentified(ResponseValue::Surface { grid, dimension, .. }) =
        &direct.response.as_ref().unwrap().estimate
    else {
        unreachable!()
    };
    assert_eq!(*dimension, 2);
    assert_eq!(grid.as_ref(), expected_grid.as_slice());
    let support = &direct.response.as_ref().unwrap().support;
    assert_eq!(support.status, SupportStatus::Supported);
    let expected_cells: Vec<SupportStatus> = fixture["contract"]["support"]["point_status"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| match value.as_str().unwrap() {
            "supported" => SupportStatus::Supported,
            "weak_overlap" => SupportStatus::WeakOverlap,
            "extrapolative" => SupportStatus::Extrapolative,
            "outside_empirical_support" => SupportStatus::OutsideEmpiricalSupport,
            other => panic!("unknown fixture support cell {other}"),
        })
        .collect();
    assert_eq!(support.point_status.as_ref().map(AsRef::as_ref), Some(expected_cells.as_slice()));
    assert!(
        direct.diagnostics.iter().all(|d| d.code.as_ref() != "exec.identify.cached"),
        "fresh execution must identify"
    );
    assert!(
        click.diagnostics.iter().any(|d| d.code.as_ref() == "exec.identify.cached"),
        "prepared estimate_series must reuse identification"
    );

    let surface = direct.response.as_ref().unwrap();
    let ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) =
        &surface.estimate
    else {
        unreachable!()
    };
    let projection = mean[2] - mean[0];
    let expected_projection =
        fixture["contract"]["pulse_effect_projection"]["contrast"].as_f64().unwrap();
    assert!((projection - expected_projection).abs() <= atol);
}

#[test]
fn pulse_and_single_step_sustained_match_surface_projection() {
    let fixture = fixture();
    let (series, graph) = temporal_fixture_series();
    let atol = fixture["tolerance"]["atol"].as_f64().unwrap();
    let expected = fixture["contract"]["pulse_effect_projection"]["contrast"].as_f64().unwrap();
    let sustained_expected =
        fixture["contract"]["sustained_effect_projection"]["contrast"].as_f64().unwrap();
    assert!((expected - sustained_expected).abs() <= atol);

    let pulse = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1);
    let sustained =
        TemporalEffectQuery::sustained(VariableId::from_raw(0), VariableId::from_raw(1), 0, 1.0)
            .with_policy(TemporalPolicy::sustained(-1, -1))
            .with_horizon_steps(1);

    let ctx = ExecutionContext::for_tests(23);
    let pulse_study = Study::series(series.clone())
        .graph(graph.clone())
        .temporal_query(pulse)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let pulse_direct = pulse_study.run(&ctx).unwrap();
    let pulse_prepared = pulse_study.prepare(&ctx).unwrap();
    let pulse_click = pulse_prepared.estimate_series(&series, &ctx).unwrap();

    let sustained_study = Study::series(series.clone())
        .graph(graph)
        .temporal_query(sustained)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let sustained_direct = sustained_study.run(&ctx).unwrap();
    let sustained_prepared = sustained_study.prepare(&ctx).unwrap();
    let sustained_click = sustained_prepared.estimate_series(&series, &ctx).unwrap();

    for (label, result) in [
        ("pulse direct", &pulse_direct),
        ("pulse prepared", &pulse_click),
        ("sustained direct", &sustained_direct),
        ("sustained prepared", &sustained_click),
    ] {
        assert!(
            (result.estimate.ate - expected).abs() <= atol,
            "{label}: ate={}, expected={expected}",
            result.estimate.ate
        );
    }
    assert!(
        pulse_direct.diagnostics.iter().all(|d| d.code.as_ref() != "exec.identify.cached"),
        "fresh Pulse must identify"
    );
    assert!(
        pulse_click.diagnostics.iter().any(|d| d.code.as_ref() == "exec.identify.cached"),
        "prepared Pulse must reuse identification"
    );
    assert!(
        sustained_click.diagnostics.iter().any(|d| d.code.as_ref() == "exec.identify.cached"),
        "prepared Sustained must reuse identification"
    );
}

fn noisy_pulse_series(n: usize) -> (TimeSeriesData, TemporalDag) {
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut rng = 0x00C0_FFEE_u64;
    let mut gauss = || {
        rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let u = ((rng >> 11) as f64) / ((1u64 << 53) as f64);
        rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let v = ((rng >> 11) as f64) / ((1u64 << 53) as f64);
        (-2.0 * u.max(1e-12).ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    };
    for i in 1..n {
        t[i] = 0.4 * gauss();
        y[i] = 0.8 * t[i - 1] + 0.35 * gauss();
    }
    let mut builder = CausalSchemaBuilder::new();
    builder
        .add_variable(
            "t",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    builder
        .add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    let schema = builder.build().unwrap();
    let columns = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(t), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(y), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, columns, None, None).unwrap();
    let series = TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap();
    let mut graph = TemporalDag::empty();
    let t1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(t1, y0).unwrap();
    (series, graph)
}

fn pointwise_halfwidths(result: &antecedent::result::StudyResult) -> Vec<f64> {
    let ResponseUncertainty::PointwiseBand { lower, upper, .. } =
        &result.response.as_ref().expect("response").uncertainty
    else {
        panic!("expected pointwise band on temporal surface");
    };
    lower.iter().zip(upper.iter()).map(|(lo, hi)| (hi - lo) * 0.5).collect()
}

#[test]
fn pulse_sustained_and_surface_share_study_bootstrap_ses() {
    let (series, graph) = noisy_pulse_series(200);
    let ctx = ExecutionContext::for_tests(29);
    let pulse = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1);
    let sustained =
        TemporalEffectQuery::sustained(VariableId::from_raw(0), VariableId::from_raw(1), 0, 1.0)
            .with_policy(TemporalPolicy::sustained(-1, -1))
            .with_horizon_steps(1);
    let surface_query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from([0.0, 1.0])),
        ),
    })
    .with_temporal(TemporalResponseSpec::new(vec![1], TemporalPolicy::pulse(-1), None).unwrap());

    let run_pulse = |reps: u32| {
        Study::series(series.clone())
            .graph(graph.clone())
            .temporal_query(pulse.clone())
            .refute(RefuteSuite::None)
            .bootstrap_replicates(reps)
            .build()
            .unwrap()
            .run(&ctx)
            .unwrap()
    };
    let run_sustained = |reps: u32| {
        Study::series(series.clone())
            .graph(graph.clone())
            .temporal_query(sustained.clone())
            .refute(RefuteSuite::None)
            .bootstrap_replicates(reps)
            .build()
            .unwrap()
            .run(&ctx)
            .unwrap()
    };
    let run_surface = |reps: u32| {
        Study::series(series.clone())
            .graph(graph.clone())
            .query(CausalQuery::Response(surface_query.clone()))
            .refute(RefuteSuite::None)
            .bootstrap_replicates(reps)
            .build()
            .unwrap()
            .run(&ctx)
            .unwrap()
    };

    let pulse_analytic = run_pulse(0);
    let pulse_boot = run_pulse(40);
    let se_pulse_a = pulse_analytic.estimate.se_analytic;
    let se_pulse_b = pulse_boot.estimate.se_bootstrap.expect("pulse bootstrap SE");
    assert!(se_pulse_a.is_finite() && se_pulse_a > 0.0, "analytic pulse se={se_pulse_a}");
    assert!(se_pulse_b.is_finite() && se_pulse_b > 0.0, "bootstrap pulse se={se_pulse_b}");
    assert!(
        (se_pulse_a - se_pulse_b).abs() > 1e-8,
        "Pulse must use Study bootstrap when replicates > 0 (analytic={se_pulse_a}, boot={se_pulse_b})"
    );

    let se_sustained_b = run_sustained(40).estimate.se_bootstrap.expect("sustained bootstrap SE");
    assert!(se_sustained_b.is_finite() && se_sustained_b > 0.0);
    let pulse_sustained_ratio = se_pulse_b / se_sustained_b;
    assert!(
        (0.25..=4.0).contains(&pulse_sustained_ratio),
        "Pulse and single-step Sustained SEs must be comparable (pulse={se_pulse_b}, sustained={se_sustained_b})"
    );

    let hw_a = pointwise_halfwidths(&run_surface(0));
    let hw_b = pointwise_halfwidths(&run_surface(40));
    assert!(hw_a.iter().all(|w| w.is_finite() && *w > 0.0), "analytic surface bands={hw_a:?}");
    assert!(hw_b.iter().all(|w| w.is_finite() && *w > 0.0), "bootstrap surface bands={hw_b:?}");
    assert!(
        hw_a.iter().zip(&hw_b).any(|(a, b)| (a - b).abs() > 1e-8),
        "surface SEs must follow Study bootstrap (analytic={hw_a:?}, boot={hw_b:?})"
    );
    let z = 1.959_963_984_540_054;
    let surface_se = hw_b.iter().copied().fold(0.0_f64, f64::max) / z;
    let ratio = se_pulse_b / surface_se;
    assert!(
        (0.25..=4.0).contains(&ratio),
        "Pulse SE and dose×horizon surface SE must be comparable (pulse={se_pulse_b}, surface={surface_se})"
    );
}

#[test]
fn temporal_intervention_path_matches_fixture() {
    let fixture = fixture();
    let (series, graph) = temporal_fixture_series();
    let expected: Vec<f64> = fixture["contract"]["intervention_paths"]["set_1"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap())
        .collect();
    let atol = fixture["tolerance"]["atol"].as_f64().unwrap();
    let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(1),
        interventions: Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(1.0))]),
    })
    .with_temporal(temporal_spec(&fixture));
    let result = Study::series(series)
        .graph(graph)
        .query(CausalQuery::Response(query))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(22))
        .unwrap();

    assert_surface(&result, &expected, atol, "estimate.temporal_response.intervention_gcomp");
}

// ---- GAP3: fixture keys that exist but were never read ----

#[test]
fn temporal_intervention_path_soft_constant_matches_fixture() {
    let fixture = fixture();
    let (series, graph) = temporal_fixture_series();
    let expected: Vec<f64> = fixture["contract"]["intervention_paths"]["soft_constant_1"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap())
        .collect();
    let atol = fixture["tolerance"]["atol"].as_f64().unwrap();
    let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(1),
        interventions: Arc::from([Intervention::soft(
            VariableId::from_raw(0),
            MechanismOverride::constant(1.0),
        )]),
    })
    .with_temporal(temporal_spec(&fixture));
    let result = Study::series(series)
        .graph(graph)
        .query(CausalQuery::Response(query))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(22))
        .unwrap();

    assert_surface(&result, &expected, atol, "estimate.temporal_response.intervention_gcomp");
}

#[test]
fn temporal_intervention_path_soft_additive_shift_matches_fixture() {
    let fixture = fixture();
    let (series, graph) = temporal_fixture_series();
    let expected: Vec<f64> = fixture["contract"]["intervention_paths"]["soft_additive_shift_1"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap())
        .collect();
    let atol = fixture["tolerance"]["atol"].as_f64().unwrap();
    let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(1),
        interventions: Arc::from([Intervention::soft(
            VariableId::from_raw(0),
            MechanismOverride::additive_shift(1.0),
        )]),
    })
    .with_temporal(temporal_spec(&fixture));
    let result = Study::series(series)
        .graph(graph)
        .query(CausalQuery::Response(query))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(22))
        .unwrap();

    assert_surface(&result, &expected, atol, "estimate.temporal_response.intervention_gcomp");
}

#[test]
fn temporal_single_step_sequence_matches_bare_intervention() {
    let fixture = fixture();
    let (series, graph) = temporal_fixture_series();
    let expected: Vec<f64> = fixture["contract"]["intervention_paths"]["set_1"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap())
        .collect();
    let atol = fixture["tolerance"]["atol"].as_f64().unwrap();
    let seq = InterventionSequence::new(vec![SequencedIntervention {
        intervention: Intervention::set(VariableId::from_raw(0), Value::f64(1.0)),
        temporal: TemporalPolicy::pulse(0),
    }]);
    let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(1),
        interventions: Arc::from([Intervention::Sequence(seq)]),
    })
    .with_temporal(temporal_spec(&fixture));
    let result = Study::series(series)
        .graph(graph)
        .query(CausalQuery::Response(query))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(22))
        .unwrap();

    assert_surface(&result, &expected, atol, "estimate.temporal_response.intervention_gcomp");
}

// ---- GAP2: refusal paths were unasserted (end-to-end through the facade) ----

/// (a) Multi-step `Sequence` (>1 step, same variable) must refuse. Before the fix this
/// silently collapsed to the last step, so `Sequence([Set(t=0), Set(t=5)])` returned the
/// same answer as `Set(t=5)` instead of erroring. If that collapse ever returns, `.unwrap_err()`
/// below panics instead of silently passing.
#[test]
fn multi_step_sequence_refuses_end_to_end() {
    let fixture = fixture();
    let (series, graph) = temporal_fixture_series();
    let v = VariableId::from_raw(0);
    let seq = InterventionSequence::new(vec![
        SequencedIntervention {
            intervention: Intervention::set(v, Value::f64(0.0)),
            temporal: TemporalPolicy::pulse(0),
        },
        SequencedIntervention {
            intervention: Intervention::set(v, Value::f64(5.0)),
            temporal: TemporalPolicy::pulse(0),
        },
    ]);
    let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(1),
        interventions: Arc::from([Intervention::Sequence(seq)]),
    })
    .with_temporal(temporal_spec(&fixture));
    let err = Study::series(series)
        .graph(graph)
        .query(CausalQuery::Response(query))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(22))
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("multi-step Sequence") && msg.contains("not licensed"),
        "unexpected error content: {msg}"
    );
}

/// (b) A `Sequence` nested inside a `Sequence` must refuse via the depth guard rather
/// than silently recursing into a leaf.
#[test]
fn nested_sequence_refuses_end_to_end() {
    let fixture = fixture();
    let (series, graph) = temporal_fixture_series();
    let v = VariableId::from_raw(0);
    let inner = InterventionSequence::new(vec![SequencedIntervention {
        intervention: Intervention::set(v, Value::f64(1.0)),
        temporal: TemporalPolicy::pulse(0),
    }]);
    let outer = InterventionSequence::new(vec![SequencedIntervention {
        intervention: Intervention::Sequence(inner),
        temporal: TemporalPolicy::pulse(0),
    }]);
    let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(1),
        interventions: Arc::from([Intervention::Sequence(outer)]),
    })
    .with_temporal(temporal_spec(&fixture));
    let err = Study::series(series)
        .graph(graph)
        .query(CausalQuery::Response(query))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(22))
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("Sequence") && msg.contains("nested") && msg.contains("not licensed"),
        "unexpected error content: {msg}"
    );
}

/// (c) Multi-step `Sustained{from,until}` with `until > from` is enforced by
/// `refuse_multi_step_schedule` in `temporal_adjustment.rs`, which was previously
/// referenced by no test anywhere in the repo. Assert it refuses at the estimation layer.
#[test]
fn multi_step_sustained_refuses_end_to_end() {
    let (series, graph) = temporal_fixture_series();
    let temporal =
        TemporalResponseSpec::new(vec![1u32], TemporalPolicy::sustained(-1, 0), None).unwrap();
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from(vec![0.0_f64, 1.0])),
        ),
    })
    .with_temporal(temporal);
    let err = Study::series(series)
        .graph(graph)
        .query(CausalQuery::Response(query))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(22))
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("Sustained") && msg.contains("multiple steps"),
        "unexpected error content: {msg}"
    );
}

/// (e) A `TargetPopulation` other than `AllObserved` must refuse on the temporal path.
#[test]
fn target_population_other_than_all_observed_refuses_end_to_end() {
    let fixture = fixture();
    let (series, graph) = temporal_fixture_series();
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from(vec![0.0_f64, 1.0])),
        ),
    })
    .with_temporal(temporal_spec(&fixture))
    .with_target_population(TargetPopulation::Treated);
    let err = Study::series(series)
        .graph(graph)
        .query(CausalQuery::Response(query))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(22))
        .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("AllObserved"), "unexpected error content: {msg}");
}

// (f) A `Sequence` spanning multiple target variables must refuse. Note: at the
// facade level this is actually caught even earlier than temporal resolution — a
// `Sequence` whose steps target different variables has no unique `primary_variable`,
// so `Study` refuses with "response query has no treatment/outcome pair" before ever
// reaching the temporal estimator. The estimator-level "multiple target variables"
// refusal in `resolve_sequence` is covered directly (with the private helper it lives
// on) by `sequence_multiple_target_variables_fails_closed` in
// `crates/antecedent-estimate/src/temporal_response.rs`.

#[test]
fn temporal_dose_horizon_bands_match_fixture() {
    use antecedent_core::ResponseUncertainty;

    let fixture = fixture();
    let (series, graph) = temporal_fixture_series();
    let doses: Vec<f64> = fixture["contract"]["dose_grid"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap())
        .collect();
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from(doses)),
        ),
    })
    .with_temporal(temporal_spec(&fixture));
    let result = Study::series(series)
        .graph(graph)
        .query(CausalQuery::Response(query))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(21))
        .unwrap();
    let response = result.response.as_ref().expect("response payload");
    let ResponseUncertainty::PointwiseBand { lower, upper, .. } = &response.uncertainty else {
        panic!("expected pointwise bands");
    };
    let expected_lower: Vec<f64> = fixture["contract"]["surface"]["lower"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap())
        .collect();
    let expected_upper: Vec<f64> = fixture["contract"]["surface"]["upper"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap())
        .collect();
    let atol = fixture["tolerance"]["atol"].as_f64().unwrap();
    for (i, (&lo, &hi)) in lower.iter().zip(upper.iter()).enumerate() {
        assert!(
            (lo - expected_lower[i]).abs() <= atol,
            "lower[{i}]={lo}, expected={}",
            expected_lower[i]
        );
        assert!(
            (hi - expected_upper[i]).abs() <= atol,
            "upper[{i}]={hi}, expected={}",
            expected_upper[i]
        );
    }
    // dose=0 horizon=1: index 0 — band width must be strictly positive (regression guard).
    assert!(upper[0] - lower[0] > 0.0, "dose=0 band width must be positive");
}
