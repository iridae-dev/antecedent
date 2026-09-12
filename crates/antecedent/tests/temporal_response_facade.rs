//! End-to-end temporal-response facade conformance.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(clippy::cast_precision_loss, clippy::many_single_char_names)]

use std::sync::Arc;

use antecedent::{RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, CausalSchemaBuilder, ContinuousDomain, ExecutionContext, GridSpec, Intervention,
    InterventionSequence, Lag, MeasurementSpec, MechanismOverride, MemoryBudget,
    ResponseFunctional, ResponseIdentification, ResponseQuery, ResponseUncertainty, ResponseValue,
    RoleHint, SequencedIntervention, SmallRoleSet, SupportStatus, TargetPopulation,
    TemporalEffectQuery, TemporalPolicy, TemporalResponseSpec, Value, ValueType, VariableId,
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
fn temporal_surface_honors_execution_memory_budget() {
    let (series, graph) = temporal_fixture_series();
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from([0.0, 1.0])),
        ),
    })
    .with_temporal(TemporalResponseSpec::new(vec![1, 2], TemporalPolicy::pulse(-1), None).unwrap());
    let study = Study::series(series)
        .graph(graph)
        .query(CausalQuery::Response(query))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let mut ctx = ExecutionContext::for_tests(21);
    ctx.memory = MemoryBudget { soft_limit_bytes: None, hard_limit_bytes: Some(159) };

    let error = study.run(&ctx).unwrap_err();
    assert!(error.to_string().contains("at least 160 output bytes"), "got {error}");
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

// ---- GAP2: multi-step Sequence is a sequential overlay, never last-step collapse ----

fn sequence_steps(variable: VariableId, values: &[f64]) -> Intervention {
    Intervention::Sequence(InterventionSequence::new(
        values
            .iter()
            .map(|&value| SequencedIntervention {
                intervention: Intervention::set(variable, Value::f64(value)),
                temporal: TemporalPolicy::pulse(0),
            })
            .collect::<Vec<_>>(),
    ))
}

/// Two-step `Set(1)` at consecutive times ending at the pulse origin is
/// `Y_s = 1 + 2 T_{s-1} + 3 T_{s-2}` with both lagged treatments set: `[6, 4]`.
/// Last-step-only `Set(1)` is `[3, 4]`. Collapse to the last step fails this pin.
#[test]
fn multi_step_sequence_matches_two_step_truth_and_does_not_collapse() {
    let fixture = fixture();
    let (series, graph) = temporal_fixture_series();
    let expected: Vec<f64> = fixture["contract"]["intervention_paths"]["sequence_two_step_set_1"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap())
        .collect();
    let last_step: Vec<f64> = fixture["contract"]["intervention_paths"]["set_1"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap())
        .collect();
    let atol = fixture["tolerance"]["atol"].as_f64().unwrap();
    assert_ne!(expected.as_slice(), last_step.as_slice());

    let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(1),
        interventions: Arc::from([sequence_steps(VariableId::from_raw(0), &[1.0, 1.0])]),
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
    assert!(
        result.diagnostics.iter().any(|d| d.code.as_ref() == "estimate.temporal.sequence_overlay"),
        "sequential overlay diagnostic missing: {:?}",
        result.diagnostics.iter().map(|d| d.code.as_ref()).collect::<Vec<_>>()
    );
    assert_eq!(result.identification.estimands[0].method.as_ref(), "temporal.backdoor.unfolded");
    let response = result.response.as_ref().unwrap();
    assert_eq!(response.support.status, SupportStatus::Extrapolative);
    assert!(matches!(response.uncertainty, ResponseUncertainty::None));
    assert!(response.support.warnings.iter().any(|warning| {
        warning.code.as_ref() == "response.temporal.sequence_joint_support_unassessed"
    }));
    assert!(response.support.warnings.iter().any(|warning| {
        warning.code.as_ref() == "response.temporal.sequence_uncertainty_unavailable"
    }));
}

#[test]
fn single_step_sequence_uses_its_explicit_policy_not_the_outer_policy() {
    let (series, graph) = temporal_fixture_series();
    let sequence = Intervention::Sequence(InterventionSequence::new([SequencedIntervention {
        intervention: Intervention::set(VariableId::from_raw(0), Value::f64(1.0)),
        temporal: TemporalPolicy::pulse(-2),
    }]));
    let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(1),
        interventions: Arc::from([sequence]),
    })
    .with_temporal(TemporalResponseSpec::new(vec![1u32], TemporalPolicy::pulse(-1), None).unwrap());
    let result = Study::series(series)
        .graph(graph)
        .query(CausalQuery::Response(query))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(22))
        .unwrap();

    // Y_s = 1 + 2 T_{s-1} + 3 T_{s-2}; E[T_{s-1}] = 0 and do(T_{s-2}=1).
    assert_surface(&result, &[4.0], 1e-10, "estimate.temporal_response.intervention_gcomp");
    let response = result.response.as_ref().unwrap();
    let overlay = response
        .support
        .diagnostics
        .iter()
        .find(|diagnostic| diagnostic.id.as_ref() == "response.temporal.sequence_overlay")
        .expect("resolved overlay provenance");
    assert!((overlay.values[1] + 2.0).abs() < f64::EPSILON);
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

fn joint_ab_series() -> (TimeSeriesData, TemporalDag) {
    let n = 242;
    let a: Vec<f64> = (0..n)
        .map(|i| match i % 4 {
            0 | 2 => 0.0,
            1 => 1.0,
            3 => -1.0,
            _ => unreachable!(),
        })
        .collect();
    let b: Vec<f64> = (0..n)
        .map(|i| match i % 4 {
            0 | 1 => 0.0,
            2 => 1.0,
            3 => -1.0,
            _ => unreachable!(),
        })
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|i: usize| {
            1.0 + 2.0 * i.checked_sub(1).map_or(0.0, |j| a[j])
                + 4.0 * i.checked_sub(1).map_or(0.0, |j| b[j])
        })
        .collect();
    let mut builder = CausalSchemaBuilder::new();
    for (name, hint) in [
        ("a", RoleHint::TreatmentCandidate),
        ("b", RoleHint::TreatmentCandidate),
        ("y", RoleHint::OutcomeCandidate),
    ] {
        builder
            .add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(hint),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
    }
    let schema = builder.build().unwrap();
    let columns = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(a), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(b), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(2), Arc::from(y), ValidityBitmap::all_valid(n))
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
    let a1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let b1 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut graph, VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(a1, y0).unwrap();
    graph.insert_directed(b1, y0).unwrap();
    (series, graph)
}

/// Joint single-time Sequence: `Y_s = 1 + 2 A_{s-1} + 4 B_{s-1}`, both set to 1 → 7.
#[test]
fn joint_sequence_matches_structural_level() {
    let (series, graph) = joint_ab_series();
    let seq = InterventionSequence::new(vec![
        SequencedIntervention {
            intervention: Intervention::set(VariableId::from_raw(0), Value::f64(1.0)),
            temporal: TemporalPolicy::pulse(0),
        },
        SequencedIntervention {
            intervention: Intervention::set(VariableId::from_raw(1), Value::f64(1.0)),
            temporal: TemporalPolicy::pulse(0),
        },
    ]);
    let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(2),
        interventions: Arc::from([Intervention::Sequence(seq)]),
    })
    .with_temporal(TemporalResponseSpec::new(vec![1u32], TemporalPolicy::pulse(-1), None).unwrap());
    let result = Study::series(series)
        .graph(graph)
        .query(CausalQuery::Response(query))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(22))
        .unwrap();
    assert_surface(&result, &[7.0], 1e-10, "estimate.temporal_response.intervention_gcomp");
    assert_eq!(result.identification.estimands[0].method.as_ref(), "temporal.backdoor.unfolded");
}

#[test]
fn additive_shift_preserves_the_counterfactual_parent_mechanism() {
    let n = 200usize;
    let x: Vec<f64> = (0..n).map(|i| (i % 17) as f64 / 16.0).collect();
    let m: Vec<f64> = x.iter().map(|&value| 1.0 + 2.0 * value).collect();
    let y: Vec<f64> =
        (0..n).map(|i| i.checked_sub(1).map_or(0.0, |previous| 3.0 * m[previous])).collect();
    let series = TimeSeriesData::from_f64_columns(
        [("x", x.as_slice()), ("m", m.as_slice()), ("y", y.as_slice())],
        1,
    )
    .unwrap();
    let mut graph = TemporalDag::empty();
    let x0 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
    let m0 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let m1 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut graph, VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(x0, m0).unwrap();
    graph.insert_directed(m1, y0).unwrap();

    let seq = InterventionSequence::new(vec![
        SequencedIntervention {
            intervention: Intervention::set(VariableId::from_raw(0), Value::f64(2.0)),
            temporal: TemporalPolicy::pulse(0),
        },
        SequencedIntervention {
            intervention: Intervention::soft(
                VariableId::from_raw(1),
                MechanismOverride::additive_shift(1.0),
            ),
            temporal: TemporalPolicy::pulse(0),
        },
    ]);
    let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(2),
        interventions: Arc::from([Intervention::Sequence(seq)]),
    })
    .with_temporal(TemporalResponseSpec::new(vec![1u32], TemporalPolicy::pulse(-1), None).unwrap());
    let result = Study::series(series)
        .graph(graph)
        .query(CausalQuery::Response(query))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(23))
        .unwrap();

    // M := (1 + 2*X) + 1, so do(X=2) gives M=6 and Y=3*M=18.
    // Replacing M by its factual marginal mean plus one gives the wrong answer.
    assert_surface(&result, &[18.0], 1e-9, "estimate.temporal_response.intervention_gcomp");
}

#[test]
fn joint_sequence_refuses_when_a_coordinate_is_not_on_the_graph() {
    let (series, graph) = joint_ab_series();
    let seq = InterventionSequence::new(vec![
        SequencedIntervention {
            intervention: Intervention::set(VariableId::from_raw(0), Value::f64(1.0)),
            temporal: TemporalPolicy::pulse(0),
        },
        SequencedIntervention {
            intervention: Intervention::set(VariableId::from_raw(3), Value::f64(1.0)),
            temporal: TemporalPolicy::pulse(0),
        },
    ]);
    let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(2),
        interventions: Arc::from([Intervention::Sequence(seq)]),
    })
    .with_temporal(TemporalResponseSpec::new(vec![1u32], TemporalPolicy::pulse(-1), None).unwrap());
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
        msg.contains("not in temporal graph")
            || msg.contains("UnknownVariable")
            || msg.contains("unknown"),
        "unexpected error content: {msg}"
    );
}

#[test]
fn joint_sequence_refuses_when_a_coordinate_is_the_outcome() {
    let (series, graph) = joint_ab_series();
    let seq = InterventionSequence::new(vec![
        SequencedIntervention {
            intervention: Intervention::set(VariableId::from_raw(0), Value::f64(1.0)),
            temporal: TemporalPolicy::pulse(0),
        },
        SequencedIntervention {
            intervention: Intervention::set(VariableId::from_raw(2), Value::f64(1.0)),
            temporal: TemporalPolicy::pulse(0),
        },
    ]);
    let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(2),
        interventions: Arc::from([Intervention::Sequence(seq)]),
    })
    .with_temporal(TemporalResponseSpec::new(vec![1u32], TemporalPolicy::pulse(-1), None).unwrap());
    let err = Study::series(series)
        .graph(graph)
        .query(CausalQuery::Response(query))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(22))
        .unwrap_err();
    assert!(err.to_string().contains("same variable"), "unexpected error content: {err}");
}

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

#[test]
#[allow(clippy::too_many_lines)]
fn bayesian_temporal_response_and_sustained_window_known_truth() {
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/bayesian/response_surfaces/expected.json"
    ))
    .unwrap();
    let (series, graph) = temporal_fixture_series();
    let ctx = ExecutionContext::for_tests(31);
    for accepted in [false, true] {
        for intervention in [false, true] {
            let functional = if intervention {
                ResponseFunctional::InterventionResponse {
                    outcome: VariableId::from_raw(1),
                    interventions: Arc::from([Intervention::set(
                        VariableId::from_raw(0),
                        Value::f64(1.0),
                    )]),
                }
            } else {
                ResponseFunctional::MeanCurve {
                    outcome: VariableId::from_raw(1),
                    treatment: ContinuousDomain::new(
                        VariableId::from_raw(0),
                        GridSpec::Values(Arc::from([0.0, 1.0])),
                    ),
                }
            };
            let query = ResponseQuery::new(functional).with_temporal(temporal_spec(&fixture()));
            let builder = Study::series(series.clone());
            let builder = if accepted {
                builder.graph(antecedent::AcceptedGraph::temporal_dag(graph.clone()))
            } else {
                builder.graph(graph.clone())
            };
            let result = builder
                .query(query)
                .inference(antecedent::InferenceMode::Bayesian(
                    antecedent::BayesianConfig::conjugate().n_draws(4096),
                ))
                .refute(RefuteSuite::None)
                .build()
                .unwrap()
                .prepare(&ctx)
                .unwrap()
                .estimate_series(&series, &ctx)
                .unwrap();
            let expected: Vec<_> = pin["temporal_mean"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            assert_surface(
                &result,
                if intervention { &expected[2..] } else { &expected },
                pin["temporal_tolerance"].as_f64().unwrap(),
                "estimate.response.temporal.bayesian",
            );
        }
        for bayesian in [false, true] {
            let query =
                TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
                    .with_policy(TemporalPolicy::sustained(-2, -1));
            let builder = Study::series(series.clone());
            let builder = if accepted {
                builder.graph(antecedent::AcceptedGraph::temporal_dag(graph.clone()))
            } else {
                builder.graph(graph.clone())
            };
            let builder = if bayesian {
                builder.inference(antecedent::InferenceMode::Bayesian(
                    antecedent::BayesianConfig::conjugate().n_draws(2048),
                ))
            } else {
                builder
            };
            let result = builder
                .query(query)
                .refute(RefuteSuite::None)
                .bootstrap_replicates(20)
                .build()
                .unwrap()
                .prepare(&ctx)
                .unwrap()
                .estimate_series(&series, &ctx)
                .unwrap();
            assert!(
                (result.estimate.ate - pin["window_effect"].as_f64().unwrap()).abs()
                    < pin["window_tolerance"].as_f64().unwrap()
            );
            assert_eq!(result.posterior.is_some(), bayesian);
        }
        let seq_query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
            outcome: VariableId::from_raw(1),
            interventions: Arc::from([sequence_steps(VariableId::from_raw(0), &[1.0, 1.0])]),
        })
        .with_temporal(temporal_spec(&fixture()));
        let builder = Study::series(series.clone());
        let builder = if accepted {
            builder.graph(antecedent::AcceptedGraph::temporal_dag(graph.clone()))
        } else {
            builder.graph(graph.clone())
        };
        let seq_result = builder
            .query(seq_query)
            .inference(antecedent::InferenceMode::Bayesian(
                antecedent::BayesianConfig::conjugate().n_draws(4096),
            ))
            .refute(RefuteSuite::None)
            .build()
            .unwrap()
            .prepare(&ctx)
            .unwrap()
            .estimate_series(&series, &ctx)
            .unwrap();
        let two_step: Vec<f64> =
            fixture()["contract"]["intervention_paths"]["sequence_two_step_set_1"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap())
                .collect();
        assert_surface(
            &seq_result,
            &two_step,
            pin["temporal_tolerance"].as_f64().unwrap(),
            "estimate.temporal_response.intervention_gcomp",
        );
        assert!(
            seq_result.posterior.is_none(),
            "a scalar last-horizon posterior must not masquerade as the multi-horizon surface"
        );
        assert!(
            seq_result.diagnostics.iter().any(|d| {
                d.code.as_ref() == "estimate.temporal.sequence_posterior_not_attached"
            })
        );
        assert!(
            seq_result
                .diagnostics
                .iter()
                .any(|d| d.code.as_ref() == "estimate.temporal.sequence_overlay")
        );
        assert_ne!(seq_result.estimand.method.as_ref(), "bayesian.gcomp");
    }
}

#[test]
fn bayesian_sequence_band_matches_posterior_quantiles() {
    let (series, graph) = temporal_fixture_series();
    let mut temporal = temporal_spec(&fixture());
    temporal.horizons = Arc::from([2]);
    let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(1),
        interventions: Arc::from([sequence_steps(VariableId::from_raw(0), &[1.0, 1.0])]),
    })
    .with_temporal(temporal);
    let result = Study::series(series)
        .graph(graph)
        .query(query)
        .inference(antecedent::InferenceMode::Bayesian(
            antecedent::BayesianConfig::conjugate().n_draws(512),
        ))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(83))
        .unwrap();
    let posterior = result.posterior.as_ref().unwrap();
    let ResponseUncertainty::PointwiseBand { lower, upper, .. } =
        &result.response.as_ref().unwrap().uncertainty
    else {
        panic!("missing band")
    };
    assert_eq!(lower[0].to_bits(), posterior.summaries.q025[0].to_bits());
    assert_eq!(upper[0].to_bits(), posterior.summaries.q975[0].to_bits());
}
