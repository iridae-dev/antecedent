use antecedent_core::{
    Assumption, AssumptionSet, CausalResponse, ContinuousDomain, ExecutionContext, GridSpec,
    IdentificationStatus, Intervention, InterventionSequence, MechanismOverride,
    ResponseFunctional, ResponseIdentification, ResponseQuery, ResponseUncertainty, ResponseValue,
    SupportStatus, TargetPopulation, TemporalEffectQuery, TemporalResponseSpec, Value, VariableId,
};
use antecedent_data::{TemporalIndexer, TimeSeriesData};
use antecedent_expr::IdentifiedEstimand;
use antecedent_stats::{CompiledDesign, LeastSquaresWorkspace};

use crate::adjustment::PreparedEstimationProblem;
use crate::overlap::OverlapPolicy;
use crate::temporal_block::{AlignedRows, aligned_block_bootstrap};
use crate::temporal_sequential::SequentialNodeOverlay;
use crate::util::{range, sample_std};

use super::*;
use antecedent_core::TemporalPolicy;

#[test]
fn soft_constant_resolves_to_set() {
    let v = VariableId::from_raw(0);
    let (t, level, shift) =
        resolve_one(&Intervention::soft(v, MechanismOverride::constant(1.5)), 0).unwrap();
    assert_eq!(t, v);
    assert_eq!(level, Some(1.5));
    assert!(shift.abs() < f64::EPSILON);
}

#[test]
fn soft_unknown_family_refuses() {
    let v = VariableId::from_raw(0);
    let err = resolve_one(
        &Intervention::soft(v, MechanismOverride::named("linear_gaussian", vec![1.0])),
        0,
    )
    .unwrap_err();
    assert!(err.to_string().contains("not licensed"));
}

#[test]
fn non_finite_and_ambiguous_soft_parameters_fail_closed() {
    let v = VariableId::from_raw(0);
    let non_finite = resolve_one(&Intervention::set(v, Value::f64(f64::NAN)), 0).unwrap_err();
    assert!(non_finite.to_string().contains("finite numeric"));

    let too_many = resolve_one(
        &Intervention::soft(v, MechanismOverride::named("additive_shift", vec![1.0, 2.0])),
        0,
    )
    .unwrap_err();
    assert!(too_many.to_string().contains("exactly one finite parameter"));
}

#[test]
fn multi_step_sequence_is_consecutive_overlays_not_last_step() {
    use antecedent_core::SequencedIntervention;

    let v = VariableId::from_raw(0);
    let spec = TemporalResponseSpec::new(vec![1u32], TemporalPolicy::pulse(-1), None).unwrap();
    let seq = InterventionSequence {
        steps: Arc::from(vec![
            SequencedIntervention {
                intervention: Intervention::set(v, Value::f64(0.0)),
                temporal: TemporalPolicy::pulse(0),
            },
            SequencedIntervention {
                intervention: Intervention::set(v, Value::f64(5.0)),
                temporal: TemporalPolicy::pulse(0),
            },
        ]),
    };
    let TemporalInterventionPlan::Sequential { overlays } =
        plan_temporal_intervention(&[Intervention::Sequence(seq)], &spec).unwrap()
    else {
        panic!("expected sequential overlays");
    };
    assert_eq!(overlays.len(), 2);
    assert_eq!(overlays[0].offset, -2);
    assert_eq!(overlays[0].level, Some(0.0));
    assert_eq!(overlays[1].offset, -1);
    assert_eq!(overlays[1].level, Some(5.0));
    assert_ne!(overlays[0].level, overlays[1].level);
}

#[test]
fn implicit_single_step_pulse_zero_attaches_to_spec_origin() {
    use antecedent_core::SequencedIntervention;

    let variable = VariableId::from_raw(0);
    let spec = TemporalResponseSpec::new(vec![1u32], TemporalPolicy::pulse(-1), None).unwrap();
    let sequence = InterventionSequence::new([SequencedIntervention {
        intervention: Intervention::set(variable, Value::f64(1.0)),
        temporal: TemporalPolicy::pulse(0),
    }]);
    let TemporalInterventionPlan::Sequential { overlays } =
        plan_temporal_intervention(&[Intervention::Sequence(sequence)], &spec).unwrap()
    else {
        panic!("an implicit Pulse(0) Sequence must resolve to overlays");
    };
    assert_eq!(
        overlays,
        vec![SequentialNodeOverlay { variable, offset: -1, level: Some(1.0), shift: 0.0 }]
    );
}

#[test]
fn sequence_step_order_is_the_schedule_not_a_set() {
    use antecedent_core::SequencedIntervention;

    let variable = VariableId::from_raw(0);
    let spec = TemporalResponseSpec::new(vec![1u32], TemporalPolicy::pulse(-1), None).unwrap();
    let forward = InterventionSequence::new([
        SequencedIntervention {
            intervention: Intervention::set(variable, Value::f64(0.0)),
            temporal: TemporalPolicy::pulse(0),
        },
        SequencedIntervention {
            intervention: Intervention::set(variable, Value::f64(5.0)),
            temporal: TemporalPolicy::pulse(0),
        },
    ]);
    let reversed = InterventionSequence::new([
        SequencedIntervention {
            intervention: Intervention::set(variable, Value::f64(5.0)),
            temporal: TemporalPolicy::pulse(0),
        },
        SequencedIntervention {
            intervention: Intervention::set(variable, Value::f64(0.0)),
            temporal: TemporalPolicy::pulse(0),
        },
    ]);
    let TemporalInterventionPlan::Sequential { overlays: a } =
        plan_temporal_intervention(&[Intervention::Sequence(forward)], &spec).unwrap()
    else {
        panic!("expected sequential overlays");
    };
    let TemporalInterventionPlan::Sequential { overlays: b } =
        plan_temporal_intervention(&[Intervention::Sequence(reversed)], &spec).unwrap()
    else {
        panic!("expected sequential overlays");
    };
    assert_eq!(a[0].offset, -2);
    assert_eq!(a[1].offset, -1);
    assert_eq!(a[0].level, Some(0.0));
    assert_eq!(a[1].level, Some(5.0));
    assert_eq!(b[0].offset, -2);
    assert_eq!(b[1].offset, -1);
    assert_eq!(b[0].level, Some(5.0));
    assert_eq!(b[1].level, Some(0.0));
    assert_ne!(a, b);
}

#[test]
fn single_step_sequence_preserves_explicit_pulse_offset() {
    use antecedent_core::SequencedIntervention;

    let variable = VariableId::from_raw(0);
    let spec = TemporalResponseSpec::new(vec![1u32], TemporalPolicy::pulse(-1), None).unwrap();
    let sequence = InterventionSequence::new([SequencedIntervention {
        intervention: Intervention::set(variable, Value::f64(2.0)),
        temporal: TemporalPolicy::pulse(-2),
    }]);
    let TemporalInterventionPlan::Sequential { overlays } =
        plan_temporal_intervention(&[Intervention::Sequence(sequence)], &spec).unwrap()
    else {
        panic!("an explicit Sequence must resolve to overlays");
    };
    assert_eq!(
        overlays,
        vec![SequentialNodeOverlay { variable, offset: -2, level: Some(2.0), shift: 0.0 }]
    );
}

#[test]
fn single_step_sequence_expands_explicit_sustained_policy() {
    use antecedent_core::SequencedIntervention;

    let variable = VariableId::from_raw(0);
    let spec = TemporalResponseSpec::new(vec![1u32], TemporalPolicy::pulse(-1), None).unwrap();
    let sequence = InterventionSequence::new([SequencedIntervention {
        intervention: Intervention::shift(variable, Value::f64(0.5)),
        temporal: TemporalPolicy::sustained(-3, -1),
    }]);
    let TemporalInterventionPlan::Sequential { overlays } =
        plan_temporal_intervention(&[Intervention::Sequence(sequence)], &spec).unwrap()
    else {
        panic!("an explicit Sequence must resolve to overlays");
    };
    assert_eq!(overlays.iter().map(|overlay| overlay.offset).collect::<Vec<_>>(), vec![-3, -2, -1]);
    assert!(
        overlays
            .iter()
            .all(|overlay| overlay.level.is_none() && (overlay.shift - 0.5).abs() < f64::EPSILON)
    );
}

#[test]
fn nested_sequence_fails_closed() {
    use antecedent_core::SequencedIntervention;

    let v = VariableId::from_raw(0);
    let inner = InterventionSequence {
        steps: Arc::from(vec![SequencedIntervention {
            intervention: Intervention::set(v, Value::f64(0.0)),
            temporal: antecedent_core::TemporalPolicy::pulse(0),
        }]),
    };
    let outer = InterventionSequence {
        steps: Arc::from(vec![SequencedIntervention {
            intervention: Intervention::Sequence(inner),
            temporal: antecedent_core::TemporalPolicy::pulse(0),
        }]),
    };
    let spec = TemporalResponseSpec::new(vec![1u32], TemporalPolicy::pulse(-1), None).unwrap();
    let err = plan_temporal_intervention(&[Intervention::Sequence(outer)], &spec).unwrap_err();
    assert!(err.to_string().contains("not licensed"));
}

// ---- GAP1: uncertainty is computed but was never asserted anywhere ----
//
// The band was fixed from a wrong formula (ATE-coefficient SE scaled by dose,
// which gave a ZERO-WIDTH 95% interval at dose 0) to the correct linear-functional
// variance `cbar(a)' Sigma cbar(a)`. These tests would fail against the old formula.

use antecedent_core::{
    CausalSchemaBuilder, Lag, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex, ValidityBitmap,
};
use antecedent_graph::{TemporalDag, ensure_lagged};
use antecedent_identify::TemporalBackdoorIdentifier;

use super::assumptions::HorizonTempering;
use super::assumptions::temporal_bayesian_tempering_assumption;
use super::band::HORIZON_BOOTSTRAP_STREAM;
use super::plan::resolve_one;

/// Deterministic AR(2)-ish series: `t` is a mildly autocorrelated continuous
/// treatment, `y` depends on `t` lagged 1 and 2 steps. Non-degenerate `t` mean.
fn synthetic_series(n: usize) -> (TimeSeriesData, TemporalDag) {
    series_with_noise(n, 0.0, 0)
}

/// Standard-normal-ish draw (sum of twelve uniforms) from a seeded linear congruential state.
fn lcg_normal(state: &mut u64) -> f64 {
    let mut uniform = || {
        *state =
            state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((*state >> 11) as f64) / ((1_u64 << 53) as f64)
    };
    (0..12).map(|_| uniform()).sum::<f64>() - 6.0
}

/// `synthetic_series` with innovations of standard deviation `sd` on the treatment and the
/// outcome (`sd = 0` is the deterministic series).
fn series_with_noise(n: usize, sd: f64, seed: u64) -> (TimeSeriesData, TemporalDag) {
    let mut noise_state = seed.wrapping_mul(2_654_435_761).wrapping_add(17);
    let mut b = CausalSchemaBuilder::new();
    for name in ["t", "y"] {
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
    let schema = b.build().unwrap();
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 2..n {
        t[i] = 0.3 + 0.2 * t[i - 1] + 0.05 * (i as f64).sin();
        y[i] = 1.0 + 2.0 * t[i - 1] + 3.0 * t[i - 2] + 0.01 * (i as f64).cos();
        if sd > 0.0 {
            t[i] += 0.5 * sd * lcg_normal(&mut noise_state) + 0.6 * (t[i - 1] - 0.375);
            y[i] += sd * lcg_normal(&mut noise_state);
        }
    }
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(t), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(y), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let data = TimeSeriesData::try_new(
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
    (data, graph)
}

fn identify(graph: &TemporalDag, horizon_steps: u32) -> (IdentifiedEstimand, TemporalIndexer) {
    let id_query =
        TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
            .with_horizon_steps(horizon_steps)
            .with_policy(TemporalPolicy::pulse(0));
    let id_res = TemporalBackdoorIdentifier::new().identify_temporal(graph, &id_query).unwrap();
    let estimand = id_res.result.estimands.first().cloned().expect("identified estimand");
    (estimand, id_res.indexer)
}

#[test]
fn public_entry_point_revalidates_query_and_identification_status() {
    let temporal = TemporalResponseSpec::new(vec![1u32], TemporalPolicy::pulse(0), None).unwrap();
    let invalid_query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(0),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from([-1.0, 1.0])),
        ),
    })
    .with_temporal(temporal.clone());
    let (data, _) = synthetic_series(100);
    let err = TemporalResponseEstimator::new()
        .estimate(
            &data,
            &[],
            &invalid_query,
            IdentificationStatus::NonparametricallyIdentified,
            AssumptionSet::new(),
            &ExecutionContext::for_tests(7),
        )
        .unwrap_err();
    assert!(err.to_string().contains("same variable"), "got {err}");

    let valid_query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from([-1.0, 1.0])),
        ),
    })
    .with_temporal(temporal);
    let err = TemporalResponseEstimator::new()
        .estimate(
            &data,
            &[],
            &valid_query,
            IdentificationStatus::NotIdentified,
            AssumptionSet::new(),
            &ExecutionContext::for_tests(7),
        )
        .unwrap_err();
    assert!(err.to_string().contains("point identification"), "got {err}");
}

#[test]
fn response_preserves_the_exact_caller_estimand() {
    let (data, graph) = synthetic_series(300);
    let (estimand, indexer) = identify(&graph, 1);
    let temporal = TemporalResponseSpec::new(vec![1u32], TemporalPolicy::pulse(0), None).unwrap();
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Linspace { start: -1.0, end: 1.0, points: 3 },
        ),
    })
    .with_temporal(temporal);
    let result = TemporalResponseEstimator::new()
        .estimate(
            &data,
            &[(&estimand, &indexer)],
            &query,
            IdentificationStatus::NonparametricallyIdentified,
            AssumptionSet::new(),
            &ExecutionContext::for_tests(7),
        )
        .unwrap();
    assert_eq!(result.estimand, query.functional);
}

/// (a) `lower < mean < upper` strictly for every cell of the dose x horizon surface,
/// and the band is symmetric about the mean.
/// (b) A dose of exactly 0.0 in the grid produces a STRICTLY POSITIVE band width —
/// the direct regression guard for the old zero-width-at-dose-0 bug.
#[test]
fn uncertainty_band_strict_and_zero_dose_has_positive_width() {
    let (data, graph) = synthetic_series(400);
    let (estimand, indexer) = identify(&graph, 4);
    let doses = vec![-1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
    let n_h = 3;
    let temporal =
        TemporalResponseSpec::new(vec![1u32, 2, 4], TemporalPolicy::pulse(0), None).unwrap();
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from(doses.clone())),
        ),
    })
    .with_temporal(temporal);
    let est = TemporalResponseEstimator::new().with_bootstrap_replicates(60);
    let result = est
        .estimate(
            &data,
            &[(&estimand, &indexer), (&estimand, &indexer), (&estimand, &indexer)],
            &query,
            IdentificationStatus::NonparametricallyIdentified,
            AssumptionSet::new(),
            &ExecutionContext::for_tests(7),
        )
        .unwrap();

    assert!(
        result.assumptions.entries.iter().any(|r| matches!(
            &r.assumption,
            Assumption::ParametricRestriction(p) if p.id.as_ref() == "ols.homoskedastic.pointwise"
        )),
        "temporal response must record the homoskedastic pointwise OLS assumption"
    );
    assert!(result.assumptions.entries.iter().any(|r| matches!(
        &r.assumption,
        Assumption::ParametricRestriction(p) if p.id.as_ref() == "ols.linear_additive.gcomp"
    )));

    let ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) =
        &result.estimate
    else {
        panic!("expected point-identified surface");
    };
    let ResponseUncertainty::PointwiseBand { lower, upper, .. } = &result.uncertainty else {
        panic!(
            "expected PointwiseBand uncertainty on the temporal MeanCurve path — this is \
             exactly the GAP1 regression this test guards against"
        );
    };
    assert_eq!(mean.len(), doses.len() * n_h);
    assert_eq!(mean.len(), lower.len());
    assert_eq!(mean.len(), upper.len());

    for i in 0..mean.len() {
        assert!(lower[i] < mean[i], "cell {i}: lower {} not < mean {}", lower[i], mean[i]);
        assert!(mean[i] < upper[i], "cell {i}: mean {} not < upper {}", mean[i], upper[i]);
        // Symmetric about the mean by construction (mean +/- z*se); assert it holds.
        assert!(
            (mean[i] - lower[i] - (upper[i] - mean[i])).abs() < 1e-9,
            "cell {i}: band not symmetric about mean (lower half {}, upper half {})",
            mean[i] - lower[i],
            upper[i] - mean[i]
        );
    }

    // (b): dose == 0.0 must produce a strictly positive band width. Under the old
    // (buggy) formula — ATE-coefficient SE scaled by dose — the width at dose 0.0
    // was exactly zero.
    let zero_idx = doses.iter().position(|&d| d == 0.0).unwrap();
    for h in 0..n_h {
        let idx = zero_idx * n_h + h;
        let width = upper[idx] - lower[idx];
        assert!(
            width > 1e-9,
            "dose=0.0 horizon-slot {h}: band width {width} is not strictly positive \
             (regression guard: old formula gave a zero-width interval at dose 0)"
        );
    }
}

/// (c) The published bootstrap half-width grows as the dose moves away from the
/// observed treatment mean — the standard widening of a regression band away from
/// the design centroid. A band scaled by |dose| (minimized at dose 0) fails this
/// whenever the treatment mean is nonzero.
#[test]
fn bootstrap_band_widens_away_from_treatment_mean() {
    let (data, graph) = synthetic_series(400);
    let (estimand, indexer) = identify(&graph, 2);
    let temporal = TemporalResponseSpec::new(vec![2u32], TemporalPolicy::pulse(0), None).unwrap();
    let est = TemporalResponseEstimator::new().with_bootstrap_replicates(120);
    let fitted = est
        .fit_horizon(
            &SurfaceProblem {
                data: &data,
                identifications: &[(&estimand, &indexer)],
                outcome: VariableId::from_raw(1),
                treatment: VariableId::from_raw(0),
                temporal: &temporal,
            },
            &estimand,
            2,
            &indexer,
            &ExecutionContext::for_tests(7),
            &mut LeastSquaresWorkspace::default(),
        )
        .unwrap();
    let center = fitted.treatment_mean();
    assert!(
        (0.2..0.8).contains(&center),
        "test fixture assumes a positive treatment mean below 1; got {center}"
    );
    // Increasing grid: center − 1 < 0 < center < center + 1 < center + 3.
    let doses = [center - 1.0, 0.0, center, center + 1.0, center + 3.0];
    let response = est
        .estimate(
            &data,
            &[(&estimand, &indexer)],
            &surface_query(&doses, vec![2u32]),
            IdentificationStatus::NonparametricallyIdentified,
            AssumptionSet::new(),
            &ExecutionContext::for_tests(7),
        )
        .unwrap();
    let (mean, lower, _) = pointwise(&response);
    let half: Vec<f64> = mean.iter().zip(&lower).map(|(m, l)| m - l).collect();
    let (below, zero, at, near, far) = (half[0], half[1], half[2], half[3], half[4]);
    assert!(at < near, "half-width should grow away from center: {half:?}");
    assert!(near < far, "and keep growing further out: {half:?}");
    assert!(at < below, "on both sides of the center: {half:?}");
    assert!(at < zero, "the minimum is at the centroid, not at dose 0: {half:?}");
}

// ---- GAP2: refusal paths were unasserted ----

/// (f) A `Sequence` spanning multiple target variables must refuse. Exercised directly
/// against `resolve_sequence` (rather than end-to-end through `Study`) because a
/// cross-variable `Sequence` has no unique `primary_variable`, so the facade already
/// refuses earlier ("no treatment/outcome pair") before temporal resolution runs.
#[test]
fn sequence_multiple_target_variables_fails_closed() {
    use antecedent_core::SequencedIntervention;

    let v0 = VariableId::from_raw(0);
    let v1 = VariableId::from_raw(1);
    let seq = InterventionSequence {
        steps: Arc::from(vec![
            SequencedIntervention {
                intervention: Intervention::set(v0, Value::f64(1.0)),
                temporal: antecedent_core::TemporalPolicy::pulse(0),
            },
            SequencedIntervention {
                intervention: Intervention::set(v1, Value::f64(2.0)),
                temporal: antecedent_core::TemporalPolicy::pulse(0),
            },
        ]),
    };
    let spec = TemporalResponseSpec::new(vec![1u32], TemporalPolicy::pulse(-1), None).unwrap();
    let TemporalInterventionPlan::Sequential { overlays } =
        plan_temporal_intervention(&[Intervention::Sequence(seq)], &spec).unwrap()
    else {
        panic!("joint Sequence must be sequential overlays");
    };
    assert_eq!(overlays.len(), 2);
    assert_eq!(overlays[0].variable, v0);
    assert_eq!(overlays[1].variable, v1);
    assert_eq!(overlays[0].offset, -1);
    assert_eq!(overlays[1].offset, -1);
}

/// (d) Empty dose grid must refuse with a specific message, not silently produce an
/// empty (or garbage) surface.
#[test]
fn empty_dose_grid_refuses() {
    let (data, graph) = synthetic_series(300);
    let (estimand, indexer) = identify(&graph, 2);
    let temporal = TemporalResponseSpec::new(vec![1u32], TemporalPolicy::pulse(0), None).unwrap();
    let est = TemporalResponseEstimator::new();
    let err = est
        .estimate_mean_curve(
            &SurfaceProblem {
                data: &data,
                identifications: &[(&estimand, &indexer)],
                outcome: VariableId::from_raw(1),
                treatment: VariableId::from_raw(0),
                temporal: &temporal,
            },
            &[],
            IdentificationStatus::NonparametricallyIdentified,
            AssumptionSet::new(),
            &ExecutionContext::for_tests(1),
        )
        .unwrap_err();
    assert!(err.to_string().contains("dose grid must be non-empty"), "unexpected error: {err}");
}

/// The union of per-horizon treatment ranges would call dose 1.5 supported
/// here; the cell grid must not.
#[test]
fn surface_support_uses_per_horizon_ranges_not_union() {
    let temporal =
        TemporalResponseSpec::new(vec![1u32, 2, 8], TemporalPolicy::pulse(0), None).unwrap();
    let ranges = [(-2.0, 2.0), (-1.0, 1.0), (-0.4, 0.4)];
    let report = mean_curve_support(&[1.5], &temporal, &ranges);
    assert_eq!(report.status, SupportStatus::Extrapolative);
    assert_eq!(
        report.point_status.as_ref().map(AsRef::as_ref),
        Some(
            [
                SupportStatus::Supported,
                SupportStatus::OutsideEmpiricalSupport,
                SupportStatus::OutsideEmpiricalSupport,
            ]
            .as_slice()
        )
    );
    assert!(
        report
            .warnings
            .iter()
            .any(|w| w.code.as_ref() == "response.temporal.partial_horizon_support")
    );
    let ranges_diag = report
        .diagnostics
        .iter()
        .find(|d| d.id.as_ref() == "response.temporal.horizon_treatment_range")
        .expect("horizon treatment range diagnostic");
    assert_eq!(ranges_diag.values.as_ref(), [-2.0, 2.0, -1.0, 1.0, -0.4, 0.4].as_slice());
}

#[test]
fn surface_support_all_outside_and_all_supported() {
    let temporal =
        TemporalResponseSpec::new(vec![1u32, 2], TemporalPolicy::pulse(0), None).unwrap();
    let ranges = [(-1.0, 1.0), (-0.5, 0.5)];
    let outside = mean_curve_support(&[10.0], &temporal, &ranges);
    assert_eq!(outside.status, SupportStatus::OutsideEmpiricalSupport);
    assert!(
        outside
            .point_status
            .as_ref()
            .unwrap()
            .iter()
            .all(|s| { *s == SupportStatus::OutsideEmpiricalSupport })
    );
    assert!(
        outside.warnings.iter().any(|w| w.code.as_ref() == "response.outside_empirical_support")
    );

    let inside = mean_curve_support(&[0.0], &temporal, &ranges);
    assert_eq!(inside.status, SupportStatus::Supported);
    assert!(inside.warnings.is_empty());
    assert!(inside.point_status.as_ref().unwrap().iter().all(|s| *s == SupportStatus::Supported));
}

#[test]
fn intervention_support_is_one_cell_per_horizon() {
    let temporal =
        TemporalResponseSpec::new(vec![1u32, 2, 8], TemporalPolicy::pulse(0), None).unwrap();
    let ranges = [(-2.0, 2.0), (-1.0, 1.0), (-0.4, 0.4)];
    let report = intervention_support(&[1.5, 1.5, 1.5], Some(1.5), 0.0, &temporal, &ranges);
    assert_eq!(report.status, SupportStatus::Extrapolative);
    assert_eq!(report.point_status.as_ref().unwrap().len(), 3);
    let layout = report
        .diagnostics
        .iter()
        .find(|diagnostic| {
            diagnostic.id.as_ref() == "response.temporal.intervention_horizon_layout"
        })
        .expect("intervention layout");
    assert_eq!(layout.values.as_ref(), &[3.0]);
}

#[test]
fn shift_support_checks_the_shifted_law_not_only_its_mean() {
    let temporal = TemporalResponseSpec::new(vec![1u32], TemporalPolicy::pulse(0), None).unwrap();
    let ranges = [(0.0, 1.0)];

    // E[A] + 0.4 = 0.9 is inside the observed range, but the shifted
    // treatment law spans [0.4, 1.4] and therefore extrapolates.
    let partial = intervention_support(&[0.9], None, 0.4, &temporal, &ranges);
    assert_eq!(partial.status, SupportStatus::Extrapolative);
    assert_eq!(partial.point_status.as_deref(), Some(&[SupportStatus::Extrapolative][..]));
    assert!(partial.warnings.iter().any(|warning| {
        warning.code.as_ref() == "response.temporal.shift_distribution_extrapolative"
    }));
    assert_eq!(partial.query_region.minima.as_ref(), &[0.4, 1.0]);
    assert_eq!(partial.query_region.maxima.as_ref(), &[1.4, 1.0]);
    let shifted = partial
        .diagnostics
        .iter()
        .find(|diagnostic| diagnostic.id.as_ref() == "response.temporal.shifted_treatment_range")
        .expect("shifted treatment range");
    assert_eq!(shifted.values.as_ref(), &[0.4, 1.4]);

    // A disjoint shifted law gets the stronger outside-support status.
    let outside = intervention_support(&[2.5], None, 2.0, &temporal, &ranges);
    assert_eq!(outside.status, SupportStatus::OutsideEmpiricalSupport);
    assert_eq!(
        outside.point_status.as_deref(),
        Some(&[SupportStatus::OutsideEmpiricalSupport][..])
    );

    let identity = intervention_support(&[0.5], None, 0.0, &temporal, &ranges);
    assert_eq!(identity.status, SupportStatus::Supported);
}

/// Extreme T only at the end of the series. A longer-horizon pulse looks
/// further back, so that spike is inside the h=1 treatment column and
/// outside a long-horizon window.
fn spike_then_quiet_series(n: usize) -> (TimeSeriesData, TemporalDag) {
    let mut b = CausalSchemaBuilder::new();
    for name in ["t", "y"] {
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
    let schema = b.build().unwrap();
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    for (i, t_i) in t.iter_mut().enumerate() {
        *t_i = if i >= n.saturating_sub(7) { 10.0 } else { 0.05 * (i as f64).sin() };
    }
    for i in 2..n {
        y[i] = 1.0 + 2.0 * t[i - 1] + 3.0 * t[i - 2];
    }
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(t), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(y), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let data = TimeSeriesData::try_new(
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
    (data, graph)
}

#[test]
fn late_treatment_spike_is_horizon_specific_support() {
    let (data, graph) = spike_then_quiet_series(80);
    let (estimand, indexer) = identify(&graph, 8);
    let temporal =
        TemporalResponseSpec::new(vec![1u32, 8], TemporalPolicy::pulse(0), None).unwrap();
    let est = TemporalResponseEstimator::new();
    let mut ws = LeastSquaresWorkspace::default();
    let ctx = ExecutionContext::for_tests(11);
    let short = est
        .fit_horizon(
            &SurfaceProblem {
                data: &data,
                identifications: &[(&estimand, &indexer)],
                outcome: VariableId::from_raw(1),
                treatment: VariableId::from_raw(0),
                temporal: &temporal,
            },
            &estimand,
            1,
            &indexer,
            &ctx,
            &mut ws,
        )
        .unwrap();
    let long = est
        .fit_horizon(
            &SurfaceProblem {
                data: &data,
                identifications: &[(&estimand, &indexer)],
                outcome: VariableId::from_raw(1),
                treatment: VariableId::from_raw(0),
                temporal: &temporal,
            },
            &estimand,
            8,
            &indexer,
            &ctx,
            &mut ws,
        )
        .unwrap();
    let short_range = range(&short.prepared.treatment);
    let long_range = range(&long.prepared.treatment);
    assert!(
        short_range.1 > long_range.1 + 1.0,
        "long horizon should miss the late spike: short={short_range:?} long={long_range:?}"
    );
    let dose = long_range.1 + (short_range.1 - long_range.1) * 0.5;
    let result = est
        .estimate_mean_curve(
            &SurfaceProblem {
                data: &data,
                identifications: &[(&estimand, &indexer), (&estimand, &indexer)],
                outcome: VariableId::from_raw(1),
                treatment: VariableId::from_raw(0),
                temporal: &temporal,
            },
            &[0.0, dose],
            IdentificationStatus::NonparametricallyIdentified,
            AssumptionSet::new(),
            &ctx,
        )
        .unwrap();
    assert_eq!(result.support.status, SupportStatus::Extrapolative);
    let cells = result.support.point_status.as_ref().expect("temporal point_status");
    // dose-major: (0, h=1), (0, h=8), (dose, h=1), (dose, h=8)
    assert_eq!(cells[0], SupportStatus::Supported);
    assert_eq!(cells[1], SupportStatus::Supported);
    assert_eq!(cells[2], SupportStatus::Supported);
    assert_eq!(cells[3], SupportStatus::OutsideEmpiricalSupport);
    // The union envelope would have classified `dose` as supported.
    assert!(dose >= short_range.0 && dose <= short_range.1);
    assert!(dose < long_range.0 || dose > long_range.1);
}

fn surface_query(doses: &[f64], horizons: Vec<u32>) -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from(doses.to_vec())),
        ),
    })
    .with_temporal(TemporalResponseSpec::new(horizons, TemporalPolicy::pulse(0), None).unwrap())
}

fn pointwise(response: &CausalResponse) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) =
        &response.estimate
    else {
        panic!("expected surface");
    };
    let ResponseUncertainty::PointwiseBand { lower, upper, .. } = &response.uncertainty else {
        panic!("expected pointwise band");
    };
    (mean.to_vec(), lower.to_vec(), upper.to_vec())
}

fn diagnostic<'a>(response: &'a CausalResponse, id: &str) -> Option<&'a [f64]> {
    response.support.diagnostics.iter().find(|d| d.id.as_ref() == id).map(|d| d.values.as_ref())
}

/// Zero replicates publish no band (the analytic OLS band would treat
/// lag-aligned rows as independent) and say so; requested replicates publish
/// the circular-block band around the same full-sample point estimate.
#[test]
fn zero_replicates_withhold_the_band_and_the_bootstrap_publishes_it() {
    let (data, graph) = synthetic_series(240);
    let (estimand, indexer) = identify(&graph, 3);
    let query = surface_query(&[0.0, 1.0], vec![3u32]);
    let run = |replicates| {
        TemporalResponseEstimator::new()
            .with_bootstrap_replicates(replicates)
            .estimate(
                &data,
                &[(&estimand, &indexer)],
                &query,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
                &ExecutionContext::for_tests(13),
            )
            .unwrap()
    };
    let analytic = run(0);
    assert!(matches!(analytic.uncertainty, ResponseUncertainty::None));
    assert!(
        analytic
            .support
            .warnings
            .iter()
            .any(|w| w.code.as_ref() == TEMPORAL_RESPONSE_BAND_WITHHELD),
        "a withheld band must be diagnosed"
    );
    let ResponseIdentification::PointIdentified(ResponseValue::Surface { mean: mu_a, .. }) =
        &analytic.estimate
    else {
        panic!("expected surface");
    };
    let (mu_b, lo_b, _) = pointwise(&run(40));
    let se_b = mu_b[1] - lo_b[1];
    assert!(se_b.is_finite() && se_b > 0.0, "bootstrap half-width={se_b}");
    assert!(
        (mu_a[0] - mu_b[0]).abs() < 1e-12,
        "point estimate must stay full-sample OLS (analytic={}, boot={})",
        mu_a[0],
        mu_b[0]
    );
}

/// The bootstrap surface publishes a simultaneous band next to the pointwise
/// band; it contains the pointwise band at every cell and names its construction.
/// The analytic path withholds it explicitly.
#[test]
fn bootstrap_surface_publishes_simultaneous_band_alongside_pointwise() {
    let (data, graph) = synthetic_series(300);
    let (estimand, indexer) = identify(&graph, 3);
    let doses = [-1.0, 0.0, 1.0, 2.0];
    let query = surface_query(&doses, vec![1u32, 2, 3]);
    let ids = [(&estimand, &indexer), (&estimand, &indexer), (&estimand, &indexer)];
    let boot = TemporalResponseEstimator::new()
        .with_bootstrap_replicates(80)
        .estimate(
            &data,
            &ids,
            &query,
            IdentificationStatus::NonparametricallyIdentified,
            AssumptionSet::new(),
            &ExecutionContext::for_tests(3),
        )
        .unwrap();
    let (mean, lower, upper) = pointwise(&boot);
    let sim_lo = diagnostic(&boot, SIMULTANEOUS_BAND_LOWER).expect("simultaneous lower");
    let sim_hi = diagnostic(&boot, SIMULTANEOUS_BAND_UPPER).expect("simultaneous upper");
    let critical = diagnostic(&boot, SIMULTANEOUS_BAND_CRITICAL).expect("critical");
    assert_eq!(sim_lo.len(), mean.len());
    assert_eq!(sim_hi.len(), mean.len());
    assert!((critical[0] - 0.95).abs() < 1e-12);
    assert!((critical[2] - 80.0).abs() < 1e-12);
    assert!(critical[1] >= antecedent_stats::NORMAL_Q975, "sup-t critical {} < z", critical[1]);
    for cell in 0..mean.len() {
        assert!(sim_lo[cell] <= lower[cell] && upper[cell] <= sim_hi[cell], "cell {cell}");
        let half_pointwise = mean[cell] - lower[cell];
        let half_sim = mean[cell] - sim_lo[cell];
        assert!(
            (half_sim / half_pointwise - critical[1] / antecedent_stats::NORMAL_Q975).abs() < 1e-9,
            "simultaneous band must scale the same bootstrap SE by the sup-t critical value"
        );
    }
    assert!(boot.assumptions.entries.iter().any(|r| matches!(
        &r.assumption,
        Assumption::ParametricRestriction(p) if p.id.as_ref() == "temporal_response.block_bootstrap"
    )));

    let analytic = TemporalResponseEstimator::new()
        .estimate(
            &data,
            &ids,
            &query,
            IdentificationStatus::NonparametricallyIdentified,
            AssumptionSet::new(),
            &ExecutionContext::for_tests(3),
        )
        .unwrap();
    assert!(diagnostic(&analytic, SIMULTANEOUS_BAND_LOWER).is_none());
    assert!(
        analytic
            .support
            .warnings
            .iter()
            .any(|warning| warning.code.as_ref() == SIMULTANEOUS_BAND_WITHHELD)
    );
}

/// Run the three-horizon surface of `synthetic_series` with joint replicates retained, under
/// the bootstrap or the block-weight posterior.
fn joint_surface(weighted: bool, retain: bool) -> CausalResponse {
    let (data, graph) = synthetic_series(300);
    let (estimand, indexer) = identify(&graph, 3);
    let query = surface_query(&[-1.0, 0.0, 1.0, 2.0], vec![1u32, 2, 3]);
    let ids = [(&estimand, &indexer), (&estimand, &indexer), (&estimand, &indexer)];
    TemporalResponseEstimator::new()
        .with_bootstrap_replicates(80)
        .with_block_weight_posterior(weighted)
        .with_joint_replicates(retain)
        .estimate(
            &data,
            &ids,
            &query,
            IdentificationStatus::NonparametricallyIdentified,
            AssumptionSet::new(),
            &ExecutionContext::for_tests(3),
        )
        .unwrap()
}

/// The joint replicates are retained only on request, are replicate-major in the cell layout of
/// the mean surface, and every row is one realization of all horizons: the weighted draws centre
/// on the point surface and spread about as the bootstrap does, and their band reads as credible.
#[test]
fn joint_horizon_replicates_are_retained_on_request_and_couple_the_horizons() {
    let off = joint_surface(false, false);
    assert!(diagnostic(&off, JOINT_HORIZON_REPLICATES).is_none(), "nothing retained by default");

    let boot = joint_surface(false, true);
    let layout = diagnostic(&boot, JOINT_HORIZON_LAYOUT).expect("layout");
    let values = diagnostic(&boot, JOINT_HORIZON_REPLICATES).expect("replicates");
    let (replicates, cells) = (layout[0] as usize, layout[1] as usize);
    assert_eq!(cells, 12, "four doses by three horizons");
    assert!(replicates >= 40);
    assert_eq!(values.len(), replicates * cells);
    let horizon_of_cell: Vec<usize> = layout[2..].iter().map(|h| *h as usize).collect();
    assert_eq!(horizon_of_cell, [0, 1, 2].repeat(4), "dose-major, then horizon");
    let (mean, lower, upper) = pointwise(&boot);
    let ResponseUncertainty::PointwiseBand { interpretation, .. } = &boot.uncertainty else {
        panic!("expected a pointwise band");
    };
    assert_eq!(*interpretation, antecedent_core::IntervalInterpretation::Confidence);
    for cell in 0..cells {
        let column: Vec<f64> = (0..replicates).map(|r| values[r * cells + cell]).collect();
        let sd = antecedent_stats::sample_std(&column);
        // The replicates resample the window every horizon shares, while the point surface
        // fits each horizon on all its own rows, so the replicate centre need not equal the
        // point: what the band reads is the spread of the replicates about the point.
        assert!(column.iter().all(|v| v.is_finite()), "cell {cell}");
        let half = (upper[cell] - lower[cell]) / 2.0;
        assert!((half / (antecedent_stats::NORMAL_Q975 * sd) - 1.0).abs() < 1e-9, "cell {cell}");
    }

    let posterior = joint_surface(true, true);
    let weighted = diagnostic(&posterior, JOINT_HORIZON_REPLICATES).expect("weighted replicates");
    let weighted_layout = diagnostic(&posterior, JOINT_HORIZON_LAYOUT).expect("layout");
    assert_eq!(weighted_layout[1] as usize, cells);
    assert_eq!(weighted.len(), weighted_layout[0] as usize * cells, "replicate-major layout");
    assert!(weighted.iter().all(|v| v.is_finite()));
    let ResponseUncertainty::PointwiseBand { interpretation, lower: wl, upper: wu, .. } =
        &posterior.uncertainty
    else {
        panic!("expected a pointwise band");
    };
    assert_eq!(*interpretation, antecedent_core::IntervalInterpretation::Credible);
    let (point, ..) = pointwise(&posterior);
    for cell in 0..cells {
        assert!((point[cell] - mean[cell]).abs() < 1e-12, "same full-sample point surface");
        let ratio = (wu[cell] - wl[cell]) / (upper[cell] - lower[cell]);
        assert!(
            (0.25..4.0).contains(&ratio),
            "cell {cell}: the weighted posterior must spread about as the bootstrap does, \
             width ratio {ratio}"
        );
    }
}

/// `synthetic_series` with `gap` missing time points starting at `at` (both columns).
fn gapped_series(n: usize, at: usize, gap: usize) -> (TimeSeriesData, TemporalDag) {
    let (data, graph) = synthetic_series(n);
    let mut bytes = vec![0xFFu8; n.div_ceil(8)];
    for row in at..at + gap {
        bytes[row / 8] &= !(1 << (row % 8));
    }
    let validity = ValidityBitmap::from_bytes(bytes, n).unwrap();
    let columns = [VariableId::from_raw(0), VariableId::from_raw(1)]
        .into_iter()
        .map(|id| {
            let values: Vec<f64> = data.float64_slice(id).unwrap().to_vec();
            OwnedColumn::Float64(
                Float64Column::new(id, Arc::from(values), validity.clone()).unwrap(),
            )
        })
        .collect();
    let storage =
        OwnedColumnarStorage::try_new(data.schema().clone(), columns, None, None).unwrap();
    let gapped = TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap();
    (gapped, graph)
}

/// On a series with missing time points the block-weight posterior puts its weights on blocks of
/// the original time axis and only on the times observed in every horizon: it publishes joint
/// replicates, none of which is built from a row inside the gap.
#[test]
fn the_block_weight_posterior_runs_on_a_series_with_missing_time_points() {
    let (data, graph) = gapped_series(300, 120, 6);
    let (estimand, indexer) = identify(&graph, 2);
    let query = surface_query(&[0.0, 1.0], vec![1u32, 2]);
    let ids = [(&estimand, &indexer), (&estimand, &indexer)];
    let response = TemporalResponseEstimator::new()
        .with_allow_gaps(true)
        .with_bootstrap_replicates(60)
        .with_block_weight_posterior(true)
        .with_joint_replicates(true)
        .estimate(
            &data,
            &ids,
            &query,
            IdentificationStatus::NonparametricallyIdentified,
            AssumptionSet::new(),
            &ExecutionContext::for_tests(5),
        )
        .unwrap();
    let layout = diagnostic(&response, JOINT_HORIZON_LAYOUT).expect("layout");
    assert!(layout[0] >= 40.0, "replicates {}", layout[0]);
    assert_eq!(layout[1] as usize, 4, "two doses by two horizons");
    let ResponseUncertainty::PointwiseBand { interpretation, .. } = &response.uncertainty else {
        panic!("expected a pointwise band");
    };
    assert_eq!(*interpretation, antecedent_core::IntervalInterpretation::Credible);
    let values = diagnostic(&response, JOINT_HORIZON_REPLICATES).expect("replicates");
    assert!(values.iter().all(|v| v.is_finite()));
}

/// Every horizon of one replicate resamples the same calendar times, in circular
/// runs of the block length over the common window, and every window time is an
/// equally likely draw (no row gets a second chance to start a block).
#[test]
fn surface_bootstrap_rows_are_calendar_aligned_over_the_common_window() {
    // Series of 20 times; horizon A keeps rows at times 2..20, horizon B at 4..20.
    let rows = [AlignedRows { first_time: 2, rows: 18 }, AlignedRows { first_time: 4, rows: 16 }];
    let block = 5;
    let ctx = ExecutionContext::for_tests(11);
    let mut hits = [0usize; 20];
    let boot =
        aligned_block_bootstrap(&rows, block, 2000, HORIZON_BOOTSTRAP_STREAM, &ctx, |maps| {
            assert_eq!(maps[0].len(), 16, "every horizon resamples the common window");
            assert_eq!(maps[1].len(), 16);
            let times: Vec<usize> = maps[1].iter().map(|&row| 4 + row).collect();
            for (r, (&a, &b)) in maps[0].iter().zip(&maps[1]).enumerate() {
                assert_eq!(2 + a, 4 + b, "row {r}: horizons disagree on calendar time");
            }
            for chunk in times.chunks(block) {
                for pair in chunk.windows(2) {
                    // Consecutive times inside a block, wrapping within the window 4..20.
                    assert_eq!(pair[1], 4 + (pair[0] - 4 + 1) % 16, "block broke contiguity");
                }
            }
            for &time in &times {
                assert!((4..20).contains(&time), "time {time} outside the common window");
                hits[time] += 1;
            }
            Some(vec![0.0])
        })
        .expect("the horizons share a window");
    assert_eq!(boot.draws.len(), 2000);
    assert_eq!(boot.rows, 16);
    let expected = 2000.0;
    for (time, &count) in hits.iter().enumerate().skip(4) {
        let ratio = count as f64 / expected;
        assert!((0.85..1.15).contains(&ratio), "time {time} drawn {count} times");
    }
    assert_eq!(hits[..4].iter().sum::<usize>(), 0);

    // max(span, ceil(sqrt(n))), capped at n.
    assert_eq!(temporal_block_length(3, 160), 13);
    assert_eq!(temporal_block_length(20, 160), 20);
    assert_eq!(temporal_block_length(1, 144), 12);
    assert_eq!(temporal_block_length(9, 4), 4);
}

/// The response block length keeps the rule on short-memory scores, lengthens it on a
/// persistent score, and reports when the n/3 cap binds.
#[test]
fn response_block_length_lengthens_on_persistent_scores() {
    /// The stated rule, written out from its definition and not from the product code:
    /// `rule = max(span, ceil(sqrt(rows)))` (capped at `rows`), `testing = ceil(b_PW rows^(1/6))`
    /// with `b_PW` the largest Politis-White flat-top block length over the scores, and
    /// `length = max(rule, min(testing, rows / 3))` (at most `rows`).
    fn politis_white(scores: &[f64]) -> usize {
        let n = scores.len();
        let nf = n as f64;
        let mean = scores.iter().sum::<f64>() / nf;
        let centered: Vec<f64> = scores.iter().map(|s| s - mean).collect();
        let autocov = |k: usize| -> f64 {
            centered[..n - k].iter().zip(&centered[k..]).map(|(a, b)| a * b).sum::<f64>() / nf
        };
        let gamma0 = autocov(0);
        let k_n = 5usize.max(nf.log10().sqrt().ceil() as usize);
        let m_max = (nf.sqrt().ceil() as usize + k_n).min(n - 1);
        let threshold = 2.0 * (nf.log10() / nf).sqrt();
        let rho: Vec<f64> = (0..=m_max).map(|k| autocov(k) / gamma0).collect();
        // first m after which K_n consecutive autocorrelations are insignificant
        let m_hat = (0..=m_max.saturating_sub(k_n))
            .find(|&m| (1..=k_n).all(|j| rho.get(m + j).is_none_or(|r| r.abs() < threshold)))
            .unwrap_or(m_max.saturating_sub(k_n));
        let big_m = (2 * m_hat).min(m_max).max(1);
        let (mut g0, mut big_g) = (gamma0, 0.0);
        for k in 1..=big_m {
            let t = k as f64 / big_m as f64;
            let weight = if t <= 0.5 { 1.0 } else { 2.0 * (1.0 - t) } * rho[k] * gamma0;
            g0 += 2.0 * weight;
            big_g += 2.0 * k as f64 * weight;
        }
        let b = (2.0 * big_g * big_g / (4.0 / 3.0 * g0 * g0)).cbrt() * nf.cbrt();
        let cap = (3.0 * nf.sqrt()).min(nf / 3.0).ceil();
        b.clamp(1.0, cap.max(1.0)).ceil() as usize
    }
    let stated = |span: usize, rows: usize, scores: &[&[f64]]| -> (usize, usize, usize) {
        let rule = span.max((rows as f64).sqrt().ceil() as usize).min(rows).max(1);
        let pw = scores.iter().map(|s| politis_white(s)).max().unwrap_or(0);
        let testing = (pw as f64 * (rows as f64).powf(1.0 / 6.0)).ceil() as usize;
        (rule.max(testing.min(rows / 3)).min(rows.max(1)), rule, testing)
    };

    let n = 400;
    let white = gaussian_series(n, 0.0, 3);
    let persistent = gaussian_series(n, 0.9, 5);
    let short = ResponseBlockLength::new(2, n, &[&white]);
    let (length, rule, testing) = stated(2, n, &[&white]);
    assert_eq!(short.rule, 20);
    assert_eq!(short.length, 20, "{short:?}");
    assert_eq!((short.length, short.rule, short.testing), (length, rule, testing), "{short:?}");
    assert!(!short.capped());
    let long = ResponseBlockLength::new(2, n, &[&white, &persistent]);
    let (length, rule, testing) = stated(2, n, &[&white, &persistent]);
    assert_eq!((long.length, long.rule, long.testing), (length, rule, testing), "{long:?}");
    assert!(long.length > long.rule, "{long:?}");
    assert!(long.length <= n / 3);
    assert_eq!(long.length, long.testing.min(n / 3));
    let tight = ResponseBlockLength::new(2, 60, &[&persistent[..60]]);
    let (length, rule, testing) = stated(2, 60, &[&persistent[..60]]);
    assert_eq!((tight.length, tight.rule, tight.testing), (length, rule, testing), "{tight:?}");
    assert!(tight.length <= 20, "{tight:?}");
    assert_eq!(tight.capped(), tight.testing > tight.length);
    // The stated rule evaluates to these lengths (rule, testing) on the fixed series.
    assert_eq!((short.length, short.rule, short.testing), (20, 20, 3));
    assert_eq!((long.length, long.rule, long.testing), (52, 20, 52));
    assert_eq!((tight.length, tight.rule, tight.testing), (18, 8, 18));
}

/// Deterministic AR(1) series with Gaussian innovations (Box–Muller on splitmix64).
fn gaussian_series(n: usize, rho: f64, seed: u64) -> Vec<f64> {
    let mut state = seed;
    let mut next = || crate::splitmix::splitmix64_unit(&mut state);
    let mut out = Vec::with_capacity(n);
    let mut previous = 0.0;
    for _ in 0..n {
        let (u, v) = (next().max(1e-12), next());
        let innovation = (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos();
        previous = rho * previous + innovation;
        out.push(previous);
    }
    out
}

fn fitted_horizon(
    treatment: &[f64],
    covariates: &[(VariableId, &[f64])],
    outcome: &[f64],
) -> FittedHorizon {
    let design = CompiledDesign::linear_adjustment(treatment, covariates, outcome, &[])
        .expect("balanced design");
    let prepared = PreparedEstimationProblem {
        design,
        method: Arc::from("test"),
        adjustment_set: Arc::from(covariates.iter().map(|(id, _)| *id).collect::<Vec<_>>()),
        overlap: OverlapPolicy::ExplicitOverride,
        treatment_delta: 1.0,
        target_population: TargetPopulation::AllObserved,
        treatment: Arc::from(treatment.to_vec()),
        preparation: antecedent_data::materialize::PreparationReceipt::composed_finite(
            treatment.len(),
        )
        .expect("row count fits u64"),
        active: 1.0,
        control: 0.0,
        treatment_levels: None,
        sampling_weights: None,
    };
    FittedHorizon::fit(prepared, &mut LeastSquaresWorkspace::default()).expect("OLS fit")
}

/// The level influence must be `O_p(1)` in every term: the OLS piece is
/// `n · x_t'(X'X)⁻¹ c · ê_t`, not the raw leverage weight. Without the `n`, a long
/// series makes the residual term ~`1/n` times the mean term, and duplicating every
/// row (same level, twice the rows) would shrink the residual contribution by 2.
#[test]
fn level_influence_is_op1_in_regression_and_mean_terms() {
    // Intercept + treatment, Shift(0): level is ȳ, so the influence is exactly y_t − ȳ.
    let n = 200usize;
    let a: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
    let y: Vec<f64> =
        (0..n).map(|i| 2.0 + 3.0 * a[i] + if i % 2 == 0 { 0.5 } else { -0.5 }).collect();
    let fitted = fitted_horizon(&a, &[], &y);
    let psi = fitted.level_influence(CellEval::Shift(0.0)).unwrap();
    let ybar = y.iter().sum::<f64>() / n as f64;
    for t in 0..n {
        assert!((psi[t] - (y[t] - ybar)).abs() < 1e-10, "t={t}: {} vs {}", psi[t], y[t] - ybar);
    }

    // One covariate, Dose(a): both contributions stay the same order on a long series.
    let z_id = VariableId::from_raw(2);
    let z: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
    let a2: Vec<f64> = (0..n).map(|i| if i % 4 < 2 { 1.0 } else { -1.0 }).collect();
    let y2: Vec<f64> =
        (0..n).map(|i| 1.0 + 2.0 * a2[i] + 4.0 * z[i] + 0.25 * ((i % 3) as f64 - 1.0)).collect();
    let fitted2 = fitted_horizon(&a2, &[(z_id, z.as_slice())], &y2);
    let dose = 0.5;
    let psi2 = fitted2.level_influence(CellEval::Dose(dose)).unwrap();
    let means: Vec<f64> =
        (0..n).map(|t| fitted2.coefs[2] * (z[t] - fitted2.column_means[2])).collect();
    let reg: Vec<f64> = psi2.iter().zip(&means).map(|(p, m)| p - m).collect();
    let rms = |v: &[f64]| (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt();
    let (r_reg, r_means) = (rms(&reg), rms(&means));
    assert!(r_reg > 0.05 && r_means > 0.05, "reg={r_reg} means={r_means}");
    let ratio = r_reg / r_means;
    assert!(
        (0.05..20.0).contains(&ratio),
        "regression term must not be ~1/n times the mean term: ratio={ratio}"
    );

    // Repeat every row: level unchanged, n → 2n. Influence on matching rows must stay
    // put (not shrink by 2 as a missing-n residual term would).
    let a_rep: Vec<f64> = a2.iter().flat_map(|&x| [x, x]).collect();
    let z_rep: Vec<f64> = z.iter().flat_map(|&x| [x, x]).collect();
    let y_rep: Vec<f64> = y2.iter().flat_map(|&x| [x, x]).collect();
    let fitted_rep = fitted_horizon(&a_rep, &[(z_id, z_rep.as_slice())], &y_rep);
    let psi_rep = fitted_rep.level_influence(CellEval::Dose(dose)).unwrap();
    for t in 0..n {
        assert!(
            (psi_rep[2 * t] - psi2[t]).abs() < 1e-8,
            "row {t}: repeated {} vs original {}",
            psi_rep[2 * t],
            psi2[t]
        );
    }
}

/// The horizon caches one Gram matrix and reads its OLS residuals for the estimating
/// scores; both must equal the definitions (`x_a'x_b` and, by the normal equations,
/// `Σ_t x_tj ê_t = 0`).
#[test]
fn horizon_caches_gram_and_scores_use_the_held_residuals() {
    let n = 60usize;
    let z_id = VariableId::from_raw(2);
    let a: Vec<f64> = (0..n).map(|i| ((i * 7 % 11) as f64) - 5.0).collect();
    let z: Vec<f64> = (0..n).map(|i| ((i * 5 % 13) as f64) * 0.5 - 3.0).collect();
    let y: Vec<f64> =
        (0..n).map(|i| 1.0 + 0.7 * a[i] - 0.4 * z[i] + 0.3 * ((i * 3 % 7) as f64 - 3.0)).collect();
    let fitted = fitted_horizon(&a, &[(z_id, z.as_slice())], &y);
    let cols: [Vec<f64>; 3] = [vec![1.0; n], a.clone(), z.clone()];
    // The cached factor reproduces the Gram matrix by definition (L times L-transpose
    // equals x_a'x_b) and solves against it as a per-call factorization would.
    let chol = fitted.gram_chol.as_ref().expect("the design is full rank");
    let mut gram = [0.0; 9];
    for r in 0..3 {
        for c in 0..3 {
            let dot: f64 = (0..n).map(|t| cols[r][t] * cols[c][t]).sum();
            gram[r * 3 + c] = dot;
            let llt: f64 = (0..3).map(|k| chol[r * 3 + k] * chol[c * 3 + k]).sum();
            assert!((llt - dot).abs() < 1e-8 * (1.0 + dot.abs()), "gram[{r}][{c}]");
        }
    }
    let rhs = [1.0, -2.0, 0.5];
    let cached = antecedent_stats::chol_solve(chol, 3, &rhs).unwrap();
    let direct = crate::util::solve_spd(&gram, &rhs, 3).unwrap();
    for k in 0..3 {
        assert!((cached[k] - direct[k]).abs() < 1e-10, "solve component {k}");
    }
    let scores = horizon_scores(std::slice::from_ref(&fitted));
    // 3 normal-equation series, then the 2 centered non-intercept columns.
    assert_eq!(scores.len(), 5);
    for (j, series) in scores.iter().take(3).enumerate() {
        let total: f64 = series.iter().sum();
        assert!(total.abs() < 1e-8, "normal equation {j}: sum {total}");
        for (t, &value) in series.iter().enumerate().take(n) {
            assert!((value - cols[j][t] * fitted.residuals[t]).abs() < 1e-12);
        }
    }
}

#[test]
fn block_dispersion_inflation_is_the_fixed_b_ratio_times_hc1() {
    // 159 rows in blocks of 13, 3 coefficients: circular-Bartlett fixed-b ratio at
    // b = 13/159 times the HC1 factor sqrt(159/156).
    let b = 13.0 / 159.0;
    let intercept = crate::temporal_block::FIXED_B_INTERCEPT;
    let circular = (intercept + 2.4389 * b + 3.7072 * b * b - 2.1055 * b * b * b) / intercept;
    let expected = circular * (159.0_f64 / 156.0).sqrt();
    assert!((block_dispersion_inflation(159, 13, 3) - expected).abs() < 1e-12);
    assert!(expected > 1.12 && expected < 1.13, "{expected}");
    // Long series with short blocks: no inflation to speak of.
    assert!((block_dispersion_inflation(1_000_000, 1_000, 3) - 1.0).abs() < 2e-3);
    // Barely one block: the b = 1 critical value, never "no inflation".
    assert!(block_dispersion_inflation(6, 6, 2) > 1.9);
    let center = [1.0, 2.0];
    let mut draws = vec![vec![2.0, 1.0]];
    inflate_replicates(&center, &mut draws, 1.5);
    assert_eq!(draws[0], vec![2.5, 0.5]);
}

#[test]
fn max_deviation_band_uses_the_monte_carlo_rank_and_refuses_thin_draws() {
    let center = [0.0, 10.0];
    // Replicate r deviates by r/100 SD-units in cell 0 only: maxima are sorted
    // deviations, so the critical value is the ceil(0.95·(B+1))-th of them.
    let draws: Vec<Vec<f64>> =
        (0..99).map(|r| vec![f64::from(r) - 49.0, 10.0 + f64::from(r % 2)]).collect();
    let band = max_deviation_band(&center, &draws, 0.95).unwrap();
    assert_eq!(band.replicates, 99);
    let sd0 = sample_std(&draws.iter().map(|d| d[0]).collect::<Vec<_>>());
    let sd1 = sample_std(&draws.iter().map(|d| d[1]).collect::<Vec<_>>());
    let mut maxima: Vec<f64> = draws
        .iter()
        .map(|d| ((d[0] - center[0]).abs() / sd0).max((d[1] - center[1]).abs() / sd1))
        .collect();
    maxima.sort_by(f64::total_cmp);
    assert!((band.critical - maxima[94]).abs() < 1e-15);
    assert!((band.upper[0] - band.critical * sd0).abs() < 1e-12);
    assert!((band.lower[1] - (10.0 - band.critical * sd1)).abs() < 1e-12);
    let columns: Vec<Vec<f64>> =
        (0..2).map(|cell| draws.iter().map(|d| d[cell]).collect()).collect();
    let column_refs: Vec<&[f64]> = columns.iter().map(Vec::as_slice).collect();
    assert_eq!(max_deviation_band_columns(&center, &column_refs, 0.95).unwrap(), band);
    assert!(max_deviation_band(&center, &draws[..39], 0.95).is_err());
    let flat: Vec<Vec<f64>> = (0..50).map(|r| vec![f64::from(r), 10.0]).collect();
    let flat_band = max_deviation_band(&center, &flat, 0.95).unwrap();
    assert!((flat_band.lower[1] - 10.0).abs() < 1e-12);
    assert!((flat_band.upper[1] - 10.0).abs() < 1e-12);
    assert!(flat_band.upper[0] > flat_band.lower[0]);
    // Perfectly correlated, light-tailed (uniform) cells: the Monte Carlo rank sits near
    // 1.65 SD, below z; the band is floored at the one-cell normal band.
    let uniform: Vec<Vec<f64>> =
        (0..99).map(|r| vec![f64::from(r) - 49.0, 10.0 + f64::from(r) - 49.0]).collect();
    let floored = max_deviation_band(&center, &uniform, 0.95).unwrap();
    assert!(
        (floored.critical - antecedent_stats::NORMAL_Q975).abs() < 1e-12,
        "{}",
        floored.critical
    );
}

#[test]
fn bayesian_tempering_assumption_discloses_capped_and_inestimable_horizons() {
    let horizons = [1u32, 2u32, 3u32];
    let tempering = vec![
        HorizonTempering { kappa: 1.0, capped: false, inestimable: true },
        HorizonTempering { kappa: 7.5, capped: true, inestimable: false },
        HorizonTempering { kappa: 2.0, capped: false, inestimable: false },
    ];
    let record = temporal_bayesian_tempering_assumption(&horizons, &tempering);
    let Assumption::ParametricRestriction(p) = &record.assumption else {
        panic!("expected a parametric restriction");
    };
    let text = p.description.as_ref();
    assert!(text.contains("h=1: 1.0000 inestimable"), "{text}");
    assert!(text.contains("h=2: 7.5000 capped"), "{text}");
    assert!(text.contains("h=3: 2.0000"), "{text}");
    assert!(
        text.contains("REML") && text.contains("n/(p+2)"),
        "must describe the actual fitted formula (REML AR(q), capped at n/(p+2)): {text}"
    );
    assert!(
        !text.contains(
            "the largest autoregressive-prewhitened (AR(1), plus a \
             BIC-selected AR(q <= 4)) Newey-West long-run-variance ratio of that horizon's \
             grid-cell level scores, floored at 1"
        ),
        "must not keep the wrong pre-REML formula text: {text}"
    );
}

#[test]
fn bayesian_response_retains_draws_only_for_a_single_horizon() {
    let column = vec![0.5, 1.5, 2.5];
    let one = super::bayes::retained_response_draws(std::slice::from_ref(&column), 1)
        .expect("one horizon retains its draws");
    assert_eq!(one.n_coordinates(), 1);
    assert_eq!(one.n_draws, 3);
    // Independent per-horizon fits: index r across horizons is not a joint draw, so
    // nothing is retained for a nonlinear functional to evaluate.
    let columns = vec![column.clone(), vec![9.0, 8.0, 7.0]];
    assert!(super::bayes::retained_response_draws(&columns, 2).is_none());
    // Fewer than two draws is not a retained posterior.
    assert!(super::bayes::retained_response_draws(&[vec![1.0]], 1).is_none());
}

#[test]
fn bayesian_response_refuses_an_ordinary_or_non_gaussian_likelihood_and_names_the_next_attempt() {
    let (data, graph) = synthetic_series(120);
    let (estimand, indexer) = identify(&graph, 1);
    let query = surface_query(&[-1.0, 1.0], vec![1]);
    for likelihood in [
        antecedent_prob::BayesLikelihood::BernoulliLogit,
        antecedent_prob::BayesLikelihood::PoissonLog,
    ] {
        let estimator = crate::BayesianGComputationAte {
            likelihood,
            ..crate::BayesianGComputationAte::conjugate()
        };
        let err = TemporalResponseEstimator::new()
            .estimate_bayesian(
                &data,
                TemporalIdentification {
                    horizons: &[(&estimand, &indexer)],
                    status: IdentificationStatus::NonparametricallyIdentified,
                    assumptions: AssumptionSet::new(),
                },
                &query,
                &estimator,
                &ExecutionContext::for_tests(7),
            )
            .unwrap_err();
        let text = err.to_string();
        assert!(text.contains("long-run-tempered generalized posterior"), "{text}");
        assert!(text.contains("serial dependence of the unfolded horizon residuals"), "{text}");
        assert!(text.contains("Next attempt"), "{text}");
        assert!(text.contains("GaussianIdentity"), "{text}");
    }
}

fn intervention_surface_query(horizons: Vec<u32>) -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(1),
        interventions: Arc::from(vec![Intervention::set(VariableId::from_raw(0), Value::f64(1.0))]),
    })
    .with_temporal(TemporalResponseSpec::new(horizons, TemporalPolicy::pulse(0), None).unwrap())
}

/// Two-horizon Bayesian intervention response on a noisy series: the cell means and, when
/// retained, the joint draws of the two cells.
fn two_horizon_bayesian(seed: u64) -> (Vec<f64>, Option<antecedent_core::CredibleDraws>) {
    let (data, graph) = series_with_noise(260, 1.0, seed);
    let (e1, i1) = identify(&graph, 1);
    let (e2, i2) = identify(&graph, 2);
    let response = TemporalResponseEstimator::new()
        .estimate_bayesian(
            &data,
            TemporalIdentification {
                horizons: &[(&e1, &i1), (&e2, &i2)],
                status: IdentificationStatus::NonparametricallyIdentified,
                assumptions: AssumptionSet::new(),
            },
            &intervention_surface_query(vec![1, 2]),
            &crate::BayesianGComputationAte::conjugate(),
            &ExecutionContext::for_tests(7),
        )
        .unwrap();
    let ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) =
        &response.estimate
    else {
        panic!("expected surface");
    };
    let draws = match response.uncertainty {
        ResponseUncertainty::PointwiseBand { draws, .. } => draws,
        _ => panic!("expected a pointwise band"),
    };
    (mean.to_vec(), draws)
}

fn sd_of(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    let m = values.iter().sum::<f64>() / n;
    (values.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (n - 1.0)).sqrt()
}

fn corr_of(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let cov: f64 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum();
    let va: f64 = a.iter().map(|x| (x - ma) * (x - ma)).sum();
    let vb: f64 = b.iter().map(|y| (y - mb) * (y - mb)).sum();
    cov / (va * vb).sqrt()
}

#[test]
fn a_nonlinear_cross_horizon_functional_of_the_joint_draws_matches_repeated_simulation() {
    // Independent reference: the sampling spread of the same functional of the two cell
    // estimates over fresh simulated series from one data-generating process.
    let ratio = |a: f64, b: f64| b / a;
    let (mut ref_ratio, mut ref_first, mut ref_second) = (Vec::new(), Vec::new(), Vec::new());
    for seed in 1..=60 {
        let (mean, _) = two_horizon_bayesian(1_000 + seed);
        ref_ratio.push(ratio(mean[0], mean[1]));
        ref_first.push(mean[0]);
        ref_second.push(mean[1]);
    }
    let reference_sd = sd_of(&ref_ratio);
    let reference_corr = corr_of(&ref_first, &ref_second);

    // One observed series: the joint draws are retained for both horizons together.
    let (mean, draws) = two_horizon_bayesian(5);
    let draws = draws.expect("a joint trajectory law retains its draws for a functional");
    assert_eq!(draws.n_coordinates(), 2);
    let (c1, c2) = (draws.column(0).unwrap(), draws.column(1).unwrap());
    let joint: Vec<f64> = c1.iter().zip(c2).map(|(a, b)| ratio(*a, *b)).collect();
    let joint_sd = sd_of(&joint);
    assert!(
        joint_sd / reference_sd > 0.5 && joint_sd / reference_sd < 1.8,
        "joint sd {joint_sd} vs simulation sd {reference_sd}"
    );
    // The cell draws carry the dependence the simulation shows.
    let joint_corr = corr_of(c1, c2);
    assert!(
        (joint_corr - reference_corr).abs() < 0.3,
        "joint correlation {joint_corr} vs simulation {reference_corr}"
    );
    // Pairing the same marginals independently (a rotated index) is the product of marginals;
    // where the simulation shows real dependence it is farther from the reference.
    let half = c2.len() / 2;
    let rotated: Vec<f64> =
        c1.iter().zip(c2.iter().cycle().skip(half)).map(|(a, b)| ratio(*a, *b)).collect();
    if reference_corr.abs() > 0.4 {
        let miss = |sd: f64| (sd / reference_sd).ln().abs();
        assert!(miss(joint_sd) < miss(sd_of(&rotated)), "joint must beat the product of marginals");
    }
    assert!(mean.iter().all(|m| m.is_finite()));
}
