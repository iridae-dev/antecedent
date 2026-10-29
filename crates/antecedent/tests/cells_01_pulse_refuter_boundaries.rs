// SPDX-License-Identifier: MIT OR Apache-2.0
//! Input boundaries of the pulse and of the refuter suite through the facade: a pulse on a static
//! graph class, a pulse on a categorical treatment column, and refuters on degenerate designs.
//!
//! Known answers: a pulse is indexed by a lag and a horizon, so every static graph class refuses
//! it as `not_applicable`; a categorical-typed treatment is not a real-valued lagged dose and is
//! refused as not float64; a constant treatment or an adjustment column that is an exact affine
//! copy of another leaves the design rank deficient and is refused under both refuter suites; a
//! constant outcome has effect zero to rounding, and the e-value refuter then does not pass.
use antecedent::{RefuteSuite, Study};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ExecutionContext, Lag, TemporalEffectQuery, VariableId,
};
use antecedent_data::{TabularData, TimeSeriesData};
use antecedent_graph::{Admg, Cpdag, Dag, DenseNodeId, Pag, TemporalDag, ensure_lagged};
use std::sync::Arc;

const N: usize = 160;

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn wiggle(i: usize, rate: f64) -> f64 {
    (i as f64 * rate).sin()
}

fn series_columns() -> (Vec<f64>, Vec<f64>) {
    let t: Vec<f64> = (0..N).map(|i| wiggle(i, 0.7)).collect();
    let mut y = vec![0.0; N];
    for i in 1..N {
        y[i] = 2.0 * t[i - 1] + 0.1 * wiggle(i, 0.13);
    }
    (t, y)
}

fn series(t: &[f64], y: &[f64]) -> TimeSeriesData {
    TimeSeriesData::from_f64_columns([("t", t), ("y", y)], 1).unwrap()
}

fn temporal_graph() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let t1 = ensure_lagged(&mut graph, v(0), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut graph, v(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(t1, y0).unwrap();
    graph
}

fn pulse_query() -> CausalQuery {
    CausalQuery::TemporalEffect(TemporalEffectQuery::pulse(v(0), v(1), 1.0).with_horizon_steps(2))
}

#[test]
fn cells_01_a_pulse_on_every_static_graph_class_is_not_applicable() {
    let (t, y) = series_columns();
    let node = DenseNodeId::from_raw;
    for class in ["dag", "admg", "cpdag", "pag"] {
        let builder = Study::series(series(&t, &y)).query(pulse_query());
        let builder = match class {
            "dag" => {
                let mut graph = Dag::with_variables(2);
                graph.insert_directed(node(0), node(1)).unwrap();
                builder.graph(graph)
            }
            "admg" => {
                let mut graph = Admg::with_variables(2);
                graph.insert_directed(node(0), node(1)).unwrap();
                builder.graph(graph)
            }
            "cpdag" => {
                let mut graph = Cpdag::with_variables(2);
                graph.insert_directed(node(0), node(1)).unwrap();
                builder.graph(graph)
            }
            _ => builder.graph(Pag::with_variables(2)),
        };
        let error = builder
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .map(|_| ())
            .expect_err(&format!("{class}: a pulse needs a time index"))
            .to_string();
        assert!(error.contains("not_applicable"), "{class}: {error}");
        assert!(
            error.contains("Temporal contrast queries require a temporal graph class"),
            "{class}: {error}"
        );
    }
    // The same series and query on a temporal graph run, so the refusal is the graph class's.
    let ctx = ExecutionContext::for_tests(5);
    let result = Study::series(series(&t, &y))
        .graph(temporal_graph())
        .query(pulse_query())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    assert!((result.estimate().as_effect().unwrap().ate - 2.0).abs() < 0.01);
}

fn categorical_series(y: &[f64]) -> TimeSeriesData {
    use antecedent_core::{
        CategoryDomainId, CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
    };
    use antecedent_data::{
        CategoricalColumn, CategoryCode, CategoryDomain, CategoryLevel, Float64Column, OwnedColumn,
        OwnedColumnarStorage, SamplingRegularity, TimeIndex, UnknownCategoryPolicy, ValidityBitmap,
    };
    let domain = CategoryDomain::try_new(
        CategoryDomainId::from_raw(2),
        ["a", "b", "c"].map(|label| CategoryLevel { label: Arc::from(label) }).to_vec(),
        false,
        Some(CategoryCode::from_raw(0)),
        UnknownCategoryPolicy::Fail,
    )
    .unwrap();
    let mut builder = CausalSchemaBuilder::new();
    builder
        .add_variable(
            "t",
            ValueType::Categorical,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
            None,
            Some(CategoryDomainId::from_raw(2)),
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
    let codes: Vec<CategoryCode> = (0..N).map(|i| CategoryCode::from_raw((i % 3) as u32)).collect();
    let columns = vec![
        OwnedColumn::Categorical(
            CategoricalColumn::try_new(v(0), codes, ValidityBitmap::all_valid(N), Arc::new(domain))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(v(1), Arc::<[f64]>::from(y), ValidityBitmap::all_valid(N)).unwrap(),
        ),
    ];
    let storage =
        OwnedColumnarStorage::try_new(builder.build().unwrap(), columns, None, None).unwrap();
    TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: N },
    )
    .unwrap()
}

#[test]
fn cells_01_a_categorical_treatment_is_refused_by_the_pulse_as_not_a_real_dose() {
    let (_, y) = series_columns();
    let error = Study::series(categorical_series(&y))
        .graph(temporal_graph())
        .query(pulse_query())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .map_err(|e| e.to_string())
        .and_then(|study| study.run(&ExecutionContext::for_tests(5)).map_err(|e| e.to_string()))
        .expect_err("a categorical treatment must not be read as a numeric dose");
    assert!(error.contains("not of type float64"), "{error}");
}

fn tabular_refuter_run(
    t: &[f64],
    y: &[f64],
    z: &[f64],
    z2: Option<&[f64]>,
    suite: RefuteSuite,
) -> Result<antecedent::StudyResult, String> {
    let node = DenseNodeId::from_raw;
    let mut columns = vec![("t", t), ("y", y), ("z", z)];
    let mut graph = Dag::with_variables(if z2.is_some() { 4 } else { 3 });
    for (from, to) in [(2_u32, 0_u32), (2, 1), (0, 1)] {
        graph.insert_directed(node(from), node(to)).unwrap();
    }
    if let Some(z2) = z2 {
        columns.push(("z2", z2));
        for (from, to) in [(3_u32, 0_u32), (3, 1)] {
            graph.insert_directed(node(from), node(to)).unwrap();
        }
    }
    Study::tabular(TabularData::from_f64_columns(columns).unwrap())
        .graph(graph)
        .query(AverageEffectQuery::with_levels(v(0), v(1), 0.0, 1.0))
        .refute(suite)
        .bootstrap_replicates(0)
        .build()
        .map_err(|e| e.to_string())
        .and_then(|study| study.run(&ExecutionContext::for_tests(5)).map_err(|e| e.to_string()))
}

#[test]
fn cells_01_refuters_refuse_rank_deficient_designs_and_a_constant_outcome_has_zero_effect() {
    let t: Vec<f64> = (0..N).map(|i| (i % 2) as f64).collect();
    let z: Vec<f64> = (0..N).map(|i| wiggle(i, 0.31)).collect();
    let y: Vec<f64> = (0..N).map(|i| 2.0 * t[i] + z[i] + 0.1 * wiggle(i, 0.13)).collect();
    let affine_copy: Vec<f64> = z.iter().map(|x| 2.0 * x + 1.0).collect();
    for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
        let sound = tabular_refuter_run(&t, &y, &z, None, suite).unwrap();
        assert!((sound.estimate().as_effect().unwrap().ate - 2.0).abs() < 0.05);
        let constant_treatment = vec![1.0; N];
        let error = tabular_refuter_run(&constant_treatment, &y, &z, None, suite)
            .map(|_| ())
            .expect_err("a constant treatment leaves the design rank deficient");
        assert!(error.contains("rank deficient: rank=2 ncols=3"), "{suite:?}: {error}");
        let error = tabular_refuter_run(&t, &y, &z, Some(&affine_copy), suite)
            .map(|_| ())
            .expect_err("an exactly collinear adjustment column leaves the design rank deficient");
        assert!(error.contains("rank deficient: rank=3 ncols=4"), "{suite:?}: {error}");
        let constant_outcome = vec![3.0; N];
        let flat = tabular_refuter_run(&t, &constant_outcome, &z, None, suite).unwrap();
        assert!(flat.estimate().as_effect().unwrap().ate.abs() < 1e-12);
        let evalue = flat
            .refutations
            .iter()
            .find(|report| &*report.refuter == "sensitivity.evalue")
            .expect("the e-value refuter ran");
        assert!(!evalue.passed, "{suite:?}: an effect of exactly zero cannot pass the e-value");
    }
}

fn pulse_at(magnitude: f64) -> Result<antecedent::StudyResult, String> {
    pulse_with(magnitude, &ExecutionContext::for_tests(5))
}

fn pulse_with(magnitude: f64, ctx: &ExecutionContext) -> Result<antecedent::StudyResult, String> {
    let (t, y) = series_columns();
    Study::series(series(&t, &y))
        .graph(temporal_graph())
        .query(CausalQuery::TemporalEffect(
            TemporalEffectQuery::pulse(v(0), v(1), magnitude).with_horizon_steps(2),
        ))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .map_err(|e| e.to_string())
        .and_then(|study| study.run(ctx).map_err(|e| e.to_string()))
}

#[test]
fn cells_01_a_pulse_far_outside_the_observed_range_is_flagged_then_refused_beyond_the_bound() {
    // The treatment is observed on [-1, 1] (about), the lag response is exactly 2.
    for magnitude in [0.5_f64, 0.9] {
        let result = pulse_at(magnitude).unwrap();
        assert_eq!(result.support_diagnostics().filter(|d| is_pulse_extrapolation(d)).count(), 0);
    }
    // A level 3 observed ranges beyond the interval is computed and flagged with its measures.
    let result = pulse_at(7.0).unwrap();
    let flags: Vec<_> =
        result.support_diagnostics().filter(|d| is_pulse_extrapolation(d)).collect();
    assert_eq!(flags.len(), 1);
    let field = |name: &str| -> f64 {
        flags[0].fields.iter().find(|(k, _)| &**k == name).unwrap().1.parse().unwrap()
    };
    let (lo, hi) = (field("observed_min"), field("observed_max"));
    assert!((field("active_extrapolation_ranges") - (7.0 - hi) / (hi - lo)).abs() < 1e-12);
    assert!(field("active_distance_sd") > 5.0);
    assert_eq!(field("extrapolation_bound_ranges"), 10.0);
    // Magnitude 1000 is 500 ranges out: refused with the typed code and a next attempt.
    let error = pulse_at(1000.0).expect_err("a pulse 500 ranges outside the data is refused");
    assert!(error.starts_with("reason=positivity_violated"), "{error}");
    assert!(error.contains("Next attempt"), "{error}");
    // The bound is configurable: widened it computes, tightened it refuses.
    let wide = ExecutionContext::for_tests(5).with_pulse_extrapolation_ranges(1000.0).unwrap();
    assert!(pulse_with(1000.0, &wide).is_ok());
    let tight = ExecutionContext::for_tests(5).with_pulse_extrapolation_ranges(1.0).unwrap();
    assert!(pulse_with(7.0, &tight).is_err());
    assert!(ExecutionContext::for_tests(5).with_pulse_extrapolation_ranges(f64::NAN).is_none());
}

#[test]
fn cells_01_a_pulse_with_its_control_level_outside_the_observed_range_is_not_refused() {
    // Treatment observed on about [99, 101]; control 0 is ~50 ranges below, the active level
    // 100 is inside: the contrast is written as a pulse from zero and is accepted.
    let n = N;
    let t: Vec<f64> = (0..n).map(|i| 100.0 + wiggle(i, 0.7)).collect();
    let mut y = vec![0.0; n];
    for i in 1..n {
        y[i] = 2.0 * t[i - 1] + 0.1 * wiggle(i, 0.13);
    }
    let result = Study::series(series(&t, &y))
        .graph(temporal_graph())
        .query(CausalQuery::TemporalEffect(
            TemporalEffectQuery::pulse(v(0), v(1), 100.0).with_horizon_steps(2),
        ))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(5))
        .unwrap();
    let flags: Vec<_> =
        result.support_diagnostics().filter(|d| is_pulse_extrapolation(d)).collect();
    assert_eq!(flags.len(), 1, "the control outside the range is reported, not refused");
    assert!(flags[0].message.contains("control"), "{}", flags[0].message);
}

fn is_pulse_extrapolation(d: &antecedent_core::Diagnostic) -> bool {
    &*d.code == "estimate.temporal.pulse.level_extrapolation"
}
