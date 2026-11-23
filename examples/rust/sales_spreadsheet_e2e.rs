#![allow(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::manual_map,
    clippy::match_wildcard_for_single_variants,
    clippy::doc_markdown,
    clippy::map_unwrap_or
)]
#![allow(
    clippy::cast_possible_truncation,
    reason = "test scaffolding compares exact constants and indexes with small literals"
)]
//! One accepted graph, several questions: a Bayesian effect, a path decomposition, unit-level
//! effects and a temporal pulse.
//!
//! Question: what is the effect of a campaign on revenue, how much of it passes through a
//! mediator, what is it for each unit, and what does a promotion pulse do one step later?
//!
//! Data and truth: 400 static rows from `t = 0.7 z + e`, `m = 0.6 t + 0.3 z + e`,
//! `y = 1.2 t + 0.8 m + 0.5 z + e`, so a unit change in the campaign `t` moves revenue `y` by
//! `1.2 + 0.8 * 0.6 = 1.68`, of which 0.48 passes through `m`. The temporal series has 350
//! points with `returns[t] = 0.85 promo[t - 1]` plus noise.
//!
//! Steps: a Laplace-approximated Bayesian average effect on the accepted DAG; a path-specific
//! decomposition as an `Attribute` query; unit effects as a `Simulate` query; a second,
//! conjugate Bayesian estimate on the same graph; and a Bayesian pulse effect on a `TemporalDag`.
//! The path decomposition and the unit effects are compiled study routes on the same DAG.
//!
//! Asserted: every static result is finite, a posterior is returned, and 400 unit effects come
//! back; the temporal pulse effect is within 0.25 of 0.85.
//!
//! What this establishes: one supplied graph serves each query without a structure search. Only
//! the temporal estimate is compared with its truth; the static estimates are checked for
//! shape, not accuracy.
//!
//! Run: `cargo run -p antecedent --example sales_spreadsheet_e2e`
//! Source: `examples/rust/sales_spreadsheet_e2e.rs`

use std::sync::Arc;

use antecedent::prelude::*;
use antecedent::{AttributionOutcome, CompiledExecution, RefuteSuite, SimulationOutcome};
use antecedent_core::{
    AttributeQuery, AttributeTarget, CausalQuery, CausalRng, Lag, MeasurementSpec, RoleHint,
    SimulateQuery, SimulationTarget, SmallRoleSet, TemporalPolicy, ValueType,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex, ValidityBitmap,
};
use antecedent_graph::{DenseNodeId, ensure_lagged};

fn sales_static(n: usize, seed: u64) -> (TabularData, Dag) {
    let mut rng = CausalRng::from_seed(seed);
    let mut gauss = || {
        let u1 = rng.next_f64().max(1e-12);
        let u2 = rng.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };
    let mut z = vec![0.0; n];
    let mut t = vec![0.0; n];
    let mut m = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        let zi = gauss();
        let ti = 0.7 * zi + gauss();
        let mi = 0.6 * ti + 0.3 * zi + 0.2 * gauss();
        let yi = 1.2 * ti + 0.8 * mi + 0.5 * zi + 0.3 * gauss();
        z[i] = zi;
        t[i] = ti;
        m[i] = mi;
        y[i] = yi;
    }

    let mut b = CausalSchemaBuilder::new();
    for (name, role) in [
        ("z", RoleHint::Context),
        ("t", RoleHint::TreatmentCandidate),
        ("m", RoleHint::Context),
        ("y", RoleHint::OutcomeCandidate),
    ] {
        b.add_variable(
            name,
            ValueType::Continuous,
            SmallRoleSet::from_hint(role),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(z), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(t), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(2), Arc::from(m), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(3), Arc::from(y), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let mut dag = Dag::with_variables(4);
    // z→t, z→m, z→y, t→m, t→y, m→y
    for (a, b) in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] {
        dag.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    (TabularData::new(storage), dag)
}

fn sales_temporal(n: usize, seed: u64) -> (TimeSeriesData, TemporalDag, TemporalEffectQuery) {
    let mut rng = CausalRng::from_seed(seed);
    let mut gauss = || {
        let u1 = rng.next_f64().max(1e-12);
        let u2 = rng.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        x[i] = ((i as f64) * 0.04).sin() + 0.1 * gauss();
        if i > 0 {
            y[i] = 0.85 * x[i - 1] + 0.05 * gauss();
        }
    }

    let mut b = CausalSchemaBuilder::new();
    b.add_variable(
        "promo",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "returns",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    let schema = b.build().unwrap();
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(x), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(y), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let series = TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap();

    let mut g = TemporalDag::empty();
    let p1 = ensure_lagged(&mut g, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let r0 = ensure_lagged(&mut g, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_directed(p1, r0).unwrap();
    let q = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1);
    (series, g, q)
}

/// Runs the example end to end; `main` calls it and the example test suite runs it.
pub fn run() -> Result<(), CausalError> {
    let (data, dag) = sales_static(400, 7);
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(1), VariableId::from_raw(3));

    let bayes = Study::tabular(data.clone())
        .graph(dag.clone())
        .query(query.clone())
        .inference(InferenceMode::Bayesian(BayesianConfig::laplace().n_draws(128)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()?
        .run(&ExecutionContext::for_tests(3))?;
    assert!(
        bayes.estimate().as_effect().unwrap().ate.is_finite(),
        "ate={}",
        bayes.estimate().as_effect().unwrap().ate
    );
    assert!(bayes.posterior.is_some());
    println!("Bayesian ATE={:.4} (campaign → revenue)", bayes.estimate().as_effect().unwrap().ate);

    // Path-specific decomposition of campaign → revenue through the mediator.
    let path_query = CausalQuery::Attribute(AttributeQuery::new(AttributeTarget::Paths {
        sources: Arc::from([VariableId::from_raw(1)]),
        outcome: VariableId::from_raw(3),
        max_paths: 16,
        max_len: 4,
        through: Arc::from([VariableId::from_raw(2)]),
    }));
    let path_study = Study::tabular(data.clone()).graph(dag.clone()).query(path_query).build()?;
    let path_run = CompiledExecution::run(&path_study, &ExecutionContext::for_tests(5))?;
    let Some(AttributionOutcome::Change(path)) =
        path_run.result().attribution().map(|payload| payload.outcome.clone())
    else {
        return Err(CausalError::Compile {
            message: "a paths target returns a change attribution".into(),
        });
    };
    assert!(path.total_change.is_finite());
    println!(
        "Path decompose total_change={:.4} paths={}",
        path.total_change,
        path.path_breakdown.len()
    );

    // Unit-level effects of the campaign on revenue.
    let ite_query = CausalQuery::Simulate(SimulateQuery::new(SimulationTarget::UnitEffects {
        treatment: VariableId::from_raw(1),
        outcome: VariableId::from_raw(3),
        active: 1.0,
        control: 0.0,
    }));
    let ite_study = Study::tabular(data.clone()).graph(dag.clone()).query(ite_query).build()?;
    let ite_run = CompiledExecution::run(&ite_study, &ExecutionContext::for_tests(7))?;
    let Some(SimulationOutcome::UnitEffects(ite)) =
        ite_run.result().simulation().map(|payload| payload.outcome.clone())
    else {
        return Err(CausalError::Compile {
            message: "a unit-effects target returns the effects".into(),
        });
    };
    assert_eq!(ite.unit_effects.len(), 400);
    assert!(ite.mean_ite.is_finite());
    println!("ITE mean={:.4} n={}", ite.mean_ite, ite.unit_effects.len());

    // A second estimate on the supplied graph, with the conjugate posterior. A supplied graph
    // never triggers a structure search.
    let _ = Study::tabular(data)
        .graph(dag)
        .query(query)
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()?
        .run(&ExecutionContext::for_tests(4))?;

    let (series, tdag, pulse_q) = sales_temporal(350, 11);
    let pulse = Study::series(series)
        .graph(tdag)
        .temporal_query(pulse_q)
        .inference(InferenceMode::Bayesian(BayesianConfig::laplace().n_draws(96)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()?
        .run(&ExecutionContext::for_tests(13))?;
    assert!(
        pulse.estimate().as_effect().unwrap().ate.is_finite(),
        "pulse={}",
        pulse.estimate().as_effect().unwrap().ate
    );
    println!(
        "Temporal pulse Bayesian ATE={:.4} (promo → returns)",
        pulse.estimate().as_effect().unwrap().ate
    );
    assert!(
        (pulse.estimate().as_effect().unwrap().ate - 0.85).abs() < 0.25,
        "pulse={}",
        pulse.estimate().as_effect().unwrap().ate
    );

    println!("sales_spreadsheet_e2e: ok");
    Ok(())
}

fn main() -> Result<(), CausalError> {
    run()
}
