//! Production contexts evaluate the requested replicate count.
//!
//! The calibration gates certify intervals built from the full requested
//! bootstrap under `ExecutionContext::for_tests`. These tests pin that
//! `ExecutionContext::production` reports the same replicate accounting and,
//! for the static mediation SE, the same number, so what users get is what the
//! gates measured.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::float_cmp, clippy::many_single_char_names)]

mod common;

use std::sync::Arc;

use antecedent::{InferenceMode, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    AdaptiveBootstrapBudget, AdaptiveDrawBudget, AverageEffectQuery, CausalQuery, CausalRng,
    CausalSchemaBuilder, ExecutionContext, MeasurementSpec, MediationContrast, MediationQuery,
    RoleHint, SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
};
use antecedent_graph::{Dag, DenseNodeId};
use common::driven_dgp::{Scenario, pulse, pulse_dag, pulse_series};

const IID_160: Scenario = Scenario { label: "iid n=160", rho: 0.0, n: 160, seed: 100_000 };

fn gaussian(rng: &mut CausalRng) -> f64 {
    let u1 = rng.next_f64().max(1e-12);
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn table(names: &[&str], hints: &[RoleHint], cols: Vec<Vec<f64>>) -> TabularData {
    let n = cols[0].len();
    let mut b = CausalSchemaBuilder::new();
    for (name, hint) in names.iter().zip(hints) {
        b.add_variable(
            *name,
            ValueType::Continuous,
            SmallRoleSet::from_hint(*hint),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let cols = cols
        .into_iter()
        .enumerate()
        .map(|(i, c)| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(u32::try_from(i).unwrap()),
                    Arc::from(c),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            )
        })
        .collect();
    TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap())
}

/// Confounded linear SCM with structural ATE = 2 (`z → t`, `z → y`, `t → y`).
fn confounded_scm(n: usize, seed: u64) -> (TabularData, Dag, AverageEffectQuery) {
    let mut rng = CausalRng::from_seed(seed);
    let (mut t, mut y, mut z) =
        (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    for _ in 0..n {
        let zi = gaussian(&mut rng);
        let p = 1.0 / (1.0 + (0.4 - 0.9 * zi).exp());
        let ti = if rng.next_f64() < p { 1.0 } else { 0.0 };
        y.push(2.0 * ti + zi + 0.4 * gaussian(&mut rng));
        t.push(ti);
        z.push(zi);
    }
    let data = table(
        &["t", "y", "z"],
        &[RoleHint::TreatmentCandidate, RoleHint::OutcomeCandidate, RoleHint::Context],
        vec![t, y, z],
    );
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    (data, dag, query)
}

/// Noisy linear mediation SCM: `m = 0.8 a + ε`, `y = a + 0.6 m + ε'`
/// (`a → m`, `a → y`, `m → y`), so the bootstrap SE is a genuine spread.
fn mediation_scm(n: usize, seed: u64) -> (TabularData, Dag) {
    let mut rng = CausalRng::from_seed(seed);
    let (mut a, mut m, mut y) =
        (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    for _ in 0..n {
        let ai = if rng.next_f64() < 0.5 { 1.0 } else { 0.0 };
        let mi = 0.8 * ai + 0.5 * gaussian(&mut rng);
        y.push(ai + 0.6 * mi + 0.6 * gaussian(&mut rng));
        a.push(ai);
        m.push(mi);
    }
    let data = table(
        &["a", "m", "y"],
        &[RoleHint::TreatmentCandidate, RoleHint::Context, RoleHint::OutcomeCandidate],
        vec![a, m, y],
    );
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    (data, dag)
}

fn mediation_query(contrast: MediationContrast) -> CausalQuery {
    CausalQuery::Mediation(MediationQuery::binary(
        VariableId::from_raw(0),
        VariableId::from_raw(2),
        Arc::from([VariableId::from_raw(1)]),
        contrast,
    ))
}

fn static_run(
    data: &TabularData,
    dag: &Dag,
    query: CausalQuery,
    boot: u32,
    ctx: &ExecutionContext,
) -> StudyResult {
    Study::tabular(data.clone())
        .graph(dag.clone())
        .query(query)
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(boot)
        .build()
        .unwrap()
        .run(ctx)
        .unwrap()
}

fn assert_full(result: &StudyResult, requested: u32, label: &str) {
    assert_eq!(
        result.performance.bootstrap_replicates_requested,
        Some(requested),
        "{label}: requested"
    );
    assert_eq!(result.estimate.bootstrap_replicates_ok, Some(requested), "{label}: ok count");
    assert!(!result.performance.early_stopped, "{label}: early_stopped");
    assert!(!result.estimate.bootstrap_early_stopped, "{label}: estimate early_stopped");
    assert!(result.estimate.se_bootstrap.is_some_and(|se| se.is_finite() && se > 0.0), "{label}");
}

#[test]
fn production_context_has_no_early_stop_budgets() {
    let ctx = ExecutionContext::production(1, 2);
    assert_eq!(ctx.adaptive_bootstrap, AdaptiveBootstrapBudget::disabled());
    assert_eq!(ctx.adaptive_draws, AdaptiveDrawBudget::disabled());
    assert_eq!(ctx.adaptive_bootstrap, ExecutionContext::for_tests(1).adaptive_bootstrap);
    assert_eq!(ctx.adaptive_draws, ExecutionContext::for_tests(1).adaptive_draws);
}

#[test]
fn production_static_ate_evaluates_all_199_replicates() {
    let (data, dag, query) = confounded_scm(600, 7);
    let ctx = ExecutionContext::production(5, 1);
    let result = static_run(&data, &dag, CausalQuery::AverageEffect(query), 199, &ctx);
    assert_full(&result, 199, "static ATE");
}

#[test]
fn production_static_mediation_evaluates_all_199_replicates() {
    let (data, dag) = mediation_scm(500, 11);
    let ctx = ExecutionContext::production(5, 1);
    for contrast in [MediationContrast::Total, MediationContrast::Mediated] {
        let result = static_run(&data, &dag, mediation_query(contrast), 199, &ctx);
        assert_eq!(result.logical_plan.estimator.as_deref(), Some("mediation.linear"));
        assert_full(&result, 199, "static mediation");
    }
}

#[test]
fn production_temporal_pulse_evaluates_all_199_replicates() {
    let ctx = ExecutionContext::production(7, 1);
    let result = Study::series(pulse_series(IID_160, 7, false))
        .graph(pulse_dag(false))
        .query(CausalQuery::TemporalEffect(pulse(1)))
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(199)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    assert_full(&result, 199, "temporal pulse");
}

/// The mediation SE users get at B=800 is the SE the calibration harness
/// measures: production and `for_tests` agree on the replicate count and on
/// the number itself.
#[test]
fn production_static_mediation_se_matches_calibration_context_at_b800() {
    let (data, dag) = mediation_scm(500, 11);
    let production = ExecutionContext::production(5, 1);
    let calibration = ExecutionContext::for_tests(5);
    for contrast in [MediationContrast::Total, MediationContrast::Mediated] {
        let p = static_run(&data, &dag, mediation_query(contrast), 800, &production);
        let c = static_run(&data, &dag, mediation_query(contrast), 800, &calibration);
        assert_full(&p, 800, "production mediation");
        assert_full(&c, 800, "calibration mediation");
        let (se_p, se_c) = (p.estimate.se_bootstrap.unwrap(), c.estimate.se_bootstrap.unwrap());
        assert!(
            (se_p - se_c).abs() <= 1e-9 * se_c.abs().max(1.0),
            "{contrast:?}: production SE {se_p} vs calibration SE {se_c}"
        );
        assert_eq!(p.estimate.ate, c.estimate.ate);
    }
}
