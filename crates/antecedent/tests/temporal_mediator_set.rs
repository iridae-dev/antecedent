//! Temporal mediation through a mediator set on a fixed `TemporalDag`: the mediated effect is
//! the sum of the path products over the ordered mediators, and total = direct + mediated.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, CausalSchemaBuilder, ExecutionContext, Lag, MeasurementSpec, MediationContrast,
    MediationQuery, RoleHint, SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
    TimeSeriesData, ValidityBitmap,
};
use antecedent_graph::{TemporalDag, ensure_lagged};

const N: usize = 361;
/// `a₁ = 0.8`, `a₂ = 0.3 + 0.5 · 0.8 = 0.7`, `b = (0.4, 0.6)`.
const MEDIATED_TRUTH: f64 = 0.8 * 0.4 + 0.7 * 0.6;
const DIRECT_TRUTH: f64 = 0.25;

/// `t → m1 → m2 → y` with `t → m2`, `t → y` and `m1 → y`, confounded by `z`.
fn series() -> TimeSeriesData {
    let mut schema = CausalSchemaBuilder::new();
    for (name, hint) in [
        ("t", RoleHint::TreatmentCandidate),
        ("m1", RoleHint::Context),
        ("m2", RoleHint::Context),
        ("y", RoleHint::OutcomeCandidate),
        ("z", RoleHint::Context),
    ] {
        schema
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
    let schema = schema.build().unwrap();
    let z: Vec<f64> = (0..N).map(|i| if i % 4 == 0 || i % 4 == 1 { 1.0 } else { -1.0 }).collect();
    let t: Vec<f64> =
        (0..N).map(|i| z[i] + if i % 4 == 0 || i % 4 == 2 { 1.0 } else { -1.0 }).collect();
    let (mut m1, mut m2) = (vec![0.0; N], vec![0.0; N]);
    let mut y = vec![1.0; N];
    for i in 1..N {
        let first_noise = (i % 5) as f64 - 2.0;
        let second_noise = ((3 * i) % 7) as f64 - 3.0;
        m1[i] = 0.8 * t[i - 1] + 0.6 * z[i - 1] + 0.15 * first_noise;
        m2[i] = 0.5 * m1[i] + 0.3 * t[i - 1] + 0.3 * z[i - 1] + 0.12 * second_noise;
        y[i] = 1.0 + DIRECT_TRUTH * t[i - 1] + 0.4 * m1[i] + 0.6 * m2[i] + 5.0 * z[i - 1];
    }
    let columns = [t, m1, m2, y, z]
        .into_iter()
        .enumerate()
        .map(|(id, values)| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(u32::try_from(id).unwrap()),
                    Arc::from(values),
                    ValidityBitmap::all_valid(N),
                )
                .unwrap(),
            )
        })
        .collect();
    let storage = OwnedColumnarStorage::try_new(schema, columns, None, None).unwrap();
    TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: N },
    )
    .unwrap()
}

fn graph() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let t0 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
    let t1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let m1 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let m2 = ensure_lagged(&mut graph, VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    let y0 = ensure_lagged(&mut graph, VariableId::from_raw(3), Lag::CONTEMPORANEOUS).unwrap();
    let z0 = ensure_lagged(&mut graph, VariableId::from_raw(4), Lag::CONTEMPORANEOUS).unwrap();
    let z1 = ensure_lagged(&mut graph, VariableId::from_raw(4), Lag::from_raw(1)).unwrap();
    for (from, to) in [
        (z0, t0),
        (z1, y0),
        (z1, m1),
        (z1, m2),
        (t1, y0),
        (t1, m1),
        (t1, m2),
        (m1, m2),
        (m1, y0),
        (m2, y0),
    ] {
        graph.insert_directed(from, to).unwrap();
    }
    graph
}

fn query(contrast: MediationContrast) -> MediationQuery {
    MediationQuery::binary(
        VariableId::from_raw(0),
        VariableId::from_raw(3),
        [VariableId::from_raw(1), VariableId::from_raw(2)],
        contrast,
    )
}

fn estimate(
    contrast: MediationContrast,
    inference: InferenceMode,
    suite: RefuteSuite,
) -> antecedent::StudyResult {
    let context = ExecutionContext::for_tests(821);
    Study::series(series())
        .graph(graph())
        .query(CausalQuery::Mediation(query(contrast)))
        .inference(inference)
        .refute(suite)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&context)
        .unwrap()
        .estimate_retained(&context)
        .unwrap()
}

#[test]
fn a_mediator_set_sums_its_path_products_and_decomposes_the_total() {
    let mediated =
        estimate(MediationContrast::Mediated, InferenceMode::Frequentist, RefuteSuite::None);
    let direct = estimate(MediationContrast::Direct, InferenceMode::Frequentist, RefuteSuite::None);
    let total = estimate(MediationContrast::Total, InferenceMode::Frequentist, RefuteSuite::None);
    let ate = |result: &antecedent::StudyResult| result.estimate().as_effect().unwrap().ate;
    assert!((ate(&mediated) - MEDIATED_TRUTH).abs() < 0.006, "{}", ate(&mediated));
    assert!((ate(&direct) - DIRECT_TRUTH).abs() < 1e-9, "{}", ate(&direct));
    // The three contrasts come from one set of regressions, so they decompose exactly.
    assert!((ate(&total) - (ate(&direct) + ate(&mediated))).abs() < 1e-9);
    assert_eq!(mediated.estimand.mediators.len(), 2);
}

#[test]
fn a_mediator_set_runs_the_bayesian_mechanisms_and_the_mediation_refuters() {
    let result = estimate(
        MediationContrast::Mediated,
        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(1024).prior_scale(1_000.0)),
        RefuteSuite::Cheap,
    );
    let posterior = result.posterior.as_ref().expect("a mediation posterior");
    let mean = posterior.summaries.mean[0];
    assert!((mean - MEDIATED_TRUTH).abs() < 0.008, "{mean}");
    assert!(!result.refutations.is_empty());
}
