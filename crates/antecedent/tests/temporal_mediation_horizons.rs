//! Pin: TemporalMediation I(1) ≠ I(2) on a confounded pulse-style DGP.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::float_cmp)]

use std::sync::Arc;

use antecedent::estimate::TemporalMediationEstimator;
use antecedent::identify::TemporalMediationIdentifier;
use antecedent::{AcceptedGraph, BayesianConfig, InferenceMode, PreparedStudy, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, CausalSchemaBuilder, ExecutionContext, Lag, MeasurementSpec, MediationContrast,
    MediationQuery, RoleHint, SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, LaggedColumn, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
    TimeSeriesData, ValidityBitmap,
};
use antecedent_graph::{TemporalDag, ensure_lagged};
use antecedent_identify::TemporalIdentificationResult;

const N: usize = 241;
const A: f64 = 0.8;
const B: f64 = 0.55;
const C_PRIME: f64 = 0.25;
const STRUCTURAL_MEDIATED: f64 = A * B;

fn confounded_mediation_series() -> (TimeSeriesData, TemporalDag) {
    let mut builder = CausalSchemaBuilder::new();
    for (name, hint) in [
        ("t", RoleHint::TreatmentCandidate),
        ("m", RoleHint::Context),
        ("y", RoleHint::OutcomeCandidate),
        ("z", RoleHint::Context),
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
    // Same period-4 (Z, U) as temporal_confounded_pulse, plus T→M→Y.
    let z: Vec<f64> = (0..N).map(|i| if i % 4 == 0 || i % 4 == 1 { 1.0 } else { -1.0 }).collect();
    let t: Vec<f64> =
        (0..N).map(|i| z[i] + if i % 4 == 0 || i % 4 == 2 { 1.0 } else { -1.0 }).collect();
    let mut m = vec![0.0; N];
    let mut y = vec![1.0; N];
    for i in 1..N {
        let e = (i % 5) as f64 - 2.0;
        // Z → M as well as Z → Y: omitting Z@-1 biases the path product, not
        // only the direct/total slopes (M ⊥ Z | T would leave a*b intact).
        m[i] = A * t[i - 1] + 0.6 * z[i - 1] + 0.15 * e;
        y[i] = 1.0 + C_PRIME * t[i - 1] + B * m[i] + 5.0 * z[i - 1];
    }
    let cols = vec![col(0, t), col(1, m), col(2, y), col(3, z)];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let data = TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: N },
    )
    .unwrap();
    let mut g = TemporalDag::empty();
    let t0 = ensure_lagged(&mut g, VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
    let t1 = ensure_lagged(&mut g, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let m0 = ensure_lagged(&mut g, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let y0 = ensure_lagged(&mut g, VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    let z0 = ensure_lagged(&mut g, VariableId::from_raw(3), Lag::CONTEMPORANEOUS).unwrap();
    let z1 = ensure_lagged(&mut g, VariableId::from_raw(3), Lag::from_raw(1)).unwrap();
    g.insert_directed(z0, t0).unwrap();
    g.insert_directed(z1, y0).unwrap();
    g.insert_directed(z1, m0).unwrap();
    g.insert_directed(t1, y0).unwrap();
    g.insert_directed(t1, m0).unwrap();
    g.insert_directed(m0, y0).unwrap();
    (data, g)
}

fn col(id: u32, values: Vec<f64>) -> OwnedColumn {
    OwnedColumn::Float64(
        Float64Column::new(
            VariableId::from_raw(id),
            Arc::from(values),
            ValidityBitmap::all_valid(N),
        )
        .unwrap(),
    )
}

fn query(horizons: &[u32]) -> MediationQuery {
    MediationQuery::binary(
        VariableId::from_raw(0),
        VariableId::from_raw(2),
        [VariableId::from_raw(1)],
        MediationContrast::Mediated,
    )
    .with_horizons(horizons.to_vec())
    .unwrap()
}

fn named_z(temporal: &TemporalIdentificationResult) -> Vec<(u32, i32)> {
    temporal
        .result
        .estimands
        .first()
        .map(|e| {
            e.adjustment_set
                .iter()
                .filter_map(|&dense| {
                    let key = temporal.indexer.key_of(dense.raw()).ok()?;
                    Some((key.variable.raw(), key.offset))
                })
                .collect()
        })
        .unwrap_or_default()
}

fn estimate_h1(data: &TimeSeriesData, adjustment: &[LaggedColumn]) -> f64 {
    let q = query(&[1]);
    let id = TemporalMediationIdentifier::new()
        .identify_with_horizon(&confounded_mediation_series().1, &q, 1)
        .unwrap()
        .0;
    TemporalMediationEstimator::new()
        .with_allow_natural_controlled_alias(true)
        .estimate_with_adjustment(
            data,
            &id.estimands[0],
            &q,
            adjustment,
            &[],
            &ExecutionContext::for_tests(7),
        )
        .unwrap()
        .effect
        .ate
}

fn has_cached(result: &antecedent::StudyResult) -> bool {
    result.diagnostics.iter().any(|d| d.code.as_ref() == "exec.identify.cached")
}

fn prepare_click(
    data: &TimeSeriesData,
    graph: &TemporalDag,
    q: &MediationQuery,
    inference: InferenceMode,
    refute: RefuteSuite,
    accepted: bool,
) -> (PreparedStudy, antecedent::StudyResult) {
    let ctx = ExecutionContext::for_tests(7);
    let builder = Study::series(data.clone());
    let builder = if accepted {
        builder.graph(AcceptedGraph::temporal_dag(graph.clone()))
    } else {
        builder.graph(graph.clone())
    };
    let prepared = builder
        .query(CausalQuery::Mediation(q.clone()))
        .inference(inference)
        .refute(refute)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let click = prepared.estimate_series(data, &ctx).unwrap();
    (prepared, click)
}

#[test]
fn temporal_mediation_i1_not_equal_i2_on_confounded_pulse() {
    let (data, graph) = confounded_mediation_series();
    let q = query(&[1, 2]);
    let ider = TemporalMediationIdentifier::new();
    let (id1, t1) = ider.identify_with_horizon(&graph, &q, 1).unwrap();
    let (id2, t2) = ider.identify_with_horizon(&graph, &q, 2).unwrap();
    let z1 = named_z(&t1);
    let z2 = named_z(&t2);
    assert_eq!(z1, vec![(3, -1)], "I(1) must be Z@-1, got {z1:?}");
    assert!(z2.is_empty(), "I(2) must be empty, got {z2:?}");
    assert!(id1.estimands[0].method.as_ref().starts_with("temporal_mediation."));
    assert!(id2.estimands[0].method.as_ref().starts_with("temporal_mediation."));
    assert!(
        id1.derivation.steps.iter().any(|s| {
            s.rule.as_ref() == "temporal.backdoor.unfolded" && s.detail.contains("I(1)")
        })
    );

    let ctx = ExecutionContext::for_tests(7);
    let fresh = Study::series(data.clone())
        .graph(graph.clone())
        .query(CausalQuery::Mediation(q.clone()))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    assert!((fresh.estimate.ate - STRUCTURAL_MEDIATED).abs() < 1e-6);
    assert!(
        fresh
            .diagnostics
            .iter()
            .any(|d| { d.code.as_ref() == "identify.temporal_mediation.horizon_dependent" })
    );
    assert!(fresh.diagnostics.iter().all(|d| d.code.as_ref() != "exec.identify.cached"));

    let z1_cols = [LaggedColumn { variable: VariableId::from_raw(3), lag: Lag::from_raw(1) }];
    let under_i1 = estimate_h1(&data, &z1_cols);
    let under_i2 = estimate_h1(&data, &[]);
    assert!((under_i1 - STRUCTURAL_MEDIATED).abs() < 1e-6);
    assert!(
        (under_i2 - STRUCTURAL_MEDIATED).abs() > 0.2,
        "reusing I(2)={{}} at h=1 must recover the confounded association, got {under_i2}"
    );

    for (inference, refute, accepted) in [
        (InferenceMode::Frequentist, RefuteSuite::Cheap, false),
        (InferenceMode::Frequentist, RefuteSuite::Full, true),
        (
            InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(512)),
            RefuteSuite::Cheap,
            false,
        ),
    ] {
        let (_prepared, click) =
            prepare_click(&data, &graph, &q, inference.clone(), refute, accepted);
        assert!(has_cached(&click), "prepared click must reuse exec.identify.cached");
        match inference {
            InferenceMode::Frequentist => {
                assert!((click.estimate.ate - STRUCTURAL_MEDIATED).abs() < 1e-6);
            }
            InferenceMode::Bayesian(_) => {
                assert!(
                    (click.estimate.ate - STRUCTURAL_MEDIATED).abs() < 0.08,
                    "Bayesian I(1) ate {} vs structural {STRUCTURAL_MEDIATED}",
                    click.estimate.ate
                );
            }
        }
    }
}
