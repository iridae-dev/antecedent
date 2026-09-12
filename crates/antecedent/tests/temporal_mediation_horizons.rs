//! Pin: `TemporalMediation` I(1) ≠ I(2) on a confounded pulse-style DGP.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::float_cmp)]

use std::sync::Arc;

use antecedent::estimate::TemporalMediationEstimator;
use antecedent::identify::TemporalMediationIdentifier;
use antecedent::{AcceptedGraph, BayesianConfig, InferenceMode, PreparedStudy, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, CausalSchemaBuilder, ExecutionContext, IdentificationStatus, Lag, MeasurementSpec,
    MediationContrast, MediationQuery, RoleHint, SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, LaggedColumn, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
    TimeSeriesData, ValidityBitmap,
};
use antecedent_graph::{TemporalCpdag, TemporalDag, TemporalPag, ensure_lagged};
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
    let confounder: Vec<f64> =
        (0..N).map(|i| if i % 4 == 0 || i % 4 == 1 { 1.0 } else { -1.0 }).collect();
    let treatment: Vec<f64> =
        (0..N).map(|i| confounder[i] + if i % 4 == 0 || i % 4 == 2 { 1.0 } else { -1.0 }).collect();
    let mut mediator = vec![0.0; N];
    let mut outcome = vec![1.0; N];
    for i in 1..N {
        let noise = (i % 5) as f64 - 2.0;
        // Z → M as well as Z → Y: omitting Z@-1 biases the path product, not
        // only the direct/total slopes (M ⊥ Z | T would leave a*b intact).
        mediator[i] = A * treatment[i - 1] + 0.6 * confounder[i - 1] + 0.15 * noise;
        outcome[i] = 1.0 + C_PRIME * treatment[i - 1] + B * mediator[i] + 5.0 * confounder[i - 1];
    }
    let cols = vec![col(0, treatment), col(1, mediator), col(2, outcome), col(3, confounder)];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let data = TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: N },
    )
    .unwrap();
    let mut graph = TemporalDag::empty();
    let t0 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
    let t1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let m0 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let y0 = ensure_lagged(&mut graph, VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    let z0 = ensure_lagged(&mut graph, VariableId::from_raw(3), Lag::CONTEMPORANEOUS).unwrap();
    let z1 = ensure_lagged(&mut graph, VariableId::from_raw(3), Lag::from_raw(1)).unwrap();
    graph.insert_directed(z0, t0).unwrap();
    graph.insert_directed(z1, y0).unwrap();
    graph.insert_directed(z1, m0).unwrap();
    graph.insert_directed(t1, y0).unwrap();
    graph.insert_directed(t1, m0).unwrap();
    graph.insert_directed(m0, y0).unwrap();
    (data, graph)
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
    let multi = Study::series(data.clone())
        .graph(graph.clone())
        .query(CausalQuery::Mediation(q.clone()))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    let grid = multi.mediation_grid.as_ref().expect("multi-horizon mediation grid");
    assert_eq!(grid.slices.len(), 2);
    assert!(!grid.joint_posterior);
    assert_eq!(grid.slices[0].horizon, 1);
    assert_eq!(grid.slices[1].horizon, 2);
    assert_eq!(
        grid.slices[0]
            .adjustment
            .iter()
            .map(|key| (key.variable.raw(), key.offset))
            .collect::<Vec<_>>(),
        vec![(3, -1)]
    );
    assert!(grid.slices[1].adjustment.is_empty());
    assert!(multi.estimate.ate.is_nan(), "multi-horizon results have no scalar representative");
    assert!(multi.mediation.is_none());

    let z1_cols = [LaggedColumn { variable: VariableId::from_raw(3), lag: Lag::from_raw(1) }];
    let under_i1 = estimate_h1(&data, &z1_cols);
    let under_i2 = estimate_h1(&data, &[]);
    assert!((under_i1 - STRUCTURAL_MEDIATED).abs() < 1e-6);
    assert!(
        (under_i2 - STRUCTURAL_MEDIATED).abs() > 0.2,
        "reusing I(2)={{}} at h=1 must recover the confounded association, got {under_i2}"
    );

    let q = query(&[1]);
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

    let q = query(&[1, 2]);
    let (prepared, prepared_multi) =
        prepare_click(&data, &graph, &q, InferenceMode::Frequentist, RefuteSuite::None, false);
    assert!(has_cached(&prepared_multi));
    assert_eq!(prepared_multi.mediation_grid.as_ref().unwrap().slices.len(), 2);
    let refreshed = prepared.estimate_series(&data, &ctx).unwrap();
    assert_eq!(refreshed.mediation_grid.as_ref().unwrap().slices.len(), 2);

    let (_, bayesian_multi) = prepare_click(
        &data,
        &graph,
        &q,
        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(512)),
        RefuteSuite::None,
        false,
    );
    let bayesian_grid = bayesian_multi.mediation_grid.as_ref().unwrap();
    assert_eq!(bayesian_grid.slices.len(), 2);
    assert!(!bayesian_grid.joint_posterior);
    assert!(bayesian_multi.posterior.is_none());
    assert!(bayesian_multi.mediation.is_none());
    assert!(bayesian_grid.slices.iter().all(|slice| matches!(
        slice.uncertainty,
        antecedent::estimate::TemporalMediationUncertainty::BayesianPointwise { .. }
    )));
}

#[test]
fn multi_horizon_parent_identification_is_the_conservative_join() {
    let (data, graph) = confounded_mediation_series();
    let result = Study::series(data)
        .graph(graph)
        .query(CausalQuery::Mediation(query(&[1, 2])))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(7))
        .unwrap();
    let grid = result.mediation_grid.as_ref().expect("multi-horizon mediation grid");
    assert!(grid.slices.iter().all(|slice| {
        slice.identification_status == IdentificationStatus::IdentifiedUnderParametricRestrictions
    }));
    assert_eq!(
        result.identification.status,
        IdentificationStatus::IdentifiedUnderParametricRestrictions
    );
}

#[test]
fn temporal_cpdag_mediation_returns_per_horizon_identified_sets() {
    let (data, _) = confounded_mediation_series();
    let mut graph = TemporalCpdag::empty();
    let t0 = graph.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
    let t1 = graph.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let m0 = graph.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let y0 = graph.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    let z0 = graph.add_lagged(VariableId::from_raw(3), Lag::CONTEMPORANEOUS).unwrap();
    let z1 = graph.add_lagged(VariableId::from_raw(3), Lag::from_raw(1)).unwrap();
    graph.insert_directed(z0, t0).unwrap();
    graph.insert_directed(z1, y0).unwrap();
    graph.insert_directed(z1, m0).unwrap();
    graph.insert_directed(t1, y0).unwrap();
    graph.insert_directed(t1, m0).unwrap();
    graph.insert_directed(m0, y0).unwrap();
    for accepted in [false, true] {
        let builder = Study::series(data.clone());
        let builder = if accepted {
            builder.graph(AcceptedGraph::temporal_cpdag(graph.clone()).unwrap())
        } else {
            builder.graph(graph.clone())
        };
        let result = builder
            .query(CausalQuery::Mediation(query(&[1, 2])))
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .run(&ExecutionContext::for_tests(27))
            .unwrap();
        let grid = result.mediation_grid.as_ref().expect("class mediation grid");
        assert_eq!(grid.slices.len(), 2);
        assert!(grid.slices.iter().all(|slice| {
            slice.identified_set.is_some_and(|set| set.lower.is_finite() && set.lower <= set.upper)
        }));
        assert!(result.mediation.is_none());
        assert!(result.estimate.ate.is_nan());
    }
}

#[test]
fn temporal_cpdag_mediation_cheap_and_full_run_per_completion() {
    let (data, _) = confounded_mediation_series();
    let mut graph = TemporalCpdag::empty();
    let t0 = graph.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
    let t1 = graph.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let m0 = graph.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let y0 = graph.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    let z0 = graph.add_lagged(VariableId::from_raw(3), Lag::CONTEMPORANEOUS).unwrap();
    let z1 = graph.add_lagged(VariableId::from_raw(3), Lag::from_raw(1)).unwrap();
    graph.insert_directed(z0, t0).unwrap();
    graph.insert_directed(z1, y0).unwrap();
    graph.insert_directed(z1, m0).unwrap();
    graph.insert_directed(t1, y0).unwrap();
    graph.insert_directed(t1, m0).unwrap();
    graph.insert_directed(m0, y0).unwrap();
    for (accepted, suite) in [(false, RefuteSuite::Cheap), (true, RefuteSuite::Full)] {
        let builder = Study::series(data.clone());
        let builder = if accepted {
            builder.graph(AcceptedGraph::temporal_cpdag(graph.clone()).unwrap())
        } else {
            builder.graph(graph.clone())
        };
        let result = builder
            .query(CausalQuery::Mediation(query(&[1])))
            .refute(suite)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .run(&ExecutionContext::for_tests(29))
            .unwrap();
        assert!(
            !result.refutations.is_empty(),
            "TemporalCpdag mediation {suite:?} must run per-completion refuters"
        );
        assert!(
            result
                .refutations
                .iter()
                .any(|report| { report.refuter.as_ref().starts_with("horizon.1.completion.") })
        );
        assert!(result.mediation_grid.as_ref().is_some_and(|grid| {
            grid.slices.iter().all(|slice| slice.identified_set.is_some())
        }));
        assert!(result.estimate.ate.is_nan());
    }
}

#[test]
fn temporal_pag_mediation_refuses_as_cross_world_theory_boundary() {
    let (data, _) = confounded_mediation_series();
    let mut graph = TemporalPag::empty();
    let t1 = graph.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let m0 = graph.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let y0 = graph.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(t1, m0).unwrap();
    graph.insert_directed(t1, y0).unwrap();
    graph.insert_directed(m0, y0).unwrap();
    let error = Study::series(data)
        .graph(graph)
        .query(CausalQuery::Mediation(query(&[1, 2])))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap_err();
    let message = error.to_string();
    assert!(
        message.contains("Latent-confounded")
            && message.contains("cross-world")
            && !message.contains("1.10")
            && !message.contains("1.7"),
        "TemporalPag mediation must refuse as a mathematical boundary, got {message}"
    );
}
