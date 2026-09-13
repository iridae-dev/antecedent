//! 1.7 Bayesian temporal-class envelope pins.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::too_many_lines,
    clippy::many_single_char_names
)]

use std::sync::Arc;

use antecedent::{
    BayesianConfig, ClassPrior, InferenceMode, PreparedStudy, RefuteSuite, Study, identify,
};
use antecedent_core::{
    CausalQuery, CausalSchemaBuilder, ContinuousDomain, ExecutionContext, GridSpec, Intervention,
    InterventionSequence, Lag, MeasurementSpec, ObservationAssumption, ObservationSpec,
    ResponseFunctional, ResponseQuery, RoleHint, SequencedIntervention, SmallRoleSet,
    TemporalEffectQuery, TemporalPolicy, TemporalResponseSpec, Value, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
    TimeSeriesData, ValidityBitmap,
};
use antecedent_graph::{TemporalCpdag, TemporalPag};

fn pin() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../conformance/bayesian/temporal_class_envelope/expected.json"
    ))
    .unwrap()
}

fn transfer_pin() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../conformance/bayesian/temporal_class_prior_transfer/expected.json"
    ))
    .unwrap()
}

fn observation_pin() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../conformance/response/temporal_class_observation/expected.json"
    ))
    .unwrap()
}

fn series_from_law(n: usize, two_lag: bool) -> TimeSeriesData {
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut z = vec![0.0; n];
    for i in 0..n {
        z[i] = if i % 2 == 0 { 0.0 } else { 1.0 };
        t[i] = 0.3 + 0.4 * z[i] + 0.05 * ((i as f64) * 0.017).sin();
        if i > 0 {
            y[i] = 1.0 + 2.0 * t[i - 1] + 0.5 * z[i.saturating_sub(1)];
            if two_lag && i > 1 {
                y[i] = 1.0 + 2.0 * t[i - 1] + 3.0 * t[i - 2];
            }
        }
    }
    let mut builder = CausalSchemaBuilder::new();
    for name in ["t", "y", "z"] {
        builder
            .add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
    }
    let columns = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(t), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(y), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(2), Arc::from(z), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    TimeSeriesData::try_new(
        OwnedColumnarStorage::try_new(builder.build().unwrap(), columns, None, None).unwrap(),
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap()
}

fn cpdag() -> TemporalCpdag {
    let mut g = TemporalCpdag::empty();
    let t1 = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let z1 = g.add_lagged(VariableId::from_raw(2), Lag::from_raw(1)).unwrap();
    g.insert_directed(z1, y0).unwrap();
    g.insert_directed(t1, y0).unwrap();
    g.insert_undirected(z1, t1).unwrap();
    g
}

fn two_lag_cpdag() -> TemporalCpdag {
    let mut g = TemporalCpdag::empty();
    let t1 = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let t2 = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(2)).unwrap();
    let y0 = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let z1 = g.add_lagged(VariableId::from_raw(2), Lag::from_raw(1)).unwrap();
    g.insert_directed(t1, y0).unwrap();
    g.insert_directed(t2, y0).unwrap();
    g.insert_undirected(z1, t1).unwrap();
    g
}

fn directed_pag() -> TemporalPag {
    let mut g = TemporalPag::empty();
    let t1 = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let z1 = g.add_lagged(VariableId::from_raw(2), Lag::from_raw(1)).unwrap();
    g.insert_directed(z1, t1).unwrap();
    g.insert_directed(t1, y0).unwrap();
    g
}

fn pulse_query() -> CausalQuery {
    let mut q = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0);
    q.policy = TemporalPolicy::pulse(-1);
    q.horizon_steps = 1;
    CausalQuery::TemporalEffect(q)
}

fn bayes() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64))
}

#[test]
fn temporal_class_bayesian_pulse_without_prior_is_identified_set() {
    let pin = pin();
    let data = series_from_law(usize::try_from(pin["n"].as_u64().unwrap()).unwrap(), false);
    let result = Study::series(data)
        .graph(cpdag())
        .query(pulse_query())
        .inference(bayes())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(7))
        .unwrap();
    assert!(result.posterior.is_none());
    assert!(!result.estimate.ate.is_finite());
    let structural = result.structural_response.as_ref().expect("structural envelope");
    assert_eq!(
        structural.weight_basis,
        antecedent::result::StructuralWeightBasis::CompletionEnumeration
    );
    assert!(structural.identified_set.is_some());
    assert!(structural.atoms.len() >= 2);
    assert!(
        structural
            .atoms
            .iter()
            .all(|atom| atom.posterior.as_ref().is_some_and(|p| p.draws.n_draws == 64))
    );
    assert!(
        result
            .diagnostics
            .iter()
            .any(|d| { d.code.as_ref() == "estimate.temporal_class.enumeration_not_probability" })
    );
}

#[test]
fn temporal_class_bayesian_pulse_with_class_prior_mixes() {
    let pin = pin();
    let n = usize::try_from(pin["n"].as_u64().unwrap()).unwrap();
    let data = series_from_law(n, false);
    let prior = ClassPrior::from_ordered([0.3, 0.7]).unwrap();
    let result = Study::series(data.clone())
        .graph(cpdag())
        .query(pulse_query())
        .inference(bayes())
        .class_prior(prior)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(7))
        .unwrap();
    assert!(result.posterior.is_some());
    assert!(result.estimate.ate.is_finite());
    let structural = result.structural_response.as_ref().expect("structural envelope");
    assert_eq!(
        structural.weight_basis,
        antecedent::result::StructuralWeightBasis::CallerSuppliedClassPrior
    );
    assert!(structural.unidentified_mass.abs() < 1e-9);
    let ctx = ExecutionContext::for_tests(7);
    let prepared: PreparedStudy = Study::series(data.clone())
        .graph(cpdag())
        .query(pulse_query())
        .inference(bayes())
        .class_prior(ClassPrior::from_ordered([0.3, 0.7]).unwrap())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let click = prepared.estimate_series(&data, &ctx).unwrap();
    assert!(click.diagnostics.iter().any(|d| d.code.as_ref() == "exec.identify.cached"));
    for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
        let validated = Study::series(data.clone())
            .graph(cpdag())
            .query(pulse_query())
            .inference(bayes())
            .class_prior(ClassPrior::from_ordered([0.3, 0.7]).unwrap())
            .refute(suite)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .run(&ExecutionContext::for_tests(7))
            .unwrap();
        assert!(!validated.refutations.is_empty(), "{suite:?} must run envelope refuters");
    }
}

#[test]
fn temporal_class_multi_step_is_not_last_step_collapse() {
    let pin = pin();
    let n = usize::try_from(pin["n"].as_u64().unwrap()).unwrap();
    let data = series_from_law(n, true);
    let mut multi =
        TemporalEffectQuery::sustained(VariableId::from_raw(0), VariableId::from_raw(1), 0, 1.0);
    multi.policy = TemporalPolicy::sustained(-2, -1);
    multi.horizon_steps = 1;
    let result = Study::series(data)
        .graph(two_lag_cpdag())
        .query(CausalQuery::TemporalEffect(multi))
        .inference(bayes())
        .class_prior(ClassPrior::from_ordered([0.5, 0.5]).unwrap())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(3))
        .unwrap();
    let last_step = pin["multi_step"]["last_step_effect"].as_f64().unwrap();
    let full = pin["multi_step"]["full_window_effect"].as_f64().unwrap();
    let tol = pin["multi_step"]["absolute_tolerance"].as_f64().unwrap();
    assert!(
        (result.estimate.ate - last_step).abs() > 1.0,
        "multi-step {} collapsed toward last-step {last_step}",
        result.estimate.ate
    );
    assert!(
        (result.estimate.ate - full).abs() < tol,
        "multi-step {} vs window {full}",
        result.estimate.ate
    );
}

#[test]
fn temporal_class_bayesian_response_is_identified_set() {
    let pin = pin();
    let data = series_from_law(usize::try_from(pin["n"].as_u64().unwrap()).unwrap(), false);
    let query = CausalQuery::Response(
        ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: VariableId::from_raw(1),
            treatment: ContinuousDomain::new(
                VariableId::from_raw(0),
                GridSpec::Values(Arc::from([0.0, 1.0])),
            ),
        })
        .with_temporal(
            TemporalResponseSpec::new(vec![1u32], TemporalPolicy::pulse(-1), None).unwrap(),
        ),
    );
    let result = Study::series(data)
        .graph(cpdag())
        .query(query)
        .inference(bayes())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(17))
        .unwrap();
    let structural = result.structural_response.as_ref().expect("response set");
    assert_eq!(
        structural.weight_basis,
        antecedent::result::StructuralWeightBasis::CompletionEnumeration
    );
    assert!(
        result
            .diagnostics
            .iter()
            .any(|d| { d.code.as_ref() == "estimate.envelope.response_posterior_not_mixed" })
    );
}

#[test]
fn temporal_class_prior_missing_key_refuses() {
    let pin = pin();
    let data = series_from_law(usize::try_from(pin["n"].as_u64().unwrap()).unwrap(), false);
    let err = Study::series(data)
        .graph(cpdag())
        .query(pulse_query())
        .inference(bayes())
        .class_prior(ClassPrior::from_pairs([(1_u64, 1.0)]).unwrap())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(1))
        .unwrap_err();
    assert!(err.to_string().contains("class prior"), "{err}");
}

#[test]
fn temporal_pag_directed_bayesian_pulse_identifies() {
    let pin = pin();
    let data = series_from_law(usize::try_from(pin["n"].as_u64().unwrap()).unwrap(), false);
    let result = Study::series(data)
        .graph(directed_pag())
        .query(pulse_query())
        .inference(bayes())
        .class_prior(ClassPrior::from_ordered([1.0]).unwrap())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(5))
        .unwrap();
    assert!(result.estimate.ate.is_finite());
    let structural = result.structural_response.as_ref().expect("pag envelope");
    assert!(structural.unidentified_mass.abs() < 1e-9);
}

#[test]
fn temporal_class_sequence_unidentified_coordinate_keeps_class() {
    let pin = pin();
    let data = series_from_law(usize::try_from(pin["n"].as_u64().unwrap()).unwrap(), false);
    let seq = InterventionSequence::new(vec![SequencedIntervention {
        intervention: Intervention::set(VariableId::from_raw(0), Value::f64(1.0)),
        temporal: TemporalPolicy::pulse(-1),
    }]);
    let query = CausalQuery::Response(
        ResponseQuery::new(ResponseFunctional::InterventionResponse {
            outcome: VariableId::from_raw(1),
            interventions: Arc::from([Intervention::Sequence(seq)]),
        })
        .with_temporal(
            TemporalResponseSpec::new(vec![1u32], TemporalPolicy::pulse(-1), None).unwrap(),
        ),
    );
    let result = Study::series(data)
        .graph(cpdag())
        .query(query)
        .inference(bayes())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(11))
        .unwrap();
    let structural = result.structural_response.as_ref().expect("sequence set");
    assert!(structural.atoms.iter().any(|atom| atom.value.is_some()));
    assert!(
        result.diagnostics.iter().any(|d| d.code.as_ref() == "estimate.temporal.sequence_overlay")
    );
}

#[test]
fn temporal_class_identify_response_returns_envelope() {
    let identified = identify(
        &antecedent::AcceptedGraph::from(cpdag()),
        &CausalQuery::Response(
            ResponseQuery::new(ResponseFunctional::MeanCurve {
                outcome: VariableId::from_raw(1),
                treatment: ContinuousDomain::new(
                    VariableId::from_raw(0),
                    GridSpec::Values(Arc::from([0.0, 1.0])),
                ),
            })
            .with_temporal(
                TemporalResponseSpec::new(vec![1u32], TemporalPolicy::pulse(-1), None).unwrap(),
            ),
        ),
    )
    .unwrap();
    assert!(!identified.completion_keys().is_empty());
    assert!(matches!(identified, antecedent::Identification::TemporalEnvelope { .. }));
}

#[test]
fn temporal_class_observation_does_not_reuse_complete_band() {
    let pin = observation_pin();
    assert_eq!(pin["case"].as_str(), Some("temporal_class_observation"));
    let n = 800usize;
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut r = vec![0.0; n];
    let mut z = vec![0.0; n];
    for i in 0..n {
        z[i] = if i % 2 == 0 { 0.0 } else { 1.0 };
        t[i] = 0.3 + 0.4 * z[i] + 0.05 * ((i as f64) * 0.017).sin();
        r[i] = if i % 7 == 0 { 0.0 } else { 1.0 };
        if i > 0 {
            y[i] = if r[i] > 0.5 { 1.0 + 2.0 * t[i - 1] } else { 0.0 };
        }
    }
    let data = TimeSeriesData::from_f64_columns(
        [("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice()), ("r", r.as_slice())],
        1,
    )
    .unwrap();
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from([0.0, 1.0])),
        ),
    })
    .with_temporal(TemporalResponseSpec::new(vec![1u32], TemporalPolicy::pulse(-1), None).unwrap())
    .with_observation(
        ObservationSpec::Selected {
            latent: VariableId::from_raw(1),
            observed: VariableId::from_raw(1),
            indicator: VariableId::from_raw(3),
        },
        [ObservationAssumption::OutcomeIndependentGiven(Arc::from([
            VariableId::from_raw(0),
            VariableId::from_raw(2),
        ]))],
    );
    let mut oriented = TemporalCpdag::empty();
    let t1 = oriented.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = oriented.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let z1 = oriented.add_lagged(VariableId::from_raw(2), Lag::from_raw(1)).unwrap();
    oriented.insert_directed(z1, t1).unwrap();
    oriented.insert_directed(t1, y0).unwrap();
    let result = Study::series(data)
        .graph(oriented)
        .query(CausalQuery::Response(query))
        .observation_options(antecedent_estimate::ObservationEstimatorOptions {
            selected_correction: antecedent_estimate::SelectedOutcomeCorrection::Ipw,
            ..antecedent_estimate::ObservationEstimatorOptions::default()
        })
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(13))
        .unwrap();
    let response = result.response.as_ref().expect("class observation response");
    assert!(matches!(response.uncertainty, antecedent_core::ResponseUncertainty::None));
    assert!(
        result
            .diagnostics
            .iter()
            .any(|d| { d.code.as_ref() == "estimate.temporal_class.observation_no_complete_band" })
    );
}

#[test]
fn temporal_class_prior_transfer_conflict_does_not_flip_identification() {
    let pin = transfer_pin();
    assert_eq!(pin["compatibility_filter"].as_str(), Some("PriorCatalog.filter_compatible"));
    let n = usize::try_from(pin["n"].as_u64().unwrap()).unwrap();
    let n_draws = usize::try_from(pin["n_draws"].as_u64().unwrap()).unwrap();
    let seed = pin["seed"].as_u64().unwrap();
    let source = Study::series(series_from_law(n, false))
        .graph(cpdag())
        .query(pulse_query())
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(n_draws)))
        .class_prior(ClassPrior::from_ordered([0.5, 0.5]).unwrap())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(seed))
        .unwrap();
    let bytes =
        antecedent::io::encode_causal_posterior_bytes(source.posterior.as_ref().unwrap(), "source")
            .unwrap();
    let target = Study::series(series_from_law(n, false))
        .graph(cpdag())
        .query(pulse_query())
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(n_draws).prior_from_artifact(bytes, None),
        ))
        .class_prior(ClassPrior::from_ordered([0.5, 0.5]).unwrap())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(seed.wrapping_add(1)));
    // A mapping/filter conflict may refuse or attach a conflict diagnostic.
    // Identification status must not become NotIdentified solely from the prior.
    match target {
        Ok(result) => {
            assert_ne!(format!("{:?}", result.identification.status), "NotIdentified");
        }
        Err(error) => {
            let message = error.to_string();
            assert!(
                !message.contains("NotIdentified"),
                "prior transfer must not flip identification: {message}"
            );
        }
    }
}

#[test]
fn temporal_class_sequence_refuses_prior_transfer() {
    let pin = transfer_pin();
    let n = usize::try_from(pin["n"].as_u64().unwrap()).unwrap();
    let n_draws = usize::try_from(pin["n_draws"].as_u64().unwrap()).unwrap();
    let seed = pin["seed"].as_u64().unwrap();
    let source = Study::series(series_from_law(n, false))
        .graph(cpdag())
        .query(pulse_query())
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(n_draws)))
        .class_prior(ClassPrior::from_ordered([0.5, 0.5]).unwrap())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(seed))
        .unwrap();
    let bytes =
        antecedent::io::encode_causal_posterior_bytes(source.posterior.as_ref().unwrap(), "source")
            .unwrap();
    let seq = InterventionSequence::new(vec![
        SequencedIntervention {
            intervention: Intervention::set(VariableId::from_raw(0), Value::f64(1.0)),
            temporal: TemporalPolicy::pulse(-2),
        },
        SequencedIntervention {
            intervention: Intervention::set(VariableId::from_raw(0), Value::f64(1.0)),
            temporal: TemporalPolicy::pulse(-1),
        },
    ]);
    let query = CausalQuery::Response(
        ResponseQuery::new(ResponseFunctional::InterventionResponse {
            outcome: VariableId::from_raw(1),
            interventions: Arc::from([Intervention::Sequence(seq)]),
        })
        .with_temporal(
            TemporalResponseSpec::new(vec![1u32], TemporalPolicy::sustained(-2, -1), None).unwrap(),
        ),
    );
    let err = Study::series(series_from_law(n, true))
        .graph(two_lag_cpdag())
        .query(query)
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(n_draws).prior_from_artifact(bytes, None),
        ))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(seed))
        .unwrap_err();
    let needle = pin["sequence_refuses"]["message_contains"].as_str().unwrap();
    assert!(err.to_string().contains(needle) || err.to_string().contains("transfer"), "{err}");
}

#[test]
fn class_curve_prior_is_used_and_atom_uncertainty_survives() {
    let query = CausalQuery::Response(
        ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: VariableId::from_raw(1),
            treatment: ContinuousDomain::new(
                VariableId::from_raw(0),
                GridSpec::Values(Arc::from([0.0, 1.0])),
            ),
        })
        .with_temporal(
            TemporalResponseSpec::new(vec![1, 2], TemporalPolicy::pulse(-1), None).unwrap(),
        ),
    );
    let run = |weights: [f64; 2]| {
        Study::series(series_from_law(800, false))
            .graph(cpdag())
            .query(query.clone())
            .inference(bayes())
            .class_prior(ClassPrior::from_ordered(weights).unwrap())
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .run(&ExecutionContext::for_tests(17))
            .unwrap()
    };
    let result = run([0.2, 0.8]);
    let structural = result.structural_response.as_ref().unwrap();
    assert_eq!(
        structural.weight_basis,
        antecedent::result::StructuralWeightBasis::CallerSuppliedClassPrior
    );
    let Some(antecedent_core::ResponseValue::Surface { mean, .. }) =
        &structural.conditional_on_identified
    else {
        panic!("missing class-prior curve")
    };
    assert_eq!(mean.len(), 4);
    for horizon in 0..2 {
        for cell in 0..2 {
            let expected: f64 = structural
                .atoms
                .iter()
                .filter(|atom| atom.graph_key >> 32 == horizon as u64)
                .map(|atom| {
                    assert!(atom.response.is_some());
                    let Some(antecedent_core::ResponseValue::Surface { mean, .. }) = &atom.value
                    else {
                        panic!("missing atom surface")
                    };
                    atom.weight * mean[cell]
                })
                .sum();
            assert!((mean[cell * 2 + horizon] - expected).abs() < 1e-10);
        }
    }
    for masses in [[2e-100, 8e-100], [2e307, 8e307]] {
        let scaled = run(masses);
        let Some(antecedent_core::ResponseValue::Surface { mean: scaled_mean, .. }) =
            scaled.structural_response.unwrap().conditional_on_identified
        else {
            panic!("scaled class prior lost conditional surface")
        };
        for (actual, expected) in scaled_mean.iter().zip(mean.iter()) {
            assert!((actual - expected).abs() < 1e-10);
        }
    }
}

#[test]
fn bayesian_class_selection_matches_directed_observation_likelihood() {
    let n = 400;
    let t: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.31).sin()).collect();
    let r: Vec<f64> = (0..n).map(|i| if i % 3 == 0 { 0.0 } else { 1.0 }).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| if r[i] == 0.0 { -1000.0 } else { 1.0 + 2.0 * t[i.saturating_sub(1)] })
        .collect();
    let data = TimeSeriesData::from_f64_columns(
        [("t", t.as_slice()), ("y", y.as_slice()), ("r", r.as_slice())],
        1,
    )
    .unwrap();
    let mut graph = TemporalCpdag::empty();
    let t1 = graph.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = graph.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(t1, y0).unwrap();
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from([0.0, 1.0])),
        ),
    })
    .with_temporal(TemporalResponseSpec::new(vec![1], TemporalPolicy::pulse(-1), None).unwrap())
    .with_observation(
        ObservationSpec::Selected {
            latent: VariableId::from_raw(1),
            observed: VariableId::from_raw(1),
            indicator: VariableId::from_raw(2),
        },
        [ObservationAssumption::OutcomeIndependentGiven(Arc::from([VariableId::from_raw(0)]))],
    );
    let result = Study::series(data.clone())
        .graph(graph.clone())
        .query(CausalQuery::Response(query.clone()))
        .inference(bayes())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(19))
        .unwrap();
    let atom = &result.structural_response.as_ref().unwrap().atoms[0];
    let Some(antecedent_core::ResponseValue::Surface { mean, .. }) = &atom.value else {
        panic!("missing curve")
    };
    assert!((mean[1] - mean[0] - 2.0).abs() < 0.25, "{mean:?}");
    let mut sequence_query = query;
    sequence_query.functional = ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(1),
        interventions: Arc::from([Intervention::Sequence(InterventionSequence::new(vec![
            SequencedIntervention {
                intervention: Intervention::set(VariableId::from_raw(0), Value::f64(1.0)),
                temporal: TemporalPolicy::pulse(-2),
            },
            SequencedIntervention {
                intervention: Intervention::set(VariableId::from_raw(0), Value::f64(1.0)),
                temporal: TemporalPolicy::pulse(-1),
            },
        ]))]),
    };
    let sequence = Study::series(data)
        .graph(graph)
        .query(CausalQuery::Response(sequence_query))
        .inference(bayes())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(19))
        .unwrap();
    let atom = &sequence.structural_response.as_ref().unwrap().atoms[0];
    let Some(antecedent_core::ResponseValue::Scalar(mean)) = atom.value else {
        panic!("missing sequence response")
    };
    assert!((mean - 3.0).abs() < 0.25, "{mean}");
    assert!(atom.posterior.is_some());
}

#[test]
fn temporal_class_valid_mechanism_prior_transfer_executes() {
    let data = series_from_law(800, false);
    let source = Study::series(data.clone())
        .graph(directed_pag())
        .query(pulse_query())
        .inference(bayes())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(5))
        .unwrap();
    let atom_posterior =
        source.structural_response.as_ref().unwrap().atoms[0].posterior.as_ref().unwrap();
    let bytes = antecedent::io::encode_causal_posterior_bytes(atom_posterior, "mechanism").unwrap();
    let target = Study::series(data)
        .graph(directed_pag())
        .query(pulse_query())
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(64).prior_from_artifact(bytes, None),
        ))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(6))
        .unwrap();
    let target_atom = &target.structural_response.as_ref().unwrap().atoms[0];
    assert_eq!(source.identification.status, target.identification.status);
    assert!(target_atom.posterior.is_some());
    assert!(target_atom.value.is_some());
}

#[test]
fn validation_without_class_prior_uses_finite_atom_baselines() {
    let result = Study::series(series_from_law(800, false))
        .graph(cpdag())
        .query(pulse_query())
        .inference(bayes())
        .refute(RefuteSuite::Full)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(7))
        .unwrap();
    assert!(!result.estimate.ate.is_finite());
    assert!(!result.refutations.is_empty());
    assert!(result.refutations.iter().all(|report| report.original_ate.is_finite()));
    assert!(result.refutations.iter().all(|report| report.refuter.starts_with("completion.")));
    assert!(!result.predictive_checks.is_empty());
}
