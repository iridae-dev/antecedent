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
    Assumption, AssumptionSet, CausalQuery, CausalSchemaBuilder, ContinuousDomain,
    ExecutionContext, GridSpec, IdentificationStatus, Intervention, InterventionSequence, Lag,
    MeasurementSpec, ObservationAssumption, ObservationSpec, ResponseFunctional, ResponseQuery,
    ResponseUncertainty, RoleHint, SequencedIntervention, SmallRoleSet, TemporalEffectQuery,
    TemporalPolicy, TemporalResponseSpec, Value, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
    TimeSeriesData, ValidityBitmap,
};
use antecedent_graph::{MarkedEdge, TemporalCpdag, TemporalPag};

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
    let carries_claim = |set: &AssumptionSet| {
        set.entries.iter().any(|record| {
            matches!(
                &record.assumption,
                Assumption::Custom { id, .. } if id.as_ref() == "observation.outcome_independent_given"
            )
        })
    };
    assert!(
        carries_claim(&response.assumptions),
        "the observation claim must ride the class response, not vanish with the adjusted series"
    );
    assert!(carries_claim(&result.identification.required_assumptions));
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

fn mixed_pag() -> TemporalPag {
    // t_{-1} -> y_0 confounded through z_{-1}; u_{-1} -> t_{-1} keeps the causal
    // edge visible, and w_{-1} <-> z_{-1} leaves a bidirected edge off the
    // treatment-outcome paths. Pulse adjustment on z_{-1} identifies while the
    // single completion is a genuine MAG with a bidirected edge.
    let mut g = TemporalPag::empty();
    let t1 = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let z1 = g.add_lagged(VariableId::from_raw(2), Lag::from_raw(1)).unwrap();
    let w1 = g.add_lagged(VariableId::from_raw(3), Lag::from_raw(1)).unwrap();
    let u1 = g.add_lagged(VariableId::from_raw(4), Lag::from_raw(1)).unwrap();
    g.insert_directed(t1, y0).unwrap();
    g.insert_directed(z1, t1).unwrap();
    g.insert_directed(z1, y0).unwrap();
    g.insert_directed(u1, t1).unwrap();
    g.insert_marked(MarkedEdge::bidirected(w1, z1)).unwrap();
    g
}

fn mixed_pag_series(n: usize) -> TimeSeriesData {
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut z = vec![0.0; n];
    let mut w = vec![0.0; n];
    let mut u = vec![0.0; n];
    for i in 0..n {
        z[i] = if i % 2 == 0 { 0.0 } else { 1.0 };
        w[i] = 0.5 * z[i] + 0.1 * ((i as f64) * 0.13).cos();
        u[i] = ((i as f64) * 0.29).sin();
        t[i] = 0.3 + 0.4 * z[i] + 0.2 * u[i];
        if i > 0 {
            y[i] = 1.0 + 2.0 * t[i - 1] + 0.5 * z[i - 1];
        }
    }
    TimeSeriesData::from_f64_columns(
        [
            ("t", t.as_slice()),
            ("y", y.as_slice()),
            ("z", z.as_slice()),
            ("w", w.as_slice()),
            ("u", u.as_slice()),
        ],
        1,
    )
    .unwrap()
}

fn single_step_sequence_query() -> CausalQuery {
    let seq = InterventionSequence::new(vec![SequencedIntervention {
        intervention: Intervention::set(VariableId::from_raw(0), Value::f64(1.0)),
        temporal: TemporalPolicy::pulse(-1),
    }]);
    CausalQuery::Response(
        ResponseQuery::new(ResponseFunctional::InterventionResponse {
            outcome: VariableId::from_raw(1),
            interventions: Arc::from([Intervention::Sequence(seq)]),
        })
        .with_temporal(
            TemporalResponseSpec::new(vec![1u32], TemporalPolicy::pulse(-1), None).unwrap(),
        ),
    )
}

fn two_horizon_curve_query() -> CausalQuery {
    CausalQuery::Response(
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
    )
}

#[test]
fn bidirected_completion_does_not_inherit_the_pulse_witness_for_a_schedule() {
    let graph = antecedent::AcceptedGraph::temporal_pag(mixed_pag());
    let antecedent::Identification::TemporalEnvelope { envelope: pulse, .. } =
        identify(&graph, &pulse_query()).unwrap()
    else {
        panic!("temporal class pulse identification returns an envelope")
    };
    assert!(
        pulse.envelope.cases.iter().any(|case| {
            case.graph.has_bidirected()
                && case.result.status != IdentificationStatus::GraphDependent
                && !case.result.estimands.is_empty()
        }),
        "the fixture must have a bidirected completion whose pulse witness identifies"
    );
    let antecedent::Identification::TemporalEnvelope { envelope: sequence, .. } =
        identify(&graph, &single_step_sequence_query()).unwrap()
    else {
        panic!("temporal class sequence identification returns an envelope")
    };
    assert_eq!(sequence.envelope.cases.len(), pulse.envelope.cases.len());
    for case in &sequence.envelope.cases {
        if !case.graph.has_bidirected() {
            continue;
        }
        assert_eq!(case.result.status, IdentificationStatus::GraphDependent);
        assert!(case.result.estimands.is_empty(), "no schedule certificate was ever computed");
        assert!(case.result.diagnostics.iter().any(|d| {
            d.code.as_ref() == "identify.temporal_class.bidirected_completion_uncertified"
        }));
    }
    let err = Study::series(mixed_pag_series(400))
        .graph(mixed_pag())
        .query(single_step_sequence_query())
        .inference(bayes())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(23))
        .unwrap_err();
    assert!(err.to_string().contains("no evaluable directed completion"), "{err}");
}

#[test]
fn class_curve_ordered_and_pair_priors_bind_the_same_completions() {
    let keys = identify(&antecedent::AcceptedGraph::from(cpdag()), &two_horizon_curve_query())
        .unwrap()
        .completion_keys();
    assert_eq!(keys.len(), 2);
    let run = |prior: ClassPrior| {
        Study::series(series_from_law(800, false))
            .graph(cpdag())
            .query(two_horizon_curve_query())
            .inference(bayes())
            .class_prior(prior)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .run(&ExecutionContext::for_tests(17))
            .unwrap()
    };
    let ordered = run(ClassPrior::from_ordered([0.2, 0.8]).unwrap());
    let paired = run(ClassPrior::from_pairs([(keys[0], 0.2), (keys[1], 0.8)]).unwrap());
    assert!(
        !ordered.diagnostics.iter().any(|d| d.code.as_ref() == "exec.identify.cached"),
        "a fresh study identifies every horizon; nothing was cached"
    );
    let surface = |result: &antecedent::StudyResult| {
        let structural = result.structural_response.as_ref().unwrap();
        let Some(antecedent_core::ResponseValue::Surface { mean, .. }) =
            &structural.conditional_on_identified
        else {
            panic!("class-prior curve missing")
        };
        mean.to_vec()
    };
    assert_eq!(surface(&ordered), surface(&paired));
    let weights = |result: &antecedent::StudyResult| {
        result
            .structural_response
            .as_ref()
            .unwrap()
            .atoms
            .iter()
            .map(|a| a.weight)
            .collect::<Vec<_>>()
    };
    assert_eq!(weights(&ordered), weights(&paired));
}

#[test]
fn frequentist_class_curve_replicates_ride_the_atoms() {
    let result = Study::series(series_from_law(400, false))
        .graph(cpdag())
        .query(two_horizon_curve_query())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(12)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(29))
        .unwrap();
    let response = result.response.as_ref().expect("class curve");
    assert!(matches!(response.uncertainty, ResponseUncertainty::None));
    let structural = result.structural_response.as_ref().unwrap();
    assert_eq!(structural.atoms.len(), 4);
    for atom in &structural.atoms {
        let atom_response = atom.response.as_ref().expect("atom curve retained");
        assert!(
            !matches!(atom_response.uncertainty, ResponseUncertainty::None),
            "requested replicates must produce a band on every completion atom"
        );
    }
}

#[test]
fn frequentist_multi_step_class_mixture_declares_its_missing_se() {
    let pin = pin();
    let n = usize::try_from(pin["n"].as_u64().unwrap()).unwrap();
    let mut multi =
        TemporalEffectQuery::sustained(VariableId::from_raw(0), VariableId::from_raw(1), 0, 1.0);
    multi.policy = TemporalPolicy::sustained(-2, -1);
    multi.horizon_steps = 1;
    let result = Study::series(series_from_law(n, true))
        .graph(two_lag_cpdag())
        .query(CausalQuery::TemporalEffect(multi))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(3))
        .unwrap();
    let full = pin["multi_step"]["full_window_effect"].as_f64().unwrap();
    let tol = pin["multi_step"]["absolute_tolerance"].as_f64().unwrap();
    assert!((result.estimate.ate - full).abs() < tol, "{}", result.estimate.ate);
    assert!(result.estimate.se_analytic.is_nan(), "two atoms share observations; no mixture SE");
    assert!(
        result
            .diagnostics
            .iter()
            .any(|d| d.code.as_ref() == "estimate.envelope.se_omits_between_atom_variance"),
        "the missing mixture SE must be declared, as on the Pulse envelope"
    );
    assert!(!result.estimate.assumptions.entries.is_empty());
}

fn oriented_dag() -> antecedent_graph::TemporalDag {
    let mut g = antecedent_graph::TemporalDag::empty();
    let t1 = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let z1 = g.add_lagged(VariableId::from_raw(2), Lag::from_raw(1)).unwrap();
    g.insert_directed(z1, t1).unwrap();
    g.insert_directed(z1, y0).unwrap();
    g.insert_directed(t1, y0).unwrap();
    g
}

#[test]
fn bayesian_temporal_pulse_reports_one_estimator_id_on_dag_and_class() {
    let estimator = |result: &antecedent::StudyResult| {
        result.logical_plan.estimator.as_deref().map(str::to_owned)
    };
    let dag_bayes = Study::series(series_from_law(400, false))
        .graph(oriented_dag())
        .query(pulse_query())
        .inference(bayes())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(41))
        .unwrap();
    assert_eq!(estimator(&dag_bayes).as_deref(), Some("bayesian.temporal.gcomp"));
    assert!(dag_bayes.posterior.is_some());
    let dag_freq = Study::series(series_from_law(400, false))
        .graph(oriented_dag())
        .query(pulse_query())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(41))
        .unwrap();
    assert_eq!(estimator(&dag_freq).as_deref(), Some("temporal.linear.adjustment"));
    for prior in [None, Some(ClassPrior::from_ordered([0.5, 0.5]).unwrap())] {
        let mut builder = Study::series(series_from_law(400, false))
            .graph(cpdag())
            .query(pulse_query())
            .inference(bayes())
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0);
        if let Some(prior) = prior {
            builder = builder.class_prior(prior);
        }
        let class = builder.build().unwrap().run(&ExecutionContext::for_tests(41)).unwrap();
        assert_eq!(
            estimator(&class).as_deref(),
            Some("bayesian.temporal.gcomp"),
            "a Bayesian fit must not report the Frequentist estimator whether or not a class \
             prior licenses a mixture"
        );
    }
    let class_freq = Study::series(series_from_law(400, false))
        .graph(cpdag())
        .query(pulse_query())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(41))
        .unwrap();
    assert_eq!(estimator(&class_freq).as_deref(), Some("temporal.linear.adjustment"));
    assert_eq!(
        "bayesian.temporal.gcomp".parse::<antecedent::EstimatorId>().unwrap(),
        antecedent::EstimatorId::BayesianTemporalGcomp
    );
}
