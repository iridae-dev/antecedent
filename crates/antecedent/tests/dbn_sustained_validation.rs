//! Numerical DBN sustained-window validation: both lagged actions matter.
use std::sync::Arc;

use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint, SmallRoleSet,
    TemporalEffectQuery, TemporalPolicy, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
    TimeSeriesData, ValidityBitmap,
};
use antecedent_discovery::GraphPosterior;
use antecedent_prob::InferenceDiagnostics;

fn fixture() -> (TimeSeriesData, GraphPosterior, TemporalEffectQuery) {
    let n = 402_usize;
    let treatment: Vec<f64> = (0..n)
        .map(|i| match i % 4 {
            1 => 1.0,
            3 => -1.0,
            _ => 0.0,
        })
        .collect();
    let outcome: Vec<f64> = (0..n)
        .map(|i| {
            1.0 + 2.0 * i.checked_sub(1).map_or(0.0, |j| treatment[j])
                + 3.0 * i.checked_sub(2).map_or(0.0, |j| treatment[j])
                + 0.03 * (f64::from(u32::try_from(i).unwrap()) * 0.73).sin()
        })
        .collect();
    let mut builder = CausalSchemaBuilder::new();
    for name in ["t", "y"] {
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
    let columns = [treatment, outcome]
        .into_iter()
        .enumerate()
        .map(|(i, values)| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(u32::try_from(i).unwrap()),
                    Arc::from(values),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            )
        })
        .collect();
    let data = TimeSeriesData::try_new(
        OwnedColumnarStorage::try_new(builder.build().unwrap(), columns, None, None).unwrap(),
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap();
    // Atom 0 keeps both T lags (effect 5), atom 1 only lag one (effect 2).
    // Atom 2 adds an uncertifiable treatment autoregression; retain its 0.2 mass.
    // Conditional mixture truth = (0.6*5 + 0.2*2)/0.8 = 4.25.
    let weights: Vec<f64> = vec![0.6, 0.2, 0.2];
    let posterior = GraphPosterior::new(
        2,
        weights,
        vec![0; 3],
        vec![0.0; 4],
        vec![0.0; 4],
        1.0 / (0.6 * 0.6 + 0.2 * 0.2 + 0.2 * 0.2),
        InferenceDiagnostics::analytic("sustained-validation"),
        0,
    )
    .unwrap()
    .with_lagged_marginals(2, vec![0.2, 1.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0])
    .unwrap()
    .with_lag_masks(vec![34, 2, 35])
    .unwrap();
    let query = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::sustained(-2, -1))
        .with_horizon_steps(1);
    (data, posterior, query)
}

fn assert_validation(inference: &InferenceMode, suite: RefuteSuite) {
    let (series, posterior, query) = fixture();
    let ctx = ExecutionContext::for_tests(21);
    let single = Study::series(series.clone())
        .graph_posterior(posterior.clone())
        .temporal_query(query.clone().with_policy(TemporalPolicy::sustained(-1, -1)))
        .inference(inference.clone())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    let prepared = Study::series(series.clone())
        .graph_posterior(posterior)
        .temporal_query(query)
        .inference(inference.clone())
        .refute(suite)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let result = prepared.estimate_series(&series, &ctx).unwrap();
    assert!(
        (result.estimate.ate - 4.25).abs() < 0.15,
        "{inference:?}/{suite:?}: {}",
        result.estimate.ate
    );
    assert!((single.estimate.ate - 2.0).abs() < 0.15);
    assert!(
        result.estimate.ate - single.estimate.ate > 1.8,
        "the validation route must not collapse the window"
    );
    assert!(!result.refutations.is_empty(), "{inference:?}/{suite:?}");
    assert!(
        result.refutations.iter().all(|r| r.original_ate.is_finite() && r.refuted_ate.is_finite())
    );
    if suite == RefuteSuite::Full {
        assert!(
            result.refutations.iter().any(|r| r.replicates > 0),
            "full validation must actually re-estimate perturbed data: {:?}",
            result.refutations
        );
    }
    if let Some(posterior) = &result.posterior {
        assert!((posterior.unidentified_mass - 0.2).abs() < 1e-12);
        assert!(!result.predictive_checks.is_empty());
        if suite == RefuteSuite::Full {
            assert!(posterior.prior_sensitivity.is_some());
        }
    }
}

#[test]
fn frequentist_multistep_dbn_cheap_and_full_preserve_window() {
    for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
        assert_validation(&InferenceMode::Frequentist, suite);
    }
}

#[test]
fn bayesian_multistep_dbn_cheap_and_full_preserve_window() {
    for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
        assert_validation(
            &InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(128).prior_scale(10.0)),
            suite,
        );
    }
}

#[test]
fn single_graph_full_validation_retains_both_lagged_actions() {
    use antecedent_core::Lag;
    use antecedent_graph::{TemporalDag, ensure_lagged};
    let (series, _, query) = fixture();
    let mut graph = TemporalDag::empty();
    let outcome = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    for lag in [1, 2] {
        let treatment =
            ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(lag)).unwrap();
        graph.insert_directed(treatment, outcome).unwrap();
    }
    for inference in [
        InferenceMode::Frequentist,
        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(128).prior_scale(10.0)),
    ] {
        let result = Study::series(series.clone())
            .graph(graph.clone())
            .temporal_query(query.clone())
            .inference(inference)
            .refute(RefuteSuite::Full)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .run(&ExecutionContext::for_tests(21))
            .unwrap();
        assert!((result.estimate.ate - 5.0).abs() < 0.15, "{}", result.estimate.ate);
        assert!(result.refutations.iter().any(|r| r.replicates > 0));
        if let Some(posterior) = result.posterior {
            assert!(!result.predictive_checks.is_empty());
            assert!(posterior.prior_sensitivity.is_some());
        }
    }
}
