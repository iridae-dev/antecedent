//! Regression coverage for collision-free DBN-posterior atom identity.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{BayesianConfig, CausalError, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint,
    SmallRoleSet, TemporalEffectQuery, TemporalPolicy, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
    TimeSeriesData, ValidityBitmap,
};
use antecedent_discovery::{GraphPosterior, set_edge};
use antecedent_prob::InferenceDiagnostics;

const TREATMENT: VariableId = VariableId::from_raw(0);
const OUTCOME: VariableId = VariableId::from_raw(1);
const CONFOUNDER: VariableId = VariableId::from_raw(2);
const N_SAMPLES: usize = 320;

fn confounded_series(confounder_is_observed: bool) -> (TimeSeriesData, TemporalEffectQuery) {
    let mut schema = CausalSchemaBuilder::new();
    schema
        .add_variable(
            "treatment",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    schema
        .add_variable(
            "outcome",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    schema
        .add_variable(
            "confounder",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    let schema = schema.build().unwrap();

    // At each lagged sample, P(T=1 | Z=0)=1/4 and
    // P(T=1 | Z=1)=3/4. Thus Y_t = 2 T_{t-1} + 2 Z_{t-1} + e
    // has unadjusted treatment coefficient 3 and adjusted coefficient 2.
    // Paired residuals make both population regressions exact.
    let n = N_SAMPLES + 1;
    let mut treatment = vec![0.0; n];
    let mut outcome = vec![0.0; n];
    let mut confounder = vec![0.0; n];
    let mut sample = 0;
    for _ in 0..(N_SAMPLES / 16) {
        for (z, t, count) in [(0.0, 0.0, 6), (0.0, 1.0, 2), (1.0, 0.0, 2), (1.0, 1.0, 6)] {
            for row in 0..count {
                let residual = if row % 2 == 0 { -0.2 } else { 0.2 };
                treatment[sample] = t;
                confounder[sample] = z;
                outcome[sample + 1] = 2.0 * t + 2.0 * z + residual;
                sample += 1;
            }
        }
    }
    assert_eq!(sample, N_SAMPLES);

    let confounder_validity = if confounder_is_observed {
        ValidityBitmap::all_valid(n)
    } else {
        ValidityBitmap::from_bytes(vec![0_u8; n.div_ceil(8)], n).unwrap()
    };
    let columns = vec![
        OwnedColumn::Float64(
            Float64Column::new(TREATMENT, Arc::from(treatment), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(OUTCOME, Arc::from(outcome), ValidityBitmap::all_valid(n)).unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(CONFOUNDER, Arc::from(confounder), confounder_validity).unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, columns, None, None).unwrap();
    let series = TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap();
    let query = TemporalEffectQuery::pulse(TREATMENT, OUTCOME, 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1);
    (series, query)
}

fn dbn_posterior(lag_masks: &[u64], weights: &[f64]) -> GraphPosterior {
    assert_eq!(lag_masks.len(), weights.len());
    // Both atoms share Z_t -> T_t and therefore the same public graph key.
    let contemporaneous = set_edge(0, 3, 2, 0, true);
    let mut edge_marginals = vec![0.0; 9];
    edge_marginals[6] = 1.0;
    let mut lagged_marginals = vec![0.0; 9];
    for (&weight, &mask) in weights.iter().zip(lag_masks) {
        for (bit, marginal) in lagged_marginals.iter_mut().enumerate() {
            if mask & (1_u64 << bit) != 0 {
                *marginal += weight;
            }
        }
    }
    let ess = 1.0 / weights.iter().map(|weight| weight * weight).sum::<f64>();
    GraphPosterior::new(
        3,
        weights.to_vec(),
        vec![contemporaneous; weights.len()],
        edge_marginals.clone(),
        edge_marginals,
        ess,
        InferenceDiagnostics::analytic("dbn_atom_identity"),
        0,
    )
    .unwrap()
    .with_lagged_marginals(1, lagged_marginals)
    .unwrap()
    .with_lag_masks(lag_masks.to_vec())
    .unwrap()
}

fn posterior_mean(result: &antecedent::StudyResult) -> f64 {
    let posterior = result.posterior.as_ref().expect("DBN analysis must attach a posterior");
    let effect = posterior.effect_column().expect("DBN posterior must carry an effect column");
    posterior.summaries.mean[effect]
}

fn study(
    series: &TimeSeriesData,
    query: &TemporalEffectQuery,
    posterior: GraphPosterior,
) -> antecedent::Study {
    Study::series(series.clone())
        .graph_posterior(posterior)
        .temporal_query(query.clone())
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(256).prior_scale(1_000_000.0),
        ))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
}

#[test]
fn lag_distinct_dbn_atoms_with_the_same_contemporaneous_key_keep_distinct_effects() {
    // Lag mask packing is from * n_vars + to for max_lag=1.
    let treatment_to_outcome = 1_u64 << 1;
    let confounder_to_outcome = 1_u64 << 7;
    let unadjusted = treatment_to_outcome;
    let adjusted = treatment_to_outcome | confounder_to_outcome;
    let weights = [0.25, 0.75];
    let (series, query) = confounded_series(true);
    let ctx = ExecutionContext::for_tests(73);

    let unadjusted_result =
        study(&series, &query, dbn_posterior(&[unadjusted], &[1.0])).run(&ctx).unwrap();
    let adjusted_result =
        study(&series, &query, dbn_posterior(&[adjusted], &[1.0])).run(&ctx).unwrap();
    let unadjusted_mean = posterior_mean(&unadjusted_result);
    let adjusted_mean = posterior_mean(&adjusted_result);
    assert!((unadjusted_mean - 3.0).abs() < 0.05, "unadjusted mean={unadjusted_mean}");
    assert!((adjusted_mean - 2.0).abs() < 0.05, "adjusted mean={adjusted_mean}");

    for (lag_masks, atom_weights) in [
        (vec![unadjusted, adjusted], weights.to_vec()),
        (vec![adjusted, unadjusted], vec![weights[1], weights[0]]),
    ] {
        let analysis = study(&series, &query, dbn_posterior(&lag_masks, &atom_weights));
        let fresh = analysis.clone().run(&ctx).unwrap();
        let prepared = analysis.prepare(&ctx).unwrap();
        let click = prepared.estimate_series(&series, &ctx).unwrap();
        let expected = weights[0] * unadjusted_mean + weights[1] * adjusted_mean;
        let fresh_mean = posterior_mean(&fresh);
        let click_mean = posterior_mean(&click);
        assert!(
            (fresh_mean - expected).abs() < 0.05,
            "mixture mean={fresh_mean}, expected={expected}"
        );
        assert!((click_mean - fresh_mean).abs() < 1e-12);
        assert!(fresh.posterior.as_ref().unwrap().unidentified_mass.abs() < f64::EPSILON);
        assert!(click.posterior.as_ref().unwrap().unidentified_mass.abs() < f64::EPSILON);
        assert_eq!(
            fresh
                .diagnostics
                .iter()
                .filter(|diagnostic| diagnostic.code.as_ref() == "exec.identify.cached")
                .count(),
            0
        );
        assert_eq!(
            click
                .diagnostics
                .iter()
                .filter(|diagnostic| diagnostic.code.as_ref() == "exec.identify.cached")
                .count(),
            1
        );
    }
}

#[test]
fn lag_distinct_dbn_fit_failure_demotes_the_correct_weight_in_either_order() {
    let treatment_to_outcome = 1_u64 << 1;
    let confounder_to_outcome = 1_u64 << 7;
    let unadjusted = treatment_to_outcome;
    let adjusted = treatment_to_outcome | confounder_to_outcome;
    let failed_weight = 0.7;
    let retained_weight = 1.0 - failed_weight;
    let (series, query) = confounded_series(false);
    let ctx = ExecutionContext::for_tests(73);

    // The unadjusted atom does not consume the entirely missing confounder and
    // remains estimable; the adjusted atom is softly demoted at prepare/fit.
    let retained = study(&series, &query, dbn_posterior(&[unadjusted], &[1.0])).run(&ctx).unwrap();
    assert!(retained.estimand.adjustment_set.is_empty());
    let retained_mean = posterior_mean(&retained);

    for (lag_masks, atom_weights) in [
        (vec![unadjusted, adjusted], vec![retained_weight, failed_weight]),
        (vec![adjusted, unadjusted], vec![failed_weight, retained_weight]),
    ] {
        let analysis = study(&series, &query, dbn_posterior(&lag_masks, &atom_weights));
        let fresh = analysis.clone().run(&ctx).unwrap();
        let prepared = analysis.prepare(&ctx).unwrap();
        let click = prepared.estimate_series(&series, &ctx).unwrap();
        assert!((posterior_mean(&fresh) - retained_mean).abs() < 1e-12);
        assert!((posterior_mean(&click) - retained_mean).abs() < 1e-12);
        assert!(fresh.estimand.adjustment_set.is_empty());
        assert!(click.estimand.adjustment_set.is_empty());
        assert!(
            (fresh.posterior.as_ref().unwrap().unidentified_mass - failed_weight).abs() < 1e-12
        );
        assert!(
            (click.posterior.as_ref().unwrap().unidentified_mass - failed_weight).abs() < 1e-12
        );
    }
}

#[test]
fn dbn_posterior_with_a_static_query_is_refused_rather_than_panicking() {
    // A graph posterior on the prepared handle is licensed for tabular
    // AverageEffect and series TemporalEffect only. A series study that pairs a
    // DBN posterior with a static AverageEffect query must answer with a typed
    // refusal at build or prepare, never reach an internal unreachable branch.
    let treatment_to_outcome = 1_u64 << 1;
    let (series, _query) = confounded_series(true);
    let ctx = ExecutionContext::for_tests(73);
    let built = Study::series(series)
        .graph_posterior(dbn_posterior(&[treatment_to_outcome], &[1.0]))
        .query(AverageEffectQuery::binary_ate(TREATMENT, OUTCOME))
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(16)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build();
    let error = match built {
        Err(error) => error,
        Ok(study) => {
            study.prepare(&ctx).expect_err("a static query on a DBN posterior must not prepare")
        }
    };
    assert!(
        matches!(error, CausalError::Support { .. } | CausalError::Unsupported { .. }),
        "expected a typed support refusal, got {error:?}"
    );
}
