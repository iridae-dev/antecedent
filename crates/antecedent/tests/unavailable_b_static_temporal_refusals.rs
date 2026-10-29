// SPDX-License-Identifier: MIT OR Apache-2.0
//! A pulse or sustained contrast declared on a static graph is refused with the typed
//! not-applicable reason, never estimated as a one-shot do().
//!
//! A pulse and a sustained contrast are indexed by a lag and a horizon, so a static `Dag` has no
//! time index for either. The same data and graph answer a static average effect (the valid
//! control), and each temporal query declared over them is refused at build, under both
//! inference modes and with or without resampling uncertainty.

use antecedent::{
    BayesianConfig, CausalError, InferenceMode, RefuteSuite, Study, StudyBuilder, SupportRefusal,
};
use antecedent_core::{AverageEffectQuery, CausalQuery, TemporalEffectQuery, VariableId};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};

fn id(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn data() -> TabularData {
    let t: Vec<f64> = (0..40).map(|i| f64::from(i % 2)).collect();
    let y: Vec<f64> = (0..40).map(|i| 2.0 * f64::from(i % 2) + 0.05 * f64::from(i % 7)).collect();
    TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice())]).unwrap()
}

fn graph() -> Dag {
    let mut dag = Dag::with_variables(2);
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    dag
}

fn study(query: CausalQuery, inference: InferenceMode, replicates: u32) -> StudyBuilder {
    Study::tabular(data())
        .graph(graph())
        .query(query)
        .inference(inference)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(replicates)
}

fn pulse() -> CausalQuery {
    CausalQuery::TemporalEffect(TemporalEffectQuery::pulse(id(0), id(1), 1.0))
}

fn sustained() -> CausalQuery {
    CausalQuery::TemporalEffect(TemporalEffectQuery::sustained(id(0), id(1), 2, 1.0))
}

fn assert_not_applicable(
    query: CausalQuery,
    inference: InferenceMode,
    replicates: u32,
    what: &str,
) {
    let error = match study(query, inference, replicates).build() {
        Ok(_) => panic!("{what}: a temporal contrast on a static graph was accepted"),
        Err(error) => error,
    };
    assert!(
        matches!(&error, CausalError::Support { id: SupportRefusal::NotApplicable, .. }),
        "{what}: {error}"
    );
    let text = error.to_string();
    assert!(text.contains("time index") || text.contains("temporal"), "{what}: {text}");
}

#[test]
fn unavailable_b_the_static_control_builds_and_runs_where_the_temporal_contrast_is_refused() {
    let control = CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(id(0), id(1)));
    let result = study(control, InferenceMode::Frequentist, 0)
        .build()
        .expect("the static average effect is the valid control")
        .run(&antecedent_core::ExecutionContext::for_tests(11))
        .unwrap();
    assert!(result.estimate().as_effect().unwrap().ate.is_finite());
}

#[test]
fn unavailable_b_pulse_on_a_static_graph_is_refused_under_both_inference_modes() {
    for (inference, label) in [
        (InferenceMode::Frequentist, "frequentist"),
        (InferenceMode::Bayesian(BayesianConfig::laplace()), "bayesian"),
    ] {
        assert_not_applicable(pulse(), inference, 0, label);
    }
}

#[test]
fn unavailable_b_pulse_on_a_static_graph_is_refused_whatever_resampling_uncertainty_is_asked() {
    for replicates in [0, 50, 200] {
        for (inference, label) in [
            (InferenceMode::Frequentist, "frequentist"),
            (InferenceMode::Bayesian(BayesianConfig::laplace()), "bayesian"),
        ] {
            assert_not_applicable(
                pulse(),
                inference,
                replicates,
                &format!("{label} replicates {replicates}"),
            );
        }
    }
}

#[test]
fn unavailable_b_sustained_on_a_static_graph_is_refused_under_both_inference_modes() {
    for (inference, label) in [
        (InferenceMode::Frequentist, "frequentist"),
        (InferenceMode::Bayesian(BayesianConfig::laplace()), "bayesian"),
    ] {
        assert_not_applicable(sustained(), inference, 0, label);
    }
}

#[test]
fn unavailable_b_sustained_on_a_static_graph_is_refused_whatever_resampling_uncertainty_is_asked() {
    for replicates in [0, 50, 200] {
        for (inference, label) in [
            (InferenceMode::Frequentist, "frequentist"),
            (InferenceMode::Bayesian(BayesianConfig::laplace()), "bayesian"),
        ] {
            assert_not_applicable(
                sustained(),
                inference,
                replicates,
                &format!("{label} replicates {replicates}"),
            );
        }
    }
}
