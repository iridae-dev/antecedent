//! TemporalCpdag/Pag graph-posterior Pulse/Sustained: policy, mass, and suites.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::float_cmp,
    clippy::many_single_char_names
)]

use antecedent::{
    BayesianConfig, CausalQuery, InferenceMode, MediationContrast, RefuteSuite,
    StructuralAggregationPolicy, StructuralWeightBasis, Study,
};
use antecedent_core::{
    ContinuousDomain, ExecutionContext, GridSpec, IdentificationStatus, Intervention,
    MediationQuery, ResponseFunctional, ResponseQuery, TemporalEffectQuery, TemporalPolicy,
    TemporalResponseSpec, Value, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_discovery::{
    GraphPosterior, GraphPosteriorAtomKind, set_edge, temporal_cpdag_from_dbn_masks,
    temporal_pag_from_dbn_masks,
};
use antecedent_prob::InferenceDiagnostics;

fn series() -> TimeSeriesData {
    let n = 48usize;
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut z = vec![0.0; n];
    for i in 0..n {
        let phase = f64::from(u32::try_from(i).unwrap()) * 0.37;
        z[i] = phase.sin();
        t[i] = f64::from(u8::from(phase.cos() > 0.0));
        if i > 0 {
            y[i] = 1.4 * t[i - 1] + 0.6 * z[i - 1] + 0.1 * (phase * 1.7).sin();
        }
    }
    TimeSeriesData::from_f64_columns(
        [("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())],
        1,
    )
    .unwrap()
}

/// Non-degenerate law where adjusting for `z@-1` changes the pulse effect.
fn disagreeing_series() -> TimeSeriesData {
    let n = 300usize;
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut z = vec![0.0; n];
    let mut r = vec![0.0; n];
    for i in 0..n {
        let s = i as f64;
        r[i] = (s * 0.29).sin();
        z[i] = 0.4 * r[i] + (s * 0.13).cos() + 0.3 * (s * 1.7).sin();
        t[i] = 0.3 + 0.5 * r[i] + 0.2 * z[i] + 0.4 * (s * 2.3).cos();
        if i > 0 {
            y[i] = 1.0 + 2.0 * t[i - 1] + 0.6 * z[i - 1] + 0.1 * (s * 0.77).sin();
        }
    }
    TimeSeriesData::from_f64_columns(
        [("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())],
        1,
    )
    .unwrap()
}

fn t_to_y_lag_mask() -> u64 {
    lag_bit(3, 0, 1)
}

fn z_confounds_t_contemp() -> u64 {
    set_edge(0, 3, 2, 0, true)
}

fn z_confounds_lag_mask() -> u64 {
    identified_lag_mask() | lag_bit(3, 2, 0)
}

fn disagreeing_posterior(kind: GraphPosteriorAtomKind) -> GraphPosterior {
    class_posterior(
        kind,
        &[0.5, 0.5],
        &[z_confounds_t_contemp(), 0],
        &[z_confounds_lag_mask(), t_to_y_lag_mask()],
    )
}

fn pulse() -> TemporalEffectQuery {
    TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
}

fn licensed_pulse() -> TemporalEffectQuery {
    TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(antecedent_core::TemporalPolicy::pulse(-1))
        .with_horizon_steps(1)
        .with_max_history_lag(Some(1))
}

fn sustained() -> TemporalEffectQuery {
    TemporalEffectQuery::sustained(VariableId::from_raw(0), VariableId::from_raw(1), 0, 1.0)
}

fn lag_bit(n: usize, from: usize, to: usize) -> u64 {
    1u64 << (from * n + to)
}

fn class_posterior(
    kind: GraphPosteriorAtomKind,
    weights: &[f64],
    adjacency: &[u64],
    lag_masks: &[u64],
) -> GraphPosterior {
    let n = 3;
    GraphPosterior::new(
        n,
        weights.to_vec(),
        adjacency.to_vec(),
        vec![0.0; n * n],
        vec![0.0; n * n],
        1.0,
        InferenceDiagnostics::analytic("temporal_class_posterior"),
        0,
    )
    .unwrap()
    .with_atom_kind(kind)
    .with_lagged_marginals(1, vec![0.0; n * n])
    .unwrap()
    .with_lag_masks(lag_masks.to_vec())
    .unwrap()
}

fn identified_lag_mask() -> u64 {
    // t@1 → y@0 and z@1 → y@0
    lag_bit(3, 0, 1) | lag_bit(3, 2, 1)
}

fn fixture_temporal_effect_pin() -> serde_json::Value {
    let expected: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/bayesian/known_truth_mixtures/expected.json"
    ))
    .unwrap();
    expected["temporal_effect"].clone()
}

fn fixture_temporal_effect_series(pin: &serde_json::Value) -> TimeSeriesData {
    let n = usize::try_from(pin["n"].as_u64().unwrap()).unwrap();
    let mut state = pin["seed"].as_u64().unwrap();
    let mut pressure = vec![0.0; n];
    let mut defect = vec![0.0; n];
    for t in 0..n {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let u = (state >> 33) as f64 / (1u64 << 31) as f64;
        pressure[t] = u * 2.0 - 1.0;
        if t > 0 {
            defect[t] = 0.9 * pressure[t - 1];
        }
    }
    TimeSeriesData::from_f64_columns(
        [("pressure", pressure.as_slice()), ("defect", defect.as_slice())],
        1,
    )
    .unwrap()
}

fn reconstruct_temporal_class(contemp: u64, lag: u64, n: usize) {
    let variables: Vec<VariableId> = (0..n).map(|i| VariableId::from_raw(i as u32)).collect();
    temporal_cpdag_from_dbn_masks(contemp, lag, n, 1, &variables)
        .expect("TemporalCpdag reconstructs from DBN masks");
    temporal_pag_from_dbn_masks(contemp, lag, 0, n, 1, &variables)
        .expect("TemporalPag reconstructs from DBN masks");
}

fn fixture_class_posterior(
    kind: GraphPosteriorAtomKind,
    pin: &serde_json::Value,
) -> GraphPosterior {
    let weights: Vec<f64> =
        pin["posterior_weights"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let identified_c = pin["identified_atom"]["contemporaneous_mask"].as_u64().unwrap();
    let unidentified_c = pin["unidentified_atom"]["contemporaneous_mask"].as_u64().unwrap();
    let identified_l = pin["identified_atom"]["lag_mask"].as_u64().unwrap();
    let unidentified_l = pin["unidentified_atom"]["lag_mask"].as_u64().unwrap();
    reconstruct_temporal_class(identified_c, identified_l, 2);
    reconstruct_temporal_class(unidentified_c, unidentified_l, 2);
    let lagged_marginals: Vec<f64> = pin["lagged_edge_marginals"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap())
        .collect();
    GraphPosterior::new(
        2,
        weights,
        vec![identified_c, unidentified_c],
        vec![0.0; 4],
        vec![0.0; 4],
        1.0,
        InferenceDiagnostics::analytic("known_truth_mixtures"),
        0,
    )
    .unwrap()
    .with_atom_kind(kind)
    .with_lagged_marginals(1, lagged_marginals)
    .unwrap()
    .with_lag_masks(vec![identified_l, unidentified_l])
    .unwrap()
}

fn fixture_pulse() -> TemporalEffectQuery {
    TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1)
        .with_max_history_lag(Some(1))
}

fn fixture_sustained() -> TemporalEffectQuery {
    TemporalEffectQuery::sustained(VariableId::from_raw(0), VariableId::from_raw(1), 0, 1.0)
        .with_policy(TemporalPolicy::sustained(-1, -1))
        .with_horizon_steps(1)
        .with_max_history_lag(Some(1))
}

fn two_completion_contemp() -> u64 {
    // t — z at lag 0
    set_edge(set_edge(0, 3, 0, 2, true), 3, 2, 0, true)
}

fn has_policy(result: &antecedent::StudyResult, policy: StructuralAggregationPolicy) -> bool {
    result.diagnostics.iter().any(|d| {
        d.code.as_ref() == "estimate.graph_posterior.structural_aggregation"
            && d.message.contains(policy.as_str())
    })
}

fn run_on(
    data: &TimeSeriesData,
    gp: GraphPosterior,
    query: TemporalEffectQuery,
    inference: InferenceMode,
    refute: RefuteSuite,
    bootstrap_replicates: u32,
) -> antecedent::StudyResult {
    let ctx = ExecutionContext::for_tests(1);
    Study::series(data.clone())
        .graph_posterior(gp)
        .query(query)
        .inference(inference)
        .refute(refute)
        .bootstrap_replicates(bootstrap_replicates)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap()
}

fn run(
    gp: GraphPosterior,
    query: TemporalEffectQuery,
    inference: InferenceMode,
    refute: RefuteSuite,
) -> antecedent::StudyResult {
    run_on(&series(), gp, query, inference, refute, 0)
}

fn run_disagreeing(
    gp: GraphPosterior,
    query: TemporalEffectQuery,
    inference: InferenceMode,
    refute: RefuteSuite,
) -> antecedent::StudyResult {
    run_on(&disagreeing_series(), gp, query, inference, refute, 0)
}

fn has_diagnostic(result: &antecedent::StudyResult, code: &str) -> bool {
    result.diagnostics.iter().any(|d| d.code.as_ref() == code)
}

#[test]
fn temporal_cpdag_graph_posterior_pulse_mixes() {
    let identified = identified_lag_mask();
    // Autoregressive t_{t-1}→t_t hits the finite-history certification
    // boundary, so that atom stays unidentified (empty graphs identify).
    let unidentified = identified | lag_bit(3, 0, 0);
    let gp = class_posterior(
        GraphPosteriorAtomKind::Cpdag,
        &[0.8, 0.2],
        &[two_completion_contemp(), 0],
        &[identified, unidentified],
    );
    let variables = [VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2)];
    let reconstructed =
        temporal_cpdag_from_dbn_masks(two_completion_contemp(), identified, 3, 1, &variables)
            .unwrap();
    assert!(reconstructed.undirected_edge_count() >= 1);
    for inference in [
        InferenceMode::Frequentist,
        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(16)),
    ] {
        let result = run(gp.clone(), pulse(), inference, RefuteSuite::None);
        let mixture = result.structural_response.as_ref().expect("class posterior mixture");
        assert_eq!(mixture.weight_basis, StructuralWeightBasis::PosteriorProbability);
        assert!(
            (mixture.unidentified_mass - 0.2).abs() < 1e-12,
            "unidentified posterior mass must be retained: {}",
            mixture.unidentified_mass
        );
        assert!(result.diagnostics.iter().any(|d| {
            d.code.as_ref() == "estimate.graph_posterior.structural_aggregation"
                && d.message.contains("completion enumeration is not posterior probability")
        }));
        assert!(
            result
                .diagnostics
                .iter()
                .any(|d| { d.code.as_ref() == "estimate.graph_posterior.temporal_class_envelope" })
        );
        assert_eq!(result.identification.status, IdentificationStatus::GraphDependent);
        assert!(mixture.atoms.iter().any(|atom| atom.value.is_some()));
        assert!(mixture.atoms.iter().any(|atom| atom.value.is_none()));
    }
}

fn visible_pag_contemp() -> u64 {
    // z → t and z → y at lag 0: unfolding puts an arrowhead into t@-1.
    set_edge(set_edge(0, 3, 2, 0, true), 3, 2, 1, true)
}

#[test]
fn temporal_pag_graph_posterior_licensed_pulse_runs() {
    let gp = class_posterior(
        GraphPosteriorAtomKind::Pag,
        &[1.0],
        &[visible_pag_contemp()],
        &[lag_bit(3, 0, 1)],
    );
    let result = run(gp, licensed_pulse(), InferenceMode::Frequentist, RefuteSuite::None);
    assert!(
        result.structural_response.is_some(),
        "licensed pulse(-1) TemporalPag GP must identify"
    );
    assert!(
        result.estimate.ate.is_finite()
            || result.identification.status != antecedent_core::IdentificationStatus::NotIdentified
    );
}

#[test]
fn temporal_class_graph_posterior_pulse_and_sustained_run() {
    let identified = identified_lag_mask();
    for kind in [GraphPosteriorAtomKind::Cpdag, GraphPosteriorAtomKind::Pag] {
        let gp = class_posterior(kind, &[1.0], &[0], &[identified]);
        if kind == GraphPosteriorAtomKind::Pag {
            let variables =
                [VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2)];
            temporal_pag_from_dbn_masks(0, identified, 0, 3, 1, &variables).unwrap();
        }
        for query in [pulse(), sustained()] {
            for inference in [
                InferenceMode::Frequentist,
                InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(8)),
            ] {
                let result = run(gp.clone(), query.clone(), inference, RefuteSuite::None);
                assert!(
                    result.structural_response.is_some(),
                    "{kind:?} {:?} missing mixture",
                    query.policy
                );
                assert!(
                    result.estimate.ate.is_finite()
                        || result
                            .diagnostics
                            .iter()
                            .any(|d| d.message.contains("IdentifiedSetEnvelope")
                                || d.message.contains("graph_dependent_atoms")),
                    "{kind:?} produced neither a scalar nor a class policy"
                );
            }
        }
    }
}

#[test]
fn temporal_class_graph_posterior_cheap_and_full_run() {
    let pin = fixture_temporal_effect_pin();
    let fixture_series = fixture_temporal_effect_series(&pin);
    reconstruct_temporal_class(
        pin["identified_atom"]["contemporaneous_mask"].as_u64().unwrap(),
        pin["identified_atom"]["lag_mask"].as_u64().unwrap(),
        2,
    );
    reconstruct_temporal_class(
        pin["unidentified_atom"]["contemporaneous_mask"].as_u64().unwrap(),
        pin["unidentified_atom"]["lag_mask"].as_u64().unwrap(),
        2,
    );
    for kind in [GraphPosteriorAtomKind::Cpdag, GraphPosteriorAtomKind::Pag] {
        let (data, gp, queries) = if kind == GraphPosteriorAtomKind::Cpdag {
            (
                fixture_series.clone(),
                fixture_class_posterior(kind, &pin),
                vec![fixture_pulse(), fixture_sustained()],
            )
        } else {
            (
                series(),
                class_posterior(kind, &[1.0], &[0], &[identified_lag_mask()]),
                vec![pulse(), sustained()],
            )
        };
        for query in queries {
            for inference in [
                InferenceMode::Frequentist,
                InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(8)),
            ] {
                for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
                    let result =
                        run_on(&data, gp.clone(), query.clone(), inference.clone(), suite, 0);
                    assert!(
                        result.diagnostics.iter().any(|d| {
                            d.code.as_ref() == "refute.envelope.temporal_class_posterior"
                        }),
                        "{kind:?} {suite:?} must run temporal class-posterior refuters"
                    );
                    assert!(
                        !result.diagnostics.iter().any(|d| {
                            d.message.contains("linear.adjustment.ate")
                                && d.code.as_ref().contains("ate")
                        }),
                        "ATE refuters must not be attached to temporal class posterior"
                    );
                }
            }
        }
    }
}

fn assert_disagreeing_estimands_withhold_scalar() {
    for kind in [GraphPosteriorAtomKind::Cpdag, GraphPosteriorAtomKind::Pag] {
        let gp = disagreeing_posterior(kind);
        for query in [pulse(), sustained()] {
            let result = run_disagreeing(
                gp.clone(),
                query.clone(),
                InferenceMode::Frequentist,
                RefuteSuite::None,
            );
            assert!(
                has_policy(&result, StructuralAggregationPolicy::GraphDependentAtoms),
                "{kind:?} {:?}: disagreeing atoms must not scalar-mix",
                query.policy
            );
            assert!(
                !result.estimate.ate.is_finite(),
                "{kind:?} {:?}: scalar ate must be withheld",
                query.policy
            );
            let mixture = result.structural_response.as_ref().unwrap();
            assert_eq!(mixture.weight_basis, StructuralWeightBasis::PosteriorProbability);
            assert!(mixture.conditional_on_identified.is_none());
            assert!(
                mixture.identified_set.is_some(),
                "{kind:?} {:?}: identified set must be published",
                query.policy
            );
            assert!(
                mixture.atoms.iter().filter(|atom| atom.value.is_some()).count() >= 2,
                "{kind:?} {:?}: both atoms must contribute values",
                query.policy
            );
            assert!(
                result.diagnostics.iter().any(|d| {
                    d.code.as_ref() == "estimate.graph_posterior.structural_aggregation"
                }),
                "{kind:?} {:?}: structural_aggregation diagnostic required",
                query.policy
            );
        }
    }
}

fn assert_same_estimand_scalar() {
    let identified = identified_lag_mask();
    for kind in [GraphPosteriorAtomKind::Cpdag, GraphPosteriorAtomKind::Pag] {
        let gp = class_posterior(kind, &[1.0], &[0], &[identified]);
        for query in [pulse(), sustained()] {
            let result =
                run(gp.clone(), query.clone(), InferenceMode::Frequentist, RefuteSuite::None);
            assert!(
                has_policy(&result, StructuralAggregationPolicy::SameEstimandWeightedMean),
                "{kind:?} {:?}: single identified atom must scalar-mix",
                query.policy
            );
            assert!(
                result.estimate.ate.is_finite(),
                "{kind:?} {:?}: scalar ate required",
                query.policy
            );
            let mixture = result.structural_response.as_ref().unwrap();
            let conditional = match mixture.conditional_on_identified.as_ref() {
                Some(antecedent_core::ResponseValue::Scalar(value)) => *value,
                _ => panic!("{kind:?} {:?}: conditional_on_identified", query.policy),
            };
            assert!((conditional - result.estimate.ate).abs() < 1e-12);
            assert!(
                result.diagnostics.iter().any(|d| {
                    d.code.as_ref() == "estimate.graph_posterior.structural_aggregation"
                }),
                "{kind:?} {:?}: structural_aggregation diagnostic required",
                query.policy
            );
        }
    }
}

#[test]
fn temporal_class_graph_posterior_disagreeing_estimands_withhold_scalar() {
    assert_disagreeing_estimands_withhold_scalar();
}

#[test]
fn temporal_class_graph_posterior_same_estimand_scalar() {
    assert_same_estimand_scalar();
}

#[test]
fn temporal_class_graph_posterior_numeric_pins() {
    let pin = fixture_temporal_effect_pin();
    let fixture_series = fixture_temporal_effect_series(&pin);
    reconstruct_temporal_class(
        pin["identified_atom"]["contemporaneous_mask"].as_u64().unwrap(),
        pin["identified_atom"]["lag_mask"].as_u64().unwrap(),
        2,
    );
    reconstruct_temporal_class(
        pin["unidentified_atom"]["contemporaneous_mask"].as_u64().unwrap(),
        pin["unidentified_atom"]["lag_mask"].as_u64().unwrap(),
        2,
    );
    let effect_truth = pin["expected_effect_given_identified"].as_f64().unwrap();
    let unidentified_truth = pin["expected_unidentified_mass"].as_f64().unwrap();
    let tolerance = pin["effect_abs_tolerance"].as_f64().unwrap();
    for kind in [GraphPosteriorAtomKind::Cpdag, GraphPosteriorAtomKind::Pag] {
        let (data, gp, pin_numeric) = if kind == GraphPosteriorAtomKind::Cpdag {
            (fixture_series.clone(), fixture_class_posterior(kind, &pin), true)
        } else {
            (series(), class_posterior(kind, &[1.0], &[0], &[identified_lag_mask()]), false)
        };
        for query in [fixture_pulse(), fixture_sustained()] {
            for inference in [
                InferenceMode::Frequentist,
                InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(16)),
            ] {
                let result = run_on(
                    &data,
                    gp.clone(),
                    if pin_numeric {
                        query.clone()
                    } else if matches!(query.policy, TemporalPolicy::Pulse { .. }) {
                        pulse()
                    } else {
                        sustained()
                    },
                    inference,
                    RefuteSuite::None,
                    0,
                );
                assert_eq!(result.support_status.unwrap().as_str(), "licensed", "{kind:?}");
                if pin_numeric {
                    assert!(
                        (result.estimate.ate - effect_truth).abs() < tolerance,
                        "{kind:?} {:?}: effect {} vs {effect_truth}",
                        query.policy,
                        result.estimate.ate
                    );
                    let mixture = result.structural_response.as_ref().expect("mixture");
                    assert!(
                        (mixture.unidentified_mass - unidentified_truth).abs() < 1e-12,
                        "{kind:?}: unidentified mass {}",
                        mixture.unidentified_mass
                    );
                } else {
                    assert!(result.structural_response.is_some(), "{kind:?} mixture");
                }
            }
        }
    }
    assert_disagreeing_estimands_withhold_scalar();
    assert_same_estimand_scalar();
}

#[test]
fn temporal_class_graph_posterior_shared_block_se_when_mixable() {
    const SHARED_BLOCK: &str = "estimate.temporal_class.frequentist.shared_block";
    let identified = identified_lag_mask();
    for kind in [GraphPosteriorAtomKind::Cpdag, GraphPosteriorAtomKind::Pag] {
        let gp = class_posterior(kind, &[1.0], &[0], &[identified]);
        for query in [pulse(), sustained()] {
            let result = run_on(
                &series(),
                gp.clone(),
                query.clone(),
                InferenceMode::Frequentist,
                RefuteSuite::None,
                32,
            );
            assert!(
                has_policy(&result, StructuralAggregationPolicy::SameEstimandWeightedMean),
                "{kind:?} {:?}: scalar mix required for shared block",
                query.policy
            );
            assert!(
                has_diagnostic(&result, SHARED_BLOCK),
                "{kind:?} {:?}: must publish shared circular-block SE",
                query.policy
            );
            let se = result.estimate.se_bootstrap.expect("se_bootstrap");
            assert!(se.is_finite() && se > 0.0, "{kind:?} {:?}: positive block SE", query.policy);
            assert!(
                result
                    .diagnostics
                    .iter()
                    .all(|d| d.code.as_ref() != "estimate.graph_posterior.joint_if_se")
                    || result.estimate.influence.is_some(),
                "joint-IF diagnostic only when influence is published"
            );
        }
    }
}

#[test]
fn temporal_class_graph_posterior_shared_block_withheld_when_graph_dependent() {
    const SHARED_BLOCK: &str = "estimate.temporal_class.frequentist.shared_block";
    let gp = disagreeing_posterior(GraphPosteriorAtomKind::Cpdag);
    let result = run_on(
        &disagreeing_series(),
        gp,
        pulse(),
        InferenceMode::Frequentist,
        RefuteSuite::None,
        32,
    );
    assert!(has_policy(&result, StructuralAggregationPolicy::GraphDependentAtoms));
    assert!(!result.estimate.ate.is_finite());
    assert!(!has_diagnostic(&result, SHARED_BLOCK));
    assert!(result.estimate.se_bootstrap.is_none());
}

fn mediation_series(n: usize) -> TimeSeriesData {
    let mut t = vec![0.0; n];
    let mut m = vec![0.0; n];
    let mut y = vec![0.0; n];
    for (i, slot) in t.iter_mut().enumerate() {
        *slot = (0.071 * i as f64).sin() + 0.35 * (0.137 * i as f64).cos();
    }
    for i in 1..n {
        m[i] = 0.8 * t[i - 1] + 0.12 * (0.43 * i as f64).sin();
        y[i] = 0.25 * t[i - 1] + 0.55 * m[i] + 0.09 * (0.29 * i as f64).cos();
    }
    TimeSeriesData::from_f64_columns(
        [("t", t.as_slice()), ("m", m.as_slice()), ("y", y.as_slice())],
        1,
    )
    .unwrap()
}

fn mediation_query() -> MediationQuery {
    MediationQuery::binary(
        VariableId::from_raw(0),
        VariableId::from_raw(2),
        [VariableId::from_raw(1)],
        MediationContrast::Mediated,
    )
}

fn known_truth_cpdag_mediation_posterior(
    pin: &serde_json::Value,
    weights: &[f64],
) -> GraphPosterior {
    let identified_c = pin["identified_atom"]["contemporaneous_mask"].as_u64().unwrap();
    let unidentified_c = pin["unidentified_atom"]["contemporaneous_mask"].as_u64().unwrap();
    let identified_l = pin["identified_atom"]["lag_mask"].as_u64().unwrap();
    let unidentified_l = pin["unidentified_atom"]["lag_mask"].as_u64().unwrap();
    let contemporaneous_marginals = vec![0.0; 9];
    let lagged_marginals: Vec<f64> = pin["lagged_edge_marginals"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap())
        .collect();
    reconstruct_temporal_class(identified_c, identified_l, 3);
    GraphPosterior::new(
        3,
        weights.to_vec(),
        vec![identified_c, unidentified_c],
        contemporaneous_marginals.clone(),
        contemporaneous_marginals,
        1.0 / weights.iter().map(|w| w * w).sum::<f64>(),
        InferenceDiagnostics::analytic("known_truth_mixtures"),
        0,
    )
    .unwrap()
    .with_atom_kind(GraphPosteriorAtomKind::Cpdag)
    .with_lagged_marginals(1, lagged_marginals)
    .unwrap()
    .with_lag_masks(vec![identified_l, unidentified_l])
    .unwrap()
}

fn run_mediation(
    data: &TimeSeriesData,
    gp: GraphPosterior,
    query: MediationQuery,
    inference: InferenceMode,
    refute: RefuteSuite,
    bootstrap_replicates: u32,
) -> antecedent::StudyResult {
    Study::series(data.clone())
        .graph_posterior(gp)
        .query(CausalQuery::Mediation(query))
        .inference(inference)
        .refute(refute)
        .bootstrap_replicates(bootstrap_replicates)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(91))
        .unwrap()
}

#[test]
fn temporal_class_graph_posterior_mediation_envelope() {
    let expected: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/bayesian/known_truth_mixtures/expected.json"
    ))
    .unwrap();
    let pin = &expected["temporal_mediation"];
    let n = usize::try_from(pin["n"].as_u64().unwrap()).unwrap();
    let weights: Vec<f64> =
        pin["posterior_weights"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let effect_truth = pin["expected_effect_given_identified"].as_f64().unwrap();
    let unidentified_truth = pin["expected_unidentified_mass"].as_f64().unwrap();
    let tolerance = pin["effect_abs_tolerance"].as_f64().unwrap();
    let series = mediation_series(n);
    let gp = known_truth_cpdag_mediation_posterior(pin, &weights);
    let result = run_mediation(
        &series,
        gp,
        mediation_query(),
        InferenceMode::Frequentist,
        RefuteSuite::None,
        40,
    );
    assert_eq!(result.support_status.unwrap().as_str(), "licensed");
    assert!(
        (result.estimate.ate - effect_truth).abs() < tolerance,
        "mediated {} vs truth {effect_truth}",
        result.estimate.ate
    );
    assert_eq!(result.identification.status, IdentificationStatus::GraphDependent);
    assert!(has_policy(&result, StructuralAggregationPolicy::SameEstimandWeightedMean));
    let structural = result.structural_response.as_ref().expect("structural mixture");
    assert!((structural.unidentified_mass - unidentified_truth).abs() < 1e-12);
    assert!(result.estimate.se_bootstrap.is_some_and(|se| se.is_finite() && se > 0.0));
}

#[test]
fn temporal_class_graph_posterior_mediation_cheap_and_full_run() {
    let expected: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/bayesian/known_truth_mixtures/expected.json"
    ))
    .unwrap();
    let pin = &expected["temporal_mediation"];
    let weights: Vec<f64> =
        pin["posterior_weights"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let series = mediation_series(usize::try_from(pin["n"].as_u64().unwrap()).unwrap());
    let gp = known_truth_cpdag_mediation_posterior(pin, &weights);
    for inference in [
        InferenceMode::Frequentist,
        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(16)),
    ] {
        for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
            let result =
                run_mediation(&series, gp.clone(), mediation_query(), inference.clone(), suite, 0);
            assert_eq!(result.support_status.unwrap().as_str(), "licensed");
            assert!(
                result.diagnostics.iter().any(|d| {
                    d.code.as_ref() == "refute.envelope.temporal_class_mediation_posterior"
                }),
                "{inference:?} {suite:?} must request temporal class mediation refuters"
            );
            if matches!(inference, InferenceMode::Frequentist) {
                assert!(
                    !result.refutations.is_empty(),
                    "{inference:?} {suite:?} must run mediation refuters"
                );
            }
        }
    }
}

#[test]
fn bayesian_temporal_mediation_keeps_partial_mass_posterior_conditional() {
    let expected: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/bayesian/known_truth_mixtures/expected.json"
    ))
    .unwrap();
    let pin = &expected["temporal_mediation"];
    let gp = known_truth_cpdag_mediation_posterior(pin, &[0.8, 0.2]);
    let result = run_mediation(
        &mediation_series(320),
        gp,
        mediation_query(),
        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(32)),
        RefuteSuite::None,
        0,
    );
    assert!(result.estimate.ate.is_finite());
    assert!(result.posterior.is_none());
    assert_eq!(result.identification.status, IdentificationStatus::GraphDependent);
    assert!(
        result
            .diagnostics
            .iter()
            .any(|d| d.code.as_ref() == "estimate.graph_posterior.posterior_withheld")
    );
}

fn response_spec() -> TemporalResponseSpec {
    TemporalResponseSpec::new([1], TemporalPolicy::pulse(-1), Some(1)).unwrap()
}

fn mean_curve() -> CausalQuery {
    CausalQuery::Response(
        ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: VariableId::from_raw(1),
            treatment: ContinuousDomain::new(
                VariableId::from_raw(0),
                GridSpec::Values(std::sync::Arc::from([0.0, 1.0])),
            ),
        })
        .with_temporal(response_spec()),
    )
}

fn intervention_response() -> CausalQuery {
    CausalQuery::Response(
        ResponseQuery::new(ResponseFunctional::InterventionResponse {
            outcome: VariableId::from_raw(1),
            interventions: std::sync::Arc::from([Intervention::set(
                VariableId::from_raw(0),
                Value::f64(1.0),
            )]),
        })
        .with_temporal(response_spec()),
    )
}

fn run_response(
    gp: GraphPosterior,
    query: CausalQuery,
    inference: InferenceMode,
    refute: RefuteSuite,
) -> antecedent::StudyResult {
    Study::series(series())
        .graph_posterior(gp)
        .query(query)
        .inference(inference)
        .refute(refute)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(1))
        .unwrap()
}

#[test]
fn temporal_class_graph_posterior_response_mixes() {
    let identified = identified_lag_mask();
    reconstruct_temporal_class(0, identified, 3);
    reconstruct_temporal_class(visible_pag_contemp(), lag_bit(3, 0, 1), 3);
    for kind in [GraphPosteriorAtomKind::Cpdag, GraphPosteriorAtomKind::Pag] {
        let gp = if kind == GraphPosteriorAtomKind::Pag {
            class_posterior(kind, &[1.0], &[visible_pag_contemp()], &[lag_bit(3, 0, 1)])
        } else {
            class_posterior(kind, &[1.0], &[0], &[identified])
        };
        for query in [mean_curve(), intervention_response()] {
            for inference in [
                InferenceMode::Frequentist,
                InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(16)),
            ] {
                let result = run_response(gp.clone(), query.clone(), inference, RefuteSuite::None);
                assert_eq!(result.support_status.unwrap().as_str(), "licensed", "{kind:?}");
                assert!(result.response.is_some(), "{kind:?}");
                let mixture = result.structural_response.as_ref().expect("class response mixture");
                assert_eq!(mixture.weight_basis, StructuralWeightBasis::PosteriorProbability);
                assert!(
                    has_policy(&result, StructuralAggregationPolicy::SameEstimandWeightedMean)
                        || has_policy(&result, StructuralAggregationPolicy::IdentifiedSetEnvelope)
                        || has_policy(&result, StructuralAggregationPolicy::GraphDependentAtoms)
                );
                assert!(result.diagnostics.iter().any(|d| {
                    d.code.as_ref() == "estimate.response.temporal_class_graph_posterior"
                }));
            }
        }
    }
}

#[test]
fn temporal_class_graph_posterior_intervention_response_cheap_and_full_mix_atom_refuters() {
    let identified = identified_lag_mask();
    reconstruct_temporal_class(0, identified, 3);
    reconstruct_temporal_class(visible_pag_contemp(), lag_bit(3, 0, 1), 3);
    for kind in [GraphPosteriorAtomKind::Cpdag, GraphPosteriorAtomKind::Pag] {
        let gp = if kind == GraphPosteriorAtomKind::Pag {
            class_posterior(kind, &[1.0], &[visible_pag_contemp()], &[lag_bit(3, 0, 1)])
        } else {
            class_posterior(kind, &[1.0], &[0], &[identified])
        };
        for inference in [
            InferenceMode::Frequentist,
            InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(16)),
        ] {
            for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
                let result =
                    run_response(gp.clone(), intervention_response(), inference.clone(), suite);
                assert_eq!(result.support_status.unwrap().as_str(), "licensed", "{kind:?}");
                assert!(
                    !result.refutations.is_empty()
                        || result.diagnostics.iter().any(|d| {
                            d.code.as_ref() == "refute.envelope.temporal_class_graph_posterior"
                        }),
                    "{kind:?} {inference:?} {suite:?} missing Pulse-native IR reports: {:?}",
                    result.diagnostics.iter().map(|d| d.code.as_ref()).collect::<Vec<_>>()
                );
            }
        }
    }
}

#[test]
fn capped_temporal_class_posterior_does_not_claim_full_class_scope() {
    let gp = class_posterior(
        GraphPosteriorAtomKind::Cpdag,
        &[1.0],
        &[two_completion_contemp()],
        &[identified_lag_mask()],
    );
    let result = Study::series(series())
        .graph_posterior(gp)
        .query(pulse())
        .max_completions(1)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(1))
        .unwrap();
    let mixture = result.structural_response.as_ref().unwrap();
    assert!(mixture.truncated_atoms > 0);
    assert!(!mixture.full_mass_scope);
    assert_eq!(result.identification.status, IdentificationStatus::GraphDependent);
}
