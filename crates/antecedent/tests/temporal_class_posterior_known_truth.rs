//! Known-truth checks for temporal graph-posterior queries under every validation level.
//!
//! The expected values are the structural coefficients of the data-generating laws
//! recorded in `conformance/bayesian/known_truth_mixtures`: `defect_t = 0.9 *
//! pressure_{t-1}` (pulse, sustained and intervention-response effect 0.9), the mediated
//! path product `0.8 * 0.55 = 0.44`, and, for the PAG atoms that need a visibility
//! witness, the least-squares slope implied by `y_t = 1.4 t_{t-1} + w_t`. Each test
//! executes the cheap and full coordinates it cites and compares the published effect and
//! the retained unidentified posterior mass with those closed forms; no production
//! estimator is called to produce an expectation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::too_many_lines,
    reason = "test scaffolding builds deterministic series and compares closed-form constants"
)]

use std::sync::Arc;

use antecedent::{
    BayesianConfig, CausalQuery, InferenceMode, MediationContrast, RefuteSuite, Study,
};
use antecedent_core::{
    ExecutionContext, Intervention, MediationQuery, ResponseFunctional, ResponseQuery,
    ResponseValue, TemporalEffectQuery, TemporalPolicy, TemporalResponseSpec, Value, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_discovery::adjacency_mask::set_edge;
use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind, SamplingSummary};
use antecedent_prob::InferenceDiagnostics;

const CHECKED: [RefuteSuite; 2] = [RefuteSuite::Cheap, RefuteSuite::Full];

fn fixture() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../conformance/bayesian/known_truth_mixtures/expected.json"
    ))
    .unwrap()
}

fn floats(value: &serde_json::Value) -> Vec<f64> {
    value.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()
}

fn lag_bit(n: usize, from: usize, to: usize) -> u64 {
    1u64 << (from * n + to)
}

fn sampling(weights: &[f64]) -> SamplingSummary {
    SamplingSummary {
        ess: 1.0 / weights.iter().map(|w| w * w).sum::<f64>(),
        diagnostics: InferenceDiagnostics::analytic("temporal_class_posterior_known_truth"),
        rejected_invalid: 0,
    }
}

/// `pressure` iid uniform on [-1, 1], `defect_t = 0.9 pressure_{t-1}` (fixture generator).
fn pressure_defect_series(pin: &serde_json::Value) -> TimeSeriesData {
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

/// Structural coefficient of the PAG series: `y_t = PAG_COEFFICIENT t_{t-1} + w_t`.
const PAG_COEFFICIENT: f64 = 1.4;

/// `t` iid uniform on [-1, 1], `z` a distractor, `y_t = 1.4 t_{t-1} + 0.1 sin(1.7 s)` on 400 rows.
/// Returns the series, the exact least-squares slope of `y_t` on `t_{t-1}` implied by the
/// structural equation, `1.4 + cov(t_{t-1}, w) / var(t_{t-1})` (the atom keeps only
/// `t@-1 -> y`, so the identified adjustment is empty and the estimator is that slope), and a
/// Bayesian tolerance `5 * sigma_w / (sqrt(n) sd_x) / sqrt(draws) + 1e-3` derived from the
/// slope's sampling sd and the draw count. At horizon one the pulse and sustained queries
/// name the same estimand, so both are held to this one value.
fn pag_series(draws: usize) -> (TimeSeriesData, f64, f64) {
    let n = 400usize;
    let mut state = 12_345u64;
    let (mut t, mut y, mut z, mut w) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        t[i] = ((state >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0;
        z[i] = (i as f64 * 0.37).sin();
        w[i] = 0.1 * (i as f64 * 0.37 * 1.7).sin();
        if i > 0 {
            y[i] = PAG_COEFFICIENT * t[i - 1] + w[i];
        }
    }
    let rows = n - 1;
    let x: Vec<f64> = (0..rows).map(|i| t[i]).collect();
    let noise: Vec<f64> = (1..n).map(|i| w[i]).collect();
    let m = rows as f64;
    let (mx, mw) = (x.iter().sum::<f64>() / m, noise.iter().sum::<f64>() / m);
    let var_x = x.iter().map(|v| (v - mx).powi(2)).sum::<f64>() / m;
    let cov_xw = x.iter().zip(&noise).map(|(a, b)| (a - mx) * (b - mw)).sum::<f64>() / m;
    let var_w = noise.iter().map(|v| (v - mw).powi(2)).sum::<f64>() / m;
    let slope = PAG_COEFFICIENT + cov_xw / var_x;
    let sd_slope = (var_w / (m * var_x)).sqrt();
    let tolerance = 5.0 * sd_slope / (draws as f64).sqrt() + 1e-3;
    let data = TimeSeriesData::from_f64_columns(
        [("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())],
        1,
    )
    .unwrap();
    (data, slope, tolerance)
}

/// `T` (sinusoid) -> `M_t = 0.8 T_{t-1}`, `Y_t = 0.25 T_{t-1} + 0.55 M_t`.
fn mediation_series(n: usize) -> TimeSeriesData {
    let (mut t, mut m, mut y) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
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

fn visible_pag_contemp() -> u64 {
    // z -> t and z -> y at lag 0: unfolding puts an arrowhead into t@-1 (a visibility witness).
    set_edge(set_edge(0, 3, 2, 0, true), 3, 2, 1, true)
        .to_word()
        .expect("a three-variable graph fits one word")
}

/// Atoms read from the fixture's masks: the identified lag-one graph and the
/// autoregressive graph that stays unidentified (`NotCertified`).
fn fixture_posterior(
    kind: Option<GraphPosteriorAtomKind>,
    pin: &serde_json::Value,
    variables: usize,
) -> GraphPosterior {
    let weights = floats(&pin["posterior_weights"]);
    let contemporaneous = [
        pin["identified_atom"]["contemporaneous_mask"].as_u64().unwrap(),
        pin["unidentified_atom"]["contemporaneous_mask"].as_u64().unwrap(),
    ];
    let lags = [
        pin["identified_atom"]["lag_mask"].as_u64().unwrap(),
        pin["unidentified_atom"]["lag_mask"].as_u64().unwrap(),
    ];
    let cells = variables * variables;
    let mut posterior = GraphPosterior::from_atoms(
        variables,
        weights.clone(),
        contemporaneous.to_vec(),
        vec![0.0; cells],
        vec![0.0; cells],
        sampling(&weights),
    )
    .unwrap();
    if let Some(kind) = kind {
        posterior = posterior.with_atom_kind(kind);
    }
    posterior
        .with_lagged_marginals(1, floats(&pin["lagged_edge_marginals"]))
        .unwrap()
        .with_lag_masks(lags.to_vec())
        .unwrap()
}

fn pag_posterior(weights: &[f64]) -> GraphPosterior {
    let identified = lag_bit(3, 0, 1);
    let autoregressive = identified | lag_bit(3, 0, 0);
    GraphPosterior::from_atoms(
        3,
        weights.to_vec(),
        vec![visible_pag_contemp(), visible_pag_contemp()],
        vec![0.0; 9],
        vec![0.0; 9],
        sampling(weights),
    )
    .unwrap()
    .with_atom_kind(GraphPosteriorAtomKind::Pag)
    .with_lagged_marginals(1, vec![0.0; 9])
    .unwrap()
    .with_lag_masks(vec![identified, autoregressive])
    .unwrap()
}

fn pulse() -> TemporalEffectQuery {
    TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1)
        .with_max_history_lag(Some(1))
}

fn sustained() -> TemporalEffectQuery {
    TemporalEffectQuery::sustained(VariableId::from_raw(0), VariableId::from_raw(1), 0, 1.0)
        .with_policy(TemporalPolicy::sustained(-1, -1))
        .with_horizon_steps(1)
        .with_max_history_lag(Some(1))
}

fn run(
    data: &TimeSeriesData,
    gp: GraphPosterior,
    query: CausalQuery,
    inference: InferenceMode,
    suite: RefuteSuite,
    seed: u64,
) -> antecedent::StudyResult {
    Study::series(data.clone())
        .graph_posterior(gp)
        .query(query)
        .inference(inference)
        .refute(suite)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(seed))
        .unwrap()
}

/// The built study `run` executes, for a test that also pins the cell it compiles to.
fn posterior_study(
    data: &TimeSeriesData,
    gp: GraphPosterior,
    query: CausalQuery,
    inference: InferenceMode,
    suite: RefuteSuite,
) -> Study {
    Study::series(data.clone())
        .graph_posterior(gp)
        .query(query)
        .inference(inference)
        .refute(suite)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
}

/// The support-matrix cell the study compiles to (`query:graph_class:structure:inference:
/// validation`), so a test pins the coordinate it executes rather than inferring it.
fn compiled_cell(study: &Study) -> String {
    study
        .compile(&ExecutionContext::for_tests(0))
        .unwrap()
        .lowered()
        .expect("compile attaches a lowered program")
        .scientific()
        .support_coordinate()
        .expect("the study is on the support-matrix axis")
        .to_owned()
}

/// The matrix validation axis of a checked suite.
fn matrix_validation(suite: RefuteSuite) -> &'static str {
    if suite == RefuteSuite::Full { "full" } else { "cheap" }
}

fn bayes(draws: usize) -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(draws).prior_scale(1_000_000.0))
}

fn unidentified_mass(result: &antecedent::StudyResult) -> f64 {
    result.structural_response.as_ref().expect("structural mixture").unidentified_mass
}

// ----------------------------------------------------------------------- Pulse / Sustained

/// `PulseEffect` and `SustainedEffect` x `{TemporalCpdag, TemporalPag}` x
/// `graph_posterior` x both inferences x `{cheap, full}`: the published effect equals
/// the structural coefficient (0.9 on the fixture series, 1.4 on the PAG series) and the
/// autoregressive atom's mass stays unidentified at the fixture's 0.3.
#[test]
fn temporal_class_pulse_and_sustained_match_the_structural_coefficient_under_cheap_and_full() {
    let fixture = fixture();
    let pin = &fixture["temporal_effect"];
    let effect_truth = pin["expected_effect_given_identified"].as_f64().unwrap();
    let unidentified_truth = pin["expected_unidentified_mass"].as_f64().unwrap();
    let band = pin["effect_abs_tolerance"].as_f64().unwrap();
    let weights = floats(&pin["posterior_weights"]);
    let cpdag_data = pressure_defect_series(pin);
    let (pag_data, pag_truth, pag_bayes_tolerance) = pag_series(64);
    for query in [pulse(), sustained()] {
        for (mode, inference) in
            [("frequentist", InferenceMode::Frequentist), ("bayesian", bayes(64))]
        {
            for suite in CHECKED {
                let label = format!("{:?} {mode} {suite:?}", query.policy);

                let result = run(
                    &cpdag_data,
                    fixture_posterior(Some(GraphPosteriorAtomKind::Cpdag), pin, 2),
                    CausalQuery::TemporalEffect(query.clone()),
                    inference.clone(),
                    suite,
                    1,
                );
                let ate = result.estimate().as_effect().unwrap().ate;
                assert!((ate - effect_truth).abs() < band, "TemporalCpdag {label}: {ate}");
                assert!(
                    (unidentified_mass(&result) - unidentified_truth).abs() < 1e-12,
                    "TemporalCpdag {label}: unidentified mass"
                );
                assert!(!result.refutations.is_empty(), "TemporalCpdag {label}: refuters ran");

                let result = run(
                    &pag_data,
                    pag_posterior(&weights),
                    CausalQuery::TemporalEffect(query.clone()),
                    inference.clone(),
                    suite,
                    1,
                );
                let ate = result.estimate().as_effect().unwrap().ate;
                let pag_tolerance = if mode == "bayesian" { pag_bayes_tolerance } else { 1e-6 };
                assert!(
                    (ate - pag_truth).abs() < pag_tolerance,
                    "TemporalPag {label}: {ate} vs structural slope {pag_truth}"
                );
                assert!(
                    (unidentified_mass(&result) - unidentified_truth).abs() < 1e-12,
                    "TemporalPag {label}: unidentified mass"
                );
                assert!(!result.refutations.is_empty(), "TemporalPag {label}: refuters ran");
            }
        }
    }
}

// ----------------------------------------------------------------------- Mediation

/// `TemporalMediationEffect x TemporalCpdag x graph_posterior`: the mediated path
/// product 0.8 * 0.55 = 0.44 under cheap and full, both inferences.
#[test]
fn temporal_class_mediation_matches_the_path_product_under_cheap_and_full() {
    let fixture = fixture();
    let pin = &fixture["temporal_mediation"];
    let effect_truth = pin["expected_effect_given_identified"].as_f64().unwrap();
    let unidentified_truth = pin["expected_unidentified_mass"].as_f64().unwrap();
    let band = pin["effect_abs_tolerance"].as_f64().unwrap();
    let data = mediation_series(usize::try_from(pin["n"].as_u64().unwrap()).unwrap());
    let query = MediationQuery::binary(
        VariableId::from_raw(0),
        VariableId::from_raw(2),
        [VariableId::from_raw(1)],
        MediationContrast::Mediated,
    );
    for (mode, inference) in [("Frequentist", InferenceMode::Frequentist), ("Bayesian", bayes(256))]
    {
        for suite in CHECKED {
            let label = format!("{mode} {suite:?}");
            let built = posterior_study(
                &data,
                fixture_posterior(Some(GraphPosteriorAtomKind::Cpdag), pin, 3),
                CausalQuery::Mediation(query.clone()),
                inference.clone(),
                suite,
            );
            assert_eq!(
                compiled_cell(&built),
                format!(
                    "TemporalMediationEffect:TemporalCpdag:graph_posterior:{mode}:{}",
                    matrix_validation(suite)
                ),
                "{label}: the class-atom posterior runs the TemporalCpdag cell"
            );
            let result = built.run(&ExecutionContext::for_tests(91)).unwrap();
            let ate = result.estimate().as_effect().unwrap().ate;
            assert!((ate - effect_truth).abs() < band, "{label}: mediated {ate} vs {effect_truth}");
            assert!(
                (unidentified_mass(&result) - unidentified_truth).abs() < 1e-12,
                "{label}: unidentified mass"
            );
        }
    }
}

// ----------------------------------------------------------------------- Response

/// `InterventionResponse x TemporalDag x graph_posterior x both inferences x
/// {cheap, full}`: `do(pressure = 1)` at lag one gives `defect = 0.9`.
#[test]
fn temporal_dag_graph_posterior_intervention_response_matches_the_structural_level() {
    let fixture = fixture();
    let pin = &fixture["temporal_effect"];
    let effect_truth = pin["expected_effect_given_identified"].as_f64().unwrap();
    let unidentified_truth = pin["expected_unidentified_mass"].as_f64().unwrap();
    let band = pin["effect_abs_tolerance"].as_f64().unwrap();
    let data = pressure_defect_series(pin);
    let q = pulse();
    let response = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: q.outcome,
        interventions: Arc::from([Intervention::set(q.treatment, Value::f64(1.0))]),
    })
    .with_temporal(
        TemporalResponseSpec::new(vec![q.horizon_steps], q.policy.clone(), q.max_history_lag)
            .unwrap(),
    );
    for (mode, inference) in [("Frequentist", InferenceMode::Frequentist), ("Bayesian", bayes(64))]
    {
        for suite in CHECKED {
            let label = format!("{mode} {suite:?}");
            let built = posterior_study(
                &data,
                fixture_posterior(None, pin, 2),
                CausalQuery::Response(response.clone()),
                inference.clone(),
                suite,
            );
            assert_eq!(
                compiled_cell(&built),
                format!(
                    "InterventionResponse:TemporalDag:graph_posterior:{mode}:{}",
                    matrix_validation(suite)
                ),
                "{label}: the DAG-atom posterior on a series runs the TemporalDag cell"
            );
            let result = built.run(&ExecutionContext::for_tests(11)).unwrap();
            let atoms: Vec<f64> = result
                .structural_response
                .as_ref()
                .expect("structural response")
                .atoms
                .iter()
                .filter_map(|atom| match atom.value.as_ref()? {
                    ResponseValue::Scalar(v) => Some(*v),
                    ResponseValue::Surface { mean, .. } => mean.first().copied(),
                    _ => None,
                })
                .collect();
            assert_eq!(atoms.len(), 1, "{label}: exactly the lag-one atom identifies");
            assert!(
                (atoms[0] - effect_truth).abs() < band,
                "{label}: level {} vs {effect_truth}",
                atoms[0]
            );
            assert!(
                (unidentified_mass(&result) - unidentified_truth).abs() < 1e-12,
                "{label}: unidentified mass"
            );
            assert!(!result.refutations.is_empty(), "{label}: refuters ran");
        }
    }
}
