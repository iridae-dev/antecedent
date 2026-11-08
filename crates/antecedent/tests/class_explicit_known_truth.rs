//! Known-truth checks for explicit and accepted CPDAG/PAG static queries.
//!
//! Expected values are the closed forms recorded in
//! `conformance/bayesian/known_truth_mixtures`: the confounded contrast 3 and the
//! Z-adjusted coefficient 2 of `Y = 2T + 2Z` (a CPDAG whose `Z - T` edge has two
//! equally weighted completions publishes their mean, 2.5), and the response curves
//! `1 + 2 t` on a design where `Z` is independent of `T`. Every (graph class x
//! structure x inference x validation) coordinate cited is executed; no production
//! estimator is called to produce an expectation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::too_many_lines,
    reason = "test scaffolding keeps one scenario per body and compares closed-form constants"
)]

use std::sync::Arc;

use antecedent::{AcceptedGraph, BayesianConfig, InferenceMode, RefuteSuite, Study, StudyBuilder};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ConditionalEffectQuery, ContinuousDomain, ExecutionContext,
    GridSpec, Intervention, ResponseFunctional, ResponseIdentification, ResponseQuery,
    ResponseValue, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Cpdag, DenseNodeId, Pag};

const SUITES: [RefuteSuite; 3] = [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full];

fn fixture() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../conformance/bayesian/known_truth_mixtures/expected.json"
    ))
    .unwrap()
}

fn floats(value: &serde_json::Value) -> Vec<f64> {
    value.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()
}

fn d(i: u32) -> DenseNodeId {
    DenseNodeId::from_raw(i)
}

/// Confounded `(z, t)` design: `Y = 2T + 2Z +/- 0.2`, counts 6/2/2/6.
fn confounded_data(n: usize) -> TabularData {
    let (mut treatment, mut outcome, mut confounder) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..(n / 16) {
        for (z, t, count) in [(0.0, 0.0, 6), (0.0, 1.0, 2), (1.0, 0.0, 2), (1.0, 1.0, 6)] {
            for row in 0..count {
                let epsilon = if row % 2 == 0 { -0.2 } else { 0.2 };
                treatment.push(t);
                confounder.push(z);
                outcome.push(2.0 * t + 2.0 * z + epsilon);
            }
        }
    }
    TabularData::from_f64_columns([
        ("t", treatment.as_slice()),
        ("y", outcome.as_slice()),
        ("z", confounder.as_slice()),
    ])
    .unwrap()
}

/// `Z` independent of `T`, `Y = 2T + 2Z +/- 0.2`: the effect is 2 under every adjustment set.
fn balanced_data(n: usize) -> TabularData {
    let (mut treatment, mut outcome, mut covariate) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..(n / 8) {
        for (z, t) in [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)] {
            for row in 0..2 {
                let epsilon = if row % 2 == 0 { -0.2 } else { 0.2 };
                treatment.push(t);
                covariate.push(z);
                outcome.push(2.0 * t + 2.0 * z + epsilon);
            }
        }
    }
    TabularData::from_f64_columns([
        ("t", treatment.as_slice()),
        ("y", outcome.as_slice()),
        ("z", covariate.as_slice()),
    ])
    .unwrap()
}

/// The fixture's `static_response` design with `Z` independent of `T`:
/// `E[Y | do(t)] = 1 + 2t` for every adjustment set.
fn independent_response_data(pin: &serde_json::Value) -> TabularData {
    let levels = usize::try_from(pin["treatment_levels"].as_u64().unwrap()).unwrap();
    let span = pin["treatment_span"].as_f64().unwrap();
    let offset = pin["z_offset"].as_f64().unwrap();
    let noise = pin["outcome_noise"].as_f64().unwrap();
    let (mut t, mut y, mut z) = (Vec::new(), Vec::new(), Vec::new());
    for k in 0..levels {
        let tv = -span + 2.0 * span * k as f64 / (levels - 1) as f64;
        for (z_sign, e_sign) in [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)] {
            let zv = z_sign * offset;
            t.push(tv);
            z.push(zv);
            y.push(1.0 + 2.0 * tv + 1.5 * zv + e_sign * noise);
        }
    }
    TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())])
        .unwrap()
}

/// `z - t` undirected with `z -> y`, `t -> y`: completions `z -> t` (effect 2) and `t -> z` (3).
fn two_completion_cpdag() -> Cpdag {
    let mut g = Cpdag::with_variables(3);
    g.insert_directed(d(2), d(1)).unwrap();
    g.insert_directed(d(0), d(1)).unwrap();
    g.insert_undirected(d(2), d(0)).unwrap();
    g
}

/// `z -> t -> y` with `z` not adjacent to `y`: a visible edge, one completion.
fn visible_pag() -> Pag {
    let mut g = Pag::with_variables(3);
    g.insert_directed(d(2), d(0)).unwrap();
    g.insert_directed(d(0), d(1)).unwrap();
    g
}

#[derive(Clone)]
enum Class {
    Cpdag(Cpdag),
    Pag(Pag),
}

fn study(
    data: &TabularData,
    class: &Class,
    accepted: bool,
    query: CausalQuery,
    inference: InferenceMode,
    suite: RefuteSuite,
) -> StudyBuilder {
    let builder = Study::tabular(data.clone());
    let builder = match (class, accepted) {
        (Class::Cpdag(g), true) => builder.graph(AcceptedGraph::from(g.clone())),
        (Class::Cpdag(g), false) => builder.graph(g.clone()),
        (Class::Pag(g), true) => builder.graph(AcceptedGraph::from(g.clone())),
        (Class::Pag(g), false) => builder.graph(g.clone()),
    };
    builder.query(query).inference(inference).refute(suite).bootstrap_replicates(0)
}

fn execute(builder: StudyBuilder) -> antecedent::StudyResult {
    builder.build().unwrap().run(&ExecutionContext::for_tests(3)).unwrap()
}

fn bayes(draws: usize) -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(draws).prior_scale(1_000_000.0))
}

fn response_values(result: &antecedent::StudyResult) -> Vec<f64> {
    if let Some(mixture) = result.structural_response.as_ref() {
        let atoms: Vec<(f64, &ResponseValue)> = mixture
            .atoms
            .iter()
            .filter_map(|atom| atom.value.as_ref().map(|value| (atom.weight, value)))
            .collect();
        let total: f64 = atoms.iter().map(|(w, _)| w).sum();
        let width = atoms.first().map_or(0, |(_, v)| points(v).len());
        if width > 0 {
            return (0..width)
                .map(|i| atoms.iter().map(|(w, v)| w * points(v)[i]).sum::<f64>() / total)
                .collect();
        }
    }
    match &result.response().expect("response payload").estimate {
        ResponseIdentification::PointIdentified(value)
        | ResponseIdentification::PartiallyIdentified(value) => points(value),
        other => panic!("unexpected identification payload {other:?}"),
    }
}

fn points(value: &ResponseValue) -> Vec<f64> {
    match value {
        ResponseValue::Scalar(v) => vec![*v],
        ResponseValue::Surface { mean, .. } => mean.to_vec(),
        other => panic!("unexpected response value {other:?}"),
    }
}

fn assert_validation_ran(label: &str, result: &antecedent::StudyResult, suite: RefuteSuite) {
    match suite {
        RefuteSuite::None => {
            assert!(result.refutations.is_empty(), "{label}: none emits no refuters")
        }
        _ => assert!(!result.refutations.is_empty(), "{label}: refuters must execute"),
    }
}

/// `AverageEffect x {Cpdag, Pag} x {explicit, accepted} x {Frequentist, Bayesian} x
/// {none, cheap, full}`. The two-completion CPDAG publishes the mean of the closed-form
/// completion effects 3 and 2; the visible PAG on the Z-independent design publishes 2.
#[test]
fn class_average_effect_matches_the_closed_forms_explicit_and_accepted() {
    let fixture = fixture();
    let pin = &fixture["static_average_effect"];
    let n = usize::try_from(pin["n"].as_u64().unwrap()).unwrap();
    let effects = floats(&pin["identified_atom_effects"]);
    let completion_mean = f64::midpoint(effects[0], effects[1]);
    let structural = effects[1];
    let band = pin["effect_abs_tolerance"].as_f64().unwrap();
    let query = CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(
        VariableId::from_raw(0),
        VariableId::from_raw(1),
    ));
    let confounded = confounded_data(n);
    let balanced = balanced_data(n);
    let cases = [
        ("Cpdag", Class::Cpdag(two_completion_cpdag()), &confounded, completion_mean),
        ("Pag", Class::Pag(visible_pag()), &balanced, structural),
    ];
    for (name, class, data, truth) in cases {
        for accepted in [false, true] {
            for (mode, inference, tolerance) in [
                ("frequentist", InferenceMode::Frequentist, 1e-6),
                ("bayesian", bayes(1024), 1.3 * band),
            ] {
                for suite in SUITES {
                    let label = format!("{name} accepted={accepted} {mode} {suite:?}");
                    let result = execute(study(
                        data,
                        &class,
                        accepted,
                        query.clone(),
                        inference.clone(),
                        suite,
                    ));
                    let ate = result.estimate().as_effect().unwrap().ate;
                    assert!((ate - truth).abs() < tolerance, "{label}: ate {ate} vs {truth}");
                    assert_validation_ran(&label, &result, suite);
                }
            }
        }
    }
}

/// Modifier-heterogeneous design: `Y = 2T + 2Z + T*Z +/- 0.2`, so the effect of `T` at modifier
/// value `z` is `tau(z) = 2 + z` (the fixture's structural slope 2 plus an interaction of 1).
/// `confounded` uses the fixture's 6/2/2/6 `(z, t)` counts, otherwise `Z` is independent of
/// `T`; the `z = 1` cells are repeated `z1_scale` times, which moves `P(z = 1)`. The
/// modifier-averaged conditional effect is `tau(0) + P(z = 1) = 2 + P(z = 1)`, so two mixes
/// recover `tau(0)` and `tau(1)` and a modifier-free ATE (crude or adjusted) cannot match both.
/// Exact equality of two indicator values (the design only holds literal 0/1 levels).
fn same(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits()
}

fn heterogeneous_data(confounded: bool, z1_scale: usize) -> (TabularData, f64) {
    let counts: [(f64, f64, usize); 4] = if confounded {
        [(0.0, 0.0, 6), (0.0, 1.0, 2), (1.0, 0.0, 2), (1.0, 1.0, 6)]
    } else {
        [(0.0, 0.0, 4), (0.0, 1.0, 4), (1.0, 0.0, 4), (1.0, 1.0, 4)]
    };
    let (mut t, mut y, mut z) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..20 {
        for (zi, ti, count) in counts {
            let copies = if same(zi, 1.0) { z1_scale } else { 1 };
            for row in 0..(count * copies) {
                let epsilon = if row % 2 == 0 { -0.2 } else { 0.2 };
                t.push(ti);
                z.push(zi);
                y.push(2.0 * ti + 2.0 * zi + ti * zi + epsilon);
            }
        }
    }
    let share = z.iter().sum::<f64>() / z.len() as f64;
    (
        TabularData::from_f64_columns([
            ("t", t.as_slice()),
            ("y", y.as_slice()),
            ("z", z.as_slice()),
        ])
        .unwrap(),
        share,
    )
}

/// `ConditionalEffect x {Cpdag, Pag} x {explicit, accepted} x {Frequentist, Bayesian} x
/// {none, cheap, full}` on a modifier-heterogeneous design: the published value is
/// `tau(0) + P(z = 1)` at each of two modifier mixes, so both `tau(0) = 2` and `tau(1) = 3`
/// are recovered and the two mixes differ by the closed-form 0.25; a modifier-free ATE
/// (crude 2.75 on the confounded design, or the adjusted 2.5) cannot satisfy both.
#[test]
fn class_conditional_effect_matches_the_structural_slope_explicit_and_accepted() {
    let fixture = fixture();
    let pin = &fixture["static_average_effect"];
    let tau0 = floats(&pin["identified_atom_effects"])[1];
    let query = CausalQuery::ConditionalEffect(
        ConditionalEffectQuery::try_new(
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
                .with_effect_modifiers([VariableId::from_raw(2)]),
        )
        .unwrap(),
    );
    for (name, class, confounded) in [
        ("Cpdag", Class::Cpdag(two_completion_cpdag()), true),
        ("Pag", Class::Pag(visible_pag()), false),
    ] {
        for accepted in [false, true] {
            for (mode, inference, tolerance) in
                [("frequentist", InferenceMode::Frequentist, 1e-6), ("bayesian", bayes(256), 0.01)]
            {
                for suite in SUITES {
                    let mut values = Vec::new();
                    for z1_scale in [1usize, 3] {
                        let (data, share) = heterogeneous_data(confounded, z1_scale);
                        let label =
                            format!("{name} accepted={accepted} {mode} {suite:?} scale={z1_scale}");
                        let result = execute(study(
                            &data,
                            &class,
                            accepted,
                            query.clone(),
                            inference.clone(),
                            suite,
                        ));
                        let ate = result.estimate().as_effect().unwrap().ate;
                        assert!(
                            (ate - (tau0 + share)).abs() < tolerance,
                            "{label}: effect {ate} vs {}",
                            tau0 + share
                        );
                        assert_validation_ran(&label, &result, suite);
                        values.push(ate);
                    }
                    assert!(
                        (values[1] - values[0] - 0.25).abs() < 2.0 * tolerance,
                        "{name} accepted={accepted} {mode} {suite:?}: modifier dependence {values:?}"
                    );
                }
            }
        }
    }
}

/// `InterventionResponse` and `ResponseCurve` x `{Cpdag, Pag}` x `{explicit, accepted}` x
/// `{Frequentist, Bayesian}` x none: every completion of the class gives the closed-form
/// curve `1 + 2t` on the Z-independent design.
#[test]
fn class_responses_match_the_closed_form_curve_explicit_and_accepted() {
    let fixture = fixture();
    let pin = &fixture["static_response"];
    let grid = floats(&pin["treatment_grid"]);
    let adjusted = floats(&pin["identified_atom_curve_adjusted"]);
    let level = pin["intervention_level"].as_f64().unwrap();
    let index = grid.iter().position(|&v| (v - level).abs() < 1e-12).unwrap();
    let data = independent_response_data(pin);
    let draws = usize::try_from(pin["bayesian_n_draws"].as_u64().unwrap()).unwrap();
    let curve = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(grid.clone().into()),
        ),
    });
    let at_level = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(1),
        interventions: Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(level))]),
    });
    let modes = [
        (
            "frequentist",
            InferenceMode::Frequentist,
            pin["frequentist_abs_tolerance"].as_f64().unwrap(),
        ),
        (
            "bayesian",
            InferenceMode::Bayesian(
                BayesianConfig::conjugate()
                    .n_draws(draws)
                    .prior_scale(pin["bayesian_prior_scale"].as_f64().unwrap()),
            ),
            pin["bayesian_abs_tolerance"].as_f64().unwrap(),
        ),
    ];
    for (name, class) in
        [("Cpdag", Class::Cpdag(two_completion_cpdag())), ("Pag", Class::Pag(visible_pag()))]
    {
        for accepted in [false, true] {
            for (mode, inference, tolerance) in &modes {
                for (query_label, query, truth) in [
                    ("ResponseCurve", curve.clone(), adjusted.clone()),
                    ("InterventionResponse", at_level.clone(), vec![adjusted[index]]),
                ] {
                    let label = format!("{name} accepted={accepted} {mode} {query_label}");
                    let result = execute(study(
                        &data,
                        &class,
                        accepted,
                        CausalQuery::Response(query),
                        inference.clone(),
                        RefuteSuite::None,
                    ));
                    let got = response_values(&result);
                    assert_eq!(got.len(), truth.len(), "{label}");
                    for (g, want) in got.iter().zip(&truth) {
                        assert!((g - want).abs() < *tolerance, "{label}: {g} vs {want}");
                    }
                }
            }
        }
    }
}
