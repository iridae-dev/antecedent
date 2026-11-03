//! Known-truth checks for CPDAG/PAG (and DAG) graph-posterior static queries.
//!
//! Every expected value comes from the closed forms recorded in
//! `conformance/bayesian/known_truth_mixtures`: the structural coefficient 2 of
//! `Y = 2T + 2Z`, the confounded contrast 3, and the response curves
//! `1 + 2.75 t` (direct) and `1 + 2 t` (Z-adjusted). The tests execute every
//! (graph class x inference x validation) coordinate they cite and compare each
//! atom's published value, and the retained unidentified posterior mass, with
//! those closed forms. No production estimator is called to produce an
//! expectation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    clippy::too_many_lines,
    reason = "test scaffolding compares closed-form constants and keeps one scenario per body"
)]

use std::sync::Arc;

use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ConditionalEffectQuery, ContinuousDomain, ExecutionContext,
    GridSpec, Intervention, ResponseFunctional, ResponseQuery, ResponseValue, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_discovery::adjacency_mask::{
    adjacency_mask_from_cpdag, adjacency_masks_from_pag, set_edge,
};
use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind, SamplingSummary};
use antecedent_graph::{Cpdag, Dag, DenseNodeId, Pag};
use antecedent_prob::InferenceDiagnostics;

const SUITES: [RefuteSuite; 3] = [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full];

fn pin() -> serde_json::Value {
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

fn sampling(weights: &[f64]) -> SamplingSummary {
    SamplingSummary {
        ess: 1.0 / weights.iter().map(|w| w * w).sum::<f64>(),
        diagnostics: InferenceDiagnostics::analytic("class_posterior_known_truth"),
        rejected_invalid: 0,
    }
}

// ----------------------------------------------------------------------- designs

/// Confounded `(z, t)` design of the fixture: `Y = 2T + 2Z +/- 0.2`, counts 6/2/2/6.
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

/// `Z` independent of `T` (equal cell counts), `Y = 2T + 2Z +/- 0.2`: the
/// structural coefficient 2 is the effect under every adjustment set.
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

/// The fixture's `static_response` design; `z_slope` is the fixture's 0.5
/// (confounded) or 0 (Z independent of T, so `E[Y | do(t)] = 1 + 2t`).
fn response_data(pin: &serde_json::Value, z_slope: f64) -> TabularData {
    let levels = usize::try_from(pin["treatment_levels"].as_u64().unwrap()).unwrap();
    let span = pin["treatment_span"].as_f64().unwrap();
    let offset = pin["z_offset"].as_f64().unwrap();
    let noise = pin["outcome_noise"].as_f64().unwrap();
    let (mut t, mut y, mut z) = (Vec::new(), Vec::new(), Vec::new());
    for k in 0..levels {
        let tv = -span + 2.0 * span * k as f64 / (levels - 1) as f64;
        for (z_sign, e_sign) in [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)] {
            let zv = z_slope * tv + z_sign * offset;
            t.push(tv);
            z.push(zv);
            y.push(1.0 + 2.0 * tv + 1.5 * zv + e_sign * noise);
        }
    }
    TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())])
        .unwrap()
}

// ----------------------------------------------------------------------- graphs

fn dag_cpdag(edges: &[(u32, u32)]) -> Cpdag {
    let mut g = Dag::with_variables(3);
    for &(a, b) in edges {
        g.insert_directed(d(a), d(b)).unwrap();
    }
    Cpdag::from_dag(&g)
}

fn cpdag_posterior(weights: &[f64], graphs: &[Cpdag]) -> GraphPosterior {
    let masks: Vec<_> = graphs.iter().map(|g| adjacency_mask_from_cpdag(g).unwrap()).collect();
    GraphPosterior::from_atoms(
        3,
        weights.to_vec(),
        masks,
        vec![0.0; 9],
        vec![0.0; 9],
        sampling(weights),
    )
    .unwrap()
    .with_atom_kind(GraphPosteriorAtomKind::Cpdag)
}

fn pag_of(edges: &[(u32, u32)]) -> Pag {
    let mut g = Pag::with_variables(3);
    for &(a, b) in edges {
        g.insert_directed(d(a), d(b)).unwrap();
    }
    g
}

fn pag_posterior(weights: &[f64], graphs: &[Pag]) -> GraphPosterior {
    let (mut masks, mut marks) = (Vec::new(), Vec::new());
    for graph in graphs {
        let (adj, mark) = adjacency_masks_from_pag(graph).unwrap();
        masks.push(adj);
        marks.push(mark);
    }
    GraphPosterior::from_atoms(
        3,
        weights.to_vec(),
        masks,
        vec![0.0; 9],
        vec![0.0; 9],
        sampling(weights),
    )
    .unwrap()
    .with_atom_kind(GraphPosteriorAtomKind::Pag)
    .with_mark_atoms(marks)
    .unwrap()
}

/// The fixture's three disagreeing atoms: `T -> Y` (direct), `Z -> T, Z -> Y, T -> Y`
/// (adjusted) and the reverse-causal `Y -> T` (unidentified), as CPDAG atoms.
fn cpdag_atoms(weights: &[f64]) -> GraphPosterior {
    cpdag_posterior(
        weights,
        &[dag_cpdag(&[(0, 1)]), dag_cpdag(&[(0, 1), (2, 0), (2, 1)]), dag_cpdag(&[(1, 0)])],
    )
}

/// One identified PAG atom (`Z -> T -> Y`, `Z` not adjacent to `Y`) and the
/// unidentified reverse-causal `Y -> T`.
fn pag_atoms(weights: &[f64]) -> GraphPosterior {
    pag_posterior(weights, &[pag_of(&[(2, 0), (0, 1)]), pag_of(&[(1, 0)])])
}

/// The same three atoms as the DAG graph-posterior fixture.
fn dag_atoms(weights: &[f64]) -> GraphPosterior {
    let direct = set_edge(0, 3, 0, 1, true);
    let adjusted = set_edge(set_edge(set_edge(0, 3, 0, 1, true), 3, 2, 0, true), 3, 2, 1, true);
    let unidentified = set_edge(0, 3, 1, 0, true);
    GraphPosterior::from_atoms(
        3,
        weights.to_vec(),
        vec![direct, adjusted, unidentified],
        vec![0.0; 9],
        vec![0.0; 9],
        sampling(weights),
    )
    .unwrap()
}

// ----------------------------------------------------------------------- helpers

fn bayes(draws: usize, prior_scale: f64) -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(draws).prior_scale(prior_scale))
}

fn run(
    data: &TabularData,
    gp: GraphPosterior,
    query: CausalQuery,
    inference: InferenceMode,
    suite: RefuteSuite,
    seed: u64,
) -> antecedent::StudyResult {
    Study::tabular(data.clone())
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

fn values_of(value: &ResponseValue) -> Vec<f64> {
    match value {
        ResponseValue::Scalar(v) => vec![*v],
        ResponseValue::Surface { mean, .. } => mean.to_vec(),
        other => panic!("expected a scalar or surface response, got {other:?}"),
    }
}

/// Values of the atoms that identified the query, in publication order.
fn identified_atoms(result: &antecedent::StudyResult) -> Vec<Vec<f64>> {
    result
        .structural_response
        .as_ref()
        .expect("structural response")
        .atoms
        .iter()
        .filter_map(|atom| atom.value.as_ref().map(values_of))
        .collect()
}

fn assert_matches_some_atom(label: &str, atoms: &[Vec<f64>], truth: &[f64], tolerance: f64) {
    assert!(
        atoms.iter().any(|atom| {
            atom.len() == truth.len()
                && atom.iter().zip(truth).all(|(got, want)| (got - want).abs() < tolerance)
        }),
        "{label}: no atom matches the closed form {truth:?}; atoms {atoms:?}"
    );
}

fn assert_unidentified_mass(label: &str, result: &antecedent::StudyResult, truth: f64) {
    let mass = result.structural_response.as_ref().unwrap().unidentified_mass;
    assert!((mass - truth).abs() < 1e-9, "{label}: unidentified mass {mass} vs {truth}");
}

fn assert_validation_ran(label: &str, result: &antecedent::StudyResult, suite: RefuteSuite) {
    match suite {
        RefuteSuite::None => {
            assert!(result.refutations.is_empty(), "{label}: none must emit no refuters");
        }
        _ => assert!(!result.refutations.is_empty(), "{label}: refuters must execute"),
    }
}

fn level_query(level: f64) -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(1),
        interventions: Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(level))]),
    })
}

fn curve_query(grid: &[f64]) -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(grid.to_vec().into()),
        ),
    })
}

fn inferences(bayesian_tolerance: f64, frequentist_tolerance: f64) -> Vec<(&'static str, InferenceMode, f64)> {
    vec![
        ("frequentist", InferenceMode::Frequentist, frequentist_tolerance),
        ("bayesian", bayes(256, 1_000_000.0), bayesian_tolerance),
    ]
}

// ----------------------------------------------------------------------- AverageEffect

/// `AverageEffect x {Cpdag, Pag} x graph_posterior x {Frequentist, Bayesian} x
/// {none, cheap, full}`: the confounded CPDAG atoms publish the closed-form
/// effects 3 (direct) and 2 (Z-adjusted); the PAG atom on a Z-independent design
/// publishes the structural coefficient 2; the reverse-causal atom's mass stays
/// unidentified at the fixture's 0.2.
#[test]
fn class_graph_posterior_average_effect_matches_the_closed_forms_on_every_cell() {
    let pin = pin();
    let ate = &pin["static_average_effect"];
    let n = usize::try_from(ate["n"].as_u64().unwrap()).unwrap();
    let weights = floats(&ate["posterior_weights"]);
    let effects = floats(&ate["identified_atom_effects"]);
    let (direct, adjusted) = (effects[0], effects[1]);
    let unidentified = ate["expected_unidentified_mass"].as_f64().unwrap();
    let bayesian_tolerance = ate["effect_abs_tolerance"].as_f64().unwrap();
    let query = CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(
        VariableId::from_raw(0),
        VariableId::from_raw(1),
    ));
    let confounded = confounded_data(n);
    let balanced = balanced_data(n);
    let pag_weights = [weights[0] + weights[1], weights[2]];
    for (mode, inference, tolerance) in inferences(bayesian_tolerance, 1e-6) {
        for suite in SUITES {
            let label = format!("Cpdag {mode} {suite:?}");
            let result = run(
                &confounded,
                cpdag_atoms(&weights),
                query.clone(),
                inference.clone(),
                suite,
                11,
            );
            let atoms = identified_atoms(&result);
            assert_eq!(atoms.len(), 2, "{label}: two identified atoms");
            assert_matches_some_atom(&label, &atoms, &[direct], tolerance);
            assert_matches_some_atom(&label, &atoms, &[adjusted], tolerance);
            assert_unidentified_mass(&label, &result, unidentified);
            assert_validation_ran(&label, &result, suite);

            let label = format!("Pag {mode} {suite:?}");
            let result =
                run(&balanced, pag_atoms(&pag_weights), query.clone(), inference.clone(), suite, 11);
            let atoms = identified_atoms(&result);
            assert_eq!(atoms.len(), 1, "{label}: one identified atom");
            assert_matches_some_atom(&label, &atoms, &[adjusted], tolerance);
            assert_unidentified_mass(&label, &result, unidentified);
            assert_validation_ran(&label, &result, suite);
        }
    }
}

// ----------------------------------------------------------------------- ConditionalEffect

/// Modifier-heterogeneous design: `Y = 2T + 2Z + T*Z +/- 0.2`, so the effect of `T` at modifier
/// value `z` is `tau(z) = 2 + z` (the fixture's structural slope 2 plus an interaction of 1).
/// `confounded` uses the fixture's 6/2/2/6 `(z, t)` counts, otherwise `Z` is independent of
/// `T`; the `z = 1` cells are repeated `z1_scale` times, which moves `P(z = 1)`. The
/// modifier-averaged conditional effect is `tau(0) + P(z = 1) = 2 + P(z = 1)`, so two mixes
/// recover `tau(0)` and `tau(1)` and a modifier-free ATE (crude or adjusted) cannot match both.
fn heterogeneous_data(confounded: bool, z1_scale: usize) -> (TabularData, f64) {
    let counts: [(f64, f64, usize); 4] = if confounded {
        [(0.0, 0.0, 6), (0.0, 1.0, 2), (1.0, 0.0, 2), (1.0, 1.0, 6)]
    } else {
        [(0.0, 0.0, 4), (0.0, 1.0, 4), (1.0, 0.0, 4), (1.0, 1.0, 4)]
    };
    let (mut t, mut y, mut z) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..20 {
        for (zi, ti, count) in counts {
            let copies = if zi == 1.0 { z1_scale } else { 1 };
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
        TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())])
            .unwrap(),
        share,
    )
}

/// `ConditionalEffect x {Cpdag, Pag} x graph_posterior x {Frequentist, Bayesian} x {none,
/// cheap, full}` on a design whose effect depends on the modifier: every identified atom
/// publishes `tau(0) + P(z = 1)` at each of two modifier mixes (so `tau(0) = 2` and
/// `tau(1) = 3` are both recovered), and the two mixes differ by exactly the closed-form
/// 0.25, which a modifier-free effect could not reproduce.
#[test]
fn class_graph_posterior_conditional_effect_matches_the_structural_slope_on_every_cell() {
    let pin = pin();
    let ate = &pin["static_average_effect"];
    let weights = floats(&ate["posterior_weights"]);
    let tau0 = floats(&ate["identified_atom_effects"])[1];
    let unidentified = ate["expected_unidentified_mass"].as_f64().unwrap();
    let query = CausalQuery::ConditionalEffect(
        ConditionalEffectQuery::try_new(
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
                .with_effect_modifiers([VariableId::from_raw(2)]),
        )
        .unwrap(),
    );
    let pag_weights = [weights[0] + weights[1], weights[2]];
    for (mode, inference, tolerance) in inferences(0.01, 1e-6) {
        for suite in SUITES {
            let mut cpdag_means = Vec::new();
            let mut pag_means = Vec::new();
            for z1_scale in [1usize, 3] {
                let (confounded, share) = heterogeneous_data(true, z1_scale);
                let (balanced, balanced_share) = heterogeneous_data(false, z1_scale);
                let label = format!("Cpdag {mode} {suite:?} z1_scale={z1_scale}");
                let result = run(
                    &confounded,
                    cpdag_atoms(&weights),
                    query.clone(),
                    inference.clone(),
                    suite,
                    12,
                );
                let atoms = identified_atoms(&result);
                assert_eq!(atoms.len(), 2, "{label}");
                for atom in &atoms {
                    assert!(
                        (atom[0] - (tau0 + share)).abs() < tolerance,
                        "{label}: atom {atom:?} vs {}",
                        tau0 + share
                    );
                }
                cpdag_means.push(atoms[0][0]);
                assert_unidentified_mass(&label, &result, unidentified);
                assert_validation_ran(&label, &result, suite);

                let label = format!("Pag {mode} {suite:?} z1_scale={z1_scale}");
                let result = run(
                    &balanced,
                    pag_atoms(&pag_weights),
                    query.clone(),
                    inference.clone(),
                    suite,
                    12,
                );
                let atoms = identified_atoms(&result);
                assert_eq!(atoms.len(), 1, "{label}");
                assert!(
                    (atoms[0][0] - (tau0 + balanced_share)).abs() < tolerance,
                    "{label}: atom {:?} vs {}",
                    atoms[0],
                    tau0 + balanced_share
                );
                pag_means.push(atoms[0][0]);
                assert_unidentified_mass(&label, &result, unidentified);
                assert_validation_ran(&label, &result, suite);
            }
            // The effect moves with the modifier distribution: P(z = 1) is 0.5 then 0.75.
            for (name, means) in [("Cpdag", &cpdag_means), ("Pag", &pag_means)] {
                assert!(
                    (means[1] - means[0] - 0.25).abs() < 2.0 * tolerance,
                    "{name} {mode} {suite:?}: modifier dependence {means:?}"
                );
            }
        }
    }
}

// ----------------------------------------------------------------------- Responses

/// `InterventionResponse` and `ResponseCurve` x `{Cpdag, Pag}` x `graph_posterior`,
/// both inferences. Confounded atoms: direct curve `1 + 2.75 t`, adjusted
/// `1 + 2 t`; the Z-independent PAG design: `1 + 2 t`. Cheap/full are executed for
/// the intervention level.
#[test]
fn class_graph_posterior_responses_match_the_closed_form_curves() {
    let pin = pin();
    let resp = &pin["static_response"];
    let weights = floats(&resp["posterior_weights"]);
    let grid = floats(&resp["treatment_grid"]);
    let direct = floats(&resp["identified_atom_curve_direct"]);
    let adjusted = floats(&resp["identified_atom_curve_adjusted"]);
    let unidentified = resp["expected_unidentified_mass"].as_f64().unwrap();
    let level = resp["intervention_level"].as_f64().unwrap();
    let level_index = grid.iter().position(|&v| (v - level).abs() < 1e-12).unwrap();
    let confounded = response_data(resp, resp["z_slope"].as_f64().unwrap());
    let independent = response_data(resp, 0.0);
    let pag_weights = [weights[0] + weights[1], weights[2]];
    let seed = resp["seed"].as_u64().unwrap();
    let draws = usize::try_from(resp["bayesian_n_draws"].as_u64().unwrap()).unwrap();
    let prior_scale = resp["bayesian_prior_scale"].as_f64().unwrap();
    let modes = [
        ("frequentist", InferenceMode::Frequentist, resp["frequentist_abs_tolerance"].as_f64().unwrap()),
        ("bayesian", bayes(draws, prior_scale), resp["bayesian_abs_tolerance"].as_f64().unwrap()),
    ];
    for (mode, inference, tolerance) in modes {
        for (query_label, query, take) in [
            ("ResponseCurve", CausalQuery::Response(curve_query(&grid)), None),
            ("InterventionResponse", CausalQuery::Response(level_query(level)), Some(level_index)),
        ] {
            let pick = |curve: &[f64]| -> Vec<f64> {
                take.map_or_else(|| curve.to_vec(), |i| vec![curve[i]])
            };
            let suites: &[RefuteSuite] = if take.is_some() { &SUITES } else { &[RefuteSuite::None] };
            for &suite in suites {
                let label = format!("Cpdag {query_label} {mode} {suite:?}");
                let result = run(
                    &confounded,
                    cpdag_atoms(&weights),
                    query.clone(),
                    inference.clone(),
                    suite,
                    seed,
                );
                let atoms = identified_atoms(&result);
                assert_eq!(atoms.len(), 2, "{label}");
                assert_matches_some_atom(&label, &atoms, &pick(&direct), tolerance);
                assert_matches_some_atom(&label, &atoms, &pick(&adjusted), tolerance);
                assert_unidentified_mass(&label, &result, unidentified);
                assert_validation_ran(&label, &result, suite);

                let label = format!("Pag {query_label} {mode} {suite:?}");
                let result = run(
                    &independent,
                    pag_atoms(&pag_weights),
                    query.clone(),
                    inference.clone(),
                    suite,
                    seed,
                );
                let atoms = identified_atoms(&result);
                assert_eq!(atoms.len(), 1, "{label}");
                assert_matches_some_atom(&label, &atoms, &pick(&adjusted), tolerance);
                assert_unidentified_mass(&label, &result, unidentified);
                assert_validation_ran(&label, &result, suite);
            }
        }
    }
}

/// `InterventionResponse x Dag x graph_posterior x Frequentist x {cheap, full}`:
/// the atoms' published levels equal the closed forms `1 + 2.75 t` and `1 + 2 t`
/// at the fixture's intervention level while the per-atom refuters run.
#[test]
fn dag_graph_posterior_intervention_response_matches_the_closed_form_levels_under_refuters() {
    let pin = pin();
    let resp = &pin["static_response"];
    let weights = floats(&resp["posterior_weights"]);
    let grid = floats(&resp["treatment_grid"]);
    let direct = floats(&resp["identified_atom_curve_direct"]);
    let adjusted = floats(&resp["identified_atom_curve_adjusted"]);
    let level = resp["intervention_level"].as_f64().unwrap();
    let index = grid.iter().position(|&v| (v - level).abs() < 1e-12).unwrap();
    let data = response_data(resp, resp["z_slope"].as_f64().unwrap());
    let tolerance = resp["frequentist_abs_tolerance"].as_f64().unwrap();
    for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
        let label = format!("Dag frequentist {suite:?}");
        let result = run(
            &data,
            dag_atoms(&weights),
            CausalQuery::Response(level_query(level)),
            InferenceMode::Frequentist,
            suite,
            resp["seed"].as_u64().unwrap(),
        );
        let atoms = identified_atoms(&result);
        assert_eq!(atoms.len(), 2, "{label}");
        assert_matches_some_atom(&label, &atoms, &[direct[index]], tolerance);
        assert_matches_some_atom(&label, &atoms, &[adjusted[index]], tolerance);
        assert_unidentified_mass(&label, &result, resp["expected_unidentified_mass"].as_f64().unwrap());
        assert!(!result.refutations.is_empty(), "{label}: refuters must execute");
    }
}
