//! Derivatives, ADMG conditional effects, counterfactuals and dynamic conditional, derivative and
//! cross-world points over a graph posterior.
//!
//! Every expected value is read from the explicit cell each atom is answered by (a study over
//! the atom's own supplied structure), counted from one contingency table, enumerated from an
//! exact latent-variable model or pinned by the conformance fixtures, never from the mixture:
//! the posterior moves only which structure answers, how much posterior mass identifies, and how
//! the atoms' values are weighted.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::too_many_lines,
    clippy::needless_range_loop,
    reason = "test scaffolding compares constants computed by counting small contingency tables"
)]

#[path = "../../antecedent-identify/tests/support/latent_scm.rs"]
mod latent_scm;

use std::sync::Arc;

use antecedent::{BayesianConfig, CellStatus, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ConditionalEffectQuery, CounterfactualQuery, DerivativeScale,
    ExecutionContext, IdentificationStatus, Intervention, InterventionalDistributionQuery,
    NestedCounterfactualQuery, PathSpecificEffectQuery, ResponseFunctional, ResponseIdentification,
    ResponseQuery, ResponseUncertainty, ResponseValue, TemporalEffectQuery, TemporalPolicy,
    TemporalResponseSpec, Value, VariableId,
};
use antecedent_data::{TabularData, TimeSeriesData};
use antecedent_discovery::{
    GraphPosterior, GraphPosteriorAtomKind, SamplingSummary, adjacency_mask_from_admg, set_edge,
};
use antecedent_estimate::ContinuousResponseOptions;
use antecedent_graph::{Admg, Dag, DenseNodeId};
use antecedent_prob::InferenceDiagnostics;
use latent_scm::{LatentScm, Rng};

/// Graph-posterior estimation recurses deeply; run each body on a large stack.
fn on_large_stack(body: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name("graph-posterior-functional".into())
        .stack_size(64 * 1024 * 1024)
        .spawn(body)
        .expect("spawn evidence thread")
        .join()
        .expect("evidence body panicked");
}

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn node(raw: u32) -> DenseNodeId {
    DenseNodeId::from_raw(raw)
}

fn sampling(label: &str) -> SamplingSummary {
    SamplingSummary {
        ess: 1.0,
        diagnostics: InferenceDiagnostics::analytic(label),
        rejected_invalid: 0,
    }
}

fn has(result: &antecedent::StudyResult, code: &str) -> bool {
    result.diagnostics.iter().any(|d| d.code.as_ref() == code)
}

fn ate(result: &antecedent::StudyResult) -> f64 {
    result.estimate.as_effect().unwrap().ate
}

fn value_of(result: &antecedent::StudyResult, key: u64) -> Option<ResponseValue> {
    let mixture = result.structural_response.as_ref().expect("a structural mixture");
    mixture.atoms.iter().find(|atom| atom.graph_key == key).expect("atom present").value.clone()
}

// ---------------------------------------------------------------------------
// ConditionalEffect over a posterior of Admg atoms.
// ---------------------------------------------------------------------------

/// `(z, x, y)` levels with their counts; every stratum holds both treatment arms.
fn conditional_cells() -> Vec<([f64; 3], usize)> {
    let counts = [30, 10, 15, 25, 12, 28, 8, 52];
    counts
        .iter()
        .enumerate()
        .map(|(index, count)| {
            ([((index >> 2) & 1) as f64, ((index >> 1) & 1) as f64, (index & 1) as f64], *count)
        })
        .collect()
}

fn conditional_data() -> TabularData {
    let mut columns: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for (levels, count) in conditional_cells() {
        for (column, level) in columns.iter_mut().zip(levels) {
            column.extend(std::iter::repeat_n(level, count));
        }
    }
    TabularData::from_f64_columns([
        ("z", columns[0].as_slice()),
        ("x", columns[1].as_slice()),
        ("y", columns[2].as_slice()),
    ])
    .unwrap()
}

/// `sum_z P(z) (E[Y | x = 1, z] - E[Y | x = 0, z])`, by counting.
fn stratified_difference() -> f64 {
    let cells = conditional_cells();
    let total: usize = cells.iter().map(|c| c.1).sum();
    let mut value = 0.0;
    for z in [0.0, 1.0] {
        let stratum: usize = cells.iter().filter(|c| c.0[0] == z).map(|c| c.1).sum();
        let mean = |x: f64| {
            let arm: usize = cells.iter().filter(|c| c.0[0] == z && c.0[1] == x).map(|c| c.1).sum();
            let ones: usize = cells
                .iter()
                .filter(|c| c.0[0] == z && c.0[1] == x && c.0[2] == 1.0)
                .map(|c| c.1)
                .sum();
            ones as f64 / arm as f64
        };
        value += stratum as f64 / total as f64 * (mean(1.0) - mean(0.0));
    }
    value
}

/// `E[Y | x = 1] - E[Y | x = 0]`, by counting: the contrast of a graph that leaves the modifier
/// out of the mechanism.
fn pooled_difference() -> f64 {
    let cells = conditional_cells();
    let mean = |x: f64| {
        let arm: usize = cells.iter().filter(|c| c.0[1] == x).map(|c| c.1).sum();
        let ones: usize = cells.iter().filter(|c| c.0[1] == x && c.0[2] == 1.0).map(|c| c.1).sum();
        ones as f64 / arm as f64
    };
    mean(1.0) - mean(0.0)
}

fn admg_of(n: u32, directed: &[(u32, u32)], bidirected: &[(u32, u32)]) -> Admg {
    let mut graph = Admg::with_variables(n);
    for &(from, to) in directed {
        graph.insert_directed(node(from), node(to)).unwrap();
    }
    for &(left, right) in bidirected {
        graph.insert_bidirected(node(left), node(right)).unwrap();
    }
    graph
}

/// `z -> x`, `z -> y`, `x -> y`, `z <-> x`: conditioning on `z` closes the back door.
fn confounded_modifier() -> Admg {
    admg_of(3, &[(0, 1), (0, 2), (1, 2)], &[(0, 1)])
}

/// `x -> y` with the modifier outside the mechanism.
fn modifier_outside() -> Admg {
    admg_of(3, &[(1, 2)], &[])
}

/// `x -> y` with `x <-> y`: the bow arc, which no conditional contrast identifies.
fn bow_arc() -> Admg {
    admg_of(3, &[(1, 2)], &[(1, 2)])
}

fn admg_posterior(graphs: &[Admg], weights: &[f64]) -> GraphPosterior {
    let masks: Vec<u64> =
        graphs.iter().map(|graph| adjacency_mask_from_admg(graph).unwrap()).collect();
    let n = 3;
    GraphPosterior::new(
        n,
        weights.to_vec(),
        masks,
        vec![0.0; n * n],
        vec![0.0; n * n],
        sampling("graph_posterior_functional_admg"),
    )
    .unwrap()
    .with_atom_kind(GraphPosteriorAtomKind::Admg)
}

fn conditional_query() -> CausalQuery {
    CausalQuery::ConditionalEffect(
        ConditionalEffectQuery::try_new(
            AverageEffectQuery::binary_ate(v(1), v(2)).with_effect_modifiers([v(0)]),
        )
        .unwrap(),
    )
}

fn conditional_study(
    posterior: GraphPosterior,
    inference: InferenceMode,
    replicates: u32,
) -> Study {
    Study::tabular(conditional_data())
        .graph_posterior(posterior)
        .query(conditional_query())
        .inference(inference)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(replicates)
        .build()
        .unwrap()
}

fn explicit_conditional(graph: Admg, inference: InferenceMode, replicates: u32) -> Study {
    Study::tabular(conditional_data())
        .graph(graph)
        .query(conditional_query())
        .inference(inference)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(replicates)
        .build()
        .unwrap()
}

#[test]
fn the_admg_conditional_mixture_weights_the_atoms_and_states_the_lost_mass() {
    on_large_stack(|| {
        let weights = [0.5, 0.3, 0.2];
        let graphs = [confounded_modifier(), modifier_outside(), bow_arc()];
        let posterior = admg_posterior(&graphs, &weights);
        let keys = posterior.graph_keys.clone();
        // The two identified atoms disagree, so the weights are visible in the value.
        let (stratified, pooled) = (stratified_difference(), pooled_difference());
        assert!((stratified - 0.259_259_259_259_259_3).abs() < 1e-12, "{stratified}");
        assert!((pooled - 0.295).abs() < 1e-12, "{pooled}");
        let expected = (0.5 * stratified + 0.3 * pooled) / 0.8;

        let study = conditional_study(posterior, InferenceMode::Frequentist, 80);
        assert_eq!(study.support_status(), Some(CellStatus::Refused));
        let ctx = ExecutionContext::for_tests(8_101);
        let result = study.run(&ctx).unwrap();
        let effect = result.estimate.as_effect().unwrap();
        assert!((effect.ate - expected).abs() < 1e-9, "{} vs {expected}", effect.ate);
        assert_eq!(result.support_status, Some(CellStatus::Refused));
        assert_eq!(result.logical_plan.estimator.as_deref(), Some("functional.effect"));
        assert_eq!(result.identification.status, IdentificationStatus::GraphDependent);

        let mixture = result.structural_response.as_ref().expect("a structural mixture");
        assert!((mixture.identified_mass - 0.8).abs() < 1e-9, "{}", mixture.identified_mass);
        let lost = mixture.unidentified_mass + mixture.unevaluable_mass;
        assert!((lost - 0.2).abs() < 1e-9, "{lost}");
        assert_eq!(mixture.atoms.len(), 3);
        for (key, truth) in [(keys[0], stratified), (keys[1], pooled)] {
            match value_of(&result, key) {
                Some(ResponseValue::Scalar(value)) => {
                    assert!((value - truth).abs() < 1e-12, "{value} vs {truth}")
                }
                other => panic!("an identified atom carries its own contrast, got {other:?}"),
            }
        }
        assert!(value_of(&result, keys[2]).is_none(), "the bow arc identifies nothing");

        // Several atoms and no row influence: the interval is the spread of the draw-wise
        // mixture over shared resamples.
        let se = effect.se_bootstrap().expect("a shared-replicate standard error");
        assert!(se.is_finite() && se > 0.01 && se < 0.2, "{se}");
        assert!(has(&result, "estimate.graph_posterior.atom_route"));
        assert!(has(&result, "estimate.graph_posterior.structural_aggregation"));
    });
}

#[test]
fn one_identified_admg_atom_is_the_explicit_conditional_cell() {
    on_large_stack(|| {
        let ctx = ExecutionContext::for_tests(8_102);
        let explicit = explicit_conditional(confounded_modifier(), InferenceMode::Frequentist, 40)
            .run(&ctx)
            .unwrap();
        let mixed = conditional_study(
            admg_posterior(&[confounded_modifier()], &[1.0]),
            InferenceMode::Frequentist,
            40,
        )
        .run(&ctx)
        .unwrap();
        assert!((ate(&mixed) - ate(&explicit)).abs() < 1e-12);
        assert!((ate(&mixed) - stratified_difference()).abs() < 1e-12);
        assert_eq!(mixed.identification.status, explicit.identification.status);
        let mixture = mixed.structural_response.as_ref().unwrap();
        assert!((mixture.identified_mass - 1.0).abs() < 1e-12);
        assert!(mixture.unidentified_mass.abs() < 1e-12 && mixture.unevaluable_mass.abs() < 1e-12);
        // The atom's own sampling uncertainty is the mixture's.
        assert_eq!(
            mixed.estimate.as_effect().unwrap().se_bootstrap().is_some(),
            explicit.estimate.as_effect().unwrap().se_bootstrap().is_some()
        );
    });
}

#[test]
fn a_posterior_no_atom_of_which_identifies_the_contrast_refuses_with_the_identifier() {
    on_large_stack(|| {
        let study = conditional_study(
            admg_posterior(&[bow_arc(), bow_arc()], &[0.5, 0.5]),
            InferenceMode::Frequentist,
            0,
        );
        let error = study.run(&ExecutionContext::for_tests(8_103)).unwrap_err();
        assert!(error.is_identification_refusal(), "{error}");
    });
}

#[test]
fn the_bayesian_conditional_mixture_pools_the_atoms_draw_by_draw() {
    on_large_stack(|| {
        let weights = [0.6, 0.4];
        let posterior = admg_posterior(&[confounded_modifier(), modifier_outside()], &weights);
        let study = conditional_study(
            posterior,
            InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(256).prior_scale(1_000.0)),
            0,
        );
        let ctx = ExecutionContext::for_tests(8_104);
        let result = study.run(&ctx).unwrap();
        let pooled = result.posterior.as_ref().expect("the atoms ran in one context");
        assert_eq!(pooled.draws.n_draws, 256);
        // The atoms are Dirichlet posteriors of the plug-in contrasts, so the pooled posterior
        // centres on the weighted plug-in contrast.
        let mixed_plug_in = 0.6 * stratified_difference() + 0.4 * pooled_difference();
        assert!((ate(&result) - mixed_plug_in).abs() < 0.03, "{} vs {mixed_plug_in}", ate(&result));
        assert!(result.estimate.as_effect().unwrap().uncertainty.analytic.se > 0.0);
        assert!(has(&result, "estimate.graph_posterior.atom_route"));
    });
}

// ---------------------------------------------------------------------------
// Derivatives over a posterior of Dag atoms.
// ---------------------------------------------------------------------------

fn derivative_data() -> TabularData {
    let a: Vec<_> = (0..400)
        .map(|i| 2.0 + (f64::from(i) * 0.71).sin() + 0.2 * (f64::from(i) * 0.13).cos())
        .collect();
    let b: Vec<_> = (0..400).map(|i| (f64::from(i) * 1.13).cos()).collect();
    let y: Vec<_> = a.iter().zip(&b).map(|(a, b)| 5.0 + 2.0 * a - 0.5 * b).collect();
    let w: Vec<_> = a.iter().zip(&b).map(|(a, b)| 1.0 + 0.25 * a + 1.5 * b).collect();
    TabularData::from_f64_columns([
        ("a", a.as_slice()),
        ("b", b.as_slice()),
        ("y", y.as_slice()),
        ("w", w.as_slice()),
    ])
    .unwrap()
}

/// `a, b -> y, w`: `b` is not a confounder of `a -> y`.
fn derivative_dag_plain() -> Dag {
    let mut graph = Dag::with_variables(4);
    for (from, to) in [(0, 2), (0, 3), (1, 2), (1, 3)] {
        graph.insert_directed(node(from), node(to)).unwrap();
    }
    graph
}

/// `b -> a` as well: `b` confounds `a -> y`, so the back door adjusts for it.
fn derivative_dag_confounded() -> Dag {
    let mut graph = derivative_dag_plain();
    graph.insert_directed(node(1), node(0)).unwrap();
    graph
}

fn dag_mask(edges: &[(usize, usize)]) -> u64 {
    edges.iter().fold(0_u64, |mask, &(from, to)| set_edge(mask, 4, from, to, true))
}

const PLAIN_EDGES: [(usize, usize); 4] = [(0, 2), (0, 3), (1, 2), (1, 3)];
const CONFOUNDED_EDGES: [(usize, usize); 5] = [(0, 2), (0, 3), (1, 2), (1, 3), (1, 0)];
/// `a -> y` and `y -> a`: not a DAG, so the sample is not a member of the posterior's class.
const CYCLIC_EDGES: [(usize, usize); 2] = [(0, 2), (2, 0)];

fn dag_posterior(atoms: &[&[(usize, usize)]], weights: &[f64]) -> GraphPosterior {
    let masks: Vec<u64> = atoms.iter().map(|edges| dag_mask(edges)).collect();
    GraphPosterior::new(
        4,
        weights.to_vec(),
        masks,
        vec![0.0; 16],
        vec![0.0; 16],
        sampling("graph_posterior_functional_dag"),
    )
    .unwrap()
}

fn point_derivative() -> ResponseFunctional {
    ResponseFunctional::PointDerivative {
        outcome: v(2),
        treatment: v(0),
        at: 2.0,
        order: 1,
        scale: DerivativeScale::Identity,
    }
}

fn jacobian() -> ResponseFunctional {
    ResponseFunctional::Jacobian {
        outcomes: Arc::from([v(2), v(3)]),
        treatments: Arc::from([v(0), v(1)]),
        at: Arc::from([2.0, 0.0]),
        scale: DerivativeScale::Identity,
    }
}

fn options() -> ContinuousResponseOptions {
    ContinuousResponseOptions { bandwidth: Some(0.35), ..Default::default() }
}

fn derivative_posterior_study(
    posterior: GraphPosterior,
    functional: ResponseFunctional,
    inference: InferenceMode,
) -> Study {
    Study::tabular(derivative_data())
        .graph_posterior(posterior)
        .query(CausalQuery::Response(ResponseQuery::new(functional)))
        .response_options(options())
        .inference(inference)
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
}

fn explicit_derivative(
    graph: Dag,
    functional: ResponseFunctional,
    inference: InferenceMode,
) -> antecedent::StudyResult {
    Study::tabular(derivative_data())
        .graph(graph)
        .query(CausalQuery::Response(ResponseQuery::new(functional)))
        .response_options(options())
        .inference(inference)
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(8_201))
        .unwrap()
}

fn coordinates(result: &antecedent::StudyResult) -> Vec<f64> {
    let response = result.response().expect("a response payload");
    let ResponseIdentification::PointIdentified(value) = &response.estimate else {
        panic!("not point identified: {:?}", response.estimate)
    };
    match value {
        ResponseValue::Scalar(x) => vec![*x],
        ResponseValue::Vector(values) | ResponseValue::Jacobian { values, .. } => values.to_vec(),
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn a_derivative_mixture_is_the_weighted_sum_of_the_explicit_atoms() {
    on_large_stack(|| {
        let ctx = ExecutionContext::for_tests(8_201);
        let plain = explicit_derivative(
            derivative_dag_plain(),
            point_derivative(),
            InferenceMode::Frequentist,
        );
        let confounded = explicit_derivative(
            derivative_dag_confounded(),
            point_derivative(),
            InferenceMode::Frequentist,
        );
        let (e1, e2) = (coordinates(&plain)[0], coordinates(&confounded)[0]);
        // The linear fixture has slope 2 in `a`; the adjustment sets differ only in whether the
        // back door through `b` is closed.
        assert!((e2 - 2.0).abs() < 0.3, "{e2}");
        let expected = (0.5 * e1 + 0.3 * e2) / 0.8;

        // A cyclic sample is not a member of the class: unidentified posterior mass.
        let posterior =
            dag_posterior(&[&PLAIN_EDGES, &CONFOUNDED_EDGES, &CYCLIC_EDGES], &[0.5, 0.3, 0.2]);
        let study = derivative_posterior_study(
            posterior.clone(),
            point_derivative(),
            InferenceMode::Frequentist,
        );
        assert_eq!(study.support_status(), Some(CellStatus::Refused));
        let result = study.run(&ctx).unwrap();
        let mixture = result.structural_response.as_ref().expect("a structural mixture");
        assert!((mixture.identified_mass - 0.8).abs() < 1e-9, "{}", mixture.identified_mass);
        assert!((mixture.unidentified_mass - 0.2).abs() < 1e-9);
        match &mixture.conditional_on_identified {
            Some(ResponseValue::Scalar(value)) => {
                assert!((value - expected).abs() < 1e-9, "{value} vs {expected}")
            }
            other => panic!("expected the conditional mixture, got {other:?}"),
        }
        assert_eq!(result.identification.status, IdentificationStatus::GraphDependent);
        let response = result.response().unwrap();
        assert!(
            matches!(response.estimate, ResponseIdentification::GraphDependent(_)),
            "lost mass makes the response graph dependent, not a point"
        );
        for (key, truth) in [(posterior.graph_keys[0], e1), (posterior.graph_keys[1], e2)] {
            match value_of(&result, key) {
                Some(ResponseValue::Scalar(value)) => assert!((value - truth).abs() < 1e-9),
                other => panic!("an identified atom carries its own derivative, got {other:?}"),
            }
        }
        assert!(value_of(&result, posterior.graph_keys[2]).is_none());
        assert!(has(&result, "estimate.graph_posterior.atom_route"));

        // With every sample identified the response is a point, and its interval is the joint
        // influence of the two atoms on the shared rows.
        let both = derivative_posterior_study(
            dag_posterior(&[&PLAIN_EDGES, &CONFOUNDED_EDGES], &[0.6, 0.4]),
            point_derivative(),
            InferenceMode::Frequentist,
        )
        .run(&ctx)
        .unwrap();
        let expected = 0.6 * e1 + 0.4 * e2;
        assert!((coordinates(&both)[0] - expected).abs() < 1e-9);
        assert!((ate(&both) - expected).abs() < 1e-9);
        assert_ne!(both.identification.status, IdentificationStatus::GraphDependent);
        let ResponseUncertainty::Scalar { standard_error, lower, upper, .. } =
            both.response().unwrap().uncertainty
        else {
            panic!("several atoms with row influence publish a scalar interval")
        };
        assert!(standard_error.is_finite() && standard_error >= 0.0);
        assert!(lower <= expected + 1e-9 && expected <= upper + 1e-9);
    });
}

#[test]
fn a_jacobian_mixture_is_mixed_coordinate_by_coordinate() {
    on_large_stack(|| {
        let plain =
            explicit_derivative(derivative_dag_plain(), jacobian(), InferenceMode::Frequentist);
        let confounded = explicit_derivative(
            derivative_dag_confounded(),
            jacobian(),
            InferenceMode::Frequentist,
        );
        let (first, second) = (coordinates(&plain), coordinates(&confounded));
        assert_eq!(first.len(), 4);
        let result = derivative_posterior_study(
            dag_posterior(&[&PLAIN_EDGES, &CONFOUNDED_EDGES], &[0.7, 0.3]),
            jacobian(),
            InferenceMode::Frequentist,
        )
        .run(&ExecutionContext::for_tests(8_202))
        .unwrap();
        let got = coordinates(&result);
        assert_eq!(got.len(), 4);
        for j in 0..4 {
            let want = 0.7 * first[j] + 0.3 * second[j];
            assert!((got[j] - want).abs() < 1e-9, "[{j}] {} vs {want}", got[j]);
        }
        let ResponseValue::Jacobian { outcomes, treatments, .. } = result
            .response()
            .and_then(|response| match &response.estimate {
                ResponseIdentification::PointIdentified(value) => Some(value.clone()),
                _ => None,
            })
            .unwrap()
        else {
            panic!("a Jacobian mixes to a Jacobian")
        };
        assert_eq!((outcomes, treatments), (2, 2));
    });
}

#[test]
fn the_bayesian_derivative_mixture_pools_the_atoms_credible_draws() {
    on_large_stack(|| {
        let inference = || {
            InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64).prior_scale(1_000.0))
        };
        let result = derivative_posterior_study(
            dag_posterior(&[&PLAIN_EDGES, &CONFOUNDED_EDGES], &[0.6, 0.4]),
            point_derivative(),
            inference(),
        )
        .run(&ExecutionContext::for_tests(8_203))
        .unwrap();
        let mixture = result.structural_response.as_ref().unwrap();
        assert!((mixture.identified_mass - 1.0).abs() < 1e-9);
        let ResponseUncertainty::Scalar { draws: Some(draws), lower, upper, .. } =
            result.response().unwrap().uncertainty.clone()
        else {
            panic!("the pooled credible draws are published")
        };
        assert_eq!(draws.n_draws, 64);
        let value = coordinates(&result)[0];
        let mean = draws.values.iter().sum::<f64>() / draws.values.len() as f64;
        let sd = (draws.values.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / (draws.values.len() as f64 - 1.0))
            .sqrt();
        assert!((mean - value).abs() <= 0.1 * sd + 1e-9, "{mean} vs {value}");
        assert!(lower <= value + 1e-9 && value <= upper + 1e-9);
        // Each atom keeps its own full response.
        for atom in &mixture.atoms {
            assert!(atom.response.is_some());
        }
    });
}

#[test]
fn an_intervention_level_over_several_coordinates_is_answered_atom_by_atom() {
    on_large_stack(|| {
        let level = ResponseFunctional::InterventionResponse {
            outcome: v(2),
            interventions: Arc::from([
                Intervention::set(v(0), Value::f64(2.0)),
                Intervention::set(v(1), Value::f64(0.0)),
            ]),
        };
        let explicit = Study::tabular(derivative_data())
            .graph(derivative_dag_plain())
            .query(CausalQuery::Response(ResponseQuery::new(level.clone())))
            .response_options(options())
            .refute(RefuteSuite::None)
            .build();
        // The explicit route decides whether a Dag executes a two-coordinate level; the
        // posterior executes exactly when it does, and refuses with its refusal otherwise.
        let posterior = derivative_posterior_study(
            dag_posterior(&[&PLAIN_EDGES], &[1.0]),
            level,
            InferenceMode::Frequentist,
        );
        let ctx = ExecutionContext::for_tests(8_204);
        match explicit.and_then(|study| study.run(&ctx)) {
            Ok(reference) => {
                let mixed = posterior.run(&ctx).unwrap();
                assert!((coordinates(&mixed)[0] - coordinates(&reference)[0]).abs() < 1e-9);
            }
            Err(_) => assert!(posterior.run(&ctx).is_err()),
        }
    });
}

// ---------------------------------------------------------------------------
// Counterfactual over a posterior of Admg atoms.
// ---------------------------------------------------------------------------

const ROWS: f64 = 60_000.0;

/// The observational joint of `scm` as a table of about `ROWS` rows, one count per cell.
fn table_of(scm: &LatentScm, names: &[&str]) -> TabularData {
    let joint = scm.observational();
    let cards = &scm.cards;
    let mut columns: Vec<Vec<f64>> = vec![Vec::new(); cards.len()];
    for (cell, probability) in joint.iter().enumerate() {
        let count = (probability * ROWS).round() as usize;
        let mut rest = cell;
        let mut levels = vec![0_usize; cards.len()];
        for variable in (0..cards.len()).rev() {
            levels[variable] = rest % cards[variable];
            rest /= cards[variable];
        }
        for _ in 0..count {
            for variable in 0..cards.len() {
                columns[variable].push(levels[variable] as f64);
            }
        }
    }
    TabularData::from_f64_columns(
        names.iter().zip(&columns).map(|(name, column)| (*name, column.as_slice())),
    )
    .unwrap()
}

/// `E[Y_x | X = 0] - E[Y | X = 0]` enumerated over the model's exogenous states.
fn enumerated_effect(scm: &LatentScm, x: usize, y: usize) -> f64 {
    let mean = |numerators: Vec<f64>| {
        let total: f64 = numerators.iter().sum();
        numerators.iter().enumerate().map(|(level, n)| level as f64 * n).sum::<f64>() / total
    };
    mean(scm.ett_numerators(x, 1, 0, y)) - mean(scm.ett_numerators(x, 0, 0, y))
}

fn counterfactual_query() -> CausalQuery {
    CausalQuery::Counterfactual(CounterfactualQuery::new(
        v(2),
        [Intervention::set(v(0), Value::f64(1.0))],
    ))
}

#[test]
fn a_counterfactual_mixture_weights_the_enumerated_atoms_and_states_the_lost_mass() {
    on_large_stack(|| {
        // x -> m -> y with x <-> y: the front door identifies the effect on the treated.
        let scm = LatentScm::random(&mut Rng::new(8), &[2, 3, 2], &[(0, 1), (1, 2)], &[(0, 2)], 3);
        let data = table_of(&scm, &["x", "m", "y"]);
        let front_door = admg_of(3, &[(0, 1), (1, 2)], &[(0, 2)]);
        // x -> m -> y with no latent: the effect on the treated is the observed contrast.
        let unconfounded = admg_of(3, &[(0, 1), (1, 2)], &[]);
        // The bow arc leaves the effect on the treated unidentified.
        let hedge = admg_of(3, &[(0, 2), (0, 1)], &[(0, 2)]);
        let truth_front_door = enumerated_effect(&scm, 0, 2);
        // The latent-free atom is answered by the explicit cell over the same table.
        let counted = Study::tabular(data.clone())
            .graph(unconfounded.clone())
            .query(counterfactual_query())
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .run(&ExecutionContext::for_tests(8_300))
            .unwrap()
            .effect();
        let expected = (0.5 * truth_front_door + 0.3 * counted) / 0.8;

        let masks: Vec<u64> = [&front_door, &unconfounded, &hedge]
            .into_iter()
            .map(|graph| adjacency_mask_from_admg(graph).unwrap())
            .collect();
        let posterior = GraphPosterior::new(
            3,
            vec![0.5, 0.3, 0.2],
            masks,
            vec![0.0; 9],
            vec![0.0; 9],
            sampling("graph_posterior_functional_counterfactual"),
        )
        .unwrap()
        .with_atom_kind(GraphPosteriorAtomKind::Admg);
        let keys = posterior.graph_keys.clone();
        let study = Study::tabular(data)
            .graph_posterior(posterior)
            .query(counterfactual_query())
            .refute(RefuteSuite::None)
            .build()
            .unwrap();
        assert_eq!(study.support_status(), Some(CellStatus::Refused));
        let result = study.run(&ExecutionContext::for_tests(8_301)).unwrap();
        assert!((result.effect() - expected).abs() < 1e-3, "{} vs {expected}", result.effect());
        match value_of(&result, keys[0]) {
            Some(ResponseValue::Scalar(value)) => {
                assert!((value - truth_front_door).abs() < 1e-3, "{value} vs {truth_front_door}");
            }
            other => panic!("the front-door atom carries its enumerated effect, got {other:?}"),
        }
        match value_of(&result, keys[1]) {
            Some(ResponseValue::Scalar(value)) => assert!((value - counted).abs() < 1e-3),
            other => panic!("the unconfounded atom carries the observed contrast, got {other:?}"),
        }
        assert!(value_of(&result, keys[2]).is_none());
        let mixture = result.structural_response.as_ref().unwrap();
        assert!((mixture.identified_mass - 0.8).abs() < 1e-9, "{}", mixture.identified_mass);
        assert!(((mixture.unidentified_mass + mixture.unevaluable_mass) - 0.2).abs() < 1e-9);
        assert_eq!(result.identification.status, IdentificationStatus::GraphDependent);
        assert!(has(&result, "estimate.graph_posterior.atom_route"));
    });
}

// ---------------------------------------------------------------------------
// Conditional, derivative and cross-world points over a posterior of TemporalDag atoms.
// ---------------------------------------------------------------------------

fn series_from(columns: &[(&'static str, Vec<f64>)]) -> TimeSeriesData {
    let refs: Vec<(&str, &[f64])> =
        columns.iter().map(|(name, values)| (*name, values.as_slice())).collect();
    TimeSeriesData::from_f64_columns(refs, 1).unwrap()
}

fn conditional_series() -> TimeSeriesData {
    let n = 80;
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];
    let mut y = vec![0.0; n];
    for t in 0..n {
        x[t] = ((t as f64) * 0.37).sin();
        w[t] = if t % 2 == 0 { 0.0 } else { 1.0 };
        if t > 0 {
            y[t] = 1.5 * x[t - 1] + 0.2 * w[t] + 0.8 * x[t - 1] * w[t];
        }
    }
    series_from(&[("x", x), ("y", y), ("w", w)])
}

fn derivative_series() -> TimeSeriesData {
    let n = 80;
    let mut x = vec![0.0; n];
    let mut b = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut w = vec![0.0; n];
    for t in 0..n {
        x[t] = 1.0 + ((t as f64) * 0.37).sin();
        b[t] = if t % 2 == 0 { 0.0 } else { 1.0 };
        if t > 0 {
            y[t] = 3.0 + 2.0 * x[t - 1] + 0.5 * b[t - 1];
            w[t] = 1.0 + x[t - 1] - b[t - 1];
        }
    }
    series_from(&[("x", x), ("b", b), ("y", y), ("v", w)])
}

fn cross_series() -> TimeSeriesData {
    let n = 200;
    let mut x = vec![0.0; n];
    let mut m = vec![0.0; n];
    let mut y = vec![0.0; n];
    for t in 0..n {
        x[t] = ((t as f64) * 0.37).sin();
        if t > 0 {
            let shock = if t % 2 == 0 { 0.3 } else { -0.3 };
            m[t] = 0.5 * x[t - 1] + shock;
            y[t] = 1.5 * x[t - 1] + 4.0 * m[t];
        }
    }
    series_from(&[("x", x), ("m", m), ("y", y)])
}

fn lag_bit(n: usize, from: usize, to: usize) -> u64 {
    1_u64 << (from * n + to)
}

/// One sample of a dynamic posterior: its contemporaneous edges, lagged edges and weight.
struct Template {
    contemporaneous: Vec<(usize, usize)>,
    lagged: Vec<(usize, usize)>,
    weight: f64,
}

fn dbn_posterior(n: usize, templates: &[Template]) -> GraphPosterior {
    let adjacency: Vec<u64> = templates
        .iter()
        .map(|t| {
            t.contemporaneous.iter().fold(0_u64, |mask, &(f, to)| set_edge(mask, n, f, to, true))
        })
        .collect();
    let lag_masks: Vec<u64> = templates
        .iter()
        .map(|t| t.lagged.iter().fold(0_u64, |mask, &(f, to)| mask | lag_bit(n, f, to)))
        .collect();
    let weights: Vec<f64> = templates.iter().map(|t| t.weight).collect();
    let cell = n * n;
    GraphPosterior::new(
        n,
        weights,
        adjacency,
        vec![0.0; cell],
        vec![0.0; cell],
        sampling("graph_posterior_functional_dbn"),
    )
    .unwrap()
    .with_lagged_marginals(1, vec![0.0; cell])
    .unwrap()
    .with_lag_masks(lag_masks)
    .unwrap()
}

fn conditional_temporal_query() -> TemporalEffectQuery {
    TemporalEffectQuery::pulse(v(0), v(1), 1.0)
        .with_horizon_steps(1)
        .with_max_history_lag(Some(1))
        .with_effect_modifiers([v(2)])
        .with_policy(TemporalPolicy::pulse(-1))
}

#[test]
fn the_dynamic_conditional_mixture_states_the_lost_mass_and_publishes_a_shared_interval() {
    on_large_stack(|| {
        let pin: serde_json::Value = serde_json::from_str(include_str!(
            "../../../conformance/estimate/temporal_conditional/expected.json"
        ))
        .unwrap();
        let truth = pin["marginal_effect"].as_f64().unwrap();
        let correct =
            |weight| Template { contemporaneous: vec![(2, 1)], lagged: vec![(0, 1)], weight };
        // `w -> y` and `y -> w`: not a DAG, so the sample identifies nothing.
        let cyclic =
            Template { contemporaneous: vec![(2, 1), (1, 2)], lagged: vec![(0, 1)], weight: 0.3 };
        let posterior = dbn_posterior(3, &[correct(0.4), correct(0.3), cyclic]);
        let study = Study::series(conditional_series())
            .graph_posterior(posterior)
            .query(CausalQuery::TemporalEffect(conditional_temporal_query()))
            .bootstrap_replicates(60)
            .refute(RefuteSuite::None)
            .build()
            .unwrap();
        assert_eq!(study.support_status(), Some(CellStatus::Refused));
        let result = study.run(&ExecutionContext::for_tests(8_401)).unwrap();
        assert!((ate(&result) - truth).abs() < 0.05, "{} vs {truth}", ate(&result));
        assert_eq!(result.identification.status, IdentificationStatus::GraphDependent);
        let mixture = result.structural_response.as_ref().unwrap();
        // The two identical samples are one atom with their combined weight.
        assert!((mixture.identified_mass - 0.7).abs() < 1e-9, "{}", mixture.identified_mass);
        assert!((mixture.unidentified_mass - 0.3).abs() < 1e-9);
        assert_eq!(result.logical_plan.estimator.as_deref(), Some("temporal.linear.adjustment"));
        assert!(has(&result, "estimate.graph_posterior.dbn_route"));
        assert!(result.posterior.is_none(), "a Frequentist point carries no posterior");
        assert!(result.interval.is_some());
    });
}

#[test]
fn the_bayesian_dynamic_conditional_pools_the_atoms_draws() {
    on_large_stack(|| {
        let pin: serde_json::Value = serde_json::from_str(include_str!(
            "../../../conformance/estimate/temporal_conditional/expected.json"
        ))
        .unwrap();
        let truth = pin["marginal_effect"].as_f64().unwrap();
        let template =
            |weight| Template { contemporaneous: vec![(2, 1)], lagged: vec![(0, 1)], weight };
        let result = Study::series(conditional_series())
            .graph_posterior(dbn_posterior(3, &[template(0.5), template(0.5)]))
            .query(CausalQuery::TemporalEffect(conditional_temporal_query()))
            .inference(InferenceMode::Bayesian(
                BayesianConfig::conjugate().prior_scale(1_000.0).n_draws(32),
            ))
            .refute(RefuteSuite::None)
            .build()
            .unwrap()
            .run(&ExecutionContext::for_tests(8_402))
            .unwrap();
        assert!((ate(&result) - truth).abs() < 0.05);
    });
}

fn attach(functional: ResponseFunctional) -> ResponseQuery {
    ResponseQuery::new(functional)
        .with_temporal(TemporalResponseSpec::new([1], TemporalPolicy::pulse(-1), Some(1)).unwrap())
}

#[test]
fn the_dynamic_derivative_mixture_matches_the_pinned_slope() {
    on_large_stack(|| {
        let pin: serde_json::Value = serde_json::from_str(include_str!(
            "../../../conformance/estimate/temporal_derivative/expected.json"
        ))
        .unwrap();
        let tolerance = pin["tolerance"].as_f64().unwrap();
        let truth = pin["point"].as_f64().unwrap();
        let template = |weight| Template {
            contemporaneous: vec![],
            lagged: vec![(0, 2), (1, 2), (0, 3), (1, 3)],
            weight,
        };
        let functional = ResponseFunctional::PointDerivative {
            outcome: v(2),
            treatment: v(0),
            at: 2.0,
            order: 1,
            scale: DerivativeScale::Identity,
        };
        for inference in [
            InferenceMode::Frequentist,
            InferenceMode::Bayesian(BayesianConfig::conjugate().prior_scale(1_000.0).n_draws(16)),
        ] {
            let result = Study::series(derivative_series())
                .graph_posterior(dbn_posterior(4, &[template(0.6), template(0.4)]))
                .query(CausalQuery::Response(attach(functional.clone())))
                .inference(inference)
                .refute(RefuteSuite::None)
                .build()
                .unwrap()
                .run(&ExecutionContext::for_tests(8_403))
                .unwrap();
            let response = result.response().unwrap();
            let ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) =
                &response.estimate
            else {
                panic!("a scalar derivative is a point")
            };
            assert!((value - truth).abs() < tolerance, "{value} vs {truth}");
            assert!(has(&result, "estimate.graph_posterior.dbn_route"));
        }
    });
}

#[test]
fn the_dynamic_cross_world_points_match_the_linear_mechanism() {
    on_large_stack(|| {
        let pin: serde_json::Value = serde_json::from_str(include_str!(
            "../../../conformance/estimate/temporal_cross_world/expected.json"
        ))
        .unwrap();
        let tolerance = pin["tolerance"].as_f64().unwrap();
        let (x, m, y) = (v(0), v(1), v(2));
        let queries = [
            (
                "nested",
                CausalQuery::NestedCounterfactual(NestedCounterfactualQuery::new(x, m, y).unwrap()),
            ),
            (
                "path",
                CausalQuery::PathSpecific(
                    PathSpecificEffectQuery::binary(x, y).with_path_nodes([m]),
                ),
            ),
            (
                "counterfactual",
                CausalQuery::Counterfactual(
                    CounterfactualQuery::new(y, Arc::from([Intervention::set(x, Value::f64(1.0))]))
                        .with_control_level(0.0),
                ),
            ),
            (
                "distribution",
                CausalQuery::Distribution(InterventionalDistributionQuery::new(
                    y,
                    [Intervention::set(x, Value::f64(1.0))],
                )),
            ),
        ];
        let template = |weight| Template {
            contemporaneous: vec![(1, 2)],
            lagged: vec![(0, 1), (0, 2)],
            weight,
        };
        for (key, query) in queries {
            let truth = pin[key].as_f64().unwrap();
            let result = Study::series(cross_series())
                .graph_posterior(dbn_posterior(3, &[template(0.7), template(0.3)]))
                .query(query)
                .refute(RefuteSuite::None)
                .build()
                .unwrap()
                .run(&ExecutionContext::for_tests(8_404))
                .unwrap();
            assert!((ate(&result) - truth).abs() < tolerance, "{key}: {} vs {truth}", ate(&result));
            assert!(has(&result, "estimate.graph_posterior.dbn_route"), "{key}");
            let mixture = result.structural_response.as_ref().unwrap();
            assert!((mixture.identified_mass - 1.0).abs() < 1e-9, "{key}");
        }
    });
}

// ---------------------------------------------------------------------------
// Cheap and full validation of the mixed cells.
// ---------------------------------------------------------------------------

fn frontdoor_pin() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../conformance/estimate/admg_frontdoor_functional/expected.json"
    ))
    .unwrap()
}

fn frontdoor_data(pin: &serde_json::Value) -> TabularData {
    let columns: Vec<&str> =
        pin["columns"].as_array().unwrap().iter().map(|c| c.as_str().unwrap()).collect();
    let mut values: Vec<Vec<f64>> = vec![Vec::new(); columns.len()];
    for cell in pin["contingency_table"].as_array().unwrap() {
        let count = usize::try_from(cell["count"].as_u64().unwrap()).unwrap();
        for (i, name) in columns.iter().enumerate() {
            values[i].extend(std::iter::repeat_n(cell[*name].as_f64().unwrap(), count));
        }
    }
    let pairs: Vec<(&str, &[f64])> =
        columns.iter().zip(values.iter()).map(|(n, c)| (*n, c.as_slice())).collect();
    TabularData::from_f64_columns(pairs).unwrap()
}

fn frontdoor_admg(pin: &serde_json::Value) -> Admg {
    let columns: Vec<&str> =
        pin["columns"].as_array().unwrap().iter().map(|c| c.as_str().unwrap()).collect();
    let index =
        |name: &str| node(u32::try_from(columns.iter().position(|c| *c == name).unwrap()).unwrap());
    let mut admg = Admg::with_variables(u32::try_from(columns.len()).unwrap());
    for edge in pin["graph"]["directed_edges"].as_array().unwrap() {
        admg.insert_directed(index(edge[0].as_str().unwrap()), index(edge[1].as_str().unwrap()))
            .unwrap();
    }
    for edge in pin["graph"]["bidirected_edges"].as_array().unwrap() {
        admg.insert_bidirected(index(edge[0].as_str().unwrap()), index(edge[1].as_str().unwrap()))
            .unwrap();
    }
    admg
}

#[test]
fn a_mixed_response_curve_is_refuted_by_refitting_the_whole_procedure() {
    on_large_stack(|| {
        let pin = frontdoor_pin();
        let data = frontdoor_data(&pin);
        let admg = frontdoor_admg(&pin);
        let n = admg.node_count();
        // A cyclic sample is not a member of the class: unidentified posterior mass.
        let cyclic = {
            let mut mask = 0_u64;
            mask = set_edge(mask, n, 0, 1, true);
            mask = set_edge(mask, n, 1, 2, true);
            set_edge(mask, n, 2, 0, true)
        };
        let posterior = GraphPosterior::new(
            n,
            vec![0.8, 0.2],
            vec![adjacency_mask_from_admg(&admg).unwrap(), cyclic],
            vec![0.0; n * n],
            vec![0.0; n * n],
            sampling("graph_posterior_functional_curve"),
        )
        .unwrap()
        .with_atom_kind(GraphPosteriorAtomKind::Admg);
        let curve = ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: v(2),
            treatment: antecedent_core::ContinuousDomain::new(
                v(0),
                antecedent_core::GridSpec::Values(Arc::from([0.0, 1.0])),
            ),
        });
        let run = |suite| {
            Study::tabular(data.clone())
                .graph_posterior(posterior.clone())
                .query(CausalQuery::Response(curve.clone()))
                .refute(suite)
                .bootstrap_replicates(0)
                .build()
                .unwrap()
                .run(&ExecutionContext::for_tests(8_501))
                .unwrap()
        };
        let plain = run(RefuteSuite::None);
        for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
            let result = run(suite);
            assert_eq!(result.support_status, Some(CellStatus::Refused));
            // The refuted curve is the curve the study publishes.
            let (a, b) = (result.response().unwrap(), plain.response().unwrap());
            assert_eq!(a.estimate, b.estimate);
            assert!(has(&result, "refute.functional.state"), "{suite:?}");
            let support = result
                .refutations
                .iter()
                .find(|r| r.refuter.as_ref() == "curve.atom_support")
                .expect("the support of the curve over the atoms");
            assert!(support.passed && support.replicates == 2);
            assert!((support.refuted_ate - 0.8).abs() < 1e-9, "{}", support.refuted_ate);
            assert!(result.refutations.len() >= 2, "{suite:?}: a whole-curve refuter ran");
        }
    });
}

#[test]
fn a_static_derivative_mixture_mixes_the_atoms_refuter_reports_by_weight() {
    on_large_stack(|| {
        let result = Study::tabular(derivative_data())
            .graph_posterior(dag_posterior(&[&PLAIN_EDGES, &CONFOUNDED_EDGES], &[0.6, 0.4]))
            .query(CausalQuery::Response(ResponseQuery::new(point_derivative())))
            .response_options(options())
            .refute(RefuteSuite::Cheap)
            .build()
            .unwrap()
            .run(&ExecutionContext::for_tests(8_502))
            .unwrap();
        assert_eq!(result.support_status, Some(CellStatus::Refused));
        assert!(has(&result, "refute.graph_posterior.per_atom"));
        assert!(!result.refutations.is_empty());
        // One report per refuter, not one per atom and refuter.
        let mut ids: Vec<&str> = result.refutations.iter().map(|r| r.refuter.as_ref()).collect();
        let before = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), before, "reports of one refuter are mixed: {ids:?}");
    });
}

#[test]
fn a_dynamic_derivative_is_refuted_atom_by_atom_and_mixed() {
    on_large_stack(|| {
        let template = |weight, lagged: Vec<(usize, usize)>| Template {
            contemporaneous: vec![],
            lagged,
            weight,
        };
        let functional = ResponseFunctional::PointDerivative {
            outcome: v(2),
            treatment: v(0),
            at: 2.0,
            order: 1,
            scale: DerivativeScale::Identity,
        };
        let posterior = dbn_posterior(
            4,
            &[
                template(0.6, vec![(0, 2), (1, 2), (0, 3), (1, 3)]),
                template(0.4, vec![(0, 2), (0, 3)]),
            ],
        );
        for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
            let result = Study::series(derivative_series())
                .graph_posterior(posterior.clone())
                .query(CausalQuery::Response(attach(functional.clone())))
                .refute(suite)
                .bootstrap_replicates(20)
                .build()
                .unwrap()
                .run(&ExecutionContext::for_tests(8_503))
                .unwrap();
            assert_eq!(result.support_status, Some(CellStatus::Refused), "{suite:?}");
            assert!(has(&result, "refute.graph_posterior.per_atom"), "{suite:?}");
            assert!(!result.refutations.is_empty(), "{suite:?}");
            let mut ids: Vec<&str> =
                result.refutations.iter().map(|r| r.refuter.as_ref()).collect();
            let before = ids.len();
            ids.sort_unstable();
            ids.dedup();
            assert_eq!(ids.len(), before, "{suite:?}: one mixed report per refuter {ids:?}");
        }
    });
}

#[test]
fn several_horizons_share_one_band_and_match_the_single_horizon_points() {
    on_large_stack(|| {
        let template = |weight| Template {
            contemporaneous: vec![],
            lagged: vec![(0, 2), (1, 2), (0, 3), (1, 3)],
            weight,
        };
        let functional = ResponseFunctional::PointDerivative {
            outcome: v(2),
            treatment: v(0),
            at: 2.0,
            order: 1,
            scale: DerivativeScale::Identity,
        };
        let with_horizons = |horizons: &[u32]| {
            ResponseQuery::new(functional.clone()).with_temporal(
                TemporalResponseSpec::new(horizons.to_vec(), TemporalPolicy::pulse(-1), Some(1))
                    .unwrap(),
            )
        };
        let run = |horizons: &[u32]| {
            Study::series(derivative_series())
                .graph_posterior(dbn_posterior(4, &[template(0.6), template(0.4)]))
                .query(CausalQuery::Response(with_horizons(horizons)))
                .refute(RefuteSuite::None)
                .bootstrap_replicates(40)
                .build()
                .unwrap()
                .run(&ExecutionContext::for_tests(8_504))
                .unwrap()
        };
        let joint = run(&[1, 2]);
        let first = run(&[1]);
        let response = joint.response().unwrap();
        let ResponseIdentification::PointIdentified(ResponseValue::Vector(values)) =
            &response.estimate
        else {
            panic!("several horizons publish one value per horizon: {:?}", response.estimate)
        };
        assert_eq!(values.len(), 2);
        // Horizons are never mixed: the first horizon is the single-horizon answer.
        let ResponseIdentification::PointIdentified(ResponseValue::Scalar(alone)) =
            &first.response().unwrap().estimate
        else {
            panic!("one horizon publishes a scalar")
        };
        assert!((values[0] - alone).abs() < 1e-9, "{} vs {alone}", values[0]);
        let ResponseUncertainty::PointwiseBand { lower, upper, .. } = &response.uncertainty else {
            panic!("a joint band over the horizons: {:?}", response.uncertainty)
        };
        assert_eq!((lower.len(), upper.len()), (2, 2));
        for k in 0..2 {
            assert!(lower[k] <= values[k] + 1e-9 && values[k] <= upper[k] + 1e-9);
        }
        // A Jacobian over several horizons is not a per-horizon slope.
        let refused = Study::series(derivative_series())
            .graph_posterior(dbn_posterior(4, &[template(1.0)]))
            .query(CausalQuery::Response(
                ResponseQuery::new(ResponseFunctional::Jacobian {
                    outcomes: Arc::from([v(2)]),
                    treatments: Arc::from([v(0)]),
                    at: Arc::from([2.0]),
                    scale: DerivativeScale::Identity,
                })
                .with_temporal(
                    TemporalResponseSpec::new([1, 2], TemporalPolicy::pulse(-1), Some(1)).unwrap(),
                ),
            ))
            .refute(RefuteSuite::None)
            .build()
            .and_then(|study| study.run(&ExecutionContext::for_tests(8_505)));
        assert!(refused.is_err());
    });
}
