//! `ConditionalEffect` on a latent-confounded static ADMG.
//!
//! The effect is the IDC contrast `E[Y | do(active), Z] - E[Y | do(control), Z]`, averaged over
//! the observed law of the pre-treatment modifiers `Z` (the quantity the Dag/Cpdag/Pag
//! conditional routes report). Every expected value below is computed from the contingency
//! table by counting, never by calling the library: a stratified difference when the modifier
//! closes the back door, and a front-door formula within strata of the modifier when it does not.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::float_cmp,
    clippy::too_many_lines,
    reason = "test scaffolding compares constants computed by counting small contingency tables"
)]

use antecedent::analysis::RetainedOperation;
use antecedent::{
    AcceptedGraph, BayesianConfig, InferenceMode, RefuteSuite, StructureSource, Study,
};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ConditionalEffectQuery, ExecutionContext, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Admg, DenseNodeId, TemporalAdmg};

fn var(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

fn node(i: u32) -> DenseNodeId {
    DenseNodeId::from_raw(i)
}

fn bayes() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(256).prior_scale(1_000.0))
}

/// Expand `(cell levels, count)` rows into named f64 columns.
fn expand(names: &[&str], cells: &[(Vec<f64>, usize)]) -> TabularData {
    let mut columns: Vec<Vec<f64>> = vec![Vec::new(); names.len()];
    for (levels, count) in cells {
        for (column, level) in columns.iter_mut().zip(levels) {
            column.extend(std::iter::repeat_n(*level, *count));
        }
    }
    TabularData::from_f64_columns(
        names.iter().zip(&columns).map(|(name, column)| (*name, column.as_slice())),
    )
    .unwrap()
}

fn conditional(treatment: u32, outcome: u32, modifiers: &[u32]) -> CausalQuery {
    let modifiers: Vec<VariableId> = modifiers.iter().map(|&m| var(m)).collect();
    CausalQuery::ConditionalEffect(
        ConditionalEffectQuery::try_new(
            AverageEffectQuery::binary_ate(var(treatment), var(outcome))
                .with_effect_modifiers(modifiers),
        )
        .unwrap(),
    )
}

/// `Z -> X`, `Z -> Y`, `X -> Y`, `Z <-> X`: a latent common cause of the modifier and the
/// treatment. Conditioning on `Z` closes the back door of `X -> Y`.
fn confounded_modifier_graph() -> Admg {
    let mut graph = Admg::with_variables(3);
    graph.insert_directed(node(0), node(1)).unwrap();
    graph.insert_directed(node(0), node(2)).unwrap();
    graph.insert_directed(node(1), node(2)).unwrap();
    graph.insert_bidirected(node(0), node(1)).unwrap();
    graph
}

/// Cells `(z, x, y)` with their counts. Every stratum holds both treatment arms.
fn confounded_modifier_cells() -> Vec<(Vec<f64>, usize)> {
    let counts = [30, 10, 15, 25, 12, 28, 8, 52];
    let mut cells = Vec::new();
    for (index, count) in counts.iter().enumerate() {
        let (z, x, y) = ((index >> 2) & 1, (index >> 1) & 1, index & 1);
        cells.push((vec![z as f64, x as f64, y as f64], *count));
    }
    cells
}

/// `sum_z P(z) (E[Y | x=1, z] - E[Y | x=0, z])`, by counting.
fn stratified_difference(cells: &[(Vec<f64>, usize)]) -> f64 {
    let total: usize = cells.iter().map(|c| c.1).sum();
    let mut value = 0.0;
    for z in [0.0, 1.0] {
        let in_stratum = |c: &&(Vec<f64>, usize)| c.0[0] == z;
        let stratum: usize = cells.iter().filter(in_stratum).map(|c| c.1).sum();
        let mean = |x: f64| {
            let arm: usize =
                cells.iter().filter(in_stratum).filter(|c| c.0[1] == x).map(|c| c.1).sum();
            let ones: usize = cells
                .iter()
                .filter(in_stratum)
                .filter(|c| c.0[1] == x && c.0[2] == 1.0)
                .map(|c| c.1)
                .sum();
            ones as f64 / arm as f64
        };
        value += stratum as f64 / total as f64 * (mean(1.0) - mean(0.0));
    }
    value
}

fn ate(result: &antecedent::StudyResult) -> f64 {
    result.estimate.as_effect().unwrap().ate
}

#[test]
fn a_latent_confounded_modifier_gives_the_stratified_difference() {
    let cells = confounded_modifier_cells();
    let data = expand(&["z", "x", "y"], &cells);
    let expected = stratified_difference(&cells);
    // The two strata have different effects, so a pooled difference or a one-stratum answer
    // would not reproduce the weighted average.
    assert!((expected - 0.259_259_259_259_259_3).abs() < 1e-12, "{expected}");
    let ctx = ExecutionContext::for_tests(41);
    let study = Study::tabular(data.clone())
        .graph(confounded_modifier_graph())
        .query(conditional(1, 2, &[0]))
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let fresh = study.clone().run(&ctx).unwrap();
    assert_eq!(fresh.logical_plan.identifier.as_deref(), Some("general.id"));
    assert_eq!(fresh.logical_plan.estimator.as_deref(), Some("functional.effect"));
    assert!((ate(&fresh) - expected).abs() < 1e-12, "{} vs {expected}", ate(&fresh));
    assert!(fresh.posterior.is_none(), "a Frequentist run attaches no posterior");
    assert!(
        fresh
            .diagnostics
            .iter()
            .any(|d| d.code.as_ref() == "estimate.functional.free_variables_averaged"),
        "the modifier average must be disclosed"
    );
    // The prepared handle retains the checked program and reuses the identification.
    let prepared = study.prepare(&ctx).unwrap();
    assert!(matches!(
        prepared.retained_operation(),
        RetainedOperation::FunctionalEffect(..) | RetainedOperation::PathSpecificEffect(..)
    ));
    let click = prepared.estimate(&data, &ctx).unwrap();
    assert!((ate(&click) - expected).abs() < 1e-12);
    assert!(click.diagnostics.iter().any(|d| d.code.as_ref() == "exec.identify.cached"));
}

#[test]
#[allow(
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "the quantile is inside [0, 1] and the draw count fits an index"
)]
fn the_bayesian_arm_is_a_dirichlet_posterior_of_the_same_contrast() {
    let cells = confounded_modifier_cells();
    let data = expand(&["z", "x", "y"], &cells);
    let expected = stratified_difference(&cells);
    let ctx = ExecutionContext::for_tests(41);
    let study = Study::tabular(data.clone())
        .graph(confounded_modifier_graph())
        .query(conditional(1, 2, &[0]))
        .inference(bayes())
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    for result in vec![
        study.clone().run(&ctx).unwrap(),
        study.prepare(&ctx).unwrap().estimate(&data, &ctx).unwrap(),
    ] {
        assert_eq!(result.logical_plan.estimator.as_deref(), Some("functional.effect"));
        let posterior = result.posterior.as_ref().expect("a Bayesian run attaches its posterior");
        assert_eq!(posterior.diagnostics.backend_id.as_ref(), "functional.dirichlet");
        let column = posterior.effect_column().expect("effect draws");
        let mut draws = posterior.draws.column(column).unwrap().to_vec();
        let mean = draws.iter().sum::<f64>() / draws.len() as f64;
        // Rubin's Bayesian bootstrap is centred on the plug-in; its Monte Carlo error with 256
        // draws and an effect sd near 0.07 is about 0.005.
        assert!((mean - expected).abs() < 0.03, "posterior mean {mean} vs {expected}");
        draws.sort_by(f64::total_cmp);
        let at = |q: f64| draws[((draws.len() - 1) as f64 * q).round() as usize];
        assert!(at(0.05) <= expected && expected <= at(0.95), "90% interval around the plug-in");
        assert!(at(0.95) - at(0.05) > 0.05, "a posterior has spread, not a point");
    }
}

/// `Z -> X`, `Z -> M`, `Z -> Y`, `X -> M`, `M -> Y`, `X <-> Y`: the front-door graph with a
/// pre-treatment modifier. `P(y | do(x), z)` needs IDC's line 1 and the front-door functional.
fn frontdoor_modifier_graph() -> Admg {
    let mut graph = Admg::with_variables(4);
    for (a, b) in [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)] {
        graph.insert_directed(node(a), node(b)).unwrap();
    }
    graph.insert_bidirected(node(1), node(3)).unwrap();
    graph
}

/// Counts of a law with a strong front-door effect: `P(z) = 1/2`, `P(x=1 | z) = .3 + .4 z`,
/// `P(m=1 | x, z) = .2 + .5 x + .1 z`, `P(y=1 | m, z, x) = .15 + .5 m + .1 z + .1 x`, scaled by
/// 4000 and rounded (the dependence of `y` on `x` stands for the latent `x <-> y` arrow).
#[allow(clippy::cast_sign_loss, reason = "each cell count is a non-negative rounded mass")]
fn frontdoor_modifier_cells() -> Vec<(Vec<f64>, usize)> {
    (0..16usize)
        .map(|i| {
            let (z, x, m, y) = ((i >> 3) & 1, (i >> 2) & 1, (i >> 1) & 1, i & 1);
            let (zf, xf, mf) = (z as f64, x as f64, m as f64);
            let bernoulli = |p: f64, level: usize| if level == 1 { p } else { 1.0 - p };
            let mass = 0.5
                * bernoulli(0.3 + 0.4 * zf, x)
                * bernoulli(0.2 + 0.5 * xf + 0.1 * zf, m)
                * bernoulli(0.15 + 0.5 * mf + 0.1 * zf + 0.1 * xf, y);
            (vec![zf, xf, mf, y as f64], (4000.0 * mass).round() as usize)
        })
        .collect()
}

/// `sum_z P(z) (P(y=1 | do(1), z) - P(y=1 | do(0), z))` with the front-door formula
/// `P(y | do(x), z) = sum_m P(m | x, z) sum_x' P(y | x', m, z) P(x' | z)`, by counting.
fn frontdoor_within_strata(cells: &[(Vec<f64>, usize)]) -> f64 {
    let n = |z: f64, x: Option<f64>, m: Option<f64>, y: Option<f64>| -> f64 {
        cells
            .iter()
            .filter(|c| {
                c.0[0] == z
                    && x.is_none_or(|v| c.0[1] == v)
                    && m.is_none_or(|v| c.0[2] == v)
                    && y.is_none_or(|v| c.0[3] == v)
            })
            .map(|c| c.1 as f64)
            .sum()
    };
    let total = n(0.0, None, None, None) + n(1.0, None, None, None);
    let mut value = 0.0;
    for z in [0.0, 1.0] {
        let p_do = |x: f64| -> f64 {
            let mut sum = 0.0;
            for m in [0.0, 1.0] {
                let p_m = n(z, Some(x), Some(m), None) / n(z, Some(x), None, None);
                let mut inner = 0.0;
                for x_prime in [0.0, 1.0] {
                    let p_y = n(z, Some(x_prime), Some(m), Some(1.0))
                        / n(z, Some(x_prime), Some(m), None);
                    inner += p_y * n(z, Some(x_prime), None, None) / n(z, None, None, None);
                }
                sum += p_m * inner;
            }
            sum
        };
        value += n(z, None, None, None) / total * (p_do(1.0) - p_do(0.0));
    }
    value
}

#[test]
fn a_front_door_effect_within_strata_of_the_modifier_matches_the_counted_formula() {
    let cells = frontdoor_modifier_cells();
    let data = expand(&["z", "x", "m", "y"], &cells);
    let expected = frontdoor_within_strata(&cells);
    // Distinct from the naive stratified contrast (0.35), so an adjustment-for-z answer cannot
    // pass.
    assert!((expected - 0.25).abs() < 1e-9, "{expected}");
    let ctx = ExecutionContext::for_tests(43);
    let study = Study::tabular(data.clone())
        .graph(frontdoor_modifier_graph())
        .query(conditional(1, 3, &[0]))
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let fresh = study.clone().run(&ctx).unwrap();
    assert!(
        (ate(&fresh) - expected).abs() < 1e-12,
        "functional {} vs front-door formula {expected}",
        ate(&fresh)
    );
    let click = study.prepare(&ctx).unwrap().estimate(&data, &ctx).unwrap();
    assert!((ate(&click) - ate(&fresh)).abs() < 1e-14);
}

#[test]
fn a_bow_is_not_identified_and_a_prior_does_not_change_that() {
    // `X <-> Y` with `X -> Y` and an unrelated modifier: the hedge blocks both arms.
    let mut graph = Admg::with_variables(3);
    graph.insert_directed(node(0), node(1)).unwrap();
    graph.insert_bidirected(node(0), node(1)).unwrap();
    let cells: Vec<(Vec<f64>, usize)> = (0..8usize)
        .map(|i| (vec![((i >> 2) & 1) as f64, ((i >> 1) & 1) as f64, (i & 1) as f64], 5 + i))
        .collect();
    let data = expand(&["x", "y", "w"], &cells);
    let ctx = ExecutionContext::for_tests(44);
    for inference in [InferenceMode::Frequentist, bayes()] {
        let outcome = Study::tabular(data.clone())
            .graph(graph.clone())
            .query(conditional(0, 1, &[2]))
            .inference(inference)
            .refute(RefuteSuite::None)
            .build()
            .and_then(|study| study.run(&ctx));
        let error = outcome.expect_err("a bow is not identified");
        assert!(error.to_string().contains("effect_not_identified"), "{error}");
    }
}

#[test]
fn a_post_treatment_modifier_is_refused_and_names_the_distribution_query() {
    let mut graph = Admg::with_variables(3);
    graph.insert_directed(node(0), node(1)).unwrap();
    graph.insert_directed(node(1), node(2)).unwrap();
    graph.insert_bidirected(node(0), node(2)).unwrap();
    let cells: Vec<(Vec<f64>, usize)> = (0..8usize)
        .map(|i| (vec![((i >> 2) & 1) as f64, ((i >> 1) & 1) as f64, (i & 1) as f64], 6 + i))
        .collect();
    let data = expand(&["x", "m", "y"], &cells);
    let ctx = ExecutionContext::for_tests(45);
    let error = Study::tabular(data)
        .graph(graph)
        .query(conditional(0, 2, &[1]))
        .refute(RefuteSuite::None)
        .build()
        .and_then(|study| study.run(&ctx))
        .expect_err("a mediator is not a pre-treatment modifier");
    let text = error.to_string();
    assert!(text.contains("pre-treatment"), "{text}");
    assert!(text.contains("InterventionalDistribution"), "{text}");
}

#[test]
fn a_temporal_admg_contrast_is_refused_and_names_the_licensed_routes() {
    let graph = confounded_modifier_graph();
    let temporal = TemporalAdmg::from_static_contemporaneous(&graph).unwrap();
    let data = expand(&["z", "x", "y"], &confounded_modifier_cells());
    let ctx = ExecutionContext::for_tests(46);
    let error = Study::tabular(data)
        .graph(temporal)
        .query(conditional(1, 2, &[0]))
        .refute(RefuteSuite::None)
        .build()
        .and_then(|study| study.run(&ctx))
        .expect_err("a temporal contrast on a TemporalAdmg is not identified");
    assert!(error.to_string().contains("TemporalAdmg"), "{error}");
}

/// The static ADMG average effect is the same Dirichlet posterior with the same procedure id,
/// so a conditional posterior and an average-effect posterior are told apart by their query,
/// not by a different engine.
#[test]
fn the_static_admg_average_effect_posterior_names_its_procedure() {
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/admg_frontdoor_functional/expected.json"
    ))
    .unwrap();
    let columns: Vec<&str> =
        pin["columns"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    let cells: Vec<(Vec<f64>, usize)> = pin["contingency_table"]
        .as_array()
        .unwrap()
        .iter()
        .map(|cell| {
            (
                columns.iter().map(|name| cell[*name].as_f64().unwrap()).collect(),
                usize::try_from(cell["count"].as_u64().unwrap()).unwrap(),
            )
        })
        .collect();
    let data = expand(&columns, &cells);
    let mut graph = Admg::with_variables(3);
    graph.insert_directed(node(0), node(1)).unwrap();
    graph.insert_directed(node(1), node(2)).unwrap();
    graph.insert_bidirected(node(0), node(2)).unwrap();
    let ctx = ExecutionContext::for_tests(47);
    let query = AverageEffectQuery::with_levels(var(0), var(2), 0.0, 1.0);
    let build = |inference: InferenceMode| {
        Study::tabular(data.clone())
            .graph(graph.clone())
            .query(query.clone())
            .inference(inference)
            .refute(RefuteSuite::None)
            .build()
            .unwrap()
            .run(&ctx)
            .unwrap()
    };
    let bayesian = build(bayes());
    let frequentist = build(InferenceMode::Frequentist);
    let posterior = bayesian.posterior.as_ref().expect("Bayesian ADMG posterior");
    assert_eq!(posterior.diagnostics.backend_id.as_ref(), "functional.dirichlet");
    assert!(
        posterior.assumptions.entries.iter().any(|record| matches!(
            &record.assumption,
            antecedent_core::Assumption::ParametricRestriction(p)
                if p.id.as_ref() == "functional.empirical_support_prior"
        )),
        "the posterior states its row-law prior"
    );
    assert!(frequentist.posterior.is_none());
    assert_eq!(bayesian.logical_plan.estimator, frequentist.logical_plan.estimator);
}

/// Drive one coordinate through prepare, estimate and refresh on the retained plan, with the
/// builder dropped, and compare the contrast with the counted stratified difference.
fn verify_conditional_coordinate(coordinate: &str, accepted: bool, bayesian: bool) {
    let cells = confounded_modifier_cells();
    let data = expand(&["z", "x", "y"], &cells);
    let expected = stratified_difference(&cells);
    let ctx = ExecutionContext::for_tests(41);
    let base = Study::tabular(data.clone())
        .query(conditional(1, 2, &[0]))
        .inference(if bayesian { bayes() } else { InferenceMode::Frequentist })
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0);
    let graph = confounded_modifier_graph();
    let builder = if accepted { base.graph(AcceptedGraph::from(graph)) } else { base.graph(graph) };
    let study = builder.clone().build().unwrap_or_else(|error| panic!("{coordinate}: {error}"));
    let mut prepared = study.prepare(&ctx).unwrap_or_else(|error| panic!("{coordinate}: {error}"));
    drop(builder);
    drop(study);

    assert_eq!(
        prepared.structure_source(),
        if accepted { StructureSource::Accepted } else { StructureSource::Explicit },
        "{coordinate} structure source"
    );
    assert_eq!(prepared.support_status(), Some(antecedent::CellStatus::Licensed), "{coordinate}");
    assert_eq!(prepared.plan().logical.record.identifier.as_deref(), Some("general.id"));
    assert_eq!(prepared.plan().logical.record.estimator.as_deref(), Some("functional.effect"));
    let result =
        prepared.estimate(&data, &ctx).unwrap_or_else(|error| panic!("{coordinate}: {error}"));
    let tolerance = if bayesian { 0.03 } else { 1e-12 };
    assert!(
        (result.effect() - expected).abs() < tolerance,
        "{coordinate}: {} vs {expected}",
        result.effect()
    );
    assert_eq!(result.posterior.is_some(), bayesian, "{coordinate} posterior presence");
    let refreshed = prepared.refresh(data, &ctx).unwrap();
    assert_eq!(refreshed.effect().to_bits(), result.effect().to_bits(), "{coordinate} refresh");
}

#[test]
fn admg_conditional_effect_coordinates_execute_from_the_retained_plan() {
    verify_conditional_coordinate("ConditionalEffect:Admg:explicit:Frequentist:none", false, false);
    verify_conditional_coordinate("ConditionalEffect:Admg:accepted:Frequentist:none", true, false);
    verify_conditional_coordinate("ConditionalEffect:Admg:explicit:Bayesian:none", false, true);
    verify_conditional_coordinate("ConditionalEffect:Admg:accepted:Bayesian:none", true, true);
}
