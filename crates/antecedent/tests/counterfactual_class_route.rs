//! A counterfactual query on an ADMG, a temporal ADMG, a PAG and a CPDAG.
//!
//! Latent confounding breaks unit-level noise abduction, so on a latent-confounded structure the
//! query is the population-level cross-world mean `E[Y_x | X = x'] - E[Y | X = x']` decided by
//! ID*. The references are exact enumerations of finite latent-variable structural models (the
//! shared fixture in `antecedent-identify/tests/support`), which use no identification code: the
//! observational joint of a model becomes the table the study reads, and the effect on the treated
//! is enumerated over the model's exogenous states. A CPDAG assumes causal sufficiency and runs
//! the per-unit route on every DAG completion; its references are the known slopes of the
//! generating equations in each completion.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::too_many_lines,
    clippy::needless_range_loop,
    reason = "test fixtures: small exact indices, enumerated probabilities and counts"
)]

#[path = "../../antecedent-identify/tests/support/latent_scm.rs"]
mod latent_scm;

use antecedent::{BayesianConfig, CellStatus, Identification, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, CounterfactualQuery, ExecutionContext, Intervention, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Admg, Cpdag, Dag, DenseNodeId, Pag, TemporalAdmg};
use latent_scm::{LatentScm, Rng};

const ROWS: f64 = 60_000.0;

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn d(raw: u32) -> DenseNodeId {
    DenseNodeId::from_raw(raw)
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(41)
}

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

fn query(outcome: u32, treatment: u32) -> CausalQuery {
    CausalQuery::Counterfactual(CounterfactualQuery::new(
        v(outcome),
        [Intervention::set(v(treatment), Value::f64(1.0))],
    ))
}

fn admg(n: u32, directed: &[(u32, u32)], bidirected: &[(u32, u32)]) -> Admg {
    let mut graph = Admg::with_variables(n);
    for &(a, b) in directed {
        graph.insert_directed(d(a), d(b)).unwrap();
    }
    for &(a, b) in bidirected {
        graph.insert_bidirected(d(a), d(b)).unwrap();
    }
    graph
}

fn study(
    data: TabularData,
    graph: impl antecedent::IntoGraphInput,
    query: CausalQuery,
    inference: InferenceMode,
    replicates: u32,
) -> Study {
    Study::tabular(data)
        .graph(graph)
        .query(query)
        .inference(inference)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(replicates)
        .build()
        .unwrap()
}

fn front_door() -> (LatentScm, TabularData) {
    let scm = LatentScm::random(&mut Rng::new(8), &[2, 3, 2], &[(0, 1), (1, 2)], &[(0, 2)], 3);
    let data = table_of(&scm, &["x", "m", "y"]);
    (scm, data)
}

fn front_door_admg() -> Admg {
    admg(3, &[(0, 1), (1, 2)], &[(0, 2)])
}

fn has(result: &antecedent::StudyResult, code: &str) -> bool {
    result.diagnostics.iter().any(|d| d.code.as_ref() == code)
}

#[test]
fn an_admg_counterfactual_is_the_enumerated_effect_on_the_treated() {
    let (scm, data) = front_door();
    let truth = enumerated_effect(&scm, 0, 2);
    let built = study(data, front_door_admg(), query(2, 0), InferenceMode::Frequentist, 0);
    // The cell executes unlicensed: no evidence row names it.
    assert_eq!(built.support_status(), Some(CellStatus::Refused));
    let result = built.run(&ctx()).unwrap();
    assert!((result.effect() - truth).abs() < 5e-4, "{} vs {truth}", result.effect());
    // The estimand is the population-level cross-world mean, not a per-unit counterfactual.
    assert!(has(&result, "estimate.counterfactual.population_level"));
    assert!(has(&result, "estimate.counterfactual.id_star"));
    assert!(result.counterfactual().is_none(), "no per-unit effects are published");
    // No bootstrap was asked for, so the result says it carries no interval.
    assert!(has(&result, "estimate.counterfactual.uncertainty_unavailable"));
    // A different treatment contrast is a different truth: moving the units observed at level 0
    // to level 1 is not moving the units observed at level 1 to level 0.
    let reversed = CausalQuery::Counterfactual(
        CounterfactualQuery::new(v(2), [Intervention::set(v(0), Value::f64(0.0))])
            .with_control_level(1.0),
    );
    let (_, data) = front_door();
    let flipped = study(data, front_door_admg(), reversed, InferenceMode::Frequentist, 0)
        .run(&ctx())
        .unwrap();
    let numerators = scm.ett_numerators(0, 0, 1, 2);
    let observed = scm.ett_numerators(0, 1, 1, 2);
    let mean = |n: &[f64]| {
        n.iter().enumerate().map(|(l, p)| l as f64 * p).sum::<f64>() / n.iter().sum::<f64>()
    };
    assert!(
        (flipped.effect() - (mean(&numerators) - mean(&observed))).abs() < 5e-4,
        "{} vs {}",
        flipped.effect(),
        mean(&numerators) - mean(&observed)
    );
}

#[test]
fn the_frequentist_bootstrap_resamples_the_identified_functional() {
    let (scm, data) = front_door();
    let truth = enumerated_effect(&scm, 0, 2);
    let point = study(data.clone(), front_door_admg(), query(2, 0), InferenceMode::Frequentist, 0)
        .run(&ctx())
        .unwrap();
    let result = study(data, front_door_admg(), query(2, 0), InferenceMode::Frequentist, 80)
        .run(&ctx())
        .unwrap();
    assert!((result.effect() - truth).abs() < 5e-4);
    assert_eq!(result.effect(), point.effect(), "the bootstrap does not move the point");
    let effect = result.estimate().as_effect().unwrap();
    let se = effect.se_bootstrap().expect("a bootstrap standard error");
    // The sampling standard deviation of an effect from ~60k rows is of order 1/sqrt(n).
    assert!(se > 1e-4 && se < 0.05, "{se}");
    assert!(effect.uncertainty.bootstrap.replicates_ok.unwrap() >= 40);
    assert!(has(&result, "estimate.counterfactual.bootstrap"));
    assert!(!has(&result, "estimate.counterfactual.uncertainty_unavailable"));
}

#[test]
fn the_bayesian_bootstrap_posterior_centres_on_the_identified_functional() {
    let (scm, data) = front_door();
    let truth = enumerated_effect(&scm, 0, 2);
    let inference = InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(300));
    let result = study(data, front_door_admg(), query(2, 0), inference, 0).run(&ctx()).unwrap();
    let posterior = result.posterior.as_ref().expect("a posterior");
    assert_eq!(posterior.draws.n_draws, 300);
    assert!((result.effect() - truth).abs() < 0.01, "{} vs {truth}", result.effect());
    let sd = result.estimate().as_effect().unwrap().uncertainty.analytic.se;
    assert!(sd > 1e-4 && sd < 0.05, "{sd}");
    assert!(has(&result, "estimate.counterfactual.bayesian_bootstrap"));
}

#[test]
fn a_temporal_admg_reads_its_contemporaneous_projection_and_resamples_in_blocks() {
    let (scm, data) = front_door();
    let truth = enumerated_effect(&scm, 0, 2);
    let temporal = TemporalAdmg::from_static_contemporaneous(&front_door_admg()).unwrap();
    let result =
        study(data, temporal, query(2, 0), InferenceMode::Frequentist, 40).run(&ctx()).unwrap();
    assert!((result.effect() - truth).abs() < 5e-4, "{} vs {truth}", result.effect());
    let effect = result.estimate().as_effect().unwrap();
    assert!(effect.se_bootstrap().is_some_and(|se| se.is_finite()));
}

#[test]
fn an_unidentified_admg_returns_the_typed_id_star_refusal_with_its_witness() {
    // The bow arc: the effect on the treated is not identified from P(V).
    let scm = LatentScm::random(&mut Rng::new(2), &[2, 2], &[(0, 1)], &[(0, 1)], 3);
    let data = table_of(&scm, &["x", "y"]);
    let error =
        study(data, admg(2, &[(0, 1)], &[(0, 1)]), query(1, 0), InferenceMode::Frequentist, 0)
            .run(&ctx())
            .unwrap_err()
            .to_string();
    assert!(
        error.contains("reason=route_not_supported: counterfactual_id.conflicting_subscripts"),
        "{error}"
    );
    assert!(error.contains("not a proof of non-identifiability"), "{error}");
    assert!(!error.contains("cross_world_not_identified"), "{error}");
}

#[test]
fn a_pag_is_read_through_the_admgs_its_mag_stands_for() {
    // w -> x -> y with the edge w -> x invisible: the MAG stands for the ADMG without a latent
    // cause and the one with `w <-> x`. Both identify the effect on the treated of x on y, and
    // the enumerated model carries the latent, so the point is the model's truth.
    let scm = LatentScm::random(&mut Rng::new(4), &[2, 2, 2], &[(0, 1), (1, 2)], &[(0, 1)], 3);
    let truth = enumerated_effect(&scm, 1, 2);
    let data = table_of(&scm, &["w", "x", "y"]);
    let mut pag = Pag::with_variables(3);
    pag.insert_directed(d(0), d(1)).unwrap();
    pag.insert_directed(d(1), d(2)).unwrap();
    let result = study(data, pag, query(2, 1), InferenceMode::Frequentist, 0).run(&ctx()).unwrap();
    assert!((result.effect() - truth).abs() < 5e-4, "{} vs {truth}", result.effect());
    assert!(matches!(
        result.certificate.as_ref().map(|certificate| &certificate.identification),
        Some(Identification::Envelope { .. })
    ));
    assert!(has(&result, "identify.pag.envelope"));
    assert!(result.structural_response.is_none(), "the candidates agree: a point, not a set");
}

#[test]
fn a_pag_whose_mag_admits_a_bow_arc_candidate_is_refused_not_read_as_a_point() {
    // x -> y alone: the edge is invisible, so the MAG also stands for `x <-> y`, which ID* does
    // not identify; no functional is correct for every model the MAG represents.
    let scm = LatentScm::random(&mut Rng::new(2), &[2, 2], &[(0, 1)], &[], 3);
    let data = table_of(&scm, &["x", "y"]);
    let mut pag = Pag::with_variables(2);
    pag.insert_directed(d(0), d(1)).unwrap();
    let error = study(data, pag, query(1, 0), InferenceMode::Frequentist, 0)
        .run(&ctx())
        .unwrap_err()
        .to_string();
    assert!(error.contains("reason=route_not_supported: counterfactual_id."), "{error}");
}

#[test]
fn a_query_the_population_level_route_cannot_state_is_refused_with_the_contract_reason() {
    let (_, data) = front_door();
    let shifted = CausalQuery::Counterfactual(CounterfactualQuery::new(
        v(2),
        [Intervention::shift(v(0), Value::f64(1.0))],
    ));
    let error = study(data, front_door_admg(), shifted, InferenceMode::Frequentist, 0)
        .run(&ctx())
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("reason=route_not_supported: counterfactual_id.query_outside_contract"),
        "{error}"
    );
}

/// `x` on a grid, `y = 1 + 2 x` up to a small deterministic wiggle.
fn chain_table() -> TabularData {
    let x: Vec<f64> = (0..80).map(|i| f64::from(i) / 20.0 - 2.0).collect();
    let y: Vec<f64> =
        x.iter().enumerate().map(|(i, x)| 1.0 + 2.0 * x + 0.01 * (i as f64).sin()).collect();
    TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())]).unwrap()
}

#[test]
fn a_cpdag_whose_completions_disagree_publishes_the_mean_and_the_identified_set() {
    // x - y: the completions x -> y and y -> x are Markov equivalent. In the first the unit
    // effect of setting x to 1 on y is the slope 2; in the second y is a root and is unmoved.
    let mut cpdag = Cpdag::with_variables(2);
    cpdag.insert_undirected(d(0), d(1)).unwrap();
    let result = study(chain_table(), cpdag, query(1, 0), InferenceMode::Frequentist, 0)
        .run(&ctx())
        .unwrap();
    assert!((result.effect() - 1.0).abs() < 0.1, "{}", result.effect());
    let mixture = result.structural_response.as_ref().expect("completions disagree");
    assert_eq!(mixture.atoms.len(), 2);
    let set = mixture.identified_set.as_ref().expect("an identified set");
    assert!(set.lower[0].abs() < 0.05, "{:?}", set.lower);
    assert!((set.upper[0] - 2.0).abs() < 0.1, "{:?}", set.upper);
    assert!(has(&result, "estimate.counterfactual.completions_disagree"));
    // No interval pools two completions' values, and no per-unit effect belongs to either.
    assert!(!result.estimate().as_effect().unwrap().uncertainty.analytic.se.is_finite());
    assert!(result.counterfactual().is_none());
    assert!(matches!(
        result.certificate.as_ref().map(|certificate| &certificate.identification),
        Some(Identification::CpdagEnvelope { .. })
    ));
}

#[test]
fn a_cpdag_with_one_completion_is_the_per_unit_route_on_that_dag() {
    // x -> m <- w: the collider is compelled, so the class is one DAG.
    let w: Vec<f64> = (0..80).map(|i| ((f64::from(i) * 0.7).sin() * 3.0).round() / 3.0).collect();
    let x: Vec<f64> = (0..80).map(|i| f64::from(i) / 20.0 - 2.0).collect();
    let m: Vec<f64> = x
        .iter()
        .zip(&w)
        .enumerate()
        .map(|(i, (x, w))| 1.0 * x + 0.5 * w + 0.01 * (f64::from(i as u32) * 1.3).cos())
        .collect();
    let data = TabularData::from_f64_columns([
        ("x", x.as_slice()),
        ("w", w.as_slice()),
        ("m", m.as_slice()),
    ])
    .unwrap();
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(d(0), d(2)).unwrap();
    dag.insert_directed(d(1), d(2)).unwrap();
    let cpdag = Cpdag::from_dag(&dag);
    let class =
        study(data.clone(), cpdag, query(2, 0), InferenceMode::Frequentist, 0).run(&ctx()).unwrap();
    // The unit effect of moving x from 0 to 1 on `m = x + 0.5 w + noise` is the slope 1.
    assert!((class.effect() - 1.0).abs() < 0.1, "{}", class.effect());
    assert!(class.structural_response.is_none(), "one completion publishes a point");
    assert!(has(&class, "estimate.counterfactual.completions_agree"));
    // It is the same value the supplied-DAG route publishes.
    let supplied =
        study(data, dag, query(2, 0), InferenceMode::Frequentist, 0).run(&ctx()).unwrap();
    assert!((class.effect() - supplied.effect()).abs() < 1e-12);
    assert!(class.counterfactual().is_some(), "the unit effects of the single completion");
}

/// The class routes publish a contracted result that an independent consumer reads back.
#[test]
fn every_class_counterfactual_publishes_an_artifact_a_consumer_reads() {
    let (_, front) = front_door();
    let pag_scm = LatentScm::random(&mut Rng::new(4), &[2, 2, 2], &[(0, 1), (1, 2)], &[(0, 1)], 3);
    let pag_data = table_of(&pag_scm, &["w", "x", "y"]);
    let mut pag = Pag::with_variables(3);
    pag.insert_directed(d(0), d(1)).unwrap();
    pag.insert_directed(d(1), d(2)).unwrap();
    let mut cpdag = Cpdag::with_variables(2);
    cpdag.insert_undirected(d(0), d(1)).unwrap();
    let temporal = TemporalAdmg::from_static_contemporaneous(&front_door_admg()).unwrap();
    for (mode, inference) in [
        ("frequentist", InferenceMode::Frequentist),
        ("bayesian", InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(20))),
    ] {
        let cases = [
            ("admg", study(front.clone(), front_door_admg(), query(2, 0), inference.clone(), 0)),
            (
                "temporal-admg",
                study(front.clone(), temporal.clone(), query(2, 0), inference.clone(), 0),
            ),
            ("pag", study(pag_data.clone(), pag.clone(), query(2, 1), inference.clone(), 0)),
            ("cpdag", study(chain_table(), cpdag.clone(), query(1, 0), inference.clone(), 0)),
        ];
        for (class, built) in cases {
            let label = format!("{class} {mode}");
            let prepared = built.prepare(&ctx()).unwrap_or_else(|e| panic!("{label}: {e}"));
            let result =
                prepared.estimate_retained(&ctx()).unwrap_or_else(|e| panic!("{label}: {e}"));
            let artifact = prepared
                .encode_contracted_result(&result, &format!("class-counterfactual-{class}"), &ctx())
                .unwrap_or_else(|e| panic!("{label}: {e}"));
            let consumed = antecedent_io::consume_analysis_result(&artifact)
                .unwrap_or_else(|e| panic!("{label}: {e}"));
            assert_eq!(consumed.body.estimate, Some(result.effect()), "{label}");
        }
    }
}

/// One hard intervention on `treatment` to 1 against a control of 0, read on every outcome.
fn outcomes_query(outcomes: &[u32], treatment: u32) -> CausalQuery {
    let mut query = CounterfactualQuery::new(
        v(outcomes[0]),
        [Intervention::set(v(treatment), Value::f64(1.0))],
    )
    .with_control_level(0.0);
    query.outcomes = outcomes.iter().map(|&outcome| v(outcome)).collect::<Vec<_>>().into();
    CausalQuery::Counterfactual(query)
}

/// Two outcomes on a latent-confounded structure are each identified from the same graph and
/// published as one contrast per outcome, in the query's order; the first outcome's contrast is
/// bit for bit the one-outcome result.
#[test]
fn two_outcomes_on_an_admg_are_each_the_enumerated_effect_on_the_treated() {
    let (scm, data) = front_door();
    let (mediator_truth, outcome_truth) =
        (enumerated_effect(&scm, 0, 1), enumerated_effect(&scm, 0, 2));
    // The outcome first, the mediator second: the order is the query's, not the graph's.
    let result = study(
        data.clone(),
        front_door_admg(),
        outcomes_query(&[2, 1], 0),
        InferenceMode::Frequentist,
        0,
    )
    .run(&ctx())
    .unwrap();
    assert!(result.counterfactual().is_none(), "a contrast set is not the single form");
    let set = result.counterfactual_contrasts().expect("a contrast set payload");
    assert_eq!(set.treatments.as_ref(), [v(0)]);
    assert_eq!(set.contrasts.iter().map(|c| c.outcome).collect::<Vec<_>>(), [v(2), v(1)]);
    let (first, second) = (&set.contrasts[0], &set.contrasts[1]);
    assert!((first.mean_ite - outcome_truth).abs() < 5e-4, "{} vs {outcome_truth}", first.mean_ite);
    assert!(
        (second.mean_ite - mediator_truth).abs() < 5e-4,
        "{} vs {mediator_truth}",
        second.mean_ite
    );
    // A population-level contrast has no per-unit effects, flags or intervals.
    for contrast in &set.contrasts {
        assert!(contrast.unit_effects.is_empty() && contrast.unit_extrapolative.is_none());
        assert!(contrast.unit_effect_intervals.is_none() && contrast.standard_error.is_none());
    }
    // The scalar estimate is the first outcome's, and equals the one-outcome route's.
    let single = study(data.clone(), front_door_admg(), query(2, 0), InferenceMode::Frequentist, 0)
        .run(&ctx())
        .unwrap();
    assert!(single.counterfactual_contrasts().is_none(), "one outcome keeps the scalar result");
    assert_eq!(first.mean_ite.to_bits(), single.effect().to_bits());
    assert_eq!(result.effect().to_bits(), single.effect().to_bits());
    assert!(has(&result, "estimate.counterfactual.outcomes"));
    assert!(has(&result, "estimate.counterfactual.population_level"));

    // A temporal ADMG reads the same projection, so it answers the same two outcomes.
    let temporal = TemporalAdmg::from_static_contemporaneous(&front_door_admg()).unwrap();
    let projected =
        study(data, temporal, outcomes_query(&[2, 1], 0), InferenceMode::Frequentist, 0)
            .run(&ctx())
            .unwrap();
    let projected = projected.counterfactual_contrasts().expect("a contrast set payload");
    for (a, b) in projected.contrasts.iter().zip(&set.contrasts) {
        assert_eq!((a.outcome, a.mean_ite.to_bits()), (b.outcome, b.mean_ite.to_bits()));
    }
}

/// The resampled cells carry one uncertainty per outcome: a standard error for each outcome of a
/// row bootstrap, and one posterior column per outcome of the Bayesian bootstrap.
#[test]
fn two_outcomes_on_an_admg_carry_their_own_uncertainty() {
    let (_, data) = front_door();
    let bootstrapped = study(
        data.clone(),
        front_door_admg(),
        outcomes_query(&[2, 1], 0),
        InferenceMode::Frequentist,
        60,
    )
    .run(&ctx())
    .unwrap();
    let set = bootstrapped.counterfactual_contrasts().unwrap();
    for contrast in &set.contrasts {
        assert!(contrast.standard_error.is_some_and(|se| se > 1e-5 && se < 0.05));
    }
    assert!(!has(&bootstrapped, "estimate.counterfactual.uncertainty_unavailable"));

    let inference = InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(100));
    let bayesian = study(data, front_door_admg(), outcomes_query(&[2, 1], 0), inference, 0)
        .run(&ctx())
        .unwrap();
    let posterior = bayesian.posterior.as_ref().expect("a posterior");
    assert_eq!(posterior.summaries.mean.len(), 2, "one effect column per outcome");
    let set = bayesian.counterfactual_contrasts().unwrap();
    for (index, contrast) in set.contrasts.iter().enumerate() {
        assert!((contrast.mean_ite - posterior.summaries.mean[index]).abs() < 1e-12);
        assert!(contrast.standard_error.is_some_and(|se| se > 1e-5 && se < 0.05));
    }
}

/// A joint baseline of simultaneous interventions is refused on a latent-confounded structure
/// with the reason the identification does not state it, not as a blanket one-outcome refusal.
#[test]
fn a_joint_baseline_on_an_admg_is_refused_with_the_reason_it_cannot_be_stated() {
    let (_, data) = front_door();
    let joint = CausalQuery::Counterfactual(
        CounterfactualQuery::new(
            v(2),
            [Intervention::set(v(0), Value::f64(1.0)), Intervention::set(v(1), Value::f64(1.0))],
        )
        .with_joint_control([
            Intervention::set(v(0), Value::f64(0.0)),
            Intervention::set(v(1), Value::f64(0.0)),
        ]),
    );
    let error = study(data, front_door_admg(), joint, InferenceMode::Frequentist, 0)
        .run(&ctx())
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("reason=route_not_supported: counterfactual_id.query_outside_contract"),
        "{error}"
    );
    assert!(error.contains("intervenes on 2 variables at once"), "{error}");
    assert!(error.contains("one treatment per outcome event"), "{error}");
    assert!(!error.contains("one outcome"), "not the blanket one-outcome refusal: {error}");
}

/// `x -> m <- w`: the collider is compelled, so the CPDAG is one DAG and `m = x + 0.5 w + noise`.
fn collider() -> (TabularData, Dag) {
    let w: Vec<f64> = (0..80).map(|i| ((f64::from(i) * 0.7).sin() * 3.0).round() / 3.0).collect();
    let x: Vec<f64> = (0..80).map(|i| f64::from(i) / 20.0 - 2.0).collect();
    let m: Vec<f64> = x
        .iter()
        .zip(&w)
        .enumerate()
        .map(|(i, (x, w))| 1.0 * x + 0.5 * w + 0.01 * (f64::from(i as u32) * 1.3).cos())
        .collect();
    let data = TabularData::from_f64_columns([
        ("x", x.as_slice()),
        ("w", w.as_slice()),
        ("m", m.as_slice()),
    ])
    .unwrap();
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(d(0), d(2)).unwrap();
    dag.insert_directed(d(1), d(2)).unwrap();
    (data, dag)
}

/// A CPDAG runs the per-unit route on every DAG completion, so it answers several outcomes and a
/// joint baseline: one completion publishes the supplied-DAG contrast set on that DAG.
#[test]
fn a_cpdag_answers_several_outcomes_and_a_joint_baseline_through_its_completions() {
    let (data, dag) = collider();
    let cpdag = Cpdag::from_dag(&dag);
    // Two outcomes under one intervention: `m` moves by the slope 1, the root `w` not at all.
    let outcomes = outcomes_query(&[2, 1], 0);
    let class = study(data.clone(), cpdag.clone(), outcomes.clone(), InferenceMode::Frequentist, 0)
        .run(&ctx())
        .unwrap();
    let supplied = study(data.clone(), dag.clone(), outcomes, InferenceMode::Frequentist, 0)
        .run(&ctx())
        .unwrap();
    let (set, reference) = (
        class.counterfactual_contrasts().expect("a contrast set payload"),
        supplied.counterfactual_contrasts().unwrap(),
    );
    assert_eq!(set.contrasts.iter().map(|c| c.outcome).collect::<Vec<_>>(), [v(2), v(1)]);
    assert!((set.contrasts[0].mean_ite - 1.0).abs() < 0.1, "{}", set.contrasts[0].mean_ite);
    assert!(set.contrasts[1].mean_ite.abs() < 1e-6, "{}", set.contrasts[1].mean_ite);
    for (a, b) in set.contrasts.iter().zip(&reference.contrasts) {
        assert_eq!(a.mean_ite.to_bits(), b.mean_ite.to_bits());
        assert_eq!(a.unit_effects.len(), b.unit_effects.len());
    }
    assert!(has(&class, "estimate.counterfactual.completions_agree"));
    assert!(class.structural_response.is_none(), "one completion publishes a point");

    // A joint baseline moving both causes of `m` from 0 to 1 moves it by 1 + 0.5.
    let joint = CausalQuery::Counterfactual(
        CounterfactualQuery::new(
            v(2),
            [Intervention::set(v(0), Value::f64(1.0)), Intervention::set(v(1), Value::f64(1.0))],
        )
        .with_joint_control([
            Intervention::set(v(0), Value::f64(0.0)),
            Intervention::set(v(1), Value::f64(0.0)),
        ]),
    );
    let class = study(data, cpdag, joint, InferenceMode::Frequentist, 0).run(&ctx()).unwrap();
    let set = class.counterfactual_contrasts().expect("a contrast set payload");
    assert_eq!(set.treatments.as_ref(), [v(0), v(1)]);
    assert!((set.primary().mean_ite - 1.5).abs() < 0.1, "{}", set.primary().mean_ite);
}
