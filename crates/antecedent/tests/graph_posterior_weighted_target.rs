//! A graph-posterior study for a declared target population, against a hand mixture.
//!
//! The planted effect is `tau(z) = 1 + 0.5 z` of a binary exposure `t` whose assignment and
//! outcome both depend on a covariate `z`. The target is a Gaussian-kernel neighbourhood of
//! `z`, declared with `weighted_target`. The posterior holds three structures over `(z, t, y)`:
//!
//! * `z -> t, z -> y, t -> y` (mass 0.6): the covariate is adjusted, the effect is estimated;
//! * `z -> t, z -> y` (mass 0.3): no directed path from `t` to `y`, so the effect is exactly zero
//!   on every population;
//! * `t -> y` (mass 0.1): the covariate is not in the adjustment set, so the target's weights
//!   are not identified by it and the structure is refused for the target. Its mass is reported
//!   as unanswered and is not redistributed.
//!
//! The route must equal the hand mixture of single-structure studies over the answered mass,
//! with the standard error `sqrt((sum w se)^2 + between-structure variance)`.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, reason = "small graph node counts")]
#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    reason = "integration test with a planted truth and a hand-composed reference"
)]

mod common;

use antecedent::inference::{KernelFamily, TransportKernel};
use antecedent::{AverageEffectQuery, RefuteSuite, Study, StudyResult, WeightedTargetDeclaration};
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_discovery::adjacency_mask::{AdjacencyMask, set_edge};
use antecedent_discovery::{GraphPosterior, SamplingSummary};
use antecedent_estimate::AipwAte;
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_prob::InferenceDiagnostics;
use common::calibration::{gaussian, uniform};

const ROWS: usize = 3_000;
const Z: usize = 0;
const T: usize = 1;
const Y: usize = 2;
const CENTRE: f64 = 0.5;
const BANDWIDTH: f64 = 0.7;
/// Masses of the three structures: estimated, no path, refused for the target.
const MASSES: [f64; 3] = [0.6, 0.3, 0.1];

fn v(raw: usize) -> VariableId {
    VariableId::from_raw(u32::try_from(raw).unwrap())
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(17)
}

fn tau(z: f64) -> f64 {
    1.0 + 0.5 * z
}

fn data() -> TabularData {
    let mut normal = gaussian(91);
    let mut next = uniform(91, 1);
    let z: Vec<f64> = (0..ROWS).map(|_| normal()).collect();
    let t: Vec<f64> =
        z.iter().map(|&zi| f64::from(u8::from(next() < 1.0 / (1.0 + (-0.8 * zi).exp())))).collect();
    let y: Vec<f64> = (0..ROWS).map(|i| 0.5 * z[i] + t[i] * tau(z[i]) + 0.5 * normal()).collect();
    TabularData::from_f64_columns([("z", z.as_slice()), ("t", t.as_slice()), ("y", y.as_slice())])
        .unwrap()
}

fn target() -> WeightedTargetDeclaration {
    WeightedTargetDeclaration::kernel(
        TransportKernel::new(KernelFamily::Gaussian, BANDWIDTH).unwrap(),
        vec![v(Z)],
        vec![CENTRE],
        None,
    )
    .unwrap()
}

/// The directed edges of the three structures.
fn structures() -> [Vec<(usize, usize)>; 3] {
    [vec![(Z, T), (Z, Y), (T, Y)], vec![(Z, T), (Z, Y)], vec![(T, Y)]]
}

fn mask_of(edges: &[(usize, usize)]) -> AdjacencyMask {
    let mut mask = set_edge(0_u64, 3, edges[0].0, edges[0].1, true);
    for &(a, b) in &edges[1..] {
        mask = set_edge(mask, 3, a, b, true);
    }
    mask
}

fn posterior() -> GraphPosterior {
    let masks: Vec<AdjacencyMask> = structures().iter().map(|edges| mask_of(edges)).collect();
    let mut marginals = vec![0.0; 9];
    for (edges, weight) in structures().iter().zip(MASSES) {
        for &(a, b) in edges {
            marginals[a * 3 + b] += weight;
        }
    }
    GraphPosterior::from_atoms(
        3,
        MASSES.to_vec(),
        masks,
        marginals.clone(),
        marginals,
        SamplingSummary {
            ess: 1.0 / MASSES.iter().map(|w| w * w).sum::<f64>(),
            diagnostics: InferenceDiagnostics::analytic("weighted_target_known_truth"),
            rejected_invalid: 0,
        },
    )
    .unwrap()
    .with_algorithm("weighted_target_known_truth")
}

fn dag(edges: &[(usize, usize)]) -> Dag {
    let mut dag = Dag::with_variables(3);
    for &(a, b) in edges {
        dag.insert_directed(DenseNodeId::from_raw(a as u32), DenseNodeId::from_raw(b as u32))
            .unwrap();
    }
    dag
}

fn estimator() -> AipwAte {
    AipwAte::new().with_bootstrap_replicates(0)
}

fn query() -> AverageEffectQuery {
    AverageEffectQuery::binary_ate(v(T), v(Y))
}

/// One structure, one study, the same target.
fn single(edges: &[(usize, usize)]) -> (f64, f64) {
    let result = Study::tabular(data())
        .graph(dag(edges))
        .estimator(estimator())
        .refute(RefuteSuite::None)
        .weighted_target(target())
        .query(query())
        .build()
        .unwrap()
        .run(&ctx())
        .unwrap();
    let effect = result.estimate().as_effect().unwrap();
    (effect.ate, effect.se_analytic())
}

fn route() -> StudyResult {
    Study::tabular(data())
        .graph_posterior(posterior())
        .estimator(estimator())
        .refute(RefuteSuite::None)
        .weighted_target(target())
        .query(query())
        .build()
        .unwrap()
        .run(&ctx())
        .unwrap()
}

#[test]
fn the_route_equals_the_hand_mixture_over_the_answered_mass() {
    let result = route();
    let structural = result.structural_response.as_ref().expect("the structural mixture");
    assert!(structural.unidentified_mass < 1e-12, "{}", structural.unidentified_mass);
    // The covariate-free structure is refused for the target: its mass is reported unanswered.
    assert!(
        (structural.unevaluable_mass - MASSES[2]).abs() < 1e-12,
        "{}",
        structural.unevaluable_mass
    );

    let (tau_a, se_a) = single(&structures()[0]);
    let answered = MASSES[0] + MASSES[1];
    let mixed = MASSES[0] * tau_a / answered;
    let within = (MASSES[0] * se_a) / answered;
    let between = (MASSES[0] * (tau_a - mixed).powi(2) + MASSES[1] * mixed.powi(2)) / answered;
    let se = (within * within + between).sqrt();

    let effect = result.estimate().as_effect().unwrap();
    assert!((effect.ate - mixed).abs() < 1e-9, "route {} against hand {mixed}", effect.ate);
    assert!(
        (effect.se_analytic() - se).abs() < 1e-9 * se.max(1.0),
        "route se {} against hand {se}",
        effect.se_analytic()
    );
    // The mixture is the effect on the declared population, not on every row: it sits near the
    // answered share of the closed-form kernel-weighted effect of the estimated structure.
    assert!(between > 0.0);
}

#[test]
fn the_route_names_the_target_and_the_mixture_in_its_record() {
    let result = route();
    let effect = result.estimate().as_effect().unwrap();
    let ids: Vec<String> = effect
        .provenance
        .assumptions
        .entries
        .iter()
        .filter_map(|record| match &record.assumption {
            antecedent_core::Assumption::ParametricRestriction(declared) => {
                Some(declared.id.to_string())
            }
            _ => None,
        })
        .collect();
    assert!(ids.iter().any(|id| id == "estimate.weighted_target_population"), "{ids:?}");
    assert!(
        result
            .diagnostics
            .iter()
            .any(|d| d.code.as_ref() == "estimate.graph_posterior.weighted_target_mixture"
                && d.message.contains("unanswered_mass=0.1")),
        "the mixture note states the unanswered mass"
    );
}

#[test]
fn a_study_without_a_target_keeps_the_default_mixture_rule() {
    // Without a declared population the structures disagree on their adjustment sets and the
    // scalar is withheld, exactly as before the target seam existed.
    let result = Study::tabular(data())
        .graph_posterior(posterior())
        .refute(RefuteSuite::None)
        .query(query())
        .build()
        .unwrap()
        .run(&ctx())
        .unwrap();
    assert!(
        result
            .diagnostics
            .iter()
            .all(|d| d.code.as_ref() != "estimate.graph_posterior.weighted_target_mixture")
    );
}

// ---- the Bayesian path ----------------------------------------------------------------------

fn bayesian() -> antecedent::inference::InferenceMode {
    antecedent::inference::InferenceMode::Bayesian(
        antecedent::inference::BayesianConfig::conjugate().n_draws(2_000).prior_scale(100.0),
    )
}

fn single_bayesian_mean(edges: &[(usize, usize)]) -> f64 {
    let result = Study::tabular(data())
        .graph(dag(edges))
        .inference(bayesian())
        .refute(RefuteSuite::None)
        .weighted_target(target())
        .query(query())
        .build()
        .unwrap()
        .run(&ctx())
        .unwrap();
    result.estimate().as_effect().unwrap().ate
}

#[test]
fn the_bayesian_route_equals_the_hand_mixture_of_per_structure_posteriors() {
    let result = Study::tabular(data())
        .graph_posterior(posterior())
        .inference(bayesian())
        .refute(RefuteSuite::None)
        .weighted_target(target())
        .query(query())
        .build()
        .unwrap()
        .run(&ctx())
        .unwrap();
    // The structure the target's covariate cannot be adjusted in is refused: its mass is
    // reported as unanswered and not redistributed.
    let structural = result.structural_response.as_ref().expect("the structural mixture");
    assert!(
        (structural.unevaluable_mass - MASSES[2]).abs() < 1e-9,
        "unanswered mass {}",
        structural.unevaluable_mass
    );
    let published = result.posterior.as_ref().expect("a Bayesian mixture posterior");
    assert!((published.unevaluable_mass - MASSES[2]).abs() < 1e-9);

    // The structure without a path contributes a point mass at exactly zero.
    let answered = MASSES[0] + MASSES[1];
    let mean_a = single_bayesian_mean(&structures()[0]);
    let hand = MASSES[0] * mean_a / answered;
    let effect = result.estimate().as_effect().unwrap();
    assert!(
        (effect.ate - hand).abs() < 1e-3,
        "route {} against the hand mixture {hand} of {mean_a} and a point mass at zero",
        effect.ate
    );
}

/// A posterior whose one structure identifies the planted effect with the target's covariate in
/// the adjustment set, so every refuter replicate has that structure to re-estimate.
fn refutable_posterior() -> GraphPosterior {
    let masks = [mask_of(&structures()[0])];
    let mut marginals = vec![0.0; 9];
    for &(a, b) in &structures()[0] {
        marginals[a * 3 + b] = 1.0;
    }
    GraphPosterior::from_atoms(
        3,
        vec![1.0],
        masks,
        marginals.clone(),
        marginals,
        SamplingSummary {
            ess: 1.0,
            diagnostics: InferenceDiagnostics::analytic("weighted_target_refuters"),
            rejected_invalid: 0,
        },
    )
    .unwrap()
    .with_algorithm("weighted_target_refuters")
}

fn refuted(bayes: bool, suite: RefuteSuite) -> StudyResult {
    let builder = Study::tabular(data())
        .graph_posterior(refutable_posterior())
        .refute(suite)
        .weighted_target(target())
        .query(query());
    let builder =
        if bayes { builder.inference(bayesian()) } else { builder.estimator(estimator()) };
    builder.build().unwrap().run(&ctx()).unwrap()
}

fn report<'a>(result: &'a StudyResult, id: &str) -> &'a antecedent_validate::RefutationReport {
    result
        .refutations
        .iter()
        .find(|report| &*report.refuter == id)
        .unwrap_or_else(|| panic!("no {id} report in {:?}", result.refutations))
}

/// The all-rows effect of the planted `tau(z) = 1 + 0.5 z` is about 1; the kernel target around
/// `z = 0.5` has a larger one, so a refuter that re-estimated all rows would sit near 1.
fn assert_targeted(result: &StudyResult, id: &str) {
    let reported = report(result, id);
    let targeted = result.estimate().as_effect().unwrap().ate;
    assert!(
        (reported.refuted_ate - targeted).abs() < 0.15,
        "{id} re-estimated {} against the targeted effect {targeted}",
        reported.refuted_ate
    );
    assert!(
        (reported.refuted_ate - 1.0).abs() > 0.1,
        "{id} re-estimated {}, the all-rows effect",
        reported.refuted_ate
    );
}

#[test]
fn every_refuter_re_estimates_the_targeted_effect_under_frequentist_inference() {
    let result = refuted(false, RefuteSuite::Full);
    assert_targeted(&result, "random.common_cause");
    assert_targeted(&result, "data.subset");
    // The placebo treatments centre on zero whatever the population.
    assert!(report(&result, "placebo.treatment").refuted_ate.abs() < 0.3);
}

#[test]
fn every_refuter_re_estimates_the_targeted_effect_under_bayesian_inference() {
    let result = refuted(true, RefuteSuite::Full);
    assert_targeted(&result, "random.common_cause");
    assert_targeted(&result, "data.subset");
    assert!(report(&result, "placebo.treatment").refuted_ate.abs() < 0.3);
}

#[test]
fn the_cheap_and_placebo_suites_run_for_a_target_too() {
    for bayes in [false, true] {
        for suite in [RefuteSuite::Cheap, RefuteSuite::PlaceboAndRcc] {
            let result = refuted(bayes, suite);
            assert!(!result.refutations.is_empty(), "{suite:?} under bayes={bayes}");
        }
    }
}
