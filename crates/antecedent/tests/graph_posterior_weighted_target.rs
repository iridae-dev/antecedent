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
//! SPDX-License-Identifier: MIT OR Apache-2.0

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
