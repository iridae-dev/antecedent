//! Known-truth tests of a weighted target population through the study API.
//!
//! The planted effect is heterogeneous, `tau(z) = 1 + z`, so the effect for a population is the
//! weighted mean of `tau` over that population's row weights. Every expected value is that
//! closed form on the planted rows (and the hand-computed Kish quantities of the weights), never
//! a call into the estimator under test. The target is declared on the study as a kernel over
//! each row's distance to the target centre, the way a Python caller declares it.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    reason = "test scaffolding compares identical computations and indexes with small literals"
)]

use antecedent::inference::{KernelFamily, TargetOverlap, TransportKernel};
use antecedent::{
    EstimatorId, PreparedStudy, RefuteSuite, Study, StudyResult, WeightedTargetDeclaration,
};
use antecedent_core::{
    AverageEffectQuery, DistributionRef, ExecutionContext, PopulationRegistry, StreamDomain,
    TargetPopulation, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_kernels::standard_normal;

const Z: u32 = 2;

struct Fixture {
    data: TabularData,
    graph: Dag,
    query: AverageEffectQuery,
    z: Vec<f64>,
}

fn fixture(n: usize, seed: u64) -> Fixture {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Test, 0x31);
    let mut z = vec![0.0; n];
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        let zi = standard_normal(&mut rng);
        z[i] = zi;
        let p = 1.0 / (1.0 + (-(0.9 * zi)).exp());
        t[i] = f64::from(rng.next_f64() < p);
        y[i] = (1.0 + zi) * t[i] + zi + 0.3 * standard_normal(&mut rng);
    }
    let pairs = [("t", t), ("y", y), ("z", z.clone())];
    let borrowed: Vec<(&str, &[f64])> = pairs.iter().map(|(n, v)| (*n, v.as_slice())).collect();
    let data = TabularData::from_f64_columns(borrowed).unwrap();
    let mut graph = Dag::with_variables(3);
    for (from, to) in [(2, 0), (2, 1), (0, 1)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    Fixture { data, graph, query, z }
}

fn kernel(family: KernelFamily, bandwidth: f64) -> TransportKernel {
    TransportKernel::new(family, bandwidth).unwrap()
}

/// The declared kernel target of `z` around `centre`.
fn declaration(family: KernelFamily, bandwidth: f64, centre: f64) -> WeightedTargetDeclaration {
    WeightedTargetDeclaration::kernel(
        kernel(family, bandwidth),
        vec![VariableId::from_raw(Z)],
        vec![centre],
        None,
    )
    .unwrap()
}

/// The kernel weights of `z`, computed here by hand from the rows' distances to `centre`.
fn kernel_weights(z: &[f64], family: KernelFamily, bandwidth: f64, centre: f64) -> Vec<f64> {
    let distances: Vec<f64> = z.iter().map(|zi| (zi - centre).abs()).collect();
    kernel(family, bandwidth).weights(&distances).unwrap()
}

/// Closed-form effect of the weighted target: the weighted mean of `tau(z) = 1 + z`.
fn closed_form(z: &[f64], weights: &[f64]) -> f64 {
    let mass: f64 = weights.iter().sum();
    weights.iter().zip(z).map(|(w, zi)| w * (1.0 + zi)).sum::<f64>() / mass
}

fn builder(fx: &Fixture) -> antecedent::StudyBuilder {
    Study::tabular(fx.data.clone())
        .graph(fx.graph.clone())
        .estimator(EstimatorId::Aipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
}

fn run(
    fx: &Fixture,
    target: Option<WeightedTargetDeclaration>,
    seed: u64,
) -> Result<StudyResult, antecedent::CausalError> {
    let mut builder = builder(fx);
    if let Some(target) = target {
        builder = builder.weighted_target(target);
    }
    builder.query(fx.query.clone()).build()?.run(&ExecutionContext::for_tests(seed))
}

fn ate_and_se(result: &StudyResult) -> (f64, f64) {
    let effect = result.estimate().as_effect().unwrap();
    (effect.ate, effect.se_analytic())
}

#[test]
fn a_kernel_neighbourhood_effect_equals_the_closed_form_weighted_effect() {
    let fx = fixture(3_000, 21);
    let weights = kernel_weights(&fx.z, KernelFamily::Gaussian, 0.5, 1.0);
    let truth = closed_form(&fx.z, &weights);
    let result = run(&fx, Some(declaration(KernelFamily::Gaussian, 0.5, 1.0)), 21).unwrap();
    let (ate, se) = ate_and_se(&result);
    assert!((ate - truth).abs() < 4.0 * se, "ate {ate} se {se} truth {truth}");
    // The whole-population effect is another number: the target moved the answer.
    let everyone = run(&fx, None, 21).unwrap();
    assert!((ate_and_se(&everyone).0 - ate).abs() > 0.5);
}

#[test]
fn the_effect_tracks_the_closed_form_across_bandwidths_and_families() {
    let fx = fixture(4_000, 22);
    for (family, bandwidth) in [
        (KernelFamily::Gaussian, 0.4),
        (KernelFamily::Gaussian, 1.2),
        (KernelFamily::Epanechnikov, 1.5),
        (KernelFamily::Tricube, 2.0),
    ] {
        let weights = kernel_weights(&fx.z, family, bandwidth, 0.5);
        let truth = closed_form(&fx.z, &weights);
        let result = run(&fx, Some(declaration(family, bandwidth, 0.5)), 22).unwrap();
        let (ate, se) = ate_and_se(&result);
        assert!((ate - truth).abs() < 4.0 * se, "{family:?} h={bandwidth}: {ate} vs {truth}");
    }
}

#[test]
fn a_declared_target_is_the_custom_distribution_of_the_same_weights() {
    let fx = fixture(2_500, 26);
    let declared = declaration(KernelFamily::Gaussian, 0.5, 1.0);
    let weights = kernel_weights(&fx.z, KernelFamily::Gaussian, 0.5, 1.0);
    let by_declaration = run(&fx, Some(declared), 26).unwrap();
    // The existing mechanism, registered by hand under another id.
    let href = DistributionRef::from_raw(9);
    let mut registry = PopulationRegistry::new();
    registry.insert_distribution_with_dependence(href, weights, [VariableId::from_raw(Z)]);
    let by_registry = builder(&fx)
        .population_registry(registry)
        .query(fx.query.clone().with_target_population(TargetPopulation::CustomDistribution(href)))
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(26))
        .unwrap();
    assert_eq!(ate_and_se(&by_declaration), ate_and_se(&by_registry));
}

#[test]
fn the_report_holds_the_hand_computed_kish_quantities_and_the_declaration() {
    let fx = fixture(2_000, 23);
    let weights = kernel_weights(&fx.z, KernelFamily::Gaussian, 0.5, 1.0);
    let prepared: PreparedStudy = builder(&fx)
        .weighted_target(declaration(KernelFamily::Gaussian, 0.5, 1.0))
        .query(fx.query.clone())
        .build()
        .unwrap()
        .prepare(&ExecutionContext::for_tests(23))
        .unwrap();
    let report = prepared.weighted_target_report().unwrap();
    let mass: f64 = weights.iter().sum();
    let squares: f64 = weights.iter().map(|w| w * w).sum();
    let max = weights.iter().copied().fold(0.0_f64, f64::max);
    let overlap = report.overlap();
    assert_eq!(overlap.n_rows, weights.len());
    assert_eq!(overlap.n_support, weights.iter().filter(|w| **w > 0.0).count());
    assert!((overlap.kish_ess - mass * mass / squares).abs() < 1e-9 * overlap.kish_ess);
    assert!((overlap.max_weight_share - max / mass).abs() < 1e-15);
    assert_eq!(report.family(), Some(KernelFamily::Gaussian));
    assert_eq!(report.center(), Some(&[1.0][..]));
    assert_eq!(report.scale(), Some(&[1.0][..]));
    // The kernel is part of the identity: another bandwidth is another claim.
    let other = builder(&fx)
        .weighted_target(declaration(KernelFamily::Gaussian, 0.8, 1.0))
        .query(fx.query.clone())
        .build()
        .unwrap()
        .prepare(&ExecutionContext::for_tests(23))
        .unwrap();
    assert_ne!(report.identity(), other.weighted_target_report().unwrap().identity());
}

#[test]
fn a_target_without_overlap_with_the_rows_refuses() {
    let fx = fixture(2_000, 24);
    let weights = kernel_weights(&fx.z, KernelFamily::Gaussian, 0.3, 6.0);
    assert!(TargetOverlap::from_weights(&weights).failure(0.05).is_some());
    let error = run(&fx, Some(declaration(KernelFamily::Gaussian, 0.3, 6.0)), 24).unwrap_err();
    let text = error.to_string();
    assert!(text.contains("transport_target_overlap_failed"), "{text}");
    assert!(text.contains("Next attempt"), "{text}");
}

#[test]
fn a_declared_ess_fraction_floor_is_applied() {
    let fx = fixture(2_000, 27);
    let strict = declaration(KernelFamily::Gaussian, 0.5, 1.0).with_min_ess_fraction(0.9).unwrap();
    let error = run(&fx, Some(strict), 27).unwrap_err();
    assert!(error.to_string().contains("transport_target_overlap_failed"), "{error}");
}

#[test]
fn tag_populations_are_unchanged_by_the_weighted_machinery() {
    let fx = fixture(2_000, 25);
    let default = run(&fx, None, 25).unwrap();
    let explicit = builder(&fx)
        .query(fx.query.clone().with_target_population(TargetPopulation::AllObserved))
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(25))
        .unwrap();
    assert_eq!(ate_and_se(&default), ate_and_se(&explicit));
    assert!(default.retarget_population().is_none());
}
