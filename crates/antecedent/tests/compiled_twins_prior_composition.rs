//! The compiled external prior composition and transport families and the direct
//! `antecedent_prob` functions return the same composed prior, discounts, effective sample
//! sizes, transported sources, transport outcomes, target overlap and bandwidth sensitivity rows,
//! bit for bit, and refuse the same inputs the same way.
//!
//! The fixtures are those of the kernel's own tests and of the known-truth tests: one and two
//! Gaussian sources at fixed discounts, a moment-matched mixture, a correlated (dense) source, a
//! dropped zero-discount source, Beta and Gamma count sources, a tag-defined target with a unit
//! reweighting (default and declared coefficient, a residual-scaled source), claim-only and
//! forced-zero transports, a row-weighted and a kernel-weighted target, and Gaussian and
//! Epanechnikov bandwidth sensitivities. Neither path reads the context's seed. Each compiled
//! program names its route, and the route's unit is the capability record that owns external
//! prior composition and transport.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::float_cmp, reason = "the bits are compared exactly")]

use std::sync::Arc;

use antecedent::Study;
use antecedent::planner::PlanRoute;
use antecedent_core::ExecutionContext;
use antecedent_prob::{
    BandwidthSensitivityRow, BetaHyperparameters, CoefficientCorrelation, ComposedPrior,
    ExternalPriorSource, ExternalPriorWeight, GammaHyperparameters, GaussianCoefficientPrior,
    KernelFamily, PriorSet, PriorSpec, TargetOverlap, TransportAdjustment, TransportContext,
    TransportKernel, TransportOutcome, TransportPolicy, WeightedTarget, apply_transport,
    apply_weighted_target_transport, compose_beta_power_priors, compose_external_priors,
    compose_gamma_power_priors, compose_with_transport, compose_with_weighted_target_transport,
    kernel_bandwidth_sensitivity,
};

const UNIT: &str = "prior.external_composition";

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(23)
}

fn gaussian(mean: &[f64], variance: &[f64]) -> PriorSet {
    let mut prior = PriorSet::new();
    prior.push(PriorSpec::GaussianCoefficients(GaussianCoefficientPrior {
        mean: Arc::from(mean.to_vec()),
        variance: Arc::from(variance.to_vec()),
    }));
    prior
}

fn source(
    id: &str,
    mean: &[f64],
    variance: &[f64],
    weight: ExternalPriorWeight,
    ess: Option<f64>,
) -> ExternalPriorSource {
    ExternalPriorSource { id: Arc::from(id), prior: gaussian(mean, variance), weight, ess }
}

fn power(alpha: f64) -> ExternalPriorWeight {
    ExternalPriorWeight::power(alpha).unwrap()
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn option_bits(values: &[Option<f64>]) -> Vec<Option<u64>> {
    values.iter().map(|value| value.map(f64::to_bits)).collect()
}

fn same_prior(direct: &PriorSet, compiled: &PriorSet) {
    let (left, right) =
        (direct.gaussian_coefficients().unwrap(), compiled.gaussian_coefficients().unwrap());
    assert_eq!(bits(&left.mean), bits(&right.mean), "mean");
    assert_eq!(bits(&left.variance), bits(&right.variance), "variance");
    assert_eq!(
        direct.coefficient_correlation().map(|c| bits(c.matrix())),
        compiled.coefficient_correlation().map(|c| bits(c.matrix())),
        "correlation"
    );
    assert_eq!(direct, compiled);
}

fn same_composed(direct: &ComposedPrior, compiled: &ComposedPrior) {
    same_prior(&direct.prior, &compiled.prior);
    assert_eq!(direct.source_ids, compiled.source_ids);
    assert_eq!(bits(&direct.alphas_requested), bits(&compiled.alphas_requested));
    assert_eq!(bits(&direct.alphas_applied), bits(&compiled.alphas_applied));
    assert_eq!(option_bits(&direct.mixture_weights), option_bits(&compiled.mixture_weights));
    assert_eq!(option_bits(&direct.effective_ess), option_bits(&compiled.effective_ess));
    assert_eq!(direct.composed_ess.map(f64::to_bits), compiled.composed_ess.map(f64::to_bits));
    assert_eq!(direct.kish_ess.map(f64::to_bits), compiled.kish_ess.map(f64::to_bits));
    assert_eq!(direct, compiled);
}

fn same_sources(direct: &[ExternalPriorSource], compiled: &[ExternalPriorSource]) {
    assert_eq!(direct.len(), compiled.len());
    for (left, right) in direct.iter().zip(compiled) {
        assert_eq!(left.id, right.id);
        assert_eq!(left.weight.alpha.to_bits(), right.weight.alpha.to_bits());
        assert_eq!(
            left.weight.mixture_weight.map(f64::to_bits),
            right.weight.mixture_weight.map(f64::to_bits)
        );
        assert_eq!(left.ess.map(f64::to_bits), right.ess.map(f64::to_bits));
        same_prior(&left.prior, &right.prior);
    }
}

fn same_outcomes(direct: &[TransportOutcome], compiled: &[TransportOutcome]) {
    assert_eq!(direct.len(), compiled.len());
    for (left, right) in direct.iter().zip(compiled) {
        assert_eq!(left.alpha_override.map(f64::to_bits), right.alpha_override.map(f64::to_bits));
        assert_eq!(left, right);
    }
}

fn same_overlap(direct: &TargetOverlap, compiled: &TargetOverlap) {
    assert_eq!(direct.n_rows, compiled.n_rows);
    assert_eq!(direct.n_support, compiled.n_support);
    assert_eq!(direct.support_fraction.to_bits(), compiled.support_fraction.to_bits());
    assert_eq!(direct.kish_ess.to_bits(), compiled.kish_ess.to_bits());
    assert_eq!(direct.ess_fraction.to_bits(), compiled.ess_fraction.to_bits());
    assert_eq!(direct.max_weight_share.to_bits(), compiled.max_weight_share.to_bits());
}

fn same_rows(direct: &[BandwidthSensitivityRow], compiled: &[BandwidthSensitivityRow]) {
    assert_eq!(direct.len(), compiled.len());
    for (left, right) in direct.iter().zip(compiled) {
        assert_eq!(left.bandwidth.to_bits(), right.bandwidth.to_bits());
        assert_eq!(left.weighted_mean.map(f64::to_bits), right.weighted_mean.map(f64::to_bits));
        assert_eq!(left.mean_variance.map(f64::to_bits), right.mean_variance.map(f64::to_bits));
        match (&left.overlap, &right.overlap) {
            (Some(l), Some(r)) => same_overlap(l, r),
            (None, None) => {}
            _ => panic!("overlap present on one path only"),
        }
        assert_eq!(left.overlap_ok, right.overlap_ok);
        assert_eq!(left.refusal, right.refusal);
    }
}

fn dense_source(alpha: f64, mixture: Option<f64>) -> ExternalPriorSource {
    let mut prior = gaussian(&[1.0, -1.0], &[0.04, 0.09]);
    prior.push(PriorSpec::CoefficientCorrelation(
        CoefficientCorrelation::new(2, vec![1.0, -0.8, -0.8, 1.0]).unwrap(),
    ));
    let weight = match mixture {
        Some(w) => ExternalPriorWeight::power_mixture(alpha, w).unwrap(),
        None => power(alpha),
    };
    ExternalPriorSource { id: Arc::from("dense"), prior, weight, ess: None }
}

fn gaussian_cases() -> Vec<(Vec<ExternalPriorSource>, PriorSet)> {
    vec![
        (vec![source("a", &[2.0], &[1.0], power(0.5), Some(40.0))], gaussian(&[0.0], &[4.0])),
        (
            vec![
                source("a", &[2.0, -1.0], &[1.0, 0.5], power(0.5), Some(40.0)),
                source("b", &[3.0, 0.0], &[2.0, 1.0], power(0.25), Some(80.0)),
            ],
            gaussian(&[0.0, 1.0], &[4.0, 2.0]),
        ),
        (
            vec![source(
                "m",
                &[10.0],
                &[1.0],
                ExternalPriorWeight::power_mixture(0.5, 0.4).unwrap(),
                Some(50.0),
            )],
            gaussian(&[0.0], &[100.0]),
        ),
        (vec![dense_source(0.5, None)], PriorSet::weakly_informative(2)),
        (vec![dense_source(1.0, Some(0.5))], gaussian(&[0.0, 0.0], &[1.0, 1.0])),
        (
            vec![
                source("a", &[2.0], &[1.0], power(0.5), Some(40.0)),
                source("b", &[5.0], &[1.0], power(0.0), Some(999.0)),
            ],
            gaussian(&[0.0], &[4.0]),
        ),
    ]
}

#[test]
fn compiled_gaussian_composition_equals_the_direct_composition_bit_for_bit() {
    let ctx = ctx();
    for (sources, baseline) in gaussian_cases() {
        let direct = compose_external_priors(&sources, &baseline).unwrap();
        let compiled =
            Study::compile_external_prior_composition(sources.clone(), baseline.clone(), &ctx)
                .unwrap()
                .compose(&ctx)
                .unwrap();
        same_composed(&direct, &compiled);
    }
    // A refused composition is refused the same way.
    let mixed_modes = vec![
        source("a", &[1.0], &[1.0], power(1.0), None),
        source("b", &[2.0], &[1.0], ExternalPriorWeight::power_mixture(1.0, 0.5).unwrap(), None),
    ];
    let baseline = gaussian(&[0.0], &[4.0]);
    let direct = compose_external_priors(&mixed_modes, &baseline).unwrap_err();
    let compiled = Study::compile_external_prior_composition(mixed_modes, baseline, &ctx)
        .unwrap()
        .compose(&ctx)
        .unwrap_err();
    assert_eq!(direct, compiled);
}

#[test]
fn compiled_count_compositions_equal_the_direct_compositions_bit_for_bit() {
    let ctx = ctx();
    let beta_cases: Vec<(BetaHyperparameters, Vec<(u64, u64, f64)>)> = vec![
        (BetaHyperparameters { alpha: 1.0, beta: 1.0 }, vec![(30, 100, 0.5), (12, 40, 0.25)]),
        (BetaHyperparameters { alpha: 2.5, beta: 0.5 }, vec![(7, 9, 0.0)]),
        (BetaHyperparameters { alpha: 0.3, beta: 0.7 }, vec![(1, 3, 1.0 / 3.0), (0, 5, 0.9)]),
        (BetaHyperparameters { alpha: 1.0, beta: 1.0 }, vec![(11, 10, 0.5)]),
        (BetaHyperparameters { alpha: 1.0, beta: 1.0 }, vec![(3, 10, 1.5)]),
    ];
    for (base, sources) in beta_cases {
        let direct = compose_beta_power_priors(base, &sources);
        let compiled =
            Study::compile_beta_power_prior_composition(base, sources, &ctx).unwrap().compose(&ctx);
        match (direct, compiled) {
            (Ok(left), Ok(right)) => {
                assert_eq!(left.alpha.to_bits(), right.alpha.to_bits());
                assert_eq!(left.beta.to_bits(), right.beta.to_bits());
            }
            (Err(left), Err(right)) => assert_eq!(left, right),
            (left, right) => panic!("paths disagree: {left:?} vs {right:?}"),
        }
    }
    let gamma_cases: Vec<(GammaHyperparameters, Vec<(u64, f64, f64)>)> = vec![
        (GammaHyperparameters { shape: 2.0, rate: 1.0 }, vec![(50, 25.0, 0.4), (10, 8.0, 1.0)]),
        (GammaHyperparameters { shape: 3.0, rate: 2.0 }, vec![]),
        (GammaHyperparameters { shape: 0.5, rate: 0.1 }, vec![(7, 0.3, 0.7)]),
        (GammaHyperparameters { shape: 1.0, rate: 1.0 }, vec![(3, -1.0, 0.5)]),
    ];
    for (base, sources) in gamma_cases {
        let direct = compose_gamma_power_priors(base, &sources);
        let compiled = Study::compile_gamma_power_prior_composition(base, sources, &ctx)
            .unwrap()
            .compose(&ctx);
        match (direct, compiled) {
            (Ok(left), Ok(right)) => {
                assert_eq!(left.shape.to_bits(), right.shape.to_bits());
                assert_eq!(left.rate.to_bits(), right.rate.to_bits());
            }
            (Err(left), Err(right)) => assert_eq!(left, right),
            (left, right) => panic!("paths disagree: {left:?} vs {right:?}"),
        }
    }
}

#[test]
fn compiled_tag_transport_equals_the_direct_transport_bit_for_bit() {
    let ctx = ctx();
    let reweight =
        TransportAdjustment::new((0..25).map(f64::from).collect::<Vec<_>>(), vec![1.0; 25])
            .unwrap();
    let graded = TransportAdjustment::new(
        (0..60).map(|i| f64::from(i % 7) + 0.25 * f64::from(i)).collect::<Vec<_>>(),
        (0..60).map(|i| 1.0 + f64::from(i % 5)).collect::<Vec<_>>(),
    )
    .unwrap();
    let thin = TransportAdjustment::new(vec![1.0; 15], vec![1.0; 15]).unwrap();
    let two = vec![
        source("s", &[0.3, 1.0], &[0.7, 0.4], power(0.5), None),
        source("t", &[-0.2, 2.0], &[0.5, 0.9], power(0.75), Some(12.0)),
    ];
    let mut scaled = two.clone();
    for item in &mut scaled {
        item.prior.push(PriorSpec::KnownResidualVariance(2.0));
    }
    let baseline = gaussian(&[0.0, 0.0], &[100.0, 100.0]);
    let tagged = [Some("source"), Some("target")];
    let untagged = [Some("us"), None];
    let contexts = [
        TransportContext {
            source_populations: &tagged,
            target_population: Some("target"),
            policy: Some(TransportPolicy::InvariantConditionalOutcome),
            adjustment: Some(&reweight),
            coef_index: None,
        },
        TransportContext {
            source_populations: &untagged,
            target_population: Some("eu"),
            policy: Some(TransportPolicy::InvariantEffectModifiers),
            adjustment: Some(&graded),
            coef_index: Some(0),
        },
        TransportContext {
            source_populations: &untagged,
            target_population: Some("eu"),
            policy: Some(TransportPolicy::InvariantPropensity),
            adjustment: None,
            coef_index: None,
        },
        TransportContext {
            source_populations: &untagged,
            target_population: Some("eu"),
            policy: Some(TransportPolicy::InvariantEffectModifiers),
            adjustment: None,
            coef_index: None,
        },
        TransportContext {
            source_populations: &tagged,
            target_population: Some("target"),
            policy: Some(TransportPolicy::InvariantConditionalOutcome),
            adjustment: Some(&thin),
            coef_index: None,
        },
        TransportContext {
            source_populations: &untagged,
            target_population: Some("eu"),
            policy: None,
            adjustment: None,
            coef_index: None,
        },
        TransportContext {
            source_populations: &tagged,
            target_population: Some("target"),
            policy: Some(TransportPolicy::InvariantConditionalOutcome),
            adjustment: Some(&reweight),
            coef_index: Some(5),
        },
    ];
    for sources in [&two, &scaled] {
        for context in &contexts {
            let compiled = Study::compile_prior_tag_transport(
                sources.clone(),
                baseline.clone(),
                context,
                &ctx,
            )
            .unwrap();
            match (apply_transport(sources, context), compiled.apply_transport(&ctx)) {
                (Ok((left, left_outcomes)), Ok((right, right_outcomes))) => {
                    same_sources(&left, &right);
                    same_outcomes(&left_outcomes, &right_outcomes);
                }
                (Err(left), Err(right)) => assert_eq!(left, right),
                (left, right) => panic!("paths disagree: {left:?} vs {right:?}"),
            }
            match (
                compose_with_transport(sources, &baseline, context),
                compiled.compose_with_transport(&ctx),
            ) {
                (Ok((left, left_outcomes)), Ok((right, right_outcomes))) => {
                    same_composed(&left, &right);
                    same_outcomes(&left_outcomes, &right_outcomes);
                }
                (Err(left), Err(right)) => assert_eq!(left, right),
                (left, right) => panic!("paths disagree: {left:?} vs {right:?}"),
            }
        }
    }
}

fn weighted_targets() -> Vec<WeightedTarget> {
    let effects: Vec<f64> = (0..60).map(|i| 0.5 * f64::from(i % 6)).collect();
    let weights: Vec<f64> = (0..60).map(|i| if i < 30 { 2.0 } else { 1.0 }).collect();
    let row = WeightedTarget::from_row_weights(effects, weights).unwrap();
    let xs: Vec<f64> = (0..400).map(|i| -2.0 + 0.01 * f64::from(i)).collect();
    let distances: Vec<f64> = xs.iter().map(|x| x.abs()).collect();
    let kernel = WeightedTarget::from_kernel(
        xs.clone(),
        &distances,
        TransportKernel::new(KernelFamily::Tricube, 0.5).unwrap(),
    )
    .unwrap();
    let narrow = WeightedTarget::from_kernel(
        xs,
        &distances,
        TransportKernel::new(KernelFamily::Gaussian, 0.004).unwrap(),
    )
    .unwrap()
    .with_min_ess_fraction(0.0)
    .unwrap();
    vec![row, kernel, narrow]
}

#[test]
fn compiled_weighted_target_transport_equals_the_direct_transport_bit_for_bit() {
    let ctx = ctx();
    let mut residual = source("r", &[4.0], &[0.25], power(0.5), None);
    residual.prior.push(PriorSpec::KnownResidualVariance(2.0));
    let source_sets = [
        vec![residual],
        vec![
            source("a", &[0.0, 1.0], &[1.0, 0.5], power(1.0), None),
            source("b", &[2.0, -1.0], &[0.3, 0.2], power(0.4), Some(30.0)),
        ],
    ];
    for target in weighted_targets() {
        for sources in &source_sets {
            let width = sources[0].prior.gaussian_coefficients().unwrap().len();
            let baseline = gaussian(&vec![0.0; width], &vec![10.0; width]);
            for (policy, coef_index) in [
                (TransportPolicy::InvariantConditionalOutcome, None),
                (TransportPolicy::InvariantEffectModifiers, Some(0)),
                (TransportPolicy::InvariantPropensity, Some(7)),
            ] {
                let compiled = Study::compile_prior_weighted_target_transport(
                    sources.clone(),
                    baseline.clone(),
                    policy,
                    target.clone(),
                    coef_index,
                    &ctx,
                )
                .unwrap();
                assert_eq!(compiled.target(), &target);
                match (
                    apply_weighted_target_transport(sources, policy, &target, coef_index),
                    compiled.apply_weighted_target_transport(&ctx),
                ) {
                    (Ok((left, left_overlap)), Ok((right, right_overlap))) => {
                        same_sources(&left, &right);
                        same_overlap(&left_overlap, &right_overlap);
                    }
                    (Err(left), Err(right)) => assert_eq!(left, right),
                    (left, right) => panic!("paths disagree: {left:?} vs {right:?}"),
                }
                match (
                    compose_with_weighted_target_transport(
                        sources, &baseline, policy, &target, coef_index,
                    ),
                    compiled.compose_with_weighted_target_transport(&ctx),
                ) {
                    (Ok((left, left_overlap)), Ok((right, right_overlap))) => {
                        same_composed(&left, &right);
                        same_overlap(&left_overlap, &right_overlap);
                    }
                    (Err(left), Err(right)) => assert_eq!(left, right),
                    (left, right) => panic!("paths disagree: {left:?} vs {right:?}"),
                }
            }
        }
    }
}

#[test]
fn compiled_bandwidth_sensitivity_equals_the_direct_rows_bit_for_bit() {
    let ctx = ctx();
    let xs: Vec<f64> = (0..400).map(|i| -2.0 + 0.01 * f64::from(i)).collect();
    let distances: Vec<f64> = xs.iter().map(|x| x.abs()).collect();
    let far: Vec<f64> = (0..400).map(|i| 3.0 + 0.01 * f64::from(i)).collect();
    let cases: Vec<(KernelFamily, Vec<f64>, Vec<f64>, Vec<f64>, f64)> = vec![
        (KernelFamily::Gaussian, vec![0.05, 0.5, 2.0], distances.clone(), xs.clone(), 0.05),
        (
            KernelFamily::Epanechnikov,
            vec![0.02, 0.1, 0.5, 1.0],
            distances.clone(),
            xs.clone(),
            0.05,
        ),
        (KernelFamily::Tricube, vec![0.3, 1.5], distances, xs.clone(), 0.2),
        (KernelFamily::Epanechnikov, vec![0.5], far.clone(), xs.clone(), 0.05),
        (KernelFamily::Gaussian, vec![], far.clone(), xs.clone(), 0.05),
        (KernelFamily::Gaussian, vec![1.0], far, xs[..10].to_vec(), 0.05),
    ];
    for (family, bandwidths, distances, effects, floor) in cases {
        let direct = kernel_bandwidth_sensitivity(family, &bandwidths, &distances, &effects, floor);
        let compiled = Study::compile_kernel_bandwidth_sensitivity(
            family, bandwidths, distances, effects, floor, &ctx,
        )
        .unwrap()
        .sensitivity(&ctx);
        match (direct, compiled) {
            (Ok(left), Ok(right)) => same_rows(&left, &right),
            (Err(left), Err(right)) => assert_eq!(left, right),
            (left, right) => panic!("paths disagree: {left:?} vs {right:?}"),
        }
    }
}

#[test]
fn each_compiled_program_names_its_route_and_reports_its_unit() {
    let ctx = ctx();
    let (sources, baseline) = gaussian_cases().remove(0);
    let gaussian_program =
        Study::compile_external_prior_composition(sources.clone(), baseline.clone(), &ctx).unwrap();
    let beta = Study::compile_beta_power_prior_composition(
        BetaHyperparameters { alpha: 1.0, beta: 1.0 },
        vec![(30, 100, 0.5)],
        &ctx,
    )
    .unwrap();
    let gamma = Study::compile_gamma_power_prior_composition(
        GammaHyperparameters { shape: 2.0, rate: 1.0 },
        vec![(50, 25.0, 0.4)],
        &ctx,
    )
    .unwrap();
    let populations = [Some("us")];
    let context = TransportContext {
        source_populations: &populations,
        target_population: Some("eu"),
        policy: Some(TransportPolicy::InvariantEffectModifiers),
        adjustment: None,
        coef_index: None,
    };
    let tag = Study::compile_prior_tag_transport(sources.clone(), baseline.clone(), &context, &ctx)
        .unwrap();
    let weighted = Study::compile_prior_weighted_target_transport(
        sources,
        baseline,
        TransportPolicy::InvariantConditionalOutcome,
        weighted_targets().remove(0),
        None,
        &ctx,
    )
    .unwrap();
    let sensitivity = Study::compile_kernel_bandwidth_sensitivity(
        KernelFamily::Gaussian,
        vec![0.5, 2.0],
        (0..50).map(|i| 0.1 * f64::from(i)).collect(),
        (0..50).map(|i| 1.0 + 0.2 * f64::from(i)).collect(),
        0.05,
        &ctx,
    )
    .unwrap();
    for (view, route) in [
        (gaussian_program.plan_view(), PlanRoute::ExternalPriorComposition),
        (beta.plan_view(), PlanRoute::ExternalPriorComposition),
        (gamma.plan_view(), PlanRoute::ExternalPriorComposition),
        (tag.plan_view(), PlanRoute::ExternalPriorTransport),
        (weighted.plan_view(), PlanRoute::ExternalPriorTransport),
        (sensitivity.plan_view(), PlanRoute::ExternalPriorTransport),
    ] {
        assert_eq!(view.route(), route);
        let decision = view.unit_support().expect("the route is bound to a unit");
        // Whether the unit is supported is the record's earned status, read from the support
        // model; the program only has to name the unit.
        assert_eq!(decision.unit(), UNIT);
    }
    // Equal declarations compile to the same program identity; a different discount does not.
    let again = Study::compile_beta_power_prior_composition(
        BetaHyperparameters { alpha: 1.0, beta: 1.0 },
        vec![(30, 100, 0.5)],
        &ctx,
    )
    .unwrap();
    let other = Study::compile_beta_power_prior_composition(
        BetaHyperparameters { alpha: 1.0, beta: 1.0 },
        vec![(30, 100, 0.25)],
        &ctx,
    )
    .unwrap();
    assert_eq!(beta.lowered().identity(), again.lowered().identity());
    assert_ne!(beta.lowered().identity(), other.lowered().identity());
    // A run consults the unit: on a supported unit it passes and returns the result.
    assert!(gaussian_program.compose(&ctx).is_ok());
    assert!(beta.compose(&ctx).is_ok());
    assert!(gamma.compose(&ctx).is_ok());
    assert!(tag.apply_transport(&ctx).is_ok());
    assert!(tag.compose_with_transport(&ctx).is_ok());
    assert!(weighted.apply_weighted_target_transport(&ctx).is_ok());
    assert!(weighted.compose_with_weighted_target_transport(&ctx).is_ok());
    assert!(sensitivity.sensitivity(&ctx).is_ok());
}
