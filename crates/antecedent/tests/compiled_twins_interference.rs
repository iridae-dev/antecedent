//! The direct interference estimators and their compiled routes are one computation.
//!
//! Each direct entry (`estimate_interference_with_context`, `estimate_saturation_interference`,
//! `estimate_observational_exposure`) is run on the fixtures the route tests use and compared,
//! bit for bit, with the point estimates, variances, standard errors and interval bounds the
//! prepared study publishes. The compiled route calls the same kernel with the same inputs.
//! The only difference is the exposure-probability Monte Carlo seed of the Bernoulli design: the
//! direct entry takes the caller's seed and the route derives it from the execution context, so
//! the direct call here is given the derived seed.
#![allow(
    deprecated,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    reason = "integration test builds fixtures from small nonnegative counts and calls the direct entries it compares"
)]
// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(deprecated, reason = "exercises the deprecated direct facade path")]

use std::sync::Arc;

use antecedent::estimate::{
    ObservationalExposureSpec, estimate_interference_with_context, estimate_observational_exposure,
    estimate_saturation_interference,
};
use antecedent::support::CellStatus;
use antecedent::{InterferenceSpec, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    AssignmentDesign, CausalQuery, ExecutionContext, ExposureLevel, ExposureMapping,
    ExposurePropensityProvenance, InterferenceFunctional, InterferenceQuery, StreamDomain,
    VariableId,
};
use antecedent_data::{NetworkData, NetworkEdge, TabularData};
use antecedent_graph::Dag;

fn same(direct: f64, compiled: f64, label: &str) {
    assert_eq!(direct.to_bits(), compiled.to_bits(), "{label}: direct {direct} vs {compiled}");
}

fn contrast(from: (f64, f64), to: (f64, f64)) -> InterferenceFunctional {
    InterferenceFunctional::ExposureContrast {
        outcome: VariableId::from_raw(0),
        from: ExposureLevel { own: from.0, neighbors: from.1 },
        to: ExposureLevel { own: to.0, neighbors: to.1 },
    }
}

fn compiled(
    data: &TabularData,
    query: &InterferenceQuery,
    edges: Vec<NetworkEdge>,
    assignment: &[bool],
    ctx: &ExecutionContext,
) -> StudyResult {
    let network = NetworkData::try_new(data.clone(), edges).unwrap();
    Study::tabular(data.clone())
        .graph(Dag::with_variables(1))
        .query(CausalQuery::Interference(query.clone()))
        .interference(InterferenceSpec { network, assignment: Arc::from(assignment) })
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .prepare(ctx)
        .unwrap()
        .estimate(data, ctx)
        .unwrap()
}

fn clique_edges(clusters: usize, size: usize) -> Vec<NetworkEdge> {
    (0..clusters)
        .flat_map(|cluster| {
            let first = cluster * size;
            (first..first + size).flat_map(move |from| {
                (first..first + size).filter(move |&to| to != from).map(move |to| {
                    NetworkEdge::new(
                        u64::try_from(from).expect("row id"),
                        u64::try_from(to).expect("row id"),
                        1.0,
                    )
                })
            })
        })
        .collect()
}

#[test]
fn design_based_route_equals_the_direct_estimator() {
    let n = 12_usize;
    let assignment =
        [false, false, false, true, true, false, false, false, false, true, true, false];
    let edges = (0..n)
        .flat_map(|unit| {
            let next = (unit + 1) % n;
            [
                NetworkEdge::new(unit as u64, next as u64, 1.0),
                NetworkEdge::new(next as u64, unit as u64, 1.0),
            ]
        })
        .collect::<Vec<_>>();
    let outcomes = (0..n)
        .map(|unit| {
            let neighbors = [(unit + n - 1) % n, (unit + 1) % n]
                .iter()
                .filter(|&&other| assignment[other])
                .count() as f64;
            1.0 + 0.25 * unit as f64 + 2.0 * f64::from(assignment[unit]) + 1.5 * neighbors
        })
        .collect::<Vec<_>>();
    let data = TabularData::from_f64_columns([("y", outcomes.as_slice())]).unwrap();
    let designs = [
        (
            "bernoulli",
            AssignmentDesign::Bernoulli { probabilities: Arc::from(vec![0.5; n]) },
            ExposureMapping::NeighborCount,
            (1.0, 1.0),
        ),
        (
            "complete",
            AssignmentDesign::CompleteRandomization { treated: 4 },
            ExposureMapping::NeighborFraction,
            (1.0, 0.5),
        ),
    ];
    for (label, design, exposure, to) in designs {
        let query = InterferenceQuery::new(design, exposure, contrast((0.0, 0.0), to));
        let ctx = ExecutionContext::for_tests(241);
        let result = compiled(&data, &query, edges.clone(), &assignment, &ctx);
        // The route draws the exposure-probability seed from the execution context's stream.
        let seed = ctx.rng.stream_for(StreamDomain::Transport, 0x1F7E).next_u64();
        let network = NetworkData::try_new(data.clone(), edges.clone()).unwrap();
        let direct =
            estimate_interference_with_context(&query, &network, &assignment, seed, &ctx).unwrap();
        let route = result.interference().unwrap();
        same(
            direct.contrast.horvitz_thompson,
            route.contrast.horvitz_thompson,
            &format!("{label} ht"),
        );
        same(direct.contrast.hajek, route.contrast.hajek, &format!("{label} hajek"));
        same(
            direct.contrast.conservative_variance,
            route.contrast.conservative_variance,
            &format!("{label} variance"),
        );
        same(
            direct.minimum_exposure_probability,
            route.minimum_exposure_probability,
            &format!("{label} min prob"),
        );
        assert_eq!(direct.from_probability_method, route.from_probability_method);
        assert_eq!(direct.to_probability_method, route.to_probability_method);
        let effect = result.estimate().as_effect().unwrap();
        same(direct.contrast.horvitz_thompson, effect.ate, &format!("{label} effect"));
        same(
            direct.contrast.conservative_variance.sqrt(),
            effect.se_analytic(),
            &format!("{label} se"),
        );
    }
}

fn saturation_fixture(
    clusters: usize,
    high: usize,
    from: (f64, f64),
    to: (f64, f64),
) -> (TabularData, Vec<bool>, InterferenceQuery, Vec<NetworkEdge>) {
    let labels = (0..clusters).flat_map(|cluster| [cluster as u32; 3]).collect::<Vec<u32>>();
    let patterns =
        [[false, false, false], [true, false, false], [true, true, false], [true, true, true]];
    let assignment = (0..clusters).flat_map(|cluster| patterns[cluster % 4]).collect::<Vec<_>>();
    let realized = (0..clusters)
        .flat_map(|cluster| [if cluster < clusters - high { 0.2 } else { 0.8 }; 3])
        .collect::<Vec<_>>();
    let outcomes = (0..assignment.len())
        .map(|unit| {
            let first = unit / 3 * 3;
            let fraction = (first..first + 3)
                .filter(|&other| other != unit && assignment[other])
                .count() as f64
                / 2.0;
            5.0 + 0.1 * (unit / 3 % 9) as f64 + 2.0 * f64::from(assignment[unit]) + 3.0 * fraction
        })
        .collect::<Vec<_>>();
    let data = TabularData::from_f64_columns([("y", outcomes.as_slice())]).unwrap();
    let query = InterferenceQuery::new(
        AssignmentDesign::TwoStageSaturation {
            clusters: Arc::from(labels),
            low_probability: 0.2,
            high_probability: 0.8,
            high_clusters: high,
            realized_saturation: Arc::from(realized),
        },
        ExposureMapping::NeighborFraction,
        contrast(from, to),
    );
    (data, assignment, query, clique_edges(clusters, 3))
}

#[test]
fn saturation_route_equals_the_direct_estimator_with_and_without_an_interval() {
    let ctx = ExecutionContext::for_tests(241);
    // Four clusters withhold the interval; forty-eight carry the supported one.
    for (clusters, high, expect_interval) in [(4_usize, 2_usize, false), (48, 24, true)] {
        for (name, from, to) in [
            ("direct", (0.0, 0.5), (1.0, 0.5)),
            ("spillover", (0.0, 0.0), (0.0, 1.0)),
            ("total", (0.0, 0.0), (1.0, 1.0)),
        ] {
            let label = format!("{clusters} clusters {name}");
            let (data, assignment, query, edges) = saturation_fixture(clusters, high, from, to);
            let result = compiled(&data, &query, edges.clone(), &assignment, &ctx);
            let network = NetworkData::try_new(data.clone(), edges).unwrap();
            let direct = estimate_saturation_interference(&query, &network, &assignment).unwrap();
            let route = result.interference().unwrap();
            same(
                direct.estimate.contrast.horvitz_thompson,
                route.contrast.horvitz_thompson,
                &label,
            );
            same(direct.estimate.contrast.hajek, route.contrast.hajek, &label);
            same(
                direct.estimate.contrast.conservative_variance,
                route.contrast.conservative_variance,
                &label,
            );
            same(
                direct.estimate.minimum_exposure_probability,
                route.minimum_exposure_probability,
                &label,
            );
            let inference = result.interference_inference().unwrap();
            assert_eq!(inference.method, "saturation_cluster_neyman_welch", "{label}");
            assert_eq!(inference.from_exposed_units, direct.from_exposed_units, "{label}");
            assert_eq!(inference.to_exposed_units, direct.to_exposed_units, "{label}");
            assert_eq!(inference.from_exposed_clusters, direct.from_exposed_clusters, "{label}");
            assert_eq!(inference.to_exposed_clusters, direct.to_exposed_clusters, "{label}");
            assert_eq!(
                inference.interval_unavailable_reason, direct.interval_unavailable_reason,
                "{label}"
            );
            match (&direct.pointwise_interval, &inference.interval) {
                (Some(kernel), Some(published)) => {
                    assert!(expect_interval, "{label}");
                    same(kernel.bounds[0], published.lower, &label);
                    same(kernel.bounds[1], published.upper, &label);
                    same(kernel.standard_error, published.standard_error, &label);
                    same(kernel.degrees_of_freedom, published.degrees_of_freedom, &label);
                    assert_eq!(
                        [kernel.low_clusters, kernel.high_clusters],
                        published.first_stage_arm_clusters,
                        "{label}"
                    );
                    // The coverage row the interval is recorded under: design
                    // `two_stage_saturation`, method `saturation_cluster_neyman_welch`.
                    assert_eq!(
                        inference.graphless_support_status,
                        Some(CellStatus::Supported),
                        "{label}"
                    );
                }
                (None, None) => {
                    assert!(!expect_interval, "{label}");
                    assert_eq!(inference.graphless_support_status, None, "{label}");
                }
                other => panic!("{label}: interval presence differs: {other:?}"),
            }
        }
    }
}

fn observational_fixture(
    provenance: ExposurePropensityProvenance,
    clusters: usize,
) -> (TabularData, Vec<bool>, InterferenceQuery, Vec<NetworkEdge>) {
    let mut outcomes = Vec::new();
    let mut assignment = Vec::new();
    let mut labels = Vec::new();
    let mut edges = Vec::new();
    for cluster in 0..clusters {
        let treated = cluster % 2 == 1;
        let baseline = 1.0 + (cluster % 7) as f64 / 10.0;
        for unit in 0..2 {
            outcomes.push(baseline + 2.0 * f64::from(treated));
            assignment.push(treated);
            labels.push(cluster as u32);
            edges.push(NetworkEdge::new(
                u64::try_from(2 * cluster + 1 - unit).expect("row id"),
                u64::try_from(2 * cluster + unit).expect("row id"),
                1.0,
            ));
        }
    }
    let data = TabularData::from_f64_columns([("y", outcomes.as_slice())]).unwrap();
    let query = InterferenceQuery::new(
        AssignmentDesign::ObservedExposure {
            clusters: Arc::from(labels),
            propensity_from: Arc::from(vec![0.5; 2 * clusters]),
            propensity_to: Arc::from(vec![0.5; 2 * clusters]),
            provenance,
            assume_network_exchangeability: true,
        },
        ExposureMapping::NeighborCount,
        contrast((0.0, 0.0), (1.0, 1.0)),
    );
    (data, assignment, query, edges)
}

#[test]
fn observational_route_equals_the_direct_estimator_with_and_without_an_interval() {
    let ctx = ExecutionContext::for_tests(353);
    // Known probabilities on sixty clusters publish the interval; externally estimated ones
    // stay point-only.
    for (provenance, expect_interval) in [
        (ExposurePropensityProvenance::Known, true),
        (ExposurePropensityProvenance::ExternallyEstimated, false),
    ] {
        let label = format!("{provenance:?}");
        let (data, assignment, query, edges) = observational_fixture(provenance, 60);
        let result = compiled(&data, &query, edges.clone(), &assignment, &ctx);
        let network = NetworkData::try_new(data.clone(), edges).unwrap();
        let AssignmentDesign::ObservedExposure {
            clusters,
            propensity_from,
            propensity_to,
            provenance,
            ..
        } = &query.assignment
        else {
            unreachable!("observed exposure fixture")
        };
        let InterferenceFunctional::ExposureContrast { outcome, from, to } = query.functional;
        let direct = estimate_observational_exposure(
            &network,
            outcome,
            &ObservationalExposureSpec {
                assignment: &assignment,
                clusters,
                exposure: &query.exposure,
                from,
                to,
                propensity_from,
                propensity_to,
                propensity_provenance: *provenance,
            },
        )
        .unwrap();
        let route = result.interference().unwrap();
        same(direct.horvitz_thompson, route.contrast.horvitz_thompson, &label);
        same(direct.hajek, route.contrast.hajek, &label);
        same(direct.cluster_robust_variance, route.contrast.conservative_variance, &label);
        same(direct.minimum_exposure_probability, route.minimum_exposure_probability, &label);
        let effect = result.estimate().as_effect().unwrap();
        same(direct.horvitz_thompson, effect.ate, &label);
        let inference = result.interference_inference().unwrap();
        assert_eq!(inference.method, "observational_known_exposure_cluster_t", "{label}");
        assert_eq!(inference.from_exposed_units, direct.from_exposed_units, "{label}");
        assert_eq!(inference.to_exposed_units, direct.to_exposed_units, "{label}");
        assert_eq!(inference.from_exposed_clusters, direct.from_exposed_clusters, "{label}");
        assert_eq!(inference.to_exposed_clusters, direct.to_exposed_clusters, "{label}");
        assert_eq!(
            inference.interval_unavailable_reason, direct.interval_unavailable_reason,
            "{label}"
        );
        match (&direct.pointwise_interval, &inference.interval) {
            (Some(kernel), Some(published)) => {
                assert!(expect_interval, "{label}");
                same(kernel.bounds[0], published.lower, &label);
                same(kernel.bounds[1], published.upper, &label);
                same(kernel.standard_error, published.standard_error, &label);
                same(kernel.degrees_of_freedom, published.degrees_of_freedom, &label);
                same(kernel.standard_error, effect.se_analytic(), &label);
                // The coverage row the interval is recorded under: design
                // `observational_known_exposure`, method
                // `observational_known_exposure_cluster_t`.
                assert_eq!(
                    inference.graphless_support_status,
                    Some(CellStatus::Supported),
                    "{label}"
                );
            }
            (None, None) => {
                assert!(!expect_interval, "{label}");
                assert!(effect.se_analytic().is_nan(), "{label}");
            }
            other => panic!("{label}: interval presence differs: {other:?}"),
        }
    }
}
