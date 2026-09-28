//! Retained off-axis interference exposure-contrast intervals: the cluster
//! total effect, the exact two-stage saturation contrast, and the known-exposure
//! observational contrast each carry an exact graphless license that survives the
//! artifact round trip and cannot be forged onto a withheld interval.
#![allow(
    clippy::float_cmp,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    reason = "integration test asserts exact deterministic estimates and builds fixtures from small nonnegative counts"
)]
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::support::CellStatus;
use antecedent::{InterferenceSpec, RefuteSuite, Study};
use antecedent_core::{
    AssignmentDesign, CausalQuery, ExecutionContext, ExposureLevel, ExposureMapping,
    ExposurePropensityProvenance, InterferenceFunctional, InterferenceQuery, VariableId,
};
use antecedent_data::{NetworkData, NetworkEdge, TabularData};
use antecedent_graph::Dag;

fn run(
    data: &TabularData,
    query: InterferenceQuery,
    edges: Vec<NetworkEdge>,
    assignment: Vec<bool>,
    ctx: &ExecutionContext,
) -> (antecedent::PreparedStudy, antecedent::StudyResult) {
    let network = NetworkData::try_new(data.clone(), edges).unwrap();
    let prepared = Study::tabular(data.clone())
        .graph(Dag::with_variables(1))
        .query(CausalQuery::Interference(query))
        .interference(InterferenceSpec { network, assignment: Arc::from(assignment) })
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .prepare(ctx)
        .unwrap();
    let result = prepared.estimate(data, ctx).unwrap();
    (prepared, result)
}

/// Encode a licensed result, decode it, assert the wire carries the license, then
/// strip the interval while keeping the licensed status and prove the re-encode
/// (which revalidates) rejects the forged license.
fn assert_license_round_trips_and_refuses_forgery(
    prepared: &antecedent::PreparedStudy,
    result: &antecedent::StudyResult,
    label: &str,
    ctx: &ExecutionContext,
) {
    assert_eq!(
        result.interference_inference.as_ref().unwrap().graphless_support_status,
        Some(CellStatus::Licensed),
        "{label}: a supported interval must be licensed",
    );
    let bytes = prepared.encode_contracted_result(result, label, ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    let inference = body.interference_inference.as_ref().unwrap();
    assert_eq!(inference.graphless_support_status.as_deref(), Some("licensed"), "{label}");
    assert!(inference.interval.is_some(), "{label}");
    // A licensed status with the interval withheld is not evidenced: revalidation
    // on re-encode must refuse it.
    let mut forged = body.clone();
    let forged_inference = forged.interference_inference.as_mut().unwrap();
    forged_inference.interval = None;
    forged_inference.interval_unavailable_reason = Some("forged withheld interval".into());
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &forged,
            header.variable_names.clone(),
            "forged-interference-license",
        )
        .is_err(),
        "{label}: forged license on a withheld interval must be refused",
    );
}

#[test]
fn licensed_cluster_total_interval_round_trips_and_refuses_forgery() {
    const K: usize = 16;
    const TREATED: usize = 8;
    let clusters = (0..K).flat_map(|cluster| [cluster as u32; 2]).collect::<Vec<_>>();
    let assignment = (0..K).flat_map(|cluster| [cluster < TREATED; 2]).collect::<Vec<_>>();
    let edges = (0..K)
        .flat_map(|cluster| {
            let first = (cluster * 2) as u32;
            [
                NetworkEdge { from: first, to: first + 1, weight: 1.0 },
                NetworkEdge { from: first + 1, to: first, weight: 1.0 },
            ]
        })
        .collect::<Vec<_>>();
    let outcomes = (0..K * 2)
        .map(|unit| {
            let cluster = unit / 2;
            let base = 0.5 * cluster as f64 + 0.1 * (unit % 2) as f64;
            base + if cluster < TREATED { 3.0 } else { 0.0 }
        })
        .collect::<Vec<_>>();
    let data = TabularData::from_f64_columns([("y", outcomes.as_slice())]).unwrap();
    let query = InterferenceQuery::new(
        AssignmentDesign::ClusterRandomization {
            clusters: Arc::from(clusters),
            treated_clusters: TREATED,
        },
        ExposureMapping::NeighborFraction,
        InterferenceFunctional::ExposureContrast {
            outcome: VariableId::from_raw(0),
            from: ExposureLevel { own: 0.0, neighbors: 0.0 },
            to: ExposureLevel { own: 1.0, neighbors: 1.0 },
        },
    );
    let ctx = ExecutionContext::for_tests(611);
    let (prepared, result) = run(&data, query, edges, assignment, &ctx);
    let inference = result.interference_inference.as_ref().unwrap();
    assert_eq!(inference.method, "cluster_total_neyman_welch");
    assert_eq!(inference.interval.as_ref().unwrap().first_stage_arm_clusters, [8, 8]);
    assert_license_round_trips_and_refuses_forgery(&prepared, &result, "cluster-total", &ctx);
}

#[test]
fn thin_cluster_total_cannot_forge_a_license() {
    // Four clusters (two per arm) is below the eight-cluster interval floor, so no
    // interval and no license exist; a forged licensed status must be refused.
    const K: usize = 4;
    let clusters = (0..K).flat_map(|cluster| [cluster as u32; 2]).collect::<Vec<_>>();
    let assignment = (0..K).flat_map(|cluster| [cluster < 2; 2]).collect::<Vec<_>>();
    let edges = (0..K)
        .flat_map(|cluster| {
            let first = (cluster * 2) as u32;
            [
                NetworkEdge { from: first, to: first + 1, weight: 1.0 },
                NetworkEdge { from: first + 1, to: first, weight: 1.0 },
            ]
        })
        .collect::<Vec<_>>();
    let outcomes = (0..K * 2)
        .map(|unit| 0.5 * (unit / 2) as f64 + if unit / 2 < 2 { 3.0 } else { 0.0 })
        .collect::<Vec<_>>();
    let data = TabularData::from_f64_columns([("y", outcomes.as_slice())]).unwrap();
    let query = InterferenceQuery::new(
        AssignmentDesign::ClusterRandomization {
            clusters: Arc::from(clusters),
            treated_clusters: 2,
        },
        ExposureMapping::NeighborFraction,
        InterferenceFunctional::ExposureContrast {
            outcome: VariableId::from_raw(0),
            from: ExposureLevel { own: 0.0, neighbors: 0.0 },
            to: ExposureLevel { own: 1.0, neighbors: 1.0 },
        },
    );
    let ctx = ExecutionContext::for_tests(612);
    let (prepared, result) = run(&data, query, edges, assignment, &ctx);
    let inference = result.interference_inference.as_ref().unwrap();
    assert!(inference.interval.is_none());
    assert_eq!(inference.graphless_support_status, None);
    let bytes = prepared.encode_contracted_result(&result, "thin-cluster", &ctx).unwrap();
    let (_, header, mut body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    body.interference_inference.as_mut().unwrap().graphless_support_status =
        Some("licensed".into());
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &body,
            header.variable_names,
            "forged-thin-cluster"
        )
        .is_err()
    );
}

#[test]
fn licensed_saturation_interval_round_trips_and_refuses_forgery() {
    let clusters = (0..48).flat_map(|cluster| [cluster; 3]).collect::<Vec<u32>>();
    let patterns =
        [[false, false, false], [true, false, false], [true, true, false], [true, true, true]];
    let assignment = (0..48).flat_map(|cluster| patterns[cluster % 4]).collect::<Vec<_>>();
    let realized =
        (0..48).flat_map(|cluster| [if cluster < 24 { 0.2 } else { 0.8 }; 3]).collect::<Vec<_>>();
    let edges = (0..48)
        .flat_map(|cluster| {
            let first = cluster * 3;
            (first..first + 3).flat_map(move |from| {
                (first..first + 3).filter(move |&to| to != from).map(move |to| NetworkEdge {
                    from: from as u32,
                    to: to as u32,
                    weight: 1.0,
                })
            })
        })
        .collect::<Vec<_>>();
    let outcomes = (0..assignment.len())
        .map(|unit| {
            let first = unit / 3 * 3;
            let neighbor_fraction = (first..first + 3)
                .filter(|&other| other != unit && assignment[other])
                .count() as f64
                / 2.0;
            5.0 + 0.1 * (unit / 3 % 9) as f64
                + 2.0 * f64::from(assignment[unit])
                + 3.0 * neighbor_fraction
        })
        .collect::<Vec<_>>();
    let data = TabularData::from_f64_columns([("y", outcomes.as_slice())]).unwrap();
    let query = InterferenceQuery::new(
        AssignmentDesign::TwoStageSaturation {
            clusters: Arc::from(clusters),
            low_probability: 0.2,
            high_probability: 0.8,
            high_clusters: 24,
            realized_saturation: Arc::from(realized),
        },
        ExposureMapping::NeighborFraction,
        InterferenceFunctional::ExposureContrast {
            outcome: VariableId::from_raw(0),
            from: ExposureLevel { own: 0.0, neighbors: 0.0 },
            to: ExposureLevel { own: 1.0, neighbors: 1.0 },
        },
    );
    let ctx = ExecutionContext::for_tests(613);
    let (prepared, result) = run(&data, query, edges, assignment, &ctx);
    let inference = result.interference_inference.as_ref().unwrap();
    assert_eq!(inference.method, "saturation_cluster_neyman_welch");
    assert_eq!(inference.interval.as_ref().unwrap().first_stage_arm_clusters, [24, 24]);
    assert_license_round_trips_and_refuses_forgery(&prepared, &result, "saturation", &ctx);
}

#[test]
fn licensed_observational_known_interval_round_trips_and_refuses_forgery() {
    let mut outcomes = Vec::new();
    let mut assignment = Vec::new();
    let mut clusters = Vec::new();
    let mut edges = Vec::new();
    for cluster in 0..60 {
        let treated = cluster % 2 == 1;
        let baseline = 1.0 + (cluster % 7) as f64 / 10.0;
        for unit in 0..2 {
            outcomes.push(baseline + 2.0 * f64::from(treated));
            assignment.push(treated);
            clusters.push(cluster as u32);
            edges.push(NetworkEdge {
                from: (2 * cluster + 1 - unit) as u32,
                to: (2 * cluster + unit) as u32,
                weight: 1.0,
            });
        }
    }
    let data = TabularData::from_f64_columns([("y", outcomes.as_slice())]).unwrap();
    let query = InterferenceQuery::new(
        AssignmentDesign::ObservedExposure {
            clusters: Arc::from(clusters),
            propensity_from: Arc::from(vec![0.5; 120]),
            propensity_to: Arc::from(vec![0.5; 120]),
            provenance: ExposurePropensityProvenance::Known,
            assume_network_exchangeability: true,
        },
        ExposureMapping::NeighborCount,
        InterferenceFunctional::ExposureContrast {
            outcome: VariableId::from_raw(0),
            from: ExposureLevel { own: 0.0, neighbors: 0.0 },
            to: ExposureLevel { own: 1.0, neighbors: 1.0 },
        },
    );
    let ctx = ExecutionContext::for_tests(614);
    let (prepared, result) = run(&data, query, edges, assignment, &ctx);
    let inference = result.interference_inference.as_ref().unwrap();
    assert_eq!(inference.method, "observational_known_exposure_cluster_t");
    assert_eq!(inference.interval.as_ref().unwrap().first_stage_arm_clusters, [30, 30]);
    assert_license_round_trips_and_refuses_forgery(&prepared, &result, "observational-known", &ctx);
}

#[test]
fn externally_estimated_observational_stays_point_only() {
    // Externally fitted exposure probabilities carry a declared, unverifiable
    // network-confounding assumption, so no interval and no license are issued.
    let mut outcomes = Vec::new();
    let mut assignment = Vec::new();
    let mut clusters = Vec::new();
    let mut edges = Vec::new();
    for cluster in 0..60 {
        let treated = cluster % 2 == 1;
        for unit in 0..2 {
            outcomes.push(1.0 + 2.0 * f64::from(treated));
            assignment.push(treated);
            clusters.push(cluster as u32);
            edges.push(NetworkEdge {
                from: (2 * cluster + 1 - unit) as u32,
                to: (2 * cluster + unit) as u32,
                weight: 1.0,
            });
        }
    }
    let data = TabularData::from_f64_columns([("y", outcomes.as_slice())]).unwrap();
    let query = InterferenceQuery::new(
        AssignmentDesign::ObservedExposure {
            clusters: Arc::from(clusters),
            propensity_from: Arc::from(vec![0.5; 120]),
            propensity_to: Arc::from(vec![0.5; 120]),
            provenance: ExposurePropensityProvenance::ExternallyEstimated,
            assume_network_exchangeability: true,
        },
        ExposureMapping::NeighborCount,
        InterferenceFunctional::ExposureContrast {
            outcome: VariableId::from_raw(0),
            from: ExposureLevel { own: 0.0, neighbors: 0.0 },
            to: ExposureLevel { own: 1.0, neighbors: 1.0 },
        },
    );
    let ctx = ExecutionContext::for_tests(615);
    let (prepared, result) = run(&data, query, edges, assignment, &ctx);
    let inference = result.interference_inference.as_ref().unwrap();
    assert!(inference.interval.is_none());
    assert_eq!(inference.graphless_support_status, None);
    assert!(inference.interval_unavailable_reason.unwrap().contains("known, fixed"));
    // A forged license on the externally-fitted point result is refused.
    let bytes = prepared.encode_contracted_result(&result, "external-obs", &ctx).unwrap();
    let (_, header, mut body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    body.interference_inference.as_mut().unwrap().graphless_support_status =
        Some("licensed".into());
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &body,
            header.variable_names,
            "forged-external-obs"
        )
        .is_err()
    );
}
