//! Retained two-stage saturation design through prepare, analyze, and artifact transport.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{EstimatorId, InterferenceSpec, RefuteSuite, Study};
use antecedent_core::{AssignmentDesign, CausalQuery, ExecutionContext, ExposureLevel, ExposureMapping, InterferenceFunctional, InterferenceQuery, VariableId};
use antecedent_data::{NetworkData, NetworkEdge, TabularData};
use antecedent_graph::Dag;
use antecedent_io::{consume_analysis_result, interference_query_from_wire, interference_query_to_wire};

fn fixture(from: (f64, f64), to: (f64, f64)) -> (TabularData, Vec<bool>, InterferenceQuery, Vec<NetworkEdge>) {
    let clusters = [0_u32, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3];
    let assignment = vec![false, false, false, true, false, false, true, true, false, true, true, true];
    let edges = (0..4).flat_map(|cluster| {
        let first = cluster * 3;
        (first..first + 3).flat_map(move |from| {
            (first..first + 3).filter(move |&to| to != from).map(move |to| NetworkEdge { from: from as u32, to: to as u32, weight: 1.0 })
        })
    }).collect::<Vec<_>>();
    let outcomes = (0..12).map(|i| {
        let first = i / 3 * 3;
        let neighbors = (first..first + 3).filter(|&j| j != i && assignment[j]).count() as f64 / 2.0;
        let own = if assignment[i] { 1.0 } else { 0.0 };
        1.0 + 2.0 * own + 3.0 * neighbors + 4.0 * own * neighbors
    }).collect::<Vec<_>>();
    let data = TabularData::from_f64_columns([("y", outcomes.as_slice())]).unwrap();
    let query = InterferenceQuery::new(
        AssignmentDesign::TwoStageSaturation {
            clusters: Arc::from(clusters), low_probability: 0.2, high_probability: 0.8,
            high_clusters: 2, realized_saturation: Arc::from([0.2; 6].into_iter().chain([0.8; 6]).collect::<Vec<_>>()),
        },
        ExposureMapping::NeighborFraction,
        InterferenceFunctional::ExposureContrast {
            outcome: VariableId::from_raw(0),
            from: ExposureLevel { own: from.0, neighbors: from.1 },
            to: ExposureLevel { own: to.0, neighbors: to.1 },
        },
    );
    (data, assignment, query, edges)
}

#[test]
fn saturation_direct_spillover_total_retain_truth_without_interval() {
    let ctx = ExecutionContext::for_tests(241);
    for (from, to, truth) in [
        ((0.0, 0.5), (1.0, 0.5), 4.0),
        ((0.0, 0.0), (0.0, 1.0), 3.0),
        ((0.0, 0.0), (1.0, 1.0), 9.0),
    ] {
        let (data, assignment, query, edges) = fixture(from, to);
        let wire = interference_query_to_wire(&query).unwrap();
        assert_eq!(interference_query_from_wire(&wire).unwrap(), query);
        let network = NetworkData::try_new(data.clone(), edges).unwrap();
        let study = Study::tabular(data.clone())
            .graph(Dag::with_variables(1))
            .query(CausalQuery::Interference(query.clone()))
            .interference(InterferenceSpec { network, assignment: Arc::from(assignment) })
            .refute(RefuteSuite::None)
            .build().unwrap();
        let prepared = study.prepare(&ctx).unwrap();
        assert_eq!(prepared.checked_interference_info().unwrap().estimator, EstimatorId::InterferenceSaturationExact);
        let result = prepared.estimate(&data, &ctx).unwrap();
        assert_eq!(result.support_status, None);
        assert!((result.interference.as_ref().unwrap().contrast.hajek - truth).abs() < 1e-12);
        assert!(result.interference.as_ref().unwrap().contrast.conservative_variance > 0.0);
        assert!(result.estimate.se_analytic.is_nan());
        let inference = result.interference_inference.as_ref().unwrap();
        assert!(inference.interval.is_none());
        assert!(inference.interval_unavailable_reason.unwrap().contains("eight independent clusters"));
        assert!(result.estimate.assumptions.entries.iter().any(|a| matches!(&a.assumption, antecedent_core::Assumption::Custom { id, .. } if id.as_ref() == "interference.two_stage_saturation")));
        let artifact = prepared.encode_contracted_result(&result, "saturation-exact", &ctx).unwrap();
        let consumed = consume_analysis_result(&artifact).unwrap();
        assert_eq!(consumed.body.query, antecedent_io::CausalQueryWire::Interference(wire));
        let wire_inference = consumed.body.interference_inference.as_ref().unwrap();
        assert!(wire_inference.interval.is_none());
        assert_eq!(wire_inference.interval_unavailable_reason.as_deref(), inference.interval_unavailable_reason);
        let mut tampered = artifact.clone();
        let middle = tampered.len() / 2;
        tampered[middle] ^= 0x40;
        assert!(consume_analysis_result(&tampered).is_err());
    }
}

#[test]
fn saturation_rejects_cross_cluster_edges_at_prepare() {
    let (data, assignment, query, mut edges) = fixture((0.0, 0.5), (1.0, 0.5));
    edges.push(NetworkEdge { from: 0, to: 3, weight: 1.0 });
    let network = NetworkData::try_new(data.clone(), edges).unwrap();
    let study = Study::tabular(data)
        .graph(Dag::with_variables(1))
        .query(CausalQuery::Interference(query))
        .interference(InterferenceSpec { network, assignment: Arc::from(assignment) })
        .refute(RefuteSuite::None)
        .build().unwrap();
    assert!(study.prepare(&ExecutionContext::for_tests(241)).is_err());
}

#[test]
fn supported_saturation_interval_survives_retained_artifact_round_trip() {
    let clusters = (0..80).flat_map(|cluster| [cluster; 3]).collect::<Vec<u32>>();
    let patterns = [[false, false, false], [true, false, false], [true, true, false], [true, true, true]];
    let assignment = (0..80).flat_map(|cluster| patterns[cluster % 4]).collect::<Vec<_>>();
    let realized = (0..80).flat_map(|cluster| [if cluster < 40 { 0.2 } else { 0.8 }; 3]).collect::<Vec<_>>();
    let edges = (0..80).flat_map(|cluster| {
        let first = cluster * 3;
        (first..first + 3).flat_map(move |from| {
            (first..first + 3).filter(move |&to| to != from).map(move |to| NetworkEdge {
                from: from as u32, to: to as u32, weight: 1.0,
            })
        })
    }).collect::<Vec<_>>();
    let outcomes = (0..assignment.len()).map(|unit| {
        let first = unit / 3 * 3;
        let neighbor_fraction = (first..first + 3).filter(|&other| other != unit && assignment[other]).count() as f64 / 2.0;
        5.0 + 0.1 * (unit / 3 % 9) as f64 + 2.0 * f64::from(assignment[unit]) + 3.0 * neighbor_fraction
    }).collect::<Vec<_>>();
    let data = TabularData::from_f64_columns([("y", outcomes.as_slice())]).unwrap();
    let ctx = ExecutionContext::for_tests(241);
    for (name, from, to, truth) in [
        ("direct", (0.0, 0.5), (1.0, 0.5), 2.0),
        ("spillover", (0.0, 0.0), (0.0, 1.0), 3.0),
        ("total", (0.0, 0.0), (1.0, 1.0), 5.0),
    ] {
        let network = NetworkData::try_new(data.clone(), edges.clone()).unwrap();
        let query = InterferenceQuery::new(
            AssignmentDesign::TwoStageSaturation {
                clusters: Arc::from(clusters.clone()), low_probability: 0.2, high_probability: 0.8,
                high_clusters: 40, realized_saturation: Arc::from(realized.clone()),
            },
            ExposureMapping::NeighborFraction,
            InterferenceFunctional::ExposureContrast {
                outcome: VariableId::from_raw(0),
                from: ExposureLevel { own: from.0, neighbors: from.1 },
                to: ExposureLevel { own: to.0, neighbors: to.1 },
            },
        );
        let study = Study::tabular(data.clone())
            .graph(Dag::with_variables(1))
            .query(CausalQuery::Interference(query))
            .interference(InterferenceSpec { network, assignment: Arc::from(assignment.clone()) })
            .refute(RefuteSuite::None)
            .build().unwrap();
        let prepared = study.prepare(&ctx).unwrap();
        let result = prepared.estimate(&data, &ctx).unwrap();
        let inference = result.interference_inference.as_ref().unwrap();
        let interval = inference.interval.as_ref().expect("adequate independent cluster and exposure support");
        assert_eq!(interval.first_stage_arm_clusters, [40, 40], "{name}");
        assert!(interval.lower < truth && truth < interval.upper, "{name}");
        assert!(interval.lower < result.interference.as_ref().unwrap().contrast.horvitz_thompson);
        assert!(result.interference.as_ref().unwrap().contrast.horvitz_thompson < interval.upper);
        let artifact = prepared.encode_contracted_result(&result, name, &ctx).unwrap();
        let consumed = consume_analysis_result(&artifact).unwrap();
        let wire = consumed.body.interference_inference.unwrap();
        assert_eq!(wire.method, "saturation_cluster_neyman_welch", "{name}");
        assert_eq!(wire.interval.unwrap().lower, interval.lower, "{name}");
        let mut tampered = artifact;
        let middle = tampered.len() / 2;
        tampered[middle] ^= 0x40;
        assert!(consume_analysis_result(&tampered).is_err(), "{name}");
    }
}
