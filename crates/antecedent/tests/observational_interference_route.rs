//! Retained supplied-propensity observational network exposure through the study lifecycle.
#![allow(
    clippy::float_cmp,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    reason = "integration test asserts exact deterministic estimates and builds fixtures from small nonnegative counts"
)]
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{EstimatorId, InterferenceSpec, RefuteSuite, Study};
use antecedent_core::{
    AssignmentDesign, CausalQuery, ExecutionContext, ExposureLevel, ExposureMapping,
    ExposurePropensityProvenance, InterferenceFunctional, InterferenceQuery, VariableId,
};
use antecedent_data::{NetworkData, NetworkEdge, TabularData};
use antecedent_graph::Dag;
use antecedent_io::{
    consume_analysis_result, interference_query_from_wire, interference_query_to_wire,
};

fn fixture() -> (TabularData, Vec<bool>, InterferenceQuery, Vec<NetworkEdge>) {
    let data = TabularData::from_f64_columns([("y", &[0.0, 0.0, 6.0, 6.0][..])]).unwrap();
    let assignment = vec![false, false, true, true];
    let query = InterferenceQuery::new(
        AssignmentDesign::ObservedExposure {
            clusters: Arc::from([0, 0, 1, 1]),
            propensity_from: Arc::from([0.5; 4]),
            propensity_to: Arc::from([0.5; 4]),
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
    let edges = vec![
        NetworkEdge { from: 0, to: 1, weight: 1.0 },
        NetworkEdge { from: 1, to: 0, weight: 1.0 },
        NetworkEdge { from: 2, to: 3, weight: 1.0 },
        NetworkEdge { from: 3, to: 2, weight: 1.0 },
    ];
    (data, assignment, query, edges)
}

#[test]
fn observational_network_contrast_retains_truth_assumptions_and_artifact() {
    let (data, assignment, query, edges) = fixture();
    let wire = interference_query_to_wire(&query).unwrap();
    assert_eq!(interference_query_from_wire(&wire).unwrap(), query);
    let network = NetworkData::try_new(data.clone(), edges).unwrap();
    let study = Study::tabular(data.clone())
        .graph(Dag::with_variables(1))
        .query(CausalQuery::Interference(query))
        .interference(InterferenceSpec { network, assignment: Arc::from(assignment) })
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(352);
    let prepared = study.prepare(&ctx).unwrap();
    assert_eq!(
        prepared.checked_interference_info().unwrap().estimator,
        EstimatorId::InterferenceObservationalIpw
    );
    let result = prepared.estimate(&data, &ctx).unwrap();
    assert_eq!(result.support_status, None);
    assert!((result.estimate.ate - 6.0).abs() < 1e-12);
    assert!(result.estimate.se_analytic.is_nan());
    let observed = result.interference.as_ref().unwrap();
    assert!((observed.contrast.hajek - 6.0).abs() < 1e-12);
    assert!((observed.contrast.conservative_variance - 36.0).abs() < 1e-12);
    assert_eq!(
        observed.from_probability_method,
        antecedent_stats::ExposureProbabilityMethod::SuppliedExternallyEstimated
    );
    assert!(result.interference_inference.as_ref().unwrap().interval.is_none());
    assert!(
        result
            .interference_inference
            .as_ref()
            .unwrap()
            .interval_unavailable_reason
            .unwrap()
            .contains("known, fixed")
    );
    assert!(result.estimate.assumptions.entries.iter().any(|entry| matches!(&entry.assumption, antecedent_core::Assumption::Custom { id, .. } if id.as_ref() == "interference.network_exchangeability")));
    assert!(result.diagnostics.iter().any(|d| {
        d.code.as_ref() == "estimate.interference.observational_ipw"
            && d.fields
                .iter()
                .any(|(key, value)| key.as_ref() == "from_exposed_units" && value.as_ref() == "2")
    }));
    let artifact =
        prepared.encode_contracted_result(&result, "observational-network", &ctx).unwrap();
    let consumed = consume_analysis_result(&artifact).unwrap();
    assert_eq!(consumed.body.query, antecedent_io::CausalQueryWire::Interference(wire));
    assert!(consumed.body.standard_error.is_none());
    assert!(consumed.body.interval_lower.is_none());
    let mut tampered = artifact;
    let middle = tampered.len() / 2;
    tampered[middle] ^= 0x40;
    assert!(consume_analysis_result(&tampered).is_err());
}

#[test]
fn known_probabilities_retain_pointwise_interval_and_artifact() {
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
    let network = NetworkData::try_new(data.clone(), edges).unwrap();
    let study = Study::tabular(data.clone())
        .graph(Dag::with_variables(1))
        .query(CausalQuery::Interference(query))
        .interference(InterferenceSpec { network, assignment: Arc::from(assignment) })
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(353);
    let prepared = study.prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let interval = result.interference_inference.as_ref().unwrap().interval.as_ref().unwrap();
    assert!(interval.lower < 2.0 && interval.upper > 2.0);
    assert_eq!(interval.degrees_of_freedom, 59.0);
    assert_eq!(interval.first_stage_arm_clusters, [30, 30]);
    let artifact = prepared.encode_contracted_result(&result, "known-network", &ctx).unwrap();
    let consumed = consume_analysis_result(&artifact).unwrap();
    let wire = consumed.body.interference_inference.unwrap();
    assert_eq!(wire.method, "observational_known_exposure_cluster_t");
    assert!(wire.interval.is_some());
    assert!(consumed.body.standard_error.is_some());
}

#[test]
fn observational_network_refuses_missing_assumption_and_cross_cluster_edge() {
    let (data, assignment, mut query, mut edges) = fixture();
    if let AssignmentDesign::ObservedExposure { assume_network_exchangeability, .. } =
        &mut query.assignment
    {
        *assume_network_exchangeability = false;
    }
    assert!(query.validate().is_err());
    let (_, _, query, _) = fixture();
    edges.push(NetworkEdge { from: 0, to: 2, weight: 1.0 });
    let network = NetworkData::try_new(data.clone(), edges).unwrap();
    let study = Study::tabular(data)
        .graph(Dag::with_variables(1))
        .query(CausalQuery::Interference(query))
        .interference(InterferenceSpec { network, assignment: Arc::from(assignment) })
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    assert!(study.prepare(&ExecutionContext::for_tests(352)).is_err());
}
