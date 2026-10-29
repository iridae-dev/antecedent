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
use antecedent::{BayesianConfig, InferenceMode, InterferenceSpec, RefuteSuite, Study};
use antecedent_core::{
    AssignmentDesign, CausalQuery, ExecutionContext, ExposureLevel, ExposureMapping,
    ExposurePropensityProvenance, InterferenceFunctional, InterferenceQuery, VariableId,
};
use antecedent_data::{NetworkData, NetworkEdge, TabularData};
use antecedent_graph::Dag;

#[path = "common/retained_graphless.rs"]
mod retained_graphless;

/// Mean of `outcomes` over the units picked by `to` minus the mean over those picked by
/// `from`: the exposure contrast written out from its definition.
fn group_mean_contrast(outcomes: &[f64], to: impl Fn(usize) -> bool, from: impl Fn(usize) -> bool) -> f64 {
    let mean = |pick: &dyn Fn(usize) -> bool| {
        let chosen = (0..outcomes.len()).filter(|&u| pick(u)).map(|u| outcomes[u]).collect::<Vec<_>>();
        chosen.iter().sum::<f64>() / chosen.len() as f64
    };
    mean(&to) - mean(&from)
}

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
        result.interference_inference().unwrap().graphless_support_status,
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
    let error = antecedent_io::encode_analysis_result_artifact(
        &forged,
        header.variable_names.clone(),
        "forged-interference-license",
    )
    .unwrap_err();
    assert!(
        format!("{error:?}").contains("graphless license does not match the published interval support"),
        "{label}: forged license on a withheld interval must be refused for that reason: {error:?}",
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
                NetworkEdge::new(u64::from(first), u64::from(first + 1), 1.0),
                NetworkEdge::new(u64::from(first + 1), u64::from(first), 1.0),
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
    retained_graphless::assert_retained_graphless_program(
        "licensed_cluster_total_interval_round_tr",
        &prepared,
        &result,
        &ctx,
    );
    // Treated clusters 0..8 against control 8..16: the constant effect 3 plus the base shift
    // 0.5 * (3.5 - 11.5) = -4, so the contrast is -1 exactly.
    let expected = group_mean_contrast(&outcomes, |u| u / 2 < TREATED, |u| u / 2 >= TREATED);
    assert!((expected - -1.0).abs() < 1e-12, "{expected}");
    assert!((result.effect() - expected).abs() < 1e-9, "{} vs {expected}", result.effect());
    let inference = result.interference_inference().unwrap();
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
                NetworkEdge::new(u64::from(first), u64::from(first + 1), 1.0),
                NetworkEdge::new(u64::from(first + 1), u64::from(first), 1.0),
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
    let inference = result.interference_inference().unwrap();
    assert!(inference.interval.is_none());
    assert_eq!(inference.graphless_support_status, None);
    let bytes = prepared.encode_contracted_result(&result, "thin-cluster", &ctx).unwrap();
    let (_, header, mut body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    body.interference_inference.as_mut().unwrap().graphless_support_status =
        Some("licensed".into());
    let error = antecedent_io::encode_analysis_result_artifact(
        &body,
        header.variable_names,
        "forged-thin-cluster",
    )
    .unwrap_err();
    assert!(
        format!("{error:?}")
            .contains("interference graphless license does not match the published interval support"),
        "{error:?}"
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
                (first..first + 3).filter(move |&to| to != from).map(move |to| {
                    NetworkEdge::new(
                        u64::try_from(from).expect("row id"),
                        u64::try_from(to).expect("row id"),
                        1.0,
                    )
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
    retained_graphless::assert_retained_graphless_program(
        "licensed_saturation_interval_round_trips",
        &prepared,
        &result,
        &ctx,
    );
    // Exposure (own 0, neighbours 0) is a unit of a no-treatment cluster (cluster % 4 == 0);
    // (own 1, neighbours 1) is a unit of an all-treated cluster (cluster % 4 == 3). Half the
    // clusters are low (0.2) and half high (0.8), so each exposure has marginal probability
    // (0.8^3 + 0.2^3) / 2 = 0.26 and the Horvitz-Thompson contrast over the 144 units is
    // sum(1[all treated] y - 1[none treated] y) / (0.26 * 144).
    let pi = (0.8_f64.powi(3) + 0.2_f64.powi(3)) / 2.0;
    let total: f64 = (0..outcomes.len())
        .map(|u| match u / 3 % 4 {
            3 => outcomes[u],
            0 => -outcomes[u],
            _ => 0.0,
        })
        .sum();
    let expected = total / (pi * outcomes.len() as f64);
    assert!((expected - 4.807_692_307_692_3).abs() < 1e-9, "{expected}");
    assert!((result.effect() - expected).abs() < 1e-9, "{} vs {expected}", result.effect());
    let inference = result.interference_inference().unwrap();
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
    retained_graphless::assert_retained_graphless_program(
        "licensed_observational_known_interval_ro",
        &prepared,
        &result,
        &ctx,
    );
    // Known propensity 1/2 for both exposures: the contrast is the treated minus untreated
    // mean, 2 plus the baseline difference between odd and even clusters.
    let expected = group_mean_contrast(&outcomes, |u| (u / 2) % 2 == 1, |u| (u / 2) % 2 == 0);
    assert!((result.effect() - expected).abs() < 1e-9, "{} vs {expected}", result.effect());
    let inference = result.interference_inference().unwrap();
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
    let inference = result.interference_inference().unwrap();
    assert!(inference.interval.is_none());
    assert_eq!(inference.graphless_support_status, None);
    assert!(inference.interval_unavailable_reason.unwrap().contains("known, fixed"));
    // A forged license on the externally-fitted point result is refused.
    let bytes = prepared.encode_contracted_result(&result, "external-obs", &ctx).unwrap();
    let (_, header, mut body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    body.interference_inference.as_mut().unwrap().graphless_support_status =
        Some("licensed".into());
    let error = antecedent_io::encode_analysis_result_artifact(
        &body,
        header.variable_names,
        "forged-external-obs",
    )
    .unwrap_err();
    assert!(
        format!("{error:?}")
            .contains("interference graphless license does not match the published interval support"),
        "{error:?}"
    );
}

/// Sixteen two-unit clusters, eight treated, with a constant cluster total effect of 3.
fn cluster_fixture() -> (TabularData, InterferenceQuery, Vec<NetworkEdge>, Vec<bool>) {
    const K: usize = 16;
    const TREATED: usize = 8;
    let clusters = (0..K).flat_map(|cluster| [cluster as u32; 2]).collect::<Vec<_>>();
    let assignment = (0..K).flat_map(|cluster| [cluster < TREATED; 2]).collect::<Vec<_>>();
    let edges = (0..K)
        .flat_map(|cluster| {
            let first = (cluster * 2) as u32;
            [
                NetworkEdge::new(u64::from(first), u64::from(first + 1), 1.0),
                NetworkEdge::new(u64::from(first + 1), u64::from(first), 1.0),
            ]
        })
        .collect::<Vec<_>>();
    let outcomes = (0..K * 2)
        .map(|unit| {
            let cluster = unit / 2;
            let base = 0.5 * ((cluster * 7) % K) as f64 + 0.1 * (unit % 2) as f64;
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
    (data, query, edges, assignment)
}

#[test]
fn cluster_bayesian_bootstrap_centres_on_the_design_estimate_with_a_matching_spread() {
    let (data, query, edges, assignment) = cluster_fixture();
    let ctx = ExecutionContext::for_tests(613);
    let (_, frequentist) = run(&data, query.clone(), edges.clone(), assignment.clone(), &ctx);
    let network = NetworkData::try_new(data.clone(), edges).unwrap();
    let prepared = Study::tabular(data.clone())
        .graph(Dag::with_variables(1))
        .query(CausalQuery::Interference(query))
        .interference(InterferenceSpec { network, assignment: Arc::from(assignment) })
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(8_000)))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    retained_graphless::assert_retained_graphless_program(
        "cluster_bayesian_bootstrap",
        &prepared,
        &result,
        &ctx,
    );
    assert_eq!(
        result.logical_plan.estimator.as_deref(),
        Some("interference.cluster_bayesian_bootstrap")
    );
    assert_eq!(result.support_status(), None, "the Bayesian bootstrap route is unlicensed");
    let posterior = result.posterior.as_ref().unwrap();
    let column = posterior.effect_column().unwrap();
    let design_effect = frequentist.effect();
    // The Bayesian bootstrap of the arm means centres on the design-based contrast.
    // Treated clusters 0..8 have mean (7c mod 16) 6.5 and control clusters 8..16 have 8.5,
    // so the contrast is 3 + 0.5 * (6.5 - 8.5) = 2.
    assert!((design_effect - 2.0).abs() < 1e-9, "{design_effect}");
    assert!((posterior.summaries.mean[column] - design_effect).abs() < 0.05, "{}", posterior.summaries.mean[column]);
    // Its spread is of the order of the conservative cluster Neyman standard error.
    let neyman_se =
        frequentist.interference_inference().unwrap().interval.as_ref().unwrap().standard_error;
    let sd = posterior.summaries.sd[column];
    assert!(sd > 0.7 * neyman_se && sd < 1.1 * neyman_se, "{sd} vs {neyman_se}");
}

#[test]
fn cluster_bayesian_bootstrap_refuses_a_thin_arm_at_prepare() {
    let (data, mut query, edges, mut assignment) = cluster_fixture();
    if let AssignmentDesign::ClusterRandomization { treated_clusters, .. } = &mut query.assignment {
        *treated_clusters = 1;
    }
    for flag in assignment.iter_mut().skip(2) {
        *flag = false;
    }
    let network = NetworkData::try_new(data.clone(), edges).unwrap();
    let ctx = ExecutionContext::for_tests(614);
    let outcome = Study::tabular(data)
        .graph(Dag::with_variables(1))
        .query(CausalQuery::Interference(query))
        .interference(InterferenceSpec { network, assignment: Arc::from(assignment) })
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(200)))
        .refute(RefuteSuite::None)
        .build()
        .and_then(|study| study.prepare(&ctx));
    let error = outcome.err().expect("a thin arm must be refused at prepare");
    assert!(
        format!("{error:?}").contains("at least two observed clusters in each arm"),
        "{error:?}"
    );
}
