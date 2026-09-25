//! Public lifecycle evidence for explicit Unknown-tier average effects.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::float_cmp)]

use antecedent::{RefuteSuite, Study};
use antecedent_core::{AverageEffectQuery, ExecutionContext};
use antecedent_data::{TableView, TabularData};
use antecedent_graph::{TieredBackground, WithinTier};

mod common;
use common::calibration::gaussian;

fn unknown_data(n: usize, seed: u64, shift: f64) -> TabularData {
    let mut g = gaussian(seed);
    let (mut era, mut t, mut m, mut y) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        era[i] = g();
        t[i] = 0.8 * era[i] + g();
        m[i] = t[i] + 0.2 * era[i] + 0.5 * g();
        y[i] = -t[i] + 2.0 * m[i] + 0.2 * era[i] + g() + shift;
    }
    TabularData::from_f64_columns([
        ("era", era.as_slice()),
        ("t", t.as_slice()),
        ("m", m.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap()
}

fn background(data: &TabularData) -> TieredBackground {
    TieredBackground::from_named(
        data.schema(),
        &[vec!["era"], vec!["t", "m"], vec!["y"]],
        WithinTier::Unknown,
    )
    .unwrap()
}

fn query(data: &TabularData) -> AverageEffectQuery {
    AverageEffectQuery::binary_ate(
        data.schema().id_of("t").unwrap(),
        data.schema().id_of("y").unwrap(),
    )
}

#[test]
fn unknown_tiered_average_executes_from_retained_plan_refreshes_and_refuses_unreplayable_artifact()
{
    let data = unknown_data(4_000, 23, 0.0);
    let ctx = ExecutionContext::for_tests(23);
    let builder = Study::tabular(data.clone())
        .tiered_background(background(&data))
        .unwrap()
        .query(query(&data))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let mut prepared = builder.prepare(&ctx).unwrap();
    let plan = prepared.checked_unknown_tiered_average_info().expect("sealed two-scenario plan");
    assert_eq!(plan.query, query(&data));
    assert_eq!(plan.tier_count, 3);
    assert_eq!(plan.scenario_count, 2);
    assert_eq!(plan.estimator.as_str(), "linear.adjustment.ate");
    assert_eq!(plan.identifier.as_str(), "generalized.adjustment");
    assert!(!plan.plan_id.is_empty());
    drop(builder);

    let first = prepared.estimate(&data, &ctx).unwrap();
    assert_eq!(first.support_status.unwrap().as_str(), "licensed");
    assert_eq!(format!("{:?}", first.identification.status), "GraphDependent");
    assert!(
        first.estimate.ate.is_nan(),
        "ambiguous scenarios must not be collapsed to a scalar ATE"
    );
    let values = first.estimate.scenario_effects.as_ref().expect("scenario values");
    let covariance = first.estimate.joint_covariance.as_ref().expect("shared joint covariance");
    let bands = first.estimate.scenario_intervals.as_ref().expect("joint simultaneous intervals");
    let structure = first.structural_response.as_ref().expect("scenario mass receipt");
    let truths = [1.0, -1.0];
    assert_eq!(values.len(), 2);
    assert_eq!(bands.len(), 2);
    assert_eq!(structure.atoms.len(), 2);
    assert_eq!(structure.identified_mass, 1.0);
    assert_eq!(structure.unidentified_mass, 0.0);
    assert_eq!(structure.unevaluable_mass, 0.0);
    assert_eq!(structure.subsampled_out_mass, 0.0);
    for i in 0..2 {
        assert!(covariance.se(i).is_finite() && covariance.se(i) > 0.0);
        assert!((values[i] - truths[i]).abs() < 4.0 * covariance.se(i));
        assert!(bands[i].0 <= truths[i] && truths[i] <= bands[i].1);
    }

    let refreshed_data = unknown_data(4_000, 23, 0.7);
    let refreshed = prepared.refresh(refreshed_data, &ctx).unwrap();
    assert!(refreshed.estimate.ate.is_nan());
    assert_eq!(refreshed.estimate.scenario_effects.as_ref().unwrap().len(), 2);
    assert!(refreshed.estimate.joint_covariance.is_some());

    let artifact =
        prepared.encode_contracted_result(&refreshed, "unknown-tiered-average", &ctx).unwrap();
    let consumed = antecedent_io::consume_analysis_result(&artifact).unwrap();
    assert!(consumed.acceptance.unresolved.iter().any(|reason| {
        reason.as_ref() == "dependencies.checked_unknown_tiered_average_operation"
    }));
    assert!(!consumed.acceptance.accepts_as_verified_program());
}
