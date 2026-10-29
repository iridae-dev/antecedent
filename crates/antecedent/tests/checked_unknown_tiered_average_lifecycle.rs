//! Public lifecycle evidence for explicit Unknown-tier average effects.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::float_cmp, reason = "lifecycle pins compare bitwise-identical replayed estimates")]

use antecedent::analysis::RetainedOperation;
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
    std::thread::Builder::new()
        .name("checked-unknown-tiered-average".into())
        .stack_size(8 * 1024 * 1024)
        .spawn(unknown_tiered_average_lifecycle_body)
        .unwrap()
        .join()
        .unwrap();
}

fn unknown_tiered_average_lifecycle_body() {
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
    let RetainedOperation::UnknownTieredAverage(plan) = prepared.retained_operation() else {
        panic!("sealed two-scenario plan")
    };
    assert_eq!(plan.query, query(&data));
    assert_eq!(plan.tier_count, 3);
    assert_eq!(plan.scenario_count, 2);
    assert_eq!(plan.estimator.as_str(), "linear.adjustment.ate");
    assert_eq!(plan.identifier.as_str(), "generalized.adjustment");
    assert!(!plan.plan_id.is_empty());
    drop(builder);

    let first = prepared.estimate(&data, &ctx).unwrap();
    assert_eq!(first.support_status().unwrap().as_str(), "licensed");
    assert_eq!(format!("{:?}", first.identification.status), "GraphDependent");
    assert!(
        first.estimate().as_effect().unwrap().ate.is_nan(),
        "ambiguous scenarios must not be collapsed to a scalar ATE"
    );
    let values =
        first.estimate().as_effect().unwrap().scenarios.effects.as_ref().expect("scenario values");
    let covariance = first
        .estimate()
        .as_effect()
        .unwrap()
        .joint
        .covariance
        .as_ref()
        .expect("shared joint covariance");
    let bands = first
        .estimate()
        .as_effect()
        .unwrap()
        .scenarios
        .intervals
        .as_ref()
        .expect("joint simultaneous intervals");
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

    // A refresh on a different draw (another seed, a shifted outcome level) is recomputed, not
    // replayed: its scenario values equal a fresh study of the same table and differ from the
    // first run's, and both still bracket the truth.
    let refreshed_data = unknown_data(4_000, 24, 0.7);
    let refreshed = rebind_estimate(&mut prepared, refreshed_data.clone(), &ctx).unwrap();
    assert!(refreshed.estimate().as_effect().unwrap().ate.is_nan());
    let refreshed_values =
        refreshed.estimate().as_effect().unwrap().scenarios.effects.as_ref().unwrap();
    assert_eq!(refreshed_values.len(), 2);
    let refreshed_covariance =
        refreshed.estimate().as_effect().unwrap().joint.covariance.as_ref().unwrap();
    let fresh = Study::tabular(refreshed_data.clone())
        .tiered_background(background(&refreshed_data))
        .unwrap()
        .query(query(&refreshed_data))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap()
        .estimate(&refreshed_data, &ctx)
        .unwrap();
    let fresh_values = fresh.estimate().as_effect().unwrap().scenarios.effects.as_ref().unwrap();
    for i in 0..2 {
        assert_eq!(
            refreshed_values[i], fresh_values[i],
            "scenario {i}: refresh equals a fresh run"
        );
        assert_ne!(refreshed_values[i], values[i], "scenario {i}: another draw moves the value");
        assert!((refreshed_values[i] - truths[i]).abs() < 4.0 * refreshed_covariance.se(i));
    }

    let artifact =
        prepared.encode_contracted_result(&refreshed, "unknown-tiered-average", &ctx).unwrap();
    let consumed = antecedent_io::consume_analysis_result(&artifact).unwrap();
    assert!(consumed.acceptance.unresolved.iter().any(|reason| {
        reason.as_ref() == "dependencies.checked_unknown_tiered_average_operation"
    }));
    assert!(!consumed.acceptance.accepts_as_verified_program());
}

/// Bind `data` on a new branch, materialize and estimate it, then adopt the branch as the handle.
/// A failure leaves the handle unchanged, as the immutable lifecycle does.
fn rebind_estimate(
    prepared: &mut antecedent::PreparedStudy,
    data: antecedent_data::TabularData,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<antecedent::StudyResult, antecedent::CausalError> {
    let branch = prepared.rebind_data(data, ctx)?.materialize(ctx)?;
    let result = branch.estimate_retained(ctx)?;
    *prepared = branch;
    Ok(result)
}
