//! The treatment-propensity nuisance product of a prepared study, and the refusal of a
//! weighting-effect posterior, through the public flow.
//!
//! The fixture is the two-cell propensity: 100 rows at `z = 0` with 30 treated and 100 rows at
//! `z = 1` with 60 treated. The references are the Newton mode of the logistic model under the
//! stated Gaussian prior and the score moments of a large Laplace sample, derived outside the
//! estimator. The product is a posterior of a nuisance, never of an effect, and it is retained
//! as an artifact that a loader refits.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::float_cmp, reason = "exact sealed copies and exact zero band probabilities")]

#[path = "treatment_score_row_semantics.rs"]
mod row_semantics;

use antecedent::analysis::array_fits::{
    ArrayFit, ArrayFitArtifactWire, ArrayFitConsumeLimits, ArrayFitError, ArrayFitRequestWire,
    TreatmentScoreSpec,
};
use antecedent::prelude::{CausalSchemaBuilder, TabularData};
use antecedent::{
    AverageEffectQuery, BayesianConfig, Dag, EstimatorId, InferenceMode, RefuteSuite, Study,
};
use antecedent_core::ExecutionContext;

fn table() -> TabularData {
    let z: Vec<f64> = (0..200).map(|i| if i >= 100 { 1.0 } else { 0.0 }).collect();
    let t: Vec<f64> = (0..200)
        .map(|i| {
            let treated = if i >= 100 { i - 100 < 60 } else { i < 30 };
            if treated { 1.0 } else { 0.0 }
        })
        .collect();
    let y: Vec<f64> = (0..200)
        .map(|i| 1.0 + 2.0 * t[i] + 0.1 * f64::from(u32::try_from(i).unwrap() % 7))
        .collect();
    // Column order is the schema order of `builder`: treatment, outcome, then the context.
    TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())])
        .unwrap()
}

fn builder() -> antecedent::StudyBuilder {
    let schema = CausalSchemaBuilder::new()
        .continuous("t")
        .treatment()
        .continuous("y")
        .outcome()
        .continuous("z")
        .context()
        .build()
        .unwrap();
    let dag = Dag::from_named_edges(&schema, &[("z", "t"), ("z", "y"), ("t", "y")]).unwrap();
    let query =
        AverageEffectQuery::binary_ate(schema.id_of("t").unwrap(), schema.id_of("y").unwrap());
    Study::tabular(table())
        .graph(dag)
        .query(query)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
}

fn spec(prior_sd: f64) -> TreatmentScoreSpec {
    TreatmentScoreSpec { prior_sd, draws: 4000, seed: 13, clip: 0.35, summary_rows: vec![0, 150] }
}

#[test]
fn the_product_matches_newton_modes_and_laplace_score_moments() {
    let ctx = ExecutionContext::for_tests(3);
    let prepared = builder().build().unwrap().prepare(&ctx).unwrap();
    let product = prepared.treatment_score_posterior(&table(), &spec(2.0)).unwrap();
    // Rows 0 and 150 are the two cells; Newton modes 0.305098 and 0.596960, score means
    // 0.306897 and 0.596070, sds 0.045356 and 0.048366, outside-band 0.8296 and 0.1330.
    let value = |name: &str| product.value(name).unwrap();
    assert!(
        (value("row0.at_mode") - 0.305_098).abs() < 1e-4,
        "{} {} {:?} {:?} {:?}",
        value("row0.at_mode"),
        value("row1.at_mode"),
        product.value("mode.0"),
        product.value("mode.1"),
        product.value("mode.2")
    );
    assert!((value("row1.at_mode") - 0.596_960).abs() < 1e-4);
    assert!((value("row0.mean") - 0.306_897).abs() < 0.004);
    assert!((value("row1.mean") - 0.596_070).abs() < 0.004);
    assert!((value("row0.sd") / 0.045_356 - 1.0).abs() < 0.06);
    assert!((value("row1.sd") / 0.048_366 - 1.0).abs() < 0.06);
    assert!((value("row0.outside_band") - 0.8296).abs() < 0.03);
    assert!((value("row1.outside_band") - 0.1330).abs() < 0.03);
    // A nuisance product: no effect draws, and its own labels say so.
    assert!(product.draws().is_empty());
    assert!(product.result().uncertainty.contains("not_an_effect_posterior"));
    assert_eq!(product.result().licensing, "unlicensed");
    assert_eq!(product.result().calibration, "unmeasured");
    // The retained request is the design of the prepared study: intercept then z.
    let ArrayFitRequestWire::PropensityPosterior(request) = product.request() else {
        panic!("a propensity request");
    };
    assert_eq!(request.nrows, 200);
    assert_eq!(request.x_colmajor.len(), 400);
    assert_eq!(request.prior_sd, 2.0);
}

#[test]
fn the_product_is_retained_reloaded_by_refitting_and_refused_when_forged() {
    let ctx = ExecutionContext::for_tests(3);
    let prepared = builder().build().unwrap().prepare(&ctx).unwrap();
    let product = prepared.treatment_score_posterior(&table(), &spec(2.0)).unwrap();
    let bytes = product.export().unwrap();
    let limits = ArrayFitConsumeLimits::default();
    assert_eq!(ArrayFit::consume(&bytes, limits).unwrap(), product);

    let mut wire = ArrayFitArtifactWire::decode(&bytes).unwrap();
    wire.result.values[0].value += 0.01;
    assert_eq!(
        ArrayFit::consume(&wire.export().unwrap(), limits).unwrap_err(),
        ArrayFitError::ResultMismatch
    );
    let mut wire = ArrayFitArtifactWire::decode(&bytes).unwrap();
    if let ArrayFitRequestWire::PropensityPosterior(request) = &mut wire.request {
        request.treatment[0] = 1.0 - request.treatment[0];
    }
    wire.request_digest = wire.expected_request_digest().unwrap();
    assert_eq!(
        ArrayFit::consume(&wire.export().unwrap(), limits).unwrap_err(),
        ArrayFitError::ResultMismatch
    );
}

#[test]
fn a_summary_row_outside_the_data_is_refused() {
    let ctx = ExecutionContext::for_tests(3);
    let prepared = builder().build().unwrap().prepare(&ctx).unwrap();
    let mut bad = spec(2.0);
    bad.summary_rows = vec![200];
    let error = prepared.treatment_score_posterior(&table(), &bad).unwrap_err();
    assert!(
        error.to_string().contains("a summary row is outside the prepared data"),
        "{error}"
    );
    // The last valid row (199) is accepted: the bound is on the row count, not off by one.
    let mut edge = spec(2.0);
    edge.summary_rows = vec![199];
    assert!(prepared.treatment_score_posterior(&table(), &edge).is_ok());
}

#[test]
fn a_bayesian_weighting_estimator_is_the_typed_boundary_with_its_next_attempt() {
    let error = builder()
        .estimator(EstimatorId::PropensityWeighting)
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(2000)))
        .build()
        .unwrap_err();
    let text = error.to_string();
    assert!(text.contains("route_not_supported"), "{text}");
    assert!(text.contains("penalized_propensity.effect_posterior"), "{text}");
    assert!(text.contains("ancillary"), "{text}");
    assert!(text.contains("Next attempt"), "{text}");
    assert!(text.contains("Bayesian g-computation"), "{text}");
}
