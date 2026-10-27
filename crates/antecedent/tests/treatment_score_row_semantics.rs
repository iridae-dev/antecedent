//! Nuisance likelihood rows follow the study cohort; prediction rows retain physical positions.
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(clippy::float_cmp, reason = "identical selected designs must produce identical fits")]

use antecedent::analysis::array_fits::{ArrayFitRequestWire, TreatmentScoreSpec};
use antecedent::prelude::TabularData;
use antecedent::{AverageEffectQuery, Dag, RefuteSuite, Study};
use antecedent_core::ExecutionContext;
use antecedent_data::{MissingDeclaration, MissingPolicy, TableView, ValidityBitmap};

fn fixture() -> TabularData {
    let z: Vec<_> = (0..240).map(|i| f64::from(i >= 100)).collect();
    let t: Vec<_> = (0..240).map(|i| f64::from(if i < 100 { i < 30 } else { i < 160 })).collect();
    let y: Vec<_> = t.iter().zip(&z).map(|(&t, &z)| 2.0 * t + z).collect();
    TabularData::from_f64_columns([("t", t.as_slice()), ("y", &y), ("z", &z)]).unwrap()
}

fn builder(data: TabularData, policy: Option<MissingPolicy>) -> antecedent::StudyBuilder {
    let schema = data.schema();
    let graph = Dag::from_named_edges(schema, &[("z", "t"), ("z", "y"), ("t", "y")]).unwrap();
    let query =
        AverageEffectQuery::binary_ate(schema.id_of("t").unwrap(), schema.id_of("y").unwrap());
    let mut builder = Study::tabular(data).graph(graph).query(query).refute(RefuteSuite::None);
    if let Some(policy) = policy {
        builder = builder.missing(MissingDeclaration::new(policy));
    }
    builder
}

fn prepare(data: TabularData, policy: Option<MissingPolicy>) -> antecedent::PreparedStudy {
    builder(data, policy)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ExecutionContext::for_tests(3))
        .unwrap()
}

fn spec(rows: Vec<usize>) -> TreatmentScoreSpec {
    TreatmentScoreSpec { prior_sd: 2.0, draws: 40, seed: 13, clip: 0.35, summary_rows: rows }
}

fn filtered(data: &TabularData, rows: &[usize]) -> TabularData {
    let columns: Vec<_> = ["t", "y", "z"]
        .into_iter()
        .map(|name| {
            let values = data.float64_cow(data.schema().id_of(name).unwrap()).unwrap();
            (name, rows.iter().map(|&row| values[row]).collect::<Vec<_>>())
        })
        .collect();
    TabularData::from_f64_columns(columns.iter().map(|(name, values)| (*name, values.as_slice())))
        .unwrap()
}

#[test]
fn masked_likelihood_matches_physical_filter_but_summary_rows_stay_physical() {
    let data = fixture();
    let masked = data
        .with_analysis_mask(ValidityBitmap::from_bools(
            &(0..240).map(|i| i < 200).collect::<Vec<_>>(),
        ))
        .unwrap();
    let selected = filtered(&data, &(0..200).collect::<Vec<_>>());
    let actual =
        prepare(masked, None).treatment_score_posterior_retained(&spec(vec![0, 150, 239])).unwrap();
    let reference = prepare(selected, None)
        .treatment_score_posterior_retained(&spec(vec![0, 150, 150]))
        .unwrap();
    assert_eq!(actual.request(), reference.request());
    assert_eq!(actual.result(), reference.result());
    // Independent two-parameter Newton solve: solve
    // 30-100p0 + 60-100p1 - b0/4 = 0 and 60-100p1 - b1/4 = 0,
    // p0=logistic(b0), p1=logistic(b0+b1). It gives b0=-0.8231390550060228,
    // b1=1.2159538315986123, p0=.3050977322165116, p1=.5969601154210035.
    assert!((actual.value("row0.at_mode").unwrap() - 0.305_098).abs() < 1e-4);
    assert!((actual.value("row2.at_mode").unwrap() - 0.596_960).abs() < 1e-4);
}

#[test]
fn missing_outcome_selects_the_fit_cohort_without_becoming_a_predictor() {
    let data = fixture();
    let outcome = data.schema().id_of("y").unwrap();
    let mut values = data.float64_values(outcome).unwrap();
    values[10] = f64::NAN;
    let missing = data.with_replaced_float(outcome, values.into()).unwrap();
    let selected = filtered(&data, &(0..240).filter(|&i| i != 10).collect::<Vec<_>>());
    let actual = prepare(missing, Some(MissingPolicy::CompleteCase))
        .treatment_score_posterior_retained(&spec(vec![10]))
        .unwrap();
    let reference =
        prepare(selected, None).treatment_score_posterior_retained(&spec(vec![0])).unwrap();
    assert_eq!(actual.request(), reference.request());
    let ArrayFitRequestWire::PropensityPosterior(request) = actual.request() else {
        panic!("expected propensity posterior");
    };
    assert_eq!(request.nrows, 239);
    assert_eq!(request.x_colmajor.len(), 478);
}

#[test]
fn error_policy_and_missing_summary_predictors_are_not_silently_dropped() {
    let data = fixture();
    let predictor = data.schema().id_of("z").unwrap();
    let mut values = data.float64_values(predictor).unwrap();
    values[10] = f64::NAN;
    let missing = data.with_replaced_float(predictor, values.into()).unwrap();
    let error = prepare(missing.clone(), Some(MissingPolicy::ErrorOnMissing))
        .treatment_score_posterior_retained(&spec(vec![0]))
        .unwrap_err();
    assert!(error.to_string().contains("missing_policy_required"));
    let error = prepare(missing, Some(MissingPolicy::CompleteCase))
        .treatment_score_posterior_retained(&spec(vec![10]))
        .unwrap_err();
    assert!(error.to_string().contains("summary row"));
}

#[test]
fn nonunit_storage_weights_refuse_instead_of_changing_population_silently() {
    let weighted = fixture().with_weights(vec![2.0; 240].into()).unwrap();
    let error =
        prepare(weighted, None).treatment_score_posterior_retained(&spec(vec![0])).unwrap_err();
    assert!(error.to_string().contains("sampling"));
}

#[test]
fn declared_sampling_weights_refuse_even_when_the_storage_is_unweighted() {
    use antecedent_estimate::{LinearAdjustmentAte, SamplingWeights};
    let estimator = LinearAdjustmentAte::new()
        .with_se_kind(antecedent_estimate::AnalyticSeKind::Hc3)
        .with_sampling_weights(SamplingWeights::new(vec![1.0; 240]).unwrap());
    let prepared = builder(fixture(), None)
        .estimator(estimator)
        .build()
        .unwrap()
        .prepare(&ExecutionContext::for_tests(3))
        .unwrap();
    let error = prepared.treatment_score_posterior_retained(&spec(vec![0])).unwrap_err();
    assert!(error.to_string().contains("sampling"));
}
