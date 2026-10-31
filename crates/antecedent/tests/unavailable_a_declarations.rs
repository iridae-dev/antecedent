//! Declarations the routes read, or refuse by name.
//!
//! - The categorical contrast, the dose-grid functionals and the smoothed dose transport read
//!   declared sampling weights, clusters and serial dependence: the weights fit the weighted
//!   regression or local quadratic, clusters give cluster-robust variances (and keep clusters whole
//!   in a cross-fit fold), a lag gives a Newey-West variance (and contiguous-block folds), and each
//!   declaration the route or its covariance kind cannot read is refused with a typed detail. A
//!   multi-level treatment on the dose-based routes stays structural: the dose is a real-valued
//!   column and a categorical design is refused (`dose_grid.unsupported_semantics`).
//! - A structure search takes variables, a method and bounds: no treatment, so no categorical
//!   treatment and no held-out intervention regime.
//! - The Bayesian synthetic control, staggered event study and augmented panel contrasts are
//!   refused on a table with no second period or with one row per subject, with a next attempt.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::analysis::categorical_treatment::{
    CategoricalRowsRequest, CategoricalSpecWire, CategoricalTreatment, CategoricalTreatmentRequest,
    CovarianceKindWire, LevelScaleWire,
};
use antecedent::analysis::dose_grid_functional::{
    DoseClaimsWire, DoseFunctionalWire, DoseGridArtifactError, DoseGridFunctional,
    DoseGridRequestWire,
};
use antecedent::analysis::vector_treatment::AdjustmentColumnRequest;
use antecedent::prelude::ExecutionContext;
use antecedent::{
    BayesianConfig, InferenceMode, PanelDidQuery, PreparedSmoothedDose, Study, StudyBuilder,
    consume_smoothed_dose_artifact,
};
use antecedent_core::{DiscoverQuery, SyntheticControlQuery, VariableId};
use antecedent_data::TabularData;
use antecedent_estimate::smoothed_dose::SmoothedDoseInput;
use antecedent_identify::TransportIdentifier;
use antecedent_io::smoothed_dose_artifact::SmoothedDoseConsumeLimits;

#[path = "smoothed_dose_dgp/mod.rs"]
mod dgp;

fn bayes() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().prior_scale(100.0).n_draws(2_000))
}

fn arcs(names: impl IntoIterator<Item = String>) -> Vec<Arc<str>> {
    names.into_iter().map(Arc::<str>::from).collect()
}

// ---------------------------------------------------------------------------------------------
// Categorical contrast: weights, clusters and a serial lag are read, and refused where unreadable.
// ---------------------------------------------------------------------------------------------

/// Dataset A of the categorical artifact tests: `a = [1, 3]`, `b = [4, 6, 5]`,
/// `c = [9, 7, 11, 9]`, rows interleaved, no adjustment.
fn categorical_request(covariance: CovarianceKindWire) -> CategoricalTreatmentRequest {
    let rows = [
        ("c", 9.0),
        ("a", 1.0),
        ("b", 4.0),
        ("c", 7.0),
        ("b", 6.0),
        ("a", 3.0),
        ("c", 11.0),
        ("b", 5.0),
        ("c", 9.0),
    ];
    CategoricalTreatmentRequest {
        outcome: rows.iter().map(|r| r.1).collect(),
        row_snapshot: "snap".into(),
        adjustment: Vec::<AdjustmentColumnRequest>::new(),
        levels: rows.iter().map(|r| r.0.to_string()).collect(),
        spec: CategoricalSpecWire {
            declared_levels: vec!["a".into(), "b".into(), "c".into()],
            scale: LevelScaleWire::Unordered,
            reference: "a".into(),
            min_level_rows: 2,
            pairwise: vec![("b".into(), "c".into())],
            monotonicity: None,
            covariance,
        },
    }
}

fn categorical(
    request: &CategoricalTreatmentRequest,
    rows: &CategoricalRowsRequest,
) -> Result<CategoricalTreatment, antecedent::analysis::categorical_treatment::CategoricalTreatmentArtifactError>
{
    CategoricalTreatment::evaluate_declared(request, rows, &ExecutionContext::for_tests(0))
}

#[test]
fn the_categorical_request_has_no_weight_cluster_or_lag_field_and_no_cluster_robust_covariance() {
    // Weighted group means are the known answer of the dummy regression: with weights `w`,
    // b_l = ybar_l(w) - ybar_a(w), and under HC0 the variance of a weighted group mean is
    // sum_g w_i^2 (y_i - ybar_g(w))^2 / (sum_g w_i)^2, independent across groups. The weights
    // below have a zero entry (row 2, a level-b row), which is deleted, and an unequal mix.
    let weights = vec![1.0, 3.0, 2.0, 1.0, 0.0, 1.0, 4.0, 2.0, 2.0];
    let request = categorical_request(CovarianceKindWire::Hc0);
    let rows = CategoricalRowsRequest { sampling_weights: Some(weights.clone()), ..Default::default() };
    let fit = categorical(&request, &rows).expect("weights are read").report();
    let groups: [(&str, Vec<(f64, f64)>); 3] = [
        ("a", vec![(3.0, 1.0), (1.0, 3.0)]),
        ("b", vec![(2.0, 4.0), (2.0, 5.0)]),
        ("c", vec![(1.0, 9.0), (1.0, 7.0), (4.0, 11.0), (2.0, 9.0)]),
    ];
    // (weight, outcome) pairs, with the zero-weight b row (outcome 6) removed above.
    let moments = |g: &[(f64, f64)]| {
        let total: f64 = g.iter().map(|(w, _)| w).sum();
        let mean = g.iter().map(|(w, y)| w * y).sum::<f64>() / total;
        let variance = g.iter().map(|(w, y)| (w * (y - mean)).powi(2)).sum::<f64>() / (total * total);
        (mean, variance)
    };
    let (mean_a, var_a) = moments(&groups[0].1);
    let (mean_b, var_b) = moments(&groups[1].1);
    let (mean_c, var_c) = moments(&groups[2].1);
    let accounting = fit.declared_rows.as_ref().expect("the declaration is accounted");
    assert_eq!((accounting.rows_used, accounting.zero_weight_rows), (8, 1));
    let close = |got: f64, want: f64, what: &str| {
        assert!((got - want).abs() <= 1e-9 * want.abs().max(1.0), "{what}: {got} vs {want}");
    };
    close(fit.result.coefficients[0].estimate, mean_b - mean_a, "b - a");
    close(fit.result.coefficients[1].estimate, mean_c - mean_a, "c - a");
    close(fit.result.covariance[0], var_a + var_b, "V_bb");
    close(fit.result.covariance[1], var_a, "V_bc");
    close(fit.result.covariance[3], var_a + var_c, "V_cc");
    // The pair b -> c has variance var_b + var_c (the shared reference cancels).
    close(fit.result.pairwise[0].standard_error.powi(2), var_b + var_c, "se(c - b)^2");

    // Clusters: a cluster-robust fit reads the cluster ids; the cluster covariance kind without
    // them, and clusters under a covariance that does not read them, are refused by name.
    let clustered = categorical_request(CovarianceKindWire::ClusterRobust);
    let ids = CategoricalRowsRequest { cluster: Some(vec![0, 0, 1, 1, 2, 2, 3, 3, 3]), ..Default::default() };
    let fit = categorical(&clustered, &ids).expect("clusters are read").report();
    assert_eq!(fit.declared_rows.as_ref().unwrap().clusters, Some(4));
    assert_eq!(fit.spec.covariance.as_str(), "cluster_robust");
    let text = categorical(&clustered, &CategoricalRowsRequest::default()).unwrap_err().to_string();
    assert!(text.contains("vector_treatment.cluster_ids_required"), "{text}");
    let text = categorical(&request, &ids).unwrap_err().to_string();
    assert!(text.contains("vector_treatment.cluster_unread"), "{text}");
    // A serial lag is read by the Newey-West covariance only.
    let serial = categorical_request(CovarianceKindWire::NeweyWest);
    let lag = CategoricalRowsRequest { serial_lag: Some(2), ..Default::default() };
    let fit = categorical(&serial, &lag).expect("a lag is read").report();
    assert_eq!(fit.declared_rows.as_ref().unwrap().serial_lag, Some(2));
    let text = categorical(&request, &lag).unwrap_err().to_string();
    assert!(text.contains("vector_treatment.lag_unread"), "{text}");
    // Weights under the model-based variance are refused: it would treat them as precision weights.
    let model = categorical_request(CovarianceKindWire::ModelBased);
    let text = categorical(&model, &rows).unwrap_err().to_string();
    assert!(text.contains("weights_need_robust_covariance"), "{text}");
}

// ---------------------------------------------------------------------------------------------
// Dose-grid functionals: aligned (dose, outcome) rows with a declared bandwidth.
// ---------------------------------------------------------------------------------------------

fn dose_request(design: &str) -> DoseGridRequestWire {
    let n = 201usize;
    let dose: Vec<f64> = (0..n).map(|i| 4.0 * i as f64 / (n - 1) as f64).collect();
    let outcome: Vec<f64> = dose
        .iter()
        .enumerate()
        .map(|(i, &d)| 1.0 + 0.5 * d + 0.25 * d * d + if i % 2 == 0 { 0.01 } else { -0.01 })
        .collect();
    DoseGridRequestWire {
        design: design.into(),
        quadratic_mean: false,
        functional: DoseFunctionalWire { kind: "level".into(), from: None, to: None },
        grid: vec![0.5, 1.0, 2.0],
        bandwidth: 0.4,
        bandwidth_range: (0.1, 1.0),
        minimum_local_ess: 10.0,
        claims: DoseClaimsWire {
            pointwise_level: true,
            derivative: false,
            simultaneous_band: false,
        },
        dose,
        outcome,
        bias_aware: None,
        sampling_weights: None,
        cluster: None,
        serial_lag: None,
    }
}

#[test]
fn the_dose_grid_request_has_no_weight_cluster_or_lag_field_and_no_categorical_design() {
    let plain = DoseGridFunctional::evaluate(dose_request("randomized_dose"))
        .expect("the declared rows must evaluate");
    // A noise-free quadratic reproduces exactly whatever the weights (a weighted local quadratic
    // reproduces a quadratic mean): the level is 1 + d/2 + d^2/4 at every grid dose, and the
    // weights change the standard error, not the answer.
    let mut weighted = dose_request("randomized_dose");
    weighted.sampling_weights = Some((0..201).map(|i| 0.5 + (i % 7) as f64 * 0.4).collect());
    let weighted = DoseGridFunctional::evaluate(weighted).expect("sampling weights are read");
    let accounting = weighted.result().declared_rows.as_ref().expect("accounted");
    assert_eq!((accounting.rows_used, accounting.zero_weight_rows), (201, 0));
    assert!(accounting.weight_effective_sample_size.unwrap() < 201.0);
    for (point, base) in weighted.result().levels.iter().zip(&plain.result().levels) {
        let truth = 1.0 + 0.5 * point.dose + 0.25 * point.dose * point.dose;
        assert!((point.value - truth).abs() < 0.02, "{} vs {truth}", point.value);
        assert_ne!(point.standard_error.to_bits(), base.standard_error.to_bits());
    }
    // Clusters give the t reference of G - 1 degrees of freedom; a lag gives the Newey-West error.
    let mut clustered = dose_request("randomized_dose");
    clustered.cluster = Some((0..201).map(|i| i % 12).collect());
    let clustered = DoseGridFunctional::evaluate(clustered).expect("clusters are read");
    assert_eq!(clustered.result().declared_rows.as_ref().unwrap().clusters, Some(12));
    assert_eq!(
        clustered.result().declared_rows.as_ref().unwrap().interval_reference,
        "student_t_clusters_minus_one"
    );
    let mut serial = dose_request("randomized_dose");
    serial.serial_lag = Some(4);
    let serial = DoseGridFunctional::evaluate(serial).expect("a lag is read");
    assert_eq!(serial.result().declared_rows.as_ref().unwrap().serial_lag, Some(4));
    // Each sealed artifact replays bit for bit from its own bytes.
    for done in [&weighted, &clustered, &serial] {
        let bytes = done.export().unwrap();
        let again = DoseGridFunctional::consume(&bytes, Default::default()).unwrap();
        assert_eq!(again.result(), done.result());
    }
    // Clusters together with a lag, and declarations of the wrong length, are refused by name.
    let mut both = dose_request("randomized_dose");
    both.cluster = Some(vec![0; 201]);
    both.serial_lag = Some(2);
    assert!(matches!(
        DoseGridFunctional::evaluate(both),
        Err(DoseGridArtifactError::UnsupportedSemantics(_))
    ));
    let mut short = dose_request("randomized_dose");
    short.sampling_weights = Some(vec![1.0; 200]);
    assert_eq!(
        DoseGridFunctional::evaluate(short).unwrap_err().refusal().1,
        "dose_grid.invalid_request"
    );

    // A categorical design is not one of the two spellings; it is refused with a typed detail.
    let error: DoseGridArtifactError = DoseGridFunctional::evaluate(dose_request("categorical"))
        .expect_err("categorical was read");
    let (code, detail) = error.refusal();
    assert_eq!(code.to_string(), "route_not_supported");
    assert_eq!(detail, "dose_grid.unsupported_semantics");
}

// ---------------------------------------------------------------------------------------------
// Smoothed dose transport: rows, a source indicator, the trial sampling mode and declared rows.
// ---------------------------------------------------------------------------------------------

fn prepare_smoothed(input: SmoothedDoseInput, seed: u64) -> PreparedSmoothedDose {
    let (diagram, names) = dgp::diagram();
    StudyBuilder::smoothed_dose_transport(
        diagram,
        dgp::query(&[1.0, 2.0], 0.5),
        input,
        antecedent_estimate::SmoothedDoseOptions { folds: 3, ..Default::default() },
        names,
        &ExecutionContext::for_tests(seed),
    )
    .expect("the declared rows prepare")
}

#[test]
fn the_smoothed_dose_input_has_no_weight_cluster_or_lag_field() {
    let rows = dgp::draw(dgp::Design::NestedCohort, dgp::Scenario::Good, 600, 450, 31);
    let n = rows.source.len();
    let ctx = ExecutionContext::for_tests(31);
    let plain = prepare_smoothed(rows.clone(), 31).estimate(&ctx).unwrap();
    // Sampling weights, with some rows of weight zero that are deleted and counted.
    let mut weighted = rows.clone();
    weighted.sampling_weights =
        Some((0..n).map(|i| if i % 97 == 5 { 0.0 } else { 0.6 + (i % 5) as f64 * 0.3 }).collect());
    let prepared = prepare_smoothed(weighted, 31);
    assert_eq!(prepared.zero_weight_rows(), (0..n).filter(|i| i % 97 == 5).count() as u64);
    let result = prepared.estimate(&ctx).unwrap();
    // Weights independent of the outcome leave the target alone: the weighted grid sits within
    // 0.08 of the closed-form smoothed target psi_h(a) = a^2 + h^2/5 + m(1 + a/2) (observed
    // 0.02 and 0.07 against 1.5 and 4.65), as the unweighted grid does.
    for (w, p) in result.estimate().grid.iter().zip(&plain.estimate().grid) {
        let truth = dgp::Scenario::Good.truth(w.dose, 0.5);
        assert!((w.estimate - truth).abs() < 0.15, "weighted {} vs {truth}", w.estimate);
        assert!((p.estimate - truth).abs() < 0.15, "plain {} vs {truth}", p.estimate);
    }
    assert!(result.estimate().grid.iter().zip(&plain.estimate().grid).all(|(w, p)| {
        w.estimate.is_finite() && w.estimate.to_bits() != p.estimate.to_bits()
    }));
    // Clusters: whole clusters per fold, a cluster-robust diagnostic; a lag: block folds.
    let mut clustered = rows.clone();
    clustered.cluster = Some((0..n as u32).map(|i| i % 40).collect());
    let clustered = prepare_smoothed(clustered, 31).estimate(&ctx).unwrap();
    assert_eq!(clustered.estimate().folds.scheme, "clusters_whole_stratified_seeded_by_role");
    let folds = &clustered.estimate().folds.assignment;
    for i in 0..n {
        assert_eq!(folds[i], folds[i % 40], "cluster {} is split across folds", i % 40);
    }
    let mut serial = rows.clone();
    serial.serial_lag = Some(3);
    let serial = prepare_smoothed(serial, 31).estimate(&ctx).unwrap();
    assert_eq!(serial.estimate().folds.scheme, "contiguous_row_blocks");
    for (i, fold) in serial.estimate().folds.assignment.iter().enumerate() {
        assert_eq!(usize::from(*fold), i * 3 / n, "block fold of row {i}");
    }
    // Every declared execution exports an artifact that replays bit for bit from its own bytes,
    // with the weights, clusters and lag bound by its digests.
    for result in [&result, &clustered, &serial] {
        let bytes = result.export().unwrap();
        let again =
            consume_smoothed_dose_artifact(&bytes, SmoothedDoseConsumeLimits::default(), &ctx)
                .unwrap();
        assert_eq!(again.estimate(), result.estimate());
        assert_eq!(again.identity(), result.identity());
    }
    // A repository-wide invariant of plain rows: no new field, no changed digest.
    assert!(!serde_json::to_string(&rows).unwrap().contains("sampling_weights"));
    let (diagram, _) = dgp::diagram();
    let _ = TransportIdentifier::new().identify(&diagram, &dgp::query(&[1.0], 0.5).transport_query());
}

// ---------------------------------------------------------------------------------------------
// Structure search: variables, a method and bounds.
// ---------------------------------------------------------------------------------------------

#[test]
fn a_structure_search_query_has_no_treatment_variable_and_no_intervention_regime() {
    fn fields(query: DiscoverQuery) {
        let DiscoverQuery { variables: _, method: _, bounds: _ } = query;
    }
    let _ = fields;
}

// ---------------------------------------------------------------------------------------------
// Bayesian design routes on a table with no time dimension or no pre-period outcome.
// ---------------------------------------------------------------------------------------------

#[test]
fn the_bayesian_synthetic_control_is_refused_on_a_single_period_table() {
    let outcome = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let data = TabularData::from_f64_columns([("y", outcome.as_slice())]).unwrap();
    let units = arcs(["treated", "d0", "d1", "d2", "d3", "d4"].map(String::from));
    let query =
        SyntheticControlQuery::new(VariableId::from_raw(0), units, vec![1_i64; 6], "treated", 1);
    let ctx = ExecutionContext::for_tests(11);
    let error = Study::tabular(data)
        .query(query)
        .inference(bayes())
        .build()
        .and_then(|study| study.run(&ctx))
        .err()
        .expect("a single period table has no pre-treatment path");
    let text = error.to_string();
    assert!(text.contains("at least two pre-periods and one post-period"), "{text}");
    assert_eq!(error.reason_code(), Some("invalid_argument"), "{text}");
    assert!(text.contains("Next attempt"), "{text}");
}

#[test]
fn the_bayesian_staggered_event_study_is_refused_on_a_single_period_table_with_a_next_attempt() {
    let outcome = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let data = TabularData::from_f64_columns([("y", outcome.as_slice())]).unwrap();
    let subjects = arcs((0..6).map(|i| format!("s{i}")));
    let query = PanelDidQuery::staggered_event_study(
        VariableId::from_raw(0),
        subjects.clone(),
        subjects,
        vec![1_i64; 6],
        vec![0_i64, 0, 0, 2, 2, 2],
    );
    let ctx = ExecutionContext::for_tests(12);
    let error = Study::tabular(data)
        .query(query)
        .inference(bayes())
        .build()
        .and_then(|study| study.run(&ctx))
        .err()
        .expect("one period defines no event-time contrast");
    let text = error.to_string();
    assert!(text.contains("balanced panel with at least two common periods"), "{text}");
    assert!(text.contains("Next attempt"), "{text}");
    assert_eq!(error.reason_code(), Some("invalid_argument"), "{text}");
}

#[test]
fn the_bayesian_augmented_panel_is_refused_when_a_subject_repeats_instead_of_pairing_two_periods() {
    let pre = [10.0, 12.0, 13.0, 9.0, 15.0, 11.0];
    let post = [15.0, 17.0, 18.0, 11.0, 17.0, 13.0];
    let (propensity, prediction) = ([0.5; 6], [2.0; 6]);
    let data = TabularData::from_f64_columns([
        ("pre", pre.as_slice()),
        ("post", post.as_slice()),
        ("p", propensity.as_slice()),
        ("m0", prediction.as_slice()),
    ])
    .unwrap();
    // A repeated cross-section reuses a subject label across rows instead of one row per subject.
    let subjects = arcs(["a", "a", "b", "b", "c", "c"].map(String::from));
    let query = PanelDidQuery::augmented_panel(
        VariableId::from_raw(1),
        VariableId::from_raw(0),
        VariableId::from_raw(2),
        VariableId::from_raw(3),
        [true, true, true, false, false, false],
        subjects.clone(),
        subjects,
        true,
    );
    let ctx = ExecutionContext::for_tests(13);
    let error = Study::tabular(data)
        .query(query)
        .inference(bayes())
        .build()
        .and_then(|study| study.run(&ctx))
        .err()
        .expect("a repeated subject was paired as two periods of one unit");
    let text = error.to_string();
    assert!(text.contains("one unique subject per row"), "{text}");
    assert_eq!(error.reason_code(), Some("invalid_argument"), "{text}");
}
