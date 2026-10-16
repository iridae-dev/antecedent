//! Posterior means of the joint, categorical, and local-quadratic coefficient fits.
//! A wide prior stays near the frequentist pin. A mapped prior refuses. The
//! consumer replays the mean from normal equations and does not see the rows.

use antecedent_estimate::EstimationError;
use antecedent_estimate::categorical_treatment::{
    CategoricalTreatmentInput, CategoricalTreatmentSpec, LevelScale,
};
use antecedent_estimate::posterior_mean::{
    LocalQuadraticNamespace, PosteriorRequest, categorical_treatment_posterior,
    consume_coefficient_posterior, consume_local_quadratic_posterior, local_quadratic_posterior,
    vector_treatment_posterior,
};
use antecedent_estimate::vector_treatment::{
    DesignSummary, TreatmentColumn, VectorTreatmentInput, summarize_vector_design,
};

const SCALE: f64 = 1_000.0;

fn prior() -> PosteriorRequest {
    PosteriorRequest::IndependentNormal { scale: SCALE }
}

fn refusal(error: EstimationError) -> String {
    let EstimationError::Refused { code, message } = error else {
        panic!("expected a coded refusal, got {error:?}");
    };
    format!("{code} {message}")
}

fn treatment(name: &str, values: &[f64]) -> TreatmentColumn {
    TreatmentColumn {
        name: name.into(),
        values: values.to_vec(),
        adjustment_set: vec![],
        row_snapshot: "snap".into(),
    }
}

fn vector_input() -> VectorTreatmentInput {
    VectorTreatmentInput {
        outcome: vec![3.0, 2.0, 5.0, 1.0, 0.0, 4.0],
        row_snapshot: "snap".into(),
        adjustment: vec![],
        treatments: vec![
            treatment("t1", &[1.0, 0.0, 1.0, 0.0, 0.0, 1.0]),
            treatment("t2", &[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
        ],
    }
}

fn hand_summary() -> DesignSummary {
    DesignSummary {
        n_rows: 6,
        first_treatment: 1,
        gram: vec![6.0, 3.0, 3.0, 3.0, 3.0, 2.0, 3.0, 2.0, 3.0],
        xty: vec![15.0, 12.0, 10.0],
        rss: 3.25,
    }
}

#[test]
fn vector_posterior_mean_stays_near_the_hand_solved_coefficients() {
    let mean = vector_treatment_posterior(&vector_input(), prior()).unwrap();
    assert!((mean[0] - 2.75).abs() < 1e-3, "{mean:?}");
    assert!((mean[1] - 0.75).abs() < 1e-3, "{mean:?}");
    let shrunk = vector_treatment_posterior(
        &vector_input(),
        PosteriorRequest::IndependentNormal { scale: 0.05 },
    )
    .unwrap();
    assert!(shrunk[0].abs() < mean[0].abs());
}

#[test]
fn vector_posterior_refuses_a_mapped_prior() {
    let message = refusal(
        vector_treatment_posterior(&vector_input(), PosteriorRequest::MappedCoefficient)
            .unwrap_err(),
    );
    assert!(message.contains("route_not_supported"), "{message}");
    assert!(message.contains("vector_posterior.mapped_prior"), "{message}");
}

#[test]
fn vector_posterior_consumer_replays_the_hand_normal_equations() {
    let produced = vector_treatment_posterior(&vector_input(), prior()).unwrap();
    let consumed = consume_coefficient_posterior(&hand_summary(), SCALE).unwrap();
    assert_eq!(produced.len(), consumed.len());
    for (left, right) in produced.iter().zip(&consumed) {
        assert!((left - right).abs() < 1e-9, "{produced:?} vs {consumed:?}");
    }
    let _ = summarize_vector_design(&vector_input()).unwrap();
}

fn categorical_input() -> CategoricalTreatmentInput {
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
    CategoricalTreatmentInput {
        outcome: rows.iter().map(|row| row.1).collect(),
        row_snapshot: "snap".into(),
        adjustment: vec![],
        levels: rows.iter().map(|row| row.0.to_string()).collect(),
    }
}

fn categorical_spec() -> CategoricalTreatmentSpec {
    CategoricalTreatmentSpec {
        declared_levels: ["a", "b", "c"].into_iter().map(str::to_string).collect(),
        scale: LevelScale::Unordered,
        reference: "a".into(),
        min_level_rows: 2,
        pairwise: vec![],
        monotonicity: None,
        covariance: antecedent_estimate::vector_treatment::VectorCovariance::ModelBased,
    }
}

#[test]
fn categorical_posterior_mean_stays_near_the_group_mean_differences() {
    let mean = categorical_treatment_posterior(&categorical_input(), &categorical_spec(), prior())
        .unwrap();
    assert!((mean[0] - 3.0).abs() < 1e-3, "{mean:?}");
    assert!((mean[1] - 7.0).abs() < 1e-3, "{mean:?}");
}

#[test]
fn categorical_posterior_refuses_an_external_power_prior() {
    let message = refusal(
        categorical_treatment_posterior(
            &categorical_input(),
            &categorical_spec(),
            PosteriorRequest::ExternalPower,
        )
        .unwrap_err(),
    );
    assert!(message.contains("route_not_supported"), "{message}");
    assert!(message.contains("categorical_posterior.external_power_prior"), "{message}");
}

#[test]
fn categorical_posterior_consumer_replays_the_dummy_summary() {
    let produced =
        categorical_treatment_posterior(&categorical_input(), &categorical_spec(), prior())
            .unwrap();
    let summary = antecedent_estimate::categorical_treatment::summarize_categorical_design(
        &categorical_input(),
        &categorical_spec(),
    )
    .unwrap();
    let consumed = consume_coefficient_posterior(&summary.design, SCALE).unwrap();
    for (left, right) in produced.iter().zip(&consumed) {
        assert!((left - right).abs() < 1e-9);
    }
}

fn line() -> (Vec<f64>, Vec<f64>) {
    let dose: Vec<f64> = (0..24_i32).map(f64::from).collect();
    let outcome = dose.iter().map(|d| 1.0 + 2.0 * d).collect();
    (dose, outcome)
}

#[test]
fn quadratic_posterior_mean_stays_near_the_linear_dose_response() {
    let (dose, outcome) = line();
    let (level, derivative) = local_quadratic_posterior(
        &dose,
        &outcome,
        10.0,
        3.0,
        prior(),
        LocalQuadraticNamespace::QuadraticDose,
    )
    .unwrap();
    assert!((level - 21.0).abs() < 0.05, "{level}");
    assert!((derivative - 2.0).abs() < 0.05, "{derivative}");
}

#[test]
fn quadratic_posterior_refuses_an_unmeasured_band() {
    let (dose, outcome) = line();
    let message = refusal(
        local_quadratic_posterior(
            &dose,
            &outcome,
            10.0,
            3.0,
            PosteriorRequest::UnmeasuredBand,
            LocalQuadraticNamespace::QuadraticDose,
        )
        .unwrap_err(),
    );
    assert!(message.contains("cell_not_licensed"), "{message}");
    assert!(message.contains("quadratic_posterior.unmeasured_band"), "{message}");
}

#[test]
fn quadratic_posterior_consumer_replays_the_local_normal_equations() {
    let (dose, outcome) = line();
    let (level, derivative) = local_quadratic_posterior(
        &dose,
        &outcome,
        10.0,
        3.0,
        prior(),
        LocalQuadraticNamespace::QuadraticDose,
    )
    .unwrap();
    let mut gram = [[0.0; 3]; 3];
    let mut rhs = [0.0; 3];
    let mut weight_sum = 0.0;
    let mut rss = 0.0;
    for (&x, &y) in dose.iter().zip(&outcome) {
        let dx = x - 10.0;
        let w = (-0.5 * (dx / 3.0).powi(2)).exp();
        let row = [1.0, dx, dx * dx];
        let fitted = 21.0 + 2.0 * dx;
        rss += w * (y - fitted).powi(2);
        for j in 0..3 {
            rhs[j] += w * row[j] * y;
            for k in 0..3 {
                gram[j][k] += w * row[j] * row[k];
            }
        }
        weight_sum += w;
    }
    let (consumed_level, consumed_derivative) =
        consume_local_quadratic_posterior(gram, rhs, weight_sum, rss, SCALE).unwrap();
    assert!((consumed_level - level).abs() < 1e-6, "{consumed_level} vs {level}");
    assert!((consumed_derivative - derivative).abs() < 1e-6);
}

#[test]
fn dose_grid_posterior_mean_stays_near_the_linear_dose_response() {
    let (dose, outcome) = line();
    let (level, derivative) = local_quadratic_posterior(
        &dose,
        &outcome,
        8.0,
        2.5,
        prior(),
        LocalQuadraticNamespace::DoseGrid,
    )
    .unwrap();
    assert!((level - 17.0).abs() < 0.05, "{level}");
    assert!((derivative - 2.0).abs() < 0.05, "{derivative}");
}

#[test]
fn dose_grid_posterior_refuses_a_mapped_prior() {
    let (dose, outcome) = line();
    let message = refusal(
        local_quadratic_posterior(
            &dose,
            &outcome,
            8.0,
            2.5,
            PosteriorRequest::MappedCoefficient,
            LocalQuadraticNamespace::DoseGrid,
        )
        .unwrap_err(),
    );
    assert!(message.contains("route_not_supported"), "{message}");
    assert!(message.contains("dose_grid_posterior.mapped_prior"), "{message}");
}

#[test]
fn dose_grid_posterior_consumer_replays_the_local_normal_equations() {
    let (dose, outcome) = line();
    let (level, derivative) = local_quadratic_posterior(
        &dose,
        &outcome,
        8.0,
        2.5,
        prior(),
        LocalQuadraticNamespace::DoseGrid,
    )
    .unwrap();
    let mut gram = [[0.0; 3]; 3];
    let mut rhs = [0.0; 3];
    let mut weight_sum = 0.0;
    for (&x, &y) in dose.iter().zip(&outcome) {
        let dx = x - 8.0;
        let w = (-0.5 * (dx / 2.5).powi(2)).exp();
        let row = [1.0, dx, dx * dx];
        for j in 0..3 {
            rhs[j] += w * row[j] * y;
            for k in 0..3 {
                gram[j][k] += w * row[j] * row[k];
            }
        }
        weight_sum += w;
    }
    let (consumed_level, consumed_derivative) =
        consume_local_quadratic_posterior(gram, rhs, weight_sum, 0.0, SCALE).unwrap();
    assert!((consumed_level - level).abs() < 1e-6);
    assert!((consumed_derivative - derivative).abs() < 1e-6);
}
