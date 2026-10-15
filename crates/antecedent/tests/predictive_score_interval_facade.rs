//! The interval of an expected predictive score through the facade: published inside a
//! measured coverage record and refused, with a typed reason, everywhere else.
//!
//! The measured designs (see `predictive_score_interval_calibration.rs`) are fixed
//! Gaussian forecasts against iid outcomes (400 to 1600 cases), clustered outcomes in
//! clusters of five (800 to 3200 cases) and a stationary AR(1) series (1600 to 6400
//! cases), at level 0.95, for the CRPS, the log score and the CRPS difference of two
//! forecasts. The serial CRPS mean interval was measured as a named boundary and is not
//! licensed.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "case counts and simulated count outcomes are small and non-negative"
)]

use antecedent::analysis::predictive_assessment::{
    DependenceDeclaration, FitDeclaration, ForecastLaw, Horizon, LawSource, ObservationKey,
    Outcome, PredictiveAssessmentSpec, PredictiveTarget, PredictorLaws,
    PreparedPredictiveAssessment, RealizedObservations, ScoreIntervalQuery, ScoreKind,
    ValidationDesign,
};
use antecedent_core::ExecutionContext;

const CLUSTER_SIZE: usize = 5;

fn key(i: u64) -> ObservationKey {
    let minted = ObservationKey::from_key("predictive-score-interval-facade", &i.to_le_bytes());
    ObservationKey::from_bytes(*minted.as_bytes())
}

/// A deterministic standard-normal stream (xorshift and Box-Muller).
struct Normal(u64);

impl Normal {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 + 0.5) / 9_007_199_254_740_992.0
    }

    fn draw(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

#[derive(Clone, Copy)]
enum Design {
    Iid,
    Serial,
    Cluster,
}

fn spec(design: Design, n: usize, fit: FitDeclaration, declare: bool) -> PredictiveAssessmentSpec {
    let mut normal = Normal(0x9e37_79b9_7f4a_7c15);
    let values: Vec<f64> = match design {
        Design::Iid => (0..n).map(|_| normal.draw()).collect(),
        Design::Serial => {
            let mut previous = normal.draw();
            (0..n)
                .map(|_| {
                    previous = 0.5 * previous + 0.75_f64.sqrt() * normal.draw();
                    previous
                })
                .collect()
        }
        Design::Cluster => {
            let mut shared = 0.0;
            (0..n)
                .map(|i| {
                    if i % CLUSTER_SIZE == 0 {
                        shared = normal.draw();
                    }
                    0.5_f64.sqrt() * shared + 0.5_f64.sqrt() * normal.draw()
                })
                .collect()
        }
    };
    let target = PredictiveTarget::new("outcome", Horizon::Steps(1), "population", "unit");
    let ids: Vec<ObservationKey> = (0..n as u64).map(key).collect();
    let fitted_on = ids.clone();
    let predictor = |name: &str, mean: f64, sd: f64, fit: FitDeclaration| PredictorLaws {
        name: name.to_owned(),
        source: LawSource::Native { model: name.to_owned() },
        target: target.clone(),
        laws: vec![ForecastLaw::Gaussian { mean, sd }; n],
        fit,
    };
    let in_sample = matches!(fit, FitDeclaration::FittedOn(_));
    let fit_a = if in_sample { FitDeclaration::FittedOn(fitted_on) } else { fit };
    let observations = RealizedObservations {
        target: target.clone(),
        ids,
        outcomes: values.into_iter().map(Outcome::Real).collect(),
    };
    let validation = match design {
        Design::Cluster => ValidationDesign::Blocked {
            blocks: (0..n).map(|i| (i / CLUSTER_SIZE) as u32).collect(),
        },
        _ => ValidationDesign::HeldOutSplit,
    };
    let mut spec = PredictiveAssessmentSpec::new(
        observations,
        vec![
            predictor("forecast_a", 0.3, 1.2, fit_a),
            predictor("forecast_b", 0.0, 1.5, FitDeclaration::NotFitted),
        ],
        validation,
        vec![ScoreKind::Crps, ScoreKind::Log],
    );
    if declare {
        spec.dependence = Some(match design {
            Design::Iid => DependenceDeclaration::Iid,
            Design::Serial => DependenceDeclaration::Serial {
                time_index: (0..n as u64).collect(),
                truncation_lag: (n as f64).cbrt().floor() as u32,
            },
            Design::Cluster => DependenceDeclaration::Cluster {
                labels: (0..n).map(|i| (i / CLUSTER_SIZE) as u32).collect(),
            },
        });
    }
    spec
}

fn assess(spec: &PredictiveAssessmentSpec) -> PreparedPredictiveAssessment {
    PreparedPredictiveAssessment::assess(spec, &ExecutionContext::for_tests(3)).expect("assessed")
}

fn mean_query(score: ScoreKind) -> ScoreIntervalQuery {
    ScoreIntervalQuery { predictor: "forecast_a".to_owned(), reference: None, score, level: 0.95 }
}

#[test]
fn an_expected_score_interval_inside_a_measured_design_is_bound_to_its_record() {
    let cases = [
        (Design::Iid, 800, "iid", "student_t_iid"),
        (Design::Cluster, 1600, "cluster", "student_t_cluster_robust"),
    ];
    for (design, n, dependence, construction) in cases {
        let prepared = assess(&spec(design, n, FitDeclaration::NotFitted, true));
        for score in [ScoreKind::Crps, ScoreKind::Log] {
            let interval = prepared.expected_score_interval(&mean_query(score)).expect("measured");
            assert!(interval.lower < interval.mean && interval.mean < interval.upper);
            assert_eq!(interval.cases, n);
            assert_eq!(interval.dependence, dependence);
            assert_eq!(interval.construction, construction);
            assert!(interval.coverage_record.starts_with("cov.predictive_score_interval."));
            assert!(interval.coverage_record.contains(&format!(".{construction}.")));
            assert!(!interval.calibration_sha.is_empty());
            assert!(interval.assumptions.contains("finite variance"));
        }
        let difference = prepared
            .expected_score_interval(&ScoreIntervalQuery {
                predictor: "forecast_b".to_owned(),
                reference: Some("forecast_a".to_owned()),
                score: ScoreKind::Crps,
                level: 0.95,
            })
            .expect("measured difference");
        assert!(difference.lower < difference.upper);
        assert_eq!(difference.reference.as_deref(), Some("forecast_a"));
    }
}

#[test]
fn the_serial_log_score_and_difference_are_measured_but_the_serial_crps_mean_is_a_boundary() {
    let prepared = assess(&spec(Design::Serial, 1600, FitDeclaration::NotFitted, true));
    prepared.expected_score_interval(&mean_query(ScoreKind::Log)).expect("measured");
    prepared
        .expected_score_interval(&ScoreIntervalQuery {
            predictor: "forecast_b".to_owned(),
            reference: Some("forecast_a".to_owned()),
            score: ScoreKind::Crps,
            level: 0.95,
        })
        .expect("measured");
    let refused = prepared
        .expected_score_interval(&mean_query(ScoreKind::Crps))
        .expect_err("a named boundary licenses nothing");
    assert_eq!(refused.code, "cell_not_licensed");
    assert_eq!(refused.detail, "predictive_assessment.interval_scope_outside_measured");
}

#[test]
fn a_design_no_record_measured_is_refused_with_the_typed_reason() {
    // Below the measured number of cases, at another level, and for a different dependence
    // declaration at the same cases.
    let small = assess(&spec(Design::Iid, 100, FitDeclaration::NotFitted, true));
    let refused = small.expected_score_interval(&mean_query(ScoreKind::Crps)).unwrap_err();
    assert_eq!(refused.code, "cell_not_licensed");
    assert_eq!(refused.detail, "predictive_assessment.interval_scope_outside_measured");
    let measured = assess(&spec(Design::Iid, 800, FitDeclaration::NotFitted, true));
    let mut ninety = mean_query(ScoreKind::Crps);
    ninety.level = 0.9;
    assert_eq!(
        measured.expected_score_interval(&ninety).unwrap_err().detail,
        "predictive_assessment.interval_scope_outside_measured"
    );
    let serial_short = assess(&spec(Design::Serial, 800, FitDeclaration::NotFitted, true));
    assert_eq!(
        serial_short.expected_score_interval(&mean_query(ScoreKind::Log)).unwrap_err().detail,
        "predictive_assessment.interval_scope_outside_measured"
    );
}

#[test]
fn an_in_sample_predictor_and_an_undeclared_dependence_are_refused() {
    let in_sample = assess(&spec(Design::Iid, 800, FitDeclaration::FittedOn(Vec::new()), true));
    let refused = in_sample.expected_score_interval(&mean_query(ScoreKind::Crps)).unwrap_err();
    assert_eq!(refused.code, "evidence_double_counted");
    assert_eq!(refused.detail, "predictive_assessment.in_sample");
    let undeclared = assess(&spec(Design::Iid, 800, FitDeclaration::NotFitted, false));
    let refused = undeclared.expected_score_interval(&mean_query(ScoreKind::Crps)).unwrap_err();
    assert_eq!(refused.code, "sampling_dependence_unknown");
    assert_eq!(refused.detail, "predictive_assessment.interval_dependence_undeclared");
}

#[test]
fn the_unscoped_request_stays_refused() {
    let prepared = assess(&spec(Design::Iid, 800, FitDeclaration::NotFitted, true));
    let refused = prepared.request_score_interval().unwrap_err();
    assert_eq!(refused.code, "cell_not_licensed");
    assert_eq!(refused.detail, "predictive_assessment.interval_not_licensed");
}
