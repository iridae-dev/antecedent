//! Calibration diagnostics and score intervals of the predictive assessment against 60-digit
//! evaluation of their definitions (`oracle_data/predictive_assessment_reference.py`).
//!
//! Gaussian, Bernoulli, finite real and ensemble laws: the PIT histogram with its descriptors and
//! the mid-PIT values, the central-interval coverage and width at several levels (including
//! Bernoulli probabilities and levels whose cumulative probability equals the tail up to a
//! rounding), the reliability table, the mean predictive spread; and the Student-t interval of
//! a mean CRPS and of a CRPS difference under iid, serial (a shuffled time index, so that the
//! scores must be put in time order) and cluster dependence.

use antecedent_core::ExecutionContext;
use antecedent_validate::predictive_assessment::{
    DependenceDeclaration, FitDeclaration, ForecastLaw, Horizon, LawSource, ObservationKey,
    Outcome, PredictiveAssessmentReport, PredictiveAssessmentSpec, PredictiveTarget, PredictorLaws,
    RealizedObservations, ScoreIntervalQuery, ScoreKind, ValidationDesign, assess_predictive_laws,
    construct_expected_score_interval,
};
use serde_json::Value;

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/predictive_assessment_reference.json"))
        .expect("frozen reference")
}

fn vec_of(value: &Value) -> Vec<f64> {
    value.as_array().expect("array").iter().map(|v| v.as_f64().expect("number")).collect()
}

fn key(i: u64) -> ObservationKey {
    let minted = ObservationKey::from_key("predictive-assessment-reference", &i.to_le_bytes());
    ObservationKey::from_bytes(*minted.as_bytes())
}

fn target() -> PredictiveTarget {
    PredictiveTarget::new("outcome", Horizon::Contemporaneous, "cohort", "units")
}

fn predictor(name: &str, laws: Vec<ForecastLaw>) -> PredictorLaws {
    PredictorLaws {
        name: name.to_owned(),
        source: LawSource::Native { model: name.to_owned() },
        target: target(),
        laws,
        fit: FitDeclaration::NotFitted,
    }
}

fn spec(
    outcomes: Vec<Outcome>,
    predictors: Vec<PredictorLaws>,
    scores: Vec<ScoreKind>,
    pit_bins: usize,
    levels: Vec<f64>,
) -> PredictiveAssessmentSpec {
    let n = outcomes.len() as u64;
    let mut spec = PredictiveAssessmentSpec::new(
        RealizedObservations { target: target(), ids: (0..n).map(key).collect(), outcomes },
        predictors,
        ValidationDesign::SimulationStream { stream: "reference".into() },
        scores,
    );
    spec.dependence = Some(DependenceDeclaration::Iid);
    spec.pit_bins = pit_bins;
    spec.interval_levels = levels;
    spec
}

fn close(got: f64, want: f64, relative: f64, what: &str) {
    assert!((got - want).abs() <= 1e-12 + relative * want.abs(), "{what}: {got} vs {want}");
}

fn check_pit(report: &PredictiveAssessmentReport, predictor: usize, want: &Value) {
    let pit = report.predictors[predictor].diagnostics.pit.as_ref().expect("pit");
    let counts = vec_of(&want["counts"]);
    assert_eq!(pit.counts.len(), counts.len());
    for (i, (g, w)) in pit.counts.iter().zip(&counts).enumerate() {
        close(*g, *w, 1e-11, &format!("pit count {i}"));
    }
    close(
        pit.max_bin_deviation,
        want["max_bin_deviation"].as_f64().unwrap(),
        1e-10,
        "max deviation",
    );
    close(pit.chi_square, want["chi_square"].as_f64().unwrap(), 1e-10, "chi square");
    close(pit.ks_distance, want["ks"].as_f64().unwrap(), 1e-11, "ks");
    let mut got = pit.mid_pit.clone();
    got.sort_by(f64::total_cmp);
    let mut mids = vec_of(&want["mid_pit"]);
    mids.sort_by(f64::total_cmp);
    for (i, (g, w)) in got.iter().zip(&mids).enumerate() {
        close(*g, *w, 1e-11, &format!("mid pit {i}"));
    }
}

fn check_coverage(report: &PredictiveAssessmentReport, predictor: usize, want: &Value) {
    let coverage = report.predictors[predictor].diagnostics.coverage.as_ref().expect("coverage");
    for (got, want) in coverage.iter().zip(want.as_array().unwrap()) {
        let level = want["level"].as_f64().unwrap();
        assert_eq!(got.level, level);
        close(
            got.empirical_coverage,
            want["coverage"].as_f64().unwrap(),
            1e-12,
            &format!("coverage at {level}"),
        );
        close(got.mean_width, want["width"].as_f64().unwrap(), 1e-10, &format!("width at {level}"));
    }
}

#[test]
fn gaussian_diagnostics_equal_the_definitions() {
    let r = reference();
    let g = &r["gauss"];
    let means = vec_of(&g["means"]);
    let outcomes: Vec<Outcome> = vec_of(&g["outcomes"]).into_iter().map(Outcome::Real).collect();
    let sd = g["sd_a"].as_f64().unwrap();
    let spec = spec(
        outcomes,
        vec![predictor("a", means.iter().map(|m| ForecastLaw::gaussian(*m, sd)).collect())],
        vec![ScoreKind::Crps],
        g["bins"].as_u64().unwrap() as usize,
        vec_of(&g["levels"]),
    );
    let report = assess_predictive_laws(&spec, &ExecutionContext::for_tests(1)).unwrap();
    check_pit(&report, 0, &g["pit"]);
    check_coverage(&report, 0, &g["coverage"]);
    close(report.predictors[0].diagnostics.mean_predictive_sd.unwrap(), sd, 1e-14, "sd");
    close(
        report.predictors[0].scores[0].mean,
        g["crps_mean"].as_f64().unwrap(),
        1e-10,
        "mean crps",
    );
}

#[test]
fn bernoulli_diagnostics_include_the_rounding_boundary_of_the_central_interval() {
    let r = reference();
    let b = &r["bern"];
    let outcomes: Vec<Outcome> =
        vec_of(&b["y"]).into_iter().map(|y| Outcome::Category(y as u32)).collect();
    let mut s = spec(
        outcomes,
        vec![predictor("b", vec_of(&b["p"]).into_iter().map(ForecastLaw::bernoulli).collect())],
        vec![ScoreKind::Brier],
        b["pit_bins"].as_u64().unwrap() as usize,
        vec_of(&b["levels"]),
    );
    s.reliability_bins = b["reliability_bins"].as_u64().unwrap() as usize;
    let report = assess_predictive_laws(&s, &ExecutionContext::for_tests(1)).unwrap();
    check_pit(&report, 0, &b["pit"]);
    // Probability 0.8 at level 0.6 has `1 - p` equal to the tail 0.2 only up to a rounding.
    check_coverage(&report, 0, &b["coverage"]);
    let reliability = report.predictors[0].diagnostics.reliability.as_ref().unwrap();
    for (got, want) in reliability.bins.iter().zip(b["reliability"].as_array().unwrap()) {
        assert_eq!(got.count, want["count"].as_u64().unwrap());
        close(got.mean_forecast, want["mean_forecast"].as_f64().unwrap(), 1e-12, "mean forecast");
        close(got.observed_frequency, want["observed"].as_f64().unwrap(), 1e-12, "observed");
    }
    close(reliability.weighted_gap, b["weighted_gap"].as_f64().unwrap(), 1e-11, "weighted gap");
    close(
        report.predictors[0].diagnostics.mean_predictive_sd.unwrap(),
        b["mean_sd"].as_f64().unwrap(),
        1e-12,
        "mean sd",
    );
}

#[test]
fn a_two_point_finite_law_has_the_interval_of_the_bernoulli_law() {
    // The same forecast written two ways must have the same central interval at every level.
    let outcomes: Vec<Outcome> = (0..8).map(|i| Outcome::Real(f64::from(i % 2))).collect();
    let ps = [0.8, 0.9, 0.7, 0.6, 0.95, 0.05, 0.5, 0.3];
    let levels: Vec<f64> = (1..99).map(|i| f64::from(i) / 100.0).filter(|l| *l >= 0.5).collect();
    let bern = spec(
        (0..8).map(|i| Outcome::Category(i % 2)).collect(),
        vec![predictor("b", ps.iter().map(|p| ForecastLaw::bernoulli(*p)).collect())],
        vec![ScoreKind::Crps],
        4,
        levels.clone(),
    );
    let fin = spec(
        outcomes,
        vec![predictor(
            "f",
            ps.iter()
                .map(|p| ForecastLaw::FiniteReal {
                    support: vec![0.0, 1.0],
                    probabilities: vec![1.0 - p, *p],
                })
                .collect(),
        )],
        vec![ScoreKind::Crps],
        4,
        levels,
    );
    let rb = assess_predictive_laws(&bern, &ExecutionContext::for_tests(1)).unwrap();
    let rf = assess_predictive_laws(&fin, &ExecutionContext::for_tests(1)).unwrap();
    let cb = rb.predictors[0].diagnostics.coverage.as_ref().unwrap();
    let cf = rf.predictors[0].diagnostics.coverage.as_ref().unwrap();
    for (a, b) in cb.iter().zip(cf) {
        assert_eq!(a.mean_width, b.mean_width, "width at level {}", a.level);
        assert_eq!(a.empirical_coverage, b.empirical_coverage, "coverage at level {}", a.level);
    }
}

#[test]
fn finite_and_ensemble_diagnostics_equal_the_definitions() {
    let r = reference();
    let f = &r["fin"];
    let support = vec_of(&f["support"]);
    let laws: Vec<ForecastLaw> = f["probs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| ForecastLaw::FiniteReal { support: support.clone(), probabilities: vec_of(p) })
        .collect();
    let outcomes: Vec<Outcome> = vec_of(&f["y"]).into_iter().map(Outcome::Real).collect();
    let s = spec(
        outcomes,
        vec![predictor("f", laws)],
        vec![ScoreKind::Crps],
        f["pit_bins"].as_u64().unwrap() as usize,
        vec_of(&f["levels"]),
    );
    let report = assess_predictive_laws(&s, &ExecutionContext::for_tests(1)).unwrap();
    check_pit(&report, 0, &f["pit"]);
    check_coverage(&report, 0, &f["coverage"]);
    close(
        report.predictors[0].diagnostics.mean_predictive_sd.unwrap(),
        f["mean_sd"].as_f64().unwrap(),
        1e-12,
        "sd",
    );

    let e = &r["ens"];
    let laws: Vec<ForecastLaw> = e["draws"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| ForecastLaw::EmpiricalSample { draws: vec_of(d) })
        .collect();
    let outcomes: Vec<Outcome> = vec_of(&e["y"]).into_iter().map(Outcome::Real).collect();
    let s = spec(
        outcomes,
        vec![predictor("e", laws)],
        vec![ScoreKind::Crps],
        e["pit_bins"].as_u64().unwrap() as usize,
        vec_of(&e["levels"]),
    );
    let report = assess_predictive_laws(&s, &ExecutionContext::for_tests(1)).unwrap();
    check_pit(&report, 0, &e["pit"]);
    check_coverage(&report, 0, &e["coverage"]);
    close(
        report.predictors[0].diagnostics.mean_predictive_sd.unwrap(),
        e["mean_sd"].as_f64().unwrap(),
        1e-12,
        "sd",
    );
}

#[test]
fn score_intervals_are_the_student_t_intervals_of_the_declared_dependence() {
    let r = reference();
    let g = &r["gauss"];
    let means = vec_of(&g["means"]);
    let mean_b = vec_of(&g["mean_b"]);
    let outcomes: Vec<Outcome> = vec_of(&g["outcomes"]).into_iter().map(Outcome::Real).collect();
    let n = means.len();
    let time_index: Vec<u64> = vec_of(&g["time_index"]).into_iter().map(|v| v as u64).collect();
    let labels: Vec<u32> = vec_of(&g["labels"]).into_iter().map(|v| v as u32).collect();
    for (name, dependence) in [
        ("iid", DependenceDeclaration::Iid),
        (
            "serial",
            DependenceDeclaration::Serial { time_index: time_index.clone(), truncation_lag: 3 },
        ),
        ("cluster", DependenceDeclaration::Cluster { labels: labels.clone() }),
    ] {
        let mut s = spec(
            outcomes.clone(),
            vec![
                predictor(
                    "a",
                    means
                        .iter()
                        .map(|m| ForecastLaw::gaussian(*m, g["sd_a"].as_f64().unwrap()))
                        .collect(),
                ),
                predictor(
                    "b",
                    mean_b
                        .iter()
                        .map(|m| ForecastLaw::gaussian(*m, g["sd_b"].as_f64().unwrap()))
                        .collect(),
                ),
            ],
            vec![ScoreKind::Crps],
            8,
            vec![0.5],
        );
        s.dependence = Some(dependence);
        let report = assess_predictive_laws(&s, &ExecutionContext::for_tests(1)).unwrap();
        for level in [0.9, 0.5] {
            let want_set = &g["intervals"][level.to_string()];
            for (suffix, reference) in [("mean", None), ("diff", Some("b".to_owned()))] {
                // The difference is a minus b: the candidate is `a`, the reference `b`.
                let query = ScoreIntervalQuery {
                    predictor: "a".into(),
                    reference,
                    score: ScoreKind::Crps,
                    level,
                };
                let got = construct_expected_score_interval(&s, &report, &query).unwrap().interval;
                let want = &want_set[format!("{name}_{suffix}")];
                let tag = format!("{name} {suffix} level {level}");
                assert_eq!(got.cases, n);
                close(got.mean, want["mean"].as_f64().unwrap(), 1e-10, &format!("{tag} mean"));
                close(got.standard_error, want["se"].as_f64().unwrap(), 1e-9, &format!("{tag} se"));
                close(
                    got.degrees_of_freedom,
                    want["df"].as_f64().unwrap(),
                    1e-13,
                    &format!("{tag} df"),
                );
                close(got.lower, want["lower"].as_f64().unwrap(), 1e-7, &format!("{tag} lower"));
                close(got.upper, want["upper"].as_f64().unwrap(), 1e-7, &format!("{tag} upper"));
            }
        }
    }
}
