//! Predictive assessment: closed-form and hand-calculated scores, calibrated and
//! miscalibrated generators, the proper-score property on exact finite laws, and
//! the refusals of the contract.
//!
//! Oracles:
//!
//! * Gaussian CRPS `sigma (z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi))`:
//!   `N(0,1)` at `0` gives `(sqrt 2 - 1) / sqrt(pi) = 0.233694977...`, at `1` gives
//!   `0.602441357...`; the expected CRPS of the true `N(0,1)` is `1/sqrt(pi)`.
//! * Bernoulli `p = 0.7`: Brier `0.09` (hit) and `0.49` (miss); log score `-ln 0.7`
//!   and `-ln 0.3`; the CRPS of a binary law equals the Brier score.
//! * Finite law `{0: 1/2, 1: 1/2}` at `0.5`: CRPS `1/4` by integrating
//!   `(F - 1{t >= y})^2` and by the energy form `1/2 - 1/4`.
//! * Sample `{0, 1, 2}` at `1.5`: energy form `5/6 - 4/9 = 7/18`; fair `5/6 - 2/3 = 1/6`.

use antecedent_core::{CausalRng, ExecutionContext};
use antecedent_kernels::standard_normal;
use antecedent_validate::predictive_assessment::{
    DependenceDeclaration, EnsembleCrpsForm, FiniteTruth, FitDeclaration, ForecastLaw, Horizon,
    LawSource, ObservationKey, Outcome, OverlapPolicy, OverlapStatus, PredictiveAssessmentError,
    PredictiveAssessmentReport, PredictiveAssessmentSpec, PredictiveReplicates, PredictiveTarget,
    PredictorLaws, ProviderIdentity, ProviderTrustLabel, RealizedObservations,
    ScoreIntervalRequest, ScoreKind, ValidationDesign, assess_predictive_laws, brier_binary,
    crps_gaussian, crps_weighted_atoms, detail, distribution_drift, energy_score_weighted,
    exact_expected_score, log_score_gaussian, score_case, variance_of_mean,
};

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(7)
}

fn key(i: u64) -> ObservationKey {
    ObservationKey::from_key("predictive-assessment-test", &i.to_le_bytes())
}

fn target(horizon: Horizon) -> PredictiveTarget {
    PredictiveTarget::new("outcome", horizon, "cohort", "mmHg")
}

fn observations(outcomes: Vec<Outcome>, horizon: Horizon) -> RealizedObservations {
    RealizedObservations {
        target: target(horizon),
        ids: (0..outcomes.len() as u64).map(key).collect(),
        outcomes,
    }
}

fn predictor(name: &str, laws: Vec<ForecastLaw>, horizon: Horizon) -> PredictorLaws {
    PredictorLaws {
        name: name.to_owned(),
        source: LawSource::Native { model: name.to_owned() },
        target: target(horizon),
        laws,
        fit: FitDeclaration::NotFitted,
    }
}

fn close(a: f64, b: f64, tolerance: f64) -> bool {
    (a - b).abs() <= tolerance
}

fn refusal(
    result: Result<PredictiveAssessmentReport, PredictiveAssessmentError>,
) -> (String, String) {
    let error = result.expect_err("the request must be refused");
    (error.reason_code().to_owned(), error.detail().to_owned())
}

fn mean_score(report: &PredictiveAssessmentReport, predictor: usize, kind: ScoreKind) -> f64 {
    report.predictors[predictor]
        .scores
        .iter()
        .find(|s| s.kind == kind)
        .expect("score requested")
        .mean
}

// -- closed forms -----------------------------------------------------------------------

#[test]
fn gaussian_crps_matches_the_closed_form_hand_values() {
    let at_zero = (2.0_f64.sqrt() - 1.0) / std::f64::consts::PI.sqrt();
    assert!(close(crps_gaussian(0.0, 1.0, 0.0), at_zero, 1e-12));
    assert!(close(crps_gaussian(0.0, 1.0, 0.0), 0.233_694_977_255_109, 1e-12));
    assert!(close(crps_gaussian(0.0, 1.0, 1.0), 0.602_441_357_627_616_3, 1e-12));
    // Location-scale: CRPS(N(mu, s^2), y) = s CRPS(N(0, 1), (y - mu) / s).
    assert!(close(crps_gaussian(1.0, 2.0, -1.0), 2.0 * 0.602_441_357_627_616_3, 1e-12));
    assert!(close(crps_gaussian(1.0, 2.0, -1.0), crps_gaussian(0.0, 1.0, 1.0) * 2.0, 1e-12));
}

#[test]
fn gaussian_crps_equals_the_integral_of_the_squared_distribution_difference() {
    // Midpoint quantile atoms of N(0, 1) approximate the law; CRPS is the integral form.
    let atoms: Vec<(f64, f64)> = {
        let m = 4000_usize;
        (0..m)
            .map(|i| {
                let p = (i as f64 + 0.5) / m as f64;
                (antecedent_kernels::norm_inv(p), 1.0 / m as f64)
            })
            .collect()
    };
    let approximate = crps_weighted_atoms(&atoms, 1.0);
    assert!(close(approximate, 0.602_441_357_627_616_3, 2e-3));
}

#[test]
fn gaussian_log_score_matches_the_density() {
    // -ln N(1; 0, 2^2) = ln 2 + (1/2) ln(2 pi) + 1/8.
    assert!(close(log_score_gaussian(0.0, 2.0, 1.0), 1.737_085_713_764_618, 1e-12));
    assert!(close(log_score_gaussian(0.0, 1.0, 0.0), 0.918_938_533_204_672_7, 1e-12));
}

#[test]
fn standalone_scoring_refuses_malformed_inputs_and_non_finite_scores() {
    let form = EnsembleCrpsForm::EmpiricalLaw;
    let malformed_law = ForecastLaw::Gaussian { mean: 0.0, sd: -1.0 };
    assert!(score_case(ScoreKind::Log, form, &malformed_law, &Outcome::Real(0.0)).is_err());

    let law = ForecastLaw::gaussian(0.0, 1.0);
    assert!(score_case(ScoreKind::Crps, form, &law, &Outcome::Real(f64::NAN)).is_err());

    // Finite inputs can still overflow an intermediate standardized residual.
    let extreme = ForecastLaw::gaussian(f64::MAX, 1.0);
    assert!(score_case(ScoreKind::Log, form, &extreme, &Outcome::Real(-f64::MAX)).is_err());
}

#[test]
fn exact_expected_score_refuses_malformed_finite_truths() {
    let forecast = ForecastLaw::gaussian(0.0, 1.0);
    let non_finite_support =
        FiniteTruth::Real { support: vec![f64::NAN], probabilities: vec![1.0] };
    assert!(exact_expected_score(ScoreKind::Crps, &forecast, &non_finite_support).is_err());

    let overflowing_dimension =
        FiniteTruth::Vector { dimension: usize::MAX, atoms: vec![], probabilities: vec![1.0] };
    assert!(exact_expected_score(ScoreKind::Energy, &forecast, &overflowing_dimension).is_err());
}

#[test]
fn bernoulli_brier_log_and_crps_match_hand_values() {
    let law = ForecastLaw::bernoulli(0.7);
    let form = EnsembleCrpsForm::EmpiricalLaw;
    let hit = Outcome::Category(1);
    let miss = Outcome::Category(0);
    assert!(close(score_case(ScoreKind::Brier, form, &law, &hit).unwrap(), 0.09, 1e-12));
    assert!(close(score_case(ScoreKind::Brier, form, &law, &miss).unwrap(), 0.49, 1e-12));
    assert!(close(brier_binary(0.7, true), 0.09, 1e-12));
    assert!(close(
        score_case(ScoreKind::Log, form, &law, &hit).unwrap(),
        0.356_674_943_938_732_4,
        1e-12
    ));
    assert!(close(
        score_case(ScoreKind::Log, form, &law, &miss).unwrap(),
        1.203_972_804_325_936,
        1e-12
    ));
    // The CRPS of a binary law is its Brier score.
    assert!(close(score_case(ScoreKind::Crps, form, &law, &hit).unwrap(), 0.09, 1e-12));
    assert!(close(score_case(ScoreKind::Crps, form, &law, &miss).unwrap(), 0.49, 1e-12));
}

#[test]
fn categorical_brier_and_log_match_hand_values() {
    let law = ForecastLaw::Categorical { probabilities: vec![0.5, 0.3, 0.2] };
    let form = EnsembleCrpsForm::EmpiricalLaw;
    // Outcome 1: (0.5)^2 + (0.3 - 1)^2 + (0.2)^2 = 0.25 + 0.49 + 0.04 = 0.78.
    let brier = score_case(ScoreKind::Brier, form, &law, &Outcome::Category(1)).unwrap();
    assert!(close(brier, 0.78, 1e-12));
    let log = score_case(ScoreKind::Log, form, &law, &Outcome::Category(2)).unwrap();
    assert!(close(log, -(0.2_f64.ln()), 1e-12));
}

#[test]
fn finite_law_and_ensemble_crps_agree_with_the_energy_form() {
    let form = EnsembleCrpsForm::EmpiricalLaw;
    let law = ForecastLaw::FiniteReal { support: vec![0.0, 1.0], probabilities: vec![0.5, 0.5] };
    let crps = score_case(ScoreKind::Crps, form, &law, &Outcome::Real(0.5)).unwrap();
    assert!(close(crps, 0.25, 1e-12));
    // Energy form of the same law as a two-draw sample: E|X - y| - E|X - X'| / 2 = 1/2 - 1/4.
    let sample = ForecastLaw::EmpiricalSample { draws: vec![1.0, 0.0] };
    let crps_sample = score_case(ScoreKind::Crps, form, &sample, &Outcome::Real(0.5)).unwrap();
    assert!(close(crps_sample, 0.25, 1e-12));

    let ensemble = ForecastLaw::EmpiricalSample { draws: vec![2.0, 0.0, 1.0] };
    let proper = score_case(ScoreKind::Crps, form, &ensemble, &Outcome::Real(1.5)).unwrap();
    assert!(close(proper, 7.0 / 18.0, 1e-12));
    let fair = score_case(ScoreKind::Crps, EnsembleCrpsForm::Fair, &ensemble, &Outcome::Real(1.5))
        .unwrap();
    assert!(close(fair, 1.0 / 6.0, 1e-12));
}

#[test]
fn energy_score_matches_hand_values_and_reduces_to_the_crps_in_one_dimension() {
    // Atoms (0,0) and (1,0) with mass 1/2 each, y = (0,0): 1/2 * 1 - (1/2)(1/2)(1) = 1/4.
    let atoms = [0.0, 0.0, 1.0, 0.0];
    let weights = [0.5, 0.5];
    assert!(close(energy_score_weighted(&atoms, &weights, 2, &[0.0, 0.0]), 0.25, 1e-12));
    // One dimension: the energy score is the CRPS.
    let line = [0.0, 1.0];
    assert!(close(energy_score_weighted(&line, &weights, 1, &[0.5]), 0.25, 1e-12));
    // A joint sample is scored as the law it is.
    let joint = ForecastLaw::JointSample { dimension: 2, draws: atoms.to_vec() };
    let value = score_case(
        ScoreKind::Energy,
        EnsembleCrpsForm::EmpiricalLaw,
        &joint,
        &Outcome::Vector(vec![0.0, 0.0]),
    )
    .unwrap();
    assert!(close(value, 0.25, 1e-12));
}

// -- the proper-score property on exact finite laws -------------------------------------

#[test]
fn the_true_finite_law_has_the_smallest_expected_score() {
    let support = vec![-1.0, 0.0, 2.0];
    let truth_probabilities = vec![0.2, 0.5, 0.3];
    let truth =
        FiniteTruth::Real { support: support.clone(), probabilities: truth_probabilities.clone() };
    let honest = ForecastLaw::FiniteReal {
        support: support.clone(),
        probabilities: truth_probabilities.clone(),
    };
    let perturbations = [
        vec![0.3, 0.4, 0.3],
        vec![0.2, 0.4, 0.4],
        vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        vec![0.1, 0.7, 0.2],
    ];
    for kind in [ScoreKind::Log, ScoreKind::Crps] {
        let best = exact_expected_score(kind, &honest, &truth).unwrap();
        for probabilities in &perturbations {
            let other = ForecastLaw::FiniteReal {
                support: support.clone(),
                probabilities: probabilities.clone(),
            };
            let value = exact_expected_score(kind, &other, &truth).unwrap();
            assert!(value > best + 1e-6, "{kind:?}: {value} must exceed the true law's {best}");
        }
    }
    // Entropy of the true law is the log-score minimum (Gibbs): -sum p ln p.
    let entropy: f64 = truth_probabilities.iter().map(|p| -p * p.ln()).sum();
    let log_best = exact_expected_score(ScoreKind::Log, &honest, &truth).unwrap();
    assert!(close(log_best, entropy, 1e-12));
}

#[test]
fn the_expected_brier_score_of_a_bernoulli_forecast_is_variance_plus_squared_error() {
    // E Brier(q) = p (1 - q)^2 + (1 - p) q^2 = p (1 - p) + (q - p)^2.
    let p = 0.3;
    let truth = FiniteTruth::Category { probabilities: vec![1.0 - p, p] };
    for q in [0.0, 0.1, 0.2, 0.3, 0.35, 0.5, 0.9, 1.0] {
        let value =
            exact_expected_score(ScoreKind::Brier, &ForecastLaw::bernoulli(q), &truth).unwrap();
        assert!(close(value, p * (1.0 - p) + (q - p) * (q - p), 1e-12));
    }
    let honest =
        exact_expected_score(ScoreKind::Brier, &ForecastLaw::bernoulli(p), &truth).unwrap();
    let other =
        exact_expected_score(ScoreKind::Brier, &ForecastLaw::bernoulli(0.4), &truth).unwrap();
    assert!(other > honest);
}

#[test]
fn the_energy_score_is_proper_on_a_finite_joint_law() {
    let atoms = vec![0.0, 0.0, 1.0, 0.0, 0.0, 2.0];
    let truth = FiniteTruth::Vector {
        dimension: 2,
        atoms: atoms.clone(),
        probabilities: vec![0.5, 0.25, 0.25],
    };
    let honest = ForecastLaw::JointFinite {
        dimension: 2,
        atoms: atoms.clone(),
        probabilities: vec![0.5, 0.25, 0.25],
    };
    let best = exact_expected_score(ScoreKind::Energy, &honest, &truth).unwrap();
    for probabilities in [vec![0.25, 0.5, 0.25], vec![0.4, 0.2, 0.4], vec![0.8, 0.1, 0.1]] {
        let other = ForecastLaw::JointFinite { dimension: 2, atoms: atoms.clone(), probabilities };
        let value = exact_expected_score(ScoreKind::Energy, &other, &truth).unwrap();
        assert!(value > best + 1e-6);
    }
}

// -- variance under the declared dependence ---------------------------------------------

#[test]
fn the_variance_of_a_mean_difference_follows_the_declared_dependence() {
    // iid: s^2 / n = 2.5 / 5.
    let d = [1.0, 2.0, 3.0, 4.0, 5.0];
    let iid = variance_of_mean(&d, &DependenceDeclaration::Iid, None).unwrap();
    assert!(close(iid, 0.5, 1e-12));

    // Serial with lag 1 on (1, 2, 3, 4): gamma_0 = 1.25, gamma_1 = 0.3125, weight 1/2,
    // long-run 1.25 + 2 * 0.5 * 0.3125 = 1.5625, divided by n = 4.
    let serial = DependenceDeclaration::Serial { time_index: vec![0, 1, 2, 3], truncation_lag: 1 };
    let value = variance_of_mean(&[1.0, 2.0, 3.0, 4.0], &serial, None).unwrap();
    assert!(close(value, 0.390_625, 1e-12));

    // Cluster: d = (1, 3 | 2, 4), mean 5/2; centred cluster sums -1 and +1; G/(G-1) = 2;
    // 2 * (1 + 1) / 16 = 1/4.
    let cluster = DependenceDeclaration::Cluster { labels: vec![0, 0, 1, 1] };
    let value = variance_of_mean(&[1.0, 3.0, 2.0, 4.0], &cluster, Some(&[0, 0, 1, 1])).unwrap();
    assert!(close(value, 0.25, 1e-12));

    // One cluster has no cluster-robust variance.
    let error = variance_of_mean(&[1.0, 3.0], &cluster, Some(&[5, 5])).unwrap_err();
    assert_eq!(error.detail(), detail::TOO_FEW_CLUSTERS);
}

// -- calibrated and miscalibrated generators --------------------------------------------

fn gaussian_outcomes(n: usize, seed: u64) -> Vec<Outcome> {
    let mut rng = CausalRng::from_seed(seed);
    (0..n).map(|_| Outcome::Real(standard_normal(&mut rng))).collect()
}

fn gaussian_laws(n: usize, mean: f64, sd: f64) -> Vec<ForecastLaw> {
    vec![ForecastLaw::gaussian(mean, sd); n]
}

fn generator_spec(n: usize) -> PredictiveAssessmentSpec {
    let outcomes = gaussian_outcomes(n, 20_240_611);
    let mut spec = PredictiveAssessmentSpec::new(
        observations(outcomes, Horizon::Contemporaneous),
        vec![
            predictor("calibrated", gaussian_laws(n, 0.0, 1.0), Horizon::Contemporaneous),
            predictor("overconfident", gaussian_laws(n, 0.0, 0.5), Horizon::Contemporaneous),
            predictor("biased", gaussian_laws(n, 1.0, 1.0), Horizon::Contemporaneous),
        ],
        ValidationDesign::SimulationStream { stream: "standard-normal seed 20240611".into() },
        vec![ScoreKind::Log, ScoreKind::Crps],
    );
    spec.dependence = Some(DependenceDeclaration::Iid);
    spec.pit_bins = 10;
    spec.interval_levels = vec![0.5, 0.9];
    spec
}

#[test]
fn a_calibrated_generator_has_a_uniform_pit_and_nominal_coverage() {
    let n = 2000;
    let report = assess_predictive_laws(&generator_spec(n), &ctx()).unwrap();
    let diagnostics = &report.predictors[0].diagnostics;
    assert_eq!(diagnostics.label, "diagnostic");
    assert_eq!(diagnostics.calibration, "unmeasured");
    let pit = diagnostics.pit.as_ref().unwrap();
    // The KS distance of n = 2000 uniform PIT values exceeds 0.05 with probability ~1e-10.
    assert!(pit.ks_distance < 0.05, "ks {}", pit.ks_distance);
    assert!(pit.max_bin_deviation < 0.03, "bin deviation {}", pit.max_bin_deviation);
    let coverage = diagnostics.coverage.as_ref().unwrap();
    assert!(close(coverage[0].empirical_coverage, 0.5, 0.05));
    assert!(close(coverage[1].empirical_coverage, 0.9, 0.04));
    // The expected CRPS of the true law is 1 / sqrt(pi).
    let expected = 1.0 / std::f64::consts::PI.sqrt();
    assert!(close(mean_score(&report, 0, ScoreKind::Crps), expected, 0.04));
    // The expected log score of the true law is the entropy (1 + ln 2 pi) / 2.
    let entropy = 0.5 * (1.0 + (2.0 * std::f64::consts::PI).ln());
    assert!(close(mean_score(&report, 0, ScoreKind::Log), entropy, 0.08));
}

#[test]
fn miscalibrated_generators_are_flagged_by_the_diagnostics() {
    let n = 2000;
    let report = assess_predictive_laws(&generator_spec(n), &ctx()).unwrap();
    let calibrated = report.predictors[0].diagnostics.pit.as_ref().unwrap();
    let overconfident = report.predictors[1].diagnostics.pit.as_ref().unwrap();
    let biased = report.predictors[2].diagnostics.pit.as_ref().unwrap();
    // Population KS distances are 0.161 (sd 0.5 forecast of unit-sd outcomes) and 0.383 (mean 1).
    assert!(overconfident.ks_distance > 0.10, "ks {}", overconfident.ks_distance);
    assert!(biased.ks_distance > 0.25, "ks {}", biased.ks_distance);
    assert!(overconfident.chi_square > 10.0 * calibrated.chi_square);
    // The 90% interval of an overconfident law covers about 59% of the outcomes.
    let coverage = report.predictors[1].diagnostics.coverage.as_ref().unwrap();
    assert!(coverage[1].empirical_coverage < 0.70, "coverage {}", coverage[1].empirical_coverage);
    // Sharpness is reported separately: the overconfident law is sharper and worse.
    let sharp = report.predictors[1].diagnostics.mean_predictive_sd.unwrap();
    let honest = report.predictors[0].diagnostics.mean_predictive_sd.unwrap();
    assert!(close(sharp, 0.5, 1e-12) && close(honest, 1.0, 1e-12));
    // Both miscalibrated laws score worse than the calibrated one, on both scores.
    for kind in [ScoreKind::Log, ScoreKind::Crps] {
        assert!(mean_score(&report, 0, kind) < mean_score(&report, 1, kind));
        assert!(mean_score(&report, 0, kind) < mean_score(&report, 2, kind));
    }
}

#[test]
fn score_differences_are_paired_and_state_their_dependence() {
    let n = 2000;
    let report = assess_predictive_laws(&generator_spec(n), &ctx()).unwrap();
    // Default comparisons: every other predictor against the first.
    assert_eq!(report.differences.len(), 4);
    for difference in &report.differences {
        assert_eq!(difference.reference, "calibrated");
        assert_eq!(difference.shared_observations, n);
        assert_eq!(difference.pairing, "paired_on_shared_observations");
        assert_eq!(difference.dependence, "iid");
        assert_eq!(difference.interval, "not_reported");
        assert_eq!(difference.label, "diagnostic");
        assert!(difference.mean_difference > 0.0);
        assert!(difference.standard_error.unwrap() > 0.0);
        assert!(difference.dm_statistic.unwrap() > 3.0);
    }
    // The mean difference is the difference of the mean scores.
    let crps_overconfident = report
        .differences
        .iter()
        .find(|d| d.candidate == "overconfident" && d.score == ScoreKind::Crps)
        .unwrap();
    assert!(close(
        crps_overconfident.mean_difference,
        mean_score(&report, 1, ScoreKind::Crps) - mean_score(&report, 0, ScoreKind::Crps),
        1e-12,
    ));
    // The exact expected CRPS of N(0, 0.5^2) under N(0, 1) is 0.60997; the calibrated one
    // is 1/sqrt(pi); their difference is about 0.0458.
    assert!(close(crps_overconfident.mean_difference, 0.0458, 0.02));
}

#[test]
fn no_declared_dependence_withholds_every_standard_error() {
    let mut spec = generator_spec(200);
    spec.dependence = None;
    let report = assess_predictive_laws(&spec, &ctx()).unwrap();
    assert_eq!(report.dependence, "undeclared");
    for difference in &report.differences {
        assert!(difference.standard_error.is_none());
        assert!(difference.dm_statistic.is_none());
        assert_eq!(difference.dependence, "undeclared");
    }
    // The point scores are still reported.
    assert!(mean_score(&report, 0, ScoreKind::Crps).is_finite());
}

// -- the shape of the report ------------------------------------------------------------

#[test]
fn every_report_is_point_only_unmeasured_and_not_a_causal_claim() {
    let report = assess_predictive_laws(&generator_spec(50), &ctx()).unwrap();
    assert_eq!(report.family, "predictive_assessment");
    assert_eq!(report.inference_claim, "point_only");
    assert_eq!(report.causal_validity, "not_established");
    assert_eq!(report.uncertainty.sampling_interval, "not reported");
    assert_eq!(report.uncertainty.reason_code, "cell_not_licensed");
    let interval = report.request_score_interval().unwrap_err();
    assert_eq!(interval.reason_code(), "cell_not_licensed");
    assert_eq!(interval.detail(), detail::INTERVAL_NOT_LICENSED);
    let causal = report.request_causal_validity().unwrap_err();
    assert_eq!(causal.detail(), detail::NOT_A_CAUSAL_CLAIM);
    let probability = report.request_score_as_probability().unwrap_err();
    assert_eq!(probability.detail(), detail::SCORE_NOT_PROBABILITY);
}

#[test]
fn a_score_interval_request_is_refused_with_cell_not_licensed() {
    let mut spec = generator_spec(50);
    spec.score_interval =
        Some(ScoreIntervalRequest { level: 0.95, method: "block bootstrap".to_owned() });
    let (code, detail_slot) = refusal(assess_predictive_laws(&spec, &ctx()));
    assert_eq!(code, "cell_not_licensed");
    assert_eq!(detail_slot, detail::INTERVAL_NOT_LICENSED);
}

// -- target, horizon and law-source refusals --------------------------------------------

#[test]
fn a_mismatched_target_or_horizon_is_refused_not_converted() {
    let n = 5;
    let outcomes = gaussian_outcomes(n, 1);
    let base = |laws_target: PredictiveTarget| {
        let mut p = predictor("p", gaussian_laws(n, 0.0, 1.0), Horizon::Contemporaneous);
        p.target = laws_target;
        PredictiveAssessmentSpec::new(
            observations(outcomes.clone(), Horizon::Contemporaneous),
            vec![p],
            ValidationDesign::HeldOutSplit,
            vec![ScoreKind::Crps],
        )
    };
    let units = PredictiveTarget::new("outcome", Horizon::Contemporaneous, "cohort", "kPa");
    let (code, slot) = refusal(assess_predictive_laws(&base(units), &ctx()));
    assert_eq!(
        (code.as_str(), slot.as_str()),
        ("quantity_semantics_mismatch", detail::TARGET_MISMATCH)
    );

    let population = PredictiveTarget::new("outcome", Horizon::Contemporaneous, "other", "mmHg");
    let (_, slot) = refusal(assess_predictive_laws(&base(population), &ctx()));
    assert_eq!(slot, detail::TARGET_MISMATCH);

    let horizon = PredictiveTarget::new("outcome", Horizon::Steps(3), "cohort", "mmHg");
    let (code, slot) = refusal(assess_predictive_laws(&base(horizon), &ctx()));
    assert_eq!(
        (code.as_str(), slot.as_str()),
        ("quantity_semantics_mismatch", detail::HORIZON_MISMATCH)
    );
}

#[test]
fn a_callback_law_and_an_incomplete_provider_identity_are_refused_by_name() {
    let n = 4;
    let mut spec = PredictiveAssessmentSpec::new(
        observations(gaussian_outcomes(n, 2), Horizon::Contemporaneous),
        vec![predictor("callback", gaussian_laws(n, 0.0, 1.0), Horizon::Contemporaneous)],
        ValidationDesign::HeldOutSplit,
        vec![ScoreKind::Crps],
    );
    spec.predictors[0].source = LawSource::Callback { description: "python closure".into() };
    let (code, slot) = refusal(assess_predictive_laws(&spec, &ctx()));
    assert_eq!(
        (code.as_str(), slot.as_str()),
        ("route_not_supported", detail::LAW_CALLBACK_NOT_BUILT)
    );

    let identity = ProviderIdentity {
        provider_id: "svc".into(),
        object_id: "law-1".into(),
        version_id: "v3".into(),
        snapshot_id: String::new(),
        request_id: "req".into(),
    };
    spec.predictors[0].source =
        LawSource::Provider { identity, trust: ProviderTrustLabel::ExternallyAttested };
    let (_, slot) = refusal(assess_predictive_laws(&spec, &ctx()));
    assert_eq!(slot, detail::PROVIDER_IDENTITY_INCOMPLETE);
}

#[test]
fn a_score_a_law_does_not_support_is_refused() {
    let n = 3;
    let sample = ForecastLaw::EmpiricalSample { draws: vec![0.0, 1.0, 2.0] };
    let spec = PredictiveAssessmentSpec::new(
        observations(gaussian_outcomes(n, 3), Horizon::Contemporaneous),
        vec![predictor("sample", vec![sample; n], Horizon::Contemporaneous)],
        ValidationDesign::HeldOutSplit,
        vec![ScoreKind::Log],
    );
    let (code, slot) = refusal(assess_predictive_laws(&spec, &ctx()));
    assert_eq!((code.as_str(), slot.as_str()), ("route_not_supported", detail::SCORE_NOT_DEFINED));
}

#[test]
fn an_infinite_log_score_is_refused_rather_than_averaged() {
    let spec = PredictiveAssessmentSpec::new(
        observations(vec![Outcome::Category(1)], Horizon::Contemporaneous),
        vec![predictor("sure", vec![ForecastLaw::bernoulli(0.0)], Horizon::Contemporaneous)],
        ValidationDesign::HeldOutSplit,
        vec![ScoreKind::Log],
    );
    let (_, slot) = refusal(assess_predictive_laws(&spec, &ctx()));
    assert_eq!(slot, detail::LOG_SCORE_INFINITE);
}

// -- designs, dependence and folds ------------------------------------------------------

fn rolling_spec(
    dependence: Option<DependenceDeclaration>,
    origins: Vec<u64>,
    horizon: u32,
) -> PredictiveAssessmentSpec {
    let n = origins.len();
    let mut spec = PredictiveAssessmentSpec::new(
        observations(gaussian_outcomes(n, 11), Horizon::Steps(horizon)),
        vec![
            predictor("a", gaussian_laws(n, 0.0, 1.0), Horizon::Steps(horizon)),
            predictor("b", gaussian_laws(n, 0.2, 1.0), Horizon::Steps(horizon)),
        ],
        ValidationDesign::RollingOrigin { origins, horizon_steps: horizon },
        vec![ScoreKind::Crps],
    );
    spec.dependence = dependence;
    spec
}

#[test]
fn iid_is_refused_for_rolling_origin_and_blocked_designs() {
    let spec = rolling_spec(Some(DependenceDeclaration::Iid), vec![1, 2, 3, 4, 5, 6], 1);
    let (code, slot) = refusal(assess_predictive_laws(&spec, &ctx()));
    assert_eq!(
        (code.as_str(), slot.as_str()),
        ("sampling_dependence_unknown", detail::IID_NOT_LICENSED)
    );

    let n = 6;
    let mut blocked = PredictiveAssessmentSpec::new(
        observations(gaussian_outcomes(n, 5), Horizon::Contemporaneous),
        vec![predictor("a", gaussian_laws(n, 0.0, 1.0), Horizon::Contemporaneous)],
        ValidationDesign::Blocked { blocks: vec![0, 0, 1, 1, 2, 2] },
        vec![ScoreKind::Crps],
    );
    blocked.dependence = Some(DependenceDeclaration::Iid);
    let (_, slot) = refusal(assess_predictive_laws(&blocked, &ctx()));
    assert_eq!(slot, detail::IID_NOT_LICENSED);
}

#[test]
fn a_serial_truncation_lag_shorter_than_the_horizon_is_refused() {
    let dependence =
        DependenceDeclaration::Serial { time_index: vec![1, 2, 3, 4, 5, 6], truncation_lag: 1 };
    let spec = rolling_spec(Some(dependence), vec![1, 2, 3, 4, 5, 6], 4);
    let (_, slot) = refusal(assess_predictive_laws(&spec, &ctx()));
    assert_eq!(slot, detail::SERIAL_LAG_BELOW_HORIZON);
}

#[test]
fn rolling_origin_reports_a_score_per_origin_and_a_serial_standard_error() {
    let origins = vec![10, 10, 20, 20, 30, 30];
    let dependence =
        DependenceDeclaration::Serial { time_index: vec![1, 2, 3, 4, 5, 6], truncation_lag: 1 };
    let report =
        assess_predictive_laws(&rolling_spec(Some(dependence), origins, 2), &ctx()).unwrap();
    let folds = &report.predictors[0].scores[0].folds;
    assert_eq!(folds.iter().map(|f| f.fold).collect::<Vec<_>>(), vec![10, 20, 30]);
    assert!(folds.iter().all(|f| f.cases == 2));
    let overall = report.predictors[0].scores[0].mean;
    let fold_mean = folds.iter().map(|f| f.mean).sum::<f64>() / 3.0;
    assert!(close(overall, fold_mean, 1e-12));
    // Case order is a time order, so the PIT drift between the two halves is reported.
    let drift = report.predictors[0].diagnostics.drift.as_ref().unwrap();
    assert_eq!((drift.reference_cases, drift.comparison_cases), (3, 3));
    assert!(drift.ks_distance >= 0.0 && drift.ks_distance <= 1.0);
    // A held-out split has no time order, so no drift is reported.
    let held_out = assess_predictive_laws(&generator_spec(40), &ctx()).unwrap();
    assert!(held_out.predictors[0].diagnostics.drift.is_none());
    let difference = &report.differences[0];
    assert_eq!(difference.dependence, "serial");
    assert!(difference.standard_error.is_some());
}

#[test]
fn a_single_cluster_cannot_give_a_cluster_standard_error() {
    let n = 6;
    let mut spec = PredictiveAssessmentSpec::new(
        observations(gaussian_outcomes(n, 9), Horizon::Contemporaneous),
        vec![
            predictor("a", gaussian_laws(n, 0.0, 1.0), Horizon::Contemporaneous),
            predictor("b", gaussian_laws(n, 0.5, 1.0), Horizon::Contemporaneous),
        ],
        ValidationDesign::Blocked { blocks: vec![1; 6] },
        vec![ScoreKind::Crps],
    );
    spec.dependence = Some(DependenceDeclaration::Cluster { labels: vec![1; 6] });
    let (code, slot) = refusal(assess_predictive_laws(&spec, &ctx()));
    assert_eq!((code.as_str(), slot.as_str()), ("too_few_clusters", detail::TOO_FEW_CLUSTERS));
}

// -- overlap with the data a law was fitted on ------------------------------------------

fn overlap_spec(fit: FitDeclaration, policy: OverlapPolicy) -> PredictiveAssessmentSpec {
    let n = 6;
    let mut p = predictor("model", gaussian_laws(n, 0.0, 1.0), Horizon::Contemporaneous);
    p.fit = fit;
    let mut spec = PredictiveAssessmentSpec::new(
        observations(gaussian_outcomes(n, 4), Horizon::Contemporaneous),
        vec![p],
        ValidationDesign::HeldOutSplit,
        vec![ScoreKind::Crps],
    );
    spec.overlap_policy = policy;
    spec
}

#[test]
fn overlap_is_detected_from_row_identities_and_labelled() {
    // Fitted on rows 100..110: disjoint from the assessed rows 0..6.
    let disjoint = FitDeclaration::FittedOn((100..110).map(key).collect());
    let report =
        assess_predictive_laws(&overlap_spec(disjoint, OverlapPolicy::RequireHeldOut), &ctx())
            .unwrap();
    assert_eq!(report.predictors[0].overlap, OverlapStatus::HeldOutVerified { fitted_on: 10 });
    assert_eq!(report.predictors[0].overlap.label(), "held_out_verified");

    // Fitted on rows 4..10: rows 4 and 5 are assessed too.
    let shared = FitDeclaration::FittedOn((4..10).map(key).collect());
    let labelled =
        assess_predictive_laws(&overlap_spec(shared.clone(), OverlapPolicy::LabelOnly), &ctx())
            .unwrap();
    assert_eq!(labelled.predictors[0].overlap, OverlapStatus::InSample { shared: 2, assessed: 6 });
    let (code, slot) = refusal(assess_predictive_laws(
        &overlap_spec(shared, OverlapPolicy::RequireHeldOut),
        &ctx(),
    ));
    assert_eq!((code.as_str(), slot.as_str()), ("evidence_double_counted", detail::IN_SAMPLE));
}

#[test]
fn an_undeclared_fit_is_unverified_never_held_out() {
    let undeclared = FitDeclaration::Undeclared { reason: "provider declared no fit data".into() };
    let report =
        assess_predictive_laws(&overlap_spec(undeclared.clone(), OverlapPolicy::LabelOnly), &ctx())
            .unwrap();
    assert_eq!(report.predictors[0].overlap.label(), "unverified");
    let (code, slot) = refusal(assess_predictive_laws(
        &overlap_spec(undeclared, OverlapPolicy::RequireHeldOut),
        &ctx(),
    ));
    assert_eq!(
        (code.as_str(), slot.as_str()),
        ("evidence_overlap_unknown", detail::OVERLAP_UNKNOWN)
    );
}

#[test]
fn predictors_fitted_on_shared_rows_state_their_fit_overlap() {
    let n = 4;
    let mut a = predictor("a", gaussian_laws(n, 0.0, 1.0), Horizon::Contemporaneous);
    a.fit = FitDeclaration::FittedOn((100..110).map(key).collect());
    let mut b = predictor("b", gaussian_laws(n, 0.1, 1.0), Horizon::Contemporaneous);
    b.fit = FitDeclaration::FittedOn((105..115).map(key).collect());
    let mut spec = PredictiveAssessmentSpec::new(
        observations(gaussian_outcomes(n, 6), Horizon::Contemporaneous),
        vec![a, b],
        ValidationDesign::HeldOutSplit,
        vec![ScoreKind::Crps],
    );
    spec.dependence = Some(DependenceDeclaration::Iid);
    let report = assess_predictive_laws(&spec, &ctx()).unwrap();
    assert_eq!(report.differences[0].fit_overlap, Some(5));
}

// -- reliability, PIT of discrete laws, drift, replicates, invariances -------------------

#[test]
fn reliability_and_the_discrete_pit_match_hand_values() {
    // Five forecasts of 0.2 with one event: one bin holds all five, observed frequency 1/5.
    let outcomes = vec![
        Outcome::Category(1),
        Outcome::Category(0),
        Outcome::Category(0),
        Outcome::Category(0),
        Outcome::Category(0),
    ];
    let mut spec = PredictiveAssessmentSpec::new(
        observations(outcomes, Horizon::Contemporaneous),
        vec![predictor("event", vec![ForecastLaw::bernoulli(0.2); 5], Horizon::Contemporaneous)],
        ValidationDesign::HeldOutSplit,
        vec![ScoreKind::Brier, ScoreKind::Log],
    );
    spec.dependence = Some(DependenceDeclaration::Iid);
    spec.reliability_bins = 10;
    spec.pit_bins = 2;
    let report = assess_predictive_laws(&spec, &ctx()).unwrap();
    let reliability = report.predictors[0].diagnostics.reliability.as_ref().unwrap();
    let bin = &reliability.bins[2];
    assert_eq!(bin.count, 5);
    assert!(close(bin.mean_forecast, 0.2, 1e-12) && close(bin.observed_frequency, 0.2, 1e-12));
    assert!(close(reliability.weighted_gap, 0.0, 1e-12));
    // Brier: (0.8)^2 + 4 (0.2)^2 = 0.64 + 0.16 = 0.8, mean 0.16.
    assert!(close(mean_score(&report, 0, ScoreKind::Brier), 0.16, 1e-12));
    // PIT with two bins: the event outcome has interval (0.8, 1) -> bin 1; each non-event has
    // (0, 0.8) spread over both bins in proportion 0.5/0.8 and 0.3/0.8.
    let pit = report.predictors[0].diagnostics.pit.as_ref().unwrap();
    let expected_bin0 = 4.0 * 0.5 / 0.8;
    assert!(close(pit.counts[0], expected_bin0, 1e-12));
    assert!(close(pit.counts[0] + pit.counts[1], 5.0, 1e-12));
}

#[test]
fn drift_compares_the_pit_of_two_windows() {
    let same = distribution_drift(&[0.1, 0.4, 0.6, 0.9], &[0.1, 0.4, 0.6, 0.9], 4).unwrap();
    assert!(close(same.ks_distance, 0.0, 1e-12));
    assert!(close(same.population_stability_index, 0.0, 1e-12));
    assert_eq!(same.label, "diagnostic");
    assert_eq!(same.route, "distribution_drift");
    let moved = distribution_drift(&[0.1, 0.2, 0.3], &[0.7, 0.8, 0.9], 4).unwrap();
    assert!(close(moved.ks_distance, 1.0, 1e-12));
    assert!(moved.population_stability_index > 1.0);
}

#[test]
fn drift_refuses_empty_windows_values_outside_the_unit_interval_and_bad_bins() {
    for (reference, comparison, bins) in [
        (vec![], vec![0.5], 4),
        (vec![0.5], vec![], 4),
        (vec![1.5], vec![0.5], 4),
        (vec![0.5], vec![f64::NAN], 4),
        (vec![0.5], vec![0.5], 1),
    ] {
        let error = distribution_drift(&reference, &comparison, bins).unwrap_err();
        assert_eq!(error.detail(), detail::DIAGNOSTIC_NOT_DEFINED);
        assert_eq!(error.reason_code(), "invalid_argument");
    }
}

#[test]
fn posterior_predictive_replicates_are_scored_as_sample_laws() {
    // Two rows, three replicates each: row 0 draws {0, 1, 2}, row 1 draws {5, 5, 5}.
    let replicates = PredictiveReplicates::new(3, 2, vec![0.0, 5.0, 1.0, 5.0, 2.0, 5.0]).unwrap();
    assert_eq!(replicates.laws().len(), 2);
    let predictor = replicates.into_predictor(
        "ppc",
        "bayesian_linear",
        target(Horizon::Contemporaneous),
        FitDeclaration::Undeclared { reason: "fitted on the scored rows".into() },
    );
    let spec = PredictiveAssessmentSpec::new(
        observations(vec![Outcome::Real(1.5), Outcome::Real(5.0)], Horizon::Contemporaneous),
        vec![predictor],
        ValidationDesign::HeldOutSplit,
        vec![ScoreKind::Crps],
    );
    let report = assess_predictive_laws(&spec, &ctx()).unwrap();
    // Row 0: 7/18 (hand value above); row 1: a point mass at the outcome, CRPS 0.
    assert!(close(mean_score(&report, 0, ScoreKind::Crps), 7.0 / 36.0, 1e-12));
    assert!(PredictiveReplicates::new(2, 2, vec![0.0; 3]).is_err());
}

#[test]
fn the_assessment_is_invariant_to_case_order_and_deterministic() {
    let n = 40;
    let outcomes = gaussian_outcomes(n, 31);
    let build = |order: &[usize]| {
        let laws_a: Vec<ForecastLaw> =
            order.iter().map(|&i| ForecastLaw::gaussian(i as f64 * 0.01, 1.0)).collect();
        let laws_b: Vec<ForecastLaw> =
            order.iter().map(|&i| ForecastLaw::gaussian(0.0, 1.0 + i as f64 * 0.01)).collect();
        let mut spec = PredictiveAssessmentSpec::new(
            RealizedObservations {
                target: target(Horizon::Contemporaneous),
                ids: order.iter().map(|&i| key(i as u64)).collect(),
                outcomes: order.iter().map(|&i| outcomes[i].clone()).collect(),
            },
            vec![
                predictor("a", laws_a, Horizon::Contemporaneous),
                predictor("b", laws_b, Horizon::Contemporaneous),
            ],
            ValidationDesign::HeldOutSplit,
            vec![ScoreKind::Log, ScoreKind::Crps],
        );
        spec.dependence = Some(DependenceDeclaration::Iid);
        spec
    };
    let forward: Vec<usize> = (0..n).collect();
    let mut shuffled: Vec<usize> = (0..n).collect();
    shuffled.reverse();
    shuffled.rotate_left(7);
    let first = assess_predictive_laws(&build(&forward), &ctx()).unwrap();
    let second = assess_predictive_laws(&build(&shuffled), &ctx()).unwrap();
    let again = assess_predictive_laws(&build(&forward), &ctx()).unwrap();
    assert_eq!(first.identity, second.identity);
    for kind in [ScoreKind::Log, ScoreKind::Crps] {
        for predictor in 0..2 {
            assert_eq!(
                mean_score(&first, predictor, kind).to_bits(),
                mean_score(&second, predictor, kind).to_bits()
            );
        }
    }
    assert_eq!(first, again);
    assert_eq!(first.first_replay_difference(&second), None);
}

#[test]
fn a_changed_law_or_observation_changes_the_identity() {
    let base = assess_predictive_laws(&generator_spec(30), &ctx()).unwrap();
    let mut shifted = generator_spec(30);
    shifted.predictors[1].laws[3] = ForecastLaw::gaussian(0.0, 0.500_000_1);
    let changed = assess_predictive_laws(&shifted, &ctx()).unwrap();
    assert_ne!(base.identity.digest, changed.identity.digest);
    assert_eq!(base.identity.observations, changed.identity.observations);
    assert_eq!(base.identity.first_difference(&changed.identity), Some("predictor_laws"));
    assert!(base.first_replay_difference(&changed).is_some());
}

#[test]
fn malformed_inputs_are_refused() {
    let n = 3;
    // Duplicate observation identities.
    let mut spec = PredictiveAssessmentSpec::new(
        observations(gaussian_outcomes(n, 1), Horizon::Contemporaneous),
        vec![predictor("a", gaussian_laws(n, 0.0, 1.0), Horizon::Contemporaneous)],
        ValidationDesign::HeldOutSplit,
        vec![ScoreKind::Crps],
    );
    spec.observations.ids[1] = spec.observations.ids[0];
    let (_, slot) = refusal(assess_predictive_laws(&spec, &ctx()));
    assert_eq!(slot, detail::INVALID_OBSERVATIONS);

    // A non-positive sd.
    let bad = PredictiveAssessmentSpec::new(
        observations(gaussian_outcomes(n, 1), Horizon::Contemporaneous),
        vec![predictor("a", vec![ForecastLaw::gaussian(0.0, 0.0); n], Horizon::Contemporaneous)],
        ValidationDesign::HeldOutSplit,
        vec![ScoreKind::Crps],
    );
    let (code, slot) = refusal(assess_predictive_laws(&bad, &ctx()));
    assert_eq!((code.as_str(), slot.as_str()), ("invalid_argument", detail::INVALID_LAW));

    // No scores requested, and a cancelled context.
    let empty = PredictiveAssessmentSpec::new(
        observations(gaussian_outcomes(n, 1), Horizon::Contemporaneous),
        vec![predictor("a", gaussian_laws(n, 0.0, 1.0), Horizon::Contemporaneous)],
        ValidationDesign::HeldOutSplit,
        Vec::new(),
    );
    assert!(assess_predictive_laws(&empty, &ctx()).is_err());
    let cancelled = ctx();
    cancelled.cancellation.cancel();
    let (code, slot) = refusal(assess_predictive_laws(&generator_spec(10), &cancelled));
    assert_eq!((code.as_str(), slot.as_str()), ("cancelled_no_claim", detail::CANCELLED));
}

#[test]
fn a_rolling_origin_design_must_match_the_target_horizon() {
    // The observations and laws are one step ahead; the design evaluates three steps ahead.
    let n = 6;
    let mut spec = PredictiveAssessmentSpec::new(
        observations(gaussian_outcomes(n, 13), Horizon::Steps(1)),
        vec![predictor("a", gaussian_laws(n, 0.0, 1.0), Horizon::Steps(1))],
        ValidationDesign::RollingOrigin { origins: (1..=6).collect(), horizon_steps: 3 },
        vec![ScoreKind::Crps],
    );
    spec.dependence =
        Some(DependenceDeclaration::Serial { time_index: (1..=6).collect(), truncation_lag: 2 });
    let (code, slot) = refusal(assess_predictive_laws(&spec, &ctx()));
    assert_eq!(
        (code.as_str(), slot.as_str()),
        ("quantity_semantics_mismatch", detail::HORIZON_MISMATCH)
    );
}

#[test]
fn a_malformed_diagnostic_configuration_is_refused() {
    let n = 5;
    let build = |pit_bins: usize, levels: Vec<f64>| {
        let mut spec = PredictiveAssessmentSpec::new(
            observations(gaussian_outcomes(n, 21), Horizon::Contemporaneous),
            vec![predictor("a", gaussian_laws(n, 0.0, 1.0), Horizon::Contemporaneous)],
            ValidationDesign::HeldOutSplit,
            vec![ScoreKind::Crps],
        );
        spec.pit_bins = pit_bins;
        spec.interval_levels = levels;
        spec
    };
    for spec in
        [build(1, vec![0.5]), build(10, vec![1.5]), build(10, vec![0.0]), build(5000, vec![])]
    {
        let (code, slot) = refusal(assess_predictive_laws(&spec, &ctx()));
        assert_eq!(code, "invalid_argument");
        assert_eq!(slot, detail::DIAGNOSTIC_NOT_DEFINED);
    }
    // No interval levels is a valid request: coverage is simply absent.
    let report = assess_predictive_laws(&build(4, vec![]), &ctx()).unwrap();
    assert!(report.predictors[0].diagnostics.coverage.is_none());
}

#[test]
fn declared_bounds_on_predictors_atoms_and_joint_dimension_are_refused() {
    let one = |name: String| predictor(&name, gaussian_laws(1, 0.0, 1.0), Horizon::Contemporaneous);
    let spec = PredictiveAssessmentSpec::new(
        observations(gaussian_outcomes(1, 31), Horizon::Contemporaneous),
        (0..33).map(|i| one(format!("p{i}"))).collect(),
        ValidationDesign::HeldOutSplit,
        vec![ScoreKind::Crps],
    );
    let (code, slot) = refusal(assess_predictive_laws(&spec, &ctx()));
    assert_eq!((code.as_str(), slot.as_str()), ("route_not_supported", detail::BOUNDS_EXCEEDED));

    let too_many_draws = ForecastLaw::EmpiricalSample { draws: vec![0.0; 65_537] };
    let spec = PredictiveAssessmentSpec::new(
        observations(gaussian_outcomes(1, 32), Horizon::Contemporaneous),
        vec![predictor("big", vec![too_many_draws], Horizon::Contemporaneous)],
        ValidationDesign::HeldOutSplit,
        vec![ScoreKind::Crps],
    );
    let (code, slot) = refusal(assess_predictive_laws(&spec, &ctx()));
    assert_eq!((code.as_str(), slot.as_str()), ("invalid_argument", detail::INVALID_LAW));

    let wide = ForecastLaw::JointSample { dimension: 65, draws: vec![0.0; 65] };
    let spec = PredictiveAssessmentSpec::new(
        RealizedObservations {
            target: target(Horizon::Contemporaneous),
            ids: vec![key(0)],
            outcomes: vec![Outcome::Vector(vec![0.0; 65])],
        },
        vec![predictor("wide", vec![wide], Horizon::Contemporaneous)],
        ValidationDesign::HeldOutSplit,
        vec![ScoreKind::Energy],
    );
    let (code, slot) = refusal(assess_predictive_laws(&spec, &ctx()));
    assert_eq!((code.as_str(), slot.as_str()), ("invalid_argument", detail::INVALID_LAW));
}
