//! Known-truth tests of exposure misclassification, soft exposures, covariate-dependent
//! detection and latent-class repeat codings.

use super::*;
use antecedent_core::CausalRng;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

fn rates(se: f64, sp: f64) -> BinaryErrorRates {
    BinaryErrorRates::new(se, sp, 0.0, 0.0).unwrap()
}

fn rates_with_variance(se: f64, sp: f64, se_var: f64, sp_var: f64) -> BinaryErrorRates {
    BinaryErrorRates::new(se, sp, se_var, sp_var).unwrap()
}

// ---------------------------------------------------------------------------
// Exposure misclassification
// ---------------------------------------------------------------------------

/// Cases Se 0.9 / Sp 0.8, controls Se 0.7 / Sp 0.95; true exposure shares 0.4 and 0.2.
/// The recorded shares are exactly 0.48 and 0.18.
fn exact_counts() -> ExposureOutcomeCounts {
    ExposureOutcomeCounts { exposed_cases: 480, cases: 1000, exposed_controls: 180, controls: 1000 }
}

#[test]
fn outcome_specific_rates_recover_the_planted_exposure_odds_ratio() {
    let structure =
        ExposureErrorStructure::ByOutcome { controls: rates(0.7, 0.95), cases: rates(0.9, 0.8) };
    let fit = corrected_exposure_association(&exact_counts(), &structure, 0.1).unwrap();
    assert!(close(fit.prevalence_cases, 0.4, 1e-12));
    assert!(close(fit.prevalence_controls, 0.2, 1e-12));
    assert!(close(fit.log_odds_ratio, (8.0_f64 / 3.0).ln(), 1e-12));
    assert!(close(fit.odds_ratio, 8.0 / 3.0, 1e-12));
    assert!(fit.differential);
    // The recorded association is attenuated towards the null here.
    let naive = fit.naive_log_odds_ratio.unwrap();
    assert!(close(naive, (0.48_f64 / 0.52).ln() - (0.18_f64 / 0.82).ln(), 1e-12));
    // Hand sampling variance: g_y^2 p_y (1 - p_y) / (n J_y^2), g = 1 / (pi (1 - pi)).
    let g1 = 1.0 / (0.4 * 0.6);
    let g0 = 1.0 / (0.2 * 0.8);
    let expected = g1 * g1 * 0.48 * 0.52 / 1000.0 / (0.7 * 0.7)
        + g0 * g0 * 0.18 * 0.82 / 1000.0 / (0.65 * 0.65);
    assert!(close(fit.variance_sampling, expected, 1e-12), "{fit:?}");
    assert!(fit.variance_rates == 0.0);
}

#[test]
fn applying_one_pair_of_rates_to_error_that_differs_by_outcome_misses_the_truth() {
    let wrong = ExposureErrorStructure::Shared(rates(0.8, 0.9));
    let fit = corrected_exposure_association(&exact_counts(), &wrong, 0.1).unwrap();
    assert!(!fit.differential);
    assert!((fit.log_odds_ratio - (8.0_f64 / 3.0).ln()).abs() > 0.3, "{fit:?}");
}

/// The corrected log odds ratio as an explicit function of the recorded shares and the rates
/// `[se_cases, sp_cases, se_controls, sp_controls]`, written independently of the library.
fn log_or_by_hand(p1: f64, p0: f64, r: [f64; 4]) -> f64 {
    let pi1 = (p1 + r[1] - 1.0) / (r[0] + r[1] - 1.0);
    let pi0 = (p0 + r[3] - 1.0) / (r[2] + r[3] - 1.0);
    (pi1 / (1.0 - pi1)).ln() - (pi0 / (1.0 - pi0)).ln()
}

#[test]
fn delta_method_variance_equals_a_numerical_gradient_for_both_structures() {
    let (se1, sp1, se0, sp0) = (0.9, 0.8, 0.7, 0.95);
    let variances = [4e-4, 9e-4, 6e-4, 2e-4];
    let counts = exact_counts();
    let structure = ExposureErrorStructure::ByOutcome {
        controls: rates_with_variance(se0, sp0, variances[2], variances[3]),
        cases: rates_with_variance(se1, sp1, variances[0], variances[1]),
    };
    let fit = corrected_exposure_association(&counts, &structure, 0.1).unwrap();
    let base = [se1, sp1, se0, sp0];
    let (p1, p0) = (0.48, 0.18);
    let h = 1e-6;
    let mut expected_rates = 0.0;
    for (slot, variance) in variances.iter().enumerate() {
        let (mut up, mut down) = (base, base);
        up[slot] += h;
        down[slot] -= h;
        let d = (log_or_by_hand(p1, p0, up) - log_or_by_hand(p1, p0, down)) / (2.0 * h);
        expected_rates += d * d * variance;
    }
    assert!(
        close(fit.variance_rates, expected_rates, 1e-8),
        "{} vs {expected_rates}",
        fit.variance_rates
    );
    let d_p1 = (log_or_by_hand(p1 + h, p0, base) - log_or_by_hand(p1 - h, p0, base)) / (2.0 * h);
    let d_p0 = (log_or_by_hand(p1, p0 + h, base) - log_or_by_hand(p1, p0 - h, base)) / (2.0 * h);
    let expected_sampling =
        d_p1 * d_p1 * p1 * (1.0 - p1) / 1000.0 + d_p0 * d_p0 * p0 * (1.0 - p0) / 1000.0;
    assert!(close(fit.variance_sampling, expected_sampling, 1e-8));

    // One shared pair: the derivatives in the shared rates add before they are squared.
    let shared = ExposureErrorStructure::Shared(rates_with_variance(0.8, 0.9, 4e-4, 9e-4));
    let fit = corrected_exposure_association(&counts, &shared, 0.1).unwrap();
    let by_hand = |se: f64, sp: f64| log_or_by_hand(p1, p0, [se, sp, se, sp]);
    let d_se = (by_hand(0.8 + h, 0.9) - by_hand(0.8 - h, 0.9)) / (2.0 * h);
    let d_sp = (by_hand(0.8, 0.9 + h) - by_hand(0.8, 0.9 - h)) / (2.0 * h);
    let expected = d_se * d_se * 4e-4 + d_sp * d_sp * 9e-4;
    assert!(close(fit.variance_rates, expected, 1e-8), "{} vs {expected}", fit.variance_rates);
}

#[test]
fn a_youden_index_at_or_below_the_floor_in_either_stratum_refuses() {
    let weak_controls =
        ExposureErrorStructure::ByOutcome { controls: rates(0.55, 0.5), cases: rates(0.9, 0.8) };
    let err = corrected_exposure_association(&exact_counts(), &weak_controls, 0.1).unwrap_err();
    assert!(matches!(err, MeasurementErrorError::YoudenTooSmall { .. }), "{err:?}");
    let weak_cases =
        ExposureErrorStructure::ByOutcome { controls: rates(0.7, 0.95), cases: rates(0.5, 0.5) };
    assert!(matches!(
        corrected_exposure_association(&exact_counts(), &weak_cases, 0.1),
        Err(MeasurementErrorError::YoudenTooSmall { .. })
    ));
}

#[test]
fn validation_samples_give_binomial_rate_variances_per_outcome_level() {
    let structure = exposure_rates_from_validation((70, 30, 190, 10), (90, 10, 80, 20)).unwrap();
    let ExposureErrorStructure::ByOutcome { controls, cases } = structure else {
        panic!("outcome-specific rates expected");
    };
    assert!(close(controls.sensitivity, 0.7, 1e-12));
    assert!(close(controls.specificity, 0.95, 1e-12));
    assert!(close(controls.sensitivity_variance, 0.7 * 0.3 / 100.0, 1e-15));
    assert!(close(cases.specificity_variance, 0.8 * 0.2 / 100.0, 1e-15));
    assert!(exposure_rates_from_validation((0, 0, 5, 5), (1, 1, 1, 1)).is_err());
}

// ---------------------------------------------------------------------------
// Soft exposures
// ---------------------------------------------------------------------------

fn uniform_stream(seed: u64) -> impl FnMut() -> f64 {
    let mut rng = CausalRng::from_seed(seed);
    move || rng.next_f64()
}

fn normal_from(next: &mut impl FnMut() -> f64) -> f64 {
    let u1 = 1.0 - next();
    let u2 = next();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// `q ~ U(0.05, 0.95)`, `X ~ Bernoulli(q)`, `Y = 1 + 2 X + 0.5 Z + N(0, 1)`.
struct SoftSample {
    y: Vec<f64>,
    q: Vec<f64>,
    z: Vec<f64>,
}

fn soft_sample(n: usize, seed: u64) -> SoftSample {
    let mut next = uniform_stream(seed);
    let (mut y, mut q, mut z) = (vec![], vec![], vec![]);
    for _ in 0..n {
        let qi = 0.05 + 0.9 * next();
        let xi = if next() < qi { 1.0 } else { 0.0 };
        let zi = normal_from(&mut next);
        y.push(1.0 + 2.0 * xi + 0.5 * zi + normal_from(&mut next));
        q.push(qi);
        z.push(zi);
    }
    SoftSample { y, q, z }
}

#[test]
fn posterior_mean_substitution_matches_least_squares_on_the_probabilities() {
    let s = soft_sample(4000, 5);
    let data =
        SoftExposureData { outcome: &s.y, probability: &s.q, covariates: &s.z, covariate_count: 1 };
    let fit = soft_exposure_substitution(&data).unwrap();
    // Independent normal equations by Gauss-Jordan on (1, q, z).
    let n = s.y.len();
    let mut a = vec![vec![0.0; 4]; 3];
    for i in 0..n {
        let v = [1.0, s.q[i], s.z[i]];
        for r in 0..3 {
            for c in 0..3 {
                a[r][c] += v[r] * v[c];
            }
            a[r][3] += v[r] * s.y[i];
        }
    }
    for p in 0..3 {
        let pivot = a[p][p];
        for c in 0..4 {
            a[p][c] /= pivot;
        }
        for r in 0..3 {
            if r != p {
                let f = a[r][p];
                for c in 0..4 {
                    a[r][c] -= f * a[p][c];
                }
            }
        }
    }
    for k in 0..3 {
        assert!(close(fit.coefficients[k], a[k][3], 1e-8), "{k}: {:?}", fit.coefficients);
    }
    // The exposure-uncertainty term is a share of, never more than, the total.
    assert!(fit.variance_exposure_uncertainty >= 0.0);
    assert!(fit.variance_exposure_uncertainty <= fit.variance);
    assert!(close(fit.variance_exposure_uncertainty + fit.variance_sampling, fit.variance, 1e-15));
    assert!(close(
        fit.mean_posterior_variance,
        s.q.iter().map(|q| q * (1.0 - q)).sum::<f64>() / n as f64,
        1e-12
    ));
}

#[test]
fn substitution_on_calibrated_probabilities_recovers_the_coefficient_where_a_draw_does_not() {
    // Berkson structure: the slope on q is exactly the slope on the exposure when q is the
    // exposure's conditional mean (here planted, with a large sample).
    let s = soft_sample(60_000, 17);
    let data =
        SoftExposureData { outcome: &s.y, probability: &s.q, covariates: &s.z, covariate_count: 1 };
    let fit = soft_exposure_substitution(&data).unwrap();
    assert!(close(fit.coefficients[1], 2.0, 0.08), "{:?}", fit.coefficients);
    // The same regression on one 0/1 draw given q alone is attenuated.
    let mut next = uniform_stream(99);
    let draw: Vec<f64> = s.q.iter().map(|q| if next() < *q { 1.0 } else { 0.0 }).collect();
    let noisy = SoftExposureData {
        outcome: &s.y,
        probability: &draw,
        covariates: &s.z,
        covariate_count: 1,
    };
    let attenuated = soft_exposure_substitution(&noisy).unwrap();
    assert!(attenuated.coefficients[1] < 1.5, "{:?}", attenuated.coefficients);
}

#[test]
fn imputation_over_outcome_conditioned_draws_recovers_the_coefficient_with_rubin_variance() {
    let s = soft_sample(3000, 23);
    let data =
        SoftExposureData { outcome: &s.y, probability: &s.q, covariates: &s.z, covariate_count: 1 };
    let draws = soft_exposure_outcome_conditioned_draws(&data, 30, 4).unwrap();
    assert_eq!(draws.draws.len(), 30 * s.y.len());
    assert!(draws.draws.iter().all(|d| *d == 0.0 || *d == 1.0));
    let combined = soft_exposure_imputation(&s.y, &draws.draws, 30, &s.z, 1).unwrap();
    let d = combined.dim;
    // Rubin: total = within + (1 + 1/S) between.
    let total = combined.within[d + 1] + (1.0 + 1.0 / 30.0) * combined.between[d + 1];
    assert!(close(combined.covariance[d + 1], total, 1e-12));
    // Reproducible from the seed.
    let again = soft_exposure_outcome_conditioned_draws(&data, 30, 4).unwrap();
    assert_eq!(draws.draws, again.draws);
    // A single imputation cannot be combined.
    assert!(soft_exposure_outcome_conditioned_draws(&data, 1, 4).is_err());
}

#[test]
fn reliability_reports_a_calibrated_and_a_miscalibrated_probability_set() {
    // Calibrated: every unit's expected coding share equals its stated probability.
    // Ten units per bin at q = 0.05, 0.15, ...; each unit has 4 codings whose share equals q
    // rounded to the nearest quarter would not be exact, so use 20 codings per unit with
    // exactly round(20 q) ones.
    let mut probability = Vec::new();
    let mut codings: Vec<u8> = Vec::new();
    let mut sizes = Vec::new();
    for b in 0..10 {
        let q = 0.05 + 0.1 * b as f64;
        for _ in 0..10 {
            probability.push(q);
            let ones = (20.0 * q).round() as usize;
            codings.extend((0..20).map(|k| u8::from(k < ones)));
            sizes.push(20);
        }
    }
    let report = soft_exposure_reliability(&probability, &codings, &sizes, 10, None).unwrap();
    assert_eq!(report.bins.len(), 10);
    assert_eq!(report.units, 100);
    // Rounding 20 q to an integer leaves gaps of at most 0.025.
    assert!(report.max_abs_gap <= 0.025 + 1e-12, "{}", report.max_abs_gap);
    // Overconfident probabilities: stated 0.9 for units whose codings are 0.5 ones.
    let mut probability = vec![0.9; 40];
    probability.extend(vec![0.1; 40]);
    let mut codings = Vec::new();
    for _ in 0..80 {
        codings.extend([1_u8, 0, 1, 0]);
    }
    let sizes = vec![4_u32; 80];
    let report = soft_exposure_reliability(&probability, &codings, &sizes, 10, None).unwrap();
    assert_eq!(report.bins.len(), 2);
    assert!(close(report.max_abs_gap, 0.4, 1e-12));
    // A declared reference coding with sensitivity 0.5 and specificity 0.5 expects 0.5 at any q.
    let reference = rates(0.5, 0.5);
    let report =
        soft_exposure_reliability(&probability, &codings, &sizes, 10, Some(&reference)).unwrap();
    assert!(report.max_abs_gap < 1e-12);
    // Malformed inputs refuse.
    assert!(soft_exposure_reliability(&[0.5], &[2], &[1], 5, None).is_err());
    assert!(soft_exposure_reliability(&[0.5], &[1], &[0], 5, None).is_err());
}

// ---------------------------------------------------------------------------
// Detection that depends on evidence density
// ---------------------------------------------------------------------------

/// `per` units at each covariate value 0..=3; `detected_at[c]` of them recorded as present.
fn detection_rows(per: usize, detected_at: [usize; 4]) -> (Vec<bool>, Vec<f64>) {
    let mut detected = Vec::new();
    let mut covariate = Vec::new();
    for (c, &hits) in detected_at.iter().enumerate() {
        for k in 0..per {
            detected.push(k < hits);
            covariate.push(c as f64);
        }
    }
    (detected, covariate)
}

fn table() -> DetectionFunction {
    DetectionFunction::Table { knots: vec![0.0, 3.0], values: vec![0.4, 1.0] }
}

#[test]
fn a_fully_declared_detection_function_recovers_the_planted_prevalence() {
    // d(c) = 0.4, 0.6, 0.8, 1.0; prevalence 0.3 gives 120, 180, 240, 300 detections of 1000.
    let (detected, covariate) = detection_rows(1000, [120, 180, 240, 300]);
    let data = DetectionData { detected: &detected, covariate: &covariate };
    let result = detection_adjusted_prevalence(
        &data,
        &DetectionDeclaration::Full(table()),
        &FalsePositiveRate::NONE,
        0.1,
    )
    .unwrap();
    let DetectionAdjusted::Point(point) = result else { panic!("a point is expected") };
    assert!(close(point.prevalence, 0.3, 1e-12));
    assert!(close(point.observed_rate, 0.21, 1e-12));
    assert!(close(point.mean_detection, 0.7, 1e-12));
    // Hand Bernoulli variance: sum p_i (1 - p_i) / G^2 with p_i = 0.3 d_i, G = 2800.
    let by_hand: f64 =
        [0.4, 0.6, 0.8, 1.0].iter().map(|d| 1000.0 * (0.3 * d) * (1.0 - 0.3 * d)).sum::<f64>()
            / (2800.0 * 2800.0);
    assert!(close(point.variance_sampling, by_hand, 1e-15), "{point:?}");
    assert!(point.variance_parameters == 0.0);
}

#[test]
fn a_false_positive_rate_shifts_the_inversion_and_adds_its_variance() {
    // Detected share = pi d + (1 - pi) f with f = 0.05, pi = 0.3: 0.3 d + 0.035.
    let hits = [155, 215, 275, 335];
    let (detected, covariate) = detection_rows(1000, hits);
    let data = DetectionData { detected: &detected, covariate: &covariate };
    let fp = FalsePositiveRate { rate: 0.05, variance: 1e-4 };
    let result =
        detection_adjusted_prevalence(&data, &DetectionDeclaration::Full(table()), &fp, 0.1)
            .unwrap();
    let DetectionAdjusted::Point(point) = result else { panic!("a point is expected") };
    assert!(close(point.prevalence, 0.3, 1e-12));
    // d pi / d f = -n (1 - pi) / G with G = sum d - n f = 2800 - 200.
    let slope = -4000.0 * 0.7 / 2600.0;
    assert!(close(point.variance_parameters, slope * slope * 1e-4, 1e-12), "{point:?}");
}

#[test]
fn logistic_detection_variance_equals_a_numerical_gradient_of_the_estimator() {
    let covariance = [[4e-3, 1e-3], [1e-3, 2e-3]];
    let (detected, covariate) = detection_rows(500, [60, 110, 190, 240]);
    let data = DetectionData { detected: &detected, covariate: &covariate };
    let function = |a: f64, b: f64| DetectionFunction::Logistic {
        intercept: a,
        slope: b,
        intercept_variance: covariance[0][0],
        covariance: covariance[0][1],
        slope_variance: covariance[1][1],
    };
    let prevalence = |a: f64, b: f64| {
        let r = detection_adjusted_prevalence(
            &data,
            &DetectionDeclaration::Full(function(a, b)),
            &FalsePositiveRate::NONE,
            0.05,
        )
        .unwrap();
        let DetectionAdjusted::Point(p) = r else { panic!("a point is expected") };
        p
    };
    let (a, b) = (-1.0, 0.9);
    let h = 1e-6;
    let da = (prevalence(a + h, b).prevalence - prevalence(a - h, b).prevalence) / (2.0 * h);
    let db = (prevalence(a, b + h).prevalence - prevalence(a, b - h).prevalence) / (2.0 * h);
    let expected =
        da * da * covariance[0][0] + 2.0 * da * db * covariance[0][1] + db * db * covariance[1][1];
    let got = prevalence(a, b);
    assert!(
        close(got.variance_parameters, expected, 1e-10),
        "{} vs {expected}",
        got.variance_parameters
    );
}

#[test]
fn a_partially_declared_monotone_function_gives_sharp_bounds_that_contain_the_truth() {
    let (detected, covariate) = detection_rows(1000, [120, 180, 240, 300]);
    let data = DetectionData { detected: &detected, covariate: &covariate };
    let declaration = DetectionDeclaration::Partial {
        direction: DetectionDirection::Nondecreasing,
        anchors: vec![
            DetectionAnchor { covariate: 0.0, lower: 0.3, upper: 0.5 },
            DetectionAnchor { covariate: 3.0, lower: 0.9, upper: 1.0 },
        ],
    };
    let result =
        detection_adjusted_prevalence(&data, &declaration, &FalsePositiveRate::NONE, 0.1).unwrap();
    let DetectionAdjusted::Bounds(bounds) = result else { panic!("bounds are expected") };
    // lower d = (0.3, 0.3, 0.3, 0.9), upper d = (0.5, 1, 1, 1): means 0.45 and 0.875.
    assert!(close(bounds.mean_detection_lower, 0.45, 1e-12));
    assert!(close(bounds.mean_detection_upper, 0.875, 1e-12));
    assert!(close(bounds.lower, 840.0 / 3500.0, 1e-12));
    assert!(close(bounds.upper, 840.0 / 1800.0, 1e-12));
    assert!(bounds.lower < 0.3 && 0.3 < bounds.upper);
    // Tightening the declaration to the truth collapses the bounds onto the point.
    let exact = DetectionDeclaration::Partial {
        direction: DetectionDirection::Nondecreasing,
        anchors: vec![
            DetectionAnchor { covariate: 0.0, lower: 0.4, upper: 0.4 },
            DetectionAnchor { covariate: 1.0, lower: 0.6, upper: 0.6 },
            DetectionAnchor { covariate: 2.0, lower: 0.8, upper: 0.8 },
            DetectionAnchor { covariate: 3.0, lower: 1.0, upper: 1.0 },
        ],
    };
    let r = detection_adjusted_prevalence(&data, &exact, &FalsePositiveRate::NONE, 0.1).unwrap();
    let DetectionAdjusted::Bounds(collapsed) = r else { panic!("bounds are expected") };
    assert!(close(collapsed.lower, 0.3, 1e-12) && close(collapsed.upper, 0.3, 1e-12));
}

#[test]
fn detection_that_cannot_identify_or_contradicts_itself_refuses() {
    let (detected, covariate) = detection_rows(100, [10, 10, 10, 10]);
    let data = DetectionData { detected: &detected, covariate: &covariate };
    let weak = DetectionFunction::Table { knots: vec![0.0, 3.0], values: vec![0.05, 0.08] };
    assert!(matches!(
        detection_adjusted_prevalence(
            &data,
            &DetectionDeclaration::Full(weak),
            &FalsePositiveRate::NONE,
            0.1
        ),
        Err(MeasurementErrorError::YoudenTooSmall { .. })
    ));
    let contradictory = DetectionDeclaration::Partial {
        direction: DetectionDirection::Nondecreasing,
        anchors: vec![
            DetectionAnchor { covariate: 0.0, lower: 0.8, upper: 0.9 },
            DetectionAnchor { covariate: 1.0, lower: 0.1, upper: 0.2 },
        ],
    };
    assert!(
        detection_adjusted_prevalence(&data, &contradictory, &FalsePositiveRate::NONE, 0.1)
            .is_err()
    );
    // Detection that would need a prevalence above one.
    let (all, cov) = detection_rows(100, [100, 100, 100, 100]);
    let impossible = DetectionDeclaration::Full(DetectionFunction::Table {
        knots: vec![0.0, 3.0],
        values: vec![0.4, 0.4],
    });
    let d = DetectionData { detected: &all, covariate: &cov };
    let r = detection_adjusted_prevalence(&d, &impossible, &FalsePositiveRate::NONE, 0.1).unwrap();
    let DetectionAdjusted::Point(p) = r else { panic!("a point is expected") };
    assert!(p.outside_unit_interval);
}

// ---------------------------------------------------------------------------
// Latent-class repeat codings
// ---------------------------------------------------------------------------

/// Rows from exact pattern probabilities: `n * P(pattern)` rounded to whole units.
fn rows_from_model(
    n: usize,
    pi: &[f64],
    theta: &[Vec<Vec<f64>>],
    categories: usize,
) -> (Vec<u8>, usize) {
    let k = theta.len();
    let mut codings = Vec::new();
    let mut units = 0;
    let patterns = categories.pow(k as u32);
    for pattern in 0..patterns {
        let digits: Vec<usize> =
            (0..k).map(|c| (pattern / categories.pow((k - 1 - c) as u32)) % categories).collect();
        let probability: f64 = pi
            .iter()
            .enumerate()
            .map(|(level, p)| p * (0..k).map(|c| theta[c][level][digits[c]]).product::<f64>())
            .sum();
        let count = (n as f64 * probability).round() as usize;
        for _ in 0..count {
            codings.extend(digits.iter().map(|d| *d as u8));
            units += 1;
        }
    }
    (codings, units)
}

#[test]
fn three_binary_codings_recover_the_planted_latent_class_model() {
    let pi = [0.7, 0.3];
    let (se, sp) = ([0.9, 0.8, 0.85], [0.95, 0.9, 0.85]);
    let theta: Vec<Vec<Vec<f64>>> =
        (0..3).map(|k| vec![vec![sp[k], 1.0 - sp[k]], vec![1.0 - se[k], se[k]]]).collect();
    let (codings, units) = rows_from_model(60_000, &pi, &theta, 2);
    let data = LatentClassData {
        codings: &codings,
        units,
        coding_count: 3,
        categories: 2,
        latent_levels: 2,
    };
    let fit = fit_latent_class(
        &data,
        &LatentClassConstraints::default(),
        &LatentClassSettings::default(),
    )
    .unwrap();
    assert!(fit.identification.kruskal);
    assert!(close(fit.prevalence[1], 0.3, 2e-3), "{:?}", fit.prevalence);
    for k in 0..3 {
        assert!(
            close(fit.response_of(k, 1, 1), se[k], 3e-3),
            "se {k}: {}",
            fit.response_of(k, 1, 1)
        );
        assert!(
            close(fit.response_of(k, 0, 0), sp[k], 3e-3),
            "sp {k}: {}",
            fit.response_of(k, 0, 0)
        );
    }
    let se_report = fit.standard_errors.as_ref().expect("the score matrix is invertible");
    assert!(se_report.prevalence[1] > 0.0 && se_report.prevalence[1] < 0.01);
    // The posterior of a unit that every coding marks present favours the upper level.
    let all_ones = (0..units).find(|i| codings[i * 3..i * 3 + 3] == [1, 1, 1]).unwrap();
    assert!(fit.posterior[all_ones * 2 + 1] > 0.9);
}

#[test]
fn ordinal_codings_recover_a_three_level_facet() {
    let pi = [0.5, 0.3, 0.2];
    let law = vec![vec![0.8, 0.15, 0.05], vec![0.15, 0.7, 0.15], vec![0.05, 0.15, 0.8]];
    let theta = vec![law.clone(), law.clone(), law];
    let (codings, units) = rows_from_model(90_000, &pi, &theta, 3);
    let data = LatentClassData {
        codings: &codings,
        units,
        coding_count: 3,
        categories: 3,
        latent_levels: 3,
    };
    let fit = fit_latent_class(
        &data,
        &LatentClassConstraints::default(),
        &LatentClassSettings::default(),
    )
    .unwrap();
    for (level, truth) in pi.iter().enumerate() {
        assert!(close(fit.prevalence[level], *truth, 1e-2), "{:?}", fit.prevalence);
    }
    assert!(close(fit.response_of(0, 2, 2), 0.8, 1e-2));
}

#[test]
fn two_codings_identify_only_under_declared_constraints() {
    let pi = [0.7, 0.3];
    let law = vec![vec![0.95, 0.05], vec![0.1, 0.9]];
    let theta = vec![law.clone(), law];
    let (codings, units) = rows_from_model(100_000, &pi, &theta, 2);
    let data = LatentClassData {
        codings: &codings,
        units,
        coding_count: 2,
        categories: 2,
        latent_levels: 2,
    };
    for constraints in [
        LatentClassConstraints::default(),
        LatentClassConstraints { exchangeable: true, lowest_level: None },
        LatentClassConstraints { exchangeable: false, lowest_level: Some(vec![0.95, 0.05]) },
    ] {
        assert!(matches!(
            fit_latent_class(&data, &constraints, &LatentClassSettings::default()),
            Err(MeasurementErrorError::InsufficientReplicates(_))
        ));
    }
    let constraints =
        LatentClassConstraints { exchangeable: true, lowest_level: Some(vec![0.95, 0.05]) };
    let fit = fit_latent_class(&data, &constraints, &LatentClassSettings::default()).unwrap();
    assert!(fit.identification.constrained);
    assert!(close(fit.prevalence[1], 0.3, 3e-3), "{:?}", fit.prevalence);
    assert!(close(fit.response_of(0, 1, 1), 0.9, 3e-3));
    assert!(close(fit.response_of(1, 1, 1), 0.9, 3e-3));
}

#[test]
fn kruskal_condition_counts_the_codings_each_design_needs() {
    assert!(!kruskal_identified(2, 2, 2));
    assert!(kruskal_identified(2, 2, 3));
    assert!(kruskal_identified(3, 3, 3));
    assert!(!kruskal_identified(3, 2, 3));
    assert!(!kruskal_identified(3, 2, 4));
    assert!(kruskal_identified(3, 2, 5));
}

#[test]
fn codings_that_do_not_separate_the_classes_refuse() {
    // Three independent fair coins carry no information about any latent class.
    let (codings, units) =
        rows_from_model(40_000, &[0.5, 0.5], &vec![vec![vec![0.5, 0.5], vec![0.5, 0.5]]; 3], 2);
    let data = LatentClassData {
        codings: &codings,
        units,
        coding_count: 3,
        categories: 2,
        latent_levels: 2,
    };
    assert!(
        fit_latent_class(
            &data,
            &LatentClassConstraints::default(),
            &LatentClassSettings::default()
        )
        .is_err()
    );
}
