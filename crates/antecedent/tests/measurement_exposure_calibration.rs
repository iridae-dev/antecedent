//! Repeated-sampling coverage of the intervals published by the exposure-error routes, each
//! scored against an analytically known truth at realistic validation and replicate sizes.
//!
//! Every test scores the Wald interval `point -/+ z * se` the route reports at 0.95 and binds
//! the construction it reports. A replicate whose correction refuses is a skip and counts as a
//! miss.
//!
//! * exposure misclassification that differs by outcome: true exposure shares 0.4 (cases) and
//!   0.2 (controls), sensitivities 0.9 and 0.7 and specificities 0.8 and 0.95, each estimated
//!   from its own validation sample of `m` truly exposed and `m` truly unexposed units, truth
//!   the log odds ratio `ln(8/3)`; and the same with one shared pair of rates (sensitivity 0.85,
//!   specificity 0.9, shares 0.35 and 0.2) from one validation sample;
//! * per-unit soft exposures: `q ~ U(0.05, 0.95)`, `X ~ Bernoulli(q)`,
//!   `Y = 1 + 2 X + 0.5 Z + N(0, 1)`, truth the exposure coefficient 2, by posterior-mean
//!   substitution (HC0 variance) and by multiple imputation over outcome-conditioned draws
//!   (Rubin's total variance);
//! * detection that depends on a count: a condition of prevalence 0.3 detected with probability
//!   `expit(-1 + 0.8 c)` at count `c`, the function declared exactly and with its two
//!   parameters estimated (their covariance propagated), truth 0.3;
//! * repeat codings as replicates: three binary codings of a binary facet (prevalence 0.3) and
//!   three ordinal codings of a three-level facet (top level 0.2), each under conditional
//!   independence, scored on the prevalence of the highest latent level with the delta-method
//!   interval of the outer-product covariance.
//!
//! The routes publish no runtime calibration key for the interval, so their constructions are
//! declared here in the form the other facade-level harnesses without a runtime key use.
//!
//! Ignored tests run via `scripts/measure_calibration.sh`.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    reason = "bounded replicate indices in a calibration fixture"
)]
#![allow(clippy::cast_precision_loss, reason = "bounded simulation sizes and small indices")]
#![allow(
    clippy::needless_range_loop,
    reason = "simulation fixtures index parallel columns by row and draw"
)]

mod common;

use antecedent_core::ExecutionContext;
use antecedent_estimate::measurement::{
    SoftExposureCombination, correct_detection_dependent_prevalence,
    correct_exposure_misclassification, correct_soft_exposure_effect, fit_repeat_codings,
};
use antecedent_stats::measurement_error::{
    BinaryErrorRates, DetectionAdjusted, DetectionData, DetectionDeclaration, DetectionFunction,
    ExposureErrorStructure, ExposureOutcomeCounts, FalsePositiveRate, LatentClassConstraints,
    LatentClassData, LatentClassSettings, SoftExposureData,
};
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, gaussian, grid_n,
    grid_seed, map_replicates, n_sim, uniform,
};

// ---------------------------------------------------------------------------
// Shared scoring
// ---------------------------------------------------------------------------

/// One replicate's reported interval, the construction it was reported under and its row count.
struct Reading {
    interval: (f64, f64),
    construction: Construction,
    rows: u64,
}

/// Score `replicate` over `n_sim()` datasets against `truth` and assert nominal coverage;
/// `None` is a skip.
fn measure(
    test: &'static str,
    dgp: &'static str,
    interval: &'static str,
    truth: f64,
    seed_base: u64,
    replicate: impl Fn(u64) -> Option<Reading> + Sync,
) {
    let mut tally = CoverageTally::for_record(RecordKey { test, dgp, interval }, REPORTED_LEVEL);
    let readings = map_replicates(n_sim(), |rep| replicate(grid_seed(seed_base + rep)));
    for reading in readings {
        let Some(reading) = reading else {
            tally.skip();
            continue;
        };
        tally.bind_validation(
            &reading.construction,
            ScopeFacts {
                row_count: reading.rows,
                replicates_ok: None,
                posterior_draws: None,
                unidentified_mass: 0.0,
            },
            "not_applicable",
        );
        tally.record(Some(reading.interval), truth);
    }
    tally.assert();
}

#[allow(
    clippy::too_many_arguments,
    reason = "one value per axis of the calibration key, all of them required"
)]
fn construction(
    estimator: &str,
    interval_method: &str,
    se_kind: &str,
    dependence: &str,
    inference: &str,
    functional: &str,
) -> Construction {
    Construction {
        query: "MeasurementCorrectedFunctional".into(),
        graph_class: "Dag".into(),
        structure: "fixed".into(),
        modality: "tabular".into(),
        inference: inference.into(),
        estimator: estimator.into(),
        interval_method: interval_method.into(),
        se_kind: se_kind.into(),
        dependence: dependence.into(),
        posterior: String::new(),
        functional: functional.into(),
        identification: "point".into(),
        reported_level: REPORTED_LEVEL,
        unit: None,
    }
}

const WALD: &str = "delta_method_wald";

fn binomial(next: &mut impl FnMut() -> f64, n: u64, p: f64) -> u64 {
    (0..n).filter(|_| next() < p).count() as u64
}

fn ctx(seed: u64) -> ExecutionContext {
    ExecutionContext::for_tests(seed)
}

fn expit(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// ---------------------------------------------------------------------------
// Exposure misclassification
// ---------------------------------------------------------------------------

/// Validation counts `(true_positive, false_negative, true_negative, false_positive)` of `m`
/// truly exposed and `m` truly unexposed units.
fn validation(
    next: &mut impl FnMut() -> f64,
    m: u64,
    sensitivity: f64,
    specificity: f64,
) -> (u64, u64, u64, u64) {
    let tp = binomial(next, m, sensitivity);
    let tn = binomial(next, m, specificity);
    (tp, m - tp, tn, m - tn)
}

fn recorded(share: f64, sensitivity: f64, specificity: f64) -> f64 {
    sensitivity * share + (1.0 - specificity) * (1.0 - share)
}

fn rates_of(counts: (u64, u64, u64, u64)) -> Option<BinaryErrorRates> {
    BinaryErrorRates::from_validation_counts(counts.0, counts.1, counts.2, counts.3).ok()
}

fn exposure_reading(
    seed: u64,
    structure: ExposureErrorStructure,
    counts: ExposureOutcomeCounts,
    dependence: &str,
) -> Option<Reading> {
    let fit = correct_exposure_misclassification(&counts, &structure, 0.1, &ctx(seed)).ok()?;
    Some(Reading {
        interval: fit.estimate.wald_interval(REPORTED_LEVEL).ok()?,
        construction: construction(
            "measurement.exposure_misclassification.corrected_log_odds_ratio",
            WALD,
            "delta_method_with_validation_variance",
            dependence,
            "Frequentist",
            "exposure_log_odds_ratio",
        ),
        rows: counts.cases + counts.controls,
    })
}

fn exposure_by_outcome_replicate(seed: u64) -> Option<Reading> {
    let (se1, sp1, se0, sp0) = (0.9, 0.8, 0.7, 0.95);
    let (share1, share0) = (0.4, 0.2);
    let m = grid_n(150) as u64;
    let per_outcome = grid_n(1000) as u64;
    let mut next = uniform(seed, 1);
    let cases = rates_of(validation(&mut next, m, se1, sp1))?;
    let controls = rates_of(validation(&mut next, m, se0, sp0))?;
    let counts = ExposureOutcomeCounts {
        exposed_cases: binomial(&mut next, per_outcome, recorded(share1, se1, sp1)),
        cases: per_outcome,
        exposed_controls: binomial(&mut next, per_outcome, recorded(share0, se0, sp0)),
        controls: per_outcome,
    };
    exposure_reading(
        seed,
        ExposureErrorStructure::ByOutcome { controls, cases },
        counts,
        "declared_differential_by_outcome",
    )
}

fn exposure_shared_replicate(seed: u64) -> Option<Reading> {
    let (sensitivity, specificity) = (0.85, 0.9);
    let (share1, share0) = (0.35, 0.2);
    let m = grid_n(200) as u64;
    let per_outcome = grid_n(1000) as u64;
    let mut next = uniform(seed, 1);
    let rates = rates_of(validation(&mut next, m, sensitivity, specificity))?;
    let counts = ExposureOutcomeCounts {
        exposed_cases: binomial(&mut next, per_outcome, recorded(share1, sensitivity, specificity)),
        cases: per_outcome,
        exposed_controls: binomial(
            &mut next,
            per_outcome,
            recorded(share0, sensitivity, specificity),
        ),
        controls: per_outcome,
    };
    exposure_reading(
        seed,
        ExposureErrorStructure::Shared(rates),
        counts,
        "declared_non_differential",
    )
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn measurement_exposure_differential_by_outcome_wald_l95() {
    measure(
        "measurement_exposure_differential_by_outcome_wald_l95",
        "exposure_error_by_outcome_validated_log_odds_ratio",
        WALD,
        (8.0_f64 / 3.0).ln(),
        0x4E01_0000,
        exposure_by_outcome_replicate,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn measurement_exposure_shared_rates_wald_l95() {
    let truth = ((0.35_f64 / 0.65) / (0.2 / 0.8)).ln();
    measure(
        "measurement_exposure_shared_rates_wald_l95",
        "exposure_error_shared_validated_log_odds_ratio",
        WALD,
        truth,
        0x4E02_0000,
        exposure_shared_replicate,
    );
}

// ---------------------------------------------------------------------------
// Soft exposures
// ---------------------------------------------------------------------------

fn soft_replicate(seed: u64, imputed: bool) -> Option<Reading> {
    let n = grid_n(if imputed { 400 } else { 600 });
    let mut next = uniform(seed, 1);
    let mut normal = gaussian(seed);
    let (mut y, mut q, mut z) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::new());
    for _ in 0..n {
        let qi = 0.05 + 0.9 * next();
        let x = if next() < qi { 1.0 } else { 0.0 };
        let zi = normal();
        y.push(1.0 + 2.0 * x + 0.5 * zi + normal());
        q.push(qi);
        z.push(zi);
    }
    let data =
        SoftExposureData { outcome: &y, probability: &q, covariates: &z, covariate_count: 1 };
    let combination = if imputed {
        SoftExposureCombination::MultipleImputation { draw_count: 30, seed }
    } else {
        SoftExposureCombination::PosteriorMean
    };
    let effect = correct_soft_exposure_effect(&data, &combination, &ctx(seed)).ok()?;
    Some(Reading {
        interval: effect.estimate.wald_interval(REPORTED_LEVEL).ok()?,
        construction: construction(
            if imputed {
                "measurement.soft_exposure.multiple_imputation"
            } else {
                "measurement.soft_exposure.posterior_mean_substitution"
            },
            WALD,
            if imputed { "rubin_total_variance" } else { "hc0_sandwich_given_posterior_means" },
            "declared_non_differential",
            "Frequentist",
            "soft_exposure_coefficient",
        ),
        rows: n as u64,
    })
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn measurement_soft_exposure_posterior_mean_wald_l95() {
    measure(
        "measurement_soft_exposure_posterior_mean_wald_l95",
        "calibrated_exposure_probability_coefficient_2",
        WALD,
        2.0,
        0x4E03_0000,
        |seed| soft_replicate(seed, false),
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn measurement_soft_exposure_multiple_imputation_wald_l95() {
    measure(
        "measurement_soft_exposure_multiple_imputation_wald_l95",
        "calibrated_exposure_probability_outcome_conditioned_draws_coefficient_2",
        WALD,
        2.0,
        0x4E04_0000,
        |seed| soft_replicate(seed, true),
    );
}

// ---------------------------------------------------------------------------
// Detection that depends on a count
// ---------------------------------------------------------------------------

fn detection_replicate(seed: u64, estimated: bool) -> Option<Reading> {
    let n = grid_n(2000);
    let (a, b) = (-1.0, 0.8);
    let prevalence = 0.3;
    let mut next = uniform(seed, 1);
    let mut detected = Vec::with_capacity(n);
    let mut covariate = Vec::with_capacity(n);
    for _ in 0..n {
        let c = (next() * 6.0).floor();
        let present = next() < prevalence;
        detected.push(present && next() < expit(a + b * c));
        covariate.push(c);
    }
    let function = if estimated {
        // The function's parameters estimated from a calibration sample of `m` units known to
        // have the condition: a logistic fit by Newton steps.
        let m = grid_n(400);
        let mut counts = vec![0_u64; 6];
        let mut hits = vec![0_u64; 6];
        for k in 0..m {
            let c = k % 6;
            counts[c] += 1;
            if next() < expit(a + b * c as f64) {
                hits[c] += 1;
            }
        }
        logistic_fit(&counts, &hits)?
    } else {
        DetectionFunction::Logistic {
            intercept: a,
            slope: b,
            intercept_variance: 0.0,
            covariance: 0.0,
            slope_variance: 0.0,
        }
    };
    let data = DetectionData { detected: &detected, covariate: &covariate };
    let result = correct_detection_dependent_prevalence(
        &data,
        &DetectionDeclaration::Full(function),
        &FalsePositiveRate::NONE,
        0.1,
        &ctx(seed),
    )
    .ok()?;
    let DetectionAdjusted::Point(_) = result.adjusted else { return None };
    Some(Reading {
        interval: result.estimate?.wald_interval(REPORTED_LEVEL).ok()?,
        construction: construction(
            "measurement.detection_dependent.declared_function",
            WALD,
            if estimated {
                "delta_method_with_declared_function_variance"
            } else {
                "bernoulli_sampling_given_declared_function"
            },
            "declared_covariate_dependent_detection",
            "Frequentist",
            "detection_adjusted_prevalence",
        ),
        rows: n as u64,
    })
}

/// Logistic fit of detection on the count from grouped successes, with the inverse information
/// as the parameter covariance.
fn logistic_fit(counts: &[u64], hits: &[u64]) -> Option<DetectionFunction> {
    let (mut a, mut b) = (0.0_f64, 0.0_f64);
    let mut info = [[0.0_f64; 2]; 2];
    for _ in 0..50 {
        let mut score = [0.0_f64; 2];
        info = [[0.0; 2]; 2];
        for (c, (&n, &h)) in counts.iter().zip(hits).enumerate() {
            let c = c as f64;
            let p = expit(a + b * c);
            let w = n as f64 * p * (1.0 - p);
            score[0] += h as f64 - n as f64 * p;
            score[1] += c * (h as f64 - n as f64 * p);
            info[0][0] += w;
            info[0][1] += w * c;
            info[1][0] += w * c;
            info[1][1] += w * c * c;
        }
        let det = info[0][0] * info[1][1] - info[0][1] * info[1][0];
        if !(det.is_finite() && det.abs() > 1e-12) {
            return None;
        }
        let step_a = (info[1][1] * score[0] - info[0][1] * score[1]) / det;
        let step_b = (-info[1][0] * score[0] + info[0][0] * score[1]) / det;
        a += step_a;
        b += step_b;
        if step_a.abs().max(step_b.abs()) < 1e-10 {
            break;
        }
    }
    let det = info[0][0] * info[1][1] - info[0][1] * info[1][0];
    if !(a.is_finite() && b.is_finite() && det.abs() > 1e-12) {
        return None;
    }
    Some(DetectionFunction::Logistic {
        intercept: a,
        slope: b,
        intercept_variance: info[1][1] / det,
        covariance: -info[0][1] / det,
        slope_variance: info[0][0] / det,
    })
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn measurement_detection_declared_function_wald_l95() {
    measure(
        "measurement_detection_declared_function_wald_l95",
        "count_dependent_detection_declared_exactly_prevalence_0_3",
        WALD,
        0.3,
        0x4E05_0000,
        |seed| detection_replicate(seed, false),
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn measurement_detection_estimated_function_wald_l95() {
    measure(
        "measurement_detection_estimated_function_wald_l95",
        "count_dependent_detection_estimated_prevalence_0_3",
        WALD,
        0.3,
        0x4E06_0000,
        |seed| detection_replicate(seed, true),
    );
}

// ---------------------------------------------------------------------------
// Repeat codings as replicates
// ---------------------------------------------------------------------------

/// Draw a category from `law` with the uniform `u`.
fn category(law: &[f64], u: f64) -> u8 {
    let mut acc = 0.0;
    for (k, p) in law.iter().enumerate() {
        acc += p;
        if u < acc {
            return k as u8;
        }
    }
    (law.len() - 1) as u8
}

fn latent_class_replicate(seed: u64, ordinal: bool) -> Option<Reading> {
    let mut next = uniform(seed, 1);
    let (n, levels, categories, prevalence): (usize, usize, usize, Vec<f64>) = if ordinal {
        (grid_n(3000), 3, 3, vec![0.5, 0.3, 0.2])
    } else {
        (grid_n(1500), 2, 2, vec![0.7, 0.3])
    };
    // Class-conditional response laws by coding and latent level.
    let laws: Vec<Vec<Vec<f64>>> = if ordinal {
        let law = vec![vec![0.8, 0.15, 0.05], vec![0.15, 0.7, 0.15], vec![0.05, 0.15, 0.8]];
        vec![law.clone(), law.clone(), law]
    } else {
        let (se, sp) = ([0.9, 0.85, 0.8], [0.95, 0.9, 0.9]);
        (0..3).map(|k| vec![vec![sp[k], 1.0 - sp[k]], vec![1.0 - se[k], se[k]]]).collect()
    };
    let mut codings = Vec::with_capacity(n * 3);
    for _ in 0..n {
        let level = category(&prevalence, next()) as usize;
        for k in 0..3 {
            codings.push(category(&laws[k][level], next()));
        }
    }
    let data = LatentClassData {
        codings: &codings,
        units: n,
        coding_count: 3,
        categories,
        latent_levels: levels,
    };
    let fit = fit_repeat_codings(
        &data,
        &LatentClassConstraints::default(),
        &LatentClassSettings { seed, ..LatentClassSettings::default() },
        &ctx(seed),
    )
    .ok()?;
    let errors = fit.standard_errors.as_ref()?;
    let top = levels - 1;
    let z = antecedent_stats::special::normal_ppf(0.5 + REPORTED_LEVEL / 2.0);
    let (point, se) = (fit.prevalence[top], errors.prevalence[top]);
    Some(Reading {
        interval: (point - z * se, point + z * se),
        construction: construction(
            if ordinal {
                "measurement.repeat_codings.ordinal_latent_class"
            } else {
                "measurement.repeat_codings.binary_latent_class"
            },
            WALD,
            "outer_product_score_covariance",
            "declared_conditional_independence",
            "Frequentist",
            "latent_level_prevalence",
        ),
        rows: n as u64,
    })
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn measurement_repeat_codings_binary_latent_class_wald_l95() {
    measure(
        "measurement_repeat_codings_binary_latent_class_wald_l95",
        "three_conditionally_independent_binary_codings_prevalence_0_3",
        WALD,
        0.3,
        0x4E07_0000,
        |seed| latent_class_replicate(seed, false),
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn measurement_repeat_codings_ordinal_latent_class_wald_l95() {
    measure(
        "measurement_repeat_codings_ordinal_latent_class_wald_l95",
        "three_conditionally_independent_ordinal_codings_top_level_0_2",
        WALD,
        0.2,
        0x4E08_0000,
        |seed| latent_class_replicate(seed, true),
    );
}
