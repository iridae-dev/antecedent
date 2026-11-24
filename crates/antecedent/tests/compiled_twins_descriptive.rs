//! The raw contrast, the raw-versus-adjusted comparison, the reporting-scale transform and the
//! matched case-control conditional odds ratio are the same computation as their compiled
//! programs.
//!
//! Each test calls the kernel directly and through `compile_descriptive(..).evaluate(..)` on the
//! fixtures the kernel's own tests use (hand-arithmetic arms, the hand-computed log risk ratio,
//! the 1:1 discordant-pair closed form and the mixed-set brute-force profile), and compares every
//! published number bit for bit: arm summaries, the difference, the Welch standard error, the
//! gap, each transformed value and gradient, the delta-method covariance and Wald interval, and
//! the odds ratio, its information, set counts and Wald interval. Neither path draws randomness,
//! so the seed does not enter. No coverage record measures these intervals (their calibration
//! slot is `not_measured` on both paths), so there is no coverage key to compare.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::float_cmp, reason = "the bits are compared exactly")]

use antecedent::analysis::descriptive_programs::{
    DescriptiveOutcome, DescriptiveProgramError, DescriptiveRequest, compile_descriptive,
};
use antecedent_core::ExecutionContext;
use antecedent_estimate::{
    AdjustedEstimate, ArmSummary, Availability, ConditionalOddsRatio, MeanPair, RawContrast,
    ReportingScale, ReportingTransform, compare_raw_adjusted, conditional_odds_ratio, raw_contrast,
    transform_mean_pair,
};

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(0)
}

fn bits(value: f64) -> u64 {
    value.to_bits()
}

fn compiled(request: DescriptiveRequest) -> Result<DescriptiveOutcome, DescriptiveProgramError> {
    let ctx = ctx();
    compile_descriptive(request, &ctx)?.evaluate(&ctx)
}

/// Control `[1, 2, 3]`, active `[4, 6, 8]`.
fn two_arms() -> (Vec<f64>, Vec<f64>) {
    (vec![1.0, 2.0, 3.0, 4.0, 6.0, 8.0], vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
}

/// Active `[1, 1, 0, 1]`, control `[0, 0, 1, 0]`.
fn binary_arms() -> (Vec<f64>, Vec<f64>) {
    (vec![1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0], vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
}

fn assert_arm(label: &str, direct: &ArmSummary, compiled: &ArmSummary) {
    assert_eq!(direct.n, compiled.n, "{label}: rows");
    assert_eq!(bits(direct.mean), bits(compiled.mean), "{label}: mean");
    assert_eq!(bits(direct.m2), bits(compiled.m2), "{label}: sum of squares");
}

fn assert_raw(direct: &RawContrast, compiled: &RawContrast) {
    assert_arm("active", &direct.active, &compiled.active);
    assert_arm("control", &direct.control, &compiled.control);
    assert_eq!(bits(direct.difference), bits(compiled.difference), "raw difference");
    assert_eq!(direct.se.available().map(|v| bits(*v)), compiled.se.available().map(|v| bits(*v)));
    assert_eq!(direct.se.unavailable(), compiled.se.unavailable(), "typed absent SE");
}

fn assert_transform(direct: &ReportingTransform, compiled: &ReportingTransform, level: f64) {
    assert_eq!(direct.contrasts.len(), compiled.contrasts.len());
    for (i, (a, b)) in direct.contrasts.iter().zip(&compiled.contrasts).enumerate() {
        assert_eq!(a.scale, b.scale, "scale {i}");
        assert_eq!(bits(a.value), bits(b.value), "value {i}");
        assert_eq!(a.gradient.map(bits), b.gradient.map(bits), "gradient {i}");
        assert_eq!(direct.se(i).map(bits), compiled.se(i).map(bits), "se {i}");
        assert_eq!(
            direct.interval(i, level).map(|(lo, hi)| (bits(lo), bits(hi))),
            compiled.interval(i, level).map(|(lo, hi)| (bits(lo), bits(hi))),
            "interval {i}"
        );
    }
    match (&direct.covariance, &compiled.covariance) {
        (Availability::Available(a), Availability::Available(b)) => {
            assert_eq!(
                a.iter().copied().map(bits).collect::<Vec<_>>(),
                b.iter().copied().map(bits).collect::<Vec<_>>(),
                "delta-method covariance"
            );
        }
        (Availability::Unavailable(a), Availability::Unavailable(b)) => assert_eq!(a, b),
        _ => panic!("the covariance is available on one path only"),
    }
}

const SCALES: [ReportingScale; 4] = [
    ReportingScale::MeanDifference,
    ReportingScale::RiskDifference,
    ReportingScale::LogRiskRatio,
    ReportingScale::LogOddsRatio,
];

#[test]
fn the_raw_contrast_equals_the_compiled_program() {
    let (y, t) = two_arms();
    let direct = raw_contrast(&y, &t, &ctx()).unwrap();
    let DescriptiveOutcome::RawContrast(program) =
        compiled(DescriptiveRequest::RawContrast { outcome: y, treatment: t }).unwrap()
    else {
        panic!("a raw contrast")
    };
    assert_raw(&direct, &program);
}

#[test]
fn the_raw_versus_adjusted_comparison_equals_the_compiled_program() {
    let (y, t) = two_arms();
    let adjusted = AdjustedEstimate::mean_difference(3.5, Some(0.4));
    let direct = compare_raw_adjusted(&y, &t, adjusted, &ctx()).unwrap();
    let DescriptiveOutcome::Comparison(program) =
        compiled(DescriptiveRequest::CompareRawAdjusted { outcome: y, treatment: t, adjusted })
            .unwrap()
    else {
        panic!("a comparison")
    };
    assert_raw(&direct.raw, &program.raw);
    assert_eq!(bits(direct.gap), bits(program.gap), "gap");
    assert_eq!(direct.adjusted, program.adjusted, "adjusted estimate as supplied");
    assert_eq!(direct.gap_interval, program.gap_interval, "typed absent gap interval");
}

#[test]
fn a_refused_comparison_is_refused_identically_by_the_compiled_program() {
    let (y, t) = two_arms();
    let adjusted = AdjustedEstimate {
        all_observed_population: false,
        ..AdjustedEstimate::mean_difference(3.0, None)
    };
    let direct = compare_raw_adjusted(&y, &t, adjusted, &ctx()).unwrap_err();
    let Err(DescriptiveProgramError::Estimation(program)) =
        compiled(DescriptiveRequest::CompareRawAdjusted { outcome: y, treatment: t, adjusted })
    else {
        panic!("the kernel refusal, unchanged")
    };
    assert_eq!(direct, program);
}

#[test]
fn a_mean_pair_transform_equals_the_compiled_program() {
    for pair in [MeanPair::new(0.4, 0.2, Some([0.01, 0.002, 0.004])), MeanPair::new(0.4, 0.2, None)]
    {
        let direct = transform_mean_pair(&pair, &SCALES).unwrap();
        let DescriptiveOutcome::Transform(program) =
            compiled(DescriptiveRequest::TransformMeanPair { pair, scales: SCALES.to_vec() })
                .unwrap()
        else {
            panic!("a transform")
        };
        assert_transform(&direct, &program, 0.95);
    }
}

#[test]
fn a_raw_arm_transform_equals_the_compiled_program() {
    let (y, t) = binary_arms();
    let raw = raw_contrast(&y, &t, &ctx()).unwrap();
    let direct = transform_mean_pair(&raw.mean_pair(), &SCALES).unwrap();
    let DescriptiveOutcome::RawArmsTransform(program_raw, program) =
        compiled(DescriptiveRequest::RawArmsTransform {
            outcome: y,
            treatment: t,
            scales: SCALES.to_vec(),
        })
        .unwrap()
    else {
        panic!("raw arms and their transform")
    };
    assert_raw(&raw, &program_raw);
    assert_transform(&direct, &program, 0.9);
}

// --- matched case-control ------------------------------------------------------------------

type Set = Vec<(f64, f64)>;

fn rows(sets: &[Set]) -> (Vec<String>, Vec<f64>, Vec<f64>) {
    let mut stratum = Vec::new();
    let (mut case, mut exposed) = (Vec::new(), Vec::new());
    for (i, set) in sets.iter().enumerate() {
        for &(y, x) in set {
            stratum.push(format!("set{i}"));
            case.push(y);
            exposed.push(x);
        }
    }
    (stratum, case, exposed)
}

fn repeat(set: &[(f64, f64)], times: usize) -> Vec<Set> {
    vec![set.to_vec(); times]
}

fn assert_odds_ratio(direct: &ConditionalOddsRatio, program: &ConditionalOddsRatio) {
    assert_eq!(bits(direct.log_odds_ratio), bits(program.log_odds_ratio), "log odds ratio");
    assert_eq!(bits(direct.odds_ratio), bits(program.odds_ratio), "odds ratio");
    assert_eq!(bits(direct.information), bits(program.information), "information");
    assert_eq!(direct.counts, program.counts, "set counts");
    for level in [0.9, 0.95] {
        assert_eq!(
            direct.odds_ratio_interval(level).map(|(lo, hi)| (bits(lo), bits(hi))),
            program.odds_ratio_interval(level).map(|(lo, hi)| (bits(lo), bits(hi))),
            "odds-ratio interval at {level}"
        );
        assert_eq!(
            direct.log_odds_ratio_interval(level).map(|(lo, hi)| (bits(lo), bits(hi))),
            program.log_odds_ratio_interval(level).map(|(lo, hi)| (bits(lo), bits(hi))),
            "log-scale interval at {level}"
        );
    }
}

fn matched_twin(sets: &[Set]) {
    let (stratum, case, exposed) = rows(sets);
    let direct = conditional_odds_ratio(&stratum, &case, &exposed, &ctx()).unwrap();
    let DescriptiveOutcome::MatchedOddsRatio(program) =
        compiled(DescriptiveRequest::MatchedOddsRatio { stratum, case, exposed }).unwrap()
    else {
        panic!("a conditional odds ratio")
    };
    assert_odds_ratio(&direct, &program);
}

#[test]
fn the_one_to_one_conditional_odds_ratio_equals_the_compiled_program() {
    let mut sets = repeat(&[(1.0, 1.0), (0.0, 0.0)], 12);
    sets.extend(repeat(&[(1.0, 0.0), (0.0, 1.0)], 4));
    matched_twin(&sets);
}

#[test]
fn the_mixed_set_conditional_odds_ratio_equals_the_compiled_program() {
    let mut sets = repeat(&[(1.0, 1.0), (0.0, 0.0), (0.0, 0.0)], 3);
    sets.extend(repeat(&[(1.0, 0.0), (0.0, 1.0), (0.0, 0.0)], 2));
    sets.extend(repeat(&[(1.0, 1.0), (0.0, 1.0), (0.0, 0.0)], 2));
    sets.push(vec![(1.0, 1.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]);
    sets.extend(repeat(&[(1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0)], 2));
    sets.push(vec![(1.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 0.0)]);
    sets.push(vec![(1.0, 1.0), (1.0, 0.0), (0.0, 0.0), (0.0, 0.0)]);
    sets.push(vec![(1.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)]);
    sets.push(vec![(1.0, 1.0), (1.0, 1.0), (0.0, 0.0), (0.0, 1.0)]);
    sets.push(vec![(1.0, 0.0), (0.0, 0.0), (0.0, 0.0)]);
    sets.push(vec![(1.0, 1.0), (0.0, 1.0)]);
    sets.push(vec![(0.0, 1.0), (0.0, 0.0)]);
    sets.push(vec![(1.0, 1.0)]);
    matched_twin(&sets);
}

#[test]
fn a_matched_refusal_is_refused_identically_by_the_compiled_program() {
    // Every set concordant: no informative set.
    let sets = repeat(&[(1.0, 1.0), (0.0, 1.0)], 3);
    let (stratum, case, exposed) = rows(&sets);
    let direct = conditional_odds_ratio(&stratum, &case, &exposed, &ctx()).unwrap_err();
    let Err(DescriptiveProgramError::Estimation(program)) =
        compiled(DescriptiveRequest::MatchedOddsRatio { stratum, case, exposed })
    else {
        panic!("the kernel refusal, unchanged")
    };
    assert_eq!(direct, program);
}

#[test]
fn each_family_lowers_to_its_own_route() {
    let ctx = ctx();
    let (y, t) = two_arms();
    let raw =
        compile_descriptive(DescriptiveRequest::RawContrast { outcome: y, treatment: t }, &ctx)
            .unwrap();
    assert_eq!(raw.lowered().route(), "DescriptiveComparison");
    let matched = compile_descriptive(
        DescriptiveRequest::MatchedOddsRatio {
            stratum: vec!["a".into(), "a".into()],
            case: vec![1.0, 0.0],
            exposed: vec![1.0, 0.0],
        },
        &ctx,
    )
    .unwrap();
    assert_eq!(matched.lowered().route(), "MatchedCaseControl");
}
