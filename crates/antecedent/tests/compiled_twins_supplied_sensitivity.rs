//! The compiled sensitivity ranges of supplied evidence against the direct kernel calls, and
//! against values computed here without the kernels.
//!
//! * `MatchedDesignSensitivity`: Rosenbaum bounds of matched pairs (`rosenbaum_range`), of a
//!   vector of matched designs (`rosenbaum_vector_range`) and of matched sets
//!   (`matched_sets_upper_p` on the grid, then `matched_sets_break_even`). The twins assert bit
//!   equality with the direct calls on the same inputs. The independent checks enumerate every
//!   sign vector of the pairs (the exact signed-rank law under `Gamma`, with average ranks
//!   computed here), so the bound at `Gamma = 1` is the exact permutation p-value, and every
//!   binary hidden covariate of a matched set (the worst-case expectation under `Gamma`).
//! * `SuppliedEstimateEValue`: E-values of a supplied ratio estimate (`evalue_range`), of a vector
//!   of estimates (`evalue_vector_range`, `component_evalue_range`) and of posterior draws
//!   (`evalue_posterior_range`, `component_evalue_posterior_range`). The independent checks are
//!   the closed form `RR + sqrt(RR (RR - 1))` of `VanderWeele` and Ding (2017) with the
//!   odds-ratio and hazard-ratio conversions, the bounding factor `s^2 / (2 s - 1)`, and the
//!   posterior median and retention probability counted here.
//!
//! These ranges are assumption ranges of the supplied evidence; there is no calibration basis to
//! compare.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    reason = "bits are compared exactly; the fixtures are small"
)]

use antecedent::analysis::supplied_sensitivity_programs::{
    MATCHED_REENTRY_DETAIL, MatchedDesign, MatchedSensitivityOutcome, SuppliedEValueOutcome,
    SuppliedEstimate, TerminalError, compile_matched_design_sensitivity,
    compile_supplied_estimate_evalue,
};
use antecedent_core::{ExecutionContext, SemanticDigest};
use antecedent_validate::rosenbaum::DEFAULT_GAMMA_GRID;
use antecedent_validate::target_native::{RefutationTarget, TargetFamily, TargetQuadrant};
use antecedent_validate::{
    AssumptionRange, ComponentModel, EffectInterval, EffectScale, MatchedSet, MatchedSetScore,
    component_evalue_posterior_range, component_evalue_range, evalue_posterior_range, evalue_range,
    evalue_vector_range, matched_sets_break_even, matched_sets_upper_p, rosenbaum_range,
    rosenbaum_vector_range,
};

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(0x5355_5050)
}

fn target(family: TargetFamily, quadrant: TargetQuadrant, labels: &[&str]) -> RefutationTarget {
    RefutationTarget::new(family, quadrant, labels.iter().copied(), [7_u8; 32]).unwrap()
}

fn scalar_target() -> RefutationTarget {
    target(TargetFamily::ScalarAverageEffect, TargetQuadrant::STATIC_FREQUENTIST, &["effect"])
}

fn same_range(compiled: &AssumptionRange, direct: &AssumptionRange) {
    assert_eq!(compiled.target.identity(), direct.target.identity(), "target");
    assert_eq!(compiled.assumption, direct.assumption);
    assert_eq!(compiled.tested.len(), direct.tested.len());
    for (a, b) in compiled.tested.iter().zip(direct.tested.iter()) {
        assert_eq!(a.parameter.to_bits(), b.parameter.to_bits());
        assert_eq!(a.target.to_bits(), b.target.to_bits(), "target at {}", a.parameter);
        assert_eq!(a.test.map(f64::to_bits), b.test.map(f64::to_bits), "test at {}", a.parameter);
    }
    assert_eq!(compiled.break_even.map(f64::to_bits), direct.break_even.map(f64::to_bits));
    assert_eq!(
        compiled.break_even_significance.map(f64::to_bits),
        direct.break_even_significance.map(f64::to_bits)
    );
    assert_eq!(compiled.identity, direct.identity);
}

fn close(a: f64, b: f64, tolerance: f64) -> bool {
    (a - b).abs() <= tolerance * a.abs().max(b.abs()).max(1.0)
}

// ------------------------------------------------------------------ matched designs

/// Pair differences with ties, a zero and both signs.
const PAIRS: [f64; 13] = [1.5, -0.5, 2.0, 2.0, 0.0, 3.1, -1.5, 0.7, 4.2, 2.6, -2.6, 0.9, 1.1];

/// The exact upper-bound p-value of the signed-rank statistic under `gamma`, by enumerating
/// every sign vector of the non-zero pairs with average ranks of `|d|` computed here.
fn enumerated_bound(differences: &[f64], gamma: f64) -> f64 {
    let nonzero: Vec<f64> = differences.iter().copied().filter(|d| *d != 0.0).collect();
    let n = nonzero.len();
    let ranks: Vec<f64> = nonzero
        .iter()
        .map(|d| {
            let below = nonzero.iter().filter(|e| e.abs() < d.abs()).count() as f64;
            let tied = nonzero.iter().filter(|e| e.abs() == d.abs()).count() as f64;
            below + (tied + 1.0) / 2.0
        })
        .collect();
    let observed: f64 = nonzero.iter().zip(&ranks).filter(|(d, _)| **d > 0.0).map(|(_, r)| r).sum();
    let p = gamma / (1.0 + gamma);
    let mut total = 0.0;
    for mask in 0_u32..(1 << n) {
        let positives = mask.count_ones() as i32;
        let statistic: f64 = (0..n).filter(|i| mask & (1 << i) != 0).map(|i| ranks[i]).sum();
        if statistic >= observed {
            total += p.powi(positives) * (1.0 - p).powi(n as i32 - positives);
        }
    }
    total
}

fn sets() -> Vec<MatchedSet> {
    vec![
        MatchedSet { outcomes: vec![5.0, 1.0, 2.0], treated: 0 },
        MatchedSet { outcomes: vec![3.0, 4.0, 0.0, 1.5], treated: 1 },
        MatchedSet { outcomes: vec![2.0, 2.5], treated: 1 },
        MatchedSet { outcomes: vec![6.0, 0.5, 1.0], treated: 0 },
    ]
}

#[test]
fn compiled_matched_designs_equal_the_direct_bounds() {
    let gammas = DEFAULT_GAMMA_GRID.to_vec();
    // Pairs.
    let program = compile_matched_design_sensitivity(
        MatchedDesign::Pairs {
            target: scalar_target(),
            differences: PAIRS.to_vec(),
            gammas: gammas.clone(),
            alpha: 0.05,
        },
        &ctx(),
    )
    .unwrap();
    assert_eq!(program.lowered().route(), "MatchedDesignSensitivity");
    let MatchedSensitivityOutcome::Pairs { analysis, range } = program.execute(&ctx()).unwrap()
    else {
        panic!("a matched-pair outcome");
    };
    let (direct, direct_range) =
        rosenbaum_range(scalar_target(), &PAIRS, &gammas, 0.05, None).unwrap();
    assert_eq!(analysis, direct);
    assert_eq!(format!("{analysis:?}"), format!("{direct:?}"));
    same_range(&range, &direct_range);

    // A vector of designs.
    let vector = target(
        TargetFamily::ConditionalEffect,
        TargetQuadrant::STATIC_FREQUENTIST,
        &["young", "old"],
    );
    let columns = vec![PAIRS.to_vec(), vec![1.0, 2.0, 3.0, -0.25, 4.0, 5.0, 6.0, 0.5]];
    let program = compile_matched_design_sensitivity(
        MatchedDesign::PairVector {
            target: vector.clone(),
            differences: columns.clone(),
            gammas: gammas.clone(),
            alpha: 0.05,
        },
        &ctx(),
    )
    .unwrap();
    let MatchedSensitivityOutcome::PairVector { analyses, range } =
        program.execute(&ctx()).unwrap()
    else {
        panic!("a matched-pair vector outcome");
    };
    let (direct, direct_range) =
        rosenbaum_vector_range(vector, &columns, &gammas, 0.05, None).unwrap();
    assert_eq!(analyses, direct);
    same_range(&range, &direct_range);

    // Matched sets, both scores.
    for score in [MatchedSetScore::Outcomes, MatchedSetScore::AlignedRanks] {
        let program = compile_matched_design_sensitivity(
            MatchedDesign::Sets { sets: sets(), score, gammas: gammas.clone(), alpha: 0.1 },
            &ctx(),
        )
        .unwrap();
        let MatchedSensitivityOutcome::Sets { grid, gamma_break_even } =
            program.execute(&ctx()).unwrap()
        else {
            panic!("a matched-set outcome");
        };
        let direct: Vec<_> =
            gammas.iter().map(|&g| matched_sets_upper_p(&sets(), score, g).unwrap()).collect();
        assert_eq!(grid.len(), direct.len());
        for (a, b) in grid.iter().zip(&direct) {
            assert_eq!(a.p_value.to_bits(), b.p_value.to_bits());
            assert_eq!(a.expectation.to_bits(), b.expectation.to_bits());
            assert_eq!(a.variance.to_bits(), b.variance.to_bits());
            assert_eq!(a.statistic.to_bits(), b.statistic.to_bits());
        }
        assert_eq!(
            gamma_break_even.map(f64::to_bits),
            matched_sets_break_even(&sets(), score, 0.1).unwrap().map(f64::to_bits)
        );
    }
}

#[test]
fn the_gamma_one_bound_is_the_exact_signed_rank_permutation_p_value() {
    let gammas = vec![1.0, 1.5, 2.0, 3.0];
    let program = compile_matched_design_sensitivity(
        MatchedDesign::Pairs {
            target: scalar_target(),
            differences: PAIRS.to_vec(),
            gammas: gammas.clone(),
            alpha: 0.05,
        },
        &ctx(),
    )
    .unwrap();
    let MatchedSensitivityOutcome::Pairs { analysis, range } = program.execute(&ctx()).unwrap()
    else {
        panic!("a matched-pair outcome");
    };
    assert_eq!(analysis.pairs, 12);
    assert!(analysis.exact);
    for &(gamma, p) in &analysis.grid {
        let expected = enumerated_bound(&PAIRS, gamma);
        assert!(close(p, expected, 1e-12), "gamma {gamma}: {p} against {expected}");
    }
    // At Gamma = 1 the bound is the randomization p-value: the share of the 2^12 equally likely
    // sign vectors whose statistic reaches the observed one.
    let permutation = enumerated_bound(&PAIRS, 1.0);
    assert!(close(analysis.grid[0].1, permutation, 1e-12));
    assert!(close(range.tested[0].target, permutation, 1e-12));
    // The break-even Gamma is where the enumerated bound crosses alpha.
    let star = analysis.gamma_break_even.unwrap();
    assert!(enumerated_bound(&PAIRS, star) <= 0.05 + 1e-12);
    assert!(enumerated_bound(&PAIRS, star * 1.001) > 0.05);
    // All positive pairs: the bound is (Gamma / (1 + Gamma))^n.
    let program = compile_matched_design_sensitivity(
        MatchedDesign::Pairs {
            target: scalar_target(),
            differences: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            gammas: vec![1.0, 2.0],
            alpha: 0.05,
        },
        &ctx(),
    )
    .unwrap();
    let MatchedSensitivityOutcome::Pairs { analysis, .. } = program.execute(&ctx()).unwrap() else {
        panic!("a matched-pair outcome");
    };
    assert!(close(analysis.grid[0].1, 0.5_f64.powi(6), 1e-15));
    assert!(close(analysis.grid[1].1, (2.0_f64 / 3.0).powi(6), 1e-15));
}

#[test]
fn matched_sets_follow_the_randomization_moments_and_the_worst_hidden_covariate() {
    let program = compile_matched_design_sensitivity(
        MatchedDesign::Sets {
            sets: sets(),
            score: MatchedSetScore::Outcomes,
            gammas: vec![1.0, 2.5],
            alpha: 0.05,
        },
        &ctx(),
    )
    .unwrap();
    let MatchedSensitivityOutcome::Sets { grid, .. } = program.execute(&ctx()).unwrap() else {
        panic!("a matched-set outcome");
    };
    // The statistic is the treated units' outcomes: 5 + 4 + 2.5 + 6.
    assert_eq!(grid[0].statistic, 17.5);
    // Gamma = 1: each set's treated unit is uniform, so the moments are the set means and the
    // population variances.
    let mut mean = 0.0;
    let mut variance = 0.0;
    for set in sets() {
        let n = set.outcomes.len() as f64;
        let m = set.outcomes.iter().sum::<f64>() / n;
        mean += m;
        variance += set.outcomes.iter().map(|y| (y - m) * (y - m)).sum::<f64>() / n;
    }
    assert!(close(grid[0].expectation, mean, 1e-12));
    assert!(close(grid[0].variance, variance, 1e-12));
    // Gamma = 2.5: the worst-case expectation of each set is the largest mean over every binary
    // hidden covariate u (not constant), the treated unit drawn with odds Gamma^u.
    let gamma = 2.5;
    let worst: f64 = sets()
        .iter()
        .map(|set| {
            let n = set.outcomes.len();
            (1_u32..(1 << n) - 1)
                .map(|mask| {
                    let w: Vec<f64> =
                        (0..n).map(|i| if mask & (1 << i) != 0 { gamma } else { 1.0 }).collect();
                    let total: f64 = w.iter().sum();
                    w.iter().zip(&set.outcomes).map(|(w, y)| w * y).sum::<f64>() / total
                })
                .fold(f64::NEG_INFINITY, f64::max)
        })
        .sum();
    assert!(close(grid[1].expectation, worst, 1e-12));
}

#[test]
fn a_kernel_refusal_is_the_same_refusal_on_both_paths() {
    let zeros = vec![0.0, 0.0, 0.0];
    let program = compile_matched_design_sensitivity(
        MatchedDesign::Pairs {
            target: scalar_target(),
            differences: zeros.clone(),
            gammas: vec![1.0],
            alpha: 0.05,
        },
        &ctx(),
    )
    .unwrap();
    let Err(TerminalError::Kernel(compiled)) = program.execute(&ctx()) else {
        panic!("the kernel refuses all-zero differences");
    };
    let direct = rosenbaum_range(scalar_target(), &zeros, &[1.0], 0.05, None).unwrap_err();
    assert_eq!(compiled.to_string(), direct.to_string());

    let program = compile_matched_design_sensitivity(
        MatchedDesign::Sets {
            sets: sets(),
            score: MatchedSetScore::Outcomes,
            gammas: vec![0.5],
            alpha: 0.05,
        },
        &ctx(),
    )
    .unwrap();
    let Err(TerminalError::Kernel(compiled)) = program.execute(&ctx()) else {
        panic!("the kernel refuses Gamma below one");
    };
    let direct = matched_sets_upper_p(&sets(), MatchedSetScore::Outcomes, 0.5).unwrap_err();
    assert_eq!(compiled.to_string(), direct.to_string());

    // Inside another program's lowered dispatch the compiler refuses with the family's detail.
    let scoped = ctx().enter_lowered_dispatch(SemanticDigest::from_bytes([3_u8; 32]));
    let refusal = compile_matched_design_sensitivity(
        MatchedDesign::Pairs {
            target: scalar_target(),
            differences: PAIRS.to_vec(),
            gammas: vec![1.0],
            alpha: 0.05,
        },
        &scoped,
    )
    .unwrap_err();
    let text = refusal.to_string();
    assert!(text.contains("public_builder_reentry"), "{text}");
    assert!(text.contains(MATCHED_REENTRY_DETAIL), "{text}");
}

#[test]
fn the_matched_program_identity_is_the_declared_design() {
    let compile = |differences: Vec<f64>, gammas: Vec<f64>, alpha: f64| {
        compile_matched_design_sensitivity(
            MatchedDesign::Pairs { target: scalar_target(), differences, gammas, alpha },
            &ctx(),
        )
        .unwrap()
        .identity()
    };
    let base = compile(PAIRS.to_vec(), vec![1.0, 2.0], 0.05);
    assert_eq!(base, compile(PAIRS.to_vec(), vec![1.0, 2.0], 0.05));
    let mut changed = PAIRS.to_vec();
    changed[3] = 2.000_000_1;
    assert_ne!(base, compile(changed, vec![1.0, 2.0], 0.05));
    assert_ne!(base, compile(PAIRS.to_vec(), vec![1.0, 2.5], 0.05));
    assert_ne!(base, compile(PAIRS.to_vec(), vec![1.0, 2.0], 0.1));
    let other_target = compile_matched_design_sensitivity(
        MatchedDesign::Pairs {
            target: target(
                TargetFamily::ScalarAverageEffect,
                TargetQuadrant::STATIC_BAYESIAN,
                &["effect"],
            ),
            differences: PAIRS.to_vec(),
            gammas: vec![1.0, 2.0],
            alpha: 0.05,
        },
        &ctx(),
    )
    .unwrap();
    assert_ne!(base, other_target.identity());
    let a = compile_matched_design_sensitivity(
        MatchedDesign::Sets {
            sets: sets(),
            score: MatchedSetScore::Outcomes,
            gammas: vec![1.0],
            alpha: 0.05,
        },
        &ctx(),
    )
    .unwrap();
    let b = compile_matched_design_sensitivity(
        MatchedDesign::Sets {
            sets: sets(),
            score: MatchedSetScore::AlignedRanks,
            gammas: vec![1.0],
            alpha: 0.05,
        },
        &ctx(),
    )
    .unwrap();
    assert_ne!(a.identity(), b.identity());
    assert_eq!(a.lowered().route(), "MatchedDesignSensitivity");
}

// ------------------------------------------------------------------ supplied estimates

fn ratio(estimate: f64, lower: f64, upper: f64) -> EffectInterval {
    EffectInterval::Ratio { estimate, lower: Some(lower), upper: Some(upper) }
}

fn evalue_of(rr: f64) -> f64 {
    let rr = if rr >= 1.0 { rr } else { 1.0 / rr };
    rr + (rr * (rr - 1.0)).sqrt()
}

#[test]
fn compiled_supplied_estimates_equal_the_direct_evalues() {
    for (interval, scale) in [
        (ratio(3.9, 1.8, 8.7), EffectScale::RiskRatio),
        (ratio(0.4, 0.2, 0.8), EffectScale::OddsRatio { common_outcome: true }),
        (ratio(2.0, 0.9, 4.0), EffectScale::HazardRatio { common_outcome: false }),
        (
            EffectInterval::Standardized { estimate: 0.4, standard_error: Some(0.1) },
            EffectScale::StandardizedMeanDifference,
        ),
    ] {
        let program = compile_supplied_estimate_evalue(
            SuppliedEstimate::Scalar { target: scalar_target(), interval, scale },
            &ctx(),
        )
        .unwrap();
        assert_eq!(program.lowered().route(), "SuppliedEstimateEValue");
        let SuppliedEValueOutcome::Scalar { bounds, range } = program.execute(&ctx()).unwrap()
        else {
            panic!("a scalar outcome");
        };
        let (direct, direct_range) = evalue_range(scalar_target(), interval, scale).unwrap();
        assert_eq!(format!("{bounds:?}"), format!("{direct:?}"));
        same_range(&range, &direct_range);
    }

    let vector = target(
        TargetFamily::ConditionalEffect,
        TargetQuadrant::STATIC_FREQUENTIST,
        &["young", "old"],
    );
    let coordinates = vec![ratio(3.9, 1.8, 8.7), ratio(2.0, 1.2, 3.3)];
    for model in [None, Some(ComponentModel::Mediation), Some(ComponentModel::MarginalStructural)] {
        let program = compile_supplied_estimate_evalue(
            SuppliedEstimate::Vector {
                target: vector.clone(),
                coordinates: coordinates.clone(),
                scale: EffectScale::RiskRatio,
                model,
            },
            &ctx(),
        )
        .unwrap();
        let SuppliedEValueOutcome::Vector { bounds, range } = program.execute(&ctx()).unwrap()
        else {
            panic!("a vector outcome");
        };
        let (direct, direct_range) = match model {
            Some(model) => {
                component_evalue_range(vector.clone(), &coordinates, EffectScale::RiskRatio, model)
            }
            None => evalue_vector_range(vector.clone(), &coordinates, EffectScale::RiskRatio),
        }
        .unwrap();
        assert_eq!(format!("{bounds:?}"), format!("{direct:?}"));
        same_range(&range, &direct_range);
    }

    let posterior_target =
        target(TargetFamily::ConditionalEffect, TargetQuadrant::STATIC_BAYESIAN, &["a", "b"]);
    let draws: Vec<Vec<f64>> = vec![
        (0..400).map(|i| (0.3 + 0.002 * f64::from(i)).exp()).collect(),
        (0..400).map(|i| (1.1 - 0.004 * f64::from(i)).exp()).collect(),
    ];
    for model in [None, Some(ComponentModel::MarginalStructural)] {
        let program = compile_supplied_estimate_evalue(
            SuppliedEstimate::Posterior {
                target: posterior_target.clone(),
                draws: draws.clone(),
                scale: EffectScale::OddsRatio { common_outcome: false },
                level: 0.9,
                model,
            },
            &ctx(),
        )
        .unwrap();
        let SuppliedEValueOutcome::Posterior { summary, range } = program.execute(&ctx()).unwrap()
        else {
            panic!("a posterior outcome");
        };
        let scale = EffectScale::OddsRatio { common_outcome: false };
        let (direct, direct_range) = match model {
            Some(model) => component_evalue_posterior_range(
                posterior_target.clone(),
                &draws,
                scale,
                0.9,
                model,
            ),
            None => evalue_posterior_range(posterior_target.clone(), &draws, scale, 0.9),
        }
        .unwrap();
        assert_eq!(format!("{summary:?}"), format!("{direct:?}"));
        same_range(&range, &direct_range);
    }
}

#[test]
fn ratio_evalues_are_the_closed_form_of_vanderweele_and_ding() {
    let run = |interval: EffectInterval, scale: EffectScale| {
        let program = compile_supplied_estimate_evalue(
            SuppliedEstimate::Scalar { target: scalar_target(), interval, scale },
            &ctx(),
        )
        .unwrap();
        let SuppliedEValueOutcome::Scalar { bounds, range } = program.execute(&ctx()).unwrap()
        else {
            panic!("a scalar outcome");
        };
        (bounds, range)
    };
    // A harmful risk ratio: E = RR + sqrt(RR (RR - 1)), the limit nearest the null is the lower.
    let (bounds, range) = run(ratio(3.9, 1.8, 8.7), EffectScale::RiskRatio);
    assert!(close(bounds.point, 3.9 + (3.9_f64 * 2.9).sqrt(), 1e-12));
    assert!(close(bounds.limit.unwrap(), 1.8 + (1.8_f64 * 0.8).sqrt(), 1e-12));
    assert!(close(range.break_even.unwrap(), bounds.point, 0.0));
    // At confounder strength s the worst-case adjusted risk ratio is RR / (s^2 / (2 s - 1)),
    // capped at the null.
    for point in &range.tested {
        let s = point.parameter;
        let factor = s * s / (2.0 * s - 1.0);
        assert!(close(point.target, (3.9 / factor).max(1.0), 1e-12), "strength {s}");
        assert!(close(point.test.unwrap(), (1.8 / factor).max(1.0), 1e-12), "strength {s}");
    }
    // A protective risk ratio reads its reciprocal and its upper limit.
    let (bounds, _) = run(ratio(0.4, 0.2, 0.8), EffectScale::RiskRatio);
    assert!(close(bounds.point, evalue_of(1.0 / 0.4), 1e-12));
    assert!(close(bounds.limit.unwrap(), evalue_of(1.0 / 0.8), 1e-12));
    // An interval containing the null has limit E-value 1.
    let (bounds, _) = run(ratio(1.6, 0.9, 2.8), EffectScale::RiskRatio);
    assert_eq!(bounds.limit, Some(1.0));
    // A common-outcome odds ratio converts by its square root.
    let (bounds, _) = run(ratio(4.0, 2.25, 9.0), EffectScale::OddsRatio { common_outcome: true });
    assert!(close(bounds.point, evalue_of(2.0), 1e-12));
    assert!(close(bounds.limit.unwrap(), evalue_of(1.5), 1e-12));
    // A common-outcome hazard ratio converts by (1 - 0.5^sqrt(HR)) / (1 - 0.5^sqrt(1 / HR)).
    let hr = 2.0_f64;
    let rr = (1.0 - 0.5_f64.powf(hr.sqrt())) / (1.0 - 0.5_f64.powf((1.0 / hr).sqrt()));
    let (bounds, _) = run(
        EffectInterval::Ratio { estimate: hr, lower: None, upper: None },
        EffectScale::HazardRatio { common_outcome: true },
    );
    assert!(close(bounds.risk_ratio, rr, 1e-12));
    assert!(close(bounds.point, evalue_of(rr), 1e-12));
    assert_eq!(bounds.limit, None);
}

#[test]
fn posterior_evalues_read_the_median_and_the_counted_retention() {
    // 2001 risk-ratio draws exp(-0.2005 + 0.001 i): the median is exp(0.7995) and 1800 draws
    // lie above the null.
    let draws: Vec<f64> = (0..=2000).map(|i| (-0.2005 + 0.001 * f64::from(i)).exp()).collect();
    let program = compile_supplied_estimate_evalue(
        SuppliedEstimate::Posterior {
            target: target(
                TargetFamily::ConditionalEffect,
                TargetQuadrant::STATIC_BAYESIAN,
                &["a"],
            ),
            draws: vec![draws.clone()],
            scale: EffectScale::RiskRatio,
            level: 0.9,
            model: None,
        },
        &ctx(),
    )
    .unwrap();
    let SuppliedEValueOutcome::Posterior { summary, range } = program.execute(&ctx()).unwrap()
    else {
        panic!("a posterior outcome");
    };
    let median = (-0.2005_f64 + 0.001 * 1000.0).exp();
    assert!(close(summary.median_risk_ratio[0], median, 1e-12));
    assert!(close(summary.bounds[0].point, evalue_of(median), 1e-12));
    assert!(close(range.break_even.unwrap(), evalue_of(median), 1e-12));
    // The 5% quantile of the equally spaced draws is draw 100.
    let lower = (-0.2005_f64 + 0.1).exp();
    assert!(close(summary.credible_interval[0].0, lower, 1e-9));
    // At strength 1 (no confounding) the retention is the share of draws above the null; at
    // strength s a draw is retained when it stays above the bounding factor s^2 / (2 s - 1).
    for &(strength, probability) in &summary.retention {
        let factor = strength * strength / (2.0 * strength - 1.0);
        let kept = draws.iter().filter(|rr| **rr / factor > 1.0).count();
        assert!(close(probability, kept as f64 / 2001.0, 1e-15), "strength {strength}");
    }
    assert!(close(summary.retention[0].1, 1800.0 / 2001.0, 1e-15));
}

#[test]
fn a_supplied_estimate_refusal_is_the_kernels() {
    let interval = ratio(-1.0, -2.0, 0.5);
    let program = compile_supplied_estimate_evalue(
        SuppliedEstimate::Scalar {
            target: scalar_target(),
            interval,
            scale: EffectScale::RiskRatio,
        },
        &ctx(),
    )
    .unwrap();
    let Err(TerminalError::Kernel(compiled)) = program.execute(&ctx()) else {
        panic!("the kernel refuses a non-positive ratio");
    };
    let direct = evalue_range(scalar_target(), interval, EffectScale::RiskRatio).unwrap_err();
    assert_eq!(compiled.to_string(), direct.to_string());
    // Two coordinates against a one-label target.
    let program = compile_supplied_estimate_evalue(
        SuppliedEstimate::Vector {
            target: scalar_target(),
            coordinates: vec![ratio(2.0, 1.5, 3.0), ratio(3.0, 2.0, 4.0)],
            scale: EffectScale::RiskRatio,
            model: None,
        },
        &ctx(),
    )
    .unwrap();
    assert!(matches!(program.execute(&ctx()), Err(TerminalError::Kernel(_))));
    // Inside another program's lowered dispatch the compiler refuses.
    let scoped = ctx().enter_lowered_dispatch(SemanticDigest::from_bytes([5_u8; 32]));
    let refusal = compile_supplied_estimate_evalue(
        SuppliedEstimate::Scalar {
            target: scalar_target(),
            interval: ratio(2.0, 1.5, 3.0),
            scale: EffectScale::RiskRatio,
        },
        &scoped,
    )
    .unwrap_err();
    assert!(refusal.to_string().contains("public_builder_reentry"));
}

#[test]
fn the_estimate_program_identity_is_the_declared_estimate() {
    let compile = |interval: EffectInterval, scale: EffectScale| {
        compile_supplied_estimate_evalue(
            SuppliedEstimate::Scalar { target: scalar_target(), interval, scale },
            &ctx(),
        )
        .unwrap()
    };
    let base = compile(ratio(2.0, 1.5, 3.0), EffectScale::RiskRatio);
    assert_eq!(base.lowered().route(), "SuppliedEstimateEValue");
    assert_eq!(base.identity(), compile(ratio(2.0, 1.5, 3.0), EffectScale::RiskRatio).identity());
    assert_ne!(base.identity(), compile(ratio(2.0, 1.5, 3.1), EffectScale::RiskRatio).identity());
    assert_ne!(
        base.identity(),
        compile(ratio(2.0, 1.5, 3.0), EffectScale::OddsRatio { common_outcome: false }).identity()
    );
    assert_ne!(
        base.identity(),
        compile(
            EffectInterval::Ratio { estimate: 2.0, lower: Some(1.5), upper: None },
            EffectScale::RiskRatio
        )
        .identity()
    );
    let vector = |model: Option<ComponentModel>| {
        compile_supplied_estimate_evalue(
            SuppliedEstimate::Vector {
                target: target(
                    TargetFamily::ConditionalEffect,
                    TargetQuadrant::STATIC_FREQUENTIST,
                    &["a", "b"],
                ),
                coordinates: vec![ratio(2.0, 1.5, 3.0), ratio(3.0, 2.0, 4.0)],
                scale: EffectScale::RiskRatio,
                model,
            },
            &ctx(),
        )
        .unwrap()
        .identity()
    };
    assert_ne!(vector(None), vector(Some(ComponentModel::Mediation)));
    assert_ne!(
        vector(Some(ComponentModel::Mediation)),
        vector(Some(ComponentModel::MarginalStructural))
    );
}
