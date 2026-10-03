//! 2.2 E7: conditional odds ratio from matched case-control sets, against hand-enumerated
//! closed forms and an independent brute-force profile likelihood, plus the typed refusals.

#![allow(clippy::float_cmp, reason = "rows are coded exactly 0.0 or 1.0")]

use antecedent_core::ExecutionContext;
use antecedent_estimate::{
    EstimationError, MATCHED_ESTIMAND, MATCHED_SAMPLING, conditional_odds_ratio,
    parse_matched_estimand, parse_matched_sampling, refuse_matched_interval,
};

/// `(case, exposed)` members of one matched set.
type Set = Vec<(f64, f64)>;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

/// Flatten labelled sets into the parallel rows the estimator takes.
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

fn fit(sets: &[Set]) -> Result<antecedent_estimate::ConditionalOddsRatio, EstimationError> {
    let (stratum, case, exposed) = rows(sets);
    conditional_odds_ratio(&stratum, &case, &exposed, &ctx())
}

fn repeat(set: &[(f64, f64)], times: usize) -> Vec<Set> {
    vec![set.to_vec(); times]
}

fn refused(error: EstimationError) -> (&'static str, String) {
    let EstimationError::Refused { code, message } = error else {
        panic!("expected a coded refusal, got {error:?}");
    };
    (code, message)
}

/// Independent oracle: the conditional log likelihood written as the probability of the
/// observed case subset among every same-size subset of the set (bitmask enumeration),
/// maximized by ternary search. It shares no code with the hypergeometric-weight solver.
fn brute_force_log_odds(sets: &[Set]) -> f64 {
    let loglik = |beta: f64| -> f64 {
        sets.iter()
            .map(|set| {
                let n = set.len();
                let k = set.iter().filter(|m| m.0 == 1.0).count();
                let observed: f64 = set.iter().filter(|m| m.0 == 1.0).map(|m| m.1).sum();
                let denominator: f64 = (0_u32..(1 << n))
                    .filter(|mask| mask.count_ones() as usize == k)
                    .map(|mask| {
                        let exposed: f64 =
                            (0..n).filter(|j| ((mask >> j) & 1) == 1).map(|j| set[j].1).sum();
                        (beta * exposed).exp()
                    })
                    .sum();
                beta * observed - denominator.ln()
            })
            .sum()
    };
    let (mut lo, mut hi) = (-8.0_f64, 8.0_f64);
    for _ in 0..200 {
        let (a, b) = (lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0);
        if loglik(a) < loglik(b) {
            lo = a;
        } else {
            hi = b;
        }
    }
    0.5 * (lo + hi)
}

#[test]
fn one_to_one_matching_reduces_to_the_discordant_pair_ratio() {
    // b = 7 pairs with the case exposed, c = 3 with the control exposed.
    let mut sets = repeat(&[(1.0, 1.0), (0.0, 0.0)], 7);
    sets.extend(repeat(&[(1.0, 0.0), (0.0, 1.0)], 3));
    // Concordant pairs (both exposed, both unexposed) carry no information.
    sets.extend(repeat(&[(1.0, 1.0), (0.0, 1.0)], 4));
    sets.extend(repeat(&[(1.0, 0.0), (0.0, 0.0)], 5));
    // A set of one member and a set with no case are outcome-degenerate.
    sets.push(vec![(1.0, 1.0)]);
    sets.push(vec![(0.0, 1.0), (0.0, 0.0)]);
    let fitted = fit(&sets).unwrap();
    assert!((fitted.odds_ratio - 7.0 / 3.0).abs() < 1e-9, "{}", fitted.odds_ratio);
    assert!((fitted.log_odds_ratio - (7.0_f64 / 3.0).ln()).abs() < 1e-9);
    let counts = fitted.counts;
    assert_eq!(counts.total, 21);
    assert_eq!(counts.informative, 10);
    assert_eq!(counts.exposure_concordant, 9);
    assert_eq!(counts.outcome_degenerate, 2);
    assert_eq!(counts.singleton, 1);
    assert_eq!(
        counts.informative + counts.exposure_concordant + counts.outcome_degenerate,
        counts.total
    );
}

#[test]
fn one_to_two_sets_of_one_type_have_a_closed_form() {
    // One type (n = 3, k = 1, t = 1): P(case exposed) = theta / (theta + 2), so the
    // conditional MLE from `a` of `m` sets with the case exposed is theta = 2a / (m - a).
    for (a, m, truth) in [(3_usize, 5_usize, 3.0_f64), (1, 4, 2.0 / 3.0), (4, 6, 4.0)] {
        let mut sets = repeat(&[(1.0, 1.0), (0.0, 0.0), (0.0, 0.0)], a);
        sets.extend(repeat(&[(1.0, 0.0), (0.0, 1.0), (0.0, 0.0)], m - a));
        let fitted = fit(&sets).unwrap();
        assert!((fitted.odds_ratio - truth).abs() < 1e-8, "a {a} m {m}: {}", fitted.odds_ratio);
    }
}

#[test]
fn mixed_one_to_m_and_k_to_m_sets_match_the_brute_force_profile_likelihood() {
    let mut sets = repeat(&[(1.0, 1.0), (0.0, 0.0), (0.0, 0.0)], 3);
    sets.extend(repeat(&[(1.0, 0.0), (0.0, 1.0), (0.0, 0.0)], 2));
    sets.extend(repeat(&[(1.0, 1.0), (0.0, 1.0), (0.0, 0.0)], 2));
    sets.push(vec![(1.0, 1.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]);
    sets.extend(repeat(&[(1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0)], 2));
    sets.push(vec![(1.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 0.0)]);
    sets.push(vec![(1.0, 1.0), (1.0, 0.0), (0.0, 0.0), (0.0, 0.0)]);
    sets.push(vec![(1.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)]);
    sets.push(vec![(1.0, 1.0), (1.0, 1.0), (0.0, 0.0), (0.0, 1.0)]);
    // Sets that contribute nothing, including a singleton and an all-case set.
    sets.push(vec![(1.0, 0.0), (0.0, 0.0), (0.0, 0.0)]);
    sets.push(vec![(1.0, 1.0), (0.0, 1.0)]);
    sets.push(vec![(0.0, 1.0), (0.0, 0.0)]);
    sets.push(vec![(1.0, 1.0)]);
    let fitted = fit(&sets).unwrap();
    let oracle = brute_force_log_odds(&sets);
    assert!(
        (fitted.log_odds_ratio - oracle).abs() < 1e-6,
        "solver {} vs brute force {oracle}",
        fitted.log_odds_ratio
    );
    assert_eq!(fitted.counts.informative, 14);
    assert_eq!(fitted.counts.outcome_degenerate, 2);
    assert_eq!(fitted.counts.exposure_concordant, 2);
    assert_eq!(fitted.counts.singleton, 1);
}

#[test]
fn concordant_and_degenerate_sets_leave_the_estimate_unchanged() {
    let mut base = repeat(&[(1.0, 1.0), (0.0, 0.0), (0.0, 0.0)], 3);
    base.extend(repeat(&[(1.0, 0.0), (0.0, 1.0), (0.0, 0.0)], 2));
    let plain = fit(&base).unwrap();
    let mut padded = base.clone();
    padded.extend(repeat(&[(1.0, 1.0), (0.0, 1.0), (0.0, 1.0)], 50));
    padded.extend(repeat(&[(1.0, 0.0), (0.0, 0.0), (0.0, 0.0)], 50));
    padded.extend(repeat(&[(0.0, 1.0), (0.0, 0.0)], 20));
    padded.extend(repeat(&[(1.0, 1.0)], 10));
    let with_padding = fit(&padded).unwrap();
    assert_eq!(plain.log_odds_ratio.to_bits(), with_padding.log_odds_ratio.to_bits());
    assert_eq!(plain.counts.informative, with_padding.counts.informative);
    assert_eq!(with_padding.counts.exposure_concordant, 100);
    assert_eq!(with_padding.counts.outcome_degenerate, 30);
    assert_eq!(with_padding.counts.singleton, 10);
}

#[test]
fn a_set_order_or_label_change_does_not_move_the_estimate() {
    let sets = vec![
        vec![(1.0, 1.0), (0.0, 0.0)],
        vec![(1.0, 1.0), (0.0, 0.0)],
        vec![(1.0, 0.0), (0.0, 1.0)],
    ];
    let (stratum, case, exposed) = rows(&sets);
    let forward = conditional_odds_ratio(&stratum, &case, &exposed, &ctx()).unwrap();
    let reversed_strata: Vec<String> = stratum.iter().rev().map(|s| format!("z-{s}")).collect();
    let reversed_case: Vec<f64> = case.iter().rev().copied().collect();
    let reversed_exposed: Vec<f64> = exposed.iter().rev().copied().collect();
    let backward =
        conditional_odds_ratio(&reversed_strata, &reversed_case, &reversed_exposed, &ctx())
            .unwrap();
    assert!((forward.odds_ratio - 2.0).abs() < 1e-9);
    assert_eq!(forward.log_odds_ratio.to_bits(), backward.log_odds_ratio.to_bits());
}

#[test]
fn no_informative_set_is_refused_with_the_counts() {
    let mut sets = repeat(&[(1.0, 1.0), (0.0, 1.0)], 3);
    sets.extend(repeat(&[(1.0, 0.0), (0.0, 0.0)], 2));
    sets.push(vec![(1.0, 1.0)]);
    let (code, message) = refused(fit(&sets).unwrap_err());
    assert_eq!(code, "effect_not_identified");
    assert!(message.contains("matched_case_control.no_informative_sets"), "{message}");
    assert!(message.contains("6 matched sets"), "{message}");
}

#[test]
fn a_monotone_likelihood_has_no_finite_estimate() {
    // Only case-exposed discordant pairs: the odds ratio diverges; only control-exposed: zero.
    let up = fit(&repeat(&[(1.0, 1.0), (0.0, 0.0)], 4)).unwrap_err();
    let down = fit(&repeat(&[(1.0, 0.0), (0.0, 1.0)], 4)).unwrap_err();
    for error in [up, down] {
        let (code, message) = refused(error);
        assert_eq!(code, "route_not_supported");
        assert!(message.contains("matched_case_control.estimate_not_finite"), "{message}");
    }
}

#[test]
fn malformed_rows_are_refused() {
    let two = || vec!["a".to_string(), "a".to_string()];
    let cases: Vec<(Vec<String>, Vec<f64>, Vec<f64>)> = vec![
        (two(), vec![1.0, 0.0], vec![1.0]),
        (two(), vec![1.0, 2.0], vec![1.0, 0.0]),
        (two(), vec![1.0, 0.0], vec![f64::NAN, 0.0]),
        (two(), vec![1.0, 0.5], vec![1.0, 0.0]),
        (vec![], vec![], vec![]),
    ];
    for (stratum, case, exposed) in cases {
        let (code, message) =
            refused(conditional_odds_ratio(&stratum, &case, &exposed, &ctx()).unwrap_err());
        assert_eq!(code, "invalid_argument");
        assert!(message.contains("matched_case_control.invalid_data"), "{message}");
    }
}

#[test]
fn risk_scale_estimands_are_refused_without_prevalence() {
    assert!(parse_matched_estimand(MATCHED_ESTIMAND).is_ok());
    for name in ["population_risk", "absolute_risk", "risk_difference", "risk_ratio"] {
        let (code, message) = refused(parse_matched_estimand(name).unwrap_err());
        assert_eq!(code, "effect_not_identified", "{name}");
        assert!(message.contains("matched_case_control.absolute_risk_not_identified"), "{message}");
    }
    let (code, message) = refused(parse_matched_estimand("hazard_ratio").unwrap_err());
    assert_eq!(code, "route_not_supported");
    assert!(message.contains("matched_case_control.estimand_not_supported"), "{message}");
}

#[test]
fn only_the_declared_sampling_design_is_accepted() {
    assert!(parse_matched_sampling(MATCHED_SAMPLING).is_ok());
    for name in ["cohort", "unmatched_case_control", ""] {
        let (code, message) = refused(parse_matched_sampling(name).unwrap_err());
        assert_eq!(code, "invalid_argument");
        assert!(message.contains("matched_case_control.sampling_design"), "{message}");
    }
}

#[test]
fn the_interval_route_is_closed() {
    let (code, message) = refused(refuse_matched_interval());
    assert_eq!(code, "cell_not_licensed");
    assert!(message.contains("matched_case_control.interval_withheld"), "{message}");
}

#[test]
fn a_cancelled_solve_is_a_budget_stop_never_a_verdict() {
    let mut sets = repeat(&[(1.0, 1.0), (0.0, 0.0)], 7);
    sets.extend(repeat(&[(1.0, 0.0), (0.0, 1.0)], 3));
    let (stratum, case, exposed) = rows(&sets);
    let cancelled = ctx();
    cancelled.cancellation.cancel();
    let (code, message) =
        refused(conditional_odds_ratio(&stratum, &case, &exposed, &cancelled).unwrap_err());
    assert_eq!(code, "transport_budget_cancel");
    assert!(message.contains("matched_case_control.budget"), "{message}");
    // The same rows then solve under a live context.
    assert!(conditional_odds_ratio(&stratum, &case, &exposed, &ctx()).is_ok());
}
