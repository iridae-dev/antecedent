//! Independent oracles for the adaptive-inference numerics.
//!
//! Every asserted value has an oracle that is not the code under test:
//! closed forms (incomplete beta of integer shapes, the integer-shape sum for
//! the probability that one Beta variable exceeds another, the uniform-Beta
//! integrals), values integrated with `SciPy`'s adaptive quadrature and written
//! down here as constants, published boundaries (O'Brien-Fleming-type
//! alpha-spending boundaries of Lan and `DeMets`, as in the `gsDesign`
//! vignette; the Armitage, `McPherson` and Rowe five-look inflation), exact
//! enumeration of a two-arm Thompson-sampling tree, hand-computed
//! augmented-inverse-probability scores, and fixed-seed simulation with a
//! known truth.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::needless_range_loop, reason = "enumerations index parallel per-arm arrays by arm")]

use antecedent_core::CausalRng;
use antecedent_kernels::standard_normal;
use antecedent_stats::adaptive_inference::{
    AdaptiveInferenceError, BetaPosterior, Estimand, OutcomeModelPrior, Sidedness,
    SpendingFunction, WeightScheme, adaptive_scores, best_arm_probabilities, boundaries,
    bounded_mean_confidence_sequence, crossing_probabilities, expected_losses, first_crossing,
    mixture_hoeffding_radius, probability_superior, regularized_incomplete_beta, weighted_estimate,
};
use antecedent_stats::ln_gamma;

fn never() -> bool {
    false
}

fn close(actual: f64, expected: f64, tolerance: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{what}: expected {expected}, got {actual} (tolerance {tolerance})"
    );
}

fn post(alpha: f64, beta: f64) -> BetaPosterior {
    BetaPosterior::new(alpha, beta).expect("valid posterior")
}

// -- Beta posterior ---------------------------------------------------------------------------

#[test]
fn counts_update_the_beta_posterior_and_its_moments() {
    let p = BetaPosterior::from_counts(1.0, 1.0, 7, 3).unwrap();
    close(p.alpha(), 8.0, 0.0, "alpha");
    close(p.beta(), 4.0, 0.0, "beta");
    close(p.mean(), 8.0 / 12.0, 1e-15, "mean");
    // alpha beta / ((alpha + beta)^2 (alpha + beta + 1)) = 32 / (144 * 13).
    close(p.variance(), 32.0 / (144.0 * 13.0), 1e-15, "variance");
    assert!(BetaPosterior::new(0.5, 1.0).is_err(), "a shape below one is refused");
    assert!(BetaPosterior::new(f64::NAN, 1.0).is_err());
}

#[test]
fn incomplete_beta_matches_closed_forms_and_scipy() {
    let x: f64 = 0.37;
    // Integer shapes: I_x(a, b) = sum_{j=a}^{a+b-1} C(a+b-1, j) x^j (1-x)^(a+b-1-j).
    for (a, b) in [(1_u32, 1_u32), (2, 1), (1, 3), (3, 2), (4, 5), (6, 6)] {
        let n = a + b - 1;
        let mut expected = 0.0;
        for j in a..=n {
            let mut binomial = 1.0;
            for i in 0..j {
                binomial *= f64::from(n - i) / f64::from(i + 1);
            }
            expected += binomial
                * x.powi(i32::try_from(j).unwrap())
                * (1.0 - x).powi(i32::try_from(n - j).unwrap());
        }
        let value = regularized_incomplete_beta(x, f64::from(a), f64::from(b)).unwrap();
        close(value, expected, 1e-12, &format!("I_x({a},{b})"));
    }
    // Non-integer and large shapes against scipy.special.betainc.
    close(
        regularized_incomplete_beta(0.3, 2.5, 4.0).unwrap(),
        0.352_197_585_906_767_2,
        1e-12,
        "I_0.3(2.5,4)",
    );
    close(
        regularized_incomplete_beta(0.55, 30.0, 20.0).unwrap(),
        0.232_888_606_786_569_3,
        1e-12,
        "I_0.55(30,20)",
    );
    close(
        regularized_incomplete_beta(0.56, 500.0, 400.0).unwrap(),
        0.604_862_187_242_888_3,
        1e-9,
        "I_0.56(500,400)",
    );
    assert!(regularized_incomplete_beta(1.5, 1.0, 1.0).is_err());
    assert!(regularized_incomplete_beta(0.5, 0.0, 1.0).is_err());
}

/// `P(theta_b > theta_a)` for integer shapes (Evan Miller's finite sum):
/// `sum_{i=0}^{alpha_b - 1} B(alpha_a + i, beta_a + beta_b) /
/// ((beta_b + i) B(1 + i, beta_b) B(alpha_a, beta_a))`.
fn miller_b_exceeds_a(a: (u32, u32), b: (u32, u32)) -> f64 {
    let ln_b = |x: f64, y: f64| ln_gamma(x) + ln_gamma(y) - ln_gamma(x + y);
    let (aa, ba) = (f64::from(a.0), f64::from(a.1));
    let bb = f64::from(b.1);
    (0..b.0)
        .map(|i| {
            let i = f64::from(i);
            (ln_b(aa + i, ba + bb) - ln_b(1.0 + i, bb) - ln_b(aa, ba)).exp() / (bb + i)
        })
        .sum()
}

#[test]
fn superiority_matches_closed_forms_and_the_integer_sum() {
    // theta_a ~ Beta(2,1) against a uniform arm: E[theta_a] = 2/3.
    close(
        probability_superior(post(2.0, 1.0), post(1.0, 1.0), &never).unwrap(),
        2.0 / 3.0,
        1e-10,
        "2/3",
    );
    // Beta(3,2) against Beta(2,4): exactly 5/6.
    close(
        probability_superior(post(3.0, 2.0), post(2.0, 4.0), &never).unwrap(),
        5.0 / 6.0,
        1e-10,
        "5/6",
    );
    // Beta(8,4) against Beta(5,7): scipy quad 0.9008097165991901.
    close(
        probability_superior(post(8.0, 4.0), post(5.0, 7.0), &never).unwrap(),
        0.900_809_716_599_190_1,
        1e-9,
        "(8,4) vs (5,7)",
    );
    for (a, b) in [((8_u32, 4_u32), (5_u32, 7_u32)), ((3, 2), (2, 4)), ((20, 12), (14, 9))] {
        let library = probability_superior(
            post(f64::from(a.0), f64::from(a.1)),
            post(f64::from(b.0), f64::from(b.1)),
            &never,
        )
        .unwrap();
        close(library, 1.0 - miller_b_exceeds_a(a, b), 1e-9, "integer sum");
    }
    // Strongly separated large-count arms: scipy quad 0.9999563645304002.
    close(
        probability_superior(post(80.0, 40.0), post(50.0, 70.0), &never).unwrap(),
        0.999_956_364_530_400_2,
        1e-9,
        "(80,40) vs (50,70)",
    );
}

#[test]
fn best_arm_probabilities_are_exchangeable_normalized_and_match_scipy() {
    let uniform = [post(1.0, 1.0); 3];
    for p in best_arm_probabilities(&uniform, &never).unwrap() {
        close(p, 1.0 / 3.0, 1e-10, "exchangeable arms");
    }
    // Beta(2,1) against two uniforms: P(best) = E[theta^2] = 1/2, then 1/4 each.
    let p =
        best_arm_probabilities(&[post(2.0, 1.0), post(1.0, 1.0), post(1.0, 1.0)], &never).unwrap();
    close(p[0], 0.5, 1e-10, "arm 0");
    close(p[1], 0.25, 1e-10, "arm 1");
    close(p[2], 0.25, 1e-10, "arm 2");
    // scipy quad for (8,4), (5,7), (3,3).
    let p =
        best_arm_probabilities(&[post(8.0, 4.0), post(5.0, 7.0), post(3.0, 3.0)], &never).unwrap();
    close(p[0], 0.709_572_080_900_658_1, 1e-9, "arm 0");
    close(p[1], 0.064_502_417_759_927_43, 1e-9, "arm 1");
    close(p[2], 0.225_925_501_330_939_4, 1e-9, "arm 2");
    close(p.iter().sum::<f64>(), 1.0, 1e-9, "normalization");
    // Permutation invariance.
    let q =
        best_arm_probabilities(&[post(3.0, 3.0), post(8.0, 4.0), post(5.0, 7.0)], &never).unwrap();
    close(q[1], p[0], 1e-12, "relabelled arm");
    assert!(
        best_arm_probabilities(&[post(1.0, 1.0)], &never).is_err(),
        "one arm is not a comparison"
    );
}

#[test]
fn expected_loss_matches_the_uniform_integrals_and_scipy() {
    // Two uniform arms: E[max] = 2/3, loss = 2/3 - 1/2 = 1/6.
    let l = expected_losses(&[post(1.0, 1.0), post(1.0, 1.0)], &never).unwrap();
    close(l[0], 1.0 / 6.0, 1e-10, "uniform pair");
    // Three uniform arms: E[max] = 3/4, loss = 1/4.
    for loss in expected_losses(&[post(1.0, 1.0); 3], &never).unwrap() {
        close(loss, 0.25, 1e-10, "uniform triple");
    }
    // Beta(2,1) with a uniform: E[max] = 1 - 1/4; losses 3/4 - 2/3 and 3/4 - 1/2.
    let l = expected_losses(&[post(2.0, 1.0), post(1.0, 1.0)], &never).unwrap();
    close(l[0], 1.0 / 12.0, 1e-10, "beta(2,1)");
    close(l[1], 0.25, 1e-10, "uniform");
    // scipy quad for (8,4), (5,7), (3,3).
    let l = expected_losses(&[post(8.0, 4.0), post(5.0, 7.0), post(3.0, 3.0)], &never).unwrap();
    close(l[0], 0.037_731_723_564_271_79, 1e-9, "arm 0");
    close(l[1], 0.287_731_723_564_271_74, 1e-9, "arm 1");
    close(l[2], 0.204_398_390_230_927, 1e-9, "arm 2");
}

/// All histories of `steps` Thompson-sampling pulls of two Bernoulli arms
/// under uniform priors, with the exact probability of each history under the
/// prior predictive, calling `visit(probability, posterior_0, posterior_1)`.
fn enumerate_thompson(
    steps: usize,
    state: [(u32, u32); 2],
    probability: f64,
    visit: &mut dyn FnMut(f64, [(u32, u32); 2]),
) {
    if steps == 0 {
        visit(probability, state);
        return;
    }
    // Probability of pulling arm 0 given the current posteriors, by the
    // integer sum (independent of the library).
    let a = (state[0].0, state[0].1);
    let b = (state[1].0, state[1].1);
    let pull_zero = 1.0 - miller_b_exceeds_a(a, b);
    for (arm, choose) in [(0_usize, pull_zero), (1, 1.0 - pull_zero)] {
        let (s, f) = state[arm];
        let success = f64::from(s) / f64::from(s + f);
        for (succeeded, outcome) in [(true, success), (false, 1.0 - success)] {
            let mut next = state;
            if succeeded {
                next[arm].0 += 1;
            } else {
                next[arm].1 += 1;
            }
            enumerate_thompson(steps - 1, next, probability * choose * outcome, visit);
        }
    }
}

#[test]
fn thompson_posterior_is_valid_under_adaptive_assignment_by_enumeration() {
    // The posterior means and the posterior probability of superiority are
    // martingales of the prior predictive however the arms were chosen:
    // their expectation over every history equals the prior value.
    let mut total = 0.0;
    let mut mean_zero = 0.0;
    let mut mean_one = 0.0;
    let mut superiority = 0.0;
    let mut histories = 0;
    enumerate_thompson(4, [(1, 1), (1, 1)], 1.0, &mut |p, state| {
        total += p;
        mean_zero += p * f64::from(state[0].0) / f64::from(state[0].0 + state[0].1);
        mean_one += p * f64::from(state[1].0) / f64::from(state[1].0 + state[1].1);
        let library = probability_superior(
            post(f64::from(state[0].0), f64::from(state[0].1)),
            post(f64::from(state[1].0), f64::from(state[1].1)),
            &never,
        )
        .unwrap();
        let sum = 1.0 - miller_b_exceeds_a(state[0], state[1]);
        assert!((library - sum).abs() < 1e-9, "library {library} vs integer sum {sum}");
        superiority += p * library;
        histories += 1;
    });
    assert_eq!(histories, 256, "four pulls, two arms, two outcomes: 4^4 histories");
    close(total, 1.0, 1e-12, "history probabilities");
    close(mean_zero, 0.5, 1e-12, "E[posterior mean of arm 0]");
    close(mean_one, 0.5, 1e-12, "E[posterior mean of arm 1]");
    close(superiority, 0.5, 1e-9, "E[P(theta_0 > theta_1 | history)]");
}

// -- adaptively weighted scores ------------------------------------------------------------------

const HAND_ARMS: [usize; 4] = [0, 1, 1, 0];
const HAND_PROBABILITIES: [f64; 8] = [0.5, 0.5, 0.5, 0.5, 0.3, 0.7, 0.2, 0.8];
const HAND_OUTCOMES: [f64; 4] = [1.0, 0.0, 1.0, 0.0];
const PRIOR: OutcomeModelPrior = OutcomeModelPrior { mean: 0.5, strength: 1.0 };

fn hand_scores() -> Vec<f64> {
    adaptive_scores(2, &HAND_ARMS, &HAND_PROBABILITIES, &HAND_OUTCOMES, PRIOR, &never).unwrap()
}

#[test]
fn scores_and_weighted_estimates_match_the_hand_calculation() {
    // Hand calculation with exact fractions (kappa = 1, m0 = 1/2): the scores of
    // arm 0 are (3/2, 3/4, 3/4, -3) and of arm 1 (1/2, -1/2, 37/28, 1/2).
    let scores = hand_scores();
    let arm0: Vec<f64> = scores.chunks_exact(2).map(|r| r[0]).collect();
    let arm1: Vec<f64> = scores.chunks_exact(2).map(|r| r[1]).collect();
    for (got, want) in arm0.iter().zip([1.5, 0.75, 0.75, -3.0]) {
        close(*got, want, 1e-14, "arm 0 score");
    }
    for (got, want) in arm1.iter().zip([0.5, -0.5, 37.0 / 28.0, 0.5]) {
        close(*got, want, 1e-14, "arm 1 score");
    }
    let go = |estimand, scheme| {
        weighted_estimate(2, &HAND_PROBABILITIES, &scores, estimand, scheme, None, 0.05).unwrap()
    };
    let uniform0 = go(Estimand::ArmMean(0), WeightScheme::Uniform);
    close(uniform0.estimate, 0.0, 1e-14, "uniform arm 0");
    close(uniform0.variance, 0.773_437_5, 1e-14, "uniform arm 0 variance");
    let uniform1 = go(Estimand::ArmMean(1), WeightScheme::Uniform);
    close(uniform1.estimate, 0.455_357_142_857_142_85, 1e-14, "uniform arm 1");
    close(uniform1.variance, 0.104_173_309_948_979_58, 1e-14, "uniform arm 1 variance");
    let contrast = go(Estimand::Contrast { plus: 1, minus: 0 }, WeightScheme::Uniform);
    close(contrast.estimate, 0.455_357_142_857_142_85, 1e-14, "uniform contrast");
    close(contrast.variance, 0.894_351_881_377_551, 1e-14, "uniform contrast variance");
    let stab0 = go(Estimand::ArmMean(0), WeightScheme::VarianceStabilizing);
    close(stab0.estimate, 0.274_014_265_317_294_7, 1e-12, "stabilized arm 0");
    close(stab0.variance, 0.530_083_232_621_941_3, 1e-12, "stabilized arm 0 variance");
    let stab1 = go(Estimand::ArmMean(1), WeightScheme::VarianceStabilizing);
    close(stab1.estimate, 0.493_688_892_666_138_7, 1e-12, "stabilized arm 1");
    let stab_contrast =
        go(Estimand::Contrast { plus: 1, minus: 0 }, WeightScheme::VarianceStabilizing);
    close(stab_contrast.estimate, 0.288_905_842_277_246_64, 1e-12, "stabilized contrast");
    close(stab_contrast.variance, 0.774_348_577_567_450_3, 1e-9, "stabilized contrast variance");
}

#[test]
fn a_low_probability_arm_is_refused_not_clipped() {
    let scores = hand_scores();
    let outcome = weighted_estimate(
        2,
        &HAND_PROBABILITIES,
        &scores,
        Estimand::ArmMean(0),
        WeightScheme::Uniform,
        None,
        0.25,
    );
    assert!(matches!(outcome, Err(AdaptiveInferenceError::Positivity { unit: 3, arm: 0, .. })));
    let bad = adaptive_scores(2, &[0, 1], &[0.7, 0.2, 0.5, 0.5], &[0.0, 1.0], PRIOR, &never);
    assert!(
        matches!(bad, Err(AdaptiveInferenceError::InvalidParameter(_))),
        "rows must sum to one"
    );
}

/// One enumerated step: `(arm, reward, running scores)`.
type HistoryStep = (usize, f64, [f64; 2]);

/// Callback receiving a history's probability and its steps.
type Visitor<'a> = dyn FnMut(f64, &[HistoryStep]) + 'a;

/// Exact expectation, over every history of an adaptive epsilon-greedy policy
/// with Bernoulli arms of known means, of the sum of per-unit scores and of
/// `sum h (score - truth)` for variance-stabilizing weights.
fn enumerate_policy(
    steps: usize,
    truth: [f64; 2],
    epsilon: f64,
    history: &mut Vec<HistoryStep>,
    probability: f64,
    visit: &mut Visitor<'_>,
) {
    if steps == 0 {
        visit(probability, history);
        return;
    }
    // Greedy arm by running means of the history (ties go to arm 0).
    let mut sum = [0.0; 2];
    let mut count = [0.0; 2];
    for &(arm, y, _) in history.iter() {
        sum[arm] += y;
        count[arm] += 1.0;
    }
    let means = [
        if count[0] > 0.0 { sum[0] / count[0] } else { 0.5 },
        if count[1] > 0.0 { sum[1] / count[1] } else { 0.5 },
    ];
    let greedy = usize::from(means[1] > means[0]);
    let mut e = [epsilon / 2.0; 2];
    e[greedy] += 1.0 - epsilon;
    for arm in 0..2 {
        for (y, p_y) in [(1.0, truth[arm]), (0.0, 1.0 - truth[arm])] {
            history.push((arm, y, e));
            enumerate_policy(steps - 1, truth, epsilon, history, probability * e[arm] * p_y, visit);
            history.pop();
        }
    }
}

#[test]
fn weighted_score_sums_are_martingales_under_an_adaptive_policy_by_enumeration() {
    // E[ sum_i h_i (Gamma_i(a) - mu_a) ] = 0 for predictable weights h_i, with
    // the expectation taken exactly over all 4^6 histories.
    let truth = [0.3, 0.7];
    let mut total = 0.0;
    let mut uniform = [0.0; 2];
    let mut stabilized = [0.0; 2];
    let mut naive_arm_zero_mean = 0.0;
    let mut naive_mass = 0.0;
    enumerate_policy(6, truth, 0.3, &mut Vec::new(), 1.0, &mut |p, history| {
        total += p;
        let arms: Vec<usize> = history.iter().map(|h| h.0).collect();
        let outcomes: Vec<f64> = history.iter().map(|h| h.1).collect();
        let probabilities: Vec<f64> = history.iter().flat_map(|h| h.2).collect();
        let scores = adaptive_scores(2, &arms, &probabilities, &outcomes, PRIOR, &never).unwrap();
        for arm in 0..2 {
            for (unit, row) in scores.chunks_exact(2).enumerate() {
                let centred = row[arm] - truth[arm];
                uniform[arm] += p * centred;
                let h = probabilities[unit * 2 + arm].sqrt();
                stabilized[arm] += p * h * centred;
            }
        }
        // The naive sample mean of arm 0 among histories that pulled it.
        let pulled: Vec<f64> = history.iter().filter(|h| h.0 == 0).map(|h| h.1).collect();
        if !pulled.is_empty() {
            naive_arm_zero_mean += p * pulled.iter().sum::<f64>() / pulled.len() as f64;
            naive_mass += p;
        }
    });
    close(total, 1.0, 1e-12, "history probabilities");
    for arm in 0..2 {
        close(uniform[arm], 0.0, 1e-12, "uniform score sum");
        close(stabilized[arm], 0.0, 1e-12, "stabilized score sum");
    }
    // The naive per-arm sample mean is biased under this adaptive policy,
    // which is why the scores are used: the bias is far above round-off.
    let naive = naive_arm_zero_mean / naive_mass;
    assert!((naive - truth[0]).abs() > 5e-3, "naive mean {naive} should be visibly biased");
}

fn uniform01(rng: &mut CausalRng) -> f64 {
    rng.next_f64()
}

#[test]
fn simulated_epsilon_greedy_trials_recover_the_contrast_with_calibrated_variance() {
    // Whole-procedure simulation with a known truth (contrast 0.2), fixed seeds.
    let (truth, replicates, horizon, epsilon) = ([0.4, 0.6], 400_u64, 300_usize, 0.3);
    let mut estimates = Vec::new();
    let mut variances = Vec::new();
    for replicate in 0..replicates {
        let mut rng = CausalRng::from_seed(0xAD_A971_0000 + replicate);
        let (mut sum, mut count) = ([0.0_f64; 2], [0.0_f64; 2]);
        let (mut arms, mut probabilities, mut outcomes) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..horizon {
            let means = [
                if count[0] > 0.0 { sum[0] / count[0] } else { 0.5 },
                if count[1] > 0.0 { sum[1] / count[1] } else { 0.5 },
            ];
            let mut e = [epsilon / 2.0; 2];
            if (means[0] - means[1]).abs() < f64::EPSILON {
                e = [0.5, 0.5];
            } else {
                e[usize::from(means[1] > means[0])] += 1.0 - epsilon;
            }
            let arm = usize::from(uniform01(&mut rng) >= e[0]);
            let y = f64::from(u8::from(uniform01(&mut rng) < truth[arm]));
            arms.push(arm);
            probabilities.extend_from_slice(&e);
            outcomes.push(y);
            sum[arm] += y;
            count[arm] += 1.0;
        }
        let scores = adaptive_scores(2, &arms, &probabilities, &outcomes, PRIOR, &never).unwrap();
        let fit = weighted_estimate(
            2,
            &probabilities,
            &scores,
            Estimand::Contrast { plus: 1, minus: 0 },
            WeightScheme::VarianceStabilizing,
            None,
            0.01,
        )
        .unwrap();
        estimates.push(fit.estimate);
        variances.push(fit.variance);
    }
    let n = replicates as f64;
    let mean = estimates.iter().sum::<f64>() / n;
    let empirical = estimates.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let reported = variances.iter().sum::<f64>() / n;
    // Monte Carlo standard error of the mean is about 0.004.
    assert!((mean - 0.2).abs() < 0.016, "mean {mean}");
    let ratio = reported / empirical;
    assert!((0.8..1.25).contains(&ratio), "reported variance over empirical variance {ratio}");
}

// -- confidence sequence ---------------------------------------------------------------------------

#[test]
fn confidence_sequence_radius_matches_the_closed_form() {
    // sigma = 1/2: radius = 0.5 sqrt((m + n0) ln((m + n0) / (n0 alpha^2))) / m.
    close(
        mixture_hoeffding_radius(100, 0.05, 100.0).unwrap() * 100.0,
        18.281_974_356_819_244,
        1e-9,
        "u_100",
    );
    close(mixture_hoeffding_radius(1, 0.05, 10.0).unwrap(), 4.091_287_144_531_515, 1e-9, "u_1");
    close(
        mixture_hoeffding_radius(400, 0.1, 50.0).unwrap() * 400.0,
        27.663_503_228_513_648,
        1e-9,
        "u_400",
    );
    assert!(mixture_hoeffding_radius(0, 0.05, 10.0).is_err());
    assert!(mixture_hoeffding_radius(5, 1.5, 10.0).is_err());
}

#[test]
fn the_sequence_survives_optional_stopping_where_the_fixed_sample_interval_does_not() {
    // Bernoulli(0.5) outcomes with mean exactly 0.5; look after every
    // observation up to 1000 and count trajectories that ever exclude the
    // truth. The sequence's miscoverage must be at most alpha = 0.05 (it is
    // far smaller); the fixed-sample 95% interval peeked at every look is
    // violated by a large fraction of trajectories.
    let (replicates, horizon, truth) = (600_u64, 1000_usize, 0.5);
    let (mut sequence_miss, mut naive_miss) = (0, 0);
    for replicate in 0..replicates {
        let mut rng = CausalRng::from_seed(0xC5_0000 + replicate);
        let values: Vec<f64> =
            (0..horizon).map(|_| f64::from(u8::from(uniform01(&mut rng) < truth))).collect();
        let mut sequence_violated = false;
        let mut naive_violated = false;
        let mut sum = 0.0;
        for (index, x) in values.iter().enumerate() {
            sum += x;
            let m = (index + 1) as f64;
            let radius = mixture_hoeffding_radius(index as u64 + 1, 0.05, 100.0).unwrap();
            sequence_violated |= (sum / m - truth).abs() >= radius;
            naive_violated |= (sum / m - truth).abs() >= 1.959_963_984_540_054 * 0.5 / m.sqrt();
        }
        sequence_miss += usize::from(sequence_violated);
        naive_miss += usize::from(naive_violated);
        // The final-time running intersection never excludes the truth when
        // no look did, and always lies in [0, 1].
        let fit = bounded_mean_confidence_sequence(&values, 0.05, 100.0, &never);
        if let Ok(fit) = fit {
            assert!(fit.lower >= 0.0 && fit.upper <= 1.0 && fit.lower <= fit.upper);
            assert_eq!(fit.count, horizon as u64);
            if !sequence_violated {
                assert!(fit.lower <= truth && truth <= fit.upper);
            }
        }
    }
    let sequence_rate = sequence_miss as f64 / replicates as f64;
    let naive_rate = naive_miss as f64 / replicates as f64;
    assert!(sequence_rate <= 0.05, "confidence-sequence miscoverage {sequence_rate}");
    assert!(naive_rate > 0.25, "peeked fixed-sample miscoverage {naive_rate}");
}

// -- group-sequential boundaries ------------------------------------------------------------------

#[test]
fn spending_functions_match_their_closed_forms() {
    // 2 - 2 Phi(1.959963984540054 / sqrt(0.5)) and 0.05 ln(1 + (e - 1)/2).
    close(
        SpendingFunction::OBrienFlemingType.spend(0.5, 0.05),
        0.005_574_596_680_784_305,
        1e-12,
        "OBF-type at 1/2",
    );
    close(
        SpendingFunction::PocockType.spend(0.5, 0.05),
        0.031_005_725_347_913_876,
        1e-12,
        "Pocock-type at 1/2",
    );
    close(SpendingFunction::OBrienFlemingType.spend(1.0, 0.05), 0.05, 1e-12, "total");
    close(SpendingFunction::PocockType.spend(1.0, 0.05), 0.05, 1e-12, "total");
}

fn one_sided(fractions: &[f64]) -> Vec<f64> {
    boundaries(
        fractions,
        0.025,
        SpendingFunction::OBrienFlemingType,
        Sidedness::OneSidedUpper,
        &never,
    )
    .unwrap()
    .critical
}

#[test]
fn obrien_fleming_type_boundaries_match_published_values() {
    // gsDesign sfLDOF, one-sided alpha 0.025, equally spaced looks.
    let three = one_sided(&[1.0 / 3.0, 2.0 / 3.0, 1.0]);
    for (got, want) in three.iter().zip([3.7103, 2.5114, 1.9930]) {
        close(*got, want, 2e-4, "K = 3");
    }
    let five = one_sided(&[0.2, 0.4, 0.6, 0.8, 1.0]);
    for (got, want) in five.iter().zip([4.8769, 3.3570, 2.6803, 2.2898, 2.0310]) {
        close(*got, want, 2e-4, "K = 5");
    }
}

#[test]
fn two_sided_boundaries_match_an_independent_grid_recursion() {
    // Computed by the trapezoid-rule recursion on a 4001-point grid in an
    // independent implementation (SciPy and NumPy).
    let b = boundaries(
        &[1.0 / 3.0, 2.0 / 3.0, 1.0],
        0.05,
        SpendingFunction::OBrienFlemingType,
        Sidedness::TwoSidedSymmetric,
        &never,
    )
    .unwrap();
    for (got, want) in b.critical.iter().zip([3.394_757_2, 2.406_732_9, 2.015_246_9]) {
        close(*got, want, 5e-5, "two-sided OBF-type");
    }
    let pocock = boundaries(
        &[1.0 / 3.0, 2.0 / 3.0, 1.0],
        0.05,
        SpendingFunction::PocockType,
        Sidedness::TwoSidedSymmetric,
        &never,
    )
    .unwrap();
    for (got, want) in pocock.critical.iter().zip([2.279_428_2, 2.294_911_4, 2.295_938_7]) {
        close(*got, want, 5e-5, "two-sided Pocock-type");
    }
    // The spending bookkeeping adds to the total.
    close(b.cumulative_spend[2], 0.05, 1e-12, "total alpha");
    close(b.spend_increments.iter().sum::<f64>(), 0.05, 1e-12, "increments");
}

#[test]
fn crossing_probabilities_reproduce_spending_and_the_repeated_testing_inflation() {
    let fractions = [1.0 / 3.0, 2.0 / 3.0, 1.0];
    let b = boundaries(
        &fractions,
        0.05,
        SpendingFunction::OBrienFlemingType,
        Sidedness::TwoSidedSymmetric,
        &never,
    )
    .unwrap();
    let null =
        crossing_probabilities(&fractions, &b.critical, Sidedness::TwoSidedSymmetric, 0.0, &never)
            .unwrap();
    for (got, want) in null.iter().zip(&b.spend_increments) {
        close(*got, *want, 1e-9, "stage-wise type I error");
    }
    close(null.iter().sum::<f64>(), 0.05, 1e-9, "overall type I error");
    // Five equally spaced looks at the fixed-sample 1.96: Armitage, McPherson
    // and Rowe (1969) report an overall level of 0.142.
    let naive = crossing_probabilities(
        &[0.2, 0.4, 0.6, 0.8, 1.0],
        &[1.959_963_984_540_054; 5],
        Sidedness::TwoSidedSymmetric,
        0.0,
        &never,
    )
    .unwrap();
    close(naive.iter().sum::<f64>(), 0.141_689_323_103_987_9, 1e-5, "inflated level");
    // Power under drift 2.5, independent grid recursion.
    let power =
        crossing_probabilities(&fractions, &b.critical, Sidedness::TwoSidedSymmetric, 2.5, &never)
            .unwrap();
    close(power.iter().sum::<f64>(), 0.695_360_400_356_374_3, 1e-4, "power");
    // One look: the closed-form power of the two-sided z test at drift 2.8.
    let single = crossing_probabilities(
        &[1.0],
        &[1.959_963_984_540_054],
        Sidedness::TwoSidedSymmetric,
        2.8,
        &never,
    )
    .unwrap();
    close(single[0], 0.799_556_871_435_651_5, 1e-9, "single-look power");
}

#[test]
fn simulated_random_walks_cross_at_the_spent_rates() {
    let fractions = [1.0 / 3.0, 2.0 / 3.0, 1.0];
    let b = boundaries(
        &fractions,
        0.05,
        SpendingFunction::PocockType,
        Sidedness::TwoSidedSymmetric,
        &never,
    )
    .unwrap();
    let replicates = 40_000_u64;
    let mut rng = CausalRng::from_seed(0x655E_0000);
    let mut stopped = [0_u64; 3];
    for _ in 0..replicates {
        let mut s = 0.0;
        let mut previous = 0.0;
        let mut z = [0.0; 3];
        for (k, &t) in fractions.iter().enumerate() {
            s += (t - previous).sqrt() * standard_normal(&mut rng);
            previous = t;
            z[k] = s / t.sqrt();
        }
        if let Some(k) = first_crossing(&z, &b.critical, Sidedness::TwoSidedSymmetric) {
            stopped[k] += 1;
        }
    }
    for (k, want) in b.spend_increments.iter().enumerate() {
        let rate = stopped[k] as f64 / replicates as f64;
        let sd = (want * (1.0 - want) / replicates as f64).sqrt();
        assert!((rate - want).abs() < 4.0 * sd, "look {k}: rate {rate}, spent {want}");
    }
}

#[test]
fn invalid_information_fractions_are_refused() {
    for bad in [vec![], vec![0.5, 0.5, 1.0], vec![0.0, 1.0], vec![0.5, 1.2]] {
        let outcome = boundaries(
            &bad,
            0.05,
            SpendingFunction::OBrienFlemingType,
            Sidedness::TwoSidedSymmetric,
            &never,
        );
        assert!(outcome.is_err(), "{bad:?} must be refused");
    }
    let cancelled = boundaries(
        &[0.5, 1.0],
        0.05,
        SpendingFunction::OBrienFlemingType,
        Sidedness::TwoSidedSymmetric,
        &|| true,
    );
    assert_eq!(cancelled, Err(AdaptiveInferenceError::Cancelled));
}
