//! Independent oracles for the participant-level adaptive family: three or four arms,
//! participant-level randomization, three or six binary responses per participant, historical
//! informative priors, Thompson allocation with a minimum exploration probability, and
//! expected-loss, futility and recruitment-cap decisions at completed-participant batch
//! boundaries, with a posterior claim and an anytime-valid claim read at every look.
//!
//! Nothing here checks the code against itself. The beta-binomial mass function is compared with
//! the gamma-function form of the same law; the posterior update is compared with a dense
//! quadrature of the exact likelihood; the confidence-sequence readings with a hand-rolled running
//! intersection of the closed-form mixture radius; and the whole-procedure simulation with an
//! exhaustive enumeration of a small procedure's terminal states.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::needless_range_loop,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    reason = "the oracles index per-arm and per-count columns and count small integers"
)]

use std::collections::BTreeMap;

use antecedent_stats::adaptive_inference::{
    AdaptiveInferenceError as Error, AnytimeSettings, ArmPrior, BetaPrior, BorrowingPlan,
    ContractGap, CorrelationPrior, DERIVED_UNMEASURED, DecisionRules, ExactPosterior, FutilityRule,
    HistoricalPrior, LookDecision, LookReading, ParticipantDeclaration, ParticipantHistory,
    ParticipantSimulationRequest, ParticipantSpec, StopReason, apply_exploration_floor,
    beta_binomial_pmf, design_effect, simulate_participant_history, simulate_participant_procedure,
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

/// A valid three-arm declaration; `edit` changes what a test is about.
fn declaration(edit: impl FnOnce(&mut ParticipantDeclaration)) -> ParticipantDeclaration {
    let mut d = ParticipantDeclaration {
        arms: 3,
        responses: Some(3),
        correlation: Some(0.2),
        correlation_prior: None,
        borrowing: Some(BorrowingPlan::uniform(3, BetaPrior::UNIFORM)),
        enrolment: Some(vec![40, 40, 40, 40]),
        lag_looks: 0,
        min_exploration: Some(0.1),
        // The posterior expected loss of arms within a tenth of one another stays far above
        // this at every size used here, so this threshold never ends a run.
        decisions: Some(DecisionRules {
            loss_threshold: 1e-12,
            futility: None,
            participant_cap: None,
        }),
        anytime: Some(AnytimeSettings { alpha: 0.05, tuning_samples: 30.0 }),
    };
    edit(&mut d);
    d
}

fn spec(edit: impl FnOnce(&mut ParticipantDeclaration)) -> ParticipantSpec {
    ParticipantSpec::new(declaration(edit)).expect("a valid declaration")
}

fn gaps(edit: impl FnOnce(&mut ParticipantDeclaration)) -> Vec<ContractGap> {
    ParticipantSpec::new(declaration(edit)).expect_err("a refused declaration").gaps
}

fn details(gaps: &[ContractGap]) -> Vec<&'static str> {
    gaps.iter().map(|g| g.detail()).collect()
}

// -- The beta-binomial law -------------------------------------------------------------------------

/// `ln B(x, y)`.
fn ln_beta(x: f64, y: f64) -> f64 {
    ln_gamma(x) + ln_gamma(y) - ln_gamma(x + y)
}

/// `ln C(m, s)`.
fn ln_choose(m: u32, s: u32) -> f64 {
    ln_gamma(f64::from(m) + 1.0) - ln_gamma(f64::from(s) + 1.0) - ln_gamma(f64::from(m - s) + 1.0)
}

/// The beta-binomial mass function through gamma functions, from the shapes `mu kappa` and
/// `(1 - mu) kappa` with `kappa = (1 - rho) / rho`.
fn gamma_form_pmf(m: u32, mu: f64, rho: f64) -> Vec<f64> {
    if rho == 0.0 {
        return (0..=m)
            .map(|s| {
                (ln_choose(m, s) + f64::from(s) * mu.ln() + f64::from(m - s) * (1.0 - mu).ln())
                    .exp()
            })
            .collect();
    }
    let kappa = (1.0 - rho) / rho;
    let (a, b) = (mu * kappa, (1.0 - mu) * kappa);
    (0..=m)
        .map(|s| {
            (ln_choose(m, s) + ln_beta(f64::from(s) + a, f64::from(m - s) + b) - ln_beta(a, b))
                .exp()
        })
        .collect()
}

#[test]
fn the_beta_binomial_mass_function_matches_its_gamma_function_form() {
    for (m, mu, rho) in [(3, 0.4, 1.0 / 6.0), (6, 0.25, 1.0 / 7.0), (6, 0.5, 0.4), (3, 0.9, 0.05)] {
        let product = beta_binomial_pmf(m, mu, rho);
        let gamma = gamma_form_pmf(m, mu, rho);
        assert_eq!(product.len(), m as usize + 1);
        for (s, (p, g)) in product.iter().zip(&gamma).enumerate() {
            close(*p, *g, 1e-11, &format!("m={m} mu={mu} rho={rho} s={s}"));
        }
        close(product.iter().sum::<f64>(), 1.0, 1e-12, "mass sums to one");
        // Mean m mu and variance m mu (1 - mu) (1 + (m - 1) rho).
        let mean: f64 = product.iter().enumerate().map(|(s, p)| s as f64 * p).sum();
        let variance: f64 =
            product.iter().enumerate().map(|(s, p)| (s as f64 - mean).powi(2) * p).sum();
        close(mean, f64::from(m) * mu, 1e-11, "mean");
        close(
            variance,
            f64::from(m) * mu * (1.0 - mu) * design_effect(m, rho),
            1e-10,
            "variance is the binomial variance times the design effect",
        );
    }
    // Zero correlation is the binomial.
    let binomial = beta_binomial_pmf(6, 0.3, 0.0);
    let expected: Vec<f64> = (0..=6)
        .map(|s| {
            (ln_choose(6, s) + f64::from(s) * 0.3_f64.ln() + f64::from(6 - s) * 0.7_f64.ln()).exp()
        })
        .collect();
    for (p, e) in binomial.iter().zip(&expected) {
        close(*p, *e, 1e-12, "binomial");
    }
    close(design_effect(6, 0.0), 1.0, 0.0, "independent responses have no design effect");
    close(design_effect(6, 0.4), 3.0, 1e-12, "1 + 5 * 0.4");
}

// -- The posterior update --------------------------------------------------------------------------

/// Mean and standard deviation of the exact posterior of an arm mean under the beta-binomial
/// likelihood, by midpoint quadrature of the gamma-function mass function on a dense grid.
fn exact_posterior(m: u32, rho: f64, histogram: &[u32], prior: (f64, f64)) -> (f64, f64) {
    let grid = 200_000_usize;
    let mut log_weight = Vec::with_capacity(grid);
    for g in 0..grid {
        let mu = (g as f64 + 0.5) / grid as f64;
        let pmf = gamma_form_pmf(m, mu, rho);
        let log_likelihood: f64 =
            histogram.iter().zip(&pmf).map(|(&count, p)| f64::from(count) * p.ln()).sum();
        log_weight
            .push(log_likelihood + (prior.0 - 1.0) * mu.ln() + (prior.1 - 1.0) * (1.0 - mu).ln());
    }
    let peak = log_weight.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = log_weight.iter().map(|w| (w - peak).exp()).collect();
    let total: f64 = weights.iter().sum();
    let mean: f64 =
        weights.iter().enumerate().map(|(g, w)| w * (g as f64 + 0.5) / grid as f64).sum::<f64>()
            / total;
    let variance: f64 = weights
        .iter()
        .enumerate()
        .map(|(g, w)| w * ((g as f64 + 0.5) / grid as f64 - mean).powi(2))
        .sum::<f64>()
        / total;
    (mean, variance.sqrt())
}

/// The library's posterior of arm 0 given its success-count histogram (other arms empty).
fn posterior_of(spec: &ParticipantSpec, histogram: &[u32]) -> (f64, f64) {
    let mut counts = vec![vec![0_u64; spec.responses() as usize + 1]; spec.arms()];
    counts[0] = histogram.iter().map(|&c| u64::from(c)).collect();
    let posterior = spec.posterior(&counts, None, &never).expect("a posterior");
    (posterior.mean[0], posterior.sd[0])
}

#[test]
fn before_any_participant_completes_the_posterior_is_the_resolved_prior() {
    let s = spec(|d| {
        d.responses = Some(3);
        d.correlation = Some(0.3);
        d.borrowing = Some(BorrowingPlan::uniform(3, BetaPrior { alpha: 2.5, beta: 4.0 }));
    });
    let (mean, sd) = posterior_of(&s, &[0, 0, 0, 0]);
    let (a, b) = (2.5_f64, 4.0_f64);
    close(mean, a / (a + b), 1e-10, "prior mean");
    close(sd, (a * b / ((a + b).powi(2) * (a + b + 1.0))).sqrt(), 1e-10, "prior sd");
    close(s.design_effect().unwrap(), 1.6, 1e-12, "design effect");
    let counts = vec![vec![0_u64; 4]; 2];
    assert!(matches!(s.posterior(&counts, None, &never), Err(Error::Shape(_))));
    let wrong = vec![vec![0_u64; 3]; 3];
    assert!(matches!(s.posterior(&wrong, None, &never), Err(Error::Shape(_))));
}

#[test]
fn at_zero_correlation_the_posterior_is_the_conjugate_beta() {
    // Independent responses: the posterior of 12 participants' 36 responses under a Beta(2.5, 4)
    // prior is Beta(2.5 + 16, 4 + 20); the dense quadrature of the binomial likelihood agrees.
    let s = spec(|d| {
        d.responses = Some(3);
        d.correlation = Some(0.0);
        d.borrowing = Some(BorrowingPlan::uniform(3, BetaPrior { alpha: 2.5, beta: 4.0 }));
    });
    let histogram = [3_u32, 4, 3, 2];
    let (mean, sd) = posterior_of(&s, &histogram);
    let (a, b) = (18.5_f64, 24.0_f64);
    close(mean, a / (a + b), 1e-9, "conjugate mean");
    close(sd, (a * b / ((a + b).powi(2) * (a + b + 1.0))).sqrt(), 1e-9, "conjugate sd");
    let (dense_mean, dense_sd) = exact_posterior(3, 0.0, &histogram, (2.5, 4.0));
    close(mean, dense_mean, 1e-5, "mean");
    close(sd, dense_sd, 1e-5, "standard deviation");
}

#[test]
fn with_correlation_the_posterior_is_the_exact_beta_binomial_posterior() {
    // (responses, correlation, histogram of success counts, prior). The reference is the dense
    // quadrature of the gamma-function beta-binomial likelihood. The design-effect Beta update
    // differs from it by up to a fifth of a standard deviation in the centre and a few percent in
    // the width, so this is a statement the update could not pass.
    type Case = (u32, f64, Vec<u32>, (f64, f64));
    let cases: [Case; 5] = [
        (3, 0.3, vec![3, 4, 3, 2], (1.0, 1.0)),
        (3, 0.3, vec![30, 40, 30, 20], (1.0, 1.0)),
        (6, 0.4, vec![1, 1, 2, 3, 1, 1, 1], (1.0, 1.0)),
        (6, 0.4, vec![10, 10, 20, 30, 10, 10, 10], (1.0, 1.0)),
        (6, 0.2, vec![0, 0, 1, 3, 4, 2, 1], (2.5, 4.0)),
    ];
    for (responses, rho, histogram, prior) in cases {
        let s = spec(|d| {
            d.responses = Some(responses);
            d.correlation = Some(rho);
            d.borrowing =
                Some(BorrowingPlan::uniform(3, BetaPrior { alpha: prior.0, beta: prior.1 }));
        });
        let (mean, sd) = posterior_of(&s, &histogram);
        let (dense_mean, dense_sd) = exact_posterior(responses, rho, &histogram, prior);
        let what = format!("m={responses} rho={rho} n={}", histogram.iter().sum::<u32>());
        close(mean, dense_mean, 2e-6, &format!("{what}: mean"));
        close(sd, dense_sd, 2e-6, &format!("{what}: standard deviation"));
    }
}

// -- Historical priors and their effective sample size ---------------------------------------------

fn source(label: &str, successes: u64, trials: u64, discount: f64) -> HistoricalPrior {
    HistoricalPrior { source: label.to_owned(), successes, trials, discount }
}

#[test]
fn historical_sources_resolve_to_a_beta_with_their_effective_sample_size() {
    // Uniform initial prior, a source of 30/100 at discount 0.5 and one of 10/40 at discount 1:
    // alpha = 1 + 15 + 10, beta = 1 + 0.5 * 70 + 30; worth 0.5 * 100 + 40 = 90 borrowed responses.
    let arm = ArmPrior {
        initial: BetaPrior::UNIFORM,
        historical: vec![source("registry", 30, 100, 0.5), source("pilot", 10, 40, 1.0)],
    };
    let resolved = arm.resolve().expect("valid sources");
    close(resolved.prior.alpha, 26.0, 1e-12, "alpha");
    close(resolved.prior.beta, 66.0, 1e-12, "beta");
    close(resolved.effective_sample_size, 92.0, 1e-12, "prior effective sample size alpha + beta");
    close(resolved.borrowed_sample_size, 90.0, 1e-12, "borrowed sample size");
    assert_eq!(resolved.sources, arm.historical, "every source and discount is recorded");

    // A zero discount borrows nothing; a unit discount counts the source as current data.
    let ignored = ArmPrior {
        initial: BetaPrior::UNIFORM,
        historical: vec![source("registry", 30, 100, 0.0)],
    }
    .resolve()
    .unwrap();
    close(ignored.effective_sample_size, 2.0, 0.0, "a zero discount borrows nothing");
    close(ignored.borrowed_sample_size, 0.0, 0.0, "borrowed");
    let full = ArmPrior {
        initial: BetaPrior::UNIFORM,
        historical: vec![source("registry", 30, 100, 1.0)],
    }
    .resolve()
    .unwrap();
    close(full.prior.alpha, 31.0, 0.0, "alpha");
    close(full.prior.beta, 71.0, 0.0, "beta");

    // At most sixteen sources per arm.
    let sources = |n: usize| ArmPrior {
        initial: BetaPrior::UNIFORM,
        historical: (0..n).map(|i| source(&format!("study {i}"), 1, 10, 0.1)).collect(),
    };
    assert!(sources(16).resolve().is_some());
    assert!(sources(17).resolve().is_none());

    // Invalid sources resolve to nothing: they are refused, not clipped.
    for bad in [
        source("registry", 30, 100, 1.5),
        source("registry", 30, 100, -0.1),
        source("registry", 30, 100, f64::NAN),
        source("registry", 101, 100, 0.5),
        source("   ", 30, 100, 0.5),
        source("registry", 30, 100_001, 0.5),
    ] {
        let arm = ArmPrior { initial: BetaPrior::UNIFORM, historical: vec![bad] };
        assert!(arm.resolve().is_none());
    }
}

#[test]
fn borrowed_priors_enter_the_procedure_and_its_identity() {
    let plan = |discount: f64| BorrowingPlan {
        arms: vec![
            ArmPrior {
                initial: BetaPrior::UNIFORM,
                historical: vec![source("control registry", 40, 100, discount)],
            },
            ArmPrior::without_history(BetaPrior::UNIFORM),
            ArmPrior::without_history(BetaPrior::UNIFORM),
        ],
    };
    let a = spec(|d| d.borrowing = Some(plan(0.5)));
    let b = spec(|d| d.borrowing = Some(plan(0.6)));
    let none = spec(|_| {});
    // The control starts from Beta(21, 31); the treatment arms from the initial prior.
    close(a.priors()[0].prior.alpha, 21.0, 1e-12, "control alpha");
    close(a.priors()[0].prior.beta, 31.0, 1e-12, "control beta");
    close(a.priors()[1].effective_sample_size, 2.0, 0.0, "treatment prior effective size");
    assert_ne!(a.identity(), b.identity(), "the discount is part of the declaration");
    assert_ne!(a.identity(), none.identity(), "borrowing is part of the declaration");
    assert_eq!(a.identity(), spec(|d| d.borrowing = Some(plan(0.5))).identity());
    // The first posterior is the resolved prior.
    let (mean, _) = posterior_of(&a, &[0, 0, 0, 0]);
    close(mean, 21.0 / 52.0, 1e-10, "posterior mean before data");
}

#[test]
fn borrowing_is_frozen_after_the_first_look() {
    let original = spec(|_| {});
    let proposed = BorrowingPlan {
        arms: vec![
            ArmPrior {
                initial: BetaPrior::UNIFORM,
                historical: vec![source("control registry", 40, 100, 0.5)],
            },
            ArmPrior::without_history(BetaPrior::UNIFORM),
            ArmPrior::without_history(BetaPrior::UNIFORM),
        ],
    };
    // Before recruitment the plan may still be amended; the amended declaration has its own
    // identity.
    let amended = original.amend_borrowing(&proposed, 0).expect("borrowing is open before a look");
    assert_ne!(amended.identity(), original.identity());
    close(amended.priors()[0].prior.alpha, 21.0, 1e-12, "the amended prior");
    // After the first look a different plan is refused with its own detail...
    match original.amend_borrowing(&proposed, 1) {
        Err(Error::PremiseFailed { detail, .. }) => {
            assert_eq!(detail, "adaptive_inference.borrowing_frozen");
        }
        other => panic!("expected the frozen-borrowing refusal, got {other:?}"),
    }
    // ...and the same plan is not a change.
    let unchanged = original.amend_borrowing(original.borrowing(), 3).expect("no change");
    assert_eq!(unchanged.identity(), original.identity());
    // An invalid plan is refused even before the first look.
    let invalid = BorrowingPlan { arms: vec![ArmPrior::without_history(BetaPrior::UNIFORM)] };
    assert!(matches!(original.amend_borrowing(&invalid, 0), Err(Error::InvalidParameter(_))));
}

// -- The minimum exploration probability -----------------------------------------------------------

#[test]
fn the_exploration_floor_clamps_and_renormalizes_to_a_valid_distribution() {
    let rows: [Vec<f64>; 7] = [
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.97, 0.02, 0.01, 0.0],
        vec![0.5, 0.5, 0.0, 0.0],
        vec![0.25; 4],
        vec![0.6, 0.3, 0.07, 0.03],
        vec![0.0, 1.0, 0.0],
        vec![0.999_9, 0.000_1],
    ];
    for floor in [0.05, 0.1, 0.2] {
        for row in &rows {
            if floor * row.len() as f64 > 1.0 {
                continue;
            }
            let mut clamped = row.clone();
            apply_exploration_floor(&mut clamped, floor);
            let what = format!("floor {floor} row {row:?} -> {clamped:?}");
            close(clamped.iter().sum::<f64>(), 1.0, 1e-12, &what);
            assert!(clamped.iter().all(|&p| p >= floor - 1e-15), "{what}");
            // Order is preserved, and a second application changes nothing.
            for i in 0..row.len() {
                for j in 0..row.len() {
                    if row[i] > row[j] {
                        assert!(clamped[i] >= clamped[j] - 1e-15, "{what}");
                    }
                }
            }
            let mut again = clamped.clone();
            apply_exploration_floor(&mut again, floor);
            for (x, y) in clamped.iter().zip(&again) {
                close(*x, *y, 1e-12, "idempotent");
            }
        }
    }
    // Closed forms. [1, 0, 0, 0] at 0.1: the three zero arms sit on the floor and the leader
    // takes the rest, 0.7. [0.6, 0.3, 0.07, 0.03] at 0.1: one pass of clamping would leave the
    // last two at 0.1 and push the second arm to 0.3 / 0.9 * 0.8; the exact result pins them and
    // rescales the rest in proportion, 0.8 * (0.6, 0.3) / 0.9.
    let mut leader = vec![1.0, 0.0, 0.0, 0.0];
    apply_exploration_floor(&mut leader, 0.1);
    for (got, want) in leader.iter().zip([0.7, 0.1, 0.1, 0.1]) {
        close(*got, want, 1e-15, "leader");
    }
    let mut mixed = vec![0.6, 0.3, 0.07, 0.03];
    apply_exploration_floor(&mut mixed, 0.1);
    for (got, want) in mixed.iter().zip([0.8 * 0.6 / 0.9, 0.8 * 0.3 / 0.9, 0.1, 0.1]) {
        close(*got, want, 1e-12, "mixed");
    }
    // A floor of 1 / arms is equal randomization.
    let mut equal = vec![0.9, 0.05, 0.05];
    apply_exploration_floor(&mut equal, 1.0 / 3.0);
    for p in equal {
        close(p, 1.0 / 3.0, 1e-12, "equal randomization");
    }
}

fn history_of(s: &ParticipantSpec, truth: &[f64], seed: u64) -> ParticipantHistory {
    simulate_participant_history(s, truth, None, seed, &never).expect("a history")
}

#[test]
fn every_batch_is_randomized_under_probabilities_that_respect_the_floor() {
    // The expected-loss threshold is too small for a run with this much uncertainty to meet, so
    // every look is taken.
    let s = spec(|d| {
        d.min_exploration = Some(0.3);
        d.enrolment = Some(vec![60; 6]);
        d.decisions =
            Some(DecisionRules { loss_threshold: 1e-300, futility: None, participant_cap: None });
    });
    // Two weaker arms and one better: the unfloored allocation would starve the weaker arms, so
    // the floor binds from the second look on.
    let history = history_of(&s, &[0.35, 0.55, 0.35], 7);
    let readings = s.read_history(&history, &never).expect("the history replays");
    assert!(readings.len() >= 2, "the run has several looks");
    let mut bound = false;
    for reading in &readings {
        let row = &reading.assignment_probabilities;
        close(row.iter().sum::<f64>(), 1.0, 1e-12, "a distribution");
        assert!(row.iter().all(|&p| p >= 0.3 - 1e-12), "floor violated: {row:?}");
        assert!(row[1] <= 0.4 + 1e-12, "the leader cannot exceed 1 - 2 * 0.3: {row:?}");
        bound |= (row[0] - 0.3).abs() < 1e-9 || (row[2] - 0.3).abs() < 1e-9;
    }
    assert!(bound, "the floor binds at some look");
    // The recorded probabilities are what each participant was randomized under.
    let mut participant = 0;
    for (look, &size) in history.batches.iter().enumerate() {
        for _ in 0..size {
            assert_eq!(history.assignment[participant], readings[look].assignment_probabilities);
            participant += 1;
        }
    }
}

// -- Decisions at batch boundaries -----------------------------------------------------------------

fn request(replications: u32, seed: u64) -> ParticipantSimulationRequest {
    ParticipantSimulationRequest { replications, seed, true_correlation: None }
}

#[test]
fn the_recruitment_cap_ends_the_procedure_at_the_batch_boundary_that_reaches_it() {
    let s = spec(|d| {
        d.enrolment = Some(vec![30; 4]);
        d.decisions = Some(DecisionRules {
            loss_threshold: 1e-12,
            futility: None,
            participant_cap: Some(60),
        });
    });
    let simulation =
        simulate_participant_procedure(&s, &[0.3, 0.3, 0.3], &request(30, 4), &never).unwrap();
    for replicate in &simulation.replicates {
        assert_eq!(replicate.reason, StopReason::RecruitmentCap);
        assert_eq!((replicate.look, replicate.enrolled), (1, 60));
    }
    // A cap between two boundaries waits for the next one.
    let between = spec(|d| {
        d.enrolment = Some(vec![30; 4]);
        d.decisions = Some(DecisionRules {
            loss_threshold: 1e-12,
            futility: None,
            participant_cap: Some(61),
        });
    });
    let simulation =
        simulate_participant_procedure(&between, &[0.3, 0.3, 0.3], &request(10, 4), &never)
            .unwrap();
    assert!(simulation.replicates.iter().all(|r| r.look == 2 && r.enrolled == 90));
}

#[test]
fn expected_loss_decides_for_the_arm_whose_loss_fell_below_its_threshold() {
    let s = spec(|d| {
        d.enrolment = Some(vec![150, 150]);
        d.decisions =
            Some(DecisionRules { loss_threshold: 0.01, futility: None, participant_cap: None });
    });
    let simulation =
        simulate_participant_procedure(&s, &[0.1, 0.9, 0.1], &request(40, 11), &never).unwrap();
    for replicate in &simulation.replicates {
        assert_eq!(replicate.reason, StopReason::ExpectedLoss);
        assert_eq!((replicate.look, replicate.decision, replicate.enrolled), (0, 1, 150));
    }
    let summary = simulation.summarize(3).unwrap();
    close(summary.stop_fractions.expected_loss, 1.0, 1e-12, "every run decided");
    close(summary.decision_fractions[1], 1.0, 1e-9, "for the best arm");
}

#[test]
fn futility_decides_for_the_control_when_no_treatment_is_likely_to_beat_it() {
    let s = spec(|d| {
        d.enrolment = Some(vec![300, 300]);
        d.decisions = Some(DecisionRules {
            loss_threshold: 1e-12,
            futility: Some(FutilityRule { margin: 0.15, threshold: 0.1 }),
            participant_cap: None,
        });
    });
    let simulation =
        simulate_participant_procedure(&s, &[0.3, 0.3, 0.3], &request(40, 21), &never).unwrap();
    let futile: Vec<_> =
        simulation.replicates.iter().filter(|r| r.reason == StopReason::Futility).collect();
    assert!(futile.len() >= 32, "most equal-arm runs stop for futility, got {}", futile.len());
    assert!(futile.iter().all(|r| r.decision == 0), "futility decides for the control");
    assert!(futile.iter().all(|r| r.look == 0 || r.look == 1));
    // A truth with a large advantage is never futile.
    let advantage =
        simulate_participant_procedure(&s, &[0.2, 0.7, 0.2], &request(20, 21), &never).unwrap();
    assert!(advantage.replicates.iter().all(|r| r.reason != StopReason::Futility));
}

#[test]
fn rules_are_taken_in_order_expected_loss_then_futility_then_the_cap() {
    // Equal arms: the futility rule holds and so does a loose expected-loss rule, which comes first.
    let both = spec(|d| {
        d.enrolment = Some(vec![300, 300]);
        d.decisions = Some(DecisionRules {
            loss_threshold: 0.5,
            futility: Some(FutilityRule { margin: 0.15, threshold: 0.1 }),
            participant_cap: Some(300),
        });
    });
    let simulation =
        simulate_participant_procedure(&both, &[0.3, 0.3, 0.3], &request(20, 2), &never).unwrap();
    assert!(
        simulation.replicates.iter().all(|r| r.reason == StopReason::ExpectedLoss && r.look == 0)
    );
    // Without the loose loss rule, futility comes before a cap that is also reached.
    let futility_first = spec(|d| {
        d.enrolment = Some(vec![300, 300]);
        d.decisions = Some(DecisionRules {
            loss_threshold: 1e-12,
            futility: Some(FutilityRule { margin: 0.15, threshold: 0.1 }),
            participant_cap: Some(300),
        });
    });
    let simulation =
        simulate_participant_procedure(&futility_first, &[0.3, 0.3, 0.3], &request(20, 2), &never)
            .unwrap();
    assert!(
        simulation
            .replicates
            .iter()
            .all(|r| matches!(r.reason, StopReason::Futility | StopReason::RecruitmentCap)
                && r.look == 0)
    );
    assert!(
        simulation.replicates.iter().filter(|r| r.reason == StopReason::Futility).count() >= 15
    );
}

#[test]
fn nothing_is_decided_before_a_participant_has_completed() {
    // One look of delay: at the first look nobody has completed, so the posterior is the prior
    // and the prior-only expected loss (about a quarter for three arms) is above the threshold.
    let s = spec(|d| {
        d.lag_looks = 1;
        d.enrolment = Some(vec![40; 4]);
        d.decisions =
            Some(DecisionRules { loss_threshold: 0.1, futility: None, participant_cap: None });
    });
    let history = history_of(&s, &[0.1, 0.9, 0.1], 5);
    let readings = s.read_history(&history, &never).unwrap();
    let first = &readings[0];
    assert_eq!(first.completed, vec![0, 0, 0]);
    assert_eq!(first.decision, LookDecision::Continue);
    for arm in 0..3 {
        close(first.posterior.mean[arm], 0.5, 1e-10, "prior mean");
        close(first.posterior.sd[arm], (1.0_f64 / 12.0).sqrt(), 1e-9, "prior sd");
    }
    // Participants of the last batches never complete when the procedure ends.
    let last_batch = *history.batches.last().unwrap() as usize;
    assert!(history.successes[history.successes.len() - last_batch..].iter().all(Option::is_none));
    let completed: u32 = readings.last().unwrap().completed.iter().sum();
    assert_eq!(completed as usize, history.successes.iter().flatten().count());
}

// -- Confidence sequences at every look ------------------------------------------------------------

/// The closed-form radius of the normal-mixture Hoeffding sequence after `n` observations.
fn radius(n: f64, alpha: f64, tuning: f64) -> f64 {
    0.5 * ((n + tuning) * ((n + tuning) / (tuning * alpha * alpha)).ln()).sqrt() / n
}

/// The running-intersection sequence of `values` (in pull order) after each prefix.
fn running_intersection(values: &[f64], alpha: f64, tuning: f64) -> (f64, f64) {
    let (mut lower, mut upper, mut sum) = (0.0_f64, 1.0_f64, 0.0);
    for (index, value) in values.iter().enumerate() {
        sum += value;
        let n = index as f64 + 1.0;
        let mean = sum / n;
        let r = radius(n, alpha, tuning);
        lower = lower.max(mean - r);
        upper = upper.min(mean + r);
    }
    (lower, upper)
}

#[test]
fn confidence_sequence_readings_at_every_look_match_an_independent_computation() {
    let s = spec(|d| {
        d.lag_looks = 1;
        d.enrolment = Some(vec![40; 5]);
        d.anytime = Some(AnytimeSettings { alpha: 0.06, tuning_samples: 25.0 });
    });
    let history = history_of(&s, &[0.3, 0.4, 0.5], 31);
    let readings = s.read_history(&history, &never).unwrap();
    assert_eq!(readings.len(), history.batches.len(), "a reading at every performed look");
    assert!(readings.len() >= 3, "the run has several looks");
    let m = f64::from(s.responses());
    // Per-arm level: alpha over the K - 1 = 2 comparisons, each over its two arms.
    let alpha_arm = 0.06 / (2.0 * 2.0);
    for (look, reading) in readings.iter().enumerate() {
        assert_eq!(reading.look, look);
        assert_eq!(reading.posterior.paradigm, "bayesian_posterior");
        let f = &reading.frequentist;
        assert_eq!(f.paradigm, "anytime_valid_frequentist");
        assert_eq!(f.calibration, DERIVED_UNMEASURED);
        close(f.alpha, 0.06, 0.0, "family alpha");
        close(f.alpha_per_comparison, 0.03, 1e-15, "alpha per comparison");
        close(f.alpha_per_arm, alpha_arm, 1e-15, "alpha per arm");
        // Participants whose follow-up completed by this look, in enrolment order.
        let completed_batches = look; // lag of one look
        let end: usize = history.batches[..completed_batches].iter().map(|&n| n as usize).sum();
        for arm in 0..3 {
            let proportions: Vec<f64> = (0..end)
                .filter(|&i| history.arms[i] == arm)
                .map(|i| f64::from(history.successes[i].unwrap()) / m)
                .collect();
            assert_eq!(f.arms[arm].count as usize, proportions.len());
            assert_eq!(reading.completed[arm] as usize, proportions.len());
            let (lower, upper) = running_intersection(&proportions, alpha_arm, 25.0);
            close(f.arms[arm].lower, lower, 1e-12, &format!("look {look} arm {arm} lower"));
            close(f.arms[arm].upper, upper, 1e-12, &format!("look {look} arm {arm} upper"));
            match f.arms[arm].mean {
                Some(mean) => {
                    close(
                        mean,
                        proportions.iter().sum::<f64>() / proportions.len() as f64,
                        1e-12,
                        "mean",
                    );
                }
                None => assert!(proportions.is_empty()),
            }
        }
        // The comparisons against the control are the differences of the arm sequences.
        assert_eq!(f.comparisons.len(), 2);
        for comparison in &f.comparisons {
            close(comparison.lower, f.arms[comparison.arm].lower - f.arms[0].upper, 1e-15, "lower");
            close(comparison.upper, f.arms[comparison.arm].upper - f.arms[0].lower, 1e-15, "upper");
        }
        // The data-selected best treatment is the non-control arm with the highest posterior mean,
        // and its comparison is the matching member of the simultaneous family.
        let means = &reading.posterior.mean;
        let best = if means[2] > means[1] { 2 } else { 1 };
        assert_eq!(f.best_treatment, best);
        assert_eq!(f.best_treatment_comparison, f.comparisons[best - 1]);
    }
}

#[test]
fn the_simultaneous_family_covers_the_data_selected_best_treatment() {
    // Equal arms: the "best" treatment is selected from noise, which would make a naive interval
    // for it miss often. The family holds every treatment at once, so the selected contrast is
    // covered at every look and at the stopping time.
    let s = spec(|d| d.enrolment = Some(vec![40; 5]));
    let simulation =
        simulate_participant_procedure(&s, &[0.4, 0.4, 0.4], &request(200, 77), &never).unwrap();
    let summary = simulation.summarize(3).unwrap();
    assert!(summary.covered_every_look >= 0.98, "{}", summary.covered_every_look);
    assert!(summary.covered_at_stop >= 0.98, "{}", summary.covered_at_stop);
    assert!(summary.contrast_covered_at_stop >= 0.98, "{}", summary.contrast_covered_at_stop);
}

// -- Replay of a retained history ------------------------------------------------------------------

fn truncate(history: &ParticipantHistory, looks: usize, lag: usize) -> ParticipantHistory {
    let batches = history.batches[..looks].to_vec();
    let units: usize = batches.iter().map(|&n| n as usize).sum();
    let mut successes = history.successes[..units].to_vec();
    let mut index = 0;
    for (look, &size) in batches.iter().enumerate() {
        for _ in 0..size {
            if look + lag >= looks {
                successes[index] = None;
            }
            index += 1;
        }
    }
    ParticipantHistory {
        batches,
        arms: history.arms[..units].to_vec(),
        successes,
        assignment: history.assignment[..units].to_vec(),
    }
}

#[test]
fn a_retained_history_replays_and_a_tampered_one_is_refused() {
    let borrowed = BorrowingPlan {
        arms: vec![
            ArmPrior {
                initial: BetaPrior::UNIFORM,
                historical: vec![source("control registry", 40, 100, 0.5)],
            },
            ArmPrior::without_history(BetaPrior::UNIFORM),
            ArmPrior::without_history(BetaPrior::UNIFORM),
        ],
    };
    let s = spec(|d| {
        d.lag_looks = 1;
        d.enrolment = Some(vec![40; 5]);
        d.borrowing = Some(borrowed.clone());
    });
    let history = history_of(&s, &[0.35, 0.45, 0.4], 3);
    assert_eq!(history.batches.len(), 5, "no decision ends this run early");
    let readings =
        s.read_history(&history, &never).expect("a history the procedure produced replays");
    assert_eq!(readings.len(), history.batches.len());
    assert!(matches!(readings.last().unwrap().decision, LookDecision::Stop { .. }));
    assert!(readings[..readings.len() - 1].iter().all(|r| r.decision == LookDecision::Continue));
    // Deterministic: the same history reads the same.
    assert_eq!(readings, s.read_history(&history, &never).unwrap());

    let detail = |outcome: Result<Vec<LookReading>, Error>| match outcome {
        Err(Error::PremiseFailed { detail, .. }) => detail,
        other => panic!("expected a premise refusal, got {other:?}"),
    };

    // A retained probability that is not the policy's.
    let mut edited = history.clone();
    edited.assignment[0][0] += 1e-6;
    edited.assignment[0][1] -= 1e-6;
    assert_eq!(
        detail(s.read_history(&edited, &never)),
        "adaptive_inference.assignment_not_replayed"
    );

    // The same history under other borrowing: the declared priors are not the ones the
    // allocation was computed under.
    let other = s
        .amend_borrowing(
            &BorrowingPlan {
                arms: vec![
                    ArmPrior {
                        initial: BetaPrior::UNIFORM,
                        historical: vec![source("control registry", 100, 100, 1.0)],
                    },
                    ArmPrior::without_history(BetaPrior::UNIFORM),
                    ArmPrior::without_history(BetaPrior::UNIFORM),
                ],
            },
            0,
        )
        .unwrap();
    assert_eq!(
        detail(other.read_history(&history, &never)),
        "adaptive_inference.assignment_not_replayed"
    );

    // A procedure that stopped without a declared decision (a prefix of a run that went on).
    let prefix = truncate(&history, 3, 1);
    assert_eq!(detail(s.read_history(&prefix, &never)), "adaptive_inference.stopping_not_replayed");

    // A procedure that went on past a look at which the declared decisions stop it.
    let capped = spec(|d| {
        d.lag_looks = 1;
        d.enrolment = Some(vec![40; 5]);
        d.borrowing = Some(borrowed.clone());
        d.decisions = Some(DecisionRules {
            loss_threshold: 1e-12,
            futility: None,
            participant_cap: Some(80),
        });
    });
    assert_eq!(
        detail(capped.read_history(&history, &never)),
        "adaptive_inference.stopping_not_replayed"
    );

    // Structural disagreements are shape errors, not replay refusals.
    let mut wrong_batches = history.clone();
    wrong_batches.batches[0] = 39;
    assert!(matches!(s.read_history(&wrong_batches, &never), Err(Error::Shape(_))));
    // A missing response in a batch whose follow-up delay has elapsed is a partial batch,
    // refused as such rather than as a structural disagreement.
    let mut wrong_availability = history.clone();
    wrong_availability.successes[0] = None;
    assert!(matches!(
        s.read_history(&wrong_availability, &never),
        Err(Error::PremiseFailed { detail: "adaptive_inference.batch_incomplete", .. })
    ));
    let mut too_many = history.clone();
    too_many.successes[0] = Some(9);
    assert!(matches!(s.read_history(&too_many, &never), Err(Error::InvalidParameter(_))));
}

// -- The whole-procedure simulation against an exhaustive enumeration ------------------------------

#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
struct Tally {
    /// `counts[arm][s]`: completed participants of the arm with `s` successes (three responses).
    counts: [[u64; 4]; 2],
}

/// Everything the control flow of a two-arm, no-delay procedure does, enumerated: every
/// allocation and every success count of every participant, merged by the tally they lead to,
/// with the library's own posterior, allocation and loss kernels (checked above) but none of its
/// driver.
struct Enumeration<'a> {
    spec: &'a ParticipantSpec,
    pmfs: [Vec<f64>; 2],
}

type Outcomes = BTreeMap<(usize, &'static str, usize), f64>;

impl Enumeration<'_> {
    fn posterior(&self, tally: &Tally) -> ExactPosterior {
        let counts: Vec<Vec<u64>> = tally.counts.iter().map(|c| c.to_vec()).collect();
        self.spec.posterior(&counts, None, &never).unwrap()
    }

    fn row(&self, tally: &Tally) -> Vec<f64> {
        let best = self.posterior(tally).probability_best;
        let total: f64 = best.iter().map(|p| p.max(0.0)).sum();
        let mut row: Vec<f64> = best.iter().map(|p| p.max(0.0) / total).collect();
        apply_exploration_floor(&mut row, self.spec.min_exploration());
        row
    }

    fn terminal_states(&self) -> Outcomes {
        let looks = self.spec.enrolment().len();
        let mut outcomes = Outcomes::new();
        let mut live: BTreeMap<Tally, f64> = BTreeMap::from([(Tally::default(), 1.0)]);
        for look in 0..looks {
            let mut enrolled: BTreeMap<Tally, f64> = BTreeMap::new();
            for (tally, probability) in live {
                // The batch is randomized under the allocation from the tally before it.
                let row = self.row(&tally);
                let mut stage = BTreeMap::from([(tally, probability)]);
                for _ in 0..self.spec.enrolment()[look] {
                    let mut after: BTreeMap<Tally, f64> = BTreeMap::new();
                    for (t, q) in stage {
                        for arm in 0..2 {
                            for s in 0..=self.spec.responses() {
                                let p = q * row[arm] * self.pmfs[arm][s as usize];
                                if p > 0.0 {
                                    let mut next = t;
                                    next.counts[arm][s as usize] += 1;
                                    *after.entry(next).or_insert(0.0) += p;
                                }
                            }
                        }
                    }
                    stage = after;
                }
                for (t, q) in stage {
                    *enrolled.entry(t).or_insert(0.0) += q;
                }
            }
            // No delay: the batch has completed. Decide.
            let mut continuing: BTreeMap<Tally, f64> = BTreeMap::new();
            for (tally, probability) in enrolled {
                let posterior = self.posterior(&tally);
                let losses = posterior.expected_loss.clone();
                let cheapest = usize::from(losses[1] < losses[0]);
                if losses[cheapest] < self.spec.decisions().loss_threshold {
                    *outcomes.entry((look, "expected_loss", cheapest)).or_insert(0.0) +=
                        probability;
                } else if look + 1 == looks {
                    let selected = usize::from(posterior.mean[1] > posterior.mean[0]);
                    *outcomes.entry((look, "horizon", selected)).or_insert(0.0) += probability;
                } else {
                    continuing.insert(tally, probability);
                }
            }
            live = continuing;
        }
        outcomes
    }
}

#[test]
fn the_simulation_matches_an_exhaustive_enumeration_of_a_small_procedure() {
    let s = ParticipantSpec::new(ParticipantDeclaration {
        arms: 2,
        responses: Some(3),
        correlation: Some(0.3),
        correlation_prior: None,
        borrowing: Some(BorrowingPlan::uniform(2, BetaPrior::UNIFORM)),
        enrolment: Some(vec![2, 2, 2]),
        lag_looks: 0,
        min_exploration: Some(0.2),
        decisions: Some(DecisionRules {
            loss_threshold: 0.12,
            futility: None,
            participant_cap: None,
        }),
        anytime: Some(AnytimeSettings { alpha: 0.05, tuning_samples: 10.0 }),
    })
    .unwrap();
    let truth = [0.3, 0.7];
    let enumeration = Enumeration {
        spec: &s,
        pmfs: [beta_binomial_pmf(3, truth[0], 0.3), beta_binomial_pmf(3, truth[1], 0.3)],
    };
    let exact = enumeration.terminal_states();
    close(exact.values().sum::<f64>(), 1.0, 1e-9, "the enumeration is a distribution");
    assert!(exact.len() >= 3, "the procedure has a spread of terminal states: {exact:?}");

    let replications = 8_000_u32;
    let simulation =
        simulate_participant_procedure(&s, &truth, &request(replications, 2024), &never).unwrap();
    let mut observed: BTreeMap<(usize, &'static str, usize), f64> = BTreeMap::new();
    for r in &simulation.replicates {
        *observed.entry((r.look, r.reason.as_str(), r.decision)).or_insert(0.0) +=
            1.0 / f64::from(replications);
    }
    let keys: std::collections::BTreeSet<_> =
        exact.keys().chain(observed.keys()).copied().collect();
    for key in keys {
        let p = exact.get(&key).copied().unwrap_or(0.0);
        let q = observed.get(&key).copied().unwrap_or(0.0);
        let tolerance = 4.0 * (p * (1.0 - p) / f64::from(replications)).sqrt() + 1e-3;
        assert!(
            (p - q).abs() <= tolerance,
            "{key:?}: exact {p}, simulated {q} (tolerance {tolerance})"
        );
    }
}

// -- Declaration -----------------------------------------------------------------------------------

#[test]
fn every_missing_element_is_its_own_refusal() {
    let empty = ParticipantDeclaration {
        arms: 3,
        responses: None,
        correlation: None,
        correlation_prior: None,
        borrowing: None,
        enrolment: None,
        lag_looks: 0,
        min_exploration: None,
        decisions: None,
        anytime: None,
    };
    let refused = ParticipantSpec::new(empty).unwrap_err().gaps;
    assert_eq!(
        details(&refused),
        vec![
            "adaptive_inference.outcome_structure_missing",
            "adaptive_inference.within_unit_correlation_missing",
            "adaptive_inference.historical_borrowing_missing",
            "adaptive_inference.interim_analyses_missing",
            "adaptive_inference.policy_history_missing",
            "adaptive_inference.stopping_rule_missing",
            "adaptive_inference.anytime_settings_missing",
        ]
    );
    for gap in &refused {
        assert_eq!(gap.reason_code(), "adaptive_contract_incomplete");
        assert!(!gap.remedy().is_empty());
    }
}

#[test]
fn inconsistent_elements_are_refused_with_their_own_details() {
    let one = |gaps: Vec<ContractGap>| -> &'static str {
        assert_eq!(gaps.len(), 1, "{gaps:?}");
        gaps[0].detail()
    };
    assert_eq!(
        one(gaps(|d| {
            d.arms = 5;
            d.borrowing = Some(BorrowingPlan::uniform(5, BetaPrior::UNIFORM));
        })),
        "adaptive_inference.arms_invalid"
    );
    assert_eq!(one(gaps(|d| d.responses = Some(1))), "adaptive_inference.measurements_invalid");
    assert_eq!(one(gaps(|d| d.responses = Some(33))), "adaptive_inference.measurements_invalid");
    for rho in [1.0, -0.1, f64::NAN] {
        assert_eq!(
            one(gaps(|d| d.correlation = Some(rho))),
            "adaptive_inference.correlation_invalid"
        );
    }
    // The correlation is declared or estimated, never both, and a prior needs shapes of at least 1.
    assert_eq!(
        one(gaps(|d| d.correlation_prior = Some(CorrelationPrior { alpha: 1.0, beta: 1.0 }))),
        "adaptive_inference.correlation_invalid"
    );
    for prior in [(0.5, 1.0), (1.0, f64::NAN), (1.0, 1.0e5)] {
        assert_eq!(
            one(gaps(|d| {
                d.correlation = None;
                d.correlation_prior = Some(CorrelationPrior { alpha: prior.0, beta: prior.1 });
            })),
            "adaptive_inference.correlation_invalid"
        );
    }
    let bad_source = |discount: f64| {
        Some(BorrowingPlan {
            arms: vec![
                ArmPrior {
                    initial: BetaPrior::UNIFORM,
                    historical: vec![source("registry", 3, 10, discount)],
                },
                ArmPrior::without_history(BetaPrior::UNIFORM),
                ArmPrior::without_history(BetaPrior::UNIFORM),
            ],
        })
    };
    assert_eq!(
        one(gaps(|d| d.borrowing = bad_source(1.5))),
        "adaptive_inference.historical_prior_invalid"
    );
    assert_eq!(
        one(gaps(|d| d.borrowing = Some(BorrowingPlan::uniform(2, BetaPrior::UNIFORM)))),
        "adaptive_inference.historical_prior_invalid",
        "one prior per arm"
    );
    assert_eq!(
        one(gaps(
            |d| d.borrowing = Some(BorrowingPlan::uniform(3, BetaPrior { alpha: 0.5, beta: 1.0 }))
        )),
        "adaptive_inference.historical_prior_invalid"
    );
    assert_eq!(one(gaps(|d| d.lag_looks = 33)), "adaptive_inference.outcome_availability_invalid");
    for enrolment in [vec![40], vec![0, 40], vec![40, 0, 40]] {
        assert_eq!(
            one(gaps(|d| d.enrolment = Some(enrolment))),
            "adaptive_inference.looks_not_ordered"
        );
    }
    for floor in [0.0, 0.4, f64::NAN] {
        assert_eq!(
            one(gaps(|d| d.min_exploration = Some(floor))),
            "adaptive_inference.rule_mismatch"
        );
    }
    let rules =
        |loss_threshold: f64, futility: Option<FutilityRule>, participant_cap: Option<u32>| {
            Some(DecisionRules { loss_threshold, futility, participant_cap })
        };
    for decisions in [
        rules(0.0, None, None),
        rules(0.01, Some(FutilityRule { margin: 1.0, threshold: 0.1 }), None),
        rules(0.01, Some(FutilityRule { margin: 0.1, threshold: 0.5 }), None),
        rules(0.01, None, Some(0)),
        rules(0.01, None, Some(161)),
    ] {
        assert_eq!(one(gaps(|d| d.decisions = decisions)), "adaptive_inference.rule_mismatch");
    }
    for settings in [
        AnytimeSettings { alpha: 0.0, tuning_samples: 10.0 },
        AnytimeSettings { alpha: 1.0, tuning_samples: 10.0 },
        AnytimeSettings { alpha: 0.05, tuning_samples: 0.0 },
    ] {
        assert_eq!(
            one(gaps(|d| d.anytime = Some(settings))),
            "adaptive_inference.anytime_settings_invalid"
        );
    }
}

#[test]
fn a_replicate_does_not_depend_on_how_many_others_are_run() {
    let s = spec(|d| d.enrolment = Some(vec![30; 4]));
    let truth = [0.3, 0.4, 0.5];
    let short = simulate_participant_procedure(&s, &truth, &request(8, 99), &never).unwrap();
    let long = simulate_participant_procedure(&s, &truth, &request(20, 99), &never).unwrap();
    assert_eq!(short.replicates[..], long.replicates[..8]);
    let other_seed = simulate_participant_procedure(&s, &truth, &request(8, 100), &never).unwrap();
    assert_ne!(short.replicates, other_seed.replicates);
}

// -- An estimated correlation ----------------------------------------------------------------------

fn estimated(edit: impl FnOnce(&mut ParticipantDeclaration)) -> ParticipantSpec {
    spec(|d| {
        d.correlation = None;
        d.correlation_prior = Some(CorrelationPrior { alpha: 2.0, beta: 6.0 });
        edit(d);
    })
}

#[test]
fn an_estimated_correlation_is_a_different_frozen_declaration() {
    let declared = spec(|_| {});
    let learned = estimated(|_| {});
    assert_ne!(declared.identity(), learned.identity());
    assert_eq!(learned.correlation(), None);
    assert_eq!(learned.design_effect(), None);
    let prior = learned.correlation_prior().expect("an estimated correlation has a prior");
    close(prior.mean(), 0.25, 1e-15, "prior mean");
    let other = spec(|d| {
        d.correlation = None;
        d.correlation_prior = Some(CorrelationPrior { alpha: 2.0, beta: 5.0 });
    });
    assert_ne!(learned.identity(), other.identity(), "the correlation prior is declared");
    assert_eq!(learned.identity(), estimated(|_| {}).identity());
}

#[test]
fn a_procedure_that_estimates_the_correlation_reads_it_at_every_look() {
    let s = estimated(|d| d.enrolment = Some(vec![30; 3]));
    // The truth has a correlation the prior does not centre on.
    let history = simulate_participant_history(&s, &[0.3, 0.5, 0.6], Some(0.5), 11, &never)
        .expect("a history");
    let readings = s.read_history(&history, &never).expect("the history replays");
    assert!(readings.len() >= 2);
    let first = &readings[0].posterior;
    let last = &readings.last().unwrap().posterior;
    assert!(first.correlation_sd > 0.0 && last.correlation_sd > 0.0);
    assert!(
        last.correlation_sd < first.correlation_sd,
        "more participants say more about the correlation: {} then {}",
        first.correlation_sd,
        last.correlation_sd
    );
    assert!(last.correlation_mean > 0.25, "the data pull the estimate toward 0.5");
    // Declared, the correlation reads as itself with no uncertainty.
    let declared = spec(|d| d.enrolment = Some(vec![30; 3]));
    let declared_history = history_of(&declared, &[0.3, 0.5, 0.6], 11);
    let reading = &declared.read_history(&declared_history, &never).unwrap()[0].posterior;
    close(reading.correlation_mean, 0.2, 0.0, "declared correlation");
    close(reading.correlation_sd, 0.0, 0.0, "declared correlation sd");
    // A retained history does not replay under a different correlation model.
    let wrong = spec(|d| d.enrolment = Some(vec![30; 3]));
    let replayed = wrong.read_history(&history, &never);
    assert!(
        replayed.is_err()
            || replayed.unwrap().iter().zip(&readings).any(|(a, b)| a.posterior != b.posterior),
        "a different correlation model reads the history differently"
    );
}

#[test]
fn simulating_an_estimated_correlation_needs_the_true_correlation() {
    let s = estimated(|d| d.enrolment = Some(vec![15; 3]));
    assert!(matches!(
        simulate_participant_procedure(&s, &[0.3, 0.5, 0.6], &request(2, 1), &never),
        Err(Error::InvalidParameter(_))
    ));
    assert!(matches!(
        simulate_participant_history(&s, &[0.3, 0.5, 0.6], None, 1, &never),
        Err(Error::InvalidParameter(_))
    ));
    let with_truth =
        ParticipantSimulationRequest { replications: 3, seed: 1, true_correlation: Some(0.3) };
    let simulation =
        simulate_participant_procedure(&s, &[0.3, 0.5, 0.6], &with_truth, &never).unwrap();
    assert_eq!(simulation.replicates.len(), 3);
    let bad = ParticipantSimulationRequest { true_correlation: Some(1.0), ..with_truth };
    assert!(matches!(
        simulate_participant_procedure(&s, &[0.3, 0.5, 0.6], &bad, &never),
        Err(Error::InvalidParameter(_))
    ));
}

#[test]
fn the_simulation_with_an_estimated_correlation_matches_an_exhaustive_enumeration() {
    let s = ParticipantSpec::new(ParticipantDeclaration {
        arms: 2,
        responses: Some(3),
        correlation: None,
        correlation_prior: Some(CorrelationPrior { alpha: 2.0, beta: 4.0 }),
        borrowing: Some(BorrowingPlan::uniform(2, BetaPrior::UNIFORM)),
        enrolment: Some(vec![2, 2]),
        lag_looks: 0,
        min_exploration: Some(0.2),
        decisions: Some(DecisionRules {
            loss_threshold: 0.1,
            futility: None,
            participant_cap: None,
        }),
        anytime: Some(AnytimeSettings { alpha: 0.05, tuning_samples: 10.0 }),
    })
    .unwrap();
    let truth = [0.3, 0.7];
    let true_correlation = 0.4;
    let enumeration = Enumeration {
        spec: &s,
        pmfs: [
            beta_binomial_pmf(3, truth[0], true_correlation),
            beta_binomial_pmf(3, truth[1], true_correlation),
        ],
    };
    let exact = enumeration.terminal_states();
    close(exact.values().sum::<f64>(), 1.0, 1e-9, "the enumeration is a distribution");
    assert!(exact.len() >= 2, "{exact:?}");
    let replications = 3_000_u32;
    let request = ParticipantSimulationRequest {
        replications,
        seed: 77,
        true_correlation: Some(true_correlation),
    };
    let simulation = simulate_participant_procedure(&s, &truth, &request, &never).unwrap();
    let mut observed: BTreeMap<(usize, &'static str, usize), f64> = BTreeMap::new();
    for r in &simulation.replicates {
        *observed.entry((r.look, r.reason.as_str(), r.decision)).or_insert(0.0) +=
            1.0 / f64::from(replications);
    }
    let keys: std::collections::BTreeSet<_> =
        exact.keys().chain(observed.keys()).copied().collect();
    for key in keys {
        let p = exact.get(&key).copied().unwrap_or(0.0);
        let q = observed.get(&key).copied().unwrap_or(0.0);
        let tolerance = 4.0 * (p * (1.0 - p) / f64::from(replications)).sqrt() + 1e-3;
        assert!(
            (p - q).abs() <= tolerance,
            "{key:?}: exact {p}, simulated {q} (tolerance {tolerance})"
        );
    }
}

// -- Acceptance cases: the likelihood and the borrowing receipt ------------------------------------

/// Normalized weights of the dense midpoint grid over an arm mean for `prior` and the
/// beta-binomial likelihood of `histogram` (all zero: the prior alone).
fn dense_weights(m: u32, rho: f64, histogram: &[u32], prior: (f64, f64), grid: usize) -> Vec<f64> {
    let mut log_weight = Vec::with_capacity(grid);
    for g in 0..grid {
        let mu = (g as f64 + 0.5) / grid as f64;
        let pmf = gamma_form_pmf(m, mu, rho);
        let log_likelihood: f64 =
            histogram.iter().zip(&pmf).map(|(&count, p)| f64::from(count) * p.ln()).sum();
        log_weight
            .push(log_likelihood + (prior.0 - 1.0) * mu.ln() + (prior.1 - 1.0) * (1.0 - mu).ln());
    }
    let peak = log_weight.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = log_weight.iter().map(|w| (w - peak).exp()).collect();
    let total: f64 = weights.iter().sum();
    weights.into_iter().map(|w| w / total).collect()
}

/// `P(current > prior)` for independent draws from two grid distributions.
fn dense_exceedance(current: &[f64], prior: &[f64]) -> f64 {
    let (mut below, mut total) = (0.0, 0.0);
    for (c, p) in current.iter().zip(prior) {
        total += c * (below + 0.5 * p);
        below += p;
    }
    total
}

/// The success-count histogram of each arm's completed participants among the first `enrolled`
/// participants of a history without delay.
fn histograms(
    history: &ParticipantHistory,
    arms: usize,
    responses: u32,
    enrolled: usize,
) -> Vec<Vec<u32>> {
    let mut out = vec![vec![0_u32; responses as usize + 1]; arms];
    for (&arm, success) in history.arms.iter().zip(&history.successes).take(enrolled) {
        out[arm][success.expect("no delay: every participant has completed") as usize] += 1;
    }
    out
}

#[test]
fn adaptive_participant_aware_likelihood_matches_oracle() {
    // 600 participants, three arms, three responses each, correlation 0.3, in twelve batches
    // without delay, under a Beta(2.5, 4) initial prior.
    let prior = (2.5, 4.0);
    let s = spec(|d| {
        d.responses = Some(3);
        d.correlation = Some(0.3);
        d.enrolment = Some(vec![50; 12]);
        d.borrowing = Some(BorrowingPlan::uniform(3, BetaPrior { alpha: prior.0, beta: prior.1 }));
    });
    let history = simulate_participant_history(&s, &[0.3, 0.4, 0.5], None, 7_944_575, &never)
        .expect("a history");
    assert_eq!(history.arms.len(), 600);
    let readings = s.read_history(&history, &never).expect("the history replays");
    assert_eq!(readings.len(), 12);
    // The posterior at looks early, middle and last equals the dense quadrature of the
    // gamma-function beta-binomial likelihood times the prior, arm by arm.
    for look in [0, 5, 11] {
        let reading = &readings[look];
        let enrolled = 50 * (look + 1);
        assert_eq!(reading.completed.iter().sum::<u32>() as usize, enrolled);
        let by_arm = histograms(&history, 3, 3, enrolled);
        for arm in 0..3 {
            let (mean, sd) = exact_posterior(3, 0.3, &by_arm[arm], prior);
            let what = format!("look {look} arm {arm}");
            close(reading.posterior.mean[arm], mean, 2e-6, &format!("{what}: mean"));
            close(reading.posterior.sd[arm], sd, 2e-6, &format!("{what}: sd"));
        }
    }

    // At zero correlation the same update is the conjugate Beta of the pooled responses.
    let independent = spec(|d| {
        d.responses = Some(3);
        d.correlation = Some(0.0);
        d.enrolment = Some(vec![50; 12]);
        d.borrowing = Some(BorrowingPlan::uniform(3, BetaPrior { alpha: prior.0, beta: prior.1 }));
    });
    let history =
        simulate_participant_history(&independent, &[0.3, 0.4, 0.5], None, 7_944_575, &never)
            .expect("a history");
    let readings = independent.read_history(&history, &never).expect("replays");
    let last = readings.last().unwrap();
    let by_arm = histograms(&history, 3, 3, 600);
    for arm in 0..3 {
        let successes: f64 =
            by_arm[arm].iter().enumerate().map(|(s, &n)| s as f64 * f64::from(n)).sum();
        let trials = 3.0 * f64::from(by_arm[arm].iter().sum::<u32>());
        let (a, b) = (prior.0 + successes, prior.1 + trials - successes);
        close(last.posterior.mean[arm], a / (a + b), 1e-9, "conjugate mean");
        close(
            last.posterior.sd[arm],
            (a * b / ((a + b).powi(2) * (a + b + 1.0))).sqrt(),
            1e-9,
            "conjugate sd",
        );
    }
}

#[test]
#[allow(
    clippy::float_cmp,
    reason = "declared discounts and the borrowed size are copied, not recomputed"
)]
#[allow(clippy::too_many_lines, reason = "one receipt checked field by field for every arm")]
fn adaptive_historical_borrowing_receipt() {
    // Four arms, two historical sources each at a discount of one half, 1,200 participants.
    let truth = [0.3, 0.4, 0.5, 0.6];
    let plan = |shift: f64| BorrowingPlan {
        arms: truth
            .iter()
            .enumerate()
            .map(|(arm, &rate)| ArmPrior {
                initial: BetaPrior::UNIFORM,
                historical: vec![
                    source(
                        &format!("registry {arm}"),
                        ((rate + shift) * 200.0_f64).round() as u64,
                        200,
                        0.5,
                    ),
                    source(
                        &format!("pilot {arm}"),
                        ((rate + shift) * 80.0_f64).round() as u64,
                        80,
                        0.5,
                    ),
                ],
            })
            .collect(),
    };
    let run = |shift: f64| {
        let s = spec(|d| {
            d.arms = 4;
            d.responses = Some(3);
            d.correlation = Some(0.3);
            d.enrolment = Some(vec![100; 12]);
            d.borrowing = Some(plan(shift));
        });
        let history =
            simulate_participant_history(&s, &truth, None, 4_278_190, &never).expect("a history");
        (s, history)
    };

    let (s, history) = run(0.0);
    assert_eq!(history.arms.len(), 1_200);
    let receipt = s.borrowing_receipt(&history, &never).expect("a receipt");
    assert_eq!(receipt.len(), 4);
    let by_arm = histograms(&history, 4, 3, 1_200);
    let (mut agreement, mut shifted_agreement) = (Vec::new(), Vec::new());
    for (arm, entry) in receipt.iter().enumerate() {
        assert_eq!(entry.arm, arm);
        // Per-source discounts, and the effective historical sample size they are worth.
        assert_eq!(entry.sources.len(), 2);
        assert_eq!(entry.sources[0].source, format!("registry {arm}"));
        assert_eq!(entry.sources[1].source, format!("pilot {arm}"));
        assert!(entry.sources.iter().all(|source| source.discount == 0.5));
        assert_eq!((entry.sources[0].trials, entry.sources[1].trials), (200, 80));
        close(entry.effective_historical_sample_size, 0.5 * 280.0, 1e-12, "borrowed responses");
        close(entry.prior_effective_sample_size, 2.0 + 140.0, 1e-12, "prior effective size");
        assert_eq!(entry.effective_historical_sample_size, s.priors()[arm].borrowed_sample_size);
        assert_eq!(entry.current_participants, u64::from(by_arm[arm].iter().sum::<u32>()));
        // The prior-data conflict is the exact probability that the arm mean from the current
        // data alone exceeds the borrowed prior's, by a dense two-dimensional quadrature.
        let conflict = entry.conflict.expect("an arm that borrowed has a diagnostic");
        let grid = 100_000;
        let borrowed = &s.priors()[arm].prior;
        let current = dense_weights(3, 0.3, &by_arm[arm], (1.0, 1.0), grid);
        let historical =
            dense_weights(3, 0.3, &[0, 0, 0, 0], (borrowed.alpha, borrowed.beta), grid);
        let oracle = dense_exceedance(&current, &historical);
        close(conflict.current_exceeds_prior, oracle, 1e-5, &format!("arm {arm} exceedance"));
        close(
            conflict.agreement,
            2.0 * oracle.min(1.0 - oracle),
            2e-5,
            &format!("arm {arm} agreement"),
        );
        assert!((0.0..=1.0).contains(&conflict.agreement));
        agreement.push(conflict.agreement);
    }
    let means: Vec<f64> = by_arm
        .iter()
        .map(|h| {
            let n = f64::from(h.iter().sum::<u32>());
            h.iter().enumerate().map(|(s, &c)| s as f64 * f64::from(c)).sum::<f64>() / (3.0 * n)
        })
        .collect();
    assert!(
        agreement.iter().all(|&a| a > 1e-3),
        "history centred on the truth: agreement {agreement:?}, per-response means {means:?}, histograms {by_arm:?}"
    );

    // History centred 0.2 above the truth conflicts with the data of every arm: the current
    // data fall below the borrowed prior. The adaptive allocation gives the weakest arms the
    // fewest participants, so their evidence of conflict is the weakest; each arm's agreement
    // still falls far below its agreement with a centred history.
    let (shifted, shifted_history) = run(0.2);
    for entry in shifted.borrowing_receipt(&shifted_history, &never).unwrap() {
        shifted_agreement.push(entry.conflict.unwrap().agreement);
        assert!(entry.conflict.unwrap().current_exceeds_prior < 1e-2);
    }
    assert!(
        shifted_agreement.iter().zip(&agreement).all(|(&s, &c)| s < 1e-2 && s < 0.1 * c),
        "shifted {shifted_agreement:?}, centred {agreement:?}"
    );

    // An arm that borrowed nothing has no diagnostic, and so has an arm nobody completed.
    let mixed = BorrowingPlan {
        arms: vec![
            ArmPrior::without_history(BetaPrior::UNIFORM),
            ArmPrior {
                initial: BetaPrior::UNIFORM,
                historical: vec![source("registry", 40, 100, 0.5)],
            },
        ],
    };
    let two = spec(|d| {
        d.arms = 2;
        d.responses = Some(3);
        d.borrowing = Some(mixed);
        d.enrolment = Some(vec![40, 40]);
    });
    let two_history =
        simulate_participant_history(&two, &[0.4, 0.4], None, 4_278_190, &never).unwrap();
    let receipt = two.borrowing_receipt(&two_history, &never).unwrap();
    assert!(receipt[0].sources.is_empty() && receipt[0].conflict.is_none());
    close(receipt[0].effective_historical_sample_size, 0.0, 0.0, "nothing borrowed");
    assert!(receipt[1].conflict.is_some());
    let mut nobody = two_history;
    nobody.successes.iter_mut().for_each(|s| *s = None);
    let receipt = two.borrowing_receipt(&nobody, &never).unwrap();
    assert!(
        receipt.iter().all(|entry| entry.conflict.is_none() && entry.current_participants == 0)
    );
}
