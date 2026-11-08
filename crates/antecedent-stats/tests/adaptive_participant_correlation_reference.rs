//! The participant-level posterior with an estimated intra-participant correlation, read at every
//! look of a simulated history, against a joint numerical integration written here from the model.
//!
//! The model: the participants of an arm have success counts that are beta-binomial with arm mean
//! `p` and correlation `rho` (shapes `p kappa` and `(1 - p) kappa`, `kappa = (1 - rho) / rho`); the
//! arms are independent given `rho`; each arm mean has a Beta prior and `rho` has a
//! `Beta(2, 6)` prior shared by the arms. Given `rho` the marginal likelihood of an arm is the
//! integral of its prior times its likelihood over `p`, so the weight of `rho` is its prior times
//! the product of the arms' marginal likelihoods, and the arm moments are the weight-average of the
//! arm moments at each `rho`. Both integrals are midpoint sums here (a 400-point grid over `rho`
//! and a dense grid over `p`), with the mass function written as a product of rising factors, not
//! through the library's posterior code.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the oracle indexes per-arm and per-count columns and counts small integers"
)]

use antecedent_stats::adaptive_inference::{
    AnytimeSettings, ArmPrior, BetaPrior, BorrowingPlan, CorrelationPrior, DecisionRules,
    FutilityRule, HistoricalPrior, ParticipantDeclaration, ParticipantHistory, ParticipantSpec,
    simulate_participant_history,
};

const RESPONSES: usize = 3;
const ARMS: usize = 3;
const RHO_NODES: usize = 400;
const P_NODES: usize = 8_001;
const LAG: usize = 1;
const TRUTH: [f64; ARMS] = [0.35, 0.45, 0.4];
const TRUE_CORRELATION: f64 = 0.45;
const TOLERANCE: f64 = 2e-5;

fn never() -> bool {
    false
}

fn source(label: &str, successes: u64, trials: u64, discount: f64) -> HistoricalPrior {
    HistoricalPrior { source: label.to_owned(), successes, trials, discount }
}

fn declaration() -> ParticipantDeclaration {
    ParticipantDeclaration {
        arms: ARMS,
        responses: Some(RESPONSES as u32),
        correlation: None,
        correlation_prior: Some(CorrelationPrior { alpha: 2.0, beta: 6.0 }),
        borrowing: Some(BorrowingPlan {
            arms: vec![
                ArmPrior {
                    initial: BetaPrior::UNIFORM,
                    historical: vec![
                        source("control registry", 40, 100, 0.5),
                        source("earlier pilot", 9, 30, 1.0),
                    ],
                },
                ArmPrior::without_history(BetaPrior::UNIFORM),
                ArmPrior::without_history(BetaPrior::UNIFORM),
            ],
        }),
        enrolment: Some(vec![40; 5]),
        lag_looks: LAG,
        min_exploration: Some(0.15),
        decisions: Some(DecisionRules {
            loss_threshold: 1e-12,
            futility: Some(FutilityRule { margin: 0.1, threshold: 0.02 }),
            participant_cap: None,
        }),
        anytime: Some(AnytimeSettings { alpha: 0.06, tuning_samples: 30.0 }),
    }
}

/// The success-count histogram of each arm among the participants whose follow-up completed by
/// `look`, in enrolment order.
fn histograms(history: &ParticipantHistory, look: usize) -> Vec<Vec<u32>> {
    let end: usize =
        history.batches[..(look + 1).saturating_sub(LAG)].iter().map(|&n| n as usize).sum();
    let mut counts = vec![vec![0_u32; RESPONSES + 1]; ARMS];
    for i in 0..end {
        let s = history.successes[i].expect("a completed participant has a count");
        counts[history.arms[i]][s as usize] += 1;
    }
    counts
}

/// `ln` of the mass of `s` successes among three responses at arm mean `p` and correlation `rho`,
/// for `s = 0..=3`: `C(3, s) B(s + a, 3 - s + b) / B(a, b)` with `a = p kappa`, `b = (1 - p)
/// kappa`, which is `C(3, s)` times `a (a + 1) ..` over `s` factors times `b (b + 1) ..` over
/// `3 - s` factors, over `kappa (kappa + 1) ..` over three factors.
fn log_pmf(p: f64, rho: f64) -> [f64; RESPONSES + 1] {
    let kappa = (1.0 - rho) / rho;
    let (a, b) = (p * kappa, (1.0 - p) * kappa);
    let rising = |x: f64, n: usize| -> f64 { (0..n).map(|j| (x + j as f64).ln()).sum() };
    let denominator = rising(kappa, RESPONSES);
    let choose = [1.0_f64, 3.0, 3.0, 1.0];
    let mut out = [0.0; RESPONSES + 1];
    for s in 0..=RESPONSES {
        out[s] = choose[s].ln() + rising(a, s) + rising(b, RESPONSES - s) - denominator;
    }
    out
}

/// The log density of the `Beta(2, 6)` prior of the correlation up to its constant,
/// `rho (1 - rho)^5`; the constant cancels in the normalized weights.
fn log_correlation_prior(rho: f64) -> f64 {
    (2.0_f64 - 1.0) * rho.ln() + (6.0_f64 - 1.0) * (-rho).ln_1p()
}

struct Joint {
    /// Posterior mean and standard deviation of each arm mean.
    mean: Vec<f64>,
    sd: Vec<f64>,
    correlation_mean: f64,
    correlation_sd: f64,
}

#[test]
#[allow(clippy::too_many_lines, reason = "the oracle and its checks at every look stay together")]
fn an_estimated_correlation_is_integrated_jointly_with_the_arm_means() {
    let spec = ParticipantSpec::new(declaration()).expect("a valid declaration");
    // The control prior borrows two sources from a uniform start: alpha = 1 + 0.5 * 40 + 9 and
    // beta = 1 + 0.5 * 60 + 21; the other arms keep the uniform prior.
    let priors = [(30.0_f64, 52.0_f64), (1.0, 1.0), (1.0, 1.0)];
    for (arm, &(alpha, beta)) in priors.iter().enumerate() {
        assert!((spec.priors()[arm].prior.alpha - alpha).abs() < 1e-12);
        assert!((spec.priors()[arm].prior.beta - beta).abs() < 1e-12);
    }

    let history = simulate_participant_history(&spec, &TRUTH, Some(TRUE_CORRELATION), 12, &never)
        .expect("a history");
    let readings = spec.read_history(&history, &never).expect("the history replays");
    assert_eq!(readings.len(), history.batches.len());
    let counts: Vec<Vec<Vec<u32>>> =
        (0..readings.len()).map(|look| histograms(&history, look)).collect();
    for (look, reading) in readings.iter().enumerate() {
        for arm in 0..ARMS {
            assert_eq!(reading.completed[arm], counts[look][arm].iter().sum::<u32>());
        }
    }

    let p_grid: Vec<f64> = (0..P_NODES).map(|g| (g as f64 + 0.5) / P_NODES as f64).collect();
    let prior_log: Vec<Vec<f64>> = priors
        .iter()
        .map(|&(alpha, beta)| {
            p_grid.iter().map(|&p| (alpha - 1.0) * p.ln() + (beta - 1.0) * (-p).ln_1p()).collect()
        })
        .collect();

    // Per look: the log weight of each rho and, at each rho, the first two moments of every arm.
    let looks = readings.len();
    let mut log_weight = vec![Vec::with_capacity(RHO_NODES); looks];
    let mut moments = vec![Vec::with_capacity(RHO_NODES); looks];
    let rhos: Vec<f64> = (0..RHO_NODES).map(|i| (i as f64 + 0.5) / RHO_NODES as f64).collect();
    for &rho in &rhos {
        let table: Vec<[f64; RESPONSES + 1]> = p_grid.iter().map(|&p| log_pmf(p, rho)).collect();
        for look in 0..looks {
            let mut total = log_correlation_prior(rho);
            let mut per_arm = Vec::with_capacity(ARMS);
            for arm in 0..ARMS {
                let log_density: Vec<f64> = (0..P_NODES)
                    .map(|g| {
                        let mut value = prior_log[arm][g];
                        for s in 0..=RESPONSES {
                            let count = counts[look][arm][s];
                            if count > 0 {
                                value += f64::from(count) * table[g][s];
                            }
                        }
                        value
                    })
                    .collect();
                let peak = log_density.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<f64> = log_density.iter().map(|d| (d - peak).exp()).collect();
                let sum: f64 = weights.iter().sum();
                // The marginal likelihood at this rho is the midpoint sum of the density.
                total += peak + (sum / P_NODES as f64).ln();
                let mean = weights.iter().zip(&p_grid).map(|(w, p)| w * p).sum::<f64>() / sum;
                let second = weights.iter().zip(&p_grid).map(|(w, p)| w * p * p).sum::<f64>() / sum;
                per_arm.push((mean, second));
            }
            log_weight[look].push(total);
            moments[look].push(per_arm);
        }
    }

    for (look, reading) in readings.iter().enumerate() {
        let top = log_weight[look].iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let q: Vec<f64> = log_weight[look].iter().map(|w| (w - top).exp()).collect();
        let norm: f64 = q.iter().sum();
        let q: Vec<f64> = q.iter().map(|w| w / norm).collect();

        let mut joint = Joint {
            mean: Vec::new(),
            sd: Vec::new(),
            correlation_mean: q.iter().zip(&rhos).map(|(w, r)| w * r).sum(),
            correlation_sd: 0.0,
        };
        joint.correlation_sd = q
            .iter()
            .zip(&rhos)
            .map(|(w, r)| w * (r - joint.correlation_mean).powi(2))
            .sum::<f64>()
            .sqrt();
        for arm in 0..ARMS {
            let mean: f64 = (0..RHO_NODES).map(|i| q[i] * moments[look][i][arm].0).sum();
            let second: f64 = (0..RHO_NODES).map(|i| q[i] * moments[look][i][arm].1).sum();
            joint.mean.push(mean);
            joint.sd.push((second - mean * mean).sqrt());
        }

        let claim = &reading.posterior;
        for arm in 0..ARMS {
            assert!(
                (claim.mean[arm] - joint.mean[arm]).abs() <= TOLERANCE,
                "look {look} arm {arm} mean: {} vs {}",
                claim.mean[arm],
                joint.mean[arm]
            );
            assert!(
                (claim.sd[arm] - joint.sd[arm]).abs() <= TOLERANCE,
                "look {look} arm {arm} sd: {} vs {}",
                claim.sd[arm],
                joint.sd[arm]
            );
        }
        assert!(
            (claim.correlation_mean - joint.correlation_mean).abs() <= TOLERANCE,
            "look {look} correlation mean: {} vs {}",
            claim.correlation_mean,
            joint.correlation_mean
        );
        assert!(
            (claim.correlation_sd - joint.correlation_sd).abs() <= TOLERANCE,
            "look {look} correlation sd: {} vs {}",
            claim.correlation_sd,
            joint.correlation_sd
        );
        let total: f64 = claim.probability_best.iter().sum();
        assert!((total - 1.0).abs() <= 1e-8, "look {look}: probabilities of best sum to {total}");
    }

    // More completed participants say more about the correlation.
    let first = readings[1].posterior.correlation_sd;
    let last = readings.last().unwrap().posterior.correlation_sd;
    assert!(last < first, "the correlation sd fell from {first} to {last}");
}
