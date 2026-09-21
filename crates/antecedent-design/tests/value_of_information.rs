//! Preposterior value of information: EVPI, EVSI and the `ReduceDecisionRegret` ranking.
//!
//! Closed forms used as oracles (Raiffa & Schlaifer 1961, ch. 4–5):
//!
//! * EVPI on equally weighted draws: `mean_k max_a U(a,θ_k) − max_a mean_k U(a,θ_k)`.
//! * Conjugate normal, two actions `U_i = α_i + β_i θ`, prior `θ ~ N(μ₀, τ²)`, `n`
//!   observations with noise variance `σ²`: the posterior mean is preposteriorly
//!   `N(μ₀, s_n²)` with `s_n² = τ⁴ n / (nτ² + σ²)`, the expected utility difference at
//!   posterior mean `m` is `Δβ (m − μ_b)` with break-even `μ_b = (α₁ − α₂)/(β₂ − β₁)`,
//!   and `EVSI(n) = |Δβ| s_n G(|μ_b − μ₀| / s_n)` with the unit normal linear-loss
//!   integral `G(u) = φ(u) − u (1 − Φ(u))`. `EVPI` is the same with `s = τ`.

#![allow(clippy::float_cmp)]

use std::sync::Arc;

use antecedent_core::{CausalRng, ExecutionContext};
use antecedent_design::{
    AffineUtility, BinomialSignal, CandidateDesign, DecisionConstraint, DecisionPrior,
    DecisionProblem, DecisionProblemId, DecisionRegistry, DecisionSignal, DesignCost, DesignError,
    DesignEvaluationContext, DesignObjective, DesignRankConfig, DesignRanker, GaussianMeanSignal,
    PreposteriorAnalysis, SamplingPlan, ScoreEvaluation, Utility, evaluate_decision,
};
use antecedent_kernels::{norm_pdf, norm_sf};
use antecedent_prob::{GraphIdentFlag, WeightedGraphSamples};

/// `U(a, θ) = scale · a · θ + shift`.
struct Linear {
    shift: f64,
    scale: f64,
}

impl Utility<f64, f64> for Linear {
    fn evaluate_batch(&self, a: &[f64], o: &[f64], out: &mut [f64]) -> Result<(), DesignError> {
        for (i, ai) in a.iter().enumerate() {
            for (j, oj) in o.iter().enumerate() {
                out[i * o.len() + j] = self.scale * ai * oj + self.shift;
            }
        }
        Ok(())
    }
}

/// `U(a, θ) = scale · (a · (θ − 0.45)) + shift`: abstain (a = 0) or bet (a = 1).
struct Bet {
    shift: f64,
    scale: f64,
}

impl Utility<f64, f64> for Bet {
    fn evaluate_batch(&self, a: &[f64], o: &[f64], out: &mut [f64]) -> Result<(), DesignError> {
        for (i, ai) in a.iter().enumerate() {
            for (j, oj) in o.iter().enumerate() {
                out[i * o.len() + j] = self.scale * ai * (oj - 0.45) + self.shift;
            }
        }
        Ok(())
    }
}

struct Never;

impl DecisionConstraint<f64, f64> for Never {
    fn name(&self) -> &'static str {
        "never"
    }
    fn satisfaction_batch(&self, actions: &[f64], _outcomes: &[f64], out: &mut [f64]) {
        out[..actions.len()].fill(0.0);
    }
}

fn unit_g(u: f64) -> f64 {
    norm_pdf(u) - u * norm_sf(u)
}

fn coin_problem(shift: f64, scale: f64) -> DecisionProblem<f64, f64> {
    DecisionProblem::new(vec![0.0, 1.0], Arc::new(Bet { shift, scale }), vec![])
}

fn coin_prior() -> DecisionPrior<f64> {
    DecisionPrior::Draws(vec![0.2, 0.5, 0.8, 0.35])
}

// -- reviewer cases -----------------------------------------------------------------

/// Outcomes [-1, 3], U = a·θ, a ∈ {1, 2}: EVPI = (max(−1,−2) + max(3,6))/2 − 2 = 0.5.
#[test]
fn unconstrained_expected_regret_is_evpi() {
    let problem =
        DecisionProblem::new(vec![1.0, 2.0], Arc::new(Linear { shift: 0.0, scale: 1.0 }), vec![]);
    let evaluation = evaluate_decision(&problem, &[-1.0, 3.0]).expect("decision");
    assert!((evaluation.posterior_regret - 0.5).abs() < 1e-15, "{evaluation:?}");
    let prior = DecisionPrior::Draws(vec![-1.0, 3.0]);
    let signal = GaussianMeanSignal::new(1.0).unwrap();
    let analysis = PreposteriorAnalysis::new(&problem, &prior, &signal).expect("analysis");
    assert!((analysis.expected_value_of_perfect_information() - 0.5).abs() < 1e-15);
    assert_eq!(analysis.bayes_action(), 1);
    assert!((analysis.prior_expected_utility() - 2.0).abs() < 1e-15);
}

/// Every action inadmissible: a typed refusal at utility shifts 0 and +100 (the old
/// score reported regret 2 and 102, moving with the utility's origin).
#[test]
fn all_inadmissible_is_refused_at_every_utility_origin() {
    for shift in [0.0, 100.0] {
        let problem = DecisionProblem::new(
            vec![1.0, 2.0],
            Arc::new(Linear { shift, scale: 1.0 }),
            vec![Arc::new(Never) as Arc<dyn DecisionConstraint<f64, f64>>],
        );
        let err = evaluate_decision(&problem, &[-1.0, 3.0]).unwrap_err();
        assert!(matches!(err, DesignError::NoAdmissibleAction(_)), "{err:?}");
        let prior = DecisionPrior::Draws(vec![-1.0, 3.0]);
        let signal = GaussianMeanSignal::new(1.0).unwrap();
        let err = PreposteriorAnalysis::new(&problem, &prior, &signal).err().expect("refused");
        assert!(matches!(err, DesignError::NoAdmissibleAction(_)), "{err:?}");
    }
}

// -- invariances --------------------------------------------------------------------

#[test]
fn evsi_is_shift_invariant_and_scale_equivariant() {
    let prior = coin_prior();
    let binomial = BinomialSignal;
    let gaussian = GaussianMeanSignal::new(0.09).unwrap();
    let base_problem = coin_problem(0.0, 1.0);
    let base_bin = PreposteriorAnalysis::new(&base_problem, &prior, &binomial).unwrap();
    let base_gau = PreposteriorAnalysis::new(&base_problem, &prior, &gaussian).unwrap();
    for (shift, scale) in [(100.0, 1.0), (-1.0e3, 1.0), (0.0, 3.0), (42.0, 0.25)] {
        let problem = coin_problem(shift, scale);
        let bin = PreposteriorAnalysis::new(&problem, &prior, &binomial).unwrap();
        let gau = PreposteriorAnalysis::new(&problem, &prior, &gaussian).unwrap();
        let tol = 1e-9 * scale.max(1.0);
        assert!(
            (bin.expected_value_of_perfect_information()
                - scale * base_bin.expected_value_of_perfect_information())
            .abs()
                < tol
        );
        for n in [1, 3, 12, 50] {
            let exact = bin.exact_evsi(n).unwrap().unwrap();
            let base = base_bin.exact_evsi(n).unwrap().unwrap();
            assert!((exact - scale * base).abs() < tol, "n={n} {shift} {scale}: {exact} {base}");
            // Same seed → same simulated state and statistic → same draw, up to scale.
            let seed = 77 + n;
            let draw = gau.sample_evsi(n, &mut CausalRng::from_seed(seed)).unwrap();
            let base_draw = base_gau.sample_evsi(n, &mut CausalRng::from_seed(seed)).unwrap();
            assert!((draw - scale * base_draw).abs() < tol, "n={n}: {draw} {base_draw}");
        }
    }
}

// -- bounds and limits --------------------------------------------------------------

#[test]
fn finite_signal_evsi_is_monotone_below_evpi_and_converges_to_it() {
    let problem = coin_problem(0.0, 1.0);
    let prior = coin_prior();
    let analysis = PreposteriorAnalysis::new(&problem, &prior, &BinomialSignal).unwrap();
    let evpi = analysis.expected_value_of_perfect_information();
    // Draws {0.2, 0.5, 0.8, 0.35}; bet pays θ − 0.45: prior EU 0.0125 (bet),
    // perfect information pays mean(0, 0.05, 0.35, 0) = 0.1, EVPI = 0.0875.
    assert!((evpi - 0.0875).abs() < 1e-15, "{evpi}");
    assert_eq!(analysis.evaluation(5), ScoreEvaluation::Exact);
    let mut previous = 0.0;
    for n in [1, 2, 4, 8, 16, 32, 64, 128, 256] {
        let sample_value = analysis.exact_evsi(n).unwrap().unwrap();
        assert!(sample_value >= previous - 1e-15, "n={n}: {sample_value} < {previous}");
        assert!(sample_value < evpi, "n={n}: {sample_value} >= {evpi}");
        previous = sample_value;
    }
    let far = analysis.exact_evsi(4_000).unwrap().unwrap();
    assert!((evpi - far).abs() < 1e-9, "{far} vs {evpi}");
    assert_eq!(analysis.exact_evsi(0).unwrap(), Some(0.0));
}

/// A single Bernoulli flip enumerated by hand: y ∈ {0, 1}.
#[test]
fn finite_signal_evsi_matches_hand_enumeration() {
    let problem = coin_problem(0.0, 1.0);
    let draws = [0.2, 0.5, 0.8, 0.35];
    let prior = DecisionPrior::Draws(draws.to_vec());
    let analysis = PreposteriorAnalysis::new(&problem, &prior, &BinomialSignal).unwrap();
    // E_y[max(0, E[θ − 0.45 | y])] − max(0, E[θ − 0.45]).
    let prior_value = (draws.iter().sum::<f64>() / 4.0 - 0.45).max(0.0);
    let heads: f64 = draws.iter().map(|t| t / 4.0).sum();
    let bet_given_heads: f64 = draws.iter().map(|t| t * (t - 0.45) / 4.0).sum::<f64>() / heads;
    let bet_given_tails: f64 =
        draws.iter().map(|t| (1.0 - t) * (t - 0.45) / 4.0).sum::<f64>() / (1.0 - heads);
    let expected =
        heads * bet_given_heads.max(0.0) + (1.0 - heads) * bet_given_tails.max(0.0) - prior_value;
    let got = analysis.exact_evsi(1).unwrap().unwrap();
    assert!(expected > 0.0);
    assert!((got - expected).abs() < 1e-15, "{got} vs {expected}");
}

/// Monte Carlo EVSI on draws with a Gaussian signal is unbiased: compare its mean with
/// a numerical-quadrature reference `∫ max_a Σ_k p_k gap_ak N(y; θ_k, σ²/n) dy`.
#[test]
fn monte_carlo_evsi_matches_quadrature_reference() {
    let problem = coin_problem(0.0, 1.0);
    let draws = [0.2, 0.5, 0.8, 0.35];
    let prior = DecisionPrior::Draws(draws.to_vec());
    let noise_variance = 0.2;
    let signal = GaussianMeanSignal::new(noise_variance).unwrap();
    let analysis = PreposteriorAnalysis::new(&problem, &prior, &signal).unwrap();
    assert_eq!(analysis.evaluation(3), ScoreEvaluation::MonteCarlo);
    assert_eq!(analysis.exact_evsi(3).unwrap(), None);
    let bayes_mean = draws.iter().sum::<f64>() / 4.0 - 0.45;
    assert!(bayes_mean > 0.0, "prior Bayes action is the bet");
    for n in [1_u64, 3, 10] {
        let sd = (noise_variance / n as f64).sqrt();
        // Relative to the bet, abstaining gains 0.45 − θ_k.
        let (lo, hi, steps) = (-3.0, 4.0, 400_000);
        let h = (hi - lo) / f64::from(steps);
        let mut reference = 0.0;
        for i in 0..=steps {
            let y = lo + h * f64::from(i);
            let gain: f64 =
                draws.iter().map(|t| 0.25 * (0.45 - t) * norm_pdf((y - t) / sd) / sd).sum();
            let weight = if i == 0 || i == steps { 0.5 } else { 1.0 };
            reference += weight * gain.max(0.0) * h;
        }
        let mut rng = CausalRng::from_seed(20_260_916 + n);
        let reps = 200_000;
        let (mut sum, mut sumsq) = (0.0, 0.0);
        for _ in 0..reps {
            let x = analysis.sample_evsi(n, &mut rng).unwrap();
            assert!(x >= 0.0);
            sum += x;
            sumsq += x * x;
        }
        let mean = sum / f64::from(reps);
        let se = ((sumsq / f64::from(reps) - mean * mean) / f64::from(reps)).sqrt();
        assert!(
            (mean - reference).abs() < 4.0 * se,
            "n={n}: MC {mean} ± {se} vs quadrature {reference}"
        );
        assert!(reference < analysis.expected_value_of_perfect_information());
    }
}

// -- conjugate normal ---------------------------------------------------------------

fn normal_problem() -> DecisionProblem<usize, f64> {
    let utility = AffineUtility::new(vec![1.0, -0.5], vec![0.2, 1.1]).unwrap();
    DecisionProblem::new(vec![0, 1], Arc::new(utility), vec![])
}

const MU0: f64 = 1.0;
const TAU2: f64 = 2.0;
const SIGMA2: f64 = 3.0;

fn closed_form_two_action(s: f64) -> f64 {
    let (alpha, beta) = ([1.0_f64, -0.5], [0.2_f64, 1.1]);
    let delta_beta = beta[1] - beta[0];
    let breakeven = (alpha[0] - alpha[1]) / delta_beta;
    delta_beta.abs() * s * unit_g((breakeven - MU0).abs() / s)
}

#[test]
fn conjugate_normal_evsi_matches_closed_form_and_is_monotone() {
    let problem = normal_problem();
    let prior = DecisionPrior::Normal { mean: MU0, variance: TAU2 };
    let signal = GaussianMeanSignal::new(SIGMA2).unwrap();
    let analysis = PreposteriorAnalysis::new(&problem, &prior, &signal).unwrap();
    // Prior EU: action 0 → 1.2, action 1 → 0.6.
    assert_eq!(analysis.bayes_action(), 0);
    let evpi = analysis.expected_value_of_perfect_information();
    let evpi_closed = closed_form_two_action(TAU2.sqrt());
    assert!((evpi - evpi_closed).abs() < 1e-14, "{evpi} vs {evpi_closed}");
    let mut previous = 0.0;
    for k in 0..=14 {
        let n = 1_u64 << k;
        assert_eq!(analysis.evaluation(n), ScoreEvaluation::Exact);
        let nf = n as f64;
        let s = (TAU2 * TAU2 * nf / (nf * TAU2 + SIGMA2)).sqrt();
        let sample_value = analysis.exact_evsi(n).unwrap().unwrap();
        let closed = closed_form_two_action(s);
        assert!((sample_value - closed).abs() < 1e-14, "n={n}: {sample_value} vs {closed}");
        assert!(sample_value > previous, "n={n}: not increasing ({sample_value} <= {previous})");
        assert!(sample_value < evpi, "n={n}: EVSI {sample_value} >= EVPI {evpi}");
        previous = sample_value;
    }
    let limit = analysis.exact_evsi(1_000_000_000_000).unwrap().unwrap();
    assert!((evpi - limit).abs() < 1e-10, "{limit} vs {evpi}");
    assert_eq!(analysis.exact_evsi(0).unwrap(), Some(0.0));
}

/// Three affine actions: the envelope sum matches direct quadrature of
/// `E[max_a (α_a + β_a (μ₀ + sZ))] − max_a (α_a + β_a μ₀)`.
#[test]
fn conjugate_normal_three_actions_matches_quadrature() {
    let (alpha, beta) = (vec![0.0, 0.8, -1.5], vec![0.0, -0.6, 1.4]);
    let utility = AffineUtility::new(alpha.clone(), beta.clone()).unwrap();
    let problem = DecisionProblem::new(vec![0, 1, 2], Arc::new(utility), vec![]);
    let prior = DecisionPrior::Normal { mean: 0.3, variance: 1.7 };
    let signal = GaussianMeanSignal::new(0.9).unwrap();
    let analysis = PreposteriorAnalysis::new(&problem, &prior, &signal).unwrap();
    let prior_best = (0..3).map(|a| alpha[a] + beta[a] * 0.3).fold(f64::NEG_INFINITY, f64::max);
    for n in [1_u64, 5, 40] {
        let nf = n as f64;
        let s = (1.7 * 1.7 * nf / (nf * 1.7 + 0.9)).sqrt();
        let (lo, hi, steps) = (-12.0, 12.0, 240_000);
        let h = (hi - lo) / f64::from(steps);
        let mut reference = 0.0;
        for i in 0..=steps {
            let z = lo + h * f64::from(i);
            let m = 0.3 + s * z;
            let best = (0..3).map(|a| alpha[a] + beta[a] * m).fold(f64::NEG_INFINITY, f64::max);
            let weight = if i == 0 || i == steps { 0.5 } else { 1.0 };
            reference += weight * best * norm_pdf(z) * h;
        }
        reference -= prior_best;
        let evsi = analysis.exact_evsi(n).unwrap().unwrap();
        assert!((evsi - reference).abs() < 1e-8, "n={n}: {evsi} vs {reference}");
    }
}

#[test]
fn conjugate_normal_refuses_outside_its_model() {
    let prior = DecisionPrior::Normal { mean: 0.0, variance: 1.0 };
    let gaussian = GaussianMeanSignal::new(1.0).unwrap();
    // A utility without declared affine coefficients.
    let opaque =
        DecisionProblem::new(vec![1.0, 2.0], Arc::new(Linear { shift: 0.0, scale: 1.0 }), vec![]);
    assert!(matches!(
        PreposteriorAnalysis::new(&opaque, &prior, &gaussian).err(),
        Some(DesignError::Config(_))
    ));
    // A non-Gaussian signal.
    assert!(matches!(
        PreposteriorAnalysis::new(
            &normal_problem(),
            &prior,
            &BinomialSignal as &dyn DecisionSignal<f64>
        )
        .err(),
        Some(DesignError::Config(_))
    ));
    // Constraints need draws of the state.
    let constrained = DecisionProblem::new(
        vec![1.0, 2.0],
        Arc::new(Linear { shift: 0.0, scale: 1.0 }),
        vec![Arc::new(Never) as Arc<dyn DecisionConstraint<f64, f64>>],
    );
    assert!(matches!(
        PreposteriorAnalysis::new(&constrained, &prior, &gaussian).err(),
        Some(DesignError::Config(_))
    ));
}

// -- ranking ------------------------------------------------------------------------

fn sampling(n: u64) -> CandidateDesign {
    CandidateDesign::IncreaseSamplingRate(SamplingPlan {
        additional_samples: n,
        cost: DesignCost::zero(),
        tag: n,
    })
}

#[test]
fn ranking_scores_are_the_conjugate_normal_evsi() {
    let graphs =
        WeightedGraphSamples::new(vec![1.0], vec![GraphIdentFlag::Identified], vec![1]).unwrap();
    let registry = DecisionRegistry {
        problems: vec![Some(normal_problem())],
        prior: DecisionPrior::Normal { mean: MU0, variance: TAU2 },
        signal: Arc::new(GaussianMeanSignal::new(SIGMA2).unwrap()),
    };
    let eval = DesignEvaluationContext {
        graphs: &graphs,
        effect_width: None,
        model_loglik: None,
        decisions: Some(&registry),
        query_id_unlock: None,
        env_id_unlock: None,
        identified_under_intervention: None,
        graph_features: None,
    };
    let candidates: Vec<CandidateDesign> = [4_u64, 64, 1].into_iter().map(sampling).collect();
    let ranking = DesignRanker::new()
        .with_config(DesignRankConfig {
            min_batches: 2,
            max_batches: 2,
            batch_size: 2,
            rank_uncertainty_threshold: 0.0,
        })
        .rank(
            &DesignObjective::ReduceDecisionRegret { decision: DecisionProblemId::from_raw(0) },
            &candidates,
            &eval,
            &ExecutionContext::for_tests(3),
        )
        .expect("rank");
    let order: Vec<usize> = ranking.ranked.iter().map(|r| r.candidate_index).collect();
    assert_eq!(order, vec![1, 0, 2]);
    for ranked in ranking.ranked.iter() {
        let n = [4.0, 64.0, 1.0][ranked.candidate_index];
        let s = (TAU2 * TAU2 * n / (n * TAU2 + SIGMA2)).sqrt();
        assert!((ranked.score - closed_form_two_action(s)).abs() < 1e-14);
        assert_eq!(ranked.evaluation, ScoreEvaluation::Exact);
        assert_eq!(ranked.monte_carlo.stderr, 0.0);
    }
}
