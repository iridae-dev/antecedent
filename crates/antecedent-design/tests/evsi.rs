//! F12: EVSI through the existing `ReduceDecisionRegret` path with declared
//! integration, cost mapping, ties, truncation and source-overlap refusals.
//!
//! Independent oracles (constants written out; nothing is read from the code under test):
//!
//! * Two-action Gaussian-mean decision. `U(stay, theta) = 0`, `U(treat, theta) = theta`,
//!   prior `theta ~ N(mu0, tau^2)` with `mu0 = 1/2`, `tau^2 = 1`; `n = 4` observations of
//!   noise variance `sigma^2 = 12`. The posterior mean is preposteriorly `N(mu0, s^2)` with
//!   `s^2 = tau^2 * n tau^2 / (n tau^2 + sigma^2) = 4 / 16 = 1/4`, so `s = 1/2`. The break-even
//!   state is `mu_b = 0`, so `z = |mu0 - mu_b| / s = 1`, and
//!   `EVSI = |delta beta| * s * G(z)` with the unit normal linear-loss integral
//!   `G(z) = phi(z) - z * (1 - Phi(z))`; EVSI is the same with `s` replaced by `tau` and
//!   `z = |mu0 - mu_b| / tau = 1/2` for EVPI. Standard normal values:
//!   `phi(1) = 0.24197072451914337`, `1 - Phi(1) = 0.15865525393145707`,
//!   `phi(1/2) = 0.35206532676429952`, `1 - Phi(1/2) = 0.30853753872598690`. Hence
//!   `G(1) = 0.24197072451914337 - 0.15865525393145707 = 0.08331547058768630`,
//!   `EVSI = 0.5 * G(1) = 0.04165773529384315`, and
//!   `G(1/2) = 0.35206532676429952 - 0.5 * 0.30853753872598690 = 0.19779655740130607 = EVPI`.
//! * Binomial signal with a two-point prior `theta in {1/4, 3/4}` (equal weight), `n = 2`
//!   trials, bet (`U = theta - 1/2`) or abstain (`U = 0`). Outcomes `y = 0, 1, 2` have
//!   `P(y | 1/4) = (9/16, 6/16, 1/16)` and `P(y | 3/4) = (1/16, 6/16, 9/16)`, so
//!   `P(y) = (5/16, 6/16, 5/16)` and `P(theta = 1/4 | y) = (9/10, 1/2, 1/10)`. The bet's
//!   posterior value is `(-1/5, 0, 1/5)`; the prior value of both actions is 0. Then
//!   `EVSI = 5/16 * 1/5 = 1/16` and `EVPI = 1/2 * (0 + 1/4) = 1/8`.
//! * Frozen F12 record: binary state, guess the state (utility 1 if right), prior 1/2, signal
//!   accuracy `a`. Current value 1/2, post-signal value `a`, `EVSI = a - 1/2`, `EVPI = 1/2`.
//!   `a = 3/4` gives `EVSI = 1/4` and, with a utility-unit cost 1/10, net `3/20`.

use std::sync::Arc;

use antecedent_core::{
    CancellationToken, ExternalCapability, ExternalScientificObject, ExternalTrustState,
    ProviderObjectIdentity, QuantityRole, ScientificQuantity, SignalProviderContract,
};
use antecedent_design::evsi::{
    CostToUtilityMap, EvsiCandidate, EvsiError, EvsiRequest, IntegrationMethod, RankingBasis,
    StudyCostSpec, evaluate_evsi,
};
use antecedent_design::signal::{
    ExternalDecisionValues, ExternalLaw, ExternalSignal, ExternalSignalBody, NativeBinomialSignal,
    NativeGaussianMeanSignal, SignalLimits, SignalProvider, SignalRequest, SignalTrustLabel,
    SignalUpdateMode,
};
use antecedent_design::{
    AffineUtility, CandidateDesign, DecisionPrior, DecisionProblem, DesignCost, DesignRankConfig,
    SamplingPlan,
};

const EVSI_GAUSSIAN: f64 = 0.041_657_735_293_843_15;
const EVPI_GAUSSIAN: f64 = 0.197_796_557_401_306_07;

fn near(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

fn quantity(role: QuantityRole, id: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: id.into(),
        variable_name: id.into(),
        role,
        units: "dimensionless".into(),
        population_id: "target".into(),
        regime_id: "observational".into(),
        horizon: 0,
        functional_id: "state".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn signal_request(candidate: &str, n: u64) -> SignalRequest {
    SignalRequest {
        candidate_id: candidate.into(),
        prior_id: "prior-1".into(),
        state_quantity: quantity(QuantityRole::Outcome, "schema:theta"),
        observation_quantity: quantity(QuantityRole::Outcome, "schema:signal"),
        sample_size: n,
        rng_seed: 11,
        evidence_lineage: vec!["snapshot:a".into()],
        conditional_independence: "iid_given_state".into(),
        limits: SignalLimits::default(),
    }
}

fn sampling(n: u64) -> CandidateDesign {
    CandidateDesign::IncreaseSamplingRate(SamplingPlan {
        additional_samples: n,
        cost: DesignCost::zero(),
        tag: 0,
    })
}

fn external(
    req: &SignalRequest,
    capabilities: Vec<ExternalCapability>,
    law: ExternalLaw,
) -> Arc<dyn SignalProvider> {
    let object = ExternalScientificObject::Signal(SignalProviderContract {
        identity: ProviderObjectIdentity {
            provider_id: "lab".into(),
            object_id: format!("signal-{}", req.candidate_id),
            version_id: "v1".into(),
            snapshot_id: "snap".into(),
            request_id: req.fingerprint(),
        },
        candidate_id: req.candidate_id.clone(),
        prior_id: req.prior_id.clone(),
        observation: req.observation_quantity.clone(),
        capabilities,
    });
    let trust = ExternalTrustState::attest(&object, "lab-qa").unwrap();
    let body = ExternalSignalBody {
        sample_size: req.sample_size,
        state_quantity: req.state_quantity.clone(),
        observation_quantity: req.observation_quantity.clone(),
        law,
    };
    Arc::new(ExternalSignal::new(object, trust, body).unwrap())
}

const FACTOR: [ExternalCapability; 3] =
    [ExternalCapability::Sample, ExternalCapability::Factor, ExternalCapability::Update];
const UPDATE: [ExternalCapability; 2] = [ExternalCapability::Sample, ExternalCapability::Update];

fn candidate(
    id: &str,
    n: u64,
    provider: Arc<dyn SignalProvider>,
    cost: f64,
    unit: &str,
) -> EvsiCandidate {
    EvsiCandidate {
        semantic_id: id.into(),
        design: sampling(n),
        signal_request: signal_request(id, n),
        provider,
        cost: StudyCostSpec { amount: cost, unit: unit.into() },
        reused_observation_ids: vec!["obs-future".into()],
    }
}

fn request(
    action_ids: &[&str],
    candidates: Vec<EvsiCandidate>,
    cost_map: Option<CostToUtilityMap>,
) -> EvsiRequest {
    EvsiRequest {
        decision_contract_identity: "contract-1".into(),
        utility_unit: "utility".into(),
        action_ids: action_ids.iter().map(|s| (*s).to_owned()).collect(),
        candidates,
        cost_map,
        require_net_value: false,
        prior_observation_ids: vec!["obs-prior".into()],
        rank_config: DesignRankConfig {
            min_batches: 4,
            max_batches: 4,
            batch_size: 4,
            rank_uncertainty_threshold: 0.0,
        },
        rng_seed: 5,
        mc_error_tolerance: 1e-3,
        tie_tolerance: 1e-12,
        max_candidates: 16,
    }
}

fn utility_map() -> CostToUtilityMap {
    CostToUtilityMap {
        cost_unit: "utility".into(),
        utility_unit: "utility".into(),
        utility_per_cost: 1.0,
    }
}

/// Bet (`U = scale * (theta - 1/2) + shift`) or abstain (`U = shift`).
fn bet_problem(scale: f64, shift: f64) -> DecisionProblem<usize, f64> {
    let utility = AffineUtility::new(vec![shift, shift - 0.5 * scale], vec![0.0, scale]).unwrap();
    DecisionProblem::new(vec![0, 1], Arc::new(utility), vec![])
}

fn bet_prior() -> DecisionPrior<f64> {
    DecisionPrior::Draws(vec![0.25, 0.75])
}

const BET_ACTIONS: [&str; 2] = ["abstain", "bet"];

fn binomial_likelihood() -> ExternalLaw {
    ExternalLaw::Likelihood {
        states: vec![0.25, 0.75],
        statistics: vec![0.0, 1.0, 2.0],
        probabilities: vec![vec![0.5625, 0.0625], vec![0.375, 0.375], vec![0.0625, 0.5625]],
    }
}

fn binomial_posterior() -> ExternalLaw {
    ExternalLaw::Posterior {
        states: vec![0.25, 0.75],
        statistics: vec![0.0, 1.0, 2.0],
        predictive: vec![0.3125, 0.375, 0.3125],
        posterior: vec![vec![0.9, 0.1], vec![0.5, 0.5], vec![0.1, 0.9]],
    }
}

fn binomial_values() -> ExternalLaw {
    ExternalLaw::DecisionValues(ExternalDecisionValues {
        branch_probabilities: vec![0.3125, 0.375, 0.3125],
        action_ids: vec!["abstain".into(), "bet".into()],
        values: vec![vec![0.0, -0.2], vec![0.0, 0.0], vec![0.0, 0.2]],
    })
}

fn run(
    problem: &DecisionProblem<usize, f64>,
    prior: &DecisionPrior<f64>,
    req: &EvsiRequest,
) -> Result<antecedent_design::evsi::EvsiReport, EvsiError> {
    evaluate_evsi(problem, prior, req, &CancellationToken::new())
}

fn gaussian_problem(scale: f64, shift: f64) -> DecisionProblem<usize, f64> {
    // stay: U = shift; treat: U = scale * theta + shift (positive affine image of 0 and theta).
    let utility = AffineUtility::new(vec![shift, shift], vec![0.0, scale]).unwrap();
    DecisionProblem::new(vec![0, 1], Arc::new(utility), vec![])
}

fn gaussian_request(scale_unused: f64) -> EvsiRequest {
    let _ = scale_unused;
    let provider: Arc<dyn SignalProvider> = Arc::new(NativeGaussianMeanSignal::new(12.0).unwrap());
    request(&["stay", "treat"], vec![candidate("gauss", 4, provider, 0.0, "utility")], None)
}

#[test]
fn f12_gaussian_mean_matches_the_closed_form_normal_linear_loss_value() {
    let prior = DecisionPrior::Normal { mean: 0.5, variance: 1.0 };
    let report = run(&gaussian_problem(1.0, 0.0), &prior, &gaussian_request(1.0)).unwrap();
    let row = &report.candidates[0];
    assert!(near(row.evsi, EVSI_GAUSSIAN, 1e-7), "{} vs {EVSI_GAUSSIAN}", row.evsi);
    assert!(near(row.evpi, EVPI_GAUSSIAN, 1e-7), "{} vs {EVPI_GAUSSIAN}", row.evpi);
    assert_eq!(row.integration.method, IntegrationMethod::Exact);
    assert!(row.integration.converged);
    assert!(near(row.integration.stderr, 0.0, 0.0));
    assert_eq!(row.sample_size, 4);
    assert_eq!(row.provider_trust, SignalTrustLabel::NativeLicensed);
    assert_eq!(row.update_mode, SignalUpdateMode::NativeUpdate);
    // Prior value of treating is mu0 = 1/2 > 0, so the Bayes action is "treat" (index 1).
    assert_eq!(report.bayes_action, 1);
    assert!(near(report.prior_expected_utility, 0.5, 1e-12));
    assert!(row.net_value.is_none() && row.study_cost_utility.is_none());
    assert_eq!(report.basis, RankingBasis::Evsi);
    assert!(
        row.assumptions
            .iter()
            .any(|a| a == "terminal_action_set_identical_before_and_after_information")
    );
}

#[test]
fn f12_binomial_enumeration_matches_native_and_every_external_update_mode() {
    let problem = bet_problem(1.0, 0.0);
    let prior = bet_prior();
    let req_for = |id: &str| signal_request(id, 2);
    let providers: Vec<(&str, Arc<dyn SignalProvider>, SignalUpdateMode, SignalTrustLabel)> = vec![
        (
            "native",
            Arc::new(NativeBinomialSignal) as Arc<dyn SignalProvider>,
            SignalUpdateMode::NativeUpdate,
            SignalTrustLabel::NativeLicensed,
        ),
        (
            "ext_likelihood",
            external(&req_for("ext_likelihood"), FACTOR.to_vec(), binomial_likelihood()),
            SignalUpdateMode::NativeUpdate,
            SignalTrustLabel::ExternallyAttested,
        ),
        (
            "ext_posterior",
            external(&req_for("ext_posterior"), UPDATE.to_vec(), binomial_posterior()),
            SignalUpdateMode::ExternalPosterior,
            SignalTrustLabel::ExternallyAttested,
        ),
        (
            "ext_values",
            external(&req_for("ext_values"), UPDATE.to_vec(), binomial_values()),
            SignalUpdateMode::ExternalDecisionValues,
            SignalTrustLabel::ExternallyAttested,
        ),
    ];
    for (id, provider, mode, trust) in providers {
        let req = request(&BET_ACTIONS, vec![candidate(id, 2, provider, 0.0, "utility")], None);
        let report = run(&problem, &prior, &req).unwrap();
        let row = &report.candidates[0];
        // EVSI = 5/16 * 1/5 = 1/16; EVPI = 1/8; both from the enumeration above.
        assert!(near(row.evsi, 1.0 / 16.0, 1e-12), "{id}: {}", row.evsi);
        assert!(near(row.evpi, 0.125, 1e-12), "{id}: {}", row.evpi);
        assert_eq!(row.update_mode, mode, "{id}");
        assert_eq!(row.provider_trust, trust, "{id}");
        assert_eq!(row.signal_receipt.request_fingerprint, req_for(id).fingerprint(), "{id}");
        assert_eq!(
            row.signal_receipt.provider_identity.is_some(),
            trust != SignalTrustLabel::NativeLicensed,
            "{id}"
        );
        assert_eq!(
            row.integration.method,
            if mode == SignalUpdateMode::ExternalDecisionValues {
                IntegrationMethod::ExternallyComputed
            } else {
                IntegrationMethod::Exact
            },
            "{id}"
        );
        if mode.externally_computed() {
            assert!(
                row.assumptions
                    .iter()
                    .any(|a| a == "update_computed_externally_not_verified_by_antecedent")
            );
        }
    }
}

#[test]
fn f12_evsi_is_nonnegative_bounded_by_evpi_and_positive_affine_invariant() {
    let prior = bet_prior();
    let provider: Arc<dyn SignalProvider> = Arc::new(NativeBinomialSignal);
    let req =
        request(&BET_ACTIONS, vec![candidate("c", 2, Arc::clone(&provider), 0.0, "utility")], None);
    let base = run(&bet_problem(1.0, 0.0), &prior, &req).unwrap();
    let base_row = &base.candidates[0];
    assert!(base_row.evsi >= 0.0 && base_row.evsi <= base_row.evpi + 1e-12);
    // U' = 3 U + 10 scales EVSI and EVPI by 3 and leaves the action set unchanged.
    let scaled = run(&bet_problem(3.0, 10.0), &prior, &req).unwrap();
    let row = &scaled.candidates[0];
    assert!(near(row.evsi, 3.0 / 16.0, 1e-12), "{}", row.evsi);
    assert!(near(row.evpi, 3.0 / 8.0, 1e-12), "{}", row.evpi);
    assert_eq!(scaled.action_ids, base.action_ids);
    assert_eq!(scaled.bayes_action, base.bayes_action);
    // Gaussian: U' = 2 U + 5 doubles the closed-form values.
    let gaussian_prior = DecisionPrior::Normal { mean: 0.5, variance: 1.0 };
    let doubled =
        run(&gaussian_problem(2.0, 5.0), &gaussian_prior, &gaussian_request(2.0)).unwrap();
    let row = &doubled.candidates[0];
    assert!(near(row.evsi, 2.0 * EVSI_GAUSSIAN, 1e-7));
    assert!(near(row.evpi, 2.0 * EVPI_GAUSSIAN, 1e-7));
    assert!(row.evsi <= row.evpi);
}

#[test]
fn f12_monte_carlo_is_labelled_with_error_replicates_ess_and_convergence() {
    // Bet pays theta - 0.45 over draws {0.2, 0.5, 0.8}: EVPI = (0 + 0.05 + 0.35) / 3 - 0.05 = 1/12.
    let utility = AffineUtility::new(vec![0.0, -0.45], vec![0.0, 1.0]).unwrap();
    let problem = DecisionProblem::new(vec![0_usize, 1], Arc::new(utility), vec![]);
    let prior = DecisionPrior::Draws(vec![0.2, 0.5, 0.8]);
    let provider: Arc<dyn SignalProvider> = Arc::new(NativeGaussianMeanSignal::new(0.25).unwrap());
    let mut req = request(&BET_ACTIONS, vec![candidate("mc", 4, provider, 0.0, "utility")], None);
    req.rank_config = DesignRankConfig {
        min_batches: 16,
        max_batches: 16,
        batch_size: 64,
        rank_uncertainty_threshold: 0.0,
    };
    req.mc_error_tolerance = 1.0;
    let report = run(&problem, &prior, &req).unwrap();
    let row = &report.candidates[0];
    assert_eq!(row.integration.method, IntegrationMethod::MonteCarlo);
    assert_eq!(row.integration.replicates, 16 * 64);
    assert_eq!(row.integration.ess, Some(1024.0));
    assert!(row.integration.stderr > 0.0);
    assert!(row.integration.converged);
    assert!(near(row.evpi, 1.0 / 12.0, 1e-12));
    let slack = 4.0 * row.integration.stderr + 1e-9;
    assert!(row.evsi >= -slack && row.evsi <= row.evpi + slack, "{} / {}", row.evsi, row.evpi);
    // The same seed reproduces the estimate exactly; a zero tolerance is not converged.
    let again = run(&problem, &prior, &req).unwrap();
    assert!(near(again.candidates[0].evsi, row.evsi, 0.0));
    req.mc_error_tolerance = 0.0;
    let strict = run(&problem, &prior, &req).unwrap();
    assert!(!strict.candidates[0].integration.converged);
}

/// The frozen F12 decision: guess the binary state; accuracy `a` gives EVSI `a - 1/2`.
fn guess_problem() -> DecisionProblem<usize, f64> {
    // guess0 pays 1 - theta, guess1 pays theta.
    let utility = AffineUtility::new(vec![1.0, 0.0], vec![-1.0, 1.0]).unwrap();
    DecisionProblem::new(vec![0, 1], Arc::new(utility), vec![])
}

fn accuracy_law(accuracy: f64) -> ExternalLaw {
    ExternalLaw::Posterior {
        states: vec![0.0, 1.0],
        statistics: vec![0.0, 1.0],
        predictive: vec![0.5, 0.5],
        posterior: vec![vec![accuracy, 1.0 - accuracy], vec![1.0 - accuracy, accuracy]],
    }
}

fn guess_candidate(id: &str, accuracy: f64, cost: f64, unit: &str) -> EvsiCandidate {
    let req = signal_request(id, 1);
    candidate(id, 1, external(&req, UPDATE.to_vec(), accuracy_law(accuracy)), cost, unit)
}

fn guess_prior() -> DecisionPrior<f64> {
    DecisionPrior::Draws(vec![0.0, 1.0])
}

const GUESS_ACTIONS: [&str; 2] = ["guess0", "guess1"];

#[test]
fn f12_frozen_record_values_and_net_value_only_with_a_cost_mapping() {
    let problem = guess_problem();
    let prior = guess_prior();
    // Without a mapping: value and cost are reported separately.
    let separate = run(
        &problem,
        &prior,
        &request(&GUESS_ACTIONS, vec![guess_candidate("study", 0.75, 0.1, "usd")], None),
    )
    .unwrap();
    let row = &separate.candidates[0];
    assert!(near(row.evsi, 0.25, 1e-12));
    assert!(near(row.evpi, 0.5, 1e-12));
    assert!(near(separate.prior_expected_utility, 0.5, 1e-12));
    assert!(row.net_value.is_none() && row.study_cost_utility.is_none());
    assert_eq!(row.study_cost, StudyCostSpec { amount: 0.1, unit: "usd".into() });
    assert_eq!(separate.basis, RankingBasis::Evsi);

    // With a valid utility-unit mapping: net = 1/4 - 1/10 = 3/20.
    let net = run(
        &problem,
        &prior,
        &request(
            &GUESS_ACTIONS,
            vec![guess_candidate("study", 0.75, 0.1, "utility")],
            Some(utility_map()),
        ),
    )
    .unwrap();
    let row = &net.candidates[0];
    assert!(near(row.study_cost_utility.unwrap(), 0.1, 1e-12));
    assert!(near(row.net_value.unwrap(), 0.15, 1e-12));
    assert_eq!(net.basis, RankingBasis::NetValue);

    // A scaling map: 2 utility per currency unit prices the same cost at 1/5.
    let currency = CostToUtilityMap {
        cost_unit: "usd".into(),
        utility_unit: "utility".into(),
        utility_per_cost: 2.0,
    };
    let priced = run(
        &problem,
        &prior,
        &request(&GUESS_ACTIONS, vec![guess_candidate("study", 0.75, 0.1, "usd")], Some(currency)),
    )
    .unwrap();
    assert!(near(priced.candidates[0].net_value.unwrap(), 0.25 - 0.2, 1e-12));
}

#[test]
fn f12_incompatible_cost_units_and_missing_maps_refuse() {
    let problem = guess_problem();
    let prior = guess_prior();
    // Currency cost with a utility-unit map: units differ.
    let error = run(
        &problem,
        &prior,
        &request(
            &GUESS_ACTIONS,
            vec![guess_candidate("study", 0.75, 0.1, "usd")],
            Some(utility_map()),
        ),
    )
    .unwrap_err();
    assert!(matches!(error, EvsiError::CostUnitsMismatch(_)), "{error:?}");
    let refusal = error.to_refusal();
    assert_eq!(refusal.detail, "evsi.cost_units_mismatch");
    assert_eq!(refusal.code, "design_cost_units_mismatch");

    // A map into a different utility unit than the decision's.
    let wrong_target = CostToUtilityMap {
        cost_unit: "usd".into(),
        utility_unit: "qaly".into(),
        utility_per_cost: 1.0,
    };
    let error = run(
        &problem,
        &prior,
        &request(
            &GUESS_ACTIONS,
            vec![guess_candidate("study", 0.75, 0.1, "usd")],
            Some(wrong_target),
        ),
    )
    .unwrap_err();
    assert!(matches!(error, EvsiError::CostUnitsMismatch(_)), "{error:?}");

    // Net value demanded without any mapping.
    let mut demanding =
        request(&GUESS_ACTIONS, vec![guess_candidate("study", 0.75, 0.1, "usd")], None);
    demanding.require_net_value = true;
    let error = run(&problem, &prior, &demanding).unwrap_err();
    assert_eq!(error, EvsiError::CostMapRequired);
    assert_eq!(error.to_refusal().detail, "evsi.cost_map_required");

    // Nonpositive rates and negative costs are invalid, not silently clamped.
    let free = CostToUtilityMap { utility_per_cost: 0.0, ..utility_map() };
    let error = run(
        &problem,
        &prior,
        &request(&GUESS_ACTIONS, vec![guess_candidate("study", 0.75, 0.1, "utility")], Some(free)),
    )
    .unwrap_err();
    assert!(matches!(error, EvsiError::InvalidCost(_)));
    let error = run(
        &problem,
        &prior,
        &request(&GUESS_ACTIONS, vec![guess_candidate("study", 0.75, -1.0, "utility")], None),
    )
    .unwrap_err();
    assert!(matches!(error, EvsiError::InvalidCost(_)));
}

#[test]
fn f12_changed_terminal_action_set_and_incoherent_values_refuse() {
    let problem = bet_problem(1.0, 0.0);
    let prior = bet_prior();
    let id = "values";
    let values = |actions: [&str; 2], bet: [f64; 3]| {
        ExternalLaw::DecisionValues(ExternalDecisionValues {
            branch_probabilities: vec![0.3125, 0.375, 0.3125],
            action_ids: actions.iter().map(|s| (*s).to_owned()).collect(),
            values: vec![vec![0.0, bet[0]], vec![0.0, bet[1]], vec![0.0, bet[2]]],
        })
    };
    let attempt = |law: ExternalLaw| {
        let provider = external(&signal_request(id, 2), UPDATE.to_vec(), law);
        run(
            &problem,
            &prior,
            &request(&BET_ACTIONS, vec![candidate(id, 2, provider, 0.0, "utility")], None),
        )
        .unwrap_err()
    };
    // The provider evaluates a different action set ("other" instead of "bet").
    let error = attempt(values(["abstain", "other"], [-0.2, 0.0, 0.2]));
    assert_eq!(error, EvsiError::ActionSetChanged);
    assert_eq!(error.to_refusal().detail, "evsi.action_set_changed");
    // Branch values that do not average to the prior value (0.3125 * (-0.2 + 0.5) != 0).
    let error = attempt(values(["abstain", "bet"], [-0.2, 0.0, 0.5]));
    assert_eq!(error, EvsiError::DecisionValuesIncoherent(id.into()));
    assert_eq!(error.to_refusal().detail, "evsi.decision_values_incoherent");

    // A request whose action ids do not align with the problem is invalid.
    let provider: Arc<dyn SignalProvider> = Arc::new(NativeBinomialSignal);
    let error = run(
        &problem,
        &prior,
        &request(&["abstain"], vec![candidate("c", 2, provider, 0.0, "utility")], None),
    )
    .unwrap_err();
    assert!(matches!(error, EvsiError::InvalidRequest(_)));
}

#[test]
fn f12_external_law_for_other_prior_draws_or_sample_size_refuses() {
    let problem = bet_problem(1.0, 0.0);
    let prior = bet_prior();
    let other_states = ExternalLaw::Likelihood {
        states: vec![0.1, 0.9],
        statistics: vec![0.0, 1.0, 2.0],
        probabilities: vec![vec![0.81, 0.01], vec![0.18, 0.18], vec![0.01, 0.81]],
    };
    let provider = external(&signal_request("c", 2), FACTOR.to_vec(), other_states);
    let error = run(
        &problem,
        &prior,
        &request(&BET_ACTIONS, vec![candidate("c", 2, provider, 0.0, "utility")], None),
    )
    .unwrap_err();
    assert_eq!(error, EvsiError::PriorStatesMismatch("c".into()));
    assert_eq!(error.to_refusal().detail, "evsi.prior_states_mismatch");

    // The candidate design's declared sample size must equal its signal request's.
    let provider: Arc<dyn SignalProvider> = Arc::new(NativeBinomialSignal);
    let mut mismatched = candidate("c", 2, provider, 0.0, "utility");
    mismatched.design = sampling(3);
    let error = run(&problem, &prior, &request(&BET_ACTIONS, vec![mismatched], None)).unwrap_err();
    assert_eq!(error, EvsiError::SampleSizeMismatch("c".into()));
}

#[test]
fn f12_source_overlap_between_prior_and_study_data_refuses() {
    let provider: Arc<dyn SignalProvider> = Arc::new(NativeBinomialSignal);
    let mut overlapping = candidate("c", 2, provider, 0.0, "utility");
    overlapping.reused_observation_ids = vec!["obs-future".into(), "obs-prior".into()];
    let error =
        run(&bet_problem(1.0, 0.0), &bet_prior(), &request(&BET_ACTIONS, vec![overlapping], None))
            .unwrap_err();
    assert_eq!(error, EvsiError::SourceOverlap(vec!["obs-prior".into()]));
    assert_eq!(error.to_refusal().detail, "evsi.source_overlap");

    // The successful path reports the (empty) overlap diagnostic it checked.
    let provider: Arc<dyn SignalProvider> = Arc::new(NativeBinomialSignal);
    let report = run(
        &bet_problem(1.0, 0.0),
        &bet_prior(),
        &request(&BET_ACTIONS, vec![candidate("c", 2, provider, 0.0, "utility")], None),
    )
    .unwrap();
    let overlap = &report.candidates[0].source_overlap;
    assert!(overlap.overlapping.is_empty());
    assert_eq!(overlap.observations_checked, 2);
}

#[test]
fn f12_signal_provider_refusals_surface_with_their_namespace() {
    // The provider's exact-request check fires when the request was altered after the
    // external object was bound to its fingerprint.
    let problem = bet_problem(1.0, 0.0);
    let prior = bet_prior();
    let provider = external(&signal_request("c", 2), FACTOR.to_vec(), binomial_likelihood());
    let mut altered = candidate("c", 2, provider, 0.0, "utility");
    altered.signal_request.rng_seed = 99;
    let error = run(&problem, &prior, &request(&BET_ACTIONS, vec![altered], None)).unwrap_err();
    assert_eq!(error.to_refusal().detail, "signal_provider.request_fingerprint_mismatch");
}

#[test]
fn f12_ties_truncation_and_cancellation_are_reported() {
    let problem = bet_problem(1.0, 0.0);
    let prior = bet_prior();
    let make = |id: &str| {
        let provider: Arc<dyn SignalProvider> = Arc::new(NativeBinomialSignal);
        candidate(id, 2, provider, 0.0, "utility")
    };
    // Tied candidates (identical signal, size and cost) are flagged uncertain and ordered
    // by semantic id, whatever order they were supplied in.
    let report =
        run(&problem, &prior, &request(&BET_ACTIONS, vec![make("b"), make("a")], None)).unwrap();
    let ids: Vec<&str> = report.candidates.iter().map(|c| c.semantic_id.as_str()).collect();
    assert_eq!(ids, vec!["a", "b"]);
    assert_eq!(report.ties, vec![("a".to_owned(), "b".to_owned())]);
    assert!(report.candidates.iter().all(|c| c.rank_uncertain));
    assert!(!report.search.truncated);

    // A bounded search evaluates the first `max_candidates` in canonical order and says so.
    let mut bounded = request(&BET_ACTIONS, vec![make("c"), make("a"), make("b")], None);
    bounded.max_candidates = 2;
    let report = run(&problem, &prior, &bounded).unwrap();
    assert!(report.search.truncated);
    assert_eq!(report.search.supplied, 3);
    assert_eq!(report.search.evaluated, 2);
    assert_eq!(report.search.unevaluated_ids, vec!["c".to_owned()]);
    assert_eq!(report.candidates.len(), 2);

    // Cancellation before evaluation produces no claim.
    let token = CancellationToken::new();
    token.cancel();
    let error =
        evaluate_evsi(&problem, &prior, &request(&BET_ACTIONS, vec![make("a")], None), &token)
            .unwrap_err();
    assert_eq!(error, EvsiError::Cancelled);
    assert_eq!(error.to_refusal().detail, "evsi.cancelled");

    // Duplicate semantic ids are ambiguous.
    let error = run(&problem, &prior, &request(&BET_ACTIONS, vec![make("a"), make("a")], None))
        .unwrap_err();
    assert_eq!(error, EvsiError::DuplicateCandidate("a".into()));
}

#[test]
fn f12_candidates_rank_by_net_value_and_larger_samples_are_worth_more() {
    // Two-point prior, bet pays theta - 1/2, equal prior weights: after outcome y the
    // marginal-weighted bet value is (1/2)(1/4)(P(y | 3/4) - P(y | 1/4)), so
    // EVSI(n) = (1/8) * sum_y max(0, P(y | 3/4) - P(y | 1/4)).
    // n = 2: only y = 2 contributes, (9 - 1)/16 = 1/2, so EVSI = 1/16.
    // n = 4: P(y | 1/4) = (81, 108, 54, 12, 1)/256 and P(y | 3/4) is the reverse; y = 3 gives
    // (108 - 12)/256 and y = 4 gives (81 - 1)/256, y = 2 ties at 0, so the sum is 176/256 and
    // EVSI = 176/2048 = 11/128 < EVPI = 1/8.
    let problem = bet_problem(1.0, 0.0);
    let prior = bet_prior();
    let make = |id: &str, n: u64, cost: f64| {
        let provider: Arc<dyn SignalProvider> = Arc::new(NativeBinomialSignal);
        candidate(id, n, provider, cost, "utility")
    };
    let report = run(
        &problem,
        &prior,
        &request(
            &BET_ACTIONS,
            vec![make("small", 2, 0.0), make("large", 4, 0.0)],
            Some(utility_map()),
        ),
    )
    .unwrap();
    let large = report.candidates.iter().find(|c| c.semantic_id == "large").unwrap();
    let small = report.candidates.iter().find(|c| c.semantic_id == "small").unwrap();
    assert!(near(small.evsi, 1.0 / 16.0, 1e-12), "{}", small.evsi);
    assert!(near(large.evsi, 11.0 / 128.0, 1e-12), "{}", large.evsi);
    assert!(large.evsi <= large.evpi + 1e-12);
    assert_eq!(report.candidates[0].semantic_id, "large");
    // A cost above the extra value reverses the net-value order.
    let report = run(
        &problem,
        &prior,
        &request(
            &BET_ACTIONS,
            vec![make("small", 2, 0.0), make("large", 4, 1.0)],
            Some(utility_map()),
        ),
    )
    .unwrap();
    assert_eq!(report.candidates[0].semantic_id, "small");
    assert!(report.candidates[0].net_value.unwrap() > report.candidates[1].net_value.unwrap());
}
