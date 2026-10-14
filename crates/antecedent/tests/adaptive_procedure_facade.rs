//! Derived uncertainty of an adaptive procedure through the facade: the
//! declaration with one refusal per missing element, the adapter from the
//! retained analysis contract, the exact law, the per-paradigm reports with
//! their premise refusals, and the simulation of the weighted estimator.
//!
//! Oracles are independent: hand enumerations of one- and two-unit trees,
//! closed forms of the Beta integrals (`B(2, 3) = 1/12`), the normal
//! distribution function for a single-look stagewise p-value, and theorems
//! (the posterior mean is a martingale; an anytime-valid sequence excludes the
//! truth with probability at most alpha).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::adaptive_inference::{
    AdaptiveDesignDeclaration, AdaptiveInferenceRefusal, AdaptiveInferenceSession,
    AssignmentFamily, DelayMechanism, InterimAnalysis, MonitoringStrategy, OutcomeAvailability,
    OutcomeStructure, PolicyEpoch, ProcedureCell, StoppingRuleDeclaration, StoppingRuleKind,
    StoppingSpec,
};
use antecedent::analysis::adaptive_procedure_declaration::{
    AdaptiveProcedure, ProcedureElements, Retention,
};
use antecedent::analysis::adaptive_procedure_report::{
    ProcedureReportBody, ProcedureReportRequest,
};
use antecedent_core::{ExecutionContext, SemanticDigest};
use antecedent_stats::adaptive_inference::group_sequential::{Sidedness, StagewiseLook};
use antecedent_stats::adaptive_inference::{
    AssignmentRule, Attribution, BetaPrior, DERIVED_UNMEASURED, Estimand, ExposureModel,
    FinalSelection, OutcomeModelPrior, OutcomeObservation, ProcedureDeclaration, SimulationRequest,
    StoppingRule, WeightScheme,
};

const TOL: f64 = 1e-9;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(3)
}

fn declare(
    assignment: AssignmentRule,
    enrolment: &[u32],
    observation: OutcomeObservation,
    stopping: StoppingRule,
    selection: FinalSelection,
) -> Result<AdaptiveProcedure, Vec<AdaptiveInferenceRefusal>> {
    AdaptiveProcedure::declare(ProcedureDeclaration {
        arms: 2,
        context_probabilities: None,
        assignment: Some(assignment),
        enrolment: Some(enrolment.to_vec()),
        observation: Some(observation),
        exposure: Some(ExposureModel::none(2)),
        stopping: Some(stopping),
        selection: Some(selection),
    })
}

fn even() -> AssignmentRule {
    AssignmentRule::Fixed { probabilities: vec![0.5, 0.5] }
}

fn thompson() -> AssignmentRule {
    AssignmentRule::ThompsonBeta { prior: BetaPrior::UNIFORM }
}

fn report_of(
    procedure: &AdaptiveProcedure,
    request: &ProcedureReportRequest,
) -> Result<
    antecedent::analysis::adaptive_procedure_report::ProcedureReport,
    AdaptiveInferenceRefusal,
> {
    procedure.report(request, &ctx())
}

fn assert_premise(
    result: Result<
        antecedent::analysis::adaptive_procedure_report::ProcedureReport,
        AdaptiveInferenceRefusal,
    >,
    detail: &str,
) {
    let refusal = result.expect_err("the paradigm's premise must fail");
    assert_eq!(refusal.code, "cell_not_licensed");
    assert_eq!(refusal.detail, detail);
    assert!(!refusal.message.is_empty());
}

// -- declaration -------------------------------------------------------------------------------------

#[test]
fn each_missing_element_is_its_own_refusal() {
    let refusals = AdaptiveProcedure::declare(ProcedureDeclaration {
        arms: 2,
        context_probabilities: None,
        assignment: None,
        enrolment: None,
        observation: None,
        exposure: None,
        stopping: None,
        selection: None,
    })
    .expect_err("an empty declaration is refused");
    let details: Vec<&str> = refusals.iter().map(|r| r.detail.as_str()).collect();
    assert_eq!(
        details,
        [
            "adaptive_inference.policy_history_missing",
            "adaptive_inference.interim_analyses_missing",
            "adaptive_inference.outcome_availability_missing",
            "adaptive_inference.exposure_history_missing",
            "adaptive_inference.stopping_rule_missing",
            "adaptive_inference.policy_selection_missing",
        ]
    );
    assert!(refusals.iter().all(|r| r.code == "adaptive_contract_incomplete"));
    assert!(refusals.iter().all(|r| !r.message.is_empty()));
}

#[test]
fn an_invalid_element_is_refused_with_its_own_detail() {
    let refusals = declare(
        AssignmentRule::Fixed { probabilities: vec![0.7, 0.7] },
        &[2],
        OutcomeObservation::COMPLETE,
        StoppingRule::NoStopping,
        FinalSelection::NoSelection,
    )
    .expect_err("probabilities that do not sum to one");
    assert_eq!(refusals.len(), 1);
    assert_eq!(refusals[0].detail, "adaptive_inference.rule_mismatch");
    assert_eq!(refusals[0].code, "invalid_argument");
}

#[test]
fn the_identity_changes_with_a_changed_element_of_unchanged_shape() {
    let build = |epsilon: f64| {
        declare(
            AssignmentRule::EpsilonGreedy { epsilon, prior: BetaPrior::UNIFORM },
            &[1, 1, 1],
            OutcomeObservation::COMPLETE,
            StoppingRule::NoStopping,
            FinalSelection::NoSelection,
        )
        .unwrap()
    };
    let a = build(0.1);
    assert_eq!(a.identity(), build(0.1).identity());
    assert_ne!(a.identity(), build(0.2).identity());
    assert_eq!(a.identity().len(), 64);
    assert!(a.contract_identity().is_none());
}

// -- exact law and fixed-sample report ---------------------------------------------------------------

#[test]
fn the_fixed_design_contrast_law_matches_the_hand_enumeration() {
    // Two units, each to arm 1 or arm 0 with probability 1/2. Contrast defined only when the
    // arms receive one unit each (probability 1/2); then y1 - y0 with y1 ~ Bern(0.6),
    // y0 ~ Bern(0.2): +1 w.p. 0.48, 0 w.p. 0.12 + 0.32, -1 w.p. 0.08.
    let procedure = declare(
        even(),
        &[2],
        OutcomeObservation::COMPLETE,
        StoppingRule::NoStopping,
        FinalSelection::NoSelection,
    )
    .unwrap();
    let report = report_of(
        &procedure,
        &ProcedureReportRequest::FixedSample {
            truth: vec![0.2, 0.6],
            plus: 1,
            minus: 0,
            observed: None,
        },
    )
    .unwrap();
    assert_eq!(report.calibration, DERIVED_UNMEASURED);
    assert_eq!(report.procedure_identity, procedure.identity());
    let ProcedureReportBody::FixedSample { law, ordering } = &report.body else {
        panic!("fixed-sample report body");
    };
    assert!(ordering.is_none());
    assert!((law.undefined_mass - 0.5).abs() < TOL);
    let at = |v: f64| law.support.iter().find(|(x, _)| (x - v).abs() < 1e-12).map(|s| s.1);
    assert!((at(-1.0).unwrap() - 0.04).abs() < TOL);
    assert!((at(0.0).unwrap() - 0.22).abs() < TOL);
    assert!((at(1.0).unwrap() - 0.24).abs() < TOL);
    assert!((law.mean().unwrap() - 0.4).abs() < TOL);
}

#[test]
fn the_exact_law_carries_the_terminal_atoms_and_work() {
    let procedure = declare(
        even(),
        &[1, 1],
        OutcomeObservation::delayed(1),
        StoppingRule::NoStopping,
        FinalSelection::NoSelection,
    )
    .unwrap();
    let law = procedure.law(&[0.3, 0.7], &ctx()).unwrap();
    assert_eq!(law.calibration, DERIVED_UNMEASURED);
    assert_eq!(law.law.cumulative_enrolment, vec![1, 2]);
    let mass: f64 = law.law.atoms.iter().map(|a| a.probability).sum();
    assert!((mass - 1.0).abs() < TOL);
    // Lag one: at the horizon only the first unit's outcome is revealed, so every atom has one
    // observation and the second unit is pending.
    assert!(
        law.law.atoms.iter().all(|a| a.observed.iter().map(|&n| u32::from(n)).sum::<u32>() == 1)
    );
    assert!(law.law.work_units > 0);
    let wrong = procedure.law(&[0.3], &ctx()).expect_err("truth of the wrong shape");
    assert_eq!(wrong.code, "invalid_argument");
}

// -- premise refusals --------------------------------------------------------------------------------

#[test]
fn every_paradigm_whose_premise_fails_is_refused_with_cell_not_licensed() {
    let adaptive_stopped = declare(
        thompson(),
        &[1, 1, 1],
        OutcomeObservation::COMPLETE,
        StoppingRule::PosteriorProbabilityBest { threshold: 0.9, prior: BetaPrior::UNIFORM },
        FinalSelection::NoSelection,
    )
    .unwrap();
    assert_premise(
        report_of(
            &adaptive_stopped,
            &ProcedureReportRequest::FixedSample {
                truth: vec![0.3, 0.6],
                plus: 1,
                minus: 0,
                observed: None,
            },
        ),
        "adaptive_inference.fixed_sample_design_adaptive",
    );
    assert_premise(
        report_of(
            &adaptive_stopped,
            &ProcedureReportRequest::GroupSequential {
                truth: vec![0.3, 0.6],
                level: 0.95,
                observed: None,
            },
        ),
        "adaptive_inference.group_sequential_assignment_adaptive",
    );
    assert_premise(
        report_of(
            &adaptive_stopped,
            &ProcedureReportRequest::DecisionTheoretic { truth: vec![0.3, 0.6] },
        ),
        "adaptive_inference.rule_mismatch",
    );

    let fixed_stopped = declare(
        even(),
        &[2, 2],
        OutcomeObservation::COMPLETE,
        StoppingRule::ResourceCap { max_units: 2 },
        FinalSelection::NoSelection,
    )
    .unwrap();
    assert_premise(
        report_of(
            &fixed_stopped,
            &ProcedureReportRequest::FixedSample {
                truth: vec![0.3, 0.6],
                plus: 1,
                minus: 0,
                observed: None,
            },
        ),
        "adaptive_inference.fixed_sample_data_dependent_stopping",
    );

    let lossy = declare(
        thompson(),
        &[1, 1],
        OutcomeObservation { lag_looks: 0, retain_success: 1.0, retain_failure: 0.5 },
        StoppingRule::NoStopping,
        FinalSelection::NoSelection,
    )
    .unwrap();
    assert_premise(
        report_of(
            &lossy,
            &ProcedureReportRequest::BayesianPosterior {
                prior: BetaPrior::UNIFORM,
                observed: None,
            },
        ),
        "adaptive_inference.posterior_premise_outcome_dependent_delay",
    );
    assert_premise(
        report_of(
            &lossy,
            &ProcedureReportRequest::AnytimeValid {
                alpha: 0.1,
                tuning_samples: 10.0,
                observed: None,
                truth: Some(vec![0.3, 0.6]),
            },
        ),
        "adaptive_inference.outcome_dependent_delay",
    );
    let refusal = lossy
        .simulate(
            &[0.3, 0.6],
            &SimulationRequest {
                replications: 10,
                seed: 1,
                estimand: Estimand::Contrast { plus: 1, minus: 0 },
                scheme: WeightScheme::Uniform,
                outcome_prior: OutcomeModelPrior { mean: 0.5, strength: 2.0 },
                probability_floor: 0.01,
            },
            &ctx(),
        )
        .expect_err("adaptive weighting premise");
    assert_eq!(refusal.code, "cell_not_licensed");
    assert_eq!(refusal.detail, "adaptive_inference.outcome_dependent_delay");

    let contextual = AdaptiveProcedure::declare(ProcedureDeclaration {
        arms: 2,
        context_probabilities: Some(vec![0.5, 0.5]),
        assignment: Some(thompson()),
        enrolment: Some(vec![2]),
        observation: Some(OutcomeObservation::COMPLETE),
        exposure: Some(ExposureModel::none(2)),
        stopping: Some(StoppingRule::NoStopping),
        selection: Some(FinalSelection::NoSelection),
    })
    .unwrap();
    assert_premise(
        report_of(
            &contextual,
            &ProcedureReportRequest::AnytimeValid {
                alpha: 0.1,
                tuning_samples: 10.0,
                observed: None,
                truth: None,
            },
        ),
        "adaptive_inference.anytime_premise_contexts",
    );
}

// -- decision-theoretic and Bayesian reports ---------------------------------------------------------

#[test]
fn the_expected_loss_rule_decides_with_hand_integrated_losses() {
    // All units go to arm 0 (probabilities 1, 0). After one outcome the posterior of arm 0 is
    // Beta(2, 1) (success) or Beta(1, 2) (failure); arm 1 stays uniform. The smallest expected
    // loss is B(2, 3) = 1/12 < 0.1 in both cases, so the rule stops at the first look and decides
    // arm 0 after a success and arm 1 after a failure.
    let procedure = declare(
        AssignmentRule::Fixed { probabilities: vec![1.0, 0.0] },
        &[1, 1],
        OutcomeObservation::COMPLETE,
        StoppingRule::PosteriorExpectedLoss { loss_threshold: 0.1, prior: BetaPrior::UNIFORM },
        FinalSelection::NoSelection,
    )
    .unwrap();
    let report =
        report_of(&procedure, &ProcedureReportRequest::DecisionTheoretic { truth: vec![0.7, 0.4] })
            .unwrap();
    let ProcedureReportBody::DecisionTheoretic(d) = &report.body else { panic!("body") };
    assert!((d.stop_probabilities[0] - 1.0).abs() < 1e-8);
    assert!(d.horizon_probability.abs() < 1e-8);
    assert!((d.expected_enrolled - 1.0).abs() < 1e-8);
    assert!((d.decision_probabilities[0] - 0.7).abs() < 1e-8);
    assert!((d.decision_probabilities[1] - 0.3).abs() < 1e-8);
    assert!((d.wrong_decision_probability - 0.3).abs() < 1e-8);
    assert!((d.expected_regret - 0.3 * (0.7 - 0.4)).abs() < 1e-8);
    assert_eq!(report.calibration, DERIVED_UNMEASURED);
}

#[test]
fn the_posterior_mean_is_a_martingale_of_an_adaptive_stopped_procedure() {
    // Thompson assignment, a stopping rule on observed counts and an ignorable delay: the
    // terminal posterior mean has expectation equal to the prior mean in every cell.
    let procedure = declare(
        thompson(),
        &[1, 1, 1, 1],
        OutcomeObservation::delayed(1),
        StoppingRule::PosteriorProbabilityBest { threshold: 0.8, prior: BetaPrior::UNIFORM },
        FinalSelection::HighestObservedMean,
    )
    .unwrap();
    let report = report_of(
        &procedure,
        &ProcedureReportRequest::BayesianPosterior {
            prior: BetaPrior { alpha: 2.0, beta: 1.0 },
            observed: Some(vec![3, 1, 1, 2]),
        },
    )
    .unwrap();
    let ProcedureReportBody::Bayesian { posterior, mean_gaps } = &report.body else {
        panic!("body")
    };
    for gap in mean_gaps {
        assert!((gap.prior_mean - 2.0 / 3.0).abs() < TOL);
        assert!(gap.gap.abs() < 1e-9, "gap {} in cell {}", gap.gap, gap.cell);
    }
    // Conjugate closed form at the observed counts: Beta(2 + s, 1 + f).
    let posterior = posterior.as_ref().unwrap();
    assert!((posterior.posteriors[0].mean() - 5.0 / 7.0).abs() < TOL);
    assert!((posterior.posteriors[1].mean() - 3.0 / 6.0).abs() < TOL);
}

// -- anytime-valid, group-sequential -----------------------------------------------------------------

#[test]
fn the_anytime_bounds_exclude_the_truth_with_probability_at_most_alpha() {
    let procedure = declare(
        thompson(),
        &[1; 8],
        OutcomeObservation::COMPLETE,
        StoppingRule::PosteriorProbabilityBest { threshold: 0.9, prior: BetaPrior::UNIFORM },
        FinalSelection::NoSelection,
    )
    .unwrap();
    for truth in [[0.0, 1.0], [0.3, 0.6], [0.5, 0.5], [0.95, 0.05]] {
        let report = report_of(
            &procedure,
            &ProcedureReportRequest::AnytimeValid {
                alpha: 0.1,
                tuning_samples: 8.0,
                observed: Some(vec![2, 1, 1, 3]),
                truth: Some(truth.to_vec()),
            },
        )
        .unwrap();
        let ProcedureReportBody::AnytimeValid { report: bounds, exclusion_probability } =
            &report.body
        else {
            panic!("body")
        };
        let probability = exclusion_probability.unwrap();
        assert!((0.0..=0.1 + 1e-12).contains(&probability), "{truth:?}: {probability}");
        let bounds = bounds.as_ref().unwrap();
        assert!((bounds.alpha_per_cell - 0.05).abs() < TOL);
        // Observed means 2/3 and 1/4 lie inside their own bounds.
        for b in &bounds.bounds {
            let mean = b.mean.unwrap();
            assert!(b.lower <= mean && mean <= b.upper);
        }
    }
}

#[test]
fn a_single_look_stagewise_inference_is_the_normal_interval() {
    // One look: the upper p-value at drift theta is 1 - Phi(z - theta). The median-unbiased
    // estimate is z; the 95% limits are z -/+ 1.959964; the one-sided p-value at zero is
    // 1 - Phi(1.5) = 0.0668072.
    let procedure = declare(
        even(),
        &[10],
        OutcomeObservation::COMPLETE,
        StoppingRule::WaldBoundaries { critical: vec![1.96], sidedness: Sidedness::OneSidedUpper },
        FinalSelection::NoSelection,
    )
    .unwrap();
    let report = report_of(
        &procedure,
        &ProcedureReportRequest::GroupSequential {
            truth: vec![0.5, 0.5],
            level: 0.95,
            observed: Some(StagewiseLook { look: 0, statistic: 1.5 }),
        },
    )
    .unwrap();
    let ProcedureReportBody::GroupSequential {
        critical,
        information_fractions,
        stop_probabilities,
        horizon_probability,
        stagewise,
        ..
    } = &report.body
    else {
        panic!("body")
    };
    assert_eq!(critical, &vec![1.96]);
    assert_eq!(information_fractions, &vec![1.0]);
    assert!((stop_probabilities[0] + horizon_probability - 1.0).abs() < TOL);
    let s = stagewise.unwrap();
    assert!((s.median_unbiased - 1.5).abs() < 1e-5);
    assert!((s.lower - (1.5 - 1.959_963_985)).abs() < 1e-4);
    assert!((s.upper - (1.5 + 1.959_963_985)).abs() < 1e-4);
    assert!((s.p_value_at_null - 0.066_807_2).abs() < 1e-6);
}

// -- simulation --------------------------------------------------------------------------------------

fn weighting_request(replications: u32) -> SimulationRequest {
    SimulationRequest {
        replications,
        seed: 20_240_607,
        estimand: Estimand::Contrast { plus: 1, minus: 0 },
        scheme: WeightScheme::Uniform,
        outcome_prior: OutcomeModelPrior { mean: 0.5, strength: 2.0 },
        probability_floor: 0.01,
    }
}

#[test]
fn the_simulated_weighted_estimate_recovers_the_contrast_and_replays() {
    let procedure = declare(
        AssignmentRule::EpsilonGreedy { epsilon: 0.5, prior: BetaPrior::UNIFORM },
        &[10, 10, 10, 10],
        OutcomeObservation::COMPLETE,
        StoppingRule::NoStopping,
        FinalSelection::HighestObservedMean,
    )
    .unwrap();
    let request = ProcedureReportRequest::AdaptiveWeighting {
        truth: vec![0.3, 0.6],
        simulation: weighting_request(600),
        z: 1.96,
    };
    let first = report_of(&procedure, &request).unwrap();
    let ProcedureReportBody::AdaptiveWeighting { summary, truth_value, replications, .. } =
        &first.body
    else {
        panic!("body")
    };
    assert!((truth_value - 0.3).abs() < TOL);
    assert_eq!(*replications, 600);
    assert!(summary.defined > 550);
    assert!(
        summary.bias.abs() <= 5.0 * summary.monte_carlo_standard_error + 0.02,
        "bias {} against MCSE {}",
        summary.bias,
        summary.monte_carlo_standard_error
    );
    // The same seed and request replay to the same report.
    assert_eq!(first, report_of(&procedure, &request).unwrap());
    // A changed seed at unchanged shape changes the replicates.
    let mut other = weighting_request(600);
    other.seed += 1;
    let moved = report_of(
        &procedure,
        &ProcedureReportRequest::AdaptiveWeighting {
            truth: vec![0.3, 0.6],
            simulation: other,
            z: 1.96,
        },
    )
    .unwrap();
    assert_ne!(first.body, moved.body);
}

#[test]
fn a_deterministic_rule_has_no_weighted_estimate() {
    let procedure = declare(
        AssignmentRule::UpperConfidenceBound { exploration: 2.0 },
        &[1; 6],
        OutcomeObservation::COMPLETE,
        StoppingRule::NoStopping,
        FinalSelection::HighestObservedMean,
    )
    .unwrap();
    let refusal = procedure
        .simulate(&[0.3, 0.6], &weighting_request(5), &ctx())
        .expect_err("assignment probabilities of zero and one violate the floor");
    assert_eq!(refusal.code, "adaptive_assumption_violated");
    assert_eq!(refusal.detail, "adaptive_inference.positivity_floor");
}

// -- the adapter from the retained contract ----------------------------------------------------------

fn contract_declaration(exposure: Vec<u32>) -> AdaptiveDesignDeclaration {
    let assigned = vec![0_u32, 1, 0, 1, 0, 1];
    AdaptiveDesignDeclaration {
        label: "even".to_owned(),
        cell: ProcedureCell {
            assignment: AssignmentFamily::FixedRandomization,
            outcome: OutcomeStructure::SingleOutcome,
            monitoring: MonitoringStrategy::FixedInterimLooks,
            stopping: StoppingRuleKind::NoStopping,
        },
        arm_labels: vec!["control".to_owned(), "treatment".to_owned()],
        assigned_arm: assigned,
        policy_history: Some(vec![PolicyEpoch {
            first_unit: 0,
            policy_identity: SemanticDigest::from_bytes([7; 32]),
        }]),
        assignment_probabilities: Some(vec![vec![0.5, 0.5]; 6]),
        exposure: Some(exposure),
        availability: Some(OutcomeAvailability {
            available_at_look: vec![Some(0), Some(0), Some(0), Some(1), Some(1), Some(1)],
            delay: DelayMechanism::Immediate,
        }),
        outcomes: vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0], vec![0.0], vec![1.0]],
        interim_analyses: Some(vec![
            InterimAnalysis { units_enrolled: 3, information_fraction: 0.5 },
            InterimAnalysis { units_enrolled: 6, information_fraction: 1.0 },
        ]),
        stopping_rule: Some(StoppingRuleDeclaration {
            kind: StoppingRuleKind::NoStopping,
            spec: StoppingSpec::NoStopping,
            ended_by_rule: false,
        }),
        probability_floor: 0.01,
    }
}

fn elements() -> ProcedureElements {
    ProcedureElements {
        assignment: Some(even()),
        selection: Some(FinalSelection::NoSelection),
        ..ProcedureElements::default()
    }
}

#[test]
fn the_adapter_takes_the_schedule_from_the_contract_and_ties_the_identity() {
    let session =
        AdaptiveInferenceSession::prepare(contract_declaration(vec![0, 1, 0, 1, 0, 1])).unwrap();
    let procedure = session.procedure(&elements(), &ctx()).unwrap();
    assert_eq!(procedure.spec().enrolment(), &[3, 3]);
    assert_eq!(procedure.spec().observation().lag_looks, 0);
    assert_eq!(procedure.contract_identity(), Some(session.contract_identity().as_str()));
    assert!(procedure.taken_from_contract().contains(&"enrolment_schedule"));
    assert_eq!(procedure.spec().family_name(), "fixed_randomization");
    // The derived report is bound to both identities and the contract's cell.
    let report = report_of(
        &procedure,
        &ProcedureReportRequest::FixedSample {
            truth: vec![0.3, 0.7],
            plus: 1,
            minus: 0,
            observed: None,
        },
    )
    .unwrap();
    assert_eq!(report.contract_identity.as_deref(), Some(session.contract_identity().as_str()));
    // Six units were enrolled whatever the data, so the law carries enrolled == 6 at the horizon.
    let law = procedure.law(&[0.3, 0.7], &ctx()).unwrap();
    assert!(law.law.atoms.iter().all(|a| a.enrolled == 6 && !a.rule_stopped));
}

#[test]
fn the_adapter_refuses_each_missing_element_and_each_disagreement() {
    let session =
        AdaptiveInferenceSession::prepare(contract_declaration(vec![0, 1, 0, 1, 0, 1])).unwrap();
    let missing = session
        .procedure(&ProcedureElements::default(), &ctx())
        .expect_err("no assignment rule and no selection");
    let details: Vec<&str> = missing.iter().map(|r| r.detail.as_str()).collect();
    assert!(details.contains(&"adaptive_inference.policy_history_missing"));
    assert!(details.contains(&"adaptive_inference.policy_selection_missing"));

    let mismatch = session
        .procedure(
            &ProcedureElements {
                assignment: Some(AssignmentRule::Fixed { probabilities: vec![0.6, 0.4] }),
                ..elements()
            },
            &ctx(),
        )
        .expect_err("realized rows do not replay the declared probabilities");
    assert_eq!(mismatch[0].detail, "adaptive_inference.rule_mismatch");

    let wrong_family = session
        .procedure(&ProcedureElements { assignment: Some(thompson()), ..elements() }, &ctx())
        .expect_err("a Thompson rule against a fixed-randomization cell");
    assert_eq!(wrong_family[0].detail, "adaptive_inference.rule_mismatch");

    // Exposure that differs from the assignments needs a declared failure model.
    let failing =
        AdaptiveInferenceSession::prepare(contract_declaration(vec![0, 0, 0, 1, 0, 1])).unwrap();
    let missing_exposure = failing.procedure(&elements(), &ctx()).expect_err("no exposure model");
    assert_eq!(missing_exposure[0].detail, "adaptive_inference.exposure_history_missing");
    let with_model = failing
        .procedure(
            &ProcedureElements {
                exposure: Some(ExposureModel {
                    failure_probability: vec![0.0, 0.4],
                    fallback_arm: 0,
                    attribution: Attribution::Assigned,
                }),
                ..elements()
            },
            &ctx(),
        )
        .unwrap();
    assert_eq!(with_model.spec().exposure().fallback_arm, 0);
    // Delivered-arm retention disagrees with the contract's immediate availability.
    let retained = session
        .procedure(
            &ProcedureElements {
                retention: Some(Retention { success: 1.0, failure: 0.5 }),
                ..elements()
            },
            &ctx(),
        )
        .expect_err("outcome-dependent retention against an immediate contract");
    assert_eq!(retained[0].detail, "adaptive_inference.outcome_availability_invalid");
}
