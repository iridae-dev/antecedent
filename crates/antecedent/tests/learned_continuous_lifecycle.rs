//! Learned continuous-outcome trial transport (2.2A cell X4): known-truth execution under
//! both sampling designs, overlap refusals, the estimator menu, thread-budget invariance,
//! the prepared lifecycle, the independently consumed artifact and the closed interval route.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "learned_continuous_dgp/mod.rs"]
mod dgp;

use antecedent::{PreparedLearnedContinuous, StudyBuilder, consume_learned_continuous_artifact};
use antecedent_core::{ExecutionContext, NonZeroThreadCount, Parallelism};
use antecedent_estimate::{
    EstimationError, LearnedContinuousOptions, LearnerSpec, LinearSpec,
    learned_continuous_interval_internal,
};
use antecedent_io::IoError;
use antecedent_io::learned_continuous_artifact::{
    LearnedContinuousArtifactError as Refusal, LearnedContinuousArtifactWire,
    LearnedContinuousConsumeLimits,
};
use dgp::{Design, Scenario, draw, graph};

const DESIGNS: [Design; 2] = [Design::NestedCohort, Design::IndependentSamples];

fn options() -> LearnedContinuousOptions {
    LearnedContinuousOptions {
        outcome: LearnerSpec::Linear(LinearSpec::default()),
        folds: 3,
        ..LearnedContinuousOptions::default()
    }
}

fn prepare(
    design: Design,
    scenario: Scenario,
    n: (usize, usize),
    options: LearnedContinuousOptions,
    seed: u64,
) -> Result<PreparedLearnedContinuous, IoError> {
    let (diagram, query, names) = graph();
    let ctx = ExecutionContext::for_tests(seed);
    StudyBuilder::learned_continuous_transport(
        diagram,
        query,
        draw(design, scenario, n.0, n.1, seed),
        options,
        names,
        &ctx,
    )
}

fn refused_code(error: IoError) -> (&'static str, String) {
    match error {
        IoError::Refused { code, message } => (code, message),
        other => panic!("expected a coded refusal, got {other}"),
    }
}

#[test]
fn good_overlap_known_truth_under_both_designs() {
    for design in DESIGNS {
        let prepared = prepare(design, Scenario::Good, (1000, 600), options(), 11).unwrap();
        let ctx = ExecutionContext::for_tests(11);
        let result = prepared.estimate(&ctx).unwrap();
        let estimate = result.estimate();
        assert!(
            (estimate.estimate - Scenario::Good.truth()).abs() < 0.3,
            "{design:?}: {} vs {}",
            estimate.estimate,
            Scenario::Good.truth()
        );
        // The interval is not requested; nothing is attached either way.
        assert_eq!(estimate.uncertainty.status, "point_only");
        assert!(!estimate.uncertainty.available());
        // Source-membership and treatment overlap are distinct diagnostics.
        assert!(estimate.overlap.selection.probability_min > 0.05);
        assert!((estimate.overlap.treatment.probability_min - 0.5).abs() < 1e-12);
        assert!(
            (estimate.overlap.selection.probability_min
                - estimate.overlap.treatment.probability_min)
                .abs()
                > 1e-3
        );
        // Provider implementation/version for every fitted nuisance, and the fold scheme.
        assert_eq!(estimate.provenance.len(), 3 * 3);
        assert!(estimate.provenance.iter().all(|p| !p.implementation.is_empty()));
        assert_eq!(estimate.folds.count, 3);
        assert_eq!(estimate.folds.assignment.len(), 1600);
    }
}

/// Plug-in over the target rows: what an estimator that trusts the outcome learner alone
/// (no membership correction) reports.
fn outcome_plug_in(
    estimate: &antecedent_estimate::LearnedContinuousEstimate,
    source: &[bool],
) -> f64 {
    let target: Vec<usize> = (0..source.len()).filter(|i| !source[*i]).collect();
    target.iter().map(|&i| estimate.mu1[i] - estimate.mu0[i]).sum::<f64>() / target.len() as f64
}

/// Trial-only mean contrast: an estimator that ignores membership altogether.
fn trial_only_contrast(input: &antecedent_estimate::TrialAipwInput) -> f64 {
    let mean = |arm: bool| {
        let ys: Vec<f64> = (0..input.source.len())
            .filter(|&i| input.source[i] && input.treatment[i] == arm)
            .map(|i| input.outcome[i])
            .collect();
        ys.iter().sum::<f64>() / ys.len() as f64
    };
    mean(true) - mean(false)
}

#[test]
fn misspecified_nuisance_cases_match_only_the_claimed_robustness() {
    // Model double robustness: one nuisance family wrong, the other right. Each scenario
    // has a real target shift, so an estimator that trusted only the wrong family would
    // land far from truth while the augmented estimator stays close. Both families wrong
    // at once is outside the theorem and is not asserted.
    const TOLERANCE: f64 = 0.45;
    let (n, seed) = ((4000, 3000), 5);
    for design in DESIGNS {
        // Outcome learner wrong (linear on a quadratic effect), membership right: the
        // augmented estimate is near truth; the outcome plug-in and the trial-only
        // contrast (both blind to the participation correction) are far from it.
        let scenario = Scenario::MisspecifiedOutcome;
        let input = draw(design, scenario, n.0, n.1, seed);
        let result = prepare(design, scenario, n, options(), seed)
            .unwrap()
            .estimate(&ExecutionContext::for_tests(seed))
            .unwrap();
        let aipw = result.estimate().estimate;
        let plug_in = outcome_plug_in(result.estimate(), &input.source);
        let trial_only = trial_only_contrast(&input);
        assert!((aipw - scenario.truth()).abs() < TOLERANCE, "{design:?} outcome-wrong: {aipw}");
        for (name, alternative) in [("plug-in", plug_in), ("trial-only", trial_only)] {
            assert!(
                (alternative - scenario.truth()).abs() > 2.5 * TOLERANCE,
                "{design:?}: the {name} alternative {alternative} is not visibly wrong"
            );
        }
        // The analytic limits of the alternatives are what the DGP documents.
        assert!((plug_in - scenario.linear_plug_in()).abs() < 0.5, "{design:?}: {plug_in}");
        assert!((trial_only - scenario.trial_only_contrast()).abs() < 0.5);

        // Membership model wrong (quadratic log-odds vs linear logistic), outcome right:
        // near truth, while ignoring membership (the trial contrast) is far from it.
        let scenario = Scenario::MisspecifiedMembership;
        let input = draw(design, scenario, n.0, n.1, seed);
        let result = prepare(design, scenario, n, options(), seed)
            .unwrap()
            .estimate(&ExecutionContext::for_tests(seed))
            .unwrap();
        let aipw = result.estimate().estimate;
        let trial_only = trial_only_contrast(&input);
        assert!((aipw - scenario.truth()).abs() < TOLERANCE, "{design:?} membership-wrong: {aipw}");
        assert!(
            (trial_only - scenario.truth()).abs() > 2.0 * TOLERANCE,
            "{design:?}: ignoring membership ({trial_only}) is not visibly wrong"
        );
        assert!((trial_only - scenario.trial_only_contrast()).abs() < 0.5);
    }
}

#[test]
fn weak_overlap_refuses_rather_than_extrapolating() {
    for design in DESIGNS {
        let prepared = prepare(design, Scenario::WeakOverlap, (1000, 600), options(), 3).unwrap();
        let error = prepared.estimate(&ExecutionContext::for_tests(3)).unwrap_err();
        let (code, message) = refused_code(error);
        assert_eq!(code, "transport_support_failure");
        assert!(message.starts_with("learned_transport.membership_overlap"), "{message}");
    }
    // Known randomization probabilities outside the declared bound refuse at preparation.
    let (diagram, query, names) = graph();
    let mut input = draw(Design::IndependentSamples, Scenario::Good, 300, 200, 1);
    input.randomization[0] = 0.01;
    let error = StudyBuilder::learned_continuous_transport(
        diagram,
        query,
        input,
        options(),
        names,
        &ExecutionContext::for_tests(1),
    )
    .unwrap_err();
    let (code, message) = refused_code(error);
    assert_eq!(code, "transport_support_failure");
    assert!(message.starts_with("learned_transport.treatment_overlap"), "{message}");
}

#[test]
fn declared_bounds_and_designs_refuse_with_their_details() {
    let too_many_folds = LearnedContinuousOptions { folds: 21, ..options() };
    let (code, message) = refused_code(
        prepare(Design::IndependentSamples, Scenario::Good, (400, 300), too_many_folds, 1)
            .unwrap_err(),
    );
    assert_eq!(code, "route_not_supported");
    assert!(message.starts_with("learned_transport.bounds_exceeded"), "{message}");
    let too_many_replicates = LearnedContinuousOptions { bootstrap: 2001, ..options() };
    let (code, message) = refused_code(
        prepare(Design::NestedCohort, Scenario::Good, (400, 300), too_many_replicates, 1)
            .unwrap_err(),
    );
    assert_eq!(
        (code, message.starts_with("learned_transport.bounds_exceeded")),
        ("route_not_supported", true)
    );
    // Clustered or undeclared sampling and heterogeneous targets are refused by name.
    let error = antecedent_estimate::parse_learned_continuous_sampling("clustered").unwrap_err();
    assert!(
        matches!(&error, EstimationError::Refused { code: "sampling_dependence_unknown", message } if message.starts_with("learned_transport.non_iid_design"))
    );
    let error = antecedent_estimate::parse_learned_continuous_target("cate").unwrap_err();
    assert!(
        matches!(&error, EstimationError::Refused { code: "route_not_supported", message } if message.starts_with("learned_transport.cate_requested"))
    );
    // A replicate request below the floor keeps the point and withholds the interval.
    let below = LearnedContinuousOptions { bootstrap: 50, ..options() };
    let result = prepare(Design::NestedCohort, Scenario::Good, (400, 300), below, 2)
        .unwrap()
        .estimate(&ExecutionContext::for_tests(2))
        .unwrap();
    let uncertainty = &result.estimate().uncertainty;
    assert_eq!(uncertainty.reason, "estimator_inference_mismatch");
    assert_eq!(uncertainty.detail.as_deref(), Some("learned_transport.bootstrap_below_floor"));
    assert!(result.estimate().estimate.is_finite());
}

#[test]
#[allow(clippy::too_many_lines, reason = "one scenario per assertion group")]
fn menu_lists_refused_estimators_with_reasons_consistent_with_the_graph() {
    let (diagram, query, _) = graph();
    let menu = StudyBuilder::learned_continuous_menu(&diagram, &query, None).unwrap();
    let by_name = |name: &str| menu.entries.iter().find(|e| e.estimator == name).unwrap();
    assert!(by_name("learned_trial_aipw").eligible);
    assert!(by_name("trial_ipw_supplied_probabilities").eligible);
    assert!(!by_name("dr_learner_cate").eligible);
    assert_eq!(
        by_name("dr_learner_cate").refusal.as_ref().unwrap().detail.as_deref(),
        Some("learned_transport.cate_requested")
    );
    assert!(!by_name("exact_finite_law_evaluator").eligible);
    // Selection stays manual: no entry is ranked or recommended.
    assert_eq!(menu.selection, "manual");
    assert!(!serde_json::to_string(&menu).unwrap().contains("recommend"));
    for entry in &menu.entries {
        assert_eq!(entry.eligible, entry.refusal.is_none(), "{}", entry.estimator);
        assert!(!entry.required_laws.is_empty() && !entry.nuisance_tasks.is_empty());
        assert!(!entry.support_requirements.is_empty() && !entry.sampling_design.is_empty());
        assert!(!entry.uncertainty_status.is_empty());
    }
    // Computed fields follow the certificate, learners, thresholds, sampling and route status.
    let learned = by_name("learned_trial_aipw");
    assert!(
        learned.required_laws.iter().any(|l| l.contains("X=[v0]")),
        "{:?}",
        learned.required_laws
    );
    assert!(learned.required_graph_conditions.iter().any(|c| c.contains("standardizers [v0]")));
    assert!(
        learned.nuisance_tasks.iter().any(|t| t.contains("learner ridge")),
        "{:?}",
        learned.nuisance_tasks
    );
    assert!(learned.nuisance_tasks.iter().all(|t| t.contains("default")));
    assert!(learned.static_fields.is_empty());
    assert!(by_name("dr_learner_cate").static_fields.contains(&"required_laws".to_owned()));
    let tuned = LearnedContinuousOptions {
        outcome: LearnerSpec::Linear(LinearSpec::default()),
        membership: LearnerSpec::Logistic(antecedent_estimate::LogisticSpec::default()),
        folds: 7,
        min_membership_probability: 0.1,
        min_treatment_probability: 0.2,
        bootstrap: 300,
        ..LearnedContinuousOptions::default()
    };
    let id = antecedent_identify::TransportIdentifier::new().identify(&diagram, &query).unwrap();
    let context =
        |options, sampling| antecedent_estimate::MenuContext { options, learners: None, sampling };
    let menu_tuned = antecedent_estimate::transport_estimator_menu_with(
        &id,
        &query,
        &context(Some(tuned), Some(antecedent_estimate::TrialSampling::NestedCohort)),
    );
    let tuned_entry = &menu_tuned.entries[0];
    assert!(
        tuned_entry.nuisance_tasks[0].contains("learner linear")
            && tuned_entry.nuisance_tasks[0].contains("7-fold")
    );
    assert!(tuned_entry.nuisance_tasks.iter().all(|t| !t.contains("default")));
    assert!(tuned_entry.support_requirements[0].contains("at least 0.1 "));
    assert!(tuned_entry.support_requirements[1].contains("[0.2, 0.8]"));
    assert_eq!(tuned_entry.sampling_design, ["nested_cohort"]);
    assert!(tuned_entry.uncertainty_status.starts_with("withheld: cell_not_licensed"));
    assert!(tuned_entry.uncertainty_status.contains("300 bootstrap replicates"));
    let below = LearnedContinuousOptions { bootstrap: 50, ..tuned };
    let menu_below = antecedent_estimate::transport_estimator_menu_with(
        &id,
        &query,
        &context(Some(below), None),
    );
    assert!(menu_below.entries[0].uncertainty_status.contains("bootstrap_below_floor"));
    assert_eq!(menu_below.entries[0].sampling_design.len(), 2);
    assert_ne!(menu.entries[0].nuisance_tasks, tuned_entry.nuisance_tasks);
    assert_ne!(menu.entries[0].support_requirements, tuned_entry.support_requirements);
    assert_ne!(menu.entries[0].uncertainty_status, tuned_entry.uncertainty_status);
    // A direct certificate (no selection) names no standardizer.
    let mut direct_graph = antecedent_graph::Admg::with_variables(3);
    for (from, to) in [(0, 2), (1, 2)] {
        direct_graph
            .insert_directed(
                antecedent_graph::DenseNodeId::from_raw(from),
                antecedent_graph::DenseNodeId::from_raw(to),
            )
            .unwrap();
    }
    let direct = antecedent_graph::SelectionDiagram::try_new(
        direct_graph,
        Vec::<antecedent_core::VariableId>::new(),
    )
    .unwrap();
    let direct_menu = StudyBuilder::learned_continuous_menu(&direct, &query, None).unwrap();
    let direct_entry = &direct_menu.entries[0];
    {
        assert!(direct_entry.eligible, "{:?}", direct_entry.refusal);
        assert!(
            direct_entry.required_laws[0].contains("no standardizer"),
            "{:?}",
            direct_entry.required_laws
        );
        assert_ne!(direct_entry.required_laws, menu.entries[0].required_laws);
        // A direct certificate admits no covariates (validate_trial_aipw refuses any), so
        // its membership and randomization laws are marginal and the graph condition says so.
        assert!(
            direct_entry.required_laws[1].contains("marginal")
                && direct_entry.required_laws[2]
                    == "known randomization probabilities P(A=1 | S=1)",
            "{:?}",
            direct_entry.required_laws
        );
        assert!(
            direct_entry.required_graph_conditions.iter().any(|c| c.contains("admits none")),
            "{:?}",
            direct_entry.required_graph_conditions
        );
        assert!(
            !direct_entry.required_graph_conditions.iter().any(|c| c.contains("standardizers"))
        );
        let direct_ipw = &direct_menu.entries[1];
        assert_eq!(
            direct_ipw.required_laws[0],
            format!("supplied {}", direct_entry.required_laws[1])
        );
        assert_eq!(direct_ipw.required_laws.len(), 2);
    }
    // A graph whose selection acts on the outcome certifies no estimator.
    let mut graph = antecedent_graph::Admg::with_variables(3);
    for (from, to) in [(0, 2), (1, 2)] {
        graph
            .insert_directed(
                antecedent_graph::DenseNodeId::from_raw(from),
                antecedent_graph::DenseNodeId::from_raw(to),
            )
            .unwrap();
    }
    let blocked = antecedent_graph::SelectionDiagram::try_new(
        graph,
        vec![antecedent_core::VariableId::from_raw(2)],
    )
    .unwrap();
    let menu = StudyBuilder::learned_continuous_menu(&blocked, &query, None).unwrap();
    for name in ["learned_trial_aipw", "trial_ipw_supplied_probabilities"] {
        let entry = menu.entries.iter().find(|e| e.estimator == name).unwrap();
        assert!(!entry.eligible, "{name}");
        assert_eq!(entry.refusal.as_ref().unwrap().code, "transport_not_certified");
        // An uncertified graph still names what each estimator would require.
        assert!(entry.required_laws.iter().all(|l| l.starts_with("not derivable")), "{name}");
        assert!(!entry.required_laws.is_empty() && !entry.required_graph_conditions.is_empty());
    }
    // An unsupported learner refuses only the learned entry.
    let bad = LearnerSpec::Ridge(antecedent_estimate::RidgeSpec { lambda: -1.0 });
    let menu = StudyBuilder::learned_continuous_menu(
        &diagram,
        &query,
        Some((bad, LearnerSpec::Logistic(antecedent_estimate::LogisticSpec::default()))),
    )
    .unwrap();
    assert!(!menu.entries[0].eligible && menu.entries[1].eligible);
    // The prepared study exposes the same menu without fitting.
    let prepared = prepare(Design::NestedCohort, Scenario::Good, (400, 300), options(), 1).unwrap();
    assert!(prepared.estimator_menu().entries[0].eligible);
}

#[test]
fn thread_budget_changes_leave_seeded_results_identical() {
    // Both designs, the point and the whole-estimator bootstrap replicates.
    let (diagram, query, _) = graph();
    for design in DESIGNS {
        let input = draw(design, Scenario::Good, 160, 100, 21);
        let id =
            antecedent_identify::TransportIdentifier::new().identify(&diagram, &query).unwrap();
        let options = LearnedContinuousOptions { bootstrap: 199, ..options() };
        let serial = ExecutionContext::for_tests(9);
        let mut parallel = ExecutionContext::for_tests(9);
        parallel.parallelism = Parallelism::bounded(NonZeroThreadCount::new(4).unwrap());
        let a = learned_continuous_interval_internal(&id, &input, &options, &serial).unwrap();
        let b = learned_continuous_interval_internal(&id, &input, &options, &parallel).unwrap();
        assert_eq!(a.replicates, b.replicates);
        assert_eq!(a.failures, b.failures);
        assert_eq!(a.interval, b.interval);
        assert_eq!(a.estimate.to_bits(), b.estimate.to_bits());
        assert!(a.replicates.len() > 150 && a.interval.is_some());
        // The public point route is invariant too.
        let point = |ctx: &ExecutionContext| {
            antecedent_estimate::estimate_learned_continuous(&id, &input, &options, ctx).unwrap()
        };
        assert_eq!(point(&serial), point(&parallel));
    }
}

/// Fit one nuisance on `train` rows of `input` alone (intercept plus the covariates, the
/// only design `fit_point` builds) and predict at every row of `at`.
fn refit_and_predict(
    input: &antecedent_estimate::TrialAipwInput,
    train: &[usize],
    target: &[f64],
    spec: LearnerSpec,
    task: antecedent_learn::PredictionTask,
    at: &antecedent_estimate::TrialAipwInput,
    rows: &[usize],
) -> Vec<f64> {
    use antecedent_learn::{DesignView, TargetView, resolve_for};
    let design = |source: &antecedent_estimate::TrialAipwInput, idx: &[usize]| {
        let mut d = vec![1.0; idx.len()];
        for col in &source.covariates {
            d.extend(idx.iter().map(|i| col[*i]));
        }
        d
    };
    let ctx = ExecutionContext::for_tests(0);
    let cols = input.features.len() + 1;
    let train_x = design(input, train);
    let train_y: Vec<f64> = train.iter().map(|i| target[*i]).collect();
    let factory = resolve_for(spec, task).unwrap();
    let fitted = factory
        .fit(
            DesignView::from_column_major(&train_x, train.len(), cols).unwrap(),
            TargetView::new(&train_y),
            None,
            &ctx,
        )
        .unwrap();
    let test_x = design(at, rows);
    let mut out = vec![0.0; rows.len()];
    fitted
        .predict(DesignView::from_column_major(&test_x, rows.len(), cols).unwrap(), &mut out, &ctx)
        .unwrap();
    out
}

/// Every out-of-fold prediction of `estimate` at fold `k` equals the prediction of a model
/// fitted on the rows outside fold `k` of `trained` alone, evaluated at the covariates
/// of `at` (the same rows, possibly with their own values edited).
fn assert_held_in_only(
    estimate: (&[f64], &[f64], &[f64]),
    trained: &antecedent_estimate::TrialAipwInput,
    at: &antecedent_estimate::TrialAipwInput,
    folds: &[u16],
    k: u16,
    options: &LearnedContinuousOptions,
) {
    use antecedent_learn::PredictionTask::{BinaryProbability, Regression};
    let n = trained.source.len();
    let held_out: Vec<usize> = (0..n).filter(|i| folds[*i] == k).collect();
    let outside = |eligible: &dyn Fn(usize) -> bool| -> Vec<usize> {
        (0..n).filter(|i| folds[*i] != k && eligible(*i)).collect()
    };
    let source: Vec<f64> = trained.source.iter().map(|s| f64::from(*s)).collect();
    let membership = refit_and_predict(
        trained,
        &outside(&|_| true),
        &source,
        options.membership,
        BinaryProbability,
        at,
        &held_out,
    );
    for (row, expected) in held_out.iter().zip(&membership) {
        assert!((estimate.0[*row] - expected).abs() < 1e-9, "membership row {row}");
    }
    for arm in [false, true] {
        let mu = refit_and_predict(
            trained,
            &outside(&|i| trained.source[i] && trained.treatment[i] == arm),
            &trained.outcome,
            options.outcome,
            Regression,
            at,
            &held_out,
        );
        let stored = if arm { estimate.2 } else { estimate.1 };
        for (row, expected) in held_out.iter().zip(&mu) {
            assert!((stored[*row] - expected).abs() < 1e-9, "mu{} row {row}", u8::from(arm));
        }
    }
}

fn triple(e: &antecedent_estimate::LearnedContinuousEstimate) -> (&[f64], &[f64], &[f64]) {
    (&e.membership, &e.mu0, &e.mu1)
}

/// What this proves: every stored nuisance prediction for a fold is the prediction of a
/// model fitted on the other folds alone, and editing that fold's own outcomes, covariates
/// or membership labels cannot reach it (it is recomputed from the held-in side).
///
/// What it does not prove: `fit_point` performs no preprocessing beyond prepending an
/// intercept column, so "preprocessing inside folds" holds trivially today. Any future
/// full-sample transform (a scale or a mean over all rows) would move the held-out
/// predictions off the reference refit below and fail this test for learners that are
/// not invariant to it (ridge and logistic are not scale-invariant).
#[test]
fn every_nuisance_prediction_is_out_of_fold_and_fold_edits_never_reach_it() {
    let (diagram, query, _) = graph();
    let id = antecedent_identify::TransportIdentifier::new().identify(&diagram, &query).unwrap();
    let options = options();
    let input = draw(Design::IndependentSamples, Scenario::Good, 300, 200, 4);
    let ctx = ExecutionContext::for_tests(4);
    let base =
        antecedent_estimate::estimate_learned_continuous(&id, &input, &options, &ctx).unwrap();
    let folds = base.folds.assignment.clone();
    // One fold assignment is shared by every nuisance and recorded with the estimate.
    assert_eq!(folds, antecedent_estimate::learned_trial::trial_fold_assignment(&input, 3));
    for k in 0..3u16 {
        assert_held_in_only(triple(&base), &input, &input, &folds, k, &options);
    }
    let fold0: Vec<usize> = (0..input.source.len()).filter(|i| folds[*i] == 0).collect();
    assert!(!fold0.is_empty());

    // Edit everything about fold 0's own rows that a leak could read: outcomes and
    // covariates (a mean or scale over all rows would move with them). The fold
    // assignment never depends on outcomes or covariates, so it is unchanged.
    let mut edited = input.clone();
    for &i in &fold0 {
        if edited.source[i] {
            edited.outcome[i] += 100.0;
        }
        edited.covariates[0][i] = 7.0 + 3.0 * edited.covariates[0][i];
    }
    assert_eq!(antecedent_estimate::learned_trial::trial_fold_assignment(&edited, 3), folds);
    // The overlap gate would refuse the shifted covariates; the ungated point fit is what
    // holds the out-of-fold identity.
    let moved =
        antecedent_estimate::estimate_trial_aipw(&id, &edited, &options.trial_options(0), &ctx)
            .unwrap();
    // Fold 0's predictions still come from the model fitted on the untouched held-in rows
    // (the base rows outside fold 0), now evaluated at the edited covariates.
    assert_held_in_only(
        (&moved.membership, &moved.mu0, &moved.mu1),
        &input,
        &edited,
        &folds,
        0,
        &options,
    );
    // The other folds train on the edited rows, so their predictions do move.
    assert!(
        (0..input.source.len())
            .any(|i| folds[i] != 0 && base.mu1[i].to_bits() != moved.mu1[i].to_bits())
    );

    // Perturb membership labels in fold 0: trial arm-0 rows of fold 0 become target rows.
    // Labels define the fold strata, so the assignment is recomputed for the edited data;
    // the out-of-fold identity must hold under that assignment.
    let mut relabelled = input.clone();
    let flipped: Vec<usize> = fold0
        .iter()
        .copied()
        .filter(|i| input.source[*i] && !input.treatment[*i])
        .take(15)
        .collect();
    assert_eq!(flipped.len(), 15);
    for &i in &flipped {
        relabelled.source[i] = false;
    }
    let relabelled_folds =
        antecedent_estimate::learned_trial::trial_fold_assignment(&relabelled, 3);
    assert_ne!(relabelled_folds, folds, "labels are what the fold strata are built from");
    let after =
        antecedent_estimate::estimate_learned_continuous(&id, &relabelled, &options, &ctx).unwrap();
    for k in 0..3u16 {
        assert_held_in_only(
            triple(&after),
            &relabelled,
            &relabelled,
            &relabelled_folds,
            k,
            &options,
        );
    }
}

#[test]
fn estimate_reuses_the_prepared_certificate_and_refresh_keeps_it() {
    let prepared =
        prepare(Design::IndependentSamples, Scenario::Good, (500, 300), options(), 7).unwrap();
    let ctx = ExecutionContext::for_tests(7);
    let first = prepared.estimate(&ctx).unwrap();
    assert_eq!(first.identity(), prepared.estimate(&ctx).unwrap().identity());
    let fresh = draw(Design::IndependentSamples, Scenario::Good, 500, 300, 8);
    let refreshed = prepared.refresh(fresh.clone(), &ctx).unwrap();
    assert_eq!(refreshed.identification(), prepared.identification());
    assert_ne!(refreshed.estimate(&ctx).unwrap().identity(), first.identity());
    // A changed sampling design or schema needs a new preparation.
    let mut nested = fresh.clone();
    nested.sampling = antecedent_estimate::TrialSampling::NestedCohort;
    assert!(prepared.refresh(nested, &ctx).is_err());
    let mut bad = fresh;
    bad.randomization.fill(0.0);
    assert!(prepared.refresh(bad, &ctx).is_err());
}

fn exported(design: Design) -> (Vec<u8>, LearnedContinuousArtifactWire) {
    let prepared = prepare(design, Scenario::Good, (400, 300), options(), 13).unwrap();
    let bytes = prepared.estimate(&ExecutionContext::for_tests(13)).unwrap().export().unwrap();
    let wire = LearnedContinuousArtifactWire::decode(&bytes).unwrap();
    (bytes, wire)
}

fn refused(bytes: &[u8]) -> Refusal {
    match consume_learned_continuous_artifact(bytes, LearnedContinuousConsumeLimits::default()) {
        Err(IoError::LearnedContinuous(kind)) => kind,
        other => {
            panic!("expected a typed refusal, got {:?}", other.map(|r| r.identity().to_owned()))
        }
    }
}

#[test]
fn the_exported_point_is_replayed_by_an_independent_consumer() {
    for design in DESIGNS {
        let (bytes, wire) = exported(design);
        let consumed =
            consume_learned_continuous_artifact(&bytes, LearnedContinuousConsumeLimits::default())
                .unwrap();
        assert_eq!(consumed.estimate(), &wire.result);
        assert_eq!(consumed.identity(), wire.identity().unwrap());
        // The artifact carries the certificate, provider provenance, folds and the point.
        assert_eq!(wire.certificate.formula, "standardize");
        assert_eq!(wire.result.provenance.len(), 9);
        assert_eq!(wire.result.folds.scheme, "stratified_round_robin_source_arm_and_target");
        assert_eq!(wire.variable_names, ["z", "a", "y"]);
        assert_eq!(wire.result.uncertainty.status, "point_only");
        // Consumption re-exports to the same bytes without any fitting.
        assert_eq!(consumed.export().unwrap(), bytes);
    }
}

fn mutate(
    wire: &LearnedContinuousArtifactWire,
    edit: impl Fn(&mut LearnedContinuousArtifactWire),
    rebind: bool,
) -> Vec<u8> {
    let mut wire = wire.clone();
    edit(&mut wire);
    if rebind {
        wire.premises_digest = wire.expected_premises_digest().unwrap();
        wire.data_digest = wire.expected_data_digest().unwrap();
        wire.evidence_digest = wire.expected_evidence_digest().unwrap();
    }
    wire.export().unwrap()
}

#[test]
#[allow(clippy::too_many_lines, reason = "one scenario per assertion group")]
fn a_mutated_artifact_fails_independent_consumption_with_a_typed_error() {
    let (bytes, original) = exported(Design::NestedCohort);
    // Any edit of the stored result without re-digesting fails the evidence digest.
    let point = mutate(&original, |w| w.result.estimate += 1e-9, false);
    assert_eq!(refused(&point), Refusal::EvidenceMismatch);
    let unbound_provenance =
        mutate(&original, |w| w.result.provenance[0].version = "9.9".into(), false);
    assert_eq!(refused(&unbound_provenance), Refusal::EvidenceMismatch);
    let unbound_folds = mutate(&original, |w| w.result.folds.assignment[0] += 1, false);
    assert_eq!(refused(&unbound_folds), Refusal::EvidenceMismatch);
    // With the digests rewritten consistently, the replayed checks are what fail.
    let point = mutate(&original, |w| w.result.estimate += 1e-9, true);
    assert_eq!(refused(&point), Refusal::PointMismatch);
    // A treated trial row enters the score through its weight and its arm's regression.
    let row = (0..original.input.source.len())
        .find(|i| original.input.source[*i] && original.input.treatment[*i])
        .unwrap();
    let membership = mutate(&original, |w| w.result.membership[row] *= 0.99, true);
    assert_eq!(refused(&membership), Refusal::PointMismatch);
    let mu = mutate(&original, |w| w.result.mu1[row] += 0.5, true);
    assert_eq!(refused(&mu), Refusal::PointMismatch);
    let diagnostics = mutate(&original, |w| w.result.diagnostics.membership_logloss += 1e-6, true);
    assert_eq!(refused(&diagnostics), Refusal::DiagnosticsMismatch);
    let overlap = mutate(&original, |w| w.result.overlap.selection.probability_min += 1e-6, true);
    assert_eq!(refused(&overlap), Refusal::OverlapMismatch);
    let folds = mutate(
        &original,
        |w| {
            let other = w.result.folds.assignment.iter().position(|f| *f != 0).unwrap();
            w.result.folds.assignment.swap(0, other);
        },
        true,
    );
    assert_eq!(refused(&folds), Refusal::FoldMismatch);
    let status =
        mutate(&original, |w| w.result.uncertainty.reason = "no_interval_requested_x".into(), true);
    assert!(matches!(refused(&status), Refusal::UncertaintyMismatch(_)));
    // Provenance is bound and checked against the learner specs.
    let dropped = mutate(
        &original,
        |w| {
            w.result.provenance.pop();
        },
        true,
    );
    assert_eq!(refused(&dropped), Refusal::ProvenanceMismatch);
    let other_learner = mutate(&original, |w| w.result.provenance[0].spec = "ridge".into(), true);
    assert_eq!(refused(&other_learner), Refusal::ProvenanceMismatch);
    let mixed_role = mutate(&original, |w| w.result.provenance[1].version = "0.0".into(), true);
    assert_eq!(refused(&mixed_role), Refusal::ProvenanceMismatch);
    // What replay cannot detect: a producer-recorded provider version that is
    // self-consistent within its role and re-digested (the consumer never refits), and
    // nuisance predictions edited together with the point, diagnostics and overlap they
    // imply. This documents the limit rather than asserting a refusal.
    let renamed_version = mutate(
        &original,
        |w| {
            for p in &mut w.result.provenance[..3] {
                p.version = "recorded-elsewhere".into();
            }
        },
        true,
    );
    assert!(
        consume_learned_continuous_artifact(
            &renamed_version,
            LearnedContinuousConsumeLimits::default()
        )
        .is_ok(),
        "the recorded provider version is producer evidence, not replayed"
    );
    // The mutation that shifts a single held-out prediction is not consistent with the
    // stored point, so the score replay catches it even when re-digested (above); only a
    // full consistent rewrite (not constructed here) would pass without a refit.
    // Premises: learner specs, folds, thresholds, seed, limits' inputs and names.
    let learner = mutate(
        &original,
        |w| w.options.outcome = LearnerSpec::Ridge(antecedent_estimate::RidgeSpec::default()),
        false,
    );
    assert_eq!(refused(&learner), Refusal::PremisesMismatch);
    let threshold = mutate(&original, |w| w.options.min_membership_probability = 0.01, false);
    assert_eq!(refused(&threshold), Refusal::PremisesMismatch);
    let seed = mutate(&original, |w| w.seed += 1, false);
    assert_eq!(refused(&seed), Refusal::PremisesMismatch);
    let names = mutate(&original, |w| w.variable_names.swap(0, 2), false);
    assert_eq!(refused(&names), Refusal::PremisesMismatch);
    let query = mutate(&original, |w| w.query.source_population = "elsewhere".into(), false);
    assert_eq!(refused(&query), Refusal::PremisesMismatch);
    assert_eq!(original.check_variable_names(&original.variable_names.clone()), Ok(()));
    let mut relabelled = original.variable_names.clone();
    relabelled.swap(0, 2);
    assert_eq!(original.check_variable_names(&relabelled), Err(Refusal::NamesMismatch));
    // Data identity: rows are bound by the data digest.
    let rows = mutate(&original, |w| w.input.outcome[row] += 1.0, false);
    assert_eq!(refused(&rows), Refusal::DataIdentityMismatch);
    // Consistent rewrites of both digests must still reproduce the checks.
    let forged_rows = mutate(&original, |w| w.input.outcome[row] += 1.0, true);
    assert_eq!(refused(&forged_rows), Refusal::PointMismatch);
    let forged_folds = mutate(&original, |w| w.options.folds = 4, true);
    assert_eq!(refused(&forged_folds), Refusal::FoldMismatch);
    let forged_threshold = mutate(&original, |w| w.options.min_membership_probability = 0.49, true);
    assert_eq!(refused(&forged_threshold), Refusal::OverlapRefused);
    let forged_bound = mutate(&original, |w| w.options.folds = 21, true);
    assert!(matches!(refused(&forged_bound), Refusal::LimitsExceeded(_)));
    let forged_certificate = mutate(&original, |w| w.certificate.rule = "forged".into(), true);
    assert!(matches!(refused(&forged_certificate), Refusal::ProofMismatch(_)));
    let forged_bootstrap = mutate(&original, |w| w.options.bootstrap = 199, true);
    assert!(matches!(refused(&forged_bootstrap), Refusal::UncertaintyMismatch(_)));
    let features = mutate(&original, |w| w.required_features.push("interval".into()), false);
    assert!(matches!(refused(&features), Refusal::UnsupportedSemantics(_)));
    let limits = LearnedContinuousConsumeLimits { max_rows: 10, ..Default::default() };
    assert!(matches!(
        consume_learned_continuous_artifact(&bytes, limits),
        Err(IoError::LearnedContinuous(Refusal::LimitsExceeded(_)))
    ));
    // An artifact that carries an interval fails to decode: the format has no such field.
    let value: ciborium::Value = ciborium::from_reader(bytes.as_slice()).unwrap();
    let ciborium::Value::Map(mut entries) = value else { panic!("a map") };
    entries.push((ciborium::Value::Text("interval".into()), ciborium::Value::Array(vec![])));
    let mut with_interval = Vec::new();
    ciborium::into_writer(&ciborium::Value::Map(entries), &mut with_interval).unwrap();
    assert!(
        consume_learned_continuous_artifact(
            &with_interval,
            LearnedContinuousConsumeLimits::default()
        )
        .is_err()
    );
    // A different version is refused before its payload is interpreted.
    let version = mutate(&original, |w| w.version = 3, false);
    assert!(matches!(
        consume_learned_continuous_artifact(&version, LearnedContinuousConsumeLimits::default()),
        Err(IoError::UnsupportedVersion { version: 3 })
    ));
}

#[test]
fn the_analytic_interval_replays_and_the_legacy_bootstrap_refuses() {
    let prepared =
        prepare(Design::IndependentSamples, Scenario::Good, (400, 300), options(), 2).unwrap();
    let ctx = ExecutionContext::for_tests(2);
    let interval = prepared.interval(&ctx).unwrap();
    assert_eq!(interval.wire().version, 2);
    assert!(interval.estimate().uncertainty.available());
    let (low, high) = interval.estimate().interval.unwrap();
    assert!(low < interval.estimate().estimate && interval.estimate().estimate < high);
    let replay = consume_learned_continuous_artifact(
        &interval.export().unwrap(),
        LearnedContinuousConsumeLimits::default(),
    )
    .unwrap();
    assert_eq!(replay.estimate().interval, interval.estimate().interval);
    let forged = mutate(
        interval.wire(),
        |w| {
            w.result.interval.as_mut().unwrap().0 -= 0.01;
        },
        true,
    );
    assert!(matches!(
        consume_learned_continuous_artifact(&forged, LearnedContinuousConsumeLimits::default()),
        Err(IoError::LearnedContinuous(Refusal::IntervalMismatch))
    ));
    // The old percentile-bootstrap request remains withheld.
    let requested = LearnedContinuousOptions { bootstrap: 199, ..options() };
    let result = prepare(Design::IndependentSamples, Scenario::Good, (400, 300), requested, 2)
        .unwrap()
        .estimate(&ctx)
        .unwrap();
    let uncertainty = &result.estimate().uncertainty;
    assert_eq!(
        (uncertainty.status.as_str(), uncertainty.reason.as_str()),
        ("withheld", "cell_not_licensed")
    );
    assert!(!uncertainty.available() && result.estimate().estimate.is_finite());
    let (code, message) = refused_code(
        prepare(Design::IndependentSamples, Scenario::Good, (400, 300), requested, 2)
            .unwrap()
            .interval(&ctx)
            .unwrap_err(),
    );
    assert_eq!(code, "estimator_inference_mismatch");
    assert!(message.starts_with("learned_transport.bootstrap_not_licensed"));
}

#[test]
fn the_estimator_layer_returns_the_point_with_recorded_diagnostics_and_refuses_directly() {
    let (diagram, query, _) = graph();
    let id = antecedent_identify::TransportIdentifier::new().identify(&diagram, &query).unwrap();
    for design in DESIGNS {
        let input = draw(design, Scenario::Good, 1000, 600, 21);
        let ctx = ExecutionContext::for_tests(21);
        let direct =
            antecedent_estimate::estimate_learned_continuous(&id, &input, &options(), &ctx)
                .unwrap();
        assert!((direct.estimate - Scenario::Good.truth()).abs() < 0.3, "{design:?}");
        // Recorded evidence is per row, per fold and per fitted nuisance.
        let rows = input.source.len();
        assert_eq!(
            (direct.membership.len(), direct.mu0.len(), direct.mu1.len()),
            (rows, rows, rows)
        );
        assert_eq!(direct.folds.count, 3);
        assert_eq!(
            direct.folds.assignment,
            antecedent_estimate::learned_trial::trial_fold_assignment(&input, 3)
        );
        assert_eq!(direct.provenance.len(), 9);
        assert!(direct.membership.iter().all(|p| *p > 0.0 && *p < 1.0));
        assert!(direct.diagnostics.membership_logloss.is_finite());
        assert_eq!(direct.uncertainty.status, "point_only");
        // A requested interval is reported as withheld and never attached.
        let with_bootstrap = LearnedContinuousOptions { bootstrap: 300, ..options() };
        let withheld =
            antecedent_estimate::estimate_learned_continuous(&id, &input, &with_bootstrap, &ctx)
                .unwrap();
        assert!(!withheld.uncertainty.available());
        assert_ne!(withheld.uncertainty.status, "point_only");
        assert_eq!(withheld.estimate.to_bits(), direct.estimate.to_bits());
        // The seeded estimator layer and the prepared facade agree bit for bit.
        let prepared = prepare(design, Scenario::Good, (1000, 600), options(), 21).unwrap();
        let facade = prepared.estimate(&ctx).unwrap();
        assert_eq!(facade.estimate().estimate.to_bits(), direct.estimate.to_bits());
    }
    // The layer validates and refuses on its own, without the facade.
    let input = draw(Design::IndependentSamples, Scenario::WeakOverlap, 1000, 600, 3);
    let error = antecedent_estimate::estimate_learned_continuous(
        &id,
        &input,
        &options(),
        &ExecutionContext::for_tests(3),
    )
    .unwrap_err();
    assert!(
        matches!(&error, EstimationError::Refused { code: "transport_support_failure", message } if message.starts_with("learned_transport.membership_overlap")),
        "{error:?}"
    );
    let input = draw(Design::NestedCohort, Scenario::Good, 400, 300, 1);
    let too_many = LearnedContinuousOptions { folds: 21, ..options() };
    let error = antecedent_estimate::estimate_learned_continuous(
        &id,
        &input,
        &too_many,
        &ExecutionContext::for_tests(1),
    )
    .unwrap_err();
    assert!(
        matches!(&error, EstimationError::Refused { code: "route_not_supported", message } if message.starts_with("learned_transport.bounds_exceeded")),
        "{error:?}"
    );
}

#[test]
fn the_free_menu_function_follows_its_learner_argument_without_the_builder() {
    let (diagram, query, _) = graph();
    let id = antecedent_identify::TransportIdentifier::new().identify(&diagram, &query).unwrap();
    let defaulted = antecedent_estimate::transport_estimator_menu(&id, &query, None);
    let learned = &defaulted.entries[0];
    assert_eq!(learned.estimator, "learned_trial_aipw");
    assert!(learned.eligible && learned.refusal.is_none());
    assert!(learned.nuisance_tasks.iter().all(|t| t.contains("default: none requested")));
    // The same certificate under explicit learners lists them and drops the default marker.
    let learners = Some((
        LearnerSpec::Linear(LinearSpec::default()),
        LearnerSpec::Logistic(antecedent_estimate::LogisticSpec::default()),
    ));
    let explicit = antecedent_estimate::transport_estimator_menu(&id, &query, learners);
    let named = &explicit.entries[0];
    assert!(named.nuisance_tasks.iter().any(|t| t.contains("learner linear")));
    assert!(named.nuisance_tasks.iter().any(|t| t.contains("learner logistic")));
    assert!(named.nuisance_tasks.iter().all(|t| !t.contains("default")));
    assert_ne!(learned.nuisance_tasks, named.nuisance_tasks);
    // The free function is a pure function of its inputs: it fits nothing and repeats.
    assert_eq!(
        serde_json::to_string(&explicit).unwrap(),
        serde_json::to_string(&antecedent_estimate::transport_estimator_menu(
            &id, &query, learners
        ))
        .unwrap()
    );
    // Entry set and manual selection do not depend on the learners.
    let names = |m: &antecedent_estimate::EstimatorMenu| {
        m.entries.iter().map(|e| e.estimator.clone()).collect::<Vec<_>>()
    };
    assert_eq!(names(&defaulted), names(&explicit));
    assert_eq!(explicit.selection, "manual");
    // The builder constructor is a thin wrapper over the free function: same menu.
    let via_builder = StudyBuilder::learned_continuous_menu(&diagram, &query, learners).unwrap();
    assert_eq!(
        serde_json::to_string(&via_builder).unwrap(),
        serde_json::to_string(&explicit).unwrap()
    );
}

#[test]
fn the_io_wire_consumes_bytes_directly_under_its_own_limits() {
    let (bytes, original) = exported(Design::IndependentSamples);
    let limits = LearnedContinuousConsumeLimits::default();
    // Bytes to verified replay through the io type, no facade.
    let consumed = LearnedContinuousArtifactWire::consume(&bytes, limits).unwrap();
    assert_eq!(consumed.wire.result, original.result);
    assert_eq!(consumed.wire.identity().unwrap(), original.identity().unwrap());
    let (diagram, query, _) = graph();
    let id = antecedent_identify::TransportIdentifier::new().identify(&diagram, &query).unwrap();
    assert_eq!(consumed.identification, id);
    let coded = |bytes: &[u8], limits| match LearnedContinuousArtifactWire::consume(bytes, limits) {
        Err(IoError::LearnedContinuous(kind)) => kind,
        Err(other) => panic!("expected a typed refusal, got {other}"),
        Ok(_) => panic!("expected a refusal"),
    };
    // The consumer's own bounds bind, whatever the artifact stores.
    let rows = original.input.source.len();
    assert_eq!(
        coded(&bytes, LearnedContinuousConsumeLimits { max_rows: rows - 1, ..limits }),
        Refusal::LimitsExceeded("row count")
    );
    assert_eq!(
        coded(&bytes, LearnedContinuousConsumeLimits { max_features: 0, ..limits }),
        Refusal::LimitsExceeded("feature count")
    );
    assert!(
        LearnedContinuousArtifactWire::consume(
            &bytes,
            LearnedContinuousConsumeLimits { max_rows: rows, ..limits }
        )
        .is_ok()
    );
    // A foreign feature marker is refused before any replay.
    let foreign = mutate(&original, |w| w.required_features = vec!["other_v9".into()], false);
    assert!(matches!(coded(&foreign, limits), Refusal::UnsupportedSemantics(_)));
    // A future version fails at decode, and undecodable bytes are not a typed artifact refusal.
    let future = mutate(&original, |w| w.version = 3, false);
    assert!(matches!(
        LearnedContinuousArtifactWire::consume(&future, limits),
        Err(IoError::UnsupportedVersion { .. })
    ));
    assert!(!matches!(
        LearnedContinuousArtifactWire::consume(&bytes[..bytes.len() / 2], limits),
        Err(IoError::LearnedContinuous(_))
    ));
    // Replayed evidence is checked at this layer: an unbound point edit fails the digest.
    let point = mutate(&original, |w| w.result.estimate += 1e-9, false);
    assert_eq!(coded(&point, limits), Refusal::EvidenceMismatch);
}

/// Source identity re-sealed: a relabelled source population is a different
/// premise set. The rows carry no population name, so replay cannot contradict
/// the label itself; the consumed identity differs from the producer's, and a
/// consumer naming the populations it expects refuses it.
#[test]
fn a_resealed_source_relabel_changes_the_consumed_identity() {
    let (bytes, original) = exported(Design::NestedCohort);
    let consume = |bytes: &[u8]| {
        consume_learned_continuous_artifact(bytes, LearnedContinuousConsumeLimits::default())
    };
    let live = consume(&bytes).unwrap();
    let relabelled = mutate(&original, |w| w.query.source_population = "elsewhere".into(), true);
    match consume(&relabelled) {
        Ok(forged) => assert_ne!(forged.identity(), live.identity()),
        Err(error) => panic!("unexpected: {error:?}"),
    }
    // A consumer that names the populations it expects refuses the relabel.
    let (source, target) =
        (original.query.source_population.clone(), original.query.target_population.clone());
    original.check_population_names(&source, &target).unwrap();
    let forged = LearnedContinuousArtifactWire::decode(&relabelled).unwrap();
    assert_eq!(forged.check_population_names(&source, &target), Err(Refusal::NamesMismatch));
}

/// Cross-format: a 2.2 learned-continuous artifact is never read as a 2.1
/// learned-trial artifact or as a smoothed-dose grid, nor the reverse.
#[test]
fn a_learned_continuous_artifact_is_refused_by_other_trial_consumers() {
    let (bytes, _) = exported(Design::NestedCohort);
    assert!(antecedent::LearnedTrialResult::consume(&bytes).is_err());
    assert!(
        antecedent::consume_smoothed_dose_artifact(
            &bytes,
            antecedent_io::smoothed_dose_artifact::SmoothedDoseConsumeLimits::default(),
            &ExecutionContext::for_tests(0)
        )
        .is_err()
    );
}
