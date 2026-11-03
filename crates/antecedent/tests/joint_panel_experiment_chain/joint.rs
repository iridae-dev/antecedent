// SPDX-License-Identifier: MIT OR Apache-2.0
//! Joint conditioning of one retained panel posterior on several mapped experimental components:
//! an independent reference for the joint likelihood, the consequence of retaining dependence,
//! refusals, current authority, reuse and invalidation, ranking, artifacts, assumption-bounded
//! evidence, and the retained-experiment adapter over two kinds of compiled evidence.
use super::matrix::translated_fixture;
use super::*;
use antecedent::decision_evidence::{
    CompiledJointExcursion, CompiledMatchedPairDesign, Exposure, JointEvidenceComponent,
    JointExcursionDesign, MatchedPairDesign, OutcomeComponent, RetainedExperiment,
    compile_and_evidence_from_joint_excursion, compile_and_evidence_from_matched_pairs,
    compile_joint_excursion, compile_matched_pair_design,
    condition_panel_jointly_on_retained_evidence,
};

fn functionals(original: &JointPanelResult) -> [JointPanelResponseFunctional; 2] {
    let horizon = original.request().worlds[0].program.horizon();
    [
        JointPanelResponseFunctional::TerminalMean,
        JointPanelResponseFunctional::WindowAverage { from: horizon - 3, until: horizon - 1 },
    ]
}

/// Values each fixture's paired mapping predicts the experiment would observe, per draw slot.
fn predicted(original: &JointPanelResult, fixture: &ExperimentFixture) -> Vec<f64> {
    let law = original.numerical_project_law_source(&fixture.update.functional).unwrap();
    let width = fixture.coefficients.len();
    law.draws()
        .chunks_exact(width)
        .enumerate()
        .map(|(slot, row)| {
            let state = row.iter().zip(&fixture.coefficients).map(|(v, c)| v * c).sum::<f64>();
            (state - fixture.update.mapping.intercepts[slot]) / fixture.update.mapping.slopes[slot]
        })
        .collect()
}

fn covariance(sd: [f64; 2], rho: f64) -> Vec<f64> {
    vec![sd[0] * sd[0], rho * sd[0] * sd[1], rho * sd[0] * sd[1], sd[1] * sd[1]]
}

fn total_variation(a: &[f64], b: &[f64]) -> f64 {
    0.5 * a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f64>()
}

fn sum_of_components() -> UtilityExpr {
    UtilityExpr::sum(UtilityExpr::Input(0), UtilityExpr::Input(1))
}

#[allow(
    clippy::too_many_lines,
    reason = "the bounded reference fixture preserves its exact calculations and local structure"
)]
pub fn verify_joint_conditioning(
    prepared: &PreparedStudy<JointPanelState>,
    original: &JointPanelResult,
    ctx: &ExecutionContext,
) {
    let fixtures: Vec<_> = functionals(original)
        .into_iter()
        .map(|f| translated_fixture(original, f, true, 0))
        .collect();
    let ledger = fixtures[0].ledger.clone();
    let evidence = fixtures[0].update.evidence.clone();
    let predictions: Vec<_> = fixtures.iter().map(|f| predicted(original, f)).collect();
    let observed: Vec<f64> =
        predictions.iter().map(|p| p.iter().sum::<f64>() / p.len() as f64).collect();
    let sd: Vec<f64> = predictions
        .iter()
        .map(|p| {
            let high = p.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let low = p.iter().copied().fold(f64::INFINITY, f64::min);
            (high - low) / 6.0
        })
        .collect();
    let sd = [sd[0], sd[1]];
    assert!(sd[0] > 0.0 && sd[1] > 0.0, "fixture must retain uncertainty in both components");
    let components: Vec<_> = fixtures
        .iter()
        .enumerate()
        .map(|(i, f)| {
            // Two measured outcomes: distinct experimental quantities, one experiment.
            let mut bridge = f.update.bridge.clone();
            bridge.source.quantity.variable_name = format!("outcome_{i}");
            bridge.check_support().unwrap();
            PanelJointExperimentComponent {
                bridge,
                functional: f.update.functional.clone(),
                mapping: f.update.mapping.clone(),
            }
        })
        .collect();
    let update_with = |sampling_covariance: Vec<f64>| PanelJointExperimentUpdate {
        components: components.clone(),
        observed_means: observed.clone(),
        sampling_covariance,
        sample_size: 100,
        modular_policy: PanelExperimentModularPolicy::FrozenJointImportance,
        evidence: evidence.clone(),
        contrast_bias_bounds: None,
    };
    let joint = original
        .condition_joint_experiment(update_with(covariance(sd, 0.8)), &ledger, ctx)
        .unwrap();
    let independent = original
        .condition_joint_experiment(update_with(covariance(sd, 0.0)), &ledger, ctx)
        .unwrap();

    // Independent arithmetic for the bivariate Gaussian likelihood on the retained draws.
    let prior = original.numerical_project_law_source(&fixtures[0].update.functional).unwrap();
    let count = prior.n_draws();
    let prior_weights =
        prior.metadata().weights.clone().unwrap_or_else(|| vec![1.0 / count as f64; count]);
    let cov = covariance(sd, 0.8);
    let (v1, c, v2) = (cov[0], cov[1], cov[3]);
    let determinant = v1 * v2 - c * c;
    let log_weights = (0..count)
        .map(|slot| {
            let (r1, r2) = (observed[0] - predictions[0][slot], observed[1] - predictions[1][slot]);
            prior_weights[slot].ln()
                - 0.5 * (v2 * r1 * r1 - 2.0 * c * r1 * r2 + v1 * r2 * r2) / determinant
        })
        .collect::<Vec<_>>();
    let maximum = log_weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mass = log_weights.iter().map(|w| (w - maximum).exp()).sum::<f64>();
    let reference = log_weights.iter().map(|w| (w - maximum).exp() / mass).collect::<Vec<_>>();
    let actual = joint.posterior_weights(&ledger).unwrap();
    assert!(
        reference.iter().zip(actual).all(|(a, b)| (a - b).abs() < 1e-9),
        "joint update must equal the independently computed bivariate Gaussian conditioning"
    );

    // Retaining dependence changes the consequential result relative to assuming independence.
    let separate = independent.posterior_weights(&ledger).unwrap();
    assert!(
        total_variation(actual, separate) > 1e-6,
        "dependence must move the posterior: tv {}",
        total_variation(actual, separate)
    );
    let with_dependence = joint.combination(&[1.0, 1.0], &ledger).unwrap();
    let without_dependence = independent.combination(&[1.0, 1.0], &ledger).unwrap();
    assert_eq!(
        with_dependence.prior_variance.to_bits(),
        without_dependence.prior_variance.to_bits()
    );
    assert!(
        (with_dependence.posterior_variance - without_dependence.posterior_variance).abs()
            > 1e-9 * with_dependence.prior_variance.max(f64::MIN_POSITIVE),
        "a derived quantity's uncertainty must depend on the declared covariance: {with_dependence:?} vs {without_dependence:?}"
    );
    assert_ne!(joint.dependency_identity(), independent.dependency_identity());

    // Reuse: a different derived quantity or decision problem reuses the same update and fit;
    // a different bridge, observation or mapping is a different update.
    let product = UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(1));
    let nonlinear = joint.derived_quantity(&product, &ledger).unwrap();
    assert!(nonlinear.posterior_mean.is_finite() && nonlinear.posterior_variance >= 0.0);
    let again = original
        .condition_joint_experiment(update_with(covariance(sd, 0.8)), &ledger, ctx)
        .unwrap();
    assert_eq!(again.dependency_identity(), joint.dependency_identity());
    assert_eq!(
        joint.law_source(0, &ledger).unwrap().dependency_identity(),
        joint.dependency_identity(),
        "every utility evaluated on this update names the same dependency"
    );
    assert_eq!(prepared.fits_performed(), 1, "joint updating reuses the original fitted mechanism");
    let mut changed_observation = update_with(covariance(sd, 0.8));
    changed_observation.observed_means[0] += sd[0];
    let changed_observation =
        original.condition_joint_experiment(changed_observation, &ledger, ctx).unwrap();
    assert_ne!(changed_observation.dependency_identity(), joint.dependency_identity());
    let mut changed_mapping = update_with(covariance(sd, 0.8));
    changed_mapping.components[1].mapping.intercepts[0] += 1e-3;
    assert!(
        original.condition_joint_experiment(changed_mapping, &ledger, ctx).is_err(),
        "a mapping that departs from its declared moments is not the supported bridge"
    );
    let mut changed_bridge = update_with(covariance(sd, 0.8));
    changed_bridge.components[1].bridge.assumptions[0].statement.push_str(" (revised)");
    let changed_bridge = original.condition_joint_experiment(changed_bridge, &ledger, ctx).unwrap();
    assert_ne!(changed_bridge.dependency_identity(), joint.dependency_identity());
    assert_eq!(prepared.fits_performed(), 1);

    // Only the named quantities are informed; everything else stays model-conditional.
    let informed = joint.informed_quantities();
    assert_eq!(informed.len(), 2);
    assert_eq!(informed[0], components[0].bridge.target.quantity);
    assert!(joint.law_source(2, &ledger).is_err());
    for index in 0..2 {
        let source = joint.law_source(index, &ledger).unwrap();
        source.authorize(ScientificConsumptionScope::ModelConditional, &ledger).unwrap();
        assert!(source.authorize(ScientificConsumptionScope::IdentifiedCausal, &ledger).is_err());
    }
    assert!(joint.scope().contains("calibration_unmeasured"));

    // Ranking a further study of one component uses the jointly updated weights.
    let coefficients = &fixtures[0].coefficients;
    let threshold = {
        let law = joint.law_source(0, &ledger).unwrap();
        let _ = law;
        0.0
    };
    let problem = DecisionProblem::new(
        vec![0, 1],
        Arc::new(ContrastUtility { coefficients: coefficients.clone(), threshold }),
        vec![],
    );
    let candidates = [PanelFutureSample {
        id: "next".into(),
        sample_size: 4,
        observation_variance: 1.0,
        cost_utility: Some(0.1),
    }];
    let ranking = joint.rank_next_samples(0, &problem, &candidates, 16, &ledger, ctx).unwrap();
    assert_eq!(ranking.len(), 1);
    assert!(ranking[0].gross_evsi.is_finite() && ranking[0].gross_evsi >= 0.0);
    assert_eq!(ranking[0].dependency_identity, joint.dependency_identity());
    assert!(joint.rank_next_samples(2, &problem, &candidates, 16, &ledger, ctx).is_err());

    // Artifact: replay recomputes the weights from the retained inputs alone; a changed input,
    // or a withdrawn experiment, cannot be consumed as the current update.
    let artifact = joint.export(&ledger, ctx).unwrap();
    let replayed = PanelJointExperimentResult::replay(&artifact, ctx).unwrap();
    assert!(!replayed.grants_current_authority());
    assert_eq!(replayed.dependency_identity(), joint.dependency_identity());
    assert!(replayed.weights().iter().zip(actual).all(|(a, b)| a.to_bits() == b.to_bits()));
    joint.consume_current(&artifact, 1, &ledger, ctx).unwrap();
    let mut tampered: serde_json::Value = serde_json::from_slice(&artifact).unwrap();
    tampered["observed"][0] = serde_json::json!(
        (f64::from_bits(tampered["observed"][0].as_u64().unwrap()) + 1.0).to_bits()
    );
    let tampered = serde_json::to_vec(&tampered).unwrap();
    assert!(PanelJointExperimentResult::replay(&tampered, ctx).is_err());
    assert!(joint.consume_current(&tampered, 0, &ledger, ctx).is_err());
    let other = independent.export(&ledger, ctx).unwrap();
    assert!(joint.consume_current(&other, 0, &ledger, ctx).is_err());

    // Refusals.
    let mut single = update_with(covariance(sd, 0.8));
    single.components.truncate(1);
    single.observed_means.truncate(1);
    single.sampling_covariance = vec![sd[0] * sd[0]];
    assert!(original.condition_joint_experiment(single, &ledger, ctx).is_err());
    let mut repeated = update_with(covariance(sd, 0.8));
    repeated.components[1].bridge.source.quantity =
        repeated.components[0].bridge.source.quantity.clone();
    assert!(original.condition_joint_experiment(repeated, &ledger, ctx).is_err());
    assert!(
        original
            .condition_joint_experiment(update_with(covariance(sd, 1.0)), &ledger, ctx)
            .is_err(),
        "a singular covariance is not a joint sampling law"
    );
    let mut asymmetric = covariance(sd, 0.8);
    asymmetric[1] *= 0.5;
    assert!(original.condition_joint_experiment(update_with(asymmetric), &ledger, ctx).is_err());
    let mut misaligned = update_with(covariance(sd, 0.8));
    misaligned.sampling_covariance.pop();
    assert!(original.condition_joint_experiment(misaligned, &ledger, ctx).is_err());

    // Assumption-bounded evidence is a sensitivity, never a likelihood: it answers no decision.
    let bounded_with = |bound: f64| {
        let mut update = update_with(covariance(sd, 0.8));
        update.contrast_bias_bounds = Some(vec![bound * sd[0], bound * sd[1]]);
        original.condition_joint_experiment(update, &ledger, ctx).unwrap()
    };
    let bounded = bounded_with(1.0);
    assert!(bounded.scope().contains("no_likelihood_claim"));
    assert!(bounded.law_source(0, &ledger).is_err());
    assert!(bounded.posterior_weights(&ledger).is_err());
    assert!(bounded.derived_quantity(&sum_of_components(), &ledger).is_err());
    assert!(bounded.rank_next_samples(0, &problem, &candidates, 16, &ledger, ctx).is_err());
    assert!(joint.bias_sensitivity(&sum_of_components(), &ledger).is_err());
    let narrow = bounded_with(0.5).bias_sensitivity(&sum_of_components(), &ledger).unwrap();
    let wide = bounded.bias_sensitivity(&sum_of_components(), &ledger).unwrap();
    assert_eq!(wide.settings, 5);
    assert!((wide.nominal - narrow.nominal).abs() < 1e-12);
    assert!(wide.enumerated_minimum <= narrow.enumerated_minimum + 1e-12);
    assert!(wide.enumerated_maximum >= narrow.enumerated_maximum - 1e-12);
    assert!(wide.enumerated_maximum - wide.enumerated_minimum > 1e-9, "{wide:?}");
    let none = bounded_with(0.0).bias_sensitivity(&sum_of_components(), &ledger).unwrap();
    assert_eq!(none.enumerated_minimum.to_bits(), none.enumerated_maximum.to_bits());
    let mut negative = update_with(covariance(sd, 0.8));
    negative.contrast_bias_bounds = Some(vec![-1.0, 0.0]);
    assert!(original.condition_joint_experiment(negative, &ledger, ctx).is_err());

    // Observations already behind the experiment cannot also be an independent prior origin.
    let mut overlapping = ledger.clone();
    let reused = overlapping
        .add_source(
            SourceDeclaration::new("data_prior", SourceKind::FittedModel)
                .with_observations([ObservationId::from_key("geo", b"observed")]),
        )
        .unwrap();
    let mut counted_twice = update_with(covariance(sd, 0.8));
    counted_twice.evidence.prior_origins.push(reused);
    assert!(original.condition_joint_experiment(counted_twice, &overlapping, ctx).is_err());

    // Withdrawing the shared experiment ends current use of every component, not only one.
    let mut withdrawn = ledger.clone();
    withdrawn.withdraw_source(evidence.experiment, "withdrawn").unwrap();
    assert!(joint.law_source(0, &withdrawn).is_err());
    assert!(joint.law_source(1, &withdrawn).is_err());
    assert!(joint.posterior_weights(&withdrawn).is_err());
    assert!(joint.combination(&[1.0, 1.0], &withdrawn).is_err());
    assert!(joint.rank_next_samples(0, &problem, &candidates, 16, &withdrawn, ctx).is_err());
    assert!(joint.export(&withdrawn, ctx).is_err());
    assert!(joint.consume_current(&artifact, 0, &withdrawn, ctx).is_err());
}

fn excursion_source(original: &JointPanelResult, ctx: &ExecutionContext) -> RetainedExperiment {
    let units = 40usize;
    let periods = 3usize;
    let n = units * periods;
    let assignment: Vec<_> = (0..n).map(|i| (i * 7 + i / periods) % 11 < 5).collect();
    let treated_before = |i: usize| i % periods > 0 && assignment[i - 1];
    let first: Vec<_> = (0..n)
        .map(|i| {
            1.0 + (i / periods) as f64 * 0.01
                + if treated_before(i) { 0.2 } else { 0.0 }
                + 0.03 * ((i * 13) % 7) as f64
        })
        .collect();
    let second: Vec<_> = (0..n)
        .map(|i| {
            2.0 - (i / periods) as f64 * 0.005
                + if treated_before(i) { 0.1 } else { 0.0 }
                + 0.04 * ((i * 29 + 3) % 11) as f64
                + 0.02 * ((i / periods * 5) % 3) as f64
                + 0.5 * 0.03 * ((i * 13) % 7) as f64
        })
        .collect();
    let design: CompiledJointExcursion = compile_joint_excursion(JointExcursionDesign {
        unit_ids: (0..units).flat_map(|u| (0..periods).map(move |_| format!("geo-{u}"))).collect(),
        periods: (0..units).flat_map(|_| 0..i64::try_from(periods).unwrap()).collect(),
        treatment_probabilities: vec![0.5; n],
        lag: 1,
        min_probability: 0.1,
        assignment_law_id: "weekly-law".into(),
        independent_units_assumption_id: "independent-geos".into(),
        administrative_followup_assumption_id: "administrative".into(),
        consistency_assumption_id: "consistency".into(),
    })
    .unwrap();
    let target = original.experiment_quantity(&JointPanelResponseFunctional::TerminalMean).unwrap();
    compile_and_evidence_from_joint_excursion(
        &design,
        &[&first, &second],
        &assignment,
        "excursion",
        &target.population_id,
        ("on", "off"),
        target.horizon,
        outcome_components(original),
        Exposure::NoBetweenUnitInterference { assumption_id: "isolation".into() },
        ctx,
    )
    .unwrap()
    .into()
}

fn outcome_components(original: &JointPanelResult) -> Vec<OutcomeComponent> {
    let target = original.experiment_quantity(&JointPanelResponseFunctional::TerminalMean).unwrap();
    ["first", "second"]
        .iter()
        .map(|name| OutcomeComponent {
            name: (*name).into(),
            variable_id: target.variable_id.clone(),
            units: target.units.clone(),
        })
        .collect()
}

fn pairs_source(
    original: &JointPanelResult,
    exposure: Exposure,
    ctx: &ExecutionContext,
) -> RetainedExperiment {
    let n = 60usize;
    let design: CompiledMatchedPairDesign = compile_matched_pair_design(MatchedPairDesign {
        cluster_ids: (0..n).map(|i| format!("c{i}")).collect(),
        pairs: (0..n / 2).map(|i| (2 * i, 2 * i + 1)).collect(),
        matching_variables: vec!["pre".into()],
        treatment_probabilities: vec![0.5; n],
        cluster_sizes: vec![1; n],
        preassignment_assumption_id: "fixed".into(),
        independent_pairs_assumption_id: "independent".into(),
        consistency_assumption_id: "consistency".into(),
    })
    .unwrap();
    let assignment: Vec<_> = (0..n).map(|i| i % 2 == (i / 2) % 2).collect();
    let first: Vec<f64> = (0..n)
        .map(|i| {
            2.0 + (i % 7) as f64 / 5.0
                + (i / 2) as f64 / 10.0
                + if assignment[i] { 0.3 } else { 0.0 }
        })
        .collect();
    let second: Vec<f64> = (0..n)
        .map(|i| {
            4.0 + (i % 5) as f64 / 3.0 - (i / 2) as f64 / 20.0
                + if assignment[i] { 0.2 } else { 0.0 }
        })
        .collect();
    let target = original.experiment_quantity(&JointPanelResponseFunctional::TerminalMean).unwrap();
    compile_and_evidence_from_matched_pairs(
        &design,
        &[&first, &second],
        &assignment,
        "pairs",
        &target.population_id,
        ("on", "off"),
        target.horizon,
        outcome_components(original),
        exposure,
        ctx,
    )
    .unwrap()
    .into()
}

/// One retained experiment of two outcomes updates a panel together, with the evidence's own
/// covariance, through the same exact source/target/likelihood checks as a single component.
#[allow(
    clippy::assigning_clones,
    clippy::too_many_lines,
    reason = "the bounded reference fixture preserves its exact calculations and local structure"
)]
fn verify_retained_update(
    original: &JointPanelResult,
    ctx: &ExecutionContext,
    retained: &RetainedExperiment,
    observation: &'static [u8],
) {
    let target = original.experiment_quantity(&JointPanelResponseFunctional::TerminalMean).unwrap();
    let fixtures: Vec<_> = functionals(original)
        .into_iter()
        .map(|f| translated_fixture(original, f, true, 0))
        .collect();
    let mut ledger = fixtures[0].ledger.clone();
    let mut evidence = fixtures[0].update.evidence.clone();
    let observed = || ObservationId::from_key("retained", observation);
    let experiment = ledger
        .add_source(
            SourceDeclaration::new(retained.dependency_identity().to_hex(), SourceKind::Experiment)
                .with_observations([observed()]),
        )
        .unwrap();
    evidence.experiment = experiment;
    let components = |assumption: &str| -> Vec<JointEvidenceComponent> {
        ["first", "second"]
            .iter()
            .zip(&fixtures)
            .map(|(name, f)| {
                let mut bridge = f.update.bridge.clone();
                bridge.source.quantity = retained.component_quantity(name).unwrap();
                bridge.evidence_scope = BridgeScope {
                    population_id: target.population_id.clone(),
                    regime_id: bridge.source.quantity.regime_id.clone(),
                };
                let translation = bridge.estimand_bridge.as_mut().unwrap();
                translation.experiment_id = retained.dependency_identity().to_hex();
                translation.experiment.population_id = target.population_id.clone();
                translation.experiment.exposure_regime = bridge.source.quantity.regime_id.clone();
                bridge.assumptions.push(BridgeAssumption {
                    id: assumption.into(),
                    kind: AssumptionKind::Additional,
                    statement: "The experimental estimates have the explicitly declared joint Gaussian sampling model".into(),
                    evidence: None,
                });
                JointEvidenceComponent {
                    component: (*name).into(),
                    bridge,
                    functional: f.update.functional.clone(),
                    mapping: f.update.mapping.clone(),
                }
            })
            .collect()
    };
    let run = |parts: Vec<JointEvidenceComponent>,
               evidence: PanelExperimentEvidence,
               assumption: &str,
               ledger: &EvidenceLedger| {
        condition_panel_jointly_on_retained_evidence(
            original,
            retained.clone(),
            parts,
            evidence,
            assumption,
            ledger,
            ctx,
        )
    };
    assert!(
        run(components("joint-gaussian"), evidence.clone(), "missing-assumption", &ledger).is_err(),
        "the sampling model needs its own declared assumption"
    );
    let mut single = components("joint-gaussian");
    single.truncate(1);
    assert!(run(single, evidence.clone(), "joint-gaussian", &ledger).is_err());
    let mut repeated = components("joint-gaussian");
    repeated[1].component = "first".into();
    assert!(run(repeated, evidence.clone(), "joint-gaussian", &ledger).is_err());
    let mut wrong_source = components("joint-gaussian");
    wrong_source[0].bridge.estimand_bridge.as_mut().unwrap().experiment_id = "other-source".into();
    assert!(run(wrong_source, evidence.clone(), "joint-gaussian", &ledger).is_err());
    let mut overlapping = ledger.clone();
    let refit = overlapping
        .add_source(
            SourceDeclaration::new("refit_on_experiment", SourceKind::FittedModel)
                .with_observations([observed()]),
        )
        .unwrap();
    let mut counted_twice = evidence.clone();
    counted_twice.prior_origins.push(refit);
    assert!(
        run(components("joint-gaussian"), counted_twice, "joint-gaussian", &overlapping).is_err(),
        "observations behind the experiment cannot be an independent origin of the model"
    );
    let updated = run(components("joint-gaussian"), evidence, "joint-gaussian", &ledger).unwrap();
    assert_eq!(updated.likelihood_assumption_id(), "joint-gaussian");
    assert_eq!(updated.result().informed_quantities().len(), 2);
    let bounded = updated.result().scope().contains("no_likelihood_claim");
    if bounded {
        // Interference is only bounded: a sensitivity, never a decision source.
        assert!(updated.law_source(0, &ledger).is_err());
        let range = updated.result().bias_sensitivity(&sum_of_components(), &ledger).unwrap();
        assert!(range.enumerated_maximum > range.enumerated_minimum);
    } else {
        let conditional = updated.law_source(0, &ledger).unwrap();
        assert!(
            conditional.authorize(ScientificConsumptionScope::IdentifiedCausal, &ledger).is_err()
        );
    }
    ledger.withdraw_source(experiment, "withdrawn").unwrap();
    assert!(updated.law_source(0, &ledger).is_err());
    assert!(updated.law_source(1, &ledger).is_err());
}

pub fn verify_joint_excursion_update(original: &JointPanelResult, ctx: &ExecutionContext) {
    let source = excursion_source(original, ctx);
    verify_retained_update(original, ctx, &source, b"excursion");
}

/// The adapter is not excursion-specific: matched pairs update the panel the same way, and a
/// pair experiment whose interference is only bounded gives a sensitivity, not a likelihood.
pub fn verify_joint_pairs_update(original: &JointPanelResult, ctx: &ExecutionContext) {
    let isolated = pairs_source(
        original,
        Exposure::NoBetweenUnitInterference { assumption_id: "isolation".into() },
        ctx,
    );
    verify_retained_update(original, ctx, &isolated, b"pairs");
    let bounded = pairs_source(
        original,
        Exposure::BoundedSpillover { assumption_id: "spill".into(), bias_bounds: vec![0.05, 0.05] },
        ctx,
    );
    verify_retained_update(original, ctx, &bounded, b"bounded-pairs");
}
