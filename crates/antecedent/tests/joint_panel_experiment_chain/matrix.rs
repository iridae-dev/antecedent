// SPDX-License-Identifier: MIT OR Apache-2.0
//! Independent finite-law references across every admitted response projection,
//! deterministic/dependent mappings and changed exposure/population translations.
use super::*;
mod acceptance_record {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../conformance/response/joint_panel/acceptance_record.rs"
    ));
}

pub(super) fn exercise(
    prepared: &PreparedStudy<JointPanelState>,
    original: &JointPanelResult,
    ctx: &ExecutionContext,
) {
    verify_fitless_native_law_adapter(original, ctx);
    let horizon = original.request().worlds[0].program.horizon();
    let functionals = [
        JointPanelResponseFunctional::TerminalMean,
        JointPanelResponseFunctional::WindowAverage { from: horizon - 3, until: horizon - 1 },
        JointPanelResponseFunctional::WeightedSum {
            coefficients: (0..horizon).map(|t| if t + 2 >= horizon { 0.5 } else { 0.0 }).collect(),
        },
    ];
    for (functional_index, functional) in functionals.into_iter().enumerate() {
        for uncertain in [false, true] {
            for translation_case in 0..5 {
                let fixture =
                    translated_fixture(original, functional.clone(), uncertain, translation_case);
                let updated = original
                    .condition_experiment(fixture.update.clone(), &fixture.ledger, ctx)
                    .expect("declared finite paired translation executes without a refit");
                verify_independent_likelihood_reference(original, &updated, &fixture);
                let feasible_actions = verify_constrained_decision(&updated, &fixture);
                let free_study = verify_next_study(&updated, &fixture, vec![0, 1], 0.0, ctx);
                let constrained_study =
                    verify_next_study(&updated, &fixture, feasible_actions, 1e-7, ctx);
                assert_ne!(
                    free_study["admissible_action_identity"],
                    constrained_study["admissible_action_identity"]
                );
                verify_evidence_changes_decision(original, &updated, &fixture, ctx);
                if functional_index == 0 || translation_case == 3 {
                    verify_regime_studies(
                        original,
                        &updated,
                        &fixture,
                        translation_case,
                        uncertain,
                        ctx,
                    );
                }
                // Replay binds the translated quantities, the exact mapping slots,
                // observed evidence and the weighted law, not merely an output mean.
                verify_experiment_artifacts(prepared, &updated, &fixture, ctx);
                let functional_name =
                    ["terminal_mean", "window_average", "weighted_sum"][functional_index];
                acceptance_record::record(
                    &format!("retained_panel::{functional_name}"),
                    uncertain,
                    translation_case,
                    "retained_panel",
                    &original.dependency_identity().to_hex(),
                    serde_json::json!({"constrained_next_study":constrained_study,"unconstrained_next_study":free_study,"independent_likelihood_and_posterior":"checked","constrained_utility_and_ties":"checked","nested_next_study_reference":"checked_with_quadrature_bound_and_mcse","fresh_original_fit_and_update_replay":"checked","changed_evidence":"checked"}),
                );
            }
        }
    }
    assert_eq!(prepared.fits_performed(), 1);
}

fn translated_fixture(
    original: &JointPanelResult,
    functional: JointPanelResponseFunctional,
    uncertain: bool,
    translation_case: usize,
) -> ExperimentFixture {
    let mut fixture = experiment_fixture(original);
    let law = original.numerical_project_law_source(&functional).unwrap();
    let width = fixture.coefficients.len();
    let states = law
        .draws()
        .chunks_exact(width)
        .map(|row| row.iter().zip(&fixture.coefficients).map(|(v, c)| v * c).sum::<f64>())
        .collect::<Vec<_>>();
    let n = states.len() as f64;
    let mean = |x: &[f64]| x.iter().sum::<f64>() / n;
    let cov = |x: &[f64], y: &[f64]| {
        let (mx, my) = (mean(x), mean(y));
        x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum::<f64>() / n
    };
    let state_mean = mean(&states);
    // Each translation case is its own measurement scale. The interaction contrast is flat in
    // time once the adstock has filled, so a regime that differs only by window predicts the
    // same observations; a regime is distinguished by its paired mapping.
    let slope_base = 1.2 + 0.1 * translation_case as f64;
    let intercept_base = 0.15 + 0.01 * translation_case as f64;
    let slopes = states
        .iter()
        .map(|v| slope_base + if uncertain { 0.04 * (v - state_mean).tanh() } else { 0.0 })
        .collect::<Vec<_>>();
    let intercepts = states
        .iter()
        .map(|v| intercept_base + if uncertain { 0.02 * (v - state_mean).tanh() } else { 0.0 })
        .collect::<Vec<_>>();
    let target = original.experiment_quantity(&functional).unwrap();
    let mut bridge =
        declared_bridge(&target, mean(&slopes), cov(&slopes, &slopes), cov(&slopes, &states));
    bridge.mapping = BridgeMapping::Affine { intercept: mean(&intercepts), slope: mean(&slopes) };
    let uncertainty = bridge.mapping_uncertainty.as_mut().unwrap();
    uncertainty.intercept_variance = cov(&intercepts, &intercepts);
    // Both are affine in the same state, so the sample correlation is exactly one and the
    // covariance sits on the Cauchy-Schwarz bound, where rounding can put its square one ulp
    // above the product of the variances. State it strictly inside the bound.
    let bound = (uncertainty.intercept_variance * uncertainty.slope_variance).sqrt();
    uncertainty.intercept_slope_covariance =
        cov(&intercepts, &slopes).clamp(-bound, bound) * (1.0 - 1e-9);
    let horizon = original.request().worlds[0].program.horizon();
    let (from, until) = match &functional {
        JointPanelResponseFunctional::TerminalMean => (horizon - 1, horizon - 1),
        JointPanelResponseFunctional::WindowAverage { from, until } => (*from, *until),
        JointPanelResponseFunctional::WeightedSum { .. } => (0, horizon - 1),
        _ => unreachable!(),
    };
    let translation = bridge.estimand_bridge.as_mut().unwrap();
    translation.target.time_start = format!("program_period:{from}");
    translation.target.time_end = format!("program_period:{until}");
    translation.target.duration = (until - from + 1) as f64;
    translation.experiment = translation.target.clone();
    match translation_case {
        0 => {}
        1 => {
            translation.experiment.population_id = "reference_experiment_population".into();
            bridge.evidence_scope.population_id = translation.experiment.population_id.clone();
            bridge.source.quantity.population_id = translation.experiment.population_id.clone();
        }
        2 => {
            translation.experiment.time_start = "historical_period:0".into();
            translation.experiment.time_end = "historical_period:1".into();
            translation.experiment.duration = 2.0;
            translation.recency_delta = 10.0;
            bridge.source.quantity.horizon = 2;
        }
        3 => {
            translation.experiment.dose = 0.5;
            translation.spend_delta = 0.5;
        }
        4 => {
            translation.experiment.duration = 2.0;
            bridge.source.quantity.horizon = 2;
        }
        _ => unreachable!(),
    }
    if [1, 2, 4].contains(&translation_case) {
        bridge.transport=Some(TransportabilityPremise {id:"declared_estimand_translation".into(),statement:"The declared paired affine mapping carries the exact experimental population/time/duration setting to the retained target".into(),established_in:bridge.evidence_scope.clone(),applies_to:BridgeScope {population_id:target.population_id.clone(),regime_id:target.regime_id.clone()},evidence:Some("explicit_reference_transport_model".into())});
    }
    bridge.check_licensing().unwrap();
    if [1, 2, 4].contains(&translation_case) {
        let mut unsupported = bridge.clone();
        unsupported.transport = None;
        assert!(
            unsupported.check_licensing().is_err(),
            "changed experimental population/horizon requires an explicit transport premise"
        );
        let mut wrong_source = bridge.clone();
        wrong_source.source.quantity.population_id = "unrelated_population".into();
        assert!(
            wrong_source.check_licensing().is_err(),
            "experimental endpoint must match the translated experimental population"
        );
    }
    let mapped =
        states.iter().enumerate().map(|(i, v)| (v - intercepts[i]) / slopes[i]).collect::<Vec<_>>();
    fixture.update.observed_mean = mapped.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    fixture.update.observation_variance = likelihood_variance(&mapped, 20, 0.01);
    fixture.update.sample_size = 20;
    fixture.update.functional = functional;
    fixture.update.bridge = bridge;
    fixture.update.mapping.intercepts = intercepts;
    fixture.update.mapping.slopes = slopes;
    fixture
}

/// A future study may run in another population, window, dose, duration, exposure, recency or
/// spend than the observed experiment. It is valued through its own licensed bridge and paired
/// mapping, the update it would cause and the decision that update changes, and each stage is
/// checked here against arithmetic that calls no production update or ranking helper.
fn verify_regime_studies(
    original: &JointPanelResult,
    updated: &PanelExperimentResult,
    fixture: &ExperimentFixture,
    translation_case: usize,
    uncertain: bool,
    ctx: &ExecutionContext,
) {
    let horizon = original.request().worlds[0].program.horizon();
    let future_functional = match &fixture.update.functional {
        JointPanelResponseFunctional::TerminalMean => {
            JointPanelResponseFunctional::WindowAverage { from: horizon - 3, until: horizon - 1 }
        }
        _ => JointPanelResponseFunctional::TerminalMean,
    };
    let future_case = (translation_case + 2) % 5;
    let future = translated_fixture(original, future_functional.clone(), uncertain, future_case);
    let mut ledger = fixture.ledger.clone();
    let mapping_evidence = ledger
        .add_source(
            SourceDeclaration::new("future_regime_mapping", SourceKind::FittedModel)
                .with_observations([ObservationId::from_key("future_mapping", b"source")]),
        )
        .unwrap();
    let regime = PanelFutureRegime {
        functional: future_functional.clone(),
        bridge: future.update.bridge.clone(),
        mapping: future.update.mapping.clone(),
        mapping_evidence,
    };
    let width = fixture.coefficients.len();
    let dot = |row: &[f64]| row.iter().zip(&fixture.coefficients).map(|(v, c)| v * c).sum::<f64>();
    let (_, posterior_means) = updated.response_means(&fixture.ledger).unwrap();
    let center = dot(posterior_means);
    let problem = DecisionProblem::new(
        vec![0, 1],
        Arc::new(ContrastUtility { coefficients: fixture.coefficients.clone(), threshold: center }),
        vec![],
    );
    let sample = |id: &str, cost: f64| PanelFutureSample {
        id: id.into(),
        sample_size: 4,
        observation_variance: 1.0,
        cost_utility: Some(cost),
    };
    let studies = vec![
        PanelFutureStudy { sample: sample("observed_estimand", 0.02), regime: None },
        PanelFutureStudy { sample: sample("regime_cheap", 0.01), regime: Some(regime.clone()) },
        PanelFutureStudy { sample: sample("regime_dear", 0.2), regime: Some(regime.clone()) },
    ];
    let ranking = updated.rank_next_studies(&problem, &studies, 32, &ledger, ctx).unwrap();
    assert_eq!(ranking.len(), 3);
    // Independent references: the regime's own predicted observations per draw slot.
    let regime_law = original.numerical_project_law_source(&future_functional).unwrap();
    let regime_means = regime_law
        .draws()
        .chunks_exact(width)
        .enumerate()
        .map(|(slot, row)| {
            (dot(row) - regime.mapping.intercepts[slot]) / regime.mapping.slopes[slot]
        })
        .collect::<Vec<_>>();
    let observed_law = updated.numerical_law_source(&fixture.ledger).unwrap();
    let observed_means = observed_law
        .draws()
        .chunks_exact(width)
        .enumerate()
        .map(|(slot, row)| {
            (dot(row) - fixture.update.mapping.intercepts[slot])
                / fixture.update.mapping.slopes[slot]
        })
        .collect::<Vec<_>>();
    assert!(
        regime_means.iter().zip(&observed_means).any(|(a, b)| (a - b).abs() > 1e-9),
        "a different regime must predict different observations"
    );
    let (reference_regime, regime_error) =
        reference_study_value_for_means(updated, fixture, &regime_means, center);
    let (reference_observed, observed_error) =
        reference_study_value_for_means(updated, fixture, &observed_means, center);
    for row in &ranking {
        let (reference, error, cost) = match row.value.id.as_str() {
            "observed_estimand" => (reference_observed, observed_error, 0.02),
            "regime_cheap" => (reference_regime, regime_error, 0.01),
            "regime_dear" => (reference_regime, regime_error, 0.2),
            other => panic!("unexpected study {other}"),
        };
        let tolerance = error + 6.0 * row.value.standard_error + 1e-9;
        assert!(
            (row.value.gross_evsi - reference).abs() <= tolerance,
            "{}: production {}, independent reference {reference} +/- {tolerance}",
            row.value.id,
            row.value.gross_evsi
        );
        assert!((row.value.net_value.unwrap() - (reference - cost)).abs() <= tolerance);
        assert_eq!(row.regime_identity.is_some(), row.value.id != "observed_estimand");
    }
    let identity =
        |id: &str| ranking.iter().find(|row| row.value.id == id).unwrap().regime_identity;
    assert_eq!(identity("regime_cheap"), identity("regime_dear"));
    let mut other_mapping = regime.clone();
    other_mapping.mapping.slopes.iter_mut().for_each(|slope| *slope *= 1.0 + 1e-3);
    let changed = updated.rank_next_studies(
        &problem,
        &[PanelFutureStudy { sample: sample("regime_cheap", 0.01), regime: Some(other_mapping) }],
        2,
        &ledger,
        ctx,
    );
    // A different paired mapping is a different study, whether or not its moments still validate.
    if let Ok(changed) = changed {
        assert_ne!(changed[0].regime_identity, identity("regime_cheap"));
    }
    // The ranking is only as good as the update it prices: execute it for the two extreme
    // outcomes, check the posterior against independent arithmetic, and require the response
    // and the decision to move.
    let spread = regime_means.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - regime_means.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(spread > 0.0);
    let decisive = PanelFutureStudy {
        sample: PanelFutureSample {
            id: "decisive".into(),
            sample_size: 1,
            observation_variance: (spread / 20.0).powi(2),
            cost_utility: Some(0.0),
        },
        regime: Some(regime.clone()),
    };
    let outcome = |y: f64| updated.update_for_future_study(&decisive, y, &ledger).unwrap();
    let high = outcome(regime_means.iter().copied().fold(f64::NEG_INFINITY, f64::max));
    let low = outcome(regime_means.iter().copied().fold(f64::INFINITY, f64::min));
    let prior_weights = observed_law.metadata().weights.clone().unwrap();
    for update in [&high, &low] {
        let variance = (spread / 20.0).powi(2);
        let anchor = regime_means
            .iter()
            .copied()
            .min_by(|a, b| {
                (update.observed_mean - a).abs().total_cmp(&(update.observed_mean - b).abs())
            })
            .unwrap();
        let unnormalized = regime_means
            .iter()
            .zip(&prior_weights)
            .map(|(m, w)| {
                w * ((m - anchor) * (2.0 * update.observed_mean - m - anchor) / (2.0 * variance))
                    .exp()
            })
            .collect::<Vec<_>>();
        let total = unnormalized.iter().sum::<f64>();
        for (reference, actual) in unnormalized.iter().zip(&update.posterior_weights) {
            assert!((reference / total - actual).abs() < 1e-10);
        }
        assert!((update.posterior_weights.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert_eq!(update.regime_identity, identity("regime_cheap"));
    }
    assert!((dot(&high.prior_response_means) - center).abs() < 1e-9);
    assert!(
        dot(&high.response_means) > center && center > dot(&low.response_means),
        "opposite outcomes of the same study must move the joint response across the threshold"
    );
    // The decision rule the problem encodes (change when the contrast beats the threshold)
    // therefore flips between the two outcomes.
    assert_ne!(dot(&high.response_means) > center, dot(&low.response_means) > center);
    // Unlicensed, unaligned or stale regimes refuse; none falls back to the observed estimand.
    let rank = |regime: PanelFutureRegime, ledger: &EvidenceLedger| {
        updated.rank_next_studies(
            &problem,
            &[PanelFutureStudy { sample: sample("candidate", 0.0), regime: Some(regime) }],
            2,
            ledger,
            ctx,
        )
    };
    if [1, 2, 4].contains(&future_case) {
        let mut no_transport = regime.clone();
        no_transport.bridge.transport = None;
        assert!(rank(no_transport, &ledger).is_err(), "a changed regime needs its transport");
    }
    let mut short = regime.clone();
    short.mapping.intercepts.pop();
    short.mapping.slopes.pop();
    assert!(rank(short, &ledger).is_err());
    let mut foreign = regime.clone();
    foreign.mapping.dependency = antecedent_core::SemanticDigest::from_bytes([7; 32]);
    assert!(rank(foreign, &ledger).is_err());
    let mut mismatched = regime.clone();
    mismatched.functional = fixture.update.functional.clone();
    assert!(rank(mismatched, &ledger).is_err());
    let mut withdrawn = ledger.clone();
    withdrawn.withdraw_source(mapping_evidence, "withdrawn future mapping").unwrap();
    assert!(rank(regime.clone(), &withdrawn).is_err());
    let partly_costed = updated.rank_next_studies(
        &problem,
        &[
            PanelFutureStudy { sample: sample("costed", 0.1), regime: Some(regime.clone()) },
            PanelFutureStudy {
                sample: PanelFutureSample { cost_utility: None, ..sample("free", 0.0) },
                regime: Some(regime),
            },
        ],
        2,
        &ledger,
        ctx,
    );
    assert!(partly_costed.is_err());
}

fn verify_constrained_decision(
    updated: &PanelExperimentResult,
    fixture: &ExperimentFixture,
) -> Vec<usize> {
    let law = updated.numerical_law_source(&fixture.ledger).unwrap();
    let weights = law.metadata().weights.as_ref().unwrap();
    let width = fixture.coefficients.len();
    let values = law
        .draws()
        .chunks_exact(width)
        .map(|row| row.iter().zip(&fixture.coefficients).map(|(v, c)| v * c).sum::<f64>())
        .collect::<Vec<_>>();
    let center = values.iter().zip(weights).map(|(v, w)| v * w).sum::<f64>();
    let costs = [0.0, center - 1e-7, center + 1e-7];
    let coordinates = updated.quantities(&fixture.ledger).unwrap();
    let contrast =
        fixture.coefficients.iter().enumerate().fold(UtilityExpr::Const(0.0), |sum, (i, c)| {
            UtilityExpr::sum(
                sum,
                UtilityExpr::product(UtilityExpr::Const(*c), UtilityExpr::Input(i)),
            )
        });
    let contract = DecisionContract {
        actions: costs
            .iter()
            .enumerate()
            .map(|(i, c)| DecisionAction {
                id: format!("action_{i}"),
                kind: ActionKind::Policy,
                inputs: coordinates.clone(),
                utility: if i == 0 {
                    zero_utility()
                } else {
                    UtilityExpr::sum(contrast.clone(), UtilityExpr::Const(-c))
                },
            })
            .collect(),
        utility_units: coordinates[0].units.clone(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![HardConstraint {
            id: "binding_operational_limit".into(),
            expr: UtilityExpr::sum(zero_utility(), UtilityExpr::Const(1.0)),
            bound: 0.0,
            min_probability: 1.0,
            units: "declared_operational_unit".into(),
            applies_to: vec!["action_1".into()],
        }],
        target_population: coordinates[0].population_id.clone(),
        horizon: coordinates[0].horizon,
        structural_policy: StructuralPolicy::ReportOnly,
    };
    let actual = updated.decide(&contract, &fixture.ledger).unwrap().result;
    let mut unconstrained = contract.clone();
    unconstrained.constraints.clear();
    let free = updated.decide(&unconstrained, &fixture.ledger).unwrap().result;
    assert!(
        matches!(free.verdict,antecedent_design::decision_eval::Verdict::UniquelyOptimal(ref id) if id=="action_1")
    );
    let mut tied = unconstrained;
    tied.actions[1].utility = UtilityExpr::sum(contrast.clone(), UtilityExpr::Const(-center));
    let tie_result = updated.decide(&tied, &fixture.ledger).unwrap().result;
    assert!(
        tie_result.actions[0].expected_utility.abs() < 1e-10
            && tie_result.actions[1].expected_utility.abs() < 1e-10
    );
    assert_eq!(actual.actions.len(), 3);
    for (i, action) in actual.actions.iter().enumerate() {
        let reference = if i == 0 {
            0.0
        } else {
            values.iter().zip(weights).map(|(v, w)| w * (v - costs[i])).sum::<f64>()
        };
        assert!((action.expected_utility - reference).abs() < 1e-9);
        assert_eq!(action.admissible, i != 1);
    }
    // Enumeration of all feasible actions is an independent constrained optimum.
    assert!(
        matches!(actual.verdict,antecedent_design::decision_eval::Verdict::UniquelyOptimal(ref id) if id=="action_0")
    );
    let reference_evpi =
        values.iter().zip(weights).map(|(v, w)| w * (v - costs[2]).max(0.0)).sum::<f64>();
    assert!((actual.evpi.unwrap() - reference_evpi).abs() < 1e-9);
    actual.actions.iter().enumerate().filter_map(|(i, a)| a.admissible.then_some(i)).collect()
}

fn verify_evidence_changes_decision(
    original: &JointPanelResult,
    high: &PanelExperimentResult,
    fixture: &ExperimentFixture,
    ctx: &ExecutionContext,
) {
    let law = original.numerical_project_law_source(&fixture.update.functional).unwrap();
    let width = fixture.coefficients.len();
    let mut lower = fixture.update.clone();
    lower.observed_mean = law
        .draws()
        .chunks_exact(width)
        .enumerate()
        .map(|(i, row)| {
            let v = row.iter().zip(&fixture.coefficients).map(|(v, c)| v * c).sum::<f64>();
            (v - lower.mapping.intercepts[i]) / lower.mapping.slopes[i]
        })
        .fold(f64::INFINITY, f64::min);
    let low = original.condition_experiment(lower.clone(), &fixture.ledger, ctx).unwrap();
    let dot = |row: &[f64]| row.iter().zip(&fixture.coefficients).map(|(v, c)| v * c).sum::<f64>();
    let low_mean = dot(low.response_means(&fixture.ledger).unwrap().1);
    let high_mean = dot(high.response_means(&fixture.ledger).unwrap().1);
    assert!(
        high_mean > low_mean + 1e-9,
        "opposite observed evidence changes the actual posterior contrast"
    );
    let changed = ExperimentFixture {
        update: lower,
        ledger: fixture.ledger.clone(),
        coefficients: fixture.coefficients.clone(),
    };
    verify_independent_likelihood_reference(original, &low, &changed);
}

fn verify_next_study(
    updated: &PanelExperimentResult,
    fixture: &ExperimentFixture,
    actions: Vec<usize>,
    cost_shift: f64,
    ctx: &ExecutionContext,
) -> serde_json::Value {
    let mean = updated
        .response_means(&fixture.ledger)
        .unwrap()
        .1
        .iter()
        .zip(&fixture.coefficients)
        .map(|(v, c)| v * c)
        .sum::<f64>()
        + cost_shift;
    let action_ids = actions.clone();
    let problem = DecisionProblem::new(
        actions,
        Arc::new(ContrastUtility { coefficients: fixture.coefficients.clone(), threshold: mean }),
        vec![],
    );
    let candidates = [
        PanelFutureSample {
            id: "small_cost".into(),
            sample_size: 4,
            observation_variance: 1.0,
            cost_utility: Some(0.02),
        },
        PanelFutureSample {
            id: "large_cost".into(),
            sample_size: 4,
            observation_variance: 1.0,
            cost_utility: Some(0.07),
        },
    ];
    let ranking =
        updated.rank_next_samples(&problem, &candidates, 32, &fixture.ledger, ctx).unwrap();
    let (reference, error) = reference_next_study_value(updated, fixture, mean);
    assert_eq!(ranking[0].id, "small_cost");
    for candidate in &ranking {
        let tolerance = error + 6.0 * candidate.standard_error + 1e-9;
        assert!((candidate.gross_evsi - reference).abs() <= tolerance);
        let cost = if candidate.id == "small_cost" { 0.02 } else { 0.07 };
        assert!((candidate.net_value.unwrap() - (reference - cost)).abs() <= tolerance);
    }
    serde_json::json!({"evsi_reference":reference,"evsi_absolute_error":(ranking[0].gross_evsi-reference).abs(),"quadrature_error_bound":error,"evsi_mcse":ranking[0].standard_error,"admissible_action_identity":ranking[0].admissible_action_identity.to_hex(),"operational_action_ids":action_ids,"terminal_cost":mean,"net_value":ranking[0].net_value})
}

fn verify_fitless_native_law_adapter(original: &JointPanelResult, ctx: &ExecutionContext) {
    let fixture = translated_fixture(original, JointPanelResponseFunctional::TerminalMean, true, 1);
    let source = original.project_law_source(&fixture.update.functional).unwrap();
    assert_eq!(
        JointLawExperimentResult::quantity(&source).unwrap(),
        fixture.update.bridge.target.quantity
    );
    let generic = JointLawExperimentResult::condition(
        source,
        JointLawExperimentRequest {
            bridge: fixture.update.bridge.clone(),
            observed_mean: fixture.update.observed_mean,
            observation_variance: fixture.update.observation_variance,
            sample_size: fixture.update.sample_size,
            mapping: fixture.update.mapping.clone(),
            evidence: fixture.update.evidence.clone(),
        },
        &fixture.ledger,
        ctx,
    )
    .unwrap();
    let panel =
        original.condition_experiment(fixture.update.clone(), &fixture.ledger, ctx).unwrap();
    let actual = generic.numerical_law_source(&fixture.ledger).unwrap();
    let expected = panel.numerical_law_source(&fixture.ledger).unwrap();
    assert_eq!(actual.draws(), expected.draws());
    assert_eq!(actual.metadata().weights, expected.metadata().weights);
    let mut tiny = ExecutionContext::for_tests(1);
    tiny.set_memory_limit(Some(1));
    assert!(generic.export(&fixture.ledger, &tiny).is_err());
    let artifact = generic.export(&fixture.ledger, ctx).unwrap();
    assert!(JointLawExperimentResult::replay(&artifact, &tiny).is_err());
    let historical = JointLawExperimentResult::replay(&artifact, ctx).unwrap();
    assert!(!historical.grants_current_authority());
    assert_eq!(historical.numerical_law().metadata(), actual.metadata());
    assert_eq!(historical.numerical_law().draws(), actual.draws());
    generic.consume_current(&artifact, &fixture.ledger, ctx).unwrap();
    let mut withdrawn = fixture.ledger.clone();
    withdrawn.withdraw_source(fixture.update.evidence.experiment, "withdrawn experiment").unwrap();
    assert!(generic.consume_current(&artifact, &withdrawn, ctx).is_err());
    let scientific = generic.law_source(&fixture.ledger).unwrap();
    assert!(
        scientific
            .authorize(ScientificConsumptionScope::IdentifiedCausal, &fixture.ledger)
            .is_err()
    );
    let center = actual
        .draws()
        .chunks_exact(fixture.coefficients.len())
        .zip(actual.metadata().weights.as_ref().unwrap())
        .map(|(row, w)| w * row.iter().zip(&fixture.coefficients).map(|(v, c)| v * c).sum::<f64>())
        .sum::<f64>();
    let problem = DecisionProblem::new(
        vec![0, 1],
        Arc::new(ContrastUtility { coefficients: fixture.coefficients.clone(), threshold: center }),
        vec![],
    );
    let samples = [PanelFutureSample {
        id: "design_law_next_study".into(),
        sample_size: 4,
        observation_variance: 1.0,
        cost_utility: Some(0.03),
    }];
    let generic_ranking =
        generic.rank_next_samples(&problem, &samples, 32, &fixture.ledger, ctx).unwrap();
    let panel_ranking =
        panel.rank_next_samples(&problem, &samples, 32, &fixture.ledger, ctx).unwrap();
    assert_eq!(generic_ranking[0].gross_evsi, panel_ranking[0].gross_evsi);
    assert_eq!(generic_ranking[0].net_value, panel_ranking[0].net_value);
}

#[test]
fn finite_experiment_likelihood_matches_separately_generated_reference() {
    let reference: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../conformance/response/joint_panel/experiment_reference.json"
    ))
    .unwrap();
    let weights = antecedent_prob::finite_belief::AffineGaussianExperiment {
        observed_mean: 0.7,
        observation_variance: 0.6,
        sample_size: 3,
        target_values: &[-1.0, 0.5, 2.0],
        intercepts: &[-0.1, 0.0, 0.1],
        slopes: &[0.8, 1.0, 1.2],
    }
    .condition(&[0.2, 0.5, 0.3])
    .unwrap();
    for (actual, expected) in weights.iter().zip(reference["posterior_weights"].as_array().unwrap())
    {
        assert!((actual - expected.as_f64().unwrap()).abs() < 1e-14);
    }
}

/// Native randomization posterior → explicit translation → experiment update →
/// constrained decision → future-data ranking. The original panel fit supplies
/// only the exact scientific world definitions; no panel estimates enter this chain.
fn native_randomized_chain(target: &antecedent_core::ScientificTarget, ctx: &ExecutionContext) {
    use antecedent::analysis::joint_panel::designs::{
        JointDesignProof, JointDesignWorldBinding, PreparedJointDesign,
    };
    use antecedent_estimate::joint_design::{
        compiled::{JointDesignInference, JointDesignRecipe},
        randomized::{JointRandomizedDesign, JointRandomizedUnit, RandomizedGaussianPrior},
    };
    let worlds = (0..4).map(|i| format!("assigned_complete_program_{i}")).collect::<Vec<_>>();
    for sampling_unit in ["participant_factorial", "participant_multiarm", "geographical_cluster"] {
        let truth = [0.0, 1.0, 2.0, 4.0];
        let units = (0..64)
            .map(|i| JointRandomizedUnit {
                id: format!("{sampling_unit}:{i}"),
                arm: i % 4,
                probabilities: vec![0.25; 4],
                outcome: truth[i % 4] + if (i / 4) % 2 == 0 { 0.1 } else { -0.1 },
                population_weight: 1.0,
            })
            .collect::<Vec<_>>();
        let TargetForm::Contrast(target_contrast) = target.form() else {
            panic!("contrast");
        };
        let source_binding =
            antecedent::analysis::joint_panel::designs::JointDesignSourceBinding::from_world(
                &target_contrast.worlds().worlds()[0],
                "The prespecified complete-program target population".into(),
            )
            .unwrap();
        let design=JointRandomizedDesign {worlds:worlds.clone(),sampling_unit:source_binding.sampling_unit.clone(),population:"The prespecified complete-program target population".into(),consistency:"Each assignment key denotes its exact retained target multichannel schedule and measured terminal outcome".into(),between_unit_isolation:"Independent units with no exposure outside assigned complete-program cluster".into()};
        let proof = JointDesignProof {
            version: 1,
            longitudinal_arms: vec![],
            source_binding,
            recipe: JointDesignRecipe::Randomized { design, units: units.clone() },
            worlds: {
                let TargetForm::Contrast(contrast) = target.form() else {
                    panic!("contrast");
                };
                contrast
                    .worlds()
                    .worlds()
                    .iter()
                    .zip(&worlds)
                    .map(|(world, key)| JointDesignWorldBinding {
                        key: key.clone(),
                        world: world.id().raw(),
                        target_identity:
                            antecedent::analysis::joint_panel::designs::joint_design_world_identity(
                                world,
                            )
                            .to_hex(),
                    })
                    .collect()
            },
        };
        let prepared = PreparedJointDesign::compile(target.clone(), proof, ctx).unwrap();
        let frequentist = prepared.execute(JointDesignInference::Frequentist, ctx).unwrap();
        assert!(
            frequentist.posterior_source().is_err(),
            "design sampling uncertainty is not a Bayesian prior"
        );
        let prior = RandomizedGaussianPrior {
            mean: vec![0.0; 4],
            variance: vec![4.0; 4],
            outcome_variance: vec![1.0; units.len()],
            draws: 512,
            seed: 77,
        };
        let result =
            prepared.execute(JointDesignInference::RandomizedGaussian { prior }, ctx).unwrap();
        let source = result.posterior_source().unwrap();
        let ScientificSourcePayload::Law(prior_law) = source.as_numerical_source() else {
            panic!("native posterior law");
        };
        for arm in 0..4 {
            let count = units.iter().filter(|u| u.arm == arm).count() as f64;
            let sum = units.iter().filter(|u| u.arm == arm).map(|u| u.outcome).sum::<f64>();
            let variance = 1.0 / (0.25 + count);
            let expected = variance * sum;
            assert!(
                (prior_law.mean(arm).unwrap() - expected).abs() < 8.0 * (variance / 512.0).sqrt()
            );
        }
        let (family, case) = match sampling_unit {
            "participant_factorial" => ("factorial_multi_arm", "compiled_world_factorial"),
            "participant_multiarm" => ("factorial_multi_arm", "compiled_world_multi_arm"),
            _ => ("geo_experiment", "compiled_world_geo"),
        };
        exercise_native_source(&result, family, case, ctx);
        record_world_refusal(&frequentist, family, case);
        let artifact = result.export_artifact().unwrap();
        prepared.consume_current(&artifact, ctx).unwrap();
    }
}

fn exercise_native_source(
    design: &antecedent::analysis::joint_panel::designs::JointDesignResult,
    family: &str,
    case: &str,
    ctx: &ExecutionContext,
) {
    for uncertain in [false, true] {
        for translation in 0..5 {
            exercise_native_source_variant(design, uncertain, translation, family, case, ctx);
        }
    }
}
fn exercise_native_source_variant(
    design: &antecedent::analysis::joint_panel::designs::JointDesignResult,
    uncertain: bool,
    translation: usize,
    family: &str,
    case: &str,
    ctx: &ExecutionContext,
) {
    let source = design.posterior_source().unwrap();
    let quantity = JointLawExperimentResult::quantity(&source).unwrap();
    let ScientificSourcePayload::Law(prior) = source.as_numerical_source() else {
        panic!("actual posterior");
    };
    let TargetForm::Contrast(contrast) = source.target().form() else {
        panic!("contrast");
    };
    let projection = source.projection().unwrap();
    let coefficients = projection
        .outputs
        .iter()
        .map(|o| {
            let world = projection.basis[o.terms[0].0].world;
            contrast.terms().iter().find(|t| t.world == world).map_or(0.0, |t| t.coefficient)
        })
        .collect::<Vec<_>>();
    let values = prior
        .draws()
        .chunks_exact(coefficients.len())
        .map(|row| row.iter().zip(&coefficients).map(|(v, c)| v * c).sum::<f64>())
        .collect::<Vec<_>>();
    let n = values.len();
    let center = values.iter().sum::<f64>() / n as f64;
    let slopes = values
        .iter()
        .map(|v| if uncertain { 1.0 + 0.05 * (v - center).tanh() } else { 1.0 })
        .collect::<Vec<_>>();
    let slope_mean = slopes.iter().sum::<f64>() / n as f64;
    let variance = slopes.iter().map(|s| (s - slope_mean).powi(2)).sum::<f64>() / n as f64;
    let covariance =
        slopes.iter().zip(&values).map(|(s, v)| (s - slope_mean) * (v - center)).sum::<f64>()
            / n as f64;
    let mut bridge = declared_bridge(&quantity, slope_mean, variance, covariance);
    let setting = bridge.estimand_bridge.as_mut().unwrap();
    setting.target.time_start = format!("program_period:{}", quantity.horizon.saturating_sub(1));
    setting.target.time_end = setting.target.time_start.clone();
    setting.experiment = setting.target.clone();
    match translation {
        0 => {}
        1 => {
            bridge.source.quantity.population_id = "historical_experiment_population".into();
            bridge.evidence_scope.population_id = bridge.source.quantity.population_id.clone();
            bridge.estimand_bridge.as_mut().unwrap().experiment.population_id =
                bridge.source.quantity.population_id.clone();
        }
        2 => {
            bridge.source.quantity.horizon = if quantity.horizon == 2 { 3 } else { 2 };
            let e = bridge.estimand_bridge.as_mut().unwrap();
            e.experiment.time_start = "historical_period:0".into();
            e.experiment.time_end =
                format!("historical_period:{}", bridge.source.quantity.horizon - 1);
            e.experiment.duration = f64::from(bridge.source.quantity.horizon);
            e.recency_delta = 10.0;
        }
        3 => {
            let e = bridge.estimand_bridge.as_mut().unwrap();
            e.experiment.dose = 0.5;
            e.spend_delta = 0.5;
        }
        4 => {
            bridge.source.quantity.horizon = if quantity.horizon == 2 { 3 } else { 2 };
            let e = bridge.estimand_bridge.as_mut().unwrap();
            e.experiment.duration = 2.0;
            e.experiment.time_start = "experimental_window:start".into();
            e.experiment.time_end = "experimental_window:end".into();
        }
        _ => unreachable!(),
    }
    if [1, 2, 4].contains(&translation) {
        bridge.transport=Some(TransportabilityPremise {id:"explicit_estimand_transport".into(),statement:"The retained paired mapping carries this specific experimental population/time/duration setting to the exact native target".into(),established_in:bridge.evidence_scope.clone(),applies_to:BridgeScope {population_id:quantity.population_id.clone(),regime_id:quantity.regime_id.clone()},evidence:Some("separately_supplied_transport_model".into())});
    }
    bridge.check_licensing().unwrap();
    let mut ledger = EvidenceLedger::new();
    let model = ledger
        .add_source(
            SourceDeclaration::new(source.dependency_identity().to_hex(), SourceKind::FittedModel)
                .with_observations([ObservationId::from_key("native-design", b"original")]),
        )
        .unwrap();
    let experiment = ledger
        .add_source(
            SourceDeclaration::new("independent-experimental-likelihood", SourceKind::Experiment)
                .with_observations([ObservationId::from_key("native-experiment", b"new")]),
        )
        .unwrap();
    let mapping = ledger
        .add_source(
            SourceDeclaration::new("paired-mapping", SourceKind::FittedModel)
                .with_observations([ObservationId::from_key("native-mapping", b"translation")]),
        )
        .unwrap();
    let request = JointLawExperimentRequest {
        bridge,
        observed_mean: center + 0.2,
        observation_variance: 0.3,
        sample_size: 8,
        mapping: PanelExperimentMapping {
            dependency: source.dependency_identity(),
            intercepts: vec![0.0; n],
            slopes: slopes.clone(),
        },
        evidence: PanelExperimentEvidence { model, experiment, mapping, prior_origins: vec![] },
    };
    let updated = design.condition_experiment(request, &ledger, ctx).unwrap();
    let posterior = updated.numerical_law_source(&ledger).unwrap();
    let predicted = values.iter().zip(&slopes).map(|(v, s)| v / s).collect::<Vec<_>>();
    let anchor = predicted
        .iter()
        .min_by(|a, b| ((center + 0.2) - **a).abs().total_cmp(&((center + 0.2) - **b).abs()))
        .unwrap();
    let masses = predicted
        .iter()
        .map(|m| ((m - anchor) * (2.0 * (center + 0.2) - m - anchor) / (2.0 * 0.3 / 8.0)).exp())
        .collect::<Vec<_>>();
    let norm = masses.iter().sum::<f64>();
    for (actual, reference) in posterior.metadata().weights.as_ref().unwrap().iter().zip(&masses) {
        assert!((actual - reference / norm).abs() < 1e-12);
    }
    let current = values
        .iter()
        .zip(posterior.metadata().weights.as_ref().unwrap())
        .map(|(v, w)| v * w)
        .sum::<f64>();
    let coordinates = projection.outputs.iter().map(|o| o.quantity.clone()).collect::<Vec<_>>();
    let expression =
        coefficients.iter().enumerate().fold(UtilityExpr::Const(-current), |e, (i, c)| {
            UtilityExpr::sum(e, UtilityExpr::product(UtilityExpr::Const(*c), UtilityExpr::Input(i)))
        });
    let contract = DecisionContract {
        actions: vec![
            DecisionAction {
                id: "hold".into(),
                kind: ActionKind::Policy,
                inputs: coordinates.clone(),
                utility: zero_utility(),
            },
            DecisionAction {
                id: "act".into(),
                kind: ActionKind::Policy,
                inputs: coordinates.clone(),
                utility: expression,
            },
        ],
        utility_units: quantity.units.clone(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: quantity.population_id.clone(),
        horizon: quantity.horizon,
        structural_policy: StructuralPolicy::ReportOnly,
    };
    let decision = updated.decide(&contract, &ledger).unwrap();
    assert!(decision.result.actions.iter().all(|a| a.expected_utility.abs() < 1e-10));
    let mut constrained = contract.clone();
    constrained.constraints.push(HardConstraint {
        id: "binding_capacity".into(),
        expr: UtilityExpr::sum(zero_utility(), UtilityExpr::Const(1.0)),
        bound: 0.0,
        min_probability: 1.0,
        units: "capacity".into(),
        applies_to: vec!["act".into()],
    });
    let limited = updated.decide(&constrained, &ledger).unwrap();
    assert!(!limited.result.actions[1].admissible);
    assert!(
        matches!(limited.result.verdict,antecedent_design::decision_eval::Verdict::UniquelyOptimal(ref id) if id=="hold")
    );
    for perturbation in [-1e-7, 1e-7] {
        let mut weak = contract.clone();
        weak.actions[1].utility =
            UtilityExpr::sum(weak.actions[1].utility.clone(), UtilityExpr::Const(perturbation));
        let d = updated.decide(&weak, &ledger).unwrap();
        assert!((d.result.actions[1].expected_utility - perturbation).abs() < 1e-10);
        let expected = if perturbation > 0.0 { "act" } else { "hold" };
        assert!(
            matches!(d.result.verdict,antecedent_design::decision_eval::Verdict::UniquelyOptimal(ref id) if id==expected)
        );
    }
    let problem = DecisionProblem::new(
        vec![0, 1],
        Arc::new(ContrastUtility { coefficients: coefficients.clone(), threshold: current }),
        vec![],
    );
    let samples = [
        PanelFutureSample {
            id: "cheap".into(),
            sample_size: 4,
            observation_variance: 1.0,
            cost_utility: Some(0.01),
        },
        PanelFutureSample {
            id: "costly".into(),
            sample_size: 4,
            observation_variance: 1.0,
            cost_utility: Some(0.1),
        },
    ];
    let ranking = updated.rank_next_samples(&problem, &samples, 128, &ledger, ctx).unwrap();
    assert_eq!(ranking[0].id, "cheap");
    assert_eq!(ranking[0].gross_evsi, ranking[1].gross_evsi);
    let feasible_actions = limited
        .result
        .actions
        .iter()
        .enumerate()
        .filter_map(|(i, a)| a.admissible.then_some(i))
        .collect::<Vec<_>>();
    assert_eq!(feasible_actions, vec![0]);
    let constrained_problem = DecisionProblem::new(
        feasible_actions,
        Arc::new(ContrastUtility { coefficients: coefficients.clone(), threshold: current }),
        vec![],
    );
    let constrained_ranking =
        updated.rank_next_samples(&constrained_problem, &samples, 8, &ledger, ctx).unwrap();
    assert_ne!(
        constrained_ranking[0].admissible_action_identity,
        ranking[0].admissible_action_identity
    );
    assert!(
        ranking[0].gross_evsi > constrained_ranking[0].gross_evsi + 1e-12,
        "changing operational feasibility must change future information value"
    );
    for row in &constrained_ranking {
        assert!(row.gross_evsi.abs() < 1e-12);
        let cost = if row.id == "cheap" { 0.01 } else { 0.1 };
        assert!((row.net_value.unwrap() + cost).abs() < 1e-12);
    }
    let sigma = 0.5;
    let lower = predicted.iter().copied().fold(f64::INFINITY, f64::min) - 8.0 * sigma;
    let upper = predicted.iter().copied().fold(f64::NEG_INFINITY, f64::max) + 8.0 * sigma;
    let step = (upper - lower) / 2048.0;
    let weighted_gains = values
        .iter()
        .zip(posterior.metadata().weights.as_ref().unwrap())
        .map(|(v, w)| w * (v - current))
        .collect::<Vec<_>>();
    let density = 1.0 / (sigma * (2.0 * std::f64::consts::PI).sqrt());
    let mut reference = 0.0;
    for cell in 0..2048 {
        let y = lower + (f64::from(cell) + 0.5) * step;
        reference += step
            * predicted
                .iter()
                .zip(&weighted_gains)
                .map(|(m, g)| g * density * (-0.5 * ((y - m) / sigma).powi(2)).exp())
                .sum::<f64>()
                .max(0.0);
    }
    reference -= weighted_gains.iter().sum::<f64>().max(0.0);
    let absolute = weighted_gains.iter().map(|g| g.abs()).sum::<f64>();
    let error = step / 2.0 * absolute * (2.0 / std::f64::consts::PI).sqrt() / sigma
        + absolute * 2.0 * (-32.0_f64).exp() / (8.0 * (2.0 * std::f64::consts::PI).sqrt());
    for study in &ranking {
        assert!((study.gross_evsi - reference).abs() <= error + 6.0 * study.standard_error + 1e-10);
    }
    let artifact = updated.export(&ledger, ctx).unwrap();
    let replay = JointLawExperimentResult::replay(&artifact, ctx).unwrap();
    assert_fresh_native_law_replay(&artifact, posterior);
    assert_eq!(replay.numerical_law().metadata(), posterior.metadata());
    updated.consume_current(&artifact, &ledger, ctx).unwrap();
    let mut altered: serde_json::Value = serde_json::from_slice(&artifact).unwrap();
    altered["sample_size"] = serde_json::json!(9);
    assert!(updated.consume_current(&serde_json::to_vec(&altered).unwrap(), &ledger, ctx).is_err());
    altered["version"] = serde_json::json!(3);
    assert!(JointLawExperimentResult::replay(&serde_json::to_vec(&altered).unwrap(), ctx).is_err());
    ledger.withdraw_source(experiment, "experiment withdrawn").unwrap();
    assert!(updated.consume_current(&artifact, &ledger, ctx).is_err());
    acceptance_record::record(
        &format!("{family}::bayesian"),
        uncertain,
        translation,
        case,
        &design.dependency_identity().to_hex(),
        serde_json::json!({"constrained_next_study":{"evsi_reference":0.0,"evsi_absolute_error":constrained_ranking[0].gross_evsi.abs(),"quadrature_error_bound":0.0,"evsi_mcse":constrained_ranking[0].standard_error,"admissible_action_identity":constrained_ranking[0].admissible_action_identity.to_hex(),"operational_action_ids":[0],"net_value":constrained_ranking[0].net_value},"admissible_action_identity":ranking[0].admissible_action_identity.to_hex(),"native_posterior_reference":"checked","likelihood_and_posterior_reference":"checked","utility_constraints_ties":"checked","evsi_reference":reference,"evsi_absolute_error":(ranking[0].gross_evsi-reference).abs(),"quadrature_error_bound":error,"evsi_mcse":ranking[0].standard_error,"fresh_original_design_and_update_replay":"checked","withdrawal_and_tamper":"checked"}),
    );
}

#[test]
fn native_randomized_designs_execute_the_experiment_to_decision_reference_chain() {
    let ctx = ExecutionContext::for_tests(77);
    let data = super::super::fixture::source(false);
    let request = super::super::fixture::request(
        &data,
        false,
        antecedent_estimate::joint_panel::PanelInference::Bayesian {
            prior: antecedent_estimate::joint_panel::PanelPrior::default(),
            sampler: super::super::fixture::posterior_sampling(),
        },
    );
    native_randomized_chain(&request.target, &ctx);
}

#[test]
fn longitudinal_treatment_affected_history_executes_the_full_experiment_chain() {
    use antecedent::analysis::joint_panel::designs::{
        JointDesignArm, JointDesignProof, JointDesignWorldBinding, PreparedJointDesign,
        joint_design_world_identity,
    };
    use antecedent_core::assumption::AssumptionSet;
    use antecedent_core::{
        Conditioning, GraphFamily, HistoryWindow, InferenceSelection, InitialLaw, Intervention,
        InterventionContrast, InterventionSequence, InterventionWorld, ModelAssumptions,
        ObservationUnit, OutcomeQuantity, QuantityUnits, ScientificTarget, ScientificWorldId,
        SequencedIntervention, TargetPopulation, TemporalPolicy, TimeFrame, Value, VariableId,
        WorldAction, WorldBinding, WorldCollection,
    };
    use antecedent_estimate::joint_design::{
        compiled::{JointDesignInference, JointDesignRecipe},
        longitudinal::{LongitudinalDirichletPrior, LongitudinalEpisode, LongitudinalJointDesign},
    };
    let ctx = ExecutionContext::for_tests(123);
    let binding = WorldBinding::new(
        TargetPopulation::AllObserved,
        ObservationUnit::new("independent_episode").unwrap(),
        OutcomeQuantity::mean(VariableId::from_raw(1), QuantityUnits::dimensionless()).unwrap(),
        Conditioning::unconditional(),
    )
    .unwrap();
    let time = TimeFrame::temporal(
        2,
        HistoryWindow::planner_default(),
        InitialLaw::contemporaneous_observed(),
    )
    .unwrap();
    let worlds = (0..2)
        .map(|a| {
            let steps = (0..2)
                .map(|t| {
                    SequencedIntervention::new(
                        Intervention::set(VariableId::from_raw(0), Value::f64(f64::from(a))),
                        TemporalPolicy::pulse(t),
                    )
                })
                .collect::<Vec<_>>();
            InterventionWorld::new(
                ScientificWorldId::from_raw(a),
                WorldAction::schedule(
                    Intervention::Sequence(InterventionSequence::new(steps)),
                    TemporalPolicy::sustained(0, 1),
                )
                .unwrap(),
                a == 0,
                binding.clone(),
                time.clone(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let target = ScientificTarget::new(
        TargetForm::Contrast(
            InterventionContrast::difference(
                WorldCollection::new(worlds.clone()).unwrap(),
                ScientificWorldId::from_raw(1),
                ScientificWorldId::from_raw(0),
                ScientificWorldId::from_raw(0),
            )
            .unwrap(),
        ),
        ModelAssumptions::new(GraphFamily::Dag, AssumptionSet::new()),
        InferenceSelection::Bayesian,
    );
    let mut episodes = Vec::new();
    // All full histories have genuine observed support. A0 changes the next
    // covariate distribution (3:1 vs1:3), so freezing L1 would change the target.
    for l0 in 0..2 {
        for a0 in 0..2 {
            for l1 in 0..2 {
                for a1 in 0..2 {
                    for y in 0..2 {
                        let transition_repeats = if l1 == a0 { 6 } else { 2 };
                        let successes = 1 + l1 + a1;
                        let outcome_repeats = if y == 1 { successes } else { 4 - successes };
                        for r in 0..transition_repeats * outcome_repeats {
                            episodes.push(LongitudinalEpisode {
                                id: format!("{l0}:{a0}:{l1}:{a1}:{y}:{r}"),
                                covariates: vec![l0, l1],
                                assignments: vec![a0, a1],
                                outcome: y as f64,
                            });
                        }
                    }
                }
            }
        }
    }
    let design=LongitudinalJointDesign {worlds:vec![("reference".into(),vec![0,0]),("active".into(),vec![1,1])],sequential_exchangeability:"Both assignments are independent of subsequent potential outcomes conditional on the entire retained covariate and assignment history".into(),consistency:"The arm dictionary specifies exact versions of both recorded interventions".into(),sampling_population:"Independent complete episodes from the declared observed target population".into()};
    let source_binding =
        antecedent::analysis::joint_panel::designs::JointDesignSourceBinding::from_world(
            &worlds[0],
            design.sampling_population.clone(),
        )
        .unwrap();
    let proof = JointDesignProof {
        version: 1,
        source_binding,
        recipe: JointDesignRecipe::Longitudinal { design, episodes },
        worlds: worlds
            .iter()
            .zip(["reference", "active"])
            .map(|(w, key)| JointDesignWorldBinding {
                key: key.into(),
                world: w.id().raw(),
                target_identity: joint_design_world_identity(w).to_hex(),
            })
            .collect(),
        longitudinal_arms: vec![
            JointDesignArm { arm: 0, levels: vec![(0, 0.0)] },
            JointDesignArm { arm: 1, levels: vec![(0, 1.0)] },
        ],
    };
    let prepared = PreparedJointDesign::compile(target, proof, &ctx).unwrap();
    let frequentist = prepared.execute(JointDesignInference::Frequentist, &ctx).unwrap();
    assert!((frequentist.law().estimate().means[0] - 0.3125).abs() < 1e-12);
    assert!((frequentist.law().estimate().means[1] - 0.6875).abs() < 1e-12);
    let result = prepared
        .execute(
            JointDesignInference::LongitudinalDirichlet {
                prior: LongitudinalDirichletPrior {
                    covariate_states: vec![vec![0, 1], vec![0, 1]],
                    outcome_values: vec![0.0, 1.0],
                    concentration: 1.0,
                    draws: 512,
                    seed: 123,
                },
            },
            &ctx,
        )
        .unwrap();
    let source = result.posterior_source().unwrap();
    let ScientificSourcePayload::Law(law) = source.as_numerical_source() else {
        panic!("Dirichlet mechanism law");
    };
    // Independent table-count reference. The proper posterior transition mean is
    // 17/66 or49/66, and terminal beta means are7/26,1/2,19/26.
    // Mechanism tables are independent under their explicit product prior, so
    // E[g-formula] is the product of these separately counted posterior means.
    let expected = [
        (49.0 / 66.0) * (7.0 / 26.0) + (17.0 / 66.0) * 0.5,
        (17.0 / 66.0) * 0.5 + (49.0 / 66.0) * (19.0 / 26.0),
    ];
    for (i, reference) in expected.iter().enumerate() {
        assert!((law.mean(i).unwrap() - reference).abs() < 0.025);
    }
    assert!((law.mean(1).unwrap() - law.mean(0).unwrap()) > 0.25);
    exercise_native_source(
        &result,
        "sequential_observational",
        "compiled_world_longitudinal",
        &ctx,
    );
    record_world_refusal(&frequentist, "sequential_observational", "compiled_world_longitudinal");
    prepared.consume_current(&result.export_artifact().unwrap(), &ctx).unwrap();
}

/// Release-candidate measurement of the actual native fit→bridge→update→decision
/// chain. Its finite empirical posterior error is included in the measurement;
/// implementation runs do not authorize a release calibration claim.
#[test]
#[ignore = "Q08/E14 stable-candidate full experiment-chain measurement"]
fn native_experiment_decision_full_procedure_calibration_l95() {
    use antecedent::analysis::joint_panel::designs::{
        JointDesignProof, JointDesignWorldBinding, PreparedJointDesign, joint_design_world_identity,
    };
    use antecedent_estimate::joint_design::{
        compiled::{JointDesignInference, JointDesignRecipe},
        randomized::{JointRandomizedDesign, JointRandomizedUnit, RandomizedGaussianPrior},
    };
    let source_data = super::super::fixture::source(false);
    let template = super::super::fixture::request(
        &source_data,
        false,
        antecedent_estimate::joint_panel::PanelInference::Frequentist { draws: 2 },
    );
    let TargetForm::Contrast(contrast) = template.target.form() else {
        panic!("contrast");
    };
    let worlds = contrast.worlds().worlds();
    let keys = (0..4).map(|i| format!("whole_program_{i}")).collect::<Vec<_>>();
    let mut covered = 0usize;
    let mut decisions_changed = 0usize;
    let repetitions = std::env::var("ANTECEDENT_CALIBRATION_NSIM")
        .ok()
        .map(|v| v.parse::<u64>().expect("ANTECEDENT_CALIBRATION_NSIM must be an integer"))
        .unwrap_or(400);
    assert!(
        repetitions >= 400,
        "stable-candidate full-chain measurement requires at least400 independent data sets"
    );
    for replicate in 0..repetitions {
        let ctx = ExecutionContext::for_tests(13000 + replicate);
        let mut rng = ctx.rng.stream(0x5E1E_C710_0000 + replicate);
        let truth =
            (0..4).map(|_| 2.0 * antecedent_kernels::standard_normal(&mut rng)).collect::<Vec<_>>();
        let units = (0..64)
            .map(|i| {
                let arm = antecedent_kernels::sample_categorical(&mut rng, &[0.25; 4]).unwrap();
                JointRandomizedUnit {
                    id: format!("original:{replicate}:{i}"),
                    arm,
                    probabilities: vec![0.25; 4],
                    outcome: truth[arm] + antecedent_kernels::standard_normal(&mut rng),
                    population_weight: 1.0,
                }
            })
            .collect::<Vec<_>>();
        let source_binding =
            antecedent::analysis::joint_panel::designs::JointDesignSourceBinding::from_world(
                &worlds[0],
                "prespecified finite Gaussian arm-mean target population".into(),
            )
            .unwrap();
        let recipe = JointDesignRecipe::Randomized {
            design: JointRandomizedDesign {
                worlds: keys.clone(),
                sampling_unit: source_binding.sampling_unit.clone(),
                population: "prespecified finite Gaussian arm-mean target population".into(),
                consistency:
                    "Each known arm code is the complete retained multichannel world program".into(),
                between_unit_isolation:
                    "Independent randomized clusters, no between-cluster interference".into(),
            },
            units: units.clone(),
        };
        let proof = JointDesignProof {
            version: 1,
            recipe,
            source_binding,
            worlds: worlds
                .iter()
                .zip(&keys)
                .map(|(w, key)| JointDesignWorldBinding {
                    key: key.clone(),
                    world: w.id().raw(),
                    target_identity: joint_design_world_identity(w).to_hex(),
                })
                .collect(),
            longitudinal_arms: vec![],
        };
        let prepared = PreparedJointDesign::compile(template.target.clone(), proof, &ctx).unwrap();
        let fit = prepared
            .execute(
                JointDesignInference::RandomizedGaussian {
                    prior: RandomizedGaussianPrior {
                        mean: vec![0.0; 4],
                        variance: vec![4.0; 4],
                        outcome_variance: vec![1.0; 64],
                        draws: 2048,
                        seed: 19000 + replicate,
                    },
                },
                &ctx,
            )
            .unwrap();
        let source = fit.posterior_source().unwrap();
        let quantity = JointLawExperimentResult::quantity(&source).unwrap();
        let coefficients = worlds
            .iter()
            .map(|w| contrast.terms().iter().find(|t| t.world == w.id()).unwrap().coefficient)
            .collect::<Vec<_>>();
        let true_effect = truth.iter().zip(&coefficients).map(|(v, c)| v * c).sum::<f64>();
        let observed =
            true_effect + (0.3_f64 / 8.0).sqrt() * antecedent_kernels::standard_normal(&mut rng);
        let mut ledger = EvidenceLedger::new();
        let model = ledger
            .add_source(
                SourceDeclaration::new(
                    source.dependency_identity().to_hex(),
                    SourceKind::FittedModel,
                )
                .with_observations([ObservationId::from_key("original", &replicate.to_le_bytes())]),
            )
            .unwrap();
        let experiment = ledger
            .add_source(
                SourceDeclaration::new("independent_new_experiment", SourceKind::Experiment)
                    .with_observations([ObservationId::from_key(
                        "new-experiment",
                        &replicate.to_le_bytes(),
                    )]),
            )
            .unwrap();
        let mapping = ledger
            .add_source(
                SourceDeclaration::new("identity_translation_reference", SourceKind::FittedModel)
                    .with_observations([ObservationId::from_key(
                        "identity-reference",
                        b"fixed-external-model",
                    )]),
            )
            .unwrap();
        let bridge = declared_bridge(&quantity, 1.0, 0.0, 0.0);
        let request = JointLawExperimentRequest {
            bridge,
            observed_mean: observed,
            observation_variance: 0.3,
            sample_size: 8,
            mapping: PanelExperimentMapping {
                dependency: source.dependency_identity(),
                intercepts: vec![0.0; 2048],
                slopes: vec![1.0; 2048],
            },
            evidence: PanelExperimentEvidence { model, experiment, mapping, prior_origins: vec![] },
        };
        let updated = fit.condition_experiment(request, &ledger, &ctx).unwrap();
        let law = updated.numerical_law_source(&ledger).unwrap();
        let values = law
            .draws()
            .chunks_exact(4)
            .map(|row| row.iter().zip(&coefficients).map(|(v, c)| v * c).sum::<f64>())
            .collect::<Vec<_>>();
        let weights = law.metadata().weights.as_ref().unwrap();
        let mut ordered = values.iter().copied().zip(weights.iter().copied()).collect::<Vec<_>>();
        ordered.sort_by(|a, b| a.0.total_cmp(&b.0));
        let quantile = |p: f64| {
            let mut total = 0.0;
            for (v, w) in &ordered {
                total += w;
                if total >= p {
                    return *v;
                }
            }
            ordered.last().unwrap().0
        };
        covered += usize::from(quantile(0.025) <= true_effect && true_effect <= quantile(0.975));
        let ScientificSourcePayload::Law(prior) = source.as_numerical_source() else {
            panic!("prior law");
        };
        let before = prior
            .draws()
            .chunks_exact(4)
            .map(|row| row.iter().zip(&coefficients).map(|(v, c)| v * c).sum::<f64>())
            .sum::<f64>()
            / 2048.0;
        let after = values.iter().zip(weights).map(|(v, w)| v * w).sum::<f64>();
        decisions_changed += usize::from((before > 0.0) != (after > 0.0));
        let coordinates = law
            .quantities()
            .iter()
            .cloned()
            .map(antecedent_contract::ScientificQuantity::try_from)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let expr = coefficients.iter().enumerate().fold(UtilityExpr::Const(0.0), |e, (i, c)| {
            UtilityExpr::sum(e, UtilityExpr::product(UtilityExpr::Const(*c), UtilityExpr::Input(i)))
        });
        let contract = DecisionContract {
            actions: vec![
                DecisionAction {
                    id: "hold".into(),
                    kind: ActionKind::Policy,
                    inputs: coordinates.clone(),
                    utility: zero_utility(),
                },
                DecisionAction {
                    id: "act".into(),
                    kind: ActionKind::Policy,
                    inputs: coordinates.clone(),
                    utility: expr,
                },
            ],
            utility_units: quantity.units.clone(),
            criterion: DecisionCriterion::PosteriorExpectedUtility,
            constraints: vec![],
            target_population: quantity.population_id.clone(),
            horizon: quantity.horizon,
            structural_policy: StructuralPolicy::ReportOnly,
        };
        let decision = updated.decide(&contract, &ledger).unwrap();
        assert!((decision.result.actions[1].expected_utility - after).abs() < 1e-10);
        let problem = DecisionProblem::new(
            vec![0, 1],
            Arc::new(ContrastUtility { coefficients, threshold: 0.0 }),
            vec![],
        );
        let ranking = updated
            .rank_next_samples(
                &problem,
                &[
                    PanelFutureSample {
                        id: "cheap".into(),
                        sample_size: 4,
                        observation_variance: 1.0,
                        cost_utility: Some(0.01),
                    },
                    PanelFutureSample {
                        id: "expensive".into(),
                        sample_size: 4,
                        observation_variance: 1.0,
                        cost_utility: Some(0.1),
                    },
                ],
                16,
                &ledger,
                &ctx,
            )
            .unwrap();
        assert_eq!(ranking[0].id, "cheap");
        let artifact = updated.export(&ledger, &ctx).unwrap();
        updated.consume_current(&artifact, &ledger, &ctx).unwrap();
        prepared.consume_current(&fit.export_artifact().unwrap(), &ctx).unwrap();
    }
    let coverage = covered as f64 / repetitions as f64;
    eprintln!(
        "native_experiment_decision_full_procedure: seeds={repetitions} covered={covered} coverage={coverage} changed_decisions={decisions_changed}; release license requires stable candidate and registered gate measurements"
    );
    assert!(
        (0.90..=0.99).contains(&coverage),
        "unmeasured candidate failed the declared broad calibration tolerance"
    );
    assert!(
        decisions_changed > 0,
        "new independent evidence must sometimes change actual terminal actions"
    );
}

#[test]
fn experimental_translation_can_change_real_endpoint_population_and_horizon_only_with_transport() {
    let quantity = antecedent_contract::ScientificQuantity {
        variable_id: "outcome".into(),
        variable_name: "outcome".into(),
        role: antecedent_contract::QuantityRole::Outcome,
        units: "utility_unit".into(),
        population_id: "target_population".into(),
        regime_id: "target_joint_regime".into(),
        horizon: 24,
        functional_id: "joint_contrast".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    };
    let mut translated = declared_bridge(&quantity, 1.0, 0.0, 0.0);
    translated.source.quantity.population_id = "historical_trial_population".into();
    translated.source.quantity.horizon = 2;
    translated.evidence_scope.population_id = "historical_trial_population".into();
    let estimand = translated.estimand_bridge.as_mut().unwrap();
    estimand.experiment.population_id = "historical_trial_population".into();
    estimand.experiment.time_start = "historical_period:0".into();
    estimand.experiment.time_end = "historical_period:1".into();
    estimand.experiment.duration = 2.0;
    translated.transport=Some(TransportabilityPremise {id:"population_and_horizon_model".into(),statement:"The supplied affine map carries this historical trial response to the target population at period24".into(),established_in:translated.evidence_scope.clone(),applies_to:BridgeScope {population_id:quantity.population_id.clone(),regime_id:quantity.regime_id.clone()},evidence:Some("separately_established_translation_model".into())});
    assert!(translated.check_licensing().unwrap().transported);
    let mut no_transport = translated.clone();
    no_transport.transport = None;
    assert!(no_transport.check_licensing().is_err());
    let mut wrong_scope = translated.clone();
    wrong_scope.transport.as_mut().unwrap().applies_to.population_id = "another_population".into();
    assert!(wrong_scope.check_licensing().is_err());
    let mut wrong_source = translated.clone();
    wrong_source.source.quantity.population_id = "unbound_population".into();
    assert!(wrong_source.check_licensing().is_err());
    let mut missing_translation = translated.clone();
    missing_translation.estimand_bridge = None;
    assert!(missing_translation.check_licensing().is_err());
    let mut horizon_only = translated;
    horizon_only.source.quantity.population_id = quantity.population_id.clone();
    horizon_only.evidence_scope.population_id = quantity.population_id.clone();
    horizon_only.estimand_bridge.as_mut().unwrap().experiment.population_id =
        quantity.population_id.clone();
    horizon_only.transport.as_mut().unwrap().established_in = horizon_only.evidence_scope.clone();
    assert!(horizon_only.check_licensing().unwrap().transported);
    horizon_only.transport = None;
    assert!(
        horizon_only.check_licensing().is_err(),
        "same-population horizon extrapolation still needs its explicit transport model"
    );
}

fn assert_fresh_native_law_replay(
    bytes: &[u8],
    law: &antecedent_io::distribution_artifact::DistributionArtifact,
) {
    static SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let sequence = SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let directory = std::env::temp_dir()
        .join(format!("antecedent-native-experiment-replay-{}-{sequence}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    std::fs::write(directory.join("execution.json"), bytes).unwrap();
    let expected = (
        law.draws().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        law.metadata().weights.as_ref().unwrap().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
    );
    std::fs::write(directory.join("expected.json"), serde_json::to_vec(&expected).unwrap())
        .unwrap();
    // The child must select this very test; its libtest name carries the module path.
    let prefix =
        module_path!().split_once("::").map_or(String::new(), |(_, rest)| format!("{rest}::"));
    let test_name = format!("{prefix}native_law_fresh_process_consumer");
    let child = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", &test_name, "--nocapture"])
        .env("ANTECEDENT_NATIVE_EXPERIMENT_REPLAY", &directory)
        .output()
        .unwrap();
    assert!(
        child.status.success(),
        "independent native experiment replay: {}",
        String::from_utf8_lossy(&child.stderr)
    );
    assert!(
        String::from_utf8_lossy(&child.stdout).contains("running 1 test"),
        "the child selected no test: {}",
        String::from_utf8_lossy(&child.stdout)
    );
    assert!(directory.join("consumer-ran").exists());
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn native_law_fresh_process_consumer() {
    let Some(directory) = std::env::var_os("ANTECEDENT_NATIVE_EXPERIMENT_REPLAY") else {
        return;
    };
    let directory = std::path::PathBuf::from(directory);
    let bytes = std::fs::read(directory.join("execution.json")).unwrap();
    let expected: (Vec<u64>, Vec<u64>) =
        serde_json::from_slice(&std::fs::read(directory.join("expected.json")).unwrap()).unwrap();
    let result = JointLawExperimentResult::replay(&bytes, &ExecutionContext::for_tests(1)).unwrap();
    assert!(!result.grants_current_authority());
    assert!(
        result.replays_original_design(),
        "fresh native artifact must replay the original data/design/posterior recipe"
    );
    assert_eq!(
        result.numerical_law().draws().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        expected.0
    );
    assert_eq!(
        result
            .numerical_law()
            .metadata()
            .weights
            .as_ref()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        expected.1
    );
    std::fs::write(directory.join("consumer-ran"), b"verified").unwrap();
}

fn zero_utility() -> UtilityExpr {
    UtilityExpr::difference(UtilityExpr::Input(0), UtilityExpr::Input(0))
}

fn record_world_refusal(
    design: &antecedent::analysis::joint_panel::designs::JointDesignResult,
    family: &str,
    case: &str,
) {
    for uncertain in [false, true] {
        for translation in 0..5 {
            assert!(
                design.posterior_source().is_err(),
                "sampling distribution is not a Bayesian prior"
            );
            acceptance_record::record(
                &format!("{family}::frequentist"),
                uncertain,
                translation,
                case,
                &design.dependency_identity().to_hex(),
                serde_json::json!({"sampling_law_prior_guard":"expected_scoped_refusal","downstream_stages":"not_applicable_sampling_law_is_not_a_prior"}),
            );
        }
    }
}
