//! Independent references for the compiled event source and world composition.

use antecedent::analysis::events::{
    Arrival, EventBelief, EventBinding, EventError, EventForecastRequest, EventHistory,
    EventHypothesis, EventLawMeaning, EventObservation, EventSource, EventWindow,
    EventWorldRequest, HawkesFitSpec, HawkesParameters, PathFunctional, PathOutput,
};
use antecedent::planner::PlanRoute;
use antecedent::{ExecutionContext, Study};
use antecedent_model::mechanism_expr::MechanismExpression;
use antecedent_model::nonlinear_sde::{
    SdeAction, SdeDiffusion, SdeInput, SdeInterpolation, SdeScheme, SdeSimulationPolicy, SdeSystem,
    SdeWorld,
};

fn history(end: f64, events: Vec<Arrival>) -> EventHistory {
    EventHistory {
        catalog: vec!["mention".into()],
        time_unit: "hour".into(),
        origin: "origin".into(),
        source: "news".into(),
        empty_origin: 0.0,
        window: EventWindow { start: 0.0, end },
        left: vec![],
        events,
        gaps: vec![],
    }
}

fn parameters(baseline: f64, branching: f64) -> HawkesParameters {
    HawkesParameters {
        baseline: vec![baseline],
        branching: vec![branching],
        decay: vec![1.0],
    }
}

#[test]
fn conditional_likelihood_matches_poisson_and_direct_asymmetric_reference() {
    let ctx = ExecutionContext::for_tests(0);
    let observed = history(
        2.0,
        vec![
            Arrival { time: 0.5, kind: 0 },
            Arrival { time: 1.5, kind: 0 },
        ],
    );
    let poisson =
        Study::prepare_event_likelihood(observed.clone(), parameters(2.0, 0.0), &ctx).unwrap();
    assert_eq!(poisson.plan_view().route(), PlanRoute::EventProcess);
    assert!((poisson.execute(&ctx).unwrap() - (2.0 * 2.0_f64.ln() - 4.0)).abs() < 1e-13);
    let mut two = observed;
    two.catalog.push("reshare".into());
    two.events[1].kind = 1;
    let rates = HawkesParameters {
        baseline: vec![2.0, 3.0],
        branching: vec![0.0, 0.1, 0.4, 0.0],
        decay: vec![1.0, 2.0, 4.0, 1.0],
    };
    let expected = 2.0_f64.ln() + (3.0 + 0.4 * 4.0 * (-4.0_f64).exp()).ln()
        - 10.0
        - 0.4 * (1.0 - (-6.0_f64).exp())
        - 0.1 * (1.0 - (-1.0_f64).exp());
    let actual = Study::prepare_event_likelihood(two, rates, &ctx)
        .unwrap()
        .execute(&ctx)
        .unwrap();
    assert!((actual - expected).abs() < 1e-13);
}

fn fixture_request(paths: Vec<Vec<Arrival>>) -> EventForecastRequest {
    EventForecastRequest {
        history: history(1.0, vec![]),
        forecast: EventWindow {
            start: 1.0,
            end: 3.0,
        },
        paths: paths.len(),
        source: EventSource::Fixture { paths },
        max_events: 128,
        seed: 51,
    }
}

fn worlds() -> EventWorldRequest {
    let path = |world: &str, functional, units: &str| PathOutput {
        state: "x".into(),
        world: world.into(),
        variable_id: "example.x".into(),
        units: units.into(),
        functional,
    };
    EventWorldRequest {
        system: SdeSystem {
            variables: vec!["x".into()],
            inputs: vec![SdeInput {
                name: "news".into(),
                interpolation: SdeInterpolation::PiecewiseConstant,
                times: vec![1.0],
                values: vec![0.0],
                envelope: Some([0.0, 16.0]),
            }],
            parameters: vec![],
            drift: vec![MechanismExpression::Variable { variable: 1 }],
            diffusion: vec![SdeDiffusion {
                expression: MechanismExpression::Constant { value: 0.0 },
                square_root: false,
            }],
            boundaries: vec![],
        },
        population: "example".into(),
        horizon: 2,
        time_unit: "hour".into(),
        origin: "origin".into(),
        policy: SdeSimulationPolicy {
            start: 1.0,
            step: 1.0,
            steps: 2,
            paths: 2,
            output_every: 1,
            scheme: SdeScheme::EulerMaruyama,
            seed: 92,
            weak_tolerance: 0.01,
            check_step: true,
            retain_final_samples: false,
        },
        initial: vec![0.0],
        worlds: vec![SdeWorld {
            id: "hold".into(),
            actions: vec![SdeAction {
                variable: 0,
                times: vec![2.0],
                values: vec![0.25],
            }],
            inputs: vec![],
        }],
        bindings: vec![EventBinding {
            kind: 0,
            input: "news".into(),
            magnitude: 1.0,
            decay: 1.0,
        }],
        outputs: vec![
            path("baseline", PathFunctional::Maximum, "score"),
            path("baseline", PathFunctional::Integral, "score-hour"),
            path(
                "baseline",
                PathFunctional::Exceeds { threshold: 0.4 },
                "dimensionless",
            ),
            path("hold", PathFunctional::Maximum, "score"),
        ],
    }
}

#[test]
fn fixture_worlds_match_off_grid_path_oracle_and_hawkes_composes() {
    let ctx = ExecutionContext::for_tests(0);
    let request = fixture_request(vec![
        vec![Arrival {
            time: 1.25,
            kind: 0,
        }],
        vec![],
    ]);
    let plan = Study::prepare_event_forecast(request.clone(), None, &ctx).unwrap();
    let law = plan.execute(&ctx).unwrap();
    assert_eq!(law.identity(), plan.execute(&ctx).unwrap().identity());
    let result = Study::prepare_event_worlds(&law, worlds(), None, &ctx)
        .unwrap()
        .execute(&ctx)
        .unwrap();
    // Euler drift reads times 1 and 2. The input arrives at 1.25, so the three
    // coarse states are 0, 0 and exp(-0.75). The held world is 0, .25, .25.
    let peak = (-0.75_f64).exp();
    let expected = [peak, peak / 2.0, 1.0, 0.25, 0.0, 0.0, 0.0, 0.25];
    for (&value, truth) in result.distribution.draws().iter().zip(expected) {
        assert!((value - truth).abs() < 1e-13);
    }
    assert!(result.distribution.alignment_identity().is_ok());
    let mut native = request;
    native.source = EventSource::Declared {
        parameters: parameters(0.0, 0.2),
    };
    let new_law = Study::prepare_event_forecast(native, None, &ctx)
        .unwrap()
        .execute(&ctx)
        .unwrap();
    assert_eq!(new_law.meaning(), EventLawMeaning::Fixed);
    let new_result = Study::prepare_event_worlds(&new_law, worlds(), None, &ctx)
        .unwrap()
        .execute(&ctx)
        .unwrap();
    assert_ne!(
        new_result.distribution.alignment_identity().unwrap(),
        result.distribution.alignment_identity().unwrap()
    );
    assert_eq!(new_result.distribution.draws()[0], 0.0);
}

#[test]
fn rejects_leakage_unstable_parameters_and_unverified_external_source() {
    let ctx = ExecutionContext::for_tests(0);
    let mut request = fixture_request(vec![vec![], vec![]]);
    request.source = EventSource::Declared {
        parameters: parameters(1.0, 1.0),
    };
    assert!(matches!(
        Study::prepare_event_forecast(request.clone(), None, &ctx),
        Err(EventError::Unstable)
    ));
    request.source = EventSource::Declared {
        parameters: parameters(1.0, 0.0),
    };
    request.history.events.push(Arrival { time: 1.0, kind: 0 });
    assert!(matches!(
        Study::prepare_event_forecast(request, None, &ctx),
        Err(EventError::Invalid(_))
    ));
    let mut request = fixture_request(vec![vec![], vec![]]);
    request.source = EventSource::External {
        identity: antecedent_contract::ProviderObjectIdentity {
            provider_id: "provider".into(),
            object_id: "paths".into(),
            version_id: "v1".into(),
            snapshot_id: "snapshot".into(),
            request_id: "request".into(),
        },
        paths: vec![vec![], vec![]],
    };
    assert!(matches!(
        Study::prepare_event_forecast(request, None, &ctx),
        Err(EventError::Unverified)
    ));
}

#[test]
fn actual_update_matches_poisson_bayes_factor_and_rejects_reuse() {
    let ctx = ExecutionContext::for_tests(0);
    let first = EventBelief {
        history: history(1.0, vec![]),
        hypotheses: vec![
            EventHypothesis {
                weight: 0.5,
                parameters: parameters(1.0, 0.0),
            },
            EventHypothesis {
                weight: 0.5,
                parameters: parameters(2.0, 0.0),
            },
        ],
    };
    let observation = EventObservation {
        window: EventWindow {
            start: 1.0,
            end: 2.0,
        },
        source: "news".into(),
        events: vec![
            Arrival {
                time: 1.25,
                kind: 0,
            },
            Arrival {
                time: 1.75,
                kind: 0,
            },
        ],
    };
    let forecast = EventWindow {
        start: 2.0,
        end: 3.0,
    };
    let (law, updated) =
        Study::prepare_event_update(first, observation.clone(), forecast, 16, 64, 43, &ctx)
            .unwrap()
            .execute_with_belief(&ctx)
            .unwrap();
    let odds = 4.0 / std::f64::consts::E;
    assert!((updated.hypotheses[1].weight - odds / (1.0 + odds)).abs() < 1e-13);
    assert_eq!(law.paths().len(), 16);
    assert!(Study::prepare_event_update(updated, observation, forecast, 16, 64, 43, &ctx).is_err());
}

#[test]
fn fixed_decay_fit_recovers_independent_cluster_fixture() {
    let ctx = ExecutionContext::for_tests(0);
    let (training, _) = common_truth::branching_history(77432, 2500.0, 2501.0);
    let mut request = fixture_request(vec![vec![], vec![]]);
    request.history = training;
    request.forecast = EventWindow {
        start: 2500.0,
        end: 2501.0,
    };
    request.source = EventSource::Fitted {
        fit: HawkesFitSpec {
            decay: vec![1.0],
            column_cap: 0.90,
            iterations: 1400,
        },
        refits: 0,
    };
    let law = Study::prepare_event_forecast(request, None, &ctx)
        .unwrap()
        .execute(&ctx)
        .unwrap();
    let fit = law.fit().unwrap();
    assert!(!fit.weak_identification);
    assert!((fit.parameters.baseline[0] - 0.9).abs() < 0.14);
    assert!((fit.parameters.branching[0] - 0.45).abs() < 0.12);
}

#[path = "events_support/cluster_oracle.rs"]
mod common_truth;

#[test]
fn unmeasured_count_interval_is_withheld() {
    let ctx = ExecutionContext::for_tests(0);
    let request = fixture_request(vec![vec![], vec![]]);
    let plan = Study::prepare_event_forecast(request, None, &ctx).unwrap();
    let error = plan.count_interval(0, &ctx).unwrap_err();
    assert!(matches!(error, EventError::IntervalUnmeasured));
    assert_eq!(error.reason_code(), Some("not_supported"));
}

#[test]
fn asymmetric_stationary_rates_match_two_type_reference() {
    let ctx = ExecutionContext::for_tests(0);
    let mut history = history(1.0, vec![]);
    history.catalog.push("reshare".into());
    let request = EventForecastRequest {
        history,
        forecast: EventWindow {
            start: 1.0,
            end: 101.0,
        },
        source: EventSource::Declared {
            parameters: HawkesParameters {
                baseline: vec![0.5, 1.0],
                branching: vec![0.1, 0.2, 0.6, 0.1],
                decay: vec![2.0; 4],
            },
        },
        paths: 512,
        max_events: 1000,
        seed: 45,
    };
    let law = Study::prepare_event_forecast(request, None, &ctx)
        .unwrap()
        .execute(&ctx)
        .unwrap();
    // Solve .9 x - .2 y = .5 and -.6 x + .9 y = 1 independently.
    let expected = [0.65 / 0.69, 1.2 / 0.69];
    let mut counts = [0.0; 2];
    for path in law.paths() {
        for event in path.iter().filter(|event| event.time >= 21.0) {
            counts[event.kind] += 1.0;
        }
    }
    for (count, target) in counts.into_iter().zip(expected) {
        assert!((count / (80.0 * 512.0) - target).abs() < 0.035);
    }
}

#[test]
fn same_stream_signal_changes_compiled_terminal_choice() {
    use antecedent::analysis::events::EventStudyRequest;
    use antecedent_design::{
        ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy,
        UtilityExpr,
    };
    let ctx = ExecutionContext::for_tests(0);
    let mut world = worlds();
    world.policy.paths = 256;
    world.policy.start = 2.0;
    world.policy.step = 0.25;
    world.policy.steps = 4;
    world.worlds[0].actions[0].times = vec![2.0];
    let open = DecisionAction {
        id: "baseline".into(),
        kind: ActionKind::Regime,
        inputs: vec![world.outputs[0].quantity(&world.population, world.horizon)],
        utility: UtilityExpr::Neg(Box::new(UtilityExpr::Input(0))),
    };
    let hold = DecisionAction {
        id: "hold".into(),
        kind: ActionKind::Intervention,
        inputs: vec![world.outputs[3].quantity(&world.population, world.horizon)],
        utility: UtilityExpr::Neg(Box::new(UtilityExpr::Input(0))),
    };
    let decision = DecisionContract {
        actions: vec![open, hold],
        utility_units: "score".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: world.population.clone(),
        horizon: world.horizon,
        structural_policy: StructuralPolicy::ReportOnly,
    };
    let prior = EventBelief {
        history: history(1.0, vec![]),
        hypotheses: vec![EventHypothesis {
            weight: 1.0,
            parameters: parameters(0.4, 0.2),
        }],
    };
    let request = EventStudyRequest {
        decision_time: 2.0,
        end: 3.0,
        kind: 0,
        minimum_count: 1,
        paths: 256,
        max_events: 128,
        seed: 54,
    };
    let result = Study::prepare_event_study(prior, request, world, decision, None, &ctx)
        .unwrap()
        .execute(&ctx)
        .unwrap();
    assert_eq!(
        result.branch_actions,
        [Some("baseline".into()), Some("hold".into())]
    );
    assert!(result.gross_value > 0.01);
}

#[test]
fn holdout_changes_scores_without_changing_the_forecast_at_cutoff() {
    use antecedent::analysis::events::{ForecastAssessmentRequest, ForecastOrigin};
    let ctx = ExecutionContext::for_tests(0);
    let (history, _) = common_truth::branching_history(83761, 310.0, 312.0);
    let mut request = ForecastAssessmentRequest {
        history,
        fit: HawkesFitSpec {
            decay: vec![1.0],
            column_cap: 0.9,
            iterations: 1400,
        },
        origins: vec![
            ForecastOrigin {
                cutoff: 300.0,
                end: 302.0,
                validation: true,
            },
            ForecastOrigin {
                cutoff: 308.0,
                end: 310.0,
                validation: false,
            },
        ],
        kind: 0,
        minimum_count: 1,
        paths: 128,
        max_events: 128,
        seed: 31,
    };
    let original = Study::prepare_event_assessment(request.clone(), &ctx)
        .unwrap()
        .execute(&ctx)
        .unwrap();
    let insert = request
        .history
        .events
        .partition_point(|event| event.time < 309.125);
    request.history.events.insert(
        insert,
        Arrival {
            time: 309.125,
            kind: 0,
        },
    );
    let changed = Study::prepare_event_assessment(request, &ctx)
        .unwrap()
        .execute(&ctx)
        .unwrap();
    let (before, after) = (&original.scores[1], &changed.scores[1]);
    assert_eq!(before.training_events, after.training_events);
    assert_eq!(before.threshold_probability, after.threshold_probability);
    assert_eq!(before.observed_count + 1, after.observed_count);
    assert_ne!(before.log_likelihood, after.log_likelihood);
}

#[test]
fn verified_external_events_bind_payload_and_provider_version() {
    use antecedent::analysis::events::external_execution_contract;
    use antecedent_contract::{
        ProviderObjectIdentity, VerificationProbe, VerificationProbeKind as Probe,
        VerificationSubject, verify_provider_requests,
    };
    let ctx = ExecutionContext::for_tests(0);
    let mut request = fixture_request(vec![
        vec![Arrival {
            time: 1.25,
            kind: 0,
        }],
        vec![],
    ]);
    request.source = EventSource::External {
        identity: ProviderObjectIdentity {
            provider_id: "event provider".into(),
            object_id: "paths".into(),
            version_id: "v1".into(),
            snapshot_id: "snapshot".into(),
            request_id: "pending".into(),
        },
        paths: vec![
            vec![Arrival {
                time: 1.25,
                kind: 0,
            }],
            vec![],
        ],
    };
    let request_id = request.provider_request_id().unwrap();
    let EventSource::External { identity, .. } = &mut request.source else {
        unreachable!()
    };
    identity.request_id = request_id;
    let declaration = external_execution_contract(identity.clone());
    let probes =
        [Probe::KnownTruth, Probe::Shape, Probe::SeededBehavior].map(|kind| VerificationProbe {
            kind,
            observed: 1.0,
            expected: 1.0,
            tolerance: 0.0,
        });
    let receipt = verify_provider_requests(
        "event fixture oracle",
        &[(VerificationSubject::Execution(&declaration), &probes)],
    )
    .unwrap();
    let law = Study::prepare_event_forecast(request.clone(), Some(&receipt), &ctx)
        .unwrap()
        .execute(&ctx)
        .unwrap();
    assert_eq!(law.meaning(), EventLawMeaning::VerifiedExternal);
    assert_eq!(law.provider_receipt(), Some(receipt.digest().as_str()));
    let EventSource::External { identity, .. } = &mut request.source else {
        unreachable!()
    };
    identity.version_id = "v2".into();
    assert!(matches!(
        Study::prepare_event_forecast(request, Some(&receipt), &ctx),
        Err(EventError::Unverified)
    ));
}
