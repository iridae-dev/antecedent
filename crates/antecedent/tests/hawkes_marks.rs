//! Horizon, mark and structural-reading checks for the Hawkes compiler.

use antecedent::Study;
use antecedent::analysis::events::{
    EventBinding, EventForecastRequest, EventHistory, EventSource, EventWindow, EventWorldRequest,
    PathFunctional, PathOutput,
};
use antecedent::prelude::ExecutionContext;
use antecedent_model::mechanism_expr::MechanismExpression;
use antecedent_model::nonlinear_sde::{
    SdeDiffusion, SdeInput, SdeInterpolation, SdeScheme, SdeSimulationPolicy, SdeSystem, SdeWorld,
};

fn history() -> EventHistory {
    EventHistory {
        catalog: vec!["mention".into()],
        time_unit: "hour".into(),
        origin: "origin".into(),
        source: "news".into(),
        empty_origin: 0.0,
        window: EventWindow { start: 0.0, end: 1.0 },
        left: vec![],
        events: vec![],
        gaps: vec![],
    }
}

#[test]
fn sixteen_intervention_worlds_stay_inside_the_sde_world_cap() {
    let ctx = ExecutionContext::for_tests(0);
    let request = EventForecastRequest {
        history: history(),
        forecast: EventWindow { start: 1.0, end: 3.0 },
        source: EventSource::Fixture { paths: vec![vec![], vec![]] },
        paths: 2,
        max_events: 8,
        seed: 7,
    };
    let law = Study::prepare_event_forecast(request, None, &ctx).unwrap().execute(&ctx).unwrap();
    let worlds = EventWorldRequest {
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
        worlds: (0..16)
            .map(|i| SdeWorld { id: format!("hold-{i}"), actions: vec![], inputs: vec![] })
            .collect(),
        bindings: vec![EventBinding { kind: 0, input: "news".into(), magnitude: 1.0, decay: 1.0 }],
        outputs: vec![PathOutput {
            state: "x".into(),
            world: "baseline".into(),
            variable_id: "example.x".into(),
            units: "score".into(),
            functional: PathFunctional::Maximum,
        }],
    };
    Study::prepare_event_worlds(&law, worlds, None, &ctx).unwrap();
}

mod compiled_programs {
    //! Independent checks of the marked, offspring and immigrant-challenge programs. Each
    //! expected value is written out here from the defining formula of the conditional
    //! intensity: a directly summed likelihood, a closed-form mean count or offspring
    //! functional, or the maximum-likelihood point of a piecewise-constant Poisson rate.

    use antecedent::Study;
    use antecedent::analysis::events::{
        Arrival, BaselineFeature, DiagnosticStatus, EventError, EventHistory, EventWindow,
        ExcitationChallenge, ExcitationSubject, ExcludedShock, HawkesFitSpec, HawkesParameters,
        MarkPrototype, MarkedArrival, MarkedFitSpec, MarkedHistory, MarkedParameters,
        ReleaseComparisonRequest,
    };
    use antecedent::prelude::ExecutionContext;

    /// Two-type catalog arrivals on `[0, 4)`.
    const ARRIVALS: [(f64, usize); 7] =
        [(0.3, 0), (0.9, 1), (1.4, 0), (2.2, 1), (2.25, 0), (3.1, 1), (3.7, 0)];
    /// Baseline of each type.
    const BASELINE: [f64; 2] = [0.5, 0.8];
    /// Lifetime offspring of target `t` from source `s`, at `[t][s]`.
    const BRANCHING: [[f64; 2]; 2] = [[0.2, 0.1], [0.3, 0.25]];
    /// Decay of each source type.
    const DECAY: [f64; 2] = [1.5, 0.7];
    const END: f64 = 4.0;

    /// The conditional log likelihood summed term by term: the log intensity at each arrival,
    /// less the integrated intensity of every type over the window.
    fn direct_log_likelihood() -> f64 {
        let mut total = 0.0;
        for (i, &(time, kind)) in ARRIVALS.iter().enumerate() {
            let mut intensity = BASELINE[kind];
            for &(earlier, source) in &ARRIVALS[..i] {
                intensity += BRANCHING[kind][source]
                    * DECAY[source]
                    * (-DECAY[source] * (time - earlier)).exp();
            }
            total += intensity.ln();
        }
        for target in 0..2 {
            total -= BASELINE[target] * END;
            for &(time, source) in &ARRIVALS {
                total -= BRANCHING[target][source] * (1.0 - (-DECAY[source] * (END - time)).exp());
            }
        }
        total
    }

    #[test]
    fn one_hot_marks_reproduce_the_directly_summed_catalog_likelihood() {
        let ctx = ExecutionContext::for_tests(0);
        let expected = direct_log_likelihood();
        let one_hot = |k: usize| MarkPrototype {
            source: (0..2).map(|j| f64::from(u8::from(j == k))).collect(),
            target: (0..2).map(|j| f64::from(u8::from(j == k))).collect(),
            mass: 1.0,
        };
        let marked = MarkedHistory {
            time_unit: "hour".into(),
            origin: "origin".into(),
            source: "log".into(),
            empty_origin: 0.0,
            window: EventWindow { start: 0.0, end: END },
            left: vec![],
            events: ARRIVALS
                .iter()
                .map(|&(time, kind)| MarkedArrival { time, prototype: kind })
                .collect(),
            prototypes: vec![one_hot(0), one_hot(1)],
        };
        let parameters = MarkedParameters {
            baseline: BASELINE.to_vec(),
            excitation: BRANCHING.iter().flatten().copied().collect(),
            decay: DECAY.to_vec(),
        };
        let compiled = Study::prepare_marked_likelihood(marked, parameters, &ctx).unwrap();
        assert_eq!(compiled.plan_view().route().as_str(), "MarkedEventProcess");
        let value = compiled.execute(&ctx).unwrap();
        assert!((value - expected).abs() < 1e-9, "marked {value} against direct {expected}");

        let catalog = EventHistory {
            catalog: vec!["a".into(), "b".into()],
            time_unit: "hour".into(),
            origin: "origin".into(),
            source: "log".into(),
            empty_origin: 0.0,
            window: EventWindow { start: 0.0, end: END },
            left: vec![],
            events: ARRIVALS.iter().map(|&(time, kind)| Arrival { time, kind }).collect(),
            gaps: vec![],
        };
        let parameters = HawkesParameters {
            baseline: BASELINE.to_vec(),
            branching: BRANCHING.iter().flatten().copied().collect(),
            decay: vec![DECAY[0], DECAY[1], DECAY[0], DECAY[1]],
        };
        let value = Study::prepare_event_likelihood(catalog, parameters, &ctx)
            .unwrap()
            .execute(&ctx)
            .unwrap();
        assert!((value - expected).abs() < 1e-9, "catalog {value} against direct {expected}");
    }

    #[test]
    fn marked_draws_match_the_closed_form_mean_count() {
        // One mark: intensity mu + alpha beta sum exp(-beta (t - s)) from an empty start. The
        // mean intensity solves m' = -beta (1 - alpha) (m - mu / (1 - alpha)), so the mean count
        // on [0, T) is mu T / (1 - alpha) - alpha mu (1 - exp(-beta (1 - alpha) T)) / (beta
        // (1 - alpha)^2).
        let ctx = ExecutionContext::for_tests(0);
        let (mu, alpha, beta, horizon) = (2.0_f64, 0.5_f64, 2.0_f64, 3.0_f64);
        let history = MarkedHistory {
            time_unit: "hour".into(),
            origin: "origin".into(),
            source: "log".into(),
            empty_origin: 0.0,
            window: EventWindow { start: 0.0, end: 1.0 },
            left: vec![],
            events: vec![],
            prototypes: vec![MarkPrototype { source: vec![1.0], target: vec![1.0], mass: 1.0 }],
        };
        let parameters =
            MarkedParameters { baseline: vec![mu], excitation: vec![alpha], decay: vec![beta] };
        let paths = 4000;
        let drawn = Study::prepare_marked_draw(
            history,
            parameters,
            EventWindow { start: 1.0, end: 1.0 + horizon },
            paths,
            200,
            17,
            &ctx,
        )
        .unwrap()
        .execute(&ctx)
        .unwrap();
        assert_eq!(drawn.len(), paths);
        let mean = drawn.iter().map(Vec::len).sum::<usize>() as f64 / paths as f64;
        let rate = 1.0 - alpha;
        let expected = mu * horizon / rate
            - alpha * mu * (1.0 - (-beta * rate * horizon).exp()) / (beta * rate * rate);
        // The count's variance is below mu T / (1 - alpha)^3 = 48, so its standard error is
        // below 0.11 over 4000 paths.
        assert!((mean - expected).abs() < 0.6, "mean count {mean} against {expected}");
    }

    #[test]
    fn compiled_offspring_match_the_closed_forms() {
        let ctx = ExecutionContext::for_tests(0);
        let (alpha, beta, horizon) = (0.4_f64, 1.0_f64, 2.0_f64);
        let univariate =
            HawkesParameters { baseline: vec![0.5], branching: vec![alpha], decay: vec![beta] };
        let compiled = Study::prepare_excitation_offspring(univariate, 0, horizon, &ctx).unwrap();
        assert_eq!(compiled.plan_view().route().as_str(), "HawkesExcitation");
        let offspring = compiled.execute(&ctx).unwrap();
        // Descendant intensity of one event is alpha beta exp(-beta (1 - alpha) t).
        let direct = alpha * (1.0 - (-beta * horizon).exp());
        let lifetime = alpha / (1.0 - alpha);
        let within = lifetime * (1.0 - (-beta * (1.0 - alpha) * horizon).exp());
        assert!((offspring.direct[0] - direct).abs() < 1e-12);
        assert!((offspring.lifetime_direct[0] - alpha).abs() < 1e-12);
        assert!((offspring.lifetime_descendants[0] - lifetime).abs() < 1e-9);
        assert!((offspring.horizon_descendants[0] - within).abs() < 1e-6);

        let marked = Study::prepare_marked_offspring(
            MarkedParameters { baseline: vec![0.5], excitation: vec![alpha], decay: vec![beta] },
            vec![MarkPrototype { source: vec![1.0], target: vec![1.0], mass: 1.0 }],
            0,
            horizon,
            &ctx,
        )
        .unwrap()
        .execute(&ctx)
        .unwrap();
        assert!((marked.direct[0] - direct).abs() < 1e-12);
        assert!((marked.lifetime_descendants[0] - lifetime).abs() < 1e-9);
        assert!((marked.horizon_descendants[0] - within).abs() < 1e-6);

        // Two types: the lifetime descendants of source 0 are the first column of
        // (I - A)^{-1} less the initiator, by the 2x2 inverse.
        let pair = HawkesParameters {
            baseline: vec![0.2, 0.3],
            branching: vec![0.1, 0.2, 0.3, 0.05],
            decay: vec![1.0, 2.0, 1.5, 0.5],
        };
        let offspring =
            Study::prepare_excitation_offspring(pair, 0, 1.0, &ctx).unwrap().execute(&ctx).unwrap();
        let determinant = 0.9 * 0.95 - 0.2 * 0.3;
        assert!((offspring.lifetime_descendants[0] - (0.95 / determinant - 1.0)).abs() < 1e-9);
        assert!((offspring.lifetime_descendants[1] - 0.3 / determinant).abs() < 1e-9);
        assert!((offspring.direct[1] - 0.3 * (1.0 - (-1.5_f64).exp())).abs() < 1e-12);
    }

    /// Five arrivals one apart on [0, 5), then fifteen a third apart on [5, 10).
    fn stepped_history() -> EventHistory {
        let first = (0..5).map(|i| f64::from(i) + 0.5);
        let second = (0..15).map(|j| 5.0 + (f64::from(j) + 0.5) / 3.0);
        EventHistory {
            catalog: vec!["post".into()],
            time_unit: "hour".into(),
            origin: "origin".into(),
            source: "log".into(),
            empty_origin: 0.0,
            window: EventWindow { start: 0.0, end: 10.0 },
            left: vec![],
            events: first.chain(second).map(|time| Arrival { time, kind: 0 }).collect(),
            gaps: vec![],
        }
    }

    fn stepped_challenge(tolerance: f64) -> ExcitationChallenge {
        ExcitationChallenge {
            history: stepped_history(),
            fit: HawkesFitSpec { decay: vec![0.5], column_cap: 0.9, iterations: 2000 },
            alternative: BaselineFeature { knots: vec![0.0, 5.0], values: vec![0.0, 1.0] },
            source: 0,
            target: 0,
            horizon: 1.0,
            tolerance,
        }
    }

    #[test]
    fn an_immigrant_step_explains_regular_arrivals_and_withholds_the_constant_mass() {
        // Inside each half the arrivals are evenly spaced, so with a declared step feature the
        // likelihood is maximized by the piecewise-constant Poisson rate (1 then 3) with no
        // excitation: the excitation gradient at zero is a sum of x / (exp(x) - 1) < 1 terms
        // against a compensator of about one per arrival. The alternative's direct mass is
        // therefore exactly zero, while the constant-baseline fit reads the step as excitation.
        let ctx = ExecutionContext::for_tests(0);
        let prepared = Study::prepare_excitation_challenge(stepped_challenge(0.05), &ctx).unwrap();
        assert_eq!(prepared.plan_view().route().as_str(), "HawkesExcitation");
        let reading = prepared.execute(&ctx).unwrap();
        assert_eq!(reading.alternative_direct, 0.0);
        assert_eq!(reading.status, DiagnosticStatus::Withheld);
        assert_eq!(reading.reason, Some("immigrant alternative moved the horizon mass"));
        let ExcitationSubject::Catalog { parameters, .. } = prepared.kernel(&ctx).unwrap() else {
            panic!("a catalog challenge offers a catalog kernel");
        };
        let alpha = parameters.branching[0];
        assert!(alpha > 0.05 && alpha < 0.9, "constant-baseline excitation {alpha}");
        assert!((reading.conditional_direct - alpha * (1.0 - (-0.5_f64).exp())).abs() < 1e-12);
        assert!(reading.conditional_direct > 0.05);

        // Under a tolerance the movement satisfies, the release comparison scores one added
        // event over each remaining horizon of the publisher's window in closed form.
        let compared = Study::prepare_release_comparison(
            ReleaseComparisonRequest {
                parameters: parameters.clone(),
                challenge: stepped_challenge(1.0),
                earlier: 10.0,
                later: 13.0,
                window_end: 15.0,
            },
            &ctx,
        )
        .unwrap()
        .execute(&ctx)
        .unwrap();
        assert_eq!(compared.reading.status, DiagnosticStatus::Agreed);
        let choice = compared.comparison.expect("an agreed reading licenses the comparison");
        let beta = 0.5_f64;
        let direct = |remaining: f64| alpha * (1.0 - (-beta * remaining).exp());
        let descendants = |remaining: f64| {
            alpha / (1.0 - alpha) * (1.0 - (-beta * (1.0 - alpha) * remaining).exp())
        };
        assert!((choice.earlier_direct - direct(5.0)).abs() < 1e-12);
        assert!((choice.later_direct - direct(2.0)).abs() < 1e-12);
        assert!((choice.earlier_descendants - descendants(5.0)).abs() < 1e-6);
        assert!((choice.later_descendants - descendants(2.0)).abs() < 1e-6);
        assert_eq!(choice.greater_descendant_release, Some(0));
    }

    #[test]
    fn each_new_program_runs_under_its_own_unit() {
        let ctx = ExecutionContext::for_tests(0);
        let bound = |view: antecedent::planner::PlanView, route: &str, unit: &str| {
            assert_eq!(view.route().as_str(), route);
            let decision = view.unit_support().expect("the route is bound to a unit");
            assert_eq!(decision.unit(), unit);
        };
        let history = MarkedHistory {
            time_unit: "hour".into(),
            origin: "origin".into(),
            source: "log".into(),
            empty_origin: 0.0,
            window: EventWindow { start: 0.0, end: 1.0 },
            left: vec![],
            events: vec![MarkedArrival { time: 0.5, prototype: 0 }],
            prototypes: vec![MarkPrototype { source: vec![1.0], target: vec![1.0], mass: 1.0 }],
        };
        let parameters =
            MarkedParameters { baseline: vec![1.0], excitation: vec![0.2], decay: vec![1.0] };
        bound(
            Study::prepare_marked_likelihood(history, parameters, &ctx).unwrap().plan_view(),
            "MarkedEventProcess",
            "forecast.marked_event_process",
        );
        bound(
            Study::prepare_excitation_challenge(stepped_challenge(0.05), &ctx).unwrap().plan_view(),
            "HawkesExcitation",
            "forecast.hawkes_excitation",
        );
        let shock = ExcludedShock {
            history: EventHistory {
                catalog: vec!["release".into(), "activity".into()],
                ..stepped_history()
            },
            feature: BaselineFeature { knots: vec![0.0, 5.0], values: vec![0.0, 1.0] },
            source: 0,
            target: 1,
            horizon: 1.0,
            instrument: "launch_window".into(),
        };
        bound(
            Study::prepare_shock_query(shock, &ctx).unwrap().plan_view(),
            "HawkesParameterBridge",
            "forecast.hawkes_parameter_bridge",
        );
    }

    fn one_mark(events: Vec<MarkedArrival>) -> MarkedHistory {
        MarkedHistory {
            time_unit: "hour".into(),
            origin: "origin".into(),
            source: "log".into(),
            empty_origin: 0.0,
            window: EventWindow { start: 0.0, end: 1.0 },
            left: vec![],
            events,
            prototypes: vec![MarkPrototype { source: vec![1.0], target: vec![1.0], mass: 1.0 }],
        }
    }

    #[test]
    fn marked_declarations_outside_the_model_refuse_before_execution() {
        let ctx = ExecutionContext::for_tests(0);
        // A lifetime mass of one is critical: the prototype branching matrix has no positive
        // subinvariance witness.
        let critical =
            MarkedParameters { baseline: vec![1.0], excitation: vec![1.0], decay: vec![1.0] };
        assert!(matches!(
            Study::prepare_marked_likelihood(one_mark(vec![]), critical, &ctx),
            Err(EventError::Unstable)
        ));
        let tied = vec![
            MarkedArrival { time: 0.5, prototype: 0 },
            MarkedArrival { time: 0.5, prototype: 0 },
        ];
        let stable =
            MarkedParameters { baseline: vec![1.0], excitation: vec![0.2], decay: vec![1.0] };
        assert!(matches!(
            Study::prepare_marked_likelihood(one_mark(tied), stable, &ctx),
            Err(EventError::Invalid(_))
        ));
        // The reproduction cap must lie strictly inside (0, 1).
        let open = MarkedFitSpec { decay: vec![1.0], reproduction_cap: 1.0, iterations: 100 };
        assert!(matches!(
            Study::prepare_marked_fit(
                one_mark(vec![MarkedArrival { time: 0.5, prototype: 0 }]),
                open,
                &ctx
            ),
            Err(EventError::Invalid(_))
        ));
    }

    #[test]
    fn excitation_queries_outside_the_kernel_refuse_and_a_withheld_reading_compares_nothing() {
        let ctx = ExecutionContext::for_tests(0);
        let kernel =
            || HawkesParameters { baseline: vec![0.5], branching: vec![0.4], decay: vec![1.0] };
        assert!(matches!(
            Study::prepare_excitation_offspring(kernel(), 1, 1.0, &ctx),
            Err(EventError::Invalid(_))
        ));
        assert!(matches!(
            Study::prepare_excitation_offspring(kernel(), 0, 0.0, &ctx),
            Err(EventError::Invalid(_))
        ));
        // The stepped challenge withholds at tolerance 0.05, so no release is scored.
        let compared = Study::prepare_release_comparison(
            ReleaseComparisonRequest {
                parameters: kernel(),
                challenge: stepped_challenge(0.05),
                earlier: 10.0,
                later: 13.0,
                window_end: 15.0,
            },
            &ctx,
        )
        .unwrap()
        .execute(&ctx)
        .unwrap();
        assert_eq!(compared.reading.status, DiagnosticStatus::Withheld);
        assert!(compared.comparison.is_none());
    }
}
