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
