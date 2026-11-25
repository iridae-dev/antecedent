//! The direct cross-world seal and the compiled `CrossWorldFiniteExecution` route give the same
//! answer.
//!
//! `Study::compile_cross_world_execution` binds the request's wire form to the
//! `cross_world_execution.finite_markovian_scm` unit and seals through the same kernel function
//! as `CrossWorldExecution::seal`, so the value, label, identity and artifact bytes agree bit for
//! bit. The fixture is the mediation model of `cross_world_execution_facade` (natural direct
//! effect 0.25).

#![allow(deprecated, reason = "the deprecated direct seal is the path under comparison")]

use antecedent::Study;
use antecedent::analysis::cross_world_execution::{
    CrossWorldExecution, CrossWorldExecutionArtifactError, ExogenousInference, ExogenousVariable,
    FactualUnit, FiniteExecutionRequest, FiniteQuantity, FiniteScm, FiniteVariable, FiniteWorld,
    NoiseCoupling, WorldRoute,
};
use antecedent::planner::PlanRoute;
use antecedent::prelude::ExecutionContext;
use antecedent::scientific::RequestedReading;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(21)
}

fn mediation() -> FiniteScm {
    FiniteScm {
        population: "trial-population".to_owned(),
        history: "baseline-regime".to_owned(),
        exogenous: vec![
            ExogenousVariable { name: "u_x".to_owned(), pmf: vec![0.5, 0.5] },
            ExogenousVariable { name: "u_m".to_owned(), pmf: vec![0.3, 0.4, 0.1, 0.2] },
            ExogenousVariable { name: "u_y".to_owned(), pmf: vec![0.75, 0.25] },
        ],
        variables: vec![
            FiniteVariable {
                name: "x".to_owned(),
                levels: 2,
                parents: vec![],
                exogenous: 0,
                table: vec![0, 1],
            },
            FiniteVariable {
                name: "m".to_owned(),
                levels: 2,
                parents: vec![0],
                exogenous: 1,
                table: vec![0, 0, 0, 1, 1, 0, 1, 1],
            },
            FiniteVariable {
                name: "y".to_owned(),
                levels: 2,
                parents: vec![0, 1],
                exogenous: 2,
                table: vec![0, 1, 0, 1, 0, 0, 1, 1],
            },
        ],
    }
}

fn request() -> FiniteExecutionRequest {
    let scm = mediation();
    let route = WorldRoute { parent: 1, child: 2, source: 1 };
    FiniteExecutionRequest {
        evidence_history: scm.history.clone(),
        scm,
        units: vec![FactualUnit { unit: "cohort-mean".to_owned(), observed: vec![] }],
        worlds: vec![
            FiniteWorld {
                name: "x1_with_m_of_x0".to_owned(),
                interventions: vec![(0, 1)],
                routes: vec![route],
            },
            FiniteWorld { name: "x0".to_owned(), interventions: vec![(0, 0)], routes: vec![] },
        ],
        quantity: FiniteQuantity::MeanContrast { plus: (0, 2), minus: (1, 2) },
        coupling: NoiseCoupling::SharedAbducedExogenous,
        exogenous: ExogenousInference::ExactEnumeration,
        target_population: None,
        reading: RequestedReading::Point,
    }
}

#[test]
fn the_compiled_seal_reproduces_the_direct_seal() {
    let mut request = request();
    request.reading = RequestedReading::Interval;
    let compiled = Study::compile_cross_world_execution((&request).into(), &ctx()).unwrap();
    let Err(CrossWorldExecutionArtifactError::Refused(refusal)) = compiled.seal(&ctx()) else {
        panic!("an interval without an interval construction must be refused");
    };
    assert_eq!(refusal.code, "uncertainty_representation_insufficient");
    assert_eq!(refusal.detail, "joint_law_propagation.no_interval_construction");

    request.reading = RequestedReading::Point;
    let direct = CrossWorldExecution::seal(&request, &ctx()).unwrap();
    let compiled = Study::compile_cross_world_execution((&request).into(), &ctx()).unwrap();
    assert_eq!(compiled.plan_view().route(), PlanRoute::CrossWorldFiniteExecution);
    let sealed = compiled.seal(&ctx()).unwrap();
    assert_eq!(sealed.result().value.to_bits(), direct.result().value.to_bits());
    assert!((sealed.result().value - 0.25).abs() < 1e-12);
    assert_eq!(sealed.result().label, direct.result().label);
    assert_eq!(sealed.identity(), direct.identity());
    assert_eq!(sealed.to_bytes("twin").unwrap(), direct.to_bytes("twin").unwrap());
}
