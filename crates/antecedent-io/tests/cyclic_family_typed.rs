//! Typed cyclic worlds, experiment-recovered laws and dynamic-fit inference,
//! executed through the family program and replayed from encoded bytes.
//!
//! The equilibrium numbers are those of `A = 0.4 B + U_A`, `B = 0.5 A + U_B`: a unit
//! shock on `U_A` gives `A = 1.25`, `B = 0.625`; `do(A = 2)` gives `B = 1`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::execution_attempt::{PhysicalEvent, observe_with_context};
use antecedent_core::{CanonicalWriter, ExecutionContext};
use antecedent_estimate::cyclic_dynamic_inference::{
    DynamicBootstrapMethod, DynamicPosteriorOptions, DynamicPrior, DynamicQuantity,
    OuInferenceProblem,
};
use antecedent_estimate::cyclic_dynamics::OuParameter;
use antecedent_estimate::cyclic_likelihood::{CyclicBootstrapDesign, CyclicFitOptions};
use antecedent_io::cyclic_family_artifact::{
    CertificateDemandWire, CoefficientPriorWire, CompiledCyclicFamilyProgram, CyclicComponentWire,
    CyclicFamilyAnswer, CyclicFamilyRequest, CyclicInterventionWire, CyclicInterventionWorld,
    CyclicWorldOutcomeAnswer, CyclicWorldWire, DisturbancePriorWire, DynamicInferenceAnswer,
    DynamicMethodWire, DynamicProblemWire, ExperimentRegimeWire, ExperimentalFitOptionsWire,
    LinearCyclicWire, RecoveredDrawAnswer, RecoveredDrawOriginWire, RecoveredDrawStatusWire,
    RecoveredSolvedBy, RecoveryInferenceWire, RecoveryPriorWire, RegimeClampWire, RegimeSampleWire,
    ResamplingRuleWire, SamplingDesignWire, UnitColumnWire, consume_cyclic_family,
};
use antecedent_io::cyclic_system_artifact::CyclicArtifactError;
use antecedent_model::continuous_cyclic::OrnsteinUhlenbeck;
use antecedent_model::nonlinear_equilibrium::{
    Activation, Equation, EquilibriumModel, EquilibriumOutcome, RegimeRow, SolvePolicy,
};

/// A recovered-linear request as an earlier encoder wrote it, before draw origins.
const SUPPLIED_RECOVERED_LINEAR: &str = concat!(
    r#"{"family":"recovered_linear","nodes":[0,1],"draws":[{"coefficients":"#,
    r#"[{"into":0,"from":1,"value":0.4},{"into":1,"from":0,"value":0.5}],"#,
    r#""intercept":[0.0,0.0],"noise_mean":[1.0,0.0],"noise_covariance":[1.0,0.0,0.0,1.0],"#,
    r#""unresolved":[],"weight":1.0,"marked_inadmissible":false}],"#,
    r#""worlds":[{"clamp_nodes":[],"clamp_values":[]},{"clamp_nodes":[0],"clamp_values":[2.0]}],"#,
    r#""outcome":1,"source_id":"clamp-experiments","certifying_evidence":["held-out-regime"]}"#,
);

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!((actual - expected).abs() <= tolerance, "{actual} vs {expected} (±{tolerance})");
}

fn refusal_text(error: CyclicArtifactError) -> String {
    match error {
        CyclicArtifactError::Invalid(message) => message,
        other => panic!("expected a typed refusal, got {other:?}"),
    }
}

fn set(variable: u32, value: f64) -> CyclicInterventionWire {
    CyclicInterventionWire::Set { variable, value }
}

fn world(id: u32, interventions: Vec<CyclicInterventionWire>) -> CyclicWorldWire {
    CyclicWorldWire { world: id, interventions, initial_states: Vec::new() }
}

/// Deterministic xorshift normal stream; no library random state.
struct Stream(u64);

impl Stream {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 + 0.5) / (1_u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

#[test]
fn a_supplied_recovered_request_keeps_its_encoding_identity_and_replay() {
    let request: CyclicFamilyRequest = serde_json::from_str(SUPPLIED_RECOVERED_LINEAR).unwrap();
    let CyclicFamilyRequest::RecoveredLinear { uncertainty, .. } = &request else {
        panic!("recovered linear request");
    };
    assert!(uncertainty.is_supplied());
    assert_eq!(serde_json::to_string(&request).unwrap(), SUPPLIED_RECOVERED_LINEAR);
    let (report, bytes) = request.export(&ctx()).unwrap();
    assert_eq!(
        report.identity,
        CanonicalWriter::digest_bytes_hex(
            "antecedent.cyclic_family.request.v1",
            SUPPLIED_RECOVERED_LINEAR.as_bytes()
        )
    );
    let CyclicFamilyAnswer::RecoveredLinear(answer) = &report.answer else {
        panic!("recovered linear answer");
    };
    match &answer.worlds[0].draws[0].outcome {
        RecoveredDrawAnswer::Equilibrium {
            solved_by: RecoveredSolvedBy::Cyclic { .. },
            mean,
            ..
        } => {
            close(mean[0], 1.25, 1e-10);
            close(mean[1], 0.625, 1e-10);
        }
        other => panic!("open world was {other:?}"),
    }
    assert_eq!(consume_cyclic_family(&bytes, &report.identity, &ctx()).unwrap(), report);
    let worlds = request.intervention_worlds(&report).unwrap();
    assert!(matches!(
        &worlds[1],
        CyclicInterventionWorld::Recovered {
            world: 1,
            origin: RecoveredDrawOriginWire::Supplied,
            ..
        }
    ));
}

fn linear_request(
    coefficients: Vec<f64>,
    worlds: Vec<CyclicWorldWire>,
    certificate: CertificateDemandWire,
) -> CyclicFamilyRequest {
    let n = (1..=coefficients.len()).find(|n| n * n == coefficients.len()).unwrap();
    let mut covariance = vec![0.0; n * n];
    for i in 0..n {
        covariance[i * n + i] = 1.0;
    }
    CyclicFamilyRequest::LinearWorlds {
        variables: (10..).take(n).collect(),
        model: LinearCyclicWire {
            coefficients,
            intercept: vec![0.0; n],
            noise_mean: vec![0.0; n],
            noise_covariance: covariance,
        },
        component: CyclicComponentWire {
            component_id: "feedback".into(),
            source_id: "specification".into(),
            certifying_evidence: vec!["independent-bench-test".into()],
        },
        worlds,
        certificate,
    }
}

#[test]
fn linear_worlds_are_keyed_by_world_and_keep_distinct_actions_and_certificates() {
    // K[into * 2 + from]: A = 0.4 B, B = 0.5 A, variables 10 (A) and 11 (B).
    let request = linear_request(
        vec![0.0, 0.4, 0.5, 0.0],
        vec![
            world(7, vec![]),
            world(8, vec![set(10, 2.0)]),
            world(9, vec![CyclicInterventionWire::Shift { variable: 10, delta: 1.0 }]),
        ],
        CertificateDemandWire::Required,
    );
    let program = CompiledCyclicFamilyProgram::compile(&request, &ctx()).unwrap();
    let (report, bytes) = program.export(&ctx()).unwrap();
    let CyclicFamilyAnswer::Worlds(worlds) = &report.answer else { panic!("typed worlds") };
    assert_eq!(worlds.iter().map(|w| w.world).collect::<Vec<_>>(), [7, 8, 9]);
    for answer in worlds {
        assert!(answer.certificate.certified, "{answer:?}");
        assert_eq!(answer.certificate.basis.as_deref(), Some("linear_family_contraction"));
        close(answer.failed_mass, 0.0, 0.0);
        assert_eq!(answer.physical_stability, "unclaimed");
    }
    let mean = |index: usize| match &worlds[index].outcome {
        CyclicWorldOutcomeAnswer::GaussianLaw { mean, .. } => mean.clone(),
        other => panic!("world {index} was {other:?}"),
    };
    close(mean(1)[0], 2.0, 1e-10);
    close(mean(1)[1], 1.0, 1e-10);
    // A shock is not a clamp: the feedback through B stays active.
    close(mean(2)[0], 1.25, 1e-10);
    close(mean(2)[1], 0.625, 1e-10);
    assert_eq!(worlds[1].actions[0].kind, "hard_clamp");
    assert_eq!(worlds[2].actions[0].kind, "additive_shock");
    assert_eq!(
        consume_cyclic_family(&bytes, &report.identity, &ExecutionContext::for_tests(5)).unwrap(),
        report
    );
}

#[test]
fn a_clamp_that_leaves_no_unique_solution_is_failed_mass_not_a_dropped_world() {
    // A = B + C, B = A, C = A: the full system solves; do(C = 1) leaves 0 = 1 + U_A + U_B.
    let request = linear_request(
        vec![0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        vec![world(0, vec![]), world(1, vec![set(12, 1.0)])],
        CertificateDemandWire::ReportUncertified,
    );
    let report =
        CompiledCyclicFamilyProgram::compile(&request, &ctx()).unwrap().execute(&ctx()).unwrap();
    let CyclicFamilyAnswer::Worlds(worlds) = &report.answer else { panic!("typed worlds") };
    assert!(!worlds[0].certificate.certified);
    assert!(worlds[0].certificate.next_attempt.is_some());
    assert!(matches!(worlds[1].outcome, CyclicWorldOutcomeAnswer::NoSolution { .. }));
    close(worlds[1].failed_mass, 1.0, 0.0);
}

#[test]
fn typed_world_refusals_name_their_next_attempt() {
    let unbound = linear_request(
        vec![0.0, 0.4, 0.5, 0.0],
        vec![world(0, vec![set(99, 1.0)])],
        CertificateDemandWire::Required,
    );
    let message = refusal_text(CompiledCyclicFamilyProgram::compile(&unbound, &ctx()).unwrap_err());
    assert!(message.contains("next attempt"), "{message}");
    let repeated = linear_request(
        vec![0.0, 0.4, 0.5, 0.0],
        vec![world(3, vec![]), world(3, vec![set(10, 1.0)])],
        CertificateDemandWire::Required,
    );
    let message =
        refusal_text(CompiledCyclicFamilyProgram::compile(&repeated, &ctx()).unwrap_err());
    assert!(message.contains("next attempt"), "{message}");
}

fn feedback_model() -> EquilibriumModel {
    let equation = |id: &str, weights: Vec<f64>| Equation {
        id: id.into(),
        intercept: 0.0,
        weights,
        activation: Activation::Identity,
        expression: None,
        parameters: vec![],
    };
    EquilibriumModel {
        equations: vec![equation("a", vec![0.0, 0.4]), equation("b", vec![0.5, 0.0])],
        noise_covariance: vec![1.0, 0.0, 0.0, 1.0],
    }
}

#[test]
fn equilibrium_worlds_share_their_disturbance_draws() {
    let request = CyclicFamilyRequest::EquilibriumWorlds {
        variables: vec![0, 1],
        model: feedback_model(),
        worlds: vec![world(0, vec![]), world(1, vec![set(0, 2.0)])],
        noise: vec![vec![0.0, 0.0], vec![1.0, 0.0]],
        initial: vec![0.0, 0.0],
        policy: SolvePolicy::default(),
    };
    let (report, bytes) =
        CompiledCyclicFamilyProgram::compile(&request, &ctx()).unwrap().export(&ctx()).unwrap();
    let CyclicFamilyAnswer::Worlds(worlds) = &report.answer else { panic!("typed worlds") };
    let solved = |world: usize, draw: usize| match &worlds[world].outcome {
        CyclicWorldOutcomeAnswer::EquilibriumDraws { outcomes } => match &outcomes[draw] {
            EquilibriumOutcome::Solved(result) => result.values.clone(),
            other => panic!("draw {draw} was {other:?}"),
        },
        other => panic!("world {world} was {other:?}"),
    };
    close(solved(0, 1)[0], 1.25, 1e-6);
    close(solved(0, 1)[1], 0.625, 1e-6);
    close(solved(1, 1)[1], 1.0, 1e-6);
    assert_eq!(consume_cyclic_family(&bytes, &report.identity, &ctx()).unwrap(), report);
}

/// One regime of `A = 0.4 B + U_A`, `B = 0.5 A + U_B` with randomized clamps.
fn simulate(clamped: Option<u32>, rows: usize, stream: &mut Stream) -> Vec<Vec<f64>> {
    let mut a = Vec::with_capacity(rows);
    let mut b = Vec::with_capacity(rows);
    for _ in 0..rows {
        let (ua, ub) = (stream.normal(), stream.normal());
        let (x, y) = match clamped {
            None => {
                let x = (ua + 0.4 * ub) / 0.8;
                (x, 0.5 * x + ub)
            }
            Some(0) => {
                let x = 2.0 + 1.5 * stream.normal();
                (x, 0.5 * x + ub)
            }
            Some(_) => {
                let y = 3.0 + 1.5 * stream.normal();
                (0.4 * y + ua, y)
            }
        };
        a.push(x);
        b.push(y);
    }
    vec![a, b]
}

fn recovery_request(
    designs: &[Option<u32>],
    inference: RecoveryInferenceWire,
) -> CyclicFamilyRequest {
    let mut stream = Stream(0x9e37_79b9_7f4a_7c15);
    let mut regimes = Vec::new();
    let mut samples = Vec::new();
    for (index, clamped) in designs.iter().enumerate() {
        let source = u64::try_from(index).unwrap() + 1;
        regimes.push(ExperimentRegimeWire {
            clamps: clamped
                .map(|unit| RegimeClampWire { unit, value: if unit == 0 { 2.0 } else { 3.0 } })
                .into_iter()
                .collect(),
            units: vec![0, 1],
            population: 0,
            dependent: false,
            source,
        });
        let columns = simulate(*clamped, 400, &mut stream);
        samples.push(RegimeSampleWire {
            source,
            columns: columns
                .into_iter()
                .enumerate()
                .map(|(unit, values)| UnitColumnWire { unit: u32::try_from(unit).unwrap(), values })
                .collect(),
            sampling_units: None,
        });
    }
    CyclicFamilyRequest::ExperimentalRecovery {
        regimes,
        dropped: vec![],
        samples,
        options: ExperimentalFitOptionsWire::default(),
        inference,
        worlds: vec![world(1, vec![]), world(2, vec![set(0, 2.0)])],
        outcome: 1,
        source_id: "clamp-experiments".into(),
        certifying_evidence: vec!["held-out-regime".into()],
    }
}

const COMPLETE: &[Option<u32>] = &[None, Some(0), Some(1)];

#[test]
fn a_recovered_law_is_fitted_at_execution_and_replayed_from_its_draws() {
    let request =
        recovery_request(COMPLETE, RecoveryInferenceWire::AsymptoticNormal { draws: 64, seed: 7 });
    let program = CompiledCyclicFamilyProgram::compile(&request, &ctx()).unwrap();
    let first = program.execute(&ctx()).unwrap();
    let CyclicFamilyAnswer::ExperimentalRecovery(report) = &first.answer else {
        panic!("experimental recovery answer");
    };
    assert_eq!(report.law_meaning.as_deref(), Some("sampling_law"));
    assert!(matches!(
        report.law.origin,
        RecoveredDrawOriginWire::AsymptoticNormal { draws: 64, seed: 7 }
    ));
    assert_eq!(report.law.draws.len(), 64);
    assert_eq!(report.worlds.iter().map(|w| w.world).collect::<Vec<_>>(), [1, 2]);
    let clamped = &report.worlds[1].answer;
    close(clamped.mixture_mean.unwrap(), 1.0, 0.2);
    assert!(clamped.between_draw_variance.unwrap() > 0.0);
    // The same seed gives the same law on another context.
    let other = CompiledCyclicFamilyProgram::compile(&request, &ExecutionContext::for_tests(8))
        .unwrap()
        .execute(&ExecutionContext::for_tests(8))
        .unwrap();
    assert_eq!(other, first);
    let (exported, bytes) = program.export(&ctx()).unwrap();
    assert_eq!(
        consume_cyclic_family(&bytes, &exported.identity, &ExecutionContext::for_tests(3)).unwrap(),
        exported
    );

    // Changing only the worlds reuses the recovered law.
    let mut retargeted = request.clone();
    if let CyclicFamilyRequest::ExperimentalRecovery { worlds, .. } = &mut retargeted {
        *worlds = vec![world(5, vec![set(0, -1.0)])];
    }
    let rebound = program.rebind(&retargeted, &ctx()).unwrap();
    let reused = observe_with_context(&ctx(), || rebound.execute(&ctx()));
    assert!(reused.report.physical(PhysicalEvent::CacheHit).is_some_and(|hits| hits >= 1));
    let reused = reused.result.unwrap();
    let CyclicFamilyAnswer::ExperimentalRecovery(again) = &reused.answer else { panic!() };
    assert_eq!(again.law, report.law);
    close(again.worlds[0].answer.mixture_mean.unwrap(), -0.5, 0.2);
}

#[test]
fn recovered_uncertainty_keeps_its_meaning_and_unresolved_rows_stay_unresolved() {
    let posterior = recovery_request(
        COMPLETE,
        RecoveryInferenceWire::Posterior {
            prior: RecoveryPriorWire {
                coefficients: CoefficientPriorWire::Flat,
                disturbance: DisturbancePriorWire::Known { variances: vec![1.0, 1.0] },
            },
            draws: 32,
            burn_in: 0,
            seed: 3,
        },
    );
    let report = posterior.execute(&ctx()).unwrap();
    let CyclicFamilyAnswer::ExperimentalRecovery(law) = &report.answer else { panic!() };
    assert_eq!(law.law_meaning.as_deref(), Some("parameter_posterior"));

    let bootstrap = recovery_request(
        COMPLETE,
        RecoveryInferenceWire::Bootstrap {
            replicates: 12,
            seed: 4,
            design: SamplingDesignWire::IndependentSubjectTrajectories,
            rule: ResamplingRuleWire::RowSubjectIid,
            block_length: None,
        },
    );
    let report = bootstrap.execute(&ctx()).unwrap();
    let CyclicFamilyAnswer::ExperimentalRecovery(law) = &report.answer else { panic!() };
    assert_eq!(law.law_meaning.as_deref(), Some("sampling_law"));
    assert_eq!(law.law.draws.len(), 12);

    // Without a regime that clamps B, the row into A is uncovered: no draw completes it.
    let uncovered = recovery_request(
        &[None, Some(0)],
        RecoveryInferenceWire::AsymptoticNormal { draws: 16, seed: 9 },
    );
    let (report, bytes) = uncovered.export(&ctx()).unwrap();
    let CyclicFamilyAnswer::ExperimentalRecovery(law) = &report.answer else { panic!() };
    assert!(
        law.law
            .draw_status
            .iter()
            .all(|status| matches!(status, RecoveredDrawStatusWire::Unresolved { .. }))
    );
    let clamped = &law.worlds[1].answer;
    assert!(clamped.mixture_mean.is_none());
    close(clamped.withheld_mass, 1.0, 1e-12);
    assert_eq!(consume_cyclic_family(&bytes, &report.identity, &ctx()).unwrap(), report);
}

#[test]
fn a_recovered_law_refuses_actions_other_than_hard_clamps_at_compilation() {
    let mut request =
        recovery_request(COMPLETE, RecoveryInferenceWire::AsymptoticNormal { draws: 4, seed: 1 });
    if let CyclicFamilyRequest::ExperimentalRecovery { worlds, .. } = &mut request {
        worlds.push(world(3, vec![CyclicInterventionWire::Shift { variable: 0, delta: 1.0 }]));
    }
    let message = refusal_text(CompiledCyclicFamilyProgram::compile(&request, &ctx()).unwrap_err());
    assert!(message.contains("hard clamps only") && message.contains("next attempt"), "{message}");
}

fn ou_problem() -> OuInferenceProblem {
    let mut stream = Stream(0x5eed_0f00);
    OuInferenceProblem {
        template: OrnsteinUhlenbeck { drift: vec![-1.0], center: vec![0.0], diffusion: vec![4.0] },
        parameters: vec![
            OuParameter::Drift { row: 0, column: 0 },
            OuParameter::Center { variable: 0 },
        ],
        rows: (0..300)
            .map(|_| RegimeRow { values: vec![0.5 + stream.normal()], clamps: vec![] })
            .collect(),
        options: CyclicFitOptions::default(),
    }
}

#[test]
fn dynamic_inference_shares_its_fit_across_methods_and_replays() {
    let quantities = vec![DynamicQuantity::OuStationaryMean { clamps: vec![], variable: 0 }];
    let information = CyclicFamilyRequest::DynamicInference {
        problem: DynamicProblemWire::Ou(ou_problem()),
        method: DynamicMethodWire::Information {
            design: SamplingDesignWire::IndependentSubjectTrajectories,
        },
        quantities: quantities.clone(),
    };
    let program = CompiledCyclicFamilyProgram::compile(&information, &ctx()).unwrap();
    let first = program.execute(&ctx()).unwrap();
    let CyclicFamilyAnswer::DynamicInference { fit, inference } = &first.answer else {
        panic!("dynamic inference");
    };
    let DynamicInferenceAnswer::Information(observed) = inference.as_ref() else {
        panic!("observed information");
    };
    close(observed.quantity_estimates[0].unwrap(), 0.5, 0.25);
    assert!(observed.quantity_standard_errors[0].is_some());

    let bootstrap = CyclicFamilyRequest::DynamicInference {
        problem: DynamicProblemWire::Ou(ou_problem()),
        method: DynamicMethodWire::Bootstrap {
            kind: DynamicBootstrapMethod::Nonparametric,
            replicates: 20,
            design: SamplingDesignWire::IndependentSubjectTrajectories,
            resampling: ResamplingRuleWire::RowSubjectIid,
            stratification: CyclicBootstrapDesign::FixedAssignments,
            seed: 11,
        },
        quantities: quantities.clone(),
    };
    let rebound = program.rebind(&bootstrap, &ctx()).unwrap();
    let attempt = observe_with_context(&ctx(), || rebound.execute(&ctx()));
    assert!(attempt.report.physical(PhysicalEvent::CacheHit).is_some_and(|hits| hits >= 1));
    let second = attempt.result.unwrap();
    let CyclicFamilyAnswer::DynamicInference { fit: reused, inference } = &second.answer else {
        panic!("dynamic inference");
    };
    let DynamicInferenceAnswer::SamplingLaw(law) = inference.as_ref() else {
        panic!("sampling law");
    };
    assert_eq!(reused, fit);
    assert_eq!(law.replicates.len(), 20);
    let (exported, bytes) = rebound.export(&ctx()).unwrap();
    assert_eq!(
        consume_cyclic_family(&bytes, &exported.identity, &ExecutionContext::for_tests(6)).unwrap(),
        exported
    );

    let posterior = CyclicFamilyRequest::DynamicInference {
        problem: DynamicProblemWire::Ou(ou_problem()),
        method: DynamicMethodWire::Posterior {
            prior: DynamicPrior { mean: vec![-1.0, 0.0], standard_deviation: vec![10.0, 10.0] },
            options: DynamicPosteriorOptions {
                chains: 2,
                warmup: 50,
                draws: 50,
                proposal_scale: None,
                start_spread: 1.0,
            },
            design: SamplingDesignWire::IndependentSubjectTrajectories,
            seed: 13,
        },
        quantities,
    };
    let report = posterior.execute(&ctx()).unwrap();
    let CyclicFamilyAnswer::DynamicInference { inference, .. } = &report.answer else {
        panic!("dynamic inference");
    };
    let DynamicInferenceAnswer::Posterior(draws) = inference.as_ref() else {
        panic!("posterior");
    };
    assert_eq!(draws.quantity_draws.len(), 100);
}
