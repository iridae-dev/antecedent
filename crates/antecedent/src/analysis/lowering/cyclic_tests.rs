//! Cyclic family lowering: every family lowers to a complete program with a
//! stable identity, preparation obtains its plan from that lowering, results carry
//! the lowered identity, and the version-two family record carries a contract an
//! independent consumer verifies.
//!
//! The equilibrium numbers are those of `A = 0.4 B + U_A`, `B = 0.5 A + U_B`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{ExecutionContext, IdentityDomain};
use antecedent_estimate::cyclic_dynamic_inference::{DynamicQuantity, OuInferenceProblem};
use antecedent_estimate::cyclic_dynamics::OuParameter;
use antecedent_estimate::cyclic_likelihood::CyclicFitOptions;
use antecedent_io::contract_section::cyclic::{
    CyclicIdentificationWire, CyclicTargetWire, verify_cyclic_contract_against_body,
};
use antecedent_io::cyclic_family_artifact::{
    CertificateDemandWire, CompiledCyclicFamilyProgram, CyclicComponentWire, CyclicFamilyRequest,
    CyclicInitialStateWire, CyclicInterventionWire, CyclicWorldWire, DynamicMethodWire,
    DynamicProblemWire, ExperimentRegimeWire, ExperimentalFitOptionsWire, LinearCyclicWire,
    RecoveredCoefficientWire, RecoveredDrawOriginWire, RecoveredDrawWire,
    RecoveredInterventionWire, RecoveryInferenceWire, RegimeClampWire, RegimeSampleWire,
    SamplingDesignWire, UnitColumnWire, consume_cyclic_family, consume_cyclic_family_contract,
};
use antecedent_io::cyclic_system_artifact::CyclicExecutionRequest;
use antecedent_model::continuous_cyclic::{OdePolicy, OdeSystem, OrnsteinUhlenbeck};
use antecedent_model::nonlinear_equilibrium::{
    Activation, Clamp, Equation, EquilibriumModel, RegimeRow, SolvePolicy,
};

use super::{lower_cyclic_family, lower_cyclic_system};
use crate::Study;
use crate::planner::lowered::{
    LoweringDepth, NonGraphicalIdentification, ProblemTarget, UncertaintyReading,
};

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

fn set(variable: u32, value: f64) -> CyclicInterventionWire {
    CyclicInterventionWire::Set { variable, value }
}

fn world(id: u32, interventions: Vec<CyclicInterventionWire>) -> CyclicWorldWire {
    CyclicWorldWire { world: id, interventions, initial_states: Vec::new() }
}

fn equation(id: &str, weights: Vec<f64>) -> Equation {
    Equation {
        id: id.into(),
        intercept: 0.0,
        weights,
        activation: Activation::Identity,
        expression: None,
        parameters: vec![],
    }
}

fn feedback_model() -> EquilibriumModel {
    EquilibriumModel {
        equations: vec![equation("a", vec![0.0, 0.4]), equation("b", vec![0.5, 0.0])],
        noise_covariance: vec![1.0, 0.0, 0.0, 1.0],
    }
}

fn linear_worlds(worlds: Vec<CyclicWorldWire>) -> CyclicFamilyRequest {
    CyclicFamilyRequest::LinearWorlds {
        variables: vec![10, 11],
        model: LinearCyclicWire {
            coefficients: vec![0.0, 0.4, 0.5, 0.0],
            intercept: vec![0.0; 2],
            noise_mean: vec![0.0; 2],
            noise_covariance: vec![1.0, 0.0, 0.0, 1.0],
        },
        component: CyclicComponentWire {
            component_id: "feedback".into(),
            source_id: "specification".into(),
            certifying_evidence: vec!["independent-bench-test".into()],
        },
        worlds,
        certificate: CertificateDemandWire::Required,
    }
}

fn typed_linear() -> CyclicFamilyRequest {
    linear_worlds(vec![
        world(7, vec![]),
        world(8, vec![set(10, 2.0)]),
        world(9, vec![CyclicInterventionWire::Shift { variable: 10, delta: 1.0 }]),
    ])
}

fn equilibrium_worlds() -> CyclicFamilyRequest {
    CyclicFamilyRequest::EquilibriumWorlds {
        variables: vec![0, 1],
        model: feedback_model(),
        worlds: vec![world(0, vec![]), world(1, vec![set(0, 2.0)])],
        noise: vec![vec![0.0, 0.0], vec![1.0, 0.0]],
        initial: vec![0.0, 0.0],
        policy: SolvePolicy::default(),
    }
}

fn ode_worlds() -> CyclicFamilyRequest {
    CyclicFamilyRequest::OdeWorlds {
        variables: vec![0, 1],
        model: OdeSystem { equations: feedback_model().equations, decay: vec![1.0, 1.0] },
        initial: vec![0.0, 0.0],
        initial_laws: vec!["a0".into(), "b0".into()],
        times: vec![0.0, 0.5, 1.0],
        worlds: vec![
            world(0, vec![]),
            CyclicWorldWire {
                world: 1,
                interventions: Vec::new(),
                initial_states: vec![CyclicInitialStateWire { law: "a0".into(), level: 1.0 }],
            },
        ],
        policy: OdePolicy::default(),
    }
}

fn recovered_linear() -> CyclicFamilyRequest {
    CyclicFamilyRequest::RecoveredLinear {
        nodes: vec![0, 1],
        draws: vec![RecoveredDrawWire {
            coefficients: vec![
                RecoveredCoefficientWire { into: 0, from: 1, value: 0.4 },
                RecoveredCoefficientWire { into: 1, from: 0, value: 0.5 },
            ],
            intercept: vec![0.0, 0.0],
            noise_mean: vec![1.0, 0.0],
            noise_covariance: vec![1.0, 0.0, 0.0, 1.0],
            unresolved: Vec::new(),
            weight: 1.0,
            marked_inadmissible: false,
        }],
        worlds: vec![
            RecoveredInterventionWire { clamp_nodes: vec![], clamp_values: vec![] },
            RecoveredInterventionWire { clamp_nodes: vec![0], clamp_values: vec![2.0] },
        ],
        outcome: 1,
        source_id: "clamp-experiments".into(),
        certifying_evidence: vec!["held-out-regime".into()],
        uncertainty: RecoveredDrawOriginWire::Supplied,
    }
}

/// Deterministic regime rows of the feedback model: passive, `A` clamped, `B` clamped.
fn regime_columns(clamped: Option<u32>, rows: usize) -> Vec<Vec<f64>> {
    let mut a = Vec::with_capacity(rows);
    let mut b = Vec::with_capacity(rows);
    for row in 0..rows {
        let t = f64::from(u32::try_from(row).unwrap());
        let (ua, ub) = ((t * 0.37).sin(), (t * 0.91).cos());
        let (x, y) = match clamped {
            None => {
                let x = (ua + 0.4 * ub) / 0.8;
                (x, 0.5 * x + ub)
            }
            Some(0) => {
                let x = 2.0 + (t * 0.53).sin();
                (x, 0.5 * x + ub)
            }
            Some(_) => {
                let y = 3.0 + (t * 0.29).cos();
                (0.4 * y + ua, y)
            }
        };
        a.push(x);
        b.push(y);
    }
    vec![a, b]
}

fn experimental_recovery(designs: &[Option<u32>]) -> CyclicFamilyRequest {
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
        samples.push(RegimeSampleWire {
            source,
            columns: regime_columns(*clamped, 60)
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
        inference: RecoveryInferenceWire::AsymptoticNormal { draws: 8, seed: 7 },
        worlds: vec![world(1, vec![]), world(2, vec![set(0, 2.0)])],
        outcome: 1,
        source_id: "clamp-experiments".into(),
        certifying_evidence: vec!["held-out-regime".into()],
    }
}

fn dynamic_inference() -> CyclicFamilyRequest {
    CyclicFamilyRequest::DynamicInference {
        problem: DynamicProblemWire::Ou(OuInferenceProblem {
            template: OrnsteinUhlenbeck {
                drift: vec![-1.0],
                center: vec![0.0],
                diffusion: vec![4.0],
            },
            parameters: vec![
                OuParameter::Drift { row: 0, column: 0 },
                OuParameter::Center { variable: 0 },
            ],
            rows: (0..40)
                .map(|row| RegimeRow {
                    values: vec![0.5 + (f64::from(row) * 0.7).sin()],
                    clamps: vec![],
                })
                .collect(),
            options: CyclicFitOptions::default(),
        }),
        method: DynamicMethodWire::Information {
            design: SamplingDesignWire::IndependentSubjectTrajectories,
        },
        quantities: vec![DynamicQuantity::OuStationaryMean { clamps: vec![], variable: 0 }],
    }
}

fn system_request() -> CyclicExecutionRequest {
    CyclicExecutionRequest {
        model: feedback_model(),
        fit: None,
        worlds: vec![vec![], vec![Clamp { variable: 0, value: 2.0 }]],
        noise: vec![vec![1.0, 0.0]],
        initial: vec![0.0; 2],
        policy: SolvePolicy::default(),
        factual: None,
    }
}

fn families() -> Vec<CyclicFamilyRequest> {
    vec![
        typed_linear(),
        equilibrium_worlds(),
        ode_worlds(),
        recovered_linear(),
        experimental_recovery(&[None, Some(0), Some(1)]),
        dynamic_inference(),
        CyclicFamilyRequest::System { request: system_request() },
        CyclicFamilyRequest::OuStationary {
            model: OrnsteinUhlenbeck { drift: vec![-2.0], center: vec![0.0], diffusion: vec![4.0] },
        },
    ]
}

fn lower(request: &CyclicFamilyRequest) -> crate::planner::lowered::LoweredProgram {
    lower_cyclic_family(&CompiledCyclicFamilyProgram::compile(request, &ctx()).unwrap()).unwrap()
}

#[test]
fn every_cyclic_family_lowers_to_a_complete_program_with_a_stable_identity() {
    for request in families() {
        let family = request.family_name();
        let first = lower(&request);
        let again = lower(&request);
        assert_eq!(first.depth(), LoweringDepth::Complete, "{family}");
        assert_eq!(first.cyclic_family(), Some(family));
        assert_eq!(first.route(), "CyclicFamily");
        assert_eq!(first.identity(), again.identity(), "{family}: identity is not stable");
        assert!(
            first.non_graphical_identification().is_some(),
            "{family}: a feedback program carries no graphical identification product"
        );
        assert!(!first.identification_retained(), "{family}");
        assert!(!first.scientific().licensed(), "{family}: lowering grants no license");
        assert_eq!(first.scientific().graph_class(), "cyclic");
        assert!(!first.physical().operators().is_empty(), "{family}");
        let output = first.physical().operators().last().unwrap().estimates().output();
        assert!(output.is_some_and(|bytes| bytes > 0), "{family}: output bytes are admitted");
    }
}

#[test]
fn keyed_worlds_lower_to_feedback_targets() {
    let linear = lower(&typed_linear());
    let ProblemTarget::Targets(targets) = linear.scientific().target() else {
        panic!("typed worlds answer every variable");
    };
    assert_eq!(targets.len(), 2);
    for target in targets.iter() {
        assert_eq!(target.assumptions().graph(), antecedent_core::GraphFamily::Cyclic);
    }
    let recovered = lower(&experimental_recovery(&[None, Some(0), Some(1)]));
    let ProblemTarget::Target(target) = recovered.scientific().target() else {
        panic!("a recovery with a declared outcome lowers one target");
    };
    let antecedent_core::TargetForm::Worlds(worlds) = target.form() else {
        panic!("keyed worlds");
    };
    let clamp = worlds.worlds()[1].action().structural_actions().unwrap();
    assert_eq!(clamp[0].kind_name(), "hard_clamp");
    assert!(worlds.worlds()[1].time().is_equilibrium());
    let ode = lower(&ode_worlds());
    let ProblemTarget::Targets(targets) = ode.scientific().target() else {
        panic!("typed worlds answer every variable");
    };
    let antecedent_core::TargetForm::Worlds(worlds) = targets[0].form() else {
        panic!("keyed worlds");
    };
    let changed = &worlds.worlds()[1];
    assert_eq!(changed.time().continuous_horizon(), Some(1.0));
    assert_eq!(
        changed.action().structural_actions().unwrap()[0].initial_state_change(),
        Some(("a0", 1.0))
    );
}

#[test]
fn changing_a_world_or_its_action_kind_moves_the_lowered_identity() {
    let clamp = lower(&linear_worlds(vec![world(0, vec![set(10, 1.0)])]));
    let other_level = lower(&linear_worlds(vec![world(0, vec![set(10, 2.0)])]));
    let shock = lower(&linear_worlds(vec![world(
        0,
        vec![CyclicInterventionWire::Shift { variable: 10, delta: 1.0 }],
    )]));
    assert_ne!(clamp.identity(), other_level.identity());
    assert_ne!(clamp.identity(), shock.identity());
}

#[test]
fn an_experimental_recovery_lowers_its_design_and_unresolved_directions() {
    let complete = lower(&experimental_recovery(&[None, Some(0), Some(1)]));
    let Some(NonGraphicalIdentification::ExperimentalRecovery { unresolved, .. }) =
        complete.non_graphical_identification()
    else {
        panic!("experimental recovery identification");
    };
    assert!(unresolved.is_empty());
    assert_eq!(complete.claim().uncertainty(), UncertaintyReading::Sampling);
    let uncovered = lower(&experimental_recovery(&[None, Some(0)]));
    let Some(NonGraphicalIdentification::ExperimentalRecovery { unresolved, .. }) =
        uncovered.non_graphical_identification()
    else {
        panic!("experimental recovery identification");
    };
    // Without a regime that clamps B, the row into A stays unresolved.
    assert_eq!(unresolved.len(), 1);
    assert_eq!(unresolved[0].0.raw(), 0);
    assert_eq!(unresolved[0].1.iter().map(|id| id.raw()).collect::<Vec<_>>(), [1]);
    assert_ne!(complete.identity(), uncovered.identity());
    let specified = lower(&typed_linear());
    assert!(matches!(
        specified.non_graphical_identification(),
        Some(NonGraphicalIdentification::SpecifiedModelConditional { .. })
    ));
    assert_eq!(specified.claim().uncertainty(), UncertaintyReading::None);
}

#[test]
fn preparation_obtains_its_plan_from_the_lowering_and_results_carry_it() {
    let ctx = ctx();
    for request in [typed_linear(), recovered_linear()] {
        let prepared = Study::prepare_cyclic_family(&request, &ctx).unwrap();
        let lowered = prepared.lowered().identity();
        assert_eq!(lowered, lower(&request).identity());
        assert_eq!(prepared.inspect().lowered_program_identity, lowered);
        assert_eq!(prepared.inspect().preparation_evaluations, 0);
        let report = prepared.estimate(&ctx).unwrap();
        assert_eq!(report.lowered_program_identity(), Some(lowered));
        let (exported, _) = prepared.export(&ctx).unwrap();
        assert_eq!(exported.lowered_program_identity(), Some(lowered));
    }
    let recovered = Study::prepare_recovered_linear(&recovered_linear(), &ctx).unwrap();
    assert_eq!(recovered.inspect().identification, "specified_model_conditional");
    let experimental = Study::prepare_experimental_recovery(
        &experimental_recovery(&[None, Some(0), Some(1)]),
        &ctx,
    )
    .unwrap();
    assert_eq!(experimental.inspect().identification, "experimental_recovery");
    assert_eq!(
        experimental.lowered().identity(),
        lower(&experimental_recovery(&[None, Some(0), Some(1)])).identity()
    );

    let system = Study::prepare_cyclic(system_request(), &ctx).unwrap();
    let lowered = system.lowered().identity();
    assert_eq!(lowered, lower_cyclic_system(&system_request(), &ctx).unwrap().identity());
    assert_eq!(system.inspect().unwrap().lowered_program_identity, lowered);
    assert_eq!(system.estimate(&ctx).unwrap().lowered_program_identity(), Some(lowered));
    let clamp = vec![vec![Clamp { variable: 0, value: 3.0 }]];
    let retargeted = system.retarget_worlds(clamp, vec![vec![1.0, 0.0]], vec![0.0; 2]).unwrap();
    assert_ne!(retargeted.lowered().identity(), lowered, "new worlds lower a new program");
    assert!(system.contract().unwrap().identities.program.is_some());
}

#[test]
fn the_contract_binds_every_layer_and_the_recovery_derivation() {
    let ctx = ctx();
    let prepared = Study::prepare_experimental_recovery(
        &experimental_recovery(&[None, Some(0), Some(1)]),
        &ctx,
    )
    .unwrap();
    let contract = prepared.contract().unwrap();
    assert!(contract.identities.identification_product.is_some());
    assert!(contract.identities.program.is_some());
    assert!(contract.recovery.is_some(), "an experimental recovery carries a recovery layer");
    assert_eq!(contract.lowered_program_identity, prepared.lowered().identity());
    assert!(matches!(
        contract.section.identification,
        CyclicIdentificationWire::ExperimentalRecovery { .. }
    ));
    assert!(matches!(contract.section.target, CyclicTargetWire::Worlds { .. }));
    let refs = contract.identities.refs();
    for domain in [
        IdentityDomain::Target,
        IdentityDomain::Identification,
        IdentityDomain::IdentificationProduct,
        IdentityDomain::Program,
        IdentityDomain::InferenceBinding,
        IdentityDomain::Observation,
        IdentityDomain::DataSnapshot,
    ] {
        assert!(refs.iter().any(|item| item.domain == domain), "{domain}");
    }

    // One changed experiment value moves the data snapshot, the recovery and the
    // program, and keeps the target and the regime contract.
    let mut changed = experimental_recovery(&[None, Some(0), Some(1)]);
    if let CyclicFamilyRequest::ExperimentalRecovery { samples, .. } = &mut changed {
        samples[0].columns[0].values[0] += 0.25;
    }
    let moved = Study::prepare_experimental_recovery(&changed, &ctx).unwrap().contract().unwrap();
    assert_ne!(moved.identities.data_snapshot, contract.identities.data_snapshot);
    assert_ne!(moved.recovery, contract.recovery);
    assert_ne!(moved.identities.program, contract.identities.program);
    assert_eq!(moved.identities.target, contract.identities.target);
    assert_eq!(
        moved.section.data_snapshot.recovery.as_ref().map(|recovery| recovery.regime_contract),
        contract.section.data_snapshot.recovery.as_ref().map(|recovery| recovery.regime_contract)
    );

    let specified =
        Study::prepare_cyclic_family(&typed_linear(), &ctx).unwrap().contract().unwrap();
    assert!(specified.recovery.is_none());
}

#[test]
fn a_version_two_record_is_verified_against_its_body_and_replayed() {
    let ctx = ctx();
    let prepared = Study::prepare_cyclic_family(&typed_linear(), &ctx).unwrap();
    let (report, bytes) = prepared.export(&ctx).unwrap();
    let (contract, replayed) =
        consume_cyclic_family_contract(&bytes, &report.identity, &ExecutionContext::for_tests(5))
            .unwrap();
    assert_eq!(replayed, report);
    assert_eq!(replayed.lowered_program_identity(), None, "a replay is not a lowered dispatch");
    assert_eq!(contract, prepared.contract().unwrap().section);
    let request_bytes = serde_json::to_vec(&typed_linear()).unwrap();
    assert!(verify_cyclic_contract_against_body(&contract, &request_bytes).is_empty());
    assert_eq!(consume_cyclic_family(&bytes, &report.identity, &ctx).unwrap(), report);
}

#[test]
fn a_tampered_contract_is_refused() {
    let ctx = ctx();
    let request = typed_linear();
    let request_bytes = serde_json::to_vec(&request).unwrap();
    let contract =
        Study::prepare_cyclic_family(&request, &ctx).unwrap().contract().unwrap().section;

    let mut retargeted = contract.clone();
    retargeted.identities.target[0] ^= 1;
    let reasons = verify_cyclic_contract_against_body(&retargeted, &request_bytes);
    assert!(reasons.iter().any(|reason| &**reason == "identities.target"), "{reasons:?}");
    assert!(reasons.iter().any(|reason| &**reason == "seal"), "{reasons:?}");

    let mut unsolved = contract.clone();
    unsolved.solution.policy = "selected_root".into();
    let reasons = verify_cyclic_contract_against_body(&unsolved, &request_bytes);
    assert!(reasons.iter().any(|reason| &**reason == "cyclic_solution"), "{reasons:?}");

    let other = serde_json::to_vec(&linear_worlds(vec![world(0, vec![set(10, 5.0)])])).unwrap();
    let reasons = verify_cyclic_contract_against_body(&contract, &other);
    assert!(reasons.iter().any(|reason| &**reason == "body.request_identity"), "{reasons:?}");
}

mod composition {
    use antecedent_contract::{QuantityRole, ScientificQuantity};

    use super::{ctx, typed_linear};
    use crate::Study;
    use crate::analysis::interface_contract::{
        ComponentConnection, ConsumedPort, ContractOperation, InterfaceContract, ProducedPort,
        SupportDomain,
    };
    use crate::analysis::{check_interface_composition, check_interface_composition_with_cyclic};

    fn quantity() -> ScientificQuantity {
        ScientificQuantity {
            variable_id: "schema:b".into(),
            variable_name: "b".into(),
            role: QuantityRole::Outcome,
            units: "1".into(),
            population_id: "system".into(),
            regime_id: "do(a=2)".into(),
            horizon: 0,
            functional_id: "mean".into(),
            conditioning: Vec::new(),
            transform_id: "identity".into(),
        }
    }

    fn produced() -> ProducedPort {
        ProducedPort::new(
            "out",
            quantity(),
            "cyclic:equilibrium_mean",
            SupportDomain::Real,
            [ContractOperation::Mean],
        )
    }

    fn consumed() -> ConsumedPort {
        ConsumedPort::new(
            "in",
            quantity(),
            "cyclic:equilibrium_mean",
            SupportDomain::Real,
            [ContractOperation::Mean],
        )
    }

    fn component(name: &str, consumes: bool, produces: bool) -> InterfaceContract {
        InterfaceContract {
            component: name.into(),
            produces: if produces { vec![produced()] } else { vec![] },
            consumes: if consumes { vec![consumed()] } else { vec![] },
        }
    }

    #[test]
    fn a_certified_cyclic_product_enters_as_one_component_and_component_cycles_stay_refused() {
        let ctx = ctx();
        let contract =
            Study::prepare_cyclic_family(&typed_linear(), &ctx).unwrap().contract().unwrap();
        let product = contract.as_component("feedback_model");
        let components =
            [component("feedback_model", false, true), component("decision", true, false)];
        let connections = [ComponentConnection::new("feedback_model", "out", "decision", "in")];
        let composed = check_interface_composition_with_cyclic(
            &components,
            &connections,
            std::slice::from_ref(&product),
            None,
        )
        .expect("a certified cyclic product is one feed-forward component");
        assert_eq!(composed.order, ["feedback_model", "decision"]);
        let plain = check_interface_composition(&components, &connections).unwrap();
        assert_ne!(composed.identity, plain.identity, "the product's identity is bound");

        let unknown = contract.as_component("elsewhere");
        let refusals =
            check_interface_composition_with_cyclic(&components, &connections, &[unknown], None)
                .expect_err("a product names a declared component");
        assert_eq!(refusals[0].detail, "interface_contract.unknown_component");

        let looped = [component("feedback_model", true, true), component("decision", true, true)];
        let loop_connections = [
            ComponentConnection::new("feedback_model", "out", "decision", "in"),
            ComponentConnection::new("decision", "out", "feedback_model", "in"),
        ];
        let refusals =
            check_interface_composition_with_cyclic(&looped, &loop_connections, &[product], None)
                .expect_err("feedback among components is refused");
        assert_eq!(refusals[0].code, "interface_contract_cyclic_composition");
        assert_eq!(
            refusals[0].offending.as_deref(),
            Some("decision -> feedback_model -> decision")
        );
        assert!(
            refusals[0].remedy.as_deref().is_some_and(|remedy| remedy.contains("CyclicComponent"))
        );
    }
}
