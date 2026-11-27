//! Public compiler, decision and portable replay of patient-level principal surrogacy.
#![allow(clippy::float_cmp, reason = "fixtures intentionally pin exact replay and declared values")]

use antecedent::Study;
use antecedent::analysis::composition::{self, BundleBuilder, SuppliedSources};
use antecedent::analysis::kernel_programs::patient_surrogate::{
    ObservedSurrogatePatient, PatientSurrogateDesign, PatientSurrogateError as E,
    PatientSurrogateEvidence as Input, PatientSurrogateProgramError as P, PatientSurrogateReading,
    PatientSurrogateRequest, PrincipalSurrogateError, SurrogateThresholdReading,
    consume_patient_surrogate,
};
use antecedent_core::ExecutionContext;

fn design() -> PatientSurrogateDesign {
    PatientSurrogateDesign {
        trial_id: "trial-001".into(),
        population: "adults".into(),
        surrogate: "early".into(),
        outcome: "terminal_benefit".into(),
        surrogate_visit: 1,
        outcome_visit: 4,
        control: "usual-care".into(),
        treatment: "active".into(),
        consistency_and_no_interference: true,
        randomized_exchangeability: true,
        complete_joint_follow_up: true,
        monotone_surrogate: true,
    }
}

fn law(control_cells: [u32; 4], treatment_cells: [u32; 4]) -> PatientSurrogateRequest {
    PatientSurrogateRequest {
        design: design(),
        evidence: Input::PopulationJointLaw {
            evidence_id: "joint-population-law".into(),
            control_cells,
            treatment_cells,
        },
    }
}

#[test]
fn equal_trial_and_joint_arm_summaries_allow_opposite_individual_surrogate_answers() {
    // Rows and columns are (S,Y) = 00,01,10,11. Each latent table has four patient types
    // of equal mass. Their fully joint observable arm laws, not just the means, agree.
    let no_improver_benefit = [(0, 0), (0, 2), (1, 3), (2, 3)];
    let all_improvers_benefit = [(0, 3), (0, 3), (1, 0), (2, 2)];
    let observable = |table: [(usize, usize); 4]| {
        let mut arms = [[0_u32; 4]; 2];
        for (control, treated) in table {
            arms[0][control] += 1;
            arms[1][treated] += 1;
        }
        arms
    };
    let hidden = |table: [(usize, usize); 4]| {
        let improvers = table.iter().filter(|(a, b)| *a < 2 && *b >= 2).count();
        let beneficiers = table.iter().filter(|&&(a, b)| a == 0 && b == 3).count();
        (beneficiers, improvers)
    };
    assert_eq!(observable(no_improver_benefit), observable(all_improvers_benefit));
    assert_eq!(hidden(no_improver_benefit), (0, 2));
    assert_eq!(hidden(all_improvers_benefit), (2, 2));
    let [control, treated] = observable(no_improver_benefit);
    let ctx = ExecutionContext::for_tests(16);
    let plan = Study::compile_patient_surrogate(law(control, treated), &ctx).unwrap();
    assert_eq!(plan.lowered().route(), "PatientSurrogateIdentification");
    let result = plan.execute(&ctx).unwrap();
    assert_eq!(result.reading(), PatientSurrogateReading::PopulationIdentification);
    assert_eq!(result.bounds().endpoints(), [0.0, 1.0]);
    assert_eq!(result.bounds().improver_probability(), 0.5);
    assert!(matches!(result.point(), Err(E::PointNotIdentified)));
    assert_eq!(result.decide(1, 2).unwrap().reading, SurrogateThresholdReading::Ambiguous);
}

#[test]
fn retained_patient_artifact_binds_bytes_before_program_and_result() {
    use antecedent::analysis::kernel_programs::patient_surrogate::StoredPatientSurrogate;
    use antecedent_io::{IoError, RetainedArtifactIdentity};

    let ctx = ExecutionContext::for_tests(16);
    let plan = Study::compile_patient_surrogate(law([3, 0, 1, 0], [0, 0, 1, 3]), &ctx).unwrap();
    let program = plan.lowered().identity();
    let (bytes, result) = plan.export_with_identity("baseline", &ctx).unwrap();
    let retained = RetainedArtifactIdentity::of_bytes(&bytes);
    let decoded = StoredPatientSurrogate::decode_expecting(&bytes, &retained).unwrap();
    assert_eq!(decoded.program, program);
    let stored =
        StoredPatientSurrogate::open_expecting(&bytes, program, result, &retained).unwrap();
    stored.verify(program, &plan.execute(&ctx).unwrap()).unwrap();

    let (relabeled, same_result) = plan.export_with_identity("follow-up", &ctx).unwrap();
    assert_eq!(same_result, result);
    assert_ne!(bytes, relabeled);
    StoredPatientSurrogate::open(&relabeled, program, result).unwrap();
    for candidate in [relabeled.as_slice(), b"invalid container".as_slice()] {
        for read in [
            StoredPatientSurrogate::decode_expecting(candidate, &retained),
            StoredPatientSurrogate::open_expecting(candidate, program, result, &retained),
        ] {
            assert!(matches!(
                read,
                Err(E::Artifact(IoError::Refused { code: "external_binding_mismatch", .. }))
            ));
        }
    }
    assert!(matches!(
        StoredPatientSurrogate::open_expecting(&bytes, program, program, &retained),
        Err(E::ReplayMismatch)
    ));
}

#[test]
fn threshold_decision_consumes_exact_bounds_and_replays_under_both_identities() {
    let ctx = ExecutionContext::for_tests(16);
    let input = law([3, 0, 1, 0], [0, 0, 1, 3]);
    let plan = Study::compile_patient_surrogate(input, &ctx).unwrap();
    let expected_program = plan.lowered().identity();
    let original = plan.execute(&ctx).unwrap();
    let bytes = plan.export("patient-surrogate", &ctx).unwrap();
    let replayed =
        consume_patient_surrogate(&bytes, expected_program, original.identity(), &ctx).unwrap();
    let bounds = replayed.bounds();
    assert_eq!(
        (bounds.lower_numerator, bounds.upper_numerator, bounds.improver_numerator),
        (8, 12, 12)
    );
    let tied = replayed.decide(2, 3).unwrap();
    assert_eq!(tied.reading, SurrogateThresholdReading::AllAtOrAbove);
    assert_eq!(tied.source, original.identity());
    assert_eq!(replayed.decide(4, 5).unwrap().reading, SurrogateThresholdReading::Ambiguous);
    assert_eq!(replayed.identified_set_source().unwrap().claim_identity, original.identity());

    let mut builder = BundleBuilder::new();
    builder.add_detected_artifact(Some("surrogate"), &bytes).unwrap();
    let graph = builder.build().unwrap();
    let graph_bytes = composition::export_bundle(&graph, "surrogate-graph").unwrap();
    let consumed =
        composition::consume(&graph_bytes, graph.identity(), &SuppliedSources::default()).unwrap();
    consumed.require_verified().unwrap();
    assert_eq!(consumed.value("surrogate", "lower"), Some(2.0 / 3.0));
    assert_eq!(consumed.value("surrogate", "upper"), Some(1.0));

    let alternate =
        Study::compile_patient_surrogate(law([1, 2, 1, 0], [1, 0, 1, 2]), &ctx).unwrap();
    let other = alternate.export("replacement", &ctx).unwrap();
    assert!(matches!(
        consume_patient_surrogate(&other, expected_program, original.identity(), &ctx),
        Err(P::Scientific(E::ReplayMismatch))
    ));
    assert_eq!(
        alternate.execute(&ctx).unwrap().decide(3, 4).unwrap().reading,
        SurrogateThresholdReading::AllBelow
    );
}

#[test]
fn empirical_joint_counts_cannot_license_a_population_point_or_decision() {
    let ctx = ExecutionContext::for_tests(16);
    let mut sample = law([0, 4, 0, 0], [0, 0, 0, 4]);
    let point =
        Study::compile_patient_surrogate(sample.clone(), &ctx).unwrap().execute(&ctx).unwrap();
    assert_eq!(point.point().unwrap(), 0.0);
    sample.evidence = Input::CompletePatientSample {
        patients: vec![
            ObservedSurrogatePatient {
                id: "reference".into(),
                treated: false,
                surrogate: false,
                outcome: true,
            },
            ObservedSurrogatePatient {
                id: "active".into(),
                treated: true,
                surrogate: true,
                outcome: true,
            },
        ],
    };
    let sample = Study::compile_patient_surrogate(sample, &ctx).unwrap().execute(&ctx).unwrap();
    assert_eq!(sample.bounds().endpoints(), [0.0, 0.0]);
    assert_eq!(sample.reading(), PatientSurrogateReading::EmpiricalPlugin);
    assert!(matches!(sample.point(), Err(E::DecisionReading)));
    assert!(matches!(sample.decide(0, 1), Err(E::DecisionReading)));

    let empty = Study::compile_patient_surrogate(law([2, 0, 2, 0], [0, 2, 0, 2]), &ctx).unwrap();
    assert!(matches!(
        empty.execute(&ctx),
        Err(P::Scientific(E::Identification(PrincipalSurrogateError::EmptyImproverStratum)))
    ));
    let mut incomplete = law([0, 4, 0, 0], [0, 0, 0, 4]);
    incomplete.design.complete_joint_follow_up = false;
    assert!(matches!(
        Study::compile_patient_surrogate(incomplete, &ctx),
        Err(P::Scientific(E::Declaration(_)))
    ));
}

#[test]
fn identified_set_bridge_requires_the_replayed_patient_bounds() {
    use antecedent::analysis::composition::NodeStatus;
    use antecedent::analysis::kernel_programs::bridge_images::BridgeImageRequest;
    use antecedent_contract::bridge::{
        AssumptionKind, BridgeAssumption, BridgeContract, BridgeEndpoint, BridgeKind,
        BridgeMapping, BridgeScope, FixedBridgeDomain, QuantityClass, ValueInterval, ValueKind,
    };

    let ctx = ExecutionContext::for_tests(16);
    let plan = Study::compile_patient_surrogate(law([3, 0, 1, 0], [0, 0, 1, 3]), &ctx).unwrap();
    let patient = plan.export("original-patient", &ctx).unwrap();
    let identified = plan.execute(&ctx).unwrap().identified_set_source().unwrap();
    let mut utility = identified.quantity.clone();
    utility.variable_id = "utility".into();
    utility.variable_name = "utility".into();
    utility.units = "score".into();
    let contract = BridgeContract {
        kind: BridgeKind::ScientificDecisionUtility,
        source: BridgeEndpoint {
            quantity: identified.quantity.clone(),
            class: QuantityClass::ScientificEndpoint,
        },
        target: BridgeEndpoint { quantity: utility, class: QuantityClass::DecisionUtility },
        source_value: ValueKind::IdentifiedSet,
        target_value: ValueKind::IdentifiedSet,
        evidence_scope: BridgeScope {
            population_id: identified.quantity.population_id.clone(),
            regime_id: identified.quantity.regime_id.clone(),
        },
        mapping: BridgeMapping::Affine { intercept: 1.0, slope: 3.0 },
        assumptions: vec![BridgeAssumption {
            id: "utility-scale".into(),
            kind: AssumptionKind::UtilityMapping,
            statement: "one unit of the endpoint contributes three units of utility".into(),
            evidence: Some("declared-choice-rule".into()),
        }],
        transport: None,
        mapping_uncertainty: None,
        source_dependence: None,
        estimand_bridge: None,
    };
    for changed in [false, true] {
        let mut source = identified.clone();
        if changed {
            source.set.lower = 0.5;
        }
        let image = Study::compile_bridge_image(
            BridgeImageRequest::IdentifiedSet {
                bridge: Box::new(contract.clone()),
                source,
                domain: FixedBridgeDomain(ValueInterval { low: 0.0, high: 1.0 }),
            },
            &ctx,
        )
        .unwrap()
        .export("mapped-patient", &ctx)
        .unwrap();

        let mut builder = BundleBuilder::new();
        builder.add_detected_artifact(Some("patient"), &patient).unwrap();
        builder.add_detected_artifact(Some("image"), &image).unwrap();
        builder.connect("patient", "image").unwrap();
        let bundle = builder.build().unwrap();
        let replayed = composition::consume(
            &composition::export_bundle(&bundle, "patient-image").unwrap(),
            bundle.identity(),
            &SuppliedSources::default(),
        )
        .unwrap();
        if changed {
            assert!(matches!(replayed.node("image").unwrap().status, NodeStatus::Failed { .. }));
        } else {
            assert!(replayed.all_replayed());
            assert_eq!(replayed.value("image", "lower"), Some(3.0));
            assert_eq!(replayed.value("image", "upper"), Some(4.0));
        }
    }
}
