//! Independent finite-SCM evidence for negative restricted-experiment claims.
//!
//! These tests enumerate exogenous states directly. They do not use the
//! identifier or expression evaluator to calculate any probability.
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use antecedent_core::{
    DependenceGroup, DistributionAvailability, Environment, EvidenceCatalog, EvidenceKind,
    EvidenceRegime, ExecutionContext, InterventionAssignment, RegimeBinding, RegimeId, RegimeKind,
    SamplingDesign, Value, VariableCoordinate, VariableDomain, VariableId,
};
use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram};
use antecedent_identify::{
    ClassicalTransportQuery, ClassicalTransportResult, SidLimits, ZTransportDecision,
    ZTransportMissingEvidence, ZTransportQuery, decide_z_transport_with_catalog,
    identify_classical_transport, validate_z_transport_query,
};
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
enum OutcomeMechanism {
    CopiesObservedTreatment,
    CopiesTreatmentLatent,
}

#[derive(Clone, Copy, Debug)]
struct Regime {
    /// `None` means observational; otherwise the intervention is applied.
    do_x: Option<u8>,
    do_z: Option<u8>,
    do_w: Option<u8>,
}

type Law = [f64; 16];

fn enumerate_law(mechanism: OutcomeMechanism, regime: Regime) -> Law {
    let mut law = [0.0; 16];
    // U, Vz, and Vw are mutually independent fair binary exogenous variables.
    for u in 0..=1_u8 {
        for vz in 0..=1_u8 {
            for vw in 0..=1_u8 {
                let x = regime.do_x.unwrap_or(u);
                let y = match mechanism {
                    OutcomeMechanism::CopiesObservedTreatment => x,
                    OutcomeMechanism::CopiesTreatmentLatent => u,
                };
                let z = regime.do_z.unwrap_or(vz);
                let w = regime.do_w.unwrap_or(vw);
                let cell = (x as usize) * 8 + (y as usize) * 4 + (z as usize) * 2 + w as usize;
                law[cell] += 1.0 / 8.0;
            }
        }
    }
    law
}

fn target_y_risk(mechanism: OutcomeMechanism, x: u8) -> f64 {
    (0..=1_u8)
        .map(|u| {
            let y = match mechanism {
                OutcomeMechanism::CopiesObservedTreatment => x,
                OutcomeMechanism::CopiesTreatmentLatent => u,
            };
            f64::from(y) * 0.5
        })
        .sum()
}

fn all_binary_regimes_for_zw() -> Vec<Regime> {
    let mut regimes = vec![Regime { do_x: None, do_z: None, do_w: None }];
    // Complete source family for every non-empty subset of {Z, W}, including
    // both joint assignments. Intervention levels are enumerated exhaustively.
    for z in 0..=1 {
        regimes.push(Regime { do_x: None, do_z: Some(z), do_w: None });
    }
    for w in 0..=1 {
        regimes.push(Regime { do_x: None, do_z: None, do_w: Some(w) });
    }
    for z in 0..=1 {
        for w in 0..=1 {
            regimes.push(Regime { do_x: None, do_z: Some(z), do_w: Some(w) });
        }
    }
    regimes
}

fn fig2b_style_contract() -> (SelectionDiagram, ZTransportQuery) {
    // X -> Y and X <-> Y form the treatment/outcome hedge. Z and W are
    // disconnected from that component. There are no selection targets, so a
    // source intervention on X would transport under the invariant graph.
    let mut graph = Admg::with_variables(4);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    graph.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let diagram = SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from([])).unwrap();
    let query = ZTransportQuery {
        outcomes: Arc::from([VariableId::from_raw(1)]),
        treatments: Arc::from([VariableId::from_raw(0)]),
        controllable: Arc::from([VariableId::from_raw(2), VariableId::from_raw(3)]),
        // The obstruction is about the full experimental family; no one
        // particular source assignment is selected as a formula input.
        experiment_assignment: Arc::from([]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    };
    (diagram, query)
}

fn full_experiment_catalog(
    omit_joint_regime: bool,
    omit_target_observation: bool,
) -> EvidenceCatalog {
    let variables = (0..4)
        .map(|raw| VariableCoordinate {
            variable: VariableId::from_raw(raw),
            domain: VariableDomain::Binary,
            unit: None,
        })
        .collect::<Vec<_>>();
    let source = Environment::try_new("source", variables.clone(), []).unwrap();
    let target = Environment::try_new("target", variables, []).unwrap();
    let measured: Vec<_> = (0..4).map(VariableId::from_raw).collect();
    let mut regimes = Vec::new();
    let mut bindings = Vec::new();
    let mut next_id = 0_u32;
    let mut add_regime = |kind, population: &str, assignments: Vec<(u32, u8)>| {
        if omit_joint_regime && assignments.len() == 2 && assignments == [(2, 0), (3, 0)] {
            return;
        }
        let id = RegimeId::from_raw(next_id);
        next_id += 1;
        let interventions = assignments
            .iter()
            .map(|(variable, _)| VariableId::from_raw(*variable))
            .collect::<Vec<_>>();
        let values = assignments
            .iter()
            .map(|(variable, level)| InterventionAssignment {
                variable: VariableId::from_raw(*variable),
                value: Value::Bool(*level == 1),
            })
            .collect::<Vec<_>>();
        regimes.push(
            EvidenceRegime::try_new(
                id,
                kind,
                EvidenceKind::Available,
                interventions,
                values,
                measured.clone(),
                population,
                DistributionAvailability::Joint,
            )
            .unwrap(),
        );
        bindings.push(RegimeBinding {
            dataset_identity: None,
            regime: id,
            snapshot_identity: Arc::from(format!("snapshot-{}", id.raw())),
            schema_names: Arc::from([]),
            sampling: SamplingDesign::Independent,
            weights: None,
            dependence: DependenceGroup::IndependentStudies,
        });
    };
    if !omit_target_observation {
        add_regime(RegimeKind::Observational, "target", Vec::new());
    }
    // Enumerate every nonempty intervention subset of {Z,W} and all of its
    // concrete binary assignments, including each joint do(Z,W) law.
    for mask in 1..4_u32 {
        let selected = (0..2_u32).filter(|bit| mask & (1 << bit) != 0).collect::<Vec<_>>();
        let assignment_count = 1_u32 << selected.len();
        for assignment_mask in 0..assignment_count {
            let assignments = selected
                .iter()
                .enumerate()
                .map(|(index, bit)| (2 + *bit, ((assignment_mask >> index) & 1) as u8))
                .collect();
            add_regime(RegimeKind::Experimental, "source", assignments);
        }
    }
    EvidenceCatalog::try_new([source, target], regimes, bindings, None).unwrap()
}

#[test]
fn two_models_match_target_observations_and_complete_zw_experiments_but_disagree_on_do_x() {
    let first = OutcomeMechanism::CopiesObservedTreatment;
    let second = OutcomeMechanism::CopiesTreatmentLatent;
    let observational = Regime { do_x: None, do_z: None, do_w: None };

    let (diagram, query) = fig2b_style_contract();
    validate_z_transport_query(&diagram, &query).unwrap();
    let unrestricted = identify_classical_transport(
        &diagram,
        &ClassicalTransportQuery {
            outcomes: Arc::from([VariableId::from_raw(1)]),
            treatments: Arc::from([VariableId::from_raw(0)]),
            source: Arc::from("source"),
            target: Arc::from("target"),
        },
        SidLimits::default(),
        &ExecutionContext::for_tests(17),
    )
    .unwrap();
    assert!(
        matches!(unrestricted, ClassicalTransportResult::Identified(_)),
        "unrestricted source experiments should identify through the no-selection graph"
    );

    let limits = SidLimits::default();
    let ctx = ExecutionContext::for_tests(18);
    let catalog = full_experiment_catalog(false, false);
    let decision =
        decide_z_transport_with_catalog(&diagram, &query, &catalog, limits, &ctx).unwrap();
    let ZTransportDecision::ProvenNonTransportable(obstruction) = decision else {
        panic!("complete {query:?} source family must reach a checked TRz obstruction")
    };
    let record = obstruction.to_record();
    let replayed = antecedent_identify::ZTransportObstruction::from_record_checked(
        record.clone(),
        &diagram,
        &query,
        &catalog,
        limits,
        &ExecutionContext::for_tests(19),
    )
    .unwrap();
    assert_eq!(record, replayed.to_record());

    let mut changed_terminal = record.clone();
    changed_terminal.terminal.rules.push("tampered.rule".into());
    assert!(
        antecedent_identify::ZTransportObstruction::from_record_checked(
            changed_terminal,
            &diagram,
            &query,
            &catalog,
            limits,
            &ExecutionContext::for_tests(21),
        )
        .is_err(),
        "changed terminal reasoning must not replay as a checked obstruction"
    );
    let mut changed_family = record.clone();
    changed_family.family_regimes.pop();
    assert!(
        antecedent_identify::ZTransportObstruction::from_record_checked(
            changed_family,
            &diagram,
            &query,
            &catalog,
            limits,
            &ExecutionContext::for_tests(22),
        )
        .is_err(),
        "changed source-family identities must not replay as a checked obstruction"
    );

    let exhausted = decide_z_transport_with_catalog(
        &diagram,
        &query,
        &catalog,
        SidLimits { steps: 0, depth: limits.depth },
        &ExecutionContext::for_tests(23),
    )
    .unwrap_err();
    assert_eq!(exhausted.to_string(), "z_transport.exhausted_computation");

    let incomplete = full_experiment_catalog(true, false);
    assert!(matches!(
        decide_z_transport_with_catalog(
            &diagram,
            &query,
            &incomplete,
            limits,
            &ExecutionContext::for_tests(20),
        )
        .unwrap(),
        ZTransportDecision::MissingEvidence {
            missing: ZTransportMissingEvidence::SourceExperiment(_),
        }
    ));

    let missing_target = full_experiment_catalog(false, true);
    assert!(matches!(
        decide_z_transport_with_catalog(
            &diagram,
            &query,
            &missing_target,
            limits,
            &ExecutionContext::for_tests(24),
        )
        .unwrap(),
        ZTransportDecision::MissingEvidence {
            missing: ZTransportMissingEvidence::TargetObservationalJoint,
        }
    ));

    // The target observational law is identical in the two SCMs.
    assert_eq!(enumerate_law(first, observational), enumerate_law(second, observational));

    // Source experiments cover every assignment for each non-empty subset of
    // the declared controllable set {Z, W}, including the joint do(Z,W) laws.
    for regime in all_binary_regimes_for_zw() {
        assert_eq!(enumerate_law(first, regime), enumerate_law(second, regime));
    }

    // Yet their target interventional outcome laws disagree. In the first SCM
    // Y follows X; in the second it follows the unobserved common cause U.
    assert_eq!(target_y_risk(first, 0), 0.0);
    assert_eq!(target_y_risk(first, 1), 1.0);
    assert_eq!(target_y_risk(second, 0), 0.5);
    assert_eq!(target_y_risk(second, 1), 0.5);
}

#[test]
fn completed_search_with_a_control_connected_to_the_hedge_stays_uncertified() {
    let (mut diagram, query) = fig2b_style_contract();
    let mut graph = diagram.causal_graph().clone();
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    diagram = SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from([])).unwrap();

    let decision = decide_z_transport_with_catalog(
        &diagram,
        &query,
        &full_experiment_catalog(false, false),
        SidLimits::default(),
        &ExecutionContext::for_tests(27),
    )
    .unwrap();
    assert!(matches!(
        decision,
        ZTransportDecision::NotCertified {
            reason: "z_transport.negative_outside_checked_hedge_subset"
        }
    ));
}

#[test]
fn disconnected_control_obstruction_contains_a_replayable_target_hedge() {
    let (diagram, query) = fig2b_style_contract();
    let selected = SelectionDiagram::try_new(
        diagram.causal_graph().clone(),
        Arc::<[VariableId]>::from((0..4).map(VariableId::from_raw).collect::<Vec<_>>()),
    )
    .unwrap();
    let classical_query = ClassicalTransportQuery {
        outcomes: Arc::clone(&query.outcomes),
        treatments: Arc::clone(&query.treatments),
        source: Arc::clone(&query.source),
        target: Arc::clone(&query.target),
    };
    let result = identify_classical_transport(
        &selected,
        &classical_query,
        SidLimits::default(),
        &ExecutionContext::for_tests(29),
    )
    .unwrap();
    let ClassicalTransportResult::ProvenNonTransportable(hedge) = result else {
        panic!("the all-selected diagram must return its checked nested-forest witness")
    };
    assert_eq!(hedge.larger.nodes.as_ref(), &[VariableId::from_raw(0), VariableId::from_raw(1)]);
    assert_eq!(hedge.smaller.nodes.as_ref(), &[VariableId::from_raw(1)]);

    // Dropping the selector-membership premise leaves the ordinary ID hedge:
    // F={X,Y}, F'={Y}, common root Y, with X only in F.
    assert!(hedge.larger.nodes.contains(&query.treatments[0]));
    assert!(!hedge.smaller.nodes.contains(&query.treatments[0]));
    assert!(hedge.smaller.nodes.iter().all(|node| hedge.larger.nodes.contains(node)));
}

#[test]
fn disconnected_controls_cannot_resolve_a_selected_hedge_component() {
    let (diagram, query) = fig2b_style_contract();
    let diagram = SelectionDiagram::try_new(
        diagram.causal_graph().clone(),
        Arc::<[VariableId]>::from([VariableId::from_raw(1)]),
    )
    .unwrap();
    let decision = decide_z_transport_with_catalog(
        &diagram,
        &query,
        &full_experiment_catalog(false, false),
        SidLimits::default(),
        &ExecutionContext::for_tests(31),
    )
    .unwrap();
    assert!(matches!(decision, ZTransportDecision::ProvenNonTransportable(_)));

    // The selected target component still has the observationally equivalent
    // SCM pair, while controls Z and W live in separate ADMG components.
    assert_eq!(
        enumerate_law(
            OutcomeMechanism::CopiesObservedTreatment,
            Regime { do_x: None, do_z: Some(0), do_w: Some(1) }
        ),
        enumerate_law(
            OutcomeMechanism::CopiesTreatmentLatent,
            Regime { do_x: None, do_z: Some(0), do_w: Some(1) }
        )
    );
    assert_ne!(
        target_y_risk(OutcomeMechanism::CopiesObservedTreatment, 0),
        target_y_risk(OutcomeMechanism::CopiesTreatmentLatent, 0)
    );
}

#[test]
fn a_source_experiment_that_controls_x_separates_the_two_models() {
    let first = OutcomeMechanism::CopiesObservedTreatment;
    let second = OutcomeMechanism::CopiesTreatmentLatent;
    for x in 0..=1 {
        let regime = Regime { do_x: Some(x), do_z: None, do_w: None };
        assert_ne!(enumerate_law(first, regime), enumerate_law(second, regime));
        assert_eq!(target_y_risk(first, x), f64::from(x));
        assert_eq!(target_y_risk(second, x), 0.5);
    }
}

#[test]
fn negative_decision_refuses_a_continuous_observed_coordinate() {
    let (diagram, query) = fig2b_style_contract();
    let mut catalog = full_experiment_catalog(false, false);
    let mut environments = catalog.environments.to_vec();
    for environment in &mut environments {
        let mut coordinates = environment.variables.to_vec();
        coordinates[1].domain = VariableDomain::Continuous;
        *environment = Environment::try_new(environment.identity.clone(), coordinates, []).unwrap();
    }
    catalog.environments = environments.into();
    let error = decide_z_transport_with_catalog(
        &diagram,
        &query,
        &catalog,
        SidLimits::default(),
        &ExecutionContext::for_tests(25),
    )
    .unwrap_err();
    assert!(error.to_string().starts_with("z_transport.unsupported_domain"), "{error}");
}
