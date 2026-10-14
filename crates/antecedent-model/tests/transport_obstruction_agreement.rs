//! The composition-side refusal of a carried mechanism and the finite
//! counterfactual-transport obstruction are two views of one declared
//! selection diagram. This test runs both on the shared fixture and requires
//! them to agree, configuration by configuration:
//!
//! * a selection node into the outcome: the composition refuses to carry the
//!   outcome mechanism, and the finite derivation returns a verified two-model
//!   obstruction for the effect of the treatment from the source experiment;
//! * a selection node into the treatment only: the composition carries the
//!   outcome mechanism, and the finite derivation identifies the effect.
//!
//! Both sides are driven by the same `selection_targets` of the same diagram;
//! the finite problem declares the confounder as the bidirected edge the
//! fixture's shared latent cause stands for.
//!
//! Needs `antecedent-identify` as a dev-dependency of this crate.

mod transport_fixture;

use std::sync::Arc;

use antecedent_core::{
    CounterfactualEvent, CounterfactualEventQuery, DistributionAvailability, Environment,
    EvidenceCatalog, EvidenceKind, EvidenceRegime, ExecutionContext, ExogenousCoupling,
    InterventionAssignment, RegimeBinding, RegimeId, RegimeKind, VariableId, WorldId, WorldSpec,
};
use antecedent_identify::{
    COUNTERFACTUAL_TRANSPORT_DEFAULT_LIMITS, CounterfactualTransportDecision,
    CounterfactualTransportProblem, CounterfactualTransportQuery, CounterfactualTransportSource,
    identify_counterfactual_transport,
};
use antecedent_model::mechanism_transport::{
    DETAIL_SELECTION_INTO_MECHANISM, MechanismTransportRequest, selection_targets,
    transport_linear_gaussian,
};
use antecedent_model::compose::PosteriorDependence;
use transport_fixture::{
    SOURCE, Selection, TARGET, VARIABLES, diagram, edges, outcome_invariance, source_x, source_y,
    target_context, target_x, variable_id,
};

fn v(id: u32) -> VariableId {
    VariableId::from_raw(id)
}

fn regime(id: u32, population: &str, interventions: &[u32], measured: &[u32]) -> EvidenceRegime {
    let kind = if interventions.is_empty() { RegimeKind::Observational } else { RegimeKind::Experimental };
    EvidenceRegime::try_new(
        RegimeId::from_raw(id),
        kind,
        EvidenceKind::Available,
        interventions.iter().map(|&i| v(i)).collect::<Vec<_>>(),
        Vec::<InterventionAssignment>::new(),
        measured.iter().map(|&i| v(i)).collect::<Vec<_>>(),
        population,
        DistributionAvailability::Joint,
    )
    .unwrap()
}

/// The finite derivation's decision for `E[y | do(x = 1)]` in the target, from
/// the target observational law, a source experiment on `x` and the source
/// observational law, under the same selection diagram as the composition.
fn finite_decision(selection: Selection) -> CounterfactualTransportDecision {
    let x = variable_id("x");
    let y = variable_id("y");
    let problem = CounterfactualTransportProblem::new(
        &VARIABLES.iter().map(|(_, id)| (v(*id), vec![0.0, 1.0])).collect::<Vec<_>>(),
        &[(v(x), v(y))],
        &[(v(x), v(y))],
    )
    .unwrap();
    let catalog = EvidenceCatalog::try_new(
        Vec::<Environment>::new(),
        vec![
            regime(0, TARGET, &[], &[x, y]),
            regime(1, SOURCE, &[x], &[y]),
            regime(2, SOURCE, &[], &[x, y]),
        ],
        Vec::<RegimeBinding>::new(),
        None,
    )
    .unwrap();
    // The same diagram drives the finite side's selection targets.
    let selection_ids: Vec<VariableId> =
        selection_targets(&diagram(selection)).iter().map(|name| v(variable_id(name))).collect();
    let event = CounterfactualEventQuery::new(
        vec![WorldSpec::new([(v(x), 1.0)], []).unwrap()],
        ExogenousCoupling::SharedLatentExogenous,
        [CounterfactualEvent::new(WorldId::new(0), v(y), 1.0).unwrap()],
        [],
    )
    .unwrap();
    let query = CounterfactualTransportQuery {
        event,
        target: Arc::from(TARGET),
        sources: vec![CounterfactualTransportSource {
            population: Arc::from(SOURCE),
            selection_targets: selection_ids.into(),
            controllable: vec![v(x)].into(),
        }]
        .into(),
    };
    let ctx = ExecutionContext::for_tests(1);
    identify_counterfactual_transport(
        &problem,
        &query,
        &catalog,
        COUNTERFACTUAL_TRANSPORT_DEFAULT_LIMITS,
        &ctx,
    )
    .unwrap()
}

fn composition_request(selection: Selection) -> MechanismTransportRequest {
    MechanismTransportRequest {
        target: target_context(),
        diagram: diagram(selection),
        source_mechanisms: vec![source_x(), source_y()],
        // The treatment mechanism is refitted in the target in both
        // configurations: the finite derivation reads the target law of x.
        target_mechanisms: vec![target_x()],
        edges: edges(),
        invariances: vec![outcome_invariance()],
        dependence: PosteriorDependence::Unknown,
    }
}

#[test]
fn a_selection_arrow_into_the_outcome_is_a_composition_refusal_and_a_verified_obstruction() {
    let selection = Selection::OutcomeSelected;

    let refusal = transport_linear_gaussian(&composition_request(selection)).unwrap_err();
    let refused: Vec<(&str, &str)> =
        refusal.findings().iter().map(|f| (f.detail, f.subject.as_str())).collect();
    assert_eq!(refused, vec![(DETAIL_SELECTION_INTO_MECHANISM, "y")]);

    match finite_decision(selection) {
        CounterfactualTransportDecision::ProvenNonTransportable(obstruction) => {
            let check = obstruction.verify().unwrap();
            assert_ne!(check.first_value, check.second_value);
        }
        other => panic!("expected an obstruction, got {other:?}"),
    }
}

#[test]
fn a_selection_arrow_into_the_treatment_only_carries_the_mechanism_and_identifies_the_effect() {
    let selection = Selection::TreatmentSelected;

    let carried = transport_linear_gaussian(&composition_request(selection)).unwrap();
    assert_eq!(carried.transported_variables(), vec!["y"]);

    let decision = finite_decision(selection);
    assert!(decision.derivation().is_some(), "{decision:?}");
}

#[test]
fn both_views_depend_on_the_diagram_alone_not_on_the_label_of_the_selected_variable() {
    // Swapping which variable carries the selection node swaps both verdicts
    // together; there is no configuration on which they disagree.
    for selection in Selection::ALL {
        let composition_refuses = transport_linear_gaussian(&composition_request(selection)).is_err();
        let finite_obstructed = matches!(
            finite_decision(selection),
            CounterfactualTransportDecision::ProvenNonTransportable(_)
        );
        assert_eq!(composition_refuses, finite_obstructed, "{selection:?}");
    }
}
