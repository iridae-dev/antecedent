// SPDX-License-Identifier: MIT OR Apache-2.0
//! Operation projections from the compiled plan layers.

use super::{
    Composition, EvidenceDecisionProgram, OperationLayer, OperationView, PhysicalView,
    StatisticalProgram,
};

fn composition_label(composition: Composition) -> String {
    match composition {
        Composition::Single => "single".to_owned(),
        Composition::AtomMixture { atoms } => format!("atom_mixture({atoms})"),
        Composition::ScenarioSet { scenarios } => format!("scenario_set({scenarios})"),
        Composition::Transported => "transported".to_owned(),
        Composition::Pooled { sources } => format!("pooled({sources})"),
        Composition::Chained { stages } => format!("chained({stages})"),
    }
}

/// The statistical operations of a program: its operators, the models it fits and
/// the folds, resampling and posterior computation it runs.
pub(super) fn statistical_operations(program: &StatisticalProgram) -> Vec<OperationView> {
    let mut operations = Vec::new();
    for operator in program.operators() {
        let mut view = OperationView::new(
            OperationLayer::Statistical,
            operator.effect().as_str(),
            operator.graph().as_str(),
        );
        view.inputs = operator.inputs().iter().map(|role| role.as_str()).collect();
        view.outputs = operator.outputs().iter().map(|role| role.as_str()).collect();
        operations.push(view);
    }
    for model in program.models() {
        let mut view =
            OperationView::new(OperationLayer::Statistical, "fit_model", model.role().as_str());
        view.outputs = vec!["fit_handle"];
        operations.push(view);
    }
    if let Some(folds) = program.folds() {
        operations.push(OperationView::new(
            OperationLayer::Statistical,
            "cross_fit",
            format!("{} folds by {}", folds.folds(), folds.unit().as_str()),
        ));
    }
    if let Some(resampling) = program.resampling() {
        operations.push(OperationView::new(
            OperationLayer::Statistical,
            "resample",
            format!(
                "{} {} replicates by {}",
                resampling.replicates(),
                resampling.method().as_str(),
                resampling.unit().as_str()
            ),
        ));
    }
    if let Some(posterior) = program.posterior() {
        operations.push(OperationView::new(
            OperationLayer::Statistical,
            "compute_posterior",
            format!("{} with {} draws", posterior.algorithm().as_str(), posterior.draws()),
        ));
    }
    operations
}

/// The evidence and decision operations of a program: reading each source, composing
/// them, the decision and its value of information, the simulation of a planned
/// experiment and each stage of a propagation chain.
pub(super) fn evidence_operations(program: &EvidenceDecisionProgram) -> Vec<OperationView> {
    let mut operations = Vec::new();
    for source in program.sources() {
        operations.push(OperationView::new(
            OperationLayer::Evidence,
            "read_source",
            source.role().as_str(),
        ));
    }
    operations.push(OperationView::new(
        OperationLayer::Evidence,
        "compose",
        composition_label(program.composition()),
    ));
    if let Some(decision) = program.decision() {
        operations.push(OperationView::new(
            OperationLayer::Evidence,
            "decide",
            format!(
                "{} over {} actions and {} constraints",
                decision.carrier().as_str(),
                decision.actions(),
                decision.constraints()
            ),
        ));
        if let Some(evsi) = decision.information_value() {
            operations.push(OperationView::new(
                OperationLayer::Evidence,
                "value_of_information",
                format!(
                    "{} candidates, {} draws each",
                    evsi.candidates().len(),
                    evsi.rank().draws_per_candidate()
                ),
            ));
        }
    }
    if let Some(quote) =
        program.planning().and_then(super::super::lowered::PlanningProgram::simulation)
    {
        operations.push(OperationView::new(
            OperationLayer::Evidence,
            "simulate",
            format!("{} replicates", quote.replicates()),
        ));
    }
    if let Some(chain) = program.chain() {
        for stage in &chain.plan().stages {
            operations.push(OperationView::new(
                OperationLayer::Evidence,
                "propagate",
                format!(
                    "{}: {} as {}",
                    stage.id,
                    stage.kind.as_str(),
                    stage.representation.as_str()
                ),
            ));
        }
    }
    operations
}

/// The physical operations of a plan: each operator's effect, the buffers it
/// materializes and the backend it runs on.
pub(super) fn physical_operations(physical: &PhysicalView) -> Vec<OperationView> {
    let mut operations = Vec::new();
    for operator in physical.operators() {
        operations.push(OperationView::new(
            OperationLayer::Physical,
            operator.effect().as_str(),
            operator.capability().to_string(),
        ));
    }
    for (buffer, materialization) in physical.materializations() {
        operations.push(OperationView::new(
            OperationLayer::Physical,
            "materialize",
            format!("{buffer}: {materialization:?}"),
        ));
    }
    if let Some(backend) = physical.backend() {
        operations.push(OperationView::new(
            OperationLayer::Physical,
            "run_on_backend",
            backend.selected().as_str(),
        ));
    }
    operations
}
