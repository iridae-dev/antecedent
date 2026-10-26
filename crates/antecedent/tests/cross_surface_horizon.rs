//! Two-step horizon through the scientific target and an independent consumer.
//!
//! The licensed horizon is two. The target law is the observational marginal of
//! the history-indexed experiments. Unit histories and source experiments do not
//! supply it. The numerical check is the fixture's enumerated structural risk.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "../../antecedent-estimate/tests/common/mod.rs"]
mod common;

use antecedent::{
    EvidenceSearch, PreparedTemporalTransport, SequenceEndpoints, StudyBuilder,
    consume_temporal_transport_artifact,
};
use antecedent_core::scientific_target::HistoryIndexedTargetLaw;
use antecedent_core::{
    AssumptionSet, Conditioning, ExecutionContext, GraphFamily, HistoryWindow, InferenceSelection,
    InitialLaw, Intervention, InterventionWorld, ModelAssumptions, ObservationUnit,
    OutcomeQuantity, QuantityUnits, ScientificTarget, ScientificWorldId, SearchLimits, TargetForm,
    TargetPopulation, TemporalPolicy, TimeFrame, Value, VariableId, WorldAction, WorldBinding,
};
use antecedent_estimate::temporal_dependent_interval::{
    SequenceHistory, TemporalUnitPanel, UnitHistories,
};
use antecedent_estimate::temporal_history_fit::fit_binary_history_joint;
use antecedent_estimate::temporal_transport::TEMPORAL_INFERENCE_CLAIM;
use antecedent_expr::{ExactEvaluationLimits, ExactTransportData};
use antecedent_identify::sid::temporal_sequence::TemporalSequenceSpec;
use antecedent_io::IoError;
use antecedent_io::temporal_transport_artifact::TemporalTransportConsumeLimits;
use common::temporal_fixture::{laws, sequence, source_scm, spec, target_scm, truth};

const BUDGET: SearchLimits = SearchLimits { operations: 100_000, depth: 256 };

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(3)
}

fn experiment_coordinates(spec: &TemporalSequenceSpec) -> Vec<VariableId> {
    let slots = spec.slots();
    let mut coordinates = slots.variables();
    coordinates.retain(|id| *id != slots.outcome);
    coordinates
}

fn endpoints(sequence: &[Value]) -> SequenceEndpoints<'_> {
    SequenceEndpoints { sequence, source: "source", target: "target" }
}

fn search() -> EvidenceSearch {
    EvidenceSearch { catalog: common::temporal_fixture::catalog(), budget: BUDGET }
}

fn scientific_target(horizon: u32, law: Option<HistoryIndexedTargetLaw>) -> ScientificTarget {
    let sequence = spec();
    let slots = sequence.slots();
    let time = TimeFrame::temporal(
        horizon,
        HistoryWindow::max_lag(horizon),
        InitialLaw::contemporaneous_observed(),
    )
    .unwrap();
    let binding = WorldBinding::new(
        TargetPopulation::AllObserved,
        ObservationUnit::new("series").unwrap(),
        OutcomeQuantity::mean(slots.outcome, QuantityUnits::dimensionless()).unwrap(),
        Conditioning::unconditional(),
    )
    .unwrap();
    let action = WorldAction::schedule(
        Intervention::set(slots.actions[0], Value::f64(1.0)),
        TemporalPolicy::pulse(0),
    )
    .unwrap();
    let world =
        InterventionWorld::new(ScientificWorldId::from_raw(1), action, false, binding, time)
            .unwrap();
    let target = ScientificTarget::new(
        TargetForm::World(world),
        ModelAssumptions::new(GraphFamily::TemporalAdmg, AssumptionSet::new()),
        InferenceSelection::Unspecified,
    );
    match law {
        Some(law) => target.with_history_indexed_law(law),
        None => target,
    }
}

fn transport_data() -> ExactTransportData {
    laws(&source_scm(), &target_scm(), &[])
}

fn declared_law(data: &ExactTransportData) -> HistoryIndexedTargetLaw {
    let coordinates = experiment_coordinates(&spec());
    StudyBuilder::history_indexed_target_law(data, "target", &coordinates).unwrap()
}

fn prepare(
    data: ExactTransportData,
    target: &ScientificTarget,
    sequence: &[Value],
) -> Result<PreparedTemporalTransport, IoError> {
    StudyBuilder::temporal_transport_sequence_for_target(
        &spec(),
        endpoints(sequence),
        search(),
        data,
        target,
        ExactEvaluationLimits::default(),
        &ctx(),
    )
}

fn refusal(error: &IoError) -> String {
    error.to_string()
}

#[test]
fn history_indexed_law_licenses_the_two_step_horizon_and_reloads() {
    let spec = spec();
    assert_eq!(spec.horizon(), 2);
    let coordinates = experiment_coordinates(&spec);
    assert_eq!(coordinates.len(), 5);
    let data = transport_data();
    let law = declared_law(&data);
    assert_eq!(law.snapshot(), "target-0");
    assert_eq!(law.coordinates(), coordinates.as_slice());

    let bare = scientific_target(2, None);
    assert!(bare.history_indexed_law().is_none());
    assert_eq!(bare.shared_temporal_horizon(), Some(2));
    let missing = prepare(data.clone(), &bare, &sequence(0.0, 1.0)).unwrap_err();
    let missing = refusal(&missing);
    assert!(missing.contains("temporal_transport.target_law_missing"), "{missing}");
    assert!(missing.contains("unit histories do not supply the target law"), "{missing}");

    let source_laws = data
        .laws()
        .iter()
        .filter(|law| !law.interventions().is_empty())
        .cloned()
        .collect::<Vec<_>>();
    let source_only = ExactTransportData::try_new(source_laws, 4096).unwrap();
    let with_law = scientific_target(2, Some(law.clone()));
    assert_ne!(bare.identity(), with_law.identity());
    let from_experiments = prepare(source_only, &with_law, &sequence(0.0, 1.0)).unwrap_err();
    let from_experiments = refusal(&from_experiments);
    assert!(
        from_experiments.contains("temporal_transport.target_law_missing"),
        "{from_experiments}"
    );
    assert!(
        from_experiments.contains("unit histories do not supply the target law"),
        "{from_experiments}"
    );

    let panel = TemporalUnitPanel::new(
        "unit-histories",
        Some(vec![UnitHistories {
            unit_id: 1,
            histories: vec![SequenceHistory::two_step(1, 0, 1, 0, 1, 1.0)],
        }]),
    )
    .unwrap();
    let histories =
        ExactTransportData::try_new(vec![fit_binary_history_joint(&panel).unwrap()], 4096).unwrap();
    let from_histories = prepare(histories, &with_law, &sequence(0.0, 1.0)).unwrap_err();
    let from_histories = refusal(&from_histories);
    assert!(from_histories.contains("temporal_transport.target_law_missing"), "{from_histories}");
    assert!(
        from_histories.contains("unit histories do not supply the target law"),
        "{from_histories}"
    );

    let wrong = HistoryIndexedTargetLaw::new(
        "not-the-observational-marginal",
        coordinates.iter().copied(),
        [(vec![0; coordinates.len()], 1.0)],
    )
    .unwrap();
    let mismatched = scientific_target(2, Some(wrong));
    let mismatch = prepare(data.clone(), &mismatched, &sequence(0.0, 1.0)).unwrap_err();
    let mismatch = refusal(&mismatch);
    assert!(mismatch.contains("temporal_transport.target_law_mismatch"), "{mismatch}");

    let short = scientific_target(1, Some(law));
    assert_eq!(short.shared_temporal_horizon(), Some(1));
    let horizon = prepare(data.clone(), &short, &sequence(0.0, 1.0)).unwrap_err();
    let horizon = refusal(&horizon);
    assert!(horizon.contains("temporal_transport.horizon"), "{horizon}");

    let licensed = scientific_target(2, Some(declared_law(&data)));
    assert_eq!(licensed.shared_temporal_horizon(), Some(2));
    let sequence = sequence(0.0, 1.0);
    let prepared = prepare(data, &licensed, &sequence).unwrap();
    let report = prepared.estimate(&ctx()).unwrap();
    assert_eq!(report.horizon, 2);
    assert_eq!(report.inference_claim, TEMPORAL_INFERENCE_CLAIM);
    assert!(report.support.target_law_used);
    assert!((report.mean - truth(&target_scm(), [0, 1])).abs() < 1e-12);
    let bytes = prepared.export(&report).unwrap();
    let mean = report.mean.to_bits();
    let probabilities = report.distribution.probabilities.clone();
    drop(prepared);
    drop(report);
    let consumed = consume_temporal_transport_artifact(
        &bytes,
        TemporalTransportConsumeLimits::default(),
        &ctx(),
    )
    .unwrap();
    assert_eq!(consumed.mean.to_bits(), mean);
    assert_eq!(consumed.horizon, 2);
    assert!(consumed.support.target_law_used);
    assert_eq!(consumed.inference_claim, TEMPORAL_INFERENCE_CLAIM);
    for (left, right) in consumed.distribution.probabilities.iter().zip(probabilities.iter()) {
        assert_eq!(left.to_bits(), right.to_bits());
    }
}
