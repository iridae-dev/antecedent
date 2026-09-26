//! Contracts-first: identities, inspection, transformation preview, claims.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::too_many_lines)]
#![allow(
    clippy::float_cmp,
    reason = "test scaffolding compares exact constants and indexes with small literals"
)]

use std::sync::Arc;

use antecedent::discovery::{
    DiscoverParams, MultiDatasetConstraints, StaticDiscoverParams, discover_ges, discover_pc,
    discover_pcmci_plus,
};
use antecedent::discovery_defaults::resolve_ci;
use antecedent::state::{
    DataBatchRef, InterventionRecord, apply_state_event, new_antecedent_state,
    publish_recomputed_results, result_lineage_fingerprint,
};
use antecedent::{
    AcceptedGraph, BayesianConfig, EstimatorId, IdentifierId, InferenceMode, IntoGraphInput,
    OperationKind, OperationReadiness, RefuteSuite, SemanticApplicability, StructureSource, Study,
    StudyResult,
};
use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSource, AssumptionStatus,
    AverageEffectQuery, CacheBudget, CausalQuery, CausalRng, CausalSchemaBuilder, ClaimDomainAxis,
    ClaimKind, ClaimOperation, ConditionalEffectQuery, ConsumerProfile, ContinuousDomain,
    CounterfactualQuery, DerivedClaimOutcome, DomainStatus, ExecutionContext, ExecutionReceipt,
    ExecutionRequestState, GridSpec, HostOperation, IdentificationStatus, Intervention,
    InterventionSequence, InterventionalDistributionQuery, Lag, MeasurementSpec, MediationContrast,
    MediationQuery, ObligationKind, ObservationAssumption, ObservationSpec,
    PathSpecificEffectQuery, ProgressSink, QueryId, RequestIdentity, ResponseFunctional,
    ResponseIdentification, ResponseQuery, ResponseValue, RoleHint, SemanticDigest,
    SequencedIntervention, SharedEvidenceRef, SlotAvailability, SmallRoleSet,
    TEMPORAL_OBSERVATION_UNLICENSED, TemporalEffectQuery, TemporalPolicy, TemporalResponseSpec,
    TransformIntent, Value, ValueType, VariableId, compose_claims,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TableView, TabularData,
    TimeIndex, TimeSeriesData, ValidityBitmap,
};
use antecedent_discovery::{GraphPosterior, set_edge};
use antecedent_graph::{
    Admg, Cpdag, CpdagReview, Dag, DenseNodeId, Endpoint, MarkedEdge, MiddleMark, Pag,
    TemporalCpdag, TemporalCpdagReview, TemporalDag, TemporalPag, ensure_lagged,
};
use antecedent_io::query_wire::{
    CausalQueryWire, InterventionWire, TargetPopulationWire, ValueWire,
};
use antecedent_io::{
    DesignVariableRole, DesignVariableSummary, EstimandFingerprint, PriorCatalog, PriorMapping,
    PriorSourceMeta, PriorSourceRef, TargetDesign, TemporalCoordinates, accept_claim,
    causal_query_to_wire, consume_analysis_result, executed_functional_labels, project_claim_host,
    project_lossy_scalar,
};
use antecedent_prob::InferenceDiagnostics;

mod common;

// The confounded static ATE study five suites run, in one owner.
use common::fixtures::{
    confounded_scm, mixture_graph_posterior, pinned_pag as identified_pag,
    pinned_pag_series as identified_pag_series,
};

fn study(data: TabularData, dag: Dag, query: AverageEffectQuery) -> Study {
    Study::tabular(data).graph(dag).query(query).bootstrap_replicates(0).build().unwrap()
}

#[derive(Default)]
struct RecordingProgress(std::sync::Mutex<Vec<String>>);

impl ProgressSink for RecordingProgress {
    fn report(&self, _fraction: f64, stage: &str) {
        self.0.lock().unwrap().push(stage.to_owned());
    }
}

fn identify_computations(sink: &RecordingProgress) -> usize {
    sink.0.lock().unwrap().iter().filter(|stage| stage.as_str() == "identify.compute").count()
}

fn assert_executed_binary_ate(query: &CausalQueryWire, names: &[String]) {
    let CausalQueryWire::AverageEffect {
        treatment,
        outcome,
        control,
        active,
        target_population,
        outcome_functional,
        ..
    } = query
    else {
        panic!("executed functional must be AverageEffect, got {query:?}");
    };
    assert_eq!(names[*treatment as usize], "t");
    assert_eq!(names[*outcome as usize], "y");
    assert_eq!(*treatment, 0);
    assert_eq!(*outcome, 1);
    match control {
        InterventionWire::Set { variable, value: ValueWire::Float64(value) } => {
            assert_eq!(*variable, *treatment);
            assert_eq!(value.to_bits(), 0.0f64.to_bits());
        }
        other => panic!("control must be do(t=0), got {other:?}"),
    }
    match active {
        InterventionWire::Set { variable, value: ValueWire::Float64(value) } => {
            assert_eq!(*variable, *treatment);
            assert_eq!(value.to_bits(), 1.0f64.to_bits());
        }
        other => panic!("active must be do(t=1), got {other:?}"),
    }
    assert!(
        matches!(target_population, TargetPopulationWire::AllObserved),
        "population must stay AllObserved, got {target_population:?}"
    );
    assert!(
        matches!(outcome_functional, antecedent_io::query_wire::OutcomeFunctionalWire::Mean),
        "temporal coordinates are absent on static AverageEffect"
    );
}

#[test]
fn inspect_does_not_identify() {
    let (data, dag, query) = confounded_scm(64, 7);
    let built = study(data, dag, query);
    let inspected = built.inspect().unwrap();
    assert!(matches!(inspected.reasoning.identification, SlotAvailability::Unavailable { .. }));
    assert!(inspected.identities.identification_product.is_none());
    assert!(matches!(inspected.reasoning.uncertainty, SlotAvailability::Unavailable { .. }));
    assert_eq!(inspected.support_status.unwrap().as_str(), "licensed");
}

#[test]
fn prepared_contract_uses_cached_identification() {
    let ctx = ExecutionContext::for_tests(1);
    let (data, dag, query) = confounded_scm(64, 11);
    let prepared = study(data, dag, query).prepare(&ctx).unwrap();
    let contract = prepared.contract().unwrap();
    let product = contract.identities.identification_product.expect("prepared product");
    assert_ne!(Some(contract.identities.target), contract.identities.program);
    assert_ne!(contract.identities.identification, product);
    match &contract.reasoning.identification {
        SlotAvailability::Available(slot) => {
            assert_eq!(slot.status, IdentificationStatus::NonparametricallyIdentified);
            assert_eq!(slot.identified_mass, 1.0);
            assert_eq!(slot.unidentified_mass, 0.0);
            assert!(!slot.search_capped);
        }
        _ => panic!("prepared identification must be available"),
    }
}

#[test]
fn prepared_inspect_is_the_cheap_pre_prepare_record() {
    let ctx = ExecutionContext::for_tests(1);
    let (data, dag, query) = confounded_scm(64, 12);
    let before = study(data.clone(), dag.clone(), query.clone()).inspect().unwrap();
    let prepared = study(data, dag, query).prepare(&ctx).unwrap();
    let cheap = prepared.inspect().unwrap();
    assert!(cheap.identities.identification_product.is_none());
    assert!(matches!(cheap.reasoning.identification, SlotAvailability::Unavailable { .. }));
    assert_eq!(cheap.capability().operation, OperationKind::Inspect);
    assert_eq!(cheap.identities, before.identities, "inspect must not read prepared caches");
    let contract = prepared.contract().unwrap();
    assert!(contract.identities.identification_product.is_some());
    assert!(matches!(contract.reasoning.identification, SlotAvailability::Available(_)));
    assert_eq!(contract.identities.target, cheap.identities.target);
    assert!(contract.identities.program.is_some());
    assert_eq!(
        cheap.identities.program, None,
        "cheap inspection has no identification products, so no program identity"
    );
}

#[test]
fn different_graph_is_a_different_program() {
    let ctx = ExecutionContext::for_tests(1);
    let (data, dag, query) = confounded_scm(48, 13);
    let mut flipped = Dag::with_variables(3);
    flipped.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    flipped.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    flipped.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let a = study(data.clone(), dag, query.clone()).prepare(&ctx).unwrap();
    let b = study(data, flipped, query).prepare(&ctx).unwrap();
    let ca = a.contract().unwrap();
    let cb = b.contract().unwrap();
    assert_eq!(ca.identities.target, cb.identities.target);
    assert_ne!(ca.identities.identification, cb.identities.identification);
    assert_ne!(ca.identities.program, cb.identities.program);
}

#[test]
fn prior_change_keeps_identification_and_changes_binding() {
    let ctx = ExecutionContext::for_tests(1);
    let (data, dag, query) = confounded_scm(48, 17);
    let frequentist = study(data.clone(), dag.clone(), query.clone()).prepare(&ctx).unwrap();
    let bayesian = Study::tabular(data)
        .graph(dag)
        .query(query)
        .bootstrap_replicates(0)
        .inference(antecedent::InferenceMode::Bayesian(antecedent::BayesianConfig::laplace()))
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let cf = frequentist.contract().unwrap();
    let cb = bayesian.contract().unwrap();
    assert_eq!(cf.identities.target, cb.identities.target);
    assert_eq!(cf.identities.identification, cb.identities.identification);
    assert_ne!(cf.identities.inference_binding, cb.identities.inference_binding);
    assert_ne!(cf.identities.program, cb.identities.program);
}

#[test]
fn preview_does_not_authorize_a_changed_program() {
    let ctx = ExecutionContext::for_tests(1);
    let (data, dag, query) = confounded_scm(32, 19);
    let prepared = study(data, dag, query).prepare(&ctx).unwrap();
    let contract = prepared.contract().unwrap();
    let preview = prepared.preview_transform(TransformIntent::CompatibleDataReplace).unwrap();
    assert!(preview.binds_program(contract.identities.program.expect("prepared program")));
    assert!(!preview.refused);
    let support = preview.layer(antecedent_core::SemanticLayer::Support).unwrap();
    assert!(support.effects.contains(&antecedent_core::TransformEffect::Invalidates));
    assert!(support.effects.contains(&antecedent_core::TransformEffect::RequiresReestimation));
    let identification = preview.layer(antecedent_core::SemanticLayer::Identification).unwrap();
    assert!(identification.effects.contains(&antecedent_core::TransformEffect::Preserves));
}

#[test]
fn estimate_claim_preserves_identities() {
    let ctx = ExecutionContext::for_tests(1);
    let (data, dag, query) = confounded_scm(80, 23);
    let prepared = study(data.clone(), dag, query).prepare(&ctx).unwrap();
    let contract = prepared.contract().unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let claim = result.claim(&contract, &ctx).unwrap();
    assert_eq!(claim.identities.program, contract.identities.program);
    assert_eq!(claim.identities.target, contract.identities.target);
    assert!(claim.value.is_some_and(f64::is_finite));
    assert!(matches!(claim.reasoning.identification, SlotAvailability::Available(_)));
}

#[test]
fn inspect_and_prepare_share_target_identity() {
    let ctx = ExecutionContext::for_tests(1);
    let (data, dag, query) = confounded_scm(32, 29);
    let built = study(data, dag, query);
    let inspected = built.inspect().unwrap();
    let prepared = built.prepare(&ctx).unwrap();
    let contract = prepared.contract().unwrap();
    assert_eq!(inspected.identities.target, contract.identities.target);
    assert_eq!(inspected.identities.identification, contract.identities.identification);
    assert!(inspected.identities.identification_product.is_none());
    assert!(contract.identities.identification_product.is_some());
    assert!(inspected.identities.program.is_none());
    assert!(contract.identities.program.is_some());
}

#[test]
fn acceptance_history_preserves_semantics_and_remains_inspectable() {
    let ctx = ExecutionContext::for_tests(1);
    let (data, dag, query) = confounded_scm(256, 31);
    let accepted = antecedent::AcceptedGraph::dag(dag.clone());
    let reviewed = accepted.replace(accepted.clone().with_schema(data.schema()));
    let explicit = study(data.clone(), dag, query.clone()).prepare(&ctx).unwrap();
    let first = Study::tabular(data.clone())
        .graph(accepted)
        .query(query.clone())
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let second = Study::tabular(data.clone())
        .graph(reviewed)
        .query(query)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let explicit_contract = explicit.contract().unwrap();
    let a = first.contract().unwrap();
    let b = second.contract().unwrap();
    assert_eq!(a.accepted_version, 1);
    assert_eq!(b.accepted_version, 2);
    assert!(a.accepted_variable_names.is_none());
    assert_eq!(
        b.accepted_variable_names.as_deref().unwrap(),
        &[Arc::from("t"), Arc::from("y"), Arc::from("z")]
    );
    assert_eq!(explicit_contract.structure_source, antecedent::StructureSource::Explicit);
    assert_eq!(a.structure_source, antecedent::StructureSource::Accepted);
    assert_eq!(explicit_contract.identities, a.identities);
    assert_eq!(a.identities, b.identities);
    let original = first.estimate(&data, &ctx).unwrap();
    let reaccepted = second.estimate(&data, &ctx).unwrap();
    assert_eq!(original.effect(), reaccepted.effect());
    assert!((original.effect() - 2.0).abs() < 0.2);
}

#[test]
fn refresh_same_shape_changes_snapshot_and_expires_preview() {
    let ctx = ExecutionContext::for_tests(1);
    let (data, dag, query) = confounded_scm(256, 31);
    let mut prepared = study(data.clone(), dag.clone(), query.clone()).prepare(&ctx).unwrap();
    let before = prepared.contract().unwrap();
    let preview = before.preview_transform(TransformIntent::CompatibleDataReplace);
    assert!(preview.binds_contract(&before.identities));
    let original = prepared.estimate(&data, &ctx).unwrap();
    let treatment = data.float64_slice(VariableId::from_raw(0)).unwrap();
    let outcome = data.float64_slice(VariableId::from_raw(1)).unwrap();
    let changed = data
        .with_replaced_float(
            VariableId::from_raw(1),
            outcome.iter().zip(treatment).map(|(y, t)| y + t).collect::<Vec<_>>().into(),
        )
        .unwrap();
    let refreshed = prepared.refresh(changed.clone(), &ctx).unwrap();
    let after = prepared.contract().unwrap();
    assert_eq!(before.identities.target, after.identities.target);
    assert_eq!(before.identities.identification, after.identities.identification);
    assert_eq!(before.identities.identification_product, after.identities.identification_product);
    assert_eq!(before.identities.program, after.identities.program);
    assert_eq!(before.identities.inference_binding, after.identities.inference_binding);
    assert_eq!(before.identities.observation, after.identities.observation);
    assert_ne!(before.identities.data_snapshot, after.identities.data_snapshot);
    assert!(!preview.binds_contract(&after.identities));
    let fresh = study(changed.clone(), dag, query).prepare(&ctx).unwrap();
    let fresh_result = fresh.estimate(&changed, &ctx).unwrap();
    assert!((refreshed.effect() - fresh_result.effect()).abs() < 1e-12);
    assert!((refreshed.estimate.se_analytic - fresh_result.estimate.se_analytic).abs() < 1e-12);
    assert_eq!(after.identities, fresh.contract().unwrap().identities);
    assert!((refreshed.effect() - original.effect() - 1.0).abs() < 1e-10);
    assert!((refreshed.effect() - 3.0).abs() < 0.2);
    let failure = data.with_replaced_float(VariableId::from_raw(0), vec![0.0; 256].into()).unwrap();
    assert!(prepared.refresh(failure, &ctx).is_err());
    assert_eq!(after.identities, prepared.contract().unwrap().identities);
}

#[test]
fn same_shape_inspection_distinguishes_masks_and_weights() {
    let (data, dag, query) = confounded_scm(32, 37);
    let baseline = study(data.clone(), dag.clone(), query.clone()).inspect().unwrap();
    let masked = data
        .with_analysis_mask(
            ValidityBitmap::from_bytes([0xfe, 0xff, 0xff, 0xff].as_slice(), 32).unwrap(),
        )
        .unwrap();
    let weighted = TabularData::new(
        OwnedColumnarStorage::try_new(
            data.schema().clone(),
            data.storage().columns().to_vec(),
            None,
            Some(vec![2.0; 32].into()),
        )
        .unwrap(),
    );
    let ctx = ExecutionContext::for_tests(1);
    let program = |data: TabularData| {
        study(data, dag.clone(), query.clone())
            .prepare(&ctx)
            .unwrap()
            .contract()
            .unwrap()
            .identities
    };
    let baseline_program = program(data.clone()).program;
    assert!(baseline_program.is_some());
    for (label, changed) in [("mask", masked), ("weights", weighted)] {
        let contract = study(changed.clone(), dag.clone(), query.clone()).inspect().unwrap();
        assert_eq!(baseline.identities.identification, contract.identities.identification);
        assert_ne!(baseline.identities.data_snapshot, contract.identities.data_snapshot);
        let changed_program = program(changed).program;
        if label == "mask" {
            // The checked physical design is bound to the complete-case row
            // count, so masking changes this executable program identity while
            // leaving the causal target and identification stable.
            assert_ne!(baseline_program, changed_program, "changed={label}");
        } else {
            assert_eq!(baseline_program, changed_program, "changed={label}");
        }
    }
}

#[test]
fn dag_average_effect_vertical_path_is_independently_accepted() {
    let ctx = ExecutionContext::for_tests(1);
    let (data, dag, query) = confounded_scm(80, 41);
    let built = study(data.clone(), dag, query);
    let inspected = built.inspect().unwrap();
    assert!(matches!(inspected.reasoning.identification, SlotAvailability::Unavailable { .. }));
    let prepared = built.prepare(&ctx).unwrap();
    let contract = prepared.contract().unwrap();
    assert_eq!(inspected.identities.target, contract.identities.target);
    let filter = prepared.preview_transform(TransformIntent::NewConditionalQuery).unwrap();
    assert!(
        filter.obligations.iter().any(|item| item.id.as_ref() == "transform.population_or_query")
    );
    let graph_change = prepared.preview_transform(TransformIntent::ChangeGraph).unwrap();
    let identification =
        graph_change.layer(antecedent_core::SemanticLayer::Identification).unwrap();
    assert!(
        identification
            .effects
            .contains(&antecedent_core::TransformEffect::RequiresReidentification)
    );
    let result = prepared.estimate(&data, &ctx).unwrap();
    let claim = result.claim(&contract, &ctx).unwrap();
    assert!(claim.value.is_some_and(|value| (value - 2.0).abs() < 0.2));
    let bytes = prepared.encode_contracted_result(&result, "dag-ate", &ctx).unwrap();
    let consumed = antecedent_io::consume_analysis_result(&bytes).unwrap();
    assert!(!consumed.acceptance.accepts_as_verified_program());
    assert_eq!(
        consumed.acceptance.unresolved.as_ref(),
        &[std::sync::Arc::<str>::from("dependencies.linear_fit_sufficient_statistics")]
    );
    assert_eq!(
        consumed.contract.as_ref().map(|section| section.identities.program),
        Some(*contract.identities.program.expect("prepared program").as_bytes())
    );
    assert_eq!(consumed.body.estimate, Some(result.effect()));
    let names: Vec<String> =
        data.schema().variables().iter().map(|variable| variable.name.to_string()).collect();
    let section = consumed.contract.as_ref().expect("verified contract");
    assert_executed_binary_ate(&section.target.query, &names);
    assert_eq!(section.target.query, consumed.body.query);
    assert_eq!(section.target.query, consumed.body.identification.query);
    assert_eq!(section.target.schema.variable_names(), names);
    assert!(result.estimate.se_analytic.is_finite());
    assert!((result.effect() - 2.0).abs() < 1.96 * result.estimate.se_analytic);
    let body = result.analysis_result_wire(prepared.query()).unwrap();
    let mut legacy = Vec::new();
    antecedent_io::encode_analysis_result_artifact(&body, names, "legacy")
        .unwrap()
        .write_to(&mut legacy)
        .unwrap();
    let old = antecedent_io::consume_analysis_result(&legacy).unwrap();
    assert_eq!(old.body.estimate, consumed.body.estimate);
    assert!(!old.acceptance.accepts_as_verified_program());
}

#[test]
fn dag_average_effect_vertical_path_binds_functional_after_retarget() {
    let ctx = ExecutionContext::for_tests(1);
    let (data, dag, query) = confounded_scm(256, 41);
    let prepared = Study::tabular(data.clone())
        .graph(dag)
        .query(query)
        .estimator(EstimatorId::Aipw)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let contract = prepared.contract().unwrap();
    let estimated = prepared.estimate(&data, &ctx).unwrap();
    let preview = prepared.preview_transform(TransformIntent::Retarget).unwrap();
    let weights = vec![1.0; data.row_count()];
    let retargeted = prepared.apply_retarget(&preview, &weights, &[], &ctx).unwrap();
    assert_eq!(retargeted.certificate.as_ref().map(|c| &c.query), Some(prepared.query()));
    assert_eq!(
        contract.identities.identification_product,
        prepared.contract().unwrap().identities.identification_product
    );
    assert!(retargeted.diagnostics.iter().any(|d| d.code.as_ref() == "exec.identify.cached"));
    assert!(retargeted.estimate.se_analytic.is_finite());
    assert!((retargeted.effect() - 2.0).abs() < 0.35);
    assert!((retargeted.effect() - estimated.effect()).abs() < 1e-10);
    let bytes = prepared.encode_contracted_result(&retargeted, "dag-ate-retarget", &ctx).unwrap();
    let consumed = antecedent_io::consume_analysis_result(&bytes).unwrap();
    assert!(consumed.acceptance.accepts_as_verified_program());
    let names: Vec<String> =
        data.schema().variables().iter().map(|variable| variable.name.to_string()).collect();
    let section = consumed.contract.as_ref().expect("verified contract");
    assert_executed_binary_ate(&section.target.query, &names);
    assert_eq!(section.target.query, consumed.body.query);
    assert_eq!(section.target.query, consumed.body.identification.query);
    assert_eq!(
        section.identities.program,
        *contract.identities.program.expect("prepared program").as_bytes()
    );
}

#[test]
fn dag_average_effect_vertical_path_counts_identify_compute() {
    let sink = Arc::new(RecordingProgress::default());
    let mut ctx = ExecutionContext::for_tests(1);
    ctx.progress = Some(Arc::clone(&sink) as Arc<dyn ProgressSink>);
    let (data, dag, query) = confounded_scm(128, 41);
    let built = study(data.clone(), dag, query);
    let _ = built.inspect().unwrap();
    assert_eq!(identify_computations(&sink), 0, "inspect must not identify");
    let mut prepared = built.prepare(&ctx).unwrap();
    assert_eq!(identify_computations(&sink), 1, "prepare identifies once");
    let first = prepared.estimate(&data, &ctx).unwrap();
    assert_eq!(identify_computations(&sink), 1);
    let refuted = prepared.refute(&first, &data, RefuteSuite::Cheap, &ctx).unwrap();
    assert_eq!(refuted.effect().to_bits(), first.effect().to_bits());
    assert_eq!(identify_computations(&sink), 1, "refute must not re-identify");
    let compatible = data
        .with_replaced_float(
            VariableId::from_raw(1),
            data.float64_slice(VariableId::from_raw(1)).unwrap().to_vec().into(),
        )
        .unwrap();
    let refreshed = prepared.refresh(compatible.clone(), &ctx).unwrap();
    assert_eq!(identify_computations(&sink), 1, "compatible refresh must not re-identify");
    let second = prepared.estimate(&compatible, &ctx).unwrap();
    assert_eq!(identify_computations(&sink), 1);
    assert!(second.diagnostics.iter().any(|d| d.code.as_ref() == "exec.identify.cached"));
    assert!((refreshed.effect() - first.effect()).abs() < 1e-12);
    assert!((second.effect() - 2.0).abs() < 0.25);
}

#[test]
fn admg_latent_confounding_does_not_upgrade_identification() {
    let ctx = ExecutionContext::for_tests(1);
    let (data, _, query) = confounded_scm(48, 43);
    let mut admg = Admg::with_variables(3);
    admg.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    admg.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let built = Study::tabular(data.clone())
        .graph(admg)
        .query(query)
        .identifier(IdentifierId::GeneralId)
        .estimator(EstimatorId::FunctionalEffect)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    assert_eq!(built.inspect().unwrap().graph_class.as_str(), "Admg");
    match built.prepare(&ctx) {
        Ok(prepared) => {
            let contract = prepared.contract().unwrap();
            match &contract.reasoning.identification {
                SlotAvailability::Available(slot) => {
                    assert_eq!(slot.status, IdentificationStatus::NotIdentified);
                    assert_eq!(slot.identified_mass, 0.0);
                }
                other => panic!("expected available NotIdentified, got {other:?}"),
            }
            if let Ok(result) = prepared.estimate(&data, &ctx) {
                assert_eq!(result.claim(&contract, &ctx).unwrap().kind, ClaimKind::Incomplete);
            }
        }
        Err(err) => {
            let text = err.to_string();
            assert!(text.contains("identif"), "{text}");
        }
    }
}

#[test]
fn admg_frontdoor_contract_records_functional_identification() {
    let ctx = ExecutionContext::for_tests(1);
    let n = 64;
    let mut rng = CausalRng::from_seed(47);
    let mut t = Vec::with_capacity(n);
    let mut m = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let u = if rng.next_f64() < 0.5 { 1.0 } else { 0.0 };
        let ti = if rng.next_f64() < 0.2 + 0.6 * u { 1.0 } else { 0.0 };
        let mi = if rng.next_f64() < 0.25 + 0.5 * ti { 1.0 } else { 0.0 };
        let yi = if rng.next_f64() < 0.1 + 0.5 * mi + 0.3 * u { 1.0 } else { 0.0 };
        t.push(ti);
        m.push(mi);
        y.push(yi);
    }
    let mut builder = CausalSchemaBuilder::new();
    builder
        .add_variable(
            "t",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    builder
        .add_variable(
            "m",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    builder
        .add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    let schema = builder.build().unwrap();
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(t), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(m), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(2), Arc::from(y), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let data = TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
    let mut admg = Admg::with_variables(3);
    admg.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    admg.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    admg.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(2));
    let prepared = Study::tabular(data)
        .graph(admg)
        .query(query)
        .identifier(IdentifierId::GeneralId)
        .estimator(EstimatorId::FunctionalEffect)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let contract = prepared.contract().unwrap();
    match &contract.reasoning.identification {
        SlotAvailability::Available(slot) => {
            assert_eq!(slot.status, IdentificationStatus::NonparametricallyIdentified);
            assert_eq!(slot.identified_mass, 1.0);
        }
        other => panic!("front-door ADMG must identify, got {other:?}"),
    }
    assert!(contract.identities.identification_product.is_some());
}

#[test]
fn consume_rehashes_identification_product_from_stored_payloads() {
    let ctx = ExecutionContext::for_tests(1);
    let (data, dag, query) = confounded_scm(80, 53);
    let prepared = study(data.clone(), dag, query).prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let bytes = prepared.encode_contracted_result(&result, "rehash", &ctx).unwrap();
    let consumed = antecedent_io::consume_analysis_result(&bytes).unwrap();
    assert!(!consumed.acceptance.accepts_as_verified_program());
    assert_eq!(
        consumed.acceptance.unresolved.as_ref(),
        &[std::sync::Arc::<str>::from("dependencies.linear_fit_sufficient_statistics")]
    );
    assert!(consumed.contract.as_ref().unwrap().identification_product.is_some());
    assert_eq!(
        consumed
            .contract
            .as_ref()
            .unwrap()
            .reasoning
            .support
            .value
            .as_ref()
            .and_then(|slot| { slot.matrix_coordinate.as_deref() }),
        Some("AverageEffect:Dag:explicit:Frequentist:full")
    );
}

#[test]
fn execution_identity_follows_the_estimate_context() {
    let (data, dag, query) = confounded_scm(48, 59);
    let prepared =
        study(data.clone(), dag, query).prepare(&ExecutionContext::for_tests(1)).unwrap();
    let contract = prepared.contract().unwrap();
    let result = prepared.estimate(&data, &ExecutionContext::for_tests(1)).unwrap();
    let a = result.claim(&contract, &ExecutionContext::for_tests(1)).unwrap();
    let b = result.claim(&contract, &ExecutionContext::for_tests(2)).unwrap();
    assert_ne!(a.execution, b.execution);
    assert_ne!(a.claim_id, b.claim_id);
}

#[test]
fn execution_identity_is_advertised_and_bound() {
    let ctx = ExecutionContext::for_tests(3);
    let (data, dag, query) = confounded_scm(48, 63);
    let prepared = study(data.clone(), dag, query).prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let bytes = prepared.encode_contracted_result(&result, "execution", &ctx).unwrap();
    let consumed = consume_analysis_result(&bytes).unwrap();
    assert!(!consumed.acceptance.accepts_as_verified_program());
    assert_eq!(
        consumed.acceptance.unresolved.as_ref(),
        &[std::sync::Arc::<str>::from("dependencies.linear_fit_sufficient_statistics")]
    );
    let section = consumed.contract.expect("contract");
    let advertised = section.identities.execution.expect("execution identity");
    assert_eq!(Some(advertised), section.claim.as_ref().unwrap().execution);
    assert_eq!(
        antecedent_io::execution_digest(section.execution.as_ref().expect("execution payload"))
            .unwrap()
            .as_bytes(),
        &advertised
    );
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    let mut tampered = section.clone();
    tampered.identities.execution.as_mut().unwrap()[0] ^= 0xff;
    let consumed = consume_analysis_result(&rewrite_contract_section(
        &body,
        header.variable_names.clone(),
        "tampered",
        &tampered,
    ))
    .unwrap();
    assert!(!consumed.acceptance.accepts_as_verified_program());
    assert!(consumed.acceptance.unresolved.iter().any(|item| &**item == "identities.execution"));
}

#[test]
fn apply_refresh_requires_a_bound_preview() {
    let ctx = ExecutionContext::for_tests(1);
    let (data, dag, query) = confounded_scm(64, 61);
    let mut prepared = study(data.clone(), dag, query).prepare(&ctx).unwrap();
    let preview = prepared.preview_transform(TransformIntent::CompatibleDataReplace).unwrap();
    let stale = prepared.preview_transform(TransformIntent::ChangeGraph).unwrap();
    assert!(prepared.apply_refresh(&stale, data.clone(), &ctx).is_err());
    let refreshed = prepared.apply_refresh(&preview, data, &ctx).unwrap();
    assert!(refreshed.effect().is_finite());
}

#[test]
fn unweighted_class_aggregation_is_refused() {
    let ctx = ExecutionContext::for_tests(1);
    let (data, dag, query) = confounded_scm(48, 67);
    let cpdag = antecedent_graph::Cpdag::from_dag(&dag);
    let prepared = Study::tabular(data)
        .graph(cpdag)
        .query(query)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let preview = prepared.preview_transform(TransformIntent::AverageUnweightedClass).unwrap();
    assert!(preview.refused);
    assert!(
        preview
            .obligations
            .iter()
            .any(|item| item.id.as_ref() == "transform.unweighted_class_prior")
    );
}

#[test]
fn design_ranking_composes_the_existing_ranker() {
    use antecedent::design::{
        CandidateDesign, DesignCost, DesignEvaluationContext, DesignObjective, DesignRankConfig,
        DesignRanker, SamplingPlan,
    };
    use antecedent_prob::{GraphIdentFlag, WeightedGraphSamples};

    let ctx = ExecutionContext::for_tests(1);
    let (data, dag, query) = confounded_scm(32, 71);
    let prepared = study(data, dag, query).prepare(&ctx).unwrap();
    let graphs = WeightedGraphSamples::new(
        vec![0.6, 0.4],
        vec![GraphIdentFlag::Identified, GraphIdentFlag::Unidentified],
        vec![1, 2],
    )
    .unwrap();
    let candidates = vec![CandidateDesign::IncreaseSamplingRate(SamplingPlan {
        additional_samples: 1,
        cost: DesignCost::zero(),
        tag: 1,
    })];
    let ranker = DesignRanker::new().with_config(DesignRankConfig {
        min_batches: 1,
        max_batches: 2,
        batch_size: 2,
        rank_uncertainty_threshold: 0.5,
    });
    let eval = DesignEvaluationContext::<(), ()> {
        graphs: &graphs,
        effect_width: None,
        model_loglik: None,
        decisions: None,
        query_id_unlock: None,
        env_id_unlock: None,
        identified_under_intervention: None,
        graph_features: None,
    };
    let ranking = prepared
        .rank_designs(&ranker, &DesignObjective::ReduceGraphEntropy, &candidates, &eval, &ctx)
        .unwrap();
    assert_eq!(ranking.ranked.len(), 1);
}

#[test]
fn apply_retarget_rejects_a_stale_preview() {
    let ctx = ExecutionContext::for_tests(1);
    let (data, dag, query) = confounded_scm(48, 73);
    let prepared = study(data, dag, query).prepare(&ctx).unwrap();
    let stale = prepared.preview_transform(TransformIntent::ChangeGraph).unwrap();
    let err = prepared.apply_retarget(&stale, &[1.0, 1.0], &[], &ctx).unwrap_err();
    assert!(err.to_string().contains("preview"));
}

fn schema_names(data: &TabularData) -> Vec<String> {
    data.schema().variables().iter().map(|variable| variable.name.to_string()).collect()
}

fn consume_licensed_family(
    built: &Study,
    estimate: impl FnOnce(&antecedent::PreparedStudy) -> StudyResult,
    artifact_id: &str,
    ctx: &ExecutionContext,
) -> (StudyResult, antecedent_io::AnalysisResultConsumption) {
    let inspected = built.inspect().unwrap();
    assert!(matches!(inspected.reasoning.identification, SlotAvailability::Unavailable { .. }));
    assert_eq!(inspected.support_status.unwrap().as_str(), "licensed");
    let prepared = built.prepare(ctx).unwrap();
    let contract = prepared.contract().unwrap();
    assert_eq!(inspected.identities.target, contract.identities.target);
    let preview = prepared.preview_transform(TransformIntent::CompatibleDataReplace).unwrap();
    assert!(!preview.refused);
    assert!(preview.binds_program(contract.identities.program.expect("prepared program")));
    let result = estimate(&prepared);
    let claim = result.claim(&contract, ctx).unwrap();
    assert_eq!(claim.identities.program, contract.identities.program);
    assert_eq!(claim.identities.target, contract.identities.target);
    let bytes = prepared.encode_contracted_result(&result, artifact_id, ctx).unwrap();
    let consumed = consume_analysis_result(&bytes).unwrap();
    let named_nonportable_dependency = |reason: &std::sync::Arc<str>| {
        matches!(
            reason.as_ref(),
            "dependencies.checked_glm_operation"
                | "dependencies.checked_rd_operation"
                | "dependencies.checked_propensity_operation"
                | "dependencies.checked_derivative_response_operation"
                | "dependencies.checked_response_grid_operation"
                | "dependencies.checked_intervention_response_operation"
                | "dependencies.checked_temporal_response_operation"
                | "dependencies.checked_conditional_effect_operation"
                | "dependencies.checked_mediation_operation"
                | "dependencies.checked_bayesian_dag_ate_operation"
                | "dependencies.checked_temporal_dag_effect_operation"
                | "dependencies.checked_temporal_class_effect_operation"
                | "dependencies.checked_temporal_class_response_operation"
                | "dependencies.checked_temporal_graph_posterior_response_operation"
                | "dependencies.fitted_counterfactual_mechanisms"
                // The portable result is readable and preserves the posterior
                // claim, but does not carry joint factor draws for replay.
                | "dependencies.distribution_posterior_factor_draws"
                | "dependencies.functional_effect_posterior_draws"
                // The plan records fitted design roles, but no independent
                // consumer can recompute the numeric fit without replay rows.
                | "dependencies.linear_fit_sufficient_statistics"
        )
    };
    let known_nonportable_dependency =
        consumed.acceptance.unresolved.iter().any(named_nonportable_dependency);
    if known_nonportable_dependency {
        assert!(!consumed.acceptance.accepts_as_verified_program());
        assert!(consumed.acceptance.unresolved.iter().all(named_nonportable_dependency));
    } else {
        assert!(consumed.acceptance.accepts_as_verified_program());
    }
    let section = consumed.contract.as_ref().expect("readable contract");
    let expected = executed_functional_labels(&causal_query_to_wire(prepared.query()).unwrap());
    assert_eq!(executed_functional_labels(&section.target.query), expected);
    assert_eq!(section.target.query, consumed.body.query);
    (result, consumed)
}

fn conditional_effect_fixture() -> (TabularData, Dag, ConditionalEffectQuery) {
    let n = 120usize;
    let t: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
    let w: Vec<f64> = (0..n).map(|i| (i % 5) as f64).collect();
    let y: Vec<f64> = t.iter().zip(&w).map(|(ti, wi)| 1.0 + 2.0 * ti + 0.5 * ti * wi).collect();
    let data = TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("w", w.as_slice()),
    ])
    .unwrap();
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    let query = ConditionalEffectQuery::try_new(
        AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
            .with_effect_modifiers([VariableId::from_raw(2)]),
    )
    .unwrap();
    (data, dag, query)
}

#[test]
fn licensed_family_conditional_effect_frequentist_consumes() {
    let ctx = ExecutionContext::for_tests(1);
    let (data, dag, query) = conditional_effect_fixture();
    let built = Study::tabular(data.clone())
        .graph(dag)
        .query(CausalQuery::ConditionalEffect(query))
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let (result, consumed) = consume_licensed_family(
        &built,
        |prepared| prepared.estimate(&data, &ctx).unwrap(),
        "ce-f",
        &ctx,
    );
    assert!((result.effect() - 3.0).abs() < 1e-8);
    assert_eq!(consumed.body.estimate, Some(result.effect()));
    assert_eq!(
        consumed.acceptance.unresolved.as_ref(),
        &[std::sync::Arc::<str>::from("dependencies.checked_conditional_effect_operation")]
    );
    let labels: std::collections::HashMap<_, _> =
        executed_functional_labels(&consumed.contract.as_ref().unwrap().target.query)
            .into_iter()
            .collect();
    assert_eq!(labels["query_kind"], "conditional_effect");
    assert_eq!(labels["treatment"], "0");
    assert_eq!(labels["outcome"], "1");
    assert_eq!(schema_names(&data), ["t", "y", "w"]);
}

#[test]
fn licensed_family_conditional_effect_bayesian_consumes() {
    let ctx = ExecutionContext::for_tests(1);
    let (data, dag, query) = conditional_effect_fixture();
    let built = Study::tabular(data.clone())
        .graph(dag)
        .query(CausalQuery::ConditionalEffect(query))
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64)))
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let (result, consumed) = consume_licensed_family(
        &built,
        |prepared| prepared.estimate(&data, &ctx).unwrap(),
        "ce-b",
        &ctx,
    );
    assert!((result.effect() - 3.0).abs() < 0.15);
    assert!(result.posterior.is_some());
    assert_eq!(consumed.body.estimate, Some(result.effect()));
}

fn staged_static_kinds() -> (TabularData, Dag, serde_json::Value) {
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/staged_static_kinds/expected.json"
    ))
    .unwrap();
    let a: Vec<_> = (0..500).map(|i| (f64::from(i) * 0.71).sin()).collect();
    let m: Vec<_> =
        a.iter().enumerate().map(|(i, ai)| 2.0 * ai + (i as f64 * 1.13).cos()).collect();
    let y: Vec<_> = a
        .iter()
        .zip(&m)
        .enumerate()
        .map(|(i, (ai, mi))| 3.0 * ai + 4.0 * mi + 0.1 * (i as f64 * 0.31).sin())
        .collect();
    let data = TabularData::from_f64_columns([
        ("a", a.as_slice()),
        ("m", m.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap();
    let mut dag = Dag::with_variables(3);
    for (source, target) in [(0, 1), (0, 2), (1, 2)] {
        dag.insert_directed(DenseNodeId::from_raw(source), DenseNodeId::from_raw(target)).unwrap();
    }
    (data, dag, pin)
}

#[test]
fn licensed_family_mediation_frequentist_consumes() {
    let ctx = ExecutionContext::for_tests(13);
    let (data, dag, pin) = staged_static_kinds();
    let mut query = MediationQuery::binary(
        VariableId::from_raw(0),
        VariableId::from_raw(2),
        Arc::from([VariableId::from_raw(1)]),
        MediationContrast::NaturalDirect,
    );
    query.control =
        Intervention::set(query.treatment, Value::f64(pin["control"].as_f64().unwrap()));
    query.active = Intervention::set(query.treatment, Value::f64(pin["active"].as_f64().unwrap()));
    let built = Study::tabular(data.clone())
        .graph(dag)
        .query(CausalQuery::Mediation(query))
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let (result, consumed) = consume_licensed_family(
        &built,
        |prepared| prepared.estimate(&data, &ctx).unwrap(),
        "med-f",
        &ctx,
    );
    assert!(
        (result.estimate.ate - pin["direct"].as_f64().unwrap()).abs()
            < pin["tolerance"].as_f64().unwrap()
    );
    assert_eq!(consumed.body.estimate, Some(result.estimate.ate));
    assert_eq!(
        consumed.acceptance.unresolved.as_ref(),
        &[std::sync::Arc::<str>::from("dependencies.checked_mediation_operation")]
    );
    let labels: std::collections::HashMap<_, _> =
        executed_functional_labels(&consumed.contract.as_ref().unwrap().target.query)
            .into_iter()
            .collect();
    assert_eq!(labels["query_kind"], "mediation");
    assert_eq!(labels["treatment"], "0");
    assert_eq!(labels["outcome"], "2");
}

#[test]
fn licensed_family_counterfactual_frequentist_consumes() {
    let ctx = ExecutionContext::for_tests(13);
    let (data, dag, pin) = staged_static_kinds();
    let query = CounterfactualQuery::new(
        VariableId::from_raw(2),
        Arc::from([Intervention::set(
            VariableId::from_raw(0),
            Value::f64(pin["active"].as_f64().unwrap()),
        )]),
    )
    .with_control_level(pin["control"].as_f64().unwrap());
    let built = Study::tabular(data.clone())
        .graph(dag)
        .query(CausalQuery::Counterfactual(query))
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let (result, consumed) = consume_licensed_family(
        &built,
        |prepared| prepared.estimate(&data, &ctx).unwrap(),
        "cf-f",
        &ctx,
    );
    let cf = result.counterfactual.as_ref().expect("counterfactual payload");
    assert!((cf.mean_ite - pin["counterfactual_mean"].as_f64().unwrap()).abs() < 0.03);
    assert_eq!(consumed.body.estimate, Some(result.effect()));
    let labels: std::collections::HashMap<_, _> =
        executed_functional_labels(&consumed.contract.as_ref().unwrap().target.query)
            .into_iter()
            .collect();
    assert_eq!(labels["query_kind"], "counterfactual");
    assert_eq!(labels["outcome"], "2");
}

fn path_specific_fixture() -> (TabularData, Dag, PathSpecificEffectQuery) {
    let mut t = Vec::new();
    let mut m = Vec::new();
    let mut y = Vec::new();
    for (ti, mi, yi, count) in [
        (0.0, 0.0, 0.0, 40),
        (0.0, 0.0, 1.0, 10),
        (0.0, 1.0, 0.0, 10),
        (0.0, 1.0, 1.0, 40),
        (1.0, 0.0, 0.0, 10),
        (1.0, 0.0, 1.0, 10),
        (1.0, 1.0, 0.0, 10),
        (1.0, 1.0, 1.0, 70),
    ] {
        for _ in 0..count {
            t.push(ti);
            m.push(mi);
            y.push(yi);
        }
    }
    let data = TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("m", m.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap();
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    let query = PathSpecificEffectQuery::binary(VariableId::from_raw(0), VariableId::from_raw(2))
        .with_path_nodes([VariableId::from_raw(1)]);
    (data, dag, query)
}

#[test]
fn licensed_family_path_specific_frequentist_consumes() {
    let ctx = ExecutionContext::for_tests(1);
    let (data, dag, query) = path_specific_fixture();
    let built = Study::tabular(data.clone())
        .graph(dag)
        .query(CausalQuery::PathSpecific(query))
        .identifier(IdentifierId::PathSpecificNatural)
        .estimator(EstimatorId::FunctionalEffect)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let (result, consumed) = consume_licensed_family(
        &built,
        |prepared| prepared.estimate(&data, &ctx).unwrap(),
        "path-f",
        &ctx,
    );
    assert!(result.effect().is_finite());
    assert_eq!(consumed.body.estimate, Some(result.effect()));
    let labels: std::collections::HashMap<_, _> =
        executed_functional_labels(&consumed.contract.as_ref().unwrap().target.query)
            .into_iter()
            .collect();
    assert_eq!(labels["query_kind"], "path_specific");
    assert_eq!(labels["treatment"], "0");
    assert_eq!(labels["outcome"], "2");
}

fn distribution_fixture() -> (TabularData, Dag, InterventionalDistributionQuery) {
    let mut t = Vec::new();
    let mut y = Vec::new();
    let mut z = Vec::new();
    for (zv, tv, yv, count) in [
        (0.0, 0.0, 0.0, 21),
        (0.0, 0.0, 1.0, 9),
        (0.0, 1.0, 0.0, 4),
        (0.0, 1.0, 1.0, 16),
        (1.0, 0.0, 0.0, 12),
        (1.0, 0.0, 1.0, 3),
        (1.0, 1.0, 0.0, 14),
        (1.0, 1.0, 1.0, 21),
    ] {
        t.extend(std::iter::repeat_n(tv, count));
        y.extend(std::iter::repeat_n(yv, count));
        z.extend(std::iter::repeat_n(zv, count));
    }
    let data = TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
    ])
    .unwrap();
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let query = InterventionalDistributionQuery::new(
        VariableId::from_raw(1),
        [Intervention::set(VariableId::from_raw(0), Value::f64(1.0))],
    );
    (data, dag, query)
}

#[test]
fn licensed_family_distribution_frequentist_consumes() {
    let ctx = ExecutionContext::for_tests(1);
    let (data, dag, query) = distribution_fixture();
    let built = Study::tabular(data.clone())
        .graph(dag)
        .query(CausalQuery::Distribution(query))
        .identifier(IdentifierId::GeneralId)
        .estimator(EstimatorId::FunctionalDistribution)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let (result, consumed) = consume_licensed_family(
        &built,
        |prepared| prepared.estimate(&data, &ctx).unwrap(),
        "dist-f",
        &ctx,
    );
    let dist = result.distribution.as_ref().expect("distribution payload");
    assert!((dist.mean - 0.7).abs() < 0.08);
    assert_eq!(consumed.body.estimate, Some(result.effect()));
    let contract = consumed.contract.as_ref().unwrap();
    let retained = contract.program.as_ref().unwrap().functional_program.as_ref().unwrap();
    assert_eq!(retained.source, retained.executable);
    assert!(
        antecedent_io::functional_program_from_wire(
            retained,
            antecedent_expr::ProgramLimits::default(),
        )
        .is_ok()
    );

    // Recompute both the payload digest and section seal after tampering so
    // independent consumption must validate the retained program structure.
    let mut forged = contract.clone();
    let retained = forged.program.as_mut().unwrap().functional_program.as_mut().unwrap();
    retained.arena.nodes.clear();
    forged.identities.program =
        *antecedent_io::program_digest(forged.program.as_ref().unwrap()).unwrap().as_bytes();
    forged.seal = antecedent_io::contract_seal(
        &forged.identities,
        &forged.reasoning,
        &forged.graph_class,
        &forged.structure_source,
        forged.identifier.as_deref(),
        forged.estimator.as_deref(),
    )
    .unwrap();
    let forged_bytes = rewrite_contract_section(
        &consumed.body,
        schema_names(&data),
        "dist-f-tampered-program",
        &forged,
    );
    let forged = consume_analysis_result(&forged_bytes).unwrap();
    assert!(
        forged
            .acceptance
            .unresolved
            .iter()
            .any(|item| item.as_ref() == "program.functional_program")
    );
    // A different *well-formed* expression is also invalid when it no longer
    // names the selected identification product. Rehashing cannot repair that
    // causal binding.
    let mut forged = contract.clone();
    let retained = forged.program.as_mut().unwrap().functional_program.as_mut().unwrap();
    assert!(retained.arena.nodes.len() > 1);
    let alternate = if retained.source == 0 { 1 } else { 0 };
    retained.source = alternate;
    retained.executable = alternate;
    assert!(
        antecedent_io::functional_program_from_wire(
            retained,
            antecedent_expr::ProgramLimits::default(),
        )
        .is_ok()
    );
    forged.identities.program =
        *antecedent_io::program_digest(forged.program.as_ref().unwrap()).unwrap().as_bytes();
    forged.seal = antecedent_io::contract_seal(
        &forged.identities,
        &forged.reasoning,
        &forged.graph_class,
        &forged.structure_source,
        forged.identifier.as_deref(),
        forged.estimator.as_deref(),
    )
    .unwrap();
    let forged_bytes = rewrite_contract_section(
        &consumed.body,
        schema_names(&data),
        "dist-f-valid-wrong-program",
        &forged,
    );
    let forged = consume_analysis_result(&forged_bytes).unwrap();
    assert!(
        forged
            .acceptance
            .unresolved
            .iter()
            .any(|item| item.as_ref() == "program.functional_binding")
    );
    let labels: std::collections::HashMap<_, _> =
        executed_functional_labels(&consumed.contract.as_ref().unwrap().target.query)
            .into_iter()
            .collect();
    assert_eq!(labels["query_kind"], "distribution");
    assert_eq!(labels["outcome"], "1");
}

#[test]
fn licensed_family_distribution_bayesian_consumes() {
    let ctx = ExecutionContext::for_tests(18);
    let (data, dag, query) = distribution_fixture();
    let built = Study::tabular(data.clone())
        .graph(dag)
        .query(CausalQuery::Distribution(query))
        .identifier(IdentifierId::GeneralId)
        .estimator(EstimatorId::FunctionalDistribution)
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(128)))
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let (result, consumed) = consume_licensed_family(
        &built,
        |prepared| prepared.estimate(&data, &ctx).unwrap(),
        "dist-b",
        &ctx,
    );
    let dist = result.distribution.as_ref().expect("distribution payload");
    assert!((dist.mean - 0.7).abs() < 0.08);
    assert!(result.posterior.is_some());
    assert_eq!(consumed.body.estimate, Some(result.effect()));
}

fn response_curve_fixture() -> (TabularData, Dag, ResponseQuery) {
    let n = 240;
    let z: Vec<f64> = (0..n).map(|i| (i as f64 / 17.0).sin()).collect();
    let treatment: Vec<f64> =
        (0..n).map(|i| z[i] + (i as f64 / 11.0).cos() + (i % 7) as f64 * 0.03).collect();
    let outcome: Vec<f64> = (0..n)
        .map(|i| 1.0 + 2.0 * treatment[i] + 0.8 * z[i] + (i as f64 / 13.0).sin() * 0.05)
        .collect();
    let data = TabularData::from_f64_columns([
        ("treatment", treatment.as_slice()),
        ("outcome", outcome.as_slice()),
        ("confounder", z.as_slice()),
    ])
    .unwrap();
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(vec![-0.5, 0.0, 0.5].into()),
        ),
    });
    (data, dag, query)
}

#[test]
fn licensed_family_response_curve_frequentist_consumes() {
    let ctx = ExecutionContext::for_tests(50);
    let (data, dag, query) = response_curve_fixture();
    let built = Study::tabular(data.clone())
        .graph(dag)
        .query(CausalQuery::Response(query))
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let (result, consumed) = consume_licensed_family(
        &built,
        |prepared| prepared.estimate(&data, &ctx).unwrap(),
        "resp-f",
        &ctx,
    );
    let ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) =
        &result.response.as_ref().expect("response payload").estimate
    else {
        panic!("expected a point-identified response surface");
    };
    assert_eq!(mean.len(), 3);
    for (got, truth) in mean.iter().zip([0.0, 1.0, 2.0]) {
        assert!((got - truth).abs() < 0.25, "grid point {got} vs {truth}");
    }
    let labels: std::collections::HashMap<_, _> =
        executed_functional_labels(&consumed.contract.as_ref().unwrap().target.query)
            .into_iter()
            .collect();
    assert_eq!(labels["query_kind"], "response");
    assert_eq!(labels["temporal_coordinates"], "none");
}

fn mixture_data() -> TabularData {
    let mut treatment = Vec::new();
    let mut outcome = Vec::new();
    let mut confounder = Vec::new();
    for _ in 0..(160 / 16) {
        for (z, t, count) in [(0.0, 0.0, 6), (0.0, 1.0, 2), (1.0, 0.0, 2), (1.0, 1.0, 6)] {
            for row in 0..count {
                let epsilon = if row % 2 == 0 { -0.2 } else { 0.2 };
                treatment.push(t);
                confounder.push(z);
                outcome.push(2.0 * t + 2.0 * z + epsilon);
            }
        }
    }
    TabularData::from_f64_columns([
        ("t", treatment.as_slice()),
        ("y", outcome.as_slice()),
        ("z", confounder.as_slice()),
    ])
    .unwrap()
}

#[test]
fn licensed_family_conditional_graph_posterior_preserves_mass() {
    let ctx = ExecutionContext::for_tests(18);
    let data = mixture_data();
    let query = ConditionalEffectQuery::try_new(
        AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
            .with_effect_modifiers([VariableId::from_raw(2)]),
    )
    .unwrap();
    let built = Study::tabular(data.clone())
        .graph_posterior(mixture_graph_posterior())
        .query(CausalQuery::ConditionalEffect(query))
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let (result, consumed) = consume_licensed_family(
        &built,
        |prepared| prepared.estimate(&data, &ctx).unwrap(),
        "ce-gp",
        &ctx,
    );
    assert_eq!(result.identification.status, IdentificationStatus::GraphDependent);
    assert!(result.effect().is_nan(), "graph-dependent estimands withhold a scalar");
    let structural = result.structural_response.as_ref().unwrap();
    let values: Vec<_> = structural
        .atoms
        .iter()
        .filter_map(|atom| match atom.value {
            Some(antecedent_core::ResponseValue::Scalar(value)) => Some(value),
            _ => None,
        })
        .collect();
    assert_eq!(values.len(), 2);
    assert!(values.iter().all(|value| (value - 2.0).abs() < 0.15));
    let slot = consumed
        .contract
        .as_ref()
        .and_then(|section| section.reasoning.identification.value.as_ref())
        .expect("consumed identification slot");
    assert!((slot.unidentified_mass - 0.2).abs() < 1e-9);
    assert!((slot.identified_mass - 0.8).abs() < 1e-9);
    let claim = result.claim(&built.prepare(&ctx).unwrap().contract().unwrap(), &ctx).unwrap();
    match &claim.reasoning.identification {
        SlotAvailability::Available(slot) => {
            assert_eq!(slot.weight_basis.as_deref(), Some("posterior_probability"));
            assert!(slot.full_mass_scope);
        }
        other => panic!("graph-posterior mass must remain available, got {other:?}"),
    }
    assert_eq!(claim.kind, ClaimKind::Bounds);
}

#[test]
fn licensed_family_cpdag_conditional_refuses_unweighted_average() {
    let ctx = ExecutionContext::for_tests(4);
    let (data, dag, query) = conditional_effect_fixture();
    let cpdag = Cpdag::from_dag(&dag);
    let built = Study::tabular(data.clone())
        .graph(cpdag)
        .query(CausalQuery::ConditionalEffect(query))
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let inspected = built.inspect().unwrap();
    assert_eq!(inspected.graph_class.as_str(), "Cpdag");
    let (result, consumed) = consume_licensed_family(
        &built,
        |prepared| prepared.estimate(&data, &ctx).unwrap(),
        "ce-cpdag",
        &ctx,
    );
    assert!((result.effect() - 3.0).abs() < 0.2);
    let prepared = built.prepare(&ctx).unwrap();
    let preview = prepared.preview_transform(TransformIntent::AverageUnweightedClass).unwrap();
    assert!(preview.refused);
    assert!(
        preview
            .obligations
            .iter()
            .any(|item| item.id.as_ref() == "transform.unweighted_class_prior")
    );
    if let Some(mixture) = &result.structural_response {
        assert_eq!(mixture.weight_basis.as_str(), "completion_enumeration");
        assert_ne!(mixture.weight_basis.as_str(), "posterior_probability");
    }
    let labels: std::collections::HashMap<_, _> =
        executed_functional_labels(&consumed.contract.as_ref().unwrap().target.query)
            .into_iter()
            .collect();
    assert_eq!(labels["query_kind"], "conditional_effect");
}

fn lag1_series() -> (TimeSeriesData, TemporalDag) {
    let n = 60usize;
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; n];
    for t in 1..n {
        x[t] = (t as f64 * 0.05).sin();
        y[t] = 0.5 * x[t - 1];
    }
    let data = TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())]).unwrap();
    let series = TimeSeriesData::try_new(
        data.storage().clone(),
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap();
    let mut graph = TemporalDag::empty();
    let x1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(x1, y0).unwrap();
    (series, graph)
}

fn pulse_query() -> TemporalEffectQuery {
    TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1)
}

fn consume_series_family(
    series: &TimeSeriesData,
    graph: impl IntoGraphInput,
    query: TemporalEffectQuery,
    artifact_id: &str,
    inference: InferenceMode,
) -> (StudyResult, antecedent_io::AnalysisResultConsumption) {
    consume_series_family_refute(series, graph, query, artifact_id, inference, RefuteSuite::None)
}

fn consume_series_family_refute(
    series: &TimeSeriesData,
    graph: impl IntoGraphInput,
    query: TemporalEffectQuery,
    artifact_id: &str,
    inference: InferenceMode,
    refute: RefuteSuite,
) -> (StudyResult, antecedent_io::AnalysisResultConsumption) {
    let ctx = ExecutionContext::for_tests(1);
    let built = Study::series(series.clone())
        .graph(graph)
        .temporal_query(query)
        .inference(inference)
        .refute(refute)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    consume_licensed_family(
        &built,
        |prepared| prepared.estimate_series(series, &ctx).unwrap(),
        artifact_id,
        &ctx,
    )
}

#[test]
fn licensed_family_pulse_temporal_consumes() {
    let (series, graph) = lag1_series();
    let (result, consumed) =
        consume_series_family(&series, graph, pulse_query(), "pulse-f", InferenceMode::Frequentist);
    assert!(result.effect().is_finite());
    assert!((result.effect() - 0.5).abs() < 0.15);
    assert_eq!(
        consumed.acceptance.unresolved.as_ref(),
        &[std::sync::Arc::<str>::from("dependencies.checked_temporal_dag_effect_operation")]
    );
    let labels: std::collections::HashMap<_, _> =
        executed_functional_labels(&consumed.contract.as_ref().unwrap().target.query)
            .into_iter()
            .collect();
    assert_eq!(labels["query_kind"], "temporal_effect");
    assert!(labels["temporal_coordinates"].contains("horizon:1"));
}

#[test]
fn licensed_family_sustained_temporal_consumes() {
    let (series, graph) = lag1_series();
    let query = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::sustained(-1, -1))
        .with_horizon_steps(1);
    let (result, consumed) =
        consume_series_family(&series, graph, query, "sustained-f", InferenceMode::Frequentist);
    assert!((result.effect() - 0.5).abs() < 0.15);
    assert_eq!(
        consumed.acceptance.unresolved.as_ref(),
        &[std::sync::Arc::<str>::from("dependencies.checked_temporal_dag_effect_operation")]
    );
    let labels: std::collections::HashMap<_, _> =
        executed_functional_labels(&consumed.contract.as_ref().unwrap().target.query)
            .into_iter()
            .collect();
    assert_eq!(labels["query_kind"], "temporal_effect");
    assert!(labels["temporal_coordinates"].contains("horizon:1"));
}

fn static_discover_params() -> StaticDiscoverParams {
    StaticDiscoverParams {
        alpha: 0.05,
        max_cond_size: 3,
        fdr: None,
        ci: resolve_ci("parcorr", None).unwrap(),
        screen_pc: false,
        max_subset: None,
    }
}

fn temporal_discover_params(max_lag: u32) -> DiscoverParams {
    DiscoverParams {
        max_lag,
        alpha: 0.05,
        fdr: None,
        ci: resolve_ci("parcorr", None).unwrap(),
        multi_dataset: MultiDatasetConstraints::default(),
        max_cond_size: 2,
    }
}

/// Accept directed discovery marks; leave undirected marks as the class.
fn accept_static_class(mut review: CpdagReview) -> AcceptedGraph {
    let pending = review.pending_edges.clone();
    for &(from, to) in pending.iter() {
        review = review.accept_edge(from, to);
    }
    assert_eq!(review.graph.conflict_edge_count(), 0);
    AcceptedGraph::accept(review).expect("undirected marks stay on the CPDAG class")
}

fn accept_temporal_class(mut review: TemporalCpdagReview) -> AcceptedGraph {
    let pending = review.pending_edges.clone();
    for &(from, to) in pending.iter() {
        review = review.accept_edge(from, to);
    }
    assert_eq!(review.graph.conflict_edge_count(), 0);
    AcceptedGraph::accept(review).expect("undirected marks stay on the TemporalCpdag class")
}

#[test]
fn composition_pc_and_ges_accepted_class_stays_cpdag() {
    let ctx = ExecutionContext::for_tests(7);
    let (data, dag, query) = confounded_scm(400, 7);
    let vars = [VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2)];
    let params = static_discover_params();

    let pc = discover_pc(&data, &vars, &params, &ctx).unwrap();
    let undirected = pc.review.pending_undirected.len();
    let class = accept_static_class(pc.review);
    assert_eq!(class.class().as_str(), "Cpdag");
    assert_eq!(class.algorithm_id(), Some("pc"));
    if undirected > 0 {
        assert!(class.as_cpdag().unwrap().undirected_edge_count() > 0);
    }

    let built = Study::tabular(data.clone())
        .graph(class)
        .query(query.clone())
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let inspected = built.inspect().unwrap();
    assert_eq!(inspected.graph_class.as_str(), "Cpdag");
    assert_eq!(inspected.structure_source, StructureSource::Accepted);
    assert_eq!(inspected.discovery_algorithm.as_deref(), Some("pc"));
    let prepared = built.prepare(&ctx).unwrap();
    let unweighted = prepared.preview_transform(TransformIntent::AverageUnweightedClass).unwrap();
    assert!(unweighted.refused, "class path must not average over completions");
    let (result, _) = consume_licensed_family(
        &built,
        |prepared| prepared.estimate(&data, &ctx).unwrap(),
        "pc-class",
        &ctx,
    );
    assert!(result.effect().is_finite());

    let dag_built = Study::tabular(data.clone())
        .graph(AcceptedGraph::dag(dag))
        .query(query.clone())
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    assert_eq!(dag_built.inspect().unwrap().graph_class.as_str(), "Dag");
    let class_ids = prepared.contract().unwrap().identities;
    let dag_ids = dag_built.prepare(&ctx).unwrap().contract().unwrap().identities;
    assert_eq!(class_ids.target, dag_ids.target);
    assert_ne!(class_ids.program, dag_ids.program);
    let (dag_result, _) = consume_licensed_family(
        &dag_built,
        |prepared| prepared.estimate(&data, &ctx).unwrap(),
        "pc-dag",
        &ctx,
    );
    assert!((dag_result.effect() - 2.0).abs() < 0.2);

    let ges = discover_ges(&data, &vars, &params, &ctx).unwrap();
    let ges_class = accept_static_class(ges.review);
    assert_eq!(ges_class.class().as_str(), "Cpdag");
    assert_eq!(ges_class.algorithm_id(), Some("ges"));
    let ges_built = Study::tabular(data.clone())
        .graph(ges_class)
        .query(query)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let ges_inspected = ges_built.inspect().unwrap();
    assert_eq!(ges_inspected.graph_class.as_str(), "Cpdag");
    assert_eq!(ges_inspected.discovery_algorithm.as_deref(), Some("ges"));
    let (ges_result, _) = consume_licensed_family(
        &ges_built,
        |prepared| prepared.estimate(&data, &ctx).unwrap(),
        "ges-class",
        &ctx,
    );
    assert!(ges_result.effect().is_finite());
}

fn lag1_discoverable_series() -> (TimeSeriesData, TemporalDag) {
    let n = 160usize;
    let mut rng = CausalRng::from_seed(11);
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; n];
    for t in 1..n {
        let u1 = rng.next_f64().max(1e-12);
        let u2 = rng.next_f64();
        x[t] = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        y[t] = 0.8 * x[t - 1]
            + 0.05
                * (-2.0 * rng.next_f64().max(1e-12).ln()).sqrt()
                * (2.0 * std::f64::consts::PI * rng.next_f64()).cos();
    }
    let data = TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())]).unwrap();
    let series = TimeSeriesData::try_new(
        data.storage().clone(),
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap();
    let mut graph = TemporalDag::empty();
    let x1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(x1, y0).unwrap();
    (series, graph)
}

#[test]
fn composition_pcmci_plus_accepted_temporal_cpdag_retains_class() {
    let ctx = ExecutionContext::for_tests(11);
    let (series, dag) = lag1_discoverable_series();
    let vars = [VariableId::from_raw(0), VariableId::from_raw(1)];
    let discovered =
        discover_pcmci_plus(&series, &vars, &temporal_discover_params(1), &ctx).unwrap();
    let undirected = discovered.review.pending_undirected.len();
    let class = accept_temporal_class(discovered.review);
    assert_eq!(class.class().as_str(), "TemporalCpdag");
    assert_eq!(class.algorithm_id(), Some("pcmci_plus"));
    if undirected > 0 {
        assert!(class.as_temporal_cpdag().unwrap().undirected_edge_count() > 0);
    }

    let query = pulse_query().with_max_history_lag(Some(4));
    let built = Study::series(series.clone())
        .graph(class)
        .temporal_query(query.clone())
        .inference(InferenceMode::Frequentist)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let inspected = built.inspect().unwrap();
    assert_eq!(inspected.graph_class.as_str(), "TemporalCpdag");
    assert_eq!(inspected.structure_source, StructureSource::Accepted);
    assert_eq!(inspected.discovery_algorithm.as_deref(), Some("pcmci_plus"));
    let (result, consumed) = consume_licensed_family(
        &built,
        |prepared| prepared.estimate_series(&series, &ctx).unwrap(),
        "pcmci-plus-class",
        &ctx,
    );
    assert!(result.effect().is_finite());
    let labels: std::collections::HashMap<_, _> =
        executed_functional_labels(&consumed.contract.as_ref().unwrap().target.query)
            .into_iter()
            .collect();
    assert_eq!(labels["query_kind"], "temporal_effect");

    let dag_built = Study::series(series.clone())
        .graph(AcceptedGraph::temporal_dag(dag))
        .temporal_query(query)
        .inference(InferenceMode::Frequentist)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let class_ids = built.prepare(&ctx).unwrap().contract().unwrap().identities;
    let dag_ids = dag_built.prepare(&ctx).unwrap().contract().unwrap().identities;
    assert_eq!(class_ids.target, dag_ids.target);
    assert_ne!(class_ids.program, dag_ids.program);
    assert_eq!(dag_built.inspect().unwrap().graph_class.as_str(), "TemporalDag");
}

#[test]
fn composition_two_programs_compare_only_on_shared_data() {
    let ctx = ExecutionContext::for_tests(18);
    let data = mixture_data();
    let query = ConditionalEffectQuery::try_new(
        AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
            .with_effect_modifiers([VariableId::from_raw(2)]),
    )
    .unwrap();
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();

    let dag_built = Study::tabular(data.clone())
        .graph(AcceptedGraph::dag(dag))
        .query(CausalQuery::ConditionalEffect(query.clone()))
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let mixture_built = Study::tabular(data.clone())
        .graph_posterior(mixture_graph_posterior())
        .query(CausalQuery::ConditionalEffect(query))
        .bootstrap_replicates(0)
        .build()
        .unwrap();

    let dag_inspected = dag_built.inspect().unwrap();
    let mixture_inspected = mixture_built.inspect().unwrap();
    assert_eq!(dag_inspected.identities.target, mixture_inspected.identities.target);
    assert_eq!(dag_inspected.structure_source, StructureSource::Accepted);
    assert_eq!(mixture_inspected.structure_source, StructureSource::GraphPosterior);

    let dag_prepared = dag_built.prepare(&ctx).unwrap();
    let dag_contract = dag_prepared.contract().unwrap();
    let dag_result = dag_prepared.estimate(&data, &ctx).unwrap();
    let mixture_prepared = mixture_built.prepare(&ctx).unwrap();
    let mixture_contract = mixture_prepared.contract().unwrap();
    let mixture_result = mixture_prepared.estimate(&data, &ctx).unwrap();
    assert_eq!(dag_contract.identities.target, mixture_contract.identities.target);
    assert_ne!(dag_contract.identities.program, mixture_contract.identities.program);
    assert_ne!(dag_contract.identities.identification, mixture_contract.identities.identification);

    let left = dag_result.claim(&dag_contract, &ctx).unwrap();
    let right = mixture_result.claim(&mixture_contract, &ctx).unwrap();
    assert_eq!(right.kind, ClaimKind::Bounds);
    match &right.reasoning.identification {
        SlotAvailability::Available(slot) => {
            assert!((slot.unidentified_mass - 0.2).abs() < 1e-9);
            assert!((slot.identified_mass - 0.8).abs() < 1e-9);
        }
        other => panic!("mixture identification must remain available, got {other:?}"),
    }
    assert_eq!(mixture_result.identification.status, IdentificationStatus::GraphDependent);
    match compose_claims(&[&left, &right], ClaimOperation::Compare, None, None) {
        DerivedClaimOutcome::Comparison(report) => {
            assert!(!report.operation.synthesizes());
        }
        other => panic!("expected comparison, got {other:?}"),
    }
    match compose_claims(&[&left, &right], ClaimOperation::Pool, None, None) {
        DerivedClaimOutcome::Refused { restriction, .. } => {
            assert_eq!(&*restriction, "unlicensed_synthesis");
        }
        other => panic!("expected pool refusal, got {other:?}"),
    }
}

#[test]
fn composition_score_table_retarget_reuses_scores_and_refuses_illegal_weights() {
    let ctx = ExecutionContext::for_tests(41);
    let (data, dag, query) = confounded_scm(256, 41);
    let built = Study::tabular(data.clone())
        .graph(AcceptedGraph::dag(dag))
        .query(query)
        .estimator(EstimatorId::Aipw)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    assert_eq!(built.inspect().unwrap().support_status.unwrap().as_str(), "licensed");
    let mut prepared = built.prepare(&ctx).unwrap();
    let before = prepared.contract().unwrap();
    let score_before = prepared.score_reuse_identity().unwrap().expect("score key");
    let preview = prepared.preview_transform(TransformIntent::Retarget).unwrap();
    assert!(!preview.refused);
    assert!(preview.binds_program(before.identities.program.expect("prepared program")));
    let weights = vec![1.0; data.row_count()];
    let retargeted = prepared.apply_retarget(&preview, &weights, &[], &ctx).unwrap();
    assert_eq!(
        prepared.contract().unwrap().identities.identification,
        before.identities.identification
    );
    assert_eq!(prepared.score_reuse_identity().unwrap().expect("score key"), score_before);
    assert!(retargeted.diagnostics.iter().any(|d| d.code.as_ref() == "exec.identify.cached"));
    assert!((retargeted.effect() - 2.0).abs() < 0.35);
    let illegal = prepared.retarget(&weights, &[VariableId::from_raw(0)], &ctx);
    assert!(illegal.is_err(), "treatment-dependent weights must refuse");

    let (other, _, _) = confounded_scm(256, 42);
    assert_eq!(other.row_count(), data.row_count());
    prepared.refresh(other, &ctx).unwrap();
    let after = prepared.contract().unwrap();
    assert_eq!(before.identities.identification, after.identities.identification);
    assert_eq!(before.identities.program, after.identities.program);
    assert_ne!(before.identities.data_snapshot, after.identities.data_snapshot);
    assert_ne!(prepared.score_reuse_identity().unwrap().expect("score key"), score_before);
}

#[test]
fn composition_artifact_reload_in_separate_process() {
    const FLAG: &str = "ANTECEDENT_I6_RELOAD";
    if let Ok(path) = std::env::var(FLAG) {
        let bytes = std::fs::read(path).unwrap();
        let consumed = consume_analysis_result(&bytes).unwrap();
        assert!(!consumed.acceptance.accepts_as_verified_program());
        assert_eq!(
            consumed.acceptance.unresolved.as_ref(),
            &[std::sync::Arc::<str>::from("dependencies.linear_fit_sufficient_statistics")]
        );
        let section = consumed.contract.as_ref().expect("reloaded contract");
        assert_eq!(section.graph_class.as_str(), "Dag");
        assert!(section.reasoning.identification.value.is_some());
        assert!(section.reasoning.support.value.is_some());
        assert!(section.reasoning.uncertainty.value.is_some());
        assert!(section.reasoning.assumptions.value.is_some());
        println!("I6_RELOAD_OK");
        return;
    }

    let ctx = ExecutionContext::for_tests(41);
    let (data, dag, query) = confounded_scm(80, 41);
    let built = study(data.clone(), dag, query);
    let prepared = built.prepare(&ctx).unwrap();
    let contract = prepared.contract().unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let bytes = prepared.encode_contracted_result(&result, "i6-reload", &ctx).unwrap();
    let path = std::env::temp_dir().join(format!("antecedent-i6-{}.bin", std::process::id()));
    std::fs::write(&path, bytes).unwrap();

    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .env(FLAG, &path)
        .args(["composition_artifact_reload_in_separate_process", "--exact", "--nocapture"])
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&path);
    assert!(
        output.status.success(),
        "child reload failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("I6_RELOAD_OK"), "{stdout}");
    let consumed = consume_analysis_result(
        &prepared.encode_contracted_result(&result, "i6-parent", &ctx).unwrap(),
    )
    .unwrap();
    assert_eq!(
        consumed.contract.as_ref().map(|section| section.identities.program),
        Some(*contract.identities.program.expect("prepared program").as_bytes())
    );
}

#[test]
fn composition_prior_transfer_keeps_identification_and_refuses_mismatch() {
    use antecedent_prob::{
        ExternalPriorSource, ExternalPriorWeight, GaussianCoefficientPrior, PriorSet, PriorSpec,
        compose_external_priors,
    };
    use antecedent_validate::ConflictPolicy;

    let ctx = ExecutionContext::for_tests(1);
    let (data, dag, query) = confounded_scm(160, 19);
    let source = Study::tabular(data.clone())
        .graph(dag.clone())
        .query(query.clone())
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(32)))
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap()
        .estimate(&data, &ctx)
        .unwrap();
    let bytes = antecedent::io::encode_causal_posterior_bytes(
        source.posterior.as_ref().expect("source posterior"),
        "static-source",
    )
    .unwrap();
    let isotropic = Study::tabular(data.clone())
        .graph(dag.clone())
        .query(query.clone())
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(32)))
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let transferred = Study::tabular(data.clone())
        .graph(dag)
        .query(query)
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(32).prior_from_artifact(
                bytes.clone(),
                Some(PriorMapping::IdenticalCoefficientSubspace),
            ),
        ))
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    assert_eq!(transferred.inspect().unwrap().support_status.unwrap().as_str(), "licensed");
    let iso = isotropic.prepare(&ctx).unwrap();
    let xfer = transferred.prepare(&ctx).unwrap();
    let iso_ids = iso.contract().unwrap().identities;
    let xfer_ids = xfer.contract().unwrap().identities;
    assert_eq!(iso_ids.identification, xfer_ids.identification);
    assert_eq!(iso_ids.target, xfer_ids.target);
    assert_ne!(iso_ids.inference_binding, xfer_ids.inference_binding);
    match (
        &iso.contract().unwrap().reasoning.identification,
        &xfer.contract().unwrap().reasoning.identification,
    ) {
        (SlotAvailability::Available(a), SlotAvailability::Available(b)) => {
            assert_eq!(a.status, b.status);
            assert_eq!(a.unidentified_mass, b.unidentified_mass);
        }
        other => panic!("prepared identification must stay available, got {other:?}"),
    }
    let transferred_result = xfer.estimate(&data, &ctx).unwrap();
    assert!(transferred_result.posterior.is_some());
    assert_eq!(transferred_result.identification.status, source.identification.status);

    let catalog = PriorCatalog::from_sources(vec![PriorSourceRef::with_bytes(
        PriorSourceMeta::new(
            "static-source",
            EstimandFingerprint::new("ate", "t", "y"),
            "NonparametricallyIdentified",
        )
        .with_design(vec![
            DesignVariableSummary::new("t", DesignVariableRole::Treatment),
            DesignVariableSummary::new("y", DesignVariableRole::Outcome),
        ]),
        bytes,
    )]);
    catalog
        .require_usable(&TargetDesign::new(EstimandFingerprint::new("ate", "t", "y"), ["t", "y"]))
        .expect("same-design ATE catalog");
    let wrong_outcome = catalog
        .require_usable(&TargetDesign::new(EstimandFingerprint::new("ate", "t", "z"), ["t", "z"]));
    assert!(wrong_outcome.is_err(), "outcome mismatch must refuse");

    let (series, graph) = lag1_series();
    let pulse_bytes = pulse_prior_bytes(&series, &graph, &ctx);
    let pulse_catalog = PriorCatalog::from_sources(vec![PriorSourceRef::with_bytes(
        PriorSourceMeta::new(
            "pulse-source",
            EstimandFingerprint::new("pulse", "x", "y")
                .with_temporal(TemporalCoordinates::new([1], 1)),
            "NonparametricallyIdentified",
        )
        .with_design(vec![
            DesignVariableSummary::new("x", DesignVariableRole::Treatment),
            DesignVariableSummary::new("y", DesignVariableRole::Outcome),
        ]),
        pulse_bytes.clone(),
    )]);
    let lag_mismatch = pulse_catalog.require_usable(&TargetDesign::new(
        EstimandFingerprint::new("pulse", "x", "y").with_temporal(TemporalCoordinates::new([2], 2)),
        ["x", "y"],
    ));
    assert!(lag_mismatch.is_err(), "lag mismatch must refuse");

    let bad_map = Study::series(series.clone())
        .graph(graph.clone())
        .temporal_query(pulse_query())
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(32).prior_from_artifact(
                pulse_bytes.clone(),
                Some(PriorMapping::NamedParameters {
                    pairs: vec![("coef_x@lag1".into(), "not_a_coefficient".into())],
                }),
            ),
        ))
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let prepared_bad = bad_map.prepare(&ctx).unwrap();
    let iso_pulse = Study::series(series.clone())
        .graph(graph.clone())
        .temporal_query(pulse_query())
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(32)))
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    assert_eq!(
        prepared_bad.contract().unwrap().identities.identification,
        iso_pulse.contract().unwrap().identities.identification
    );
    let err = prepared_bad.estimate_series(&series, &ctx).unwrap_err();
    assert!(
        err.to_string().contains("unknown target coefficient"),
        "mapping mismatch must refuse at hydrate, got {err}"
    );

    let mut source_prior = PriorSet::new();
    source_prior.push(PriorSpec::GaussianCoefficients(GaussianCoefficientPrior {
        mean: Arc::from(vec![0.0, 8.0]),
        variance: Arc::from(vec![0.01, 0.01]),
    }));
    let sources = Arc::<[ExternalPriorSource]>::from(vec![ExternalPriorSource {
        id: Arc::from("conflict-bank"),
        prior: source_prior,
        weight: ExternalPriorWeight::power(1.0).unwrap(),
        ess: None,
    }]);
    let composed = compose_external_priors(&sources, &PriorSet::weakly_informative(2)).unwrap();
    let conflicted = Study::series(series.clone())
        .graph(graph)
        .temporal_query(pulse_query())
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(32).prior_from_composed(
                sources,
                composed,
                Some(ConflictPolicy::try_new(0.05, 1.0).unwrap()),
            ),
        ))
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let conflict_prepared = conflicted.prepare(&ctx).unwrap();
    assert_eq!(
        conflict_prepared.contract().unwrap().identities.identification,
        iso_pulse.contract().unwrap().identities.identification
    );
    let conflict_result = conflict_prepared.estimate_series(&series, &ctx).unwrap();
    assert_ne!(format!("{:?}", conflict_result.identification.status), "NotIdentified");
    assert!(
        conflict_result.posterior.as_ref().is_some_and(|p| p.conflict_summary.is_some())
            || conflict_result
                .diagnostics
                .iter()
                .any(|d| d.code.as_ref() == "bayes.prior_bank.conflict"),
        "conflict policy must surface a diagnostic, not flip identification"
    );
}

#[test]
fn composition_adversarial_boundaries_refuse_stronger_claims() {
    let ctx = ExecutionContext::for_tests(31);
    let (data, dag, query) = confounded_scm(128, 31);
    let mut prepared = study(data.clone(), dag.clone(), query.clone()).prepare(&ctx).unwrap();
    let before = prepared.contract().unwrap();
    let preview = prepared.preview_transform(TransformIntent::CompatibleDataReplace).unwrap();
    assert!(preview.binds_contract(&before.identities));
    let changed = {
        let treatment = data.float64_slice(VariableId::from_raw(0)).unwrap();
        let outcome = data.float64_slice(VariableId::from_raw(1)).unwrap();
        data.with_replaced_float(
            VariableId::from_raw(1),
            outcome.iter().zip(treatment).map(|(y, t)| y + t).collect::<Vec<_>>().into(),
        )
        .unwrap()
    };
    prepared.refresh(changed.clone(), &ctx).unwrap();
    let stale = prepared.apply_refresh(&preview, changed, &ctx);
    assert!(
        stale.is_err_and(|err| err.to_string().contains("does not bind")),
        "stale preview must refuse, not refresh"
    );

    let graph_change = prepared.preview_transform(TransformIntent::ChangeGraph).unwrap();
    let identification =
        graph_change.layer(antecedent_core::SemanticLayer::Identification).unwrap();
    assert!(
        identification
            .effects
            .contains(&antecedent_core::TransformEffect::RequiresReidentification)
    );

    let mixture = Study::tabular(mixture_data())
        .graph_posterior(mixture_graph_posterior())
        .query(CausalQuery::ConditionalEffect(
            ConditionalEffectQuery::try_new(
                AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
                    .with_effect_modifiers([VariableId::from_raw(2)]),
            )
            .unwrap(),
        ))
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let mixture_prepared = mixture.prepare(&ctx).unwrap();
    let drop_mass =
        mixture_prepared.preview_transform(TransformIntent::AverageUnweightedClass).unwrap();
    assert!(drop_mass.refused, "unidentified mass must not be dropped by unweighted average");
    let mixture_result = mixture_prepared.estimate(&mixture_data(), &ctx).unwrap();
    let mixture_claim = mixture_result.claim(&mixture_prepared.contract().unwrap(), &ctx).unwrap();
    match &mixture_claim.reasoning.identification {
        SlotAvailability::Available(slot) => {
            assert!((slot.unidentified_mass - 0.2).abs() < 1e-9);
        }
        other => panic!("mixture mass must remain available, got {other:?}"),
    }

    let ctx_b = ExecutionContext::for_tests(32);
    let claim_prepared = study(data.clone(), dag, query).prepare(&ctx).unwrap();
    let contract = claim_prepared.contract().unwrap();
    let left = claim_prepared.estimate(&data, &ctx).unwrap().claim(&contract, &ctx).unwrap();
    let right = claim_prepared.estimate(&data, &ctx_b).unwrap().claim(&contract, &ctx_b).unwrap();
    assert!(SharedEvidenceRef::forwarded_duplicate(left.claim_id, left.claim_id));
    assert!(!SharedEvidenceRef::forwarded_duplicate(left.claim_id, right.claim_id));
    match compose_claims(&[&left, &right], ClaimOperation::Contrast, None, None) {
        DerivedClaimOutcome::Refused { restriction, .. } => {
            assert_eq!(&*restriction, "marginal_intervals_do_not_determine_contrast");
        }
        other => panic!("shared-data contrast without alignment must refuse, got {other:?}"),
    }
    let result = claim_prepared.estimate(&data, &ctx).unwrap();
    let bytes = claim_prepared.encode_contracted_result(&result, "adv-lossy", &ctx).unwrap();
    let (sender, _) = accept_claim(&bytes, &ConsumerProfile::full()).unwrap();
    let (scalar, lossy) = project_lossy_scalar(&sender);
    assert!(!scalar.accepts_as_claim);
    assert!(!lossy.equivalent_claim());

    assert_ne!(
        std::mem::discriminant(&antecedent_core::ResponseUncertainty::PointwiseBand {
            level: 0.95,
            lower: Arc::from([0.0]),
            upper: Arc::from([1.0]),
            interpretation: antecedent_core::IntervalInterpretation::Confidence,
            draws: None,
        }),
        std::mem::discriminant(&antecedent_core::ResponseUncertainty::SimultaneousBand {
            level: 0.95,
            lower: Arc::from([0.0]),
            upper: Arc::from([1.0]),
            replicates: 100,
            interpretation: antecedent_core::IntervalInterpretation::Confidence,
        })
    );

    let (refused_data, _, refused_query) = conditional_effect_fixture();
    let mut admg = Admg::with_variables(3);
    admg.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    admg.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    let refused = Study::tabular(refused_data)
        .graph(admg)
        .query(CausalQuery::ConditionalEffect(refused_query))
        .bootstrap_replicates(0)
        .inspect()
        .unwrap();
    assert_eq!(refused.support_status.unwrap().as_str(), "refused");
    assert!(!matches!(refused.support_status.unwrap().as_str(), "licensed"));
}

#[test]
fn licensed_family_refused_cell_inspects_as_structured_refusal() {
    let (data, _, query) = conditional_effect_fixture();
    let mut admg = Admg::with_variables(3);
    admg.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    admg.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    let inspected = Study::tabular(data.clone())
        .graph(admg.clone())
        .query(CausalQuery::ConditionalEffect(query.clone()))
        .bootstrap_replicates(0)
        .inspect()
        .unwrap();
    assert_eq!(inspected.support_status.unwrap().as_str(), "refused");
    assert!(matches!(inspected.reasoning.identification, SlotAvailability::Unavailable { .. }));
    assert!(inspected.identities.identification_product.is_none());
    let err = Study::tabular(data)
        .graph(admg)
        .query(CausalQuery::ConditionalEffect(query))
        .bootstrap_replicates(0)
        .build()
        .unwrap_err();
    assert!(matches!(err, antecedent::CausalError::Support { .. }), "{err}");
}

#[test]
fn licensed_family_na_cell_inspects_as_incomplete_contract() {
    let (data, dag, _) = confounded_scm(32, 3);
    let inspected = Study::tabular(data.clone())
        .graph(dag.clone())
        .query(TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0))
        .bootstrap_replicates(0)
        .inspect()
        .unwrap();
    match inspected.support_status.unwrap() {
        antecedent::CellStatus::NotApplicable { reason } => {
            assert!(reason.contains("temporal"), "{reason}");
        }
        other => panic!("Pulse on a static DAG must be n/a, got {other:?}"),
    }
    assert!(matches!(inspected.reasoning.identification, SlotAvailability::Unavailable { .. }));
    assert!(inspected.identities.identification_product.is_none());
    let err = Study::tabular(data)
        .graph(dag)
        .query(TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0))
        .bootstrap_replicates(0)
        .build()
        .unwrap_err();
    match err {
        antecedent::CausalError::Support {
            id: antecedent::SupportRefusal::NotApplicable, ..
        } => {}
        other => panic!("expected n/a support refusal, got {other}"),
    }
}

#[test]
fn capability_preflight_and_build_agree_on_refused_blockers() {
    let (data, _, query) = conditional_effect_fixture();
    let mut admg = Admg::with_variables(3);
    admg.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    admg.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    let builder = Study::tabular(data)
        .graph(admg)
        .query(CausalQuery::ConditionalEffect(query))
        .bootstrap_replicates(0);
    let preflight = builder.capability().unwrap();
    assert_eq!(preflight.applicability, SemanticApplicability::Unlicensed);
    assert!(!preflight.is_executable());
    assert_eq!(preflight.primary_blocker_id(), Some("support.refused"));
    assert!(preflight.blockers.iter().all(|blocker| blocker.scientific));
    assert!(
        preflight
            .neighbors
            .iter()
            .any(|n| n.coordinate.as_ref().starts_with("AverageEffect:Admg:")),
        "{:?}",
        preflight.neighbors
    );
    assert!(preflight.neighbors.iter().all(|n| !n.coordinate.as_ref().contains(":Dag:")));
    let inspected = builder.clone().inspect().unwrap();
    assert_eq!(inspected.support_status.unwrap().as_str(), "refused");
    assert!(matches!(inspected.reasoning.identification, SlotAvailability::Unavailable { .. }));
    assert_eq!(inspected.capability().primary_blocker_id(), preflight.primary_blocker_id());
    let err = builder.build().unwrap_err();
    assert_eq!(
        err.blocker_id().as_ref().map(|blocker| blocker.id.as_ref()),
        preflight.primary_blocker_id()
    );
}

#[test]
fn capability_preflight_and_build_agree_on_na_blockers() {
    let (data, dag, _) = confounded_scm(32, 3);
    let builder = Study::tabular(data)
        .graph(dag)
        .query(TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0))
        .bootstrap_replicates(0);
    let preflight = builder.capability().unwrap();
    assert_eq!(preflight.applicability, SemanticApplicability::Impossible);
    assert_eq!(preflight.primary_blocker_id(), Some("support.not_applicable"));
    assert!(
        preflight.neighbors.iter().any(|n| n.coordinate.as_ref().starts_with("AverageEffect:Dag:")),
        "{:?}",
        preflight.neighbors
    );
    assert!(
        preflight.neighbors.iter().all(|n| n.coordinate.as_ref().split(':').nth(1) == Some("Dag"))
    );
    let err = builder.build().unwrap_err();
    assert_eq!(
        err.blocker_id().as_ref().map(|blocker| blocker.id.as_ref()),
        Some("support.not_applicable")
    );
}

#[test]
fn capability_inspect_cannot_open_a_refused_cell() {
    let (data, _, query) = conditional_effect_fixture();
    let mut admg = Admg::with_variables(3);
    admg.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    admg.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    let inspected = Study::tabular(data.clone())
        .graph(admg.clone())
        .query(CausalQuery::ConditionalEffect(query.clone()))
        .bootstrap_replicates(0)
        .inspect()
        .unwrap();
    let report = inspected.capability();
    assert_eq!(report.applicability, SemanticApplicability::Unlicensed);
    assert!(!report.is_executable());
    assert!(matches!(inspected.reasoning.identification, SlotAvailability::Unavailable { .. }));
    assert!(inspected.identities.identification_product.is_none());
    assert!(
        Study::tabular(data)
            .graph(admg)
            .query(CausalQuery::ConditionalEffect(query))
            .bootstrap_replicates(0)
            .build()
            .is_err()
    );
}

#[test]
fn capability_no_graph_has_no_neighbors() {
    let (data, _, query) = confounded_scm(16, 11);
    let report = Study::tabular(data).query(query).capability().unwrap();
    assert_eq!(report.readiness, Some(OperationReadiness::BindingMissing));
    assert_eq!(report.primary_blocker_id(), Some("binding.graph"));
    assert!(report.neighbors.is_empty());
    assert!(!report.is_executable());
}

#[test]
fn capability_cpdag_does_not_recommend_dag_relabel() {
    let (data, _, _) = confounded_scm(16, 13);
    let mut cpdag = Cpdag::with_variables(3);
    cpdag.insert_undirected(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    cpdag.insert_undirected(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    cpdag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let report = Study::tabular(data)
        .graph(cpdag)
        .query(TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0))
        .bootstrap_replicates(0)
        .capability()
        .unwrap();
    assert_eq!(report.applicability, SemanticApplicability::Impossible);
    assert!(
        report.neighbors.iter().all(|n| n.coordinate.as_ref().split(':').nth(1) == Some("Cpdag"))
    );
    assert!(report.neighbors.iter().all(|n| !n.coordinate.as_ref().contains(":Dag:")));
}

#[test]
fn capability_licensed_cell_is_executable_and_agrees_with_prepare() {
    let ctx = ExecutionContext::for_tests(1);
    let (data, dag, query) = confounded_scm(32, 17);
    let builder = Study::tabular(data).graph(dag).query(query).bootstrap_replicates(0);
    let preflight = builder.capability().unwrap();
    assert_eq!(preflight.applicability, SemanticApplicability::Licensed);
    assert_eq!(preflight.readiness, Some(OperationReadiness::Executable));
    assert!(preflight.is_executable());
    assert!(preflight.neighbors.is_empty());
    let prepared = builder.build().unwrap().prepare(&ctx).unwrap();
    let prepared_report = prepared.capability().unwrap();
    assert_eq!(prepared_report.applicability, SemanticApplicability::Licensed);
    assert!(prepared_report.is_executable());
    assert_eq!(prepared_report.coordinate, preflight.coordinate);
}

#[test]
fn capability_rank_designs_uses_scoped_operation_not_matrix_row() {
    let ctx = ExecutionContext::for_tests(1);
    let (data, dag, query) = confounded_scm(32, 19);
    let inspected = Study::tabular(data.clone())
        .graph(dag.clone())
        .query(query.clone())
        .bootstrap_replicates(0)
        .inspect()
        .unwrap();
    let rank_preflight = inspected.capability_for(OperationKind::RankDesigns);
    assert!(!OperationKind::RankDesigns.uses_matrix_row());
    assert_eq!(rank_preflight.primary_blocker_id(), Some("binding.identification_product"));
    assert!(
        rank_preflight
            .coordinate
            .as_deref()
            .is_some_and(|coord| coord.starts_with("AverageEffect:Dag:")),
        "{:?}",
        rank_preflight.coordinate
    );

    let (data, _, cond) = conditional_effect_fixture();
    let mut admg = Admg::with_variables(3);
    admg.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    admg.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    let refused = Study::tabular(data)
        .graph(admg)
        .query(CausalQuery::ConditionalEffect(cond))
        .bootstrap_replicates(0)
        .inspect()
        .unwrap();
    let refused_rank = refused.capability_for(OperationKind::RankDesigns);
    assert_eq!(refused_rank.primary_blocker_id(), Some("operation.unlicensed"));
    assert!(
        refused_rank
            .coordinate
            .as_deref()
            .is_some_and(|coord| coord.starts_with("ConditionalEffect:Admg:")),
        "scoped ops keep the real matrix coordinate, got {:?}",
        refused_rank.coordinate
    );

    let (data, dag, query) = confounded_scm(32, 23);
    let prepared = Study::tabular(data)
        .graph(dag)
        .query(query)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let prepared_rank = prepared.contract().unwrap().capability_for(OperationKind::RankDesigns);
    assert!(prepared_rank.is_executable());
}

fn temporal_dose_horizon() -> (TimeSeriesData, TemporalDag, serde_json::Value) {
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/response/temporal_dose_horizon/expected.json"
    ))
    .unwrap();
    let n = usize::try_from(pin["generation"]["n"].as_u64().unwrap()).unwrap();
    let t: Vec<f64> = (0..n)
        .map(|i| match i % 4 {
            0 | 2 => 0.0,
            1 => 1.0,
            3 => -1.0,
            _ => unreachable!(),
        })
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            1.0 + 2.0 * i.checked_sub(1).map_or(0.0, |j| t[j])
                + 3.0 * i.checked_sub(2).map_or(0.0, |j| t[j])
        })
        .collect();
    let series = TimeSeriesData::try_new(
        TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice())])
            .unwrap()
            .storage()
            .clone(),
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap();
    let mut graph = TemporalDag::empty();
    let t1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let t2 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(2)).unwrap();
    let y0 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(t1, y0).unwrap();
    graph.insert_directed(t2, y0).unwrap();
    (series, graph, pin)
}

fn temporal_spec(pin: &serde_json::Value) -> TemporalResponseSpec {
    let horizons: Vec<u32> = pin["contract"]["horizons"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| u32::try_from(value.as_u64().unwrap()).unwrap())
        .collect();
    let at = i32::try_from(pin["contract"]["policy"]["at"].as_i64().unwrap()).unwrap();
    TemporalResponseSpec::new(horizons, TemporalPolicy::pulse(at), None).unwrap()
}

fn pin_f64s(pin: &serde_json::Value, path: &[&str]) -> Vec<f64> {
    let mut node = pin;
    for key in path {
        node = &node[*key];
    }
    node.as_array().unwrap().iter().map(|value| value.as_f64().unwrap()).collect()
}

fn consume_temporal_response(
    series: &TimeSeriesData,
    graph: impl IntoGraphInput,
    query: ResponseQuery,
    artifact_id: &str,
) -> (StudyResult, antecedent_io::AnalysisResultConsumption) {
    let ctx = ExecutionContext::for_tests(21);
    let built = Study::series(series.clone())
        .graph(graph)
        .query(CausalQuery::Response(query))
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    consume_licensed_family(
        &built,
        |prepared| prepared.estimate_series(series, &ctx).unwrap(),
        artifact_id,
        &ctx,
    )
}

#[test]
fn licensed_family_temporal_response_curve_consumes() {
    let (series, graph, pin) = temporal_dose_horizon();
    let doses = pin_f64s(&pin, &["contract", "dose_grid"]);
    let expected = pin_f64s(&pin, &["contract", "surface", "mean"]);
    let atol = pin["tolerance"]["atol"].as_f64().unwrap();
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from(doses)),
        ),
    })
    .with_temporal(temporal_spec(&pin));
    let (result, consumed) = consume_temporal_response(&series, graph, query, "tresp-f");
    assert_eq!(
        consumed.acceptance.unresolved.as_ref(),
        &[std::sync::Arc::<str>::from("dependencies.checked_temporal_response_operation")]
    );
    let ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) =
        &result.response.as_ref().expect("temporal response").estimate
    else {
        panic!("expected a point-identified temporal response surface");
    };
    assert_eq!(mean.len(), expected.len());
    for (got, truth) in mean.iter().zip(&expected) {
        assert!((got - truth).abs() <= atol, "surface {got} vs {truth}");
    }
    let labels: std::collections::HashMap<_, _> =
        executed_functional_labels(&consumed.contract.as_ref().unwrap().target.query)
            .into_iter()
            .collect();
    assert_eq!(labels["query_kind"], "response");
    assert_ne!(labels["temporal_coordinates"], "none");
}

#[test]
fn licensed_family_temporal_sequence_consumes() {
    let (series, graph, pin) = temporal_dose_horizon();
    let expected = pin_f64s(&pin, &["contract", "intervention_paths", "sequence_two_step_set_1"]);
    let last_step = pin_f64s(&pin, &["contract", "intervention_paths", "set_1"]);
    let atol = pin["tolerance"]["atol"].as_f64().unwrap();
    assert_ne!(expected.as_slice(), last_step.as_slice());
    let sequence = Intervention::Sequence(InterventionSequence::new(
        [1.0, 1.0]
            .into_iter()
            .map(|value| SequencedIntervention {
                intervention: Intervention::set(VariableId::from_raw(0), Value::f64(value)),
                temporal: TemporalPolicy::pulse(0),
            })
            .collect::<Vec<_>>(),
    ));
    let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(1),
        interventions: Arc::from([sequence]),
    })
    .with_temporal(temporal_spec(&pin));
    let (result, consumed) = consume_temporal_response(&series, graph, query, "tseq-f");
    let ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) =
        &result.response.as_ref().expect("sequence response").estimate
    else {
        panic!("expected a point-identified sequence surface");
    };
    for (got, truth) in mean.iter().zip(&expected) {
        assert!((got - truth).abs() <= atol, "sequence {got} vs {truth}");
    }
    let labels: std::collections::HashMap<_, _> =
        executed_functional_labels(&consumed.contract.as_ref().unwrap().target.query)
            .into_iter()
            .collect();
    assert_eq!(labels["query_kind"], "response");
    assert_ne!(labels["temporal_coordinates"], "none");
}

#[test]
fn licensed_family_mediation_bayesian_consumes() {
    let ctx = ExecutionContext::for_tests(13);
    let (data, dag, pin) = staged_static_kinds();
    let mut query = MediationQuery::binary(
        VariableId::from_raw(0),
        VariableId::from_raw(2),
        Arc::from([VariableId::from_raw(1)]),
        MediationContrast::NaturalDirect,
    );
    query.control =
        Intervention::set(query.treatment, Value::f64(pin["control"].as_f64().unwrap()));
    query.active = Intervention::set(query.treatment, Value::f64(pin["active"].as_f64().unwrap()));
    let built = Study::tabular(data.clone())
        .graph(dag)
        .query(CausalQuery::Mediation(query))
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64)))
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let (result, consumed) = consume_licensed_family(
        &built,
        |prepared| prepared.estimate(&data, &ctx).unwrap(),
        "med-b",
        &ctx,
    );
    assert!(result.posterior.is_some());
    assert!(
        (result.estimate.ate - pin["direct"].as_f64().unwrap()).abs()
            < pin["tolerance"].as_f64().unwrap() + 0.15
    );
    assert_eq!(consumed.body.estimate, Some(result.estimate.ate));
}

#[test]
fn licensed_family_response_curve_bayesian_consumes() {
    let ctx = ExecutionContext::for_tests(50);
    let (data, dag, query) = response_curve_fixture();
    let built = Study::tabular(data.clone())
        .graph(dag)
        .query(CausalQuery::Response(query))
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64)))
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let (result, consumed) = consume_licensed_family(
        &built,
        |prepared| prepared.estimate(&data, &ctx).unwrap(),
        "resp-b",
        &ctx,
    );
    assert_eq!(result.logical_plan.estimator.as_deref(), Some("response.bayesian"));
    let ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) =
        &result.response.as_ref().expect("response payload").estimate
    else {
        panic!("expected a point-identified Bayesian response surface");
    };
    for (got, truth) in mean.iter().zip([0.0, 1.0, 2.0]) {
        assert!((got - truth).abs() < 0.35, "grid point {got} vs {truth}");
    }
    let uncertainty = consumed
        .contract
        .as_ref()
        .unwrap()
        .reasoning
        .uncertainty
        .value
        .as_ref()
        .expect("published Bayesian band has an uncertainty disclosure");
    assert!(uncertainty.components.iter().any(|component| {
        component.source == "parameter"
            && component.target == "posterior_pointwise_band"
            && !component.omitted
    }));
    let labels: std::collections::HashMap<_, _> =
        executed_functional_labels(&consumed.contract.as_ref().unwrap().target.query)
            .into_iter()
            .collect();
    assert_eq!(labels["query_kind"], "response");
}

#[test]
fn licensed_family_counterfactual_bayesian_consumes() {
    let ctx = ExecutionContext::for_tests(13);
    let (data, dag, pin) = staged_static_kinds();
    let query = CounterfactualQuery::new(
        VariableId::from_raw(2),
        Arc::from([Intervention::set(
            VariableId::from_raw(0),
            Value::f64(pin["active"].as_f64().unwrap()),
        )]),
    )
    .with_control_level(pin["control"].as_f64().unwrap());
    let built = Study::tabular(data.clone())
        .graph(dag)
        .query(CausalQuery::Counterfactual(query))
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64)))
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let (result, consumed) = consume_licensed_family(
        &built,
        |prepared| prepared.estimate(&data, &ctx).unwrap(),
        "cf-b",
        &ctx,
    );
    assert!(result.posterior.is_some());
    let cf = result.counterfactual.as_ref().expect("counterfactual payload");
    assert!((cf.mean_ite - pin["counterfactual_mean"].as_f64().unwrap()).abs() < 0.15);
    assert_eq!(consumed.body.estimate, Some(result.effect()));
    assert!(
        consumed
            .acceptance
            .unresolved
            .iter()
            .any(|reason| { reason.as_ref() == "dependencies.fitted_counterfactual_mechanisms" })
    );
}

#[test]
fn licensed_family_conditional_effect_cheap_validation_consumes() {
    let ctx = ExecutionContext::for_tests(1);
    let (data, dag, query) = conditional_effect_fixture();
    let built = Study::tabular(data.clone())
        .graph(dag)
        .query(CausalQuery::ConditionalEffect(query))
        .refute(RefuteSuite::Cheap)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let inspected = built.inspect().unwrap();
    let coordinate = match &inspected.reasoning.support {
        SlotAvailability::Available(slot) => slot.matrix_coordinate.as_deref().unwrap_or(""),
        other => panic!("inspect must publish support, got {other:?}"),
    };
    assert!(coordinate.contains("cheap"), "cheap validation must appear on {coordinate}");
    let (result, consumed) = consume_licensed_family(
        &built,
        |prepared| prepared.estimate(&data, &ctx).unwrap(),
        "ce-cheap",
        &ctx,
    );
    assert!(!result.refutations.is_empty());
    assert!((result.effect() - 3.0).abs() < 1e-8);
    assert_eq!(consumed.body.estimate, Some(result.effect()));
}

fn pag_pin() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../conformance/estimate/pag_ate_envelope_identified/expected.json"
    ))
    .unwrap()
}

fn pag_from_pin(pin: &serde_json::Value) -> (TabularData, Pag) {
    let columns: Vec<&str> =
        pin["columns"].as_array().unwrap().iter().map(|value| value.as_str().unwrap()).collect();
    let mut values: Vec<Vec<f64>> = vec![Vec::new(); columns.len()];
    for cell in pin["contingency_table"].as_array().unwrap() {
        let count = usize::try_from(cell["count"].as_u64().unwrap()).unwrap();
        for (index, name) in columns.iter().enumerate() {
            values[index].extend(std::iter::repeat_n(cell[*name].as_f64().unwrap(), count));
        }
    }
    let pairs: Vec<(&str, &[f64])> =
        columns.iter().zip(values.iter()).map(|(name, col)| (*name, col.as_slice())).collect();
    let data = TabularData::from_f64_columns(pairs).unwrap();
    let index = |name: &str| {
        u32::try_from(columns.iter().position(|column| *column == name).unwrap()).unwrap()
    };
    let endpoint = |mark: &str| match mark {
        "tail" => Endpoint::Tail,
        "arrow" => Endpoint::Arrow,
        "circle" => Endpoint::Circle,
        other => panic!("unknown endpoint {other}"),
    };
    let mut pag = Pag::with_variables(u32::try_from(columns.len()).unwrap());
    for edge in pin["graph"]["marked_edges"].as_array().unwrap() {
        pag.insert_marked(MarkedEdge {
            a: DenseNodeId::from_raw(index(edge[0].as_str().unwrap())),
            b: DenseNodeId::from_raw(index(edge[1].as_str().unwrap())),
            at_a: endpoint(edge[2].as_str().unwrap()),
            at_b: endpoint(edge[3].as_str().unwrap()),
            middle: MiddleMark::Empty,
        })
        .unwrap();
    }
    (data, pag)
}

#[test]
fn licensed_family_pag_conditional_refuses_unweighted_average() {
    let ctx = ExecutionContext::for_tests(1);
    let pin = pag_pin();
    let (data, pag) = pag_from_pin(&pin);
    let modifier = u32::try_from(
        pin["columns"]
            .as_array()
            .unwrap()
            .iter()
            .position(|name| name.as_str() == pin["conditional"]["modifier"].as_str())
            .unwrap(),
    )
    .unwrap();
    let treatment = u32::try_from(
        pin["columns"]
            .as_array()
            .unwrap()
            .iter()
            .position(|name| name.as_str() == pin["query"]["treatment"].as_str())
            .unwrap(),
    )
    .unwrap();
    let outcome = u32::try_from(
        pin["columns"]
            .as_array()
            .unwrap()
            .iter()
            .position(|name| name.as_str() == pin["query"]["outcome"].as_str())
            .unwrap(),
    )
    .unwrap();
    let query = ConditionalEffectQuery::try_new(
        AverageEffectQuery::with_levels(
            VariableId::from_raw(treatment),
            VariableId::from_raw(outcome),
            pin["query"]["control_level"].as_f64().unwrap(),
            pin["query"]["active_level"].as_f64().unwrap(),
        )
        .with_effect_modifiers([VariableId::from_raw(modifier)]),
    )
    .unwrap();
    let built = Study::tabular(data.clone())
        .graph(pag)
        .query(CausalQuery::ConditionalEffect(query))
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let inspected = built.inspect().unwrap();
    assert_eq!(inspected.graph_class.as_str(), "Pag");
    let (result, consumed) = consume_licensed_family(
        &built,
        |prepared| prepared.estimate(&data, &ctx).unwrap(),
        "ce-pag",
        &ctx,
    );
    assert_eq!(result.identification.status, IdentificationStatus::GraphDependent);
    let freq = &pin["conditional"]["frequentist"];
    assert!(
        (result.effect() - freq["expected_ate"].as_f64().unwrap()).abs()
            < freq["absolute_tolerance"].as_f64().unwrap()
    );
    let slot = consumed
        .contract
        .as_ref()
        .and_then(|section| section.reasoning.identification.value.as_ref())
        .expect("consumed PAG identification slot");
    let unidentified = pin["identification"]["unidentified_mass"].as_f64().unwrap();
    let identified = pin["identification"]["identified_mass"].as_f64().unwrap();
    let total = identified + unidentified;
    assert!((slot.unidentified_mass - unidentified / total).abs() < 1e-9);
    assert!((slot.identified_mass - identified / total).abs() < 1e-9);
    let prepared = built.prepare(&ctx).unwrap();
    let preview = prepared.preview_transform(TransformIntent::AverageUnweightedClass).unwrap();
    assert!(preview.refused);
    assert!(
        preview
            .obligations
            .iter()
            .any(|item| item.id.as_ref() == "transform.unweighted_class_prior")
    );
}

#[test]
fn licensed_family_response_observation_obligations_inspect() {
    let ctx = ExecutionContext::for_tests(50);
    let n = 240;
    let confounder: Vec<f64> = (0..n).map(|i| (i as f64 / 17.0).sin()).collect();
    let treatment: Vec<f64> =
        (0..n).map(|i| confounder[i] + (i as f64 / 11.0).cos() + (i % 7) as f64 * 0.03).collect();
    let outcome: Vec<f64> = (0..n)
        .map(|i| 1.0 + 2.0 * treatment[i] + 0.8 * confounder[i] + (i as f64 / 13.0).sin() * 0.05)
        .collect();
    let selected = vec![1.0; n];
    let observed = TabularData::from_f64_columns([
        ("treatment", treatment.as_slice()),
        ("outcome", outcome.as_slice()),
        ("confounder", confounder.as_slice()),
        ("selected", selected.as_slice()),
    ])
    .unwrap();
    let mut graph = Dag::with_variables(4);
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(vec![-0.5, 0.0, 0.5].into()),
        ),
    })
    .with_observation(
        ObservationSpec::Selected {
            latent: VariableId::from_raw(1),
            observed: VariableId::from_raw(1),
            indicator: VariableId::from_raw(3),
        },
        [ObservationAssumption::OutcomeIndependentGiven(Arc::from([
            VariableId::from_raw(0),
            VariableId::from_raw(2),
        ]))],
    );
    let built = Study::tabular(observed.clone())
        .graph(graph)
        .query(CausalQuery::Response(query))
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let inspected = built.inspect().unwrap();
    match &inspected.reasoning.assumptions {
        SlotAvailability::Available(slot) => {
            assert!(
                slot.obligations.iter().any(|obligation| {
                    obligation.id.as_ref() == "observation.0"
                        && obligation.description.as_ref() == "outcome_independent_given"
                        && obligation.kind == ObligationKind::UserAssertion
                        && obligation.status == AssumptionStatus::Declared
                }),
                "declared observation must enter as an unresolved user assertion, not a MAR proof"
            );
        }
        other => panic!("inspect must publish assumption obligations, got {other:?}"),
    }
    assert!(matches!(inspected.reasoning.identification, SlotAvailability::Unavailable { .. }));
    let (result, consumed) = consume_licensed_family(
        &built,
        |prepared| prepared.estimate(&observed, &ctx).unwrap(),
        "resp-obs",
        &ctx,
    );
    let ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) =
        &result.response.as_ref().expect("observed response").estimate
    else {
        panic!("expected a point-identified observed response");
    };
    assert_eq!(mean.len(), 3);
    let observation = consumed
        .contract
        .as_ref()
        .and_then(|section| section.observation.as_ref())
        .expect("consumed observation identity");
    assert!(
        observation.observation.iter().any(|tag| tag.starts_with("outcome_independent_given:")),
        "observation tags keep their conditioning variables: {:?}",
        observation.observation
    );
    let prepared = built.prepare(&ctx).unwrap();
    match &prepared.contract().unwrap().reasoning.identification {
        SlotAvailability::Available(slot) => {
            assert_eq!(slot.status, IdentificationStatus::NonparametricallyIdentified);
        }
        other => panic!("prepared identification must stay graphical, got {other:?}"),
    }
}

#[test]
fn licensed_family_prior_bank_does_not_create_identification() {
    let ctx = ExecutionContext::for_tests(1);
    let (series, graph) = lag1_series();
    let source = Study::series(series.clone())
        .graph(graph.clone())
        .temporal_query(pulse_query())
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(32)))
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap()
        .estimate_series(&series, &ctx)
        .unwrap();
    let bytes = antecedent::io::encode_causal_posterior_bytes(
        source.posterior.as_ref().expect("source posterior"),
        "source",
    )
    .unwrap();
    let isotropic = Study::series(series.clone())
        .graph(graph.clone())
        .temporal_query(pulse_query())
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(32)))
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let transferred = Study::series(series.clone())
        .graph(graph)
        .temporal_query(pulse_query())
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(32).prior_from_artifact(bytes, None),
        ))
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let inspected = transferred.inspect().unwrap();
    assert_eq!(inspected.support_status.unwrap().as_str(), "licensed");
    assert!(matches!(inspected.reasoning.identification, SlotAvailability::Unavailable { .. }));
    let iso = isotropic.prepare(&ctx).unwrap();
    let xfer = transferred.prepare(&ctx).unwrap();
    let iso_contract = iso.contract().unwrap();
    let xfer_contract = xfer.contract().unwrap();
    assert_eq!(iso_contract.identities.identification, xfer_contract.identities.identification);
    assert_eq!(iso_contract.identities.target, xfer_contract.identities.target);
    assert_ne!(
        iso_contract.identities.inference_binding,
        xfer_contract.identities.inference_binding
    );
    match (&iso_contract.reasoning.identification, &xfer_contract.reasoning.identification) {
        (SlotAvailability::Available(a), SlotAvailability::Available(b)) => {
            assert_eq!(a.status, b.status);
            assert_eq!(a.unidentified_mass, b.unidentified_mass);
        }
        other => panic!("prepared identification must stay available, got {other:?}"),
    }
    let (result, consumed) = consume_licensed_family(
        &transferred,
        |prepared| prepared.estimate_series(&series, &ctx).unwrap(),
        "pulse-prior",
        &ctx,
    );
    assert!(result.posterior.is_some());
    let binding = consumed
        .contract
        .as_ref()
        .and_then(|section| section.inference_binding.as_ref())
        .expect("consumed inference binding");
    assert!(
        binding
            .bayesian
            .as_ref()
            .is_some_and(|bayes| bayes.prior_artifact.is_some() && bayes.prior.is_none()),
        "a transferred prior binds the artifact bytes it hydrates from"
    );
    let slot = consumed
        .contract
        .as_ref()
        .and_then(|section| section.reasoning.identification.value.as_ref())
        .expect("consumed identification slot");
    assert_eq!(slot.unidentified_mass, 0.0);
}

fn lag1_temporal_cpdag() -> TemporalCpdag {
    let (_, graph) = lag1_series();
    TemporalCpdag::from_temporal_dag(&graph)
}

fn lag1_temporal_pag() -> TemporalPag {
    let mut graph = TemporalPag::empty();
    let x1 = graph.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = graph.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(x1, y0).unwrap();
    graph
}

fn lag1_gp_posterior() -> GraphPosterior {
    GraphPosterior::new(
        2,
        vec![1.0],
        vec![0],
        vec![0.0; 4],
        vec![0.0; 4],
        1.0,
        InferenceDiagnostics::analytic("v110_gp_pulse"),
        0,
    )
    .unwrap()
    .with_lagged_marginals(1, vec![0.0, 1.0, 0.0, 0.0])
    .unwrap()
    .with_lag_masks(vec![2])
    .unwrap()
}

fn intervention_response_fixture() -> (TabularData, Dag, ResponseQuery, f64, f64) {
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/response/intervention_response/expected.json"
    ))
    .unwrap();
    let n = usize::try_from(pin["generation"]["n"].as_u64().unwrap()).unwrap();
    let z: Vec<f64> = (0..n).map(|i| (i as f64 / 17.0).sin()).collect();
    let treatment: Vec<f64> = (0..n).map(|i| z[i] + (i as f64 / 11.0).cos()).collect();
    let outcome: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * treatment[i] + 0.8 * z[i]).collect();
    let data = TabularData::from_f64_columns([
        ("t", treatment.as_slice()),
        ("y", outcome.as_slice()),
        ("z", z.as_slice()),
    ])
    .unwrap();
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(1),
        interventions: Arc::from([Intervention::set(
            VariableId::from_raw(0),
            Value::f64(pin["contract"]["intervention"]["value"].as_f64().unwrap()),
        )]),
    });
    (
        data,
        dag,
        query,
        pin["contract"]["true_response"].as_f64().unwrap(),
        pin["tolerance"]["truth_absolute"].as_f64().unwrap(),
    )
}

fn inspect_coordinate(built: &Study) -> String {
    match &built.inspect().unwrap().reasoning.support {
        SlotAvailability::Available(slot) => {
            slot.matrix_coordinate.as_deref().unwrap_or("").to_string()
        }
        other => panic!("inspect must publish support, got {other:?}"),
    }
}

fn consume_gp_series(
    series: &TimeSeriesData,
    posterior: GraphPosterior,
    query: TemporalEffectQuery,
    artifact_id: &str,
    inference: InferenceMode,
) -> StudyResult {
    let ctx = ExecutionContext::for_tests(1);
    let built = Study::series(series.clone())
        .graph_posterior(posterior)
        .temporal_query(query)
        .inference(inference)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let inspected = built.inspect().unwrap();
    assert_eq!(inspected.support_status.unwrap().as_str(), "licensed");
    assert!(matches!(inspected.reasoning.identification, SlotAvailability::Unavailable { .. }));
    let prepared = built.prepare(&ctx).unwrap();
    let contract = prepared.contract().unwrap();
    assert_eq!(inspected.identities.target, contract.identities.target);
    let preview = prepared.preview_transform(TransformIntent::CompatibleDataReplace).unwrap();
    assert!(!preview.refused);
    let result = prepared.estimate_series(series, &ctx).unwrap();
    let claim = result.claim(&contract, &ctx).unwrap();
    assert_eq!(claim.identities.program, contract.identities.program);
    let bytes = prepared
        .encode_contracted_result(&result, artifact_id, &ctx)
        .expect("GP temporal encode must consume lagged-node products through the DBN namespace");
    let consumed = consume_analysis_result(&bytes).unwrap();
    assert!(consumed.acceptance.accepts_as_verified_program());
    assert_eq!(consumed.contract.as_ref().unwrap().graph_class.as_str(), "TemporalDag");
    result
}

fn class_envelope_series() -> TimeSeriesData {
    let n = 80usize;
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut z = vec![0.0; n];
    for i in 0..n {
        z[i] = if i % 2 == 0 { 0.0 } else { 1.0 };
        t[i] = 0.3 + 0.4 * z[i] + 0.05 * ((i as f64) * 0.017).sin();
        if i > 0 {
            y[i] = 1.0 + 2.0 * t[i - 1] + 0.5 * z[i - 1];
        }
    }
    TimeSeriesData::from_f64_columns(
        [("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())],
        1,
    )
    .unwrap()
}

fn class_envelope_cpdag() -> TemporalCpdag {
    let mut graph = TemporalCpdag::empty();
    let t1 = graph.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = graph.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let z1 = graph.add_lagged(VariableId::from_raw(2), Lag::from_raw(1)).unwrap();
    graph.insert_directed(z1, y0).unwrap();
    graph.insert_directed(t1, y0).unwrap();
    graph.insert_undirected(z1, t1).unwrap();
    graph
}

fn class_envelope_pag() -> TemporalPag {
    let mut graph = TemporalPag::empty();
    let t1 = graph.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = graph.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let z1 = graph.add_lagged(VariableId::from_raw(2), Lag::from_raw(1)).unwrap();
    graph.insert_directed(z1, y0).unwrap();
    graph.insert_directed(t1, y0).unwrap();
    graph.insert_circle_circle_with_middle(z1, t1, antecedent_graph::MiddleMark::Empty).unwrap();
    graph
}

#[test]
fn licensed_family_pulse_bayesian_consumes() {
    let (series, graph) = lag1_series();
    let (result, consumed) = consume_series_family(
        &series,
        graph,
        pulse_query(),
        "pulse-b",
        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(32)),
    );
    assert!(result.posterior.is_some());
    assert!((result.effect() - 0.5).abs() < 0.2);
    let labels: std::collections::HashMap<_, _> =
        executed_functional_labels(&consumed.contract.as_ref().unwrap().target.query)
            .into_iter()
            .collect();
    assert_eq!(labels["query_kind"], "temporal_effect");
    assert!(labels["temporal_coordinates"].contains("horizon:1"));
}

#[test]
fn licensed_family_sustained_bayesian_consumes() {
    let (series, graph) = lag1_series();
    let query = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::sustained(-1, -1))
        .with_horizon_steps(1);
    let (result, consumed) = consume_series_family(
        &series,
        graph,
        query,
        "sustained-b",
        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(32)),
    );
    assert!(result.posterior.is_some());
    assert!((result.effect() - 0.5).abs() < 0.2);
    let labels: std::collections::HashMap<_, _> =
        executed_functional_labels(&consumed.contract.as_ref().unwrap().target.query)
            .into_iter()
            .collect();
    assert_eq!(labels["query_kind"], "temporal_effect");
}

#[test]
fn licensed_family_pulse_cheap_validation_consumes() {
    let (series, graph) = lag1_series();
    let built = Study::series(series.clone())
        .graph(graph.clone())
        .temporal_query(pulse_query())
        .refute(RefuteSuite::Cheap)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    assert!(inspect_coordinate(&built).contains("cheap"));
    let (result, consumed) = consume_series_family_refute(
        &series,
        graph,
        pulse_query(),
        "pulse-cheap",
        InferenceMode::Frequentist,
        RefuteSuite::Cheap,
    );
    assert!(!result.refutations.is_empty());
    assert!((result.effect() - 0.5).abs() < 0.15);
    assert_eq!(consumed.body.estimate, Some(result.effect()));
}

#[test]
fn licensed_family_pulse_full_validation_consumes() {
    let (series, graph) = lag1_series();
    let built = Study::series(series.clone())
        .graph(graph.clone())
        .temporal_query(pulse_query())
        .refute(RefuteSuite::Full)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    assert!(inspect_coordinate(&built).contains("full"));
    let (result, consumed) = consume_series_family_refute(
        &series,
        graph,
        pulse_query(),
        "pulse-full",
        InferenceMode::Frequentist,
        RefuteSuite::Full,
    );
    assert!(!result.refutations.is_empty());
    assert!((result.effect() - 0.5).abs() < 0.15);
    assert_eq!(consumed.body.estimate, Some(result.effect()));
}

#[test]
fn licensed_family_intervention_response_static_consumes() {
    let ctx = ExecutionContext::for_tests(52);
    let (data, dag, query, truth, tolerance) = intervention_response_fixture();
    let built = Study::tabular(data.clone())
        .graph(dag)
        .query(CausalQuery::Response(query))
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let (result, consumed) = consume_licensed_family(
        &built,
        |prepared| prepared.estimate(&data, &ctx).unwrap(),
        "ir-f",
        &ctx,
    );
    let ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) =
        result.response.as_ref().expect("intervention response").estimate
    else {
        panic!("expected a scalar intervention response");
    };
    assert!((value - truth).abs() <= tolerance, "value={value} truth={truth}");
    let labels: std::collections::HashMap<_, _> =
        executed_functional_labels(&consumed.contract.as_ref().unwrap().target.query)
            .into_iter()
            .collect();
    assert_eq!(labels["query_kind"], "response");
    assert_eq!(labels["temporal_coordinates"], "none");
}

#[test]
fn licensed_family_intervention_response_cheap_validation_consumes() {
    let ctx = ExecutionContext::for_tests(52);
    let (data, dag, query, truth, tolerance) = intervention_response_fixture();
    let built = Study::tabular(data.clone())
        .graph(dag)
        .query(CausalQuery::Response(query))
        .refute(RefuteSuite::Cheap)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    assert!(inspect_coordinate(&built).contains("cheap"));
    let (result, consumed) = consume_licensed_family(
        &built,
        |prepared| prepared.estimate(&data, &ctx).unwrap(),
        "ir-cheap",
        &ctx,
    );
    assert!(!result.refutations.is_empty());
    let ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) =
        result.response.as_ref().expect("intervention response").estimate
    else {
        panic!("expected a scalar intervention response");
    };
    assert!((value - truth).abs() <= tolerance);
    assert!(
        consumed
            .acceptance
            .unresolved
            .iter()
            .any(|reason| reason.as_ref() == "dependencies.checked_intervention_response_operation")
    );
}

#[test]
fn licensed_family_path_specific_bayesian_consumes() {
    let ctx = ExecutionContext::for_tests(18);
    let (data, dag, query) = path_specific_fixture();
    let built = Study::tabular(data.clone())
        .graph(dag)
        .query(CausalQuery::PathSpecific(query))
        .identifier(IdentifierId::PathSpecificNatural)
        .estimator(EstimatorId::FunctionalEffect)
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64)))
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let (result, consumed) = consume_licensed_family(
        &built,
        |prepared| prepared.estimate(&data, &ctx).unwrap(),
        "path-b",
        &ctx,
    );
    assert!(result.posterior.is_some());
    assert!(result.effect().is_finite());
    assert_eq!(consumed.body.estimate, Some(result.effect()));
    let labels: std::collections::HashMap<_, _> =
        executed_functional_labels(&consumed.contract.as_ref().unwrap().target.query)
            .into_iter()
            .collect();
    assert_eq!(labels["query_kind"], "path_specific");
}

#[test]
fn licensed_family_path_specific_cheap_validation_consumes() {
    let ctx = ExecutionContext::for_tests(1);
    let (data, dag, query) = path_specific_fixture();
    let built = Study::tabular(data.clone())
        .graph(dag)
        .query(CausalQuery::PathSpecific(query))
        .identifier(IdentifierId::PathSpecificNatural)
        .estimator(EstimatorId::FunctionalEffect)
        .refute(RefuteSuite::Cheap)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    assert!(inspect_coordinate(&built).contains("cheap"));
    let (result, consumed) = consume_licensed_family(
        &built,
        |prepared| prepared.estimate(&data, &ctx).unwrap(),
        "path-cheap",
        &ctx,
    );
    assert!(!result.refutations.is_empty());
    assert!(result.effect().is_finite());
    assert_eq!(consumed.body.estimate, Some(result.effect()));
}

#[test]
fn licensed_family_temporal_cpdag_pulse_consumes() {
    let (series, _) = lag1_series();
    let (result, consumed) = consume_series_family(
        &series,
        lag1_temporal_cpdag(),
        pulse_query(),
        "pulse-tcpdag",
        InferenceMode::Frequentist,
    );
    assert!((result.effect() - 0.5).abs() < 0.15);
    assert_eq!(consumed.contract.as_ref().unwrap().graph_class.as_str(), "TemporalCpdag");
    let labels: std::collections::HashMap<_, _> =
        executed_functional_labels(&consumed.contract.as_ref().unwrap().target.query)
            .into_iter()
            .collect();
    assert!(labels["temporal_coordinates"].contains("horizon:1"));
}

#[test]
fn licensed_family_temporal_pag_pulse_inspects_without_identified_mass() {
    let (series, _) = lag1_series();
    let built = Study::series(series.clone())
        .graph(lag1_temporal_pag())
        .temporal_query(pulse_query())
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let inspected = built.inspect().unwrap();
    assert_eq!(inspected.support_status.unwrap().as_str(), "licensed");
    assert_eq!(inspected.graph_class.as_str(), "TemporalPag");
    assert!(matches!(inspected.reasoning.identification, SlotAvailability::Unavailable { .. }));
    let ctx = ExecutionContext::for_tests(1);
    let err = built
        .prepare(&ctx)
        .and_then(|prepared| prepared.estimate_series(&series, &ctx))
        .unwrap_err();
    assert!(
        err.to_string().contains("no identified mass"),
        "directed-only TemporalPag Pulse must refuse without inventing identification, got {err}"
    );
    assert!(matches!(
        built.inspect().unwrap().reasoning.identification,
        SlotAvailability::Unavailable { .. }
    ));
}

#[test]
fn licensed_family_temporal_cpdag_response_curve_consumes() {
    let series = class_envelope_series();
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from([0.0, 1.0])),
        ),
    })
    .with_temporal(TemporalResponseSpec::new(vec![1], TemporalPolicy::pulse(-1), None).unwrap());
    let ctx = ExecutionContext::for_tests(21);
    let built = Study::series(series.clone())
        .graph(class_envelope_cpdag())
        .query(CausalQuery::Response(query))
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let inspected = built.inspect().unwrap();
    assert_eq!(inspected.support_status.unwrap().as_str(), "licensed");
    assert_eq!(inspected.graph_class.as_str(), "TemporalCpdag");
    let prepared = built.prepare(&ctx).unwrap();
    let result = prepared.estimate_series(&series, &ctx).unwrap();
    assert!(result.response.is_some());
    match prepared.encode_contracted_result(&result, "tresp-tcpdag", &ctx) {
        Ok(bytes) => {
            let consumed = consume_analysis_result(&bytes).unwrap();
            // The class response executes from a retained completion proof that
            // the portable artifact does not carry; an independent consumer must
            // name that missing operation rather than accept a program-less replay.
            assert!(consumed.acceptance.unresolved.iter().any(|dependency| {
                dependency.as_ref() == "dependencies.checked_temporal_class_response_operation"
            }));
            assert!(!consumed.acceptance.accepts_as_verified_program());
            assert_eq!(consumed.contract.as_ref().unwrap().graph_class.as_str(), "TemporalCpdag");
        }
        Err(err) => {
            let message = err.to_string();
            assert!(
                message.contains("variable id") && message.contains("header bounds"),
                "TemporalCpdag response must consume or refuse lagged-node products, got {err}"
            );
        }
    }
}

#[test]
fn licensed_family_temporal_pag_response_curve_inspects() {
    let series = class_envelope_series();
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from([0.0, 1.0])),
        ),
    })
    .with_temporal(TemporalResponseSpec::new(vec![1], TemporalPolicy::pulse(-1), None).unwrap());
    let ctx = ExecutionContext::for_tests(21);
    let built = Study::series(series.clone())
        .graph(class_envelope_pag())
        .query(CausalQuery::Response(query))
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let inspected = built.inspect().unwrap();
    assert_eq!(inspected.support_status.unwrap().as_str(), "licensed");
    assert_eq!(inspected.graph_class.as_str(), "TemporalPag");
    match built.prepare(&ctx).and_then(|prepared| prepared.estimate_series(&series, &ctx)) {
        Ok(result) => assert!(result.response.is_some()),
        Err(err) => {
            assert!(
                err.to_string().contains("no identified mass")
                    || err.to_string().contains("not identified")
                    || err.to_string().contains("no evaluable atom"),
                "non-identifying TemporalPag response must refuse without inventing identification, got {err}"
            );
        }
    }
}

fn identified_pag_pin() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../conformance/estimate/temporal_class_envelope/identified_pag.json"
    ))
    .unwrap()
}

#[test]
fn licensed_family_temporal_pag_pulse_consumes() {
    let pin = identified_pag_pin();
    let series = identified_pag_series(&pin);
    let expected = pin["pulse_ate"].as_f64().unwrap();
    let tolerance = pin["absolute_tolerance"].as_f64().unwrap();
    let (result, consumed) = consume_series_family(
        &series,
        identified_pag(&pin),
        pulse_query(),
        "pulse-tpag",
        InferenceMode::Frequentist,
    );
    assert!(
        (result.effect() - expected).abs() <= tolerance,
        "mixture {} vs {expected}",
        result.effect()
    );
    assert_eq!(consumed.contract.as_ref().unwrap().graph_class.as_str(), "TemporalPag");
    let slot = consumed
        .contract
        .as_ref()
        .and_then(|section| section.reasoning.identification.value.as_ref())
        .expect("consumed TemporalPag identification slot");
    let identified = pin["identification"]["identified_mass"].as_f64().unwrap();
    let unidentified = pin["identification"]["unidentified_mass"].as_f64().unwrap();
    let total = identified + unidentified;
    assert!((slot.identified_mass - identified / total).abs() < 1e-9);
    assert!((slot.unidentified_mass - unidentified / total).abs() < 1e-9);
    let labels: std::collections::HashMap<_, _> =
        executed_functional_labels(&consumed.contract.as_ref().unwrap().target.query)
            .into_iter()
            .collect();
    assert!(labels["temporal_coordinates"].contains("horizon:1"));
}

#[test]
fn licensed_family_temporal_pag_response_curve_consumes() {
    let pin = identified_pag_pin();
    let series = identified_pag_series(&pin);
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from([0.0, 1.0])),
        ),
    })
    .with_temporal(TemporalResponseSpec::new(vec![1], TemporalPolicy::pulse(-1), None).unwrap());
    let ctx = ExecutionContext::for_tests(21);
    let built = Study::series(series.clone())
        .graph(identified_pag(&pin))
        .query(CausalQuery::Response(query))
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let (result, consumed) = consume_licensed_family(
        &built,
        |prepared| prepared.estimate_series(&series, &ctx).unwrap(),
        "tresp-tpag",
        &ctx,
    );
    assert!(result.response.is_some());
    assert_eq!(consumed.contract.as_ref().unwrap().graph_class.as_str(), "TemporalPag");
}

#[test]
fn licensed_family_gp_pulse_temporal_consumes() {
    let (series, _) = lag1_series();
    let result = consume_gp_series(
        &series,
        lag1_gp_posterior(),
        pulse_query(),
        "pulse-gp",
        InferenceMode::Frequentist,
    );
    assert!((result.effect() - 0.5).abs() < 0.15);
}

#[test]
fn licensed_family_gp_pulse_bayesian_consumes() {
    let (series, _) = lag1_series();
    let result = consume_gp_series(
        &series,
        lag1_gp_posterior(),
        pulse_query(),
        "pulse-gp-b",
        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(32)),
    );
    assert!(result.posterior.is_some());
    assert!((result.effect() - 0.5).abs() < 0.2);
}

#[test]
fn licensed_family_response_independent_given_observation_inspect() {
    let ctx = ExecutionContext::for_tests(50);
    let n = 240;
    let confounder: Vec<f64> = (0..n).map(|i| (i as f64 / 17.0).sin()).collect();
    let treatment: Vec<f64> =
        (0..n).map(|i| confounder[i] + (i as f64 / 11.0).cos() + (i % 7) as f64 * 0.03).collect();
    let outcome: Vec<f64> = (0..n)
        .map(|i| 1.0 + 2.0 * treatment[i] + 0.8 * confounder[i] + (i as f64 / 13.0).sin() * 0.05)
        .collect();
    let selected = vec![1.0; n];
    let observed = TabularData::from_f64_columns([
        ("treatment", treatment.as_slice()),
        ("outcome", outcome.as_slice()),
        ("confounder", confounder.as_slice()),
        ("selected", selected.as_slice()),
    ])
    .unwrap();
    let mut graph = Dag::with_variables(4);
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(vec![-0.5, 0.0, 0.5].into()),
        ),
    })
    .with_observation(
        ObservationSpec::Selected {
            latent: VariableId::from_raw(1),
            observed: VariableId::from_raw(1),
            indicator: VariableId::from_raw(3),
        },
        [ObservationAssumption::IndependentGiven(Arc::from([
            VariableId::from_raw(0),
            VariableId::from_raw(2),
        ]))],
    );
    let built = Study::tabular(observed.clone())
        .graph(graph)
        .query(CausalQuery::Response(query))
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let inspected = built.inspect().unwrap();
    match &inspected.reasoning.assumptions {
        SlotAvailability::Available(slot) => {
            assert!(
                slot.obligations.iter().any(|obligation| {
                    obligation.id.as_ref() == "observation.0"
                        && obligation.description.as_ref() == "independent_given"
                        && obligation.kind == ObligationKind::UserAssertion
                        && obligation.status == AssumptionStatus::Declared
                }),
                "IndependentGiven must enter as a declared user assertion"
            );
        }
        other => panic!("inspect must publish assumption obligations, got {other:?}"),
    }
    assert_eq!(inspected.support_status.unwrap().as_str(), "licensed");
    let err =
        built.prepare(&ctx).and_then(|prepared| prepared.estimate(&observed, &ctx)).unwrap_err();
    assert!(
        err.to_string().contains("OutcomeIndependentGiven"),
        "Selected + IndependentGiven must refuse without treating missingness as MAR, got {err}"
    );
}

#[test]
fn licensed_family_response_structural_observation_inspect() {
    let ctx = ExecutionContext::for_tests(50);
    let (data, dag, query) = response_curve_fixture();
    let query = query.with_observation(
        ObservationSpec::Complete,
        [ObservationAssumption::Structural(Arc::from("gaussian_observation_likelihood"))],
    );
    let built = Study::tabular(data.clone())
        .graph(dag)
        .query(CausalQuery::Response(query))
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let inspected = built.inspect().unwrap();
    assert_eq!(inspected.support_status.unwrap().as_str(), "licensed");
    match &inspected.reasoning.assumptions {
        SlotAvailability::Available(slot) => {
            assert!(
                slot.obligations.iter().any(|obligation| {
                    obligation.description.as_ref() == "structural:gaussian_observation_likelihood"
                }),
                "structural observation must bind a declared obligation"
            );
        }
        other => panic!("inspect must publish assumption obligations, got {other:?}"),
    }
    assert!(matches!(inspected.reasoning.identification, SlotAvailability::Unavailable { .. }));
    match built.prepare(&ctx).and_then(|prepared| prepared.estimate(&data, &ctx)) {
        Ok(_) => {
            let (_, consumed) = consume_licensed_family(
                &built,
                |prepared| prepared.estimate(&data, &ctx).unwrap(),
                "resp-struct",
                &ctx,
            );
            let observation = consumed
                .contract
                .as_ref()
                .and_then(|section| section.observation.as_ref())
                .expect("consumed observation identity");
            assert!(
                observation
                    .observation
                    .iter()
                    .any(|tag| tag == "structural:gaussian_observation_likelihood")
            );
        }
        Err(err) => {
            assert!(
                err.to_string().contains("observation")
                    || err.to_string().contains("structural")
                    || matches!(err, antecedent::CausalError::Unsupported { .. })
                    || matches!(err, antecedent::CausalError::Support { .. }),
                "structural pair must refuse structurally, got {err}"
            );
        }
    }
}

#[test]
fn licensed_family_temporal_response_observation_independent_given() {
    let ctx = ExecutionContext::for_tests(21);
    let (series, graph, pin) = temporal_dose_horizon();
    let n = series.row_count();
    let t: Vec<f64> = (0..n)
        .map(|i| match i % 4 {
            0 | 2 => 0.0,
            1 => 1.0,
            3 => -1.0,
            _ => unreachable!(),
        })
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            1.0 + 2.0 * i.checked_sub(1).map_or(0.0, |j| t[j])
                + 3.0 * i.checked_sub(2).map_or(0.0, |j| t[j])
        })
        .collect();
    let selected = vec![1.0; n];
    let observed = TimeSeriesData::try_new(
        TabularData::from_f64_columns([
            ("t", t.as_slice()),
            ("y", y.as_slice()),
            ("selected", selected.as_slice()),
        ])
        .unwrap()
        .storage()
        .clone(),
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap();
    let doses = pin_f64s(&pin, &["contract", "dose_grid"]);
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from(doses)),
        ),
    })
    .with_temporal(temporal_spec(&pin))
    .with_observation(
        ObservationSpec::Selected {
            latent: VariableId::from_raw(1),
            observed: VariableId::from_raw(1),
            indicator: VariableId::from_raw(2),
        },
        [ObservationAssumption::OutcomeIndependentGiven(Arc::from([VariableId::from_raw(0)]))],
    );
    let built = Study::series(observed.clone())
        .graph(graph)
        .query(CausalQuery::Response(query))
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let inspected = built.inspect().unwrap();
    assert_eq!(inspected.support_status.unwrap().as_str(), "licensed");
    match &inspected.reasoning.assumptions {
        SlotAvailability::Available(slot) => {
            assert!(slot.obligations.iter().any(|obligation| {
                obligation.description.as_ref() == "outcome_independent_given"
            }));
        }
        other => panic!("inspect must publish temporal observation obligations, got {other:?}"),
    }
    let (result, consumed) = consume_licensed_family(
        &built,
        |prepared| prepared.estimate_series(&observed, &ctx).unwrap(),
        "tresp-obs",
        &ctx,
    );
    assert!(result.response.is_some());
    let observation = consumed
        .contract
        .as_ref()
        .and_then(|section| section.observation.as_ref())
        .expect("consumed temporal observation identity");
    assert!(
        observation.observation.iter().any(|tag| tag.starts_with("outcome_independent_given:")),
        "observation tags keep their conditioning variables: {:?}",
        observation.observation
    );
}

fn pulse_prior_bytes(
    series: &TimeSeriesData,
    graph: &TemporalDag,
    ctx: &ExecutionContext,
) -> Vec<u8> {
    let source = Study::series(series.clone())
        .graph(graph.clone())
        .temporal_query(pulse_query())
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(32)))
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(ctx)
        .unwrap()
        .estimate_series(series, ctx)
        .unwrap();
    antecedent::io::encode_causal_posterior_bytes(
        source.posterior.as_ref().expect("source posterior"),
        "source",
    )
    .unwrap()
}

fn consume_mapped_prior(
    series: &TimeSeriesData,
    graph: TemporalDag,
    mapping: PriorMapping,
    artifact_id: &str,
) {
    let ctx = ExecutionContext::for_tests(1);
    let expected = mapping.clone();
    let bytes = pulse_prior_bytes(series, &graph, &ctx);
    let isotropic = Study::series(series.clone())
        .graph(graph.clone())
        .temporal_query(pulse_query())
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(32)))
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let transferred = Study::series(series.clone())
        .graph(graph)
        .temporal_query(pulse_query())
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(32).prior_from_artifact(bytes, Some(mapping)),
        ))
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let iso = isotropic.prepare(&ctx).unwrap().contract().unwrap();
    let xfer = transferred.prepare(&ctx).unwrap().contract().unwrap();
    assert_eq!(iso.identities.identification, xfer.identities.identification);
    assert_ne!(iso.identities.inference_binding, xfer.identities.inference_binding);
    let (_, consumed) = consume_licensed_family(
        &transferred,
        |prepared| prepared.estimate_series(series, &ctx).unwrap(),
        artifact_id,
        &ctx,
    );
    let binding = consumed
        .contract
        .as_ref()
        .and_then(|section| section.inference_binding.as_ref())
        .expect("consumed inference binding");
    assert_eq!(
        binding.bayesian.as_ref().and_then(|bayes| bayes.prior_mapping.clone()),
        Some(antecedent_io::PriorMappingIdentityWire::from_mapping(&expected)),
        "the mapping is bound structurally, not as a joined string"
    );
}

#[test]
fn licensed_family_prior_mapping_identical_subspace_does_not_create_identification() {
    let (series, graph) = lag1_series();
    consume_mapped_prior(
        &series,
        graph,
        PriorMapping::IdenticalCoefficientSubspace,
        "pulse-map-ident",
    );
}

#[test]
fn licensed_family_prior_mapping_effect_functional_does_not_create_identification() {
    let (series, graph) = lag1_series();
    consume_mapped_prior(
        &series,
        graph,
        PriorMapping::EffectFunctional { source_quantity: "ate".into() },
        "pulse-map-fn",
    );
}

#[test]
fn licensed_family_prior_mapping_named_parameters_does_not_create_identification() {
    let (series, graph) = lag1_series();
    consume_mapped_prior(
        &series,
        graph,
        PriorMapping::NamedParameters { pairs: vec![("coef_x@lag1".into(), "coef_x@lag1".into())] },
        "pulse-map-named",
    );
}

fn right_censored_temporal_curve() -> (TimeSeriesData, TemporalDag, ResponseQuery) {
    let n = 240usize;
    let mut t = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut c = Vec::with_capacity(n);
    let mut event = Vec::with_capacity(n);
    for i in 0..n {
        let tv = ((i as f64) * 0.04).sin();
        t.push(tv);
        let t1 = if i >= 1 { t[i - 1] } else { 0.0 };
        let latent = 5.0 + 2.0 * t1 + 0.1 * ((i as f64) * 0.17).sin();
        let censor = if i % 5 == 0 { latent - 0.25 } else { latent + 4.0 };
        y.push(latent.min(censor));
        c.push(censor);
        event.push(if latent <= censor { 1.0 } else { 0.0 });
    }
    let series = TimeSeriesData::from_f64_columns(
        [
            ("t", t.as_slice()),
            ("y", y.as_slice()),
            ("c", c.as_slice()),
            ("event", event.as_slice()),
        ],
        1,
    )
    .unwrap();
    let mut graph = TemporalDag::empty();
    let t1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(t1, y0).unwrap();
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from([-0.5, 0.0, 0.5])),
        ),
    })
    .with_temporal(TemporalResponseSpec::new(vec![1], TemporalPolicy::pulse(-1), None).unwrap())
    .with_observation(
        ObservationSpec::RightCensored {
            latent: VariableId::from_raw(1),
            observed: VariableId::from_raw(1),
            censoring: VariableId::from_raw(2),
            event: VariableId::from_raw(3),
        },
        [ObservationAssumption::IndependentGiven(Arc::from([]))],
    );
    (series, graph, query)
}

fn inspect_observation_pair(
    spec: ObservationSpec,
    assumption: ObservationAssumption,
    spec_tag: &str,
    assumption_tag: &str,
) -> (Study, TimeSeriesData) {
    let n = 80usize;
    let t: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.04).sin()).collect();
    let y: Vec<f64> = (0..n).map(|i| 1.0 + if i > 0 { t[i - 1] } else { 0.0 }).collect();
    let aux = vec![0.5; n];
    let series = TimeSeriesData::from_f64_columns(
        [("t", t.as_slice()), ("y", y.as_slice()), ("lo", aux.as_slice()), ("hi", aux.as_slice())],
        1,
    )
    .unwrap();
    let mut graph = TemporalDag::empty();
    let t1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(t1, y0).unwrap();
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from([-0.5, 0.5])),
        ),
    })
    .with_temporal(TemporalResponseSpec::new(vec![1], TemporalPolicy::pulse(-1), None).unwrap())
    .with_observation(spec, [assumption]);
    let built = Study::series(series.clone())
        .graph(graph)
        .query(CausalQuery::Response(query))
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let inspected = built.inspect().unwrap();
    match &inspected.reasoning.assumptions {
        SlotAvailability::Available(slot) => {
            assert!(
                slot.obligations.iter().any(|obligation| {
                    obligation.description.as_ref() == spec_tag
                        && obligation.kind == ObligationKind::UserAssertion
                        && obligation.status == AssumptionStatus::Declared
                }),
                "observation spec must enter as a declared obligation, got {:?}",
                slot.obligations
            );
            assert!(
                slot.obligations.iter().any(|obligation| {
                    obligation.description.as_ref() == assumption_tag
                        && obligation.kind == ObligationKind::UserAssertion
                        && obligation.status == AssumptionStatus::Declared
                }),
                "observation assumption must enter as a declared user assertion"
            );
        }
        other => panic!("inspect must publish observation obligations, got {other:?}"),
    }
    assert!(matches!(inspected.reasoning.identification, SlotAvailability::Unavailable { .. }));
    (built, series)
}

#[test]
fn licensed_family_response_right_censored_observation_consumes() {
    let ctx = ExecutionContext::for_tests(21);
    let (series, graph, query) = right_censored_temporal_curve();
    let built = Study::series(series.clone())
        .graph(graph)
        .query(CausalQuery::Response(query))
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let inspected = built.inspect().unwrap();
    match &inspected.reasoning.assumptions {
        SlotAvailability::Available(slot) => {
            assert!(slot.obligations.iter().any(|obligation| {
                obligation.description.as_ref() == "right_censored"
                    && obligation.kind == ObligationKind::UserAssertion
                    && obligation.status == AssumptionStatus::Declared
            }));
            assert!(
                slot.obligations
                    .iter()
                    .any(|obligation| { obligation.description.as_ref() == "independent_given" })
            );
        }
        other => panic!("inspect must publish right-censored obligations, got {other:?}"),
    }
    let (result, consumed) = consume_licensed_family(
        &built,
        |prepared| prepared.estimate_series(&series, &ctx).unwrap(),
        "tresp-rc",
        &ctx,
    );
    assert!(result.response.is_some());
    let observation = consumed
        .contract
        .as_ref()
        .and_then(|section| section.observation.as_ref())
        .expect("consumed right-censored observation identity");
    assert!(
        observation.observation.iter().any(|tag| tag.starts_with("right_censored:latent=")),
        "censoring bindings stay in the observation identity: {:?}",
        observation.observation
    );
    assert!(observation.observation.iter().any(|tag| tag.starts_with("independent_given:")));
}

#[test]
fn licensed_family_response_interval_censored_observation_inspects() {
    let (built, series) = inspect_observation_pair(
        ObservationSpec::IntervalCensored {
            latent: VariableId::from_raw(1),
            lower: VariableId::from_raw(2),
            upper: VariableId::from_raw(3),
        },
        ObservationAssumption::IndependentGiven(Arc::from([])),
        "interval_censored",
        "independent_given",
    );
    let ctx = ExecutionContext::for_tests(21);
    let err = built
        .prepare(&ctx)
        .and_then(|prepared| prepared.estimate_series(&series, &ctx))
        .unwrap_err();
    assert!(
        err.to_string().contains(TEMPORAL_OBSERVATION_UNLICENSED)
            || err.to_string().contains("interval")
            || matches!(err, antecedent::CausalError::Unsupported { .. }),
        "IntervalCensored must refuse without treating bounds as MAR, got {err}"
    );
}

#[test]
fn licensed_family_response_truncated_observation_inspects() {
    let (built, series) = inspect_observation_pair(
        ObservationSpec::Truncated {
            latent: VariableId::from_raw(1),
            observed: VariableId::from_raw(1),
            lower: Some(VariableId::from_raw(2)),
            upper: Some(VariableId::from_raw(3)),
        },
        ObservationAssumption::IndependentGiven(Arc::from([])),
        "truncated",
        "independent_given",
    );
    let ctx = ExecutionContext::for_tests(21);
    let err = built
        .prepare(&ctx)
        .and_then(|prepared| prepared.estimate_series(&series, &ctx))
        .unwrap_err();
    assert!(
        err.to_string().contains(TEMPORAL_OBSERVATION_UNLICENSED)
            || err.to_string().contains("truncat")
            || matches!(err, antecedent::CausalError::Unsupported { .. }),
        "Truncated must refuse without treating bounds as MAR, got {err}"
    );
}

#[test]
fn licensed_family_prior_catalog_does_not_create_identification() {
    let ctx = ExecutionContext::for_tests(1);
    let (series, graph) = lag1_series();
    let bytes = pulse_prior_bytes(&series, &graph, &ctx);
    let catalog = PriorCatalog::from_sources(vec![PriorSourceRef::with_bytes(
        PriorSourceMeta::new(
            "catalog-source",
            EstimandFingerprint::new("pulse", "x", "y")
                .with_temporal(TemporalCoordinates::new([1], 1)),
            "NonparametricallyIdentified",
        )
        .with_design(vec![
            DesignVariableSummary::new("x", DesignVariableRole::Treatment),
            DesignVariableSummary::new("y", DesignVariableRole::Outcome),
        ]),
        bytes.clone(),
    )]);
    let fingerprint =
        EstimandFingerprint::new("pulse", "x", "y").with_temporal(TemporalCoordinates::new([1], 1));
    let target = TargetDesign::new(fingerprint, ["x", "y"]);
    let reports = catalog.filter_compatible(&target);
    assert!(reports[0].is_usable(), "catalog must select the named source, got {:?}", reports[0]);
    let chosen = catalog.require_usable(&target).expect("PriorCatalog.filter_compatible");
    assert_eq!(chosen.meta.artifact_id, "catalog-source");
    let isotropic = Study::series(series.clone())
        .graph(graph.clone())
        .temporal_query(pulse_query())
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(32)))
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let transferred = Study::series(series.clone())
        .graph(graph)
        .temporal_query(pulse_query())
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(32).prior_from_artifact(bytes, None),
        ))
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let inspected = transferred.inspect().unwrap();
    assert_eq!(inspected.support_status.unwrap().as_str(), "licensed");
    assert!(matches!(inspected.reasoning.identification, SlotAvailability::Unavailable { .. }));
    let iso = isotropic.prepare(&ctx).unwrap().contract().unwrap();
    let xfer = transferred.prepare(&ctx).unwrap().contract().unwrap();
    assert_eq!(iso.identities.identification, xfer.identities.identification);
    assert_ne!(iso.identities.inference_binding, xfer.identities.inference_binding);
    let (_, consumed) = consume_licensed_family(
        &transferred,
        |prepared| prepared.estimate_series(&series, &ctx).unwrap(),
        "pulse-catalog",
        &ctx,
    );
    let binding = consumed
        .contract
        .as_ref()
        .and_then(|section| section.inference_binding.as_ref())
        .expect("consumed catalog inference binding");
    assert!(
        binding
            .bayesian
            .as_ref()
            .is_some_and(|bayes| bayes.prior_artifact.is_some() && bayes.prior.is_none()),
        "a transferred prior binds the artifact bytes it hydrates from"
    );
    let slot = consumed
        .contract
        .as_ref()
        .and_then(|section| section.reasoning.identification.value.as_ref())
        .expect("consumed identification slot");
    assert_eq!(slot.unidentified_mass, 0.0);
}

fn xy_series_len(n: usize) -> TimeSeriesData {
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; n];
    for t in 1..n {
        x[t] = (t as f64 * 0.05).sin();
        y[t] = 0.5 * x[t - 1];
    }
    TimeSeriesData::try_new(
        TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())])
            .unwrap()
            .storage()
            .clone(),
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap()
}

fn cheap_response_curve() -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from(vec![0.0_f64, 1.0])),
        ),
    })
    .with_temporal(TemporalResponseSpec::new(vec![1], TemporalPolicy::pulse(-1), None).unwrap())
}

#[test]
fn failed_series_refresh_then_estimate_on_old_data() {
    let (series, graph) = lag1_series();
    let ctx = ExecutionContext::for_tests(1);
    let mut prepared = Study::series(series.clone())
        .graph(graph)
        .temporal_query(pulse_query())
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let before = prepared.contract().unwrap().identities;
    let first = prepared.estimate_series(&series, &ctx).unwrap();
    assert!((first.effect() - 0.5).abs() < 0.15);
    assert!(prepared.refresh_series(xy_series_len(2), &ctx).is_err());
    assert_eq!(before, prepared.contract().unwrap().identities);
    let recovered = prepared.estimate_series(&series, &ctx).unwrap();
    assert!((recovered.effect() - first.effect()).abs() < 1e-12);
    assert_eq!(before, prepared.contract().unwrap().identities);
}

#[test]
#[allow(clippy::too_many_lines)]
fn temporal_state_events_keep_lineage_and_refuse_stale_publish() {
    let ctx = ExecutionContext::for_tests(1);
    let pulse_series = xy_series_len(60);
    let curve_series = xy_series_len(60);
    let (_, pulse_graph) = lag1_series();
    let curve_query = cheap_response_curve();
    let mut pulse = Study::series(pulse_series.clone())
        .graph_posterior(lag1_gp_posterior())
        .temporal_query(pulse_query())
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let mut curve = Study::series(curve_series.clone())
        .graph(pulse_graph.clone())
        .query(CausalQuery::Response(curve_query.clone()))
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let pulse_id = pulse.contract().unwrap().identities;
    let curve_id = curve.contract().unwrap().identities;
    let pulse_est = pulse.estimate_series(&pulse_series, &ctx).unwrap();
    let curve_est = curve.estimate_series(&curve_series, &ctx).unwrap();
    assert!(pulse_est.effect().is_finite());
    assert!(curve_est.response.as_ref().is_some());

    let mut state = new_antecedent_state(CacheBudget::new(1 << 20));
    apply_state_event(
        &mut state,
        antecedent::state::StateEvent::AppendData(DataBatchRef {
            id: Arc::from("t0"),
            nrows: 60,
            bytes: 960,
        }),
    )
    .unwrap();
    let pulse_q = state.queries.register(CausalQuery::TemporalEffect(pulse_query()));
    let curve_q = state.queries.register(CausalQuery::Response(curve_query));
    let v0 = state.version;
    publish_recomputed_results(
        &mut state,
        v0,
        &[(pulse_q, &pulse_id, 32), (curve_q, &curve_id, 32)],
    )
    .unwrap();
    assert!(!state.is_stale(pulse_q) && !state.is_stale(curve_q));

    apply_state_event(
        &mut state,
        antecedent::state::StateEvent::AppendData(DataBatchRef {
            id: Arc::from("t1"),
            nrows: 20,
            bytes: 320,
        }),
    )
    .unwrap();
    assert!(state.is_stale(pulse_q) && state.is_stale(curve_q));
    let stale_publish =
        publish_recomputed_results(&mut state, v0, &[(pulse_q, &pulse_id, 32)]).unwrap_err();
    assert!(stale_publish.to_string().contains("stale commit"));
    assert!(state.is_stale(pulse_q));

    assert!(pulse.refresh_series(xy_series_len(2), &ctx).is_err());
    assert_eq!(pulse.contract().unwrap().identities, pulse_id);
    assert!(state.is_stale(pulse_q), "failed recompute must not publish freshness");

    let longer = xy_series_len(80);
    let pulse_refresh = pulse.refresh_series(longer.clone(), &ctx).unwrap();
    let curve_refresh = curve.refresh_series(longer.clone(), &ctx).unwrap();
    let pulse_after = pulse.contract().unwrap().identities;
    let curve_after = curve.contract().unwrap().identities;
    assert_eq!(pulse_id.identification, pulse_after.identification);
    assert_eq!(curve_id.identification, curve_after.identification);
    assert_ne!(pulse_id.data_snapshot, pulse_after.data_snapshot);
    assert_ne!(curve_id.data_snapshot, curve_after.data_snapshot);
    let pulse_again = pulse.estimate_series(&longer, &ctx).unwrap();
    assert!((pulse_again.effect() - pulse_refresh.effect()).abs() < 1e-12);
    assert!(curve_refresh.response.as_ref().is_some());

    let v1 = state.version;
    publish_recomputed_results(
        &mut state,
        v1,
        &[(pulse_q, &pulse_after, 32), (curve_q, &curve_after, 32)],
    )
    .unwrap();
    assert!(!state.is_stale(pulse_q));
    assert_ne!(result_lineage_fingerprint(&pulse_id), result_lineage_fingerprint(&pulse_after));
    // The lineage digest covers every layer, not a 32-bit prefix of two.
    let att = antecedent_core::ContractIdentities::new(
        SemanticDigest::from_bytes([9; 32]),
        pulse_after.identification,
        pulse_after.identification_product,
        pulse_after.program,
        pulse_after.inference_binding,
        pulse_after.observation,
        pulse_after.data_snapshot,
    );
    assert_ne!(result_lineage_fingerprint(&pulse_after), result_lineage_fingerprint(&att));
    let rebound = antecedent_core::ContractIdentities::new(
        pulse_after.target,
        pulse_after.identification,
        pulse_after.identification_product,
        pulse_after.program,
        SemanticDigest::from_bytes([8; 32]),
        pulse_after.observation,
        pulse_after.data_snapshot,
    );
    assert_ne!(result_lineage_fingerprint(&pulse_after), result_lineage_fingerprint(&rebound));

    let next_data = state.data_catalog.version.next();
    apply_state_event(&mut state, antecedent::state::StateEvent::ReplaceData(next_data)).unwrap();
    assert!(state.is_stale(pulse_q) && state.is_stale(curve_q));
    let after_replace = state.version;
    // Republishing the snapshot recorded before the data event is not a
    // recomputation, whatever the caller passes.
    let not_recomputed = publish_recomputed_results(
        &mut state,
        after_replace,
        &[(pulse_q, &pulse_after, 32), (curve_q, &curve_after, 32)],
    )
    .unwrap_err();
    assert!(format!("{not_recomputed}").contains("before the latest data event"));
    assert!(state.is_stale(pulse_q) && state.is_stale(curve_q));
    let recomputed = pulse.refresh_series(xy_series_len(90), &ctx).unwrap();
    assert!(recomputed.effect().is_finite());
    let pulse_recomputed = pulse.contract().unwrap().identities;
    publish_recomputed_results(&mut state, after_replace, &[(pulse_q, &pulse_recomputed, 32)])
        .unwrap();
    assert!(!state.is_stale(pulse_q));

    apply_state_event(
        &mut state,
        antecedent::state::StateEvent::RecordIntervention(InterventionRecord {
            id: Arc::from("do-x"),
            fingerprint: 1,
        }),
    )
    .unwrap();
    assert!(state.is_stale(pulse_q) && state.is_stale(curve_q));
    apply_state_event(
        &mut state,
        antecedent::state::StateEvent::UpdateAssumption(AssumptionRecord {
            assumption: Assumption::Stationarity,
            source: AssumptionSource::UserDeclared,
            scope: AssumptionScope::Global,
            status: AssumptionStatus::Declared,
        }),
    )
    .unwrap();
    assert!(state.is_stale(pulse_q) && state.is_stale(curve_q));
    assert_eq!(
        pulse.contract().unwrap().identities.data_snapshot,
        pulse_recomputed.data_snapshot,
        "historical prepared lineage stays on the last successful refresh"
    );
}

fn rewrite_contract_section(
    body: &antecedent_io::AnalysisResultWire,
    names: Vec<String>,
    artifact_id: &str,
    contract: &antecedent_io::AnalysisResultContractWire,
) -> Vec<u8> {
    let mut artifact =
        antecedent_io::encode_analysis_result_artifact(body, names, artifact_id).unwrap();
    let (descriptor, packed) = antecedent_io::pack_section_shared(
        antecedent_io::CONTRACT_SECTION,
        "application/cbor",
        antecedent_io::to_cbor(contract).unwrap().into(),
        antecedent_io::CompressPolicy::Auto,
    );
    artifact.manifest.sections.push(descriptor);
    artifact.sections.push(packed);
    let mut bytes = Vec::new();
    artifact.write_to(&mut bytes).unwrap();
    bytes
}

#[test]
fn claim_handoff_sender_restricted_forward_preserves_losses() {
    let ctx = ExecutionContext::for_tests(1);
    let (data, dag, query) = confounded_scm(80, 41);
    let prepared = study(data.clone(), dag, query).prepare(&ctx).unwrap();
    let contract = prepared.contract().unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let claim = result.claim(&contract, &ctx).unwrap();
    assert_eq!(claim.kind, ClaimKind::Point);
    assert_eq!(claim.domain(ClaimDomainAxis::Identification), DomainStatus::Identified);
    assert_eq!(claim.domain(ClaimDomainAxis::Evaluated), DomainStatus::Evaluated);
    assert!(claim.evidence.iter().any(|item| item.starts_with("snapshot:")));
    assert_eq!(HostOperation::AcceptClaim.as_str(), "accept_claim");

    let bytes = prepared.encode_contracted_result(&result, "claim-handoff", &ctx).unwrap();
    let (sender, lossless) = accept_claim(&bytes, &ConsumerProfile::full()).unwrap();
    assert!(!sender.acceptance.accepts_as_verified_program());
    assert_eq!(
        sender.acceptance.unresolved.as_ref(),
        &[std::sync::Arc::<str>::from("dependencies.linear_fit_sufficient_statistics")]
    );
    assert_eq!(lossless.input, claim.claim_id);
    assert_eq!(lossless.output, None);
    assert_eq!(
        lossless.unresolved.as_ref(),
        &[std::sync::Arc::<str>::from("dependencies.linear_fit_sufficient_statistics")]
    );
    assert!(!lossless.equivalent_claim());
    let host = project_claim_host(&sender);
    assert!(!host.accepts_as_claim);
    assert!(
        host.value.is_number(),
        "the readable point remains present despite the replay refusal"
    );
    assert_eq!(host.identification_domain, serde_json::Value::Null);

    let (forwarding, stored) = accept_claim(&bytes, &ConsumerProfile::forwarding()).unwrap();
    assert!(!forwarding.acceptance.accepts_as_claim());
    assert!(forwarding.acceptance.supported_operations.iter().any(|op| &**op == "forward"));
    assert!(!stored.equivalent_claim());

    let (decoded, header, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    let names = header.variable_names.clone();
    let mut section =
        antecedent_io::decode_analysis_result_contract(&decoded).unwrap().expect("contract");
    section.format = 99;
    let unknown = rewrite_contract_section(&body, names.clone(), "unknown", &section);
    let (restricted, unknown_receipt) =
        accept_claim(&unknown, &ConsumerProfile::restricted()).unwrap();
    assert!(!restricted.acceptance.recognized);
    assert!(!restricted.acceptance.accepts_as_claim());
    assert!(restricted.acceptance.supported_operations.iter().any(|op| &**op == "forward"));

    let (forwarded, forward_receipt) =
        accept_claim(&unknown, &ConsumerProfile::forwarding()).unwrap();
    assert!(!forwarded.acceptance.accepts_as_claim());
    let chained = unknown_receipt.chain(&forward_receipt);
    assert!(!chained.equivalent_claim());
    let reserialized = chained.chain(&antecedent_core::HandoffReceipt::lossless(
        chained.output.unwrap_or(claim.claim_id),
        "reserialize",
    ));
    assert!(!reserialized.equivalent_claim());
    assert!(reserialized.omitted.iter().any(|field| &**field == "required_semantics"));

    let mut missing_contract =
        antecedent_io::decode_analysis_result_contract(&decoded).unwrap().expect("contract");
    missing_contract.identification = None;
    let missing_bytes = rewrite_contract_section(&body, names, "missing", &missing_contract);
    let (missing, missing_receipt) =
        accept_claim(&missing_bytes, &ConsumerProfile::full()).unwrap();
    assert!(!missing.acceptance.accepts_as_claim());
    assert!(missing.acceptance.unresolved.iter().any(|item| item.starts_with("identities.")));
    assert!(!missing_receipt.equivalent_claim());

    let (scalar, lossy) = project_lossy_scalar(&sender);
    assert!(!scalar.accepts_as_claim);
    assert!(!lossy.equivalent_claim());
    let forwarded_scalar = lossy.chain(&stored);
    assert!(!forwarded_scalar.equivalent_claim());

    let offline = claim.freshness(Some(claim.identities.data_snapshot), None, None, false, false);
    assert!(offline.historically_valid);
    assert_eq!(offline.currently_applicable, None);
    assert_eq!(offline.as_of_snapshot, Some(claim.identities.data_snapshot));
    assert!(!offline.locally_ready);
    let superseded = claim.freshness(
        Some(claim.identities.data_snapshot),
        Some(claim.identities.data_snapshot),
        Some(claim.claim_id),
        false,
        true,
    );
    assert_eq!(superseded.currently_applicable, Some(false));
    assert_eq!(superseded.superseded_by, Some(claim.claim_id));
}

#[test]
fn derived_claim_compare_is_not_synthesis() {
    let ctx_a = ExecutionContext::for_tests(1);
    let ctx_b = ExecutionContext::for_tests(2);
    let (data, dag, query) = confounded_scm(80, 41);
    let prepared = study(data.clone(), dag, query).prepare(&ctx_a).unwrap();
    let contract = prepared.contract().unwrap();
    let result_a = prepared.estimate(&data, &ctx_a).unwrap();
    let result_b = prepared.estimate(&data, &ctx_b).unwrap();
    let left = result_a.claim(&contract, &ctx_a).unwrap();
    let right = result_b.claim(&contract, &ctx_b).unwrap();
    assert_ne!(left.claim_id, right.claim_id);
    assert!(left.evidence_ref().same_source(&right.evidence_ref()));
    assert!(SharedEvidenceRef::forwarded_duplicate(left.claim_id, left.claim_id));
    assert!(!SharedEvidenceRef::forwarded_duplicate(left.claim_id, right.claim_id));

    match compose_claims(&[&left, &right], ClaimOperation::Compare, None, None) {
        DerivedClaimOutcome::Comparison(report) => {
            assert!(report.compatible);
            assert!(!report.operation.synthesizes());
        }
        other => panic!("expected comparison, got {other:?}"),
    }
    match compose_claims(&[&left, &right], ClaimOperation::Pool, None, None) {
        DerivedClaimOutcome::Refused { restriction, parents, .. } => {
            assert_eq!(&*restriction, "unlicensed_synthesis");
            assert_eq!(parents.as_ref(), [left.claim_id, right.claim_id]);
        }
        other => panic!("expected pool refusal, got {other:?}"),
    }
    match compose_claims(&[&left, &right], ClaimOperation::Contrast, None, None) {
        DerivedClaimOutcome::Refused { restriction, .. } => {
            assert_eq!(&*restriction, "marginal_intervals_do_not_determine_contrast");
        }
        other => panic!("expected contrast refusal, got {other:?}"),
    }
}

fn tiny_ranker() -> antecedent::design::DesignRanker {
    use antecedent::design::{DesignRankConfig, DesignRanker};
    DesignRanker::new().with_config(DesignRankConfig {
        min_batches: 1,
        max_batches: 2,
        batch_size: 2,
        rank_uncertainty_threshold: 0.5,
    })
}

fn empty_eval(
    graphs: &antecedent_prob::WeightedGraphSamples,
) -> antecedent::design::DesignEvaluationContext<'_, (), ()> {
    antecedent::design::DesignEvaluationContext {
        graphs,
        effect_width: None,
        model_loglik: None,
        decisions: None,
        query_id_unlock: None,
        env_id_unlock: None,
        identified_under_intervention: None,
        graph_features: None,
    }
}

#[test]
fn design_rank_pulse_bayesian_pins_synthetic_width_order() {
    use antecedent::design::{
        CandidateDesign, DesignCost, DesignObjective, EffectWidthContext, SamplingPlan,
    };

    let ctx = ExecutionContext::for_tests(1);
    let (series, graph) = lag1_series();
    let prepared = Study::series(series)
        .graph(graph)
        .temporal_query(pulse_query())
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(32)))
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let width = EffectWidthContext {
        xtx: Arc::from([1.0]),
        sigma2: 1.0,
        treatment_col: 0,
        n: 16,
        measure_columns: None,
        intervention_design: None,
        environment_grams: None,
    };
    let graphs = antecedent_prob::WeightedGraphSamples::new(
        vec![1.0],
        vec![antecedent_prob::GraphIdentFlag::Identified],
        vec![1],
    )
    .unwrap();
    let eval = antecedent::design::DesignEvaluationContext::<(), ()> {
        graphs: &graphs,
        effect_width: Some(&width),
        ..empty_eval(&graphs)
    };
    let candidates = vec![
        CandidateDesign::IncreaseSamplingRate(SamplingPlan {
            additional_samples: 4,
            cost: DesignCost::zero(),
            tag: 1,
        }),
        CandidateDesign::IncreaseSamplingRate(SamplingPlan {
            additional_samples: 64,
            cost: DesignCost::zero(),
            tag: 2,
        }),
        CandidateDesign::Measure(antecedent::design::MeasurementPlan {
            variables: Arc::from([VariableId::from_raw(0)]),
            cost: DesignCost::zero(),
            tag: 3,
        }),
    ];
    let preview = prepared
        .preview_rank_designs(
            &DesignObjective::ReduceEffectPosteriorWidth { query: QueryId::from_raw(0) },
            &candidates,
            &eval,
            None,
        )
        .unwrap();
    assert!(!preview.unresolved);
    assert_eq!(preview.violations.len(), 1);
    assert_eq!(preview.violations[0].candidate_index, 2);
    assert_eq!(preview.violations[0].constraint.as_ref(), "unlicensed_candidate");
    assert!(preview.obligations.iter().any(|item| item.id.as_ref() == "design.update_rule"));

    let ranking = prepared
        .rank_designs(
            &tiny_ranker(),
            &DesignObjective::ReduceEffectPosteriorWidth { query: QueryId::from_raw(0) },
            &candidates,
            &eval,
            &ctx,
        )
        .unwrap();
    assert_eq!(ranking.ranked.len(), 2);
    assert_eq!(ranking.ranked[0].candidate_index, 1);
    assert!(ranking.ranked[0].score > ranking.ranked[1].score);
    assert_eq!(ranking.violations.len(), 1);
    assert_eq!(ranking.violations[0].candidate_index, 2);
}

/// `ReduceDecisionRegret` through a prepared study scores the conjugate-normal
/// expected value of sample information of each candidate's sample size, and
/// records a candidate without a sample size as unlicensed instead of scoring it.
#[test]
fn design_rank_decision_regret_scores_value_of_sample_information() {
    use antecedent::design::{
        AffineUtility, CandidateDesign, DecisionPrior, DecisionProblem, DecisionProblemId,
        DecisionRegistry, DesignCost, DesignObjective, GaussianMeanSignal, SamplingPlan,
        ScoreEvaluation,
    };

    let ctx = ExecutionContext::for_tests(1);
    let (series, graph) = lag1_series();
    let prepared = Study::series(series)
        .graph(graph)
        .temporal_query(pulse_query())
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(32)))
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    // Keep (U = 1 + 0.2θ) or switch (U = −0.5 + 1.1θ); θ ~ N(1, 2), noise variance 3.
    let utility = AffineUtility::new(vec![1.0, -0.5], vec![0.2, 1.1]).unwrap();
    let registry = DecisionRegistry {
        problems: vec![Some(DecisionProblem::new(vec![0_usize, 1], Arc::new(utility), vec![]))],
        prior: DecisionPrior::Normal { mean: 1.0, variance: 2.0 },
        signal: Arc::new(GaussianMeanSignal::new(3.0).unwrap()),
    };
    let graphs = antecedent_prob::WeightedGraphSamples::new(
        vec![1.0],
        vec![antecedent_prob::GraphIdentFlag::Identified],
        vec![1],
    )
    .unwrap();
    let eval = antecedent::design::DesignEvaluationContext {
        graphs: &graphs,
        effect_width: None,
        model_loglik: None,
        decisions: Some(&registry),
        query_id_unlock: None,
        env_id_unlock: None,
        identified_under_intervention: None,
        graph_features: None,
    };
    let sampling = |n: u64| {
        CandidateDesign::IncreaseSamplingRate(SamplingPlan {
            additional_samples: n,
            cost: DesignCost::zero(),
            tag: n,
        })
    };
    let candidates = vec![
        sampling(4),
        sampling(64),
        CandidateDesign::Measure(antecedent::design::MeasurementPlan {
            variables: Arc::from([VariableId::from_raw(0)]),
            cost: DesignCost::zero(),
            tag: 3,
        }),
    ];
    let objective =
        DesignObjective::ReduceDecisionRegret { decision: DecisionProblemId::from_raw(0) };
    let preview = prepared.preview_rank_designs(&objective, &candidates, &eval, None).unwrap();
    assert!(!preview.unresolved);
    assert_eq!(preview.violations.len(), 1);
    assert_eq!(preview.violations[0].candidate_index, 2);
    assert_eq!(preview.violations[0].constraint.as_ref(), "unlicensed_candidate");

    let ranking =
        prepared.rank_designs(&tiny_ranker(), &objective, &candidates, &eval, &ctx).unwrap();
    assert_eq!(ranking.ranked.len(), 2);
    assert_eq!(ranking.ranked[0].candidate_index, 1);
    assert_eq!(ranking.violations.len(), 1);
    // EVSI(n) = |Δβ| s_n G(|μ_b − μ₀|/s_n), s_n² = τ⁴n/(nτ² + σ²), G(u) = φ(u) − u(1 − Φ(u)).
    for ranked in ranking.ranked.iter() {
        let n = [4.0, 64.0][ranked.candidate_index];
        let s = (4.0 * n / (2.0 * n + 3.0_f64)).sqrt();
        let u = (1.5_f64 / 0.9 - 1.0).abs() / s;
        let g = antecedent_kernels::norm_pdf(u) - u * antecedent_kernels::norm_sf(u);
        assert!((ranked.score - 0.9 * s * g).abs() < 1e-14, "{ranked:?}");
        assert_eq!(ranked.evaluation, ScoreEvaluation::Exact);
    }
}

#[test]
fn design_rank_pulse_window_refuses_incomparable_width() {
    use antecedent::design::{CandidateDesign, DesignCost, DesignObjective, SamplingPlan};

    let ctx = ExecutionContext::for_tests(1);
    let (series, graph) = lag1_series();
    let prepared = Study::series(series)
        .graph(graph)
        .temporal_query(pulse_query())
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(32)))
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let graphs = antecedent_prob::WeightedGraphSamples::new(
        vec![1.0],
        vec![antecedent_prob::GraphIdentFlag::Identified],
        vec![1],
    )
    .unwrap();
    let eval = empty_eval(&graphs);
    let candidates = vec![CandidateDesign::IncreaseSamplingRate(SamplingPlan {
        additional_samples: 8,
        cost: DesignCost::zero(),
        tag: 1,
    })];
    let other = SemanticDigest::from_bytes([7; 32]);
    let err = prepared
        .rank_designs_bound(
            &tiny_ranker(),
            &DesignObjective::ReduceEffectPosteriorWidth { query: QueryId::from_raw(0) },
            &candidates,
            &eval,
            &ctx,
            Some(other),
        )
        .unwrap_err();
    assert!(err.to_string().contains("incomparable estimands"), "{err}");
}

#[test]
fn design_rank_width_without_model_is_unresolved() {
    use antecedent::design::{CandidateDesign, DesignCost, DesignObjective, SamplingPlan};

    let ctx = ExecutionContext::for_tests(1);
    let (series, graph) = lag1_series();
    let prepared = Study::series(series)
        .graph(graph)
        .temporal_query(
            TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
                .with_policy(TemporalPolicy::sustained(-1, -1))
                .with_horizon_steps(1),
        )
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(32)))
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let graphs = antecedent_prob::WeightedGraphSamples::new(
        vec![1.0],
        vec![antecedent_prob::GraphIdentFlag::Identified],
        vec![1],
    )
    .unwrap();
    let eval = empty_eval(&graphs);
    let candidates = vec![CandidateDesign::IncreaseSamplingRate(SamplingPlan {
        additional_samples: 8,
        cost: DesignCost::zero(),
        tag: 1,
    })];
    let ranking = prepared
        .rank_designs(
            &tiny_ranker(),
            &DesignObjective::ReduceEffectPosteriorWidth { query: QueryId::from_raw(0) },
            &candidates,
            &eval,
            &ctx,
        )
        .unwrap();
    assert!(ranking.ranked.is_empty());
    assert_eq!(ranking.violations[0].constraint.as_ref(), "unresolved_utility");
    assert_eq!(ranking.violations[0].detail.as_ref(), "missing_effect_width_model");
}

#[test]
fn design_rank_graph_posterior_ate_preserves_unidentified_mass() {
    use antecedent::design::{
        CandidateDesign, DesignCost, DesignObjective, MeasurementPlan, SamplingPlan,
    };

    let ctx = ExecutionContext::for_tests(1);
    let data = mixture_data();
    let prepared = Study::tabular(data)
        .graph_posterior(mixture_graph_posterior())
        .query(AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)))
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let graphs = antecedent_prob::WeightedGraphSamples::new(
        vec![0.6, 0.4],
        vec![
            antecedent_prob::GraphIdentFlag::Identified,
            antecedent_prob::GraphIdentFlag::Unidentified,
        ],
        vec![1, 2],
    )
    .unwrap();
    let query = QueryId::from_raw(0);
    let unlock = [(query, Arc::from([VariableId::from_raw(2)]))];
    let eval = antecedent::design::DesignEvaluationContext::<(), ()> {
        graphs: &graphs,
        query_id_unlock: Some(&unlock),
        ..empty_eval(&graphs)
    };
    let candidates = vec![
        CandidateDesign::IncreaseSamplingRate(SamplingPlan {
            additional_samples: 1_000,
            cost: DesignCost::zero(),
            tag: 1,
        }),
        CandidateDesign::Measure(MeasurementPlan {
            variables: Arc::from([VariableId::from_raw(2)]),
            cost: DesignCost::zero(),
            tag: 2,
        }),
    ];
    let preview = prepared
        .preview_rank_designs(
            &DesignObjective::IncreaseIdentificationProbability { query },
            &candidates,
            &eval,
            None,
        )
        .unwrap();
    assert!((preview.unidentified_mass.unwrap() - 0.4).abs() < 1e-12);
    assert_eq!(preview.violations.len(), 1);
    assert_eq!(preview.violations[0].candidate_index, 0);
    assert!(preview.violations[0].detail.contains("observational units"));

    let ranking = prepared
        .rank_designs(
            &tiny_ranker(),
            &DesignObjective::IncreaseIdentificationProbability { query },
            &candidates,
            &eval,
            &ctx,
        )
        .unwrap();
    assert_eq!(ranking.ranked.len(), 1);
    assert_eq!(ranking.ranked[0].candidate_index, 1);
    assert!((graphs.unidentified_mass() - 0.4).abs() < 1e-12);
    assert_eq!(ranking.violations[0].constraint.as_ref(), "unlicensed_candidate");
}

#[test]
fn composition_request_lifecycle_duplicate_conflict_cancel_and_order() {
    let ctx = ExecutionContext::for_tests(47);
    let (data, dag, query) = confounded_scm(64, 47);
    let prepared = study(data.clone(), dag, query).prepare(&ctx).unwrap();
    let contract = prepared.contract().unwrap();
    let claim = prepared.estimate(&data, &ctx).unwrap().claim(&contract, &ctx).unwrap();
    let execution =
        antecedent_io::execution_digest(&antecedent_io::execution_identity_from_context(&ctx))
            .unwrap();
    let key = SemanticDigest::from_bytes([11; 32]);
    let request = RequestIdentity::new(key, contract.identities, execution);
    assert!(!request.conflicts_with(&RequestIdentity::new(key, contract.identities, execution)));
    let other_data = antecedent_core::ContractIdentities::new(
        contract.identities.target,
        contract.identities.identification,
        contract.identities.identification_product,
        contract.identities.program,
        contract.identities.inference_binding,
        contract.identities.observation,
        SemanticDigest::from_bytes([12; 32]),
    );
    assert!(request.conflicts_with(&RequestIdentity::new(key, other_data, execution)));

    let pending = ExecutionReceipt::new(request.clone(), ExecutionRequestState::Pending);
    let running = ExecutionReceipt::new(request.clone(), ExecutionRequestState::Running);
    let completed = ExecutionReceipt::new(request.clone(), ExecutionRequestState::Completed);
    let cancelled = ExecutionReceipt::new(request.clone(), ExecutionRequestState::Cancelled);
    assert!(!pending.state.publishes_claim());
    assert!(!running.state.publishes_claim());
    assert!(!cancelled.state.publishes_claim());
    assert!(completed.state.publishes_claim());

    let mut current: Option<ExecutionReceipt> = None;
    for receipt in [pending, running, completed.clone(), cancelled] {
        if receipt.state.publishes_claim() {
            assert!(current.as_ref().is_none_or(|prev| !prev.state.publishes_claim()
                || (!prev.request.conflicts_with(&receipt.request))));
            current = Some(receipt);
        } else if current.as_ref().is_some_and(|prev| prev.state.publishes_claim()) {
            assert!(
                !receipt.state.publishes_claim(),
                "cancel or partial must not replace a published claim"
            );
        }
    }
    assert_eq!(current.unwrap().state, ExecutionRequestState::Completed);
    assert_eq!(claim.identities.program, contract.identities.program);
}

#[test]
fn composition_inspect_and_metadata_read_do_not_identify() {
    let sink = Arc::new(RecordingProgress::default());
    let mut ctx = ExecutionContext::for_tests(53);
    ctx.progress = Some(Arc::clone(&sink) as Arc<dyn ProgressSink>);
    let (data, dag, query) = confounded_scm(64, 53);
    let built = study(data.clone(), dag, query);
    let inspected = built.inspect().unwrap();
    assert!(matches!(inspected.reasoning.identification, SlotAvailability::Unavailable { .. }));
    assert!(inspected.identities.identification_product.is_none());
    let _ = built.capability().unwrap();
    assert_eq!(identify_computations(&sink), 0, "inspect/capability must not identify");

    let prepared = built.prepare(&ctx).unwrap();
    assert_eq!(identify_computations(&sink), 1);
    let result = prepared.estimate(&data, &ctx).unwrap();
    let bytes = prepared.encode_contracted_result(&result, "j-meta", &ctx).unwrap();
    let before = identify_computations(&sink);
    let artifact = antecedent_io::EncodedArtifact::read_from(bytes.as_slice()).unwrap();
    let section =
        antecedent_io::decode_analysis_result_contract(&artifact).unwrap().expect("contract");
    assert_eq!(identify_computations(&sink), before, "metadata-only read must not identify");
    assert_eq!(
        section.identities.program,
        *prepared.contract().unwrap().identities.program.expect("prepared program").as_bytes()
    );
}

#[test]
fn composition_prepared_estimate_matches_fresh() {
    let ctx = ExecutionContext::for_tests(59);
    let (data, dag, query) = confounded_scm(128, 59);
    let prepared = study(data.clone(), dag.clone(), query.clone()).prepare(&ctx).unwrap();
    let prepared_result = prepared.estimate(&data, &ctx).unwrap();
    let fresh =
        study(data.clone(), dag, query).prepare(&ctx).unwrap().estimate(&data, &ctx).unwrap();
    assert!((prepared_result.effect() - fresh.effect()).abs() < 1e-12);
    assert!((prepared_result.estimate.se_analytic - fresh.estimate.se_analytic).abs() < 1e-12);
}

/// A record for the construction `basis` describes, or `None`.
fn record_for(
    basis: &antecedent_io::calibration::CalibrationBasisWire,
) -> Option<&'static antecedent_io::coverage_records_data::CoverageRecord> {
    let key = &basis.key;
    antecedent::coverage_records_data::RECORDS.iter().find(|record| {
        record.query == key.query
            && record.graph_class == key.graph_class
            && record.structure == key.structure
            && record.inference == key.inference
            && record.estimator == key.estimator
            && record.interval_method == key.interval_method
            && record.se_kind == key.se_kind
            && record.dependence == key.dependence
            && record.posterior == key.posterior
            && record.functional == key.functional
            && record.identification == key.identification
            && (record.nominal - key.level).abs() < 1e-9
    })
}

/// A matching historical record is calibrated only while its measured facets
/// still attest this tree. The branch can be tested before its next deliberate
/// calibration run without turning a stale record into a coverage claim.
fn assert_measured_record_status(
    status: &str,
    reason: Option<&str>,
    record: &antecedent_io::coverage_records_data::CoverageRecord,
) {
    if antecedent_io::calibration::record_attests_current_code(record) {
        assert_eq!(status, "calibrated");
    } else {
        assert_eq!(status, "scope_not_assessed");
        assert_eq!(reason, Some("coverage_record_not_attesting"));
    }
}

fn calibration_of(
    prepared: &antecedent::PreparedStudy,
    result: &StudyResult,
    ctx: &ExecutionContext,
) -> antecedent_core::CalibrationView {
    let contract = prepared.contract().unwrap();
    result.claim(&contract, ctx).unwrap().calibration
}

/// The Bayesian g-computation construction `bayesian_gcomp_misspecification_probe`
/// measures on its correctly specified law: conjugate Gaussian, prior scale 10,
/// 400 draws, 500 rows. Its record is what an execution of the same
/// construction must be calibrated against.
fn gcomp_measured_study(
    n: usize,
    draws: usize,
    seed: u64,
) -> (antecedent::PreparedStudy, TabularData) {
    let ctx = ExecutionContext::for_tests(seed);
    let (data, dag, query) = confounded_scm(n, seed);
    let prepared = Study::tabular(data.clone())
        .graph(dag)
        .query(query)
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(draws).prior_scale(10.0),
        ))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    (prepared, data)
}

#[test]
fn calibration_slot_is_calibrated_inside_record_scope() {
    let ctx = ExecutionContext::for_tests(21);
    let (prepared, data) = gcomp_measured_study(500, 400, 21);
    let result = prepared.estimate(&data, &ctx).unwrap();
    let contract = prepared.contract().unwrap();
    let basis = &result.calibration_bases(&contract).unwrap()[0];
    let record =
        record_for(basis).unwrap_or_else(|| panic!("no coverage record measures {:#?}", basis.key));
    assert!(!record.boundary, "{} is a boundary record", record.id);
    let claim = calibration_of(&prepared, &result, &ctx);
    assert_measured_record_status(claim.status.as_ref(), claim.reason.as_deref(), record);
    assert_eq!(claim.record_id.as_deref(), Some(record.id));
    assert_eq!(claim.calibration_sha.as_deref(), Some(record.calibration_sha));
    assert_eq!(claim.observed, Some(record.observed));
}

#[test]
fn calibration_slot_is_scope_not_assessed_outside_the_measured_sample_size() {
    let ctx = ExecutionContext::for_tests(22);
    let (prepared, data) = gcomp_measured_study(50, 400, 22);
    let result = prepared.estimate(&data, &ctx).unwrap();
    let claim = calibration_of(&prepared, &result, &ctx);
    assert_eq!(claim.status.as_ref(), "scope_not_assessed", "{claim:#?}");
    assert_eq!(claim.reason.as_deref(), Some("sample_size_outside_measured_range"));
    assert!(claim.record_id.is_some());
}

#[test]
fn missing_row_padding_does_not_upgrade_calibration() {
    let ctx = ExecutionContext::for_tests(24);
    let mut t = vec![0.0; 100];
    let mut y = vec![0.0; 100];
    for i in 0..100 {
        t[i] = (i as f64) * 0.1;
        y[i] = 2.0 * t[i] + ((i % 5) as f64 - 2.0) * 0.1;
    }
    let run = |pad: usize| {
        let mut tp = t.clone();
        let mut yp = y.clone();
        tp.extend(std::iter::repeat(f64::NAN).take(pad));
        yp.extend(std::iter::repeat(f64::NAN).take(pad));
        let data =
            TabularData::from_f64_columns([("t", tp.as_slice()), ("y", yp.as_slice())]).unwrap();
        let mut dag = Dag::with_variables(2);
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let prepared = Study::tabular(data.clone())
            .graph(dag)
            .query(CausalQuery::average_effect(AverageEffectQuery::binary_ate(
                VariableId::from_raw(0),
                VariableId::from_raw(1),
            )))
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .prepare(&ctx)
            .unwrap();
        let result = prepared.estimate(&data, &ctx).unwrap();
        let claim = calibration_of(&prepared, &result, &ctx);
        let contract = prepared.contract().unwrap();
        let basis = &result.calibration_bases(&contract).unwrap()[0];
        assert_eq!(basis.scope.row_count, 100);
        assert_eq!(contract.row_count, (100 + pad) as u64);
        assert_eq!(artifact_analysis_n(&prepared, &result, &ctx), 100);
        (
            result.effect(),
            result.estimate.se_analytic,
            claim.status.to_string(),
            result.estimate.n_obs,
        )
    };
    let (e0, se0, status0, n0) = run(0);
    let (e1, se1, status1, n1) = run(400);
    let (e2, se2, status2, n2) = run(900);
    assert_eq!(n0, Some(100));
    assert_eq!(n1, Some(100));
    assert_eq!(n2, Some(100));
    assert!((e0 - e1).abs() < 1e-12 && (e0 - e2).abs() < 1e-12);
    assert!((se0 - se1).abs() < 1e-12 && (se0 - se2).abs() < 1e-12);
    assert_eq!(status0, status1);
    assert_eq!(status0, status2);
}

fn artifact_analysis_n(
    prepared: &antecedent::PreparedStudy,
    result: &StudyResult,
    ctx: &ExecutionContext,
) -> u64 {
    let bytes = prepared.encode_contracted_result(result, "r11-n", ctx).unwrap();
    let consumed = consume_analysis_result(&bytes).unwrap();
    consumed
        .contract
        .as_ref()
        .and_then(|section| section.claim.as_ref())
        .and_then(|claim| claim.calibration.basis.as_ref())
        .map(|basis| basis.scope.row_count)
        .expect("encoded claim carries an analysis-sample calibration basis")
}

#[test]
fn matching_missing_row_padding_does_not_upgrade_calibration() {
    let ctx = ExecutionContext::for_tests(25);
    let mut t = vec![0.0; 100];
    let mut y = vec![0.0; 100];
    let mut z = vec![0.0; 100];
    for i in 0..100 {
        t[i] = (i % 2) as f64;
        z[i] = (i as f64) * 0.05;
        y[i] = 2.0 * t[i] + z[i];
    }
    let run = |pad: usize| {
        let mut tp = t.clone();
        let mut yp = y.clone();
        let mut zp = z.clone();
        tp.extend(std::iter::repeat(f64::NAN).take(pad));
        yp.extend(std::iter::repeat(f64::NAN).take(pad));
        zp.extend(std::iter::repeat(f64::NAN).take(pad));
        let data = TabularData::from_f64_columns([
            ("t", tp.as_slice()),
            ("y", yp.as_slice()),
            ("z", zp.as_slice()),
        ])
        .unwrap();
        let mut dag = Dag::with_variables(3);
        dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
        dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let prepared = Study::tabular(data.clone())
            .graph(dag)
            .query(CausalQuery::average_effect(AverageEffectQuery::binary_ate(
                VariableId::from_raw(0),
                VariableId::from_raw(1),
            )))
            .estimator(antecedent_estimate::PropensityMatching {
                bootstrap_replicates: 0,
                se_kind: antecedent_estimate::AnalyticSeKind::Homoskedastic,
                ..antecedent_estimate::PropensityMatching::new()
            })
            .refute(RefuteSuite::None)
            .build()
            .unwrap()
            .prepare(&ctx)
            .unwrap();
        let result = prepared.estimate(&data, &ctx).unwrap();
        let contract = prepared.contract().unwrap();
        let basis = &result.calibration_bases(&contract).unwrap()[0];
        assert!(basis.scope.row_count < contract.row_count || pad == 0);
        assert_eq!(basis.scope.row_count, result.estimate.n_obs.unwrap());
        assert_eq!(artifact_analysis_n(&prepared, &result, &ctx), basis.scope.row_count);
        (result.effect(), result.estimate.n_obs, basis.scope.row_count)
    };
    let (e0, n0, scope0) = run(0);
    let (e1, n1, scope1) = run(400);
    let (e2, n2, scope2) = run(900);
    assert_eq!(n0, n1);
    assert_eq!(n0, n2);
    assert_eq!(scope0, scope1);
    assert_eq!(scope0, scope2);
    assert!((e0 - e1).abs() < 1e-10 && (e0 - e2).abs() < 1e-10);
}

#[test]
fn temporal_calibration_scope_uses_lag_aligned_n() {
    let ctx = ExecutionContext::for_tests(26);
    let (series, graph) = lag1_series();
    let snapshot_n = series.row_count() as u64;
    let prepared = Study::series(series.clone())
        .graph(graph)
        .temporal_query(pulse_query())
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let result = prepared.estimate_series(&series, &ctx).unwrap();
    let contract = prepared.contract().unwrap();
    let analysis_n = result.estimate.n_obs.expect("temporal pulse records lag-aligned n");
    assert!(analysis_n < snapshot_n, "lag alignment must drop at least the history window");
    assert_eq!(contract.row_count, snapshot_n);
    let basis = &result.calibration_bases(&contract).unwrap()[0];
    assert_eq!(basis.scope.row_count, analysis_n);
    assert_eq!(artifact_analysis_n(&prepared, &result, &ctx), analysis_n);
}

#[test]
fn calibration_slot_is_not_calibrated_with_fewer_posterior_draws_than_measured() {
    let ctx = ExecutionContext::for_tests(23);
    let (measured, data) = gcomp_measured_study(500, 400, 23);
    let calibrated = calibration_of(&measured, &measured.estimate(&data, &ctx).unwrap(), &ctx);
    let contract = measured.contract().unwrap();
    let result = measured.estimate(&data, &ctx).unwrap();
    let basis = &result.calibration_bases(&contract).unwrap()[0];
    assert_measured_record_status(
        calibrated.status.as_ref(),
        calibrated.reason.as_deref(),
        record_for(basis).unwrap(),
    );

    let (few, data) = gcomp_measured_study(500, 4, 23);
    let claim = calibration_of(&few, &few.estimate(&data, &ctx).unwrap(), &ctx);
    assert_ne!(claim.status.as_ref(), "calibrated", "4 draws is not the measured construction");
    assert_eq!(claim.reason.as_deref(), Some("posterior_draws_below_measured"));
}

/// The static natural-mediation law `v19_static_calibration::mediation_data`
/// measures: `t = 0.5x + e`, `m = 0.6t + 0.8e`, `y = 0.4t + 0.5m + 0.5x + e`,
/// 400 rows, 200 bootstrap replicates.
fn mediation_measured_study(
    n: usize,
    replicates: u32,
    seed: u64,
) -> (antecedent::PreparedStudy, TabularData) {
    let ctx = ExecutionContext::for_tests(seed);
    let mut rng = CausalRng::from_seed(seed);
    let normal = |rng: &mut CausalRng| {
        let u1 = rng.next_f64().max(1e-12);
        let u2 = rng.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };
    let (mut t, mut m, mut y, mut x) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        x[i] = normal(&mut rng);
        t[i] = 0.5 * x[i] + normal(&mut rng);
        m[i] = 0.6 * t[i] + 0.8 * normal(&mut rng);
        y[i] = 0.4 * t[i] + 0.5 * m[i] + 0.5 * x[i] + normal(&mut rng);
    }
    let data = TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("m", m.as_slice()),
        ("y", y.as_slice()),
        ("x", x.as_slice()),
    ])
    .unwrap();
    let mut dag = Dag::with_variables(4);
    for (a, b) in [(3, 0), (3, 2), (0, 1), (0, 2), (1, 2)] {
        dag.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    let query = CausalQuery::Mediation(MediationQuery::binary(
        VariableId::from_raw(0),
        VariableId::from_raw(2),
        Arc::from([VariableId::from_raw(1)]),
        MediationContrast::NaturalDirect,
    ));
    let prepared = Study::tabular(data.clone())
        .graph(dag)
        .query(query)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(replicates)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    (prepared, data)
}

#[test]
fn calibration_slot_is_not_calibrated_with_fewer_bootstrap_replicates_than_measured() {
    let ctx = ExecutionContext::for_tests(24);
    let (measured, data) = mediation_measured_study(400, 200, 24);
    let result = measured.estimate(&data, &ctx).unwrap();
    let contract = measured.contract().unwrap();
    let basis = &result.calibration_bases(&contract).unwrap()[0];
    assert_eq!(basis.key.interval_method, "bootstrap_se");
    let record =
        record_for(basis).unwrap_or_else(|| panic!("no coverage record measures {:#?}", basis.key));
    assert!(record.replicates_min >= 2, "{} measured no replicate floor", record.id);
    let calibrated = calibration_of(&measured, &result, &ctx);
    assert_measured_record_status(calibrated.status.as_ref(), calibrated.reason.as_deref(), record);

    let (few, data) = mediation_measured_study(400, 2, 24);
    let claim = calibration_of(&few, &few.estimate(&data, &ctx).unwrap(), &ctx);
    assert_ne!(
        claim.status.as_ref(),
        "calibrated",
        "a two-replicate bootstrap is not the {} construction",
        record.id
    );
    assert_eq!(claim.reason.as_deref(), Some("resampling_replicates_below_measured"));
}

#[test]
fn an_unmeasured_bootstrap_construction_is_unavailable() {
    // The review's reproducer: AIPW with two bootstrap replicates reported
    // `calibrated` against a record whose cited test never ran AIPW.
    let ctx = ExecutionContext::for_tests(28);
    let (data, dag, query) = confounded_scm(1200, 28);
    let prepared = Study::tabular(data.clone())
        .graph(dag)
        .query(query)
        .estimator(EstimatorId::Aipw)
        .bootstrap_replicates(2)
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let claim = calibration_of(&prepared, &result, &ctx);
    assert_ne!(claim.status.as_ref(), "calibrated", "{claim:#?}");
}

#[test]
fn calibration_slot_is_unavailable_with_code_when_no_record() {
    // A median contrast: the same estimator and interval, a functional no
    // coverage test measures.
    let ctx = ExecutionContext::for_tests(7);
    let (data, dag, query) = confounded_scm(600, 7);
    let query = query.with_outcome_functional(antecedent_core::OutcomeFunctional::quantile(0.5));
    let prepared = Study::tabular(data.clone())
        .graph(dag)
        .query(query)
        .estimator(EstimatorId::Aipw)
        .bootstrap_replicates(0)
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let claim = calibration_of(&prepared, &result, &ctx);
    assert_eq!(claim.status.as_ref(), "unavailable");
    assert!(claim.reason.is_some());
    assert!(claim.record_id.is_none());
}

#[test]
fn partially_identified_answers_are_not_calibrated_against_point_records() {
    let ctx = ExecutionContext::for_tests(25);
    let (data, dag, query) = confounded_scm(500, 25);
    let point = Study::tabular(data.clone())
        .graph(dag.clone())
        .query(query.clone())
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(400).prior_scale(10.0),
        ))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let result = point.estimate(&data, &ctx).unwrap();
    let contract = point.contract().unwrap();
    let mut basis = result.calibration_bases(&contract).unwrap()[0].clone();
    assert_eq!(basis.key.identification, "point");
    assert!(
        record_for(&basis).is_some(),
        "the registry must measure this construction: {:?}",
        basis.key
    );
    let point_slot = antecedent_io::calibration::calibration_slot(&basis);
    assert_measured_record_status(
        &point_slot.status,
        point_slot.reason.as_deref(),
        record_for(&basis).unwrap(),
    );
    basis.key.identification = "partial".into();
    basis.scope.unidentified_mass = 0.5;
    let partial = antecedent_io::calibration::calibration_slot(&basis);
    assert_ne!(partial.status, "calibrated", "{partial:#?}");
    assert_eq!(partial.reason.as_deref(), Some("identification_not_measured"));
}

#[test]
fn calibration_slot_only_matches_the_level_it_measured() {
    let ctx = ExecutionContext::for_tests(26);
    let (prepared, data) = gcomp_measured_study(500, 400, 26);
    let result = prepared.estimate(&data, &ctx).unwrap();
    let contract = prepared.contract().unwrap();
    let mut basis = result.calibration_bases(&contract).unwrap()[0].clone();
    assert!(
        record_for(&basis).is_some(),
        "the registry must measure the construction this study builds: {:?}",
        basis.key
    );
    let point_slot = antecedent_io::calibration::calibration_slot(&basis);
    assert_measured_record_status(
        &point_slot.status,
        point_slot.reason.as_deref(),
        record_for(&basis).unwrap(),
    );
    basis.key.level = 0.99;
    let other = antecedent_io::calibration::calibration_slot(&basis);
    assert_ne!(other.status, "calibrated");
    assert_eq!(other.reason.as_deref(), Some("interval_level_not_measured"));
}

#[test]
fn calibration_slot_is_covered_by_claim_id() {
    let ctx = ExecutionContext::for_tests(8);
    let (prepared, data) = gcomp_measured_study(500, 400, 8);
    let result = prepared.estimate(&data, &ctx).unwrap();
    let bytes = prepared.encode_contracted_result(&result, "cal-slot", &ctx).unwrap();
    let consumed = consume_analysis_result(&bytes).unwrap();
    assert_eq!(
        consumed.acceptance.unresolved.as_ref(),
        &[std::sync::Arc::<str>::from("dependencies.checked_bayesian_dag_ate_operation")]
    );
    let original = consumed.contract.as_ref().unwrap().claim.as_ref().unwrap().calibration.clone();

    // Re-encode the same artifact with an edited calibration slot and the
    // original claim_id: claim_id covers the slot, so consume must refuse.
    let (artifact, _, _body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    let mut contract = antecedent_io::decode_analysis_result_contract(&artifact).unwrap().unwrap();
    let claim = contract.claim.as_mut().unwrap();
    claim.calibration.status = if original.status == "calibrated" {
        "scope_not_assessed".into()
    } else {
        "calibrated".into()
    };
    // Repack the section bytes directly: the producer's own encoder already
    // refuses the edit, and this test is about the independent consumer.
    let mut forged = artifact;
    let (descriptor, section) = antecedent_io::pack_section_shared(
        antecedent_io::CONTRACT_SECTION,
        "application/cbor",
        antecedent_io::to_cbor(&contract).unwrap().into(),
        antecedent_io::CompressPolicy::Auto,
    );
    forged.manifest.sections.retain(|item| item.id != antecedent_io::CONTRACT_SECTION);
    forged.sections.retain(|item| item.id != antecedent_io::CONTRACT_SECTION);
    forged.manifest.sections.push(descriptor);
    forged.sections.push(section);
    let mut forged_bytes = Vec::new();
    forged.write_to(&mut forged_bytes).unwrap();
    match consume_analysis_result(&forged_bytes) {
        // The claim digest covers the slot, so the edited artifact either fails
        // to decode as a contract at all or is refused as a verified program.
        Err(err) => assert!(format!("{err}").contains("claim"), "{err}"),
        Ok(consumed) => {
            assert!(
                !consumed.acceptance.accepts_as_verified_program(),
                "an edited calibration status must not verify: {:?}",
                consumed.acceptance.unresolved
            );
            assert!(
                consumed.acceptance.unresolved.iter().any(|item| item.starts_with("claim.")),
                "{:?}",
                consumed.acceptance.unresolved
            );
        }
    }
}

#[test]
fn producer_and_consumer_agree_on_the_calibration_slot() {
    let ctx = ExecutionContext::for_tests(27);
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("tabular", {
            let (prepared, data) = gcomp_measured_study(500, 400, 27);
            let result = prepared.estimate(&data, &ctx).unwrap();
            prepared.encode_contracted_result(&result, "cal-tabular", &ctx).unwrap()
        }),
        ("series", {
            let (series, graph) = lag1_series();
            let prepared = Study::series(series.clone())
                .graph(graph)
                .temporal_query(pulse_query())
                .refute(RefuteSuite::None)
                .bootstrap_replicates(24)
                .build()
                .unwrap()
                .prepare(&ctx)
                .unwrap();
            let result = prepared.estimate_series(&series, &ctx).unwrap();
            prepared.encode_contracted_result(&result, "cal-series", &ctx).unwrap()
        }),
        ("panel", {
            let (panel, graph) = lag1_panel();
            let prepared = Study::panel(panel.clone())
                .graph(graph)
                .temporal_query(pulse_query().with_max_history_lag(Some(1)))
                .refute(RefuteSuite::None)
                .bootstrap_replicates(0)
                .build()
                .unwrap()
                .prepare(&ctx)
                .unwrap();
            let result = prepared.estimate_panel(&panel, &ctx).unwrap();
            let contract = prepared.contract().unwrap();
            let basis = &result.calibration_bases(&contract).unwrap()[0];
            assert_eq!(basis.key.dependence, "panel_cluster", "a panel execution clusters by unit");
            prepared.encode_contracted_result(&result, "cal-panel", &ctx).unwrap()
        }),
    ];
    for (label, bytes) in cases {
        let consumed = consume_analysis_result(&bytes).unwrap();
        let claim = consumed.contract.as_ref().unwrap().claim.as_ref().unwrap();
        assert_eq!(
            antecedent_io::calibration::rederive_calibration(&claim.calibration),
            claim.calibration,
            "{label}: the consumer re-derives a different slot"
        );
        if label == "tabular" {
            assert_eq!(
                consumed.acceptance.unresolved.as_ref(),
                &[std::sync::Arc::<str>::from("dependencies.checked_bayesian_dag_ate_operation")]
            );
        } else if label == "series" {
            assert_eq!(
                consumed.acceptance.unresolved.as_ref(),
                &[std::sync::Arc::<str>::from(
                    "dependencies.checked_temporal_dag_effect_operation"
                )]
            );
        } else {
            assert!(
                consumed.acceptance.accepts_as_verified_program(),
                "{label}: {:?}",
                consumed.acceptance.unresolved
            );
        }
    }
}

/// Three units of [`lag1_series`] with shifted phases.
fn lag1_panel() -> (antecedent_data::PanelData, TemporalDag) {
    let (_, graph) = lag1_series();
    let unit = |unit_id: u32, phase: f64| {
        let n = 80usize;
        let mut x = vec![0.0; n];
        let mut y = vec![0.0; n];
        for t in 1..n {
            x[t] = (t as f64 * 0.05 + phase).sin();
            y[t] = 0.5 * x[t - 1] + 0.01 * phase;
        }
        let data =
            TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())]).unwrap();
        antecedent_data::PanelUnit {
            unit_id,
            series: TimeSeriesData::try_new(
                data.storage().clone(),
                TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
            )
            .unwrap(),
        }
    };
    let panel =
        antecedent_data::PanelData::try_new(Arc::from([unit(0, 0.0), unit(1, 0.7), unit(2, 1.4)]))
            .unwrap();
    (panel, graph)
}

#[test]
fn every_licensed_cell_has_a_calibration_record_or_code() {
    let licensed = include_str!("../../../parity/support_licensed.toml");
    let mut missing = Vec::new();
    for block in licensed.split("[[cell]]").skip(1) {
        let has_record = block.contains("calibration =");
        let has_reason = block.contains("calibration_reason =");
        // A record list beside `boundary_record` says what was measured and that none
        // of it is a nominal pass; any other pairing is one too many or too few.
        let boundary_only = block.contains("calibration_reason = \"boundary_record\"");
        if has_record == has_reason && !(has_record && boundary_only) {
            missing.push(block.lines().take(6).collect::<Vec<_>>().join(" "));
        }
        if has_record {
            for line in block.lines() {
                let line = line.trim();
                if let Some(rest) = line.strip_prefix("calibration =") {
                    for id in rest.split(['[', ']', ',', '"']).filter(|s| s.starts_with("cov.")) {
                        assert!(
                            antecedent::coverage_records_data::RECORDS
                                .iter()
                                .any(|row| row.id == id),
                            "missing coverage record {id}"
                        );
                    }
                }
            }
        }
    }
    assert!(
        missing.is_empty(),
        "cells missing exactly one of calibration / calibration_reason: {missing:?}"
    );
}

#[test]
fn every_calibration_reason_code_is_in_the_vocabulary() {
    // `RECORD_NOT_ATTESTING` is runtime-only (see `antecedent_io::calibration`'s
    // module doc): it is deliberately absent from `parity/reason_codes.toml`,
    // because whether a record attests is generated with the registry
    // (`ATTESTING_RECORD_IDS`) and is not part of the checked-in vocabulary.
    let vocabulary = include_str!("../../../parity/reason_codes.toml");
    for code in antecedent_io::calibration::REASON_CODES {
        if code == antecedent_io::calibration::RECORD_NOT_ATTESTING {
            continue;
        }
        assert!(
            vocabulary.contains(&format!("id = \"{code}\"")),
            "{code} is not in parity/reason_codes.toml"
        );
    }
}

#[test]
fn estimator_spec_changes_inference_binding_and_program_stays() {
    let ctx = ExecutionContext::for_tests(9);
    let (data, dag, query) = confounded_scm(96, 9);
    let a = Study::tabular(data.clone())
        .graph(dag.clone())
        .query(query.clone())
        .estimator(antecedent_estimate::LinearAdjustmentAte::new().with_bootstrap_replicates(0))
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let b = Study::tabular(data)
        .graph(dag)
        .query(query)
        .estimator(
            antecedent_estimate::LinearAdjustmentAte::new()
                .with_bootstrap_replicates(0)
                .with_se_kind(antecedent_estimate::AnalyticSeKind::Hc1),
        )
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let ca = a.contract().unwrap();
    let cb = b.contract().unwrap();
    assert_ne!(ca.identities.inference_binding, cb.identities.inference_binding);
}

#[test]
fn rd_config_changes_identification() {
    let ctx = ExecutionContext::for_tests(10);
    let (data, dag, query) = confounded_scm(96, 10);
    let a = Study::tabular(data.clone())
        .graph(dag.clone())
        .query(query.clone())
        .rd_config(VariableId::from_raw(2), 0.0, 1.0)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let b = Study::tabular(data)
        .graph(dag)
        .query(query)
        .rd_config(VariableId::from_raw(2), 0.0, 2.0)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    assert_ne!(
        a.contract().unwrap().identities.identification,
        b.contract().unwrap().identities.identification
    );
}

#[test]
fn constant_nonunit_weights_do_not_change_target() {
    assert!(!antecedent_estimate::changes_target(&[7.0; 8]));
    assert!(antecedent_estimate::changes_target(&[1.0, 2.0, 3.0]));
    assert!(antecedent_estimate::changes_target(&[f64::NAN, 1.0]));
}

fn aipw_prepared(
    data: &TabularData,
    dag: Dag,
    query: AverageEffectQuery,
) -> antecedent::PreparedStudy {
    Study::tabular(data.clone())
        .graph(dag)
        .query(query)
        .estimator(EstimatorId::Aipw)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ExecutionContext::for_tests(12))
        .unwrap()
}

fn contract_of(bytes: &[u8]) -> antecedent_io::AnalysisResultContractWire {
    consume_analysis_result(bytes).unwrap().contract.expect("contract section")
}

fn z_weights(data: &TabularData, scale: f64) -> Vec<f64> {
    let z = data.float64_slice(VariableId::from_raw(2)).unwrap();
    z.iter().map(|zi| (zi / scale).exp()).collect()
}

#[test]
fn encoder_refuses_result_population_mismatch() {
    let ctx = ExecutionContext::for_tests(11);
    let (data, dag, query) = confounded_scm(96, 11);
    let average = aipw_prepared(&data, dag.clone(), query.clone());
    let treated = aipw_prepared(
        &data,
        dag,
        query.with_target_population(antecedent_core::TargetPopulation::Treated),
    );
    // A genuine treated-population execution is not the average handle's target.
    let treated_result = treated.estimate(&data, &ctx).unwrap();
    let err = average.encode_contracted_result(&treated_result, "mismatch", &ctx).unwrap_err();
    assert!(format!("{err}").contains("program"), "{err}");
    // A genuine retarget does not become the target of another handle.
    let retargeted =
        average.retarget(&z_weights(&data, 3.0), &[VariableId::from_raw(2)], &ctx).unwrap();
    assert!(treated.encode_contracted_result(&retargeted, "mismatch", &ctx).is_err());
    assert!(treated.encode_contracted_result(&treated_result, "ok", &ctx).is_ok());
}

#[test]
fn encoder_refuses_a_result_from_another_program() {
    let ctx = ExecutionContext::for_tests(3);
    let (data, dag, query) = confounded_scm(400, 3);
    let confounded = study(data.clone(), dag, query.clone()).prepare(&ctx).unwrap();
    // Same query and data, a graph without the confounding edge: a different
    // program that produces a different number.
    let mut unconfounded_graph = Dag::with_variables(3);
    unconfounded_graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    unconfounded_graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    let unconfounded = study(data.clone(), unconfounded_graph, query).prepare(&ctx).unwrap();
    let foreign = unconfounded.estimate(&data, &ctx).unwrap();
    let own = confounded.estimate(&data, &ctx).unwrap();
    assert!((foreign.effect() - own.effect()).abs() > 0.1, "the programs disagree");
    let err = confounded.encode_contracted_result(&foreign, "foreign", &ctx).unwrap_err();
    assert!(format!("{err}").contains("program"), "{err}");
    assert!(confounded.encode_contracted_result(&own, "own", &ctx).is_ok());
    // A result no prepared handle executed cannot be certified either.
    let mut detached = own.clone();
    detached.executed_contract = None;
    let err = confounded.encode_contracted_result(&detached, "detached", &ctx).unwrap_err();
    assert!(format!("{err}").contains("prepared handle"), "{err}");
}

#[test]
fn encoder_refuses_a_result_computed_on_another_snapshot() {
    let ctx = ExecutionContext::for_tests(5);
    let (data, dag, query) = confounded_scm(1200, 5);
    let prepared = study(data.clone(), dag, query).prepare(&ctx).unwrap();
    let (small, _, _) = confounded_scm(50, 99);
    let other = prepared.estimate(&small, &ctx).unwrap();
    let err = prepared.encode_contracted_result(&other, "other-data", &ctx).unwrap_err();
    assert!(format!("{err}").contains("data snapshot"), "{err}");
    let consumed = consume_analysis_result(
        &prepared
            .encode_contracted_result(&prepared.estimate(&data, &ctx).unwrap(), "own", &ctx)
            .unwrap(),
    )
    .unwrap();
    let snapshot = consumed
        .contract
        .as_ref()
        .and_then(|section| section.data_snapshot.as_ref())
        .expect("data snapshot");
    assert_eq!(snapshot.row_count, 1200);
}

#[test]
fn nonconstant_retarget_exports_with_target_weights_identity() {
    let ctx = ExecutionContext::for_tests(12);
    let (data, dag, query) = confounded_scm(128, 12);
    let prepared = aipw_prepared(&data, dag, query);
    let z = [VariableId::from_raw(2)];
    let preview = prepared.preview_transform(TransformIntent::Retarget).unwrap();
    let weights = z_weights(&data, 3.0);
    let retargeted = prepared.apply_retarget(&preview, &weights, &z, &ctx).unwrap();
    let binding = retargeted.row_weights.clone().expect("row-weight binding");
    let bytes = prepared.encode_contracted_result(&retargeted, "reweight", &ctx).unwrap();
    let consumed = consume_analysis_result(&bytes).unwrap();
    // The row-weight retarget is still a separate composition whose checked
    // lowering has not migrated. The artifact remains readable and its weight
    // identity is verifiable, but consumption must not upgrade the AIPW program.
    assert!(!consumed.acceptance.accepts_as_verified_program());
    assert!(
        consumed
            .acceptance
            .unresolved
            .iter()
            .any(|item| item.as_ref() == "program.checked_aipw_lowering")
    );
    let section = consumed.contract.expect("contract");
    let target_weights = section.identities.target_weights.expect("target_weights identity");
    assert_eq!(target_weights, *binding.target_weights.as_bytes());
    assert_eq!(
        antecedent_io::target_weights_digest(&section.target_weights.as_ref().unwrap().identity)
            .unwrap()
            .as_bytes(),
        &target_weights
    );
    let carried = section.target_weights.as_ref().unwrap();
    assert_eq!(carried.values, weights);
    assert_eq!(carried.identity.data_snapshot, section.identities.data_snapshot);
    assert_eq!(Some(carried.identity.score_reuse), section.identities.score_reuse);
    assert_eq!(
        Some(carried.identity.score_reuse),
        prepared.score_reuse_identity().unwrap().map(|digest| *digest.as_bytes())
    );
    assert_eq!(carried.identity.depends_on, vec![2]);
    match &section.target.query {
        CausalQueryWire::AverageEffect { target_population, .. } => assert_eq!(
            target_population,
            &TargetPopulationWire::RowWeights { weights: target_weights, depends_on: vec![2] }
        ),
        other => panic!("expected AverageEffect, got {other:?}"),
    }

    // Two different weight vectors give two target-weights identities.
    let other = prepared.retarget(&z_weights(&data, 2.0), &z, &ctx).unwrap();
    let other_bytes = prepared.encode_contracted_result(&other, "reweight-2", &ctx).unwrap();
    let other_section = contract_of(&other_bytes);
    assert_ne!(other_section.identities.target_weights, Some(target_weights));
    assert_ne!(other_section.identities.target, section.identities.target);

    // Constant weights of any scale keep the original target.
    let estimated = prepared.estimate(&data, &ctx).unwrap();
    let plain = contract_of(&prepared.encode_contracted_result(&estimated, "plain", &ctx).unwrap());
    let constant = prepared.retarget(&vec![7.0; weights.len()], &[], &ctx).unwrap();
    assert!(constant.row_weights.is_none());
    let constant =
        contract_of(&prepared.encode_contracted_result(&constant, "constant", &ctx).unwrap());
    assert!(constant.identities.target_weights.is_none());
    assert_eq!(constant.identities.target, plain.identities.target);
    assert!(plain.identities.target_weights.is_none());
    assert_eq!(plain.identities.score_reuse, section.identities.score_reuse);
}

#[test]
fn consume_rederives_target_weights_and_refuses_substitutions() {
    let ctx = ExecutionContext::for_tests(16);
    let (data, dag, query) = confounded_scm(96, 16);
    let prepared = aipw_prepared(&data, dag, query);
    let z = [VariableId::from_raw(2)];
    let retargeted = prepared.retarget(&z_weights(&data, 3.0), &z, &ctx).unwrap();
    let bytes = prepared.encode_contracted_result(&retargeted, "reweight", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    let names = header.variable_names.clone();
    let section = contract_of(&bytes);
    let refused = |contract: &antecedent_io::AnalysisResultContractWire, label: &str| {
        let consumed =
            consume_analysis_result(&rewrite_contract_section(&body, names.clone(), "t", contract))
                .unwrap();
        assert!(!consumed.acceptance.accepts_as_verified_program(), "{label}");
        assert!(
            consumed.acceptance.unresolved.iter().any(|item| &**item == label),
            "{label}: {:?}",
            consumed.acceptance.unresolved
        );
    };
    let mut values = section.clone();
    values.target_weights.as_mut().unwrap().values[0] *= 2.0;
    refused(&values, "target_weights.values");

    let mut dropped = section.clone();
    dropped.target_weights = None;
    refused(&dropped, "identities.target_weights");

    let mut both_dropped = section.clone();
    both_dropped.target_weights = None;
    both_dropped.identities.target_weights = None;
    refused(&both_dropped, "target.population");

    let other = prepared.retarget(&z_weights(&data, 2.0), &z, &ctx).unwrap();
    let other = contract_of(&prepared.encode_contracted_result(&other, "other", &ctx).unwrap());
    let mut swapped = section.clone();
    swapped.target_weights = other.target_weights.clone();
    swapped.identities.target_weights = other.identities.target_weights;
    refused(&swapped, "target.population");

    let mut rebound = section.clone();
    rebound.target_weights.as_mut().unwrap().identity.data_snapshot[0] ^= 0xff;
    rebound.identities.target_weights = Some(
        *antecedent_io::target_weights_digest(&rebound.target_weights.as_ref().unwrap().identity)
            .unwrap()
            .as_bytes(),
    );
    refused(&rebound, "target_weights.data_snapshot");

    let mut score = section.clone();
    score.score_reuse = None;
    refused(&score, "identities.score_reuse");
}

#[test]
fn row_weights_reexecute_only_on_their_own_snapshot() {
    let ctx = ExecutionContext::for_tests(17);
    let (data, dag, query) = confounded_scm(96, 17);
    let (other_data, _, _) = confounded_scm(96, 71);
    let prepared = aipw_prepared(&data, dag, query);
    let z = [VariableId::from_raw(2)];
    let weights = z_weights(&data, 3.0);
    let retargeted = prepared.retarget(&weights, &z, &ctx).unwrap();
    let bytes = prepared.encode_contracted_result(&retargeted, "reweight", &ctx).unwrap();
    let carried = contract_of(&bytes).target_weights.expect("weights travel");

    let again = prepared.reexecute_retarget(&carried, &ctx).unwrap();
    assert_eq!(again.effect().to_bits(), retargeted.effect().to_bits());
    assert_eq!(again.row_weights, retargeted.row_weights);

    let mut refreshed = prepared.clone();
    refreshed.refresh(other_data, &ctx).unwrap();
    let err = refreshed.reexecute_retarget(&carried, &ctx).unwrap_err();
    assert!(format!("{err}").contains("reason=row_weights_bound_to_snapshot"), "{err}");
    // A retarget computed on one snapshot cannot be exported through a handle
    // holding another snapshot of the same shape.
    let err = refreshed.encode_contracted_result(&retargeted, "stale", &ctx).unwrap_err();
    assert!(format!("{err}").contains("reason=row_weights_bound_to_snapshot"), "{err}");
}

#[test]
fn retarget_preserves_identification_digest() {
    let ctx = ExecutionContext::for_tests(13);
    let (data, dag, query) = confounded_scm(128, 13);
    let prepared = aipw_prepared(&data, dag, query);
    let estimated = prepared.estimate(&data, &ctx).unwrap();
    let before =
        contract_of(&prepared.encode_contracted_result(&estimated, "before", &ctx).unwrap());
    let preview = prepared.preview_transform(TransformIntent::Retarget).unwrap();
    let mut weights = vec![1.0; data.row_count()];
    weights[1] = 3.0;
    let retargeted =
        prepared.apply_retarget(&preview, &weights, &[VariableId::from_raw(2)], &ctx).unwrap();
    let after =
        contract_of(&prepared.encode_contracted_result(&retargeted, "after", &ctx).unwrap());
    assert_ne!(before.identities.target, after.identities.target, "retarget changes the target");
    assert_eq!(before.identities.identification, after.identities.identification);
    assert_eq!(before.identification, after.identification);
    assert_eq!(before.identities.identification_product, after.identities.identification_product);
}

#[test]
fn retarget_report_matches_identity_deltas() {
    use antecedent_core::{SemanticLayer, TransformEffect};
    let ctx = ExecutionContext::for_tests(14);
    let (data, dag, query) = confounded_scm(128, 14);
    let prepared = Study::tabular(data.clone())
        .graph(dag)
        .query(query)
        .estimator(EstimatorId::Aipw)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let estimated = prepared.estimate(&data, &ctx).unwrap();
    let before_bytes =
        prepared.encode_contracted_result(&estimated, "before-retarget", &ctx).unwrap();
    let before = consume_analysis_result(&before_bytes).unwrap().contract.unwrap();
    let preview = prepared.preview_transform(TransformIntent::Retarget).unwrap();
    assert!(preview.layer(SemanticLayer::Identification).is_some_and(|layer| {
        layer.effects.iter().any(|effect| *effect == TransformEffect::Preserves)
    }));
    assert!(preview.layer(SemanticLayer::Data).is_some_and(|layer| {
        layer.effects.iter().any(|effect| *effect == TransformEffect::Preserves)
    }));
    assert!(preview.layer(SemanticLayer::Inference).is_some_and(|layer| {
        layer.effects.iter().any(|effect| *effect == TransformEffect::Preserves)
    }));
    let mut weights = vec![1.0; data.row_count()];
    weights[0] = 2.0;
    let retargeted =
        prepared.apply_retarget(&preview, &weights, &[VariableId::from_raw(2)], &ctx).unwrap();
    let bytes = prepared.encode_contracted_result(&retargeted, "retarget-report", &ctx).unwrap();
    let after = consume_analysis_result(&bytes).unwrap().contract.unwrap();
    match &after.target.query {
        CausalQueryWire::AverageEffect { target_population, .. } => {
            assert!(matches!(target_population, TargetPopulationWire::RowWeights { .. }));
        }
        other => panic!("expected AverageEffect, got {other:?}"),
    }
    assert_eq!(before.identities.identification, after.identities.identification);
    assert_eq!(before.identities.data_snapshot, after.identities.data_snapshot);
    assert_eq!(before.identities.inference_binding, after.identities.inference_binding);
    assert_ne!(before.identities.target, after.identities.target);
}

#[test]
fn custom_distribution_weights_enter_target_identity() {
    use antecedent_core::{DistributionRef, PopulationRegistry};
    let ctx = ExecutionContext::for_tests(15);
    let (data, dag, query) = confounded_scm(64, 15);
    let mut first = PopulationRegistry::new();
    first.insert_distribution_with_dependence(
        DistributionRef::from_raw(1),
        vec![1.0; data.row_count()],
        [VariableId::from_raw(2)],
    );
    let mut second = PopulationRegistry::new();
    let mut weights = vec![1.0; data.row_count()];
    weights[0] = 3.0;
    second.insert_distribution_with_dependence(
        DistributionRef::from_raw(1),
        weights,
        [VariableId::from_raw(2)],
    );
    let query = query.with_target_population(
        antecedent_core::TargetPopulation::CustomDistribution(DistributionRef::from_raw(1)),
    );
    let a = Study::tabular(data.clone())
        .graph(dag.clone())
        .query(query.clone())
        .population_registry(first)
        .estimator(EstimatorId::Aipw)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let b = Study::tabular(data)
        .graph(dag)
        .query(query)
        .population_registry(second)
        .estimator(EstimatorId::Aipw)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    assert_ne!(a.contract().unwrap().identities.target, b.contract().unwrap().identities.target);
}

#[test]
fn multi_env_prepares_and_exports() {
    let ctx = ExecutionContext::for_tests(16);
    let (series, graph) = lag1_series();
    let multi = antecedent_data::MultiEnvironmentData::try_new(std::sync::Arc::from([
        series.clone(),
        series,
    ]))
    .unwrap();
    let prepared = Study::series_multi(multi)
        .graph(graph)
        .temporal_query(pulse_query())
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let (again, _) = lag1_series();
    let again_multi = antecedent_data::MultiEnvironmentData::try_new(std::sync::Arc::from([
        again.clone(),
        again,
    ]))
    .unwrap();
    let result = prepared.estimate_multi_env(&again_multi, &ctx).unwrap();
    let bytes = prepared.encode_contracted_result(&result, "multi-env", &ctx).unwrap();
    let consumed = consume_analysis_result(&bytes).unwrap();
    assert!(consumed.acceptance.accepts_as_verified_program());
}

/// Weighted posterior over the same three graphs, for identity comparisons.
fn weighted_posterior(weights: [f64; 3]) -> GraphPosterior {
    let direct = set_edge(0, 3, 0, 1, true);
    let adjusted = set_edge(set_edge(set_edge(0, 3, 0, 1, true), 3, 2, 0, true), 3, 2, 1, true);
    let unidentified = set_edge(0, 3, 1, 0, true);
    let marginals = vec![0.0; 9];
    GraphPosterior::new(
        3,
        weights.to_vec(),
        vec![direct, adjusted, unidentified],
        marginals.clone(),
        marginals,
        1.0 / weights.iter().map(|w| w * w).sum::<f64>(),
        InferenceDiagnostics::analytic("identity_weights"),
        0,
    )
    .unwrap()
}

fn posterior_study(
    data: &TabularData,
    posterior: GraphPosterior,
    query: &AverageEffectQuery,
) -> Study {
    Study::tabular(data.clone())
        .graph_posterior(posterior)
        .query(query.clone())
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64)))
        .bootstrap_replicates(0)
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
}

#[test]
fn target_population_moves_target_and_program_not_identification() {
    let ctx = ExecutionContext::for_tests(14);
    let (data, dag, query) = confounded_scm(400, 14);
    let prepared = |query: AverageEffectQuery| {
        Study::tabular(data.clone())
            .graph(dag.clone())
            .query(query)
            .estimator(EstimatorId::Aipw)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .prepare(&ctx)
            .unwrap()
    };
    let all_observed = prepared(query.clone());
    let treated =
        prepared(query.with_target_population(antecedent_core::TargetPopulation::Treated));
    let (a, b) =
        (all_observed.contract().unwrap().identities, treated.contract().unwrap().identities);
    assert_ne!(a.target, b.target);
    assert_ne!(a.program, b.program, "ATE and ATT are different programs");
    assert_eq!(a.identification, b.identification, "the population-free question is shared");
    assert_eq!(a.identification_product, b.identification_product);
    assert_eq!(a.inference_binding, b.inference_binding, "no structure in the inference layer");
    let execution =
        antecedent_io::execution_digest(&antecedent_io::execution_identity_from_context(&ctx))
            .unwrap();
    let key = SemanticDigest::from_bytes([13; 32]);
    assert!(
        RequestIdentity::new(key, a, execution)
            .conflicts_with(&RequestIdentity::new(key, b, execution)),
        "an idempotency key reused for another estimand is a conflict"
    );
}

#[test]
fn graph_posterior_weights_and_atoms_enter_identification() {
    let ctx = ExecutionContext::for_tests(21);
    let (data, dag, query) = confounded_scm(400, 21);
    let _ = dag;
    let base =
        posterior_study(&data, weighted_posterior([0.5, 0.3, 0.2]), &query).prepare(&ctx).unwrap();
    let reweighted =
        posterior_study(&data, weighted_posterior([0.2, 0.3, 0.5]), &query).prepare(&ctx).unwrap();
    let a = base.contract().unwrap().identities;
    let b = reweighted.contract().unwrap().identities;
    assert_ne!(a.identification, b.identification, "posterior weights are premises");
    assert_ne!(a.program, b.program);
    let first = base.estimate(&data, &ctx).unwrap();
    let second = reweighted.estimate(&data, &ctx).unwrap();
    assert!(first.effect().is_nan() && second.effect().is_nan());
    let weights = |result: &StudyResult| {
        result
            .structural_response
            .as_ref()
            .unwrap()
            .atoms
            .iter()
            .map(|atom| (atom.graph_key, atom.weight))
            .collect::<Vec<_>>()
    };
    assert_ne!(weights(&first), weights(&second), "reweighting changes the structural result");
    let claim_id = |prepared: &antecedent::PreparedStudy, result: &StudyResult| {
        let consumed = consume_analysis_result(
            &prepared.encode_contracted_result(result, "mix", &ctx).unwrap(),
        )
        .unwrap();
        assert!(consumed.acceptance.accepts_as_claim(), "{:?}", consumed.acceptance.unresolved);
        consumed.contract.unwrap().claim.unwrap().claim_id
    };
    assert_ne!(
        claim_id(&base, &first),
        claim_id(&reweighted, &second),
        "two mixtures with different weights are two claims"
    );
}

#[test]
fn claim_id_binds_the_data_snapshot_and_the_whole_result() {
    let ctx = ExecutionContext::for_tests(41);
    let (data, dag, query) = confounded_scm(200, 41);
    let (other_data, _, _) = confounded_scm(200, 99);
    let claim_id = |data: &TabularData| {
        let prepared = posterior_study(data, weighted_posterior([0.5, 0.3, 0.2]), &query)
            .prepare(&ctx)
            .unwrap();
        let result = prepared.estimate(data, &ctx).unwrap();
        let bytes = prepared.encode_contracted_result(&result, "snapshot", &ctx).unwrap();
        let consumed = consume_analysis_result(&bytes).unwrap();
        assert!(consumed.acceptance.accepts_as_claim(), "{:?}", consumed.acceptance.unresolved);
        let contract = consumed.contract.unwrap();
        assert_ne!(contract.claim.as_ref().unwrap().kind, "point", "a mixture is not a point");
        (contract.claim.unwrap().claim_id, bytes)
    };
    let _ = dag;
    let (first, bytes) = claim_id(&data);
    let (second, _) = claim_id(&other_data);
    assert_ne!(first, second, "two data snapshots are two claims");
    assert!(
        !SharedEvidenceRef::forwarded_duplicate(
            SemanticDigest::from_bytes(first),
            SemanticDigest::from_bytes(second)
        ),
        "independent executions must not merge as forwarded duplicates"
    );

    // Every body field a verified consume reports is inside the claim id.
    let (decoded, header, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    let section = antecedent_io::decode_analysis_result_contract(&decoded).unwrap().unwrap();
    let edits: [fn(&mut antecedent_io::AnalysisResultWire); 3] = [
        |body| body.standard_error = Some(body.standard_error.unwrap_or(1.0) / 100.0),
        |body| body.estimate = Some(body.estimate.unwrap_or(0.0) + 1.0),
        // Search effort is execution detail the product excludes; only the
        // result digest inside the claim id covers it.
        |body| body.identification.candidates_examined += 1,
    ];
    for edit in edits {
        let mut tampered = body.clone();
        edit(&mut tampered);
        let consumed = consume_analysis_result(&rewrite_contract_section(
            &tampered,
            header.variable_names.clone(),
            "tampered",
            &section,
        ))
        .unwrap();
        assert!(
            !consumed.acceptance.accepts_as_claim(),
            "tampered body accepted: {:?}",
            consumed.acceptance.unresolved
        );
    }
}

#[test]
fn a_tampered_verified_artifact_is_refused() {
    let ctx = ExecutionContext::for_tests(7);
    let (data, dag, query) = confounded_scm(200, 7);
    let prepared = Study::tabular(data.clone())
        .graph(dag)
        .query(query)
        .estimator(EstimatorId::Aipw)
        .bootstrap_replicates(0)
        .refute(RefuteSuite::Full)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let bytes = prepared.encode_contracted_result(&result, "orig", &ctx).unwrap();
    let original = consume_analysis_result(&bytes).unwrap();
    assert!(original.acceptance.accepts_as_claim(), "{:?}", original.acceptance.unresolved);
    let section = original.contract.clone().unwrap();
    let names = original.header.variable_names.clone();

    let edits: [fn(&mut antecedent_io::AnalysisResultContractWire); 6] = [
        |section| {
            let claim = section.claim.as_mut().unwrap();
            claim.identification_domain = "identified".into();
            claim.support_domain = "supported".into();
            claim.evaluated_domain = "evaluated".into();
        },
        |section| section.graph_class = "Pag".into(),
        |section| section.estimator = Some("propensity_weighting".into()),
        |section| section.identifier = Some("frontdoor".into()),
        |section| {
            section.reasoning.support.value.as_mut().unwrap().empirical = "supported".into();
        },
        |section| {
            section.reasoning.assumptions.value.as_mut().unwrap().obligations.clear();
        },
    ];
    for edit in edits {
        let mut tampered = section.clone();
        edit(&mut tampered);
        let consumed = consume_analysis_result(&rewrite_contract_section(
            &original.body,
            names.clone(),
            "tampered",
            &tampered,
        ))
        .unwrap();
        assert!(
            !consumed.acceptance.accepts_as_claim(),
            "tampered contract accepted: {:?}",
            consumed.acceptance.unresolved
        );
        let (_, receipt) = accept_claim(
            &rewrite_contract_section(&original.body, names.clone(), "tampered", &tampered),
            &ConsumerProfile::full(),
        )
        .unwrap();
        assert!(!receipt.equivalent_claim(), "a tampered artifact is not a lossless exchange");
    }

    // Re-encoding a tampered section through the writer is refused too.
    let mut tampered = section;
    tampered.claim.as_mut().unwrap().support_domain = "supported".into();
    assert!(
        antecedent_io::encode_analysis_result_artifact_with_contract(
            &original.body,
            names,
            "tampered",
            Some(&tampered)
        )
        .is_err()
    );
}

#[test]
fn a_failed_overlap_check_does_not_support_the_claim() {
    let ctx = ExecutionContext::for_tests(23);
    let (data, dag, query) = confounded_scm(200, 23);
    // A graph that drops the confounder edge into treatment: the overlap
    // refuter runs and fails on this data.
    let mut misspecified = Dag::with_variables(3);
    misspecified.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    misspecified.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let mut seen_failure = false;
    for graph in [dag, misspecified] {
        let prepared = Study::tabular(data.clone())
            .graph(graph)
            .query(query.clone())
            .estimator(EstimatorId::Aipw)
            .bootstrap_replicates(0)
            .refute(RefuteSuite::Full)
            .build()
            .unwrap()
            .prepare(&ctx)
            .unwrap();
        let result = prepared.estimate(&data, &ctx).unwrap();
        let claim = result.claim(&prepared.contract().unwrap(), &ctx).unwrap();
        let failed: Vec<&str> = result
            .refutations
            .iter()
            .filter(|report| report.refuter.starts_with("overlap.") && !report.passed)
            .map(|report| report.refuter.as_ref())
            .collect();
        if failed.is_empty() {
            assert_ne!(
                claim.domain(ClaimDomainAxis::Support),
                DomainStatus::Contradicted,
                "no failed support check means no contradiction"
            );
        } else {
            seen_failure = true;
            assert_eq!(
                claim.domain(ClaimDomainAxis::Support),
                DomainStatus::Contradicted,
                "a failed overlap check is not support: {failed:?}"
            );
            assert_ne!(claim.domain(ClaimDomainAxis::Support), DomainStatus::Supported);
        }
    }
    assert!(seen_failure, "the misspecified graph must fail an overlap check");
}

#[test]
fn bayesian_backend_prior_and_budgets_move_the_layer_that_owns_them() {
    let ctx = ExecutionContext::for_tests(31);
    let (data, dag, query) = confounded_scm(300, 31);
    let bayes = |config: BayesianConfig| {
        Study::tabular(data.clone())
            .graph(dag.clone())
            .query(query.clone())
            .inference(InferenceMode::Bayesian(config))
            .bootstrap_replicates(0)
            .refute(RefuteSuite::None)
            .build()
            .unwrap()
            .prepare(&ctx)
            .unwrap()
            .contract()
            .unwrap()
            .identities
    };
    let conjugate = bayes(BayesianConfig::conjugate().n_draws(64));
    let laplace = bayes(BayesianConfig::laplace().n_draws(64));
    assert_ne!(
        conjugate.inference_binding, laplace.inference_binding,
        "the posterior backend is an inference binding"
    );
    assert_eq!(conjugate.identification, laplace.identification);
    assert_eq!(conjugate.program, laplace.program, "a backend is not a licensed commitment");
    let scaled = bayes(BayesianConfig::conjugate().n_draws(64).prior_scale(0.01));
    assert_ne!(conjugate.inference_binding, scaled.inference_binding);
    let drawn = bayes(BayesianConfig::conjugate().n_draws(65));
    assert_ne!(conjugate.inference_binding, drawn.inference_binding);
    let explicit = bayes(
        BayesianConfig::conjugate()
            .n_draws(64)
            .prior(antecedent_prob::PriorSet::weakly_informative(4)),
    );
    assert_ne!(conjugate.inference_binding, explicit.inference_binding, "prior contents bind");

    // Execution budgets and determinism are execution lineage, not program.
    let identity = antecedent_io::execution_identity_from_context(&ctx);
    let mut adaptive = ExecutionContext::for_tests(31);
    adaptive.adaptive_bootstrap = antecedent_core::AdaptiveBootstrapBudget::enabled_default();
    let mut fast = ExecutionContext::for_tests(31);
    fast.determinism = antecedent_core::Determinism::PreferFast;
    let digest = |wire: &antecedent_io::ExecutionIdentityWire| {
        antecedent_io::execution_digest(wire).unwrap()
    };
    assert_ne!(
        digest(&identity),
        digest(&antecedent_io::execution_identity_from_context(&adaptive)),
        "an adaptive early-stop budget changes execution lineage"
    );
    assert_ne!(
        digest(&identity),
        digest(&antecedent_io::execution_identity_from_context(&fast)),
        "determinism policy changes execution lineage"
    );
}

#[test]
fn response_bandwidth_moves_the_inference_binding_not_the_program() {
    let (data, dag, _) = confounded_scm(200, 37);
    let curve = |bandwidth: Option<f64>| {
        let options = antecedent_estimate::ContinuousResponseOptions {
            bandwidth,
            ..antecedent_estimate::ContinuousResponseOptions::default()
        };
        Study::tabular(data.clone())
            .graph(dag.clone())
            .query(CausalQuery::Response(ResponseQuery::new(ResponseFunctional::PointDerivative {
                treatment: VariableId::from_raw(0),
                outcome: VariableId::from_raw(1),
                at: 0.5,
                order: 1,
                scale: antecedent_core::DerivativeScale::Identity,
            })))
            .response_options(options)
            .bootstrap_replicates(0)
            .refute(RefuteSuite::None)
            .build()
            .unwrap()
            .inspect()
            .unwrap()
            .identities
    };
    let default = curve(None);
    let narrow = curve(Some(0.25));
    let wide = curve(Some(0.75));
    assert_ne!(narrow.inference_binding, wide.inference_binding, "bandwidth is a numeric knob");
    assert_ne!(default.inference_binding, narrow.inference_binding);
    assert_eq!(narrow.program, wide.program, "a bandwidth is not a licensed commitment");
    assert_eq!(narrow.identification, wide.identification);
    assert_eq!(narrow.target, wide.target);
}

#[test]
fn unlicensed_operation_blocker_ids_come_from_constants_not_prose() {
    let (data, _, query) = conditional_effect_fixture();
    let mut admg = Admg::with_variables(3);
    admg.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    admg.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    let inspected = Study::tabular(data)
        .graph(admg)
        .query(CausalQuery::ConditionalEffect(query))
        .bootstrap_replicates(0)
        .inspect()
        .unwrap();
    for operation in [OperationKind::Retarget, OperationKind::Export, OperationKind::RankDesigns] {
        let report = inspected.capability_for(operation);
        assert_eq!(
            report.primary_blocker_id(),
            Some("operation.unlicensed"),
            "{operation:?} on a refused cell"
        );
        let message = report.blockers.first().map(|blocker| blocker.reason.to_string()).unwrap();
        assert_eq!(
            antecedent::CausalError::Unsupported { message: Box::leak(message.into_boxed_str()) }
                .blocker_id()
                .map(|blocker| blocker.id.to_string()),
            Some("operation.unlicensed".into()),
            "the refusal a capability report prints maps back to the same id"
        );
    }
    // A message that merely mentions the same words is a different refusal.
    let reworded = antecedent::CausalError::Unsupported {
        message: "this retarget would need a licensed prepared contract first",
    };
    assert_eq!(reworded.blocker_id().map(|blocker| blocker.id.to_string()), None);
}
