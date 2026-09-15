//! 1.10 contracts-first: identities, inspection, transformation preview, claims.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::float_cmp, clippy::many_single_char_names)]

use std::sync::Arc;

use antecedent::state::{
    DataBatchRef, InterventionRecord, apply_state_event, new_antecedent_state,
    publish_recomputed_results, result_lineage_fingerprint,
};
use antecedent::{
    BayesianConfig, EstimatorId, IdentifierId, InferenceMode, IntoGraphInput, OperationKind,
    OperationReadiness, RefuteSuite, SemanticApplicability, Study, StudyResult,
};
use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSource, AssumptionStatus,
    AverageEffectQuery, CacheBudget, CausalQuery, CausalRng, CausalSchemaBuilder, ClaimDomainAxis,
    ClaimKind, ClaimOperation, ConditionalEffectQuery, ConsumerProfile, ContinuousDomain,
    CounterfactualQuery, DerivedClaimOutcome, DomainStatus, ExecutionContext, GridSpec,
    HostOperation, IdentificationStatus, Intervention, InterventionSequence,
    InterventionalDistributionQuery, Lag, MeasurementSpec, MediationContrast, MediationQuery,
    ObligationKind, ObservationAssumption, ObservationSpec, PathSpecificEffectQuery, ProgressSink,
    ResponseFunctional, ResponseIdentification, ResponseQuery, ResponseValue, RoleHint,
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
    Admg, Cpdag, Dag, DenseNodeId, Endpoint, MarkedEdge, MiddleMark, Pag, TemporalCpdag,
    TemporalDag, TemporalPag, ensure_lagged,
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

fn confounded_scm(n: usize, seed: u64) -> (TabularData, Dag, AverageEffectQuery) {
    let mut rng = CausalRng::from_seed(seed);
    let mut t = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    for _ in 0..n {
        let u1 = rng.next_f64().max(1e-12);
        let u2 = rng.next_f64();
        let zi = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        let logit = -0.4 + 0.9 * zi;
        let p = 1.0 / (1.0 + (-logit).exp());
        let ti = if rng.next_f64() < p { 1.0 } else { 0.0 };
        let e = (-2.0 * rng.next_f64().max(1e-12).ln()).sqrt()
            * (2.0 * std::f64::consts::PI * rng.next_f64()).cos()
            * 0.4;
        z.push(zi);
        t.push(ti);
        y.push(2.0 * ti + zi + e);
    }
    let mut b = CausalSchemaBuilder::new();
    b.add_variable(
        "t",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "y",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "z",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::Context),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    let schema = b.build().unwrap();
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(t), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(y), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(2), Arc::from(z), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let data = TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    (data, dag, query)
}

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
    assert_ne!(contract.identities.target, contract.identities.program);
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
    assert!(preview.binds_program(contract.identities.program));
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
    assert_ne!(inspected.identities.program, contract.identities.program);
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
    for changed in [masked, weighted] {
        let contract = study(changed, dag.clone(), query.clone()).inspect().unwrap();
        assert_eq!(baseline.identities.program, contract.identities.program);
        assert_ne!(baseline.identities.data_snapshot, contract.identities.data_snapshot);
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
    assert!(consumed.acceptance.accepts_as_verified_program());
    assert_eq!(
        consumed.contract.as_ref().map(|section| section.identities.program),
        Some(*contract.identities.program.as_bytes())
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
    assert_eq!(section.identities.program, *contract.identities.program.as_bytes());
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
    assert!(consumed.acceptance.accepts_as_verified_program());
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
    assert!(preview.binds_program(contract.identities.program));
    let result = estimate(&prepared);
    let claim = result.claim(&contract, ctx).unwrap();
    assert_eq!(claim.identities.program, contract.identities.program);
    assert_eq!(claim.identities.target, contract.identities.target);
    let bytes = prepared.encode_contracted_result(&result, artifact_id, ctx).unwrap();
    let consumed = consume_analysis_result(&bytes).unwrap();
    assert!(consumed.acceptance.accepts_as_verified_program());
    let section = consumed.contract.as_ref().expect("verified contract");
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

fn mixture_graph_posterior() -> GraphPosterior {
    let weights = [0.5, 0.3, 0.2];
    let direct = set_edge(0, 3, 0, 1, true);
    let adjusted = set_edge(set_edge(set_edge(0, 3, 0, 1, true), 3, 2, 0, true), 3, 2, 1, true);
    let unidentified = set_edge(0, 3, 1, 0, true);
    let mut marginals = vec![0.0; 9];
    marginals[1] = weights[0] + weights[1];
    marginals[3] = weights[2];
    marginals[6] = weights[1];
    marginals[7] = weights[1];
    GraphPosterior::new(
        3,
        weights.to_vec(),
        vec![direct, adjusted, unidentified],
        marginals.clone(),
        marginals,
        1.0 / weights.iter().map(|w| w * w).sum::<f64>(),
        InferenceDiagnostics::analytic("known_truth_mixtures"),
        0,
    )
    .unwrap()
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
    assert!((result.effect() - 2.0).abs() < 0.15);
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
    assert_eq!(claim.kind, ClaimKind::Mixture);
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
    let labels: std::collections::HashMap<_, _> =
        executed_functional_labels(&consumed.contract.as_ref().unwrap().target.query)
            .into_iter()
            .collect();
    assert_eq!(labels["query_kind"], "temporal_effect");
    assert!(labels["temporal_coordinates"].contains("horizon:1"));
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
    assert!(observation.observation.iter().any(|tag| tag == "outcome_independent_given"));
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
    assert_eq!(binding.prior_mapping.as_deref(), Some("prior_artifact"));
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
    assert!(consumed.acceptance.accepts_as_verified_program());
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
            assert!(consumed.acceptance.accepts_as_verified_program());
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

fn identified_pag_series(pin: &serde_json::Value) -> TimeSeriesData {
    let n = usize::try_from(pin["n"].as_u64().unwrap()).unwrap();
    let names: Vec<&str> =
        pin["columns"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    let mut cols = vec![vec![0.0; n]; names.len()];
    let [t, y, z, m, v] = [0, 1, 2, 3, 4];
    for i in 0..n {
        let x = i as f64;
        cols[z][i] = (0.37 * x).sin() + 0.5 * (1.3 * x).cos();
        cols[t][i] = 0.6 * cols[z][i] + 0.8 * (0.23 * x + 0.4).sin();
        cols[v][i] = 0.5 * cols[t][i] + (0.41 * x).cos();
        cols[m][i] = 0.7 * cols[z][i] + 0.6 * (0.29 * x + 0.2).cos();
        if i > 0 {
            cols[y][i] = 1.0 + 2.0 * cols[t][i - 1] + 1.5 * cols[m][i - 1] + 0.3 * (0.53 * x).sin();
        }
    }
    TimeSeriesData::from_f64_columns(names.iter().copied().zip(cols.iter().map(Vec::as_slice)), 1)
        .unwrap()
}

fn identified_pag(pin: &serde_json::Value) -> TemporalPag {
    let names: Vec<&str> =
        pin["columns"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    let mark = |m: &str| match m {
        "tail" => Endpoint::Tail,
        "arrow" => Endpoint::Arrow,
        "circle" => Endpoint::Circle,
        other => panic!("unknown endpoint {other}"),
    };
    let node = |g: &mut TemporalPag, name: &str, lag: &serde_json::Value| {
        let var = u32::try_from(names.iter().position(|c| *c == name).unwrap()).unwrap();
        let lag = Lag::from_raw(u32::try_from(lag.as_u64().unwrap()).unwrap());
        g.add_lagged(VariableId::from_raw(var), lag).unwrap()
    };
    let mut graph = TemporalPag::empty();
    for edge in pin["marked_edges"].as_array().unwrap() {
        let a = node(&mut graph, edge[0].as_str().unwrap(), &edge[1]);
        let b = node(&mut graph, edge[2].as_str().unwrap(), &edge[3]);
        graph
            .insert_marked(MarkedEdge {
                a,
                b,
                at_a: mark(edge[4].as_str().unwrap()),
                at_b: mark(edge[5].as_str().unwrap()),
                middle: MiddleMark::Empty,
            })
            .unwrap();
    }
    graph
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
    assert!(observation.observation.iter().any(|tag| tag == "outcome_independent_given"));
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
    expected_tag: &str,
    artifact_id: &str,
) {
    let ctx = ExecutionContext::for_tests(1);
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
    assert_eq!(binding.prior_mapping.as_deref(), Some(expected_tag));
}

#[test]
fn licensed_family_prior_mapping_identical_subspace_does_not_create_identification() {
    let (series, graph) = lag1_series();
    consume_mapped_prior(
        &series,
        graph,
        PriorMapping::IdenticalCoefficientSubspace,
        "identical_coefficient_subspace",
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
        "effect_functional:ate",
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
        "named_parameters:coef_x@lag1->coef_x@lag1",
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
    assert!(observation.observation.iter().any(|tag| tag == "right_censored"));
    assert!(observation.observation.iter().any(|tag| tag == "independent_given"));
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
    assert_eq!(binding.prior_mapping.as_deref(), Some("prior_artifact"));
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
        &[
            (pulse_q, result_lineage_fingerprint(&pulse_id), 32),
            (curve_q, result_lineage_fingerprint(&curve_id), 32),
        ],
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
    let stale_publish = publish_recomputed_results(
        &mut state,
        v0,
        &[(pulse_q, result_lineage_fingerprint(&pulse_id), 32)],
    )
    .unwrap_err();
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
        &[
            (pulse_q, result_lineage_fingerprint(&pulse_after), 32),
            (curve_q, result_lineage_fingerprint(&curve_after), 32),
        ],
    )
    .unwrap();
    assert!(!state.is_stale(pulse_q));
    assert_ne!(result_lineage_fingerprint(&pulse_id), result_lineage_fingerprint(&pulse_after));

    let next_data = state.data_catalog.version.next();
    apply_state_event(&mut state, antecedent::state::StateEvent::ReplaceData(next_data)).unwrap();
    assert!(state.is_stale(pulse_q) && state.is_stale(curve_q));
    let after_replace = state.version;
    publish_recomputed_results(
        &mut state,
        after_replace,
        &[
            (pulse_q, result_lineage_fingerprint(&pulse_after), 32),
            (curve_q, result_lineage_fingerprint(&curve_after), 32),
        ],
    )
    .unwrap();

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
        pulse_after.data_snapshot,
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
    assert!(sender.acceptance.accepts_as_verified_program());
    assert!(lossless.equivalent_claim());
    let host = project_claim_host(&sender);
    assert!(host.accepts_as_claim);
    assert_ne!(host.value, serde_json::Value::Null);
    assert_eq!(host.identification_domain, serde_json::Value::String("identified".into()));

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
