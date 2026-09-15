//! 1.10 contracts-first: identities, inspection, transformation preview, claims.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::float_cmp, clippy::many_single_char_names)]

use std::sync::Arc;

use antecedent::{EstimatorId, IdentifierId, RefuteSuite, Study};
use antecedent_core::{
    AverageEffectQuery, CausalRng, CausalSchemaBuilder, ClaimKind, ExecutionContext,
    IdentificationStatus, MeasurementSpec, ProgressSink, RoleHint, SlotAvailability, SmallRoleSet,
    TransformIntent, ValueType, VariableId,
};
use antecedent_io::query_wire::{CausalQueryWire, InterventionWire, TargetPopulationWire, ValueWire};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TableView, TabularData, ValidityBitmap,
};
use antecedent_graph::{Admg, Dag, DenseNodeId};

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
        consumed.contract.as_ref().unwrap().reasoning.support.value.as_ref().and_then(|slot| {
            slot.matrix_coordinate.as_deref()
        }),
        Some("AverageEffect:Dag:explicit:Frequentist:full")
    );
}

#[test]
fn execution_identity_follows_the_estimate_context() {
    let (data, dag, query) = confounded_scm(48, 59);
    let prepared = study(data.clone(), dag, query).prepare(&ExecutionContext::for_tests(1)).unwrap();
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
