//! Portable CATE prediction: executed fit counts, SCM anchors and process isolation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{RefuteSuite, Study};
use antecedent_core::{AverageEffectQuery, ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_estimate::DrLearner;
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_learn::fit_counts::count_resolved_fits;

fn fixture(effect: f64) -> (TabularData, Dag) {
    let n = 1600_u32;
    let mut rng = antecedent_core::CausalRng::from_seed(19);
    let mut z = Vec::new();
    let mut t = Vec::new();
    let mut y = Vec::new();
    for i in 0..n {
        let x = f64::from(i % 21) / 10.0 - 1.0;
        let a = f64::from(rng.next_f64() < 0.5);
        z.push(x);
        t.push(a);
        y.push(a * (effect + 0.4 * x) + x);
    }
    let data = TabularData::from_f64_columns([
        ("z", z.as_slice()),
        ("t", t.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap();
    let mut graph = Dag::with_variables(3);
    for (a, b) in [(0, 1), (0, 2), (1, 2)] {
        graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    (data, graph)
}

fn fitted_bytes(effect: f64) -> Vec<u8> {
    let (data, graph) = fixture(effect);
    let ctx = ExecutionContext::for_tests(3);
    let prepared = Study::tabular(data.clone())
        .graph(graph)
        .query(AverageEffectQuery::binary_ate(VariableId::from_raw(1), VariableId::from_raw(2)))
        .estimator(DrLearner::new())
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    prepared.encode_contracted_result(&result, "portable-cate", &ctx).unwrap()
}

fn fit_bytes(effect: f64) -> (Vec<u8>, u64) {
    count_resolved_fits(|| fitted_bytes(effect))
}

fn predict_bytes(bytes: &[u8]) -> (Vec<f64>, String) {
    let consumed = antecedent_io::replay_analysis_result(bytes).unwrap();
    // The serialized predictor remains usable for model-point inference, but
    // this body does not independently replay the fitted causal estimator.
    assert!(!consumed.acceptance.accepts_as_replayed_program());
    assert!(!consumed.acceptance.replayed_references);
    assert!(
        consumed
            .acceptance
            .unresolved
            .iter()
            .any(|reason| reason.as_ref() == "dependencies.checked_estimator_operation"),
        "{:?}",
        consumed.acceptance.unresolved
    );
    let claim = consumed.contract.as_ref().unwrap().claim.as_ref().unwrap();
    let parent = antecedent_core::to_hex(&claim.claim_id);
    let model = consumed.body.fitted_effect.unwrap();
    let ctx = ExecutionContext::for_tests(3);
    let values = model.predict(&[VariableId::from_raw(0)], &[&[-1.0, 0.0, 1.0]], 3, &ctx).unwrap();
    // Independent SCM truth E[Y(1)-Y(0)|z] = 2 + 0.4 z. The outcome is noiseless and exactly
    // linear in (t, t*z, z), so the fitted CATE deviates from the structural value only by
    // fitting error (observed <= 7.1e-5); 5e-4 is about seven times that and excludes a wrong
    // slope (0.4 -> 0.3 moves the z=+-1 anchors by 0.1) or intercept.
    for (z, (value, truth)) in [-1.0_f64, 0.0, 1.0].iter().zip(values.iter().zip([1.6, 2.0, 2.4])) {
        let closed_form = 2.0 + 0.4 * z;
        assert!((closed_form - truth).abs() < 1e-12);
        assert!((value - truth).abs() < 5e-4, "CATE SCM anchor at z={z}: {value} vs {truth}");
    }
    assert!(model.predict(&[VariableId::from_raw(1)], &[&[0.0]], 1, &ctx).is_err());
    assert!(model.predict(&[VariableId::from_raw(0)], &[&[f64::NAN]], 1, &ctx).is_err());
    (values, parent)
}

#[test]
fn portable_predictor_zero_refits_and_fresh_process_match_full_run() {
    let (bytes, fits) = fit_bytes(2.0);
    assert!(fits > 1, "resolved nuisance and final-stage models actually fit");
    assert_eq!(bytes, fitted_bytes(2.0), "instrumentation preserves the complete claim bytes");
    let (one, resume_fits) = count_resolved_fits(|| predict_bytes(&bytes));
    assert_eq!(resume_fits, 0);
    let (rerun, rerun_fits) = fit_bytes(2.0);
    assert_eq!(rerun_fits, fits);
    assert_eq!(one, predict_bytes(&rerun));
    let directory =
        std::env::temp_dir().join(format!("antecedent-portable-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join("model.ant");
    std::fs::write(&path, &bytes).unwrap();
    // The child must select this very test: under a suite crate it is a module member, so its
    // libtest name carries the module path (a bare name would match nothing and "pass").
    let prefix =
        module_path!().split_once("::").map_or(String::new(), |(_, rest)| format!("{rest}::"));
    let test_name = format!("{prefix}portable_predictor_child");
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", &test_name, "--nocapture"])
        .env("ANTECEDENT_PORTABLE_TEST_ARTIFACT", &path)
        .output()
        .unwrap();
    std::fs::remove_dir_all(&directory).unwrap();
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("running 1 test"),
        "the child selected no test: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    // Refit requires compatible supplied raw data; changing outcomes cannot reuse this parent.
    let (changed, changed_fits) = fit_bytes(3.0);
    assert!(changed_fits > 1);
    let changed = antecedent_io::replay_analysis_result(&changed).unwrap();
    let changed_parent =
        antecedent_core::to_hex(&changed.contract.unwrap().claim.unwrap().claim_id);
    assert_ne!(changed_parent, one.1);
}

#[test]
fn portable_predictor_child() {
    let Some(path) = std::env::var_os("ANTECEDENT_PORTABLE_TEST_ARTIFACT") else { return };
    let bytes = std::fs::read(path).unwrap();
    let (_, fits) = count_resolved_fits(|| predict_bytes(&bytes));
    assert_eq!(fits, 0);
}

/// Replay of the cross-fitted DR fit from the full-source export.
///
/// A consumer holding the complete original table can run the declared fit again; only a
/// replay that reproduces the body clears the missing estimator operation. Every digest a
/// forger can compute is recomputed in the negative tests, so a refusal is the replay's
/// verdict and not a stale checksum.
mod source_replay {
    use std::sync::{Arc, OnceLock};

    use antecedent::analysis::cohort_decision::FittedEffectModel;
    use antecedent_core::CancellationToken;
    use antecedent_data::TableView;
    use antecedent_estimate::OverlapPolicy;
    use antecedent_io_core::{
        AnalysisResultConsumption, AnalysisResultContractWire, AnalysisResultHeader,
        AnalysisResultWire, CONTRACT_SECTION, CheckedDrLoweringWire, CheckedLearnerWire,
        ClaimIdentityWire, CompressPolicy, DrReplayRefusal, EstimatorSpecWire,
        SOURCE_WITNESS_SECTION, SourceReplayBounds, SourceWitnessLimits, SourceWitnessWire,
        claim_digest, contract_seal, data_snapshot_digest, decode_analysis_result_artifact,
        decode_analysis_result_contract, decode_source_witness_section,
        encode_analysis_result_artifact, encode_analysis_result_artifact_with_contract_and_source,
        inference_binding_digest, pack_section_shared, program_digest, replay_analysis_result,
        replay_analysis_result_with_source, result_digest, score_reuse_digest, to_cbor,
        verify_checked_dr_replay,
    };
    use antecedent_learn::{LearnerSpec, LinearSpec, PredictionMap, RidgeSpec};

    use super::*;

    const SEED: u64 = 3;
    const OPERATION: &str = "dependencies.checked_estimator_operation";
    const MISMATCH: &str = "dependencies.checked_dr_replay_mismatch";
    const SOURCE: &str = "dependencies.checked_dr_replay_source";
    const RESOURCE: &str = "dependencies.checked_dr_replay_resource";
    const BINDING: &str = "program.checked_dr_binding";
    /// Reasons that would say a recomputable digest went stale instead of the replay failing.
    const STALE: [&str; 7] = [
        "claim.id",
        "claim.value",
        "contract.seal",
        "identities.program",
        "identities.data_snapshot",
        "identities.inference_binding",
        "identities.score_reuse",
    ];

    struct Export {
        compact: Vec<u8>,
        sourced: Vec<u8>,
    }

    fn ctx() -> ExecutionContext {
        ExecutionContext::for_tests(SEED)
    }

    fn query() -> AverageEffectQuery {
        AverageEffectQuery::binary_ate(VariableId::from_raw(1), VariableId::from_raw(2))
    }

    fn export_with(
        data: &TabularData,
        graph: Dag,
        query: AverageEffectQuery,
        learner: DrLearner,
    ) -> Export {
        let ctx = ctx();
        let prepared = Study::tabular(data.clone())
            .graph(graph)
            .query(query)
            .estimator(learner)
            .refute(RefuteSuite::None)
            .build()
            .unwrap()
            .prepare(&ctx)
            .unwrap();
        let result = prepared.estimate(data, &ctx).unwrap();
        Export {
            compact: prepared.encode_contracted_result(&result, "compact", &ctx).unwrap(),
            sourced: prepared
                .encode_contracted_result_with_source(&result, data, "sourced", &ctx)
                .unwrap(),
        }
    }

    fn base() -> &'static Export {
        static BASE: OnceLock<Export> = OnceLock::new();
        BASE.get_or_init(|| {
            let (data, graph) = fixture(2.0);
            export_with(&data, graph, query(), DrLearner::new())
        })
    }

    #[derive(Clone)]
    struct Parts {
        header: AnalysisResultHeader,
        body: AnalysisResultWire,
        contract: AnalysisResultContractWire,
        witness: SourceWitnessWire,
    }

    fn parts(bytes: &[u8]) -> Parts {
        let (artifact, header, body) = decode_analysis_result_artifact(bytes).unwrap();
        let contract = decode_analysis_result_contract(&artifact).unwrap().unwrap();
        let witness = decode_source_witness_section(&artifact, &SourceWitnessLimits::default())
            .unwrap()
            .unwrap();
        Parts { header, body, contract, witness }
    }

    /// The artifact bytes of `parts`: the contract section and, when asked, the witness.
    fn assemble(parts: &Parts, with_witness: bool) -> Vec<u8> {
        let mut artifact = encode_analysis_result_artifact(
            &parts.body,
            parts.header.variable_names.clone(),
            "tampered",
        )
        .unwrap();
        let mut sections = vec![(CONTRACT_SECTION, to_cbor(&parts.contract).unwrap())];
        if with_witness {
            sections.push((SOURCE_WITNESS_SECTION, to_cbor(&parts.witness).unwrap()));
        }
        for (id, payload) in sections {
            let (descriptor, section) =
                pack_section_shared(id, "application/cbor", payload.into(), CompressPolicy::Auto);
            artifact.manifest.sections.push(descriptor);
            artifact.sections.push(section);
        }
        let mut bytes = Vec::new();
        artifact.write_to(&mut bytes).unwrap();
        bytes
    }

    /// Recompute the program identity, the seal, the claimed value and the claim id.
    fn reseal(contract: &mut AnalysisResultContractWire, body: &AnalysisResultWire) {
        contract.identities.program =
            *program_digest(contract.program.as_ref().unwrap()).unwrap().as_bytes();
        contract.seal = contract_seal(
            &contract.identities,
            &contract.reasoning,
            &contract.graph_class,
            &contract.structure_source,
            contract.identifier.as_deref(),
            contract.estimator.as_deref(),
        )
        .unwrap();
        let claim = contract.claim.as_mut().unwrap();
        claim.value_bits = body.estimate.map(f64::to_bits);
        claim.claim_id = *claim_digest(&ClaimIdentityWire::new(
            contract.seal,
            claim,
            result_digest(body).unwrap(),
        ))
        .unwrap()
        .as_bytes();
    }

    /// Point the snapshot at a different table and recompute every identity keyed by it.
    fn rebind_snapshot(parts: &mut Parts, content: [u8; 32]) {
        let snapshot = parts.contract.data_snapshot.as_mut().unwrap();
        snapshot.partitions[0].content = content;
        let digest = *data_snapshot_digest(snapshot).unwrap().as_bytes();
        parts.contract.identities.data_snapshot = digest;
        if let Some(score) = parts.contract.score_reuse.as_mut() {
            score.data_snapshot = digest;
            parts.contract.identities.score_reuse =
                Some(*score_reuse_digest(score).unwrap().as_bytes());
        }
        reseal(&mut parts.contract, &parts.body);
    }

    #[allow(
        clippy::redundant_closure_for_method_calls,
        reason = "the bounded reference fixture preserves its exact calculations and local structure"
    )]
    fn unresolved(bytes: &[u8]) -> Vec<String> {
        replay_analysis_result(bytes)
            .unwrap()
            .acceptance
            .unresolved
            .iter()
            .map(|reason| reason.to_string())
            .collect()
    }

    fn has(reasons: &[String], reason: &str) -> bool {
        reasons.iter().any(|candidate| candidate == reason)
    }

    #[allow(
        clippy::redundant_closure_for_method_calls,
        reason = "the bounded reference fixture preserves its exact calculations and local structure"
    )]
    fn assert_refused_by_replay(bytes: &[u8], verdict: &str) {
        let consumed = replay_analysis_result(bytes).unwrap();
        let reasons: Vec<String> =
            consumed.acceptance.unresolved.iter().map(|reason| reason.to_string()).collect();
        assert!(has(&reasons, verdict), "{reasons:?}");
        assert!(has(&reasons, OPERATION), "{reasons:?}");
        assert!(!consumed.acceptance.accepts_as_claim());
        for stale in STALE {
            assert!(!has(&reasons, stale), "{stale} in {reasons:?}");
        }
    }

    fn declared(consumed: &AnalysisResultConsumption) -> Option<CheckedDrLoweringWire> {
        consumed.contract.as_ref()?.program.as_ref()?.checked_dr_lowering.clone()
    }

    #[test]
    #[allow(
        clippy::redundant_closure_for_method_calls,
        reason = "the bounded reference fixture preserves its exact calculations and local structure"
    )]
    fn a_full_source_export_is_replayed_and_verified_while_the_compact_export_is_not() {
        let export = base();
        let compact = replay_analysis_result(&export.compact).unwrap();
        assert!(has(
            &compact.acceptance.unresolved.iter().map(|r| r.to_string()).collect::<Vec<_>>(),
            OPERATION
        ));
        assert!(!compact.acceptance.accepts_as_claim());

        let sourced = replay_analysis_result(&export.sourced).unwrap();
        assert!(sourced.acceptance.unresolved.is_empty(), "{:?}", sourced.acceptance.unresolved);
        assert!(sourced.acceptance.accepts_as_claim());
        // The body and the contract are those of the compact export; the witness is the only
        // difference, and the declaration sits in the program of both.
        assert_eq!(sourced.body, compact.body);
        assert_eq!(sourced.contract, compact.contract);
        let lowering = declared(&sourced).expect("a supported execution declares its operation");
        assert_eq!(declared(&compact), Some(lowering.clone()));

        // The declaration is the executed configuration of `DrLearner::new()`, read per field.
        assert_eq!(
            (lowering.treatment, lowering.outcome, lowering.adjustment.clone()),
            (1, 2, vec![0])
        );
        assert_eq!((lowering.population.as_str(), lowering.folds), ("all_observed", 5));
        assert_eq!(
            lowering.outcome_learner,
            CheckedLearnerWire::from_spec(LearnerSpec::Ridge(RidgeSpec { lambda: 1.0 })).unwrap()
        );
        assert_eq!(
            (lowering.overlap_clip_bits, lowering.overlap_trim_bits, lowering.overlap_declared),
            (Some(0.01_f64.to_bits()), None, false)
        );
        assert!(lowering.active_level_above_control && lowering.intercept);

        // Independent oracle from the witnessed table: every row's potential outcomes are
        // known, so the sample average effect is `2 + 0.4 mean(z)`.
        let witness = parts(&export.sourced).witness;
        let z = &witness.columns[0].values;
        let mean_z = z.iter().sum::<f64>() / z.len() as f64;
        let published = sourced.body.estimate.unwrap();
        assert!((published - (2.0 + 0.4 * mean_z)).abs() < 5e-3, "{published}");

        // The verified claim is a usable parent; the compact one is not.
        assert!(FittedEffectModel::load(&export.sourced).is_ok());
        match FittedEffectModel::load(&export.compact) {
            Err(refusal) => assert_eq!(
                refusal.detail,
                antecedent_core::reason_details::cohort_decision::PARENT_NOT_VERIFIED
            ),
            Ok(_) => panic!("a compact export must not be a verified parent"),
        }
    }

    #[test]
    fn the_declaration_follows_the_executed_configuration_and_a_trimmed_fit_replays() {
        let (data, graph) = fixture(2.0);
        let learner = DrLearner::new()
            .with_folds(3)
            .with_outcome(LearnerSpec::Linear(LinearSpec::default()))
            .with_overlap(OverlapPolicy::RequireDiagnostics { clip: Some(0.05), trim: Some(0.02) });
        let export = export_with(&data, graph, query(), learner);
        let sourced = replay_analysis_result(&export.sourced).unwrap();
        let lowering = declared(&sourced).expect("a configured fit declares its operation");
        assert_eq!(lowering.folds, 3);
        assert_eq!(
            lowering.outcome_learner,
            CheckedLearnerWire::from_spec(LearnerSpec::Linear(LinearSpec::default())).unwrap()
        );
        assert_eq!(
            (lowering.overlap_clip_bits, lowering.overlap_trim_bits, lowering.overlap_declared),
            (Some(0.05_f64.to_bits()), Some(0.02_f64.to_bits()), true)
        );
        assert_eq!(sourced.body.crossfit_folds, Some(3));
        assert_eq!(sourced.body.crossfit_seed, Some(SEED));
        assert!(sourced.acceptance.unresolved.is_empty(), "{:?}", sourced.acceptance.unresolved);
        assert!(has(&unresolved(&export.compact), OPERATION));
    }

    #[test]
    #[allow(
        clippy::implicit_clone,
        clippy::redundant_closure_for_method_calls,
        reason = "the bounded reference fixture preserves its exact calculations and local structure"
    )]
    fn a_treatment_coded_outside_zero_and_one_declares_nothing_and_stays_unresolved() {
        let (data, graph) = fixture(2.0);
        let z = data.float64_values(VariableId::from_raw(0)).unwrap().to_vec();
        let t: Vec<f64> =
            data.float64_values(VariableId::from_raw(1)).unwrap().iter().map(|a| 2.0 * a).collect();
        let y = data.float64_values(VariableId::from_raw(2)).unwrap().to_vec();
        let coded = TabularData::from_f64_columns([
            ("z", z.as_slice()),
            ("t", t.as_slice()),
            ("y", y.as_slice()),
        ])
        .unwrap();
        let levels = AverageEffectQuery::with_levels(
            VariableId::from_raw(1),
            VariableId::from_raw(2),
            0.0,
            2.0,
        );
        let export = export_with(&coded, graph, levels, DrLearner::new());
        let sourced = replay_analysis_result(&export.sourced).unwrap();
        assert_eq!(declared(&sourced), None);
        let reasons: Vec<String> =
            sourced.acceptance.unresolved.iter().map(|r| r.to_string()).collect();
        assert!(has(&reasons, OPERATION), "{reasons:?}");
        assert!(!sourced.acceptance.accepts_as_claim());
    }

    /// Edit the witnessed columns. Left unsealed, the snapshot no longer binds the table.
    /// With the snapshot, score-reuse, program and claim digests recomputed, the table binds
    /// and the replay is what refuses the body.
    fn assert_witness_edit_refused(edit: impl Fn(&mut Vec<Vec<f64>>)) {
        let original = parts(&base().sourced);
        let names = original.header.variable_names.clone();
        let mut columns: Vec<Vec<f64>> =
            original.witness.columns.iter().map(|column| column.values.clone()).collect();
        edit(&mut columns);
        let mut edited = original.witness.clone();
        for (wire, values) in edited.columns.iter_mut().zip(&columns) {
            wire.values.clone_from(values);
        }

        let mut unsealed = original.clone();
        unsealed.witness = edited.clone();
        assert_refused_by_replay(&assemble(&unsealed, true), SOURCE);

        let content = TabularData::from_f64_columns(
            names.iter().zip(&columns).map(|(name, values)| (name.as_str(), values.as_slice())),
        )
        .unwrap()
        .storage()
        .content_digest();
        let mut resealed = original;
        resealed.witness = edited;
        rebind_snapshot(&mut resealed, content);
        assert_refused_by_replay(&assemble(&resealed, true), MISMATCH);
    }

    #[test]
    fn one_changed_outcome_value_in_the_witness_is_refused() {
        assert_witness_edit_refused(|columns| columns[2][100] += 0.5);
    }

    #[test]
    fn one_changed_treatment_value_in_the_witness_is_refused() {
        assert_witness_edit_refused(|columns| columns[1][11] = 1.0 - columns[1][11]);
    }

    #[test]
    fn one_changed_covariate_value_in_the_witness_is_refused() {
        assert_witness_edit_refused(|columns| columns[0][7] += 0.3);
    }

    #[test]
    fn two_swapped_rows_in_the_witness_are_refused() {
        assert_witness_edit_refused(|columns| {
            for column in columns.iter_mut() {
                column.swap(5, 912);
            }
        });
    }

    #[test]
    #[allow(
        clippy::implicit_clone,
        reason = "the bounded reference fixture preserves its exact calculations and local structure"
    )]
    fn an_unused_extra_column_is_carried_and_verified_and_a_dropped_one_is_refused() {
        let (data, _) = fixture(2.0);
        let spare: Vec<f64> = (0..data.row_count()).map(|i| (i % 7) as f64).collect();
        let columns: Vec<Vec<f64>> = (0..3)
            .map(|id| data.float64_values(VariableId::from_raw(id)).unwrap().to_vec())
            .collect();
        let wide = TabularData::from_f64_columns([
            ("z", columns[0].as_slice()),
            ("t", columns[1].as_slice()),
            ("y", columns[2].as_slice()),
            ("spare", spare.as_slice()),
        ])
        .unwrap();
        let mut graph = Dag::with_variables(4);
        for (a, b) in [(0, 1), (0, 2), (1, 2)] {
            graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        let export = export_with(&wide, graph, query(), DrLearner::new());
        let consumed = replay_analysis_result(&export.sourced).unwrap();
        assert!(consumed.acceptance.unresolved.is_empty(), "{:?}", consumed.acceptance.unresolved);
        let original = parts(&export.sourced);
        assert_eq!(original.witness.columns.len(), 4);
        // The estimator runs on a table projected to the columns it needs; the exported model
        // still names the adjustment variable by its id in the full table, and consumers
        // resolve that id to the right header name.
        let model = original.body.fitted_effect.as_ref().unwrap();
        assert_eq!(model.features, vec![0]);
        assert_eq!(original.header.variable_names[0], "z");
        assert_eq!(
            FittedEffectModel::load(&export.sourced).unwrap().features(),
            vec!["z".to_owned()]
        );

        let mut dropped = original;
        dropped.witness.columns.pop();
        assert_refused_by_replay(&assemble(&dropped, true), SOURCE);
    }

    #[test]
    fn a_tampered_point_estimate_is_refused_after_the_claim_is_resealed() {
        let mut tampered = parts(&base().sourced);
        let point = tampered.body.estimate.unwrap();
        tampered.body.estimate = Some(f64::from_bits(point.to_bits() + 1));
        reseal(&mut tampered.contract, &tampered.body);
        assert_refused_by_replay(&assemble(&tampered, true), MISMATCH);
    }

    #[test]
    fn a_tampered_standard_error_is_refused_after_the_claim_is_resealed() {
        let mut tampered = parts(&base().sourced);
        let published = tampered.body.standard_error.expect("the route publishes its error");
        assert!(published > 0.0);
        tampered.body.standard_error = Some(published * 0.5);
        reseal(&mut tampered.contract, &tampered.body);
        assert_refused_by_replay(&assemble(&tampered, true), MISMATCH);

        let mut dropped = parts(&base().sourced);
        dropped.body.standard_error = None;
        reseal(&mut dropped.contract, &dropped.body);
        assert_refused_by_replay(&assemble(&dropped, true), MISMATCH);

        let mut interval = parts(&base().sourced);
        interval.body.interval_lower = Some(0.0);
        interval.body.interval_upper = Some(4.0);
        reseal(&mut interval.contract, &interval.body);
        let reasons = unresolved(&assemble(&interval, true));
        assert!(has(&reasons, "dependencies.checked_dr_replay_unsupported"), "{reasons:?}");
        assert!(has(&reasons, OPERATION), "{reasons:?}");
    }

    #[test]
    fn a_tampered_cate_vector_is_refused_after_the_claim_is_resealed() {
        let mut tampered = parts(&base().sourced);
        let mut cate = tampered.body.cate.as_ref().unwrap().to_vec();
        cate[3] = f64::from_bits(cate[3].to_bits() + 1);
        tampered.body.cate = Some(Arc::from(cate));
        reseal(&mut tampered.contract, &tampered.body);
        assert_refused_by_replay(&assemble(&tampered, true), MISMATCH);
    }

    fn assert_model_edit_refused(edit: impl Fn(&mut antecedent_estimate::FittedEffect)) {
        let mut tampered = parts(&base().sourced);
        edit(Arc::make_mut(tampered.body.fitted_effect.as_mut().unwrap()));
        reseal(&mut tampered.contract, &tampered.body);
        assert_refused_by_replay(&assemble(&tampered, true), MISMATCH);
    }

    #[test]
    fn a_tampered_predictor_coefficient_is_refused_after_the_claim_is_resealed() {
        assert_model_edit_refused(|model| {
            let PredictionMap::Linear { coefficients, .. } = &mut model.predictor.model else {
                panic!("the DR-learner's linear final stage retains a linear map");
            };
            coefficients[1] += 1e-9;
        });
    }

    #[test]
    fn a_tampered_feature_schema_or_intercept_is_refused_after_the_claim_is_resealed() {
        assert_model_edit_refused(|model| model.features = vec![1]);
        assert_model_edit_refused(|model| {
            model.intercept = false;
            model.predictor.columns -= 1;
            if let PredictionMap::Linear { coefficients, .. } = &mut model.predictor.model {
                coefficients.remove(0);
            }
        });
    }

    #[test]
    fn a_tampered_support_box_is_refused_after_the_claim_is_resealed() {
        assert_model_edit_refused(|model| model.support.as_mut().unwrap().ranges[0].hi += 1.0);
    }

    #[test]
    fn a_tampered_recorded_seed_is_refused_after_the_claim_is_resealed() {
        let mut tampered = parts(&base().sourced);
        tampered.body.crossfit_seed = Some(SEED + 1);
        reseal(&mut tampered.contract, &tampered.body);
        assert_refused_by_replay(&assemble(&tampered, true), MISMATCH);
    }

    #[test]
    fn a_tampered_declared_fold_count_is_refused_after_the_program_is_resealed() {
        let mut tampered = parts(&base().sourced);
        let declared =
            tampered.contract.program.as_mut().unwrap().checked_dr_lowering.as_mut().unwrap();
        declared.folds = 4;
        // The forger also rewrites the body's cross-fit record to agree with the declaration.
        tampered.body.crossfit_folds = Some(4);
        reseal(&mut tampered.contract, &tampered.body);
        // The estimator configuration committed in the program still says three folds, so the
        // declaration no longer binds to it.
        assert_refused_by_replay(&assemble(&tampered, true), BINDING);
    }

    #[test]
    fn a_wrong_learner_in_the_declaration_is_refused_even_when_the_binding_is_rewritten_to_match() {
        let wrong = LearnerSpec::Ridge(RidgeSpec { lambda: 4.0 });
        let mut tampered = parts(&base().sourced);
        tampered
            .contract
            .program
            .as_mut()
            .unwrap()
            .checked_dr_lowering
            .as_mut()
            .unwrap()
            .outcome_learner = CheckedLearnerWire::from_spec(wrong).unwrap();
        reseal(&mut tampered.contract, &tampered.body);
        // Declaration alone: the inference binding still names the executed learner.
        assert_refused_by_replay(&assemble(&tampered, true), BINDING);

        // Declaration and binding rewritten consistently, every digest recomputed: the replay
        // of the wrong learner does not produce the body.
        let binding = tampered.contract.inference_binding.as_mut().unwrap();
        let Some(EstimatorSpecWire::DrLearner(config)) = binding.estimator_spec.as_mut() else {
            panic!("a configured DR-learner records its configuration in the inference binding");
        };
        config.outcome_config = Some(wrong.identity());
        tampered.contract.identities.inference_binding =
            *inference_binding_digest(tampered.contract.inference_binding.as_ref().unwrap())
                .unwrap()
                .as_bytes();
        reseal(&mut tampered.contract, &tampered.body);
        assert_refused_by_replay(&assemble(&tampered, true), MISMATCH);
    }

    #[test]
    #[allow(
        clippy::redundant_closure_for_method_calls,
        reason = "the bounded reference fixture preserves its exact calculations and local structure"
    )]
    fn an_oversize_witness_is_refused_before_any_fit() {
        let bytes = &base().sourced;
        let bounds = SourceReplayBounds {
            witness: SourceWitnessLimits { max_bytes: 4096, ..SourceWitnessLimits::default() },
            ..SourceReplayBounds::default()
        };
        let (consumed, fits) =
            count_resolved_fits(|| replay_analysis_result_with_source(bytes, &bounds).unwrap());
        assert_eq!(fits, 0);
        let reasons: Vec<String> =
            consumed.acceptance.unresolved.iter().map(|r| r.to_string()).collect();
        assert!(has(&reasons, SOURCE), "{reasons:?}");
        assert!(has(&reasons, OPERATION), "{reasons:?}");
        assert!(!consumed.acceptance.accepts_as_claim());
    }

    #[test]
    #[allow(
        clippy::redundant_closure_for_method_calls,
        reason = "the bounded reference fixture preserves its exact calculations and local structure"
    )]
    fn a_cancelled_replay_is_cancelled_not_verified_and_not_a_mismatch() {
        let token = CancellationToken::new();
        token.cancel();
        let bounds = SourceReplayBounds { cancellation: token, ..SourceReplayBounds::default() };
        let bytes = &base().sourced;
        let (consumed, fits) =
            count_resolved_fits(|| replay_analysis_result_with_source(bytes, &bounds).unwrap());
        assert_eq!(fits, 0);
        let reasons: Vec<String> =
            consumed.acceptance.unresolved.iter().map(|r| r.to_string()).collect();
        assert!(has(&reasons, RESOURCE), "{reasons:?}");
        assert!(has(&reasons, OPERATION), "{reasons:?}");
        assert!(!has(&reasons, MISMATCH), "{reasons:?}");
        assert!(!consumed.acceptance.accepts_as_claim());
        let original = parts(bytes);
        assert_eq!(
            verify_checked_dr_replay(
                &original.header,
                &original.body,
                &original.contract,
                &original.witness,
                &bounds
            ),
            Err(DrReplayRefusal::Cancelled)
        );
    }

    #[test]
    fn the_export_replays_its_own_source_and_refuses_one_that_does_not_reproduce_the_body() {
        let original = parts(&base().sourced);
        let names = original.header.variable_names.clone();
        assert!(
            encode_analysis_result_artifact_with_contract_and_source(
                &original.body,
                names.clone(),
                "same",
                Some(&original.contract),
                Some(&original.witness),
            )
            .is_ok()
        );
        let mut tampered = original.clone();
        let point = tampered.body.estimate.unwrap();
        tampered.body.estimate = Some(f64::from_bits(point.to_bits() + 1));
        reseal(&mut tampered.contract, &tampered.body);
        assert!(
            encode_analysis_result_artifact_with_contract_and_source(
                &tampered.body,
                names.clone(),
                "forged",
                Some(&tampered.contract),
                Some(&tampered.witness),
            )
            .is_err()
        );
        // The same forgery with no source encodes as it always did: it stays unresolved.
        assert!(
            encode_analysis_result_artifact_with_contract_and_source(
                &tampered.body,
                names,
                "forged-compact",
                Some(&tampered.contract),
                None,
            )
            .is_ok()
        );
    }

    #[test]
    fn a_dropped_witness_leaves_the_reason_and_a_garbled_one_is_never_ignored() {
        let original = parts(&base().sourced);
        // The same contract and body without the witness section is the compact export.
        let without = assemble(&original, false);
        assert!(has(&unresolved(&without), OPERATION));
        // A malformed witness section is never silently ignored when the contract declares
        // a lowering: the artifact stays unresolved with the source reason.
        let mut garbled = original;
        garbled.witness.columns[0].values.pop();
        assert_refused_by_replay(&assemble(&garbled, true), SOURCE);
    }
}
