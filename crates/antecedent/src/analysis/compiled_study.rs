//! Persistence of a portable prepared tabular average-effect study.
//!
//! Opening starts a new compilation after verifying the retained invariant and source.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{CausalQuery, Determinism, ExecutionContext, KernelPolicy};
use antecedent_data::{TableView, TabularData};
use antecedent_io::catalog_role::SOURCE_ROWS;
use antecedent_io::{
    CausalQueryWire, DagWire, ExecutionIdentityWire, IoError, RetainedArtifactIdentity, SchemaWire,
    causal_query_from_wire, causal_query_to_wire, dag_from_wire, dag_to_wire, execution_digest,
    execution_identity_from_context, from_cbor, schema_from_wire, schema_to_wire, to_cbor,
};
use antecedent_io_core::source_witness::{SourceWitnessLimits, SourceWitnessWire};
use serde::{Deserialize, Serialize};

use super::PreparedStudy;
use crate::analysis::builder::DataInput;
use crate::estimator_spec::EstimatorSpec;
use crate::{CausalError, EstimatorId, InferenceMode, RefuteSuite, Study};

pub use antecedent_io::compiled_study::{
    CATALOG_SECTION, CATALOG_VERSION, CLAIM_SECTION, FORMAT, INVARIANT_SECTION, KIND, OpenedStudy,
    Placement, StudyBody, StudyInvariant, StudyParts, UnresolvedBody, WrittenStudy,
    accept_source_body, body_section, layer5_quote, source_content_digest, write,
};

const CONTINUATION_VERSION: u16 = 1;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Continuation {
    version: u16,
    graph: DagWire,
    query: CausalQueryWire,
    identifier: Option<String>,
    estimator: Option<String>,
    resolved_estimator: String,
    refute: Option<String>,
    bootstrap_replicates: u32,
    execution: ExecutionIdentityWire,
    schema: SchemaWire,
    source: [u8; 32],
    inference: [u8; 32],
    lowered_program: [u8; 32],
    estimate_bits: u64,
}

fn refused(message: impl Into<String>) -> IoError {
    IoError::Refused {
        code: antecedent_core::reason_code!("invalid_argument"),
        message: message.into(),
    }
}

fn map_causal(error: CausalError) -> IoError {
    refused(error.to_string())
}

impl Continuation {
    fn context(&self) -> Result<ExecutionContext, IoError> {
        let execution = &self.execution;
        let mut ctx = ExecutionContext::production(execution.seed, execution.threads);
        ctx.kernel_policy = KernelPolicy {
            allow_portable_optimized: execution.kernel.allow_portable_optimized,
            allow_arch_simd: execution.kernel.allow_arch_simd,
            force_scalar: execution.kernel.force_scalar,
        };
        if execution.determinism == "prefer_fast" {
            ctx.determinism = Determinism::PreferFast;
        }
        if execution_identity_from_context(&ctx) != *execution {
            return Err(refused("the saved execution policy is unavailable for replay"));
        }
        Ok(ctx)
    }

    fn prepare(
        &self,
        invariant: &StudyInvariant,
        source: &[u8],
        ctx: &ExecutionContext,
    ) -> Result<PreparedStudy, IoError> {
        if self.version != CONTINUATION_VERSION {
            return Err(IoError::UnsupportedVersion { version: self.version.into() });
        }
        let witness: SourceWitnessWire = from_cbor(source)?;
        let storage =
            witness.to_storage(&self.schema.variable_names(), &SourceWitnessLimits::default())?;
        if storage.content_digest() != self.source {
            return Err(refused("resolved source does not match the study invariant"));
        }
        let storage = storage
            .with_schema(schema_from_wire(&self.schema)?)
            .map_err(|error| refused(error.to_string()))?;
        let query = causal_query_from_wire(&self.query)?;
        if !matches!(query, CausalQuery::AverageEffect(_)) {
            return Err(refused("the saved query has no executable continuation"));
        }
        let suite = match self.refute.as_deref() {
            None => RefuteSuite::None,
            Some("overlap+evalue") => RefuteSuite::Cheap,
            Some("placebo+rcc") => RefuteSuite::PlaceboAndRcc,
            Some("validation.full") => RefuteSuite::Full,
            Some(_) => return Err(refused("unknown saved validation suite")),
        };
        let mut builder = Study::tabular(TabularData::new(storage))
            .graph(dag_from_wire(&self.graph)?)
            .query(query)
            .refute(suite)
            .bootstrap_replicates(self.bootstrap_replicates);
        if let Some(estimator) = &self.estimator {
            builder = builder.estimator(
                estimator.parse::<EstimatorId>().map_err(|_| refused("unknown saved estimator"))?,
            );
        }
        if let Some(identifier) = &self.identifier {
            builder = builder
                .identifier(identifier.parse().map_err(|_| refused("unknown saved identifier"))?);
        }
        let prepared = builder.build().and_then(|study| study.prepare(ctx)).map_err(map_causal)?;
        let plan = prepared.plan();
        if plan.logical.record.estimator.as_deref() != Some(self.resolved_estimator.as_str())
            || plan
                .lowered_program_identity()
                .as_ref()
                .map(antecedent_core::SemanticDigest::as_bytes)
                != Some(&self.lowered_program)
        {
            return Err(refused("the reconstructed estimator does not match the study invariant"));
        }
        let identities = prepared.contract().map_err(map_causal)?.identities;
        if identities.program.as_ref().map(antecedent_core::SemanticDigest::as_bytes)
            != Some(&invariant.program)
            || *identities.data_snapshot.as_bytes() != invariant.data_snapshot
            || *identities.inference_binding.as_bytes() != self.inference
        {
            return Err(refused("the reconstructed contract does not match the study invariant"));
        }
        Ok(prepared)
    }
}

impl PreparedStudy {
    /// Save an executed frequentist average-effect study and its sealed claim.
    ///
    /// The portable route uses a supplied DAG, a named estimator with builder defaults,
    /// and a table of finite, unmasked, unweighted float64 values. `source_uri` places the table at an
    /// opaque URI. The resolver must supply the encoded `source_rows` body at that URI.
    ///
    /// # Errors
    /// The configuration cannot be reconstructed, the result is from another study or execution,
    /// or artifact encoding fails.
    pub fn save_compiled_study(
        &self,
        result: &crate::StudyResult,
        ctx: &ExecutionContext,
        source_uri: Option<&str>,
    ) -> Result<WrittenStudy, IoError> {
        let study = self.study();
        let DataInput::Tabular(data) = &study.data else {
            return Err(refused("compiled continuation requires a tabular study"));
        };
        let plan = self.plan();
        let graph = study
            .graph
            .as_dag()
            .filter(|_| study.structure_source == crate::support::StructureSource::Explicit)
            .ok_or_else(|| refused("compiled continuation requires an explicitly supplied DAG"))?;
        if !matches!(self.query(), CausalQuery::AverageEffect(_))
            || !matches!(study.settings.inference, InferenceMode::Frequentist)
            || !matches!(
                study.declared.estimator_spec.as_deref(),
                None | Some(EstimatorSpec::Default(_))
            )
            || data.unit_key().is_some()
        {
            return Err(refused("this prepared estimator has no executable continuation"));
        }
        let schema = schema_to_wire(data.schema());
        let names = schema.variable_names();
        let witness = SourceWitnessWire::from_storage(data.storage())?;
        let source = to_cbor(&witness)?;
        let execution = execution_identity_from_context(ctx);
        if result.execution_identity != Some(execution_digest(&execution)?) {
            return Err(refused("the supplied execution does not match the saved result"));
        }
        let (claim_contract, body) =
            self.execution_record_with_quantity_labels(result, ctx, None).map_err(map_causal)?;
        let estimate = body
            .estimate
            .filter(|estimate| estimate.is_finite())
            .ok_or_else(|| refused("compiled continuation requires a finite point estimate"))?;
        let continuation = Continuation {
            version: CONTINUATION_VERSION,
            graph: dag_to_wire(graph)?,
            query: causal_query_to_wire(self.query())?,
            identifier: study.declared.identifier.map(|id| id.as_str().to_owned()),
            estimator: study.declared.estimator.map(|id| id.as_str().to_owned()),
            resolved_estimator: plan
                .logical
                .record
                .estimator
                .as_deref()
                .ok_or_else(|| refused("the study has no resolved estimator"))?
                .to_owned(),
            refute: study.settings.refute.validation_suite_id().map(str::to_owned),
            bootstrap_replicates: study.settings.bootstrap_replicates,
            execution,
            schema,
            source: data.storage().content_digest(),
            inference: claim_contract.identities.inference_binding,
            lowered_program: *plan
                .lowered_program_identity()
                .ok_or_else(|| refused("the study has no lowered program"))?
                .as_bytes(),
            estimate_bits: estimate.to_bits(),
        };
        let invariant = StudyInvariant {
            program: claim_contract.identities.program,
            data_snapshot: claim_contract.identities.data_snapshot,
            continuation: to_cbor(&continuation)?,
        };
        continuation.prepare(&invariant, &source, &continuation.context()?)?;
        let claim = antecedent_io_core::encode_analysis_result_artifact_with_provenance(
            &body,
            names.clone(),
            "compiled-study-claim",
            Some(&claim_contract),
            None,
            Some(&result.method_provenance()),
        )?;
        let mut claim_bytes = Vec::new();
        claim.write_to(&mut claim_bytes)?;
        let invariant = invariant.encode()?;
        let bodies = [StudyBody {
            role: SOURCE_ROWS.into(),
            bytes: source,
            external_uri: source_uri.map(str::to_owned),
            required: true,
            summary_only: false,
            names,
        }];
        write(StudyParts { invariant: &invariant, bodies: &bodies, claim: Some(&claim_bytes) })
    }
}

/// Open a compiled study under the invariant identity returned by [`write`].
///
/// A portable study with its required source present reconstructs the prepared
/// estimator and executes it. An unresolved source leaves the bound claim available.
///
/// # Errors
/// Invalid identities or bodies, an unavailable execution policy, or a failed replay.
pub fn open(
    bytes: &[u8],
    expected_identity: &str,
    resolve: &dyn Fn(&str) -> Option<Vec<u8>>,
    caller_supplied: bool,
) -> Result<OpenedStudy, IoError> {
    let opened =
        antecedent_io::compiled_study::open(bytes, expected_identity, resolve, caller_supplied)?;
    replay(opened)
}

/// Open and replay a compiled study under a separately retained invariant identity.
///
/// # Errors
/// As [`open`].
pub fn open_expecting(
    bytes: &[u8],
    retained: &RetainedArtifactIdentity,
    resolve: &dyn Fn(&str) -> Option<Vec<u8>>,
    caller_supplied: bool,
) -> Result<OpenedStudy, IoError> {
    let opened =
        antecedent_io::compiled_study::open_expecting(bytes, retained, resolve, caller_supplied)?;
    replay(opened)
}

fn replay(mut opened: OpenedStudy) -> Result<OpenedStudy, IoError> {
    let Some(invariant) = StudyInvariant::decode(&opened.invariant)? else { return Ok(opened) };
    if invariant.continuation.is_empty() {
        return Ok(opened);
    }
    let recipe: Continuation = from_cbor(&invariant.continuation)?;
    let claim = opened
        .claim
        .as_ref()
        .ok_or_else(|| refused("the saved estimator requires its sealed claim"))?;
    let claim_identities = &claim
        .contract
        .as_ref()
        .ok_or_else(|| refused("the saved estimator requires its claim contract"))?
        .identities;
    if claim_identities.inference_binding != recipe.inference
        || claim_identities.execution != Some(*execution_digest(&recipe.execution)?.as_bytes())
        || claim.body.estimate.map(f64::to_bits) != Some(recipe.estimate_bits)
    {
        return Err(refused("the sealed claim execution does not match the study invariant"));
    }
    match opened.source_names.as_ref() {
        Some(names) if names != &recipe.schema.variable_names() => {
            return Err(refused("source names do not match the study invariant"));
        }
        None => opened.unresolved.push(UnresolvedBody { role: SOURCE_ROWS.into(), uri: None }),
        _ => {}
    }
    if !opened.unresolved.is_empty()
        || opened.refit_refused.is_some()
        || opened.callback_requires_caller
    {
        return Ok(opened);
    }
    let Some(source) = opened.bodies.get(SOURCE_ROWS) else { return Ok(opened) };
    let ctx = recipe.context()?;
    let prepared = recipe.prepare(&invariant, source, &ctx)?;
    let result = prepared.estimate_retained(&ctx).map_err(map_causal)?;
    let estimate = result.estimate.as_effect().map(|estimate| estimate.ate);
    if estimate.map(f64::to_bits) != Some(recipe.estimate_bits) {
        return Err(refused("the saved estimator did not reproduce its estimate"));
    }
    opened.layer5_bytes = prepared.plan().memory_quote.as_ref().map(|quote| quote.total());
    opened.resumable = true;
    opened.estimate = estimate;
    Ok(opened)
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{AverageEffectQuery, VariableId};
    use antecedent_graph::{Dag, DenseNodeId};

    fn data(extra: f64) -> TabularData {
        let z: Vec<_> = (0..240).map(|i| (f64::from(i) * 0.37).sin()).collect();
        let t: Vec<_> = z
            .iter()
            .enumerate()
            .map(|(i, z)| f64::from(u8::from(z + 0.8 * (i as f64 * 1.31).sin() > 0.0)))
            .collect();
        let y: Vec<_> = t
            .iter()
            .zip(&z)
            .enumerate()
            .map(|(i, (t, z))| (2.0 + extra) * t + z + 0.1 * (i as f64 * 2.17).sin())
            .collect();
        TabularData::from_f64_columns([
            ("t", t.as_slice()),
            ("y", y.as_slice()),
            ("z", z.as_slice()),
        ])
        .unwrap()
    }

    fn prepared(
        data: TabularData,
        estimator: EstimatorId,
        ctx: &ExecutionContext,
    ) -> PreparedStudy {
        let mut graph = Dag::with_variables(3);
        for (source, target) in [(2, 0), (2, 1), (0, 1)] {
            graph
                .insert_directed(DenseNodeId::from_raw(source), DenseNodeId::from_raw(target))
                .unwrap();
        }
        Study::tabular(data)
            .graph(graph)
            .query(AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)))
            .refute(RefuteSuite::None)
            .bootstrap_replicates(if estimator == EstimatorId::PropensityWeighting {
                20
            } else {
                0
            })
            .estimator(estimator)
            .build()
            .unwrap()
            .prepare(ctx)
            .unwrap()
    }

    #[test]
    fn fresh_process_replays_the_saved_estimator() {
        const CHILD: &str = "ANTECEDENT_COMPILED_STUDY_CHILD";
        if let Ok(input) = std::env::var(CHILD) {
            let mut input = input.splitn(3, ':');
            let identity = input.next().unwrap();
            let estimate_bits = input.next().unwrap().parse::<u64>().unwrap();
            let path = input.next().unwrap();
            let bytes = std::fs::read(path).unwrap();
            let opened = open(&bytes, identity, &|_| None, false).unwrap();
            assert!(opened.resumable);
            assert!(opened.layer5_bytes.is_some_and(|bytes| bytes > 0));
            assert_eq!(opened.estimate.map(f64::to_bits), Some(estimate_bits));
            assert!(opened.claim.unwrap().is_intact());
            return;
        }
        let ctx = ExecutionContext::for_tests(237);
        let prefix = module_path!().split_once("::").map_or("", |(_, remainder)| remainder);
        let test_name = format!("{prefix}::fresh_process_replays_the_saved_estimator");
        let mut previous_result = None;
        for estimator in [EstimatorId::LinearAdjustmentAte, EstimatorId::PropensityWeighting] {
            let prepared = prepared(data(0.0), estimator, &ctx);
            let result = prepared.estimate_retained(&ctx).unwrap();
            let point = result.estimate.as_effect().unwrap().ate;
            // The balanced binary treatment in the first column has mean one half.
            assert!((point - 0.5).abs() > 1.0);
            if let Some(previous) = previous_result.as_ref() {
                assert!(prepared.save_compiled_study(previous, &ctx, None).is_err());
            }
            let saved = prepared.save_compiled_study(&result, &ctx, None).unwrap();
            let path = std::env::temp_dir().join(format!(
                "compiled-study-{}-{}",
                std::process::id(),
                estimator.as_str()
            ));
            std::fs::write(&path, &saved.bytes).unwrap();
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", &test_name, "--nocapture"])
                .env(
                    CHILD,
                    format!("{}:{}:{}", saved.expected_identity, point.to_bits(), path.display()),
                )
                .output()
                .unwrap();
            std::fs::remove_file(path).unwrap();
            assert!(
                output.status.success(),
                "{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(String::from_utf8_lossy(&output.stdout).contains("running 1 test"));
            previous_result = Some(result);
        }
    }

    #[test]
    fn opening_with_a_missing_source_preserves_the_claim_and_substitution_refuses() {
        let ctx = ExecutionContext::for_tests(1);
        let source = data(0.0);
        let other = data(1.0);
        let prepared = prepared(source.clone(), EstimatorId::LinearAdjustmentAte, &ctx);
        let mut different_result =
            self::prepared(other.clone(), EstimatorId::LinearAdjustmentAte, &ctx)
                .estimate_retained(&ctx)
                .unwrap();
        assert!(prepared.save_compiled_study(&different_result, &ctx, None).is_err());
        different_result = prepared.estimate_retained(&ctx).unwrap();
        let saved =
            prepared.save_compiled_study(&different_result, &ctx, Some("study://source")).unwrap();
        let absent = open(&saved.bytes, &saved.expected_identity, &|_| None, false).unwrap();
        assert!(!absent.resumable);
        assert_eq!(absent.unresolved[0].uri.as_deref(), Some("study://source"));
        assert!(absent.claim.as_ref().is_some_and(antecedent_io::AnalysisResultLoad::is_intact));

        let source_bytes =
            to_cbor(&SourceWitnessWire::from_storage(source.storage()).unwrap()).unwrap();
        let resolved = |uri: &str| (uri == "study://source").then(|| source_bytes.clone());
        let retained = RetainedArtifactIdentity::from_hex(&saved.expected_identity).unwrap();
        let reopened = open(&saved.bytes, &saved.expected_identity, &resolved, false).unwrap();
        let bound = open_expecting(&saved.bytes, &retained, &resolved, false).unwrap();
        assert!(bound.resumable && reopened.resumable);
        assert_eq!(bound.estimate.map(f64::to_bits), reopened.estimate.map(f64::to_bits));
        assert_eq!(bound.layer5_bytes, reopened.layer5_bytes);
        let wrong = RetainedArtifactIdentity::of_bytes(&saved.bytes);
        let unexpected_resolve = |_: &str| panic!("an unbound invariant must refuse resolution");
        let mismatch =
            open_expecting(&saved.bytes, &wrong, &unexpected_resolve, false).unwrap_err();
        assert!(mismatch.to_string().contains("expected identity"), "{mismatch}");

        let claim = prepared.encode_contracted_result(&different_result, "claim", &ctx).unwrap();
        let substitute = |bytes, names: &[&str]| {
            let body = StudyBody {
                role: SOURCE_ROWS.into(),
                bytes,
                external_uri: None,
                required: true,
                summary_only: false,
                names: names.iter().map(ToString::to_string).collect(),
            };
            write(StudyParts { invariant: &absent.invariant, bodies: &[body], claim: Some(&claim) })
                .unwrap()
        };
        let replaced = to_cbor(&SourceWitnessWire::from_storage(other.storage()).unwrap()).unwrap();
        let tampered = substitute(replaced, &["t", "y", "z"]);
        let error = open(&tampered.bytes, &saved.expected_identity, &|_| None, false).unwrap_err();
        assert!(error.to_string().contains("resolved source"), "{error}");
        let tampered = substitute(source_bytes, &["t", "renamed", "z"]);
        let error = open(&tampered.bytes, &saved.expected_identity, &|_| None, false).unwrap_err();
        assert!(error.to_string().contains("source names"), "{error}");
    }

    #[test]
    fn an_opaque_invariant_without_bodies_does_not_resume() {
        let written = write(StudyParts { invariant: b"opaque", bodies: &[], claim: None }).unwrap();
        let opened = open(&written.bytes, &written.expected_identity, &|_| None, false).unwrap();
        assert!(!opened.resumable);
        assert!(opened.estimate.is_none());
    }
}
