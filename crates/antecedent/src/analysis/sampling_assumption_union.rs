//! Source-declared whole-chain bootstrap union for an assumption envelope.
//!
//! Each bootstrap replicate reruns the same complete propagation declaration
//! with new sampling rows while keeping interval-valued assumptions fixed.
//! The basic one-sided bootstrap bounds for the lower and upper endpoints use
//! a Bonferroni split, so one interval is formed at the final composed
//! coordinate. Bootstrap accuracy and whole-schedule resampling provenance
//! still require calibration; this route reports `not_measured` and does not
//! promote an assumption enclosure to a confidence claim.

use std::collections::BTreeSet;
use std::sync::Arc;

use antecedent_contract::{ClaimKind, DataClassification};
use antecedent_contract::propagation_chain::ChainRefusal;
use antecedent_core::{
    ByteEstimates, CanonicalWriter, ExecutionContext, PhysicalEffect,
    PhysicalOperator, ScoreFrequentist, ScoreModel, SemanticDigest,
};
use antecedent_io::IoError;
use antecedent_io::propagation_chain_record::PropagationChainRecordWire;
use antecedent_kernels::quantile_type7_sorted;
use serde::{Deserialize, Serialize};

use super::propagation_chain::{
    PropagationChain, PropagationChainPlan, compile_propagation_chain,
};
use crate::planner::lowered::{
    ClaimDemand, Composition, DesignShape, EvidenceDecisionProgram,
    EvidenceRole, EvidenceSourceRef, FunctionalKind, IdentificationLayer,
    LoweredProgram, NonGraphicalIdentification, PhysicalLayer, ProblemTarget,
    ProgramCell, ProgramIdentityScope, ScientificProblem,
    StandaloneCompositionLoweredParts, UncertaintyReading,
};
use crate::planner::statistical::{Realization, StatisticalProgram};
use crate::planner::{PlanStage, PlanView};

const FEATURE: &str = "sampling_assumption_union_v1";
const MAX_ARTIFACT_BYTES: usize = 80 * 1024 * 1024;

/// Prespecified original chain and whole-chain bootstrap replicates. The
/// replicate identifiers and resampling design are provenance declarations,
/// not proof that the replicate generator was run correctly.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SamplingAssumptionUnionRequest {
    /// Original assumption-envelope chain draft.
    pub original: PropagationChainRecordWire,
    /// Re-fitted chain drafts, one per whole-schedule bootstrap replicate.
    pub bootstrap: Vec<PropagationChainRecordWire>,
    /// Distinct identities assigned before inspecting interval coverage.
    pub replicate_ids: Vec<String>,
    /// Prespecified resampling scheme and sampling-unit identity.
    pub resampling_design_identity: String,
    /// Nominal two-sided confidence level; actual coverage is not measured.
    pub level: f64,
}

/// Refusal of an incompatible chain schedule or portable replay.
#[derive(Debug, thiserror::Error)]
pub enum SamplingAssumptionUnionError {
    /// The source-declared whole-chain schedule is unsupported.
    #[error("sampling assumption union: {0}")]
    Invalid(&'static str),
    /// A propagation chain refused its declaration or arithmetic.
    #[error("sampling assumption union chain: {0}")]
    Chain(#[from] ChainRefusal),
    /// Artifact encoding or decoding refusal.
    #[error("sampling assumption union artifact: {0}")]
    Io(#[from] IoError),
}

/// One compiler program covering the original and all declared resampling
/// chains at the same terminal coordinate.
#[derive(Clone, Debug)]
pub struct CompiledSamplingAssumptionUnion {
    request: SamplingAssumptionUnionRequest,
    original: PropagationChainPlan,
    bootstrap: Vec<PropagationChainPlan>,
    seed: u64,
    lowered: Arc<LoweredProgram>,
}

/// A bootstrap interval candidate for the whole assumption-identified set.
/// It has no earned coverage status.
#[derive(Clone, Debug)]
pub struct SamplingAssumptionUnionResult {
    /// Plug-in lower endpoint, the original chain's weighted lower mean.
    pub original_lower: f64,
    /// Plug-in upper endpoint, the original chain's weighted upper mean.
    pub original_upper: f64,
    /// Basic-bootstrap lower endpoint after the one-sided alpha split.
    pub union_lower: f64,
    /// Basic-bootstrap upper endpoint after the one-sided alpha split.
    pub union_upper: f64,
    /// Nominal confidence level; not a measured coverage statement.
    pub level: f64,
    /// Always `not_measured` until a matching whole-route record is earned.
    pub calibration: &'static str,
    /// Original and replicate chain identities in declared order.
    pub chain_identities: Vec<String>,
    /// Full six-layer compiler identity.
    pub lowered_program_identity: SemanticDigest,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SamplingAssumptionUnionWire {
    version: u32,
    required_feature: String,
    request: SamplingAssumptionUnionRequest,
    seed: u64,
    program_identity: [u8; 32],
    result_fingerprint: [u8; 32],
}

fn compatible(
    original: &PropagationChainRecordWire,
    replicate: &PropagationChainRecordWire,
) -> bool {
    original.construction == replicate.construction
        && original.reading == replicate.reading
        && original.output == replicate.output
        && original.output_quantity == replicate.output_quantity
        && original.joint_outputs == replicate.joint_outputs
        && original.stages == replicate.stages
        && original.changes == replicate.changes
        && original.sources.len() == replicate.sources.len()
        && original.sources.iter().zip(&replicate.sources).all(|(left, right)| {
            left.id == right.id
                && left.representation == right.representation
                && left.meaning == right.meaning
                && left.complete_disclosure == right.complete_disclosure
                && (left.representation != "interval" || left.values == right.values)
        })
}

/// Bind the original chain, every full bootstrap rerun and the exact output
/// coordinate in a new compiler program. The raw replicate declarations are
/// checked for matching stages and fixed assumption intervals before
/// numerical execution.
///
/// # Errors
/// Missing sampling or assumption source, incompatible rerun, malformed
/// level, oversized replicate schedule or chain compiler refusal.
pub fn compile_sampling_assumption_union(
    request: SamplingAssumptionUnionRequest,
    ctx: &ExecutionContext,
) -> Result<CompiledSamplingAssumptionUnion, SamplingAssumptionUnionError> {
    let original = &request.original;
    let mut ids = BTreeSet::new();
    if original.reading != "assumption_envelope"
        || original.construction != "joint_draws"
        || original.output_quantity.is_none()
        || !original.sources.iter().any(|s| s.representation == "interval")
        || !original.sources.iter().any(|s| {
            s.representation == "draws" && s.meaning.as_deref() == Some("sampling")
        })
        || request.bootstrap.len() < 100 || request.bootstrap.len() > 4096
        || request.bootstrap.len() != request.replicate_ids.len()
        || request.resampling_design_identity.trim().is_empty()
        || !request.level.is_finite() || !(0.0..1.0).contains(&request.level)
        || request.replicate_ids.iter().any(|id| {
            id.trim().is_empty() || !ids.insert(id.clone())
        })
        || request.bootstrap.iter().any(|replicate| !compatible(original, replicate))
    {
        return Err(SamplingAssumptionUnionError::Invalid("invalid whole-chain resampling declaration"));
    }
    let source_bytes = antecedent_io::to_cbor(&request)?;
    if source_bytes.len() > MAX_ARTIFACT_BYTES {
        return Err(SamplingAssumptionUnionError::Invalid("whole-chain resampling declaration exceeds its byte bound"));
    }
    let source_digest = CanonicalWriter::digest_bytes(
        "antecedent.sampling_assumption_union.source.v1", &source_bytes,
    );
    let seed = ctx.rng.master_seed();
    let mut writer = CanonicalWriter::new("antecedent.sampling_assumption_union.question.v1");
    writer.bytes(&source_digest).word(seed);
    let query = writer.finish();
    let original_plan = compile_propagation_chain(request.original.clone(), ctx)?;
    let bootstrap = request.bootstrap.iter().cloned()
        .map(|draft| compile_propagation_chain(draft, ctx))
        .collect::<Result<Vec<_>, _>>()?;
    let design = DesignShape::quoted(request.bootstrap.len() as u64,
        request.original.sources.len() as u64, true);
    let scientific = ScientificProblem {
        runtime_query: None, identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::IdentifiedSet, outcome: None, query,
        },
        plan_id: Arc::from("sampling_assumption_union.v1"),
        data: DataClassification::Tabular,
        graph_class: "declared_whole_chain_bootstrap_assumption_set",
        query_variables: Arc::from([]), source_digest: Some(source_digest),
        support_coordinate: None, supported: false, support_status: None,
    };
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SharedEvidenceContract { contract: query },
    ));
    let score = ScoreModel::new("whole_chain_assumption_endpoint_bootstrap")
        .map_err(|_| SamplingAssumptionUnionError::Invalid("invalid bootstrap score identity"))?;
    let statistical = StatisticalProgram {
        identifier: None, estimator: None,
        realization: Realization::Frequentist(ScoreFrequentist::new(score)),
        models: Arc::from([]), folds: None, resampling: None,
        posterior: None, design, operators: Arc::from([]),
    };
    let evidence = EvidenceDecisionProgram {
        sources: Arc::from([EvidenceSourceRef {
            role: EvidenceRole::ChainSource, digest: Some(source_digest),
        }]),
        composition: Composition::Chained { stages: request.bootstrap.len() + 1 },
        decision: None, planning: None, chain: None,
    };
    let physical = PhysicalLayer {
        operators: Arc::from([
            PhysicalOperator::declare(PhysicalEffect::Read)
                .estimates(ByteEstimates::absent().with_output(source_bytes.len() as u64))
                .build().map_err(|_| SamplingAssumptionUnionError::Invalid("invalid bootstrap read operator"))?,
            PhysicalOperator::declare(PhysicalEffect::Select)
                .build().map_err(|_| SamplingAssumptionUnionError::Invalid("invalid resampling operator"))?,
        ]),
        design,
    };
    let lowered = Arc::new(StandaloneCompositionLoweredParts {
        cell: ProgramCell::SamplingAssumptionUnion,
        scientific, identification, statistical, evidence, physical,
        claim: ClaimDemand { uncertainty: UncertaintyReading::Sampling,
            kind: Some(ClaimKind::Bounds) },
    }.finish());
    Ok(CompiledSamplingAssumptionUnion {
        request, original: original_plan, bootstrap, seed, lowered,
    })
}

fn endpoint_means(chain: &PropagationChain) -> Result<(f64, f64), SamplingAssumptionUnionError> {
    let (lower, upper, weights) = chain.output_envelope().ok_or(
        SamplingAssumptionUnionError::Invalid("a chain lost its terminal assumption envelope"),
    )?;
    if lower.is_empty() || lower.len() != upper.len()
        || weights.is_some_and(|w| w.len() != lower.len())
    {
        return Err(SamplingAssumptionUnionError::Invalid("invalid terminal envelope rows"));
    }
    let mut total = 0.0;
    let mut lo = 0.0;
    let mut hi = 0.0;
    for row in 0..lower.len() {
        let l = f64::from_bits(lower[row]);
        let u = f64::from_bits(upper[row]);
        let weight = weights.map_or(1.0, |w| f64::from_bits(w[row]));
        if !l.is_finite() || !u.is_finite() || l > u
            || !weight.is_finite() || weight < 0.0
        {
            return Err(SamplingAssumptionUnionError::Invalid("nonfinite or reversed terminal envelope"));
        }
        total += weight;
        lo += weight * l;
        hi += weight * u;
    }
    if !total.is_finite() || total <= 0.0 {
        return Err(SamplingAssumptionUnionError::Invalid("terminal envelope has no positive weight"));
    }
    Ok((lo / total, hi / total))
}

fn fingerprint(result: &SamplingAssumptionUnionResult) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("antecedent.sampling_assumption_union.result.v1");
    writer.word(result.original_lower.to_bits())
        .word(result.original_upper.to_bits())
        .word(result.union_lower.to_bits()).word(result.union_upper.to_bits())
        .word(result.level.to_bits()).text(result.calibration)
        .bytes(result.lowered_program_identity.as_bytes())
        .texts(&result.chain_identities);
    writer.finish()
}

impl CompiledSamplingAssumptionUnion {
    /// Complete compiler program for the final coordinate and resampling law.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram { &self.lowered }

    /// Inspect all layers before any chain arithmetic.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }

    /// Rerun the original and every resampled chain, then form basic-bootstrap
    /// lower and upper endpoint bounds at alpha/2 each. The result remains
    /// `not_measured` until its exact schedule and coordinate earn coverage.
    ///
    /// # Errors
    /// Any nested compiler or invalid endpoint refusal.
    pub fn execute(
        &self, ctx: &ExecutionContext,
    ) -> Result<SamplingAssumptionUnionResult, SamplingAssumptionUnionError> {
        if ctx.rng.master_seed() != self.seed {
            return Err(SamplingAssumptionUnionError::Invalid("resampling execution seed changed"));
        }
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        let original = self.original.execute(ctx)?;
        let (original_lower, original_upper) = endpoint_means(&original)?;
        let mut lows = Vec::with_capacity(self.bootstrap.len());
        let mut highs = Vec::with_capacity(self.bootstrap.len());
        let mut chain_identities = vec![original.identity().to_owned()];
        for plan in &self.bootstrap {
            if scoped.cancellation.is_cancelled() {
                return Err(SamplingAssumptionUnionError::Invalid("whole-chain resampling was cancelled"));
            }
            let chain = plan.execute(ctx)?;
            let (lower, upper) = endpoint_means(&chain)?;
            lows.push(lower);
            highs.push(upper);
            chain_identities.push(chain.identity().to_owned());
        }
        lows.sort_by(f64::total_cmp);
        highs.sort_by(f64::total_cmp);
        let tail = (1.0 - self.request.level) / 2.0;
        let union_lower = 2.0 * original_lower - quantile_type7_sorted(&lows, 1.0 - tail);
        let union_upper = 2.0 * original_upper - quantile_type7_sorted(&highs, tail);
        if !union_lower.is_finite() || !union_upper.is_finite() || union_lower > union_upper {
            return Err(SamplingAssumptionUnionError::Invalid("bootstrap endpoint union is nonfinite or reversed"));
        }
        Ok(SamplingAssumptionUnionResult {
            original_lower, original_upper, union_lower, union_upper,
            level: self.request.level, calibration: "not_measured",
            chain_identities, lowered_program_identity: self.lowered.identity(),
        })
    }

    /// Seal the complete original and resampling declarations after checking
    /// a fresh result. The independent consumer recompiles every chain.
    ///
    /// # Errors
    /// A changed result or artifact over its byte limit.
    pub fn export(
        &self, result: &SamplingAssumptionUnionResult, ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, SamplingAssumptionUnionError> {
        let live = self.execute(ctx)?;
        if fingerprint(result) != fingerprint(&live) {
            return Err(SamplingAssumptionUnionError::Invalid("sampling union result changed"));
        }
        let bytes = antecedent_io::to_cbor(&SamplingAssumptionUnionWire {
            version: 1, required_feature: FEATURE.into(),
            request: self.request.clone(), seed: self.seed,
            program_identity: *self.lowered.identity().as_bytes(),
            result_fingerprint: fingerprint(&live),
        })?;
        if bytes.len() > MAX_ARTIFACT_BYTES {
            return Err(SamplingAssumptionUnionError::Invalid("sampling union artifact exceeds its bound"));
        }
        Ok(bytes)
    }
}

/// Independently reconstruct the whole-chain resampling program and every
/// nested chain from a portable artifact and externally retained identity.
///
/// # Errors
/// Changed schedule, chain execution, terminal coordinate or result.
pub fn consume_sampling_assumption_union(
    bytes: &[u8], expected_program: SemanticDigest, ctx: &ExecutionContext,
) -> Result<SamplingAssumptionUnionResult, SamplingAssumptionUnionError> {
    if bytes.len() > MAX_ARTIFACT_BYTES {
        return Err(SamplingAssumptionUnionError::Invalid("sampling union artifact exceeds its bound"));
    }
    let wire: SamplingAssumptionUnionWire = antecedent_io::from_cbor(bytes)?;
    if wire.version != 1 || wire.required_feature != FEATURE
        || wire.program_identity != *expected_program.as_bytes()
        || wire.seed != ctx.rng.master_seed()
    {
        return Err(SamplingAssumptionUnionError::Invalid("sampling union feature, identity or seed differs"));
    }
    let plan = compile_sampling_assumption_union(wire.request, ctx)?;
    if plan.lowered().identity() != expected_program {
        return Err(SamplingAssumptionUnionError::Invalid("recompiled sampling union differs"));
    }
    let result = plan.execute(ctx)?;
    if fingerprint(&result) != wire.result_fingerprint {
        return Err(SamplingAssumptionUnionError::Invalid("sampling union result differs on replay"));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_contract::propagation_chain::{
        ChainReading, LawMeaning, StageKind, TableConstruction,
    };
    use antecedent_io::propagation_chain_record::{
        ChainOpWire, ChainSourceWire, ChainStageWire,
    };
    use antecedent_io::quantity_wire::ScientificQuantityWire;

    fn draft(shift: f64) -> PropagationChainRecordWire {
        PropagationChain::draft(
            TableConstruction::JointDraws, ChainReading::AssumptionEnvelope,
            "answer",
            vec![
                ChainSourceWire::draws(
                    "estimate", LawMeaning::Sampling, "joint-sampling-table",
                    &[1.0 + shift, 2.0 + shift, 3.0 + shift],
                ),
                ChainSourceWire::interval("bias", 0.0, 1.0),
            ],
            vec![ChainStageWire::new(
                "answer", StageKind::DerivedQuantity, &["estimate", "bias"],
                ChainOpWire::binary("sum"),
            )],
        ).with_output_quantity(ScientificQuantityWire {
            version: 1, variable_id: "study.estimate_plus_bias".into(),
            variable_name: "Estimate plus bias".into(), role: "outcome".into(),
            units: "points".into(), population_id: "target".into(),
            regime_id: "observed".into(), horizon: 1,
            functional_id: "mean".into(), conditioning: Vec::new(),
            transform_id: "identity".into(),
        })
    }

    #[test]
    #[ignore = "source-only full-chain bootstrap design; coverage has not been earned"]
    fn whole_chain_bootstrap_union_replays_at_the_final_coordinate() {
        let ctx = ExecutionContext::for_tests(11);
        let request = SamplingAssumptionUnionRequest {
            original: draft(0.0),
            bootstrap: (0..100).map(|i| draft((i as f64 - 49.5) * 0.002)).collect(),
            replicate_ids: (0..100).map(|i| format!("resample-{i:03}")).collect(),
            resampling_design_identity: "fixed-row-bootstrap-seed-11-v1".into(),
            level: 0.95,
        };
        let plan = compile_sampling_assumption_union(request, &ctx)
            .expect("whole-chain union compiler");
        assert_eq!(plan.plan_view().route().as_str(), "SamplingAssumptionUnion");
        let result = plan.execute(&ctx).expect("endpoint bootstrap union");
        assert_eq!(result.calibration, "not_measured");
        assert!(result.union_lower < result.original_lower);
        assert!(result.union_upper > result.original_upper);
        let artifact = plan.export(&result, &ctx).expect("portable union artifact");
        let replay = consume_sampling_assumption_union(
            &artifact, plan.lowered().identity(), &ctx,
        ).expect("independent whole-chain replay");
        assert_eq!(fingerprint(&replay), fingerprint(&result));
    }
}
