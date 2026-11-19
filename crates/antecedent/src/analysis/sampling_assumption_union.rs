//! Source-declared whole-chain bootstrap union for an assumption envelope.
//!
//! Each bootstrap replicate reruns the same complete propagation declaration
//! with new sampling rows while keeping interval-valued assumptions fixed.
//! The basic one-sided bootstrap bounds for the lower and upper endpoints use
//! a Bonferroni split, so one interval is formed at the final composed
//! coordinate. Bootstrap accuracy and whole-schedule resampling provenance
//! still require calibration; this route reports `not_measured` and does not
//! promote an assumption enclosure to a confidence claim.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use antecedent_contract::{ClaimKind, DataClassification};
use antecedent_contract::propagation_chain::ChainRefusal;
use antecedent_core::{
    ByteEstimates, CalibrationBasis, CanonicalWriter, ExecutionContext, PhysicalEffect,
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
const IID_FEATURE: &str = "iid_sampling_assumption_union_v1";
const STRATIFIED_MOVING_BLOCK_FEATURE: &str = "stratified_moving_block_sampling_assumption_union_v1";
const PANEL_BLOCK_FEATURE: &str = "panel_block_sampling_assumption_union_v1";
const MOVING_BLOCK_FEATURE: &str = "moving_block_sampling_assumption_union_v1";
const STRATIFIED_CLUSTER_FEATURE: &str = "stratified_cluster_sampling_assumption_union_v1";
const CLUSTER_FEATURE: &str = "cluster_sampling_assumption_union_v1";
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

/// One raw IID unit table and a deterministic whole-chain bootstrap schedule.
/// All sampling-draw sources are columns of the same unit table; their rows
/// are resampled together. Fixed point and interval sources are never sampled.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct IidSamplingAssumptionUnionRequest {
    /// Draft chain over raw IID unit rows and fixed assumptions.
    pub original: PropagationChainRecordWire,
    /// Distinct unit identities in the common row order of every sampling source.
    pub unit_ids: Vec<String>,
    /// Number of complete-chain bootstrap resamples.
    pub bootstrap_replicates: usize,
    /// Nominal two-sided confidence level; no coverage is inferred from it.
    pub level: f64,
}

/// Compiler-owned IID resampling and full-chain endpoint union.
#[derive(Clone, Debug)]
pub struct CompiledIidSamplingAssumptionUnion {
    request: IidSamplingAssumptionUnionRequest,
    original: PropagationChainPlan,
    seed: u64,
    lowered: Arc<LoweredProgram>,
}

/// Unmeasured joint-endpoint coverage candidate for a fixed assumption set.
#[derive(Clone, Debug)]
pub struct IidSamplingAssumptionCalibrationCandidate {
    /// Complete compiler-generated bootstrap execution.
    pub result: SamplingAssumptionUnionResult,
    /// Executed route and fixed assumption template used by a future record.
    pub basis: CalibrationBasis,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct IidSamplingAssumptionUnionWire {
    version: u32,
    required_feature: String,
    request: IidSamplingAssumptionUnionRequest,
    seed: u64,
    program_identity: [u8; 32],
    result_fingerprint: [u8; 32],
}

fn validate_raw_unit_source(
    draft: &PropagationChainRecordWire, unit_ids: &[String],
    bootstrap_replicates: usize, level: f64,
) -> Result<(), SamplingAssumptionUnionError> {
    let n = unit_ids.len();
    let mut units = BTreeSet::new();
    let mut alignment = None;
    let mut sampling = 0usize;
    let mut assumptions = 0usize;
    if n < 20 || n > 1_000_000
        || bootstrap_replicates < 100 || bootstrap_replicates > 4096
        || n.checked_mul(bootstrap_replicates).is_none_or(|work| work > 50_000_000)
        || !level.is_finite() || !(0.0..1.0).contains(&level)
        || unit_ids.iter().any(|id| id.trim().is_empty() || !units.insert(id.as_str()))
        || draft.reading != "assumption_envelope" || draft.construction != "joint_draws"
        || draft.output_quantity.is_none() || !draft.results.is_empty()
        || !draft.identity.is_empty() || draft.output_draws.is_some()
        || draft.output_weights.is_some() || draft.output_lower_bounds.is_some()
        || draft.output_upper_bounds.is_some() || draft.joint_output_draws.is_some()
        || draft.joint_output_weights.is_some()
    {
        return Err(SamplingAssumptionUnionError::Invalid("invalid raw-unit IID bootstrap declaration"));
    }
    for source in &draft.sources {
        if !source.complete_disclosure || !source.factors.is_empty() {
            return Err(SamplingAssumptionUnionError::Invalid("raw IID unit columns need complete, factor-free disclosure"));
        }
        match source.representation.as_str() {
            "draws" if source.meaning.as_deref() == Some("sampling")
                && source.values.len() == n
                && source.alignment.as_deref().is_some_and(|id| !id.is_empty()) => {
                let current = source.alignment.as_deref().ok_or(
                    SamplingAssumptionUnionError::Invalid("raw sampling column lost alignment"),
                )?;
                if alignment.is_some_and(|previous| previous != current) {
                    return Err(SamplingAssumptionUnionError::Invalid("IID unit columns are not jointly aligned"));
                }
                alignment = Some(current);
                sampling += 1;
            }
            "interval" if source.values.len() == 2 => assumptions += 1,
            "point" if source.values.len() == 1 => {},
            _ => return Err(SamplingAssumptionUnionError::Invalid(
                "IID bootstrap accepts only aligned raw sampling columns and fixed point or interval sources",
            )),
        }
    }
    if sampling == 0 || assumptions == 0 {
        return Err(SamplingAssumptionUnionError::Invalid("IID bootstrap needs raw units and an assumption interval"));
    }
    Ok(())
}

/// Compile a raw IID sample and its complete resampling algorithm before
/// drawing any bootstrap rows. The central chain compiler checks the original
/// terminal coordinate and every generated replicate is recompiled at execute.
///
/// # Errors
/// Malformed row identities, unaligned columns, non-IID source declaration,
/// unsupported chain, level or resource bound.
pub fn compile_iid_sampling_assumption_union(
    request: IidSamplingAssumptionUnionRequest, ctx: &ExecutionContext,
) -> Result<CompiledIidSamplingAssumptionUnion, SamplingAssumptionUnionError> {
    validate_raw_unit_source(&request.original, &request.unit_ids,
        request.bootstrap_replicates, request.level)?;
    let source_bytes = antecedent_io::to_cbor(&request)?;
    if source_bytes.len() > MAX_ARTIFACT_BYTES {
        return Err(SamplingAssumptionUnionError::Invalid("IID bootstrap source exceeds its byte bound"));
    }
    let original = compile_propagation_chain(request.original.clone(), ctx)?;
    let source_digest = CanonicalWriter::digest_bytes(
        "antecedent.iid_sampling_assumption_union.source.v1", &source_bytes,
    );
    let seed = ctx.rng.master_seed();
    let mut writer = CanonicalWriter::new("antecedent.iid_sampling_assumption_union.question.v1");
    writer.bytes(&source_digest).word(seed);
    let query = writer.finish();
    let design = DesignShape::quoted(request.unit_ids.len() as u64,
        request.original.sources.len() as u64, true);
    let scientific = ScientificProblem {
        runtime_query: None, identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::IdentifiedSet, outcome: None, query,
        },
        plan_id: Arc::from("iid_sampling_assumption_union.v1"),
        data: DataClassification::Tabular,
        graph_class: "compiler_resampled_iid_assumption_set",
        query_variables: Arc::from([]), source_digest: Some(source_digest),
        support_coordinate: None, supported: false, support_status: None,
    };
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SharedEvidenceContract { contract: query },
    ));
    let score = ScoreModel::new("iid_whole_chain_assumption_endpoint_bootstrap")
        .map_err(|_| SamplingAssumptionUnionError::Invalid("invalid IID bootstrap score identity"))?;
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
        composition: Composition::Chained { stages: request.bootstrap_replicates + 1 },
        decision: None, planning: None, chain: None,
    };
    let physical = PhysicalLayer {
        operators: Arc::from([
            PhysicalOperator::declare(PhysicalEffect::Read)
                .estimates(ByteEstimates::absent().with_output(source_bytes.len() as u64))
                .build().map_err(|_| SamplingAssumptionUnionError::Invalid("invalid IID read operator"))?,
            PhysicalOperator::declare(PhysicalEffect::Select)
                .build().map_err(|_| SamplingAssumptionUnionError::Invalid("invalid IID resampling operator"))?,
        ]), design,
    };
    let lowered = Arc::new(StandaloneCompositionLoweredParts {
        cell: ProgramCell::IidSamplingAssumptionUnion,
        scientific, identification, statistical, evidence, physical,
        claim: ClaimDemand { uncertainty: UncertaintyReading::Sampling,
            kind: Some(ClaimKind::Bounds) },
    }.finish());
    Ok(CompiledIidSamplingAssumptionUnion { request, original, seed, lowered })
}

fn uniform_row(rng: &mut antecedent_core::CausalRng, n: usize) -> usize {
    let bound = n as u64;
    let threshold = bound.wrapping_neg() % bound;
    loop {
        let value = rng.next_u64();
        if value >= threshold { return (value % bound) as usize; }
    }
}

impl CompiledIidSamplingAssumptionUnion {
    /// Six-layer identity of the compiler-generated IID bootstrap.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram { &self.lowered }

    /// Inspect the retained program without sampling rows.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }

    /// Resample every raw-unit column jointly, recompile the whole chain for
    /// every replicate and form the two endpoint bounds. Coverage remains
    /// `not_measured` until the exact algorithm earns a record.
    ///
    /// # Errors
    /// Changed seed, cancellation, nested chain or nonfinite bound.
    pub fn execute(
        &self, ctx: &ExecutionContext,
    ) -> Result<SamplingAssumptionUnionResult, SamplingAssumptionUnionError> {
        if ctx.rng.master_seed() != self.seed {
            return Err(SamplingAssumptionUnionError::Invalid("IID bootstrap execution seed changed"));
        }
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        let original = self.original.execute(ctx)?;
        let (original_lower, original_upper) = endpoint_means(&original)?;
        let mut lows = Vec::with_capacity(self.request.bootstrap_replicates);
        let mut highs = Vec::with_capacity(self.request.bootstrap_replicates);
        let mut chain_identities = vec![original.identity().to_owned()];
        let n = self.request.unit_ids.len();
        let mut rng = scoped.rng.stream(0x1144_4253_5452_4150);
        for replicate in 0..self.request.bootstrap_replicates {
            if scoped.cancellation.is_cancelled() {
                return Err(SamplingAssumptionUnionError::Invalid("IID whole-chain bootstrap was cancelled"));
            }
            let indices: Vec<usize> = (0..n).map(|_| uniform_row(&mut rng, n)).collect();
            let mut draft = self.request.original.clone();
            for (resampled, source) in draft.sources.iter_mut().zip(&self.request.original.sources) {
                if source.representation == "draws" {
                    resampled.values = indices.iter().map(|&row| source.values[row]).collect();
                    resampled.alignment = Some(format!("iid-bootstrap:{}:{replicate}", self.lowered.identity().to_hex()));
                }
            }
            let chain = compile_propagation_chain(draft, ctx)?.execute(ctx)?;
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
            return Err(SamplingAssumptionUnionError::Invalid("IID bootstrap endpoint union is nonfinite or reversed"));
        }
        Ok(SamplingAssumptionUnionResult {
            original_lower, original_upper, union_lower, union_upper,
            level: self.request.level, calibration: "not_measured",
            chain_identities, lowered_program_identity: self.lowered.identity(),
        })
    }

    /// Expose the exact whole-chain construction for a prospective coverage
    /// experiment. The event to score is containment of both true set
    /// endpoints; this does not activate or earn a confidence claim.
    ///
    /// # Errors
    /// A route refusal or a malformed fixed-stage template.
    pub fn calibration_candidate(
        &self, ctx: &ExecutionContext,
    ) -> Result<IidSamplingAssumptionCalibrationCandidate, SamplingAssumptionUnionError> {
        let result = self.execute(ctx)?;
        let mut template = self.request.original.clone();
        for source in &mut template.sources {
            if source.representation == "draws" {
                source.values.clear();
                source.alignment = Some("iid_raw_units".into());
            }
        }
        let template_digest = CanonicalWriter::digest_bytes(
            "antecedent.iid_sampling_assumption_union.template.v1",
            &antecedent_io::to_cbor(&template)?,
        );
        let functional = format!(
            "joint_set_endpoints:template={}:replicates={}",
            SemanticDigest::from_bytes(template_digest).to_hex(),
            self.request.bootstrap_replicates,
        );
        let basis = CalibrationBasis::new(
            [
                "IdentifiedSet", "None", "fixed", "tabular", "Frequentist",
                IID_FEATURE, "bonferroni_basic_bootstrap", "",
                "iid_units_joint_resampling", "", functional.as_str(),
            ].map(Arc::from),
            self.request.level, Arc::from("partial"), self.request.unit_ids.len() as u64,
            Some(self.request.bootstrap_replicates as u32), None, 0.0,
        ).with_validation("none");
        Ok(IidSamplingAssumptionCalibrationCandidate { result, basis })
    }

    /// Seal the raw rows, unit identities, resampling rule and executed result.
    ///
    /// # Errors
    /// A changed run or oversized artifact refuses.
    pub fn export(
        &self, result: &SamplingAssumptionUnionResult, ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, SamplingAssumptionUnionError> {
        let live = self.execute(ctx)?;
        if fingerprint(result) != fingerprint(&live) {
            return Err(SamplingAssumptionUnionError::Invalid("IID bootstrap result changed"));
        }
        let bytes = antecedent_io::to_cbor(&IidSamplingAssumptionUnionWire {
            version: 1, required_feature: IID_FEATURE.into(),
            request: self.request.clone(), seed: self.seed,
            program_identity: *self.lowered.identity().as_bytes(),
            result_fingerprint: fingerprint(&live),
        })?;
        if bytes.len() > MAX_ARTIFACT_BYTES {
            return Err(SamplingAssumptionUnionError::Invalid("IID bootstrap artifact exceeds its byte bound"));
        }
        Ok(bytes)
    }
}

/// Independently recompile and rerun all IID bootstrap draws from a portable
/// source declaration and an externally retained six-layer program identity.
///
/// # Errors
/// A foreign feature, changed row schedule, chain or result refuses.
pub fn consume_iid_sampling_assumption_union(
    bytes: &[u8], expected_program: SemanticDigest, ctx: &ExecutionContext,
) -> Result<SamplingAssumptionUnionResult, SamplingAssumptionUnionError> {
    if bytes.len() > MAX_ARTIFACT_BYTES {
        return Err(SamplingAssumptionUnionError::Invalid("IID bootstrap artifact exceeds its byte bound"));
    }
    let wire: IidSamplingAssumptionUnionWire = antecedent_io::from_cbor(bytes)?;
    if wire.version != 1 || wire.required_feature != IID_FEATURE
        || wire.program_identity != *expected_program.as_bytes()
        || wire.seed != ctx.rng.master_seed()
    {
        return Err(SamplingAssumptionUnionError::Invalid("IID bootstrap feature, identity or seed differs"));
    }
    let plan = compile_iid_sampling_assumption_union(wire.request, ctx)?;
    if plan.lowered().identity() != expected_program {
        return Err(SamplingAssumptionUnionError::Invalid("recompiled IID bootstrap program differs"));
    }
    let live = plan.execute(ctx)?;
    if fingerprint(&live) != wire.result_fingerprint {
        return Err(SamplingAssumptionUnionError::Invalid("IID bootstrap result differs on replay"));
    }
    Ok(live)
}

/// Raw sampling rows grouped into independent clusters. Every bootstrap draw
/// selects complete clusters with replacement, keeping their member rows and
/// every aligned source column together. Cluster independence is a declared
/// design assumption, not inferred from the rows.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ClusterSamplingAssumptionUnionRequest {
    /// Complete chain over raw unit rows and fixed assumptions.
    pub original: PropagationChainRecordWire,
    /// Distinct row identities in the common sampling-source order.
    pub unit_ids: Vec<String>,
    /// Cluster identity for each row, aligned with `unit_ids`.
    pub cluster_ids: Vec<String>,
    /// Number of whole-chain cluster bootstrap replicates.
    pub bootstrap_replicates: usize,
    /// Nominal two-sided level; coverage remains unmeasured.
    pub level: f64,
}

/// Retained cluster resampling and complete-chain endpoint union.
#[derive(Clone, Debug)]
pub struct CompiledClusterSamplingAssumptionUnion {
    request: ClusterSamplingAssumptionUnionRequest,
    groups: Vec<Vec<usize>>,
    original: PropagationChainPlan,
    seed: u64,
    lowered: Arc<LoweredProgram>,
}

/// Candidate for simultaneous identified-set endpoint coverage by cluster.
#[derive(Clone, Debug)]
pub struct ClusterSamplingAssumptionCalibrationCandidate {
    /// Complete compiler-generated cluster bootstrap result.
    pub result: SamplingAssumptionUnionResult,
    /// Exact fixed chain template and cluster-design calibration coordinate.
    pub basis: CalibrationBasis,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ClusterSamplingAssumptionUnionWire {
    version: u32,
    required_feature: String,
    request: ClusterSamplingAssumptionUnionRequest,
    seed: u64,
    program_identity: [u8; 32],
    result_fingerprint: [u8; 32],
}

/// Compile complete-cluster resampling before drawing any replicate rows.
/// Every generated chain is recompiled at execution, and the portable
/// consumer reconstructs the same cluster schedule from the raw declaration.
///
/// # Errors
/// Malformed unit or cluster identities, invalid fixed assumptions, unsupported
/// chain, level or resource bound.
pub fn compile_cluster_sampling_assumption_union(
    request: ClusterSamplingAssumptionUnionRequest, ctx: &ExecutionContext,
) -> Result<CompiledClusterSamplingAssumptionUnion, SamplingAssumptionUnionError> {
    validate_raw_unit_source(&request.original, &request.unit_ids,
        request.bootstrap_replicates, request.level)?;
    if request.cluster_ids.len() != request.unit_ids.len()
        || request.cluster_ids.iter().any(|id| id.trim().is_empty())
    {
        return Err(SamplingAssumptionUnionError::Invalid("cluster identities must align with raw units"));
    }
    let mut grouped: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (row, id) in request.cluster_ids.iter().enumerate() {
        grouped.entry(id).or_default().push(row);
    }
    let groups: Vec<Vec<usize>> = grouped.into_values().collect();
    let largest = groups.iter().map(Vec::len).max().unwrap_or(0);
    if groups.len() < 10 || groups.len() > request.unit_ids.len()
        || groups.iter().all(|group| group.len() == 1)
        || largest.checked_mul(groups.len())
            .and_then(|rows| rows.checked_mul(request.bootstrap_replicates))
            .is_none_or(|work| work > 50_000_000)
    {
        return Err(SamplingAssumptionUnionError::Invalid("cluster bootstrap needs bounded independent groups with repeated members"));
    }
    let source_bytes = antecedent_io::to_cbor(&request)?;
    if source_bytes.len() > MAX_ARTIFACT_BYTES {
        return Err(SamplingAssumptionUnionError::Invalid("cluster bootstrap source exceeds its byte bound"));
    }
    let original = compile_propagation_chain(request.original.clone(), ctx)?;
    let source_digest = CanonicalWriter::digest_bytes(
        "antecedent.cluster_sampling_assumption_union.source.v1", &source_bytes,
    );
    let seed = ctx.rng.master_seed();
    let mut writer = CanonicalWriter::new("antecedent.cluster_sampling_assumption_union.question.v1");
    writer.bytes(&source_digest).word(seed);
    let query = writer.finish();
    let design = DesignShape::quoted(largest.saturating_mul(groups.len()) as u64,
        request.original.sources.len() as u64, true);
    let scientific = ScientificProblem {
        runtime_query: None, identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::IdentifiedSet, outcome: None, query,
        },
        plan_id: Arc::from("cluster_sampling_assumption_union.v1"),
        data: DataClassification::Tabular,
        graph_class: "compiler_resampled_cluster_assumption_set",
        query_variables: Arc::from([]), source_digest: Some(source_digest),
        support_coordinate: None, supported: false, support_status: None,
    };
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SharedEvidenceContract { contract: query },
    ));
    let score = ScoreModel::new("cluster_whole_chain_assumption_endpoint_bootstrap")
        .map_err(|_| SamplingAssumptionUnionError::Invalid("invalid cluster bootstrap score identity"))?;
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
        composition: Composition::Chained { stages: request.bootstrap_replicates + 1 },
        decision: None, planning: None, chain: None,
    };
    let physical = PhysicalLayer {
        operators: Arc::from([
            PhysicalOperator::declare(PhysicalEffect::Read)
                .estimates(ByteEstimates::absent().with_output(source_bytes.len() as u64))
                .build().map_err(|_| SamplingAssumptionUnionError::Invalid("invalid cluster read operator"))?,
            PhysicalOperator::declare(PhysicalEffect::Select)
                .build().map_err(|_| SamplingAssumptionUnionError::Invalid("invalid cluster resampling operator"))?,
        ]), design,
    };
    let lowered = Arc::new(StandaloneCompositionLoweredParts {
        cell: ProgramCell::ClusterSamplingAssumptionUnion,
        scientific, identification, statistical, evidence, physical,
        claim: ClaimDemand { uncertainty: UncertaintyReading::Sampling,
            kind: Some(ClaimKind::Bounds) },
    }.finish());
    Ok(CompiledClusterSamplingAssumptionUnion {
        request, groups, original, seed, lowered,
    })
}

impl CompiledClusterSamplingAssumptionUnion {
    /// Complete six-layer cluster resampling program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram { &self.lowered }

    /// Inspect the fixed schedule before sampling any clusters.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }

    /// Resample entire clusters, rerun every aligned chain source and form the
    /// unmeasured Bonferroni basic-bootstrap identified-set union.
    ///
    /// # Errors
    /// Changed seed, cancellation, nested chain or nonfinite bound.
    pub fn execute(
        &self, ctx: &ExecutionContext,
    ) -> Result<SamplingAssumptionUnionResult, SamplingAssumptionUnionError> {
        if ctx.rng.master_seed() != self.seed {
            return Err(SamplingAssumptionUnionError::Invalid("cluster bootstrap execution seed changed"));
        }
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        let original = self.original.execute(ctx)?;
        let (original_lower, original_upper) = endpoint_means(&original)?;
        let mut lows = Vec::with_capacity(self.request.bootstrap_replicates);
        let mut highs = Vec::with_capacity(self.request.bootstrap_replicates);
        let mut chain_identities = vec![original.identity().to_owned()];
        let mut rng = scoped.rng.stream(0x434C_5553_5445_5253);
        for replicate in 0..self.request.bootstrap_replicates {
            if scoped.cancellation.is_cancelled() {
                return Err(SamplingAssumptionUnionError::Invalid("cluster bootstrap was cancelled"));
            }
            let mut indices = Vec::new();
            for _ in 0..self.groups.len() {
                let group = &self.groups[uniform_row(&mut rng, self.groups.len())];
                indices.extend(group.iter().copied());
            }
            let mut draft = self.request.original.clone();
            for (resampled, source) in draft.sources.iter_mut().zip(&self.request.original.sources) {
                if source.representation == "draws" {
                    resampled.values = indices.iter().map(|&row| source.values[row]).collect();
                    resampled.alignment = Some(format!(
                        "cluster-bootstrap:{}:{replicate}", self.lowered.identity().to_hex(),
                    ));
                }
            }
            let chain = compile_propagation_chain(draft, ctx)?.execute(ctx)?;
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
            return Err(SamplingAssumptionUnionError::Invalid("cluster bootstrap union is nonfinite or reversed"));
        }
        Ok(SamplingAssumptionUnionResult {
            original_lower, original_upper, union_lower, union_upper,
            level: self.request.level, calibration: "not_measured",
            chain_identities, lowered_program_identity: self.lowered.identity(),
        })
    }

    /// Exact cluster-design and fixed-chain coordinate for a prospective
    /// simultaneous endpoint coverage record.
    ///
    /// # Errors
    /// Complete-chain execution or template encoding refused.
    pub fn calibration_candidate(
        &self, ctx: &ExecutionContext,
    ) -> Result<ClusterSamplingAssumptionCalibrationCandidate, SamplingAssumptionUnionError> {
        let result = self.execute(ctx)?;
        let mut template = self.request.original.clone();
        for source in &mut template.sources {
            if source.representation == "draws" {
                source.values.clear();
                source.alignment = Some("cluster_raw_units".into());
            }
        }
        let template_digest = CanonicalWriter::digest_bytes(
            "antecedent.cluster_sampling_assumption_union.template.v1",
            &antecedent_io::to_cbor(&template)?,
        );
        let mut writer = CanonicalWriter::new(
            "antecedent.cluster_sampling_assumption_union.calibration_scope.v1",
        );
        writer.bytes(&template_digest).word(self.request.bootstrap_replicates as u64)
            .word(self.groups.len() as u64);
        for group in &self.groups {
            writer.word(group.len() as u64);
            for &row in group { writer.word(row as u64); }
        }
        let scope = SemanticDigest::from_bytes(writer.finish()).to_hex();
        let functional = format!("joint_set_endpoints:cluster_scope={scope}");
        let basis = CalibrationBasis::new(
            [
                "IdentifiedSet", "None", "fixed", "tabular", "Frequentist",
                CLUSTER_FEATURE, "bonferroni_basic_cluster_bootstrap", "",
                "independent_clusters_whole_group_resampling", "", functional.as_str(),
            ].map(Arc::from),
            self.request.level, Arc::from("partial"), self.request.unit_ids.len() as u64,
            Some(self.request.bootstrap_replicates as u32), None, 0.0,
        ).with_validation("none");
        Ok(ClusterSamplingAssumptionCalibrationCandidate { result, basis })
    }

    /// Seal the raw rows, cluster schedule and complete result for replay.
    ///
    /// # Errors
    /// A changed run or oversized artifact refuses.
    pub fn export(
        &self, result: &SamplingAssumptionUnionResult, ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, SamplingAssumptionUnionError> {
        if fingerprint(&self.execute(ctx)?) != fingerprint(result) {
            return Err(SamplingAssumptionUnionError::Invalid("cluster bootstrap result changed"));
        }
        let bytes = antecedent_io::to_cbor(&ClusterSamplingAssumptionUnionWire {
            version: 1, required_feature: CLUSTER_FEATURE.into(),
            request: self.request.clone(), seed: self.seed,
            program_identity: *self.lowered.identity().as_bytes(),
            result_fingerprint: fingerprint(result),
        })?;
        if bytes.len() > MAX_ARTIFACT_BYTES {
            return Err(SamplingAssumptionUnionError::Invalid("cluster bootstrap artifact exceeds its byte bound"));
        }
        Ok(bytes)
    }
}

/// Recompile and rerun the complete cluster schedule under a separately
/// retained program identity.
///
/// # Errors
/// A foreign feature, changed raw data, cluster mapping, seed or bound.
pub fn consume_cluster_sampling_assumption_union(
    bytes: &[u8], expected_program: SemanticDigest, ctx: &ExecutionContext,
) -> Result<SamplingAssumptionUnionResult, SamplingAssumptionUnionError> {
    if bytes.len() > MAX_ARTIFACT_BYTES {
        return Err(SamplingAssumptionUnionError::Invalid("cluster bootstrap artifact exceeds its byte bound"));
    }
    let wire: ClusterSamplingAssumptionUnionWire = antecedent_io::from_cbor(bytes)?;
    if wire.version != 1 || wire.required_feature != CLUSTER_FEATURE
        || wire.program_identity != *expected_program.as_bytes()
        || wire.seed != ctx.rng.master_seed()
    {
        return Err(SamplingAssumptionUnionError::Invalid("cluster bootstrap feature, identity or seed differs"));
    }
    let plan = compile_cluster_sampling_assumption_union(wire.request, ctx)?;
    if plan.lowered().identity() != expected_program {
        return Err(SamplingAssumptionUnionError::Invalid("recompiled cluster bootstrap program differs"));
    }
    let live = plan.execute(ctx)?;
    if fingerprint(&live) != wire.result_fingerprint {
        return Err(SamplingAssumptionUnionError::Invalid("cluster bootstrap result differs on replay"));
    }
    Ok(live)
}

/// Clustered sampling rows with fixed, prespecified strata. Each replicate
/// samples the original number of complete clusters independently with replacement
/// within each stratum, preserving all member rows and aligned columns. Clusters
/// are assumed independent across and identically distributed within each stratum;
/// this is not inferred from rows. Unequal cluster sizes are retained, so replicate
/// row counts may vary and the terminal mean remains row-weighted.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StratifiedClusterSamplingAssumptionUnionRequest {
    /// Complete chain over raw unit rows and fixed assumptions.
    pub original: PropagationChainRecordWire,
    /// Distinct row identities in the common sampling-source order.
    pub unit_ids: Vec<String>,
    /// Cluster identity for each row, aligned with `unit_ids`.
    pub cluster_ids: Vec<String>,
    /// Stratum for each row; every cluster must belong to exactly one stratum.
    pub stratum_ids: Vec<String>,
    /// Number of whole-chain cluster bootstrap replicates.
    pub bootstrap_replicates: usize,
    /// Nominal two-sided level; coverage remains unmeasured.
    pub level: f64,
}

/// Retained cluster resampling and complete-chain endpoint union.
#[derive(Clone, Debug)]
pub struct CompiledStratifiedClusterSamplingAssumptionUnion {
    request: StratifiedClusterSamplingAssumptionUnionRequest,
    strata: Vec<Vec<Vec<usize>>>,
    original: PropagationChainPlan,
    seed: u64,
    lowered: Arc<LoweredProgram>,
}

/// Candidate for simultaneous identified-set endpoint coverage by cluster.
#[derive(Clone, Debug)]
pub struct StratifiedClusterSamplingAssumptionCalibrationCandidate {
    /// Complete compiler-generated cluster bootstrap result.
    pub result: SamplingAssumptionUnionResult,
    /// Exact fixed chain template and cluster-design calibration coordinate.
    pub basis: CalibrationBasis,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct StratifiedClusterSamplingAssumptionUnionWire {
    version: u32,
    required_feature: String,
    request: StratifiedClusterSamplingAssumptionUnionRequest,
    seed: u64,
    program_identity: [u8; 32],
    result_fingerprint: [u8; 32],
}

/// Compile stratified whole-cluster resampling before drawing any replicate rows.
/// Every generated chain is recompiled at execution, and the portable
/// consumer reconstructs the same cluster schedule from the raw declaration.
///
/// # Errors
/// Malformed unit or cluster identities, invalid fixed assumptions, unsupported
/// chain, level or resource bound.
pub fn compile_stratified_cluster_sampling_assumption_union(
    request: StratifiedClusterSamplingAssumptionUnionRequest, ctx: &ExecutionContext,
) -> Result<CompiledStratifiedClusterSamplingAssumptionUnion, SamplingAssumptionUnionError> {
    validate_raw_unit_source(&request.original, &request.unit_ids,
        request.bootstrap_replicates, request.level)?;
    if request.cluster_ids.len() != request.unit_ids.len()
        || request.stratum_ids.len() != request.unit_ids.len()
        || request.cluster_ids.iter().chain(&request.stratum_ids).any(|id| id.trim().is_empty())
    {
        return Err(SamplingAssumptionUnionError::Invalid("stratum and cluster identities must align with raw units"));
    }
    let mut owner: BTreeMap<&str, &str> = BTreeMap::new();
    let mut grouped: BTreeMap<&str, BTreeMap<&str, Vec<usize>>> = BTreeMap::new();
    for row in 0..request.unit_ids.len() {
        let cluster = request.cluster_ids[row].as_str();
        let stratum = request.stratum_ids[row].as_str();
        if owner.insert(cluster, stratum).is_some_and(|previous| previous != stratum) {
            return Err(SamplingAssumptionUnionError::Invalid("a cluster cannot cross strata"));
        }
        grouped.entry(stratum).or_default().entry(cluster).or_default().push(row);
    }
    let strata: Vec<Vec<Vec<usize>>> = grouped.into_values()
        .map(|groups| groups.into_values().collect()).collect();
    let max_rows = strata.iter().try_fold(0usize, |sum, groups| {
        groups.iter().map(Vec::len).max().and_then(|largest|
            largest.checked_mul(groups.len())).and_then(|rows| sum.checked_add(rows))
    });
    if strata.len() < 2 || strata.iter().any(|groups| groups.len() < 2)
        || strata.iter().flatten().all(|group| group.len() == 1)
        || max_rows.and_then(|rows| rows.checked_mul(request.bootstrap_replicates))
            .is_none_or(|work| work > 50_000_000)
    {
        return Err(SamplingAssumptionUnionError::Invalid("stratified cluster bootstrap needs at least two strata, two clusters per stratum, and bounded work"));
    }
    let max_rows = max_rows.ok_or(SamplingAssumptionUnionError::Invalid("stratified cluster size overflow"))?;
    let source_bytes = antecedent_io::to_cbor(&request)?;
    if source_bytes.len() > MAX_ARTIFACT_BYTES {
        return Err(SamplingAssumptionUnionError::Invalid("cluster bootstrap source exceeds its byte bound"));
    }
    let original = compile_propagation_chain(request.original.clone(), ctx)?;
    let source_digest = CanonicalWriter::digest_bytes(
        "antecedent.stratified_cluster_sampling_assumption_union.source.v1", &source_bytes,
    );
    let seed = ctx.rng.master_seed();
    let mut writer = CanonicalWriter::new("antecedent.stratified_cluster_sampling_assumption_union.question.v1");
    writer.bytes(&source_digest).word(seed);
    let query = writer.finish();
    let design = DesignShape::quoted(max_rows as u64,
        request.original.sources.len() as u64, true);
    let scientific = ScientificProblem {
        runtime_query: None, identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::IdentifiedSet, outcome: None, query,
        },
        plan_id: Arc::from("stratified_cluster_sampling_assumption_union.v1"),
        data: DataClassification::Tabular,
        graph_class: "compiler_resampled_stratified_cluster_assumption_set",
        query_variables: Arc::from([]), source_digest: Some(source_digest),
        support_coordinate: None, supported: false, support_status: None,
    };
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SharedEvidenceContract { contract: query },
    ));
    let score = ScoreModel::new("stratified_cluster_whole_chain_assumption_endpoint_bootstrap")
        .map_err(|_| SamplingAssumptionUnionError::Invalid("invalid cluster bootstrap score identity"))?;
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
        composition: Composition::Chained { stages: request.bootstrap_replicates + 1 },
        decision: None, planning: None, chain: None,
    };
    let physical = PhysicalLayer {
        operators: Arc::from([
            PhysicalOperator::declare(PhysicalEffect::Read)
                .estimates(ByteEstimates::absent().with_output(source_bytes.len() as u64))
                .build().map_err(|_| SamplingAssumptionUnionError::Invalid("invalid cluster read operator"))?,
            PhysicalOperator::declare(PhysicalEffect::Select)
                .build().map_err(|_| SamplingAssumptionUnionError::Invalid("invalid cluster resampling operator"))?,
        ]), design,
    };
    let lowered = Arc::new(StandaloneCompositionLoweredParts {
        cell: ProgramCell::ClusterSamplingAssumptionUnion,
        scientific, identification, statistical, evidence, physical,
        claim: ClaimDemand { uncertainty: UncertaintyReading::Sampling,
            kind: Some(ClaimKind::Bounds) },
    }.finish());
    Ok(CompiledStratifiedClusterSamplingAssumptionUnion {
        request, strata, original, seed, lowered,
    })
}

fn stratified_cluster_indices(strata: &[Vec<Vec<usize>>], rng: &mut antecedent_core::CausalRng) -> Vec<usize> {
    let mut indices = Vec::new();
    for groups in strata {
        for _ in 0..groups.len() {
            indices.extend(groups[uniform_row(rng, groups.len())].iter().copied());
        }
    }
    indices
}

impl CompiledStratifiedClusterSamplingAssumptionUnion {
    /// Complete six-layer cluster resampling program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram { &self.lowered }

    /// Inspect the fixed schedule before sampling any clusters.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }

    /// Resample entire clusters within strata, rerun every aligned chain source and form the
    /// unmeasured Bonferroni basic-bootstrap identified-set union.
    ///
    /// # Errors
    /// Changed seed, cancellation, nested chain or nonfinite bound.
    pub fn execute(
        &self, ctx: &ExecutionContext,
    ) -> Result<SamplingAssumptionUnionResult, SamplingAssumptionUnionError> {
        if ctx.rng.master_seed() != self.seed {
            return Err(SamplingAssumptionUnionError::Invalid("cluster bootstrap execution seed changed"));
        }
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        let original = self.original.execute(ctx)?;
        let (original_lower, original_upper) = endpoint_means(&original)?;
        let mut lows = Vec::with_capacity(self.request.bootstrap_replicates);
        let mut highs = Vec::with_capacity(self.request.bootstrap_replicates);
        let mut chain_identities = vec![original.identity().to_owned()];
        let mut rng = scoped.rng.stream(0x5354_5241_5443_4C55);
        for replicate in 0..self.request.bootstrap_replicates {
            if scoped.cancellation.is_cancelled() {
                return Err(SamplingAssumptionUnionError::Invalid("cluster bootstrap was cancelled"));
            }
            let indices = stratified_cluster_indices(&self.strata, &mut rng);
            let mut draft = self.request.original.clone();
            for (resampled, source) in draft.sources.iter_mut().zip(&self.request.original.sources) {
                if source.representation == "draws" {
                    resampled.values = indices.iter().map(|&row| source.values[row]).collect();
                    resampled.alignment = Some(format!(
                        "stratified-cluster-bootstrap:{}:{replicate}", self.lowered.identity().to_hex(),
                    ));
                }
            }
            let chain = compile_propagation_chain(draft, ctx)?.execute(ctx)?;
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
            return Err(SamplingAssumptionUnionError::Invalid("cluster bootstrap union is nonfinite or reversed"));
        }
        Ok(SamplingAssumptionUnionResult {
            original_lower, original_upper, union_lower, union_upper,
            level: self.request.level, calibration: "not_measured",
            chain_identities, lowered_program_identity: self.lowered.identity(),
        })
    }

    /// Exact cluster-design and fixed-chain coordinate for a prospective
    /// simultaneous endpoint coverage record.
    ///
    /// # Errors
    /// Complete-chain execution or template encoding refused.
    pub fn calibration_candidate(
        &self, ctx: &ExecutionContext,
    ) -> Result<StratifiedClusterSamplingAssumptionCalibrationCandidate, SamplingAssumptionUnionError> {
        let result = self.execute(ctx)?;
        let mut template = self.request.original.clone();
        for source in &mut template.sources {
            if source.representation == "draws" {
                source.values.clear();
                source.alignment = Some("stratified_cluster_raw_units".into());
            }
        }
        let template_digest = CanonicalWriter::digest_bytes(
            "antecedent.stratified_cluster_sampling_assumption_union.template.v1",
            &antecedent_io::to_cbor(&template)?,
        );
        let mut writer = CanonicalWriter::new(
            "antecedent.stratified_cluster_sampling_assumption_union.calibration_scope.v1",
        );
        writer.bytes(&template_digest).word(self.request.bootstrap_replicates as u64)
            .word(self.strata.len() as u64);
        for groups in &self.strata {
            writer.word(groups.len() as u64);
            for group in groups {
                writer.word(group.len() as u64);
                for &row in group { writer.word(row as u64); }
            }
        }
        let scope = SemanticDigest::from_bytes(writer.finish()).to_hex();
        let functional = format!("joint_set_endpoints:stratified_cluster_scope={scope}");
        let basis = CalibrationBasis::new(
            [
                "IdentifiedSet", "None", "fixed", "tabular", "Frequentist",
                STRATIFIED_CLUSTER_FEATURE, "bonferroni_basic_stratified_cluster_bootstrap", "",
                "independent_clusters_within_fixed_strata", "", functional.as_str(),
            ].map(Arc::from),
            self.request.level, Arc::from("partial"), self.request.unit_ids.len() as u64,
            Some(self.request.bootstrap_replicates as u32), None, 0.0,
        ).with_validation("none");
        Ok(StratifiedClusterSamplingAssumptionCalibrationCandidate { result, basis })
    }

    /// Seal the raw rows, cluster schedule and complete result for replay.
    ///
    /// # Errors
    /// A changed run or oversized artifact refuses.
    pub fn export(
        &self, result: &SamplingAssumptionUnionResult, ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, SamplingAssumptionUnionError> {
        if fingerprint(&self.execute(ctx)?) != fingerprint(result) {
            return Err(SamplingAssumptionUnionError::Invalid("cluster bootstrap result changed"));
        }
        let bytes = antecedent_io::to_cbor(&StratifiedClusterSamplingAssumptionUnionWire {
            version: 1, required_feature: STRATIFIED_CLUSTER_FEATURE.into(),
            request: self.request.clone(), seed: self.seed,
            program_identity: *self.lowered.identity().as_bytes(),
            result_fingerprint: fingerprint(result),
        })?;
        if bytes.len() > MAX_ARTIFACT_BYTES {
            return Err(SamplingAssumptionUnionError::Invalid("cluster bootstrap artifact exceeds its byte bound"));
        }
        Ok(bytes)
    }
}

/// Recompile and rerun the complete cluster schedule under a separately
/// retained program identity.
///
/// # Errors
/// A foreign feature, changed raw data, cluster mapping, seed or bound.
pub fn consume_stratified_cluster_sampling_assumption_union(
    bytes: &[u8], expected_program: SemanticDigest, ctx: &ExecutionContext,
) -> Result<SamplingAssumptionUnionResult, SamplingAssumptionUnionError> {
    if bytes.len() > MAX_ARTIFACT_BYTES {
        return Err(SamplingAssumptionUnionError::Invalid("cluster bootstrap artifact exceeds its byte bound"));
    }
    let wire: StratifiedClusterSamplingAssumptionUnionWire = antecedent_io::from_cbor(bytes)?;
    if wire.version != 1 || wire.required_feature != STRATIFIED_CLUSTER_FEATURE
        || wire.program_identity != *expected_program.as_bytes()
        || wire.seed != ctx.rng.master_seed()
    {
        return Err(SamplingAssumptionUnionError::Invalid("cluster bootstrap feature, identity or seed differs"));
    }
    let plan = compile_stratified_cluster_sampling_assumption_union(wire.request, ctx)?;
    if plan.lowered().identity() != expected_program {
        return Err(SamplingAssumptionUnionError::Invalid("recompiled cluster bootstrap program differs"));
    }
    let live = plan.execute(ctx)?;
    if fingerprint(&live) != wire.result_fingerprint {
        return Err(SamplingAssumptionUnionError::Invalid("cluster bootstrap result differs on replay"));
    }
    Ok(live)
}

/// One equally spaced ordered series. Rows and all aligned sampling columns are
/// resampled together using overlapping, non-circular blocks drawn with replacement.
/// Stationarity and sufficiently weak temporal dependence are declared assumptions,
/// not verified from these rows. The last selected block is truncated to series length.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MovingBlockSamplingAssumptionUnionRequest {
    /// Complete chain over raw unit rows and fixed assumptions.
    pub original: PropagationChainRecordWire,
    /// Distinct row identities in chronological, equally spaced sampling-source order.
    pub unit_ids: Vec<String>,
    /// Prespecified number of consecutive rows in each overlapping block.
    pub block_length: usize,
    /// Number of whole-chain moving-block bootstrap replicates.
    pub bootstrap_replicates: usize,
    /// Nominal two-sided level; coverage remains unmeasured.
    pub level: f64,
}

/// Retained moving-block resampling and complete-chain endpoint union.
#[derive(Clone, Debug)]
pub struct CompiledMovingBlockSamplingAssumptionUnion {
    request: MovingBlockSamplingAssumptionUnionRequest,
    original: PropagationChainPlan,
    seed: u64,
    lowered: Arc<LoweredProgram>,
}

/// Candidate for simultaneous identified-set endpoint coverage by moving-block.
#[derive(Clone, Debug)]
pub struct MovingBlockSamplingAssumptionCalibrationCandidate {
    /// Complete compiler-generated moving-block bootstrap result.
    pub result: SamplingAssumptionUnionResult,
    /// Exact fixed chain template and moving-block-design calibration coordinate.
    pub basis: CalibrationBasis,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct MovingBlockSamplingAssumptionUnionWire {
    version: u32,
    required_feature: String,
    request: MovingBlockSamplingAssumptionUnionRequest,
    seed: u64,
    program_identity: [u8; 32],
    result_fingerprint: [u8; 32],
}

/// Compile complete-moving-block resampling before drawing any replicate rows.
/// Every generated chain is recompiled at execution, and the portable
/// consumer reconstructs the same moving-block schedule from the raw declaration.
///
/// # Errors
/// Malformed unit or moving-block identities, invalid fixed assumptions, unsupported
/// chain, level or resource bound.
pub fn compile_moving_block_sampling_assumption_union(
    request: MovingBlockSamplingAssumptionUnionRequest, ctx: &ExecutionContext,
) -> Result<CompiledMovingBlockSamplingAssumptionUnion, SamplingAssumptionUnionError> {
    validate_raw_unit_source(&request.original, &request.unit_ids,
        request.bootstrap_replicates, request.level)?;
    let n = request.unit_ids.len();
    if request.block_length < 2 || request.block_length > n / 2 {
        return Err(SamplingAssumptionUnionError::Invalid("moving-block length must be between 2 and half the series length"));
    }
    let source_bytes = antecedent_io::to_cbor(&request)?;
    if source_bytes.len() > MAX_ARTIFACT_BYTES {
        return Err(SamplingAssumptionUnionError::Invalid("moving-block bootstrap source exceeds its byte bound"));
    }
    let original = compile_propagation_chain(request.original.clone(), ctx)?;
    let source_digest = CanonicalWriter::digest_bytes(
        "antecedent.moving_block_sampling_assumption_union.source.v1", &source_bytes,
    );
    let seed = ctx.rng.master_seed();
    let mut writer = CanonicalWriter::new("antecedent.moving_block_sampling_assumption_union.question.v1");
    writer.bytes(&source_digest).word(seed);
    let query = writer.finish();
    let design = DesignShape::quoted(request.unit_ids.len() as u64,
        request.original.sources.len() as u64, true);
    let scientific = ScientificProblem {
        runtime_query: None, identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::IdentifiedSet, outcome: None, query,
        },
        plan_id: Arc::from("moving_block_sampling_assumption_union.v1"),
        data: DataClassification::Tabular,
        graph_class: "compiler_resampled_moving_block_assumption_set",
        query_variables: Arc::from([]), source_digest: Some(source_digest),
        support_coordinate: None, supported: false, support_status: None,
    };
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SharedEvidenceContract { contract: query },
    ));
    let score = ScoreModel::new("moving_block_whole_chain_assumption_endpoint_bootstrap")
        .map_err(|_| SamplingAssumptionUnionError::Invalid("invalid moving-block bootstrap score identity"))?;
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
        composition: Composition::Chained { stages: request.bootstrap_replicates + 1 },
        decision: None, planning: None, chain: None,
    };
    let physical = PhysicalLayer {
        operators: Arc::from([
            PhysicalOperator::declare(PhysicalEffect::Read)
                .estimates(ByteEstimates::absent().with_output(source_bytes.len() as u64))
                .build().map_err(|_| SamplingAssumptionUnionError::Invalid("invalid moving-block read operator"))?,
            PhysicalOperator::declare(PhysicalEffect::Select)
                .build().map_err(|_| SamplingAssumptionUnionError::Invalid("invalid moving-block resampling operator"))?,
        ]), design,
    };
    let lowered = Arc::new(StandaloneCompositionLoweredParts {
        cell: ProgramCell::MovingBlockSamplingAssumptionUnion,
        scientific, identification, statistical, evidence, physical,
        claim: ClaimDemand { uncertainty: UncertaintyReading::Sampling,
            kind: Some(ClaimKind::Bounds) },
    }.finish());
    Ok(CompiledMovingBlockSamplingAssumptionUnion {
        request, original, seed, lowered,
    })
}

impl CompiledMovingBlockSamplingAssumptionUnion {
    /// Complete six-layer moving-block resampling program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram { &self.lowered }

    /// Inspect the fixed schedule before sampling any moving-blocks.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }

    /// Resample consecutive moving blocks, rerun every aligned chain source and form the
    /// unmeasured Bonferroni basic-bootstrap identified-set union.
    ///
    /// # Errors
    /// Changed seed, cancellation, nested chain or nonfinite bound.
    pub fn execute(
        &self, ctx: &ExecutionContext,
    ) -> Result<SamplingAssumptionUnionResult, SamplingAssumptionUnionError> {
        if ctx.rng.master_seed() != self.seed {
            return Err(SamplingAssumptionUnionError::Invalid("moving-block bootstrap execution seed changed"));
        }
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        let original = self.original.execute(ctx)?;
        let (original_lower, original_upper) = endpoint_means(&original)?;
        let mut lows = Vec::with_capacity(self.request.bootstrap_replicates);
        let mut highs = Vec::with_capacity(self.request.bootstrap_replicates);
        let mut chain_identities = vec![original.identity().to_owned()];
        let mut rng = scoped.rng.stream(0x4D4F_5649_4E47_424C);
        for replicate in 0..self.request.bootstrap_replicates {
            if scoped.cancellation.is_cancelled() {
                return Err(SamplingAssumptionUnionError::Invalid("moving-block bootstrap was cancelled"));
            }
            let n = self.request.unit_ids.len();
            let mut indices = Vec::with_capacity(n);
            while indices.len() < n {
                let start = uniform_row(&mut rng, n - self.request.block_length + 1);
                let take = self.request.block_length.min(n - indices.len());
                indices.extend(start..start + take);
            }
            let mut draft = self.request.original.clone();
            for (resampled, source) in draft.sources.iter_mut().zip(&self.request.original.sources) {
                if source.representation == "draws" {
                    resampled.values = indices.iter().map(|&row| source.values[row]).collect();
                    resampled.alignment = Some(format!(
                        "moving-block-bootstrap:{}:{replicate}", self.lowered.identity().to_hex(),
                    ));
                }
            }
            let chain = compile_propagation_chain(draft, ctx)?.execute(ctx)?;
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
            return Err(SamplingAssumptionUnionError::Invalid("moving-block bootstrap union is nonfinite or reversed"));
        }
        Ok(SamplingAssumptionUnionResult {
            original_lower, original_upper, union_lower, union_upper,
            level: self.request.level, calibration: "not_measured",
            chain_identities, lowered_program_identity: self.lowered.identity(),
        })
    }

    /// Exact moving-block-design and fixed-chain coordinate for a prospective
    /// simultaneous endpoint coverage record.
    ///
    /// # Errors
    /// Complete-chain execution or template encoding refused.
    pub fn calibration_candidate(
        &self, ctx: &ExecutionContext,
    ) -> Result<MovingBlockSamplingAssumptionCalibrationCandidate, SamplingAssumptionUnionError> {
        let result = self.execute(ctx)?;
        let mut template = self.request.original.clone();
        for source in &mut template.sources {
            if source.representation == "draws" {
                source.values.clear();
                source.alignment = Some("moving_block_raw_series".into());
            }
        }
        let template_digest = CanonicalWriter::digest_bytes(
            "antecedent.moving_block_sampling_assumption_union.template.v1",
            &antecedent_io::to_cbor(&template)?,
        );
        let mut writer = CanonicalWriter::new(
            "antecedent.moving_block_sampling_assumption_union.calibration_scope.v1",
        );
        writer.bytes(&template_digest).word(self.request.bootstrap_replicates as u64)
            .word(self.request.unit_ids.len() as u64)
            .word(self.request.block_length as u64);
        let scope = SemanticDigest::from_bytes(writer.finish()).to_hex();
        let functional = format!("joint_set_endpoints:moving_block_scope={scope}");
        let basis = CalibrationBasis::new(
            [
                "IdentifiedSet", "None", "fixed", "tabular", "Frequentist",
                MOVING_BLOCK_FEATURE, "bonferroni_basic_moving_block_bootstrap", "",
                "stationary_weak_dependence_moving_blocks", "", functional.as_str(),
            ].map(Arc::from),
            self.request.level, Arc::from("partial"), self.request.unit_ids.len() as u64,
            Some(self.request.bootstrap_replicates as u32), None, 0.0,
        ).with_validation("none");
        Ok(MovingBlockSamplingAssumptionCalibrationCandidate { result, basis })
    }

    /// Seal the raw rows, moving-block schedule and complete result for replay.
    ///
    /// # Errors
    /// A changed run or oversized artifact refuses.
    pub fn export(
        &self, result: &SamplingAssumptionUnionResult, ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, SamplingAssumptionUnionError> {
        if fingerprint(&self.execute(ctx)?) != fingerprint(result) {
            return Err(SamplingAssumptionUnionError::Invalid("moving-block bootstrap result changed"));
        }
        let bytes = antecedent_io::to_cbor(&MovingBlockSamplingAssumptionUnionWire {
            version: 1, required_feature: MOVING_BLOCK_FEATURE.into(),
            request: self.request.clone(), seed: self.seed,
            program_identity: *self.lowered.identity().as_bytes(),
            result_fingerprint: fingerprint(result),
        })?;
        if bytes.len() > MAX_ARTIFACT_BYTES {
            return Err(SamplingAssumptionUnionError::Invalid("moving-block bootstrap artifact exceeds its byte bound"));
        }
        Ok(bytes)
    }
}

/// Recompile and rerun the complete moving-block schedule under a separately
/// retained program identity.
///
/// # Errors
/// A foreign feature, changed raw data, moving-block mapping, seed or bound.
pub fn consume_moving_block_sampling_assumption_union(
    bytes: &[u8], expected_program: SemanticDigest, ctx: &ExecutionContext,
) -> Result<SamplingAssumptionUnionResult, SamplingAssumptionUnionError> {
    if bytes.len() > MAX_ARTIFACT_BYTES {
        return Err(SamplingAssumptionUnionError::Invalid("moving-block bootstrap artifact exceeds its byte bound"));
    }
    let wire: MovingBlockSamplingAssumptionUnionWire = antecedent_io::from_cbor(bytes)?;
    if wire.version != 1 || wire.required_feature != MOVING_BLOCK_FEATURE
        || wire.program_identity != *expected_program.as_bytes()
        || wire.seed != ctx.rng.master_seed()
    {
        return Err(SamplingAssumptionUnionError::Invalid("moving-block bootstrap feature, identity or seed differs"));
    }
    let plan = compile_moving_block_sampling_assumption_union(wire.request, ctx)?;
    if plan.lowered().identity() != expected_program {
        return Err(SamplingAssumptionUnionError::Invalid("recompiled moving-block bootstrap program differs"));
    }
    let live = plan.execute(ctx)?;
    if fingerprint(&live) != wire.result_fingerprint {
        return Err(SamplingAssumptionUnionError::Invalid("moving-block bootstrap result differs on replay"));
    }
    Ok(live)
}

/// Multiple independent, equally spaced ordered series. Each stratum's rows
/// appear in chronological order in the common source table, including when
/// series are interleaved. Overlapping non-circular blocks are sampled within
/// each series with replacement; the last block is truncated to its length.
/// Independence across series and stationarity and sufficiently weak dependence
/// within each series are declared, not tested.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StratifiedMovingBlockSamplingAssumptionUnionRequest {
    /// Complete chain over raw unit rows and fixed assumptions.
    pub original: PropagationChainRecordWire,
    /// Distinct row identities in chronological, equally spaced sampling-source order.
    pub unit_ids: Vec<String>,
    /// Series or stratum identity per row, in the common sampling-source order.
    pub stratum_ids: Vec<String>,
    /// Prespecified common number of consecutive rows per within-series block.
    pub block_length: usize,
    /// Number of whole-chain moving-block bootstrap replicates.
    pub bootstrap_replicates: usize,
    /// Nominal two-sided level; coverage remains unmeasured.
    pub level: f64,
}

/// Retained moving-block resampling and complete-chain endpoint union.
#[derive(Clone, Debug)]
pub struct CompiledStratifiedMovingBlockSamplingAssumptionUnion {
    request: StratifiedMovingBlockSamplingAssumptionUnionRequest,
    strata: Vec<Vec<usize>>,
    original: PropagationChainPlan,
    seed: u64,
    lowered: Arc<LoweredProgram>,
}

/// Candidate for simultaneous identified-set endpoint coverage by moving-block.
#[derive(Clone, Debug)]
pub struct StratifiedMovingBlockSamplingAssumptionCalibrationCandidate {
    /// Complete compiler-generated moving-block bootstrap result.
    pub result: SamplingAssumptionUnionResult,
    /// Exact fixed chain template and moving-block-design calibration coordinate.
    pub basis: CalibrationBasis,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct StratifiedMovingBlockSamplingAssumptionUnionWire {
    version: u32,
    required_feature: String,
    request: StratifiedMovingBlockSamplingAssumptionUnionRequest,
    seed: u64,
    program_identity: [u8; 32],
    result_fingerprint: [u8; 32],
}

/// Compile stratified moving-block resampling before drawing any replicate rows.
/// Every generated chain is recompiled at execution, and the portable
/// consumer reconstructs the same moving-block schedule from the raw declaration.
///
/// # Errors
/// Malformed unit or moving-block identities, invalid fixed assumptions, unsupported
/// chain, level or resource bound.
pub fn compile_stratified_moving_block_sampling_assumption_union(
    request: StratifiedMovingBlockSamplingAssumptionUnionRequest, ctx: &ExecutionContext,
) -> Result<CompiledStratifiedMovingBlockSamplingAssumptionUnion, SamplingAssumptionUnionError> {
    validate_raw_unit_source(&request.original, &request.unit_ids,
        request.bootstrap_replicates, request.level)?;
    let n = request.unit_ids.len();
    if request.stratum_ids.len() != n || request.stratum_ids.iter().any(|id| id.trim().is_empty()) {
        return Err(SamplingAssumptionUnionError::Invalid("strata must align with all sampling rows"));
    }
    let mut grouped: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (row, stratum) in request.stratum_ids.iter().enumerate() {
        grouped.entry(stratum.as_str()).or_default().push(row);
    }
    let strata: Vec<Vec<usize>> = grouped.into_values().collect();
    if strata.len() < 2 || request.block_length < 2
        || strata.iter().any(|rows| rows.len() < 20 || request.block_length > rows.len() / 2)
    {
        return Err(SamplingAssumptionUnionError::Invalid("stratified moving blocks need two series of at least 20 rows and a common block length between 2 and half each series"));
    }
    let source_bytes = antecedent_io::to_cbor(&request)?;
    if source_bytes.len() > MAX_ARTIFACT_BYTES {
        return Err(SamplingAssumptionUnionError::Invalid("moving-block bootstrap source exceeds its byte bound"));
    }
    let original = compile_propagation_chain(request.original.clone(), ctx)?;
    let source_digest = CanonicalWriter::digest_bytes(
        "antecedent.stratified_moving_block_sampling_assumption_union.source.v1", &source_bytes,
    );
    let seed = ctx.rng.master_seed();
    let mut writer = CanonicalWriter::new("antecedent.stratified_moving_block_sampling_assumption_union.question.v1");
    writer.bytes(&source_digest).word(seed);
    let query = writer.finish();
    let design = DesignShape::quoted(request.unit_ids.len() as u64,
        request.original.sources.len() as u64, true);
    let scientific = ScientificProblem {
        runtime_query: None, identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::IdentifiedSet, outcome: None, query,
        },
        plan_id: Arc::from("stratified_moving_block_sampling_assumption_union.v1"),
        data: DataClassification::Tabular,
        graph_class: "compiler_resampled_stratified_moving_block_assumption_set",
        query_variables: Arc::from([]), source_digest: Some(source_digest),
        support_coordinate: None, supported: false, support_status: None,
    };
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SharedEvidenceContract { contract: query },
    ));
    let score = ScoreModel::new("stratified_moving_block_whole_chain_assumption_endpoint_bootstrap")
        .map_err(|_| SamplingAssumptionUnionError::Invalid("invalid moving-block bootstrap score identity"))?;
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
        composition: Composition::Chained { stages: request.bootstrap_replicates + 1 },
        decision: None, planning: None, chain: None,
    };
    let physical = PhysicalLayer {
        operators: Arc::from([
            PhysicalOperator::declare(PhysicalEffect::Read)
                .estimates(ByteEstimates::absent().with_output(source_bytes.len() as u64))
                .build().map_err(|_| SamplingAssumptionUnionError::Invalid("invalid moving-block read operator"))?,
            PhysicalOperator::declare(PhysicalEffect::Select)
                .build().map_err(|_| SamplingAssumptionUnionError::Invalid("invalid moving-block resampling operator"))?,
        ]), design,
    };
    let lowered = Arc::new(StandaloneCompositionLoweredParts {
        cell: ProgramCell::MovingBlockSamplingAssumptionUnion,
        scientific, identification, statistical, evidence, physical,
        claim: ClaimDemand { uncertainty: UncertaintyReading::Sampling,
            kind: Some(ClaimKind::Bounds) },
    }.finish());
    Ok(CompiledStratifiedMovingBlockSamplingAssumptionUnion {
        request, strata, original, seed, lowered,
    })
}

fn stratified_moving_block_indices(
    strata: &[Vec<usize>], block_length: usize, rng: &mut antecedent_core::CausalRng,
) -> Vec<usize> {
    let total: usize = strata.iter().map(Vec::len).sum();
    let mut indices = vec![0; total];
    for rows in strata {
        let mut filled = 0;
        while filled < rows.len() {
            let start = uniform_row(rng, rows.len() - block_length + 1);
            let take = block_length.min(rows.len() - filled);
            for offset in 0..take { indices[rows[filled + offset]] = rows[start + offset]; }
            filled += take;
        }
    }
    indices
}

impl CompiledStratifiedMovingBlockSamplingAssumptionUnion {
    /// Complete six-layer moving-block resampling program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram { &self.lowered }

    /// Inspect the fixed schedule before sampling any moving-blocks.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }

    /// Resample consecutive moving blocks, rerun every aligned chain source and form the
    /// unmeasured Bonferroni basic-bootstrap identified-set union.
    ///
    /// # Errors
    /// Changed seed, cancellation, nested chain or nonfinite bound.
    pub fn execute(
        &self, ctx: &ExecutionContext,
    ) -> Result<SamplingAssumptionUnionResult, SamplingAssumptionUnionError> {
        if ctx.rng.master_seed() != self.seed {
            return Err(SamplingAssumptionUnionError::Invalid("moving-block bootstrap execution seed changed"));
        }
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        let original = self.original.execute(ctx)?;
        let (original_lower, original_upper) = endpoint_means(&original)?;
        let mut lows = Vec::with_capacity(self.request.bootstrap_replicates);
        let mut highs = Vec::with_capacity(self.request.bootstrap_replicates);
        let mut chain_identities = vec![original.identity().to_owned()];
        let mut rng = scoped.rng.stream(0x5354_524D_4F56_424C);
        for replicate in 0..self.request.bootstrap_replicates {
            if scoped.cancellation.is_cancelled() {
                return Err(SamplingAssumptionUnionError::Invalid("moving-block bootstrap was cancelled"));
            }
            let indices = stratified_moving_block_indices(&self.strata, self.request.block_length, &mut rng);
            let mut draft = self.request.original.clone();
            for (resampled, source) in draft.sources.iter_mut().zip(&self.request.original.sources) {
                if source.representation == "draws" {
                    resampled.values = indices.iter().map(|&row| source.values[row]).collect();
                    resampled.alignment = Some(format!(
                        "stratified-moving-block-bootstrap:{}:{replicate}", self.lowered.identity().to_hex(),
                    ));
                }
            }
            let chain = compile_propagation_chain(draft, ctx)?.execute(ctx)?;
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
            return Err(SamplingAssumptionUnionError::Invalid("moving-block bootstrap union is nonfinite or reversed"));
        }
        Ok(SamplingAssumptionUnionResult {
            original_lower, original_upper, union_lower, union_upper,
            level: self.request.level, calibration: "not_measured",
            chain_identities, lowered_program_identity: self.lowered.identity(),
        })
    }

    /// Exact moving-block-design and fixed-chain coordinate for a prospective
    /// simultaneous endpoint coverage record.
    ///
    /// # Errors
    /// Complete-chain execution or template encoding refused.
    pub fn calibration_candidate(
        &self, ctx: &ExecutionContext,
    ) -> Result<StratifiedMovingBlockSamplingAssumptionCalibrationCandidate, SamplingAssumptionUnionError> {
        let result = self.execute(ctx)?;
        let mut template = self.request.original.clone();
        for source in &mut template.sources {
            if source.representation == "draws" {
                source.values.clear();
                source.alignment = Some("stratified_moving_block_raw_series".into());
            }
        }
        let template_digest = CanonicalWriter::digest_bytes(
            "antecedent.stratified_moving_block_sampling_assumption_union.template.v1",
            &antecedent_io::to_cbor(&template)?,
        );
        let mut writer = CanonicalWriter::new(
            "antecedent.stratified_moving_block_sampling_assumption_union.calibration_scope.v1",
        );
        writer.bytes(&template_digest).word(self.request.bootstrap_replicates as u64)
            .word(self.request.unit_ids.len() as u64).word(self.request.block_length as u64)
            .word(self.strata.len() as u64);
        for rows in &self.strata {
            writer.word(rows.len() as u64);
            for &row in rows { writer.word(row as u64); }
        }
        let scope = SemanticDigest::from_bytes(writer.finish()).to_hex();
        let functional = format!("joint_set_endpoints:stratified_moving_block_scope={scope}");
        let basis = CalibrationBasis::new(
            [
                "IdentifiedSet", "None", "fixed", "tabular", "Frequentist",
                STRATIFIED_MOVING_BLOCK_FEATURE, "bonferroni_basic_stratified_moving_block_bootstrap", "",
                "independent_series_stationary_weak_dependence_moving_blocks", "", functional.as_str(),
            ].map(Arc::from),
            self.request.level, Arc::from("partial"), self.request.unit_ids.len() as u64,
            Some(self.request.bootstrap_replicates as u32), None, 0.0,
        ).with_validation("none");
        Ok(StratifiedMovingBlockSamplingAssumptionCalibrationCandidate { result, basis })
    }

    /// Seal the raw rows, moving-block schedule and complete result for replay.
    ///
    /// # Errors
    /// A changed run or oversized artifact refuses.
    pub fn export(
        &self, result: &SamplingAssumptionUnionResult, ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, SamplingAssumptionUnionError> {
        if fingerprint(&self.execute(ctx)?) != fingerprint(result) {
            return Err(SamplingAssumptionUnionError::Invalid("moving-block bootstrap result changed"));
        }
        let bytes = antecedent_io::to_cbor(&StratifiedMovingBlockSamplingAssumptionUnionWire {
            version: 1, required_feature: STRATIFIED_MOVING_BLOCK_FEATURE.into(),
            request: self.request.clone(), seed: self.seed,
            program_identity: *self.lowered.identity().as_bytes(),
            result_fingerprint: fingerprint(result),
        })?;
        if bytes.len() > MAX_ARTIFACT_BYTES {
            return Err(SamplingAssumptionUnionError::Invalid("moving-block bootstrap artifact exceeds its byte bound"));
        }
        Ok(bytes)
    }
}

/// Recompile and rerun the complete moving-block schedule under a separately
/// retained program identity.
///
/// # Errors
/// A foreign feature, changed raw data, moving-block mapping, seed or bound.
pub fn consume_stratified_moving_block_sampling_assumption_union(
    bytes: &[u8], expected_program: SemanticDigest, ctx: &ExecutionContext,
) -> Result<SamplingAssumptionUnionResult, SamplingAssumptionUnionError> {
    if bytes.len() > MAX_ARTIFACT_BYTES {
        return Err(SamplingAssumptionUnionError::Invalid("moving-block bootstrap artifact exceeds its byte bound"));
    }
    let wire: StratifiedMovingBlockSamplingAssumptionUnionWire = antecedent_io::from_cbor(bytes)?;
    if wire.version != 1 || wire.required_feature != STRATIFIED_MOVING_BLOCK_FEATURE
        || wire.program_identity != *expected_program.as_bytes()
        || wire.seed != ctx.rng.master_seed()
    {
        return Err(SamplingAssumptionUnionError::Invalid("moving-block bootstrap feature, identity or seed differs"));
    }
    let plan = compile_stratified_moving_block_sampling_assumption_union(wire.request, ctx)?;
    if plan.lowered().identity() != expected_program {
        return Err(SamplingAssumptionUnionError::Invalid("recompiled moving-block bootstrap program differs"));
    }
    let live = plan.execute(ctx)?;
    if fingerprint(&live) != wire.result_fingerprint {
        return Err(SamplingAssumptionUnionError::Invalid("moving-block bootstrap result differs on replay"));
    }
    Ok(live)
}

/// Independent, identically distributed panels with equal-length, equally spaced
/// ordered observations. Each bootstrap replicate resamples panels with replacement,
/// then draws overlapping non-circular blocks within each selected panel. All
/// sampling-source columns use the same indices, and the last block is truncated.
/// Independence and exchangeability of panels, and stationarity and sufficiently
/// weak dependence within each panel, are declared assumptions, not tested.
/// The two-stage bootstrap is a candidate only under an appropriate panel/time
/// asymptotic regime; its extra within-panel resampling can overstate variance
/// when the panel bootstrap already captures that variability.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PanelBlockSamplingAssumptionUnionRequest {
    /// Complete chain over raw unit rows and fixed assumptions.
    pub original: PropagationChainRecordWire,
    /// Distinct row identities in chronological, equally spaced sampling-source order.
    pub unit_ids: Vec<String>,
    /// Panel identity per row, in chronological order within each panel.
    pub panel_ids: Vec<String>,
    /// Prespecified common number of consecutive rows per within-series block.
    pub block_length: usize,
    /// Number of whole-chain moving-block bootstrap replicates.
    pub bootstrap_replicates: usize,
    /// Nominal two-sided level; coverage remains unmeasured.
    pub level: f64,
}

/// Retained moving-block resampling and complete-chain endpoint union.
#[derive(Clone, Debug)]
pub struct CompiledPanelBlockSamplingAssumptionUnion {
    request: PanelBlockSamplingAssumptionUnionRequest,
    panels: Vec<Vec<usize>>,
    original: PropagationChainPlan,
    seed: u64,
    lowered: Arc<LoweredProgram>,
}

/// Candidate for simultaneous identified-set endpoint coverage by moving-block.
#[derive(Clone, Debug)]
pub struct PanelBlockSamplingAssumptionCalibrationCandidate {
    /// Complete compiler-generated moving-block bootstrap result.
    pub result: SamplingAssumptionUnionResult,
    /// Exact fixed chain template and moving-block-design calibration coordinate.
    pub basis: CalibrationBasis,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PanelBlockSamplingAssumptionUnionWire {
    version: u32,
    required_feature: String,
    request: PanelBlockSamplingAssumptionUnionRequest,
    seed: u64,
    program_identity: [u8; 32],
    result_fingerprint: [u8; 32],
}

/// Compile two-stage panel and within-panel block resampling before drawing rows.
/// Every generated chain is recompiled at execution, and the portable
/// consumer reconstructs the same moving-block schedule from the raw declaration.
///
/// # Errors
/// Malformed unit or moving-block identities, invalid fixed assumptions, unsupported
/// chain, level or resource bound.
pub fn compile_panel_block_sampling_assumption_union(
    request: PanelBlockSamplingAssumptionUnionRequest, ctx: &ExecutionContext,
) -> Result<CompiledPanelBlockSamplingAssumptionUnion, SamplingAssumptionUnionError> {
    validate_raw_unit_source(&request.original, &request.unit_ids,
        request.bootstrap_replicates, request.level)?;
    let n = request.unit_ids.len();
    if request.panel_ids.len() != n || request.panel_ids.iter().any(|id| id.trim().is_empty()) {
        return Err(SamplingAssumptionUnionError::Invalid("panels must align with all sampling rows"));
    }
    let mut grouped: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (row, panel) in request.panel_ids.iter().enumerate() {
        grouped.entry(panel.as_str()).or_default().push(row);
    }
    let panels: Vec<Vec<usize>> = grouped.into_values().collect();
    if panels.len() < 10 || request.block_length < 2
        || panels.iter().any(|rows| rows.len() < 20 || rows.len() != panels[0].len()
            || request.block_length > rows.len() / 2)
    {
        return Err(SamplingAssumptionUnionError::Invalid("panel bootstrap needs ten equal-length panels of at least 20 rows and block length between 2 and half the panel length"));
    }
    let source_bytes = antecedent_io::to_cbor(&request)?;
    if source_bytes.len() > MAX_ARTIFACT_BYTES {
        return Err(SamplingAssumptionUnionError::Invalid("moving-block bootstrap source exceeds its byte bound"));
    }
    let original = compile_propagation_chain(request.original.clone(), ctx)?;
    let source_digest = CanonicalWriter::digest_bytes(
        "antecedent.panel_block_sampling_assumption_union.source.v1", &source_bytes,
    );
    let seed = ctx.rng.master_seed();
    let mut writer = CanonicalWriter::new("antecedent.panel_block_sampling_assumption_union.question.v1");
    writer.bytes(&source_digest).word(seed);
    let query = writer.finish();
    let design = DesignShape::quoted(request.unit_ids.len() as u64,
        request.original.sources.len() as u64, true);
    let scientific = ScientificProblem {
        runtime_query: None, identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::IdentifiedSet, outcome: None, query,
        },
        plan_id: Arc::from("panel_block_sampling_assumption_union.v1"),
        data: DataClassification::Tabular,
        graph_class: "compiler_resampled_panel_block_assumption_set",
        query_variables: Arc::from([]), source_digest: Some(source_digest),
        support_coordinate: None, supported: false, support_status: None,
    };
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SharedEvidenceContract { contract: query },
    ));
    let score = ScoreModel::new("panel_block_whole_chain_assumption_endpoint_bootstrap")
        .map_err(|_| SamplingAssumptionUnionError::Invalid("invalid moving-block bootstrap score identity"))?;
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
        composition: Composition::Chained { stages: request.bootstrap_replicates + 1 },
        decision: None, planning: None, chain: None,
    };
    let physical = PhysicalLayer {
        operators: Arc::from([
            PhysicalOperator::declare(PhysicalEffect::Read)
                .estimates(ByteEstimates::absent().with_output(source_bytes.len() as u64))
                .build().map_err(|_| SamplingAssumptionUnionError::Invalid("invalid moving-block read operator"))?,
            PhysicalOperator::declare(PhysicalEffect::Select)
                .build().map_err(|_| SamplingAssumptionUnionError::Invalid("invalid moving-block resampling operator"))?,
        ]), design,
    };
    let lowered = Arc::new(StandaloneCompositionLoweredParts {
        cell: ProgramCell::MovingBlockSamplingAssumptionUnion,
        scientific, identification, statistical, evidence, physical,
        claim: ClaimDemand { uncertainty: UncertaintyReading::Sampling,
            kind: Some(ClaimKind::Bounds) },
    }.finish());
    Ok(CompiledPanelBlockSamplingAssumptionUnion {
        request, panels, original, seed, lowered,
    })
}

fn panel_block_indices(
    panels: &[Vec<usize>], block_length: usize, rng: &mut antecedent_core::CausalRng,
) -> Vec<usize> {
    let total: usize = panels.iter().map(Vec::len).sum();
    let mut indices = vec![0; total];
    for destination in panels {
        let source = &panels[uniform_row(rng, panels.len())];
        let mut filled = 0;
        while filled < destination.len() {
            let start = uniform_row(rng, source.len() - block_length + 1);
            let take = block_length.min(destination.len() - filled);
            for offset in 0..take { indices[destination[filled + offset]] = source[start + offset]; }
            filled += take;
        }
    }
    indices
}

impl CompiledPanelBlockSamplingAssumptionUnion {
    /// Complete six-layer moving-block resampling program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram { &self.lowered }

    /// Inspect the fixed schedule before sampling any moving-blocks.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }

    /// Resample panels and then their consecutive blocks, rerun the aligned chain and form the
    /// unmeasured Bonferroni basic-bootstrap identified-set union.
    ///
    /// # Errors
    /// Changed seed, cancellation, nested chain or nonfinite bound.
    pub fn execute(
        &self, ctx: &ExecutionContext,
    ) -> Result<SamplingAssumptionUnionResult, SamplingAssumptionUnionError> {
        if ctx.rng.master_seed() != self.seed {
            return Err(SamplingAssumptionUnionError::Invalid("moving-block bootstrap execution seed changed"));
        }
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        let original = self.original.execute(ctx)?;
        let (original_lower, original_upper) = endpoint_means(&original)?;
        let mut lows = Vec::with_capacity(self.request.bootstrap_replicates);
        let mut highs = Vec::with_capacity(self.request.bootstrap_replicates);
        let mut chain_identities = vec![original.identity().to_owned()];
        let mut rng = scoped.rng.stream(0x5041_4E45_4C42_4C4B);
        for replicate in 0..self.request.bootstrap_replicates {
            if scoped.cancellation.is_cancelled() {
                return Err(SamplingAssumptionUnionError::Invalid("moving-block bootstrap was cancelled"));
            }
            let indices = panel_block_indices(&self.panels, self.request.block_length, &mut rng);
            let mut draft = self.request.original.clone();
            for (resampled, source) in draft.sources.iter_mut().zip(&self.request.original.sources) {
                if source.representation == "draws" {
                    resampled.values = indices.iter().map(|&row| source.values[row]).collect();
                    resampled.alignment = Some(format!(
                        "panel-block-bootstrap:{}:{replicate}", self.lowered.identity().to_hex(),
                    ));
                }
            }
            let chain = compile_propagation_chain(draft, ctx)?.execute(ctx)?;
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
            return Err(SamplingAssumptionUnionError::Invalid("moving-block bootstrap union is nonfinite or reversed"));
        }
        Ok(SamplingAssumptionUnionResult {
            original_lower, original_upper, union_lower, union_upper,
            level: self.request.level, calibration: "not_measured",
            chain_identities, lowered_program_identity: self.lowered.identity(),
        })
    }

    /// Exact moving-block-design and fixed-chain coordinate for a prospective
    /// simultaneous endpoint coverage record.
    ///
    /// # Errors
    /// Complete-chain execution or template encoding refused.
    pub fn calibration_candidate(
        &self, ctx: &ExecutionContext,
    ) -> Result<PanelBlockSamplingAssumptionCalibrationCandidate, SamplingAssumptionUnionError> {
        let result = self.execute(ctx)?;
        let mut template = self.request.original.clone();
        for source in &mut template.sources {
            if source.representation == "draws" {
                source.values.clear();
                source.alignment = Some("panel_block_raw_units".into());
            }
        }
        let template_digest = CanonicalWriter::digest_bytes(
            "antecedent.panel_block_sampling_assumption_union.template.v1",
            &antecedent_io::to_cbor(&template)?,
        );
        let mut writer = CanonicalWriter::new(
            "antecedent.panel_block_sampling_assumption_union.calibration_scope.v1",
        );
        writer.bytes(&template_digest).word(self.request.bootstrap_replicates as u64)
            .word(self.request.unit_ids.len() as u64).word(self.request.block_length as u64)
            .word(self.panels.len() as u64);
        for rows in &self.panels {
            writer.word(rows.len() as u64);
            for &row in rows { writer.word(row as u64); }
        }
        let scope = SemanticDigest::from_bytes(writer.finish()).to_hex();
        let functional = format!("joint_set_endpoints:panel_block_scope={scope}");
        let basis = CalibrationBasis::new(
            [
                "IdentifiedSet", "None", "fixed", "tabular", "Frequentist",
                PANEL_BLOCK_FEATURE, "bonferroni_basic_panel_block_bootstrap", "",
                "iid_panels_stationary_weak_dependence_two_stage_blocks", "", functional.as_str(),
            ].map(Arc::from),
            self.request.level, Arc::from("partial"), self.request.unit_ids.len() as u64,
            Some(self.request.bootstrap_replicates as u32), None, 0.0,
        ).with_validation("none");
        Ok(PanelBlockSamplingAssumptionCalibrationCandidate { result, basis })
    }

    /// Seal the raw rows, moving-block schedule and complete result for replay.
    ///
    /// # Errors
    /// A changed run or oversized artifact refuses.
    pub fn export(
        &self, result: &SamplingAssumptionUnionResult, ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, SamplingAssumptionUnionError> {
        if fingerprint(&self.execute(ctx)?) != fingerprint(result) {
            return Err(SamplingAssumptionUnionError::Invalid("moving-block bootstrap result changed"));
        }
        let bytes = antecedent_io::to_cbor(&PanelBlockSamplingAssumptionUnionWire {
            version: 1, required_feature: PANEL_BLOCK_FEATURE.into(),
            request: self.request.clone(), seed: self.seed,
            program_identity: *self.lowered.identity().as_bytes(),
            result_fingerprint: fingerprint(result),
        })?;
        if bytes.len() > MAX_ARTIFACT_BYTES {
            return Err(SamplingAssumptionUnionError::Invalid("moving-block bootstrap artifact exceeds its byte bound"));
        }
        Ok(bytes)
    }
}

/// Recompile and rerun the complete moving-block schedule under a separately
/// retained program identity.
///
/// # Errors
/// A foreign feature, changed raw data, moving-block mapping, seed or bound.
pub fn consume_panel_block_sampling_assumption_union(
    bytes: &[u8], expected_program: SemanticDigest, ctx: &ExecutionContext,
) -> Result<SamplingAssumptionUnionResult, SamplingAssumptionUnionError> {
    if bytes.len() > MAX_ARTIFACT_BYTES {
        return Err(SamplingAssumptionUnionError::Invalid("moving-block bootstrap artifact exceeds its byte bound"));
    }
    let wire: PanelBlockSamplingAssumptionUnionWire = antecedent_io::from_cbor(bytes)?;
    if wire.version != 1 || wire.required_feature != PANEL_BLOCK_FEATURE
        || wire.program_identity != *expected_program.as_bytes()
        || wire.seed != ctx.rng.master_seed()
    {
        return Err(SamplingAssumptionUnionError::Invalid("moving-block bootstrap feature, identity or seed differs"));
    }
    let plan = compile_panel_block_sampling_assumption_union(wire.request, ctx)?;
    if plan.lowered().identity() != expected_program {
        return Err(SamplingAssumptionUnionError::Invalid("recompiled moving-block bootstrap program differs"));
    }
    let live = plan.execute(ctx)?;
    if fingerprint(&live) != wire.result_fingerprint {
        return Err(SamplingAssumptionUnionError::Invalid("moving-block bootstrap result differs on replay"));
    }
    Ok(live)
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
    #[ignore = "source-only compiler-owned IID bootstrap assertion; coverage has not been earned"]
    fn iid_raw_units_are_resampled_and_whole_chain_replayed() {
        let ctx = ExecutionContext::for_tests(31);
        let mut original = draft(0.0);
        original.sources[0].values = (0..40)
            .map(|row| (1.0 + row as f64 / 40.0).to_bits()).collect();
        let request = IidSamplingAssumptionUnionRequest {
            original,
            unit_ids: (0..40).map(|row| format!("unit-{row}")).collect(),
            bootstrap_replicates: 100,
            level: 0.95,
        };
        let plan = compile_iid_sampling_assumption_union(request, &ctx).unwrap();
        assert_eq!(plan.plan_view().route().as_str(), "IidSamplingAssumptionUnion");
        let result = plan.execute(&ctx).unwrap();
        assert!(result.union_lower <= result.original_lower);
        assert!(result.union_upper >= result.original_upper);
        assert_eq!(result.chain_identities.len(), 101);
        assert_eq!(result.calibration, "not_measured");
        let bytes = plan.export(&result, &ctx).unwrap();
        let replay = consume_iid_sampling_assumption_union(
            &bytes, result.lowered_program_identity, &ctx,
        ).unwrap();
        assert_eq!(fingerprint(&replay), fingerprint(&result));
    }

    #[test]
    #[ignore = "source-only compiler-owned cluster bootstrap; coverage has not been earned"]
    fn dependent_unit_pairs_are_resampled_as_complete_clusters() {
        let ctx = ExecutionContext::for_tests(37);
        let mut original = draft(0.0);
        original.sources[0].values = (0..40)
            .map(|row| (1.0 + (row / 2) as f64 / 20.0).to_bits()).collect();
        let request = ClusterSamplingAssumptionUnionRequest {
            original,
            unit_ids: (0..40).map(|row| format!("unit-{row}")).collect(),
            cluster_ids: (0..40).map(|row| format!("cluster-{}", row / 2)).collect(),
            bootstrap_replicates: 100, level: 0.95,
        };
        let plan = compile_cluster_sampling_assumption_union(request, &ctx).unwrap();
        assert_eq!(plan.plan_view().route().as_str(), "ClusterSamplingAssumptionUnion");
        let candidate = plan.calibration_candidate(&ctx).unwrap();
        assert_eq!(candidate.result.calibration, "not_measured");
        assert_eq!(candidate.result.chain_identities.len(), 101);
        assert_eq!(candidate.basis.interval_method.as_ref(),
            "bonferroni_basic_cluster_bootstrap");
        let bytes = plan.export(&candidate.result, &ctx).unwrap();
        let replay = consume_cluster_sampling_assumption_union(
            &bytes, candidate.result.lowered_program_identity, &ctx,
        ).unwrap();
        assert_eq!(fingerprint(&replay), fingerprint(&candidate.result));
    }

    #[test]
    #[ignore = "source-only stratified whole-cluster bootstrap; coverage has not been earned"]
    fn stratified_clusters_are_kept_complete_and_replayed() {
        let ctx = ExecutionContext::for_tests(43);
        let mut original = draft(0.0);
        original.sources[0].values = (0..40).map(|row| (row as f64 / 40.0).to_bits()).collect();
        let request = StratifiedClusterSamplingAssumptionUnionRequest {
            original, unit_ids: (0..40).map(|i| format!("unit-{i}")).collect(),
            cluster_ids: (0..40).map(|i| format!("cluster-{}", i / 2)).collect(),
            stratum_ids: (0..40).map(|i| format!("stratum-{}", i / 20)).collect(),
            bootstrap_replicates: 100, level: 0.95,
        };
        let plan = compile_stratified_cluster_sampling_assumption_union(request.clone(), &ctx).unwrap();
        let mut rng = antecedent_core::CausalRng::from_seed(43);
        let indices = stratified_cluster_indices(&plan.strata, &mut rng);
        assert_eq!(indices.len(), 40);
        assert!(indices.chunks_exact(2).all(|pair| pair[1] == pair[0] + 1 && pair[0] % 2 == 0));
        assert!(indices[..20].iter().all(|&i| i < 20));
        assert!(indices[20..].iter().all(|&i| i >= 20));
        let candidate = plan.calibration_candidate(&ctx).unwrap();
        assert_eq!(candidate.result.calibration, "not_measured");
        assert_eq!(candidate.result.chain_identities.len(), 101);
        assert_eq!(candidate.basis.interval_method.as_ref(), "bonferroni_basic_stratified_cluster_bootstrap");
        let artifact = plan.export(&candidate.result, &ctx).unwrap();
        let replay = consume_stratified_cluster_sampling_assumption_union(
            &artifact, plan.lowered().identity(), &ctx,
        ).unwrap();
        assert_eq!(fingerprint(&candidate.result), fingerprint(&replay));
        let mut changed = request.clone();
        changed.stratum_ids[0] = "wrong".into();
        assert!(compile_stratified_cluster_sampling_assumption_union(changed, &ctx).is_err());
        let mut changed = request;
        changed.stratum_ids.reverse();
        let other = compile_stratified_cluster_sampling_assumption_union(changed, &ctx).unwrap();
        assert_ne!(plan.lowered().identity(), other.lowered().identity());
        assert!(consume_stratified_cluster_sampling_assumption_union(&artifact, other.lowered().identity(), &ctx).is_err());
    }

    #[test]
    #[ignore = "source-only two-stage panel block bootstrap; coverage has not been earned"]
    fn resampled_panels_keep_one_source_panel_and_consecutive_blocks() {
        let ctx = ExecutionContext::for_tests(53);
        let mut original = draft(0.0);
        original.sources[0].values = (0..200).map(|i| (i as f64 / 200.0).to_bits()).collect();
        let request = PanelBlockSamplingAssumptionUnionRequest {
            original, unit_ids: (0..200).map(|i| format!("row-{i}")).collect(),
            panel_ids: (0..200).map(|i| format!("panel-{:02}", i % 10)).collect(),
            block_length: 4, bootstrap_replicates: 100, level: 0.95,
        };
        let plan = compile_panel_block_sampling_assumption_union(request.clone(), &ctx).unwrap();
        let mut rng = antecedent_core::CausalRng::from_seed(53);
        let indices = panel_block_indices(&plan.panels, 4, &mut rng);
        for destination in &plan.panels {
            let selected: Vec<_> = destination.iter().map(|&row| indices[row]).collect();
            assert!(selected.iter().all(|row| row % 10 == selected[0] % 10));
            assert!(selected.chunks(4).all(|block| block.windows(2).all(|pair| pair[1] == pair[0] + 10)));
        }
        let candidate = plan.calibration_candidate(&ctx).unwrap();
        assert_eq!(candidate.result.calibration, "not_measured");
        assert_eq!(candidate.result.chain_identities.len(), 101);
        assert_eq!(candidate.basis.interval_method.as_ref(), "bonferroni_basic_panel_block_bootstrap");
        let artifact = plan.export(&candidate.result, &ctx).unwrap();
        let replay = consume_panel_block_sampling_assumption_union(&artifact, plan.lowered().identity(), &ctx).unwrap();
        assert_eq!(fingerprint(&candidate.result), fingerprint(&replay));
        let mut changed = request.clone(); changed.block_length = 5;
        let other = compile_panel_block_sampling_assumption_union(changed, &ctx).unwrap();
        assert_ne!(plan.lowered().identity(), other.lowered().identity());
        assert!(consume_panel_block_sampling_assumption_union(&artifact, other.lowered().identity(), &ctx).is_err());
        let mut invalid = request; invalid.panel_ids[0] = "extra-panel".into();
        assert!(compile_panel_block_sampling_assumption_union(invalid, &ctx).is_err());
    }

    #[test]
    #[ignore = "source-only stratified moving-block bootstrap; coverage has not been earned"]
    fn stratified_moving_blocks_stay_within_interleaved_series_and_replay() {
        let ctx = ExecutionContext::for_tests(47);
        let mut original = draft(0.0);
        original.sources[0].values = (0..40).map(|i| (i as f64 / 40.0).to_bits()).collect();
        let request = StratifiedMovingBlockSamplingAssumptionUnionRequest {
            original, unit_ids: (0..40).map(|i| format!("row-{i}")).collect(),
            stratum_ids: (0..40).map(|i| format!("series-{}", i % 2)).collect(),
            block_length: 4, bootstrap_replicates: 100, level: 0.95,
        };
        let plan = compile_stratified_moving_block_sampling_assumption_union(request.clone(), &ctx).unwrap();
        let mut rng = antecedent_core::CausalRng::from_seed(47);
        let indices = stratified_moving_block_indices(&plan.strata, 4, &mut rng);
        for rows in &plan.strata {
            for chunk in rows.chunks(4) {
                let selected: Vec<_> = chunk.iter().map(|&row| indices[row]).collect();
                assert!(selected.iter().all(|&row| row % 2 == rows[0] % 2));
                assert!(selected.windows(2).all(|pair| pair[1] == pair[0] + 2));
            }
        }
        let candidate = plan.calibration_candidate(&ctx).unwrap();
        assert_eq!(candidate.result.calibration, "not_measured");
        assert_eq!(candidate.result.chain_identities.len(), 101);
        assert_eq!(candidate.basis.interval_method.as_ref(), "bonferroni_basic_stratified_moving_block_bootstrap");
        let artifact = plan.export(&candidate.result, &ctx).unwrap();
        let replay = consume_stratified_moving_block_sampling_assumption_union(
            &artifact, plan.lowered().identity(), &ctx,
        ).unwrap();
        assert_eq!(fingerprint(&candidate.result), fingerprint(&replay));
        let mut changed = request.clone();
        changed.block_length = 5;
        let other = compile_stratified_moving_block_sampling_assumption_union(changed, &ctx).unwrap();
        assert_ne!(plan.lowered().identity(), other.lowered().identity());
        assert!(consume_stratified_moving_block_sampling_assumption_union(&artifact, other.lowered().identity(), &ctx).is_err());
        let mut invalid = request;
        invalid.stratum_ids[0].clear();
        assert!(compile_stratified_moving_block_sampling_assumption_union(invalid, &ctx).is_err());
    }

    #[test]
    #[ignore = "source-only compiler-owned moving-block bootstrap; coverage has not been earned"]
    fn moving_blocks_replay_and_bind_the_declared_design() {
        let ctx = ExecutionContext::for_tests(41);
        let mut original = draft(0.0);
        original.sources[0].values = (0..40)
            .map(|row| (1.0 + row as f64 / 40.0).to_bits()).collect();
        let request = MovingBlockSamplingAssumptionUnionRequest {
            original,
            unit_ids: (0..40).map(|row| format!("time-{row}")).collect(),
            block_length: 5, bootstrap_replicates: 100, level: 0.95,
        };
        let plan = compile_moving_block_sampling_assumption_union(request.clone(), &ctx).unwrap();
        assert_eq!(plan.plan_view().route().as_str(), "MovingBlockSamplingAssumptionUnion");
        let candidate = plan.calibration_candidate(&ctx).unwrap();
        assert_eq!(candidate.result.calibration, "not_measured");
        assert_eq!(candidate.result.chain_identities.len(), 101);
        assert_eq!(candidate.basis.interval_method.as_ref(),
            "bonferroni_basic_moving_block_bootstrap");
        let bytes = plan.export(&candidate.result, &ctx).unwrap();
        let replay = consume_moving_block_sampling_assumption_union(
            &bytes, candidate.result.lowered_program_identity, &ctx,
        ).unwrap();
        assert_eq!(fingerprint(&replay), fingerprint(&candidate.result));
        let mut changed = request.clone();
        changed.block_length = 6;
        let other = compile_moving_block_sampling_assumption_union(changed, &ctx).unwrap();
        assert_ne!(other.lowered().identity(), plan.lowered().identity());
        assert!(consume_moving_block_sampling_assumption_union(
            &bytes, other.lowered().identity(), &ctx,
        ).is_err());
        let mut invalid = request;
        invalid.block_length = 1;
        assert!(compile_moving_block_sampling_assumption_union(invalid, &ctx).is_err());
    }

    #[test]
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
