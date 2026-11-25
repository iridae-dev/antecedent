//! Canonical bytes of the digests a lowered program already stores.
//!
//! [`encode_invariant`](super::super::encode_invariant) is untouched and remains the only
//! `CanonicalWriter` on `antecedent.lowered_program`. Each child buffer is the framed
//! preimage of a digest that function already writes, or the CBOR of an identification
//! product whose existing digest is `identification_product_digest_wire`. A decoder reads
//! the buffer and [`check_children`] compares it to the live program.
//!
//! Runtime callbacks, bare arena ids, `Arc` graphs and `CausalState` are not bytes here.
//! `PerExecution` and `Deferred` are recorded as those variants; [`resume_identification`]
//! identifies them again.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    GENERALIZED_BAYES_TAG, ORDINARY_BAYESIAN_LIKELIHOOD_TAG, PARTIAL_IDENTIFICATION_ENVELOPE_TAG,
    SCORE_FREQUENTIST_TAG, SemanticDigest,
};
use antecedent_identify::IdentificationResult;
use antecedent_io_core::identity::{
    IdentificationProductWire, decode_identification_product, identification_product_canonical_bytes,
    identification_product_wire,
};

use super::super::planning::canon::{CanonBuf, CanonPut, framed_digest};
use super::super::{
    EvidenceDecisionProgram, IdentificationLayer, LoweredProgram, NonGraphicalIdentification,
    ProblemTarget, ProgramIdentityScope, RetainedIdentification,
};
use super::{
    Composition, CostMapping, DecisionProgram, EvsiProgram, PreferenceDesignLawFact,
    PreferenceJointLawFact, UtilityForm,
};
use crate::analysis::IdentificationCache;
use crate::planner::statistical::{
    FoldPlan, NestedFitLaw, PosteriorPlan, Realization, ResamplingPlan, StatisticalProgram,
};

const STATISTICAL_V2: &str = "antecedent.lowered_program.statistical.v2";
const IDENTIFICATION_CHILD: &str = "antecedent.lowered_program.identification_child.v1";
const JOINT_SCENARIOS: &str = "joint_scenarios_v1";

/// Why a child buffer was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ChildCodecError {
    detail: &'static str,
}

impl ChildCodecError {
    const fn new(detail: &'static str) -> Self {
        Self { detail }
    }
}

impl std::fmt::Display for ChildCodecError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.detail)
    }
}

fn err(detail: &'static str) -> ChildCodecError {
    ChildCodecError::new(detail)
}

/// A nested digest whose preimage is not in this codec.
///
/// `owner` is the function that already hashes that digest. The bytes stay with that
/// function; [`check_children`] only checks that the live program still carries `digest`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Referent {
    /// Role inside the lowered program.
    pub role: &'static str,
    /// The digest the program already stores.
    pub digest: [u8; 32],
    /// Function that owns the preimage.
    pub owner: &'static str,
}

/// What the child bytes leave in the process.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ProcessLocal {
    /// The program binds a runtime callback. The callback is not in the bytes.
    pub runtime_callback: bool,
    /// A retained cache or execution proof holds an `Arc` graph. The graph is not in the bytes.
    pub arc_graph: bool,
    /// A compiled product or a deferred failure holds an expression arena.
    /// Bare arena ids are not a child payload.
    pub arena_in_process: bool,
    /// `CausalState` is never a child payload.
    pub causal_state_omitted: bool,
}

/// Child bytes of one lowered program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Children {
    /// Frames of [`StatisticalProgram::lowered_layer_digest`].
    pub statistical: Vec<u8>,
    /// Frames of [`EvidenceDecisionProgram::identity`].
    pub evidence: Vec<u8>,
    /// Explicit identification record, including per-execution and deferred.
    pub identification: Vec<u8>,
    /// Frames of a checked realization identity, when the statistical layer embeds one.
    pub realization: Option<Vec<u8>>,
    /// Nested digests whose preimage is owned elsewhere.
    pub referents: Vec<Referent>,
    /// Process-local state the bytes do not carry.
    pub process_local: ProcessLocal,
}

/// How resume treats layer two.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum IdentificationResume {
    /// Identify again. The compile-time failure, when there was one, is not in the bytes.
    Reidentify {
        /// Which explicit variant was stored.
        kind: ReidentifyKind,
        /// The reason compile recorded.
        reason: String,
    },
    /// The retained record decoded. Preparation installs it and does not identify again.
    Retained,
    /// A non-graphical record decoded.
    NonGraphical,
    /// The program claims no identification.
    NotClaimed {
        /// Refusal text, when compile stored one.
        refusal: Option<String>,
    },
}

/// A layer two that resume identifies again.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReidentifyKind {
    /// [`IdentificationLayer::PerExecution`].
    PerExecution,
    /// [`IdentificationLayer::Deferred`].
    Deferred,
}

/// Encode every child of `program`.
///
/// # Errors
///
/// An identification-product wire cannot be encoded.
pub(crate) fn resolve_children(program: &LoweredProgram) -> Result<Children, ChildCodecError> {
    Ok(Children {
        statistical: statistical_bytes(program.statistical()),
        evidence: evidence_bytes(program.evidence()),
        identification: identification_bytes(program.identification_layer())?,
        realization: realization_preimage(program.statistical().realization()),
        referents: referents(program),
        process_local: process_local(program),
    })
}

/// Decode `children` and compare each layer to `program`.
///
/// # Errors
///
/// A buffer is not the preimage of the live digest, or a decoded field differs.
pub(crate) fn check_children(
    program: &LoweredProgram,
    children: &Children,
) -> Result<(), ChildCodecError> {
    if children.statistical != statistical_bytes(program.statistical()) {
        return Err(err("statistical child bytes are not the live layer"));
    }
    if framed_digest(&children.statistical) != *program.statistical().lowered_layer_digest().as_bytes()
    {
        return Err(err("statistical bytes do not hash to lowered_layer_digest"));
    }
    check_statistical(program.statistical(), &children.statistical)?;
    if children.evidence != evidence_bytes(program.evidence()) {
        return Err(err("evidence child bytes are not the live layer"));
    }
    if framed_digest(&children.evidence) != *program.evidence().identity().as_bytes() {
        return Err(err("evidence bytes do not hash to evidence.identity"));
    }
    check_evidence(program.evidence(), &children.evidence)?;
    if children.identification != identification_bytes(program.identification_layer())? {
        return Err(err("identification child bytes are not the live layer"));
    }
    let resume = resume_identification(&children.identification)?;
    check_resume(program.identification_layer(), &resume)?;
    check_identification(program.identification_layer(), &children.identification)?;
    check_realization(program.statistical().realization(), &children.realization)?;
    if children.referents != referents(program) {
        return Err(err("a nested digest does not match the live program"));
    }
    if children.process_local != process_local(program) || !children.process_local.causal_state_omitted
    {
        return Err(err("process-local flags do not match the live program"));
    }
    Ok(())
}

/// Read the identification child and decide whether resume identifies again.
///
/// # Errors
///
/// The buffer is not an identification child.
pub(crate) fn resume_identification(bytes: &[u8]) -> Result<IdentificationResume, ChildCodecError> {
    let mut frames = Frames::new(bytes);
    frames.expect_tag(IDENTIFICATION_CHILD)?;
    match frames.text()? {
        "deferred" => Ok(IdentificationResume::Reidentify {
            kind: ReidentifyKind::Deferred,
            reason: frames.text()?.to_owned(),
        }),
        "per_execution" => Ok(IdentificationResume::Reidentify {
            kind: ReidentifyKind::PerExecution,
            reason: frames.text()?.to_owned(),
        }),
        "retained" => {
            skip_retained(&mut frames)?;
            frames.expect_end()?;
            Ok(IdentificationResume::Retained)
        }
        "non_graphical" => {
            skip_non_graphical(&mut frames)?;
            frames.expect_end()?;
            Ok(IdentificationResume::NonGraphical)
        }
        "not_claimed" => {
            let refusal = frames.opt_text()?;
            frames.expect_end()?;
            Ok(IdentificationResume::NotClaimed { refusal })
        }
        _ => Err(err("unknown identification child")),
    }
}

fn statistical_bytes(program: &StatisticalProgram) -> Vec<u8> {
    let mut buf = CanonBuf::new(STATISTICAL_V2);
    write_opt_text(&mut buf, program.identifier().map(|id| id.as_str()));
    write_opt_text(&mut buf, program.estimator().map(|id| id.as_str()));
    write_realization(&mut buf, program.realization());
    buf.word(program.models().len() as u64);
    for model in program.models() {
        buf.text(model.role().as_str()).text(model.family()).word(model.inputs().len() as u64);
        for input in model.inputs() {
            buf.word(u64::from(input.raw()));
        }
        write_opt_word(&mut buf, model.output().map(|output| u64::from(output.raw())));
    }
    match program.folds() {
        Some(folds) => {
            buf.word(1).word(folds.folds() as u64).text(folds.unit().as_str());
            write_opt_word(&mut buf, folds.embargo().map(|gap| gap as u64));
        }
        None => {
            buf.word(0);
        }
    }
    match program.resampling() {
        Some(plan) => {
            buf.word(1)
                .text(plan.method().as_str())
                .word(u64::from(plan.replicates()))
                .text(plan.unit().as_str());
        }
        None => {
            buf.word(0);
        }
    }
    match program.posterior() {
        Some(plan) => {
            buf.word(1)
                .text(plan.algorithm().as_str())
                .word(plan.draws() as u64)
                .text(plan.prior().name());
        }
        None => {
            buf.word(0);
        }
    }
    write_opt_word(&mut buf, program.design().columns());
    buf.text(program.design().source().as_str()).word(program.operators().len() as u64);
    buf.into_vec()
}

fn write_opt_text<S: CanonPut>(sink: &mut S, value: Option<&str>) {
    sink.opt_text(value);
}

fn write_opt_word<S: CanonPut>(sink: &mut S, value: Option<u64>) {
    match value {
        Some(word) => {
            sink.word(1).word(word);
        }
        None => {
            sink.word(0);
        }
    }
}

fn write_realization<S: CanonPut>(sink: &mut S, realization: &Realization) {
    match realization {
        Realization::Frequentist(realization) => {
            sink.text("frequentist").bytes(realization.identity().as_bytes());
        }
        Realization::Bayesian(realization) => {
            sink.text("bayesian").bytes(realization.identity().as_bytes());
        }
        Realization::GeneralizedBayes(realization) => {
            sink.text("generalized_bayes").bytes(realization.identity().as_bytes());
        }
        Realization::Envelope(realization) => {
            sink.text("envelope").bytes(realization.identity().as_bytes());
        }
        Realization::GraphImputationV1 => {
            sink.text("graph_imputation_v1");
        }
        Realization::CategoricalPosteriorCandidateV1 => {
            sink.text("categorical_posterior_candidate_v1");
        }
        Realization::SequentialKnownLawMonteCarlo => {
            sink.text("sequential_known_law_monte_carlo_v1");
        }
        Realization::SequentialTrajectoryBootstrap => {
            sink.text("sequential_trajectory_bootstrap_v1");
        }
        Realization::ExactPoint { evaluator } => {
            sink.text("exact_point").text(evaluator);
        }
        Realization::DeclaredPosterior { evaluator } => {
            sink.text("declared_posterior").text(evaluator);
        }
        Realization::DeclaredPredictive { evaluator } => {
            sink.text("declared_predictive").text(evaluator);
        }
        Realization::NestedFit { producer_program, method, law } => {
            sink.text("nested_fit").bytes(producer_program.as_bytes()).text(method).text(match law {
                NestedFitLaw::Posterior => "posterior",
                NestedFitLaw::Predictive => "predictive",
            });
        }
        Realization::DeclaredAssumptionEnvelope { evaluator, draw_meaning } => {
            sink.text("declared_assumption_envelope").text(evaluator).word(draw_meaning.tag());
        }
    }
}

fn realization_preimage(realization: &Realization) -> Option<Vec<u8>> {
    let buf = match realization {
        Realization::Frequentist(inner) => {
            let mut buf = CanonBuf::new(SCORE_FREQUENTIST_TAG);
            buf.text(inner.score().name());
            buf
        }
        Realization::Bayesian(inner) => {
            let mut buf = CanonBuf::new(ORDINARY_BAYESIAN_LIKELIHOOD_TAG);
            buf.text(inner.likelihood().name()).text(inner.prior().name());
            buf
        }
        Realization::GeneralizedBayes(inner) => {
            let mut buf = CanonBuf::new(GENERALIZED_BAYES_TAG);
            buf.text(inner.loss().name())
                .text(inner.prior().name())
                .word(inner.learning_rate().to_bits());
            buf
        }
        Realization::Envelope(inner) => {
            let mut buf = CanonBuf::new(PARTIAL_IDENTIFICATION_ENVELOPE_TAG);
            buf.text(inner.name());
            buf
        }
        Realization::GraphImputationV1
        | Realization::CategoricalPosteriorCandidateV1
        | Realization::SequentialKnownLawMonteCarlo
        | Realization::SequentialTrajectoryBootstrap
        | Realization::ExactPoint { .. }
        | Realization::DeclaredPosterior { .. }
        | Realization::DeclaredPredictive { .. }
        | Realization::NestedFit { .. }
        | Realization::DeclaredAssumptionEnvelope { .. } => return None,
    };
    Some(buf.into_vec())
}

fn evidence_bytes(program: &EvidenceDecisionProgram) -> Vec<u8> {
    let mut buf = CanonBuf::new(super::EVIDENCE_PROGRAM_TAG);
    program.encode_identity(&mut buf);
    buf.into_vec()
}

fn identification_bytes(layer: &IdentificationLayer) -> Result<Vec<u8>, ChildCodecError> {
    let mut buf = CanonBuf::new(IDENTIFICATION_CHILD);
    match layer {
        IdentificationLayer::Deferred { reason, failure: _ } => {
            buf.text("deferred").text(reason);
        }
        IdentificationLayer::PerExecution { reason } => {
            buf.text("per_execution").text(reason);
        }
        IdentificationLayer::Retained(retained) => write_retained(&mut buf, retained)?,
        IdentificationLayer::NonGraphical(identification) => {
            buf.text("non_graphical");
            write_non_graphical(&mut buf, identification);
        }
        IdentificationLayer::NotClaimed { refusal } => {
            buf.text("not_claimed").opt_text(refusal.as_deref());
        }
    }
    Ok(buf.into_vec())
}

fn write_retained(
    buf: &mut CanonBuf,
    retained: &RetainedIdentification,
) -> Result<(), ChildCodecError> {
    buf.text("retained");
    match &retained.product {
        Some(product) => {
            buf.word(1).bytes(product.identity().as_bytes());
        }
        None => {
            buf.word(0);
        }
    }
    buf.word(retained.caches.len() as u64);
    for cache in retained.caches.iter() {
        buf.text(cache.kind());
    }
    let wires = cache_wires(&retained.caches)?;
    buf.word(wires.len() as u64);
    for wire in &wires {
        let bytes = identification_product_canonical_bytes(wire).map_err(|_| {
            err("identification product canonical bytes do not hash to the existing digest")
        })?;
        buf.bytes(&bytes);
    }
    Ok(())
}

fn write_non_graphical(buf: &mut CanonBuf, identification: &NonGraphicalIdentification) {
    buf.text(identification.as_str());
    match identification {
        NonGraphicalIdentification::SpecifiedModelConditional { model } => {
            buf.bytes(model);
        }
        NonGraphicalIdentification::ExperimentalRecovery { design, unresolved } => {
            buf.bytes(design).word(unresolved.len() as u64);
            for (target, missing) in unresolved.iter() {
                buf.word(u64::from(target.raw())).word(missing.len() as u64);
                for source in missing.iter() {
                    buf.word(u64::from(source.raw()));
                }
            }
        }
        NonGraphicalIdentification::CyclicIoScm { source } => {
            buf.bytes(source);
        }
        NonGraphicalIdentification::SharedEvidenceContract { contract } => {
            buf.bytes(contract);
        }
        NonGraphicalIdentification::DeclaredPolicyEvaluation { request }
        | NonGraphicalIdentification::GraphCheckedPolicyEvaluation { request } => {
            buf.bytes(request);
        }
    }
}

fn cache_wires(caches: &[IdentificationCache]) -> Result<Vec<IdentificationProductWire>, ChildCodecError> {
    let mut wires = Vec::new();
    for cache in caches {
        match cache {
            IdentificationCache::Static(cache) => push_wire(&cache.identification, &mut wires)?,
            IdentificationCache::Pag(cache) => push_wire(&cache.identification, &mut wires)?,
            IdentificationCache::Cpdag(cache) => push_wire(&cache.identification, &mut wires)?,
            IdentificationCache::Temporal(cache) => {
                for horizon in cache.by_horizon.iter() {
                    push_wire(&horizon.identification, &mut wires)?;
                }
            }
            IdentificationCache::TemporalClass(_) | IdentificationCache::Transport(_) => {}
            IdentificationCache::GraphPosterior(cache) => {
                for atom in cache.atoms.iter() {
                    push_wire(&atom.identification, &mut wires)?;
                }
                for class in cache.class_atoms.iter() {
                    push_wire(&class.identification, &mut wires)?;
                    for case in class.cases.iter() {
                        push_wire(case.identification.as_ref(), &mut wires)?;
                    }
                }
            }
            IdentificationCache::DbnPosterior(cache) => {
                for atom in cache.atoms.iter() {
                    push_wire(&atom.identification, &mut wires)?;
                    if let Some(horizons) = &atom.horizons {
                        for horizon in horizons.by_horizon.iter() {
                            push_wire(&horizon.identification, &mut wires)?;
                        }
                    }
                }
            }
            IdentificationCache::TemporalClassPosterior(cache) => {
                for atom in cache.class_atoms.iter() {
                    push_wire(&atom.identification, &mut wires)?;
                }
            }
        }
    }
    Ok(wires)
}

fn push_wire(
    result: &IdentificationResult,
    wires: &mut Vec<IdentificationProductWire>,
) -> Result<(), ChildCodecError> {
    wires.push(identification_product_wire(result, false).map_err(|_| err("identification product wire"))?);
    Ok(())
}

fn referents(program: &LoweredProgram) -> Vec<Referent> {
    let mut out = Vec::new();
    match program.scientific().target() {
        ProblemTarget::Target(target) => push_target(&mut out, target),
        ProblemTarget::Targets(targets) => {
            for target in targets.iter() {
                push_target(&mut out, target);
            }
        }
        ProblemTarget::QueryFunctional { query, .. } => out.push(Referent {
            role: "scientific.query",
            digest: *query,
            owner: "query_digest",
        }),
    }
    if let Some(digest) = program.scientific().source_digest() {
        out.push(Referent { role: "scientific.source", digest, owner: "the source rows" });
    }
    if let IdentificationLayer::Retained(retained) = program.identification_layer() {
        if let Some(product) = &retained.product {
            out.push(Referent {
                role: "identification.compiled_product",
                digest: *product.identity().as_bytes(),
                owner: "CompiledIdentificationProduct::identity",
            });
        }
        if let Some(digest) = retained.key {
            out.push(Referent {
                role: "identification.key",
                digest,
                owner: "retention_key",
            });
        }
        for seal in retained.seals.iter().copied().flatten() {
            out.push(Referent { role: "identification.seal", digest: seal, owner: "seal_cache" });
        }
    }
    if let Some(identification) = program.non_graphical_identification() {
        let digest = match identification {
            NonGraphicalIdentification::SpecifiedModelConditional { model } => *model,
            NonGraphicalIdentification::ExperimentalRecovery { design, .. } => *design,
            NonGraphicalIdentification::CyclicIoScm { source } => *source,
            NonGraphicalIdentification::SharedEvidenceContract { contract } => *contract,
            NonGraphicalIdentification::DeclaredPolicyEvaluation { request }
            | NonGraphicalIdentification::GraphCheckedPolicyEvaluation { request } => *request,
        };
        out.push(Referent {
            role: "identification.non_graphical",
            digest,
            owner: "the object named by the non-graphical identification",
        });
    }
    for source in program.evidence().sources() {
        if let Some(digest) = source.digest() {
            out.push(Referent { role: "evidence.source", digest, owner: "the source content" });
        }
    }
    if let Some(planning) = program.evidence().planning() {
        out.push(Referent {
            role: "evidence.planning",
            digest: *planning.request_identity().as_bytes(),
            owner: "PlanningRequest::identity",
        });
    }
    if let Realization::NestedFit { producer_program, .. } = program.statistical().realization() {
        out.push(Referent {
            role: "statistical.nested_fit.producer",
            digest: *producer_program.as_bytes(),
            owner: "the producer program identity",
        });
    }
    if let Some(decision) = program.evidence().decision() {
        if let Some(joint) = decision.utility().preference_facts().and_then(|facts| facts.joint_law())
        {
            out.push(Referent {
                role: "evidence.preference.joint_program",
                digest: *joint.program().as_bytes(),
                owner: "the joint chain program identity",
            });
        }
    }
    if let Some(digest) = program.binding() {
        out.push(Referent { role: "program.binding", digest, owner: "the study binding" });
    }
    out
}

fn push_target(out: &mut Vec<Referent>, target: &antecedent_core::ScientificTarget) {
    out.push(Referent {
        role: "scientific.target",
        digest: *target.identity().as_bytes(),
        owner: "ScientificTarget::identity",
    });
}

fn process_local(program: &LoweredProgram) -> ProcessLocal {
    let runtime_callback = program.identity_scope()
        == ProgramIdentityScope::RuntimeCallbackDeclarationOnly
        || program.scientific().runtime_query.is_some();
    let (arc_graph, arena_in_process) = match program.identification_layer() {
        IdentificationLayer::Deferred { failure, .. } => (false, failure.is_some()),
        IdentificationLayer::Retained(retained) => (
            retained.execution.is_some() || retained.caches.iter().any(cache_holds_graph),
            retained.product.is_some(),
        ),
        IdentificationLayer::PerExecution { .. }
        | IdentificationLayer::NonGraphical(_)
        | IdentificationLayer::NotClaimed { .. } => (false, false),
    };
    ProcessLocal {
        runtime_callback,
        arc_graph,
        arena_in_process,
        causal_state_omitted: true,
    }
}

fn cache_holds_graph(cache: &IdentificationCache) -> bool {
    match cache {
        IdentificationCache::Pag(_)
        | IdentificationCache::Cpdag(_)
        | IdentificationCache::TemporalClass(_)
        | IdentificationCache::GraphPosterior(_)
        | IdentificationCache::DbnPosterior(_)
        | IdentificationCache::TemporalClassPosterior(_) => true,
        IdentificationCache::Static(_)
        | IdentificationCache::Temporal(_)
        | IdentificationCache::Transport(_) => false,
    }
}

fn check_resume(layer: &IdentificationLayer, resume: &IdentificationResume) -> Result<(), ChildCodecError> {
    let matches = match (layer, resume) {
        (
            IdentificationLayer::Deferred { reason, .. },
            IdentificationResume::Reidentify { kind: ReidentifyKind::Deferred, reason: got },
        ) => same_text_str(got, reason),
        (
            IdentificationLayer::PerExecution { reason },
            IdentificationResume::Reidentify { kind: ReidentifyKind::PerExecution, reason: got },
        ) => same_text_str(got, reason),
        (IdentificationLayer::Retained(_), IdentificationResume::Retained) => true,
        (IdentificationLayer::NonGraphical(_), IdentificationResume::NonGraphical) => true,
        (
            IdentificationLayer::NotClaimed { refusal },
            IdentificationResume::NotClaimed { refusal: got },
        ) => same_refusal(got, refusal),
        (
            IdentificationLayer::Deferred { .. }
            | IdentificationLayer::PerExecution { .. }
            | IdentificationLayer::Retained(_)
            | IdentificationLayer::NonGraphical(_)
            | IdentificationLayer::NotClaimed { .. },
            IdentificationResume::Reidentify { .. }
            | IdentificationResume::Retained
            | IdentificationResume::NonGraphical
            | IdentificationResume::NotClaimed { .. },
        ) => false,
    };
    if matches { Ok(()) } else { Err(err("resume does not match the live identification")) }
}

fn same_text_str(got: &str, expected: &str) -> bool {
    got == expected
}

fn same_refusal(got: &Option<String>, expected: &Option<Arc<str>>) -> bool {
    got.as_deref() == expected.as_deref()
}

fn check_realization(
    realization: &Realization,
    bytes: &Option<Vec<u8>>,
) -> Result<(), ChildCodecError> {
    match realization {
        Realization::Frequentist(inner) => same_preimage(bytes, inner.identity()),
        Realization::Bayesian(inner) => same_preimage(bytes, inner.identity()),
        Realization::GeneralizedBayes(inner) => same_preimage(bytes, inner.identity()),
        Realization::Envelope(inner) => same_preimage(bytes, inner.identity()),
        Realization::GraphImputationV1
        | Realization::CategoricalPosteriorCandidateV1
        | Realization::SequentialKnownLawMonteCarlo
        | Realization::SequentialTrajectoryBootstrap
        | Realization::ExactPoint { .. }
        | Realization::DeclaredPosterior { .. }
        | Realization::DeclaredPredictive { .. }
        | Realization::NestedFit { .. }
        | Realization::DeclaredAssumptionEnvelope { .. } => {
            if bytes.is_some() {
                Err(err("this realization has no nested digest"))
            } else {
                Ok(())
            }
        }
    }
}

fn same_preimage(bytes: &Option<Vec<u8>>, digest: SemanticDigest) -> Result<(), ChildCodecError> {
    let Some(bytes) = bytes else {
        return Err(err("missing realization bytes"));
    };
    if framed_digest(bytes) == *digest.as_bytes() {
        Ok(())
    } else {
        Err(err("realization bytes do not hash to the existing digest"))
    }
}

fn check_statistical(program: &StatisticalProgram, bytes: &[u8]) -> Result<(), ChildCodecError> {
    let mut frames = Frames::new(bytes);
    frames.expect_tag(STATISTICAL_V2)?;
    same_opt(&frames.opt_text()?, program.identifier().map(|id| id.as_str()))?;
    same_opt(&frames.opt_text()?, program.estimator().map(|id| id.as_str()))?;
    check_realization_frames(&mut frames, program.realization())?;
    let models = frames.count()?;
    if models != program.models().len() {
        return Err(err("model count"));
    }
    for model in program.models() {
        if frames.text()? != model.role().as_str() || frames.text()? != model.family() {
            return Err(err("model"));
        }
        let inputs = frames.count()?;
        if inputs != model.inputs().len() {
            return Err(err("model inputs"));
        }
        for input in model.inputs() {
            if frames.word()? != u64::from(input.raw()) {
                return Err(err("model input"));
            }
        }
        same_opt_word(frames.opt_word()?, model.output().map(|output| u64::from(output.raw())))?;
    }
    check_folds(&mut frames, program.folds())?;
    check_resampling(&mut frames, program.resampling())?;
    check_posterior(&mut frames, program.posterior())?;
    same_opt_word(frames.opt_word()?, program.design().columns())?;
    if frames.text()? != program.design().source().as_str() {
        return Err(err("design source"));
    }
    if frames.word()? != program.operators().len() as u64 {
        return Err(err("operator count"));
    }
    frames.expect_end()
}

fn check_realization_frames(frames: &mut Frames<'_>, realization: &Realization) -> Result<(), ChildCodecError> {
    match realization {
        Realization::Frequentist(inner) => tagged_digest(frames, "frequentist", inner.identity()),
        Realization::Bayesian(inner) => tagged_digest(frames, "bayesian", inner.identity()),
        Realization::GeneralizedBayes(inner) => {
            tagged_digest(frames, "generalized_bayes", inner.identity())
        }
        Realization::Envelope(inner) => tagged_digest(frames, "envelope", inner.identity()),
        Realization::GraphImputationV1 => same_text(frames, "graph_imputation_v1"),
        Realization::CategoricalPosteriorCandidateV1 => {
            same_text(frames, "categorical_posterior_candidate_v1")
        }
        Realization::SequentialKnownLawMonteCarlo => {
            same_text(frames, "sequential_known_law_monte_carlo_v1")
        }
        Realization::SequentialTrajectoryBootstrap => {
            same_text(frames, "sequential_trajectory_bootstrap_v1")
        }
        Realization::ExactPoint { evaluator } => {
            same_text(frames, "exact_point")?;
            same_text(frames, evaluator)
        }
        Realization::DeclaredPosterior { evaluator } => {
            same_text(frames, "declared_posterior")?;
            same_text(frames, evaluator)
        }
        Realization::DeclaredPredictive { evaluator } => {
            same_text(frames, "declared_predictive")?;
            same_text(frames, evaluator)
        }
        Realization::NestedFit { producer_program, method, law } => {
            same_text(frames, "nested_fit")?;
            if frames.digest()? != *producer_program.as_bytes() {
                return Err(err("nested fit producer"));
            }
            same_text(frames, method)?;
            same_text(
                frames,
                match law {
                    NestedFitLaw::Posterior => "posterior",
                    NestedFitLaw::Predictive => "predictive",
                },
            )
        }
        Realization::DeclaredAssumptionEnvelope { evaluator, draw_meaning } => {
            same_text(frames, "declared_assumption_envelope")?;
            same_text(frames, evaluator)?;
            if frames.word()? == draw_meaning.tag() {
                Ok(())
            } else {
                Err(err("assumption envelope"))
            }
        }
    }
}

fn tagged_digest(frames: &mut Frames<'_>, tag: &str, digest: SemanticDigest) -> Result<(), ChildCodecError> {
    same_text(frames, tag)?;
    if frames.digest()? == *digest.as_bytes() {
        Ok(())
    } else {
        Err(err("realization digest"))
    }
}

fn same_text(frames: &mut Frames<'_>, expected: &str) -> Result<(), ChildCodecError> {
    if frames.text()? == expected { Ok(()) } else { Err(err("text")) }
}

fn check_folds(frames: &mut Frames<'_>, folds: Option<FoldPlan>) -> Result<(), ChildCodecError> {
    match (frames.word()?, folds) {
        (0, None) => Ok(()),
        (1, Some(folds)) => {
            if frames.word()? != folds.folds() as u64 || frames.text()? != folds.unit().as_str() {
                return Err(err("folds"));
            }
            same_opt_word(frames.opt_word()?, folds.embargo().map(|gap| gap as u64))
        }
        (0 | 1, _) => Err(err("folds")),
        _ => Err(err("folds presence")),
    }
}

fn check_resampling(
    frames: &mut Frames<'_>,
    plan: Option<ResamplingPlan>,
) -> Result<(), ChildCodecError> {
    match (frames.word()?, plan) {
        (0, None) => Ok(()),
        (1, Some(plan)) => {
            if frames.text()? == plan.method().as_str()
                && frames.word()? == u64::from(plan.replicates())
                && frames.text()? == plan.unit().as_str()
            {
                Ok(())
            } else {
                Err(err("resampling"))
            }
        }
        (0 | 1, _) => Err(err("resampling")),
        _ => Err(err("resampling presence")),
    }
}

fn check_posterior(frames: &mut Frames<'_>, plan: Option<&PosteriorPlan>) -> Result<(), ChildCodecError> {
    match (frames.word()?, plan) {
        (0, None) => Ok(()),
        (1, Some(plan)) => {
            if frames.text()? == plan.algorithm().as_str()
                && frames.word()? == plan.draws() as u64
                && frames.text()? == plan.prior().name()
            {
                Ok(())
            } else {
                Err(err("posterior"))
            }
        }
        (0 | 1, _) => Err(err("posterior")),
        _ => Err(err("posterior presence")),
    }
}

fn check_evidence(program: &EvidenceDecisionProgram, bytes: &[u8]) -> Result<(), ChildCodecError> {
    let mut frames = Frames::new(bytes);
    frames.expect_tag(super::EVIDENCE_PROGRAM_TAG)?;
    let sources = frames.count()?;
    if sources != program.sources().len() {
        return Err(err("evidence sources"));
    }
    for source in program.sources() {
        if frames.text()? != source.role().as_str() {
            return Err(err("evidence role"));
        }
        match (frames.word()?, source.digest()) {
            (0, None) => {}
            (1, Some(digest)) => {
                if frames.digest()? != digest {
                    return Err(err("evidence source digest"));
                }
            }
            (0 | 1, _) => return Err(err("evidence source digest")),
            (_, _) => return Err(err("evidence source presence")),
        }
    }
    check_composition(&mut frames, program.composition())?;
    match frames.word()? {
        0 if program.decision().is_none() => {}
        1 => check_decision(&mut frames, program.decision().ok_or(err("decision"))?)?,
        _ => return Err(err("decision presence")),
    }
    if let Some(planning) = program.planning() {
        same_text(&mut frames, "planning")?;
        if frames.digest()? != *planning.request_identity().as_bytes() {
            return Err(err("planning request"));
        }
    }
    if let Some(chain) = program.chain() {
        check_chain(&mut frames, chain)?;
    }
    frames.expect_end()
}

fn check_composition(frames: &mut Frames<'_>, composition: Composition) -> Result<(), ChildCodecError> {
    match composition {
        Composition::Single => same_text(frames, "single"),
        Composition::AtomMixture { atoms } => {
            same_text(frames, "atom_mixture")?;
            same_word(frames, atoms as u64)
        }
        Composition::ScenarioSet { scenarios } => {
            same_text(frames, "scenario_set")?;
            same_word(frames, scenarios as u64)
        }
        Composition::Transported => same_text(frames, "transported"),
        Composition::Pooled { sources } => {
            same_text(frames, "pooled")?;
            same_word(frames, sources as u64)
        }
        Composition::Chained { stages } => {
            same_text(frames, "chained")?;
            same_word(frames, stages as u64)
        }
    }
}

fn check_decision(frames: &mut Frames<'_>, decision: &DecisionProgram) -> Result<(), ChildCodecError> {
    same_text(frames, decision.carrier().as_str())?;
    check_utility(frames, decision.utility())?;
    same_word(frames, decision.actions() as u64)?;
    same_word(frames, decision.constraints() as u64)?;
    if frames.real_bits()? != decision.chance_threshold().to_bits() {
        return Err(err("chance threshold"));
    }
    same_opt(&frames.opt_text()?, decision.utility_unit())?;
    same_opt(&frames.opt_text()?, decision.contract_identity())?;
    let scenario = if frames.peek_word()? == JOINT_SCENARIOS.len() as u64 {
        let at = frames.at;
        if frames.text()? == JOINT_SCENARIOS {
            Some(frames.text()?.to_owned())
        } else {
            frames.at = at;
            None
        }
    } else {
        None
    };
    same_opt(&scenario, decision.scenario_source_identity())?;
    match frames.word()? {
        0 if decision.information_value().is_none() => Ok(()),
        1 => check_evsi(frames, decision.information_value().ok_or(err("evsi"))?),
        _ => Err(err("evsi presence")),
    }
}

fn check_utility(frames: &mut Frames<'_>, utility: &UtilityForm) -> Result<(), ChildCodecError> {
    match utility {
        UtilityForm::Affine(coefficients) => {
            same_text(frames, "affine")?;
            same_word(frames, coefficients.len() as u64)?;
            for &(intercept, slope) in coefficients.iter() {
                if frames.real_bits()? != intercept.to_bits() || frames.real_bits()? != slope.to_bits() {
                    return Err(err("affine utility"));
                }
            }
            Ok(())
        }
        UtilityForm::Opaque => same_text(frames, "opaque"),
        UtilityForm::PreferenceValue { quantities, means_suffice, facts } => {
            same_text(frames, "preference_value")?;
            same_word(frames, u64::from(*means_suffice))?;
            let got = frames.texts()?;
            if got.len() != quantities.len() || got.iter().zip(quantities.iter()).any(|(g, q)| g != q.as_ref())
            {
                return Err(err("preference quantities"));
            }
            check_joint(frames, facts.joint_law())?;
            check_design_law(frames, facts.design_law())
        }
    }
}

fn check_joint(frames: &mut Frames<'_>, joint: Option<&PreferenceJointLawFact>) -> Result<(), ChildCodecError> {
    match frames.word()? {
        0 if joint.is_none() => Ok(()),
        1 => {
            let Some(joint) = joint else { return Err(err("joint law")); };
            same_text(frames, "joint_law")?;
            same_text(frames, joint.declaration())?;
            if frames.digest()? == *joint.program().as_bytes() {
                Ok(())
            } else {
                Err(err("joint program"))
            }
        }
        _ => Err(err("joint presence")),
    }
}

fn check_design_law(
    frames: &mut Frames<'_>,
    law: Option<&PreferenceDesignLawFact>,
) -> Result<(), ChildCodecError> {
    match frames.word()? {
        0 if law.is_none() => Ok(()),
        1 => {
            let Some(law) = law else { return Err(err("design law")); };
            same_text(frames, "design_law")?;
            same_text(frames, law.kind())?;
            same_text(frames, law.unit())?;
            same_word(frames, law.replicates() as u64)?;
            same_word(frames, law.seed())?;
            same_opt(&frames.opt_text()?, law.sampling())
        }
        _ => Err(err("design law presence")),
    }
}

fn check_evsi(frames: &mut Frames<'_>, evsi: &EvsiProgram) -> Result<(), ChildCodecError> {
    same_list(frames, evsi.candidates().iter().map(|id| id.as_ref()))?;
    same_list(frames, evsi.prior_observation_ids().iter().map(|id| id.as_ref()))?;
    same_word(frames, evsi.rng_seed())?;
    if frames.real_bits()? != evsi.mc_error_tolerance().to_bits() {
        return Err(err("evsi tolerance"));
    }
    same_word(frames, u64::from(evsi.require_net_value()))?;
    if frames.real_bits()? != evsi.tie_tolerance().to_bits() {
        return Err(err("evsi tie"));
    }
    same_word(frames, evsi.max_candidates())?;
    let rank = evsi.rank();
    same_word(frames, u64::from(rank.min_batches()))?;
    same_word(frames, u64::from(rank.max_batches()))?;
    same_word(frames, u64::from(rank.batch_size()))?;
    if frames.real_bits()? != rank.rank_uncertainty_threshold().to_bits() {
        return Err(err("evsi rank"));
    }
    match frames.word()? {
        0 if evsi.cost_map().is_none() => Ok(()),
        1 => check_cost(frames, evsi.cost_map().ok_or(err("cost map"))?),
        _ => Err(err("cost map presence")),
    }
}

fn check_cost(frames: &mut Frames<'_>, cost: &CostMapping) -> Result<(), ChildCodecError> {
    same_text(frames, cost.cost_unit())?;
    same_text(frames, cost.utility_unit())?;
    if frames.real_bits()? == cost.utility_per_cost().to_bits() {
        Ok(())
    } else {
        Err(err("cost map"))
    }
}

fn check_chain(frames: &mut Frames<'_>, chain: &super::super::chain::ChainProgram) -> Result<(), ChildCodecError> {
    same_text(frames, chain.declaration_identity())?;
    same_text(frames, chain.plan().label)?;
    same_word(frames, chain.plan().stages.len() as u64)?;
    for stage in &chain.plan().stages {
        same_text(frames, &stage.id)?;
        same_text(frames, stage.kind.as_str())?;
        same_text(frames, stage.representation.as_str())?;
        same_list(frames, stage.sources.iter().map(String::as_str))?;
        match (&stage.dependence, frames.word()?) {
            (None, 0) => {}
            (Some(dependence), 1) => {
                same_text(frames, dependence.status.identity_word())?;
                same_list(frames, dependence.shared.iter().map(String::as_str))?;
            }
            (None, 1) | (Some(_), 0) => return Err(err("stage dependence")),
            _ => return Err(err("stage dependence presence")),
        }
    }
    Ok(())
}

fn check_identification(layer: &IdentificationLayer, bytes: &[u8]) -> Result<(), ChildCodecError> {
    let mut frames = Frames::new(bytes);
    frames.expect_tag(IDENTIFICATION_CHILD)?;
    match layer {
        IdentificationLayer::Deferred { reason, failure: _ } => {
            same_text(&mut frames, "deferred")?;
            same_text(&mut frames, reason)?;
        }
        IdentificationLayer::PerExecution { reason } => {
            same_text(&mut frames, "per_execution")?;
            same_text(&mut frames, reason)?;
        }
        IdentificationLayer::Retained(retained) => {
            same_text(&mut frames, "retained")?;
            check_retained_body(&mut frames, retained)?;
        }
        IdentificationLayer::NonGraphical(identification) => {
            same_text(&mut frames, "non_graphical")?;
            check_non_graphical_body(&mut frames, identification)?;
        }
        IdentificationLayer::NotClaimed { refusal } => {
            same_text(&mut frames, "not_claimed")?;
            same_opt(&frames.opt_text()?, refusal.as_deref())?;
        }
    }
    frames.expect_end()
}

fn check_retained_body(frames: &mut Frames<'_>, retained: &RetainedIdentification) -> Result<(), ChildCodecError> {
    match (&retained.product, frames.word()?) {
        (None, 0) => {}
        (Some(product), 1) if frames.digest()? == *product.identity().as_bytes() => {}
        _ => return Err(err("compiled product digest")),
    }
    let kinds = frames.count()?;
    if kinds != retained.caches.len() {
        return Err(err("cache count"));
    }
    for cache in retained.caches.iter() {
        same_text(frames, cache.kind())?;
    }
    let wires = cache_wires(&retained.caches)?;
    let count = frames.count()?;
    if count != wires.len() {
        return Err(err("product wire count"));
    }
    for wire in &wires {
        let bytes = frames.bytes()?;
        let decoded = decode_identification_product(bytes).map_err(|_| err("product wire decode"))?;
        if decoded != *wire {
            return Err(err("product wire"));
        }
    }
    Ok(())
}

fn check_non_graphical_body(
    frames: &mut Frames<'_>,
    identification: &NonGraphicalIdentification,
) -> Result<(), ChildCodecError> {
    same_text(frames, identification.as_str())?;
    match identification {
        NonGraphicalIdentification::SpecifiedModelConditional { model } => same_digest(frames, model),
        NonGraphicalIdentification::ExperimentalRecovery { design, unresolved } => {
            same_digest(frames, design)?;
            same_word(frames, unresolved.len() as u64)?;
            for (target, missing) in unresolved.iter() {
                same_word(frames, u64::from(target.raw()))?;
                same_word(frames, missing.len() as u64)?;
                for source in missing.iter() {
                    same_word(frames, u64::from(source.raw()))?;
                }
            }
            Ok(())
        }
        NonGraphicalIdentification::CyclicIoScm { source } => same_digest(frames, source),
        NonGraphicalIdentification::SharedEvidenceContract { contract } => same_digest(frames, contract),
        NonGraphicalIdentification::DeclaredPolicyEvaluation { request }
        | NonGraphicalIdentification::GraphCheckedPolicyEvaluation { request } => {
            same_digest(frames, request)
        }
    }
}

fn skip_retained(frames: &mut Frames<'_>) -> Result<(), ChildCodecError> {
    if frames.word()? == 1 {
        let _ = frames.digest()?;
    }
    let kinds = frames.count()?;
    for _ in 0..kinds {
        let _ = frames.text()?;
    }
    let wires = frames.count()?;
    for _ in 0..wires {
        let _ = frames.bytes()?;
    }
    Ok(())
}

fn skip_non_graphical(frames: &mut Frames<'_>) -> Result<(), ChildCodecError> {
    match frames.text()? {
        "specified_model_conditional" | "cyclic_ioscm" | "shared_evidence_contract"
        | "declared_policy_evaluation" | "graph_checked_policy_evaluation" => {
            let _ = frames.digest()?;
            Ok(())
        }
        "experimental_recovery" => {
            let _ = frames.digest()?;
            let rows = frames.count()?;
            for _ in 0..rows {
                let _ = frames.word()?;
                let missing = frames.count()?;
                for _ in 0..missing {
                    let _ = frames.word()?;
                }
            }
            Ok(())
        }
        _ => Err(err("non-graphical child")),
    }
}

fn same_opt(got: &Option<String>, expected: Option<&str>) -> Result<(), ChildCodecError> {
    if got.as_deref() == expected { Ok(()) } else { Err(err("optional text")) }
}

fn same_opt_word(got: Option<u64>, expected: Option<u64>) -> Result<(), ChildCodecError> {
    if got == expected { Ok(()) } else { Err(err("optional word")) }
}

fn same_word(frames: &mut Frames<'_>, expected: u64) -> Result<(), ChildCodecError> {
    if frames.word()? == expected { Ok(()) } else { Err(err("word")) }
}

fn same_digest(frames: &mut Frames<'_>, expected: &[u8; 32]) -> Result<(), ChildCodecError> {
    if frames.digest()? == *expected { Ok(()) } else { Err(err("digest")) }
}

fn same_list<'a>(
    frames: &mut Frames<'_>,
    expected: impl Iterator<Item = &'a str>,
) -> Result<(), ChildCodecError> {
    let got = frames.texts()?;
    let expected: Vec<&str> = expected.collect();
    if got.len() == expected.len() && got.iter().zip(expected.iter()).all(|(g, e)| g == e) {
        Ok(())
    } else {
        Err(err("text list"))
    }
}

struct Frames<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Frames<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, at: 0 }
    }

    fn rest(&self) -> usize {
        self.bytes.len().saturating_sub(self.at)
    }

    fn word(&mut self) -> Result<u64, ChildCodecError> {
        if self.rest() < 8 {
            return Err(err("truncated word"));
        }
        let mut le = [0_u8; 8];
        le.copy_from_slice(&self.bytes[self.at..self.at + 8]);
        self.at += 8;
        Ok(u64::from_le_bytes(le))
    }

    fn peek_word(&self) -> Result<u64, ChildCodecError> {
        if self.rest() < 8 {
            return Err(err("truncated word"));
        }
        let mut le = [0_u8; 8];
        le.copy_from_slice(&self.bytes[self.at..self.at + 8]);
        Ok(u64::from_le_bytes(le))
    }

    fn bytes(&mut self) -> Result<&'a [u8], ChildCodecError> {
        let len = usize::try_from(self.word()?).map_err(|_| err("length does not fit"))?;
        if len > self.rest() {
            return Err(err("truncated bytes"));
        }
        let start = self.at;
        self.at += len;
        Ok(&self.bytes[start..self.at])
    }

    fn text(&mut self) -> Result<&'a str, ChildCodecError> {
        std::str::from_utf8(self.bytes()?).map_err(|_| err("text is not utf-8"))
    }

    fn digest(&mut self) -> Result<[u8; 32], ChildCodecError> {
        self.bytes()?.try_into().map_err(|_| err("digest is not 32 bytes"))
    }

    fn count(&mut self) -> Result<usize, ChildCodecError> {
        let n = usize::try_from(self.word()?).map_err(|_| err("count does not fit"))?;
        if self.rest() < n.saturating_mul(8) {
            return Err(err("count exceeds the remaining bytes"));
        }
        Ok(n)
    }

    fn texts(&mut self) -> Result<Vec<String>, ChildCodecError> {
        let n = self.count()?;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.text()?.to_owned());
        }
        Ok(out)
    }

    fn opt_text(&mut self) -> Result<Option<String>, ChildCodecError> {
        match self.word()? {
            0 => Ok(None),
            1 => Ok(Some(self.text()?.to_owned())),
            _ => Err(err("optional text presence")),
        }
    }

    fn opt_word(&mut self) -> Result<Option<u64>, ChildCodecError> {
        match self.word()? {
            0 => Ok(None),
            1 => Ok(Some(self.word()?)),
            _ => Err(err("optional word presence")),
        }
    }

    fn real_bits(&mut self) -> Result<u64, ChildCodecError> {
        self.word()
    }

    fn expect_tag(&mut self, tag: &str) -> Result<(), ChildCodecError> {
        same_text(self, tag).map_err(|_| err("child tag"))
    }

    fn expect_end(&self) -> Result<(), ChildCodecError> {
        if self.at == self.bytes.len() { Ok(()) } else { Err(err("trailing child bytes")) }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use antecedent_contract::DataClassification;

    use super::super::super::{
        CarrierDemand, ClaimDemand, Composition, DecisionProgram, DesignShape,
        EvidenceDecisionProgram, EvidenceRole, EvidenceSourceRef, EvsiProgram, FunctionalKind,
        IdentificationLayer, LoweredParts, LoweringDepth, NonGraphicalIdentification, PhysicalLayer,
        ProblemTarget, ProgramIdentityScope, RetainedIdentification, ScientificProblem,
        UncertaintyReading,
    };
    use super::super::RankSchedule;
    use super::{
        check_children, check_realization, check_statistical, framed_digest, realization_preimage,
        resolve_children, resume_identification, statistical_bytes, IdentificationResume,
        ReidentifyKind,
    };
    use crate::analysis::lowering::RouteCell;
    use crate::planner::statistical::{
        DeclaredProgram, FoldPlan, ModelRole, ModelSlot, Realization, ResampleUnit, StatisticalProgram,
    };
    use crate::{EstimatorId, IdentifierId, InferenceMode};

    fn scientific(scope: ProgramIdentityScope) -> ScientificProblem {
        ScientificProblem {
            target: ProblemTarget::QueryFunctional {
                kind: FunctionalKind::Association,
                outcome: None,
                query: [1; 32],
            },
            identity_scope: scope,
            runtime_query: None,
            plan_id: Arc::from("child"),
            data: DataClassification::Tabular,
            graph_class: "dag",
            query_variables: Arc::from([antecedent_core::VariableId::from_raw(0)]),
            source_digest: Some([9; 32]),
            support_coordinate: None,
            supported: true,
            support_status: None,
        }
    }

    fn statistical() -> StatisticalProgram {
        StatisticalProgram::declared(DeclaredProgram {
            identifier: Some(IdentifierId::BackdoorAdjustment),
            estimator: Some(EstimatorId::LinearAdjustmentAte),
            inference: &InferenceMode::Frequentist,
            bootstrap_replicates: 2,
            sampling_design: None,
            design: DesignShape::quoted(4, 2, true),
            operators: Vec::new(),
        })
        .expect("program")
    }

    fn decision() -> DecisionProgram {
        DecisionProgram {
            carrier: CarrierDemand::Mean,
            utility: DecisionProgram::utility_form(None),
            actions: 2,
            constraints: 0,
            chance_threshold: 0.95,
            utility_unit: None,
            contract_identity: None,
            information_value: Some(EvsiProgram {
                candidates: Arc::from([Arc::from("a")]),
                prior_observation_ids: Arc::<[Arc<str>]>::from([]),
                rng_seed: 3,
                mc_error_tolerance: 0.01,
                require_net_value: false,
                tie_tolerance: 0.0,
                max_candidates: 8,
                rank: RankSchedule {
                    min_batches: 1,
                    max_batches: 4,
                    batch_size: 16,
                    rank_uncertainty_threshold: 0.0,
                },
                cost_map: None,
                monte_carlo: EvsiProgram::quote_tasks(1, 1).expect("quote"),
            }),
            scenario_source: Some(Arc::from("src")),
        }
    }

    fn evidence() -> EvidenceDecisionProgram {
        EvidenceDecisionProgram {
            sources: Arc::from([EvidenceSourceRef {
                role: EvidenceRole::Observations,
                digest: Some([7; 32]),
            }]),
            composition: Composition::AtomMixture { atoms: 2 },
            decision: Some(decision()),
            planning: None,
            chain: None,
        }
    }

    fn finish(scope: ProgramIdentityScope, layer: IdentificationLayer) -> super::super::super::LoweredProgram {
        LoweredParts {
            cell: RouteCell::StaticAteDag,
            depth: LoweringDepth::Complete,
            scientific: scientific(scope),
            identification: layer,
            statistical: statistical(),
            evidence: evidence(),
            physical: PhysicalLayer {
                operators: Arc::from(Vec::new()),
                design: DesignShape::not_design_based("child"),
            },
            claim: ClaimDemand { uncertainty: UncertaintyReading::None, kind: None },
        }
        .finish()
    }

    #[test]
    fn layer_bytes_hash_to_the_existing_digests_and_match_the_live_program() {
        let program = finish(
            ProgramIdentityScope::Semantic,
            IdentificationLayer::PerExecution { reason: "each execution" },
        );
        let identity = program.identity();
        let children = resolve_children(&program).expect("children");
        check_children(&program, &children).expect("check");
        assert_eq!(program.identity(), identity);
        assert_eq!(
            framed_digest(&children.statistical),
            *program.statistical().lowered_layer_digest().as_bytes()
        );
        assert_eq!(framed_digest(&children.evidence), *program.evidence().identity().as_bytes());
        assert!(matches!(
            resume_identification(&children.identification).expect("resume"),
            IdentificationResume::Reidentify { kind: ReidentifyKind::PerExecution, ref reason }
                if reason == "each execution"
        ));
        assert!(!children.process_local.runtime_callback);
        assert!(children.process_local.causal_state_omitted);
        assert!(!children.process_local.arc_graph);
        assert!(children.referents.iter().any(|referent| {
            referent.role == "scientific.query" && referent.digest == [1; 32]
        }));
        let mut tampered = children.clone();
        tampered.statistical[0] ^= 1;
        assert!(check_children(&program, &tampered).is_err());
    }

    #[test]
    fn a_runtime_callback_is_not_part_of_the_child_bytes() {
        let semantic = finish(
            ProgramIdentityScope::Semantic,
            IdentificationLayer::PerExecution { reason: "each execution" },
        );
        let callback = finish(
            ProgramIdentityScope::RuntimeCallbackDeclarationOnly,
            IdentificationLayer::PerExecution { reason: "each execution" },
        );
        assert_eq!(semantic.identity(), callback.identity());
        let semantic = resolve_children(&semantic).expect("semantic");
        let callback = resolve_children(&callback).expect("callback");
        assert_eq!(semantic.statistical, callback.statistical);
        assert_eq!(semantic.evidence, callback.evidence);
        assert_eq!(semantic.identification, callback.identification);
        assert!(!semantic.process_local.runtime_callback);
        assert!(callback.process_local.runtime_callback);
    }

    #[test]
    fn deferred_identification_is_reidentified_and_a_retained_product_is_not() {
        let deferred = finish(
            ProgramIdentityScope::Semantic,
            IdentificationLayer::Deferred { reason: "compile retained nothing", failure: None },
        );
        let children = resolve_children(&deferred).expect("deferred");
        check_children(&deferred, &children).expect("check deferred");
        assert!(matches!(
            resume_identification(&children.identification).expect("resume"),
            IdentificationResume::Reidentify { kind: ReidentifyKind::Deferred, ref reason }
                if reason == "compile retained nothing"
        ));
        assert!(!children.process_local.arena_in_process);

        let retained = RetainedIdentification::new(None, Vec::new(), Some([3; 32])).expect("retain");
        let program = finish(
            ProgramIdentityScope::Semantic,
            IdentificationLayer::Retained(Arc::new(retained)),
        );
        let children = resolve_children(&program).expect("retained");
        check_children(&program, &children).expect("check retained");
        assert_eq!(
            resume_identification(&children.identification).expect("resume"),
            IdentificationResume::Retained
        );
        assert!(children.referents.iter().any(|referent| {
            referent.role == "identification.key" && referent.owner == "retention_key"
        }));
    }

    #[test]
    fn a_non_graphical_digest_is_checked_against_the_live_program() {
        let program = finish(
            ProgramIdentityScope::Semantic,
            IdentificationLayer::NonGraphical(Arc::new(
                NonGraphicalIdentification::SpecifiedModelConditional { model: [4; 32] },
            )),
        );
        let children = resolve_children(&program).expect("children");
        check_children(&program, &children).expect("check");
        assert_eq!(
            resume_identification(&children.identification).expect("resume"),
            IdentificationResume::NonGraphical
        );
        assert!(children.referents.iter().any(|referent| {
            referent.digest == [4; 32] && referent.owner.contains("non-graphical")
        }));
    }

    #[test]
    fn bayesian_and_exact_realizations_keep_their_digests() {
        let inference = InferenceMode::Bayesian(crate::BayesianConfig::laplace());
        let bayesian = StatisticalProgram::declared(DeclaredProgram {
            identifier: None,
            estimator: None,
            inference: &inference,
            bootstrap_replicates: 0,
            sampling_design: None,
            design: DesignShape::not_design_based("child"),
            operators: Vec::new(),
        })
        .expect("bayesian");
        let bytes = statistical_bytes(&bayesian);
        assert_eq!(framed_digest(&bytes), *bayesian.lowered_layer_digest().as_bytes());
        check_statistical(&bayesian, &bytes).expect("bayesian frames");
        let preimage = realization_preimage(bayesian.realization()).expect("prior bytes");
        check_realization(bayesian.realization(), &Some(preimage)).expect("bayesian preimage");

        let mut exact = statistical();
        exact.realization = Realization::ExactPoint { evaluator: Arc::from("evaluator") };
        exact.models = Arc::from([ModelSlot {
            role: ModelRole::Outcome,
            family: Arc::from("ols"),
            inputs: Arc::from([antecedent_core::VariableId::from_raw(0)]),
            output: Some(antecedent_core::VariableId::from_raw(1)),
        }]);
        exact.folds = Some(FoldPlan { folds: 2, unit: ResampleUnit::Row, embargo: Some(1) });
        let bytes = statistical_bytes(&exact);
        assert_eq!(framed_digest(&bytes), *exact.lowered_layer_digest().as_bytes());
        check_statistical(&exact, &bytes).expect("exact frames");
        assert!(realization_preimage(&exact.realization).is_none());
    }
}
