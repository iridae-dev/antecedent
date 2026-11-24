//! Several outcomes in one source.
//!
//! Separately compiled studies of distinct outcomes meet as one mean source when means
//! suffice (the expectation of a value affine in the outcomes reads each outcome's mean on
//! its own), and otherwise only under a declared dependence: under distinct evidence each
//! study's posterior covariance is kept within it, nothing is shared between studies, and the
//! propagation chain draws one aligned law from the moment-matched family table of their
//! posterior summaries; under shared evidence the declared cross-study blocks enter the
//! family table's covariance beside them. The evidence ledger judges at compile whether the
//! studies share observations, and the chain is declared, checked and compiled at compile with
//! its family table deferred; execution binds each study's posterior moments to it. Rows of
//! separately produced draws are never paired by index.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::ExternalRefusal;
use antecedent_contract::propagation_chain::ChainRefusal;
use antecedent_core::reason_details::{decision_evaluation, native_claims};
use antecedent_core::{ExecutionContext, SemanticDigest};
use antecedent_design::decision_contract::SourceRepresentation;
use antecedent_design::decision_eval::{MeanEstimate, MeanSource};
use antecedent_design::preference_contract::PreferenceContract;
use antecedent_io::distribution_artifact::DistributionArtifact;
use antecedent_io::quantity_wire::ScientificQuantityWire;

use super::evidence::{CrossStudyCovariance, SharedEvidence, StudyOverlaps};
use super::joint::PreferenceJointLaw;
use super::{
    BoundStudy, CompiledPreferenceDecision, PreferenceDemand, PreferenceRouteError, STAGE_COMPILE,
    STAGE_EVALUATE, StudyEntry, compile_entries,
};
use crate::analysis::native_claims::{NativeDecisionInput, NativeDecisionSource};
use crate::analysis::preference_decision::chain_joint_law;
use crate::analysis::propagation_chain::ChainTableBinding;
use crate::{CausalError, PreparedStudy, Study};

/// How separately estimated outcomes depend on each other.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutcomeDependence {
    /// No dependence is declared. Each outcome's mean is read on its own, which answers the
    /// expectation of a value affine in the outcomes; a contract that needs a joint law is
    /// refused at compile.
    Undeclared,
    /// The caller declares that the studies rest on distinct evidence: each study's
    /// posterior covariance is kept within it and nothing is shared between studies. The
    /// joint law is the propagation chain's family table matched to those moments, with
    /// `draws` aligned rows drawn under `seed`.
    ///
    /// The evidence ledger still checks the declaration where it can: studies whose analysed
    /// rows come from one source table, or that are keyed in one identifier space, are refused
    /// when they share rows or units. Studies of different tables without unit keys cannot be
    /// compared, so they are distinct on this declaration alone, and the plan states that
    /// basis for the pair (`declared_distinct`).
    DistinctEvidence {
        /// Aligned rows of the joined law.
        draws: usize,
        /// Seed of the family table.
        seed: u64,
    },
    /// Some of the studies share observations and the covariance between their estimates is
    /// declared ([`SharedEvidence`]): the joint law is the chain's family table matched to
    /// each study's posterior moments with the declared cross-study blocks beside them.
    SharedEvidence {
        /// Aligned rows of the joined law.
        draws: usize,
        /// Seed of the family table.
        seed: u64,
    },
}

pub(super) fn unpaired(supplied: &str) -> PreferenceRouteError {
    ExternalRefusal::new(
        antecedent_core::reason_code!("joint_law_required"),
        STAGE_COMPILE,
        decision_evaluation::JOINT_LAW_REQUIRED,
    )
    .with_expected("aligned joint draws of every outcome the decision reads")
    .with_supplied(supplied)
    .with_remedy(
        "declare distinct evidence when the studies rest on distinct rows, or estimate the \
         outcomes jointly in one design",
    )
    .into()
}

/// Check that separately estimated outcomes can be joined as the demand needs: a declared
/// dependence, at least two aligned rows, and the evidence ledger's judgment of whether the
/// studies share observations ([`super::evidence::judge`]). Returns the pairs that share
/// observations, when the ledger judged them.
pub(super) fn check_dependence(
    studies: &[BoundStudy],
    demand: &PreferenceDemand,
    (dependence, shared): (OutcomeDependence, &[CrossStudyCovariance]),
    ctx: &ExecutionContext,
) -> Result<Option<StudyOverlaps>, PreferenceRouteError> {
    if studies.len() < 2 || demand.representation == SourceRepresentation::Mean {
        return Ok(None);
    }
    let (OutcomeDependence::DistinctEvidence { draws, .. }
    | OutcomeDependence::SharedEvidence { draws, .. }) = dependence
    else {
        return Err(unpaired("separately estimated outcomes with no declared dependence"));
    };
    if draws < 2 {
        return Err(unpaired("a joined law of fewer than two aligned rows"));
    }
    super::evidence::judge(studies, dependence, shared, ctx).map(Some)
}

/// Compile a decision across the preference set of `contract` whose quantities are the
/// outcomes of several studies, one outcome per study.
///
/// Every study is bound, lowered with the decision in layer four and prepared as
/// [`Study::compile_for_preference_decision`] does. When means suffice the outcomes'
/// means are read side by side; otherwise `dependence` must declare how the outcomes depend
/// on each other.
///
/// # Errors
/// Every refusal of [`Study::compile_for_preference_decision`] for each study; two studies
/// of the same outcome; a graph posterior among the studies; a contract that needs a joint
/// law with no declared dependence (`joint_law_required`); distinct evidence declared for
/// studies that share rows of one source table or units of one identifier space
/// (`evidence_double_counted`); a unit key on one study and not on a study of another table,
/// or keys of different identifier spaces (`evidence_overlap_unknown`); a unit key naming the
/// treatment or outcome (`preference_compile.unit_key_invalid`); a joined law whose family
/// table exceeds its work budget (`moment_matched_table_work_budget`), at compile.
pub fn compile_preference_decision_across_studies(
    studies: &[&Study],
    contract: &PreferenceContract,
    dependence: OutcomeDependence,
    ctx: &ExecutionContext,
) -> Result<CompiledPreferenceDecision, PreferenceRouteError> {
    let mut entries = Vec::with_capacity(studies.len());
    for study in studies {
        entries.push(StudyEntry { study, causal: study.inspect()?, prepared: None });
    }
    compile_entries(&entries, contract, (dependence, &[]), ctx)
}

/// Compile a decision across the preference set of `contract` whose quantities are the
/// outcomes of several studies, some of which share observations, with the covariance
/// between their estimates declared.
///
/// As [`compile_preference_decision_across_studies`] under
/// [`OutcomeDependence::SharedEvidence`]: the evidence ledger judges which studies share
/// observations, every such pair needs its block in `shared`, and the propagation chain
/// receives the blocks as the cross-study part of its declared covariance. The within-study
/// blocks are each study's posterior covariance, so the chain checks the whole matrix (its
/// diagonal against each study's spread, and positive semi-definiteness) when it draws.
///
/// # Errors
/// Every refusal of [`compile_preference_decision_across_studies`]; a malformed block or one
/// naming studies that share nothing (`preference_compile.cross_study_covariance_invalid`);
/// studies that share observations with no declared block
/// (`preference_compile.studies_share_evidence`).
pub fn compile_preference_decision_with_shared_evidence(
    studies: &[&Study],
    contract: &PreferenceContract,
    shared: &SharedEvidence,
    ctx: &ExecutionContext,
) -> Result<CompiledPreferenceDecision, PreferenceRouteError> {
    let mut entries = Vec::with_capacity(studies.len());
    for study in studies {
        entries.push(StudyEntry { study, causal: study.inspect()?, prepared: None });
    }
    let dependence = OutcomeDependence::SharedEvidence { draws: shared.draws, seed: shared.seed };
    compile_entries(&entries, contract, (dependence, &shared.cross), ctx)
}

/// Compile a decision across the preference set of `contract` whose quantities are the
/// outcomes of several prepared studies, one outcome per study: the same binding, demand,
/// lowering and checks as [`compile_preference_decision_across_studies`] (and, with
/// `cross`, as [`compile_preference_decision_with_shared_evidence`]), against each handle's
/// frozen contract. A handle executes its study's single structure.
///
/// `cross` declares the covariance of every pair of studies that share observations and is
/// read only under [`OutcomeDependence::SharedEvidence`].
///
/// # Errors
/// As [`compile_preference_decision_across_studies`] and
/// [`compile_preference_decision_with_shared_evidence`].
pub fn compile_preference_decision_across_prepared_studies(
    studies: &[&PreparedStudy],
    contract: &PreferenceContract,
    (dependence, cross): (OutcomeDependence, &[CrossStudyCovariance]),
    ctx: &ExecutionContext,
) -> Result<CompiledPreferenceDecision, PreferenceRouteError> {
    let mut entries = Vec::with_capacity(studies.len());
    for prepared in studies {
        let study = prepared.study();
        let handle = study.graph_posterior.is_none().then_some(*prepared);
        entries.push(StudyEntry { study, causal: prepared.contract()?, prepared: handle });
    }
    let cross =
        if matches!(dependence, OutcomeDependence::SharedEvidence { .. }) { cross } else { &[] };
    compile_entries(&entries, contract, (dependence, cross), ctx)
}

/// Join the sources of separately estimated outcomes: their means side by side, or one
/// aligned law drawn by the chain compiled at compile, its deferred family table bound to each
/// study's posterior moments. Returns the source and the chain's compiled program identity.
pub(super) fn join_studies(
    inputs: &[NativeDecisionInput],
    studies: &[BoundStudy],
    joint: Option<&PreferenceJointLaw>,
    ctx: &ExecutionContext,
) -> Result<(NativeDecisionSource, Option<SemanticDigest>), PreferenceRouteError> {
    let means: Vec<&MeanSource> = inputs
        .iter()
        .filter_map(|input| match &input.source {
            NativeDecisionSource::Mean(means) => Some(means),
            NativeDecisionSource::JointLaw(_) => None,
        })
        .collect();
    if means.len() == inputs.len() {
        return Ok((NativeDecisionSource::Mean(side_by_side(&means)), None));
    }
    let Some(joint) = joint else {
        return Err(unpaired("separately estimated outcomes with no declared dependence"));
    };
    let mut laws = Vec::with_capacity(inputs.len());
    for (input, study) in inputs.iter().zip(studies) {
        let NativeDecisionSource::JointLaw(law) = &input.source else {
            return Err(unpaired("a mean source among the laws to join"));
        };
        let declared: Vec<ScientificQuantityWire> =
            study.coordinates.iter().map(ScientificQuantityWire::from).collect();
        if law.quantities() != declared.as_slice() {
            return Err(law_refusal(format!(
                "study {}: the law's coordinates are not the ones the chain was declared over",
                study.binding.outcome_id
            )));
        }
        laws.push(law.as_ref());
    }
    let binding = table_binding(&laws, ctx)?;
    let chain =
        joint.program().execute_bound(&binding, ctx).map_err(|refusal| chain_refusal(&refusal))?;
    let law = chain_joint_law(&chain)?;
    Ok((NativeDecisionSource::JointLaw(Box::new(law)), Some(chain.lowered_program_identity())))
}

/// Means of separately estimated outcomes side by side; each mean stays its own study's.
fn side_by_side(means: &[&MeanSource]) -> MeanSource {
    let join = |field: fn(&MeanSource) -> &str| {
        means.iter().map(|source| field(source)).collect::<Vec<_>>().join("+")
    };
    let estimate = match means.first() {
        Some(first) if means.iter().all(|source| source.estimate == first.estimate) => {
            first.estimate
        }
        _ => MeanEstimate::Frequentist,
    };
    MeanSource {
        coordinates: means.iter().flat_map(|source| source.coordinates.iter().cloned()).collect(),
        means: means.iter().flat_map(|source| source.means.iter().copied()).collect(),
        estimate,
        provider_id: join(|source| source.provider_id.as_str()),
        snapshot_id: join(|source| source.snapshot_id.as_str()),
        causal_contract_id: join(|source| source.causal_contract_id.as_str()),
        rng_id: join(|source| source.rng_id.as_str()),
    }
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// Sample covariance (divisor `n - 1`) of two aligned columns of `n >= 2` equally weighted
/// draws: the convention the family table matches its rows to, so an equally weighted reader
/// of the joined rows sees each study's own covariance.
fn covariance(x: &[f64], y: &[f64]) -> f64 {
    let (mx, my) = (mean(x), mean(y));
    let divisor = x.len().saturating_sub(1).max(1) as f64;
    x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum::<f64>() / divisor
}

fn chain_refusal(refusal: &ChainRefusal) -> PreferenceRouteError {
    ExternalRefusal::new(refusal.code(), STAGE_EVALUATE, refusal.detail())
        .with_supplied(refusal.message_with_next_attempt())
        .into()
}

pub(super) fn law_refusal(supplied: String) -> PreferenceRouteError {
    ExternalRefusal::new(
        antecedent_core::reason_code!("invalid_argument"),
        STAGE_EVALUATE,
        native_claims::JOINT_LAW_INVALID,
    )
    .with_supplied(supplied)
    .with_remedy("retain finite, aligned, equally weighted draws of every coordinate")
    .into()
}

/// The posterior moments of every study's law, in the deferred table's slot and group order:
/// each column's mean, and each study's sample covariance within it (divisor `n - 1`). The
/// family table's rows have exactly that sample covariance (divisor `rows - 1`), so the
/// joined law reproduces each study's own covariance under one convention.
fn table_binding(
    laws: &[&DistributionArtifact],
    ctx: &ExecutionContext,
) -> Result<ChainTableBinding, PreferenceRouteError> {
    let mut means = Vec::new();
    let mut within = Vec::with_capacity(laws.len());
    for (study, law) in laws.iter().enumerate() {
        let [n, width] = law.shape();
        let masked = law.metadata().supported.as_ref().is_some_and(|mask| mask.contains(&false));
        if law.metadata().weights.is_some() || masked || n < 2 {
            return Err(law_refusal(format!(
                "study {study}: weighted, partly unsupported or fewer than two draws; a joined \
                 law never drops a weight or a support mask"
            )));
        }
        let mut columns = Vec::with_capacity(width);
        for j in 0..width {
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled { stage: super::CANCELLED_STAGE }.into());
            }
            let values: Vec<f64> = (0..n).map(|draw| law.draws()[draw * width + j]).collect();
            means.push(mean(&values));
            columns.push(values);
        }
        let mut block = vec![0.0; width * width];
        for (p, x) in columns.iter().enumerate() {
            for (q, y) in columns.iter().enumerate() {
                block[p * width + q] = covariance(x, y);
            }
        }
        within.push(block);
    }
    Ok(ChainTableBinding { means, within })
}
