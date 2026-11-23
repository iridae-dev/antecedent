//! Several outcomes in one source.
//!
//! Separately compiled studies of distinct outcomes meet as one mean source when means
//! suffice (the expectation of a value affine in the outcomes reads each outcome's mean on
//! its own), and otherwise only under a declared dependence: under distinct evidence each
//! study's posterior covariance is kept within it, nothing is shared between studies, and the
//! propagation chain draws one aligned law from the moment-matched family table of their
//! posterior summaries; under shared evidence the declared cross-study blocks enter the
//! family table's covariance beside them. The evidence ledger judges at compile whether the
//! studies share observations, and the chain is declared and checked at compile. Rows of
//! separately produced draws are never paired by index.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::ExternalRefusal;
use antecedent_contract::dependence::SharedFactorKind;
use antecedent_contract::propagation_chain::{ChainReading, ChainRefusal, LawMeaning, StageKind};
use antecedent_core::reason_details::{decision_evaluation, native_claims};
use antecedent_core::{ExecutionContext, SemanticDigest};
use antecedent_design::decision_contract::SourceRepresentation;
use antecedent_design::decision_eval::{MeanEstimate, MeanSource};
use antecedent_design::preference_contract::PreferenceContract;
use antecedent_io::distribution_artifact::DistributionArtifact;
use antecedent_io::propagation_chain_record::{
    ChainCoefficientWire, ChainJointOutputWire, ChainOpWire, ChainStageWire,
};
use antecedent_io::quantity_wire::ScientificQuantityWire;

use super::evidence::{CrossStudyCovariance, SharedEvidence};
use super::{
    BoundStudy, CompiledPreferenceDecision, PreferenceDemand, PreferenceRouteError, STAGE_COMPILE,
    STAGE_EVALUATE, StudyEntry, compile_entries,
};
use crate::analysis::native_claims::{NativeDecisionInput, NativeDecisionSource};
use crate::analysis::preference_decision::chain_joint_law;
use crate::analysis::propagation_chain::{
    FamilyDependence, FamilyEstimate, compile_propagation_chain, family_table,
};
use crate::{CausalError, PreparedStudy, Study};

/// How separately estimated outcomes depend on each other.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutcomeDependence {
    /// No dependence is declared. Each outcome's mean is read on its own, which answers the
    /// expectation of a value affine in the outcomes; a contract that needs a joint law is
    /// refused at compile.
    Undeclared,
    /// The studies rest on distinct evidence (their tables share no rows): each study's
    /// posterior covariance is kept within it and nothing is shared between studies. The
    /// joint law is the propagation chain's family table matched to those moments, with
    /// `draws` aligned rows drawn under `seed`.
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
/// studies share observations ([`super::evidence::judge`]).
pub(super) fn check_dependence(
    studies: &[BoundStudy],
    demand: &PreferenceDemand,
    (dependence, shared): (OutcomeDependence, &[CrossStudyCovariance]),
    ctx: &ExecutionContext,
) -> Result<(), PreferenceRouteError> {
    if studies.len() < 2 || demand.representation == SourceRepresentation::Mean {
        return Ok(());
    }
    let (OutcomeDependence::DistinctEvidence { draws, .. }
    | OutcomeDependence::SharedEvidence { draws, .. }) = dependence
    else {
        return Err(unpaired("separately estimated outcomes with no declared dependence"));
    };
    if draws < 2 {
        return Err(unpaired("a joined law of fewer than two aligned rows"));
    }
    super::evidence::judge(studies, dependence, shared, ctx)
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
/// studies whose tables share rows (`evidence_double_counted`).
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
/// aligned law from the chain under the declared distinct evidence. Returns the source and
/// the chain's program identity.
pub(super) fn join_studies(
    inputs: &[NativeDecisionInput],
    studies: &[BoundStudy],
    (dependence, shared): (OutcomeDependence, &[CrossStudyCovariance]),
    alignment: &str,
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
    let (OutcomeDependence::DistinctEvidence { draws, seed }
    | OutcomeDependence::SharedEvidence { draws, seed }) = dependence
    else {
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
        laws.push((law.as_ref(), study.observations.as_str()));
    }
    let (law, program) = chain_law(&laws, (draws, seed, shared), alignment, ctx)?;
    Ok((NativeDecisionSource::JointLaw(Box::new(law)), Some(program)))
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

/// Covariance of two aligned columns of equally weighted draws.
fn covariance(x: &[f64], y: &[f64]) -> f64 {
    let (mx, my) = (mean(x), mean(y));
    x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum::<f64>() / x.len() as f64
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

/// Each study's covariance within it, and the declared block between studies that share
/// observations; zero between the others.
fn declared_covariance(
    columns: &[(usize, Vec<f64>)],
    offsets: &[usize],
    shared: &[CrossStudyCovariance],
) -> Result<Vec<f64>, PreferenceRouteError> {
    let k = columns.len();
    let mut declared = vec![0.0; k * k];
    for (i, (left, x)) in columns.iter().enumerate() {
        for (j, (right, y)) in columns.iter().enumerate() {
            if left == right {
                declared[i * k + j] = covariance(x, y);
            }
        }
    }
    for block in shared {
        let (Some(&first), Some(&second)) = (offsets.get(block.first), offsets.get(block.second))
        else {
            return Err(law_refusal(
                "a declared block names a study the chain does not join".to_owned(),
            ));
        };
        let rows = columns.iter().filter(|(study, _)| *study == block.first).count();
        let cols = columns.iter().filter(|(study, _)| *study == block.second).count();
        if block.covariance.len() != rows * cols {
            return Err(law_refusal(
                "a declared block misaligned with the studies' laws".to_owned(),
            ));
        }
        for p in 0..rows {
            for q in 0..cols {
                let term = block.covariance[p * cols + q];
                declared[(first + p) * k + second + q] = term;
                declared[(second + q) * k + first + p] = term;
            }
        }
    }
    Ok(declared)
}

/// The posterior summaries of every study's law as a family table (each study's covariance
/// within it, the declared cross-study blocks between studies that share observations and
/// nothing between the others), drawn into one aligned law by the propagation chain whose
/// declaration the compile checked. Returns the law and the chain's program identity.
fn chain_law(
    laws: &[(&DistributionArtifact, &str)],
    (draws, seed, shared): (usize, u64, &[CrossStudyCovariance]),
    alignment: &str,
    ctx: &ExecutionContext,
) -> Result<(DistributionArtifact, SemanticDigest), PreferenceRouteError> {
    let mut estimates = Vec::new();
    let mut columns: Vec<(usize, Vec<f64>)> = Vec::new();
    let mut outputs = Vec::new();
    let mut offsets = Vec::with_capacity(laws.len());
    for (study, (law, rows)) in laws.iter().enumerate() {
        let [n, width] = law.shape();
        let masked = law.metadata().supported.as_ref().is_some_and(|mask| mask.contains(&false));
        if law.metadata().weights.is_some() || masked || n < 2 {
            return Err(law_refusal(format!(
                "study {study}: weighted, partly unsupported or fewer than two draws; a joined \
                 law never drops a weight or a support mask"
            )));
        }
        offsets.push(columns.len());
        for j in 0..width {
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled { stage: super::CANCELLED_STAGE }.into());
            }
            let values: Vec<f64> = (0..n).map(|draw| law.draws()[draw * width + j]).collect();
            let stage = super::joint::stage_id(study, j);
            let spread = covariance(&values, &values).sqrt();
            estimates.push(
                FamilyEstimate::new(
                    &format!("{stage}_source"),
                    mean(&values),
                    spread,
                    LawMeaning::Posterior,
                )
                .with_factor(SharedFactorKind::Row, rows),
            );
            let quantity = law.quantities().get(j).cloned().ok_or_else(|| {
                law_refusal(format!("study {study}: column {j} has no coordinate"))
            })?;
            outputs.push(ChainJointOutputWire { stage, quantity });
            columns.push((study, values));
        }
    }
    let declared = declared_covariance(&columns, &offsets, shared)?;
    let table = family_table(
        alignment,
        &estimates,
        &FamilyDependence::Covariance(declared),
        None,
        draws,
        seed,
    )
    .map_err(|refusal| chain_refusal(&refusal))?;
    let stages = outputs
        .iter()
        .map(|output| {
            ChainStageWire::new(
                &output.stage,
                StageKind::DerivedQuantity,
                &[format!("{}_source", output.stage).as_str()],
                ChainOpWire::affine(
                    ChainCoefficientWire::fixed(1.0),
                    ChainCoefficientWire::fixed(0.0),
                ),
            )
        })
        .collect();
    let Some((output, quantity)) =
        outputs.first().map(|first| (first.stage.clone(), first.quantity.clone()))
    else {
        return Err(law_refusal("no column to join".to_owned()));
    };
    let draft = table
        .draft(ChainReading::Posterior, &output, stages)
        .with_output_quantity(quantity)
        .with_joint_outputs(outputs);
    let chain = compile_propagation_chain(draft, ctx)
        .and_then(|plan| plan.execute(ctx))
        .map_err(|refusal| chain_refusal(&refusal))?;
    let law = chain_joint_law(&chain)?;
    Ok((law, chain.lowered_program_identity()))
}
