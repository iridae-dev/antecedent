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
//! The outcomes of one compiled design are read from the design's own laws, with the decision
//! attached to layer four of the design's program and its carrier checked at compile:
//!
//! * a static-regime design: its arm means, or its aligned whole-subject bootstrap refits as
//!   one aligned law over the arm coordinates, with any failed refit refused rather than
//!   dropped;
//! * a cluster-randomized design and a matched-pair or switchback experiment: their arm means,
//!   under an exposure free of interference. Neither has an aligned joint law: the cluster law
//!   retains per-arm conservative Neyman covariance bounds
//!   (`antecedent_estimate::joint_design::cluster_decision::JointClusterLaw`) and the
//!   experiment law a conservative design-based covariance estimate of its arm means
//!   (`antecedent_estimate::randomized_joint::matched_pair_joint_contrast`, the switchback
//!   ITT variances), and the covariance of the potential outcomes across arms that a law
//!   would need is not identified by a randomized design. A demand for a joint law of these
//!   designs is refused at compile naming the design.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::dependence::SharedFactorKind;
use antecedent_contract::propagation_chain::{ChainReading, ChainRefusal, LawMeaning, StageKind};
use antecedent_contract::{ExternalRefusal, ScientificQuantity};
use antecedent_core::reason_details::{decision_evaluation, native_claims, preference_compile};
use antecedent_core::{ExecutionContext, SemanticDigest};
use antecedent_design::decision_contract::SourceRepresentation;
use antecedent_design::decision_eval::{MeanEstimate, MeanSource};
use antecedent_design::preference_contract::PreferenceContract;
use antecedent_estimate::joint_design::compiled::{
    JointDesignInference, JointDesignRecipe, JointNativeDesignLaw,
};
use antecedent_estimate::joint_design::experiment::ExperimentExposure;
use antecedent_estimate::joint_design::static_regime::JointStaticRegimeBootstrap;
use antecedent_estimate::temporal_design::regime::MIN_REPLICATES;
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io::propagation_chain_record::{
    ChainCoefficientWire, ChainJointOutputWire, ChainOpWire, ChainStageWire,
};
use antecedent_io::quantity_wire::{DistributionMeaningWire, ScientificQuantityWire};

use super::evidence::{CrossStudyCovariance, SharedEvidence};
use super::{
    BoundStudy, CompiledPreferenceDecision, DemandedQuantity, PreferenceDemand,
    PreferenceRouteError, STAGE_BIND, STAGE_COMPILE, STAGE_EVALUATE, Sources, StudyEntry,
    arm_coordinate, bound_identity, compile_entries, coordinate_refusal, decision_program,
    demand_of, quantity_inputs,
};
use crate::analysis::joint_panel::designs::PreparedNativeJointDesign;
use crate::analysis::native_claims::{NativeDecisionInput, NativeDecisionSource};
use crate::analysis::preference_decision::chain_joint_law;
use crate::analysis::propagation_chain::{
    FamilyDependence, FamilyEstimate, compile_propagation_chain, family_table,
};
use crate::planner::PlanView;
use crate::planner::lowered::{CarrierDemand, DecisionProgram};
use crate::planner::statistical::Realization;
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

fn unpaired(supplied: &str) -> PreferenceRouteError {
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

fn law_refusal(supplied: String) -> PreferenceRouteError {
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

// ---------------------------------------------------------------------------------------------
// One design, several outcomes
// ---------------------------------------------------------------------------------------------

/// Aligned whole-subject bootstrap refits a static-regime design draws when the decision
/// needs a joint law of its outcomes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DesignBootstrap {
    /// Complete refits; at least the design's floor of 99.
    pub replicates: usize,
    /// Resampling seed.
    pub seed: u64,
}

/// The designs whose arm coordinates a decision reads.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DesignKind {
    /// A static-regime design over recorded longitudinal histories.
    StaticRegime,
    /// A cluster-randomized design.
    Cluster,
    /// A matched-pair or switchback experiment.
    Experiment,
}

/// A compiled design read as the source of a decision, entering the design's program with the
/// decision attached to layer four.
#[derive(Clone, Debug)]
pub(super) struct DesignSource {
    design: PreparedNativeJointDesign,
    coordinates: Vec<ScientificQuantity>,
    inference: JointDesignInference,
}

impl DesignSource {
    pub(super) fn plan_view(&self) -> PlanView {
        self.design.plan_view()
    }

    /// Execute the design once and read the representation the demand named.
    pub(super) fn execute(
        &self,
        representation: SourceRepresentation,
        ctx: &ExecutionContext,
    ) -> Result<(NativeDecisionSource, Vec<SemanticDigest>), PreferenceRouteError> {
        let result = self.design.execute(self.inference.clone(), ctx)?;
        let programs = vec![result.program_identity()];
        let dependency = result.dependency_identity().to_hex();
        let source = match (result.law(), representation) {
            (JointNativeDesignLaw::StaticRegime(point), SourceRepresentation::Mean) => {
                NativeDecisionSource::Mean(self.arm_means(point.means.clone(), &dependency)?)
            }
            (JointNativeDesignLaw::DecisionCluster(law), SourceRepresentation::Mean) => {
                let means = law.treated_means.iter().chain(&law.control_means).copied().collect();
                NativeDecisionSource::Mean(self.arm_means(means, &dependency)?)
            }
            (JointNativeDesignLaw::DecisionExperiment(law), SourceRepresentation::Mean) => {
                NativeDecisionSource::Mean(self.arm_means(law.means.clone(), &dependency)?)
            }
            (JointNativeDesignLaw::StaticRegimeBootstrap(refits), _) => {
                NativeDecisionSource::JointLaw(Box::new(self.refit_law(refits, &dependency)?))
            }
            _ => return Err(law_refusal("the design returned another law".to_owned())),
        };
        Ok((source, programs))
    }

    /// The arm means, treated arm then control arm, as a mean source over the arm
    /// coordinates.
    fn arm_means(
        &self,
        means: Vec<f64>,
        dependency: &str,
    ) -> Result<MeanSource, PreferenceRouteError> {
        if means.len() != self.coordinates.len() {
            return Err(law_refusal("arm means misaligned with the arm coordinates".to_owned()));
        }
        Ok(MeanSource {
            coordinates: self.coordinates.clone(),
            means,
            estimate: MeanEstimate::Frequentist,
            provider_id: format!("native_joint_design:{}", self.design.program_identity().to_hex()),
            snapshot_id: self.design.dependency().to_hex(),
            causal_contract_id: dependency.to_owned(),
            rng_id: "none:arm_means".to_owned(),
        })
    }

    /// The aligned refits as one law over the arm coordinates: one row per refit, treated
    /// arm then control arm. A failed refit is refused, never dropped.
    fn refit_law(
        &self,
        refits: &JointStaticRegimeBootstrap,
        dependency: &str,
    ) -> Result<DistributionArtifact, PreferenceRouteError> {
        let failed = refits.failures.iter().filter(|failure| failure.is_some()).count();
        if failed > 0 {
            return Err(law_refusal(format!(
                "{failed} of {} refits failed; a failed slot is never dropped",
                refits.realizations.len()
            )));
        }
        let width = self.coordinates.len();
        let mut rows = Vec::with_capacity(refits.realizations.len() * width);
        for realization in &refits.realizations {
            match realization {
                Some(row) if row.len() == width => rows.extend_from_slice(row),
                _ => return Err(law_refusal("a refit misaligned with the arms".to_owned())),
            }
        }
        let invalid = |error: antecedent_io::IoError| law_refusal(error.to_string());
        let identity = DistributionIdentity::new(
            DistributionMeaningWire::Bootstrap,
            &self.coordinates,
            DrawAlignment::Joint,
            DistributionProvenance {
                source_id: dependency.to_owned(),
                provider_id: "antecedent.native_joint_design".to_owned(),
                rng_id: format!("{:?}", self.inference),
                snapshot_id: self.design.dependency().to_hex(),
                causal_contract_id: self.design.program_identity().to_hex(),
            },
        )
        .map_err(invalid)?;
        DistributionArtifact::new(
            DistributionMetadata {
                version: 1,
                identity,
                axes: ["draw".to_owned(), "quantity".to_owned()],
                shape: [refits.realizations.len(), width],
                weights: None,
                supported: None,
                calibration: DistributionCalibration::NotMeasured,
                trust: DistributionTrust::Unverified,
                legacy_posterior: None,
                legacy_bindings: None,
            },
            rows,
        )
        .map_err(invalid)
    }
}

fn unsupported_design(design: &PreparedNativeJointDesign) -> PreferenceRouteError {
    ExternalRefusal::new(
        antecedent_core::reason_code!("route_not_supported"),
        STAGE_BIND,
        preference_compile::DESIGN_NOT_SUPPORTED,
    )
    .with_expected(
        "a static-regime, cluster-randomized, matched-pair or switchback design of several \
         outcomes",
    )
    .with_supplied(design.identification().method.clone())
    .into()
}

fn bounded_exposure(design: &PreparedNativeJointDesign) -> PreferenceRouteError {
    ExternalRefusal::new(
        antecedent_core::reason_code!("effect_not_identified"),
        STAGE_COMPILE,
        preference_compile::DESIGN_EXPOSURE_BOUNDED,
    )
    .with_offending(design.identification().method.clone())
    .with_expected("an exposure free of between-unit interference")
    .with_supplied("declared bias bounds on the arm means or their contrast")
    .with_remedy("declare no interference, or decide over the bounded set with a robust decision")
    .into()
}

/// The refusal of a joint-law demand on a design whose law carries arm means and a
/// conservative design-based covariance, never an aligned joint law.
fn law_not_aligned(design: &PreparedNativeJointDesign) -> PreferenceRouteError {
    ExternalRefusal::new(
        antecedent_core::reason_code!("joint_law_required"),
        STAGE_COMPILE,
        preference_compile::DESIGN_LAW_NOT_ALIGNED,
    )
    .with_offending(design.identification().method.clone())
    .with_expected("aligned joint draws of every outcome under both arms")
    .with_supplied(
        "arm means with a conservative design-based covariance estimate; the covariance of \
         the potential outcomes across arms is not identified by the randomization",
    )
    .with_remedy(
        "decide with a value affine in the outcomes and an expectation criterion, which the \
         arm means answer",
    )
    .into()
}

/// The arm coordinates of `outcomes` under `treated` then `control`.
fn arm_coordinates(
    outcomes: &[antecedent_estimate::joint_design::excursion::ExcursionOutcome],
    (population, horizon): (&str, u32),
    (treated, control): (&str, &str),
) -> Vec<ScientificQuantity> {
    let mut coordinates = Vec::with_capacity(2 * outcomes.len());
    for regime in [treated, control] {
        for outcome in outcomes {
            coordinates.push(arm_coordinate(
                &outcome.variable_id,
                &outcome.component_name,
                &outcome.units,
                population,
                regime,
                horizon,
            ));
        }
    }
    coordinates
}

const fn free_of_interference(exposure: &ExperimentExposure) -> bool {
    matches!(exposure, ExperimentExposure::NoInterference { .. })
}

/// The arm coordinates of a design: every outcome under the treated regime, then every
/// outcome under the control regime, at the design's horizon (for a static regime, the
/// length of its active path).
fn design_coordinates(
    design: &PreparedNativeJointDesign,
) -> Result<(Vec<ScientificQuantity>, DesignKind), PreferenceRouteError> {
    match design.recipe() {
        JointDesignRecipe::StaticRegime { recipe } => {
            let horizon = u32::try_from(recipe.active_path.len())
                .map_err(|_| law_refusal("regime horizon too large".to_owned()))?;
            let coordinates = arm_coordinates(
                &recipe.outcomes,
                (&recipe.population, horizon),
                (&recipe.treated_regime_id, &recipe.control_regime_id),
            );
            Ok((coordinates, DesignKind::StaticRegime))
        }
        JointDesignRecipe::DecisionCluster { recipe } => {
            if !free_of_interference(&recipe.exposure) {
                return Err(bounded_exposure(design));
            }
            let coordinates = arm_coordinates(
                &recipe.outcomes,
                (&recipe.population, recipe.horizon),
                (&recipe.treated_regime_id, &recipe.control_regime_id),
            );
            Ok((coordinates, DesignKind::Cluster))
        }
        JointDesignRecipe::DecisionExperiment { recipe } => {
            if !free_of_interference(&recipe.exposure) {
                return Err(bounded_exposure(design));
            }
            let coordinates = arm_coordinates(
                &recipe.outcomes,
                (&recipe.population, recipe.horizon),
                (&recipe.treated_regime_id, &recipe.control_regime_id),
            );
            Ok((coordinates, DesignKind::Experiment))
        }
        _ => Err(unsupported_design(design)),
    }
}

/// Bind every quantity input to an arm coordinate of the design.
fn bind_design(
    contract: &PreferenceContract,
    coordinates: &[ScientificQuantity],
) -> Result<Vec<DemandedQuantity>, PreferenceRouteError> {
    let mut demanded: Vec<DemandedQuantity> = Vec::new();
    for (action, attribute, quantity) in quantity_inputs(contract) {
        let Some(coordinate) =
            coordinates.iter().find(|c| quantity.require_same_coordinate(c).is_ok())
        else {
            return Err(coordinate_refusal(coordinates, (action, attribute), quantity));
        };
        let read = (action.to_owned(), attribute.to_owned());
        match demanded.iter_mut().find(|d| d.quantity == *coordinate) {
            Some(entry) => entry.read_by.push(read),
            None => demanded.push(DemandedQuantity {
                quantity: coordinate.clone(),
                source: 0,
                read_by: vec![read],
            }),
        }
    }
    Ok(demanded)
}

/// The inference that supplies the representation the demand names, and the carrier the
/// decision demands of the design's program.
fn design_inference(
    design: &PreparedNativeJointDesign,
    kind: DesignKind,
    representation: SourceRepresentation,
    bootstrap: Option<DesignBootstrap>,
) -> Result<(JointDesignInference, CarrierDemand), PreferenceRouteError> {
    match (representation, kind, bootstrap) {
        (SourceRepresentation::Mean, _, _) => {
            Ok((JointDesignInference::Frequentist, CarrierDemand::Mean))
        }
        (_, DesignKind::Cluster | DesignKind::Experiment, _) => Err(law_not_aligned(design)),
        (_, DesignKind::StaticRegime, Some(DesignBootstrap { replicates, seed }))
            if replicates >= MIN_REPLICATES =>
        {
            Ok((
                JointDesignInference::StaticRegimeBootstrap { replicates, seed },
                CarrierDemand::DecisionDraws,
            ))
        }
        (_, DesignKind::StaticRegime, Some(_)) => {
            Err(unpaired("fewer whole-subject refits than the design's floor of 99"))
        }
        (_, DesignKind::StaticRegime, None) => {
            Err(unpaired("the design's point law carries arm means only"))
        }
    }
}

/// Check, at compile, that the design's realization supplies the source the decision program
/// demands: a mean of an affine value from the design's exact point functional, or aligned
/// draws from its whole-subject refits.
fn require_design_carrier(
    design: &PreparedNativeJointDesign,
    program: &DecisionProgram,
    inference: &JointDesignInference,
) -> Result<(), PreferenceRouteError> {
    let exact_point =
        matches!(design.lowered().statistical().realization(), Realization::ExactPoint { .. });
    let supplied = match program.carrier() {
        CarrierDemand::Mean => {
            exact_point
                && program.utility().is_affine()
                && matches!(inference, JointDesignInference::Frequentist)
        }
        CarrierDemand::DecisionDraws => {
            matches!(inference, JointDesignInference::StaticRegimeBootstrap { .. })
        }
        CarrierDemand::ParameterPosterior
        | CarrierDemand::OutcomeLaw
        | CarrierDemand::PredictiveLaw
        | CarrierDemand::IdentifiedSet => false,
    };
    if supplied { Ok(()) } else { Err(law_not_aligned(design)) }
}

impl PreparedNativeJointDesign {
    /// Compile a decision across the preference set of `contract` whose quantities are the
    /// arm means of this compiled design's outcomes.
    ///
    /// Each quantity must be one outcome under the treated or the control regime, in the
    /// design's population and units, at its horizon (a static regime's active path, a
    /// randomized design's declared horizon), as a mean on the identity scale. The decision
    /// is attached to layer four of the design's program, so its plan view shows the
    /// decision's quantities, carrier and parameters, and the carrier is checked against the
    /// realization at compile. When means suffice the design's frequentist point law
    /// answers; otherwise a static-regime design's `bootstrap` declares the aligned
    /// whole-subject refits that supply one law of every outcome under both arms. Nothing is
    /// executed.
    ///
    /// # Errors
    /// A design that is not a static regime, a cluster-randomized design or a matched-pair or
    /// switchback experiment (`preference_compile.design_not_supported`); a randomized design
    /// whose exposure declares bias bounds (`preference_compile.design_exposure_bounded`); an
    /// invalid contract; a quantity that is not one of the design's arm coordinates (naming
    /// the action and the attribute); a contract that needs a joint law of a cluster or
    /// experiment design, which has none (`preference_compile.design_law_not_aligned`); a
    /// static-regime contract that needs a joint law with no bootstrap declared, or fewer
    /// refits than the design's floor (`joint_law_required`).
    pub fn compile_for_preference_decision(
        &self,
        contract: &PreferenceContract,
        bootstrap: Option<DesignBootstrap>,
        ctx: &ExecutionContext,
    ) -> Result<CompiledPreferenceDecision, PreferenceRouteError> {
        crate::analysis::lowering::refuse_reentry(ctx)?;
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: super::CANCELLED_STAGE }.into());
        }
        let contract_identity = contract.identity().map_err(|e| e.to_refusal())?;
        let (coordinates, kind) = design_coordinates(self)?;
        let quantities = bind_design(contract, &coordinates)?;
        let (mut demand, _) = demand_of(contract, quantities, false)?;
        let (inference, carrier) = design_inference(self, kind, demand.representation, bootstrap)?;
        demand.carrier = Some(carrier);
        let causal = vec![(self.program_identity().to_hex(), self.dependency().to_hex())];
        let identity = bound_identity(&contract_identity, &causal);
        let read: Vec<&DemandedQuantity> = demand.quantities.iter().collect();
        let stated = super::facts::value_facts(contract, &demand, &read, None, None);
        let program = decision_program(contract, &identity, stated, (&demand, carrier));
        require_design_carrier(self, &program, &inference)?;
        let lowered = self.lowered().with_decision(program).ok_or_else(|| {
            PreferenceRouteError::from(
                ExternalRefusal::new(
                    antecedent_core::reason_code!("invalid_argument"),
                    STAGE_COMPILE,
                    preference_compile::PROGRAM_CARRIES_DECISION,
                )
                .with_offending(self.program_identity().to_hex()),
            )
        })?;
        Ok(CompiledPreferenceDecision {
            contract: contract.clone(),
            contract_identity,
            demand,
            identity,
            causal,
            dependence: OutcomeDependence::Undeclared,
            sources: Sources::Design(Box::new(DesignSource {
                design: self.with_lowered(Arc::new(lowered)),
                coordinates,
                inference,
            })),
            shared: Arc::from(Vec::new()),
            joint: None,
        })
    }
}
