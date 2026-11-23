//! Several outcomes in one source.
//!
//! Separately compiled studies of distinct outcomes meet as one mean source when means
//! suffice (the expectation of a value affine in the outcomes reads each outcome's mean on
//! its own), and otherwise only under a declared dependence: under distinct evidence each
//! study's posterior covariance is kept within it, nothing is shared between studies, and the
//! propagation chain draws one aligned law from the moment-matched family table of their
//! posterior summaries. Rows of separately produced draws are never paired by index.
//!
//! The outcomes of one compiled static-regime design are read from the design's own laws:
//! its arm means, or its aligned whole-subject bootstrap refits as one aligned law over the
//! arm coordinates, with any failed refit refused rather than dropped.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::dependence::SharedFactorKind;
use antecedent_contract::propagation_chain::{ChainReading, ChainRefusal, LawMeaning, StageKind};
use antecedent_contract::{ExternalRefusal, ScientificQuantity};
use antecedent_core::reason_details::{decision_evaluation, decision_evidence, native_claims};
use antecedent_core::{ExecutionContext, SemanticDigest};
use antecedent_design::decision_contract::SourceRepresentation;
use antecedent_design::decision_eval::{MeanEstimate, MeanSource};
use antecedent_design::preference_contract::PreferenceContract;
use antecedent_estimate::joint_design::compiled::{
    JointDesignInference, JointDesignRecipe, JointNativeDesignLaw,
};
use antecedent_estimate::joint_design::static_regime::{
    JointStaticRegimeBootstrap, JointStaticRegimeLaw,
};
use antecedent_estimate::temporal_design::regime::MIN_REPLICATES;
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io::propagation_chain_record::{
    ChainCoefficientWire, ChainJointOutputWire, ChainOpWire, ChainStageWire,
};
use antecedent_io::quantity_wire::DistributionMeaningWire;

use super::{
    BoundStudy, CompiledPreferenceDecision, DemandedQuantity, PreferenceDemand,
    PreferenceRouteError, STAGE_BIND, STAGE_COMPILE, STAGE_EVALUATE, Sources, StudyEntry,
    arm_coordinate, bound_identity, compile_entries, coordinate_refusal, demand_of,
    quantity_inputs,
};
use crate::Study;
use crate::analysis::joint_panel::designs::PreparedNativeJointDesign;
use crate::analysis::native_claims::{NativeDecisionInput, NativeDecisionSource};
use crate::analysis::preference_decision::chain_joint_law;
use crate::analysis::propagation_chain::{
    FamilyDependence, FamilyEstimate, compile_propagation_chain, family_table,
};
use crate::planner::PlanView;

/// How separately estimated outcomes depend on each other.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutcomeDependence {
    /// No dependence is declared. Each outcome's mean is read on its own, which answers the
    /// expectation of a value affine in the outcomes; a contract that needs a joint law is
    /// refused at compile.
    Undeclared,
    /// The studies rest on distinct evidence (their data snapshots differ): each study's
    /// posterior covariance is kept within it and nothing is shared between studies. The
    /// joint law is the propagation chain's family table matched to those moments, with
    /// `draws` aligned rows drawn under `seed`.
    DistinctEvidence {
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

/// Check that separately estimated outcomes can be joined as the demand needs.
pub(super) fn check_dependence(
    studies: &[BoundStudy],
    demand: &PreferenceDemand,
    dependence: OutcomeDependence,
) -> Result<(), PreferenceRouteError> {
    if studies.len() < 2 || demand.representation == SourceRepresentation::Mean {
        return Ok(());
    }
    let OutcomeDependence::DistinctEvidence { draws, .. } = dependence else {
        return Err(unpaired("separately estimated outcomes with no declared dependence"));
    };
    if draws < 2 {
        return Err(unpaired("a joined law of fewer than two aligned rows"));
    }
    for (index, study) in studies.iter().enumerate() {
        if let Some(other) = studies[index + 1..].iter().find(|o| o.snapshot == study.snapshot) {
            return Err(ExternalRefusal::new(
                antecedent_core::reason_code!("joint_law_required"),
                STAGE_COMPILE,
                decision_evidence::DEPENDENCE_UNKNOWN,
            )
            .with_offending(format!(
                "{} and {}",
                study.binding.outcome_id, other.binding.outcome_id
            ))
            .with_supplied("one data snapshot behind both outcomes")
            .with_remedy(
                "outcomes estimated from the same rows are not distinct evidence; estimate \
                 them jointly in one design",
            )
            .into());
        }
    }
    Ok(())
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
/// law with no declared dependence, or with distinct evidence declared for studies of the
/// same data snapshot (`joint_law_required`).
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
    compile_entries(&entries, contract, dependence, ctx)
}

/// Join the sources of separately estimated outcomes: their means side by side, or one
/// aligned law from the chain under the declared distinct evidence. Returns the source and
/// the chain's program identity.
pub(super) fn join_studies(
    inputs: &[NativeDecisionInput],
    studies: &[BoundStudy],
    dependence: OutcomeDependence,
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
    let OutcomeDependence::DistinctEvidence { draws, seed } = dependence else {
        return Err(unpaired("separately estimated outcomes with no declared dependence"));
    };
    let mut laws = Vec::with_capacity(inputs.len());
    for (input, study) in inputs.iter().zip(studies) {
        let NativeDecisionSource::JointLaw(law) = &input.source else {
            return Err(unpaired("a mean source among the laws to join"));
        };
        laws.push((law.as_ref(), study.snapshot.as_str()));
    }
    let (law, program) = chain_law(&laws, (draws, seed), alignment, ctx)?;
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

/// The posterior summaries of every study's law as a family table under distinct evidence
/// (each study's covariance within it, nothing between studies), drawn into one aligned law
/// by the propagation chain. Returns the law and the chain's program identity.
fn chain_law(
    laws: &[(&DistributionArtifact, &str)],
    (draws, seed): (usize, u64),
    alignment: &str,
    ctx: &ExecutionContext,
) -> Result<(DistributionArtifact, SemanticDigest), PreferenceRouteError> {
    let mut estimates = Vec::new();
    let mut columns: Vec<(usize, Vec<f64>)> = Vec::new();
    let mut outputs = Vec::new();
    for (study, (law, rows)) in laws.iter().enumerate() {
        let [n, width] = law.shape();
        let masked = law.metadata().supported.as_ref().is_some_and(|mask| mask.contains(&false));
        if law.metadata().weights.is_some() || masked || n < 2 {
            return Err(law_refusal(format!(
                "study {study}: weighted, partly unsupported or fewer than two draws; a joined \
                 law never drops a weight or a support mask"
            )));
        }
        for j in 0..width {
            let values: Vec<f64> = (0..n).map(|draw| law.draws()[draw * width + j]).collect();
            let stage = format!("study_{study}_column_{j}");
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
    let k = columns.len();
    let mut declared = vec![0.0; k * k];
    for (i, (left, x)) in columns.iter().enumerate() {
        for (j, (right, y)) in columns.iter().enumerate() {
            if left == right {
                declared[i * k + j] = covariance(x, y);
            }
        }
    }
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

/// A compiled static-regime design read as the source of a decision.
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
                NativeDecisionSource::Mean(self.arm_means(point, &dependency)?)
            }
            (JointNativeDesignLaw::StaticRegimeBootstrap(refits), _) => {
                NativeDecisionSource::JointLaw(Box::new(self.refit_law(refits, &dependency)?))
            }
            _ => return Err(law_refusal("the design returned another law".to_owned())),
        };
        Ok((source, programs))
    }

    fn arm_means(
        &self,
        point: &JointStaticRegimeLaw,
        dependency: &str,
    ) -> Result<MeanSource, PreferenceRouteError> {
        if point.means.len() != self.coordinates.len() {
            return Err(law_refusal("arm means misaligned with the arm coordinates".to_owned()));
        }
        Ok(MeanSource {
            coordinates: self.coordinates.clone(),
            means: point.means.clone(),
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

/// The arm coordinates of a static-regime design: every outcome under the treated regime,
/// then every outcome under the control regime, at the horizon of the active path.
fn design_coordinates(
    design: &PreparedNativeJointDesign,
) -> Result<Vec<ScientificQuantity>, PreferenceRouteError> {
    let JointDesignRecipe::StaticRegime { recipe } = design.recipe() else {
        return Err(ExternalRefusal::new(
            antecedent_core::reason_code!("route_not_supported"),
            STAGE_BIND,
            decision_evidence::UNSUPPORTED_ASSIGNMENT_DESIGN,
        )
        .with_expected("a static-regime design of several outcomes")
        .with_supplied(design.identification().method.clone())
        .into());
    };
    let horizon = u32::try_from(recipe.active_path.len())
        .map_err(|_| law_refusal("regime horizon too large".to_owned()))?;
    let mut coordinates = Vec::with_capacity(2 * recipe.outcomes.len());
    for regime in [&recipe.treated_regime_id, &recipe.control_regime_id] {
        for outcome in &recipe.outcomes {
            coordinates.push(arm_coordinate(
                &outcome.variable_id,
                &outcome.component_name,
                &outcome.units,
                &recipe.population,
                regime,
                horizon,
            ));
        }
    }
    Ok(coordinates)
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

impl PreparedNativeJointDesign {
    /// Compile a decision across the preference set of `contract` whose quantities are the
    /// arm means of this compiled static-regime design's outcomes.
    ///
    /// Each quantity must be one outcome under the treated or the control regime, in the
    /// design's population and units, at the horizon of its active path, as a mean on the
    /// identity scale. When means suffice the design's frequentist point law answers;
    /// otherwise `bootstrap` declares the aligned whole-subject refits that supply one law of
    /// every outcome under both arms. Nothing is executed.
    ///
    /// # Errors
    /// A design that is not a static regime; an invalid contract; a quantity that is not one
    /// of the design's arm coordinates (naming the action and the attribute); a contract that
    /// needs a joint law with no bootstrap declared, or fewer refits than the design's floor.
    pub fn compile_for_preference_decision(
        &self,
        contract: &PreferenceContract,
        bootstrap: Option<DesignBootstrap>,
        ctx: &ExecutionContext,
    ) -> Result<CompiledPreferenceDecision, PreferenceRouteError> {
        crate::analysis::lowering::refuse_reentry(ctx)?;
        let contract_identity = contract.identity().map_err(|e| e.to_refusal())?;
        let coordinates = design_coordinates(self)?;
        let quantities = bind_design(contract, &coordinates)?;
        let (mut demand, _) = demand_of(contract, quantities, false)?;
        demand.carrier = None;
        let inference = match (demand.representation, bootstrap) {
            (SourceRepresentation::Mean, _) => JointDesignInference::Frequentist,
            (_, Some(DesignBootstrap { replicates, seed })) if replicates >= MIN_REPLICATES => {
                JointDesignInference::StaticRegimeBootstrap { replicates, seed }
            }
            (_, Some(_)) => {
                return Err(unpaired("fewer whole-subject refits than the design's floor of 99"));
            }
            (_, None) => {
                return Err(unpaired("the design's point law carries arm means only"));
            }
        };
        let causal = vec![(self.program_identity().to_hex(), self.dependency().to_hex())];
        let identity = bound_identity(&contract_identity, &causal);
        Ok(CompiledPreferenceDecision {
            contract: contract.clone(),
            contract_identity,
            demand,
            identity,
            causal,
            dependence: OutcomeDependence::Undeclared,
            sources: Sources::Design(Box::new(DesignSource {
                design: self.clone(),
                coordinates,
                inference,
            })),
        })
    }
}
