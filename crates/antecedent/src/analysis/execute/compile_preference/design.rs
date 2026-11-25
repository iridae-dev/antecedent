//! One design, several outcomes.
//!
//! The outcomes of one compiled design are read from the design's own laws, with the decision
//! attached to layer four of the design's program and its carrier checked at compile:
//!
//! * a static-regime design: its arm means, or its aligned whole-subject bootstrap refits as
//!   one aligned law over the arm coordinates, with any failed refit refused rather than
//!   dropped;
//! * a cluster-randomized design and a matched-pair or switchback experiment: their arm means,
//!   under an exposure free of interference. Randomization alone identifies the arm means and
//!   a conservative covariance bound for their linear combinations, but not the dependence of
//!   the arm-mean estimators across arms, which rests on the joint distribution of the
//!   potential outcomes under both arms. A decision that needs a joint law of these designs
//!   therefore reads one only under a declared premise that the independent units (clusters,
//!   pairs or sequences) were sampled independently from a larger population: then the
//!   independent-unit bootstrap
//!   (`antecedent_estimate::joint_design::unit_bootstrap::joint_unit_bootstrap`) refits the
//!   whole estimator on resampled units and its aligned refits are the law, with any failed
//!   refit refused rather than dropped. Without the premise the demand is refused at compile
//!   naming the design.
//!
//! The resampling law (its kind, unit, refits, seed and premise) is stated in the decision
//! program, so it enters the program's identity, and the premise enters the evaluation's
//! assumptions.
//!
//! The same compile answers an ordinary decision contract, whose actions each read their own
//! arm coordinates through their own utility
//! ([`PreparedNativeJointDesign::compile_for_contract_decision`]).
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::{ExternalRefusal, QuantityRole, ScientificQuantity};
use antecedent_core::SemanticDigest;
use antecedent_core::reason_details::preference_compile;
use antecedent_core::{CanonicalWriter, ExecutionContext};
use antecedent_design::decision_contract::{DecisionContract, SourceRepresentation};
use antecedent_design::decision_eval::{MeanEstimate, MeanSource};
use antecedent_design::preference_contract::PreferenceContract;
use antecedent_design::preference_evaluation::PreferenceEvaluation;
use antecedent_estimate::joint_design::compiled::{
    JointDesignInference, JointDesignRecipe, JointNativeDesignLaw,
};
use antecedent_estimate::joint_design::experiment::ExperimentExposure;
use antecedent_estimate::joint_design::unit_bootstrap::{
    IndependentUnit, MIN_INDEPENDENT_UNITS, MIN_REFITS, independent_units,
};
use antecedent_estimate::temporal_design::regime::MIN_REPLICATES;
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io::quantity_wire::DistributionMeaningWire;

use super::bind::{bound_identity, coordinate_refusal, quantity_inputs};
use super::outcomes::{OutcomeDependence, law_refusal, unpaired};
use super::{
    CompiledContract, CompiledDecision, CompiledPreferenceDecision, CompiledPreferenceEvaluation,
    DemandedQuantity, IDENTITY_TRANSFORM, MEAN_FUNCTIONAL, PreferenceRouteError, STAGE_BIND,
    STAGE_COMPILE, Sources, checked_executed, decision_program, demand_of, parent_of,
};
use crate::CausalError;
use crate::analysis::joint_panel::designs::PreparedNativeJointDesign;
use crate::analysis::native_claims::NativeDecisionSource;
use crate::planner::PlanView;
use crate::planner::lowered::{
    CarrierDemand, Composition, DecisionProgram, PreferenceDesignLawFact,
};
use crate::planner::statistical::Realization;

/// Domain of the digest naming a refit law's resampling recipe.
const RESAMPLING_TAG: &str = "antecedent.preference_route.design_resampling.v1";
/// Kind of the whole-subject refits of a static-regime design.
const STATIC_REGIME_BOOTSTRAP: &str = "static_regime_bootstrap";
/// Kind of the independent-unit bootstrap of a randomized design.
const INDEPENDENT_UNIT_BOOTSTRAP: &str = "independent_unit_bootstrap";

/// The refits a compiled design draws when the decision needs a joint law of its outcomes:
/// aligned whole-subject refits of a static-regime design, or the independent-unit bootstrap
/// of a cluster-randomized, matched-pair or switchback design.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DesignBootstrap {
    /// Complete refits; at least the floor of 99.
    pub replicates: usize,
    /// Resampling seed.
    pub seed: u64,
    /// Assumption identity of the declared premise that the design's independent units
    /// (subjects, clusters, pairs or sequences) were sampled independently from a larger
    /// population. Required for a randomized design, whose arm means carry no joint law
    /// under randomization alone; stated with a static-regime design when given. The premise
    /// enters the decision program's identity and the evaluation's assumptions.
    pub sampling: Option<String>,
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

/// What one execution of a design produced: the source, the compiled program identity, and the
/// identity the executed result reported, aligned with it.
type DesignExecution = (NativeDecisionSource, Vec<SemanticDigest>, Vec<Option<SemanticDigest>>);

/// A compiled design read as the source of a decision, entering the design's program with the
/// decision attached to layer four.
#[derive(Clone, Debug)]
pub(super) struct DesignSource {
    design: PreparedNativeJointDesign,
    coordinates: Vec<ScientificQuantity>,
    inference: JointDesignInference,
    /// The declared sampling premise the law rests on.
    sampling: Option<String>,
}

impl DesignSource {
    pub(super) fn plan_view(&self) -> PlanView {
        self.design.plan_view()
    }

    /// Execute the design once and read the representation the demand named. Returns the
    /// source, the compiled program's identity and the identity the executed result reports,
    /// refusing a result of another program (`preference_compile.executed_program_mismatch`).
    pub(super) fn execute(
        &self,
        representation: SourceRepresentation,
        ctx: &ExecutionContext,
    ) -> Result<DesignExecution, PreferenceRouteError> {
        let result = self.design.execute(self.inference.clone(), ctx)?;
        let compiled = self.design.program_identity();
        let ran = checked_executed(Some(compiled), Some(result.program_identity()))?;
        let dependency = result.dependency_identity().to_hex();
        if let Some(refits) = result.unit_bootstrap() {
            let rng_id = self.resampling_id(Some(refits.unit))?;
            let law =
                self.refit_law((&refits.realizations, &refits.failures), rng_id, &dependency)?;
            return Ok((NativeDecisionSource::JointLaw(Box::new(law)), vec![compiled], vec![ran]));
        }
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
                let rng_id = self.resampling_id(None)?;
                NativeDecisionSource::JointLaw(Box::new(self.refit_law(
                    (&refits.realizations, &refits.failures),
                    rng_id,
                    &dependency,
                )?))
            }
            _ => return Err(law_refusal("the design returned another law".to_owned())),
        };
        Ok((source, vec![compiled], vec![ran]))
    }

    /// The resampling recipe a refit law names as its `rng_id`: the recipe's kind and a
    /// canonical digest of its kind, resampled unit, refits and seed. `unit` is the unit an
    /// independent-unit bootstrap resampled, and `None` for the whole-subject refits of a
    /// static-regime design. A law that does not match the declared inference is refused.
    fn resampling_id(&self, unit: Option<IndependentUnit>) -> Result<String, PreferenceRouteError> {
        resampling_recipe(&self.inference, unit)
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

    /// Add the declared sampling premise to the assumptions the evaluation rests on.
    pub(super) fn state_premise(&self, evaluation: &mut CompiledPreferenceEvaluation) {
        let Some(premise) = &self.sampling else {
            return;
        };
        let assumptions = match evaluation {
            CompiledPreferenceEvaluation::Single(PreferenceEvaluation::Exact(result)) => {
                &mut result.assumptions
            }
            CompiledPreferenceEvaluation::Single(PreferenceEvaluation::Certified(result)) => {
                &mut result.assumptions
            }
            CompiledPreferenceEvaluation::Ordinary(result) => &mut result.assumptions,
            CompiledPreferenceEvaluation::Structures(_)
            | CompiledPreferenceEvaluation::Sampled(_) => return,
        };
        assumptions.push(premise.clone());
        assumptions.sort();
        assumptions.dedup();
    }

    /// The aligned refits as one law over the arm coordinates: one row per refit, treated
    /// arm then control arm, labelled with the resampling recipe `rng_id`. A failed refit is
    /// refused, never dropped.
    fn refit_law(
        &self,
        (realizations, failures): (&[Option<Vec<f64>>], &[Option<String>]),
        rng_id: String,
        dependency: &str,
    ) -> Result<DistributionArtifact, PreferenceRouteError> {
        let failed = failures.iter().filter(|failure| failure.is_some()).count();
        if failed > 0 {
            return Err(law_refusal(format!(
                "{failed} of {} refits failed; a failed slot is never dropped",
                realizations.len()
            )));
        }
        let width = self.coordinates.len();
        let mut rows = Vec::with_capacity(realizations.len() * width);
        for realization in realizations {
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
                rng_id,
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
                shape: [realizations.len(), width],
                weights: None,
                supported: Some(refit_support(&rows, width)),
                calibration: REFIT_LAW_CALIBRATION,
                trust: REFIT_LAW_TRUST,
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

/// The refusal of a joint-law demand on a randomized design with no declared sampling
/// premise: under randomization alone the cross-arm dependence of the arm-mean estimators is
/// not identified.
fn sampling_undeclared(design: &PreparedNativeJointDesign) -> PreferenceRouteError {
    ExternalRefusal::new(
        antecedent_core::reason_code!("joint_law_required"),
        STAGE_COMPILE,
        preference_compile::DESIGN_SAMPLING_UNDECLARED,
    )
    .with_offending(design.identification().method.clone())
    .with_expected(
        "a declared premise that the independent units (clusters, pairs or sequences) were \
         sampled independently from a larger population, with at least 99 refits",
    )
    .with_supplied(
        "arm means of a randomized design; under randomization alone the cross-arm dependence \
         of the arm-mean estimators is not identified",
    )
    .with_remedy(
        "declare the sampling premise and the refits to read the independent-unit bootstrap \
         law, or decide with a value affine in the outcomes and an expectation criterion, \
         which the arm means answer",
    )
    .into()
}

/// The refusal of an independent-unit bootstrap over fewer units than its floor.
fn units_too_few(
    design: &PreparedNativeJointDesign,
    unit: IndependentUnit,
    units: usize,
) -> PreferenceRouteError {
    let stratum = if unit == IndependentUnit::Cluster { " in each arm" } else { "" };
    ExternalRefusal::new(
        antecedent_core::reason_code!("too_few_clusters"),
        STAGE_COMPILE,
        preference_compile::DESIGN_UNITS_TOO_FEW,
    )
    .with_offending(design.identification().method.clone())
    .with_expected(format!("at least {MIN_INDEPENDENT_UNITS} {}s{stratum}", unit.name()))
    .with_supplied(format!("{units} {}s{stratum}", unit.name()))
    .with_remedy(
        "decide with a value affine in the outcomes and an expectation criterion, which the \
         arm means answer, or collect more independent units",
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
    contract: &CompiledContract,
    coordinates: &[ScientificQuantity],
) -> Result<Vec<DemandedQuantity>, PreferenceRouteError> {
    let mut demanded: Vec<DemandedQuantity> = Vec::new();
    for (action, attribute, quantity) in quantity_inputs(contract) {
        let Some(coordinate) =
            coordinates.iter().find(|c| quantity.require_same_coordinate(c).is_ok())
        else {
            return Err(coordinate_refusal(coordinates, (action, attribute.as_str()), quantity));
        };
        let read = (action.to_owned(), attribute);
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

/// The fact stating a design's resampling law.
fn design_law_fact(
    (kind, unit): (&str, &str),
    replicates: usize,
    seed: u64,
    sampling: Option<&str>,
) -> PreferenceDesignLawFact {
    PreferenceDesignLawFact {
        kind: Arc::from(kind),
        unit: Arc::from(unit),
        replicates,
        seed,
        sampling: sampling.map(Arc::from),
    }
}

/// The inference that supplies the representation the demand names, the carrier the
/// decision demands of the design's program, and the resampling law it states.
fn design_inference(
    design: &PreparedNativeJointDesign,
    kind: DesignKind,
    representation: SourceRepresentation,
    bootstrap: Option<DesignBootstrap>,
) -> Result<
    (JointDesignInference, CarrierDemand, Option<PreferenceDesignLawFact>),
    PreferenceRouteError,
> {
    if representation == SourceRepresentation::Mean {
        return Ok((JointDesignInference::Frequentist, CarrierDemand::Mean, None));
    }
    match (kind, bootstrap) {
        (DesignKind::StaticRegime, Some(refits)) if refits.replicates >= MIN_REPLICATES => {
            let DesignBootstrap { replicates, seed, sampling } = refits;
            Ok((
                JointDesignInference::StaticRegimeBootstrap { replicates, seed },
                CarrierDemand::DecisionDraws,
                Some(design_law_fact(
                    ("static_regime_refits", "subject"),
                    replicates,
                    seed,
                    sampling.as_deref(),
                )),
            ))
        }
        (DesignKind::StaticRegime, Some(_)) => {
            Err(unpaired("fewer whole-subject refits than the design's floor of 99"))
        }
        (DesignKind::StaticRegime, None) => {
            Err(unpaired("the design's point law carries arm means only"))
        }
        (
            DesignKind::Cluster | DesignKind::Experiment,
            Some(DesignBootstrap { replicates, seed, sampling: Some(sampling) }),
        ) if !sampling.trim().is_empty() => {
            let (unit, units) =
                independent_units(design.recipe()).ok_or_else(|| unsupported_design(design))?;
            if units < MIN_INDEPENDENT_UNITS {
                return Err(units_too_few(design, unit, units));
            }
            if replicates < MIN_REFITS {
                return Err(unpaired("fewer independent-unit refits than the floor of 99"));
            }
            Ok((
                JointDesignInference::IndependentUnitBootstrap {
                    replicates,
                    seed,
                    sampling: sampling.clone(),
                },
                CarrierDemand::DecisionDraws,
                Some(design_law_fact(
                    ("independent_unit_bootstrap", unit.name()),
                    replicates,
                    seed,
                    Some(&sampling),
                )),
            ))
        }
        (DesignKind::Cluster | DesignKind::Experiment, _) => Err(sampling_undeclared(design)),
    }
}

/// Check, at compile, that the design's realization supplies the source the decision program
/// demands: a mean of an affine value from the design's exact point functional, or aligned
/// draws from its whole-subject or independent-unit refits.
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
                && program.utility().means_suffice()
                && matches!(inference, JointDesignInference::Frequentist)
        }
        CarrierDemand::DecisionDraws => matches!(
            inference,
            JointDesignInference::StaticRegimeBootstrap { .. }
                | JointDesignInference::IndependentUnitBootstrap { .. }
        ),
        CarrierDemand::ParameterPosterior
        | CarrierDemand::OutcomeLaw
        | CarrierDemand::PredictiveLaw
        | CarrierDemand::IdentifiedSet => false,
    };
    if supplied {
        Ok(())
    } else {
        Err(unpaired("the design's realization does not supply the carrier the decision demands"))
    }
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
    /// answers; otherwise `bootstrap` declares the refits that supply one law of every
    /// outcome under both arms: a static-regime design's aligned whole-subject refits, or,
    /// under a declared sampling premise, a randomized design's independent-unit bootstrap.
    /// The resampling law is stated in the decision program and enters its identity. Nothing
    /// is executed.
    ///
    /// # Errors
    /// A design that is not a static regime, a cluster-randomized design or a matched-pair or
    /// switchback experiment (`preference_compile.design_not_supported`); a randomized design
    /// whose exposure declares bias bounds (`preference_compile.design_exposure_bounded`); an
    /// invalid contract; a quantity that is not one of the design's arm coordinates (naming
    /// the action and the attribute); a contract that needs a joint law of a cluster or
    /// experiment design with no sampling premise declared
    /// (`preference_compile.design_sampling_undeclared`) or with fewer independent units than
    /// the bootstrap's floor (`preference_compile.design_units_too_few`); a contract that
    /// needs a joint law with no refits declared, or fewer refits than the floor of 99
    /// (`joint_law_required`).
    pub fn compile_for_preference_decision(
        &self,
        contract: &PreferenceContract,
        bootstrap: Option<DesignBootstrap>,
        ctx: &ExecutionContext,
    ) -> Result<CompiledPreferenceDecision, PreferenceRouteError> {
        self.compile_design_decision(CompiledContract::Tradeoff(contract.clone()), bootstrap, ctx)
    }

    /// Compile an ordinary decision contract whose quantities are the arm means of this
    /// compiled design's outcomes: the same binding, carrier check and lowering into layer
    /// four as [`Self::compile_for_preference_decision`], each action reading its own arm
    /// coordinates through its own utility. Executed, the decision is the contract's own
    /// evaluation on the source the design supplies, with its result and contract identity.
    ///
    /// # Errors
    /// As [`Self::compile_for_preference_decision`], a quantity being named by its action and
    /// input position (`input k`); a criterion that reads structure or identified-set inputs
    /// (`decision_evaluation.structure_inputs_required`).
    pub fn compile_for_contract_decision(
        &self,
        contract: &DecisionContract,
        bootstrap: Option<DesignBootstrap>,
        ctx: &ExecutionContext,
    ) -> Result<CompiledDecision, PreferenceRouteError> {
        self.compile_design_decision(CompiledContract::Ordinary(contract.clone()), bootstrap, ctx)
    }

    fn compile_design_decision(
        &self,
        contract: CompiledContract,
        bootstrap: Option<DesignBootstrap>,
        ctx: &ExecutionContext,
    ) -> Result<CompiledPreferenceDecision, PreferenceRouteError> {
        crate::analysis::lowering::refuse_reentry(ctx)?;
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: super::CANCELLED_STAGE }.into());
        }
        let contract_identity = contract.identity()?;
        let (coordinates, kind) = design_coordinates(self)?;
        let quantities = bind_design(&contract, &coordinates)?;
        let (mut demand, _) = demand_of(&contract, quantities, false)?;
        let (inference, carrier, design_law) =
            design_inference(self, kind, demand.representation, bootstrap)?;
        let sampling = design_law.as_ref().and_then(|law| law.sampling().map(str::to_owned));
        demand.carrier = Some(carrier);
        let causal = vec![(self.program_identity().to_hex(), self.dependency().to_hex())];
        let identity = bound_identity(&contract_identity, &causal);
        let read: Vec<&DemandedQuantity> = demand.quantities.iter().collect();
        let stated = super::facts::value_facts(&contract, &demand, &read, None, None, design_law);
        let program = decision_program(&contract, &identity, stated, (&demand, carrier));
        require_design_carrier(self, &program, &inference)?;
        let lowered = self.lowered().with_decision(program.clone()).ok_or_else(|| {
            PreferenceRouteError::from(
                ExternalRefusal::new(
                    antecedent_core::reason_code!("invalid_argument"),
                    STAGE_COMPILE,
                    preference_compile::PROGRAM_CARRIES_DECISION,
                )
                .with_offending(self.program_identity().to_hex()),
            )
        })?;
        let design = self.with_lowered(Arc::new(lowered));
        // The one child is the design's program with the decision in its layer four, the
        // program the execution runs.
        let children = [Some(design.program_identity())];
        let parent = parent_of(
            &contract,
            (&demand, program),
            (identity.as_str(), contract_identity.as_str()),
            (children.as_slice(), &[] as &[f64]),
            Composition::Single,
        )?;
        Ok(CompiledPreferenceDecision {
            contract,
            contract_identity,
            demand,
            identity,
            causal,
            dependence: OutcomeDependence::Undeclared,
            sources: Sources::Design(Box::new(DesignSource {
                design,
                coordinates,
                inference,
                sampling,
            })),
            shared: Arc::from(Vec::new()),
            joint: None,
            parent,
        })
    }
}

/// The coordinate of one outcome of a design under one arm.
fn arm_coordinate(
    variable_id: &str,
    name: &str,
    units: &str,
    population: &str,
    regime: &str,
    horizon: u32,
) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable_id.to_owned(),
        variable_name: name.to_owned(),
        role: QuantityRole::Outcome,
        units: units.to_owned(),
        population_id: population.to_owned(),
        regime_id: regime.to_owned(),
        horizon,
        functional_id: MEAN_FUNCTIONAL.to_owned(),
        conditioning: Vec::new(),
        transform_id: IDENTITY_TRANSFORM.to_owned(),
    }
}

/// The support a refitted design law states for each arm coordinate, from the refits
/// themselves: an arm is supported only when every refit of the study's own data produced a
/// finite estimate for it. Failed refits are refused before the law is built, so the mask is
/// the evidence the refits hold, never a default: a law without a stated mask would read as
/// missing evidence wherever a decision input reads it.
fn refit_support(rows: &[f64], width: usize) -> Vec<bool> {
    (0..width)
        .map(|column| {
            !rows.is_empty() && rows.chunks_exact(width).all(|row| row[column].is_finite())
        })
        .collect()
}

/// A refitted design law states no coverage record: its draws are realizations of the
/// compiled design, not a measured interval construction.
const REFIT_LAW_CALIBRATION: DistributionCalibration = DistributionCalibration::NotMeasured;
/// A refitted design law is the library's own computation, asserted by no verified provider.
const REFIT_LAW_TRUST: DistributionTrust = DistributionTrust::Unverified;

#[cfg(test)]
mod standing_tests {
    use super::*;

    #[test]
    fn site_a_refitted_design_law_claims_no_calibration_or_verified_trust() {
        assert_eq!(REFIT_LAW_CALIBRATION, DistributionCalibration::NotMeasured);
        assert_ne!(REFIT_LAW_CALIBRATION, DistributionCalibration::Measured);
        assert_ne!(REFIT_LAW_CALIBRATION, DistributionCalibration::Exact);
        assert_eq!(REFIT_LAW_TRUST, DistributionTrust::Unverified);
    }

    #[test]
    fn site_a_refitted_design_law_states_support_from_its_refits() {
        // Every refit estimated both arms: both are supported, stated, not defaulted.
        assert_eq!(refit_support(&[1.0, 2.0, 1.5, 2.5], 2), vec![true, true]);
        // An arm some refit could not estimate is not supported; the other arm keeps its own.
        assert_eq!(refit_support(&[1.0, f64::NAN, 1.5, 2.5], 2), vec![true, false]);
        assert_eq!(refit_support(&[1.0, 2.0, f64::INFINITY, 2.5], 2), vec![false, true]);
        // No refit is no evidence.
        assert_eq!(refit_support(&[], 2), vec![false, false]);
    }
}

/// The resampling recipe a refit law of `inference` names as its `rng_id`, `unit` being the
/// unit it resampled (`None` for whole-subject refits): see [`DesignSource::resampling_id`].
fn resampling_recipe(
    inference: &JointDesignInference,
    unit: Option<IndependentUnit>,
) -> Result<String, PreferenceRouteError> {
    let mut writer = CanonicalWriter::new(RESAMPLING_TAG);
    let kind = match (inference, unit) {
        (JointDesignInference::StaticRegimeBootstrap { replicates, seed }, None) => {
            writer.text(STATIC_REGIME_BOOTSTRAP).word(0).word(*replicates as u64).word(*seed);
            STATIC_REGIME_BOOTSTRAP
        }
        (
            JointDesignInference::IndependentUnitBootstrap { replicates, seed, sampling: _ },
            Some(unit),
        ) => {
            writer
                .text(INDEPENDENT_UNIT_BOOTSTRAP)
                .word(1)
                .text(unit.name())
                .word(*replicates as u64)
                .word(*seed);
            INDEPENDENT_UNIT_BOOTSTRAP
        }
        // Any other pairing is a law the declared inference does not draw.
        _ => return Err(law_refusal("the design returned another law".to_owned())),
    };
    Ok(format!("{kind}:{}", writer.finish_hex()))
}

#[cfg(test)]
mod site_tests {
    use super::*;
    use crate::identity_laws::all_distinct;

    /// `resampling_recipe`: the recipe's kind, its resampled unit, refits and seed each enter
    /// in a fixed, framed position; the sampling premise does not change the refits, so it
    /// enters the decision's identity, not the recipe.
    #[test]
    fn site_design_resampling_recipe_names_kind_unit_refits_and_seed() {
        let regime =
            |replicates, seed| JointDesignInference::StaticRegimeBootstrap { replicates, seed };
        let units =
            |replicates, seed, sampling: &str| JointDesignInference::IndependentUnitBootstrap {
                replicates,
                seed,
                sampling: sampling.to_owned(),
            };
        let recipe = |inference: &JointDesignInference, unit| {
            resampling_recipe(inference, unit).expect("a recipe of the declared inference")
        };
        all_distinct(&[
            recipe(&regime(199, 7), None),
            recipe(&regime(200, 7), None),
            recipe(&regime(199, 8), None),
            recipe(&units(199, 7, "a"), Some(IndependentUnit::Cluster)),
            recipe(&units(199, 7, "a"), Some(IndependentUnit::Pair)),
            recipe(&units(199, 7, "a"), Some(IndependentUnit::Sequence)),
            recipe(&units(200, 7, "a"), Some(IndependentUnit::Pair)),
            recipe(&units(199, 8, "a"), Some(IndependentUnit::Pair)),
        ]);
        assert_eq!(
            recipe(&units(199, 7, "a"), Some(IndependentUnit::Pair)),
            recipe(&units(199, 7, "b"), Some(IndependentUnit::Pair)),
        );
        assert!(resampling_recipe(&regime(199, 7), Some(IndependentUnit::Pair)).is_err());
        assert!(resampling_recipe(&units(199, 7, "a"), None).is_err());
    }
}
