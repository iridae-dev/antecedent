//! One design, several outcomes.
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

use antecedent_contract::{ExternalRefusal, QuantityRole, ScientificQuantity};
use antecedent_core::ExecutionContext;
use antecedent_core::SemanticDigest;
use antecedent_core::reason_details::preference_compile;
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
use antecedent_io::quantity_wire::DistributionMeaningWire;

use super::bind::{bound_identity, coordinate_refusal, quantity_inputs};
use super::outcomes::{OutcomeDependence, law_refusal, unpaired};
use super::{
    CompiledPreferenceDecision, DemandedQuantity, IDENTITY_TRANSFORM, MEAN_FUNCTIONAL,
    PreferenceRouteError, STAGE_BIND, STAGE_COMPILE, Sources, decision_program, demand_of,
};
use crate::CausalError;
use crate::analysis::joint_panel::designs::PreparedNativeJointDesign;
use crate::analysis::native_claims::NativeDecisionSource;
use crate::planner::PlanView;
use crate::planner::lowered::{CarrierDemand, DecisionProgram};
use crate::planner::statistical::Realization;

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
