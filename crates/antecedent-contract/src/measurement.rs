//! Semantic contract for measurement models: what a model licenses, for
//! which population, regime, arm and representation, and what stays refused.
//!
//! The numerical correction is elsewhere; this module decides whether a
//! request *may* use a [`MeasurementModel`] at all, and says which request it
//! refused and what declaration or evidence would license it. Refusals are
//! [`ExternalRefusal`]s under the `measurement_model` detail namespace, each
//! with a registered reason code.
//!
//! The six measurement kinds ([`MeasurementKind`]) are distinct in refusal:
//! a proxy offered as the construct, an observed column given an error model,
//! a learned vector treated as intervenable and a derived column offered as a
//! latent construct are different refusals, not one generic mismatch.
//!
//! The extension of [`crate::QuantityRole`] is deliberately *not* made: a
//! scientific quantity keeps its role, and the measurement kind and the
//! representation-version coordinate live here, beside it, so no exhaustive
//! match over roles in another crate is broken.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use antecedent_core::measurement_model::{
    ArmBinding, DifferentialAxis, ErrorStructure, MeasurementKind, MeasurementModel,
    MeasurementModelError, MeasurementParameters, MeasurementRoute, RepresentationMapping,
    RepresentationVersion,
};

use crate::interface_contract::{
    CalibrationStatus, ContractOperation, DependenceRequirement, EvidenceBasis, InterfaceContract,
    PortFacts,
};
use crate::{
    ExternalCapability, ExternalPosteriorKind, ExternalRefusal, ExternalScientificObject,
    ExternalTrustState, ScientificQuantity,
};

/// A downstream use of a corrected measurement.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum MeasurementOperation {
    /// The coefficient of the latent regressor in a linear outcome model with
    /// declared error-free adjustment covariates.
    LinearAdjustedEffect,
    /// A linear functional of the corrected coefficient vector.
    LinearFunctionalOfCoefficients,
    /// The prevalence of a binary latent condition.
    LatentPrevalence,
    /// A two-by-two risk difference for a misclassified binary outcome.
    RiskDifference,
    /// Outcome means and contrasts by latent category.
    CategoryOutcomeMeans,
    /// Conditioning on the latent construct inside an identification proof.
    ConditionOnConstruct,
    /// An intervention on the representation's coordinates.
    InterveneOnRepresentation,
    /// Use in a population or regime other than the model's binding.
    TransportToOtherPopulation,
    /// A nonlinear outcome model in the corrected regressor.
    NonlinearOutcomeModel,
    /// A confidence or credible interval.
    Interval,
}

impl MeasurementOperation {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LinearAdjustedEffect => "linear_adjusted_effect",
            Self::LinearFunctionalOfCoefficients => "linear_functional_of_coefficients",
            Self::LatentPrevalence => "latent_prevalence",
            Self::RiskDifference => "risk_difference",
            Self::CategoryOutcomeMeans => "category_outcome_means",
            Self::ConditionOnConstruct => "condition_on_construct",
            Self::InterveneOnRepresentation => "intervene_on_representation",
            Self::TransportToOtherPopulation => "transport_to_other_population",
            Self::NonlinearOutcomeModel => "nonlinear_outcome_model",
            Self::Interval => "interval",
        }
    }
}

/// What a route assumes, which operations it may enter, and what stays refused.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MeasurementRouteCard {
    /// The route.
    pub route: MeasurementRoute,
    /// Assumptions the route's correction rests on.
    pub assumptions: &'static [&'static str],
    /// Downstream operations the route may enter.
    pub allowed: &'static [MeasurementOperation],
    /// Operations that stay refused, with what would license each.
    pub refused: &'static [(MeasurementOperation, &'static str)],
}

const CLASSICAL_REFUSED: &[(MeasurementOperation, &str)] = &[
    (
        MeasurementOperation::NonlinearOutcomeModel,
        "regression calibration is exact for a linear outcome model only; a nonlinear model needs its own measurement-error route",
    ),
    (
        MeasurementOperation::ConditionOnConstruct,
        "the corrected coefficient is a statistical functional; conditioning on the construct in an identification proof needs the identification layer to accept the latent variable",
    ),
    (
        MeasurementOperation::InterveneOnRepresentation,
        "an intervention is declared on the construct, never on the recorded indicator",
    ),
    (
        MeasurementOperation::TransportToOtherPopulation,
        "a reliability measured in one population applies to another only under a declared measurement-invariance premise",
    ),
    (
        MeasurementOperation::Interval,
        "no measured coverage for the whole measurement-plus-estimation method; the point and an uncalibrated standard error are reported",
    ),
];

const BINARY_REFUSED: &[(MeasurementOperation, &str)] = &[
    (
        MeasurementOperation::ConditionOnConstruct,
        "per-unit conditioning on the latent class needs a posterior over the unit's class, not a population-level inversion",
    ),
    (
        MeasurementOperation::InterveneOnRepresentation,
        "an intervention is declared on the construct, never on the recorded indicator",
    ),
    (
        MeasurementOperation::TransportToOtherPopulation,
        "sensitivity and specificity measured in one population apply to another only under a declared invariance premise",
    ),
    (
        MeasurementOperation::Interval,
        "no measured coverage for the whole measurement-plus-estimation method; the point and an uncalibrated standard error are reported",
    ),
];

const REPEATED_REFUSED: &[(MeasurementOperation, &str)] = &[
    (
        MeasurementOperation::NonlinearOutcomeModel,
        "the replicate-based variance corrects a linear outcome model only",
    ),
    (
        MeasurementOperation::ConditionOnConstruct,
        "the corrected coefficient is a statistical functional; the identification layer must accept the latent variable",
    ),
    (
        MeasurementOperation::InterveneOnRepresentation,
        "an intervention is declared on the construct, never on the recorded indicator",
    ),
    (
        MeasurementOperation::TransportToOtherPopulation,
        "a replicate variance measured in one population applies to another only under a declared invariance premise",
    ),
    (
        MeasurementOperation::Interval,
        "no measured coverage for the whole measurement-plus-estimation method; the point and an uncalibrated standard error are reported",
    ),
];

const CATEGORICAL_REFUSED: &[(MeasurementOperation, &str)] = &[
    (
        MeasurementOperation::ConditionOnConstruct,
        "per-unit soft assignments are not consumed; a calibrated per-unit posterior is a separate route",
    ),
    (
        MeasurementOperation::InterveneOnRepresentation,
        "an intervention is declared on the construct, never on the assigned label",
    ),
    (
        MeasurementOperation::TransportToOtherPopulation,
        "a confusion matrix measured in one population applies to another only under a declared invariance premise",
    ),
    (
        MeasurementOperation::Interval,
        "no measured coverage for the whole measurement-plus-estimation method; the point and an uncalibrated standard error are reported",
    ),
];

const POSTERIOR_REFUSED: &[(MeasurementOperation, &str)] = &[
    (
        MeasurementOperation::NonlinearOutcomeModel,
        "the posterior mean substitutes exactly for a linear outcome model only",
    ),
    (
        MeasurementOperation::InterveneOnRepresentation,
        "an intervention is declared on the construct, never on the provider's coordinates",
    ),
    (
        MeasurementOperation::TransportToOtherPopulation,
        "the posterior is bound to the population and regime it was computed for",
    ),
    (
        MeasurementOperation::Interval,
        "no measured coverage; the provider's own parameter uncertainty is not propagated by this route",
    ),
];

/// The assumptions, allowed operations and refused operations of a route.
#[must_use]
pub fn route_card(route: MeasurementRoute) -> MeasurementRouteCard {
    match route {
        MeasurementRoute::ClassicalContinuousError => MeasurementRouteCard {
            route,
            assumptions: &[
                "additive error W = X + U with U independent of X, the covariates and the outcome",
                "error variance known, or estimated from an independent validation or replicate sample",
                "linear outcome model in the construct with error-free covariates",
                "model parameters established for the same population, regime and arm as the data",
            ],
            allowed: &[
                MeasurementOperation::LinearAdjustedEffect,
                MeasurementOperation::LinearFunctionalOfCoefficients,
            ],
            refused: CLASSICAL_REFUSED,
        },
        MeasurementRoute::BinaryMisclassification => MeasurementRouteCard {
            route,
            assumptions: &[
                "sensitivity and specificity known, or estimated from an independent validation sample",
                "Youden index above the declared minimum",
                "non-differential error, or an explicit per-arm differential declaration",
                "misclassification of the outcome; exposure misclassification is a separate closed route",
            ],
            allowed: &[
                MeasurementOperation::LatentPrevalence,
                MeasurementOperation::RiskDifference,
            ],
            refused: BINARY_REFUSED,
        },
        MeasurementRoute::RepeatedMeasurements => MeasurementRouteCard {
            route,
            assumptions: &[
                "replicates share one latent value and have independent, equal-variance errors",
                "enough units with at least two replicates to identify the within variance",
                "linear outcome model in the construct with error-free covariates",
            ],
            allowed: &[
                MeasurementOperation::LinearAdjustedEffect,
                MeasurementOperation::LinearFunctionalOfCoefficients,
            ],
            refused: REPEATED_REFUSED,
        },
        MeasurementRoute::ProbabilisticCategorical => MeasurementRouteCard {
            route,
            assumptions: &[
                "assigned label depends on the true category and not on the outcome (non-differential)",
                "declared column-stochastic confusion matrix, exact or from a validation sample",
                "confusion matrix invertible within the declared condition bound",
            ],
            allowed: &[
                MeasurementOperation::LatentPrevalence,
                MeasurementOperation::CategoryOutcomeMeans,
            ],
            refused: CATEGORICAL_REFUSED,
        },
        MeasurementRoute::ExternalLatentPosterior => MeasurementRouteCard {
            route,
            assumptions: &[
                "the provider's posterior is the law of the construct given the indicator data (and the outcome when declared)",
                "the provider object is verified or attested; no native trust is inferred from metadata",
                "linear outcome model in the construct with error-free covariates",
            ],
            allowed: &[
                MeasurementOperation::LinearAdjustedEffect,
                MeasurementOperation::LinearFunctionalOfCoefficients,
            ],
            refused: POSTERIOR_REFUSED,
        },
    }
}

/// A refusal of a measurement request.
pub type MeasurementRefusal = ExternalRefusal;

const STAGE_DECLARE: &str = "declare";
const STAGE_BIND: &str = "bind";

fn invalid(detail: &str) -> MeasurementRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("measurement_model_invalid"),
        STAGE_DECLARE,
        detail,
    )
    .with_remedy("declare a well-formed measurement model for the indicator and its construct")
}

/// Map a model-construction failure to its typed refusal.
#[must_use]
pub fn model_refusal(error: &MeasurementModelError) -> MeasurementRefusal {
    match error {
        MeasurementModelError::BlankField(name) => {
            invalid("measurement_model.blank_field").with_offending(*name)
        }
        MeasurementModelError::ObservedIndicator => ExternalRefusal::new(
            antecedent_core::reason_code!("measurement_model_invalid"),
            STAGE_DECLARE,
            "measurement_model.indicator_observed",
        )
        .with_remedy(
            "an observed column has no error model; declare it noisy or misclassified, or use it directly",
        ),
        MeasurementModelError::ProxyOfferedAsConstruct { offered } => ExternalRefusal::new(
            antecedent_core::reason_code!("proxy_offered_as_construct"),
            STAGE_DECLARE,
            "measurement_model.proxy_as_construct",
        )
        .with_supplied(offered.as_str())
        .with_expected(MeasurementKind::LatentScientific.as_str())
        .with_remedy(
            "name the latent scientific construct and relate the proxy to it by a measurement model",
        ),
        MeasurementModelError::IndicatorKindNotAccepted { route, kind } => {
            invalid("measurement_model.indicator_kind")
                .with_offending(route.as_str())
                .with_supplied(kind.as_str())
        }
        MeasurementModelError::RouteParameterMismatch => {
            invalid("measurement_model.route_parameters")
        }
        MeasurementModelError::InvalidParameter(what) => {
            invalid("measurement_model.parameter").with_offending(*what)
        }
        MeasurementModelError::StructureParameterMismatch => {
            ExternalRefusal::new(
                antecedent_core::reason_code!("differential_error_not_pooled"),
                STAGE_DECLARE,
                "measurement_model.structure_parameters",
            )
            .with_remedy(
                "declare shared parameters as non-differential, or one parameter set per arm as differential by arm",
            )
        }
        _ => invalid("measurement_model.unrecognized"),
    }
}

/// What an analysis asks of a measurement model.
#[derive(Clone, Debug)]
pub struct MeasurementRequest<'a> {
    /// Variable the analysis treats as the construct.
    pub construct_id: &'a str,
    /// Population the analysis is about.
    pub population_id: &'a str,
    /// Regime the analysis is about.
    pub regime_id: &'a str,
    /// Arm the analysis is about; `None` for all arms.
    pub arm_id: Option<&'a str>,
    /// Representation the analysis expects the indicator in.
    pub representation: &'a RepresentationVersion,
    /// Error structure the analysis assumes.
    pub assumed_structure: ErrorStructure,
    /// Operation the corrected value will enter.
    pub operation: MeasurementOperation,
}

/// Check one request against one model.
///
/// `mapping` is the explicit representation mapping the caller supplies when
/// the requested representation differs from the model's.
///
/// # Errors
/// The first failed check, as a coded structured refusal: a malformed model; a
/// proxy offered as the construct; a different population, regime or arm; a
/// representation change without an explicit mapping; a non-differential
/// analysis of a differential model (or the reverse); an operation the route
/// does not license.
#[allow(
    clippy::result_large_err,
    reason = "a refusal is the cold path of a once-per-call check; boxing the error would not pay"
)]
pub fn check_measurement_request(
    model: &MeasurementModel,
    request: &MeasurementRequest<'_>,
    mapping: Option<&RepresentationMapping>,
) -> Result<(), MeasurementRefusal> {
    model.validate().map_err(|error| model_refusal(&error))?;

    if request.construct_id == model.indicator.id {
        return Err(ExternalRefusal::new(
            antecedent_core::reason_code!("proxy_offered_as_construct"),
            STAGE_BIND,
            "measurement_model.indicator_as_construct",
        )
        .with_offending(request.construct_id)
        .with_expected(model.construct.id.clone())
        .with_supplied(model.indicator.id.clone())
        .with_remedy("request the construct by its identity; the measured column is its proxy"));
    }
    if request.construct_id != model.construct.id {
        return Err(binding_mismatch(
            "measurement_model.construct_mismatch",
            &model.construct.id,
            request.construct_id,
        ));
    }
    if request.population_id != model.binding.population_id {
        return Err(binding_mismatch(
            "measurement_model.population_mismatch",
            &model.binding.population_id,
            request.population_id,
        ));
    }
    if request.regime_id != model.binding.regime_id {
        return Err(binding_mismatch(
            "measurement_model.regime_mismatch",
            &model.binding.regime_id,
            request.regime_id,
        ));
    }
    if let ArmBinding::Arm(bound) = &model.binding.arm {
        if request.arm_id != Some(bound.as_str()) {
            return Err(binding_mismatch(
                "measurement_model.arm_mismatch",
                bound,
                request.arm_id.unwrap_or("all_arms"),
            ));
        }
    }
    check_representation_transfer(
        &model.representation,
        request.representation,
        mapping,
        request.operation == MeasurementOperation::InterveneOnRepresentation,
        model.indicator.kind,
    )?;
    check_structure(model.structure, request.assumed_structure)?;
    check_operation(model.route, request.operation)
}

fn binding_mismatch(detail: &str, expected: &str, supplied: &str) -> MeasurementRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("measurement_binding_mismatch"),
        STAGE_BIND,
        detail,
    )
    .with_expected(expected)
    .with_supplied(supplied)
    .with_remedy(
        "establish the measurement model for this population, regime and arm, or declare the invariance premise under a transport contract",
    )
}

/// Check a representation change.
///
/// A representation with the same coordinate system and version needs no
/// mapping. Any change needs an explicit [`RepresentationMapping`] whose ends
/// are exactly the two versions; intervention meaning crosses only a mapping
/// that declares it. Coordinates of a learned vector representation carry no
/// intervention meaning of their own.
///
/// # Errors
/// A missing or mismatched mapping, or intervention meaning requested through
/// a mapping that does not carry it.
#[allow(
    clippy::result_large_err,
    reason = "a refusal is the cold path of a once-per-call check; boxing the error would not pay"
)]
pub fn check_representation_transfer(
    model_representation: &RepresentationVersion,
    requested: &RepresentationVersion,
    mapping: Option<&RepresentationMapping>,
    requires_intervention_meaning: bool,
    indicator_kind: MeasurementKind,
) -> Result<(), MeasurementRefusal> {
    let unmapped = |detail: &str| {
        ExternalRefusal::new(
            antecedent_core::reason_code!("representation_mapping_required"),
            STAGE_BIND,
            detail,
        )
        .with_expected(format!(
            "{}@{}",
            model_representation.coordinate_system_id, model_representation.version
        ))
        .with_supplied(format!("{}@{}", requested.coordinate_system_id, requested.version))
    };
    if model_representation == requested {
        if requires_intervention_meaning
            && indicator_kind == MeasurementKind::LearnedVectorRepresentation
        {
            return Err(unmapped("measurement_model.learned_coordinates_not_intervenable")
                .with_remedy(
                    "declare the intervention on the scientific construct; coordinates of a learned representation have no intervention meaning of their own",
                ));
        }
        return Ok(());
    }
    let Some(mapping) = mapping else {
        return Err(unmapped("measurement_model.representation_unmapped").with_remedy(
            "supply an explicit mapping between the two representation versions; a changed coordinate system or version is a different quantity",
        ));
    };
    if &mapping.from != model_representation || &mapping.to != requested {
        return Err(unmapped("measurement_model.mapping_endpoints").with_remedy(
            "the mapping must run exactly from the model's representation to the requested one",
        ));
    }
    if requires_intervention_meaning && !mapping.carries_intervention_meaning() {
        return Err(unmapped("measurement_model.intervention_meaning_not_carried")
            .with_remedy(
                "a relabeling carries values only; declare evidence that interventions correspond across the mapping",
            ));
    }
    Ok(())
}

#[allow(
    clippy::result_large_err,
    reason = "a refusal is the cold path of a once-per-call check; boxing the error would not pay"
)]
fn check_structure(
    declared: ErrorStructure,
    assumed: ErrorStructure,
) -> Result<(), MeasurementRefusal> {
    if declared == assumed {
        if let ErrorStructure::Differential(axis) = declared {
            if axis != DifferentialAxis::Arm {
                return Err(ExternalRefusal::new(
                    antecedent_core::reason_code!("differential_error_not_pooled"),
                    STAGE_BIND,
                    "measurement_model.differential_axis_unsupported",
                )
                .with_offending(axis.as_str())
                .with_remedy(
                    "only error that differs by arm has a bounded route; other axes stay refused rather than pooled",
                ));
            }
        }
        return Ok(());
    }
    let (detail, remedy) = match declared {
        ErrorStructure::Differential(_) => (
            "measurement_model.differential_declared",
            "the model is declared differential; analyse it with its per-arm parameters, not the non-differential route",
        ),
        _ => (
            "measurement_model.differential_undeclared",
            "the analysis assumes error that differs; declare that structure with its own parameters",
        ),
    };
    Err(ExternalRefusal::new(
        antecedent_core::reason_code!("differential_error_not_pooled"),
        STAGE_BIND,
        detail,
    )
    .with_remedy(remedy))
}

#[allow(
    clippy::result_large_err,
    reason = "a refusal is the cold path of a once-per-call check; boxing the error would not pay"
)]
fn check_operation(
    route: MeasurementRoute,
    operation: MeasurementOperation,
) -> Result<(), MeasurementRefusal> {
    let card = route_card(route);
    if card.allowed.contains(&operation) {
        return Ok(());
    }
    let why = card
        .refused
        .iter()
        .find(|(refused, _)| *refused == operation)
        .map_or("not an operation this route enters", |(_, why)| *why);
    Err(ExternalRefusal::new(
        antecedent_core::reason_code!("cell_not_licensed"),
        STAGE_BIND,
        "measurement_model.operation_not_licensed",
    )
    .with_offending(operation.as_str())
    .with_supplied(route.as_str())
    .with_remedy(why))
}

/// Name of the consumed port of a measurement model's interface contract.
pub const MEASUREMENT_INDICATOR_PORT: &str = "indicator";
/// Name of the produced port of a measurement model's interface contract.
pub const MEASUREMENT_CONSTRUCT_PORT: &str = "construct";

/// What a measurement model's declaration does not state but its interface
/// contract needs, and what the correction offers.
#[derive(Clone, Debug, PartialEq)]
pub struct MeasurementInterfaceBasis {
    /// Facts of the recorded indicator the correction consumes. Its
    /// measurement models are empty: the indicator is the raw record.
    pub indicator: PortFacts,
    /// Facts of the corrected construct value. The model's identity is added
    /// to its measurement models.
    pub construct: PortFacts,
    /// What the correction consumes of the indicator estimate and offers of
    /// the construct value.
    pub operations: BTreeSet<ContractOperation>,
    /// The calibration the corrected value holds; `None` is unmeasured when it
    /// carries uncertainty and point-only otherwise.
    pub calibration: Option<CalibrationStatus>,
    /// The evidence the corrected value rests on.
    pub evidence: EvidenceBasis,
}

/// A measurement model as one instance of the interface-contract schema: the
/// indicator quantity it consumes and the construct quantity it produces,
/// established under the model's identity, so a consumer that relies on a
/// corrected value states the model it relies on and the one connection check
/// compares them.
///
/// The model must bind the two quantities it is projected over: the
/// indicator and construct identities, and the model's population and regime
/// for both ends. Those refuse under the `measurement_model.*` details that
/// [`check_measurement_request`] uses. An interval offered by a route that
/// does not license one refuses as `measurement_model.operation_not_licensed`.
///
/// # Errors
/// A malformed model, a quantity that is not the model's indicator or
/// construct, a population or regime other than the model's binding, or an
/// operation the route does not license.
#[allow(
    clippy::result_large_err,
    reason = "a refusal is the cold path of a once-per-call check; boxing the error would not pay"
)]
pub fn measurement_interface_contract(
    component: impl Into<String>,
    model: &MeasurementModel,
    indicator: &ScientificQuantity,
    construct: &ScientificQuantity,
    basis: &MeasurementInterfaceBasis,
) -> Result<InterfaceContract, MeasurementRefusal> {
    model.validate().map_err(|error| model_refusal(&error))?;
    if indicator.variable_id != model.indicator.id {
        return Err(binding_mismatch(
            "measurement_model.indicator_mismatch",
            &model.indicator.id,
            &indicator.variable_id,
        ));
    }
    if construct.variable_id != model.construct.id {
        return Err(binding_mismatch(
            "measurement_model.construct_mismatch",
            &model.construct.id,
            &construct.variable_id,
        ));
    }
    for quantity in [indicator, construct] {
        if quantity.population_id != model.binding.population_id {
            return Err(binding_mismatch(
                "measurement_model.population_mismatch",
                &model.binding.population_id,
                &quantity.population_id,
            ));
        }
        if quantity.regime_id != model.binding.regime_id {
            return Err(binding_mismatch(
                "measurement_model.regime_mismatch",
                &model.binding.regime_id,
                &quantity.regime_id,
            ));
        }
    }
    if basis.operations.contains(&ContractOperation::Interval) {
        check_operation(model.route, MeasurementOperation::Interval)?;
    }
    let identity = model.identities().model.to_hex();
    let mut construct_facts = basis.construct.clone();
    construct_facts.measurement_models.insert(identity);
    let produced = construct_facts.produced(
        MEASUREMENT_CONSTRUCT_PORT,
        construct,
        basis.operations.clone(),
        ["transformation"],
        basis.calibration.clone(),
        basis.evidence.clone(),
    );
    let consumed = basis.indicator.consumed(
        MEASUREMENT_INDICATOR_PORT,
        indicator,
        basis.operations.clone(),
        DependenceRequirement::Unconstrained,
    );
    Ok(InterfaceContract {
        component: component.into(),
        produces: vec![produced],
        consumes: vec![consumed],
    })
}

/// Refuse a measured column used as the construct when no measurement model
/// relates them.
///
/// # Errors
/// A noisy, misclassified, assigned or learned-vector indicator, or a derived
/// column, requested as a latent construct without a model.
#[allow(
    clippy::result_large_err,
    reason = "a refusal is the cold path of a once-per-call check; boxing the error would not pay"
)]
pub fn check_unmodelled_use(
    kind: MeasurementKind,
    requested_as_construct: bool,
) -> Result<(), MeasurementRefusal> {
    if !requested_as_construct {
        return Ok(());
    }
    match kind {
        MeasurementKind::Observed | MeasurementKind::LatentScientific => Ok(()),
        other => Err(ExternalRefusal::new(
            antecedent_core::reason_code!("proxy_offered_as_construct"),
            STAGE_BIND,
            "measurement_model.unmodelled_proxy",
        )
        .with_supplied(other.as_str())
        .with_remedy(
            "declare a measurement model relating the measured quantity to the construct, or ask the question about the measured quantity itself",
        )),
    }
}

/// How an external posterior is trusted once it passed the binding checks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum LatentPosteriorTrust {
    /// The supplier asserted it; no numerical check was performed here.
    ExternallyAttested,
    /// The exact provider request passed object-level verification.
    ExactRequestVerified,
}

impl LatentPosteriorTrust {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ExternallyAttested => "externally_attested",
            Self::ExactRequestVerified => "exact_request_verified",
        }
    }
}

/// Check that an external posterior object may supply a model's latent
/// variable, and report how far it is trusted.
///
/// Native licensing is never held by a provider: a state that claims it, and
/// a posterior whose identity, kind, coordinates or draw operation differ
/// from what the model names, refuse.
///
/// # Errors
/// A model of another route, a different provider object or request, a
/// posterior that is not over parameters, a missing construct coordinate in
/// the model's population and regime, a missing sampling operation, or a trust
/// state that is native (or unverified against this object) refuse.
#[allow(
    clippy::result_large_err,
    reason = "a refusal is the cold path of a once-per-call check; boxing the error would not pay"
)]
pub fn check_latent_posterior(
    model: &MeasurementModel,
    object: &ExternalScientificObject,
    trust: &ExternalTrustState,
) -> Result<LatentPosteriorTrust, MeasurementRefusal> {
    model.validate().map_err(|error| model_refusal(&error))?;
    let MeasurementParameters::ExternalPosterior { reference, .. } = &model.parameters else {
        return Err(invalid("measurement_model.route_parameters")
            .with_supplied(model.route.as_str())
            .with_expected(MeasurementRoute::ExternalLatentPosterior.as_str()));
    };
    let posterior_refusal = |detail: &str| {
        ExternalRefusal::new(
            antecedent_core::reason_code!("measurement_posterior_not_trusted"),
            STAGE_BIND,
            detail,
        )
    };
    let identity = object.identity();
    if identity.provider_id != reference.provider_id
        || identity.object_id != reference.object_id
        || identity.version_id != reference.version_id
        || identity.snapshot_id != reference.snapshot_id
        || identity.request_id != reference.request_id
    {
        return Err(posterior_refusal("measurement_model.posterior_identity")
            .with_expected(format!("{}/{}", reference.provider_id, reference.object_id))
            .with_supplied(format!("{}/{}", identity.provider_id, identity.object_id))
            .with_remedy("supply the exact provider object the model names"));
    }
    object.validate().map_err(|_| {
        posterior_refusal("measurement_model.posterior_declaration")
            .with_remedy("the provider declaration is incomplete or inconsistent")
    })?;
    let ExternalScientificObject::Posterior(posterior) = object else {
        return Err(posterior_refusal("measurement_model.posterior_kind")
            .with_remedy("the latent variable is supplied by a posterior object"));
    };
    if posterior.kind != ExternalPosteriorKind::Parameter {
        return Err(posterior_refusal("measurement_model.posterior_kind").with_remedy(
            "supply a posterior over the latent coordinates (a parameter posterior)",
        ));
    }
    if !posterior.capabilities.contains(&ExternalCapability::Sample) {
        return Err(posterior_refusal("measurement_model.posterior_capability")
            .with_capability(ExternalCapability::Sample)
            .with_remedy(
                "the route consumes joint posterior draws; declare the sampling operation",
            ));
    }
    let covers = |quantity: &ScientificQuantity| {
        quantity.variable_id == model.construct.id
            && quantity.population_id == model.binding.population_id
            && quantity.regime_id == model.binding.regime_id
    };
    if !posterior.quantities.iter().any(covers) {
        return Err(posterior_refusal("measurement_model.posterior_quantity")
            .with_expected(format!(
                "{}@{}/{}",
                model.construct.id, model.binding.population_id, model.binding.regime_id
            ))
            .with_remedy(
                "the posterior must declare the construct in the model's population and regime",
            ));
    }
    match trust {
        ExternalTrustState::NativeLicensed => {
            Err(posterior_refusal("measurement_model.posterior_native_trust")
                .with_remedy("a provider object is never natively licensed; attest or verify it"))
        }
        ExternalTrustState::ExternallyAttested { .. } => {
            Ok(LatentPosteriorTrust::ExternallyAttested)
        }
        ExternalTrustState::ExactRequestVerified(_) => {
            if trust.verifies(object) {
                Ok(LatentPosteriorTrust::ExactRequestVerified)
            } else {
                Err(posterior_refusal("measurement_model.posterior_receipt")
                    .with_remedy("the verification receipt does not cover this exact object"))
            }
        }
    }
}

/// Map a numerical refusal from the correction to its coded refusal.
///
/// `kind` is one of `youden`, `reliability`, `confusion`, `replicates`.
#[must_use]
pub fn identification_refusal(kind: &str, observed: f64, bound: f64) -> MeasurementRefusal {
    let (detail, remedy) = match kind {
        "youden" => (
            "measurement_model.youden_too_small",
            "the indicator is not informative enough to invert; a more accurate measurement or a larger declared minimum-evidence study is needed",
        ),
        "reliability" => (
            "measurement_model.reliability_too_small",
            "the error variance leaves too little latent variance; a more reliable measurement or averaged replicates is needed",
        ),
        "confusion" => (
            "measurement_model.confusion_ill_conditioned",
            "the confusion matrix cannot be inverted within the declared condition bound",
        ),
        _ => (
            "measurement_model.insufficient_replicates",
            "replicates on more units are needed to identify the within-unit variance",
        ),
    };
    ExternalRefusal::new(
        antecedent_core::reason_code!("measurement_not_identified"),
        STAGE_BIND,
        detail,
    )
    .with_supplied(format!("{observed}"))
    .with_expected(format!("{bound}"))
    .with_remedy(remedy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::measurement_model::{
        BinaryArmRates, MappingSemantics, MeasurementBinding, MeasurementVariable,
        ParameterEvidence,
    };

    fn model() -> MeasurementModel {
        MeasurementModel {
            model_id: "m".into(),
            route: MeasurementRoute::BinaryMisclassification,
            indicator: MeasurementVariable {
                id: "schema:flag".into(),
                kind: MeasurementKind::NoisyMisclassified,
                units: "binary".into(),
            },
            construct: MeasurementVariable {
                id: "construct:disease".into(),
                kind: MeasurementKind::LatentScientific,
                units: "binary".into(),
            },
            binding: MeasurementBinding {
                population_id: "clinic-a".into(),
                regime_id: "observational".into(),
                arm: ArmBinding::AllArms,
            },
            structure: ErrorStructure::NonDifferential,
            parameters: MeasurementParameters::Binary {
                rates: vec![BinaryArmRates {
                    arm_id: None,
                    sensitivity: 0.9,
                    specificity: 0.8,
                    sensitivity_variance: 0.0,
                    specificity_variance: 0.0,
                }],
                min_youden: 0.1,
            },
            representation: RepresentationVersion {
                coordinate_system_id: "scalar".into(),
                version: 1,
            },
            evidence: ParameterEvidence::Declared { declarer: "lab".into() },
        }
    }

    fn request(rep: &RepresentationVersion) -> MeasurementRequest<'_> {
        MeasurementRequest {
            construct_id: "construct:disease",
            population_id: "clinic-a",
            regime_id: "observational",
            arm_id: None,
            representation: rep,
            assumed_structure: ErrorStructure::NonDifferential,
            operation: MeasurementOperation::LatentPrevalence,
        }
    }

    fn rep(version: u32) -> RepresentationVersion {
        RepresentationVersion { coordinate_system_id: "scalar".into(), version }
    }

    #[test]
    fn matching_request_is_accepted() {
        let r = rep(1);
        assert_eq!(check_measurement_request(&model(), &request(&r), None), Ok(()));
    }

    #[test]
    fn proxy_offered_as_construct_is_its_own_refusal() {
        let r = rep(1);
        let mut req = request(&r);
        req.construct_id = "schema:flag";
        let refusal = check_measurement_request(&model(), &req, None).unwrap_err();
        assert_eq!(refusal.detail, "measurement_model.indicator_as_construct");
        assert_eq!(refusal.code, "proxy_offered_as_construct");
        assert_eq!(refusal.validate(), Ok(()));

        let mut bad = model();
        bad.construct.kind = MeasurementKind::LearnedVectorRepresentation;
        let refusal = check_measurement_request(&bad, &request(&r), None).unwrap_err();
        assert_eq!(refusal.detail, "measurement_model.proxy_as_construct");
        assert_eq!(refusal.supplied.as_deref(), Some("learned_vector_representation"));

        let unmodelled =
            check_unmodelled_use(MeasurementKind::NoisyMisclassified, true).unwrap_err();
        assert_eq!(unmodelled.detail, "measurement_model.unmodelled_proxy");
        assert!(check_unmodelled_use(MeasurementKind::NoisyMisclassified, false).is_ok());
        assert!(check_unmodelled_use(MeasurementKind::LatentScientific, true).is_ok());
    }

    #[test]
    fn population_regime_arm_and_construct_bindings_are_each_checked() {
        let r = rep(1);
        let mut req = request(&r);
        req.population_id = "clinic-b";
        assert_eq!(
            check_measurement_request(&model(), &req, None).unwrap_err().detail,
            "measurement_model.population_mismatch"
        );
        let mut req = request(&r);
        req.regime_id = "do(a=1)";
        assert_eq!(
            check_measurement_request(&model(), &req, None).unwrap_err().detail,
            "measurement_model.regime_mismatch"
        );
        let mut arm_model = model();
        arm_model.binding.arm = ArmBinding::Arm("treated".into());
        let mut req = request(&r);
        req.arm_id = Some("control");
        assert_eq!(
            check_measurement_request(&arm_model, &req, None).unwrap_err().detail,
            "measurement_model.arm_mismatch"
        );
        req.arm_id = Some("treated");
        assert!(check_measurement_request(&arm_model, &req, None).is_ok());
        let mut req = request(&r);
        req.construct_id = "construct:other";
        assert_eq!(
            check_measurement_request(&model(), &req, None).unwrap_err().detail,
            "measurement_model.construct_mismatch"
        );
    }

    #[test]
    fn representation_version_change_needs_an_explicit_mapping() {
        let new = rep(2);
        let refusal = check_measurement_request(&model(), &request(&new), None).unwrap_err();
        assert_eq!(refusal.code, "representation_mapping_required");
        assert_eq!(refusal.detail, "measurement_model.representation_unmapped");

        let mapping = RepresentationMapping {
            from: rep(1),
            to: rep(2),
            mapping_id: "relabel".into(),
            semantics: MappingSemantics::CoordinateRelabeling,
        };
        assert!(check_measurement_request(&model(), &request(&new), Some(&mapping)).is_ok());

        let wrong = RepresentationMapping { from: rep(3), ..mapping.clone() };
        assert_eq!(
            check_measurement_request(&model(), &request(&new), Some(&wrong)).unwrap_err().detail,
            "measurement_model.mapping_endpoints"
        );

        // Intervention meaning does not cross a relabeling.
        let refusal = check_representation_transfer(
            &rep(1),
            &rep(2),
            Some(&mapping),
            true,
            MeasurementKind::NoisyMisclassified,
        )
        .unwrap_err();
        assert_eq!(refusal.detail, "measurement_model.intervention_meaning_not_carried");
        let preserving = RepresentationMapping {
            semantics: MappingSemantics::InterventionPreserving { evidence_id: "exp".into() },
            ..mapping
        };
        assert!(
            check_representation_transfer(
                &rep(1),
                &rep(2),
                Some(&preserving),
                true,
                MeasurementKind::NoisyMisclassified,
            )
            .is_ok()
        );
        // Learned coordinates are not intervenable even at the same version.
        let refusal = check_representation_transfer(
            &rep(1),
            &rep(1),
            None,
            true,
            MeasurementKind::LearnedVectorRepresentation,
        )
        .unwrap_err();
        assert_eq!(refusal.detail, "measurement_model.learned_coordinates_not_intervenable");
    }

    #[test]
    fn differential_and_non_differential_are_never_silently_pooled() {
        let r = rep(1);
        let mut differential = model();
        differential.structure = ErrorStructure::Differential(DifferentialAxis::Arm);
        differential.parameters = MeasurementParameters::Binary {
            rates: vec![
                BinaryArmRates {
                    arm_id: Some("a".into()),
                    sensitivity: 0.9,
                    specificity: 0.8,
                    sensitivity_variance: 0.0,
                    specificity_variance: 0.0,
                },
                BinaryArmRates {
                    arm_id: Some("b".into()),
                    sensitivity: 0.7,
                    specificity: 0.9,
                    sensitivity_variance: 0.0,
                    specificity_variance: 0.0,
                },
            ],
            min_youden: 0.1,
        };
        let refusal = check_measurement_request(&differential, &request(&r), None).unwrap_err();
        assert_eq!(refusal.code, "differential_error_not_pooled");
        assert_eq!(refusal.detail, "measurement_model.differential_declared");
        let mut diff_request = request(&r);
        diff_request.assumed_structure = ErrorStructure::Differential(DifferentialAxis::Arm);
        assert!(check_measurement_request(&differential, &diff_request, None).is_ok());
        let refusal = check_measurement_request(&model(), &diff_request, None).unwrap_err();
        assert_eq!(refusal.detail, "measurement_model.differential_undeclared");
        // Differential by time is representable but has no route.
        let mut by_time = request(&r);
        by_time.assumed_structure = ErrorStructure::Differential(DifferentialAxis::Time);
        let mut time_model = model();
        time_model.structure = ErrorStructure::Differential(DifferentialAxis::Time);
        // The parameter layout check refuses before the axis check.
        assert!(check_measurement_request(&time_model, &by_time, None).is_err());
    }

    #[test]
    fn operations_outside_the_route_are_refused_with_what_would_license_them() {
        let r = rep(1);
        for operation in [
            MeasurementOperation::Interval,
            MeasurementOperation::ConditionOnConstruct,
            MeasurementOperation::InterveneOnRepresentation,
            MeasurementOperation::LinearAdjustedEffect,
        ] {
            let mut req = request(&r);
            req.operation = operation;
            let refusal = check_measurement_request(&model(), &req, None).unwrap_err();
            assert_eq!(refusal.detail, "measurement_model.operation_not_licensed");
            assert_eq!(refusal.code, "cell_not_licensed");
            assert!(refusal.remedy.is_some());
        }
        for route in MeasurementRoute::ALL {
            let card = route_card(route);
            assert_eq!(card.route, route);
            assert!(!card.allowed.is_empty() && !card.assumptions.is_empty());
            assert!(card.refused.iter().any(|(op, _)| *op == MeasurementOperation::Interval));
            for (op, _) in card.refused {
                assert!(!card.allowed.contains(op));
            }
        }
    }

    #[test]
    fn observed_indicator_and_blank_fields_are_distinct_refusals() {
        let mut observed = model();
        observed.indicator.kind = MeasurementKind::Observed;
        assert_eq!(
            model_refusal(&observed.validate().unwrap_err()).detail,
            "measurement_model.indicator_observed"
        );
        let mut blank = model();
        blank.binding.population_id = " ".into();
        let refusal = model_refusal(&blank.validate().unwrap_err());
        assert_eq!(refusal.detail, "measurement_model.blank_field");
        assert_eq!(refusal.offending.as_deref(), Some("binding.population_id"));
    }
}
