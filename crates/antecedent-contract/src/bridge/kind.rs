//! The six bridge kinds, the classes of quantity they connect, the assumption
//! each kind rests on, and the value kinds a bridge never interconverts.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::mapping::MappingKind;

/// What a scientific quantity is, for the purpose of a bridge.
///
/// A [`crate::ScientificQuantity`] says which variable, unit, population,
/// regime and functional a value is about. It does not say whether the value
/// is a construct, a recorded proxy, a stated preference or a prediction; the
/// class does, and a bridge connects two classes.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum QuantityClass {
    /// The scientific construct itself, never recorded directly.
    LatentConstruct,
    /// A recorded value that differs from the construct by error or
    /// misclassification.
    MeasuredOutcome,
    /// A stated, elicited or modelled preference, intention or valuation.
    StatedPreference,
    /// Behavior that was realized (a choice made, a dose taken, an outcome
    /// that occurred).
    RealizedBehavior,
    /// An endpoint measured earlier or more cheaply than the outcome it is
    /// offered for.
    SurrogateEndpoint,
    /// The terminal outcome a surrogate is offered for.
    TerminalOutcome,
    /// An associational or predictive functional of a fitted predictive
    /// model (a conditional mean or law given features).
    PredictiveQuantity,
    /// The importance or attribution of a feature in a predictive model.
    PredictiveImportance,
    /// An uncalibrated output of a model (a score, a margin, a rank).
    ModelScore,
    /// A quantity defined under an intervention.
    CausalQuantity,
    /// A parameter of a fitted model.
    ModelParameter,
    /// The response of an outcome to a declared intervention.
    InterventionResponse,
    /// A scientific endpoint or effect on its own natural scale.
    ScientificEndpoint,
    /// The utility a decision maker assigns to an outcome or an effect.
    DecisionUtility,
}

impl QuantityClass {
    /// Every class.
    pub const ALL: [Self; 14] = [
        Self::LatentConstruct,
        Self::MeasuredOutcome,
        Self::StatedPreference,
        Self::RealizedBehavior,
        Self::SurrogateEndpoint,
        Self::TerminalOutcome,
        Self::PredictiveQuantity,
        Self::PredictiveImportance,
        Self::ModelScore,
        Self::CausalQuantity,
        Self::ModelParameter,
        Self::InterventionResponse,
        Self::ScientificEndpoint,
        Self::DecisionUtility,
    ];

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LatentConstruct => "latent_construct",
            Self::MeasuredOutcome => "measured_outcome",
            Self::StatedPreference => "stated_preference",
            Self::RealizedBehavior => "realized_behavior",
            Self::SurrogateEndpoint => "surrogate_endpoint",
            Self::TerminalOutcome => "terminal_outcome",
            Self::PredictiveQuantity => "predictive_quantity",
            Self::PredictiveImportance => "predictive_importance",
            Self::ModelScore => "model_score",
            Self::CausalQuantity => "causal_quantity",
            Self::ModelParameter => "model_parameter",
            Self::InterventionResponse => "intervention_response",
            Self::ScientificEndpoint => "scientific_endpoint",
            Self::DecisionUtility => "decision_utility",
        }
    }

    /// Parse the spelling of [`Self::as_str`].
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|class| class.as_str() == name)
    }

    pub(crate) const fn tag(self) -> u64 {
        match self {
            Self::LatentConstruct => 1,
            Self::MeasuredOutcome => 2,
            Self::StatedPreference => 3,
            Self::RealizedBehavior => 4,
            Self::SurrogateEndpoint => 5,
            Self::TerminalOutcome => 6,
            Self::PredictiveQuantity => 7,
            Self::PredictiveImportance => 8,
            Self::ModelScore => 9,
            Self::CausalQuantity => 10,
            Self::ModelParameter => 11,
            Self::InterventionResponse => 12,
            Self::ScientificEndpoint => 13,
            Self::DecisionUtility => 14,
        }
    }
}

/// The kind of object a value is. They are not interchangeable.
///
/// A mean, a predictive law, a posterior, an identified set and a joint outcome
/// law answer different questions; supplying one where another is required is
/// refused rather than approximated. No bridge relaxes this: a bridge carries
/// a value of one kind to a quantity of the same kind.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum ValueKind {
    /// A mean (an expectation, effect or point functional of one quantity).
    Mean,
    /// A predictive law of one outcome.
    PredictiveLaw,
    /// A posterior over a parameter or a causal functional.
    Posterior,
    /// An identified set (bounds) of a functional.
    IdentifiedSet,
    /// The joint law of several outcomes, with their dependence.
    JointOutcomeLaw,
}

impl ValueKind {
    /// Every kind.
    pub const ALL: [Self; 5] = [
        Self::Mean,
        Self::PredictiveLaw,
        Self::Posterior,
        Self::IdentifiedSet,
        Self::JointOutcomeLaw,
    ];

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Mean => "mean",
            Self::PredictiveLaw => "predictive_law",
            Self::Posterior => "posterior",
            Self::IdentifiedSet => "identified_set",
            Self::JointOutcomeLaw => "joint_outcome_law",
        }
    }

    /// Parse the spelling of [`Self::as_str`].
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.as_str() == name)
    }

    pub(crate) const fn tag(self) -> u64 {
        match self {
            Self::Mean => 1,
            Self::PredictiveLaw => 2,
            Self::Posterior => 3,
            Self::IdentifiedSet => 4,
            Self::JointOutcomeLaw => 5,
        }
    }
}

/// The kind of statement a declared bridge assumption makes.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum AssumptionKind {
    /// A measurement model relates the recorded quantity to the construct.
    /// Names the model identity as evidence.
    MeasurementModel,
    /// Stated and realized behavior are linked: the declared route by which a
    /// stated preference determines the realized choice (no hypothetical
    /// bias, or a declared calibration of it).
    BehavioralLinkage,
    /// The surrogate is validated for the terminal outcome at the trial or
    /// unit level. Names the validation evidence.
    SurrogateValidation,
    /// The causal quantity is identified from the predictive source under a
    /// checked or declared identification (adjustment, exchangeability,
    /// positivity, consistency). Names the identification identity.
    Identification,
    /// The model parameter has a stated structural meaning as the response to
    /// the declared intervention.
    ParameterInterpretation,
    /// A declared utility function maps the endpoint to decision utility.
    /// Names the utility function identity.
    UtilityMapping,
    /// A further declared assumption of this bridge.
    Additional,
}

impl AssumptionKind {
    /// Every kind.
    pub const ALL: [Self; 7] = [
        Self::MeasurementModel,
        Self::BehavioralLinkage,
        Self::SurrogateValidation,
        Self::Identification,
        Self::ParameterInterpretation,
        Self::UtilityMapping,
        Self::Additional,
    ];

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::MeasurementModel => "measurement_model",
            Self::BehavioralLinkage => "behavioral_linkage",
            Self::SurrogateValidation => "surrogate_validation",
            Self::Identification => "identification",
            Self::ParameterInterpretation => "parameter_interpretation",
            Self::UtilityMapping => "utility_mapping",
            Self::Additional => "additional",
        }
    }

    /// Parse the spelling of [`Self::as_str`].
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.as_str() == name)
    }

    /// Whether an assumption of this kind must name the evidence or identity
    /// it rests on.
    #[must_use]
    pub const fn requires_evidence(self) -> bool {
        matches!(
            self,
            Self::MeasurementModel
                | Self::SurrogateValidation
                | Self::Identification
                | Self::UtilityMapping
        )
    }

    pub(crate) const fn tag(self) -> u64 {
        match self {
            Self::MeasurementModel => 1,
            Self::BehavioralLinkage => 2,
            Self::SurrogateValidation => 3,
            Self::Identification => 4,
            Self::ParameterInterpretation => 5,
            Self::UtilityMapping => 6,
            Self::Additional => 7,
        }
    }
}

/// One of the six bridge kinds.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum BridgeKind {
    /// Latent construct and the measured outcome recorded for it, through a
    /// measurement model.
    LatentMeasuredOutcome,
    /// A stated preference or behavioral quantity and realized behavior.
    PreferenceRealizedBehavior,
    /// A surrogate endpoint and the terminal outcome.
    SurrogateTerminalOutcome,
    /// A predictive quantity and the causal quantity it is offered as.
    PredictiveCausalQuantity,
    /// A model parameter and the intervention response it encodes.
    ParameterInterventionResponse,
    /// A scientific endpoint and the decision utility assigned to it.
    ScientificDecisionUtility,
}

impl BridgeKind {
    /// Every kind.
    pub const ALL: [Self; 6] = [
        Self::LatentMeasuredOutcome,
        Self::PreferenceRealizedBehavior,
        Self::SurrogateTerminalOutcome,
        Self::PredictiveCausalQuantity,
        Self::ParameterInterventionResponse,
        Self::ScientificDecisionUtility,
    ];

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LatentMeasuredOutcome => "latent_measured_outcome",
            Self::PreferenceRealizedBehavior => "preference_realized_behavior",
            Self::SurrogateTerminalOutcome => "surrogate_terminal_outcome",
            Self::PredictiveCausalQuantity => "predictive_causal_quantity",
            Self::ParameterInterventionResponse => "parameter_intervention_response",
            Self::ScientificDecisionUtility => "scientific_decision_utility",
        }
    }

    /// Parse the spelling of [`Self::as_str`].
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.as_str() == name)
    }

    pub(crate) const fn tag(self) -> u64 {
        match self {
            Self::LatentMeasuredOutcome => 1,
            Self::PreferenceRealizedBehavior => 2,
            Self::SurrogateTerminalOutcome => 3,
            Self::PredictiveCausalQuantity => 4,
            Self::ParameterInterventionResponse => 5,
            Self::ScientificDecisionUtility => 6,
        }
    }

    /// The `(source, target)` class pairs the kind connects.
    ///
    /// A pair listed here is not thereby licensed: the importance and score
    /// pairs of the predictive kind are listed so the refusal that closes
    /// them is the named one, not a malformed-declaration refusal.
    #[must_use]
    pub const fn endpoint_pairs(self) -> &'static [(QuantityClass, QuantityClass)] {
        use QuantityClass as C;
        match self {
            Self::LatentMeasuredOutcome => &[
                (C::MeasuredOutcome, C::LatentConstruct),
                (C::LatentConstruct, C::MeasuredOutcome),
            ],
            Self::PreferenceRealizedBehavior => &[(C::StatedPreference, C::RealizedBehavior)],
            Self::SurrogateTerminalOutcome => &[(C::SurrogateEndpoint, C::TerminalOutcome)],
            Self::PredictiveCausalQuantity => &[
                (C::PredictiveQuantity, C::CausalQuantity),
                (C::PredictiveImportance, C::CausalQuantity),
                (C::ModelScore, C::PredictiveQuantity),
            ],
            Self::ParameterInterventionResponse => &[(C::ModelParameter, C::InterventionResponse)],
            Self::ScientificDecisionUtility => &[(C::ScientificEndpoint, C::DecisionUtility)],
        }
    }

    /// The assumption the kind rests on. Without it the request is refused
    /// under the named substitution refusal of its class pair.
    #[must_use]
    pub const fn principal_assumption(self) -> AssumptionKind {
        match self {
            Self::LatentMeasuredOutcome => AssumptionKind::MeasurementModel,
            Self::PreferenceRealizedBehavior => AssumptionKind::BehavioralLinkage,
            Self::SurrogateTerminalOutcome => AssumptionKind::SurrogateValidation,
            Self::PredictiveCausalQuantity => AssumptionKind::Identification,
            Self::ParameterInterventionResponse => AssumptionKind::ParameterInterpretation,
            Self::ScientificDecisionUtility => AssumptionKind::UtilityMapping,
        }
    }

    /// The mappings that may carry a value across a bridge of this kind.
    #[must_use]
    pub const fn allowed_mappings(self) -> &'static [MappingKind] {
        match self {
            Self::LatentMeasuredOutcome => &[MappingKind::MeasurementCorrection],
            Self::PreferenceRealizedBehavior => &[MappingKind::Identity, MappingKind::Affine],
            Self::SurrogateTerminalOutcome => &[MappingKind::TrialLevelLinear, MappingKind::Affine],
            Self::PredictiveCausalQuantity => &[MappingKind::Identity],
            Self::ParameterInterventionResponse => &[MappingKind::Affine, MappingKind::Identity],
            Self::ScientificDecisionUtility => &[MappingKind::Affine],
        }
    }

    /// Whether the regime may differ between the source and target quantity.
    ///
    /// A predictive quantity or a fitted parameter is observational or
    /// fitted-regime by construction and its target is an interventional
    /// regime; every other kind keeps the regime.
    #[must_use]
    pub const fn regime_may_change(self) -> bool {
        matches!(self, Self::PredictiveCausalQuantity | Self::ParameterInterventionResponse)
    }
}
