//! The six bridge kinds, the classes of quantity they connect, the assumption
//! each kind rests on, and the value kinds a bridge never interconverts.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::mapping::MappingKind;
use antecedent_core::named_variants;

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

named_variants! {
    QuantityClass {
        LatentConstruct => "latent_construct",
        MeasuredOutcome => "measured_outcome",
        StatedPreference => "stated_preference",
        RealizedBehavior => "realized_behavior",
        SurrogateEndpoint => "surrogate_endpoint",
        TerminalOutcome => "terminal_outcome",
        PredictiveQuantity => "predictive_quantity",
        PredictiveImportance => "predictive_importance",
        ModelScore => "model_score",
        CausalQuantity => "causal_quantity",
        ModelParameter => "model_parameter",
        InterventionResponse => "intervention_response",
        ScientificEndpoint => "scientific_endpoint",
        DecisionUtility => "decision_utility",
    }
}

impl QuantityClass {
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

named_variants! {
    ValueKind {
        Mean => "mean",
        PredictiveLaw => "predictive_law",
        Posterior => "posterior",
        IdentifiedSet => "identified_set",
        JointOutcomeLaw => "joint_outcome_law",
    }
}

impl ValueKind {
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

named_variants! {
    AssumptionKind {
        MeasurementModel => "measurement_model",
        BehavioralLinkage => "behavioral_linkage",
        SurrogateValidation => "surrogate_validation",
        Identification => "identification",
        ParameterInterpretation => "parameter_interpretation",
        UtilityMapping => "utility_mapping",
        Additional => "additional",
    }
}

impl AssumptionKind {
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

named_variants! {
    BridgeKind {
        LatentMeasuredOutcome => "latent_measured_outcome",
        PreferenceRealizedBehavior => "preference_realized_behavior",
        SurrogateTerminalOutcome => "surrogate_terminal_outcome",
        PredictiveCausalQuantity => "predictive_causal_quantity",
        ParameterInterventionResponse => "parameter_intervention_response",
        ScientificDecisionUtility => "scientific_decision_utility",
    }
}

impl BridgeKind {
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
