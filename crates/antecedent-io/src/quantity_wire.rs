//! Versioned wire coordinates for 2.3 scientific quantities.
//!
//! This wire carries semantics only. It does not turn a numerical payload into
//! a licensed causal estimate or reinterpret a 2.2 posterior artifact.

use antecedent_core::{DistributionMeaning, QuantityCondition, QuantityRole, ScientificQuantity};
use serde::{Deserialize, Serialize};

/// First version of the scientific quantity wire contract.
pub const QUANTITY_WIRE_VERSION: u16 = 1;

/// Portable semantic identity of one scalar quantity.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScientificQuantityWire {
    /// Wire format version.
    pub version: u16,
    /// Stable schema-qualified variable ID.
    pub variable_id: String,
    /// Display name, never the sole identity.
    pub variable_name: String,
    /// Stable role tag.
    pub role: String,
    /// Declared units; no implicit conversion.
    pub units: String,
    /// Population identity.
    pub population_id: String,
    /// Intervention or regime identity.
    pub regime_id: String,
    /// Ordered causal horizon.
    pub horizon: u32,
    /// Functional identity.
    pub functional_id: String,
    /// Canonically ordered conditions.
    pub conditioning: Vec<QuantityConditionWire>,
    /// Scale or transform identity.
    pub transform_id: String,
}

/// A portable condition bound by stable variable and value IDs.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuantityConditionWire {
    /// Stable variable ID.
    pub variable_id: String,
    /// Stable value or event ID.
    pub value_id: String,
}

impl From<&ScientificQuantity> for ScientificQuantityWire {
    fn from(value: &ScientificQuantity) -> Self {
        let role = match value.role {
            QuantityRole::Treatment => "treatment",
            QuantityRole::Outcome => "outcome",
            QuantityRole::Covariate => "covariate",
            QuantityRole::Mediator => "mediator",
            QuantityRole::Selection => "selection",
            QuantityRole::Utility => "utility",
        };
        Self {
            version: QUANTITY_WIRE_VERSION,
            variable_id: value.variable_id.clone(),
            variable_name: value.variable_name.clone(),
            role: role.into(),
            units: value.units.clone(),
            population_id: value.population_id.clone(),
            regime_id: value.regime_id.clone(),
            horizon: value.horizon,
            functional_id: value.functional_id.clone(),
            conditioning: value
                .conditioning
                .iter()
                .map(|condition| QuantityConditionWire {
                    variable_id: condition.variable_id.clone(),
                    value_id: condition.value_id.clone(),
                })
                .collect(),
            transform_id: value.transform_id.clone(),
        }
    }
}

impl TryFrom<ScientificQuantityWire> for ScientificQuantity {
    type Error = &'static str;

    fn try_from(wire: ScientificQuantityWire) -> Result<Self, Self::Error> {
        if wire.version != QUANTITY_WIRE_VERSION {
            return Err("unsupported quantity wire version");
        }
        let role = match wire.role.as_str() {
            "treatment" => QuantityRole::Treatment,
            "outcome" => QuantityRole::Outcome,
            "covariate" => QuantityRole::Covariate,
            "mediator" => QuantityRole::Mediator,
            "selection" => QuantityRole::Selection,
            "utility" => QuantityRole::Utility,
            _ => return Err("unsupported quantity role"),
        };
        let quantity = Self {
            variable_id: wire.variable_id,
            variable_name: wire.variable_name,
            role,
            units: wire.units,
            population_id: wire.population_id,
            regime_id: wire.regime_id,
            horizon: wire.horizon,
            functional_id: wire.functional_id,
            conditioning: wire
                .conditioning
                .into_iter()
                .map(|condition| QuantityCondition {
                    variable_id: condition.variable_id,
                    value_id: condition.value_id,
                })
                .collect(),
            transform_id: wire.transform_id,
        };
        quantity.validate().map_err(|_| "invalid quantity coordinate")?;
        Ok(quantity)
    }
}

/// Stable wire tag for the meaning of distribution draws.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DistributionMeaningWire {
    /// Posterior over model parameters.
    ParameterPosterior,
    /// Posterior over causal functionals.
    CausalFunctionalPosterior,
    /// Observational posterior predictive outcomes.
    PosteriorPredictive,
    /// Interventional posterior predictive outcomes.
    InterventionalPredictive,
    /// Repeated-sampling estimator distribution.
    EstimatorSampling,
    /// Bootstrap distribution conditional on one snapshot.
    Bootstrap,
    /// Empirical observed outcome law.
    EmpiricalOutcome,
}

impl From<DistributionMeaning> for DistributionMeaningWire {
    fn from(meaning: DistributionMeaning) -> Self {
        match meaning {
            DistributionMeaning::ParameterPosterior => Self::ParameterPosterior,
            DistributionMeaning::CausalFunctionalPosterior => Self::CausalFunctionalPosterior,
            DistributionMeaning::PosteriorPredictive => Self::PosteriorPredictive,
            DistributionMeaning::InterventionalPredictive => Self::InterventionalPredictive,
            DistributionMeaning::EstimatorSampling => Self::EstimatorSampling,
            DistributionMeaning::Bootstrap => Self::Bootstrap,
            DistributionMeaning::EmpiricalOutcome => Self::EmpiricalOutcome,
        }
    }
}

impl From<DistributionMeaningWire> for DistributionMeaning {
    fn from(meaning: DistributionMeaningWire) -> Self {
        match meaning {
            DistributionMeaningWire::ParameterPosterior => Self::ParameterPosterior,
            DistributionMeaningWire::CausalFunctionalPosterior => Self::CausalFunctionalPosterior,
            DistributionMeaningWire::PosteriorPredictive => Self::PosteriorPredictive,
            DistributionMeaningWire::InterventionalPredictive => Self::InterventionalPredictive,
            DistributionMeaningWire::EstimatorSampling => Self::EstimatorSampling,
            DistributionMeaningWire::Bootstrap => Self::Bootstrap,
            DistributionMeaningWire::EmpiricalOutcome => Self::EmpiricalOutcome,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantity_wire_preserves_semantics_and_refuses_mutations() {
        let quantity = ScientificQuantity {
            variable_id: "schema:y".into(),
            variable_name: "outcome".into(),
            role: QuantityRole::Outcome,
            units: "kg".into(),
            population_id: "target".into(),
            regime_id: "do(a=1)".into(),
            horizon: 2,
            functional_id: "mean".into(),
            conditioning: vec![],
            transform_id: "identity".into(),
        };
        let wire = ScientificQuantityWire::from(&quantity);
        let bytes = serde_json::to_vec(&wire).unwrap();
        let decoded: ScientificQuantityWire = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(ScientificQuantity::try_from(decoded.clone()), Ok(quantity));
        let mut altered = decoded;
        altered.version = 2;
        assert!(ScientificQuantity::try_from(altered).is_err());
        let mut altered = wire;
        altered.units.clear();
        assert!(ScientificQuantity::try_from(altered).is_err());
        assert!(
            serde_json::from_slice::<ScientificQuantityWire>(br#"{"version":1,"unknown":"field"}"#)
                .is_err()
        );
    }

    #[test]
    fn distribution_meaning_round_trips_without_reinterpretation() {
        let wire = DistributionMeaningWire::from(DistributionMeaning::Bootstrap);
        assert_eq!(serde_json::to_string(&wire).unwrap(), "\"bootstrap\"");
        assert_eq!(DistributionMeaning::from(wire), DistributionMeaning::Bootstrap);
    }
}
