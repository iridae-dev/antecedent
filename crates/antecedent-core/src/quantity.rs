//! Stable semantic coordinates for scientific values and distributions.
//!
//! A display label or array position is never sufficient to identify a causal
//! quantity. These types deliberately make no identification or calibration
//! claim; a producer must bind them to a separately checked causal contract.

/// The causal role of a variable in a quantity.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum QuantityRole {
    /// A treatment or action variable.
    Treatment,
    /// A response or outcome variable.
    Outcome,
    /// A measured covariate.
    Covariate,
    /// A mediator on a specified causal path.
    Mediator,
    /// A selection or observation indicator.
    Selection,
    /// A utility or cost quantity.
    Utility,
}

/// A condition bound by stable variable and value IDs.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct QuantityCondition {
    /// Stable variable identity, not its display label.
    pub variable_id: String,
    /// Stable value or event identity.
    pub value_id: String,
}

/// The full semantic coordinate of one scalar scientific quantity.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ScientificQuantity {
    /// Stable schema-qualified variable identity.
    pub variable_id: String,
    /// Human-readable label, never used alone for matching.
    pub variable_name: String,
    /// Variable's causal role.
    pub role: QuantityRole,
    /// Declared unit identity, including `dimensionless` when appropriate.
    pub units: String,
    /// Stable population identity.
    pub population_id: String,
    /// Exact intervention or regime identity; `observational` is explicit.
    pub regime_id: String,
    /// Number of ordered causal steps to this quantity; zero is static.
    pub horizon: u32,
    /// Functional identity, such as `mean`, `risk`, or `outcome`.
    pub functional_id: String,
    /// Explicit conditioning coordinates in canonical order.
    pub conditioning: Vec<QuantityCondition>,
    /// Declared scale and transform, such as `identity` or `log_odds`.
    pub transform_id: String,
}

/// Why two quantity coordinates cannot be composed directly.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuantityMismatch {
    /// A required semantic identity is empty or conditioning is ambiguous.
    InvalidCoordinate,
    /// The stable variable or role differs.
    Variable,
    /// The units differ; implicit conversion is prohibited.
    Units,
    /// The source and target populations differ.
    Population,
    /// The interventions or regimes differ.
    Regime,
    /// The time horizons differ.
    Horizon,
    /// The causal functionals differ.
    Functional,
    /// The conditioning events differ.
    Conditioning,
    /// The scales or transforms differ.
    Transform,
}

impl ScientificQuantity {
    /// Validate required identities and canonical conditioning before binding.
    ///
    /// # Errors
    /// Returns an error for empty identity fields or duplicate/unsorted conditions.
    pub fn validate(&self) -> Result<(), QuantityMismatch> {
        if [
            &self.variable_id,
            &self.variable_name,
            &self.units,
            &self.population_id,
            &self.regime_id,
            &self.functional_id,
            &self.transform_id,
        ]
        .into_iter()
        .any(|value| value.trim().is_empty())
        {
            return Err(QuantityMismatch::InvalidCoordinate);
        }
        if self.conditioning.iter().any(|condition| {
            condition.variable_id.trim().is_empty() || condition.value_id.trim().is_empty()
        }) || self.conditioning.windows(2).any(|pair| pair[0].variable_id >= pair[1].variable_id)
        {
            return Err(QuantityMismatch::InvalidCoordinate);
        }
        Ok(())
    }

    /// Check whether two coordinates denote the same scientific quantity.
    ///
    /// A different display name is allowed when the stable variable ID agrees.
    /// No unit conversion or causal equivalence is inferred here.
    ///
    /// # Errors
    /// Returns the first mismatched semantic dimension.
    pub fn require_same_coordinate(&self, other: &Self) -> Result<(), QuantityMismatch> {
        self.validate()?;
        other.validate()?;
        if self.variable_id != other.variable_id || self.role != other.role {
            return Err(QuantityMismatch::Variable);
        }
        if self.units != other.units {
            return Err(QuantityMismatch::Units);
        }
        if self.population_id != other.population_id {
            return Err(QuantityMismatch::Population);
        }
        if self.regime_id != other.regime_id {
            return Err(QuantityMismatch::Regime);
        }
        if self.horizon != other.horizon {
            return Err(QuantityMismatch::Horizon);
        }
        if self.functional_id != other.functional_id {
            return Err(QuantityMismatch::Functional);
        }
        if self.conditioning != other.conditioning {
            return Err(QuantityMismatch::Conditioning);
        }
        if self.transform_id != other.transform_id {
            return Err(QuantityMismatch::Transform);
        }
        Ok(())
    }
}

/// What the randomness in a distribution represents.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DistributionMeaning {
    /// Posterior uncertainty in model parameters.
    ParameterPosterior,
    /// Posterior uncertainty in a causal functional.
    CausalFunctionalPosterior,
    /// Posterior prediction of observed outcomes.
    PosteriorPredictive,
    /// Posterior prediction of outcomes under an intervention.
    InterventionalPredictive,
    /// Repeated-sampling distribution of an estimator.
    EstimatorSampling,
    /// Resampling distribution conditional on one data snapshot.
    Bootstrap,
    /// Empirical distribution of observed outcomes.
    EmpiricalOutcome,
}

impl DistributionMeaning {
    /// Whether draws can directly answer an interventional outcome threshold.
    /// A posterior over a mean effect cannot stand in for outcome draws.
    #[must_use]
    pub const fn answers_interventional_outcome_threshold(self) -> bool {
        matches!(self, Self::InterventionalPredictive)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quantity() -> ScientificQuantity {
        ScientificQuantity {
            variable_id: "schema:y".into(),
            variable_name: "outcome".into(),
            role: QuantityRole::Outcome,
            units: "kg".into(),
            population_id: "target".into(),
            regime_id: "do(a=1)".into(),
            horizon: 2,
            functional_id: "mean".into(),
            conditioning: vec![QuantityCondition {
                variable_id: "schema:z".into(),
                value_id: "high".into(),
            }],
            transform_id: "identity".into(),
        }
    }

    #[test]
    fn quantity_identity_checks_each_scientific_dimension() {
        let base = quantity();
        let mut other = base.clone();
        other.variable_name = "renamed label".into();
        assert_eq!(base.require_same_coordinate(&other), Ok(()));
        other.units = "lb".into();
        assert_eq!(base.require_same_coordinate(&other), Err(QuantityMismatch::Units));
        other = base.clone();
        other.population_id = "source".into();
        assert_eq!(base.require_same_coordinate(&other), Err(QuantityMismatch::Population));
        other = base.clone();
        other.regime_id = "observational".into();
        assert_eq!(base.require_same_coordinate(&other), Err(QuantityMismatch::Regime));
        other = base.clone();
        other.horizon = 1;
        assert_eq!(base.require_same_coordinate(&other), Err(QuantityMismatch::Horizon));
        other = base.clone();
        other.functional_id = "risk".into();
        assert_eq!(base.require_same_coordinate(&other), Err(QuantityMismatch::Functional));
        other = base.clone();
        other.conditioning.clear();
        assert_eq!(base.require_same_coordinate(&other), Err(QuantityMismatch::Conditioning));
        other = base.clone();
        other.transform_id = "log".into();
        assert_eq!(base.require_same_coordinate(&other), Err(QuantityMismatch::Transform));
    }

    #[test]
    fn posterior_over_effect_is_not_an_outcome_law() {
        assert!(
            !DistributionMeaning::CausalFunctionalPosterior
                .answers_interventional_outcome_threshold()
        );
        assert!(
            !DistributionMeaning::PosteriorPredictive.answers_interventional_outcome_threshold()
        );
        assert!(
            DistributionMeaning::InterventionalPredictive
                .answers_interventional_outcome_threshold()
        );
    }

    #[test]
    fn empty_units_and_ambiguous_conditions_are_rejected() {
        let mut value = quantity();
        value.units.clear();
        assert_eq!(value.validate(), Err(QuantityMismatch::InvalidCoordinate));
        value = quantity();
        value.conditioning.push(value.conditioning[0].clone());
        assert_eq!(value.validate(), Err(QuantityMismatch::InvalidCoordinate));
    }
}
