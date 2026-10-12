//! The typed strategy names a plan record carries.
//!
//! A [`LogicalAnalysisPlanRecord`] names its identifier and estimator by wire
//! string, and every compile path in this crate fills both. Code that routes on
//! the record used to compare those strings (or the `DEFAULT_*` string
//! constants) inline, each site choosing for itself what an absent name meant.
//! [`PlanBinding`] parses the two names once into the closed [`IdentifierId`] /
//! [`EstimatorId`] sets and gives the absent and unrecognised cases one
//! documented meaning.
//!
//! # Missing-record policy
//!
//! A slot is in one of three states ([`PlanName`]): named with a closed-set id,
//! unnamed, or named with a string outside the closed set. There is one rule
//! for the first two and one for the third:
//!
//! - **An unnamed slot reads as the route's default** ([`RouteDefaults`]), which
//!   is what the route would have run had the compiler named it. A route with
//!   no default (a design-based or otherwise single-procedure route) is never
//!   matched by an unnamed slot: [`PlanName::is`] takes no default.
//! - **An unrecognised name is never a match** and, where a site needs the id
//!   itself ([`PlanName::or`], [`PlanBinding::resolve`]), is refused with the
//!   unknown-strategy error rather than guessed at.
//!
//! No site decides this on its own: it supplies the route default, or none.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::LogicalAnalysisPlanRecord;

use crate::error::CausalError;
use crate::strategy_table::{
    DEFAULT_ADMG_ESTIMATOR_ID, DEFAULT_ADMG_IDENTIFIER_ID, DEFAULT_CONDITIONAL_ESTIMATOR_ID,
    DEFAULT_CONDITIONAL_IDENTIFIER_ID, DEFAULT_DISTRIBUTION_ESTIMATOR_ID,
    DEFAULT_DISTRIBUTION_IDENTIFIER_ID, DEFAULT_ESTIMATOR_ID, DEFAULT_IDENTIFIER_ID,
    DEFAULT_PAG_ESTIMATOR_ID, DEFAULT_PAG_IDENTIFIER_ID, DEFAULT_PATH_ESTIMATOR_ID,
    DEFAULT_PATH_IDENTIFIER_ID, DEFAULT_RESPONSE_ESTIMATOR_ID, DEFAULT_RESPONSE_IDENTIFIER_ID,
    EstimatorId, IdentifierId, UnknownStrategy,
};

use super::PhysicalExecutionPlan;

/// The identifier and estimator a route runs when its plan record names none.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RouteDefaults {
    pub(crate) identifier: IdentifierId,
    pub(crate) estimator: EstimatorId,
}

impl RouteDefaults {
    /// Static average effect on a DAG.
    pub(crate) const STATIC_ATE: Self =
        Self { identifier: DEFAULT_IDENTIFIER_ID, estimator: DEFAULT_ESTIMATOR_ID };
    /// Static conditional effect.
    pub(crate) const CONDITIONAL: Self = Self {
        identifier: DEFAULT_CONDITIONAL_IDENTIFIER_ID,
        estimator: DEFAULT_CONDITIONAL_ESTIMATOR_ID,
    };
    /// Static response curve and derivative routes.
    pub(crate) const RESPONSE: Self = Self {
        identifier: DEFAULT_RESPONSE_IDENTIFIER_ID,
        estimator: DEFAULT_RESPONSE_ESTIMATOR_ID,
    };
    /// Interventional distribution.
    pub(crate) const DISTRIBUTION: Self = Self {
        identifier: DEFAULT_DISTRIBUTION_IDENTIFIER_ID,
        estimator: DEFAULT_DISTRIBUTION_ESTIMATOR_ID,
    };
    /// Path-specific effect.
    pub(crate) const PATH: Self =
        Self { identifier: DEFAULT_PATH_IDENTIFIER_ID, estimator: DEFAULT_PATH_ESTIMATOR_ID };
    /// CPDAG / PAG class routes.
    pub(crate) const PAG: Self =
        Self { identifier: DEFAULT_PAG_IDENTIFIER_ID, estimator: DEFAULT_PAG_ESTIMATOR_ID };
    /// ADMG routes.
    pub(crate) const ADMG: Self =
        Self { identifier: DEFAULT_ADMG_IDENTIFIER_ID, estimator: DEFAULT_ADMG_ESTIMATOR_ID };
}

/// What a plan record names for one strategy slot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum PlanName<T> {
    /// The record carries no name for this slot.
    Unnamed,
    /// The record names a member of the closed set.
    Named(T),
    /// The record names a string outside the closed set; the parse refusal is
    /// kept so resolution reports exactly what parsing the name would.
    Unrecognised(UnknownStrategy),
}

/// A closed strategy-id set whose members have wire names.
pub(crate) trait StrategyId: Copy + Eq + std::str::FromStr<Err = UnknownStrategy> {
    /// The id's wire name.
    fn wire(self) -> &'static str;
}

impl StrategyId for IdentifierId {
    fn wire(self) -> &'static str {
        self.as_str()
    }
}

impl StrategyId for EstimatorId {
    fn wire(self) -> &'static str {
        self.as_str()
    }
}

impl<T: StrategyId> PlanName<T> {
    fn parse(recorded: Option<&str>) -> Self {
        match recorded.map(str::parse) {
            None => Self::Unnamed,
            Some(Ok(id)) => Self::Named(id),
            Some(Err(unknown)) => Self::Unrecognised(unknown),
        }
    }

    /// The closed-set id the record names, if it names one.
    pub(crate) fn named(&self) -> Option<T> {
        match self {
            Self::Named(id) => Some(*id),
            Self::Unnamed | Self::Unrecognised(_) => None,
        }
    }

    /// Whether the record names exactly `expected` (verification).
    pub(crate) fn is(&self, expected: T) -> bool {
        self.named() == Some(expected)
    }

    /// Whether the record's name is exactly `wire`, in the closed set or not.
    ///
    /// For the design and temporal-mediation routes, whose plan names are
    /// deliberately outside the closed id sets and so cannot be typed.
    pub(crate) fn is_wire(&self, wire: &str) -> bool {
        match self {
            Self::Named(id) => id.wire() == wire,
            Self::Unrecognised(unknown) => unknown.got == wire,
            Self::Unnamed => false,
        }
    }

    /// Whether the slot resolves to `expected`, reading an unnamed slot as
    /// `default` (resolution, without refusing an unrecognised name).
    pub(crate) fn is_or(&self, default: T, expected: T) -> bool {
        match self {
            Self::Unnamed => default == expected,
            Self::Named(id) => *id == expected,
            Self::Unrecognised(_) => false,
        }
    }

    /// The id this slot resolves to, reading an unnamed slot as `default`
    /// (resolution).
    ///
    /// # Errors
    ///
    /// The unknown-strategy compile refusal when the record names a string
    /// outside the closed set.
    pub(crate) fn or(&self, default: T) -> Result<T, CausalError> {
        match self {
            Self::Unnamed => Ok(default),
            Self::Named(id) => Ok(*id),
            Self::Unrecognised(unknown) => Err(unknown.clone().into()),
        }
    }
}

/// The identifier and estimator a plan record names, parsed once.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PlanBinding {
    pub(crate) identifier: PlanName<IdentifierId>,
    pub(crate) estimator: PlanName<EstimatorId>,
}

impl PlanBinding {
    /// Parse the names `record` carries.
    pub(crate) fn of(record: &LogicalAnalysisPlanRecord) -> Self {
        Self {
            identifier: PlanName::parse(record.identifier.as_deref()),
            estimator: PlanName::parse(record.estimator.as_deref()),
        }
    }

    /// Whether the record names exactly this identifier and estimator; the
    /// form for a route with no default.
    pub(crate) fn names(&self, identifier: IdentifierId, estimator: EstimatorId) -> bool {
        self.identifier.is(identifier) && self.estimator.is(estimator)
    }

    /// [`Self::names`] for routes whose plan names are outside the closed sets.
    pub(crate) fn names_wire(&self, identifier: &str, estimator: &str) -> bool {
        self.identifier.is_wire(identifier) && self.estimator.is_wire(estimator)
    }

    /// The identifier and estimator the record resolves to under `defaults`.
    ///
    /// # Errors
    ///
    /// The unknown-strategy compile refusal for an unrecognised name.
    pub(crate) fn resolve(
        &self,
        defaults: RouteDefaults,
    ) -> Result<(IdentifierId, EstimatorId), CausalError> {
        Ok((self.identifier.or(defaults.identifier)?, self.estimator.or(defaults.estimator)?))
    }

    /// The identifier the record resolves to under `defaults`.
    ///
    /// # Errors
    ///
    /// The unknown-strategy compile refusal for an unrecognised name.
    pub(crate) fn identifier_or(
        &self,
        defaults: RouteDefaults,
    ) -> Result<IdentifierId, CausalError> {
        self.identifier.or(defaults.identifier)
    }

    /// The estimator the record resolves to under `defaults`.
    ///
    /// # Errors
    ///
    /// The unknown-strategy compile refusal for an unrecognised name.
    pub(crate) fn estimator_or(&self, defaults: RouteDefaults) -> Result<EstimatorId, CausalError> {
        self.estimator.or(defaults.estimator)
    }

    /// Whether the record resolves to exactly `defaults`.
    pub(crate) fn runs_defaults(&self, defaults: RouteDefaults) -> bool {
        self.resolves_to(defaults, defaults.identifier, defaults.estimator)
    }

    /// Whether the record resolves to this identifier and estimator under
    /// `defaults`; the boolean form of [`Self::resolve`].
    pub(crate) fn resolves_to(
        &self,
        defaults: RouteDefaults,
        identifier: IdentifierId,
        estimator: EstimatorId,
    ) -> bool {
        self.identifier.is_or(defaults.identifier, identifier)
            && self.estimator.is_or(defaults.estimator, estimator)
    }
}

impl PhysicalExecutionPlan {
    /// The strategy names this plan's logical record carries.
    pub(crate) fn binding(&self) -> PlanBinding {
        PlanBinding::of(&self.logical.record)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(identifier: Option<&str>, estimator: Option<&str>) -> LogicalAnalysisPlanRecord {
        LogicalAnalysisPlanRecord {
            plan_id: "p".into(),
            data_classification: antecedent_contract::DataClassification::Tabular,
            discovery_algorithm: None,
            graph_review_required: false,
            identifier: identifier.map(Into::into),
            estimator: estimator.map(Into::into),
            validation_suite: None,
            query_variables: std::sync::Arc::from([]),
        }
    }

    #[test]
    fn verification_never_matches_an_unnamed_or_unrecognised_slot() {
        let named = PlanBinding::of(&record(Some("backdoor.adjustment"), Some("aipw")));
        assert!(named.names(IdentifierId::BackdoorAdjustment, EstimatorId::Aipw));
        assert!(!named.names(IdentifierId::BackdoorAdjustment, EstimatorId::LinearAdjustmentAte));
        let unnamed = PlanBinding::of(&record(None, None));
        assert!(!unnamed.names(IdentifierId::BackdoorAdjustment, EstimatorId::Aipw));
        let unknown = PlanBinding::of(&record(Some("no.such"), Some("aipw")));
        assert!(!unknown.names(IdentifierId::BackdoorAdjustment, EstimatorId::Aipw));
    }

    #[test]
    fn resolution_reads_an_unnamed_slot_as_the_default_and_refuses_an_unknown_name() {
        let defaults = RouteDefaults::STATIC_ATE;
        let pair = (defaults.identifier, defaults.estimator);
        let unnamed = PlanBinding::of(&record(None, None));
        assert_eq!(unnamed.resolve(defaults).unwrap(), pair);
        assert!(unnamed.resolves_to(defaults, pair.0, pair.1));
        let partial = PlanBinding::of(&record(None, Some("aipw")));
        assert_eq!(
            partial.resolve(defaults).unwrap(),
            (IdentifierId::BackdoorAdjustment, EstimatorId::Aipw)
        );
        let unknown = PlanBinding::of(&record(Some("no.such"), None));
        assert!(unknown.resolve(defaults).is_err());
        assert!(!unknown.resolves_to(defaults, pair.0, pair.1));
    }
}
