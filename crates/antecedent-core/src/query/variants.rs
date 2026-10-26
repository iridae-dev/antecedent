//! The closed set of causal query variants.
//!
//! Each payload type implements [`QueryVariant`] (its canonical tag, payload-free
//! kind, and validation), and a single table drives the enum, its kind mirror, the
//! constructors, the `From` impls, the canonical encoding and the validation
//! dispatch. Adding a variant is one row below plus its [`QueryVariant`]
//! behaviour; no other site enumerates the variants except the deliberately
//! exhaustive classification tables elsewhere, which fail to compile until they
//! are extended.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::canonical_encode::CanonicalEncode;
use crate::identity::CanonicalWriter;

use super::QueryError;
use super::{
    AnomalyAttributionQuery, AssociationQuery, AttributeQuery, AverageEffectQuery,
    ChangeAttributionQuery, ConditionalEffectQuery, ContinuousDoseResponseQuery,
    CounterfactualQuery, DiscoverQuery, HistoryMatchQuery, InterferenceQuery,
    InterventionalDistributionQuery, IvBoundsQuery, LocalPolynomialRatioQuery,
    LongitudinalRegimeQuery, MechanismChangeQuery, MediationQuery, NestedCounterfactualQuery,
    PanelDidQuery, PathSpecificEffectQuery, PolicyValueQuery, RandomizedEffectQuery, ResponseQuery,
    SequentialContrastQuery, SimulateQuery, SurvivalQuery, SyntheticControlQuery,
    TemporalDesignQuery, TemporalEffectQuery, TransportQuery, UnitChangeQuery,
};

/// Behaviour every [`CausalQuery`] payload type provides.
///
/// The canonical tag is the number written before the payload's own encoding.
/// Tags are part of the identity format: they are fixed per variant and never
/// reused, and the variant table below is the only place they are spelled.
pub trait QueryVariant: CanonicalEncode + Sized + Into<CausalQuery> {
    /// Canonical tag written before the payload encoding.
    const TAG: u64;
    /// The payload-free variant this type is carried as.
    const KIND: CausalQueryKind;

    /// Validate the payload (the type's own structural validation).
    ///
    /// # Errors
    ///
    /// Propagates the payload's [`QueryError`].
    fn validate_variant(&self) -> Result<(), QueryError>;
}

macro_rules! causal_queries {
    ($(
        $(#[$doc:meta])*
        $variant:ident($payload:ty) = $tag:literal, $constructor:ident, $constructor_doc:literal;
    )+) => {
        /// Top-level causal query enum.
        #[derive(Clone, Debug, PartialEq)]
        #[non_exhaustive]
        pub enum CausalQuery {
            $(
                $(#[$doc])*
                $variant($payload),
            )+
        }

        /// The variant of a [`CausalQuery`] without its payload.
        ///
        /// Unlike [`CausalQuery`] this enum is deliberately exhaustive. It exists so a
        /// per-query table (which graph classes a query is licensed on, which queries
        /// are design-based) can be one `match` with no wildcard arm: adding a
        /// [`CausalQuery`] variant extends [`CausalQuery::kind`], which extends this
        /// enum, which fails to compile every such table until its new row is written.
        #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
        pub enum CausalQueryKind {
            $(
                #[doc = concat!("[`CausalQuery::", stringify!($variant), "`].")]
                $variant,
            )+
        }

        impl CausalQuery {
            /// The payload-free variant of this query.
            #[must_use]
            pub const fn kind(&self) -> CausalQueryKind {
                match self {
                    $( Self::$variant(_) => CausalQueryKind::$variant, )+
                }
            }

            /// Validate the inner query.
            ///
            /// # Errors
            ///
            /// Propagates inner [`QueryError`].
            pub fn validate(&self) -> Result<(), QueryError> {
                match self {
                    $( Self::$variant(query) => <$payload as QueryVariant>::validate_variant(query), )+
                }
            }

            $(
                #[doc = $constructor_doc]
                #[must_use]
                pub fn $constructor(query: $payload) -> Self {
                    Self::$variant(query)
                }
            )+
        }

        /// The variant tag, then the payload's own canonical encoding.
        impl CanonicalEncode for CausalQuery {
            fn encode(&self, writer: &mut CanonicalWriter) {
                match self {
                    $(
                        Self::$variant(query) => {
                            writer.word(<$payload as QueryVariant>::TAG);
                            query.encode(writer);
                        }
                    )+
                }
            }
        }

        $(
            impl From<$payload> for CausalQuery {
                fn from(query: $payload) -> Self {
                    Self::$variant(query)
                }
            }

            impl QueryVariant for $payload {
                const TAG: u64 = $tag;
                const KIND: CausalQueryKind = CausalQueryKind::$variant;

                fn validate_variant(&self) -> Result<(), QueryError> {
                    self.validate()
                }
            }
        )+
    };
}

causal_queries! {
    /// Average / population effect (static).
    AverageEffect(AverageEffectQuery) = 1, average_effect,
        "Construct an average-effect query.";
    /// Temporal effect over a discrete horizon.
    TemporalEffect(TemporalEffectQuery) = 2, temporal_effect,
        "Construct a temporal-effect query.";
    /// Counterfactual / unit-level what-if query .
    Counterfactual(CounterfactualQuery) = 3, counterfactual,
        "Construct a counterfactual query.";
    /// Fixed-contract nested natural direct effect.
    NestedCounterfactual(NestedCounterfactualQuery) = 4, nested_counterfactual,
        "Construct a nested counterfactual query.";
    /// Anomaly attribution for one or more units .
    AnomalyAttribution(AnomalyAttributionQuery) = 5, anomaly_attribution,
        "Construct an anomaly attribution query.";
    /// Distribution / population change attribution .
    ChangeAttribution(ChangeAttributionQuery) = 6, change_attribution,
        "Construct a change attribution query.";
    /// Mechanism-change detection — not attribution .
    MechanismChange(MechanismChangeQuery) = 7, mechanism_change,
        "Construct a mechanism-change detection query.";
    /// Per-unit change attribution .
    UnitChange(UnitChangeQuery) = 8, unit_change,
        "Construct a unit-change attribution query.";
    /// Mediation (direct / mediated / natural effects).
    Mediation(MediationQuery) = 9, mediation,
        "Construct a mediation query.";
    /// Conditional average effect given modifiers.
    ConditionalEffect(ConditionalEffectQuery) = 10, conditional_effect,
        "Construct a conditional-effect query.";
    /// Interventional distribution P(Y | do(...)).
    Distribution(InterventionalDistributionQuery) = 11, distribution,
        "Construct an interventional-distribution query.";
    /// Path-specific effect / contribution.
    PathSpecific(PathSpecificEffectQuery) = 12, path_specific,
        "Construct a path-specific effect query.";
    /// Continuous response, derivative, policy response, or Jacobian.
    Response(ResponseQuery) = 13, response,
        "Construct a continuous-response query.";
    /// Structurally transported response between populations.
    Transport(TransportQuery) = 14, transport,
        "Construct a transportability query.";
    /// Randomization-based causal effect under interference.
    Interference(InterferenceQuery) = 15, interference,
        "Construct an interference query.";
    /// Bernoulli randomized two-arm intention-to-treat effect.
    RandomizedEffect(RandomizedEffectQuery) = 16, randomized_effect,
        "Construct a randomized ITT query.";
    /// Fixed randomized binary or multi-action policy value.
    PolicyValue(PolicyValueQuery) = 17, policy_value,
        "Construct a held-out policy value query.";
    /// Local conditional continuous-dose response with supplied dose density.
    ContinuousDoseResponse(ContinuousDoseResponseQuery) = 18, continuous_dose_response,
        "Construct a local conditional continuous-dose response query.";
    /// Balanced two-period panel difference in differences.
    PanelDid(PanelDidQuery) = 19, panel_did,
        "Construct a balanced-panel `DiD` query.";
    /// Balanced-panel synthetic control with one treated unit and donor pool.
    SyntheticControl(SyntheticControlQuery) = 20, synthetic_control,
        "Construct a synthetic-control query.";
    /// Fixed-bandwidth local fuzzy discontinuity or regression kink ratio.
    LocalPolynomialRatio(LocalPolynomialRatioQuery) = 21, local_polynomial_ratio,
        "Construct a local fuzzy-discontinuity or regression-kink query.";
    /// Randomized right-censored survival or competing-risk functional.
    Survival(SurvivalQuery) = 22, survival,
        "Construct a randomized survival or competing-risk query.";
    /// Prespecified longitudinal regime value over subject histories.
    LongitudinalRegime(LongitudinalRegimeQuery) = 23, longitudinal_regime,
        "Construct a prespecified longitudinal regime value query.";
    /// Observational two-decision contrast of two static regimes.
    SequentialContrast(SequentialContrastQuery) = 24, sequential_contrast,
        "Construct an observational sequential regime contrast query.";
    /// Nearest-neighbor contrast on histories, matched within calendar periods.
    HistoryMatch(HistoryMatchQuery) = 25, history_match,
        "Construct a within-period history match query.";
    /// Structure discovery: a graph, graph class or structure posterior.
    Discover(DiscoverQuery) = 26, discover,
        "Construct a structure-discovery query.";
    /// Attribution on a fitted structural causal model.
    Attribute(AttributeQuery) = 27, attribute,
        "Construct an attribution query on a fitted structural model.";
    /// Simulation on a fitted structural causal model.
    Simulate(SimulateQuery) = 28, simulate,
        "Construct a simulation query on a fitted structural model.";
    /// Excursion effects of a sequentially randomized unit-period design, or regime survival
    /// on subject trajectories.
    TemporalDesign(TemporalDesignQuery) = 29, temporal_design,
        "Construct a sequentially randomized unit-period design query.";
    /// Sharp partial-identification bounds on the average effect of a binary treatment under a
    /// binary instrument.
    IvBounds(IvBoundsQuery) = 30, iv_bounds,
        "Construct an instrumental-variable bounds query.";
    /// Noncausal association of an outcome with predictors in the observed data: a conditional
    /// law, regression coefficients, an association measure or a scored predictive model. It
    /// states no intervention and never ranks actions.
    Association(AssociationQuery) = 31, association,
        "Construct a noncausal association query.";
}
