//! Route cells and the table that maps each to its family's lowering.
//!
//! One row per `RouteInput` variant. The table generates the cell enum, the
//! classifier from a parsed route, the dispatch to a family lowering, and the
//! public [`PlanRoute`] a plan view names the cell by, with the cell's family. A
//! route added to `RouteInput` without a row here does not compile, and a row
//! always has a plan route.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::CausalQuery;

use super::{
    LoweringContext, design_based, discovery, graph_posterior, static_effect, static_response,
    structural, temporal,
};
use crate::Study;
use crate::analysis::builder::DataInput;
use crate::analysis::execute::{RouteInput, is_class_atom};
use crate::error::CausalError;
use crate::planner::lowered::LoweredProgram;
use crate::planner::plan_view::RouteFamily;

/// One compiled graph-posterior cell, mirroring the compile-time classification of
/// a study that carries a graph posterior instead of one graph.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum GraphPosteriorCell {
    /// Static response curve or intervention response over the atoms.
    StaticResponse,
    /// Static average effect over the atoms.
    AverageEffect,
    /// Static conditional effect over the atoms.
    ConditionalEffect,
    /// Temporal pulse or sustained effect over the atoms.
    TemporalEffect,
    /// Temporal response over the atoms.
    TemporalResponse,
    /// Temporal mediation over the atoms.
    TemporalMediation,
    /// A cross-world query over the atoms of a temporal class posterior.
    CrossWorld,
}

impl GraphPosteriorCell {
    #[cfg(test)]
    pub(crate) const ALL: [Self; 7] = [
        Self::StaticResponse,
        Self::AverageEffect,
        Self::ConditionalEffect,
        Self::TemporalEffect,
        Self::TemporalResponse,
        Self::TemporalMediation,
        Self::CrossWorld,
    ];

    /// Stable name, written to the licensed-route record.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::StaticResponse => "GraphPosteriorStaticResponse",
            Self::AverageEffect => "GraphPosteriorAverageEffect",
            Self::ConditionalEffect => "GraphPosteriorConditionalEffect",
            Self::TemporalEffect => "GraphPosteriorTemporalEffect",
            Self::TemporalResponse => "GraphPosteriorTemporalResponse",
            Self::TemporalMediation => "GraphPosteriorTemporalMediation",
            Self::CrossWorld => "GraphPosteriorCrossWorld",
        }
    }

    /// The cell a graph-posterior study compiles as; `None` for a study without a
    /// graph posterior or a data and query pair graph-posterior compile refuses.
    pub(crate) fn classify(study: &Study) -> Option<Self> {
        let posterior = study.graph_posterior.as_ref()?;
        Some(match (&study.data, &study.query) {
            (DataInput::Tabular(_), CausalQuery::Response(_)) => Self::StaticResponse,
            (DataInput::Tabular(_), CausalQuery::AverageEffect(_)) => Self::AverageEffect,
            (DataInput::Tabular(_), CausalQuery::ConditionalEffect(_)) => Self::ConditionalEffect,
            (DataInput::Temporal(_) | DataInput::Event(_), CausalQuery::TemporalEffect(_)) => {
                Self::TemporalEffect
            }
            (DataInput::Temporal(_) | DataInput::Event(_), CausalQuery::Response(_)) => {
                Self::TemporalResponse
            }
            (DataInput::Temporal(_) | DataInput::Event(_), CausalQuery::Mediation(_)) => {
                Self::TemporalMediation
            }
            (
                DataInput::Temporal(_) | DataInput::Event(_),
                CausalQuery::NestedCounterfactual(_)
                | CausalQuery::PathSpecific(_)
                | CausalQuery::Counterfactual(_)
                | CausalQuery::Distribution(_),
            ) if is_class_atom(posterior.atom_kind) => Self::CrossWorld,
            _ => return None,
        })
    }
}

macro_rules! route_lowerings {
    ($($cell:ident => $family:ident as $kind:ident),+ $(,)?) => {
        /// One compiled route cell: a `RouteInput` variant, or a graph-posterior cell.
        #[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
        pub(crate) enum RouteCell {
            $($cell,)+
            GraphPosterior(GraphPosteriorCell),
        }

        impl RouteCell {
            /// The cell a parsed route compiles as.
            pub(crate) fn of(input: &RouteInput<'_>) -> Self {
                match input {
                    $(RouteInput::$cell { .. } => Self::$cell,)+
                }
            }

            /// Stable name, written to the licensed-route record.
            pub(crate) const fn as_str(self) -> &'static str {
                match self {
                    $(Self::$cell => stringify!($cell),)+
                    Self::GraphPosterior(cell) => cell.as_str(),
                }
            }

            /// The family that lowers this cell.
            #[cfg(test)]
            #[allow(
                clippy::match_same_arms,
                reason = "one arm per route cell so the table reads cell by cell"
            )]
            pub(crate) const fn family(self) -> RouteFamily {
                match self {
                    $(Self::$cell => RouteFamily::$kind,)+
                    Self::GraphPosterior(_) => RouteFamily::GraphPosterior,
                }
            }

            /// The route a plan view names this cell by.
            pub(crate) const fn plan_route(self) -> PlanRoute {
                match self {
                    $(Self::$cell => PlanRoute::$cell,)+
                    Self::GraphPosterior(cell) => PlanRoute::GraphPosterior(cell),
                }
            }
        }

        /// One compiled route a plan view describes: every registered route cell,
        /// every graph-posterior cell, a cyclic family and a standalone experiment
        /// plan. Generated from the route table, so a registered cell always has a
        /// route here.
        #[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
        #[non_exhaustive]
        pub enum PlanRoute {
            $(
                #[doc = concat!("The `", stringify!($cell), "` route cell.")]
                $cell,
            )+
            /// A graph-posterior cell.
            GraphPosterior(GraphPosteriorCell),
            /// A cyclic family, by its stable family name.
            CyclicFamily(&'static str),
            /// A standalone experiment plan, by its design name.
            Planning(&'static str),
        }

        impl PlanRoute {
            /// Stable name; for a route cell, the `lowering` column of the
            /// licensed-route record.
            #[must_use]
            pub const fn as_str(self) -> &'static str {
                match self {
                    $(Self::$cell => stringify!($cell),)+
                    Self::GraphPosterior(cell) => cell.as_str(),
                    Self::CyclicFamily(_) => "CyclicFamily",
                    Self::Planning(_) => "ExperimentPlanning",
                }
            }

            /// The family that lowered the route.
            #[must_use]
            #[allow(
                clippy::match_same_arms,
                reason = "one arm per route cell so the table reads cell by cell"
            )]
            pub const fn family(self) -> RouteFamily {
                match self {
                    $(Self::$cell => RouteFamily::$kind,)+
                    Self::GraphPosterior(_) => RouteFamily::GraphPosterior,
                    Self::CyclicFamily(_) => RouteFamily::Cyclic,
                    Self::Planning(_) => RouteFamily::Planning,
                }
            }
        }

        /// Lower `cx` with the lowering its cell is registered to.
        #[allow(
            clippy::match_same_arms,
            reason = "one row per route cell so the table reads cell by cell; a family lowers several cells"
        )]
        pub(super) fn lower_cell(cx: &LoweringContext<'_>) -> Result<LoweredProgram, CausalError> {
            match cx.cell {
                $(RouteCell::$cell => $family::lower(cx),)+
                RouteCell::GraphPosterior(_) => graph_posterior::lower(cx),
            }
        }

        /// Every registered route cell with the family that lowers it.
        #[cfg(test)]
        pub(crate) const REGISTERED: &[(RouteCell, &str)] =
            &[$((RouteCell::$cell, stringify!($family)),)+];
    };
}

route_lowerings! {
    StaticAteDag => static_effect as StaticEffect,
    StaticAtePag => static_effect as StaticEffect,
    StaticAteCpdag => static_effect as StaticEffect,
    StaticAteMixed => static_effect as StaticEffect,
    ConditionalDag => static_effect as StaticEffect,
    ConditionalCpdag => static_effect as StaticEffect,
    ConditionalPag => static_effect as StaticEffect,
    ConditionalMixed => static_effect as StaticEffect,
    DistributionDag => static_effect as StaticEffect,
    DistributionMixed => static_effect as StaticEffect,
    PathSpecificDag => static_effect as StaticEffect,
    StaticResponseDag => static_response as StaticResponse,
    StaticResponsePag => static_response as StaticResponse,
    StaticResponseCpdag => static_response as StaticResponse,
    CodeterminedJoint => static_response as StaticResponse,
    MixedAdmgResponse => static_response as StaticResponse,
    StaticMediationDag => structural as Structural,
    GcmDag => structural as Structural,
    GcmTemporal => structural as Structural,
    TemporalEffectDag => temporal as Temporal,
    TemporalEffectClass => temporal as Temporal,
    TemporalResponseDag => temporal as Temporal,
    TemporalResponseClass => temporal as Temporal,
    TemporalMediationDag => temporal as Temporal,
    TemporalMediationPag => temporal as Temporal,
    TemporalMediationCpdag => temporal as Temporal,
    TemporalCrossWorldDag => temporal as Temporal,
    TemporalCrossWorldClass => temporal as Temporal,
    MultiEnvTemporalEffect => temporal as Temporal,
    PanelTemporalEffectDag => temporal as Temporal,
    PanelTemporalEffectClass => temporal as Temporal,
    PanelTemporalResponseDag => temporal as Temporal,
    PanelTemporalResponseClass => temporal as Temporal,
    TransportMixed => design_based as DesignBased,
    InterferenceDag => design_based as DesignBased,
    RandomizedEffect => design_based as DesignBased,
    PolicyValue => design_based as DesignBased,
    PanelDid => design_based as DesignBased,
    SyntheticControl => design_based as DesignBased,
    ContinuousDoseResponse => design_based as DesignBased,
    LocalPolynomialRatio => design_based as DesignBased,
    Survival => design_based as DesignBased,
    LongitudinalRegime => design_based as DesignBased,
    SequentialContrast => design_based as DesignBased,
    HistoryMatch => design_based as DesignBased,
    TemporalDesign => design_based as DesignBased,
    DiscoverStatic => discovery as Discovery,
    DiscoverTemporal => discovery as Discovery,
    GcmAttributeDag => structural as Structural,
    GcmSimulateDag => structural as Structural,
}
