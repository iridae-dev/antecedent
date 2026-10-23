//! Route cells and the table that maps each to its family's lowering.
//!
//! One row per `RouteInput` variant. The table generates the cell enum, the
//! classifier from a parsed route, and the dispatch to a family lowering, so a
//! route added to `RouteInput` without a row here does not compile.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::CausalQuery;

use super::{
    LoweringContext, design_based, graph_posterior, static_effect, static_response, structural,
    temporal,
};
use crate::Study;
use crate::analysis::builder::DataInput;
use crate::analysis::execute::{RouteInput, is_class_atom};
use crate::error::CausalError;
use crate::planner::lowered::LoweredProgram;

/// One compiled graph-posterior cell, mirroring the compile-time classification of
/// a study that carries a graph posterior instead of one graph.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) enum GraphPosteriorCell {
    StaticResponse,
    AverageEffect,
    ConditionalEffect,
    TemporalEffect,
    TemporalResponse,
    TemporalMediation,
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

    pub(crate) const fn as_str(self) -> &'static str {
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
    ($($cell:ident => $family:ident),+ $(,)?) => {
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
    StaticAteDag => static_effect,
    StaticAtePag => static_effect,
    StaticAteCpdag => static_effect,
    StaticAteMixed => static_effect,
    ConditionalDag => static_effect,
    ConditionalCpdag => static_effect,
    ConditionalPag => static_effect,
    ConditionalMixed => static_effect,
    DistributionDag => static_effect,
    DistributionMixed => static_effect,
    PathSpecificDag => static_effect,
    StaticResponseDag => static_response,
    StaticResponsePag => static_response,
    StaticResponseCpdag => static_response,
    CodeterminedJoint => static_response,
    MixedAdmgResponse => static_response,
    StaticMediationDag => structural,
    GcmDag => structural,
    GcmTemporal => structural,
    TemporalEffectDag => temporal,
    TemporalEffectClass => temporal,
    TemporalResponseDag => temporal,
    TemporalResponseClass => temporal,
    TemporalMediationDag => temporal,
    TemporalMediationPag => temporal,
    TemporalMediationCpdag => temporal,
    TemporalCrossWorldDag => temporal,
    TemporalCrossWorldClass => temporal,
    MultiEnvTemporalEffect => temporal,
    PanelTemporalEffectDag => temporal,
    PanelTemporalEffectClass => temporal,
    PanelTemporalResponseDag => temporal,
    PanelTemporalResponseClass => temporal,
    TransportMixed => design_based,
    InterferenceDag => design_based,
    RandomizedEffect => design_based,
    PolicyValue => design_based,
    PanelDid => design_based,
    SyntheticControl => design_based,
    ContinuousDoseResponse => design_based,
    LocalPolynomialRatio => design_based,
    Survival => design_based,
    LongitudinalRegime => design_based,
    SequentialContrast => design_based,
    HistoryMatch => design_based,
}
