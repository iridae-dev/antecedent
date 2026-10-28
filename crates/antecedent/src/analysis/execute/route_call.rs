//! A classified execution route together with the data and query payload its executor reads.
//!
//! Classification reads the study's data and query, and the executors then need the same
//! pieces again: the series, the panel, the query's own payload. [`Study::route_call`]
//! opens them once. A route and the data or query it was classified from can then no
//! longer disagree, so no executor re-checks for a mismatch it cannot receive.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    AnomalyAttributionQuery, AverageEffectQuery, ChangeAttributionQuery, ConditionalEffectQuery,
    CounterfactualQuery, InterferenceQuery, InterventionalDistributionQuery, MechanismChangeQuery,
    MediationQuery, PathSpecificEffectQuery, ResponseQuery, TemporalEffectQuery, TransportQuery,
    UnitChangeQuery,
};
use antecedent_data::{MultiEnvironmentData, PanelData, TabularData, TimeSeriesData};

use super::{
    AnalysisRoute, CausalError, CausalQuery, DataInput, DataModality, GraphClass,
    StaticMediationSource, Study, classify_route,
};

/// The refusal for a route whose data or query kind disagrees with the one the route was
/// classified from. Classification reads both, so reaching this means a caller swapped the
/// study's data or query between planning and execution.
pub(super) fn route_mismatch() -> CausalError {
    CausalError::Compile {
        message: "execution route does not match the study's data or query kind".into(),
    }
}

/// The refusal for a data, graph and query combination no executor serves.
pub(super) fn route_unsupported() -> CausalError {
    CausalError::Unsupported { message: "execute path unsupported for this configuration" }
}

/// A tabular route with the query payload its executor reads.
#[derive(Clone, Copy)]
pub(super) enum TabularCall<'a> {
    Response(&'a ResponseQuery),
    StaticAte(&'a AverageEffectQuery),
    Distribution(&'a InterventionalDistributionQuery),
    PathSpecific(&'a PathSpecificEffectQuery),
    Conditional(&'a ConditionalEffectQuery),
    StaticMediation(StaticMediationSource<'a>),
    Counterfactual(&'a CounterfactualQuery),
    Anomaly(&'a AnomalyAttributionQuery),
    ChangeAttribution(&'a ChangeAttributionQuery),
    MechanismChange(&'a MechanismChangeQuery),
    UnitChange(&'a UnitChangeQuery),
    Transport(&'a TransportQuery),
    Interference(&'a InterferenceQuery),
    /// A route whose executor reads the study's query itself.
    Other(AnalysisRoute),
}

impl<'a> TabularCall<'a> {
    /// Pair `route` with the payload of the `query` it was classified from.
    ///
    /// # Errors
    ///
    /// The query is not the kind the route was classified from.
    pub(super) fn of(route: AnalysisRoute, query: &'a CausalQuery) -> Result<Self, CausalError> {
        use AnalysisRoute as R;
        Ok(match (route, query) {
            (R::Response, CausalQuery::Response(q)) => Self::Response(q),
            (R::StaticAte, CausalQuery::AverageEffect(q)) => Self::StaticAte(q),
            (R::Distribution, CausalQuery::Distribution(q)) => Self::Distribution(q),
            (R::PathSpecific, CausalQuery::PathSpecific(q)) => Self::PathSpecific(q),
            (R::Conditional, CausalQuery::ConditionalEffect(q)) => Self::Conditional(q),
            (R::StaticMediation, CausalQuery::Mediation(q)) => {
                Self::StaticMediation(StaticMediationSource::Mediation(q))
            }
            (R::StaticMediation, CausalQuery::NestedCounterfactual(q)) => {
                Self::StaticMediation(StaticMediationSource::Nested(q))
            }
            (R::Counterfactual, CausalQuery::Counterfactual(q)) => Self::Counterfactual(q),
            (R::Anomaly, CausalQuery::AnomalyAttribution(q)) => Self::Anomaly(q),
            (R::ChangeAttribution, CausalQuery::ChangeAttribution(q)) => Self::ChangeAttribution(q),
            (R::MechanismChange, CausalQuery::MechanismChange(q)) => Self::MechanismChange(q),
            (R::UnitChange, CausalQuery::UnitChange(q)) => Self::UnitChange(q),
            (R::Transport, CausalQuery::Transport(q)) => Self::Transport(q),
            (R::Interference, CausalQuery::Interference(q)) => Self::Interference(q),
            (
                R::Response
                | R::StaticAte
                | R::Distribution
                | R::PathSpecific
                | R::Conditional
                | R::StaticMediation
                | R::Counterfactual
                | R::Anomaly
                | R::ChangeAttribution
                | R::MechanismChange
                | R::UnitChange
                | R::Transport
                | R::Interference,
                _,
            ) => return Err(route_mismatch()),
            (other, _) => Self::Other(other),
        })
    }

    /// The route this call was classified as.
    pub(super) const fn route(&self) -> AnalysisRoute {
        match self {
            Self::Response(_) => AnalysisRoute::Response,
            Self::StaticAte(_) => AnalysisRoute::StaticAte,
            Self::Distribution(_) => AnalysisRoute::Distribution,
            Self::PathSpecific(_) => AnalysisRoute::PathSpecific,
            Self::Conditional(_) => AnalysisRoute::Conditional,
            Self::StaticMediation(_) => AnalysisRoute::StaticMediation,
            Self::Counterfactual(_) => AnalysisRoute::Counterfactual,
            Self::Anomaly(_) => AnalysisRoute::Anomaly,
            Self::ChangeAttribution(_) => AnalysisRoute::ChangeAttribution,
            Self::MechanismChange(_) => AnalysisRoute::MechanismChange,
            Self::UnitChange(_) => AnalysisRoute::UnitChange,
            Self::Transport(_) => AnalysisRoute::Transport,
            Self::Interference(_) => AnalysisRoute::Interference,
            Self::Other(route) => *route,
        }
    }
}

/// A route classified against the study's data, holding the pieces its executor reads.
#[derive(Clone, Copy)]
pub(super) enum RouteCall<'a> {
    Tabular(TabularCall<'a>, &'a TabularData),
    CrossWorld(&'a TimeSeriesData),
    SeriesMediation {
        series: &'a TimeSeriesData,
        query: &'a MediationQuery,
    },
    SeriesEffect {
        series: &'a TimeSeriesData,
        query: &'a TemporalEffectQuery,
    },
    SeriesResponse {
        series: &'a TimeSeriesData,
        query: &'a ResponseQuery,
    },
    SeriesAssociation(&'a TimeSeriesData),
    /// An anomaly or change attribution over a temporal class or ADMG.
    ClassAttribution(&'a TimeSeriesData),
    /// An attribution family query over a supplied `TemporalDag`.
    SeriesAttribution(&'a TimeSeriesData),
    PanelEffect {
        panel: &'a PanelData,
        query: &'a TemporalEffectQuery,
    },
    PanelResponse {
        panel: &'a PanelData,
        query: &'a ResponseQuery,
    },
    MultiEnvEffect {
        multi: &'a MultiEnvironmentData,
        query: &'a TemporalEffectQuery,
    },
}

impl Study {
    /// Refuse sampling weights that the static effect route about to run would not read.
    ///
    /// Only the average effect reads sampling weights, and only the ones declared on its
    /// estimator. Observation weights carried by the table are read by the randomized and
    /// panel designs and by a single series, never by a static effect route, and weights
    /// declared on an estimator that the route does not run are read by nothing. Either would
    /// publish the unweighted population's number as the weighted population's, so both are
    /// refused with `sampling_weights.route_unsupported`.
    ///
    /// # Errors
    ///
    /// `option_not_applicable` when the table carries weights or an estimator declares them
    /// and the route is a static effect route that cannot honour them.
    pub(crate) fn refuse_unread_sampling_weights(
        &self,
        table: &TabularData,
    ) -> Result<(), CausalError> {
        let reads_declared = matches!(self.query, CausalQuery::AverageEffect(_));
        let static_effect = matches!(
            self.query,
            CausalQuery::AverageEffect(_)
                | CausalQuery::ConditionalEffect(_)
                | CausalQuery::Mediation(_)
                | CausalQuery::NestedCounterfactual(_)
                | CausalQuery::PathSpecific(_)
                | CausalQuery::Counterfactual(_)
                | CausalQuery::Response(_)
                | CausalQuery::Distribution(_)
                | CausalQuery::Transport(_)
                | CausalQuery::ContinuousDoseResponse(_)
                | CausalQuery::LocalPolynomialRatio(_)
                | CausalQuery::TemporalDesign(_)
                | CausalQuery::SyntheticControl(_)
                | CausalQuery::Survival(_)
        );
        if !static_effect {
            return Ok(());
        }
        if table.storage().weights().is_some() {
            return Err(CausalError::Estimate(
                antecedent_estimate::sampling_weights::unsupported_route(
                    "the table carries observation weights, which this static effect route does                      not read, so the estimate would describe the unweighted rows. Next attempt:                      declare the weights on the estimator of an average effect                      (with_sampling_weights), or drop them from the table",
                ),
            ));
        }
        let declared = self
            .declared
            .estimator_spec
            .as_deref()
            .is_some_and(|spec| spec.sampling_weights_calibration_label().is_some());
        if declared && !reads_declared {
            return Err(CausalError::Estimate(
                antecedent_estimate::sampling_weights::unsupported_route(
                    "sampling weights are declared on an estimator that this route does not run,                      so they would not change the estimate. Next attempt: ask the average effect                      for the weighted population, or drop the weights",
                ),
            ));
        }
        Ok(())
    }

    /// Classify the study's query against `data` and open the payload its executor reads.
    ///
    /// # Errors
    ///
    /// No executor serves the combination, or the data or query is not the kind the
    /// classification read.
    pub(super) fn route_call<'a>(
        &'a self,
        data: &'a DataInput,
    ) -> Result<RouteCall<'a>, CausalError> {
        match data {
            DataInput::Tabular(table) => self.tabular_route_call(table),
            DataInput::Temporal(series) | DataInput::Event(series) => {
                self.series_route_call(series)
            }
            DataInput::Panel(panel) => self.panel_route_call(panel),
            DataInput::MultiEnv(multi) => self.multi_env_route_call(multi),
        }
    }

    /// [`Self::route_call`] for tabular rows.
    pub(super) fn tabular_route_call<'a>(
        &'a self,
        table: &'a TabularData,
    ) -> Result<RouteCall<'a>, CausalError> {
        let route =
            classify_route(DataModality::Tabular, &self.query).ok_or_else(route_unsupported)?;
        Ok(RouteCall::Tabular(TabularCall::of(route, &self.query)?, table))
    }

    /// [`Self::route_call`] for a series or an event series.
    pub(super) fn series_route_call<'a>(
        &'a self,
        series: &'a TimeSeriesData,
    ) -> Result<RouteCall<'a>, CausalError> {
        use AnalysisRoute as R;
        let route = classify_route(DataModality::TemporalOrEvent, &self.query)
            .ok_or_else(route_unsupported)?;
        let class = self.graph.class();
        Ok(match (route, &self.query) {
            (R::TemporalCrossWorld, _) => RouteCall::CrossWorld(series),
            (R::TemporalMediation, CausalQuery::Mediation(query)) => {
                RouteCall::SeriesMediation { series, query }
            }
            (R::TemporalEffect, CausalQuery::TemporalEffect(query)) => {
                RouteCall::SeriesEffect { series, query }
            }
            (R::TemporalResponse, CausalQuery::Response(query)) => {
                RouteCall::SeriesResponse { series, query }
            }
            (R::Association, _) => RouteCall::SeriesAssociation(series),
            (R::Anomaly | R::ChangeAttribution, _)
                if matches!(
                    class,
                    GraphClass::TemporalCpdag | GraphClass::TemporalPag | GraphClass::TemporalAdmg
                ) =>
            {
                RouteCall::ClassAttribution(series)
            }
            (
                R::Anomaly
                | R::ChangeAttribution
                | R::Attribute
                | R::UnitChange
                | R::MechanismChange
                | R::Simulate,
                _,
            ) if class == GraphClass::TemporalDag => RouteCall::SeriesAttribution(series),
            (R::TemporalMediation | R::TemporalEffect | R::TemporalResponse, _) => {
                return Err(route_mismatch());
            }
            _ => return Err(route_unsupported()),
        })
    }

    /// [`Self::route_call`] for a multi-unit panel.
    fn panel_route_call<'a>(&'a self, panel: &'a PanelData) -> Result<RouteCall<'a>, CausalError> {
        use AnalysisRoute as R;
        let route =
            classify_route(DataModality::Panel, &self.query).ok_or_else(route_unsupported)?;
        Ok(match (route, &self.query) {
            (R::PanelTemporalEffect, CausalQuery::TemporalEffect(query)) => {
                RouteCall::PanelEffect { panel, query }
            }
            (R::PanelTemporalResponse, CausalQuery::Response(query)) => {
                RouteCall::PanelResponse { panel, query }
            }
            (R::PanelTemporalEffect | R::PanelTemporalResponse, _) => {
                return Err(route_mismatch());
            }
            _ => return Err(route_unsupported()),
        })
    }

    /// [`Self::route_call`] for multi-environment series.
    fn multi_env_route_call<'a>(
        &'a self,
        multi: &'a MultiEnvironmentData,
    ) -> Result<RouteCall<'a>, CausalError> {
        use AnalysisRoute as R;
        let route =
            classify_route(DataModality::MultiEnv, &self.query).ok_or_else(route_unsupported)?;
        Ok(match (route, &self.query) {
            (R::MultiEnvTemporalEffect, CausalQuery::TemporalEffect(query)) => {
                RouteCall::MultiEnvEffect { multi, query }
            }
            (R::MultiEnvTemporalEffect, _) => return Err(route_mismatch()),
            _ => return Err(route_unsupported()),
        })
    }
}
