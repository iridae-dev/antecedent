//! One parse of a study into the payload compile and one-shot dispatch share.
//!
//! [`Study::route_input`] runs admission once. The enum carries the data, query,
//! and graph the old `(route, class)` ladders then re-opened with `unreachable!`
//! and `expect`. Consumers match the variant; they do not classify again.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    AnalysisRoute, CausalError, CausalQuery, DataClassification, DataInput, GraphClass, GraphRef,
    Study, classify_analysis_route, is_gcm_route,
};
use crate::graph::{Admg, Cpdag, Dag, Pag, TemporalDag};
use antecedent_core::{
    AverageEffectQuery, ConditionalEffectQuery, ContinuousDoseResponseQuery, InterferenceQuery,
    InterventionalDistributionQuery, LocalPolynomialRatioQuery, LongitudinalRegimeQuery,
    HistoryMatchQuery, SequentialContrastQuery,
    MediationQuery, NestedCounterfactualQuery, PanelDidQuery, PathSpecificEffectQuery,
    PolicyValueQuery, RandomizedEffectQuery, ResponseQuery, SurvivalQuery, SyntheticControlQuery,
    TemporalEffectQuery, TransportQuery,
};
use antecedent_data::{MultiEnvironmentData, PanelData, TabularData, TimeSeriesData};

/// Already-extracted inputs for one `(route, graph class)` cell.
///
/// `StaticAteMixed` is a physical-compile cell. [`Study::compile_logical`]
/// still refuses it, matching the logical ladder's previous catch-all.
#[derive(Clone, Copy)]
pub(crate) enum RouteInput<'a> {
    StaticResponseDag {
        data: &'a TabularData,
        query: &'a ResponseQuery,
        graph: &'a Dag,
    },
    StaticResponsePag {
        data: &'a TabularData,
        query: &'a ResponseQuery,
        pag: &'a Pag,
    },
    StaticResponseCpdag {
        data: &'a TabularData,
        query: &'a ResponseQuery,
        cpdag: &'a Cpdag,
    },
    CodeterminedJoint {
        data: &'a TabularData,
        query: &'a ResponseQuery,
        admg: &'a Admg,
    },
    MixedAdmgResponse {
        data: &'a TabularData,
        query: &'a ResponseQuery,
    },
    StaticAteDag {
        data: &'a TabularData,
        query: &'a AverageEffectQuery,
        graph: &'a Dag,
    },
    StaticAtePag {
        data: &'a TabularData,
        query: &'a AverageEffectQuery,
        pag: &'a Pag,
    },
    StaticAteCpdag {
        data: &'a TabularData,
        query: &'a AverageEffectQuery,
        cpdag: &'a Cpdag,
    },
    StaticAteMixed {
        data: &'a TabularData,
        query: &'a AverageEffectQuery,
    },
    DistributionDag {
        data: &'a TabularData,
        query: &'a InterventionalDistributionQuery,
        graph: &'a Dag,
    },
    DistributionMixed {
        data: &'a TabularData,
        query: &'a InterventionalDistributionQuery,
    },
    PathSpecificDag {
        data: &'a TabularData,
        query: &'a PathSpecificEffectQuery,
        graph: &'a Dag,
    },
    ConditionalDag {
        data: &'a TabularData,
        query: &'a ConditionalEffectQuery,
        graph: &'a Dag,
    },
    ConditionalCpdag {
        data: &'a TabularData,
        query: &'a ConditionalEffectQuery,
        cpdag: &'a Cpdag,
    },
    ConditionalPag {
        data: &'a TabularData,
        query: &'a ConditionalEffectQuery,
        pag: &'a Pag,
    },
    /// A conditional effect on a mixed graph. The cell covers both mixed classes so that the
    /// temporal one reaches its typed refusal instead of the generic combination refusal.
    ConditionalMixed {
        data: &'a TabularData,
        query: &'a ConditionalEffectQuery,
    },
    TemporalEffectClass {
        data: &'a TimeSeriesData,
        query: &'a TemporalEffectQuery,
        classification: DataClassification,
    },
    TemporalEffectDag {
        data: &'a TimeSeriesData,
        query: &'a TemporalEffectQuery,
        graph: &'a TemporalDag,
        classification: DataClassification,
    },
    TemporalResponseDag {
        data: &'a TimeSeriesData,
        query: &'a ResponseQuery,
        graph: &'a TemporalDag,
    },
    TemporalResponseClass {
        data: &'a TimeSeriesData,
        query: &'a ResponseQuery,
    },
    MultiEnvTemporalEffect {
        multi: &'a MultiEnvironmentData,
        query: &'a TemporalEffectQuery,
        graph: &'a TemporalDag,
    },
    PanelTemporalResponseDag {
        panel: &'a PanelData,
        query: &'a ResponseQuery,
        graph: &'a TemporalDag,
    },
    PanelTemporalResponseClass {
        panel: &'a PanelData,
        query: &'a ResponseQuery,
    },
    PanelTemporalEffectClass {
        panel: &'a PanelData,
        query: &'a TemporalEffectQuery,
    },
    PanelTemporalEffectDag {
        panel: &'a PanelData,
        query: &'a TemporalEffectQuery,
        graph: &'a TemporalDag,
    },
    TemporalMediationDag {
        data: &'a TimeSeriesData,
        query: &'a MediationQuery,
        graph: &'a TemporalDag,
    },
    TemporalMediationPag,
    TemporalMediationCpdag {
        data: &'a TimeSeriesData,
        query: &'a MediationQuery,
    },
    StaticMediationDag {
        data: &'a TabularData,
        graph: &'a Dag,
        source: StaticMediationSource<'a>,
    },
    TransportMixed {
        data: &'a TabularData,
        query: &'a TransportQuery,
        class: GraphClass,
    },
    InterferenceDag {
        data: &'a TabularData,
        query: &'a InterferenceQuery,
        graph: &'a Dag,
    },
    RandomizedEffect {
        data: &'a TabularData,
        query: &'a RandomizedEffectQuery,
    },
    PolicyValue {
        data: &'a TabularData,
        query: &'a PolicyValueQuery,
    },
    PanelDid {
        data: &'a TabularData,
        query: &'a PanelDidQuery,
    },
    SyntheticControl {
        data: &'a TabularData,
        query: &'a SyntheticControlQuery,
    },
    ContinuousDoseResponse {
        data: &'a TabularData,
        query: &'a ContinuousDoseResponseQuery,
    },
    LocalPolynomialRatio {
        data: &'a TabularData,
        query: &'a LocalPolynomialRatioQuery,
    },
    Survival {
        data: &'a TabularData,
        query: &'a SurvivalQuery,
    },
    LongitudinalRegime {
        data: &'a TabularData,
        query: &'a LongitudinalRegimeQuery,
    },
    SequentialContrast {
        data: &'a TabularData,
        query: &'a SequentialContrastQuery,
    },
    HistoryMatch {
        data: &'a TabularData,
        query: &'a HistoryMatchQuery,
    },
    GcmDag {
        data: &'a TabularData,
        route: AnalysisRoute,
        graph: &'a Dag,
    },
    GcmTemporal {
        data: &'a TimeSeriesData,
    },
    TemporalCrossWorldClass {
        data: &'a TimeSeriesData,
        class: GraphClass,
    },
    TemporalCrossWorldDag {
        data: &'a TimeSeriesData,
        graph: &'a TemporalDag,
    },
}

/// Static-mediation query already opened by [`Study::route_input_classified`].
#[derive(Clone, Copy)]
pub(crate) enum StaticMediationSource<'a> {
    Mediation(&'a MediationQuery),
    Nested(&'a NestedCounterfactualQuery),
}

impl Study {
    /// Admission plus the extracted route payload.
    ///
    /// # Errors
    ///
    /// The same modality, query, and combination failures as the previous
    /// `ensure_supported_combination` plus `(route, class)` catch-all.
    pub(super) fn route_input(&self) -> Result<RouteInput<'_>, CausalError> {
        self.ensure_supported_combination()?;
        self.route_input_classified()
            .ok_or(CausalError::Unsupported { message: "unsupported data/graph/query combination" })
    }

    /// Payload extraction without a second admission pass.
    ///
    /// `None` is the logical ladder's previous catch-all: the data, query, and
    /// graph class are not one of the compiled cells. One-shot dispatch uses
    /// this so a sealed route is not refused early by compile admission.
    pub(crate) fn route_input_classified(&self) -> Option<RouteInput<'_>> {
        let route = classify_analysis_route(&self.data, &self.query)?;
        let graph = self.graph.graph_ref();
        static_route_input(self, route, graph)
            .or_else(|| temporal_route_input(self, route, graph))
            .or_else(|| design_route_input(self, route, graph))
            .or_else(|| gcm_route_input(self, route, graph))
    }
}

/// Tabular static-graph cells: response, average effect, distribution, path-specific, conditional.
fn static_route_input<'a>(
    study: &'a Study,
    route: AnalysisRoute,
    graph: GraphRef<'a>,
) -> Option<RouteInput<'a>> {
    Some(match (route, graph) {
        (AnalysisRoute::Response, GraphRef::Dag(dag)) => RouteInput::StaticResponseDag {
            data: tabular(study),
            query: response_query(study),
            graph: dag,
        },
        (AnalysisRoute::Response, GraphRef::Pag(pag)) => RouteInput::StaticResponsePag {
            data: tabular(study),
            query: response_query(study),
            pag,
        },
        (AnalysisRoute::Response, GraphRef::Cpdag(cpdag)) => RouteInput::StaticResponseCpdag {
            data: tabular(study),
            query: response_query(study),
            cpdag,
        },
        (AnalysisRoute::Response, GraphRef::Admg(admg))
            if study.tiered.as_ref().is_some_and(|background| {
                background.within_tier == antecedent_graph::WithinTier::CoDetermined
            }) =>
        {
            RouteInput::CodeterminedJoint {
                data: tabular(study),
                query: response_query(study),
                admg,
            }
        }
        (AnalysisRoute::Response, GraphRef::Admg(_) | GraphRef::TemporalAdmg(_)) => {
            RouteInput::MixedAdmgResponse { data: tabular(study), query: response_query(study) }
        }
        (AnalysisRoute::StaticAte, GraphRef::Dag(dag)) => RouteInput::StaticAteDag {
            data: tabular(study),
            query: average_effect(study),
            graph: dag,
        },
        (AnalysisRoute::StaticAte, GraphRef::Pag(pag)) => {
            RouteInput::StaticAtePag { data: tabular(study), query: average_effect(study), pag }
        }
        (AnalysisRoute::StaticAte, GraphRef::Cpdag(cpdag)) => {
            RouteInput::StaticAteCpdag { data: tabular(study), query: average_effect(study), cpdag }
        }
        (AnalysisRoute::StaticAte, GraphRef::Admg(_) | GraphRef::TemporalAdmg(_)) => {
            RouteInput::StaticAteMixed { data: tabular(study), query: average_effect(study) }
        }
        (AnalysisRoute::Distribution, GraphRef::Dag(dag)) => RouteInput::DistributionDag {
            data: tabular(study),
            query: distribution(study),
            graph: dag,
        },
        (AnalysisRoute::Distribution, GraphRef::Admg(_) | GraphRef::TemporalAdmg(_)) => {
            RouteInput::DistributionMixed { data: tabular(study), query: distribution(study) }
        }
        (AnalysisRoute::PathSpecific, GraphRef::Dag(dag)) => RouteInput::PathSpecificDag {
            data: tabular(study),
            query: path_specific(study),
            graph: dag,
        },
        (AnalysisRoute::Conditional, GraphRef::Dag(dag)) => RouteInput::ConditionalDag {
            data: tabular(study),
            query: conditional(study),
            graph: dag,
        },
        (AnalysisRoute::Conditional, GraphRef::Cpdag(cpdag)) => {
            RouteInput::ConditionalCpdag { data: tabular(study), query: conditional(study), cpdag }
        }
        (AnalysisRoute::Conditional, GraphRef::Pag(pag)) => {
            RouteInput::ConditionalPag { data: tabular(study), query: conditional(study), pag }
        }
        (AnalysisRoute::Conditional, GraphRef::Admg(_) | GraphRef::TemporalAdmg(_)) => {
            RouteInput::ConditionalMixed { data: tabular(study), query: conditional(study) }
        }
        _ => return None,
    })
}

/// Series, panel, and multi-environment temporal cells, with temporal and static mediation.
fn temporal_route_input<'a>(
    study: &'a Study,
    route: AnalysisRoute,
    graph: GraphRef<'a>,
) -> Option<RouteInput<'a>> {
    Some(match (route, graph) {
        (AnalysisRoute::TemporalEffect, GraphRef::TemporalCpdag(_) | GraphRef::TemporalPag(_)) => {
            RouteInput::TemporalEffectClass {
                data: series(study),
                query: temporal_effect(study),
                classification: series_classification(study),
            }
        }
        (AnalysisRoute::TemporalEffect, GraphRef::TemporalDag(temporal_dag)) => {
            RouteInput::TemporalEffectDag {
                data: series(study),
                query: temporal_effect(study),
                graph: temporal_dag,
                classification: series_classification(study),
            }
        }
        (AnalysisRoute::TemporalResponse, GraphRef::TemporalDag(temporal_dag)) => {
            RouteInput::TemporalResponseDag {
                data: series(study),
                query: response_query(study),
                graph: temporal_dag,
            }
        }
        (
            AnalysisRoute::TemporalResponse,
            GraphRef::TemporalCpdag(_) | GraphRef::TemporalPag(_),
        ) => {
            RouteInput::TemporalResponseClass { data: series(study), query: response_query(study) }
        }
        (AnalysisRoute::MultiEnvTemporalEffect, GraphRef::TemporalDag(temporal_dag)) => {
            RouteInput::MultiEnvTemporalEffect {
                multi: multi_env(study),
                query: temporal_effect(study),
                graph: temporal_dag,
            }
        }
        (AnalysisRoute::PanelTemporalResponse, GraphRef::TemporalDag(temporal_dag)) => {
            RouteInput::PanelTemporalResponseDag {
                panel: panel(study),
                query: response_query(study),
                graph: temporal_dag,
            }
        }
        (
            AnalysisRoute::PanelTemporalResponse,
            GraphRef::TemporalCpdag(_) | GraphRef::TemporalPag(_),
        ) => RouteInput::PanelTemporalResponseClass {
            panel: panel(study),
            query: response_query(study),
        },
        (
            AnalysisRoute::PanelTemporalEffect,
            GraphRef::TemporalCpdag(_) | GraphRef::TemporalPag(_),
        ) => RouteInput::PanelTemporalEffectClass {
            panel: panel(study),
            query: temporal_effect(study),
        },
        (AnalysisRoute::PanelTemporalEffect, GraphRef::TemporalDag(temporal_dag)) => {
            RouteInput::PanelTemporalEffectDag {
                panel: panel(study),
                query: temporal_effect(study),
                graph: temporal_dag,
            }
        }
        (AnalysisRoute::TemporalMediation, GraphRef::TemporalDag(temporal_dag)) => {
            RouteInput::TemporalMediationDag {
                data: series(study),
                query: mediation(study),
                graph: temporal_dag,
            }
        }
        (AnalysisRoute::TemporalMediation, GraphRef::TemporalPag(_)) => {
            RouteInput::TemporalMediationPag
        }
        (AnalysisRoute::TemporalMediation, GraphRef::TemporalCpdag(_)) => {
            RouteInput::TemporalMediationCpdag { data: series(study), query: mediation(study) }
        }
        (AnalysisRoute::StaticMediation, GraphRef::Dag(dag)) => RouteInput::StaticMediationDag {
            data: tabular(study),
            graph: dag,
            source: static_mediation_source(study),
        },
        _ => return None,
    })
}

/// Mixed-graph transport and the design-based (randomized, quasi-experimental) cells.
fn design_route_input<'a>(
    study: &'a Study,
    route: AnalysisRoute,
    graph: GraphRef<'a>,
) -> Option<RouteInput<'a>> {
    let class = graph.class();
    Some(match (route, graph) {
        (AnalysisRoute::Transport, GraphRef::Admg(_) | GraphRef::TemporalAdmg(_)) => {
            RouteInput::TransportMixed { data: tabular(study), query: transport(study), class }
        }
        (AnalysisRoute::Interference, GraphRef::Dag(dag)) => RouteInput::InterferenceDag {
            data: tabular(study),
            query: interference(study),
            graph: dag,
        },
        (AnalysisRoute::RandomizedEffect, GraphRef::RandomizedTrial) => {
            RouteInput::RandomizedEffect { data: tabular(study), query: randomized_effect(study) }
        }
        (AnalysisRoute::PolicyValue, GraphRef::RandomizedTrial) => {
            RouteInput::PolicyValue { data: tabular(study), query: policy_value(study) }
        }
        (AnalysisRoute::PanelDid, GraphRef::RandomizedTrial) => {
            RouteInput::PanelDid { data: tabular(study), query: panel_did(study) }
        }
        (AnalysisRoute::SyntheticControl, GraphRef::RandomizedTrial) => {
            RouteInput::SyntheticControl { data: tabular(study), query: synthetic_control(study) }
        }
        (AnalysisRoute::ContinuousDoseResponse, GraphRef::RandomizedTrial) => {
            RouteInput::ContinuousDoseResponse {
                data: tabular(study),
                query: continuous_dose(study),
            }
        }
        (AnalysisRoute::LocalPolynomialRatio, GraphRef::RandomizedTrial) => {
            RouteInput::LocalPolynomialRatio {
                data: tabular(study),
                query: local_polynomial_ratio(study),
            }
        }
        (AnalysisRoute::Survival, GraphRef::RandomizedTrial) => {
            RouteInput::Survival { data: tabular(study), query: survival(study) }
        }
        (AnalysisRoute::LongitudinalRegime, GraphRef::RandomizedTrial) => {
            RouteInput::LongitudinalRegime {
                data: tabular(study),
                query: longitudinal_regime(study),
            }
        }
        (AnalysisRoute::HistoryMatch, GraphRef::RandomizedTrial) => {
            RouteInput::HistoryMatch { data: tabular(study), query: history_match(study) }
        }
        (AnalysisRoute::SequentialContrast, GraphRef::RandomizedTrial) => {
            RouteInput::SequentialContrast { data: tabular(study), query: sequential_contrast(study) }
        }
        _ => return None,
    })
}

/// Parametric SCM cells and temporal cross-world cells.
fn gcm_route_input<'a>(
    study: &'a Study,
    route: AnalysisRoute,
    graph: GraphRef<'a>,
) -> Option<RouteInput<'a>> {
    let class = graph.class();
    Some(match (route, graph) {
        (route, GraphRef::Dag(dag)) if is_gcm_route(route) => {
            RouteInput::GcmDag { data: tabular(study), route, graph: dag }
        }
        (AnalysisRoute::Anomaly | AnalysisRoute::ChangeAttribution, GraphRef::TemporalDag(_)) => {
            RouteInput::GcmTemporal { data: series(study) }
        }
        (
            AnalysisRoute::TemporalCrossWorld,
            GraphRef::TemporalCpdag(_) | GraphRef::TemporalPag(_),
        ) => RouteInput::TemporalCrossWorldClass { data: series(study), class },
        (AnalysisRoute::TemporalCrossWorld, GraphRef::TemporalDag(temporal_dag)) => {
            RouteInput::TemporalCrossWorldDag { data: series(study), graph: temporal_dag }
        }
        _ => return None,
    })
}

fn tabular(study: &Study) -> &TabularData {
    let DataInput::Tabular(data) = &study.data else { unreachable!() };
    data
}

fn series(study: &Study) -> &TimeSeriesData {
    let (DataInput::Temporal(data) | DataInput::Event(data)) = &study.data else { unreachable!() };
    data
}

fn series_classification(study: &Study) -> DataClassification {
    match &study.data {
        DataInput::Event(_) => DataClassification::Event,
        _ => DataClassification::Temporal,
    }
}

fn panel(study: &Study) -> &PanelData {
    let DataInput::Panel(data) = &study.data else { unreachable!() };
    data
}

fn multi_env(study: &Study) -> &MultiEnvironmentData {
    let DataInput::MultiEnv(data) = &study.data else { unreachable!() };
    data
}

fn response_query(study: &Study) -> &ResponseQuery {
    let CausalQuery::Response(query) = &study.query else { unreachable!() };
    query
}

fn average_effect(study: &Study) -> &AverageEffectQuery {
    let CausalQuery::AverageEffect(query) = &study.query else { unreachable!() };
    query
}

fn distribution(study: &Study) -> &InterventionalDistributionQuery {
    let CausalQuery::Distribution(query) = &study.query else { unreachable!() };
    query
}

fn path_specific(study: &Study) -> &PathSpecificEffectQuery {
    let CausalQuery::PathSpecific(query) = &study.query else { unreachable!() };
    query
}

fn conditional(study: &Study) -> &ConditionalEffectQuery {
    let CausalQuery::ConditionalEffect(query) = &study.query else { unreachable!() };
    query
}

fn temporal_effect(study: &Study) -> &TemporalEffectQuery {
    let CausalQuery::TemporalEffect(query) = &study.query else { unreachable!() };
    query
}

fn mediation(study: &Study) -> &MediationQuery {
    let CausalQuery::Mediation(query) = &study.query else { unreachable!() };
    query
}

fn transport(study: &Study) -> &TransportQuery {
    let CausalQuery::Transport(query) = &study.query else { unreachable!() };
    query
}

fn interference(study: &Study) -> &InterferenceQuery {
    let CausalQuery::Interference(query) = &study.query else { unreachable!() };
    query
}

fn randomized_effect(study: &Study) -> &RandomizedEffectQuery {
    let CausalQuery::RandomizedEffect(query) = &study.query else { unreachable!() };
    query
}

fn policy_value(study: &Study) -> &PolicyValueQuery {
    let CausalQuery::PolicyValue(query) = &study.query else { unreachable!() };
    query
}

fn panel_did(study: &Study) -> &PanelDidQuery {
    let CausalQuery::PanelDid(query) = &study.query else { unreachable!() };
    query
}

fn synthetic_control(study: &Study) -> &SyntheticControlQuery {
    let CausalQuery::SyntheticControl(query) = &study.query else { unreachable!() };
    query
}

fn continuous_dose(study: &Study) -> &ContinuousDoseResponseQuery {
    let CausalQuery::ContinuousDoseResponse(query) = &study.query else { unreachable!() };
    query
}

fn local_polynomial_ratio(study: &Study) -> &LocalPolynomialRatioQuery {
    let CausalQuery::LocalPolynomialRatio(query) = &study.query else { unreachable!() };
    query
}

fn survival(study: &Study) -> &SurvivalQuery {
    let CausalQuery::Survival(query) = &study.query else { unreachable!() };
    query
}

fn history_match(study: &Study) -> &HistoryMatchQuery {
    let CausalQuery::HistoryMatch(query) = &study.query else { unreachable!() };
    query
}

fn sequential_contrast(study: &Study) -> &SequentialContrastQuery {
    let CausalQuery::SequentialContrast(query) = &study.query else { unreachable!() };
    query
}

fn longitudinal_regime(study: &Study) -> &LongitudinalRegimeQuery {
    let CausalQuery::LongitudinalRegime(query) = &study.query else { unreachable!() };
    query
}

fn static_mediation_source(study: &Study) -> StaticMediationSource<'_> {
    match &study.query {
        CausalQuery::Mediation(query) => StaticMediationSource::Mediation(query),
        CausalQuery::NestedCounterfactual(query) => StaticMediationSource::Nested(query),
        _ => unreachable!(),
    }
}
