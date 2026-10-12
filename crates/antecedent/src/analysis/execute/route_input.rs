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
        self.route_input_classified().ok_or_else(|| CausalError::Unsupported {
            message: "unsupported data/graph/query combination",
        })
    }

    /// Payload extraction without a second admission pass.
    ///
    /// `None` is the logical ladder's previous catch-all: the data, query, and
    /// graph class are not one of the compiled cells. One-shot dispatch uses
    /// this so a sealed route is not refused early by compile admission.
    pub(crate) fn route_input_classified(&self) -> Option<RouteInput<'_>> {
        let route = classify_analysis_route(&self.data, &self.query)?;
        let graph = self.graph.graph_ref();
        let class = graph.class();
        Some(match (route, graph) {
            (AnalysisRoute::Response, GraphRef::Dag(dag)) => RouteInput::StaticResponseDag {
                data: tabular(self),
                query: response_query(self),
                graph: dag,
            },
            (AnalysisRoute::Response, GraphRef::Pag(pag)) => RouteInput::StaticResponsePag {
                data: tabular(self),
                query: response_query(self),
                pag,
            },
            (AnalysisRoute::Response, GraphRef::Cpdag(cpdag)) => RouteInput::StaticResponseCpdag {
                data: tabular(self),
                query: response_query(self),
                cpdag,
            },
            (AnalysisRoute::Response, GraphRef::Admg(admg))
                if self.tiered.as_ref().is_some_and(|background| {
                    background.within_tier == antecedent_graph::WithinTier::CoDetermined
                }) =>
            {
                RouteInput::CodeterminedJoint {
                    data: tabular(self),
                    query: response_query(self),
                    admg,
                }
            }
            (AnalysisRoute::Response, GraphRef::Admg(_) | GraphRef::TemporalAdmg(_)) => {
                RouteInput::MixedAdmgResponse { data: tabular(self), query: response_query(self) }
            }
            (AnalysisRoute::StaticAte, GraphRef::Dag(dag)) => RouteInput::StaticAteDag {
                data: tabular(self),
                query: average_effect(self),
                graph: dag,
            },
            (AnalysisRoute::StaticAte, GraphRef::Pag(pag)) => {
                RouteInput::StaticAtePag { data: tabular(self), query: average_effect(self), pag }
            }
            (AnalysisRoute::StaticAte, GraphRef::Cpdag(cpdag)) => RouteInput::StaticAteCpdag {
                data: tabular(self),
                query: average_effect(self),
                cpdag,
            },
            (AnalysisRoute::StaticAte, GraphRef::Admg(_) | GraphRef::TemporalAdmg(_)) => {
                RouteInput::StaticAteMixed { data: tabular(self), query: average_effect(self) }
            }
            (AnalysisRoute::Distribution, GraphRef::Dag(dag)) => RouteInput::DistributionDag {
                data: tabular(self),
                query: distribution(self),
                graph: dag,
            },
            (AnalysisRoute::Distribution, GraphRef::Admg(_) | GraphRef::TemporalAdmg(_)) => {
                RouteInput::DistributionMixed { data: tabular(self), query: distribution(self) }
            }
            (AnalysisRoute::PathSpecific, GraphRef::Dag(dag)) => RouteInput::PathSpecificDag {
                data: tabular(self),
                query: path_specific(self),
                graph: dag,
            },
            (AnalysisRoute::Conditional, GraphRef::Dag(dag)) => RouteInput::ConditionalDag {
                data: tabular(self),
                query: conditional(self),
                graph: dag,
            },
            (AnalysisRoute::Conditional, GraphRef::Cpdag(cpdag)) => RouteInput::ConditionalCpdag {
                data: tabular(self),
                query: conditional(self),
                cpdag,
            },
            (AnalysisRoute::Conditional, GraphRef::Pag(pag)) => {
                RouteInput::ConditionalPag { data: tabular(self), query: conditional(self), pag }
            }
            (
                AnalysisRoute::TemporalEffect,
                GraphRef::TemporalCpdag(_) | GraphRef::TemporalPag(_),
            ) => RouteInput::TemporalEffectClass {
                data: series(self),
                query: temporal_effect(self),
                classification: series_classification(self),
            },
            (AnalysisRoute::TemporalEffect, GraphRef::TemporalDag(temporal_dag)) => {
                RouteInput::TemporalEffectDag {
                    data: series(self),
                    query: temporal_effect(self),
                    graph: temporal_dag,
                    classification: series_classification(self),
                }
            }
            (AnalysisRoute::TemporalResponse, GraphRef::TemporalDag(temporal_dag)) => {
                RouteInput::TemporalResponseDag {
                    data: series(self),
                    query: response_query(self),
                    graph: temporal_dag,
                }
            }
            (
                AnalysisRoute::TemporalResponse,
                GraphRef::TemporalCpdag(_) | GraphRef::TemporalPag(_),
            ) => RouteInput::TemporalResponseClass {
                data: series(self),
                query: response_query(self),
            },
            (AnalysisRoute::MultiEnvTemporalEffect, GraphRef::TemporalDag(temporal_dag)) => {
                RouteInput::MultiEnvTemporalEffect {
                    multi: multi_env(self),
                    query: temporal_effect(self),
                    graph: temporal_dag,
                }
            }
            (AnalysisRoute::PanelTemporalResponse, GraphRef::TemporalDag(temporal_dag)) => {
                RouteInput::PanelTemporalResponseDag {
                    panel: panel(self),
                    query: response_query(self),
                    graph: temporal_dag,
                }
            }
            (
                AnalysisRoute::PanelTemporalResponse,
                GraphRef::TemporalCpdag(_) | GraphRef::TemporalPag(_),
            ) => RouteInput::PanelTemporalResponseClass {
                panel: panel(self),
                query: response_query(self),
            },
            (
                AnalysisRoute::PanelTemporalEffect,
                GraphRef::TemporalCpdag(_) | GraphRef::TemporalPag(_),
            ) => RouteInput::PanelTemporalEffectClass {
                panel: panel(self),
                query: temporal_effect(self),
            },
            (AnalysisRoute::PanelTemporalEffect, GraphRef::TemporalDag(temporal_dag)) => {
                RouteInput::PanelTemporalEffectDag {
                    panel: panel(self),
                    query: temporal_effect(self),
                    graph: temporal_dag,
                }
            }
            (AnalysisRoute::TemporalMediation, GraphRef::TemporalDag(temporal_dag)) => {
                RouteInput::TemporalMediationDag {
                    data: series(self),
                    query: mediation(self),
                    graph: temporal_dag,
                }
            }
            (AnalysisRoute::TemporalMediation, GraphRef::TemporalPag(_)) => {
                RouteInput::TemporalMediationPag
            }
            (AnalysisRoute::TemporalMediation, GraphRef::TemporalCpdag(_)) => {
                RouteInput::TemporalMediationCpdag { data: series(self), query: mediation(self) }
            }
            (AnalysisRoute::StaticMediation, GraphRef::Dag(dag)) => {
                RouteInput::StaticMediationDag {
                    data: tabular(self),
                    graph: dag,
                    source: static_mediation_source(self),
                }
            }
            (AnalysisRoute::Transport, GraphRef::Admg(_) | GraphRef::TemporalAdmg(_)) => {
                RouteInput::TransportMixed { data: tabular(self), query: transport(self), class }
            }
            (AnalysisRoute::Interference, GraphRef::Dag(dag)) => RouteInput::InterferenceDag {
                data: tabular(self),
                query: interference(self),
                graph: dag,
            },
            (AnalysisRoute::RandomizedEffect, GraphRef::RandomizedTrial) => {
                RouteInput::RandomizedEffect { data: tabular(self), query: randomized_effect(self) }
            }
            (AnalysisRoute::PolicyValue, GraphRef::RandomizedTrial) => {
                RouteInput::PolicyValue { data: tabular(self), query: policy_value(self) }
            }
            (AnalysisRoute::PanelDid, GraphRef::RandomizedTrial) => {
                RouteInput::PanelDid { data: tabular(self), query: panel_did(self) }
            }
            (AnalysisRoute::SyntheticControl, GraphRef::RandomizedTrial) => {
                RouteInput::SyntheticControl { data: tabular(self), query: synthetic_control(self) }
            }
            (AnalysisRoute::ContinuousDoseResponse, GraphRef::RandomizedTrial) => {
                RouteInput::ContinuousDoseResponse {
                    data: tabular(self),
                    query: continuous_dose(self),
                }
            }
            (AnalysisRoute::LocalPolynomialRatio, GraphRef::RandomizedTrial) => {
                RouteInput::LocalPolynomialRatio {
                    data: tabular(self),
                    query: local_polynomial_ratio(self),
                }
            }
            (AnalysisRoute::Survival, GraphRef::RandomizedTrial) => {
                RouteInput::Survival { data: tabular(self), query: survival(self) }
            }
            (AnalysisRoute::LongitudinalRegime, GraphRef::RandomizedTrial) => {
                RouteInput::LongitudinalRegime {
                    data: tabular(self),
                    query: longitudinal_regime(self),
                }
            }
            (route, GraphRef::Dag(dag)) if is_gcm_route(route) => {
                RouteInput::GcmDag { data: tabular(self), route, graph: dag }
            }
            (
                AnalysisRoute::Anomaly | AnalysisRoute::ChangeAttribution,
                GraphRef::TemporalDag(_),
            ) => RouteInput::GcmTemporal { data: series(self) },
            (
                AnalysisRoute::TemporalCrossWorld,
                GraphRef::TemporalCpdag(_) | GraphRef::TemporalPag(_),
            ) => RouteInput::TemporalCrossWorldClass { data: series(self), class },
            (AnalysisRoute::TemporalCrossWorld, GraphRef::TemporalDag(temporal_dag)) => {
                RouteInput::TemporalCrossWorldDag { data: series(self), graph: temporal_dag }
            }
            _ => return None,
        })
    }
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
