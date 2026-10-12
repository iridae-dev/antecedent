//! One parse of a study into the payload compile and one-shot dispatch share.
//!
//! [`Study::route_input`] runs admission once. The enum carries the data, query,
//! and graph the old `(route, class)` ladders then re-opened with `unreachable!`
//! and `expect`. Consumers match the variant; they do not classify again.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    AnalysisRoute, CausalError, CausalQuery, DataClassification, DataInput, GraphClass, Study,
    classify_analysis_route, is_gcm_route,
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
        let class = self.graph.class();
        Some(match (route, class) {
            (AnalysisRoute::Response, GraphClass::Dag) => RouteInput::StaticResponseDag {
                data: tabular(self),
                query: response_query(self),
                graph: dag(self),
            },
            (AnalysisRoute::Response, GraphClass::Pag) => RouteInput::StaticResponsePag {
                data: tabular(self),
                query: response_query(self),
                pag: pag(self),
            },
            (AnalysisRoute::Response, GraphClass::Cpdag) => RouteInput::StaticResponseCpdag {
                data: tabular(self),
                query: response_query(self),
                cpdag: cpdag(self),
            },
            (AnalysisRoute::Response, GraphClass::Admg)
                if self.tiered.as_ref().is_some_and(|background| {
                    background.within_tier == antecedent_graph::WithinTier::CoDetermined
                }) =>
            {
                RouteInput::CodeterminedJoint {
                    data: tabular(self),
                    query: response_query(self),
                    admg: admg(self),
                }
            }
            (AnalysisRoute::Response, GraphClass::Admg | GraphClass::TemporalAdmg) => {
                RouteInput::MixedAdmgResponse { data: tabular(self), query: response_query(self) }
            }
            (AnalysisRoute::StaticAte, GraphClass::Dag) => RouteInput::StaticAteDag {
                data: tabular(self),
                query: average_effect(self),
                graph: dag(self),
            },
            (AnalysisRoute::StaticAte, GraphClass::Pag) => RouteInput::StaticAtePag {
                data: tabular(self),
                query: average_effect(self),
                pag: pag(self),
            },
            (AnalysisRoute::StaticAte, GraphClass::Cpdag) => RouteInput::StaticAteCpdag {
                data: tabular(self),
                query: average_effect(self),
                cpdag: cpdag(self),
            },
            (AnalysisRoute::StaticAte, GraphClass::Admg | GraphClass::TemporalAdmg) => {
                RouteInput::StaticAteMixed { data: tabular(self), query: average_effect(self) }
            }
            (AnalysisRoute::Distribution, GraphClass::Dag) => RouteInput::DistributionDag {
                data: tabular(self),
                query: distribution(self),
                graph: dag(self),
            },
            (AnalysisRoute::Distribution, GraphClass::Admg | GraphClass::TemporalAdmg) => {
                RouteInput::DistributionMixed { data: tabular(self), query: distribution(self) }
            }
            (AnalysisRoute::PathSpecific, GraphClass::Dag) => RouteInput::PathSpecificDag {
                data: tabular(self),
                query: path_specific(self),
                graph: dag(self),
            },
            (AnalysisRoute::Conditional, GraphClass::Dag) => RouteInput::ConditionalDag {
                data: tabular(self),
                query: conditional(self),
                graph: dag(self),
            },
            (AnalysisRoute::Conditional, GraphClass::Cpdag) => RouteInput::ConditionalCpdag {
                data: tabular(self),
                query: conditional(self),
                cpdag: cpdag(self),
            },
            (AnalysisRoute::Conditional, GraphClass::Pag) => RouteInput::ConditionalPag {
                data: tabular(self),
                query: conditional(self),
                pag: pag(self),
            },
            (
                AnalysisRoute::TemporalEffect,
                GraphClass::TemporalCpdag | GraphClass::TemporalPag,
            ) => RouteInput::TemporalEffectClass {
                data: series(self),
                query: temporal_effect(self),
                classification: series_classification(self),
            },
            (AnalysisRoute::TemporalEffect, GraphClass::TemporalDag) => {
                RouteInput::TemporalEffectDag {
                    data: series(self),
                    query: temporal_effect(self),
                    graph: temporal_dag(self),
                    classification: series_classification(self),
                }
            }
            (AnalysisRoute::TemporalResponse, GraphClass::TemporalDag) => {
                RouteInput::TemporalResponseDag {
                    data: series(self),
                    query: response_query(self),
                    graph: temporal_dag(self),
                }
            }
            (
                AnalysisRoute::TemporalResponse,
                GraphClass::TemporalCpdag | GraphClass::TemporalPag,
            ) => RouteInput::TemporalResponseClass {
                data: series(self),
                query: response_query(self),
            },
            (AnalysisRoute::MultiEnvTemporalEffect, GraphClass::TemporalDag) => {
                RouteInput::MultiEnvTemporalEffect {
                    multi: multi_env(self),
                    query: temporal_effect(self),
                    graph: temporal_dag(self),
                }
            }
            (AnalysisRoute::PanelTemporalResponse, GraphClass::TemporalDag) => {
                RouteInput::PanelTemporalResponseDag {
                    panel: panel(self),
                    query: response_query(self),
                    graph: temporal_dag(self),
                }
            }
            (
                AnalysisRoute::PanelTemporalResponse,
                GraphClass::TemporalCpdag | GraphClass::TemporalPag,
            ) => RouteInput::PanelTemporalResponseClass {
                panel: panel(self),
                query: response_query(self),
            },
            (
                AnalysisRoute::PanelTemporalEffect,
                GraphClass::TemporalCpdag | GraphClass::TemporalPag,
            ) => RouteInput::PanelTemporalEffectClass {
                panel: panel(self),
                query: temporal_effect(self),
            },
            (AnalysisRoute::PanelTemporalEffect, GraphClass::TemporalDag) => {
                RouteInput::PanelTemporalEffectDag {
                    panel: panel(self),
                    query: temporal_effect(self),
                    graph: temporal_dag(self),
                }
            }
            (AnalysisRoute::TemporalMediation, GraphClass::TemporalDag) => {
                RouteInput::TemporalMediationDag {
                    data: series(self),
                    query: mediation(self),
                    graph: temporal_dag(self),
                }
            }
            (AnalysisRoute::TemporalMediation, GraphClass::TemporalPag) => {
                RouteInput::TemporalMediationPag
            }
            (AnalysisRoute::TemporalMediation, GraphClass::TemporalCpdag) => {
                RouteInput::TemporalMediationCpdag { data: series(self), query: mediation(self) }
            }
            (AnalysisRoute::StaticMediation, GraphClass::Dag) => RouteInput::StaticMediationDag {
                data: tabular(self),
                graph: dag(self),
                source: static_mediation_source(self),
            },
            (AnalysisRoute::Transport, GraphClass::Admg | GraphClass::TemporalAdmg) => {
                RouteInput::TransportMixed { data: tabular(self), query: transport(self), class }
            }
            (AnalysisRoute::Interference, GraphClass::Dag) => RouteInput::InterferenceDag {
                data: tabular(self),
                query: interference(self),
                graph: dag(self),
            },
            (AnalysisRoute::RandomizedEffect, GraphClass::RandomizedTrial) => {
                RouteInput::RandomizedEffect { data: tabular(self), query: randomized_effect(self) }
            }
            (AnalysisRoute::PolicyValue, GraphClass::RandomizedTrial) => {
                RouteInput::PolicyValue { data: tabular(self), query: policy_value(self) }
            }
            (AnalysisRoute::PanelDid, GraphClass::RandomizedTrial) => {
                RouteInput::PanelDid { data: tabular(self), query: panel_did(self) }
            }
            (AnalysisRoute::SyntheticControl, GraphClass::RandomizedTrial) => {
                RouteInput::SyntheticControl { data: tabular(self), query: synthetic_control(self) }
            }
            (AnalysisRoute::ContinuousDoseResponse, GraphClass::RandomizedTrial) => {
                RouteInput::ContinuousDoseResponse {
                    data: tabular(self),
                    query: continuous_dose(self),
                }
            }
            (AnalysisRoute::LocalPolynomialRatio, GraphClass::RandomizedTrial) => {
                RouteInput::LocalPolynomialRatio {
                    data: tabular(self),
                    query: local_polynomial_ratio(self),
                }
            }
            (AnalysisRoute::Survival, GraphClass::RandomizedTrial) => {
                RouteInput::Survival { data: tabular(self), query: survival(self) }
            }
            (AnalysisRoute::LongitudinalRegime, GraphClass::RandomizedTrial) => {
                RouteInput::LongitudinalRegime {
                    data: tabular(self),
                    query: longitudinal_regime(self),
                }
            }
            (route, GraphClass::Dag) if is_gcm_route(route) => {
                RouteInput::GcmDag { data: tabular(self), route, graph: dag(self) }
            }
            (
                AnalysisRoute::Anomaly | AnalysisRoute::ChangeAttribution,
                GraphClass::TemporalDag,
            ) => RouteInput::GcmTemporal { data: series(self) },
            (
                AnalysisRoute::TemporalCrossWorld,
                GraphClass::TemporalCpdag | GraphClass::TemporalPag,
            ) => RouteInput::TemporalCrossWorldClass { data: series(self), class },
            (AnalysisRoute::TemporalCrossWorld, GraphClass::TemporalDag) => {
                RouteInput::TemporalCrossWorldDag { data: series(self), graph: temporal_dag(self) }
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

fn dag(study: &Study) -> &Dag {
    study.graph.as_dag().expect("class() == Dag implies as_dag() is Some")
}

fn pag(study: &Study) -> &Pag {
    study.graph.as_pag().expect("class() == Pag implies as_pag() is Some")
}

fn cpdag(study: &Study) -> &Cpdag {
    study.graph.as_cpdag().expect("class() == Cpdag implies as_cpdag() is Some")
}

fn admg(study: &Study) -> &Admg {
    study.graph.as_admg().expect("class() == Admg implies as_admg() is Some")
}

fn temporal_dag(study: &Study) -> &TemporalDag {
    study.graph.as_temporal_dag().expect("class() == TemporalDag implies as_temporal_dag() is Some")
}
