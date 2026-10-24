//! The sealer table: which sealed operation `Study::prepare` retains.
//!
//! Preparation offers the study to each [`Sealer`] in [`SEALERS`] order and
//! keeps the first operation one of them returns. A sealer answers `Ok(None)`
//! when its route does not admit the study, and an error only when its route
//! admits the study and cannot be sealed for it. The table is the whole
//! selection rule: there is no other place where one route outranks another.
//!
//! # Priority
//!
//! The order is the priority and it is part of the contract of preparation.
//! Routes whose admissions are disjoint may stand in any relative order; the
//! order below is the one the previous hand-written chain used, kept so that no
//! study changes route. Where two admissions genuinely overlap, the later
//! sealer declares it in [`Sealer::yields_to`] and the earlier one wins:
//!
//! - [`ExecutionKind::StaticResponseCurve`] admits any frequentist DAG mean
//!   curve at the default response procedure, which includes the studies the
//!   static DAG response operation admits first.
//!
//! Debug builds offer the study to the sealers after the winner too and refuse
//! (by assertion) an admission nobody declared, so a new route that steals
//! studies from an existing one fails the first prepare that exercises it
//! instead of silently changing which operation runs.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod adjustment;
mod bayesian;
mod design;
mod estimator;
mod posterior;
mod program;
mod response;
mod route;
mod static_query;
mod temporal;

use super::{ExecutionKind, PreparedExecution};
use crate::analysis::execute::{RouteInput, StaticMediationSource, Study};
use crate::error::CausalError;
use crate::graph::Dag;
use crate::planner::{PhysicalExecutionPlan, PlanBinding};
use antecedent_core::{
    AverageEffectQuery, ConditionalEffectQuery, ExecutionContext, InterventionalDistributionQuery,
    ResponseQuery,
};
use antecedent_data::TabularData;

/// Everything a sealer reads to decide admission and to build its operation.
#[derive(Clone, Copy)]
pub(super) struct SealContext<'a> {
    /// The study `prepare` was called on.
    pub(super) source: &'a Study,
    /// The study with the prepare-time identification caches installed. It is
    /// the study every retained operation is built from.
    pub(super) analysis: &'a Study,
    /// The physical plan compiled once for this prepare.
    pub(super) plan: &'a std::sync::Arc<PhysicalExecutionPlan>,
    /// The strategy names `plan` carries, parsed once.
    pub(super) binding: &'a PlanBinding,
    pub(super) ctx: &'a ExecutionContext,
    /// The data, query and graph payload of `source`, parsed once by the
    /// admission compile ran: `None` is a study no compiled cell covers.
    ///
    /// A sealer reads the cell it serves from here instead of re-deriving the
    /// `(data, query, graph class)` triple; the gates that are about the
    /// retained proof (identification cache) or the plan record stay in the
    /// sealer. The graph-posterior and class-envelope sealers and the
    /// attribution and class-mediation sealers do not read it: the first two
    /// groups are admitted by `compile_graph_posterior` and by their own
    /// `admits`, and the cell of an attribution carries no query.
    pub(super) route: Option<RouteInput<'a>>,
    /// The compiled cross-world operation of a tabular counterfactual query.
    pub(super) counterfactual: Option<&'a crate::gcm::CheckedCounterfactualOperation>,
    /// The compiled cross-world operation of a nested counterfactual query.
    pub(super) nested_counterfactual: Option<&'a crate::gcm::NestedCounterfactualOperation>,
}

impl<'a> SealContext<'a> {
    /// The data and query of a tabular average-effect study over a DAG.
    pub(super) fn static_ate_dag(&self) -> Option<(&'a TabularData, &'a AverageEffectQuery)> {
        match self.route {
            Some(RouteInput::StaticAteDag { data, query, .. }) => Some((data, query)),
            _ => None,
        }
    }

    /// The query of a tabular average-effect study over any graph class the
    /// static average-effect route compiles.
    pub(super) fn static_ate(&self) -> Option<&'a AverageEffectQuery> {
        match self.route {
            Some(
                RouteInput::StaticAteDag { query, .. }
                | RouteInput::StaticAtePag { query, .. }
                | RouteInput::StaticAteCpdag { query, .. }
                | RouteInput::StaticAteMixed { query, .. },
            ) => Some(query),
            _ => None,
        }
    }

    /// The query and graph of a tabular static response study over a DAG.
    pub(super) fn static_response_dag(&self) -> Option<(&'a ResponseQuery, &'a Dag)> {
        match self.route {
            Some(RouteInput::StaticResponseDag { query, graph, .. }) => Some((query, graph)),
            _ => None,
        }
    }

    /// The data and query of a tabular conditional-effect study over a DAG.
    pub(super) fn conditional_dag(&self) -> Option<(&'a TabularData, &'a ConditionalEffectQuery)> {
        match self.route {
            Some(RouteInput::ConditionalDag { data, query, .. }) => Some((data, query)),
            _ => None,
        }
    }

    /// The data and query of a tabular conditional-effect study over a DAG,
    /// CPDAG or PAG.
    pub(super) fn conditional(&self) -> Option<(&'a TabularData, &'a ConditionalEffectQuery)> {
        match self.route {
            Some(
                RouteInput::ConditionalDag { data, query, .. }
                | RouteInput::ConditionalCpdag { data, query, .. }
                | RouteInput::ConditionalPag { data, query, .. },
            ) => Some((data, query)),
            _ => None,
        }
    }

    /// The data and query of a tabular interventional-distribution study over
    /// a DAG or a mixed graph.
    pub(super) fn distribution(
        &self,
    ) -> Option<(&'a TabularData, &'a InterventionalDistributionQuery)> {
        match self.route {
            Some(
                RouteInput::DistributionDag { data, query, .. }
                | RouteInput::DistributionMixed { data, query },
            ) => Some((data, query)),
            _ => None,
        }
    }

    /// The data of a tabular static mediation study over a DAG (a nested
    /// counterfactual shares the cell and is not a mediation query).
    pub(super) fn static_mediation_dag(&self) -> Option<&'a TabularData> {
        match self.route {
            Some(RouteInput::StaticMediationDag {
                data,
                source: StaticMediationSource::Mediation(_),
                ..
            }) => Some(data),
            _ => None,
        }
    }
}

/// What one sealer answers: no admission, an admission and its operation, or
/// the refusal of an admitted study.
pub(super) type SealOutcome = Result<Option<PreparedExecution>, CausalError>;

/// One entry of the sealer table.
pub(super) struct Sealer {
    /// The route this sealer retains. Every sealed [`ExecutionKind`] has
    /// exactly one entry.
    #[cfg(any(test, debug_assertions))]
    pub(super) kind: ExecutionKind,
    pub(super) seal: fn(&SealContext<'_>) -> SealOutcome,
    /// Earlier routes this one is allowed to overlap. A study both admit goes
    /// to the earlier route; the overlap is intended, not an accident.
    #[cfg(any(test, debug_assertions))]
    pub(super) yields_to: &'static [ExecutionKind],
}

const fn sealer(kind: ExecutionKind, seal: fn(&SealContext<'_>) -> SealOutcome) -> Sealer {
    #[cfg(not(any(test, debug_assertions)))]
    let _ = kind;
    Sealer {
        #[cfg(any(test, debug_assertions))]
        kind,
        seal,
        #[cfg(any(test, debug_assertions))]
        yields_to: &[],
    }
}

/// Every sealer, highest priority first. One entry per sealed
/// [`ExecutionKind`]: the array length is that count, so a route added to the
/// execution list without an entry here does not compile.
pub(super) const SEALERS: [Sealer; ExecutionKind::SEALED_COUNT] = [
    sealer(ExecutionKind::UnknownTieredAverage, adjustment::unknown_tiered_average),
    sealer(ExecutionKind::CheckedLinear, adjustment::checked_linear),
    sealer(ExecutionKind::CheckedGlmAdjustment, adjustment::checked_glm_adjustment),
    sealer(ExecutionKind::CheckedRd, adjustment::checked_rd),
    sealer(ExecutionKind::CheckedPropensity, adjustment::checked_propensity),
    sealer(ExecutionKind::CheckedConditional, adjustment::checked_conditional),
    sealer(ExecutionKind::GraphPosteriorEffect, posterior::graph_posterior_effect),
    sealer(ExecutionKind::ClassGraphPosteriorEffect, posterior::class_graph_posterior_effect),
    sealer(ExecutionKind::BayesianGraphPosteriorAte, posterior::bayesian_graph_posterior_ate),
    // An unknown within-tier order is also a class study, so the class-effect
    // sealer admits it too; the unknown-tiered average has always won it.
    Sealer {
        #[cfg(any(test, debug_assertions))]
        kind: ExecutionKind::StaticClassEffect,
        seal: posterior::static_class_effect,
        #[cfg(any(test, debug_assertions))]
        yields_to: &[ExecutionKind::UnknownTieredAverage],
    },
    sealer(ExecutionKind::CheckedAipw, adjustment::checked_aipw),
    sealer(ExecutionKind::FrontDoorLinear, adjustment::frontdoor_linear),
    sealer(ExecutionKind::Iv, adjustment::iv),
    sealer(ExecutionKind::Counterfactual, static_query::counterfactual),
    sealer(ExecutionKind::NestedCounterfactual, static_query::nested_counterfactual),
    sealer(ExecutionKind::DerivativeResponse, response::derivative_response),
    sealer(ExecutionKind::CellAipwResponse, response::cell_aipw_response),
    sealer(ExecutionKind::StaticDagResponse, response::static_dag_response),
    sealer(ExecutionKind::BayesianStaticMediation, static_query::bayesian_static_mediation),
    sealer(ExecutionKind::StaticMediation, static_query::static_mediation),
    sealer(ExecutionKind::Attribution, static_query::attribution),
    sealer(ExecutionKind::TemporalDagResponse, temporal::temporal_dag_response),
    sealer(ExecutionKind::Interference, design::interference),
    sealer(ExecutionKind::Randomized, design::randomized),
    sealer(ExecutionKind::PolicyValue, design::policy_value),
    sealer(ExecutionKind::ContinuousDoseResponse, design::continuous_dose),
    sealer(ExecutionKind::PanelDid, design::panel_did),
    sealer(ExecutionKind::SyntheticControl, design::synthetic_control),
    sealer(ExecutionKind::LocalPolynomialRatio, design::local_polynomial_ratio),
    sealer(ExecutionKind::Survival, design::survival),
    sealer(ExecutionKind::LongitudinalRegime, design::longitudinal_regime),
    sealer(ExecutionKind::SequentialContrast, design::sequential_contrast),
    sealer(ExecutionKind::HistoryMatch, design::history_match),
    sealer(ExecutionKind::TemporalDesign, design::temporal_design),
    sealer(ExecutionKind::IvBounds, design::iv_bounds),
    sealer(ExecutionKind::TemporalMediation, temporal::temporal_mediation),
    sealer(ExecutionKind::TemporalDagEffect, temporal::temporal_dag_effect),
    sealer(ExecutionKind::BayesianTemporalDagEffect, temporal::bayesian_temporal_dag_effect),
    sealer(ExecutionKind::TemporalClassEffect, temporal::temporal_class_effect),
    sealer(ExecutionKind::Distribution, program::distribution),
    sealer(ExecutionKind::FunctionalEffect, program::functional_effect),
    sealer(ExecutionKind::PathSpecificEffect, program::path_specific_effect),
    sealer(ExecutionKind::AdmgGraphPosteriorResponse, posterior::admg_graph_posterior_response),
    sealer(ExecutionKind::AdmgResponseCurve, program::admg_response_curve),
    sealer(ExecutionKind::BayesianBasisAte, bayesian::bayesian_basis_ate),
    sealer(ExecutionKind::BayesianBasisCate, bayesian::bayesian_basis_cate),
    sealer(ExecutionKind::BayesianRobustAte, bayesian::bayesian_robust_ate),
    sealer(ExecutionKind::BayesianGcomp, bayesian::bayesian_gcomp),
    sealer(ExecutionKind::BayesianConditional, bayesian::bayesian_conditional),
    Sealer {
        #[cfg(any(test, debug_assertions))]
        kind: ExecutionKind::StaticResponseCurve,
        seal: response::static_response_curve,
        #[cfg(any(test, debug_assertions))]
        yields_to: &[ExecutionKind::StaticDagResponse],
    },
    sealer(ExecutionKind::BayesianClassConditional, posterior::bayesian_class_conditional),
    sealer(ExecutionKind::TemporalClassResponse, posterior::temporal_class_response),
    sealer(
        ExecutionKind::TemporalGraphPosteriorResponse,
        posterior::temporal_graph_posterior_response,
    ),
    sealer(ExecutionKind::TemporalClassMediation, temporal::temporal_class_mediation),
    sealer(ExecutionKind::BayesianTemporalClassEffect, posterior::bayesian_temporal_class_effect),
    sealer(ExecutionKind::TemporalGraphPosteriorEffect, posterior::temporal_graph_posterior_effect),
    sealer(ExecutionKind::StaticClassResponse, posterior::static_class_response),
    sealer(ExecutionKind::GraphPosteriorResponse, posterior::graph_posterior_response),
    sealer(ExecutionKind::BayesianSpecialist, posterior::bayesian_specialist),
    sealer(ExecutionKind::TransportTrial, posterior::transport_trial),
    sealer(ExecutionKind::Discovery, route::discovery),
    sealer(ExecutionKind::PanelTemporal, route::panel_temporal),
    sealer(ExecutionKind::MultiEnvTemporal, route::multi_env_temporal),
    sealer(ExecutionKind::TemporalCrossWorld, route::temporal_cross_world),
    sealer(ExecutionKind::TemporalAdmgEffect, route::temporal_admg_effect),
    sealer(ExecutionKind::Association, route::association),
    // Last: the estimator families the route-cell sealers above decline.
    sealer(ExecutionKind::EstimatorRoute, estimator::estimator_route),
    sealer(ExecutionKind::FreshRun, estimator::fresh_run),
];

/// Select the sealed operation for `cx`. The estimator-family sealer admits what the
/// route-cell sealers decline; a study none admits is refused, not dispatched.
pub(super) fn seal_execution(cx: &SealContext<'_>) -> Result<PreparedExecution, CausalError> {
    for entry in SEALERS.iter().enumerate() {
        let sealer = entry.1;
        let Some(execution) = (sealer.seal)(cx)? else {
            continue;
        };
        #[cfg(debug_assertions)]
        {
            assert_eq!(
                execution.kind(),
                sealer.kind,
                "the {:?} sealer returned another route",
                sealer.kind,
            );
            assert_no_undeclared_overlap(cx, sealer, &SEALERS[entry.0 + 1..]);
        }
        return Ok(execution);
    }
    let estimator = cx
        .source
        .estimator
        .or_else(|| cx.source.estimator_spec.as_deref().map(crate::EstimatorSpec::id))
        .or_else(|| cx.binding.estimator.named());
    Err(CausalError::Compile {
        message: format!(
            "no sealer admits this study; it names an estimator or route outside the sealer \
             table (estimator {estimator:?}, route parsed: {})",
            cx.route.is_some(),
        ),
    })
}

/// Offer the study to every sealer after `winner` and refuse an admission its
/// table entry did not declare.
#[cfg(debug_assertions)]
fn assert_no_undeclared_overlap(cx: &SealContext<'_>, winner: &Sealer, rest: &[Sealer]) {
    for later in rest {
        // The lowered-dispatch sealer admits every study by design.
        if later.kind == ExecutionKind::EstimatorRoute {
            continue;
        }
        match (later.seal)(cx) {
            Ok(None) => {}
            Ok(Some(_)) => assert!(
                later.yields_to.contains(&winner.kind),
                "sealers {:?} and {:?} both admit this study; the table fixes {:?} first but \
                 {:?} does not declare the overlap in `yields_to`",
                winner.kind,
                later.kind,
                winner.kind,
                later.kind,
            ),
            Err(error) => panic!(
                "sealer {:?} refuses a study the {:?} sealer admitted first: {error}",
                later.kind, winner.kind,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{SEALERS, Sealer};
    use crate::analysis::prepared::ExecutionKind;
    use std::collections::HashSet;

    /// The order below is the priority the hand-written chain used; changing
    /// it changes which operation a study that two routes admit runs, so it
    /// must be a deliberate edit of this list as well as of the table.
    const PRIORITY: &[ExecutionKind] = &[
        ExecutionKind::UnknownTieredAverage,
        ExecutionKind::CheckedLinear,
        ExecutionKind::CheckedGlmAdjustment,
        ExecutionKind::CheckedRd,
        ExecutionKind::CheckedPropensity,
        ExecutionKind::CheckedConditional,
        ExecutionKind::GraphPosteriorEffect,
        ExecutionKind::ClassGraphPosteriorEffect,
        ExecutionKind::BayesianGraphPosteriorAte,
        ExecutionKind::StaticClassEffect,
        ExecutionKind::CheckedAipw,
        ExecutionKind::FrontDoorLinear,
        ExecutionKind::Iv,
        ExecutionKind::Counterfactual,
        ExecutionKind::NestedCounterfactual,
        ExecutionKind::DerivativeResponse,
        ExecutionKind::CellAipwResponse,
        ExecutionKind::StaticDagResponse,
        ExecutionKind::BayesianStaticMediation,
        ExecutionKind::StaticMediation,
        ExecutionKind::Attribution,
        ExecutionKind::TemporalDagResponse,
        ExecutionKind::Interference,
        ExecutionKind::Randomized,
        ExecutionKind::PolicyValue,
        ExecutionKind::ContinuousDoseResponse,
        ExecutionKind::PanelDid,
        ExecutionKind::SyntheticControl,
        ExecutionKind::LocalPolynomialRatio,
        ExecutionKind::Survival,
        ExecutionKind::LongitudinalRegime,
        ExecutionKind::SequentialContrast,
        ExecutionKind::HistoryMatch,
        ExecutionKind::TemporalDesign,
        ExecutionKind::IvBounds,
        ExecutionKind::TemporalMediation,
        ExecutionKind::TemporalDagEffect,
        ExecutionKind::BayesianTemporalDagEffect,
        ExecutionKind::TemporalClassEffect,
        ExecutionKind::Distribution,
        ExecutionKind::FunctionalEffect,
        ExecutionKind::PathSpecificEffect,
        ExecutionKind::AdmgGraphPosteriorResponse,
        ExecutionKind::AdmgResponseCurve,
        ExecutionKind::BayesianBasisAte,
        ExecutionKind::BayesianBasisCate,
        ExecutionKind::BayesianRobustAte,
        ExecutionKind::BayesianGcomp,
        ExecutionKind::BayesianConditional,
        ExecutionKind::StaticResponseCurve,
        ExecutionKind::BayesianClassConditional,
        ExecutionKind::TemporalClassResponse,
        ExecutionKind::TemporalGraphPosteriorResponse,
        ExecutionKind::TemporalClassMediation,
        ExecutionKind::BayesianTemporalClassEffect,
        ExecutionKind::TemporalGraphPosteriorEffect,
        ExecutionKind::StaticClassResponse,
        ExecutionKind::GraphPosteriorResponse,
        ExecutionKind::BayesianSpecialist,
        ExecutionKind::TransportTrial,
        ExecutionKind::Discovery,
        ExecutionKind::PanelTemporal,
        ExecutionKind::MultiEnvTemporal,
        ExecutionKind::TemporalCrossWorld,
        ExecutionKind::TemporalAdmgEffect,
        ExecutionKind::Association,
        ExecutionKind::EstimatorRoute,
        ExecutionKind::FreshRun,
    ];

    #[test]
    fn table_order_is_the_documented_priority() {
        let table: Vec<_> = SEALERS.iter().map(|sealer| sealer.kind).collect();
        assert_eq!(table, PRIORITY);
    }

    #[test]
    fn every_sealed_kind_has_exactly_one_sealer() {
        let sealed: HashSet<_> = SEALERS.iter().map(|sealer| sealer.kind).collect();
        assert_eq!(sealed.len(), SEALERS.len(), "a kind has two sealers");
        let expected: HashSet<_> = ExecutionKind::ALL.iter().copied().collect();
        assert_eq!(sealed, expected);
    }

    #[test]
    fn declared_overlaps_name_higher_priority_routes() {
        let position = |kind: ExecutionKind| {
            SEALERS.iter().position(|sealer| sealer.kind == kind).expect("tabled kind")
        };
        for sealer in &SEALERS {
            let Sealer { kind, yields_to, .. } = sealer;
            for earlier in *yields_to {
                assert!(
                    position(*earlier) < position(*kind),
                    "{kind:?} yields to {earlier:?}, which the table ranks below it",
                );
            }
        }
    }
}
