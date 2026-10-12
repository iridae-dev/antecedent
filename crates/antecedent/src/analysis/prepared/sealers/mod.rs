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
mod posterior;
mod program;
mod response;
mod static_query;
mod temporal;

use super::{ExecutionKind, PreparedExecution};
use crate::analysis::execute::Study;
use crate::error::CausalError;
use crate::planner::PhysicalExecutionPlan;
use crate::strategy_table::{EstimatorId, IdentifierId};
use antecedent_core::ExecutionContext;

/// Everything a sealer reads to decide admission and to build its operation.
#[derive(Clone, Copy)]
pub(super) struct SealContext<'a> {
    /// The study `prepare` was called on.
    pub(super) source: &'a Study,
    /// The study with the prepare-time identification caches installed. It is
    /// the study every retained operation is built from.
    pub(super) analysis: &'a Study,
    /// The physical plan compiled once for this prepare.
    pub(super) plan: &'a PhysicalExecutionPlan,
    pub(super) ctx: &'a ExecutionContext,
    /// The compiled cross-world operation of a tabular counterfactual query.
    pub(super) counterfactual: Option<&'a crate::gcm::CheckedCounterfactualOperation>,
    /// The compiled cross-world operation of a nested counterfactual query.
    pub(super) nested_counterfactual: Option<&'a crate::gcm::NestedCounterfactualOperation>,
}

impl SealContext<'_> {
    /// The plan record's identifier, or `default` when it names none.
    pub(super) fn plan_identifier(&self, default: &str) -> Result<IdentifierId, CausalError> {
        Ok(self.plan.logical.record.identifier.as_deref().unwrap_or(default).parse()?)
    }

    /// The plan record's estimator, or `default` when it names none.
    pub(super) fn plan_estimator(&self, default: &str) -> Result<EstimatorId, CausalError> {
        Ok(self.plan.logical.record.estimator.as_deref().unwrap_or(default).parse()?)
    }
}

/// What one sealer answers: no admission, an admission and its operation, or
/// the refusal of an admitted study.
pub(super) type SealOutcome = Result<Option<PreparedExecution>, CausalError>;

/// One entry of the sealer table.
pub(super) struct Sealer {
    /// The route this sealer retains. Every sealed [`ExecutionKind`] has
    /// exactly one entry.
    pub(super) kind: ExecutionKind,
    pub(super) seal: fn(&SealContext<'_>) -> SealOutcome,
    /// Earlier routes this one is allowed to overlap. A study both admit goes
    /// to the earlier route; the overlap is intended, not an accident.
    pub(super) yields_to: &'static [ExecutionKind],
}

const fn sealer(kind: ExecutionKind, seal: fn(&SealContext<'_>) -> SealOutcome) -> Sealer {
    Sealer { kind, seal, yields_to: &[] }
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
    sealer(ExecutionKind::StaticClassEffect, posterior::static_class_effect),
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
        kind: ExecutionKind::StaticResponseCurve,
        seal: response::static_response_curve,
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
];

/// Select the sealed operation for `cx`, or `OrdinaryDispatch` when no sealer
/// admits it.
pub(super) fn seal_execution(cx: &SealContext<'_>) -> Result<PreparedExecution, CausalError> {
    for (index, sealer) in SEALERS.iter().enumerate() {
        let Some(execution) = (sealer.seal)(cx)? else {
            continue;
        };
        debug_assert_eq!(
            execution.kind(),
            sealer.kind,
            "the {:?} sealer returned another route",
            sealer.kind,
        );
        #[cfg(debug_assertions)]
        assert_no_undeclared_overlap(cx, sealer, &SEALERS[index + 1..]);
        return Ok(execution);
    }
    Ok(PreparedExecution::OrdinaryDispatch)
}

/// Offer the study to every sealer after `winner` and refuse an admission its
/// table entry did not declare.
#[cfg(debug_assertions)]
fn assert_no_undeclared_overlap(cx: &SealContext<'_>, winner: &Sealer, rest: &[Sealer]) {
    for later in rest {
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
        let expected: HashSet<_> = ExecutionKind::ALL
            .iter()
            .copied()
            .filter(|kind| *kind != ExecutionKind::OrdinaryDispatch)
            .collect();
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
