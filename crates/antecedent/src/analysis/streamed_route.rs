//! Which nuisance-model routes run as a streamed reduction when the resident design does not
//! fit the caller's memory limit.
//!
//! Two-stage least squares, inverse-probability weighting and cross-fitted AIPW reduce their
//! rows in closed form (see `antecedent_estimate::streamed_iv`, `streamed_propensity` and
//! `streamed_aipw`). A study streams when it
//! is a frequentist static average effect over a DAG, the estimator it resolves to has a
//! reduction and its configuration admits it, and nothing downstream needs the materialized
//! design: no refuter, no custom validator and, for weighting and AIPW, no bootstrap.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{OutcomeFunctional, TargetPopulation};
use antecedent_data::{ColumnView, TableView};

use crate::analysis::builder::RefuteSuite;
use crate::analysis::execute::{RouteInput, Study};
use crate::estimator_spec::EstimatorSpec;
use crate::planner::PhysicalExecutionPlan;
use crate::strategy_table::{EstimatorId, StaticEstimateInput, streams_static_effect};

/// The estimator `study` streams when its resident design does not fit, if it streams.
pub(crate) fn streamed_estimator(
    study: &Study,
    plan: &PhysicalExecutionPlan,
) -> Option<EstimatorId> {
    if !study.inference.is_frequentist()
        || study.graph_posterior.is_some()
        || study.tiered.is_some()
        || !study.custom_validators.is_empty()
        || !matches!(study.refute, RefuteSuite::None)
    {
        return None;
    }
    let RouteInput::StaticAteDag { data, query, .. } = study.route_input_classified()? else {
        return None;
    };
    if !matches!(query.outcome_functional, OutcomeFunctional::Mean)
        || !matches!(query.target_population, TargetPopulation::AllObserved)
    {
        return None;
    }
    let estimator = plan.logical.record.estimator.as_deref()?.parse::<EstimatorId>().ok()?;
    if !PhysicalExecutionPlan::runs_streamed_reduction(estimator)
        || matches!(estimator, EstimatorId::LinearAdjustmentAte)
    {
        // The linear adjustment sizes and streams itself when it retains an operation.
        return None;
    }
    let spec =
        study.estimator_spec.as_deref().cloned().unwrap_or(EstimatorSpec::Default(estimator));
    // Only the defaults of the point stage are read, not the estimand.
    let estimand = antecedent_expr::IdentifiedEstimand::backdoor(
        "backdoor.adjustment",
        std::sync::Arc::from([]),
        antecedent_expr::ExprId::from_raw(0),
    );
    let input = StaticEstimateInput {
        data,
        estimand: &estimand,
        query,
        bootstrap_replicates: study.bootstrap_replicates,
        overlap_policy: study.overlap_policy,
        population_registry: study.population_registry.as_ref(),
        sampling_design: study.sampling_design,
    };
    if !streams_static_effect(&spec, estimator, &input) {
        return None;
    }
    // The reduction reads float64 columns straight from the table.
    let reads_float64 = data
        .schema()
        .variables()
        .iter()
        .all(|variable| matches!(data.column(variable.id), Ok(ColumnView::Float64(_))));
    reads_float64.then_some(estimator)
}
