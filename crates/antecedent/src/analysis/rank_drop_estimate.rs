//! Opt-in rank-drop estimation: estimate on the span-preserving reduced adjustment set.
//!
//! [`estimate_with_rank_drop`] executes a [`RankDropPlan`]. It re-runs the configured
//! estimator on the design with the plan's dependent columns removed and returns the point
//! estimate together with the recorded drop, its reasons, the resulting design identity and
//! the original and reduced adjustment sets. Nothing is dropped silently, and it refuses
//! whenever the drop could change what is fitted.
//!
//! **Why a span-preserving drop cannot change the fitted nuisances.** A dropped column is an
//! exact linear combination of the retained columns and the intercept, so the retained
//! `[1 | Z_kept]` design and the original `[1 | Z]` design have the same column space.
//! Ordinary least squares depends on the design only through that space: the fitted values
//! `P y` (`P` the projector onto it) and the coefficient on the treatment are unchanged. An
//! unpenalized logistic maximum likelihood depends on the design only through the linear
//! predictor `X b`, and `X b` ranges over the same set of vectors for either design, so the
//! maximizer's fitted probabilities coincide too. Penalized or separation-ridge fits are
//! parametrization dependent, which is why the entry point refuses every estimator that is not
//! one of the two unpenalized routes and guards the cross-fitted one against separation.
//! Only the original (rank-deficient) design's *coefficients* are not identified, and the
//! estimand never reads them.
//!
//! The estimate is computed by the ordinary study path on the reduced table with a declared
//! confounder graph (each retained column points into the treatment and the outcome, the
//! treatment into the outcome); that graph only carries the retained adjustment set into the
//! estimator and no identification claim is made from it. The returned adjustment sets are
//! the declared design's, not a graph search's.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{AverageEffectQuery, ExecutionContext, VariableId, reason_code};
use antecedent_data::TableView;
use antecedent_estimate::{EstimationError, RefusalFields};
use antecedent_graph::Dag;
use serde::Serialize;

use super::builder::RefuteSuite;
use super::execute::Study;
use super::preflight::{
    PreflightInput, PropensityOutcome, RankDropPlan, RankDropPolicy, SpanCheck,
    fit_diagnostics_design, kept_adjustment_ids, plan_rank_drop, verify_span_preserved,
};
use crate::error::CausalError;
use crate::strategy_table::EstimatorId;

/// What the result is and is not.
const ESTIMATE_NOTE: &str = "point estimate on the span-preserving reduced adjustment set; the \
     dropped columns are exact linear combinations of the retained ones, so the fitted \
     projections equal the original design's; no interval, calibration or new identification \
     claim is made";

/// The estimate computed on a reduced design, with the drop that produced it.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RankDropEstimate {
    /// The recorded drop: dropped columns and their relations, original and reduced
    /// adjustment sets, priority and design identity.
    pub plan: RankDropPlan,
    /// Independent numerical check that the retained columns span the original design.
    pub span_check: SpanCheck,
    /// Estimator id that was re-run.
    pub estimator: String,
    /// `exact` for ordinary least squares, `unpenalized_logistic_no_separation` for the
    /// cross-fitted route (invariance holds while no propensity fit separates).
    pub projection_invariance: &'static str,
    /// The point estimate on the reduced design.
    pub ate: f64,
    /// What this result is and is not.
    pub note: &'static str,
}

/// Refuse with `rank_drop_not_licensed` and the plan's subject; nothing was estimated.
fn not_licensed(subject: &str, reason: &str, implicated: Vec<String>, remedy: &str) -> CausalError {
    let fields = RefusalFields {
        stage: Some("rank_drop_estimate".to_string()),
        subject: Some(subject.to_string()),
        reason: Some(reason.to_string()),
        implicated_columns: implicated,
        remedy: Some(remedy.to_string()),
        ..RefusalFields::default()
    };
    EstimationError::refused_with_fields(
        reason_code!("rank_drop_not_licensed"),
        format!("{subject}: {reason}; nothing was estimated"),
        fields,
    )
    .into()
}

fn not_licensed_estimator() -> CausalError {
    crate::unsupported_reason!(
        "route_not_supported",
        "rank_drop_estimate.estimator_not_licensed: estimation after a rank drop is licensed \
         only for the linear adjustment and the cross-fitted AIPW estimators: other nuisance \
         fits (penalized, lasso, matching, stratification) are not shown invariant to a \
         span-preserving column removal"
    )
}

/// Estimate the binary ATE on the reduced adjustment set of a declared rank drop.
///
/// Refuses before estimating when the estimator is not one of the two unpenalized routes,
/// when the plan refuses, when the retained columns do not numerically span the original
/// design, when the cross-fitted route's diagnostic propensity fit on the reduced design
/// separates or fails, or when the estimator's own certified adjustment set is not exactly the
/// retained set.
///
/// # Errors
///
/// `route_not_supported` for another estimator or a design that is not a binary contrast of
/// one treatment; `rank_drop_not_licensed` for every refusal above; the study errors from the
/// reduced estimate; cancellation.
pub fn estimate_with_rank_drop(
    input: &PreflightInput<'_>,
    policy: &RankDropPolicy,
    estimator: EstimatorId,
    ctx: &ExecutionContext,
) -> Result<RankDropEstimate, CausalError> {
    let projection_invariance = match estimator {
        EstimatorId::LinearAdjustmentAte => "exact",
        EstimatorId::Aipw => "unpenalized_logistic_no_separation",
        _ => return Err(not_licensed_estimator()),
    };
    let ([treatment], [control, active]) = (input.treatments.as_slice(), input.arms.as_slice())
    else {
        return Err(crate::unsupported_reason!(
            "route_not_supported",
            "rank_drop_estimate.shape_not_supported: estimation after a rank drop covers a \
             binary contrast of one treatment"
        ));
    };
    let (Some(&(_, control_level)), Some(&(_, active_level))) =
        (control.levels.first(), active.levels.first())
    else {
        return Err(not_licensed_estimator());
    };

    let plan = plan_rank_drop(input, policy, ctx)?;
    let kept = kept_adjustment_ids(input, &plan)?;
    let span_check = verify_span_preserved(input, &plan, &kept, ctx)?;
    if estimator == EstimatorId::Aipw {
        let reduced = PreflightInput { adjustment: kept.clone(), ..input.clone() };
        let clean = match fit_diagnostics_design(&reduced, ctx)?.propensity {
            PropensityOutcome::Fitted(fit) => {
                fit.converged && !fit.separated && !fit.boundary_saturated
            }
            PropensityOutcome::Absent { .. } => false,
        };
        if !clean {
            return Err(not_licensed(
                &plan.subject,
                "rank_drop_estimate.propensity_separates: the propensity fit on the reduced \
                 design separates, saturates, fails or does not converge, so a penalized refit \
                 could make it parametrization dependent",
                Vec::new(),
                "resolve the separation first (see diagnose_fit) or keep the original design",
            ));
        }
    }

    let mut ids: Vec<VariableId> = vec![*treatment, input.outcome];
    ids.extend_from_slice(&kept);
    let (reduced, remap) = input.data.project(&ids)?;
    let (t, y) = (remap.map(*treatment)?, remap.map(input.outcome)?);
    let names: Vec<String> = (0..ids.len())
        .map(|k| {
            let id = VariableId::from_raw(u32::try_from(k).unwrap_or(u32::MAX));
            reduced.schema().get(id).map(|meta| meta.name.to_string())
        })
        .collect::<Result<_, _>>()?;
    let (treatment_name, outcome_name) =
        (names[t.as_usize()].as_str(), names[y.as_usize()].as_str());
    let mut edges: Vec<(&str, &str)> = vec![(treatment_name, outcome_name)];
    for name in &names[2..] {
        edges.push((name.as_str(), treatment_name));
        edges.push((name.as_str(), outcome_name));
    }
    let graph = Dag::from_named_edges(reduced.schema(), &edges)?;
    let result = Study::tabular(reduced)
        .graph(graph)
        .query(AverageEffectQuery::with_levels(t, y, control_level, active_level))
        .estimator(estimator)
        .bootstrap_replicates(0)
        .refute(RefuteSuite::None)
        .build()?
        .run(ctx)?;

    let mut certified: Vec<VariableId> = result.estimand.adjustment_set.to_vec();
    certified.sort_unstable();
    let expected: Vec<VariableId> = (2..ids.len())
        .map(|k| VariableId::from_raw(u32::try_from(k).unwrap_or(u32::MAX)))
        .collect();
    if certified != expected {
        return Err(not_licensed(
            &plan.subject,
            "the estimator certified an adjustment set other than the retained columns",
            plan.kept_adjustment.clone(),
            "keep the original adjustment set",
        ));
    }
    let ate = result.estimate.as_effect().map(|effect| effect.ate).ok_or_else(|| {
        crate::compile_reason!("invalid_argument", "the reduced study returned no effect")
    })?;
    Ok(RankDropEstimate {
        plan,
        span_check,
        estimator: estimator.as_str().to_string(),
        projection_invariance,
        ate,
        note: ESTIMATE_NOTE,
    })
}
