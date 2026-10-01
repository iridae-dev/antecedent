//! Exact execution of a checked ADMG conditional transport formula (2.2B B1).
//!
//! The identifier reduces `P*(y | do(x), w)` to the joint `P*(y, w'' | do(x, w'))`
//! (rule 2 moved `w'` into the intervention set) and binds that joint to the
//! catalog. Execution compiles the joint once with the existing exact evaluator
//! at `do(x, w')` and conditions it on the requested `w''` by explicit
//! normalization. A conditioning event of zero mass is a support failure, never
//! a silently extended value. The route is point-only: exact laws carry no
//! sampling uncertainty and no interval is computed.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::EstimationError;
use antecedent_core::{ExecutionContext, Value, VariableId};
use antecedent_expr::{
    Assignment, ExactDistribution, ExactEvaluationLimits, ExactEvaluationPlan, ExactTransportData,
};
use antecedent_identify::BoundConditionalTransportFunctional;
use std::sync::Arc;

/// The request does not bind exactly the treatments and conditioned variables,
/// or names a conditioning level outside the evaluated support.
const INVALID_REQUEST: &str = "admg_transport.invalid_request";

fn invalid_request(message: &str) -> EstimationError {
    EstimationError::refused(
        antecedent_core::reason_code!("invalid_argument"),
        format!("{INVALID_REQUEST}: {message}"),
    )
}

/// A compiled conditional plan: the joint's exact plan at `do(x, w')` and the
/// conditioning levels of `w''`. Evaluating it never searches.
#[derive(Clone, Debug)]
pub struct AdmgConditionalExactPlan {
    joint: ExactEvaluationPlan,
    outcomes: Arc<[VariableId]>,
    conditioning: Vec<(VariableId, Value)>,
}

impl AdmgConditionalExactPlan {
    /// The compiled plan of the reduced joint.
    #[must_use]
    pub const fn joint_plan(&self) -> &ExactEvaluationPlan {
        &self.joint
    }
    /// Outcome coordinates of the conditional distribution.
    #[must_use]
    pub fn outcomes(&self) -> &[VariableId] {
        &self.outcomes
    }
    /// The conditioning levels `w''` the joint is normalized at.
    #[must_use]
    pub fn conditioning(&self) -> &[(VariableId, Value)] {
        &self.conditioning
    }

    /// Evaluate the joint and condition it on `w''`.
    ///
    /// # Errors
    /// An exact-evaluation refusal, a conditioning level outside the joint's
    /// support (`admg_transport.invalid_request`), or a zero-mass conditioning
    /// event (`transport_support_failure`, `admg_transport.support_failure`).
    pub fn evaluate(&self, ctx: &ExecutionContext) -> Result<ExactDistribution, EstimationError> {
        let joint = self.joint.evaluate(ctx).map_err(|e| crate::refuse_eval(&e))?;
        condition(&joint, &self.outcomes, &self.conditioning)
    }
}

/// Whether an atom's value is the requested level: equal values, or equal numbers
/// across integer and float spellings.
fn same_level(atom: &Value, level: &Value) -> bool {
    atom == level || atom.as_f64().is_some_and(|a| level.as_f64() == Some(a))
}

/// Condition an evaluated joint over `outcomes ++ w''` on the levels `conditioning`.
fn condition(
    joint: &ExactDistribution,
    outcomes: &Arc<[VariableId]>,
    conditioning: &[(VariableId, Value)],
) -> Result<ExactDistribution, EstimationError> {
    let axis = |v: VariableId| {
        joint.outcomes.iter().position(|w| *w == v).ok_or_else(|| {
            EstimationError::refused(
                antecedent_core::reason_code!("transport_numerical_failure"),
                "the evaluated joint lacks a conditioned coordinate",
            )
        })
    };
    let outcome_axes = outcomes.iter().map(|v| axis(*v)).collect::<Result<Vec<_>, _>>()?;
    let conditioning_axes = conditioning
        .iter()
        .map(|(v, x)| Ok::<_, EstimationError>((axis(*v)?, x)))
        .collect::<Result<Vec<_>, _>>()?;
    for (index, level) in &conditioning_axes {
        if !joint.atoms.iter().any(|atom| same_level(&atom[*index], level)) {
            return Err(invalid_request("a conditioning level is outside the evaluated support"));
        }
    }
    let mut atoms = Vec::new();
    let mut masses = Vec::new();
    for (atom, probability) in joint.atoms.iter().zip(joint.probabilities.iter()) {
        if conditioning_axes.iter().all(|(index, level)| same_level(&atom[*index], level)) {
            atoms.push(outcome_axes.iter().map(|i| atom[*i].clone()).collect::<Arc<[Value]>>());
            masses.push(*probability);
        }
    }
    // Fixed order and plain summation: a consumer recomputes the same bits.
    let total: f64 = masses.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return Err(EstimationError::refused(
            antecedent_core::reason_code!("transport_support_failure"),
            "admg_transport.support_failure: the conditioning event has zero mass at the requested treatment level",
        ));
    }
    Ok(ExactDistribution {
        outcomes: outcomes.clone(),
        atoms: atoms.into(),
        probabilities: masses.iter().map(|p| p / total).collect::<Vec<_>>().into(),
        support: joint.support.clone(),
    })
}

/// Validate the laws against the frozen catalog and compile the conditional
/// plan for one request, without evaluating it. The request binds exactly the
/// query's treatments and conditioned variables.
///
/// # Errors
/// Counted laws (`cell_not_licensed`, `admg_transport.interval_withheld`), a
/// request that does not bind exactly those coordinates
/// (`admg_transport.invalid_request`), provider/catalog disagreement, or a
/// compile-time resource limit.
pub fn prepare_exact_admg_conditional_transport(
    functional: &BoundConditionalTransportFunctional,
    data: ExactTransportData,
    request: &Assignment,
    limits: ExactEvaluationLimits,
    ctx: &ExecutionContext,
) -> Result<AdmgConditionalExactPlan, EstimationError> {
    // Counted laws would make the point an empirical plug-in, which this route
    // does not license (no coverage record): refused before anything compiles,
    // as the facade and Python preparations refuse them.
    if data.laws().iter().any(|law| law.empirical_counts().is_some()) {
        return Err(EstimationError::refused(
            antecedent_core::reason_code!("cell_not_licensed"),
            "admg_transport.interval_withheld: counted laws are not licensed; the route publishes exact-law points only",
        ));
    }
    let derivation = functional.derivation();
    let query = derivation.query();
    let bound: Vec<VariableId> =
        query.base.treatments.iter().chain(query.conditioned_on.iter()).copied().collect();
    if request.entries().len() != bound.len()
        || bound.iter().any(|variable| request.get(*variable).is_none())
    {
        return Err(invalid_request(
            "a request binds exactly the treatments and the conditioned variables",
        ));
    }
    let level = |v: &VariableId| {
        request.get(*v).cloned().map(|x| (*v, x)).ok_or_else(|| {
            invalid_request("a request binds exactly the treatments and the conditioned variables")
        })
    };
    let joint_request = Assignment::from_pairs(
        derivation.reduced_query().treatments.iter().map(level).collect::<Result<Vec<_>, _>>()?,
    );
    let conditioning = derivation.remaining().iter().map(level).collect::<Result<Vec<_>, _>>()?;
    let joint =
        crate::prepare_exact_transport(functional.joint(), data, joint_request, limits, ctx)
            .map_err(|e| crate::refuse_eval(&e))?;
    Ok(AdmgConditionalExactPlan { joint, outcomes: query.base.outcomes.clone(), conditioning })
}
