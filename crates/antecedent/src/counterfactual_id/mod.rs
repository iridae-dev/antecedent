//! Counterfactual identification on a bounded ADMG: the effect of treatment
//! on the treated from the observational joint.
//!
//! One cell: `P(Y_x = y | X = x')` with `x != x'`, its distribution over every
//! level of `Y` and the contrast `E[Y_x | X = x'] - E[Y | X = x']`, on an
//! explicit ADMG (or DAG) of at most six finite-discrete variables with at most
//! four levels each, from a supplied exact joint law or an empirical count
//! table. A point claim: no interval, no posterior.
//!
//! The stages are separate:
//!
//! 1. [`prepare_counterfactual_id`] decides the query once under one shared
//!    search budget ([`antecedent_identify::counterfactual_id`]) and returns
//!    the derivation, or a reason-coded refusal: `route_not_supported` with a
//!    checkable obstruction when ID* (and, for a binary treatment, the
//!    consistency complement) does not identify the query, which is not a
//!    proof of non-identifiability; `transport_budget_cancel` with a receipt;
//!    `route_not_supported` outside the contract (`path_specific_deferred` for
//!    a world that routes edges, `bounds_exceeded` over a bound);
//!    `cell_not_licensed` for a structure outside the graph contract,
//!    `estimator_inference_mismatch` for an interval request.
//! 2. [`PreparedCounterfactualId::evaluate`] evaluates the derivation on a law
//!    without re-identifying; call it again with another law of the same
//!    variables and levels to refresh.
//! 3. [`CounterfactualIdEffect::export_artifact`] writes the
//!    `counterfactual_id_admg_v1` artifact; [`consume_counterfactual_id_artifact`]
//!    re-derives it under the stored limits, requires the identical derivation
//!    and point, and recomputes the point with a separate direct-sum evaluator
//!    that shares no evaluation code with the compiled evaluator.
//!
//! What replay does not protect against: a producer that seals a wrong graph,
//! query or law on purpose (the digests bind what was sealed, not what is
//! true); and a bug shared by the identification engine and the consumer,
//! because the consumer re-derives with the same engine and the direct-sum
//! evaluator evaluates that same re-derived functional, so such a bug replays
//! identically. The direct sum checks the functional's arithmetic, not its
//! derivation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::GraphRef;
use crate::error::{CausalError, REASON_PREFIX};
use crate::support::{IntoGraphInput, StructureSource};
use antecedent_core::{CounterfactualEventQuery, ExecutionContext, SearchLimits};
use antecedent_expr::ExactDiscreteLaw;
use antecedent_graph::Admg;
use antecedent_identify::counterfactual_id::{
    COUNTERFACTUAL_ID_DEFAULT_LIMITS, COUNTERFACTUAL_ID_MEMORY_BYTES, CounterfactualIdDerivation,
    CounterfactualIdProblem, CounterfactualIdRefusal, decide_counterfactual_id,
    evaluate_counterfactual_functional,
};
use antecedent_io::counterfactual_id_artifact::{
    CounterfactualIdLawWire, counterfactual_id_data_digest,
};

mod artifact;
#[cfg(test)]
mod tests;

use artifact::law_wire;
pub use artifact::{
    CounterfactualIdConsumeLimits, consume_counterfactual_id_artifact,
    consume_counterfactual_id_artifact_for_data,
};

/// Relative tolerance of the direct-sum cross-check against the replayed point.
pub(super) const DIRECT_SUM_TOLERANCE: f64 = 1e-10;

/// What the caller asks of the cell beyond the query.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CounterfactualIdOptions {
    /// Search limits of the identification (at most 100000 operations, depth 64).
    pub limits: SearchLimits,
    /// Memory cap of the search (further limited by the context's hard limit).
    pub memory_limit_bytes: u64,
    /// The caller asked for a sampling interval or Bayesian inference. Refused:
    /// the claim is point-only.
    pub interval_requested: bool,
}

impl Default for CounterfactualIdOptions {
    fn default() -> Self {
        Self {
            limits: COUNTERFACTUAL_ID_DEFAULT_LIMITS,
            memory_limit_bytes: COUNTERFACTUAL_ID_MEMORY_BYTES,
            interval_requested: false,
        }
    }
}

/// A reason-coded facade error from an identification refusal. Its text is
/// `reason=<code>: <namespaced detail>: <message>`, followed by the search
/// receipt's summary when the refusal is a budget or cancellation stop.
fn refusal_error(refusal: &CounterfactualIdRefusal) -> CausalError {
    CausalError::Compile { message: format!("{REASON_PREFIX}{refusal}") }
}

/// A decided query, ready to evaluate on any law of its variables and levels.
#[derive(Clone, Debug)]
pub struct PreparedCounterfactualId {
    problem: CounterfactualIdProblem,
    query: CounterfactualEventQuery,
    derivation: CounterfactualIdDerivation,
    names: Vec<String>,
}

/// Decide `query` on `graph` once.
///
/// `names` names the graph's variables in id order; `levels` lists each
/// variable's levels.
///
/// # Errors
///
/// A reason-coded [`CausalError`] (see the module documentation).
pub fn prepare_counterfactual_id(
    graph: impl IntoGraphInput,
    names: Vec<String>,
    levels: Vec<Vec<f64>>,
    query: &CounterfactualEventQuery,
    options: CounterfactualIdOptions,
    ctx: &ExecutionContext,
) -> Result<PreparedCounterfactualId, CausalError> {
    if options.interval_requested {
        return Err(refusal_error(&CounterfactualIdRefusal::interval_requested()));
    }
    let (structure, source) = graph.into_graph_input();
    let admg: Admg = match (source, structure.graph_ref()) {
        (StructureSource::Explicit, GraphRef::Admg(admg)) => admg.clone(),
        (StructureSource::Explicit, GraphRef::Dag(dag)) => antecedent_identify::dag_to_admg(dag)
            .map_err(|e| {
                refusal_error(&CounterfactualIdRefusal::graph(format!("the DAG is invalid: {e}")))
            })?,
        _ => {
            return Err(refusal_error(&CounterfactualIdRefusal::graph(
                "only a supplied explicit ADMG or DAG is licensed; accepted, uncertain, \
                 equivalence-class and temporal structures are refused",
            )));
        }
    };
    let problem = CounterfactualIdProblem::from_admg(&admg, levels)
        .map_err(|refusal| refusal_error(&refusal))?;
    check_names(&names, problem.node_count()).map_err(|refusal| refusal_error(&refusal))?;
    let derivation =
        decide_counterfactual_id(&problem, query, options.limits, options.memory_limit_bytes, ctx)
            .map_err(|refusal| refusal_error(&refusal))?;
    Ok(PreparedCounterfactualId { problem, query: query.clone(), derivation, names })
}

fn check_names(names: &[String], n: usize) -> Result<(), CounterfactualIdRefusal> {
    let distinct: std::collections::BTreeSet<&String> = names.iter().collect();
    if names.len() != n || distinct.len() != n || names.iter().any(String::is_empty) {
        return Err(CounterfactualIdRefusal::invalid_query(format!(
            "{} names for a graph of {n} variables; names must be distinct and non-empty",
            names.len()
        )));
    }
    Ok(())
}

impl PreparedCounterfactualId {
    /// The derivation (identified once, reused by every evaluation).
    #[must_use]
    pub const fn derivation(&self) -> &CounterfactualIdDerivation {
        &self.derivation
    }

    /// The decided query.
    #[must_use]
    pub const fn query(&self) -> &CounterfactualEventQuery {
        &self.query
    }

    /// Evaluate the prepared derivation on `law` (no re-identification). The law
    /// must cover exactly the prepared variables and levels (axes in any order)
    /// and be a supplied exact law or an empirical count table.
    ///
    /// # Errors
    ///
    /// `invalid_argument` (`counterfactual_id.invalid_query`) for another law,
    /// origin or a zero-probability conditioning event;
    /// `counterfactual_id.positivity_violation` when a conditional the
    /// functional evaluates is on a zero-mass event; the typed cancellation.
    pub fn evaluate(
        &self,
        law: &ExactDiscreteLaw,
        ctx: &ExecutionContext,
    ) -> Result<CounterfactualIdEffect, CausalError> {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: "counterfactual_id" });
        }
        let law_wire = law_wire(law).map_err(|refusal| refusal_error(&refusal))?;
        let point = evaluate_counterfactual_functional(&self.problem, &self.derivation, law, ctx)
            .map_err(|refusal| {
            if refusal.receipt.is_some() {
                CausalError::Cancelled { stage: "counterfactual_id" }
            } else {
                refusal_error(&refusal)
            }
        })?;
        let data_digest = counterfactual_id_data_digest(&law_wire)?;
        let distribution = point
            .numerators
            .iter()
            .map(|(level, p)| (*level, p / point.conditioning_probability))
            .collect();
        Ok(CounterfactualIdEffect {
            probability: point.probability,
            conditioning_probability: point.conditioning_probability,
            outcome_distribution: distribution,
            counterfactual_mean: point.counterfactual_mean,
            observed_mean: point.observed_mean,
            effect: point.effect,
            numerators: point.numerators,
            data_digest,
            independently_verified: false,
            prepared: self.clone(),
            law: law_wire,
        })
    }
}

/// The point, its outcome distribution and contrast, and the derivation.
#[derive(Clone, Debug)]
pub struct CounterfactualIdEffect {
    /// `P(Y_x = y | X = x')` (0 for an exact zero).
    pub probability: f64,
    /// `P(X = x')`.
    pub conditioning_probability: f64,
    /// `(level, P(Y_x = level | X = x'))` for every level of `Y`, declared order.
    pub outcome_distribution: Vec<(f64, f64)>,
    /// `E[Y_x | X = x']`.
    pub counterfactual_mean: Option<f64>,
    /// `E[Y | X = x']`.
    pub observed_mean: Option<f64>,
    /// `E[Y_x | X = x'] - E[Y | X = x']`.
    pub effect: Option<f64>,
    /// `(level, P(Y_x = level, X = x'))`.
    pub numerators: Vec<(f64, f64)>,
    /// Digest of the law the point was computed on; compare it with your own.
    pub data_digest: String,
    /// The consumer recomputed the point with the separate direct-sum evaluator
    /// and it agreed. `false` on a freshly evaluated effect.
    pub independently_verified: bool,
    prepared: PreparedCounterfactualId,
    law: CounterfactualIdLawWire,
}

impl CounterfactualIdEffect {
    /// The derivation the point was evaluated from.
    #[must_use]
    pub const fn derivation(&self) -> &CounterfactualIdDerivation {
        &self.prepared.derivation
    }

    /// The variable names, in id order.
    #[must_use]
    pub fn names(&self) -> &[String] {
        &self.prepared.names
    }
}
