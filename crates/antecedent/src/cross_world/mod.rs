//! Cross-world edge contrasts on a fixed Markovian DAG: check, coupled
//! operation, result.
//!
//! One cell: a path-specific edge-intervention contrast (which of the graph's
//! edges see the intervened value and which see the baseline) on a supplied
//! explicit DAG, evaluated as `E[Y_1] - E[Y_0]` under one abduced exogenous term
//! per unit and variable shared by both worlds. The natural direct and indirect
//! effects are special cases. It is a point claim: no interval, no Bayesian
//! posterior, no latent confounding, no accepted or uncertain structure.
//!
//! The three stages are separate and separately testable:
//!
//! 1. [`check`] decides whether the graph and query license the estimand and
//!    returns a machine-checkable witness
//!    ([`antecedent_identify::cross_world`]).
//! 2. [`antecedent_counterfactual::cross_world::evaluate_cross_world`] abducts,
//!    acts and predicts as one operation over shared exogenous terms.
//! 3. [`CrossWorldEffect`] carries the point beside the witness and exports an
//!    artifact a consumer replays ([`consume_cross_world_artifact`]): same
//!    evaluator, plus a separate closed-form recomputation (ordinary least
//!    squares, or the basis regression by QR) for both mechanism families.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{CrossWorldQuery, ExecutionContext, VariableId};
use antecedent_counterfactual::{
    CounterfactualEngine, CounterfactualError, NoiseInferenceKind,
    cross_world::evaluate_cross_world,
};
use antecedent_data::{ColumnView, TableView, TabularData};
use antecedent_graph::Dag;
use antecedent_identify::cross_world::{CrossWorldRefusal, CrossWorldWitness, check_cross_world};
pub use antecedent_io::cross_world_artifact::CROSS_WORLD_MAX_ROWS;
use antecedent_io::cross_world_artifact::cross_world_data_digest;
use antecedent_model::CompiledCausalModel;

use crate::GraphRef;
use crate::error::CausalError;
use crate::gcm::{NestedOutcomeMechanism, fit_nested_mechanisms_polled};
use crate::support::{IntoGraphInput, StructureSource};

mod artifact;
mod closed_form;
#[cfg(test)]
mod tests;

pub use artifact::{consume_cross_world_artifact, consume_cross_world_artifact_for_data};

/// Stage id reported on a [`CausalError::Cancelled`] raised by this cell.
const STAGE_CROSS_WORLD: &str = "cross_world";

/// What the caller asks of the cell beyond the query.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CrossWorldOptions {
    /// Structural mechanism family fitted for every variable.
    pub mechanism: NestedOutcomeMechanism,
    /// The caller asked for a sampling interval or Bayesian inference. Refused:
    /// the claim is point-only.
    pub interval_requested: bool,
}

impl Default for CrossWorldOptions {
    fn default() -> Self {
        Self { mechanism: NestedOutcomeMechanism::LinearGaussian, interval_requested: false }
    }
}

/// The point, its per-unit contrasts and the derivation that licensed it.
#[derive(Clone, Debug)]
pub struct CrossWorldEffect {
    /// Mean per-unit contrast.
    pub point: f64,
    /// Per-unit contrasts under the shared abduced exogenous terms.
    pub unit_effects: Arc<[f64]>,
    /// Derivation of the estimand.
    pub witness: CrossWorldWitness,
    /// The evaluated query.
    pub query: CrossWorldQuery,
    /// Mechanism family fitted.
    pub mechanism: NestedOutcomeMechanism,
    /// How exogenous terms were obtained.
    pub noise: NoiseInferenceKind,
    /// Digest of the factual table this point was computed on
    /// ([`cross_world_data_digest`]); compare it with a digest of your own data.
    pub data_digest: String,
    /// The consumer recomputed the point with a separate implementation (ordinary
    /// least squares for the linear-Gaussian family, a Givens-QR basis regression
    /// for the non-separable family) and it agreed. `false` on a freshly evaluated
    /// effect, and after consumption when the design is singular or ill
    /// conditioned beyond the check's floor (replay-verified only).
    pub independently_verified: bool,
    names: Vec<String>,
    columns: Vec<Vec<f64>>,
}

fn refusal_error(refusal: CrossWorldRefusal) -> CausalError {
    let CrossWorldRefusal { detail, message, .. } = refusal;
    match detail {
        "cross_world.query_outside_contract" => crate::compile_reason!(
            "route_not_supported",
            "cross_world.query_outside_contract: {message}"
        ),
        "cross_world.recanting_witness" => crate::compile_reason!(
            "cross_world_not_identified",
            "cross_world.recanting_witness: {message}"
        ),
        "cross_world.graph_outside_contract" => crate::compile_reason!(
            "cell_not_licensed",
            "cross_world.graph_outside_contract: {message}"
        ),
        _ => crate::compile_reason!("invalid_argument", "cross_world.invalid_query: {message}"),
    }
}

/// Stage one: check `query` on `graph`.
///
/// # Errors
///
/// A reason-coded refusal: `cross_world_not_identified` for a recanting witness
/// (the only nonidentification finding), `route_not_supported` for a query shape
/// outside the two-world contract, `cell_not_licensed` for a graph outside it,
/// `invalid_argument` for a malformed query.
pub fn check(graph: &Dag, query: &CrossWorldQuery) -> Result<CrossWorldWitness, CausalError> {
    check_cross_world(graph, query).map_err(refusal_error)
}

fn float_columns(data: &TabularData) -> Result<(Vec<String>, Vec<Vec<f64>>), CausalError> {
    let invalid = |message: String| {
        crate::compile_reason!("invalid_argument", "cross_world.invalid_query: {message}")
    };
    let mut names = Vec::new();
    let mut columns = Vec::new();
    for variable in data.schema().variables() {
        let ColumnView::Float64(column) =
            data.column(variable.id).map_err(|e| invalid(e.to_string()))?
        else {
            return Err(invalid(format!("variable {} is not a float64 column", variable.name)));
        };
        let values: Vec<f64> = column.values.as_slice().to_vec();
        if values.iter().any(|x| !x.is_finite()) {
            return Err(invalid(format!(
                "variable {} has missing or non-finite values",
                variable.name
            )));
        }
        names.push(variable.name.to_string());
        columns.push(values);
    }
    Ok((names, columns))
}

/// Evaluate `query` on `graph` and `data`: check, then one coupled
/// abduction-action-prediction operation, then the result.
///
/// # Errors
///
/// A reason-coded refusal (see [`check`]), `estimator_inference_mismatch` when
/// an interval or Bayesian inference is requested, `cell_not_licensed` for an
/// accepted or non-DAG structure, `invalid_argument` (`cross_world.invalid_query`)
/// when abduction is ill posed (a mechanism that cannot be inverted on the data,
/// or exogenous terms that are not exact inversions), [`CausalError::Cancelled`]
/// when the context is cancelled, or a fitting failure.
pub fn evaluate_cross_world_effect(
    graph: impl IntoGraphInput,
    data: &TabularData,
    query: &CrossWorldQuery,
    options: CrossWorldOptions,
    ctx: &ExecutionContext,
) -> Result<CrossWorldEffect, CausalError> {
    evaluate_effect_probed(graph, data, query, options, ctx, &|_| {})
}

/// [`evaluate_cross_world_effect`] with a hook called at every cancellation poll
/// of the fit (with the poll's index, before the token is read), so a test can
/// cancel the context in the middle of the fit.
fn evaluate_effect_probed(
    graph: impl IntoGraphInput,
    data: &TabularData,
    query: &CrossWorldQuery,
    options: CrossWorldOptions,
    ctx: &ExecutionContext,
    on_fit_poll: &dyn Fn(usize),
) -> Result<CrossWorldEffect, CausalError> {
    if options.interval_requested {
        return Err(crate::unsupported_reason!(
            "estimator_inference_mismatch",
            "cross_world.interval_requested: the cross-world edge contrast is a point claim"
        ));
    }
    let (structure, source) = graph.into_graph_input();
    let dag = match structure.graph_ref() {
        GraphRef::Dag(dag) if source == StructureSource::Explicit => dag,
        _ => {
            return Err(crate::unsupported_reason!(
                "cell_not_licensed",
                "cross_world.graph_outside_contract: only a supplied explicit DAG is licensed; \
                 accepted, uncertain, latent-confounded, equivalence-class and temporal \
                 structures are refused"
            ));
        }
    };
    if data.schema().variables().len() != dag.node_count() {
        return Err(crate::compile_reason!(
            "invalid_argument",
            "cross_world.invalid_query: the table has {} variables for a graph of {}",
            data.schema().variables().len(),
            dag.node_count()
        ));
    }
    let witness = check(dag, query)?;
    let (names, columns) = float_columns(data)?;
    let data_digest = cross_world_data_digest(&names, &columns)?;
    poll(ctx)?;
    // Stage two: fit the structural model, then abduct/act/predict as one operation.
    let compiled = CompiledCausalModel::compile(dag.clone()).map_err(CausalError::rejected)?;
    let outcome = VariableId::from_raw(witness.outcome);
    // The fit polls the token before every node, candidate family and
    // cross-validation fold of the model crate's fitting loops.
    let fit_polls = std::sync::atomic::AtomicUsize::new(0);
    let store = fit_nested_mechanisms_polled(
        &compiled,
        data,
        options.mechanism,
        outcome,
        STAGE_CROSS_WORLD,
        &|| {
            on_fit_poll(fit_polls.fetch_add(1, std::sync::atomic::Ordering::Relaxed));
            ctx.cancellation.is_cancelled()
        },
    )?;
    poll(ctx)?;
    let engine = CounterfactualEngine::new(compiled.with_mechanisms(store));
    let evaluation = evaluate_cross_world(&engine, data, query, ctx).map_err(evaluation_error)?;
    let unit_effects: Arc<[f64]> = Arc::from(evaluation.unit_effects());
    Ok(CrossWorldEffect {
        point: evaluation.point(),
        unit_effects,
        witness,
        query: query.clone(),
        mechanism: options.mechanism,
        noise: evaluation.noise,
        data_digest,
        independently_verified: false,
        names,
        columns,
    })
}

fn poll(ctx: &ExecutionContext) -> Result<(), CausalError> {
    if ctx.cancellation.is_cancelled() {
        Err(CausalError::Cancelled { stage: STAGE_CROSS_WORLD })
    } else {
        Ok(())
    }
}

/// Map an evaluator failure: ill-posed abduction keeps the `cross_world.invalid_query`
/// detail, cancellation is the facade's typed cancellation, anything else is a
/// compile failure.
fn evaluation_error(error: CounterfactualError) -> CausalError {
    match error {
        CounterfactualError::AbductionNotExact { message } => crate::compile_reason!(
            "invalid_argument",
            "cross_world.invalid_query: abduction is not exact inversion: {message}"
        ),
        CounterfactualError::Cancelled => CausalError::Cancelled { stage: STAGE_CROSS_WORLD },
        other => CausalError::rejected(other),
    }
}
