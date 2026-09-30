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
//!    evaluator, plus a separate closed-form OLS cross-check for the
//!    linear-Gaussian family.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{CrossWorldQuery, ExecutionContext, VariableId};
use antecedent_counterfactual::{
    CounterfactualEngine, CounterfactualError, NoiseInferenceKind,
    cross_world::evaluate_cross_world,
};
use antecedent_data::{ColumnView, TableView, TabularData};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_identify::cross_world::{CrossWorldRefusal, CrossWorldWitness, check_cross_world};
pub use antecedent_io::cross_world_artifact::CROSS_WORLD_MAX_ROWS;
use antecedent_io::cross_world_artifact::{
    CROSS_WORLD_ARTIFACT_FEATURE, CROSS_WORLD_ARTIFACT_VERSION, CROSS_WORLD_ESTIMAND,
    CrossWorldArtifactError, CrossWorldArtifactWire, CrossWorldQueryWire, cross_world_data_digest,
    cross_world_identity,
};
use antecedent_model::CompiledCausalModel;

use crate::GraphClass;
use crate::error::CausalError;
use crate::gcm::{NestedOutcomeMechanism, fit_nested_mechanisms};
use crate::support::{IntoGraphInput, StructureSource};

/// Stage id reported on a [`CausalError::Cancelled`] raised by this cell.
const STAGE_CROSS_WORLD: &str = "cross_world";

/// Relative tolerance of the closed-form cross-check against the replayed point.
const CLOSED_FORM_TOLERANCE: f64 = 1e-6;

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
    /// The consumer recomputed the point with a separate closed-form
    /// implementation and it agreed (linear-Gaussian family only; always `false`
    /// on a freshly evaluated effect).
    pub independently_verified: bool,
    names: Vec<String>,
    columns: Vec<Vec<f64>>,
}

fn mechanism_tag(mechanism: NestedOutcomeMechanism) -> &'static str {
    match mechanism {
        NestedOutcomeMechanism::LinearGaussian => "linear_gaussian",
        NestedOutcomeMechanism::NonSeparableBasis => "non_separable_basis",
    }
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
    if options.interval_requested {
        return Err(crate::unsupported_reason!(
            "estimator_inference_mismatch",
            "cross_world.interval_requested: the cross-world edge contrast is a point claim"
        ));
    }
    let (structure, source) = graph.into_graph_input();
    let Some(dag) = structure
        .as_dag()
        .filter(|_| source == StructureSource::Explicit && structure.class() == GraphClass::Dag)
    else {
        return Err(crate::unsupported_reason!(
            "cell_not_licensed",
            "cross_world.graph_outside_contract: only a supplied explicit DAG is licensed; \
             accepted, uncertain, latent-confounded, equivalence-class and temporal structures \
             are refused"
        ));
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
    let compiled = CompiledCausalModel::compile(dag.clone())
        .map_err(|e| CausalError::Compile { message: e.to_string() })?;
    let outcome = VariableId::from_raw(witness.outcome);
    let store = fit_nested_mechanisms(&compiled, data, options.mechanism, outcome)?;
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
        other => CausalError::Compile { message: other.to_string() },
    }
}

impl CrossWorldEffect {
    /// Export the replayable artifact.
    ///
    /// # Errors
    ///
    /// [`CausalError::Resource`] when the table exceeds the format's bounds
    /// (the consumer would refuse it), or an encoding failure.
    pub fn export_artifact(&self) -> Result<Vec<u8>, CausalError> {
        self.artifact_wire().and_then(|wire| wire.export()).map_err(|error| match error {
            CrossWorldArtifactError::LimitsExceeded(what) => CausalError::Resource {
                message: format!(
                    "the cross-world artifact format does not carry this table ({what} exceed \
                     the format bound of {CROSS_WORLD_MAX_ROWS} rows and {} variables)",
                    antecedent_identify::cross_world::CROSS_WORLD_MAX_NODES
                ),
            },
            other => CausalError::Serialization(antecedent_io::IoError::Convert(other.to_string())),
        })
    }

    fn artifact_wire(&self) -> Result<CrossWorldArtifactWire, CrossWorldArtifactError> {
        CrossWorldArtifactWire {
            version: CROSS_WORLD_ARTIFACT_VERSION,
            required_features: vec![CROSS_WORLD_ARTIFACT_FEATURE.into()],
            variable_names: self.names.clone(),
            graph_edges: self.witness.graph_edges.clone(),
            query: CrossWorldQueryWire::from_query(&self.query),
            mechanism: mechanism_tag(self.mechanism).into(),
            estimand: CROSS_WORLD_ESTIMAND.into(),
            columns: self.columns.clone(),
            witness: self.witness.clone(),
            point_bits: self.point.to_bits(),
            data_digest: String::new(),
            premises_digest: String::new(),
        }
        .sealed()
    }
}

/// Replay a cross-world artifact from its premises and accept it only when the
/// digests, the witness and the point all replay.
///
/// This is deterministic replay by the same evaluator, plus, for the
/// linear-Gaussian family, a separate closed-form OLS recomputation of the point
/// ([`CrossWorldEffect::independently_verified`]). It detects corruption and
/// re-sealed edits that change the derivation or the point; it does not detect a
/// producer that seals a wrong table or query on purpose.
///
/// # Errors
///
/// A [`CrossWorldArtifactError`] naming the failed check.
pub fn consume_cross_world_artifact(
    bytes: &[u8],
    ctx: &ExecutionContext,
) -> Result<CrossWorldEffect, CrossWorldArtifactError> {
    consume_inner(bytes, None, ctx)
}

/// [`consume_cross_world_artifact`] that also requires the artifact's factual
/// table to have the caller's data digest ([`cross_world_data_digest`]).
///
/// # Errors
///
/// As [`consume_cross_world_artifact`], and
/// [`CrossWorldArtifactError::DataIdentityMismatch`] when the artifact was
/// computed on other data.
pub fn consume_cross_world_artifact_for_data(
    bytes: &[u8],
    expected_data_digest: &str,
    ctx: &ExecutionContext,
) -> Result<CrossWorldEffect, CrossWorldArtifactError> {
    consume_inner(bytes, Some(expected_data_digest), ctx)
}

fn consume_inner(
    bytes: &[u8],
    expected_data_digest: Option<&str>,
    ctx: &ExecutionContext,
) -> Result<CrossWorldEffect, CrossWorldArtifactError> {
    let wire = CrossWorldArtifactWire::decode(bytes)?;
    wire.check_bounds()?;
    let n = wire.variable_names.len();
    if cross_world_identity(&wire).map_err(|e| CrossWorldArtifactError::Malformed(e.to_string()))?
        != wire.premises_digest
    {
        return Err(CrossWorldArtifactError::PremisesMismatch);
    }
    let table_digest = cross_world_data_digest(&wire.variable_names, &wire.columns)
        .map_err(|e| CrossWorldArtifactError::Malformed(e.to_string()))?;
    if table_digest != wire.data_digest
        || expected_data_digest.is_some_and(|expected| expected != table_digest)
    {
        return Err(CrossWorldArtifactError::DataIdentityMismatch);
    }
    if wire.estimand != CROSS_WORLD_ESTIMAND {
        return Err(CrossWorldArtifactError::UnsupportedSemantics("estimand"));
    }
    let mechanism = match wire.mechanism.as_str() {
        "linear_gaussian" => NestedOutcomeMechanism::LinearGaussian,
        "non_separable_basis" => NestedOutcomeMechanism::NonSeparableBasis,
        _ => return Err(CrossWorldArtifactError::UnsupportedSemantics("mechanism family")),
    };
    let query = wire.query.to_query()?;
    let bound = u32::try_from(n).unwrap_or(u32::MAX);
    let mut dag = Dag::with_variables(bound);
    for &(a, b) in &wire.graph_edges {
        if a >= bound || b >= bound {
            return Err(CrossWorldArtifactError::Malformed(
                "edge names an unknown variable".into(),
            ));
        }
        dag.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b))
            .map_err(|e| CrossWorldArtifactError::Malformed(e.to_string()))?;
    }
    // The witness is verified against its own stored graph, so the graph the
    // replay uses must be that one.
    let mut stored_edges = wire.graph_edges.clone();
    stored_edges.sort_unstable();
    stored_edges.dedup();
    if stored_edges != wire.witness.graph_edges {
        return Err(CrossWorldArtifactError::WitnessMismatch(
            "the witness was derived on a different graph than the artifact stores".into(),
        ));
    }
    wire.witness.verify(n, &query).map_err(CrossWorldArtifactError::WitnessMismatch)?;
    let data = TabularData::from_f64_columns(
        wire.variable_names.iter().map(String::as_str).zip(wire.columns.iter().map(Vec::as_slice)),
    )
    .map_err(|e| CrossWorldArtifactError::Malformed(e.to_string()))?;
    let mut effect = evaluate_cross_world_effect(
        dag,
        &data,
        &query,
        CrossWorldOptions { mechanism, interval_requested: false },
        ctx,
    )
    .map_err(|e| CrossWorldArtifactError::Replay(e.to_string()))?;
    if effect.point.to_bits() != wire.point_bits || effect.witness != wire.witness {
        return Err(CrossWorldArtifactError::PointMismatch);
    }
    if mechanism == NestedOutcomeMechanism::LinearGaussian {
        if let Some(closed) =
            closed_form_linear_gaussian_point(n, &wire.graph_edges, &wire.columns, &query)
        {
            if !closed_form_agrees(closed, effect.point) {
                return Err(CrossWorldArtifactError::IndependentCheckMismatch);
            }
            effect.independently_verified = true;
        }
    }
    Ok(effect)
}

/// Ordinary least squares of `columns[v]` on `columns[parents]` with an intercept,
/// by the centered normal equations solved with Gaussian elimination and partial
/// pivoting: `(intercept, slopes, residuals)`, or `None` when the system is
/// numerically singular.
fn ols_on_parents(
    columns: &[Vec<f64>],
    v: usize,
    parents: &[usize],
) -> Option<(f64, Vec<f64>, Vec<f64>)> {
    let rows = columns[v].len();
    let p = parents.len();
    let mean = |c: &[f64]| c.iter().sum::<f64>() / rows as f64;
    let ybar = mean(&columns[v]);
    let xbar: Vec<f64> = parents.iter().map(|&j| mean(&columns[j])).collect();
    // Augmented [S | b] with S = Xc'Xc and b = Xc'yc.
    let mut system = vec![vec![0.0; p + 1]; p];
    for a in 0..p {
        for b in 0..p {
            system[a][b] = (0..rows)
                .map(|i| (columns[parents[a]][i] - xbar[a]) * (columns[parents[b]][i] - xbar[b]))
                .sum();
        }
        system[a][p] =
            (0..rows).map(|i| (columns[parents[a]][i] - xbar[a]) * (columns[v][i] - ybar)).sum();
    }
    let scale = (0..p).map(|a| system[a][a]).fold(0.0_f64, f64::max);
    for col in 0..p {
        let pivot =
            (col..p).max_by(|&a, &b| system[a][col].abs().total_cmp(&system[b][col].abs()))?;
        if system[pivot][col].abs() <= 1e-12 * scale.max(f64::MIN_POSITIVE) {
            return None;
        }
        system.swap(col, pivot);
        for r in 0..p {
            if r != col {
                let factor = system[r][col] / system[col][col];
                for c in col..=p {
                    let value = system[col][c];
                    system[r][c] -= factor * value;
                }
            }
        }
    }
    let beta: Vec<f64> = (0..p).map(|a| system[a][p] / system[a][a]).collect();
    let intercept = ybar - beta.iter().zip(&xbar).map(|(b, x)| b * x).sum::<f64>();
    let residual = (0..rows)
        .map(|i| {
            columns[v][i]
                - intercept
                - beta.iter().enumerate().map(|(a, b)| b * columns[parents[a]][i]).sum::<f64>()
        })
        .collect();
    Some((intercept, beta, residual))
}

/// Whether the closed-form point agrees with the replayed one.
fn closed_form_agrees(closed: f64, replayed: f64) -> bool {
    (closed - replayed).abs() <= CLOSED_FORM_TOLERANCE * (1.0 + replayed.abs())
}

/// The point of a linear-Gaussian cross-world contrast from first principles,
/// sharing no code with the fitting or evaluation path: ordinary least squares
/// per variable on its parents by the centered normal equations (Gaussian
/// elimination with partial pivoting), each unit's exogenous term is its
/// residual, and each world is evaluated node by node in topological order with
/// every edge reading the world its route names.
///
/// `None` when a normal-equation system is numerically singular (the design has
/// no unique least-squares solution), so no closed form exists to compare with.
fn closed_form_linear_gaussian_point(
    n: usize,
    edges: &[(u32, u32)],
    columns: &[Vec<f64>],
    query: &CrossWorldQuery,
) -> Option<f64> {
    let rows = columns.first()?.len();
    let rows_f = rows as f64;
    let parents: Vec<Vec<usize>> = (0..n)
        .map(|v| edges.iter().filter(|e| e.1 as usize == v).map(|e| e.0 as usize).collect())
        .collect();
    // Topological order (Kahn).
    let mut indegree: Vec<usize> = parents.iter().map(Vec::len).collect();
    let mut order = Vec::with_capacity(n);
    let mut ready: Vec<usize> = (0..n).filter(|&v| indegree[v] == 0).collect();
    while let Some(v) = ready.pop() {
        order.push(v);
        for &(a, b) in edges {
            if a as usize == v {
                indegree[b as usize] -= 1;
                if indegree[b as usize] == 0 {
                    ready.push(b as usize);
                }
            }
        }
    }
    if order.len() != n {
        return None;
    }
    // Per variable: intercept, slopes (aligned with `parents[v]`) and residuals.
    let mut intercept = vec![0.0; n];
    let mut slopes: Vec<Vec<f64>> = vec![Vec::new(); n];
    let mut residual: Vec<Vec<f64>> = vec![Vec::new(); n];
    for v in 0..n {
        (intercept[v], slopes[v], residual[v]) = ols_on_parents(columns, v, &parents[v])?;
    }
    let worlds = query.worlds();
    let mut values = vec![vec![Vec::<f64>::new(); n]; worlds.len()];
    for &v in &order {
        let variable = VariableId::from_raw(u32::try_from(v).ok()?);
        for (w, world) in worlds.iter().enumerate() {
            let column: Vec<f64> = if let Some(level) = world.intervention_of(variable) {
                vec![level; rows]
            } else {
                (0..rows)
                    .map(|i| {
                        let mut y = intercept[v] + residual[v][i];
                        for (a, &j) in parents[v].iter().enumerate() {
                            let parent = VariableId::from_raw(u32::try_from(j).ok()?);
                            let source = world
                                .routes()
                                .iter()
                                .find(|r| r.parent == parent && r.child == variable)
                                .map_or(w, |r| r.source.index());
                            y += slopes[v][a] * values[source][j][i];
                        }
                        Some(y)
                    })
                    .collect::<Option<Vec<f64>>>()?
            };
            values[w][v] = column;
        }
    }
    let read =
        |o: antecedent_core::WorldObservation| &values[o.world.index()][o.variable.raw() as usize];
    let (plus, minus) = (read(query.plus()), read(query.minus()));
    Some(plus.iter().zip(minus).map(|(a, b)| a - b).sum::<f64>() / rows_f)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gcm::NestedCounterfactualOperation;
    use antecedent_core::NestedCounterfactualQuery;

    #[test]
    fn ill_posed_abduction_keeps_the_invalid_query_detail() {
        let error =
            evaluation_error(CounterfactualError::AbductionNotExact { message: "probe".into() });
        let text = error.to_string();
        assert!(
            text.contains("cross_world.invalid_query: abduction is not exact inversion: probe"),
            "{text}"
        );
        assert!(text.contains("invalid_argument"), "{text}");
        assert!(matches!(
            evaluation_error(CounterfactualError::Cancelled),
            CausalError::Cancelled { stage: "cross_world" }
        ));
    }

    fn mediation_columns(noisy: bool) -> Vec<Vec<f64>> {
        let n = 400usize;
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
        let m: Vec<f64> =
            x.iter().enumerate().map(|(i, x)| 0.8 * x + (i as f64 * 0.91).cos()).collect();
        let y: Vec<f64> = x
            .iter()
            .zip(&m)
            .enumerate()
            .map(|(i, (x, m))| {
                1.7 * x + 4.0 * m + if noisy { (i as f64 * 1.7 + 0.3).sin() } else { 0.0 }
            })
            .collect();
        vec![x, m, y]
    }

    /// The closed-form implementation is checked against the structural truth
    /// (the sample disturbances are only approximately orthogonal to the
    /// regressors, so within 0.1) and against the evaluator (1e-9) for every edge
    /// set: the outcome disturbance cancels, the mediator path contributes
    /// 4 * 0.8 * 3.
    #[test]
    fn the_closed_form_recomputation_matches_the_structural_truth() {
        let edges = [(0, 1), (0, 2), (1, 2)];
        let v = VariableId::from_raw;
        let all = edges.map(|(a, b)| (v(a), v(b)));
        for noisy in [false, true] {
            let columns = mediation_columns(noisy);
            for mask in 0u32..8 {
                let chosen: Vec<_> = all
                    .iter()
                    .enumerate()
                    .filter(|(k, _)| (mask >> k) & 1 == 1)
                    .map(|(_, e)| *e)
                    .collect();
                let query =
                    CrossWorldQuery::path_specific(v(0), v(2), -1.0, 2.0, &all, &chosen).unwrap();
                let truth = 1.7 * 3.0 * f64::from(u8::from(chosen.contains(&all[1])))
                    + 4.0
                        * 0.8
                        * 3.0
                        * f64::from(u8::from(chosen.contains(&all[0]) && chosen.contains(&all[2])));
                let closed =
                    closed_form_linear_gaussian_point(3, &edges, &columns, &query).unwrap();
                assert!(
                    (closed - truth).abs() < 0.1,
                    "noisy={noisy} mask={mask}: {closed} vs {truth}"
                );
                let data = TabularData::from_f64_columns([
                    ("x", columns[0].as_slice()),
                    ("m", columns[1].as_slice()),
                    ("y", columns[2].as_slice()),
                ])
                .unwrap();
                let mut graph = Dag::with_variables(3);
                for (a, b) in edges {
                    graph
                        .insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b))
                        .unwrap();
                }
                let evaluated = evaluate_cross_world_effect(
                    graph,
                    &data,
                    &query,
                    CrossWorldOptions::default(),
                    &ExecutionContext::for_tests(1),
                )
                .unwrap()
                .point;
                assert!(closed_form_agrees(closed, evaluated), "{mask}: {closed} vs {evaluated}");
                assert!((closed - evaluated).abs() < 1e-9, "{mask}: {closed} vs {evaluated}");
            }
        }
    }

    /// A singular design has no closed form to compare with, and the comparison
    /// tolerance is relative.
    #[test]
    fn the_closed_form_declines_singular_designs_and_compares_relatively() {
        let v = VariableId::from_raw;
        let x: Vec<f64> = (0..50).map(|i| (f64::from(i) * 0.37).sin()).collect();
        let y: Vec<f64> = (0..50).map(|i| (f64::from(i) * 1.1).cos()).collect();
        // The outcome has two identical parents: X twice cannot happen in a DAG,
        // so make the mediator an exact copy of the treatment.
        let columns = vec![x.clone(), x, y];
        let all = [(v(0), v(1)), (v(0), v(2)), (v(1), v(2))];
        let query = CrossWorldQuery::path_specific(v(0), v(2), 0.0, 1.0, &all, &all[1..2]).unwrap();
        assert!(
            closed_form_linear_gaussian_point(3, &[(0, 1), (0, 2), (1, 2)], &columns, &query)
                .is_none()
        );
        assert!(closed_form_agrees(1000.0, 1000.0 + 5e-4));
        assert!(!closed_form_agrees(1000.0, 1000.0 + 5e-2));
        assert!(!closed_form_agrees(0.0, 1e-3));
    }

    /// The natural direct effect as an edge contrast equals the existing nested
    /// operation on both the separable and the non-separable mechanism.
    #[test]
    fn natural_direct_edge_contrast_reproduces_the_existing_nested_operation() {
        let n = 2000usize;
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin() * 3.0).collect();
        let m: Vec<f64> =
            x.iter().enumerate().map(|(i, x)| 0.8 * x + (i as f64 * 0.91).cos()).collect();
        // Outcome noise, so the regression covers a per-unit outcome disturbance.
        let y: Vec<f64> = x
            .iter()
            .zip(&m)
            .enumerate()
            .map(|(i, (x, m))| 1.7 * x + 0.5 * m + 0.9 * x * m * m + (i as f64 * 1.7 + 0.3).sin())
            .collect();
        let data = TabularData::from_f64_columns([
            ("x", x.as_slice()),
            ("m", m.as_slice()),
            ("y", y.as_slice()),
        ])
        .unwrap();
        let mut graph = Dag::with_variables(3);
        for (a, b) in [(0, 1), (0, 2), (1, 2)] {
            graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        let (t, mid, out) =
            (VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2));
        let nested = NestedCounterfactualQuery::with_levels(t, mid, out, -1.0, 2.0).unwrap();
        let direct = CrossWorldQuery::natural_direct(t, mid, out, -1.0, 2.0).unwrap();
        let ctx = ExecutionContext::for_tests(8);
        for mechanism in
            [NestedOutcomeMechanism::LinearGaussian, NestedOutcomeMechanism::NonSeparableBasis]
        {
            let mut operation =
                NestedCounterfactualOperation::compile(graph.clone(), nested).unwrap();
            if mechanism == NestedOutcomeMechanism::NonSeparableBasis {
                operation = operation.with_non_separable_outcome();
            }
            let existing = operation.execute(&data, &ctx).unwrap();
            let mine = evaluate_cross_world_effect(
                graph.clone(),
                &data,
                &direct,
                CrossWorldOptions { mechanism, interval_requested: false },
                &ctx,
            )
            .unwrap()
            .point;
            assert!((mine - existing).abs() < 1e-9, "{mechanism:?}: {mine} vs {existing}");
        }
    }
}
