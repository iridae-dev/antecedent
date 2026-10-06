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
use crate::gcm::{NestedOutcomeMechanism, fit_nested_mechanisms_polled};
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
    /// The consumer recomputed the point with a separate implementation (ordinary
    /// least squares for the linear-Gaussian family, a Givens-QR basis regression
    /// for the non-separable family) and it agreed. `false` on a freshly evaluated
    /// effect, and after consumption when the design is singular or ill
    /// conditioned beyond the check's floor (replay-verified only).
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
/// This is deterministic replay by the same evaluator, plus, for both mechanism
/// families, a separate closed-form recomputation of the point
/// ([`CrossWorldEffect::independently_verified`]; skipped for a singular or
/// ill-conditioned design). It detects corruption and
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
    // Both mechanism families are recomputed from the stored table by a separate
    // implementation; `None` (a singular or refused design) leaves the effect
    // replay-verified only.
    if let Some(closed) = closed_form_point(
        n,
        &wire.graph_edges,
        &wire.columns,
        &query,
        mechanism,
        wire.witness.outcome,
    ) {
        if !closed_form_agrees(closed, effect.point) {
            return Err(CrossWorldArtifactError::IndependentCheckMismatch);
        }
        effect.independently_verified = true;
    }
    Ok(effect)
}

/// Ordinary least squares of `columns[v]` on `columns[parents]` with an intercept,
/// by the centered normal equations solved with Gaussian elimination and partial
/// pivoting: `(intercept, slopes)`, or `None` when the system is numerically
/// singular.
fn ols_on_parents(columns: &[Vec<f64>], v: usize, parents: &[usize]) -> Option<(f64, Vec<f64>)> {
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
    Some((intercept, beta))
}

// The non-separable family's estimator, restated from its specification and not
// from the fitter's code. The outcome is regressed on the expansion of its
// standardized parents: each parent's main effect, and for a parent with at
// least `SPLINE_MIN_DISTINCT` distinct standardized values its square, cube and
// one truncated cubic per interior knot (the 0.25 / 0.5 / 0.75 sample quantiles
// of the standardized column, nearest rank, duplicates removed); then, for every
// ordered pair of distinct parents (j, l), the main effect of j times every
// column of l, keeping the main-effect product z_j z_l once per pair. The
// least-squares problem is on columns scaled to unit root-mean-square, with a
// relative ridge on every column but the intercept: minimize
// |y - Xb|^2 + SPLINE_CONDITIONING_RIDGE * n * |b_columns|^2. The ridge is part of
// the estimator (the truncated power basis is nearly collinear), so the check
// solves the same penalized problem and differs only in numerics.
const SPLINE_KNOT_QUANTILES: [f64; 3] = [0.25, 0.50, 0.75];
const SPLINE_MIN_DISTINCT: usize = 5;
const SPLINE_CONDITIONING_RIDGE: f64 = 1e-6;
/// Rows required per fitted column; the fitter refuses below it, so no
/// comparison exists for such a table.
const SPLINE_MIN_ROWS_PER_COLUMN: usize = 10;

/// A fitted non-separable outcome mean: `intercept + coeffs . phi(z(parents))`.
struct SplineMean {
    intercept: f64,
    coeffs: Vec<f64>,
    centers: Vec<f64>,
    scales: Vec<f64>,
    knots: Vec<Vec<f64>>,
}

/// The expansion columns (no intercept) of one standardized parent row.
fn spline_columns(z: &[f64], knots: &[Vec<f64>], out: &mut Vec<f64>) {
    out.clear();
    let mut own: Vec<Vec<f64>> = Vec::with_capacity(z.len());
    for (p, &zp) in z.iter().enumerate() {
        let mut cols = vec![zp];
        if !knots[p].is_empty() {
            cols.push(zp * zp);
            cols.push(zp * zp * zp);
            cols.extend(knots[p].iter().map(|&t| if zp > t { (zp - t).powi(3) } else { 0.0 }));
        }
        own.push(cols);
    }
    for cols in &own {
        out.extend_from_slice(cols);
    }
    for (j, &zj) in z.iter().enumerate() {
        for (l, cols) in own.iter().enumerate() {
            if l == j {
                continue;
            }
            for (k, &c) in cols.iter().enumerate() {
                if k == 0 && l < j {
                    continue;
                }
                out.push(zj * c);
            }
        }
    }
}

impl SplineMean {
    fn mean(&self, raw: &[f64], z: &mut Vec<f64>, cols: &mut Vec<f64>) -> f64 {
        z.clear();
        z.extend(raw.iter().enumerate().map(|(p, x)| (x - self.centers[p]) / self.scales[p]));
        spline_columns(z, &self.knots, cols);
        self.intercept + self.coeffs.iter().zip(cols.iter()).map(|(b, c)| b * c).sum::<f64>()
    }
}

/// Interior knots per parent on the standardized scale: the sample quantiles in
/// `SPLINE_KNOT_QUANTILES` (nearest rank, duplicates removed), none for a parent
/// with fewer than `SPLINE_MIN_DISTINCT` distinct standardized values.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the rank is a rounded quantile of a non-empty vector, so it is non-negative and below its length"
)]
fn spline_knots(
    columns: &[Vec<f64>],
    parents: &[usize],
    centers: &[f64],
    scales: &[f64],
) -> Vec<Vec<f64>> {
    let mut knots: Vec<Vec<f64>> = Vec::with_capacity(parents.len());
    for (a, &j) in parents.iter().enumerate() {
        let mut sorted: Vec<f64> =
            columns[j].iter().map(|x| (x - centers[a]) / scales[a]).collect();
        sorted.sort_by(f64::total_cmp);
        let mut distinct = sorted.clone();
        distinct.dedup_by(|x, y| (*x - *y).abs() < 1e-9);
        if distinct.len() < SPLINE_MIN_DISTINCT {
            knots.push(Vec::new());
            continue;
        }
        let mut chosen: Vec<f64> = SPLINE_KNOT_QUANTILES
            .iter()
            .map(|q| sorted[((sorted.len() - 1) as f64 * q).round() as usize])
            .collect();
        chosen.dedup_by(|x, y| (*x - *y).abs() < 1e-9);
        knots.push(chosen);
    }
    knots
}

/// Fit the non-separable outcome mean of `columns[v]` on `columns[parents]` by a
/// streaming Givens QR of the ridge-augmented, unit-scaled design (no normal
/// equations, so the condition number is not squared): the augmented design's
/// smallest singular value is at least `sqrt(ridge)` against a largest of at most
/// `sqrt(n * ncols)`, a condition number below about `sqrt(ncols / 1e-6)`, so the
/// coefficients carry a relative error of order `1e-16 * cond`, far below the
/// `CLOSED_FORM_TOLERANCE`. `None` for a table the fitter refuses (fewer than two
/// parents, no parent with enough distinct values, too few rows per column) or a
/// numerically singular design.
fn fit_spline_mean(columns: &[Vec<f64>], v: usize, parents: &[usize]) -> Option<SplineMean> {
    let (n, p) = (columns[v].len(), parents.len());
    if p < 2 || n == 0 {
        return None;
    }
    let nf = n as f64;
    let mut centers = Vec::with_capacity(p);
    let mut scales = Vec::with_capacity(p);
    for &j in parents {
        let mean = columns[j].iter().sum::<f64>() / nf;
        let sd = (columns[j].iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / nf).sqrt();
        centers.push(mean);
        scales.push(if sd.is_finite() && sd > 1e-12 { sd } else { 1.0 });
    }
    let knots = spline_knots(columns, parents, &centers, &scales);
    if knots.iter().all(Vec::is_empty) {
        return None;
    }
    let row_of = |i: usize, z: &mut Vec<f64>, out: &mut Vec<f64>| {
        z.clear();
        z.extend(
            parents.iter().enumerate().map(|(a, &j)| (columns[j][i] - centers[a]) / scales[a]),
        );
        spline_columns(z, &knots, out);
    };
    let (mut z, mut cols) = (Vec::new(), Vec::new());
    row_of(0, &mut z, &mut cols);
    let m = 1 + cols.len();
    if n < m.saturating_mul(SPLINE_MIN_ROWS_PER_COLUMN) {
        return None;
    }
    // Pass one: root-mean-square of every non-intercept column.
    let mut rms = vec![0.0_f64; m];
    for i in 0..n {
        row_of(i, &mut z, &mut cols);
        for (c, x) in cols.iter().enumerate() {
            rms[1 + c] += x * x;
        }
    }
    let scale: Vec<f64> = rms
        .iter()
        .enumerate()
        .map(|(c, s)| {
            let r = (s / nf).sqrt();
            if c > 0 && r.is_finite() && r > 1e-12 { r } else { 1.0 }
        })
        .collect();
    // Pass two: rotate every data row, then every ridge row, into R and Q'y.
    let mut r = vec![vec![0.0_f64; m]; m];
    let mut rhs = vec![0.0_f64; m];
    let mut rotate = |row: &mut [f64], target: f64| {
        let mut t = target;
        for k in 0..m {
            let (a, b) = (r[k][k], row[k]);
            if b == 0.0 {
                continue;
            }
            let h = a.hypot(b);
            let (c, s) = (a / h, b / h);
            for (rj, xj) in r[k][k..].iter_mut().zip(row[k..].iter_mut()) {
                let (u, w) = (*rj, *xj);
                *rj = c * u + s * w;
                *xj = -s * u + c * w;
            }
            let (u, w) = (rhs[k], t);
            rhs[k] = c * u + s * w;
            t = -s * u + c * w;
        }
    };
    let mut row = vec![0.0_f64; m];
    for i in 0..n {
        row_of(i, &mut z, &mut cols);
        row[0] = 1.0;
        for (c, x) in cols.iter().enumerate() {
            row[1 + c] = x / scale[1 + c];
        }
        rotate(&mut row, columns[v][i]);
    }
    let sqrt_ridge = (SPLINE_CONDITIONING_RIDGE * nf).sqrt();
    for j in 1..m {
        row.fill(0.0);
        row[j] = sqrt_ridge;
        rotate(&mut row, 0.0);
    }
    let largest = (0..m).map(|k| r[k][k].abs()).fold(0.0_f64, f64::max);
    if !largest.is_finite() || (0..m).any(|k| r[k][k].abs() <= 1e-13 * largest) {
        return None;
    }
    let mut beta = vec![0.0_f64; m];
    for k in (0..m).rev() {
        let tail: f64 = ((k + 1)..m).map(|j| r[k][j] * beta[j]).sum();
        beta[k] = (rhs[k] - tail) / r[k][k];
    }
    let coeffs: Vec<f64> = (1..m).map(|c| beta[c] / scale[c]).collect();
    if beta.iter().any(|b| !b.is_finite()) {
        return None;
    }
    Some(SplineMean { intercept: beta[0], coeffs, centers, scales, knots })
}

/// One variable's fitted mean function in the independent check.
enum NodeMean {
    Linear { intercept: f64, slopes: Vec<f64> },
    Spline(SplineMean),
}

/// Whether the closed-form point agrees with the replayed one.
fn closed_form_agrees(closed: f64, replayed: f64) -> bool {
    (closed - replayed).abs() <= CLOSED_FORM_TOLERANCE * (1.0 + replayed.abs())
}

/// The point of a cross-world contrast from first principles, sharing no code
/// with the fitting or evaluation path: per variable a least-squares mean on its
/// parents (ordinary least squares by the centered normal equations with
/// Gaussian elimination and partial pivoting; for `outcome` under the
/// non-separable family the ridge-stabilized basis regression of
/// [`fit_spline_mean`] by Givens QR), each unit's exogenous term is its residual
/// about that mean, and each world is evaluated node by node in topological
/// order with every edge reading the world its route names.
///
/// Tolerance argument. The linear solve works on centered normal equations with
/// pivot floor `1e-12` of the largest diagonal, so its coefficient error is of
/// order `1e-16 * cond(S)`; the check declines (`None`) rather than compare when
/// the design is beyond that floor. The basis solve is bounded above (condition
/// number below about `5e3` by the ridge). Both errors enter the point through
/// one linear map of the residuals, so a relative `1e-6` on `1 + |point|` leaves
/// several orders of magnitude for legitimate rounding and is still far below any
/// mistake in a basis column, a knot, a scale, the ridge, a route or a noise
/// term (each of which moves the point by at least a fraction of the fit error).
///
/// `None` when a system is numerically singular or the fitter itself would have
/// refused the table, so no closed form exists to compare with.
fn closed_form_point(
    n: usize,
    edges: &[(u32, u32)],
    columns: &[Vec<f64>],
    query: &CrossWorldQuery,
    mechanism: NestedOutcomeMechanism,
    outcome: u32,
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
    // Per variable: the mean function and each unit's residual (exogenous term).
    let mut fits: Vec<NodeMean> = Vec::with_capacity(n);
    let mut residual: Vec<Vec<f64>> = vec![Vec::new(); n];
    let (mut z, mut cols) = (Vec::new(), Vec::new());
    for v in 0..n {
        let fit = if mechanism == NestedOutcomeMechanism::NonSeparableBasis && v == outcome as usize
        {
            NodeMean::Spline(fit_spline_mean(columns, v, &parents[v])?)
        } else {
            let (intercept, slopes) = ols_on_parents(columns, v, &parents[v])?;
            NodeMean::Linear { intercept, slopes }
        };
        let mut pv = vec![0.0; parents[v].len()];
        residual[v] = (0..rows)
            .map(|i| {
                for (a, &j) in parents[v].iter().enumerate() {
                    pv[a] = columns[j][i];
                }
                columns[v][i] - fit.mean(&pv, &mut z, &mut cols)
            })
            .collect();
        fits.push(fit);
    }
    let worlds = query.worlds();
    let mut values = vec![vec![Vec::<f64>::new(); n]; worlds.len()];
    let mut pv = Vec::new();
    for &v in &order {
        let variable = VariableId::from_raw(u32::try_from(v).ok()?);
        for (w, world) in worlds.iter().enumerate() {
            let column: Vec<f64> = if let Some(level) = world.intervention_of(variable) {
                vec![level; rows]
            } else {
                let sources: Vec<usize> = parents[v]
                    .iter()
                    .map(|&j| {
                        let parent = VariableId::from_raw(u32::try_from(j).ok()?);
                        Some(
                            world
                                .routes()
                                .iter()
                                .find(|r| r.parent == parent && r.child == variable)
                                .map_or(w, |r| r.source.index()),
                        )
                    })
                    .collect::<Option<Vec<usize>>>()?;
                (0..rows)
                    .map(|i| {
                        pv.clear();
                        pv.extend(parents[v].iter().zip(&sources).map(|(&j, &s)| values[s][j][i]));
                        fits[v].mean(&pv, &mut z, &mut cols) + residual[v][i]
                    })
                    .collect()
            };
            values[w][v] = column;
        }
    }
    let read =
        |o: antecedent_core::WorldObservation| &values[o.world.index()][o.variable.raw() as usize];
    let (plus, minus) = (read(query.plus()), read(query.minus()));
    Some(plus.iter().zip(minus).map(|(a, b)| a - b).sum::<f64>() / rows_f)
}

impl NodeMean {
    fn mean(&self, parent_values: &[f64], z: &mut Vec<f64>, cols: &mut Vec<f64>) -> f64 {
        match self {
            Self::Linear { intercept, slopes } => {
                let mut y = *intercept;
                for (b, x) in slopes.iter().zip(parent_values) {
                    y += b * x;
                }
                y
            }
            Self::Spline(spline) => spline.mean(parent_values, z, cols),
        }
    }
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
                let closed = closed_form_point(
                    3,
                    &edges,
                    &columns,
                    &query,
                    NestedOutcomeMechanism::LinearGaussian,
                    2,
                )
                .unwrap();
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
            closed_form_point(
                3,
                &[(0, 1), (0, 2), (1, 2)],
                &columns,
                &query,
                NestedOutcomeMechanism::LinearGaussian,
                2
            )
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

    /// A noisy non-separable table: the outcome interacts the treatment with the
    /// squared mediator, with a per-unit outcome disturbance.
    fn non_separable_table(n: usize) -> (TabularData, Vec<Vec<f64>>, Dag) {
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin() * 3.0).collect();
        let m: Vec<f64> =
            x.iter().enumerate().map(|(i, x)| 0.8 * x + (i as f64 * 0.91).cos()).collect();
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
        (data, vec![x, m, y], graph)
    }

    /// The independent basis regression (Givens QR, restated basis, knots,
    /// scaling and ridge) reproduces the fitter's point for every edge set, and
    /// the check is not vacuous: the point is far from the linear closed form.
    #[test]
    fn the_non_separable_closed_form_matches_the_fit_for_every_edge_set() {
        let v = VariableId::from_raw;
        let (data, columns, graph) = non_separable_table(1500);
        let edges = [(0, 1), (0, 2), (1, 2)];
        let all = edges.map(|(a, b)| (v(a), v(b)));
        let ctx = ExecutionContext::for_tests(4);
        let mut worst = 0.0_f64;
        for mask in 0u32..8 {
            let chosen: Vec<_> = all
                .iter()
                .enumerate()
                .filter(|(k, _)| (mask >> k) & 1 == 1)
                .map(|(_, e)| *e)
                .collect();
            let query =
                CrossWorldQuery::path_specific(v(0), v(2), -1.0, 2.0, &all, &chosen).unwrap();
            let replayed = evaluate_cross_world_effect(
                graph.clone(),
                &data,
                &query,
                CrossWorldOptions {
                    mechanism: NestedOutcomeMechanism::NonSeparableBasis,
                    interval_requested: false,
                },
                &ctx,
            )
            .unwrap()
            .point;
            let closed = closed_form_point(
                3,
                &edges,
                &columns,
                &query,
                NestedOutcomeMechanism::NonSeparableBasis,
                2,
            )
            .expect("a well-conditioned table has a closed form");
            worst = worst.max((closed - replayed).abs() / (1.0 + replayed.abs()));
            assert!(closed_form_agrees(closed, replayed), "{mask}: {closed} vs {replayed}");
            let linear = closed_form_point(
                3,
                &edges,
                &columns,
                &query,
                NestedOutcomeMechanism::LinearGaussian,
                2,
            )
            .unwrap();
            if mask == 0b101 {
                // The natural indirect edge set reads the interaction, which a
                // linear outcome cannot represent.
                assert!(
                    (linear - replayed).abs() > 1.0,
                    "the linear closed form must not pass as the basis one: {linear} vs {replayed}"
                );
            }
        }
        assert!(worst < 1e-8, "independent solves should agree far inside the tolerance: {worst}");
    }

    /// Mutating the fitted mechanism (an edited outcome coefficient, and an edited
    /// knot) gives a point the replay would accept, because the same evaluator
    /// recomputes it from the same edited mechanism, while the independent
    /// recomputation from the table disagrees.
    #[test]
    fn an_edited_mechanism_is_caught_by_the_independent_check_but_not_by_replay() {
        use antecedent_model::{MechanismSlot, ParentBasis};
        let v = VariableId::from_raw;
        let (data, columns, graph) = non_separable_table(1500);
        let edges = [(0, 1), (0, 2), (1, 2)];
        let all = edges.map(|(a, b)| (v(a), v(b)));
        let query =
            CrossWorldQuery::path_specific(v(0), v(2), -1.0, 2.0, &all, &all[1..2]).unwrap();
        let compiled = CompiledCausalModel::compile(graph).unwrap();
        let store = crate::gcm::fit_nested_mechanisms(
            &compiled,
            &data,
            NestedOutcomeMechanism::NonSeparableBasis,
            v(2),
        )
        .unwrap();
        let outcome = compiled.dense_of(v(2)).unwrap();
        let MechanismSlot::LinearBasis { intercept, basis, coeffs, sigma } =
            store.get(outcome).clone()
        else {
            panic!("the outcome is a basis mechanism");
        };
        let closed = closed_form_point(
            3,
            &edges,
            &columns,
            &query,
            NestedOutcomeMechanism::NonSeparableBasis,
            2,
        )
        .unwrap();
        let ctx = ExecutionContext::for_tests(5);
        let point_of = |slot: MechanismSlot| {
            let engine = CounterfactualEngine::new(
                compiled.clone().with_mechanisms(store.with_replaced(outcome, slot).unwrap()),
            );
            let first = evaluate_cross_world(&engine, &data, &query, &ctx).unwrap().point();
            // Replay by the same evaluator over the same (edited) mechanism is
            // bit-identical, so replay alone accepts the edit.
            let second = evaluate_cross_world(&engine, &data, &query, &ctx).unwrap().point();
            assert_eq!(first.to_bits(), second.to_bits());
            first
        };
        let untouched = point_of(MechanismSlot::LinearBasis {
            intercept,
            basis: basis.clone(),
            coeffs: coeffs.clone(),
            sigma,
        });
        assert!(closed_form_agrees(closed, untouched), "unmutated: {closed} vs {untouched}");
        // (1) The coefficient of the first cross-parent product column edited.
        let mut edited: Vec<f64> = coeffs.to_vec();
        let product = basis
            .terms()
            .iter()
            .position(|t| matches!(t, antecedent_model::BasisTerm::Product { .. }))
            .expect("a cross-parent product column");
        edited[product] += 0.5;
        let coefficient = point_of(MechanismSlot::LinearBasis {
            intercept,
            basis: basis.clone(),
            coeffs: Arc::from(edited),
            sigma,
        });
        assert!(!closed_form_agrees(closed, coefficient), "coefficient edit: {coefficient}");
        // (2) One knot of the basis moved.
        let mut knots: Vec<Arc<[f64]>> = basis.knots().to_vec();
        let moved: Vec<f64> = knots[1].iter().map(|k| k + 0.4).collect();
        knots[1] = Arc::from(moved);
        let shifted = ParentBasis::new(
            basis.n_parents(),
            Arc::from(basis.centers().to_vec()),
            Arc::from(basis.scales().to_vec()),
            Arc::from(knots),
            Arc::from(basis.terms().to_vec()),
        )
        .unwrap();
        let knot = point_of(MechanismSlot::LinearBasis {
            intercept,
            basis: shifted,
            coeffs: coeffs.clone(),
            sigma,
        });
        assert!(!closed_form_agrees(closed, knot), "knot edit: {knot}");
    }

    /// The consumer reports the non-separable family as independently verified,
    /// and declines (replay only) when the outcome design has no closed form.
    #[test]
    fn the_non_separable_closed_form_declines_designs_the_fitter_refuses() {
        let v = VariableId::from_raw;
        let edges = [(0, 1), (0, 2), (1, 2)];
        let all = edges.map(|(a, b)| (v(a), v(b)));
        let query = CrossWorldQuery::path_specific(v(0), v(2), 0.0, 1.0, &all, &all[..1]).unwrap();
        let (_, columns, _) = non_separable_table(1500);
        // Too few rows for the columns of the expansion.
        let small: Vec<Vec<f64>> = columns.iter().map(|c| c[..60].to_vec()).collect();
        let decline = |columns: &[Vec<f64>], outcome: u32| {
            closed_form_point(
                3,
                &edges,
                columns,
                &query,
                NestedOutcomeMechanism::NonSeparableBasis,
                outcome,
            )
        };
        assert!(decline(&small, 2).is_none());
        // Every parent of the outcome binary: too few distinct values for a spline.
        let binary = |k: usize| -> Vec<f64> { (0..1500).map(|i| ((i / k) % 2) as f64).collect() };
        let flat = vec![binary(1), binary(3), columns[2].clone()];
        let direct = [(v(0), v(2)), (v(1), v(2))];
        let flat_query =
            CrossWorldQuery::path_specific(v(0), v(2), 0.0, 1.0, &direct, &direct[..1]).unwrap();
        assert!(
            closed_form_point(
                3,
                &[(0, 2), (1, 2)],
                &flat,
                &flat_query,
                NestedOutcomeMechanism::NonSeparableBasis,
                2
            )
            .is_none()
        );
        // An outcome with a single parent has no cross-parent product.
        assert!(
            closed_form_point(
                3,
                &[(0, 1), (1, 2)],
                &columns,
                &query,
                NestedOutcomeMechanism::NonSeparableBasis,
                2
            )
            .is_none()
        );
    }

    /// A cancellation raised in the middle of the fit stops it with the typed
    /// error and no effect, for both mechanism families; the same evaluation
    /// without cancellation polls the token many more times and answers.
    #[test]
    fn a_context_cancelled_mid_fit_stops_the_fit_without_a_result() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let v = VariableId::from_raw;
        let (data, _, graph) = non_separable_table(400);
        let all = [(v(0), v(1)), (v(0), v(2)), (v(1), v(2))];
        let query = CrossWorldQuery::path_specific(v(0), v(2), 0.0, 1.0, &all, &all[..1]).unwrap();
        for mechanism in
            [NestedOutcomeMechanism::LinearGaussian, NestedOutcomeMechanism::NonSeparableBasis]
        {
            let options = CrossWorldOptions { mechanism, interval_requested: false };
            let total = AtomicUsize::new(0);
            let ok = evaluate_effect_probed(
                graph.clone(),
                &data,
                &query,
                options,
                &ExecutionContext::for_tests(6),
                &|_| {
                    total.fetch_add(1, Ordering::Relaxed);
                },
            )
            .unwrap();
            let total = total.into_inner();
            // Three nodes: one poll per node and candidate family, and one per fold.
            assert!(total >= 20, "{mechanism:?}: only {total} fit polls");
            assert!(ok.point.is_finite());
            for at in [0, 1, total / 3, total / 2, total - 1] {
                let ctx = ExecutionContext::for_tests(6);
                let seen = AtomicUsize::new(0);
                let outcome =
                    evaluate_effect_probed(graph.clone(), &data, &query, options, &ctx, &|k| {
                        seen.fetch_add(1, Ordering::Relaxed);
                        if k == at {
                            ctx.cancellation.cancel();
                        }
                    });
                assert!(
                    matches!(outcome, Err(CausalError::Cancelled { stage: "cross_world" })),
                    "{mechanism:?} at poll {at}: {outcome:?}"
                );
                assert_eq!(
                    seen.into_inner(),
                    at + 1,
                    "{mechanism:?}: the fit kept going after the cancelling poll {at}"
                );
            }
        }
    }
}
