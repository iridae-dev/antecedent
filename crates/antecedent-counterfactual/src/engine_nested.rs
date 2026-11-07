// SPDX-License-Identifier: MIT OR Apache-2.0
//! Nested counterfactual queries and frozen-world evaluation.

use super::{
    AbductionMissingPolicy, Arc, CounterfactualEngine, CounterfactualError, CounterfactualResult,
    CounterfactualWorld, DenseNodeId, ExecutionContext, ExogenousPosterior, Intervention,
    InterventionOverlay, MechanismSlot, MechanismWorkspace, PREDICT_STREAM, StreamDomain,
    TabularData, ValueBatchMut, VariableId, WorldEvaluation, complete_outcome_mean,
    evaluate_world_into,
};

pub(super) fn interventions_overlap(outer: &[Intervention], inner: &[Intervention]) -> bool {
    outer
        .iter()
        .filter_map(Intervention::primary_variable)
        .any(|ov| inner.iter().any(|i| i.primary_variable() == Some(ov)))
}

/// Interventions, frozen set and outcome of a nested counterfactual query.
#[derive(Debug, Clone, Copy)]
pub struct NestedSpec<'a> {
    /// Outer-world interventions.
    pub outer: &'a [Intervention],
    /// Inner-world interventions.
    pub inner: &'a [Intervention],
    /// Nodes held at their outer-world value in the inner world.
    pub frozen: &'a [VariableId],
    /// Outcome variable.
    pub outcome: VariableId,
}

/// Nested counterfactual (twin-network) under shared abduced noise.
///
/// The query `Y_{inner, F_{outer}}`: the outcome under the `inner` interventions
/// while every variable in `frozen` keeps the value it takes in the `outer` world.
/// For the mediation contrast `Y_{x, M_{x'}}` take `outer = do(X = x')`,
/// `inner = do(X = x)` and `frozen = {M}`.
///
/// 1. Abduct exogenous noise once from factual data.
/// 2. Evaluate the **outer** world.
/// 3. Freeze each node of `frozen` at its outer counterfactual value. Soft/stochastic
///    outer interventions are realized first; freezes store those realized values.
/// 4. Evaluate the **inner** world with those freezes plus `inner`. Every node that is
///    not frozen — including nodes between an inner target and the outcome, and side
///    paths such as a second mediator — is re-evaluated under the inner intervention.
///
/// Freezing more than the mediators changes the estimand (it holds every frozen path
/// at the outer world), which is why the set is an argument rather than inferred.
/// Supports Set, Shift, Soft, and Stochastic interventions.
///
/// # Errors
///
/// Unsupported intervention kinds, a frozen node that is an `inner` target or the
/// outcome, unknown variables, abduction/predict failures.
pub fn nested_counterfactual(
    engine: &CounterfactualEngine,
    data: &TabularData,
    spec: &NestedSpec<'_>,
    ws: &mut MechanismWorkspace,
    ctx: &ExecutionContext,
) -> Result<f64, CounterfactualError> {
    // Intervention-kind validation lives in the `_with_exo` body.
    let exo = engine.abduct(data, AbductionMissingPolicy::Error, ctx)?;
    nested_counterfactual_with_exo(engine, &exo, spec, ws, ctx)
}

/// [`nested_counterfactual`] against a caller-supplied [`ExogenousPosterior`]
/// (abduct once, evaluate many nested queries).
///
/// # Errors
///
/// Unsupported intervention kinds, an invalid frozen set, or predict failures.
#[allow(
    clippy::needless_range_loop,
    reason = "index-based loop over parallel arrays; the subscript also names the matrix coordinate"
)]
pub fn nested_counterfactual_with_exo(
    engine: &CounterfactualEngine,
    exo: &ExogenousPosterior,
    spec: &NestedSpec<'_>,
    ws: &mut MechanismWorkspace,
    ctx: &ExecutionContext,
) -> Result<f64, CounterfactualError> {
    let NestedSpec { outer, inner, frozen, outcome } = *spec;
    for iv in outer.iter().chain(inner.iter()) {
        match iv {
            Intervention::Set { .. }
            | Intervention::Shift { .. }
            | Intervention::Soft { .. }
            | Intervention::Stochastic { .. } => {}
            Intervention::Sequence(_) => {
                return Err(CounterfactualError::model_msg(
                    "nested_counterfactual does not accept Sequence inside outer/inner; \
                     pass flat intervention slices",
                ));
            }
            _ => {
                return Err(CounterfactualError::model_msg(format!(
                    "nested_counterfactual unsupported intervention kind: {iv:?}"
                )));
            }
        }
    }
    let n_nodes = exo.n_nodes;
    let n_units = exo.n_units;
    let outcome_dense = engine.model.dense_of(outcome).ok_or_else(|| {
        CounterfactualError::model_msg(format!("unknown outcome variable {outcome}"))
    })?;

    let mut inner_targets = vec![false; n_nodes];
    for iv in inner {
        if let Some(v) = iv.primary_variable() {
            if let Some(d) = engine.model.dense_of(v) {
                inner_targets[d.as_usize()] = true;
            }
        }
    }
    let mut is_frozen = vec![false; n_nodes];
    for &var in frozen {
        let node = engine
            .model
            .dense_of(var)
            .ok_or_else(|| CounterfactualError::model_msg(format!("unknown frozen node {var}")))?
            .as_usize();
        if node == outcome_dense.as_usize() || inner_targets[node] {
            return Err(CounterfactualError::model_msg(format!(
                "frozen node {var} is the outcome or an inner intervention target; a node cannot \
                 be both held at its outer value and re-evaluated"
            )));
        }
        is_frozen[node] = true;
    }

    let outer_world =
        CounterfactualWorld { unit_rows: None, interventions: Arc::from(outer.to_vec()) };
    // Full retention: the freeze step below reads every frozen node's column.
    let outer_res = engine.predict(exo, &[outer_world], &[], true, ws, ctx)?;

    let mut freeze_ivs: Vec<Intervention> = Vec::new();
    let mut need_unitwise = false;
    for node in 0..n_nodes {
        if !is_frozen[node] {
            continue;
        }
        let start = node * n_units;
        let col = &outer_res.values[start..start + n_units];
        // Only a truly constant finite column can be represented by one hard
        // intervention without changing any unit's frozen value.
        match column_is_finite_constant(col) {
            Some(first) => {
                let var = engine.model.output_layout.variables[node];
                freeze_ivs.push(Intervention::set(var, antecedent_core::Value::f64(first)));
            }
            None if col.is_empty() => {}
            None => {
                need_unitwise = true;
                break;
            }
        }
    }

    if need_unitwise {
        // Column-frozen single pass. Per unit, the historical loop hard-set
        // every frozen node to its outer value and ran a full-table `predict`
        // to read one row — O(n_units² · nodes). Mechanism evaluation is
        // row-wise for the non-temporal families, so freezing every unit at
        // once (frozen nodes keep their outer *columns*; every other node is
        // re-evaluated) produces the identical per-row values in one
        // O(n_units · nodes) pass. Stochastic inner interventions also match
        // bit-for-bit: each per-unit `predict` call restarted the same `0xCF_01`
        // stream and sampled full columns in topological order, exactly as the
        // single pass does.
        let overlay = InterventionOverlay::from_interventions(&engine.model, inner)?;
        let row_independent = nested_rows_independent(engine, &overlay, &is_frozen, n_nodes)?;

        if row_independent {
            let frozen =
                FrozenOuter { outer_res: &outer_res, overlay: &overlay, is_frozen: &is_frozen };
            return nested_column_frozen_mean(engine, exo, &frozen, outcome_dense, ws, ctx);
        }

        // Row-coupled (temporal) mechanisms cannot be evaluated per unit by
        // freezing: the historical fallback hard-set every frozen node to
        // unit u's outer value across *all* rows and ran a full-column
        // evaluation — a state-space mechanism then rolls its recursion over
        // that constant-filled column with every unit's abduced noise, so
        // "unit u's counterfactual" mixed other units' noise and a fabricated
        // constant history. Refuse rather than return a contaminated number;
        // a true single-series slice evaluation is the follow-up.
        return Err(CounterfactualError::model_msg(
            "nested counterfactual over row-coupled (temporal) mechanisms with \
             unit-varying outer values is not supported: per-unit freezing \
             contaminates state-space evaluation across rows. Refusing rather \
             than returning a silently mixed result.",
        ));
    }

    let mut combined = freeze_ivs;
    combined.extend_from_slice(inner);
    let world = CounterfactualWorld { unit_rows: None, interventions: Arc::from(combined) };
    let res = engine.predict(exo, &[world], &[outcome], true, ws, ctx)?;
    complete_outcome_mean(&res, 0, outcome_dense)
}

/// Whether every mechanism the nested pass would evaluate is row-independent.
/// Row-coupled (temporal) families make `out[u]` depend on other rows, which
/// the column-frozen pass cannot reproduce; those refuse. Overlay-driven nodes
/// (hard set / stochastic / soft) never consult the fitted mechanism, so they are
/// row-independent regardless of family.
#[allow(
    clippy::needless_range_loop,
    reason = "index-based loop over parallel arrays; the subscript also names the matrix coordinate"
)]
fn nested_rows_independent(
    engine: &CounterfactualEngine,
    overlay: &InterventionOverlay,
    is_frozen: &[bool],
    n_nodes: usize,
) -> Result<bool, CounterfactualError> {
    let mut row_independent = true;
    for node in 0..n_nodes {
        if is_frozen[node]
            || overlay.hard_set[node].is_some()
            || overlay.stochastic[node].is_some()
            || overlay.soft[node].is_some()
        {
            continue;
        }
        let slot = engine.model.mechanisms.get(DenseNodeId::from_raw(
            u32::try_from(node)
                .map_err(|_| CounterfactualError::model_msg("node index exceeds u32 capacity"))?,
        ));
        row_independent &= matches!(
            slot,
            MechanismSlot::LinearGaussian { .. }
                | MechanismSlot::HierarchicalLinear { .. }
                | MechanismSlot::Bvar { .. }
                | MechanismSlot::Constant { .. }
                | MechanismSlot::Discrete { .. }
                | MechanismSlot::GaussianProcess { .. }
                // A basis expansion reads only its own row's parents and
                // disturbance, exactly like the linear family it extends.
                | MechanismSlot::LinearBasis { .. }
                | MechanismSlot::DiscreteBasis { .. }
                // Tree ensembles read only their own row's parents and disturbance.
                | MechanismSlot::TreeAdditive { .. }
                | MechanismSlot::TreeCategorical { .. }
                // A state-space mechanism's abduced noise is its residual path, so
                // evaluation is `mean(parents) + noise[row]`: row-wise like the linear family.
                | MechanismSlot::LinearGaussianStateSpace { .. }
                | MechanismSlot::ConditionalLinearGaussianStateSpace { .. }
        );
    }
    Ok(row_independent)
}

/// `Some(value)` when every entry is finite and equal to the first.
/// Signed zeroes compare equal; empty and non-finite columns are not constants.
#[allow(
    clippy::float_cmp,
    reason = "exact equality preserves discrete values and normalization boundaries"
)]
pub(super) fn column_is_finite_constant(col: &[f64]) -> Option<f64> {
    let &first = col.first()?;
    (first.is_finite() && col.iter().all(|&v| v == first)).then_some(first)
}

/// The outer world, the inner overlay and the frozen-node mask a nested evaluation reads.
#[derive(Clone, Copy)]
struct FrozenOuter<'a> {
    outer_res: &'a CounterfactualResult,
    overlay: &'a InterventionOverlay,
    is_frozen: &'a [bool],
}

/// One-pass evaluation of the unit-wise nested counterfactual: frozen nodes
/// keep their outer counterfactual *columns*; every other node is re-evaluated
/// row-wise with the abduced noise. Bit-identical to the per-unit
/// freeze-and-predict loop for row-independent mechanisms (see the caller's
/// derivation and the differential test).
fn nested_column_frozen_mean(
    engine: &CounterfactualEngine,
    exo: &ExogenousPosterior,
    frozen: &FrozenOuter<'_>,
    outcome_dense: DenseNodeId,
    ws: &mut MechanismWorkspace,
    ctx: &ExecutionContext,
) -> Result<f64, CounterfactualError> {
    let FrozenOuter { outer_res, overlay, is_frozen } = *frozen;
    let n_units = exo.n_units;
    let n_nodes = exo.n_nodes;
    let mut rng = ctx.rng.stream_for(StreamDomain::Counterfactual, PREDICT_STREAM);
    let mut values_buf = outer_res.values[..n_nodes * n_units].to_vec();
    let mut values = ValueBatchMut::new(n_units, n_nodes, &mut values_buf)?;
    let mut parent_buf = Vec::new();
    evaluate_world_into(
        WorldEvaluation { model: &engine.model, exo, overlay, frozen: Some(is_frozen), ctx },
        &mut rng,
        &mut values,
        &mut parent_buf,
        ws,
    )?;
    let start = outcome_dense.as_usize() * n_units;
    let mut sum = 0.0;
    let mut count = 0usize;
    for &v in &values_buf[start..start + n_units] {
        if v.is_finite() {
            sum += v;
            count += 1;
        }
    }
    if count != n_units {
        return Err(CounterfactualError::model_msg(format!(
            "nested CF outcome is non-finite for {} of {} units; refusing to average the rest",
            n_units - count,
            n_units
        )));
    }
    Ok(sum / count as f64)
}

/// Alias for [`nested_counterfactual`] (historical name).
///
/// # Errors
///
/// As [`nested_counterfactual`].
pub fn nested_hard_counterfactual(
    engine: &CounterfactualEngine,
    data: &TabularData,
    spec: &NestedSpec<'_>,
    ws: &mut MechanismWorkspace,
    ctx: &ExecutionContext,
) -> Result<f64, CounterfactualError> {
    nested_counterfactual(engine, data, spec, ws, ctx)
}
