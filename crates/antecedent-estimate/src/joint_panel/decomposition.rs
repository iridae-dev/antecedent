//! Exact reference-world decomposition through the compiled panel response owner.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{PanelData, PanelDraw, PanelError, PanelProgram, check_cancelled, invalid, likelihood};
use antecedent_core::ExecutionContext;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Four disjoint blocks replaced when constructing a reference world.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChangeComponent {
    /// Schedules and initial exposure histories.
    Exposure,
    /// Every saturation parameter (half-saturations and Hill shapes), excluded from the
    /// mechanism block.
    Saturation,
    /// Scenario covariates, seasons and population weights.
    Population,
    /// All remaining response parameters, lag weights and group effects.
    Mechanism,
}
impl ChangeComponent {
    fn index(self) -> usize {
        match self {
            Self::Exposure => 0,
            Self::Saturation => 1,
            Self::Population => 2,
            Self::Mechanism => 3,
        }
    }
}

/// Explicit program and population law at one endpoint.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecompositionEndpoint {
    /// Complete schedules and group-specific histories/context.
    pub program: PanelProgram,
    /// Nonnegative weights in source group order; must sum to one.
    pub population_weights: Vec<f64>,
}

/// One *paired* mechanism realization across the two endpoints.
/// A stationary realization contains one draw per endpoint; a temporal one
/// contains one draw per response period. Independently fitted marginal draws
/// cannot be paired by index and called a joint law.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairedPanelPath {
    /// Before path, either length one or the complete response horizon.
    pub before: Vec<PanelDraw>,
    /// After path on the same uncertainty realization.
    pub after: Vec<PanelDraw>,
}

/// Explicit mapping from response periods to two parts of one fitted mechanism path.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MechanismPeriodMap {
    /// Before parameter index at each requested response period.
    pub before: Vec<usize>,
    /// After parameter index at each requested response period.
    pub after: Vec<usize>,
}
impl MechanismPeriodMap {
    /// Build two paths from one jointly fitted temporal realization.
    ///
    /// # Errors
    /// Missing periods, unequal/empty response axes or an out-of-range parameter index.
    pub fn from_temporal(
        &self,
        draw: &super::time_varying::TemporalPanelDraw,
    ) -> Result<PairedPanelPath, PanelError> {
        if self.before.is_empty() || self.before.len() != self.after.len() {
            return Err(invalid("mechanism period maps must have equal positive lengths"));
        }
        let path = |indices: &[usize]| {
            indices
                .iter()
                .map(|index| {
                    let parameters = draw.parameters.get(*index).ok_or_else(|| {
                        invalid("mechanism period is outside the fitted temporal path")
                    })?;
                    Ok(PanelDraw {
                        parameters: parameters.clone(),
                        group_intercepts: draw.baseline.group_intercepts.clone(),
                    })
                })
                .collect::<Result<Vec<_>, PanelError>>()
        };
        Ok(PairedPanelPath { before: path(&self.before)?, after: path(&self.after)? })
    }
    /// Bind a stationary realization; only period zero exists in its parameter path.
    ///
    /// # Errors
    /// Unequal/empty axes or a nonzero stationary mechanism index.
    pub fn from_invariant(&self, draw: &PanelDraw) -> Result<PairedPanelPath, PanelError> {
        if self.before.is_empty()
            || self.before.len() != self.after.len()
            || self.before.iter().chain(&self.after).any(|i| *i != 0)
        {
            return Err(invalid(
                "stationary mechanism maps require equal positive zero-index axes",
            ));
        }
        Ok(PairedPanelPath { before: vec![draw.clone()], after: vec![draw.clone()] })
    }
}

/// Explicit response functional and the requested sequential replacement order.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PanelDecompositionRequest {
    /// Before schedules, histories and population law.
    pub before: DecompositionEndpoint,
    /// After schedules, histories and population law.
    pub after: DecompositionEndpoint,
    /// Finite time coefficients, one per common response period.
    pub time_weights: Vec<f64>,
    /// A permutation of the four components.
    pub order: [ChangeComponent; 4],
}

/// Decomposition of one complete joint mechanism realization.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReferenceDecomposition {
    /// All 16 hybrid responses; bit order is exposure, saturation, population, mechanism.
    pub hybrids: [f64; 16],
    /// After minus before response under the requested time functional.
    pub total: f64,
    /// Sequential contributions, indexed by component rather than order position.
    pub sequential: [f64; 4],
    /// Exact average over all 24 sequential orderings.
    pub shapley: [f64; 4],
    /// Explicit component permutation associated with each `by_order` row.
    pub orderings: Vec<[ChangeComponent; 4]>,
    /// Contributions for all orderings in lexicographic component-index order.
    pub by_order: Vec<[f64; 4]>,
    /// Min/max contribution across orders; this is *not* a statistical interval.
    pub order_range: [[f64; 2]; 4],
}

/// Retained point and joint uncertainty decompositions. Numerical evaluation
/// alone grants neither causal identification nor an interval coverage license.
#[derive(Clone, Debug)]
pub struct PanelDecomposition {
    /// Point mechanism decomposition.
    pub point: ReferenceDecomposition,
    /// Original uncertainty slots; a failed parent slot stays failed.
    pub draws: Vec<Option<ReferenceDecomposition>>,
    /// Original nonnegative finite-law weights, when the parent law supplies them.
    pub weights: Option<Vec<f64>>,
    retention: Arc<antecedent_core::execution::AdmissionLease>,
}
/// Joint moments of the four contributions, calculated from aligned complete law slots.
/// Order sensitivity is intentionally absent: it is a separate deterministic quantity.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DecompositionJointUncertainty {
    /// Whether covariance describes an explicitly weighted finite law (rather than a sample estimate).
    pub weighted_finite_law: bool,
    /// Requested slots, including failures; no failed realization is replaced or renumbered.
    pub requested: usize,
    /// Complete slots contributing to these descriptive moments.
    pub complete: usize,
    /// Mean Shapley contributions.
    pub shapley_mean: [f64; 4],
    /// Sample covariance in component order, preserving dependence and cancellation.
    pub shapley_covariance: [[f64; 4]; 4],
    /// Mean contributions for the caller's sequential ordering.
    pub sequential_mean: [f64; 4],
    /// Joint sample covariance for the requested sequential ordering.
    pub sequential_covariance: [[f64; 4]; 4],
    /// Sample variance of the total change; equals the sum of the joint covariance.
    pub total_variance: f64,
}
/// Running joint moments of the contributions, in the exact arithmetic order of a retained law.
#[derive(Default)]
struct MomentAccumulator {
    count: u32,
    means: [[f64; 4]; 2],
    products: [[[f64; 4]; 4]; 2],
    total_mean: f64,
    total_product: f64,
}
impl MomentAccumulator {
    fn push(&mut self, draw: &ReferenceDecomposition) -> Option<()> {
        self.count = self.count.checked_add(1)?;
        let n = f64::from(self.count);
        let total_delta = draw.total - self.total_mean;
        self.total_mean += total_delta / n;
        self.total_product += total_delta * (draw.total - self.total_mean);
        for (kind, values) in [draw.shapley, draw.sequential].iter().enumerate() {
            let delta: [f64; 4] = std::array::from_fn(|j| values[j] - self.means[kind][j]);
            for j in 0..4 {
                self.means[kind][j] += delta[j] / n;
            }
            for j in 0..4 {
                for k in 0..4 {
                    self.products[kind][j][k] += delta[j] * (values[k] - self.means[kind][k]);
                }
            }
        }
        Some(())
    }
    fn finish(mut self, requested: usize) -> Option<DecompositionJointUncertainty> {
        if self.count < 2 {
            return None;
        }
        let denominator = f64::from(self.count - 1);
        for product in &mut self.products {
            for row in product {
                for value in row {
                    *value /= denominator;
                }
            }
        }
        Some(DecompositionJointUncertainty {
            weighted_finite_law: false,
            requested,
            complete: usize::try_from(self.count).ok()?,
            shapley_mean: self.means[0],
            sequential_mean: self.means[1],
            shapley_covariance: self.products[0],
            sequential_covariance: self.products[1],
            total_variance: self.total_product / denominator,
        })
    }
}

impl PanelDecomposition {
    /// Descriptive joint uncertainty from the retained law. A covariance is not a
    /// coverage claim. Failed slots remain visible; fewer than two complete slots
    /// cannot define a sample covariance and return `None`.
    #[must_use]
    pub fn joint_uncertainty(&self) -> Option<DecompositionJointUncertainty> {
        if let Some(weights) = &self.weights {
            return weighted_joint_uncertainty(&self.draws, weights);
        }
        let mut accumulator = MomentAccumulator::default();
        for draw in self.draws.iter().flatten() {
            accumulator.push(draw)?;
        }
        accumulator.finish(self.draws.len())
    }
    /// Retained admission, excluding the caller's separately owned input law.
    #[must_use]
    pub fn admitted_bytes(&self) -> u64 {
        self.retention.remainder().bytes()
    }
}

fn weighted_joint_uncertainty(
    draws: &[Option<ReferenceDecomposition>],
    weights: &[f64],
) -> Option<DecompositionJointUncertainty> {
    if draws.len() != weights.len() {
        return None;
    }
    let total = weights.iter().sum::<f64>();
    if !total.is_finite() || total <= 0.0 {
        return None;
    }
    let mut means = [[0.0; 4]; 2];
    let mut total_mean = 0.0;
    let mut complete = 0;
    for (draw, weight) in draws.iter().zip(weights) {
        if !weight.is_finite() || *weight < 0.0 {
            return None;
        }
        if *weight == 0.0 {
            continue;
        }
        let draw = draw.as_ref()?;
        complete += 1;
        let normalized = weight / total;
        for (kind, values) in [draw.shapley, draw.sequential].iter().enumerate() {
            for j in 0..4 {
                means[kind][j] += normalized * values[j];
            }
        }
        total_mean += normalized * draw.total;
    }
    let mut covariance = [[[0.0; 4]; 4]; 2];
    let mut total_variance = 0.0;
    for (draw, weight) in draws.iter().zip(weights) {
        if *weight == 0.0 {
            continue;
        }
        let draw = draw.as_ref()?;
        let normalized = weight / total;
        total_variance += normalized * (draw.total - total_mean).powi(2);
        for (kind, values) in [draw.shapley, draw.sequential].iter().enumerate() {
            for j in 0..4 {
                for k in 0..4 {
                    covariance[kind][j][k] +=
                        normalized * (values[j] - means[kind][j]) * (values[k] - means[kind][k]);
                }
            }
        }
    }
    Some(DecompositionJointUncertainty {
        weighted_finite_law: true,
        requested: draws.len(),
        complete,
        shapley_mean: means[0],
        sequential_mean: means[1],
        shapley_covariance: covariance[0],
        sequential_covariance: covariance[1],
        total_variance,
    })
}

/// Decompose one retained stationary law without inventing a paired independent fit.
///
/// # Errors
/// Invalid period maps, inputs, numerical failure, resources or cancellation.
pub fn decompose_invariant_law(
    data: &PanelData,
    request: &PanelDecompositionRequest,
    periods: &MechanismPeriodMap,
    point: &PanelDraw,
    draws: &[Option<PanelDraw>],
    ctx: &ExecutionContext,
) -> Result<PanelDecomposition, PanelError> {
    map_law(data, request, periods, point, draws, |draw| periods.from_invariant(draw), ctx)
}

/// Decompose one retained temporal law, preserving model/path/selection slots.
///
/// # Errors
/// Invalid period maps, inputs, numerical failure, resources or cancellation.
pub fn decompose_temporal_law(
    data: &PanelData,
    request: &PanelDecompositionRequest,
    periods: &MechanismPeriodMap,
    point: &super::time_varying::TemporalPanelDraw,
    draws: &[Option<super::time_varying::TemporalPanelDraw>],
    ctx: &ExecutionContext,
) -> Result<PanelDecomposition, PanelError> {
    map_law(data, request, periods, point, draws, |draw| periods.from_temporal(draw), ctx)
}

fn map_law<T>(
    data: &PanelData,
    request: &PanelDecompositionRequest,
    periods: &MechanismPeriodMap,
    point: &T,
    draws: &[Option<T>],
    map: impl Fn(&T) -> Result<PairedPanelPath, PanelError>,
    ctx: &ExecutionContext,
) -> Result<PanelDecomposition, PanelError> {
    check_cancelled(ctx)?;
    if periods.before.len() != request.time_weights.len()
        || periods.after.len() != request.time_weights.len()
    {
        return Err(invalid("mechanism maps do not match the response horizon"));
    }
    let bytes = likelihood::parameter_bytes(data)?
        .checked_add(
            data.groups.len().checked_mul(8).ok_or_else(|| invalid("path quote overflow"))?,
        )
        .and_then(|n| n.checked_add(std::mem::size_of::<PanelDraw>()))
        .and_then(|n| n.checked_mul(request.time_weights.len()))
        .and_then(|n| n.checked_mul(2))
        .and_then(|n| n.checked_mul(draws.len().checked_add(1)?))
        .and_then(|n| {
            n.checked_add(std::mem::size_of::<Option<PairedPanelPath>>() * (draws.len() + 1))
        })
        .ok_or_else(|| invalid("paired mechanism path quote overflow"))?;
    super::check_memory(ctx, bytes as u64)?;
    let _temporary = ctx
        .acquire_admission(0, bytes as u64)
        .map_err(|error| PanelError::Resource(error.to_string()))?;
    let point = map(point)?;
    let mut paired = Vec::with_capacity(draws.len());
    for draw in draws {
        check_cancelled(ctx)?;
        paired.push(draw.as_ref().map(&map).transpose()?);
    }
    decompose_panel_law(data, request, &point, &paired, ctx)
}

/// Evaluate all reference worlds using one compiled interaction plan, preserving
/// paired parameter/path uncertainty across worlds and components.
///
/// # Errors
/// Refuses mismatched horizons/groups, invalid paths, weights or orders, numerical
/// overflow, cancellation and exhausted resources. Scientific identification must
/// be checked by the compiler before these numbers are consumed causally.
pub fn decompose_panel_law(
    data: &PanelData,
    request: &PanelDecompositionRequest,
    point: &PairedPanelPath,
    draws: &[Option<PairedPanelPath>],
    ctx: &ExecutionContext,
) -> Result<PanelDecomposition, PanelError> {
    check_cancelled(ctx)?;
    validate_request(data, request)?;
    let compiled = likelihood::CompiledPanel::new(data)?;
    let result_bytes = (draws.len() as u64)
        .checked_add(1)
        .and_then(|n| {
            n.checked_mul(
                (std::mem::size_of::<Option<ReferenceDecomposition>>()
                    + std::mem::size_of::<f64>()
                    + 24 * (std::mem::size_of::<[f64; 4]>()
                        + std::mem::size_of::<[ChangeComponent; 4]>())) as u64,
            )
        })
        .ok_or_else(|| invalid("decomposition retention size overflow"))?;
    super::check_memory(ctx, result_bytes)?;
    let retention = Arc::new(
        ctx.acquire_admission(0, result_bytes)
            .map_err(|error| PanelError::Resource(error.to_string()))?,
    );
    let scratch_bytes = super::response_workspace_bytes(&compiled)?
        .checked_add(likelihood::parameter_bytes(data)? as u64)
        .and_then(|n| n.checked_add(scenario_workspace_bytes(request).ok()?))
        .ok_or_else(|| invalid("decomposition workspace overflow"))?;
    super::check_memory(
        ctx,
        result_bytes
            .checked_add(scratch_bytes)
            .ok_or_else(|| invalid("decomposition peak overflow"))?,
    )?;
    let _scratch = ctx
        .acquire_admission(0, scratch_bytes)
        .map_err(|error| PanelError::Resource(error.to_string()))?;
    let point = evaluate(&compiled, request, point, ctx)?;
    let mut results = Vec::with_capacity(draws.len());
    for draw in draws {
        check_cancelled(ctx)?;
        results
            .push(draw.as_ref().map(|path| evaluate(&compiled, request, path, ctx)).transpose()?);
    }
    Ok(PanelDecomposition { point, draws: results, weights: None, retention })
}

fn validate_request(
    data: &PanelData,
    request: &PanelDecompositionRequest,
) -> Result<(), PanelError> {
    data.validate()?;
    let horizon = request.time_weights.len();
    if horizon == 0
        || request.time_weights.iter().any(|v| !v.is_finite())
        || request.time_weights.iter().all(|v| *v == 0.0)
        || request.before.program.horizon() != horizon
        || request.after.program.horizon() != horizon
    {
        return Err(invalid("decomposition requires one explicit nonzero common time functional"));
    }
    validate_order(request.order)?;
    for endpoint in [&request.before, &request.after] {
        super::validate_panel_programs(data, std::slice::from_ref(&endpoint.program))?;
        if endpoint.population_weights.len() != data.groups.len()
            || endpoint.population_weights.iter().any(|w| !w.is_finite() || *w < 0.0)
            || (endpoint.population_weights.iter().sum::<f64>() - 1.0).abs() > 1e-12
        {
            return Err(invalid("population weights must cover source groups and sum to one"));
        }
    }
    Ok(())
}

/// Point decomposition and joint moments of a law, without retaining any per-draw table.
#[derive(Clone, Debug, PartialEq)]
pub struct StreamedDecomposition {
    /// Point mechanism decomposition.
    pub point: ReferenceDecomposition,
    /// The moments [`PanelDecomposition::joint_uncertainty`] would report for the retained law.
    pub joint: Option<DecompositionJointUncertainty>,
    /// Law slots, including failed ones.
    pub requested: usize,
    /// Failed parent slots, left failed.
    pub failed: usize,
}

/// Streamed [`decompose_invariant_law`]: each draw's paired path is formed, evaluated, folded
/// into the joint moments and dropped, so neither the paired paths nor the per-draw
/// decompositions of the whole law are ever held. Moments equal the retained route's.
///
/// # Errors
/// Invalid period maps, inputs, numerical failure, resources or cancellation.
pub fn decompose_invariant_law_streamed(
    data: &PanelData,
    request: &PanelDecompositionRequest,
    periods: &MechanismPeriodMap,
    point: &PanelDraw,
    draws: &[Option<PanelDraw>],
    ctx: &ExecutionContext,
) -> Result<StreamedDecomposition, PanelError> {
    decompose_law_streamed(
        data,
        request,
        periods,
        point,
        draws,
        |draw| periods.from_invariant(draw),
        ctx,
    )
}

/// Streamed [`decompose_temporal_law`], keeping selection and path coupling per draw.
///
/// # Errors
/// Invalid period maps, inputs, numerical failure, resources or cancellation.
pub fn decompose_temporal_law_streamed(
    data: &PanelData,
    request: &PanelDecompositionRequest,
    periods: &MechanismPeriodMap,
    point: &super::time_varying::TemporalPanelDraw,
    draws: &[Option<super::time_varying::TemporalPanelDraw>],
    ctx: &ExecutionContext,
) -> Result<StreamedDecomposition, PanelError> {
    decompose_law_streamed(
        data,
        request,
        periods,
        point,
        draws,
        |draw| periods.from_temporal(draw),
        ctx,
    )
}

fn decompose_law_streamed<T>(
    data: &PanelData,
    request: &PanelDecompositionRequest,
    periods: &MechanismPeriodMap,
    point: &T,
    draws: &[Option<T>],
    map: impl Fn(&T) -> Result<PairedPanelPath, PanelError>,
    ctx: &ExecutionContext,
) -> Result<StreamedDecomposition, PanelError> {
    check_cancelled(ctx)?;
    if periods.before.len() != request.time_weights.len()
        || periods.after.len() != request.time_weights.len()
    {
        return Err(invalid("mechanism maps do not match the response horizon"));
    }
    validate_request(data, request)?;
    let compiled = likelihood::CompiledPanel::new(data)?;
    // One paired path and one decomposition are live at a time.
    let path_bytes = likelihood::parameter_bytes(data)?
        .checked_add(data.groups.len().checked_mul(8).ok_or_else(|| invalid("quote overflow"))?)
        .and_then(|n| n.checked_add(std::mem::size_of::<PanelDraw>()))
        .and_then(|n| n.checked_mul(request.time_weights.len()))
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| invalid("paired mechanism path quote overflow"))?;
    let scratch_bytes = super::response_workspace_bytes(&compiled)?
        .checked_add(likelihood::parameter_bytes(data)? as u64)
        .and_then(|n| n.checked_add(scenario_workspace_bytes(request).ok()?))
        .and_then(|n| n.checked_add(path_bytes as u64))
        .and_then(|n| n.checked_add(std::mem::size_of::<ReferenceDecomposition>() as u64 * 2))
        .ok_or_else(|| invalid("decomposition workspace overflow"))?;
    super::check_memory(ctx, scratch_bytes)?;
    let _scratch = ctx
        .acquire_admission(0, scratch_bytes)
        .map_err(|error| PanelError::Resource(error.to_string()))?;
    let point = evaluate(&compiled, request, &map(point)?, ctx)?;
    let mut accumulator = MomentAccumulator::default();
    let mut failed = 0;
    let mut overflow = false;
    for draw in draws {
        check_cancelled(ctx)?;
        match draw {
            None => failed += 1,
            Some(draw) => {
                let decomposed = evaluate(&compiled, request, &map(draw)?, ctx)?;
                overflow |= accumulator.push(&decomposed).is_none();
            }
        }
    }
    let joint = if overflow { None } else { accumulator.finish(draws.len()) };
    Ok(StreamedDecomposition { point, joint, requested: draws.len(), failed })
}

fn scenario_workspace_bytes(request: &PanelDecompositionRequest) -> Result<u64, PanelError> {
    let mut peak = 0_usize;
    for endpoint in [&request.before, &request.after] {
        for scenario in &endpoint.program.scenarios {
            let mut bytes = std::mem::size_of_val(scenario)
                .checked_add(scenario.group.len())
                .and_then(|n| n.checked_add(scenario.confounder.len().checked_mul(8)?))
                .and_then(|n| {
                    n.checked_add(scenario.season.len().checked_mul(std::mem::size_of::<usize>())?)
                })
                .ok_or_else(|| invalid("scenario workspace quote overflow"))?;
            for history in &scenario.history {
                bytes = bytes
                    .checked_add(2 * std::mem::size_of_val(history))
                    .and_then(|n| n.checked_add(history.id.len().checked_mul(2)?))
                    .ok_or_else(|| invalid("history workspace quote overflow"))?;
            }
            peak = peak.max(bytes);
        }
    }
    u64::try_from(peak).map_err(|_| invalid("scenario workspace quote overflow"))
}

fn validate_order(order: [ChangeComponent; 4]) -> Result<(), PanelError> {
    let mut seen = [false; 4];
    for component in order {
        if std::mem::replace(&mut seen[component.index()], true) {
            return Err(invalid("replacement order must be a permutation of four components"));
        }
    }
    Ok(())
}
fn validate_path(
    data: &likelihood::CompiledPanel<'_>,
    path: &[PanelDraw],
    horizon: usize,
) -> Result<(), PanelError> {
    if path.len() != 1 && path.len() != horizon {
        return Err(invalid(
            "mechanism path must be stationary or cover the full response horizon",
        ));
    }
    for draw in path {
        likelihood::validate_parameters(data, &draw.parameters)?;
        if draw.group_intercepts.len() != data.groups.len()
            || draw.group_intercepts.iter().any(|v| !v.is_finite())
        {
            return Err(invalid("decomposition group-effect axis mismatch"));
        }
    }
    Ok(())
}
fn evaluate(
    data: &likelihood::CompiledPanel<'_>,
    request: &PanelDecompositionRequest,
    path: &PairedPanelPath,
    ctx: &ExecutionContext,
) -> Result<ReferenceDecomposition, PanelError> {
    let horizon = request.time_weights.len();
    validate_path(data, &path.before, horizon)?;
    validate_path(data, &path.after, horizon)?;
    // Identical component bindings describe the same reference world. Reuse its
    // actual compiled response, rather than reevaluating equivalent worlds on every
    // realization (particularly important for full posterior/selection laws).
    let exposure_equal = request.before.program.channels == request.after.program.channels
        && request
            .before
            .program
            .scenarios
            .iter()
            .zip(&request.after.program.scenarios)
            .all(|(a, b)| a.group == b.group && a.history == b.history);
    let population_equal = request.before.population_weights == request.after.population_weights
        && request.before.program.scenarios.iter().zip(&request.after.program.scenarios).all(
            |(a, b)| a.group == b.group && a.confounder == b.confounder && a.season == b.season,
        );
    let saturation_equal = (0..horizon).all(|time| {
        path.before[if path.before.len() == 1 { 0 } else { time }]
            .parameters
            .saturation_equal(&path.after[if path.after.len() == 1 { 0 } else { time }].parameters)
    });
    let inactive = usize::from(exposure_equal)
        | (usize::from(saturation_equal) << 1)
        | (usize::from(population_equal) << 2)
        | (usize::from(path.before == path.after) << 3);
    let mut hybrids = [0.0; 16];
    for mask in 0..16 {
        let canonical = mask & !inactive;
        if canonical != mask {
            hybrids[mask] = hybrids[canonical];
            continue;
        }
        let response = &mut hybrids[mask];
        check_cancelled(ctx)?;
        let exposure = if mask & 1 == 0 { &request.before } else { &request.after };
        let population = if mask & 4 == 0 { &request.before } else { &request.after };
        let mechanism = if mask & 8 == 0 { &path.before } else { &path.after };
        let saturation = if mask & 2 == 0 { &path.before } else { &path.after };
        // Scenarios are matched by stable group identity, never by input ordering.
        for (group, weight) in data.groups.iter().zip(&population.population_weights) {
            let context = population
                .program
                .scenarios
                .iter()
                .find(|s| s.group == group.id)
                .ok_or_else(|| invalid("population group absent"))?;
            let history = exposure
                .program
                .scenarios
                .iter()
                .find(|s| s.group == group.id)
                .ok_or_else(|| invalid("exposure group absent"))?;
            let group_index = data
                .groups
                .iter()
                .position(|g| g.id == group.id)
                .ok_or_else(|| invalid("unknown source group"))?;
            let mut scenario = context.clone();
            scenario.history.clone_from(&history.history);
            for (time, time_weight) in request.time_weights.iter().enumerate() {
                check_cancelled(ctx)?;
                let draw = &mechanism[if mechanism.len() == 1 { 0 } else { time }];
                let sat = &saturation[if saturation.len() == 1 { 0 } else { time }];
                let mut parameters = draw.parameters.clone();
                parameters.copy_saturation_from(&sat.parameters);
                *response += weight
                    * time_weight
                    * (likelihood::program_response(
                        data,
                        &parameters,
                        &exposure.program,
                        &scenario,
                        time,
                    ) + draw.group_intercepts[group_index]);
            }
        }
    }
    decompose_values(hybrids, request.order)
}

/// Decompose a compiled joint response law whose sixteen coordinates have already
/// passed the owning scientific compiler's component and world checks.
///
/// # Errors
/// Invalid responses/order, resources or cancellation.
pub fn decompose_hybrid_law(
    point: [f64; 16],
    draws: &[Option<[f64; 16]>],
    order: [ChangeComponent; 4],
    ctx: &ExecutionContext,
) -> Result<PanelDecomposition, PanelError> {
    decompose_weighted_hybrid_law(point, draws, None, order, ctx)
}

/// Preserve explicit weights from a compiled finite joint law while decomposing it.
/// Zero-weight failed slots may be retained; positive-weight failures refuse.
/// # Errors
/// Invalid weights, responses/order, resources or cancellation.
pub fn decompose_weighted_hybrid_law(
    point: [f64; 16],
    draws: &[Option<[f64; 16]>],
    weights: Option<&[f64]>,
    order: [ChangeComponent; 4],
    ctx: &ExecutionContext,
) -> Result<PanelDecomposition, PanelError> {
    if let Some(weights) = weights {
        let total = weights.iter().sum::<f64>();
        if weights.len() != draws.len()
            || !total.is_finite()
            || total <= 0.0
            || weights.iter().any(|weight| !weight.is_finite() || *weight < 0.0)
            || weights.iter().zip(draws).any(|(weight, draw)| *weight > 0.0 && draw.is_none())
        {
            return Err(invalid("invalid or incomplete weighted hybrid law"));
        }
    }
    check_cancelled(ctx)?;
    let bytes = (draws.len() as u64)
        .checked_add(1)
        .and_then(|n| {
            n.checked_mul(
                (std::mem::size_of::<Option<ReferenceDecomposition>>()
                    + std::mem::size_of::<f64>()
                    + 24 * (std::mem::size_of::<[f64; 4]>()
                        + std::mem::size_of::<[ChangeComponent; 4]>())) as u64,
            )
        })
        .ok_or_else(|| invalid("hybrid law retention overflow"))?;
    super::check_memory(ctx, bytes)?;
    let retention = Arc::new(
        ctx.acquire_admission(0, bytes).map_err(|error| PanelError::Resource(error.to_string()))?,
    );
    let point = decompose_values(point, order)?;
    let mut values = Vec::with_capacity(draws.len());
    for draw in draws {
        check_cancelled(ctx)?;
        values.push(draw.map(|row| decompose_values(row, order)).transpose()?);
    }
    Ok(PanelDecomposition {
        point,
        draws: values,
        weights: weights.map(<[f64]>::to_vec),
        retention,
    })
}

/// Exact linear propagation of a joint hybrid-response covariance.
/// Covariance meaning (design-based, sampling, posterior, or approximation) is inherited.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReferenceDecompositionCovariance {
    /// Joint covariance of the exact Shapley contributions.
    pub shapley: [[f64; 4]; 4],
    /// Joint covariance of contributions in the requested ordering.
    pub sequential: [[f64; 4]; 4],
    /// Joint covariances in the same lexicographic ordering as `by_order`.
    pub by_order: [[[f64; 4]; 4]; 24],
    /// Variance of after minus before, including their dependence.
    pub total: f64,
}
/// Propagate a supplied *joint* hybrid covariance, using exact linear decomposition maps.
/// The caller must retain the originating method's inferential assumptions.
/// # Errors
/// Nonfinite/asymmetric covariance, invalid ordering or arithmetic overflow.
pub fn propagate_hybrid_covariance(
    covariance: &[[f64; 16]; 16],
    order: [ChangeComponent; 4],
) -> Result<ReferenceDecompositionCovariance, PanelError> {
    if covariance.iter().flatten().any(|value| !value.is_finite())
        || (0..16).any(|j| {
            covariance[j][j] < 0.0
                || (0..16).any(|k| {
                    (covariance[j][k] - covariance[k][j]).abs()
                        > 1e-10 * (1.0 + covariance[j][k].abs() + covariance[k][j].abs())
                })
        })
    {
        return Err(invalid(
            "joint hybrid covariance must be finite, symmetric with nonnegative variances",
        ));
    }
    let matrix: [f64; 256] = std::array::from_fn(|index| covariance[index / 16][index % 16]);
    let mut factor = [0.0; 256];
    antecedent_kernels::cholesky::cholesky_psd_into(&matrix, 16, 1e-10, &mut factor)
        .map_err(|_| invalid("joint hybrid covariance is not positive semidefinite"))?;
    let basis = (0..16)
        .map(|j| {
            let mut values = [0.0; 16];
            values[j] = 1.0;
            decompose_values(values, order)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let transform = |coefficients: &[[f64; 4]; 16]| {
        std::array::from_fn(|j| {
            std::array::from_fn(|k| {
                (0..16)
                    .map(|a| {
                        (0..16)
                            .map(|b| coefficients[a][j] * covariance[a][b] * coefficients[b][k])
                            .sum::<f64>()
                    })
                    .sum::<f64>()
            })
        })
    };
    let shapley = transform(&std::array::from_fn(|j| basis[j].shapley));
    let sequential = transform(&std::array::from_fn(|j| basis[j].sequential));
    let by_order =
        std::array::from_fn(|index| transform(&std::array::from_fn(|j| basis[j].by_order[index])));
    let total = covariance[15][15] + covariance[0][0] - covariance[0][15] - covariance[15][0];
    if !total.is_finite()
        || shapley
            .iter()
            .flatten()
            .chain(sequential.iter().flatten())
            .chain(by_order.iter().flatten().flatten())
            .any(|value| !value.is_finite())
    {
        return Err(invalid("hybrid covariance propagation overflow"));
    }
    Ok(ReferenceDecompositionCovariance { shapley, sequential, by_order, total })
}

/// Exact four-factor Shapley/sequential decomposition of a complete hybrid table.
///
/// # Errors
/// Nonfinite hybrid values, invalid order or arithmetic overflow.
pub fn decompose_values(
    hybrids: [f64; 16],
    order: [ChangeComponent; 4],
) -> Result<ReferenceDecomposition, PanelError> {
    validate_order(order)?;
    if hybrids.iter().any(|v| !v.is_finite()) {
        return Err(invalid("nonfinite hybrid response"));
    }
    let contribution = |order: [usize; 4]| {
        let mut mask = 0;
        let mut values = [0.0; 4];
        for component in order {
            let next = mask | (1 << component);
            values[component] = hybrids[next] - hybrids[mask];
            mask = next;
        }
        values
    };
    let sequential = contribution(order.map(ChangeComponent::index));
    let mut by_order = Vec::with_capacity(24);
    let mut orderings = Vec::with_capacity(24);
    let components = [
        ChangeComponent::Exposure,
        ChangeComponent::Saturation,
        ChangeComponent::Population,
        ChangeComponent::Mechanism,
    ];
    for a in 0..4 {
        for b in 0..4 {
            for c in 0..4 {
                for d in 0..4 {
                    if a != b && a != c && a != d && b != c && b != d && c != d {
                        by_order.push(contribution([a, b, c, d]));
                        orderings.push([
                            components[a],
                            components[b],
                            components[c],
                            components[d],
                        ]);
                    }
                }
            }
        }
    }
    let mut shapley = [0.0; 4];
    let mut order_range = [[f64::INFINITY, f64::NEG_INFINITY]; 4];
    for values in &by_order {
        for j in 0..4 {
            shapley[j] += values[j] / 24.0;
            order_range[j][0] = order_range[j][0].min(values[j]);
            order_range[j][1] = order_range[j][1].max(values[j]);
        }
    }
    let total = hybrids[15] - hybrids[0];
    if !total.is_finite()
        || shapley
            .iter()
            .chain(&sequential)
            .chain(by_order.iter().flatten())
            .any(|v| !v.is_finite())
    {
        return Err(invalid("decomposition arithmetic overflow"));
    }
    Ok(ReferenceDecomposition {
        hybrids,
        total,
        sequential,
        shapley,
        orderings,
        by_order,
        order_range,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    const ORDER: [ChangeComponent; 4] = [
        ChangeComponent::Exposure,
        ChangeComponent::Saturation,
        ChangeComponent::Population,
        ChangeComponent::Mechanism,
    ];

    #[test]
    fn interaction_is_shared_exactly_and_order_sensitivity_is_separate() {
        let hybrids = std::array::from_fn(|mask| {
            2.0 * (mask & 1) as f64
                + 3.0 * ((mask >> 1) & 1) as f64
                + if mask & 3 == 3 { 8.0 } else { 0.0 }
        });
        let result = decompose_values(hybrids, ORDER).unwrap();
        assert_eq!(result.sequential, [2.0, 11.0, 0.0, 0.0]);
        for (actual, expected) in result.shapley.iter().zip([6.0, 7.0, 0.0, 0.0]) {
            assert!((actual - expected).abs() < 1e-12);
        }
        assert_eq!(result.order_range[0], [2.0, 10.0]);
        assert_eq!(result.by_order.len(), 24);
        for order in result.by_order {
            assert_eq!(order.iter().sum::<f64>(), result.total);
        }
    }

    #[test]
    fn compiled_population_change_is_not_mechanism_change_and_failed_slots_survive() {
        let (data, draw, _) = super::super::tests::fixture();
        let program = super::super::tests::programs(&data).remove(3);
        let mut after = vec![0.0; data.groups.len()];
        after[0] = 1.0;
        let request = PanelDecompositionRequest {
            before: DecompositionEndpoint {
                program: program.clone(),
                population_weights: vec![0.25; 4],
            },
            after: DecompositionEndpoint { program, population_weights: after },
            time_weights: vec![1.0 / 24.0; 24],
            order: ORDER,
        };
        let path = PairedPanelPath { before: vec![draw.clone()], after: vec![draw] };
        let result = decompose_panel_law(
            &data,
            &request,
            &path,
            &[Some(path.clone()), None],
            &ExecutionContext::for_tests(9),
        )
        .unwrap();
        assert!(result.point.total.abs() > 1e-6);
        assert!((result.point.shapley[2] - result.point.total).abs() < 1e-12);
        for j in [0, 1, 3] {
            assert!(result.point.shapley[j].abs() < 1e-12);
        }
        assert!(result.draws[1].is_none());
        assert_eq!(result.draws[0].as_ref().unwrap(), &result.point);
        let mut limited = ExecutionContext::for_tests(9);
        limited.set_memory_limit(Some(1));
        assert!(matches!(
            decompose_panel_law(&data, &request, &path, &[], &limited),
            Err(PanelError::Resource(_))
        ));
        let cancelled = ExecutionContext::for_tests(9);
        cancelled.cancellation.cancel();
        assert!(matches!(
            decompose_panel_law(&data, &request, &path, &[], &cancelled),
            Err(PanelError::Cancelled)
        ));
    }

    #[test]
    fn joint_covariance_preserves_component_dependence_and_total_cancellation() {
        let (data, draw, _) = super::super::tests::fixture();
        let programs = super::super::tests::programs(&data);
        let request = PanelDecompositionRequest {
            before: DecompositionEndpoint {
                program: programs[0].clone(),
                population_weights: vec![0.25; 4],
            },
            after: DecompositionEndpoint {
                program: programs[3].clone(),
                population_weights: vec![0.25; 4],
            },
            time_weights: vec![1.0 / 24.0; 24],
            order: ORDER,
        };
        let point = PairedPanelPath { before: vec![draw.clone()], after: vec![draw.clone()] };
        let mut paths = Vec::new();
        for multiplier in [0.5, 1.0, 2.0] {
            let mut changed = draw.clone();
            changed.parameters.channels[0].coefficient *= multiplier;
            changed.parameters.half_saturation *= multiplier;
            paths.push(Some(PairedPanelPath { before: vec![draw.clone()], after: vec![changed] }));
        }
        paths.insert(1, None);
        let result =
            decompose_panel_law(&data, &request, &point, &paths, &ExecutionContext::for_tests(82))
                .unwrap();
        let joint = result.joint_uncertainty().unwrap();
        assert_eq!((joint.requested, joint.complete), (4, 3));
        let draws = result.draws.iter().flatten().collect::<Vec<_>>();
        let mean = draws.iter().map(|draw| draw.total).sum::<f64>() / 3.0;
        let independent_variance =
            draws.iter().map(|draw| (draw.total - mean).powi(2)).sum::<f64>() / 2.0;
        let hybrid_mean: [f64; 16] =
            std::array::from_fn(|j| draws.iter().map(|draw| draw.hybrids[j]).sum::<f64>() / 3.0);
        let hybrid_covariance: [[f64; 16]; 16] = std::array::from_fn(|j| {
            std::array::from_fn(|k| {
                draws
                    .iter()
                    .map(|draw| {
                        (draw.hybrids[j] - hybrid_mean[j]) * (draw.hybrids[k] - hybrid_mean[k])
                    })
                    .sum::<f64>()
                    / 2.0
            })
        });
        let propagated = propagate_hybrid_covariance(&hybrid_covariance, ORDER).unwrap();
        for j in 0..4 {
            for k in 0..4 {
                assert!((propagated.shapley[j][k] - joint.shapley_covariance[j][k]).abs() < 1e-10);
                assert!(
                    (propagated.sequential[j][k] - joint.sequential_covariance[j][k]).abs() < 1e-10
                );
            }
        }
        assert!((propagated.total - independent_variance).abs() < 1e-10);
        assert!((joint.total_variance - independent_variance).abs() < 1e-12);
        for covariance in [joint.shapley_covariance, joint.sequential_covariance] {
            assert!(
                (covariance.iter().flatten().sum::<f64>() - independent_variance).abs() < 1e-10
            );
            for j in 0..4 {
                for k in 0..4 {
                    assert!((covariance[j][k] - covariance[k][j]).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn temporal_parameter_mapping_preserves_selected_realizations_and_all_hybrid_totals() {
        use super::super::time_varying::{TemporalModelChoice, TemporalPanelDraw};
        let (data, draw, _) = super::super::tests::fixture();
        let programs = super::super::tests::programs(&data);
        let request = PanelDecompositionRequest {
            before: DecompositionEndpoint {
                program: programs[0].clone(),
                population_weights: vec![0.25; 4],
            },
            after: DecompositionEndpoint {
                program: programs[3].clone(),
                population_weights: vec![0.25; 4],
            },
            time_weights: vec![1.0 / 24.0; 24],
            order: ORDER,
        };
        let periods = MechanismPeriodMap { before: vec![0; 24], after: (0..24).collect() };
        let temporal = |choice, shift| {
            let parameters = (0..24)
                .map(|time| {
                    let mut value = draw.parameters.clone();
                    value.intercept += shift * time as f64;
                    value
                })
                .collect();
            TemporalPanelDraw { baseline: draw.clone(), parameters, choice }
        };
        let point = temporal(TemporalModelChoice::NoChange, 0.0);
        let laws = [
            Some(temporal(TemporalModelChoice::Piecewise { at: 12 }, 0.1)),
            None,
            Some(temporal(TemporalModelChoice::Smooth { penalty: 1.0 }, 0.2)),
        ];
        let result = decompose_temporal_law(
            &data,
            &request,
            &periods,
            &point,
            &laws,
            &ExecutionContext::for_tests(83),
        )
        .unwrap();
        assert_eq!(
            result.draws.iter().map(Option::is_some).collect::<Vec<_>>(),
            vec![true, false, true]
        );
        for draw in result.draws.iter().flatten() {
            assert!((draw.shapley.iter().sum::<f64>() - draw.total).abs() < 1e-10);
            for order in &draw.by_order {
                assert!((order.iter().sum::<f64>() - draw.total).abs() < 1e-10);
            }
            assert!(draw.shapley[3] > 1.0);
            assert!(draw.shapley[2].abs() < 1e-12);
        }
        assert!(result.joint_uncertainty().is_some());
    }

    #[test]
    fn streamed_moments_equal_the_retained_law_without_holding_paths_or_tables() {
        use super::super::time_varying::{TemporalModelChoice, TemporalPanelDraw};
        let (data, draw, _) = super::super::tests::fixture();
        let programs = super::super::tests::programs(&data);
        let request = PanelDecompositionRequest {
            before: DecompositionEndpoint {
                program: programs[0].clone(),
                population_weights: vec![0.25; 4],
            },
            after: DecompositionEndpoint {
                program: programs[3].clone(),
                population_weights: vec![0.25; 4],
            },
            time_weights: vec![1.0 / 24.0; 24],
            order: ORDER,
        };
        let periods = MechanismPeriodMap { before: vec![0; 24], after: (0..24).collect() };
        let temporal = |choice, shift: f64| {
            let parameters = (0..24)
                .map(|time| {
                    let mut value = draw.parameters.clone();
                    value.intercept += shift * time as f64;
                    value.half_saturation *= 1.0 + shift;
                    value
                })
                .collect();
            TemporalPanelDraw { baseline: draw.clone(), parameters, choice }
        };
        let point = temporal(TemporalModelChoice::NoChange, 0.0);
        let laws = vec![
            Some(temporal(TemporalModelChoice::Piecewise { at: 12 }, 0.1)),
            None,
            Some(temporal(TemporalModelChoice::Smooth { penalty: 1.0 }, 0.2)),
            Some(temporal(TemporalModelChoice::Piecewise { at: 8 }, 0.35)),
        ];
        let ctx = ExecutionContext::for_tests(86);
        let retained =
            decompose_temporal_law(&data, &request, &periods, &point, &laws, &ctx).unwrap();
        let streamed =
            decompose_temporal_law_streamed(&data, &request, &periods, &point, &laws, &ctx)
                .unwrap();
        assert_eq!(streamed.point, retained.point);
        assert_eq!(streamed.joint, retained.joint_uncertainty());
        assert!(streamed.joint.is_some());
        assert_eq!((streamed.requested, streamed.failed), (4, 1));
        // A stationary law streams through the same period map and agrees with its retained form.
        let stationary = MechanismPeriodMap { before: vec![0; 24], after: vec![0; 24] };
        let draws = vec![Some(draw.clone()), None, Some(draw.clone())];
        let retained =
            decompose_invariant_law(&data, &request, &stationary, &draw, &draws, &ctx).unwrap();
        let streamed =
            decompose_invariant_law_streamed(&data, &request, &stationary, &draw, &draws, &ctx)
                .unwrap();
        assert_eq!(streamed.point, retained.point);
        assert_eq!(streamed.joint, retained.joint_uncertainty());
        // A bad period map or a cancelled context refuses before any draw is evaluated.
        let short = MechanismPeriodMap { before: vec![0; 3], after: vec![0; 3] };
        assert!(
            decompose_invariant_law_streamed(&data, &request, &short, &draw, &draws, &ctx).is_err()
        );
        let cancelled = ExecutionContext::for_tests(86);
        cancelled.cancellation.cancel();
        assert!(
            decompose_temporal_law_streamed(&data, &request, &periods, &point, &laws, &cancelled)
                .is_err()
        );
    }

    #[test]
    fn weighted_finite_law_preserves_importance_weights_and_joint_variance() {
        let table = |exposure: f64, mechanism: f64| {
            std::array::from_fn(|mask| {
                (if mask & 1 == 0 { 0.0 } else { exposure })
                    + (if mask & 8 == 0 { 0.0 } else { mechanism })
            })
        };
        let draws = [Some(table(2.0, 1.0)), Some(table(10.0, -3.0))];
        let point = table(8.0, -2.0);
        let result = decompose_weighted_hybrid_law(
            point,
            &draws,
            Some(&[1.0, 3.0]),
            ORDER,
            &ExecutionContext::for_tests(84),
        )
        .unwrap();
        assert_eq!(result.weights, Some(vec![1.0, 3.0]));
        let moments = result.joint_uncertainty().unwrap();
        assert!(moments.weighted_finite_law);
        assert_eq!(moments.shapley_mean, [8.0, 0.0, 0.0, -2.0]);
        assert!((moments.shapley_covariance[0][0] - 12.0).abs() < 1e-12);
        assert!((moments.shapley_covariance[0][3] + 6.0).abs() < 1e-12);
        assert!((moments.total_variance - 3.0).abs() < 1e-12);
        assert!(
            decompose_weighted_hybrid_law(
                point,
                &[draws[0], None],
                Some(&[1.0, 3.0]),
                ORDER,
                &ExecutionContext::for_tests(84)
            )
            .is_err()
        );
    }

    #[test]
    fn bad_order_and_nonfinite_values_refuse() {
        let mut order = ORDER;
        order[3] = order[0];
        assert!(decompose_values([0.0; 16], order).is_err());
        assert!(decompose_values([f64::NAN; 16], ORDER).is_err());
    }

    #[test]
    fn disjoint_component_changes_recover_the_only_changed_block_in_static_and_temporal_forms() {
        for lag in [0, 2] {
            let (mut data, mut draw, _) = super::super::tests::fixture();
            data.lag_order = lag;
            if lag == 0 {
                draw.parameters.rho = 0.0;
                for channel in &mut draw.parameters.channels {
                    channel.lag = [1.0, 0.0, 0.0];
                }
            }
            let programs = super::super::tests::programs(&data);
            for component in 0..4 {
                let before = DecompositionEndpoint {
                    program: programs[3].clone(),
                    population_weights: vec![0.25; 4],
                };
                let mut after = before.clone();
                let mut after_draw = draw.clone();
                match component {
                    0 => after.program = programs[0].clone(),
                    1 => after_draw.parameters.half_saturation *= 2.0,
                    2 => after.population_weights = vec![1.0, 0.0, 0.0, 0.0],
                    3 => after_draw.parameters.intercept += 1.0,
                    _ => unreachable!(),
                }
                let request = PanelDecompositionRequest {
                    before,
                    after,
                    time_weights: vec![1.0 / 24.0; 24],
                    order: ORDER,
                };
                let path = PairedPanelPath { before: vec![draw.clone()], after: vec![after_draw] };
                let result = decompose_panel_law(
                    &data,
                    &request,
                    &path,
                    &[],
                    &ExecutionContext::for_tests(9),
                )
                .unwrap();
                assert!(result.point.total.abs() > 1e-6);
                for j in 0..4 {
                    let expected = if j == component { result.point.total } else { 0.0 };
                    assert!(
                        (result.point.shapley[j] - expected).abs() < 1e-10,
                        "lag={lag}, changed={component}, contribution={j}"
                    );
                }
            }
        }
    }
}
