// SPDX-License-Identifier: MIT OR Apache-2.0
//! Gaussian AR(1) precision plus rank-one random-intercept integration.
use super::family::{PanelFamilyParameters, PanelSpatialKernel, SPATIAL_JITTER};
use super::{
    PanelChannelParameters, PanelData, PanelError, PanelGroup, PanelInteractionParameter,
    PanelParameters, PanelProgram, PanelScenario, invalid,
};

/// Bound once at a numerical execution boundary; never constructs a causal model or builder.
pub(super) struct CompiledPanel<'a> {
    pub source: &'a PanelData,
    pub(super) simulated_outcomes: Option<Vec<Vec<f64>>>,
    features: Vec<antecedent_model::program::NodeProgram>,
    /// The portable expression of every feature, for exact partial derivatives.
    pub(super) expressions: Vec<antecedent_core::ValueExpr>,
    /// The expression variable of every channel, in channel order.
    pub(super) axis: Vec<antecedent_core::VariableId>,
    /// The channel columns of every feature that is the plain product of two channels.
    products: Vec<Option<(usize, usize)>>,
}
impl std::ops::Deref for CompiledPanel<'_> {
    type Target = PanelData;
    fn deref(&self) -> &PanelData {
        self.source
    }
}
impl<'a> CompiledPanel<'a> {
    pub fn new(source: &'a PanelData) -> Result<Self, PanelError> {
        let axis = source.channel_variables()?;
        let features = source
            .interactions
            .iter()
            .map(|feature| {
                antecedent_model::program::NodeProgram::compile_columns(
                    &axis,
                    &feature.expression.to_domain(),
                )
                .map_err(|error| invalid(&format!("programmable feature compilation: {error:?}")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let expressions: Vec<_> =
            source.interactions.iter().map(|f| f.expression.to_domain()).collect();
        let products =
            expressions.iter().map(|expression| product_columns(&axis, expression)).collect();
        Ok(Self { source, simulated_outcomes: None, features, expressions, axis, products })
    }
    /// The active outcome column for a group, with the source design shared across refits.
    pub(super) fn outcome(&self, group: usize) -> &[f64] {
        self.simulated_outcomes
            .as_ref()
            .map_or_else(|| self.groups[group].outcome.as_slice(), |columns| &columns[group])
    }

    pub(super) fn initial(&self) -> PanelParameters {
        initial_with(self, |group| self.outcome(group))
    }

    /// One interaction feature at the saturated channel values, as the mechanism evaluates it.
    pub fn feature_value(
        &self,
        index: usize,
        saturated: &[f64],
        values: &mut Vec<f64>,
        scratch: &mut Vec<f64>,
    ) -> f64 {
        if let Some(columns) = self.products[index] {
            return product_value(self.axis.len(), saturated, columns);
        }
        self.features[index]
            .evaluate_columns_with_scratch(saturated, values, scratch, None)
            .unwrap_or(f64::NAN)
    }
    pub fn workspace_bytes(&self) -> Result<usize, PanelError> {
        let feature = self.features.iter().try_fold(0_usize, |peak, program| {
            let bytes =
                program.column_workspace_bytes().map_err(|error| invalid(&error.to_string()))?;
            Ok::<_, PanelError>(peak.max(
                usize::try_from(bytes).map_err(|_| invalid("feature workspace size overflow"))?,
            ))
        })?;
        let spatial = if self.family.spatial.is_some() {
            // Correlation, two Cholesky factors, the conditional covariance and solves.
            self.groups.len().checked_mul(self.groups.len()).and_then(|n| n.checked_mul(64))
        } else {
            Some(0)
        };
        self.channels
            .len()
            .checked_mul(16)
            .and_then(|n| n.checked_add(feature))
            .and_then(|n| n.checked_add(spatial?))
            .ok_or_else(|| invalid("feature workspace size overflow"))
    }
    pub fn plan_bytes(&self) -> Result<usize, PanelError> {
        self.features.iter().try_fold(0_usize, |sum, program| {
            let bytes = program.column_plan_bytes().map_err(|error| invalid(&error.to_string()))?;
            sum.checked_add(
                usize::try_from(bytes).map_err(|_| invalid("feature plan size overflow"))?,
            )
            .ok_or_else(|| invalid("feature plan size overflow"))
        })
    }
}

pub(super) fn parameter_count(data: &PanelData) -> usize {
    layout(data).total
}
/// Positions of every coordinate in the unconstrained parameter vector. The reviewed family
/// keeps the original order: linear block, lag logits, log k, atanh rho, log sigma, log tau.
/// Broader-family coordinates sit between log k and rho (shapes, class half-saturations)
/// or after log tau (spatial range).
pub(super) struct Layout {
    pub base: usize,
    pub lag: usize,
    pub k0: usize,
    pub shape: usize,
    pub shape_len: usize,
    pub class: usize,
    pub class_len: usize,
    pub rho: Option<usize>,
    pub sigma: usize,
    pub tau: usize,
    pub range: Option<usize>,
    pub total: usize,
}
pub(super) fn layout(data: &PanelData) -> Layout {
    let d = data.channels.len();
    let base = linear_count(data);
    let lagged = data.lag_order == 2;
    let lag_len = match (lagged, data.family.is_geometric()) {
        (false, _) => 0,
        (true, true) => d,
        (true, false) => 2 * d,
    };
    let k0 = base + lag_len;
    let shape = k0 + 1;
    let shape_len = if data.family.is_hill() { d } else { 0 };
    let class = shape + shape_len;
    let class_len = data.family.classes() - 1;
    let mut next = class + class_len;
    let rho = lagged.then_some(next);
    if lagged {
        next += 1;
    }
    let (sigma, tau) = (next, next + 1);
    next += 2;
    let range = data.family.spatial.is_some().then_some(next);
    if range.is_some() {
        next += 1;
    }
    Layout {
        base,
        lag: base,
        k0,
        shape,
        shape_len,
        class,
        class_len,
        rho,
        sigma,
        tau,
        range,
        total: next,
    }
}
pub(super) fn linear_count(data: &PanelData) -> usize {
    data.channels.len() + data.interactions.len() + data.seasons + 1
}
pub(super) fn validate_parameters(data: &PanelData, p: &PanelParameters) -> Result<(), PanelError> {
    if p.channels.len() != data.channels.len()
        || p.interactions.len() != data.interactions.len()
        || p.channels
            .iter()
            .zip(&data.channels)
            .any(|(p, d)| p.id != d.id || !p.coefficient.is_finite())
        || p.interactions
            .iter()
            .zip(&data.interactions)
            .any(|(p, d)| p.id != d.id || !p.coefficient.is_finite())
        || [p.intercept, p.confounder]
            .iter()
            .chain(&p.season_effects)
            .any(|value| !value.is_finite())
        || p.season_effects.len() != data.seasons
        || p.season_effects[0] != 0.0
        || ![p.half_saturation, p.innovation_sd, p.group_sd]
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
        || !p.rho.is_finite()
        || p.rho.abs() >= 1.0
        || (data.lag_order == 0 && p.rho != 0.0)
    {
        return Err(invalid(
            "mechanism channel/product identities, coefficients, scales or seasons",
        ));
    }
    for channel in &p.channels {
        let weights = channel.lag;
        if weights.iter().any(|value| !value.is_finite() || *value < 0.0)
            || (weights.iter().sum::<f64>() - 1.0).abs() > 1e-10
            || (data.lag_order == 0
                && weights
                    .iter()
                    .zip([1.0_f64, 0.0, 0.0])
                    .any(|(actual, expected)| actual.to_bits() != expected.to_bits()))
            || (data.lag_order == 2 && weights.iter().any(|value| *value <= 0.0))
        {
            return Err(invalid("lag weights must match the declared simplex lag order"));
        }
        if data.family.is_geometric() {
            // w = (1, theta, theta^2) / (1 + theta + theta^2) with theta in (0,1).
            let tolerance = 1e-9 * weights[0] * weights[2];
            if (weights[1] * weights[1] - weights[0] * weights[2]).abs() > tolerance
                // The decay reaches one in floating point when its logit is large: the
                // uniform weights are the closed limit of the family, not a different law.
                || weights[1] > weights[0] * (1.0 + 1e-12)
            {
                return Err(invalid("lag weights must be a decaying geometric adstock"));
            }
        }
    }
    let family = &p.family;
    let hill = if data.family.is_hill() { data.channels.len() } else { 0 };
    let spatial = data.family.spatial.is_some();
    if family.hill_shape.len() != hill
        || family.class_half_saturation.len() != data.family.classes() - 1
        || family
            .hill_shape
            .iter()
            .chain(&family.class_half_saturation)
            .any(|value| !value.is_finite() || *value <= 0.0)
        || (spatial && !(family.spatial_range.is_finite() && family.spatial_range > 0.0))
        || (!spatial && family.spatial_range != 0.0)
    {
        return Err(invalid(
            "Hill shapes, class half-saturations and spatial range must match the declared family",
        ));
    }
    Ok(())
}
pub(super) fn encode(data: &PanelData, p: &PanelParameters) -> Result<Vec<f64>, PanelError> {
    validate_parameters(data, p)?;
    let mut x = vec![p.intercept];
    x.extend(p.channels.iter().map(|channel| channel.coefficient));
    x.extend(p.interactions.iter().map(|interaction| interaction.coefficient));
    x.push(p.confounder);
    x.extend_from_slice(&p.season_effects[1..]);
    if data.lag_order == 2 {
        for channel in &p.channels {
            let w = channel.lag;
            if data.family.is_geometric() {
                let theta = w[1] / w[0];
                x.push((theta / (1.0 - theta)).ln());
            } else {
                x.extend([(w[0] / w[2]).ln(), (w[1] / w[2]).ln()]);
            }
        }
    }
    x.push(p.half_saturation.ln());
    x.extend(p.family.hill_shape.iter().map(|shape| shape.ln()));
    x.extend(p.family.class_half_saturation.iter().map(|k| k.ln()));
    if data.lag_order == 2 {
        x.push(p.rho.atanh());
    }
    x.extend([p.innovation_sd.ln(), p.group_sd.ln()]);
    if data.family.spatial.is_some() {
        x.push(p.family.spatial_range.ln());
    }
    Ok(x)
}
fn simplex(x: &[f64]) -> [f64; 3] {
    let max = x[0].max(x[1]).max(0.0);
    let u = [(x[0] - max).exp(), (x[1] - max).exp(), (-max).exp()];
    let sum = u.iter().sum::<f64>();
    u.map(|value| value / sum)
}
/// Normalized truncated geometric weights from the logit of the decay.
pub(super) fn geometric(logit: f64) -> [f64; 3] {
    let theta = crate::util::sigmoid(logit);
    let u = [1.0, theta, theta * theta];
    let sum = u.iter().sum::<f64>();
    u.map(|value| value / sum)
}
/// `exp` of a log-scale coordinate as a positive finite scale. The optimizer may run a flat
/// direction (a variance component at its boundary) past the range where `exp` underflows
/// to zero or overflows to infinity; those coordinates decode to the smallest normal and the
/// largest finite scale, which are the representable boundary values of the same flat
/// likelihood, instead of making the whole parameter vector invalid.
pub(super) fn positive_exp(log_scale: f64) -> f64 {
    log_scale.exp().clamp(f64::MIN_POSITIVE, f64::MAX)
}
/// `d positive_exp(x) / d x` divided by `positive_exp(x)`: one inside the representable range
/// and zero where the decoded scale is held at a boundary value.
pub(super) fn positive_exp_log_slope(log_scale: f64) -> f64 {
    let scale = log_scale.exp();
    if scale > f64::MIN_POSITIVE && scale < f64::MAX { 1.0 } else { 0.0 }
}
pub(super) fn decode(data: &PanelData, x: &[f64]) -> Option<PanelParameters> {
    let layout = layout(data);
    if x.len() != layout.total || x.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let d = data.channels.len();
    let i = data.interactions.len();
    let base = layout.base;
    let mut season_effects = vec![0.0];
    season_effects.extend_from_slice(&x[2 + d + i..base]);
    let p = PanelParameters {
        intercept: x[0],
        channels: data
            .channels
            .iter()
            .enumerate()
            .map(|(index, channel)| PanelChannelParameters {
                id: channel.id.clone(),
                coefficient: x[1 + index],
                lag: if data.lag_order != 2 {
                    [1.0, 0.0, 0.0]
                } else if data.family.is_geometric() {
                    geometric(x[layout.lag + index])
                } else {
                    simplex(&x[layout.lag + 2 * index..layout.lag + 2 * index + 2])
                },
            })
            .collect(),
        interactions: data
            .interactions
            .iter()
            .enumerate()
            .map(|(index, interaction)| PanelInteractionParameter {
                id: interaction.id.clone(),
                coefficient: x[1 + d + index],
            })
            .collect(),
        confounder: x[1 + d + i],
        season_effects,
        half_saturation: positive_exp(x[layout.k0]),
        rho: layout.rho.map_or(0.0, |index| x[index].tanh()),
        innovation_sd: positive_exp(x[layout.sigma]),
        group_sd: positive_exp(x[layout.tau]),
        family: PanelFamilyParameters {
            hill_shape: x[layout.shape..layout.shape + layout.shape_len]
                .iter()
                .map(|value| positive_exp(*value))
                .collect(),
            class_half_saturation: x[layout.class..layout.class + layout.class_len]
                .iter()
                .map(|value| positive_exp(*value))
                .collect(),
            spatial_range: layout.range.map_or(0.0, |index| positive_exp(x[index])),
        },
    };
    validate_parameters(data, &p).ok().map(|()| p)
}
pub(super) fn initial(data: &PanelData) -> PanelParameters {
    initial_with(data, |group| &data.groups[group].outcome)
}

fn initial_with<'a>(data: &PanelData, outcome: impl Fn(usize) -> &'a [f64]) -> PanelParameters {
    let analysed = || (0..data.groups.len()).flat_map(&outcome).filter(|v| v.is_finite());
    let n = analysed().count() as f64;
    let mean = analysed().sum::<f64>() / n;
    let variance = analysed().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let scale = variance.sqrt().max(0.1);
    let lag = match (data.lag_order == 2, data.family.is_geometric()) {
        (false, _) => [1.0, 0.0, 0.0],
        (true, false) => [0.6, 0.3, 0.1],
        // theta = 1/2
        (true, true) => [4.0 / 7.0, 2.0 / 7.0, 1.0 / 7.0],
    };
    let mut exposures: Vec<_> = data
        .groups
        .iter()
        .enumerate()
        .flat_map(|(index, group)| {
            let outcome = outcome(index);
            group.channels.iter().flat_map(move |values| {
                values.iter().zip(outcome).filter(|(_, y)| y.is_finite()).map(|(v, _)| v)
            })
        })
        .copied()
        .filter(|v| *v > 0.0)
        .collect();
    exposures.sort_by(f64::total_cmp);
    let half_saturation = exposures.get(exposures.len() / 2).copied().unwrap_or(1.0);
    PanelParameters {
        intercept: mean,
        channels: data
            .channels
            .iter()
            .map(|c| PanelChannelParameters { id: c.id.clone(), coefficient: 0.0, lag })
            .collect(),
        interactions: data
            .interactions
            .iter()
            .map(|c| PanelInteractionParameter { id: c.id.clone(), coefficient: 0.0 })
            .collect(),
        confounder: 0.0,
        season_effects: vec![0.0; data.seasons],
        half_saturation,
        rho: if data.lag_order == 2 { 0.1 } else { 0.0 },
        innovation_sd: scale * 0.8,
        group_sd: scale * 0.3,
        family: PanelFamilyParameters {
            hill_shape: if data.family.is_hill() {
                vec![1.0; data.channels.len()]
            } else {
                Vec::new()
            },
            class_half_saturation: vec![half_saturation; data.family.classes() - 1],
            spatial_range: if data.family.spatial.is_some() {
                data.family.median_distance()
            } else {
                0.0
            },
        },
    }
}
/// The value the compiled column program of `left * right` returns, without running the
/// executor: it refuses an input row of the wrong width or with a non-finite value in any
/// column and a non-finite result (each read by the mechanism as NaN), and otherwise applies
/// the same `ArithOp::apply`, so the bits are identical.
fn product_value(width: usize, saturated: &[f64], (left, right): (usize, usize)) -> f64 {
    if saturated.len() != width || saturated.iter().any(|value| !value.is_finite()) {
        return f64::NAN;
    }
    let value = antecedent_core::ArithOp::Mul.apply(saturated[left], saturated[right]);
    if value.is_finite() { value } else { f64::NAN }
}
/// The channel columns of a feature written as the product of two channel variables.
fn product_columns(
    axis: &[antecedent_core::VariableId],
    expression: &antecedent_core::ValueExpr,
) -> Option<(usize, usize)> {
    use antecedent_core::{ArithOp, ValueExpr};
    let ValueExpr::Arith { op: ArithOp::Mul, lhs, rhs } = expression else {
        return None;
    };
    let (ValueExpr::Var(left), ValueExpr::Var(right)) = (lhs.as_ref(), rhs.as_ref()) else {
        return None;
    };
    let column = |id| axis.iter().position(|variable| variable == id);
    Some((column(left)?, column(right)?))
}
pub(super) fn exposure(values: &[f64], history: [f64; 2], weights: [f64; 3], time: usize) -> f64 {
    weights
        .into_iter()
        .enumerate()
        .map(|(lag, weight)| {
            weight * if lag <= time { values[time - lag] } else { history[2 + time - lag] }
        })
        .sum()
}
/// Saturated lag-weighted exposure; `shape` is `None` for Michaelis-Menten.
pub(super) fn saturate(x: f64, k: f64, shape: Option<f64>) -> f64 {
    match shape {
        None => x / (x + k),
        Some(shape) => {
            if x <= 0.0 {
                return 0.0;
            }
            let z = (x / k).powf(shape);
            if z.is_infinite() { 1.0 } else { z / (1.0 + z) }
        }
    }
}
#[cfg(test)]
pub(super) fn response(
    data: &CompiledPanel<'_>,
    p: &PanelParameters,
    exposures: &[f64],
    z: f64,
    season: usize,
    class: usize,
) -> f64 {
    combine(p, mechanism(data, p, exposures, class), z, season)
}
/// Adds the exposure-independent terms in the order the response is defined by: intercept,
/// saturated main effects, interaction products, confounder, season.
pub(super) fn combine(
    p: &PanelParameters,
    (main, products): (f64, f64),
    z: f64,
    season: usize,
) -> f64 {
    p.intercept + main + products + p.confounder * z + p.season_effects[season]
}
/// The saturated main effects and interaction products of one exposure vector.
#[cfg(test)]
pub(super) fn mechanism(
    data: &CompiledPanel<'_>,
    p: &PanelParameters,
    exposures: &[f64],
    class: usize,
) -> (f64, f64) {
    mechanism_with(data, p, exposures, class, &mut ResponseScratch::default())
}
/// Reusable buffers of one response evaluation. Reuse never changes a value: every buffer is
/// cleared or fully overwritten before it is read.
#[derive(Default)]
pub(super) struct ResponseScratch {
    exposures: Vec<f64>,
    saturated: Vec<f64>,
    values: Vec<f64>,
    scratch: Vec<f64>,
}
fn mechanism_with(
    data: &CompiledPanel<'_>,
    p: &PanelParameters,
    exposures: &[f64],
    class: usize,
    buffers: &mut ResponseScratch,
) -> (f64, f64) {
    let k = p.half_saturation_of(class);
    let ResponseScratch { saturated, values, scratch, .. } = buffers;
    saturated.clear();
    saturated.extend(
        exposures
            .iter()
            .enumerate()
            .map(|(index, x)| saturate(*x, k, p.family.hill_shape.get(index).copied())),
    );
    let saturated: &[f64] = saturated;
    let main = p.channels.iter().zip(saturated).map(|(c, x)| c.coefficient * x).sum::<f64>();
    let products = p
        .interactions
        .iter()
        .take(data.features.len())
        .enumerate()
        .map(|(index, param)| {
            // The common production expression executor validates every numerical result.
            param.coefficient * data.feature_value(index, saturated, values, scratch)
        })
        .sum::<f64>();
    (main, products)
}
pub(super) fn observed_response(
    data: &CompiledPanel<'_>,
    p: &PanelParameters,
    group: &PanelGroup,
    group_index: usize,
    time: usize,
) -> f64 {
    observed_response_with(data, p, group, group_index, time, &mut ResponseScratch::default())
}
/// `observed_response` over caller-owned buffers: the same operations in the same order.
pub(super) fn observed_response_with(
    data: &CompiledPanel<'_>,
    p: &PanelParameters,
    group: &PanelGroup,
    group_index: usize,
    time: usize,
    buffers: &mut ResponseScratch,
) -> f64 {
    let mut exposures = std::mem::take(&mut buffers.exposures);
    exposures.clear();
    exposures.extend(
        group
            .channels
            .iter()
            .zip(&group.history)
            .zip(&p.channels)
            .map(|((values, history), channel)| exposure(values, *history, channel.lag, time)),
    );
    let mechanism = mechanism_with(data, p, &exposures, data.family.class_of(group_index), buffers);
    buffers.exposures = exposures;
    combine(p, mechanism, group.confounder[time], group.season[time])
}
/// The mechanism value of the last exposure vector a scenario saw. A program holds its
/// schedule constant over stretches (and every adstock window fills after the carryover
/// order), so consecutive periods with bit-identical exposures share one evaluation of the
/// saturation and interaction program; the value is the same number the direct call returns.
#[derive(Default)]
pub(super) struct MechanismMemo {
    exposures: Vec<f64>,
    candidate: Vec<f64>,
    class: usize,
    value: Option<(f64, f64)>,
    buffers: ResponseScratch,
}
impl MechanismMemo {
    /// Drops the remembered value (kept buffers never change a value): the next call evaluates
    /// the mechanism. A memo reused across different parameters must forget between them.
    pub(super) fn forget(&mut self) {
        self.value = None;
    }
}
/// Where one scenario reads each channel: the index of the program's schedule and of the
/// scenario's history for every source channel, in channel order, and the scenario group's
/// class. Resolved once per scenario instead of by identity at every period; the indices are
/// those the identity search returns, so the response is the same number.
pub(super) struct ScenarioColumns {
    program: Vec<usize>,
    history: Vec<usize>,
    class: usize,
}
impl ScenarioColumns {
    /// The columns against the source channel order, which every validated parameter value
    /// shares (`validate_parameters` checks the identities in order).
    pub(super) fn resolve(
        data: &CompiledPanel<'_>,
        program: &PanelProgram,
        scenario: &PanelScenario,
    ) -> Self {
        Self::resolve_ids(
            data,
            data.channels.iter().map(|channel| channel.id.as_str()),
            program,
            scenario,
        )
    }
    fn resolve_ids<'a>(
        data: &CompiledPanel<'_>,
        ids: impl Iterator<Item = &'a str>,
        program: &PanelProgram,
        scenario: &PanelScenario,
    ) -> Self {
        let (mut columns, mut history) = (Vec::new(), Vec::new());
        for id in ids {
            columns.push(
                program
                    .channels
                    .iter()
                    .position(|c| c.id == id)
                    .expect("validated program channel"),
            );
            history.push(
                scenario
                    .history
                    .iter()
                    .position(|c| c.id == id)
                    .expect("validated scenario history"),
            );
        }
        let class = match data.family.classes() {
            1 => 0,
            _ => data
                .groups
                .iter()
                .position(|group| group.id == scenario.group)
                .map_or(0, |index| data.family.class_of(index)),
        };
        Self { program: columns, history, class }
    }
}
pub(super) fn program_response(
    data: &CompiledPanel<'_>,
    p: &PanelParameters,
    program: &PanelProgram,
    scenario: &PanelScenario,
    time: usize,
) -> f64 {
    program_response_memo(data, p, program, scenario, time, &mut MechanismMemo::default())
}
pub(super) fn program_response_memo(
    data: &CompiledPanel<'_>,
    p: &PanelParameters,
    program: &PanelProgram,
    scenario: &PanelScenario,
    time: usize,
    memo: &mut MechanismMemo,
) -> f64 {
    // Resolved by the parameter value's own channel identities: this entry point does not
    // require a validated value.
    let columns = ScenarioColumns::resolve_ids(
        data,
        p.channels.iter().map(|channel| channel.id.as_str()),
        program,
        scenario,
    );
    program_response_columns(data, p, program, scenario, &columns, time, memo)
}
/// `program_response_memo` with the scenario's columns resolved by the caller, for a
/// parameter value whose channels are in the source order (`validate_parameters`).
pub(super) fn program_response_columns(
    data: &CompiledPanel<'_>,
    p: &PanelParameters,
    program: &PanelProgram,
    scenario: &PanelScenario,
    columns: &ScenarioColumns,
    time: usize,
    memo: &mut MechanismMemo,
) -> f64 {
    let mut exposures = std::mem::take(&mut memo.candidate);
    exposures.clear();
    exposures.extend(p.channels.iter().enumerate().map(|(index, channel)| {
        let values = &program.channels[columns.program[index]].values;
        let history = scenario.history[columns.history[index]].values;
        exposure(values, history, channel.lag, time)
    }));
    let class = columns.class;
    let same = memo.value.is_some()
        && memo.class == class
        && memo.exposures.len() == exposures.len()
        && memo.exposures.iter().zip(&exposures).all(|(a, b)| a.to_bits() == b.to_bits());
    let mechanism_value = if same {
        memo.value.unwrap_or((f64::NAN, f64::NAN))
    } else {
        let value = mechanism_with(data, p, &exposures, class, &mut memo.buffers);
        std::mem::swap(&mut memo.exposures, &mut exposures);
        memo.class = class;
        memo.value = Some(value);
        value
    };
    memo.candidate = exposures;
    combine(p, mechanism_value, scenario.confounder[time], scenario.season[time])
}
/// Conservatively retained heap and inline bytes for one complete global parameter value.
pub(super) fn parameter_bytes(data: &PanelData) -> Result<usize, PanelError> {
    data.channels
        .iter()
        .try_fold(std::mem::size_of::<PanelParameters>() + data.seasons * 8, |n, c| {
            n.checked_add(std::mem::size_of::<PanelChannelParameters>())
                .and_then(|n| n.checked_add(c.id.len()))
                .ok_or_else(|| invalid("parameter storage overflow"))
        })?
        .checked_add(data.interactions.iter().try_fold(0_usize, |n, c| {
            n.checked_add(std::mem::size_of::<PanelInteractionParameter>())
                .and_then(|n| n.checked_add(c.id.len()))
                .ok_or_else(|| invalid("interaction storage overflow"))
        })?)
        .and_then(|n| {
            // Family coordinates only; the reviewed family keeps its original quote.
            let extra = if data.family.is_reviewed() {
                0
            } else {
                8 * (data.channels.len() + data.family.classes())
            };
            n.checked_add(extra)
        })
        .ok_or_else(|| invalid("parameter storage overflow"))
}

pub(super) struct GaussianGroup {
    pub log_likelihood: f64,
    pub intercept_mean: f64,
    pub intercept_variance: f64,
}
pub(super) fn group_likelihood(
    data: &CompiledPanel<'_>,
    group_index: usize,
    p: &PanelParameters,
) -> Option<GaussianGroup> {
    let group = &data.groups[group_index];
    let mut buffers = ResponseScratch::default();
    let residuals: Vec<_> = (0..group.outcome.len())
        .map(|time| {
            group.outcome[time]
                - observed_response_with(data, p, group, group_index, time, &mut buffers)
        })
        .collect();
    let mut value = group_marginal(group, &residuals, p.rho, p.innovation_sd, p.group_sd)?;
    value.log_likelihood *= group.weight;
    value.log_likelihood.is_finite().then_some(value)
}
/// One group's marginal Gaussian law. Periods whose outcome is excluded (`NaN`) are
/// marginalized exactly; a non-finite residual at an analysed period is a numerical failure.
pub(super) fn group_marginal(
    group: &PanelGroup,
    residuals: &[f64],
    rho: f64,
    sigma: f64,
    tau: f64,
) -> Option<GaussianGroup> {
    outcome_marginal(&group.outcome, residuals, rho, sigma, tau)
}

fn outcome_marginal(
    outcome: &[f64],
    residuals: &[f64],
    rho: f64,
    sigma: f64,
    tau: f64,
) -> Option<GaussianGroup> {
    if outcome.iter().all(|value| value.is_finite()) {
        return ar1_random_intercept(residuals, rho, sigma, tau);
    }
    let observed = observed_periods(outcome, residuals)?;
    ar1_random_intercept_at(&observed, rho, sigma, tau)
}
/// `(period, residual)` of every analysed period, or `None` if one of them is not finite.
fn observed_periods(outcome: &[f64], residuals: &[f64]) -> Option<Vec<(usize, f64)>> {
    let mut observed = Vec::with_capacity(residuals.len());
    for (time, (&outcome, &residual)) in outcome.iter().zip(residuals).enumerate() {
        if outcome.is_finite() {
            if !residual.is_finite() {
                return None;
            }
            observed.push((time, residual));
        }
    }
    Some(observed)
}
/// Stationary AR(1) precision factors over analysed periods `t_1 < ... < t_m` at their true
/// time distances: `e_1 = sqrt(1-rho^2) r_1 / sigma` and, for a step of `d` periods,
/// `e_k = (r_k - rho^d r_{k-1}) / sqrt(v_d)` with `v_d = sigma^2 (1-rho^{2d})/(1-rho^2)`.
/// For consecutive periods this is exactly the tridiagonal precision of the closed form.
fn ar1_moments_at(observed: &[(usize, f64)], rho: f64, sigma: f64) -> Option<Ar1Moments> {
    if observed.is_empty()
        || !rho.is_finite()
        || rho.abs() >= 1.0
        || !sigma.is_finite()
        || sigma <= 0.0
        || observed.iter().any(|(_, value)| !value.is_finite())
    {
        return None;
    }
    let stationary = 1.0 - rho * rho;
    let variance = sigma * sigma;
    let first = stationary.sqrt() / sigma;
    let (mut ones, mut cross, mut quad) =
        (first * first, first * first * observed[0].1, (first * observed[0].1).powi(2));
    let mut log_det = variance.ln() - stationary.ln();
    for pair in observed.windows(2) {
        let distance = i32::try_from(pair[1].0 - pair[0].0).ok()?;
        let decay = rho.powi(distance);
        let step_variance = variance * (1.0 - rho.powi(2 * distance.min(1_000_000))) / stationary;
        let scale = 1.0 / step_variance.sqrt();
        let for_ones = (1.0 - decay) * scale;
        let for_residual = (pair[1].1 - decay * pair[0].1) * scale;
        ones += for_ones * for_ones;
        cross += for_ones * for_residual;
        quad += for_residual * for_residual;
        log_det += step_variance.ln();
    }
    Some(Ar1Moments { ones, cross, quad, log_det })
}
/// Random-intercept integration over analysed periods at their true time distances.
fn ar1_random_intercept_at(
    observed: &[(usize, f64)],
    rho: f64,
    sigma: f64,
    tau: f64,
) -> Option<GaussianGroup> {
    if !tau.is_finite() || tau <= 0.0 {
        return None;
    }
    let moments = ar1_moments_at(observed, rho, sigma)?;
    let tau2 = tau * tau;
    let shrinkage = 1.0 + tau2 * moments.ones;
    let conditional_variance = tau2 / shrinkage;
    let intercept_mean = conditional_variance * moments.cross;
    // Completing the square, as in the contiguous closed form: the centred quadratic form is
    // taken from the centred residuals, never as a difference of two large forms.
    let stationary = 1.0 - rho * rho;
    let first = stationary.sqrt() / sigma;
    let mut centered = (first * (observed[0].1 - intercept_mean)).powi(2);
    for pair in observed.windows(2) {
        let distance = i32::try_from(pair[1].0 - pair[0].0).ok()?;
        let decay = rho.powi(distance);
        let step_variance =
            sigma * sigma * (1.0 - rho.powi(2 * distance.min(1_000_000))) / stationary;
        centered += ((pair[1].1 - intercept_mean) - decay * (pair[0].1 - intercept_mean)).powi(2)
            / step_variance;
    }
    let quad = centered + (tau * moments.cross / shrinkage).powi(2);
    let log_det = moments.log_det + (tau2 * moments.ones).ln_1p();
    let log_likelihood =
        -0.5 * (observed.len() as f64 * std::f64::consts::TAU.ln() + log_det + quad);
    if !log_likelihood.is_finite()
        || !conditional_variance.is_finite()
        || conditional_variance < 0.0
    {
        return None;
    }
    Some(GaussianGroup { log_likelihood, intercept_mean, intercept_variance: conditional_variance })
}
/// Residual vector of every group under one mechanism.
pub(super) fn residuals(data: &CompiledPanel<'_>, p: &PanelParameters) -> Vec<Vec<f64>> {
    let mut buffers = ResponseScratch::default();
    data.groups
        .iter()
        .enumerate()
        .map(|(index, group)| {
            (0..group.outcome.len())
                .map(|time| {
                    group.outcome[time]
                        - observed_response_with(data, p, group, index, time, &mut buffers)
                })
                .collect()
        })
        .collect()
}

/// Closed-form stationary AR(1) precision and Sherman-Morrison intercept update.
pub(super) fn ar1_random_intercept(
    r: &[f64],
    rho: f64,
    sigma: f64,
    tau: f64,
) -> Option<GaussianGroup> {
    if r.is_empty()
        || !rho.is_finite()
        || rho.abs() >= 1.0
        || !sigma.is_finite()
        || sigma <= 0.0
        || !tau.is_finite()
        || tau <= 0.0
        || r.iter().any(|value| !value.is_finite())
    {
        return None;
    }
    let stationary = 1.0 - rho * rho;
    let mut one_r = stationary * r[0];
    for pair in r.windows(2) {
        let innovation = pair[1] - rho * pair[0];
        one_r += (1.0 - rho) * innovation;
    }
    let variance = sigma * sigma;
    let one_one = (stationary + (r.len() - 1) as f64 * (1.0 - rho).powi(2)) / variance;
    one_r /= variance;
    let tau2 = tau * tau;
    let shrinkage = 1.0 + tau2 * one_one;
    let conditional_variance = tau2 / shrinkage;
    let intercept_mean = conditional_variance * one_r;
    // Completing the square avoids subtracting two large nearly equal quadratic
    // forms when an intercept explains most of a group's residual mean.
    let mut centered_quad = stationary * (r[0] - intercept_mean).powi(2);
    for pair in r.windows(2) {
        centered_quad += ((pair[1] - intercept_mean) - rho * (pair[0] - intercept_mean)).powi(2);
    }
    let quad = centered_quad / variance + (tau * one_r / shrinkage).powi(2);
    let log_det = r.len() as f64 * variance.ln() - stationary.ln() + (tau2 * one_one).ln_1p();
    let log_likelihood = -0.5 * (r.len() as f64 * std::f64::consts::TAU.ln() + log_det + quad);
    if !log_likelihood.is_finite()
        || !conditional_variance.is_finite()
        || conditional_variance < 0.0
    {
        return None;
    }
    Some(GaussianGroup { log_likelihood, intercept_mean, intercept_variance: conditional_variance })
}
/// Detail of the marginal Gaussian law a caller needs.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Detail {
    /// Only the log likelihood.
    Value,
    /// The conditional means of the group intercepts.
    Mean,
    /// Means and everything needed to draw the intercepts jointly.
    Draw,
}
/// Marginal Gaussian law of all residuals with the group intercepts integrated out.
pub(super) struct Marginal {
    pub log_likelihood: f64,
    pub intercept_mean: Vec<f64>,
    /// Conditional variances (independent groups) or the diagonal of the conditional covariance.
    pub intercept_variance: Vec<f64>,
    /// Lower Cholesky factor of the joint conditional covariance (spatial structure only).
    pub joint_factor: Option<Vec<f64>>,
}
impl Marginal {
    /// One conditional intercept per group; a draw when an RNG is supplied, else the means.
    /// Independent groups consume one normal each, in group order.
    pub fn intercepts(&self, rng: Option<&mut antecedent_core::CausalRng>) -> Option<Vec<f64>> {
        let mut rng = rng;
        let draw = if let Some(factor) = &self.joint_factor {
            let n = self.intercept_mean.len();
            let z: Option<Vec<f64>> = rng
                .as_deref_mut()
                .map(|rng| (0..n).map(|_| antecedent_kernels::standard_normal(rng)).collect());
            (0..n)
                .map(|i| {
                    self.intercept_mean[i]
                        + z.as_ref()
                            .map_or(0.0, |z| (0..=i).map(|j| factor[i * n + j] * z[j]).sum::<f64>())
                })
                .collect::<Vec<_>>()
        } else {
            self.intercept_mean
                .iter()
                .zip(&self.intercept_variance)
                .map(|(mean, variance)| {
                    mean + rng.as_deref_mut().map_or(0.0, |rng| {
                        variance.sqrt() * antecedent_kernels::standard_normal(rng)
                    })
                })
                .collect()
        };
        draw.iter().all(|value| value.is_finite()).then_some(draw)
    }
}
/// Sufficient statistics of a stationary AR(1) residual vector against the unit vector.
struct Ar1Moments {
    /// `1' S^-1 1`.
    ones: f64,
    /// `1' S^-1 r`.
    cross: f64,
    /// `r' S^-1 r`.
    quad: f64,
    /// `log |S|`.
    log_det: f64,
}
fn ar1_moments(r: &[f64], rho: f64, sigma: f64) -> Option<Ar1Moments> {
    if r.is_empty()
        || !rho.is_finite()
        || rho.abs() >= 1.0
        || !sigma.is_finite()
        || sigma <= 0.0
        || r.iter().any(|value| !value.is_finite())
    {
        return None;
    }
    let stationary = 1.0 - rho * rho;
    let mut cross = stationary * r[0];
    let mut quad = stationary * r[0] * r[0];
    for pair in r.windows(2) {
        let innovation = pair[1] - rho * pair[0];
        cross += (1.0 - rho) * innovation;
        quad += innovation * innovation;
    }
    let variance = sigma * sigma;
    Some(Ar1Moments {
        ones: (stationary + (r.len() - 1) as f64 * (1.0 - rho).powi(2)) / variance,
        cross: cross / variance,
        quad: quad / variance,
        log_det: r.len() as f64 * variance.ln() - stationary.ln(),
    })
}
/// Spatial correlation `K(range)` of the group intercepts with a tiny diagonal regularization.
pub(super) fn spatial_correlation(data: &PanelData, range: f64) -> Option<Vec<f64>> {
    let spatial = data.family.spatial.as_ref()?;
    if !range.is_finite() || range <= 0.0 {
        return None;
    }
    let groups = data.groups.len();
    let mut kernel: Vec<f64> = spatial
        .distances
        .iter()
        .map(|distance| match spatial.kernel {
            PanelSpatialKernel::Exponential => (-distance / range).exp(),
            PanelSpatialKernel::Gaussian => (-(distance / range).powi(2)).exp(),
        })
        .collect();
    for g in 0..groups {
        kernel[g * groups + g] += SPATIAL_JITTER;
    }
    kernel.iter().all(|value| value.is_finite()).then_some(kernel)
}
/// Marginal law with spatially correlated intercepts `u ~ N(0, tau^2 K)`.
///
/// With `A = blockdiag(S_g^-1)`, `D = diag(1' S_g^-1 1)` and `b_g = 1' S_g^-1 r_g`, the
/// Woodbury identity gives `r'(S + Z V Z')^-1 r = sum_g r_g' S_g^-1 r_g - b'(V^-1 + D)^-1 b`
/// and `log|S + Z V Z'| = sum_g log|S_g| + log|I + D^(1/2) V D^(1/2)|` for `V = tau^2 K`.
fn spatial_marginal(
    data: &CompiledPanel<'_>,
    p: &PanelParameters,
    residuals: &[Vec<f64>],
    detail: Detail,
) -> Option<Marginal> {
    let groups = residuals.len();
    if !p.group_sd.is_finite() || p.group_sd <= 0.0 {
        return None;
    }
    let kernel = spatial_correlation(data, p.family.spatial_range)?;
    let (mut ones, mut cross) = (Vec::with_capacity(groups), Vec::with_capacity(groups));
    let (mut quad, mut log_det, mut rows) = (0.0, 0.0, 0.0);
    for (index, r) in residuals.iter().enumerate() {
        let outcome = data.outcome(index);
        let (moments, analysed) = if outcome.iter().all(|value| value.is_finite()) {
            (ar1_moments(r, p.rho, p.innovation_sd)?, r.len())
        } else {
            let observed = observed_periods(outcome, r)?;
            (ar1_moments_at(&observed, p.rho, p.innovation_sd)?, observed.len())
        };
        ones.push(moments.ones);
        cross.push(moments.cross);
        quad += moments.quad;
        log_det += moments.log_det;
        rows += analysed as f64;
    }
    let tau2 = p.group_sd * p.group_sd;
    let root: Vec<f64> = ones.iter().map(|value| value.sqrt()).collect();
    let mut m = vec![0.0; groups * groups];
    for i in 0..groups {
        for j in 0..groups {
            m[i * groups + j] =
                f64::from(u8::from(i == j)) + root[i] * tau2 * kernel[i * groups + j] * root[j];
        }
    }
    let mut factor = vec![0.0; groups * groups];
    antecedent_kernels::cholesky_spd_into(&m, groups, &mut factor).ok()?;
    let log_det_m = antecedent_stats::chol_log_det(&factor, groups);
    let apply = |matrix: &[f64], vector: &[f64]| -> Vec<f64> {
        (0..groups)
            .map(|i| (0..groups).map(|j| matrix[i * groups + j] * vector[j]).sum::<f64>())
            .collect()
    };
    let vb: Vec<f64> = apply(&kernel, &cross).into_iter().map(|value| tau2 * value).collect();
    let w: Vec<f64> = vb.iter().zip(&root).map(|(v, r)| v * r).collect();
    let y = antecedent_stats::chol_solve(&factor, groups, &w)?;
    // `b'Vb - w'M^-1 w` equals `c'(I - M^-1)c` for `c = D^(-1/2) b` (since
    // `G (I + G)^-1 = I - (I + G)^-1` for `G = D^(1/2) V D^(1/2)`). That form stays bounded
    // by `|c|^2` as the intercept scale grows, where the difference of the two large
    // quadratic forms loses every digit.
    let scaled: Vec<f64> = cross.iter().zip(&root).map(|(b, r)| b / r).collect();
    let solved_scaled = antecedent_stats::chol_solve(&factor, groups, &scaled)?;
    let between: f64 = scaled.iter().map(|c| c * c).sum::<f64>()
        - scaled.iter().zip(&solved_scaled).map(|(a, b)| a * b).sum::<f64>();
    let log_likelihood =
        -0.5 * (rows * std::f64::consts::TAU.ln() + (log_det + log_det_m) + (quad - between));
    if !log_likelihood.is_finite() {
        return None;
    }
    if detail == Detail::Value {
        return Some(Marginal {
            log_likelihood,
            intercept_mean: Vec::new(),
            intercept_variance: Vec::new(),
            joint_factor: None,
        });
    }
    let inner: Vec<f64> =
        cross.iter().zip(root.iter().zip(&y)).map(|(b, (r, y))| b - r * y).collect();
    let intercept_mean: Vec<f64> =
        apply(&kernel, &inner).into_iter().map(|value| tau2 * value).collect();
    // C = V - U' M^-1 U with U = D^(1/2) V, via one forward solve per column of U.
    let mut cov: Vec<f64> = kernel.iter().map(|value| tau2 * value).collect();
    let mut solved = vec![0.0; groups * groups];
    let mut column = vec![0.0; groups];
    let mut z = vec![0.0; groups];
    for j in 0..groups {
        for i in 0..groups {
            column[i] = root[i] * tau2 * kernel[i * groups + j];
        }
        antecedent_kernels::forward_solve_into(&factor, groups, &column, &mut z);
        for i in 0..groups {
            solved[i * groups + j] = z[i];
        }
    }
    for i in 0..groups {
        for k in 0..groups {
            let shrink: f64 =
                (0..groups).map(|m| solved[m * groups + i] * solved[m * groups + k]).sum();
            cov[i * groups + k] -= shrink;
        }
    }
    let intercept_variance: Vec<f64> = (0..groups).map(|i| cov[i * groups + i]).collect();
    if intercept_variance.iter().any(|value| !value.is_finite() || *value <= 0.0) {
        return None;
    }
    let joint_factor = if detail == Detail::Draw {
        let mut lower = vec![0.0; groups * groups];
        antecedent_kernels::cholesky_spd_into(&cov, groups, &mut lower).ok()?;
        Some(lower)
    } else {
        None
    };
    Some(Marginal { log_likelihood, intercept_mean, intercept_variance, joint_factor })
}
/// Marginal law of all residuals, independent per group or spatially correlated.
pub(super) fn marginal(
    data: &CompiledPanel<'_>,
    p: &PanelParameters,
    residuals: &[Vec<f64>],
    detail: Detail,
) -> Option<Marginal> {
    if data.family.spatial.is_some() {
        return spatial_marginal(data, p, residuals, detail);
    }
    let mut total = 0.0;
    // A value-only evaluation never reads the conditional intercepts: do not collect them.
    let keep = detail != Detail::Value;
    let capacity = if keep { residuals.len() } else { 0 };
    let mut intercept_mean = Vec::with_capacity(capacity);
    let mut intercept_variance = Vec::with_capacity(capacity);
    for (index, (group, r)) in data.groups.iter().zip(residuals).enumerate() {
        let value = outcome_marginal(data.outcome(index), r, p.rho, p.innovation_sd, p.group_sd)?;
        total += group.weight * value.log_likelihood;
        if keep {
            intercept_mean.push(value.intercept_mean);
            intercept_variance.push(value.intercept_variance);
        }
    }
    total.is_finite().then_some(Marginal {
        log_likelihood: total,
        intercept_mean,
        intercept_variance,
        joint_factor: None,
    })
}
/// Whether the group intercepts are independent or jointly drawn from a spatial prior.
pub(super) enum SpatialPriorIntercepts {
    Independent,
    Correlated(Vec<f64>),
}

/// Group intercepts drawn from the fitted prior; a factorization failure refuses the draw.
pub(super) fn spatial_prior_intercepts(
    data: &PanelData,
    p: &PanelParameters,
    rng: &mut antecedent_core::CausalRng,
) -> Option<SpatialPriorIntercepts> {
    if data.family.spatial.is_none() {
        return Some(SpatialPriorIntercepts::Independent);
    }
    let groups = data.groups.len();
    let kernel = spatial_correlation(data, p.family.spatial_range)?;
    let mut lower = vec![0.0; groups * groups];
    antecedent_kernels::cholesky_spd_into(&kernel, groups, &mut lower).ok()?;
    let z: Vec<f64> = (0..groups).map(|_| antecedent_kernels::standard_normal(rng)).collect();
    Some(SpatialPriorIntercepts::Correlated(
        (0..groups)
            .map(|i| p.group_sd * (0..=i).map(|j| lower[i * groups + j] * z[j]).sum::<f64>())
            .collect(),
    ))
}
pub(super) fn log_likelihood(data: &CompiledPanel<'_>, x: &[f64]) -> f64 {
    let Some(p) = decode(data, x) else {
        return f64::NEG_INFINITY;
    };
    let residuals = residuals(data, &p);
    marginal(data, &p, &residuals, Detail::Value)
        .map_or(f64::NEG_INFINITY, |value| value.log_likelihood)
}
#[cfg(test)]
mod product_feature_tests {
    use super::{product_columns, product_value};
    use antecedent_core::{ValueExpr, VariableId};
    use antecedent_model::program::NodeProgram;

    /// The direct product returns the compiled program's bits, including every refusal.
    #[test]
    fn product_feature_matches_the_column_program_bit_for_bit() {
        let axis: Vec<_> = (0..3).map(VariableId::from_raw).collect();
        let expression = ValueExpr::var(axis[2]).mul(ValueExpr::var(axis[0]));
        let columns = product_columns(&axis, &expression).expect("a channel product");
        assert_eq!(columns, (2, 0));
        let program = NodeProgram::compile_columns(&axis, &expression).expect("compiles");
        let rows: [&[f64]; 8] = [
            &[0.25, 0.5, 0.75],
            &[0.0, 1.0, -0.0],
            &[1e-300, 0.3, 1e-300],
            &[1e200, 0.0, 1e200],
            &[f64::NAN, 0.5, 0.5],
            &[0.5, f64::INFINITY, 0.5],
            &[0.1, 0.2],
            &[0.1 + 0.2, 0.7, 1.0 / 3.0],
        ];
        for row in rows {
            let direct = product_value(axis.len(), row, columns);
            let executed = program.evaluate_columns(row, None).unwrap_or(f64::NAN);
            assert_eq!(direct.to_bits(), executed.to_bits(), "row {row:?}");
        }
    }

    /// Only a product of two channel variables takes the direct path.
    #[test]
    fn other_features_keep_the_column_program() {
        let axis: Vec<_> = (0..2).map(VariableId::from_raw).collect();
        let outside = ValueExpr::var(axis[0]).mul(ValueExpr::var(VariableId::from_raw(7)));
        let sum = ValueExpr::var(axis[0]).add(ValueExpr::var(axis[1]));
        assert_eq!(product_columns(&axis, &outside), None);
        assert_eq!(product_columns(&axis, &sum), None);
    }
}
