// SPDX-License-Identifier: MIT OR Apache-2.0
//! Gaussian AR(1) precision plus rank-one random-intercept integration.
use super::{
    PanelChannelParameters, PanelData, PanelError, PanelGroup, PanelInteractionParameter,
    PanelParameters, PanelProgram, PanelScenario, invalid,
};

/// Bound once at a numerical execution boundary; never constructs a causal model or builder.
pub(super) struct CompiledPanel<'a> {
    pub source: &'a PanelData,
    features: Vec<antecedent_model::program::NodeProgram>,
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
        Ok(Self { source, features })
    }
    pub fn workspace_bytes(&self) -> Result<usize, PanelError> {
        let feature = self.features.iter().try_fold(0_usize, |peak, program| {
            let bytes =
                program.column_workspace_bytes().map_err(|error| invalid(&error.to_string()))?;
            Ok::<_, PanelError>(peak.max(
                usize::try_from(bytes).map_err(|_| invalid("feature workspace size overflow"))?,
            ))
        })?;
        self.channels
            .len()
            .checked_mul(16)
            .and_then(|n| n.checked_add(feature))
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
    linear_count(data) + if data.lag_order == 2 { 2 * data.channels.len() + 4 } else { 3 }
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
            x.extend([(w[0] / w[2]).ln(), (w[1] / w[2]).ln()]);
        }
    }
    x.push(p.half_saturation.ln());
    if data.lag_order == 2 {
        x.push(p.rho.atanh());
    }
    x.extend([p.innovation_sd.ln(), p.group_sd.ln()]);
    Ok(x)
}
fn simplex(x: &[f64]) -> [f64; 3] {
    let max = x[0].max(x[1]).max(0.0);
    let u = [(x[0] - max).exp(), (x[1] - max).exp(), (-max).exp()];
    let sum = u.iter().sum::<f64>();
    u.map(|value| value / sum)
}
pub(super) fn decode(data: &PanelData, x: &[f64]) -> Option<PanelParameters> {
    if x.len() != parameter_count(data) || x.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let d = data.channels.len();
    let i = data.interactions.len();
    let base = linear_count(data);
    let tail = base + if data.lag_order == 2 { 2 * d } else { 0 };
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
                lag: if data.lag_order == 2 {
                    simplex(&x[base + 2 * index..base + 2 * index + 2])
                } else {
                    [1.0, 0.0, 0.0]
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
        half_saturation: x[tail].exp(),
        rho: if data.lag_order == 2 { x[tail + 1].tanh() } else { 0.0 },
        innovation_sd: x[tail + if data.lag_order == 2 { 2 } else { 1 }].exp(),
        group_sd: x[tail + if data.lag_order == 2 { 3 } else { 2 }].exp(),
    };
    validate_parameters(data, &p).ok().map(|()| p)
}
pub(super) fn initial(data: &PanelData) -> PanelParameters {
    let n = data.groups.iter().map(|g| g.outcome.len()).sum::<usize>() as f64;
    let mean = data.groups.iter().flat_map(|g| &g.outcome).sum::<f64>() / n;
    let variance =
        data.groups.iter().flat_map(|g| &g.outcome).map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let scale = variance.sqrt().max(0.1);
    let lag = if data.lag_order == 2 { [0.6, 0.3, 0.1] } else { [1.0, 0.0, 0.0] };
    let mut exposures: Vec<_> = data
        .groups
        .iter()
        .flat_map(|g| g.channels.iter().flatten())
        .copied()
        .filter(|v| *v > 0.0)
        .collect();
    exposures.sort_by(f64::total_cmp);
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
        half_saturation: exposures.get(exposures.len() / 2).copied().unwrap_or(1.0),
        rho: if data.lag_order == 2 { 0.1 } else { 0.0 },
        innovation_sd: scale * 0.8,
        group_sd: scale * 0.3,
    }
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
pub(super) fn response(
    data: &CompiledPanel<'_>,
    p: &PanelParameters,
    exposures: &[f64],
    z: f64,
    season: usize,
) -> f64 {
    let saturated: Vec<_> = exposures.iter().map(|x| x / (x + p.half_saturation)).collect();
    let main = p.channels.iter().zip(&saturated).map(|(c, x)| c.coefficient * x).sum::<f64>();
    let mut values = Vec::new();
    let mut scratch = Vec::new();
    let products = data
        .features
        .iter()
        .zip(&p.interactions)
        .map(|(program, param)| {
            // The common production expression executor validates every numerical result.
            param.coefficient
                * program
                    .evaluate_columns_with_scratch(&saturated, &mut values, &mut scratch, None)
                    .unwrap_or(f64::NAN)
        })
        .sum::<f64>();
    p.intercept + main + products + p.confounder * z + p.season_effects[season]
}
pub(super) fn observed_response(
    data: &CompiledPanel<'_>,
    p: &PanelParameters,
    group: &PanelGroup,
    time: usize,
) -> f64 {
    let exposures: Vec<_> = group
        .channels
        .iter()
        .zip(&group.history)
        .zip(&p.channels)
        .map(|((values, history), channel)| exposure(values, *history, channel.lag, time))
        .collect();
    response(data, p, &exposures, group.confounder[time], group.season[time])
}
pub(super) fn program_response(
    data: &CompiledPanel<'_>,
    p: &PanelParameters,
    program: &PanelProgram,
    scenario: &PanelScenario,
    time: usize,
) -> f64 {
    let exposures: Vec<_> = p
        .channels
        .iter()
        .map(|channel| {
            let values = &program
                .channels
                .iter()
                .find(|c| c.id == channel.id)
                .expect("validated program channel")
                .values;
            let history = scenario
                .history
                .iter()
                .find(|c| c.id == channel.id)
                .expect("validated scenario history")
                .values;
            exposure(values, history, channel.lag, time)
        })
        .collect();
    response(data, p, &exposures, scenario.confounder[time], scenario.season[time])
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
        .ok_or_else(|| invalid("parameter storage overflow"))
}

pub(super) struct GaussianGroup {
    pub log_likelihood: f64,
    pub intercept_mean: f64,
    pub intercept_variance: f64,
}
pub(super) fn group_likelihood(
    data: &CompiledPanel<'_>,
    group: usize,
    p: &PanelParameters,
) -> Option<GaussianGroup> {
    let group = &data.groups[group];
    let residuals: Vec<_> = (0..group.outcome.len())
        .map(|time| group.outcome[time] - observed_response(data, p, group, time))
        .collect();
    ar1_random_intercept(&residuals, p.rho, p.innovation_sd, p.group_sd)
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
    let conditional_variance = tau2 / (1.0 + tau2 * one_one);
    let intercept_mean = conditional_variance * one_r;
    // Completing the square avoids subtracting two large nearly equal quadratic
    // forms when an intercept explains most of a group's residual mean.
    let mut centered_quad = stationary * (r[0] - intercept_mean).powi(2);
    for pair in r.windows(2) {
        centered_quad += ((pair[1] - intercept_mean) - rho * (pair[0] - intercept_mean)).powi(2);
    }
    let quad = centered_quad / variance + (intercept_mean / tau).powi(2);
    let log_det = r.len() as f64 * variance.ln() - stationary.ln() + (tau2 * one_one).ln_1p();
    let log_likelihood = -0.5 * (r.len() as f64 * std::f64::consts::TAU.ln() + log_det + quad);
    if !log_likelihood.is_finite()
        || !conditional_variance.is_finite()
        || conditional_variance <= 0.0
    {
        return None;
    }
    Some(GaussianGroup { log_likelihood, intercept_mean, intercept_variance: conditional_variance })
}
pub(super) fn log_likelihood(data: &CompiledPanel<'_>, x: &[f64]) -> f64 {
    let Some(p) = decode(data, x) else {
        return f64::NEG_INFINITY;
    };
    let mut total = 0.0;
    for group in 0..data.groups.len() {
        let Some(value) = group_likelihood(data, group, &p) else {
            return f64::NEG_INFINITY;
        };
        total += value.log_likelihood;
    }
    if total.is_finite() { total } else { f64::NEG_INFINITY }
}
