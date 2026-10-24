//! Explicit model-conditional piecewise/smooth attempts on the same panel likelihood.
//!
//! Candidate selection is part of the fitted procedure: frequentist replicates rerun it;
//! Bayesian draws retain discrete model choice jointly with path and mechanism parameters.
//! Neither fitting a change nor a prior makes that change causally identified. Explanatory
//! contrasts require independently checked reference-world support in the compiler.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
use super::{
    PanelData, PanelDraw, PanelError, PanelFitOptions, PanelInference, PanelParameters,
    PanelProgram, PanelResponse, check_cancelled, invalid,
};
use super::{inference, likelihood};
use antecedent_core::{ExecutionContext, SemanticDigest};
use antecedent_kernels::standard_normal;
use antecedent_prob::{
    all_chains_moved, mcmc_summary,
    metropolis::{MetropolisOptions, sample_metropolis_dispersed},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// The stationary residual law and group-intercept hierarchy do not silently vary.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PanelMeanParameter {
    /// Baseline outcome intercept.
    Intercept,
    /// Effectiveness of one declared channel.
    Channel {
        /// Stable channel identity.
        id: String,
    },
    /// Coefficient of one declared programmable feature.
    Interaction {
        /// Stable feature identity.
        id: String,
    },
    /// Additive observable-confounder coefficient.
    Confounder,
    /// Seasonal effect, with the reference level zero excluded.
    Season {
        /// Nonreference dummy level.
        level: usize,
    },
    /// Additive logratio of one channel's lag weight to lag-two.
    Lag {
        /// Stable channel identity.
        channel: String,
        /// Current or lag-one numerator coordinate, 0 or 1.
        coordinate: usize,
    },
    /// Log of the shared positive half-saturation.
    LogHalfSaturation,
}

/// Every choice is prespecified; data-dependent selection remains in the procedure.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemporalPanelSpec {
    /// Mean-mechanism coordinates that can vary jointly.
    pub varying: Vec<PanelMeanParameter>,
    /// Candidate first periods of the changed segment; no-change is always included.
    pub change_points: Vec<usize>,
    /// Minimum observed periods on either side in every group.
    pub minimum_segment: usize,
    /// Minimum best-versus-next BIC gap before retaining a localized piecewise change.
    pub weak_change_gap: f64,
    /// Piecewise-linear smooth-path knots, including zero and the last observed time.
    pub smooth_knots: Vec<usize>,
    /// Positive prespecified difference-penalty grid; uncertainty includes its selection.
    pub smoothing: Vec<f64>,
    /// Proper Gaussian path-offset prior scale for Bayesian model/path uncertainty.
    pub path_prior_sd: f64,
}

/// A candidate is a statistical specification, not an identified explanation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum TemporalModelChoice {
    /// No time-varying parameter.
    NoChange,
    /// Constant parameters before/after this declared change candidate.
    Piecewise {
        /// First changed period.
        at: usize,
    },
    /// Regularized continuous parameter path on declared knots.
    Smooth {
        /// Selected difference penalty / prior precision multiplier.
        penalty: f64,
    },
}

/// Complete selected realization shared across programs/reference worlds.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemporalPanelDraw {
    /// Baseline global parameters and same conditional group effects for every time/world.
    pub baseline: PanelDraw,
    /// Complete mean-mechanism parameter at every declared response time.
    pub parameters: Vec<PanelParameters>,
    /// Actual selected specification for this draw/replicate.
    pub choice: TemporalModelChoice,
}

/// A numerical candidate attempt, including failed local fits.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemporalPanelAttempt {
    /// Candidate attempted.
    pub choice: TemporalModelChoice,
    /// Log likelihood excluding regularization, if finite.
    pub log_likelihood: Option<f64>,
    /// Declared penalized BIC selection score (lower is preferred), if finite.
    pub criterion: Option<f64>,
    /// Local score tolerance was actually met.
    pub converged: bool,
    /// Numerical failure remains explicit.
    pub failure: Option<String>,
}

/// Complete attempt/selection uncertainty; not a claim of change-point identification.
#[derive(Clone, Debug)]
pub struct TemporalPanelFit {
    /// Retained point specification and mechanism.
    pub point: TemporalPanelDraw,
    /// Failed fits/replicates keep their original slots.
    pub draws: Vec<Option<TemporalPanelDraw>>,
    /// Initial no-change/piecewise attempts, followed by smooth attempts when needed.
    pub attempts: Vec<TemporalPanelAttempt>,
    /// Whether weak change localization started the required smooth attempt.
    pub smooth_fallback: bool,
    /// Choice counts from retained whole-procedure replicates/posterior draws.
    pub selection_counts: Vec<(TemporalModelChoice, usize)>,
    /// Posterior max R-hat, bulk ESS and tail ESS, when sampled.
    pub posterior_diagnostics: Option<[f64; 3]>,
    /// Actual movement and declared posterior thresholds passed.
    pub posterior_converged: Option<bool>,
    /// Which global/path/change/smoothing/selection uncertainty was retained.
    pub uncertainty: String,
    /// Always model conditional; an engine cannot identify an attribution.
    pub claim: String,
    /// Calibration remains unmeasured.
    pub calibration: String,
    /// Frozen data/history/group dependency.
    pub source_identity: SemanticDigest,
    retention: Arc<antecedent_core::execution::AdmissionLease>,
    retained_volume: u64,
}

/// The response change alone does not distinguish these explanations.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PanelExplanation {
    /// Changed assignments under the same mechanism/history law.
    Exposure,
    /// Changed saturation parameters under fixed reference assignments.
    Saturation,
    /// Changed population/group weighting.
    Composition,
    /// Changed mean-mechanism parameters under matched reference worlds.
    Mechanism,
}

/// Required next attempt instead of a unique unsupported causal decomposition.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoarserPanelContrast {
    /// Explanation whose finer attribution is unsupported.
    pub explanation: PanelExplanation,
    /// Scientific obligation the compiler must attempt next.
    pub next_attempt: String,
}

fn coordinates(data: &PanelData, spec: &TemporalPanelSpec) -> Result<Vec<usize>, PanelError> {
    let base = likelihood::linear_count(data);
    let mut result = Vec::new();
    for coordinate in &spec.varying {
        let index = match coordinate {
            PanelMeanParameter::Intercept => 0,
            PanelMeanParameter::Channel { id } => {
                1 + data
                    .channels
                    .iter()
                    .position(|channel| &channel.id == id)
                    .ok_or_else(|| invalid("unknown varying channel"))?
            }
            PanelMeanParameter::Interaction { id } => {
                1 + data.channels.len()
                    + data
                        .interactions
                        .iter()
                        .position(|term| &term.id == id)
                        .ok_or_else(|| invalid("unknown varying programmable feature"))?
            }
            PanelMeanParameter::Confounder => 1 + data.channels.len() + data.interactions.len(),
            PanelMeanParameter::Season { level } if *level > 0 && *level < data.seasons => {
                1 + data.channels.len() + data.interactions.len() + level
            }
            PanelMeanParameter::Lag { channel, coordinate }
                if data.lag_order == 2 && *coordinate < 2 =>
            {
                base + 2 * data
                    .channels
                    .iter()
                    .position(|entry| &entry.id == channel)
                    .ok_or_else(|| invalid("unknown varying lag channel"))?
                    + coordinate
            }
            PanelMeanParameter::LogHalfSaturation => {
                base + if data.lag_order == 2 { 2 * data.channels.len() } else { 0 }
            }
            _ => {
                return Err(invalid(
                    "time-varying coordinate is outside the declared mean-mechanism family",
                ));
            }
        };
        if result.contains(&index) {
            return Err(invalid("duplicate varying mean-mechanism coordinate"));
        }
        result.push(index);
    }
    Ok(result)
}

/// Validate the complete candidate/fallback specification without fitting.
///
/// # Errors
/// Static questions, empty/duplicate paths, unsupported dependence changes and malformed candidate windows are refused.
pub fn validate_temporal_spec(
    data: &PanelData,
    spec: &TemporalPanelSpec,
) -> Result<(), PanelError> {
    data.validate()?;
    let coordinates = coordinates(data, spec)?;
    let min = data.groups.iter().map(|group| group.outcome.len()).min().unwrap_or(0);
    let max = data.groups.iter().map(|group| group.outcome.len()).max().unwrap_or(0);
    if data.lag_order != 2
        || coordinates.is_empty()
        || spec.change_points.len() > 64
        || spec.minimum_segment < 3
        || !spec.weak_change_gap.is_finite()
        || spec.weak_change_gap < 0.0
        || !spec.path_prior_sd.is_finite()
        || spec.path_prior_sd <= 0.0
        || spec.change_points.windows(2).any(|pair| pair[0] >= pair[1])
        || spec.change_points.iter().any(|&point| {
            point < spec.minimum_segment
                || point.checked_add(spec.minimum_segment).is_none_or(|end| end > min)
        })
        || spec.smooth_knots.len() < 2
        || spec.smooth_knots.len() > 32
        || spec.smooth_knots[0] != 0
        || spec.smooth_knots.last().copied() != max.checked_sub(1)
        || spec.smooth_knots.windows(2).any(|pair| pair[0] >= pair[1])
        || spec.smoothing.is_empty()
        || spec.smoothing.len() > 16
        || spec.smoothing.iter().any(|value| !value.is_finite() || *value <= 0.0)
        || spec.smoothing.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(invalid(
            "temporal candidate windows, smooth knots, penalty grid or proper path prior",
        ));
    }
    Ok(())
}

impl TemporalPanelSpec {
    /// Canonical dependency of every change, smoothing, selection and path-prior choice.
    ///
    /// # Errors
    /// Refuses a specification that cannot execute on this source panel.
    pub fn identity(&self, data: &PanelData) -> Result<SemanticDigest, PanelError> {
        validate_temporal_spec(data, self)?;
        let mut writer = antecedent_core::CanonicalWriter::new(
            "antecedent.estimate.temporal_joint_panel.spec.v2",
        );
        writer.word(self.varying.len() as u64);
        for coordinate in &self.varying {
            match coordinate {
                PanelMeanParameter::Intercept => {
                    writer.word(0);
                }
                PanelMeanParameter::Channel { id } => {
                    writer.word(1).text(id);
                }
                PanelMeanParameter::Interaction { id } => {
                    writer.word(2).text(id);
                }
                PanelMeanParameter::Confounder => {
                    writer.word(3);
                }
                PanelMeanParameter::Season { level } => {
                    writer.word(4).word(*level as u64);
                }
                PanelMeanParameter::Lag { channel, coordinate } => {
                    writer.word(5).text(channel).word(*coordinate as u64);
                }
                PanelMeanParameter::LogHalfSaturation => {
                    writer.word(8);
                }
            }
        }
        writer.word(self.change_points.len() as u64);
        for &point in &self.change_points {
            writer.word(point as u64);
        }
        writer.word(self.minimum_segment as u64).word(self.weak_change_gap.to_bits());
        writer.word(self.smooth_knots.len() as u64);
        for &knot in &self.smooth_knots {
            writer.word(knot as u64);
        }
        writer.word(self.smoothing.len() as u64);
        for &penalty in &self.smoothing {
            writer.word(penalty.to_bits());
        }
        writer.word(self.path_prior_sd.to_bits());
        Ok(SemanticDigest::from_bytes(writer.finish()))
    }
}

fn offsets(spec: &TemporalPanelSpec, choice: &TemporalModelChoice) -> usize {
    match choice {
        TemporalModelChoice::NoChange => 0,
        TemporalModelChoice::Piecewise { .. } => spec.varying.len(),
        TemporalModelChoice::Smooth { .. } => spec.varying.len() * (spec.smooth_knots.len() - 1),
    }
}
fn path_value(
    spec: &TemporalPanelSpec,
    choice: &TemporalModelChoice,
    delta: &[f64],
    coordinate: usize,
    time: usize,
) -> f64 {
    match choice {
        TemporalModelChoice::NoChange => 0.0,
        TemporalModelChoice::Piecewise { at } => {
            if time < *at {
                0.0
            } else {
                delta[coordinate]
            }
        }
        TemporalModelChoice::Smooth { .. } => {
            let knots = &spec.smooth_knots;
            let right = knots.partition_point(|&point| point < time).max(1).min(knots.len() - 1);
            let left = right - 1;
            let fraction = (time - knots[left]) as f64 / (knots[right] - knots[left]) as f64;
            let at = |knot| {
                if knot == 0 { 0.0 } else { delta[(knot - 1) * spec.varying.len() + coordinate] }
            };
            at(left) * (1.0 - fraction) + at(right) * fraction
        }
    }
}
fn parameters_at(
    data: &PanelData,
    spec: &TemporalPanelSpec,
    choice: &TemporalModelChoice,
    values: &[f64],
    time: usize,
) -> Option<PanelParameters> {
    let p = likelihood::parameter_count(data);
    let mut current = values.get(..p)?.to_vec();
    let selected = coordinates(data, spec).ok()?;
    let delta = values.get(p..p + offsets(spec, choice))?;
    for (coordinate, &index) in selected.iter().enumerate() {
        current[index] += path_value(spec, choice, delta, coordinate, time);
    }
    likelihood::decode(data, &current)
}
fn conditional_groups(
    data: &likelihood::CompiledPanel<'_>,
    spec: &TemporalPanelSpec,
    choice: &TemporalModelChoice,
    values: &[f64],
) -> Option<Vec<likelihood::GaussianGroup>> {
    let base = likelihood::decode(data, values.get(..likelihood::parameter_count(data))?)?;
    data.groups
        .iter()
        .map(|group| {
            let residuals: Vec<_> = (0..group.outcome.len())
                .map(|time| {
                    let p = parameters_at(data, spec, choice, values, time)?;
                    Some(group.outcome[time] - likelihood::observed_response(data, &p, group, time))
                })
                .collect::<Option<Vec<_>>>()?;
            likelihood::ar1_random_intercept(
                &residuals,
                base.rho,
                base.innovation_sd,
                base.group_sd,
            )
        })
        .collect()
}
fn objective(
    data: &likelihood::CompiledPanel<'_>,
    spec: &TemporalPanelSpec,
    choice: &TemporalModelChoice,
    values: &[f64],
) -> f64 {
    let Some(groups) = conditional_groups(data, spec, choice, values) else {
        return f64::NEG_INFINITY;
    };
    groups.iter().map(|group| group.log_likelihood).sum()
}
fn penalty(spec: &TemporalPanelSpec, choice: &TemporalModelChoice, delta: &[f64]) -> f64 {
    let TemporalModelChoice::Smooth { penalty } = choice else {
        return 0.0;
    };
    let mut result = 0.0;
    for coordinate in 0..spec.varying.len() {
        let mut previous = 0.0;
        for knot in 1..spec.smooth_knots.len() {
            let next = delta[(knot - 1) * spec.varying.len() + coordinate];
            result += penalty * (next - previous).powi(2);
            previous = next;
        }
    }
    0.5 * result
}
struct Selected {
    values: Vec<f64>,
    choice: TemporalModelChoice,
    attempts: Vec<TemporalPanelAttempt>,
    smooth: bool,
}
fn fit_candidates(
    data: &likelihood::CompiledPanel<'_>,
    spec: &TemporalPanelSpec,
    options: &PanelFitOptions,
    ctx: &ExecutionContext,
) -> Result<Selected, PanelError> {
    let p = likelihood::parameter_count(data);
    let initial = options.initial.clone().unwrap_or_else(|| likelihood::initial(data));
    let start = likelihood::encode(data, &initial)?;
    let n = data.groups.iter().map(|group| group.outcome.len()).sum::<usize>() as f64;
    let choices = std::iter::once(TemporalModelChoice::NoChange)
        .chain(spec.change_points.iter().map(|&at| TemporalModelChoice::Piecewise { at }));
    let mut attempts = Vec::new();
    let mut successes = Vec::new();
    let try_fit = |choice: TemporalModelChoice,
                   attempts: &mut Vec<TemporalPanelAttempt>|
     -> Result<Option<(f64, Vec<f64>, TemporalModelChoice)>, PanelError> {
        check_cancelled(ctx)?;
        let mut initial = start.clone();
        initial.resize(p + offsets(spec, &choice), 0.0);
        let function = |values: &[f64]| {
            objective(data, spec, &choice, values) - penalty(spec, &choice, &values[p..])
        };
        match inference::optimize_target(initial, options.optimizer, ctx, &function) {
            Ok(fit) => {
                let ll = objective(data, spec, &choice, &fit.values);
                let criterion = -2.0 * fit.likelihood + (fit.values.len() as f64) * n.ln();
                attempts.push(TemporalPanelAttempt {
                    choice: choice.clone(),
                    log_likelihood: Some(ll),
                    criterion: Some(criterion),
                    converged: fit.converged,
                    failure: (!fit.converged)
                        .then(|| format!("local score tolerance not met: {}", fit.score_norm)),
                });
                Ok(fit.converged.then_some((criterion, fit.values, choice)))
            }
            Err(PanelError::Cancelled) => Err(PanelError::Cancelled),
            Err(error) => {
                attempts.push(TemporalPanelAttempt {
                    choice,
                    log_likelihood: None,
                    criterion: None,
                    converged: false,
                    failure: Some(error.to_string()),
                });
                Ok(None)
            }
        }
    };
    for choice in choices {
        if let Some(fit) = try_fit(choice, &mut attempts)? {
            successes.push(fit);
        }
    }
    successes.sort_by(|a, b| a.0.total_cmp(&b.0));
    let weak = successes.len() < 2
        || successes.first().is_none_or(|fit| matches!(fit.2, TemporalModelChoice::NoChange))
        || successes.get(1).is_some_and(|second| second.0 - successes[0].0 <= spec.weak_change_gap);
    if weak {
        for &penalty in &spec.smoothing {
            if let Some(fit) = try_fit(TemporalModelChoice::Smooth { penalty }, &mut attempts)? {
                successes.push(fit);
            }
        }
        // After a weak change attempt, the smooth family is an actual next attempt;
        // retain its best converged fit, not a renamed piecewise/no-change result.
        successes.retain(|fit| matches!(fit.2, TemporalModelChoice::Smooth { .. }));
        successes.sort_by(|a, b| a.0.total_cmp(&b.0));
    }
    let Some((_, values, choice)) = successes.into_iter().next() else {
        return Err(PanelError::Numerical(format!(
            "no temporal candidate met numerical tolerance; {} attempted",
            attempts.len()
        )));
    };
    Ok(Selected { values, choice, attempts, smooth: weak })
}

fn realization(
    data: &likelihood::CompiledPanel<'_>,
    spec: &TemporalPanelSpec,
    choice: &TemporalModelChoice,
    values: &[f64],
    rng: Option<&mut antecedent_core::CausalRng>,
) -> Option<TemporalPanelDraw> {
    let groups = conditional_groups(data, spec, choice, values)?;
    let mut rng = rng;
    let group_intercepts = groups
        .iter()
        .map(|group| {
            group.intercept_mean
                + rng
                    .as_deref_mut()
                    .map_or(0.0, |rng| group.intercept_variance.sqrt() * standard_normal(rng))
        })
        .collect();
    let baseline = PanelDraw {
        parameters: likelihood::decode(data, &values[..likelihood::parameter_count(data)])?,
        group_intercepts,
    };
    let horizon = data.groups.iter().map(|group| group.outcome.len()).max()?;
    let parameters = (0..horizon)
        .map(|time| parameters_at(data, spec, choice, values, time))
        .collect::<Option<Vec<_>>>()?;
    Some(TemporalPanelDraw { baseline, parameters, choice: choice.clone() })
}

fn bootstrap_data(
    data: &likelihood::CompiledPanel<'_>,
    draw: &TemporalPanelDraw,
    ctx: &ExecutionContext,
    replicate: usize,
) -> Result<PanelData, PanelError> {
    let mut simulated = data.source.clone();
    let mut rng =
        ctx.rng.stream(antecedent_core::derive_stream(0x5830_3342_4f4f_5453, replicate as u64));
    let base = &draw.baseline.parameters;
    for group in &mut simulated.groups {
        check_cancelled(ctx)?;
        let intercept = base.group_sd * standard_normal(&mut rng);
        let mut error =
            base.innovation_sd / (1.0 - base.rho.powi(2)).sqrt() * standard_normal(&mut rng);
        for time in 0..group.outcome.len() {
            let p = &draw.parameters[time];
            error = base.rho * error + base.innovation_sd * standard_normal(&mut rng);
            group.outcome[time] =
                likelihood::observed_response(data, p, group, time) + intercept + error;
        }
    }
    Ok(simulated)
}

fn candidate_choices(spec: &TemporalPanelSpec) -> Vec<TemporalModelChoice> {
    std::iter::once(TemporalModelChoice::NoChange)
        .chain(spec.change_points.iter().map(|&at| TemporalModelChoice::Piecewise { at }))
        .chain(spec.smoothing.iter().map(|&penalty| TemporalModelChoice::Smooth { penalty }))
        .collect()
}
fn choice_index(auxiliary: f64, n: usize) -> Option<usize> {
    // Keep the discrete model choice explicit while reusing the existing symmetric
    // random-walk sampler: a uniform auxiliary on (0,1) gives equal prior bin mass.
    if !auxiliary.is_finite() || auxiliary <= 0.0 || auxiliary >= 1.0 {
        return None;
    }
    (0..n).find(|&index| auxiliary < (index + 1) as f64 / n as f64)
}
fn path_log_prior(spec: &TemporalPanelSpec, choice: &TemporalModelChoice, delta: &[f64]) -> f64 {
    let sigma2 = spec.path_prior_sd.powi(2);
    match choice {
        TemporalModelChoice::Smooth { penalty } => {
            let mut total = 0.5 * delta.len() as f64 * penalty.ln();
            for coordinate in 0..spec.varying.len() {
                let mut previous = 0.0;
                for knot in 1..spec.smooth_knots.len() {
                    let next = delta[(knot - 1) * spec.varying.len() + coordinate];
                    total -= 0.5 * penalty * (next - previous).powi(2) / sigma2;
                    previous = next;
                }
            }
            total
        }
        _ => -0.5 * delta.iter().map(|value| value * value / sigma2).sum::<f64>(),
    }
}

/// Fit both named specification attempts and retain whole-procedure uncertainty.
///
/// Bayesian priors cover the prespecified union of no-change, piecewise and smooth models
/// before observing data. The numerical point's weak-change rule does not truncate that prior.
/// Frequentist bootstrap replicates repeat the point's complete selection/fallback procedure.
///
/// # Errors
/// Refuses malformed specifications, numerical failure of every required attempt, missing
/// resource admission or cancellation. No causal-change identification is inferred here.
pub fn fit_temporal_panel(
    data: &PanelData,
    spec: &TemporalPanelSpec,
    options: &PanelFitOptions,
    ctx: &ExecutionContext,
) -> Result<TemporalPanelFit, PanelError> {
    ctx.try_map_indexed(1, |_, worker| fit_temporal_inner(data, spec, options, worker))
        .map_err(super::map_execution_error)?
        .pop()
        .ok_or_else(|| invalid("missing admitted temporal worker"))
}
#[allow(
    clippy::too_many_lines,
    reason = "retains the actual attempted candidates, realization-specific uncertainty and resource owner together"
)]
fn fit_temporal_inner(
    data: &PanelData,
    spec: &TemporalPanelSpec,
    options: &PanelFitOptions,
    ctx: &ExecutionContext,
) -> Result<TemporalPanelFit, PanelError> {
    validate_temporal_spec(data, spec)?;
    super::validate_fit_options(data, options)?;
    check_cancelled(ctx)?;
    let compiled = likelihood::CompiledPanel::new(data)?;
    let data = &compiled;
    let p = likelihood::parameter_count(data);
    let max_time = data.groups.iter().map(|group| group.outcome.len()).max().unwrap_or(0);
    let rows = data.groups.iter().map(|group| group.outcome.len()).sum::<usize>();
    let path_parameters = spec.varying.len() * (spec.smooth_knots.len() - 1);
    let slots = match &options.inference {
        PanelInference::Frequentist { draws } | PanelInference::FrequentistBootstrap { draws } => {
            *draws
        }
        PanelInference::Bayesian { sampler, .. } => sampler
            .chains
            .checked_mul(sampler.draws)
            .ok_or_else(|| invalid("temporal posterior count overflow"))?,
    };
    if p + path_parameters + 1 > 1024 {
        return Err(invalid("time-varying target exceeds bounded sampler coordinate limit"));
    }
    let parameter_bytes = likelihood::parameter_bytes(data)?;
    let one_draw = max_time
        .checked_mul(parameter_bytes)
        .and_then(|n| {
            n.checked_add(
                parameter_bytes + data.groups.len() * 8 + std::mem::size_of::<TemporalPanelDraw>(),
            )
        })
        .ok_or_else(|| invalid("temporal retained realization overflow"))?;
    let retained = slots
        .checked_add(1)
        .and_then(|count| count.checked_mul(one_draw))
        .and_then(|count| count.checked_add(4096))
        .ok_or_else(|| invalid("temporal retained volume overflow"))?;
    let scratch = rows
        .checked_mul(128)
        .and_then(|value| value.checked_add(data.workspace_bytes().ok()?))
        .and_then(|value| value.checked_add(data.plan_bytes().ok()?))
        .and_then(|count| count.checked_add((p + path_parameters + 1) * (slots + 256) * 32))
        .ok_or_else(|| invalid("temporal scratch volume overflow"))?;
    let retained =
        u64::try_from(retained).map_err(|_| invalid("temporal retained volume overflow"))?;
    super::check_memory(
        ctx,
        retained
            .checked_add(
                u64::try_from(scratch).map_err(|_| invalid("temporal scratch volume overflow"))?,
            )
            .ok_or_else(|| invalid("temporal fit peak overflow"))?,
    )?;
    let retention = Arc::new(
        ctx.acquire_admission(0, retained)
            .map_err(|error| PanelError::Resource(error.to_string()))?,
    );
    let _scratch = ctx
        .acquire_admission(
            0,
            u64::try_from(scratch).map_err(|_| invalid("temporal scratch volume overflow"))?,
        )
        .map_err(|error| PanelError::Resource(error.to_string()))?;
    let selected = fit_candidates(data, spec, options, ctx)?;
    let point = realization(data, spec, &selected.choice, &selected.values, None)
        .ok_or_else(|| PanelError::Numerical("invalid selected temporal point".into()))?;
    let mut posterior_diagnostics = None;
    let mut posterior_converged = None;
    let (draws, uncertainty) = match &options.inference {
        PanelInference::Frequentist { draws } | PanelInference::FrequentistBootstrap { draws } => {
            let mut output = Vec::with_capacity(*draws);
            for replicate in 0..*draws {
                check_cancelled(ctx)?;
                let simulated = bootstrap_data(data, &point, ctx, replicate)?;
                let compiled_simulated = likelihood::CompiledPanel::new(&simulated)?;
                let draw = match fit_candidates(&compiled_simulated, spec, options, ctx) {
                    Ok(fit) => {
                        let mut rng = ctx.rng.stream(antecedent_core::derive_stream(
                            0x5830_3352_414e_444f,
                            replicate as u64,
                        ));
                        realization(
                            &compiled_simulated,
                            spec,
                            &fit.choice,
                            &fit.values,
                            Some(&mut rng),
                        )
                    }
                    Err(PanelError::Cancelled) => return Err(PanelError::Cancelled),
                    Err(_) => None,
                };
                output.push(draw);
            }
            (output, "hierarchical model-based parametric bootstrap of the complete candidate/change/smooth-penalty selection procedure; conditional random-effect uncertainty retained; not calibrated coverage or a posterior".to_owned())
        }
        PanelInference::Bayesian { prior, sampler } => {
            let choices = candidate_choices(spec);
            let selected_index = choices
                .iter()
                .position(|choice| *choice == selected.choice)
                .ok_or_else(|| invalid("selected temporal model is outside the declared prior"))?;
            let mut initial = selected.values.clone();
            initial.resize(p + path_parameters, 0.0);
            initial.push((selected_index as f64 + 0.5) / choices.len() as f64);
            let target = |values: &[f64]| {
                let Some(index) = choice_index(values[p + path_parameters], choices.len()) else {
                    return f64::NEG_INFINITY;
                };
                objective(data, spec, &choices[index], values)
                    + inference::log_prior(data, &values[..p], prior)
                    + path_log_prior(spec, &choices[index], &values[p..p + path_parameters])
            };
            let samples = sample_metropolis_dispersed(
                &initial,
                10.0 * sampler.proposal_scale,
                MetropolisOptions {
                    chains: sampler.chains,
                    warmup: sampler.warmup,
                    draws: sampler.draws,
                    proposal_scale: sampler.proposal_scale,
                },
                ctx,
                target,
            )
            .map_err(|error| {
                if ctx.cancellation.is_cancelled() {
                    PanelError::Cancelled
                } else {
                    PanelError::Numerical(error.to_string())
                }
            })?;
            let dimensions = initial.len();
            let summary = mcmc_summary(&samples.samples, sampler.chains, sampler.draws, dimensions);
            posterior_diagnostics =
                Some([summary.max_rhat, summary.min_bulk_ess, summary.min_tail_ess]);
            posterior_converged = Some(
                summary.max_rhat <= sampler.max_rhat
                    && summary.min_bulk_ess >= sampler.min_ess
                    && summary.min_tail_ess >= sampler.min_ess
                    && all_chains_moved(
                        &samples.samples,
                        sampler.chains,
                        sampler.draws,
                        dimensions,
                    ),
            );
            let mut output = Vec::with_capacity(slots);
            for (slot, values) in samples.samples.chunks_exact(dimensions).enumerate() {
                check_cancelled(ctx)?;
                let mut rng = ctx
                    .rng
                    .stream(antecedent_core::derive_stream(0x5830_3350_4f53_5445, slot as u64));
                output.push(choice_index(values[p + path_parameters], choices.len()).and_then(
                    |index| realization(data, spec, &choices[index], values, Some(&mut rng)),
                ));
            }
            (output, "explicit-prior posterior jointly over no-change/change location, global parameters, piecewise or smooth path, smoothing-grid choice and conditional group random effects; fixed prior over every declared candidate; no identified-change or causal-attribution claim".to_owned())
        }
    };
    let mut selection_counts: Vec<(TemporalModelChoice, usize)> =
        candidate_choices(spec).into_iter().map(|choice| (choice, 0)).collect();
    for draw in draws.iter().flatten() {
        if let Some((_, count)) =
            selection_counts.iter_mut().find(|(choice, _)| *choice == draw.choice)
        {
            *count += 1;
        }
    }
    check_cancelled(ctx)?;
    Ok(TemporalPanelFit {
        point,
        draws,
        attempts: selected.attempts,
        smooth_fallback: selected.smooth,
        selection_counts,
        posterior_diagnostics,
        posterior_converged,
        uncertainty,
        claim: "model_conditional_temporal_mechanism_only".into(),
        calibration: "unmeasured".into(),
        source_identity: data.identity()?,
        retention,
        retained_volume: retained,
    })
}

impl TemporalPanelFit {
    /// Include this existing fit reservation in the current execution observation.
    pub fn observe_current(&self) {
        self.retention.observe_current();
    }
    /// Conservatively admitted retained fit volume, not allocator instrumentation or RSS.
    #[must_use]
    pub fn retained_bytes(&self) -> u64 {
        self.retention.observe_current();
        self.retained_volume
    }
}

/// Evaluate the same path/random-effect realization in every declared reference world.
///
/// # Errors
/// Refuses implicit extrapolation outside the retained path, scenario mismatch, cancellation and resource failure.
pub fn evaluate_temporal_programs(
    data: &PanelData,
    draw: &TemporalPanelDraw,
    programs: &[PanelProgram],
    ctx: &ExecutionContext,
) -> Result<Vec<PanelResponse>, PanelError> {
    ctx.try_map_indexed(1, |_, worker| evaluate_temporal_inner(data, draw, programs, worker))
        .map_err(super::map_execution_error)?
        .pop()
        .ok_or_else(|| invalid("missing admitted temporal response worker"))
}
fn evaluate_temporal_inner(
    data: &PanelData,
    draw: &TemporalPanelDraw,
    programs: &[PanelProgram],
    ctx: &ExecutionContext,
) -> Result<Vec<PanelResponse>, PanelError> {
    super::validate_panel_programs(data, programs)?;
    let compiled = likelihood::CompiledPanel::new(data)?;
    let data = &compiled;
    let bytes = super::response_storage_bytes(programs)?
        .checked_add(super::response_workspace_bytes(data)?)
        .ok_or_else(|| invalid("temporal response peak overflow"))?;
    super::check_memory(ctx, bytes)?;
    let _working =
        ctx.acquire_admission(0, bytes).map_err(|error| PanelError::Resource(error.to_string()))?;
    evaluate_temporal_bound(data, draw, programs, ctx)
}
pub(super) fn evaluate_temporal_bound(
    data: &likelihood::CompiledPanel<'_>,
    draw: &TemporalPanelDraw,
    programs: &[PanelProgram],
    ctx: &ExecutionContext,
) -> Result<Vec<PanelResponse>, PanelError> {
    if draw.baseline.group_intercepts.len() != data.groups.len()
        || draw.baseline.group_intercepts.iter().any(|value| !value.is_finite())
        || programs.iter().any(|program| program.horizon() > draw.parameters.len())
    {
        return Err(invalid(
            "temporal path/group axis mismatch or unrequested future extrapolation",
        ));
    }
    let mut result = Vec::with_capacity(programs.len());
    for program in programs {
        let mut means = vec![0.0; program.horizon()];
        for scenario in &program.scenarios {
            let group = data
                .groups
                .iter()
                .position(|group| group.id == scenario.group)
                .ok_or_else(|| invalid("unfitted temporal scenario group"))?;
            for (time, mean) in means.iter_mut().enumerate() {
                check_cancelled(ctx)?;
                let p = &draw.parameters[time];
                likelihood::validate_parameters(data, p)?;
                *mean += likelihood::program_response(data, p, program, scenario, time)
                    + draw.baseline.group_intercepts[group];
            }
        }
        for mean in &mut means {
            *mean /= data.groups.len() as f64;
            if !mean.is_finite() {
                return Err(PanelError::Numerical("temporal response overflow".into()));
            }
        }
        result.push(PanelResponse { program: program.id.clone(), means });
    }
    Ok(result)
}

/// Record the necessary coarser attempt when the design cannot distinguish explanations.
///
/// This component cannot accept a boolean claiming scientific support. The compiler must
/// separately bind identified reference-world products before exposing a causal explanation.
#[must_use]
pub fn require_coarser_contrast(explanation: PanelExplanation) -> CoarserPanelContrast {
    CoarserPanelContrast { explanation, next_attempt: "attempt the coarser total response contrast under matched population, assignments, saturation, initial histories and outcome units; retain every unsupported world and refuse a unique mechanism-change attribution".into() }
}

/// Evaluate every temporal mechanism draw using one immutable compiled expression plan.
///
/// # Errors
/// Unsupported extrapolation, invalid axes, point numerical failure, resources or cancellation.
pub fn evaluate_temporal_law(
    data: &PanelData,
    point: &TemporalPanelDraw,
    draws: &[Option<TemporalPanelDraw>],
    programs: &[PanelProgram],
    ctx: &ExecutionContext,
) -> Result<super::PanelLawResponses, PanelError> {
    ctx.try_map_indexed(1, |_, worker| {
        super::evaluate_law(data, point, draws, programs, worker, evaluate_temporal_bound)
    })
    .map_err(super::map_execution_error)?
    .pop()
    .ok_or_else(|| invalid("missing admitted temporal law worker"))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn specification() -> TemporalPanelSpec {
        TemporalPanelSpec {
            varying: vec![PanelMeanParameter::Interaction { id: "ab".into() }],
            change_points: vec![8, 12, 16],
            minimum_segment: 4,
            weak_change_gap: 4.0,
            smooth_knots: vec![0, 8, 16, 23],
            smoothing: vec![0.5, 2.0],
            path_prior_sd: 1.0,
        }
    }
    #[test]
    fn changed_mechanism_likelihood_and_response_share_lags_histories_and_reference_worlds() {
        let (mut data, baseline, _) = super::super::tests::fixture();
        let spec = specification();
        let choice = TemporalModelChoice::Piecewise { at: 12 };
        let mut values = likelihood::encode(&data, &baseline.parameters).unwrap();
        values.push(2.0);
        for group in &mut data.groups {
            for time in 12..group.outcome.len() {
                let a = likelihood::exposure(
                    &group.channels[0],
                    group.history[0],
                    baseline.parameters.channels[0].lag,
                    time,
                );
                let b = likelihood::exposure(
                    &group.channels[1],
                    group.history[1],
                    baseline.parameters.channels[1].lag,
                    time,
                );
                group.outcome[time] += 2.0 * a / (a + 2.0) * b / (b + 2.0);
            }
        }
        let compiled = likelihood::CompiledPanel::new(&data).unwrap();
        let correct = objective(&compiled, &spec, &choice, &values);
        let mut unchanged = values.clone();
        *unchanged.last_mut().unwrap() = 0.0;
        assert!(correct > objective(&compiled, &spec, &choice, &unchanged));
        let parameters = (0..24)
            .map(|time| parameters_at(&data, &spec, &choice, &values, time).unwrap())
            .collect();
        let draw = TemporalPanelDraw { baseline, parameters, choice };
        let responses = evaluate_temporal_programs(
            &data,
            &draw,
            &super::super::tests::programs(&data),
            &ExecutionContext::for_tests(8),
        )
        .unwrap();
        let law = evaluate_temporal_law(
            &data,
            &draw,
            &[Some(draw.clone()), None],
            &super::super::tests::programs(&data),
            &ExecutionContext::for_tests(8),
        )
        .unwrap();
        assert_eq!(law.point(), responses);
        assert_eq!(law.draws()[0].as_deref(), Some(responses.as_slice()));
        assert!(law.draws()[1].is_none());
        let delta = |time| {
            responses[3].means[time] - responses[1].means[time] - responses[2].means[time]
                + responses[0].means[time]
        };
        assert!((delta(11) - 2.0 / 3.0).abs() < 1e-12);
        assert!((delta(12) - 3.5 * 4.0 / 9.0).abs() < 1e-12);
    }
    #[test]
    fn smooth_path_and_selection_prior_are_explicit_not_selected_only_uncertainty() {
        let spec = specification();
        let smooth = TemporalModelChoice::Smooth { penalty: 2.0 };
        let delta = [0.8, 1.6, 2.3];
        assert!((path_value(&spec, &smooth, &delta, 0, 4) - 0.4).abs() < 1e-14);
        assert!((path_value(&spec, &smooth, &delta, 0, 20) - 2.0).abs() < 1e-14);
        assert!(penalty(&spec, &smooth, &delta) > 0.0);
        let choices = candidate_choices(&spec);
        assert_eq!(choices.len(), 6);
        for index in 0..choices.len() {
            assert_eq!(
                choice_index((index as f64 + 0.5) / choices.len() as f64, choices.len()),
                Some(index)
            );
        }
        assert!(choice_index(0.0, choices.len()).is_none());
        assert!(choice_index(1.0, choices.len()).is_none());
        // Proper smoothing precision includes its normalization, not just a
        // penalty that would distort the model-choice posterior.
        assert!((path_log_prior(&spec, &smooth, &[0.0; 3]) - 1.5 * 2.0_f64.ln()).abs() < 1e-14);
    }
    #[test]
    fn unsupported_specifications_and_unique_explanations_remain_separate_attempts() {
        let (mut data, _, _) = super::super::tests::fixture();
        let mut spec = specification();
        assert!(validate_temporal_spec(&data, &spec).is_ok());
        spec.varying = vec![PanelMeanParameter::Season { level: 0 }];
        assert!(validate_temporal_spec(&data, &spec).is_err());
        data.lag_order = 0;
        assert!(validate_temporal_spec(&data, &specification()).is_err());
        let next = require_coarser_contrast(PanelExplanation::Mechanism);
        assert_eq!(next.explanation, PanelExplanation::Mechanism);
        assert!(next.next_attempt.contains("coarser total response contrast"));
    }
}
