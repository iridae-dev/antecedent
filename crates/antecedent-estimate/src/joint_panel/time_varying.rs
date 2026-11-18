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
use antecedent_kernels::{cholesky_spd_into, standard_normal};
use antecedent_prob::{all_chains_moved, mcmc_summary};
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
    /// Logit of one channel's geometric adstock decay (geometric family).
    GeometricDecay {
        /// Stable channel identity.
        channel: String,
    },
    /// Log of one channel's Hill shape (Hill family).
    HillShape {
        /// Stable channel identity.
        channel: String,
    },
    /// Log of a nonreference saturation class's half-saturation (class-specific family).
    ClassLogHalfSaturation {
        /// Class index, 1 or more; class zero is `LogHalfSaturation`.
        class: usize,
    },
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
    /// A best-versus-next BIC gap at or below this value triggers smooth attempts;
    /// they also run when fewer than two initial fits converge or no-change leads.
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
    /// Selection score `-2 * (log likelihood - path penalty) + k * ln(n)`, if finite.
    /// The lowest score among all converged attempted candidates is retained.
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
    /// Whether weak change localization started smooth attempts, regardless of the selected point.
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
    /// Calibration remains `not_measured`.
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
    let layout = likelihood::layout(data);
    let channel_position = |id: &String, what: &'static str| {
        data.channels.iter().position(|entry| &entry.id == id).ok_or_else(|| invalid(what))
    };
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
                if data.lag_order == 2 && !data.family.is_geometric() && *coordinate < 2 =>
            {
                layout.lag
                    + 2 * channel_position(channel, "unknown varying lag channel")?
                    + coordinate
            }
            PanelMeanParameter::GeometricDecay { channel }
                if data.lag_order == 2 && data.family.is_geometric() =>
            {
                layout.lag + channel_position(channel, "unknown varying decay channel")?
            }
            PanelMeanParameter::LogHalfSaturation => layout.k0,
            PanelMeanParameter::HillShape { channel } if data.family.is_hill() => {
                layout.shape + channel_position(channel, "unknown varying Hill channel")?
            }
            PanelMeanParameter::ClassLogHalfSaturation { class }
                if *class >= 1 && *class < data.family.classes() =>
            {
                layout.class + class - 1
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
        || spec.smooth_knots[0] != 0
        || spec.smooth_knots.last().copied() != max.checked_sub(1)
        || spec.smooth_knots.windows(2).any(|pair| pair[0] >= pair[1])
        || spec.smoothing.is_empty()
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
                PanelMeanParameter::GeometricDecay { channel } => {
                    writer.word(9).text(channel);
                }
                PanelMeanParameter::HillShape { channel } => {
                    writer.word(10).text(channel);
                }
                PanelMeanParameter::ClassLogHalfSaturation { class } => {
                    writer.word(11).word(*class as u64);
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
/// The knots on either side of a period and the interpolation fraction toward the right one.
fn smooth_bracket(knots: &[usize], time: usize) -> (usize, usize, f64) {
    let right = knots.partition_point(|&point| point < time).max(1).min(knots.len() - 1);
    let left = right - 1;
    (left, right, (time - knots[left]) as f64 / (knots[right] - knots[left]) as f64)
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
            let (left, right, fraction) = smooth_bracket(&spec.smooth_knots, time);
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
/// The decoded mechanism of every period, with periods whose mechanism coordinates are
/// bit-identical sharing one decoded value (every period of a no-change path, every period on
/// one side of a change point). Decoding is a pure function of the coordinates, so sharing never
/// changes a value; `share_equal` exists so a test can compare against the unshared path.
struct PeriodMechanisms {
    distinct: Vec<PanelParameters>,
    of_period: Vec<usize>,
}
impl PeriodMechanisms {
    fn new(
        data: &PanelData,
        spec: &TemporalPanelSpec,
        choice: &TemporalModelChoice,
        values: &[f64],
        periods: usize,
        share_equal: bool,
    ) -> Option<Self> {
        let p = likelihood::parameter_count(data);
        let global = values.get(..p)?;
        let selected = coordinates(data, spec).ok()?;
        let delta = values.get(p..p + offsets(spec, choice))?;
        let mut coordinates_of: Vec<Vec<f64>> = Vec::new();
        let mut distinct = Vec::new();
        let mut of_period = Vec::with_capacity(periods);
        let mut current = global.to_vec();
        for time in 0..periods {
            current.copy_from_slice(global);
            for (coordinate, &index) in selected.iter().enumerate() {
                current[index] += path_value(spec, choice, delta, coordinate, time);
            }
            let same = if share_equal {
                coordinates_of.iter().position(|earlier| {
                    earlier.iter().zip(&current).all(|(a, b)| a.to_bits() == b.to_bits())
                })
            } else {
                None
            };
            let index = if let Some(index) = same {
                index
            } else {
                distinct.push(likelihood::decode(data, &current)?);
                coordinates_of.push(current.clone());
                distinct.len() - 1
            };
            of_period.push(index);
        }
        Some(Self { distinct, of_period })
    }
    fn at(&self, time: usize) -> &PanelParameters {
        &self.distinct[self.of_period[time]]
    }
}
fn conditional_state(
    data: &likelihood::CompiledPanel<'_>,
    spec: &TemporalPanelSpec,
    choice: &TemporalModelChoice,
    values: &[f64],
    detail: likelihood::Detail,
) -> Option<likelihood::Marginal> {
    conditional_state_with(data, spec, choice, values, detail, true)
}
fn conditional_state_with(
    data: &likelihood::CompiledPanel<'_>,
    spec: &TemporalPanelSpec,
    choice: &TemporalModelChoice,
    values: &[f64],
    detail: likelihood::Detail,
    share_equal: bool,
) -> Option<likelihood::Marginal> {
    let base = likelihood::decode(data, values.get(..likelihood::parameter_count(data))?)?;
    // The mechanism at a period does not depend on the group: decode it once per distinct
    // period mechanism instead of once per row. A period no group reaches is never decoded.
    let periods = data.groups.iter().map(|group| group.outcome.len()).max().unwrap_or(0);
    let mechanisms = PeriodMechanisms::new(data, spec, choice, values, periods, share_equal)?;
    let mut buffers = likelihood::ResponseScratch::default();
    let residuals = data
        .groups
        .iter()
        .enumerate()
        .map(|(index, group)| {
            (0..group.outcome.len())
                .map(|time| {
                    group.outcome[time]
                        - likelihood::observed_response_with(
                            data,
                            mechanisms.at(time),
                            group,
                            index,
                            time,
                            &mut buffers,
                        )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    likelihood::marginal(data, &base, &residuals, detail)
}
fn objective(
    data: &likelihood::CompiledPanel<'_>,
    spec: &TemporalPanelSpec,
    choice: &TemporalModelChoice,
    values: &[f64],
) -> f64 {
    conditional_state(data, spec, choice, values, likelihood::Detail::Value)
        .map_or(f64::NEG_INFINITY, |marginal| marginal.log_likelihood)
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
/// Gradient of `objective - penalty` with respect to the global coordinates followed by the
/// path offsets. The path enters only as `current_t = global + path_value(t)` on the varying
/// mechanism coordinates, so each period's mechanism derivative is shared between the global
/// coordinate and the offsets that move it.
fn score(
    data: &likelihood::CompiledPanel<'_>,
    spec: &TemporalPanelSpec,
    choice: &TemporalModelChoice,
    values: &[f64],
) -> Option<Vec<f64>> {
    score_with(data, spec, choice, values, true)
}
fn score_with(
    data: &likelihood::CompiledPanel<'_>,
    spec: &TemporalPanelSpec,
    choice: &TemporalModelChoice,
    values: &[f64],
    share_equal: bool,
) -> Option<Vec<f64>> {
    let p = likelihood::parameter_count(data);
    let selected = coordinates(data, spec).ok()?;
    let delta = values.get(p..p + offsets(spec, choice))?;
    let global = values.get(..p)?;
    let periods = data.groups.iter().map(|group| group.outcome.len()).max().unwrap_or(0);
    let slot_x: Vec<Vec<f64>> = (0..periods)
        .map(|time| {
            let mut current = global.to_vec();
            for (coordinate, &index) in selected.iter().enumerate() {
                current[index] += path_value(spec, choice, delta, coordinate, time);
            }
            current
        })
        .collect();
    let score =
        super::gradient::independent_score_with(data, global, &slot_x, &|time| time, share_equal)?;
    let mut output = score.shared;
    output.extend(std::iter::repeat_n(0.0, delta.len()));
    for slot in &score.slots {
        for (target, value) in output.iter_mut().zip(slot) {
            *target += value;
        }
    }
    let varying = spec.varying.len();
    for (time, slot) in score.slots.iter().enumerate() {
        for (coordinate, &index) in selected.iter().enumerate() {
            let through = slot[index];
            match choice {
                TemporalModelChoice::NoChange => {}
                TemporalModelChoice::Piecewise { at } => {
                    if time >= *at {
                        output[p + coordinate] += through;
                    }
                }
                TemporalModelChoice::Smooth { .. } => {
                    let (left, right, fraction) = smooth_bracket(&spec.smooth_knots, time);
                    if left > 0 {
                        output[p + (left - 1) * varying + coordinate] += through * (1.0 - fraction);
                    }
                    output[p + (right - 1) * varying + coordinate] += through * fraction;
                }
            }
        }
    }
    if let TemporalModelChoice::Smooth { penalty } = choice {
        // d/d delta_k of 0.5 * penalty * sum_k (delta_k - delta_(k-1))^2, delta_0 = 0.
        let knots = spec.smooth_knots.len();
        for coordinate in 0..varying {
            let at = |knot: usize| {
                if knot == 0 { 0.0 } else { delta[(knot - 1) * varying + coordinate] }
            };
            for knot in 1..knots {
                let mut slope = penalty * (at(knot) - at(knot - 1));
                if knot + 1 < knots {
                    slope -= penalty * (at(knot + 1) - at(knot));
                }
                output[p + (knot - 1) * varying + coordinate] -= slope;
            }
        }
    }
    output.iter().all(|value| value.is_finite()).then_some(output)
}
struct Selected {
    values: Vec<f64>,
    choice: TemporalModelChoice,
    attempts: Vec<TemporalPanelAttempt>,
    smooth: bool,
}
/// `polish` (frequentist fits) lets a candidate that has not met the score tolerance within the
/// quasi-Newton budget finish with Newton steps on its score's curvature
/// ([`inference::polished_local_fit`]); a Bayesian fit only seeds its sampler here and keeps the
/// plain search.
fn fit_candidates(
    data: &likelihood::CompiledPanel<'_>,
    spec: &TemporalPanelSpec,
    options: &PanelFitOptions,
    polish: bool,
    ctx: &ExecutionContext,
) -> Result<Selected, PanelError> {
    let p = likelihood::parameter_count(data);
    let initial = options.initial.clone().unwrap_or_else(|| likelihood::initial(data));
    let start = likelihood::encode(data, &initial)?;
    let n = data.groups.iter().map(|group| group.outcome.len()).sum::<usize>() as f64;
    let try_fit = |choice: TemporalModelChoice,
                   attempts: &mut Vec<TemporalPanelAttempt>|
     -> Result<Option<(f64, Vec<f64>, TemporalModelChoice)>, PanelError> {
        check_cancelled(ctx)?;
        let mut initial = start.clone();
        initial.resize(p + offsets(spec, &choice), 0.0);
        let function = |values: &[f64]| {
            objective(data, spec, &choice, values) - penalty(spec, &choice, &values[p..])
        };
        let gradient = |values: &[f64]| {
            check_cancelled(ctx)?;
            if data.family.spatial.is_some() {
                inference::numerical_score(values, ctx, &function)
            } else {
                // An exact derivative that overflows on a degenerate design (finite objective)
                // falls back to central differences for that evaluation, as the static fit does.
                match score(data, spec, &choice, values) {
                    Some(exact) if exact.iter().all(|value| value.is_finite()) => Ok(exact),
                    _ => inference::numerical_score(values, ctx, &function),
                }
            }
        };
        let optimized = if polish {
            inference::polished_local_fit(initial, options.optimizer, ctx, &function, &gradient)
        } else {
            inference::optimize_target_exact(initial, options.optimizer, ctx, &function, &gradient)
        };
        match optimized {
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
    select_candidates(spec, try_fit)
}

// Keep the attempt policy and the selection policy distinct: triggering smooth
// fits does not erase a converged incumbent from the same score comparison.
fn select_candidates(
    spec: &TemporalPanelSpec,
    mut try_fit: impl FnMut(
        TemporalModelChoice,
        &mut Vec<TemporalPanelAttempt>,
    ) -> Result<Option<(f64, Vec<f64>, TemporalModelChoice)>, PanelError>,
) -> Result<Selected, PanelError> {
    let choices = std::iter::once(TemporalModelChoice::NoChange)
        .chain(spec.change_points.iter().map(|&at| TemporalModelChoice::Piecewise { at }));
    let mut attempts = Vec::new();
    let mut successes = Vec::new();
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
        successes.sort_by(|a, b| a.0.total_cmp(&b.0));
    }
    let Some((_, values, choice)) = successes.into_iter().next() else {
        let message = if weak {
            "no eligible temporal candidate met numerical tolerance after the required smooth attempt"
        } else {
            "no temporal candidate met numerical tolerance"
        };
        return Err(PanelError::Numerical(format!("{message}; {} attempted", attempts.len())));
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
    let detail = if rng.is_some() { likelihood::Detail::Draw } else { likelihood::Detail::Mean };
    let marginal = conditional_state(data, spec, choice, values, detail)?;
    let group_intercepts = marginal.intercepts(rng)?;
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
    let spatial = likelihood::spatial_prior_intercepts(data, base, &mut rng)
        .ok_or_else(|| PanelError::Numerical("spatial correlation is not factorable".into()))?;
    for (index, group) in simulated.groups.iter_mut().enumerate() {
        check_cancelled(ctx)?;
        let intercept = match &spatial {
            likelihood::SpatialPriorIntercepts::Correlated(draws) => draws[index],
            likelihood::SpatialPriorIntercepts::Independent => {
                base.group_sd * standard_normal(&mut rng)
            }
        };
        let mut error =
            base.innovation_sd / (1.0 - base.rho.powi(2)).sqrt() * standard_normal(&mut rng);
        for time in 0..group.outcome.len() {
            let p = &draw.parameters[time];
            error = base.rho * error + base.innovation_sd * standard_normal(&mut rng);
            group.outcome[time] =
                likelihood::observed_response(data, p, group, index, time) + intercept + error;
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

/// Share of sampler iterations that propose a change of specification rather than a move
/// inside the current one.
const SPECIFICATION_MOVE_PROBABILITY: f64 = 0.25;

/// A specification's posterior location and scale: the coordinates `center + lower * z` map a
/// standardized state `z` onto the full coordinate vector.
struct Whitening {
    center: Vec<f64>,
    /// Row-major lower Cholesky factor of the local posterior covariance, or the identity
    /// where the curvature is unavailable.
    lower: Vec<f64>,
    /// Log determinant of `lower`: the Jacobian of `z -> center + lower * z`.
    log_det: f64,
}

impl Whitening {
    fn position(&self, z: &[f64]) -> Vec<f64> {
        let d = self.center.len();
        (0..d)
            .map(|row| {
                self.center[row]
                    + self.lower[row * d..=row * d + row]
                        .iter()
                        .zip(z)
                        .map(|(l, z)| l * z)
                        .sum::<f64>()
            })
            .collect()
    }
}

/// Locate one specification's posterior mode and local covariance from its own density.
///
/// A random walk in raw coordinates cannot mix on a posterior whose directions differ in scale
/// by orders of magnitude (a sharp mechanism parameter beside a weakly identified path
/// coordinate), and started at raw offsets it never reaches the bulk, so the chains disagree
/// and the posterior fails its diagnostics. Each specification is therefore whitened at its own
/// mode, exactly as the invariant posterior is.
fn whiten(
    start: Vec<f64>,
    density: &impl Fn(&[f64]) -> f64,
    options: &PanelFitOptions,
    ctx: &ExecutionContext,
) -> Result<Whitening, PanelError> {
    let d = start.len();
    let center = match inference::optimize_target(start.clone(), options.optimizer, ctx, density) {
        Ok(fit) => fit.values,
        Err(PanelError::Cancelled) => return Err(PanelError::Cancelled),
        Err(_) => start,
    };
    let covariance = match inference::information_inverse_target(&center, ctx, density) {
        Ok(inverse) => inverse,
        Err(PanelError::Cancelled) => return Err(PanelError::Cancelled),
        Err(_) => None,
    };
    let mut lower = vec![0.0; d * d];
    let factored = covariance
        .as_deref()
        .is_some_and(|covariance| cholesky_spd_into(covariance, d, &mut lower).is_ok());
    if !factored {
        lower.fill(0.0);
        for j in 0..d {
            lower[j * d + j] = 1.0;
        }
    }
    let log_det = (0..d).map(|j| lower[j * d + j].ln()).sum();
    Ok(Whitening { center, lower, log_det })
}

/// Metropolis over (specification, standardized state): a random walk in the standardized
/// state of the current specification, and a move to another specification chosen uniformly
/// that keeps the standardized state, so the proposal is symmetric. The target in that space
/// is the specification's density times the Jacobian of its whitening, which makes the
/// specification's marginal probability exactly its posterior mass.
///
/// Returns the draws in `(chain * draws + draw) * (d + 1) + coordinate` layout, with the last
/// coordinate a uniform within the bin of the drawn specification (bin `i` of `n` holds
/// specification `i`), the encoding the retained slots decode.
fn sample_model_mixture(
    whitenings: &[Whitening],
    density: &impl Fn(usize, &[f64]) -> f64,
    sampler: &super::PanelSamplingOptions,
    ctx: &ExecutionContext,
) -> Result<Vec<f64>, PanelError> {
    let n = whitenings.len();
    let d = whitenings.first().map_or(0, |w| w.center.len());
    if n == 0 || d == 0 || sampler.chains == 0 {
        return Err(invalid("empty temporal posterior state"));
    }
    let log_target = |model: usize, z: &[f64]| {
        let w = &whitenings[model];
        density(model, &w.position(z)) + w.log_det
    };
    let mut samples = Vec::with_capacity(sampler.chains * sampler.draws * (d + 1));
    for chain in 0..sampler.chains {
        let mut rng =
            ctx.rng.stream(antecedent_core::derive_stream(0x5830_3354_5750_4d58, chain as u64));
        let mut start = None;
        for attempt in 0..64 {
            let model = (chain + attempt) % n;
            let z: Vec<f64> =
                (0..d).map(|_| inference::CHAIN_START_SPREAD * standard_normal(&mut rng)).collect();
            let logp = log_target(model, &z);
            if logp.is_finite() {
                start = Some((model, z, logp));
                break;
            }
        }
        let (mut model, mut z, mut logp) = start
            .ok_or_else(|| PanelError::Numerical("dispersed posterior start density".into()))?;
        for iteration in 0..sampler.warmup + sampler.draws {
            check_cancelled(ctx)?;
            let (next_model, next_z) = if n > 1 && rng.next_f64() < SPECIFICATION_MOVE_PROBABILITY {
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    clippy::cast_precision_loss,
                    reason = "a uniform draw in [0, 1) times a small specification count is a non-negative index"
                )]
                let other = ((rng.next_f64() * (n - 1) as f64) as usize).min(n - 2);
                (if other >= model { other + 1 } else { other }, z.clone())
            } else {
                (
                    model,
                    z.iter()
                        .map(|v| v + sampler.proposal_scale * standard_normal(&mut rng))
                        .collect(),
                )
            };
            let next_logp = log_target(next_model, &next_z);
            if next_logp.is_nan() || next_logp == f64::INFINITY {
                return Err(PanelError::Numerical("nonfinite temporal posterior density".into()));
            }
            if rng.next_f64().ln() < next_logp - logp {
                model = next_model;
                z = next_z;
                logp = next_logp;
            }
            if iteration >= sampler.warmup {
                samples.extend(whitenings[model].position(&z));
                samples.push((model as f64 + 0.05 + 0.9 * rng.next_f64()) / n as f64);
            }
        }
    }
    Ok(samples)
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
    let TemporalFitAdmission { p, path_parameters, slots, retained, retention, scratch: _scratch } =
        admit_temporal_fit(data, spec, options, ctx)?;
    // A frequentist fit's interval rests on every candidate meeting the score tolerance, at the
    // point and in each bootstrap refit; a posterior only starts its sampler from the point.
    let polish = matches!(
        options.inference,
        PanelInference::Frequentist { .. } | PanelInference::FrequentistBootstrap { .. }
    );
    let selected = fit_candidates(data, spec, options, polish, ctx)?;
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
                let draw = match fit_candidates(&compiled_simulated, spec, options, polish, ctx) {
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
            // The density of one specification over the full coordinate vector: global
            // parameters followed by every path coordinate. Coordinates a specification does
            // not use are held by their prior alone, so every specification shares one
            // dimension and the model choice is a plain discrete state.
            let density = |index: usize, values: &[f64]| {
                objective(data, spec, &choices[index], values)
                    + inference::log_prior(data, &values[..p], prior)
                    + path_log_prior(spec, &choices[index], &values[p..p + path_parameters])
            };
            let coordinates = p + path_parameters;
            let dimensions = coordinates + 1;
            let mut start = selected.values.clone();
            start.truncate(p);
            start.resize(coordinates, 0.0);
            let mut whitenings = Vec::with_capacity(choices.len());
            for index in 0..choices.len() {
                check_cancelled(ctx)?;
                whitenings.push(whiten(
                    start.clone(),
                    &|values: &[f64]| density(index, values),
                    options,
                    ctx,
                )?);
            }
            let samples = sample_model_mixture(&whitenings, &density, sampler, ctx)?;
            let summary = mcmc_summary(&samples, sampler.chains, sampler.draws, dimensions);
            posterior_diagnostics =
                Some([summary.max_rhat, summary.min_bulk_ess, summary.min_tail_ess]);
            posterior_converged = Some(
                summary.max_rhat <= sampler.max_rhat
                    && summary.min_bulk_ess >= sampler.min_ess
                    && summary.min_tail_ess >= sampler.min_ess
                    && all_chains_moved(&samples, sampler.chains, sampler.draws, dimensions),
            );
            let mut output = Vec::with_capacity(slots);
            for (slot, values) in samples.chunks_exact(dimensions).enumerate() {
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
        calibration: "not_measured".into(),
        source_identity: data.identity()?,
        retention,
        retained_volume: retained,
    })
}

struct TemporalFitAdmission {
    p: usize,
    path_parameters: usize,
    slots: usize,
    retained: u64,
    retention: Arc<antecedent_core::execution::AdmissionLease>,
    scratch: antecedent_core::execution::AdmissionLease,
}

fn admit_temporal_fit(
    data: &likelihood::CompiledPanel<'_>,
    spec: &TemporalPanelSpec,
    options: &PanelFitOptions,
    ctx: &ExecutionContext,
) -> Result<TemporalFitAdmission, PanelError> {
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
    let scratch = ctx
        .acquire_admission(
            0,
            u64::try_from(scratch).map_err(|_| invalid("temporal scratch volume overflow"))?,
        )
        .map_err(|error| PanelError::Resource(error.to_string()))?;
    Ok(TemporalFitAdmission { p, path_parameters, slots, retained, retention, scratch })
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
            let columns = likelihood::ScenarioColumns::resolve(data, program, scenario);
            // The parameters change with the period, so the memo forgets before every call and
            // only its buffers are reused.
            let mut memo = likelihood::MechanismMemo::default();
            for (time, mean) in means.iter_mut().enumerate() {
                check_cancelled(ctx)?;
                let p = &draw.parameters[time];
                likelihood::validate_parameters(data, p)?;
                memo.forget();
                *mean += likelihood::program_response_columns(
                    data, p, program, scenario, &columns, time, &mut memo,
                ) + draw.baseline.group_intercepts[group];
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
    // Exercise the production candidate coordinator with deterministic fit outcomes;
    // no optimizer tolerances or random samples can mask a selection regression.
    fn selected_from_scores(
        spec: &TemporalPanelSpec,
        scores: &[Option<f64>],
    ) -> Result<Selected, PanelError> {
        let mut slot = 0;
        let selected = select_candidates(spec, |choice, attempts| {
            let score = scores[slot];
            slot += 1;
            attempts.push(TemporalPanelAttempt {
                choice: choice.clone(),
                log_likelihood: score.map(|s| -s / 2.0),
                criterion: score,
                converged: score.is_some(),
                failure: score.is_none().then(|| "local fit failed".to_owned()),
            });
            Ok(score.map(|s| (s, vec![s], choice)))
        });
        assert_eq!(slot, scores.len(), "every required candidate was attempted");
        selected
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "the fixed reference fixture preserves its exact values and bounded setup"
    )]
    fn smooth_attempts_do_not_discard_the_best_no_change_candidate() {
        let spec = specification();
        let fit = selected_from_scores(
            &spec,
            &[Some(1.0), Some(4.0), Some(5.0), Some(6.0), Some(2.0), Some(3.0)],
        )
        .unwrap();
        assert!(fit.smooth);
        assert_eq!(fit.choice, TemporalModelChoice::NoChange);
        assert_eq!(fit.values, vec![1.0]);
        assert_eq!(fit.attempts.len(), 6);
        assert!(
            matches!(&fit.attempts[4].choice, TemporalModelChoice::Smooth { penalty } if *penalty == 0.5)
        );
        assert!(fit.attempts.iter().all(|attempt| attempt.converged && attempt.failure.is_none()));
    }

    #[test]
    fn failed_smooth_attempts_keep_the_best_converged_piecewise_incumbent() {
        let spec = specification();
        let fit = selected_from_scores(&spec, &[Some(5.0), Some(2.0), Some(3.0), None, None, None])
            .unwrap();
        assert!(fit.smooth);
        assert_eq!(fit.choice, TemporalModelChoice::Piecewise { at: 8 });
        assert_eq!(fit.values, vec![2.0]);
        assert_eq!(fit.attempts.len(), 6);
        assert!(
            fit.attempts[4..].iter().all(|attempt| !attempt.converged && attempt.failure.is_some())
        );
    }

    #[test]
    fn strongly_localized_piecewise_fit_does_not_run_smooth_attempts() {
        let spec = specification();
        let fit =
            selected_from_scores(&spec, &[Some(12.0), Some(1.0), Some(8.0), Some(9.0)]).unwrap();
        assert!(!fit.smooth);
        assert_eq!(fit.choice, TemporalModelChoice::Piecewise { at: 8 });
        assert_eq!(fit.attempts.len(), 4);
    }

    #[test]
    fn a_better_smooth_candidate_wins_and_all_failed_attempts_refuse() {
        let spec = specification();
        let fit = selected_from_scores(
            &spec,
            &[Some(5.0), Some(2.0), Some(3.0), None, Some(1.0), Some(4.0)],
        )
        .unwrap();
        assert_eq!(fit.choice, TemporalModelChoice::Smooth { penalty: 0.5 });
        assert!(fit.smooth);
        let Err(error) = selected_from_scores(&spec, &[None, None, None, None, None, None]) else {
            panic!("all failed candidates must refuse")
        };
        assert!(matches!(&error, PanelError::Numerical(_)));
        assert!(error.to_string().contains("6 attempted"));
    }

    #[test]
    fn public_fit_retains_no_change_and_reports_the_smooth_attempts() {
        // The reference panel has a stationary interaction of 1.5. Start each
        // candidate at that same known mechanism with zero path offsets. A
        // deliberately loose declared score tolerance isolates selection from
        // optimizer trajectory and makes the BIC ordering deterministic.
        let (data, truth, _) = super::super::tests::fixture();
        let spec = specification();
        let options = PanelFitOptions {
            optimizer: super::super::PanelOptimizerOptions {
                max_iterations: 1,
                gradient_tolerance: 1e100,
            },
            inference: PanelInference::Frequentist { draws: 1 },
            initial: Some(truth.parameters.clone()),
        };
        let ctx = ExecutionContext::for_tests(79);
        let fit = fit_temporal_panel(&data, &spec, &options, &ctx).unwrap();
        assert_eq!(fit.point.choice, TemporalModelChoice::NoChange);
        assert!(fit.smooth_fallback);
        assert_eq!(fit.attempts.len(), 1 + spec.change_points.len() + spec.smoothing.len());
        assert_eq!(
            fit.attempts
                .iter()
                .filter(|a| matches!(&a.choice, TemporalModelChoice::Smooth { .. }))
                .count(),
            spec.smoothing.len()
        );
        let selected_score = fit
            .attempts
            .iter()
            .find(|a| a.choice == TemporalModelChoice::NoChange)
            .unwrap()
            .criterion
            .unwrap();
        assert!(selected_score.is_finite());
        assert!(
            fit.attempts
                .iter()
                .filter(|a| a.converged)
                .all(|a| a.criterion.is_some_and(|score| score >= selected_score))
        );
        assert!(fit.point.parameters.windows(2).all(|p| p[0] == p[1]));
        assert_eq!(fit.draws.len(), 1);
        assert!(fit.draws[0].is_some());
        assert!(fit.uncertainty.contains("bootstrap"));
        let responses = evaluate_temporal_programs(
            &data,
            &fit.point,
            &super::super::tests::programs(&data),
            &ctx,
        )
        .unwrap();
        let interaction = responses[3].means[12] - responses[1].means[12] - responses[2].means[12]
            + responses[0].means[12];
        assert!(interaction.is_finite());
        assert!((interaction - 2.0 / 3.0).abs() < 1e-10);
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
    #[allow(
        clippy::float_cmp,
        clippy::too_many_lines,
        reason = "the fixed reference fixture preserves its exact values and bounded setup"
    )]
    fn broader_family_coordinates_vary_through_the_same_path_and_refit_a_shape_change() {
        use super::super::family::{
            PanelAdstock, PanelFamily, PanelSaturation, PanelSaturationScope,
        };
        use super::super::family_tests::{hand_mean, simulate, truth};
        let family = PanelFamily {
            adstock: PanelAdstock::Geometric,
            saturation: PanelSaturation::Hill,
            scope: PanelSaturationScope::Classes { assignment: vec![0, 0, 0, 1, 1, 1] },
            spatial: None,
        };
        let base = truth(&family);
        let mut data = simulate(family, &base, 6, 24, 77);
        let spec = TemporalPanelSpec {
            varying: vec![PanelMeanParameter::HillShape { channel: "0".into() }],
            change_points: vec![8, 12, 16],
            minimum_segment: 4,
            weak_change_gap: 4.0,
            smooth_knots: vec![0, 8, 16, 23],
            smoothing: vec![0.5, 2.0],
            path_prior_sd: 1.0,
        };
        validate_temporal_spec(&data, &spec).unwrap();
        let layout = likelihood::layout(&data);
        assert_eq!(coordinates(&data, &spec).unwrap(), vec![layout.shape]);
        // The other family coordinates map to their own positions, or are refused when the
        // declared family lacks them.
        let mut other = spec.clone();
        other.varying = vec![
            PanelMeanParameter::ClassLogHalfSaturation { class: 1 },
            PanelMeanParameter::GeometricDecay { channel: "1".into() },
            PanelMeanParameter::LogHalfSaturation,
        ];
        assert_eq!(
            coordinates(&data, &other).unwrap(),
            vec![layout.class, layout.lag + 1, layout.k0]
        );
        for refused in [
            PanelMeanParameter::ClassLogHalfSaturation { class: 2 },
            PanelMeanParameter::ClassLogHalfSaturation { class: 0 },
            PanelMeanParameter::Lag { channel: "0".into(), coordinate: 0 },
        ] {
            other.varying = vec![refused];
            assert!(coordinates(&data, &other).is_err());
        }
        let mut reviewed = data.clone();
        reviewed.family = PanelFamily::default();
        for refused in [
            PanelMeanParameter::HillShape { channel: "0".into() },
            PanelMeanParameter::GeometricDecay { channel: "0".into() },
            PanelMeanParameter::ClassLogHalfSaturation { class: 1 },
        ] {
            other.varying = vec![refused];
            assert!(coordinates(&reviewed, &other).is_err());
        }
        assert_ne!(spec.identity(&data).unwrap(), {
            let mut changed = spec.clone();
            changed.varying = vec![PanelMeanParameter::GeometricDecay { channel: "0".into() }];
            changed.identity(&data).unwrap()
        });
        // A shape change at period twelve: shift the mean of every later period by the
        // hand-computed difference between the changed and baseline Hill mechanisms.
        let delta = 0.6;
        let mut changed = base.clone();
        changed.family.hill_shape[0] *= f64::exp(delta);
        for (g, group) in data.groups.iter_mut().enumerate() {
            for time in 12..group.outcome.len() {
                let (after, before) = (
                    hand_mean(&changed, &data.family, g, group, time),
                    hand_mean(&base, &data.family, g, group, time),
                );
                group.outcome[time] += after - before;
            }
        }
        let choice = TemporalModelChoice::Piecewise { at: 12 };
        let mut values = likelihood::encode(&data, &base).unwrap();
        values.push(delta);
        let compiled = likelihood::CompiledPanel::new(&data).unwrap();
        let before = parameters_at(&data, &spec, &choice, &values, 11).unwrap();
        let after = parameters_at(&data, &spec, &choice, &values, 12).unwrap();
        assert!((before.family.hill_shape[0] - 2.0).abs() < 1e-12);
        assert!((after.family.hill_shape[0] - 2.0 * f64::exp(delta)).abs() < 1e-12);
        assert_eq!(after.family.hill_shape[1], before.family.hill_shape[1]);
        assert_eq!(after.family.class_half_saturation, before.family.class_half_saturation);
        let mut unchanged = values.clone();
        *unchanged.last_mut().unwrap() = 0.0;
        assert!(
            objective(&compiled, &spec, &choice, &values)
                > objective(&compiled, &spec, &choice, &unchanged)
        );
        // The complete selection procedure on the same mechanism attempts every candidate.
        let options = PanelFitOptions {
            optimizer: super::super::PanelOptimizerOptions {
                max_iterations: 800,
                gradient_tolerance: 5e-2,
            },
            inference: PanelInference::Frequentist { draws: 0 },
            initial: Some(base.clone()),
        };
        let fit =
            fit_temporal_panel(&data, &spec, &options, &ExecutionContext::for_tests(61)).unwrap();
        let log_likelihood = |wanted: TemporalModelChoice| {
            fit.attempts
                .iter()
                .find(|attempt| attempt.choice == wanted)
                .and_then(|attempt| attempt.log_likelihood)
                .unwrap()
        };
        assert!(
            log_likelihood(TemporalModelChoice::Piecewise { at: 12 })
                > log_likelihood(TemporalModelChoice::NoChange) + 10.0
        );
        assert_eq!(fit.claim, "model_conditional_temporal_mechanism_only");
        let path = &fit.point.parameters;
        assert!(
            path[20].family.hill_shape[0] / path[2].family.hill_shape[0] > 1.2,
            "the fitted shape must rise after the change: {} then {}",
            path[2].family.hill_shape[0],
            path[20].family.hill_shape[0]
        );
    }
    #[test]
    fn analytic_path_score_matches_central_differences_for_every_specification() {
        use super::super::family::{
            PanelAdstock, PanelFamily, PanelSaturation, PanelSaturationScope,
        };
        use super::super::family_tests::{
            assert_scores_agree, five_point_gradient, simulate, truth,
        };
        let family = PanelFamily {
            adstock: PanelAdstock::Geometric,
            saturation: PanelSaturation::Hill,
            scope: PanelSaturationScope::Classes { assignment: vec![0, 0, 0, 1, 1, 1] },
            spatial: None,
        };
        let base = truth(&family);
        let mut broad = simulate(family, &base, 6, 24, 90);
        broad.groups[1].weight = 2.0;
        broad.groups[2].outcome[5] = f64::NAN;
        broad.groups[4].outcome[11] = f64::NAN;
        let broad_start = likelihood::encode(&broad, &base).unwrap();
        let broad_spec = TemporalPanelSpec {
            varying: vec![
                PanelMeanParameter::HillShape { channel: "0".into() },
                PanelMeanParameter::GeometricDecay { channel: "1".into() },
                PanelMeanParameter::ClassLogHalfSaturation { class: 1 },
                PanelMeanParameter::Intercept,
            ],
            ..specification()
        };
        let (reviewed, baseline, _) = super::super::tests::fixture();
        let reviewed_start = likelihood::encode(&reviewed, &baseline.parameters).unwrap();
        let cases = [(broad, broad_spec, broad_start), (reviewed, specification(), reviewed_start)];
        let mut state = 17_u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / (1_u64 << 53) as f64 - 0.5
        };
        for (data, spec, start) in cases {
            let compiled = likelihood::CompiledPanel::new(&data).unwrap();
            let p = start.len();
            for choice in [
                TemporalModelChoice::NoChange,
                TemporalModelChoice::Piecewise { at: 12 },
                TemporalModelChoice::Smooth { penalty: 2.0 },
            ] {
                for point in 0..3 {
                    let mut values = start.clone();
                    values.resize(p + offsets(&spec, &choice), 0.0);
                    for value in &mut values {
                        *value += 0.3 * next();
                    }
                    let function = |values: &[f64]| {
                        objective(&compiled, &spec, &choice, values)
                            - penalty(&spec, &choice, &values[p..])
                    };
                    assert!(function(&values).is_finite());
                    let analytic = score(&compiled, &spec, &choice, &values).unwrap();
                    let numeric = five_point_gradient(&values, &function);
                    assert_scores_agree(&analytic, &numeric, &format!("{choice:?} point {point}"));
                }
            }
        }
    }
    #[test]
    fn shared_period_decoding_reproduces_per_period_decoding_bit_for_bit() {
        let (data, baseline, _) = super::super::tests::fixture();
        let spec = specification();
        let compiled = likelihood::CompiledPanel::new(&data).unwrap();
        let p = likelihood::parameter_count(&data);
        let start = likelihood::encode(&data, &baseline.parameters).unwrap();
        let mut state = 5_u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / (1_u64 << 53) as f64 - 0.5
        };
        for choice in [
            TemporalModelChoice::NoChange,
            TemporalModelChoice::Piecewise { at: 12 },
            TemporalModelChoice::Smooth { penalty: 2.0 },
        ] {
            let mut values = start.clone();
            values.resize(p + offsets(&spec, &choice), 0.0);
            for value in &mut values {
                *value += 0.2 * next();
            }
            for detail in [likelihood::Detail::Value, likelihood::Detail::Mean] {
                let shared =
                    conditional_state_with(&compiled, &spec, &choice, &values, detail, true)
                        .unwrap();
                let separate =
                    conditional_state_with(&compiled, &spec, &choice, &values, detail, false)
                        .unwrap();
                assert_eq!(shared.log_likelihood.to_bits(), separate.log_likelihood.to_bits());
                assert_eq!(shared.intercept_mean.len(), separate.intercept_mean.len());
                for (a, b) in shared.intercept_mean.iter().zip(&separate.intercept_mean) {
                    assert_eq!(a.to_bits(), b.to_bits());
                }
            }
            let shared = score_with(&compiled, &spec, &choice, &values, true).unwrap();
            let separate = score_with(&compiled, &spec, &choice, &values, false).unwrap();
            assert_eq!(shared.len(), separate.len());
            for (a, b) in shared.iter().zip(&separate) {
                assert_eq!(a.to_bits(), b.to_bits(), "{choice:?}");
            }
        }
    }
    #[test]
    fn both_gradient_sources_reach_the_same_temporal_optimum() {
        let (data, baseline, _) = super::super::tests::fixture();
        let spec = specification();
        let choice = TemporalModelChoice::Piecewise { at: 12 };
        let compiled = likelihood::CompiledPanel::new(&data).unwrap();
        let p = likelihood::parameter_count(&data);
        let mut start = likelihood::encode(&data, &baseline.parameters).unwrap();
        start.resize(p + offsets(&spec, &choice), 0.0);
        let function = |values: &[f64]| {
            objective(&compiled, &spec, &choice, values) - penalty(&spec, &choice, &values[p..])
        };
        let options =
            super::super::PanelOptimizerOptions { max_iterations: 3_000, gradient_tolerance: 1e-3 };
        let ctx = ExecutionContext::for_tests(3);
        let numeric = inference::optimize_target(start.clone(), options, &ctx, &function).unwrap();
        let analytic =
            inference::optimize_target_with_score(start, options, &ctx, &function, &|values| {
                inference::checked_score(score(&compiled, &spec, &choice, values))
            })
            .unwrap();
        assert!(numeric.converged && analytic.converged);
        assert!(
            (numeric.likelihood - analytic.likelihood).abs()
                <= 1e-6 * (1.0 + numeric.likelihood.abs()),
            "{} vs {}",
            numeric.likelihood,
            analytic.likelihood
        );
    }
    #[test]
    fn broader_family_posterior_selects_jointly_with_the_family_coordinates_and_group_axes() {
        use super::super::family::{
            PanelAdstock, PanelFamily, PanelSaturation, PanelSaturationScope,
        };
        use super::super::family_tests::{simulate, truth};
        use super::super::{PanelOptimizerOptions, PanelPrior, PanelSamplingOptions};
        let family = PanelFamily {
            adstock: PanelAdstock::Geometric,
            saturation: PanelSaturation::Hill,
            scope: PanelSaturationScope::Classes { assignment: vec![0, 0, 0, 1, 1, 1] },
            spatial: None,
        };
        let base = truth(&family);
        let data = simulate(family, &base, 6, 24, 78);
        let spec = TemporalPanelSpec {
            varying: vec![PanelMeanParameter::HillShape { channel: "0".into() }],
            change_points: vec![8, 12, 16],
            minimum_segment: 4,
            weak_change_gap: 4.0,
            smooth_knots: vec![0, 8, 16, 23],
            smoothing: vec![0.5, 2.0],
            path_prior_sd: 1.0,
        };
        let options = PanelFitOptions {
            optimizer: PanelOptimizerOptions { max_iterations: 300, gradient_tolerance: 5e-2 },
            inference: PanelInference::Bayesian {
                prior: PanelPrior::default(),
                sampler: PanelSamplingOptions {
                    chains: 2,
                    warmup: 30,
                    draws: 30,
                    proposal_scale: 0.2,
                    max_rhat: 5.0,
                    min_ess: 1.0,
                },
            },
            initial: Some(base),
        };
        let fit =
            fit_temporal_panel(&data, &spec, &options, &ExecutionContext::for_tests(62)).unwrap();
        assert_eq!(fit.draws.len(), 60);
        let retained = fit.draws.iter().flatten().collect::<Vec<_>>();
        assert!(retained.len() >= 50);
        assert!(fit.posterior_diagnostics.is_some());
        assert_eq!(
            fit.selection_counts.iter().map(|(_, count)| count).sum::<usize>(),
            retained.len(),
            "every retained draw records the specification it was drawn under"
        );
        for draw in retained {
            assert_eq!(draw.parameters.len(), 24);
            assert_eq!(draw.baseline.group_intercepts.len(), 6);
            assert!(draw.parameters.iter().all(|p| {
                p.family.hill_shape.len() == 2
                    && p.family.class_half_saturation
                        == draw.baseline.parameters.family.class_half_saturation
                    && p.channels == draw.baseline.parameters.channels
            }));
            if matches!(draw.choice, TemporalModelChoice::NoChange) {
                assert!(draw.parameters.windows(2).all(|w| w[0] == w[1]));
            }
        }
        assert!(fit.uncertainty.contains("posterior"));
    }
    /// Gaussian specifications of different location, scale and prior weight: the sampler's
    /// specification frequencies must equal the weights, which holds only when the whitening
    /// Jacobian is part of the target.
    #[test]
    fn specification_sampler_recovers_posterior_masses_across_whitenings() {
        use super::super::PanelSamplingOptions;
        let centers: [[f64; 2]; 3] = [[0.0, 0.0], [0.5, -0.3], [1.0, 1.0]];
        let scales: [[f64; 2]; 3] = [[1.0, 1.0], [0.1, 0.2], [0.15, 0.7]];
        let weights: [f64; 3] = [1.0, 2.0, 3.0];
        let whitenings: Vec<Whitening> = (0..3)
            .map(|m| Whitening {
                center: centers[m].to_vec(),
                lower: vec![scales[m][0], 0.0, 0.0, scales[m][1]],
                log_det: scales[m][0].ln() + scales[m][1].ln(),
            })
            .collect();
        let density = |m: usize, x: &[f64]| {
            let quadratic: f64 =
                (0..2).map(|j| ((x[j] - centers[m][j]) / scales[m][j]).powi(2)).sum();
            weights[m].ln() - 0.5 * quadratic - scales[m][0].ln() - scales[m][1].ln()
        };
        let sampler = PanelSamplingOptions {
            chains: 4,
            warmup: 500,
            draws: 6_000,
            proposal_scale: 0.8,
            max_rhat: 1.05,
            min_ess: 100.0,
        };
        let samples =
            sample_model_mixture(&whitenings, &density, &sampler, &ExecutionContext::for_tests(5))
                .unwrap();
        assert_eq!(samples.len(), 4 * 6_000 * 3);
        let mut counts = [0.0_f64; 3];
        for row in samples.chunks_exact(3) {
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "the stored specification coordinate is a uniform in [0, 1)"
            )]
            let slot = (row[2] * 3.0) as usize;
            counts[slot] += 1.0;
        }
        let total: f64 = counts.iter().sum();
        for (count, weight) in counts.iter().zip(weights) {
            assert!((count / total - weight / 6.0).abs() < 0.04, "{counts:?}");
        }
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

#[cfg(test)]
#[path = "time_varying_dependence_reference_tests.rs"]
mod dependence_reference_tests;
