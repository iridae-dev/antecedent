//! Survival under static treatment regimes from discrete-time subject trajectories with
//! time-varying confounding.
//!
//! # Data and target
//!
//! Each subject contributes one row per period `k = 0, 1, ...` while at risk: the confounders
//! `L_k` measured at the start of the period, the treatment `A_k`, then whether the subject was
//! censored during the period (`C_k`, the event then unobserved) and, if not, whether the event
//! occurred (`Y_k`). A subject's rows end at its first event or censoring or at the end of its
//! follow-up. The target is `S_1(K) - S_0(K)` with `S_a(K) = P(T(a) > K)`: the probability of
//! surviving the first `K` periods under the static regime "treat every period" (`a = 1`) or
//! "never treat" (`a = 0`). It is identified under sequential exchangeability of treatment and
//! censoring given the measured history, positivity of both regimes and consistency; these are
//! declared, not checked.
//!
//! # Frequentist procedure
//!
//! Pooled logistic models of treatment `P(A_k | L_k, A_{k-1}, k)` and censoring
//! `P(C_k | L_k, A_k, k)` give cumulative inverse-probability weights of regime-consistent,
//! uncensored person-periods; the weighted discrete hazard of each period is the weighted event
//! share, and survival is the product of one minus the hazards. The uncertainty is a
//! whole-procedure subject bootstrap: every replicate resamples subjects with replacement and
//! refits both models before recomputing the weights and hazards, so the propensity and
//! censoring fits are inside the interval. Its coverage is not measured here.
//!
//! # Bayesian procedure
//!
//! A joint likelihood over the observed trajectory factorizes into the confounder transition
//! law (Gaussian linear `L_{k,c} | L_{k-1}, A_{k-1}, k`, conjugate normal-inverse-gamma), the
//! treatment law and the censoring law (pooled logistic) and the hazard (pooled logistic
//! `Y_k | L_k, A_k, k`). Each factor's posterior is computed from its own likelihood: exact for
//! the confounders, a Laplace approximation at the posterior mode under an isotropic Gaussian
//! coefficient prior for the three logistic factors. Survival under a regime is the parametric
//! g-formula: per posterior draw, every subject's baseline confounders (reweighted by a
//! Bayesian-bootstrap draw over subjects) are carried forward through the drawn confounder law
//! under the regime, and the drawn hazards are multiplied out. Under sequential exchangeability
//! the treatment and censoring factors do not enter the g-formula; their posteriors are kept to
//! report the posterior positivity of the regimes. The posterior is uncalibrated and the prior
//! supplies no identifying premise.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{ExecutionContext, StreamDomain};

use super::linalg::{GaussianRegression, LogisticFit, fit_gaussian, fit_logistic, lower_times};
use super::{PosteriorSummary, invalid, summarize_draws};
use crate::error::EstimationError;
use crate::util::refuse;

/// Stream key of the subject bootstrap.
const BOOTSTRAP_STREAM: u64 = 0x5A1F_B007_0000_0000;
/// Stream key of the posterior g-formula draws.
const POSTERIOR_STREAM: u64 = 0x5A1F_6F00_0000_0000;
/// Largest share of bootstrap replicates allowed to fail.
const MAX_FAILED_SHARE: f64 = 0.05;
/// Fewest bootstrap replicates of the whole-procedure interval.
pub const MIN_REPLICATES: usize = 99;

/// Coordinate names of a regime survival fit.
pub const SURVIVAL_COORDINATES: [&str; 3] =
    ["survival_difference", "always_treat_survival", "never_treat_survival"];

/// Row-aligned subject trajectories.
#[derive(Clone, Copy, Debug)]
pub struct TrajectoryRows<'a> {
    /// Dense subject index of each row.
    pub unit: &'a [usize],
    /// Period of each row, counted from each subject's start at 0.
    pub period: &'a [i64],
    /// Treatment in the period.
    pub treatment: &'a [bool],
    /// Censored during the period (the event then unobserved).
    pub censored: &'a [bool],
    /// Event during the period.
    pub event: &'a [bool],
    /// Confounder columns, each row-aligned.
    pub confounders: &'a [Vec<f64>],
}

/// One subject's ordered person-periods up to the horizon.
#[derive(Clone, Debug)]
struct Subject {
    /// `[period][confounder]`.
    confounders: Vec<Vec<f64>>,
    treatment: Vec<bool>,
    censored: Vec<bool>,
    event: Vec<bool>,
}

/// Validated trajectories.
#[derive(Clone, Debug)]
pub struct Trajectories {
    subjects: Vec<Subject>,
    confounders: usize,
    horizon: usize,
}

impl Trajectories {
    /// Group, order and validate the rows of every subject, keeping periods below `horizon`.
    ///
    /// # Errors
    /// Misaligned rows, a horizon of zero, a subject whose periods do not run `0, 1, ...`
    /// without gaps, a row after an event or censoring, an event in a censored period, a
    /// non-finite confounder, or fewer than two subjects.
    pub fn new(rows: TrajectoryRows<'_>, horizon: usize) -> Result<Self, EstimationError> {
        let n = rows.unit.len();
        if n == 0
            || horizon == 0
            || [rows.period.len(), rows.treatment.len(), rows.censored.len(), rows.event.len()]
                .iter()
                .any(|len| *len != n)
            || rows.confounders.iter().any(|column| column.len() != n)
        {
            return Err(invalid("trajectory arrays must be aligned and the horizon positive"));
        }
        let units = rows.unit.iter().copied().max().map_or(0, |max| max + 1);
        let mut by_unit: Vec<Vec<usize>> = vec![Vec::new(); units];
        for (row, unit) in rows.unit.iter().enumerate() {
            by_unit[*unit].push(row);
        }
        let mut subjects = Vec::with_capacity(units);
        for mut members in by_unit.into_iter().filter(|members| !members.is_empty()) {
            members.sort_by_key(|row| rows.period[*row]);
            let mut subject = Subject {
                confounders: Vec::new(),
                treatment: Vec::new(),
                censored: Vec::new(),
                event: Vec::new(),
            };
            for (expected, &row) in members.iter().enumerate() {
                if rows.period[row] != i64::try_from(expected).unwrap_or(i64::MAX) {
                    return Err(invalid(
                        "each subject's periods must run 0, 1, 2, ... without gaps or repeats",
                    ));
                }
                if expected > 0 && (subject.censored[expected - 1] || subject.event[expected - 1]) {
                    return Err(invalid("a subject has a row after its event or censoring"));
                }
                if rows.censored[row] && rows.event[row] {
                    return Err(invalid("an event cannot be observed in a censored period"));
                }
                let values: Vec<f64> = rows.confounders.iter().map(|column| column[row]).collect();
                if values.iter().any(|value| !value.is_finite()) {
                    return Err(invalid("every confounder value must be finite"));
                }
                subject.confounders.push(values);
                subject.treatment.push(rows.treatment[row]);
                subject.censored.push(rows.censored[row]);
                subject.event.push(rows.event[row]);
            }
            let keep = subject.treatment.len().min(horizon);
            subject.confounders.truncate(keep);
            subject.treatment.truncate(keep);
            subject.censored.truncate(keep);
            subject.event.truncate(keep);
            subjects.push(subject);
        }
        if subjects.len() < super::MIN_CLUSTERS {
            return Err(invalid("the survival design needs at least two subjects"));
        }
        Ok(Self { subjects, confounders: rows.confounders.len(), horizon })
    }

    /// Number of subjects.
    #[must_use]
    pub fn subjects(&self) -> usize {
        self.subjects.len()
    }

    /// Number of person-periods kept.
    #[must_use]
    pub fn person_periods(&self) -> usize {
        self.subjects.iter().map(|subject| subject.treatment.len()).sum()
    }

    fn period_feature(&self, k: usize) -> f64 {
        k as f64 / self.horizon as f64
    }

    /// `[1, L_k, A_{k-1}, k/K]`.
    fn treatment_features(&self, subject: &Subject, k: usize, out: &mut Vec<f64>) {
        out.push(1.0);
        out.extend_from_slice(&subject.confounders[k]);
        out.push(if k > 0 && subject.treatment[k - 1] { 1.0 } else { 0.0 });
        out.push(self.period_feature(k));
    }

    /// `[1, L_k, a, k/K]` for the censoring and hazard models.
    fn current_features(&self, confounders: &[f64], treated: bool, k: usize, out: &mut Vec<f64>) {
        out.push(1.0);
        out.extend_from_slice(confounders);
        out.push(if treated { 1.0 } else { 0.0 });
        out.push(self.period_feature(k));
    }

    /// `[1, L_{k-1}, A_{k-1}, k/K]` for the confounder transition.
    fn transition_features(
        &self,
        previous: &[f64],
        previously_treated: bool,
        k: usize,
        out: &mut Vec<f64>,
    ) {
        out.push(1.0);
        out.extend_from_slice(previous);
        out.push(if previously_treated { 1.0 } else { 0.0 });
        out.push(self.period_feature(k));
    }

    fn width(&self) -> usize {
        self.confounders + 3
    }
}

/// A fitted regime survival contrast.
#[derive(Clone, Debug, PartialEq)]
pub struct SurvivalFit {
    /// `[S_1(K) - S_0(K), S_1(K), S_0(K)]`.
    pub estimate: [f64; 3],
    /// Bootstrap standard deviation of each coordinate (frequentist).
    pub standard_error: Option<[f64; 3]>,
    /// Percentile interval of each coordinate (frequentist).
    pub interval: Option<[(f64, f64); 3]>,
    /// Successful and failed bootstrap replicates (frequentist).
    pub replicates: (usize, usize),
    /// Posterior (Bayesian).
    pub posterior: Option<PosteriorSummary>,
    /// Smallest fitted probability of a regime-consistent action (frequentist point fit).
    pub minimum_regime_probability: f64,
    /// Posterior share of draws whose smallest regime-consistent action probability falls
    /// below the floor (Bayesian).
    pub posterior_positivity_shortfall: Option<f64>,
    /// Subjects and person-periods.
    pub subjects: usize,
    /// Person-periods kept below the horizon.
    pub person_periods: usize,
}

/// The weighted procedure on subject weights (multiplicities in a bootstrap replicate).
#[allow(
    clippy::float_cmp,
    reason = "a subject absent from a bootstrap replicate has a weight of exactly zero"
)]
fn ipw_procedure(
    data: &Trajectories,
    subject_weights: Option<&[f64]>,
    floor: f64,
) -> Result<([f64; 3], f64), EstimationError> {
    let width = data.width();
    let weight = |i: usize| subject_weights.map_or(1.0, |weights| weights[i]);
    let (mut tx, mut ty, mut tw) = (Vec::new(), Vec::new(), Vec::new());
    let (mut cx, mut cy, mut cw) = (Vec::new(), Vec::new(), Vec::new());
    for (i, subject) in data.subjects.iter().enumerate() {
        let w = weight(i);
        if w == 0.0 {
            continue;
        }
        for k in 0..subject.treatment.len() {
            data.treatment_features(subject, k, &mut tx);
            ty.push(f64::from(u8::from(subject.treatment[k])));
            tw.push(w);
            data.current_features(&subject.confounders[k], subject.treatment[k], k, &mut cx);
            cy.push(f64::from(u8::from(subject.censored[k])));
            cw.push(w);
        }
    }
    let treatment = fit_logistic(&tx, width, &ty, Some(&tw), 0.0)?;
    let censoring = if cy.iter().any(|value| *value > 0.0) {
        Some(fit_logistic(&cx, width, &cy, Some(&cw), 0.0)?)
    } else {
        None
    };
    let mut survival = [1.0_f64; 2];
    let mut minimum = 1.0_f64;
    let mut features = Vec::with_capacity(width);
    for (arm_index, arm) in [true, false].into_iter().enumerate() {
        let mut numerator = vec![0.0; data.horizon];
        let mut denominator = vec![0.0; data.horizon];
        for (i, subject) in data.subjects.iter().enumerate() {
            let w = weight(i);
            if w == 0.0 {
                continue;
            }
            let mut cumulative = 1.0;
            for k in 0..subject.treatment.len() {
                if subject.treatment[k] != arm {
                    break;
                }
                features.clear();
                data.treatment_features(subject, k, &mut features);
                let p_treat = probability(&treatment, &features);
                let p_arm = if arm { p_treat } else { 1.0 - p_treat };
                minimum = minimum.min(p_arm);
                if p_arm < floor {
                    return Err(refuse(
                        antecedent_core::reason_code!("positivity_violated"),
                        "regime_survival.positivity",
                        &format!(
                            "a regime-consistent treatment probability {p_arm} is below the floor \
                             {floor}. Next attempt: a lower declared floor if the design \
                             supports it, or a regime the data support"
                        ),
                    ));
                }
                cumulative /= p_arm;
                if let Some(model) = &censoring {
                    features.clear();
                    data.current_features(&subject.confounders[k], arm, k, &mut features);
                    let p_censor = probability(model, &features);
                    cumulative /= (1.0 - p_censor).max(floor);
                }
                if subject.censored[k] {
                    break;
                }
                numerator[k] += w * cumulative * f64::from(u8::from(subject.event[k]));
                denominator[k] += w * cumulative;
            }
        }
        for (k, (events, at_risk)) in numerator.iter().zip(&denominator).enumerate() {
            if *at_risk <= 0.0 {
                return Err(refuse(
                    antecedent_core::reason_code!("arm_not_populated"),
                    "regime_survival.no_regime_support",
                    &format!(
                        "no uncensored subject followed the {} regime through period {k}",
                        if arm { "always-treat" } else { "never-treat" }
                    ),
                ));
            }
            survival[arm_index] *= 1.0 - events / at_risk;
        }
    }
    Ok(([survival[0] - survival[1], survival[0], survival[1]], minimum))
}

fn probability(fit: &LogisticFit, features: &[f64]) -> f64 {
    let eta: f64 = fit.coefficients.iter().zip(features).map(|(a, b)| a * b).sum();
    antecedent_kernels::sigmoid(eta)
}

/// Frequentist regime survival with a whole-procedure subject bootstrap.
///
/// # Errors
/// Fewer than [`MIN_REPLICATES`] replicates, a positivity floor outside `(0, 0.5]`, a model fit
/// that fails, a period no regime-consistent subject reaches, more than five percent failed
/// replicates, admission or cancellation.
pub fn fit_regime_survival(
    data: &Trajectories,
    floor: f64,
    replicates: usize,
    ctx: &ExecutionContext,
) -> Result<SurvivalFit, EstimationError> {
    if !(floor.is_finite() && floor > 0.0 && floor <= 0.5) {
        return Err(invalid("the positivity floor must be in (0, 0.5]"));
    }
    if replicates < MIN_REPLICATES {
        return Err(refuse(
            antecedent_core::reason_code!("required_option_missing"),
            "regime_survival.bootstrap_replicates",
            "the regime survival interval refits the treatment and censoring models in every \
             subject-bootstrap replicate and needs at least 99 replicates. Next attempt: set \
             bootstrap_replicates to 99 or more",
        ));
    }
    let (estimate, minimum) = ipw_procedure(data, None, floor)?;
    let n = data.subjects();
    let _lease = crate::util::admit_array::<f64>(ctx, replicates, 3 + n)?;
    let mut columns: [Vec<f64>; 3] = [
        Vec::with_capacity(replicates),
        Vec::with_capacity(replicates),
        Vec::with_capacity(replicates),
    ];
    let mut counts = vec![0.0; n];
    let mut failed = 0_usize;
    for replicate in 0..replicates {
        if ctx.cancellation.is_cancelled() {
            return Err(antecedent_stats::StatsError::Cancelled.into());
        }
        let mut rng =
            ctx.rng.stream_for(StreamDomain::Resample, BOOTSTRAP_STREAM | replicate as u64);
        counts.iter_mut().for_each(|value| *value = 0.0);
        for _ in 0..n {
            counts[antecedent_kernels::unbiased_index(&mut rng, n)] += 1.0;
        }
        match ipw_procedure(data, Some(&counts), floor) {
            Ok((values, _)) => {
                for (column, value) in columns.iter_mut().zip(values) {
                    column.push(value);
                }
            }
            Err(_) => failed += 1,
        }
    }
    if failed as f64 > MAX_FAILED_SHARE * replicates as f64 {
        return Err(refuse(
            antecedent_core::reason_code!("arm_not_populated"),
            "regime_survival.bootstrap_failures",
            &format!(
                "{failed} of {replicates} subject-bootstrap replicates could not refit the \
                 procedure (positivity or regime support). Next attempt: more subjects, a \
                 shorter horizon or a lower floor"
            ),
        ));
    }
    let mut standard_error = [0.0; 3];
    let mut interval = [(0.0, 0.0); 3];
    for (j, column) in columns.iter_mut().enumerate() {
        standard_error[j] = crate::util::sample_std(column);
        column.sort_by(f64::total_cmp);
        interval[j] = antecedent_stats::equal_tail_interval_sorted(
            column,
            super::NOMINAL_LEVEL,
            antecedent_stats::QuantileRule::ExchangeableRank,
        );
    }
    Ok(SurvivalFit {
        estimate,
        standard_error: Some(standard_error),
        interval: Some(interval),
        replicates: (replicates - failed, failed),
        posterior: None,
        minimum_regime_probability: minimum,
        posterior_positivity_shortfall: None,
        subjects: n,
        person_periods: data.person_periods(),
    })
}

/// The four likelihood factors' posteriors.
struct JointPosterior {
    transitions: Vec<GaussianRegression>,
    hazard: LogisticFit,
    treatment: LogisticFit,
    censoring: Option<LogisticFit>,
}

fn fit_joint(data: &Trajectories, prior_precision: f64) -> Result<JointPosterior, EstimationError> {
    let width = data.width();
    let (mut hx, mut hy) = (Vec::new(), Vec::new());
    let (mut tx, mut ty) = (Vec::new(), Vec::new());
    let (mut cx, mut cy) = (Vec::new(), Vec::new());
    let mut lx = Vec::new();
    let mut ly: Vec<Vec<f64>> = vec![Vec::new(); data.confounders];
    for subject in &data.subjects {
        for k in 0..subject.treatment.len() {
            data.treatment_features(subject, k, &mut tx);
            ty.push(f64::from(u8::from(subject.treatment[k])));
            data.current_features(&subject.confounders[k], subject.treatment[k], k, &mut cx);
            cy.push(f64::from(u8::from(subject.censored[k])));
            if !subject.censored[k] {
                data.current_features(&subject.confounders[k], subject.treatment[k], k, &mut hx);
                hy.push(f64::from(u8::from(subject.event[k])));
            }
            if k > 0 {
                data.transition_features(
                    &subject.confounders[k - 1],
                    subject.treatment[k - 1],
                    k,
                    &mut lx,
                );
                for (c, column) in ly.iter_mut().enumerate() {
                    column.push(subject.confounders[k][c]);
                }
            }
        }
    }
    let hazard = fit_logistic(&hx, width, &hy, None, prior_precision)?;
    let treatment = fit_logistic(&tx, width, &ty, None, prior_precision)?;
    let censoring = if cy.iter().any(|value| *value > 0.0) {
        Some(fit_logistic(&cx, width, &cy, None, prior_precision)?)
    } else {
        None
    };
    let transitions = if data.horizon > 1 && data.confounders > 0 {
        if ly[0].len() < width + 1 {
            return Err(refuse(
                antecedent_core::reason_code!("invalid_argument"),
                "regime_survival.too_few_transitions",
                "too few observed period-to-period transitions to fit the confounder law",
            ));
        }
        ly.iter()
            .map(|column| fit_gaussian(&lx, width, column, prior_precision))
            .collect::<Result<Vec<_>, _>>()?
    } else {
        Vec::new()
    };
    Ok(JointPosterior { transitions, hazard, treatment, censoring })
}

fn draw_coefficients(
    mode: &[f64],
    factor: &[f64],
    scale: f64,
    rng: &mut antecedent_core::CausalRng,
) -> Vec<f64> {
    let p = mode.len();
    let mut z = vec![0.0; p];
    antecedent_kernels::fill_standard_normal(rng, &mut z);
    let shift = lower_times(factor, p, &z);
    mode.iter().zip(shift).map(|(m, s)| m + scale * s).collect()
}

/// Bayesian regime survival from the joint likelihood of confounders, treatment, censoring and
/// hazard, evaluated by the parametric g-formula per posterior draw.
///
/// # Errors
/// A non-positive prior scale, fewer than two draws, a factor fit that fails, admission or
/// cancellation.
#[allow(
    clippy::too_many_lines,
    reason = "one posterior draw loop whose per-draw buffers and shortfall count feed the summary"
)]
pub fn fit_regime_survival_posterior(
    data: &Trajectories,
    floor: f64,
    prior_scale: f64,
    draws: usize,
    ctx: &ExecutionContext,
) -> Result<SurvivalFit, EstimationError> {
    if !(prior_scale.is_finite() && prior_scale > 0.0) || draws < 2 {
        return Err(invalid("the prior scale must be positive and at least two draws taken"));
    }
    if !(floor.is_finite() && floor > 0.0 && floor <= 0.5) {
        return Err(invalid("the positivity floor must be in (0, 0.5]"));
    }
    let precision = 1.0 / (prior_scale * prior_scale);
    let joint = fit_joint(data, precision)?;
    let n = data.subjects();
    let width = data.width();
    let _lease = crate::util::admit_array::<f64>(ctx, draws, 4 + n)?;
    let mut columns: Vec<Vec<f64>> = (0..3).map(|_| Vec::with_capacity(draws)).collect();
    let mut shortfall = 0_usize;
    let mut features = Vec::with_capacity(width);
    let mut noise = vec![0.0; data.confounders];
    for draw in 0..draws {
        if ctx.cancellation.is_cancelled() {
            return Err(antecedent_stats::StatsError::Cancelled.into());
        }
        let mut rng = ctx.rng.stream_for(StreamDomain::Bayesian, POSTERIOR_STREAM | draw as u64);
        let hazard = draw_coefficients(
            &joint.hazard.coefficients,
            &joint.hazard.covariance_factor,
            1.0,
            &mut rng,
        );
        let treatment = draw_coefficients(
            &joint.treatment.coefficients,
            &joint.treatment.covariance_factor,
            1.0,
            &mut rng,
        );
        let _censoring = joint.censoring.as_ref().map(|model| {
            draw_coefficients(&model.coefficients, &model.covariance_factor, 1.0, &mut rng)
        });
        let transitions: Vec<(Vec<f64>, f64)> = joint
            .transitions
            .iter()
            .map(|law| {
                let variance = antecedent_kernels::sample_inv_gamma(law.shape, law.rate, &mut rng);
                let coefficients =
                    draw_coefficients(&law.mean, &law.covariance_factor, variance.sqrt(), &mut rng);
                (coefficients, variance.sqrt())
            })
            .collect();
        // Posterior positivity of the regimes on the observed histories.
        let mut smallest = 1.0_f64;
        for subject in &data.subjects {
            for k in 0..subject.treatment.len() {
                features.clear();
                data.treatment_features(subject, k, &mut features);
                let eta: f64 = treatment.iter().zip(&features).map(|(a, b)| a * b).sum();
                let p = antecedent_kernels::sigmoid(eta);
                smallest = smallest.min(p.min(1.0 - p));
            }
        }
        if smallest < floor {
            shortfall += 1;
        }
        let weights = antecedent_kernels::bayesian_bootstrap_weights(&mut rng, n, true);
        let mut survival = [0.0_f64; 2];
        let total: f64 = weights.iter().sum();
        for (i, subject) in data.subjects.iter().enumerate() {
            // One noise path per subject shared by both regimes (common random numbers).
            let mut paths = [subject.confounders[0].clone(), subject.confounders[0].clone()];
            let mut surviving = [1.0_f64; 2];
            for k in 0..data.horizon {
                if k > 0 {
                    antecedent_kernels::fill_standard_normal(&mut rng, &mut noise);
                }
                for (arm_index, arm) in [true, false].into_iter().enumerate() {
                    if k > 0 {
                        let previous = paths[arm_index].clone();
                        for (c, (coefficients, sd)) in transitions.iter().enumerate() {
                            features.clear();
                            data.transition_features(&previous, arm, k, &mut features);
                            let mean: f64 =
                                coefficients.iter().zip(&features).map(|(a, b)| a * b).sum();
                            paths[arm_index][c] = mean + sd * noise[c];
                        }
                    }
                    features.clear();
                    data.current_features(&paths[arm_index], arm, k, &mut features);
                    let eta: f64 = hazard.iter().zip(&features).map(|(a, b)| a * b).sum();
                    surviving[arm_index] *= 1.0 - antecedent_kernels::sigmoid(eta);
                }
            }
            for (total_arm, value) in survival.iter_mut().zip(surviving) {
                *total_arm += weights[i] * value;
            }
        }
        let treated = survival[0] / total;
        let untreated = survival[1] / total;
        if !(treated.is_finite() && untreated.is_finite()) {
            return Err(EstimationError::stats_msg("a posterior survival draw is not finite"));
        }
        columns[0].push(treated - untreated);
        columns[1].push(treated);
        columns[2].push(untreated);
    }
    let posterior = summarize_draws(columns, "joint_likelihood_parametric_g_formula");
    Ok(SurvivalFit {
        estimate: [posterior.mean[0], posterior.mean[1], posterior.mean[2]],
        standard_error: None,
        interval: None,
        replicates: (0, 0),
        posterior: Some(posterior),
        minimum_regime_probability: f64::NAN,
        posterior_positivity_shortfall: Some(shortfall as f64 / draws as f64),
        subjects: n,
        person_periods: data.person_periods(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform(seed: u64) -> f64 {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x3C6E_F372_FE94_F82B;
        state ^= state >> 31;
        state = state.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        state ^= state >> 29;
        (state >> 11) as f64 / (1_u64 << 53) as f64
    }

    fn expit(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    type Rows = (Vec<usize>, Vec<i64>, Vec<bool>, Vec<bool>, Vec<bool>, Vec<f64>);

    /// Time-varying confounding: `L_k = 0.5 L_{k-1} - 0.4 A_{k-1} + e`; treatment more likely
    /// when `L_k` is high; hazard `expit(-2.5 + 0.8 L_k - 0.7 A_k)`; light random censoring.
    fn trajectories(subjects: u64, horizon: u64) -> Rows {
        let (mut unit, mut period, mut treat, mut cens, mut event, mut l) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for i in 0..subjects {
            let mut current = uniform(i * 101) * 2.0 - 1.0;
            let mut previous_a = false;
            for k in 0..horizon {
                if k > 0 {
                    current = 0.5 * current - 0.4 * f64::from(u8::from(previous_a))
                        + (uniform(i * 101 + k + 50_000) - 0.5);
                }
                let a = uniform(i * 101 + k + 90_000) < expit(0.8 * current);
                let c = uniform(i * 101 + k + 130_000) < 0.03;
                let y = !c
                    && uniform(i * 101 + k + 170_000)
                        < expit(-2.5 + 0.8 * current - 0.7 * f64::from(u8::from(a)));
                unit.push(usize::try_from(i).unwrap());
                period.push(i64::try_from(k).unwrap());
                treat.push(a);
                cens.push(c);
                event.push(y);
                l.push(current);
                previous_a = a;
                if c || y {
                    break;
                }
            }
        }
        (unit, period, treat, cens, event, l)
    }

    /// Monte Carlo truth of `S_1(K) - S_0(K)` from the generating law.
    fn truth(horizon: u64) -> f64 {
        let mut totals = [0.0; 2];
        let draws = 200_000_u64;
        for (index, arm) in [true, false].into_iter().enumerate() {
            for i in 0..draws {
                let mut current = uniform(i * 7 + 1) * 2.0 - 1.0;
                let mut surviving = 1.0;
                for k in 0..horizon {
                    if k > 0 {
                        current = 0.5 * current - 0.4 * f64::from(u8::from(arm))
                            + (uniform(i * 7 + k * 1_000_003 + 3) - 0.5);
                    }
                    surviving *= 1.0 - expit(-2.5 + 0.8 * current - 0.7 * f64::from(u8::from(arm)));
                }
                totals[index] += surviving;
            }
        }
        (totals[0] - totals[1]) / draws as f64
    }

    #[test]
    fn both_procedures_recover_the_regime_contrast() {
        let horizon = 4;
        let (unit, period, treat, cens, event, l) = trajectories(1500, horizon);
        let columns = vec![l];
        let rows = TrajectoryRows {
            unit: &unit,
            period: &period,
            treatment: &treat,
            censored: &cens,
            event: &event,
            confounders: &columns,
        };
        let data = Trajectories::new(rows, usize::try_from(horizon).unwrap()).unwrap();
        let target = truth(horizon);
        let ctx = ExecutionContext::for_tests(13);
        let fit = fit_regime_survival(&data, 0.02, 99, &ctx).unwrap();
        let se = fit.standard_error.unwrap()[0];
        assert!(
            (fit.estimate[0] - target).abs() < 4.0 * se + 0.01,
            "{:?} vs {target}",
            fit.estimate
        );
        let bayes = fit_regime_survival_posterior(&data, 0.02, 10.0, 200, &ctx).unwrap();
        let posterior = bayes.posterior.unwrap();
        assert!(
            (posterior.mean[0] - target).abs() < 4.0 * posterior.standard_deviation[0] + 0.01,
            "{} vs {target}",
            posterior.mean[0]
        );
        assert!(fit_regime_survival(&data, 0.02, 10, &ctx).is_err());
    }
}
